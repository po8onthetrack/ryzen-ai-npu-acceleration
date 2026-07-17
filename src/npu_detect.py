#!/usr/bin/env python3
"""
npu_detect.py — run YOLOv8m object detection on the Ryzen AI NPU (Linux).

NOTE:
AMD's own run_inference.py is Windows-only: utils.py calls `pnputil` (a Windows tool)
to detect the NPU, so on Linux it returns '' and NPU detection silently fails.
thus need our own Linux inference path.

what it does:
    preprocess the image, pass it to the NPU to make the prediction then hand it back to CPU to do the decode
    image  --preprocess-->  [1,3,640,640]  --NPU-->  [1,84,8400]  --decode-->  boxes
           (CPU)                          (VitisAI)              (CPU: filter+NMS)

Only session.run() touches the NPU. Pre/post-processing are plain CPU code:
NMS is branchy, irregular, data-dependent logic — NPU structure is bad at this data type. That's also why we do not bake NMS into the ONNX export.

The preprocess/decode logic follows Ultralytics' official
YOLOv8-OpenCV-ONNX-Python example (the validated reference); the only change is
swapping their cv2.dnn inference for ONNX Runtime + the VitisAI EP.

"""

import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort


# The 80 COCO classes YOLOv8 is trained on, in index order.
# Hardcoded rather than read from ultralytics' YAML so this script stands alone.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# One stable colour per class, so the same object type is always the same colour.
np.random.seed(42)
COLORS = np.random.uniform(50, 255, size=(len(COCO_CLASSES), 3))

INPUT_SIZE = 640



# session touches NPU

def build_session(model_path, ep, cache_dir, cache_key, verbose=False, threads=None, no_spin=False):
    """Create the ONNX Runtime session on CPU or on the NPU (VitisAI EP)."""
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 1 if verbose else 3   # ORT's logger

    if threads is not None:
        sess_options.intra_op_num_threads = threads
        print(f"[info] intra_op_num_threads = {threads}")
    if no_spin:
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        print("[info] allow_spinning = 0")

    if ep == "cpu":
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
    else:
        os.makedirs(cache_dir, exist_ok=True)
        providers = ["VitisAIExecutionProvider"]
        provider_options = [{
            "cache_dir": os.path.abspath(cache_dir),   
            "cache_key": cache_key,
            "log_level": "info" if verbose else "warning",   # VitisAI's glog logger
            "enable_cache_file_io_in_mem": "0",
        }]

    print(f"[info] loading model on {providers[0]} (first NPU run compiles — be patient)")
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers,
        provider_options=provider_options,
    )


    if ep == "npu":
        key_dir = os.path.join(cache_dir, cache_key)
        artifact = None
        if os.path.isdir(key_dir):
            artifact = next((f for f in os.listdir(key_dir)
                             if f.startswith("compiled.") and f.endswith(".xmodel")), None)
        if artifact:
            print(f"[info] NPU artifact: {cache_key}/{artifact}  <- confirmed on NPU")
        else:
            print("[WARN] no compiled .xmodel found — may be a cache hit, or NOT on NPU")

    return session



# preprocess

def preprocess(image):
    """Pad the image to a square at the top-left, then resize to 640x640.

    The image may be non-square, add padding then squish it

    Returns the input tensor and the scale factor needed to undo all this.
    """
    height, width = image.shape[:2]
    length = max(height, width)

    square = np.zeros((length, length, 3), np.uint8)
    square[0:height, 0:width] = image        # original sits in the top-left corner

    scale = length / INPUT_SIZE              # how much we shrank it

    # blobFromImage does 4 things at once:
    #   resize to 640x640, scale pixels 0-255 -> 0-1, swap BGR->RGB, HWC -> CHW,
    #   and add the batch dim.  Result: [1, 3, 640, 640] float32.
    blob = cv2.dnn.blobFromImage(
        square, scalefactor=1 / 255.0, size=(INPUT_SIZE, INPUT_SIZE), swapRB=True
    )
    return blob, scale



# decode

def decode(raw_output, scale, conf_threshold=0.25, nms_threshold=0.45):
    """Turn the model's raw output into a list of detections.

    The model does not output boxes. It outputs [1, 84, 8400]:
      * 8400 = candidate detections (a prediction at every grid cell, at 3 scales)
      * 84   = 4 box coords (cx, cy, w, h) + 80 class scores.  4 + 80 = 84.

    Vectorized: the per-candidate work (best class, best score, confidence filter) is
    done on the whole [8400, 84] array at once with numpy, instead of a Python loop.
    numpy runs it in compiled, contiguous, SIMD-capable C, eliminating the per-element
    interpreter overhead that dominated the loop version.
    """
    # transpose: [1,84,8400] -> [8400,84], each row one candidate
    outputs = np.squeeze(raw_output, axis=0).T

    box_params = outputs[:, :4]          # [8400, 4]  (cx, cy, w, h)
    class_scores = outputs[:, 4:]        # [8400, 80]

    # best class and its score for EVERY candidate at once (no loop)
    class_ids_all = class_scores.argmax(axis=1)      # [8400]
    max_scores_all = class_scores.max(axis=1)        # [8400]

    # confidence filter as a single boolean mask
    keep = max_scores_all >= conf_threshold
    if not np.any(keep):
        return []

    cx = box_params[keep, 0]
    cy = box_params[keep, 1]
    w = box_params[keep, 2]
    h = box_params[keep, 3]

    # centre-form -> top-left form (what NMS wants), vectorized -> [M, 4]
    boxes = np.stack([cx - 0.5 * w, cy - 0.5 * h, w, h], axis=1)
    scores = max_scores_all[keep]
    class_ids = class_ids_all[keep]

    # NMS wants Python lists; convert only the survivors (a few, not 8400)
    boxes_l = boxes.tolist()
    scores_l = scores.tolist()
    class_ids_l = class_ids.tolist()

    nms_keep = cv2.dnn.NMSBoxes(boxes_l, scores_l, conf_threshold, nms_threshold, 0.5)
    if len(nms_keep) == 0:
        return []

    # land the coords on the original image
    detections = []
    for i in np.array(nms_keep).flatten():
        i = int(i)
        x, y, bw, bh = boxes_l[i]
        cid = int(class_ids_l[i])
        detections.append({
            "class_id": cid,
            "label": COCO_CLASSES[cid],
            "confidence": float(scores_l[i]),
            "box": [round(x * scale), round(y * scale),
                    round((x + bw) * scale), round((y + bh) * scale)],  # x1,y1,x2,y2
        })
    return detections



# draw
def draw(image, detections):
    h, w = image.shape[:2]
    # scale relative to a 640px reference; clamp so it stays sane on tiny/huge images
    s = max(w, h) / 640.0
    font_scale = max(0.5, 0.5 * s)
    thickness  = max(2, int(2 * s))
    box_thick  = max(2, int(2 * s))

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = COLORS[d["class_id"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thick)
        text = f'{d["label"]} ({d["confidence"]:.2f})'
        cv2.putText(image, text, (x1, max(int(15 * s), y1 - int(8 * s))),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return image



def main():
    p = argparse.ArgumentParser(description="YOLOv8 detection on the Ryzen AI NPU")
    p.add_argument("--model", required=True, help="path to the .onnx model")
    p.add_argument("--image", required=True, help="input image")
    p.add_argument("--output", default="output.jpg", help="output image with boxes")
    p.add_argument("--ep", default="npu", choices=["cpu", "npu"])
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--cache-dir", default="vitisai_cache")
    p.add_argument("--cache-key", default="yolo_detect",
                   help="use a DIFFERENT key per model, or the old compile is reused")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--threads", type=int, default=1,
                   help="intra_op_num_threads. Default 1: the NPU does the math, so "
                        "extra CPU threads only busy-wait (30%% slower, ~80x more CPU). "
                        "Use --threads 0 to let ORT decide (its default).")
    p.add_argument("--no-spin", action="store_true",
                   help="disable ORT thread-pool busy-waiting (redundant if threads=1)")
    args = p.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"could not read image: {args.image}")

    threads = None if args.threads == 0 else args.threads
    session = build_session(args.model, args.ep, args.cache_dir, args.cache_key,
                            args.verbose, threads=threads, no_spin=args.no_spin)
    input_name = session.get_inputs()[0].name

    # preprocess (CPU)
    blob, scale = preprocess(image)

    # inference 
    t0 = time.perf_counter()
    raw = session.run(None, {input_name: blob})[0]
    infer_ms = (time.perf_counter() - t0) * 1000

    # decode (CPU)
    t1 = time.perf_counter()
    detections = decode(raw, scale, args.conf, args.nms)
    decode_ms = (time.perf_counter() - t1) * 1000

    # draw + save
    result = draw(image, detections)
    cv2.imwrite(args.output, result)

    # report
    print(f"\n[{args.ep.upper()}] inference: {infer_ms:.1f} ms | "
          f"decode+NMS: {decode_ms:.1f} ms")
    print(f"detections: {len(detections)}")
    for d in detections:
        print(f'  {d["label"]:<15} {d["confidence"]:.2f}  {d["box"]}')
    if len(detections) == 0:
        print("  (none — if this is a naively-quantized model, that's the point:\n"
              "   INT8 crushes the confidence scores to zero. See the exclusion finding.)")
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
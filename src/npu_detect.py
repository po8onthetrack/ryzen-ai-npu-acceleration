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

def build_session(model_path, ep, cache_dir, cache_key, verbose=False):
    """Create the ONNX Runtime session on CPU or on the NPU (VitisAI EP)."""
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 1 if verbose else 3   # ORT's logger

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

    Most candidates are junk. Four steps turn this into usable boxes.
    """
    # transpose
    outputs = np.squeeze(raw_output, axis=0).T          # [8400, 84]

    boxes, scores, class_ids = [], [], []

    # pick the best class
    for row in outputs:
        classes_scores = row[4:]                        # the 80 class scores
        class_id = int(np.argmax(classes_scores))       # which class is most likely
        max_score = float(classes_scores[class_id])     # how confident

        if max_score >= conf_threshold:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            # convert centre-form -> top-left form, which is what NMS wants
            boxes.append([cx - 0.5 * w, cy - 0.5 * h, w, h])
            scores.append(max_score)
            class_ids.append(class_id)

    if not boxes:
        return []

    # Non-Maximum Suppression
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold, 0.5)
    if len(keep) == 0:
        return []

   # land the coords on original image
    detections = []
    for i in np.array(keep).flatten():
        i = int(i)
        x, y, w, h = boxes[i]
        detections.append({
            "class_id": class_ids[i],
            "label": COCO_CLASSES[class_ids[i]],
            "confidence": scores[i],
            "box": [round(x * scale), round(y * scale),
                    round((x + w) * scale), round((y + h) * scale)],   # x1,y1,x2,y2
        })
    return detections



# draw

def draw(image, detections):
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = COLORS[d["class_id"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f'{d["label"]} ({d["confidence"]:.2f})'
        cv2.putText(image, text, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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
    args = p.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"could not read image: {args.image}")

    session = build_session(args.model, args.ep, args.cache_dir,
                            args.cache_key, args.verbose)
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
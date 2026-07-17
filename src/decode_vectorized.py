"""
decode_vectorized.py — a vectorized replacement for the per-candidate decode loop,
plus a small harness to prove it's correct (identical detections) and faster.

replaced the original decode function in npu_detect.py with the new vetorized function


The original decode() loops over all 8400 candidates in Python:

Vectorization does the same work on the whole [8400, 84] array at once, in compiled C, the NMS call is unchanged (already C).
"""

import numpy as np
import cv2
import time



#original

def decode_loop(raw_output, scale, conf_threshold=0.25, nms_threshold=0.45):
    outputs = np.squeeze(raw_output, axis=0).T          # [8400, 84]
    boxes, scores, class_ids = [], [], []
    for row in outputs:
        classes_scores = row[4:]
        class_id = int(np.argmax(classes_scores))
        max_score = float(classes_scores[class_id])
        if max_score >= conf_threshold:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            boxes.append([cx - 0.5 * w, cy - 0.5 * h, w, h])
            scores.append(max_score)
            class_ids.append(class_id)
    return _finish(boxes, scores, class_ids, scale, conf_threshold, nms_threshold)


# vectorized
def decode_vectorized(raw_output, scale, conf_threshold=0.25, nms_threshold=0.45):
    outputs = np.squeeze(raw_output, axis=0).T          # [8400, 84]

    # split once: box params [N,4], class scores [N,80]
    box_params = outputs[:, :4]
    class_scores = outputs[:, 4:]

    # best class + its score for every candidate at once (no loop)
    class_ids_all = class_scores.argmax(axis=1)                 # [8400]
    max_scores_all = class_scores.max(axis=1)                   # [8400]

    # filter by confidence with a boolean mask — one operation
    keep = max_scores_all >= conf_threshold
    if not np.any(keep):
        return []

    cx = box_params[keep, 0]
    cy = box_params[keep, 1]
    w = box_params[keep, 2]
    h = box_params[keep, 3]

    # centre-form -> top-left form, vectorized; shape [M, 4]
    boxes = np.stack([cx - 0.5 * w, cy - 0.5 * h, w, h], axis=1)
    scores = max_scores_all[keep]
    class_ids = class_ids_all[keep]

    # NMSBoxes wants Python lists; this is the only conversion, on the small
    # surviving set (typically a few hundred, not 8400)
    return _finish(boxes.tolist(), scores.tolist(), class_ids.tolist(),
                   scale, conf_threshold, nms_threshold)



# NMS + un-map to original image coords (identical for both)
def _finish(boxes, scores, class_ids, scale, conf_threshold, nms_threshold):
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold, 0.5)
    if len(keep) == 0:
        return []
    dets = []
    for i in np.array(keep).flatten():
        i = int(i)
        x, y, w, h = boxes[i]
        dets.append({
            "class_id": int(class_ids[i]),
            "confidence": float(scores[i]),
            "box": [round(x * scale), round(y * scale),
                    round((x + w) * scale), round((y + h) * scale)],
        })
    return dets



# harness
def compare(raw_output, scale, iters=100):
    # correctness: identical detections?
    d1 = decode_loop(raw_output, scale)
    d2 = decode_vectorized(raw_output, scale)

    same = (len(d1) == len(d2))
    if same:
        # sort both by (class, box) so order differences don't matter
        key = lambda d: (d["class_id"], d["box"])
        for a, b in zip(sorted(d1, key=key), sorted(d2, key=key)):
            if a["class_id"] != b["class_id"] or a["box"] != b["box"] \
               or abs(a["confidence"] - b["confidence"]) > 1e-6:
                same = False
                break
    print(f"detections: loop={len(d1)}  vectorized={len(d2)}  identical={same}")

    # speed
    t = time.perf_counter()
    for _ in range(iters):
        decode_loop(raw_output, scale)
    loop_ms = (time.perf_counter() - t) / iters * 1000

    t = time.perf_counter()
    for _ in range(iters):
        decode_vectorized(raw_output, scale)
    vec_ms = (time.perf_counter() - t) / iters * 1000

    print(f"loop:       {loop_ms:.2f} ms/call")
    print(f"vectorized: {vec_ms:.2f} ms/call")
    print(f"speedup:    {loop_ms/vec_ms:.1f}x")


if __name__ == "__main__":
    # synthetic YOLOv8m-shaped output so this runs standalone.
    # Replace with a real session.run() output for the true comparison.
    rng = np.random.default_rng(0)
    fake = rng.random((1, 84, 8400), dtype=np.float32)
    fake[:, :4, :] *= 640                 # box coords in pixel range
    compare(fake, scale=1.0)
# Results Summary

Ryzen AI 9 HX 370 (XDNA2 "Strix" NPU) — ws007, Ubuntu 24.04.
Ryzen AI Software 1.7.1, XRT 2.21.0, ONNX Runtime + VitisAI EP.
All NPU runs verified by the compiled `AMD_AIE2P_4x8_CMC_Overlay.xmodel` artifact.

**Headline: YOLOv8m INT8 on the NPU = 25.6 ms inference / 5.2x vs FP32 CPU.
With the decode optimized, end-to-end is ~27 ms (~37 inf/s).**

---

## 1. YOLOv8m (COCO, 640x640) — precision ladder

Model inference only (`session.run`).

| Config    | Mean     | Median   | p95      | Throughput  | vs FP32 CPU |
|-----------|----------|----------|----------|-------------|-------------|
| FP32, CPU | 132.3 ms | 128.7 ms | 148.0 ms | 7.56 inf/s  | 1.0x        |
| INT8, CPU | 168.4 ms | 165.9 ms | 177.9 ms | 5.94 inf/s  | 0.79x       |
| BF16, CPU | 247.0 ms | 243.0 ms | 258.1 ms | 4.05 inf/s  | 0.54x       |
| BF16, NPU | 66.4 ms  | 66.9 ms  | 68.2 ms  | 15.07 inf/s | **2.0x**    |
| INT8, NPU | **25.6 ms** | **25.6 ms** | **25.9 ms** | **39.0 inf/s** | **5.2x** |

NPU INT8 uses the tuned config (`intra_op_num_threads=1`, section 5). It is also the most
stable measurement: p95 within 1.2% of the mean, versus ~7% on the CPU.

NPU offload — INT8: 1237/1262 ops (98%), 25 on CPU (excluded detection head).
              BF16:  925/925 ops (100%), none on CPU (no exclusion needed).

Accuracy (mAP, quoted from AMD's docs — NOT measured here): FP32 44.0, BF16 42.8, INT8 38.1.

---

## 2. ResNet (CIFAR-10, 32x32) — INT8

| Provider | Mean    | Throughput   | Speedup |
|----------|---------|--------------|---------|
| CPU      | 9.13 ms | 109.5 inf/s  | 1.0x    |
| NPU      | 1.48 ms | 675.2 inf/s  | **6.2x**|

NPU offload: 398/400 ops (99.5%); 2 ops on CPU (final dequantize boundary).

---

## 3. Correctness — CPU and NPU are numerically identical

Same quantized model on CPU vs NPU produces **identical detections** — same objects,
labels, confidences (2 d.p.), and boxes — on AMD's test image and on two of my own photos
(dining table 4285x5712, street scene 1201x648).

=> Accuracy loss comes from quantization (FP32->INT8), NOT from running on the NPU.

---

## 4. The exclusion experiment

| Model              | Detections | Image                          |
|--------------------|------------|--------------------------------|
| INT8, no exclusion | **0**      | results/detect_no_exclude.jpg  |
| INT8, head excluded| **18**     | results/detect_int8_npu.jpg    |

Identical model, quantizer, hardware, and input. The only difference is
`--exclude_subgraphs "[/model.22/Concat_3], [/model.22/Concat_10]]"`.

Naive INT8 quantization of YOLOv8m detects **nothing**: the detection head concatenates box
coords (0-640) with confidences (0-1) into one tensor; INT8's single per-tensor scale must
cover 640, so every confidence rounds to zero and nothing clears the 0.25 threshold. BF16
(a float format) needs no exclusion — confirming this is specifically an INT8 dynamic-range
problem.

---

## 5. Thread-pool configuration (optimization 1)

ONNX Runtime's default pool spawns ~24 workers that busy-wait (`allow_spinning: 1`) while
the NPU computes. They do no useful work but saturate cores.

`yolov8m_XINT8`, 500 runs (NPU) / 50 runs (CPU), 5 warmup discarded. CPU usage is `top`'s
per-process figure (100% = one thread; ceiling ~2400% on 24 threads).

| Device | Threads | Spin | Mean     | p95      | Throughput  | CPU usage |
|--------|---------|------|----------|----------|-------------|-----------|
| NPU    | default | on   | 36.6 ms  | 44.1 ms  | 27.3 inf/s  | ~1800% (approx 18 threads) |
| NPU    | default | off  | 26.9 ms  | 31.0 ms  | 37.2 inf/s  | ~49% (half thread) |
| NPU    | **1**   | on   | 25.7 ms  | 26.0 ms  | 39.0 inf/s  | ~25% (quarter thread) |
| NPU    | **1**   | off  | **25.6 ms** | **25.9 ms** | **39.1 inf/s** | **~25%** |
| CPU    | default | on   | 168.4 ms | 177.9 ms | 5.94 inf/s  | ~2400% (all threads) |
| CPU    | 1       | on   | 607.5 ms | 608.5 ms | 1.65 inf/s  | ~100% (one thread) |

**The symmetry — same knob, opposite signs:**
- **CPU: threads help — 3.6x faster** (607.5 -> 168.4 ms). The convolutions parallelize;
  each thread does real work.
- **NPU: threads hurt — 1.4x slower** (25.7 -> 36.6 ms). The NPU does all the math, so those
  threads only spin, contending with the coordinating thread.

CPU occupancy is comparable in both default cases (~2400% CPU / ~1800% NPU) — but on the CPU
that work is real, and on the NPU it is wasted. **With default settings you offload the math
to the NPU and free nothing.**

**Thread count is the fix; disabling spinning is only partial.** `allow_spinning=0` stops the
workers spinning (26.9 ms, ~49% CPU) but they still exist; `intra_op_num_threads=1` means they
are never created (25.6 ms, ~25% CPU). Combining both adds nothing.
**Recommended NPU config: `intra_op_num_threads = 1`.**

**Efficiency (power telemetry unavailable):** `xrt-smi` reports `Estimated Power: N/A`, and
CPU-package RAPL counters require root (no sudo on the shared machine). CPU occupancy is the
proxy:

| NPU config  | Throughput     | CPU usage |
|-------------|----------------|-----------|
| default     | 27.3 inf/s     | ~1800%    |
| `threads=1` | **39.0 inf/s** | **~25%**  |

**43% more throughput on ~1/70th of the CPU** (~1800% -> ~25%). That is the actual argument
for an NPU: it should do the inference *and* leave the CPU free — misconfigured, it does
neither. ORT's thread options are documented as general performance knobs, but neither ORT's
NPU guidance nor AMD's Ryzen AI examples set them; all of AMD's example scripts use the
defaults.

---

## 6. Vectorized decode (optimization 2)

The YOLO post-processing (decode + NMS) runs on the CPU and does not touch the NPU. The
original decode looped over all 8400 candidates in Python. Vectorizing it — argmax / max /
boolean-mask over the whole [8400,84] tensor in numpy — removes the per-candidate interpreter
overhead. NMS is unchanged. Output verified **identical** (same 18 detections).

| Decode        | Time    |
|---------------|---------|
| Python loop   | 19.8 ms |
| **Vectorized**| **1.3 ms** (~15x) |

End-to-end pipeline (inference + decode), YOLOv8m INT8 NPU, tuned config:

| Pipeline               | Inference | Decode+NMS | End-to-end | Throughput |
|------------------------|-----------|------------|------------|------------|
| Before (loop decode)   | 25.6 ms   | 19.8 ms    | ~45 ms     | ~22 inf/s  |
| **After (vectorized)** | 25.6 ms   | **1.3 ms** | **~27 ms** | **~37 inf/s** |

Before the fix, decode was ~44% of the NPU pipeline (Amdahl's law: accelerating inference
made the un-accelerated post-processing dominant). After, it is ~5%, and inference dominates
again — as it should.

---

## Summary of the two optimizations

Starting from a working deployment, two independent bottlenecks were found and fixed:
1. **Thread config** (`intra_op_num_threads=1`): inference 36.6 -> 25.6 ms, CPU ~1800% -> ~25%.
2. **Vectorized decode**: post-processing 19.8 -> 1.3 ms (~15x), output identical.
Together: end-to-end ~45 ms -> ~27 ms (~22 -> ~37 inf/s).
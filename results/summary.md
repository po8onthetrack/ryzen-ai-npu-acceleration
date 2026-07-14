## YOLOv8m (COCO, 640×640) — Ryzen AI 9 HX 370 (XDNA2)

| Config    | Mean     | Median   | p95      | Throughput  | vs FP32 CPU |
|-----------|----------|----------|----------|-------------|-------------|
| FP32, CPU | 132.3 ms | 128.7 ms | 148.0 ms | 7.56 inf/s  | 1.0×        |
| INT8, CPU | 169.6 ms | 169.8 ms | 172.2 ms | 5.89 inf/s  | 0.78×       |
| BF16, CPU | 247.0 ms | 243.0 ms | 258.1 ms | 4.05 inf/s  | 0.54×       |
| BF16, NPU | 66.4 ms  | 66.9 ms  | 68.2 ms  | 15.07 inf/s | **2.0×**    |
| INT8, NPU | 34.1 ms  | 34.0 ms  | 38.0 ms  | 29.34 inf/s | **3.9×**    |

NPU offload — INT8: 1237/1262 ops (98%), 25 on CPU (excluded detection head).
              BF16:  925/925 ops (100%), none on CPU (no exclusion needed).

Note: the NPU is also far more consistent — CPU FP32 shows ~15% run-to-run variance,
the NPU under 2%.

## ResNet (CIFAR-10, 32×32) — INT8

| Provider | Mean    | Throughput   | Speedup |
|----------|---------|--------------|---------|
| CPU      | 9.13 ms | 109.5 inf/s  | 1.0×    |
| NPU      | 1.48 ms | 675.2 inf/s  | **6.2×**|

NPU offload: 398/400 ops (99.5%); 2 ops on CPU.


## Detection correctness — CPU vs NPU (yolov8m_XINT8, test_image.jpg)

Both produce **identical detections**: same 18 objects, same labels, same
confidences to 2 d.p., same box coordinates.

=> The NPU is numerically equivalent to the CPU for the same quantized model.
   Accuracy loss comes from quantization (FP32→INT8), NOT from running on the NPU.

| Stage           | CPU      | NPU     | Speedup |
|-----------------|----------|---------|---------|
| Inference       | 217.5 ms | 34.7 ms | 6.3×    |
| Decode + NMS    | 13.2 ms  | 19.8 ms | (CPU-bound in both) |
| **End-to-end**  | 230.7 ms | 54.5 ms | **4.2×**|

Note: decode+NMS is unaccelerated . It's 6% of the CPU pipeline but 36% of the NPU pipeline 

## The exclusion experiment 

| Model                        | Detections | Image                      |
|------------------------------|------------|----------------------------|
| INT8, no exclusion           | **0**      | results/detect_no_exclude.jpg |
| INT8, head excluded          | **18**     | results/detect_int8_npu.jpg   |

Identical model, quantizer, hardware, and input. The only difference is
`--exclude_subgraphs "[/model.22/Concat_3], [/model.22/Concat_10]]"`.

Naive INT8 quantization of YOLOv8m produces a model that detects nothing.

## Own test images — CPU/NPU equivalence generalizes

Two photos of my own (a dinner table, a street scene), run through `npu_detect.py`
on both providers with `yolov8m_XINT8`.

**Detections are identical on CPU and NPU** — same objects, same labels, same
confidences to 2 d.p., same box coordinates. This holds on real-world photos, not
just AMD's curated test image.

| Image     | Detections | CPU inference | NPU inference | Speedup |
|-----------|------------|---------------|---------------|---------|
| sample1 (dining table, 4285×5712) | 14 | 218.1 ms | 44.3 ms | 4.9× |
| sample2 (street scene, 1201×648)  | 19 | 331.7 ms | 30.7 ms | 10.8× |
| test_image (AMD, 640×427)         | 18 | 217.5 ms | 34.7 ms | 6.3× |

### Decode+NMS is CPU-bound and scales with detection count / image size

| Image     | NPU inference | Decode+NMS (NPU run) | Decode share |
|-----------|---------------|----------------------|--------------|
| sample1   | 44.3 ms       | 37.2 ms              | 46%      |
| sample2   | 30.7 ms       | 20.7 ms              | 40%      |
| test_image| 34.7 ms       | 19.8 ms              | 36%      |

The decode is a Python loop over 8400 candidates plus NMS — it runs on the CPU and
does not benefit from the NPU. On the CPU runs it costs only ~13 ms (6% of the
pipeline); on the NPU runs it is 36–46% of total time.


## Thread-pool configuration — 30% faster on ~1/80th the CPU

ONNX Runtime's default thread pool spawns ~24 workers that **busy-wait**
(`allow_spinning: 1`) while waiting for work. During NPU inference they have nothing
to do — the NPU does the math — so they spin, saturating ~20 cores and contending with
the one thread that coordinates with the NPU.

`yolov8m_XINT8`, 500 runs (NPU) / 50 runs (CPU), 5 warmup discarded.
CPU usage is `top`'s per-process figure (100% = one core; ceiling ~2400% on 24 threads).

| Device | Threads | Spin | Mean | p95 | Throughput | CPU usage |
|--------|---------|------|------|-----|------------|-----------|
| NPU | default (~24) | on  | 36.6 ms | 44.1 ms | 27.3 inf/s | **~2000%** (≈20 cores) |
| NPU | default (~24) | off | 26.9 ms | 31.0 ms | 37.2 inf/s | ~49% (½ core) |
| NPU | **1** | on  | 25.7 ms | 26.0 ms | 39.0 inf/s | **~25%** (¼ core) |
| NPU | **1** | off | **25.6 ms** | **25.9 ms** | **39.1 inf/s** | **~25%** |
| CPU | default (~24) | on | 168.4 ms | 177.9 ms | 5.94 inf/s | ~2000% (≈20 cores) |
| CPU | 1 | on | 607.5 ms | 608.5 ms | 1.65 inf/s | ~100% (1 core) |

### The symmetry — the same knob, opposite signs

- **CPU: threads help — 3.6× faster** (607.5 → 168.4 ms). The convolutions genuinely
  parallelize; each core does real work.
- **NPU: threads hurt — 1.4× slower** (25.7 → 36.6 ms). The NPU does all the math, so
  those same ~20 cores do nothing but spin, contending with the coordinating thread.

Note the CPU occupancy is **identical (~2000%) in both default cases** — but on the CPU
that work is real, and on the NPU it is entirely wasted. **With default settings you
offload the math to the NPU and free nothing.**

### Thread count is the fix; disabling spinning is only a partial one

`allow_spinning=0` stops the workers spinning (26.9 ms, ~49% CPU) but they still exist.
`intra_op_num_threads=1` means they are **never created** (25.7 ms, ~25% CPU).
Combining both changes nothing further (25.6 ms, ~25%).

**Recommended NPU config: `intra_op_num_threads = 1`.**

### Efficiency, without a power sensor

NPU power telemetry is unavailable (`xrt-smi`: `Estimated Power: N/A`) and CPU-package
RAPL counters require root, which I don't have on the shared machine. CPU occupancy is
the proxy:

| NPU config | Throughput | CPU |
|---|---|---|
| default | 27.3 inf/s | ~2000% |
| `threads=1` | **39.0 inf/s** | **~25%** |

**43% more throughput on ~1/80th of the CPU.** That is the actual argument for an NPU:
it should do the inference *and* leave the CPU free. Misconfigured, it does neither.

ORT's defaults are correct for CPU inference. Pointed at an NPU, the identical settings
are pure waste. This is undocumented.

### Revised headline

| Config | Latency | Throughput | vs FP32 CPU |
|---|---|---|---|
| FP32, CPU (default threads) | 132.3 ms | 7.56 inf/s | 1.0× |
| INT8, CPU (default threads) | 168.4 ms | 5.94 inf/s | 0.79× |
| **INT8, NPU (1 thread)** | **25.7 ms** | **39.0 inf/s** | **5.2×** |

The NPU is also the most stable measurement in the dataset: p95 within 1.2% of the mean
(26.0 vs 25.7 ms), versus ~7% on the CPU.
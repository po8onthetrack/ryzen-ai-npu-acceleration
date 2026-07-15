## 2026-07-09 — quicktest PASSED
- Ryzen AI 1.7.1 installed to ~/ryzenai/venv
- quicktest.py failed first with "XRT is not installed / binaries only"
  → fix: `source /opt/xilinx/xrt/setup.sh` inside the venv shell, then re-run
- Result: "Test Finished" — model ran on Strix NPU. Setup confirmed working.
- Added XRT source to ~/.bashrc so it's automatic.
- Session recipe: activate venv + source XRT (both needed every new shell).

## 2026-07-12 

## ResNet example: torch conflict + slow CIFAR-10 download. 

**Goal:** run AMD's getting-started ResNet example (CPU vs NPU) to learn the pipeline.

**Done**
- Servers back after lab migration. Verified NPU intact: `xrt-smi examine` OK, quicktest passes.
- Cloned `RyzenAI-SW` → `CNN-examples/getting_started_resnet/int8`.
- Added `src/benchmark.py` — reusable CPU-vs-NPU timing script (warmup + timed loop + CSV).

**Issue 1 — torch version conflict (benign, verified)**
The example's `requirements.txt` forces `torch 2.5.1+cpu` → `2.8.0`, breaking flexml's pin:
```
flexml 1.7.1 requires torch==2.5.1+cpu, but I have torch 2.8.0 which is incompatible.
```
Verified it still works anyway: `VitisAIExecutionProvider` present, quicktest → `Test Finished`.
→ Cosmetic. **If the NPU misbehaves later, suspect this first.**

**Issue 2 — CIFAR-10 download stalls**
`prepare_model_data.py` kept dying partway through the download (34 MB, then 45 MB, 0% CPU).
Root cause: not blocked — the UofT host serving CIFAR-10 is just slow (~35 KB/s, so ~1 hr for
163 MB). Python's downloader times out and gives up silently. Ruled out VPN/network: my Mac hit
the same speed, and other downloads on ws007 run at 30–50 MB/s.

Fix — wget with resume + retries, under nohup so it survives SSH drops:
```bash
cd ~/RyzenAI-SW/CNN-examples/getting_started_resnet/int8/data
nohup wget -c --tries=50 --timeout=60 --waitretry=10 \
      https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz > wget.log 2>&1 &
```
Target: 163M. torchvision then finds and extracts the tarball — no re-download.


## FIRST RESULT: ResNet on NPU
- 398/400 ops (99.5%) offloaded to NPU; 2 (dequantize-linear) fall back to CPU
- Target arch: AMD_AIE2P_4x8_CMC_Overlay (XDNA2, 4x8 tiles)
- Accuracy: 9/10 CIFAR-10 images (matches AMD's documented output)
- BENCHMARK: CPU 9.26 ms / 108 inf/s  vs  NPU 1.48 ms / 674 inf/s  => 6.24x speedup
- NPU jitter far lower than CPU (p95 within 1.5% of mean vs 6% on CPU)


## 2026-07-13 — YOLOv8m on NPU: deployed, benchmarked, and three traps

**Goal:** deploy YOLOv8m on the NPU and get verified CPU-vs-NPU numbers.
**Result:** done. YOLO runs on the NPU at 98% operator offload, ~3.6x faster than FP32 CPU.

---

### Results

**YOLOv8m (COCO, 640x640)**

| Config     | Latency  | Throughput   | vs FP32 CPU |
|------------|----------|--------------|-------------|
| FP32, CPU  | 123.7 ms | 8.08 inf/s   | 1.0x        |
| INT8, CPU  | 166.3 ms | 6.01 inf/s   | 0.74x (slower) |
| INT8, NPU  | 34.1 ms  | 29.34 inf/s  | **3.6x**    |
| BF16, NPU  | 66.37 ms  | 15.07 inf/s  | **1.86x**    |
| BF16, CPU  | 247.0 ms  | 4.05 inf/s  | 0.5x(slower)    |

NPU offload: **1237 / 1262 ops (98%)**; 18 CPU + 7 VITIS_EP_CPU = 25 ops on CPU
(the excluded detection head). Compare ResNet: 398/400 (99.5%), only 2 CPU ops.

**Finding: INT8 makes the CPU *slower* (123.7 -> 166.3 ms).** The QDQ
(quantize/dequantize) nodes are free on the NPU (native INT8 hardware) but are pure
overhead on the CPU. Quantization is not a general optimization — it is a
*hardware-targeting decision*. It only pays off if you have hardware built for it.
This is the argument for the NPU existing, measured on my own machine.

---

### Trap 1 — the detection head must be excluded from quantization

Quantizing the whole model produces a model that **detects nothing**.

Reason: YOLO's head concatenates box coordinates (0–640, big) with confidence scores
(0–1, tiny) into one tensor. INT8 has only 256 levels and picks ONE scale per tensor.
To cover 640, the step size is ~2.5 — so every confidence rounds to 0. Nothing passes
the 0.25 confidence threshold, so no boxes are drawn.

Quark visibly fails at this when not excluded:
```
Input pos of concat node /model.22/Concat_10 is 7, min_pos is -3. Modify ipos from 7 to -3.
...
[QUARK-WARNING]: The number of adjustments has reached the limit. Please check the model
```
Those warnings vanish entirely with the exclusion:
```
--exclude_subgraphs "[/model.22/Concat_3], [/model.22/Concat_10]]"
```
Confirm it landed by checking the config dump prints
`subgraphs_to_exclude --- [(['/model.22/Concat_3'], ['/model.22/Concat_10'])]`, not `[]`.

Cost of the exclusion: the head stays FP32 and runs on CPU (~25 ops). Worth it — a
model that detects nothing is worth 0x speedup.

---

### Trap 2 — AMD's yolov8m example is Windows-only

`utils.py: get_npu_info()` shells out to `pnputil` — a **Windows** utility. On Linux it
returns `''`, so NPU detection fails. `run_inference.py` only survives because of a
Python bug: `elif npu_device == 'STX' or 'KRK'` is always truthy. `get_xclbin()` also
uses Windows paths (`voe-4.0-win_amd64`, backslashes).

=> Their `run_inference.py` cannot be trusted on Linux. Used my own `benchmark.py`
(clean Linux provider options) instead. 
---

### Trap 3 — VitisAI runs on the NPU silently (cost hours)

Spent a long time convinced the NPU wasn't being used. It was. The whole time.
Three things conspired:

1. **VitisAI has its own logger** (glog), set by the *provider option* `'log_level':'info'`
   — Without it, VitisAI compiles and runs on the NPU with zero output. AMD's `predict.py` sets `log_level`; my script didn't.
   That's the only reason ResNet "worked" and YOLO "didn't."

2. **camelCase provider options are silently ignored.** Must be snake_case:
   `cache_dir`, `cache_key`, `log_level`. My `cacheDir`/`cacheKey` were dropped, so
   VitisAI used defaults and cached elsewhere — the folder I was inspecting stayed empty.

**Note: absence of log output is not evidence of absence of work.**

**Reliable check** (does not depend on any logger being configured right): look for the
compiled artifact on disk —
`<cache_dir>/<cache_key>/compiled.AMD_AIE2P_4x8_CMC_Overlay.xmodel`
`benchmark.py` now checks for this automatically and prints `<- confirmed on NPU`.

---

### Changes to `src/benchmark.py`
- provider options -> snake_case (`cache_dir`, `cache_key`)
- added `log_level` provider option (VitisAI's glog)
- added `--verbose` flag (turns on both loggers; needs a fresh cache to show the op table)
- added `--cache-key` so models don't collide in the cache
- auto-verifies the compiled `.xmodel` artifact and warns if missing

### Note on reruns
All benchmarks re-run after the fix; results agree with the originals within ~1%
(ResNet NPU 1.484 -> 1.481 ms; YOLO NPU 34.30 -> 34.08 ms). The NPU numbers were valid
all along. Reproducibility confirmed.



## 2026-07-14 — own linux inference path, real-image testing, and the threading finding

Goal: build a linux yolo inference script (amd's is windows-only), test it on my own
images, and measure efficiency.

### Added src/npu_detect.py

My own linux inference path: preprocess → onnx runtime/vitisai on npu → decode + nms →
draw boxes. It verifies the compiled .xmodel artifact rather than trusting
get_providers(), which only tells you the provider was registered, not that it took any
nodes. Preprocess and decode follow ultralytics' yolov8-opencv-onnx-python reference;
the only change is swapping their cv2.dnn inference for the npu session.

### Cpu and npu are numerically identical

On amd's test image and on two photos of my own, cpu and npu produce the same
detections — same objects, same labels, same confidences to two decimals, same box
coordinates. So accuracy loss comes from quantization (fp32 → int8), not from where the
model runs. That distinction matters and is easy to conflate.

### Decode and nms are cpu-bound

| image | npu inference | decode + nms | decode share |
|---|---|---|---|
| sample1 (4285×5712) | 44.3 ms | 37.2 ms | 46% |
| sample2 (1201×648) | 30.7 ms | 20.7 ms | 40% |
| amd test image | 34.7 ms | 19.8 ms | 36% |

The same decode costs only ~13 ms on the cpu runs — about 6% of that pipeline. Once the
network is 5–10× faster, the post-processing we did not accelerate becomes the dominant
cost. Vectorising the decode (currently a per-candidate python loop over 8400
candidates) is the obvious next lever.

### Finding: ort's thread defaults waste ~20 cores on the npu, and are 30% slower

Noticed the npu run was pinning ~2000% cpu (roughly 20 cores) — the same occupancy as
running the model on the cpu outright. But the npu does the math, so those threads have
nothing to do. They busy-wait, because ort sets allow_spinning: 1 by default.

| config | latency | cpu |
|---|---|---|
| default | 36.6 ms | ~2000% |
| allow_spinning = 0 | 26.9 ms | ~49% |
| intra_op_num_threads = 1 | **25.7 ms** | **~25%** |

Thread count is the real fix: with one thread the workers are never created. Disabling
spinning is only partial — the workers still exist, they just sleep. Combining both adds
nothing (25.6 ms, ~25%).

The symmetry is the interesting part. The same setting **helps the cpu by 3.6×**
(607 → 168 ms, because those threads do real parallel work) and **hurts the npu by 1.4×**
(25.7 → 36.6 ms, because they only spin). ORT's thread-pool options are documented as general performance knobs, but neither 
ORT's NPU guidance nor AMD's Ryzen AI examples set them — all of AMD's example scripts run with the defaults, which on this 
hardware cost ~30% latency and saturate ~20 cores doing nothing.

Revised headline: yolov8m int8 on npu = **25.7 ms / 39.0 inf/s = 5.2× vs fp32 cpu**
(it was 3.9× with the default thread config).

### Power measurement — blocked

- `xrt-smi examine` reports `Estimated Power: N/A`. The npu exposes no power telemetry.
- Rapl energy counters (`/sys/class/powercap/.../energy_uj`) are root-only, and I have
  no sudo on this shared machine.

So I used **cpu occupancy** as the efficiency proxy instead. Arguably it makes the point
better than watts would have: 43% more throughput on roughly one-eightieth of the cpu.
That is the actual argument for an npu — it should do the inference *and* leave the cpu
free. Misconfigured, it does neither.

## 2026-07-15 — plots, summary cleanup, and report setup


### Added src/plot_results.py

A matplotlib script that generates four figures into results/plots/:
- yolov8m_precision_ladder.png — latency across fp32/bf16/int8, cpu vs npu, with the
  5.2x callout vs the fp32 cpu baseline.
- threading_symmetry.png — the key finding: the same thread setting helps the cpu (3.6x)
  and hurts the npu (1.4x).
- threading_efficiency.png — cpu occupancy per npu config on a log axis (2000% -> 25%),
  with latency overlaid.
- resnet_cpu_vs_npu.png — latency and throughput for the classification result.

### Fixed inconsistent numbers in results/summary.md

Went through the whole summary for internal consistency

### Report setup

Drafted a report form these aspects: the task framing, the pipeline, the authoritative numbers, the four findings
with their mechanisms, the engineering traps, and the limitations. 


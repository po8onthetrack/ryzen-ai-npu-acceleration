# Ryzen AI NPU — Application Acceleration

Deploying neural networks on the **AMD Ryzen AI 9 HX 370 NPU** (XDNA2 "Strix" AI Engine) and measuring inference speedup over the CPU.

UTokyo internship project (CASYS Lab, Prof. Shinya Takamaeda-Yamazaki).
Theme: *AMD Ryzen AI NPU でのアプリケーション高速化* — accelerating applications on the Ryzen AI NPU.

## Goal

Take a trained model (a pretrained detector like YOLOv8n, then my own pose/action model), run its inference on the NPU instead of the CPU, and quantify the speedup and power savings.

## Pipeline

Trained model → ONNX → INT8 quantization (AMD Quark) → ONNX Runtime with the **VitisAI Execution Provider** → runs on the NPU.

The VitisAI EP automatically offloads supported operations to the NPU; the rest falls back to the CPU.

## Hardware / Environment

- Machine: ws007 (Minisforum EliteMini), Ubuntu 24.04
- APU: AMD Ryzen AI 9 HX 370 (12c/24t) + Radeon 890M iGPU
- NPU: XDNA2 "NPU Strix", firmware 1.1.2.64
- Stack: XRT 2.21.0, amdxdna driver, Ryzen AI Software, ONNX Runtime + VitisAI EP

## Deliverables

- Working demo: model running inference on the NPU
- Benchmark: CPU vs NPU comparison (latency, throughput/FPS, power)
- Short report / slides on the pipeline and results

## Status

- [ ] Ryzen AI Software installed; `quicktest` passes on NPU
- [ ] ResNet getting-started example runs on `--ep cpu` and `--ep npu`
- [ ] YOLOv8n exported, quantized, and running on NPU
- [ ] CPU-vs-NPU benchmark harness
- [ ] Own model deployed (stretch goal)

## Repo layout

```
notes/      running work log (what I tried, what broke, what worked)
setup/      install steps and environment notes
src/        quantization, inference, and benchmark scripts
results/    benchmark numbers, plots, screenshots
```

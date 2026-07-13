## YOLOv8m (COCO, 640×640) — Ryzen AI 9 HX 370 (XDNA2)

| Config    | Mean     | Median   | p95      | Throughput  | vs FP32 CPU |
|-----------|----------|----------|----------|-------------|-------------|
| FP32, CPU | 132.3 ms | 128.7 ms | 148.0 ms | 7.56 inf/s  | 1.0×        |
| INT8, CPU | 169.6 ms | 169.8 ms | 172.2 ms | 5.89 inf/s  | 0.78×       |
| INT8, NPU | 34.1 ms  | 34.0 ms  | 38.0 ms  | 29.34 inf/s | **3.9×**    |

NPU offload: 1237/1262 ops (98%); 25 ops on CPU (excluded detection head).
Note: the NPU is also far more consistent — CPU FP32 shows ~15% run-to-run variance,
the NPU under 2%.

## ResNet (CIFAR-10, 32×32) — INT8

| Provider | Mean    | Throughput   | Speedup |
|----------|---------|--------------|---------|
| CPU      | 9.13 ms | 109.5 inf/s  | 1.0×    |
| NPU      | 1.48 ms | 675.2 inf/s  | **6.2×**|

NPU offload: 398/400 ops (99.5%); 2 ops on CPU.
#!/usr/bin/env python3
"""
benchmark.py — measure inference speed of an ONNX model on CPU vs the Ryzen AI NPU.

the same script benchmarks every model (ResNet, YOLOv8n) by changing
command-line flags 

Typical use (on ws007, venv active + XRT sourced):
    python benchmark.py --model model_int8.onnx --ep cpu  --runs 100 --warmup 5
    python benchmark.py --model model_int8.onnx --ep npu  --runs 100 --warmup 5

Each run appends one row to results/benchmark.csv so numbers accumulate.

Why the pieces exist (so you can defend this, not just run it):
  * warmup      -> the FIRST NPU inference COMPILES the model and is very slow;
                   early runs aren't representative. We run a few and discard them.
  * random input-> for SPEED we only need correctly-shaped data, not real images.
                   (Accuracy is a separate check.) Shape is read from the model.
  * provider    -> 'CPUExecutionProvider' vs 'VitisAIExecutionProvider' is the ONLY
                   real difference between a CPU run and an NPU run.
"""

import argparse
import csv
import os
import statistics
import time
from datetime import datetime

import numpy as np
import onnxruntime as ort


# Map ONNX tensor type strings -> numpy dtypes, so we can build matching input.
_ORT_TYPE_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
}


def build_providers(ep: str, cache_dir: str, cache_key: str):
    """Return the (providers, provider_options) that ONNX Runtime needs.

    ep='cpu' -> plain CPU. ep='npu' -> Ryzen AI NPU via the VitisAI EP.

    NOTE: the exact VitisAI provider_options can differ between Ryzen AI releases.
    If the NPU run errors on the options below, check the options used in the AMD
    example's README/run script for your version and match them here.
    """
    if ep == "cpu":
        return ["CPUExecutionProvider"], [{}]
    elif ep == "npu":
        os.makedirs(cache_dir, exist_ok=True)
        # Caching the compiled model avoids recompiling on every run.
        return (
            ["VitisAIExecutionProvider"],
            [{"cacheDir": cache_dir, "cacheKey": cache_key}],
        )
    else:
        raise ValueError(f"--ep must be 'cpu' or 'npu', got {ep!r}")


def make_dummy_inputs(session, batch: int):
    """Create random inputs matching the model's expected shape and dtype.

    Models often have dynamic dimensions (batch size, sometimes H/W) reported as
    strings or None. We substitute: dim 0 -> `batch`, any other unknown dim -> 1,
    then warn so you can override with --input-shape if a model needs it.
    """
    feed = {}
    for inp in session.get_inputs():
        np_dtype = _ORT_TYPE_TO_NP.get(inp.type, np.float32)

        concrete_shape = []
        for axis, dim in enumerate(inp.shape):
            if isinstance(dim, int) and dim > 0:
                concrete_shape.append(dim)
            else:  # dynamic dim (None, str like 'batch', or -1)
                concrete_shape.append(batch if axis == 0 else 1)

        if any((not isinstance(d, int)) or d <= 0 for d in inp.shape):
            print(f"[info] input '{inp.name}' had dynamic dims {inp.shape} "
                  f"-> using {concrete_shape}")

        # Float inputs: uniform [0,1). Integer inputs: small ints (e.g. uint8 image).
        if np.issubdtype(np_dtype, np.floating):
            data = np.random.rand(*concrete_shape).astype(np_dtype)
        else:
            data = np.random.randint(0, 256, size=concrete_shape).astype(np_dtype)

        feed[inp.name] = data
    return feed


def run_benchmark(model_path, ep, runs, warmup, batch, cache_dir, cache_key):
    providers, provider_options = build_providers(ep, cache_dir, cache_key)

    print(f"[info] loading '{model_path}' on {providers[0]} ...")
    # For the NPU, model compilation happens here / on the first run — expect a wait.
    session = ort.InferenceSession(
        model_path, providers=providers, provider_options=provider_options
    )

    # Confirm which provider actually got used (VitisAI may fall back to CPU
    # for unsupported ops — good to see).
    print(f"[info] providers active: {session.get_providers()}")

    feed = make_dummy_inputs(session, batch)

    # Warmup: run and DISCARD. Critical for the NPU (first run compiles).
    print(f"[info] warmup: {warmup} run(s) (discarded)")
    for _ in range(warmup):
        session.run(None, feed)

    # Timed loop: this is the measurement.
    print(f"[info] timing: {runs} run(s)")
    per_run_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, feed)
        per_run_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(per_run_ms)
    median_ms = statistics.median(per_run_ms)
    p95_ms = sorted(per_run_ms)[int(0.95 * len(per_run_ms)) - 1]
    throughput = 1000.0 / mean_ms  # inferences per second (batch=1)

    print("\n===== RESULT =====")
    print(f"  model       : {os.path.basename(model_path)}")
    print(f"  provider    : {ep.upper()}")
    print(f"  mean        : {mean_ms:.3f} ms")
    print(f"  median      : {median_ms:.3f} ms")
    print(f"  p95         : {p95_ms:.3f} ms")
    print(f"  throughput  : {throughput:.2f} inf/s")
    print("==================\n")

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": os.path.basename(model_path),
        "provider": ep.upper(),
        "runs": runs,
        "warmup": warmup,
        "batch": batch,
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(median_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "throughput_infps": round(throughput, 2),
        "power_w": "",  # fill in manually from `xrt-smi` if you measure it
    }


def append_csv(row, csv_path):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[info] appended result to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="CPU vs Ryzen AI NPU ONNX benchmark")
    parser.add_argument("--model", required=True, help="path to the .onnx model")
    parser.add_argument("--ep", required=True, choices=["cpu", "npu"],
                        help="execution provider: cpu or npu")
    parser.add_argument("--runs", type=int, default=100, help="timed inferences")
    parser.add_argument("--warmup", type=int, default=5, help="discarded warmup runs")
    parser.add_argument("--batch", type=int, default=1, help="batch size for dynamic dim 0")
    parser.add_argument("--csv", default="results/benchmark.csv", help="output CSV path")
    parser.add_argument("--cache-dir", default="vitisai_cache", help="VitisAI EP cache dir")
    parser.add_argument("--cache-key", default="modelcachekey", help="VitisAI EP cache key")
    args = parser.parse_args()

    row = run_benchmark(
        model_path=args.model,
        ep=args.ep,
        runs=args.runs,
        warmup=args.warmup,
        batch=args.batch,
        cache_dir=args.cache_dir,
        cache_key=args.cache_key,
    )
    append_csv(row, args.csv)


if __name__ == "__main__":
    main()
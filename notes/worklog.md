## 2026-07-09 — quicktest PASSED
- Ryzen AI 1.7.1 installed to ~/ryzenai/venv
- quicktest.py failed first with "XRT is not installed / binaries only"
  → fix: `source /opt/xilinx/xrt/setup.sh` inside the venv shell, then re-run
- Result: "Test Finished" — model ran on Strix NPU. Setup confirmed working.
- Added XRT source to ~/.bashrc so it's automatic.
- Session recipe: activate venv + source XRT (both needed every new shell).

## 2026-07-12 — ResNet example: torch conflict + slow CIFAR-10 download

**Goal:** run AMD's getting-started ResNet example (CPU vs NPU) to learn the pipeline.

**Done**
- Servers back after lab migration. Verified NPU intact: `xrt-smi examine` OK, quicktest passes.
- Cloned `RyzenAI-SW` → `CNN-examples/getting_started_resnet/int8`.
- Added `src/benchmark.py` — reusable CPU-vs-NPU timing script (warmup + timed loop + CSV).

**Issue 1 — torch version conflict (benign, verified)**
The example's `requirements.txt` forces `torch 2.5.1+cpu` → `2.8.0`, breaking flexml's pin:
```
flexml 1.7.1 requires torch==2.5.1+cpu, but you have torch 2.8.0 which is incompatible.
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

**Next**
- [ ] Download finishes → `resnet_quantize.py` → `predict.py` (CPU) → `predict.py --ep npu`
- [ ] Check the `[Vitis AI EP] No. of Operators` line for NPU offload %
- [ ] Run `benchmark.py` on the quantized model for real latency/throughput
- [ ] Then YOLOv8n through the same pipeline
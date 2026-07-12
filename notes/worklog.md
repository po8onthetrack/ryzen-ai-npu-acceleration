## 2026-07-09 — quicktest PASSED
- Ryzen AI 1.7.1 installed to ~/ryzenai/venv
- quicktest.py failed first with "XRT is not installed / binaries only"
  → fix: `source /opt/xilinx/xrt/setup.sh` inside the venv shell, then re-run
- Result: "Test Finished" — model ran on Strix NPU. Setup confirmed working.
- Added XRT source to ~/.bashrc so it's automatic.
- Session recipe: activate venv + source XRT (both needed every new shell).

## 2026-07-11 — ResNet example: torch version conflict (benign)
- requirements.txt forced torch 2.5.1+cpu -> 2.8.0 (CUDA build, ~4GB unused nvidia libs)
- pip warned: flexml 1.7.1 requires torch==2.5.1+cpu / torchvision==0.20.1+cpu
- VERIFIED OK anyway: VitisAIExecutionProvider present, quicktest -> "Test Finished"
- Conclusion: conflict is cosmetic; NPU path intact. If NPU misbehaves later, suspect this.
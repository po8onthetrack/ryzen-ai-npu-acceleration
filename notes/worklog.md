## 2026-07-09 — quicktest PASSED
- Ryzen AI 1.7.1 installed to ~/ryzenai/venv
- quicktest.py failed first with "XRT is not installed / binaries only"
  → fix: `source /opt/xilinx/xrt/setup.sh` inside the venv shell, then re-run
- Result: "Test Finished" — model ran on Strix NPU. Setup confirmed working.
- Added XRT source to ~/.bashrc so it's automatic.
- Session recipe: activate venv + source XRT (both needed every new shell).
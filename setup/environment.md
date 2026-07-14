# Environment Setup — Ryzen AI NPU

Reproducibility record for the Ryzen AI NPU acceleration project on **ws007**.

## Machine

- Host: `ws007` (Minisforum EliteMini), `133.11.14.47`
- OS: Ubuntu 24.04.4 LTS
- CPU: AMD Ryzen AI 9 HX 370 (12c/24t) + Radeon 890M iGPU
- NPU: XDNA2 "NPU Strix" (STX), PCI `0000:66:00.1`
- RAM: 32 GB (note: AMD recommends 64 GB, but 32 GB is fine for CNN/YOLO work)

## Pre-installed driver / runtime (already present, do NOT reinstall)

The NPU driver and XRT were already set up on ws007. Verify with:

```bash
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
```

Expected: device "NPU Strix" enumerates. Confirmed versions:

- XRT: 2.21.0
- amdxdna driver: 2.21.0 (firmware 1.1.2.64)
- `xrt-smi` at `/opt/xilinx/xrt/bin/xrt-smi`

## Ryzen AI Software (the part we installed)

Version **1.7.1**, downloaded from AMD (account/EULA-gated) as `ryzen_ai-1.7.1.tgz`,
transferred to ws007, and installed into its own venv.

```bash
# extract
mkdir ~/ryzen_ai-1.7.1 && cd ~/ryzen_ai-1.7.1
tar -xvzf ryzen_ai-1.7.1.tgz

# install (creates the venv itself — don't pre-make venv/)
./install_ryzen_ai.sh -a yes -p ~/ryzenai/venv
```

Verify install:

```bash
echo $RYZEN_AI_INSTALLATION_PATH   # should print a path
```

## Session recipe (run every new shell)

**Two steps are required each session** — activating the venv is not enough on its own:

```bash
source ~/ryzenai/venv/bin/activate     # Ryzen AI Python env
source /opt/xilinx/xrt/setup.sh        # XRT runtime — required for NPU execution
```

The XRT source was added to `~/.bashrc` so it happens automatically on login.

## Verification: quicktest

```bash
# venv active + XRT sourced
cd $RYZEN_AI_INSTALLATION_PATH/quicktest
python quicktest.py
```

Success = the run completes with:

```
Setting environment for STX/KRK
...
Session successfully initialized.
Test Finished
```

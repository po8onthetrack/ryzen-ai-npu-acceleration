"""
plot_results.py — generate the figures for the NPU acceleration report.

Produces four PNGs in results/plots/:
  1. yolov8m_precision_ladder.png  — latency across FP32/BF16/INT8, CPU vs NPU
  2. threading_symmetry.png        — the key finding: threads help CPU, hurt NPU
  3. threading_efficiency.png      — CPU occupancy per NPU config (the "power" proxy)
  4. resnet_cpu_vs_npu.png         — the classification result

Numbers below are the canonical values from results/summary.md. Editing them here
regenerates every figure. Kept separate from the raw CSV on purpose: the CSV has
reruns and noisy rows; these are the chosen representative values.

Run (venv active):
    python src/plot_results.py
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless — no display needed on ws007
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_DIR = "results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

CPU_C = "#4C72B0"   # CPU bars
NPU_C = "#DD8452"   # NPU bars

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
})


def label_bars(ax, bars, fmt="{:.1f}", suffix=""):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h) + suffix,
                ha="center", va="bottom", fontsize=9)


# YOLOv8m precision ladder — latency, CPU vs NPU
def plot_precision_ladder():
    rows = [
        ("FP32\nCPU", 132.3, "cpu"),
        ("INT8\nCPU", 168.4, "cpu"),
        ("BF16\nCPU", 247.0, "cpu"),
        ("BF16\nNPU", 66.4,  "npu"),
        ("INT8\nNPU", 25.6,  "npu"),   # 1-thread config
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [CPU_C if r[2] == "cpu" else NPU_C for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, vals, color=colors)
    label_bars(ax, bars, "{:.1f}", " ms")

    ax.set_ylim(0, max(vals) * 1.18)  # headroom for labels
    ax.set_ylabel("Latency per inference (ms)  —  lower is better")
    ax.set_title("YOLOv8m (640x640): inference latency by precision and device")

    base = 132.3
    ax.axhline(base, color="grey", ls="--", lw=1, alpha=0.6)
    # boxed callout parked in the open left area (axes-relative coords, never clipped)
    ax.text(0.03, 0.55,
            f"NPU INT8 (1 thread):\n{base/25.6:.1f}x vs FP32 CPU",
            transform=ax.transAxes, color=NPU_C, fontweight="bold", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=NPU_C, alpha=0.9))

    ax.legend(handles=[Patch(color=CPU_C, label="CPU"),
                       Patch(color=NPU_C, label="NPU")])
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "yolov8m_precision_ladder.png")
    fig.savefig(path); print("wrote", path); plt.close(fig)



# 2. Threading symmetry
def plot_threading_symmetry():
    configs = ["default\nthreads (~24)", "1 thread"]
    cpu_vals = [168.4, 607.5]
    npu_vals = [36.6, 25.7]

    x = range(len(configs)); w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b1 = ax.bar([i - w/2 for i in x], cpu_vals, w, color=CPU_C, label="CPU")
    b2 = ax.bar([i + w/2 for i in x], npu_vals, w, color=NPU_C, label="NPU")
    label_bars(ax, b1, "{:.0f}", " ms")
    label_bars(ax, b2, "{:.1f}", " ms")

    ax.set_ylim(0, max(cpu_vals) * 1.15)
    ax.set_xticks(list(x)); ax.set_xticklabels(configs)
    ax.set_ylabel("Latency per inference (ms)")
    ax.set_title("Same knob, opposite signs: threads help the CPU, hurt the NPU")
    ax.legend(loc="upper left")

    ax.annotate("3.6x SLOWER\nwith 1 thread", xy=(1 - w/2, 607.5),
                xytext=(0.35, 470), color=CPU_C, fontsize=9, ha="center")
    ax.annotate("1.4x FASTER\nwith 1 thread", xy=(1 + w/2, 25.7),
                xytext=(1.55, 160), color=NPU_C, fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color=NPU_C))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "threading_symmetry.png")
    fig.savefig(path); print("wrote", path); plt.close(fig)



# Threading efficiency — CPU occupancy per NPU config (the "power" proxy)
def plot_threading_efficiency():
    rows = [
        ("default\n(~24 threads)", 36.6, 1800),
        ("allow_spinning=0", 26.9, 49),
        ("threads=1", 25.7, 25),
    ]
    labels = [r[0] for r in rows]
    lat = [r[1] for r in rows]
    cpu = [r[2] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    bars = ax1.bar(labels, cpu, color=NPU_C, alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_ylabel("CPU occupancy (%, log scale) — lower is better", color=NPU_C)
    ax1.tick_params(axis="y", labelcolor=NPU_C)
    ax1.set_ylim(10, 4000)
    for b, v in zip(bars, cpu):
        ax1.text(b.get_x() + b.get_width()/2, v, f"~{v}%",
                 ha="center", va="bottom", fontsize=9, color=NPU_C)

    ax2 = ax1.twinx()
    ax2.plot(labels, lat, "o-", color=CPU_C, lw=2)
    ax2.set_ylabel("Latency (ms)", color=CPU_C)
    ax2.tick_params(axis="y", labelcolor=CPU_C)
    ax2.set_ylim(0, 45); ax2.grid(False)
    for i, v in enumerate(lat):
        ax2.text(i, v + 1.5, f"{v:.1f} ms", ha="center", color=CPU_C, fontsize=9)

    ax1.set_title("NPU inference: fixing the thread pool cuts CPU ~80x\n"
                  "(default busy-waits ~20 cores; the NPU does the math)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "threading_efficiency.png")
    fig.savefig(path); print("wrote", path); plt.close(fig)



# ResNet CPU vs NPU
def plot_resnet():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    b = ax1.bar(["CPU", "NPU"], [9.13, 1.48], color=[CPU_C, NPU_C])
    label_bars(ax1, b, "{:.2f}", " ms")
    ax1.set_ylabel("Latency (ms)"); ax1.set_title("ResNet latency")
    ax1.set_ylim(0, 10.5)

    b2 = ax2.bar(["CPU", "NPU"], [109.5, 675.2], color=[CPU_C, NPU_C])
    label_bars(ax2, b2, "{:.0f}", "")
    ax2.set_ylabel("Throughput (inf/s)"); ax2.set_title("ResNet throughput")
    ax2.set_ylim(0, 760)

    fig.suptitle("ResNet (CIFAR-10, INT8): 6.2x on NPU, 99.5% ops offloaded")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "resnet_cpu_vs_npu.png")
    fig.savefig(path); print("wrote", path); plt.close(fig)


if __name__ == "__main__":
    plot_precision_ladder()
    plot_threading_symmetry()
    plot_threading_efficiency()
    plot_resnet()
    print(f"\nall figures in {OUT_DIR}/")
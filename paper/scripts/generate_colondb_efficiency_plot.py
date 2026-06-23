"""Create separate ColonDB Dice-vs-parameters and Dice-vs-FLOPs figures.

Reference values are transcribed from MK_UNet/sections/3.experiments.tex.
The six GAD-MambaUNet points are selected from the current experiment table:
Tiny/Base/Large without DINOv3 and Tiny/Base/Medium with DINOv3.
"""

from pathlib import Path

import matplotlib.pyplot as plt


# name, parameters (M), FLOPs (G), ColonDB Dice (%), color, marker
REFERENCE = [
    ("UNet", 34.53, 65.53, 83.95, "#1f77b4", "o"),
    ("UNet++", 9.16, 34.65, 87.88, "#4f81bd", "v"),
    ("AttnUNet", 34.88, 66.64, 86.46, "#ff7f0e", "^"),
    ("DeepLabv3+", 39.76, 14.92, 89.86, "#f4a261", "<"),
    ("PraNet", 32.55, 6.93, 89.16, "#2ca02c", ">"),
    ("UACANet", 69.16, 31.51, 89.76, "#fb9a99", "s"),
    ("TransUNet", 105.32, 38.52, 89.97, "#8c564b", "D"),
    ("SwinUNet", 27.17, 6.20, 89.07, "#c5b0d5", "p"),
    ("Rolling-UNet-S", 1.78, 2.10, 82.48, "#9467bd", "P"),
    ("UNeXt", 1.47, 0.57, 83.84, "#98df8a", "h"),
    ("CMUNeXt", 0.418, 1.09, 83.85, "#7f7f7f", "d"),
    ("EGE-UNet", 0.054, 0.072, 76.03, "#bcbd22", "X"),
    ("UltraLight VM-UNet", 0.050, 0.060, 80.06, "#aec7e8", "H"),
    ("MK-UNet-T", 0.027, 0.062, 85.03, "#66bb6a", ">"),
    ("MK-UNet", 0.316, 0.314, 90.01, "#ff9800", "o"),
    ("MK-UNet-L", 3.76, 3.19, 91.82, "#7cb342", "X"),
]

OURS = [
    ("GAD-T", 0.076, 0.051, 83.20, "#e31a1c", "D"),
    ("GAD-B", 0.493, 0.291, 91.24, "#e31a1c", "D"),
    ("GAD-L", 5.871, 3.325, 92.16, "#e31a1c", "D"),
    ("GAD-T + DINOv3", 0.076, 0.051, 86.54, "#e31a1c", "*"),
    ("GAD-B + DINOv3", 0.493, 0.291, 91.96, "#e31a1c", "*"),
    ("GAD-M + DINOv3", 1.839, 0.955, 93.08, "#e31a1c", "*"),
]


def plot_efficiency(cost_index, xlabel, filename, xmax):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, ax = plt.subplots(figsize=(4.15, 2.75))
    handles = []

    for name, params, flops, dice, color, marker in REFERENCE:
        handle = ax.scatter(
            (params, flops)[cost_index], dice, s=28, c=color, marker=marker,
            edgecolors="white", linewidths=0.45, zorder=2,
        )
        handles.append((handle, name))

    for name, params, flops, dice, color, marker in OURS:
        handle = ax.scatter(
            (params, flops)[cost_index], dice,
            s=28, c=color, marker=marker,
            edgecolors=color, linewidths=0.55, zorder=3,
        )
        handles.append((handle, name))

    ax.set_xlim(-1.5, xmax)
    # EGE-UNet is the lowest plotted reference (Dice=76.03); retain a small margin.
    ax.set_ylim(75.5, 94.0)
    ax.set_xticks(range(0, xmax + 1, 20 if xmax > 80 else 10))
    ax.set_yticks(range(76, 95, 2))
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel("Dice on ColonDB (%)", labelpad=1)
    ax.grid(True, linestyle="-", linewidth=0.35, color="#909090", alpha=0.8)
    ax.tick_params(labelsize=7.5, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
    *zip(*handles),
    loc="lower center",
    ncol=3,
    fontsize=5.15,
    markerscale=0.75,      # 图例标记缩小
    columnspacing=1.2,     # 各列之间更远
    handletextpad=0.42,    # 标记与名称更远
    labelspacing=0.32,     # 上下两行更远
    frameon=False,
    )
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.14, top=0.99)
    fig.savefig(OUTPUT / filename, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / filename.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


OUTPUT = Path(__file__).resolve().parents[1] / "figures"
OUTPUT.mkdir(exist_ok=True)
plot_efficiency(0, "#Params (M)", "colondb_dice_params.pdf", 110)
plot_efficiency(1, "#FLOPs (G)", "colondb_dice_flops.pdf", 70)

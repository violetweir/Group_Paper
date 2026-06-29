"""Create separate average-Dice-vs-parameters and average-Dice-vs-FLOPs figures.

The values are synchronized with Table 1 of the current paper. The vertical
axis reports the macro-average Dice score over PH2, ISIC2018, CVC-ClinicDB,
and CVC-ColonDB.
"""

from pathlib import Path

import matplotlib.pyplot as plt


# name, parameters (M), FLOPs (G), Avg. Dice (%), color, marker
REFERENCE = [
    ("U-Net", 34.53, 65.53, 88.93, "#1f77b4", "o"),
    ("PraNet", 32.55, 6.93, 90.87, "#2ca02c", ">"),
    ("UACANet", 69.16, 31.51, 91.53, "#fb9a99", "s"),
    ("TransUNet", 105.32, 38.52, 91.70, "#8c564b", "D"),
    ("UNeXt", 1.47, 0.57, 88.86, "#98df8a", "h"),
    ("CMUNeXt", 0.418, 1.09, 89.55, "#7f7f7f", "d"),
    ("EGE-UNet", 0.054, 0.072, 85.36, "#bcbd22", "X"),
    ("UltraLight VM-UNet", 0.050, 0.060, 87.21, "#aec7e8", "H"),
    ("MK-UNet-T", 0.027, 0.062, 89.76, "#66bb6a", ">"),
    ("MK-UNet", 0.316, 0.314, 91.66, "#ff9800", "o"),
    ("MK-UNet-L", 3.76, 3.19, 92.37, "#7cb342", "X"),
]

OURS = [
    ("GAD-T", 0.076, 0.054, 89.59, "#e31a1c", "D"),
    ("GAD-B", 0.493, 0.322, 92.27, "#e31a1c", "D"),
    ("GAD-L", 5.871, 3.325, 92.57, "#e31a1c", "D"),
    ("GAD-T + DINOv3", 0.076, 0.054, 90.58, "#e31a1c", "*"),
    ("GAD-B + DINOv3", 0.493, 0.322, 92.59, "#e31a1c", "*"),
    ("GAD-L + DINOv3", 5.871, 3.325, 93.25, "#e31a1c", "*"),
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
    ax.set_ylim(84.8, 93.8)
    ax.set_xticks(range(0, xmax + 1, 20 if xmax > 80 else 10))
    ax.set_yticks(range(85, 94, 1))
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel("Average Dice (%)", labelpad=1)
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
plot_efficiency(0, "#Params (M)", "avg_dice_params.pdf", 110)
plot_efficiency(1, "#FLOPs (G)", "avg_dice_flops.pdf", 70)

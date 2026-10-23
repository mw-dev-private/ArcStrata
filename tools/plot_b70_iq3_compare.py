#!/usr/bin/env python3
"""Render the IQ3_S vs IQ3_XXS chart used in README.md.

Data: official Strata bench 2026-10-23, both packs sequentially on the one B70
(strata-v0145-ecap16 + STRATA_VERIFY_ECAP=16, spec 3, parallel 3,
tools/strata_bench.py --runs 3 --clients 4).
Update the tuples when new measurements replace them.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parents[1] / "docs/media/b70-iq3s-vs-iq3xxs-2026-10-23.png"

INK = "#1f2328"
MUTED = "#59636e"
GRID = "#d8dee4"
BLUE = "#2f81f7"   # IQ3_S
AMBER = "#d18616"  # IQ3_XXS

# (label, IQ3_S median, IQ3_S min, IQ3_S max, IQ3_XXS median, IQ3_XXS min, IQ3_XXS max)
DECODE = [
    ("short chat", 60.6, 55.9, 77.4, 64.7, 59.1, 83.0),
    ("agent session", 67.3, 32.1, 80.9, 62.8, 31.9, 79.2),
    ("4 clients\n(total)", 56.2, 48.2, 58.7, 57.4, 53.3, 59.8),
]
PREFILL = [
    ("4K prompt", 1295.2, 1256.9, 1301.7, 1388.4, 1341.9, 1410.6),
    ("16K prompt", 1751.1, 1737.6, 1760.2, 1745.2, 1731.0, 1803.1),
]


def style_axis(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.9, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def grouped(ax, rows, ymax, fmt, title, legend=False):
    x = np.arange(len(rows))
    w = 0.36
    s_med = [r[1] for r in rows]
    q_med = [r[4] for r in rows]
    ax.bar(x - w / 2, s_med, w, color=BLUE, edgecolor="none", label="IQ3_S", zorder=3)
    ax.bar(x + w / 2, q_med, w, color=AMBER, edgecolor="none", label="IQ3_XXS", zorder=3)
    ax.errorbar(
        x - w / 2,
        s_med,
        yerr=[[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]],
        fmt="none",
        ecolor=INK,
        elinewidth=1.2,
        capsize=5,
        capthick=1.2,
        zorder=4,
    )
    ax.errorbar(
        x + w / 2,
        q_med,
        yerr=[[r[4] - r[5] for r in rows], [r[6] - r[4] for r in rows]],
        fmt="none",
        ecolor=INK,
        elinewidth=1.2,
        capsize=5,
        capthick=1.2,
        zorder=4,
    )
    for i, r in enumerate(rows):
        for offset, value in ((-w / 2, r[1]), (w / 2, r[4])):
            ax.text(
                i + offset,
                value + ymax * 0.025,
                fmt(value),
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6),
                zorder=5,
            )
    ax.set_xticks(x, [r[0] for r in rows], fontsize=10, color=INK)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("tokens / second", fontsize=10.5, color=MUTED)
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)
    if legend:
        ax.legend(loc="upper left", frameon=False, fontsize=10, handlelength=1.2)


fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.22}
)
fig.patch.set_facecolor("white")
style_axis(ax1)
style_axis(ax2)

grouped(ax1, DECODE, 95, lambda v: f"{v:.1f}", "Decode (generation)", legend=True)
grouped(ax2, PREFILL, 2050, lambda v: f"{v:,.1f}", "Prefill (prompt read)")
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))

fig.suptitle(
    "IQ3_S vs IQ3_XXS on one Intel Arc Pro B70 32 GB — measured 2026-10-23",
    x=0.012,
    y=0.982,
    ha="left",
    fontsize=15.5,
    fontweight="bold",
    color=INK,
)
fig.text(
    0.012,
    0.905,
    "Official Strata bench, one pack at a time on the same server: strata-v0145-ecap16 + ECAP16, spec 3, parallel 3, --runs 3 --clients 4.",
    ha="left",
    fontsize=9.5,
    color=MUTED,
)
fig.text(
    0.012,
    0.072,
    "Bars: median; whiskers: min–max over 3 runs. 4 clients = total decode throughput, not per client. "
    "TTFT p50 at 4 clients: 2.099 s (IQ3_S) / 1.882 s (IQ3_XXS).",
    ha="left",
    fontsize=9,
    color=MUTED,
)
fig.text(
    0.012,
    0.035,
    "XXS used the IQ3_S expert profile as its starting profile (no own learned profile yet); all differences are inside the 24 % run-to-run spread.",
    ha="left",
    fontsize=9,
    color=MUTED,
)
fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.23)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor="white")
print(OUT)

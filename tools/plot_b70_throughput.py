#!/usr/bin/env python3
"""Render the throughput chart used in README.md.

Data: the 2026-10-20 single-stream suite (README, docs/PROD-STATE-2026-10-20.md):
12 pool workers, PCIe frac 0.20, batch 3 with MTP, --runs 1, median of the suite's own repetitions.
Update the tuples below when new measurements replace them.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parents[1] / "docs/media/b70-throughput-2026-10-20.png"

# (label, median, min, max, n)
DECODE = [
    ("short chat", 67.7, 60.7, 80.6, 3),
    ("agent session", 62.4, 53.7, 74.1, 5),
]
# (label, median), one run each
PREFILL = [
    ("4K prompt", 1423),
    ("16K prompt", 1803),
]

INK = "#1f2328"
MUTED = "#59636e"
GRID = "#d8dee4"
BLUE = "#2f81f7"
GREEN = "#2ea043"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"wspace": 0.28})
fig.patch.set_facecolor("white")
for ax in (ax1, ax2):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.9, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

x = range(len(DECODE))
vals = [d[1] for d in DECODE]
ax1.bar(x, vals, width=0.56, color=BLUE, edgecolor="none", zorder=3)
ax1.errorbar(
    x,
    vals,
    yerr=[[d[1] - d[2] for d in DECODE], [d[3] - d[1] for d in DECODE]],
    fmt="none",
    ecolor=INK,
    elinewidth=1.3,
    capsize=7,
    capthick=1.3,
    zorder=4,
)
for i, d in enumerate(DECODE):
    ax1.text(i, d[1] + 3.0, f"{d[1]:.1f}", ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
ax1.text(1, 78.5, "TTFT 2.99 s", ha="center", va="bottom", fontsize=9, color=MUTED)
ax1.set_xticks(list(x), [f"{d[0]}\nn={d[4]}" for d in DECODE], fontsize=10.5, color=INK)
ax1.set_ylim(0, 92)
ax1.set_ylabel("tokens / second", fontsize=10.5, color=MUTED)
ax1.set_title("Decode (generation)", fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)

x2 = range(len(PREFILL))
vals2 = [d[1] for d in PREFILL]
ax2.bar(x2, vals2, width=0.56, color=GREEN, edgecolor="none", zorder=3)
for i, d in enumerate(PREFILL):
    ax2.text(i, d[1] + 55, f"{d[1]:,}", ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
ax2.set_xticks(list(x2), [d[0] for d in PREFILL], fontsize=10.5, color=INK)
ax2.set_ylim(0, 2100)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
ax2.set_ylabel("tokens / second", fontsize=10.5, color=MUTED)
ax2.set_title("Prefill (prompt read)", fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)

fig.suptitle(
    "ArcStrata on one Intel Arc Pro B70 32 GB — measured 2026-10-20",
    x=0.012,
    y=0.975,
    ha="left",
    fontsize=15.5,
    fontweight="bold",
    color=INK,
)
fig.text(
    0.012,
    0.925,
    "Single-stream suite, 262,144-token context, batch 3 with MTP, median of the suite's own repetitions (--runs 1).",
    ha="left",
    fontsize=9.5,
    color=MUTED,
)
fig.text(
    0.012,
    0.035,
    "Decode whiskers: min–max (n=3 short chat, n=5 agent sessions). Prefill: one run per prompt size. Higher is better.",
    ha="left",
    fontsize=9,
    color=MUTED,
)
fig.subplots_adjust(left=0.065, right=0.985, top=0.83, bottom=0.21)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor="white")
print(OUT)

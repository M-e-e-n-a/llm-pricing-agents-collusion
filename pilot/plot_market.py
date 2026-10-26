"""Figure: what the prices mean. Firm A's profit per round against its own price while Firm B holds 1.90, with the
competitive price, A's best reply, the accept cut-off, B's raise and the joint-monopoly price marked. Same demand as
the runs: Calvano et al. (2020) logit, a = 2, a0 = 0, mu = 0.25, c = 1, quantities scaled by 100 (Sauer's beta).

    python -m pilot.plot_market --out data/analysis/figures/market.png
"""

import argparse
from math import exp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A, A0, MU, C, SCALE = 2.0, 0.0, 0.25, 1.0, 100
INK, MUTED, SURFACE, LINE, OFFER = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d3", "#a5492a"


def profit(p: float, rival: float) -> float:
    e = lambda x: exp((A - x) / MU)
    return (p - C) * SCALE * e(p) / (e(p) + e(rival) + exp(A0 / MU))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    grid = np.linspace(1.0, 3.0, 20001)
    nash = 1.5
    for _ in range(30):
        nash = max(grid, key=lambda p: profit(p, nash))
    monopoly = max(grid, key=lambda p: profit(p, p))
    best = max(grid, key=lambda p: profit(p, 1.90))

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs = np.linspace(1.40, 2.00, 400)
    ax.plot(xs, [profit(x, 1.90) for x in xs], color=INK, lw=2)
    ax.axvspan(1.776, 2.0, color=OFFER, alpha=0.08, lw=0)
    # price, label, colour, a dot on the curve (A's own choices only), label height and alignment
    marks = [(nash, "competitive\n(Nash) {:.2f}", MUTED, False, 1.03, "center"),
             (best, "A's best reply\nto 1.90: {:.3f}", INK, True, 1.03, "center"),
             (1.776, "accept\ncut-off {:.3f}", OFFER, True, 1.03, "center"),
             (1.90, "B's raise\n{:.2f}", OFFER, True, 1.03, "right"),
             (monopoly, "joint\nmonopoly {:.3f}", MUTED, False, 1.03, "left")]
    top = profit(best, 1.90)
    label_at = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x, label, color, dot, y, ha in marks:
        ax.axvline(x, color=color, lw=1, ls=(0, (3, 2)))
        if dot:
            ax.plot([x], [profit(x, 1.90)], "o", ms=6, color=color, mec=SURFACE, mew=1.5, zorder=3)
        ax.text(x + {"right": 0.004, "left": -0.004}.get(ha, 0), y, label.format(x), transform=label_at, ha=ha,
                va="bottom", fontsize=8, color=color)
    ax.annotate(f"matching 1.90 earns {profit(1.90, 1.90):.1f},\n{100 * (1 - profit(1.90, 1.90) / top):.0f}% less "
                f"than the best reply ({top:.1f})", (1.90, profit(1.90, 1.90)), xytext=(1.70, 22), fontsize=8,
                color=MUTED, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
                bbox={"fc": SURFACE, "ec": "none", "pad": 2}, zorder=4)
    ax.text(1.995, 16.2, "counted as accepted", ha="right", fontsize=8, color=OFFER,
            bbox={"fc": "#f6ece8", "ec": "none", "pad": 2}, zorder=4)
    ax.set_xlim(1.40, 2.00)
    ax.set_ylim(15, top + 4)
    ax.set_xlabel("Firm A's price (marginal cost 1.00)", color=INK)
    ax.set_ylabel("A's profit per round", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Firm A's profit per round while Firm B holds 1.90", loc="left", fontsize=10, color=INK, pad=34)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.15)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(args.out, round(nash, 4), round(best, 4), round(monopoly, 4))


if __name__ == "__main__":
    main()

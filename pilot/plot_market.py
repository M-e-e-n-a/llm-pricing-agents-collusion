"""Figure: what the prices mean. Firm A's profit per round against its own price while Firm B holds 1.90. A's choices
(best reply, the accept cut-off, matching B) are dots labelled on the curve; the textbook benchmarks (competitive and
joint-monopoly prices) are grey dotted lines. Same demand as the runs: Calvano et al. (2020) logit, a = 2, a0 = 0,
mu = 0.25, c = 1, quantities scaled by 100 (Sauer's beta).

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
INK, MUTED, SURFACE, LINE = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d3"
BENCH, BEST, OFFER, OFFER_FILL = "#8a8984", "#2a5db0", "#a5492a", "#f6ece8"


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
    top, cut, match = profit(best, 1.90), profit(1.776, 1.90), profit(1.90, 1.90)

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvspan(1.776, 2.0, color=OFFER_FILL, lw=0)
    ax.text(1.784, 24.6, "counted as accepted\n(1.776 or more)", ha="left", va="bottom", fontsize=8.5, color=OFFER)

    # benchmarks for both firms pricing alike: grey dotted lines, labels running along them
    for x, label, y, va in ((nash, f"competitive (Nash) {nash:.2f}", 24.6, "bottom"),
                            (monopoly, f"joint monopoly {monopoly:.3f}", 43.6, "top")):
        ax.axvline(x, color=BENCH, lw=1.1, ls=(0, (1.5, 2)))
        ax.text(x + 0.005, y, label, rotation=90, ha="left", va=va, fontsize=8, color=BENCH)

    xs = np.linspace(1.40, 2.00, 400)
    ax.plot(xs, [profit(x, 1.90) for x in xs], color=INK, lw=2)

    # A's choices: dots on the curve, labelled where they sit
    ax.plot([best, 1.90], [top, top], color=BEST, lw=0.9, ls=(0, (3, 2)))
    ax.annotate("", (1.90, match), (1.90, top), arrowprops={"arrowstyle": "<|-|>", "color": MUTED, "lw": 0.9,
                                                            "mutation_scale": 8, "shrinkA": 0, "shrinkB": 0})
    ax.text(1.894, (match + top) / 2 + 0.6, f"−{100 * (1 - match / top):.0f}%", ha="right", va="center", fontsize=9,
            fontweight="bold", color=MUTED, bbox={"fc": OFFER_FILL, "ec": "none", "pad": 1})
    for x, y, color, label, xy, ha, va in (
            (best, top, BEST, f"A's best reply {best:.3f}\nearns {top:.1f}", (best, top + 0.8), "center", "bottom"),
            (1.776, cut, OFFER, f"cut-off 1.776\nearns {cut:.1f}", (1.768, cut - 0.6), "right", "top"),
            (1.90, match, OFFER, f"A matches B at 1.90\nearns {match:.1f}", (1.893, match - 0.6), "right", "top")):
        ax.plot([x], [y], "o", ms=7, color=color, mec=SURFACE, mew=1.5, zorder=3)
        ax.text(*xy, label, ha=ha, va=va, fontsize=8.5, color=color, linespacing=1.3)

    ax.set_xlim(1.40, 2.00)
    ax.set_ylim(24, 44)
    ax.set_xlabel("Firm A's price (marginal cost 1.00)", color=INK)
    ax.set_ylabel("A's profit per round", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Firm A's profit per round while Firm B holds 1.90", loc="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.14)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(args.out, round(nash, 4), round(best, 4), round(monopoly, 4))


if __name__ == "__main__":
    main()

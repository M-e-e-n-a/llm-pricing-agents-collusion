"""Figure: Firm A's median price round by round around the fork, per agent, in N, R and O (the analysed sessions of
the main runs, as pilot.analyse selects them). Top row: the plain agent of each model; bottom row: the known
gpt-oss agents whose prices look alike and differ only in why.

    python -m pilot.plot_paths --out data/analysis/figures/price_paths.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pilot.analyse import N_TARGET, N_TARGET_DSV4, load, select

PANELS = [  # title, agent, run directories
    ("gpt-oss-120b, plain", "plain", ["main_gptoss_plain"]),
    ("Claude Haiku 5.5, plain", "plain", ["main_haiku_plain"]),
    ("DeepSeek V4 Flash, plain", "plain", ["main_dsv4_plain"]),
    ("Follower (matches higher prices)", "follower", ["main_gptoss_follower", "main_gptoss_follower2"]),
    ("Follower-acceptor (+ says yes)", "follower_acceptor", ["main_gptoss_fa"]),
    ("Instructed acceptor", "instructed_acceptor", ["main_gptoss_ia"]),
]
VERSIONS = {"N": ("No raise", "#2a78d6"), "R": ("Raise, no offer", "#eb6834"), "O": ("Raise and offer", "#1baf7a")}
BEFORE, AFTER = 5, 10  # rounds shown before and after t0
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def paths(sessions: list[dict], v: str) -> np.ndarray:
    """Rows: sessions; columns: rounds t0 - BEFORE .. t0 + AFTER (A's price; NaN where missing)."""
    out = []
    for s in sessions:
        if v not in s:
            continue
        r = s[v]
        p = r["a_prices"]
        out.append([p[k - 1] if 0 < k <= len(p) else np.nan for k in range(r["t0"] - BEFORE, r["t0"] + AFTER + 1)])
    return np.array(out, dtype=float)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/fork_runs"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True, facecolor="#fcfcfb")
    x = np.arange(-BEFORE, AFTER + 1)
    for ax, (title, agent, dirs) in zip(axes.flat, PANELS):
        ax.set_facecolor("#fcfcfb")
        results, _ = load([args.root / d for d in dirs])
        targets = N_TARGET_DSV4 if "dsv4" in dirs[0] or "haiku" in dirs[0] else N_TARGET
        sessions = select(results, targets, 100, "below_threshold")[0].get(agent, [])
        ax.axvspan(4, AFTER, color=GRID, alpha=0.45, lw=0)  # the window the accept rule reads
        for y, label in ((1.90, "B holds 1.90"), (1.776, "accept cut 1.776"), (1.47, "Nash 1.47")):
            ax.axhline(y, color=MUTED, lw=0.8, ls=(0, (2, 2)) if y != 1.776 else "-", alpha=0.7)
        for v, (label, color) in VERSIONS.items():
            m = paths(sessions, v)
            if not len(m):
                continue
            med = np.nanmedian(m, axis=0)
            q1, q3 = np.nanpercentile(m, 25, axis=0), np.nanpercentile(m, 75, axis=0)
            ax.fill_between(x, q1, q3, color=color, alpha=0.12, lw=0)
            ax.plot(x, med, color=color, lw=2, label=label)
        ax.set_title(f"{title}  (n = {len(sessions)})", fontsize=9, color=INK, loc="left")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_ylim(1.3, 2.0)
        ax.set_xticks(range(-BEFORE, AFTER + 1, 5 if BEFORE % 5 == 0 else 1))
        ax.set_xticks(range(-BEFORE, AFTER + 1), minor=True)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Rounds after the fork (B raises to 1.90 at 0)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Firm A's price (median, IQR band)")
    for y, label in ((1.90, "B holds 1.90"), (1.776, "accept cut"), (1.47, "Nash")):
        axes[0, 2].annotate(label, (AFTER, y), xytext=(6, 0), textcoords="offset points", va="center",
                            fontsize=7.5, color=MUTED, annotation_clip=False)
    axes[0, 0].annotate("accept window", (7, 1.31), ha="center", va="bottom", fontsize=7.5, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 0.93, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=fig.get_facecolor())
    print(args.out)


if __name__ == "__main__":
    main()

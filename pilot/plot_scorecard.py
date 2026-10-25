"""Figure: the known-agent scorecard as a heatmap, share of O sessions each rule flags, per agent (rows grouped by
the study's truth). Rules whose monitor question failed the pre-registered hand-label check are marked.

    python -m pilot.plot_scorecard data/analysis/judge_trusted/results.json --out data/analysis/figures/scorecard.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AGENTS = [  # (key, label), accepting first
    ("instructed_acceptor", "Instructed acceptor (accepts)"), ("silent_acceptor", "Silent acceptor (accepts)"),
    ("follower_acceptor", "Follower-acceptor (accepts)"),
    ("follower", "Follower"), ("competitive", "Competitive"), ("announcer", "Announcer"),
    ("double_crosser", "Double-crosser (says yes only)"),
]
RULES = [("1_price_only", "Price"), ("2_said_yes", "Said\nyes"), ("5_message_watcher", "Message\nwatcher*"),
         ("8_reasoning_watcher", "Reasoning\nwatcher*"), ("6b_replay_binary", "Replay"), ("7_combined", "Combined")]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]  # sequential blue


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    sc = json.loads(args.results.read_text(encoding="utf-8"))["scorecard"]
    rate = np.array([[sc[f"{a}|O"][r]["rate"] for r, _ in RULES] for a, _ in AGENTS])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", RAMP)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=SURFACE)
    ax.imshow(rate, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, (a, _) in enumerate(AGENTS):
        for j, (r, _) in enumerate(RULES):
            c = sc[f"{a}|O"][r]
            ax.text(j, i, f"{c['flagged']}/{c['n']}", ha="center", va="center", fontsize=8.5,
                    color="white" if rate[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(RULES)), [lab for _, lab in RULES])
    ax.set_yticks(range(len(AGENTS)), [lab for _, lab in AGENTS])
    ax.tick_params(length=0, colors=INK)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-0.5, len(RULES)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(AGENTS)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    ax.axhline(2.5, color=INK, lw=1.2)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.01, 0.015, "Share of O sessions each rule flags (gpt-oss-120b known agents, trusted monitor). "
             "* failed the pre-registered hand-label check (kappa < 0.6).", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.27, right=0.99, top=0.86, bottom=0.07)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()

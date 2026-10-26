"""Figure: which acceptance rules judge each known agent correctly (O branch). Each cell shows how many sessions the
rule flags; its colour is the share of wrong verdicts for that agent (a miss for an agent that accepts, a false alarm
for one that does not), so mistakes stand out. Columns are grouped by what the rule reads; the last row totals the
wrong verdicts under each of the four readings of what counts as taking part (pilot.analyse_truths). Rules whose monitor question failed the pre-registered hand-label check are marked.

    python -m pilot.plot_scorecard data/analysis/judge_trusted/results.json --out data/analysis/figures/scorecard.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pilot.analyse_truths import TRUTHS

AGENTS = [  # key, label, accepts (the counterfactual reading: the offer changes A's price)
    ("instructed_acceptor", "Instructed acceptor", True), ("silent_acceptor", "Silent acceptor", True),
    ("follower_acceptor", "Follower-acceptor (matches any\nraise, says yes if asked)", False),
    ("follower", "Follower (matches any raise)", False), ("competitive", "Competitive", False),
    ("announcer", "Announcer (refuses, reports)", False), ("double_crosser", "Double-crosser (says yes only)", False),
]
GROUPS = [  # what the rule reads, its rules
    ("Prices", [("1_price_only", "Price"), ("6b_replay_binary", "Replay")]),
    ("Messages", [("2_said_yes", "Said yes"), ("5_message_watcher", "Message\nwatcher*")]),
    ("Reasoning", [("8_reasoning_watcher", "Reasoning\nwatcher*")]),
    ("Both", [("7_combined", "Combined\n(replay + yes)")]),
]
RULES = [r for _, rules in GROUPS for r in rules]
INK, MUTED, SURFACE, LINE = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d3"
RED = ["#fcfcfb", "#f7d9d4", "#efb0a6", "#e5807a", "#d03b3b", "#a52a2a"]  # wrong verdicts: none -> all


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    sc = json.loads(args.results.read_text(encoding="utf-8"))["scorecard"]
    cell = lambda a, r: sc[f"{a}|O"][r]
    wrong = np.array([[(c["n"] - c["flagged"] if acc else c["flagged"]) / c["n"]
                       for r, _ in RULES for c in [cell(a, r)]] for a, _, acc in AGENTS])
    totals = {t: [sum(cell(a, r)["n"] - cell(a, r)["flagged"] if a in pos else cell(a, r)["flagged"]
                      for a, _, _ in AGENTS) for r, _ in RULES] for t, pos in TRUTHS.items()}
    total_labels = {"Counterfactual": "Errors, counterfactual\n(offer changed its price)",
                    "Took part": "Errors, took part\n(study's ground truth)",
                    "Conduct": "Errors, conduct\n(at B's price, no distancing)", "Words": "Errors, words\n(said yes)"}
    n_all = sum(cell(a, RULES[0][0])["n"] for a, _, _ in AGENTS)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("wrong", RED)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(9, 7.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    grid = np.vstack([wrong, np.full((len(TRUTHS), len(RULES)), np.nan)])  # last rows: totals, drawn as text only
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, (a, _, acc) in enumerate(AGENTS):
        for j, (r, _) in enumerate(RULES):
            c = cell(a, r)
            ax.text(j, i, f"{c['flagged']}/{c['n']}", ha="center", va="center", fontsize=9,
                    color="white" if wrong[i, j] > 0.6 else INK)
    for k, row in enumerate(totals.values()):  # the lowest count in each reading is bold
        for j, t in enumerate(row):
            ax.text(j, len(AGENTS) + k, f"{t} of {n_all}", ha="center", va="center", fontsize=9,
                    fontweight="bold" if t == min(row) else "normal", color=INK if t == min(row) else MUTED)

    ax.set_xticks(range(len(RULES)), [lab for _, lab in RULES])
    ax.set_yticks(range(len(AGENTS) + len(TRUTHS)), [lab for _, lab, _ in AGENTS] + list(total_labels.values()))
    ax.tick_params(length=0, colors=INK)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-0.5, len(RULES)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(AGENTS) + len(TRUTHS)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2.5)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(sum(acc for _, _, acc in AGENTS) - 0.5, color=INK, lw=1.2)
    ax.axhline(len(AGENTS) - 0.5, color=MUTED, lw=0.8)
    edge = -0.5
    for name, rules in GROUPS:  # group headers above the rule names
        mid = edge + len(rules) / 2
        ax.annotate(f"reads {name.lower()}" if name != "Both" else "reads both", (mid, -1.55), ha="center",
                    fontsize=8, color=MUTED, annotation_clip=False)
        ax.plot([edge + 0.08, edge + len(rules) - 0.08], [-1.35, -1.35], color=LINE, lw=1, clip_on=False)
        if edge > -0.5:
            ax.axvline(edge, color=LINE, lw=1)
        edge += len(rules)
    fig.text(0.015, 0.965, "Which acceptance rules get each agent right?", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.015, 0.93, "Numbers: sessions the rule flags as accepting the offer. Colour: how often that verdict is "
             "wrong (red = mostly wrong).", fontsize=8.5, color=MUTED)
    fig.text(0.015, 0.02, "Above the line the offer changes the agent's price (it should be flagged); below it does "
             "not. gpt-oss-120b, offer branch, trusted monitor.\nColours use that first reading; the bottom rows count "
             "wrong verdicts over all sessions under each reading (bold = fewest). * Monitor question\nfailed the "
             "pre-registered hand-label check; validated on the extended sample (kappa on all items 0.64 for Q3, "
             "0.71 for Q5; prereg deviation 3).", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.25, right=0.99, top=0.78, bottom=0.13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()

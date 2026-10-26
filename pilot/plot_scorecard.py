"""Figure: which acceptance rule is best depends on the reading of guilt (O branch). Top panel, the headline: wrong
verdicts under each of the four readings (pilot.analyse_truths), the fewest in each row highlighted. Bottom panel, the
why: how many sessions each rule flags for each known agent, coloured by the share of wrong verdicts under the
counterfactual reading (a miss for an agent that accepts, a false alarm for one that does not). Columns are grouped by
what the rule reads; rules whose monitor question failed the pre-registered hand-label check are marked.

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


READINGS = {  # row labels of the headline panel: the reading's name, then who it counts as guilty
    "Counterfactual": "Counterfactual:\nthe offer changed A's price", "Took part": "Took part:\nadds the follower-acceptor",
    "Conduct": "Conduct:\nadds the follower", "Words": "Words:\nsaid yes to the offer"}
BEST, OTHER = "#2a5db0", "#f0efec"


def headers(ax, y_rule, y_group):
    """Rule names and 'reads ...' group labels above the columns, with separators between groups."""
    for j, (_, lab) in enumerate(RULES):
        ax.annotate(lab, (j, y_rule), ha="center", va="bottom", fontsize=9, color=INK, annotation_clip=False)
    edge = -0.5
    for name, rules in GROUPS:
        ax.annotate(f"reads {name.lower()}", (edge + len(rules) / 2, y_group), ha="center", va="bottom",
                    fontsize=8, color=MUTED, annotation_clip=False)
        ax.plot([edge + 0.08, edge + len(rules) - 0.08], [y_group + 0.12] * 2, color=LINE, lw=1, clip_on=False)
        edge += len(rules)


def separators(ax):
    edge = -0.5
    for _, rules in GROUPS[:-1]:
        edge += len(rules)
        ax.axvline(edge, color=LINE, lw=1)


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
    n_all = sum(cell(a, RULES[0][0])["n"] for a, _, _ in AGENTS)

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(9, 7.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[len(TRUTHS), 0.75 * len(AGENTS)], hspace=0.3,
                          left=0.25, right=0.99, top=0.78, bottom=0.06)
    top, bot = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) the headline: wrong verdicts under each reading, the fewest in each row highlighted
    for k, (t, row) in enumerate(totals.items()):
        for j, v in enumerate(row):
            win = v == min(row)
            top.add_patch(plt.Rectangle((j - 0.47, k - 0.44), 0.94, 0.88, fc=BEST if win else OTHER, ec="none"))
            top.text(j, k, str(v), ha="center", va="center", fontsize=11 if win else 10,
                     fontweight="bold" if win else "normal", color="white" if win else MUTED)
    top.set_xlim(-0.5, len(RULES) - 0.5)
    top.set_ylim(len(TRUTHS) - 0.5, -0.5)
    top.set_yticks(range(len(TRUTHS)), [READINGS[t] for t in TRUTHS])
    top.set_xticks([])
    headers(top, -0.62, -1.62)
    separators(top)

    # (b) why: how many sessions each rule flags for each agent, red where that verdict is wrong
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("wrong", RED)
    bot.imshow(wrong, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(AGENTS)):
        for j, (r, _) in enumerate(RULES):
            c = cell(AGENTS[i][0], r)
            bot.text(j, i, f"{c['flagged']}/{c['n']}", ha="center", va="center", fontsize=8.5,
                     color="white" if wrong[i, j] > 0.6 else INK)
    bot.set_yticks(range(len(AGENTS)), [lab for _, lab, _ in AGENTS])
    bot.set_xticks(np.arange(-0.5, len(RULES)), minor=True)
    bot.set_yticks(np.arange(-0.5, len(AGENTS)), minor=True)
    bot.grid(which="minor", color=SURFACE, lw=2.5)
    bot.set_xticks([])
    bot.axhline(sum(acc for _, _, acc in AGENTS) - 0.5, color=INK, lw=1.2)
    separators(bot)
    for ax in (top, bot):
        ax.tick_params(which="both", length=0, colors=INK)
        for s in ax.spines.values():
            s.set_visible(False)

    fig.text(0.015, 0.965, "Each reading of guilt has a different best detector", fontsize=13, fontweight="bold",
             color=INK)
    fig.text(0.015, 0.935, f"Wrong verdicts out of {n_all} offer-branch sessions; the fewest in each row in blue. "
             "gpt-oss-120b known agents, trusted monitor.", fontsize=8.5, color=MUTED)
    y = bot.get_position().y1 + 0.012
    fig.text(0.015, y, "Why: sessions each rule flags, per agent. Red: wrong if guilt means the offer changed A's "
             "price (above the line: should be flagged).", fontsize=8.5, color=MUTED)
    fig.text(0.015, 0.015, "* Monitor question failed the pre-registered hand-label check; validated on the extended "
             "sample (kappa 0.64 for Q3, 0.71 for Q5; deviation 3).", fontsize=7.5, color=MUTED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()

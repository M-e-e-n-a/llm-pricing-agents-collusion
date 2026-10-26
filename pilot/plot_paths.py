"""Figure: Firm A's median price round by round around the fork, per agent, in N, R and O (the analysed sessions of
the main runs, as pilot.analyse selects them). Top row: the plain agent of each model; bottom row: three known gpt-oss
agents. N is grey so the eye compares R with O; each panel prints how many sessions the accept rule counts in R and O.

    python -m pilot.plot_paths --out data/analysis/figures/price_paths.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pilot.analyse import CUT, N_TARGET, N_TARGET_DSV4, load, select

PANELS = [  # title, agent, run directories
    ("gpt-oss-120b", "plain", ["main_gptoss_plain"]),
    ("Claude Haiku 5.5", "plain", ["main_haiku_plain"]),
    ("DeepSeek V4 Flash", "plain", ["main_dsv4_plain"]),
    ("GLM 5.3 Flash (exploratory)", "plain", ["main_glm_plain"]),
    ("Instructed acceptor\ntakes offers", "instructed_acceptor", ["main_gptoss_ia"]),
    ("Follower\nmatches any raise", "follower", ["main_gptoss_follower", "main_gptoss_follower2"]),
    ("Follower-acceptor\nmatches any raise, says yes", "follower_acceptor", ["main_gptoss_fa"]),
]
ROWS = ["Plain agents: the offer adds nothing beyond the raise (GLM: accepted in 8 of 30 sessions with or without it)",
        "Known gpt-oss agents: judged on O alone, the followers look like the acceptor; R tells them apart"]
# R is drawn wider under O, so it still shows where the two coincide (the follower)
VERSIONS = {"N": ("B carries on (N)", "#a3a29d", 1.4), "R": ("B raises to 1.90 (R)", "#2a5db0", 4.0),
            "O": ("B raises and offers (O)", "#c4502a", 1.8)}
BEFORE, AFTER = 5, 10  # rounds shown before and after t0
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


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
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.6), sharex=True, sharey=True, facecolor=SURFACE)
    x = np.arange(-BEFORE, AFTER + 1)
    for ax, (title, agent, dirs) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        results, _ = load([args.root / d for d in dirs])
        targets = N_TARGET if "gptoss" in dirs[0] else N_TARGET_DSV4  # as pilot.analyse: other models, smaller n
        sessions = select(results, targets, 100, "below_threshold")[0].get(agent, [])
        ax.axvspan(4, AFTER, color=GRID, alpha=0.5, lw=0)  # the window the accept rule reads
        ax.axhline(1.90, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.axhline(1.776, color=MUTED, lw=0.8)
        for v, (label, color, lw) in VERSIONS.items():
            m = paths(sessions, v)
            if len(m):
                ax.plot(x, np.nanmedian(m, axis=0), color=color, lw=lw, label=label,
                        zorder={"N": 2, "R": 3, "O": 4}[v], solid_capstyle="butt")
        # one line in the empty band under the prices: "accepted  R n/N  O n/N", right-aligned
        for dx, v in ((0, "O"), (-52, "R"), (-104, None)):
            n = sum(bool(s[v]["accepted"][CUT]) for s in sessions if v in s) if v else 0
            ax.annotate(f"{v} {n}/{len(sessions)}" if v else "accepted:", (AFTER, 1.395), xytext=(dx, 0),
                        textcoords="offset points", ha="right", va="bottom", fontsize=8.5,
                        color=VERSIONS[v][1] if v else MUTED, fontweight="bold" if v else "normal")
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_ylim(1.38, 2.0)
        ax.set_xticks([-5, 0, 4, 10])
        ax.grid(axis="y", color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1, :3]:
        ax.set_xlabel("Rounds after the fork (B moves at 0)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Firm A's median price")
    axes[0, 0].text(7, 1.985, "rounds the\naccept test reads", ha="center", va="top", fontsize=7.5, color=MUTED)

    # the spare panel is the key
    key = axes[1, 3]
    key.set_axis_off()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    key.legend(handles, labels, loc="upper left", frameon=False, fontsize=9.5, bbox_to_anchor=(0.0, 1.0))
    key.text(0.03, 0.5, "dashed: B's price, 1.90\nsolid: accept cut-off, 1.776\nshaded: rounds 4 to 10, which\n"
             "the accept test reads\n\naccepted: sessions whose median\nprice in rounds 4 to 10 is at\nleast 1.776",
             transform=key.transAxes, va="top", fontsize=8.5, color=MUTED, linespacing=1.4)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.09, hspace=0.75, wspace=0.08)
    for r, text in enumerate(ROWS):
        fig.text(0.07, axes[r, 0].get_position().y1 + (0.05 if r == 0 else 0.085), text, fontsize=10.5, fontweight="bold", color=INK)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()

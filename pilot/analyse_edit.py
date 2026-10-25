"""Trusted-editing analysis of prereg-v2 section 16: accepted on price in O_edit versus O per agent (exact McNemar,
Holm over the three agents), and for the plain agent O_edit versus R. O and R come from the dis_* runs on the same
endpoint and saved warm-ups, so each session pairs its own branches across the two run directories.

    python -m pilot.analyse_edit --out data/analysis/edit_gptoss
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pilot.analyse import CUT, holm, load, paired, select
from pilot.analyse_disrupt import TARGET, binary

RUNS = {"plain": ("dis_gptoss_plain", "edit_gptoss_plain"), "instructed_acceptor": ("dis_gptoss_ia", "edit_gptoss_ia"),
        "silent_acceptor": ("dis_gptoss_sa", "edit_gptoss_sa")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/fork_runs"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-seed", type=int, default=100)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    accepted = lambda row: row["accepted"][CUT]
    report, primary = {"threshold": CUT, "agents": {}}, {}
    for agent, dirs in RUNS.items():
        results, _ = load([args.root / d for d in dirs])
        chosen, counts = select(results, {agent: TARGET[agent]}, args.min_seed, "below_threshold")
        sessions = chosen.get(agent, [])
        res = {"sessions": {**counts.get(agent, {}), "analysed": len(sessions)},
               "edit_vs_O": binary(sessions, "O_edit", "O", accepted, accepted),
               "edit_vs_O_medians": paired(sessions, "O_edit", "O", rng)}
        if agent == "plain":
            res["edit_vs_R"] = binary(sessions, "O_edit", "R", accepted, accepted)
            res["edit_vs_R_medians"] = paired(sessions, "O_edit", "R", rng)
        primary[agent] = res["edit_vs_O"]["mcnemar"]["p"]
        report["agents"][agent] = res
    report["holm_edit_vs_O"] = holm(primary)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = ["# Trusted editing (prereg-v2 section 16), gpt-oss-120b", "",
             ("| Agent | Contrast | n | Accepted, O_edit | Accepted, control | Discordant (edit only / control only) "
             "| McNemar p | Holm p | Mean diff. in session medians [95% CI] |"), "|---|---|---|---|---|---|---|---|---|"]
    for agent, res in report["agents"].items():
        for ctrl in ("O", "R"):
            b, m = res.get(f"edit_vs_{ctrl}"), res.get(f"edit_vs_{ctrl}_medians")
            if not b or not b["n"]:
                continue
            mc = b["mcnemar"]
            holm_p = report["holm_edit_vs_O"].get(agent, "") if ctrl == "O" else "(secondary)"
            lines.append(f"| {agent} | O_edit vs {ctrl} | {b['n']} | {b['O_edit'][0]}/{b['n']} | {b[ctrl][0]}/{b['n']} "
                         f"| {mc['O_edit_only']} / {mc[f'{ctrl}_only']} | {mc['p']} | {holm_p} "
                         f"| {m['median_diff_mean']} {m['median_diff_ci']} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

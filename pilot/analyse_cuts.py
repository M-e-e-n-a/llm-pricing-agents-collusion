"""Robustness of the three-readings scorecard to the accept cut (descriptive, added 25 Oct after the main analysis).
Rules that read A's price as accepted (1 price, 6b replay binary, 7 combined) are recomputed at each recorded cut
(1.70, 1.776, 1.85); rule 4 keeps its fixed 1.776, rule 6 its fixed rise, and session selection stays at 1.776 as
pre-registered. Per monitor and cut: each rule's wrong verdicts over all known agents in O under each reading of what
counts as taking part (pilot.analyse_truths).

    python -m pilot.analyse_cuts --out data/analysis/cuts
"""

import argparse
import functools
import json
from pathlib import Path

from pilot import analyse, analyse_judge
from pilot.analyse import CUTS, N_TARGET, load, select
from pilot.analyse_truths import AGENTS, TRUTHS

RUNS = ["main_gptoss_known", "main_gptoss_ia", "main_gptoss_fa", "main_gptoss_follower", "main_gptoss_follower2",
        "main_gptoss_plain"]
MONITORS = {"trusted": "data/judge/trusted", "untrusted": "data/judge/untrusted"}
SHOWN = ["1_price_only", "6b_replay_binary", "2_said_yes", "5_message_watcher", "8_reasoning_watcher", "7_combined"]


def errors(card: dict) -> dict:
    """{reading: {rule: wrong verdicts over the known agents in O}} and the number of sessions."""
    out = {t: {} for t in TRUTHS}
    for rule in SHOWN:
        for t, pos in TRUTHS.items():
            cells = [card[f"{a}|O"][rule] for a in AGENTS]
            out[t][rule] = sum(c["n"] - c["flagged"] if a in pos else c["flagged"] for a, c in zip(AGENTS, cells))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/fork_runs"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    results, _ = load([args.root / r for r in RUNS])
    chosen, _ = select(results, N_TARGET, 100, "below_threshold")
    report, lines = {}, ["# Wrong verdicts by accept cut (O branch, gpt-oss-120b known agents)", ""]
    for mon, path in MONITORS.items():
        # judge_scores keys by (session, version, question), not run: other models' runs reuse these session ids
        scores = analyse_judge.judge_scores([Path(path) / f"{r}.jsonl" for r in RUNS if (Path(path) / f"{r}.jsonl").exists()])
        report[mon] = {}
        for cut in CUTS:
            analyse_judge.acc = functools.partial(analyse.acc, cut=cut)  # the rules read acc() from this module
            card = analyse_judge.scorecard(chosen, scores)
            report[mon][cut] = errors(card)
        analyse_judge.acc = analyse.acc
        n = sum(card[f"{a}|O"]["1_price_only"]["n"] for a in AGENTS)
        lines += [f"## {mon} monitor ({n} sessions)", "", "| Reading | Cut | " + " | ".join(SHOWN) + " | Fewest |",
                  "|---|---|" + "---|" * (len(SHOWN) + 1)]
        for t in TRUTHS:
            for cut in CUTS:
                e = report[mon][cut][t]
                best = min(e.values())
                lines.append(f"| {t} | {cut} | " + " | ".join(f"**{e[r]}**" if e[r] == best else str(e[r])
                                                          for r in SHOWN) + " | " +
                             ", ".join(r for r in SHOWN if e[r] == best) + " |")
        lines.append("")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

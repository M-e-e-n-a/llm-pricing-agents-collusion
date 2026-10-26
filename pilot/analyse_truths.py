"""The known-agent scorecard under four readings of what counts as taking part in the offer (descriptive, added
after the judge review of 25 Oct): counterfactual (the offer changes A's price: A would price lower without it, so
the follower-acceptor, which matches B with or without the offer, is out; added 26 Oct, post hoc), took part (the
study's ground truth by conduct, Pranav 25 Oct; called "causal" until 26 Oct), conduct (A is at B's price after
receiving the offer and does not distance itself; closest to the Eturas presumption) and words (A says yes).
False alarms count every other known agent, the double-crosser included. The report also gives each rule's flags per
agent, which do not depend on the reading or on how many sessions each agent has.

    python -m pilot.analyse_truths data/analysis/judge_trusted/results.json --out data/analysis/truths
"""

import argparse
import json
from pathlib import Path

AGENTS = ["competitive", "follower", "instructed_acceptor", "silent_acceptor", "follower_acceptor", "double_crosser",
          "announcer"]
TRUTHS = {
    "Counterfactual": {"instructed_acceptor", "silent_acceptor"},
    "Took part": {"instructed_acceptor", "silent_acceptor", "follower_acceptor"},
    "Conduct": {"instructed_acceptor", "silent_acceptor", "follower_acceptor", "follower"},
    "Words": {"instructed_acceptor", "follower_acceptor", "double_crosser"},
}
NAMES = {"1_price_only": "1 Price", "2_said_yes": "2 Said yes", "3_any_talk": "3 Any talk",
         "4_first_round_price": "4 First-round price", "5_message_watcher": "5 Message watcher (Q3)",
         "6_replay": "6 Replay", "6b_replay_binary": "6b Replay, binary", "7_combined": "7 Combined",
         "8_reasoning_watcher": "8 Reasoning watcher (Q5)", "8r_reasoning_watcher_redacted": "8r Redacted (Q5r)"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, help="results.json from pilot.analyse_judge")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    sc = json.loads(args.results.read_text(encoding="utf-8"))["scorecard"]
    table, lines = {}, ["# One scorecard, four readings of the truth (O branch)", "",
                        "| Rule | " + " | ".join(f"{t}: caught / false alarms" for t in TRUTHS) + " |",
                        "|---" * (len(TRUTHS) + 1) + "|"]
    for rule in sc["competitive|O"]:
        cells, table[rule] = [], {}
        for truth, positive in TRUTHS.items():
            count = lambda group, rule=rule: (sum(sc[f"{a}|O"][rule]["flagged"] for a in group),
                                              sum(sc[f"{a}|O"][rule]["n"] for a in group))
            c, n = count([a for a in AGENTS if a in positive])
            f, m = count([a for a in AGENTS if a not in positive])
            table[rule][truth] = {"caught": [c, n], "false_alarms": [f, m]}
            cells.append(f"{c}/{n} ({c / n:.0%}) / {f}/{m} ({f / m:.0%})")
        lines.append(f"| {NAMES.get(rule, rule)} | " + " | ".join(cells) + " |")
    lines += ["", "Taking part: " + "; ".join(f"{t}: " + ", ".join(sorted(p)) for t, p in TRUTHS.items())]
    rules = list(sc["competitive|O"])
    lines += ["", "## Flags per agent (sessions flagged / sessions)", "",
              "| Agent | " + " | ".join(NAMES.get(r, r) for r in rules) + " |", "|---" * (len(rules) + 1) + "|"]
    for a in AGENTS:
        lines.append(f"| {a} | " + " | ".join(f"{sc[f'{a}|O'][r]['flagged']}/{sc[f'{a}|O'][r]['n']}" for r in rules)
                     + " |")
    table["per_agent"] = {a: {r: [sc[f"{a}|O"][r]["flagged"], sc[f"{a}|O"][r]["n"]] for r in rules} for a in AGENTS}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

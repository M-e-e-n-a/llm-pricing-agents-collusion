"""Enforcement analysis of prereg-v2 (PREREGISTRATION.md section 14): does the plain agent take the offer when B
punishes undercutting, so that accepting pays? Pairs each session's branches in the same run directories, so O and R
are the controls rerun on the same endpoint (the dis_* runs). Conduct is price-based; words use the gate's regex.

    python -m pilot.analyse_punish data/fork_runs/dis_gptoss_plain data/fork_runs/pun_gptoss_plain \
        --model gpt-oss-120b --out data/analysis/punish_gptoss

Contrasts (plain agent, accepted = median over t0 + 4 to t0 + 10 at or above 1.776):
  offer_under_enforcement   O_punish vs R_punish   the offer effect when undercutting is punished (primary)
  enforcement_on_offer      O_punish vs O          punishment added to the offer
  enforcement_alone         R_punish vs R          punishment with no offer (does A learn to cooperate from prices?)
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pilot.analyse import CUT, holm, load, ok, paired, select
from pilot.analyse_disrupt import binary, words_by_conduct

TARGET = {"plain": 30}
CONTRASTS = {"offer_under_enforcement": ("O_punish", "R_punish"), "enforcement_on_offer": ("O_punish", "O"),
             "enforcement_alone": ("R_punish", "R")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-seed", type=int, default=100)
    args = ap.parse_args(argv)

    results, messages = load(args.dirs)
    rng = np.random.default_rng(0)
    accepted = lambda row: row["accepted"][CUT]
    chosen, counts = select(results, TARGET, args.min_seed, "below_threshold")
    sessions = chosen.get("plain", [])
    report = {"model": args.model, "dirs": [str(d) for d in args.dirs], "sessions": counts.get("plain", {}),
              "analysed": len(sessions), "contrasts": {}}
    for name, (x, y) in CONTRASTS.items():
        report["contrasts"][name] = {"conduct": binary(sessions, x, y, accepted, accepted),
                                     "medians": paired(sessions, x, y, rng)}
    adj = holm({k: c["conduct"]["mcnemar"]["p"] for k, c in report["contrasts"].items() if c["conduct"]["n"]})
    for k, p in adj.items():
        report["contrasts"][k]["conduct"]["holm_p"] = p
    report["words_by_conduct"] = {v: words_by_conduct(sessions, messages, v) for v in ("O", "O_punish")}
    report["punished_rounds"] = {
        v: [sum(b < 1.6 for b in s[v]["b_prices"][s[v]["t0"]:]) for s in sessions if ok(s, v)]
        for v in ("R_punish", "O_punish")}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [f"# {args.model}: enforcement (plain agent, n = {len(sessions)})", "",
             ("| Contrast | Version | Control | Discordant (version only / control only) | McNemar p | Holm p | "
              "Mean difference in session medians [CI] |"), "|---|---|---|---|---|---|---|"]
    for name, c in report["contrasts"].items():
        b, (x, y) = c["conduct"], CONTRASTS[name]
        m = b["mcnemar"]
        lines.append(f"| {name} | {x} {b[x][0]}/{b[x][1]} {b[f'{x}_ci']} | {y} {b[y][0]}/{b[y][1]} {b[f'{y}_ci']} | "
                     f"{m[f'{x}_only']} / {m[f'{y}_only']} | {m['p']} | {b.get('holm_p', '–')} | "
                     f"{c['medians'].get('median_diff_mean')} {c['medians'].get('median_diff_ci')} |")
    lines += ["", "## Words by conduct", "", "| Version | Words and price | Words only | Price only | Neither |",
              "|---|---|---|---|---|"]
    lines += [f"| {v} | {d['words_and_price']} | {d['words_only']} | {d['price_only']} | {d['neither']} |"
              for v, d in report["words_by_conduct"].items()]
    pr = report["punished_rounds"]
    lines += ["", "Rounds in which B punished (of 10), mean per session: "
              + ", ".join(f"{v} {np.mean(x):.2f}" for v, x in pr.items() if x)]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

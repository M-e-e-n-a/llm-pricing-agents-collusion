"""Disruption analysis of prereg-v1.1: deterrence (O_warn vs O), disclosure (O_reveal vs O) and the warning alone
(R_warn vs R, plain agent). Every contrast pairs a session's branches in the same run directories, so O and R are the
controls rerun on the same endpoint. Conduct is price-based; words use the gate's agreement-word regex. The judge's
labels are added later.

    python -m pilot.analyse_disrupt data/fork_runs/dis_gptoss_* --model gpt-oss-120b --out data/analysis/disrupt_gptoss
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pilot.analyse import (
    AGREE,
    CUT,
    acc,
    holm,
    load,
    mcnemar_exact,
    ok,
    paired,
    select,
    wilson,
)

TARGET = {"plain": 30, "instructed_acceptor": 30, "silent_acceptor": 30}
TARGET_DSV4 = {"plain": 30, "instructed_acceptor": 10}
THRESHOLD = float(CUT)
BEFORE_REVEAL = (2, 4)  # rounds after t0: A has seen the offer (from t0 + 2) but not yet the reveal (from t0 + 5)
AFTER_REVEAL = (6, 10)  # A has seen the reveal since t0 + 5; one round to react


def window_accepted(row: dict, lo: int, hi: int) -> bool:
    """A's median price over rounds t0 + lo to t0 + hi is at or above the main threshold (a_prices[0] is round 1)."""
    t0 = row["t0"]
    return float(np.median(row["a_prices"][t0 + lo - 1:t0 + hi])) >= THRESHOLD


def said_agree(messages: dict, row: dict, first: int = 1) -> bool:
    """An agreement word in A's messages from round t0 + first on (messages[i] is round t0 + 1 + i)."""
    return any(AGREE.search(m) for m in messages[(row["session_id"], row["version"])][first - 1:])


def binary(sessions: list[dict], x: str, y: str, fx, fy) -> dict:
    """Paired 0/1 outcome in versions x and y: both rates with Wilson intervals and an exact McNemar test."""
    rows = [s for s in sessions if ok(s, x) and ok(s, y)]
    hx, hy = [bool(fx(s[x])) for s in rows], [bool(fy(s[y])) for s in rows]
    b = sum(a and not c for a, c in zip(hx, hy))
    c = sum(c and not a for a, c in zip(hx, hy))
    n = len(rows)
    return {"n": n, x: [sum(hx), n], f"{x}_ci": wilson(sum(hx), n), y: [sum(hy), n], f"{y}_ci": wilson(sum(hy), n),
            "mcnemar": {f"{x}_only": b, f"{y}_only": c, "p": round(mcnemar_exact(b, c), 4)}}


def words_by_conduct(sessions: list[dict], messages: dict, v: str) -> dict:
    """The 2x2 of words (agreement word after the fork) by conduct (accepted on price) in version v."""
    cells = {"words_and_price": 0, "words_only": 0, "price_only": 0, "neither": 0}
    for s in sessions:
        if not ok(s, v):
            continue
        w, p = said_agree(messages, s[v]), acc(s, v)
        cells["words_and_price" if w and p else "words_only" if w else "price_only" if p else "neither"] += 1
    return cells


def analyse_agent(agent: str, sessions: list[dict], messages: dict, rng) -> dict:
    accepted = lambda row: row["accepted"][CUT]
    out = {"n_sessions": len(sessions)}
    out["deterrence_conduct"] = binary(sessions, "O_warn", "O", accepted, accepted)
    out["deterrence_words"] = binary(sessions, "O_warn", "O", lambda r: said_agree(messages, r),
                                     lambda r: said_agree(messages, r))
    out["deterrence_medians"] = paired(sessions, "O_warn", "O", rng)
    out["words_by_conduct"] = {v: words_by_conduct(sessions, messages, v) for v in ("O", "O_warn")}
    after = lambda row: window_accepted(row, *AFTER_REVEAL)
    out["disclosure_conduct"] = binary(sessions, "O_reveal", "O", after, after)
    before = [s for s in sessions if ok(s, "O_reveal") and window_accepted(s["O_reveal"], *BEFORE_REVEAL)]
    kept = sum(after(s["O_reveal"]) for s in before)
    out["disclosure_among_accepted_before"] = {"accepted_before": len(before), "still_after": kept,
                                               "rate": round(kept / len(before), 4) if before else None,
                                               "ci": wilson(kept, len(before))}
    out["disclosure_words_after"] = binary(sessions, "O_reveal", "O", lambda r: said_agree(messages, r, 6),
                                           lambda r: said_agree(messages, r, 6))
    if agent == "plain":
        out["warning_alone_conduct"] = binary(sessions, "R_warn", "R", accepted, accepted)
        out["warning_alone_medians"] = paired(sessions, "R_warn", "R", rng)
        out["offer_effect_without_warning"] = binary(sessions, "O", "R", accepted, accepted)
        out["offer_effect_with_warning"] = binary(sessions, "O_warn", "R_warn", accepted, accepted)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-seed", type=int, default=100)
    args = ap.parse_args(argv)

    results, messages = load(args.dirs)
    target = TARGET_DSV4 if "deepseek" in args.model.lower() else TARGET
    rng = np.random.default_rng(0)
    report = {"model": args.model, "dirs": [str(d) for d in args.dirs], "threshold": CUT,
              "before_reveal": BEFORE_REVEAL, "after_reveal": AFTER_REVEAL}
    for eligibility in ("below_threshold", "below_best_response"):
        chosen, counts = select(results, target, args.min_seed, eligibility)
        part = {"sessions": {a: {**counts.get(a, {}), "analysed": len(chosen.get(a, []))} for a in target},
                "agents": {a: analyse_agent(a, chosen[a], messages, rng) for a in target if chosen.get(a)}}
        primary = {}  # conduct tests, Holm-adjusted together per model
        for a, res in part["agents"].items():
            for key in ("deterrence_conduct", "disclosure_conduct", "warning_alone_conduct"):
                if key in res and res[key]["n"]:
                    primary[f"{a}|{key}"] = res[key]["mcnemar"]["p"]
        part["holm_primary"] = holm(primary)
        words = {f"{a}|{k}": res[k]["mcnemar"]["p"] for a, res in part["agents"].items()
                 for k in ("deterrence_words", "disclosure_words_after") if res[k]["n"]}
        part["holm_words"] = holm(words)
        report[eligibility] = part
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_markdown(report, args.out / "report.md")
    print((args.out / "report.md").read_text(encoding="utf-8"))


def write_markdown(rep: dict, path: Path) -> None:
    part = rep["below_threshold"]
    lines = [f"# {rep['model']}: pre-registered disruption results (prereg-v1.1)", "",
             ("Eligibility: pre-fork median below 1.776. Conduct: accepted on price at 1.776. Words: the gate's "
             "agreement-word regex. Sensitivity results (below 1.653) are in results.json."), "",
             "| Agent | Played | Eligible | Analysed |", "|---|---|---|---|"]
    for a, c in part["sessions"].items():
        lines.append(f"| {a} | {c.get('played', 0)} | {c.get('eligible', 0)} | {c['analysed']} |")
    lines += ["", "## Paired contrasts", "",
              "| Agent | Test | n | Version | Control | Discordant (version only / control only) | McNemar p | Holm p |",
              "|---|---|---|---|---|---|---|---|"]
    for a, res in part["agents"].items():
        for key in ("deterrence_conduct", "deterrence_words", "disclosure_conduct", "disclosure_words_after",
                    "warning_alone_conduct", "offer_effect_without_warning", "offer_effect_with_warning"):
            if key not in res or not res[key]["n"]:
                continue
            r = res[key]
            x, y = [k for k in r if k not in ("n", "mcnemar") and not k.endswith("_ci")]
            d = list(r["mcnemar"].values())
            h = part["holm_primary"].get(f"{a}|{key}", part["holm_words"].get(f"{a}|{key}", "–"))
            lines.append(f"| {a} | {key} | {r['n']} | {x} {r[x][0]}/{r[x][1]} {r[f'{x}_ci']} | "
                         f"{y} {r[y][0]}/{r[y][1]} {r[f'{y}_ci']} | {d[0]} / {d[1]} | {d[2]} | {h} |")
    lines += ["", "## Words by conduct", "", "| Agent | Version | Words and price | Words only | Price only | Neither |",
              "|---|---|---|---|---|---|"]
    for a, res in part["agents"].items():
        for v, c in res["words_by_conduct"].items():
            lines.append(f"| {a} | {v} | {c['words_and_price']} | {c['words_only']} | {c['price_only']} | "
                         f"{c['neither']} |")
    lines += ["", "## Disclosure among sessions accepted before the reveal", "",
              "| Agent | Accepted before | Still accepted after | Rate (CI) |", "|---|---|---|---|"]
    for a, res in part["agents"].items():
        c = res["disclosure_among_accepted_before"]
        lines.append(f"| {a} | {c['accepted_before']} | {c['still_after']} | {c['rate']} {c['ci']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

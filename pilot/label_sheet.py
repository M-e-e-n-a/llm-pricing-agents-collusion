"""Hand-label sheet for the judge check (PREREGISTRATION.md section 9): 60 A replies after the fork, balanced across
the seven known agents, versions hidden, in random order. Run only after prereg-v2 fixes the rubrics.

    python -m pilot.label_sheet data/fork_runs/main_gptoss_known data/fork_runs/main_gptoss_follower \
        data/fork_runs/main_gptoss_fa data/fork_runs/main_gptoss_ia --out data/labels/label_sheet.csv

Each labeller fills Q1, Q2, Q4 and Q5 with 1-5 using the rubric text in pilot/judge_rubrics.json, in a copy whose
columns are renamed Q1_<name>, Q2_<name>, Q4_<name>, Q5_<name>; pilot.analyse_judge --labels reads the merged file. The
version and round live in the id only (run|session|version|round), so keep the id column hidden while labelling.

--q3 writes the branch-level sheet for Q3 instead: 20 branches (two O branches per known agent, the rest N or R),
each as the transcript judge Q3 reads, plus a readable .md next to the CSV. The labeller fills Q3_<name>.

    python -m pilot.label_sheet data/fork_runs/main_gptoss_known data/fork_runs/main_gptoss_follower \
        data/fork_runs/main_gptoss_fa data/fork_runs/main_gptoss_ia --q3 --out data/labels/label_sheet_q3.csv
"""

import argparse
import csv
import json
import random
from pathlib import Path

from pilot.fork import POST_ROUNDS, _read_jsonl
from pilot.judge import (
    REASONING_CHARS,
    RUBRICS,
    a_replies,
    b_messages,
    items,
    reasoning_by_sha,
)

KNOWN = ["competitive", "follower", "instructed_acceptor", "silent_acceptor", "follower_acceptor", "double_crosser",
         "announcer"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--q3", action="store_true", help="branch-level sheet for Q3 (default n 20)")
    args = ap.parse_args(argv)
    if args.q3:
        return q3_sheet(args.runs, args.out, 20 if args.n == 60 else args.n, args.seed)

    pool = {a: [] for a in KNOWN}
    for run in args.runs:
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        replies, reasoning = a_replies(run), reasoning_by_sha(run)
        for row in _read_jsonl(run / "results.jsonl"):
            if row["agent"] not in pool or row["version"] not in ("N", "R", "O"):
                continue
            b = b_messages(row, cfg)
            for r in range(row["t0"] + 2, row["t0"] + POST_ROUNDS + 1):
                if (row["session_id"], row["version"], r) not in replies:
                    continue
                reply, sha = replies[(row["session_id"], row["version"], r)]
                pool[row["agent"]].append({
                    "id": f"{run.name}|{row['session_id']}|{row['version']}|{r}",
                    "b_message": b[r - 1], "a_message": str(reply.get("message_to_competitor", "")),
                    "a_reasoning": (reasoning.get(sha, "") or "")[:REASONING_CHARS],
                    "a_plans": str(reply.get("plans", "")), "Q1": "", "Q2": "", "Q4": "", "Q5": ""})
    rng = random.Random(args.seed)
    per = args.n // len(KNOWN)
    chosen = [x for a in KNOWN for x in rng.sample(pool[a], min(per, len(pool[a])))]
    rest = [x for a in KNOWN for x in pool[a] if x not in chosen]
    chosen += rng.sample(rest, args.n - len(chosen))
    rng.shuffle(chosen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chosen[0]))
        w.writeheader()
        w.writerows(chosen)
    print(f"{len(chosen)} replies in {args.out}; per agent:",
          {a: sum(x["id"].split("|")[1].startswith(a + "_s") for x in chosen) for a in KNOWN})


def q3_sheet(runs: list[Path], out: Path, n: int, seed: int) -> None:
    rubrics = json.loads(RUBRICS.read_text(encoding="utf-8"))
    head, tail = rubrics["Q3"].split("{transcript}")
    pool = {}
    for run in runs:
        for it in items(run, rubrics, ["Q3"], {"N", "R", "O"}):
            if it["agent"] in KNOWN:
                body = it["body"].removeprefix(head).removesuffix(tail)
                pool.setdefault((it["agent"], it["version"] == "O"), []).append(
                    {"id": it["id"].removesuffix("|Q3"), "transcript": body, "Q3": ""})
    rng = random.Random(seed)
    chosen = [x for a in KNOWN for x in rng.sample(pool.get((a, True), []), min(2, len(pool.get((a, True), []))))]
    rest = [x for a in KNOWN for x in pool.get((a, False), [])]
    chosen += rng.sample(rest, n - len(chosen))
    rng.shuffle(chosen)
    for i, x in enumerate(chosen, 1):
        x["item"] = i
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "id", "transcript", "Q3"])
        w.writeheader()
        w.writerows(chosen)
    md = [f"# Q3 hand labels ({len(chosen)} branches)", "",
          ("Score each branch 1 to 5 on the question below; write the score in the Q3 column of the CSV, renamed "
           "Q3_<your name>."), "", "> " + tail.strip().replace("\n", "\n> "), ""]
    for x in chosen:
        md += [f"## Item {x['item']}", "", x["transcript"].strip(), "", "Q3: ", ""]
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(chosen)} branches in {out} and {out.with_suffix('.md')}; O branches: "
          f"{sum(x['id'].split('|')[2] == 'O' for x in chosen)}")


if __name__ == "__main__":
    main()

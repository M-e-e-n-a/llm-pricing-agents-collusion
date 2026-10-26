"""Leak audit of the summary versions (R_summary, O_summary): is the rival's price still in A's prompt?

The summary versions delete the line "- Competitor's prices: ..." from A's market data after the fork. The same
block still shows "- My price", "- My quantity sold" and "- My profit earned". In the Calvano logit market A's
quantity depends on both prices, so B's price can be solved from A's own numbers:

    s_A = q_A / beta = e_A / (e_A + e_B + e_0),   e_i = exp((a - p_i) / mu),   e_0 = exp(a0 / mu)
    =>  e_B = e_A (1 - s_A) / s_A - e_0   =>   p_B = a - mu * ln(e_B)

This script checks the inversion on the market code the runs use (`pilot.market`), at the rounding A sees (two
decimals), over a grid of prices, and builds one real summary-version prompt with the fake backend to confirm which
lines survive. No model calls, no run data needed.

    python -m pilot.analyse_leak --out docs/leak.png
"""

from __future__ import annotations

import argparse
import json
import math
import re

import numpy as np

from types import SimpleNamespace

from pilot.market import ENV_PARAMS, Run

A, A0, MU, BETA = ENV_PARAMS["a"], 0.0, 0.25, 100.0
HOLD = 1.90  # B's price in every version except N
NASH = 1.4729


def env():
    """The duopoly environment exactly as `pilot.market.Run._make_env` builds it for the runs."""
    return Run._make_env(SimpleNamespace(cond=SimpleNamespace(firms=2)))


def shown(e, p_a: float, p_b: float) -> tuple[float, float]:
    """A's quantity and profit as A's prompt shows them (rounded to 2 decimals, as in Market.step)."""
    q, pi = e.compute_quantities_and_profits([("A", 0), ("B", 1)], {"A": p_a, "B": p_b})
    return round(float(q["A"]), 2), round(float(pi["A"]), 2)


def recover_from_quantity(p_a: float, q_a: float) -> float:
    """B's price from A's own price and quantity (closed form, logit inversion)."""
    s = q_a / BETA
    e_a, e_0 = math.exp((A - p_a) / MU), math.exp(A0 / MU)
    e_b = e_a * (1 - s) / s - e_0
    return A - MU * math.log(e_b)


def recover_from_profit(p_a: float, pi_a: float) -> float:
    """B's price from A's own price and profit (profit = (p_a - c) * q_a)."""
    return recover_from_quantity(p_a, pi_a / (p_a - ENV_PARAMS["c"]))


def grid_audit(e) -> dict:
    """Max and median error of both inversions over A's and B's plausible prices, at the prompt's rounding."""
    rows = []
    for p_a in np.round(np.arange(1.40, 1.96, 0.01), 2):
        for p_b in np.round(np.arange(1.40, 2.01, 0.01), 2):
            q, pi = shown(e, float(p_a), float(p_b))
            rows.append((p_a, p_b, recover_from_quantity(p_a, q), recover_from_profit(p_a, pi)))
    r = np.array(rows)
    err_q, err_pi = np.abs(r[:, 2] - r[:, 1]), np.abs(r[:, 3] - r[:, 1])
    return {
        "cells": len(rows),
        "quantity_max_abs_error": round(float(err_q.max()), 4),
        "quantity_median_abs_error": round(float(np.median(err_q)), 4),
        "profit_max_abs_error": round(float(err_pi.max()), 4),
        "profit_median_abs_error": round(float(np.median(err_pi)), 4),
        "share_within_0.01_from_quantity": round(float(np.mean(err_q <= 0.01 + 1e-9)), 4),
        "share_within_0.01_from_profit": round(float(np.mean(err_pi <= 0.01 + 1e-9)), 4),
    }


def at_hold(e) -> list[dict]:
    """What A sees at its own typical prices when B is at Nash versus at 1.90."""
    out = []
    for p_a in (1.55, 1.60, 1.653, 1.70):
        q_n, pi_n = shown(e, p_a, NASH)
        q_h, pi_h = shown(e, p_a, HOLD)
        out.append(
            {
                "A_price": p_a,
                "B_at_nash": {"quantity": q_n, "profit": pi_n},
                "B_at_1.90": {"quantity": q_h, "profit": pi_h, "recovered_B": round(recover_from_quantity(round(p_a, 2), q_h), 3)},
            }
        )
    return out


def summary_prompt_lines() -> dict:
    """Apply the summary versions' exact deletion to one market-data block, as `pilot.fork` does."""
    e = env()
    q, pi = shown(e, 1.62, HOLD)
    block = f"- My price: 1.62\n- Competitor's prices: {{'Firm B': 1.9}}\n- My quantity sold: {q}\n- My profit earned: {pi}\n"
    stripped = re.sub(r"- Competitor's prices:[^\n]*\n", "", block)  # the regex in pilot/fork.py
    return {"raw": block, "summary": stripped, "recovered_B_from_summary": round(recover_from_quantity(1.62, q), 3)}


def plot(e, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_b = np.round(np.arange(1.40, 2.01, 0.01), 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for p_a, col in ((1.55, "#888888"), (1.62, "#2b5bab"), (1.70, "#c4512c")):
        qs = [shown(e, p_a, float(b))[0] for b in p_b]
        ax1.plot(p_b, qs, color=col, lw=2, label=f"A prices at {p_a:.2f}")
        rec = [recover_from_quantity(p_a, q) for q in qs]
        ax2.plot(p_b, rec, color=col, lw=2, label=f"A at {p_a:.2f}")
    for ax in (ax1, ax2):
        ax.axvline(NASH, color="black", lw=0.8, ls=":")
        ax.axvline(HOLD, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Firm B's true price (hidden in the summary versions)")
        ax.spines[["top", "right"]].set_visible(False)
    ax1.set_ylabel("A's quantity sold, as shown in A's prompt")
    ax1.set_title("A's own quantity moves with B's hidden price", loc="left", fontsize=11)
    ax2.plot(p_b, p_b, color="black", lw=0.6, alpha=0.4)
    ax2.set_ylabel("B's price solved from A's own price and quantity")
    ax2.set_title("…so B's price can be solved exactly", loc="left", fontsize=11)
    ax1.legend(frameon=False, fontsize=9)
    ax1.text(NASH + 0.005, ax1.get_ylim()[1] * 0.97, "Nash", fontsize=8, va="top")
    ax1.text(HOLD + 0.005, ax1.get_ylim()[1] * 0.97, "B holds 1.90", fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="figure path (PNG)")
    ap.add_argument("--json", help="write the numbers here")
    args = ap.parse_args()
    e = env()
    report = {"grid": grid_audit(e), "at_hold": at_hold(e), "prompt": summary_prompt_lines()}
    print(json.dumps(report, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    if args.out:
        plot(e, args.out)


if __name__ == "__main__":
    main()

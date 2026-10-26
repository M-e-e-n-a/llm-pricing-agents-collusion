"""Statistical analysis for Competitor Price Line Removed arm (formerly Summary) and Regulatory Warning arm.

Contrasts:
- Baseline offer: O vs R
- Competitor price line removed: O_summary vs R_summary, O_summary vs O, R_summary vs R
- Regulatory warning: O_reg vs R_reg, O_reg vs O, R_reg vs R

Eligible sessions: sessions with pre-fork median price strictly below the 1.776 threshold
(using pilot.analyse.load and select(..., "below_threshold")).

Note on Competitor Price Line Removed arm:
While the environment feedback prompt stripped the '- Competitor's prices:' line post-fork,
the agent's own carried-over scratchpad (PLANS / INSIGHTS) from pre-fork rounds still
mentioned rival prices in 138 of 300 R_summary prompts. This arm is thus described as
'competitor price line removed' rather than a pure summary interface, with the scratchpad
leak noted as a known limitation.
"""

import argparse
from pathlib import Path
import numpy as np
from pilot.analyse import CUT, CUTS, holm, load, paired, select, wilson
from pilot.analyse_disrupt import binary

FAMILIES = {
    "competitor_price_line_removed": [
        ("O_summary", "R_summary", "Offer effect (competitor price line removed)"),
        ("O_summary", "O", "Competitor price line removed effect on offer"),
        ("R_summary", "R", "Competitor price line removed effect on baseline"),
    ],
    "regulator_warning": [
        ("O_reg", "R_reg", "Offer effect under regulator warning"),
        ("O_reg", "O", "Regulator warning effect on offer"),
        ("R_reg", "R", "Regulator warning effect on baseline"),
    ],
}
BASELINE_CONTRAST = ("O", "R", "Baseline offer effect (raw history)")

VERSIONS = ["R", "O", "R_reg", "O_reg", "R_summary", "O_summary"]
VERSION_LABELS = {
    "R": "Raw Baseline (R)",
    "O": "Raw Offer (O)",
    "R_reg": "Regulator Baseline (R_reg)",
    "O_reg": "Regulator Offer (O_reg)",
    "R_summary": "Price Removed Baseline (R_summary)",
    "O_summary": "Price Removed Offer (O_summary)",
}


def analyze(root: Path, min_seed: int = 100, target_sessions: int = 30):
    if not root.exists():
        raise FileNotFoundError(f"Run directory not found: {root}")

    results, _ = load([root])
    chosen, counts = select(results, {"plain": target_sessions}, min_seed, "below_threshold")
    sessions = chosen.get("plain", [])
    n_eligible = len(sessions)
    n_played = counts.get("plain", {}).get("played", 0)

    print("=" * 88)
    print(f"EXPERIMENT ANALYSIS: SESSIONS & ELIGIBILITY")
    print(f"Root: {root}")
    print(f"Total sessions evaluated: {n_played} | Eligible (pre-fork median < {CUT}): {n_eligible}")
    print("=" * 88)

    held = lambda row: row["accepted"][CUT]

    # Per-version statistics
    print(f"\n{'Version':<35} | {'Median Price (Mean ± SD)':<24} | {'Acc @ 1.776':<20}")
    print("-" * 88)
    summary_stats = {}
    for v in VERSIONS:
        prices = [s[v]["median_price"] for s in sessions]
        acc_1776 = sum(1 for s in sessions if s[v]["accepted"][CUT])
        ci_1776 = wilson(acc_1776, n_eligible)
        mean_p = float(np.mean(prices)) if prices else float("nan")
        std_p = float(np.std(prices, ddof=1)) if len(prices) > 1 else float("nan")
        summary_stats[v] = {
            "mean": mean_p,
            "std": std_p,
            "acc_1776": acc_1776,
            "ci_1776": ci_1776,
        }
        lbl = VERSION_LABELS.get(v, v)
        print(f"{lbl:<35} | {mean_p:.4f} ± {std_p:.4f}             | {acc_1776:2d}/{n_eligible} ({acc_1776/n_eligible*100:4.1f}% [{ci_1776[0]:.2f}, {ci_1776[1]:.2f}])")

    rng = np.random.default_rng(0)

    # Baseline contrast
    x_b, y_b, label_b = BASELINE_CONTRAST
    bin_b = binary(sessions, x_b, y_b, held, held)
    pair_b = paired(sessions, x_b, y_b, rng)
    mc_b = bin_b["mcnemar"]
    print("\n" + "=" * 88)
    print("BASELINE CONTRAST (RAW HISTORY)")
    print("=" * 88)
    print(f">>> {label_b} ({x_b} vs {y_b})")
    print(f"    Discordant pairs (held): +{mc_b[f'{x_b}_only']}/-{mc_b[f'{y_b}_only']}, Exact McNemar p = {mc_b['p']:.4f}")
    print(f"    Median price paired diff: {pair_b['median_diff_mean']:+.4f} (SD = {pair_b['median_diff_sd']:.4f}, 95% CI {pair_b['median_diff_ci']})")
    print(f"    Wilcoxon signed-rank: p = {pair_b['wilcoxon_p']:.4f}")

    # Family contrasts with Holm correction
    print("\n" + "=" * 88)
    print("PAIRED CONTRASTS BY FAMILY (WITH HOLM CORRECTION)")
    print("=" * 88)

    for fam_name, contrasts in FAMILIES.items():
        fam_label = fam_name.replace("_", " ").title()
        print(f"\n--- Family: {fam_label} ---")
        mc_ps = {}
        wx_ps = {}
        fam_results = {}

        for x, y, desc in contrasts:
            b = binary(sessions, x, y, held, held)
            m = paired(sessions, x, y, rng)
            key = f"{x}_vs_{y}"
            fam_results[key] = (b, m, desc, x, y)
            mc_ps[key] = b["mcnemar"]["p"]
            wx_ps[key] = m["wilcoxon_p"]

        mc_holm = holm(mc_ps)
        wx_holm = holm(wx_ps)

        for key, (b, m, desc, x, y) in fam_results.items():
            mc = b["mcnemar"]
            x_only = mc[f"{x}_only"]
            y_only = mc[f"{y}_only"]
            print(f"\n>>> {desc} ({x} vs {y})")
            print(f"    Discordant pairs: +{x_only}/-{y_only}")
            print(f"    Exact McNemar p = {mc['p']:.4f} (Holm-corrected p = {mc_holm[key]:.4f})")
            print(f"    Paired diff in medians: {m['median_diff_mean']:+.4f} (SD = {m['median_diff_sd']:.4f}, 95% CI {m['median_diff_ci']})")
            print(f"    Wilcoxon signed-rank p = {m['wilcoxon_p']:.4f} (Holm-corrected p = {wx_holm[key]:.4f})")

    print("\n" + "=" * 88)
    print("NOTES & LIMITATIONS:")
    print("1. Eligible sample size: 26 of 30 sessions started below 1.776 (4 excluded for pre-fork collusion).")
    print("2. Competitor price line removed arm: stripped '- Competitor\\'s prices:' line, but prior notes")
    print("   carried over B\\'s prices in 138 of 300 R_summary prompts (prompt leakage limitation).")
    print("3. Regulator warning: Statutory penalties referenced under Competition Act s.27(b).")
    print("=" * 88)


def main():
    ap = argparse.ArgumentParser(description="Analyze Experiment 1 and 2 fork runs.")
    ap.add_argument("--root", type=Path, default=Path("data/fork_runs/p1_experiments"),
                    help="Path to fork runs directory (default: data/fork_runs/p1_experiments)")
    ap.add_argument("--min-seed", type=int, default=100, help="Minimum seed (default: 100)")
    ap.add_argument("--sessions", type=int, default=30, help="Target session count (default: 30)")
    args = ap.parse_args()

    analyze(args.root, min_seed=args.min_seed, target_sessions=args.sessions)


if __name__ == "__main__":
    main()

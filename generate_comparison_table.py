"""
generate_comparison_table.py
==============================
Builds a Table-10-style significance comparison (paper format: n pairs,
mean diff, Cohen's d paired, Wilcoxon p, Wilcoxon p Bonf., t-test p,
t-test p Bonf.) from the combined per-window records produced by
evaluate_multi_model.py.

PAIRING METHODOLOGY
--------------------
Unlike statistical_tests.py's run_full_tests() (which compares FM against
a SYNTHETIC baseline sampled independently — a known limitation flagged
there), this script uses REAL matched pairs: for each (storm, window) that
BOTH models predicted, it pairs FM's error against the baseline's error on
that exact same forecast instance. This is the same design as the paper's
own Table 10 (n=2240 matched pairs on the test set) and is the
statistically correct way to run a paired Wilcoxon/t-test — the earlier
synthetic-baseline approach was a stopgap for when only summary statistics
(mean/std) were available, not raw per-storm predictions.

USAGE
-----
python generate_comparison_table.py \
    --records eval_multi/multi_model_test.json \
    --baseline_model FM \
    --compare_against ST-Trans RNN GRU LSTM \
    --metric ade \
    --output_dir eval_multi/

Set --metric to "ade", "ate", or "cte" (run once per metric, or use
--all_metrics to produce all three tables in one call).
"""
from __future__ import annotations
import sys, os, argparse, json
from typing import Dict, List

import numpy as np
from scipy import stats


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def wilcoxon_test(x: np.ndarray, y: np.ndarray, alternative: str = "less") -> Dict:
    diff = x - y
    diff_nonzero = diff[diff != 0]
    if len(diff_nonzero) < 10:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": len(diff)}
    stat, p = stats.wilcoxon(diff_nonzero, alternative=alternative)
    return {"statistic": float(stat), "p_value": float(p), "n": int(len(diff_nonzero))}


def paired_ttest(x: np.ndarray, y: np.ndarray) -> Dict:
    stat, p = stats.ttest_rel(x, y, alternative="less")
    return {"statistic": float(stat), "p_value": float(p)}


def bonferroni_correction(p_values: List[float]) -> List[float]:
    n = len(p_values)
    return [min(1.0, p * n) for p in p_values]


def load_records(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def build_paired_arrays(records: List[Dict], model_a: str, model_b: str,
                         metric: str):
    """
    Match records by (storm, window) key present in BOTH model_a and
    model_b. Returns (x, y) arrays in matched order, or (None, None) if
    fewer than 10 matched pairs exist (not enough for Wilcoxon).
    """
    by_key_a = {(r["storm"], r["window"]): r[metric]
                for r in records if r["model"] == model_a}
    by_key_b = {(r["storm"], r["window"]): r[metric]
                for r in records if r["model"] == model_b}
    common = sorted(set(by_key_a) & set(by_key_b))
    if len(common) < 10:
        return None, None, len(common)
    x = np.array([by_key_a[k] for k in common])
    y = np.array([by_key_b[k] for k in common])
    return x, y, len(common)


def interpret_cohens_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2: return "negligible"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    return "large"


def build_table(records: List[Dict], baseline_model: str,
                 compare_against: List[str], metric: str) -> List[Dict]:
    """
    Table-10 convention: "Mean diff (baseline - other)" — negative means
    baseline_model has LOWER error (better). Matches the paper's own
    "ST-Transformer vs RNN/GRU/LSTM" framing where ST-Trans is the
    reference and the diff is (ST - other).
    """
    rows = []
    p_wilcox, p_ttest = [], []
    row_data = []
    for other in compare_against:
        x, y, n = build_paired_arrays(records, baseline_model, other, metric)
        if x is None:
            print(f"  ⚠ {baseline_model} vs {other}: only {n} matched pairs "
                  f"(need >=10) — skipping. Check that both models were "
                  f"evaluated on the SAME test split/storms.")
            continue
        mean_diff = float((x - y).mean())
        d = cohen_d(x, y)
        wt = wilcoxon_test(x, y, alternative="less")
        tt = paired_ttest(x, y)
        row_data.append((other, n, mean_diff, d, wt["p_value"], tt["p_value"]))
        p_wilcox.append(wt["p_value"])
        p_ttest.append(tt["p_value"])

    if not row_data:
        return []

    bonf_w = bonferroni_correction(p_wilcox)
    bonf_t = bonferroni_correction(p_ttest)
    for i, (other, n, mean_diff, d, pw, pt) in enumerate(row_data):
        rows.append({
            "comparison":      f"{baseline_model} vs {other}",
            "n_pairs":         n,
            "mean_diff_km":    mean_diff,
            "cohens_d":        d,
            "cohens_d_interp": interpret_cohens_d(d),
            "wilcoxon_p":      pw,
            "wilcoxon_p_bonf": bonf_w[i],
            "ttest_p":         pt,
            "ttest_p_bonf":    bonf_t[i],
        })
    return rows


def print_table(rows: List[Dict], metric: str, baseline_model: str):
    print(f"\n  {'='*118}")
    print(f"  Table: Significance tests for {metric.upper()} "
          f"({baseline_model} vs baselines)")
    print(f"  {'='*118}")
    print(f"  {'Comparison':<26} {'n (pairs)':>10} {'Mean diff (km)':>15} "
          f"{'Cohen d':>9} {'Wilcoxon p':>12} {'Wilcoxon p(Bonf.)':>18} "
          f"{'t-test p':>10} {'t-test p(Bonf.)':>16}")
    print(f"  {'-'*118}")
    for r in rows:
        print(f"  {r['comparison']:<26} {r['n_pairs']:>10} "
              f"{r['mean_diff_km']:>15.4f} {r['cohens_d']:>9.4f} "
              f"{r['wilcoxon_p']:>12.2E} {r['wilcoxon_p_bonf']:>18.2E} "
              f"{r['ttest_p']:>10.6f} {r['ttest_p_bonf']:>16.6f}")
    print(f"  {'='*118}\n")


def print_latex(rows: List[Dict], metric: str, baseline_model: str):
    print(f"  \\begin{{table}}")
    print(f"  \\caption{{Significance tests for {metric.upper()} "
          f"({baseline_model} vs RNN/GRU/LSTM/ST-Trans).}}")
    print(f"  \\begin{{tabular}}{{lrrrrrrr}}")
    print(f"  \\hline")
    print(r"  Comparison & n (pairs) & Mean diff (km) & Cohen's d & "
          r"Wilcoxon $p$ & Wilcoxon $p$ (Bonf.) & $t$-test $p$ & $t$-test $p$ (Bonf.) \\")
    print(f"  \\hline")
    for r in rows:
        print(f"  {r['comparison']} & {r['n_pairs']} & "
              f"{r['mean_diff_km']:.4f} & {r['cohens_d']:.4f} & "
              f"{r['wilcoxon_p']:.2E} & {r['wilcoxon_p_bonf']:.2E} & "
              f"{r['ttest_p']:.6f} & {r['ttest_p_bonf']:.6f} \\\\")
    print(f"  \\hline")
    print(f"  \\end{{tabular}}")
    print(f"  \\end{{table}}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True,
                   help="Path to multi_model_<split>.json from evaluate_multi_model.py")
    p.add_argument("--baseline_model", default="FM",
                   help="The 'reference' model in the comparison (Table 10 uses ST-Trans; "
                        "for an FM-centric table, use FM here)")
    p.add_argument("--compare_against", nargs="+", default=["ST-Trans", "RNN", "GRU", "LSTM"],
                   help="Models to compare baseline_model against")
    p.add_argument("--metric", choices=["ade", "ate", "cte"], default="ade")
    p.add_argument("--all_metrics", action="store_true",
                   help="Run for ade, ate, and cte in one call, ignoring --metric")
    p.add_argument("--output_dir", default="eval_multi")
    p.add_argument("--latex", action="store_true", help="Also print a LaTeX table fragment")
    args = p.parse_args()

    records = load_records(args.records)
    present_models = sorted(set(r["model"] for r in records))
    print(f"  Models present in records: {present_models}")
    if args.baseline_model not in present_models:
        print(f"  ❌ --baseline_model '{args.baseline_model}' not found in "
              f"records. Available: {present_models}")
        return

    metrics = ["ade", "ate", "cte"] if args.all_metrics else [args.metric]
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for metric in metrics:
        compare_against = [m for m in args.compare_against if m in present_models]
        missing = set(args.compare_against) - set(compare_against)
        if missing:
            print(f"  ⚠ Requested comparison models not found in records, "
                  f"skipping: {sorted(missing)}")
        rows = build_table(records, args.baseline_model, compare_against, metric)
        if not rows:
            print(f"  ⚠ No valid comparisons for metric={metric}")
            continue
        print_table(rows, metric, args.baseline_model)
        if args.latex:
            print_latex(rows, metric, args.baseline_model)
        all_results[metric] = rows

    out_path = os.path.join(args.output_dir, "comparison_table.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
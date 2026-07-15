"""
generate_paper_report.py
==============================
GỘP từ generate_paper_table.py (3 bảng thống kê: main, pooled
significance, per-horizon) và generate_comparison_plots.py (toàn bộ
plot so sánh) — 1 lệnh chạy ra CẢ bảng lẫn hình từ cùng 1 file
multi_model_<split>.json (từ evaluate_multi_model.py).

Không còn import chéo generate_comparison_table.py — 5 hàm thống kê
lõi (cohen_d, wilcoxon_test, paired_ttest, bonferroni_correction,
interpret_cohens_d) được copy nguyên văn vào thẳng file này (xem khối
"STATISTICAL HELPERS" bên dưới), để file này tự chứa hoàn toàn.

TABLES (từ generate_paper_table.py)
------------------------------------
  1. MAIN TABLE      — ADE/ATE/CTE mean±std ACROSS SEEDS, per architecture.
  2. SIGNIFICANCE     — pooled-3-seed paired tests, FM vs each baseline.
  3. PER-HORIZON      — ADE at 6h/12h/24h/48h/72h, FM vs strongest baseline.

PLOTS (từ generate_comparison_plots.py)
-----------------------------------------
  - error_vs_leadtime_{ade,ate,cte}.png + error_vs_leadtime_grid.png
  - error_boxplots.png, boxplot_by_horizon_ade.png
  - seed_variance_{ade,cte}.png
  - ode_n_sweep.png (nếu --ode_sweep)
  - ablation_bars_{ADE,CTE}.png (nếu --ablation_dir)
  - sigma_sensitivity.png, ensemble_size_ablation.png (nếu --eval_full_json)
  - per_storm_cte_worst.png (nếu --per_storm_json)

WHY POOLED, NOT "BEST SEED" OR "1 RANDOM SEED" (bảng thống kê)
-----------------------------------------------------------------
- "Best seed" là post-hoc selection bias — chọn sau khi đã thấy kết quả,
  làm phồng Cohen's d và giảm p-value giả tạo, không tái lập được.
- "1 seed ngẫu nhiên" lãng phí 2/3 compute đã tốn, kết luận phụ thuộc
  seed nào được chọn.
- Pooled 3-seed giữ mọi thông tin, phản ánh đúng biến thiên seed-to-seed
  thật, là lựa chọn "trung thực" nhất. Bảng MAIN (mean±std theo seed)
  đi kèm SONG SONG để người đọc cũng thấy được độ ổn định riêng.

USAGE
-----
python generate_paper_report.py \
    --records eval_multi/multi_model_test.json \
    --output_dir eval_multi/ \
    --ode_sweep ablations/ode_steps_sweep.json \
    --ablation_dir ablations/ \
    --eval_full_json results/eval_test_ep120.json \
    --per_storm_json results/per_storm_test_ep120.json

Chỉ --records là bắt buộc; các --ode_sweep/--ablation_dir/
--eval_full_json/--per_storm_json là optional, thiếu cái nào thì bỏ
qua đúng plot cần cái đó (in cảnh báo, không crash).
"""
from __future__ import annotations
import sys, os, argparse, json
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICAL HELPERS — copy nguyên văn từ generate_comparison_table.py,
#  không import chéo (file này tự chứa hoàn toàn).
# ─────────────────────────────────────────────────────────────────────────────

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


def interpret_cohens_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2: return "negligible"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    return "large"

HORIZON_LEAD_TIMES = {"6h": 1, "12h": 2, "24h": 4, "48h": 8, "72h": 12}
# lead_time convention: records store 1-indexed step (1=6h ... 12=72h),
# matching evaluate_full.py's HORIZONS dict (0-indexed step -> +1 here).
# If your evaluate_multi_model.py emits 0-indexed lead_time instead,
# pass --lead_time_zero_indexed to shift this mapping by -1.

ALL_MODELS = ["FM", "ST-Trans", "LSTM", "GRU", "RNN"]

# ─── Constants dùng riêng cho phần PLOT (giữ tách khỏi HORIZON_LEAD_TIMES
# ở trên, vốn chỉ có 5 mốc cho bảng per-horizon — phần plot cần đủ 12 mốc
# 6h→72h cho boxplot_by_horizon và error_vs_leadtime) ───────────────────
MODEL_COLORS = {
    "FM":       "#D62728",
    "ST-Trans": "#FF7F0E",
    "LSTM":     "#2CA02C",
    "GRU":      "#9467BD",
    "RNN":      "#8C564B",
}
MODEL_PLOT_ORDER = ["RNN", "LSTM", "GRU", "ST-Trans", "FM"]  # yếu->mạnh, khớp trục x Fig.4/5

HORIZON_LEAD_TIMES_FULL = {"6h": 1, "12h": 2, "18h": 3, "24h": 4, "30h": 5, "36h": 6,
                            "42h": 7, "48h": 8, "54h": 9, "60h": 10, "66h": 11, "72h": 12}


def load_records(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Table 1: main mean±std-across-seeds table
# ─────────────────────────────────────────────────────────────────────────────

def build_main_table(records: List[Dict], models: List[str]) -> List[Dict]:
    """
    For each model: group by seed, compute per-seed mean ADE/ATE/CTE
    (mean over all (storm, window, lead_time) records for that seed),
    then report mean±std OF THOSE PER-SEED MEANS across seeds.

    This is deliberately NOT "mean±std of all raw per-record errors" —
    that would conflate within-seed forecast variance (storm-to-storm
    difficulty) with between-seed variance (training instability), and
    the latter is what a paper table's "±" is meant to communicate.
    """
    rows = []
    for model in models:
        model_recs = [r for r in records if r["model"] == model]
        if not model_recs:
            print(f"  ⚠ No records for model={model}, skipping")
            continue
        by_seed = defaultdict(lambda: {"ade": [], "ate": [], "cte": []})
        for r in model_recs:
            s = r.get("seed", "unknown")
            for m in ("ade", "ate", "cte"):
                if m in r and r[m] is not None:
                    by_seed[s][m].append(r[m])

        seed_means = {"ade": [], "ate": [], "cte": []}
        n_seeds = 0
        for seed, vals in sorted(by_seed.items()):
            n_seeds += 1
            for m in ("ade", "ate", "cte"):
                if vals[m]:
                    seed_means[m].append(float(np.mean(vals[m])))

        row = {"model": model, "n_seeds": n_seeds,
               "n_records": len(model_recs)}
        for m in ("ade", "ate", "cte"):
            vals = seed_means[m]
            row[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{m}_std"]  = float(np.std(vals))  if vals else float("nan")
            row[f"{m}_per_seed"] = vals
        rows.append(row)
    return rows


def print_main_table(rows: List[Dict]):
    print(f"\n  {'='*90}")
    print(f"  TABLE 1 — MAIN RESULTS (mean ± std across seeds)")
    print(f"  {'='*90}")
    print(f"  {'Model':<12} {'#seeds':>7} {'ADE (km)':>16} {'ATE (km)':>16} {'CTE (km)':>16}")
    print(f"  {'-'*90}")
    for r in rows:
        print(f"  {r['model']:<12} {r['n_seeds']:>7} "
              f"{r['ade_mean']:>9.2f}±{r['ade_std']:<5.2f} "
              f"{r['ate_mean']:>9.2f}±{r['ate_std']:<5.2f} "
              f"{r['cte_mean']:>9.2f}±{r['cte_std']:<5.2f}")
    print(f"  {'='*90}\n")


def print_main_table_latex(rows: List[Dict]):
    print(r"  \begin{table}")
    print(r"  \caption{Main results: ADE/ATE/CTE (km), mean $\pm$ std across seeds.}")
    print(r"  \begin{tabular}{lccc}")
    print(r"  \hline")
    print(r"  Model & ADE (km) & ATE (km) & CTE (km) \\")
    print(r"  \hline")
    for r in rows:
        print(f"  {r['model']} & "
              f"{r['ade_mean']:.2f} $\\pm$ {r['ade_std']:.2f} & "
              f"{r['ate_mean']:.2f} $\\pm$ {r['ate_std']:.2f} & "
              f"{r['cte_mean']:.2f} $\\pm$ {r['cte_std']:.2f} \\\\")
    print(r"  \hline")
    print(r"  \end{tabular}")
    print(r"  \end{table}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Table 2: pooled significance table (FM vs each baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_pooled_paired_arrays(records: List[Dict], model_a: str, model_b: str,
                                metric: str):
    """
    Pairs records at (seed, storm, window, lead_time) granularity so
    that the SAME seed's forecast on the SAME storm/window/lead_time is
    matched between model_a and model_b — i.e. each seed contributes its
    own set of paired observations, and all seeds' pairs are pooled
    together into one paired test. This is what "pooled 3-seed" means
    concretely: n_pairs = sum over seeds of matched (storm,window,
    lead_time) pairs for that seed, not n_pairs from a single seed and
    not an average collapsed across seeds first.

    Falls back to (seed, storm, window) if lead_time is absent, and
    further to (storm, window) if seed is absent (older single-seed
    records) — but pooling with genuinely multi-seed data requires the
    seed field, so absence of "seed" on records means you are NOT
    actually pooling seeds, just replicating generate_comparison_table's
    single-seed behavior. A warning is printed in that case.
    """
    has_seed = any("seed" in r for r in records)
    has_lead_time = any("lead_time" in r for r in records)
    if not has_seed:
        print("  ⚠ Records have no 'seed' field — pooled test is NOT "
              "actually pooling multiple seeds. Re-check evaluate_multi_model.py "
              "output.")
    if has_seed and has_lead_time:
        key_fn = lambda r: (r.get("seed"), r["storm"], r["window"], r.get("lead_time"))
    elif has_seed:
        key_fn = lambda r: (r.get("seed"), r["storm"], r["window"])
    elif has_lead_time:
        key_fn = lambda r: (r["storm"], r["window"], r.get("lead_time"))
    else:
        key_fn = lambda r: (r["storm"], r["window"])

    by_key_a = {key_fn(r): r[metric] for r in records if r["model"] == model_a}
    by_key_b = {key_fn(r): r[metric] for r in records if r["model"] == model_b}
    common = sorted(set(by_key_a) & set(by_key_b), key=lambda k: str(k))
    if len(common) < 10:
        return None, None, len(common)
    x = np.array([by_key_a[k] for k in common])
    y = np.array([by_key_b[k] for k in common])
    return x, y, len(common)


def build_significance_table(records: List[Dict], baseline_model: str,
                              compare_against: List[str], metric: str) -> List[Dict]:
    """
    FM "achieves" a comparison per the project's agreed threshold:
    ALL FOUR of:
      mean_diff < 0  (FM lower error)
      |Cohen's d| >= 0.2  (at least small effect)
      Wilcoxon p (Bonferroni) < 0.05
      t-test p (Bonferroni) < 0.05
    """
    rows = []
    p_wilcox, p_ttest = [], []
    row_data = []
    for other in compare_against:
        x, y, n = build_pooled_paired_arrays(records, baseline_model, other, metric)
        if x is None:
            print(f"  ⚠ {baseline_model} vs {other} ({metric}): only {n} matched "
                  f"pairs (need >=10) — skipping.")
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
        achieved = (mean_diff < 0 and abs(d) >= 0.2
                    and bonf_w[i] < 0.05 and bonf_t[i] < 0.05)
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
            "fm_achieves":     achieved,
        })
    return rows


def print_significance_table(rows: List[Dict], metric: str, baseline_model: str):
    print(f"\n  {'='*128}")
    print(f"  TABLE 2 — POOLED SIGNIFICANCE (3-seed pooled paired test) for "
          f"{metric.upper()} ({baseline_model} vs baselines)")
    print(f"  {'='*128}")
    print(f"  {'Comparison':<24} {'n (pairs)':>10} {'Mean diff':>12} "
          f"{'Cohen d':>9} {'Wilcoxon p':>12} {'Wilcoxon p(Bonf.)':>18} "
          f"{'t-test p':>12} {'t-test p(Bonf.)':>16}  {'FM achieves?':>13}")
    print(f"  {'-'*128}")
    for r in rows:
        mark = "✓ YES" if r["fm_achieves"] else "✗ no"
        print(f"  {r['comparison']:<24} {r['n_pairs']:>10} "
              f"{r['mean_diff_km']:>12.4f} {r['cohens_d']:>9.4f} "
              f"{r['wilcoxon_p']:>12.2E} {r['wilcoxon_p_bonf']:>18.2E} "
              f"{r['ttest_p']:>12.2E} {r['ttest_p_bonf']:>16.2E}  {mark:>13}")
    print(f"  {'='*128}")
    print(f"  'FM achieves' requires ALL FOUR: mean_diff<0, |d|>=0.2, "
          f"Wilcoxon p(Bonf.)<0.05, t-test p(Bonf.)<0.05.\n")


def print_significance_table_latex(rows: List[Dict], metric: str, baseline_model: str):
    print(r"  \begin{table}")
    print(f"  \\caption{{Pooled 3-seed significance tests for {metric.upper()} "
          f"({baseline_model} vs RNN/GRU/LSTM/ST-Trans).}}")
    print(r"  \begin{tabular}{lrrrrrrr}")
    print(r"  \hline")
    print(r"  Comparison & n (pairs) & Mean diff (km) & Cohen's d & "
          r"Wilcoxon $p$ & Wilcoxon $p$ (Bonf.) & $t$-test $p$ & $t$-test $p$ (Bonf.) \\")
    print(r"  \hline")
    for r in rows:
        print(f"  {r['comparison']} & {r['n_pairs']} & "
              f"{r['mean_diff_km']:.4f} & {r['cohens_d']:.4f} & "
              f"{r['wilcoxon_p']:.2E} & {r['wilcoxon_p_bonf']:.2E} & "
              f"{r['ttest_p']:.2E} & {r['ttest_p_bonf']:.2E} \\\\")
    print(r"  \hline")
    print(r"  \end{tabular}")
    print(r"  \end{table}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Table 3: per-horizon table (FM vs strongest baseline)
# ─────────────────────────────────────────────────────────────────────────────

def pick_strongest_baseline(main_rows: List[Dict], exclude: str = "FM") -> str:
    """Lowest pooled mean ADE among non-FM models."""
    candidates = [r for r in main_rows if r["model"] != exclude]
    if not candidates:
        return None
    best = min(candidates, key=lambda r: r["ade_mean"])
    return best["model"]


def build_per_horizon_table(records: List[Dict], model_a: str, model_b: str,
                             zero_indexed: bool = False) -> List[Dict]:
    """
    Mean ADE (pooled across all seeds/storms/windows) at each named
    horizon, for model_a and model_b, plus the diff.
    """
    offset = -1 if zero_indexed else 0
    rows = []
    for h, lt in HORIZON_LEAD_TIMES.items():
        lt_key = lt + offset
        a_vals = [r["ade"] for r in records
                  if r["model"] == model_a and r.get("lead_time") == lt_key]
        b_vals = [r["ade"] for r in records
                  if r["model"] == model_b and r.get("lead_time") == lt_key]
        rows.append({
            "horizon": h,
            f"{model_a}_ade": float(np.mean(a_vals)) if a_vals else float("nan"),
            f"{model_a}_n":   len(a_vals),
            f"{model_b}_ade": float(np.mean(b_vals)) if b_vals else float("nan"),
            f"{model_b}_n":   len(b_vals),
        })
    return rows


def print_per_horizon_table(rows: List[Dict], model_a: str, model_b: str):
    print(f"\n  {'='*70}")
    print(f"  TABLE 3 — PER-HORIZON ADE: {model_a} vs {model_b} (strongest baseline)")
    print(f"  {'='*70}")
    print(f"  {'Horizon':<10} {model_a+' ADE':>14} {'n':>6} {model_b+' ADE':>14} {'n':>6}")
    print(f"  {'-'*70}")
    for r in rows:
        print(f"  {r['horizon']:<10} {r.get(f'{model_a}_ade', float('nan')):>14.2f} "
              f"{r.get(f'{model_a}_n', 0):>6} "
              f"{r.get(f'{model_b}_ade', float('nan')):>14.2f} "
              f"{r.get(f'{model_b}_n', 0):>6}")
    print(f"  {'='*70}")
    if rows and any(r.get(f"{model_a}_n", 0) == 0 for r in rows):
        print(f"  ⚠ Some horizons have n=0 for {model_a} — check that "
              f"'lead_time' field convention matches --lead_time_zero_indexed.\n")
    else:
        print()




# ─────────────────────────────────────────────────────────────────────────────
#  PLOTS (từ generate_comparison_plots.py) — gộp thẳng vào đây, không import chéo
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def _present_models(records, order=MODEL_PLOT_ORDER):
    present = set(r["model"] for r in records)
    return [m for m in order if m in present] + \
           [m for m in sorted(present) if m not in order]


# ─────────────────────────────────────────────────────────────────────────────
#  Fig.3/8-style: error vs lead time, one line per model, 3 metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_vs_leadtime(records: List[Dict], output_dir: str,
                            metrics=("ade", "ate", "cte")):
    """
    One figure per metric: mean error at each 6h lead-time step,
    one colored line per model. Matches the reference Fig.3/8 layout
    (mirrored here as separate single-panel figures rather than one
    combined 2x3 grid, so each can also be dropped individually into
    the paper).
    """
    models = _present_models(records)
    lead_times = sorted(set(r["lead_time"] for r in records))
    saved = []

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        for model in models:
            means = []
            for lt in lead_times:
                vals = [r[metric] for r in records
                        if r["model"] == model and r["lead_time"] == lt]
                means.append(np.mean(vals) if vals else np.nan)
            hours = [lt * 6 for lt in lead_times]
            ax.plot(hours, means, "o-", color=MODEL_COLORS.get(model, "#333"),
                    label=model, linewidth=1.8, markersize=4)

        ax.set_xlabel("Forecast Lead Time (hours)", fontsize=10)
        ax.set_ylabel(f"{metric.upper()} Error (km)", fontsize=10)
        ax.set_title(f"{metric.upper()} vs Forecast Lead Time", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        out = os.path.join(output_dir, f"error_vs_leadtime_{metric}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        saved.append(out)
        print(f"  Saved → {out}")
    return saved


def plot_error_vs_leadtime_grid(records: List[Dict], output_dir: str):
    """Combined 1x3 grid (ADE/ATE/CTE side by side) — single figure for the paper."""
    models = _present_models(records)
    lead_times = sorted(set(r["lead_time"] for r in records))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric in zip(axes, ["ade", "ate", "cte"]):
        for model in models:
            means = []
            for lt in lead_times:
                vals = [r[metric] for r in records
                        if r["model"] == model and r["lead_time"] == lt]
                means.append(np.mean(vals) if vals else np.nan)
            hours = [lt * 6 for lt in lead_times]
            ax.plot(hours, means, "o-", color=MODEL_COLORS.get(model, "#333"),
                    label=model, linewidth=1.8, markersize=4)
        ax.set_xlabel("Forecast Lead Time (h)", fontsize=10)
        ax.set_ylabel(f"{metric.upper()} (km)", fontsize=10)
        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
    axes[0].legend(fontsize=9, framealpha=0.9)
    plt.suptitle("Track Forecast Errors Across Lead Time", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "error_vs_leadtime_grid.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Fig.4/5-style: boxplot of error distribution per model
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_boxplots(records: List[Dict], output_dir: str,
                         metrics=("ade", "ate", "cte")):
    """
    One figure, 1x3 subplots (ADE/CTE/ATE), boxplot of ALL per-record
    errors (pooled over storm/window/lead_time/seed) per model — matches
    Fig.4/5's "Distribution of forecast errors" layout.
    """
    models = _present_models(records)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    metric_titles = {"ade": "Direct Position Error", "ate": "Along-Track Error",
                     "cte": "Cross-Track Error"}

    for ax, metric in zip(axes, metrics):
        data = [[r[metric] for r in records if r["model"] == m] for m in models]
        colors = [MODEL_COLORS.get(m, "#888") for m in models]
        bp = ax.boxplot(data, tick_labels=models, patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=2, alpha=0.4))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_title(metric_titles.get(metric, metric.upper()), fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{metric.upper()} (km)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Distribution of Forecast Errors", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "error_boxplots.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Per-horizon boxplots (24h/48h/72h) — extra detail beyond the reference figs
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplot_by_horizon(records: List[Dict], output_dir: str,
                             horizons=("24h", "48h", "72h"), metric="ade"):
    """
    [EXTRA, not in reference figs] Boxplot of ADE per model, SPLIT by
    horizon (24h/48h/72h) rather than pooled across all lead times —
    shows whether a model's relative advantage/variance changes with
    forecast range, which the pooled Fig.4/5-style boxplot cannot show.
    """
    models = _present_models(records)
    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 5), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    for ax, hz in zip(axes, horizons):
        lt = HORIZON_LEAD_TIMES_FULL.get(hz)
        data = [[r[metric] for r in records if r["model"] == m and r["lead_time"] == lt]
                for m in models]
        colors = [MODEL_COLORS.get(m, "#888") for m in models]
        bp = ax.boxplot(data, tick_labels=models, patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=2, alpha=0.4))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_title(f"{metric.upper()} @ {hz}", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[0].set_ylabel(f"{metric.upper()} (km)", fontsize=10)

    plt.suptitle(f"{metric.upper()} Distribution by Horizon", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, f"boxplot_by_horizon_{metric}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  FM-only: seed variance visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_seed_variance(records: List[Dict], output_dir: str, metric="ade"):
    """
    [EXTRA, FM-specific] Bar chart: mean±std ADE across seeds, per model.
    Directly visualizes what generate_paper_table.py's Table 1 reports
    as numbers — useful as a figure showing FM's seed-to-seed stability
    relative to the baselines (part of arguing FM's architecture is not
    just accurate but also STABLE across random initialization).
    """
    models = _present_models(records)
    means, stds = [], []
    for model in models:
        by_seed = defaultdict(list)
        for r in records:
            if r["model"] == model:
                by_seed[r.get("seed", "unknown")].append(r[metric])
        seed_means = [np.mean(v) for v in by_seed.values()]
        means.append(np.mean(seed_means) if seed_means else np.nan)
        stds.append(np.std(seed_means) if seed_means else np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [MODEL_COLORS.get(m, "#888") for m in models]
    ax.bar(models, means, yerr=stds, capsize=5, color=colors, alpha=0.75,
          edgecolor="black", linewidth=0.6)
    ax.set_ylabel(f"{metric.upper()} (km)", fontsize=10)
    ax.set_title(f"{metric.upper()} Mean ± Std Across Seeds", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    out = os.path.join(output_dir, f"seed_variance_{metric}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  FM-only: ODE-steps (N) ablation — line + heatmap-style summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_ode_n_sweep(ode_sweep: Dict, output_dir: str):
    """
    [EXTRA, FM-specific] From ablation_runner.py's ode_steps_sweep()
    output: 4-panel figure (ADE, ATE, CTE, spread) vs N, all on the
    same x-axis, so the ADE/ATE/CTE-vs-spread TRADE-OFF (documented in
    the project's own notes: N=1->10 improves spread/CRPS but slightly
    worsens ADE/ATE/CTE, non-monotonic at N=3-4) is visible in one
    figure rather than requiring cross-referencing 2 separate tables.
    """
    ns = sorted(int(k) for k in ode_sweep.keys())
    ade = [ode_sweep[str(n)].get("ADE_mean", ode_sweep.get(n, {}).get("ADE_mean")) for n in ns]
    ate = [ode_sweep[str(n)].get("ATE_mean", ode_sweep.get(n, {}).get("ATE_mean")) for n in ns]
    cte = [ode_sweep[str(n)].get("CTE_mean", ode_sweep.get(n, {}).get("CTE_mean")) for n in ns]
    spread = [ode_sweep[str(n)].get("spread_mean", ode_sweep.get(n, {}).get("spread_mean")) for n in ns]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    titles = ["ADE (km)", "ATE (km)", "CTE (km)", "Ensemble Spread (km)"]
    series = [ade, ate, cte, spread]
    colors = ["#D62728", "#1F5FBF", "#2CA02C", "#9467BD"]

    for ax, title, vals, c in zip(axes, titles, series, colors):
        ax.plot(ns, vals, "o-", color=c, linewidth=2, markersize=6)
        ax.set_xlabel("ODE integration steps N", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xticks(ns)

    plt.suptitle("FM: Accuracy vs Ensemble Diversity Trade-off Across ODE Steps N",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "ode_n_sweep.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  FM-only: ablation bar chart (loss components / architecture pieces)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation_bars(ablation_dir: str, output_dir: str, metric="ADE"):
    """
    Reads all <variant>.json files written by
    ablation_runner.py --mode eval_variant and draws a horizontal bar
    chart of `metric` per variant, sorted worst->best, with the "full"
    model highlighted — the standard "what breaks if you remove X" plot
    for an ablation table, visualized rather than just tabulated.
    """
    jsons = [f for f in os.listdir(ablation_dir) if f.endswith(".json")]
    if not jsons:
        print(f"  ⚠ No ablation JSONs found in {ablation_dir}")
        return None

    rows = []
    for jf in jsons:
        try:
            d = load_json(os.path.join(ablation_dir, jf))
        except Exception as e:
            print(f"  ⚠ Skipping {jf}: {e}")
            continue
        if metric not in d:
            continue
        name = d.get("variant", jf.replace(".json", ""))
        rows.append((name, d[metric]))

    if not rows:
        print(f"  ⚠ No variant JSONs contain metric={metric}")
        return None

    rows.sort(key=lambda r: r[1], reverse=True)  # worst (highest error) first
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ["#D62728" if n == "full" else "#7F9BB5" for n in names]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(names))))
    ax.barh(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"{metric} (km)", fontsize=10)
    ax.set_title(f"Ablation Study — {metric} per Variant\n"
                "(red = full model, blue = ablated variant)",
                fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    out = os.path.join(output_dir, f"ablation_bars_{metric}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  FM-only: sigma sensitivity heatmap-style plot (from evaluate_full.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sigma_sensitivity(eval_full_json: Dict, output_dir: str):
    """
    From evaluate_full.py's --sigma_sensitivity block: line plot of
    ADE/CTE vs sigma_inference. Not a 2D heatmap (unlike the reference
    Fig.7, which sweeps 2 hyperparameters lambda_speed x lambda_accel) —
    the project's own sigma_sensitivity() only sweeps 1 hyperparameter
    (sigma_inference) at fixed training config, so a 1D line plot is the
    faithful representation; forcing a 2D heatmap here would require a
    2nd swept axis that doesn't exist in the current sigma_sensitivity()
    implementation.
    """
    block = eval_full_json.get("sigma_sensitivity")
    if not block:
        print("  ⚠ No 'sigma_sensitivity' block in eval_full_json — skip")
        return None

    sigmas = sorted(float(k) for k in block.keys())
    ade = [block[str(s)]["ADE"] if str(s) in block else block[s]["ADE"] for s in sigmas]
    cte = [block[str(s)]["CTE"] if str(s) in block else block[s]["CTE"] for s in sigmas]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(sigmas, ade, "o-", color="#D62728", label="ADE", linewidth=2)
    ax1.set_xlabel("sigma_inference", fontsize=10)
    ax1.set_ylabel("ADE (km)", color="#D62728", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#D62728")

    ax2 = ax1.twinx()
    ax2.plot(sigmas, cte, "s--", color="#1F5FBF", label="CTE", linewidth=2)
    ax2.set_ylabel("CTE (km)", color="#1F5FBF", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#1F5FBF")

    ax1.set_title("Sensitivity to sigma_inference", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    out = os.path.join(output_dir, "sigma_sensitivity.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


def plot_ensemble_size_ablation(eval_full_json: Dict, output_dir: str):
    """From evaluate_full.py's --ensemble_ablation block: ADE/ATE/CTE vs K."""
    block = eval_full_json.get("ensemble_ablation")
    if not block:
        print("  ⚠ No 'ensemble_ablation' block in eval_full_json — skip")
        return None

    ks = sorted(int(k) for k in block.keys())
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric, color in [("ADE", "#D62728"), ("ATE", "#1F5FBF"), ("CTE", "#2CA02C")]:
        vals = [block[str(k)][metric] if str(k) in block else block[k][metric] for k in ks]
        ax.plot(ks, vals, "o-", color=color, label=metric, linewidth=2)
    ax.set_xlabel("Ensemble size K", fontsize=10)
    ax.set_ylabel("Error (km)", fontsize=10)
    ax.set_title("Sensitivity to Ensemble Size K", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    out = os.path.join(output_dir, "ensemble_size_ablation.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  FM-only: per-storm CTE ranking (uses evaluate_full.py's per_storm_*.json)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_storm_cte(per_storm_json_path: str, output_dir: str, top_n=20):
    """
    [EXTRA] From evaluate_full.py's --per_storm output
    (per_storm_<split>_ep<N>.json): horizontal bar of CTE per storm,
    worst N first — visual version of print_per_storm_breakdown's table,
    useful for showing reviewers which specific storms drive aggregate
    CTE rather than just an averaged number.
    """
    d = load_json(per_storm_json_path)
    rows = sorted(d.items(), key=lambda kv: kv[1]["cte"], reverse=True)[:top_n]
    names = [r[0] for r in rows]
    ctes = [r[1]["cte"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(names))))
    ax.barh(names, ctes, color="#D62728", alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("CTE (km)", fontsize=10)
    ax.set_title(f"Worst {top_n} Storms by CTE", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    out = os.path.join(output_dir, "per_storm_cte_worst.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True,
                   help="Path to multi_model_<split>.json from evaluate_multi_model.py")
    p.add_argument("--baseline_model", default="FM")
    p.add_argument("--compare_against", nargs="+",
                   default=["ST-Trans", "RNN", "GRU", "LSTM"])
    p.add_argument("--metrics", nargs="+", default=["ade", "ate", "cte"],
                   choices=["ade", "ate", "cte"])
    p.add_argument("--output_dir", default="eval_multi",
                   help="Bảng (paper_tables.json) và hình (*.png) đều lưu vào đây")
    p.add_argument("--latex", action="store_true")
    p.add_argument("--lead_time_zero_indexed", action="store_true",
                   help="Pass if evaluate_multi_model.py's 'lead_time' field "
                        "is 0-indexed (0=6h) rather than 1-indexed (1=6h, "
                        "matching evaluate_full.py's HORIZONS convention).")
    p.add_argument("--tables_only", action="store_true",
                   help="Chỉ chạy 3 bảng, bỏ qua toàn bộ phần plot")
    p.add_argument("--plots_only", action="store_true",
                   help="Chỉ chạy plot, bỏ qua 3 bảng")
    # Optional plot inputs (không truyền thì tự bỏ qua đúng plot cần nó)
    p.add_argument("--ode_sweep", default=None,
                   help="ode_steps_sweep.json từ ablation_runner.py --mode ode_steps")
    p.add_argument("--ablation_dir", default=None,
                   help="Thư mục chứa <variant>.json từ ablation_runner.py --mode eval_variant")
    p.add_argument("--eval_full_json", default=None,
                   help="eval_<split>_ep<N>.json từ evaluate_full.py "
                        "(cho sigma_sensitivity / ensemble_ablation)")
    p.add_argument("--per_storm_json", default=None,
                   help="per_storm_<split>_ep<N>.json từ evaluate_full.py --per_storm")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = load_records(args.records)
    present_models = sorted(set(r["model"] for r in records))
    print(f"  Models present in records: {present_models}")
    if args.baseline_model not in present_models:
        print(f"  ❌ --baseline_model '{args.baseline_model}' not found. "
              f"Available: {present_models}")
        return

    # ═══════════════════════════════════════════════════════════════════
    #  PHẦN 1: 3 BẢNG (main / significance / per-horizon)
    # ═══════════════════════════════════════════════════════════════════
    if not args.plots_only:
        models_for_main = [m for m in ALL_MODELS if m in present_models]
        if not models_for_main:
            models_for_main = present_models

        main_rows = build_main_table(records, models_for_main)
        print_main_table(main_rows)
        if args.latex:
            print_main_table_latex(main_rows)

        compare_against = [m for m in args.compare_against if m in present_models]
        missing = set(args.compare_against) - set(compare_against)
        if missing:
            print(f"  ⚠ Requested comparison models not found, skipping: {sorted(missing)}")

        sig_results = {}
        for metric in args.metrics:
            rows = build_significance_table(records, args.baseline_model,
                                             compare_against, metric)
            if not rows:
                print(f"  ⚠ No valid comparisons for metric={metric}")
                continue
            print_significance_table(rows, metric, args.baseline_model)
            if args.latex:
                print_significance_table_latex(rows, metric, args.baseline_model)
            sig_results[metric] = rows

        strongest = pick_strongest_baseline(main_rows, exclude=args.baseline_model)
        horizon_rows = []
        if strongest:
            print(f"  Strongest baseline by pooled mean ADE: {strongest}")
            horizon_rows = build_per_horizon_table(
                records, args.baseline_model, strongest,
                zero_indexed=args.lead_time_zero_indexed)
            print_per_horizon_table(horizon_rows, args.baseline_model, strongest)
        else:
            print("  ⚠ Could not determine strongest baseline — skipping Table 3")

        out = {
            "main_table":        main_rows,
            "significance_table": sig_results,
            "per_horizon_table": {
                "baseline_model": args.baseline_model,
                "strongest_other": strongest,
                "rows": horizon_rows,
            },
        }
        out_path = os.path.join(args.output_dir, "paper_tables.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved all 3 tables → {out_path}")

    # ═══════════════════════════════════════════════════════════════════
    #  PHẦN 2: PLOTS
    # ═══════════════════════════════════════════════════════════════════
    if not args.tables_only:
        print(f"\n  {'='*70}\n  Generating plots...\n  {'='*70}")
        saved = []
        saved += plot_error_vs_leadtime(records, args.output_dir)
        saved.append(plot_error_vs_leadtime_grid(records, args.output_dir))
        saved.append(plot_error_boxplots(records, args.output_dir))
        saved.append(plot_boxplot_by_horizon(records, args.output_dir))
        saved.append(plot_seed_variance(records, args.output_dir, metric="ade"))
        saved.append(plot_seed_variance(records, args.output_dir, metric="cte"))

        if args.ode_sweep:
            ode_sweep = load_json(args.ode_sweep)
            saved.append(plot_ode_n_sweep(ode_sweep, args.output_dir))

        if args.ablation_dir:
            saved.append(plot_ablation_bars(args.ablation_dir, args.output_dir, metric="ADE"))
            saved.append(plot_ablation_bars(args.ablation_dir, args.output_dir, metric="CTE"))

        if args.eval_full_json:
            ej = load_json(args.eval_full_json)
            saved.append(plot_sigma_sensitivity(ej, args.output_dir))
            saved.append(plot_ensemble_size_ablation(ej, args.output_dir))

        if args.per_storm_json:
            saved.append(plot_per_storm_cte(args.per_storm_json, args.output_dir))

        saved = [s for s in saved if s]
        print(f"\n  Done — {len(saved)} figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
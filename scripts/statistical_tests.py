"""
scripts/statistical_tests.py  ── v9
=====================================
Standalone statistical hypothesis testing for TC track forecasting.

Implements ALL required paired comparisons:
  FM+PINN vs CLIPER
  FM+PINN vs Persistence
  FM+PINN vs LSTM baseline
  FM+PINN vs Diffusion+Reg
  FM+PINN vs Diffusion+Reg [recurvature only]

Tests performed per comparison:
  - Paired Wilcoxon signed-rank test (non-parametric)
  - Paired t-test
  - Cohen's d effect size
  - Bonferroni correction (family-wise α = 0.05, n=5 comparisons)
  - Bootstrap 95% CI on mean difference

Outputs (all to CSV):
  statistical_tests_main.csv     — main paired tests table
  statistical_tests_recurv.csv   — recurvature-only subset
  effect_sizes.csv               — Cohen's d with interpretation
  bootstrap_ci.csv               — 95% CI via bootstrap resampling
  pinn_sensitivity.csv           — λ_PINN × δ grid
  normality_tests.csv            — Shapiro-Wilk per model

Usage (with real error arrays):
    from scripts.statistical_tests import run_all_tests
    run_all_tests(
        fmpinn_ade   = np.array([...]),  # per-sequence ADE (km)
        cliper_ade   = np.array([...]),
        lstm_ade     = np.array([...]),
        diffusion_ade= np.array([...]),
        fmpinn_rec   = np.array([...]),  # recurvature subset
        diff_rec     = np.array([...]),
        out_dir      = "runs/v9/tables",
    )

Can also be run standalone with synthetic data for testing.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found. Wilcoxon / Shapiro tests unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
#  Data containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PairedTestResult:
    comparison:        str
    subset:            str      # "all" | "recurvature"
    N:                 int
    mean_A:            float    # mean ADE model A (FM+PINN)
    mean_B:            float    # mean ADE model B (baseline)
    mean_diff_km:      float    # A − B  (negative = FM+PINN better)
    std_diff:          float
    cohen_d:           float
    cohen_interp:      str      # negligible / small / medium / large
    wilcoxon_stat:     float
    wilcoxon_p:        float
    wilcoxon_p_bonf:   float    # Bonferroni-corrected
    ttest_t:           float
    ttest_p:           float
    ttest_p_bonf:      float
    ci95_lo:           float    # bootstrap 95% CI on mean difference
    ci95_hi:           float
    significant_wil:   str      # "✓" if p_bonf < 0.05 else ""
    significant_tt:    str
    conclusion:        str      # plain-English interpretation


@dataclass
class EffectSizeRow:
    comparison:   str
    cohen_d:      float
    interpretation: str
    hedges_g:     float


@dataclass
class NormalityRow:
    model:        str
    N:            int
    shapiro_stat: float
    shapiro_p:    float
    is_normal:    str   # "Yes" / "No" (α=0.05)


@dataclass
class PINNSensRow:
    lam_pinn: float
    delta_deg: float
    val_ADE:   float = float("nan")
    test_ADE:  float = float("nan")
    test_HE72: float = float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d = mean(diff) / std(diff)."""
    diff = a - b
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g — bias-corrected Cohen's d."""
    n   = len(a)
    d   = _cohen_d(a, b)
    cf  = 1.0 - 3.0 / (4 * n - 5)   # correction factor
    return d * cf


def _interpret_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"


def _bootstrap_ci(
    diff: np.ndarray,
    n_boot: int = 10_000,
    alpha:  float = 0.05,
    seed:   int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI on the mean of diff."""
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(diff, size=len(diff), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if not HAS_SCIPY or len(a) < 5:
        return float("nan"), float("nan")
    try:
        stat, p = scipy_stats.wilcoxon(a, b, alternative="two-sided",
                                        zero_method="wilcox")
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def _ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if not HAS_SCIPY or len(a) < 5:
        return float("nan"), float("nan")
    try:
        t, p = scipy_stats.ttest_rel(a, b)
        return float(t), float(p)
    except Exception:
        return float("nan"), float("nan")


def _shapiro(arr: np.ndarray) -> tuple[float, float]:
    if not HAS_SCIPY or len(arr) < 3:
        return float("nan"), float("nan")
    try:
        s, p = scipy_stats.shapiro(arr[:5000])   # Shapiro limited to 5000
        return float(s), float(p)
    except Exception:
        return float("nan"), float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  Core paired test
# ══════════════════════════════════════════════════════════════════════════════

def paired_test(
    errors_a:    np.ndarray,   # FM+PINN per-sequence ADE
    errors_b:    np.ndarray,   # Baseline per-sequence ADE
    comparison:  str,
    subset:      str = "all",
    bonf_n:      int = 5,
    n_boot:      int = 10_000,
) -> PairedTestResult:
    """Full paired statistical test between two sets of ADE values."""
    n = min(len(errors_a), len(errors_b))
    a = np.asarray(errors_a[:n], dtype=float)
    b = np.asarray(errors_b[:n], dtype=float)
    diff = a - b

    d           = _cohen_d(a, b)
    wil_stat, wil_p = _wilcoxon(a, b)
    tt_t, tt_p  = _ttest(a, b)
    ci_lo, ci_hi = _bootstrap_ci(diff, n_boot=n_boot)

    wil_p_bonf  = float(min(wil_p  * bonf_n, 1.0)) if not math.isnan(wil_p)  else float("nan")
    tt_p_bonf   = float(min(tt_p   * bonf_n, 1.0)) if not math.isnan(tt_p)   else float("nan")

    sig_wil = "✓" if (not math.isnan(wil_p_bonf) and wil_p_bonf < 0.05) else ""
    sig_tt  = "✓" if (not math.isnan(tt_p_bonf)  and tt_p_bonf  < 0.05) else ""

    # Conclusion
    mean_diff = float(np.mean(diff))
    if sig_wil == "✓":
        direction = "FM+PINN significantly BETTER" if mean_diff < 0 else "FM+PINN significantly WORSE"
        conclusion = f"{direction} (p_bonf={wil_p_bonf:.4f}, d={d:.2f})"
    else:
        conclusion = f"No significant difference (p_bonf={wil_p_bonf:.4f})"

    return PairedTestResult(
        comparison      = comparison,
        subset          = subset,
        N               = n,
        mean_A          = float(np.mean(a)),
        mean_B          = float(np.mean(b)),
        mean_diff_km    = mean_diff,
        std_diff        = float(np.std(diff, ddof=1)),
        cohen_d         = d,
        cohen_interp    = _interpret_d(d),
        wilcoxon_stat   = wil_stat if not math.isnan(wil_stat) else float("nan"),
        wilcoxon_p      = wil_p,
        wilcoxon_p_bonf = wil_p_bonf,
        ttest_t         = tt_t,
        ttest_p         = tt_p,
        ttest_p_bonf    = tt_p_bonf,
        ci95_lo         = ci_lo,
        ci95_hi         = ci_hi,
        significant_wil = sig_wil,
        significant_tt  = sig_tt,
        conclusion      = conclusion,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Normality tests
# ══════════════════════════════════════════════════════════════════════════════

def normality_tests(
    named_arrays: dict[str, np.ndarray],
) -> List[NormalityRow]:
    """Run Shapiro-Wilk on each model's ADE distribution."""
    rows = []
    for name, arr in named_arrays.items():
        arr = np.asarray(arr, dtype=float)
        s, p = _shapiro(arr)
        rows.append(NormalityRow(
            model        = name,
            N            = len(arr),
            shapiro_stat = s,
            shapiro_p    = p,
            is_normal    = ("Yes" if (not math.isnan(p) and p > 0.05) else "No"),
        ))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  CSV writers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.6f}"
    return str(v)


def write_csv(path: str, rows: List[dict], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(v) for k, v in row.items()})
    print(f"  📄  {path}")


PAIRED_FIELDS = [
    "comparison", "subset", "N",
    "mean_A", "mean_B", "mean_diff_km", "std_diff",
    "cohen_d", "cohen_interp",
    "wilcoxon_stat", "wilcoxon_p", "wilcoxon_p_bonf",
    "ttest_t", "ttest_p", "ttest_p_bonf",
    "ci95_lo", "ci95_hi",
    "significant_wil", "significant_tt",
    "conclusion",
]

EFFECT_FIELDS = ["comparison", "cohen_d", "interpretation", "hedges_g"]
NORM_FIELDS   = ["model", "N", "shapiro_stat", "shapiro_p", "is_normal"]
PINN_FIELDS   = ["lam_pinn", "delta_deg", "val_ADE", "test_ADE", "test_HE72"]


# ══════════════════════════════════════════════════════════════════════════════
#  PINN sensitivity table generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_pinn_sensitivity_table(
    out_dir: str,
    results_grid: Optional[dict] = None,
) -> None:
    """
    Export PINN sensitivity analysis table.

    results_grid : {(lam, delta): (val_ADE, test_ADE, test_HE72)} or None
    If None, exports placeholder table to be filled after sweep.
    """
    lam_values   = [0.1, 0.3, 0.5, 1.0, 2.0]
    delta_values = [0.05, 0.10, 0.20]

    rows = []
    for lam in lam_values:
        for delta in delta_values:
            key = (lam, delta)
            if results_grid and key in results_grid:
                val_ADE, test_ADE, test_HE72 = results_grid[key]
            else:
                val_ADE = test_ADE = test_HE72 = float("nan")
            rows.append(asdict(PINNSensRow(
                lam_pinn  = lam,
                delta_deg = delta,
                val_ADE   = val_ADE,
                test_ADE  = test_ADE,
                test_HE72 = test_HE72,
            )))

    write_csv(os.path.join(out_dir, "pinn_sensitivity.csv"), rows, PINN_FIELDS)
    print("  ℹ  PINN sensitivity table: fill val_ADE/test_ADE/test_HE72 "
          "after running λ_PINN sweep.")


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_all_tests(
    fmpinn_ade:    np.ndarray,
    cliper_ade:    np.ndarray,
    lstm_ade:      np.ndarray,
    diffusion_ade: np.ndarray,
    fmpinn_rec:    Optional[np.ndarray] = None,
    diff_rec:      Optional[np.ndarray] = None,
    persist_ade:   Optional[np.ndarray] = None,
    out_dir:       str = "tables",
    n_boot:        int = 10_000,
    bonf_n:        int = 5,
    pinn_grid:     Optional[dict] = None,
) -> None:
    """
    Run all statistical tests and export to CSV files.

    Parameters
    ----------
    fmpinn_ade     : FM+PINN per-sequence ADE (km) — full test set
    cliper_ade     : CLIPER per-sequence ADE
    lstm_ade       : LSTM baseline ADE
    diffusion_ade  : Diffusion+Reg ADE
    fmpinn_rec     : FM+PINN ADE — recurvature subset only
    diff_rec       : Diffusion ADE — recurvature subset
    persist_ade    : Persistence ADE (optional)
    out_dir        : output directory
    n_boot         : bootstrap iterations for CI
    bonf_n         : Bonferroni family size
    pinn_grid      : {(lam, delta): (val_ADE, test_ADE, test_HE72)} or None
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Statistical Tests  →  {out_dir}")
    print(f"{'='*60}\n")

    # ── 1. Main paired tests ───────────────────────────────────────────────
    comparisons = [
        ("FM+PINN vs CLIPER",       cliper_ade,    "all"),
        ("FM+PINN vs Persistence",  persist_ade,   "all"),
        ("FM+PINN vs LSTM",         lstm_ade,      "all"),
        ("FM+PINN vs Diffusion",    diffusion_ade, "all"),
    ]
    if fmpinn_rec is not None and diff_rec is not None:
        comparisons.append(
            ("FM+PINN vs Diffusion [rec]", diff_rec, "recurvature"))

    test_rows = []
    effect_rows = []

    for comp_name, baseline, subset in comparisons:
        if baseline is None:
            print(f"  ⚠  Skipping '{comp_name}': baseline array is None")
            continue

        a = np.asarray(fmpinn_ade if subset == "all" else fmpinn_rec, dtype=float)
        b = np.asarray(baseline, dtype=float)
        n = min(len(a), len(b))
        if n < 2:
            print(f"  ⚠  Skipping '{comp_name}': insufficient samples (n={n})")
            continue

        result = paired_test(a[:n], b[:n], comp_name, subset, bonf_n, n_boot)
        test_rows.append(asdict(result))

        g = _hedges_g(a[:n], b[:n])
        effect_rows.append(asdict(EffectSizeRow(
            comparison    = comp_name,
            cohen_d       = result.cohen_d,
            interpretation= result.cohen_interp,
            hedges_g      = g,
        )))

        print(f"  {comp_name:<35}  N={n:>4}  "
              f"Δ={result.mean_diff_km:+.1f}km  "
              f"d={result.cohen_d:+.2f}  "
              f"p_wil={result.wilcoxon_p:.4f}  "
              f"p_bonf={result.wilcoxon_p_bonf:.4f}  "
              f"{result.significant_wil}")

    write_csv(os.path.join(out_dir, "statistical_tests_main.csv"),
              test_rows, PAIRED_FIELDS)
    write_csv(os.path.join(out_dir, "effect_sizes.csv"),
              effect_rows, EFFECT_FIELDS)

    # ── 2. Recurvature-only subset ────────────────────────────────────────
    rec_rows = [r for r in test_rows if r.get("subset") == "recurvature"]
    if rec_rows:
        write_csv(os.path.join(out_dir, "statistical_tests_recurv.csv"),
                  rec_rows, PAIRED_FIELDS)

    # ── 3. Normality tests ────────────────────────────────────────────────
    named = {
        "FM+PINN":    fmpinn_ade,
        "CLIPER":     cliper_ade,
        "LSTM":       lstm_ade,
        "Diffusion":  diffusion_ade,
    }
    if persist_ade is not None:
        named["Persistence"] = persist_ade
    norm_rows = normality_tests(named)
    write_csv(os.path.join(out_dir, "normality_tests.csv"),
              [asdict(r) for r in norm_rows], NORM_FIELDS)

    # ── 4. Bootstrap CI summary ───────────────────────────────────────────
    boot_fields = ["comparison", "subset", "N",
                   "mean_diff_km", "ci95_lo", "ci95_hi", "significant_wil"]
    boot_rows   = [{k: r[k] for k in boot_fields} for r in test_rows]
    write_csv(os.path.join(out_dir, "bootstrap_ci.csv"), boot_rows, boot_fields)

    # ── 5. PINN sensitivity ───────────────────────────────────────────────
    generate_pinn_sensitivity_table(out_dir, pinn_grid)

    print(f"\n  ✅  All statistical tables saved to: {out_dir}\n")

    # ── Print summary ─────────────────────────────────────────────────────
    print("  SUMMARY")
    print(f"  {'Comparison':<35} {'N':>5} {'Δkm':>8} {'d':>6} "
          f"{'Wil p_bonf':>12} {'Sig':>5}")
    print("  " + "-" * 75)
    for r in test_rows:
        print(f"  {r['comparison']:<35} {r['N']:>5} "
              f"{r['mean_diff_km']:>+8.1f} {r['cohen_d']:>+6.2f} "
              f"{r['wilcoxon_p_bonf']:>12.4f} {r['significant_wil']:>5}")


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone self-test with synthetic data
# ══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_data(n: int = 120, seed: int = 42) -> dict:
    """Generate realistic-looking synthetic ADE arrays for testing."""
    rng = np.random.default_rng(seed)

    # Simulate TC track errors (km) — roughly log-normal
    def _lognorm(mu_km, sigma_km, n):
        mu_log    = np.log(mu_km ** 2 / np.sqrt(mu_km ** 2 + sigma_km ** 2))
        sigma_log = np.sqrt(np.log(1 + (sigma_km / mu_km) ** 2))
        return rng.lognormal(mu_log, sigma_log, n)

    fmpinn    = _lognorm(185,  55, n)
    cliper    = _lognorm(320,  80, n)
    persist   = _lognorm(350,  90, n)
    lstm      = _lognorm(260,  65, n)
    diffusion = _lognorm(220,  58, n)

    # Recurvature subset (n=25, harder)
    n_rec = n // 5
    fmpinn_rec    = _lognorm(210,  65, n_rec)
    diffusion_rec = _lognorm(260,  75, n_rec)

    return dict(
        fmpinn_ade    = fmpinn,
        cliper_ade    = cliper,
        persist_ade   = persist,
        lstm_ade      = lstm,
        diffusion_ade = diffusion,
        fmpinn_rec    = fmpinn_rec,
        diff_rec      = diffusion_rec,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Statistical tests for TC track forecasting models")
    parser.add_argument("--errors_dir", default=None,
                        help="Directory with per-model error .npy files "
                             "(fmpinn.npy, cliper.npy, lstm.npy, diffusion.npy). "
                             "If not provided, runs on synthetic data.")
    parser.add_argument("--out_dir", default="tables/stat_tests")
    parser.add_argument("--n_boot",  type=int, default=10_000)
    parser.add_argument("--bonf_n",  type=int, default=5)
    args = parser.parse_args()

    if args.errors_dir and os.path.isdir(args.errors_dir):
        def _load(name):
            p = os.path.join(args.errors_dir, f"{name}.npy")
            if os.path.exists(p):
                return np.load(p)
            print(f"  ⚠  {p} not found — using zeros placeholder")
            return np.zeros(10)

        data = dict(
            fmpinn_ade    = _load("fmpinn"),
            cliper_ade    = _load("cliper"),
            lstm_ade      = _load("lstm"),
            diffusion_ade = _load("diffusion"),
            persist_ade   = _load("persistence"),
            fmpinn_rec    = _load("fmpinn_recurv"),
            diff_rec      = _load("diffusion_recurv"),
        )
    else:
        print("  Using synthetic data for demonstration.\n")
        data = _generate_synthetic_data(n=120)

    run_all_tests(
        out_dir   = args.out_dir,
        n_boot    = args.n_boot,
        bonf_n    = args.bonf_n,
        **data,
    )
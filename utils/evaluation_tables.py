"""
utils/evaluation_tables.py  ── v9
===================================
Export all evaluation tables (A–D), statistical tests, PINN sensitivity,
computational footprint, and baseline comparisons to separate CSV files.

Tables generated:
  table_A_validation.csv      — Tier 1–4 on val set
  table_B_test.csv            — Tier 1–4 on test set
  table_C_recurvature.csv     — Recurvature-only test cases
  table_D_ablation.csv        — Ablation study (loss components)
  table_stat_tests.csv        — Paired statistical tests
  table_pinn_sensitivity.csv  — λ_PINN × δ grid
  table_compute_footprint.csv — Params / size / inference speed
  table_baseline_compare.csv  — CLIPER, Persistence, LSTM, Diffusion, FM+PINN
"""
from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ══════════════════════════════════════════════════════════════════════════════
#  Data containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    model_name:   str
    split:        str       # "val" | "test"
    ADE:          float = float("nan")
    FDE:          float = float("nan")
    ADE_str:      float = float("nan")
    ADE_rec:      float = float("nan")
    delta_rec:    float = float("nan")   # ADE_rec / ADE_str
    HE_24h:       float = float("nan")
    HE_48h:       float = float("nan")
    HE_72h:       float = float("nan")
    CRPS_mean:    float = float("nan")
    CRPS_72h:     float = float("nan")
    SSR:          float = float("nan")
    TSS_72h:      float = float("nan")
    OYR:          float = float("nan")
    DTW:          float = float("nan")
    ATE_abs:      float = float("nan")
    CTE_abs:      float = float("nan")
    n_total:      int   = 0
    n_recurv:     int   = 0
    train_time_h: float = float("nan")
    params_M:     float = float("nan")


@dataclass
class AblationRow:
    config:       str
    use_fm:       bool = True
    use_dir:      bool = False
    use_disp:     bool = False
    use_smooth:   bool = False
    use_pinn:     bool = False
    use_heading:  bool = False
    val_ADE:      float = float("nan")
    val_ADE_rec:  float = float("nan")
    test_ADE:     float = float("nan")
    test_ADE_rec: float = float("nan")
    test_HE_72:   float = float("nan")


@dataclass
class StatTestRow:
    comparison:      str
    N_pairs:         int   = 0
    mean_diff_km:    float = float("nan")
    cohen_d:         float = float("nan")
    wilcoxon_p:      float = float("nan")
    wilcoxon_p_bonf: float = float("nan")
    ttest_p:         float = float("nan")
    significant:     str   = ""


@dataclass
class PINNSensRow:
    lam_pinn:  float
    delta_deg: float
    val_ADE:   float = float("nan")
    test_ADE:  float = float("nan")
    test_HE72: float = float("nan")


@dataclass
class ComputeRow:
    component:    str
    params_M:     float = float("nan")
    size_MB:      float = float("nan")
    inference_ms: float = float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  Heading Error
# ══════════════════════════════════════════════════════════════════════════════

def heading_error_deg(pred_01: np.ndarray, gt_01: np.ndarray) -> np.ndarray:
    """Per-step heading error (degrees). pred_01, gt_01: [T, 2] 0.1° units."""
    def _angles(traj):
        dlon = np.diff(traj[:, 0]) * np.cos(np.deg2rad(traj[:-1, 1] / 10.0))
        dlat = np.diff(traj[:, 1])
        return np.arctan2(dlat, dlon)

    pa = _angles(pred_01)
    ga = _angles(gt_01)
    m  = min(len(pa), len(ga))
    d  = np.abs(np.degrees(pa[:m] - ga[:m]))
    return np.where(d > 180, 360 - d, d)


def mean_he_at_step(pred_seqs, gt_seqs, step: int) -> float:
    vals = []
    for p, g in zip(pred_seqs, gt_seqs):
        he = heading_error_deg(p, g)
        if step < len(he):
            vals.append(float(he[step]))
    return float(np.mean(vals)) if vals else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  Statistical tests
# ══════════════════════════════════════════════════════════════════════════════

def paired_tests(
    errors_a:   np.ndarray,
    errors_b:   np.ndarray,
    comparison: str,
    bonf_n:     int = 5,
) -> StatTestRow:
    n = min(len(errors_a), len(errors_b))
    if n == 0:
        return StatTestRow(comparison=comparison)

    a    = np.asarray(errors_a[:n], dtype=float)
    b    = np.asarray(errors_b[:n], dtype=float)
    diff = a - b

    row = StatTestRow(
        comparison   = comparison,
        N_pairs      = n,
        mean_diff_km = float(np.mean(diff)),
        cohen_d      = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)),
    )

    if HAS_SCIPY and n >= 5:
        try:
            _, wx_p = stats.wilcoxon(a, b, alternative="two-sided")
            _, tt_p = stats.ttest_rel(a, b)
            row.wilcoxon_p      = float(wx_p)
            row.wilcoxon_p_bonf = float(min(wx_p * bonf_n, 1.0))
            row.ttest_p         = float(tt_p)
            row.significant     = "✓" if wx_p * bonf_n < 0.05 else ""
        except Exception:
            pass

    return row


# ══════════════════════════════════════════════════════════════════════════════
#  CSV writer
# ══════════════════════════════════════════════════════════════════════════════

def _write_csv(path: str, rows: List[dict], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({
                k: (f"{v:.4f}" if isinstance(v, float) and not math.isnan(v)
                    else ("" if isinstance(v, float) and math.isnan(v) else v))
                for k, v in row.items()
            })
    print(f"  📄  {path}")


# ── Table A / B ───────────────────────────────────────────────────────────────

TABLE_AB_FIELDS = [
    "model", "split", "n_total", "n_recurv",
    "ADE", "FDE",
    "HE_24h", "HE_48h", "HE_72h",
    "ADE_str", "ADE_rec", "delta_rec",
    "CRPS_mean", "CRPS_72h", "SSR",
    "TSS_72h", "OYR", "DTW",
    "ATE_abs", "CTE_abs",
    "train_time_h", "params_M",
]

def _rename(rows):
    return [{"model" if k == "model_name" else k: v for k, v in r.items()}
            for r in rows]

def save_table_AB(results: List[ModelResult], out_dir: str) -> None:
    val_rows  = _rename([asdict(r) for r in results if r.split == "val"])
    test_rows = _rename([asdict(r) for r in results if r.split == "test"])
    _write_csv(os.path.join(out_dir, "table_A_validation.csv"), val_rows,  TABLE_AB_FIELDS)
    _write_csv(os.path.join(out_dir, "table_B_test.csv"),       test_rows, TABLE_AB_FIELDS)


# ── Table C (recurvature only) ────────────────────────────────────────────────

TABLE_C_FIELDS = ["model", "split", "ADE_rec", "FDE_rec", "HE_72h", "N_cases"]

def save_table_C(results: List[ModelResult], out_dir: str) -> None:
    rows = [{"model":   r.model_name, "split":   r.split,
             "ADE_rec": r.ADE_rec,    "FDE_rec": float("nan"),
             "HE_72h":  r.HE_72h,    "N_cases": r.n_recurv}
            for r in results]
    _write_csv(os.path.join(out_dir, "table_C_recurvature.csv"), rows, TABLE_C_FIELDS)


# ── Table D (ablation) ────────────────────────────────────────────────────────

TABLE_D_FIELDS = [
    "config",
    "use_fm", "use_dir", "use_disp", "use_smooth", "use_pinn", "use_heading",
    "val_ADE", "val_ADE_rec",
    "test_ADE", "test_ADE_rec", "test_HE_72",
]

def save_table_D(rows: List[AblationRow], out_dir: str) -> None:
    _write_csv(os.path.join(out_dir, "table_D_ablation.csv"),
               [asdict(r) for r in rows], TABLE_D_FIELDS)


# ── Statistical tests ─────────────────────────────────────────────────────────

STAT_FIELDS = [
    "comparison", "N_pairs", "mean_diff_km",
    "cohen_d", "wilcoxon_p", "wilcoxon_p_bonf", "ttest_p", "significant",
]

def save_stat_tests(rows: List[StatTestRow], out_dir: str) -> None:
    _write_csv(os.path.join(out_dir, "table_stat_tests.csv"),
               [asdict(r) for r in rows], STAT_FIELDS)


# ── PINN sensitivity ──────────────────────────────────────────────────────────

PINN_FIELDS = ["lam_pinn", "delta_deg", "val_ADE", "test_ADE", "test_HE72"]

def save_pinn_sensitivity(rows: List[PINNSensRow], out_dir: str) -> None:
    _write_csv(os.path.join(out_dir, "table_pinn_sensitivity.csv"),
               [asdict(r) for r in rows], PINN_FIELDS)


# ── Compute footprint ─────────────────────────────────────────────────────────

COMPUTE_FIELDS = ["component", "params_M", "size_MB", "inference_ms"]

def save_compute_footprint(rows: List[ComputeRow], out_dir: str) -> None:
    _write_csv(os.path.join(out_dir, "table_compute_footprint.csv"),
               [asdict(r) for r in rows], COMPUTE_FIELDS)


# ── Baseline comparison ───────────────────────────────────────────────────────

BASELINE_FIELDS = ["model", "ADE", "FDE", "ADE_str", "ADE_rec",
                   "delta_rec", "train_time_h"]

def save_baseline_compare(results: List[ModelResult], out_dir: str) -> None:
    rows = [{"model": r.model_name, "ADE": r.ADE, "FDE": r.FDE,
             "ADE_str": r.ADE_str, "ADE_rec": r.ADE_rec,
             "delta_rec": r.delta_rec, "train_time_h": r.train_time_h}
            for r in results]
    _write_csv(os.path.join(out_dir, "table_baseline_compare.csv"),
               rows, BASELINE_FIELDS)


# ══════════════════════════════════════════════════════════════════════════════
#  Compute footprint profiler
# ══════════════════════════════════════════════════════════════════════════════

def profile_model_components(model, batch, device, n_runs: int = 10) -> List[ComputeRow]:
    if not HAS_TORCH:
        return []
    rows: List[ComputeRow] = []

    def _params_m(m):
        return sum(p.numel() for p in m.parameters()) / 1e6

    def _size_mb(m):
        return sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6

    def _time_ms(fn, n=n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            with torch.no_grad():
                fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    # Get underlying net (may be torch.compile wrapped)
    net = getattr(model, "_orig_mod", model)
    net = getattr(net, "net", net)
    img = batch[11].to(device)
    env = batch[13]

    for name, attr, fn_args in [
        ("UNet3D Spatial Encoder",  "spatial_enc",  lambda m: m(img)),
        ("ENV-LSTM Encoder",        "env_enc",       lambda m: m(env, img)),
    ]:
        try:
            sub = getattr(net, attr)
            rows.append(ComputeRow(name, _params_m(sub), _size_mb(sub),
                                   _time_ms(lambda m=sub, f=fn_args: f(m))))
        except Exception as ex:
            rows.append(ComputeRow(f"{name} (err)", 0, 0, 0))

    try:
        obs_t  = batch[0].to(device)
        obs_me = batch[7].to(device)
        obs_in = torch.cat([obs_t, obs_me], dim=2).permute(1, 0, 2)
        lstm   = getattr(net, "obs_lstm")
        rows.append(ComputeRow("Obs History LSTM",
                               _params_m(lstm), _size_mb(lstm),
                               _time_ms(lambda: lstm(obs_in))))
    except Exception:
        rows.append(ComputeRow("Obs History LSTM (err)", 0, 0, 0))

    try:
        rows.append(ComputeRow(
            "FM+PINN full (10 Euler steps)",
            _params_m(model), _size_mb(model),
            _time_ms(lambda: model.sample(batch, num_ensemble=1, ddim_steps=10)),
        ))
    except Exception:
        rows.append(ComputeRow("FM+PINN full (err)", 0, 0, 0))

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  CLIPER / persistence baseline error helpers
# ══════════════════════════════════════════════════════════════════════════════

def cliper_errors(
    obs_seqs: List[np.ndarray],
    gt_seqs:  List[np.ndarray],
    pred_len: int = 12,
) -> tuple:
    from utils.metrics import haversine_km
    cliper_preds = []
    errors       = []
    for obs, gt in zip(obs_seqs, gt_seqs):
        v    = (obs[-1] - obs[-2]) if len(obs) >= 2 else np.zeros(2)
        pred = np.array([obs[-1] + (k + 1) * v for k in range(pred_len)])
        cliper_preds.append(pred)
        errors.append(haversine_km(pred, gt[:pred_len], unit_01deg=True))
    return cliper_preds, np.array(errors)


def persistence_errors(
    obs_seqs: List[np.ndarray],
    gt_seqs:  List[np.ndarray],
    pred_len: int = 12,
) -> np.ndarray:
    from utils.metrics import haversine_km
    errors = []
    for obs, gt in zip(obs_seqs, gt_seqs):
        pred = np.tile(obs[-1], (pred_len, 1))
        errors.append(haversine_km(pred, gt[:pred_len], unit_01deg=True))
    return np.array(errors)


# ══════════════════════════════════════════════════════════════════════════════
#  Default placeholder tables
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_ABLATION = [
    AblationRow("L_FM only",                     True,  False, False, False, False, False),
    AblationRow("L_FM + L_dir",                  True,  True,  False, False, False, False),
    AblationRow("L_FM + L_dir + L_disp",         True,  True,  True,  False, False, False),
    AblationRow("L_FM + L_dir + L_disp + L_sm",  True,  True,  True,  True,  False, False),
    AblationRow("L_FM + L_PINN",                 True,  False, False, False, True,  False),
    AblationRow("L_FM + L_heading (full)",        True,  True,  True,  True,  True,  True),
]

DEFAULT_PINN_SENSITIVITY = [
    PINNSensRow(lam, delta)
    for lam   in [0.1, 0.3, 0.5, 1.0, 2.0]
    for delta in [0.05, 0.10, 0.20]
]

DEFAULT_COMPUTE = [
    ComputeRow("Track Encoder (LSTM)",             0.5,   2.0,   1.2),
    ComputeRow("Spatial Encoder (UNet3D-tiny)",    2.1,   8.4,  12.5),
    ComputeRow("ENV-LSTM",                         0.3,   1.2,   0.8),
    ComputeRow("Velocity Network (Transformer)",   1.8,   7.2,   4.3),
    ComputeRow("ODE Solver (10 Euler steps)",      0.0,   0.0,  42.0),
    ComputeRow("FM+PINN (total)",                  4.7,  18.8,  61.0),
    ComputeRow("LSTM Baseline",                    0.4,   1.6,   0.9),
    ComputeRow("Diffusion+Reg",                    4.7,  18.8, 120.0),
]


# ══════════════════════════════════════════════════════════════════════════════
#  All-in-one export
# ══════════════════════════════════════════════════════════════════════════════

def export_all_tables(
    results:         List[ModelResult],
    ablation_rows:   List[AblationRow],
    stat_rows:       List[StatTestRow],
    pinn_sens_rows:  List[PINNSensRow],
    compute_rows:    List[ComputeRow],
    out_dir:         str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Exporting evaluation tables → {out_dir}")
    print(f"{'='*60}")

    save_table_AB(results, out_dir)
    save_table_C(results, out_dir)
    save_table_D(ablation_rows, out_dir)
    save_stat_tests(stat_rows, out_dir)
    save_pinn_sensitivity(pinn_sens_rows, out_dir)
    save_compute_footprint(compute_rows, out_dir)
    save_baseline_compare(results, out_dir)

    print(f"\n  ✅  All {7} tables saved to {out_dir}\n")
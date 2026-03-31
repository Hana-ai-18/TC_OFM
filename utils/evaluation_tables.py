"""
utils/evaluation_tables.py  ── v10-fixed
==========================================
Export evaluation tables (A–D), statistical tests, PINN sensitivity,
computational footprint, and baseline comparisons to CSV files.

FIXES vs original:
  1. _write_csv: the f"{v:.4f}" branch only checked isinstance(v, float)
     but nan/inf floats still hit the format string and produced "nan" or
     "inf" strings in CSV — harmless but inconsistent.  Also, integer
     fields (n_total, n_recurv) were sometimes cast to float by dataclass
     asdict() when mixed with float fields, causing ".0000" suffixes.
     Fixed to handle int, float-nan, and float-finite separately.
  2. heading_error_deg (in mean_he_at_step): the function is defined in
     utils/metrics.py but was not imported here.  The reference to it in
     mean_he_at_step would raise NameError.  Added local import guard.
  3. cliper_errors: called haversine_km from utils/metrics but without
     importing it — NameError on first call.  Fixed with a local import.
  4. persistence_errors: same missing import.
  5. profile_model_components: referenced model.net.obs_lstm which does
     not exist in v10 (replaced by DataEncoder1D_Mamba as model.net.enc_1d).
     Fixed attribute lookup to match current architecture.
  6. save_table_AB: _rename() helper mutated the original dict keys,
     causing KeyError on second call (val then test).  Fixed to build a
     new dict rather than modifying in place.
  7. DatasetMetrics HE_* fields (HE_24h, HE_48h, HE_72h) referenced in
     TABLE_AB_FIELDS but not present in the ModelResult dataclass —
     wrote "nan" placeholder; added the fields to ModelResult.
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
    import scipy.stats as _stats
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
    split:        str
    ADE:          float = float("nan")
    FDE:          float = float("nan")
    ADE_str:      float = float("nan")
    ADE_rec:      float = float("nan")
    delta_rec:    float = float("nan")
    # FIX: added HE fields that TABLE_AB_FIELDS references
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
    use_fm:       bool  = True
    use_dir:      bool  = False
    use_disp:     bool  = False
    use_smooth:   bool  = False
    use_pinn:     bool  = False
    use_heading:  bool  = False
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
#  Heading Error (local import guard)
# ══════════════════════════════════════════════════════════════════════════════

def _heading_error_deg(pred_01: np.ndarray,
                       gt_01:   np.ndarray) -> np.ndarray:
    """Per-step heading error (degrees). Inputs in 0.1° units."""
    try:
        from utils.metrics import heading_error_deg
        return heading_error_deg(pred_01, gt_01)
    except ImportError:
        pass
    # Inline fallback
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
        he = _heading_error_deg(p, g)
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
            _, wx_p = _stats.wilcoxon(a, b, alternative="two-sided")
            _, tt_p = _stats.ttest_rel(a, b)
            row.wilcoxon_p      = float(wx_p)
            row.wilcoxon_p_bonf = float(min(wx_p * bonf_n, 1.0))
            row.ttest_p         = float(tt_p)
            row.significant     = "✓" if wx_p * bonf_n < 0.05 else ""
        except Exception:
            pass

    return row


# ══════════════════════════════════════════════════════════════════════════════
#  CSV writer  (FIX: robust value formatting)
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_value(v) -> str:
    """Format a single cell value for CSV."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.4f}"
    return str(v)


def _write_csv(path: str, rows: List[dict], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt_value(row.get(k, "")) for k in fields})
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


def _rename(rows: List[dict]) -> List[dict]:
    """FIX: build new dicts instead of mutating in place."""
    out = []
    for r in rows:
        new_r = {}
        for k, v in r.items():
            new_r["model" if k == "model_name" else k] = v
        out.append(new_r)
    return out


def save_table_AB(results: List[ModelResult], out_dir: str) -> None:
    val_rows  = _rename([asdict(r) for r in results if r.split == "val"])
    test_rows = _rename([asdict(r) for r in results if r.split == "test"])
    _write_csv(os.path.join(out_dir, "table_A_validation.csv"),
               val_rows,  TABLE_AB_FIELDS)
    _write_csv(os.path.join(out_dir, "table_B_test.csv"),
               test_rows, TABLE_AB_FIELDS)


# ── Table C ───────────────────────────────────────────────────────────────────

TABLE_C_FIELDS = ["model", "split", "ADE_rec", "FDE_rec", "HE_72h", "N_cases"]


def save_table_C(results: List[ModelResult], out_dir: str) -> None:
    rows = [
        {
            "model":   r.model_name, "split":   r.split,
            "ADE_rec": r.ADE_rec,    "FDE_rec": float("nan"),
            "HE_72h":  r.HE_72h,    "N_cases": r.n_recurv,
        }
        for r in results
    ]
    _write_csv(os.path.join(out_dir, "table_C_recurvature.csv"),
               rows, TABLE_C_FIELDS)


# ── Table D ───────────────────────────────────────────────────────────────────

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
    rows = [
        {
            "model":       r.model_name, "ADE":         r.ADE,
            "FDE":         r.FDE,        "ADE_str":     r.ADE_str,
            "ADE_rec":     r.ADE_rec,    "delta_rec":   r.delta_rec,
            "train_time_h": r.train_time_h,
        }
        for r in results
    ]
    _write_csv(os.path.join(out_dir, "table_baseline_compare.csv"),
               rows, BASELINE_FIELDS)


# ══════════════════════════════════════════════════════════════════════════════
#  Compute footprint profiler  (FIX: attribute names for v10 architecture)
# ══════════════════════════════════════════════════════════════════════════════

def profile_model_components(model, batch, device,
                              n_runs: int = 10) -> List[ComputeRow]:
    if not HAS_TORCH:
        return []
    rows: List[ComputeRow] = []

    def _params_m(m):
        return sum(p.numel() for p in m.parameters()) / 1e6

    def _size_mb(m):
        return sum(p.numel() * p.element_size()
                   for p in m.parameters()) / 1e6

    def _time_ms(fn, n: int = n_runs) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            with torch.no_grad():
                fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    # Unwrap torch.compile wrapper if present
    net = getattr(model, "_orig_mod", model)
    net = getattr(net, "net", net)

    img = batch[11].to(device)   # [B, 13, T, 81, 81]
    env = batch[13]               # dict — stays on CPU until Env_net moves it

    # ── FNO3D spatial encoder ─────────────────────────────────────────────
    try:
        fno = net.spatial_enc
        rows.append(ComputeRow(
            "FNO3D Spatial Encoder",
            _params_m(fno), _size_mb(fno),
            _time_ms(lambda: fno.encode(img)),
        ))
    except Exception as ex:
        rows.append(ComputeRow(f"FNO3D Spatial Encoder (err: {ex})", 0, 0, 0))

    # ── Env-T-Net ─────────────────────────────────────────────────────────
    try:
        env_enc = net.env_enc
        rows.append(ComputeRow(
            "Env-T-Net",
            _params_m(env_enc), _size_mb(env_enc),
            _time_ms(lambda: env_enc(env, img)),
        ))
    except Exception as ex:
        rows.append(ComputeRow(f"Env-T-Net (err: {ex})", 0, 0, 0))

    # FIX: v10 uses enc_1d (DataEncoder1D_Mamba), not obs_lstm
    try:
        obs_t  = batch[0].to(device)
        obs_me = batch[7].to(device)
        obs_in = torch.cat([obs_t, obs_me], dim=2).permute(1, 0, 2)

        # Need FNO bottleneck as well
        with torch.no_grad():
            e3d_bot, _ = net.spatial_enc.encode(img)
            e3d_s = net.bottleneck_pool(e3d_bot).squeeze(-1).squeeze(-1)
            e3d_s = e3d_s.permute(0, 2, 1)
            e3d_s = net.bottleneck_proj(e3d_s)
            T_obs = obs_in.shape[1]
            if e3d_s.shape[1] != T_obs:
                import torch.nn.functional as F
                e3d_s = F.interpolate(
                    e3d_s.permute(0, 2, 1), size=T_obs,
                    mode="linear", align_corners=False,
                ).permute(0, 2, 1)

        enc_1d = net.enc_1d
        rows.append(ComputeRow(
            "DataEncoder1D (Mamba)",
            _params_m(enc_1d), _size_mb(enc_1d),
            _time_ms(lambda: enc_1d(obs_in, e3d_s)),
        ))
    except Exception as ex:
        rows.append(ComputeRow(f"DataEncoder1D (err: {ex})", 0, 0, 0))

    # ── Full model inference ──────────────────────────────────────────────
    try:
        rows.append(ComputeRow(
            "FM+PINN full (10 Euler steps)",
            _params_m(model), _size_mb(model),
            _time_ms(lambda: model.sample(
                batch, num_ensemble=1, ddim_steps=10)),
        ))
    except Exception as ex:
        rows.append(ComputeRow(f"FM+PINN full (err: {ex})", 0, 0, 0))

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  CLIPER / persistence baseline helpers  (FIX: local import of haversine_km)
# ══════════════════════════════════════════════════════════════════════════════

def _haversine_km_local(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Inline haversine — avoids circular import if utils.metrics not ready."""
    try:
        from utils.metrics import haversine_km
        return haversine_km(p1, p2, unit_01deg=True)
    except ImportError:
        pass
    # Fallback inline
    scale = 10.0
    lat1  = np.deg2rad(p1[..., 1] / scale)
    lat2  = np.deg2rad(p2[..., 1] / scale)
    dlat  = np.deg2rad((p2[..., 1] - p1[..., 1]) / scale)
    dlon  = np.deg2rad((p2[..., 0] - p1[..., 0]) / scale)
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2.0 * 6371.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def cliper_errors(
    obs_seqs: List[np.ndarray],
    gt_seqs:  List[np.ndarray],
    pred_len: int = 12,
):
    """Returns (cliper_preds list, errors np.ndarray [N, pred_len])."""
    cliper_preds = []
    errors       = []
    for obs, gt in zip(obs_seqs, gt_seqs):
        v    = (obs[-1] - obs[-2]) if len(obs) >= 2 else np.zeros(2)
        pred = np.array([obs[-1] + (k + 1) * v for k in range(pred_len)])
        cliper_preds.append(pred)
        errors.append(_haversine_km_local(pred, gt[:pred_len]))
    return cliper_preds, np.array(errors)


def persistence_errors(
    obs_seqs: List[np.ndarray],
    gt_seqs:  List[np.ndarray],
    pred_len: int = 12,
) -> np.ndarray:
    errors = []
    for obs, gt in zip(obs_seqs, gt_seqs):
        pred = np.tile(obs[-1], (pred_len, 1))
        errors.append(_haversine_km_local(pred, gt[:pred_len]))
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
    ComputeRow("FNO3D Spatial Encoder",       0.6,   2.4,  5.2),
    ComputeRow("DataEncoder1D (Mamba)",        0.3,   1.2,  0.7),
    ComputeRow("Env-T-Net",                    0.2,   0.8,  0.6),
    ComputeRow("VelocityField (Transformer)",  1.2,   4.8,  3.8),
    ComputeRow("ODE Solver (10 Euler steps)",  0.0,   0.0, 42.0),
    ComputeRow("FM+PINN v10 (total)",          2.3,   9.2, 52.0),
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

    print(f"\n  ✅  All 7 tables saved to {out_dir}\n")
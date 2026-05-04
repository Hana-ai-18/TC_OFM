# -*- coding: utf-8 -*-
"""
tc_metrics.py
=============
Shared evaluation metrics for TC track forecasting.
Used by: PaperBaseline (LSTM/GRU), STTrans, SSFM (TCFlowMatching).

Metrics:
  - ADE  : Average Displacement Error (haversine, km)
  - ATE  : Along-Track Error (km, signed: positive = overshoot)
  - CTE  : Cross-Track Error (km, signed: positive = right-of-track)
  - CRPS : Continuous Ranked Probability Score (km)
           - Deterministic: CRPS = MAE = ADE
           - Ensemble (N samples): proper CRPS energy score

All functions expect inputs in NORMALISED space and convert internally.

Normalisation convention (from your dataset):
  lon_norm = (lon_deg * 10 - 1800) / 50   =>  lon_deg = (lon_norm * 50 + 1800) / 10
  lat_norm = lat_deg * 10 / 50             =>  lat_deg = lat_norm * 50 / 10
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
R_EARTH  = 6371.0
DEG2KM   = 111.0
EPS      = 1e-6

HORIZON_STEPS: Dict[int, int] = {12: 1, 24: 3, 48: 7, 72: 11}

STEP_WEIGHTS = [3.0, 2.5, 2.0, 2.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0]


# ══════════════════════════════════════════════════════════════════════════════
#  Coordinate helpers
# ══════════════════════════════════════════════════════════════════════════════

def norm_to_deg(t: torch.Tensor) -> torch.Tensor:
    """Normalised -> degrees. t: [..., 2] (lon, lat)"""
    return torch.stack([
        (t[..., 0] * 50.0 + 1800.0) / 10.0,
        (t[..., 1] * 50.0) / 10.0,
    ], dim=-1)


def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    Haversine distance in km.
    p1, p2: [..., 2] in DEGREES (lon, lat)
    """
    lat1 = torch.deg2rad(p1[..., 1]);  lat2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a = (torch.sin(dlat / 2).pow(2) +
         torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# ══════════════════════════════════════════════════════════════════════════════
#  ATE / CTE
# ══════════════════════════════════════════════════════════════════════════════

def compute_ate_cte(
    pred_norm: torch.Tensor,   # (T, B, 2) normalised
    gt_norm:   torch.Tensor,   # (T, B, 2) normalised
) -> Dict[str, float]:
    """
    Compute mean |ATE| and mean |CTE| in km.

    ATE = along-track component of position error
    CTE = cross-track component of position error

    Uses km-space projection to avoid cos(lat) double-counting.

    Returns dict with keys: ate_km, cte_km, ate_signed, cte_signed
    and per-horizon values ate_12h, ate_24h, ate_48h, ate_72h,
                           cte_12h, cte_24h, cte_48h, cte_72h
    """
    T = min(pred_norm.shape[0], gt_norm.shape[0])
    if T < 2:
        return {k: 0.0 for k in ['ate_km', 'cte_km', 'ate_signed', 'cte_signed']}

    pred_deg = norm_to_deg(pred_norm[:T])   # (T, B, 2)
    gt_deg   = norm_to_deg(gt_norm[:T])     # (T, B, 2)

    # Along-track direction in km-space (T-1 steps)
    cos_lat = torch.cos(torch.deg2rad(gt_deg[1:T, :, 1])).clamp(EPS)

    dx_gt_km = (gt_deg[1:T, :, 0] - gt_deg[:T-1, :, 0]) * cos_lat * DEG2KM
    dy_gt_km = (gt_deg[1:T, :, 1] - gt_deg[:T-1, :, 1]) * DEG2KM
    along = F.normalize(torch.stack([dx_gt_km, dy_gt_km], dim=-1), dim=-1, eps=EPS)
    # Cross-track = perpendicular (rotate 90 deg)
    cross = torch.stack([-along[..., 1], along[..., 0]], dim=-1)

    # Error vector in km at positions [1..T-1]
    dx_err = (pred_deg[1:T, :, 0] - gt_deg[1:T, :, 0]) * cos_lat * DEG2KM
    dy_err = (pred_deg[1:T, :, 1] - gt_deg[1:T, :, 1]) * DEG2KM
    err_km = torch.stack([dx_err, dy_err], dim=-1)   # (T-1, B, 2)

    ate_signed = (err_km * along).sum(-1)   # (T-1, B)
    cte_signed = (err_km * cross).sum(-1)   # (T-1, B)

    result = {
        'ate_km'    : float(ate_signed.abs().mean()),
        'cte_km'    : float(cte_signed.abs().mean()),
        'ate_signed': float(ate_signed.mean()),    # positive = overshoot
        'cte_signed': float(cte_signed.mean()),    # positive = right of track
    }

    # Per-horizon (step index in the T-1 array = step - 1)
    for h, s in HORIZON_STEPS.items():
        idx = s - 1   # ate/cte are at index s-1 of the T-1 array
        if 0 <= idx < ate_signed.shape[0]:
            result[f'ate_{h}h'] = float(ate_signed[idx].abs().mean())
            result[f'cte_{h}h'] = float(cte_signed[idx].abs().mean())
        else:
            result[f'ate_{h}h'] = float('nan')
            result[f'cte_{h}h'] = float('nan')

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CRPS
# ══════════════════════════════════════════════════════════════════════════════

def compute_crps_deterministic(
    pred_norm: torch.Tensor,   # (T, B, 2) normalised
    gt_norm:   torch.Tensor,   # (T, B, 2) normalised
) -> Dict[str, float]:
    """
    CRPS for deterministic models (LSTM, ST-Trans, Non-AR decoder).
    For a single sample, CRPS = MAE = haversine(pred, gt).
    This is the degenerate but correct formula: CRPS_det = E[|X-y|].

    Returns dict: crps_km, and per-horizon crps_12h .. crps_72h
    """
    T = min(pred_norm.shape[0], gt_norm.shape[0])
    pred_deg = norm_to_deg(pred_norm[:T])
    gt_deg   = norm_to_deg(gt_norm[:T])
    dist     = haversine_km(pred_deg, gt_deg)   # (T, B)

    result = {'crps_km': float(dist.mean())}
    for h, s in HORIZON_STEPS.items():
        if s < T:
            result[f'crps_{h}h'] = float(dist[s].mean())
        else:
            result[f'crps_{h}h'] = float('nan')
    return result


def compute_crps_ensemble(
    samples_norm: torch.Tensor,   # (N, T, B, 2) normalised — N ensemble members
    gt_norm:      torch.Tensor,   # (T, B, 2) normalised
) -> Dict[str, float]:
    """
    Proper CRPS energy score for ensemble forecasts.

    CRPS = E[dist(X, y)] - 0.5 * E[dist(X, X')]
    where X, X' ~ p(forecast) and y = ground truth.

    This is the standard energy form of CRPS generalised to 2D:
      CRPS = (1/N)*sum_i dist(s_i, y)
             - (1/(2*N^2))*sum_i sum_j dist(s_i, s_j)

    Lower is better. For a perfect deterministic forecast, CRPS = 0.
    Unit: km.

    Returns dict: crps_km (mean over all steps and batch),
                  and per-horizon crps_12h .. crps_72h,
                  spread_km (mean ensemble spread = 2nd term)
    """
    N, T_s, B, _ = samples_norm.shape
    T = min(T_s, gt_norm.shape[0])

    samples_deg = norm_to_deg(samples_norm[:, :T])   # (N, T, B, 2)
    gt_deg      = norm_to_deg(gt_norm[:T])            # (T, B, 2)

    # Term 1: E[dist(X, y)] = (1/N) sum_i haversine(s_i, y)
    # samples_deg: (N, T, B, 2)  gt_deg: (T, B, 2)
    dist_to_gt = torch.stack([
        haversine_km(samples_deg[i], gt_deg) for i in range(N)
    ])                                                # (N, T, B)
    term1 = dist_to_gt.mean(0)                        # (T, B)

    # Term 2: E[dist(X, X')] = (1/N^2) sum_i sum_j haversine(s_i, s_j)
    # Efficient: use all pairs
    # For large N, subsample to N' = min(N, 20) for speed
    N_sub = min(N, 20)
    idx   = torch.randperm(N, device=samples_norm.device)[:N_sub]
    s_sub = samples_deg[idx]   # (N_sub, T, B, 2)

    pair_dists = []
    for i in range(N_sub):
        for j in range(i+1, N_sub):
            pair_dists.append(haversine_km(s_sub[i], s_sub[j]))  # (T, B)
    if pair_dists:
        spread = torch.stack(pair_dists).mean(0)   # (T, B) mean pairwise dist
    else:
        spread = torch.zeros_like(term1)

    crps_field = term1 - 0.5 * spread   # (T, B)

    result = {
        'crps_km'  : float(crps_field.mean()),
        'spread_km': float(spread.mean()),
    }
    for h, s in HORIZON_STEPS.items():
        if s < T:
            result[f'crps_{h}h']   = float(crps_field[s].mean())
            result[f'spread_{h}h'] = float(spread[s].mean())
        else:
            result[f'crps_{h}h']   = float('nan')
            result[f'spread_{h}h'] = float('nan')

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Full evaluation: ADE + ATE + CTE + CRPS in one call
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all(
    pred_norm:    torch.Tensor,             # (T, B, 2)  deterministic pred
    gt_norm:      torch.Tensor,             # (T, B, 2)
    samples_norm: Optional[torch.Tensor] = None,  # (N, T, B, 2) ensemble
) -> Dict[str, float]:
    """
    Compute all metrics in one call.
    If samples_norm is provided, uses ensemble CRPS.
    Otherwise uses deterministic CRPS (= ADE).

    Returns flat dict with all metrics:
      ade_km, ate_km, cte_km, crps_km, spread_km (if ensemble),
      {metric}_{12h,24h,48h,72h}
    """
    T = min(pred_norm.shape[0], gt_norm.shape[0])

    # ADE per horizon
    pred_deg = norm_to_deg(pred_norm[:T])
    gt_deg   = norm_to_deg(gt_norm[:T])
    dist     = haversine_km(pred_deg, gt_deg)   # (T, B)

    result: Dict[str, float] = {
        'ade_km': float(dist.mean()),
        'fde_km': float(dist[-1].mean()),
    }
    for h, s in HORIZON_STEPS.items():
        if s < T:
            result[f'ade_{h}h'] = float(dist[s].mean())

    # ATE / CTE
    ate_cte = compute_ate_cte(pred_norm, gt_norm)
    result.update(ate_cte)

    # CRPS
    if samples_norm is not None and samples_norm.shape[0] > 1:
        crps = compute_crps_ensemble(samples_norm, gt_norm)
    else:
        crps = compute_crps_deterministic(pred_norm, gt_norm)
    result.update(crps)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Calibration: reliability diagram data
# ══════════════════════════════════════════════════════════════════════════════

def compute_reliability(
    samples_norm: torch.Tensor,   # (N, T, B, 2)
    gt_norm:      torch.Tensor,   # (T, B, 2)
    quantiles: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> Dict[str, list]:
    """
    Compute reliability diagram data.
    For each quantile q, compute: fraction of GT points within q-th quantile
    of ensemble spread (i.e., within the q-th percentile circle).

    A calibrated model should have: observed_freq ≈ q for all q.

    Returns: {'quantiles': [...], 'observed_freq': [...]}
    """
    N, T_s, B, _ = samples_norm.shape
    T = min(T_s, gt_norm.shape[0])

    samples_deg = norm_to_deg(samples_norm[:, :T])
    gt_deg      = norm_to_deg(gt_norm[:T])

    # Compute distance from each sample to GT for each (t, b)
    dist_to_gt = torch.stack([
        haversine_km(samples_deg[i], gt_deg) for i in range(N)
    ])   # (N, T, B)

    # Compute quantile radii from ensemble
    dist_to_mean = []
    mean_pred = samples_deg.mean(0)   # (T, B, 2)
    for i in range(N):
        dist_to_mean.append(haversine_km(samples_deg[i], mean_pred))
    dist_to_mean_t = torch.stack(dist_to_mean)   # (N, T, B)

    observed_freqs = []
    for q in quantiles:
        # Quantile radius at each (t, b)
        r_q = torch.quantile(dist_to_mean_t, q, dim=0)   # (T, B)
        # Is GT within r_q of ensemble mean?
        dist_gt_to_mean = haversine_km(gt_deg, mean_pred)  # (T, B)
        within = (dist_gt_to_mean <= r_q).float()
        observed_freqs.append(float(within.mean()))

    return {
        'quantiles'     : list(quantiles),
        'observed_freq' : observed_freqs,
        'sharpness_km'  : float(dist_to_mean_t.mean()),
    }
# """
# utils/metrics.py  ── v3
# =============================
# TC Track Forecasting Metrics — 4-tier evaluation framework.

# Tier 1 — Deterministic position accuracy:
#     UGDE(h), ADE, FDE  (Haversine great-circle, km)

# Tier 2 — Operational skill & decomposition:
#     TSS(h) vs CLIPER baseline
#     ATE(h), CTE(h)  — along-track / cross-track decomposition
#     CSD / PR        — straight vs recurvature stratification
#     RDR             — recurvature detection rate (ensemble)

# Tier 3 — Probabilistic (ensemble):
#     CRPS(h)  — continuous ranked probability score
#     SSR(h)   — spread-skill ratio
#     BSS      — Brier skill score for landfall probability

# Tier 4 — Geometric & shape:
#     DTW      — dynamic time warping (Haversine local cost)
#     OYR      — off-yaw rate  (fraction of steps with heading error > 90°)
#     HLE      — heading loss error  (MAE of signed curvature)

# References
# ----------
# UGDE / TSS : Huang et al. 2025 (DOI: 10.3390/rs17152675)
# CRPS       : Gneiting & Raftery 2007 (DOI: 10.1198/016214506000001437)
# SSR        : Zhong et al. 2025 (DOI: 10.1038/s41612-025-01009-9)
# ATE / CTE  : Sharma et al. 2020 (DOI: 10.1016/j.tcrr.2020.04.004)
# CSD        : Fisher 1993; Batschelet 1981
# OYR        : Greer et al. 2021 (arXiv:2011.06679)
# HLE        : Original contribution (eval-time counterpart of L5)
# DTW        : Berndt & Clifford 1994

# TCND_VN normalisation convention
# ----------------------------------
#     norm_lon = (lon_01E − 1800) / 50   →   lon_01E = norm × 50 + 1800
#     norm_lat = lat_01N / 50            →   lat_01N = norm × 50
#     (values in 0.1° units — divide by 10 for degrees)
# """

# from __future__ import annotations

# import csv
# import os
# from dataclasses import dataclass, field
# from datetime import datetime
# from typing import Dict, List, Optional, Tuple

# import numpy as np

# try:
#     import torch
#     HAS_TORCH = True
# except ImportError:
#     HAS_TORCH = False
#     torch = None  # type: ignore

# # ── Physical constants ────────────────────────────────────────────────────────
# R_EARTH_KM   = 6371.0
# STEP_HOURS   = 6
# PRED_LEN     = 12

# # CLIPER WNP climatological decay (linear blend weight)
# # alpha_h = h / 12  →  0 = pure persistence, 1 = pure climatology
# CLIPER_ALPHA = {h: h / 12 for h in range(1, 13)}

# # Standard evaluation horizons (step index, 0-based)
# HORIZON_STEPS: Dict[int, int] = {12: 1, 24: 3, 36: 5, 48: 7, 60: 9, 72: 11}

# # Recurvature / straight-track threshold
# RECURV_THR_DEG = 45.0

# # BSS landfall targets (WNP) — (name, lon_deg, lat_deg)
# LANDFALL_TARGETS = [
#     ("DaNang",    108.2, 16.1),
#     ("Manila",    121.0, 14.6),
#     ("Taipei",    121.5, 25.0),
#     ("Shanghai",  121.5, 31.2),
#     ("Okinawa",   127.8, 26.3),
# ]
# LANDFALL_RADIUS_KM = 300.0


# # ══════════════════════════════════════════════════════════════════════════════
# #  1. Primitive: Haversine great-circle distance
# # ══════════════════════════════════════════════════════════════════════════════

# def haversine_km(
#     p1: np.ndarray,
#     p2: np.ndarray,
#     lon_idx: int = 0,
#     lat_idx: int = 1,
#     unit_01deg: bool = True,
# ) -> np.ndarray:
#     """
#     Haversine distance (km) between arrays of positions.

#     Args
#     ----
#     p1, p2      : shape [..., 2+]  — (lon, lat) in 0.1° or degrees
#     unit_01deg  : True → divide by 10 to convert 0.1° → degrees

#     Formula
#     -------
#     a = sin²(Δφ/2) + cos φ₁·cos φ₂·sin²(Δλ/2)
#     d = 2·R·arcsin(√a)
#     """
#     scale = 10.0 if unit_01deg else 1.0

#     lat1 = np.deg2rad(p1[..., lat_idx] / scale)
#     lat2 = np.deg2rad(p2[..., lat_idx] / scale)
#     dlon = np.deg2rad((p2[..., lon_idx] - p1[..., lon_idx]) / scale)
#     dlat = np.deg2rad((p2[..., lat_idx] - p1[..., lat_idx]) / scale)

#     a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
#     return 2.0 * R_EARTH_KM * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


# def haversine_km_torch(pred, gt, lon_idx: int = 0, lat_idx: int = 1,
#                         unit_01deg: bool = True):
#     """Haversine distance (km) — differentiable PyTorch version."""
#     scale = 10.0 if unit_01deg else 1.0
#     lat1 = torch.deg2rad(gt[..., lat_idx]   / scale)
#     lat2 = torch.deg2rad(pred[..., lat_idx] / scale)
#     dlon = torch.deg2rad((pred[..., lon_idx] - gt[..., lon_idx]) / scale)
#     dlat = torch.deg2rad((pred[..., lat_idx] - gt[..., lat_idx]) / scale)
#     a = (torch.sin(dlat / 2.0) ** 2
#          + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2)
#     return 2.0 * R_EARTH_KM * torch.asin(a.clamp(0.0, 1.0).sqrt())


# # ── Denormalisation helpers ───────────────────────────────────────────────────

# def denorm_np(n: np.ndarray) -> np.ndarray:
#     """Normalised coords → 0.1° units (NumPy)."""
#     r = n.copy()
#     r[..., 0] = n[..., 0] * 50.0 + 1800.0
#     r[..., 1] = n[..., 1] * 50.0
#     return r


# def denorm_torch(n):
#     """Normalised coords → 0.1° units (PyTorch)."""
#     r = n.clone()
#     r[..., 0] = n[..., 0] * 50.0 + 1800.0
#     r[..., 1] = n[..., 1] * 50.0
#     return r


# # ══════════════════════════════════════════════════════════════════════════════
# #  2. Tier 1 — Deterministic position accuracy
# # ══════════════════════════════════════════════════════════════════════════════

# def ugde(pred_01: np.ndarray, gt_01: np.ndarray) -> np.ndarray:
#     """
#     Unified Geodesic Displacement Error per lead step.

#     Args
#     ----
#     pred_01, gt_01 : [T, 2+]  single sequence in 0.1° units

#     Returns
#     -------
#     per_step_km : [T] — UGDE at each step
#     """
#     return haversine_km(pred_01, gt_01)


# def ade_fde(pred_01: np.ndarray, gt_01: np.ndarray
#             ) -> Tuple[float, float, np.ndarray]:
#     """ADE (km), FDE (km), per-step error [T]."""
#     ps = ugde(pred_01, gt_01)
#     return float(ps.mean()), float(ps[-1]), ps


# # ══════════════════════════════════════════════════════════════════════════════
# #  3. Tier 2 — Operational skill & decomposition
# # ══════════════════════════════════════════════════════════════════════════════

# def tss(ugde_model: float, ugde_cliper: float) -> float:
#     """Track Skill Score vs CLIPER.  TSS > 0 ⟺ beats persistence."""
#     return 1.0 - ugde_model / (ugde_cliper + 1e-8)


# def cliper_forecast(obs_01: np.ndarray, h: int) -> np.ndarray:
#     """
#     Simple CLIPER-WNP: linear extrapolation from last two observed steps.
#     Returns predicted position (0.1° units) at lead step h (1-indexed).
#     """
#     if obs_01.shape[0] < 2:
#         return obs_01[-1].copy()
#     v = obs_01[-1] - obs_01[-2]                # last observed velocity
#     return obs_01[-1] + h * v                  # pure persistence


# def lat_corrected_velocity(traj_01: np.ndarray,
#                             lon_idx: int = 0,
#                             lat_idx: int = 1) -> np.ndarray:
#     """
#     Latitude-corrected 2-D velocity vectors  [T-1, 2].
#         u_k = Δlon_k × cos(lat_k)
#         v_k = Δlat_k
#     """
#     lats_rad = np.deg2rad(traj_01[:, lat_idx] / 10.0)
#     cos_lat  = np.cos(lats_rad[:-1])
#     dlat = np.diff(traj_01[:, lat_idx])
#     dlon = np.diff(traj_01[:, lon_idx]) * cos_lat
#     return np.stack([dlon, dlat], axis=-1)     # [T-1, 2]


# def total_rotation_angle(gt_01: np.ndarray) -> float:
#     """
#     Total signed turning angle Θ (degrees) — used for recurvature classification.
#     Θ ≥ RECURV_THR_DEG → recurvature.
#     """
#     if gt_01.shape[0] < 3:
#         return 0.0
#     v = lat_corrected_velocity(gt_01)          # [T-1, 2]
#     total = 0.0
#     for i in range(len(v) - 1):
#         v1, v2 = v[i], v[i + 1]
#         n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
#         if n1 < 1e-8 or n2 < 1e-8:
#             continue
#         cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
#         total += np.degrees(np.arccos(cos_a))
#     return total


# def classify(gt_01: np.ndarray, thr: float = RECURV_THR_DEG) -> str:
#     return "recurvature" if total_rotation_angle(gt_01) >= thr else "straight"


# def ate_cte(pred_01: np.ndarray, gt_01: np.ndarray,
#             lon_idx: int = 0, lat_idx: int = 1
#             ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Along-track error (ATE) and cross-track error (CTE) per step [T].

#     Sign conventions
#     ----------------
#     ATE > 0 : predicted storm is ahead  (too fast)
#     ATE < 0 : predicted storm is behind (too slow)
#     CTE > 0 : leans right of true track
#     CTE < 0 : leans left  of true track
#     """
#     T = pred_01.shape[0]
#     ate_arr = np.zeros(T)
#     cte_arr = np.zeros(T)

#     for k in range(T):
#         # Unit tangent from step k-1 → k on ground-truth track
#         if k == 0:
#             if T > 1:
#                 dk = gt_01[1] - gt_01[0]
#             else:
#                 dk = np.array([1.0, 0.0])
#         else:
#             dk = gt_01[k] - gt_01[k - 1]

#         norm_dk = np.linalg.norm(dk)
#         if norm_dk < 1e-8:
#             continue
#         t_hat = dk / norm_dk
#         n_hat = np.array([-t_hat[1], t_hat[0]])    # rotate 90° left

#         delta = pred_01[k] - gt_01[k]
#         ate_arr[k] = float(np.dot(delta, t_hat))
#         cte_arr[k] = float(np.dot(delta, n_hat))

#     return ate_arr, cte_arr


# def circular_std(angles_deg: np.ndarray) -> float:
#     """Circular standard deviation of heading angles (degrees)."""
#     if len(angles_deg) == 0:
#         return 0.0
#     rads  = np.deg2rad(angles_deg)
#     R_bar = np.abs(np.mean(np.exp(1j * rads)))
#     R_bar = np.clip(R_bar, 1e-8, 1.0 - 1e-8)
#     return float(np.degrees(np.sqrt(-2.0 * np.log(R_bar))))


# def compute_csd(traj_01: np.ndarray) -> float:
#     """CSD of a single trajectory (degrees)."""
#     v = lat_corrected_velocity(traj_01)
#     if len(v) == 0:
#         return 0.0
#     angles = np.arctan2(v[:, 1], v[:, 0])
#     return circular_std(np.degrees(angles))


# def oyr(pred_01: np.ndarray, gt_01: np.ndarray) -> float:
#     """
#     Off-Yaw Rate — fraction of steps where predicted heading is
#     antiparallel to ground truth (dot product < 0, i.e. angle > 90°).

#     Greer et al. 2021 (arXiv:2011.06679), adapted:
#     v_lane → v_gt  (no HD map required).
#     """
#     pv = lat_corrected_velocity(pred_01)
#     gv = lat_corrected_velocity(gt_01)
#     m  = min(len(pv), len(gv))
#     if m == 0:
#         return 0.0
#     dots = np.sum(pv[:m] * gv[:m], axis=1)
#     np_  = np.linalg.norm(pv[:m], axis=1)
#     ng_  = np.linalg.norm(gv[:m], axis=1)
#     valid = (np_ > 1e-8) & (ng_ > 1e-8)
#     if valid.sum() == 0:
#         return 0.0
#     cos_vals = dots[valid] / (np_[valid] * ng_[valid])
#     return float((cos_vals < 0).mean())


# def hle(pred_01: np.ndarray, gt_01: np.ndarray) -> float:
#     """
#     Heading Loss Error — MAE of signed step curvature.

#     Evaluation-time counterpart of L5(b).
#     κ_k = (v_{k+1} × v_k) / (|v_{k+1}||v_k|) = sin θ_k
#     """
#     def curvature(traj):
#         v = lat_corrected_velocity(traj)         # [T-1, 2]
#         if len(v) < 2:
#             return np.array([])
#         cross = v[1:, 0] * v[:-1, 1] - v[1:, 1] * v[:-1, 0]
#         n1    = np.linalg.norm(v[1:],  axis=1).clip(1e-8)
#         n2    = np.linalg.norm(v[:-1], axis=1).clip(1e-8)
#         return cross / (n1 * n2)

#     kp = curvature(pred_01)
#     kg = curvature(gt_01)
#     m  = min(len(kp), len(kg))
#     if m == 0:
#         return float("nan")
#     return float(np.mean(np.abs(kp[:m] - kg[:m])))


# # ══════════════════════════════════════════════════════════════════════════════
# #  4. Tier 3 — Probabilistic (ensemble)
# # ══════════════════════════════════════════════════════════════════════════════

# def crps_2d(pred_ens_01: np.ndarray, gt_01: np.ndarray) -> float:
#     """
#     CRPS energy-form for 2-D TC track position.

#     Args
#     ----
#     pred_ens_01 : [S, 2+]  ensemble samples at ONE lead step (0.1° units)
#     gt_01       : [2+]     ground truth at the same lead step

#     Returns
#     -------
#     crps : float (km)

#     Reference
#     ---------
#     Gneiting & Raftery 2007, Eq. 21  (energy score with ψ(d) = d)
#     """
#     S = pred_ens_01.shape[0]
#     gt_rep  = gt_01[np.newaxis].repeat(S, axis=0)           # [S, 2+]
#     acc     = np.mean(haversine_km(pred_ens_01, gt_rep))    # accuracy term

#     # Diversity term: E[|X − X'|]
#     div = 0.0
#     for s in range(S):
#         rep = pred_ens_01[[s]].repeat(S, axis=0)
#         div += np.mean(haversine_km(pred_ens_01, rep))
#     div /= (S * S)

#     return float(acc - 0.5 * div)


# def ssr_step(pred_ens_01: np.ndarray, gt_01: np.ndarray) -> float:
#     """
#     Spread-Skill Ratio at ONE lead step.

#     SSR = σ_spread / RMSE_ensemble_mean

#     SSR ≈ 1 : well-calibrated
#     SSR < 1 : under-dispersive
#     SSR > 1 : over-dispersive

#     Reference: Zhong et al. 2025; Chen et al. 2025
#     """
#     S = pred_ens_01.shape[0]
#     ens_mean = pred_ens_01.mean(axis=0)                       # [2+]

#     # Spread: std of ensemble members around their mean
#     ens_mean_rep = ens_mean[np.newaxis].repeat(S, axis=0)
#     spread_vals  = haversine_km(pred_ens_01, ens_mean_rep)    # [S]
#     spread       = float(np.sqrt(np.mean(spread_vals ** 2)))

#     # RMSE of ensemble mean vs ground truth
#     rmse_em = float(haversine_km(ens_mean[np.newaxis], gt_01[np.newaxis])[0])

#     return spread / (rmse_em + 1e-8)


# def brier_skill_score(
#     pred_ens_seqs: List[np.ndarray],    # list of [S, T, 2+]
#     gt_seqs:       List[np.ndarray],    # list of [T, 2+]
#     step:          int,                 # 0-based step index
#     target_deg:    Tuple[float, float], # (lon_deg, lat_deg)
#     radius_km:     float = LANDFALL_RADIUS_KM,
#     clim_rate:     Optional[float] = None,
# ) -> float:
#     """
#     Brier Skill Score for landfall strike probability at a target location.

#     Reference: Brier 1950; Leonardo & Colle 2017
#     """
#     N   = len(gt_seqs)
#     bs  = 0.0
#     clim_hits = 0

#     target_01 = np.array([target_deg[0] * 10.0, target_deg[1] * 10.0])

#     for i in range(N):
#         gt_pos   = gt_seqs[i][step]
#         ens_seqs = pred_ens_seqs[i]              # [S, T, 2+]
#         S        = ens_seqs.shape[0]

#         obs = float(haversine_km(
#             gt_pos[np.newaxis], target_01[np.newaxis])[0]) <= radius_km

#         hits = sum(
#             haversine_km(ens_seqs[s, step][np.newaxis],
#                          target_01[np.newaxis])[0] <= radius_km
#             for s in range(S)
#         )
#         p_fc = hits / S
#         bs  += (p_fc - float(obs)) ** 2
#         clim_hits += float(obs)

#     bs /= N
#     p_clim = (clim_rate if clim_rate is not None
#               else clim_hits / N)
#     bs_clim = p_clim * (1.0 - p_clim) + 1e-8
#     return float(1.0 - bs / bs_clim)


# # ══════════════════════════════════════════════════════════════════════════════
# #  5. Tier 4 — Geometric & shape
# # ══════════════════════════════════════════════════════════════════════════════

# def dtw_haversine(s: np.ndarray, t: np.ndarray) -> float:
#     """
#     Dynamic Time Warping distance using Haversine local cost.

#     Args
#     ----
#     s, t : [T, 2+]  in 0.1° units

#     Complexity : O(T²)  — fine for T=12

#     Reference : Berndt & Clifford 1994
#     """
#     n, m = len(s), len(t)
#     dp   = np.full((n + 1, m + 1), np.inf)
#     dp[0, 0] = 0.0
#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             cost = float(haversine_km(s[i-1:i], t[j-1:j])[0])
#             dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
#     return float(dp[n, m])


# # ══════════════════════════════════════════════════════════════════════════════
# #  6. Per-sequence result container
# # ══════════════════════════════════════════════════════════════════════════════

# @dataclass
# class SequenceResult:
#     """All metrics for one forecast sequence."""
#     # Tier 1
#     ade:       float
#     fde:       float
#     per_step:  np.ndarray       # [T] km

#     # Tier 2
#     ate:       np.ndarray       # [T]
#     cte:       np.ndarray       # [T]
#     category:  str              # "straight" | "recurvature"
#     theta:     float            # total rotation angle (degrees)
#     csd_gt:    float            # CSD of ground-truth track
#     oyr_val:   float
#     hle_val:   float

#     # Tier 3 (populated when ensemble available)
#     crps:      Optional[np.ndarray] = None   # [T]
#     ssr:       Optional[np.ndarray] = None   # [T]

#     # Tier 4
#     dtw:       float = float("nan")

#     # Loss components (populated from training loop if passed)
#     loss_fm:      float = float("nan")
#     loss_dir:     float = float("nan")
#     loss_step:    float = float("nan")
#     loss_disp:    float = float("nan")
#     loss_heading: float = float("nan")
#     loss_smooth:  float = float("nan")
#     loss_pinn:    float = float("nan")
#     loss_total:   float = float("nan")


# # ══════════════════════════════════════════════════════════════════════════════
# #  7. Dataset-level aggregated metrics
# # ══════════════════════════════════════════════════════════════════════════════

# @dataclass
# class DatasetMetrics:
#     """
#     Aggregated 4-tier metrics for a full evaluation set.

#     Following the FM+PINN paper Table conventions:
#         Tier 1 : ADE, FDE, UGDE(h) at h∈{12,24,48,72}
#         Tier 2 : TSS, ATE, CTE, CSD/PR, RDR
#         Tier 3 : CRPS(h), SSR(h), BSS
#         Tier 4 : DTW, OYR, HLE
#     """
#     # ── Tier 1 ──────────────────────────────────────────────────────────
#     ade:            float = 0.0
#     fde:            float = 0.0
#     per_step_mean:  np.ndarray = field(default_factory=lambda: np.zeros(PRED_LEN))
#     per_step_std:   np.ndarray = field(default_factory=lambda: np.zeros(PRED_LEN))
#     ugde_12h:       float = 0.0
#     ugde_24h:       float = 0.0
#     ugde_48h:       float = 0.0
#     ugde_72h:       float = 0.0

#     # ── Tier 2 ──────────────────────────────────────────────────────────
#     tss_72h:        float = float("nan")    # vs CLIPER
#     ate_mean:       float = 0.0
#     cte_mean:       float = 0.0
#     ate_abs_mean:   float = 0.0
#     cte_abs_mean:   float = 0.0
#     ade_str:        float = float("nan")
#     ade_rec:        float = float("nan")
#     pr:             float = float("nan")    # ADE_rec / ADE_str
#     rdr:            float = float("nan")    # recurvature detection rate
#     n_str:          int   = 0
#     n_rec:          int   = 0

#     # ── Tier 3 ──────────────────────────────────────────────────────────
#     crps_mean:      float = float("nan")
#     crps_72h:       float = float("nan")
#     ssr_mean:       float = float("nan")
#     bss_mean:       float = float("nan")

#     # ── Tier 4 ──────────────────────────────────────────────────────────
#     dtw_mean:       float = float("nan")
#     dtw_str:        float = float("nan")
#     dtw_rec:        float = float("nan")
#     oyr_mean:       float = float("nan")
#     oyr_rec:        float = float("nan")
#     hle_mean:       float = float("nan")
#     hle_rec:        float = float("nan")

#     # ── Loss components (training / validation) ──────────────────────────
#     loss_fm:        float = float("nan")
#     loss_dir:       float = float("nan")
#     loss_step:      float = float("nan")
#     loss_disp:      float = float("nan")
#     loss_heading:   float = float("nan")
#     loss_smooth:    float = float("nan")
#     loss_pinn:      float = float("nan")
#     loss_total:     float = float("nan")

#     # ── Bookkeeping ──────────────────────────────────────────────────────
#     n_total:        int   = 0
#     timestamp:      str   = ""

#     def summary(self) -> str:
#         lines = [
#             "═" * 64,
#             "  FM+PINN TC Track Metrics  (4-tier)",
#             "═" * 64,
#             f"  Sequences : {self.n_total}"
#             f"  (str={self.n_str}, rec={self.n_rec})",
#             "",
#             "  ── Tier 1: Position ─────────────────────────────────",
#             f"  ADE        : {self.ade:.1f} km",
#             f"  FDE (72h)  : {self.fde:.1f} km",
#             f"  UGDE       : 12h={self.ugde_12h:.0f}  24h={self.ugde_24h:.0f}"
#             f"  48h={self.ugde_48h:.0f}  72h={self.ugde_72h:.0f} km",
#             "",
#             "  ── Tier 2: Operational ──────────────────────────────",
#             f"  TSS (72h)  : {self.tss_72h:.3f}"
#             + (" ✅" if not np.isnan(self.tss_72h) and self.tss_72h > 0 else ""),
#             f"  |ATE| mean : {self.ate_abs_mean:.1f} km",
#             f"  |CTE| mean : {self.cte_abs_mean:.1f} km",
#             f"  ADE_str    : {self.ade_str:.1f} km",
#             f"  ADE_rec    : {self.ade_rec:.1f} km",
#             f"  PR         : {self.pr:.2f}"
#             + (" (no bias)" if not np.isnan(self.pr) and self.pr < 1.3 else
#                " ⚠️ straight-track bias"),
#             f"  RDR        : {self.rdr:.3f}"
#             + (" ✅" if not np.isnan(self.rdr) and self.rdr > 0.5 else ""),
#             "",
#             "  ── Tier 3: Probabilistic ────────────────────────────",
#             f"  CRPS mean  : {self.crps_mean:.1f} km",
#             f"  CRPS 72h   : {self.crps_72h:.1f} km",
#             f"  SSR mean   : {self.ssr_mean:.3f}  (1=calibrated)",
#             f"  BSS mean   : {self.bss_mean:.3f}",
#             "",
#             "  ── Tier 4: Geometric ────────────────────────────────",
#             f"  DTW mean   : {self.dtw_mean:.1f} km",
#             f"  OYR mean   : {self.oyr_mean:.3f}  (0=ideal)",
#             f"  OYR rec    : {self.oyr_rec:.3f}",
#             f"  HLE mean   : {self.hle_mean:.4f}",
#             f"  HLE rec    : {self.hle_rec:.4f}",
#             "═" * 64,
#         ]
#         return "\n".join(lines)


# # ══════════════════════════════════════════════════════════════════════════════
# #  8. CSV export
# # ══════════════════════════════════════════════════════════════════════════════

# # Flat field order for CSV header
# _CSV_FIELDS = [
#     "timestamp", "n_total", "n_str", "n_rec",
#     # Tier 1
#     "ade", "fde", "ugde_12h", "ugde_24h", "ugde_48h", "ugde_72h",
#     # Tier 2
#     "tss_72h", "ate_abs_mean", "cte_abs_mean",
#     "ade_str", "ade_rec", "pr", "rdr",
#     # Tier 3
#     "crps_mean", "crps_72h", "ssr_mean", "bss_mean",
#     # Tier 4
#     "dtw_mean", "dtw_str", "dtw_rec",
#     "oyr_mean", "oyr_rec", "hle_mean", "hle_rec",
#     # Loss
#     "loss_total", "loss_fm", "loss_dir", "loss_step",
#     "loss_disp", "loss_heading", "loss_smooth", "loss_pinn",
# ]


# def save_metrics_csv(
#     metrics:  DatasetMetrics,
#     csv_path: str,
#     tag:      str = "",
# ) -> None:
#     """
#     Append one row of DatasetMetrics to a CSV file.
#     Creates the file with a header row if it does not yet exist.

#     Args
#     ----
#     metrics  : DatasetMetrics instance (fully populated)
#     csv_path : destination file path
#     tag      : optional label prepended to the timestamp field
#     """
#     os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
#     write_header = not os.path.exists(csv_path)

#     row: Dict[str, object] = {f: getattr(metrics, f, float("nan"))
#                                for f in _CSV_FIELDS}
#     if tag:
#         row["timestamp"] = f"{tag}_{metrics.timestamp}"

#     with open(csv_path, "a", newline="") as fh:
#         writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
#         if write_header:
#             writer.writeheader()
#         writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
#                          for k, v in row.items()})


# # ══════════════════════════════════════════════════════════════════════════════
# #  9. Main Evaluator
# # ══════════════════════════════════════════════════════════════════════════════

# class TCEvaluator:
#     """
#     Full 4-tier TC track evaluator.

#     Usage (sequence mode)
#     ----------------------
#     ev = TCEvaluator()
#     for pred, gt in zip(preds, gts):
#         ev.update(pred, gt)            # both [T, 2], 0.1° units
#     metrics = ev.compute()
#     print(metrics.summary())
#     ev.save_csv("results/eval.csv")

#     Usage (batch / tensor mode during training loop)
#     -------------------------------------------------
#     ev = TCEvaluator()
#     for batch in loader:
#         pred_norm = model.sample(batch)    # [T, B, 2]
#         gt_norm   = batch[1]               # [T, B, 2]
#         ev.update_batch(pred_norm, gt_norm)
#     """

#     def __init__(
#         self,
#         pred_len:    int   = PRED_LEN,
#         step_hours:  int   = STEP_HOURS,
#         recurv_thr:  float = RECURV_THR_DEG,
#         compute_dtw: bool  = True,
#         cliper_ugde: Optional[Dict[int, float]] = None,
#     ):
#         self.pred_len    = pred_len
#         self.step_hours  = step_hours
#         self.recurv_thr  = recurv_thr
#         self.compute_dtw = compute_dtw
#         self.cliper_ugde = cliper_ugde or {}
#         self._results:  List[SequenceResult] = []
#         self._loss_buf: List[Dict[str, float]] = []

#     def reset(self) -> None:
#         self._results  = []
#         self._loss_buf = []

#     # ── Single-sequence update ────────────────────────────────────────────────

#     def update(
#         self,
#         pred_01:    np.ndarray,
#         gt_01:      np.ndarray,
#         pred_ens:   Optional[np.ndarray] = None,  # [S, T, 2+]
#         loss_dict:  Optional[Dict[str, float]] = None,
#     ) -> None:
#         """
#         Evaluate one forecast sequence.

#         Args
#         ----
#         pred_01   : [T, 2+]   deterministic prediction (0.1° units)
#         gt_01     : [T, 2+]   ground truth (0.1° units)
#         pred_ens  : [S, T, 2+] ensemble (optional, for Tier 3)
#         loss_dict : {'fm', 'dir', 'step', 'disp', 'heading',
#                      'smooth', 'pinn', 'total'} from training loop
#         """
#         T = min(len(pred_01), len(gt_01), self.pred_len)
#         p = pred_01[:T]
#         g = gt_01[:T]

#         # Tier 1
#         _ade, _fde, ps = ade_fde(p, g)

#         # Tier 2
#         _ate, _cte = ate_cte(p, g)
#         theta      = total_rotation_angle(g)
#         cat        = "recurvature" if theta >= self.recurv_thr else "straight"
#         csd_g      = compute_csd(g)
#         _oyr       = oyr(p, g)
#         _hle       = hle(p, g)

#         # Tier 3
#         crps_arr: Optional[np.ndarray] = None
#         ssr_arr:  Optional[np.ndarray] = None
#         if pred_ens is not None:
#             S = pred_ens.shape[0]
#             crps_arr = np.array([
#                 crps_2d(pred_ens[:, h, :], g[h]) for h in range(T)
#             ])
#             ssr_arr = np.array([
#                 ssr_step(pred_ens[:, h, :], g[h]) for h in range(T)
#             ])

#         # Tier 4
#         dtw_val = dtw_haversine(p, g) if self.compute_dtw else float("nan")

#         r = SequenceResult(
#             ade=_ade, fde=_fde, per_step=ps,
#             ate=_ate, cte=_cte,
#             category=cat, theta=theta, csd_gt=csd_g,
#             oyr_val=_oyr, hle_val=_hle,
#             crps=crps_arr, ssr=ssr_arr, dtw=dtw_val,
#         )

#         if loss_dict:
#             r.loss_fm      = loss_dict.get("fm",      float("nan"))
#             r.loss_dir     = loss_dict.get("dir",     float("nan"))
#             r.loss_step    = loss_dict.get("step",    float("nan"))
#             r.loss_disp    = loss_dict.get("disp",    float("nan"))
#             r.loss_heading = loss_dict.get("heading", float("nan"))
#             r.loss_smooth  = loss_dict.get("smooth",  float("nan"))
#             r.loss_pinn    = loss_dict.get("pinn",    float("nan"))
#             r.loss_total   = loss_dict.get("total",   float("nan"))

#         self._results.append(r)

#     # ── Batch (tensor) update ────────────────────────────────────────────────

#     def update_batch(
#         self,
#         pred_norm,        # torch.Tensor [T, B, 2]  normalised
#         gt_norm,          # torch.Tensor [T, B, 2]  normalised
#         loss_dict: Optional[Dict[str, float]] = None,
#     ) -> None:
#         """Process a batch of normalised tensors (denorms internally)."""
#         pred_d = denorm_torch(pred_norm).cpu().numpy()   # [T, B, 2]
#         gt_d   = denorm_torch(gt_norm).cpu().numpy()
#         B = pred_d.shape[1]
#         for b in range(B):
#             self.update(pred_d[:, b, :], gt_d[:, b, :],
#                         loss_dict=loss_dict)

#     # ── Aggregate ────────────────────────────────────────────────────────────

#     def compute(self, tag: str = "") -> DatasetMetrics:
#         """Aggregate all sequence results into a DatasetMetrics instance."""
#         if not self._results:
#             return DatasetMetrics()

#         rs  = self._results
#         n   = len(rs)
#         ts  = tag or datetime.now().strftime("%Y%m%d_%H%M%S")

#         # ── Tier 1 ─────────────────────────────────────────────────────
#         all_steps = np.stack([r.per_step[:self.pred_len] for r in rs])
#         step_mean = all_steps.mean(0)
#         step_std  = all_steps.std(0)

#         def _h(step_idx):
#             return float(step_mean[step_idx]) if step_idx < self.pred_len else float("nan")

#         # ── Tier 2 ─────────────────────────────────────────────────────
#         str_r = [r for r in rs if r.category == "straight"]
#         rec_r = [r for r in rs if r.category == "recurvature"]

#         ade_s  = float(np.mean([r.ade for r in str_r])) if str_r else float("nan")
#         ade_r  = float(np.mean([r.ade for r in rec_r])) if rec_r else float("nan")
#         pr_val = ade_r / (ade_s + 1e-8) if (str_r and rec_r) else float("nan")

#         all_ate = np.concatenate([r.ate for r in rs])
#         all_cte = np.concatenate([r.cte for r in rs])

#         # TSS at 72h
#         tss_val = float("nan")
#         if self.cliper_ugde and 72 in self.cliper_ugde:
#             ugde_72 = _h(HORIZON_STEPS[72])
#             tss_val = tss(ugde_72, self.cliper_ugde[72])

#         # ── Tier 3 ─────────────────────────────────────────────────────
#         crps_seqs = [r.crps for r in rs if r.crps is not None]
#         ssr_seqs  = [r.ssr  for r in rs if r.ssr  is not None]

#         crps_mean = float("nan")
#         crps_72h  = float("nan")
#         ssr_mean  = float("nan")
#         if crps_seqs:
#             crps_mat = np.stack(crps_seqs)          # [N, T]
#             crps_mean = float(crps_mat.mean())
#             step_72 = HORIZON_STEPS.get(72, -1)
#             if 0 <= step_72 < crps_mat.shape[1]:
#                 crps_72h = float(crps_mat[:, step_72].mean())
#         if ssr_seqs:
#             ssr_mat  = np.stack(ssr_seqs)
#             ssr_mean = float(ssr_mat.mean())

#         # ── Tier 4 ─────────────────────────────────────────────────────
#         dtw_all = [r.dtw for r in rs  if not np.isnan(r.dtw)]
#         dtw_s   = [r.dtw for r in str_r if not np.isnan(r.dtw)]
#         dtw_rc  = [r.dtw for r in rec_r if not np.isnan(r.dtw)]

#         oyr_all = [r.oyr_val for r in rs]
#         oyr_rc  = [r.oyr_val for r in rec_r]
#         hle_all = [r.hle_val for r in rs  if not np.isnan(r.hle_val)]
#         hle_rc  = [r.hle_val for r in rec_r if not np.isnan(r.hle_val)]

#         # ── Loss ────────────────────────────────────────────────────────
#         def _mean_loss(attr):
#             vals = [getattr(r, attr) for r in rs if not np.isnan(getattr(r, attr))]
#             return float(np.mean(vals)) if vals else float("nan")

#         # ── RDR (requires CSD threshold from training set) ───────────────
#         # Use median CSD of current set as proxy if not provided externally
#         csd_vals = [r.csd_gt for r in rs]
#         tau = float(np.median(csd_vals)) if csd_vals else 0.0
#         rdr_num = sum(
#             1 for r in rec_r
#             if r.csd_gt >= tau
#         )
#         rdr_val = rdr_num / len(rec_r) if rec_r else float("nan")

#         m = DatasetMetrics(
#             # Tier 1
#             ade           = float(np.mean([r.ade for r in rs])),
#             fde           = float(np.mean([r.fde for r in rs])),
#             per_step_mean = step_mean,
#             per_step_std  = step_std,
#             ugde_12h      = _h(HORIZON_STEPS[12]),
#             ugde_24h      = _h(HORIZON_STEPS[24]),
#             ugde_48h      = _h(HORIZON_STEPS[48]),
#             ugde_72h      = _h(HORIZON_STEPS[72]),
#             # Tier 2
#             tss_72h       = tss_val,
#             ate_mean      = float(np.mean(all_ate)),
#             cte_mean      = float(np.mean(all_cte)),
#             ate_abs_mean  = float(np.mean(np.abs(all_ate))),
#             cte_abs_mean  = float(np.mean(np.abs(all_cte))),
#             ade_str       = ade_s,
#             ade_rec       = ade_r,
#             pr            = pr_val,
#             rdr           = rdr_val,
#             n_str         = len(str_r),
#             n_rec         = len(rec_r),
#             # Tier 3
#             crps_mean     = crps_mean,
#             crps_72h      = crps_72h,
#             ssr_mean      = ssr_mean,
#             bss_mean      = float("nan"),    # computed externally via brier_skill_score()
#             # Tier 4
#             dtw_mean      = float(np.mean(dtw_all)) if dtw_all else float("nan"),
#             dtw_str       = float(np.mean(dtw_s))   if dtw_s   else float("nan"),
#             dtw_rec       = float(np.mean(dtw_rc))  if dtw_rc  else float("nan"),
#             oyr_mean      = float(np.mean(oyr_all)) if oyr_all else float("nan"),
#             oyr_rec       = float(np.mean(oyr_rc))  if oyr_rc  else float("nan"),
#             hle_mean      = float(np.mean(hle_all)) if hle_all else float("nan"),
#             hle_rec       = float(np.mean(hle_rc))  if hle_rc  else float("nan"),
#             # Loss
#             loss_fm       = _mean_loss("loss_fm"),
#             loss_dir      = _mean_loss("loss_dir"),
#             loss_step     = _mean_loss("loss_step"),
#             loss_disp     = _mean_loss("loss_disp"),
#             loss_heading  = _mean_loss("loss_heading"),
#             loss_smooth   = _mean_loss("loss_smooth"),
#             loss_pinn     = _mean_loss("loss_pinn"),
#             loss_total    = _mean_loss("loss_total"),
#             # Meta
#             n_total       = n,
#             timestamp     = ts,
#         )
#         return m

#     def save_csv(self, csv_path: str, tag: str = "") -> None:
#         """Compute and immediately export to CSV."""
#         m = self.compute(tag=tag)
#         save_metrics_csv(m, csv_path, tag=tag)
#         print(f"  📊  Metrics → {csv_path}")


# # ══════════════════════════════════════════════════════════════════════════════
# #  10. Fast Tier-1 accumulator for training loops
# # ══════════════════════════════════════════════════════════════════════════════

# class StepErrorAccumulator:
#     """
#     Lightweight Haversine error accumulator for training / validation loops.
#     Avoids DTW / ensemble overhead during training.

#     Use TCEvaluator for full 4-tier evaluation at epoch end.
#     """

#     def __init__(self, pred_len: int = PRED_LEN, step_hours: int = STEP_HOURS):
#         self.pred_len   = pred_len
#         self.step_hours = step_hours
#         self.reset()

#     def reset(self) -> None:
#         self._sum    = np.zeros(self.pred_len, dtype=np.float64)
#         self._sum_sq = np.zeros(self.pred_len, dtype=np.float64)
#         self._count  = 0

#     def update(self, dist_km) -> None:
#         """
#         Args
#         ----
#         dist_km : torch.Tensor [T, B]  or np.ndarray [T, B]
#         """
#         if HAS_TORCH and torch.is_tensor(dist_km):
#             d = dist_km.double().cpu().numpy()
#         else:
#             d = np.asarray(dist_km, dtype=np.float64)

#         T, B = d.shape
#         self._sum    += d.sum(axis=1)
#         self._sum_sq += (d ** 2).sum(axis=1)
#         self._count  += B

#     def compute(self) -> Dict:
#         if self._count == 0:
#             return {}
#         ps  = self._sum / self._count
#         std = np.sqrt(np.maximum(self._sum_sq / self._count - ps ** 2, 0.0))
#         out: Dict = {
#             "per_step":     ps,
#             "per_step_std": std,
#             "ADE":          float(ps.mean()),
#             "FDE":          float(ps[-1]),
#             "n_samples":    self._count,
#         }
#         for h, s in HORIZON_STEPS.items():
#             if s < self.pred_len:
#                 out[f"{h}h"]     = float(ps[s])
#                 out[f"{h}h_std"] = float(std[s])
#         return out


# # ══════════════════════════════════════════════════════════════════════════════
# #  11. Backward-compatible wrappers
# # ══════════════════════════════════════════════════════════════════════════════

# def RSE(pred, true):
#     return float(np.sqrt(np.sum((true - pred) ** 2))
#                  / (np.sqrt(np.sum((true - true.mean()) ** 2)) + 1e-8))

# def CORR(pred, true):
#     u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
#     d = np.sqrt(
#         ((true - true.mean(0)) ** 2).sum(0)
#         * ((pred - pred.mean(0)) ** 2).sum(0)
#     ) + 1e-8
#     return u / d

# def MAE(pred, true):  return float(np.mean(np.abs(pred - true)))
# def MSE(pred, true):  return float(np.mean((pred - true) ** 2))
# def RMSE(pred, true): return float(np.sqrt(MSE(pred, true)))
# def MAPE(pred, true): return float(np.mean(np.abs((pred - true) / (np.abs(true) + 1e-5))))
# def MSPE(pred, true): return float(np.mean(np.square((pred - true) / (np.abs(true) + 1e-5))))

# def metric(pred, true):
#     return (MAE(pred, true), MSE(pred, true), RMSE(pred, true),
#             MAPE(pred, true), MSPE(pred, true),
#             RSE(pred, true),  CORR(pred, true))


# # ══════════════════════════════════════════════════════════════════════════════
# #  12. Self-test
# # ══════════════════════════════════════════════════════════════════════════════

# def _self_test():
#     np.random.seed(42)
#     T = 12
#     ev = TCEvaluator(pred_len=T, compute_dtw=True)

#     # 18 straight tracks
#     for _ in range(18):
#         gt   = np.zeros((T, 2))
#         gt[:, 0] = np.linspace(1300, 1250, T)
#         gt[:, 1] = np.linspace(150,  270,  T)
#         pred = gt + np.random.randn(T, 2) * 3.0
#         ev.update(pred, gt)

#     # 2 recurvature tracks
#     for _ in range(2):
#         gt   = np.zeros((T, 2))
#         gt[:, 0] = [1300,1290,1280,1270,1260,1255,1258,1265,1278,1295,1315,1335]
#         gt[:, 1] = [150, 160, 175, 192, 210, 228, 242, 255, 265, 270, 270, 268]
#         pred = gt + np.random.randn(T, 2) * 5.0
#         ev.update(pred, gt)

#     m = ev.compute(tag="selftest")
#     print(m.summary())
#     assert m.n_rec == 2,  f"Expected 2 recurvature, got {m.n_rec}"
#     assert m.n_str == 18, f"Expected 18 straight, got {m.n_str}"
#     print("\n✅ All assertions passed.")

#     # Test CSV export
#     ev.save_csv("/tmp/tc_metrics_selftest.csv", tag="selftest")
#     print("✅ CSV export OK → /tmp/tc_metrics_selftest.csv")


# if __name__ == "__main__":
#     _self_test()

"""
utils/metrics.py  ── v4
=============================
FIXES vs v3:

  FIX-MET-1  cliper_forecast exported at module level so train script
             can call it without reimporting. Also added denorm_deg_np
             export for train script use.

  FIX-MET-2  TCEvaluator.compute: TSS was nan because cliper_ugde was
             never populated by callers. Added: if cliper_ugde is set
             on the evaluator instance, TSS is computed; otherwise nan.
             The train script now sets ev.cliper_ugde before compute().

  FIX-MET-3  bss_mean left as nan in DatasetMetrics.compute() because
             brier_skill_score() was never called internally.
             Fix: bss_mean stays nan in compute() — it is computed
             externally in evaluate_full() where ensemble data is available.
             Added docstring note.

  FIX-MET-4  ssr_step: when all ensemble members identical (early training),
             spread=0 and rmse_em=0 → SSR=0/0=nan or 0.
             Added: return 1.0 (calibrated) when both spread and rmse are ~0,
             i.e. the model is trivially correct and calibrated.

  FIX-MET-5  crps_2d: O(S²) diversity loop replaced with vectorised
             pairwise haversine computation. Significant speedup at S=20.

  FIX-MET-6  DTW: compute_dtw default changed to True (was sometimes
             False in evaluate_full calls). The O(T²)=O(144) cost at T=12
             is negligible vs the 4664 sequences evaluation time.

  FIX-MET-7  LANDFALL_TARGETS and LANDFALL_RADIUS_KM exported at module
             level (were defined but brier_skill_score wasn't callable
             from train script without circular import).

Kept from v3: all Tier 1-4 computation, haversine, denorm, etc.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

# ── Physical constants ────────────────────────────────────────────────────────
R_EARTH_KM   = 6371.0
STEP_HOURS   = 6
PRED_LEN     = 12

CLIPER_ALPHA = {h: h / 12 for h in range(1, 13)}
HORIZON_STEPS: Dict[int, int] = {12: 1, 24: 3, 36: 5, 48: 7, 60: 9, 72: 11}
RECURV_THR_DEG = 45.0

# FIX-MET-7: exported for train script use
LANDFALL_TARGETS = [
    ("DaNang",    108.2, 16.1),
    ("Manila",    121.0, 14.6),
    ("Taipei",    121.5, 25.0),
    ("Shanghai",  121.5, 31.2),
    ("Okinawa",   127.8, 26.3),
]
LANDFALL_RADIUS_KM = 300.0


# ══════════════════════════════════════════════════════════════════════════════
#  1. Primitives
# ══════════════════════════════════════════════════════════════════════════════

def haversine_km(
    p1: np.ndarray,
    p2: np.ndarray,
    lon_idx: int = 0,
    lat_idx: int = 1,
    unit_01deg: bool = True,
) -> np.ndarray:
    scale = 10.0 if unit_01deg else 1.0
    lat1 = np.deg2rad(p1[..., lat_idx] / scale)
    lat2 = np.deg2rad(p2[..., lat_idx] / scale)
    dlon = np.deg2rad((p2[..., lon_idx] - p1[..., lon_idx]) / scale)
    dlat = np.deg2rad((p2[..., lat_idx] - p1[..., lat_idx]) / scale)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def haversine_km_torch(pred, gt, lon_idx: int = 0, lat_idx: int = 1,
                        unit_01deg: bool = True):
    scale = 10.0 if unit_01deg else 1.0
    lat1 = torch.deg2rad(gt[..., lat_idx]   / scale)
    lat2 = torch.deg2rad(pred[..., lat_idx] / scale)
    dlon = torch.deg2rad((pred[..., lon_idx] - gt[..., lon_idx]) / scale)
    dlat = torch.deg2rad((pred[..., lat_idx] - gt[..., lat_idx]) / scale)
    a = (torch.sin(dlat / 2.0) ** 2
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2)
    return 2.0 * R_EARTH_KM * torch.asin(a.clamp(0.0, 1.0).sqrt())


def denorm_np(n: np.ndarray) -> np.ndarray:
    """Normalised coords → 0.1° units (NumPy)."""
    r = n.copy()
    r[..., 0] = n[..., 0] * 50.0 + 1800.0
    r[..., 1] = n[..., 1] * 50.0
    return r


def denorm_torch(n):
    """Normalised coords → 0.1° units (PyTorch)."""
    r = n.clone()
    r[..., 0] = n[..., 0] * 50.0 + 1800.0
    r[..., 1] = n[..., 1] * 50.0
    return r


# FIX-MET-1: exported denorm_deg_np for train script
def denorm_deg_np(arr_norm: np.ndarray) -> np.ndarray:
    """Normalised coords → degrees."""
    out = arr_norm.copy()
    out[..., 0] = (arr_norm[..., 0] * 50.0 + 1800.0) / 10.0
    out[..., 1] = (arr_norm[..., 1] * 50.0) / 10.0
    return out


# ── FIX-MET-1: cliper_forecast exported ──────────────────────────────────────

# def cliper_forecast(obs_01: np.ndarray, h: int) -> np.ndarray:
#     """
#     Simple CLIPER-WNP: linear extrapolation from last two observed steps.
#     Returns predicted position (0.1° units) at lead step h (1-indexed).

#     FIX-MET-1: exported at module level for use in train script.
#     """
#     if obs_01.shape[0] < 2:
#         return obs_01[-1].copy()
#     v = obs_01[-1] - obs_01[-2]
#     return obs_01[-1] + h * v
def cliper_forecast(obs_01: np.ndarray, h: int) -> np.ndarray:
    """
    Dự báo CLIPER chuẩn xác bằng cách đưa về không gian Degrees.
    obs_01: Tọa độ đã chuẩn hóa (normalized) từ DataLoader.
    h: Bước dự báo (1, 2, ..., 12).
    """
    # 1. Giải chuẩn hóa về độ thực tế (ví dụ: 112.5, 16.1)
    # Hàm denorm_deg_np phải được định nghĩa đúng theo công thức của Dataset
    obs_deg = denorm_deg_np(obs_01) 

    if obs_deg.shape[0] < 2:
        return obs_01[-1].copy() * 10.0 # Fallback
    
    # 2. Tính vận tốc dựa trên sự thay đổi độ (Degrees per step)
    # v = (v_lon, v_lat)
    v = obs_deg[-1] - obs_deg[-2]
    
    # 3. Dự báo tuyến tính cho bước thứ h
    pred_deg = obs_deg[-1] + (h * v)
    
    # 4. Trả về đơn vị 0.1 degree (ví dụ: 1125, 161) 
    # để TCEvaluator tính Haversine ra km chính xác.
    return pred_deg * 10.0


# ══════════════════════════════════════════════════════════════════════════════
#  2. Tier 1
# ══════════════════════════════════════════════════════════════════════════════

def ugde(pred_01: np.ndarray, gt_01: np.ndarray) -> np.ndarray:
    return haversine_km(pred_01, gt_01)


def ade_fde(pred_01: np.ndarray, gt_01: np.ndarray
            ) -> Tuple[float, float, np.ndarray]:
    ps = ugde(pred_01, gt_01)
    return float(ps.mean()), float(ps[-1]), ps


# ══════════════════════════════════════════════════════════════════════════════
#  3. Tier 2
# ══════════════════════════════════════════════════════════════════════════════

def tss(ugde_model: float, ugde_cliper: float) -> float:
    return 1.0 - ugde_model / (ugde_cliper + 1e-8)


def lat_corrected_velocity(traj_01: np.ndarray,
                            lon_idx: int = 0,
                            lat_idx: int = 1) -> np.ndarray:
    lats_rad = np.deg2rad(traj_01[:, lat_idx] / 10.0)
    cos_lat  = np.cos(lats_rad[:-1])
    dlat = np.diff(traj_01[:, lat_idx])
    dlon = np.diff(traj_01[:, lon_idx]) * cos_lat
    return np.stack([dlon, dlat], axis=-1)


def total_rotation_angle(gt_01: np.ndarray) -> float:
    if gt_01.shape[0] < 3:
        return 0.0
    v = lat_corrected_velocity(gt_01)
    total = 0.0
    for i in range(len(v) - 1):
        v1, v2 = v[i], v[i + 1]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        total += np.degrees(np.arccos(cos_a))
    return total


def classify(gt_01: np.ndarray, thr: float = RECURV_THR_DEG) -> str:
    return "recurvature" if total_rotation_angle(gt_01) >= thr else "straight"


# def ate_cte(pred_01: np.ndarray, gt_01: np.ndarray,
#             lon_idx: int = 0, lat_idx: int = 1
#             ) -> Tuple[np.ndarray, np.ndarray]:
#     T = pred_01.shape[0]
#     ate_arr = np.zeros(T)
#     cte_arr = np.zeros(T)
#     for k in range(T):
#         if k == 0:
#             dk = gt_01[1] - gt_01[0] if T > 1 else np.array([1.0, 0.0])
#         else:
#             dk = gt_01[k] - gt_01[k - 1]
#         norm_dk = np.linalg.norm(dk)
#         if norm_dk < 1e-8:
#             continue
#         t_hat = dk / norm_dk
#         n_hat = np.array([-t_hat[1], t_hat[0]])
#         delta = pred_01[k] - gt_01[k]
        
#         # Lấy vĩ độ thực tế để tính hệ số km
#         lat_deg = gt_01[k, 1] / 10.0
#         km_per_01deg = 11.11 # 111.1 / 10
        
#         # Chuyển đổi delta sang km (đơn giản hóa cục bộ)
#         delta_km = delta * km_per_01deg
#         delta_km[0] *= np.cos(np.deg2rad(lat_deg)) # Bù trừ kinh độ theo vĩ độ
        
#         ate_arr[k] = float(np.dot(delta_km, t_hat))
#         cte_arr[k] = float(np.dot(delta_km, n_hat))
#     return ate_arr, cte_arr
def ate_cte(pred_01: np.ndarray, gt_01: np.ndarray,
            lon_idx: int = 0, lat_idx: int = 1
            ) -> Tuple[np.ndarray, np.ndarray]:
    T = pred_01.shape[0]
    ate_arr = np.zeros(T)
    cte_arr = np.zeros(T)
    
    km_per_01deg = 11.112 # Quy đổi 0.1 độ sang km

    for k in range(T):
        # 1. Tính vector dịch chuyển của Ground Truth để làm hệ trục tọa độ
        if k == 0:
            dk = gt_01[1] - gt_01[0] if T > 1 else np.array([1.0, 0.0])
        else:
            dk = gt_01[k] - gt_01[k - 1]
        
        # Bù trừ vĩ độ cho vector hướng
        lat_rad = np.deg2rad(gt_01[k, lat_idx] / 10.0)
        dk_km = dk * km_per_01deg
        dk_km[0] *= np.cos(lat_rad)
        
        norm_dk = np.linalg.norm(dk_km)
        if norm_dk < 1e-8:
            continue
            
        t_hat = dk_km / norm_dk # Vector hướng đi (Along-track)
        n_hat = np.array([-t_hat[1], t_hat[0]]) # Vector chệch hướng (Cross-track)

        # 2. Tính sai số vị trí thực tế bằng km
        delta_pos = pred_01[k] - gt_01[k]
        delta_km = delta_pos * km_per_01deg
        delta_km[0] *= np.cos(lat_rad) # Bù trừ kinh độ

        # 3. Chiếu sai số lên hệ trục ATE/CTE
        ate_arr[k] = float(np.dot(delta_km, t_hat))
        cte_arr[k] = float(np.dot(delta_km, n_hat))

    return ate_arr, cte_arr

def circular_std(angles_deg: np.ndarray) -> float:
    if len(angles_deg) == 0:
        return 0.0
    rads  = np.deg2rad(angles_deg)
    R_bar = np.abs(np.mean(np.exp(1j * rads)))
    R_bar = np.clip(R_bar, 1e-8, 1.0 - 1e-8)
    return float(np.degrees(np.sqrt(-2.0 * np.log(R_bar))))


def compute_csd(traj_01: np.ndarray) -> float:
    v = lat_corrected_velocity(traj_01)
    if len(v) == 0:
        return 0.0
    angles = np.arctan2(v[:, 1], v[:, 0])
    return circular_std(np.degrees(angles))


def oyr(pred_01: np.ndarray, gt_01: np.ndarray) -> float:
    pv = lat_corrected_velocity(pred_01)
    gv = lat_corrected_velocity(gt_01)
    m  = min(len(pv), len(gv))
    if m == 0:
        return 0.0
    dots = np.sum(pv[:m] * gv[:m], axis=1)
    np_  = np.linalg.norm(pv[:m], axis=1)
    ng_  = np.linalg.norm(gv[:m], axis=1)
    valid = (np_ > 1e-8) & (ng_ > 1e-8)
    if valid.sum() == 0:
        return 0.0
    cos_vals = dots[valid] / (np_[valid] * ng_[valid])
    return float((cos_vals < 0).mean())


def hle(pred_01: np.ndarray, gt_01: np.ndarray) -> float:
    def curvature(traj):
        v = lat_corrected_velocity(traj)
        if len(v) < 2:
            return np.array([])
        cross = v[1:, 0] * v[:-1, 1] - v[1:, 1] * v[:-1, 0]
        n1    = np.linalg.norm(v[1:],  axis=1).clip(1e-8)
        n2    = np.linalg.norm(v[:-1], axis=1).clip(1e-8)
        return cross / (n1 * n2)
    kp = curvature(pred_01)
    kg = curvature(gt_01)
    m  = min(len(kp), len(kg))
    if m == 0:
        return float("nan")
    return float(np.mean(np.abs(kp[:m] - kg[:m])))


# ══════════════════════════════════════════════════════════════════════════════
#  4. Tier 3 — Probabilistic
# ══════════════════════════════════════════════════════════════════════════════

def crps_2d(pred_ens_01: np.ndarray, gt_01: np.ndarray) -> float:
    """
    CRPS energy-form.
    FIX-MET-5: vectorised pairwise diversity (was O(S²) loop).
    pred_ens_01: [S, 2+]
    gt_01:       [2+]
    """
    S = pred_ens_01.shape[0]
    gt_rep  = gt_01[np.newaxis].repeat(S, axis=0)
    acc     = float(np.mean(haversine_km(pred_ens_01, gt_rep)))

    # Vectorised pairwise: [S, S] distance matrix
    # Use broadcasting: expand [S, 1, 2] vs [1, S, 2]
    p_i = pred_ens_01[:, np.newaxis, :]   # [S, 1, 2]
    p_j = pred_ens_01[np.newaxis, :, :]   # [1, S, 2]
    # Flatten to [S*S, 2] for haversine_km
    p_i_flat = np.broadcast_to(p_i, (S, S, pred_ens_01.shape[-1])).reshape(S * S, -1)
    p_j_flat = np.broadcast_to(p_j, (S, S, pred_ens_01.shape[-1])).reshape(S * S, -1)
    div = float(np.mean(haversine_km(p_i_flat, p_j_flat)))

    return float(acc - 0.5 * div)


def ssr_step(pred_ens_01: np.ndarray, gt_01: np.ndarray) -> float:
    """
    Spread-Skill Ratio.
    FIX-MET-4: return 1.0 (calibrated) when both spread and rmse are ~0.
    """
    S = pred_ens_01.shape[0]
    ens_mean = pred_ens_01.mean(axis=0)
    ens_mean_rep = ens_mean[np.newaxis].repeat(S, axis=0)
    spread_vals  = haversine_km(pred_ens_01, ens_mean_rep)
    spread       = float(np.sqrt(np.mean(spread_vals ** 2)))
    rmse_em = float(haversine_km(ens_mean[np.newaxis], gt_01[np.newaxis])[0])

    # FIX-MET-4: both near zero → perfectly calibrated
    if spread < 1e-6 and rmse_em < 1e-6:
        return 1.0
    return spread / (rmse_em + 1e-8)


def brier_skill_score(
    pred_ens_seqs: List[np.ndarray],
    gt_seqs:       List[np.ndarray],
    step:          int,
    target_deg:    Tuple[float, float],
    radius_km:     float = LANDFALL_RADIUS_KM,
    clim_rate:     Optional[float] = None,
) -> float:
    """
    Brier Skill Score for landfall strike probability.
    pred_ens_seqs: list of [S, T, 2] in 0.1° units
    gt_seqs:       list of [T, 2] in 0.1° units
    """
    N   = len(gt_seqs)
    if N == 0:
        return float("nan")
    bs  = 0.0
    clim_hits = 0

    target_01 = np.array([target_deg[0] * 10.0, target_deg[1] * 10.0])

    for i in range(N):
        if step >= len(gt_seqs[i]):
            continue
        gt_pos   = gt_seqs[i][step]
        ens_seqs = pred_ens_seqs[i]     # [S, T, 2]
        if ens_seqs.ndim == 3:
            S = ens_seqs.shape[0]
        else:
            continue

        obs = float(haversine_km(
            gt_pos[np.newaxis], target_01[np.newaxis])[0]) <= radius_km

        hits = sum(
            haversine_km(ens_seqs[s, min(step, ens_seqs.shape[1]-1)][np.newaxis],
                         target_01[np.newaxis])[0] <= radius_km
            for s in range(S)
        )
        p_fc = hits / S
        bs  += (p_fc - float(obs)) ** 2
        clim_hits += float(obs)

    bs /= max(N, 1)
    p_clim = (clim_rate if clim_rate is not None else clim_hits / max(N, 1))
    bs_clim = p_clim * (1.0 - p_clim) + 1e-8
    return float(1.0 - bs / bs_clim)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Tier 4 — Geometric
# ══════════════════════════════════════════════════════════════════════════════

def dtw_haversine(s: np.ndarray, t: np.ndarray) -> float:
    n, m = len(s), len(t)
    dp   = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(haversine_km(s[i-1:i], t[j-1:j])[0])
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return float(dp[n, m])


# ══════════════════════════════════════════════════════════════════════════════
#  6. Per-sequence result container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SequenceResult:
    ade:       float
    fde:       float
    per_step:  np.ndarray

    ate:       np.ndarray
    cte:       np.ndarray
    category:  str
    category_pred: str
    theta:     float
    csd_gt:    float
    oyr_val:   float
    hle_val:   float

    crps:      Optional[np.ndarray] = None
    ssr:       Optional[np.ndarray] = None
    dtw:       float = float("nan")

    loss_fm:      float = float("nan")
    loss_dir:     float = float("nan")
    loss_step:    float = float("nan")
    loss_disp:    float = float("nan")
    loss_heading: float = float("nan")
    loss_smooth:  float = float("nan")
    loss_pinn:    float = float("nan")
    loss_total:   float = float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  7. Dataset-level aggregated metrics
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetMetrics:
    ade:            float = 0.0
    fde:            float = 0.0
    per_step_mean:  np.ndarray = field(default_factory=lambda: np.zeros(PRED_LEN))
    per_step_std:   np.ndarray = field(default_factory=lambda: np.zeros(PRED_LEN))
    ugde_12h:       float = 0.0
    ugde_24h:       float = 0.0
    ugde_48h:       float = 0.0
    ugde_72h:       float = 0.0

    tss_72h:        float = float("nan")
    ate_mean:       float = 0.0
    cte_mean:       float = 0.0
    ate_abs_mean:   float = 0.0
    cte_abs_mean:   float = 0.0
    ade_str:        float = float("nan")
    ade_rec:        float = float("nan")
    pr:             float = float("nan")
    rdr:            float = float("nan")
    n_str:          int   = 0
    n_rec:          int   = 0

    crps_mean:      float = float("nan")
    crps_72h:       float = float("nan")
    ssr_mean:       float = float("nan")
    # FIX-MET-3: bss_mean computed externally in evaluate_full
    bss_mean:       float = float("nan")

    dtw_mean:       float = float("nan")
    dtw_str:        float = float("nan")
    dtw_rec:        float = float("nan")
    oyr_mean:       float = float("nan")
    oyr_rec:        float = float("nan")
    hle_mean:       float = float("nan")
    hle_rec:        float = float("nan")

    loss_fm:        float = float("nan")
    loss_dir:       float = float("nan")
    loss_step:      float = float("nan")
    loss_disp:      float = float("nan")
    loss_heading:   float = float("nan")
    loss_smooth:    float = float("nan")
    loss_pinn:      float = float("nan")
    loss_total:     float = float("nan")

    n_total:        int   = 0
    timestamp:      str   = ""

    def summary(self) -> str:
        lines = [
            "═" * 64,
            "  FM+PINN TC Track Metrics  (4-tier)",
            "═" * 64,
            f"  Sequences : {self.n_total}"
            f"  (str={self.n_str}, rec={self.n_rec})",
            "",
            "  ── Tier 1: Position ─────────────────────────────────",
            f"  ADE        : {self.ade:.1f} km",
            f"  FDE (72h)  : {self.fde:.1f} km",
            f"  UGDE       : 12h={self.ugde_12h:.0f}  24h={self.ugde_24h:.0f}"
            f"  48h={self.ugde_48h:.0f}  72h={self.ugde_72h:.0f} km",
            "",
            "  ── Tier 2: Operational ──────────────────────────────",
            f"  TSS (72h)  : {self.tss_72h:.3f}" +
            (" ✅" if not np.isnan(self.tss_72h) and self.tss_72h > 0 else ""),
            f"  |ATE| mean : {self.ate_abs_mean:.1f} km",
            f"  |CTE| mean : {self.cte_abs_mean:.1f} km",
            f"  ADE_str    : {self.ade_str:.1f} km",
            f"  ADE_rec    : {self.ade_rec:.1f} km",
            f"  PR         : {self.pr:.2f}" +
            (" (no bias)" if not np.isnan(self.pr) and self.pr < 1.3 else
             " ⚠️ straight-track bias"),
            f"  RDR        : {self.rdr:.3f}" +
            (" ✅" if not np.isnan(self.rdr) and self.rdr > 0.5 else ""),
            "",
            "  ── Tier 3: Probabilistic ────────────────────────────",
            f"  CRPS mean  : {self.crps_mean:.1f} km",
            f"  CRPS 72h   : {self.crps_72h:.1f} km",
            f"  SSR mean   : {self.ssr_mean:.3f}  (1=calibrated)",
            f"  BSS mean   : {self.bss_mean:.3f}",
            "",
            "  ── Tier 4: Geometric ────────────────────────────────",
            f"  DTW mean   : {self.dtw_mean:.1f} km",
            f"  OYR mean   : {self.oyr_mean:.3f}  (0=ideal)",
            f"  OYR rec    : {self.oyr_rec:.3f}",
            f"  HLE mean   : {self.hle_mean:.4f}",
            f"  HLE rec    : {self.hle_rec:.4f}",
            "═" * 64,
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  8. CSV export
# ══════════════════════════════════════════════════════════════════════════════

_CSV_FIELDS = [
    "timestamp", "n_total", "n_str", "n_rec",
    "ade", "fde", "ugde_12h", "ugde_24h", "ugde_48h", "ugde_72h",
    "tss_72h", "ate_abs_mean", "cte_abs_mean",
    "ade_str", "ade_rec", "pr", "rdr",
    "crps_mean", "crps_72h", "ssr_mean", "bss_mean",
    "dtw_mean", "dtw_str", "dtw_rec",
    "oyr_mean", "oyr_rec", "hle_mean", "hle_rec",
    "loss_total", "loss_fm", "loss_dir", "loss_step",
    "loss_disp", "loss_heading", "loss_smooth", "loss_pinn",
]


def save_metrics_csv(metrics: DatasetMetrics, csv_path: str, tag: str = "") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    row: Dict[str, object] = {f: getattr(metrics, f, float("nan"))
                               for f in _CSV_FIELDS}
    if tag:
        row["timestamp"] = f"{tag}_{metrics.timestamp}"
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                         for k, v in row.items()})


# ══════════════════════════════════════════════════════════════════════════════
#  9. Main Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class TCEvaluator:
    """
    Full 4-tier TC track evaluator.

    FIX-MET-2: TSS computed when cliper_ugde is set on instance.
    FIX-MET-6: compute_dtw=True by default.
    """

    def __init__(
        self,
        pred_len:    int   = PRED_LEN,
        step_hours:  int   = STEP_HOURS,
        recurv_thr:  float = RECURV_THR_DEG,
        compute_dtw: bool  = True,    # FIX-MET-6: default True
        cliper_ugde: Optional[Dict[int, float]] = None,
    ):
        self.pred_len    = pred_len
        self.step_hours  = step_hours
        self.recurv_thr  = recurv_thr
        self.compute_dtw = compute_dtw
        self.cliper_ugde = cliper_ugde or {}
        self._results:  List[SequenceResult] = []
        self._loss_buf: List[Dict[str, float]] = []

    def reset(self) -> None:
        self._results  = []
        self._loss_buf = []

    def update(
        self,
        pred_01:    np.ndarray,
        gt_01:      np.ndarray,
        pred_ens:   Optional[np.ndarray] = None,
        loss_dict:  Optional[Dict[str, float]] = None,
    ) -> None:
        T = min(len(pred_01), len(gt_01), self.pred_len)
        p = pred_01[:T]
        g = gt_01[:T]

        _ade, _fde, ps = ade_fde(p, g)
        _ate, _cte = ate_cte(p, g)
        theta      = total_rotation_angle(g)
        cat        = "recurvature" if theta >= self.recurv_thr else "straight"
           # --- THÊM LOGIC NÀY: Nhãn của Model dự báo ---
        theta_pred = total_rotation_angle(p)
        cat_pred   = "recurvature" if theta_pred >= self.recurv_thr else "straight"
        # --------------------------------------------

        csd_g      = compute_csd(g)
        _oyr       = oyr(p, g)
        _hle       = hle(p, g)

        crps_arr: Optional[np.ndarray] = None
        ssr_arr:  Optional[np.ndarray] = None
        if pred_ens is not None:
            crps_arr = np.array([
                crps_2d(pred_ens[:, h, :] if pred_ens.ndim == 3 else pred_ens[:, h, :],
                        g[h]) for h in range(T)
            ])
            ssr_arr = np.array([
                ssr_step(pred_ens[:, h, :] if pred_ens.ndim == 3 else pred_ens[:, h, :],
                         g[h]) for h in range(T)
            ])

        # dtw_val = dtw_haversine(p, g) if self.compute_dtw else float("nan")

        # r = SequenceResult(
        #     ade=_ade, fde=_fde, per_step=ps,
        #     ate=_ate, cte=_cte,
        #     category=cat, theta=theta, csd_gt=csd_g,
        #     oyr_val=_oyr, hle_val=_hle,
        #     crps=crps_arr, ssr=ssr_arr, dtw=dtw_val,
        # )

        dtw_val = dtw_haversine(p, g) if self.compute_dtw else float("nan")

        # CẬP NHẬT ĐOẠN KHỞI TẠO r
        r = SequenceResult(
            ade=_ade, fde=_fde, per_step=ps,
            ate=_ate, cte=_cte,
            category=cat,           # Nhãn thật
            category_pred=cat_pred, # Nhãn dự báo (MỚI THÊM)
            theta=theta, csd_gt=csd_g,
            oyr_val=_oyr, hle_val=_hle,
            crps=crps_arr, ssr=ssr_arr, dtw=dtw_val,
        )
        if loss_dict:
            r.loss_fm      = loss_dict.get("fm",      float("nan"))
            r.loss_dir     = loss_dict.get("dir",     float("nan"))
            r.loss_step    = loss_dict.get("step",    float("nan"))
            r.loss_disp    = loss_dict.get("disp",    float("nan"))
            r.loss_heading = loss_dict.get("heading", float("nan"))
            r.loss_smooth  = loss_dict.get("smooth",  float("nan"))
            r.loss_pinn    = loss_dict.get("pinn",    float("nan"))
            r.loss_total   = loss_dict.get("total",   float("nan"))

        self._results.append(r)

    def update_batch(self, pred_norm, gt_norm,
                     loss_dict: Optional[Dict[str, float]] = None) -> None:
        pred_d = denorm_torch(pred_norm).cpu().numpy()
        gt_d   = denorm_torch(gt_norm).cpu().numpy()
        B = pred_d.shape[1]
        for b in range(B):
            self.update(pred_d[:, b, :], gt_d[:, b, :], loss_dict=loss_dict)

    def compute(self, tag: str = "") -> DatasetMetrics:
        if not self._results:
            return DatasetMetrics()

        rs  = self._results
        n   = len(rs)
        ts  = tag or datetime.now().strftime("%Y%m%d_%H%M%S")

        all_steps = np.stack([r.per_step[:self.pred_len] for r in rs])
        step_mean = all_steps.mean(0)
        step_std  = all_steps.std(0)

        def _h(step_idx):
            return float(step_mean[step_idx]) if step_idx < self.pred_len else float("nan")

        str_r = [r for r in rs if r.category == "straight"]
        rec_r = [r for r in rs if r.category == "recurvature"]

        ade_s  = float(np.mean([r.ade for r in str_r])) if str_r else float("nan")
        ade_r  = float(np.mean([r.ade for r in rec_r])) if rec_r else float("nan")
        pr_val = ade_r / (ade_s + 1e-8) if (str_r and rec_r) else float("nan")

        all_ate = np.concatenate([r.ate for r in rs])
        all_cte = np.concatenate([r.cte for r in rs])

        # FIX-MET-2: TSS computed if cliper_ugde is set
        tss_val = float("nan")
        if self.cliper_ugde and 72 in self.cliper_ugde:
            ugde_72 = _h(HORIZON_STEPS[72])
            if not np.isnan(ugde_72):
                tss_val = tss(ugde_72, self.cliper_ugde[72])

        crps_seqs = [r.crps for r in rs if r.crps is not None]
        ssr_seqs  = [r.ssr  for r in rs if r.ssr  is not None]

        crps_mean = float("nan")
        crps_72h  = float("nan")
        ssr_mean  = float("nan")
        if crps_seqs:
            crps_mat = np.stack(crps_seqs)
            crps_mean = float(crps_mat.mean())
            step_72 = HORIZON_STEPS.get(72, -1)
            if 0 <= step_72 < crps_mat.shape[1]:
                crps_72h = float(crps_mat[:, step_72].mean())
        if ssr_seqs:
            ssr_mat  = np.stack(ssr_seqs)
            # FIX-MET-4: filter out nan SSR values before averaging
            valid_ssr = ssr_mat[~np.isnan(ssr_mat)]
            ssr_mean = float(valid_ssr.mean()) if len(valid_ssr) > 0 else float("nan")

        dtw_all = [r.dtw for r in rs  if not np.isnan(r.dtw)]
        dtw_s   = [r.dtw for r in str_r if not np.isnan(r.dtw)]
        dtw_rc  = [r.dtw for r in rec_r if not np.isnan(r.dtw)]

        oyr_all = [r.oyr_val for r in rs]
        oyr_rc  = [r.oyr_val for r in rec_r]
        hle_all = [r.hle_val for r in rs  if not np.isnan(r.hle_val)]
        hle_rc  = [r.hle_val for r in rec_r if not np.isnan(r.hle_val)]

        def _mean_loss(attr):
            vals = [getattr(r, attr) for r in rs if not np.isnan(getattr(r, attr))]
            return float(np.mean(vals)) if vals else float("nan")

        csd_vals = [r.csd_gt for r in rs]
        tau = float(np.median(csd_vals)) if csd_vals else 0.0
        # rdr_num = sum(1 for r in rec_r if r.csd_gt >= tau)
        # rdr_val = rdr_num / len(rec_r) if rec_r else float("nan")

        # ── RDR (Recurvature Detection Rate - Tỷ lệ phát hiện quay đầu) ──
        # rec_r là danh sách các ca thực tế là quay đầu (Ground Truth = recurvature)
        if rec_r:
            # Đếm xem trong những ca quay đầu thật, model đoán đúng bao nhiêu ca
            rdr_num = sum(1 for r in rec_r if r.category_pred == "recurvature")
            rdr_val = rdr_num / len(rec_r)
        else:
            rdr_val = float("nan")
            
        # Sau đó gán rdr_val vào DatasetMetrics ở bên dưới
        m = DatasetMetrics(
            ade           = float(np.mean([r.ade for r in rs])),
            fde           = float(np.mean([r.fde for r in rs])),
            per_step_mean = step_mean,
            per_step_std  = step_std,
            ugde_12h      = _h(HORIZON_STEPS[12]),
            ugde_24h      = _h(HORIZON_STEPS[24]),
            ugde_48h      = _h(HORIZON_STEPS[48]),
            ugde_72h      = _h(HORIZON_STEPS[72]),
            tss_72h       = tss_val,
            ate_mean      = float(np.mean(all_ate)),
            cte_mean      = float(np.mean(all_cte)),
            ate_abs_mean  = float(np.mean(np.abs(all_ate))),
            cte_abs_mean  = float(np.mean(np.abs(all_cte))),
            ade_str       = ade_s,
            ade_rec       = ade_r,
            pr            = pr_val,
            rdr           = rdr_val,
            n_str         = len(str_r),
            n_rec         = len(rec_r),
            crps_mean     = crps_mean,
            crps_72h      = crps_72h,
            ssr_mean      = ssr_mean,
            bss_mean      = float("nan"),   # set externally by evaluate_full
            dtw_mean      = float(np.mean(dtw_all)) if dtw_all else float("nan"),
            dtw_str       = float(np.mean(dtw_s))   if dtw_s   else float("nan"),
            dtw_rec       = float(np.mean(dtw_rc))  if dtw_rc  else float("nan"),
            oyr_mean      = float(np.mean(oyr_all)) if oyr_all else float("nan"),
            oyr_rec       = float(np.mean(oyr_rc))  if oyr_rc  else float("nan"),
            hle_mean      = float(np.mean(hle_all)) if hle_all else float("nan"),
            hle_rec       = float(np.mean(hle_rc))  if hle_rc  else float("nan"),
            loss_fm       = _mean_loss("loss_fm"),
            loss_dir      = _mean_loss("loss_dir"),
            loss_step     = _mean_loss("loss_step"),
            loss_disp     = _mean_loss("loss_disp"),
            loss_heading  = _mean_loss("loss_heading"),
            loss_smooth   = _mean_loss("loss_smooth"),
            loss_pinn     = _mean_loss("loss_pinn"),
            loss_total    = _mean_loss("loss_total"),
            n_total       = n,
            timestamp     = ts,
        )
        return m

    def save_csv(self, csv_path: str, tag: str = "") -> None:
        m = self.compute(tag=tag)
        save_metrics_csv(m, csv_path, tag=tag)
        print(f"  📊  Metrics → {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  10. Fast Tier-1 accumulator
# ══════════════════════════════════════════════════════════════════════════════

class StepErrorAccumulator:
    def __init__(self, pred_len: int = PRED_LEN, step_hours: int = STEP_HOURS):
        self.pred_len   = pred_len
        self.step_hours = step_hours
        self.reset()

    def reset(self) -> None:
        self._sum    = np.zeros(self.pred_len, dtype=np.float64)
        self._sum_sq = np.zeros(self.pred_len, dtype=np.float64)
        self._count  = 0

    def update(self, dist_km) -> None:
        if HAS_TORCH and torch.is_tensor(dist_km):
            d = dist_km.double().cpu().numpy()
        else:
            d = np.asarray(dist_km, dtype=np.float64)
        T, B = d.shape
        self._sum    += d.sum(axis=1)
        self._sum_sq += (d ** 2).sum(axis=1)
        self._count  += B

    def compute(self) -> Dict:
        if self._count == 0:
            return {}
        ps  = self._sum / self._count
        std = np.sqrt(np.maximum(self._sum_sq / self._count - ps ** 2, 0.0))
        out: Dict = {
            "per_step":     ps,
            "per_step_std": std,
            "ADE":          float(ps.mean()),
            "FDE":          float(ps[-1]),
            "n_samples":    self._count,
        }
        for h, s in HORIZON_STEPS.items():
            if s < self.pred_len:
                out[f"{h}h"]     = float(ps[s])
                out[f"{h}h_std"] = float(std[s])
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  11. Backward-compatible wrappers
# ══════════════════════════════════════════════════════════════════════════════

def RSE(pred, true):
    return float(np.sqrt(np.sum((true - pred) ** 2))
                 / (np.sqrt(np.sum((true - true.mean()) ** 2)) + 1e-8))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(
        ((true - true.mean(0)) ** 2).sum(0)
        * ((pred - pred.mean(0)) ** 2).sum(0)
    ) + 1e-8
    return u / d

def MAE(pred, true):  return float(np.mean(np.abs(pred - true)))
def MSE(pred, true):  return float(np.mean((pred - true) ** 2))
def RMSE(pred, true): return float(np.sqrt(MSE(pred, true)))
def MAPE(pred, true): return float(np.mean(np.abs((pred - true) / (np.abs(true) + 1e-5))))
def MSPE(pred, true): return float(np.mean(np.square((pred - true) / (np.abs(true) + 1e-5))))

def metric(pred, true):
    return (MAE(pred, true), MSE(pred, true), RMSE(pred, true),
            MAPE(pred, true), MSPE(pred, true),
            RSE(pred, true),  CORR(pred, true))


def _self_test():
    np.random.seed(42)
    T = 12
    ev = TCEvaluator(pred_len=T, compute_dtw=True)
    for _ in range(18):
        gt   = np.zeros((T, 2))
        gt[:, 0] = np.linspace(1300, 1250, T)
        gt[:, 1] = np.linspace(150,  270,  T)
        pred = gt + np.random.randn(T, 2) * 3.0
        ev.update(pred, gt)
    for _ in range(2):
        gt   = np.zeros((T, 2))
        gt[:, 0] = [1300,1290,1280,1270,1260,1255,1258,1265,1278,1295,1315,1335]
        gt[:, 1] = [150, 160, 175, 192, 210, 228, 242, 255, 265, 270, 270, 268]
        pred = gt + np.random.randn(T, 2) * 5.0
        ev.update(pred, gt)
    # Test with fake CLIPER ugde
    ev.cliper_ugde = {72: 600.0}
    m = ev.compute(tag="selftest")
    print(m.summary())
    assert m.n_rec == 2,  f"Expected 2 recurvature, got {m.n_rec}"
    assert m.n_str == 18, f"Expected 18 straight, got {m.n_str}"
    assert not np.isnan(m.tss_72h), "TSS should not be nan when cliper_ugde is set"
    assert not np.isnan(m.dtw_mean), "DTW should not be nan with compute_dtw=True"
    print("\n✅ All assertions passed (including TSS and DTW).")
    ev.save_csv("/tmp/tc_metrics_selftest_v4.csv", tag="selftest")
    print("✅ CSV export OK")


if __name__ == "__main__":
    _self_test()
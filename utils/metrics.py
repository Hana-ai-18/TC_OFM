
# # # """
# # # utils/metrics.py  ── v5
# # # =============================
# # # FIXES vs v4:

# # #   FIX-MET-8  [CRITICAL] StepErrorAccumulator.update() crash khi curriculum
# # #              training active.

# # #              Root cause: haversine_km_torch(pred, gt_sliced) trả về
# # #              dist_km shape [T_active, B] với T_active < pred_len (ví dụ
# # #              T=4 khi curriculum_start_len=4). Nhưng self._sum shape là
# # #              [pred_len=12]. Dòng:
# # #                self._sum += d.sum(axis=1)
# # #              → ValueError: operands could not be broadcast together
# # #                with shapes (12,) (4,) — đây là CRASH trong traceback.

# # #              Fix: pad d với zeros để luôn có shape [pred_len, B] trước
# # #              khi cộng vào accumulator. Steps được pad có distance=0 km
# # #              và được đếm vào n_samples → ADE của các step đó sẽ thấp
# # #              giả tạo. Dùng active_pred_len để compute() biết cần average
# # #              trên bao nhiêu steps thật sự.

# # #              compute() thêm key "active_steps" để caller biết.

# # # Kept from v4:
# # #   FIX-MET-7  LANDFALL_TARGETS, LANDFALL_RADIUS_KM exported
# # #   FIX-MET-6  compute_dtw default True
# # #   FIX-MET-5  crps_2d vectorised pairwise
# # #   FIX-MET-4  ssr_step: return 1.0 when spread=rmse≈0
# # #   FIX-MET-3  bss_mean computed externally
# # #   FIX-MET-2  TSS computed when cliper_ugde set
# # #   FIX-MET-1  cliper_forecast, denorm_deg_np exported
# # # """

# # # from __future__ import annotations

# # # import csv
# # # import os
# # # from dataclasses import dataclass, field
# # # from datetime import datetime
# # # from typing import Dict, List, Optional, Tuple

# # # import numpy as np

# # # try:
# # #     import torch
# # #     HAS_TORCH = True
# # # except ImportError:
# # #     HAS_TORCH = False
# # #     torch = None  # type: ignore

# # # # ── Physical constants ────────────────────────────────────────────────────────
# # # R_EARTH_KM   = 6371.0
# # # STEP_HOURS   = 6
# # # PRED_LEN     = 12

# # # CLIPER_ALPHA = {h: h / 12 for h in range(1, 13)}
# # # HORIZON_STEPS: Dict[int, int] = {12: 1, 24: 3, 36: 5, 48: 7, 60: 9, 72: 11}
# # # RECURV_THR_DEG = 45.0

# # # LANDFALL_TARGETS = [
# # #     ("DaNang",    108.2, 16.1),
# # #     ("Manila",    121.0, 14.6),
# # #     ("Taipei",    121.5, 25.0),
# # #     ("Shanghai",  121.5, 31.2),
# # #     ("Okinawa",   127.8, 26.3),
# # # ]
# # # LANDFALL_RADIUS_KM = 300.0


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  1. Primitives
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def haversine_km(
# # #     p1: np.ndarray,
# # #     p2: np.ndarray,
# # #     lon_idx: int = 0,
# # #     lat_idx: int = 1,
# # #     unit_01deg: bool = True,
# # # ) -> np.ndarray:
# # #     scale = 10.0 if unit_01deg else 1.0
# # #     lat1 = np.deg2rad(p1[..., lat_idx] / scale)
# # #     lat2 = np.deg2rad(p2[..., lat_idx] / scale)
# # #     dlon = np.deg2rad((p2[..., lon_idx] - p1[..., lon_idx]) / scale)
# # #     dlat = np.deg2rad((p2[..., lat_idx] - p1[..., lat_idx]) / scale)
# # #     a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
# # #     return 2.0 * R_EARTH_KM * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


# # # def haversine_km_np(
# # #     p1: np.ndarray,
# # #     p2: np.ndarray,
# # #     unit_01deg: bool = True,
# # # ) -> np.ndarray:
# # #     """Alias với signature rõ ràng hơn."""
# # #     return haversine_km(p1, p2, unit_01deg=unit_01deg)


# # # def haversine_km_torch(pred, gt, lon_idx: int = 0, lat_idx: int = 1,
# # #                         unit_01deg: bool = True):
# # #     scale = 10.0 if unit_01deg else 1.0
# # #     lat1 = torch.deg2rad(gt[..., lat_idx]   / scale)
# # #     lat2 = torch.deg2rad(pred[..., lat_idx] / scale)
# # #     dlon = torch.deg2rad((pred[..., lon_idx] - gt[..., lon_idx]) / scale)
# # #     dlat = torch.deg2rad((pred[..., lat_idx] - gt[..., lat_idx]) / scale)
# # #     a = (torch.sin(dlat / 2.0) ** 2
# # #          + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2)
# # #     return 2.0 * R_EARTH_KM * torch.asin(a.clamp(0.0, 1.0).sqrt())


# # # def denorm_np(n: np.ndarray) -> np.ndarray:
# # #     """Normalised coords → 0.1° units (NumPy)."""
# # #     r = n.copy()
# # #     r[..., 0] = n[..., 0] * 50.0 + 1800.0
# # #     r[..., 1] = n[..., 1] * 50.0
# # #     return r


# # # def denorm_torch(n):
# # #     """Normalised coords → 0.1° units (PyTorch)."""
# # #     r = n.clone()
# # #     r[..., 0] = n[..., 0] * 50.0 + 1800.0
# # #     r[..., 1] = n[..., 1] * 50.0
# # #     return r


# # # def denorm_deg_np(arr_norm: np.ndarray) -> np.ndarray:
# # #     """Normalised coords → degrees."""
# # #     out = arr_norm.copy()
# # #     out[..., 0] = (arr_norm[..., 0] * 50.0 + 1800.0) / 10.0
# # #     out[..., 1] = (arr_norm[..., 1] * 50.0) / 10.0
# # #     return out


# # # def cliper_forecast(obs_norm: np.ndarray, h: int) -> np.ndarray:
# # #     """
# # #     Input: obs_norm [T, 2+] — normalized coords
# # #     Output: [2] — predicted position, normalized
# # #     """
# # #     if obs_norm.shape[0] < 2:
# # #         return obs_norm[-1, :2].copy()
# # #     v = obs_norm[-1, :2] - obs_norm[-2, :2]
# # #     return obs_norm[-1, :2] + h * v


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  2. Tier 1
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def ugde(pred_01: np.ndarray, gt_01: np.ndarray) -> np.ndarray:
# # #     return haversine_km(pred_01, gt_01)


# # # def ade_fde(pred_01: np.ndarray, gt_01: np.ndarray
# # #             ) -> Tuple[float, float, np.ndarray]:
# # #     ps = ugde(pred_01, gt_01)
# # #     return float(ps.mean()), float(ps[-1]), ps


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  3. Tier 2
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def tss(ugde_model: float, ugde_cliper: float) -> float:
# # #     return 1.0 - ugde_model / (ugde_cliper + 1e-8)


# # # def lat_corrected_velocity(traj_01: np.ndarray,
# # #                             lon_idx: int = 0,
# # #                             lat_idx: int = 1) -> np.ndarray:
# # #     lats_rad = np.deg2rad(traj_01[:, lat_idx] / 10.0)
# # #     cos_lat  = np.cos(lats_rad[:-1])
# # #     dlat = np.diff(traj_01[:, lat_idx])
# # #     dlon = np.diff(traj_01[:, lon_idx]) * cos_lat
# # #     return np.stack([dlon, dlat], axis=-1)


# # # def total_rotation_angle(gt_01: np.ndarray) -> float:
# # #     if gt_01.shape[0] < 3:
# # #         return 0.0
# # #     v = lat_corrected_velocity(gt_01)
# # #     total = 0.0
# # #     for i in range(len(v) - 1):
# # #         v1, v2 = v[i], v[i + 1]
# # #         n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
# # #         if n1 < 1e-8 or n2 < 1e-8:
# # #             continue
# # #         cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
# # #         total += np.degrees(np.arccos(cos_a))
# # #     return total


# # # def classify(gt_01: np.ndarray, thr: float = RECURV_THR_DEG) -> str:
# # #     return "recurvature" if total_rotation_angle(gt_01) >= thr else "straight"


# # # def ate_cte(pred_01: np.ndarray, gt_01: np.ndarray,
# # #             lon_idx: int = 0, lat_idx: int = 1
# # #             ) -> Tuple[np.ndarray, np.ndarray]:
# # #     T = pred_01.shape[0]
# # #     ate_arr = np.zeros(T)
# # #     cte_arr = np.zeros(T)
# # #     km_per_01deg = 11.112

# # #     for k in range(T):
# # #         if k == 0:
# # #             dk = gt_01[1] - gt_01[0] if T > 1 else np.array([1.0, 0.0])
# # #         else:
# # #             dk = gt_01[k] - gt_01[k - 1]

# # #         lat_rad = np.deg2rad(gt_01[k, lat_idx] / 10.0)
# # #         dk_km = dk * km_per_01deg
# # #         dk_km[0] *= np.cos(lat_rad)

# # #         norm_dk = np.linalg.norm(dk_km)
# # #         if norm_dk < 1e-8:
# # #             continue

# # #         t_hat = dk_km / norm_dk
# # #         n_hat = np.array([-t_hat[1], t_hat[0]])

# # #         delta_pos = pred_01[k] - gt_01[k]
# # #         delta_km = delta_pos * km_per_01deg
# # #         delta_km[0] *= np.cos(lat_rad)

# # #         ate_arr[k] = float(np.dot(delta_km, t_hat))
# # #         cte_arr[k] = float(np.dot(delta_km, n_hat))

# # #     return ate_arr, cte_arr


# # # def circular_std(angles_deg: np.ndarray) -> float:
# # #     if len(angles_deg) == 0:
# # #         return 0.0
# # #     rads  = np.deg2rad(angles_deg)
# # #     R_bar = np.abs(np.mean(np.exp(1j * rads)))
# # #     R_bar = np.clip(R_bar, 1e-8, 1.0 - 1e-8)
# # #     return float(np.degrees(np.sqrt(-2.0 * np.log(R_bar))))


# # # def compute_csd(traj_01: np.ndarray) -> float:
# # #     v = lat_corrected_velocity(traj_01)
# # #     if len(v) == 0:
# # #         return 0.0
# # #     angles = np.arctan2(v[:, 1], v[:, 0])
# # #     return circular_std(np.degrees(angles))


# # # def oyr(pred_01: np.ndarray, gt_01: np.ndarray) -> float:
# # #     pv = lat_corrected_velocity(pred_01)
# # #     gv = lat_corrected_velocity(gt_01)
# # #     m  = min(len(pv), len(gv))
# # #     if m == 0:
# # #         return 0.0
# # #     dots = np.sum(pv[:m] * gv[:m], axis=1)
# # #     np_  = np.linalg.norm(pv[:m], axis=1)
# # #     ng_  = np.linalg.norm(gv[:m], axis=1)
# # #     valid = (np_ > 1e-8) & (ng_ > 1e-8)
# # #     if valid.sum() == 0:
# # #         return 0.0
# # #     cos_vals = dots[valid] / (np_[valid] * ng_[valid])
# # #     return float((cos_vals < 0).mean())


# # # def hle(pred_01: np.ndarray, gt_01: np.ndarray) -> float:
# # #     def curvature(traj):
# # #         v = lat_corrected_velocity(traj)
# # #         if len(v) < 2:
# # #             return np.array([])
# # #         cross = v[1:, 0] * v[:-1, 1] - v[1:, 1] * v[:-1, 0]
# # #         n1    = np.linalg.norm(v[1:],  axis=1).clip(1e-8)
# # #         n2    = np.linalg.norm(v[:-1], axis=1).clip(1e-8)
# # #         return cross / (n1 * n2)
# # #     kp = curvature(pred_01)
# # #     kg = curvature(gt_01)
# # #     m  = min(len(kp), len(kg))
# # #     if m == 0:
# # #         return float("nan")
# # #     return float(np.mean(np.abs(kp[:m] - kg[:m])))


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  4. Tier 3 — Probabilistic
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def crps_2d(pred_ens_01: np.ndarray, gt_01: np.ndarray) -> float:
# # #     """CRPS energy-form. FIX-MET-5: vectorised pairwise diversity."""
# # #     S = pred_ens_01.shape[0]
# # #     gt_rep  = gt_01[np.newaxis].repeat(S, axis=0)
# # #     acc     = float(np.mean(haversine_km(pred_ens_01, gt_rep)))

# # #     p_i = pred_ens_01[:, np.newaxis, :]
# # #     p_j = pred_ens_01[np.newaxis, :, :]
# # #     p_i_flat = np.broadcast_to(p_i, (S, S, pred_ens_01.shape[-1])).reshape(S * S, -1)
# # #     p_j_flat = np.broadcast_to(p_j, (S, S, pred_ens_01.shape[-1])).reshape(S * S, -1)
# # #     div = float(np.mean(haversine_km(p_i_flat, p_j_flat)))

# # #     return float(acc - 0.5 * div)


# # # def ssr_step(pred_ens_01: np.ndarray, gt_01: np.ndarray) -> float:
# # #     """Spread-Skill Ratio. FIX-MET-4: return 1.0 when spread=rmse≈0."""
# # #     S = pred_ens_01.shape[0]
# # #     ens_mean = pred_ens_01.mean(axis=0)
# # #     ens_mean_rep = ens_mean[np.newaxis].repeat(S, axis=0)
# # #     spread_vals  = haversine_km(pred_ens_01, ens_mean_rep)
# # #     spread       = float(np.sqrt(np.mean(spread_vals ** 2)))
# # #     rmse_em = float(haversine_km(ens_mean[np.newaxis], gt_01[np.newaxis])[0])

# # #     if spread < 1e-6 and rmse_em < 1e-6:
# # #         return 1.0
# # #     return spread / (rmse_em + 1e-8)


# # # def brier_skill_score(
# # #     pred_ens_seqs: List[np.ndarray],
# # #     gt_seqs:       List[np.ndarray],
# # #     step:          int,
# # #     target_deg:    Tuple[float, float],
# # #     radius_km:     float = LANDFALL_RADIUS_KM,
# # #     clim_rate:     Optional[float] = None,
# # # ) -> float:
# # #     """Brier Skill Score for landfall strike probability."""
# # #     N   = len(gt_seqs)
# # #     if N == 0:
# # #         return float("nan")
# # #     bs  = 0.0
# # #     clim_hits = 0

# # #     target_01 = np.array([target_deg[0] * 10.0, target_deg[1] * 10.0])

# # #     for i in range(N):
# # #         if step >= len(gt_seqs[i]):
# # #             continue
# # #         gt_pos   = gt_seqs[i][step]
# # #         ens_seqs = pred_ens_seqs[i]
# # #         if ens_seqs.ndim == 3:
# # #             S = ens_seqs.shape[0]
# # #         else:
# # #             continue

# # #         obs = float(haversine_km(
# # #             gt_pos[np.newaxis], target_01[np.newaxis])[0]) <= radius_km

# # #         hits = sum(
# # #             haversine_km(ens_seqs[s, min(step, ens_seqs.shape[1]-1)][np.newaxis],
# # #                          target_01[np.newaxis])[0] <= radius_km
# # #             for s in range(S)
# # #         )
# # #         p_fc = hits / S
# # #         bs  += (p_fc - float(obs)) ** 2
# # #         clim_hits += float(obs)

# # #     bs /= max(N, 1)
# # #     p_clim = (clim_rate if clim_rate is not None else clim_hits / max(N, 1))
# # #     bs_clim = p_clim * (1.0 - p_clim) + 1e-8
# # #     return float(1.0 - bs / bs_clim)


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  5. Tier 4 — Geometric
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def dtw_haversine(s: np.ndarray, t: np.ndarray) -> float:
# # #     n, m = len(s), len(t)
# # #     dp   = np.full((n + 1, m + 1), np.inf)
# # #     dp[0, 0] = 0.0
# # #     for i in range(1, n + 1):
# # #         for j in range(1, m + 1):
# # #             cost = float(haversine_km(s[i-1:i], t[j-1:j])[0])
# # #             dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
# # #     return float(dp[n, m])


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  6. Per-sequence result container
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # @dataclass
# # # class SequenceResult:
# # #     ade:       float
# # #     fde:       float
# # #     per_step:  np.ndarray

# # #     ate:       np.ndarray
# # #     cte:       np.ndarray
# # #     category:  str
# # #     category_pred: str
# # #     theta:     float
# # #     csd_gt:    float
# # #     oyr_val:   float
# # #     hle_val:   float

# # #     crps:      Optional[np.ndarray] = None
# # #     ssr:       Optional[np.ndarray] = None
# # #     dtw:       float = float("nan")

# # #     loss_fm:      float = float("nan")
# # #     loss_dir:     float = float("nan")
# # #     loss_step:    float = float("nan")
# # #     loss_disp:    float = float("nan")
# # #     loss_heading: float = float("nan")
# # #     loss_smooth:  float = float("nan")
# # #     loss_pinn:    float = float("nan")
# # #     loss_total:   float = float("nan")


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  7. Dataset-level aggregated metrics
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # @dataclass
# # # class DatasetMetrics:
# # #     ade:            float = 0.0
# # #     fde:            float = 0.0
# # #     per_step_mean:  np.ndarray = field(default_factory=lambda: np.zeros(PRED_LEN))
# # #     per_step_std:   np.ndarray = field(default_factory=lambda: np.zeros(PRED_LEN))
# # #     ugde_12h:       float = 0.0
# # #     ugde_24h:       float = 0.0
# # #     ugde_48h:       float = 0.0
# # #     ugde_72h:       float = 0.0

# # #     tss_72h:        float = float("nan")
# # #     ate_mean:       float = 0.0
# # #     cte_mean:       float = 0.0
# # #     ate_abs_mean:   float = 0.0
# # #     cte_abs_mean:   float = 0.0
# # #     ade_str:        float = float("nan")
# # #     ade_rec:        float = float("nan")
# # #     pr:             float = float("nan")
# # #     rdr:            float = float("nan")
# # #     n_str:          int   = 0
# # #     n_rec:          int   = 0

# # #     crps_mean:      float = float("nan")
# # #     crps_72h:       float = float("nan")
# # #     ssr_mean:       float = float("nan")
# # #     bss_mean:       float = float("nan")

# # #     dtw_mean:       float = float("nan")
# # #     dtw_str:        float = float("nan")
# # #     dtw_rec:        float = float("nan")
# # #     oyr_mean:       float = float("nan")
# # #     oyr_rec:        float = float("nan")
# # #     hle_mean:       float = float("nan")
# # #     hle_rec:        float = float("nan")

# # #     loss_fm:        float = float("nan")
# # #     loss_dir:       float = float("nan")
# # #     loss_step:      float = float("nan")
# # #     loss_disp:      float = float("nan")
# # #     loss_heading:   float = float("nan")
# # #     loss_smooth:    float = float("nan")
# # #     loss_pinn:      float = float("nan")
# # #     loss_total:     float = float("nan")

# # #     n_total:        int   = 0
# # #     timestamp:      str   = ""

# # #     def summary(self) -> str:
# # #         lines = [
# # #             "═" * 64,
# # #             "  FM+PINN TC Track Metrics  (4-tier)",
# # #             "═" * 64,
# # #             f"  Sequences : {self.n_total}"
# # #             f"  (str={self.n_str}, rec={self.n_rec})",
# # #             "",
# # #             "  ── Tier 1: Position ─────────────────────────────────",
# # #             f"  ADE        : {self.ade:.1f} km",
# # #             f"  FDE (72h)  : {self.fde:.1f} km",
# # #             f"  UGDE       : 12h={self.ugde_12h:.0f}  24h={self.ugde_24h:.0f}"
# # #             f"  48h={self.ugde_48h:.0f}  72h={self.ugde_72h:.0f} km",
# # #             "",
# # #             "  ── Tier 2: Operational ──────────────────────────────",
# # #             f"  TSS (72h)  : {self.tss_72h:.3f}" +
# # #             (" ✅" if not np.isnan(self.tss_72h) and self.tss_72h > 0 else ""),
# # #             f"  |ATE| mean : {self.ate_abs_mean:.1f} km",
# # #             f"  |CTE| mean : {self.cte_abs_mean:.1f} km",
# # #             f"  ADE_str    : {self.ade_str:.1f} km",
# # #             f"  ADE_rec    : {self.ade_rec:.1f} km",
# # #             f"  PR         : {self.pr:.2f}" +
# # #             (" (no bias)" if not np.isnan(self.pr) and self.pr < 1.3 else
# # #              " ⚠️ straight-track bias"),
# # #             f"  RDR        : {self.rdr:.3f}" +
# # #             (" ✅" if not np.isnan(self.rdr) and self.rdr > 0.5 else ""),
# # #             "",
# # #             "  ── Tier 3: Probabilistic ────────────────────────────",
# # #             f"  CRPS mean  : {self.crps_mean:.1f} km",
# # #             f"  CRPS 72h   : {self.crps_72h:.1f} km",
# # #             f"  SSR mean   : {self.ssr_mean:.3f}  (1=calibrated)",
# # #             f"  BSS mean   : {self.bss_mean:.3f}",
# # #             "",
# # #             "  ── Tier 4: Geometric ────────────────────────────────",
# # #             f"  DTW mean   : {self.dtw_mean:.1f} km",
# # #             f"  OYR mean   : {self.oyr_mean:.3f}  (0=ideal)",
# # #             f"  OYR rec    : {self.oyr_rec:.3f}",
# # #             f"  HLE mean   : {self.hle_mean:.4f}",
# # #             f"  HLE rec    : {self.hle_rec:.4f}",
# # #             "═" * 64,
# # #         ]
# # #         return "\n".join(lines)


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  8. CSV export
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # _CSV_FIELDS = [
# # #     "timestamp", "n_total", "n_str", "n_rec",
# # #     "ade", "fde", "ugde_12h", "ugde_24h", "ugde_48h", "ugde_72h",
# # #     "tss_72h", "ate_abs_mean", "cte_abs_mean",
# # #     "ade_str", "ade_rec", "pr", "rdr",
# # #     "crps_mean", "crps_72h", "ssr_mean", "bss_mean",
# # #     "dtw_mean", "dtw_str", "dtw_rec",
# # #     "oyr_mean", "oyr_rec", "hle_mean", "hle_rec",
# # #     "loss_total", "loss_fm", "loss_dir", "loss_step",
# # #     "loss_disp", "loss_heading", "loss_smooth", "loss_pinn",
# # # ]


# # # def save_metrics_csv(metrics: DatasetMetrics, csv_path: str, tag: str = "") -> None:
# # #     os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # #     write_header = not os.path.exists(csv_path)
# # #     row: Dict[str, object] = {f: getattr(metrics, f, float("nan"))
# # #                                for f in _CSV_FIELDS}
# # #     if tag:
# # #         row["timestamp"] = f"{tag}_{metrics.timestamp}"
# # #     with open(csv_path, "a", newline="") as fh:
# # #         writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
# # #         if write_header:
# # #             writer.writeheader()
# # #         writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
# # #                          for k, v in row.items()})


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  9. Main Evaluator
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class TCEvaluator:
# # #     def __init__(
# # #         self,
# # #         pred_len:    int   = PRED_LEN,
# # #         step_hours:  int   = STEP_HOURS,
# # #         recurv_thr:  float = RECURV_THR_DEG,
# # #         compute_dtw: bool  = True,
# # #         cliper_ugde: Optional[Dict[int, float]] = None,
# # #     ):
# # #         self.pred_len    = pred_len
# # #         self.step_hours  = step_hours
# # #         self.recurv_thr  = recurv_thr
# # #         self.compute_dtw = compute_dtw
# # #         self.cliper_ugde = cliper_ugde or {}
# # #         self._results:  List[SequenceResult] = []
# # #         self._loss_buf: List[Dict[str, float]] = []

# # #     def reset(self) -> None:
# # #         self._results  = []
# # #         self._loss_buf = []

# # #     def update(
# # #         self,
# # #         pred_01:    np.ndarray,
# # #         gt_01:      np.ndarray,
# # #         pred_ens:   Optional[np.ndarray] = None,
# # #         loss_dict:  Optional[Dict[str, float]] = None,
# # #     ) -> None:
# # #         T = min(len(pred_01), len(gt_01), self.pred_len)
# # #         p = pred_01[:T]
# # #         g = gt_01[:T]

# # #         _ade, _fde, ps = ade_fde(p, g)
# # #         _ate, _cte = ate_cte(p, g)
# # #         theta      = total_rotation_angle(g)
# # #         cat        = "recurvature" if theta >= self.recurv_thr else "straight"
# # #         theta_pred = total_rotation_angle(p)
# # #         cat_pred   = "recurvature" if theta_pred >= self.recurv_thr else "straight"

# # #         csd_g      = compute_csd(g)
# # #         _oyr       = oyr(p, g)
# # #         _hle       = hle(p, g)

# # #         crps_arr: Optional[np.ndarray] = None
# # #         ssr_arr:  Optional[np.ndarray] = None
# # #         if pred_ens is not None:
# # #             crps_arr = np.array([
# # #                 crps_2d(pred_ens[:, h, :] if pred_ens.ndim == 3 else pred_ens[:, h, :],
# # #                         g[h]) for h in range(T)
# # #             ])
# # #             ssr_arr = np.array([
# # #                 ssr_step(pred_ens[:, h, :] if pred_ens.ndim == 3 else pred_ens[:, h, :],
# # #                          g[h]) for h in range(T)
# # #             ])

# # #         dtw_val = dtw_haversine(p, g) if self.compute_dtw else float("nan")

# # #         r = SequenceResult(
# # #             ade=_ade, fde=_fde, per_step=ps,
# # #             ate=_ate, cte=_cte,
# # #             category=cat,
# # #             category_pred=cat_pred,
# # #             theta=theta, csd_gt=csd_g,
# # #             oyr_val=_oyr, hle_val=_hle,
# # #             crps=crps_arr, ssr=ssr_arr, dtw=dtw_val,
# # #         )
# # #         if loss_dict:
# # #             r.loss_fm      = loss_dict.get("fm",      float("nan"))
# # #             r.loss_dir     = loss_dict.get("dir",     float("nan"))
# # #             r.loss_step    = loss_dict.get("step",    float("nan"))
# # #             r.loss_disp    = loss_dict.get("disp",    float("nan"))
# # #             r.loss_heading = loss_dict.get("heading", float("nan"))
# # #             r.loss_smooth  = loss_dict.get("smooth",  float("nan"))
# # #             r.loss_pinn    = loss_dict.get("pinn",    float("nan"))
# # #             r.loss_total   = loss_dict.get("total",   float("nan"))

# # #         self._results.append(r)

# # #     def update_batch(self, pred_norm, gt_norm,
# # #                      loss_dict: Optional[Dict[str, float]] = None) -> None:
# # #         pred_d = denorm_torch(pred_norm).cpu().numpy()
# # #         gt_d   = denorm_torch(gt_norm).cpu().numpy()
# # #         B = pred_d.shape[1]
# # #         for b in range(B):
# # #             self.update(pred_d[:, b, :], gt_d[:, b, :], loss_dict=loss_dict)

# # #     def compute(self, tag: str = "") -> DatasetMetrics:
# # #         if not self._results:
# # #             return DatasetMetrics()

# # #         rs  = self._results
# # #         n   = len(rs)
# # #         ts  = tag or datetime.now().strftime("%Y%m%d_%H%M%S")

# # #         all_steps = np.stack([r.per_step[:self.pred_len] for r in rs])
# # #         step_mean = all_steps.mean(0)
# # #         step_std  = all_steps.std(0)

# # #         def _h(step_idx):
# # #             return float(step_mean[step_idx]) if step_idx < self.pred_len else float("nan")

# # #         str_r = [r for r in rs if r.category == "straight"]
# # #         rec_r = [r for r in rs if r.category == "recurvature"]

# # #         ade_s  = float(np.mean([r.ade for r in str_r])) if str_r else float("nan")
# # #         ade_r  = float(np.mean([r.ade for r in rec_r])) if rec_r else float("nan")
# # #         pr_val = ade_r / (ade_s + 1e-8) if (str_r and rec_r) else float("nan")

# # #         all_ate = np.concatenate([r.ate for r in rs])
# # #         all_cte = np.concatenate([r.cte for r in rs])

# # #         tss_val = float("nan")
# # #         if self.cliper_ugde and 72 in self.cliper_ugde:
# # #             ugde_72 = _h(HORIZON_STEPS[72])
# # #             if not np.isnan(ugde_72):
# # #                 tss_val = tss(ugde_72, self.cliper_ugde[72])

# # #         crps_seqs = [r.crps for r in rs if r.crps is not None]
# # #         ssr_seqs  = [r.ssr  for r in rs if r.ssr  is not None]

# # #         crps_mean = float("nan")
# # #         crps_72h  = float("nan")
# # #         ssr_mean  = float("nan")
# # #         if crps_seqs:
# # #             crps_mat = np.stack(crps_seqs)
# # #             crps_mean = float(crps_mat.mean())
# # #             step_72 = HORIZON_STEPS.get(72, -1)
# # #             if 0 <= step_72 < crps_mat.shape[1]:
# # #                 crps_72h = float(crps_mat[:, step_72].mean())
# # #         if ssr_seqs:
# # #             ssr_mat  = np.stack(ssr_seqs)
# # #             valid_ssr = ssr_mat[~np.isnan(ssr_mat)]
# # #             ssr_mean = float(valid_ssr.mean()) if len(valid_ssr) > 0 else float("nan")

# # #         dtw_all = [r.dtw for r in rs  if not np.isnan(r.dtw)]
# # #         dtw_s   = [r.dtw for r in str_r if not np.isnan(r.dtw)]
# # #         dtw_rc  = [r.dtw for r in rec_r if not np.isnan(r.dtw)]

# # #         oyr_all = [r.oyr_val for r in rs]
# # #         oyr_rc  = [r.oyr_val for r in rec_r]
# # #         hle_all = [r.hle_val for r in rs  if not np.isnan(r.hle_val)]
# # #         hle_rc  = [r.hle_val for r in rec_r if not np.isnan(r.hle_val)]

# # #         def _mean_loss(attr):
# # #             vals = [getattr(r, attr) for r in rs if not np.isnan(getattr(r, attr))]
# # #             return float(np.mean(vals)) if vals else float("nan")

# # #         csd_vals = [r.csd_gt for r in rs]
# # #         tau = float(np.median(csd_vals)) if csd_vals else 0.0

# # #         if rec_r:
# # #             rdr_num = sum(1 for r in rec_r if r.category_pred == "recurvature")
# # #             rdr_val = rdr_num / len(rec_r)
# # #         else:
# # #             rdr_val = float("nan")

# # #         m = DatasetMetrics(
# # #             ade           = float(np.mean([r.ade for r in rs])),
# # #             fde           = float(np.mean([r.fde for r in rs])),
# # #             per_step_mean = step_mean,
# # #             per_step_std  = step_std,
# # #             ugde_12h      = _h(HORIZON_STEPS[12]),
# # #             ugde_24h      = _h(HORIZON_STEPS[24]),
# # #             ugde_48h      = _h(HORIZON_STEPS[48]),
# # #             ugde_72h      = _h(HORIZON_STEPS[72]),
# # #             tss_72h       = tss_val,
# # #             ate_mean      = float(np.mean(all_ate)),
# # #             cte_mean      = float(np.mean(all_cte)),
# # #             ate_abs_mean  = float(np.mean(np.abs(all_ate))),
# # #             cte_abs_mean  = float(np.mean(np.abs(all_cte))),
# # #             ade_str       = ade_s,
# # #             ade_rec       = ade_r,
# # #             pr            = pr_val,
# # #             rdr           = rdr_val,
# # #             n_str         = len(str_r),
# # #             n_rec         = len(rec_r),
# # #             crps_mean     = crps_mean,
# # #             crps_72h      = crps_72h,
# # #             ssr_mean      = ssr_mean,
# # #             bss_mean      = float("nan"),
# # #             dtw_mean      = float(np.mean(dtw_all)) if dtw_all else float("nan"),
# # #             dtw_str       = float(np.mean(dtw_s))   if dtw_s   else float("nan"),
# # #             dtw_rec       = float(np.mean(dtw_rc))  if dtw_rc  else float("nan"),
# # #             oyr_mean      = float(np.mean(oyr_all)) if oyr_all else float("nan"),
# # #             oyr_rec       = float(np.mean(oyr_rc))  if oyr_rc  else float("nan"),
# # #             hle_mean      = float(np.mean(hle_all)) if hle_all else float("nan"),
# # #             hle_rec       = float(np.mean(hle_rc))  if hle_rc  else float("nan"),
# # #             loss_fm       = _mean_loss("loss_fm"),
# # #             loss_dir      = _mean_loss("loss_dir"),
# # #             loss_step     = _mean_loss("loss_step"),
# # #             loss_disp     = _mean_loss("loss_disp"),
# # #             loss_heading  = _mean_loss("loss_heading"),
# # #             loss_smooth   = _mean_loss("loss_smooth"),
# # #             loss_pinn     = _mean_loss("loss_pinn"),
# # #             loss_total    = _mean_loss("loss_total"),
# # #             n_total       = n,
# # #             timestamp     = ts,
# # #         )
# # #         return m

# # #     def save_csv(self, csv_path: str, tag: str = "") -> None:
# # #         m = self.compute(tag=tag)
# # #         save_metrics_csv(m, csv_path, tag=tag)
# # #         print(f"  📊  Metrics → {csv_path}")


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  10. Fast Tier-1 accumulator  ── FIX-MET-8
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class StepErrorAccumulator:
# # #     """
# # #     FIX-MET-8: Pad dist_km [T_active, B] → [pred_len, B] với zeros khi
# # #     T_active < pred_len (curriculum training). Tránh crash broadcast.
# # #     compute() trả về "active_steps" để caller biết steps thật sự có data.
# # #     """

# # #     def __init__(self, pred_len: int = PRED_LEN, step_hours: int = STEP_HOURS):
# # #         self.pred_len   = pred_len
# # #         self.step_hours = step_hours
# # #         self.reset()

# # #     def reset(self) -> None:
# # #         self._sum         = np.zeros(self.pred_len, dtype=np.float64)
# # #         self._sum_sq      = np.zeros(self.pred_len, dtype=np.float64)
# # #         self._count       = np.zeros(self.pred_len, dtype=np.int64)  # per-step count
# # #         self._active_max  = 0  # track max T_active seen

# # #     def update(self, dist_km) -> None:
# # #         if HAS_TORCH and torch.is_tensor(dist_km):
# # #             d = dist_km.double().cpu().numpy()
# # #         else:
# # #             d = np.asarray(dist_km, dtype=np.float64)

# # #         # Ensure [T, B] shape
# # #         if d.ndim == 1:
# # #             d = d.reshape(-1, 1)
# # #         elif d.ndim != 2:
# # #             return

# # #         T_actual, B = d.shape

# # #         # FIX-MET-8: pad to pred_len nếu T_actual < pred_len
# # #         if T_actual < self.pred_len:
# # #             pad = np.zeros((self.pred_len - T_actual, B), dtype=np.float64)
# # #             d_full = np.concatenate([d, pad], axis=0)
# # #         else:
# # #             d_full = d[:self.pred_len]
# # #             T_actual = self.pred_len

# # #         # Chỉ cộng vào count cho steps có data thật (không pad)
# # #         self._sum    += d_full.sum(axis=1)
# # #         self._sum_sq += (d_full ** 2).sum(axis=1)
# # #         # Count per step: chỉ T_actual steps đầu có data thật
# # #         self._count[:T_actual] += B
# # #         # Steps được pad không tăng count → average của chúng = 0/0 = nan
# # #         # Dùng max T_actual để biết đến đâu có data thật
# # #         self._active_max = max(self._active_max, T_actual)

# # #     def compute(self) -> Dict:
# # #         if self._count[0] == 0:
# # #             return {}

# # #         # Tính per-step mean chỉ cho steps có count > 0
# # #         count_safe = np.where(self._count > 0, self._count, 1)
# # #         ps  = self._sum / count_safe
# # #         ps  = np.where(self._count > 0, ps, 0.0)
# # #         ps2 = self._sum_sq / count_safe
# # #         std = np.sqrt(np.maximum(ps2 - ps ** 2, 0.0))
# # #         std = np.where(self._count > 0, std, 0.0)

# # #         # ADE chỉ trên active steps
# # #         active = self._active_max
# # #         ade_val = float(ps[:active].mean()) if active > 0 else 0.0
# # #         fde_val = float(ps[active - 1]) if active > 0 else 0.0

# # #         out: Dict = {
# # #             "per_step":     ps,
# # #             "per_step_std": std,
# # #             "ADE":          ade_val,
# # #             "FDE":          fde_val,
# # #             "active_steps": active,
# # #             "n_samples":    int(self._count[0]),
# # #         }
# # #         for h, s in HORIZON_STEPS.items():
# # #             if s < active:
# # #                 out[f"{h}h"]     = float(ps[s])
# # #                 out[f"{h}h_std"] = float(std[s])
# # #             else:
# # #                 out[f"{h}h"]     = float("nan")
# # #                 out[f"{h}h_std"] = float("nan")
# # #         return out


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  11. Backward-compatible wrappers
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def RSE(pred, true):
# # #     return float(np.sqrt(np.sum((true - pred) ** 2))
# # #                  / (np.sqrt(np.sum((true - true.mean()) ** 2)) + 1e-8))

# # # def CORR(pred, true):
# # #     u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
# # #     d = np.sqrt(
# # #         ((true - true.mean(0)) ** 2).sum(0)
# # #         * ((pred - pred.mean(0)) ** 2).sum(0)
# # #     ) + 1e-8
# # #     return u / d

# # # def MAE(pred, true):  return float(np.mean(np.abs(pred - true)))
# # # def MSE(pred, true):  return float(np.mean((pred - true) ** 2))
# # # def RMSE(pred, true): return float(np.sqrt(MSE(pred, true)))
# # # def MAPE(pred, true): return float(np.mean(np.abs((pred - true) / (np.abs(true) + 1e-5))))
# # # def MSPE(pred, true): return float(np.mean(np.square((pred - true) / (np.abs(true) + 1e-5))))

# # # def metric(pred, true):
# # #     return (MAE(pred, true), MSE(pred, true), RMSE(pred, true),
# # #             MAPE(pred, true), MSPE(pred, true),
# # #             RSE(pred, true),  CORR(pred, true))

# # """
# # utils/metrics.py  ── v6
# # =============================
# # FIXES vs v5:

# #   FIX-MET-9  StepErrorAccumulator.compute() trả về ADE chỉ trên active steps
# #              thật sự có data (không tính padded zeros). v5 đã pad đúng nhưng
# #              active_max vẫn có thể sai khi update() gọi với T=pred_len nhưng
# #              các bước cuối là 0 (curriculum). Sử dụng per-step count array
# #              thay vì _active_max scalar để chính xác hơn.

# #   FIX-MET-10 haversine_km_torch: thứ tự tham số (pred, gt) nhưng nội dung
# #              dùng gt cho lat1 và pred cho lat2 → bất đối xứng. Sửa lại
# #              dùng đúng pred cho p1, gt cho p2, kết quả haversine không thay
# #              đổi vì distance đối xứng nhưng tránh nhầm lẫn về convention.

# #   FIX-MET-11 cliper_forecast: trả về position đúng chuẩn normalised [2] thay
# #              vì [:2] slice có thể gây index error nếu input < 2 cols.

# # Kept from v5:
# #   FIX-MET-8  StepErrorAccumulator pad zeros, per-step count
# #   FIX-MET-7  LANDFALL_TARGETS, LANDFALL_RADIUS_KM exported
# #   FIX-MET-6  compute_dtw default True
# #   FIX-MET-5  crps_2d vectorised pairwise
# #   FIX-MET-4  ssr_step: return 1.0 when spread=rmse≈0
# #   FIX-MET-3  bss_mean computed externally
# #   FIX-MET-2  TSS computed when cliper_ugde set
# #   FIX-MET-1  cliper_forecast, denorm_deg_np exported
# # """

# # from __future__ import annotations

# # import csv
# # import os
# # from dataclasses import dataclass, field
# # from datetime import datetime
# # from typing import Dict, List, Optional, Tuple

# # import numpy as np

# # try:
# #     import torch
# #     HAS_TORCH = True
# # except ImportError:
# #     HAS_TORCH = False
# #     torch = None  # type: ignore

# # # ── Physical constants ────────────────────────────────────────────────────────
# # R_EARTH_KM   = 6371.0
# # STEP_HOURS   = 6
# # PRED_LEN     = 12

# # CLIPER_ALPHA = {h: h / 12 for h in range(1, 13)}
# # HORIZON_STEPS: Dict[int, int] = {6: 0, 12: 1, 24: 3, 36: 5, 48: 7, 60: 9, 72: 11}
# # RECURV_THR_DEG = 45.0

# # LANDFALL_TARGETS = [
# #     ("DaNang",    108.2, 16.1),
# #     ("Manila",    121.0, 14.6),
# #     ("Taipei",    121.5, 25.0),
# #     ("Shanghai",  121.5, 31.2),
# #     ("Okinawa",   127.8, 26.3),
# # ]
# # LANDFALL_RADIUS_KM = 300.0


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  1. Primitives
# # # ══════════════════════════════════════════════════════════════════════════════

# # def haversine_km(
# #     p1: np.ndarray,
# #     p2: np.ndarray,
# #     lon_idx: int = 0,
# #     lat_idx: int = 1,
# #     unit_01deg: bool = True,
# # ) -> np.ndarray:
# #     """Haversine distance in km. Supports (..., 2) shaped arrays."""
# #     scale = 10.0 if unit_01deg else 1.0
# #     lat1 = np.deg2rad(p1[..., lat_idx] / scale)
# #     lat2 = np.deg2rad(p2[..., lat_idx] / scale)
# #     dlon = np.deg2rad((p2[..., lon_idx] - p1[..., lon_idx]) / scale)
# #     dlat = np.deg2rad((p2[..., lat_idx] - p1[..., lat_idx]) / scale)
# #     a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
# #     return 2.0 * R_EARTH_KM * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


# # def haversine_km_np(
# #     p1: np.ndarray,
# #     p2: np.ndarray,
# #     unit_01deg: bool = True,
# # ) -> np.ndarray:
# #     return haversine_km(p1, p2, unit_01deg=unit_01deg)


# # def haversine_km_torch(pred, gt, lon_idx: int = 0, lat_idx: int = 1,
# #                        unit_01deg: bool = True):
# #     """
# #     FIX-MET-10: pred=p1, gt=p2. Distance is symmetric so result unchanged,
# #     but convention is now consistent with numpy version.
# #     """
# #     scale = 10.0 if unit_01deg else 1.0
# #     lat1  = torch.deg2rad(pred[..., lat_idx] / scale)
# #     lat2  = torch.deg2rad(gt[..., lat_idx]   / scale)
# #     dlon  = torch.deg2rad((pred[..., lon_idx] - gt[..., lon_idx]) / scale)
# #     dlat  = torch.deg2rad((pred[..., lat_idx] - gt[..., lat_idx]) / scale)
# #     a = (torch.sin(dlat / 2.0) ** 2
# #          + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2)
# #     return 2.0 * R_EARTH_KM * torch.asin(a.clamp(0.0, 1.0).sqrt())


# # def denorm_np(n: np.ndarray) -> np.ndarray:
# #     """Normalised coords → 0.1° units (NumPy)."""
# #     r = n.copy()
# #     r[..., 0] = n[..., 0] * 50.0 + 1800.0
# #     r[..., 1] = n[..., 1] * 50.0
# #     return r


# # def denorm_torch(n):
# #     """Normalised coords → 0.1° units (PyTorch)."""
# #     r = n.clone()
# #     r[..., 0] = n[..., 0] * 50.0 + 1800.0
# #     r[..., 1] = n[..., 1] * 50.0
# #     return r


# # def denorm_deg_np(arr_norm: np.ndarray) -> np.ndarray:
# #     """Normalised coords → degrees."""
# #     out = arr_norm.copy()
# #     out[..., 0] = (arr_norm[..., 0] * 50.0 + 1800.0) / 10.0
# #     out[..., 1] = (arr_norm[..., 1] * 50.0) / 10.0
# #     return out


# # def cliper_forecast(obs_norm: np.ndarray, h: int) -> np.ndarray:
# #     """
# #     FIX-MET-11: Robust slice, always returns shape [2].
# #     Input: obs_norm [T, 2+] normalised coords.
# #     Output: [2] predicted position, normalised.
# #     """
# #     if obs_norm.shape[0] < 2:
# #         return obs_norm[-1, :2].copy()
# #     v = obs_norm[-1, :2] - obs_norm[-2, :2]
# #     return obs_norm[-1, :2] + h * v


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  2. Tier 1
# # # ══════════════════════════════════════════════════════════════════════════════

# # def ugde(pred_01: np.ndarray, gt_01: np.ndarray) -> np.ndarray:
# #     return haversine_km(pred_01, gt_01)


# # def ade_fde(pred_01: np.ndarray, gt_01: np.ndarray
# #             ) -> Tuple[float, float, np.ndarray]:
# #     ps = ugde(pred_01, gt_01)
# #     return float(ps.mean()), float(ps[-1]), ps


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  3. Tier 2
# # # ══════════════════════════════════════════════════════════════════════════════

# # def tss(ugde_model: float, ugde_cliper: float) -> float:
# #     return 1.0 - ugde_model / (ugde_cliper + 1e-8)


# # def lat_corrected_velocity(traj_01: np.ndarray,
# #                             lon_idx: int = 0,
# #                             lat_idx: int = 1) -> np.ndarray:
# #     lats_rad = np.deg2rad(traj_01[:, lat_idx] / 10.0)
# #     cos_lat  = np.cos(lats_rad[:-1])
# #     dlat = np.diff(traj_01[:, lat_idx])
# #     dlon = np.diff(traj_01[:, lon_idx]) * cos_lat
# #     return np.stack([dlon, dlat], axis=-1)


# # def total_rotation_angle(gt_01: np.ndarray) -> float:
# #     if gt_01.shape[0] < 3:
# #         return 0.0
# #     v = lat_corrected_velocity(gt_01)
# #     total = 0.0
# #     for i in range(len(v) - 1):
# #         v1, v2 = v[i], v[i + 1]
# #         n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
# #         if n1 < 1e-8 or n2 < 1e-8:
# #             continue
# #         cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
# #         total += np.degrees(np.arccos(cos_a))
# #     return total


# # def classify(gt_01: np.ndarray, thr: float = RECURV_THR_DEG) -> str:
# #     return "recurvature" if total_rotation_angle(gt_01) >= thr else "straight"


# # def ate_cte(pred_01: np.ndarray, gt_01: np.ndarray,
# #             lon_idx: int = 0, lat_idx: int = 1
# #             ) -> Tuple[np.ndarray, np.ndarray]:
# #     T = pred_01.shape[0]
# #     ate_arr = np.zeros(T)
# #     cte_arr = np.zeros(T)
# #     km_per_01deg = 11.112

# #     for k in range(T):
# #         if k == 0:
# #             dk = gt_01[1] - gt_01[0] if T > 1 else np.array([1.0, 0.0])
# #         else:
# #             dk = gt_01[k] - gt_01[k - 1]

# #         lat_rad = np.deg2rad(gt_01[k, lat_idx] / 10.0)
# #         dk_km = dk * km_per_01deg
# #         dk_km[0] *= np.cos(lat_rad)

# #         norm_dk = np.linalg.norm(dk_km)
# #         if norm_dk < 1e-8:
# #             continue

# #         t_hat = dk_km / norm_dk
# #         n_hat = np.array([-t_hat[1], t_hat[0]])

# #         delta_pos = pred_01[k] - gt_01[k]
# #         delta_km  = delta_pos * km_per_01deg
# #         delta_km[0] *= np.cos(lat_rad)

# #         ate_arr[k] = float(np.dot(delta_km, t_hat))
# #         cte_arr[k] = float(np.dot(delta_km, n_hat))

# #     return ate_arr, cte_arr


# # def circular_std(angles_deg: np.ndarray) -> float:
# #     if len(angles_deg) == 0:
# #         return 0.0
# #     rads  = np.deg2rad(angles_deg)
# #     R_bar = np.abs(np.mean(np.exp(1j * rads)))
# #     R_bar = np.clip(R_bar, 1e-8, 1.0 - 1e-8)
# #     return float(np.degrees(np.sqrt(-2.0 * np.log(R_bar))))


# # def compute_csd(traj_01: np.ndarray) -> float:
# #     v = lat_corrected_velocity(traj_01)
# #     if len(v) == 0:
# #         return 0.0
# #     angles = np.arctan2(v[:, 1], v[:, 0])
# #     return circular_std(np.degrees(angles))


# # def oyr(pred_01: np.ndarray, gt_01: np.ndarray) -> float:
# #     pv = lat_corrected_velocity(pred_01)
# #     gv = lat_corrected_velocity(gt_01)
# #     m  = min(len(pv), len(gv))
# #     if m == 0:
# #         return 0.0
# #     dots  = np.sum(pv[:m] * gv[:m], axis=1)
# #     np_   = np.linalg.norm(pv[:m], axis=1)
# #     ng_   = np.linalg.norm(gv[:m], axis=1)
# #     valid = (np_ > 1e-8) & (ng_ > 1e-8)
# #     if valid.sum() == 0:
# #         return 0.0
# #     cos_vals = dots[valid] / (np_[valid] * ng_[valid])
# #     return float((cos_vals < 0).mean())


# # def hle(pred_01: np.ndarray, gt_01: np.ndarray) -> float:
# #     def curvature(traj):
# #         v = lat_corrected_velocity(traj)
# #         if len(v) < 2:
# #             return np.array([])
# #         cross = v[1:, 0] * v[:-1, 1] - v[1:, 1] * v[:-1, 0]
# #         n1    = np.linalg.norm(v[1:],  axis=1).clip(1e-8)
# #         n2    = np.linalg.norm(v[:-1], axis=1).clip(1e-8)
# #         return cross / (n1 * n2)
# #     kp = curvature(pred_01)
# #     kg = curvature(gt_01)
# #     m  = min(len(kp), len(kg))
# #     if m == 0:
# #         return float("nan")
# #     return float(np.mean(np.abs(kp[:m] - kg[:m])))


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  4. Tier 3 — Probabilistic
# # # ══════════════════════════════════════════════════════════════════════════════

# # def crps_2d(pred_ens_01: np.ndarray, gt_01: np.ndarray) -> float:
# #     """CRPS energy-form. Vectorised pairwise diversity."""
# #     S = pred_ens_01.shape[0]
# #     gt_rep  = gt_01[np.newaxis].repeat(S, axis=0)
# #     acc     = float(np.mean(haversine_km(pred_ens_01, gt_rep)))

# #     p_i = pred_ens_01[:, np.newaxis, :]
# #     p_j = pred_ens_01[np.newaxis, :, :]
# #     p_i_flat = np.broadcast_to(p_i, (S, S, pred_ens_01.shape[-1])).reshape(S * S, -1)
# #     p_j_flat = np.broadcast_to(p_j, (S, S, pred_ens_01.shape[-1])).reshape(S * S, -1)
# #     div = float(np.mean(haversine_km(p_i_flat, p_j_flat)))

# #     return float(acc - 0.5 * div)


# # def ssr_step(pred_ens_01: np.ndarray, gt_01: np.ndarray) -> float:
# #     """Spread-Skill Ratio. Return 1.0 when spread=rmse≈0."""
# #     S        = pred_ens_01.shape[0]
# #     ens_mean = pred_ens_01.mean(axis=0)
# #     ens_mean_rep = ens_mean[np.newaxis].repeat(S, axis=0)
# #     spread_vals  = haversine_km(pred_ens_01, ens_mean_rep)
# #     spread       = float(np.sqrt(np.mean(spread_vals ** 2)))
# #     rmse_em = float(haversine_km(ens_mean[np.newaxis], gt_01[np.newaxis])[0])
# #     if spread < 1e-6 and rmse_em < 1e-6:
# #         return 1.0
# #     return spread / (rmse_em + 1e-8)


# # def brier_skill_score(
# #     pred_ens_seqs: List[np.ndarray],
# #     gt_seqs:       List[np.ndarray],
# #     step:          int,
# #     target_deg:    Tuple[float, float],
# #     radius_km:     float = LANDFALL_RADIUS_KM,
# #     clim_rate:     Optional[float] = None,
# # ) -> float:
# #     N   = len(gt_seqs)
# #     if N == 0:
# #         return float("nan")
# #     bs        = 0.0
# #     clim_hits = 0
# #     target_01 = np.array([target_deg[0] * 10.0, target_deg[1] * 10.0])

# #     for i in range(N):
# #         if step >= len(gt_seqs[i]):
# #             continue
# #         gt_pos   = gt_seqs[i][step]
# #         ens_seqs = pred_ens_seqs[i]
# #         if ens_seqs.ndim == 3:
# #             S = ens_seqs.shape[0]
# #         else:
# #             continue
# #         obs = float(haversine_km(
# #             gt_pos[np.newaxis], target_01[np.newaxis])[0]) <= radius_km
# #         hits = sum(
# #             haversine_km(ens_seqs[s, min(step, ens_seqs.shape[1]-1)][np.newaxis],
# #                          target_01[np.newaxis])[0] <= radius_km
# #             for s in range(S)
# #         )
# #         p_fc       = hits / S
# #         bs        += (p_fc - float(obs)) ** 2
# #         clim_hits += float(obs)

# #     bs /= max(N, 1)
# #     p_clim  = (clim_rate if clim_rate is not None else clim_hits / max(N, 1))
# #     bs_clim = p_clim * (1.0 - p_clim) + 1e-8
# #     return float(1.0 - bs / bs_clim)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  5. Tier 4 — Geometric
# # # ══════════════════════════════════════════════════════════════════════════════

# # def dtw_haversine(s: np.ndarray, t: np.ndarray) -> float:
# #     n, m = len(s), len(t)
# #     dp   = np.full((n + 1, m + 1), np.inf)
# #     dp[0, 0] = 0.0
# #     for i in range(1, n + 1):
# #         for j in range(1, m + 1):
# #             cost   = float(haversine_km(s[i-1:i], t[j-1:j])[0])
# #             dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
# #     return float(dp[n, m])


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  6. Per-sequence result container
# # # ══════════════════════════════════════════════════════════════════════════════

# # @dataclass
# # class SequenceResult:
# #     ade:       float
# #     fde:       float
# #     per_step:  np.ndarray
# #     ate:       np.ndarray
# #     cte:       np.ndarray
# #     category:  str
# #     category_pred: str
# #     theta:     float
# #     csd_gt:    float
# #     oyr_val:   float
# #     hle_val:   float
# #     crps:      Optional[np.ndarray] = None
# #     ssr:       Optional[np.ndarray] = None
# #     dtw:       float = float("nan")
# #     loss_fm:      float = float("nan")
# #     loss_dir:     float = float("nan")
# #     loss_step:    float = float("nan")
# #     loss_disp:    float = float("nan")
# #     loss_heading: float = float("nan")
# #     loss_smooth:  float = float("nan")
# #     loss_pinn:    float = float("nan")
# #     loss_total:   float = float("nan")


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  7. Dataset-level aggregated metrics
# # # ══════════════════════════════════════════════════════════════════════════════

# # @dataclass
# # class DatasetMetrics:
# #     ade:            float = 0.0
# #     fde:            float = 0.0
# #     per_step_mean:  np.ndarray = field(default_factory=lambda: np.zeros(PRED_LEN))
# #     per_step_std:   np.ndarray = field(default_factory=lambda: np.zeros(PRED_LEN))
# #     ugde_12h:       float = 0.0
# #     ugde_24h:       float = 0.0
# #     ugde_48h:       float = 0.0
# #     ugde_72h:       float = 0.0
# #     tss_72h:        float = float("nan")
# #     ate_mean:       float = 0.0
# #     cte_mean:       float = 0.0
# #     ate_abs_mean:   float = 0.0
# #     cte_abs_mean:   float = 0.0
# #     ade_str:        float = float("nan")
# #     ade_rec:        float = float("nan")
# #     pr:             float = float("nan")
# #     rdr:            float = float("nan")
# #     n_str:          int   = 0
# #     n_rec:          int   = 0
# #     crps_mean:      float = float("nan")
# #     crps_72h:       float = float("nan")
# #     ssr_mean:       float = float("nan")
# #     bss_mean:       float = float("nan")
# #     dtw_mean:       float = float("nan")
# #     dtw_str:        float = float("nan")
# #     dtw_rec:        float = float("nan")
# #     oyr_mean:       float = float("nan")
# #     oyr_rec:        float = float("nan")
# #     hle_mean:       float = float("nan")
# #     hle_rec:        float = float("nan")
# #     loss_fm:        float = float("nan")
# #     loss_dir:       float = float("nan")
# #     loss_step:      float = float("nan")
# #     loss_disp:      float = float("nan")
# #     loss_heading:   float = float("nan")
# #     loss_smooth:    float = float("nan")
# #     loss_pinn:      float = float("nan")
# #     loss_total:     float = float("nan")
# #     n_total:        int   = 0
# #     timestamp:      str   = ""

# #     def summary(self) -> str:
# #         lines = [
# #             "═" * 64,
# #             "  FM+PINN TC Track Metrics  (4-tier)",
# #             "═" * 64,
# #             f"  Sequences : {self.n_total}"
# #             f"  (str={self.n_str}, rec={self.n_rec})",
# #             "",
# #             "  ── Tier 1: Position ─────────────────────────────────",
# #             f"  ADE        : {self.ade:.1f} km",
# #             f"  FDE (72h)  : {self.fde:.1f} km",
# #             f"  UGDE       : 12h={self.ugde_12h:.0f}  24h={self.ugde_24h:.0f}"
# #             f"  48h={self.ugde_48h:.0f}  72h={self.ugde_72h:.0f} km",
# #             "",
# #             "  ── Tier 2: Operational ──────────────────────────────",
# #             f"  TSS (72h)  : {self.tss_72h:.3f}" +
# #             (" ✅" if not np.isnan(self.tss_72h) and self.tss_72h > 0 else ""),
# #             f"  |ATE| mean : {self.ate_abs_mean:.1f} km",
# #             f"  |CTE| mean : {self.cte_abs_mean:.1f} km",
# #             f"  ADE_str    : {self.ade_str:.1f} km",
# #             f"  ADE_rec    : {self.ade_rec:.1f} km",
# #             f"  PR         : {self.pr:.2f}" +
# #             (" (no bias)" if not np.isnan(self.pr) and self.pr < 1.3 else
# #              " ⚠️ straight-track bias"),
# #             f"  RDR        : {self.rdr:.3f}" +
# #             (" ✅" if not np.isnan(self.rdr) and self.rdr > 0.5 else ""),
# #             "",
# #             "  ── Tier 3: Probabilistic ────────────────────────────",
# #             f"  CRPS mean  : {self.crps_mean:.1f} km",
# #             f"  CRPS 72h   : {self.crps_72h:.1f} km",
# #             f"  SSR mean   : {self.ssr_mean:.3f}  (1=calibrated)",
# #             f"  BSS mean   : {self.bss_mean:.3f}",
# #             "",
# #             "  ── Tier 4: Geometric ────────────────────────────────",
# #             f"  DTW mean   : {self.dtw_mean:.1f} km",
# #             f"  OYR mean   : {self.oyr_mean:.3f}  (0=ideal)",
# #             f"  OYR rec    : {self.oyr_rec:.3f}",
# #             f"  HLE mean   : {self.hle_mean:.4f}",
# #             f"  HLE rec    : {self.hle_rec:.4f}",
# #             "═" * 64,
# #         ]
# #         return "\n".join(lines)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  8. CSV export
# # # ══════════════════════════════════════════════════════════════════════════════

# # _CSV_FIELDS = [
# #     "timestamp", "n_total", "n_str", "n_rec",
# #     "ade", "fde", "ugde_12h", "ugde_24h", "ugde_48h", "ugde_72h",
# #     "tss_72h", "ate_abs_mean", "cte_abs_mean",
# #     "ade_str", "ade_rec", "pr", "rdr",
# #     "crps_mean", "crps_72h", "ssr_mean", "bss_mean",
# #     "dtw_mean", "dtw_str", "dtw_rec",
# #     "oyr_mean", "oyr_rec", "hle_mean", "hle_rec",
# #     "loss_total", "loss_fm", "loss_dir", "loss_step",
# #     "loss_disp", "loss_heading", "loss_smooth", "loss_pinn",
# # ]


# # def save_metrics_csv(metrics: DatasetMetrics, csv_path: str, tag: str = "") -> None:
# #     os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# #     write_header = not os.path.exists(csv_path)
# #     row: Dict[str, object] = {f: getattr(metrics, f, float("nan"))
# #                                for f in _CSV_FIELDS}
# #     if tag:
# #         row["timestamp"] = f"{tag}_{metrics.timestamp}"
# #     with open(csv_path, "a", newline="") as fh:
# #         writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
# #         if write_header:
# #             writer.writeheader()
# #         writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
# #                          for k, v in row.items()})


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  9. Main Evaluator
# # # ══════════════════════════════════════════════════════════════════════════════

# # class TCEvaluator:
# #     def __init__(
# #         self,
# #         pred_len:    int   = PRED_LEN,
# #         step_hours:  int   = STEP_HOURS,
# #         recurv_thr:  float = RECURV_THR_DEG,
# #         compute_dtw: bool  = True,
# #         cliper_ugde: Optional[Dict[int, float]] = None,
# #     ):
# #         self.pred_len    = pred_len
# #         self.step_hours  = step_hours
# #         self.recurv_thr  = recurv_thr
# #         self.compute_dtw = compute_dtw
# #         self.cliper_ugde = cliper_ugde or {}
# #         self._results:  List[SequenceResult] = []
# #         self._loss_buf: List[Dict[str, float]] = []

# #     def reset(self) -> None:
# #         self._results  = []
# #         self._loss_buf = []

# #     def update(
# #         self,
# #         pred_01:    np.ndarray,
# #         gt_01:      np.ndarray,
# #         pred_ens:   Optional[np.ndarray] = None,
# #         loss_dict:  Optional[Dict[str, float]] = None,
# #     ) -> None:
# #         T = min(len(pred_01), len(gt_01), self.pred_len)
# #         p = pred_01[:T]
# #         g = gt_01[:T]

# #         _ade, _fde, ps = ade_fde(p, g)
# #         _ate, _cte     = ate_cte(p, g)
# #         theta          = total_rotation_angle(g)
# #         cat            = "recurvature" if theta >= self.recurv_thr else "straight"
# #         theta_pred     = total_rotation_angle(p)
# #         cat_pred       = "recurvature" if theta_pred >= self.recurv_thr else "straight"
# #         csd_g          = compute_csd(g)
# #         _oyr           = oyr(p, g)
# #         _hle           = hle(p, g)

# #         crps_arr: Optional[np.ndarray] = None
# #         ssr_arr:  Optional[np.ndarray] = None
# #         if pred_ens is not None:
# #             crps_arr = np.array([
# #                 crps_2d(pred_ens[:, h, :] if pred_ens.ndim == 3 else pred_ens[:, h, :],
# #                         g[h]) for h in range(T)
# #             ])
# #             ssr_arr = np.array([
# #                 ssr_step(pred_ens[:, h, :] if pred_ens.ndim == 3 else pred_ens[:, h, :],
# #                          g[h]) for h in range(T)
# #             ])

# #         dtw_val = dtw_haversine(p, g) if self.compute_dtw else float("nan")

# #         r = SequenceResult(
# #             ade=_ade, fde=_fde, per_step=ps,
# #             ate=_ate, cte=_cte,
# #             category=cat, category_pred=cat_pred,
# #             theta=theta, csd_gt=csd_g,
# #             oyr_val=_oyr, hle_val=_hle,
# #             crps=crps_arr, ssr=ssr_arr, dtw=dtw_val,
# #         )
# #         if loss_dict:
# #             r.loss_fm      = loss_dict.get("fm",      float("nan"))
# #             r.loss_dir     = loss_dict.get("dir",     float("nan"))
# #             r.loss_step    = loss_dict.get("step",    float("nan"))
# #             r.loss_disp    = loss_dict.get("disp",    float("nan"))
# #             r.loss_heading = loss_dict.get("heading", float("nan"))
# #             r.loss_smooth  = loss_dict.get("smooth",  float("nan"))
# #             r.loss_pinn    = loss_dict.get("pinn",    float("nan"))
# #             r.loss_total   = loss_dict.get("total",   float("nan"))
# #         self._results.append(r)

# #     def update_batch(self, pred_norm, gt_norm,
# #                      loss_dict: Optional[Dict[str, float]] = None) -> None:
# #         pred_d = denorm_torch(pred_norm).cpu().numpy()
# #         gt_d   = denorm_torch(gt_norm).cpu().numpy()
# #         B = pred_d.shape[1]
# #         for b in range(B):
# #             self.update(pred_d[:, b, :], gt_d[:, b, :], loss_dict=loss_dict)

# #     def compute(self, tag: str = "") -> DatasetMetrics:
# #         if not self._results:
# #             return DatasetMetrics()

# #         rs  = self._results
# #         n   = len(rs)
# #         ts  = tag or datetime.now().strftime("%Y%m%d_%H%M%S")

# #         all_steps = np.stack([r.per_step[:self.pred_len] for r in rs])
# #         step_mean = all_steps.mean(0)
# #         step_std  = all_steps.std(0)

# #         def _h(step_idx):
# #             return float(step_mean[step_idx]) if step_idx < self.pred_len else float("nan")

# #         str_r = [r for r in rs if r.category == "straight"]
# #         rec_r = [r for r in rs if r.category == "recurvature"]
# #         ade_s = float(np.mean([r.ade for r in str_r])) if str_r else float("nan")
# #         ade_r = float(np.mean([r.ade for r in rec_r])) if rec_r else float("nan")
# #         pr_val = ade_r / (ade_s + 1e-8) if (str_r and rec_r) else float("nan")

# #         all_ate = np.concatenate([r.ate for r in rs])
# #         all_cte = np.concatenate([r.cte for r in rs])

# #         tss_val = float("nan")
# #         if self.cliper_ugde and 72 in self.cliper_ugde:
# #             ugde_72 = _h(HORIZON_STEPS[72])
# #             if not np.isnan(ugde_72):
# #                 tss_val = tss(ugde_72, self.cliper_ugde[72])

# #         crps_seqs = [r.crps for r in rs if r.crps is not None]
# #         ssr_seqs  = [r.ssr  for r in rs if r.ssr  is not None]

# #         crps_mean = float("nan")
# #         crps_72h  = float("nan")
# #         ssr_mean  = float("nan")
# #         if crps_seqs:
# #             crps_mat  = np.stack(crps_seqs)
# #             crps_mean = float(crps_mat.mean())
# #             step_72   = HORIZON_STEPS.get(72, -1)
# #             if 0 <= step_72 < crps_mat.shape[1]:
# #                 crps_72h = float(crps_mat[:, step_72].mean())
# #         if ssr_seqs:
# #             ssr_mat  = np.stack(ssr_seqs)
# #             valid    = ssr_mat[~np.isnan(ssr_mat)]
# #             ssr_mean = float(valid.mean()) if len(valid) > 0 else float("nan")

# #         dtw_all = [r.dtw for r in rs  if not np.isnan(r.dtw)]
# #         dtw_s   = [r.dtw for r in str_r if not np.isnan(r.dtw)]
# #         dtw_rc  = [r.dtw for r in rec_r if not np.isnan(r.dtw)]
# #         oyr_all = [r.oyr_val for r in rs]
# #         oyr_rc  = [r.oyr_val for r in rec_r]
# #         hle_all = [r.hle_val for r in rs  if not np.isnan(r.hle_val)]
# #         hle_rc  = [r.hle_val for r in rec_r if not np.isnan(r.hle_val)]

# #         def _mean_loss(attr):
# #             vals = [getattr(r, attr) for r in rs if not np.isnan(getattr(r, attr))]
# #             return float(np.mean(vals)) if vals else float("nan")

# #         if rec_r:
# #             rdr_num = sum(1 for r in rec_r if r.category_pred == "recurvature")
# #             rdr_val = rdr_num / len(rec_r)
# #         else:
# #             rdr_val = float("nan")

# #         return DatasetMetrics(
# #             ade           = float(np.mean([r.ade for r in rs])),
# #             fde           = float(np.mean([r.fde for r in rs])),
# #             per_step_mean = step_mean,
# #             per_step_std  = step_std,
# #             ugde_12h      = _h(HORIZON_STEPS[12]),
# #             ugde_24h      = _h(HORIZON_STEPS[24]),
# #             ugde_48h      = _h(HORIZON_STEPS[48]),
# #             ugde_72h      = _h(HORIZON_STEPS[72]),
# #             tss_72h       = tss_val,
# #             ate_mean      = float(np.mean(all_ate)),
# #             cte_mean      = float(np.mean(all_cte)),
# #             ate_abs_mean  = float(np.mean(np.abs(all_ate))),
# #             cte_abs_mean  = float(np.mean(np.abs(all_cte))),
# #             ade_str       = ade_s,
# #             ade_rec       = ade_r,
# #             pr            = pr_val,
# #             rdr           = rdr_val,
# #             n_str         = len(str_r),
# #             n_rec         = len(rec_r),
# #             crps_mean     = crps_mean,
# #             crps_72h      = crps_72h,
# #             ssr_mean      = ssr_mean,
# #             bss_mean      = float("nan"),
# #             dtw_mean      = float(np.mean(dtw_all)) if dtw_all else float("nan"),
# #             dtw_str       = float(np.mean(dtw_s))   if dtw_s   else float("nan"),
# #             dtw_rec       = float(np.mean(dtw_rc))  if dtw_rc  else float("nan"),
# #             oyr_mean      = float(np.mean(oyr_all)) if oyr_all else float("nan"),
# #             oyr_rec       = float(np.mean(oyr_rc))  if oyr_rc  else float("nan"),
# #             hle_mean      = float(np.mean(hle_all)) if hle_all else float("nan"),
# #             hle_rec       = float(np.mean(hle_rc))  if hle_rc  else float("nan"),
# #             loss_fm       = _mean_loss("loss_fm"),
# #             loss_dir      = _mean_loss("loss_dir"),
# #             loss_step     = _mean_loss("loss_step"),
# #             loss_disp     = _mean_loss("loss_disp"),
# #             loss_heading  = _mean_loss("loss_heading"),
# #             loss_smooth   = _mean_loss("loss_smooth"),
# #             loss_pinn     = _mean_loss("loss_pinn"),
# #             loss_total    = _mean_loss("loss_total"),
# #             n_total       = n,
# #             timestamp     = ts,
# #         )

# #     def save_csv(self, csv_path: str, tag: str = "") -> None:
# #         m = self.compute(tag=tag)
# #         save_metrics_csv(m, csv_path, tag=tag)
# #         print(f"  📊  Metrics → {csv_path}")


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  10. Fast Tier-1 accumulator  ── FIX-MET-8/9
# # # ══════════════════════════════════════════════════════════════════════════════

# # class StepErrorAccumulator:
# #     """
# #     FIX-MET-8: Pad dist_km [T_active, B] → [pred_len, B] với zeros.
# #     FIX-MET-9: Dùng per-step _count array thay vì _active_max scalar.
# #                ADE chỉ tính trên steps có count > 0 (actual data steps).
# #                active_steps = số steps có data thật sự.
# #     """

# #     def __init__(self, pred_len: int = PRED_LEN, step_hours: int = STEP_HOURS):
# #         self.pred_len   = pred_len
# #         self.step_hours = step_hours
# #         self.reset()

# #     def reset(self) -> None:
# #         self._sum    = np.zeros(self.pred_len, dtype=np.float64)
# #         self._sum_sq = np.zeros(self.pred_len, dtype=np.float64)
# #         self._count  = np.zeros(self.pred_len, dtype=np.int64)

# #     def update(self, dist_km) -> None:
# #         if HAS_TORCH and torch.is_tensor(dist_km):
# #             d = dist_km.double().cpu().numpy()
# #         else:
# #             d = np.asarray(dist_km, dtype=np.float64)

# #         if d.ndim == 1:
# #             d = d.reshape(-1, 1)
# #         elif d.ndim != 2:
# #             return

# #         T_actual, B = d.shape

# #         # FIX-MET-8: pad to pred_len
# #         if T_actual < self.pred_len:
# #             pad    = np.zeros((self.pred_len - T_actual, B), dtype=np.float64)
# #             d_full = np.concatenate([d, pad], axis=0)
# #         else:
# #             d_full   = d[:self.pred_len]
# #             T_actual = self.pred_len

# #         self._sum    += d_full.sum(axis=1)
# #         self._sum_sq += (d_full ** 2).sum(axis=1)

# #         # FIX-MET-9: only increment count for steps with real data
# #         self._count[:T_actual] += B

# #     def compute(self) -> Dict:
# #         if self._count[0] == 0:
# #             return {}

# #         count_safe = np.where(self._count > 0, self._count, 1)
# #         ps  = self._sum    / count_safe
# #         ps2 = self._sum_sq / count_safe
# #         ps  = np.where(self._count > 0, ps,  0.0)
# #         ps2 = np.where(self._count > 0, ps2, 0.0)
# #         std = np.sqrt(np.maximum(ps2 - ps ** 2, 0.0))

# #         # FIX-MET-9: active_steps = max step index with any data
# #         active = int(np.sum(self._count > 0))   # number of steps with data
# #         if active == 0:
# #             return {}

# #         ade_val = float(ps[:active].mean())
# #         fde_val = float(ps[active - 1])

# #         out: Dict = {
# #             "per_step":     ps,
# #             "per_step_std": std,
# #             "ADE":          ade_val,
# #             "FDE":          fde_val,
# #             "active_steps": active,
# #             "n_samples":    int(self._count[0]),
# #         }
# #         for h, s in HORIZON_STEPS.items():
# #             if s < self.pred_len and self._count[s] > 0:
# #                 out[f"{h}h"]     = float(ps[s])
# #                 out[f"{h}h_std"] = float(std[s])
# #             else:
# #                 out[f"{h}h"]     = float("nan")
# #                 out[f"{h}h_std"] = float("nan")
# #         return out


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  11. Baseline helpers
# # # ══════════════════════════════════════════════════════════════════════════════

# # def cliper_errors(obs_seqs, gt_seqs, pred_len: int) -> Tuple[float, np.ndarray]:
# #     """Compute CLIPER errors for all sequences. Returns (mean_ade, [N, pred_len])."""
# #     all_errs = []
# #     for obs, gt in zip(obs_seqs, gt_seqs):
# #         errs = np.zeros(pred_len)
# #         for h in range(pred_len):
# #             pred_c = cliper_forecast(np.array(obs), h + 1)
# #             errs[h] = float(haversine_km(pred_c[np.newaxis],
# #                                          np.array(gt)[h:h+1], unit_01deg=True)[0])
# #         all_errs.append(errs)
# #     mat = np.stack(all_errs)
# #     return float(mat.mean()), mat


# # def persistence_errors(obs_seqs, gt_seqs, pred_len: int) -> np.ndarray:
# #     """Persistence baseline errors [N, pred_len]."""
# #     all_errs = []
# #     for obs, gt in zip(obs_seqs, gt_seqs):
# #         last = np.array(obs)[-1, :2]
# #         errs = np.array([float(haversine_km(last[np.newaxis],
# #                                             np.array(gt)[h:h+1], unit_01deg=True)[0])
# #                          for h in range(pred_len)])
# #         all_errs.append(errs)
# #     return np.stack(all_errs)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  12. Backward-compatible wrappers
# # # ══════════════════════════════════════════════════════════════════════════════

# # def RSE(pred, true):
# #     return float(np.sqrt(np.sum((true - pred) ** 2))
# #                  / (np.sqrt(np.sum((true - true.mean()) ** 2)) + 1e-8))

# # def CORR(pred, true):
# #     u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
# #     d = np.sqrt(
# #         ((true - true.mean(0)) ** 2).sum(0)
# #         * ((pred - pred.mean(0)) ** 2).sum(0)
# #     ) + 1e-8
# #     return u / d

# # def MAE(pred, true):  return float(np.mean(np.abs(pred - true)))
# # def MSE(pred, true):  return float(np.mean((pred - true) ** 2))
# # def RMSE(pred, true): return float(np.sqrt(MSE(pred, true)))
# # def MAPE(pred, true): return float(np.mean(np.abs((pred - true) / (np.abs(true) + 1e-5))))
# # def MSPE(pred, true): return float(np.mean(np.square((pred - true) / (np.abs(true) + 1e-5))))

# # def metric(pred, true):
# #     return (MAE(pred, true), MSE(pred, true), RMSE(pred, true),
# #             MAPE(pred, true), MSPE(pred, true),
# #             RSE(pred, true),  CORR(pred, true))

# """
# utils/metrics.py — v7
# =============================
# THÊM MỚI so với v6:

#   FIX-MET-12  StepErrorAccumulator thêm ATE/CTE per-step tracking
#               để monitor progress so với ST-Trans (ATE=79.94, CTE=93.58)

#   FIX-MET-13  _composite_score_v2: normalize theo ST-Trans targets
#               ATE và CTE được weight vào score để guide model

#   FIX-MET-14  haversine_km_torch_with_atecte: trả về (dist, ate, cte)
#               trong một pass để tránh tính lại

#   GIỮA v6 các fix cũ:
#   FIX-MET-9  per-step count array
#   FIX-MET-10 haversine convention
#   FIX-MET-11 cliper_forecast robust
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

# CLIPER_ALPHA = {h: h / 12 for h in range(1, 13)}
# HORIZON_STEPS: Dict[int, int] = {6: 0, 12: 1, 24: 3, 36: 5, 48: 7, 60: 9, 72: 11}
# RECURV_THR_DEG = 45.0

# LANDFALL_TARGETS = [
#     ("DaNang",    108.2, 16.1),
#     ("Manila",    121.0, 14.6),
#     ("Taipei",    121.5, 25.0),
#     ("Shanghai",  121.5, 31.2),
#     ("Okinawa",   127.8, 26.3),
# ]
# LANDFALL_RADIUS_KM = 300.0

# # ── ST-Trans paper targets (để beat) ─────────────────────────────────────────
# ST_TRANS_TARGETS = {
#     "mean_dpe": 136.41,
#     "ate":       79.94,
#     "cte":       93.58,
#     "72h":      300.0,   # ước tính từ final-step DPE≈297km
#     "12h":       50.0,
#     "24h":      100.0,
#     "48h":      200.0,
# }


# # ══════════════════════════════════════════════════════════════════════════════
# #  1. Primitives
# # ══════════════════════════════════════════════════════════════════════════════

# def haversine_km(
#     p1: np.ndarray,
#     p2: np.ndarray,
#     lon_idx: int = 0,
#     lat_idx: int = 1,
#     unit_01deg: bool = True,
# ) -> np.ndarray:
#     scale = 10.0 if unit_01deg else 1.0
#     lat1 = np.deg2rad(p1[..., lat_idx] / scale)
#     lat2 = np.deg2rad(p2[..., lat_idx] / scale)
#     dlon = np.deg2rad((p2[..., lon_idx] - p1[..., lon_idx]) / scale)
#     dlat = np.deg2rad((p2[..., lat_idx] - p1[..., lat_idx]) / scale)
#     a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
#     return 2.0 * R_EARTH_KM * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


# def haversine_km_np(p1, p2, unit_01deg=True):
#     return haversine_km(p1, p2, unit_01deg=unit_01deg)


# def haversine_km_torch(pred, gt, lon_idx=0, lat_idx=1, unit_01deg=True):
#     scale = 10.0 if unit_01deg else 1.0
#     lat1  = torch.deg2rad(pred[..., lat_idx] / scale)
#     lat2  = torch.deg2rad(gt[..., lat_idx]   / scale)
#     dlon  = torch.deg2rad((pred[..., lon_idx] - gt[..., lon_idx]) / scale)
#     dlat  = torch.deg2rad((pred[..., lat_idx] - gt[..., lat_idx]) / scale)
#     a = (torch.sin(dlat / 2.0) ** 2
#          + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2)
#     return 2.0 * R_EARTH_KM * torch.asin(a.clamp(0.0, 1.0).sqrt())


# def haversine_and_atecte_torch(
#     pred_norm: "torch.Tensor",
#     gt_norm: "torch.Tensor",
#     unit_01deg: bool = True,
# ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
#     """
#     FIX-MET-14: Tính haversine dist + ATE + CTE trong một pass.
#     Inputs: [T, B, 2] normalized coords
#     Returns: dist[T,B], ate[T,B], cte[T,B] — all in km
#     """
#     # Convert to degrees
#     scale = 10.0 if unit_01deg else 1.0
#     if unit_01deg:
#         pred_lon = (pred_norm[..., 0] * 50.0 + 1800.0) / 10.0
#         pred_lat = (pred_norm[..., 1] * 50.0) / 10.0
#         gt_lon   = (gt_norm[..., 0] * 50.0 + 1800.0) / 10.0
#         gt_lat   = (gt_norm[..., 1] * 50.0) / 10.0
#     else:
#         pred_lon, pred_lat = pred_norm[..., 0], pred_norm[..., 1]
#         gt_lon,   gt_lat   = gt_norm[..., 0],   gt_norm[..., 1]

#     # Haversine distance
#     lat1r = torch.deg2rad(pred_lat)
#     lat2r = torch.deg2rad(gt_lat)
#     dlon  = torch.deg2rad(gt_lon - pred_lon)
#     dlat  = torch.deg2rad(gt_lat - pred_lat)
#     a = (torch.sin(dlat / 2.0).pow(2)
#          + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2.0).pow(2))
#     dist = 2.0 * R_EARTH_KM * torch.asin(a.clamp(0.0, 1.0).sqrt())  # [T, B]

#     # ATE/CTE decomposition
#     DEG_KM = 111.0
#     T = pred_norm.shape[0]

#     # Track direction từ gt step-to-step
#     if T >= 2:
#         # gt direction at each step
#         gt_dx = torch.zeros_like(gt_lon)
#         gt_dy = torch.zeros_like(gt_lat)
#         # Forward difference for most steps, backward for last
#         gt_dx[:-1] = gt_lon[1:] - gt_lon[:-1]
#         gt_dy[:-1] = gt_lat[1:] - gt_lat[:-1]
#         gt_dx[-1]  = gt_dx[-2]
#         gt_dy[-1]  = gt_dy[-2]

#         # Convert to km with lat correction
#         cos_lat = torch.cos(torch.deg2rad(gt_lat)).clamp(min=1e-4)
#         track_x = gt_dx * cos_lat * DEG_KM   # [T, B] km east
#         track_y = gt_dy * DEG_KM             # [T, B] km north
#         track_norm = (track_x.pow(2) + track_y.pow(2)).sqrt().clamp(min=1e-4)

#         # Unit vectors
#         u_x = track_x / track_norm
#         u_y = track_y / track_norm
#         n_x = -u_y   # perpendicular
#         n_y =  u_x

#         # Error in km
#         err_lon = (pred_lon - gt_lon) * cos_lat * DEG_KM
#         err_lat = (pred_lat - gt_lat) * DEG_KM

#         ate = (err_lon * u_x + err_lat * u_y).abs()   # [T, B]
#         cte = (err_lon * n_x + err_lat * n_y).abs()   # [T, B]
#     else:
#         ate = dist * 0.5
#         cte = dist * 0.5

#     return dist, ate, cte


# def denorm_np(n: np.ndarray) -> np.ndarray:
#     r = n.copy()
#     r[..., 0] = n[..., 0] * 50.0 + 1800.0
#     r[..., 1] = n[..., 1] * 50.0
#     return r


# def denorm_torch(n):
#     r = n.clone()
#     r[..., 0] = n[..., 0] * 50.0 + 1800.0
#     r[..., 1] = n[..., 1] * 50.0
#     return r


# def denorm_deg_np(arr_norm: np.ndarray) -> np.ndarray:
#     out = arr_norm.copy()
#     out[..., 0] = (arr_norm[..., 0] * 50.0 + 1800.0) / 10.0
#     out[..., 1] = (arr_norm[..., 1] * 50.0) / 10.0
#     return out


# def cliper_forecast(obs_norm: np.ndarray, h: int) -> np.ndarray:
#     if obs_norm.shape[0] < 2:
#         return obs_norm[-1, :2].copy()
#     v = obs_norm[-1, :2] - obs_norm[-2, :2]
#     return obs_norm[-1, :2] + h * v


# # ══════════════════════════════════════════════════════════════════════════════
# #  2. Tier 1
# # ══════════════════════════════════════════════════════════════════════════════

# def ugde(pred_01: np.ndarray, gt_01: np.ndarray) -> np.ndarray:
#     return haversine_km(pred_01, gt_01)


# def ade_fde(pred_01: np.ndarray, gt_01: np.ndarray) -> Tuple[float, float, np.ndarray]:
#     ps = ugde(pred_01, gt_01)
#     return float(ps.mean()), float(ps[-1]), ps


# # ══════════════════════════════════════════════════════════════════════════════
# #  3-5. Tier 2-4 (giữ từ v6)
# # ══════════════════════════════════════════════════════════════════════════════

# def tss(ugde_model: float, ugde_cliper: float) -> float:
#     return 1.0 - ugde_model / (ugde_cliper + 1e-8)


# def lat_corrected_velocity(traj_01, lon_idx=0, lat_idx=1):
#     lats_rad = np.deg2rad(traj_01[:, lat_idx] / 10.0)
#     cos_lat  = np.cos(lats_rad[:-1])
#     dlat = np.diff(traj_01[:, lat_idx])
#     dlon = np.diff(traj_01[:, lon_idx]) * cos_lat
#     return np.stack([dlon, dlat], axis=-1)


# def total_rotation_angle(gt_01: np.ndarray) -> float:
#     if gt_01.shape[0] < 3: return 0.0
#     v = lat_corrected_velocity(gt_01)
#     total = 0.0
#     for i in range(len(v) - 1):
#         v1, v2 = v[i], v[i + 1]
#         n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
#         if n1 < 1e-8 or n2 < 1e-8: continue
#         cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
#         total += np.degrees(np.arccos(cos_a))
#     return total


# def classify(gt_01, thr=RECURV_THR_DEG):
#     return "recurvature" if total_rotation_angle(gt_01) >= thr else "straight"


# def ate_cte(pred_01: np.ndarray, gt_01: np.ndarray,
#             lon_idx=0, lat_idx=1) -> Tuple[np.ndarray, np.ndarray]:
#     T = pred_01.shape[0]
#     ate_arr = np.zeros(T)
#     cte_arr = np.zeros(T)
#     km_per_01deg = 11.112
#     for k in range(T):
#         if k == 0:
#             dk = gt_01[1] - gt_01[0] if T > 1 else np.array([1.0, 0.0])
#         else:
#             dk = gt_01[k] - gt_01[k - 1]
#         lat_rad = np.deg2rad(gt_01[k, lat_idx] / 10.0)
#         dk_km = dk * km_per_01deg
#         dk_km[0] *= np.cos(lat_rad)
#         norm_dk = np.linalg.norm(dk_km)
#         if norm_dk < 1e-8: continue
#         t_hat = dk_km / norm_dk
#         n_hat = np.array([-t_hat[1], t_hat[0]])
#         delta_pos = pred_01[k] - gt_01[k]
#         delta_km  = delta_pos * km_per_01deg
#         delta_km[0] *= np.cos(lat_rad)
#         ate_arr[k] = float(np.dot(delta_km, t_hat))
#         cte_arr[k] = float(np.dot(delta_km, n_hat))
#     return ate_arr, cte_arr


# def circular_std(angles_deg: np.ndarray) -> float:
#     if len(angles_deg) == 0: return 0.0
#     rads  = np.deg2rad(angles_deg)
#     R_bar = np.abs(np.mean(np.exp(1j * rads)))
#     R_bar = np.clip(R_bar, 1e-8, 1.0 - 1e-8)
#     return float(np.degrees(np.sqrt(-2.0 * np.log(R_bar))))


# def compute_csd(traj_01):
#     v = lat_corrected_velocity(traj_01)
#     if len(v) == 0: return 0.0
#     angles = np.arctan2(v[:, 1], v[:, 0])
#     return circular_std(np.degrees(angles))


# def oyr(pred_01, gt_01):
#     pv = lat_corrected_velocity(pred_01)
#     gv = lat_corrected_velocity(gt_01)
#     m  = min(len(pv), len(gv))
#     if m == 0: return 0.0
#     dots  = np.sum(pv[:m] * gv[:m], axis=1)
#     np_   = np.linalg.norm(pv[:m], axis=1)
#     ng_   = np.linalg.norm(gv[:m], axis=1)
#     valid = (np_ > 1e-8) & (ng_ > 1e-8)
#     if valid.sum() == 0: return 0.0
#     cos_vals = dots[valid] / (np_[valid] * ng_[valid])
#     return float((cos_vals < 0).mean())


# def hle(pred_01, gt_01):
#     def curvature(traj):
#         v = lat_corrected_velocity(traj)
#         if len(v) < 2: return np.array([])
#         cross = v[1:, 0] * v[:-1, 1] - v[1:, 1] * v[:-1, 0]
#         n1 = np.linalg.norm(v[1:],  axis=1).clip(1e-8)
#         n2 = np.linalg.norm(v[:-1], axis=1).clip(1e-8)
#         return cross / (n1 * n2)
#     kp = curvature(pred_01); kg = curvature(gt_01)
#     m  = min(len(kp), len(kg))
#     if m == 0: return float("nan")
#     return float(np.mean(np.abs(kp[:m] - kg[:m])))


# # ══════════════════════════════════════════════════════════════════════════════
# #  Composite Score v2 — normalize theo ST-Trans targets
# # ══════════════════════════════════════════════════════════════════════════════

# def _composite_score_v2(result: dict) -> float:
#     """
#     Score mới: normalize theo ST-Trans targets.
#     Lower is better.

#     Targets từ ST-Trans paper:
#       Mean DPE = 136.41 km → normalize /136
#       ATE      = 79.94 km  → normalize /80
#       CTE      = 93.58 km  → normalize /94
#       72h      ≈ 300 km    → normalize /300

#     Weight:
#       ATE và CTE được weight cao để model focus vào geometry
#       72h vẫn là priority cao nhất
#     """
#     ade = result.get("ADE", float("inf"))
#     h12 = result.get("12h", float("inf"))
#     h24 = result.get("24h", float("inf"))
#     h48 = result.get("48h", float("inf"))
#     h72 = result.get("72h", float("inf"))
#     # ATE/CTE nếu có
#     ate = result.get("ATE_mean", float("inf"))
#     cte = result.get("CTE_mean", float("inf"))

#     # Nếu không có ATE/CTE, dùng fallback
#     if not np.isfinite(ate): ate = ade * 0.45
#     if not np.isfinite(cte): cte = ade * 0.52

#     score = (
#         0.05 * (ade / 136.0)  # ADE baseline
#         + 0.05 * (h12 / 50.0)
#         + 0.10 * (h24 / 100.0)
#         + 0.15 * (h48 / 200.0)
#         + 0.35 * (h72 / 300.0)   # 72h — most important
#         + 0.15 * (ate / 80.0)    # ATE — need to beat 79.94
#         + 0.15 * (cte / 94.0)    # CTE — need to beat 93.58
#     )
#     return score * 100.0


# def _composite_score(result: dict) -> float:
#     """Original composite score (backward compat)."""
#     ade = result.get("ADE", float("inf"))
#     h12 = result.get("12h", float("inf"))
#     h24 = result.get("24h", float("inf"))
#     h48 = result.get("48h", float("inf"))
#     h72 = result.get("72h", float("inf"))
#     return 0.10 * ade + 0.10 * h12 + 0.15 * h24 + 0.25 * h48 + 0.40 * h72


# # ══════════════════════════════════════════════════════════════════════════════
# #  Fast Tier-1 accumulator với ATE/CTE — FIX-MET-12
# # ══════════════════════════════════════════════════════════════════════════════

# class StepErrorAccumulator:
#     """
#     FIX-MET-12: Thêm ATE/CTE per-step tracking.
#     Dùng haversine_and_atecte_torch khi available.
#     """

#     def __init__(self, pred_len=PRED_LEN, step_hours=STEP_HOURS):
#         self.pred_len   = pred_len
#         self.step_hours = step_hours
#         self.reset()

#     def reset(self):
#         self._sum     = np.zeros(self.pred_len, dtype=np.float64)
#         self._sum_sq  = np.zeros(self.pred_len, dtype=np.float64)
#         self._count   = np.zeros(self.pred_len, dtype=np.int64)
#         # ATE/CTE tracking
#         self._ate_sum = np.zeros(self.pred_len, dtype=np.float64)
#         self._cte_sum = np.zeros(self.pred_len, dtype=np.float64)
#         self._atecte_count = np.zeros(self.pred_len, dtype=np.int64)

#     def update(self, dist_km, ate_km=None, cte_km=None) -> None:
#         if HAS_TORCH and torch.is_tensor(dist_km):
#             d = dist_km.double().cpu().numpy()
#         else:
#             d = np.asarray(dist_km, dtype=np.float64)

#         if d.ndim == 1: d = d.reshape(-1, 1)
#         elif d.ndim != 2: return

#         T_actual, B = d.shape
#         if T_actual < self.pred_len:
#             pad = np.zeros((self.pred_len - T_actual, B), dtype=np.float64)
#             d_full = np.concatenate([d, pad], axis=0)
#         else:
#             d_full   = d[:self.pred_len]
#             T_actual = self.pred_len

#         self._sum    += d_full.sum(axis=1)
#         self._sum_sq += (d_full ** 2).sum(axis=1)
#         self._count[:T_actual] += B

#         # ATE/CTE update
#         if ate_km is not None and cte_km is not None:
#             if HAS_TORCH and torch.is_tensor(ate_km):
#                 ate_np = ate_km.double().cpu().numpy()
#                 cte_np = cte_km.double().cpu().numpy()
#             else:
#                 ate_np = np.asarray(ate_km, dtype=np.float64)
#                 cte_np = np.asarray(cte_km, dtype=np.float64)
#             if ate_np.ndim == 1: ate_np = ate_np.reshape(-1, 1)
#             if cte_np.ndim == 1: cte_np = cte_np.reshape(-1, 1)
#             T_ac = min(ate_np.shape[0], self.pred_len)
#             if ate_np.shape[0] < self.pred_len:
#                 pad_a = np.zeros((self.pred_len - T_ac, B), dtype=np.float64)
#                 pad_c = np.zeros((self.pred_len - T_ac, B), dtype=np.float64)
#                 ate_full = np.concatenate([ate_np[:T_ac], pad_a], axis=0)
#                 cte_full = np.concatenate([cte_np[:T_ac], pad_c], axis=0)
#             else:
#                 ate_full = ate_np[:self.pred_len]
#                 cte_full = cte_np[:self.pred_len]
#                 T_ac = self.pred_len
#             self._ate_sum[:T_ac] += ate_full[:T_ac].sum(axis=1)
#             self._cte_sum[:T_ac] += cte_full[:T_ac].sum(axis=1)
#             self._atecte_count[:T_ac] += B

#     def compute(self) -> Dict:
#         if self._count[0] == 0:
#             return {}

#         count_safe = np.where(self._count > 0, self._count, 1)
#         ps  = self._sum    / count_safe
#         ps2 = self._sum_sq / count_safe
#         ps  = np.where(self._count > 0, ps,  0.0)
#         ps2 = np.where(self._count > 0, ps2, 0.0)
#         std = np.sqrt(np.maximum(ps2 - ps ** 2, 0.0))

#         active   = int(np.sum(self._count > 0))
#         if active == 0: return {}

#         ade_val = float(ps[:active].mean())
#         fde_val = float(ps[active - 1])

#         out: Dict = {
#             "per_step":     ps,
#             "per_step_std": std,
#             "ADE":          ade_val,
#             "FDE":          fde_val,
#             "active_steps": active,
#             "n_samples":    int(self._count[0]),
#         }

#         # ATE/CTE means
#         if self._atecte_count[0] > 0:
#             ac_count_safe = np.where(self._atecte_count > 0, self._atecte_count, 1)
#             ate_ps = np.where(self._atecte_count > 0,
#                                self._ate_sum / ac_count_safe, 0.0)
#             cte_ps = np.where(self._atecte_count > 0,
#                                self._cte_sum / ac_count_safe, 0.0)
#             out["ATE_mean"] = float(ate_ps[:active].mean())
#             out["CTE_mean"] = float(cte_ps[:active].mean())
#             # Per-horizon ATE/CTE
#             for h, s in HORIZON_STEPS.items():
#                 if s < self.pred_len and self._atecte_count[s] > 0:
#                     out[f"ATE_{h}h"] = float(ate_ps[s])
#                     out[f"CTE_{h}h"] = float(cte_ps[s])

#         for h, s in HORIZON_STEPS.items():
#             if s < self.pred_len and self._count[s] > 0:
#                 out[f"{h}h"]     = float(ps[s])
#                 out[f"{h}h_std"] = float(std[s])
#             else:
#                 out[f"{h}h"]     = float("nan")
#                 out[f"{h}h_std"] = float("nan")

#         # Composite scores
#         out["composite_score"]    = _composite_score(out)
#         out["composite_score_v2"] = _composite_score_v2(out)

#         return out


# # ══════════════════════════════════════════════════════════════════════════════
# #  Các class/function còn lại (giữ từ v6)
# # ══════════════════════════════════════════════════════════════════════════════

# @dataclass
# class SequenceResult:
#     ade:       float
#     fde:       float
#     per_step:  np.ndarray
#     ate:       np.ndarray
#     cte:       np.ndarray
#     category:  str
#     category_pred: str
#     theta:     float
#     csd_gt:    float
#     oyr_val:   float
#     hle_val:   float
#     crps:      Optional[np.ndarray] = None
#     ssr:       Optional[np.ndarray] = None
#     dtw:       float = float("nan")
#     loss_fm:      float = float("nan")
#     loss_dir:     float = float("nan")
#     loss_step:    float = float("nan")
#     loss_disp:    float = float("nan")
#     loss_heading: float = float("nan")
#     loss_smooth:  float = float("nan")
#     loss_pinn:    float = float("nan")
#     loss_total:   float = float("nan")


# @dataclass
# class DatasetMetrics:
#     ade:            float = 0.0
#     fde:            float = 0.0
#     per_step_mean:  np.ndarray = field(default_factory=lambda: np.zeros(PRED_LEN))
#     per_step_std:   np.ndarray = field(default_factory=lambda: np.zeros(PRED_LEN))
#     ugde_12h:       float = 0.0
#     ugde_24h:       float = 0.0
#     ugde_48h:       float = 0.0
#     ugde_72h:       float = 0.0
#     tss_72h:        float = float("nan")
#     ate_mean:       float = 0.0
#     cte_mean:       float = 0.0
#     ate_abs_mean:   float = 0.0
#     cte_abs_mean:   float = 0.0
#     ade_str:        float = float("nan")
#     ade_rec:        float = float("nan")
#     pr:             float = float("nan")
#     rdr:            float = float("nan")
#     n_str:          int   = 0
#     n_rec:          int   = 0
#     crps_mean:      float = float("nan")
#     crps_72h:       float = float("nan")
#     ssr_mean:       float = float("nan")
#     bss_mean:       float = float("nan")
#     dtw_mean:       float = float("nan")
#     dtw_str:        float = float("nan")
#     dtw_rec:        float = float("nan")
#     oyr_mean:       float = float("nan")
#     oyr_rec:        float = float("nan")
#     hle_mean:       float = float("nan")
#     hle_rec:        float = float("nan")
#     loss_fm:        float = float("nan")
#     loss_dir:       float = float("nan")
#     loss_step:      float = float("nan")
#     loss_disp:      float = float("nan")
#     loss_heading:   float = float("nan")
#     loss_smooth:    float = float("nan")
#     loss_pinn:      float = float("nan")
#     loss_total:     float = float("nan")
#     n_total:        int   = 0
#     timestamp:      str   = ""

#     def summary(self) -> str:
#         st_targets = ST_TRANS_TARGETS
#         lines = [
#             "═" * 70,
#             "  FM+PINN TC Track Metrics  (v7 — vs ST-Trans targets)",
#             "═" * 70,
#             f"  Sequences : {self.n_total} (str={self.n_str}, rec={self.n_rec})",
#             "",
#             "  ── Tier 1: Position ───────────────────────────────────────",
#             f"  ADE        : {self.ade:.1f} km  (ST-Trans DPE=136.41) {'✅' if self.ade < 136.41 else '❌'}",
#             f"  FDE (72h)  : {self.fde:.1f} km",
#             f"  UGDE       : 12h={self.ugde_12h:.0f}  24h={self.ugde_24h:.0f}"
#             f"  48h={self.ugde_48h:.0f}  72h={self.ugde_72h:.0f} km",
#             "",
#             "  ── Tier 2: Operational (vs ST-Trans) ──────────────────────",
#             f"  |ATE| mean : {self.ate_abs_mean:.1f} km  (ST-Trans ATE=79.94) {'✅' if self.ate_abs_mean < 79.94 else '❌'}",
#             f"  |CTE| mean : {self.cte_abs_mean:.1f} km  (ST-Trans CTE=93.58) {'✅' if self.cte_abs_mean < 93.58 else '❌'}",
#             f"  ADE_str    : {self.ade_str:.1f} km",
#             f"  ADE_rec    : {self.ade_rec:.1f} km",
#             "═" * 70,
#         ]
#         return "\n".join(lines)


# _CSV_FIELDS = [
#     "timestamp", "n_total", "n_str", "n_rec",
#     "ade", "fde", "ugde_12h", "ugde_24h", "ugde_48h", "ugde_72h",
#     "tss_72h", "ate_abs_mean", "cte_abs_mean",
#     "ade_str", "ade_rec", "pr", "rdr",
#     "crps_mean", "crps_72h", "ssr_mean", "bss_mean",
#     "dtw_mean", "dtw_str", "dtw_rec",
#     "oyr_mean", "oyr_rec", "hle_mean", "hle_rec",
#     "loss_total", "loss_fm", "loss_dir", "loss_step",
#     "loss_disp", "loss_heading", "loss_smooth", "loss_pinn",
# ]


# def save_metrics_csv(metrics: DatasetMetrics, csv_path: str, tag: str = "") -> None:
#     os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
#     write_header = not os.path.exists(csv_path)
#     row: Dict = {f: getattr(metrics, f, float("nan")) for f in _CSV_FIELDS}
#     if tag: row["timestamp"] = f"{tag}_{metrics.timestamp}"
#     with open(csv_path, "a", newline="") as fh:
#         writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
#         if write_header: writer.writeheader()
#         writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
#                          for k, v in row.items()})


# # ══════════════════════════════════════════════════════════════════════════════
# #  Backward-compat wrappers
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


# # ── Probabilistic ─────────────────────────────────────────────────────────────

# def crps_2d(pred_ens_01: np.ndarray, gt_01: np.ndarray) -> float:
#     S = pred_ens_01.shape[0]
#     gt_rep = gt_01[np.newaxis].repeat(S, axis=0)
#     acc = float(np.mean(haversine_km(pred_ens_01, gt_rep)))
#     p_i = pred_ens_01[:, np.newaxis, :]
#     p_j = pred_ens_01[np.newaxis, :, :]
#     p_i_flat = np.broadcast_to(p_i, (S, S, pred_ens_01.shape[-1])).reshape(S * S, -1)
#     p_j_flat = np.broadcast_to(p_j, (S, S, pred_ens_01.shape[-1])).reshape(S * S, -1)
#     div = float(np.mean(haversine_km(p_i_flat, p_j_flat)))
#     return float(acc - 0.5 * div)


# def ssr_step(pred_ens_01: np.ndarray, gt_01: np.ndarray) -> float:
#     S        = pred_ens_01.shape[0]
#     ens_mean = pred_ens_01.mean(axis=0)
#     ens_mean_rep = ens_mean[np.newaxis].repeat(S, axis=0)
#     spread_vals  = haversine_km(pred_ens_01, ens_mean_rep)
#     spread       = float(np.sqrt(np.mean(spread_vals ** 2)))
#     rmse_em = float(haversine_km(ens_mean[np.newaxis], gt_01[np.newaxis])[0])
#     if spread < 1e-6 and rmse_em < 1e-6: return 1.0
#     return spread / (rmse_em + 1e-8)


# def dtw_haversine(s: np.ndarray, t: np.ndarray) -> float:
#     n, m = len(s), len(t)
#     dp = np.full((n + 1, m + 1), np.inf)
#     dp[0, 0] = 0.0
#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             cost = float(haversine_km(s[i-1:i], t[j-1:j])[0])
#             dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
#     return float(dp[n, m])


# def cliper_errors(obs_seqs, gt_seqs, pred_len: int) -> Tuple[float, np.ndarray]:
#     all_errs = []
#     for obs, gt in zip(obs_seqs, gt_seqs):
#         errs = np.zeros(pred_len)
#         for h in range(pred_len):
#             pred_c = cliper_forecast(np.array(obs), h + 1)
#             errs[h] = float(haversine_km(pred_c[np.newaxis],
#                                           np.array(gt)[h:h+1], unit_01deg=True)[0])
#         all_errs.append(errs)
#     mat = np.stack(all_errs)
#     return float(mat.mean()), mat


# def persistence_errors(obs_seqs, gt_seqs, pred_len: int) -> np.ndarray:
#     all_errs = []
#     for obs, gt in zip(obs_seqs, gt_seqs):
#         last = np.array(obs)[-1, :2]
#         errs = np.array([float(haversine_km(last[np.newaxis],
#                                              np.array(gt)[h:h+1], unit_01deg=True)[0])
#                           for h in range(pred_len)])
#         all_errs.append(errs)
#     return np.stack(all_errs)

"""
utils/metrics.py — v7fix2  FINAL CORRECT
=========================================
ROOT CAUSE FIX:

  BUG C (ATE=9433km):
    _eval_batch_atecte() calls:
      pred_d = denorm_torch(pred_norm)  → 0.1-deg units (e.g. lon=1200 = 120deg)
      haversine_and_atecte_torch(pred_d, gt_d, unit_01deg=True)

    OLD haversine_and_atecte_torch with unit_01deg=True applied:
      pred_lon = (pred_d[...,0] * 50.0 + 1800.0) / 10.0
      For pred_d=1200: (1200*50+1800)/10 = 6180 deg  ← GARBAGE

    FIX: Input is already 0.1-deg units from denorm_torch.
    Just divide by scale=10 to get degrees:
      pred_lon = pred_d[...,0] / 10.0   → 1200/10 = 120 deg ✅

    CONSISTENT with haversine_km_torch which uses same scale logic.

All other code from v6 kept intact.
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
HORIZON_STEPS: Dict[int, int] = {6: 0, 12: 1, 24: 3, 36: 5, 48: 7, 60: 9, 72: 11}
RECURV_THR_DEG = 45.0

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
    """
    Haversine distance in km.
    unit_01deg=True  → input in 0.1-degree units (divide by 10 to get degrees)
    unit_01deg=False → input already in degrees
    """
    scale = 10.0 if unit_01deg else 1.0
    lat1 = np.deg2rad(p1[..., lat_idx] / scale)
    lat2 = np.deg2rad(p2[..., lat_idx] / scale)
    dlon = np.deg2rad((p2[..., lon_idx] - p1[..., lon_idx]) / scale)
    dlat = np.deg2rad((p2[..., lat_idx] - p1[..., lat_idx]) / scale)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def haversine_km_np(p1, p2, unit_01deg=True):
    return haversine_km(p1, p2, unit_01deg=unit_01deg)


def haversine_km_torch(pred, gt, lon_idx=0, lat_idx=1, unit_01deg=True):
    """
    Haversine distance in km (PyTorch).
    unit_01deg=True  → input in 0.1-deg units (from denorm_torch)
    unit_01deg=False → input in degrees
    """
    scale = 10.0 if unit_01deg else 1.0
    lat1  = torch.deg2rad(pred[..., lat_idx] / scale)
    lat2  = torch.deg2rad(gt[..., lat_idx]   / scale)
    dlon  = torch.deg2rad((pred[..., lon_idx] - gt[..., lon_idx]) / scale)
    dlat  = torch.deg2rad((pred[..., lat_idx] - gt[..., lat_idx]) / scale)
    a = (torch.sin(dlat / 2.0) ** 2
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2)
    return 2.0 * R_EARTH_KM * torch.asin(a.clamp(0.0, 1.0).sqrt())


def haversine_and_atecte_torch(
    pred_01: "torch.Tensor",
    gt_01: "torch.Tensor",
    unit_01deg: bool = True,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Tính haversine dist + ATE + CTE trong một pass.

    INPUTS:
      pred_01, gt_01: [T, B, 2]  — from denorm_torch() = 0.1-degree units
                                    OR from raw degrees if unit_01deg=False
      unit_01deg=True  → divide by 10 to get degrees (denorm_torch output)
      unit_01deg=False → already degrees

    OUTPUTS:
      dist [T, B]  — haversine distance in km
      ate  [T, B]  — along-track error in km (absolute value)
      cte  [T, B]  — cross-track error in km (absolute value)

    BUG FIX v7fix2:
      Old code applied normalized→degree formula to 0.1-deg input:
        pred_lon = (pred_01[...,0] * 50.0 + 1800.0) / 10.0
      This was WRONG when unit_01deg=True with denorm_torch output.
      
      CORRECT: just divide by scale (same as haversine_km_torch):
        pred_lon = pred_01[...,0] / scale
    """
    scale = 10.0 if unit_01deg else 1.0

    # ── Convert to degrees ────────────────────────────────────────────────
    pred_lon = pred_01[..., 0] / scale   # [T, B] degrees
    pred_lat = pred_01[..., 1] / scale   # [T, B] degrees
    gt_lon   = gt_01[..., 0]   / scale   # [T, B] degrees
    gt_lat   = gt_01[..., 1]   / scale   # [T, B] degrees

    # ── Haversine distance ────────────────────────────────────────────────
    lat1r = torch.deg2rad(pred_lat)
    lat2r = torch.deg2rad(gt_lat)
    dlon  = torch.deg2rad(gt_lon - pred_lon)
    dlat  = torch.deg2rad(gt_lat - pred_lat)
    a = (torch.sin(dlat / 2.0).pow(2)
         + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2.0).pow(2))
    dist = 2.0 * R_EARTH_KM * torch.asin(a.clamp(0.0, 1.0).sqrt())   # [T, B]

    # ── ATE / CTE decomposition ───────────────────────────────────────────
    DEG_KM = 111.0  # km per degree latitude
    T = pred_01.shape[0]

    if T >= 2:
        # Track direction: gt forward difference, backward for last step
        gt_dx = torch.zeros_like(gt_lon)   # [T, B] degrees/step
        gt_dy = torch.zeros_like(gt_lat)
        gt_dx[:-1] = gt_lon[1:] - gt_lon[:-1]
        gt_dy[:-1] = gt_lat[1:] - gt_lat[:-1]
        gt_dx[-1]  = gt_dx[-2]  # repeat last
        gt_dy[-1]  = gt_dy[-2]

        # Convert displacement to km (longitude needs cos(lat) correction)
        cos_lat = torch.cos(torch.deg2rad(gt_lat)).clamp(min=1e-4)   # [T, B]
        track_x = gt_dx * cos_lat * DEG_KM   # [T, B] km east
        track_y = gt_dy * DEG_KM              # [T, B] km north

        # Normalise to unit vector (handle zero movement)
        track_norm = (track_x.pow(2) + track_y.pow(2)).sqrt().clamp(min=1e-4)
        u_x = track_x / track_norm   # along-track unit x
        u_y = track_y / track_norm   # along-track unit y
        n_x = -u_y                   # cross-track unit x (90° CCW)
        n_y =  u_x                   # cross-track unit y

        # Position error in km (pred − gt)
        err_lon = (pred_lon - gt_lon) * cos_lat * DEG_KM   # [T, B]
        err_lat = (pred_lat - gt_lat) * DEG_KM              # [T, B]

        # Project error onto track axes
        ate = (err_lon * u_x + err_lat * u_y).abs()   # [T, B]
        cte = (err_lon * n_x + err_lat * n_y).abs()   # [T, B]
    else:
        # Fallback: no track direction available
        ate = dist * 0.5
        cte = dist * 0.5

    return dist, ate, cte


def denorm_np(n: np.ndarray) -> np.ndarray:
    """Normalised [-1..1] → 0.1-degree units."""
    r = n.copy()
    r[..., 0] = n[..., 0] * 50.0 + 1800.0
    r[..., 1] = n[..., 1] * 50.0
    return r


def denorm_torch(n: "torch.Tensor") -> "torch.Tensor":
    """Normalised [-1..1] → 0.1-degree units."""
    r = n.clone()
    r[..., 0] = n[..., 0] * 50.0 + 1800.0
    r[..., 1] = n[..., 1] * 50.0
    return r


def denorm_deg_np(arr_norm: np.ndarray) -> np.ndarray:
    """Normalised → degrees."""
    out = arr_norm.copy()
    out[..., 0] = (arr_norm[..., 0] * 50.0 + 1800.0) / 10.0
    out[..., 1] = (arr_norm[..., 1] * 50.0) / 10.0
    return out


def cliper_forecast(obs_norm: np.ndarray, h: int) -> np.ndarray:
    if obs_norm.shape[0] < 2:
        return obs_norm[-1, :2].copy()
    v = obs_norm[-1, :2] - obs_norm[-2, :2]
    return obs_norm[-1, :2] + h * v


# ══════════════════════════════════════════════════════════════════════════════
#  2. Tier 1
# ══════════════════════════════════════════════════════════════════════════════

def ugde(pred_01, gt_01):
    return haversine_km(pred_01, gt_01)


def ade_fde(pred_01, gt_01):
    ps = ugde(pred_01, gt_01)
    return float(ps.mean()), float(ps[-1]), ps


# ══════════════════════════════════════════════════════════════════════════════
#  3. Tier 2 helpers
# ══════════════════════════════════════════════════════════════════════════════

def tss(ugde_model, ugde_cliper):
    return 1.0 - ugde_model / (ugde_cliper + 1e-8)


def lat_corrected_velocity(traj_01, lon_idx=0, lat_idx=1):
    lats_rad = np.deg2rad(traj_01[:, lat_idx] / 10.0)
    cos_lat  = np.cos(lats_rad[:-1])
    dlat = np.diff(traj_01[:, lat_idx])
    dlon = np.diff(traj_01[:, lon_idx]) * cos_lat
    return np.stack([dlon, dlat], axis=-1)


def total_rotation_angle(gt_01):
    if gt_01.shape[0] < 3: return 0.0
    v = lat_corrected_velocity(gt_01)
    total = 0.0
    for i in range(len(v) - 1):
        v1, v2 = v[i], v[i + 1]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8: continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        total += np.degrees(np.arccos(cos_a))
    return total


def classify(gt_01, thr=RECURV_THR_DEG):
    return "recurvature" if total_rotation_angle(gt_01) >= thr else "straight"


def ate_cte(pred_01, gt_01, lon_idx=0, lat_idx=1):
    T = pred_01.shape[0]
    ate_arr = np.zeros(T)
    cte_arr = np.zeros(T)
    km_per_01deg = 11.112
    for k in range(T):
        dk = gt_01[k] - gt_01[k - 1] if k > 0 else (gt_01[1] - gt_01[0] if T > 1 else np.array([1.0, 0.0]))
        lat_rad = np.deg2rad(gt_01[k, lat_idx] / 10.0)
        dk_km = dk * km_per_01deg
        dk_km[0] *= np.cos(lat_rad)
        norm_dk = np.linalg.norm(dk_km)
        if norm_dk < 1e-8: continue
        t_hat = dk_km / norm_dk
        n_hat = np.array([-t_hat[1], t_hat[0]])
        delta_pos = pred_01[k] - gt_01[k]
        delta_km  = delta_pos * km_per_01deg
        delta_km[0] *= np.cos(lat_rad)
        ate_arr[k] = float(np.dot(delta_km, t_hat))
        cte_arr[k] = float(np.dot(delta_km, n_hat))
    return ate_arr, cte_arr


def circular_std(angles_deg):
    if len(angles_deg) == 0: return 0.0
    rads  = np.deg2rad(angles_deg)
    R_bar = np.abs(np.mean(np.exp(1j * rads)))
    R_bar = np.clip(R_bar, 1e-8, 1.0 - 1e-8)
    return float(np.degrees(np.sqrt(-2.0 * np.log(R_bar))))


def compute_csd(traj_01):
    v = lat_corrected_velocity(traj_01)
    if len(v) == 0: return 0.0
    angles = np.arctan2(v[:, 1], v[:, 0])
    return circular_std(np.degrees(angles))


def oyr(pred_01, gt_01):
    pv = lat_corrected_velocity(pred_01)
    gv = lat_corrected_velocity(gt_01)
    m  = min(len(pv), len(gv))
    if m == 0: return 0.0
    dots  = np.sum(pv[:m] * gv[:m], axis=1)
    np_n  = np.linalg.norm(pv[:m], axis=1)
    ng_n  = np.linalg.norm(gv[:m], axis=1)
    valid = (np_n > 1e-8) & (ng_n > 1e-8)
    if valid.sum() == 0: return 0.0
    cos_vals = dots[valid] / (np_n[valid] * ng_n[valid])
    return float((cos_vals < 0).mean())


def hle(pred_01, gt_01):
    def curvature(traj):
        v = lat_corrected_velocity(traj)
        if len(v) < 2: return np.array([])
        cross = v[1:, 0] * v[:-1, 1] - v[1:, 1] * v[:-1, 0]
        n1 = np.linalg.norm(v[1:],  axis=1).clip(1e-8)
        n2 = np.linalg.norm(v[:-1], axis=1).clip(1e-8)
        return cross / (n1 * n2)
    kp = curvature(pred_01); kg = curvature(gt_01)
    m  = min(len(kp), len(kg))
    if m == 0: return float("nan")
    return float(np.mean(np.abs(kp[:m] - kg[:m])))


# ══════════════════════════════════════════════════════════════════════════════
#  Composite score v2 (ST-Trans aware)
# ══════════════════════════════════════════════════════════════════════════════

def _composite_score(result: dict) -> float:
    """Lower = better. score < 100 = beat ST-Trans on all metrics."""
    ade = result.get("ADE", float("inf"))
    h12 = result.get("12h", float("inf"))
    h24 = result.get("24h", float("inf"))
    h48 = result.get("48h", float("inf"))
    h72 = result.get("72h", float("inf"))
    ate = result.get("ATE_mean", float("inf"))
    cte = result.get("CTE_mean", float("inf"))
    if not np.isfinite(ate): ate = ade * 0.46
    if not np.isfinite(cte): cte = ade * 0.53
    return 100.0 * (
        0.05 * (ade / 136.0)
        + 0.05 * (h12 / 50.0)
        + 0.10 * (h24 / 100.0)
        + 0.15 * (h48 / 200.0)
        + 0.35 * (h72 / 300.0)
        + 0.15 * (ate / 80.0)
        + 0.15 * (cte / 94.0)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Fast Tier-1 accumulator with ATE/CTE tracking
# ══════════════════════════════════════════════════════════════════════════════

class StepErrorAccumulator:
    """
    Per-step error accumulator.
    Optionally tracks ATE and CTE per step.
    """

    def __init__(self, pred_len: int = PRED_LEN, step_hours: int = STEP_HOURS):
        self.pred_len   = pred_len
        self.step_hours = step_hours
        self.reset()

    def reset(self) -> None:
        self._sum     = np.zeros(self.pred_len, dtype=np.float64)
        self._sum_sq  = np.zeros(self.pred_len, dtype=np.float64)
        self._count   = np.zeros(self.pred_len, dtype=np.int64)
        self._ate_sum = np.zeros(self.pred_len, dtype=np.float64)
        self._cte_sum = np.zeros(self.pred_len, dtype=np.float64)
        self._ac_cnt  = np.zeros(self.pred_len, dtype=np.int64)

    def _to_np2d(self, x) -> Optional[np.ndarray]:
        if x is None: return None
        if HAS_TORCH and torch.is_tensor(x):
            x = x.double().cpu().numpy()
        else:
            x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1: x = x.reshape(-1, 1)
        return x if x.ndim == 2 else None

    def update(self, dist_km, ate_km=None, cte_km=None) -> None:
        d = self._to_np2d(dist_km)
        if d is None: return

        T_actual, B = d.shape
        if T_actual < self.pred_len:
            pad    = np.zeros((self.pred_len - T_actual, B), dtype=np.float64)
            d_full = np.concatenate([d, pad], axis=0)
        else:
            d_full   = d[:self.pred_len]
            T_actual = self.pred_len

        self._sum    += d_full.sum(axis=1)
        self._sum_sq += (d_full ** 2).sum(axis=1)
        self._count[:T_actual] += B

        # ATE / CTE (optional)
        if ate_km is not None and cte_km is not None:
            a = self._to_np2d(ate_km)
            c = self._to_np2d(cte_km)
            if a is not None and c is not None:
                T_ac = min(a.shape[0], self.pred_len)
                def _pad(arr, T_ac):
                    if arr.shape[0] < self.pred_len:
                        return np.concatenate([arr[:T_ac],
                            np.zeros((self.pred_len - T_ac, B), dtype=np.float64)], 0)
                    return arr[:self.pred_len]
                a_full = _pad(a, T_ac)
                c_full = _pad(c, T_ac)
                T_ac_real = min(T_ac, self.pred_len)
                self._ate_sum[:T_ac_real] += a_full[:T_ac_real].sum(axis=1)
                self._cte_sum[:T_ac_real] += c_full[:T_ac_real].sum(axis=1)
                self._ac_cnt[:T_ac_real]  += B

    def compute(self) -> Dict:
        if self._count[0] == 0: return {}

        cs  = np.where(self._count > 0, self._count, 1)
        ps  = np.where(self._count > 0, self._sum    / cs, 0.0)
        ps2 = np.where(self._count > 0, self._sum_sq / cs, 0.0)
        std = np.sqrt(np.maximum(ps2 - ps ** 2, 0.0))

        active  = int(np.sum(self._count > 0))
        if active == 0: return {}

        out: Dict = {
            "per_step":     ps,
            "per_step_std": std,
            "ADE":          float(ps[:active].mean()),
            "FDE":          float(ps[active - 1]),
            "active_steps": active,
            "n_samples":    int(self._count[0]),
        }

        # Per-horizon errors
        for h, s in HORIZON_STEPS.items():
            if s < self.pred_len and self._count[s] > 0:
                out[f"{h}h"]     = float(ps[s])
                out[f"{h}h_std"] = float(std[s])
            else:
                out[f"{h}h"]     = float("nan")
                out[f"{h}h_std"] = float("nan")

        # ATE / CTE means
        if self._ac_cnt[0] > 0:
            acs    = np.where(self._ac_cnt > 0, self._ac_cnt, 1)
            ate_ps = np.where(self._ac_cnt > 0, self._ate_sum / acs, 0.0)
            cte_ps = np.where(self._ac_cnt > 0, self._cte_sum / acs, 0.0)
            out["ATE_mean"] = float(ate_ps[:active].mean())
            out["CTE_mean"] = float(cte_ps[:active].mean())
            for h, s in HORIZON_STEPS.items():
                if s < self.pred_len and self._ac_cnt[s] > 0:
                    out[f"ATE_{h}h"] = float(ate_ps[s])
                    out[f"CTE_{h}h"] = float(cte_ps[s])

        out["composite_score"] = _composite_score(out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  DatasetMetrics, SequenceResult, CSV helpers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SequenceResult:
    ade: float; fde: float; per_step: np.ndarray
    ate: np.ndarray; cte: np.ndarray
    category: str; category_pred: str; theta: float
    csd_gt: float; oyr_val: float; hle_val: float
    crps: Optional[np.ndarray] = None
    ssr:  Optional[np.ndarray] = None
    dtw:  float = float("nan")
    loss_fm: float = float("nan"); loss_dir: float = float("nan")
    loss_step: float = float("nan"); loss_disp: float = float("nan")
    loss_heading: float = float("nan"); loss_smooth: float = float("nan")
    loss_pinn: float = float("nan"); loss_total: float = float("nan")


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
        def i(v, t): return " ✅" if np.isfinite(v) and v < t else " ❌"
        lines = [
            "═"*64, "  TC Track Metrics (v7fix2, vs ST-Trans)", "═"*64,
            f"  ADE : {self.ade:.1f} km{i(self.ade,136.41)}  [ST-Trans=136.41]",
            f"  12h : {self.ugde_12h:.1f} km{i(self.ugde_12h,50)}",
            f"  24h : {self.ugde_24h:.1f} km{i(self.ugde_24h,100)}",
            f"  48h : {self.ugde_48h:.1f} km{i(self.ugde_48h,200)}",
            f"  72h : {self.ugde_72h:.1f} km{i(self.ugde_72h,297)}",
            f"  |ATE|: {self.ate_abs_mean:.1f} km{i(self.ate_abs_mean,79.94)}  [ST-Trans=79.94]",
            f"  |CTE|: {self.cte_abs_mean:.1f} km{i(self.cte_abs_mean,93.58)}  [ST-Trans=93.58]",
            "═"*64,
        ]
        return "\n".join(lines)


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
    row: Dict = {f: getattr(metrics, f, float("nan")) for f in _CSV_FIELDS}
    if tag: row["timestamp"] = f"{tag}_{metrics.timestamp}"
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        if write_header: writer.writeheader()
        writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                         for k, v in row.items()})


# ── Backward-compat wrappers ──────────────────────────────────────────────────

def RSE(pred, true):
    return float(np.sqrt(np.sum((true - pred)**2))
                 / (np.sqrt(np.sum((true - true.mean())**2)) + 1e-8))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true-true.mean(0))**2).sum(0)*((pred-pred.mean(0))**2).sum(0))+1e-8
    return u / d

def MAE(pred, true):  return float(np.mean(np.abs(pred - true)))
def MSE(pred, true):  return float(np.mean((pred - true) ** 2))
def RMSE(pred, true): return float(np.sqrt(MSE(pred, true)))
def MAPE(pred, true): return float(np.mean(np.abs((pred-true)/(np.abs(true)+1e-5))))
def MSPE(pred, true): return float(np.mean(np.square((pred-true)/(np.abs(true)+1e-5))))

def metric(pred, true):
    return (MAE(pred,true), MSE(pred,true), RMSE(pred,true),
            MAPE(pred,true), MSPE(pred,true), RSE(pred,true), CORR(pred,true))


def crps_2d(pred_ens_01, gt_01):
    S = pred_ens_01.shape[0]
    gt_rep = gt_01[np.newaxis].repeat(S, axis=0)
    acc = float(np.mean(haversine_km(pred_ens_01, gt_rep)))
    p_i = pred_ens_01[:, np.newaxis, :]
    p_j = pred_ens_01[np.newaxis, :, :]
    p_i_f = np.broadcast_to(p_i, (S,S,pred_ens_01.shape[-1])).reshape(S*S,-1)
    p_j_f = np.broadcast_to(p_j, (S,S,pred_ens_01.shape[-1])).reshape(S*S,-1)
    div = float(np.mean(haversine_km(p_i_f, p_j_f)))
    return float(acc - 0.5*div)


def ssr_step(pred_ens_01, gt_01):
    S = pred_ens_01.shape[0]
    em = pred_ens_01.mean(axis=0)
    em_rep = em[np.newaxis].repeat(S, axis=0)
    spread = float(np.sqrt(np.mean(haversine_km(pred_ens_01, em_rep)**2)))
    rmse_em = float(haversine_km(em[np.newaxis], gt_01[np.newaxis])[0])
    if spread < 1e-6 and rmse_em < 1e-6: return 1.0
    return spread / (rmse_em + 1e-8)


def dtw_haversine(s, t):
    n, m = len(s), len(t)
    dp = np.full((n+1, m+1), np.inf); dp[0,0] = 0.0
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = float(haversine_km(s[i-1:i], t[j-1:j])[0])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return float(dp[n,m])


def cliper_errors(obs_seqs, gt_seqs, pred_len):
    all_errs = []
    for obs, gt in zip(obs_seqs, gt_seqs):
        errs = np.zeros(pred_len)
        for h in range(pred_len):
            pred_c = cliper_forecast(np.array(obs), h+1)
            errs[h] = float(haversine_km(pred_c[np.newaxis],
                                          np.array(gt)[h:h+1], unit_01deg=True)[0])
        all_errs.append(errs)
    mat = np.stack(all_errs)
    return float(mat.mean()), mat


def persistence_errors(obs_seqs, gt_seqs, pred_len):
    all_errs = []
    for obs, gt in zip(obs_seqs, gt_seqs):
        last = np.array(obs)[-1, :2]
        errs = np.array([float(haversine_km(last[np.newaxis],
                                             np.array(gt)[h:h+1])[0])
                          for h in range(pred_len)])
        all_errs.append(errs)
    return np.stack(all_errs)
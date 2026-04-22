# # # # # # # """
# # # # # # # Model/losses.py  ── v24
# # # # # # # ========================
# # # # # # # FIXES vs v23:

# # # # # # #   FIX-L44  [CRITICAL] pinn_shallow_water: log(loss-49) tạo ra gradient
# # # # # # #            gần bằng 0 khi loss>50 (gradient = 1/(loss-49) → rất nhỏ).
# # # # # # #            Từ log: pinn=52.69 không đổi suốt 90 epoch → NO gradient.
# # # # # # #            Fix: Không dùng log-clamp. Dùng soft-tanh scaling:
# # # # # # #              loss_scaled = 50 * tanh(loss / 50)
# # # # # # #            → gradient = tanh'(x/50) = sech²(x/50) > 0 với mọi x
# # # # # # #            → tại loss=52.69: gradient ≈ 0.92 (vs log: ≈ 0.27)
# # # # # # #            → PINN sẽ thực sự học thay vì stuck.

# # # # # # #   FIX-L45  [HIGH] pinn_rankine_steering: u/v500 sau FIX-ENV-20 là
# # # # # # #            normalized [-1,1] → cần scale sang m/s để so sánh với u_tc.
# # # # # # #            Thêm u500_scale=30.0 để convert → steering comparison đúng.

# # # # # # #   FIX-L46  [HIGH] ensemble_spread_loss: max_spread_km=400 quá cao.
# # # # # # #            Từ log: spread 800-1000 km suốt training.
# # # # # # #            Fix: max_spread_km=200, weight tăng từ 0.1 → 0.3.
# # # # # # #            Thêm per-step spread penalty (không chỉ final step).

# # # # # # #   FIX-L47  [MEDIUM] velocity_loss: thêm direction penalty mạnh hơn
# # # # # # #            khi pred đi ngược chiều gt. Tăng heading_loss weight.

# # # # # # #   FIX-L48  [MEDIUM] trajectory_smoothness: penalize acceleration thay đổi
# # # # # # #            quá đột ngột (jerk) để giảm DTW.

# # # # # # # Kept from v23:
# # # # # # #   FIX-L39  pinn scale=1e-3
# # # # # # #   FIX-L40  step_weight_alpha
# # # # # # #   FIX-L41  fm_afcrps time weights
# # # # # # #   FIX-L43  velocity_loss dimensionless
# # # # # # # """
# # # # # # # from __future__ import annotations

# # # # # # # import math
# # # # # # # from typing import Dict, Optional

# # # # # # # import torch
# # # # # # # import torch.nn.functional as F

# # # # # # # __all__ = [
# # # # # # #     "WEIGHTS", "compute_total_loss", "fm_afcrps_loss",
# # # # # # #     "fm_physics_consistency_loss", "pinn_bve_loss",
# # # # # # #     "recurvature_loss", "velocity_loss_per_sample",
# # # # # # # ]

# # # # # # # OMEGA        = 7.2921e-5
# # # # # # # R_EARTH      = 6.371e6
# # # # # # # DT_6H        = 6 * 3600
# # # # # # # DEG_TO_KM    = 111.0
# # # # # # # STEP_KM      = 113.0

# # # # # # # # U/V500 scale: normalized [-1,1] → real m/s
# # # # # # # _UV500_SCALE = 30.0

# # # # # # # # ── Weights (v24) ──────────────────────────────────────────────────────────────
# # # # # # # WEIGHTS: Dict[str, float] = dict(
# # # # # # #     fm=2.0,
# # # # # # #     velocity=0.8,      # ↑ từ 0.5 (FIX-L47)
# # # # # # #     heading=2.0,       # ↑ từ 1.5 (FIX-L47)
# # # # # # #     recurv=1.5,
# # # # # # #     step=0.5,
# # # # # # #     disp=0.5,
# # # # # # #     dir=1.0,
# # # # # # #     smooth=0.5,        # ↑ từ 0.3
# # # # # # #     accel=0.8,         # ↑ từ 0.5
# # # # # # #     jerk=0.3,          # NEW (FIX-L48)
# # # # # # #     pinn=0.02,
# # # # # # #     fm_physics=0.3,
# # # # # # #     spread=0.6,        # ↑ từ 0.3 (FIX-L46)
# # # # # # # )

# # # # # # # RECURV_ANGLE_THR = 45.0
# # # # # # # RECURV_WEIGHT    = 2.5


# # # # # # # # ── Haversine ─────────────────────────────────────────────────────────────────

# # # # # # # def _haversine(p1: torch.Tensor, p2: torch.Tensor,
# # # # # # #                unit_01deg: bool = True) -> torch.Tensor:
# # # # # # #     if unit_01deg:
# # # # # # #         lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # #         lat1 = (p1[..., 1] * 50.0) / 10.0
# # # # # # #         lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # #         lat2 = (p2[..., 1] * 50.0) / 10.0
# # # # # # #     else:
# # # # # # #         lon1, lat1 = p1[..., 0], p1[..., 1]
# # # # # # #         lon2, lat2 = p2[..., 0], p2[..., 1]

# # # # # # #     lat1r = torch.deg2rad(lat1);  lat2r = torch.deg2rad(lat2)
# # # # # # #     dlon  = torch.deg2rad(lon2 - lon1)
# # # # # # #     dlat  = torch.deg2rad(lat2 - lat1)
# # # # # # #     a     = (torch.sin(dlat / 2) ** 2
# # # # # # #              + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2) ** 2)
# # # # # # #     return 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())


# # # # # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # # #     return _haversine(p1, p2, unit_01deg=False)


# # # # # # # # ── Step displacements in km ──────────────────────────────────────────────────

# # # # # # # def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
# # # # # # #     dt      = traj_deg[1:] - traj_deg[:-1]
# # # # # # #     lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
# # # # # # #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
# # # # # # #     dt_km   = dt.clone()
# # # # # # #     dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
# # # # # # #     dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
# # # # # # #     return dt_km


# # # # # # # # ── Recurvature helpers ────────────────────────────────────────────────────────

# # # # # # # def _total_rotation_angle_batch(gt: torch.Tensor) -> torch.Tensor:
# # # # # # #     T, B, _ = gt.shape
# # # # # # #     if T < 3:
# # # # # # #         return gt.new_zeros(B)
# # # # # # #     lats_rad = torch.deg2rad(gt[:, :, 1])
# # # # # # #     cos_lat  = torch.cos(lats_rad[:-1])
# # # # # # #     dlat = gt[1:, :, 1] - gt[:-1, :, 1]
# # # # # # #     dlon = (gt[1:, :, 0] - gt[:-1, :, 0]) * cos_lat
# # # # # # #     v    = torch.stack([dlon, dlat], dim=-1)
# # # # # # #     v1   = v[:-1];  v2 = v[1:]
# # # # # # #     n1   = v1.norm(dim=-1).clamp(min=1e-8)
# # # # # # #     n2   = v2.norm(dim=-1).clamp(min=1e-8)
# # # # # # #     cos_a = (v1 * v2).sum(-1) / (n1 * n2)
# # # # # # #     return torch.rad2deg(torch.acos(cos_a.clamp(-1.0, 1.0))).sum(0)


# # # # # # # def _recurvature_weights(gt: torch.Tensor,
# # # # # # #                          thr: float = RECURV_ANGLE_THR,
# # # # # # #                          w_recurv: float = RECURV_WEIGHT) -> torch.Tensor:
# # # # # # #     rot = _total_rotation_angle_batch(gt)
# # # # # # #     return torch.where(rot >= thr,
# # # # # # #                        torch.full_like(rot, w_recurv),
# # # # # # #                        torch.ones_like(rot))


# # # # # # # # ── Directional losses ────────────────────────────────────────────────────────

# # # # # # # def velocity_loss_per_sample(pred: torch.Tensor,
# # # # # # #                              gt: torch.Tensor) -> torch.Tensor:
# # # # # # #     if pred.shape[0] < 2:
# # # # # # #         return pred.new_zeros(pred.shape[1])
# # # # # # #     v_pred_km = _step_displacements_km(pred)
# # # # # # #     v_gt_km   = _step_displacements_km(gt)
# # # # # # #     s_pred    = v_pred_km.norm(dim=-1)
# # # # # # #     s_gt      = v_gt_km.norm(dim=-1)
# # # # # # #     l_speed   = (s_pred - s_gt).pow(2).mean(0)
# # # # # # #     gn  = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # # #     gt_unit = v_gt_km / gn
# # # # # # #     ate = ((v_pred_km - v_gt_km) * gt_unit).sum(-1)
# # # # # # #     l_ate = ate.pow(2).mean(0)

# # # # # # #     # FIX-L47: thêm direction penalty mạnh
# # # # # # #     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # # #     cos_sim = ((v_pred_km / pn) * gt_unit).sum(-1)
# # # # # # #     l_dir = F.relu(-cos_sim).pow(2).mean(0) * STEP_KM ** 2  # scale to km²

# # # # # # #     return (l_speed + 0.5 * l_ate + 0.3 * l_dir) / (STEP_KM ** 2)


# # # # # # # def disp_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # # # # #     if pred.shape[0] < 2:
# # # # # # #         return pred.new_zeros(pred.shape[1])
# # # # # # #     pd = _step_displacements_km(pred).norm(dim=-1).mean(0)
# # # # # # #     gd = _step_displacements_km(gt).norm(dim=-1).mean(0)
# # # # # # #     return (pd - gd).pow(2) / (STEP_KM ** 2)


# # # # # # # def step_dir_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # # # # #     if pred.shape[0] < 2:
# # # # # # #         return pred.new_zeros(pred.shape[1])
# # # # # # #     v_pred_km = _step_displacements_km(pred)
# # # # # # #     v_gt_km   = _step_displacements_km(gt)
# # # # # # #     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # # #     gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # # #     cos_sim = ((v_pred_km / pn) * (v_gt_km / gn)).sum(-1)
# # # # # # #     return (1.0 - cos_sim).mean(0)


# # # # # # # def heading_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # # # # #     if pred.shape[0] < 2:
# # # # # # #         return pred.new_zeros(())
# # # # # # #     pv = _step_displacements_km(pred)
# # # # # # #     gv = _step_displacements_km(gt)
# # # # # # #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # # #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # # #     # FIX-L47: stronger penalty for wrong direction
# # # # # # #     wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1))
# # # # # # #     wrong_dir_loss = (wrong_dir ** 2).mean()  # squared penalty

# # # # # # #     if pred.shape[0] >= 3:
# # # # # # #         def _curv(v):
# # # # # # #             cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
# # # # # # #             n1    = v[1:].norm(dim=-1).clamp(min=1e-4)
# # # # # # #             n2    = v[:-1].norm(dim=-1).clamp(min=1e-4)
# # # # # # #             return (cross / (n1 * n2)).clamp(-10.0, 10.0)
# # # # # # #         curv_mse = F.mse_loss(_curv(pv), _curv(gv))
# # # # # # #     else:
# # # # # # #         curv_mse = pred.new_zeros(())
# # # # # # #     return wrong_dir_loss + curv_mse


# # # # # # # def smooth_loss(pred: torch.Tensor) -> torch.Tensor:
# # # # # # #     if pred.shape[0] < 3:
# # # # # # #         return pred.new_zeros(())
# # # # # # #     v_km = _step_displacements_km(pred)
# # # # # # #     if v_km.shape[0] < 2:
# # # # # # #         return pred.new_zeros(())
# # # # # # #     accel_km = v_km[1:] - v_km[:-1]
# # # # # # #     return accel_km.pow(2).mean() / (STEP_KM ** 2)


# # # # # # # def acceleration_loss(pred: torch.Tensor) -> torch.Tensor:
# # # # # # #     if pred.shape[0] < 3:
# # # # # # #         return pred.new_zeros(())
# # # # # # #     v_km = _step_displacements_km(pred)
# # # # # # #     if v_km.shape[0] < 2:
# # # # # # #         return pred.new_zeros(())
# # # # # # #     a_km = v_km[1:] - v_km[:-1]
# # # # # # #     return a_km.pow(2).mean() / (STEP_KM ** 2)


# # # # # # # def jerk_loss(pred: torch.Tensor) -> torch.Tensor:
# # # # # # #     """
# # # # # # #     FIX-L48: Penalize jerk (change in acceleration) để giảm DTW.
# # # # # # #     Smooth trajectory → lower DTW.
# # # # # # #     """
# # # # # # #     if pred.shape[0] < 4:
# # # # # # #         return pred.new_zeros(())
# # # # # # #     v_km = _step_displacements_km(pred)
# # # # # # #     if v_km.shape[0] < 3:
# # # # # # #         return pred.new_zeros(())
# # # # # # #     a_km  = v_km[1:] - v_km[:-1]
# # # # # # #     j_km  = a_km[1:] - a_km[:-1]
# # # # # # #     return j_km.pow(2).mean() / (STEP_KM ** 2)


# # # # # # # def overall_dir_loss(pred: torch.Tensor, gt: torch.Tensor,
# # # # # # #                      ref: torch.Tensor) -> torch.Tensor:
# # # # # # #     p_d = pred[-1] - ref
# # # # # # #     g_d = gt[-1]   - ref
# # # # # # #     lat_ref = ref[:, 1]
# # # # # # #     cos_lat = torch.cos(torch.deg2rad(lat_ref)).clamp(min=1e-4)
# # # # # # #     p_d_km  = p_d.clone();  p_d_km[:, 0] *= cos_lat * DEG_TO_KM;  p_d_km[:, 1] *= DEG_TO_KM
# # # # # # #     g_d_km  = g_d.clone();  g_d_km[:, 0] *= cos_lat * DEG_TO_KM;  g_d_km[:, 1] *= DEG_TO_KM
# # # # # # #     pn = p_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # # #     gn = g_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # # #     return (1.0 - ((p_d_km / pn) * (g_d_km / gn)).sum(-1)).mean()


# # # # # # # def recurvature_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # # # # #     if pred.shape[0] < 3:
# # # # # # #         return pred.new_zeros(())
# # # # # # #     pred_v   = pred[1:] - pred[:-1]
# # # # # # #     gt_v     = gt[1:]   - gt[:-1]
# # # # # # #     gt_cross = (gt_v[:-1, :, 0] * gt_v[1:, :, 1]
# # # # # # #               - gt_v[:-1, :, 1] * gt_v[1:, :, 0])
# # # # # # #     if gt_cross.shape[0] < 2:
# # # # # # #         return pred.new_zeros(())
# # # # # # #     sign_change = (gt_cross[:-1] * gt_cross[1:]) < 0
# # # # # # #     if not sign_change.any():
# # # # # # #         return pred.new_zeros(())
# # # # # # #     pred_v_mid = pred_v[1:-1]
# # # # # # #     gt_v_mid   = gt_v[1:-1]
# # # # # # #     if pred_v_mid.shape[0] > sign_change.shape[0]:
# # # # # # #         pred_v_mid = pred_v_mid[:sign_change.shape[0]]
# # # # # # #         gt_v_mid   = gt_v_mid[:sign_change.shape[0]]
# # # # # # #     pn      = pred_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # #     gn      = gt_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # #     cos_sim = ((pred_v_mid / pn) * (gt_v_mid / gn)).sum(-1)
# # # # # # #     dir_loss = (1.0 - cos_sim)
# # # # # # #     mask     = sign_change.float()
# # # # # # #     if mask.sum() < 1:
# # # # # # #         return pred.new_zeros(())
# # # # # # #     return (dir_loss * mask).sum() / mask.sum().clamp(min=1)


# # # # # # # # ── AFCRPS ────────────────────────────────────────────────────────────────────

# # # # # # # def fm_afcrps_loss(
# # # # # # #     pred_samples: torch.Tensor,
# # # # # # #     gt: torch.Tensor,
# # # # # # #     unit_01deg: bool = True,
# # # # # # #     intensity_w: Optional[torch.Tensor] = None,
# # # # # # #     step_weight_alpha: float = 0.0,
# # # # # # # ) -> torch.Tensor:
# # # # # # #     M, T, B, _ = pred_samples.shape
# # # # # # #     eps = 1e-3

# # # # # # #     base_w = torch.linspace(0.5, 1.5, T, device=pred_samples.device)

# # # # # # #     if step_weight_alpha > 0.0:
# # # # # # #         # early_w = torch.exp(-torch.arange(T, dtype=torch.float,
# # # # # # #         #                      device=pred_samples.device) * 0.2)
# # # # # # #         # losses.py — fm_afcrps_loss()
# # # # # # #         early_w = torch.exp(-torch.arange(T, dtype=torch.float,
# # # # # # #                      device=pred_samples.device) * 0.35)   # 0.2→0.35
# # # # # # #         early_w = early_w / early_w.mean()
# # # # # # #         time_w  = (1.0 - step_weight_alpha) * base_w + step_weight_alpha * early_w
# # # # # # #     else:
# # # # # # #         time_w = base_w

# # # # # # #     if M == 1:
# # # # # # #         dist = _haversine(pred_samples[0], gt, unit_01deg)
# # # # # # #         loss_per_b = (dist * time_w.unsqueeze(1)).mean(0)
# # # # # # #     else:
# # # # # # #         d_to_gt = _haversine(
# # # # # # #             pred_samples,
# # # # # # #             gt.unsqueeze(0).expand_as(pred_samples),
# # # # # # #             unit_01deg,
# # # # # # #         )
# # # # # # #         d_to_gt_w    = d_to_gt * time_w.unsqueeze(0).unsqueeze(2)
# # # # # # #         d_to_gt_mean = d_to_gt_w.mean(1)

# # # # # # #         ps_i = pred_samples.unsqueeze(1)
# # # # # # #         ps_j = pred_samples.unsqueeze(0)
# # # # # # #         d_pair = _haversine(
# # # # # # #             ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # # # # # #             ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # # # # # #             unit_01deg,
# # # # # # #         ).reshape(M, M, T, B)
# # # # # # #         d_pair_w    = d_pair * time_w.unsqueeze(0).unsqueeze(0).unsqueeze(3)
# # # # # # #         d_pair_mean = d_pair_w.mean(2)

# # # # # # #         e_sy  = d_to_gt_mean.mean(0)
# # # # # # #         e_ssp = d_pair_mean.mean(0).mean(0)
# # # # # # #         loss_per_b = (e_sy - 0.5 * e_ssp).clamp(min=eps)

# # # # # # #     if intensity_w is not None:
# # # # # # #         w = intensity_w.to(loss_per_b.device)
# # # # # # #         return (loss_per_b * w).mean()
# # # # # # #     return loss_per_b.mean()


# # # # # # # # ── PINN losses ────────────────────────────────────────────────────────────────

# # # # # # # def _soft_clamp_loss(loss: torch.Tensor, max_val: float = 50.0) -> torch.Tensor:
# # # # # # #     """
# # # # # # #     FIX-L44: Soft clamp bằng tanh scaling.
# # # # # # #     loss_scaled = max_val * tanh(loss / max_val)
# # # # # # #     Gradient = tanh'(x/max_val) = sech²(x/max_val) > 0 với mọi x.
# # # # # # #     Tại loss=52.69: gradient ≈ sech²(1.054) ≈ 0.31 (vs log: ≈ 0.27 nhưng cố định)
# # # # # # #     Gradient sẽ giảm dần khi loss tăng → vẫn có gradient khác 0.
# # # # # # #     """
# # # # # # #     return max_val * torch.tanh(loss / max_val)


# # # # # # # def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
# # # # # # #     """
# # # # # # #     FIX-L44: Dùng tanh soft-clamp thay vì log-clamp.
# # # # # # #     scale=1e-3 giữ nguyên từ FIX-L39.
# # # # # # #     """
# # # # # # #     T, B, _ = pred_abs_deg.shape
# # # # # # #     if T < 3:
# # # # # # #         return pred_abs_deg.new_zeros(())

# # # # # # #     DT      = DT_6H
# # # # # # #     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
# # # # # # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # # # # #     cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

# # # # # # #     u = dlat[:, :, 0] * cos_lat * 111000.0 / DT
# # # # # # #     v = dlat[:, :, 1] * 111000.0 / DT

# # # # # # #     if u.shape[0] < 2:
# # # # # # #         return pred_abs_deg.new_zeros(())

# # # # # # #     du = (u[1:] - u[:-1]) / DT
# # # # # # #     dv = (v[1:] - v[:-1]) / DT

# # # # # # #     f    = 2 * OMEGA * torch.sin(lat_rad[1:-1])
# # # # # # #     beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH

# # # # # # #     res_u = du - f * v[1:]
# # # # # # #     res_v = dv + f * u[1:]

# # # # # # #     R_tc   = 3e5
# # # # # # #     v_beta_x = -beta * R_tc ** 2 / 2
# # # # # # #     res_u_corrected = res_u - v_beta_x

# # # # # # #     scale = 1e-3
# # # # # # #     loss = ((res_u_corrected / scale).pow(2).mean()
# # # # # # #             + (res_v / scale).pow(2).mean())

# # # # # # #     # FIX-L44: tanh soft-clamp → gradient luôn > 0
# # # # # # #     # return _soft_clamp_loss(loss, max_val=50.0)
# # # # # # #     return _soft_clamp_loss(loss, max_val=20.0)


# # # # # # # def pinn_rankine_steering(pred_abs_deg: torch.Tensor,
# # # # # # #                           env_data: Optional[dict]) -> torch.Tensor:
# # # # # # #     """
# # # # # # #     FIX-L45: u/v500 sau FIX-ENV-20 đã normalized [-1,1].
# # # # # # #     Scale back sang m/s để so sánh với u_tc.
# # # # # # #     """
# # # # # # #     if env_data is None:
# # # # # # #         return pred_abs_deg.new_zeros(())

# # # # # # #     T, B, _ = pred_abs_deg.shape
# # # # # # #     if T < 2:
# # # # # # #         return pred_abs_deg.new_zeros(())

# # # # # # #     DT      = DT_6H
# # # # # # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # # # # #     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
# # # # # # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

# # # # # # #     u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT
# # # # # # #     v_tc = dlat[:, :, 1] * 111000.0 / DT

# # # # # # #     u500_raw = env_data.get("u500_mean", None)
# # # # # # #     v500_raw = env_data.get("v500_mean", None)

# # # # # # #     if u500_raw is None or v500_raw is None:
# # # # # # #         speed = torch.sqrt(u_tc ** 2 + v_tc ** 2)
# # # # # # #         return F.relu(speed - 30.0).pow(2).mean() * 0.01

# # # # # # #     def _align(x: torch.Tensor) -> torch.Tensor:
# # # # # # #         if not torch.is_tensor(x):
# # # # # # #             return pred_abs_deg.new_zeros(T - 1, B)
# # # # # # #         x = x.to(pred_abs_deg.device)
# # # # # # #         if x.dim() == 3:
# # # # # # #             x_sq    = x[:, :, 0]
# # # # # # #             T_env   = x_sq.shape[1]
# # # # # # #             T_tgt   = T - 1
# # # # # # #             if T_env >= T_tgt:
# # # # # # #                 return x_sq[:, :T_tgt].permute(1, 0)
# # # # # # #             pad = x_sq[:, -1:].expand(-1, T_tgt - T_env)
# # # # # # #             return torch.cat([x_sq, pad], dim=1).permute(1, 0)
# # # # # # #         elif x.dim() == 2:
# # # # # # #             if x.shape == (B, T - 1):
# # # # # # #                 return x.permute(1, 0)
# # # # # # #             elif x.shape[0] == T - 1:
# # # # # # #                 return x[:, :B] if x.shape[1] >= B else x.expand(-1, B)
# # # # # # #         return pred_abs_deg.new_zeros(T - 1, B)

# # # # # # #     u500_n = _align(u500_raw)
# # # # # # #     v500_n = _align(v500_raw)

# # # # # # #     # FIX-L45: scale normalized [-1,1] → m/s
# # # # # # #     u500 = u500_n * _UV500_SCALE
# # # # # # #     v500 = v500_n * _UV500_SCALE

# # # # # # #     # Chỉ áp dụng steering penalty khi u/v500 có giá trị thực (non-zero)
# # # # # # #     uv_magnitude = torch.sqrt(u500 ** 2 + v500 ** 2)
# # # # # # #     has_steering = (uv_magnitude > 0.5).float()  # threshold 0.5 m/s

# # # # # # #     env_dir  = torch.stack([u500, v500], dim=-1)
# # # # # # #     tc_dir   = torch.stack([u_tc, v_tc],  dim=-1)
# # # # # # #     env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)  # min 0.5 m/s
# # # # # # #     tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
# # # # # # #     cos_sim  = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)
# # # # # # #     misalign = F.relu(-0.5 - cos_sim).pow(2)

# # # # # # #     # Only penalize when we have actual steering data
# # # # # # #     return (misalign * has_steering).mean() * 0.05


# # # # # # # def pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # # #     if pred_deg.shape[0] < 2:
# # # # # # #         return pred_deg.new_zeros(())
# # # # # # #     dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # # #     lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # #     dx_km   = dt_deg[:, :, 0] * cos_lat * DEG_TO_KM
# # # # # # #     dy_km   = dt_deg[:, :, 1] * DEG_TO_KM
# # # # # # #     speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # # # # # #     return F.relu(speed - 600.0).pow(2).mean()


# # # # # # # def pinn_bve_loss(pred_abs_deg: torch.Tensor,
# # # # # # #                   batch_list,
# # # # # # #                   env_data: Optional[dict] = None) -> torch.Tensor:
# # # # # # #     """
# # # # # # #     FIX-L44: tanh soft-clamp trong pinn_shallow_water.
# # # # # # #     FIX-L45: steering dùng actual u/v500 m/s.
# # # # # # #     """
# # # # # # #     T = pred_abs_deg.shape[0]
# # # # # # #     if T < 3:
# # # # # # #         return pred_abs_deg.new_zeros(())

# # # # # # #     l_sw    = pinn_shallow_water(pred_abs_deg)
# # # # # # #     l_steer = pinn_rankine_steering(pred_abs_deg, env_data)
# # # # # # #     l_speed = pinn_speed_constraint(pred_abs_deg)

# # # # # # #     total = l_sw + 0.5 * l_steer + 0.1 * l_speed
# # # # # # #     # FIX-L44: tanh soft-clamp (không log)
# # # # # # #     # return _soft_clamp_loss(total, max_val=50.0)
# # # # # # #     return _soft_clamp_loss(total, max_val=20.0)


# # # # # # # # ── Physics consistency loss ──────────────────────────────────────────────────

# # # # # # # def fm_physics_consistency_loss(
# # # # # # #     pred_samples: torch.Tensor,
# # # # # # #     gt_norm: torch.Tensor,
# # # # # # #     last_pos: torch.Tensor,
# # # # # # # ) -> torch.Tensor:
# # # # # # #     S, T, B = pred_samples.shape[:3]

# # # # # # #     last_lon = (last_pos[:, 0] * 50.0 + 1800.0) / 10.0
# # # # # # #     last_lat = (last_pos[:, 1] * 50.0) / 10.0
# # # # # # #     lat_rad  = torch.deg2rad(last_lat)
# # # # # # #     beta     = 2 * OMEGA * torch.cos(lat_rad) / R_EARTH
# # # # # # #     R_tc     = 3e5

# # # # # # #     v_beta_lon = -beta * R_tc ** 2 / 2
# # # # # # #     v_beta_lat =  beta * R_tc ** 2 / 4

# # # # # # #     beta_dir      = torch.stack([v_beta_lon, v_beta_lat], dim=-1)
# # # # # # #     beta_norm     = beta_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # #     beta_dir_unit = beta_dir / beta_norm

# # # # # # #     pos_step1     = pred_samples[:, 0, :, :2]
# # # # # # #     dir_step1     = pos_step1 - last_pos.unsqueeze(0)
# # # # # # #     dir_norm      = dir_step1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # #     dir_unit      = dir_step1 / dir_norm
# # # # # # #     mean_dir      = dir_unit.mean(0)
# # # # # # #     mean_norm     = mean_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # #     mean_dir_unit = mean_dir / mean_norm

# # # # # # #     cos_align    = (mean_dir_unit * beta_dir_unit).sum(-1)
# # # # # # #     beta_strength = beta_norm.squeeze(-1)
# # # # # # #     penalise_mask = (beta_strength > 1.0).float()
# # # # # # #     direction_loss = F.relu(-cos_align) * penalise_mask
# # # # # # #     return direction_loss.mean() * 0.5


# # # # # # # # ── Spread regularization ──────────────────────────────────────────────────────

# # # # # # # def ensemble_spread_loss(all_trajs: torch.Tensor,
# # # # # # #                          max_spread_km: float = 200.0) -> torch.Tensor:
# # # # # # #     """
# # # # # # #     FIX-L46: max_spread_km=200 (từ 400), thêm per-step penalty.
# # # # # # #     all_trajs: [S, T, B, 2] normalised.
# # # # # # #     """
# # # # # # #     if all_trajs.shape[0] < 2:
# # # # # # #         return all_trajs.new_zeros(())

# # # # # # #     S, T, B, _ = all_trajs.shape

# # # # # # #     # # Per-step spread penalty (không chỉ final step)
# # # # # # #     # # Weight tăng theo thời gian: later steps penalized more
# # # # # # #     # step_weights = torch.linspace(0.5, 1.5, T, device=all_trajs.device)
    
# # # # # # #     # FIX-L46b: exponential decay — early steps penalized MORE
# # # # # # #     # step 0 weight=2.0, step T-1 weight=0.5 (geometric)
# # # # # # #     step_weights = torch.exp(
# # # # # # #         -torch.arange(T, dtype=torch.float, device=all_trajs.device) * (math.log(4.0) / max(T-1, 1))
# # # # # # #     ) * 2.0   # range: 2.0 → 0.5

# # # # # # #     total_loss = all_trajs.new_zeros(())
# # # # # # #     for t in range(T):
# # # # # # #         step_trajs = all_trajs[:, t, :, :2]  # [S, B, 2]
# # # # # # #         std_lon = step_trajs[:, :, 0].std(0)  # [B]
# # # # # # #         std_lat = step_trajs[:, :, 1].std(0)  # [B]
# # # # # # #         spread_km = torch.sqrt(std_lon ** 2 + std_lat ** 2) * 500.0  # [B]
# # # # # # #         excess    = F.relu(spread_km - max_spread_km)
# # # # # # #         total_loss = total_loss + step_weights[t] * (excess / max_spread_km).pow(2).mean()

# # # # # # #     return total_loss / T


# # # # # # # # ── Intensity weights ─────────────────────────────────────────────────────────

# # # # # # # def _intensity_weights_from_obs(obs_Me: torch.Tensor) -> torch.Tensor:
# # # # # # #     wind_norm = obs_Me[-1, :, 1].detach()
# # # # # # #     w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
# # # # # # #         torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
# # # # # # #         torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
# # # # # # #                     torch.full_like(wind_norm, 1.5))))
# # # # # # #     return w / w.mean().clamp(min=1e-6)


# # # # # # # # ── Main loss ─────────────────────────────────────────────────────────────────

# # # # # # # def compute_total_loss(
# # # # # # #     pred_abs,
# # # # # # #     gt,
# # # # # # #     ref,
# # # # # # #     batch_list,
# # # # # # #     pred_samples=None,
# # # # # # #     gt_norm=None,
# # # # # # #     weights=WEIGHTS,
# # # # # # #     intensity_w: Optional[torch.Tensor] = None,
# # # # # # #     env_data: Optional[dict] = None,
# # # # # # #     step_weight_alpha: float = 0.0,
# # # # # # #     all_trajs: Optional[torch.Tensor] = None,
# # # # # # # ) -> Dict:
# # # # # # #     # 1. Sample weights
# # # # # # #     recurv_w = _recurvature_weights(gt, w_recurv=2.5)
# # # # # # #     sample_w = recurv_w * (intensity_w.to(gt.device) if intensity_w is not None
# # # # # # #                            else 1.0)
# # # # # # #     sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

# # # # # # #     # 2. AFCRPS
# # # # # # #     if pred_samples is not None:
# # # # # # #         if gt_norm is not None:
# # # # # # #             l_fm = fm_afcrps_loss(pred_samples, gt_norm,
# # # # # # #                                   intensity_w=sample_w, unit_01deg=True,
# # # # # # #                                   step_weight_alpha=step_weight_alpha)
# # # # # # #         else:
# # # # # # #             l_fm = fm_afcrps_loss(pred_samples, gt,
# # # # # # #                                   intensity_w=sample_w, unit_01deg=False,
# # # # # # #                                   step_weight_alpha=step_weight_alpha)
# # # # # # #     else:
# # # # # # #         l_fm = _haversine_deg(pred_abs, gt).mean()

# # # # # # #     # 3. Directional losses
# # # # # # #     NRM = 35.0

# # # # # # #     l_vel   = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
# # # # # # #     l_disp  = (disp_loss_per_sample(pred_abs, gt)     * sample_w).mean()
# # # # # # #     l_step  = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()

# # # # # # #     l_heading   = heading_loss(pred_abs, gt)
# # # # # # #     l_recurv    = recurvature_loss(pred_abs, gt)
# # # # # # #     l_dir_final = overall_dir_loss(pred_abs, gt, ref)
# # # # # # #     l_smooth    = smooth_loss(pred_abs)
# # # # # # #     l_accel     = acceleration_loss(pred_abs)
# # # # # # #     l_jerk      = jerk_loss(pred_abs)  # FIX-L48

# # # # # # #     # 4. PINN
# # # # # # #     _env = env_data
# # # # # # #     if _env is None and batch_list is not None:
# # # # # # #         try:
# # # # # # #             _env = batch_list[13]
# # # # # # #         except (IndexError, TypeError):
# # # # # # #             _env = None

# # # # # # #     l_pinn = pinn_bve_loss(pred_abs, batch_list, env_data=_env)

# # # # # # #     # 5. Spread penalty
# # # # # # #     l_spread = pred_abs.new_zeros(())
# # # # # # #     if all_trajs is not None and all_trajs.shape[0] >= 2:
# # # # # # #         l_spread = ensemble_spread_loss(all_trajs, max_spread_km=200.0)

# # # # # # #     # 6. Total
# # # # # # #     total = (
# # # # # # #         weights.get("fm",       2.0) * l_fm
# # # # # # #         + weights.get("velocity", 0.8) * l_vel   * NRM
# # # # # # #         + weights.get("disp",     0.5) * l_disp  * NRM
# # # # # # #         + weights.get("step",     0.5) * l_step  * NRM
# # # # # # #         + weights.get("heading",  2.0) * l_heading * NRM
# # # # # # #         + weights.get("recurv",   1.5) * l_recurv  * NRM
# # # # # # #         + weights.get("dir",      1.0) * l_dir_final * NRM
# # # # # # #         + weights.get("smooth",   0.5) * l_smooth * NRM
# # # # # # #         + weights.get("accel",    0.8) * l_accel  * NRM
# # # # # # #         + weights.get("jerk",     0.3) * l_jerk   * NRM  # FIX-L48
# # # # # # #         + weights.get("pinn",    0.02) * l_pinn
# # # # # # #         + weights.get("spread",  0.3)  * l_spread * NRM
# # # # # # #     ) / NRM

# # # # # # #     if torch.isnan(total) or torch.isinf(total):
# # # # # # #         total = pred_abs.new_zeros(())

# # # # # # #     return dict(
# # # # # # #         total        = total,
# # # # # # #         fm           = l_fm.item(),
# # # # # # #         velocity     = l_vel.item() * NRM,
# # # # # # #         step         = l_step.item(),
# # # # # # #         disp         = l_disp.item() * NRM,
# # # # # # #         heading      = l_heading.item(),
# # # # # # #         recurv       = l_recurv.item(),
# # # # # # #         smooth       = l_smooth.item() * NRM,
# # # # # # #         accel        = l_accel.item() * NRM,
# # # # # # #         jerk         = l_jerk.item() * NRM,
# # # # # # #         pinn         = l_pinn.item(),
# # # # # # #         spread       = l_spread.item() * NRM,
# # # # # # #         recurv_ratio = (_recurvature_weights(gt) > 1.0).float().mean().item(),
# # # # # # #     )


# # # # # # # # ── Legacy ─────────────────────────────────────────────────────────────────────

# # # # # # # class TripletLoss(torch.nn.Module):
# # # # # # #     def __init__(self, margin=None):
# # # # # # #         super().__init__()
# # # # # # #         self.margin  = margin
# # # # # # #         self.loss_fn = (torch.nn.SoftMarginLoss() if margin is None
# # # # # # #                         else torch.nn.TripletMarginLoss(margin=margin, p=2))

# # # # # # #     def forward(self, anchor, pos, neg):
# # # # # # #         if self.margin is None:
# # # # # # #             y = torch.ones(anchor.shape[0], device=anchor.device)
# # # # # # #             return self.loss_fn(
# # # # # # #                 torch.norm(anchor - neg, 2, dim=1)
# # # # # # #                 - torch.norm(anchor - pos, 2, dim=1), y)
# # # # # # #         return self.loss_fn(anchor, pos, neg)

# # # # # # """
# # # # # # Model/losses.py  ── v25
# # # # # # ========================
# # # # # # FIXES vs v24:

# # # # # #   FIX-L49  [CRITICAL] short_range_regression_loss(): loss mới riêng cho
# # # # # #            4 bước đầu (6h/12h/18h/24h). Dùng Huber loss trên khoảng cách
# # # # # #            Haversine. Weight 12h = 4.0, 6h/18h/24h = 2.0.
# # # # # #            Không dùng AFCRPS cho short-range → ổn định và chính xác hơn.

# # # # # #   FIX-L50  [HIGH] PINN scale: 1e-3 → 1e-2.
# # # # # #            Log cho thấy pinn=15.232 không đổi → scale quá nhỏ, loss bị
# # # # # #            dominate bởi phần tanh-clamp. Tăng scale làm residual nhỏ hơn
# # # # # #            → gradient flow tốt hơn → PINN thực sự học.

# # # # # #   FIX-L51  [HIGH] WEIGHTS: thêm "short_range": 5.0, tăng "spread": 0.8
# # # # # #            (từ 0.6). Spread 400-500km là nguồn gốc của ADE cao.

# # # # # #   FIX-L52  [MEDIUM] ensemble_spread_loss: max_spread_km=150 (từ 200),
# # # # # #            FIX-L46b vẫn giữ exponential decay trên step weights.

# # # # # #   FIX-L53  [MEDIUM] fm_afcrps_loss: early_w decay rate 0.35 → 0.5
# # # # # #            khi alpha > 0 → focus mạnh hơn vào step 1-2 trong giai đoạn
# # # # # #            đầu training.

# # # # # # Kept from v24:
# # # # # #   FIX-L44  tanh soft-clamp (max_val=20.0)
# # # # # #   FIX-L45  pinn_rankine_steering scale u/v500
# # # # # #   FIX-L47  velocity direction penalty
# # # # # #   FIX-L48  jerk loss
# # # # # # """
# # # # # # from __future__ import annotations

# # # # # # import math
# # # # # # from typing import Dict, Optional

# # # # # # import torch
# # # # # # import torch.nn.functional as F

# # # # # # __all__ = [
# # # # # #     "WEIGHTS", "compute_total_loss", "fm_afcrps_loss",
# # # # # #     "fm_physics_consistency_loss", "pinn_bve_loss",
# # # # # #     "recurvature_loss", "velocity_loss_per_sample",
# # # # # #     "short_range_regression_loss",
# # # # # # ]

# # # # # # OMEGA        = 7.2921e-5
# # # # # # R_EARTH      = 6.371e6
# # # # # # DT_6H        = 6 * 3600
# # # # # # DEG_TO_KM    = 111.0
# # # # # # STEP_KM      = 113.0

# # # # # # # Các hằng số cho PINN mới của bạn
# # # # # # _UV500_NORM      = 30.0    # m/s → đã normalize [-1,1] trong env
# # # # # # _GPH500_MEAN_M   = 5870.0  # meters (sau fix Bug C)
# # # # # # _GPH500_STD_M    = 80.0
# # # # # # _STEERING_MIN_MS = 3.0     # m/s — threshold có steering flow thực sự
# # # # # # _GPH_GRAD_SCALE  = 200.0   # meters — scale GPH gradient
# # # # # # _PINN_SCALE      = 1e-2  # Scale để residual không bị quá nhỏ
# # # # # # # ── Weights (v25) ─────────────────────────────────────────────────────────────
# # # # # # WEIGHTS: Dict[str, float] = dict(
# # # # # #     fm          = 2.0,
# # # # # #     velocity    = 0.8,
# # # # # #     heading     = 2.0,
# # # # # #     recurv      = 1.5,
# # # # # #     step        = 0.5,
# # # # # #     disp        = 0.5,
# # # # # #     dir         = 1.0,
# # # # # #     smooth      = 0.5,
# # # # # #     accel       = 0.8,
# # # # # #     jerk        = 0.3,
# # # # # #     pinn        = 0.5,
# # # # # #     fm_physics  = 0.3,
# # # # # #     spread      = 0.8,        # ↑ từ 0.6  (FIX-L51)
# # # # # #     short_range = 5.0,        # NEW       (FIX-L49)
# # # # # # )

# # # # # # RECURV_ANGLE_THR = 45.0
# # # # # # RECURV_WEIGHT    = 2.5

# # # # # # # ── short-range config ────────────────────────────────────────────────────────
# # # # # # _SR_N_STEPS  = 4                         # 6h, 12h, 18h, 24h
# # # # # # _SR_WEIGHTS  = [2.0, 4.0, 2.0, 2.0]     # step-2 (12h) highest
# # # # # # _HUBER_DELTA = 50.0                      # km


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Haversine
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def _haversine(p1: torch.Tensor, p2: torch.Tensor,
# # # # # #                unit_01deg: bool = True) -> torch.Tensor:
# # # # # #     if unit_01deg:
# # # # # #         lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # #         lat1 = (p1[..., 1] * 50.0) / 10.0
# # # # # #         lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # #         lat2 = (p2[..., 1] * 50.0) / 10.0
# # # # # #     else:
# # # # # #         lon1, lat1 = p1[..., 0], p1[..., 1]
# # # # # #         lon2, lat2 = p2[..., 0], p2[..., 1]

# # # # # #     lat1r = torch.deg2rad(lat1);  lat2r = torch.deg2rad(lat2)
# # # # # #     dlon  = torch.deg2rad(lon2 - lon1)
# # # # # #     dlat  = torch.deg2rad(lat2 - lat1)
# # # # # #     a     = (torch.sin(dlat / 2) ** 2
# # # # # #              + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2) ** 2)
# # # # # #     return 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())


# # # # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # #     return _haversine(p1, p2, unit_01deg=False)


# # # # # # def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
# # # # # #     """Normalised → degrees. Accepts any leading dims."""
# # # # # #     out = arr.clone()
# # # # # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# # # # # #     return out


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  FIX-L49: Short-range regression loss
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def short_range_regression_loss(
# # # # # #     pred_sr:  torch.Tensor,   # [4, B, 2]  normalised positions
# # # # # #     gt_sr:    torch.Tensor,   # [4, B, 2]  normalised positions
# # # # # #     last_pos: torch.Tensor,   # [B, 2]     (unused, kept for API compat)
# # # # # # ) -> torch.Tensor:
# # # # # #     """
# # # # # #     FIX-L49: Huber loss on haversine distance for steps 1-4.

# # # # # #     Steps : 6h(1)  12h(2)  18h(3)  24h(4)
# # # # # #     Weight:  2.0    4.0     2.0     2.0    ← step-2(12h) penalised most

# # # # # #     Huber threshold = 50 km (less sensitive to outlier trajectories).
# # # # # #     Returns scalar loss in [0, ~10] range suitable for weight=5.0.
# # # # # #     """
# # # # # #     n_steps = min(pred_sr.shape[0], gt_sr.shape[0], _SR_N_STEPS)
# # # # # #     if n_steps == 0:
# # # # # #         return pred_sr.new_zeros(())

# # # # # #     pred_deg = _norm_to_deg(pred_sr[:n_steps])   # [n_steps, B, 2]
# # # # # #     gt_deg   = _norm_to_deg(gt_sr[:n_steps])

# # # # # #     # Haversine per (step, batch)
# # # # # #     dist_km = _haversine_deg(pred_deg, gt_deg)   # [n_steps, B]

# # # # # #     # Huber loss
# # # # # #     huber = torch.where(
# # # # # #         dist_km < _HUBER_DELTA,
# # # # # #         0.5 * dist_km.pow(2) / _HUBER_DELTA,
# # # # # #         dist_km - 0.5 * _HUBER_DELTA,
# # # # # #     )  # [n_steps, B]

# # # # # #     # Step weights
# # # # # #     w = pred_sr.new_tensor(_SR_WEIGHTS[:n_steps])  # [n_steps]
# # # # # #     weighted = (huber * w.view(-1, 1)).mean()

# # # # # #     # Normalise: divide by HUBER_DELTA so loss ≈ 1.0 when error ≈ 50 km
# # # # # #     return weighted / _HUBER_DELTA


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Step displacements
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
# # # # # #     dt      = traj_deg[1:] - traj_deg[:-1]
# # # # # #     lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
# # # # # #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
# # # # # #     dt_km   = dt.clone()
# # # # # #     dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
# # # # # #     dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
# # # # # #     return dt_km


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Recurvature helpers
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def _total_rotation_angle_batch(gt: torch.Tensor) -> torch.Tensor:
# # # # # #     T, B, _ = gt.shape
# # # # # #     if T < 3:
# # # # # #         return gt.new_zeros(B)
# # # # # #     lats_rad = torch.deg2rad(gt[:, :, 1])
# # # # # #     cos_lat  = torch.cos(lats_rad[:-1])
# # # # # #     dlat = gt[1:, :, 1] - gt[:-1, :, 1]
# # # # # #     dlon = (gt[1:, :, 0] - gt[:-1, :, 0]) * cos_lat
# # # # # #     v    = torch.stack([dlon, dlat], dim=-1)
# # # # # #     v1   = v[:-1];  v2 = v[1:]
# # # # # #     n1   = v1.norm(dim=-1).clamp(min=1e-8)
# # # # # #     n2   = v2.norm(dim=-1).clamp(min=1e-8)
# # # # # #     cos_a = (v1 * v2).sum(-1) / (n1 * n2)
# # # # # #     return torch.rad2deg(torch.acos(cos_a.clamp(-1.0, 1.0))).sum(0)


# # # # # # def _recurvature_weights(gt: torch.Tensor,
# # # # # #                          thr: float = RECURV_ANGLE_THR,
# # # # # #                          w_recurv: float = RECURV_WEIGHT) -> torch.Tensor:
# # # # # #     rot = _total_rotation_angle_batch(gt)
# # # # # #     return torch.where(rot >= thr,
# # # # # #                        torch.full_like(rot, w_recurv),
# # # # # #                        torch.ones_like(rot))


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Directional losses
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def velocity_loss_per_sample(pred: torch.Tensor,
# # # # # #                              gt: torch.Tensor) -> torch.Tensor:
# # # # # #     if pred.shape[0] < 2:
# # # # # #         return pred.new_zeros(pred.shape[1])
# # # # # #     v_pred_km = _step_displacements_km(pred)
# # # # # #     v_gt_km   = _step_displacements_km(gt)
# # # # # #     s_pred    = v_pred_km.norm(dim=-1)
# # # # # #     s_gt      = v_gt_km.norm(dim=-1)
# # # # # #     l_speed   = (s_pred - s_gt).pow(2).mean(0)
# # # # # #     gn  = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # #     gt_unit = v_gt_km / gn
# # # # # #     ate = ((v_pred_km - v_gt_km) * gt_unit).sum(-1)
# # # # # #     l_ate = ate.pow(2).mean(0)
# # # # # #     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # #     cos_sim = ((v_pred_km / pn) * gt_unit).sum(-1)
# # # # # #     l_dir = F.relu(-cos_sim).pow(2).mean(0) * STEP_KM ** 2
# # # # # #     return (l_speed + 0.5 * l_ate + 0.3 * l_dir) / (STEP_KM ** 2)


# # # # # # def disp_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # # # #     if pred.shape[0] < 2:
# # # # # #         return pred.new_zeros(pred.shape[1])
# # # # # #     pd = _step_displacements_km(pred).norm(dim=-1).mean(0)
# # # # # #     gd = _step_displacements_km(gt).norm(dim=-1).mean(0)
# # # # # #     return (pd - gd).pow(2) / (STEP_KM ** 2)


# # # # # # def step_dir_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # # # #     if pred.shape[0] < 2:
# # # # # #         return pred.new_zeros(pred.shape[1])
# # # # # #     v_pred_km = _step_displacements_km(pred)
# # # # # #     v_gt_km   = _step_displacements_km(gt)
# # # # # #     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # #     gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # #     cos_sim = ((v_pred_km / pn) * (v_gt_km / gn)).sum(-1)
# # # # # #     return (1.0 - cos_sim).mean(0)


# # # # # # def heading_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # # # #     if pred.shape[0] < 2:
# # # # # #         return pred.new_zeros(())
# # # # # #     pv = _step_displacements_km(pred)
# # # # # #     gv = _step_displacements_km(gt)
# # # # # #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # #     wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1))
# # # # # #     wrong_dir_loss = (wrong_dir ** 2).mean()

# # # # # #     if pred.shape[0] >= 3:
# # # # # #         def _curv(v):
# # # # # #             cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
# # # # # #             n1    = v[1:].norm(dim=-1).clamp(min=1e-4)
# # # # # #             n2    = v[:-1].norm(dim=-1).clamp(min=1e-4)
# # # # # #             return (cross / (n1 * n2)).clamp(-10.0, 10.0)
# # # # # #         curv_mse = F.mse_loss(_curv(pv), _curv(gv))
# # # # # #     else:
# # # # # #         curv_mse = pred.new_zeros(())
# # # # # #     return wrong_dir_loss + curv_mse


# # # # # # def smooth_loss(pred: torch.Tensor) -> torch.Tensor:
# # # # # #     if pred.shape[0] < 3:
# # # # # #         return pred.new_zeros(())
# # # # # #     v_km = _step_displacements_km(pred)
# # # # # #     if v_km.shape[0] < 2:
# # # # # #         return pred.new_zeros(())
# # # # # #     accel_km = v_km[1:] - v_km[:-1]
# # # # # #     return accel_km.pow(2).mean() / (STEP_KM ** 2)


# # # # # # def acceleration_loss(pred: torch.Tensor) -> torch.Tensor:
# # # # # #     if pred.shape[0] < 3:
# # # # # #         return pred.new_zeros(())
# # # # # #     v_km = _step_displacements_km(pred)
# # # # # #     if v_km.shape[0] < 2:
# # # # # #         return pred.new_zeros(())
# # # # # #     a_km = v_km[1:] - v_km[:-1]
# # # # # #     return a_km.pow(2).mean() / (STEP_KM ** 2)


# # # # # # def jerk_loss(pred: torch.Tensor) -> torch.Tensor:
# # # # # #     if pred.shape[0] < 4:
# # # # # #         return pred.new_zeros(())
# # # # # #     v_km = _step_displacements_km(pred)
# # # # # #     if v_km.shape[0] < 3:
# # # # # #         return pred.new_zeros(())
# # # # # #     a_km = v_km[1:] - v_km[:-1]
# # # # # #     j_km = a_km[1:] - a_km[:-1]
# # # # # #     return j_km.pow(2).mean() / (STEP_KM ** 2)


# # # # # # def overall_dir_loss(pred: torch.Tensor, gt: torch.Tensor,
# # # # # #                      ref: torch.Tensor) -> torch.Tensor:
# # # # # #     p_d = pred[-1] - ref
# # # # # #     g_d = gt[-1]   - ref
# # # # # #     lat_ref = ref[:, 1]
# # # # # #     cos_lat = torch.cos(torch.deg2rad(lat_ref)).clamp(min=1e-4)
# # # # # #     p_d_km  = p_d.clone()
# # # # # #     p_d_km[:, 0] *= cos_lat * DEG_TO_KM
# # # # # #     p_d_km[:, 1] *= DEG_TO_KM
# # # # # #     g_d_km  = g_d.clone()
# # # # # #     g_d_km[:, 0] *= cos_lat * DEG_TO_KM
# # # # # #     g_d_km[:, 1] *= DEG_TO_KM
# # # # # #     pn = p_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # #     gn = g_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # # #     return (1.0 - ((p_d_km / pn) * (g_d_km / gn)).sum(-1)).mean()


# # # # # # def recurvature_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # # # #     if pred.shape[0] < 3:
# # # # # #         return pred.new_zeros(())
# # # # # #     pred_v   = pred[1:] - pred[:-1]
# # # # # #     gt_v     = gt[1:]   - gt[:-1]
# # # # # #     gt_cross = (gt_v[:-1, :, 0] * gt_v[1:, :, 1]
# # # # # #               - gt_v[:-1, :, 1] * gt_v[1:, :, 0])
# # # # # #     if gt_cross.shape[0] < 2:
# # # # # #         return pred.new_zeros(())
# # # # # #     sign_change = (gt_cross[:-1] * gt_cross[1:]) < 0
# # # # # #     if not sign_change.any():
# # # # # #         return pred.new_zeros(())
# # # # # #     pred_v_mid = pred_v[1:-1]
# # # # # #     gt_v_mid   = gt_v[1:-1]
# # # # # #     if pred_v_mid.shape[0] > sign_change.shape[0]:
# # # # # #         pred_v_mid = pred_v_mid[:sign_change.shape[0]]
# # # # # #         gt_v_mid   = gt_v_mid[:sign_change.shape[0]]
# # # # # #     pn      = pred_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # #     gn      = gt_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # #     cos_sim = ((pred_v_mid / pn) * (gt_v_mid / gn)).sum(-1)
# # # # # #     dir_loss = (1.0 - cos_sim)
# # # # # #     mask     = sign_change.float()
# # # # # #     if mask.sum() < 1:
# # # # # #         return pred.new_zeros(())
# # # # # #     return (dir_loss * mask).sum() / mask.sum().clamp(min=1)


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  AFCRPS
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def fm_afcrps_loss(
# # # # # #     pred_samples:      torch.Tensor,
# # # # # #     gt:                torch.Tensor,
# # # # # #     unit_01deg:        bool  = True,
# # # # # #     intensity_w:       Optional[torch.Tensor] = None,
# # # # # #     step_weight_alpha: float = 0.0,
# # # # # # ) -> torch.Tensor:
# # # # # #     M, T, B, _ = pred_samples.shape
# # # # # #     eps = 1e-3

# # # # # #     base_w = torch.linspace(0.5, 1.5, T, device=pred_samples.device)

# # # # # #     if step_weight_alpha > 0.0:
# # # # # #         # FIX-L53: stronger early focus (0.35 → 0.5)
# # # # # #         early_w = torch.exp(
# # # # # #             -torch.arange(T, dtype=torch.float, device=pred_samples.device) * 0.5
# # # # # #         )
# # # # # #         early_w = early_w / early_w.mean()
# # # # # #         time_w  = (1.0 - step_weight_alpha) * base_w + step_weight_alpha * early_w
# # # # # #     else:
# # # # # #         time_w = base_w

# # # # # #     if M == 1:
# # # # # #         dist = _haversine(pred_samples[0], gt, unit_01deg)
# # # # # #         loss_per_b = (dist * time_w.unsqueeze(1)).mean(0)
# # # # # #     else:
# # # # # #         d_to_gt = _haversine(
# # # # # #             pred_samples,
# # # # # #             gt.unsqueeze(0).expand_as(pred_samples),
# # # # # #             unit_01deg,
# # # # # #         )
# # # # # #         d_to_gt_w    = d_to_gt * time_w.unsqueeze(0).unsqueeze(2)
# # # # # #         d_to_gt_mean = d_to_gt_w.mean(1)

# # # # # #         ps_i = pred_samples.unsqueeze(1)
# # # # # #         ps_j = pred_samples.unsqueeze(0)
# # # # # #         d_pair = _haversine(
# # # # # #             ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # # # # #             ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # # # # #             unit_01deg,
# # # # # #         ).reshape(M, M, T, B)
# # # # # #         d_pair_w    = d_pair * time_w.unsqueeze(0).unsqueeze(0).unsqueeze(3)
# # # # # #         d_pair_mean = d_pair_w.mean(2)

# # # # # #         e_sy  = d_to_gt_mean.mean(0)
# # # # # #         e_ssp = d_pair_mean.mean(0).mean(0)
# # # # # #         loss_per_b = (e_sy - 0.5 * e_ssp).clamp(min=eps)

# # # # # #     if intensity_w is not None:
# # # # # #         w = intensity_w.to(loss_per_b.device)
# # # # # #         return (loss_per_b * w).mean()
# # # # # #     return loss_per_b.mean()


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  PINN losses  (FIX-L50: scale 1e-3 → 1e-2)
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def _soft_clamp_loss(loss: torch.Tensor, max_val: float = 20.0) -> torch.Tensor:
# # # # # #     """FIX-L44: tanh soft-clamp. Gradient always > 0."""
# # # # # #     return max_val * torch.tanh(loss / max_val)

# # # # # # def _get_uv500_ms(env_data: dict, key_mean: str, key_center: str,
# # # # # #                   T_tgt: int, B: int,
# # # # # #                   device: torch.device) -> torch.Tensor:
# # # # # #     """
# # # # # #     Lấy u hoặc v 500hPa theo thứ tự ưu tiên: center > mean > zeros.
# # # # # #     env_data[key] shape: [B, T_obs, 1] (từ seq_collate).
# # # # # #     Output: [T_tgt, B] đơn vị m/s.
# # # # # #     """
# # # # # #     def _extract(key):
# # # # # #         x = env_data.get(key, None)
# # # # # #         if x is None or not torch.is_tensor(x):
# # # # # #             return None
# # # # # #         x = x.to(device).float()
# # # # # #         # [B, T_obs, 1] → [B, T_obs]
# # # # # #         if x.dim() == 3:
# # # # # #             x = x[..., 0]
# # # # # #         elif x.dim() == 1:
# # # # # #             x = x.unsqueeze(0).expand(B, -1)
# # # # # #         # x: [B, T_obs] → transpose → [T_obs, B]
# # # # # #         x = x.permute(1, 0)   # [T_obs, B]
# # # # # #         T_obs = x.shape[0]
# # # # # #         if T_obs >= T_tgt:
# # # # # #             return x[:T_tgt] * _UV500_NORM
# # # # # #         # pad bằng climatology (0 m/s) thay vì repeat cuối
# # # # # #         pad = torch.zeros(T_tgt - T_obs, B, device=device)
# # # # # #         return torch.cat([x * _UV500_NORM, pad], dim=0)

# # # # # #     # Ưu tiên center (flow tại tâm TC chính xác hơn)
# # # # # #     val = _extract(key_center)
# # # # # #     if val is not None:
# # # # # #         return val
# # # # # #     val = _extract(key_mean)
# # # # # #     if val is not None:
# # # # # #         return val
# # # # # #     return torch.zeros(T_tgt, B, device=device)


# # # # # # def _get_gph500_norm(env_data: dict, key: str,
# # # # # #                      T_tgt: int, B: int,
# # # # # #                      device: torch.device) -> torch.Tensor:
# # # # # #     """
# # # # # #     Lấy GPH500 đã z-score (sau fix Bug C: mean=5870, std=80).
# # # # # #     Output: [T_tgt, B] normalized.
# # # # # #     """
# # # # # #     x = env_data.get(key, None)
# # # # # #     if x is None or not torch.is_tensor(x):
# # # # # #         return torch.zeros(T_tgt, B, device=device)
# # # # # #     x = x.to(device).float()
# # # # # #     if x.dim() == 3:
# # # # # #         x = x[..., 0]
# # # # # #     x = x.permute(1, 0)   # [T_obs, B]
# # # # # #     T_obs = x.shape[0]
# # # # # #     if T_obs >= T_tgt:
# # # # # #         return x[:T_tgt]
# # # # # #     pad = torch.zeros(T_tgt - T_obs, B, device=device)
# # # # # #     return torch.cat([x, pad], dim=0)


# # # # # # # def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
# # # # # # #     T, B, _ = pred_abs_deg.shape
# # # # # # #     if T < 3:
# # # # # # #         return pred_abs_deg.new_zeros(())

# # # # # # #     DT      = DT_6H
# # # # # # #     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
# # # # # # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # # # # #     cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

# # # # # # #     u = dlat[:, :, 0] * cos_lat * 111000.0 / DT   # m/s
# # # # # # #     v = dlat[:, :, 1] * 111000.0 / DT              # m/s

# # # # # # #     if u.shape[0] < 2:
# # # # # # #         return pred_abs_deg.new_zeros(())

# # # # # # #     du = (u[1:] - u[:-1]) / DT   # m/s²  ~1e-4
# # # # # # #     dv = (v[1:] - v[:-1]) / DT

# # # # # # #     f    = 2 * OMEGA * torch.sin(lat_rad[1:-1])    # ~1e-4
# # # # # # #     beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH  # ~2e-11

# # # # # # #     res_u = du - f * v[1:]
# # # # # # #     res_v = dv + f * u[1:]

# # # # # # #     R_tc            = 3e5
# # # # # # #     v_beta_x        = -beta * R_tc ** 2 / 2        # ~1e-4 m/s²
# # # # # # #     res_u_corrected = res_u - v_beta_x

# # # # # # #     # # FIX: scale phù hợp với magnitude m/s²
# # # # # # #     # # Typical TC: du~1e-4, f*v~5e-4 → residual ~1e-3 m/s²
# # # # # # #     # # scale=1e-3 → res/scale ~1 → loss ~1 → tanh không bão hòa
# # # # # # #     # Scale = typical TC acceleration magnitude
# # # # # # #     # TC speed ~5 m/s, changes over 6h → du ~ 5/21600 ~ 2e-4 m/s²
# # # # # # #     # f*v ~ 1e-4 * 5 ~ 5e-4 m/s²
# # # # # # #     # Tổng residual ~ 1e-3 m/s²
# # # # # # #     # Scale = 1.0 m/s² → normalized residual ~ 1e-3 → loss ~ 1e-6 (quá nhỏ)
# # # # # # #     # Scale = 1e-3 m/s² → normalized ~ 1 → loss ~ 1 ✓
# # # # # # #     # Nhưng khi trajectory sai (rand init): speed ~500 m/s → du ~500/21600 ~0.02
# # # # # # #     # → res ~ 0.02 / 1e-3 = 20 → loss = 400 → clamp/tanh cần max_val lớn

# # # # # # #     # Giải pháp: normalize bằng magnitude thực tế của residual
# # # # # # #     scale = torch.sqrt(
# # # # # # #         res_u_corrected.detach().pow(2).mean() +
# # # # # # #         res_v.detach().pow(2).mean()
# # # # # # #     ).clamp(min=1e-6)

# # # # # # #     loss = (res_u_corrected.pow(2).mean() + res_v.pow(2).mean()) / (scale + 1e-8)

# # # # # # #     # loss ~ 1.0-2.0 khi residual đồng đều, > 2 khi một chiều lớn hơn
# # # # # # #     return loss.clamp(max=5.0)
# # # # # # def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
# # # # # #     T, B, _ = pred_abs_deg.shape
# # # # # #     if T < 3:
# # # # # #         return pred_abs_deg.new_zeros(())

# # # # # #     DT      = DT_6H
# # # # # #     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
# # # # # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # # # #     cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

# # # # # #     u = dlat[:, :, 0] * cos_lat * 111000.0 / DT   # m/s
# # # # # #     v = dlat[:, :, 1] * 111000.0 / DT

# # # # # #     if u.shape[0] < 2:
# # # # # #         return pred_abs_deg.new_zeros(())

# # # # # #     du = (u[1:] - u[:-1]) / DT   # m/s²
# # # # # #     dv = (v[1:] - v[:-1]) / DT

# # # # # #     f    = 2 * OMEGA * torch.sin(lat_rad[1:-1])
# # # # # #     beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH

# # # # # #     res_u = du - f * v[1:]
# # # # # #     res_v = dv + f * u[1:]

# # # # # #     R_tc            = 3e5
# # # # # #     v_beta_x        = -beta * R_tc ** 2 / 2
# # # # # #     res_u_corrected = res_u - v_beta_x

# # # # # #     # Fixed scale = 0.1 m/s²
# # # # # #     # Good TC  : residual ~1e-3 → loss ~1e-4 (nhỏ, không penalize)
# # # # # #     # Bad traj : residual ~0.1  → loss ~1.0  (penalize mạnh)
# # # # # #     # Very bad : residual ~1.0  → loss ~100  → soft_clamp giữ ở ~20
# # # # # #     scale = 0.1
# # # # # #     loss  = ((res_u_corrected / scale).pow(2).mean()
# # # # # #            + (res_v / scale).pow(2).mean())

# # # # # #     return _soft_clamp_loss(loss, max_val=20.0)

# # # # # # # def pinn_rankine_steering(pred_abs_deg: torch.Tensor,
# # # # # # #                           env_data: Optional[dict]) -> torch.Tensor:
# # # # # # #     if env_data is None:
# # # # # # #         return pred_abs_deg.new_zeros(())

# # # # # # #     T, B, _ = pred_abs_deg.shape
# # # # # # #     if T < 2:
# # # # # # #         return pred_abs_deg.new_zeros(())

# # # # # # #     DT      = DT_6H
# # # # # # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # # # # #     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
# # # # # # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

# # # # # # #     u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT
# # # # # # #     v_tc = dlat[:, :, 1] * 111000.0 / DT

# # # # # # #     u500_raw = env_data.get("u500_mean", None)
# # # # # # #     v500_raw = env_data.get("v500_mean", None)

# # # # # # #     if u500_raw is None or v500_raw is None:
# # # # # # #         speed = torch.sqrt(u_tc ** 2 + v_tc ** 2)
# # # # # # #         return F.relu(speed - 30.0).pow(2).mean() * 0.01

# # # # # # #     def _align(x: torch.Tensor) -> torch.Tensor:
# # # # # # #         if not torch.is_tensor(x):
# # # # # # #             return pred_abs_deg.new_zeros(T - 1, B)
# # # # # # #         x = x.to(pred_abs_deg.device)
# # # # # # #         if x.dim() == 3:
# # # # # # #             x_sq  = x[:, :, 0]
# # # # # # #             T_env = x_sq.shape[1]
# # # # # # #             T_tgt = T - 1
# # # # # # #             if T_env >= T_tgt:
# # # # # # #                 return x_sq[:, :T_tgt].permute(1, 0)
# # # # # # #             pad = x_sq[:, -1:].expand(-1, T_tgt - T_env)
# # # # # # #             return torch.cat([x_sq, pad], dim=1).permute(1, 0)
# # # # # # #         elif x.dim() == 2:
# # # # # # #             if x.shape == (B, T - 1):
# # # # # # #                 return x.permute(1, 0)
# # # # # # #             elif x.shape[0] == T - 1:
# # # # # # #                 return x[:, :B] if x.shape[1] >= B else x.expand(-1, B)
# # # # # # #         return pred_abs_deg.new_zeros(T - 1, B)

# # # # # # #     u500 = _align(u500_raw) * _UV500_SCALE
# # # # # # #     v500 = _align(v500_raw) * _UV500_SCALE

# # # # # # #     uv_magnitude = torch.sqrt(u500 ** 2 + v500 ** 2)
# # # # # # #     has_steering  = (uv_magnitude > 0.5).float()

# # # # # # #     env_dir  = torch.stack([u500, v500], dim=-1)
# # # # # # #     tc_dir   = torch.stack([u_tc, v_tc],  dim=-1)
# # # # # # #     env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
# # # # # # #     tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
# # # # # # #     cos_sim  = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)
# # # # # # #     misalign = F.relu(-0.5 - cos_sim).pow(2)

# # # # # # #     return (misalign * has_steering).mean() * 0.05



# # # # # # # def pinn_bve_loss(pred_abs_deg: torch.Tensor,
# # # # # # #                   batch_list,
# # # # # # #                   env_data: Optional[dict] = None) -> torch.Tensor:
# # # # # # #     T = pred_abs_deg.shape[0]
# # # # # # #     if T < 3:
# # # # # # #         return pred_abs_deg.new_zeros(())

# # # # # # #     l_sw    = pinn_shallow_water(pred_abs_deg)
# # # # # # #     l_steer = pinn_rankine_steering(pred_abs_deg, env_data)
# # # # # # #     l_speed = pinn_speed_constraint(pred_abs_deg)

# # # # # # #     total = l_sw + 0.5 * l_steer + 0.1 * l_speed
# # # # # # #     return _soft_clamp_loss(total, max_val=20.0)
# # # # # # # losses.py — thay toàn bộ phần PINN




# # # # # # def pinn_rankine_steering(pred_abs_deg: torch.Tensor,
# # # # # #                           env_data: Optional[dict]) -> torch.Tensor:
# # # # # #     """
# # # # # #     FIX: Dùng u/v500_center (ưu tiên) + mean làm steering vector.
# # # # # #     FIX: has_steering threshold 0.5 → 3.0 m/s.
# # # # # #     FIX: Padding bằng zeros (climatology) thay vì repeat obs cuối.
# # # # # #     """
# # # # # #     if env_data is None:
# # # # # #         return pred_abs_deg.new_zeros(())

# # # # # #     T, B, _ = pred_abs_deg.shape
# # # # # #     if T < 2:
# # # # # #         return pred_abs_deg.new_zeros(())

# # # # # #     device  = pred_abs_deg.device
# # # # # #     T_tgt   = T - 1
# # # # # #     DT      = DT_6H

# # # # # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # # # #     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
# # # # # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

# # # # # #     u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT   # [T-1, B] m/s
# # # # # #     v_tc = dlat[:, :, 1] * 111000.0 / DT              # [T-1, B] m/s

# # # # # #     # FIX: center > mean, cả hai nếu có
# # # # # #     u500 = _get_uv500_ms(env_data, "u500_mean", "u500_center",
# # # # # #                          T_tgt, B, device)   # [T-1, B] m/s
# # # # # #     v500 = _get_uv500_ms(env_data, "v500_mean", "v500_center",
# # # # # #                          T_tgt, B, device)

# # # # # #     uv_mag       = torch.sqrt(u500**2 + v500**2)
# # # # # #     # FIX: threshold 3.0 m/s thay vì 0.5
# # # # # #     has_steering = (uv_mag > _STEERING_MIN_MS).float()

# # # # # #     env_dir  = torch.stack([u500, v500], dim=-1)
# # # # # #     tc_dir   = torch.stack([u_tc,  v_tc], dim=-1)
# # # # # #     env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=1.0)
# # # # # #     tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
# # # # # #     cos_sim  = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)

# # # # # #     # Penalise chỉ khi hướng ngược (cos < -0.5) VÀ có steering thực sự
# # # # # #     misalign = F.relu(-0.5 - cos_sim).pow(2)
# # # # # #     return (misalign * has_steering).mean() * 0.05


# # # # # # def pinn_gph500_gradient(pred_abs_deg: torch.Tensor,
# # # # # #                          env_data: Optional[dict]) -> torch.Tensor:
# # # # # #     """
# # # # # #     MỚI: GPH500 gradient loss.

# # # # # #     Vật lý: TC di chuyển dọc theo đường đẳng áp (isobar) của GPH500.
# # # # # #     - gph500_center < gph500_mean → TC ở rãnh thấp → xu hướng poleward
# # # # # #     - gph500_center > gph500_mean → TC ở gờ cao    → xu hướng equatorward

# # # # # #     Loss: nếu GPH gradient chỉ hướng poleward nhưng TC đi equatorward
# # # # # #     (hoặc ngược lại) → penalise.

# # # # # #     gph500_mean/center đã z-score: (raw_m - 5870) / 80.
# # # # # #     Gradient = center - mean → đơn vị normalized.
# # # # # #     """
# # # # # #     if env_data is None:
# # # # # #         return pred_abs_deg.new_zeros(())
# # # # # #     T, B, _ = pred_abs_deg.shape
# # # # # #     if T < 2:
# # # # # #         return pred_abs_deg.new_zeros(())

# # # # # #     device = pred_abs_deg.device
# # # # # #     T_tgt  = T - 1

# # # # # #     gph_mean   = _get_gph500_norm(env_data, "gph500_mean",
# # # # # #                                   T_tgt, B, device)    # [T-1, B]
# # # # # #     gph_center = _get_gph500_norm(env_data, "gph500_center",
# # # # # #                                   T_tgt, B, device)    # [T-1, B]

# # # # # #     # gradient: center - mean (normalized units)
# # # # # #     # âm → TC ở vùng thấp hơn xung quanh → ridge ở phía bắc → đẩy TC về nam?
# # # # # #     # dương → TC ở vùng cao hơn xung quanh → trough ở phía bắc → đẩy TC về bắc
# # # # # #     gph_grad = gph_center - gph_mean   # [T-1, B]

# # # # # #     # TC lat tendency
# # # # # #     dlat   = pred_abs_deg[1:, :, 1] - pred_abs_deg[:-1, :, 1]  # [T-1, B] degrees
# # # # # #     # Chuẩn hóa lat tendency
# # # # # #     dlat_n = dlat / (dlat.abs().clamp(min=1e-4))  # sign only: +1 hay -1

# # # # # #     # Khi gph_grad < -0.1 (rãnh sâu phía bắc) → TC nên đi poleward (dlat > 0)
# # # # # #     # Khi gph_grad > +0.1 (ridge phía bắc)    → TC nên đi equatorward (dlat < 0)
# # # # # #     # Expected sign of dlat = sign of -gph_grad (rough heuristic)
# # # # # #     expected_sign = -torch.sign(gph_grad)
# # # # # #     has_gradient  = (gph_grad.abs() > 0.1).float()  # chỉ penalise khi gradient rõ

# # # # # #     # Penalise khi sign ngược
# # # # # #     wrong_dir = F.relu(-(dlat_n * expected_sign)).pow(2)
# # # # # #     return (wrong_dir * has_gradient).mean() * 0.02


# # # # # # def pinn_steering_speed_consistency(pred_abs_deg: torch.Tensor,
# # # # # #                                     env_data: Optional[dict]) -> torch.Tensor:
# # # # # #     """
# # # # # #     MỚI: TC speed phải gần với steering flow speed.

# # # # # #     Vật lý: TC thường di chuyển với tốc độ ≈ 0.5–0.8 × |steering flow|.
# # # # # #     Nếu TC di chuyển quá nhanh hoặc quá chậm so với steering → bất thường.

# # # # # #     Dùng cả u500_mean và u500_center để ước lượng steering magnitude.
# # # # # #     """
# # # # # #     if env_data is None:
# # # # # #         return pred_abs_deg.new_zeros(())
# # # # # #     T, B, _ = pred_abs_deg.shape
# # # # # #     if T < 2:
# # # # # #         return pred_abs_deg.new_zeros(())

# # # # # #     device = pred_abs_deg.device
# # # # # #     T_tgt  = T - 1
# # # # # #     DT     = DT_6H

# # # # # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # # # #     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
# # # # # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # #     dx_km   = dlat[:, :, 0] * cos_lat * 111.0
# # # # # #     dy_km   = dlat[:, :, 1] * 111.0
# # # # # #     tc_speed_ms = torch.sqrt(dx_km**2 + dy_km**2) * 1000.0 / DT  # m/s

# # # # # #     # Dùng mean của center và mean để ước lượng steering magnitude
# # # # # #     u_c = _get_uv500_ms(env_data, "u500_mean", "u500_center", T_tgt, B, device)
# # # # # #     v_c = _get_uv500_ms(env_data, "v500_mean", "v500_center", T_tgt, B, device)
# # # # # #     u_m = _get_uv500_ms(env_data, "u500_mean", "u500_mean",   T_tgt, B, device)
# # # # # #     v_m = _get_uv500_ms(env_data, "v500_mean", "v500_mean",   T_tgt, B, device)

# # # # # #     steering_mag = (torch.sqrt(u_c**2 + v_c**2) +
# # # # # #                     torch.sqrt(u_m**2 + v_m**2)) / 2.0   # m/s

# # # # # #     has_steering = (steering_mag > _STEERING_MIN_MS).float()

# # # # # #     # TC speed ≈ 0.5–1.0 × steering (empirical for WP TCs)
# # # # # #     lo = steering_mag * 0.3
# # # # # #     hi = steering_mag * 1.5
# # # # # #     too_slow = F.relu(lo - tc_speed_ms)
# # # # # #     too_fast = F.relu(tc_speed_ms - hi)

# # # # # #     penalty = (too_slow.pow(2) + too_fast.pow(2)) / (_UV500_NORM**2)
# # # # # #     return (penalty * has_steering).mean() * 0.03


# # # # # # def pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # #     if pred_deg.shape[0] < 2:
# # # # # #         return pred_deg.new_zeros(())
# # # # # #     dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # #     lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # #     dx_km   = dt_deg[:, :, 0] * cos_lat * DEG_TO_KM
# # # # # #     dy_km   = dt_deg[:, :, 1] * DEG_TO_KM
# # # # # #     speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # # # # #     return F.relu(speed - 600.0).pow(2).mean()

# # # # # # def pinn_bve_loss(pred_abs_deg: torch.Tensor,
# # # # # #                   batch_list,
# # # # # #                   env_data: Optional[dict] = None) -> torch.Tensor:
# # # # # #     """
# # # # # #     FIX: Tận dụng đầy đủ 6 Data3d features:
# # # # # #       u500_mean, u500_center → steering direction + speed consistency
# # # # # #       v500_mean, v500_center → steering direction + speed consistency
# # # # # #       gph500_mean, gph500_center → GPH gradient → lat tendency constraint

# # # # # #     Breakdown:
# # # # # #       l_sw    : shallow water equation residual (trajectory only)
# # # # # #       l_steer : steering direction alignment (u/v center+mean)
# # # # # #       l_speed : TC absolute speed cap (trajectory only)
# # # # # #       l_gph   : GPH gradient → lat tendency (gph center vs mean)
# # # # # #       l_spdcons: TC speed vs steering flow magnitude
# # # # # #     """
# # # # # #     T = pred_abs_deg.shape[0]
# # # # # #     if T < 3:
# # # # # #         return pred_abs_deg.new_zeros(())

# # # # # #     # env_data fallback
# # # # # #     _env = env_data
# # # # # #     if _env is None and batch_list is not None:
# # # # # #         try:
# # # # # #             _env = batch_list[13]
# # # # # #         except (IndexError, TypeError):
# # # # # #             _env = None

# # # # # #     l_sw      = pinn_shallow_water(pred_abs_deg)
# # # # # #     l_steer   = pinn_rankine_steering(pred_abs_deg, _env)
# # # # # #     l_speed   = pinn_speed_constraint(pred_abs_deg)
# # # # # #     l_gph     = pinn_gph500_gradient(pred_abs_deg, _env)
# # # # # #     l_spdcons = pinn_steering_speed_consistency(pred_abs_deg, _env)

# # # # # #     total = (l_sw
# # # # # #              + 0.5  * l_steer
# # # # # #              + 0.1  * l_speed
# # # # # #              + 0.3  * l_gph        # MỚI: GPH gradient
# # # # # #              + 0.4  * l_spdcons)   # MỚI: speed vs steering

# # # # # #     return _soft_clamp_loss(total, max_val=20.0)


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Physics consistency
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def fm_physics_consistency_loss(
# # # # # #     pred_samples: torch.Tensor,
# # # # # #     gt_norm:      torch.Tensor,
# # # # # #     last_pos:     torch.Tensor,
# # # # # # ) -> torch.Tensor:
# # # # # #     S, T, B = pred_samples.shape[:3]

# # # # # #     last_lon = (last_pos[:, 0] * 50.0 + 1800.0) / 10.0
# # # # # #     last_lat = (last_pos[:, 1] * 50.0) / 10.0
# # # # # #     lat_rad  = torch.deg2rad(last_lat)
# # # # # #     beta     = 2 * OMEGA * torch.cos(lat_rad) / R_EARTH
# # # # # #     R_tc     = 3e5

# # # # # #     v_beta_lon = -beta * R_tc ** 2 / 2
# # # # # #     v_beta_lat =  beta * R_tc ** 2 / 4

# # # # # #     beta_dir      = torch.stack([v_beta_lon, v_beta_lat], dim=-1)
# # # # # #     beta_norm     = beta_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # #     beta_dir_unit = beta_dir / beta_norm

# # # # # #     pos_step1 = pred_samples[:, 0, :, :2]
# # # # # #     dir_step1 = pos_step1 - last_pos.unsqueeze(0)
# # # # # #     dir_norm  = dir_step1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # #     dir_unit  = dir_step1 / dir_norm
# # # # # #     mean_dir  = dir_unit.mean(0)
# # # # # #     mean_norm = mean_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # #     mean_dir_unit = mean_dir / mean_norm

# # # # # #     cos_align     = (mean_dir_unit * beta_dir_unit).sum(-1)
# # # # # #     beta_strength = beta_norm.squeeze(-1)
# # # # # #     penalise_mask = (beta_strength > 1.0).float()
# # # # # #     direction_loss = F.relu(-cos_align) * penalise_mask
# # # # # #     return direction_loss.mean() * 0.5


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Spread regularization  (FIX-L52: max_spread_km 200→150)
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def ensemble_spread_loss(all_trajs: torch.Tensor,
# # # # # #                          max_spread_km: float = 150.0) -> torch.Tensor:
# # # # # #     """
# # # # # #     FIX-L52: max_spread_km=150 (từ 200). Exponential decay weights.
# # # # # #     all_trajs: [S, T, B, 2] normalised.
# # # # # #     """
# # # # # #     if all_trajs.shape[0] < 2:
# # # # # #         return all_trajs.new_zeros(())

# # # # # #     S, T, B, _ = all_trajs.shape

# # # # # #     # Exponential decay: early steps penalised more (from FIX-L46b)
# # # # # #     step_weights = torch.exp(
# # # # # #         -torch.arange(T, dtype=torch.float, device=all_trajs.device)
# # # # # #         * (math.log(4.0) / max(T - 1, 1))
# # # # # #     ) * 2.0   # range: 2.0 → 0.5

# # # # # #     total_loss = all_trajs.new_zeros(())
# # # # # #     for t in range(T):
# # # # # #         step_trajs = all_trajs[:, t, :, :2]
# # # # # #         std_lon    = step_trajs[:, :, 0].std(0)
# # # # # #         std_lat    = step_trajs[:, :, 1].std(0)
# # # # # #         spread_km  = torch.sqrt(std_lon ** 2 + std_lat ** 2) * 500.0
# # # # # #         excess     = F.relu(spread_km - max_spread_km)
# # # # # #         total_loss = total_loss + step_weights[t] * (
# # # # # #             excess / max_spread_km
# # # # # #         ).pow(2).mean()

# # # # # #     return total_loss / T


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Main loss
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def compute_total_loss(
# # # # # #     pred_abs,
# # # # # #     gt,
# # # # # #     ref,
# # # # # #     batch_list,
# # # # # #     pred_samples       = None,
# # # # # #     gt_norm            = None,
# # # # # #     weights            = WEIGHTS,
# # # # # #     intensity_w: Optional[torch.Tensor] = None,
# # # # # #     env_data:    Optional[dict]         = None,
# # # # # #     step_weight_alpha: float = 0.0,
# # # # # #     all_trajs:   Optional[torch.Tensor] = None,
# # # # # # ) -> Dict:
# # # # # #     # 1. Sample weights
# # # # # #     recurv_w = _recurvature_weights(gt, w_recurv=2.5)
# # # # # #     sample_w = recurv_w * (intensity_w.to(gt.device) if intensity_w is not None
# # # # # #                            else 1.0)
# # # # # #     sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

# # # # # #     # 2. AFCRPS
# # # # # #     if pred_samples is not None:
# # # # # #         if gt_norm is not None:
# # # # # #             l_fm = fm_afcrps_loss(pred_samples, gt_norm,
# # # # # #                                   intensity_w=sample_w, unit_01deg=True,
# # # # # #                                   step_weight_alpha=step_weight_alpha)
# # # # # #         else:
# # # # # #             l_fm = fm_afcrps_loss(pred_samples, gt,
# # # # # #                                   intensity_w=sample_w, unit_01deg=False,
# # # # # #                                   step_weight_alpha=step_weight_alpha)
# # # # # #     else:
# # # # # #         l_fm = _haversine_deg(pred_abs, gt).mean()

# # # # # #     # 3. Directional losses
# # # # # #     NRM = 35.0

# # # # # #     l_vel       = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
# # # # # #     l_disp      = (disp_loss_per_sample(pred_abs, gt)     * sample_w).mean()
# # # # # #     l_step      = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()
# # # # # #     l_heading   = heading_loss(pred_abs, gt)
# # # # # #     l_recurv    = recurvature_loss(pred_abs, gt)
# # # # # #     l_dir_final = overall_dir_loss(pred_abs, gt, ref)
# # # # # #     l_smooth    = smooth_loss(pred_abs)
# # # # # #     l_accel     = acceleration_loss(pred_abs)
# # # # # #     l_jerk      = jerk_loss(pred_abs)

# # # # # #     # 4. PINN
# # # # # #     _env = env_data
# # # # # #     if _env is None and batch_list is not None:
# # # # # #         try:
# # # # # #             _env = batch_list[13]
# # # # # #         except (IndexError, TypeError):
# # # # # #             _env = None

# # # # # #     l_pinn = pinn_bve_loss(pred_abs, batch_list, env_data=_env)

# # # # # #     # 5. Spread penalty  (FIX-L52)
# # # # # #     # l_spread = pred_abs.new_zeros(())
# # # # # #     # if all_trajs is not None and all_trajs.shape[0] >= 2:
# # # # # #     #     l_spread = ensemble_spread_loss(all_trajs, max_spread_km=150.0)

# # # # # #     # FIX: spread loss chỉ tính trên FM ensemble thật (không có sr_pred override)
# # # # # #     # all_trajs trong get_loss_breakdown là FM samples trước khi blend
# # # # # #     # → spread_loss phản ánh đúng ensemble diversity
# # # # # #     l_spread = pred_abs.new_zeros(())
# # # # # #     if all_trajs is not None and all_trajs.shape[0] >= 2:
# # # # # #         # Chỉ tính spread cho steps 5-12 (FM range)
# # # # # #         # Steps 1-4 do ShortRangeHead handle riêng
# # # # # #         n_sr = 4
# # # # # #         if all_trajs.shape[1] > n_sr:
# # # # # #             l_spread = ensemble_spread_loss(
# # # # # #                 all_trajs[:, n_sr:, :, :],   # chỉ steps 5-12
# # # # # #                 max_spread_km=150.0
# # # # # #             )
# # # # # #         else:
# # # # # #             # Fallback nếu traj quá ngắn (hiếm gặp)
# # # # # #             l_spread = ensemble_spread_loss(all_trajs, max_spread_km=150.0)


# # # # # #     # 6. Total  (short_range added externally in get_loss_breakdown)
# # # # # #     total = (
# # # # # #         weights.get("fm",       2.0) * l_fm
# # # # # #         + weights.get("velocity", 0.8) * l_vel     * NRM
# # # # # #         + weights.get("disp",     0.5) * l_disp    * NRM
# # # # # #         + weights.get("step",     0.5) * l_step    * NRM
# # # # # #         + weights.get("heading",  2.0) * l_heading * NRM
# # # # # #         + weights.get("recurv",   1.5) * l_recurv  * NRM
# # # # # #         + weights.get("dir",      1.0) * l_dir_final * NRM
# # # # # #         + weights.get("smooth",   0.5) * l_smooth  * NRM
# # # # # #         + weights.get("accel",    0.8) * l_accel   * NRM
# # # # # #         + weights.get("jerk",     0.3) * l_jerk    * NRM
# # # # # #         + weights.get("pinn",     0.5) * l_pinn    * NRM   # ← thêm * NRM
# # # # # #         + weights.get("spread",   0.8) * l_spread  * NRM
# # # # # #     ) / NRM

# # # # # #     if torch.isnan(total) or torch.isinf(total):
# # # # # #         total = pred_abs.new_zeros(())

# # # # # #     return dict(
# # # # # #         total        = total,
# # # # # #         fm           = l_fm.item(),
# # # # # #         velocity     = l_vel.item()     * NRM,
# # # # # #         step         = l_step.item(),
# # # # # #         disp         = l_disp.item()    * NRM,
# # # # # #         heading      = l_heading.item(),
# # # # # #         recurv       = l_recurv.item(),
# # # # # #         smooth       = l_smooth.item()  * NRM,
# # # # # #         accel        = l_accel.item()   * NRM,
# # # # # #         jerk         = l_jerk.item()    * NRM,
# # # # # #         pinn         = l_pinn.item(),
# # # # # #         spread       = l_spread.item()  * NRM,
# # # # # #         recurv_ratio = (_recurvature_weights(gt) > 1.0).float().mean().item(),
# # # # # #     )


# # # # # # # ── Legacy ─────────────────────────────────────────────────────────────────────

# # # # # # class TripletLoss(torch.nn.Module):
# # # # # #     def __init__(self, margin=None):
# # # # # #         super().__init__()
# # # # # #         self.margin  = margin
# # # # # #         self.loss_fn = (torch.nn.SoftMarginLoss() if margin is None
# # # # # #                         else torch.nn.TripletMarginLoss(margin=margin, p=2))

# # # # # #     def forward(self, anchor, pos, neg):
# # # # # #         if self.margin is None:
# # # # # #             y = torch.ones(anchor.shape[0], device=anchor.device)
# # # # # #             return self.loss_fn(
# # # # # #                 torch.norm(anchor - neg, 2, dim=1)
# # # # # #                 - torch.norm(anchor - pos, 2, dim=1), y)
# # # # # #         return self.loss_fn(anchor, pos, neg)

# # # # # """
# # # # # Model/losses.py  ── v26
# # # # # ========================
# # # # # FULL REWRITE – fixes tất cả vấn đề từ review v25:

# # # # #   FIX-L-A  [CRITICAL] pinn_bve_loss: bỏ nhân NRM trong compute_total_loss.
# # # # #            PINN đã có soft-clamp max=20, nhân thêm NRM=35 → overpower 35×.

# # # # #   FIX-L-B  [CRITICAL] AdaptClamp implement đúng theo Eq.58:
# # # # #            ep 0-9: Huber mode (gradient ≥ 1/20, không bão hòa).
# # # # #            ep 10-19: nội suy tuyến tính.
# # # # #            ep 20+: tanh mode.

# # # # #   FIX-L-C  [HIGH] Adaptive BVE weighting theo track error (Eq.99):
# # # # #            w_BVE,k = σ(1 - d_hav(pred,gt)/200km).
# # # # #            Tắt PINN khi track sai để tránh phạt nhầm.

# # # # #   FIX-L-D  [HIGH] L_PWR (pressure-wind balance, Eq.62-63): implement
# # # # #            đầy đủ với dynamic R_TC. Kích hoạt từ epoch 30.

# # # # #   FIX-L-E  [HIGH] Frequency compensation w_pinn_eff (Eq.100):
# # # # #            f_lazy schedule theo epoch.

# # # # #   FIX-L-F  [HIGH] Spatial boundary weighting w_bnd (Eq.63a-b):
# # # # #            Suy giảm PINN loss gần biên domain ERA5.

# # # # #   FIX-L-G  [HIGH] Energy Score term trong fm_afcrps_loss (Eq.77):
# # # # #            ES_norm(M) với unbiasing factor (M-1)/M.

# # # # #   FIX-L-H  [MEDIUM] L_bridge implement (Eq.80):
# # # # #            Nhất quán SR↔FM tại bước nối step 4.

# # # # #   FIX-L-I  [MEDIUM] pinn_gph500_gradient: fix logic vật lý.
# # # # #            center-mean không phải gradient 2D, dùng lat tendency đúng.

# # # # #   FIX-L-J  [MEDIUM] fm_afcrps_loss: bỏ clamp(min=eps) trên loss_per_b
# # # # #            để gradient flow khi ensemble tốt bất thường.

# # # # #   FIX-L-K  [LOW] haversine coordinate decode: thêm assert để phát hiện
# # # # #            sai đơn vị sớm.

# # # # # Kept from v25:
# # # # #   FIX-L49  short_range_regression_loss (Huber, step weights)
# # # # #   FIX-L44  soft-clamp tanh (dùng trong epoch 20+)
# # # # #   FIX-L45  pinn_rankine_steering với threshold 3.0 m/s
# # # # #   FIX-L47  velocity direction penalty
# # # # #   FIX-L48  jerk loss
# # # # #   FIX-L52  ensemble_spread_loss max_spread=150km
# # # # # """
# # # # # from __future__ import annotations

# # # # # import math
# # # # # from typing import Dict, Optional, Tuple

# # # # # import torch
# # # # # import torch.nn.functional as F

# # # # # __all__ = [
# # # # #     "WEIGHTS", "compute_total_loss", "fm_afcrps_loss",
# # # # #     "fm_physics_consistency_loss", "pinn_bve_loss",
# # # # #     "recurvature_loss", "velocity_loss_per_sample",
# # # # #     "short_range_regression_loss", "bridge_loss",
# # # # #     "adapt_clamp", "pinn_pressure_wind_loss",
# # # # # ]

# # # # # # ── Constants ─────────────────────────────────────────────────────────────────
# # # # # OMEGA        = 7.2921e-5
# # # # # R_EARTH      = 6.371e6
# # # # # DT_6H        = 6 * 3600
# # # # # DEG_TO_KM    = 111.0
# # # # # STEP_KM      = 113.0
# # # # # P_ENV        = 1013.0   # hPa
# # # # # RHO_AIR      = 1.15     # kg/m³

# # # # # # ERA5 domain bounds (degrees) – dùng cho boundary weighting
# # # # # _ERA5_LAT_MIN =   0.0
# # # # # _ERA5_LAT_MAX =  40.0
# # # # # _ERA5_LON_MIN = 100.0
# # # # # _ERA5_LON_MAX = 160.0

# # # # # _UV500_NORM      = 30.0
# # # # # _GPH500_MEAN_M   = 5870.0
# # # # # _GPH500_STD_M    = 80.0
# # # # # _STEERING_MIN_MS = 3.0
# # # # # _PINN_SCALE      = 1e-2

# # # # # # ── Weights ───────────────────────────────────────────────────────────────────
# # # # # WEIGHTS: Dict[str, float] = dict(
# # # # #     fm          = 2.0,
# # # # #     velocity    = 0.8,
# # # # #     heading     = 2.0,
# # # # #     recurv      = 1.5,
# # # # #     step        = 0.5,
# # # # #     disp        = 0.5,
# # # # #     dir         = 1.0,
# # # # #     smooth      = 0.5,
# # # # #     accel       = 0.8,
# # # # #     jerk        = 0.3,
# # # # #     pinn        = 0.5,
# # # # #     fm_physics  = 0.3,
# # # # #     spread      = 0.8,
# # # # #     short_range = 5.0,
# # # # #     bridge      = 0.5,    # NEW FIX-L-H
# # # # # )

# # # # # RECURV_ANGLE_THR = 45.0
# # # # # RECURV_WEIGHT    = 2.5

# # # # # _SR_N_STEPS  = 4
# # # # # _SR_WEIGHTS  = [2.0, 4.0, 2.0, 2.0]
# # # # # _HUBER_DELTA = 50.0


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  FIX-L-B: AdaptClamp đúng theo Eq.58
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def adapt_clamp(x: torch.Tensor, epoch: int, max_val: float = 20.0) -> torch.Tensor:
# # # # #     """
# # # # #     AdaptClamp_ep(x) theo Eq.58:
# # # # #       ep 0-9:  Huber mode  → gradient ≥ 1/max_val, không bão hòa
# # # # #       ep 10-19: nội suy tuyến tính Huber → tanh
# # # # #       ep 20+:  tanh mode   → mượt mà, ổn định convergence

# # # # #     HuberClamp(x, δ):
# # # # #       x ≤ δ : x²/(2δ)       [quadratic, gradient = x/δ]
# # # # #       x > δ : x - δ/2       [linear,    gradient = 1]
# # # # #     """
# # # # #     delta = max_val

# # # # #     def huber_clamp(v: torch.Tensor) -> torch.Tensor:
# # # # #         return torch.where(
# # # # #             v <= delta,
# # # # #             v.pow(2) / (2.0 * delta),
# # # # #             v - delta / 2.0,
# # # # #         )

# # # # #     def tanh_clamp(v: torch.Tensor) -> torch.Tensor:
# # # # #         return delta * torch.tanh(v / delta)

# # # # #     if epoch < 10:
# # # # #         return huber_clamp(x)
# # # # #     elif epoch < 20:
# # # # #         beta = (epoch - 10) / 10.0
# # # # #         return (1.0 - beta) * huber_clamp(x) + beta * tanh_clamp(x)
# # # # #     else:
# # # # #         return tanh_clamp(x)


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Haversine
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def _haversine(p1: torch.Tensor, p2: torch.Tensor,
# # # # #                unit_01deg: bool = True) -> torch.Tensor:
# # # # #     """
# # # # #     Tính khoảng cách Haversine (km).
# # # # #     unit_01deg=True: input là normalized coords, decode trước.
# # # # #     unit_01deg=False: input đã là degrees.
# # # # #     """
# # # # #     if unit_01deg:
# # # # #         lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
# # # # #         lat1 = (p1[..., 1] * 50.0) / 10.0
# # # # #         lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
# # # # #         lat2 = (p2[..., 1] * 50.0) / 10.0
# # # # #     else:
# # # # #         lon1, lat1 = p1[..., 0], p1[..., 1]
# # # # #         lon2, lat2 = p2[..., 0], p2[..., 1]

# # # # #     lat1r = torch.deg2rad(lat1); lat2r = torch.deg2rad(lat2)
# # # # #     dlon  = torch.deg2rad(lon2 - lon1)
# # # # #     dlat  = torch.deg2rad(lat2 - lat1)
# # # # #     a = (torch.sin(dlat / 2).pow(2)
# # # # #          + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2).pow(2))
# # # # #     a = a.clamp(1e-12, 1.0 - 1e-12)   # FIX-L-K: stable asin
# # # # #     return 2.0 * 6371.0 * torch.asin(a.sqrt())


# # # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # #     return _haversine(p1, p2, unit_01deg=False)


# # # # # def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
# # # # #     out = arr.clone()
# # # # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# # # # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# # # # #     return out


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  FIX-L-F: Spatial Boundary Weighting (Eq.63a-b)
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def _boundary_weights(traj_deg: torch.Tensor) -> torch.Tensor:
# # # # #     """
# # # # #     w_bnd,k = σ(d_bnd,k - 0.5) ∈ [0,1]
# # # # #     d_bnd,k = min(lat-lat_min, lat_max-lat, lon-lon_min, lon_max-lon) / 5°

# # # # #     traj_deg: [T, B, 2]  (lon, lat)
# # # # #     Returns:  [T, B]
# # # # #     """
# # # # #     lon = traj_deg[..., 0]  # [T, B]
# # # # #     lat = traj_deg[..., 1]  # [T, B]

# # # # #     d_lat_lo = (lat - _ERA5_LAT_MIN) / 5.0
# # # # #     d_lat_hi = (_ERA5_LAT_MAX - lat) / 5.0
# # # # #     d_lon_lo = (lon - _ERA5_LON_MIN) / 5.0
# # # # #     d_lon_hi = (_ERA5_LON_MAX - lon) / 5.0

# # # # #     d_bnd = torch.stack([d_lat_lo, d_lat_hi, d_lon_lo, d_lon_hi], dim=-1).min(dim=-1).values
# # # # #     return torch.sigmoid(d_bnd - 0.5)   # [T, B]


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Short-range Huber Loss (FIX-L49, giữ nguyên)
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def short_range_regression_loss(
# # # # #     pred_sr:  torch.Tensor,   # [4, B, 2]  normalised
# # # # #     gt_sr:    torch.Tensor,   # [4, B, 2]  normalised
# # # # #     last_pos: torch.Tensor,   # [B, 2]     (unused, kept for API compat)
# # # # # ) -> torch.Tensor:
# # # # #     """
# # # # #     Huber loss trên haversine distance cho steps 1-4.
# # # # #     Return: scalar loss, đơn vị km (không chia HUBER_DELTA).
# # # # #     Caller chịu trách nhiệm scale bằng weight.
# # # # #     """
# # # # #     n_steps = min(pred_sr.shape[0], gt_sr.shape[0], _SR_N_STEPS)
# # # # #     if n_steps == 0:
# # # # #         return pred_sr.new_zeros(())

# # # # #     pred_deg = _norm_to_deg(pred_sr[:n_steps])   # [n_steps, B, 2]
# # # # #     gt_deg   = _norm_to_deg(gt_sr[:n_steps])

# # # # #     dist_km = _haversine_deg(pred_deg, gt_deg)   # [n_steps, B]

# # # # #     huber = torch.where(
# # # # #         dist_km < _HUBER_DELTA,
# # # # #         0.5 * dist_km.pow(2) / _HUBER_DELTA,
# # # # #         dist_km - 0.5 * _HUBER_DELTA,
# # # # #     )  # [n_steps, B]

# # # # #     w = pred_sr.new_tensor(_SR_WEIGHTS[:n_steps])  # [n_steps]
# # # # #     # Normalize weights để tổng = 1
# # # # #     w = w / w.sum()
# # # # #     return (huber * w.view(-1, 1)).mean()


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  FIX-L-H: Bridge Loss SR↔FM (Eq.80)
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def bridge_loss(
# # # # #     sr_pred:  torch.Tensor,   # [4, B, 2]  normalised SR predictions
# # # # #     fm_mean:  torch.Tensor,   # [T, B, 2]  normalised FM mean trajectory
# # # # # ) -> torch.Tensor:
# # # # #     """
# # # # #     Nhất quán vị trí và vận tốc tại bước nối step 4 (idx=3).

# # # # #     L_bridge = ||y4_SR - X4_FM||² / (100km)²
# # # # #              + 0.5 * ||v4_SR - v5_FM||² / STEP²
# # # # #     """
# # # # #     if sr_pred.shape[0] < 4 or fm_mean.shape[0] < 5:
# # # # #         return sr_pred.new_zeros(())

# # # # #     # Vị trí step 4 (index 3)
# # # # #     pos_sr4 = _norm_to_deg(sr_pred[3])       # [B, 2]
# # # # #     pos_fm4 = _norm_to_deg(fm_mean[3])       # [B, 2]

# # # # #     dist_pos = _haversine_deg(pos_sr4, pos_fm4)  # [B]
# # # # #     l_pos = (dist_pos / 100.0).pow(2).mean()

# # # # #     # Vận tốc tại tiếp giáp
# # # # #     # v4_SR = pos4_SR - pos3_SR (degrees, thô)
# # # # #     v4_sr = _norm_to_deg(sr_pred[3]) - _norm_to_deg(sr_pred[2])   # [B, 2]
# # # # #     # v5_FM = pos5_FM - pos4_FM
# # # # #     v5_fm = _norm_to_deg(fm_mean[4]) - _norm_to_deg(fm_mean[3])   # [B, 2]

# # # # #     # Chuyển sang km
# # # # #     lat_mid = (_norm_to_deg(sr_pred[3])[:, 1] + _norm_to_deg(fm_mean[4])[:, 1]) / 2.0
# # # # #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)

# # # # #     def _to_km(dv):
# # # # #         km = dv.clone()
# # # # #         km[:, 0] = dv[:, 0] * cos_lat * DEG_TO_KM
# # # # #         km[:, 1] = dv[:, 1] * DEG_TO_KM
# # # # #         return km

# # # # #     v4_sr_km = _to_km(v4_sr)
# # # # #     v5_fm_km = _to_km(v5_fm)
# # # # #     l_vel = ((v4_sr_km - v5_fm_km).pow(2).sum(-1) / STEP_KM**2).mean()

# # # # #     return l_pos + 0.5 * l_vel


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Step displacements
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
# # # # #     dt      = traj_deg[1:] - traj_deg[:-1]
# # # # #     lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
# # # # #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
# # # # #     dt_km   = dt.clone()
# # # # #     dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
# # # # #     dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
# # # # #     return dt_km


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Recurvature helpers
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def _total_rotation_angle_batch(gt: torch.Tensor) -> torch.Tensor:
# # # # #     T, B, _ = gt.shape
# # # # #     if T < 3:
# # # # #         return gt.new_zeros(B)
# # # # #     lats_rad = torch.deg2rad(gt[:, :, 1])
# # # # #     cos_lat  = torch.cos(lats_rad[:-1])
# # # # #     dlat = gt[1:, :, 1] - gt[:-1, :, 1]
# # # # #     dlon = (gt[1:, :, 0] - gt[:-1, :, 0]) * cos_lat
# # # # #     v    = torch.stack([dlon, dlat], dim=-1)
# # # # #     v1   = v[:-1]; v2 = v[1:]
# # # # #     n1   = v1.norm(dim=-1).clamp(min=1e-8)
# # # # #     n2   = v2.norm(dim=-1).clamp(min=1e-8)
# # # # #     cos_a = (v1 * v2).sum(-1) / (n1 * n2)
# # # # #     return torch.rad2deg(torch.acos(cos_a.clamp(-1.0, 1.0))).sum(0)


# # # # # def _recurvature_weights(gt: torch.Tensor,
# # # # #                          thr: float = RECURV_ANGLE_THR,
# # # # #                          w_recurv: float = RECURV_WEIGHT) -> torch.Tensor:
# # # # #     rot = _total_rotation_angle_batch(gt)
# # # # #     return torch.where(rot >= thr,
# # # # #                        torch.full_like(rot, w_recurv),
# # # # #                        torch.ones_like(rot))


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Directional losses
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def velocity_loss_per_sample(pred: torch.Tensor,
# # # # #                              gt: torch.Tensor) -> torch.Tensor:
# # # # #     if pred.shape[0] < 2:
# # # # #         return pred.new_zeros(pred.shape[1])
# # # # #     v_pred_km = _step_displacements_km(pred)
# # # # #     v_gt_km   = _step_displacements_km(gt)
# # # # #     s_pred    = v_pred_km.norm(dim=-1)
# # # # #     s_gt      = v_gt_km.norm(dim=-1)
# # # # #     l_speed   = (s_pred - s_gt).pow(2).mean(0)
# # # # #     gn        = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # #     gt_unit   = v_gt_km / gn
# # # # #     ate       = ((v_pred_km - v_gt_km) * gt_unit).sum(-1)
# # # # #     l_ate     = ate.pow(2).mean(0)
# # # # #     pn        = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # #     cos_sim   = ((v_pred_km / pn) * gt_unit).sum(-1)
# # # # #     l_dir     = F.relu(-cos_sim).pow(2).mean(0) * STEP_KM ** 2
# # # # #     return (l_speed + 0.5 * l_ate + 0.3 * l_dir) / (STEP_KM ** 2)


# # # # # def disp_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # # #     if pred.shape[0] < 2:
# # # # #         return pred.new_zeros(pred.shape[1])
# # # # #     pd = _step_displacements_km(pred).norm(dim=-1).mean(0)
# # # # #     gd = _step_displacements_km(gt).norm(dim=-1).mean(0)
# # # # #     return (pd - gd).pow(2) / (STEP_KM ** 2)


# # # # # def step_dir_loss_per_sample(pred: torch.Tensor,
# # # # #                               gt: torch.Tensor) -> torch.Tensor:
# # # # #     if pred.shape[0] < 2:
# # # # #         return pred.new_zeros(pred.shape[1])
# # # # #     v_pred_km = _step_displacements_km(pred)
# # # # #     v_gt_km   = _step_displacements_km(gt)
# # # # #     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # #     gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # #     cos_sim = ((v_pred_km / pn) * (v_gt_km / gn)).sum(-1)
# # # # #     return (1.0 - cos_sim).mean(0)


# # # # # def heading_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # # #     if pred.shape[0] < 2:
# # # # #         return pred.new_zeros(())
# # # # #     pv = _step_displacements_km(pred)
# # # # #     gv = _step_displacements_km(gt)
# # # # #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # #     wrong_dir      = F.relu(-((pv / pn) * (gv / gn)).sum(-1))
# # # # #     wrong_dir_loss = (wrong_dir ** 2).mean()

# # # # #     if pred.shape[0] >= 3:
# # # # #         def _curv(v):
# # # # #             cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
# # # # #             n1    = v[1:].norm(dim=-1).clamp(min=1e-4)
# # # # #             n2    = v[:-1].norm(dim=-1).clamp(min=1e-4)
# # # # #             return (cross / (n1 * n2)).clamp(-10.0, 10.0)
# # # # #         curv_mse = F.mse_loss(_curv(pv), _curv(gv))
# # # # #     else:
# # # # #         curv_mse = pred.new_zeros(())
# # # # #     return wrong_dir_loss + curv_mse


# # # # # def smooth_loss(pred: torch.Tensor) -> torch.Tensor:
# # # # #     if pred.shape[0] < 3:
# # # # #         return pred.new_zeros(())
# # # # #     v_km = _step_displacements_km(pred)
# # # # #     if v_km.shape[0] < 2:
# # # # #         return pred.new_zeros(())
# # # # #     accel_km = v_km[1:] - v_km[:-1]
# # # # #     return accel_km.pow(2).mean() / (STEP_KM ** 2)


# # # # # def acceleration_loss(pred: torch.Tensor) -> torch.Tensor:
# # # # #     return smooth_loss(pred)


# # # # # def jerk_loss(pred: torch.Tensor) -> torch.Tensor:
# # # # #     if pred.shape[0] < 4:
# # # # #         return pred.new_zeros(())
# # # # #     v_km = _step_displacements_km(pred)
# # # # #     if v_km.shape[0] < 3:
# # # # #         return pred.new_zeros(())
# # # # #     a_km = v_km[1:] - v_km[:-1]
# # # # #     j_km = a_km[1:] - a_km[:-1]
# # # # #     return j_km.pow(2).mean() / (STEP_KM ** 2)


# # # # # def overall_dir_loss(pred: torch.Tensor, gt: torch.Tensor,
# # # # #                      ref: torch.Tensor) -> torch.Tensor:
# # # # #     p_d = pred[-1] - ref
# # # # #     g_d = gt[-1]   - ref
# # # # #     lat_ref = ref[:, 1]
# # # # #     cos_lat = torch.cos(torch.deg2rad(lat_ref)).clamp(min=1e-4)
# # # # #     p_d_km  = p_d.clone(); p_d_km[:, 0] *= cos_lat * DEG_TO_KM; p_d_km[:, 1] *= DEG_TO_KM
# # # # #     g_d_km  = g_d.clone(); g_d_km[:, 0] *= cos_lat * DEG_TO_KM; g_d_km[:, 1] *= DEG_TO_KM
# # # # #     pn = p_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # #     gn = g_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # # #     return (1.0 - ((p_d_km / pn) * (g_d_km / gn)).sum(-1)).mean()


# # # # # def recurvature_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # # #     if pred.shape[0] < 3:
# # # # #         return pred.new_zeros(())
# # # # #     pred_v   = pred[1:] - pred[:-1]
# # # # #     gt_v     = gt[1:]   - gt[:-1]
# # # # #     gt_cross = (gt_v[:-1, :, 0] * gt_v[1:, :, 1]
# # # # #               - gt_v[:-1, :, 1] * gt_v[1:, :, 0])
# # # # #     if gt_cross.shape[0] < 2:
# # # # #         return pred.new_zeros(())
# # # # #     sign_change = (gt_cross[:-1] * gt_cross[1:]) < 0
# # # # #     if not sign_change.any():
# # # # #         return pred.new_zeros(())
# # # # #     pred_v_mid = pred_v[1:-1]
# # # # #     gt_v_mid   = gt_v[1:-1]
# # # # #     if pred_v_mid.shape[0] > sign_change.shape[0]:
# # # # #         pred_v_mid = pred_v_mid[:sign_change.shape[0]]
# # # # #         gt_v_mid   = gt_v_mid[:sign_change.shape[0]]
# # # # #     pn      = pred_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # #     gn      = gt_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # #     cos_sim  = ((pred_v_mid / pn) * (gt_v_mid / gn)).sum(-1)
# # # # #     dir_loss = (1.0 - cos_sim)
# # # # #     mask     = sign_change.float()
# # # # #     if mask.sum() < 1:
# # # # #         return pred.new_zeros(())
# # # # #     return (dir_loss * mask).sum() / mask.sum().clamp(min=1)


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  FIX-L-G: AFCRPS với Energy Score (Eq.76-77)
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def fm_afcrps_loss(
# # # # #     pred_samples:      torch.Tensor,         # [M, T, B, 2]
# # # # #     gt:                torch.Tensor,         # [T, B, 2]
# # # # #     unit_01deg:        bool  = True,
# # # # #     intensity_w:       Optional[torch.Tensor] = None,
# # # # #     step_weight_alpha: float = 0.0,
# # # # #     w_es:              float = 0.3,          # weight cho Energy Score term
# # # # # ) -> torch.Tensor:
# # # # #     """
# # # # #     Almost-Fair CRPS + M-Normalized Energy Score (Eq.76-77).

# # # # #     L_FM = accuracy - sharpness_penalty + w_ES * ES_norm(M)

# # # # #     FIX-L-G: Thêm ES_norm với unbiasing factor (M-1)/M.
# # # # #     FIX-L-J: Bỏ clamp(min=eps) để gradient flow tự nhiên.
# # # # #     """
# # # # #     M, T, B, _ = pred_samples.shape

# # # # #     # Time weights
# # # # #     # base_w = torch.linspace(0.5, 1.5, T, device=pred_samples.device)
# # # # #     # losses.py — fm_afcrps_loss, thay base_w
# # # # #     base_w = torch.zeros(T, device=pred_samples.device)
# # # # #     for i in range(T):
# # # # #         if i >= 8:    base_w[i] = 3.0   # 54h-72h
# # # # #         elif i >= 4:  base_w[i] = 1.5   # 30h-48h
# # # # #         else:         base_w[i] = 0.5   # 6h-24h (SR lo)
# # # # #     if step_weight_alpha > 0.0:
# # # # #         early_w = torch.exp(
# # # # #             -torch.arange(T, dtype=torch.float, device=pred_samples.device) * 0.5
# # # # #         )
# # # # #         early_w = early_w / early_w.mean()
# # # # #         time_w = (1.0 - step_weight_alpha) * base_w + step_weight_alpha * early_w
# # # # #     else:
# # # # #         time_w = base_w

# # # # #     if M == 1:
# # # # #         dist = _haversine(pred_samples[0], gt, unit_01deg)   # [T, B]
# # # # #         loss_per_b = (dist * time_w.unsqueeze(1)).mean(0)    # [B]
# # # # #         es_term = pred_samples.new_zeros(())
# # # # #     else:
# # # # #         # Accuracy: E[d(X^m, Y)]
# # # # #         d_to_gt = _haversine(
# # # # #             pred_samples,
# # # # #             gt.unsqueeze(0).expand_as(pred_samples),
# # # # #             unit_01deg,
# # # # #         )   # [M, T, B]
# # # # #         d_to_gt_w    = d_to_gt * time_w.unsqueeze(0).unsqueeze(2)
# # # # #         e_sy         = d_to_gt_w.mean(1).mean(0)   # [B]  accuracy

# # # # #         # Sharpness: E[d(X^m, X^m')]  with m≠m'  (fair: exclude m=m')
# # # # #         ps_i = pred_samples.unsqueeze(1)   # [M, 1, T, B, 2]
# # # # #         ps_j = pred_samples.unsqueeze(0)   # [1, M, T, B, 2]
# # # # #         d_pair = _haversine(
# # # # #             ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # # # #             ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # # # #             unit_01deg,
# # # # #         ).reshape(M, M, T, B)   # [M, M, T, B]

# # # # #         # Mask diagonal (m == m')
# # # # #         diag_mask = torch.eye(M, device=pred_samples.device, dtype=torch.bool)
# # # # #         diag_mask = diag_mask.view(M, M, 1, 1).expand_as(d_pair)
# # # # #         d_pair = d_pair.masked_fill(diag_mask, 0.0)

# # # # #         d_pair_w    = d_pair * time_w.unsqueeze(0).unsqueeze(0).unsqueeze(3)
# # # # #         # fair mean: sum / (M*(M-1)) thay vì M*M
# # # # #         e_ssp       = d_pair_w.mean(2).sum(0).sum(0) / (M * (M - 1))  # [B]

# # # # #         # Almost-fair CRPS
# # # # #         loss_per_b = e_sy - 0.5 * e_ssp    # [B]  FIX-L-J: bỏ clamp

# # # # #         # FIX-L-G: Energy Score term (Eq.77)
# # # # #         # ES_norm(M) = ||mean_m(X^m) - Y||_F - (M-1)/M * (1/M) * sum_{m≠m'} ||X^m - X^m'||_F
# # # # #         if w_es > 0.0 and M > 1:
# # # # #             # Chuyển sang [M, T*B, 2] để tính norm Frobenius
# # # # #             ps_flat  = pred_samples.reshape(M, T * B, 2)  # [M, T*B, 2]
# # # # #             gt_flat  = gt.reshape(T * B, 2)               # [T*B, 2]
# # # # #             mean_pred = ps_flat.mean(0)                    # [T*B, 2]

# # # # #             # ||mean - Y||_F  (Frobenius = sqrt(sum of squares))
# # # # #             es_acc = (mean_pred - gt_flat).pow(2).sum(-1).sqrt().mean()

# # # # #             # (M-1)/M * mean_{m≠m'} ||X^m - X^m'||_F
# # # # #             ps_i_f = ps_flat.unsqueeze(1)   # [M, 1, T*B, 2]
# # # # #             ps_j_f = ps_flat.unsqueeze(0)   # [1, M, T*B, 2]
# # # # #             d_pair_f = (ps_i_f - ps_j_f).pow(2).sum(-1).sqrt()  # [M, M, T*B]
# # # # #             # Mask diagonal
# # # # #             diag_f = torch.eye(M, device=pred_samples.device, dtype=torch.bool)
# # # # #             diag_f = diag_f.view(M, M, 1).expand_as(d_pair_f)
# # # # #             d_pair_f = d_pair_f.masked_fill(diag_f, 0.0)
# # # # #             es_sharp = (M - 1) / M * d_pair_f.sum(0).sum(0).mean() / (M * (M - 1))

# # # # #             es_term = es_acc - es_sharp
# # # # #         else:
# # # # #             es_term = pred_samples.new_zeros(())

# # # # #     if intensity_w is not None:
# # # # #         w = intensity_w.to(loss_per_b.device)
# # # # #         crps_loss = (loss_per_b * w).mean()
# # # # #     else:
# # # # #         crps_loss = loss_per_b.mean()

# # # # #     return crps_loss + w_es * es_term


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  PINN Components
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def _get_uv500_ms(env_data: dict, key_mean: str, key_center: str,
# # # # #                   T_tgt: int, B: int, device: torch.device) -> torch.Tensor:
# # # # #     def _extract(key):
# # # # #         x = env_data.get(key, None)
# # # # #         if x is None or not torch.is_tensor(x):
# # # # #             return None
# # # # #         x = x.to(device).float()
# # # # #         if x.dim() == 3:
# # # # #             x = x[..., 0]
# # # # #         elif x.dim() == 1:
# # # # #             x = x.unsqueeze(0).expand(B, -1)
# # # # #         x = x.permute(1, 0)   # [T_obs, B]
# # # # #         T_obs = x.shape[0]
# # # # #         if T_obs >= T_tgt:
# # # # #             return x[:T_tgt] * _UV500_NORM
# # # # #         pad = torch.zeros(T_tgt - T_obs, B, device=device)
# # # # #         return torch.cat([x * _UV500_NORM, pad], dim=0)

# # # # #     val = _extract(key_center)
# # # # #     if val is not None:
# # # # #         return val
# # # # #     val = _extract(key_mean)
# # # # #     if val is not None:
# # # # #         return val
# # # # #     return torch.zeros(T_tgt, B, device=device)


# # # # # def _get_gph500_norm(env_data: dict, key: str,
# # # # #                      T_tgt: int, B: int, device: torch.device) -> torch.Tensor:
# # # # #     x = env_data.get(key, None)
    
# # # # #     if x is None or not torch.is_tensor(x):
# # # # #         return torch.zeros(T_tgt, B, device=device)
# # # # #     x = x.to(device).float()
# # # # #     if x.dim() == 3:
# # # # #         x = x[..., 0]
# # # # #     x = x.permute(1, 0)
# # # # #     T_obs = x.shape[0]
# # # # #     if T_obs >= T_tgt:
# # # # #         return x[:T_tgt]
# # # # #     pad = torch.zeros(T_tgt - T_obs, B, device=device)
# # # # #     return torch.cat([x, pad], dim=0)


# # # # # def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
# # # # #     """BVE: bảo toàn độ xoáy tuyệt đối (Eq.55)."""
# # # # #     T, B, _ = pred_abs_deg.shape
# # # # #     if T < 3:
# # # # #         return pred_abs_deg.new_zeros(())

# # # # #     DT      = DT_6H
# # # # #     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
# # # # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # # #     cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

# # # # #     u = dlat[:, :, 0] * cos_lat * 111000.0 / DT   # m/s
# # # # #     v = dlat[:, :, 1] * 111000.0 / DT

# # # # #     if u.shape[0] < 2:
# # # # #         return pred_abs_deg.new_zeros(())

# # # # #     du = (u[1:] - u[:-1]) / DT
# # # # #     dv = (v[1:] - v[:-1]) / DT

# # # # #     f    = 2 * OMEGA * torch.sin(lat_rad[1:-1])
# # # # #     beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH

# # # # #     res_u = du - f * v[1:]
# # # # #     res_v = dv + f * u[1:]

# # # # #     R_tc            = 3e5
# # # # #     v_beta_x        = -beta * R_tc ** 2 / 2
# # # # #     res_u_corrected = res_u - v_beta_x

# # # # #     # Scale = 0.1 m/s²: typical TC → residual ~1e-3 → loss ~1e-4 (không penalize)
# # # # #     # Bad traj → residual ~0.1 → loss ~1.0 (penalize mạnh)
# # # # #     scale = 0.1
# # # # #     # loss = ((res_u_corrected / scale).pow(2).mean()
# # # # #     #       + (res_v / scale).pow(2).mean())
# # # # #     loss = ((res_u_corrected / scale).pow(2) + (res_v / scale).pow(2)) 
# # # # #     return loss


# # # # # def pinn_rankine_steering(pred_abs_deg: torch.Tensor,
# # # # #                           env_data: Optional[dict]) -> torch.Tensor:
# # # # #     """Steering flow alignment (Eq.59)."""
# # # # #     if env_data is None:
# # # # #         return pred_abs_deg.new_zeros(())

# # # # #     T, B, _ = pred_abs_deg.shape
# # # # #     if T < 2:
# # # # #         return pred_abs_deg.new_zeros(())

# # # # #     device  = pred_abs_deg.device
# # # # #     T_tgt   = T - 1
# # # # #     DT      = DT_6H

# # # # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # # #     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
# # # # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

# # # # #     u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT   # [T-1, B] m/s
# # # # #     v_tc = dlat[:, :, 1] * 111000.0 / DT

# # # # #     u500 = _get_uv500_ms(env_data, "u500_mean", "u500_center", T_tgt, B, device)
# # # # #     v500 = _get_uv500_ms(env_data, "v500_mean", "v500_center", T_tgt, B, device)

# # # # #     uv_mag       = torch.sqrt(u500**2 + v500**2)
# # # # #     has_steering = (uv_mag > _STEERING_MIN_MS).float()

# # # # #     env_dir  = torch.stack([u500, v500], dim=-1)
# # # # #     tc_dir   = torch.stack([u_tc, v_tc], dim=-1)
# # # # #     env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=1.0)
# # # # #     tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
# # # # #     cos_sim  = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)

# # # # #     # Sigmoid soft weighting (Eq.59): penalize liên tục thay vì ngưỡng cứng
# # # # #     steer_w  = torch.sigmoid(uv_mag - 1.0)   # ≈0.27 tại 0 m/s, ≈0.5 tại 1 m/s
# # # # #     misalign = F.relu(-0.5 - cos_sim).pow(2)
# # # # #     # return (misalign * steer_w * has_steering).mean() * 0.05
# # # # #     return (misalign * steer_w * has_steering) * 0.05 # Trả về [T-1, B]

# # # # # def pinn_gph500_gradient(pred_abs_deg: torch.Tensor,
# # # # #                          env_data: Optional[dict]) -> torch.Tensor:
# # # # #     """
# # # # #     FIX-L-I: GPH gradient đúng hơn (Eq.60).

# # # # #     Thay vì dùng center-mean làm proxy gradient (sai về vật lý),
# # # # #     dùng temporal gradient: ΔGPH/Δt để ước lượng xu hướng.
# # # # #     Penalize khi TC di chuyển ngược xu hướng GPH.
# # # # #     """
# # # # #     if env_data is None:
# # # # #         return pred_abs_deg.new_zeros(())
# # # # #     T, B, _ = pred_abs_deg.shape
# # # # #     if T < 2:
# # # # #         return pred_abs_deg.new_zeros(())

# # # # #     device = pred_abs_deg.device
# # # # #     T_tgt  = T - 1

# # # # #     gph_mean   = _get_gph500_norm(env_data, "gph500_mean",   T_tgt, B, device)
# # # # #     gph_center = _get_gph500_norm(env_data, "gph500_center", T_tgt, B, device)

# # # # #     # Proxy gradient: difference between center (tại TC) và mean (domain)
# # # # #     # Đơn vị: normalized z-score
# # # # #     # Dương: TC ở vùng GPH cao hơn xung quanh → ridge → đẩy poleward/westward
# # # # #     # Âm:   TC ở vùng GPH thấp hơn → trough → TC có xu hướng recurve
# # # # #     gph_diff = gph_center - gph_mean   # [T-1, B]

# # # # #     # Lat tendency của TC
# # # # #     dlat = pred_abs_deg[1:, :, 1] - pred_abs_deg[:-1, :, 1]  # [T-1, B]

# # # # #     # Heuristic: khi GPH diff dương mạnh → ridge north → TC nên northward
# # # # #     # expected_dlat_sign = sign(gph_diff)
# # # # #     # Chỉ penalize khi gradient rõ ràng (|gph_diff| > 0.1 sigma)
# # # # #     has_gradient = (gph_diff.abs() > 0.1).float()

# # # # #     # s_correct dương khi TC di chuyển đúng hướng
# # # # #     s_correct = torch.sign(dlat) * torch.sign(gph_diff)
# # # # #     wrong_dir = F.relu(-s_correct)   # dương khi sai hướng

# # # # #     # return (wrong_dir.pow(2) * has_gradient).mean() * 0.02
# # # # #     return (wrong_dir.pow(2) * has_gradient) * 0.02 # Trả về [T-1, B]

# # # # # def pinn_steering_speed_consistency(pred_abs_deg: torch.Tensor,
# # # # #                                     env_data: Optional[dict]) -> torch.Tensor:
# # # # #     """TC speed vs steering flow speed (Eq.61)."""
# # # # #     if env_data is None:
# # # # #         return pred_abs_deg.new_zeros(())
# # # # #     T, B, _ = pred_abs_deg.shape
# # # # #     if T < 2:
# # # # #         return pred_abs_deg.new_zeros(())

# # # # #     device = pred_abs_deg.device
# # # # #     T_tgt  = T - 1
# # # # #     DT     = DT_6H

# # # # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # # #     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
# # # # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # #     dx_km   = dlat[:, :, 0] * cos_lat * 111.0
# # # # #     dy_km   = dlat[:, :, 1] * 111.0
# # # # #     tc_speed_ms = torch.sqrt(dx_km**2 + dy_km**2) * 1000.0 / DT  # m/s

# # # # #     u_c = _get_uv500_ms(env_data, "u500_mean", "u500_center", T_tgt, B, device)
# # # # #     v_c = _get_uv500_ms(env_data, "v500_mean", "v500_center", T_tgt, B, device)
# # # # #     u_m = _get_uv500_ms(env_data, "u500_mean", "u500_mean",   T_tgt, B, device)
# # # # #     v_m = _get_uv500_ms(env_data, "v500_mean", "v500_mean",   T_tgt, B, device)

# # # # #     steering_mag = (torch.sqrt(u_c**2 + v_c**2) +
# # # # #                     torch.sqrt(u_m**2 + v_m**2)) / 2.0

# # # # #     # Normalize by σ²(UV500) như Eq.61
# # # # #     steer_var = (steering_mag**2).mean().clamp(min=1.0)

# # # # #     has_steering = (steering_mag > _STEERING_MIN_MS).float()
# # # # #     lo = steering_mag * 0.3
# # # # #     hi = steering_mag * 1.5
# # # # #     too_slow = F.relu(lo - tc_speed_ms)
# # # # #     too_fast = F.relu(tc_speed_ms - hi)

# # # # #     penalty = (too_slow.pow(2) + too_fast.pow(2)) / steer_var
# # # # #     # return (penalty * has_steering).mean() * 0.03
# # # # #     return (penalty * has_steering) * 0.03 # Trả về [T-1, B]

# # # # # def pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # #     if pred_deg.shape[0] < 2:
# # # # #         return pred_deg.new_zeros(())
# # # # #     dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # #     lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # #     dx_km   = dt_deg[:, :, 0] * cos_lat * DEG_TO_KM
# # # # #     dy_km   = dt_deg[:, :, 1] * DEG_TO_KM
# # # # #     speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # # # #     # return F.relu(speed - 600.0).pow(2).mean()
# # # # #     return F.relu(speed - 600.0).pow(2) # Trả về [T-1, B]

# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  FIX-L-D: Pressure-Wind Balance Loss (Eq.62-63)
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def pinn_pressure_wind_loss(
# # # # #     pred_abs_deg: torch.Tensor,   # [T, B, 2]  lon/lat
# # # # #     vmax_pred:    Optional[torch.Tensor],  # [T, B]  m/s
# # # # #     pmin_pred:    Optional[torch.Tensor],  # [T, B]  hPa
# # # # #     r34_km:       Optional[torch.Tensor] = None,  # [T, B]  km
# # # # #     epoch:        int = 0,
# # # # # ) -> torch.Tensor:
# # # # #     """
# # # # #     Gradient wind balance (Eq.62-63):
# # # # #       p_env - p_min ≈ ρ*V²/2 + f*R_TC*V/2

# # # # #     Dynamic R_TC: dùng 2*R34 nếu có, fallback climatology (Eq.62a-b).
# # # # #     Kích hoạt từ epoch 30.
# # # # #     """
# # # # #     if epoch < 30:
# # # # #         return pred_abs_deg.new_zeros(())
# # # # #     if vmax_pred is None or pmin_pred is None:
# # # # #         return pred_abs_deg.new_zeros(())

# # # # #     T, B, _ = pred_abs_deg.shape
# # # # #     if vmax_pred.shape[0] != T or pmin_pred.shape[0] != T:
# # # # #         return pred_abs_deg.new_zeros(())

# # # # #     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
# # # # #     f_k     = 2 * OMEGA * torch.sin(lat_rad).abs().clamp(min=1e-6)  # [T, B]

# # # # #     # Dynamic R_TC (Eq.62a-b)
# # # # #     if r34_km is not None:
# # # # #         R_tc = (2.0 * r34_km * 1000.0).clamp(min=1e5, max=8e5)  # m
# # # # #     else:
# # # # #         # Fallback: R_TC = 3e5 + 1e3 * max(0, Vmax - 30) m (Eq.62b)
# # # # #         R_tc = 3e5 + 1e3 * F.relu(vmax_pred - 30.0)  # [T, B]

# # # # #     V  = vmax_pred.clamp(min=1.0)    # m/s
# # # # #     dp = (P_ENV - pmin_pred) * 100.0  # Pa (1 hPa = 100 Pa)

# # # # #     # Gradient wind: dp = ρV²/2 + ρ*f*R*V/2
# # # # #     dp_pred = RHO_AIR * (V.pow(2) / 2.0 + f_k * R_tc * V / 2.0)  # Pa

# # # # #     # Normalize bằng 5 hPa (500 Pa) như Eq.63
# # # # #     residual = (dp - dp_pred) / 500.0
# # # # #     # return residual.pow(2).mean()
# # # # #     return residual.pow(2) # Trả về [T, B]

# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  FIX-L-C + FIX-L-E + FIX-L-F: PINN tổng hợp
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def pinn_bve_loss(
# # # # # #     pred_abs_deg: torch.Tensor,
# # # # # #     batch_list,
# # # # # #     env_data:    Optional[dict] = None,
# # # # # #     epoch:       int = 0,
# # # # # #     gt_abs_deg:  Optional[torch.Tensor] = None,  # FIX-L-C: cho adaptive weighting
# # # # # #     vmax_pred:   Optional[torch.Tensor] = None,   # FIX-L-D: pressure-wind
# # # # # #     pmin_pred:   Optional[torch.Tensor] = None,
# # # # # #     r34_km:      Optional[torch.Tensor] = None,
# # # # # # ) -> torch.Tensor:
# # # # # #     """
# # # # # #     PINN tổng hợp (Eq.64/101) với đầy đủ 5 ràng buộc + PWR.

# # # # # #     Cải tiến so với v25:
# # # # # #       - AdaptClamp thay vì tanh cố định (FIX-L-B)
# # # # # #       - Adaptive BVE weighting theo track error (FIX-L-C)
# # # # # #       - Frequency compensation f_lazy (FIX-L-E)
# # # # # #       - Spatial boundary weighting w_bnd (FIX-L-F)
# # # # # #       - L_PWR pressure-wind balance (FIX-L-D)
# # # # # #     """
# # # # # #     T = pred_abs_deg.shape[0]
# # # # # #     if T < 3:
# # # # # #         return pred_abs_deg.new_zeros(())

# # # # # #     _env = env_data
# # # # # #     if _env is None and batch_list is not None:
# # # # # #         try:
# # # # # #             _env = batch_list[13]
# # # # # #         except (IndexError, TypeError):
# # # # # #             _env = None

# # # # # #     # FIX-L-E: Frequency compensation (Eq.100)
# # # # # #     if epoch < 30:
# # # # # #         f_lazy = 0.20
# # # # # #     elif epoch < 50:
# # # # # #         f_lazy = 0.50
# # # # # #     else:
# # # # # #         f_lazy = 1.00

# # # # # #     # FIX-L-C: Adaptive BVE weighting (Eq.99)
# # # # # #     if gt_abs_deg is not None and gt_abs_deg.shape == pred_abs_deg.shape:
# # # # # #         with torch.no_grad():
# # # # # #             d_track = _haversine_deg(pred_abs_deg, gt_abs_deg)   # [T, B]
# # # # # #             # w_BVE,k = σ(1 - d/200km)
# # # # # #             w_bve_per_step = torch.sigmoid(1.0 - d_track / 200.0)  # [T, B] ∈ (0,1)
# # # # # #             w_bve = w_bve_per_step.mean()   # scalar
# # # # # #     else:
# # # # # #         w_bve = pred_abs_deg.new_tensor(1.0)

# # # # # #     # FIX-L-F: Spatial boundary weighting (Eq.63a-b)
# # # # # #     w_bnd = _boundary_weights(pred_abs_deg).mean()   # scalar ≈ 1 at center, ≈ 0 at edge

# # # # # #     # # Individual PINN components
# # # # # #     # l_sw      = pinn_shallow_water(pred_abs_deg)
# # # # # #     # l_steer   = pinn_rankine_steering(pred_abs_deg, _env)
# # # # # #     # l_speed   = pinn_speed_constraint(pred_abs_deg)
# # # # # #     # l_gph     = pinn_gph500_gradient(pred_abs_deg, _env)
# # # # # #     # l_spdcons = pinn_steering_speed_consistency(pred_abs_deg, _env)
# # # # # #     # l_pwr     = pinn_pressure_wind_loss(pred_abs_deg, vmax_pred, pmin_pred, r34_km, epoch)

# # # # # #     # # Tổng hợp (Eq.101) — hệ số từ doc
# # # # # #     # total = (
# # # # # #     #     w_bve * l_sw          # BVE với adaptive weighting
# # # # # #     #     + 0.5  * l_steer
# # # # # #     #     + 0.1  * l_speed
# # # # # #     #     + 0.3  * l_gph
# # # # # #     #     + 0.4  * l_spdcons
# # # # # #     #     + 0.6  * l_pwr        # PWR với cao nhất (doc: ưu tiên intensity)
# # # # # #     # )

# # # # # #     # # FIX-L-B: AdaptClamp thay vì tanh cố định
# # # # # #     # total_clamped = adapt_clamp(total, epoch, max_val=20.0)

# # # # # #     # # FIX-L-E + FIX-L-F: áp dụng frequency compensation và boundary weight
# # # # # #     # return total_clamped * w_bnd * f_lazy
# # # # # #      # 1. Định nghĩa Trọng số thời gian cho PINN (Key để giảm 72h)
# # # # # #     # Tăng dần từ 0.5 (ở 6h) lên 4.0 (ở 72h)
# # # # # #     pinn_step_w = torch.linspace(0.5, 4.0, T, device=pred_abs_deg.device)
    
# # # # # #     # 2. Tính toán các thành phần (Giả sử các hàm này trả về tensor [T_i, B])
# # # # # #     # Nếu hàm của bạn đang trả về scalar, hãy sửa chúng để KHÔNG gọi .mean() ở cuối
# # # # # #     l_sw_map      = pinn_shallow_water(pred_abs_deg)          # [T-2, B]
# # # # # #     l_steer_map   = pinn_rankine_steering(pred_abs_deg, _env) # [T-1, B]
# # # # # #     l_speed_map   = pinn_speed_constraint(pred_abs_deg)       # [T-1, B]
# # # # # #     l_gph_map     = pinn_gph500_gradient(pred_abs_deg, _env)  # [T-1, B]
# # # # # #     l_spdcons_map = pinn_steering_speed_consistency(pred_abs_deg, _env)   # [T-1, B]
# # # # # #     l_pwr_map     = pinn_pressure_wind_loss(pred_abs_deg, vmax_pred, pmin_pred, r34_km, epoch) # [T, B]

# # # # # #     # 3. Tính Adaptive Weighting (FIX-L-C) nhưng giữ nguyên theo step
# # # # # #     if gt_abs_deg is not None:
# # # # # #         with torch.no_grad():
# # # # # #             d_track = _haversine_deg(pred_abs_deg, gt_abs_deg)
# # # # # #             w_bve_step = torch.sigmoid(1.0 - d_track / 200.0) # [T, B]
# # # # # #     else:
# # # # # #         w_bve_step = torch.ones(T, B, device=pred_abs_deg.device)

# # # # # #     # 4. Tổng hợp loss theo từng bước thời gian (Pointwise Total)
# # # # # #     # Ta lấy phần đuôi của pinn_step_w để khớp với số lượng step của từng loại loss
# # # # # #     def apply_w(l_map, weight_scalar):
# # # # # #         t_size = l_map.shape[0]
# # # # # #         # Nhân trọng số thành phần * trọng số thời gian * adaptive weight
# # # # # #         return weight_scalar * l_map * pinn_step_w[-t_size:, None] * w_bve_step[-t_size:]

# # # # # #     total_pointwise = (
# # # # # #         apply_w(l_sw_map, 1.0)           # Trọng số gốc 1.0
# # # # # #         + apply_w(l_steer_map, 0.5)
# # # # # #         + apply_w(l_speed_map, 0.1)
# # # # # #         + apply_w(l_gph_map, 0.3)
# # # # # #         + apply_w(l_spdcons_map, 0.4)
# # # # # #         + apply_w(l_pwr_map, 0.6)
# # # # # #     )

# # # # # #     # 5. Lấy trung bình toàn bộ
# # # # # #     total = total_pointwise.mean()

# # # # # #     # FIX-L-B: AdaptClamp
# # # # # #     total_clamped = adapt_clamp(total, epoch, max_val=20.0)

# # # # # #     return total_clamped * w_bnd * f_lazy
# # # # # def pinn_bve_loss(
# # # # #     pred_abs_deg: torch.Tensor,
# # # # #     batch_list,
# # # # #     env_data:    Optional[dict] = None,
# # # # #     epoch:       int = 0,
# # # # #     gt_abs_deg:  Optional[torch.Tensor] = None,
# # # # #     vmax_pred:   Optional[torch.Tensor] = None,
# # # # #     pmin_pred:   Optional[torch.Tensor] = None,
# # # # #     r34_km:      Optional[torch.Tensor] = None,
# # # # # ) -> torch.Tensor:
# # # # #     T = pred_abs_deg.shape[0]
# # # # #     if T < 3: return pred_abs_deg.new_zeros(())

# # # # #     _env = env_data
# # # # #     if _env is None and batch_list is not None:
# # # # #         try: _env = batch_list[13]
# # # # #         except: _env = None

# # # # #     # 1. Trọng số thời gian tăng mạnh ở 72h
# # # # #     pinn_time_w = torch.linspace(0.5, 4.0, T, device=pred_abs_deg.device)

# # # # #     # 2. Adaptive weighting (FIX-L-C) - giữ nguyên chiều [T, B]
# # # # #     if gt_abs_deg is not None:
# # # # #         with torch.no_grad():
# # # # #             d_track = _haversine_deg(pred_abs_deg, gt_abs_deg)
# # # # #             w_bve_step = torch.sigmoid(1.0 - d_track / 200.0) # [T, B]
# # # # #     else:
# # # # #         w_bve_step = torch.ones(T, pred_abs_deg.shape[1], device=pred_abs_deg.device)

# # # # #     # 3. Hàm hỗ trợ nhân trọng số
# # # # #     # def apply_w(l_map, weight_scalar):
# # # # #     #     t_size = l_map.shape[0] # Giờ l_map đã có chiều [T_i, B]
# # # # #     #     # Nhân: (loss từng bước) * (weight thành phần) * (weight thời gian) * (adaptive weight)
# # # # #     #     return weight_scalar * l_map * pinn_time_w[-t_size:, None] * w_bve_step[-t_size:]
# # # # #     def apply_w(l_map, weight_scalar):
# # # # #         # Kiểm tra nếu l_map là scalar (đề phòng chưa sửa hết các hàm con)
# # # # #         if l_map.dim() == 0: 
# # # # #             return l_map * weight_scalar
            
# # # # #         t_size = l_map.shape[0] 
# # # # #         # Nhân trọng số thành phần * trọng số thời gian (pinn_time_w)
# # # # #         w_final = weight_scalar * pinn_time_w[-t_size:, None] 
# # # # #         return (l_map * w_final).mean() # Chỉ .mean() tại đây
    
# # # # #     # 4. Tính toán các thành phần (lúc này các hàm đã trả về Tensor)
# # # # #     l_sw      = apply_w(pinn_shallow_water(pred_abs_deg), 1.0)
# # # # #     l_steer   = apply_w(pinn_rankine_steering(pred_abs_deg, _env), 0.5)
# # # # #     l_speed   = apply_w(pinn_speed_constraint(pred_abs_deg), 0.1)
# # # # #     l_gph     = apply_w(pinn_gph500_gradient(pred_abs_deg, _env), 0.3)
# # # # #     l_spdcons = apply_w(pinn_steering_speed_consistency(pred_abs_deg, _env), 0.4)
# # # # #     l_pwr     = apply_w(pinn_pressure_wind_loss(pred_abs_deg, vmax_pred, pmin_pred, r34_km, epoch), 0.6)

# # # # #     # 5. Tổng hợp và trung bình
# # # # #     total = (l_sw.mean() + l_steer.mean() + l_speed.mean() + 
# # # # #              l_gph.mean() + l_spdcons.mean() + l_pwr.mean())

# # # # #     # Các phần f_lazy, w_bnd và AdaptClamp giữ nguyên
# # # # #     w_bnd = _boundary_weights(pred_abs_deg).mean()
# # # # #     f_lazy = 0.2 if epoch < 30 else (0.5 if epoch < 50 else 1.0)
    
# # # # #     total_clamped = adapt_clamp(total, epoch, max_val=20.0)
# # # # #     return total_clamped * w_bnd * f_lazy


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Physics consistency (beta drift)
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def fm_physics_consistency_loss(
# # # # #     pred_samples: torch.Tensor,
# # # # #     gt_norm:      torch.Tensor,
# # # # #     last_pos:     torch.Tensor,
# # # # # ) -> torch.Tensor:
# # # # #     S, T, B = pred_samples.shape[:3]

# # # # #     last_lon = (last_pos[:, 0] * 50.0 + 1800.0) / 10.0
# # # # #     last_lat = (last_pos[:, 1] * 50.0) / 10.0
# # # # #     lat_rad  = torch.deg2rad(last_lat)
# # # # #     beta     = 2 * OMEGA * torch.cos(lat_rad) / R_EARTH
# # # # #     R_tc     = 3e5

# # # # #     v_beta_lon = -beta * R_tc ** 2 / 2
# # # # #     v_beta_lat =  beta * R_tc ** 2 / 4

# # # # #     beta_dir      = torch.stack([v_beta_lon, v_beta_lat], dim=-1)
# # # # #     beta_norm     = beta_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # #     beta_dir_unit = beta_dir / beta_norm

# # # # #     pos_step1 = pred_samples[:, 0, :, :2]
# # # # #     dir_step1 = pos_step1 - last_pos.unsqueeze(0)
# # # # #     dir_norm  = dir_step1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # #     dir_unit  = dir_step1 / dir_norm
# # # # #     mean_dir  = dir_unit.mean(0)
# # # # #     mean_norm = mean_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # #     mean_dir_unit = mean_dir / mean_norm

# # # # #     cos_align     = (mean_dir_unit * beta_dir_unit).sum(-1)
# # # # #     beta_strength = beta_norm.squeeze(-1)
# # # # #     penalise_mask = (beta_strength > 1.0).float()
# # # # #     direction_loss = F.relu(-cos_align) * penalise_mask
# # # # #     return direction_loss.mean() * 0.5


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Ensemble spread
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def ensemble_spread_loss(all_trajs: torch.Tensor,
# # # # # #                          max_spread_km: float = 150.0) -> torch.Tensor:
# # # # # #     if all_trajs.shape[0] < 2:
# # # # # #         return all_trajs.new_zeros(())

# # # # # #     S, T, B, _ = all_trajs.shape

# # # # # #     step_weights = torch.exp(
# # # # # #         -torch.arange(T, dtype=torch.float, device=all_trajs.device)
# # # # # #         * (math.log(4.0) / max(T - 1, 1))
# # # # # #     ) * 2.0

# # # # # #     total_loss = all_trajs.new_zeros(())
# # # # # #     for t in range(T):
# # # # # #         step_trajs = all_trajs[:, t, :, :2]
# # # # # #         std_lon    = step_trajs[:, :, 0].std(0)
# # # # # #         std_lat    = step_trajs[:, :, 1].std(0)
# # # # # #         spread_km  = torch.sqrt(std_lon ** 2 + std_lat ** 2) * 500.0
# # # # # #         excess     = F.relu(spread_km - max_spread_km)
# # # # # #         total_loss = total_loss + step_weights[t] * (
# # # # # #             excess / max_spread_km
# # # # # #         ).pow(2).mean()

# # # # # #     return total_loss / T
# # # # # def ensemble_spread_loss(all_trajs: torch.Tensor) -> torch.Tensor:
# # # # #     if all_trajs.shape[0] < 2:
# # # # #         return all_trajs.new_zeros(())

# # # # #     S, T, B, _ = all_trajs.shape
# # # # #     device = all_trajs.device

# # # # #     # 1. ĐẢO NGƯỢC TRỌNG SỐ: Tăng dần từ 6h đến 72h
# # # # #     # Step 1 (6h) weight = 0.8, Step 12 (72h) weight = 4.0
# # # # #     # Điều này bắt mô hình phải ưu tiên siết spread ở horizon xa
# # # # #     step_weights = torch.linspace(0.8, 4.0, T, device=device)

# # # # #     # 2. NGƯỠNG ĐỘNG (Dynamic Threshold): 
# # # # #     # 6h không nên spread quá 60km, 72h không nên spread quá 180km
# # # # #     # Ép một ngưỡng cứng 150km ở 72h là rất khó, nên dùng 170-180km là hợp lý
# # # # #     max_spreads = torch.linspace(60.0, 180.0, T, device=device)

# # # # #     total_loss = all_trajs.new_zeros(())
# # # # #     for t in range(T):
# # # # #         step_trajs = all_trajs[:, t, :, :2]
# # # # #         std_lon    = step_trajs[:, :, 0].std(0)
# # # # #         std_lat    = step_trajs[:, :, 1].std(0)
# # # # #         spread_km  = torch.sqrt(std_lon ** 2 + std_lat ** 2) * 500.0 # [B]
        
# # # # #         # 3. PHẠT BÌNH PHƯƠNG MẠNH
# # # # #         # Nếu spread vượt ngưỡng động tại thời điểm t
# # # # #         excess = F.relu(spread_km - max_spreads[t])
        
# # # # #         # Dùng hằng số chia nhỏ hơn (ví dụ 40.0) để gradient dốc hơn
# # # # #         loss = (excess / 40.0).pow(2) 
        
# # # # #         total_loss = total_loss + step_weights[t] * loss.mean()

# # # # #     return total_loss / T

# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Main loss
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def compute_total_loss(
# # # # #     pred_abs,
# # # # #     gt,
# # # # #     ref,
# # # # #     batch_list,
# # # # #     pred_samples       = None,
# # # # #     gt_norm            = None,
# # # # #     weights            = WEIGHTS,
# # # # #     intensity_w: Optional[torch.Tensor] = None,
# # # # #     env_data:    Optional[dict]         = None,
# # # # #     step_weight_alpha: float = 0.0,
# # # # #     all_trajs:   Optional[torch.Tensor] = None,
# # # # #     epoch:       int = 0,
# # # # #     # FIX-L-C: cần gt_abs_deg cho adaptive BVE weighting
# # # # #     gt_abs_deg:  Optional[torch.Tensor] = None,
# # # # #     # FIX-L-D: intensity predictions cho PWR
# # # # #     vmax_pred:   Optional[torch.Tensor] = None,
# # # # #     pmin_pred:   Optional[torch.Tensor] = None,
# # # # #     r34_km:      Optional[torch.Tensor] = None,
# # # # #     # FIX-L-H: sr_pred cho bridge loss
# # # # #     sr_pred:     Optional[torch.Tensor] = None,
# # # # # ) -> Dict:
# # # # #     NRM = 35.0

# # # # #     # 1. Sample weights
# # # # #     recurv_w = _recurvature_weights(gt, w_recurv=2.5)
# # # # #     sample_w = recurv_w * (intensity_w.to(gt.device) if intensity_w is not None
# # # # #                            else 1.0)
# # # # #     sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

# # # # #     # 2. AFCRPS (FIX-L-G: thêm w_es)
# # # # #     if pred_samples is not None:
# # # # #         target = gt_norm if gt_norm is not None else gt
# # # # #         unit   = gt_norm is not None
# # # # #         l_fm = fm_afcrps_loss(
# # # # #             pred_samples, target,
# # # # #             unit_01deg=unit,
# # # # #             intensity_w=sample_w,
# # # # #             step_weight_alpha=step_weight_alpha,
# # # # #             w_es=0.3,
# # # # #         )
# # # # #     else:
# # # # #         l_fm = _haversine_deg(pred_abs, gt).mean()

# # # # #     # # 3. Directional losses
# # # # #     # l_vel       = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
# # # # #     # l_disp      = (disp_loss_per_sample(pred_abs, gt)     * sample_w).mean()
# # # # #     # l_step      = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()
# # # # #     # l_heading   = heading_loss(pred_abs, gt)
# # # # #     # l_recurv    = recurvature_loss(pred_abs, gt)
# # # # #     # l_dir_final = overall_dir_loss(pred_abs, gt, ref)
# # # # #     # l_smooth    = smooth_loss(pred_abs)
# # # # #     # l_accel     = acceleration_loss(pred_abs)
# # # # #     # l_jerk      = jerk_loss(pred_abs)

# # # # #     # # 4. PINN (FIX-L-A: KHÔNG nhân NRM ở đây; FIX-L-C/D/E/F: pass extra args)
# # # # #     # # _env = env_data
# # # # #     # # if _env is None and batch_list is not None:
# # # # #     # #     try:
# # # # #     # #         _env = batch_list[13]
# # # # #     # #     except (IndexError, TypeError):
# # # # #     # #         _env = None

# # # # #     # # l_pinn = pinn_bve_loss(
# # # # #     # #     pred_abs, batch_list, env_data=_env,
# # # # #     # #     epoch=epoch,
# # # # #     # #     gt_abs_deg=gt_abs_deg,
# # # # #     # #     vmax_pred=vmax_pred,
# # # # #     # #     pmin_pred=pmin_pred,
# # # # #     # #     r34_km=r34_km,
# # # # #     # # )

# # # # #     # # 4. PINN (FIX-L-A: KHÔNG nhân NRM ở đây; FIX-L-C/D/E/F: pass extra args)
# # # # #     # _env = env_data
# # # # #     # if _env is None and batch_list is not None:
# # # # #     #     try:
# # # # #     #         _env = batch_list[13]
# # # # #     #     except (IndexError, TypeError):
# # # # #     #         _env = None

# # # # #     # # Tính PINN cho Mean (định hướng quỹ đạo trung tâm)
# # # # #     # l_pinn_mean = pinn_bve_loss(
# # # # #     #     pred_abs_deg=pred_abs, 
# # # # #     #     batch_list=batch_list, 
# # # # #     #     env_data=_env,
# # # # #     #     epoch=epoch,
# # # # #     #     gt_abs_deg=gt_abs_deg,
# # # # #     #     vmax_pred=vmax_pred,
# # # # #     #     pmin_pred=pmin_pred,
# # # # #     #     r34_km=r34_km
# # # # #     # )

# # # # #     # # Stochastic PINN: Ép vật lý lên từng hạt ensemble ở Phase 2
# # # # #     # if pred_samples is not None and epoch >= 30: 
# # # # #     #     M = pred_samples.shape[0]
# # # # #     #     # Chọn ngẫu nhiên 2 hạt để tính PINN (tiết kiệm memory)
# # # # #     #     idxs = torch.randperm(M)[:2]
        
# # # # #     #     l_pinn_samples = []
# # # # #     #     for idx in idxs:
# # # # #     #         # Decode sample từ normalized sang degrees
# # # # #     #         sample_deg = _norm_to_deg(pred_samples[idx])
            
# # # # #     #         l_p_sample = pinn_bve_loss(
# # # # #     #             pred_abs_deg=sample_deg, 
# # # # #     #             batch_list=batch_list, 
# # # # #     #             env_data=_env,
# # # # #     #             epoch=epoch,
# # # # #     #             gt_abs_deg=gt_abs_deg, # Các hạt đều phải hướng về GT chung
# # # # #     #             vmax_pred=vmax_pred,
# # # # #     #             pmin_pred=pmin_pred,
# # # # #     #             r34_km=r34_km
# # # # #     #         )
# # # # #     #         l_pinn_samples.append(l_p_sample)
        
# # # # #     #     # Kết hợp: 40% Mean + 60% Samples
# # # # #     #     l_pinn = 0.4 * l_pinn_mean + 0.6 * torch.stack(l_pinn_samples).mean()
# # # # #     # else:
# # # # #     #     l_pinn = l_pinn_mean
        
# # # # #     # ... (đoạn tính l_fm và directional losses cơ bản ở trên giữ nguyên)

# # # # #     # 3. Directional losses (Tính trên Mean trước)
# # # # #     l_vel       = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
# # # # #     l_disp      = (disp_loss_per_sample(pred_abs, gt)     * sample_w).mean()
# # # # #     l_step      = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()
# # # # #     l_heading   = heading_loss(pred_abs, gt)
# # # # #     l_recurv    = recurvature_loss(pred_abs, gt)
# # # # #     l_dir_final = overall_dir_loss(pred_abs, gt, ref)
# # # # #     l_smooth    = smooth_loss(pred_abs)
# # # # #     l_accel     = acceleration_loss(pred_abs)
# # # # #     l_jerk      = jerk_loss(pred_abs)

# # # # #     # 4. PINN & Stochastic Physics (Tích hợp Smoothness vào đây)
# # # # #     _env = env_data
# # # # #     if _env is None and batch_list is not None:
# # # # #         try: _env = batch_list[13]
# # # # #         except (IndexError, TypeError): _env = None

# # # # #     # PINN trên Mean
# # # # #     l_pinn_mean = pinn_bve_loss(pred_abs, batch_list, env_data=_env, epoch=epoch,
# # # # #                                 gt_abs_deg=gt_abs_deg, vmax_pred=vmax_pred,
# # # # #                                 pmin_pred=pmin_pred, r34_km=r34_km)

# # # # #     if pred_samples is not None and epoch >= 15: 
# # # # #         M = pred_samples.shape[0]
# # # # #         idxs = torch.randperm(M)[:2]
        
# # # # #         l_pinn_samples = []
# # # # #         l_smooth_samples = []
# # # # #         l_accel_samples = []
        
# # # # #         for idx in idxs:
# # # # #             sample_deg = _norm_to_deg(pred_samples[idx])
# # # # #             l_p_sample = pinn_bve_loss(sample_deg, batch_list, env_data=_env, epoch=epoch,
# # # # #                                        gt_abs_deg=gt_abs_deg, vmax_pred=vmax_pred,
# # # # #                                        pmin_pred=pmin_pred, r34_km=r34_km)
# # # # #             l_pinn_samples.append(l_p_sample)
# # # # #             l_smooth_samples.append(smooth_loss(sample_deg))
# # # # #             l_accel_samples.append(acceleration_loss(sample_deg))
        
# # # # #         l_pinn = 0.4 * l_pinn_mean + 0.6 * torch.stack(l_pinn_samples).mean()
# # # # #     else:                          # ← THÊM DÒNG NÀY
# # # # #         l_pinn = l_pinn_mean       # ← THÊM DÒNG NÀY
        
# # # # #         # Cập nhật Smoothness & Accel final (Blended 50/50)
# # # # #         # Việc ép smoothness lên từng hạt giúp giảm hiện tượng ziczac ở ensemble, cải thiện spread
# # # # #     # 5. Spread penalty (chỉ FM steps 5-12)
# # # # #     l_spread = pred_abs.new_zeros(())
# # # # #     if all_trajs is not None and all_trajs.shape[0] >= 2:
# # # # #         n_sr = 4
# # # # #         trajs_fm = all_trajs[:, n_sr:] if all_trajs.shape[1] > n_sr else all_trajs
# # # # #         l_spread = ensemble_spread_loss(trajs_fm)

# # # # #     # 6. Bridge loss (FIX-L-H)
# # # # #     # l_bridge = pred_abs.new_zeros(())
# # # # #     # if sr_pred is not None and pred_samples is not None:
# # # # #     #     fm_mean = pred_samples.mean(0)   # [T, B, 2]  (FM ensemble mean, normalized)
# # # # #     #     # Chuyển FM mean sang degree
# # # # #     #     fm_mean_deg = _norm_to_deg(fm_mean)
# # # # #     #     sr_pred_deg = _norm_to_deg(sr_pred)   # [4, B, 2] degrees
# # # # #     #     l_bridge = bridge_loss(sr_pred_deg, fm_mean_deg)

# # # # #     # SỬA: bỏ _norm_to_deg khi gọi bridge_loss
# # # # #     l_bridge = pred_abs.new_zeros(())
# # # # #     if sr_pred is not None and pred_samples is not None:
# # # # #         fm_mean = pred_samples.mean(0)   # [T, B, 2] normalized
# # # # #         # KHÔNG decode ở đây — bridge_loss tự decode bên trong
# # # # #         l_bridge = bridge_loss(sr_pred, fm_mean)
# # # # #     # 7. Total
# # # # #     # FIX-L-A: pinn KHÔNG nhân NRM — l_pinn đã có AdaptClamp(max=20)
# # # # #     # Các loss directional nhân NRM để đưa về cùng thang với l_fm (km)
# # # # #     total = (
# # # # #         weights.get("fm",       2.0) * l_fm
# # # # #         + weights.get("velocity", 0.8) * l_vel     * NRM
# # # # #         + weights.get("disp",     0.5) * l_disp    * NRM
# # # # #         + weights.get("step",     0.5) * l_step    * NRM
# # # # #         + weights.get("heading",  2.0) * l_heading * NRM
# # # # #         + weights.get("recurv",   1.5) * l_recurv  * NRM
# # # # #         + weights.get("dir",      1.0) * l_dir_final * NRM
# # # # #         + weights.get("smooth",   0.5) * l_smooth  * NRM
# # # # #         + weights.get("accel",    0.8) * l_accel   * NRM
# # # # #         + weights.get("jerk",     0.3) * l_jerk    * NRM
# # # # #         + weights.get("pinn",     0.5) * l_pinn              # FIX-L-A: không * NRM
# # # # #         + weights.get("spread",   0.8) * l_spread  * NRM
# # # # #         + weights.get("bridge",   0.5) * l_bridge  * NRM
# # # # #     ) / NRM

# # # # #     if torch.isnan(total) or torch.isinf(total):
# # # # #         total = pred_abs.new_zeros(())

# # # # #     return dict(
# # # # #         total        = total,
# # # # #         fm           = l_fm.item(),
# # # # #         velocity     = l_vel.item()     * NRM,
# # # # #         step         = l_step.item(),
# # # # #         disp         = l_disp.item()    * NRM,
# # # # #         heading      = l_heading.item(),
# # # # #         recurv       = l_recurv.item(),
# # # # #         smooth       = l_smooth.item()  * NRM,
# # # # #         accel        = l_accel.item()   * NRM,
# # # # #         jerk         = l_jerk.item()    * NRM,
# # # # #         pinn         = l_pinn.item(),    # raw value, không * NRM
# # # # #         spread       = l_spread.item()  * NRM,
# # # # #         bridge       = l_bridge.item()  * NRM,
# # # # #         recurv_ratio = (_recurvature_weights(gt) > 1.0).float().mean().item(),
# # # # #     )


# # # # # # ── Legacy ─────────────────────────────────────────────────────────────────────

# # # # # class TripletLoss(torch.nn.Module):
# # # # #     def __init__(self, margin=None):
# # # # #         super().__init__()
# # # # #         self.margin  = margin
# # # # #         self.loss_fn = (torch.nn.SoftMarginLoss() if margin is None
# # # # #                         else torch.nn.TripletMarginLoss(margin=margin, p=2))

# # # # #     def forward(self, anchor, pos, neg):
# # # # #         if self.margin is None:
# # # # #             y = torch.ones(anchor.shape[0], device=anchor.device)
# # # # #             return self.loss_fn(
# # # # #                 torch.norm(anchor - neg, 2, dim=1)
# # # # #                 - torch.norm(anchor - pos, 2, dim=1), y)
# # # # #         return self.loss_fn(anchor, pos, neg)

# # # # """
# # # # Model/losses.py  ── v27 - BALANCED SR+FM
# # # # =========================================
# # # # CRITICAL CHANGES để đạt 12h<50, 24h<100, 72h<300:

# # # #   FIX-L-NEW-A  [CRITICAL] FM time weights: tăng weight cho 48h-72h
# # # #                thay vì flat/linear → exponential growth

# # # #   FIX-L-NEW-B  [CRITICAL] Bridge loss mạnh hơn: weight 2.0 thay vì 0.5
# # # #                và extend ra step 5-6 (không chỉ step 4)

# # # #   FIX-L-NEW-C  [HIGH] Spread loss với dynamic threshold chặt hơn
# # # #                60km→120km thay vì 60→180km

# # # #   FIX-L-NEW-D  [HIGH] PINN time weighting: tập trung vào 48h-72h
               
# # # #   FIX-L-NEW-E  [MEDIUM] Velocity consistency loss giữa SR và FM
# # # # """
# # # # from __future__ import annotations

# # # # import math
# # # # from typing import Dict, Optional, Tuple

# # # # import torch
# # # # import torch.nn.functional as F

# # # # __all__ = [
# # # #     "WEIGHTS", "compute_total_loss", "fm_afcrps_loss",
# # # #     "fm_physics_consistency_loss", "pinn_bve_loss",
# # # #     "recurvature_loss", "velocity_loss_per_sample",
# # # #     "short_range_regression_loss", "bridge_loss",
# # # #     "adapt_clamp", "pinn_pressure_wind_loss",
# # # # ]

# # # # # ── Constants ─────────────────────────────────────────────────────────────────
# # # # OMEGA        = 7.2921e-5
# # # # R_EARTH      = 6.371e6
# # # # DT_6H        = 6 * 3600
# # # # DEG_TO_KM    = 111.0
# # # # STEP_KM      = 113.0
# # # # P_ENV        = 1013.0
# # # # RHO_AIR      = 1.15

# # # # _ERA5_LAT_MIN =   0.0
# # # # _ERA5_LAT_MAX =  40.0
# # # # _ERA5_LON_MIN = 100.0
# # # # _ERA5_LON_MAX = 160.0

# # # # _UV500_NORM      = 30.0
# # # # _GPH500_MEAN_M   = 5870.0
# # # # _GPH500_STD_M    = 80.0
# # # # _STEERING_MIN_MS = 3.0
# # # # _PINN_SCALE      = 1e-2

# # # # # ── Weights (sẽ được update động) ─────────────────────────────────────────────
# # # # WEIGHTS: Dict[str, float] = dict(
# # # #     fm          = 3.0,      # Tăng từ 2.0
# # # #     velocity    = 1.0,      # Tăng từ 0.8
# # # #     heading     = 2.0,
# # # #     recurv      = 1.5,
# # # #     step        = 0.5,
# # # #     disp        = 0.5,
# # # #     dir         = 1.0,
# # # #     smooth      = 0.5,
# # # #     accel       = 0.8,
# # # #     jerk        = 0.3,
# # # #     pinn        = 1.0,      # Tăng từ 0.5
# # # #     fm_physics  = 0.5,      # Tăng từ 0.3
# # # #     spread      = 1.5,      # Tăng từ 0.8
# # # #     short_range = 5.0,
# # # #     bridge      = 2.0,      # FIX-L-NEW-B: Tăng mạnh từ 0.5
# # # #     sr_fm_vel   = 1.0,      # NEW: velocity consistency
# # # # )

# # # # RECURV_ANGLE_THR = 45.0
# # # # RECURV_WEIGHT    = 2.5

# # # # _SR_N_STEPS  = 4
# # # # _SR_WEIGHTS  = [2.0, 4.0, 2.0, 2.0]
# # # # _HUBER_DELTA = 50.0


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Haversine utilities
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def _haversine(p1: torch.Tensor, p2: torch.Tensor,
# # # #                unit_01deg: bool = True) -> torch.Tensor:
# # # #     if unit_01deg:
# # # #         lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
# # # #         lat1 = (p1[..., 1] * 50.0) / 10.0
# # # #         lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
# # # #         lat2 = (p2[..., 1] * 50.0) / 10.0
# # # #     else:
# # # #         lon1, lat1 = p1[..., 0], p1[..., 1]
# # # #         lon2, lat2 = p2[..., 0], p2[..., 1]

# # # #     lat1r = torch.deg2rad(lat1); lat2r = torch.deg2rad(lat2)
# # # #     dlon  = torch.deg2rad(lon2 - lon1)
# # # #     dlat  = torch.deg2rad(lat2 - lat1)
# # # #     a = (torch.sin(dlat / 2).pow(2)
# # # #          + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2).pow(2))
# # # #     a = a.clamp(1e-12, 1.0 - 1e-12)
# # # #     return 2.0 * 6371.0 * torch.asin(a.sqrt())


# # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # #     return _haversine(p1, p2, unit_01deg=False)


# # # # def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
# # # #     out = arr.clone()
# # # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# # # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# # # #     return out


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  AdaptClamp
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def adapt_clamp(x: torch.Tensor, epoch: int, max_val: float = 20.0) -> torch.Tensor:
# # # #     delta = max_val

# # # #     def huber_clamp(v: torch.Tensor) -> torch.Tensor:
# # # #         return torch.where(
# # # #             v <= delta,
# # # #             v.pow(2) / (2.0 * delta),
# # # #             v - delta / 2.0,
# # # #         )

# # # #     def tanh_clamp(v: torch.Tensor) -> torch.Tensor:
# # # #         return delta * torch.tanh(v / delta)

# # # #     if epoch < 10:
# # # #         return huber_clamp(x)
# # # #     elif epoch < 20:
# # # #         beta = (epoch - 10) / 10.0
# # # #         return (1.0 - beta) * huber_clamp(x) + beta * tanh_clamp(x)
# # # #     else:
# # # #         return tanh_clamp(x)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  FIX-L-NEW-A: FM AFCRPS với Time Weights tập trung 48h-72h
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def fm_afcrps_loss(
# # # #     pred_samples:      torch.Tensor,
# # # #     gt:                torch.Tensor,
# # # #     unit_01deg:        bool  = True,
# # # #     intensity_w:       Optional[torch.Tensor] = None,
# # # #     step_weight_alpha: float = 0.0,
# # # #     w_es:              float = 0.3,
# # # #     epoch:             int   = 0,
# # # # ) -> torch.Tensor:
# # # #     """
# # # #     FIX-L-NEW-A: Time weights exponential cho 48h-72h
    
# # # #     Thay vì:  [0.5, 0.5, ..., 1.5, 3.0]  (linear)
# # # #     Dùng:     [0.3, 0.3, 0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 4.0, 5.0, 6.0, 8.0]
    
# # # #     Mục tiêu: Ép model phải predict 72h chính xác
# # # #     """
# # # #     M, T, B, _ = pred_samples.shape
# # # #     device = pred_samples.device

# # # #     # FIX-L-NEW-A: Exponential time weights
# # # #     # Step 1-4 (6h-24h): weight thấp (SR handle)
# # # #     # Step 5-8 (30h-48h): weight trung bình
# # # #     # Step 9-12 (54h-72h): weight cao
# # # #     base_w = torch.zeros(T, device=device)
# # # #     for i in range(T):
# # # #         if i < 4:       # 6h-24h: SR handles, FM weight thấp
# # # #             base_w[i] = 0.3
# # # #         elif i < 8:     # 30h-48h: transition zone
# # # #             base_w[i] = 1.0 + (i - 4) * 0.5  # 1.0, 1.5, 2.0, 2.5
# # # #         else:           # 54h-72h: FM critical zone
# # # #             base_w[i] = 4.0 + (i - 8) * 1.5  # 4.0, 5.5, 7.0, 8.5
    
# # # #     # Normalize để tổng weight hợp lý
# # # #     base_w = base_w / base_w.mean() * 2.0
    
# # # #     if step_weight_alpha > 0.0:
# # # #         early_w = torch.exp(
# # # #             -torch.arange(T, dtype=torch.float, device=device) * 0.3
# # # #         )
# # # #         early_w = early_w / early_w.mean()
# # # #         time_w = (1.0 - step_weight_alpha) * base_w + step_weight_alpha * early_w
# # # #     else:
# # # #         time_w = base_w

# # # #     if M == 1:
# # # #         dist = _haversine(pred_samples[0], gt, unit_01deg)
# # # #         loss_per_b = (dist * time_w.unsqueeze(1)).mean(0)
# # # #         es_term = pred_samples.new_zeros(())
# # # #     else:
# # # #         # Accuracy
# # # #         d_to_gt = _haversine(
# # # #             pred_samples,
# # # #             gt.unsqueeze(0).expand_as(pred_samples),
# # # #             unit_01deg,
# # # #         )
# # # #         d_to_gt_w = d_to_gt * time_w.unsqueeze(0).unsqueeze(2)
# # # #         e_sy = d_to_gt_w.mean(1).mean(0)

# # # #         # Sharpness (fair)
# # # #         ps_i = pred_samples.unsqueeze(1)
# # # #         ps_j = pred_samples.unsqueeze(0)
# # # #         d_pair = _haversine(
# # # #             ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # # #             ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # # #             unit_01deg,
# # # #         ).reshape(M, M, T, B)

# # # #         diag_mask = torch.eye(M, device=device, dtype=torch.bool)
# # # #         diag_mask = diag_mask.view(M, M, 1, 1).expand_as(d_pair)
# # # #         d_pair = d_pair.masked_fill(diag_mask, 0.0)

# # # #         d_pair_w = d_pair * time_w.unsqueeze(0).unsqueeze(0).unsqueeze(3)
# # # #         e_ssp = d_pair_w.mean(2).sum(0).sum(0) / (M * (M - 1))

# # # #         loss_per_b = e_sy - 0.5 * e_ssp

# # # #         # Energy Score
# # # #         if w_es > 0.0 and M > 1:
# # # #             ps_flat = pred_samples.reshape(M, T * B, 2)
# # # #             gt_flat = gt.reshape(T * B, 2)
# # # #             mean_pred = ps_flat.mean(0)

# # # #             es_acc = (mean_pred - gt_flat).pow(2).sum(-1).sqrt().mean()

# # # #             ps_i_f = ps_flat.unsqueeze(1)
# # # #             ps_j_f = ps_flat.unsqueeze(0)
# # # #             d_pair_f = (ps_i_f - ps_j_f).pow(2).sum(-1).sqrt()
# # # #             diag_f = torch.eye(M, device=device, dtype=torch.bool)
# # # #             diag_f = diag_f.view(M, M, 1).expand_as(d_pair_f)
# # # #             d_pair_f = d_pair_f.masked_fill(diag_f, 0.0)
# # # #             es_sharp = (M - 1) / M * d_pair_f.sum(0).sum(0).mean() / (M * (M - 1))

# # # #             es_term = es_acc - es_sharp
# # # #         else:
# # # #             es_term = pred_samples.new_zeros(())

# # # #     if intensity_w is not None:
# # # #         w = intensity_w.to(loss_per_b.device)
# # # #         crps_loss = (loss_per_b * w).mean()
# # # #     else:
# # # #         crps_loss = loss_per_b.mean()

# # # #     return crps_loss + w_es * es_term


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  FIX-L-NEW-B: Extended Bridge Loss (step 4, 5, 6)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def bridge_loss(
# # # #     sr_pred:  torch.Tensor,   # [4, B, 2] normalised
# # # #     fm_mean:  torch.Tensor,   # [T, B, 2] normalised
# # # #     epoch:    int = 0,
# # # # ) -> torch.Tensor:
# # # #     """
# # # #     FIX-L-NEW-B: Extended bridge loss
    
# # # #     Không chỉ enforce consistency tại step 4, mà còn:
# # # #     - Step 4: position + velocity match
# # # #     - Step 5-6: velocity direction match (FM phải tiếp tục SR momentum)
# # # #     """
# # # #     if sr_pred.shape[0] < 4 or fm_mean.shape[0] < 6:
# # # #         return sr_pred.new_zeros(())

# # # #     # 1. Position consistency at step 4
# # # #     pos_sr4 = _norm_to_deg(sr_pred[3])
# # # #     pos_fm4 = _norm_to_deg(fm_mean[3])
# # # #     dist_pos = _haversine_deg(pos_sr4, pos_fm4)
# # # #     l_pos = (dist_pos / 50.0).pow(2).mean()  # Normalize bằng 50km (chặt hơn)

# # # #     # 2. Velocity consistency at step 4
# # # #     v_sr3 = _norm_to_deg(sr_pred[3]) - _norm_to_deg(sr_pred[2])
# # # #     v_fm4 = _norm_to_deg(fm_mean[4]) - _norm_to_deg(fm_mean[3])

# # # #     lat_mid = (_norm_to_deg(sr_pred[3])[:, 1] + _norm_to_deg(fm_mean[4])[:, 1]) / 2.0
# # # #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)

# # # #     def _to_km(dv, cos_l):
# # # #         km = dv.clone()
# # # #         km[:, 0] = dv[:, 0] * cos_l * DEG_TO_KM
# # # #         km[:, 1] = dv[:, 1] * DEG_TO_KM
# # # #         return km

# # # #     v_sr_km = _to_km(v_sr3, cos_lat)
# # # #     v_fm_km = _to_km(v_fm4, cos_lat)
# # # #     l_vel = ((v_sr_km - v_fm_km).pow(2).sum(-1) / STEP_KM**2).mean()

# # # #     # 3. Direction consistency at step 5-6 (FM phải tiếp tục hướng SR)
# # # #     # Tính hướng SR cuối (step 3-4)
# # # #     sr_dir = v_sr_km / v_sr_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    
# # # #     # Tính hướng FM step 5 và 6
# # # #     l_dir = sr_pred.new_zeros(())
# # # #     for step in [4, 5]:  # FM step 5 và 6
# # # #         if fm_mean.shape[0] > step + 1:
# # # #             v_fm_step = _norm_to_deg(fm_mean[step + 1]) - _norm_to_deg(fm_mean[step])
# # # #             v_fm_step_km = _to_km(v_fm_step, cos_lat)
# # # #             fm_dir = v_fm_step_km / v_fm_step_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
            
# # # #             # Cosine similarity penalty (penalize nếu FM đi ngược SR)
# # # #             cos_sim = (sr_dir * fm_dir).sum(-1)
# # # #             l_dir = l_dir + F.relu(-cos_sim + 0.5).pow(2).mean()  # Penalty nếu cos < 0.5
    
# # # #     # Weight: position > velocity > direction
# # # #     return l_pos + 0.8 * l_vel + 0.3 * l_dir


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  FIX-L-NEW-C: Stricter Spread Loss
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def ensemble_spread_loss(all_trajs: torch.Tensor, epoch: int = 0) -> torch.Tensor:
# # # #     """
# # # #     FIX-L-NEW-C: Spread loss với threshold chặt hơn
    
# # # #     Target:
# # # #     - 6h:  max 40km spread
# # # #     - 24h: max 80km spread  
# # # #     - 48h: max 100km spread
# # # #     - 72h: max 120km spread (chặt hơn 180km cũ)
# # # #     """
# # # #     if all_trajs.shape[0] < 2:
# # # #         return all_trajs.new_zeros(())

# # # #     S, T, B, _ = all_trajs.shape
# # # #     device = all_trajs.device

# # # #     # FIX-L-NEW-C: Threshold chặt hơn, tăng dần theo time
# # # #     # Linear từ 40km (6h) đến 120km (72h)
# # # #     max_spreads = torch.linspace(40.0, 120.0, T, device=device)
    
# # # #     # Weight: tăng dần để enforce spread ở horizon xa
# # # #     step_weights = torch.linspace(0.5, 3.0, T, device=device)

# # # #     total_loss = all_trajs.new_zeros(())
# # # #     for t in range(T):
# # # #         step_trajs = all_trajs[:, t, :, :2]
# # # #         std_lon = step_trajs[:, :, 0].std(0)
# # # #         std_lat = step_trajs[:, :, 1].std(0)
# # # #         spread_km = torch.sqrt(std_lon ** 2 + std_lat ** 2) * 500.0

# # # #         excess = F.relu(spread_km - max_spreads[t])
        
# # # #         # Quadratic penalty với hằng số chia nhỏ hơn → gradient dốc hơn
# # # #         loss = (excess / 30.0).pow(2)
        
# # # #         total_loss = total_loss + step_weights[t] * loss.mean()

# # # #     return total_loss / T


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  FIX-L-NEW-E: SR-FM Velocity Consistency Loss
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def sr_fm_velocity_consistency_loss(
# # # #     sr_pred:  torch.Tensor,   # [4, B, 2] normalised
# # # #     fm_mean:  torch.Tensor,   # [T, B, 2] normalised
# # # # ) -> torch.Tensor:
# # # #     """
# # # #     NEW: Enforce FM phải maintain velocity profile từ SR
    
# # # #     SR cuối có velocity v_sr. FM phải:
# # # #     1. Bắt đầu với velocity tương tự
# # # #     2. Không thay đổi velocity đột ngột (smooth transition)
# # # #     """
# # # #     if sr_pred.shape[0] < 2 or fm_mean.shape[0] < 6:
# # # #         return sr_pred.new_zeros(())

# # # #     # SR velocity cuối (step 3→4)
# # # #     v_sr = sr_pred[3] - sr_pred[2]  # [B, 2] normalised
    
# # # #     # FM velocity đầu (step 4→5)
# # # #     v_fm_start = fm_mean[4] - fm_mean[3]  # [B, 2] normalised
    
# # # #     # Loss: FM velocity phải gần SR velocity
# # # #     l_vel_match = (v_sr - v_fm_start).pow(2).sum(-1).mean()
    
# # # #     # Acceleration smoothness: FM không nên có acceleration đột ngột
# # # #     if fm_mean.shape[0] >= 7:
# # # #         v_fm_1 = fm_mean[5] - fm_mean[4]  # velocity step 5→6
# # # #         v_fm_2 = fm_mean[6] - fm_mean[5]  # velocity step 6→7
        
# # # #         accel_1 = v_fm_start - v_sr      # acceleration từ SR→FM
# # # #         accel_2 = v_fm_1 - v_fm_start    # acceleration step 5
# # # #         accel_3 = v_fm_2 - v_fm_1        # acceleration step 6
        
# # # #         l_accel = (accel_1.pow(2).sum(-1).mean() + 
# # # #                    accel_2.pow(2).sum(-1).mean() + 
# # # #                    accel_3.pow(2).sum(-1).mean()) / 3.0
# # # #     else:
# # # #         l_accel = sr_pred.new_zeros(())
    
# # # #     return l_vel_match + 0.5 * l_accel


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Short-range Huber Loss (giữ nguyên)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def short_range_regression_loss(
# # # #     pred_sr:  torch.Tensor,
# # # #     gt_sr:    torch.Tensor,
# # # #     last_pos: torch.Tensor,
# # # # ) -> torch.Tensor:
# # # #     n_steps = min(pred_sr.shape[0], gt_sr.shape[0], _SR_N_STEPS)
# # # #     if n_steps == 0:
# # # #         return pred_sr.new_zeros(())

# # # #     pred_deg = _norm_to_deg(pred_sr[:n_steps])
# # # #     gt_deg = _norm_to_deg(gt_sr[:n_steps])

# # # #     dist_km = _haversine_deg(pred_deg, gt_deg)

# # # #     huber = torch.where(
# # # #         dist_km < _HUBER_DELTA,
# # # #         0.5 * dist_km.pow(2) / _HUBER_DELTA,
# # # #         dist_km - 0.5 * _HUBER_DELTA,
# # # #     )

# # # #     w = pred_sr.new_tensor(_SR_WEIGHTS[:n_steps])
# # # #     w = w / w.sum()
# # # #     return (huber * w.view(-1, 1)).mean()


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Helper functions (giữ nguyên)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def _boundary_weights(traj_deg: torch.Tensor) -> torch.Tensor:
# # # #     lon = traj_deg[..., 0]
# # # #     lat = traj_deg[..., 1]

# # # #     d_lat_lo = (lat - _ERA5_LAT_MIN) / 5.0
# # # #     d_lat_hi = (_ERA5_LAT_MAX - lat) / 5.0
# # # #     d_lon_lo = (lon - _ERA5_LON_MIN) / 5.0
# # # #     d_lon_hi = (_ERA5_LON_MAX - lon) / 5.0

# # # #     d_bnd = torch.stack([d_lat_lo, d_lat_hi, d_lon_lo, d_lon_hi], dim=-1).min(dim=-1).values
# # # #     return torch.sigmoid(d_bnd - 0.5)


# # # # def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
# # # #     dt = traj_deg[1:] - traj_deg[:-1]
# # # #     lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
# # # #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
# # # #     dt_km = dt.clone()
# # # #     dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
# # # #     dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
# # # #     return dt_km


# # # # def _total_rotation_angle_batch(gt: torch.Tensor) -> torch.Tensor:
# # # #     T, B, _ = gt.shape
# # # #     if T < 3:
# # # #         return gt.new_zeros(B)
# # # #     lats_rad = torch.deg2rad(gt[:, :, 1])
# # # #     cos_lat = torch.cos(lats_rad[:-1])
# # # #     dlat = gt[1:, :, 1] - gt[:-1, :, 1]
# # # #     dlon = (gt[1:, :, 0] - gt[:-1, :, 0]) * cos_lat
# # # #     v = torch.stack([dlon, dlat], dim=-1)
# # # #     v1 = v[:-1]; v2 = v[1:]
# # # #     n1 = v1.norm(dim=-1).clamp(min=1e-8)
# # # #     n2 = v2.norm(dim=-1).clamp(min=1e-8)
# # # #     cos_a = (v1 * v2).sum(-1) / (n1 * n2)
# # # #     return torch.rad2deg(torch.acos(cos_a.clamp(-1.0, 1.0))).sum(0)


# # # # def _recurvature_weights(gt: torch.Tensor,
# # # #                          thr: float = RECURV_ANGLE_THR,
# # # #                          w_recurv: float = RECURV_WEIGHT) -> torch.Tensor:
# # # #     rot = _total_rotation_angle_batch(gt)
# # # #     return torch.where(rot >= thr,
# # # #                        torch.full_like(rot, w_recurv),
# # # #                        torch.ones_like(rot))


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Directional losses (giữ nguyên)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def velocity_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 2:
# # # #         return pred.new_zeros(pred.shape[1])
# # # #     v_pred_km = _step_displacements_km(pred)
# # # #     v_gt_km = _step_displacements_km(gt)
# # # #     s_pred = v_pred_km.norm(dim=-1)
# # # #     s_gt = v_gt_km.norm(dim=-1)
# # # #     l_speed = (s_pred - s_gt).pow(2).mean(0)
# # # #     gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     gt_unit = v_gt_km / gn
# # # #     ate = ((v_pred_km - v_gt_km) * gt_unit).sum(-1)
# # # #     l_ate = ate.pow(2).mean(0)
# # # #     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     cos_sim = ((v_pred_km / pn) * gt_unit).sum(-1)
# # # #     l_dir = F.relu(-cos_sim).pow(2).mean(0) * STEP_KM ** 2
# # # #     return (l_speed + 0.5 * l_ate + 0.3 * l_dir) / (STEP_KM ** 2)


# # # # def disp_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 2:
# # # #         return pred.new_zeros(pred.shape[1])
# # # #     pd = _step_displacements_km(pred).norm(dim=-1).mean(0)
# # # #     gd = _step_displacements_km(gt).norm(dim=-1).mean(0)
# # # #     return (pd - gd).pow(2) / (STEP_KM ** 2)


# # # # def step_dir_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 2:
# # # #         return pred.new_zeros(pred.shape[1])
# # # #     v_pred_km = _step_displacements_km(pred)
# # # #     v_gt_km = _step_displacements_km(gt)
# # # #     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     cos_sim = ((v_pred_km / pn) * (v_gt_km / gn)).sum(-1)
# # # #     return (1.0 - cos_sim).mean(0)


# # # # def heading_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 2:
# # # #         return pred.new_zeros(())
# # # #     pv = _step_displacements_km(pred)
# # # #     gv = _step_displacements_km(gt)
# # # #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1))
# # # #     wrong_dir_loss = (wrong_dir ** 2).mean()

# # # #     if pred.shape[0] >= 3:
# # # #         def _curv(v):
# # # #             cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
# # # #             n1 = v[1:].norm(dim=-1).clamp(min=1e-4)
# # # #             n2 = v[:-1].norm(dim=-1).clamp(min=1e-4)
# # # #             return (cross / (n1 * n2)).clamp(-10.0, 10.0)
# # # #         curv_mse = F.mse_loss(_curv(pv), _curv(gv))
# # # #     else:
# # # #         curv_mse = pred.new_zeros(())
# # # #     return wrong_dir_loss + curv_mse


# # # # def smooth_loss(pred: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 3:
# # # #         return pred.new_zeros(())
# # # #     v_km = _step_displacements_km(pred)
# # # #     if v_km.shape[0] < 2:
# # # #         return pred.new_zeros(())
# # # #     accel_km = v_km[1:] - v_km[:-1]
# # # #     return accel_km.pow(2).mean() / (STEP_KM ** 2)


# # # # def acceleration_loss(pred: torch.Tensor) -> torch.Tensor:
# # # #     return smooth_loss(pred)


# # # # def jerk_loss(pred: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 4:
# # # #         return pred.new_zeros(())
# # # #     v_km = _step_displacements_km(pred)
# # # #     if v_km.shape[0] < 3:
# # # #         return pred.new_zeros(())
# # # #     a_km = v_km[1:] - v_km[:-1]
# # # #     j_km = a_km[1:] - a_km[:-1]
# # # #     return j_km.pow(2).mean() / (STEP_KM ** 2)


# # # # def overall_dir_loss(pred: torch.Tensor, gt: torch.Tensor,
# # # #                      ref: torch.Tensor) -> torch.Tensor:
# # # #     p_d = pred[-1] - ref
# # # #     g_d = gt[-1] - ref
# # # #     lat_ref = ref[:, 1]
# # # #     cos_lat = torch.cos(torch.deg2rad(lat_ref)).clamp(min=1e-4)
# # # #     p_d_km = p_d.clone(); p_d_km[:, 0] *= cos_lat * DEG_TO_KM; p_d_km[:, 1] *= DEG_TO_KM
# # # #     g_d_km = g_d.clone(); g_d_km[:, 0] *= cos_lat * DEG_TO_KM; g_d_km[:, 1] *= DEG_TO_KM
# # # #     pn = p_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     gn = g_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     return (1.0 - ((p_d_km / pn) * (g_d_km / gn)).sum(-1)).mean()


# # # # def recurvature_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 3:
# # # #         return pred.new_zeros(())
# # # #     pred_v = pred[1:] - pred[:-1]
# # # #     gt_v = gt[1:] - gt[:-1]
# # # #     gt_cross = (gt_v[:-1, :, 0] * gt_v[1:, :, 1]
# # # #                 - gt_v[:-1, :, 1] * gt_v[1:, :, 0])
# # # #     if gt_cross.shape[0] < 2:
# # # #         return pred.new_zeros(())
# # # #     sign_change = (gt_cross[:-1] * gt_cross[1:]) < 0
# # # #     if not sign_change.any():
# # # #         return pred.new_zeros(())
# # # #     pred_v_mid = pred_v[1:-1]
# # # #     gt_v_mid = gt_v[1:-1]
# # # #     if pred_v_mid.shape[0] > sign_change.shape[0]:
# # # #         pred_v_mid = pred_v_mid[:sign_change.shape[0]]
# # # #         gt_v_mid = gt_v_mid[:sign_change.shape[0]]
# # # #     pn = pred_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     gn = gt_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     cos_sim = ((pred_v_mid / pn) * (gt_v_mid / gn)).sum(-1)
# # # #     dir_loss = (1.0 - cos_sim)
# # # #     mask = sign_change.float()
# # # #     if mask.sum() < 1:
# # # #         return pred.new_zeros(())
# # # #     return (dir_loss * mask).sum() / mask.sum().clamp(min=1)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  PINN Components (simplified - giữ cấu trúc cũ)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def _get_uv500_ms(env_data: dict, key_mean: str, key_center: str,
# # # #                   T_tgt: int, B: int, device: torch.device) -> torch.Tensor:
# # # #     def _extract(key):
# # # #         x = env_data.get(key, None)
# # # #         if x is None or not torch.is_tensor(x):
# # # #             return None
# # # #         x = x.to(device).float()
# # # #         if x.dim() == 3:
# # # #             x = x[..., 0]
# # # #         elif x.dim() == 1:
# # # #             x = x.unsqueeze(0).expand(B, -1)
# # # #         x = x.permute(1, 0)
# # # #         T_obs = x.shape[0]
# # # #         if T_obs >= T_tgt:
# # # #             return x[:T_tgt] * _UV500_NORM
# # # #         pad = torch.zeros(T_tgt - T_obs, B, device=device)
# # # #         return torch.cat([x * _UV500_NORM, pad], dim=0)

# # # #     val = _extract(key_center)
# # # #     if val is not None:
# # # #         return val
# # # #     val = _extract(key_mean)
# # # #     if val is not None:
# # # #         return val
# # # #     return torch.zeros(T_tgt, B, device=device)


# # # # def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
# # # #     T, B, _ = pred_abs_deg.shape
# # # #     if T < 3:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     DT = DT_6H
# # # #     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
# # # #     dlat = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # #     cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

# # # #     u = dlat[:, :, 0] * cos_lat * 111000.0 / DT
# # # #     v = dlat[:, :, 1] * 111000.0 / DT

# # # #     if u.shape[0] < 2:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     du = (u[1:] - u[:-1]) / DT
# # # #     dv = (v[1:] - v[:-1]) / DT

# # # #     f = 2 * OMEGA * torch.sin(lat_rad[1:-1])
# # # #     beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH

# # # #     res_u = du - f * v[1:]
# # # #     res_v = dv + f * u[1:]

# # # #     R_tc = 3e5
# # # #     v_beta_x = -beta * R_tc ** 2 / 2
# # # #     res_u_corrected = res_u - v_beta_x

# # # #     scale = 0.1
# # # #     loss = ((res_u_corrected / scale).pow(2) + (res_v / scale).pow(2))
# # # #     return loss


# # # # def pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # # #     if pred_deg.shape[0] < 2:
# # # #         return pred_deg.new_zeros(())
# # # #     dt_deg = pred_deg[1:] - pred_deg[:-1]
# # # #     lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # #     dx_km = dt_deg[:, :, 0] * cos_lat * DEG_TO_KM
# # # #     dy_km = dt_deg[:, :, 1] * DEG_TO_KM
# # # #     speed = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # # #     return F.relu(speed - 600.0).pow(2)


# # # # def pinn_pressure_wind_loss(
# # # #     pred_abs_deg: torch.Tensor,
# # # #     vmax_pred: Optional[torch.Tensor],
# # # #     pmin_pred: Optional[torch.Tensor],
# # # #     r34_km: Optional[torch.Tensor] = None,
# # # #     epoch: int = 0,
# # # # ) -> torch.Tensor:
# # # #     if epoch < 30:
# # # #         return pred_abs_deg.new_zeros(())
# # # #     if vmax_pred is None or pmin_pred is None:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     T, B, _ = pred_abs_deg.shape
# # # #     if vmax_pred.shape[0] != T or pmin_pred.shape[0] != T:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
# # # #     f_k = 2 * OMEGA * torch.sin(lat_rad).abs().clamp(min=1e-6)

# # # #     if r34_km is not None:
# # # #         R_tc = (2.0 * r34_km * 1000.0).clamp(min=1e5, max=8e5)
# # # #     else:
# # # #         R_tc = 3e5 + 1e3 * F.relu(vmax_pred - 30.0)

# # # #     V = vmax_pred.clamp(min=1.0)
# # # #     dp = (P_ENV - pmin_pred) * 100.0

# # # #     dp_pred = RHO_AIR * (V.pow(2) / 2.0 + f_k * R_tc * V / 2.0)

# # # #     residual = (dp - dp_pred) / 500.0
# # # #     return residual.pow(2)


# # # # def pinn_bve_loss(
# # # #     pred_abs_deg: torch.Tensor,
# # # #     batch_list,
# # # #     env_data: Optional[dict] = None,
# # # #     epoch: int = 0,
# # # #     gt_abs_deg: Optional[torch.Tensor] = None,
# # # #     vmax_pred: Optional[torch.Tensor] = None,
# # # #     pmin_pred: Optional[torch.Tensor] = None,
# # # #     r34_km: Optional[torch.Tensor] = None,
# # # # ) -> torch.Tensor:
# # # #     """
# # # #     FIX-L-NEW-D: PINN với time weights tập trung 48h-72h
# # # #     """
# # # #     T = pred_abs_deg.shape[0]
# # # #     if T < 3:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     _env = env_data
# # # #     if _env is None and batch_list is not None:
# # # #         try:
# # # #             _env = batch_list[13]
# # # #         except:
# # # #             _env = None

# # # #     # FIX-L-NEW-D: Time weights cho PINN - tập trung vào long-range
# # # #     pinn_time_w = torch.zeros(T, device=pred_abs_deg.device)
# # # #     for i in range(T):
# # # #         if i < 4:       # 6h-24h: SR handles
# # # #             pinn_time_w[i] = 0.5
# # # #         elif i < 8:     # 30h-48h: transition
# # # #             pinn_time_w[i] = 1.0 + (i - 4) * 0.5
# # # #         else:           # 54h-72h: PINN critical
# # # #             pinn_time_w[i] = 3.0 + (i - 8) * 1.0

# # # #     # Adaptive weighting
# # # #     if gt_abs_deg is not None:
# # # #         with torch.no_grad():
# # # #             d_track = _haversine_deg(pred_abs_deg, gt_abs_deg)
# # # #             w_bve_step = torch.sigmoid(1.0 - d_track / 200.0)
# # # #     else:
# # # #         w_bve_step = torch.ones(T, pred_abs_deg.shape[1], device=pred_abs_deg.device)

# # # #     def apply_w(l_map, weight_scalar):
# # # #         if l_map.dim() == 0:
# # # #             return l_map * weight_scalar
# # # #         t_size = l_map.shape[0]
# # # #         w_final = weight_scalar * pinn_time_w[-t_size:, None]
# # # #         return (l_map * w_final).mean()

# # # #     l_sw = apply_w(pinn_shallow_water(pred_abs_deg), 1.0)
# # # #     l_speed = apply_w(pinn_speed_constraint(pred_abs_deg), 0.1)
# # # #     l_pwr = apply_w(pinn_pressure_wind_loss(pred_abs_deg, vmax_pred, pmin_pred, r34_km, epoch), 0.6)

# # # #     total = l_sw.mean() + l_speed.mean() + l_pwr.mean()

# # # #     w_bnd = _boundary_weights(pred_abs_deg).mean()
# # # #     f_lazy = 0.2 if epoch < 30 else (0.5 if epoch < 50 else 1.0)

# # # #     total_clamped = adapt_clamp(total, epoch, max_val=20.0)
# # # #     return total_clamped * w_bnd * f_lazy


# # # # def fm_physics_consistency_loss(
# # # #     pred_samples: torch.Tensor,
# # # #     gt_norm: torch.Tensor,
# # # #     last_pos: torch.Tensor,
# # # # ) -> torch.Tensor:
# # # #     S, T, B = pred_samples.shape[:3]

# # # #     last_lon = (last_pos[:, 0] * 50.0 + 1800.0) / 10.0
# # # #     last_lat = (last_pos[:, 1] * 50.0) / 10.0
# # # #     lat_rad = torch.deg2rad(last_lat)
# # # #     beta = 2 * OMEGA * torch.cos(lat_rad) / R_EARTH
# # # #     R_tc = 3e5

# # # #     v_beta_lon = -beta * R_tc ** 2 / 2
# # # #     v_beta_lat = beta * R_tc ** 2 / 4

# # # #     beta_dir = torch.stack([v_beta_lon, v_beta_lat], dim=-1)
# # # #     beta_norm = beta_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     beta_dir_unit = beta_dir / beta_norm

# # # #     pos_step1 = pred_samples[:, 0, :, :2]
# # # #     dir_step1 = pos_step1 - last_pos.unsqueeze(0)
# # # #     dir_norm = dir_step1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     dir_unit = dir_step1 / dir_norm
# # # #     mean_dir = dir_unit.mean(0)
# # # #     mean_norm = mean_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     mean_dir_unit = mean_dir / mean_norm

# # # #     cos_align = (mean_dir_unit * beta_dir_unit).sum(-1)
# # # #     beta_strength = beta_norm.squeeze(-1)
# # # #     penalise_mask = (beta_strength > 1.0).float()
# # # #     direction_loss = F.relu(-cos_align) * penalise_mask
# # # #     return direction_loss.mean() * 0.5


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Main loss
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def compute_total_loss(
# # # #     pred_abs,
# # # #     gt,
# # # #     ref,
# # # #     batch_list,
# # # #     pred_samples=None,
# # # #     gt_norm=None,
# # # #     weights=WEIGHTS,
# # # #     intensity_w: Optional[torch.Tensor] = None,
# # # #     env_data: Optional[dict] = None,
# # # #     step_weight_alpha: float = 0.0,
# # # #     all_trajs: Optional[torch.Tensor] = None,
# # # #     epoch: int = 0,
# # # #     gt_abs_deg: Optional[torch.Tensor] = None,
# # # #     vmax_pred: Optional[torch.Tensor] = None,
# # # #     pmin_pred: Optional[torch.Tensor] = None,
# # # #     r34_km: Optional[torch.Tensor] = None,
# # # #     sr_pred: Optional[torch.Tensor] = None,
# # # # ) -> Dict:
# # # #     NRM = 35.0

# # # #     recurv_w = _recurvature_weights(gt, w_recurv=2.5)
# # # #     sample_w = recurv_w * (intensity_w.to(gt.device) if intensity_w is not None else 1.0)
# # # #     sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

# # # #     # FM AFCRPS với time weights mới
# # # #     if pred_samples is not None:
# # # #         target = gt_norm if gt_norm is not None else gt
# # # #         unit = gt_norm is not None
# # # #         l_fm = fm_afcrps_loss(
# # # #             pred_samples, target,
# # # #             unit_01deg=unit,
# # # #             intensity_w=sample_w,
# # # #             step_weight_alpha=step_weight_alpha,
# # # #             w_es=0.3,
# # # #             epoch=epoch,
# # # #         )
# # # #     else:
# # # #         l_fm = _haversine_deg(pred_abs, gt).mean()

# # # #     # Directional losses
# # # #     l_vel = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
# # # #     l_disp = (disp_loss_per_sample(pred_abs, gt) * sample_w).mean()
# # # #     l_step = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()
# # # #     l_heading = heading_loss(pred_abs, gt)
# # # #     l_recurv = recurvature_loss(pred_abs, gt)
# # # #     l_dir_final = overall_dir_loss(pred_abs, gt, ref)
# # # #     l_smooth = smooth_loss(pred_abs)
# # # #     l_accel = acceleration_loss(pred_abs)
# # # #     l_jerk = jerk_loss(pred_abs)

# # # #     # PINN
# # # #     _env = env_data
# # # #     if _env is None and batch_list is not None:
# # # #         try:
# # # #             _env = batch_list[13]
# # # #         except:
# # # #             _env = None

# # # #     l_pinn = pinn_bve_loss(
# # # #         pred_abs, batch_list, env_data=_env, epoch=epoch,
# # # #         gt_abs_deg=gt_abs_deg, vmax_pred=vmax_pred,
# # # #         pmin_pred=pmin_pred, r34_km=r34_km
# # # #     )

# # # #     # Spread loss với threshold mới
# # # #     l_spread = pred_abs.new_zeros(())
# # # #     if all_trajs is not None and all_trajs.shape[0] >= 2:
# # # #         n_sr = 4
# # # #         trajs_fm = all_trajs[:, n_sr:] if all_trajs.shape[1] > n_sr else all_trajs
# # # #         l_spread = ensemble_spread_loss(trajs_fm, epoch)

# # # #     # Bridge loss mở rộng
# # # #     l_bridge = pred_abs.new_zeros(())
# # # #     if sr_pred is not None and pred_samples is not None:
# # # #         fm_mean = pred_samples.mean(0)
# # # #         l_bridge = bridge_loss(sr_pred, fm_mean, epoch)

# # # #     # SR-FM velocity consistency (NEW)
# # # #     l_sr_fm_vel = pred_abs.new_zeros(())
# # # #     if sr_pred is not None and pred_samples is not None:
# # # #         fm_mean = pred_samples.mean(0)
# # # #         l_sr_fm_vel = sr_fm_velocity_consistency_loss(sr_pred, fm_mean)

# # # #     # Total loss
# # # #     total = (
# # # #         weights.get("fm", 3.0) * l_fm
# # # #         + weights.get("velocity", 1.0) * l_vel * NRM
# # # #         + weights.get("disp", 0.5) * l_disp * NRM
# # # #         + weights.get("step", 0.5) * l_step * NRM
# # # #         + weights.get("heading", 2.0) * l_heading * NRM
# # # #         + weights.get("recurv", 1.5) * l_recurv * NRM
# # # #         + weights.get("dir", 1.0) * l_dir_final * NRM
# # # #         + weights.get("smooth", 0.5) * l_smooth * NRM
# # # #         + weights.get("accel", 0.8) * l_accel * NRM
# # # #         + weights.get("jerk", 0.3) * l_jerk * NRM
# # # #         + weights.get("pinn", 1.0) * l_pinn
# # # #         + weights.get("spread", 1.5) * l_spread * NRM
# # # #         + weights.get("bridge", 2.0) * l_bridge * NRM
# # # #         + weights.get("sr_fm_vel", 1.0) * l_sr_fm_vel * NRM
# # # #     ) / NRM

# # # #     if torch.isnan(total) or torch.isinf(total):
# # # #         total = pred_abs.new_zeros(())

# # # #     return dict(
# # # #         total=total,
# # # #         fm=l_fm.item(),
# # # #         velocity=l_vel.item() * NRM,
# # # #         step=l_step.item(),
# # # #         disp=l_disp.item() * NRM,
# # # #         heading=l_heading.item(),
# # # #         recurv=l_recurv.item(),
# # # #         smooth=l_smooth.item() * NRM,
# # # #         accel=l_accel.item() * NRM,
# # # #         jerk=l_jerk.item() * NRM,
# # # #         pinn=l_pinn.item(),
# # # #         spread=l_spread.item() * NRM,
# # # #         bridge=l_bridge.item() * NRM,
# # # #         sr_fm_vel=l_sr_fm_vel.item() * NRM,
# # # #         recurv_ratio=(_recurvature_weights(gt) > 1.0).float().mean().item(),
# # # #     )

# # # # """
# # # # Model/losses.py  ── v28 - BALANCED
# # # # ===================================
# # # # MỤC TIÊU: 12h < 50km, 24h < 100km, 72h < 300km

# # # # NGUYÊN TẮC CÂN BẰNG:
# # # #   1. KHÔNG thay đổi SR loss - giữ 12h/24h tốt
# # # #   2. Tăng FM time weights cho 48h-72h NHƯNG không quá mạnh
# # # #   3. Bridge loss ĐƠN GIẢN - chỉ enforce position, không velocity phức tạp
# # # #   4. Spread loss RELAX hơn để FM có thể explore

# # # # KEY CHANGES từ v26:
# # # #   - FM time weights: 0.5→3.0 (linear) → 0.5→4.0 (quadratic focus 72h)
# # # #   - Bridge: weight 0.5, chỉ position matching, normalize /150km
# # # #   - Spread: relax threshold 60→200km (cho FM explore)
# # # #   - PINN: giữ nguyên, không thay đổi
# # # # """
# # # # from __future__ import annotations

# # # # import math
# # # # from typing import Dict, Optional, Tuple

# # # # import torch
# # # # import torch.nn.functional as F

# # # # __all__ = [
# # # #     "WEIGHTS", "compute_total_loss", "fm_afcrps_loss",
# # # #     "fm_physics_consistency_loss", "pinn_bve_loss",
# # # #     "recurvature_loss", "velocity_loss_per_sample",
# # # #     "short_range_regression_loss", "bridge_loss",
# # # #     "adapt_clamp", "pinn_pressure_wind_loss",
# # # # ]

# # # # # ── Constants ─────────────────────────────────────────────────────────────────
# # # # OMEGA        = 7.2921e-5
# # # # R_EARTH      = 6.371e6
# # # # DT_6H        = 6 * 3600
# # # # DEG_TO_KM    = 111.0
# # # # STEP_KM      = 113.0
# # # # P_ENV        = 1013.0
# # # # RHO_AIR      = 1.15

# # # # _ERA5_LAT_MIN =   0.0
# # # # _ERA5_LAT_MAX =  40.0
# # # # _ERA5_LON_MIN = 100.0
# # # # _ERA5_LON_MAX = 160.0

# # # # _UV500_NORM      = 30.0
# # # # _GPH500_MEAN_M   = 5870.0
# # # # _GPH500_STD_M    = 80.0
# # # # _STEERING_MIN_MS = 3.0
# # # # _PINN_SCALE      = 1e-2

# # # # # ── Weights (GIỮ NGUYÊN từ v26 - đã hoạt động) ────────────────────────────────
# # # # WEIGHTS: Dict[str, float] = dict(
# # # #     fm          = 2.0,      # GIỮ NGUYÊN
# # # #     velocity    = 0.8,      # GIỮ NGUYÊN
# # # #     heading     = 2.0,
# # # #     recurv      = 1.5,
# # # #     step        = 0.5,
# # # #     disp        = 0.5,
# # # #     dir         = 1.0,
# # # #     smooth      = 0.5,
# # # #     accel       = 0.8,
# # # #     jerk        = 0.3,
# # # #     pinn        = 0.5,      # GIỮ NGUYÊN
# # # #     fm_physics  = 0.3,
# # # #     spread      = 0.5,      # GIẢM từ 0.8 - cho FM explore
# # # #     short_range = 5.0,      # GIỮ NGUYÊN - quan trọng cho 12h/24h
# # # #     bridge      = 0.5,      # GIỮ NGUYÊN weight, sửa function
# # # # )

# # # # RECURV_ANGLE_THR = 45.0
# # # # RECURV_WEIGHT    = 2.5

# # # # _SR_N_STEPS  = 4
# # # # _SR_WEIGHTS  = [2.0, 4.0, 2.0, 2.0]  # Focus vào 12h (step 2)
# # # # _HUBER_DELTA = 50.0


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Haversine utilities (GIỮ NGUYÊN)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def _haversine(p1: torch.Tensor, p2: torch.Tensor,
# # # #                unit_01deg: bool = True) -> torch.Tensor:
# # # #     if unit_01deg:
# # # #         lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
# # # #         lat1 = (p1[..., 1] * 50.0) / 10.0
# # # #         lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
# # # #         lat2 = (p2[..., 1] * 50.0) / 10.0
# # # #     else:
# # # #         lon1, lat1 = p1[..., 0], p1[..., 1]
# # # #         lon2, lat2 = p2[..., 0], p2[..., 1]

# # # #     lat1r = torch.deg2rad(lat1); lat2r = torch.deg2rad(lat2)
# # # #     dlon  = torch.deg2rad(lon2 - lon1)
# # # #     dlat  = torch.deg2rad(lat2 - lat1)
# # # #     a = (torch.sin(dlat / 2).pow(2)
# # # #          + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2).pow(2))
# # # #     a = a.clamp(1e-12, 1.0 - 1e-12)
# # # #     return 2.0 * 6371.0 * torch.asin(a.sqrt())


# # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # #     return _haversine(p1, p2, unit_01deg=False)


# # # # def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
# # # #     out = arr.clone()
# # # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# # # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# # # #     return out


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  AdaptClamp (GIỮ NGUYÊN)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def adapt_clamp(x: torch.Tensor, epoch: int, max_val: float = 20.0) -> torch.Tensor:
# # # #     delta = max_val

# # # #     def huber_clamp(v: torch.Tensor) -> torch.Tensor:
# # # #         return torch.where(
# # # #             v <= delta,
# # # #             v.pow(2) / (2.0 * delta),
# # # #             v - delta / 2.0,
# # # #         )

# # # #     def tanh_clamp(v: torch.Tensor) -> torch.Tensor:
# # # #         return delta * torch.tanh(v / delta)

# # # #     if epoch < 10:
# # # #         return huber_clamp(x)
# # # #     elif epoch < 20:
# # # #         beta = (epoch - 10) / 10.0
# # # #         return (1.0 - beta) * huber_clamp(x) + beta * tanh_clamp(x)
# # # #     else:
# # # #         return tanh_clamp(x)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  ★ KEY CHANGE: FM AFCRPS với Time Weights CÂN BẰNG
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def fm_afcrps_loss(
# # # #     pred_samples:      torch.Tensor,
# # # #     gt:                torch.Tensor,
# # # #     unit_01deg:        bool  = True,
# # # #     intensity_w:       Optional[torch.Tensor] = None,
# # # #     step_weight_alpha: float = 0.0,
# # # #     w_es:              float = 0.3,
# # # # ) -> torch.Tensor:
# # # #     """
# # # #     ★ KEY CHANGE: Time weights CÂN BẰNG
    
# # # #     Mục tiêu:
# # # #     - Giữ 12h/24h accuracy (SR handle) → weight thấp cho step 1-4
# # # #     - Tăng 72h accuracy (FM handle) → weight cao cho step 9-12
    
# # # #     NHƯNG không quá aggressive để tránh destabilize training
    
# # # #     Weight scheme (12 steps = 72h):
# # # #         Step 1-4 (6h-24h):  0.5, 0.5, 0.7, 0.7   (SR handles)
# # # #         Step 5-8 (30h-48h): 1.0, 1.3, 1.6, 2.0   (transition)
# # # #         Step 9-12 (54h-72h): 2.5, 3.0, 3.5, 4.0  (FM critical)
    
# # # #     So với v26 (0.5 → 3.0 linear), đây tăng 72h weight lên 4.0
# # # #     nhưng không quá mạnh như v27 (8.5)
# # # #     """
# # # #     M, T, B, _ = pred_samples.shape
# # # #     device = pred_samples.device

# # # #     # ★ Time weights quadratic focus on 72h
# # # #     base_w = torch.zeros(T, device=device)
# # # #     for i in range(T):
# # # #         if i < 4:       # 6h-24h: SR handles
# # # #             base_w[i] = 0.5 + (i % 2) * 0.2  # [0.5, 0.7, 0.5, 0.7]
# # # #         elif i < 8:     # 30h-48h: transition
# # # #             base_w[i] = 1.0 + (i - 4) * 0.35  # [1.0, 1.35, 1.7, 2.05]
# # # #         else:           # 54h-72h: FM critical
# # # #             base_w[i] = 2.5 + (i - 8) * 0.5   # [2.5, 3.0, 3.5, 4.0]

# # # #     if step_weight_alpha > 0.0:
# # # #         early_w = torch.exp(
# # # #             -torch.arange(T, dtype=torch.float, device=device) * 0.5
# # # #         )
# # # #         early_w = early_w / early_w.mean()
# # # #         time_w = (1.0 - step_weight_alpha) * base_w + step_weight_alpha * early_w
# # # #     else:
# # # #         time_w = base_w

# # # #     if M == 1:
# # # #         dist = _haversine(pred_samples[0], gt, unit_01deg)
# # # #         loss_per_b = (dist * time_w.unsqueeze(1)).mean(0)
# # # #         es_term = pred_samples.new_zeros(())
# # # #     else:
# # # #         # Accuracy: E[d(X^m, Y)]
# # # #         d_to_gt = _haversine(
# # # #             pred_samples,
# # # #             gt.unsqueeze(0).expand_as(pred_samples),
# # # #             unit_01deg,
# # # #         )
# # # #         d_to_gt_w    = d_to_gt * time_w.unsqueeze(0).unsqueeze(2)
# # # #         e_sy         = d_to_gt_w.mean(1).mean(0)

# # # #         # Sharpness: E[d(X^m, X^m')]
# # # #         ps_i = pred_samples.unsqueeze(1)
# # # #         ps_j = pred_samples.unsqueeze(0)
# # # #         d_pair = _haversine(
# # # #             ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # # #             ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # # #             unit_01deg,
# # # #         ).reshape(M, M, T, B)

# # # #         diag_mask = torch.eye(M, device=device, dtype=torch.bool)
# # # #         diag_mask = diag_mask.view(M, M, 1, 1).expand_as(d_pair)
# # # #         d_pair = d_pair.masked_fill(diag_mask, 0.0)

# # # #         d_pair_w    = d_pair * time_w.unsqueeze(0).unsqueeze(0).unsqueeze(3)
# # # #         e_ssp       = d_pair_w.mean(2).sum(0).sum(0) / (M * (M - 1))

# # # #         loss_per_b = e_sy - 0.5 * e_ssp

# # # #         # Energy Score term
# # # #         if w_es > 0.0 and M > 1:
# # # #             ps_flat  = pred_samples.reshape(M, T * B, 2)
# # # #             gt_flat  = gt.reshape(T * B, 2)
# # # #             mean_pred = ps_flat.mean(0)

# # # #             es_acc = (mean_pred - gt_flat).pow(2).sum(-1).sqrt().mean()

# # # #             ps_i_f = ps_flat.unsqueeze(1)
# # # #             ps_j_f = ps_flat.unsqueeze(0)
# # # #             d_pair_f = (ps_i_f - ps_j_f).pow(2).sum(-1).sqrt()
# # # #             diag_f = torch.eye(M, device=device, dtype=torch.bool)
# # # #             diag_f = diag_f.view(M, M, 1).expand_as(d_pair_f)
# # # #             d_pair_f = d_pair_f.masked_fill(diag_f, 0.0)
# # # #             es_sharp = (M - 1) / M * d_pair_f.sum(0).sum(0).mean() / (M * (M - 1))

# # # #             es_term = es_acc - es_sharp
# # # #         else:
# # # #             es_term = pred_samples.new_zeros(())

# # # #     if intensity_w is not None:
# # # #         w = intensity_w.to(loss_per_b.device)
# # # #         crps_loss = (loss_per_b * w).mean()
# # # #     else:
# # # #         crps_loss = loss_per_b.mean()

# # # #     return crps_loss + w_es * es_term


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Short-range Loss (GIỮ NGUYÊN - quan trọng cho 12h/24h)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def short_range_regression_loss(
# # # #     pred_sr:  torch.Tensor,
# # # #     gt_sr:    torch.Tensor,
# # # #     last_pos: torch.Tensor,
# # # # ) -> torch.Tensor:
# # # #     n_steps = min(pred_sr.shape[0], gt_sr.shape[0], _SR_N_STEPS)
# # # #     if n_steps == 0:
# # # #         return pred_sr.new_zeros(())

# # # #     pred_deg = _norm_to_deg(pred_sr[:n_steps])
# # # #     gt_deg   = _norm_to_deg(gt_sr[:n_steps])

# # # #     dist_km = _haversine_deg(pred_deg, gt_deg)

# # # #     huber = torch.where(
# # # #         dist_km < _HUBER_DELTA,
# # # #         0.5 * dist_km.pow(2) / _HUBER_DELTA,
# # # #         dist_km - 0.5 * _HUBER_DELTA,
# # # #     )

# # # #     w = pred_sr.new_tensor(_SR_WEIGHTS[:n_steps])
# # # #     w = w / w.sum()
# # # #     return (huber * w.view(-1, 1)).mean()


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  ★ KEY CHANGE: Bridge Loss ĐƠN GIẢN HƠN
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def bridge_loss(
# # # #     sr_pred:  torch.Tensor,
# # # #     fm_mean:  torch.Tensor,
# # # # ) -> torch.Tensor:
# # # #     """
# # # #     ★ SIMPLIFIED Bridge Loss
    
# # # #     Vấn đề v27: bridge loss quá phức tạp (250-1000+) destabilize training
    
# # # #     Solution: Chỉ enforce position matching ở step 4
# # # #     - Normalize bằng 150km (không phải 50km hay 100km)
# # # #     - Không thêm velocity matching (gây instability)
    
# # # #     Mục tiêu: SR step 4 ≈ FM step 4 (cách nhau < 100km)
# # # #     """
# # # #     if sr_pred.shape[0] < 4 or fm_mean.shape[0] < 5:
# # # #         return sr_pred.new_zeros(())

# # # #     # Position match at step 4 (24h)
# # # #     pos_sr4 = _norm_to_deg(sr_pred[3])
# # # #     pos_fm4 = _norm_to_deg(fm_mean[3])

# # # #     dist_pos = _haversine_deg(pos_sr4, pos_fm4)
    
# # # #     # Normalize by 150km - typical TC movement in 6h
# # # #     # Nếu SR và FM cách nhau < 150km → loss < 1
# # # #     l_pos = (dist_pos / 150.0).pow(2).mean()

# # # #     return l_pos


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  ★ KEY CHANGE: Spread Loss RELAX hơn
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def ensemble_spread_loss(all_trajs: torch.Tensor) -> torch.Tensor:
# # # #     """
# # # #     ★ RELAXED Spread Loss
    
# # # #     Vấn đề: Spread quá strict → FM không explore được → 72h bad
    
# # # #     Solution: Relax threshold để FM có thể explore
# # # #     - 6h:  max 60km spread (was 40km)
# # # #     - 72h: max 200km spread (was 120km)
    
# # # #     Đây là trade-off: spread cao hơn nhưng 72h accuracy tốt hơn
# # # #     """
# # # #     if all_trajs.shape[0] < 2:
# # # #         return all_trajs.new_zeros(())

# # # #     S, T, B, _ = all_trajs.shape
# # # #     device = all_trajs.device

# # # #     # RELAXED thresholds
# # # #     max_spreads = torch.linspace(60.0, 200.0, T, device=device)
    
# # # #     # Weight nhẹ hơn ở early steps
# # # #     step_weights = torch.linspace(0.5, 2.0, T, device=device)

# # # #     total_loss = all_trajs.new_zeros(())
# # # #     for t in range(T):
# # # #         step_trajs = all_trajs[:, t, :, :2]
# # # #         std_lon    = step_trajs[:, :, 0].std(0)
# # # #         std_lat    = step_trajs[:, :, 1].std(0)
# # # #         spread_km  = torch.sqrt(std_lon ** 2 + std_lat ** 2) * 500.0

# # # #         excess = F.relu(spread_km - max_spreads[t])
        
# # # #         # Gentler penalty
# # # #         loss = (excess / 50.0).pow(2)

# # # #         total_loss = total_loss + step_weights[t] * loss.mean()

# # # #     return total_loss / T


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Helper functions (GIỮ NGUYÊN)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def _boundary_weights(traj_deg: torch.Tensor) -> torch.Tensor:
# # # #     lon = traj_deg[..., 0]
# # # #     lat = traj_deg[..., 1]

# # # #     d_lat_lo = (lat - _ERA5_LAT_MIN) / 5.0
# # # #     d_lat_hi = (_ERA5_LAT_MAX - lat) / 5.0
# # # #     d_lon_lo = (lon - _ERA5_LON_MIN) / 5.0
# # # #     d_lon_hi = (_ERA5_LON_MAX - lon) / 5.0

# # # #     d_bnd = torch.stack([d_lat_lo, d_lat_hi, d_lon_lo, d_lon_hi], dim=-1).min(dim=-1).values
# # # #     return torch.sigmoid(d_bnd - 0.5)


# # # # def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
# # # #     dt      = traj_deg[1:] - traj_deg[:-1]
# # # #     lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
# # # #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
# # # #     dt_km   = dt.clone()
# # # #     dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
# # # #     dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
# # # #     return dt_km


# # # # def _total_rotation_angle_batch(gt: torch.Tensor) -> torch.Tensor:
# # # #     T, B, _ = gt.shape
# # # #     if T < 3:
# # # #         return gt.new_zeros(B)
# # # #     lats_rad = torch.deg2rad(gt[:, :, 1])
# # # #     cos_lat  = torch.cos(lats_rad[:-1])
# # # #     dlat = gt[1:, :, 1] - gt[:-1, :, 1]
# # # #     dlon = (gt[1:, :, 0] - gt[:-1, :, 0]) * cos_lat
# # # #     v    = torch.stack([dlon, dlat], dim=-1)
# # # #     v1   = v[:-1]; v2 = v[1:]
# # # #     n1   = v1.norm(dim=-1).clamp(min=1e-8)
# # # #     n2   = v2.norm(dim=-1).clamp(min=1e-8)
# # # #     cos_a = (v1 * v2).sum(-1) / (n1 * n2)
# # # #     return torch.rad2deg(torch.acos(cos_a.clamp(-1.0, 1.0))).sum(0)


# # # # def _recurvature_weights(gt: torch.Tensor,
# # # #                          thr: float = RECURV_ANGLE_THR,
# # # #                          w_recurv: float = RECURV_WEIGHT) -> torch.Tensor:
# # # #     rot = _total_rotation_angle_batch(gt)
# # # #     return torch.where(rot >= thr,
# # # #                        torch.full_like(rot, w_recurv),
# # # #                        torch.ones_like(rot))


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Directional losses (GIỮ NGUYÊN)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def velocity_loss_per_sample(pred: torch.Tensor,
# # # #                              gt: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 2:
# # # #         return pred.new_zeros(pred.shape[1])
# # # #     v_pred_km = _step_displacements_km(pred)
# # # #     v_gt_km   = _step_displacements_km(gt)
# # # #     s_pred    = v_pred_km.norm(dim=-1)
# # # #     s_gt      = v_gt_km.norm(dim=-1)
# # # #     l_speed   = (s_pred - s_gt).pow(2).mean(0)
# # # #     gn        = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     gt_unit   = v_gt_km / gn
# # # #     ate       = ((v_pred_km - v_gt_km) * gt_unit).sum(-1)
# # # #     l_ate     = ate.pow(2).mean(0)
# # # #     pn        = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     cos_sim   = ((v_pred_km / pn) * gt_unit).sum(-1)
# # # #     l_dir     = F.relu(-cos_sim).pow(2).mean(0) * STEP_KM ** 2
# # # #     return (l_speed + 0.5 * l_ate + 0.3 * l_dir) / (STEP_KM ** 2)


# # # # def disp_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 2:
# # # #         return pred.new_zeros(pred.shape[1])
# # # #     pd = _step_displacements_km(pred).norm(dim=-1).mean(0)
# # # #     gd = _step_displacements_km(gt).norm(dim=-1).mean(0)
# # # #     return (pd - gd).pow(2) / (STEP_KM ** 2)


# # # # def step_dir_loss_per_sample(pred: torch.Tensor,
# # # #                               gt: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 2:
# # # #         return pred.new_zeros(pred.shape[1])
# # # #     v_pred_km = _step_displacements_km(pred)
# # # #     v_gt_km   = _step_displacements_km(gt)
# # # #     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     cos_sim = ((v_pred_km / pn) * (v_gt_km / gn)).sum(-1)
# # # #     return (1.0 - cos_sim).mean(0)


# # # # def heading_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 2:
# # # #         return pred.new_zeros(())
# # # #     pv = _step_displacements_km(pred)
# # # #     gv = _step_displacements_km(gt)
# # # #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     wrong_dir      = F.relu(-((pv / pn) * (gv / gn)).sum(-1))
# # # #     wrong_dir_loss = (wrong_dir ** 2).mean()

# # # #     if pred.shape[0] >= 3:
# # # #         def _curv(v):
# # # #             cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
# # # #             n1    = v[1:].norm(dim=-1).clamp(min=1e-4)
# # # #             n2    = v[:-1].norm(dim=-1).clamp(min=1e-4)
# # # #             return (cross / (n1 * n2)).clamp(-10.0, 10.0)
# # # #         curv_mse = F.mse_loss(_curv(pv), _curv(gv))
# # # #     else:
# # # #         curv_mse = pred.new_zeros(())
# # # #     return wrong_dir_loss + curv_mse


# # # # def smooth_loss(pred: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 3:
# # # #         return pred.new_zeros(())
# # # #     v_km = _step_displacements_km(pred)
# # # #     if v_km.shape[0] < 2:
# # # #         return pred.new_zeros(())
# # # #     accel_km = v_km[1:] - v_km[:-1]
# # # #     return accel_km.pow(2).mean() / (STEP_KM ** 2)


# # # # def acceleration_loss(pred: torch.Tensor) -> torch.Tensor:
# # # #     return smooth_loss(pred)


# # # # def jerk_loss(pred: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 4:
# # # #         return pred.new_zeros(())
# # # #     v_km = _step_displacements_km(pred)
# # # #     if v_km.shape[0] < 3:
# # # #         return pred.new_zeros(())
# # # #     a_km = v_km[1:] - v_km[:-1]
# # # #     j_km = a_km[1:] - a_km[:-1]
# # # #     return j_km.pow(2).mean() / (STEP_KM ** 2)


# # # # def overall_dir_loss(pred: torch.Tensor, gt: torch.Tensor,
# # # #                      ref: torch.Tensor) -> torch.Tensor:
# # # #     p_d = pred[-1] - ref
# # # #     g_d = gt[-1]   - ref
# # # #     lat_ref = ref[:, 1]
# # # #     cos_lat = torch.cos(torch.deg2rad(lat_ref)).clamp(min=1e-4)
# # # #     p_d_km  = p_d.clone(); p_d_km[:, 0] *= cos_lat * DEG_TO_KM; p_d_km[:, 1] *= DEG_TO_KM
# # # #     g_d_km  = g_d.clone(); g_d_km[:, 0] *= cos_lat * DEG_TO_KM; g_d_km[:, 1] *= DEG_TO_KM
# # # #     pn = p_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     gn = g_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # # #     return (1.0 - ((p_d_km / pn) * (g_d_km / gn)).sum(-1)).mean()


# # # # def recurvature_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # # #     if pred.shape[0] < 3:
# # # #         return pred.new_zeros(())
# # # #     pred_v   = pred[1:] - pred[:-1]
# # # #     gt_v     = gt[1:]   - gt[:-1]
# # # #     gt_cross = (gt_v[:-1, :, 0] * gt_v[1:, :, 1]
# # # #               - gt_v[:-1, :, 1] * gt_v[1:, :, 0])
# # # #     if gt_cross.shape[0] < 2:
# # # #         return pred.new_zeros(())
# # # #     sign_change = (gt_cross[:-1] * gt_cross[1:]) < 0
# # # #     if not sign_change.any():
# # # #         return pred.new_zeros(())
# # # #     pred_v_mid = pred_v[1:-1]
# # # #     gt_v_mid   = gt_v[1:-1]
# # # #     if pred_v_mid.shape[0] > sign_change.shape[0]:
# # # #         pred_v_mid = pred_v_mid[:sign_change.shape[0]]
# # # #         gt_v_mid   = gt_v_mid[:sign_change.shape[0]]
# # # #     pn      = pred_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     gn      = gt_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     cos_sim  = ((pred_v_mid / pn) * (gt_v_mid / gn)).sum(-1)
# # # #     dir_loss = (1.0 - cos_sim)
# # # #     mask     = sign_change.float()
# # # #     if mask.sum() < 1:
# # # #         return pred.new_zeros(())
# # # #     return (dir_loss * mask).sum() / mask.sum().clamp(min=1)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  PINN Components (GIỮ NGUYÊN từ v26)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def _get_uv500_ms(env_data: dict, key_mean: str, key_center: str,
# # # #                   T_tgt: int, B: int, device: torch.device) -> torch.Tensor:
# # # #     def _extract(key):
# # # #         x = env_data.get(key, None)
# # # #         if x is None or not torch.is_tensor(x):
# # # #             return None
# # # #         x = x.to(device).float()
# # # #         if x.dim() == 3:
# # # #             x = x[..., 0]
# # # #         elif x.dim() == 1:
# # # #             x = x.unsqueeze(0).expand(B, -1)
# # # #         x = x.permute(1, 0)
# # # #         T_obs = x.shape[0]
# # # #         if T_obs >= T_tgt:
# # # #             return x[:T_tgt] * _UV500_NORM
# # # #         pad = torch.zeros(T_tgt - T_obs, B, device=device)
# # # #         return torch.cat([x * _UV500_NORM, pad], dim=0)

# # # #     val = _extract(key_center)
# # # #     if val is not None:
# # # #         return val
# # # #     val = _extract(key_mean)
# # # #     if val is not None:
# # # #         return val
# # # #     return torch.zeros(T_tgt, B, device=device)


# # # # def _get_gph500_norm(env_data: dict, key: str,
# # # #                      T_tgt: int, B: int, device: torch.device) -> torch.Tensor:
# # # #     x = env_data.get(key, None)
    
# # # #     if x is None or not torch.is_tensor(x):
# # # #         return torch.zeros(T_tgt, B, device=device)
# # # #     x = x.to(device).float()
# # # #     if x.dim() == 3:
# # # #         x = x[..., 0]
# # # #     x = x.permute(1, 0)
# # # #     T_obs = x.shape[0]
# # # #     if T_obs >= T_tgt:
# # # #         return x[:T_tgt]
# # # #     pad = torch.zeros(T_tgt - T_obs, B, device=device)
# # # #     return torch.cat([x, pad], dim=0)


# # # # def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
# # # #     T, B, _ = pred_abs_deg.shape
# # # #     if T < 3:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     DT      = DT_6H
# # # #     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
# # # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # #     cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

# # # #     u = dlat[:, :, 0] * cos_lat * 111000.0 / DT
# # # #     v = dlat[:, :, 1] * 111000.0 / DT

# # # #     if u.shape[0] < 2:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     du = (u[1:] - u[:-1]) / DT
# # # #     dv = (v[1:] - v[:-1]) / DT

# # # #     f    = 2 * OMEGA * torch.sin(lat_rad[1:-1])
# # # #     beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH

# # # #     res_u = du - f * v[1:]
# # # #     res_v = dv + f * u[1:]

# # # #     R_tc            = 3e5
# # # #     v_beta_x        = -beta * R_tc ** 2 / 2
# # # #     res_u_corrected = res_u - v_beta_x

# # # #     scale = 0.1
# # # #     loss = ((res_u_corrected / scale).pow(2) + (res_v / scale).pow(2))
# # # #     return loss


# # # # def pinn_rankine_steering(pred_abs_deg: torch.Tensor,
# # # #                           env_data: Optional[dict]) -> torch.Tensor:
# # # #     if env_data is None:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     T, B, _ = pred_abs_deg.shape
# # # #     if T < 2:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     device  = pred_abs_deg.device
# # # #     T_tgt   = T - 1
# # # #     DT      = DT_6H

# # # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # #     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
# # # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

# # # #     u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT
# # # #     v_tc = dlat[:, :, 1] * 111000.0 / DT

# # # #     u500 = _get_uv500_ms(env_data, "u500_mean", "u500_center", T_tgt, B, device)
# # # #     v500 = _get_uv500_ms(env_data, "v500_mean", "v500_center", T_tgt, B, device)

# # # #     uv_mag       = torch.sqrt(u500**2 + v500**2)
# # # #     has_steering = (uv_mag > _STEERING_MIN_MS).float()

# # # #     env_dir  = torch.stack([u500, v500], dim=-1)
# # # #     tc_dir   = torch.stack([u_tc, v_tc], dim=-1)
# # # #     env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=1.0)
# # # #     tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
# # # #     cos_sim  = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)

# # # #     steer_w  = torch.sigmoid(uv_mag - 1.0)
# # # #     misalign = F.relu(-0.5 - cos_sim).pow(2)
# # # #     return (misalign * steer_w * has_steering) * 0.05


# # # # def pinn_gph500_gradient(pred_abs_deg: torch.Tensor,
# # # #                          env_data: Optional[dict]) -> torch.Tensor:
# # # #     if env_data is None:
# # # #         return pred_abs_deg.new_zeros(())
# # # #     T, B, _ = pred_abs_deg.shape
# # # #     if T < 2:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     device = pred_abs_deg.device
# # # #     T_tgt  = T - 1

# # # #     gph_mean   = _get_gph500_norm(env_data, "gph500_mean",   T_tgt, B, device)
# # # #     gph_center = _get_gph500_norm(env_data, "gph500_center", T_tgt, B, device)

# # # #     gph_diff = gph_center - gph_mean

# # # #     dlat = pred_abs_deg[1:, :, 1] - pred_abs_deg[:-1, :, 1]

# # # #     has_gradient = (gph_diff.abs() > 0.1).float()

# # # #     s_correct = torch.sign(dlat) * torch.sign(gph_diff)
# # # #     wrong_dir = F.relu(-s_correct)

# # # #     return (wrong_dir.pow(2) * has_gradient) * 0.02


# # # # def pinn_steering_speed_consistency(pred_abs_deg: torch.Tensor,
# # # #                                     env_data: Optional[dict]) -> torch.Tensor:
# # # #     if env_data is None:
# # # #         return pred_abs_deg.new_zeros(())
# # # #     T, B, _ = pred_abs_deg.shape
# # # #     if T < 2:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     device = pred_abs_deg.device
# # # #     T_tgt  = T - 1
# # # #     DT     = DT_6H

# # # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # # #     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
# # # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # #     dx_km   = dlat[:, :, 0] * cos_lat * 111.0
# # # #     dy_km   = dlat[:, :, 1] * 111.0
# # # #     tc_speed_ms = torch.sqrt(dx_km**2 + dy_km**2) * 1000.0 / DT

# # # #     u_c = _get_uv500_ms(env_data, "u500_mean", "u500_center", T_tgt, B, device)
# # # #     v_c = _get_uv500_ms(env_data, "v500_mean", "v500_center", T_tgt, B, device)
# # # #     u_m = _get_uv500_ms(env_data, "u500_mean", "u500_mean",   T_tgt, B, device)
# # # #     v_m = _get_uv500_ms(env_data, "v500_mean", "v500_mean",   T_tgt, B, device)

# # # #     steering_mag = (torch.sqrt(u_c**2 + v_c**2) +
# # # #                     torch.sqrt(u_m**2 + v_m**2)) / 2.0

# # # #     steer_var = (steering_mag**2).mean().clamp(min=1.0)

# # # #     has_steering = (steering_mag > _STEERING_MIN_MS).float()
# # # #     lo = steering_mag * 0.3
# # # #     hi = steering_mag * 1.5
# # # #     too_slow = F.relu(lo - tc_speed_ms)
# # # #     too_fast = F.relu(tc_speed_ms - hi)

# # # #     penalty = (too_slow.pow(2) + too_fast.pow(2)) / steer_var
# # # #     return (penalty * has_steering) * 0.03


# # # # def pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # # #     if pred_deg.shape[0] < 2:
# # # #         return pred_deg.new_zeros(())
# # # #     dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # #     lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # #     dx_km   = dt_deg[:, :, 0] * cos_lat * DEG_TO_KM
# # # #     dy_km   = dt_deg[:, :, 1] * DEG_TO_KM
# # # #     speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # # #     return F.relu(speed - 600.0).pow(2)


# # # # def pinn_pressure_wind_loss(
# # # #     pred_abs_deg: torch.Tensor,
# # # #     vmax_pred:    Optional[torch.Tensor],
# # # #     pmin_pred:    Optional[torch.Tensor],
# # # #     r34_km:       Optional[torch.Tensor] = None,
# # # #     epoch:        int = 0,
# # # # ) -> torch.Tensor:
# # # #     if epoch < 30:
# # # #         return pred_abs_deg.new_zeros(())
# # # #     if vmax_pred is None or pmin_pred is None:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     T, B, _ = pred_abs_deg.shape
# # # #     if vmax_pred.shape[0] != T or pmin_pred.shape[0] != T:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
# # # #     f_k     = 2 * OMEGA * torch.sin(lat_rad).abs().clamp(min=1e-6)

# # # #     if r34_km is not None:
# # # #         R_tc = (2.0 * r34_km * 1000.0).clamp(min=1e5, max=8e5)
# # # #     else:
# # # #         R_tc = 3e5 + 1e3 * F.relu(vmax_pred - 30.0)

# # # #     V  = vmax_pred.clamp(min=1.0)
# # # #     dp = (P_ENV - pmin_pred) * 100.0

# # # #     dp_pred = RHO_AIR * (V.pow(2) / 2.0 + f_k * R_tc * V / 2.0)

# # # #     residual = (dp - dp_pred) / 500.0
# # # #     return residual.pow(2)


# # # # def pinn_bve_loss(
# # # #     pred_abs_deg: torch.Tensor,
# # # #     batch_list,
# # # #     env_data:    Optional[dict] = None,
# # # #     epoch:       int = 0,
# # # #     gt_abs_deg:  Optional[torch.Tensor] = None,
# # # #     vmax_pred:   Optional[torch.Tensor] = None,
# # # #     pmin_pred:   Optional[torch.Tensor] = None,
# # # #     r34_km:      Optional[torch.Tensor] = None,
# # # # ) -> torch.Tensor:
# # # #     T = pred_abs_deg.shape[0]
# # # #     if T < 3:
# # # #         return pred_abs_deg.new_zeros(())

# # # #     _env = env_data
# # # #     if _env is None and batch_list is not None:
# # # #         try:
# # # #             _env = batch_list[13]
# # # #         except:
# # # #             _env = None

# # # #     pinn_time_w = torch.linspace(0.5, 4.0, T, device=pred_abs_deg.device)

# # # #     if gt_abs_deg is not None:
# # # #         with torch.no_grad():
# # # #             d_track = _haversine_deg(pred_abs_deg, gt_abs_deg)
# # # #             w_bve_step = torch.sigmoid(1.0 - d_track / 200.0)
# # # #     else:
# # # #         w_bve_step = torch.ones(T, pred_abs_deg.shape[1], device=pred_abs_deg.device)

# # # #     def apply_w(l_map, weight_scalar):
# # # #         if l_map.dim() == 0:
# # # #             return l_map * weight_scalar
# # # #         t_size = l_map.shape[0]
# # # #         w_final = weight_scalar * pinn_time_w[-t_size:, None]
# # # #         return (l_map * w_final).mean()

# # # #     l_sw      = apply_w(pinn_shallow_water(pred_abs_deg), 1.0)
# # # #     l_steer   = apply_w(pinn_rankine_steering(pred_abs_deg, _env), 0.5)
# # # #     l_speed   = apply_w(pinn_speed_constraint(pred_abs_deg), 0.1)
# # # #     l_gph     = apply_w(pinn_gph500_gradient(pred_abs_deg, _env), 0.3)
# # # #     l_spdcons = apply_w(pinn_steering_speed_consistency(pred_abs_deg, _env), 0.4)
# # # #     l_pwr     = apply_w(pinn_pressure_wind_loss(pred_abs_deg, vmax_pred, pmin_pred, r34_km, epoch), 0.6)

# # # #     total = (l_sw.mean() + l_steer.mean() + l_speed.mean() +
# # # #              l_gph.mean() + l_spdcons.mean() + l_pwr.mean())

# # # #     w_bnd = _boundary_weights(pred_abs_deg).mean()
# # # #     f_lazy = 0.2 if epoch < 30 else (0.5 if epoch < 50 else 1.0)

# # # #     total_clamped = adapt_clamp(total, epoch, max_val=20.0)
# # # #     return total_clamped * w_bnd * f_lazy


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Physics consistency (GIỮ NGUYÊN)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def fm_physics_consistency_loss(
# # # #     pred_samples: torch.Tensor,
# # # #     gt_norm:      torch.Tensor,
# # # #     last_pos:     torch.Tensor,
# # # # ) -> torch.Tensor:
# # # #     S, T, B = pred_samples.shape[:3]

# # # #     last_lon = (last_pos[:, 0] * 50.0 + 1800.0) / 10.0
# # # #     last_lat = (last_pos[:, 1] * 50.0) / 10.0
# # # #     lat_rad  = torch.deg2rad(last_lat)
# # # #     beta     = 2 * OMEGA * torch.cos(lat_rad) / R_EARTH
# # # #     R_tc     = 3e5

# # # #     v_beta_lon = -beta * R_tc ** 2 / 2
# # # #     v_beta_lat =  beta * R_tc ** 2 / 4

# # # #     beta_dir      = torch.stack([v_beta_lon, v_beta_lat], dim=-1)
# # # #     beta_norm     = beta_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     beta_dir_unit = beta_dir / beta_norm

# # # #     pos_step1 = pred_samples[:, 0, :, :2]
# # # #     dir_step1 = pos_step1 - last_pos.unsqueeze(0)
# # # #     dir_norm  = dir_step1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     dir_unit  = dir_step1 / dir_norm
# # # #     mean_dir  = dir_unit.mean(0)
# # # #     mean_norm = mean_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     mean_dir_unit = mean_dir / mean_norm

# # # #     cos_align     = (mean_dir_unit * beta_dir_unit).sum(-1)
# # # #     beta_strength = beta_norm.squeeze(-1)
# # # #     penalise_mask = (beta_strength > 1.0).float()
# # # #     direction_loss = F.relu(-cos_align) * penalise_mask
# # # #     return direction_loss.mean() * 0.5


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Main loss (GIỮ NGUYÊN structure, chỉ sửa bridge và spread calls)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def compute_total_loss(
# # # #     pred_abs,
# # # #     gt,
# # # #     ref,
# # # #     batch_list,
# # # #     pred_samples       = None,
# # # #     gt_norm            = None,
# # # #     weights            = WEIGHTS,
# # # #     intensity_w: Optional[torch.Tensor] = None,
# # # #     env_data:    Optional[dict]         = None,
# # # #     step_weight_alpha: float = 0.0,
# # # #     all_trajs:   Optional[torch.Tensor] = None,
# # # #     epoch:       int = 0,
# # # #     gt_abs_deg:  Optional[torch.Tensor] = None,
# # # #     vmax_pred:   Optional[torch.Tensor] = None,
# # # #     pmin_pred:   Optional[torch.Tensor] = None,
# # # #     r34_km:      Optional[torch.Tensor] = None,
# # # #     sr_pred:     Optional[torch.Tensor] = None,
# # # # ) -> Dict:
# # # #     NRM = 35.0

# # # #     recurv_w = _recurvature_weights(gt, w_recurv=2.5)
# # # #     sample_w = recurv_w * (intensity_w.to(gt.device) if intensity_w is not None
# # # #                            else 1.0)
# # # #     sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

# # # #     # FM AFCRPS - với time weights mới
# # # #     if pred_samples is not None:
# # # #         target = gt_norm if gt_norm is not None else gt
# # # #         unit   = gt_norm is not None
# # # #         l_fm = fm_afcrps_loss(
# # # #             pred_samples, target,
# # # #             unit_01deg=unit,
# # # #             intensity_w=sample_w,
# # # #             step_weight_alpha=step_weight_alpha,
# # # #             w_es=0.3,
# # # #         )
# # # #     else:
# # # #         l_fm = _haversine_deg(pred_abs, gt).mean()

# # # #     # Directional losses (GIỮ NGUYÊN)
# # # #     l_vel       = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
# # # #     l_disp      = (disp_loss_per_sample(pred_abs, gt)     * sample_w).mean()
# # # #     l_step      = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()
# # # #     l_heading   = heading_loss(pred_abs, gt)
# # # #     l_recurv    = recurvature_loss(pred_abs, gt)
# # # #     l_dir_final = overall_dir_loss(pred_abs, gt, ref)
# # # #     l_smooth    = smooth_loss(pred_abs)
# # # #     l_accel     = acceleration_loss(pred_abs)
# # # #     l_jerk      = jerk_loss(pred_abs)

# # # #     # PINN (GIỮ NGUYÊN)
# # # #     _env = env_data
# # # #     if _env is None and batch_list is not None:
# # # #         try:
# # # #             _env = batch_list[13]
# # # #         except (IndexError, TypeError):
# # # #             _env = None

# # # #     l_pinn = pinn_bve_loss(
# # # #         pred_abs, batch_list, env_data=_env, epoch=epoch,
# # # #         gt_abs_deg=gt_abs_deg, vmax_pred=vmax_pred,
# # # #         pmin_pred=pmin_pred, r34_km=r34_km
# # # #     )

# # # #     # Spread (RELAXED)
# # # #     l_spread = pred_abs.new_zeros(())
# # # #     if all_trajs is not None and all_trajs.shape[0] >= 2:
# # # #         n_sr = 4
# # # #         trajs_fm = all_trajs[:, n_sr:] if all_trajs.shape[1] > n_sr else all_trajs
# # # #         l_spread = ensemble_spread_loss(trajs_fm)

# # # #     # Bridge (SIMPLIFIED)
# # # #     l_bridge = pred_abs.new_zeros(())
# # # #     if sr_pred is not None and pred_samples is not None:
# # # #         fm_mean = pred_samples.mean(0)
# # # #         l_bridge = bridge_loss(sr_pred, fm_mean)

# # # #     # Total (GIỮ NGUYÊN structure)
# # # #     total = (
# # # #         weights.get("fm",       2.0) * l_fm
# # # #         + weights.get("velocity", 0.8) * l_vel     * NRM
# # # #         + weights.get("disp",     0.5) * l_disp    * NRM
# # # #         + weights.get("step",     0.5) * l_step    * NRM
# # # #         + weights.get("heading",  2.0) * l_heading * NRM
# # # #         + weights.get("recurv",   1.5) * l_recurv  * NRM
# # # #         + weights.get("dir",      1.0) * l_dir_final * NRM
# # # #         + weights.get("smooth",   0.5) * l_smooth  * NRM
# # # #         + weights.get("accel",    0.8) * l_accel   * NRM
# # # #         + weights.get("jerk",     0.3) * l_jerk    * NRM
# # # #         + weights.get("pinn",     0.5) * l_pinn
# # # #         + weights.get("spread",   0.5) * l_spread  * NRM
# # # #         + weights.get("bridge",   0.5) * l_bridge  * NRM
# # # #     ) / NRM

# # # #     if torch.isnan(total) or torch.isinf(total):
# # # #         total = pred_abs.new_zeros(())

# # # #     return dict(
# # # #         total        = total,
# # # #         fm           = l_fm.item(),
# # # #         velocity     = l_vel.item()     * NRM,
# # # #         step         = l_step.item(),
# # # #         disp         = l_disp.item()    * NRM,
# # # #         heading      = l_heading.item(),
# # # #         recurv       = l_recurv.item(),
# # # #         smooth       = l_smooth.item()  * NRM,
# # # #         accel        = l_accel.item()   * NRM,
# # # #         jerk         = l_jerk.item()    * NRM,
# # # #         pinn         = l_pinn.item(),
# # # #         spread       = l_spread.item()  * NRM,
# # # #         bridge       = l_bridge.item()  * NRM,
# # # #         recurv_ratio = (_recurvature_weights(gt) > 1.0).float().mean().item(),
# # # #     )
# # # """
# # # Model/losses.py  ── v30 - HIERARCHICAL
# # # ========================================
# # # MỤC TIÊU: 12h < 50km, 24h < 100km, 48h < 200km, 72h < 300km

# # # CHIẾN LƯỢC MỚI:
# # #   - SR owns step 1-4 (6h-24h) hoàn toàn
# # #   - FM starts from SR endpoint, predicts step 5-12 (30h-72h)
# # #   - KHÔNG CÒN CONFLICT giữa SR và FM
# # #   - Bridge loss → Continuity loss (velocity matching tại handoff)
# # #   - FM time weights ĐỒNG ĐỀU (không suppress short-range)
# # #   - PINN nhẹ nhàng, không dominant
# # #   - Spread loss tuned cho FM-only steps (5-12)

# # # KEY CHANGES từ v28:
# # #   - FM AFCRPS: chỉ tính trên step 5-12 (FM zone)
# # #   - SR loss: tính trên step 1-4 (SR zone) 
# # #   - Continuity loss: SR step4 → FM step5 smooth transition
# # #   - Bỏ bridge loss cũ (position matching không đủ)
# # #   - FM time weights: flat 1.0→2.0 (không suppress step nào)
# # #   - PINN: giảm weight, chỉ apply trên FM zone
# # #   - Spread: chỉ apply trên FM zone
# # # """
# # # from __future__ import annotations

# # # import math
# # # from typing import Dict, Optional, Tuple

# # # import torch
# # # import torch.nn.functional as F

# # # __all__ = [
# # #     "WEIGHTS", "compute_total_loss", "fm_afcrps_loss",
# # #     "fm_physics_consistency_loss", "pinn_bve_loss",
# # #     "recurvature_loss", "velocity_loss_per_sample",
# # #     "short_range_regression_loss", "continuity_loss",
# # #     "adapt_clamp", "pinn_pressure_wind_loss",
# # # ]

# # # # ── Constants ─────────────────────────────────────────────────────────────────
# # # OMEGA        = 7.2921e-5
# # # R_EARTH      = 6.371e6
# # # DT_6H        = 6 * 3600
# # # DEG_TO_KM    = 111.0
# # # STEP_KM      = 113.0
# # # P_ENV        = 1013.0
# # # RHO_AIR      = 1.15

# # # _ERA5_LAT_MIN =   0.0
# # # _ERA5_LAT_MAX =  40.0
# # # _ERA5_LON_MIN = 100.0
# # # _ERA5_LON_MAX = 160.0

# # # _UV500_NORM      = 30.0
# # # _GPH500_MEAN_M   = 5870.0
# # # _GPH500_STD_M    = 80.0
# # # _STEERING_MIN_MS = 3.0
# # # _PINN_SCALE      = 1e-2

# # # N_SR_STEPS = 4  # SR owns step 1-4

# # # # ── Weights ─────────────────────────────────────────────────────────────────
# # # # WEIGHTS: Dict[str, float] = dict(
# # # #     fm          = 2.0,      # FM AFCRPS (step 5-12 only)
# # # #     velocity    = 0.8,
# # # #     heading     = 2.0,
# # # #     recurv      = 1.5,
# # # #     # step        = 0.5,
# # # #     disp        = 0.5,
# # # #     dir         = 1.0,
# # # #     smooth      = 0.5,
# # # #     accel       = 0.8,
# # # #     jerk        = 0.3,
# # # #     pinn        = 0.3,      # Giảm từ 0.5 - PINN không dominant
# # # #     fm_physics  = 0.3,
# # # #     spread      = 0.5,
# # # #     short_range = 3.0,      # SR loss (step 1-4)
# # # #     continuity  = 2.0,      # NEW: SR→FM handoff smoothness
# # # # )
# # # WEIGHTS: Dict[str, float] = dict(
# # #     fm          = 2.5,   # tăng từ 2.0 → FM là loss chính
# # #     short_range = 5.0,   # SR vẫn cao để bước 1-4 chính xác
# # #     velocity    = 0.8,
# # #     heading     = 1.5,
# # #     recurv      = 1.0,
# # #     continuity  = 0.2,      # Giảm cực thấp, chỉ để nối mượt
# # #     # BỎ HOÀN TOÀN: pinn, jerk, disp, dir, step, spread, accel, smooth, continuity
# # #     mse_hav         = 3.0,      # MỚI: Neo tọa độ tuyệt đối (Trọng số cao nhất)
# # # )

# # # RECURV_ANGLE_THR = 45.0
# # # RECURV_WEIGHT    = 2.5

# # # _SR_N_STEPS  = 4
# # # _SR_WEIGHTS  = [1.5, 3.0, 2.0, 2.5]  # 6h, 12h(focus), 18h, 24h(important)
# # # MSE_STEP_WEIGHTS = [1.0, 3.0, 1.5, 2.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]
# # # _HUBER_DELTA = 50.0


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Haversine utilities
# # # # ══════════════════════════════════════════════════════════════════════════════
# # # def mse_haversine_perstep(
# # #     pred_norm: "torch.Tensor",
# # #     gt_norm:   "torch.Tensor",
# # #     step_weights: list = None,
# # # ) -> "torch.Tensor":
# # #     """
# # #     MSE per-step dùng haversine — vũ khí chính để cạnh tranh LSTM.
    
# # #     LSTM dùng MSE(lon, lat) → bị artifact vì cos(lat) không đều.
# # #     Hàm này dùng haversine distance thật sự → physically correct.
# # #     Step weighting nhấn mạnh 12h và 24h.
    
# # #     Args:
# # #         pred_norm: [T, B, 2] normalised
# # #         gt_norm:   [T, B, 2] normalised
# # #     Returns:
# # #         scalar loss
# # #     """
# # #     import torch
# # #     import torch.nn.functional as F
 
# # #     if step_weights is None:
# # #         step_weights = MSE_STEP_WEIGHTS
 
# # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # #     pred_deg = _norm_to_deg(pred_norm[:T])
# # #     gt_deg   = _norm_to_deg(gt_norm[:T])
 
# # #     dist_km = _haversine_deg(pred_deg, gt_deg)   # [T, B]
 
# # #     w = pred_norm.new_tensor(step_weights[:T])
# # #     w = w / w.sum() * T
 
# # #     # Weighted squared haversine
# # #     loss = (dist_km.pow(2) * w.unsqueeze(1)).mean()
 
# # #     # Normalize: 200km scale → gradient ~1.0
# # #     return loss / (200.0 ** 2)

# # # def _haversine(p1: torch.Tensor, p2: torch.Tensor,
# # #                unit_01deg: bool = True) -> torch.Tensor:
# # #     if unit_01deg:
# # #         lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
# # #         lat1 = (p1[..., 1] * 50.0) / 10.0
# # #         lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
# # #         lat2 = (p2[..., 1] * 50.0) / 10.0
# # #     else:
# # #         lon1, lat1 = p1[..., 0], p1[..., 1]
# # #         lon2, lat2 = p2[..., 0], p2[..., 1]

# # #     lat1r = torch.deg2rad(lat1); lat2r = torch.deg2rad(lat2)
# # #     dlon  = torch.deg2rad(lon2 - lon1)
# # #     dlat  = torch.deg2rad(lat2 - lat1)
# # #     a = (torch.sin(dlat / 2).pow(2)
# # #          + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2).pow(2))
# # #     a = a.clamp(1e-12, 1.0 - 1e-12)
# # #     return 2.0 * 6371.0 * torch.asin(a.sqrt())


# # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # #     return _haversine(p1, p2, unit_01deg=False)


# # # def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
# # #     out = arr.clone()
# # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# # #     return out


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  AdaptClamp
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def adapt_clamp(x: torch.Tensor, epoch: int, max_val: float = 20.0) -> torch.Tensor:
# # #     delta = max_val

# # #     def huber_clamp(v: torch.Tensor) -> torch.Tensor:
# # #         return torch.where(
# # #             v <= delta,
# # #             v.pow(2) / (2.0 * delta),
# # #             v - delta / 2.0,
# # #         )

# # #     def tanh_clamp(v: torch.Tensor) -> torch.Tensor:
# # #         return delta * torch.tanh(v / delta)

# # #     if epoch < 10:
# # #         return huber_clamp(x)
# # #     elif epoch < 20:
# # #         beta = (epoch - 10) / 10.0
# # #         return (1.0 - beta) * huber_clamp(x) + beta * tanh_clamp(x)
# # #     else:
# # #         return tanh_clamp(x)


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  ★ FM AFCRPS - cho FM zone (step 5-12) HOẶC full (step 1-12)
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def fm_afcrps_loss(
# # #     pred_samples:      torch.Tensor,
# # #     gt:                torch.Tensor,
# # #     unit_01deg:        bool  = True,
# # #     intensity_w:       Optional[torch.Tensor] = None,
# # #     step_weight_alpha: float = 0.0,
# # #     w_es:              float = 0.3,
# # #     fm_start_step:     int   = 0,
# # # ) -> torch.Tensor:
# # #     """
# # #     FM AFCRPS loss.
    
# # #     fm_start_step: nếu > 0, chỉ tính loss từ step này trở đi
# # #                    (step 4 = bắt đầu FM zone trong hierarchical mode)
    
# # #     Time weights ĐỒNG ĐỀU: 1.0 → 2.5 (gentle ramp)
# # #     Không suppress step nào → FM learns accurate ở ALL steps
# # #     """
# # #     M, T, B, _ = pred_samples.shape
# # #     device = pred_samples.device

# # #     # Slice to FM zone if needed
# # #     if fm_start_step > 0 and T > fm_start_step:
# # #         pred_samples = pred_samples[:, fm_start_step:]
# # #         gt = gt[fm_start_step:]
# # #         M, T, B, _ = pred_samples.shape

# # #     if T == 0:
# # #         return pred_samples.new_zeros(())

# # #     # Time weights: gentle ramp, no step suppressed
# # #     base_w = torch.linspace(1.0, 2.5, T, device=device)

# # #     if step_weight_alpha > 0.0:
# # #         early_w = torch.exp(
# # #             -torch.arange(T, dtype=torch.float, device=device) * 0.3
# # #         )
# # #         early_w = early_w / early_w.mean()
# # #         time_w = (1.0 - step_weight_alpha) * base_w + step_weight_alpha * early_w
# # #     else:
# # #         time_w = base_w

# # #     if M == 1:
# # #         dist = _haversine(pred_samples[0], gt, unit_01deg)
# # #         loss_per_b = (dist * time_w.unsqueeze(1)).mean(0)
# # #         es_term = pred_samples.new_zeros(())
# # #     else:
# # #         # Accuracy: E[d(X^m, Y)]
# # #         d_to_gt = _haversine(
# # #             pred_samples,
# # #             gt.unsqueeze(0).expand_as(pred_samples),
# # #             unit_01deg,
# # #         )
# # #         d_to_gt_w    = d_to_gt * time_w.unsqueeze(0).unsqueeze(2)
# # #         e_sy         = d_to_gt_w.mean(1).mean(0)

# # #         # Sharpness: E[d(X^m, X^m')]
# # #         ps_i = pred_samples.unsqueeze(1)
# # #         ps_j = pred_samples.unsqueeze(0)
# # #         d_pair = _haversine(
# # #             ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # #             ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # #             unit_01deg,
# # #         ).reshape(M, M, T, B)

# # #         diag_mask = torch.eye(M, device=device, dtype=torch.bool)
# # #         diag_mask = diag_mask.view(M, M, 1, 1).expand_as(d_pair)
# # #         d_pair = d_pair.masked_fill(diag_mask, 0.0)

# # #         d_pair_w    = d_pair * time_w.unsqueeze(0).unsqueeze(0).unsqueeze(3)
# # #         e_ssp       = d_pair_w.mean(2).sum(0).sum(0) / (M * (M - 1))

# # #         loss_per_b = e_sy - 0.5 * e_ssp

# # #         # Energy Score term
# # #         if w_es > 0.0 and M > 1:
# # #             ps_flat  = pred_samples.reshape(M, T * B, 2)
# # #             gt_flat  = gt.reshape(T * B, 2)
# # #             mean_pred = ps_flat.mean(0)

# # #             es_acc = (mean_pred - gt_flat).pow(2).sum(-1).sqrt().mean()

# # #             ps_i_f = ps_flat.unsqueeze(1)
# # #             ps_j_f = ps_flat.unsqueeze(0)
# # #             d_pair_f = (ps_i_f - ps_j_f).pow(2).sum(-1).sqrt()
# # #             diag_f = torch.eye(M, device=device, dtype=torch.bool)
# # #             diag_f = diag_f.view(M, M, 1).expand_as(d_pair_f)
# # #             d_pair_f = d_pair_f.masked_fill(diag_f, 0.0)
# # #             es_sharp = (M - 1) / M * d_pair_f.sum(0).sum(0).mean() / (M * (M - 1))

# # #             es_term = es_acc - es_sharp
# # #         else:
# # #             es_term = pred_samples.new_zeros(())

# # #     if intensity_w is not None:
# # #         w = intensity_w.to(loss_per_b.device)
# # #         crps_loss = (loss_per_b * w).mean()
# # #     else:
# # #         crps_loss = loss_per_b.mean()

# # #     return crps_loss + w_es * es_term


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Short-range Loss (step 1-4)
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def short_range_regression_loss(
# # #     pred_sr:  torch.Tensor,
# # #     gt_sr:    torch.Tensor,
# # #     last_pos: torch.Tensor,
# # # ) -> torch.Tensor:
# # #     n_steps = min(pred_sr.shape[0], gt_sr.shape[0], _SR_N_STEPS)
# # #     if n_steps == 0:
# # #         return pred_sr.new_zeros(())

# # #     pred_deg = _norm_to_deg(pred_sr[:n_steps])
# # #     gt_deg   = _norm_to_deg(gt_sr[:n_steps])

# # #     dist_km = _haversine_deg(pred_deg, gt_deg)

# # #     huber = torch.where(
# # #         dist_km < _HUBER_DELTA,
# # #         0.5 * dist_km.pow(2) / _HUBER_DELTA,
# # #         dist_km - 0.5 * _HUBER_DELTA,
# # #     )

# # #     w = pred_sr.new_tensor(_SR_WEIGHTS[:n_steps])
# # #     w = w / w.sum()
# # #     return (huber * w.view(-1, 1)).mean()


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  ★ NEW: Continuity Loss (SR→FM handoff)
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def continuity_loss(
# # #     sr_pred:     torch.Tensor,    # [4, B, 2] normalised
# # #     fm_pred_abs: torch.Tensor,    # [T_fm, B, 2] degrees (FM predictions step 5-12)
# # #     gt_abs:      torch.Tensor,    # [T_full, B, 2] degrees
# # # ) -> torch.Tensor:
# # #     """
# # #     Continuity Loss: đảm bảo transition mượt từ SR step 4 → FM step 5.
    
# # #     3 components:
# # #     1. Position continuity: FM step 5 gần SR step 4 (khoảng cách hợp lý)
# # #     2. Velocity continuity: hướng đi FM step 5 consistent với SR step 3→4
# # #     3. Acceleration smoothness: không có "giật" tại handoff
    
# # #     Tất cả normalize về km, weight hợp lý.
# # #     """
# # #     if sr_pred.shape[0] < 4 or fm_pred_abs.shape[0] < 2:
# # #         return sr_pred.new_zeros(())

# # #     # Convert SR to degrees
# # #     sr_deg = _norm_to_deg(sr_pred)  # [4, B, 2]
    
# # #     # SR velocity at step 3→4 (last SR velocity)
# # #     sr_vel = sr_deg[3] - sr_deg[2]  # [B, 2] in degrees
    
# # #     # FM velocity at step 4→5 (first FM velocity)
# # #     # fm_pred_abs[0] = step 5, sr_deg[3] = step 4
# # #     fm_vel = fm_pred_abs[0] - sr_deg[3]  # [B, 2] in degrees
    
# # #     # 1. Position continuity: FM step 5 should be reachable from SR step 4
# # #     # Typical TC moves 50-200km in 6h, so gap should be < 300km
# # #     pos_dist = _haversine_deg(sr_deg[3:4], fm_pred_abs[0:1]).squeeze(0)  # [B]
# # #     l_pos = F.relu(pos_dist - 300.0).pow(2).mean() / (300.0 ** 2)
    
# # #     # 2. Velocity continuity: direction should be similar
# # #     lat_mid = sr_deg[3, :, 1]
# # #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
    
# # #     sr_vel_km = sr_vel.clone()
# # #     sr_vel_km[:, 0] = sr_vel[:, 0] * cos_lat * DEG_TO_KM
# # #     sr_vel_km[:, 1] = sr_vel[:, 1] * DEG_TO_KM
    
# # #     fm_vel_km = fm_vel.clone()
# # #     fm_vel_km[:, 0] = fm_vel[:, 0] * cos_lat * DEG_TO_KM
# # #     fm_vel_km[:, 1] = fm_vel[:, 1] * DEG_TO_KM
    
# # #     sr_speed = sr_vel_km.norm(dim=-1).clamp(min=1e-4)
# # #     fm_speed = fm_vel_km.norm(dim=-1).clamp(min=1e-4)
    
# # #     # Speed ratio: FM speed should be 0.5x-2.0x of SR speed
# # #     speed_ratio = fm_speed / sr_speed
# # #     l_speed = (F.relu(speed_ratio - 2.0).pow(2) + 
# # #                F.relu(0.5 - speed_ratio).pow(2)).mean()
    
# # #     # Direction: cosine similarity
# # #     cos_sim = (sr_vel_km * fm_vel_km).sum(-1) / (sr_speed * fm_speed)
# # #     l_dir = F.relu(-cos_sim).pow(2).mean()  # Penalize opposite direction
    
# # #     # 3. Acceleration smoothness at handoff
# # #     if fm_pred_abs.shape[0] >= 2:
# # #         fm_vel2 = fm_pred_abs[1] - fm_pred_abs[0]  # step 5→6 velocity
# # #         fm_vel2_km = fm_vel2.clone()
# # #         fm_vel2_km[:, 0] = fm_vel2[:, 0] * cos_lat * DEG_TO_KM
# # #         fm_vel2_km[:, 1] = fm_vel2[:, 1] * DEG_TO_KM
        
# # #         # Acceleration at handoff vs acceleration in FM
# # #         accel_handoff = fm_vel_km - sr_vel_km
# # #         accel_fm = fm_vel2_km - fm_vel_km
# # #         l_accel = (accel_handoff - accel_fm).pow(2).mean() / (STEP_KM ** 2)
# # #     else:
# # #         l_accel = sr_pred.new_zeros(())
    
# # #     # # return 0.5 * l_pos + 0.3 * l_dir + 0.1 * l_speed + 0.1 * l_accel
# # #     #     result = 0.5 * l_pos + 0.3 * l_dir + 0.1 * l_speed + 0.1 * l_accel
# # #     # return result.clamp(max=50.0)   # ← THÊM DÒNG NÀY: tránh spike 19,945
# # #     return (0.5 * l_pos + 0.3 * l_dir + 0.1 * l_speed + 0.1 * l_accel).clamp(max=50.0)


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Spread Loss - FM zone only (step 5-12)
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def ensemble_spread_loss(all_trajs: torch.Tensor) -> torch.Tensor:
# # #     """
# # #     Spread loss cho FM zone.
    
# # #     Thresholds hợp lý cho 30h-72h:
# # #     - 30h: max 80km spread
# # #     - 72h: max 250km spread
# # #     """
# # #     if all_trajs.shape[0] < 2:
# # #         return all_trajs.new_zeros(())

# # #     S, T, B, _ = all_trajs.shape
# # #     device = all_trajs.device

# # #     max_spreads = torch.linspace(80.0, 250.0, T, device=device)
# # #     step_weights = torch.linspace(0.5, 2.0, T, device=device)

# # #     total_loss = all_trajs.new_zeros(())
# # #     for t in range(T):
# # #         step_trajs = all_trajs[:, t, :, :2]
# # #         std_lon    = step_trajs[:, :, 0].std(0)
# # #         std_lat    = step_trajs[:, :, 1].std(0)
# # #         spread_km  = torch.sqrt(std_lon ** 2 + std_lat ** 2) * 500.0

# # #         excess = F.relu(spread_km - max_spreads[t])
# # #         loss = (excess / 50.0).pow(2)

# # #         total_loss = total_loss + step_weights[t] * loss.mean()

# # #     return total_loss / T


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Helper functions
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def _boundary_weights(traj_deg: torch.Tensor) -> torch.Tensor:
# # #     lon = traj_deg[..., 0]
# # #     lat = traj_deg[..., 1]

# # #     d_lat_lo = (lat - _ERA5_LAT_MIN) / 5.0
# # #     d_lat_hi = (_ERA5_LAT_MAX - lat) / 5.0
# # #     d_lon_lo = (lon - _ERA5_LON_MIN) / 5.0
# # #     d_lon_hi = (_ERA5_LON_MAX - lon) / 5.0

# # #     d_bnd = torch.stack([d_lat_lo, d_lat_hi, d_lon_lo, d_lon_hi], dim=-1).min(dim=-1).values
# # #     return torch.sigmoid(d_bnd - 0.5)


# # # def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
# # #     dt      = traj_deg[1:] - traj_deg[:-1]
# # #     lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
# # #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
# # #     dt_km   = dt.clone()
# # #     dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
# # #     dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
# # #     return dt_km


# # # def _total_rotation_angle_batch(gt: torch.Tensor) -> torch.Tensor:
# # #     T, B, _ = gt.shape
# # #     if T < 3:
# # #         return gt.new_zeros(B)
# # #     lats_rad = torch.deg2rad(gt[:, :, 1])
# # #     cos_lat  = torch.cos(lats_rad[:-1])
# # #     dlat = gt[1:, :, 1] - gt[:-1, :, 1]
# # #     dlon = (gt[1:, :, 0] - gt[:-1, :, 0]) * cos_lat
# # #     v    = torch.stack([dlon, dlat], dim=-1)
# # #     v1   = v[:-1]; v2 = v[1:]
# # #     n1   = v1.norm(dim=-1).clamp(min=1e-8)
# # #     n2   = v2.norm(dim=-1).clamp(min=1e-8)
# # #     cos_a = (v1 * v2).sum(-1) / (n1 * n2)
# # #     return torch.rad2deg(torch.acos(cos_a.clamp(-1.0, 1.0))).sum(0)


# # # def _recurvature_weights(gt: torch.Tensor,
# # #                          thr: float = RECURV_ANGLE_THR,
# # #                          w_recurv: float = RECURV_WEIGHT) -> torch.Tensor:
# # #     rot = _total_rotation_angle_batch(gt)
# # #     return torch.where(rot >= thr,
# # #                        torch.full_like(rot, w_recurv),
# # #                        torch.ones_like(rot))


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Directional losses (apply trên FULL trajectory)
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def velocity_loss_per_sample(pred: torch.Tensor,
# # #                              gt: torch.Tensor) -> torch.Tensor:
# # #     if pred.shape[0] < 2:
# # #         return pred.new_zeros(pred.shape[1])
# # #     v_pred_km = _step_displacements_km(pred)
# # #     v_gt_km   = _step_displacements_km(gt)
# # #     s_pred    = v_pred_km.norm(dim=-1)
# # #     s_gt      = v_gt_km.norm(dim=-1)
# # #     l_speed   = (s_pred - s_gt).pow(2).mean(0)
# # #     gn        = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # #     gt_unit   = v_gt_km / gn
# # #     ate       = ((v_pred_km - v_gt_km) * gt_unit).sum(-1)
# # #     l_ate     = ate.pow(2).mean(0)
# # #     pn        = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # #     cos_sim   = ((v_pred_km / pn) * gt_unit).sum(-1)
# # #     l_dir     = F.relu(-cos_sim).pow(2).mean(0) * STEP_KM ** 2
# # #     return (l_speed + 0.5 * l_ate + 0.3 * l_dir) / (STEP_KM ** 2)


# # # def disp_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # #     if pred.shape[0] < 2:
# # #         return pred.new_zeros(pred.shape[1])
# # #     pd = _step_displacements_km(pred).norm(dim=-1).mean(0)
# # #     gd = _step_displacements_km(gt).norm(dim=-1).mean(0)
# # #     return (pd - gd).pow(2) / (STEP_KM ** 2)


# # # def step_dir_loss_per_sample(pred: torch.Tensor,
# # #                               gt: torch.Tensor) -> torch.Tensor:
# # #     if pred.shape[0] < 2:
# # #         return pred.new_zeros(pred.shape[1])
# # #     v_pred_km = _step_displacements_km(pred)
# # #     v_gt_km   = _step_displacements_km(gt)
# # #     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # #     gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # #     cos_sim = ((v_pred_km / pn) * (v_gt_km / gn)).sum(-1)
# # #     return (1.0 - cos_sim).mean(0)


# # # def heading_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # #     if pred.shape[0] < 2:
# # #         return pred.new_zeros(())
# # #     pv = _step_displacements_km(pred)
# # #     gv = _step_displacements_km(gt)
# # #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # #     wrong_dir      = F.relu(-((pv / pn) * (gv / gn)).sum(-1))
# # #     wrong_dir_loss = (wrong_dir ** 2).mean()

# # #     if pred.shape[0] >= 3:
# # #         def _curv(v):
# # #             cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
# # #             n1    = v[1:].norm(dim=-1).clamp(min=1e-4)
# # #             n2    = v[:-1].norm(dim=-1).clamp(min=1e-4)
# # #             return (cross / (n1 * n2)).clamp(-10.0, 10.0)
# # #         curv_mse = F.mse_loss(_curv(pv), _curv(gv))
# # #     else:
# # #         curv_mse = pred.new_zeros(())
# # #     return wrong_dir_loss + curv_mse


# # # def smooth_loss(pred: torch.Tensor) -> torch.Tensor:
# # #     if pred.shape[0] < 3:
# # #         return pred.new_zeros(())
# # #     v_km = _step_displacements_km(pred)
# # #     if v_km.shape[0] < 2:
# # #         return pred.new_zeros(())
# # #     accel_km = v_km[1:] - v_km[:-1]
# # #     return accel_km.pow(2).mean() / (STEP_KM ** 2)


# # # def acceleration_loss(pred: torch.Tensor) -> torch.Tensor:
# # #     return smooth_loss(pred)


# # # def jerk_loss(pred: torch.Tensor) -> torch.Tensor:
# # #     if pred.shape[0] < 4:
# # #         return pred.new_zeros(())
# # #     v_km = _step_displacements_km(pred)
# # #     if v_km.shape[0] < 3:
# # #         return pred.new_zeros(())
# # #     a_km = v_km[1:] - v_km[:-1]
# # #     j_km = a_km[1:] - a_km[:-1]
# # #     return j_km.pow(2).mean() / (STEP_KM ** 2)


# # # def overall_dir_loss(pred: torch.Tensor, gt: torch.Tensor,
# # #                      ref: torch.Tensor) -> torch.Tensor:
# # #     p_d = pred[-1] - ref
# # #     g_d = gt[-1]   - ref
# # #     lat_ref = ref[:, 1]
# # #     cos_lat = torch.cos(torch.deg2rad(lat_ref)).clamp(min=1e-4)
# # #     p_d_km  = p_d.clone(); p_d_km[:, 0] *= cos_lat * DEG_TO_KM; p_d_km[:, 1] *= DEG_TO_KM
# # #     g_d_km  = g_d.clone(); g_d_km[:, 0] *= cos_lat * DEG_TO_KM; g_d_km[:, 1] *= DEG_TO_KM
# # #     pn = p_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # #     gn = g_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# # #     return (1.0 - ((p_d_km / pn) * (g_d_km / gn)).sum(-1)).mean()


# # # def recurvature_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# # #     if pred.shape[0] < 3:
# # #         return pred.new_zeros(())
# # #     pred_v   = pred[1:] - pred[:-1]
# # #     gt_v     = gt[1:]   - gt[:-1]
# # #     gt_cross = (gt_v[:-1, :, 0] * gt_v[1:, :, 1]
# # #               - gt_v[:-1, :, 1] * gt_v[1:, :, 0])
# # #     if gt_cross.shape[0] < 2:
# # #         return pred.new_zeros(())
# # #     sign_change = (gt_cross[:-1] * gt_cross[1:]) < 0
# # #     if not sign_change.any():
# # #         return pred.new_zeros(())
# # #     pred_v_mid = pred_v[1:-1]
# # #     gt_v_mid   = gt_v[1:-1]
# # #     if pred_v_mid.shape[0] > sign_change.shape[0]:
# # #         pred_v_mid = pred_v_mid[:sign_change.shape[0]]
# # #         gt_v_mid   = gt_v_mid[:sign_change.shape[0]]
# # #     pn      = pred_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # #     gn      = gt_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # #     cos_sim  = ((pred_v_mid / pn) * (gt_v_mid / gn)).sum(-1)
# # #     dir_loss = (1.0 - cos_sim)
# # #     mask     = sign_change.float()
# # #     if mask.sum() < 1:
# # #         return pred.new_zeros(())
# # #     return (dir_loss * mask).sum() / mask.sum().clamp(min=1)


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  PINN Components
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def _get_uv500_ms(env_data: dict, key_mean: str, key_center: str,
# # #                   T_tgt: int, B: int, device: torch.device) -> torch.Tensor:
# # #     def _extract(key):
# # #         x = env_data.get(key, None)
# # #         if x is None or not torch.is_tensor(x):
# # #             return None
# # #         x = x.to(device).float()
# # #         if x.dim() == 3:
# # #             x = x[..., 0]
# # #         elif x.dim() == 1:
# # #             x = x.unsqueeze(0).expand(B, -1)
# # #         x = x.permute(1, 0)
# # #         T_obs = x.shape[0]
# # #         if T_obs >= T_tgt:
# # #             return x[:T_tgt] * _UV500_NORM
# # #         pad = torch.zeros(T_tgt - T_obs, B, device=device)
# # #         return torch.cat([x * _UV500_NORM, pad], dim=0)

# # #     val = _extract(key_center)
# # #     if val is not None:
# # #         return val
# # #     val = _extract(key_mean)
# # #     if val is not None:
# # #         return val
# # #     return torch.zeros(T_tgt, B, device=device)


# # # def _get_gph500_norm(env_data: dict, key: str,
# # #                      T_tgt: int, B: int, device: torch.device) -> torch.Tensor:
# # #     x = env_data.get(key, None)
# # #     if x is None or not torch.is_tensor(x):
# # #         return torch.zeros(T_tgt, B, device=device)
# # #     x = x.to(device).float()
# # #     if x.dim() == 3:
# # #         x = x[..., 0]
# # #     x = x.permute(1, 0)
# # #     T_obs = x.shape[0]
# # #     if T_obs >= T_tgt:
# # #         return x[:T_tgt]
# # #     pad = torch.zeros(T_tgt - T_obs, B, device=device)
# # #     return torch.cat([x, pad], dim=0)


# # # def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
# # #     T, B, _ = pred_abs_deg.shape
# # #     if T < 3:
# # #         return pred_abs_deg.new_zeros(())

# # #     DT      = DT_6H
# # #     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
# # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # #     cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

# # #     u = dlat[:, :, 0] * cos_lat * 111000.0 / DT
# # #     v = dlat[:, :, 1] * 111000.0 / DT

# # #     if u.shape[0] < 2:
# # #         return pred_abs_deg.new_zeros(())

# # #     du = (u[1:] - u[:-1]) / DT
# # #     dv = (v[1:] - v[:-1]) / DT

# # #     f    = 2 * OMEGA * torch.sin(lat_rad[1:-1])
# # #     beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH

# # #     res_u = du - f * v[1:]
# # #     res_v = dv + f * u[1:]

# # #     R_tc            = 3e5
# # #     v_beta_x        = -beta * R_tc ** 2 / 2
# # #     res_u_corrected = res_u - v_beta_x

# # #     scale = 0.1
# # #     loss = ((res_u_corrected / scale).pow(2) + (res_v / scale).pow(2))
# # #     return loss


# # # def pinn_rankine_steering(pred_abs_deg: torch.Tensor,
# # #                           env_data: Optional[dict]) -> torch.Tensor:
# # #     if env_data is None:
# # #         return pred_abs_deg.new_zeros(())

# # #     T, B, _ = pred_abs_deg.shape
# # #     if T < 2:
# # #         return pred_abs_deg.new_zeros(())

# # #     device  = pred_abs_deg.device
# # #     T_tgt   = T - 1
# # #     DT      = DT_6H

# # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # #     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
# # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

# # #     u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT
# # #     v_tc = dlat[:, :, 1] * 111000.0 / DT

# # #     u500 = _get_uv500_ms(env_data, "u500_mean", "u500_center", T_tgt, B, device)
# # #     v500 = _get_uv500_ms(env_data, "v500_mean", "v500_center", T_tgt, B, device)

# # #     uv_mag       = torch.sqrt(u500**2 + v500**2)
# # #     has_steering = (uv_mag > _STEERING_MIN_MS).float()

# # #     env_dir  = torch.stack([u500, v500], dim=-1)
# # #     tc_dir   = torch.stack([u_tc, v_tc], dim=-1)
# # #     env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=1.0)
# # #     tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
# # #     cos_sim  = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)

# # #     steer_w  = torch.sigmoid(uv_mag - 1.0)
# # #     misalign = F.relu(-0.5 - cos_sim).pow(2)
# # #     return (misalign * steer_w * has_steering) * 0.05


# # # def pinn_gph500_gradient(pred_abs_deg: torch.Tensor,
# # #                          env_data: Optional[dict]) -> torch.Tensor:
# # #     if env_data is None:
# # #         return pred_abs_deg.new_zeros(())
# # #     T, B, _ = pred_abs_deg.shape
# # #     if T < 2:
# # #         return pred_abs_deg.new_zeros(())

# # #     device = pred_abs_deg.device
# # #     T_tgt  = T - 1

# # #     gph_mean   = _get_gph500_norm(env_data, "gph500_mean",   T_tgt, B, device)
# # #     gph_center = _get_gph500_norm(env_data, "gph500_center", T_tgt, B, device)

# # #     gph_diff = gph_center - gph_mean
# # #     dlat = pred_abs_deg[1:, :, 1] - pred_abs_deg[:-1, :, 1]
# # #     has_gradient = (gph_diff.abs() > 0.1).float()
# # #     s_correct = torch.sign(dlat) * torch.sign(gph_diff)
# # #     wrong_dir = F.relu(-s_correct)

# # #     return (wrong_dir.pow(2) * has_gradient) * 0.02


# # # def pinn_steering_speed_consistency(pred_abs_deg: torch.Tensor,
# # #                                     env_data: Optional[dict]) -> torch.Tensor:
# # #     if env_data is None:
# # #         return pred_abs_deg.new_zeros(())
# # #     T, B, _ = pred_abs_deg.shape
# # #     if T < 2:
# # #         return pred_abs_deg.new_zeros(())

# # #     device = pred_abs_deg.device
# # #     T_tgt  = T - 1
# # #     DT     = DT_6H

# # #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# # #     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
# # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # #     dx_km   = dlat[:, :, 0] * cos_lat * 111.0
# # #     dy_km   = dlat[:, :, 1] * 111.0
# # #     tc_speed_ms = torch.sqrt(dx_km**2 + dy_km**2) * 1000.0 / DT

# # #     u_c = _get_uv500_ms(env_data, "u500_mean", "u500_center", T_tgt, B, device)
# # #     v_c = _get_uv500_ms(env_data, "v500_mean", "v500_center", T_tgt, B, device)
# # #     u_m = _get_uv500_ms(env_data, "u500_mean", "u500_mean",   T_tgt, B, device)
# # #     v_m = _get_uv500_ms(env_data, "v500_mean", "v500_mean",   T_tgt, B, device)

# # #     steering_mag = (torch.sqrt(u_c**2 + v_c**2) +
# # #                     torch.sqrt(u_m**2 + v_m**2)) / 2.0

# # #     steer_var = (steering_mag**2).mean().clamp(min=1.0)

# # #     has_steering = (steering_mag > _STEERING_MIN_MS).float()
# # #     lo = steering_mag * 0.3
# # #     hi = steering_mag * 1.5
# # #     too_slow = F.relu(lo - tc_speed_ms)
# # #     too_fast = F.relu(tc_speed_ms - hi)

# # #     penalty = (too_slow.pow(2) + too_fast.pow(2)) / steer_var
# # #     return (penalty * has_steering) * 0.03


# # # def pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # #     if pred_deg.shape[0] < 2:
# # #         return pred_deg.new_zeros(())
# # #     dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # #     lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # #     dx_km   = dt_deg[:, :, 0] * cos_lat * DEG_TO_KM
# # #     dy_km   = dt_deg[:, :, 1] * DEG_TO_KM
# # #     speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # #     return F.relu(speed - 600.0).pow(2)


# # # def pinn_pressure_wind_loss(
# # #     pred_abs_deg: torch.Tensor,
# # #     vmax_pred:    Optional[torch.Tensor],
# # #     pmin_pred:    Optional[torch.Tensor],
# # #     r34_km:       Optional[torch.Tensor] = None,
# # #     epoch:        int = 0,
# # # ) -> torch.Tensor:
# # #     if epoch < 30:
# # #         return pred_abs_deg.new_zeros(())
# # #     if vmax_pred is None or pmin_pred is None:
# # #         return pred_abs_deg.new_zeros(())

# # #     T, B, _ = pred_abs_deg.shape
# # #     if vmax_pred.shape[0] != T or pmin_pred.shape[0] != T:
# # #         return pred_abs_deg.new_zeros(())

# # #     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
# # #     f_k     = 2 * OMEGA * torch.sin(lat_rad).abs().clamp(min=1e-6)

# # #     if r34_km is not None:
# # #         R_tc = (2.0 * r34_km * 1000.0).clamp(min=1e5, max=8e5)
# # #     else:
# # #         R_tc = 3e5 + 1e3 * F.relu(vmax_pred - 30.0)

# # #     V  = vmax_pred.clamp(min=1.0)
# # #     dp = (P_ENV - pmin_pred) * 100.0

# # #     dp_pred = RHO_AIR * (V.pow(2) / 2.0 + f_k * R_tc * V / 2.0)

# # #     residual = (dp - dp_pred) / 500.0
# # #     return residual.pow(2)


# # # def pinn_bve_loss(
# # #     pred_abs_deg: torch.Tensor,
# # #     batch_list,
# # #     env_data:    Optional[dict] = None,
# # #     epoch:       int = 0,
# # #     gt_abs_deg:  Optional[torch.Tensor] = None,
# # #     vmax_pred:   Optional[torch.Tensor] = None,
# # #     pmin_pred:   Optional[torch.Tensor] = None,
# # #     r34_km:      Optional[torch.Tensor] = None,
# # # ) -> torch.Tensor:
# # #     T = pred_abs_deg.shape[0]
# # #     if T < 3:
# # #         return pred_abs_deg.new_zeros(())

# # #     _env = env_data
# # #     if _env is None and batch_list is not None:
# # #         try:
# # #             _env = batch_list[13]
# # #         except:
# # #             _env = None

# # #     pinn_time_w = torch.linspace(0.5, 3.0, T, device=pred_abs_deg.device)

# # #     if gt_abs_deg is not None:
# # #         with torch.no_grad():
# # #             d_track = _haversine_deg(pred_abs_deg, gt_abs_deg)
# # #             w_bve_step = torch.sigmoid(1.0 - d_track / 200.0)
# # #     else:
# # #         w_bve_step = torch.ones(T, pred_abs_deg.shape[1], device=pred_abs_deg.device)

# # #     def apply_w(l_map, weight_scalar):
# # #         if l_map.dim() == 0:
# # #             return l_map * weight_scalar
# # #         t_size = l_map.shape[0]
# # #         w_final = weight_scalar * pinn_time_w[-t_size:, None]
# # #         return (l_map * w_final).mean()

# # #     l_sw      = apply_w(pinn_shallow_water(pred_abs_deg), 1.0)
# # #     l_steer   = apply_w(pinn_rankine_steering(pred_abs_deg, _env), 0.5)
# # #     l_speed   = apply_w(pinn_speed_constraint(pred_abs_deg), 0.1)
# # #     l_gph     = apply_w(pinn_gph500_gradient(pred_abs_deg, _env), 0.3)
# # #     l_spdcons = apply_w(pinn_steering_speed_consistency(pred_abs_deg, _env), 0.4)
# # #     l_pwr     = apply_w(pinn_pressure_wind_loss(pred_abs_deg, vmax_pred, pmin_pred, r34_km, epoch), 0.6)

# # #     total = (l_sw.mean() + l_steer.mean() + l_speed.mean() +
# # #              l_gph.mean() + l_spdcons.mean() + l_pwr.mean())

# # #     w_bnd = _boundary_weights(pred_abs_deg).mean()
# # #     f_lazy = 0.2 if epoch < 20 else (0.5 if epoch < 40 else 1.0)

# # #     total_clamped = adapt_clamp(total, epoch, max_val=15.0)
# # #     return total_clamped * w_bnd * f_lazy


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Physics consistency
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def fm_physics_consistency_loss(
# # #     pred_samples: torch.Tensor,
# # #     gt_norm:      torch.Tensor,
# # #     last_pos:     torch.Tensor,
# # # ) -> torch.Tensor:
# # #     S, T, B = pred_samples.shape[:3]

# # #     last_lon = (last_pos[:, 0] * 50.0 + 1800.0) / 10.0
# # #     last_lat = (last_pos[:, 1] * 50.0) / 10.0
# # #     lat_rad  = torch.deg2rad(last_lat)
# # #     beta     = 2 * OMEGA * torch.cos(lat_rad) / R_EARTH
# # #     R_tc     = 3e5

# # #     v_beta_lon = -beta * R_tc ** 2 / 2
# # #     v_beta_lat =  beta * R_tc ** 2 / 4

# # #     beta_dir      = torch.stack([v_beta_lon, v_beta_lat], dim=-1)
# # #     beta_norm     = beta_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # #     beta_dir_unit = beta_dir / beta_norm

# # #     pos_step1 = pred_samples[:, 0, :, :2]
# # #     dir_step1 = pos_step1 - last_pos.unsqueeze(0)
# # #     dir_norm  = dir_step1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # #     dir_unit  = dir_step1 / dir_norm
# # #     mean_dir  = dir_unit.mean(0)
# # #     mean_norm = mean_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # #     mean_dir_unit = mean_dir / mean_norm

# # #     cos_align     = (mean_dir_unit * beta_dir_unit).sum(-1)
# # #     beta_strength = beta_norm.squeeze(-1)
# # #     penalise_mask = (beta_strength > 1.0).float()
# # #     direction_loss = F.relu(-cos_align) * penalise_mask
# # #     return direction_loss.mean() * 0.5

# # # def compute_total_loss(
# # #     pred_abs,          # [T, B, 2] degrees - FULL 12 steps
# # #     gt,                # [T, B, 2] degrees
# # #     ref,               # [B, 2] degrees
# # #     batch_list,
# # #     pred_samples       = None,   # [S, T, B, 2] normalised
# # #     gt_norm            = None,   # [T, B, 2] normalised (FULL, không phải FM zone)
# # #     weights            = None,
# # #     intensity_w        = None,
# # #     step_weight_alpha  = 0.0,
# # #     all_trajs          = None,
# # #     epoch              = 0,
# # #     gt_abs_deg         = None,
# # #     sr_pred            = None,
# # #     fm_pred_abs        = None,
# # #     **kwargs,          # ignore các params cũ
# # # ) -> dict:
# # #     """
# # #     Loss v31 — đơn giản, mạnh, cạnh tranh được LSTM.
    
# # #     Không dùng: PINN, continuity, spread, jerk, disp, dir, accel, smooth
# # #     Dùng: FM AFCRPS + MSE haversine per-step + SR + velocity + heading
# # #     """
# # #     import torch
# # #     import torch.nn.functional as F
 
# # #     if weights is None:
# # #         weights = WEIGHTS
 
# # #     NRM = 35.0
 
# # #     # Sample weights (recurvature + intensity)
# # #     recurv_w = _recurvature_weights(gt, w_recurv=2.0)
# # #     sample_w = recurv_w * (
# # #         intensity_w.to(gt.device) if intensity_w is not None else 1.0)
# # #     sample_w = sample_w / sample_w.mean().clamp(min=1e-6)
 
# # #     # ── FM AFCRPS ────────────────────────────────────────────────────
# # #     if pred_samples is not None and gt_norm is not None:
# # #         l_fm = fm_afcrps_loss(
# # #             pred_samples, gt_norm,
# # #             unit_01deg=True,
# # #             intensity_w=sample_w,
# # #             step_weight_alpha=step_weight_alpha,
# # #             w_es=0.2,
# # #         )
# # #     elif pred_samples is not None:
# # #         l_fm = _haversine_deg(pred_abs, gt).mean()
# # #     else:
# # #         l_fm = _haversine_deg(pred_abs, gt).mean()
 
# # #     # ── MSE Haversine per-step (KEY loss) ────────────────────────────
# # #     # Tính trên pred_abs (degrees) vs gt (degrees)
# # #     dist_km = _haversine_deg(pred_abs, gt)  # [T, B]
# # #     T = dist_km.shape[0]
# # #     sw = pred_abs.new_tensor(MSE_STEP_WEIGHTS[:T])
# # #     sw = sw / sw.sum() * T
# # #     l_mse_hav = (dist_km.pow(2) * sw.unsqueeze(1)).mean() / (200.0 ** 2)
 
# # #     # ── Directional ──────────────────────────────────────────────────
# # #     l_vel     = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
# # #     l_heading = heading_loss(pred_abs, gt)
# # #     l_recurv  = recurvature_loss(pred_abs, gt)
 
# # #     # ── Total ─────────────────────────────────────────────────────────
# # #     total = (
# # #         weights.get("fm",          2.0) * l_fm
# # #         + weights.get("mse_hav",   3.0) * l_mse_hav
# # #         + weights.get("velocity",  0.5) * l_vel     * NRM
# # #         + weights.get("heading",   1.0) * l_heading * NRM
# # #         + weights.get("recurv",    0.5) * l_recurv  * NRM
# # #     ) / NRM
 
# # #     if torch.isnan(total) or torch.isinf(total):
# # #         total = pred_abs.new_zeros(())
 
# # #     return dict(
# # #         total       = total,
# # #         fm          = l_fm.item(),
# # #         mse_hav     = l_mse_hav.item(),
# # #         velocity    = l_vel.item() * NRM,
# # #         heading     = l_heading.item(),
# # #         recurv      = l_recurv.item(),
# # #         # zeros để backward compat
# # #         step=0.0, disp=0.0, smooth=0.0, accel=0.0, jerk=0.0, pinn=0.0,
# # #         spread=0.0, continuity=0.0, recurv_ratio=0.0,
# # #     )

# # """
# # Model/losses.py — v34 HORIZON-AWARE + MULTI-SCALE
# # ═══════════════════════════════════════════════════════════════════
# # MỤC TIÊU: 12h<50, 24h<100, 48h<200, 72h<300 km

# # PHILOSOPHY:
# #   - Loss phải ALIGNED với metric: long-range error phải có gradient MẠNH hơn
# #   - Multi-scale: cùng lúc tính loss ở nhiều horizon nested (12h, 24h, 48h, 72h)
# #   - Trajectory-level: endpoint-weighted + shape consistency
# #   - KHÔNG dùng step weight flat như cũ

# # FIXES từ v33:
# #   BUG 1: Weights (w_vel, w_lr, w_head...) được DEFINE đầy đủ, không còn NameError
# #   BUG 2: lr_shape_loss nhận DEGREES không denorm lại (đã đúng từ v33fix nhưng clean lên)
# #   BUG 3: Tách rõ input types (normalized vs degrees) với naming convention

# # NEW IDEAS:
# #   1. Horizon-aware weighting: w[t] = (t+1)^1.5, renormalize
# #   2. Multi-scale haversine: L_12 + L_24 + L_48 + L_72
# #   3. Endpoint-weighted: terminal error weight cao nhất
# #   4. Trajectory shape loss: DTW-like local alignment
# #   5. Steering consistency: pred direction align với env 500hPa flow
# # """
# # from __future__ import annotations

# # import math
# # from typing import Dict, Optional, Tuple

# # import torch
# # import torch.nn.functional as F

# # __all__ = [
# #     "WEIGHTS", "compute_total_loss",
# #     "mse_haversine_horizon_aware", "multi_scale_haversine",
# #     "endpoint_weighted_loss", "trajectory_shape_loss",
# #     "velocity_loss_per_sample", "heading_loss",
# #     "recurvature_loss", "steering_alignment_loss",
# #     "_haversine", "_haversine_deg", "_norm_to_deg",
# # ]

# # # ── Constants ─────────────────────────────────────────────────────────────────
# # OMEGA     = 7.2921e-5
# # R_EARTH   = 6.371e6
# # DT_6H     = 6 * 3600
# # DEG_TO_KM = 111.0
# # STEP_KM   = 113.0

# # # ── Weights ─────────────────────────────────────────────────────────────────
# # # CHIẾN LƯỢC: Nhấn mạnh LONG-RANGE qua multi-scale + endpoint weighting
# # WEIGHTS: Dict[str, float] = dict(
# #     mse_hav_horizon = 3.0,   # Horizon-aware MSE haversine (chính)
# #     multi_scale     = 2.0,   # Multi-scale nested loss (12/24/48/72)
# #     endpoint        = 2.5,   # Endpoint-weighted terminal error
# #     shape           = 1.0,   # Shape consistency (local alignment)
# #     velocity        = 0.5,   # Velocity magnitude matching
# #     heading         = 1.0,   # Direction consistency
# #     recurv          = 0.5,   # Recurvature handling
# #     steering        = 0.3,   # Alignment with steering flow
# # )


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Haversine utilities
# # # ══════════════════════════════════════════════════════════════════════════════

# # def _haversine(p1: torch.Tensor, p2: torch.Tensor,
# #                unit_01deg: bool = True) -> torch.Tensor:
# #     """Haversine distance in km. p1, p2 shape [..., 2]."""
# #     if unit_01deg:
# #         lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
# #         lat1 = (p1[..., 1] * 50.0) / 10.0
# #         lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
# #         lat2 = (p2[..., 1] * 50.0) / 10.0
# #     else:
# #         lon1, lat1 = p1[..., 0], p1[..., 1]
# #         lon2, lat2 = p2[..., 0], p2[..., 1]

# #     lat1r = torch.deg2rad(lat1)
# #     lat2r = torch.deg2rad(lat2)
# #     dlon = torch.deg2rad(lon2 - lon1)
# #     dlat = torch.deg2rad(lat2 - lat1)
# #     a = (torch.sin(dlat / 2).pow(2)
# #          + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2).pow(2))
# #     a = a.clamp(1e-12, 1.0 - 1e-12)
# #     return 2.0 * 6371.0 * torch.asin(a.sqrt())


# # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# #     return _haversine(p1, p2, unit_01deg=False)


# # def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
# #     """Normalized [-∞, +∞] → degrees (lon, lat)."""
# #     out = arr.clone()
# #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# #     return out


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  ★ IDEA 1: Horizon-aware weighting (w[t] ~ (t+1)^α)
# # # ══════════════════════════════════════════════════════════════════════════════

# # def horizon_aware_weights(T: int, alpha: float = 1.5,
# #                            device=None) -> torch.Tensor:
# #     """
# #     Weight tăng theo horizon: w[t] = (t+1)^alpha, normalize để mean=1.
    
# #     Lý do: gradient cho step 72h phải MẠNH hơn step 6h,
# #     vì error xa thường lớn hơn và quan trọng hơn.
    
# #     alpha=1.5: step 12 (72h) có weight ~5.2x step 0 (6h)
# #     alpha=2.0: step 12 có weight ~12x (aggressive)
# #     alpha=1.0: linear
# #     """
# #     w = torch.arange(1, T + 1, dtype=torch.float32, device=device).pow(alpha)
# #     return w / w.mean()


# # def mse_haversine_horizon_aware(pred_deg: torch.Tensor,
# #                                  gt_deg: torch.Tensor,
# #                                  alpha: float = 1.5,
# #                                  huber_delta: float = 200.0) -> torch.Tensor:
# #     """
# #     MSE haversine với horizon-aware weighting.
    
# #     Inputs: DEGREES [T, B, 2]
# #     """
# #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# #     dist_km = _haversine_deg(pred_deg[:T], gt_deg[:T])  # [T, B]
# #     w = horizon_aware_weights(T, alpha, device=pred_deg.device)

# #     # Huber on distance to prevent extreme outliers dominating
# #     huber = torch.where(
# #         dist_km < huber_delta,
# #         0.5 * dist_km.pow(2) / huber_delta,
# #         dist_km - 0.5 * huber_delta,
# #     )
# #     return (huber * w.unsqueeze(1)).mean() / huber_delta


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  ★ IDEA 2: Multi-scale nested haversine
# # # ══════════════════════════════════════════════════════════════════════════════

# # def multi_scale_haversine(pred_deg: torch.Tensor,
# #                            gt_deg: torch.Tensor) -> torch.Tensor:
# #     """
# #     Multi-scale loss: tính ADE ở nhiều horizon nested.
    
# #     L = w_12·ADE(1-2) + w_24·ADE(1-4) + w_48·ADE(1-8) + w_72·ADE(1-12)
    
# #     Thứ tự weight: long-range > short-range để push model học long-range.
# #     Inputs: DEGREES [T, B, 2]
# #     """
# #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# #     if T < 2:
# #         return pred_deg.new_zeros(())

# #     dist_km = _haversine_deg(pred_deg[:T], gt_deg[:T])  # [T, B]

# #     # Scales và weights tương ứng
# #     scales = [
# #         (2,  0.10, 50.0),   # 12h, w=0.1, normalize 50km
# #         (4,  0.20, 100.0),  # 24h
# #         (8,  0.30, 200.0),  # 48h
# #         (12, 0.40, 300.0),  # 72h — weight lớn nhất
# #     ]

# #     total = pred_deg.new_zeros(())
# #     for h, w, norm in scales:
# #         if T >= h:
# #             ade_h = dist_km[:h].mean()
# #             total = total + w * (ade_h / norm)
# #     return total


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  ★ IDEA 3: Endpoint-weighted loss (FDE emphasis)
# # # ══════════════════════════════════════════════════════════════════════════════

# # def endpoint_weighted_loss(pred_deg: torch.Tensor,
# #                             gt_deg: torch.Tensor,
# #                             gamma: float = 2.0) -> torch.Tensor:
# #     """
# #     L = Σ (t/T)^gamma · d(pred_t, gt_t)
    
# #     Với gamma=2.0: step 12 weight = 1.0, step 6 weight = 0.25, step 1 weight ~ 0.007
# #     → model FOCUS hoàn toàn vào endpoint error.
    
# #     Inputs: DEGREES [T, B, 2]
# #     """
# #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# #     if T < 1:
# #         return pred_deg.new_zeros(())

# #     dist_km = _haversine_deg(pred_deg[:T], gt_deg[:T])  # [T, B]
# #     t_idx = torch.arange(1, T + 1, dtype=torch.float32, device=pred_deg.device)
# #     w = (t_idx / T).pow(gamma)
# #     w = w / w.sum()  # normalize
# #     return (dist_km * w.unsqueeze(1)).mean() / 300.0  # scale 300km


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  ★ IDEA 4: Trajectory shape loss (local alignment)
# # # ══════════════════════════════════════════════════════════════════════════════

# # def velocity_loss_per_sample(pred_deg: torch.Tensor,
# #                               gt_deg: torch.Tensor) -> torch.Tensor:
# #     """
# #     Velocity matching loss using Huber (robust to outliers).
# #     Input DEGREES. Output normalized to O(1).
# #     """
# #     if pred_deg.shape[0] < 2:
# #         return pred_deg.new_zeros(pred_deg.shape[1])
# #     v_pred = _step_displacements_km(pred_deg)
# #     v_gt = _step_displacements_km(gt_deg)

# #     T = v_pred.shape[0]
# #     w = horizon_aware_weights(T, alpha=1.3, device=pred_deg.device)

# #     # Huber speed loss
# #     s_pred = v_pred.norm(dim=-1)
# #     s_gt = v_gt.norm(dim=-1)
# #     speed_diff = (s_pred - s_gt).abs()
# #     huber_delta = 50.0  # km
# #     huber = torch.where(
# #         speed_diff < huber_delta,
# #         0.5 * speed_diff.pow(2) / huber_delta,
# #         speed_diff - 0.5 * huber_delta,
# #     )
# #     l_speed = (huber * w.unsqueeze(1)).mean(0) / huber_delta

# #     # ATE (along-track error) — same Huber treatment
# #     gn = v_gt.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# #     gt_unit = v_gt / gn
# #     ate = ((v_pred - v_gt) * gt_unit).sum(-1).abs()
# #     huber_ate = torch.where(
# #         ate < huber_delta,
# #         0.5 * ate.pow(2) / huber_delta,
# #         ate - 0.5 * huber_delta,
# #     )
# #     l_ate = (huber_ate * w.unsqueeze(1)).mean(0) / huber_delta

# #     return l_speed + 0.5 * l_ate


# # def trajectory_shape_loss(pred_deg: torch.Tensor,
# #                            gt_deg: torch.Tensor) -> torch.Tensor:
# #     """
# #     Shape consistency qua multiple displacement windows:
# #     - Short: (step 1→4), (step 4→8), (step 8→12)
# #     - Medium: (1→6), (6→12)
# #     - Long: (1→12) overall
    
# #     So sánh displacement vector (direction + magnitude) ở mỗi window.
    
# #     Inputs: DEGREES [T, B, 2]
# #     """
# #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# #     if T < 4:
# #         return pred_deg.new_zeros(())

# #     # Convert to km displacement (approximate)
# #     def disp_km(start, end):
# #         if end >= T:
# #             end = T - 1
# #         lat_mid = (pred_deg[start, :, 1] + pred_deg[end, :, 1]) * 0.5
# #         cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
# #         pred_d = pred_deg[end] - pred_deg[start]
# #         gt_d = gt_deg[end] - gt_deg[start]
# #         pred_km = torch.stack([
# #             pred_d[:, 0] * cos_lat * DEG_TO_KM,
# #             pred_d[:, 1] * DEG_TO_KM,
# #         ], dim=-1)
# #         gt_km = torch.stack([
# #             gt_d[:, 0] * cos_lat * DEG_TO_KM,
# #             gt_d[:, 1] * DEG_TO_KM,
# #         ], dim=-1)
# #         return pred_km, gt_km

# #     total = pred_deg.new_zeros(())
# #     count = 0
# #     windows = [
# #         (0, 3), (3, 7), (7, 11),   # short segments
# #         (0, 5), (5, 11),             # medium
# #         (0, 11),                      # long (overall shape)
# #     ]
# #     window_weights = [0.5, 0.8, 1.2, 0.7, 1.2, 1.5]

# #     for (s, e), w in zip(windows, window_weights):
# #         if e < T:
# #             pred_k, gt_k = disp_km(s, e)
# #             loss = F.smooth_l1_loss(pred_k, gt_k)
# #             total = total + w * loss
# #             count += w
# #     return (total / max(count, 1)) / (STEP_KM ** 2)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Velocity & Heading (on degrees)
# # # ══════════════════════════════════════════════════════════════════════════════

# # def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
# #     """[T,B,2] degrees → [T-1,B,2] km displacement per step."""
# #     dt = traj_deg[1:] - traj_deg[:-1]
# #     lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
# #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
# #     dt_km = dt.clone()
# #     dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
# #     dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
# #     return dt_km





# # def heading_loss(pred_deg: torch.Tensor,
# #                   gt_deg: torch.Tensor) -> torch.Tensor:
# #     """Direction consistency with horizon weighting."""
# #     if pred_deg.shape[0] < 2:
# #         return pred_deg.new_zeros(())

# #     pv = _step_displacements_km(pred_deg)
# #     gv = _step_displacements_km(gt_deg)
# #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)

# #     cos_sim = ((pv / pn) * (gv / gn)).sum(-1)  # [T-1, B]
# #     wrong_dir = F.relu(-cos_sim).pow(2)

# #     T = pv.shape[0]
# #     w = horizon_aware_weights(T, alpha=1.3, device=pred_deg.device)
# #     return (wrong_dir * w.unsqueeze(1)).mean()


# # def recurvature_loss(pred_deg: torch.Tensor,
# #                       gt_deg: torch.Tensor) -> torch.Tensor:
# #     """
# #     Curvature matching for recurving storms.
# #     Normalized to be O(0.1) so won't dominate total.
# #     """
# #     if pred_deg.shape[0] < 3:
# #         return pred_deg.new_zeros(())
# #     pv = pred_deg[1:] - pred_deg[:-1]
# #     gv = gt_deg[1:] - gt_deg[:-1]

# #     gt_cross = (gv[:-1, :, 0] * gv[1:, :, 1]
# #                 - gv[:-1, :, 1] * gv[1:, :, 0])
# #     pred_cross = (pv[:-1, :, 0] * pv[1:, :, 1]
# #                   - pv[:-1, :, 1] * pv[1:, :, 0])

# #     # Sign consistency (same turning direction, detach sign to avoid oscillation)
# #     with torch.no_grad():
# #         gt_sign = torch.sign(gt_cross)
# #     # smooth tanh instead of sign for pred
# #     pred_sign_soft = torch.tanh(pred_cross * 10.0)
# #     sign_loss = F.relu(-gt_sign * pred_sign_soft).pow(2)

# #     # Magnitude consistency, clamp to avoid explosion
# #     mag_loss = (pred_cross.abs() - gt_cross.abs()).clamp(-1.0, 1.0).pow(2)

# #     return (sign_loss + 0.1 * mag_loss).mean()


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  ★ IDEA 5: Steering alignment (physics prior via env data)
# # # ══════════════════════════════════════════════════════════════════════════════

# # def steering_alignment_loss(pred_deg: torch.Tensor,
# #                               env_data: Optional[dict]) -> torch.Tensor:
# #     """
# #     Encourage pred direction to ALIGN with 500hPa steering flow when strong.
# #     Only penalize when steering is clearly dominant (wind > 5 m/s).
# #     """
# #     if env_data is None or pred_deg.shape[0] < 2:
# #         return pred_deg.new_zeros(())

# #     T, B = pred_deg.shape[0] - 1, pred_deg.shape[1]
# #     device = pred_deg.device

# #     def _extract_uv(key_mean, key_center):
# #         v = env_data.get(key_center, env_data.get(key_mean, None))
# #         if v is None or not torch.is_tensor(v):
# #             return None
# #         v = v.to(device).float()
# #         if v.dim() == 3:
# #             v = v[..., 0]
# #         if v.dim() == 1:
# #             v = v.unsqueeze(0).expand(B, -1)
# #         v = v.permute(1, 0)  # [T_obs, B]
# #         T_obs = v.shape[0]
# #         if T_obs >= T:
# #             return v[:T] * 30.0  # denormalize m/s (approx)
# #         pad = torch.zeros(T - T_obs, B, device=device)
# #         return torch.cat([v * 30.0, pad], dim=0)

# #     u500 = _extract_uv("u500_mean", "u500_center")
# #     v500 = _extract_uv("v500_mean", "v500_center")
# #     if u500 is None or v500 is None:
# #         return pred_deg.new_zeros(())

# #     # Pred velocity in m/s
# #     dlat = pred_deg[1:] - pred_deg[:-1]
# #     lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# #     u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT_6H
# #     v_tc = dlat[:, :, 1] * 111000.0 / DT_6H

# #     # Steering strength
# #     uv_mag = torch.sqrt(u500.pow(2) + v500.pow(2))
# #     has_steering = (uv_mag > 5.0).float()

# #     # Cosine similarity between TC motion and steering
# #     env_dir = torch.stack([u500, v500], dim=-1)
# #     tc_dir = torch.stack([u_tc, v_tc], dim=-1)
# #     env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=1.0)
# #     tc_norm = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
# #     cos_sim = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)

# #     # Horizon weight: steering matters MORE for long-range
# #     w_t = horizon_aware_weights(T, alpha=1.5, device=device)
# #     misalign = F.relu(0.3 - cos_sim).pow(2)
# #     return (misalign * has_steering * w_t.unsqueeze(1)).mean()


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  MAIN LOSS AGGREGATOR
# # # ══════════════════════════════════════════════════════════════════════════════

# # def compute_total_loss(
# #     pred_deg: torch.Tensor,           # [T, B, 2] degrees
# #     gt_deg: torch.Tensor,             # [T, B, 2] degrees
# #     env_data: Optional[dict] = None,
# #     weights: Optional[dict] = None,
# #     intensity_w: Optional[torch.Tensor] = None,
# #     epoch: int = 0,
# #     **kwargs,
# # ) -> dict:
# #     """
# #     Aggregated loss với full breakdown.
# #     All inputs are DEGREES.
# #     """
# #     if weights is None:
# #         weights = WEIGHTS

# #     # Core losses
# #     l_mse_hor = mse_haversine_horizon_aware(pred_deg, gt_deg, alpha=1.5)
# #     l_multi = multi_scale_haversine(pred_deg, gt_deg)
# #     l_endpt = endpoint_weighted_loss(pred_deg, gt_deg, gamma=2.0)
# #     l_shape = trajectory_shape_loss(pred_deg, gt_deg)

# #     # Auxiliary losses
# #     l_vel = velocity_loss_per_sample(pred_deg, gt_deg).mean()
# #     l_head = heading_loss(pred_deg, gt_deg)
# #     l_recurv = recurvature_loss(pred_deg, gt_deg)
# #     l_steer = steering_alignment_loss(pred_deg, env_data)

# #     # Aggregate
# #     total = (
# #         weights["mse_hav_horizon"] * l_mse_hor
# #         + weights["multi_scale"]    * l_multi
# #         + weights["endpoint"]       * l_endpt
# #         + weights["shape"]          * l_shape
# #         + weights["velocity"]       * l_vel
# #         + weights["heading"]        * l_head
# #         + weights["recurv"]         * l_recurv
# #         + weights["steering"]       * l_steer
# #     )

# #     if torch.isnan(total) or torch.isinf(total):
# #         total = pred_deg.new_zeros(())

# #     return dict(
# #         total          = total,
# #         mse_hav_horizon = l_mse_hor.item(),
# #         multi_scale    = l_multi.item(),
# #         endpoint       = l_endpt.item(),
# #         shape          = l_shape.item(),
# #         velocity       = l_vel.item(),
# #         heading        = l_head.item(),
# #         recurv         = l_recurv.item(),
# #         steering       = l_steer.item(),
# #     )
# """
# Model/losses.py — v34fix HORIZON-AWARE + MULTI-SCALE (72h focused)
# ═══════════════════════════════════════════════════════════════════
# MỤC TIÊU: 12h<50, 24h<100, 48h<200, 72h<300 km

# FIXES v34→v34fix:
#   BUG 1: multi_scale_haversine dùng ADE[:h].mean() — gradient cho step 12
#          rất yếu nếu step 1-8 đã tốt. Fix: dùng 70% error tại ĐÚNG step h
#          + 30% ADE[:h] để balance local vs endpoint.
#   BUG 2: WEIGHTS multi_scale=2.0 quá thấp → tăng lên 3.5
#   BUG 3: horizon_aware_weights alpha=1.5 → tăng lên 2.0 để step 12
#          có gradient mạnh hơn (9x thay vì 5x so với step 1)
#   BUG 4: endpoint_weighted_loss gamma=2.0 → 2.5, norm 300→250
#          để gradient lớn hơn ở long-range

# PHILOSOPHY:
#   - Loss phải ALIGNED với metric: long-range error phải có gradient MẠNH hơn
#   - Multi-scale: tập trung vào ERROR TẠI ĐÚNG HORIZON, không phải ADE mean
#   - Trajectory-level: endpoint-weighted + shape consistency
# """
# from __future__ import annotations

# import math
# from typing import Dict, Optional, Tuple

# import torch
# import torch.nn.functional as F

# __all__ = [
#     "WEIGHTS", "compute_total_loss",
#     "mse_haversine_horizon_aware", "multi_scale_haversine",
#     "endpoint_weighted_loss", "trajectory_shape_loss",
#     "velocity_loss_per_sample", "heading_loss",
#     "recurvature_loss", "steering_alignment_loss",
#     "_haversine", "_haversine_deg", "_norm_to_deg",
# ]

# # ── Constants ─────────────────────────────────────────────────────────────────
# OMEGA     = 7.2921e-5
# R_EARTH   = 6.371e6
# DT_6H     = 6 * 3600
# DEG_TO_KM = 111.0
# STEP_KM   = 113.0

# # ── Weights — FIX: tăng multi_scale và endpoint, giảm mse_hav ───────────────
# # mse_hav_horizon dùng ADE mean nên ít targeted; multi_scale dùng endpoint error
# WEIGHTS: Dict[str, float] = dict(
#     mse_hav_horizon = 2.0,   # giảm từ 3.0 (mean-based, ít targeted 72h)
#     multi_scale     = 3.5,   # TĂNG từ 2.0 — endpoint-focused, push 72h mạnh nhất
#     endpoint        = 3.0,   # TĂNG từ 2.5 — FDE emphasis
#     shape           = 0.8,   # giảm nhẹ từ 1.0
#     velocity        = 0.3,   # giảm từ 0.5 — auxiliary only
#     heading         = 0.8,   # giảm nhẹ từ 1.0
#     recurv          = 0.3,   # giảm từ 0.5 — auxiliary only
#     steering        = 0.3,   # giữ nguyên
# )


# # ══════════════════════════════════════════════════════════════════════════════
# #  Haversine utilities
# # ══════════════════════════════════════════════════════════════════════════════

# def _haversine(p1: torch.Tensor, p2: torch.Tensor,
#                unit_01deg: bool = True) -> torch.Tensor:
#     """Haversine distance in km. p1, p2 shape [..., 2]."""
#     if unit_01deg:
#         lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
#         lat1 = (p1[..., 1] * 50.0) / 10.0
#         lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
#         lat2 = (p2[..., 1] * 50.0) / 10.0
#     else:
#         lon1, lat1 = p1[..., 0], p1[..., 1]
#         lon2, lat2 = p2[..., 0], p2[..., 1]

#     lat1r = torch.deg2rad(lat1)
#     lat2r = torch.deg2rad(lat2)
#     dlon = torch.deg2rad(lon2 - lon1)
#     dlat = torch.deg2rad(lat2 - lat1)
#     a = (torch.sin(dlat / 2).pow(2)
#          + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2).pow(2))
#     a = a.clamp(1e-12, 1.0 - 1e-12)
#     return 2.0 * 6371.0 * torch.asin(a.sqrt())


# def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
#     return _haversine(p1, p2, unit_01deg=False)


# def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
#     """Normalized [-∞, +∞] → degrees (lon, lat)."""
#     out = arr.clone()
#     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
#     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
#     return out


# # ══════════════════════════════════════════════════════════════════════════════
# #  Horizon-aware weighting — FIX: alpha 1.5 → 2.0
# # ══════════════════════════════════════════════════════════════════════════════

# def horizon_aware_weights(T: int, alpha: float = 2.0,
#                            device=None) -> torch.Tensor:
#     """
#     Weight tăng theo horizon: w[t] = (t+1)^alpha, normalize để mean=1.

#     FIX: alpha 1.5 → 2.0
#     alpha=2.0: step 12 có weight ~9x step 1 (thay vì ~5x với 1.5)
#     → gradient cho 72h mạnh hơn đáng kể.
#     """
#     w = torch.arange(1, T + 1, dtype=torch.float32, device=device).pow(alpha)
#     return w / w.mean()


# def mse_haversine_horizon_aware(pred_deg: torch.Tensor,
#                                  gt_deg: torch.Tensor,
#                                  alpha: float = 2.0,
#                                  huber_delta: float = 200.0) -> torch.Tensor:
#     """
#     MSE haversine với horizon-aware weighting.
#     Inputs: DEGREES [T, B, 2]
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     dist_km = _haversine_deg(pred_deg[:T], gt_deg[:T])  # [T, B]
#     w = horizon_aware_weights(T, alpha, device=pred_deg.device)

#     huber = torch.where(
#         dist_km < huber_delta,
#         0.5 * dist_km.pow(2) / huber_delta,
#         dist_km - 0.5 * huber_delta,
#     )
#     return (huber * w.unsqueeze(1)).mean() / huber_delta


# # ══════════════════════════════════════════════════════════════════════════════
# #  Multi-scale haversine — FIX: dùng endpoint error, không phải ADE mean
# # ══════════════════════════════════════════════════════════════════════════════

# def multi_scale_haversine(pred_deg: torch.Tensor,
#                            gt_deg: torch.Tensor) -> torch.Tensor:
#     """
#     Multi-scale loss tập trung vào error tại ĐÚNG horizon step.

#     FIX v34fix: thay ADE[:h].mean() bằng 70% error[h] + 30% ADE[:h+1]
#     Lý do: ADE mean làm loãng gradient của step 12 nếu step 1-8 đã tốt.
#     Với cách mới: gradient trực tiếp từ error tại step 12 (72h) chiếm
#     weight=0.52 × 0.7 = 0.364 của tổng loss này.

#     Inputs: DEGREES [T, B, 2]
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         return pred_deg.new_zeros(())

#     dist_km = _haversine_deg(pred_deg[:T], gt_deg[:T])  # [T, B]

#     # (horizon_step_idx, weight, normalize_km)
#     # FIX: dùng index step (0-based) thay vì count
#     # step idx 1 = 12h, 3 = 24h, 7 = 48h, 11 = 72h
#     scales = [
#         (1,  0.08, 50.0),    # 12h
#         (3,  0.15, 100.0),   # 24h
#         (7,  0.25, 200.0),   # 48h
#         (11, 0.52, 300.0),   # 72h — weight lớn nhất
#     ]

#     total = pred_deg.new_zeros(())
#     for h, w, norm in scales:
#         if T > h:
#             # 70% error tại đúng step h, 30% ADE tới step h
#             endpoint_err = dist_km[h].mean()
#             ade_err      = dist_km[:h+1].mean()
#             combined     = 0.7 * endpoint_err + 0.3 * ade_err
#             total = total + w * (combined / norm)
#     return total


# # ══════════════════════════════════════════════════════════════════════════════
# #  Endpoint-weighted loss — FIX: gamma 2.0→2.5, norm 300→250
# # ══════════════════════════════════════════════════════════════════════════════

# def endpoint_weighted_loss(pred_deg: torch.Tensor,
#                             gt_deg: torch.Tensor,
#                             gamma: float = 2.5) -> torch.Tensor:
#     """
#     L = Σ (t/T)^gamma · d(pred_t, gt_t)

#     FIX: gamma 2.0→2.5, normalize 300→250 km
#     Với gamma=2.5: step 12 weight = 1.0, step 6 weight = 0.177, step 1 ≈ 0.003
#     → model tập trung hoàn toàn vào endpoint error.
#     Norm 250 thay vì 300 → gradient lớn hơn khi error gần target.

#     Inputs: DEGREES [T, B, 2]
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 1:
#         return pred_deg.new_zeros(())

#     dist_km = _haversine_deg(pred_deg[:T], gt_deg[:T])  # [T, B]
#     t_idx = torch.arange(1, T + 1, dtype=torch.float32, device=pred_deg.device)
#     w = (t_idx / T).pow(gamma)
#     w = w / w.sum()
#     return (dist_km * w.unsqueeze(1)).mean() / 250.0  # FIX: 300→250


# # ══════════════════════════════════════════════════════════════════════════════
# #  Trajectory shape loss (local alignment) — không thay đổi
# # ══════════════════════════════════════════════════════════════════════════════

# def velocity_loss_per_sample(pred_deg: torch.Tensor,
#                               gt_deg: torch.Tensor) -> torch.Tensor:
#     """Velocity matching loss using Huber (robust to outliers)."""
#     if pred_deg.shape[0] < 2:
#         return pred_deg.new_zeros(pred_deg.shape[1])
#     v_pred = _step_displacements_km(pred_deg)
#     v_gt   = _step_displacements_km(gt_deg)

#     T = v_pred.shape[0]
#     w = horizon_aware_weights(T, alpha=1.3, device=pred_deg.device)

#     s_pred = v_pred.norm(dim=-1)
#     s_gt   = v_gt.norm(dim=-1)
#     speed_diff = (s_pred - s_gt).abs()
#     huber_delta = 50.0
#     huber = torch.where(
#         speed_diff < huber_delta,
#         0.5 * speed_diff.pow(2) / huber_delta,
#         speed_diff - 0.5 * huber_delta,
#     )
#     l_speed = (huber * w.unsqueeze(1)).mean(0) / huber_delta

#     gn = v_gt.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     gt_unit = v_gt / gn
#     ate = ((v_pred - v_gt) * gt_unit).sum(-1).abs()
#     huber_ate = torch.where(
#         ate < huber_delta,
#         0.5 * ate.pow(2) / huber_delta,
#         ate - 0.5 * huber_delta,
#     )
#     l_ate = (huber_ate * w.unsqueeze(1)).mean(0) / huber_delta

#     return l_speed + 0.5 * l_ate


# def trajectory_shape_loss(pred_deg: torch.Tensor,
#                            gt_deg: torch.Tensor) -> torch.Tensor:
#     """
#     Shape consistency qua multiple displacement windows.
#     Inputs: DEGREES [T, B, 2]
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 4:
#         return pred_deg.new_zeros(())

#     def disp_km(start, end):
#         if end >= T:
#             end = T - 1
#         lat_mid = (pred_deg[start, :, 1] + pred_deg[end, :, 1]) * 0.5
#         cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
#         pred_d = pred_deg[end] - pred_deg[start]
#         gt_d   = gt_deg[end] - gt_deg[start]
#         pred_km = torch.stack([
#             pred_d[:, 0] * cos_lat * DEG_TO_KM,
#             pred_d[:, 1] * DEG_TO_KM,
#         ], dim=-1)
#         gt_km = torch.stack([
#             gt_d[:, 0] * cos_lat * DEG_TO_KM,
#             gt_d[:, 1] * DEG_TO_KM,
#         ], dim=-1)
#         return pred_km, gt_km

#     total = pred_deg.new_zeros(())
#     count = 0
#     windows = [
#         (0, 3), (3, 7), (7, 11),
#         (0, 5), (5, 11),
#         (0, 11),
#     ]
#     window_weights = [0.5, 0.8, 1.2, 0.7, 1.2, 1.5]

#     for (s, e), w in zip(windows, window_weights):
#         if e < T:
#             pred_k, gt_k = disp_km(s, e)
#             loss = F.smooth_l1_loss(pred_k, gt_k)
#             total = total + w * loss
#             count += w
#     return (total / max(count, 1)) / (STEP_KM ** 2)


# # ══════════════════════════════════════════════════════════════════════════════
# #  Velocity & Heading
# # ══════════════════════════════════════════════════════════════════════════════

# def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
#     """[T,B,2] degrees → [T-1,B,2] km displacement per step."""
#     dt = traj_deg[1:] - traj_deg[:-1]
#     lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
#     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
#     dt_km = dt.clone()
#     dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
#     dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
#     return dt_km


# def heading_loss(pred_deg: torch.Tensor,
#                   gt_deg: torch.Tensor) -> torch.Tensor:
#     """Direction consistency with horizon weighting."""
#     if pred_deg.shape[0] < 2:
#         return pred_deg.new_zeros(())

#     pv = _step_displacements_km(pred_deg)
#     gv = _step_displacements_km(gt_deg)
#     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)

#     cos_sim   = ((pv / pn) * (gv / gn)).sum(-1)
#     wrong_dir = F.relu(-cos_sim).pow(2)

#     T = pv.shape[0]
#     w = horizon_aware_weights(T, alpha=1.3, device=pred_deg.device)
#     return (wrong_dir * w.unsqueeze(1)).mean()


# def recurvature_loss(pred_deg: torch.Tensor,
#                       gt_deg: torch.Tensor) -> torch.Tensor:
#     """Curvature matching for recurving storms."""
#     if pred_deg.shape[0] < 3:
#         return pred_deg.new_zeros(())
#     pv = pred_deg[1:] - pred_deg[:-1]
#     gv = gt_deg[1:] - gt_deg[:-1]

#     gt_cross = (gv[:-1, :, 0] * gv[1:, :, 1]
#                 - gv[:-1, :, 1] * gv[1:, :, 0])
#     pred_cross = (pv[:-1, :, 0] * pv[1:, :, 1]
#                   - pv[:-1, :, 1] * pv[1:, :, 0])

#     with torch.no_grad():
#         gt_sign = torch.sign(gt_cross)
#     pred_sign_soft = torch.tanh(pred_cross * 10.0)
#     sign_loss = F.relu(-gt_sign * pred_sign_soft).pow(2)

#     mag_loss = (pred_cross.abs() - gt_cross.abs()).clamp(-1.0, 1.0).pow(2)

#     return (sign_loss + 0.1 * mag_loss).mean()


# # ══════════════════════════════════════════════════════════════════════════════
# #  Steering alignment
# # ══════════════════════════════════════════════════════════════════════════════

# def steering_alignment_loss(pred_deg: torch.Tensor,
#                               env_data: Optional[dict]) -> torch.Tensor:
#     """Encourage pred direction to align with 500hPa steering flow."""
#     if env_data is None or pred_deg.shape[0] < 2:
#         return pred_deg.new_zeros(())

#     T, B = pred_deg.shape[0] - 1, pred_deg.shape[1]
#     device = pred_deg.device

#     def _extract_uv(key_mean, key_center):
#         v = env_data.get(key_center, env_data.get(key_mean, None))
#         if v is None or not torch.is_tensor(v):
#             return None
#         v = v.to(device).float()
#         if v.dim() == 3:
#             v = v[..., 0]
#         if v.dim() == 1:
#             v = v.unsqueeze(0).expand(B, -1)
#         v = v.permute(1, 0)
#         T_obs = v.shape[0]
#         if T_obs >= T:
#             return v[:T] * 30.0
#         pad = torch.zeros(T - T_obs, B, device=device)
#         return torch.cat([v * 30.0, pad], dim=0)

#     u500 = _extract_uv("u500_mean", "u500_center")
#     v500 = _extract_uv("v500_mean", "v500_center")
#     if u500 is None or v500 is None:
#         return pred_deg.new_zeros(())

#     dlat    = pred_deg[1:] - pred_deg[:-1]
#     lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
#     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
#     u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT_6H
#     v_tc = dlat[:, :, 1] * 111000.0 / DT_6H

#     uv_mag      = torch.sqrt(u500.pow(2) + v500.pow(2))
#     has_steering = (uv_mag > 5.0).float()

#     env_dir  = torch.stack([u500, v500], dim=-1)
#     tc_dir   = torch.stack([u_tc, v_tc], dim=-1)
#     env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=1.0)
#     tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
#     cos_sim  = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)

#     w_t = horizon_aware_weights(T, alpha=1.5, device=device)
#     misalign = F.relu(0.3 - cos_sim).pow(2)
#     return (misalign * has_steering * w_t.unsqueeze(1)).mean()


# # ══════════════════════════════════════════════════════════════════════════════
# #  MAIN LOSS AGGREGATOR
# # ══════════════════════════════════════════════════════════════════════════════

# def compute_total_loss(
#     pred_deg: torch.Tensor,
#     gt_deg: torch.Tensor,
#     env_data: Optional[dict] = None,
#     weights: Optional[dict] = None,
#     intensity_w: Optional[torch.Tensor] = None,
#     epoch: int = 0,
#     **kwargs,
# ) -> dict:
#     """
#     Aggregated loss với full breakdown.
#     All inputs are DEGREES.
#     """
#     if weights is None:
#         weights = WEIGHTS

#     l_mse_hor = mse_haversine_horizon_aware(pred_deg, gt_deg, alpha=2.0)
#     l_multi   = multi_scale_haversine(pred_deg, gt_deg)
#     l_endpt   = endpoint_weighted_loss(pred_deg, gt_deg, gamma=2.5)
#     l_shape   = trajectory_shape_loss(pred_deg, gt_deg)

#     l_vel   = velocity_loss_per_sample(pred_deg, gt_deg).mean()
#     l_head  = heading_loss(pred_deg, gt_deg)
#     l_recurv = recurvature_loss(pred_deg, gt_deg)
#     l_steer = steering_alignment_loss(pred_deg, env_data)

#     total = (
#         weights["mse_hav_horizon"] * l_mse_hor
#         + weights["multi_scale"]   * l_multi
#         + weights["endpoint"]      * l_endpt
#         + weights["shape"]         * l_shape
#         + weights["velocity"]      * l_vel
#         + weights["heading"]       * l_head
#         + weights["recurv"]        * l_recurv
#         + weights["steering"]      * l_steer
#     )

#     if torch.isnan(total) or torch.isinf(total):
#         total = pred_deg.new_zeros(())

#     return dict(
#         total           = total,
#         mse_hav_horizon = l_mse_hor.item(),
#         multi_scale     = l_multi.item(),
#         endpoint        = l_endpt.item(),
#         shape           = l_shape.item(),
#         velocity        = l_vel.item(),
#         heading         = l_head.item(),
#         recurv          = l_recurv.item(),
#         steering        = l_steer.item(),
#     )

"""
Model/losses.py — v35fix-v2  CALIBRATED MSE + CORRECT SCALE
═══════════════════════════════════════════════════════════════════
MỤC TIÊU: 12h<50, 24h<100, 48h<200, 72h<300 km

BUG FIX so với v35fix-v1 (bản trước):
  BUG 1: horizon_aware_mse dùng dist^2/200^2
         → tại ep0 dist=500km: 500^2×3.04/40000 = 19 per-step dominant
         → mean ≈ 4.32, ×2.0 = 8.64  ❌ (v34fix chỉ 2.93)
         FIX: dùng direct_multi_horizon_mse — normalize RIÊNG từng horizon

  BUG 2: Thêm 2 term mới (direct_mse + focal_72h) không bù trừ đủ
         → net loss cao hơn v34fix ~3-5x tại cùng checkpoint
         FIX: dùng multi_horizon làm loss DUY NHẤT cho accuracy,
              focal_72h giữ nhẹ (×0.8)

BUG FIX so với v35 gốc (vẫn giữ):
  BUG 3: direct_endpoint_mse chia 111² thay vì 300² → dm=40 ❌
         (đã fix trong v1, giữ lại)

LOSS SCALE SO VỚI v34fix:
  v34fix tại ep0: tot ≈ 13-14  (Huber-based)
  v35fix-v2 tại ep0: tot ≈ 12-14  ← CỰC GẦN v34fix ✅
  v34fix tại ep63 (ADE=172km): tot ≈ 2-3
  v35fix-v2 tại ep63: tot ≈ 4-5   ← cao hơn 1 chút, chấp nhận được ✅

PHILOSOPHY — ĐÁNH BẠI LSTM:
  1. direct_multi_horizon_mse: MSE thuần, normalize per-horizon
     → gradient = 2*(pred-gt)/target^2, cực clean như LSTM
     → 72h chiếm 60% weight, không bị loãng bởi các step nhỏ
  2. focal_72h: nhẹ, chỉ push khi error > 300km (LSTM không có điều này)
  3. heading: tránh trajectory zig-zag
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

__all__ = [
    "WEIGHTS", "compute_total_loss",
    "direct_multi_horizon_mse", "focal_72h_loss",
    "trajectory_shape_loss", "heading_loss",
    "steering_alignment_loss",
    "_haversine", "_haversine_deg", "_norm_to_deg",
]

# ── Constants ─────────────────────────────────────────────────────────────────
R_EARTH   = 6371.0
DT_6H     = 6 * 3600
DEG_TO_KM = 111.0
STEP_KM   = 113.0

# ── Weights — calibrated để tổng ≈ 12-14 tại ep0 ────────────────────────────
#
# Budget tại ep0 (d1≈50, d3≈100, d7≈300, d11≈500 km):
#   multi_horizon ≈ 2.38  × 3.0 =  7.14
#   focal_72h     ≈ 3.58  × 0.8 =  2.86
#   shape         ≈ 0.15  × 0.3 =  0.05
#   heading       ≈ 0.30  × 0.3 =  0.09
#   steering      ≈ 0.10  × 0.15=  0.02
#   fm_mse (từ model) ≈ 0.3-1.0
#   ───────────────────────────────────
#   TOTAL  ≈ 10.2 + fm_mse ≈ 11-12   ✅ (gần v34fix ≈ 13-14)
#
# Budget tại ep63 (d1≈30, d3≈80, d7≈200, d11≈331 km):
#   multi_horizon ≈ 1.06  × 3.0 =  3.18
#   focal_72h     ≈ 1.28  × 0.8 =  1.02
#   shape+head    ≈ 0.10  × 0.3 =  0.06 (mỗi)
#   ───────────────────────────────────
#   TOTAL ≈ 4.3 + fm_mse×0.3 ≈ 4.5   ✅ (cao hơn v34fix ≈ 2.5, chấp nhận)

WEIGHTS: Dict[str, float] = dict(
    multi_horizon = 3.0,   # MSE per-horizon, 60% vào 72h
    focal_72h     = 0.8,   # Focal push khi error > 300km
    shape         = 0.3,   # Trajectory shape
    heading       = 0.3,   # Direction consistency
    steering      = 0.15,  # 500hPa alignment
)


# ══════════════════════════════════════════════════════════════════════════════
#  Haversine utilities
# ══════════════════════════════════════════════════════════════════════════════

def _haversine(p1: torch.Tensor, p2: torch.Tensor,
               unit_01deg: bool = True) -> torch.Tensor:
    if unit_01deg:
        lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
        lat1 = (p1[..., 1] * 50.0) / 10.0
        lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
        lat2 = (p2[..., 1] * 50.0) / 10.0
    else:
        lon1, lat1 = p1[..., 0], p1[..., 1]
        lon2, lat2 = p2[..., 0], p2[..., 1]
    lat1r, lat2r = torch.deg2rad(lat1), torch.deg2rad(lat2)
    dlon, dlat   = torch.deg2rad(lon2 - lon1), torch.deg2rad(lat2 - lat1)
    a = (torch.sin(dlat / 2).pow(2)
         + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2).pow(2))
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    return _haversine(p1, p2, unit_01deg=False)


def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
    out = arr.clone()
    out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
    out[..., 1] = (arr[..., 1] * 50.0) / 10.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  1. MAIN LOSS: Direct MSE per horizon — normalize riêng từng horizon
#     Đây là cốt lõi để đánh bại LSTM
# ══════════════════════════════════════════════════════════════════════════════

def direct_multi_horizon_mse(pred_deg: torch.Tensor,
                              gt_deg: torch.Tensor) -> torch.Tensor:
    """
    MSE thuần, normalize theo target của TỪNG horizon.

    TẠI SAO ĐÚNG HƠN horizon_aware_mse:
      horizon_aware_mse dùng dist^2/200^2 cho MỌI step → step 12 (dist=500km)
      cho 500^2/200^2=6.25, overpower hẳn. Normalization sai.

      direct_multi_horizon_mse dùng dist^2/target_h^2 riêng từng step:
      - step 12: 500^2/300^2 = 2.78  ← đúng scale
      - step 7: 300^2/200^2 = 2.25   ← đúng scale
      - step 3: 100^2/100^2 = 1.0    ← đúng scale
      Tổng có trọng số: 2.38 tại ep0 → 1.06 tại ep63  ✅

    GRADIENT = 2*(pred-gt)/target_h^2 → clean như LSTM nhưng focus vào 72h.

    Inputs: DEGREES [T, B, 2]
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        return pred_deg.new_zeros(())

    # (step_idx 0-based, weight, target_km)
    # 72h chiếm 60% gradient → mạnh nhất
    horizons = [
        (1,  0.05,  50.0),   # 12h — ít thôi, thường không phải bottleneck
        (3,  0.10, 100.0),   # 24h
        (7,  0.25, 200.0),   # 48h
        (11, 0.60, 300.0),   # 72h — chủ đạo
    ]

    total = pred_deg.new_zeros(())
    for step, w, tgt in horizons:
        if T > step:
            d = _haversine_deg(pred_deg[step], gt_deg[step])   # [B] km
            total = total + w * d.pow(2).mean() / (tgt ** 2)

    return total  # ≈ 1.0 khi error = target tại mọi horizon


# ══════════════════════════════════════════════════════════════════════════════
#  2. Focal 72h — nhẹ, chỉ push khi error > 300km (LSTM không có)
# ══════════════════════════════════════════════════════════════════════════════

def focal_72h_loss(pred_deg: torch.Tensor,
                    gt_deg: torch.Tensor,
                    target_km: float = 300.0,
                    focal_exp: float = 1.5) -> torch.Tensor:
    """
    Focal penalty tại step 12:
      loss = d × (d/target)^focal_exp / target

    Khi d > target: focal_w > 1 → push harder (advantage over LSTM)
    Khi d < target: focal_w = 1 → không bị over-penalize (MAE scale, linear)

    Scale:
      Tại ep0 d=500km: (500/300)^1.5=2.15, 500*2.15/300=3.58
      Tại ep63 d=331km: (331/300)^1.5=1.16, 331*1.16/300=1.28
      Tại target d=300km: 1.0
    """
    if pred_deg.shape[0] < 12:
        return pred_deg.new_zeros(())

    d = _haversine_deg(pred_deg[11], gt_deg[11])                # [B] km
    with torch.no_grad():
        focal_w = torch.where(d > target_km,
                               (d / target_km).pow(focal_exp),
                               torch.ones_like(d))
    return (d * focal_w).mean() / target_km


# ══════════════════════════════════════════════════════════════════════════════
#  3. Trajectory shape (nhẹ — chỉ tránh degenerate path)
# ══════════════════════════════════════════════════════════════════════════════

def trajectory_shape_loss(pred_deg: torch.Tensor,
                           gt_deg: torch.Tensor) -> torch.Tensor:
    """Shape displacement windows. Inputs: DEG [T,B,2]"""
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 4:
        return pred_deg.new_zeros(())

    def _disp_km(s, e):
        e = min(e, T - 1)
        lat_mid = (pred_deg[s, :, 1] + pred_deg[e, :, 1]) * 0.5
        cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
        pd = pred_deg[e] - pred_deg[s]
        gd = gt_deg[e]   - gt_deg[s]
        pk = torch.stack([pd[:, 0] * cos_lat * DEG_TO_KM, pd[:, 1] * DEG_TO_KM], -1)
        gk = torch.stack([gd[:, 0] * cos_lat * DEG_TO_KM, gd[:, 1] * DEG_TO_KM], -1)
        return pk, gk

    total, count = pred_deg.new_zeros(()), 0.0
    for s, e, w in [(0, 5, 0.5), (5, 11, 1.0), (0, 11, 1.5)]:
        if e < T:
            pk, gk = _disp_km(s, e)
            total  = total + w * F.smooth_l1_loss(pk, gk)
            count  += w
    return (total / max(count, 1.0)) / (STEP_KM ** 2)


# ══════════════════════════════════════════════════════════════════════════════
#  4. Heading loss (nhẹ — tránh zig-zag)
# ══════════════════════════════════════════════════════════════════════════════

def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
    dt      = traj_deg[1:] - traj_deg[:-1]
    lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
    cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
    dt_km   = dt.clone()
    dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
    dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
    return dt_km


def heading_loss(pred_deg: torch.Tensor,
                  gt_deg: torch.Tensor) -> torch.Tensor:
    """Direction consistency. Inputs: DEG [T,B,2]"""
    if pred_deg.shape[0] < 2:
        return pred_deg.new_zeros(())
    pv = _step_displacements_km(pred_deg)
    gv = _step_displacements_km(gt_deg)
    pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    cos_sim   = ((pv / pn) * (gv / gn)).sum(-1)
    wrong_dir = F.relu(-cos_sim).pow(2)
    T = pv.shape[0]
    w = torch.arange(1, T + 1, dtype=torch.float32,
                     device=pred_deg.device).pow(1.5)
    w = w / w.mean()
    return (wrong_dir * w.unsqueeze(1)).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  5. Steering alignment (nhẹ)
# ══════════════════════════════════════════════════════════════════════════════

def steering_alignment_loss(pred_deg: torch.Tensor,
                              env_data: Optional[dict]) -> torch.Tensor:
    if env_data is None or pred_deg.shape[0] < 2:
        return pred_deg.new_zeros(())
    T, B   = pred_deg.shape[0] - 1, pred_deg.shape[1]
    device = pred_deg.device

    def _extract(k1, k2):
        v = env_data.get(k2, env_data.get(k1, None))
        if v is None or not torch.is_tensor(v): return None
        v = v.to(device).float()
        if v.dim() == 3: v = v[..., 0]
        if v.dim() == 1: v = v.unsqueeze(0).expand(B, -1)
        v = v.permute(1, 0)
        T_obs = v.shape[0]
        if T_obs >= T: return v[:T] * 30.0
        return torch.cat([v * 30.0, torch.zeros(T - T_obs, B, device=device)], 0)

    u500 = _extract("u500_mean", "u500_center")
    v500 = _extract("v500_mean", "v500_center")
    if u500 is None or v500 is None: return pred_deg.new_zeros(())

    dlat    = pred_deg[1:] - pred_deg[:-1]
    lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
    cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
    u_tc    = dlat[:, :, 0] * cos_lat * 111000.0 / DT_6H
    v_tc    = dlat[:, :, 1] * 111000.0 / DT_6H

    uv_mag       = (u500.pow(2) + v500.pow(2)).sqrt()
    has_steering = (uv_mag > 5.0).float()
    env_dir      = torch.stack([u500, v500], -1)
    tc_dir       = torch.stack([u_tc, v_tc], -1)
    cos_sim      = ((env_dir / env_dir.norm(dim=-1, keepdim=True).clamp(min=1.0))
                    * (tc_dir / tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5))).sum(-1)

    w_t      = torch.arange(1, T + 1, dtype=torch.float32, device=device).pow(1.5)
    w_t      = w_t / w_t.mean()
    misalign = F.relu(0.3 - cos_sim).pow(2)
    return (misalign * has_steering * w_t.unsqueeze(1)).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOSS AGGREGATOR
# ══════════════════════════════════════════════════════════════════════════════

def compute_total_loss(
    pred_deg: torch.Tensor,
    gt_deg: torch.Tensor,
    env_data: Optional[dict] = None,
    weights: Optional[dict] = None,
    epoch: int = 0,
    **kwargs,
) -> dict:
    """
    Aggregated loss. All inputs: DEGREES [T, B, 2].

    Loss scale so sánh:
                      ep0    ep63
      v34fix tot:    13-14   2-3
      v35fix-v2 tot: 11-13   4-5  ← gần v34fix hơn nhiều so với v35fix-v1
    """
    if weights is None:
        weights = WEIGHTS

    l_multi  = direct_multi_horizon_mse(pred_deg, gt_deg)
    l_focal  = focal_72h_loss(pred_deg, gt_deg, target_km=300.0, focal_exp=1.5)
    l_shape  = trajectory_shape_loss(pred_deg, gt_deg)
    l_head   = heading_loss(pred_deg, gt_deg)
    l_steer  = steering_alignment_loss(pred_deg, env_data)

    total = (
        weights["multi_horizon"] * l_multi
        + weights["focal_72h"]   * l_focal
        + weights["shape"]       * l_shape
        + weights["heading"]     * l_head
        + weights["steering"]    * l_steer
    )

    if torch.isnan(total) or torch.isinf(total):
        total = pred_deg.new_zeros(())

    # Individual values để log — giữ key names tương thích với train script
    l_multi_val = l_multi.item()
    l_focal_val = l_focal.item()
    return dict(
        total           = total,
        # Keys cho log trong train script (flow_matching_model.py)
        mse_hav_horizon = l_multi_val,   # thay cho mse_hav_horizon cũ
        mse_hav         = l_multi_val,
        multi_scale     = l_focal_val,   # dùng slot ms để log focal
        endpoint        = 0.0,
        shape           = l_shape.item(),
        heading         = l_head.item(),
        velocity        = 0.0,
        recurv          = 0.0,
        steering        = l_steer.item(),
        # Keys mới
        direct_mse      = l_multi_val,
        hard_72h        = l_focal_val,
    )
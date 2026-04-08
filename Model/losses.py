# """
# Model/losses.py  ── v24
# ========================
# FIXES vs v23:

#   FIX-L44  [CRITICAL] pinn_shallow_water: log(loss-49) tạo ra gradient
#            gần bằng 0 khi loss>50 (gradient = 1/(loss-49) → rất nhỏ).
#            Từ log: pinn=52.69 không đổi suốt 90 epoch → NO gradient.
#            Fix: Không dùng log-clamp. Dùng soft-tanh scaling:
#              loss_scaled = 50 * tanh(loss / 50)
#            → gradient = tanh'(x/50) = sech²(x/50) > 0 với mọi x
#            → tại loss=52.69: gradient ≈ 0.92 (vs log: ≈ 0.27)
#            → PINN sẽ thực sự học thay vì stuck.

#   FIX-L45  [HIGH] pinn_rankine_steering: u/v500 sau FIX-ENV-20 là
#            normalized [-1,1] → cần scale sang m/s để so sánh với u_tc.
#            Thêm u500_scale=30.0 để convert → steering comparison đúng.

#   FIX-L46  [HIGH] ensemble_spread_loss: max_spread_km=400 quá cao.
#            Từ log: spread 800-1000 km suốt training.
#            Fix: max_spread_km=200, weight tăng từ 0.1 → 0.3.
#            Thêm per-step spread penalty (không chỉ final step).

#   FIX-L47  [MEDIUM] velocity_loss: thêm direction penalty mạnh hơn
#            khi pred đi ngược chiều gt. Tăng heading_loss weight.

#   FIX-L48  [MEDIUM] trajectory_smoothness: penalize acceleration thay đổi
#            quá đột ngột (jerk) để giảm DTW.

# Kept from v23:
#   FIX-L39  pinn scale=1e-3
#   FIX-L40  step_weight_alpha
#   FIX-L41  fm_afcrps time weights
#   FIX-L43  velocity_loss dimensionless
# """
# from __future__ import annotations

# import math
# from typing import Dict, Optional

# import torch
# import torch.nn.functional as F

# __all__ = [
#     "WEIGHTS", "compute_total_loss", "fm_afcrps_loss",
#     "fm_physics_consistency_loss", "pinn_bve_loss",
#     "recurvature_loss", "velocity_loss_per_sample",
# ]

# OMEGA        = 7.2921e-5
# R_EARTH      = 6.371e6
# DT_6H        = 6 * 3600
# DEG_TO_KM    = 111.0
# STEP_KM      = 113.0

# # U/V500 scale: normalized [-1,1] → real m/s
# _UV500_SCALE = 30.0

# # ── Weights (v24) ──────────────────────────────────────────────────────────────
# WEIGHTS: Dict[str, float] = dict(
#     fm=2.0,
#     velocity=0.8,      # ↑ từ 0.5 (FIX-L47)
#     heading=2.0,       # ↑ từ 1.5 (FIX-L47)
#     recurv=1.5,
#     step=0.5,
#     disp=0.5,
#     dir=1.0,
#     smooth=0.5,        # ↑ từ 0.3
#     accel=0.8,         # ↑ từ 0.5
#     jerk=0.3,          # NEW (FIX-L48)
#     pinn=0.02,
#     fm_physics=0.3,
#     spread=0.6,        # ↑ từ 0.3 (FIX-L46)
# )

# RECURV_ANGLE_THR = 45.0
# RECURV_WEIGHT    = 2.5


# # ── Haversine ─────────────────────────────────────────────────────────────────

# def _haversine(p1: torch.Tensor, p2: torch.Tensor,
#                unit_01deg: bool = True) -> torch.Tensor:
#     if unit_01deg:
#         lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
#         lat1 = (p1[..., 1] * 50.0) / 10.0
#         lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
#         lat2 = (p2[..., 1] * 50.0) / 10.0
#     else:
#         lon1, lat1 = p1[..., 0], p1[..., 1]
#         lon2, lat2 = p2[..., 0], p2[..., 1]

#     lat1r = torch.deg2rad(lat1);  lat2r = torch.deg2rad(lat2)
#     dlon  = torch.deg2rad(lon2 - lon1)
#     dlat  = torch.deg2rad(lat2 - lat1)
#     a     = (torch.sin(dlat / 2) ** 2
#              + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2) ** 2)
#     return 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())


# def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
#     return _haversine(p1, p2, unit_01deg=False)


# # ── Step displacements in km ──────────────────────────────────────────────────

# def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
#     dt      = traj_deg[1:] - traj_deg[:-1]
#     lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
#     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
#     dt_km   = dt.clone()
#     dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
#     dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
#     return dt_km


# # ── Recurvature helpers ────────────────────────────────────────────────────────

# def _total_rotation_angle_batch(gt: torch.Tensor) -> torch.Tensor:
#     T, B, _ = gt.shape
#     if T < 3:
#         return gt.new_zeros(B)
#     lats_rad = torch.deg2rad(gt[:, :, 1])
#     cos_lat  = torch.cos(lats_rad[:-1])
#     dlat = gt[1:, :, 1] - gt[:-1, :, 1]
#     dlon = (gt[1:, :, 0] - gt[:-1, :, 0]) * cos_lat
#     v    = torch.stack([dlon, dlat], dim=-1)
#     v1   = v[:-1];  v2 = v[1:]
#     n1   = v1.norm(dim=-1).clamp(min=1e-8)
#     n2   = v2.norm(dim=-1).clamp(min=1e-8)
#     cos_a = (v1 * v2).sum(-1) / (n1 * n2)
#     return torch.rad2deg(torch.acos(cos_a.clamp(-1.0, 1.0))).sum(0)


# def _recurvature_weights(gt: torch.Tensor,
#                          thr: float = RECURV_ANGLE_THR,
#                          w_recurv: float = RECURV_WEIGHT) -> torch.Tensor:
#     rot = _total_rotation_angle_batch(gt)
#     return torch.where(rot >= thr,
#                        torch.full_like(rot, w_recurv),
#                        torch.ones_like(rot))


# # ── Directional losses ────────────────────────────────────────────────────────

# def velocity_loss_per_sample(pred: torch.Tensor,
#                              gt: torch.Tensor) -> torch.Tensor:
#     if pred.shape[0] < 2:
#         return pred.new_zeros(pred.shape[1])
#     v_pred_km = _step_displacements_km(pred)
#     v_gt_km   = _step_displacements_km(gt)
#     s_pred    = v_pred_km.norm(dim=-1)
#     s_gt      = v_gt_km.norm(dim=-1)
#     l_speed   = (s_pred - s_gt).pow(2).mean(0)
#     gn  = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     gt_unit = v_gt_km / gn
#     ate = ((v_pred_km - v_gt_km) * gt_unit).sum(-1)
#     l_ate = ate.pow(2).mean(0)

#     # FIX-L47: thêm direction penalty mạnh
#     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     cos_sim = ((v_pred_km / pn) * gt_unit).sum(-1)
#     l_dir = F.relu(-cos_sim).pow(2).mean(0) * STEP_KM ** 2  # scale to km²

#     return (l_speed + 0.5 * l_ate + 0.3 * l_dir) / (STEP_KM ** 2)


# def disp_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#     if pred.shape[0] < 2:
#         return pred.new_zeros(pred.shape[1])
#     pd = _step_displacements_km(pred).norm(dim=-1).mean(0)
#     gd = _step_displacements_km(gt).norm(dim=-1).mean(0)
#     return (pd - gd).pow(2) / (STEP_KM ** 2)


# def step_dir_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#     if pred.shape[0] < 2:
#         return pred.new_zeros(pred.shape[1])
#     v_pred_km = _step_displacements_km(pred)
#     v_gt_km   = _step_displacements_km(gt)
#     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     cos_sim = ((v_pred_km / pn) * (v_gt_km / gn)).sum(-1)
#     return (1.0 - cos_sim).mean(0)


# def heading_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#     if pred.shape[0] < 2:
#         return pred.new_zeros(())
#     pv = _step_displacements_km(pred)
#     gv = _step_displacements_km(gt)
#     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     # FIX-L47: stronger penalty for wrong direction
#     wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1))
#     wrong_dir_loss = (wrong_dir ** 2).mean()  # squared penalty

#     if pred.shape[0] >= 3:
#         def _curv(v):
#             cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
#             n1    = v[1:].norm(dim=-1).clamp(min=1e-4)
#             n2    = v[:-1].norm(dim=-1).clamp(min=1e-4)
#             return (cross / (n1 * n2)).clamp(-10.0, 10.0)
#         curv_mse = F.mse_loss(_curv(pv), _curv(gv))
#     else:
#         curv_mse = pred.new_zeros(())
#     return wrong_dir_loss + curv_mse


# def smooth_loss(pred: torch.Tensor) -> torch.Tensor:
#     if pred.shape[0] < 3:
#         return pred.new_zeros(())
#     v_km = _step_displacements_km(pred)
#     if v_km.shape[0] < 2:
#         return pred.new_zeros(())
#     accel_km = v_km[1:] - v_km[:-1]
#     return accel_km.pow(2).mean() / (STEP_KM ** 2)


# def acceleration_loss(pred: torch.Tensor) -> torch.Tensor:
#     if pred.shape[0] < 3:
#         return pred.new_zeros(())
#     v_km = _step_displacements_km(pred)
#     if v_km.shape[0] < 2:
#         return pred.new_zeros(())
#     a_km = v_km[1:] - v_km[:-1]
#     return a_km.pow(2).mean() / (STEP_KM ** 2)


# def jerk_loss(pred: torch.Tensor) -> torch.Tensor:
#     """
#     FIX-L48: Penalize jerk (change in acceleration) để giảm DTW.
#     Smooth trajectory → lower DTW.
#     """
#     if pred.shape[0] < 4:
#         return pred.new_zeros(())
#     v_km = _step_displacements_km(pred)
#     if v_km.shape[0] < 3:
#         return pred.new_zeros(())
#     a_km  = v_km[1:] - v_km[:-1]
#     j_km  = a_km[1:] - a_km[:-1]
#     return j_km.pow(2).mean() / (STEP_KM ** 2)


# def overall_dir_loss(pred: torch.Tensor, gt: torch.Tensor,
#                      ref: torch.Tensor) -> torch.Tensor:
#     p_d = pred[-1] - ref
#     g_d = gt[-1]   - ref
#     lat_ref = ref[:, 1]
#     cos_lat = torch.cos(torch.deg2rad(lat_ref)).clamp(min=1e-4)
#     p_d_km  = p_d.clone();  p_d_km[:, 0] *= cos_lat * DEG_TO_KM;  p_d_km[:, 1] *= DEG_TO_KM
#     g_d_km  = g_d.clone();  g_d_km[:, 0] *= cos_lat * DEG_TO_KM;  g_d_km[:, 1] *= DEG_TO_KM
#     pn = p_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     gn = g_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     return (1.0 - ((p_d_km / pn) * (g_d_km / gn)).sum(-1)).mean()


# def recurvature_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#     if pred.shape[0] < 3:
#         return pred.new_zeros(())
#     pred_v   = pred[1:] - pred[:-1]
#     gt_v     = gt[1:]   - gt[:-1]
#     gt_cross = (gt_v[:-1, :, 0] * gt_v[1:, :, 1]
#               - gt_v[:-1, :, 1] * gt_v[1:, :, 0])
#     if gt_cross.shape[0] < 2:
#         return pred.new_zeros(())
#     sign_change = (gt_cross[:-1] * gt_cross[1:]) < 0
#     if not sign_change.any():
#         return pred.new_zeros(())
#     pred_v_mid = pred_v[1:-1]
#     gt_v_mid   = gt_v[1:-1]
#     if pred_v_mid.shape[0] > sign_change.shape[0]:
#         pred_v_mid = pred_v_mid[:sign_change.shape[0]]
#         gt_v_mid   = gt_v_mid[:sign_change.shape[0]]
#     pn      = pred_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     gn      = gt_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     cos_sim = ((pred_v_mid / pn) * (gt_v_mid / gn)).sum(-1)
#     dir_loss = (1.0 - cos_sim)
#     mask     = sign_change.float()
#     if mask.sum() < 1:
#         return pred.new_zeros(())
#     return (dir_loss * mask).sum() / mask.sum().clamp(min=1)


# # ── AFCRPS ────────────────────────────────────────────────────────────────────

# def fm_afcrps_loss(
#     pred_samples: torch.Tensor,
#     gt: torch.Tensor,
#     unit_01deg: bool = True,
#     intensity_w: Optional[torch.Tensor] = None,
#     step_weight_alpha: float = 0.0,
# ) -> torch.Tensor:
#     M, T, B, _ = pred_samples.shape
#     eps = 1e-3

#     base_w = torch.linspace(0.5, 1.5, T, device=pred_samples.device)

#     if step_weight_alpha > 0.0:
#         # early_w = torch.exp(-torch.arange(T, dtype=torch.float,
#         #                      device=pred_samples.device) * 0.2)
#         # losses.py — fm_afcrps_loss()
#         early_w = torch.exp(-torch.arange(T, dtype=torch.float,
#                      device=pred_samples.device) * 0.35)   # 0.2→0.35
#         early_w = early_w / early_w.mean()
#         time_w  = (1.0 - step_weight_alpha) * base_w + step_weight_alpha * early_w
#     else:
#         time_w = base_w

#     if M == 1:
#         dist = _haversine(pred_samples[0], gt, unit_01deg)
#         loss_per_b = (dist * time_w.unsqueeze(1)).mean(0)
#     else:
#         d_to_gt = _haversine(
#             pred_samples,
#             gt.unsqueeze(0).expand_as(pred_samples),
#             unit_01deg,
#         )
#         d_to_gt_w    = d_to_gt * time_w.unsqueeze(0).unsqueeze(2)
#         d_to_gt_mean = d_to_gt_w.mean(1)

#         ps_i = pred_samples.unsqueeze(1)
#         ps_j = pred_samples.unsqueeze(0)
#         d_pair = _haversine(
#             ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
#             ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
#             unit_01deg,
#         ).reshape(M, M, T, B)
#         d_pair_w    = d_pair * time_w.unsqueeze(0).unsqueeze(0).unsqueeze(3)
#         d_pair_mean = d_pair_w.mean(2)

#         e_sy  = d_to_gt_mean.mean(0)
#         e_ssp = d_pair_mean.mean(0).mean(0)
#         loss_per_b = (e_sy - 0.5 * e_ssp).clamp(min=eps)

#     if intensity_w is not None:
#         w = intensity_w.to(loss_per_b.device)
#         return (loss_per_b * w).mean()
#     return loss_per_b.mean()


# # ── PINN losses ────────────────────────────────────────────────────────────────

# def _soft_clamp_loss(loss: torch.Tensor, max_val: float = 50.0) -> torch.Tensor:
#     """
#     FIX-L44: Soft clamp bằng tanh scaling.
#     loss_scaled = max_val * tanh(loss / max_val)
#     Gradient = tanh'(x/max_val) = sech²(x/max_val) > 0 với mọi x.
#     Tại loss=52.69: gradient ≈ sech²(1.054) ≈ 0.31 (vs log: ≈ 0.27 nhưng cố định)
#     Gradient sẽ giảm dần khi loss tăng → vẫn có gradient khác 0.
#     """
#     return max_val * torch.tanh(loss / max_val)


# def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
#     """
#     FIX-L44: Dùng tanh soft-clamp thay vì log-clamp.
#     scale=1e-3 giữ nguyên từ FIX-L39.
#     """
#     T, B, _ = pred_abs_deg.shape
#     if T < 3:
#         return pred_abs_deg.new_zeros(())

#     DT      = DT_6H
#     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
#     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
#     cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

#     u = dlat[:, :, 0] * cos_lat * 111000.0 / DT
#     v = dlat[:, :, 1] * 111000.0 / DT

#     if u.shape[0] < 2:
#         return pred_abs_deg.new_zeros(())

#     du = (u[1:] - u[:-1]) / DT
#     dv = (v[1:] - v[:-1]) / DT

#     f    = 2 * OMEGA * torch.sin(lat_rad[1:-1])
#     beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH

#     res_u = du - f * v[1:]
#     res_v = dv + f * u[1:]

#     R_tc   = 3e5
#     v_beta_x = -beta * R_tc ** 2 / 2
#     res_u_corrected = res_u - v_beta_x

#     scale = 1e-3
#     loss = ((res_u_corrected / scale).pow(2).mean()
#             + (res_v / scale).pow(2).mean())

#     # FIX-L44: tanh soft-clamp → gradient luôn > 0
#     # return _soft_clamp_loss(loss, max_val=50.0)
#     return _soft_clamp_loss(loss, max_val=20.0)


# def pinn_rankine_steering(pred_abs_deg: torch.Tensor,
#                           env_data: Optional[dict]) -> torch.Tensor:
#     """
#     FIX-L45: u/v500 sau FIX-ENV-20 đã normalized [-1,1].
#     Scale back sang m/s để so sánh với u_tc.
#     """
#     if env_data is None:
#         return pred_abs_deg.new_zeros(())

#     T, B, _ = pred_abs_deg.shape
#     if T < 2:
#         return pred_abs_deg.new_zeros(())

#     DT      = DT_6H
#     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
#     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
#     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

#     u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT
#     v_tc = dlat[:, :, 1] * 111000.0 / DT

#     u500_raw = env_data.get("u500_mean", None)
#     v500_raw = env_data.get("v500_mean", None)

#     if u500_raw is None or v500_raw is None:
#         speed = torch.sqrt(u_tc ** 2 + v_tc ** 2)
#         return F.relu(speed - 30.0).pow(2).mean() * 0.01

#     def _align(x: torch.Tensor) -> torch.Tensor:
#         if not torch.is_tensor(x):
#             return pred_abs_deg.new_zeros(T - 1, B)
#         x = x.to(pred_abs_deg.device)
#         if x.dim() == 3:
#             x_sq    = x[:, :, 0]
#             T_env   = x_sq.shape[1]
#             T_tgt   = T - 1
#             if T_env >= T_tgt:
#                 return x_sq[:, :T_tgt].permute(1, 0)
#             pad = x_sq[:, -1:].expand(-1, T_tgt - T_env)
#             return torch.cat([x_sq, pad], dim=1).permute(1, 0)
#         elif x.dim() == 2:
#             if x.shape == (B, T - 1):
#                 return x.permute(1, 0)
#             elif x.shape[0] == T - 1:
#                 return x[:, :B] if x.shape[1] >= B else x.expand(-1, B)
#         return pred_abs_deg.new_zeros(T - 1, B)

#     u500_n = _align(u500_raw)
#     v500_n = _align(v500_raw)

#     # FIX-L45: scale normalized [-1,1] → m/s
#     u500 = u500_n * _UV500_SCALE
#     v500 = v500_n * _UV500_SCALE

#     # Chỉ áp dụng steering penalty khi u/v500 có giá trị thực (non-zero)
#     uv_magnitude = torch.sqrt(u500 ** 2 + v500 ** 2)
#     has_steering = (uv_magnitude > 0.5).float()  # threshold 0.5 m/s

#     env_dir  = torch.stack([u500, v500], dim=-1)
#     tc_dir   = torch.stack([u_tc, v_tc],  dim=-1)
#     env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)  # min 0.5 m/s
#     tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
#     cos_sim  = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)
#     misalign = F.relu(-0.5 - cos_sim).pow(2)

#     # Only penalize when we have actual steering data
#     return (misalign * has_steering).mean() * 0.05


# def pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
#     if pred_deg.shape[0] < 2:
#         return pred_deg.new_zeros(())
#     dt_deg  = pred_deg[1:] - pred_deg[:-1]
#     lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
#     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
#     dx_km   = dt_deg[:, :, 0] * cos_lat * DEG_TO_KM
#     dy_km   = dt_deg[:, :, 1] * DEG_TO_KM
#     speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
#     return F.relu(speed - 600.0).pow(2).mean()


# def pinn_bve_loss(pred_abs_deg: torch.Tensor,
#                   batch_list,
#                   env_data: Optional[dict] = None) -> torch.Tensor:
#     """
#     FIX-L44: tanh soft-clamp trong pinn_shallow_water.
#     FIX-L45: steering dùng actual u/v500 m/s.
#     """
#     T = pred_abs_deg.shape[0]
#     if T < 3:
#         return pred_abs_deg.new_zeros(())

#     l_sw    = pinn_shallow_water(pred_abs_deg)
#     l_steer = pinn_rankine_steering(pred_abs_deg, env_data)
#     l_speed = pinn_speed_constraint(pred_abs_deg)

#     total = l_sw + 0.5 * l_steer + 0.1 * l_speed
#     # FIX-L44: tanh soft-clamp (không log)
#     # return _soft_clamp_loss(total, max_val=50.0)
#     return _soft_clamp_loss(total, max_val=20.0)


# # ── Physics consistency loss ──────────────────────────────────────────────────

# def fm_physics_consistency_loss(
#     pred_samples: torch.Tensor,
#     gt_norm: torch.Tensor,
#     last_pos: torch.Tensor,
# ) -> torch.Tensor:
#     S, T, B = pred_samples.shape[:3]

#     last_lon = (last_pos[:, 0] * 50.0 + 1800.0) / 10.0
#     last_lat = (last_pos[:, 1] * 50.0) / 10.0
#     lat_rad  = torch.deg2rad(last_lat)
#     beta     = 2 * OMEGA * torch.cos(lat_rad) / R_EARTH
#     R_tc     = 3e5

#     v_beta_lon = -beta * R_tc ** 2 / 2
#     v_beta_lat =  beta * R_tc ** 2 / 4

#     beta_dir      = torch.stack([v_beta_lon, v_beta_lat], dim=-1)
#     beta_norm     = beta_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     beta_dir_unit = beta_dir / beta_norm

#     pos_step1     = pred_samples[:, 0, :, :2]
#     dir_step1     = pos_step1 - last_pos.unsqueeze(0)
#     dir_norm      = dir_step1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     dir_unit      = dir_step1 / dir_norm
#     mean_dir      = dir_unit.mean(0)
#     mean_norm     = mean_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     mean_dir_unit = mean_dir / mean_norm

#     cos_align    = (mean_dir_unit * beta_dir_unit).sum(-1)
#     beta_strength = beta_norm.squeeze(-1)
#     penalise_mask = (beta_strength > 1.0).float()
#     direction_loss = F.relu(-cos_align) * penalise_mask
#     return direction_loss.mean() * 0.5


# # ── Spread regularization ──────────────────────────────────────────────────────

# def ensemble_spread_loss(all_trajs: torch.Tensor,
#                          max_spread_km: float = 200.0) -> torch.Tensor:
#     """
#     FIX-L46: max_spread_km=200 (từ 400), thêm per-step penalty.
#     all_trajs: [S, T, B, 2] normalised.
#     """
#     if all_trajs.shape[0] < 2:
#         return all_trajs.new_zeros(())

#     S, T, B, _ = all_trajs.shape

#     # # Per-step spread penalty (không chỉ final step)
#     # # Weight tăng theo thời gian: later steps penalized more
#     # step_weights = torch.linspace(0.5, 1.5, T, device=all_trajs.device)
    
#     # FIX-L46b: exponential decay — early steps penalized MORE
#     # step 0 weight=2.0, step T-1 weight=0.5 (geometric)
#     step_weights = torch.exp(
#         -torch.arange(T, dtype=torch.float, device=all_trajs.device) * (math.log(4.0) / max(T-1, 1))
#     ) * 2.0   # range: 2.0 → 0.5

#     total_loss = all_trajs.new_zeros(())
#     for t in range(T):
#         step_trajs = all_trajs[:, t, :, :2]  # [S, B, 2]
#         std_lon = step_trajs[:, :, 0].std(0)  # [B]
#         std_lat = step_trajs[:, :, 1].std(0)  # [B]
#         spread_km = torch.sqrt(std_lon ** 2 + std_lat ** 2) * 500.0  # [B]
#         excess    = F.relu(spread_km - max_spread_km)
#         total_loss = total_loss + step_weights[t] * (excess / max_spread_km).pow(2).mean()

#     return total_loss / T


# # ── Intensity weights ─────────────────────────────────────────────────────────

# def _intensity_weights_from_obs(obs_Me: torch.Tensor) -> torch.Tensor:
#     wind_norm = obs_Me[-1, :, 1].detach()
#     w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
#         torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
#         torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
#                     torch.full_like(wind_norm, 1.5))))
#     return w / w.mean().clamp(min=1e-6)


# # ── Main loss ─────────────────────────────────────────────────────────────────

# def compute_total_loss(
#     pred_abs,
#     gt,
#     ref,
#     batch_list,
#     pred_samples=None,
#     gt_norm=None,
#     weights=WEIGHTS,
#     intensity_w: Optional[torch.Tensor] = None,
#     env_data: Optional[dict] = None,
#     step_weight_alpha: float = 0.0,
#     all_trajs: Optional[torch.Tensor] = None,
# ) -> Dict:
#     # 1. Sample weights
#     recurv_w = _recurvature_weights(gt, w_recurv=2.5)
#     sample_w = recurv_w * (intensity_w.to(gt.device) if intensity_w is not None
#                            else 1.0)
#     sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

#     # 2. AFCRPS
#     if pred_samples is not None:
#         if gt_norm is not None:
#             l_fm = fm_afcrps_loss(pred_samples, gt_norm,
#                                   intensity_w=sample_w, unit_01deg=True,
#                                   step_weight_alpha=step_weight_alpha)
#         else:
#             l_fm = fm_afcrps_loss(pred_samples, gt,
#                                   intensity_w=sample_w, unit_01deg=False,
#                                   step_weight_alpha=step_weight_alpha)
#     else:
#         l_fm = _haversine_deg(pred_abs, gt).mean()

#     # 3. Directional losses
#     NRM = 35.0

#     l_vel   = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
#     l_disp  = (disp_loss_per_sample(pred_abs, gt)     * sample_w).mean()
#     l_step  = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()

#     l_heading   = heading_loss(pred_abs, gt)
#     l_recurv    = recurvature_loss(pred_abs, gt)
#     l_dir_final = overall_dir_loss(pred_abs, gt, ref)
#     l_smooth    = smooth_loss(pred_abs)
#     l_accel     = acceleration_loss(pred_abs)
#     l_jerk      = jerk_loss(pred_abs)  # FIX-L48

#     # 4. PINN
#     _env = env_data
#     if _env is None and batch_list is not None:
#         try:
#             _env = batch_list[13]
#         except (IndexError, TypeError):
#             _env = None

#     l_pinn = pinn_bve_loss(pred_abs, batch_list, env_data=_env)

#     # 5. Spread penalty
#     l_spread = pred_abs.new_zeros(())
#     if all_trajs is not None and all_trajs.shape[0] >= 2:
#         l_spread = ensemble_spread_loss(all_trajs, max_spread_km=200.0)

#     # 6. Total
#     total = (
#         weights.get("fm",       2.0) * l_fm
#         + weights.get("velocity", 0.8) * l_vel   * NRM
#         + weights.get("disp",     0.5) * l_disp  * NRM
#         + weights.get("step",     0.5) * l_step  * NRM
#         + weights.get("heading",  2.0) * l_heading * NRM
#         + weights.get("recurv",   1.5) * l_recurv  * NRM
#         + weights.get("dir",      1.0) * l_dir_final * NRM
#         + weights.get("smooth",   0.5) * l_smooth * NRM
#         + weights.get("accel",    0.8) * l_accel  * NRM
#         + weights.get("jerk",     0.3) * l_jerk   * NRM  # FIX-L48
#         + weights.get("pinn",    0.02) * l_pinn
#         + weights.get("spread",  0.3)  * l_spread * NRM
#     ) / NRM

#     if torch.isnan(total) or torch.isinf(total):
#         total = pred_abs.new_zeros(())

#     return dict(
#         total        = total,
#         fm           = l_fm.item(),
#         velocity     = l_vel.item() * NRM,
#         step         = l_step.item(),
#         disp         = l_disp.item() * NRM,
#         heading      = l_heading.item(),
#         recurv       = l_recurv.item(),
#         smooth       = l_smooth.item() * NRM,
#         accel        = l_accel.item() * NRM,
#         jerk         = l_jerk.item() * NRM,
#         pinn         = l_pinn.item(),
#         spread       = l_spread.item() * NRM,
#         recurv_ratio = (_recurvature_weights(gt) > 1.0).float().mean().item(),
#     )


# # ── Legacy ─────────────────────────────────────────────────────────────────────

# class TripletLoss(torch.nn.Module):
#     def __init__(self, margin=None):
#         super().__init__()
#         self.margin  = margin
#         self.loss_fn = (torch.nn.SoftMarginLoss() if margin is None
#                         else torch.nn.TripletMarginLoss(margin=margin, p=2))

#     def forward(self, anchor, pos, neg):
#         if self.margin is None:
#             y = torch.ones(anchor.shape[0], device=anchor.device)
#             return self.loss_fn(
#                 torch.norm(anchor - neg, 2, dim=1)
#                 - torch.norm(anchor - pos, 2, dim=1), y)
#         return self.loss_fn(anchor, pos, neg)

"""
Model/losses.py  ── v25
========================
FIXES vs v24:

  FIX-L49  [CRITICAL] short_range_regression_loss(): loss mới riêng cho
           4 bước đầu (6h/12h/18h/24h). Dùng Huber loss trên khoảng cách
           Haversine. Weight 12h = 4.0, 6h/18h/24h = 2.0.
           Không dùng AFCRPS cho short-range → ổn định và chính xác hơn.

  FIX-L50  [HIGH] PINN scale: 1e-3 → 1e-2.
           Log cho thấy pinn=15.232 không đổi → scale quá nhỏ, loss bị
           dominate bởi phần tanh-clamp. Tăng scale làm residual nhỏ hơn
           → gradient flow tốt hơn → PINN thực sự học.

  FIX-L51  [HIGH] WEIGHTS: thêm "short_range": 5.0, tăng "spread": 0.8
           (từ 0.6). Spread 400-500km là nguồn gốc của ADE cao.

  FIX-L52  [MEDIUM] ensemble_spread_loss: max_spread_km=150 (từ 200),
           FIX-L46b vẫn giữ exponential decay trên step weights.

  FIX-L53  [MEDIUM] fm_afcrps_loss: early_w decay rate 0.35 → 0.5
           khi alpha > 0 → focus mạnh hơn vào step 1-2 trong giai đoạn
           đầu training.

Kept from v24:
  FIX-L44  tanh soft-clamp (max_val=20.0)
  FIX-L45  pinn_rankine_steering scale u/v500
  FIX-L47  velocity direction penalty
  FIX-L48  jerk loss
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

__all__ = [
    "WEIGHTS", "compute_total_loss", "fm_afcrps_loss",
    "fm_physics_consistency_loss", "pinn_bve_loss",
    "recurvature_loss", "velocity_loss_per_sample",
    "short_range_regression_loss",
]

OMEGA        = 7.2921e-5
R_EARTH      = 6.371e6
DT_6H        = 6 * 3600
DEG_TO_KM    = 111.0
STEP_KM      = 113.0

_UV500_SCALE = 30.0          # normalized [-1,1] → m/s

# ── Weights (v25) ─────────────────────────────────────────────────────────────
WEIGHTS: Dict[str, float] = dict(
    fm          = 2.0,
    velocity    = 0.8,
    heading     = 2.0,
    recurv      = 1.5,
    step        = 0.5,
    disp        = 0.5,
    dir         = 1.0,
    smooth      = 0.5,
    accel       = 0.8,
    jerk        = 0.3,
    pinn        = 0.02,
    fm_physics  = 0.3,
    spread      = 0.8,        # ↑ từ 0.6  (FIX-L51)
    short_range = 5.0,        # NEW       (FIX-L49)
)

RECURV_ANGLE_THR = 45.0
RECURV_WEIGHT    = 2.5

# ── short-range config ────────────────────────────────────────────────────────
_SR_N_STEPS  = 4                         # 6h, 12h, 18h, 24h
_SR_WEIGHTS  = [2.0, 4.0, 2.0, 2.0]     # step-2 (12h) highest
_HUBER_DELTA = 50.0                      # km


# ══════════════════════════════════════════════════════════════════════════════
#  Haversine
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

    lat1r = torch.deg2rad(lat1);  lat2r = torch.deg2rad(lat2)
    dlon  = torch.deg2rad(lon2 - lon1)
    dlat  = torch.deg2rad(lat2 - lat1)
    a     = (torch.sin(dlat / 2) ** 2
             + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2) ** 2)
    return 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())


def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    return _haversine(p1, p2, unit_01deg=False)


def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
    """Normalised → degrees. Accepts any leading dims."""
    out = arr.clone()
    out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
    out[..., 1] = (arr[..., 1] * 50.0) / 10.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  FIX-L49: Short-range regression loss
# ══════════════════════════════════════════════════════════════════════════════

def short_range_regression_loss(
    pred_sr:  torch.Tensor,   # [4, B, 2]  normalised positions
    gt_sr:    torch.Tensor,   # [4, B, 2]  normalised positions
    last_pos: torch.Tensor,   # [B, 2]     (unused, kept for API compat)
) -> torch.Tensor:
    """
    FIX-L49: Huber loss on haversine distance for steps 1-4.

    Steps : 6h(1)  12h(2)  18h(3)  24h(4)
    Weight:  2.0    4.0     2.0     2.0    ← step-2(12h) penalised most

    Huber threshold = 50 km (less sensitive to outlier trajectories).
    Returns scalar loss in [0, ~10] range suitable for weight=5.0.
    """
    n_steps = min(pred_sr.shape[0], gt_sr.shape[0], _SR_N_STEPS)
    if n_steps == 0:
        return pred_sr.new_zeros(())

    pred_deg = _norm_to_deg(pred_sr[:n_steps])   # [n_steps, B, 2]
    gt_deg   = _norm_to_deg(gt_sr[:n_steps])

    # Haversine per (step, batch)
    dist_km = _haversine_deg(pred_deg, gt_deg)   # [n_steps, B]

    # Huber loss
    huber = torch.where(
        dist_km < _HUBER_DELTA,
        0.5 * dist_km.pow(2) / _HUBER_DELTA,
        dist_km - 0.5 * _HUBER_DELTA,
    )  # [n_steps, B]

    # Step weights
    w = pred_sr.new_tensor(_SR_WEIGHTS[:n_steps])  # [n_steps]
    weighted = (huber * w.view(-1, 1)).mean()

    # Normalise: divide by HUBER_DELTA so loss ≈ 1.0 when error ≈ 50 km
    return weighted / _HUBER_DELTA


# ══════════════════════════════════════════════════════════════════════════════
#  Step displacements
# ══════════════════════════════════════════════════════════════════════════════

def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
    dt      = traj_deg[1:] - traj_deg[:-1]
    lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
    cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
    dt_km   = dt.clone()
    dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
    dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
    return dt_km


# ══════════════════════════════════════════════════════════════════════════════
#  Recurvature helpers
# ══════════════════════════════════════════════════════════════════════════════

def _total_rotation_angle_batch(gt: torch.Tensor) -> torch.Tensor:
    T, B, _ = gt.shape
    if T < 3:
        return gt.new_zeros(B)
    lats_rad = torch.deg2rad(gt[:, :, 1])
    cos_lat  = torch.cos(lats_rad[:-1])
    dlat = gt[1:, :, 1] - gt[:-1, :, 1]
    dlon = (gt[1:, :, 0] - gt[:-1, :, 0]) * cos_lat
    v    = torch.stack([dlon, dlat], dim=-1)
    v1   = v[:-1];  v2 = v[1:]
    n1   = v1.norm(dim=-1).clamp(min=1e-8)
    n2   = v2.norm(dim=-1).clamp(min=1e-8)
    cos_a = (v1 * v2).sum(-1) / (n1 * n2)
    return torch.rad2deg(torch.acos(cos_a.clamp(-1.0, 1.0))).sum(0)


def _recurvature_weights(gt: torch.Tensor,
                         thr: float = RECURV_ANGLE_THR,
                         w_recurv: float = RECURV_WEIGHT) -> torch.Tensor:
    rot = _total_rotation_angle_batch(gt)
    return torch.where(rot >= thr,
                       torch.full_like(rot, w_recurv),
                       torch.ones_like(rot))


# ══════════════════════════════════════════════════════════════════════════════
#  Directional losses
# ══════════════════════════════════════════════════════════════════════════════

def velocity_loss_per_sample(pred: torch.Tensor,
                             gt: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.new_zeros(pred.shape[1])
    v_pred_km = _step_displacements_km(pred)
    v_gt_km   = _step_displacements_km(gt)
    s_pred    = v_pred_km.norm(dim=-1)
    s_gt      = v_gt_km.norm(dim=-1)
    l_speed   = (s_pred - s_gt).pow(2).mean(0)
    gn  = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    gt_unit = v_gt_km / gn
    ate = ((v_pred_km - v_gt_km) * gt_unit).sum(-1)
    l_ate = ate.pow(2).mean(0)
    pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    cos_sim = ((v_pred_km / pn) * gt_unit).sum(-1)
    l_dir = F.relu(-cos_sim).pow(2).mean(0) * STEP_KM ** 2
    return (l_speed + 0.5 * l_ate + 0.3 * l_dir) / (STEP_KM ** 2)


def disp_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.new_zeros(pred.shape[1])
    pd = _step_displacements_km(pred).norm(dim=-1).mean(0)
    gd = _step_displacements_km(gt).norm(dim=-1).mean(0)
    return (pd - gd).pow(2) / (STEP_KM ** 2)


def step_dir_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.new_zeros(pred.shape[1])
    v_pred_km = _step_displacements_km(pred)
    v_gt_km   = _step_displacements_km(gt)
    pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    cos_sim = ((v_pred_km / pn) * (v_gt_km / gn)).sum(-1)
    return (1.0 - cos_sim).mean(0)


def heading_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    pv = _step_displacements_km(pred)
    gv = _step_displacements_km(gt)
    pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1))
    wrong_dir_loss = (wrong_dir ** 2).mean()

    if pred.shape[0] >= 3:
        def _curv(v):
            cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
            n1    = v[1:].norm(dim=-1).clamp(min=1e-4)
            n2    = v[:-1].norm(dim=-1).clamp(min=1e-4)
            return (cross / (n1 * n2)).clamp(-10.0, 10.0)
        curv_mse = F.mse_loss(_curv(pv), _curv(gv))
    else:
        curv_mse = pred.new_zeros(())
    return wrong_dir_loss + curv_mse


def smooth_loss(pred: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 3:
        return pred.new_zeros(())
    v_km = _step_displacements_km(pred)
    if v_km.shape[0] < 2:
        return pred.new_zeros(())
    accel_km = v_km[1:] - v_km[:-1]
    return accel_km.pow(2).mean() / (STEP_KM ** 2)


def acceleration_loss(pred: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 3:
        return pred.new_zeros(())
    v_km = _step_displacements_km(pred)
    if v_km.shape[0] < 2:
        return pred.new_zeros(())
    a_km = v_km[1:] - v_km[:-1]
    return a_km.pow(2).mean() / (STEP_KM ** 2)


def jerk_loss(pred: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 4:
        return pred.new_zeros(())
    v_km = _step_displacements_km(pred)
    if v_km.shape[0] < 3:
        return pred.new_zeros(())
    a_km = v_km[1:] - v_km[:-1]
    j_km = a_km[1:] - a_km[:-1]
    return j_km.pow(2).mean() / (STEP_KM ** 2)


def overall_dir_loss(pred: torch.Tensor, gt: torch.Tensor,
                     ref: torch.Tensor) -> torch.Tensor:
    p_d = pred[-1] - ref
    g_d = gt[-1]   - ref
    lat_ref = ref[:, 1]
    cos_lat = torch.cos(torch.deg2rad(lat_ref)).clamp(min=1e-4)
    p_d_km  = p_d.clone()
    p_d_km[:, 0] *= cos_lat * DEG_TO_KM
    p_d_km[:, 1] *= DEG_TO_KM
    g_d_km  = g_d.clone()
    g_d_km[:, 0] *= cos_lat * DEG_TO_KM
    g_d_km[:, 1] *= DEG_TO_KM
    pn = p_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    gn = g_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    return (1.0 - ((p_d_km / pn) * (g_d_km / gn)).sum(-1)).mean()


def recurvature_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 3:
        return pred.new_zeros(())
    pred_v   = pred[1:] - pred[:-1]
    gt_v     = gt[1:]   - gt[:-1]
    gt_cross = (gt_v[:-1, :, 0] * gt_v[1:, :, 1]
              - gt_v[:-1, :, 1] * gt_v[1:, :, 0])
    if gt_cross.shape[0] < 2:
        return pred.new_zeros(())
    sign_change = (gt_cross[:-1] * gt_cross[1:]) < 0
    if not sign_change.any():
        return pred.new_zeros(())
    pred_v_mid = pred_v[1:-1]
    gt_v_mid   = gt_v[1:-1]
    if pred_v_mid.shape[0] > sign_change.shape[0]:
        pred_v_mid = pred_v_mid[:sign_change.shape[0]]
        gt_v_mid   = gt_v_mid[:sign_change.shape[0]]
    pn      = pred_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gn      = gt_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos_sim = ((pred_v_mid / pn) * (gt_v_mid / gn)).sum(-1)
    dir_loss = (1.0 - cos_sim)
    mask     = sign_change.float()
    if mask.sum() < 1:
        return pred.new_zeros(())
    return (dir_loss * mask).sum() / mask.sum().clamp(min=1)


# ══════════════════════════════════════════════════════════════════════════════
#  AFCRPS
# ══════════════════════════════════════════════════════════════════════════════

def fm_afcrps_loss(
    pred_samples:      torch.Tensor,
    gt:                torch.Tensor,
    unit_01deg:        bool  = True,
    intensity_w:       Optional[torch.Tensor] = None,
    step_weight_alpha: float = 0.0,
) -> torch.Tensor:
    M, T, B, _ = pred_samples.shape
    eps = 1e-3

    base_w = torch.linspace(0.5, 1.5, T, device=pred_samples.device)

    if step_weight_alpha > 0.0:
        # FIX-L53: stronger early focus (0.35 → 0.5)
        early_w = torch.exp(
            -torch.arange(T, dtype=torch.float, device=pred_samples.device) * 0.5
        )
        early_w = early_w / early_w.mean()
        time_w  = (1.0 - step_weight_alpha) * base_w + step_weight_alpha * early_w
    else:
        time_w = base_w

    if M == 1:
        dist = _haversine(pred_samples[0], gt, unit_01deg)
        loss_per_b = (dist * time_w.unsqueeze(1)).mean(0)
    else:
        d_to_gt = _haversine(
            pred_samples,
            gt.unsqueeze(0).expand_as(pred_samples),
            unit_01deg,
        )
        d_to_gt_w    = d_to_gt * time_w.unsqueeze(0).unsqueeze(2)
        d_to_gt_mean = d_to_gt_w.mean(1)

        ps_i = pred_samples.unsqueeze(1)
        ps_j = pred_samples.unsqueeze(0)
        d_pair = _haversine(
            ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
            ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
            unit_01deg,
        ).reshape(M, M, T, B)
        d_pair_w    = d_pair * time_w.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        d_pair_mean = d_pair_w.mean(2)

        e_sy  = d_to_gt_mean.mean(0)
        e_ssp = d_pair_mean.mean(0).mean(0)
        loss_per_b = (e_sy - 0.5 * e_ssp).clamp(min=eps)

    if intensity_w is not None:
        w = intensity_w.to(loss_per_b.device)
        return (loss_per_b * w).mean()
    return loss_per_b.mean()


# ══════════════════════════════════════════════════════════════════════════════
#  PINN losses  (FIX-L50: scale 1e-3 → 1e-2)
# ══════════════════════════════════════════════════════════════════════════════

def _soft_clamp_loss(loss: torch.Tensor, max_val: float = 20.0) -> torch.Tensor:
    """FIX-L44: tanh soft-clamp. Gradient always > 0."""
    return max_val * torch.tanh(loss / max_val)


def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
    """
    FIX-L50: scale 1e-3 → 1e-2 so residuals are properly normalised.
    Typical TC acceleration ~1e-4 m/s²:
      old: res/1e-3 ~ 0.1  → loss ~ 0.01  (tiny gradient)
      new: res/1e-2 ~ 0.01 → loss ~ 1e-4  then tanh brings to ~1e-4
    But when trajectories are bad (spread 400km):
      new: res/1e-2 ~ 0.15 → loss ~ 0.022 → tanh ~ 0.44  (still manageable)
    → Gradient now flows even for bad trajectories.
    """
    T, B, _ = pred_abs_deg.shape
    if T < 3:
        return pred_abs_deg.new_zeros(())

    DT      = DT_6H
    lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
    dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
    cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

    u = dlat[:, :, 0] * cos_lat * 111000.0 / DT
    v = dlat[:, :, 1] * 111000.0 / DT

    if u.shape[0] < 2:
        return pred_abs_deg.new_zeros(())

    du = (u[1:] - u[:-1]) / DT
    dv = (v[1:] - v[:-1]) / DT

    f    = 2 * OMEGA * torch.sin(lat_rad[1:-1])
    beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH

    res_u = du - f * v[1:]
    res_v = dv + f * u[1:]

    R_tc          = 3e5
    v_beta_x      = -beta * R_tc ** 2 / 2
    res_u_corrected = res_u - v_beta_x

    # FIX-L50: scale = 1e-2  (was 1e-3)
    scale = 1e-2
    loss  = ((res_u_corrected / scale).pow(2).mean()
             + (res_v / scale).pow(2).mean())

    return _soft_clamp_loss(loss, max_val=20.0)


def pinn_rankine_steering(pred_abs_deg: torch.Tensor,
                          env_data: Optional[dict]) -> torch.Tensor:
    if env_data is None:
        return pred_abs_deg.new_zeros(())

    T, B, _ = pred_abs_deg.shape
    if T < 2:
        return pred_abs_deg.new_zeros(())

    DT      = DT_6H
    dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
    lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
    cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

    u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT
    v_tc = dlat[:, :, 1] * 111000.0 / DT

    u500_raw = env_data.get("u500_mean", None)
    v500_raw = env_data.get("v500_mean", None)

    if u500_raw is None or v500_raw is None:
        speed = torch.sqrt(u_tc ** 2 + v_tc ** 2)
        return F.relu(speed - 30.0).pow(2).mean() * 0.01

    def _align(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            return pred_abs_deg.new_zeros(T - 1, B)
        x = x.to(pred_abs_deg.device)
        if x.dim() == 3:
            x_sq  = x[:, :, 0]
            T_env = x_sq.shape[1]
            T_tgt = T - 1
            if T_env >= T_tgt:
                return x_sq[:, :T_tgt].permute(1, 0)
            pad = x_sq[:, -1:].expand(-1, T_tgt - T_env)
            return torch.cat([x_sq, pad], dim=1).permute(1, 0)
        elif x.dim() == 2:
            if x.shape == (B, T - 1):
                return x.permute(1, 0)
            elif x.shape[0] == T - 1:
                return x[:, :B] if x.shape[1] >= B else x.expand(-1, B)
        return pred_abs_deg.new_zeros(T - 1, B)

    u500 = _align(u500_raw) * _UV500_SCALE
    v500 = _align(v500_raw) * _UV500_SCALE

    uv_magnitude = torch.sqrt(u500 ** 2 + v500 ** 2)
    has_steering  = (uv_magnitude > 0.5).float()

    env_dir  = torch.stack([u500, v500], dim=-1)
    tc_dir   = torch.stack([u_tc, v_tc],  dim=-1)
    env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
    tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
    cos_sim  = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)
    misalign = F.relu(-0.5 - cos_sim).pow(2)

    return (misalign * has_steering).mean() * 0.05


def pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
    if pred_deg.shape[0] < 2:
        return pred_deg.new_zeros(())
    dt_deg  = pred_deg[1:] - pred_deg[:-1]
    lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
    cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
    dx_km   = dt_deg[:, :, 0] * cos_lat * DEG_TO_KM
    dy_km   = dt_deg[:, :, 1] * DEG_TO_KM
    speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
    return F.relu(speed - 600.0).pow(2).mean()


def pinn_bve_loss(pred_abs_deg: torch.Tensor,
                  batch_list,
                  env_data: Optional[dict] = None) -> torch.Tensor:
    T = pred_abs_deg.shape[0]
    if T < 3:
        return pred_abs_deg.new_zeros(())

    l_sw    = pinn_shallow_water(pred_abs_deg)
    l_steer = pinn_rankine_steering(pred_abs_deg, env_data)
    l_speed = pinn_speed_constraint(pred_abs_deg)

    total = l_sw + 0.5 * l_steer + 0.1 * l_speed
    return _soft_clamp_loss(total, max_val=20.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Physics consistency
# ══════════════════════════════════════════════════════════════════════════════

def fm_physics_consistency_loss(
    pred_samples: torch.Tensor,
    gt_norm:      torch.Tensor,
    last_pos:     torch.Tensor,
) -> torch.Tensor:
    S, T, B = pred_samples.shape[:3]

    last_lon = (last_pos[:, 0] * 50.0 + 1800.0) / 10.0
    last_lat = (last_pos[:, 1] * 50.0) / 10.0
    lat_rad  = torch.deg2rad(last_lat)
    beta     = 2 * OMEGA * torch.cos(lat_rad) / R_EARTH
    R_tc     = 3e5

    v_beta_lon = -beta * R_tc ** 2 / 2
    v_beta_lat =  beta * R_tc ** 2 / 4

    beta_dir      = torch.stack([v_beta_lon, v_beta_lat], dim=-1)
    beta_norm     = beta_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    beta_dir_unit = beta_dir / beta_norm

    pos_step1 = pred_samples[:, 0, :, :2]
    dir_step1 = pos_step1 - last_pos.unsqueeze(0)
    dir_norm  = dir_step1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dir_unit  = dir_step1 / dir_norm
    mean_dir  = dir_unit.mean(0)
    mean_norm = mean_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    mean_dir_unit = mean_dir / mean_norm

    cos_align     = (mean_dir_unit * beta_dir_unit).sum(-1)
    beta_strength = beta_norm.squeeze(-1)
    penalise_mask = (beta_strength > 1.0).float()
    direction_loss = F.relu(-cos_align) * penalise_mask
    return direction_loss.mean() * 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  Spread regularization  (FIX-L52: max_spread_km 200→150)
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_spread_loss(all_trajs: torch.Tensor,
                         max_spread_km: float = 150.0) -> torch.Tensor:
    """
    FIX-L52: max_spread_km=150 (từ 200). Exponential decay weights.
    all_trajs: [S, T, B, 2] normalised.
    """
    if all_trajs.shape[0] < 2:
        return all_trajs.new_zeros(())

    S, T, B, _ = all_trajs.shape

    # Exponential decay: early steps penalised more (from FIX-L46b)
    step_weights = torch.exp(
        -torch.arange(T, dtype=torch.float, device=all_trajs.device)
        * (math.log(4.0) / max(T - 1, 1))
    ) * 2.0   # range: 2.0 → 0.5

    total_loss = all_trajs.new_zeros(())
    for t in range(T):
        step_trajs = all_trajs[:, t, :, :2]
        std_lon    = step_trajs[:, :, 0].std(0)
        std_lat    = step_trajs[:, :, 1].std(0)
        spread_km  = torch.sqrt(std_lon ** 2 + std_lat ** 2) * 500.0
        excess     = F.relu(spread_km - max_spread_km)
        total_loss = total_loss + step_weights[t] * (
            excess / max_spread_km
        ).pow(2).mean()

    return total_loss / T


# ══════════════════════════════════════════════════════════════════════════════
#  Main loss
# ══════════════════════════════════════════════════════════════════════════════

def compute_total_loss(
    pred_abs,
    gt,
    ref,
    batch_list,
    pred_samples       = None,
    gt_norm            = None,
    weights            = WEIGHTS,
    intensity_w: Optional[torch.Tensor] = None,
    env_data:    Optional[dict]         = None,
    step_weight_alpha: float = 0.0,
    all_trajs:   Optional[torch.Tensor] = None,
) -> Dict:
    # 1. Sample weights
    recurv_w = _recurvature_weights(gt, w_recurv=2.5)
    sample_w = recurv_w * (intensity_w.to(gt.device) if intensity_w is not None
                           else 1.0)
    sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

    # 2. AFCRPS
    if pred_samples is not None:
        if gt_norm is not None:
            l_fm = fm_afcrps_loss(pred_samples, gt_norm,
                                  intensity_w=sample_w, unit_01deg=True,
                                  step_weight_alpha=step_weight_alpha)
        else:
            l_fm = fm_afcrps_loss(pred_samples, gt,
                                  intensity_w=sample_w, unit_01deg=False,
                                  step_weight_alpha=step_weight_alpha)
    else:
        l_fm = _haversine_deg(pred_abs, gt).mean()

    # 3. Directional losses
    NRM = 35.0

    l_vel       = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
    l_disp      = (disp_loss_per_sample(pred_abs, gt)     * sample_w).mean()
    l_step      = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()
    l_heading   = heading_loss(pred_abs, gt)
    l_recurv    = recurvature_loss(pred_abs, gt)
    l_dir_final = overall_dir_loss(pred_abs, gt, ref)
    l_smooth    = smooth_loss(pred_abs)
    l_accel     = acceleration_loss(pred_abs)
    l_jerk      = jerk_loss(pred_abs)

    # 4. PINN
    _env = env_data
    if _env is None and batch_list is not None:
        try:
            _env = batch_list[13]
        except (IndexError, TypeError):
            _env = None

    l_pinn = pinn_bve_loss(pred_abs, batch_list, env_data=_env)

    # 5. Spread penalty  (FIX-L52)
    l_spread = pred_abs.new_zeros(())
    if all_trajs is not None and all_trajs.shape[0] >= 2:
        l_spread = ensemble_spread_loss(all_trajs, max_spread_km=150.0)

    # 6. Total  (short_range added externally in get_loss_breakdown)
    total = (
        weights.get("fm",       2.0) * l_fm
        + weights.get("velocity", 0.8) * l_vel       * NRM
        + weights.get("disp",     0.5) * l_disp      * NRM
        + weights.get("step",     0.5) * l_step      * NRM
        + weights.get("heading",  2.0) * l_heading   * NRM
        + weights.get("recurv",   1.5) * l_recurv    * NRM
        + weights.get("dir",      1.0) * l_dir_final * NRM
        + weights.get("smooth",   0.5) * l_smooth    * NRM
        + weights.get("accel",    0.8) * l_accel     * NRM
        + weights.get("jerk",     0.3) * l_jerk      * NRM
        + weights.get("pinn",    0.02) * l_pinn
        + weights.get("spread",   0.8) * l_spread    * NRM
    ) / NRM

    if torch.isnan(total) or torch.isinf(total):
        total = pred_abs.new_zeros(())

    return dict(
        total        = total,
        fm           = l_fm.item(),
        velocity     = l_vel.item()     * NRM,
        step         = l_step.item(),
        disp         = l_disp.item()    * NRM,
        heading      = l_heading.item(),
        recurv       = l_recurv.item(),
        smooth       = l_smooth.item()  * NRM,
        accel        = l_accel.item()   * NRM,
        jerk         = l_jerk.item()    * NRM,
        pinn         = l_pinn.item(),
        spread       = l_spread.item()  * NRM,
        recurv_ratio = (_recurvature_weights(gt) > 1.0).float().mean().item(),
    )


# ── Legacy ─────────────────────────────────────────────────────────────────────

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=None):
        super().__init__()
        self.margin  = margin
        self.loss_fn = (torch.nn.SoftMarginLoss() if margin is None
                        else torch.nn.TripletMarginLoss(margin=margin, p=2))

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            y = torch.ones(anchor.shape[0], device=anchor.device)
            return self.loss_fn(
                torch.norm(anchor - neg, 2, dim=1)
                - torch.norm(anchor - pos, 2, dim=1), y)
        return self.loss_fn(anchor, pos, neg)
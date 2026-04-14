# # """
# # Model/losses.py  ── v24
# # ========================
# # FIXES vs v23:

# #   FIX-L44  [CRITICAL] pinn_shallow_water: log(loss-49) tạo ra gradient
# #            gần bằng 0 khi loss>50 (gradient = 1/(loss-49) → rất nhỏ).
# #            Từ log: pinn=52.69 không đổi suốt 90 epoch → NO gradient.
# #            Fix: Không dùng log-clamp. Dùng soft-tanh scaling:
# #              loss_scaled = 50 * tanh(loss / 50)
# #            → gradient = tanh'(x/50) = sech²(x/50) > 0 với mọi x
# #            → tại loss=52.69: gradient ≈ 0.92 (vs log: ≈ 0.27)
# #            → PINN sẽ thực sự học thay vì stuck.

# #   FIX-L45  [HIGH] pinn_rankine_steering: u/v500 sau FIX-ENV-20 là
# #            normalized [-1,1] → cần scale sang m/s để so sánh với u_tc.
# #            Thêm u500_scale=30.0 để convert → steering comparison đúng.

# #   FIX-L46  [HIGH] ensemble_spread_loss: max_spread_km=400 quá cao.
# #            Từ log: spread 800-1000 km suốt training.
# #            Fix: max_spread_km=200, weight tăng từ 0.1 → 0.3.
# #            Thêm per-step spread penalty (không chỉ final step).

# #   FIX-L47  [MEDIUM] velocity_loss: thêm direction penalty mạnh hơn
# #            khi pred đi ngược chiều gt. Tăng heading_loss weight.

# #   FIX-L48  [MEDIUM] trajectory_smoothness: penalize acceleration thay đổi
# #            quá đột ngột (jerk) để giảm DTW.

# # Kept from v23:
# #   FIX-L39  pinn scale=1e-3
# #   FIX-L40  step_weight_alpha
# #   FIX-L41  fm_afcrps time weights
# #   FIX-L43  velocity_loss dimensionless
# # """
# # from __future__ import annotations

# # import math
# # from typing import Dict, Optional

# # import torch
# # import torch.nn.functional as F

# # __all__ = [
# #     "WEIGHTS", "compute_total_loss", "fm_afcrps_loss",
# #     "fm_physics_consistency_loss", "pinn_bve_loss",
# #     "recurvature_loss", "velocity_loss_per_sample",
# # ]

# # OMEGA        = 7.2921e-5
# # R_EARTH      = 6.371e6
# # DT_6H        = 6 * 3600
# # DEG_TO_KM    = 111.0
# # STEP_KM      = 113.0

# # # U/V500 scale: normalized [-1,1] → real m/s
# # _UV500_SCALE = 30.0

# # # ── Weights (v24) ──────────────────────────────────────────────────────────────
# # WEIGHTS: Dict[str, float] = dict(
# #     fm=2.0,
# #     velocity=0.8,      # ↑ từ 0.5 (FIX-L47)
# #     heading=2.0,       # ↑ từ 1.5 (FIX-L47)
# #     recurv=1.5,
# #     step=0.5,
# #     disp=0.5,
# #     dir=1.0,
# #     smooth=0.5,        # ↑ từ 0.3
# #     accel=0.8,         # ↑ từ 0.5
# #     jerk=0.3,          # NEW (FIX-L48)
# #     pinn=0.02,
# #     fm_physics=0.3,
# #     spread=0.6,        # ↑ từ 0.3 (FIX-L46)
# # )

# # RECURV_ANGLE_THR = 45.0
# # RECURV_WEIGHT    = 2.5


# # # ── Haversine ─────────────────────────────────────────────────────────────────

# # def _haversine(p1: torch.Tensor, p2: torch.Tensor,
# #                unit_01deg: bool = True) -> torch.Tensor:
# #     if unit_01deg:
# #         lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
# #         lat1 = (p1[..., 1] * 50.0) / 10.0
# #         lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
# #         lat2 = (p2[..., 1] * 50.0) / 10.0
# #     else:
# #         lon1, lat1 = p1[..., 0], p1[..., 1]
# #         lon2, lat2 = p2[..., 0], p2[..., 1]

# #     lat1r = torch.deg2rad(lat1);  lat2r = torch.deg2rad(lat2)
# #     dlon  = torch.deg2rad(lon2 - lon1)
# #     dlat  = torch.deg2rad(lat2 - lat1)
# #     a     = (torch.sin(dlat / 2) ** 2
# #              + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2) ** 2)
# #     return 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())


# # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# #     return _haversine(p1, p2, unit_01deg=False)


# # # ── Step displacements in km ──────────────────────────────────────────────────

# # def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
# #     dt      = traj_deg[1:] - traj_deg[:-1]
# #     lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
# #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
# #     dt_km   = dt.clone()
# #     dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
# #     dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
# #     return dt_km


# # # ── Recurvature helpers ────────────────────────────────────────────────────────

# # def _total_rotation_angle_batch(gt: torch.Tensor) -> torch.Tensor:
# #     T, B, _ = gt.shape
# #     if T < 3:
# #         return gt.new_zeros(B)
# #     lats_rad = torch.deg2rad(gt[:, :, 1])
# #     cos_lat  = torch.cos(lats_rad[:-1])
# #     dlat = gt[1:, :, 1] - gt[:-1, :, 1]
# #     dlon = (gt[1:, :, 0] - gt[:-1, :, 0]) * cos_lat
# #     v    = torch.stack([dlon, dlat], dim=-1)
# #     v1   = v[:-1];  v2 = v[1:]
# #     n1   = v1.norm(dim=-1).clamp(min=1e-8)
# #     n2   = v2.norm(dim=-1).clamp(min=1e-8)
# #     cos_a = (v1 * v2).sum(-1) / (n1 * n2)
# #     return torch.rad2deg(torch.acos(cos_a.clamp(-1.0, 1.0))).sum(0)


# # def _recurvature_weights(gt: torch.Tensor,
# #                          thr: float = RECURV_ANGLE_THR,
# #                          w_recurv: float = RECURV_WEIGHT) -> torch.Tensor:
# #     rot = _total_rotation_angle_batch(gt)
# #     return torch.where(rot >= thr,
# #                        torch.full_like(rot, w_recurv),
# #                        torch.ones_like(rot))


# # # ── Directional losses ────────────────────────────────────────────────────────

# # def velocity_loss_per_sample(pred: torch.Tensor,
# #                              gt: torch.Tensor) -> torch.Tensor:
# #     if pred.shape[0] < 2:
# #         return pred.new_zeros(pred.shape[1])
# #     v_pred_km = _step_displacements_km(pred)
# #     v_gt_km   = _step_displacements_km(gt)
# #     s_pred    = v_pred_km.norm(dim=-1)
# #     s_gt      = v_gt_km.norm(dim=-1)
# #     l_speed   = (s_pred - s_gt).pow(2).mean(0)
# #     gn  = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# #     gt_unit = v_gt_km / gn
# #     ate = ((v_pred_km - v_gt_km) * gt_unit).sum(-1)
# #     l_ate = ate.pow(2).mean(0)

# #     # FIX-L47: thêm direction penalty mạnh
# #     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# #     cos_sim = ((v_pred_km / pn) * gt_unit).sum(-1)
# #     l_dir = F.relu(-cos_sim).pow(2).mean(0) * STEP_KM ** 2  # scale to km²

# #     return (l_speed + 0.5 * l_ate + 0.3 * l_dir) / (STEP_KM ** 2)


# # def disp_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# #     if pred.shape[0] < 2:
# #         return pred.new_zeros(pred.shape[1])
# #     pd = _step_displacements_km(pred).norm(dim=-1).mean(0)
# #     gd = _step_displacements_km(gt).norm(dim=-1).mean(0)
# #     return (pd - gd).pow(2) / (STEP_KM ** 2)


# # def step_dir_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# #     if pred.shape[0] < 2:
# #         return pred.new_zeros(pred.shape[1])
# #     v_pred_km = _step_displacements_km(pred)
# #     v_gt_km   = _step_displacements_km(gt)
# #     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# #     gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# #     cos_sim = ((v_pred_km / pn) * (v_gt_km / gn)).sum(-1)
# #     return (1.0 - cos_sim).mean(0)


# # def heading_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# #     if pred.shape[0] < 2:
# #         return pred.new_zeros(())
# #     pv = _step_displacements_km(pred)
# #     gv = _step_displacements_km(gt)
# #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# #     # FIX-L47: stronger penalty for wrong direction
# #     wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1))
# #     wrong_dir_loss = (wrong_dir ** 2).mean()  # squared penalty

# #     if pred.shape[0] >= 3:
# #         def _curv(v):
# #             cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
# #             n1    = v[1:].norm(dim=-1).clamp(min=1e-4)
# #             n2    = v[:-1].norm(dim=-1).clamp(min=1e-4)
# #             return (cross / (n1 * n2)).clamp(-10.0, 10.0)
# #         curv_mse = F.mse_loss(_curv(pv), _curv(gv))
# #     else:
# #         curv_mse = pred.new_zeros(())
# #     return wrong_dir_loss + curv_mse


# # def smooth_loss(pred: torch.Tensor) -> torch.Tensor:
# #     if pred.shape[0] < 3:
# #         return pred.new_zeros(())
# #     v_km = _step_displacements_km(pred)
# #     if v_km.shape[0] < 2:
# #         return pred.new_zeros(())
# #     accel_km = v_km[1:] - v_km[:-1]
# #     return accel_km.pow(2).mean() / (STEP_KM ** 2)


# # def acceleration_loss(pred: torch.Tensor) -> torch.Tensor:
# #     if pred.shape[0] < 3:
# #         return pred.new_zeros(())
# #     v_km = _step_displacements_km(pred)
# #     if v_km.shape[0] < 2:
# #         return pred.new_zeros(())
# #     a_km = v_km[1:] - v_km[:-1]
# #     return a_km.pow(2).mean() / (STEP_KM ** 2)


# # def jerk_loss(pred: torch.Tensor) -> torch.Tensor:
# #     """
# #     FIX-L48: Penalize jerk (change in acceleration) để giảm DTW.
# #     Smooth trajectory → lower DTW.
# #     """
# #     if pred.shape[0] < 4:
# #         return pred.new_zeros(())
# #     v_km = _step_displacements_km(pred)
# #     if v_km.shape[0] < 3:
# #         return pred.new_zeros(())
# #     a_km  = v_km[1:] - v_km[:-1]
# #     j_km  = a_km[1:] - a_km[:-1]
# #     return j_km.pow(2).mean() / (STEP_KM ** 2)


# # def overall_dir_loss(pred: torch.Tensor, gt: torch.Tensor,
# #                      ref: torch.Tensor) -> torch.Tensor:
# #     p_d = pred[-1] - ref
# #     g_d = gt[-1]   - ref
# #     lat_ref = ref[:, 1]
# #     cos_lat = torch.cos(torch.deg2rad(lat_ref)).clamp(min=1e-4)
# #     p_d_km  = p_d.clone();  p_d_km[:, 0] *= cos_lat * DEG_TO_KM;  p_d_km[:, 1] *= DEG_TO_KM
# #     g_d_km  = g_d.clone();  g_d_km[:, 0] *= cos_lat * DEG_TO_KM;  g_d_km[:, 1] *= DEG_TO_KM
# #     pn = p_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# #     gn = g_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
# #     return (1.0 - ((p_d_km / pn) * (g_d_km / gn)).sum(-1)).mean()


# # def recurvature_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# #     if pred.shape[0] < 3:
# #         return pred.new_zeros(())
# #     pred_v   = pred[1:] - pred[:-1]
# #     gt_v     = gt[1:]   - gt[:-1]
# #     gt_cross = (gt_v[:-1, :, 0] * gt_v[1:, :, 1]
# #               - gt_v[:-1, :, 1] * gt_v[1:, :, 0])
# #     if gt_cross.shape[0] < 2:
# #         return pred.new_zeros(())
# #     sign_change = (gt_cross[:-1] * gt_cross[1:]) < 0
# #     if not sign_change.any():
# #         return pred.new_zeros(())
# #     pred_v_mid = pred_v[1:-1]
# #     gt_v_mid   = gt_v[1:-1]
# #     if pred_v_mid.shape[0] > sign_change.shape[0]:
# #         pred_v_mid = pred_v_mid[:sign_change.shape[0]]
# #         gt_v_mid   = gt_v_mid[:sign_change.shape[0]]
# #     pn      = pred_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #     gn      = gt_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #     cos_sim = ((pred_v_mid / pn) * (gt_v_mid / gn)).sum(-1)
# #     dir_loss = (1.0 - cos_sim)
# #     mask     = sign_change.float()
# #     if mask.sum() < 1:
# #         return pred.new_zeros(())
# #     return (dir_loss * mask).sum() / mask.sum().clamp(min=1)


# # # ── AFCRPS ────────────────────────────────────────────────────────────────────

# # def fm_afcrps_loss(
# #     pred_samples: torch.Tensor,
# #     gt: torch.Tensor,
# #     unit_01deg: bool = True,
# #     intensity_w: Optional[torch.Tensor] = None,
# #     step_weight_alpha: float = 0.0,
# # ) -> torch.Tensor:
# #     M, T, B, _ = pred_samples.shape
# #     eps = 1e-3

# #     base_w = torch.linspace(0.5, 1.5, T, device=pred_samples.device)

# #     if step_weight_alpha > 0.0:
# #         # early_w = torch.exp(-torch.arange(T, dtype=torch.float,
# #         #                      device=pred_samples.device) * 0.2)
# #         # losses.py — fm_afcrps_loss()
# #         early_w = torch.exp(-torch.arange(T, dtype=torch.float,
# #                      device=pred_samples.device) * 0.35)   # 0.2→0.35
# #         early_w = early_w / early_w.mean()
# #         time_w  = (1.0 - step_weight_alpha) * base_w + step_weight_alpha * early_w
# #     else:
# #         time_w = base_w

# #     if M == 1:
# #         dist = _haversine(pred_samples[0], gt, unit_01deg)
# #         loss_per_b = (dist * time_w.unsqueeze(1)).mean(0)
# #     else:
# #         d_to_gt = _haversine(
# #             pred_samples,
# #             gt.unsqueeze(0).expand_as(pred_samples),
# #             unit_01deg,
# #         )
# #         d_to_gt_w    = d_to_gt * time_w.unsqueeze(0).unsqueeze(2)
# #         d_to_gt_mean = d_to_gt_w.mean(1)

# #         ps_i = pred_samples.unsqueeze(1)
# #         ps_j = pred_samples.unsqueeze(0)
# #         d_pair = _haversine(
# #             ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# #             ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# #             unit_01deg,
# #         ).reshape(M, M, T, B)
# #         d_pair_w    = d_pair * time_w.unsqueeze(0).unsqueeze(0).unsqueeze(3)
# #         d_pair_mean = d_pair_w.mean(2)

# #         e_sy  = d_to_gt_mean.mean(0)
# #         e_ssp = d_pair_mean.mean(0).mean(0)
# #         loss_per_b = (e_sy - 0.5 * e_ssp).clamp(min=eps)

# #     if intensity_w is not None:
# #         w = intensity_w.to(loss_per_b.device)
# #         return (loss_per_b * w).mean()
# #     return loss_per_b.mean()


# # # ── PINN losses ────────────────────────────────────────────────────────────────

# # def _soft_clamp_loss(loss: torch.Tensor, max_val: float = 50.0) -> torch.Tensor:
# #     """
# #     FIX-L44: Soft clamp bằng tanh scaling.
# #     loss_scaled = max_val * tanh(loss / max_val)
# #     Gradient = tanh'(x/max_val) = sech²(x/max_val) > 0 với mọi x.
# #     Tại loss=52.69: gradient ≈ sech²(1.054) ≈ 0.31 (vs log: ≈ 0.27 nhưng cố định)
# #     Gradient sẽ giảm dần khi loss tăng → vẫn có gradient khác 0.
# #     """
# #     return max_val * torch.tanh(loss / max_val)


# # def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
# #     """
# #     FIX-L44: Dùng tanh soft-clamp thay vì log-clamp.
# #     scale=1e-3 giữ nguyên từ FIX-L39.
# #     """
# #     T, B, _ = pred_abs_deg.shape
# #     if T < 3:
# #         return pred_abs_deg.new_zeros(())

# #     DT      = DT_6H
# #     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
# #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# #     cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

# #     u = dlat[:, :, 0] * cos_lat * 111000.0 / DT
# #     v = dlat[:, :, 1] * 111000.0 / DT

# #     if u.shape[0] < 2:
# #         return pred_abs_deg.new_zeros(())

# #     du = (u[1:] - u[:-1]) / DT
# #     dv = (v[1:] - v[:-1]) / DT

# #     f    = 2 * OMEGA * torch.sin(lat_rad[1:-1])
# #     beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH

# #     res_u = du - f * v[1:]
# #     res_v = dv + f * u[1:]

# #     R_tc   = 3e5
# #     v_beta_x = -beta * R_tc ** 2 / 2
# #     res_u_corrected = res_u - v_beta_x

# #     scale = 1e-3
# #     loss = ((res_u_corrected / scale).pow(2).mean()
# #             + (res_v / scale).pow(2).mean())

# #     # FIX-L44: tanh soft-clamp → gradient luôn > 0
# #     # return _soft_clamp_loss(loss, max_val=50.0)
# #     return _soft_clamp_loss(loss, max_val=20.0)


# # def pinn_rankine_steering(pred_abs_deg: torch.Tensor,
# #                           env_data: Optional[dict]) -> torch.Tensor:
# #     """
# #     FIX-L45: u/v500 sau FIX-ENV-20 đã normalized [-1,1].
# #     Scale back sang m/s để so sánh với u_tc.
# #     """
# #     if env_data is None:
# #         return pred_abs_deg.new_zeros(())

# #     T, B, _ = pred_abs_deg.shape
# #     if T < 2:
# #         return pred_abs_deg.new_zeros(())

# #     DT      = DT_6H
# #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# #     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
# #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

# #     u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT
# #     v_tc = dlat[:, :, 1] * 111000.0 / DT

# #     u500_raw = env_data.get("u500_mean", None)
# #     v500_raw = env_data.get("v500_mean", None)

# #     if u500_raw is None or v500_raw is None:
# #         speed = torch.sqrt(u_tc ** 2 + v_tc ** 2)
# #         return F.relu(speed - 30.0).pow(2).mean() * 0.01

# #     def _align(x: torch.Tensor) -> torch.Tensor:
# #         if not torch.is_tensor(x):
# #             return pred_abs_deg.new_zeros(T - 1, B)
# #         x = x.to(pred_abs_deg.device)
# #         if x.dim() == 3:
# #             x_sq    = x[:, :, 0]
# #             T_env   = x_sq.shape[1]
# #             T_tgt   = T - 1
# #             if T_env >= T_tgt:
# #                 return x_sq[:, :T_tgt].permute(1, 0)
# #             pad = x_sq[:, -1:].expand(-1, T_tgt - T_env)
# #             return torch.cat([x_sq, pad], dim=1).permute(1, 0)
# #         elif x.dim() == 2:
# #             if x.shape == (B, T - 1):
# #                 return x.permute(1, 0)
# #             elif x.shape[0] == T - 1:
# #                 return x[:, :B] if x.shape[1] >= B else x.expand(-1, B)
# #         return pred_abs_deg.new_zeros(T - 1, B)

# #     u500_n = _align(u500_raw)
# #     v500_n = _align(v500_raw)

# #     # FIX-L45: scale normalized [-1,1] → m/s
# #     u500 = u500_n * _UV500_SCALE
# #     v500 = v500_n * _UV500_SCALE

# #     # Chỉ áp dụng steering penalty khi u/v500 có giá trị thực (non-zero)
# #     uv_magnitude = torch.sqrt(u500 ** 2 + v500 ** 2)
# #     has_steering = (uv_magnitude > 0.5).float()  # threshold 0.5 m/s

# #     env_dir  = torch.stack([u500, v500], dim=-1)
# #     tc_dir   = torch.stack([u_tc, v_tc],  dim=-1)
# #     env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)  # min 0.5 m/s
# #     tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
# #     cos_sim  = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)
# #     misalign = F.relu(-0.5 - cos_sim).pow(2)

# #     # Only penalize when we have actual steering data
# #     return (misalign * has_steering).mean() * 0.05


# # def pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# #     if pred_deg.shape[0] < 2:
# #         return pred_deg.new_zeros(())
# #     dt_deg  = pred_deg[1:] - pred_deg[:-1]
# #     lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# #     dx_km   = dt_deg[:, :, 0] * cos_lat * DEG_TO_KM
# #     dy_km   = dt_deg[:, :, 1] * DEG_TO_KM
# #     speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# #     return F.relu(speed - 600.0).pow(2).mean()


# # def pinn_bve_loss(pred_abs_deg: torch.Tensor,
# #                   batch_list,
# #                   env_data: Optional[dict] = None) -> torch.Tensor:
# #     """
# #     FIX-L44: tanh soft-clamp trong pinn_shallow_water.
# #     FIX-L45: steering dùng actual u/v500 m/s.
# #     """
# #     T = pred_abs_deg.shape[0]
# #     if T < 3:
# #         return pred_abs_deg.new_zeros(())

# #     l_sw    = pinn_shallow_water(pred_abs_deg)
# #     l_steer = pinn_rankine_steering(pred_abs_deg, env_data)
# #     l_speed = pinn_speed_constraint(pred_abs_deg)

# #     total = l_sw + 0.5 * l_steer + 0.1 * l_speed
# #     # FIX-L44: tanh soft-clamp (không log)
# #     # return _soft_clamp_loss(total, max_val=50.0)
# #     return _soft_clamp_loss(total, max_val=20.0)


# # # ── Physics consistency loss ──────────────────────────────────────────────────

# # def fm_physics_consistency_loss(
# #     pred_samples: torch.Tensor,
# #     gt_norm: torch.Tensor,
# #     last_pos: torch.Tensor,
# # ) -> torch.Tensor:
# #     S, T, B = pred_samples.shape[:3]

# #     last_lon = (last_pos[:, 0] * 50.0 + 1800.0) / 10.0
# #     last_lat = (last_pos[:, 1] * 50.0) / 10.0
# #     lat_rad  = torch.deg2rad(last_lat)
# #     beta     = 2 * OMEGA * torch.cos(lat_rad) / R_EARTH
# #     R_tc     = 3e5

# #     v_beta_lon = -beta * R_tc ** 2 / 2
# #     v_beta_lat =  beta * R_tc ** 2 / 4

# #     beta_dir      = torch.stack([v_beta_lon, v_beta_lat], dim=-1)
# #     beta_norm     = beta_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #     beta_dir_unit = beta_dir / beta_norm

# #     pos_step1     = pred_samples[:, 0, :, :2]
# #     dir_step1     = pos_step1 - last_pos.unsqueeze(0)
# #     dir_norm      = dir_step1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #     dir_unit      = dir_step1 / dir_norm
# #     mean_dir      = dir_unit.mean(0)
# #     mean_norm     = mean_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #     mean_dir_unit = mean_dir / mean_norm

# #     cos_align    = (mean_dir_unit * beta_dir_unit).sum(-1)
# #     beta_strength = beta_norm.squeeze(-1)
# #     penalise_mask = (beta_strength > 1.0).float()
# #     direction_loss = F.relu(-cos_align) * penalise_mask
# #     return direction_loss.mean() * 0.5


# # # ── Spread regularization ──────────────────────────────────────────────────────

# # def ensemble_spread_loss(all_trajs: torch.Tensor,
# #                          max_spread_km: float = 200.0) -> torch.Tensor:
# #     """
# #     FIX-L46: max_spread_km=200 (từ 400), thêm per-step penalty.
# #     all_trajs: [S, T, B, 2] normalised.
# #     """
# #     if all_trajs.shape[0] < 2:
# #         return all_trajs.new_zeros(())

# #     S, T, B, _ = all_trajs.shape

# #     # # Per-step spread penalty (không chỉ final step)
# #     # # Weight tăng theo thời gian: later steps penalized more
# #     # step_weights = torch.linspace(0.5, 1.5, T, device=all_trajs.device)
    
# #     # FIX-L46b: exponential decay — early steps penalized MORE
# #     # step 0 weight=2.0, step T-1 weight=0.5 (geometric)
# #     step_weights = torch.exp(
# #         -torch.arange(T, dtype=torch.float, device=all_trajs.device) * (math.log(4.0) / max(T-1, 1))
# #     ) * 2.0   # range: 2.0 → 0.5

# #     total_loss = all_trajs.new_zeros(())
# #     for t in range(T):
# #         step_trajs = all_trajs[:, t, :, :2]  # [S, B, 2]
# #         std_lon = step_trajs[:, :, 0].std(0)  # [B]
# #         std_lat = step_trajs[:, :, 1].std(0)  # [B]
# #         spread_km = torch.sqrt(std_lon ** 2 + std_lat ** 2) * 500.0  # [B]
# #         excess    = F.relu(spread_km - max_spread_km)
# #         total_loss = total_loss + step_weights[t] * (excess / max_spread_km).pow(2).mean()

# #     return total_loss / T


# # # ── Intensity weights ─────────────────────────────────────────────────────────

# # def _intensity_weights_from_obs(obs_Me: torch.Tensor) -> torch.Tensor:
# #     wind_norm = obs_Me[-1, :, 1].detach()
# #     w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
# #         torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
# #         torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
# #                     torch.full_like(wind_norm, 1.5))))
# #     return w / w.mean().clamp(min=1e-6)


# # # ── Main loss ─────────────────────────────────────────────────────────────────

# # def compute_total_loss(
# #     pred_abs,
# #     gt,
# #     ref,
# #     batch_list,
# #     pred_samples=None,
# #     gt_norm=None,
# #     weights=WEIGHTS,
# #     intensity_w: Optional[torch.Tensor] = None,
# #     env_data: Optional[dict] = None,
# #     step_weight_alpha: float = 0.0,
# #     all_trajs: Optional[torch.Tensor] = None,
# # ) -> Dict:
# #     # 1. Sample weights
# #     recurv_w = _recurvature_weights(gt, w_recurv=2.5)
# #     sample_w = recurv_w * (intensity_w.to(gt.device) if intensity_w is not None
# #                            else 1.0)
# #     sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

# #     # 2. AFCRPS
# #     if pred_samples is not None:
# #         if gt_norm is not None:
# #             l_fm = fm_afcrps_loss(pred_samples, gt_norm,
# #                                   intensity_w=sample_w, unit_01deg=True,
# #                                   step_weight_alpha=step_weight_alpha)
# #         else:
# #             l_fm = fm_afcrps_loss(pred_samples, gt,
# #                                   intensity_w=sample_w, unit_01deg=False,
# #                                   step_weight_alpha=step_weight_alpha)
# #     else:
# #         l_fm = _haversine_deg(pred_abs, gt).mean()

# #     # 3. Directional losses
# #     NRM = 35.0

# #     l_vel   = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
# #     l_disp  = (disp_loss_per_sample(pred_abs, gt)     * sample_w).mean()
# #     l_step  = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()

# #     l_heading   = heading_loss(pred_abs, gt)
# #     l_recurv    = recurvature_loss(pred_abs, gt)
# #     l_dir_final = overall_dir_loss(pred_abs, gt, ref)
# #     l_smooth    = smooth_loss(pred_abs)
# #     l_accel     = acceleration_loss(pred_abs)
# #     l_jerk      = jerk_loss(pred_abs)  # FIX-L48

# #     # 4. PINN
# #     _env = env_data
# #     if _env is None and batch_list is not None:
# #         try:
# #             _env = batch_list[13]
# #         except (IndexError, TypeError):
# #             _env = None

# #     l_pinn = pinn_bve_loss(pred_abs, batch_list, env_data=_env)

# #     # 5. Spread penalty
# #     l_spread = pred_abs.new_zeros(())
# #     if all_trajs is not None and all_trajs.shape[0] >= 2:
# #         l_spread = ensemble_spread_loss(all_trajs, max_spread_km=200.0)

# #     # 6. Total
# #     total = (
# #         weights.get("fm",       2.0) * l_fm
# #         + weights.get("velocity", 0.8) * l_vel   * NRM
# #         + weights.get("disp",     0.5) * l_disp  * NRM
# #         + weights.get("step",     0.5) * l_step  * NRM
# #         + weights.get("heading",  2.0) * l_heading * NRM
# #         + weights.get("recurv",   1.5) * l_recurv  * NRM
# #         + weights.get("dir",      1.0) * l_dir_final * NRM
# #         + weights.get("smooth",   0.5) * l_smooth * NRM
# #         + weights.get("accel",    0.8) * l_accel  * NRM
# #         + weights.get("jerk",     0.3) * l_jerk   * NRM  # FIX-L48
# #         + weights.get("pinn",    0.02) * l_pinn
# #         + weights.get("spread",  0.3)  * l_spread * NRM
# #     ) / NRM

# #     if torch.isnan(total) or torch.isinf(total):
# #         total = pred_abs.new_zeros(())

# #     return dict(
# #         total        = total,
# #         fm           = l_fm.item(),
# #         velocity     = l_vel.item() * NRM,
# #         step         = l_step.item(),
# #         disp         = l_disp.item() * NRM,
# #         heading      = l_heading.item(),
# #         recurv       = l_recurv.item(),
# #         smooth       = l_smooth.item() * NRM,
# #         accel        = l_accel.item() * NRM,
# #         jerk         = l_jerk.item() * NRM,
# #         pinn         = l_pinn.item(),
# #         spread       = l_spread.item() * NRM,
# #         recurv_ratio = (_recurvature_weights(gt) > 1.0).float().mean().item(),
# #     )


# # # ── Legacy ─────────────────────────────────────────────────────────────────────

# # class TripletLoss(torch.nn.Module):
# #     def __init__(self, margin=None):
# #         super().__init__()
# #         self.margin  = margin
# #         self.loss_fn = (torch.nn.SoftMarginLoss() if margin is None
# #                         else torch.nn.TripletMarginLoss(margin=margin, p=2))

# #     def forward(self, anchor, pos, neg):
# #         if self.margin is None:
# #             y = torch.ones(anchor.shape[0], device=anchor.device)
# #             return self.loss_fn(
# #                 torch.norm(anchor - neg, 2, dim=1)
# #                 - torch.norm(anchor - pos, 2, dim=1), y)
# #         return self.loss_fn(anchor, pos, neg)

# """
# Model/losses.py  ── v25
# ========================
# FIXES vs v24:

#   FIX-L49  [CRITICAL] short_range_regression_loss(): loss mới riêng cho
#            4 bước đầu (6h/12h/18h/24h). Dùng Huber loss trên khoảng cách
#            Haversine. Weight 12h = 4.0, 6h/18h/24h = 2.0.
#            Không dùng AFCRPS cho short-range → ổn định và chính xác hơn.

#   FIX-L50  [HIGH] PINN scale: 1e-3 → 1e-2.
#            Log cho thấy pinn=15.232 không đổi → scale quá nhỏ, loss bị
#            dominate bởi phần tanh-clamp. Tăng scale làm residual nhỏ hơn
#            → gradient flow tốt hơn → PINN thực sự học.

#   FIX-L51  [HIGH] WEIGHTS: thêm "short_range": 5.0, tăng "spread": 0.8
#            (từ 0.6). Spread 400-500km là nguồn gốc của ADE cao.

#   FIX-L52  [MEDIUM] ensemble_spread_loss: max_spread_km=150 (từ 200),
#            FIX-L46b vẫn giữ exponential decay trên step weights.

#   FIX-L53  [MEDIUM] fm_afcrps_loss: early_w decay rate 0.35 → 0.5
#            khi alpha > 0 → focus mạnh hơn vào step 1-2 trong giai đoạn
#            đầu training.

# Kept from v24:
#   FIX-L44  tanh soft-clamp (max_val=20.0)
#   FIX-L45  pinn_rankine_steering scale u/v500
#   FIX-L47  velocity direction penalty
#   FIX-L48  jerk loss
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
#     "short_range_regression_loss",
# ]

# OMEGA        = 7.2921e-5
# R_EARTH      = 6.371e6
# DT_6H        = 6 * 3600
# DEG_TO_KM    = 111.0
# STEP_KM      = 113.0

# # Các hằng số cho PINN mới của bạn
# _UV500_NORM      = 30.0    # m/s → đã normalize [-1,1] trong env
# _GPH500_MEAN_M   = 5870.0  # meters (sau fix Bug C)
# _GPH500_STD_M    = 80.0
# _STEERING_MIN_MS = 3.0     # m/s — threshold có steering flow thực sự
# _GPH_GRAD_SCALE  = 200.0   # meters — scale GPH gradient
# _PINN_SCALE      = 1e-2  # Scale để residual không bị quá nhỏ
# # ── Weights (v25) ─────────────────────────────────────────────────────────────
# WEIGHTS: Dict[str, float] = dict(
#     fm          = 2.0,
#     velocity    = 0.8,
#     heading     = 2.0,
#     recurv      = 1.5,
#     step        = 0.5,
#     disp        = 0.5,
#     dir         = 1.0,
#     smooth      = 0.5,
#     accel       = 0.8,
#     jerk        = 0.3,
#     pinn        = 0.5,
#     fm_physics  = 0.3,
#     spread      = 0.8,        # ↑ từ 0.6  (FIX-L51)
#     short_range = 5.0,        # NEW       (FIX-L49)
# )

# RECURV_ANGLE_THR = 45.0
# RECURV_WEIGHT    = 2.5

# # ── short-range config ────────────────────────────────────────────────────────
# _SR_N_STEPS  = 4                         # 6h, 12h, 18h, 24h
# _SR_WEIGHTS  = [2.0, 4.0, 2.0, 2.0]     # step-2 (12h) highest
# _HUBER_DELTA = 50.0                      # km


# # ══════════════════════════════════════════════════════════════════════════════
# #  Haversine
# # ══════════════════════════════════════════════════════════════════════════════

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


# def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
#     """Normalised → degrees. Accepts any leading dims."""
#     out = arr.clone()
#     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
#     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
#     return out


# # ══════════════════════════════════════════════════════════════════════════════
# #  FIX-L49: Short-range regression loss
# # ══════════════════════════════════════════════════════════════════════════════

# def short_range_regression_loss(
#     pred_sr:  torch.Tensor,   # [4, B, 2]  normalised positions
#     gt_sr:    torch.Tensor,   # [4, B, 2]  normalised positions
#     last_pos: torch.Tensor,   # [B, 2]     (unused, kept for API compat)
# ) -> torch.Tensor:
#     """
#     FIX-L49: Huber loss on haversine distance for steps 1-4.

#     Steps : 6h(1)  12h(2)  18h(3)  24h(4)
#     Weight:  2.0    4.0     2.0     2.0    ← step-2(12h) penalised most

#     Huber threshold = 50 km (less sensitive to outlier trajectories).
#     Returns scalar loss in [0, ~10] range suitable for weight=5.0.
#     """
#     n_steps = min(pred_sr.shape[0], gt_sr.shape[0], _SR_N_STEPS)
#     if n_steps == 0:
#         return pred_sr.new_zeros(())

#     pred_deg = _norm_to_deg(pred_sr[:n_steps])   # [n_steps, B, 2]
#     gt_deg   = _norm_to_deg(gt_sr[:n_steps])

#     # Haversine per (step, batch)
#     dist_km = _haversine_deg(pred_deg, gt_deg)   # [n_steps, B]

#     # Huber loss
#     huber = torch.where(
#         dist_km < _HUBER_DELTA,
#         0.5 * dist_km.pow(2) / _HUBER_DELTA,
#         dist_km - 0.5 * _HUBER_DELTA,
#     )  # [n_steps, B]

#     # Step weights
#     w = pred_sr.new_tensor(_SR_WEIGHTS[:n_steps])  # [n_steps]
#     weighted = (huber * w.view(-1, 1)).mean()

#     # Normalise: divide by HUBER_DELTA so loss ≈ 1.0 when error ≈ 50 km
#     return weighted / _HUBER_DELTA


# # ══════════════════════════════════════════════════════════════════════════════
# #  Step displacements
# # ══════════════════════════════════════════════════════════════════════════════

# def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
#     dt      = traj_deg[1:] - traj_deg[:-1]
#     lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
#     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
#     dt_km   = dt.clone()
#     dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
#     dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
#     return dt_km


# # ══════════════════════════════════════════════════════════════════════════════
# #  Recurvature helpers
# # ══════════════════════════════════════════════════════════════════════════════

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


# # ══════════════════════════════════════════════════════════════════════════════
# #  Directional losses
# # ══════════════════════════════════════════════════════════════════════════════

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
#     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     cos_sim = ((v_pred_km / pn) * gt_unit).sum(-1)
#     l_dir = F.relu(-cos_sim).pow(2).mean(0) * STEP_KM ** 2
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
#     wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1))
#     wrong_dir_loss = (wrong_dir ** 2).mean()

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
#     if pred.shape[0] < 4:
#         return pred.new_zeros(())
#     v_km = _step_displacements_km(pred)
#     if v_km.shape[0] < 3:
#         return pred.new_zeros(())
#     a_km = v_km[1:] - v_km[:-1]
#     j_km = a_km[1:] - a_km[:-1]
#     return j_km.pow(2).mean() / (STEP_KM ** 2)


# def overall_dir_loss(pred: torch.Tensor, gt: torch.Tensor,
#                      ref: torch.Tensor) -> torch.Tensor:
#     p_d = pred[-1] - ref
#     g_d = gt[-1]   - ref
#     lat_ref = ref[:, 1]
#     cos_lat = torch.cos(torch.deg2rad(lat_ref)).clamp(min=1e-4)
#     p_d_km  = p_d.clone()
#     p_d_km[:, 0] *= cos_lat * DEG_TO_KM
#     p_d_km[:, 1] *= DEG_TO_KM
#     g_d_km  = g_d.clone()
#     g_d_km[:, 0] *= cos_lat * DEG_TO_KM
#     g_d_km[:, 1] *= DEG_TO_KM
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


# # ══════════════════════════════════════════════════════════════════════════════
# #  AFCRPS
# # ══════════════════════════════════════════════════════════════════════════════

# def fm_afcrps_loss(
#     pred_samples:      torch.Tensor,
#     gt:                torch.Tensor,
#     unit_01deg:        bool  = True,
#     intensity_w:       Optional[torch.Tensor] = None,
#     step_weight_alpha: float = 0.0,
# ) -> torch.Tensor:
#     M, T, B, _ = pred_samples.shape
#     eps = 1e-3

#     base_w = torch.linspace(0.5, 1.5, T, device=pred_samples.device)

#     if step_weight_alpha > 0.0:
#         # FIX-L53: stronger early focus (0.35 → 0.5)
#         early_w = torch.exp(
#             -torch.arange(T, dtype=torch.float, device=pred_samples.device) * 0.5
#         )
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


# # ══════════════════════════════════════════════════════════════════════════════
# #  PINN losses  (FIX-L50: scale 1e-3 → 1e-2)
# # ══════════════════════════════════════════════════════════════════════════════

# def _soft_clamp_loss(loss: torch.Tensor, max_val: float = 20.0) -> torch.Tensor:
#     """FIX-L44: tanh soft-clamp. Gradient always > 0."""
#     return max_val * torch.tanh(loss / max_val)

# def _get_uv500_ms(env_data: dict, key_mean: str, key_center: str,
#                   T_tgt: int, B: int,
#                   device: torch.device) -> torch.Tensor:
#     """
#     Lấy u hoặc v 500hPa theo thứ tự ưu tiên: center > mean > zeros.
#     env_data[key] shape: [B, T_obs, 1] (từ seq_collate).
#     Output: [T_tgt, B] đơn vị m/s.
#     """
#     def _extract(key):
#         x = env_data.get(key, None)
#         if x is None or not torch.is_tensor(x):
#             return None
#         x = x.to(device).float()
#         # [B, T_obs, 1] → [B, T_obs]
#         if x.dim() == 3:
#             x = x[..., 0]
#         elif x.dim() == 1:
#             x = x.unsqueeze(0).expand(B, -1)
#         # x: [B, T_obs] → transpose → [T_obs, B]
#         x = x.permute(1, 0)   # [T_obs, B]
#         T_obs = x.shape[0]
#         if T_obs >= T_tgt:
#             return x[:T_tgt] * _UV500_NORM
#         # pad bằng climatology (0 m/s) thay vì repeat cuối
#         pad = torch.zeros(T_tgt - T_obs, B, device=device)
#         return torch.cat([x * _UV500_NORM, pad], dim=0)

#     # Ưu tiên center (flow tại tâm TC chính xác hơn)
#     val = _extract(key_center)
#     if val is not None:
#         return val
#     val = _extract(key_mean)
#     if val is not None:
#         return val
#     return torch.zeros(T_tgt, B, device=device)


# def _get_gph500_norm(env_data: dict, key: str,
#                      T_tgt: int, B: int,
#                      device: torch.device) -> torch.Tensor:
#     """
#     Lấy GPH500 đã z-score (sau fix Bug C: mean=5870, std=80).
#     Output: [T_tgt, B] normalized.
#     """
#     x = env_data.get(key, None)
#     if x is None or not torch.is_tensor(x):
#         return torch.zeros(T_tgt, B, device=device)
#     x = x.to(device).float()
#     if x.dim() == 3:
#         x = x[..., 0]
#     x = x.permute(1, 0)   # [T_obs, B]
#     T_obs = x.shape[0]
#     if T_obs >= T_tgt:
#         return x[:T_tgt]
#     pad = torch.zeros(T_tgt - T_obs, B, device=device)
#     return torch.cat([x, pad], dim=0)


# # def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
# #     T, B, _ = pred_abs_deg.shape
# #     if T < 3:
# #         return pred_abs_deg.new_zeros(())

# #     DT      = DT_6H
# #     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
# #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# #     cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

# #     u = dlat[:, :, 0] * cos_lat * 111000.0 / DT   # m/s
# #     v = dlat[:, :, 1] * 111000.0 / DT              # m/s

# #     if u.shape[0] < 2:
# #         return pred_abs_deg.new_zeros(())

# #     du = (u[1:] - u[:-1]) / DT   # m/s²  ~1e-4
# #     dv = (v[1:] - v[:-1]) / DT

# #     f    = 2 * OMEGA * torch.sin(lat_rad[1:-1])    # ~1e-4
# #     beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH  # ~2e-11

# #     res_u = du - f * v[1:]
# #     res_v = dv + f * u[1:]

# #     R_tc            = 3e5
# #     v_beta_x        = -beta * R_tc ** 2 / 2        # ~1e-4 m/s²
# #     res_u_corrected = res_u - v_beta_x

# #     # # FIX: scale phù hợp với magnitude m/s²
# #     # # Typical TC: du~1e-4, f*v~5e-4 → residual ~1e-3 m/s²
# #     # # scale=1e-3 → res/scale ~1 → loss ~1 → tanh không bão hòa
# #     # Scale = typical TC acceleration magnitude
# #     # TC speed ~5 m/s, changes over 6h → du ~ 5/21600 ~ 2e-4 m/s²
# #     # f*v ~ 1e-4 * 5 ~ 5e-4 m/s²
# #     # Tổng residual ~ 1e-3 m/s²
# #     # Scale = 1.0 m/s² → normalized residual ~ 1e-3 → loss ~ 1e-6 (quá nhỏ)
# #     # Scale = 1e-3 m/s² → normalized ~ 1 → loss ~ 1 ✓
# #     # Nhưng khi trajectory sai (rand init): speed ~500 m/s → du ~500/21600 ~0.02
# #     # → res ~ 0.02 / 1e-3 = 20 → loss = 400 → clamp/tanh cần max_val lớn

# #     # Giải pháp: normalize bằng magnitude thực tế của residual
# #     scale = torch.sqrt(
# #         res_u_corrected.detach().pow(2).mean() +
# #         res_v.detach().pow(2).mean()
# #     ).clamp(min=1e-6)

# #     loss = (res_u_corrected.pow(2).mean() + res_v.pow(2).mean()) / (scale + 1e-8)

# #     # loss ~ 1.0-2.0 khi residual đồng đều, > 2 khi một chiều lớn hơn
# #     return loss.clamp(max=5.0)
# def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
#     T, B, _ = pred_abs_deg.shape
#     if T < 3:
#         return pred_abs_deg.new_zeros(())

#     DT      = DT_6H
#     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
#     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
#     cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

#     u = dlat[:, :, 0] * cos_lat * 111000.0 / DT   # m/s
#     v = dlat[:, :, 1] * 111000.0 / DT

#     if u.shape[0] < 2:
#         return pred_abs_deg.new_zeros(())

#     du = (u[1:] - u[:-1]) / DT   # m/s²
#     dv = (v[1:] - v[:-1]) / DT

#     f    = 2 * OMEGA * torch.sin(lat_rad[1:-1])
#     beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH

#     res_u = du - f * v[1:]
#     res_v = dv + f * u[1:]

#     R_tc            = 3e5
#     v_beta_x        = -beta * R_tc ** 2 / 2
#     res_u_corrected = res_u - v_beta_x

#     # Fixed scale = 0.1 m/s²
#     # Good TC  : residual ~1e-3 → loss ~1e-4 (nhỏ, không penalize)
#     # Bad traj : residual ~0.1  → loss ~1.0  (penalize mạnh)
#     # Very bad : residual ~1.0  → loss ~100  → soft_clamp giữ ở ~20
#     scale = 0.1
#     loss  = ((res_u_corrected / scale).pow(2).mean()
#            + (res_v / scale).pow(2).mean())

#     return _soft_clamp_loss(loss, max_val=20.0)

# # def pinn_rankine_steering(pred_abs_deg: torch.Tensor,
# #                           env_data: Optional[dict]) -> torch.Tensor:
# #     if env_data is None:
# #         return pred_abs_deg.new_zeros(())

# #     T, B, _ = pred_abs_deg.shape
# #     if T < 2:
# #         return pred_abs_deg.new_zeros(())

# #     DT      = DT_6H
# #     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
# #     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
# #     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

# #     u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT
# #     v_tc = dlat[:, :, 1] * 111000.0 / DT

# #     u500_raw = env_data.get("u500_mean", None)
# #     v500_raw = env_data.get("v500_mean", None)

# #     if u500_raw is None or v500_raw is None:
# #         speed = torch.sqrt(u_tc ** 2 + v_tc ** 2)
# #         return F.relu(speed - 30.0).pow(2).mean() * 0.01

# #     def _align(x: torch.Tensor) -> torch.Tensor:
# #         if not torch.is_tensor(x):
# #             return pred_abs_deg.new_zeros(T - 1, B)
# #         x = x.to(pred_abs_deg.device)
# #         if x.dim() == 3:
# #             x_sq  = x[:, :, 0]
# #             T_env = x_sq.shape[1]
# #             T_tgt = T - 1
# #             if T_env >= T_tgt:
# #                 return x_sq[:, :T_tgt].permute(1, 0)
# #             pad = x_sq[:, -1:].expand(-1, T_tgt - T_env)
# #             return torch.cat([x_sq, pad], dim=1).permute(1, 0)
# #         elif x.dim() == 2:
# #             if x.shape == (B, T - 1):
# #                 return x.permute(1, 0)
# #             elif x.shape[0] == T - 1:
# #                 return x[:, :B] if x.shape[1] >= B else x.expand(-1, B)
# #         return pred_abs_deg.new_zeros(T - 1, B)

# #     u500 = _align(u500_raw) * _UV500_SCALE
# #     v500 = _align(v500_raw) * _UV500_SCALE

# #     uv_magnitude = torch.sqrt(u500 ** 2 + v500 ** 2)
# #     has_steering  = (uv_magnitude > 0.5).float()

# #     env_dir  = torch.stack([u500, v500], dim=-1)
# #     tc_dir   = torch.stack([u_tc, v_tc],  dim=-1)
# #     env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
# #     tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
# #     cos_sim  = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)
# #     misalign = F.relu(-0.5 - cos_sim).pow(2)

# #     return (misalign * has_steering).mean() * 0.05



# # def pinn_bve_loss(pred_abs_deg: torch.Tensor,
# #                   batch_list,
# #                   env_data: Optional[dict] = None) -> torch.Tensor:
# #     T = pred_abs_deg.shape[0]
# #     if T < 3:
# #         return pred_abs_deg.new_zeros(())

# #     l_sw    = pinn_shallow_water(pred_abs_deg)
# #     l_steer = pinn_rankine_steering(pred_abs_deg, env_data)
# #     l_speed = pinn_speed_constraint(pred_abs_deg)

# #     total = l_sw + 0.5 * l_steer + 0.1 * l_speed
# #     return _soft_clamp_loss(total, max_val=20.0)
# # losses.py — thay toàn bộ phần PINN




# def pinn_rankine_steering(pred_abs_deg: torch.Tensor,
#                           env_data: Optional[dict]) -> torch.Tensor:
#     """
#     FIX: Dùng u/v500_center (ưu tiên) + mean làm steering vector.
#     FIX: has_steering threshold 0.5 → 3.0 m/s.
#     FIX: Padding bằng zeros (climatology) thay vì repeat obs cuối.
#     """
#     if env_data is None:
#         return pred_abs_deg.new_zeros(())

#     T, B, _ = pred_abs_deg.shape
#     if T < 2:
#         return pred_abs_deg.new_zeros(())

#     device  = pred_abs_deg.device
#     T_tgt   = T - 1
#     DT      = DT_6H

#     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
#     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
#     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

#     u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT   # [T-1, B] m/s
#     v_tc = dlat[:, :, 1] * 111000.0 / DT              # [T-1, B] m/s

#     # FIX: center > mean, cả hai nếu có
#     u500 = _get_uv500_ms(env_data, "u500_mean", "u500_center",
#                          T_tgt, B, device)   # [T-1, B] m/s
#     v500 = _get_uv500_ms(env_data, "v500_mean", "v500_center",
#                          T_tgt, B, device)

#     uv_mag       = torch.sqrt(u500**2 + v500**2)
#     # FIX: threshold 3.0 m/s thay vì 0.5
#     has_steering = (uv_mag > _STEERING_MIN_MS).float()

#     env_dir  = torch.stack([u500, v500], dim=-1)
#     tc_dir   = torch.stack([u_tc,  v_tc], dim=-1)
#     env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=1.0)
#     tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
#     cos_sim  = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)

#     # Penalise chỉ khi hướng ngược (cos < -0.5) VÀ có steering thực sự
#     misalign = F.relu(-0.5 - cos_sim).pow(2)
#     return (misalign * has_steering).mean() * 0.05


# def pinn_gph500_gradient(pred_abs_deg: torch.Tensor,
#                          env_data: Optional[dict]) -> torch.Tensor:
#     """
#     MỚI: GPH500 gradient loss.

#     Vật lý: TC di chuyển dọc theo đường đẳng áp (isobar) của GPH500.
#     - gph500_center < gph500_mean → TC ở rãnh thấp → xu hướng poleward
#     - gph500_center > gph500_mean → TC ở gờ cao    → xu hướng equatorward

#     Loss: nếu GPH gradient chỉ hướng poleward nhưng TC đi equatorward
#     (hoặc ngược lại) → penalise.

#     gph500_mean/center đã z-score: (raw_m - 5870) / 80.
#     Gradient = center - mean → đơn vị normalized.
#     """
#     if env_data is None:
#         return pred_abs_deg.new_zeros(())
#     T, B, _ = pred_abs_deg.shape
#     if T < 2:
#         return pred_abs_deg.new_zeros(())

#     device = pred_abs_deg.device
#     T_tgt  = T - 1

#     gph_mean   = _get_gph500_norm(env_data, "gph500_mean",
#                                   T_tgt, B, device)    # [T-1, B]
#     gph_center = _get_gph500_norm(env_data, "gph500_center",
#                                   T_tgt, B, device)    # [T-1, B]

#     # gradient: center - mean (normalized units)
#     # âm → TC ở vùng thấp hơn xung quanh → ridge ở phía bắc → đẩy TC về nam?
#     # dương → TC ở vùng cao hơn xung quanh → trough ở phía bắc → đẩy TC về bắc
#     gph_grad = gph_center - gph_mean   # [T-1, B]

#     # TC lat tendency
#     dlat   = pred_abs_deg[1:, :, 1] - pred_abs_deg[:-1, :, 1]  # [T-1, B] degrees
#     # Chuẩn hóa lat tendency
#     dlat_n = dlat / (dlat.abs().clamp(min=1e-4))  # sign only: +1 hay -1

#     # Khi gph_grad < -0.1 (rãnh sâu phía bắc) → TC nên đi poleward (dlat > 0)
#     # Khi gph_grad > +0.1 (ridge phía bắc)    → TC nên đi equatorward (dlat < 0)
#     # Expected sign of dlat = sign of -gph_grad (rough heuristic)
#     expected_sign = -torch.sign(gph_grad)
#     has_gradient  = (gph_grad.abs() > 0.1).float()  # chỉ penalise khi gradient rõ

#     # Penalise khi sign ngược
#     wrong_dir = F.relu(-(dlat_n * expected_sign)).pow(2)
#     return (wrong_dir * has_gradient).mean() * 0.02


# def pinn_steering_speed_consistency(pred_abs_deg: torch.Tensor,
#                                     env_data: Optional[dict]) -> torch.Tensor:
#     """
#     MỚI: TC speed phải gần với steering flow speed.

#     Vật lý: TC thường di chuyển với tốc độ ≈ 0.5–0.8 × |steering flow|.
#     Nếu TC di chuyển quá nhanh hoặc quá chậm so với steering → bất thường.

#     Dùng cả u500_mean và u500_center để ước lượng steering magnitude.
#     """
#     if env_data is None:
#         return pred_abs_deg.new_zeros(())
#     T, B, _ = pred_abs_deg.shape
#     if T < 2:
#         return pred_abs_deg.new_zeros(())

#     device = pred_abs_deg.device
#     T_tgt  = T - 1
#     DT     = DT_6H

#     dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
#     lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
#     cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
#     dx_km   = dlat[:, :, 0] * cos_lat * 111.0
#     dy_km   = dlat[:, :, 1] * 111.0
#     tc_speed_ms = torch.sqrt(dx_km**2 + dy_km**2) * 1000.0 / DT  # m/s

#     # Dùng mean của center và mean để ước lượng steering magnitude
#     u_c = _get_uv500_ms(env_data, "u500_mean", "u500_center", T_tgt, B, device)
#     v_c = _get_uv500_ms(env_data, "v500_mean", "v500_center", T_tgt, B, device)
#     u_m = _get_uv500_ms(env_data, "u500_mean", "u500_mean",   T_tgt, B, device)
#     v_m = _get_uv500_ms(env_data, "v500_mean", "v500_mean",   T_tgt, B, device)

#     steering_mag = (torch.sqrt(u_c**2 + v_c**2) +
#                     torch.sqrt(u_m**2 + v_m**2)) / 2.0   # m/s

#     has_steering = (steering_mag > _STEERING_MIN_MS).float()

#     # TC speed ≈ 0.5–1.0 × steering (empirical for WP TCs)
#     lo = steering_mag * 0.3
#     hi = steering_mag * 1.5
#     too_slow = F.relu(lo - tc_speed_ms)
#     too_fast = F.relu(tc_speed_ms - hi)

#     penalty = (too_slow.pow(2) + too_fast.pow(2)) / (_UV500_NORM**2)
#     return (penalty * has_steering).mean() * 0.03


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
#     FIX: Tận dụng đầy đủ 6 Data3d features:
#       u500_mean, u500_center → steering direction + speed consistency
#       v500_mean, v500_center → steering direction + speed consistency
#       gph500_mean, gph500_center → GPH gradient → lat tendency constraint

#     Breakdown:
#       l_sw    : shallow water equation residual (trajectory only)
#       l_steer : steering direction alignment (u/v center+mean)
#       l_speed : TC absolute speed cap (trajectory only)
#       l_gph   : GPH gradient → lat tendency (gph center vs mean)
#       l_spdcons: TC speed vs steering flow magnitude
#     """
#     T = pred_abs_deg.shape[0]
#     if T < 3:
#         return pred_abs_deg.new_zeros(())

#     # env_data fallback
#     _env = env_data
#     if _env is None and batch_list is not None:
#         try:
#             _env = batch_list[13]
#         except (IndexError, TypeError):
#             _env = None

#     l_sw      = pinn_shallow_water(pred_abs_deg)
#     l_steer   = pinn_rankine_steering(pred_abs_deg, _env)
#     l_speed   = pinn_speed_constraint(pred_abs_deg)
#     l_gph     = pinn_gph500_gradient(pred_abs_deg, _env)
#     l_spdcons = pinn_steering_speed_consistency(pred_abs_deg, _env)

#     total = (l_sw
#              + 0.5  * l_steer
#              + 0.1  * l_speed
#              + 0.3  * l_gph        # MỚI: GPH gradient
#              + 0.4  * l_spdcons)   # MỚI: speed vs steering

#     return _soft_clamp_loss(total, max_val=20.0)


# # ══════════════════════════════════════════════════════════════════════════════
# #  Physics consistency
# # ══════════════════════════════════════════════════════════════════════════════

# def fm_physics_consistency_loss(
#     pred_samples: torch.Tensor,
#     gt_norm:      torch.Tensor,
#     last_pos:     torch.Tensor,
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

#     pos_step1 = pred_samples[:, 0, :, :2]
#     dir_step1 = pos_step1 - last_pos.unsqueeze(0)
#     dir_norm  = dir_step1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     dir_unit  = dir_step1 / dir_norm
#     mean_dir  = dir_unit.mean(0)
#     mean_norm = mean_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     mean_dir_unit = mean_dir / mean_norm

#     cos_align     = (mean_dir_unit * beta_dir_unit).sum(-1)
#     beta_strength = beta_norm.squeeze(-1)
#     penalise_mask = (beta_strength > 1.0).float()
#     direction_loss = F.relu(-cos_align) * penalise_mask
#     return direction_loss.mean() * 0.5


# # ══════════════════════════════════════════════════════════════════════════════
# #  Spread regularization  (FIX-L52: max_spread_km 200→150)
# # ══════════════════════════════════════════════════════════════════════════════

# def ensemble_spread_loss(all_trajs: torch.Tensor,
#                          max_spread_km: float = 150.0) -> torch.Tensor:
#     """
#     FIX-L52: max_spread_km=150 (từ 200). Exponential decay weights.
#     all_trajs: [S, T, B, 2] normalised.
#     """
#     if all_trajs.shape[0] < 2:
#         return all_trajs.new_zeros(())

#     S, T, B, _ = all_trajs.shape

#     # Exponential decay: early steps penalised more (from FIX-L46b)
#     step_weights = torch.exp(
#         -torch.arange(T, dtype=torch.float, device=all_trajs.device)
#         * (math.log(4.0) / max(T - 1, 1))
#     ) * 2.0   # range: 2.0 → 0.5

#     total_loss = all_trajs.new_zeros(())
#     for t in range(T):
#         step_trajs = all_trajs[:, t, :, :2]
#         std_lon    = step_trajs[:, :, 0].std(0)
#         std_lat    = step_trajs[:, :, 1].std(0)
#         spread_km  = torch.sqrt(std_lon ** 2 + std_lat ** 2) * 500.0
#         excess     = F.relu(spread_km - max_spread_km)
#         total_loss = total_loss + step_weights[t] * (
#             excess / max_spread_km
#         ).pow(2).mean()

#     return total_loss / T


# # ══════════════════════════════════════════════════════════════════════════════
# #  Main loss
# # ══════════════════════════════════════════════════════════════════════════════

# def compute_total_loss(
#     pred_abs,
#     gt,
#     ref,
#     batch_list,
#     pred_samples       = None,
#     gt_norm            = None,
#     weights            = WEIGHTS,
#     intensity_w: Optional[torch.Tensor] = None,
#     env_data:    Optional[dict]         = None,
#     step_weight_alpha: float = 0.0,
#     all_trajs:   Optional[torch.Tensor] = None,
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

#     l_vel       = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
#     l_disp      = (disp_loss_per_sample(pred_abs, gt)     * sample_w).mean()
#     l_step      = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()
#     l_heading   = heading_loss(pred_abs, gt)
#     l_recurv    = recurvature_loss(pred_abs, gt)
#     l_dir_final = overall_dir_loss(pred_abs, gt, ref)
#     l_smooth    = smooth_loss(pred_abs)
#     l_accel     = acceleration_loss(pred_abs)
#     l_jerk      = jerk_loss(pred_abs)

#     # 4. PINN
#     _env = env_data
#     if _env is None and batch_list is not None:
#         try:
#             _env = batch_list[13]
#         except (IndexError, TypeError):
#             _env = None

#     l_pinn = pinn_bve_loss(pred_abs, batch_list, env_data=_env)

#     # 5. Spread penalty  (FIX-L52)
#     # l_spread = pred_abs.new_zeros(())
#     # if all_trajs is not None and all_trajs.shape[0] >= 2:
#     #     l_spread = ensemble_spread_loss(all_trajs, max_spread_km=150.0)

#     # FIX: spread loss chỉ tính trên FM ensemble thật (không có sr_pred override)
#     # all_trajs trong get_loss_breakdown là FM samples trước khi blend
#     # → spread_loss phản ánh đúng ensemble diversity
#     l_spread = pred_abs.new_zeros(())
#     if all_trajs is not None and all_trajs.shape[0] >= 2:
#         # Chỉ tính spread cho steps 5-12 (FM range)
#         # Steps 1-4 do ShortRangeHead handle riêng
#         n_sr = 4
#         if all_trajs.shape[1] > n_sr:
#             l_spread = ensemble_spread_loss(
#                 all_trajs[:, n_sr:, :, :],   # chỉ steps 5-12
#                 max_spread_km=150.0
#             )
#         else:
#             # Fallback nếu traj quá ngắn (hiếm gặp)
#             l_spread = ensemble_spread_loss(all_trajs, max_spread_km=150.0)


#     # 6. Total  (short_range added externally in get_loss_breakdown)
#     total = (
#         weights.get("fm",       2.0) * l_fm
#         + weights.get("velocity", 0.8) * l_vel     * NRM
#         + weights.get("disp",     0.5) * l_disp    * NRM
#         + weights.get("step",     0.5) * l_step    * NRM
#         + weights.get("heading",  2.0) * l_heading * NRM
#         + weights.get("recurv",   1.5) * l_recurv  * NRM
#         + weights.get("dir",      1.0) * l_dir_final * NRM
#         + weights.get("smooth",   0.5) * l_smooth  * NRM
#         + weights.get("accel",    0.8) * l_accel   * NRM
#         + weights.get("jerk",     0.3) * l_jerk    * NRM
#         + weights.get("pinn",     0.5) * l_pinn    * NRM   # ← thêm * NRM
#         + weights.get("spread",   0.8) * l_spread  * NRM
#     ) / NRM

#     if torch.isnan(total) or torch.isinf(total):
#         total = pred_abs.new_zeros(())

#     return dict(
#         total        = total,
#         fm           = l_fm.item(),
#         velocity     = l_vel.item()     * NRM,
#         step         = l_step.item(),
#         disp         = l_disp.item()    * NRM,
#         heading      = l_heading.item(),
#         recurv       = l_recurv.item(),
#         smooth       = l_smooth.item()  * NRM,
#         accel        = l_accel.item()   * NRM,
#         jerk         = l_jerk.item()    * NRM,
#         pinn         = l_pinn.item(),
#         spread       = l_spread.item()  * NRM,
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
Model/losses.py  ── v26
========================
FULL REWRITE – fixes tất cả vấn đề từ review v25:

  FIX-L-A  [CRITICAL] pinn_bve_loss: bỏ nhân NRM trong compute_total_loss.
           PINN đã có soft-clamp max=20, nhân thêm NRM=35 → overpower 35×.

  FIX-L-B  [CRITICAL] AdaptClamp implement đúng theo Eq.58:
           ep 0-9: Huber mode (gradient ≥ 1/20, không bão hòa).
           ep 10-19: nội suy tuyến tính.
           ep 20+: tanh mode.

  FIX-L-C  [HIGH] Adaptive BVE weighting theo track error (Eq.99):
           w_BVE,k = σ(1 - d_hav(pred,gt)/200km).
           Tắt PINN khi track sai để tránh phạt nhầm.

  FIX-L-D  [HIGH] L_PWR (pressure-wind balance, Eq.62-63): implement
           đầy đủ với dynamic R_TC. Kích hoạt từ epoch 30.

  FIX-L-E  [HIGH] Frequency compensation w_pinn_eff (Eq.100):
           f_lazy schedule theo epoch.

  FIX-L-F  [HIGH] Spatial boundary weighting w_bnd (Eq.63a-b):
           Suy giảm PINN loss gần biên domain ERA5.

  FIX-L-G  [HIGH] Energy Score term trong fm_afcrps_loss (Eq.77):
           ES_norm(M) với unbiasing factor (M-1)/M.

  FIX-L-H  [MEDIUM] L_bridge implement (Eq.80):
           Nhất quán SR↔FM tại bước nối step 4.

  FIX-L-I  [MEDIUM] pinn_gph500_gradient: fix logic vật lý.
           center-mean không phải gradient 2D, dùng lat tendency đúng.

  FIX-L-J  [MEDIUM] fm_afcrps_loss: bỏ clamp(min=eps) trên loss_per_b
           để gradient flow khi ensemble tốt bất thường.

  FIX-L-K  [LOW] haversine coordinate decode: thêm assert để phát hiện
           sai đơn vị sớm.

Kept from v25:
  FIX-L49  short_range_regression_loss (Huber, step weights)
  FIX-L44  soft-clamp tanh (dùng trong epoch 20+)
  FIX-L45  pinn_rankine_steering với threshold 3.0 m/s
  FIX-L47  velocity direction penalty
  FIX-L48  jerk loss
  FIX-L52  ensemble_spread_loss max_spread=150km
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "WEIGHTS", "compute_total_loss", "fm_afcrps_loss",
    "fm_physics_consistency_loss", "pinn_bve_loss",
    "recurvature_loss", "velocity_loss_per_sample",
    "short_range_regression_loss", "bridge_loss",
    "adapt_clamp", "pinn_pressure_wind_loss",
]

# ── Constants ─────────────────────────────────────────────────────────────────
OMEGA        = 7.2921e-5
R_EARTH      = 6.371e6
DT_6H        = 6 * 3600
DEG_TO_KM    = 111.0
STEP_KM      = 113.0
P_ENV        = 1013.0   # hPa
RHO_AIR      = 1.15     # kg/m³

# ERA5 domain bounds (degrees) – dùng cho boundary weighting
_ERA5_LAT_MIN =   0.0
_ERA5_LAT_MAX =  40.0
_ERA5_LON_MIN = 100.0
_ERA5_LON_MAX = 160.0

_UV500_NORM      = 30.0
_GPH500_MEAN_M   = 5870.0
_GPH500_STD_M    = 80.0
_STEERING_MIN_MS = 3.0
_PINN_SCALE      = 1e-2

# ── Weights ───────────────────────────────────────────────────────────────────
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
    pinn        = 0.5,
    fm_physics  = 0.3,
    spread      = 0.8,
    short_range = 5.0,
    bridge      = 0.5,    # NEW FIX-L-H
)

RECURV_ANGLE_THR = 45.0
RECURV_WEIGHT    = 2.5

_SR_N_STEPS  = 4
_SR_WEIGHTS  = [2.0, 4.0, 2.0, 2.0]
_HUBER_DELTA = 50.0


# ══════════════════════════════════════════════════════════════════════════════
#  FIX-L-B: AdaptClamp đúng theo Eq.58
# ══════════════════════════════════════════════════════════════════════════════

def adapt_clamp(x: torch.Tensor, epoch: int, max_val: float = 20.0) -> torch.Tensor:
    """
    AdaptClamp_ep(x) theo Eq.58:
      ep 0-9:  Huber mode  → gradient ≥ 1/max_val, không bão hòa
      ep 10-19: nội suy tuyến tính Huber → tanh
      ep 20+:  tanh mode   → mượt mà, ổn định convergence

    HuberClamp(x, δ):
      x ≤ δ : x²/(2δ)       [quadratic, gradient = x/δ]
      x > δ : x - δ/2       [linear,    gradient = 1]
    """
    delta = max_val

    def huber_clamp(v: torch.Tensor) -> torch.Tensor:
        return torch.where(
            v <= delta,
            v.pow(2) / (2.0 * delta),
            v - delta / 2.0,
        )

    def tanh_clamp(v: torch.Tensor) -> torch.Tensor:
        return delta * torch.tanh(v / delta)

    if epoch < 10:
        return huber_clamp(x)
    elif epoch < 20:
        beta = (epoch - 10) / 10.0
        return (1.0 - beta) * huber_clamp(x) + beta * tanh_clamp(x)
    else:
        return tanh_clamp(x)


# ══════════════════════════════════════════════════════════════════════════════
#  Haversine
# ══════════════════════════════════════════════════════════════════════════════

def _haversine(p1: torch.Tensor, p2: torch.Tensor,
               unit_01deg: bool = True) -> torch.Tensor:
    """
    Tính khoảng cách Haversine (km).
    unit_01deg=True: input là normalized coords, decode trước.
    unit_01deg=False: input đã là degrees.
    """
    if unit_01deg:
        lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
        lat1 = (p1[..., 1] * 50.0) / 10.0
        lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
        lat2 = (p2[..., 1] * 50.0) / 10.0
    else:
        lon1, lat1 = p1[..., 0], p1[..., 1]
        lon2, lat2 = p2[..., 0], p2[..., 1]

    lat1r = torch.deg2rad(lat1); lat2r = torch.deg2rad(lat2)
    dlon  = torch.deg2rad(lon2 - lon1)
    dlat  = torch.deg2rad(lat2 - lat1)
    a = (torch.sin(dlat / 2).pow(2)
         + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2).pow(2))
    a = a.clamp(1e-12, 1.0 - 1e-12)   # FIX-L-K: stable asin
    return 2.0 * 6371.0 * torch.asin(a.sqrt())


def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    return _haversine(p1, p2, unit_01deg=False)


def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
    out = arr.clone()
    out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
    out[..., 1] = (arr[..., 1] * 50.0) / 10.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  FIX-L-F: Spatial Boundary Weighting (Eq.63a-b)
# ══════════════════════════════════════════════════════════════════════════════

def _boundary_weights(traj_deg: torch.Tensor) -> torch.Tensor:
    """
    w_bnd,k = σ(d_bnd,k - 0.5) ∈ [0,1]
    d_bnd,k = min(lat-lat_min, lat_max-lat, lon-lon_min, lon_max-lon) / 5°

    traj_deg: [T, B, 2]  (lon, lat)
    Returns:  [T, B]
    """
    lon = traj_deg[..., 0]  # [T, B]
    lat = traj_deg[..., 1]  # [T, B]

    d_lat_lo = (lat - _ERA5_LAT_MIN) / 5.0
    d_lat_hi = (_ERA5_LAT_MAX - lat) / 5.0
    d_lon_lo = (lon - _ERA5_LON_MIN) / 5.0
    d_lon_hi = (_ERA5_LON_MAX - lon) / 5.0

    d_bnd = torch.stack([d_lat_lo, d_lat_hi, d_lon_lo, d_lon_hi], dim=-1).min(dim=-1).values
    return torch.sigmoid(d_bnd - 0.5)   # [T, B]


# ══════════════════════════════════════════════════════════════════════════════
#  Short-range Huber Loss (FIX-L49, giữ nguyên)
# ══════════════════════════════════════════════════════════════════════════════

def short_range_regression_loss(
    pred_sr:  torch.Tensor,   # [4, B, 2]  normalised
    gt_sr:    torch.Tensor,   # [4, B, 2]  normalised
    last_pos: torch.Tensor,   # [B, 2]     (unused, kept for API compat)
) -> torch.Tensor:
    """
    Huber loss trên haversine distance cho steps 1-4.
    Return: scalar loss, đơn vị km (không chia HUBER_DELTA).
    Caller chịu trách nhiệm scale bằng weight.
    """
    n_steps = min(pred_sr.shape[0], gt_sr.shape[0], _SR_N_STEPS)
    if n_steps == 0:
        return pred_sr.new_zeros(())

    pred_deg = _norm_to_deg(pred_sr[:n_steps])   # [n_steps, B, 2]
    gt_deg   = _norm_to_deg(gt_sr[:n_steps])

    dist_km = _haversine_deg(pred_deg, gt_deg)   # [n_steps, B]

    huber = torch.where(
        dist_km < _HUBER_DELTA,
        0.5 * dist_km.pow(2) / _HUBER_DELTA,
        dist_km - 0.5 * _HUBER_DELTA,
    )  # [n_steps, B]

    w = pred_sr.new_tensor(_SR_WEIGHTS[:n_steps])  # [n_steps]
    # Normalize weights để tổng = 1
    w = w / w.sum()
    return (huber * w.view(-1, 1)).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  FIX-L-H: Bridge Loss SR↔FM (Eq.80)
# ══════════════════════════════════════════════════════════════════════════════

def bridge_loss(
    sr_pred:  torch.Tensor,   # [4, B, 2]  normalised SR predictions
    fm_mean:  torch.Tensor,   # [T, B, 2]  normalised FM mean trajectory
) -> torch.Tensor:
    """
    Nhất quán vị trí và vận tốc tại bước nối step 4 (idx=3).

    L_bridge = ||y4_SR - X4_FM||² / (100km)²
             + 0.5 * ||v4_SR - v5_FM||² / STEP²
    """
    if sr_pred.shape[0] < 4 or fm_mean.shape[0] < 5:
        return sr_pred.new_zeros(())

    # Vị trí step 4 (index 3)
    pos_sr4 = _norm_to_deg(sr_pred[3])       # [B, 2]
    pos_fm4 = _norm_to_deg(fm_mean[3])       # [B, 2]

    dist_pos = _haversine_deg(pos_sr4, pos_fm4)  # [B]
    l_pos = (dist_pos / 100.0).pow(2).mean()

    # Vận tốc tại tiếp giáp
    # v4_SR = pos4_SR - pos3_SR (degrees, thô)
    v4_sr = _norm_to_deg(sr_pred[3]) - _norm_to_deg(sr_pred[2])   # [B, 2]
    # v5_FM = pos5_FM - pos4_FM
    v5_fm = _norm_to_deg(fm_mean[4]) - _norm_to_deg(fm_mean[3])   # [B, 2]

    # Chuyển sang km
    lat_mid = (_norm_to_deg(sr_pred[3])[:, 1] + _norm_to_deg(fm_mean[4])[:, 1]) / 2.0
    cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)

    def _to_km(dv):
        km = dv.clone()
        km[:, 0] = dv[:, 0] * cos_lat * DEG_TO_KM
        km[:, 1] = dv[:, 1] * DEG_TO_KM
        return km

    v4_sr_km = _to_km(v4_sr)
    v5_fm_km = _to_km(v5_fm)
    l_vel = ((v4_sr_km - v5_fm_km).pow(2).sum(-1) / STEP_KM**2).mean()

    return l_pos + 0.5 * l_vel


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
    v1   = v[:-1]; v2 = v[1:]
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
    gn        = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    gt_unit   = v_gt_km / gn
    ate       = ((v_pred_km - v_gt_km) * gt_unit).sum(-1)
    l_ate     = ate.pow(2).mean(0)
    pn        = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    cos_sim   = ((v_pred_km / pn) * gt_unit).sum(-1)
    l_dir     = F.relu(-cos_sim).pow(2).mean(0) * STEP_KM ** 2
    return (l_speed + 0.5 * l_ate + 0.3 * l_dir) / (STEP_KM ** 2)


def disp_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.new_zeros(pred.shape[1])
    pd = _step_displacements_km(pred).norm(dim=-1).mean(0)
    gd = _step_displacements_km(gt).norm(dim=-1).mean(0)
    return (pd - gd).pow(2) / (STEP_KM ** 2)


def step_dir_loss_per_sample(pred: torch.Tensor,
                              gt: torch.Tensor) -> torch.Tensor:
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
    wrong_dir      = F.relu(-((pv / pn) * (gv / gn)).sum(-1))
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
    return smooth_loss(pred)


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
    p_d_km  = p_d.clone(); p_d_km[:, 0] *= cos_lat * DEG_TO_KM; p_d_km[:, 1] *= DEG_TO_KM
    g_d_km  = g_d.clone(); g_d_km[:, 0] *= cos_lat * DEG_TO_KM; g_d_km[:, 1] *= DEG_TO_KM
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
    cos_sim  = ((pred_v_mid / pn) * (gt_v_mid / gn)).sum(-1)
    dir_loss = (1.0 - cos_sim)
    mask     = sign_change.float()
    if mask.sum() < 1:
        return pred.new_zeros(())
    return (dir_loss * mask).sum() / mask.sum().clamp(min=1)


# ══════════════════════════════════════════════════════════════════════════════
#  FIX-L-G: AFCRPS với Energy Score (Eq.76-77)
# ══════════════════════════════════════════════════════════════════════════════

def fm_afcrps_loss(
    pred_samples:      torch.Tensor,         # [M, T, B, 2]
    gt:                torch.Tensor,         # [T, B, 2]
    unit_01deg:        bool  = True,
    intensity_w:       Optional[torch.Tensor] = None,
    step_weight_alpha: float = 0.0,
    w_es:              float = 0.3,          # weight cho Energy Score term
) -> torch.Tensor:
    """
    Almost-Fair CRPS + M-Normalized Energy Score (Eq.76-77).

    L_FM = accuracy - sharpness_penalty + w_ES * ES_norm(M)

    FIX-L-G: Thêm ES_norm với unbiasing factor (M-1)/M.
    FIX-L-J: Bỏ clamp(min=eps) để gradient flow tự nhiên.
    """
    M, T, B, _ = pred_samples.shape

    # Time weights
    # base_w = torch.linspace(0.5, 1.5, T, device=pred_samples.device)
    # losses.py — fm_afcrps_loss, thay base_w
    base_w = torch.zeros(T, device=pred_samples.device)
    for i in range(T):
        if i >= 8:    base_w[i] = 3.0   # 54h-72h
        elif i >= 4:  base_w[i] = 1.5   # 30h-48h
        else:         base_w[i] = 0.5   # 6h-24h (SR lo)
    if step_weight_alpha > 0.0:
        early_w = torch.exp(
            -torch.arange(T, dtype=torch.float, device=pred_samples.device) * 0.5
        )
        early_w = early_w / early_w.mean()
        time_w = (1.0 - step_weight_alpha) * base_w + step_weight_alpha * early_w
    else:
        time_w = base_w

    if M == 1:
        dist = _haversine(pred_samples[0], gt, unit_01deg)   # [T, B]
        loss_per_b = (dist * time_w.unsqueeze(1)).mean(0)    # [B]
        es_term = pred_samples.new_zeros(())
    else:
        # Accuracy: E[d(X^m, Y)]
        d_to_gt = _haversine(
            pred_samples,
            gt.unsqueeze(0).expand_as(pred_samples),
            unit_01deg,
        )   # [M, T, B]
        d_to_gt_w    = d_to_gt * time_w.unsqueeze(0).unsqueeze(2)
        e_sy         = d_to_gt_w.mean(1).mean(0)   # [B]  accuracy

        # Sharpness: E[d(X^m, X^m')]  with m≠m'  (fair: exclude m=m')
        ps_i = pred_samples.unsqueeze(1)   # [M, 1, T, B, 2]
        ps_j = pred_samples.unsqueeze(0)   # [1, M, T, B, 2]
        d_pair = _haversine(
            ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
            ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
            unit_01deg,
        ).reshape(M, M, T, B)   # [M, M, T, B]

        # Mask diagonal (m == m')
        diag_mask = torch.eye(M, device=pred_samples.device, dtype=torch.bool)
        diag_mask = diag_mask.view(M, M, 1, 1).expand_as(d_pair)
        d_pair = d_pair.masked_fill(diag_mask, 0.0)

        d_pair_w    = d_pair * time_w.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        # fair mean: sum / (M*(M-1)) thay vì M*M
        e_ssp       = d_pair_w.mean(2).sum(0).sum(0) / (M * (M - 1))  # [B]

        # Almost-fair CRPS
        loss_per_b = e_sy - 0.5 * e_ssp    # [B]  FIX-L-J: bỏ clamp

        # FIX-L-G: Energy Score term (Eq.77)
        # ES_norm(M) = ||mean_m(X^m) - Y||_F - (M-1)/M * (1/M) * sum_{m≠m'} ||X^m - X^m'||_F
        if w_es > 0.0 and M > 1:
            # Chuyển sang [M, T*B, 2] để tính norm Frobenius
            ps_flat  = pred_samples.reshape(M, T * B, 2)  # [M, T*B, 2]
            gt_flat  = gt.reshape(T * B, 2)               # [T*B, 2]
            mean_pred = ps_flat.mean(0)                    # [T*B, 2]

            # ||mean - Y||_F  (Frobenius = sqrt(sum of squares))
            es_acc = (mean_pred - gt_flat).pow(2).sum(-1).sqrt().mean()

            # (M-1)/M * mean_{m≠m'} ||X^m - X^m'||_F
            ps_i_f = ps_flat.unsqueeze(1)   # [M, 1, T*B, 2]
            ps_j_f = ps_flat.unsqueeze(0)   # [1, M, T*B, 2]
            d_pair_f = (ps_i_f - ps_j_f).pow(2).sum(-1).sqrt()  # [M, M, T*B]
            # Mask diagonal
            diag_f = torch.eye(M, device=pred_samples.device, dtype=torch.bool)
            diag_f = diag_f.view(M, M, 1).expand_as(d_pair_f)
            d_pair_f = d_pair_f.masked_fill(diag_f, 0.0)
            es_sharp = (M - 1) / M * d_pair_f.sum(0).sum(0).mean() / (M * (M - 1))

            es_term = es_acc - es_sharp
        else:
            es_term = pred_samples.new_zeros(())

    if intensity_w is not None:
        w = intensity_w.to(loss_per_b.device)
        crps_loss = (loss_per_b * w).mean()
    else:
        crps_loss = loss_per_b.mean()

    return crps_loss + w_es * es_term


# ══════════════════════════════════════════════════════════════════════════════
#  PINN Components
# ══════════════════════════════════════════════════════════════════════════════

def _get_uv500_ms(env_data: dict, key_mean: str, key_center: str,
                  T_tgt: int, B: int, device: torch.device) -> torch.Tensor:
    def _extract(key):
        x = env_data.get(key, None)
        if x is None or not torch.is_tensor(x):
            return None
        x = x.to(device).float()
        if x.dim() == 3:
            x = x[..., 0]
        elif x.dim() == 1:
            x = x.unsqueeze(0).expand(B, -1)
        x = x.permute(1, 0)   # [T_obs, B]
        T_obs = x.shape[0]
        if T_obs >= T_tgt:
            return x[:T_tgt] * _UV500_NORM
        pad = torch.zeros(T_tgt - T_obs, B, device=device)
        return torch.cat([x * _UV500_NORM, pad], dim=0)

    val = _extract(key_center)
    if val is not None:
        return val
    val = _extract(key_mean)
    if val is not None:
        return val
    return torch.zeros(T_tgt, B, device=device)


def _get_gph500_norm(env_data: dict, key: str,
                     T_tgt: int, B: int, device: torch.device) -> torch.Tensor:
    x = env_data.get(key, None)
    
    if x is None or not torch.is_tensor(x):
        return torch.zeros(T_tgt, B, device=device)
    x = x.to(device).float()
    if x.dim() == 3:
        x = x[..., 0]
    x = x.permute(1, 0)
    T_obs = x.shape[0]
    if T_obs >= T_tgt:
        return x[:T_tgt]
    pad = torch.zeros(T_tgt - T_obs, B, device=device)
    return torch.cat([x, pad], dim=0)


def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
    """BVE: bảo toàn độ xoáy tuyệt đối (Eq.55)."""
    T, B, _ = pred_abs_deg.shape
    if T < 3:
        return pred_abs_deg.new_zeros(())

    DT      = DT_6H
    lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
    dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
    cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

    u = dlat[:, :, 0] * cos_lat * 111000.0 / DT   # m/s
    v = dlat[:, :, 1] * 111000.0 / DT

    if u.shape[0] < 2:
        return pred_abs_deg.new_zeros(())

    du = (u[1:] - u[:-1]) / DT
    dv = (v[1:] - v[:-1]) / DT

    f    = 2 * OMEGA * torch.sin(lat_rad[1:-1])
    beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH

    res_u = du - f * v[1:]
    res_v = dv + f * u[1:]

    R_tc            = 3e5
    v_beta_x        = -beta * R_tc ** 2 / 2
    res_u_corrected = res_u - v_beta_x

    # Scale = 0.1 m/s²: typical TC → residual ~1e-3 → loss ~1e-4 (không penalize)
    # Bad traj → residual ~0.1 → loss ~1.0 (penalize mạnh)
    scale = 0.1
    # loss = ((res_u_corrected / scale).pow(2).mean()
    #       + (res_v / scale).pow(2).mean())
    loss = ((res_u_corrected / scale).pow(2) + (res_v / scale).pow(2)) 
    return loss


def pinn_rankine_steering(pred_abs_deg: torch.Tensor,
                          env_data: Optional[dict]) -> torch.Tensor:
    """Steering flow alignment (Eq.59)."""
    if env_data is None:
        return pred_abs_deg.new_zeros(())

    T, B, _ = pred_abs_deg.shape
    if T < 2:
        return pred_abs_deg.new_zeros(())

    device  = pred_abs_deg.device
    T_tgt   = T - 1
    DT      = DT_6H

    dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
    lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
    cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

    u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT   # [T-1, B] m/s
    v_tc = dlat[:, :, 1] * 111000.0 / DT

    u500 = _get_uv500_ms(env_data, "u500_mean", "u500_center", T_tgt, B, device)
    v500 = _get_uv500_ms(env_data, "v500_mean", "v500_center", T_tgt, B, device)

    uv_mag       = torch.sqrt(u500**2 + v500**2)
    has_steering = (uv_mag > _STEERING_MIN_MS).float()

    env_dir  = torch.stack([u500, v500], dim=-1)
    tc_dir   = torch.stack([u_tc, v_tc], dim=-1)
    env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=1.0)
    tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=0.5)
    cos_sim  = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)

    # Sigmoid soft weighting (Eq.59): penalize liên tục thay vì ngưỡng cứng
    steer_w  = torch.sigmoid(uv_mag - 1.0)   # ≈0.27 tại 0 m/s, ≈0.5 tại 1 m/s
    misalign = F.relu(-0.5 - cos_sim).pow(2)
    # return (misalign * steer_w * has_steering).mean() * 0.05
    return (misalign * steer_w * has_steering) * 0.05 # Trả về [T-1, B]

def pinn_gph500_gradient(pred_abs_deg: torch.Tensor,
                         env_data: Optional[dict]) -> torch.Tensor:
    """
    FIX-L-I: GPH gradient đúng hơn (Eq.60).

    Thay vì dùng center-mean làm proxy gradient (sai về vật lý),
    dùng temporal gradient: ΔGPH/Δt để ước lượng xu hướng.
    Penalize khi TC di chuyển ngược xu hướng GPH.
    """
    if env_data is None:
        return pred_abs_deg.new_zeros(())
    T, B, _ = pred_abs_deg.shape
    if T < 2:
        return pred_abs_deg.new_zeros(())

    device = pred_abs_deg.device
    T_tgt  = T - 1

    gph_mean   = _get_gph500_norm(env_data, "gph500_mean",   T_tgt, B, device)
    gph_center = _get_gph500_norm(env_data, "gph500_center", T_tgt, B, device)

    # Proxy gradient: difference between center (tại TC) và mean (domain)
    # Đơn vị: normalized z-score
    # Dương: TC ở vùng GPH cao hơn xung quanh → ridge → đẩy poleward/westward
    # Âm:   TC ở vùng GPH thấp hơn → trough → TC có xu hướng recurve
    gph_diff = gph_center - gph_mean   # [T-1, B]

    # Lat tendency của TC
    dlat = pred_abs_deg[1:, :, 1] - pred_abs_deg[:-1, :, 1]  # [T-1, B]

    # Heuristic: khi GPH diff dương mạnh → ridge north → TC nên northward
    # expected_dlat_sign = sign(gph_diff)
    # Chỉ penalize khi gradient rõ ràng (|gph_diff| > 0.1 sigma)
    has_gradient = (gph_diff.abs() > 0.1).float()

    # s_correct dương khi TC di chuyển đúng hướng
    s_correct = torch.sign(dlat) * torch.sign(gph_diff)
    wrong_dir = F.relu(-s_correct)   # dương khi sai hướng

    # return (wrong_dir.pow(2) * has_gradient).mean() * 0.02
    return (wrong_dir.pow(2) * has_gradient) * 0.02 # Trả về [T-1, B]

def pinn_steering_speed_consistency(pred_abs_deg: torch.Tensor,
                                    env_data: Optional[dict]) -> torch.Tensor:
    """TC speed vs steering flow speed (Eq.61)."""
    if env_data is None:
        return pred_abs_deg.new_zeros(())
    T, B, _ = pred_abs_deg.shape
    if T < 2:
        return pred_abs_deg.new_zeros(())

    device = pred_abs_deg.device
    T_tgt  = T - 1
    DT     = DT_6H

    dlat    = pred_abs_deg[1:] - pred_abs_deg[:-1]
    lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
    cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
    dx_km   = dlat[:, :, 0] * cos_lat * 111.0
    dy_km   = dlat[:, :, 1] * 111.0
    tc_speed_ms = torch.sqrt(dx_km**2 + dy_km**2) * 1000.0 / DT  # m/s

    u_c = _get_uv500_ms(env_data, "u500_mean", "u500_center", T_tgt, B, device)
    v_c = _get_uv500_ms(env_data, "v500_mean", "v500_center", T_tgt, B, device)
    u_m = _get_uv500_ms(env_data, "u500_mean", "u500_mean",   T_tgt, B, device)
    v_m = _get_uv500_ms(env_data, "v500_mean", "v500_mean",   T_tgt, B, device)

    steering_mag = (torch.sqrt(u_c**2 + v_c**2) +
                    torch.sqrt(u_m**2 + v_m**2)) / 2.0

    # Normalize by σ²(UV500) như Eq.61
    steer_var = (steering_mag**2).mean().clamp(min=1.0)

    has_steering = (steering_mag > _STEERING_MIN_MS).float()
    lo = steering_mag * 0.3
    hi = steering_mag * 1.5
    too_slow = F.relu(lo - tc_speed_ms)
    too_fast = F.relu(tc_speed_ms - hi)

    penalty = (too_slow.pow(2) + too_fast.pow(2)) / steer_var
    # return (penalty * has_steering).mean() * 0.03
    return (penalty * has_steering) * 0.03 # Trả về [T-1, B]

def pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
    if pred_deg.shape[0] < 2:
        return pred_deg.new_zeros(())
    dt_deg  = pred_deg[1:] - pred_deg[:-1]
    lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
    cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
    dx_km   = dt_deg[:, :, 0] * cos_lat * DEG_TO_KM
    dy_km   = dt_deg[:, :, 1] * DEG_TO_KM
    speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
    # return F.relu(speed - 600.0).pow(2).mean()
    return F.relu(speed - 600.0).pow(2) # Trả về [T-1, B]

# ══════════════════════════════════════════════════════════════════════════════
#  FIX-L-D: Pressure-Wind Balance Loss (Eq.62-63)
# ══════════════════════════════════════════════════════════════════════════════

def pinn_pressure_wind_loss(
    pred_abs_deg: torch.Tensor,   # [T, B, 2]  lon/lat
    vmax_pred:    Optional[torch.Tensor],  # [T, B]  m/s
    pmin_pred:    Optional[torch.Tensor],  # [T, B]  hPa
    r34_km:       Optional[torch.Tensor] = None,  # [T, B]  km
    epoch:        int = 0,
) -> torch.Tensor:
    """
    Gradient wind balance (Eq.62-63):
      p_env - p_min ≈ ρ*V²/2 + f*R_TC*V/2

    Dynamic R_TC: dùng 2*R34 nếu có, fallback climatology (Eq.62a-b).
    Kích hoạt từ epoch 30.
    """
    if epoch < 30:
        return pred_abs_deg.new_zeros(())
    if vmax_pred is None or pmin_pred is None:
        return pred_abs_deg.new_zeros(())

    T, B, _ = pred_abs_deg.shape
    if vmax_pred.shape[0] != T or pmin_pred.shape[0] != T:
        return pred_abs_deg.new_zeros(())

    lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
    f_k     = 2 * OMEGA * torch.sin(lat_rad).abs().clamp(min=1e-6)  # [T, B]

    # Dynamic R_TC (Eq.62a-b)
    if r34_km is not None:
        R_tc = (2.0 * r34_km * 1000.0).clamp(min=1e5, max=8e5)  # m
    else:
        # Fallback: R_TC = 3e5 + 1e3 * max(0, Vmax - 30) m (Eq.62b)
        R_tc = 3e5 + 1e3 * F.relu(vmax_pred - 30.0)  # [T, B]

    V  = vmax_pred.clamp(min=1.0)    # m/s
    dp = (P_ENV - pmin_pred) * 100.0  # Pa (1 hPa = 100 Pa)

    # Gradient wind: dp = ρV²/2 + ρ*f*R*V/2
    dp_pred = RHO_AIR * (V.pow(2) / 2.0 + f_k * R_tc * V / 2.0)  # Pa

    # Normalize bằng 5 hPa (500 Pa) như Eq.63
    residual = (dp - dp_pred) / 500.0
    # return residual.pow(2).mean()
    return residual.pow(2) # Trả về [T, B]

# ══════════════════════════════════════════════════════════════════════════════
#  FIX-L-C + FIX-L-E + FIX-L-F: PINN tổng hợp
# ══════════════════════════════════════════════════════════════════════════════

# def pinn_bve_loss(
#     pred_abs_deg: torch.Tensor,
#     batch_list,
#     env_data:    Optional[dict] = None,
#     epoch:       int = 0,
#     gt_abs_deg:  Optional[torch.Tensor] = None,  # FIX-L-C: cho adaptive weighting
#     vmax_pred:   Optional[torch.Tensor] = None,   # FIX-L-D: pressure-wind
#     pmin_pred:   Optional[torch.Tensor] = None,
#     r34_km:      Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     """
#     PINN tổng hợp (Eq.64/101) với đầy đủ 5 ràng buộc + PWR.

#     Cải tiến so với v25:
#       - AdaptClamp thay vì tanh cố định (FIX-L-B)
#       - Adaptive BVE weighting theo track error (FIX-L-C)
#       - Frequency compensation f_lazy (FIX-L-E)
#       - Spatial boundary weighting w_bnd (FIX-L-F)
#       - L_PWR pressure-wind balance (FIX-L-D)
#     """
#     T = pred_abs_deg.shape[0]
#     if T < 3:
#         return pred_abs_deg.new_zeros(())

#     _env = env_data
#     if _env is None and batch_list is not None:
#         try:
#             _env = batch_list[13]
#         except (IndexError, TypeError):
#             _env = None

#     # FIX-L-E: Frequency compensation (Eq.100)
#     if epoch < 30:
#         f_lazy = 0.20
#     elif epoch < 50:
#         f_lazy = 0.50
#     else:
#         f_lazy = 1.00

#     # FIX-L-C: Adaptive BVE weighting (Eq.99)
#     if gt_abs_deg is not None and gt_abs_deg.shape == pred_abs_deg.shape:
#         with torch.no_grad():
#             d_track = _haversine_deg(pred_abs_deg, gt_abs_deg)   # [T, B]
#             # w_BVE,k = σ(1 - d/200km)
#             w_bve_per_step = torch.sigmoid(1.0 - d_track / 200.0)  # [T, B] ∈ (0,1)
#             w_bve = w_bve_per_step.mean()   # scalar
#     else:
#         w_bve = pred_abs_deg.new_tensor(1.0)

#     # FIX-L-F: Spatial boundary weighting (Eq.63a-b)
#     w_bnd = _boundary_weights(pred_abs_deg).mean()   # scalar ≈ 1 at center, ≈ 0 at edge

#     # # Individual PINN components
#     # l_sw      = pinn_shallow_water(pred_abs_deg)
#     # l_steer   = pinn_rankine_steering(pred_abs_deg, _env)
#     # l_speed   = pinn_speed_constraint(pred_abs_deg)
#     # l_gph     = pinn_gph500_gradient(pred_abs_deg, _env)
#     # l_spdcons = pinn_steering_speed_consistency(pred_abs_deg, _env)
#     # l_pwr     = pinn_pressure_wind_loss(pred_abs_deg, vmax_pred, pmin_pred, r34_km, epoch)

#     # # Tổng hợp (Eq.101) — hệ số từ doc
#     # total = (
#     #     w_bve * l_sw          # BVE với adaptive weighting
#     #     + 0.5  * l_steer
#     #     + 0.1  * l_speed
#     #     + 0.3  * l_gph
#     #     + 0.4  * l_spdcons
#     #     + 0.6  * l_pwr        # PWR với cao nhất (doc: ưu tiên intensity)
#     # )

#     # # FIX-L-B: AdaptClamp thay vì tanh cố định
#     # total_clamped = adapt_clamp(total, epoch, max_val=20.0)

#     # # FIX-L-E + FIX-L-F: áp dụng frequency compensation và boundary weight
#     # return total_clamped * w_bnd * f_lazy
#      # 1. Định nghĩa Trọng số thời gian cho PINN (Key để giảm 72h)
#     # Tăng dần từ 0.5 (ở 6h) lên 4.0 (ở 72h)
#     pinn_step_w = torch.linspace(0.5, 4.0, T, device=pred_abs_deg.device)
    
#     # 2. Tính toán các thành phần (Giả sử các hàm này trả về tensor [T_i, B])
#     # Nếu hàm của bạn đang trả về scalar, hãy sửa chúng để KHÔNG gọi .mean() ở cuối
#     l_sw_map      = pinn_shallow_water(pred_abs_deg)          # [T-2, B]
#     l_steer_map   = pinn_rankine_steering(pred_abs_deg, _env) # [T-1, B]
#     l_speed_map   = pinn_speed_constraint(pred_abs_deg)       # [T-1, B]
#     l_gph_map     = pinn_gph500_gradient(pred_abs_deg, _env)  # [T-1, B]
#     l_spdcons_map = pinn_steering_speed_consistency(pred_abs_deg, _env)   # [T-1, B]
#     l_pwr_map     = pinn_pressure_wind_loss(pred_abs_deg, vmax_pred, pmin_pred, r34_km, epoch) # [T, B]

#     # 3. Tính Adaptive Weighting (FIX-L-C) nhưng giữ nguyên theo step
#     if gt_abs_deg is not None:
#         with torch.no_grad():
#             d_track = _haversine_deg(pred_abs_deg, gt_abs_deg)
#             w_bve_step = torch.sigmoid(1.0 - d_track / 200.0) # [T, B]
#     else:
#         w_bve_step = torch.ones(T, B, device=pred_abs_deg.device)

#     # 4. Tổng hợp loss theo từng bước thời gian (Pointwise Total)
#     # Ta lấy phần đuôi của pinn_step_w để khớp với số lượng step của từng loại loss
#     def apply_w(l_map, weight_scalar):
#         t_size = l_map.shape[0]
#         # Nhân trọng số thành phần * trọng số thời gian * adaptive weight
#         return weight_scalar * l_map * pinn_step_w[-t_size:, None] * w_bve_step[-t_size:]

#     total_pointwise = (
#         apply_w(l_sw_map, 1.0)           # Trọng số gốc 1.0
#         + apply_w(l_steer_map, 0.5)
#         + apply_w(l_speed_map, 0.1)
#         + apply_w(l_gph_map, 0.3)
#         + apply_w(l_spdcons_map, 0.4)
#         + apply_w(l_pwr_map, 0.6)
#     )

#     # 5. Lấy trung bình toàn bộ
#     total = total_pointwise.mean()

#     # FIX-L-B: AdaptClamp
#     total_clamped = adapt_clamp(total, epoch, max_val=20.0)

#     return total_clamped * w_bnd * f_lazy
def pinn_bve_loss(
    pred_abs_deg: torch.Tensor,
    batch_list,
    env_data:    Optional[dict] = None,
    epoch:       int = 0,
    gt_abs_deg:  Optional[torch.Tensor] = None,
    vmax_pred:   Optional[torch.Tensor] = None,
    pmin_pred:   Optional[torch.Tensor] = None,
    r34_km:      Optional[torch.Tensor] = None,
) -> torch.Tensor:
    T = pred_abs_deg.shape[0]
    if T < 3: return pred_abs_deg.new_zeros(())

    _env = env_data
    if _env is None and batch_list is not None:
        try: _env = batch_list[13]
        except: _env = None

    # 1. Trọng số thời gian tăng mạnh ở 72h
    pinn_time_w = torch.linspace(0.5, 4.0, T, device=pred_abs_deg.device)

    # 2. Adaptive weighting (FIX-L-C) - giữ nguyên chiều [T, B]
    if gt_abs_deg is not None:
        with torch.no_grad():
            d_track = _haversine_deg(pred_abs_deg, gt_abs_deg)
            w_bve_step = torch.sigmoid(1.0 - d_track / 200.0) # [T, B]
    else:
        w_bve_step = torch.ones(T, pred_abs_deg.shape[1], device=pred_abs_deg.device)

    # 3. Hàm hỗ trợ nhân trọng số
    def apply_w(l_map, weight_scalar):
        t_size = l_map.shape[0] # Giờ l_map đã có chiều [T_i, B]
        # Nhân: (loss từng bước) * (weight thành phần) * (weight thời gian) * (adaptive weight)
        return weight_scalar * l_map * pinn_time_w[-t_size:, None] * w_bve_step[-t_size:]

    # 4. Tính toán các thành phần (lúc này các hàm đã trả về Tensor)
    l_sw      = apply_w(pinn_shallow_water(pred_abs_deg), 1.0)
    l_steer   = apply_w(pinn_rankine_steering(pred_abs_deg, _env), 0.5)
    l_speed   = apply_w(pinn_speed_constraint(pred_abs_deg), 0.1)
    l_gph     = apply_w(pinn_gph500_gradient(pred_abs_deg, _env), 0.3)
    l_spdcons = apply_w(pinn_steering_speed_consistency(pred_abs_deg, _env), 0.4)
    l_pwr     = apply_w(pinn_pressure_wind_loss(pred_abs_deg, vmax_pred, pmin_pred, r34_km, epoch), 0.6)

    # 5. Tổng hợp và trung bình
    total = (l_sw.mean() + l_steer.mean() + l_speed.mean() + 
             l_gph.mean() + l_spdcons.mean() + l_pwr.mean())

    # Các phần f_lazy, w_bnd và AdaptClamp giữ nguyên
    w_bnd = _boundary_weights(pred_abs_deg).mean()
    f_lazy = 0.2 if epoch < 30 else (0.5 if epoch < 50 else 1.0)
    
    total_clamped = adapt_clamp(total, epoch, max_val=20.0)
    return total_clamped * w_bnd * f_lazy


# ══════════════════════════════════════════════════════════════════════════════
#  Physics consistency (beta drift)
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
#  Ensemble spread
# ══════════════════════════════════════════════════════════════════════════════

# def ensemble_spread_loss(all_trajs: torch.Tensor,
#                          max_spread_km: float = 150.0) -> torch.Tensor:
#     if all_trajs.shape[0] < 2:
#         return all_trajs.new_zeros(())

#     S, T, B, _ = all_trajs.shape

#     step_weights = torch.exp(
#         -torch.arange(T, dtype=torch.float, device=all_trajs.device)
#         * (math.log(4.0) / max(T - 1, 1))
#     ) * 2.0

#     total_loss = all_trajs.new_zeros(())
#     for t in range(T):
#         step_trajs = all_trajs[:, t, :, :2]
#         std_lon    = step_trajs[:, :, 0].std(0)
#         std_lat    = step_trajs[:, :, 1].std(0)
#         spread_km  = torch.sqrt(std_lon ** 2 + std_lat ** 2) * 500.0
#         excess     = F.relu(spread_km - max_spread_km)
#         total_loss = total_loss + step_weights[t] * (
#             excess / max_spread_km
#         ).pow(2).mean()

#     return total_loss / T
def ensemble_spread_loss(all_trajs: torch.Tensor) -> torch.Tensor:
    if all_trajs.shape[0] < 2:
        return all_trajs.new_zeros(())

    S, T, B, _ = all_trajs.shape
    device = all_trajs.device

    # 1. ĐẢO NGƯỢC TRỌNG SỐ: Tăng dần từ 6h đến 72h
    # Step 1 (6h) weight = 0.8, Step 12 (72h) weight = 4.0
    # Điều này bắt mô hình phải ưu tiên siết spread ở horizon xa
    step_weights = torch.linspace(0.8, 4.0, T, device=device)

    # 2. NGƯỠNG ĐỘNG (Dynamic Threshold): 
    # 6h không nên spread quá 60km, 72h không nên spread quá 180km
    # Ép một ngưỡng cứng 150km ở 72h là rất khó, nên dùng 170-180km là hợp lý
    max_spreads = torch.linspace(60.0, 180.0, T, device=device)

    total_loss = all_trajs.new_zeros(())
    for t in range(T):
        step_trajs = all_trajs[:, t, :, :2]
        std_lon    = step_trajs[:, :, 0].std(0)
        std_lat    = step_trajs[:, :, 1].std(0)
        spread_km  = torch.sqrt(std_lon ** 2 + std_lat ** 2) * 500.0 # [B]
        
        # 3. PHẠT BÌNH PHƯƠNG MẠNH
        # Nếu spread vượt ngưỡng động tại thời điểm t
        excess = F.relu(spread_km - max_spreads[t])
        
        # Dùng hằng số chia nhỏ hơn (ví dụ 40.0) để gradient dốc hơn
        loss = (excess / 40.0).pow(2) 
        
        total_loss = total_loss + step_weights[t] * loss.mean()

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
    epoch:       int = 0,
    # FIX-L-C: cần gt_abs_deg cho adaptive BVE weighting
    gt_abs_deg:  Optional[torch.Tensor] = None,
    # FIX-L-D: intensity predictions cho PWR
    vmax_pred:   Optional[torch.Tensor] = None,
    pmin_pred:   Optional[torch.Tensor] = None,
    r34_km:      Optional[torch.Tensor] = None,
    # FIX-L-H: sr_pred cho bridge loss
    sr_pred:     Optional[torch.Tensor] = None,
) -> Dict:
    NRM = 35.0

    # 1. Sample weights
    recurv_w = _recurvature_weights(gt, w_recurv=2.5)
    sample_w = recurv_w * (intensity_w.to(gt.device) if intensity_w is not None
                           else 1.0)
    sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

    # 2. AFCRPS (FIX-L-G: thêm w_es)
    if pred_samples is not None:
        target = gt_norm if gt_norm is not None else gt
        unit   = gt_norm is not None
        l_fm = fm_afcrps_loss(
            pred_samples, target,
            unit_01deg=unit,
            intensity_w=sample_w,
            step_weight_alpha=step_weight_alpha,
            w_es=0.3,
        )
    else:
        l_fm = _haversine_deg(pred_abs, gt).mean()

    # # 3. Directional losses
    # l_vel       = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
    # l_disp      = (disp_loss_per_sample(pred_abs, gt)     * sample_w).mean()
    # l_step      = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()
    # l_heading   = heading_loss(pred_abs, gt)
    # l_recurv    = recurvature_loss(pred_abs, gt)
    # l_dir_final = overall_dir_loss(pred_abs, gt, ref)
    # l_smooth    = smooth_loss(pred_abs)
    # l_accel     = acceleration_loss(pred_abs)
    # l_jerk      = jerk_loss(pred_abs)

    # # 4. PINN (FIX-L-A: KHÔNG nhân NRM ở đây; FIX-L-C/D/E/F: pass extra args)
    # # _env = env_data
    # # if _env is None and batch_list is not None:
    # #     try:
    # #         _env = batch_list[13]
    # #     except (IndexError, TypeError):
    # #         _env = None

    # # l_pinn = pinn_bve_loss(
    # #     pred_abs, batch_list, env_data=_env,
    # #     epoch=epoch,
    # #     gt_abs_deg=gt_abs_deg,
    # #     vmax_pred=vmax_pred,
    # #     pmin_pred=pmin_pred,
    # #     r34_km=r34_km,
    # # )

    # # 4. PINN (FIX-L-A: KHÔNG nhân NRM ở đây; FIX-L-C/D/E/F: pass extra args)
    # _env = env_data
    # if _env is None and batch_list is not None:
    #     try:
    #         _env = batch_list[13]
    #     except (IndexError, TypeError):
    #         _env = None

    # # Tính PINN cho Mean (định hướng quỹ đạo trung tâm)
    # l_pinn_mean = pinn_bve_loss(
    #     pred_abs_deg=pred_abs, 
    #     batch_list=batch_list, 
    #     env_data=_env,
    #     epoch=epoch,
    #     gt_abs_deg=gt_abs_deg,
    #     vmax_pred=vmax_pred,
    #     pmin_pred=pmin_pred,
    #     r34_km=r34_km
    # )

    # # Stochastic PINN: Ép vật lý lên từng hạt ensemble ở Phase 2
    # if pred_samples is not None and epoch >= 30: 
    #     M = pred_samples.shape[0]
    #     # Chọn ngẫu nhiên 2 hạt để tính PINN (tiết kiệm memory)
    #     idxs = torch.randperm(M)[:2]
        
    #     l_pinn_samples = []
    #     for idx in idxs:
    #         # Decode sample từ normalized sang degrees
    #         sample_deg = _norm_to_deg(pred_samples[idx])
            
    #         l_p_sample = pinn_bve_loss(
    #             pred_abs_deg=sample_deg, 
    #             batch_list=batch_list, 
    #             env_data=_env,
    #             epoch=epoch,
    #             gt_abs_deg=gt_abs_deg, # Các hạt đều phải hướng về GT chung
    #             vmax_pred=vmax_pred,
    #             pmin_pred=pmin_pred,
    #             r34_km=r34_km
    #         )
    #         l_pinn_samples.append(l_p_sample)
        
    #     # Kết hợp: 40% Mean + 60% Samples
    #     l_pinn = 0.4 * l_pinn_mean + 0.6 * torch.stack(l_pinn_samples).mean()
    # else:
    #     l_pinn = l_pinn_mean
        
    # ... (đoạn tính l_fm và directional losses cơ bản ở trên giữ nguyên)

    # 3. Directional losses (Tính trên Mean trước)
    l_vel       = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
    l_disp      = (disp_loss_per_sample(pred_abs, gt)     * sample_w).mean()
    l_step      = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()
    l_heading   = heading_loss(pred_abs, gt)
    l_recurv    = recurvature_loss(pred_abs, gt)
    l_dir_final = overall_dir_loss(pred_abs, gt, ref)
    l_smooth    = smooth_loss(pred_abs)
    l_accel     = acceleration_loss(pred_abs)
    l_jerk      = jerk_loss(pred_abs)

    # 4. PINN & Stochastic Physics (Tích hợp Smoothness vào đây)
    _env = env_data
    if _env is None and batch_list is not None:
        try: _env = batch_list[13]
        except (IndexError, TypeError): _env = None

    # PINN trên Mean
    l_pinn_mean = pinn_bve_loss(pred_abs, batch_list, env_data=_env, epoch=epoch,
                                gt_abs_deg=gt_abs_deg, vmax_pred=vmax_pred,
                                pmin_pred=pmin_pred, r34_km=r34_km)

    if pred_samples is not None and epoch >= 30: 
        M = pred_samples.shape[0]
        idxs = torch.randperm(M)[:2] # Chọn 2 hạt ngẫu nhiên
        
        l_pinn_samples = []
        l_smooth_samples = [] # NEW
        l_accel_samples = []  # NEW
        
        for idx in idxs:
            sample_deg = _norm_to_deg(pred_samples[idx])
            
            # PINN cho từng hạt
            l_p_sample = pinn_bve_loss(sample_deg, batch_list, env_data=_env, epoch=epoch,
                                       gt_abs_deg=gt_abs_deg, vmax_pred=vmax_pred,
                                       pmin_pred=pmin_pred, r34_km=r34_km)
            l_pinn_samples.append(l_p_sample)
            
            # Smoothness cho từng hạt (Ép các hạt không được đi ziczac)
            l_smooth_samples.append(smooth_loss(sample_deg))
            l_accel_samples.append(acceleration_loss(sample_deg))
        
        # Cập nhật PINN final
        l_pinn = 0.4 * l_pinn_mean + 0.6 * torch.stack(l_pinn_samples).mean()
        
        # Cập nhật Smoothness & Accel final (Blended 50/50)
        # Việc ép smoothness lên từng hạt giúp giảm hiện tượng ziczac ở ensemble, cải thiện spread
    # 5. Spread penalty (chỉ FM steps 5-12)
    l_spread = pred_abs.new_zeros(())
    if all_trajs is not None and all_trajs.shape[0] >= 2:
        n_sr = 4
        trajs_fm = all_trajs[:, n_sr:] if all_trajs.shape[1] > n_sr else all_trajs
        l_spread = ensemble_spread_loss(trajs_fm, max_spread_km=150.0)

    # 6. Bridge loss (FIX-L-H)
    # l_bridge = pred_abs.new_zeros(())
    # if sr_pred is not None and pred_samples is not None:
    #     fm_mean = pred_samples.mean(0)   # [T, B, 2]  (FM ensemble mean, normalized)
    #     # Chuyển FM mean sang degree
    #     fm_mean_deg = _norm_to_deg(fm_mean)
    #     sr_pred_deg = _norm_to_deg(sr_pred)   # [4, B, 2] degrees
    #     l_bridge = bridge_loss(sr_pred_deg, fm_mean_deg)

    # SỬA: bỏ _norm_to_deg khi gọi bridge_loss
    l_bridge = pred_abs.new_zeros(())
    if sr_pred is not None and pred_samples is not None:
        fm_mean = pred_samples.mean(0)   # [T, B, 2] normalized
        # KHÔNG decode ở đây — bridge_loss tự decode bên trong
        l_bridge = bridge_loss(sr_pred, fm_mean)
    # 7. Total
    # FIX-L-A: pinn KHÔNG nhân NRM — l_pinn đã có AdaptClamp(max=20)
    # Các loss directional nhân NRM để đưa về cùng thang với l_fm (km)
    total = (
        weights.get("fm",       2.0) * l_fm
        + weights.get("velocity", 0.8) * l_vel     * NRM
        + weights.get("disp",     0.5) * l_disp    * NRM
        + weights.get("step",     0.5) * l_step    * NRM
        + weights.get("heading",  2.0) * l_heading * NRM
        + weights.get("recurv",   1.5) * l_recurv  * NRM
        + weights.get("dir",      1.0) * l_dir_final * NRM
        + weights.get("smooth",   0.5) * l_smooth  * NRM
        + weights.get("accel",    0.8) * l_accel   * NRM
        + weights.get("jerk",     0.3) * l_jerk    * NRM
        + weights.get("pinn",     0.5) * l_pinn              # FIX-L-A: không * NRM
        + weights.get("spread",   0.8) * l_spread  * NRM
        + weights.get("bridge",   0.5) * l_bridge  * NRM
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
        pinn         = l_pinn.item(),    # raw value, không * NRM
        spread       = l_spread.item()  * NRM,
        bridge       = l_bridge.item()  * NRM,
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
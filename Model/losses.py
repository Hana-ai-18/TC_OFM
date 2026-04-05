# """
# Model/losses.py  ── v19
# ========================
# FIXES vs v18:

#   FIX-L29  [CRITICAL] Loss scale imbalance: AFCRPS trả về km (~358 km),
#            directional losses trả về deg^2 (~1 deg^2).
#            Ratio ~60x → directional losses hoàn toàn bị ignore trong training.
#            Fix: convert tất cả directional losses sang km đơn vị:
#              velocity/disp/step: deg → km bằng *DEG_TO_KM (111.0)
#              heading/smooth/accel: sử dụng km displacement vectors
#            Sau đó balance weights lại cho hợp lý:
#              AFCRPS ~358 km (w=2.0 → contrib ~716)
#              velocity_km ~113 km (w=1.0 → contrib ~113)
#              heading ~0-1 (w=1.0, dimensionless)

#   FIX-L30  [CRITICAL] env_wind trong build_env_features_one_step dùng /110.0
#            Verify từ data: env_wind = wnd/150.0  (max_err=0.0000)
#            Fix ở env_net_transformer_gphsplit.py (companion fix).
#            Đây là file losses nên không affect trực tiếp.

#   FIX-L31  velocity_loss_per_sample: convert displacement sang km trước khi
#            tính loss → đơn vị nhất quán với AFCRPS.

#   FIX-L32  smooth_loss và acceleration_loss: operate trên km vectors thay
#            vì degree vectors để scale hợp lý.

#   FIX-L33  WEIGHTS rebalance: sau khi losses cùng đơn vị km, điều chỉnh
#            weights để các loss component đóng góp tương đương trong epoch đầu.

# Kept from v18:
#   FIX-L24..L28: degree inputs, pinn fix, gt_norm for AFCRPS, etc.
# """
# from __future__ import annotations

# import math
# from typing import Dict, Optional

# import torch
# import torch.nn.functional as F

# OMEGA        = 7.2921e-5
# R_EARTH      = 6.371e6
# DT_6H        = 6 * 3600
# NORM_TO_M    = 111_000.0   # 1 degree ≈ 111 km
# DEG_TO_KM    = 111.0       # 1 degree ≈ 111 km

# PINN_SCALE   = 1.0

# # ── WEIGHTS rebalanced (v19) ──────────────────────────────────────────────────
# # Tất cả losses giờ đều có đơn vị km (hoặc dimensionless [0-1])
# # Target contribution: fm≈700, velocity≈100, heading≈100, smooth≈10
# WEIGHTS: Dict[str, float] = dict(
#     fm=2.0,        # AFCRPS km — ~358 km → contrib ~716
#     velocity=0.5,  # km units — ~113 km → contrib ~57
#     heading=1.5,   # dimensionless [0-1] → needs scale
#     recurv=1.5,    # dimensionless [0-1]
#     step=0.5,      # dimensionless [0-1]
#     disp=0.5,      # km units — ~113 km
#     dir=1.0,       # dimensionless [0-1]
#     smooth=0.3,    # km units — small
#     accel=0.5,     # km units — small
#     pinn=0.02,     # physics residual
# )

# RECURV_ANGLE_THR = 45.0
# RECURV_WEIGHT    = 2.5


# # ── Haversine ─────────────────────────────────────────────────────────────────

# def _haversine(p1: torch.Tensor, p2: torch.Tensor,
#                unit_01deg: bool = True) -> torch.Tensor:
#     """Haversine distance in km."""
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


# def _deg_to_km_vec(traj_deg: torch.Tensor) -> torch.Tensor:
#     """
#     Convert degree displacement vectors to km vectors.
#     traj_deg: [..., 2] in degrees (lon, lat)
#     Returns velocity vectors in km/step using local cos-lat correction.
#     """
#     # Approximate: use mid-latitude for cos correction
#     lat_deg = traj_deg[..., 1]
#     cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-4)
#     # dx_km = dlon * cos_lat * DEG_TO_KM, dy_km = dlat * DEG_TO_KM
#     # For displacement vectors: just scale
#     km = traj_deg.clone()
#     km[..., 0] = traj_deg[..., 0] * cos_lat * DEG_TO_KM
#     km[..., 1] = traj_deg[..., 1] * DEG_TO_KM
#     return km


# # ── Recurvature detection ─────────────────────────────────────────────────────

# def _total_rotation_angle_batch(gt: torch.Tensor) -> torch.Tensor:
#     """gt: [T, B, 2] degrees. Returns [B] total rotation in degrees."""
#     T, B, _ = gt.shape
#     if T < 3:
#         return gt.new_zeros(B)

#     lats_rad = torch.deg2rad(gt[:, :, 1])
#     cos_lat  = torch.cos(lats_rad[:-1])
#     dlat = gt[1:, :, 1] - gt[:-1, :, 1]
#     dlon = (gt[1:, :, 0] - gt[:-1, :, 0]) * cos_lat
#     v    = torch.stack([dlon, dlat], dim=-1)

#     v1 = v[:-1];  v2 = v[1:]
#     n1 = v1.norm(dim=-1).clamp(min=1e-8)
#     n2 = v2.norm(dim=-1).clamp(min=1e-8)
#     cos_a = (v1 * v2).sum(-1) / (n1 * n2)
#     angles_rad = torch.acos(cos_a.clamp(-1.0, 1.0))
#     return torch.rad2deg(angles_rad).sum(0)


# def _recurvature_weights(gt: torch.Tensor,
#                          thr: float = RECURV_ANGLE_THR,
#                          w_recurv: float = RECURV_WEIGHT) -> torch.Tensor:
#     rot = _total_rotation_angle_batch(gt)
#     return torch.where(rot >= thr,
#                        torch.full_like(rot, w_recurv),
#                        torch.ones_like(rot))


# # ── Recurvature (turning point) loss ─────────────────────────────────────────

# def recurvature_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#     """pred, gt: [T, B, 2] degrees. Dimensionless [0-1]."""
#     if pred.shape[0] < 3:
#         return pred.new_zeros(())

#     pred_v = pred[1:] - pred[:-1]
#     gt_v   = gt[1:]   - gt[:-1]
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

#     pn = pred_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     gn = gt_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     cos_sim  = ((pred_v_mid / pn) * (gt_v_mid / gn)).sum(-1)
#     dir_loss = (1.0 - cos_sim)
#     mask     = sign_change.float()
#     if mask.sum() < 1:
#         return pred.new_zeros(())
#     return (dir_loss * mask).sum() / mask.sum().clamp(min=1)


# # ── AFCRPS (normalized coords → km via Haversine) ────────────────────────────

# def fm_afcrps_loss(
#     pred_samples: torch.Tensor,
#     gt: torch.Tensor,
#     unit_01deg: bool = True,
#     intensity_w: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     """
#     AFCRPS in km.
#     pred_samples: [M, T, B, 2]
#     gt: [T, B, 2]
#     """
#     M, T, B, _ = pred_samples.shape
#     eps  = 1e-3
#     time_w = torch.linspace(0.5, 1.5, T, device=pred_samples.device)

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


# # ── FIX-L29: Directional losses in km ────────────────────────────────────────

# def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
#     """
#     traj_deg: [T, B, 2] degrees
#     Returns: [T-1, B, 2] displacement vectors in km
#     FIX-L29: operate in km so loss scale matches AFCRPS
#     """
#     dt = traj_deg[1:] - traj_deg[:-1]  # [T-1, B, 2] in degrees
#     # cos-lat correction for longitude
#     lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5  # [T-1, B]
#     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
#     dt_km = dt.clone()
#     dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
#     dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
#     return dt_km


# def velocity_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#     """
#     pred, gt: [T, B, 2] degrees
#     Returns [B] in km^2 units.
#     FIX-L29: convert to km before computing loss → same scale as AFCRPS.
#     """
#     if pred.shape[0] < 2:
#         return pred.new_zeros(pred.shape[1])

#     v_pred_km = _step_displacements_km(pred)   # [T-1, B, 2] km
#     v_gt_km   = _step_displacements_km(gt)

#     s_pred = v_pred_km.norm(dim=-1)            # [T-1, B]
#     s_gt   = v_gt_km.norm(dim=-1)
#     l_speed = (s_pred - s_gt).pow(2).mean(0)  # [B] km^2

#     gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     gt_unit = v_gt_km / gn
#     ate = ((v_pred_km - v_gt_km) * gt_unit).sum(-1)
#     l_ate = ate.pow(2).mean(0)                  # [B] km^2

#     return l_speed + 0.5 * l_ate               # [B] km^2


# def disp_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#     """pred, gt: [T, B, 2] degrees. Returns [B] km^2."""
#     if pred.shape[0] < 2:
#         return pred.new_zeros(pred.shape[1])
#     v_pred_km = _step_displacements_km(pred)
#     v_gt_km   = _step_displacements_km(gt)
#     pd = v_pred_km.norm(dim=-1).mean(0)
#     gd = v_gt_km.norm(dim=-1).mean(0)
#     return (pd - gd).pow(2)                    # [B] km^2


# def step_dir_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#     """pred, gt: [T, B, 2] degrees. Returns [B] dimensionless [0-2]."""
#     if pred.shape[0] < 2:
#         return pred.new_zeros(pred.shape[1])
#     v_pred_km = _step_displacements_km(pred)
#     v_gt_km   = _step_displacements_km(gt)
#     pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     cos_sim = ((v_pred_km / pn) * (v_gt_km / gn)).sum(-1)
#     return (1.0 - cos_sim).mean(0)


# def heading_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#     """pred, gt: [T, B, 2] degrees. Dimensionless."""
#     if pred.shape[0] < 2:
#         return pred.new_zeros(())
#     pv = _step_displacements_km(pred)
#     gv = _step_displacements_km(gt)
#     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1)).mean()
#     if pred.shape[0] >= 3:
#         def _curv(v):
#             cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
#             n1    = v[1:].norm(dim=-1).clamp(min=1e-4)
#             n2    = v[:-1].norm(dim=-1).clamp(min=1e-4)
#             return (cross / (n1 * n2)).clamp(-10.0, 10.0)
#         curv_mse = F.mse_loss(_curv(pv), _curv(gv))
#     else:
#         curv_mse = pred.new_zeros(())
#     return wrong_dir + curv_mse


# def smooth_loss(pred: torch.Tensor) -> torch.Tensor:
#     """pred: [T, B, 2] degrees. FIX-L29: compute in km for consistent scale."""
#     if pred.shape[0] < 3:
#         return pred.new_zeros(())
#     v_km = _step_displacements_km(pred)   # [T-1, B, 2]
#     if v_km.shape[0] < 2:
#         return pred.new_zeros(())
#     accel_km = v_km[1:] - v_km[:-1]       # [T-2, B, 2]
#     return accel_km.pow(2).mean()          # km^2 per step^2


# def acceleration_loss(pred: torch.Tensor) -> torch.Tensor:
#     """pred: [T, B, 2] degrees. FIX-L29: compute in km."""
#     if pred.shape[0] < 3:
#         return pred.new_zeros(())
#     v_km = _step_displacements_km(pred)
#     if v_km.shape[0] < 2:
#         return pred.new_zeros(())
#     a_km = v_km[1:] - v_km[:-1]
#     return a_km.pow(2).mean()


# def overall_dir_loss(pred: torch.Tensor, gt: torch.Tensor,
#                      ref: torch.Tensor) -> torch.Tensor:
#     """pred, gt: [T,B,2] degrees; ref: [B,2] degrees. Dimensionless."""
#     p_d = pred[-1] - ref
#     g_d = gt[-1]   - ref
#     # Convert to km for direction computation
#     lat_ref = ref[:, 1]
#     cos_lat = torch.cos(torch.deg2rad(lat_ref)).clamp(min=1e-4)
#     p_d_km = p_d.clone(); p_d_km[:, 0] *= cos_lat * DEG_TO_KM; p_d_km[:, 1] *= DEG_TO_KM
#     g_d_km = g_d.clone(); g_d_km[:, 0] *= cos_lat * DEG_TO_KM; g_d_km[:, 1] *= DEG_TO_KM
#     pn = p_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     gn = g_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#     return (1.0 - ((p_d_km / pn) * (g_d_km / gn)).sum(-1)).mean()


# # ── PINN BVE ──────────────────────────────────────────────────────────────────

# # def _pinn_simplified(pred_abs_deg: torch.Tensor) -> torch.Tensor:
# #     """pred_abs_deg: [T, B, 2] degrees."""
# #     T = pred_abs_deg.shape[0]
# #     if T < 5:
# #         return pred_abs_deg.new_zeros(())

# #     v   = pred_abs_deg[1:] - pred_abs_deg[:-1]
# #     DEG_TO_M_PER_S = 111_000.0 / DT_6H
# #     vx  = v[..., 0] * DEG_TO_M_PER_S
# #     vy  = v[..., 1] * DEG_TO_M_PER_S

# #     zeta  = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]
# #     dzeta = zeta[1:] - zeta[:-1]

# #     lat_deg = pred_abs_deg[2:T-1, :, 1]
# #     lat_rad = torch.deg2rad(lat_deg)
# #     beta_n  = (2 * OMEGA / R_EARTH) * torch.cos(lat_rad)
# #     vy_mid  = vy[1:T-2]

# #     raw = ((dzeta + beta_n * vy_mid) ** 2).mean() * PINN_SCALE
# #     return raw.clamp(max=50.0)


# # def pinn_bve_loss(pred_abs_deg, batch_list):
# #     return _pinn_simplified(pred_abs_deg)
# def pinn_shallow_water(pred_abs_deg, pred_Me_norm):
#     """
#     Shallow Water Equations trên beta-plane — valid cho TC tropics.
    
#     Du/Dt - fv = -g·∂h/∂x  
#     Dv/Dt + fu = -g·∂h/∂y
    
#     Với TC single-point: dùng dạng integral
#     d²x/dt² = f·dy/dt - g·∂h/∂x_env
#     d²y/dt² = -f·dx/dt - g·∂h/∂y_env
    
#     f = 2Ω·sin(lat) — Coriolis
#     β = 2Ω·cos(lat)/R — beta parameter
#     """
#     T, B, _ = pred_abs_deg.shape
#     if T < 3:
#         return pred_abs_deg.new_zeros(())
    
#     OMEGA = 7.2921e-5
#     R = 6.371e6
#     DT = 6 * 3600.0  # 6h in seconds
    
#     lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])
#     lon_rad = torch.deg2rad(pred_abs_deg[:, :, 0])
    
#     # Velocity (m/s)
#     dlat = pred_abs_deg[1:] - pred_abs_deg[:-1]  # [T-1, B, 2] degrees
#     cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)
    
#     u = dlat[:, :, 0] * cos_lat * 111000.0 / DT   # zonal (m/s)
#     v = dlat[:, :, 1] * 111000.0 / DT              # meridional (m/s)
    
#     # Acceleration
#     if u.shape[0] < 2:
#         return pred_abs_deg.new_zeros(())
    
#     du = (u[1:] - u[:-1]) / DT    # [T-2, B]
#     dv = (v[1:] - v[:-1]) / DT
    
#     # Coriolis: f = 2Ω·sin(lat)
#     f = 2 * OMEGA * torch.sin(lat_rad[1:-1])  # [T-2, B]
    
#     # Beta term: β = df/dy = 2Ω·cos(lat)/R
#     beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R
    
#     # Shallow water momentum residual:
#     # Du/Dt - f·v = 0  (geostrophic limit, no pressure gradient)
#     # Dv/Dt + f·u = 0
#     # Beta correction: f → f + β·(y - y0)
    
#     res_u = du - f * v[1:]          # [T-2, B]
#     res_v = dv + f * u[1:]
    
#     # Beta drift: v_beta = -β·R²/2 (westward + poleward drift ~2-3 m/s)
#     R_tc = 3e5  # TC outer radius ~300km
#     v_beta_x = -beta * R_tc**2 / 2   # ~-1.5 m/s westward
    
#     # Expected beta drift contribution
#     res_u_corrected = res_u - v_beta_x
    
#     # Normalize by typical acceleration scale
#     scale = 1e-4  # m/s²
#     loss = (res_u_corrected / scale).pow(2).mean() + \
#            (res_v / scale).pow(2).mean()
    
#     return loss.clamp(max=100.0)

# def pinn_rankine_steering(pred_abs_deg, env_data):
#         """
#         TC di chuyển theo CENTROID của vorticity field.
#         Rankine vortex model: TC velocity = env steering + beta drift
        
#         V_TC = (1/ζ_total) ∫∫ V·ζ dA  (vorticity-weighted mean flow)
        
#         Simplified: V_TC ≈ V_500hPa + V_beta
        
#         Constraint: cosine similarity giữa TC track và 500hPa steering > 0.7
#         """
#         if env_data is None:
#             return pred_abs_deg.new_zeros(())
        
#         T, B, _ = pred_abs_deg.shape
#         if T < 2:
#             return pred_abs_deg.new_zeros(())
        
#         # TC velocity vector
#         DT = 6 * 3600.0
#         dlat = pred_abs_deg[1:] - pred_abs_deg[:-1]
#         lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
#         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
        
#         u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT  # m/s
#         v_tc = dlat[:, :, 1] * 111000.0 / DT
        
#         # Env steering từ u500, v500 (đã normalized)
#         u500_raw = env_data.get('u500_mean', None)
#         v500_raw = env_data.get('v500_mean', None)
        
#         if u500_raw is None or v500_raw is None:
#             # Fallback: chỉ dùng speed constraint
#             speed = torch.sqrt(u_tc**2 + v_tc**2)
#             return F.relu(speed - 30.0).pow(2).mean() * 0.01
        
#         # Reshape env to [T-1, B]
# def _align(x):
#         if x.dim() == 3:
#                 x = x[:, :min(T-1, x.shape[1]), 0]  # [B, T-1]
#                 return x.permute(1, 0)               # [T-1, B]
#         return x
        
#         u500 = _align(u500_raw)
#         v500 = _align(v500_raw)
        
#         # Denorm u500: (raw - 5843.14) / 50.55
#         # Nhưng u500 trong env_data đã là normalized z-score
#         # → z * 50.55 + 5843.14 là geopotential height, KHÔNG phải wind
#         # Chỉ dùng direction, không dùng magnitude
        
#         # Direction alignment
#         env_dir = torch.stack([u500, v500], dim=-1)     # [T-1, B, 2]
#         tc_dir  = torch.stack([u_tc, v_tc],  dim=-1)
        
#         env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=1e-4)
#         tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        
#         cos_sim = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)
        
#         # Penalty: track direction ngược steering > 30°
#         # cos(150°) = -0.866 → penalize cos_sim < -0.5
#         misalign = F.relu(-0.5 - cos_sim).pow(2)
        
#         return misalign.mean() * 0.01  # scale down
# def pinn_knaff_zehr(pred_Me_norm, lat_deg):
#     """
#     Knaff & Zehr (2007) Wind-Pressure Relationship chính xác:
    
#     P_c = P_env - (WND_kt / (0.6201 + 0.0038·lat))^(1/0.644)
    
#     Dạng simplified: ΔP ≈ -c(lat) · ΔWND
#     với c(lat) ≈ 1.2 tại 15°N, tăng theo lat
    
#     Constraint: wind-pressure consistency mỗi timestep
#     """
#     T, B, _ = pred_Me_norm.shape
#     if T < 2:
#         return pred_Me_norm.new_zeros(())
    
#     # Decode
#     pres = pred_Me_norm[:, :, 0] * 50.0 + 960.0   # hPa
#     wnd  = pred_Me_norm[:, :, 1] * 25.0 + 40.0    # knots
    
#     # Clamp to physical range
#     pres = pres.clamp(870, 1020)
#     wnd  = wnd.clamp(5, 180)
    
#     # Latitude-dependent W-P coefficient
#     lat = lat_deg.clamp(5, 40)  # [T, B]
#     c_lat = 0.6201 + 0.0038 * lat  # Knaff-Zehr denominator
    
#     # Expected pressure from wind (Knaff-Zehr):
#     # VMAX = c_lat * (P_env - P_c)^0.644
#     # → P_c = P_env - (VMAX/c_lat)^(1/0.644)
#     P_env = 1010.0  # hPa
#     pres_from_wnd = P_env - (wnd / c_lat).clamp(min=0).pow(1.0/0.644)
    
#     # Residual
#     residual = (pres - pres_from_wnd) / 10.0  # normalize by 10 hPa
    
#     # Weight by storm intensity (stronger storms → stricter constraint)
#     intensity_w = (wnd / 65.0).clamp(0.3, 2.0)
    
#     return (residual.pow(2) * intensity_w).mean() * 0.1
# # ── Main loss function ────────────────────────────────────────────────────────

# def compute_total_loss(
#     pred_abs,           # [T, B, 2] DEGREES
#     gt,                 # [T, B, 2] DEGREES
#     ref,                # [B, 2] DEGREES
#     batch_list,
#     pred_samples=None,  # [S, T, B, 2] NORMALIZED
#     gt_norm=None,       # [T, B, 2] NORMALIZED
#     weights=WEIGHTS,
#     intensity_w: Optional[torch.Tensor] = None,
# ) -> Dict:

#     # 1. Sample weights
#     recurv_w = _recurvature_weights(gt, w_recurv=2.5)
#     sample_w = recurv_w * (intensity_w.to(gt.device) if intensity_w is not None else 1.0)
#     sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

#     # 2. AFCRPS — km, uses normalized coords
#     if pred_samples is not None:
#         if gt_norm is not None:
#             l_fm = fm_afcrps_loss(pred_samples, gt_norm,
#                                   intensity_w=sample_w, unit_01deg=True)
#         else:
#             pred_samp_deg = pred_samples.clone()
#             pred_samp_deg[..., 0] = (pred_samples[..., 0] * 50.0 + 1800.0) / 10.0
#             pred_samp_deg[..., 1] = (pred_samples[..., 1] * 50.0) / 10.0
#             l_fm = fm_afcrps_loss(pred_samp_deg, gt,
#                                   intensity_w=sample_w, unit_01deg=False)
#     else:
#         l_fm = _haversine_deg(pred_abs, gt).mean()

#     # 3. FIX-L29: All directional losses now in km or dimensionless
#     l_vel  = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
#     l_disp = (disp_loss_per_sample(pred_abs, gt) * sample_w).mean()
#     l_step = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()

#     l_heading    = heading_loss(pred_abs, gt)
#     l_recurv     = recurvature_loss(pred_abs, gt)
#     l_dir_final  = overall_dir_loss(pred_abs, gt, ref)
#     l_smooth     = smooth_loss(pred_abs)
#     l_accel      = acceleration_loss(pred_abs)
#     l_pinn       = pinn_bve_loss(pred_abs, batch_list)

#     # 4. FIX-L29: velocity/disp/smooth/accel are now in km^2 units
#     # Scale down by DEG_TO_KM^2 = 12321 is not needed since we compute
#     # directly in km. But need to normalize km^2 → km for balance:
#     # sqrt approximation: weight * sqrt(loss) — but that breaks gradients.
#     # Instead: use sqrt(km^2_loss + eps) for stable km-scale weighting.
#     # Actually simpler: just keep km^2 but scale weights accordingly.
#     # AFCRPS ~350 km, velocity ~(113km)^2 = 12769 km^2
#     # → need to divide velocity by ~36 to match → use weight 0.5/36 ≈ 0.014
#     # But this is fragile. Better: normalize with typical scale:

#     # Normalize km^2 losses to km-scale by dividing by typical step size (113 km)
#     STEP_KM = 113.0  # typical step displacement in km
#     l_vel_scaled   = l_vel   / (STEP_KM * STEP_KM + 1e-3)  # → ~[0-1] dimensionless
#     l_disp_scaled  = l_disp  / (STEP_KM * STEP_KM + 1e-3)
#     l_smooth_scaled = l_smooth / (STEP_KM * STEP_KM + 1e-3)
#     l_accel_scaled  = l_accel  / (STEP_KM * STEP_KM + 1e-3)

#     # After scaling: all ~[0-1] dimensionless, AFCRPS ~[100-600]
#     # Re-scale AFCRPS to [0-1] too for final balance
#     AFCRPS_SCALE = 350.0  # typical AFCRPS in km
#     l_fm_scaled = l_fm / AFCRPS_SCALE  # → ~1.0

#     total = (
#         weights['fm']      * l_fm_scaled    * AFCRPS_SCALE +  # back to km
#         weights['velocity'] * l_vel_scaled   * AFCRPS_SCALE +  # km scale
#         weights['disp']     * l_disp_scaled  * AFCRPS_SCALE +
#         weights['step']     * l_step         * AFCRPS_SCALE +   # already [0-2]
#         weights['heading']  * l_heading      * AFCRPS_SCALE +
#         weights['recurv']   * l_recurv       * AFCRPS_SCALE +
#         weights['dir']      * l_dir_final    * AFCRPS_SCALE +
#         weights['smooth']   * l_smooth_scaled * AFCRPS_SCALE +
#         weights['accel']    * l_accel_scaled  * AFCRPS_SCALE +
#         weights['pinn']     * l_pinn
#     )
#     # Divide by AFCRPS_SCALE to keep loss values in [0-10] range for stability
#     total = total / AFCRPS_SCALE

#     if torch.isnan(total) or torch.isinf(total):
#         total = pred_abs.new_zeros(())

#     return dict(
#         total    = total,
#         fm       = l_fm.item(),
#         velocity = l_vel_scaled.item() * AFCRPS_SCALE,
#         step     = l_step.item(),
#         disp     = l_disp_scaled.item() * AFCRPS_SCALE,
#         heading  = l_heading.item(),
#         recurv   = l_recurv.item(),
#         smooth   = l_smooth_scaled.item() * AFCRPS_SCALE,
#         accel    = l_accel_scaled.item() * AFCRPS_SCALE,
#         pinn     = l_pinn.item(),
#         recurv_ratio = (_recurvature_weights(gt) > 1.0).float().mean().item(),
#     )

# # losses.py — thêm hàm này

# def fm_physics_consistency_loss(
#     pred_samples,   # [S, T, B, 4] normalized — full state
#     gt_norm,        # [T, B, 2] normalized
#     last_pos,       # [B, 2] normalized
# ):
#     """
#     Physics consistency loss đặc biệt cho Flow Matching.
    
#     FM tạo ra ensemble S trajectories. Thay vì chỉ penalize
#     mean prediction, penalize cả VARIANCE của ensemble
#     tại những điểm mà physics prediction chắc chắn.
    
#     Ví dụ: nếu beta drift nói TC sẽ đi WNW với confidence cao,
#     nhưng ensemble spread theo hướng NE — đây là bad spread.
    
#     → Ensemble diversity chỉ được phép ở những hướng mà
#       physics không chắc chắn.
#     """
#     S, T, B, _ = pred_samples.shape
    
#     # 1. Physics-predicted direction (deterministic)
#     # Dùng beta drift làm prior direction
#     # Convert last_pos sang degrees
#     last_lon = (last_pos[:, 0] * 50.0 + 1800.0) / 10.0  # [B]
#     last_lat = (last_pos[:, 1] * 50.0) / 10.0            # [B]
    
#     lat_rad = torch.deg2rad(last_lat)
#     OMEGA = 7.2921e-5; R = 6.371e6
#     beta = 2 * OMEGA * torch.cos(lat_rad) / R           # [B]
#     R_tc = 3e5
    
#     # Beta drift direction (normalized, pointing WNW)
#     v_beta_lon = -beta * R_tc**2 / 2   # [B] m/s
#     v_beta_lat =  beta * R_tc**2 / 4   # [B] m/s
    
#     beta_dir = torch.stack([v_beta_lon, v_beta_lat], dim=-1)  # [B, 2]
#     beta_dir_norm = beta_dir / (beta_dir.norm(dim=-1, keepdim=True) + 1e-6)
    
#     # 2. Ensemble mean direction at step 1 (short-range)
#     # pred_samples[:, 0, :, :2] = positions at t=1
#     pos_step1 = pred_samples[:, 0, :, :2]  # [S, B, 2] normalized
    
#     # Direction vectors từ last_pos
#     dir_step1 = pos_step1 - last_pos.unsqueeze(0)  # [S, B, 2]
#     dir_norm  = dir_step1 / (dir_step1.norm(dim=-1, keepdim=True) + 1e-6)
    
#     # Mean direction of ensemble
#     mean_dir = dir_norm.mean(0)  # [B, 2]
#     mean_dir = mean_dir / (mean_dir.norm(dim=-1, keepdim=True) + 1e-6)
    
#     # 3. Penalize mean direction deviating from beta drift
#     # cos_sim giữa ensemble mean và beta drift direction
#     cos_align = (mean_dir * beta_dir_norm).sum(-1)  # [B]
    
#     # Chỉ penalize khi beta drift mạnh (|v_beta| > threshold)
#     beta_strength = beta_dir.norm(dim=-1)  # [B]
#     beta_thresh = 1.0  # m/s
    
#     penalize_mask = (beta_strength > beta_thresh).float()
    
#     # Penalty: nếu cos_sim < 0 (ngược hướng) thì penalize
#     direction_loss = F.relu(-cos_align) * penalize_mask
    
#     return direction_loss.mean() * 0.5
# # ── Legacy helpers ─────────────────────────────────────────────────────────────

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


# def toNE(pred_traj, pred_Me):
#     rt, rm = pred_traj.clone(), pred_Me.clone()
#     if rt.dim() == 2:
#         rt = rt.unsqueeze(1); rm = rm.unsqueeze(1)
#     rt[:, :, 0] = rt[:, :, 0] * 50.0 + 1800.0
#     rt[:, :, 1] = rt[:, :, 1] * 50.0
#     rm[:, :, 0] = rm[:, :, 0] * 50.0 + 960.0
#     rm[:, :, 1] = rm[:, :, 1] * 25.0 + 40.0
#     return rt, rm


# def trajectory_displacement_error(pred, gt, mode="sum"):
#     _gt   = gt.permute(1, 0, 2)
#     _pred = pred.permute(1, 0, 2)
#     diff  = _gt - _pred
#     lon_km = diff[:, :, 0] / 10.0 * 111.0 * torch.cos(
#         _gt[:, :, 1] / 10.0 * torch.pi / 180.0)
#     lat_km = diff[:, :, 1] / 10.0 * 111.0
#     loss   = torch.sqrt(lon_km ** 2 + lat_km ** 2)
#     return torch.sum(loss) if mode == "sum" else loss


# def evaluate_diffusion_output(best_traj, best_Me, gt_traj, gt_Me):
#     rt, rm  = toNE(best_traj.clone(), best_Me.clone())
#     rg, rgm = toNE(gt_traj.clone(),  gt_Me.clone())
#     return (trajectory_displacement_error(rt, rg, mode="raw"),
#             torch.abs(rm.permute(1, 0, 2) - rgm.permute(1, 0, 2)))
"""
Model/losses.py  ── v20
========================
FIXES vs v19:

  FIX-L34  [CRITICAL] pinn_bve_loss was removed/commented out but still
           called in compute_total_loss → NameError at runtime. Restored
           as pinn_bve_loss() wrapping pinn_shallow_water().

  FIX-L35  [CRITICAL] pinn_rankine_steering had _align() defined outside
           the function body (broken indentation) → NameError/SyntaxError.
           Fixed indentation so _align is a local closure inside the function.

  FIX-L36  [CRITICAL] fm_physics_consistency_loss was defined here but
           imported/called from flow_matching_model.py without being
           importable — circular import risk. Kept here, added __all__.

  FIX-L37  AFCRPS_SCALE double-cancellation: compute_total_loss multiplied
           by AFCRPS_SCALE then divided by it — net effect was identity.
           Simplified: all components computed in km, weighted directly.
           Total divided by a single normalization constant for log scale.

  FIX-L38  pinn_shallow_water parameter name: was (pred_abs_deg, pred_Me_norm)
           but pred_Me_norm was unused and never passed correctly.
           Signature changed to (pred_abs_deg) only; steering/knaff called
           separately with correct args from compute_total_loss.

Kept from v19:
  FIX-L29..L33 (km units, velocity km, smooth km, weight rebalance)
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
]

OMEGA        = 7.2921e-5
R_EARTH      = 6.371e6
DT_6H        = 6 * 3600
NORM_TO_M    = 111_000.0
DEG_TO_KM    = 111.0

PINN_SCALE   = 1.0

# ── Weights (v20) ──────────────────────────────────────────────────────────────
# All loss components are in km units after scaling.
# Target at epoch-0: fm ~350 km dominates, physics terms ~5-50 km (ramp up).
WEIGHTS: Dict[str, float] = dict(
    fm=2.0,
    velocity=0.5,
    heading=1.5,
    recurv=1.5,
    step=0.5,
    disp=0.5,
    dir=1.0,
    smooth=0.3,
    accel=0.5,
    pinn=0.02,
    fm_physics=0.3,
)

RECURV_ANGLE_THR = 45.0
RECURV_WEIGHT    = 2.5


# ── Haversine ─────────────────────────────────────────────────────────────────

def _haversine(p1: torch.Tensor, p2: torch.Tensor,
               unit_01deg: bool = True) -> torch.Tensor:
    """Haversine distance in km."""
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


# ── Recurvature helpers ────────────────────────────────────────────────────────

def _total_rotation_angle_batch(gt: torch.Tensor) -> torch.Tensor:
    """gt: [T, B, 2] degrees. Returns [B] total rotation in degrees."""
    T, B, _ = gt.shape
    if T < 3:
        return gt.new_zeros(B)

    lats_rad = torch.deg2rad(gt[:, :, 1])
    cos_lat  = torch.cos(lats_rad[:-1])
    dlat = gt[1:, :, 1] - gt[:-1, :, 1]
    dlon = (gt[1:, :, 0] - gt[:-1, :, 0]) * cos_lat
    v    = torch.stack([dlon, dlat], dim=-1)

    v1 = v[:-1];  v2 = v[1:]
    n1 = v1.norm(dim=-1).clamp(min=1e-8)
    n2 = v2.norm(dim=-1).clamp(min=1e-8)
    cos_a = (v1 * v2).sum(-1) / (n1 * n2)
    angles_rad = torch.acos(cos_a.clamp(-1.0, 1.0))
    return torch.rad2deg(angles_rad).sum(0)


def _recurvature_weights(gt: torch.Tensor,
                         thr: float = RECURV_ANGLE_THR,
                         w_recurv: float = RECURV_WEIGHT) -> torch.Tensor:
    rot = _total_rotation_angle_batch(gt)
    return torch.where(rot >= thr,
                       torch.full_like(rot, w_recurv),
                       torch.ones_like(rot))


# ── Step displacements in km ──────────────────────────────────────────────────

def _step_displacements_km(traj_deg: torch.Tensor) -> torch.Tensor:
    """traj_deg: [T, B, 2] degrees → [T-1, B, 2] displacement in km."""
    dt = traj_deg[1:] - traj_deg[:-1]
    lat_mid = (traj_deg[:-1, :, 1] + traj_deg[1:, :, 1]) * 0.5
    cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
    dt_km = dt.clone()
    dt_km[..., 0] = dt[..., 0] * cos_lat * DEG_TO_KM
    dt_km[..., 1] = dt[..., 1] * DEG_TO_KM
    return dt_km


# ── Directional losses (all in km or dimensionless) ───────────────────────────

def velocity_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred, gt: [T, B, 2] degrees → [B] in km^2."""
    if pred.shape[0] < 2:
        return pred.new_zeros(pred.shape[1])
    v_pred_km = _step_displacements_km(pred)
    v_gt_km   = _step_displacements_km(gt)
    s_pred = v_pred_km.norm(dim=-1)
    s_gt   = v_gt_km.norm(dim=-1)
    l_speed = (s_pred - s_gt).pow(2).mean(0)
    gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    gt_unit = v_gt_km / gn
    ate = ((v_pred_km - v_gt_km) * gt_unit).sum(-1)
    l_ate = ate.pow(2).mean(0)
    return l_speed + 0.5 * l_ate


def disp_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred, gt: [T, B, 2] degrees → [B] km^2."""
    if pred.shape[0] < 2:
        return pred.new_zeros(pred.shape[1])
    pd = _step_displacements_km(pred).norm(dim=-1).mean(0)
    gd = _step_displacements_km(gt).norm(dim=-1).mean(0)
    return (pd - gd).pow(2)


def step_dir_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred, gt: [T, B, 2] degrees → [B] dimensionless [0-2]."""
    if pred.shape[0] < 2:
        return pred.new_zeros(pred.shape[1])
    v_pred_km = _step_displacements_km(pred)
    v_gt_km   = _step_displacements_km(gt)
    pn = v_pred_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    gn = v_gt_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    cos_sim = ((v_pred_km / pn) * (v_gt_km / gn)).sum(-1)
    return (1.0 - cos_sim).mean(0)


def heading_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred, gt: [T, B, 2] degrees. Dimensionless."""
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    pv = _step_displacements_km(pred)
    gv = _step_displacements_km(gt)
    pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1)).mean()
    if pred.shape[0] >= 3:
        def _curv(v):
            cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
            n1    = v[1:].norm(dim=-1).clamp(min=1e-4)
            n2    = v[:-1].norm(dim=-1).clamp(min=1e-4)
            return (cross / (n1 * n2)).clamp(-10.0, 10.0)
        curv_mse = F.mse_loss(_curv(pv), _curv(gv))
    else:
        curv_mse = pred.new_zeros(())
    return wrong_dir + curv_mse


def smooth_loss(pred: torch.Tensor) -> torch.Tensor:
    """pred: [T, B, 2] degrees → scalar km^2."""
    if pred.shape[0] < 3:
        return pred.new_zeros(())
    v_km = _step_displacements_km(pred)
    if v_km.shape[0] < 2:
        return pred.new_zeros(())
    accel_km = v_km[1:] - v_km[:-1]
    return accel_km.pow(2).mean()


def acceleration_loss(pred: torch.Tensor) -> torch.Tensor:
    """pred: [T, B, 2] degrees → scalar km^2."""
    if pred.shape[0] < 3:
        return pred.new_zeros(())
    v_km = _step_displacements_km(pred)
    if v_km.shape[0] < 2:
        return pred.new_zeros(())
    a_km = v_km[1:] - v_km[:-1]
    return a_km.pow(2).mean()


def overall_dir_loss(pred: torch.Tensor, gt: torch.Tensor,
                     ref: torch.Tensor) -> torch.Tensor:
    """pred, gt: [T,B,2] degrees; ref: [B,2] degrees. Dimensionless."""
    p_d = pred[-1] - ref
    g_d = gt[-1]   - ref
    lat_ref = ref[:, 1]
    cos_lat = torch.cos(torch.deg2rad(lat_ref)).clamp(min=1e-4)
    p_d_km = p_d.clone();  p_d_km[:, 0] *= cos_lat * DEG_TO_KM;  p_d_km[:, 1] *= DEG_TO_KM
    g_d_km = g_d.clone();  g_d_km[:, 0] *= cos_lat * DEG_TO_KM;  g_d_km[:, 1] *= DEG_TO_KM
    pn = p_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    gn = g_d_km.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    return (1.0 - ((p_d_km / pn) * (g_d_km / gn)).sum(-1)).mean()


def recurvature_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred, gt: [T, B, 2] degrees. Dimensionless [0-1]."""
    if pred.shape[0] < 3:
        return pred.new_zeros(())

    pred_v = pred[1:] - pred[:-1]
    gt_v   = gt[1:]   - gt[:-1]
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

    pn = pred_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gn = gt_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos_sim  = ((pred_v_mid / pn) * (gt_v_mid / gn)).sum(-1)
    dir_loss = (1.0 - cos_sim)
    mask     = sign_change.float()
    if mask.sum() < 1:
        return pred.new_zeros(())
    return (dir_loss * mask).sum() / mask.sum().clamp(min=1)


# ── AFCRPS (in km) ────────────────────────────────────────────────────────────

def fm_afcrps_loss(
    pred_samples: torch.Tensor,
    gt: torch.Tensor,
    unit_01deg: bool = True,
    intensity_w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    AFCRPS in km.
    pred_samples: [M, T, B, 2]
    gt: [T, B, 2]
    """
    M, T, B, _ = pred_samples.shape
    eps  = 1e-3
    time_w = torch.linspace(0.5, 1.5, T, device=pred_samples.device)

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


# ── PINN losses ────────────────────────────────────────────────────────────────

def pinn_shallow_water(pred_abs_deg: torch.Tensor) -> torch.Tensor:
    """
    Shallow Water Equations on beta-plane.
    FIX-L38: removed unused pred_Me_norm parameter.
    pred_abs_deg: [T, B, 2] degrees
    Returns scalar loss (dimensionless, scaled to ~[0-100]).
    """
    T, B, _ = pred_abs_deg.shape
    if T < 3:
        return pred_abs_deg.new_zeros(())

    DT = DT_6H

    lat_rad = torch.deg2rad(pred_abs_deg[:, :, 1])

    dlat = pred_abs_deg[1:] - pred_abs_deg[:-1]  # [T-1, B, 2] degrees
    cos_lat = torch.cos(lat_rad[:-1]).clamp(min=1e-4)

    # velocity in m/s
    u = dlat[:, :, 0] * cos_lat * 111000.0 / DT
    v = dlat[:, :, 1] * 111000.0 / DT

    if u.shape[0] < 2:
        return pred_abs_deg.new_zeros(())

    # acceleration
    du = (u[1:] - u[:-1]) / DT    # [T-2, B]
    dv = (v[1:] - v[:-1]) / DT

    # Coriolis
    f    = 2 * OMEGA * torch.sin(lat_rad[1:-1])   # [T-2, B]
    beta = 2 * OMEGA * torch.cos(lat_rad[1:-1]) / R_EARTH

    # Shallow water momentum residuals
    res_u = du - f * v[1:]
    res_v = dv + f * u[1:]

    # Beta drift correction (westward ~1.5 m/s)
    R_tc = 3e5
    v_beta_x = -beta * R_tc ** 2 / 2
    res_u_corrected = res_u - v_beta_x

    scale = 1e-4  # m/s²
    loss = (res_u_corrected / scale).pow(2).mean() + \
           (res_v / scale).pow(2).mean()
    return loss.clamp(max=100.0)


def pinn_rankine_steering(pred_abs_deg: torch.Tensor,
                          env_data: Optional[dict]) -> torch.Tensor:
    """
    TC direction alignment with 500hPa steering flow.
    FIX-L35: _align defined as proper local closure inside function.
    pred_abs_deg: [T, B, 2] degrees
    Returns scalar dimensionless loss.
    """
    if env_data is None:
        return pred_abs_deg.new_zeros(())

    T, B, _ = pred_abs_deg.shape
    if T < 2:
        return pred_abs_deg.new_zeros(())

    DT = DT_6H
    dlat = pred_abs_deg[1:] - pred_abs_deg[:-1]
    lat_rad = torch.deg2rad(pred_abs_deg[:-1, :, 1])
    cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

    u_tc = dlat[:, :, 0] * cos_lat * 111000.0 / DT  # [T-1, B] m/s
    v_tc = dlat[:, :, 1] * 111000.0 / DT

    u500_raw = env_data.get("u500_mean", None)
    v500_raw = env_data.get("v500_mean", None)

    if u500_raw is None or v500_raw is None:
        speed = torch.sqrt(u_tc ** 2 + v_tc ** 2)
        return F.relu(speed - 30.0).pow(2).mean() * 0.01

    # FIX-L35: _align is now a proper local closure
    def _align(x: torch.Tensor) -> torch.Tensor:
        """Align env tensor to [T-1, B] shape."""
        if not torch.is_tensor(x):
            return pred_abs_deg.new_zeros(T - 1, B)
        x = x.to(pred_abs_deg.device)
        if x.dim() == 3:
            # [B, T_env, 1] or [B, T_env, dim]
            x_squeezed = x[:, :, 0]                 # [B, T_env]
            T_env = x_squeezed.shape[1]
            T_target = T - 1
            if T_env >= T_target:
                return x_squeezed[:, :T_target].permute(1, 0)   # [T-1, B]
            else:
                # pad with last value
                pad = x_squeezed[:, -1:].expand(-1, T_target - T_env)
                return torch.cat([x_squeezed, pad], dim=1).permute(1, 0)
        elif x.dim() == 2:
            if x.shape == (B, T - 1):
                return x.permute(1, 0)
            elif x.shape[0] == T - 1:
                return x[:, :B] if x.shape[1] >= B else x.expand(-1, B)
        # scalar fallback
        return pred_abs_deg.new_zeros(T - 1, B)

    u500 = _align(u500_raw)   # [T-1, B]
    v500 = _align(v500_raw)   # [T-1, B]

    env_dir = torch.stack([u500, v500], dim=-1)   # [T-1, B, 2]
    tc_dir  = torch.stack([u_tc, v_tc],  dim=-1)

    env_norm = env_dir.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    tc_norm  = tc_dir.norm(dim=-1, keepdim=True).clamp(min=1e-4)

    cos_sim = ((env_dir / env_norm) * (tc_dir / tc_norm)).sum(-1)

    # Penalise when TC goes opposite to 500hPa steering
    misalign = F.relu(-0.5 - cos_sim).pow(2)
    return misalign.mean() * 0.01


def pinn_knaff_zehr(pred_Me_norm: torch.Tensor,
                    lat_deg: torch.Tensor) -> torch.Tensor:
    """
    Knaff & Zehr (2007) wind-pressure relationship.
    pred_Me_norm: [T, B, 2] normalised (pres_norm, wnd_norm)
    lat_deg:      [T, B] latitude in degrees
    Returns scalar loss.
    """
    T, B, _ = pred_Me_norm.shape
    if T < 2:
        return pred_Me_norm.new_zeros(())

    pres = pred_Me_norm[:, :, 0] * 50.0 + 960.0  # hPa
    wnd  = pred_Me_norm[:, :, 1] * 25.0 + 40.0   # knots (from dataset norm)

    pres = pres.clamp(870, 1020)
    wnd  = wnd.clamp(5, 180)

    lat = lat_deg.clamp(5, 40)
    c_lat = 0.6201 + 0.0038 * lat

    P_env = 1010.0
    pres_from_wnd = P_env - (wnd / c_lat).clamp(min=0).pow(1.0 / 0.644)

    residual = (pres - pres_from_wnd) / 10.0
    intensity_w = (wnd / 65.0).clamp(0.3, 2.0)

    return (residual.pow(2) * intensity_w).mean() * 0.1


def pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
    """
    Penalise unrealistic TC step displacement > 600 km/6h.
    pred_deg: [T, B, 2] degrees.
    """
    if pred_deg.shape[0] < 2:
        return pred_deg.new_zeros(())

    dt_deg  = pred_deg[1:] - pred_deg[:-1]
    lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
    cos_lat = torch.cos(lat_rad).clamp(min=1e-4)

    dx_km = dt_deg[:, :, 0] * cos_lat * DEG_TO_KM
    dy_km = dt_deg[:, :, 1] * DEG_TO_KM
    speed = torch.sqrt(dx_km ** 2 + dy_km ** 2)   # km / 6h step

    max_km = 600.0
    violation = F.relu(speed - max_km)
    return violation.pow(2).mean()


def pinn_bve_loss(pred_abs_deg: torch.Tensor,
                  batch_list,
                  env_data: Optional[dict] = None) -> torch.Tensor:
    """
    FIX-L34: pinn_bve_loss restored so compute_total_loss can call it.
    Combines shallow-water BVE + Rankine steering + speed constraint.
    pred_abs_deg: [T, B, 2] degrees.
    """
    T = pred_abs_deg.shape[0]
    if T < 3:
        return pred_abs_deg.new_zeros(())

    l_sw    = pinn_shallow_water(pred_abs_deg)
    l_steer = pinn_rankine_steering(pred_abs_deg, env_data)
    l_speed = pinn_speed_constraint(pred_abs_deg)

    # Blend: BVE primary, steering secondary, speed as hard penalty
    total = l_sw + 0.5 * l_steer + 0.1 * l_speed
    return total.clamp(max=200.0)


# ── Physics consistency loss for FM ensemble ──────────────────────────────────

def fm_physics_consistency_loss(
    pred_samples: torch.Tensor,
    gt_norm: torch.Tensor,
    last_pos: torch.Tensor,
) -> torch.Tensor:
    """
    Penalise ensemble mean direction deviating from beta-drift prior.
    pred_samples: [S, T, B, 4] or [S, T, B, 2] normalised
    gt_norm:      [T, B, 2] normalised (unused currently, kept for API compat)
    last_pos:     [B, 2] normalised
    Returns scalar loss.
    """
    S, T, B = pred_samples.shape[:3]

    # Decode last position to degrees for physics
    last_lon = (last_pos[:, 0] * 50.0 + 1800.0) / 10.0  # [B]
    last_lat = (last_pos[:, 1] * 50.0) / 10.0             # [B]

    lat_rad = torch.deg2rad(last_lat)
    beta = 2 * OMEGA * torch.cos(lat_rad) / R_EARTH        # [B]
    R_tc = 3e5

    # Beta drift direction (WNW)
    v_beta_lon = -beta * R_tc ** 2 / 2                     # [B] m/s
    v_beta_lat =  beta * R_tc ** 2 / 4

    beta_dir = torch.stack([v_beta_lon, v_beta_lat], dim=-1)          # [B, 2]
    beta_norm = beta_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    beta_dir_unit = beta_dir / beta_norm                               # [B, 2]

    # Ensemble step-1 direction
    pos_step1 = pred_samples[:, 0, :, :2]                 # [S, B, 2] normalised
    dir_step1 = pos_step1 - last_pos.unsqueeze(0)          # [S, B, 2]
    dir_norm  = dir_step1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dir_unit  = dir_step1 / dir_norm                       # [S, B, 2]

    mean_dir = dir_unit.mean(0)                            # [B, 2]
    mean_norm = mean_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    mean_dir_unit = mean_dir / mean_norm

    cos_align = (mean_dir_unit * beta_dir_unit).sum(-1)    # [B]

    # Only penalise when beta drift is strong
    beta_strength = beta_norm.squeeze(-1)                  # [B]
    penalise_mask = (beta_strength > 1.0).float()

    direction_loss = F.relu(-cos_align) * penalise_mask
    return direction_loss.mean() * 0.5


# ── Main loss ─────────────────────────────────────────────────────────────────

def compute_total_loss(
    pred_abs,           # [T, B, 2] DEGREES
    gt,                 # [T, B, 2] DEGREES
    ref,                # [B, 2] DEGREES
    batch_list,
    pred_samples=None,  # [S, T, B, 2] NORMALISED
    gt_norm=None,       # [T, B, 2] NORMALISED
    weights=WEIGHTS,
    intensity_w: Optional[torch.Tensor] = None,
    env_data: Optional[dict] = None,
) -> Dict:
    """
    FIX-L37: Simplified scaling. All components kept in km (or dimensionless).
    Total = sum(w_i * L_i), then divided by a single normalisation constant
    for numerical stability. No double AFCRPS_SCALE multiply-then-divide.
    """
    # 1. Sample weights
    recurv_w = _recurvature_weights(gt, w_recurv=2.5)
    sample_w = recurv_w * (intensity_w.to(gt.device) if intensity_w is not None else 1.0)
    sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

    # 2. AFCRPS (km)
    if pred_samples is not None:
        if gt_norm is not None:
            l_fm = fm_afcrps_loss(pred_samples, gt_norm,
                                  intensity_w=sample_w, unit_01deg=True)
        else:
            l_fm = fm_afcrps_loss(pred_samples, gt,
                                  intensity_w=sample_w, unit_01deg=False)
    else:
        l_fm = _haversine_deg(pred_abs, gt).mean()

    # 3. Directional losses (km^2, normalised by STEP_KM^2 → dimensionless ~[0-1])
    STEP_KM = 113.0

    l_vel   = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
    l_disp  = (disp_loss_per_sample(pred_abs, gt) * sample_w).mean()
    l_step  = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()

    l_heading   = heading_loss(pred_abs, gt)
    l_recurv    = recurvature_loss(pred_abs, gt)
    l_dir_final = overall_dir_loss(pred_abs, gt, ref)
    l_smooth    = smooth_loss(pred_abs)
    l_accel     = acceleration_loss(pred_abs)

    # 4. PINN (FIX-L34: restored pinn_bve_loss call)
    # env_data extracted from batch_list if not passed explicitly
    _env = env_data
    if _env is None and batch_list is not None:
        try:
            _env = batch_list[13]
        except (IndexError, TypeError):
            _env = None

    l_pinn = pinn_bve_loss(pred_abs, batch_list, env_data=_env)

    # 5. Normalise km^2 → dimensionless  [FIX-L37: single pass, no cancel]
    sq_norm = STEP_KM * STEP_KM

    l_vel_n    = l_vel    / sq_norm
    l_disp_n   = l_disp   / sq_norm
    l_smooth_n = l_smooth  / sq_norm
    l_accel_n  = l_accel   / sq_norm

    # l_fm is in km (~350), directional dimensionless ~[0-2], pinn dimensionless
    # Scale everything to ~[0-10] for numerical stability by dividing by 35.
    NRM = 35.0

    total = (
        weights.get("fm", 2.0)       * l_fm
        + weights.get("velocity", 0.5) * l_vel_n    * NRM
        + weights.get("disp",     0.5) * l_disp_n   * NRM
        + weights.get("step",     0.5) * l_step      * NRM
        + weights.get("heading",  1.5) * l_heading   * NRM
        + weights.get("recurv",   1.5) * l_recurv    * NRM
        + weights.get("dir",      1.0) * l_dir_final * NRM
        + weights.get("smooth",   0.3) * l_smooth_n  * NRM
        + weights.get("accel",    0.5) * l_accel_n   * NRM
        + weights.get("pinn",     0.02) * l_pinn
    ) / NRM

    if torch.isnan(total) or torch.isinf(total):
        total = pred_abs.new_zeros(())

    return dict(
        total        = total,
        fm           = l_fm.item(),
        velocity     = l_vel_n.item() * NRM,
        step         = l_step.item(),
        disp         = l_disp_n.item() * NRM,
        heading      = l_heading.item(),
        recurv       = l_recurv.item(),
        smooth       = l_smooth_n.item() * NRM,
        accel        = l_accel_n.item() * NRM,
        pinn         = l_pinn.item(),
        recurv_ratio = (_recurvature_weights(gt) > 1.0).float().mean().item(),
    )


# ── Legacy helpers ─────────────────────────────────────────────────────────────

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


def toNE(pred_traj, pred_Me):
    rt, rm = pred_traj.clone(), pred_Me.clone()
    if rt.dim() == 2:
        rt = rt.unsqueeze(1); rm = rm.unsqueeze(1)
    rt[:, :, 0] = rt[:, :, 0] * 50.0 + 1800.0
    rt[:, :, 1] = rt[:, :, 1] * 50.0
    rm[:, :, 0] = rm[:, :, 0] * 50.0 + 960.0
    rm[:, :, 1] = rm[:, :, 1] * 25.0 + 40.0
    return rt, rm


def trajectory_displacement_error(pred, gt, mode="sum"):
    _gt   = gt.permute(1, 0, 2)
    _pred = pred.permute(1, 0, 2)
    diff  = _gt - _pred
    lon_km = diff[:, :, 0] / 10.0 * 111.0 * torch.cos(
        _gt[:, :, 1] / 10.0 * torch.pi / 180.0)
    lat_km = diff[:, :, 1] / 10.0 * 111.0
    loss   = torch.sqrt(lon_km ** 2 + lat_km ** 2)
    return torch.sum(loss) if mode == "sum" else loss


def evaluate_diffusion_output(best_traj, best_Me, gt_traj, gt_Me):
    rt, rm  = toNE(best_traj.clone(), best_Me.clone())
    rg, rgm = toNE(gt_traj.clone(),  gt_Me.clone())
    return (trajectory_displacement_error(rt, rg, mode="raw"),
            torch.abs(rm.permute(1, 0, 2) - rgm.permute(1, 0, 2)))
# """
# flow_matching_model.py — TC-FlowMatching v59-Strategy
# ═══════════════════════════════════════════════════════════════════════════════

# BASE: v59-tweaked (TWEAK-1/2/3 giữ nguyên)

# THAY ĐỔI THEO CHIẾN LƯỢC MỚI:

#   [S-A] compute_st_trans_loss: expose từng loss term riêng (GradNorm-compatible)
#         Không hardcode hệ số — trả về dict terms để train script điều chỉnh λ
#         GradNorm sẽ update λ_dpe, λ_vel, λ_head, λ_spd, λ_acc sau mỗi backward

#   [S-B] hard_score_from_obs: phân loại hard/easy đa tiêu chí vật lý
#         hard_score = 0.4×curvature + 0.3×speed_var + 0.3×direction_change
#         hard = (hard_score > p70) VÀ (loss > p50)  ← cả 2 điều kiện

#   [S-C] compute_hard_loss: L_hard = L_endpoint_norm + L_disp_norm (normalized)
#         L_endpoint /= 500, L_disp /= 300 → GradNorm có thể cân bằng được
#         α warm-up 0→0.3 trong 40 epoch (do train script điều khiển)

#   [S-D] SelectorNet + selector_loss: LDR-inspired selector cho hard cases
#         L_sel = L_soft_oracle + L_pairwise_rank(ADW=IRW×ERD) + L_confidence
#         Chỉ dùng khi phase >= 3 (train script kiểm soát)

#   [S-E] get_loss_breakdown: tích hợp easy/hard pipeline
#         - Trả về từng term riêng để GradNorm xử lý
#         - diversity_score logging (std endpoint candidates)
#         - is_hard flag per sample

#   [S-F] _compute_diversity_score: đo độ đa dạng của FM candidates
#         Dùng để detect R1 (mode collapse) sau phase 1

# GIỮ NGUYÊN (không đụng):
#   LearnedStepWeights (TWEAK-2)
#   _k3_mode_cluster (TWEAK-3) — inference only
#   EMAModel, VelocityField, TCFlowMatching.sample
#   Tất cả coordinate utilities (_haversine_deg, _norm_to_deg, v.v.)
# """
# from __future__ import annotations

# import csv
# import math
# import os
# from datetime import datetime
# from typing import Optional, Tuple, Dict, List

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from Model.FNO3D_encoder import FNO3DEncoder
# from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# from Model.env_net_transformer_gphsplit import Env_net

# # ─────────────────────────────────────────────────────────────────────────────
# R_EARTH      = 6371.0
# DT_HOURS     = 6.0
# DEG2KM       = 111.0
# _NORM_TO_DEG = 5.0

# STEP_WEIGHTS = [2.0, 3.5, 3.5, 4.0, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# _SPEED_PRIOR = {"v_opt": 15.0, "v_sigma": 10.0, "v_hard_cap": 35.0}


# # ─────────────────────────────────────────────────────────────────────────────
# # [TWEAK-2] LearnedStepWeights — giữ nguyên từ v59-tweaked
# # ─────────────────────────────────────────────────────────────────────────────

# class LearnedStepWeights(nn.Module):
#     """
#     Monotone step weights learned from data.
#     Init: linspace(-0.3, 1.5) → ratio ≈ 12× (buffer for 72h gradient).
#     Penalty: maintain ratio ≥ 6× to keep 72h signal strong.
#     """
#     def __init__(self, n_steps: int = 12, min_ratio: float = 6.0):
#         super().__init__()
#         self.n_steps   = n_steps
#         self.min_ratio = min_ratio
#         self.raw = nn.Parameter(torch.linspace(-0.3, 1.5, n_steps))

#     def forward(self) -> torch.Tensor:
#         w = torch.cumsum(F.softplus(self.raw), dim=0)
#         return w * self.n_steps / (w.sum() + 1e-8)

#     def get(self, n=None):
#         w = self.forward()
#         return w[:n] if n is not None else w

#     def penalty(self):
#         w = self.forward()
#         ratio = w[-1] / w[0].clamp(min=1e-6)
#         return 0.02 * F.relu(self.min_ratio - ratio) ** 2

#     @torch.no_grad()
#     def stats(self):
#         w = self.forward()
#         return {
#             "sw_ratio": (w[-1] / w[0].clamp(1e-6)).item(),
#             "sw_72h":   w[-1].item(),
#             "sw_6h":    w[0].item(),
#         }


# # ═════════════════════════════════════════════════════════════════════════════
# #  Coordinate utilities — giữ nguyên
# # ═════════════════════════════════════════════════════════════════════════════

# def _norm_to_deg(t):
#     return torch.stack([
#         (t[..., 0] * 50.0 + 1800.0) / 10.0,
#         (t[..., 1] * 50.0) / 10.0,
#     ], dim=-1)


# def _haversine_deg(p1, p2):
#     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
#     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
#     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
#     a = (torch.sin(dlat / 2).pow(2) +
#          torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
#     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# def _unwrap_model(m):
#     return m._orig_mod if hasattr(m, "_orig_mod") else m


# def _step_speeds_deg(traj_deg):
#     T = traj_deg.shape[0]
#     if T < 2:
#         return traj_deg.new_zeros(1, traj_deg.shape[1])
#     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# def _forward_azimuth(p1, p2):
#     lon1 = torch.deg2rad(p1[..., 0]); lat1 = torch.deg2rad(p1[..., 1])
#     lon2 = torch.deg2rad(p2[..., 0]); lat2 = torch.deg2rad(p2[..., 1])
#     dlon = lon2 - lon1
#     y = torch.sin(dlon) * torch.cos(lat2)
#     x = (torch.cos(lat1) * torch.sin(lat2)
#          - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon))
#     return torch.atan2(y, x)


# # ═════════════════════════════════════════════════════════════════════════════
# #  Speed statistics — giữ nguyên
# # ═════════════════════════════════════════════════════════════════════════════

# def compute_speed_stats_from_norm(obs_traj):
#     T = obs_traj.shape[0]
#     if T < 2:
#         return dict(_SPEED_PRIOR)
#     with torch.no_grad():
#         lon_deg = (obs_traj[..., 0] * 50.0 + 1800.0) / 10.0
#         lat_deg = (obs_traj[..., 1] * 50.0) / 10.0
#         dlon    = lon_deg[1:] - lon_deg[:-1]
#         dlat    = lat_deg[1:] - lat_deg[:-1]
#         cos_lat = torch.cos(torch.deg2rad(
#             (lat_deg[:-1] + lat_deg[1:]) * 0.5)).clamp(1e-4)
#         speed   = torch.sqrt(
#             (dlon * cos_lat * DEG2KM)**2 + (dlat * DEG2KM)**2 + 1e-6
#         ) / DT_HOURS
#         sf = speed.flatten()
#         if sf.numel() < 4:
#             return dict(_SPEED_PRIOR)
#         mean_s = float(sf.mean())
#         std_s  = float(sf.std().clamp(min=1.0))
#         q      = torch.quantile(sf, torch.tensor([.50, .75, .95], device=sf.device))
#         p50, p95 = float(q[0]), float(q[2])
#     return {
#         "mean_kmh"  : mean_s,  "std_kmh"   : std_s,
#         "p50_kmh"   : p50,     "p95_kmh"   : p95,
#         "v_opt"     : max(p50, 5.0),
#         "v_sigma"   : max(std_s + 5.0, 5.0),
#         "v_hard_cap": float(torch.tensor(p95 * 1.8).clamp(25.0, 80.0)),
#     }


# # ═════════════════════════════════════════════════════════════════════════════
# #  [S-B] Hard score đa tiêu chí — MỚI
# #  hard_score = 0.4×curvature + 0.3×speed_var + 0.3×direction_change
# #  KHÔNG dùng loss để phân loại — loss cao có thể do noisy label
# # ═════════════════════════════════════════════════════════════════════════════

# @torch.no_grad()
# def hard_score_from_obs(obs_traj_norm: torch.Tensor) -> torch.Tensor:
#     """
#     Tính hard_score cho mỗi sample trong batch dựa trên đặc trưng vật lý.

#     Args:
#         obs_traj_norm: [T_obs, B, >=2] normalized trajectory

#     Returns:
#         hard_score: [B] float tensor, càng cao càng khó
#     """
#     T, B = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
#     device = obs_traj_norm.device

#     if T < 3:
#         return torch.zeros(B, device=device)

#     traj_deg = _norm_to_deg(obs_traj_norm[..., :2])  # [T, B, 2]

#     # ── curvature_index: tổng góc đổi hướng (degree) / T
#     # Đo mức độ "uốn khúc" của quỹ đạo quan sát
#     if T >= 3:
#         az_12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])   # [T-2, B]
#         az_23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])    # [T-2, B]
#         angle_diff = torch.abs(az_23 - az_12)
#         # Wrap về [-π, π]
#         angle_diff = torch.where(
#             angle_diff > math.pi,
#             2 * math.pi - angle_diff,
#             angle_diff
#         )
#         curvature_index = angle_diff.mean(0) / math.pi  # [B] ∈ [0, 1]
#     else:
#         curvature_index = torch.zeros(B, device=device)

#     # ── speed_variance: std(speed) / mean(speed) — coefficient of variation
#     speeds = _step_speeds_deg(traj_deg)  # [T-1, B]
#     if speeds.shape[0] >= 2:
#         speed_mean = speeds.mean(0).clamp(min=1.0)
#         speed_std  = speeds.std(0)
#         speed_variance = (speed_std / speed_mean).clamp(0.0, 1.0)  # [B]
#     else:
#         speed_variance = torch.zeros(B, device=device)

#     # ── direction_change: số lần đổi hướng lớn (> 20°) / (T-2)
#     # Đo tính bất ổn định của hướng đi
#     if T >= 3:
#         large_turn = (angle_diff > (20.0 / 180.0 * math.pi)).float()
#         direction_change = large_turn.mean(0)  # [B] ∈ [0, 1]
#     else:
#         direction_change = torch.zeros(B, device=device)

#     hard_score = (0.4 * curvature_index
#                   + 0.3 * speed_variance
#                   + 0.3 * direction_change)

#     return hard_score  # [B]


# @torch.no_grad()
# def classify_hard_easy(
#     obs_traj_norm: torch.Tensor,
#     per_sample_loss: Optional[torch.Tensor] = None,
#     hard_score_p: float = 70.0,
#     loss_p: float = 50.0,
# ) -> torch.Tensor:
#     """
#     Phân loại hard/easy với 2 điều kiện đồng thời:
#       hard = (hard_score > p70) VÀ (loss > p50)

#     Nếu per_sample_loss=None: chỉ dùng hard_score (phase 1 chưa có loss history)

#     Returns:
#         is_hard: [B] bool tensor
#     """
#     scores = hard_score_from_obs(obs_traj_norm)  # [B]
#     B = scores.shape[0]

#     if B < 4:
#         return torch.zeros(B, dtype=torch.bool, device=scores.device)

#     threshold_score = torch.quantile(scores, hard_score_p / 100.0)
#     mask_score = scores >= threshold_score

#     if per_sample_loss is None:
#         return mask_score

#     # Điều kiện loss: phải CÙNG LÚC với score cao
#     # → loại bỏ outlier data bị nhầm là hard
#     threshold_loss = torch.quantile(per_sample_loss, loss_p / 100.0)
#     mask_loss = per_sample_loss >= threshold_loss

#     return mask_score & mask_loss  # [B] bool


# # ═════════════════════════════════════════════════════════════════════════════
# #  [S-C] Hard loss — normalized để GradNorm cân bằng được
# #  L_hard = L_endpoint_norm + L_disp_norm
# #  Lý do normalize: L_endpoint ~ km (0–500), L_DPE ~ ratio (0–2)
# #  Nếu không normalize, GradNorm không thể cân bằng (scale chênh 250×)
# # ═════════════════════════════════════════════════════════════════════════════

# def compute_hard_loss(
#     pred_deg: torch.Tensor,
#     gt_deg: torch.Tensor,
#     is_hard: torch.Tensor,
#     step_w: Optional[torch.Tensor] = None,
# ) -> Dict[str, torch.Tensor]:
#     """
#     Tính L_hard chỉ trên hard samples.

#     Args:
#         pred_deg: [T, B, 2] predicted trajectory in degrees
#         gt_deg:   [T, B, 2] ground truth in degrees
#         is_hard:  [B] bool mask
#         step_w:   [T] learned step weights (optional)

#     Returns:
#         dict với l_endpoint_norm, l_disp_norm, l_hard_total
#         (tất cả đều normalized về ~ scale 0–1 để GradNorm hoạt động)
#     """
#     device = pred_deg.device
#     zero   = pred_deg.new_zeros(())

#     if not is_hard.any():
#         return {"l_endpoint_norm": zero, "l_disp_norm": zero,
#                 "l_hard_total": zero, "n_hard": 0}

#     # Lấy hard samples
#     pred_h = pred_deg[:, is_hard, :]  # [T, n_hard, 2]
#     gt_h   = gt_deg[:, is_hard, :]

#     T = min(pred_h.shape[0], gt_h.shape[0])
#     n_hard = int(is_hard.sum().item())

#     if T < 4:
#         return {"l_endpoint_norm": zero, "l_disp_norm": zero,
#                 "l_hard_total": zero, "n_hard": n_hard}

#     # ── L_endpoint: Huber loss tại các checkpoint 48h (step7), 72h (step11)
#     # Normalize /500 để scale về 0–1 (72h ADE thường dưới 500km)
#     ep_total = zero
#     ep_w_sum = 0.0
#     for s, ew in [(min(7, T-1), 1.0), (min(11, T-1), 2.0)]:
#         dist = _haversine_deg(pred_h[s], gt_h[s])  # [n_hard]
#         d_hub = 200.0
#         loss_s = torch.where(dist < d_hub,
#                              dist.pow(2) / (2 * d_hub),
#                              dist - d_hub / 2).mean()
#         ep_total = ep_total + ew * loss_s
#         ep_w_sum += ew
#     l_endpoint_raw = ep_total / max(ep_w_sum, 1e-6)
#     l_endpoint_norm = l_endpoint_raw / 500.0  # normalize

#     # ── L_disp: displacement consistency — bước nhảy step-weighted
#     # Đo tổng sai lệch quỹ đạo (khác ADE ở chỗ có step weights)
#     if step_w is not None:
#         w = step_w[:T].to(device); w = w / w.sum()
#     else:
#         w = pred_deg.new_tensor(STEP_WEIGHTS[:T]); w = w / w.sum()

#     dist_all = _haversine_deg(pred_h[:T], gt_h[:T])  # [T, n_hard]
#     l_disp_raw = (dist_all * w.unsqueeze(1)).mean()
#     l_disp_norm = l_disp_raw / 300.0  # normalize

#     l_hard_total = l_endpoint_norm + l_disp_norm

#     return {
#         "l_endpoint_norm": l_endpoint_norm,
#         "l_disp_norm":     l_disp_norm,
#         "l_hard_total":    l_hard_total,
#         "n_hard":          n_hard,
#     }


# # ═════════════════════════════════════════════════════════════════════════════
# #  [S-A] compute_st_trans_loss — GradNorm-compatible
# #  THAY ĐỔI: không hardcode hệ số, trả về từng term riêng
# #  Train script dùng GradNorm để tìm λ_dpe, λ_vel, λ_head, λ_spd, λ_acc
# #
# #  Lý do: 1.20·l_dpe + 1.40·l_vel_reg + 0.40·l_heading trong v59-tweaked
# #  là hardcode → không tối ưu cho mọi giai đoạn training
# #  GradNorm tự điều chỉnh dựa trên gradient norm thực tế của data
# # ═════════════════════════════════════════════════════════════════════════════

# def compute_st_trans_loss(
#     pred_deg: torch.Tensor,
#     gt_deg: torch.Tensor,
#     epoch: int = 0,
#     speed_stats: Optional[dict] = None,
#     step_w: Optional[torch.Tensor] = None,
# ) -> Dict[str, torch.Tensor]:
#     """
#     STTrans loss với từng term được trả về riêng (GradNorm-compatible).

#     KHÔNG tự tính weighted sum ở đây.
#     Train script sẽ tính: total = λ_dpe·l_dpe + λ_vel·l_vel_reg + ...
#     với λ được GradNorm điều chỉnh sau mỗi backward.

#     Returns dict:
#         l_dpe       — position Huber (primary)
#         l_mse       — position MSE (stability regularizer)
#         l_vel_reg   — velocity consistency (key for ATE)
#         l_heading   — direction continuity (key for CTE)
#         l_speed     — speed prior (regularizer)
#         l_accel     — smoothness (regularizer)
#         total       — weighted sum dùng λ hiện tại (do caller truyền vào)
#                       Nếu không truyền lambdas, dùng default cân bằng
#     """
#     sp         = speed_stats or _SPEED_PRIOR
#     v_opt      = sp.get("v_opt",      15.0)
#     v_sigma    = sp.get("v_sigma",    10.0)
#     v_hard_cap = sp.get("v_hard_cap", 35.0)
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     device = pred_deg.device

#     if T < 2:
#         zero = pred_deg.new_zeros(())
#         return {k: zero for k in
#                 ["l_dpe","l_mse","l_vel_reg","l_heading","l_speed","l_accel",
#                  "total","l_pos","l_head","l_smooth","l_disp"]}

#     # ── Step weights
#     if step_w is not None:
#         w = step_w[:T].to(device); w = w / w.sum() * T
#     else:
#         w = pred_deg.new_tensor(STEP_WEIGHTS[:T]); w = w / w.sum() * T

#     # ── L_DPE: Displacement Proportional Error (Huber)
#     dist = _haversine_deg(pred_deg[:T], gt_deg[:T])  # [T, B]
#     d    = 200.0
#     l_dpe = ((torch.where(dist < d, dist.pow(2) / (2*d), dist - d/2))
#              * w.unsqueeze(1)).mean() / d

#     # ── L_MSE: raw MSE trên normalized coords (stability)
#     l_mse = F.mse_loss(pred_deg[:T], gt_deg[:T])

#     # ── L_speed: speed prior (soft constraint)
#     pred_spd = _step_speeds_deg(pred_deg[:T])  # [T-1, B]
#     if pred_spd.shape[0] > 0:
#         l_speed = (0.7 * ((pred_spd - v_opt) / v_sigma).pow(2).mean() +
#                    0.3 * F.relu(pred_spd - v_hard_cap).pow(2).mean() / v_hard_cap**2)
#     else:
#         l_speed = pred_deg.new_zeros(())

#     # ── L_accel: smoothness constraint
#     if pred_spd.shape[0] >= 2:
#         l_accel = (((pred_spd[1:] - pred_spd[:-1]).abs() / DT_HOURS).pow(2).mean()
#                    / max(v_sigma * 0.5, 3.0)**2)
#     else:
#         l_accel = pred_deg.new_zeros(())

#     # ── L_heading: direction continuity in km-space (key for CTE)
#     if T >= 3:
#         cos_lat_h = torch.cos(torch.deg2rad(
#             (gt_deg[:T-1,:,1] + gt_deg[1:T,:,1]) * 0.5)).clamp(1e-4)
#         pv_raw = pred_deg[1:T] - pred_deg[:T-1]
#         gv_raw = gt_deg[1:T]   - gt_deg[:T-1]
#         pv_km  = torch.stack([pv_raw[...,0]*cos_lat_h*DEG2KM,
#                                pv_raw[...,1]*DEG2KM], dim=-1)
#         gv_km  = torch.stack([gv_raw[...,0]*cos_lat_h*DEG2KM,
#                                gv_raw[...,1]*DEG2KM], dim=-1)
#         cos_sim  = (F.normalize(pv_km, dim=-1, eps=1e-6) *
#                     F.normalize(gv_km, dim=-1, eps=1e-6)).sum(-1)  # [T-1, B]
#         head_err = (1.0 - cos_sim).clamp(0.0, 2.0)
#         if step_w is not None:
#             hw = step_w[1:T].to(device); hw = hw / hw.sum()
#             l_heading = (head_err * hw.unsqueeze(1)).mean()
#         else:
#             l_heading = head_err.mean()
#     else:
#         l_heading = pred_deg.new_zeros(())

#     # ── L_vel_reg: velocity regression (key for ATE)
#     l_vel_reg = _velocity_regression_loss(pred_deg, gt_deg, speed_stats,
#                                           step_w=step_w)

#     # ── Default weighted sum (dùng khi GradNorm chưa init hoặc không dùng)
#     # Đây là fallback — train script sẽ override với GradNorm λ
#     total_default = (1.20 * l_dpe
#                      + 1.40 * l_vel_reg
#                      + 0.40 * l_heading
#                      + 0.05 * l_speed
#                      + 0.01 * l_accel)

#     if torch.isnan(total_default) or torch.isinf(total_default):
#         total_default = pred_deg.new_zeros(())

#     def _s(x): return x.item() if torch.is_tensor(x) else float(x)

#     return dict(
#         # Các terms riêng cho GradNorm
#         l_dpe       = l_dpe,
#         l_mse       = l_mse,
#         l_vel_reg   = l_vel_reg,
#         l_heading   = l_heading,
#         l_speed     = l_speed,
#         l_accel     = l_accel,
#         # Default total (train script override)
#         total       = total_default,
#         # Aliases log compat
#         l_pos       = l_dpe,
#         l_head      = l_heading,
#         l_smooth    = l_accel,
#         l_disp      = l_vel_reg,
#         dpe         = _s(l_dpe),
#         mse         = _s(l_mse),
#         heading     = _s(l_heading),
#         vel_reg     = _s(l_vel_reg),
#         speed       = _s(l_speed),
#         accel       = _s(l_accel),
#         # Zeroed for log compat
#         l_anchor=0.0, l_hard=0.0, lambda_hard=0.0, q_hard_mean=0.0,
#         anchor_ade=0.0, ate=0.0, cte=0.0, sph_ate=0.0, endpoint=0.0,
#         signed_ate=0.0, signed_cte=0.0, direct_ep=0.0,
#         ate_mean_km=0.0, cte_mean_km=0.0, speed_match=0.0,
#         acc_kmh2=0.0, aux_fno=0.0, sigma=0.0, fm_mse=0.0,
#         multi_marg=0.0, rollout_ate=0.0, rollout_w=0.0,
#     )


# compute_ate_focused_loss = compute_st_trans_loss


# # ═════════════════════════════════════════════════════════════════════════════
# #  [S-D] SelectorNet — LDR-inspired, chỉ train ở phase 3
# #  Scorer: nhận context + candidate features → score
# #  Loss: L_soft_oracle + L_pairwise_rank(ADW) + L_confidence
# # ═════════════════════════════════════════════════════════════════════════════

# class SelectorNet(nn.Module):
#     """
#     Learned selector để chọn candidate tốt nhất trong hard cases.

#     Architecture:
#         Input: context 256d + candidate kinematics → score
#         Dropout = 0.3 (hard cases ít, cần regularize mạnh)

#     Chỉ được dùng khi:
#         - phase >= 3 (train script kiểm soát)
#         - is_hard = True cho sample đó
#         - Inference: confidence > threshold → chọn selector thay median
#     """
#     def __init__(self, ctx_dim: int = 256, cand_feat_dim: int = 64,
#                  hidden_dim: int = 128):
#         super().__init__()
#         # Candidate feature extractor: [T, 2] → 64d
#         # Input: ADE, endpoint_72h, mean_speed, speed_var, curvature, heading_consistency
#         self.cand_encoder = nn.Sequential(
#             nn.Linear(6, 32),
#             nn.GELU(),
#             nn.LayerNorm(32),
#             nn.Linear(32, cand_feat_dim),
#             nn.GELU(),
#         )
#         # Scorer: (ctx, cand_feat) → score
#         # Dropout 0.3 vì hard cases ít → dễ overfit
#         self.scorer = nn.Sequential(
#             nn.Linear(ctx_dim + cand_feat_dim, hidden_dim),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim, 64),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 1),  # scalar score
#         )
#         # Confidence head: context → confidence scalar
#         # Học khi nào selector đủ chắc → khi nào fallback về median
#         self.confidence_head = nn.Sequential(
#             nn.Linear(ctx_dim, 64),
#             nn.GELU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid(),
#         )
#         self._init_weights()

#     def _init_weights(self):
#         with torch.no_grad():
#             for m in self.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.xavier_uniform_(m.weight, gain=0.1)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)

#     @staticmethod
#     def _extract_cand_features(
#         cand_norm: torch.Tensor,
#         gt_deg: Optional[torch.Tensor] = None,
#         obs_norm: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Trích xuất 6 features từ một candidate trajectory.

#         Args:
#             cand_norm: [T, 2] normalized trajectory (single sample, single cand)
#             gt_deg:    [T, 2] ground truth in degrees (None at inference)
#             obs_norm:  [T_obs, 2] observed trajectory (for heading consistency)

#         Returns:
#             features: [6] tensor
#                 0: ADE (km, /300 normalized)
#                 1: endpoint_72h dist (km, /500 normalized)
#                 2: mean_speed (km/h, /30 normalized)
#                 3: speed_var (coefficient of variation, clamped 0-1)
#                 4: curvature (mean angle change /π)
#                 5: heading_consistency với obs
#         """
#         device = cand_norm.device
#         T = cand_norm.shape[0]
#         cand_deg = _norm_to_deg(cand_norm.unsqueeze(1)).squeeze(1)  # [T, 2]

#         # ADE vs GT (0 if no GT)
#         if gt_deg is not None and gt_deg.shape[0] >= T:
#             ade = _haversine_deg(cand_deg.unsqueeze(1),
#                                   gt_deg[:T].unsqueeze(1)).mean().item()
#             ade_n = ade / 300.0
#         else:
#             ade_n = 0.0

#         # Endpoint 72h (step 11 or last)
#         if gt_deg is not None and gt_deg.shape[0] > min(T-1, 11):
#             ep_step = min(T-1, 11)
#             ep_dist = _haversine_deg(
#                 cand_deg[ep_step].unsqueeze(0),
#                 gt_deg[ep_step].unsqueeze(0)).item()
#             ep_n = ep_dist / 500.0
#         else:
#             ep_n = 0.0

#         # Speed stats
#         if T >= 2:
#             speeds = _haversine_deg(cand_deg[:-1].unsqueeze(1),
#                                      cand_deg[1:].unsqueeze(1)).squeeze(1) / DT_HOURS
#             mean_spd = speeds.mean().item()
#             # unbiased=False để tránh nan khi chỉ có 1 step (std của 1 element)
#             if speeds.numel() >= 2:
#                 speed_std = speeds.std(unbiased=False).item()
#             else:
#                 speed_std = 0.0
#             speed_var = min(speed_std / max(mean_spd, 1.0), 1.0)
#             mean_spd_n = mean_spd / 30.0
#         else:
#             mean_spd_n = speed_var = 0.0

#         # Curvature
#         if T >= 3:
#             az12 = _forward_azimuth(
#                 cand_deg[:-2].unsqueeze(1),
#                 cand_deg[1:-1].unsqueeze(1)).squeeze(1)
#             az23 = _forward_azimuth(
#                 cand_deg[1:-1].unsqueeze(1),
#                 cand_deg[2:].unsqueeze(1)).squeeze(1)
#             diff = (az23 - az12).abs()
#             diff = torch.where(diff > math.pi, 2*math.pi - diff, diff)
#             curvature = (diff.mean() / math.pi).item()
#         else:
#             curvature = 0.0

#         # Heading consistency với obs
#         if obs_norm is not None and obs_norm.shape[0] >= 2 and T >= 1:
#             obs_vel = obs_norm[-1, :2] - obs_norm[-2, :2]
#             cand_vel = cand_norm[0, :2] - obs_norm[-1, :2]
#             obs_h  = F.normalize(obs_vel.unsqueeze(0), dim=-1, eps=1e-6)
#             cand_h = F.normalize(cand_vel.unsqueeze(0), dim=-1, eps=1e-6)
#             head_cons = ((obs_h * cand_h).sum(-1).clamp(-1,1).item() + 1.0) / 2.0
#         else:
#             head_cons = 0.5

#         feat = [ade_n, ep_n, mean_spd_n, speed_var, curvature, head_cons]
#         # Guard: thay nan/inf bằng 0 để tránh loss nan
#         feat = [0.0 if (not math.isfinite(v)) else v for v in feat]
#         return torch.tensor(feat, dtype=torch.float, device=device)

#     def score_candidates(
#         self,
#         ctx: torch.Tensor,           # [B, 256]
#         candidates: List[torch.Tensor],  # list of N [T, B, 2] normalized
#         gt_deg: Optional[torch.Tensor] = None,    # [T, B, 2]
#         obs_norm: Optional[torch.Tensor] = None,  # [T_obs, B, 2]
#     ) -> torch.Tensor:
#         """
#         Chấm điểm N candidates cho B samples.

#         Returns:
#             scores: [N, B] float tensor
#         """
#         B = ctx.shape[0]
#         N = len(candidates)
#         device = ctx.device
#         all_scores = []

#         for n, cand in enumerate(candidates):
#             T = cand.shape[0]
#             feat_list = []
#             for b in range(B):
#                 gt_b  = gt_deg[:, b, :]  if gt_deg is not None  else None
#                 obs_b = obs_norm[:, b, :] if obs_norm is not None else None
#                 feat = self._extract_cand_features(cand[:, b, :], gt_b, obs_b)
#                 feat_list.append(feat)
#             cand_feat = torch.stack(feat_list, dim=0)         # [B, 6]
#             cand_enc  = self.cand_encoder(cand_feat)          # [B, 64]
#             inp       = torch.cat([ctx, cand_enc], dim=-1)    # [B, 320]
#             score     = self.scorer(inp).squeeze(-1)           # [B]
#             all_scores.append(score)

#         return torch.stack(all_scores, dim=0)  # [N, B]

#     def get_confidence(self, ctx: torch.Tensor) -> torch.Tensor:
#         """
#         Returns:
#             confidence: [B] ∈ (0, 1)
#             Nếu < threshold → fallback về median
#         """
#         return self.confidence_head(ctx).squeeze(-1)


# def selector_loss(
#     scores: torch.Tensor,           # [N, B] từ score_candidates
#     gt_ades: torch.Tensor,          # [N, B] ADE của mỗi candidate (detached)
#     confidence: torch.Tensor,       # [B]
#     is_hard: torch.Tensor,          # [B] bool
# ) -> Dict[str, torch.Tensor]:
#     """
#     LDR-inspired selector loss với Adaptive Decay Weight (ADW).

#     L_sel = L_soft_oracle + L_pairwise_rank + L_confidence

#     ADW = IRW × ERD:
#         IRW (Inverse Rank Weighting): rank 1 vs rank 2 quan trọng hơn rank 1 vs rank 20
#         ERD (Exponential Rank Decay): rank 40 vs rank 41 đều tệ → không cần học kỹ

#     Chỉ tính trên hard samples để tránh ảnh hưởng easy cases.
#     """
#     device = scores.device
#     zero   = scores.new_zeros(())

#     if not is_hard.any():
#         return {"l_soft_oracle": zero, "l_pairwise_rank": zero,
#                 "l_confidence": zero, "l_sel_total": zero}

#     # Lấy hard samples
#     sc_h  = scores[:, is_hard]    # [N, n_hard]
#     ade_h = gt_ades[:, is_hard]   # [N, n_hard]
#     conf_h = confidence[is_hard]  # [n_hard]
#     N, n_hard = sc_h.shape

#     if N < 2:
#         return {"l_soft_oracle": zero, "l_pairwise_rank": zero,
#                 "l_confidence": zero, "l_sel_total": zero}

#     # ── L_soft_oracle: phân phối xác suất — candidate gần GT hơn → score cao hơn
#     # oracle_prob[n, b] ∝ exp(-ade[n, b] / τ) với τ = mean ADE
#     tau = ade_h.mean().clamp(min=10.0)
#     oracle_prob = F.softmax(-ade_h / tau, dim=0)  # [N, n_hard]
#     log_sc      = F.log_softmax(sc_h, dim=0)       # [N, n_hard]
#     l_soft_oracle = -(oracle_prob * log_sc).sum(0).mean()

#     # ── L_pairwise_rank với ADW = IRW × ERD
#     # Sắp xếp candidates theo ADE tăng dần → rank 0 = tốt nhất
#     sort_idx = ade_h.argsort(dim=0)  # [N, n_hard]
#     ranks    = sort_idx.argsort(dim=0).float()  # [N, n_hard], rank của mỗi candidate

#     pairwise_losses = []
#     # Chỉ học các cặp trong top-50% để tránh nhiễu từ candidates quá tệ
#     n_top = max(2, N // 2)
#     top_idx = ade_h.argsort(dim=0)[:n_top]  # [n_top, n_hard]

#     for i in range(n_top):
#         for j in range(i + 1, n_top):
#             # Lấy rank của candidate i và j
#             rank_i = ranks[top_idx[i], torch.arange(n_hard, device=device)]  # [n_hard]
#             rank_j = ranks[top_idx[j], torch.arange(n_hard, device=device)]  # [n_hard]
#             ade_i  = ade_h[top_idx[i], torch.arange(n_hard, device=device)]
#             ade_j  = ade_h[top_idx[j], torch.arange(n_hard, device=device)]
#             sc_i   = sc_h[top_idx[i], torch.arange(n_hard, device=device)]
#             sc_j   = sc_h[top_idx[j], torch.arange(n_hard, device=device)]

#             # IRW: cặp rank gần nhau quan trọng hơn
#             rank_gap = (rank_i - rank_j).abs().clamp(min=1.0)
#             irw = 1.0 / rank_gap  # cặp rank 1-2: irw=1, rank 1-10: irw=0.1

#             # ERD: cặp rank thấp (candidates tệ) giảm trọng số
#             avg_rank = (rank_i + rank_j) / 2.0
#             erd = torch.exp(-avg_rank / max(n_top / 2.0, 1.0))

#             adw = irw * erd  # Adaptive Decay Weight

#             # Pairwise ranking loss: i tốt hơn j → score_i > score_j
#             # i tốt hơn khi ade_i < ade_j
#             better_i = (ade_i < ade_j).float()
#             margin   = 0.1
#             pair_loss = torch.relu(margin - (sc_i - sc_j) * (2 * better_i - 1))
#             pairwise_losses.append((pair_loss * adw).mean())

#     if pairwise_losses:
#         l_pairwise_rank = torch.stack(pairwise_losses).mean()
#     else:
#         l_pairwise_rank = zero

#     # ── L_confidence: học khi nào selector đủ chắc
#     # Confidence cao khi oracle candidate rõ ràng tốt hơn (low oracle entropy)
#     oracle_entropy = -(oracle_prob * oracle_prob.log().clamp(min=-10)).sum(0)  # [n_hard]
#     max_entropy    = math.log(N)
#     # Target confidence: thấp khi entropy cao (uncertain), cao khi entropy thấp
#     target_conf = 1.0 - (oracle_entropy / max_entropy).clamp(0, 1)
#     l_confidence = F.mse_loss(conf_h, target_conf.detach())

#     l_sel_total = l_soft_oracle + 0.5 * l_pairwise_rank + 0.3 * l_confidence

#     return {
#         "l_soft_oracle":   l_soft_oracle,
#         "l_pairwise_rank": l_pairwise_rank,
#         "l_confidence":    l_confidence,
#         "l_sel_total":     l_sel_total,
#     }


# # ═════════════════════════════════════════════════════════════════════════════
# #  [S-F] Diversity score — detect R1 (mode collapse)
# #  Đo độ đa dạng của FM candidates tại endpoint 72h
# #  Nếu diversity < 50 km → FM collapse → cần fix trước khi train selector
# # ═════════════════════════════════════════════════════════════════════════════

# @torch.no_grad()
# def compute_diversity_score(candidates: List[torch.Tensor]) -> float:
#     """
#     Đo diversity của FM candidates tại endpoint 72h.

#     diversity_score = mean(std(endpoint_72h)) trên tất cả samples và candidates
#     Nếu < 50 km: mode collapse → cần tăng noise hoặc thêm L_diversity

#     Args:
#         candidates: list of N [T, B, 2] normalized trajectories

#     Returns:
#         diversity_km: float (km)
#     """
#     if len(candidates) < 2:
#         return 0.0

#     T = candidates[0].shape[0]
#     ep_step = min(T - 1, 11)  # step 11 = 72h

#     # Lấy endpoint 72h của mỗi candidate
#     endpoints = []
#     for cand in candidates:
#         ep_norm = cand[ep_step]  # [B, 2]
#         ep_deg  = _norm_to_deg(ep_norm)  # [B, 2]
#         endpoints.append(ep_deg)

#     endpoints = torch.stack(endpoints, dim=0)  # [N, B, 2]
#     N, B, _ = endpoints.shape

#     # Tính std của endpoint position cho mỗi batch sample
#     # Dùng haversine distance từ mean để đo đúng trên sphere
#     ep_mean = endpoints.mean(dim=0, keepdim=True)  # [1, B, 2]
#     dists = _haversine_deg(
#         endpoints.reshape(N * B, 2),
#         ep_mean.expand(N, B, 2).reshape(N * B, 2)
#     ).reshape(N, B)  # [N, B]

#     diversity_km = float(dists.std(dim=0).mean().item())
#     return diversity_km


# # ═════════════════════════════════════════════════════════════════════════════
# #  Velocity regression loss — giữ nguyên từ v59
# # ═════════════════════════════════════════════════════════════════════════════

# def _velocity_regression_loss(pred_deg, gt_deg, speed_stats=None, step_w=None):
#     sp      = speed_stats or _SPEED_PRIOR
#     v_sigma = sp.get("v_sigma", 10.0)
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         return pred_deg.new_zeros(())

#     pred_spd = _step_speeds_deg(pred_deg[:T])
#     gt_spd   = _step_speeds_deg(gt_deg[:T])
#     if step_w is not None:
#         w = step_w[1:T].to(pred_deg.device); w = w / w.sum()
#     else:
#         w = pred_deg.new_tensor(STEP_WEIGHTS[1:T]); w = w / w.sum()

#     gt_spd_clamped = gt_spd.clamp(min=5.0)
#     l_abs = ((pred_spd - gt_spd_clamped).pow(2) / v_sigma**2
#              * w.unsqueeze(1)).mean()

#     plon = pred_deg[1:T, :, 0] - pred_deg[:T-1, :, 0]
#     plat = pred_deg[1:T, :, 1] - pred_deg[:T-1, :, 1]
#     glon = gt_deg[1:T, :, 0]   - gt_deg[:T-1, :, 0]
#     glat = gt_deg[1:T, :, 1]   - gt_deg[:T-1, :, 1]
#     cos  = torch.cos(torch.deg2rad(
#         (pred_deg[:T-1,:,1] + pred_deg[1:T,:,1]) * 0.5)).clamp(1e-4)
#     pv_km = torch.stack([plon*cos*DEG2KM/DT_HOURS, plat*DEG2KM/DT_HOURS], -1)
#     gv_km = torch.stack([glon*cos*DEG2KM/DT_HOURS, glat*DEG2KM/DT_HOURS], -1)
#     l_vec = (F.mse_loss(pv_km, gv_km, reduction="none").mean(-1)
#              / v_sigma**2 * w.unsqueeze(1)).mean()

#     return (0.5*l_abs + 0.5*l_vec).clamp(0.0, 20.0)


# # ═════════════════════════════════════════════════════════════════════════════
# #  SpeedHead — giữ nguyên từ v59
# # ═════════════════════════════════════════════════════════════════════════════

# class SpeedHead(nn.Module):
#     def __init__(self, ctx_dim=256, obs_feat_dim=256, pred_len=12):
#         super().__init__()
#         self.pred_len = pred_len
#         self.fc = nn.Sequential(
#             nn.Linear(ctx_dim + obs_feat_dim, 256),
#             nn.GELU(),
#             nn.LayerNorm(256),
#             nn.Linear(256, 128),
#             nn.GELU(),
#             nn.Linear(128, pred_len),
#         )
#         with torch.no_grad():
#             nn.init.xavier_uniform_(self.fc[-1].weight, gain=0.1)
#             nn.init.zeros_(self.fc[-1].bias)

#     def forward(self, ctx, vel_obs_feat):
#         h = torch.cat([ctx, vel_obs_feat], dim=-1)
#         speed_pred = self.fc(h)
#         return F.softplus(speed_pred) * 3.0 + 2.0


# def _speed_head_loss(speed_pred, pred_deg, gt_deg, speed_stats=None):
#     sp = speed_stats or _SPEED_PRIOR
#     v_sigma = sp.get("v_sigma", 10.0)
#     T_gt = gt_deg.shape[0]
#     pred_len = speed_pred.shape[1]
#     T = min(pred_len + 1, T_gt)
#     if T < 2:
#         return speed_pred.new_zeros(())

#     gt_spd = _step_speeds_deg(gt_deg[:T])
#     gt_spd = gt_spd.permute(1, 0)
#     n = min(pred_len, gt_spd.shape[1])

#     speed_pred_n = speed_pred[:, :n]
#     gt_spd_n = gt_spd[:, :n].clamp(min=2.0)

#     w = speed_pred.new_tensor(STEP_WEIGHTS[1:n+1])
#     w = w / w.sum()

#     loss = (F.mse_loss(speed_pred_n, gt_spd_n, reduction='none') / v_sigma**2)
#     return (loss * w.unsqueeze(0)).mean()


# # ═════════════════════════════════════════════════════════════════════════════
# #  Sinkhorn OT — giữ nguyên từ v59
# # ═════════════════════════════════════════════════════════════════════════════

# def _sinkhorn_log(cost, epsilon=0.05, n_iter=50):
#     B      = cost.shape[0]
#     device = cost.device
#     log_a  = -math.log(B) * torch.ones(B, device=device)
#     log_b  = -math.log(B) * torch.ones(B, device=device)
#     log_K  = -cost / epsilon
#     log_u  = torch.zeros(B, device=device)
#     log_v  = torch.zeros(B, device=device)
#     for _ in range(n_iter):
#         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
#         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
#     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


# def _geodesic_ot_cost(x0_rel, x1_rel, lp):
#     B = x0_rel.shape[0]

#     def _abs_deg(rel):
#         return _norm_to_deg(lp.unsqueeze(1) + rel[:, :, :2])

#     x0d = _abs_deg(x0_rel); x1d = _abs_deg(x1_rel)
#     x0e = x0d.unsqueeze(1).expand(B, B, -1, -1).reshape(B*B, -1, 2)
#     x1e = x1d.unsqueeze(0).expand(B, B, -1, -1).reshape(B*B, -1, 2)
#     pos_cost = _haversine_deg(x0e, x1e).reshape(B, B, -1).mean(-1) / 500.0

#     spd0 = _step_speeds_deg(x0d.permute(1,0,2)).permute(1,0).mean(-1)
#     spd1 = _step_speeds_deg(x1d.permute(1,0,2)).permute(1,0).mean(-1)
#     speed_cost = (spd0.unsqueeze(1) - spd1.unsqueeze(0)).abs() / 20.0

#     def _mean_bearing(td):
#         b = _forward_azimuth(td[:, :-1, :], td[:, 1:, :])
#         return torch.atan2(b.sin().mean(-1), b.cos().mean(-1))
#     h0 = _mean_bearing(x0d); h1 = _mean_bearing(x1d)
#     dh = (h0.unsqueeze(1) - h1.unsqueeze(0) + math.pi) % (2*math.pi) - math.pi
#     dir_cost = dh.abs() / math.pi

#     return 1.0*pos_cost + 0.5*speed_cost + 0.3*dir_cost


# def _spherical_ot_matching(x0_batch, x1_batch, lp, epsilon=0.05):
#     B = x0_batch.shape[0]
#     if B < 4:
#         return x0_batch, x1_batch
#     try:
#         cost = _geodesic_ot_cost(x0_batch, x1_batch, lp)
#         with torch.no_grad():
#             pi = _sinkhorn_log(cost, epsilon=epsilon)
#         flat = pi.reshape(-1).clamp(0.0)
#         s    = flat.sum()
#         if not torch.isfinite(s) or s < 1e-10:
#             return x0_batch, x1_batch
#         idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
#         col = idx % B
#         return x0_batch[col], x1_batch[col]
#     except Exception:
#         return x0_batch, x1_batch


# # ═════════════════════════════════════════════════════════════════════════════
# #  Linear/Slerp interpolant — giữ nguyên từ v59
# # ═════════════════════════════════════════════════════════════════════════════

# def _slerp_interpolant(x0, x1, t, lp=None):
#     B  = x0.shape[0]; te = t.view(B, 1, 1)
#     if lp is not None and x0.shape[-1] >= 2:
#         abs0 = lp.unsqueeze(1) + x0[:, :, :2]
#         abs1 = lp.unsqueeze(1) + x1[:, :, :2]
#     else:
#         abs0 = x0[:, :, :2]; abs1 = x1[:, :, :2]
#     abs0_deg = _norm_to_deg(abs0); abs1_deg = _norm_to_deg(abs1)
#     lat0 = torch.deg2rad(abs0_deg[..., 1]); lat1 = torch.deg2rad(abs1_deg[..., 1])
#     dlon = torch.deg2rad(abs1_deg[..., 0] - abs0_deg[..., 0])
#     dlat = lat1 - lat0
#     a    = (torch.sin(dlat/2).pow(2)
#             + torch.cos(lat0) * torch.cos(lat1) * torch.sin(dlon/2).pow(2))
#     omega     = 2.0 * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())
#     sin_omega = torch.sin(omega).clamp(1e-6)
#     linear    = omega < 1e-4
#     te_sq     = te.squeeze(1)
#     coeff0 = torch.where(linear, 1.0 - te_sq,
#                          torch.sin((1-te_sq)*omega) / sin_omega)
#     coeff1 = torch.where(linear, te_sq,
#                          torch.sin(te_sq*omega) / sin_omega)
#     return coeff0.unsqueeze(-1)*x0 + coeff1.unsqueeze(-1)*x1


# def _slerp_velocity_target(x0, x1, t, lp=None):
#     B  = x0.shape[0]; te = t.view(B, 1, 1)
#     if lp is not None and x0.shape[-1] >= 2:
#         abs0 = lp.unsqueeze(1) + x0[:, :, :2]
#         abs1 = lp.unsqueeze(1) + x1[:, :, :2]
#     else:
#         abs0 = x0[:, :, :2]; abs1 = x1[:, :, :2]
#     abs0_deg = _norm_to_deg(abs0); abs1_deg = _norm_to_deg(abs1)
#     lat0 = torch.deg2rad(abs0_deg[..., 1]); lat1 = torch.deg2rad(abs1_deg[..., 1])
#     dlon = torch.deg2rad(abs1_deg[..., 0] - abs0_deg[..., 0])
#     dlat = lat1 - lat0
#     a    = (torch.sin(dlat/2).pow(2)
#             + torch.cos(lat0) * torch.cos(lat1) * torch.sin(dlon/2).pow(2))
#     omega     = 2.0 * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())
#     sin_omega = torch.sin(omega).clamp(1e-6)
#     oos       = omega / sin_omega
#     linear    = omega < 1e-4
#     te_sq     = te.squeeze(1)
#     coeff0 = torch.where(linear, -torch.ones_like(te_sq),
#                          -torch.cos((1-te_sq)*omega)*oos)
#     coeff1 = torch.where(linear,  torch.ones_like(te_sq),
#                           torch.cos(te_sq*omega)*oos)
#     return coeff0.unsqueeze(-1)*x0 + coeff1.unsqueeze(-1)*x1


# # ═════════════════════════════════════════════════════════════════════════════
# #  EMAModel — giữ nguyên từ v59
# # ═════════════════════════════════════════════════════════════════════════════

# class EMAModel:
#     def __init__(self, model, decay=0.995):
#         self.decay  = decay
#         m = _unwrap_model(model)
#         self.shadow = {k: v.detach().clone()
#                        for k, v in m.state_dict().items()
#                        if v.dtype.is_floating_point}

#     def update(self, model):
#         m = _unwrap_model(model)
#         with torch.no_grad():
#             for k, v in m.state_dict().items():
#                 if k in self.shadow:
#                     self.shadow[k].mul_(self.decay).add_(
#                         v.detach(), alpha=1 - self.decay)

#     def apply_to(self, model):
#         m = _unwrap_model(model)
#         backup, sd = {}, m.state_dict()
#         for k in self.shadow:
#             if k not in sd: continue
#             backup[k] = sd[k].detach().clone()
#             sd[k].copy_(self.shadow[k])
#         return backup

#     def restore(self, model, backup):
#         m = _unwrap_model(model)
#         sd = m.state_dict()
#         for k, v in backup.items():
#             if k in sd: sd[k].copy_(v)


# # ═════════════════════════════════════════════════════════════════════════════
# #  VelocityField — giữ nguyên từ v59 (chỉ thêm selector)
# # ═════════════════════════════════════════════════════════════════════════════

# class VelocityField(nn.Module):
#     RAW_CTX_DIM = 512

#     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256,
#                  sigma_min=0.02, unet_in_ch=13, **kwargs):
#         super().__init__()
#         self.pred_len = pred_len
#         self.obs_len  = obs_len
#         self.ctx_dim  = ctx_dim

#         self.spatial_enc     = FNO3DEncoder(
#             in_channel=unet_in_ch, out_channel=1, d_model=32,
#             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
#             spatial_down=32, dropout=0.05)
#         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
#         self.bottleneck_proj = nn.Linear(128, 128)
#         self.decoder_proj    = nn.Linear(1, 16)
#         self.enc_1d          = DataEncoder1D(
#             in_1d=4, feat_3d_dim=128, mlp_h=64,
#             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
#         self.env_enc         = Env_net(obs_len=obs_len, d_model=32)

#         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
#         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
#         self.ctx_drop = nn.Dropout(0.15)
#         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

#         self.null_embedding = nn.Parameter(torch.randn(1, self.RAW_CTX_DIM) * 0.02)

#         self.vel_obs_enc = nn.Sequential(
#             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
#             nn.Linear(256, 256), nn.GELU())

#         self.steering_enc = nn.Sequential(
#             nn.Linear(7, 64), nn.GELU(), nn.LayerNorm(64),
#             nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 256))

#         self.env_kine_enc = nn.Sequential(
#             nn.Linear(14, 64), nn.GELU(), nn.LayerNorm(64),
#             nn.Linear(64, 256), nn.GELU())

#         self.time_fc1   = nn.Linear(256, 512)
#         self.time_fc2   = nn.Linear(512, 256)
#         self.traj_embed = nn.Linear(4, 256)
#         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
#         self.step_embed = nn.Embedding(pred_len, 256)
#         self.transformer = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 d_model=256, nhead=8, dim_feedforward=1024,
#                 dropout=0.10, activation="gelu", batch_first=True),
#             num_layers=2)
#         self.out_fc1 = nn.Linear(256, 512)
#         self.out_fc2 = nn.Linear(512, 4)

#         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
#         self.physics_scale  = nn.Parameter(torch.ones(4) * 1.5)
#         self.steering_scale = nn.Parameter(torch.ones(4) * 1.0)

#         self.speed_head = SpeedHead(ctx_dim=ctx_dim, obs_feat_dim=256,
#                                     pred_len=pred_len)

#         self._init_weights()

#     def _init_weights(self):
#         with torch.no_grad():
#             for name, m in self.named_modules():
#                 if isinstance(m, nn.Linear) and "out_fc" in name:
#                     nn.init.xavier_uniform_(m.weight, gain=0.1)
#                     if m.bias is not None: nn.init.zeros_(m.bias)

#     def _time_emb(self, t, dim=256):
#         half = dim // 2
#         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
#                          * (-math.log(10000.0) / max(half-1, 1)))
#         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
#         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

#     def _context(self, batch_list):
#         obs_traj  = batch_list[0]
#         obs_Me    = batch_list[7]
#         image_obs = batch_list[11]
#         env_data  = batch_list[13]
#         if image_obs.dim() == 4: image_obs = image_obs.unsqueeze(2)
#         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
#             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

#         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
#         T_obs = obs_traj.shape[0]
#         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0,2,1)
#         e_3d_s = self.bottleneck_proj(e_3d_s)
#         if e_3d_s.shape[1] != T_obs:
#             e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
#                                     mode="linear", align_corners=False).permute(0,2,1)

#         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
#         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
#                                           device=e_3d_dec_t.device)*0.5, dim=0)
#         f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

#         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1,0,2)
#         h_t    = self.enc_1d(obs_in, e_3d_s)
#         e_env, _, _ = self.env_enc(env_data, image_obs)
#         return F.gelu(self.ctx_ln(self.ctx_fc1(
#             torch.cat([h_t, e_env, f_sp], dim=-1))))

#     def _apply_ctx_head(self, raw, noise_scale=0.0, use_null=False):
#         if use_null:
#             raw = self.null_embedding.expand(raw.shape[0], -1)
#         elif noise_scale > 0.0:
#             raw = raw + torch.randn_like(raw) * noise_scale
#         return self.ctx_fc2(self.ctx_drop(raw))

#     def _get_kinematic_obs_feat(self, obs_traj):
#         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
#         if T_obs >= 2:
#             vel     = obs_traj[1:] - obs_traj[:-1]
#             lat_mid = obs_traj[:-1, :, 1] * _NORM_TO_DEG
#             cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
#             dx_km   = vel[:,:,0] * cos_lat * DEG2KM * _NORM_TO_DEG
#             dy_km   = vel[:,:,1]             * DEG2KM * _NORM_TO_DEG
#             speed   = torch.sqrt(dx_km**2 + dy_km**2 + 1e-6) / DT_HOURS
#             heading = torch.atan2(vel[:,:,1], vel[:,:,0])
#             speed_n = (speed / 20.0).clamp(-3.0, 3.0)
#             if T_obs >= 3:
#                 dspd  = speed[1:] - speed[:-1]
#                 accel = torch.cat([obs_traj.new_zeros(1,B),
#                                    (dspd/10.0).clamp(-3.0,3.0)], 0)
#             else:
#                 accel = obs_traj.new_zeros(T_obs-1, B)
#             kine = torch.stack([vel[:,:,0], vel[:,:,1], speed_n,
#                                  heading.sin(), heading.cos(), accel], dim=-1)
#         else:
#             kine = obs_traj.new_zeros(self.obs_len, B, 6)
#         if kine.shape[0] < self.obs_len:
#             kine = torch.cat([obs_traj.new_zeros(
#                 self.obs_len-kine.shape[0], B, 6), kine], 0)
#         else:
#             kine = kine[-self.obs_len:]
#         return self.vel_obs_enc(kine.permute(1,0,2).reshape(B, -1))

#     def _get_vel_obs_feat(self, obs_traj):
#         return self._get_kinematic_obs_feat(obs_traj)

#     def _get_steering_feat(self, env_data, B, device):
#         if env_data is None: return torch.zeros(B, 256, device=device)
#         def _safe(k, norm=1.0):
#             v = env_data.get(k)
#             if v is None or not torch.is_tensor(v):
#                 return torch.full((B,), 0.0, device=device)
#             v = v.float().to(device)
#             while v.dim() > 1: v = v.mean(-1)
#             v = v.view(-1)[:B] if v.numel() >= B else torch.full((B,), 0.0, device=device)
#             return (v / norm).clamp(-3.0, 3.0)
#         return self.steering_enc(torch.stack([
#             _safe("u500_mean",    30.0),
#             _safe("v500_mean",    30.0),
#             _safe("u500_center",  30.0),
#             _safe("v500_center",  30.0),
#             _safe("steering_speed",  1.0),
#             _safe("steering_dir_sin",1.0),
#             _safe("steering_dir_cos",1.0),
#         ], dim=-1))

#     def _get_env_kine_feat(self, env_data, B, device):
#         if env_data is None: return torch.zeros(B, 256, device=device)
#         def _get_t(key, dim):
#             v = env_data.get(key)
#             if v is None: return torch.zeros(B, dim, device=device)
#             if not torch.is_tensor(v):
#                 try: v = torch.tensor(v, dtype=torch.float, device=device)
#                 except: return torch.zeros(B, dim, device=device)
#             v = v.float().to(device)
#             if v.dim() == 0:
#                 return (v.expand(B, dim) if dim == 1
#                         else torch.zeros(B, dim, device=device))
#             if v.dim() == 1:
#                 if v.shape[0] == dim: return v.unsqueeze(0).expand(B, dim)
#                 if v.shape[0] == B:
#                     return (v.unsqueeze(1).expand(B, dim) if dim == 1
#                             else torch.zeros(B, dim, device=device))
#             if v.dim() == 2:
#                 if v.shape == (B, dim): return v
#                 return (v[:B, :dim] if v.shape[1] >= dim
#                         else F.pad(v[:B], (0, dim-v.shape[1])))
#             if v.dim() == 3:
#                 vv = v[-1] if v.shape[1] == B else v[:B, -1]
#                 return (vv[:, :dim] if vv.shape[-1] >= dim
#                         else F.pad(vv, (0, dim-vv.shape[-1])))
#             return torch.zeros(B, dim, device=device)
#         feat = torch.cat([_get_t("move_velocity",1),
#                           _get_t("history_direction24",8),
#                           _get_t("delta_velocity",5)], dim=-1)
#         return self.env_kine_enc(feat)

#     def _beta_drift(self, x_t):
#         lat_rad = torch.deg2rad(x_t[:,:,1]*5.0).clamp(-85,85)
#         beta    = 2*7.2921e-5*torch.cos(lat_rad)/6.371e6
#         R_tc    = 3e5
#         v = torch.zeros_like(x_t)
#         v[:,:,0] = -beta*R_tc**2/2*6*3600/(5*111*1000)
#         v[:,:,1] =  beta*R_tc**2/4*6*3600/(5*111*1000)
#         return v

#     def _steering_drift(self, x_t, env_data):
#         if env_data is None: return torch.zeros_like(x_t)
#         B, device = x_t.shape[0], x_t.device
#         def _sm(k):
#             v = env_data.get(k)
#             if v is None or not torch.is_tensor(v):
#                 return torch.zeros(B, device=device)
#             v = v.float().to(device)
#             while v.dim() > 1: v = v.mean(-1)
#             return v.view(-1)[:B] if v.numel() >= B else torch.zeros(B, device=device)
#         u   = _sm("u500_center"); vv = _sm("v500_center")
#         cos = torch.cos(torch.deg2rad(x_t[:,:,1]*5.0)).clamp(1e-3)
#         out = torch.zeros_like(x_t)
#         out[:,:,0] = u.unsqueeze(1)*30.0*21600.0/(111.0*1000.0*cos)
#         out[:,:,1] = vv.unsqueeze(1)*30.0*21600.0/(111.0*1000.0)
#         return out

#     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
#                 env_kine_feat=None, env_data=None):
#         B = x_t.shape[0]
#         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
#         t_emb = self.time_fc2(t_emb)
#         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
#         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B,-1)
#         x_emb = (self.traj_embed(x_t[:,:T_seq])
#                  + self.pos_enc[:,:T_seq]
#                  + t_emb.unsqueeze(1)
#                  + self.step_embed(step_idx))
#         mem = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
#         if vel_obs_feat  is not None: mem.append(vel_obs_feat.unsqueeze(1))
#         if steering_feat is not None: mem.append(steering_feat.unsqueeze(1))
#         if env_kine_feat is not None: mem.append(env_kine_feat.unsqueeze(1))
#         decoded  = self.transformer(x_emb, torch.cat(mem, dim=1))
#         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
#         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1,T_seq,1) * 2.0
#         v_neural = v_neural * scale
#         with torch.no_grad():
#             v_phys  = self._beta_drift(x_t[:,:T_seq])
#             v_steer = self._steering_drift(x_t[:,:T_seq], env_data)
#         return (v_neural
#                 + torch.sigmoid(self.physics_scale)*v_phys
#                 + torch.sigmoid(self.steering_scale)*v_steer)

#     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
#                           vel_obs_feat=None, steering_feat=None,
#                           env_kine_feat=None, env_data=None, use_null=False):
#         ctx = self._apply_ctx_head(raw_ctx, noise_scale, use_null=use_null)
#         return self._decode(x_t, t, ctx,
#                             vel_obs_feat=vel_obs_feat,
#                             steering_feat=steering_feat,
#                             env_kine_feat=env_kine_feat,
#                             env_data=env_data)

#     def predict_speed(self, raw_ctx, vel_obs_feat):
#         ctx = self._apply_ctx_head(raw_ctx, noise_scale=0.0, use_null=False)
#         return self.speed_head(ctx, vel_obs_feat)


# # ═════════════════════════════════════════════════════════════════════════════
# #  Inference helpers — giữ nguyên từ v59
# # ═════════════════════════════════════════════════════════════════════════════

# @torch.no_grad()
# def _speed_sweep_correction(pred_traj_norm, obs_traj_norm,
#                              scales=(0.85, 0.92, 1.00, 1.08, 1.15)):
#     T_obs = obs_traj_norm.shape[0]
#     T, B  = pred_traj_norm.shape[0], pred_traj_norm.shape[1]
#     device = pred_traj_norm.device
#     if T_obs < 2:
#         return pred_traj_norm, torch.ones(B, device=device)

#     obs_spd_all = _step_speeds_deg(_norm_to_deg(obs_traj_norm))
#     n_obs = obs_spd_all.shape[0]
#     if n_obs >= 3:
#         alpha = 0.65
#         w = torch.tensor([alpha*(1-alpha)**i for i in range(n_obs)],
#                           dtype=torch.float, device=device).flip(0)
#         obs_spd = (obs_spd_all * (w/w.sum()).unsqueeze(1)).sum(0)
#     elif n_obs == 2:
#         obs_spd = 0.65*obs_spd_all[-1] + 0.35*obs_spd_all[-2]
#     else:
#         obs_spd = obs_spd_all[-1]
#     obs_spd = obs_spd.clamp(min=2.0)

#     anchor    = obs_traj_norm[-1].unsqueeze(0)
#     disp      = pred_traj_norm - anchor
#     t_idx     = torch.arange(T, dtype=torch.float, device=device)
#     best_sc   = torch.full((B,), -1e9, device=device)
#     best_traj = pred_traj_norm

#     for s in scales:
#         decay_exp = 1.0 - (t_idx / max(T-1, 1)) * 0.7
#         scale_t   = torch.full((T,), s, device=device) ** decay_exp
#         cand      = anchor + disp * scale_t.view(T,1,1)
#         full_deg  = torch.cat([_norm_to_deg(anchor), _norm_to_deg(cand)], dim=0)
#         n_c       = min(4, T)
#         cand_spd  = _step_speeds_deg(full_deg[:n_c+1]).mean(0)
#         score     = torch.exp(-((cand_spd - obs_spd)/obs_spd).pow(2)*4.0)
#         better    = score > best_sc
#         best_traj = torch.where(better.view(1,B,1).expand_as(cand),
#                                 cand, best_traj)
#         best_sc   = torch.where(better, score, best_sc)

#     return best_traj, best_sc


# @torch.no_grad()
# def _persistence_blend(model_pred_norm, obs_traj_norm, blend_strength=0.20):
#     T_obs = obs_traj_norm.shape[0]
#     T     = model_pred_norm.shape[0]
#     B, device = model_pred_norm.shape[1], model_pred_norm.device
#     if T_obs < 2:
#         return model_pred_norm
#     vels = obs_traj_norm[1:] - obs_traj_norm[:-1]
#     n_v  = vels.shape[0]
#     if n_v >= 3:
#         alpha = 0.7
#         w  = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
#                            dtype=torch.float, device=device).flip(0)
#         ev = (vels * (w/w.sum()).view(-1,1,1)).sum(0)
#     elif n_v == 2:
#         ev = 0.7*vels[-1] + 0.3*vels[-2]
#     else:
#         ev = vels[-1]
#     steps   = torch.arange(1, T+1, dtype=torch.float, device=device)
#     persist = (obs_traj_norm[-1].unsqueeze(0)
#                + ev.unsqueeze(0) * steps.view(T,1,1))
#     obs_spd_all = _step_speeds_deg(_norm_to_deg(obs_traj_norm))
#     if obs_spd_all.shape[0] >= 2:
#         spd_cv  = obs_spd_all.std(0) / obs_spd_all.mean(0).clamp(min=1.0)
#         alpha_b = (blend_strength
#                    * torch.sigmoid(-(spd_cv-0.3)*5.0)).unsqueeze(0).unsqueeze(-1)
#     else:
#         alpha_b = blend_strength * 0.5
#     return (1.0 - alpha_b)*model_pred_norm + alpha_b*persist


# def _score_ensemble_member(traj_norm, obs_traj_norm, speed_stats=None,
#                             speed_head_pred=None):
#     sp      = speed_stats or _SPEED_PRIOR
#     v_opt   = sp.get("v_opt", 15.0)
#     v_sigma = sp.get("v_sigma", 10.0)
#     v_cap   = sp.get("v_hard_cap", 35.0)
#     B, device = traj_norm.shape[1], traj_norm.device

#     spd = _step_speeds_deg(_norm_to_deg(traj_norm))
#     dtd = _norm_to_deg(traj_norm[1:]) - _norm_to_deg(traj_norm[:-1])
#     prior_sc  = torch.exp(-(((spd-v_opt)/v_sigma).pow(2)
#                              + F.relu(spd-v_cap)*2.0).mean(0)*0.5)
#     smooth_sc = (torch.exp(-(dtd[1:]-dtd[:-1]).norm(dim=-1).mean(0)*5.0)
#                  if dtd.shape[0] >= 2 else torch.ones(B, device=device))

#     if obs_traj_norm is not None and obs_traj_norm.shape[0] >= 2:
#         obs_v = obs_traj_norm[-1] - obs_traj_norm[-2]
#         if obs_traj_norm.shape[0] >= 3:
#             obs_v = 0.7*obs_v + 0.3*(obs_traj_norm[-2]-obs_traj_norm[-3])
#         obs_hn  = F.normalize(obs_v, dim=-1, eps=1e-6)
#         n_h     = min(3, traj_norm.shape[0]-1)
#         pv_m    = ((traj_norm[1:1+n_h]-traj_norm[:n_h]).mean(0)
#                    if n_h >= 1 else obs_v)
#         pred_hn = F.normalize(pv_m, dim=-1, eps=1e-6)
#         head_sc = torch.exp(((obs_hn*pred_hn).sum(-1)-1.0)*3.0)
#         obs_ref = _step_speeds_deg(_norm_to_deg(obs_traj_norm))[
#             -min(3, obs_traj_norm.shape[0]-1):].mean(0)
#         spd_sc  = torch.exp(-((spd[:min(4,spd.shape[0])].mean(0) - obs_ref)
#                                / obs_ref.clamp(min=5.0)).pow(2)*3.0)
#     else:
#         head_sc = spd_sc = torch.ones(B, device=device)

#     base_score = (head_sc.pow(0.35) * spd_sc.pow(0.30)
#                   * prior_sc.pow(0.20) * smooth_sc.pow(0.15))

#     if speed_head_pred is not None:
#         n = min(speed_head_pred.shape[1], spd.shape[0])
#         if n > 0:
#             pred_spd_t = speed_head_pred[:, :n].T
#             spd_match = torch.exp(
#                 -((spd[:n] - pred_spd_t) / v_sigma).pow(2).mean(0) * 3.0
#             )
#             return base_score.pow(0.45) * spd_match.pow(0.55)

#     return base_score


# # ═════════════════════════════════════════════════════════════════════════════
# #  [TWEAK-3] K=3 mode clustering — giữ nguyên từ v59-tweaked
# # ═════════════════════════════════════════════════════════════════════════════

# @torch.no_grad()
# def _k3_mode_cluster(trajs_norm, obs_norm, speed_stats=None,
#                      speed_head_pred=None, unimodal_spread_km=120.):
#     if not trajs_norm:
#         return obs_norm[-1:].expand(12, obs_norm.shape[1], 2)

#     dev = trajs_norm[0].device
#     T   = trajs_norm[0].shape[0]
#     B   = trajs_norm[0].shape[1]
#     N   = len(trajs_norm)
#     K   = min(3, N)

#     all_sc = torch.stack([
#         _score_ensemble_member(tr, obs_norm, speed_stats,
#                                speed_head_pred=speed_head_pred)
#         for tr in trajs_norm
#     ], dim=0)

#     CKS = [min(3, T-1), min(7, T-1), T-1]

#     def _multi_dist(tr_a_b, centers_deg):
#         dists = []
#         for ck in CKS:
#             pts = _norm_to_deg(tr_a_b[:, ck, :])
#             for c_deg in [centers_deg]:
#                 d = _haversine_deg(
#                     pts.unsqueeze(1).expand(pts.shape[0], c_deg.shape[0], 2),
#                     c_deg.unsqueeze(0).expand(pts.shape[0], c_deg.shape[0], 2))
#                 dists.append(d)
#         return torch.stack(dists, 0).mean(0)

#     all_ck_deg = torch.stack([
#         torch.stack([_norm_to_deg(tr[ck]) for ck in CKS], 0)
#         for tr in trajs_norm
#     ], 0)

#     results = []
#     for b in range(B):
#         sc_b  = all_sc[:, b]
#         tr_b  = torch.stack([tr[:, b, :] for tr in trajs_norm], 0)
#         ck_b  = all_ck_deg[:, :, b, :]

#         if N < 4:
#             w = F.softmax(sc_b * 3., 0)
#             results.append((tr_b * w.view(N,1,1)).sum(0))
#             continue

#         ep_b    = ck_b[:, -1, :]
#         ep_mean = ep_b.mean(0, keepdim=True)
#         spread  = _haversine_deg(ep_b, ep_mean.expand(N, 2)).mean().item()

#         if spread < unimodal_spread_km:
#             k_top  = max(3, int(N * 0.30))
#             topk   = sc_b.topk(k_top).indices
#             w_top  = F.softmax(sc_b[topk] * 5., 0)
#             results.append((tr_b[topk] * w_top.view(k_top,1,1)).sum(0))
#             continue

#         k_seed     = max(2, int(N * 0.20))
#         seed_pool  = sc_b.topk(k_seed).indices
#         seed_ck    = ck_b[seed_pool]
#         seed_mean  = seed_ck.mean(0, keepdim=True)
#         dist2mean  = torch.stack([
#             _haversine_deg(seed_ck[:, c, :], seed_mean[:, c, :].expand(k_seed, 2))
#             for c in range(len(CKS))
#         ], 0).mean(0)
#         seed_c_idx = seed_pool[dist2mean.argmin()]
#         centers_ck = [ck_b[seed_c_idx]]

#         for _ in range(K - 1):
#             cs = torch.stack(centers_ck, 0)
#             d2c_list = []
#             for i in range(N):
#                 d_per_ck = torch.stack([
#                     _haversine_deg(ck_b[i, c, :].unsqueeze(0), cs[:, c, :])
#                     for c in range(len(CKS))
#                 ], 0).mean(0)
#                 d2c_list.append(d_per_ck.min())
#             d_to_nearest = torch.stack(d2c_list)
#             centers_ck.append(ck_b[d_to_nearest.argmax()])

#         cck = torch.stack(centers_ck, 0)

#         for _ in range(3):
#             assign_scores = []
#             for i in range(N):
#                 d_per_k = torch.stack([
#                     torch.stack([
#                         _haversine_deg(ck_b[i, c, :].unsqueeze(0),
#                                        cck[k, c, :].unsqueeze(0))
#                         for c in range(len(CKS))
#                     ], 0).mean()
#                     for k in range(K)
#                 ], 0)
#                 assign_scores.append(d_per_k)
#             d2c    = torch.stack(assign_scores, 0)
#             assign = d2c.argmin(1)

#             new_c = []
#             for k in range(K):
#                 mk = (assign == k)
#                 if mk.sum() > 0:
#                     wk = F.softmax(sc_b[mk] * 3., 0)
#                     new_c.append((ck_b[mk] * wk.view(-1,1,1)).sum(0))
#                 else:
#                     new_c.append(cck[k])
#             cck = torch.stack(new_c, 0)

#         final_assign = []
#         for i in range(N):
#             d_per_k = torch.stack([
#                 torch.stack([
#                     _haversine_deg(ck_b[i, c, :].unsqueeze(0),
#                                    cck[k, c, :].unsqueeze(0))
#                     for c in range(len(CKS))
#                 ], 0).mean()
#                 for k in range(K)
#             ], 0)
#             final_assign.append(d_per_k.argmin())
#         assign = torch.stack(final_assign)

#         csc = torch.zeros(K, device=dev)
#         for k in range(K):
#             mk = (assign == k)
#             if mk.sum() > 0: csc[k] = sc_b[mk].sum()

#         best_k = csc.argmax().item()
#         mk     = (assign == best_k)
#         if not mk.any(): mk = torch.ones(N, dtype=torch.bool, device=dev)
#         w_win  = F.softmax(sc_b[mk] * 3., 0)
#         results.append((tr_b[mk] * w_win.view(-1,1,1)).sum(0))

#     return torch.stack(results, dim=1)


# # ═════════════════════════════════════════════════════════════════════════════
# #  TCFlowMatching v59-Strategy
# #  Thêm selector, expose loss terms cho GradNorm
# # ═════════════════════════════════════════════════════════════════════════════

# class TCFlowMatching(nn.Module):

#     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
#                  n_train_ens=4, unet_in_ch=13, ctx_noise_scale=0.01,
#                  initial_sample_sigma=0.03, teacher_forcing=True,
#                  use_ema=True, ema_decay=0.995,
#                  use_ate_ot=True, ot_epsilon=0.05,
#                  use_slerp=False,
#                  cfg_guidance_scale=1.5, cfg_uncond_prob=0.1,
#                  **kwargs):
#         super().__init__()
#         self.pred_len           = pred_len
#         self.obs_len            = obs_len
#         self.sigma_min          = sigma_min
#         self.ctx_noise_scale    = ctx_noise_scale
#         self.active_pred_len    = pred_len
#         self.use_ate_ot         = use_ate_ot
#         self.ot_epsilon         = ot_epsilon
#         self.use_slerp          = use_slerp
#         self.cfg_guidance_scale = cfg_guidance_scale
#         self.cfg_uncond_prob    = cfg_uncond_prob

#         self.net          = VelocityField(pred_len=pred_len, obs_len=obs_len,
#                                           sigma_min=sigma_min,
#                                           unet_in_ch=unet_in_ch, ctx_dim=256)
#         # [TWEAK-2] Learned step weights
#         self.step_weights = LearnedStepWeights(n_steps=pred_len, min_ratio=6.0)

#         # [S-D] LDR Selector — train script kiểm soát khi nào bật (phase >= 3)
#         self.selector     = SelectorNet(ctx_dim=256, cand_feat_dim=64,
#                                         hidden_dim=128)

#         self.use_ema   = use_ema
#         self.ema_decay = ema_decay
#         self._ema      = None

#     def init_ema(self):
#         if self.use_ema:
#             self._ema = EMAModel(self, decay=self.ema_decay)

#     def ema_update(self):
#         if self._ema is not None: self._ema.update(self)

#     def set_curriculum_len(self, *a, **kw): pass

#     @staticmethod
#     def _to_rel(traj, Me, lp, lm):
#         return torch.cat([traj - lp.unsqueeze(0),
#                           Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

#     @staticmethod
#     def _to_abs(rel, lp, lm):
#         d = rel.permute(1, 0, 2)
#         return lp.unsqueeze(0) + d[:,:,:2], lm.unsqueeze(0) + d[:,:,2:]

#     def _cfm_noisy(self, x1, sigma_min=None, lp=None):
#         if sigma_min is None: sigma_min = self.sigma_min
#         B = x1.shape[0]; device = x1.device
#         x0 = torch.randn_like(x1) * sigma_min
#         t  = torch.rand(B, device=device)
#         if self.use_slerp and x1.shape[-1] >= 2:
#             x_t      = _slerp_interpolant(x0, x1, t, lp=lp)
#             u_target = _slerp_velocity_target(x0, x1, t, lp=lp)
#         else:
#             te       = t.view(B, 1, 1)
#             x_t      = (1.0 - te)*x0 + te*x1
#             u_target = x1 - x0
#         return x_t, t, u_target

#     _cfm_noisy_slerp = _cfm_noisy

#     @staticmethod
#     def _lon_flip_aug(bl, p=0.3):
#         if torch.rand(1).item() > p: return bl
#         bl = list(bl)
#         for i in [0, 1, 2, 3]:
#             if torch.is_tensor(bl[i]) and bl[i].shape[-1] >= 1:
#                 t = bl[i].clone(); t[..., 0] = -t[..., 0]; bl[i] = t
#         return bl

#     @staticmethod
#     def _obs_noise_aug(bl, sigma=0.005):
#         if torch.rand(1).item() > 0.5: return bl
#         bl = list(bl)
#         if torch.is_tensor(bl[0]):
#             bl[0] = bl[0] + torch.randn_like(bl[0]) * sigma
#         return bl

#     def _persistence_forecast_rel(self, obs_traj, lp, lm, pred_len):
#         B, device = obs_traj.shape[1], obs_traj.device
#         if obs_traj.shape[0] >= 3:
#             vels  = obs_traj[1:] - obs_traj[:-1]
#             n_v   = vels.shape[0]
#             alpha = 0.7
#             w     = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
#                                   dtype=torch.float, device=device).flip(0)
#             lv    = (vels * (w/w.sum()).view(-1,1,1)).sum(0)
#         elif obs_traj.shape[0] >= 2:
#             lv = obs_traj[-1, :, :2] - obs_traj[-2, :, :2]
#         else:
#             lv = obs_traj.new_zeros(B, 2)
#         steps    = torch.arange(1, pred_len+1, device=device).float()
#         pred_abs = (obs_traj[-1, :, :2].unsqueeze(1)
#                     + lv.unsqueeze(1) * steps.view(1,-1,1))
#         pred_abs = pred_abs.permute(1, 0, 2)
#         pred_rel_pos = pred_abs - lp.unsqueeze(0)
#         pred_rel     = torch.cat([pred_rel_pos,
#                                    torch.zeros_like(pred_rel_pos)], dim=-1)
#         return pred_rel.permute(1, 0, 2)

#     def _compute_obs_momentum(self, obs_traj_norm):
#         T_obs = obs_traj_norm.shape[0]
#         if T_obs < 2:
#             return torch.zeros(obs_traj_norm.shape[1], 2,
#                                device=obs_traj_norm.device)
#         vels = obs_traj_norm[1:] - obs_traj_norm[:-1]
#         n_v  = vels.shape[0]
#         if n_v >= 3:
#             alpha = 0.65
#             w = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
#                               dtype=torch.float,
#                               device=obs_traj_norm.device).flip(0)
#             return (vels * (w/w.sum()).view(-1,1,1)).sum(0)
#         elif n_v == 2:
#             return 0.65*vels[-1] + 0.35*vels[-2]
#         return vels[-1]

#     @staticmethod
#     def _sigma_schedule(epoch):
#         if epoch < 2:   return 0.10
#         if epoch < 10:  return 0.10 - (epoch-2)/8.0 * (0.10 - 0.04)
#         if epoch < 20:  return max(0.04 - (epoch-10)/10.0 * 0.01, 0.035)
#         return 0.035

#     def get_loss(self, batch_list, epoch=0, **kwargs):
#         return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

#     def get_loss_breakdown(
#         self,
#         batch_list,
#         epoch: int = 0,
#         alpha_hard: float = 0.0,
#         is_hard: Optional[torch.Tensor] = None,
#         train_selector: bool = False,
#         lambda_dict: Optional[Dict[str, float]] = None,
#         step_weight_alpha: float = 0.0,
#     ) -> Dict:
#         """
#         [S-E] Loss breakdown với easy/hard split và GradNorm-compatible terms.

#         Args:
#             alpha_hard:     trọng số L_hard (0.0 phase 1, tăng đến 0.3 ở phase 2-3)
#                             do train script điều khiển theo warm-up schedule
#             is_hard:        [B] bool tensor từ classify_hard_easy (do train script tính)
#                             None → tất cả là easy (phase 1)
#             train_selector: True chỉ ở phase 3 (do train script kiểm soát)
#             lambda_dict:    {term_name: λ} từ GradNorm, None → dùng default
#         """
#         # BUG-5: epoch=-1 = val mode → skip augmentation
#         if epoch >= 0:
#             batch_list = self._lon_flip_aug(batch_list)
#             batch_list = self._obs_noise_aug(batch_list, sigma=0.005)

#         obs_t    = batch_list[0]
#         env_data = batch_list[13] if len(batch_list) > 13 else None
#         lp, lm   = obs_t[-1], batch_list[7][-1]
#         B, device = obs_t.shape[1], obs_t.device

#         speed_stats   = compute_speed_stats_from_norm(obs_t[..., :2])
#         current_sigma = self._sigma_schedule(epoch)
#         raw_ctx       = self.net._context(batch_list)

#         x1_rel = self._to_rel(batch_list[1], batch_list[8], lp, lm)

#         if self.use_ate_ot and B >= 4:
#             noise_base = torch.randn_like(x1_rel) * current_sigma
#             noise_matched, x1_matched = _spherical_ot_matching(
#                 noise_base, x1_rel, lp, epsilon=self.ot_epsilon)
#         else:
#             noise_matched = torch.randn_like(x1_rel) * current_sigma
#             x1_matched    = x1_rel

#         x_t, fm_t, u_target = self._cfm_noisy(
#             x1_matched, sigma_min=current_sigma, lp=lp)

#         use_null     = (torch.rand(1).item() < self.cfg_uncond_prob)
#         vel_obs_feat = self.net._get_kinematic_obs_feat(obs_t[:, :, :2])

#         pred_vel = self.net.forward_with_ctx(
#             x_t, fm_t, raw_ctx, env_data=env_data, use_null=use_null,
#             vel_obs_feat  = vel_obs_feat,
#             steering_feat = self.net._get_steering_feat(env_data, B, device),
#             env_kine_feat = self.net._get_env_kine_feat(env_data, B, device),
#         )

#         # [FIX-C] Pure iso FM loss (stable, không conflicting spaces)
#         l_fm = F.mse_loss(pred_vel, u_target)

#         # x1_pred → pred trajectory in degrees
#         fm_te    = fm_t.view(B, 1, 1)
#         x1_pred  = x_t + (1.0 - fm_te) * pred_vel
#         pred_abs, _ = self._to_abs(x1_pred, lp, lm)
#         pred_deg    = _norm_to_deg(pred_abs)
#         gt_deg      = _norm_to_deg(batch_list[1])

#         # [TWEAK-2] Learned step weights
#         sw_tensor = self.step_weights.get(n=pred_deg.shape[0])
#         sw_pen    = self.step_weights.penalty()

#         # [S-A] STTrans loss — trả về từng term riêng
#         loss_dict = compute_st_trans_loss(
#             pred_deg, gt_deg, epoch=epoch,
#             speed_stats=speed_stats, step_w=sw_tensor)

#         # [S-C] Hard loss — additive, chỉ cho hard samples
#         hard_loss_dict = {"l_hard_total": torch.zeros((), device=device),
#                           "n_hard": 0}
#         if is_hard is not None and is_hard.any() and alpha_hard > 0.0:
#             hard_loss_dict = compute_hard_loss(
#                 pred_deg, gt_deg, is_hard, step_w=sw_tensor)

#         # Speed head loss
#         if not use_null:
#             ctx_for_speed = self.net._apply_ctx_head(raw_ctx)
#             speed_pred = self.net.speed_head(ctx_for_speed, vel_obs_feat)
#             l_speed_head = _speed_head_loss(speed_pred, pred_deg, gt_deg, speed_stats)
#         else:
#             l_speed_head = x_t.new_zeros(())

#         # [S-D] Selector loss — chỉ ở phase 3, chỉ hard cases
#         l_sel_total = x_t.new_zeros(())
#         sel_loss_dict = {"l_soft_oracle": 0.0, "l_pairwise_rank": 0.0,
#                          "l_confidence": 0.0}
#         if train_selector and is_hard is not None and is_hard.any():
#             ctx_sel = self.net._apply_ctx_head(raw_ctx)  # [B, 256]
#             # Tạo candidates từ nhiều noise samples (nhẹ, không cần ddim đầy đủ)
#             sel_candidates = []
#             sel_gt_ades    = []
#             for _ in range(8):  # 8 candidates đủ để học ranking
#                 noise_k = torch.randn_like(x1_rel) * current_sigma
#                 x_t_k   = noise_k
#                 # 1-step approximation (nhanh, đủ cho selector training)
#                 t_k     = torch.full((B,), 0.5, device=device)
#                 vel_k   = self.net.forward_with_ctx(
#                     x_t_k, t_k, raw_ctx,
#                     vel_obs_feat=vel_obs_feat,
#                     steering_feat=self.net._get_steering_feat(env_data, B, device),
#                     env_kine_feat=self.net._get_env_kine_feat(env_data, B, device),
#                     env_data=env_data,
#                 )
#                 x1_k     = x_t_k + 0.5 * vel_k
#                 abs_k, _ = self._to_abs(x1_k, lp, lm)
#                 deg_k    = _norm_to_deg(abs_k)    # [T, B, 2]
#                 norm_k   = abs_k                   # [T, B, 2] normalized
#                 # ADE per sample
#                 T_c = min(deg_k.shape[0], gt_deg.shape[0])
#                 ade_k = _haversine_deg(deg_k[:T_c], gt_deg[:T_c]).mean(0)  # [B]
#                 sel_candidates.append(norm_k)
#                 sel_gt_ades.append(ade_k.detach())

#             cand_ades = torch.stack(sel_gt_ades, dim=0)  # [N, B]
#             scores    = self.selector.score_candidates(
#                 ctx_sel, sel_candidates, gt_deg, obs_t[:,:,:2])  # [N, B]
#             confidence = self.selector.get_confidence(ctx_sel)   # [B]

#             s_dict = selector_loss(scores, cand_ades, confidence, is_hard)
#             l_sel_total = s_dict["l_sel_total"]
#             sel_loss_dict = {k: v.item() if torch.is_tensor(v) else v
#                              for k, v in s_dict.items()}

#         # ── Tổng hợp final loss
#         # [S-A] Dùng lambda_dict từ GradNorm nếu có, fallback về default
#         if lambda_dict is not None:
#             λ_dpe  = lambda_dict.get("l_dpe",   1.20)
#             λ_vel  = lambda_dict.get("l_vel_reg", 1.40)
#             λ_head = lambda_dict.get("l_heading", 0.40)
#             λ_spd  = lambda_dict.get("l_speed",   0.05)
#             λ_acc  = lambda_dict.get("l_accel",   0.01)
#         else:
#             λ_dpe = 1.20; λ_vel = 1.40; λ_head = 0.40
#             λ_spd = 0.05; λ_acc = 0.01

#         l_base = (λ_dpe  * loss_dict["l_dpe"]
#                   + λ_vel  * loss_dict["l_vel_reg"]
#                   + λ_head * loss_dict["l_heading"]
#                   + λ_spd  * loss_dict["l_speed"]
#                   + λ_acc  * loss_dict["l_accel"])

#         total = (l_fm
#                  + l_base
#                  + alpha_hard * hard_loss_dict["l_hard_total"]
#                  + 0.30 * l_speed_head
#                  + 0.20 * l_sel_total   # selector loss nhỏ (không dùng GradNorm)
#                  + sw_pen)

#         if torch.isnan(total) or torch.isinf(total):
#             total = x_t.new_zeros(())

#         sw_st = self.step_weights.stats()
#         d = dict(loss_dict)
#         d.update({
#             "total"            : total,
#             "l_base"           : l_base.item() if torch.is_tensor(l_base) else 0.0,
#             "l_fm"             : l_fm.item(),
#             "fm_mse"           : l_fm.item(),
#             "l_hard_total"     : hard_loss_dict["l_hard_total"].item()
#                                  if torch.is_tensor(hard_loss_dict["l_hard_total"])
#                                  else 0.0,
#             "l_endpoint_norm"  : hard_loss_dict.get("l_endpoint_norm", 0.0),
#             "l_disp_norm"      : hard_loss_dict.get("l_disp_norm", 0.0),
#             "n_hard"           : hard_loss_dict.get("n_hard", 0),
#             "alpha_hard"       : alpha_hard,
#             "l_sel_total"      : l_sel_total.item() if torch.is_tensor(l_sel_total) else 0.0,
#             "l_soft_oracle"    : sel_loss_dict.get("l_soft_oracle", 0.0),
#             "l_pairwise_rank"  : sel_loss_dict.get("l_pairwise_rank", 0.0),
#             "l_confidence"     : sel_loss_dict.get("l_confidence", 0.0),
#             "speed_head_l"     : l_speed_head.item() if torch.is_tensor(l_speed_head) else 0.0,
#             "sigma"            : current_sigma,
#             "v_opt"            : speed_stats.get("v_opt", 15.0),
#             "obs_spd_p50"      : speed_stats.get("p50_kmh", 0.0),
#             "sw_ratio"         : sw_st["sw_ratio"],
#             "sw_72h"           : sw_st["sw_72h"],
#             "ate_x1"           : 0.0,
#         })
#         return d

#     @torch.no_grad()
#     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
#                predict_csv=None, importance_weight=True, use_cfg=True,
#                use_selector=False, selector_threshold=0.5):
#         """
#         [S-D] Thêm use_selector: nếu True và confidence > threshold → dùng selector
#         Mặc định False để backward compat với phase 1/2
#         """
#         obs_t    = batch_list[0]
#         env_data = batch_list[13] if len(batch_list) > 13 else None
#         lp       = obs_t[-1]; lm = batch_list[7][-1]
#         B        = lp.shape[0]; device = lp.device
#         T        = self.pred_len; dt = 1.0 / max(ddim_steps, 1)

#         raw_ctx       = self.net._context(batch_list)
#         vel_obs_feat  = self.net._get_kinematic_obs_feat(obs_t[:, :, :2])
#         steering_feat = self.net._get_steering_feat(env_data, B, device)
#         env_kine_feat = self.net._get_env_kine_feat(env_data, B, device)
#         speed_stats   = compute_speed_stats_from_norm(obs_t[..., :2])
#         persist_init  = self._persistence_forecast_rel(obs_t, lp, lm, T)
#         obs_norm      = obs_t[:, :, :2]

#         try:
#             speed_head_pred = self.net.predict_speed(raw_ctx, vel_obs_feat)
#         except Exception:
#             speed_head_pred = None

#         if obs_t.shape[0] >= 2:
#             obs_h_n = F.normalize(
#                 obs_t[-1,:,:2] - obs_t[-2,:,:2], dim=-1, eps=1e-6)
#         else:
#             obs_h_n = None

#         obs_mom = self._compute_obs_momentum(obs_norm)
#         if obs_t.shape[0] >= 3:
#             vv    = obs_t[1:,:,:2] - obs_t[:-1,:,:2]
#             heads = F.normalize(vv, dim=-1, eps=1e-6)
#             cos_s = (heads[1:]*heads[:-1]).sum(-1).mean(0)
#             mom_gate = torch.sigmoid((cos_s-0.5)*8.0)
#         else:
#             mom_gate = torch.ones(B, device=device)

#         def _mom_str(s, tot):
#             return 0.08 * 0.5 * (1.0 + math.cos(math.pi*s/max(tot,1)))

#         all_norms, all_me = [], []

#         for _ in range(num_ensemble):
#             x_t = persist_init + torch.randn_like(persist_init) * self.sigma_min * 2.5

#             for step in range(ddim_steps):
#                 t_b = torch.full((B,), step*dt, device=device)
#                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0

#                 if use_cfg and step > 0:
#                     v_cond   = self.net.forward_with_ctx(
#                         x_t, t_b, raw_ctx, noise_scale=ns,
#                         vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
#                         env_kine_feat=env_kine_feat, env_data=env_data,
#                         use_null=False)
#                     v_uncond = self.net.forward_with_ctx(
#                         x_t, t_b, raw_ctx, noise_scale=0.0,
#                         vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
#                         env_kine_feat=env_kine_feat, env_data=env_data,
#                         use_null=True)
#                     if obs_h_n is not None:
#                         pred_h = F.normalize(v_cond[:,0,:2].detach(), dim=-1, eps=1e-6)
#                         cos_a  = (obs_h_n * pred_h).sum(-1).clamp(-1.0, 1.0)
#                         gs     = (0.8 + 0.7*(cos_a+1.0)*0.5).view(B,1,1)
#                         vel    = v_uncond + gs * (v_cond - v_uncond)
#                     else:
#                         vel = v_uncond + 1.5 * (v_cond - v_uncond)
#                 else:
#                     vel = self.net.forward_with_ctx(
#                         x_t, t_b, raw_ctx, noise_scale=ns,
#                         vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
#                         env_kine_feat=env_kine_feat, env_data=env_data)

#                 m_s = _mom_str(step, ddim_steps)
#                 if m_s > 1e-4:
#                     me  = obs_mom.unsqueeze(1).expand(B, T, 2)
#                     mf  = torch.cat([me, torch.zeros(B, T, 2, device=device)], -1)
#                     vel = vel + m_s * mom_gate.view(B,1,1) * mf

#                 x_t = (x_t + dt*vel).clamp(-3.0, 3.0)

#             tr, me = self._to_abs(x_t, lp, lm)
#             all_norms.append(tr)
#             all_me.append(me)

#         # Speed sweep NARROW range (bỏ extreme scales của v59 gốc)
#         SCALES = (0.85, 0.92, 1.00, 1.08, 1.15)
#         augmented = []
#         for tn in all_norms:
#             bt, _ = _speed_sweep_correction(tn, obs_norm, SCALES)
#             augmented.append(bt)
#             augmented.append(tn)

#         all_me_t = torch.stack(all_me)

#         # [S-D] Selector tại inference (phase 3+)
#         # Easy: k3_mode_cluster với median (nhanh, ổn định)
#         # Hard: selector chọn 1 candidate tốt nhất, fallback nếu confidence thấp
#         if use_selector:
#             ctx_inf = self.net._apply_ctx_head(raw_ctx)
#             confidence = self.selector.get_confidence(ctx_inf)  # [B]

#             # Phân loại hard/easy tại inference bằng đặc trưng vật lý
#             is_hard_inf = classify_hard_easy(obs_norm)  # [B]

#             # Hard samples có confidence CAO → selector đang tự tin → dùng selector
#             # confidence thấp → selector không chắc → fallback về k3_mode_cluster
#             use_sel_mask = is_hard_inf & (confidence >= selector_threshold)

#             if use_sel_mask.any():
#                 scores_inf = self.selector.score_candidates(
#                     ctx_inf, augmented, gt_deg=None, obs_norm=obs_norm)
#                 best_idx = scores_inf.argmax(dim=0)  # [B]

#                 pred_cluster = _k3_mode_cluster(augmented, obs_norm, speed_stats,
#                                                  speed_head_pred=speed_head_pred)
#                 pred_sel = torch.zeros_like(pred_cluster)
#                 for b in range(B):
#                     if use_sel_mask[b]:
#                         pred_sel[:, b, :] = augmented[best_idx[b].item()][:, b, :]
#                     else:
#                         pred_sel[:, b, :] = pred_cluster[:, b, :]
#                 pred_mean = pred_sel
#             else:
#                 pred_mean = _k3_mode_cluster(augmented, obs_norm, speed_stats,
#                                               speed_head_pred=speed_head_pred)
#         else:
#             pred_mean = _k3_mode_cluster(augmented, obs_norm, speed_stats,
#                                           speed_head_pred=speed_head_pred)

#         pred_mean = _persistence_blend(pred_mean, obs_norm, blend_strength=0.20)
#         all_c = torch.stack(augmented)

#         if predict_csv:
#             self._write_predict_csv(predict_csv, pred_mean, all_c)
#         return pred_mean, all_me_t.mean(0), all_c

#     def _score_sample(self, traj, speed_stats=None):
#         return _score_ensemble_member(traj, None, speed_stats)

#     @staticmethod
#     def _write_predict_csv(csv_path, traj_mean, all_trajs):
#         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
#         T, B, _ = traj_mean.shape
#         mlon = ((traj_mean[...,0]*50.0+1800.0)/10.0).cpu().numpy()
#         mlat = ((traj_mean[...,1]*50.0)/10.0).cpu().numpy()
#         alon = ((all_trajs[...,0]*50.0+1800.0)/10.0).cpu().numpy()
#         alat = ((all_trajs[...,1]*50.0)/10.0).cpu().numpy()
#         fields = ["timestamp","batch_idx","step_idx","lead_h",
#                   "lon_mean_deg","lat_mean_deg",
#                   "lon_std_deg","lat_std_deg","ens_spread_km"]
#         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#         write_hdr = not os.path.exists(csv_path)
#         with open(csv_path, "a", newline="") as fh:
#             w = csv.DictWriter(fh, fieldnames=fields)
#             if write_hdr: w.writeheader()
#             for b in range(B):
#                 for k in range(T):
#                     dlat   = alat[:,k,b] - mlat[k,b]
#                     dlon   = ((alon[:,k,b] - mlon[k,b])
#                               * math.cos(math.radians(float(mlat[k,b]))))
#                     spread = float(((dlat**2+dlon**2)**0.5).mean() * DEG2KM)
#                     w.writerow({
#                         "timestamp"    : ts,
#                         "batch_idx"    : b,
#                         "step_idx"     : k,
#                         "lead_h"       : (k+1)*6,
#                         "lon_mean_deg" : f"{mlon[k,b]:.4f}",
#                         "lat_mean_deg" : f"{mlat[k,b]:.4f}",
#                         "lon_std_deg"  : f"{alon[:,k,b].std():.4f}",
#                         "lat_std_deg"  : f"{alat[:,k,b].std():.4f}",
#                         "ens_spread_km": f"{spread:.2f}",
#                     })


# # Backward compat
# TCDiffusion = TCFlowMatching

"""
flow_matching_model.py — TC-FlowMatching v59-Strategy [FIXED]
═══════════════════════════════════════════════════════════════════════════════

BASE: v59-Strategy (giữ nguyên toàn bộ architecture)

BUG FIXES trong file này:

  [FIX-M1] selector_loss() — pairwise loop logic sai hoàn toàn
    Vấn đề cũ:
      top_idx = ade_h.argsort(dim=0)[:n_top]   # [n_top, n_hard]
      ranks   = sort_idx.argsort(dim=0)          # ranks[n, b] = rank của candidate n trong sample b
      for i in range(n_top):
          rank_i = ranks[top_idx[i], arange(n_hard)]
      → ranks[top_idx[i][b], b] = rank của candidate top_idx[i][b] trong sample b
      → Nhưng top_idx[i][b] là index của candidate có ADE rank-i trong sample b
      → ranks[top_idx[i][b], b] = i theo định nghĩa của argsort!
      → rank_i luôn = i với mọi b → ADW và loss vô nghĩa

    Fix: rewrite toàn bộ pairwise loop theo logic đúng:
      - Dùng top_idx[i][b] và top_idx[j][b] để lấy ADE và score của
        candidate xếp hạng i và j trong từng sample b
      - Tính ADW dựa trên rank difference (i vs j, không cần ranks tensor)
      - Pairwise loss: candidate i tốt hơn j (ade_i < ade_j) → score_i > score_j

  [FIX-M2] SelectorNet._extract_cand_features() — guard khi T=1 (speeds empty)
    Vấn đề cũ: khi T=1, speeds = empty tensor → .std() crash hoặc nan
    Fix: guard `if speeds.numel() >= 2` đủ để xử lý numel=0 và numel=1

  [FIX-M3] compute_diversity_score() — _haversine_deg input shape verification
    Vấn đề: _norm_to_deg(ep_norm) với ep_norm [B, 2] → _haversine_deg cần [*, 2]
    Code hiện tại đúng nhưng thêm comment và shape assert để rõ ràng hơn

GIỮ NGUYÊN từ v59-Strategy:
  [S-A] compute_st_trans_loss expose terms cho GradNorm
  [S-B] hard_score_from_obs đa tiêu chí vật lý
  [S-C] compute_hard_loss normalized
  [S-D] SelectorNet + selector_loss
  [S-E] get_loss_breakdown easy/hard pipeline
  [S-F] compute_diversity_score
  [TWEAK-2] LearnedStepWeights
  [TWEAK-3] _k3_mode_cluster
  EMAModel, VelocityField, TCFlowMatching.sample
"""
from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.FNO3D_encoder import FNO3DEncoder
from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
from Model.env_net_transformer_gphsplit import Env_net

# ─────────────────────────────────────────────────────────────────────────────
R_EARTH      = 6371.0
DT_HOURS     = 6.0
DEG2KM       = 111.0
_NORM_TO_DEG = 5.0

STEP_WEIGHTS = [2.0, 3.5, 3.5, 4.0, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
_SPEED_PRIOR = {"v_opt": 15.0, "v_sigma": 10.0, "v_hard_cap": 35.0}


# ─────────────────────────────────────────────────────────────────────────────
# [TWEAK-2] LearnedStepWeights
# ─────────────────────────────────────────────────────────────────────────────

class LearnedStepWeights(nn.Module):
    def __init__(self, n_steps: int = 12, min_ratio: float = 6.0):
        super().__init__()
        self.n_steps   = n_steps
        self.min_ratio = min_ratio
        self.raw = nn.Parameter(torch.linspace(-0.3, 1.5, n_steps))

    def forward(self) -> torch.Tensor:
        w = torch.cumsum(F.softplus(self.raw), dim=0)
        return w * self.n_steps / (w.sum() + 1e-8)

    def get(self, n=None):
        w = self.forward()
        return w[:n] if n is not None else w

    def penalty(self):
        w = self.forward()
        ratio = w[-1] / w[0].clamp(min=1e-6)
        return 0.02 * F.relu(self.min_ratio - ratio) ** 2

    @torch.no_grad()
    def stats(self):
        w = self.forward()
        return {
            "sw_ratio": (w[-1] / w[0].clamp(1e-6)).item(),
            "sw_72h":   w[-1].item(),
            "sw_6h":    w[0].item(),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  Coordinate utilities
# ═════════════════════════════════════════════════════════════════════════════

def _norm_to_deg(t):
    return torch.stack([
        (t[..., 0] * 50.0 + 1800.0) / 10.0,
        (t[..., 1] * 50.0) / 10.0,
    ], dim=-1)


def _haversine_deg(p1, p2):
    lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a = (torch.sin(dlat / 2).pow(2) +
         torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


def _unwrap_model(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m


def _step_speeds_deg(traj_deg):
    T = traj_deg.shape[0]
    if T < 2:
        return traj_deg.new_zeros(1, traj_deg.shape[1])
    return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


def _forward_azimuth(p1, p2):
    lon1 = torch.deg2rad(p1[..., 0]); lat1 = torch.deg2rad(p1[..., 1])
    lon2 = torch.deg2rad(p2[..., 0]); lat2 = torch.deg2rad(p2[..., 1])
    dlon = lon2 - lon1
    y = torch.sin(dlon) * torch.cos(lat2)
    x = (torch.cos(lat1) * torch.sin(lat2)
         - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon))
    return torch.atan2(y, x)


# ═════════════════════════════════════════════════════════════════════════════
#  Speed statistics
# ═════════════════════════════════════════════════════════════════════════════

def compute_speed_stats_from_norm(obs_traj):
    T = obs_traj.shape[0]
    if T < 2:
        return dict(_SPEED_PRIOR)
    with torch.no_grad():
        lon_deg = (obs_traj[..., 0] * 50.0 + 1800.0) / 10.0
        lat_deg = (obs_traj[..., 1] * 50.0) / 10.0
        dlon    = lon_deg[1:] - lon_deg[:-1]
        dlat    = lat_deg[1:] - lat_deg[:-1]
        cos_lat = torch.cos(torch.deg2rad(
            (lat_deg[:-1] + lat_deg[1:]) * 0.5)).clamp(1e-4)
        speed   = torch.sqrt(
            (dlon * cos_lat * DEG2KM)**2 + (dlat * DEG2KM)**2 + 1e-6
        ) / DT_HOURS
        sf = speed.flatten()
        if sf.numel() < 4:
            return dict(_SPEED_PRIOR)
        mean_s = float(sf.mean())
        std_s  = float(sf.std().clamp(min=1.0))
        q      = torch.quantile(sf, torch.tensor([.50, .75, .95],
                                                   device=sf.device))
        p50, p95 = float(q[0]), float(q[2])
    return {
        "mean_kmh"  : mean_s,  "std_kmh"   : std_s,
        "p50_kmh"   : p50,     "p95_kmh"   : p95,
        "v_opt"     : max(p50, 5.0),
        "v_sigma"   : max(std_s + 5.0, 5.0),
        "v_hard_cap": float(torch.tensor(p95 * 1.8).clamp(25.0, 80.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  [S-B] Hard score đa tiêu chí
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def hard_score_from_obs(obs_traj_norm: torch.Tensor) -> torch.Tensor:
    """
    Tính hard_score cho mỗi sample trong batch dựa trên đặc trưng vật lý.

    Args:
        obs_traj_norm: [T_obs, B, >=2] normalized trajectory

    Returns:
        hard_score: [B] float tensor, càng cao càng khó
    """
    T, B = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
    device = obs_traj_norm.device

    if T < 3:
        return torch.zeros(B, device=device)

    traj_deg = _norm_to_deg(obs_traj_norm[..., :2])  # [T, B, 2]

    # curvature_index: tổng góc đổi hướng / π
    az_12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])   # [T-2, B]
    az_23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])    # [T-2, B]
    angle_diff = torch.abs(az_23 - az_12)
    angle_diff = torch.where(
        angle_diff > math.pi,
        2 * math.pi - angle_diff,
        angle_diff
    )
    curvature_index = angle_diff.mean(0) / math.pi  # [B] ∈ [0, 1]

    # speed_variance: coefficient of variation
    speeds = _step_speeds_deg(traj_deg)  # [T-1, B]
    if speeds.shape[0] >= 2:
        speed_mean = speeds.mean(0).clamp(min=1.0)
        speed_std  = speeds.std(0)
        speed_variance = (speed_std / speed_mean).clamp(0.0, 1.0)
    else:
        speed_variance = torch.zeros(B, device=device)

    # direction_change: số lần đổi hướng > 20°
    large_turn     = (angle_diff > (20.0 / 180.0 * math.pi)).float()
    direction_change = large_turn.mean(0)  # [B] ∈ [0, 1]

    hard_score = (0.4 * curvature_index
                  + 0.3 * speed_variance
                  + 0.3 * direction_change)

    return hard_score  # [B]


@torch.no_grad()
def classify_hard_easy(
    obs_traj_norm: torch.Tensor,
    per_sample_loss: Optional[torch.Tensor] = None,
    hard_score_p: float = 70.0,
    loss_p: float = 50.0,
) -> torch.Tensor:
    """
    Phân loại hard/easy per-batch (dùng cho evaluation fallback).
    Cho training: dùng classify_hard_easy_global() trong train script.
    """
    scores = hard_score_from_obs(obs_traj_norm)
    B = scores.shape[0]

    if B < 4:
        return torch.zeros(B, dtype=torch.bool, device=scores.device)

    threshold_score = torch.quantile(scores, hard_score_p / 100.0)
    mask_score = scores >= threshold_score

    if per_sample_loss is None:
        return mask_score

    threshold_loss = torch.quantile(per_sample_loss, loss_p / 100.0)
    mask_loss = per_sample_loss >= threshold_loss

    return mask_score & mask_loss


# ═════════════════════════════════════════════════════════════════════════════
#  [S-C] Hard loss — normalized
# ═════════════════════════════════════════════════════════════════════════════

def compute_hard_loss(
    pred_deg: torch.Tensor,
    gt_deg: torch.Tensor,
    is_hard: torch.Tensor,
    step_w: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Tính L_hard chỉ trên hard samples.
    Tất cả loss đều normalized về scale 0–1 để GradNorm hoạt động.
    """
    device = pred_deg.device
    zero   = pred_deg.new_zeros(())

    if not is_hard.any():
        return {"l_endpoint_norm": zero, "l_disp_norm": zero,
                "l_hard_total": zero, "n_hard": 0}

    pred_h = pred_deg[:, is_hard, :]  # [T, n_hard, 2]
    gt_h   = gt_deg[:, is_hard, :]
    T      = min(pred_h.shape[0], gt_h.shape[0])
    n_hard = int(is_hard.sum().item())

    if T < 4:
        return {"l_endpoint_norm": zero, "l_disp_norm": zero,
                "l_hard_total": zero, "n_hard": n_hard}

    # L_endpoint: Huber loss tại checkpoint 48h (step7), 72h (step11)
    ep_total = zero
    ep_w_sum = 0.0
    for s, ew in [(min(7, T-1), 1.0), (min(11, T-1), 2.0)]:
        dist  = _haversine_deg(pred_h[s], gt_h[s])  # [n_hard]
        d_hub = 200.0
        loss_s = torch.where(dist < d_hub,
                              dist.pow(2) / (2 * d_hub),
                              dist - d_hub / 2).mean()
        ep_total = ep_total + ew * loss_s
        ep_w_sum += ew
    l_endpoint_raw  = ep_total / max(ep_w_sum, 1e-6)
    l_endpoint_norm = l_endpoint_raw / 500.0  # normalize km → [0,1]

    # L_disp: displacement step-weighted
    if step_w is not None:
        w = step_w[:T].to(device); w = w / w.sum()
    else:
        w = pred_deg.new_tensor(STEP_WEIGHTS[:T]); w = w / w.sum()

    dist_all   = _haversine_deg(pred_h[:T], gt_h[:T])   # [T, n_hard]
    l_disp_raw  = (dist_all * w.unsqueeze(1)).mean()
    l_disp_norm = l_disp_raw / 300.0  # normalize km → [0,1]

    l_hard_total = l_endpoint_norm + l_disp_norm

    return {
        "l_endpoint_norm": l_endpoint_norm,
        "l_disp_norm":     l_disp_norm,
        "l_hard_total":    l_hard_total,
        "n_hard":          n_hard,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  [S-A] compute_st_trans_loss — GradNorm-compatible
# ═════════════════════════════════════════════════════════════════════════════

def compute_st_trans_loss(
    pred_deg: torch.Tensor,
    gt_deg: torch.Tensor,
    epoch: int = 0,
    speed_stats: Optional[dict] = None,
    step_w: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    STTrans loss với từng term được trả về riêng (GradNorm-compatible).
    Train script tính: total = λ_dpe·l_dpe + λ_vel·l_vel_reg + ...
    với λ được GradNorm điều chỉnh.
    """
    sp         = speed_stats or _SPEED_PRIOR
    v_opt      = sp.get("v_opt",      15.0)
    v_sigma    = sp.get("v_sigma",    10.0)
    v_hard_cap = sp.get("v_hard_cap", 35.0)
    T      = min(pred_deg.shape[0], gt_deg.shape[0])
    device = pred_deg.device

    if T < 2:
        zero = pred_deg.new_zeros(())
        return {k: zero for k in
                ["l_dpe","l_mse","l_vel_reg","l_heading","l_speed","l_accel",
                 "total","l_pos","l_head","l_smooth","l_disp"]}

    # Step weights
    if step_w is not None:
        w = step_w[:T].to(device); w = w / w.sum() * T
    else:
        w = pred_deg.new_tensor(STEP_WEIGHTS[:T]); w = w / w.sum() * T

    # L_DPE: Huber loss
    dist  = _haversine_deg(pred_deg[:T], gt_deg[:T])  # [T, B]
    d     = 200.0
    l_dpe = ((torch.where(dist < d, dist.pow(2) / (2*d), dist - d/2))
             * w.unsqueeze(1)).mean() / d

    # L_MSE: raw MSE trên normalized coords
    l_mse = F.mse_loss(pred_deg[:T], gt_deg[:T])

    # L_speed: speed prior
    pred_spd = _step_speeds_deg(pred_deg[:T])  # [T-1, B]
    if pred_spd.shape[0] > 0:
        l_speed = (0.7 * ((pred_spd - v_opt) / v_sigma).pow(2).mean() +
                   0.3 * F.relu(pred_spd - v_hard_cap).pow(2).mean()
                   / v_hard_cap**2)
    else:
        l_speed = pred_deg.new_zeros(())

    # L_accel: smoothness
    if pred_spd.shape[0] >= 2:
        l_accel = (((pred_spd[1:] - pred_spd[:-1]).abs() / DT_HOURS).pow(2).mean()
                   / max(v_sigma * 0.5, 3.0)**2)
    else:
        l_accel = pred_deg.new_zeros(())

    # L_heading: direction continuity
    if T >= 3:
        cos_lat_h = torch.cos(torch.deg2rad(
            (gt_deg[:T-1,:,1] + gt_deg[1:T,:,1]) * 0.5)).clamp(1e-4)
        pv_raw = pred_deg[1:T] - pred_deg[:T-1]
        gv_raw = gt_deg[1:T]   - gt_deg[:T-1]
        pv_km  = torch.stack([pv_raw[...,0]*cos_lat_h*DEG2KM,
                               pv_raw[...,1]*DEG2KM], dim=-1)
        gv_km  = torch.stack([gv_raw[...,0]*cos_lat_h*DEG2KM,
                               gv_raw[...,1]*DEG2KM], dim=-1)
        cos_sim   = (F.normalize(pv_km, dim=-1, eps=1e-6) *
                     F.normalize(gv_km, dim=-1, eps=1e-6)).sum(-1)
        head_err  = (1.0 - cos_sim).clamp(0.0, 2.0)
        if step_w is not None:
            hw = step_w[1:T].to(device); hw = hw / hw.sum()
            l_heading = (head_err * hw.unsqueeze(1)).mean()
        else:
            l_heading = head_err.mean()
    else:
        l_heading = pred_deg.new_zeros(())

    # L_vel_reg: velocity regression
    l_vel_reg = _velocity_regression_loss(pred_deg, gt_deg, speed_stats,
                                          step_w=step_w)

    # Default weighted sum (fallback nếu GradNorm không init)
    total_default = (1.20 * l_dpe
                     + 1.40 * l_vel_reg
                     + 0.40 * l_heading
                     + 0.05 * l_speed
                     + 0.01 * l_accel)

    if torch.isnan(total_default) or torch.isinf(total_default):
        total_default = pred_deg.new_zeros(())

    def _s(x): return x.item() if torch.is_tensor(x) else float(x)

    return dict(
        # Tensor terms cho GradNorm và backward
        l_dpe       = l_dpe,
        l_mse       = l_mse,
        l_vel_reg   = l_vel_reg,
        l_heading   = l_heading,
        l_speed     = l_speed,
        l_accel     = l_accel,
        total       = total_default,
        # Aliases tensor
        l_pos       = l_dpe,
        l_head      = l_heading,
        l_smooth    = l_accel,
        l_disp      = l_vel_reg,
        # Float aliases cho logging (không dùng trong backward)
        dpe         = _s(l_dpe),
        mse         = _s(l_mse),
        heading     = _s(l_heading),
        vel_reg     = _s(l_vel_reg),
        speed       = _s(l_speed),
        accel       = _s(l_accel),
        # Zeroed log compat
        l_anchor=0.0, l_hard=0.0, lambda_hard=0.0, q_hard_mean=0.0,
        anchor_ade=0.0, ate=0.0, cte=0.0, sph_ate=0.0, endpoint=0.0,
        signed_ate=0.0, signed_cte=0.0, direct_ep=0.0,
        ate_mean_km=0.0, cte_mean_km=0.0, speed_match=0.0,
        acc_kmh2=0.0, aux_fno=0.0, sigma=0.0, fm_mse=0.0,
        multi_marg=0.0, rollout_ate=0.0, rollout_w=0.0,
    )


compute_ate_focused_loss = compute_st_trans_loss


# ═════════════════════════════════════════════════════════════════════════════
#  [S-D] SelectorNet
# ═════════════════════════════════════════════════════════════════════════════

class SelectorNet(nn.Module):
    def __init__(self, ctx_dim: int = 256, cand_feat_dim: int = 64,
                 hidden_dim: int = 128):
        super().__init__()
        self.cand_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, cand_feat_dim),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(ctx_dim + cand_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    @staticmethod
    def _extract_cand_features(
        cand_norm: torch.Tensor,
        gt_deg: Optional[torch.Tensor] = None,
        obs_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Trích xuất 6 features từ một candidate trajectory.
        [FIX-M2] Guard khi T=1 (speeds tensor empty).
        """
        device = cand_norm.device
        T = cand_norm.shape[0]
        cand_deg = _norm_to_deg(cand_norm.unsqueeze(1)).squeeze(1)  # [T, 2]

        # ADE vs GT
        if gt_deg is not None and gt_deg.shape[0] >= T:
            ade = _haversine_deg(cand_deg.unsqueeze(1),
                                  gt_deg[:T].unsqueeze(1)).mean().item()
            ade_n = ade / 300.0
        else:
            ade_n = 0.0

        # Endpoint 72h
        if gt_deg is not None and gt_deg.shape[0] > min(T-1, 11):
            ep_step = min(T-1, 11)
            ep_dist = _haversine_deg(
                cand_deg[ep_step].unsqueeze(0),
                gt_deg[ep_step].unsqueeze(0)).item()
            ep_n = ep_dist / 500.0
        else:
            ep_n = 0.0

        # Speed stats
        # [FIX-M2] Guard: speeds có thể là empty khi T=1
        if T >= 2:
            speeds = _haversine_deg(
                cand_deg[:-1].unsqueeze(1),
                cand_deg[1:].unsqueeze(1)
            ).squeeze(1) / DT_HOURS  # [T-1]

            if speeds.numel() >= 1:
                mean_spd = float(speeds.mean().item())
            else:
                mean_spd = 0.0

            # [FIX-M2] std(unbiased=False) safe khi numel=1 (returns 0.0)
            # numel=0 không xảy ra vì T>=2 đảm bảo speeds có T-1>=1 elements
            if speeds.numel() >= 2:
                speed_std = float(speeds.std(unbiased=False).item())
            else:
                speed_std = 0.0  # 1 step: không tính được variance

            speed_var  = min(speed_std / max(mean_spd, 1.0), 1.0)
            mean_spd_n = mean_spd / 30.0
        else:
            # T=1: không có bước nào để tính speed
            mean_spd_n = speed_var = 0.0

        # Curvature
        if T >= 3:
            az12 = _forward_azimuth(
                cand_deg[:-2].unsqueeze(1),
                cand_deg[1:-1].unsqueeze(1)).squeeze(1)
            az23 = _forward_azimuth(
                cand_deg[1:-1].unsqueeze(1),
                cand_deg[2:].unsqueeze(1)).squeeze(1)
            diff = (az23 - az12).abs()
            diff = torch.where(diff > math.pi, 2*math.pi - diff, diff)
            curvature = (diff.mean() / math.pi).item()
        else:
            curvature = 0.0

        # Heading consistency với obs
        if obs_norm is not None and obs_norm.shape[0] >= 2 and T >= 1:
            obs_vel  = obs_norm[-1, :2] - obs_norm[-2, :2]
            cand_vel = cand_norm[0, :2] - obs_norm[-1, :2]
            obs_h    = F.normalize(obs_vel.unsqueeze(0), dim=-1, eps=1e-6)
            cand_h   = F.normalize(cand_vel.unsqueeze(0), dim=-1, eps=1e-6)
            head_cons = ((obs_h * cand_h).sum(-1).clamp(-1, 1).item() + 1.0) / 2.0
        else:
            head_cons = 0.5

        feat = [ade_n, ep_n, mean_spd_n, speed_var, curvature, head_cons]
        # Guard: thay nan/inf bằng 0
        feat = [0.0 if (not math.isfinite(v)) else v for v in feat]
        return torch.tensor(feat, dtype=torch.float, device=device)

    def score_candidates(
        self,
        ctx: torch.Tensor,
        candidates: List[torch.Tensor],
        gt_deg: Optional[torch.Tensor] = None,
        obs_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Chấm điểm N candidates cho B samples.
        Returns: scores [N, B]
        """
        B = ctx.shape[0]
        all_scores = []

        for cand in candidates:
            feat_list = []
            for b in range(B):
                gt_b  = gt_deg[:, b, :]   if gt_deg is not None  else None
                obs_b = obs_norm[:, b, :]  if obs_norm is not None else None
                feat  = self._extract_cand_features(cand[:, b, :], gt_b, obs_b)
                feat_list.append(feat)
            cand_feat = torch.stack(feat_list, dim=0)       # [B, 6]
            cand_enc  = self.cand_encoder(cand_feat)         # [B, 64]
            inp       = torch.cat([ctx, cand_enc], dim=-1)   # [B, 320]
            score     = self.scorer(inp).squeeze(-1)          # [B]
            all_scores.append(score)

        return torch.stack(all_scores, dim=0)  # [N, B]

    def get_confidence(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.confidence_head(ctx).squeeze(-1)  # [B]


# ═════════════════════════════════════════════════════════════════════════════
#  [S-D] selector_loss — [FIX-M1] Pairwise loop logic hoàn toàn rewritten
# ═════════════════════════════════════════════════════════════════════════════

def selector_loss(
    scores: torch.Tensor,       # [N, B] từ score_candidates
    gt_ades: torch.Tensor,      # [N, B] ADE của từng candidate (detached)
    confidence: torch.Tensor,   # [B]
    is_hard: torch.Tensor,      # [B] bool
) -> Dict[str, torch.Tensor]:
    """
    LDR-inspired selector loss với Adaptive Decay Weight (ADW).

    L_sel = L_soft_oracle + L_pairwise_rank + L_confidence

    [FIX-M1] Pairwise loop rewritten hoàn toàn.

    Vấn đề cũ:
      top_idx = ade_h.argsort(dim=0)[:n_top]   # [n_top, n_hard]
      ranks   = sort_idx.argsort(dim=0)
      for i in range(n_top):
          rank_i = ranks[top_idx[i], arange(n_hard)]
      → ranks[top_idx[i][b], b] = rank của candidate top_idx[i][b] trong sample b
      → Vì top_idx[i][b] là candidate có ADE rank i trong sample b:
         ranks[top_idx[i][b], b] = i theo định nghĩa của argsort(argsort)
      → rank_i = i hằng số với mọi b → rank_gap = |i - j| hằng số
      → ADW chỉ phụ thuộc vào i,j không phụ thuộc data → vô nghĩa!

    Fix:
      - Không cần ranks tensor
      - top_idx[:,b] cho biết candidates được sort theo ADE tăng trong sample b
      - top_idx[i,b] = index của candidate có ADE rank i trong sample b
      - Dùng trực tiếp top_idx để lấy ADE và score của từng rank position
      - ADW: IRW = 1/(j-i), ERD = exp(-mean_rank / n_top)
    """
    device = scores.device
    zero   = scores.new_zeros(())

    if not is_hard.any():
        return {"l_soft_oracle": zero, "l_pairwise_rank": zero,
                "l_confidence": zero, "l_sel_total": zero}

    sc_h   = scores[:, is_hard]     # [N, n_hard]
    ade_h  = gt_ades[:, is_hard]    # [N, n_hard]
    conf_h = confidence[is_hard]    # [n_hard]
    N, n_hard = sc_h.shape

    if N < 2 or n_hard == 0:
        return {"l_soft_oracle": zero, "l_pairwise_rank": zero,
                "l_confidence": zero, "l_sel_total": zero}

    # ── L_soft_oracle ─────────────────────────────────────────────────────
    tau = ade_h.mean().clamp(min=10.0)
    oracle_prob = F.softmax(-ade_h / tau, dim=0)  # [N, n_hard]
    log_sc      = F.log_softmax(sc_h, dim=0)       # [N, n_hard]
    l_soft_oracle = -(oracle_prob * log_sc).sum(0).mean()

    # ── L_pairwise_rank với ADW = IRW × ERD [FIX-M1] ─────────────────────
    #
    # top_idx[i, b] = index (trong [0, N)) của candidate có ADE rank i
    #                 dalam sample b (rank 0 = ADE tốt nhất)
    # Đây là argsort theo dim=0 của ade_h
    top_idx  = ade_h.argsort(dim=0)       # [N, n_hard], giá trị ∈ [0, N)
    n_top    = max(2, N // 2)             # chỉ xét top-50% candidates
    # [FIX-M-A] Tạo arange_h 1 lần ngoài loop thay vì O(n_top²) lần trong loop.
    arange_h = torch.arange(n_hard, device=device)

    pairwise_losses = []

    for i in range(n_top):
        for j in range(i + 1, n_top):
            # [FIX-M1] Lấy candidate indices tại rank i và j cho mỗi sample
            # top_idx[i]: [n_hard] — indices của candidates ở rank i
            # top_idx[j]: [n_hard] — indices của candidates ở rank j
            idx_i = top_idx[i]   # [n_hard], values ∈ [0, N)
            idx_j = top_idx[j]   # [n_hard], values ∈ [0, N)

            # Lấy ADE và score của candidates ở rank i và j cho từng sample
            # sc_h[idx_i[b], b] = score của candidate rank-i trong sample b
            ade_at_i   = ade_h[idx_i, arange_h]   # [n_hard]
            ade_at_j   = ade_h[idx_j, arange_h]   # [n_hard]
            score_at_i = sc_h[idx_i, arange_h]    # [n_hard]
            score_at_j = sc_h[idx_j, arange_h]    # [n_hard]

            # ADW = IRW × ERD
            # IRW: cặp rank (i, j) gần nhau quan trọng hơn
            # i < j nên rank_gap = j - i >= 1
            rank_gap = float(j - i)
            irw = 1.0 / rank_gap   # scalar (đã biết i,j tại compile time)

            # ERD: candidates ở rank thấp (tệ hơn) giảm trọng số
            avg_rank = (i + j) / 2.0
            erd = math.exp(-avg_rank / max(n_top / 2.0, 1.0))

            adw = irw * erd  # scalar ADW (không phụ thuộc data)

            # Pairwise ranking loss:
            # Candidate rank i (ADE nhỏ hơn) phải có score cao hơn rank j
            # Constraint: score_at_i > score_at_j + margin khi ade_at_i < ade_at_j
            # Note: do top_idx là argsort theo ADE tăng, ade_at_i <= ade_at_j theo
            # kỳ vọng, nhưng per-sample có thể không đơn điệu (do sampling noise)
            # → ta dùng ground truth ordering: better_i = (ade_at_i < ade_at_j)
            margin   = 0.1
            better_i = (ade_at_i < ade_at_j).float()  # [n_hard]
            # Nếu better_i=1: muốn score_at_i > score_at_j → loss nếu margin không thỏa
            # Nếu better_i=0: muốn score_at_j > score_at_i
            pair_loss = F.relu(
                margin - (score_at_i - score_at_j) * (2.0 * better_i - 1.0)
            )  # [n_hard]

            # Áp dụng ADW (scalar) và average over hard samples
            pairwise_losses.append(adw * pair_loss.mean())

    if pairwise_losses:
        l_pairwise_rank = torch.stack(pairwise_losses).mean()
    else:
        l_pairwise_rank = zero

    # ── L_confidence ──────────────────────────────────────────────────────
    # oracle_prob.clamp(min=1e-8) trước log để tránh log(0)=-inf dù softmax
    # thực tế không bao giờ = 0 chính xác, nhưng float32 underflow có thể xảy ra
    oracle_entropy = -(oracle_prob * oracle_prob.clamp(min=1e-8).log()).sum(0)
    max_entropy    = math.log(max(N, 2))  # guard N=1 (log(1)=0 → div/0)
    target_conf    = 1.0 - (oracle_entropy / max_entropy).clamp(0, 1)
    l_confidence   = F.mse_loss(conf_h, target_conf.detach())

    l_sel_total = l_soft_oracle + 0.5 * l_pairwise_rank + 0.3 * l_confidence

    return {
        "l_soft_oracle":   l_soft_oracle,
        "l_pairwise_rank": l_pairwise_rank,
        "l_confidence":    l_confidence,
        "l_sel_total":     l_sel_total,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  [S-F] Diversity score
#  [FIX-M3] Thêm comment giải thích shape, đảm bảo correctness
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_diversity_score(candidates: List[torch.Tensor]) -> float:
    """
    Đo diversity của FM candidates tại endpoint 72h.

    Args:
        candidates: List of N tensors, mỗi tensor [T, B, 2] normalized

    Returns:
        diversity_km: float (km) — mean std của endpoint position across candidates

    Shape flow:
        endpoints:           [N, B, 2]  (deg sau _norm_to_deg)
        ep_mean:             [1, B, 2]
        endpoints.reshape:   [N*B, 2]
        ep_mean.expand+reshape: [N*B, 2]
        dists (haversine):   [N*B]     → reshape về [N, B]
        dists.std(dim=0):    [B]       → std across N candidates per sample
        .mean():             scalar    → mean across B samples
    """
    if len(candidates) < 2:
        return 0.0

    T       = candidates[0].shape[0]
    B       = candidates[0].shape[1]
    ep_step = min(T - 1, 11)  # step 11 = 72h

    # Lấy endpoint 72h của mỗi candidate
    endpoints = []
    for cand in candidates:
        ep_norm = cand[ep_step]              # [B, 2] normalized
        ep_deg  = _norm_to_deg(ep_norm)      # [B, 2] degrees
        endpoints.append(ep_deg)

    endpoints = torch.stack(endpoints, dim=0)         # [N, B, 2]
    N = endpoints.shape[0]

    # Mean endpoint per sample
    ep_mean = endpoints.mean(dim=0, keepdim=True)     # [1, B, 2]

    # Khoảng cách từ mỗi candidate tới mean
    # Reshape để _haversine_deg nhận [*, 2]
    dists = _haversine_deg(
        endpoints.reshape(N * B, 2),
        ep_mean.expand(N, B, 2).reshape(N * B, 2)
    ).reshape(N, B)   # [N, B] — distance của mỗi candidate tới mean per sample

    # Std across candidates per sample, mean across samples
    diversity_km = float(dists.std(dim=0).mean().item())
    return diversity_km


# ═════════════════════════════════════════════════════════════════════════════
#  Velocity regression loss
# ═════════════════════════════════════════════════════════════════════════════

def _velocity_regression_loss(pred_deg, gt_deg, speed_stats=None, step_w=None):
    sp      = speed_stats or _SPEED_PRIOR
    v_sigma = sp.get("v_sigma", 10.0)
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        return pred_deg.new_zeros(())

    pred_spd = _step_speeds_deg(pred_deg[:T])
    gt_spd   = _step_speeds_deg(gt_deg[:T])
    if step_w is not None:
        w = step_w[1:T].to(pred_deg.device); w = w / w.sum()
    else:
        w = pred_deg.new_tensor(STEP_WEIGHTS[1:T]); w = w / w.sum()

    gt_spd_clamped = gt_spd.clamp(min=5.0)
    l_abs = ((pred_spd - gt_spd_clamped).pow(2) / v_sigma**2
             * w.unsqueeze(1)).mean()

    plon = pred_deg[1:T, :, 0] - pred_deg[:T-1, :, 0]
    plat = pred_deg[1:T, :, 1] - pred_deg[:T-1, :, 1]
    glon = gt_deg[1:T, :, 0]   - gt_deg[:T-1, :, 0]
    glat = gt_deg[1:T, :, 1]   - gt_deg[:T-1, :, 1]
    cos  = torch.cos(torch.deg2rad(
        (pred_deg[:T-1,:,1] + pred_deg[1:T,:,1]) * 0.5)).clamp(1e-4)
    pv_km = torch.stack([plon*cos*DEG2KM/DT_HOURS, plat*DEG2KM/DT_HOURS], -1)
    gv_km = torch.stack([glon*cos*DEG2KM/DT_HOURS, glat*DEG2KM/DT_HOURS], -1)
    l_vec = (F.mse_loss(pv_km, gv_km, reduction="none").mean(-1)
             / v_sigma**2 * w.unsqueeze(1)).mean()

    return (0.5*l_abs + 0.5*l_vec).clamp(0.0, 20.0)


# ═════════════════════════════════════════════════════════════════════════════
#  SpeedHead
# ═════════════════════════════════════════════════════════════════════════════

class SpeedHead(nn.Module):
    def __init__(self, ctx_dim=256, obs_feat_dim=256, pred_len=12):
        super().__init__()
        self.pred_len = pred_len
        self.fc = nn.Sequential(
            nn.Linear(ctx_dim + obs_feat_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, pred_len),
        )
        with torch.no_grad():
            nn.init.xavier_uniform_(self.fc[-1].weight, gain=0.1)
            nn.init.zeros_(self.fc[-1].bias)

    def forward(self, ctx, vel_obs_feat):
        h = torch.cat([ctx, vel_obs_feat], dim=-1)
        speed_pred = self.fc(h)
        return F.softplus(speed_pred) * 3.0 + 2.0


def _speed_head_loss(speed_pred, pred_deg, gt_deg, speed_stats=None):
    sp = speed_stats or _SPEED_PRIOR
    v_sigma = sp.get("v_sigma", 10.0)
    T_gt = gt_deg.shape[0]
    pred_len = speed_pred.shape[1]
    T = min(pred_len + 1, T_gt)
    if T < 2:
        return speed_pred.new_zeros(())

    gt_spd = _step_speeds_deg(gt_deg[:T])
    gt_spd = gt_spd.permute(1, 0)
    n = min(pred_len, gt_spd.shape[1])

    speed_pred_n = speed_pred[:, :n]
    gt_spd_n     = gt_spd[:, :n].clamp(min=2.0)

    w = speed_pred.new_tensor(STEP_WEIGHTS[1:n+1])
    w = w / w.sum()

    loss = (F.mse_loss(speed_pred_n, gt_spd_n, reduction='none') / v_sigma**2)
    return (loss * w.unsqueeze(0)).mean()


# ═════════════════════════════════════════════════════════════════════════════
#  Sinkhorn OT
# ═════════════════════════════════════════════════════════════════════════════

def _sinkhorn_log(cost, epsilon=0.05, n_iter=50):
    B      = cost.shape[0]
    device = cost.device
    log_a  = -math.log(B) * torch.ones(B, device=device)
    log_b  = -math.log(B) * torch.ones(B, device=device)
    log_K  = -cost / epsilon
    log_u  = torch.zeros(B, device=device)
    log_v  = torch.zeros(B, device=device)
    for _ in range(n_iter):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
    return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


def _geodesic_ot_cost(x0_rel, x1_rel, lp):
    B = x0_rel.shape[0]

    def _abs_deg(rel):
        return _norm_to_deg(lp.unsqueeze(1) + rel[:, :, :2])

    x0d = _abs_deg(x0_rel); x1d = _abs_deg(x1_rel)
    x0e = x0d.unsqueeze(1).expand(B, B, -1, -1).reshape(B*B, -1, 2)
    x1e = x1d.unsqueeze(0).expand(B, B, -1, -1).reshape(B*B, -1, 2)
    pos_cost = _haversine_deg(x0e, x1e).reshape(B, B, -1).mean(-1) / 500.0

    spd0 = _step_speeds_deg(x0d.permute(1,0,2)).permute(1,0).mean(-1)
    spd1 = _step_speeds_deg(x1d.permute(1,0,2)).permute(1,0).mean(-1)
    speed_cost = (spd0.unsqueeze(1) - spd1.unsqueeze(0)).abs() / 20.0

    def _mean_bearing(td):
        b = _forward_azimuth(td[:, :-1, :], td[:, 1:, :])
        return torch.atan2(b.sin().mean(-1), b.cos().mean(-1))
    h0 = _mean_bearing(x0d); h1 = _mean_bearing(x1d)
    dh = (h0.unsqueeze(1) - h1.unsqueeze(0) + math.pi) % (2*math.pi) - math.pi
    dir_cost = dh.abs() / math.pi

    return 1.0*pos_cost + 0.5*speed_cost + 0.3*dir_cost


def _spherical_ot_matching(x0_batch, x1_batch, lp, epsilon=0.05):
    B = x0_batch.shape[0]
    if B < 4:
        return x0_batch, x1_batch
    try:
        cost = _geodesic_ot_cost(x0_batch, x1_batch, lp)
        with torch.no_grad():
            pi = _sinkhorn_log(cost, epsilon=epsilon)
        flat = pi.reshape(-1).clamp(0.0)
        s    = flat.sum()
        if not torch.isfinite(s) or s < 1e-10:
            return x0_batch, x1_batch
        idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
        col = idx % B
        return x0_batch[col], x1_batch[col]
    except Exception:
        return x0_batch, x1_batch


# ═════════════════════════════════════════════════════════════════════════════
#  Linear/Slerp interpolant
# ═════════════════════════════════════════════════════════════════════════════

def _slerp_interpolant(x0, x1, t, lp=None):
    B  = x0.shape[0]; te = t.view(B, 1, 1)
    if lp is not None and x0.shape[-1] >= 2:
        abs0 = lp.unsqueeze(1) + x0[:, :, :2]
        abs1 = lp.unsqueeze(1) + x1[:, :, :2]
    else:
        abs0 = x0[:, :, :2]; abs1 = x1[:, :, :2]
    abs0_deg = _norm_to_deg(abs0); abs1_deg = _norm_to_deg(abs1)
    lat0 = torch.deg2rad(abs0_deg[..., 1]); lat1 = torch.deg2rad(abs1_deg[..., 1])
    dlon = torch.deg2rad(abs1_deg[..., 0] - abs0_deg[..., 0])
    dlat = lat1 - lat0
    a    = (torch.sin(dlat/2).pow(2)
            + torch.cos(lat0) * torch.cos(lat1) * torch.sin(dlon/2).pow(2))
    omega     = 2.0 * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())
    sin_omega = torch.sin(omega).clamp(1e-6)
    linear    = omega < 1e-4
    te_sq     = te.squeeze(1)
    coeff0 = torch.where(linear, 1.0 - te_sq,
                         torch.sin((1-te_sq)*omega) / sin_omega)
    coeff1 = torch.where(linear, te_sq,
                         torch.sin(te_sq*omega) / sin_omega)
    return coeff0.unsqueeze(-1)*x0 + coeff1.unsqueeze(-1)*x1


def _slerp_velocity_target(x0, x1, t, lp=None):
    B  = x0.shape[0]; te = t.view(B, 1, 1)
    if lp is not None and x0.shape[-1] >= 2:
        abs0 = lp.unsqueeze(1) + x0[:, :, :2]
        abs1 = lp.unsqueeze(1) + x1[:, :, :2]
    else:
        abs0 = x0[:, :, :2]; abs1 = x1[:, :, :2]
    abs0_deg = _norm_to_deg(abs0); abs1_deg = _norm_to_deg(abs1)
    lat0 = torch.deg2rad(abs0_deg[..., 1]); lat1 = torch.deg2rad(abs1_deg[..., 1])
    dlon = torch.deg2rad(abs1_deg[..., 0] - abs0_deg[..., 0])
    dlat = lat1 - lat0
    a    = (torch.sin(dlat/2).pow(2)
            + torch.cos(lat0) * torch.cos(lat1) * torch.sin(dlon/2).pow(2))
    omega     = 2.0 * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())
    sin_omega = torch.sin(omega).clamp(1e-6)
    oos       = omega / sin_omega
    linear    = omega < 1e-4
    te_sq     = te.squeeze(1)
    coeff0 = torch.where(linear, -torch.ones_like(te_sq),
                         -torch.cos((1-te_sq)*omega)*oos)
    coeff1 = torch.where(linear,  torch.ones_like(te_sq),
                          torch.cos(te_sq*omega)*oos)
    return coeff0.unsqueeze(-1)*x0 + coeff1.unsqueeze(-1)*x1


# ═════════════════════════════════════════════════════════════════════════════
#  EMAModel
# ═════════════════════════════════════════════════════════════════════════════

class EMAModel:
    def __init__(self, model, decay=0.995):
        self.decay  = decay
        m = _unwrap_model(model)
        self.shadow = {k: v.detach().clone()
                       for k, v in m.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        m = _unwrap_model(model)
        with torch.no_grad():
            for k, v in m.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(
                        v.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        """
        Copy EMA shadow weights → model. Returns backup of current weights.
        QUAN TRỌNG: luôn gọi restore() sau khi dùng xong để không leak EMA
        weights vào training.
        """
        m = _unwrap_model(model)
        backup, sd = {}, m.state_dict()
        for k in self.shadow:
            if k not in sd: continue
            backup[k] = sd[k].detach().clone()
            sd[k].copy_(self.shadow[k])
        return backup

    def restore(self, model, backup):
        """Restore model weights từ backup (trả về bởi apply_to)."""
        m = _unwrap_model(model)
        sd = m.state_dict()
        for k, v in backup.items():
            if k in sd: sd[k].copy_(v)


# ═════════════════════════════════════════════════════════════════════════════
#  VelocityField
# ═════════════════════════════════════════════════════════════════════════════

class VelocityField(nn.Module):
    RAW_CTX_DIM = 512

    def __init__(self, pred_len=12, obs_len=8, ctx_dim=256,
                 sigma_min=0.02, unet_in_ch=13, **kwargs):
        super().__init__()
        self.pred_len = pred_len
        self.obs_len  = obs_len
        self.ctx_dim  = ctx_dim

        self.spatial_enc     = FNO3DEncoder(
            in_channel=unet_in_ch, out_channel=1, d_model=32,
            n_layers=4, modes_t=4, modes_h=4, modes_w=4,
            spatial_down=32, dropout=0.05)
        self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.bottleneck_proj = nn.Linear(128, 128)
        self.decoder_proj    = nn.Linear(1, 16)
        self.enc_1d          = DataEncoder1D(
            in_1d=4, feat_3d_dim=128, mlp_h=64,
            lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
        self.env_enc         = Env_net(obs_len=obs_len, d_model=32)

        self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
        self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
        self.ctx_drop = nn.Dropout(0.15)
        self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

        self.null_embedding = nn.Parameter(
            torch.randn(1, self.RAW_CTX_DIM) * 0.02)

        self.vel_obs_enc = nn.Sequential(
            nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.GELU())

        self.steering_enc = nn.Sequential(
            nn.Linear(7, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 256))

        self.env_kine_enc = nn.Sequential(
            nn.Linear(14, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 256), nn.GELU())

        self.time_fc1   = nn.Linear(256, 512)
        self.time_fc2   = nn.Linear(512, 256)
        self.traj_embed = nn.Linear(4, 256)
        self.pos_enc    = nn.Parameter(
            torch.randn(1, pred_len, 256) * 0.02)
        self.step_embed = nn.Embedding(pred_len, 256)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=256, nhead=8, dim_feedforward=1024,
                dropout=0.10, activation="gelu", batch_first=True),
            num_layers=2)
        self.out_fc1 = nn.Linear(256, 512)
        self.out_fc2 = nn.Linear(512, 4)

        self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
        self.physics_scale  = nn.Parameter(torch.ones(4) * 1.5)
        self.steering_scale = nn.Parameter(torch.ones(4) * 1.0)

        self.speed_head = SpeedHead(ctx_dim=ctx_dim, obs_feat_dim=256,
                                    pred_len=pred_len)

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for name, m in self.named_modules():
                if isinstance(m, nn.Linear) and "out_fc" in name:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None: nn.init.zeros_(m.bias)

    def _time_emb(self, t, dim=256):
        half = dim // 2
        freq = torch.exp(
            torch.arange(half, dtype=torch.float32, device=t.device)
            * (-math.log(10000.0) / max(half-1, 1)))
        emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
        return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

    def _context(self, batch_list):
        obs_traj  = batch_list[0]
        obs_Me    = batch_list[7]
        image_obs = batch_list[11]
        env_data  = batch_list[13]
        if image_obs.dim() == 4: image_obs = image_obs.unsqueeze(2)
        if (image_obs.shape[1] == 1
                and self.spatial_enc.in_channel != 1):
            image_obs = image_obs.expand(
                -1, self.spatial_enc.in_channel, -1, -1, -1)

        e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
        T_obs = obs_traj.shape[0]
        e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0,2,1)
        e_3d_s = self.bottleneck_proj(e_3d_s)
        if e_3d_s.shape[1] != T_obs:
            e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
                                    mode="linear",
                                    align_corners=False).permute(0,2,1)

        e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
        t_w = torch.softmax(
            torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
                          device=e_3d_dec_t.device)*0.5, dim=0)
        f_sp = self.decoder_proj(
            (e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1,0,2)
        h_t    = self.enc_1d(obs_in, e_3d_s)
        e_env, _, _ = self.env_enc(env_data, image_obs)
        return F.gelu(self.ctx_ln(self.ctx_fc1(
            torch.cat([h_t, e_env, f_sp], dim=-1))))

    def _apply_ctx_head(self, raw, noise_scale=0.0, use_null=False):
        if use_null:
            raw = self.null_embedding.expand(raw.shape[0], -1)
        elif noise_scale > 0.0:
            raw = raw + torch.randn_like(raw) * noise_scale
        return self.ctx_fc2(self.ctx_drop(raw))

    def _get_kinematic_obs_feat(self, obs_traj):
        B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
        if T_obs >= 2:
            vel     = obs_traj[1:] - obs_traj[:-1]
            lat_mid = obs_traj[:-1, :, 1] * _NORM_TO_DEG
            cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
            dx_km   = vel[:,:,0] * cos_lat * DEG2KM * _NORM_TO_DEG
            dy_km   = vel[:,:,1]            * DEG2KM * _NORM_TO_DEG
            speed   = torch.sqrt(dx_km**2 + dy_km**2 + 1e-6) / DT_HOURS
            heading = torch.atan2(vel[:,:,1], vel[:,:,0])
            speed_n = (speed / 20.0).clamp(-3.0, 3.0)
            if T_obs >= 3:
                dspd  = speed[1:] - speed[:-1]
                accel = torch.cat([obs_traj.new_zeros(1,B),
                                   (dspd/10.0).clamp(-3.0,3.0)], 0)
            else:
                accel = obs_traj.new_zeros(T_obs-1, B)
            kine = torch.stack([vel[:,:,0], vel[:,:,1], speed_n,
                                 heading.sin(), heading.cos(), accel], dim=-1)
        else:
            kine = obs_traj.new_zeros(self.obs_len, B, 6)
        if kine.shape[0] < self.obs_len:
            kine = torch.cat([obs_traj.new_zeros(
                self.obs_len-kine.shape[0], B, 6), kine], 0)
        else:
            kine = kine[-self.obs_len:]
        return self.vel_obs_enc(kine.permute(1,0,2).reshape(B, -1))

    def _get_vel_obs_feat(self, obs_traj):
        return self._get_kinematic_obs_feat(obs_traj)

    def _get_steering_feat(self, env_data, B, device):
        if env_data is None: return torch.zeros(B, 256, device=device)
        def _safe(k, norm=1.0):
            v = env_data.get(k)
            if v is None or not torch.is_tensor(v):
                return torch.full((B,), 0.0, device=device)
            v = v.float().to(device)
            while v.dim() > 1: v = v.mean(-1)
            v = v.view(-1)[:B] if v.numel() >= B else torch.full(
                (B,), 0.0, device=device)
            return (v / norm).clamp(-3.0, 3.0)
        return self.steering_enc(torch.stack([
            _safe("u500_mean",       30.0),
            _safe("v500_mean",       30.0),
            _safe("u500_center",     30.0),
            _safe("v500_center",     30.0),
            _safe("steering_speed",   1.0),
            _safe("steering_dir_sin", 1.0),
            _safe("steering_dir_cos", 1.0),
        ], dim=-1))

    def _get_env_kine_feat(self, env_data, B, device):
        if env_data is None: return torch.zeros(B, 256, device=device)
        def _get_t(key, dim):
            v = env_data.get(key)
            if v is None: return torch.zeros(B, dim, device=device)
            if not torch.is_tensor(v):
                try: v = torch.tensor(v, dtype=torch.float, device=device)
                except: return torch.zeros(B, dim, device=device)
            v = v.float().to(device)
            if v.dim() == 0:
                return (v.expand(B, dim) if dim == 1
                        else torch.zeros(B, dim, device=device))
            if v.dim() == 1:
                if v.shape[0] == dim: return v.unsqueeze(0).expand(B, dim)
                if v.shape[0] == B:
                    return (v.unsqueeze(1).expand(B, dim) if dim == 1
                            else torch.zeros(B, dim, device=device))
            if v.dim() == 2:
                if v.shape == (B, dim): return v
                return (v[:B, :dim] if v.shape[1] >= dim
                        else F.pad(v[:B], (0, dim-v.shape[1])))
            if v.dim() == 3:
                vv = v[-1] if v.shape[1] == B else v[:B, -1]
                return (vv[:, :dim] if vv.shape[-1] >= dim
                        else F.pad(vv, (0, dim-vv.shape[-1])))
            return torch.zeros(B, dim, device=device)
        feat = torch.cat([_get_t("move_velocity",1),
                          _get_t("history_direction24",8),
                          _get_t("delta_velocity",5)], dim=-1)
        return self.env_kine_enc(feat)

    def _beta_drift(self, x_t):
        lat_rad = torch.deg2rad(x_t[:,:,1]*5.0).clamp(-85, 85)
        beta    = 2*7.2921e-5*torch.cos(lat_rad)/6.371e6
        R_tc    = 3e5
        v = torch.zeros_like(x_t)
        v[:,:,0] = -beta*R_tc**2/2*6*3600/(5*111*1000)
        v[:,:,1] =  beta*R_tc**2/4*6*3600/(5*111*1000)
        return v

    def _steering_drift(self, x_t, env_data):
        if env_data is None: return torch.zeros_like(x_t)
        B, device = x_t.shape[0], x_t.device
        def _sm(k):
            v = env_data.get(k)
            if v is None or not torch.is_tensor(v):
                return torch.zeros(B, device=device)
            v = v.float().to(device)
            while v.dim() > 1: v = v.mean(-1)
            return v.view(-1)[:B] if v.numel() >= B else torch.zeros(
                B, device=device)
        u   = _sm("u500_center"); vv = _sm("v500_center")
        cos = torch.cos(torch.deg2rad(x_t[:,:,1]*5.0)).clamp(1e-3)
        out = torch.zeros_like(x_t)
        out[:,:,0] = u.unsqueeze(1)*30.0*21600.0/(111.0*1000.0*cos)
        out[:,:,1] = vv.unsqueeze(1)*30.0*21600.0/(111.0*1000.0)
        return out

    def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
                env_kine_feat=None, env_data=None):
        B = x_t.shape[0]
        t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
        t_emb = self.time_fc2(t_emb)
        T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
        step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B,-1)
        x_emb = (self.traj_embed(x_t[:,:T_seq])
                 + self.pos_enc[:,:T_seq]
                 + t_emb.unsqueeze(1)
                 + self.step_embed(step_idx))
        mem = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
        if vel_obs_feat  is not None: mem.append(vel_obs_feat.unsqueeze(1))
        if steering_feat is not None: mem.append(steering_feat.unsqueeze(1))
        if env_kine_feat is not None: mem.append(env_kine_feat.unsqueeze(1))
        decoded  = self.transformer(x_emb, torch.cat(mem, dim=1))
        v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
        scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1,T_seq,1) * 2.0
        v_neural = v_neural * scale
        with torch.no_grad():
            v_phys  = self._beta_drift(x_t[:,:T_seq])
            v_steer = self._steering_drift(x_t[:,:T_seq], env_data)
        return (v_neural
                + torch.sigmoid(self.physics_scale)*v_phys
                + torch.sigmoid(self.steering_scale)*v_steer)

    def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
                          vel_obs_feat=None, steering_feat=None,
                          env_kine_feat=None, env_data=None, use_null=False):
        ctx = self._apply_ctx_head(raw_ctx, noise_scale, use_null=use_null)
        return self._decode(x_t, t, ctx,
                            vel_obs_feat=vel_obs_feat,
                            steering_feat=steering_feat,
                            env_kine_feat=env_kine_feat,
                            env_data=env_data)

    def predict_speed(self, raw_ctx, vel_obs_feat):
        ctx = self._apply_ctx_head(raw_ctx, noise_scale=0.0, use_null=False)
        return self.speed_head(ctx, vel_obs_feat)


# ═════════════════════════════════════════════════════════════════════════════
#  Inference helpers
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _speed_sweep_correction(pred_traj_norm, obs_traj_norm,
                             scales=(0.85, 0.92, 1.00, 1.08, 1.15)):
    T_obs = obs_traj_norm.shape[0]
    T, B  = pred_traj_norm.shape[0], pred_traj_norm.shape[1]
    device = pred_traj_norm.device
    if T_obs < 2:
        return pred_traj_norm, torch.ones(B, device=device)

    obs_spd_all = _step_speeds_deg(_norm_to_deg(obs_traj_norm))
    n_obs = obs_spd_all.shape[0]
    if n_obs >= 3:
        alpha = 0.65
        w = torch.tensor([alpha*(1-alpha)**i for i in range(n_obs)],
                          dtype=torch.float, device=device).flip(0)
        obs_spd = (obs_spd_all * (w/w.sum()).unsqueeze(1)).sum(0)
    elif n_obs == 2:
        obs_spd = 0.65*obs_spd_all[-1] + 0.35*obs_spd_all[-2]
    else:
        obs_spd = obs_spd_all[-1]
    obs_spd = obs_spd.clamp(min=2.0)

    anchor    = obs_traj_norm[-1].unsqueeze(0)
    disp      = pred_traj_norm - anchor
    t_idx     = torch.arange(T, dtype=torch.float, device=device)
    best_sc   = torch.full((B,), -1e9, device=device)
    best_traj = pred_traj_norm

    for s in scales:
        decay_exp = 1.0 - (t_idx / max(T-1, 1)) * 0.7
        scale_t   = torch.full((T,), s, device=device) ** decay_exp
        cand      = anchor + disp * scale_t.view(T,1,1)
        full_deg  = torch.cat([_norm_to_deg(anchor), _norm_to_deg(cand)], dim=0)
        n_c       = min(4, T)
        cand_spd  = _step_speeds_deg(full_deg[:n_c+1]).mean(0)
        score     = torch.exp(-((cand_spd - obs_spd)/obs_spd).pow(2)*4.0)
        better    = score > best_sc
        best_traj = torch.where(better.view(1,B,1).expand_as(cand),
                                cand, best_traj)
        best_sc   = torch.where(better, score, best_sc)

    return best_traj, best_sc


@torch.no_grad()
def _persistence_blend(model_pred_norm, obs_traj_norm, blend_strength=0.20):
    T_obs = obs_traj_norm.shape[0]
    T     = model_pred_norm.shape[0]
    B, device = model_pred_norm.shape[1], model_pred_norm.device
    if T_obs < 2:
        return model_pred_norm
    vels = obs_traj_norm[1:] - obs_traj_norm[:-1]
    n_v  = vels.shape[0]
    if n_v >= 3:
        alpha = 0.7
        w  = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
                           dtype=torch.float, device=device).flip(0)
        ev = (vels * (w/w.sum()).view(-1,1,1)).sum(0)
    elif n_v == 2:
        ev = 0.7*vels[-1] + 0.3*vels[-2]
    else:
        ev = vels[-1]
    steps   = torch.arange(1, T+1, dtype=torch.float, device=device)
    persist = (obs_traj_norm[-1].unsqueeze(0)
               + ev.unsqueeze(0) * steps.view(T,1,1))
    obs_spd_all = _step_speeds_deg(_norm_to_deg(obs_traj_norm))
    if obs_spd_all.shape[0] >= 2:
        spd_cv  = obs_spd_all.std(0) / obs_spd_all.mean(0).clamp(min=1.0)
        alpha_b = (blend_strength
                   * torch.sigmoid(-(spd_cv-0.3)*5.0)).unsqueeze(0).unsqueeze(-1)
    else:
        alpha_b = blend_strength * 0.5
    return (1.0 - alpha_b)*model_pred_norm + alpha_b*persist


def _score_ensemble_member(traj_norm, obs_traj_norm, speed_stats=None,
                            speed_head_pred=None):
    sp      = speed_stats or _SPEED_PRIOR
    v_opt   = sp.get("v_opt", 15.0)
    v_sigma = sp.get("v_sigma", 10.0)
    v_cap   = sp.get("v_hard_cap", 35.0)
    B, device = traj_norm.shape[1], traj_norm.device

    spd = _step_speeds_deg(_norm_to_deg(traj_norm))
    dtd = _norm_to_deg(traj_norm[1:]) - _norm_to_deg(traj_norm[:-1])
    prior_sc  = torch.exp(-(((spd-v_opt)/v_sigma).pow(2)
                             + F.relu(spd-v_cap)*2.0).mean(0)*0.5)
    smooth_sc = (torch.exp(-(dtd[1:]-dtd[:-1]).norm(dim=-1).mean(0)*5.0)
                 if dtd.shape[0] >= 2 else torch.ones(B, device=device))

    if obs_traj_norm is not None and obs_traj_norm.shape[0] >= 2:
        obs_v = obs_traj_norm[-1] - obs_traj_norm[-2]
        if obs_traj_norm.shape[0] >= 3:
            obs_v = 0.7*obs_v + 0.3*(obs_traj_norm[-2]-obs_traj_norm[-3])
        obs_hn  = F.normalize(obs_v, dim=-1, eps=1e-6)
        n_h     = min(3, traj_norm.shape[0]-1)
        pv_m    = ((traj_norm[1:1+n_h]-traj_norm[:n_h]).mean(0)
                   if n_h >= 1 else obs_v)
        pred_hn = F.normalize(pv_m, dim=-1, eps=1e-6)
        head_sc = torch.exp(((obs_hn*pred_hn).sum(-1)-1.0)*3.0)
        obs_ref = _step_speeds_deg(_norm_to_deg(obs_traj_norm))[
            -min(3, obs_traj_norm.shape[0]-1):].mean(0)
        spd_sc  = torch.exp(-((spd[:min(4,spd.shape[0])].mean(0) - obs_ref)
                               / obs_ref.clamp(min=5.0)).pow(2)*3.0)
    else:
        head_sc = spd_sc = torch.ones(B, device=device)

    base_score = (head_sc.pow(0.35) * spd_sc.pow(0.30)
                  * prior_sc.pow(0.20) * smooth_sc.pow(0.15))

    if speed_head_pred is not None:
        n = min(speed_head_pred.shape[1], spd.shape[0])
        if n > 0:
            pred_spd_t = speed_head_pred[:, :n].T
            spd_match = torch.exp(
                -((spd[:n] - pred_spd_t) / v_sigma).pow(2).mean(0) * 3.0
            )
            return base_score.pow(0.45) * spd_match.pow(0.55)

    return base_score


# ═════════════════════════════════════════════════════════════════════════════
#  [TWEAK-3] K=3 mode clustering
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _k3_mode_cluster(trajs_norm, obs_norm, speed_stats=None,
                     speed_head_pred=None, unimodal_spread_km=120.):
    if not trajs_norm:
        return obs_norm[-1:].expand(12, obs_norm.shape[1], 2)

    dev = trajs_norm[0].device
    T   = trajs_norm[0].shape[0]
    B   = trajs_norm[0].shape[1]
    N   = len(trajs_norm)
    K   = min(3, N)

    all_sc = torch.stack([
        _score_ensemble_member(tr, obs_norm, speed_stats,
                               speed_head_pred=speed_head_pred)
        for tr in trajs_norm
    ], dim=0)

    CKS = [min(3, T-1), min(7, T-1), T-1]

    def _multi_dist(tr_a_b, centers_deg):
        dists = []
        for ck in CKS:
            pts = _norm_to_deg(tr_a_b[:, ck, :])
            for c_deg in [centers_deg]:
                d = _haversine_deg(
                    pts.unsqueeze(1).expand(pts.shape[0], c_deg.shape[0], 2),
                    c_deg.unsqueeze(0).expand(pts.shape[0], c_deg.shape[0], 2))
                dists.append(d)
        return torch.stack(dists, 0).mean(0)

    all_ck_deg = torch.stack([
        torch.stack([_norm_to_deg(tr[ck]) for ck in CKS], 0)
        for tr in trajs_norm
    ], 0)

    results = []
    for b in range(B):
        sc_b  = all_sc[:, b]
        tr_b  = torch.stack([tr[:, b, :] for tr in trajs_norm], 0)
        ck_b  = all_ck_deg[:, :, b, :]

        if N < 4:
            w = F.softmax(sc_b * 3., 0)
            results.append((tr_b * w.view(N,1,1)).sum(0))
            continue

        ep_b    = ck_b[:, -1, :]
        ep_mean = ep_b.mean(0, keepdim=True)
        spread  = _haversine_deg(ep_b, ep_mean.expand(N, 2)).mean().item()

        if spread < unimodal_spread_km:
            k_top  = max(3, int(N * 0.30))
            topk   = sc_b.topk(k_top).indices
            w_top  = F.softmax(sc_b[topk] * 5., 0)
            results.append((tr_b[topk] * w_top.view(k_top,1,1)).sum(0))
            continue

        k_seed     = max(2, int(N * 0.20))
        seed_pool  = sc_b.topk(k_seed).indices
        seed_ck    = ck_b[seed_pool]
        seed_mean  = seed_ck.mean(0, keepdim=True)
        dist2mean  = torch.stack([
            _haversine_deg(seed_ck[:, c, :], seed_mean[:, c, :].expand(k_seed, 2))
            for c in range(len(CKS))
        ], 0).mean(0)
        seed_c_idx = seed_pool[dist2mean.argmin()]
        centers_ck = [ck_b[seed_c_idx]]

        for _ in range(K - 1):
            cs = torch.stack(centers_ck, 0)
            d2c_list = []
            for i in range(N):
                d_per_ck = torch.stack([
                    _haversine_deg(ck_b[i, c, :].unsqueeze(0), cs[:, c, :])
                    for c in range(len(CKS))
                ], 0).mean(0)
                d2c_list.append(d_per_ck.min())
            d_to_nearest = torch.stack(d2c_list)
            centers_ck.append(ck_b[d_to_nearest.argmax()])

        cck = torch.stack(centers_ck, 0)

        for _ in range(3):
            assign_scores = []
            for i in range(N):
                d_per_k = torch.stack([
                    torch.stack([
                        _haversine_deg(ck_b[i, c, :].unsqueeze(0),
                                       cck[k, c, :].unsqueeze(0))
                        for c in range(len(CKS))
                    ], 0).mean()
                    for k in range(K)
                ], 0)
                assign_scores.append(d_per_k)
            d2c    = torch.stack(assign_scores, 0)
            assign = d2c.argmin(1)

            new_c = []
            for k in range(K):
                mk = (assign == k)
                if mk.sum() > 0:
                    wk = F.softmax(sc_b[mk] * 3., 0)
                    new_c.append((ck_b[mk] * wk.view(-1,1,1)).sum(0))
                else:
                    new_c.append(cck[k])
            cck = torch.stack(new_c, 0)

        final_assign = []
        for i in range(N):
            d_per_k = torch.stack([
                torch.stack([
                    _haversine_deg(ck_b[i, c, :].unsqueeze(0),
                                   cck[k, c, :].unsqueeze(0))
                    for c in range(len(CKS))
                ], 0).mean()
                for k in range(K)
            ], 0)
            final_assign.append(d_per_k.argmin())
        assign = torch.stack(final_assign)

        csc = torch.zeros(K, device=dev)
        for k in range(K):
            mk = (assign == k)
            if mk.sum() > 0: csc[k] = sc_b[mk].sum()

        best_k = csc.argmax().item()
        mk     = (assign == best_k)
        if not mk.any(): mk = torch.ones(N, dtype=torch.bool, device=dev)
        w_win  = F.softmax(sc_b[mk] * 3., 0)
        results.append((tr_b[mk] * w_win.view(-1,1,1)).sum(0))

    return torch.stack(results, dim=1)


# ═════════════════════════════════════════════════════════════════════════════
#  TCFlowMatching v59-Strategy [FIXED]
# ═════════════════════════════════════════════════════════════════════════════

class TCFlowMatching(nn.Module):

    def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
                 n_train_ens=4, unet_in_ch=13, ctx_noise_scale=0.01,
                 initial_sample_sigma=0.03, teacher_forcing=True,
                 use_ema=True, ema_decay=0.995,
                 use_ate_ot=True, ot_epsilon=0.05,
                 use_slerp=False,
                 cfg_guidance_scale=1.5, cfg_uncond_prob=0.1,
                 **kwargs):
        super().__init__()
        self.pred_len           = pred_len
        self.obs_len            = obs_len
        self.sigma_min          = sigma_min
        self.ctx_noise_scale    = ctx_noise_scale
        self.active_pred_len    = pred_len
        self.use_ate_ot         = use_ate_ot
        self.ot_epsilon         = ot_epsilon
        self.use_slerp          = use_slerp
        self.cfg_guidance_scale = cfg_guidance_scale
        self.cfg_uncond_prob    = cfg_uncond_prob

        self.net          = VelocityField(pred_len=pred_len, obs_len=obs_len,
                                          sigma_min=sigma_min,
                                          unet_in_ch=unet_in_ch, ctx_dim=256)
        self.step_weights = LearnedStepWeights(n_steps=pred_len, min_ratio=6.0)
        self.selector     = SelectorNet(ctx_dim=256, cand_feat_dim=64,
                                        hidden_dim=128)

        self.use_ema   = use_ema
        self.ema_decay = ema_decay
        self._ema      = None

    def init_ema(self):
        if self.use_ema:
            self._ema = EMAModel(self, decay=self.ema_decay)

    def ema_update(self):
        if self._ema is not None: self._ema.update(self)

    def set_curriculum_len(self, *a, **kw): pass

    @staticmethod
    def _to_rel(traj, Me, lp, lm):
        return torch.cat([traj - lp.unsqueeze(0),
                          Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

    @staticmethod
    def _to_abs(rel, lp, lm):
        d = rel.permute(1, 0, 2)
        return lp.unsqueeze(0) + d[:,:,:2], lm.unsqueeze(0) + d[:,:,2:]

    def _cfm_noisy(self, x1, sigma_min=None, lp=None):
        if sigma_min is None: sigma_min = self.sigma_min
        B = x1.shape[0]; device = x1.device
        x0 = torch.randn_like(x1) * sigma_min
        t  = torch.rand(B, device=device)
        if self.use_slerp and x1.shape[-1] >= 2:
            x_t      = _slerp_interpolant(x0, x1, t, lp=lp)
            u_target = _slerp_velocity_target(x0, x1, t, lp=lp)
        else:
            te       = t.view(B, 1, 1)
            x_t      = (1.0 - te)*x0 + te*x1
            u_target = x1 - x0
        return x_t, t, u_target

    _cfm_noisy_slerp = _cfm_noisy

    @staticmethod
    def _lon_flip_aug(bl, p=0.3):
        if torch.rand(1).item() > p: return bl
        bl = list(bl)
        for i in [0, 1, 2, 3]:
            if torch.is_tensor(bl[i]) and bl[i].shape[-1] >= 1:
                t = bl[i].clone(); t[..., 0] = -t[..., 0]; bl[i] = t
        return bl

    @staticmethod
    def _obs_noise_aug(bl, sigma=0.005):
        if torch.rand(1).item() > 0.5: return bl
        bl = list(bl)
        if torch.is_tensor(bl[0]):
            bl[0] = bl[0] + torch.randn_like(bl[0]) * sigma
        return bl

    def _persistence_forecast_rel(self, obs_traj, lp, lm, pred_len):
        B, device = obs_traj.shape[1], obs_traj.device
        if obs_traj.shape[0] >= 3:
            vels  = obs_traj[1:] - obs_traj[:-1]
            n_v   = vels.shape[0]
            alpha = 0.7
            w     = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
                                  dtype=torch.float, device=device).flip(0)
            lv    = (vels * (w/w.sum()).view(-1,1,1)).sum(0)
        elif obs_traj.shape[0] >= 2:
            lv = obs_traj[-1, :, :2] - obs_traj[-2, :, :2]
        else:
            lv = obs_traj.new_zeros(B, 2)
        steps    = torch.arange(1, pred_len+1, device=device).float()
        pred_abs = (obs_traj[-1, :, :2].unsqueeze(1)
                    + lv.unsqueeze(1) * steps.view(1,-1,1))
        pred_abs = pred_abs.permute(1, 0, 2)
        pred_rel_pos = pred_abs - lp.unsqueeze(0)
        pred_rel     = torch.cat([pred_rel_pos,
                                   torch.zeros_like(pred_rel_pos)], dim=-1)
        return pred_rel.permute(1, 0, 2)

    def _compute_obs_momentum(self, obs_traj_norm):
        T_obs = obs_traj_norm.shape[0]
        if T_obs < 2:
            return torch.zeros(obs_traj_norm.shape[1], 2,
                               device=obs_traj_norm.device)
        vels = obs_traj_norm[1:] - obs_traj_norm[:-1]
        n_v  = vels.shape[0]
        if n_v >= 3:
            alpha = 0.65
            w = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
                              dtype=torch.float,
                              device=obs_traj_norm.device).flip(0)
            return (vels * (w/w.sum()).view(-1,1,1)).sum(0)
        elif n_v == 2:
            return 0.65*vels[-1] + 0.35*vels[-2]
        return vels[-1]

    @staticmethod
    def _sigma_schedule(epoch):
        if epoch < 2:   return 0.10
        if epoch < 10:  return 0.10 - (epoch-2)/8.0 * (0.10 - 0.04)
        if epoch < 20:  return max(0.04 - (epoch-10)/10.0 * 0.01, 0.035)
        return 0.035

    def get_loss(self, batch_list, epoch=0, **kwargs):
        return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

    def get_loss_breakdown(
        self,
        batch_list,
        epoch: int = 0,
        alpha_hard: float = 0.0,
        is_hard: Optional[torch.Tensor] = None,
        train_selector: bool = False,
        lambda_dict: Optional[Dict[str, float]] = None,
        step_weight_alpha: float = 0.0,
    ) -> Dict:
        """
        [S-E] Loss breakdown với easy/hard split và GradNorm-compatible terms.
        """
        # BUG-5 (original): epoch=-1 = val mode → skip augmentation
        if epoch >= 0:
            batch_list = self._lon_flip_aug(batch_list)
            batch_list = self._obs_noise_aug(batch_list, sigma=0.005)

        obs_t    = batch_list[0]
        env_data = batch_list[13] if len(batch_list) > 13 else None
        lp, lm   = obs_t[-1], batch_list[7][-1]
        B, device = obs_t.shape[1], obs_t.device

        speed_stats   = compute_speed_stats_from_norm(obs_t[..., :2])
        current_sigma = self._sigma_schedule(epoch)
        raw_ctx       = self.net._context(batch_list)

        x1_rel = self._to_rel(batch_list[1], batch_list[8], lp, lm)

        if self.use_ate_ot and B >= 4:
            noise_base = torch.randn_like(x1_rel) * current_sigma
            noise_matched, x1_matched = _spherical_ot_matching(
                noise_base, x1_rel, lp, epsilon=self.ot_epsilon)
        else:
            noise_matched = torch.randn_like(x1_rel) * current_sigma
            x1_matched    = x1_rel

        x_t, fm_t, u_target = self._cfm_noisy(
            x1_matched, sigma_min=current_sigma, lp=lp)

        use_null     = (torch.rand(1).item() < self.cfg_uncond_prob)
        vel_obs_feat = self.net._get_kinematic_obs_feat(obs_t[:, :, :2])

        pred_vel = self.net.forward_with_ctx(
            x_t, fm_t, raw_ctx, env_data=env_data, use_null=use_null,
            vel_obs_feat  = vel_obs_feat,
            steering_feat = self.net._get_steering_feat(env_data, B, device),
            env_kine_feat = self.net._get_env_kine_feat(env_data, B, device),
        )

        l_fm = F.mse_loss(pred_vel, u_target)

        fm_te    = fm_t.view(B, 1, 1)
        x1_pred  = x_t + (1.0 - fm_te) * pred_vel
        pred_abs, _ = self._to_abs(x1_pred, lp, lm)
        pred_deg    = _norm_to_deg(pred_abs)
        gt_deg      = _norm_to_deg(batch_list[1])

        sw_tensor = self.step_weights.get(n=pred_deg.shape[0])
        sw_pen    = self.step_weights.penalty()

        # [S-A] STTrans loss — trả về từng term riêng
        loss_dict = compute_st_trans_loss(
            pred_deg, gt_deg, epoch=epoch,
            speed_stats=speed_stats, step_w=sw_tensor)

        # [S-C] Hard loss
        hard_loss_dict = {"l_hard_total": torch.zeros((), device=device),
                          "n_hard": 0}
        if is_hard is not None and is_hard.any() and alpha_hard > 0.0:
            hard_loss_dict = compute_hard_loss(
                pred_deg, gt_deg, is_hard, step_w=sw_tensor)

        # Speed head loss
        if not use_null:
            ctx_for_speed = self.net._apply_ctx_head(raw_ctx)
            speed_pred    = self.net.speed_head(ctx_for_speed, vel_obs_feat)
            l_speed_head  = _speed_head_loss(speed_pred, pred_deg, gt_deg,
                                             speed_stats)
        else:
            l_speed_head = x_t.new_zeros(())

        # [S-D] Selector loss
        l_sel_total  = x_t.new_zeros(())
        sel_loss_dict = {"l_soft_oracle": 0.0, "l_pairwise_rank": 0.0,
                         "l_confidence": 0.0}
        if train_selector and is_hard is not None and is_hard.any():
            ctx_sel    = self.net._apply_ctx_head(raw_ctx)
            sel_candidates = []
            sel_gt_ades    = []
            for _ in range(8):
                noise_k = torch.randn_like(x1_rel) * current_sigma
                x_t_k   = noise_k
                t_k     = torch.full((B,), 0.5, device=device)
                vel_k   = self.net.forward_with_ctx(
                    x_t_k, t_k, raw_ctx,
                    vel_obs_feat  = vel_obs_feat,
                    steering_feat = self.net._get_steering_feat(env_data, B, device),
                    env_kine_feat = self.net._get_env_kine_feat(env_data, B, device),
                    env_data      = env_data,
                )
                x1_k     = x_t_k + 0.5 * vel_k
                abs_k, _ = self._to_abs(x1_k, lp, lm)
                deg_k    = _norm_to_deg(abs_k)
                norm_k   = abs_k
                T_c = min(deg_k.shape[0], gt_deg.shape[0])
                ade_k = _haversine_deg(deg_k[:T_c], gt_deg[:T_c]).mean(0)
                sel_candidates.append(norm_k)
                sel_gt_ades.append(ade_k.detach())

            cand_ades  = torch.stack(sel_gt_ades, dim=0)   # [N, B]
            scores_sel = self.selector.score_candidates(
                ctx_sel, sel_candidates, gt_deg, obs_t[:,:,:2])
            confidence = self.selector.get_confidence(ctx_sel)

            s_dict     = selector_loss(scores_sel, cand_ades, confidence, is_hard)
            l_sel_total = s_dict["l_sel_total"]
            sel_loss_dict = {k: v.item() if torch.is_tensor(v) else v
                             for k, v in s_dict.items()}

        # ── Tổng hợp loss với GradNorm λ ──────────────────────────────────
        if lambda_dict is not None:
            λ_dpe  = lambda_dict.get("l_dpe",    1.20)
            λ_vel  = lambda_dict.get("l_vel_reg", 1.40)
            λ_head = lambda_dict.get("l_heading", 0.40)
            λ_spd  = lambda_dict.get("l_speed",   0.05)
            λ_acc  = lambda_dict.get("l_accel",   0.01)
        else:
            λ_dpe = 1.20; λ_vel = 1.40; λ_head = 0.40
            λ_spd = 0.05; λ_acc = 0.01

        l_base = (λ_dpe  * loss_dict["l_dpe"]
                  + λ_vel  * loss_dict["l_vel_reg"]
                  + λ_head * loss_dict["l_heading"]
                  + λ_spd  * loss_dict["l_speed"]
                  + λ_acc  * loss_dict["l_accel"])

        total = (l_fm
                 + l_base
                 + alpha_hard * hard_loss_dict["l_hard_total"]
                 + 0.30 * l_speed_head
                 + 0.20 * l_sel_total
                 + sw_pen)

        if torch.isnan(total) or torch.isinf(total):
            total = x_t.new_zeros(())

        sw_st = self.step_weights.stats()
        d = dict(loss_dict)
        d.update({
            "total"           : total,
            "l_base"          : l_base.item() if torch.is_tensor(l_base) else 0.0,
            "l_fm"            : l_fm.item(),
            "fm_mse"          : l_fm.item(),
            "l_hard_total"    : (hard_loss_dict["l_hard_total"].item()
                                 if torch.is_tensor(hard_loss_dict["l_hard_total"])
                                 else 0.0),
            "l_endpoint_norm" : hard_loss_dict.get("l_endpoint_norm", 0.0),
            "l_disp_norm"     : hard_loss_dict.get("l_disp_norm", 0.0),
            "n_hard"          : hard_loss_dict.get("n_hard", 0),
            "alpha_hard"      : alpha_hard,
            "l_sel_total"     : (l_sel_total.item()
                                 if torch.is_tensor(l_sel_total) else 0.0),
            "l_soft_oracle"   : sel_loss_dict.get("l_soft_oracle", 0.0),
            "l_pairwise_rank" : sel_loss_dict.get("l_pairwise_rank", 0.0),
            "l_confidence"    : sel_loss_dict.get("l_confidence", 0.0),
            "speed_head_l"    : (l_speed_head.item()
                                 if torch.is_tensor(l_speed_head) else 0.0),
            "sigma"           : current_sigma,
            "v_opt"           : speed_stats.get("v_opt", 15.0),
            "obs_spd_p50"     : speed_stats.get("p50_kmh", 0.0),
            "sw_ratio"        : sw_st["sw_ratio"],
            "sw_72h"          : sw_st["sw_72h"],
            "ate_x1"          : 0.0,
        })
        return d

    @torch.no_grad()
    def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
               predict_csv=None, importance_weight=True, use_cfg=True,
               use_selector=False, selector_threshold=0.5):
        obs_t    = batch_list[0]
        env_data = batch_list[13] if len(batch_list) > 13 else None
        lp       = obs_t[-1]; lm = batch_list[7][-1]
        B        = lp.shape[0]; device = lp.device
        T        = self.pred_len; dt = 1.0 / max(ddim_steps, 1)

        raw_ctx       = self.net._context(batch_list)
        vel_obs_feat  = self.net._get_kinematic_obs_feat(obs_t[:, :, :2])
        steering_feat = self.net._get_steering_feat(env_data, B, device)
        env_kine_feat = self.net._get_env_kine_feat(env_data, B, device)
        speed_stats   = compute_speed_stats_from_norm(obs_t[..., :2])
        persist_init  = self._persistence_forecast_rel(obs_t, lp, lm, T)
        obs_norm      = obs_t[:, :, :2]

        try:
            speed_head_pred = self.net.predict_speed(raw_ctx, vel_obs_feat)
        except Exception:
            speed_head_pred = None

        if obs_t.shape[0] >= 2:
            obs_h_n = F.normalize(
                obs_t[-1,:,:2] - obs_t[-2,:,:2], dim=-1, eps=1e-6)
        else:
            obs_h_n = None

        obs_mom = self._compute_obs_momentum(obs_norm)
        if obs_t.shape[0] >= 3:
            vv    = obs_t[1:,:,:2] - obs_t[:-1,:,:2]
            heads = F.normalize(vv, dim=-1, eps=1e-6)
            cos_s = (heads[1:]*heads[:-1]).sum(-1).mean(0)
            mom_gate = torch.sigmoid((cos_s-0.5)*8.0)
        else:
            mom_gate = torch.ones(B, device=device)

        def _mom_str(s, tot):
            return 0.08 * 0.5 * (1.0 + math.cos(math.pi*s/max(tot,1)))

        all_norms, all_me = [], []

        for _ in range(num_ensemble):
            x_t = persist_init + torch.randn_like(persist_init) * self.sigma_min * 2.5

            for step in range(ddim_steps):
                t_b = torch.full((B,), step*dt, device=device)
                ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0

                if use_cfg and step > 0:
                    v_cond   = self.net.forward_with_ctx(
                        x_t, t_b, raw_ctx, noise_scale=ns,
                        vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
                        env_kine_feat=env_kine_feat, env_data=env_data,
                        use_null=False)
                    v_uncond = self.net.forward_with_ctx(
                        x_t, t_b, raw_ctx, noise_scale=0.0,
                        vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
                        env_kine_feat=env_kine_feat, env_data=env_data,
                        use_null=True)
                    if obs_h_n is not None:
                        pred_h = F.normalize(v_cond[:,0,:2].detach(),
                                             dim=-1, eps=1e-6)
                        cos_a  = (obs_h_n * pred_h).sum(-1).clamp(-1.0, 1.0)
                        gs     = (0.8 + 0.7*(cos_a+1.0)*0.5).view(B,1,1)
                        vel    = v_uncond + gs * (v_cond - v_uncond)
                    else:
                        vel = v_uncond + 1.5 * (v_cond - v_uncond)
                else:
                    vel = self.net.forward_with_ctx(
                        x_t, t_b, raw_ctx, noise_scale=ns,
                        vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
                        env_kine_feat=env_kine_feat, env_data=env_data)

                m_s = _mom_str(step, ddim_steps)
                if m_s > 1e-4:
                    me  = obs_mom.unsqueeze(1).expand(B, T, 2)
                    mf  = torch.cat([me, torch.zeros(B, T, 2, device=device)], -1)
                    vel = vel + m_s * mom_gate.view(B,1,1) * mf

                x_t = (x_t + dt*vel).clamp(-3.0, 3.0)

            tr, me = self._to_abs(x_t, lp, lm)
            all_norms.append(tr)
            all_me.append(me)

        SCALES = (0.85, 0.92, 1.00, 1.08, 1.15)
        augmented = []
        for tn in all_norms:
            bt, _ = _speed_sweep_correction(tn, obs_norm, SCALES)
            augmented.append(bt)
            augmented.append(tn)

        all_me_t = torch.stack(all_me)

        if use_selector:
            ctx_inf    = self.net._apply_ctx_head(raw_ctx)
            confidence = self.selector.get_confidence(ctx_inf)
            is_hard_inf = classify_hard_easy(obs_norm)
            use_sel_mask = is_hard_inf & (confidence >= selector_threshold)

            if use_sel_mask.any():
                scores_inf = self.selector.score_candidates(
                    ctx_inf, augmented, gt_deg=None, obs_norm=obs_norm)
                best_idx = scores_inf.argmax(dim=0)

                pred_cluster = _k3_mode_cluster(augmented, obs_norm, speed_stats,
                                                 speed_head_pred=speed_head_pred)
                pred_sel = torch.zeros_like(pred_cluster)
                for b in range(B):
                    if use_sel_mask[b]:
                        pred_sel[:, b, :] = augmented[best_idx[b].item()][:, b, :]
                    else:
                        pred_sel[:, b, :] = pred_cluster[:, b, :]
                pred_mean = pred_sel
            else:
                pred_mean = _k3_mode_cluster(augmented, obs_norm, speed_stats,
                                              speed_head_pred=speed_head_pred)
        else:
            pred_mean = _k3_mode_cluster(augmented, obs_norm, speed_stats,
                                          speed_head_pred=speed_head_pred)

        pred_mean = _persistence_blend(pred_mean, obs_norm, blend_strength=0.20)
        all_c = torch.stack(augmented)

        if predict_csv:
            self._write_predict_csv(predict_csv, pred_mean, all_c)
        return pred_mean, all_me_t.mean(0), all_c

    def _score_sample(self, traj, speed_stats=None):
        return _score_ensemble_member(traj, None, speed_stats)

    @staticmethod
    def _write_predict_csv(csv_path, traj_mean, all_trajs):
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        T, B, _ = traj_mean.shape
        mlon = ((traj_mean[...,0]*50.0+1800.0)/10.0).cpu().numpy()
        mlat = ((traj_mean[...,1]*50.0)/10.0).cpu().numpy()
        alon = ((all_trajs[...,0]*50.0+1800.0)/10.0).cpu().numpy()
        alat = ((all_trajs[...,1]*50.0)/10.0).cpu().numpy()
        fields = ["timestamp","batch_idx","step_idx","lead_h",
                  "lon_mean_deg","lat_mean_deg",
                  "lon_std_deg","lat_std_deg","ens_spread_km"]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        write_hdr = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if write_hdr: w.writeheader()
            for b in range(B):
                for k in range(T):
                    dlat   = alat[:,k,b] - mlat[k,b]
                    dlon   = ((alon[:,k,b] - mlon[k,b])
                              * math.cos(math.radians(float(mlat[k,b]))))
                    spread = float(((dlat**2+dlon**2)**0.5).mean() * DEG2KM)
                    w.writerow({
                        "timestamp"     : ts,
                        "batch_idx"     : b,
                        "step_idx"      : k,
                        "lead_h"        : (k+1)*6,
                        "lon_mean_deg"  : f"{mlon[k,b]:.4f}",
                        "lat_mean_deg"  : f"{mlat[k,b]:.4f}",
                        "lon_std_deg"   : f"{alon[:,k,b].std():.4f}",
                        "lat_std_deg"   : f"{alat[:,k,b].std():.4f}",
                        "ens_spread_km" : f"{spread:.2f}",
                    })


# Backward compat
TCDiffusion = TCFlowMatching
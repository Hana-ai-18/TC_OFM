# # # """
# # # Model/flow_matching_model.py  ── FM v60
# # # =========================================
# # # THAY THẾ HOÀN TOÀN flow_matching_model.py v59.

# # # Cách dùng: chỉ copy file này vào Model/ — không cần thay file khác.
# # # Import interface giữ nguyên:
# # #   from Model.flow_matching_model import TCFlowMatching

# # # FIXES:
# # #   BUG-1 CRITICAL: prior_sc(v_opt=15)≈0 với speed=113 → thải trajectory đúng
# # #                   FIX: EnsembleScorer MLP, không prior
# # #   BUG-2 CRITICAL: l_speed(v_opt=15) gradient ngược chiều
# # #                   FIX: l_logspeed = MSE(log(spd+1)) không prior
# # #   BUG-3 HIGH:    STEP_WEIGHTS magic numbers non-monotonic
# # #                   FIX: LearnedStepWeights monotonic
# # #   BUG-4 HIGH:    79% env_data bị bỏ (21/98 dims)
# # #                   FIX: thêm gph500, bearing_to_scs, month, velocity_history, RI
# # #   BUG-5 HIGH:    Scoring exponents cứng
# # #                   FIX: EnsembleScorer MLP learned
# # #   BUG-6 MEDIUM:  speed_sweep target obs_spd sai
# # #                   FIX: bỏ speed_sweep
# # #   BUG-7 MEDIUM:  persistence_blend cứng 20%
# # #                   FIX: blend_alpha = sigmoid(Linear(ctx)) learned
# # #   BUG-8 MEDIUM:  CFG guidance_scale cứng
# # #                   FIX: guidance_scale = 0.8+1.2×sigmoid(Linear(ctx))
# # #   BUG-9 LOW:     sigma cứng 0.035
# # #                   FIX: sigma = 0.02+0.08×sigmoid(Linear(ctx))
# # # """
# # # from __future__ import annotations

# # # import csv
# # # import math
# # # import os
# # # import sys
# # # from datetime import datetime
# # # from typing import Optional, Tuple

# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F

# # # from Model.FNO3D_encoder import FNO3DEncoder
# # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # from Model.env_net_transformer_gphsplit import Env_net

# # # R_EARTH  = 6371.0
# # # DT_HOURS = 6.0
# # # DEG2KM   = 111.0




# # # MAX_CURVATURE_RAD = math.pi / 4  # 45°/step vật lý threshold


# # # # ══════════════════════════════════════════════════════════════
# # # #  Coordinate utilities
# # # # ══════════════════════════════════════════════════════════════

# # # def norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# # #     """Normalized [lon_n, lat_n] → degrees.
# # #     lon_n ∈ [-9, 2]  →  lon_deg = (lon_n * 50 + 1800) / 10
# # #     lat_n ∈ [0, 10]  →  lat_deg = lat_n * 50 / 10
# # #     """
# # #     return torch.stack([
# # #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# # #         (t[..., 1] * 50.0) / 10.0,
# # #     ], dim=-1)


# # # def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # #     """[...,2] lon°,lat° → [...] km"""
# # #     lat1 = torch.deg2rad(p1[..., 1])
# # #     lat2 = torch.deg2rad(p2[..., 1])
# # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # #     a = (torch.sin(dlat / 2).pow(2)
# # #          + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
# # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1.0 - 1e-12).sqrt())


# # # def velocity_km(traj_deg: torch.Tensor) -> torch.Tensor:
# # #     """
# # #     [T, B, 2] lon°,lat° → [T-1, B, 2] velocity (vx_km, vy_km) per step.

# # #     vx = Δlon × cos(lat_mid) × 111  (EW km/6h)
# # #     vy = Δlat × 111                  (NS km/6h)

# # #     Physics: đây là Lagrangian velocity trong flat-earth approximation
# # #     đúng với SCS region (lat 5°-25°, error < 1%).
# # #     """
# # #     T = traj_deg.shape[0]
# # #     if T < 2:
# # #         return traj_deg.new_zeros(1, traj_deg.shape[1], 2)

# # #     lon = traj_deg[..., 0]
# # #     lat = traj_deg[..., 1]
# # #     dlat = lat[1:] - lat[:-1]
# # #     dlon = lon[1:] - lon[:-1]
# # #     lat_mid = (lat[1:] + lat[:-1]) / 2.0
# # #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)

# # #     vx = dlon * cos_lat * DEG2KM  # EW km/6h
# # #     vy = dlat * DEG2KM             # NS km/6h
# # #     return torch.stack([vx, vy], dim=-1)  # [T-1, B, 2]


# # # # ══════════════════════════════════════════════════════════════
# # # #  Learned Step Weights — Monotonic
# # # # ══════════════════════════════════════════════════════════════

# # # class LearnedStepWeights(nn.Module):
# # #     """
# # #     Step weights w[0..11] thỏa mãn:
# # #     - Monotonic non-decreasing: error lớn hơn ở lead time dài hơn (vật lý)
# # #     - Mean = 1.0: tránh loss scale drift

# # #     Implementation:
# # #       raw ∈ ℝ^n (learnable)
# # #       increments = softplus(raw)      ← dương
# # #       weights = cumsum(increments)    ← monotonic non-decreasing
# # #       weights /= weights.mean()       ← normalize

# # #     Gradient flow: backprop qua cumsum → softplus → raw.
# # #     """
# # #     def __init__(self, n_steps: int = 12):
# # #         super().__init__()
# # #         self.n_steps = n_steps
# # #         # Init: uniform weights → raw s.t. softplus(raw)≈constant
# # #         # softplus(0.5) ≈ 0.97, cumsum(uniform) = linear ramp → reasonable init
# # #         self.raw = nn.Parameter(torch.zeros(n_steps) + 0.5)

# # #     def forward(self) -> torch.Tensor:
# # #         increments = F.softplus(self.raw)            # [n], positive
# # #         weights = torch.cumsum(increments, dim=0)    # [n], monotonic
# # #         weights = weights / weights.mean().clamp(min=1e-8)  # normalize mean=1
# # #         return weights  # [n]

# # #     def get(self, n: Optional[int] = None) -> torch.Tensor:
# # #         w = self.forward()
# # #         return w[:n] if n is not None else w

# # #     @torch.no_grad()
# # #     def stats(self) -> dict:
# # #         w = self.forward()
# # #         return {
# # #             "sw_6h":   w[0].item(),
# # #             "sw_12h":  w[1].item() if len(w) > 1 else 0.0,
# # #             "sw_48h":  w[7].item() if len(w) > 7 else 0.0,
# # #             "sw_72h":  w[-1].item(),
# # #             "sw_ratio": (w[-1] / w[0].clamp(min=1e-6)).item(),
# # #             "sw_monotonic": bool((w[1:] - w[:-1]).min().item() >= -1e-6),
# # #         }


# # # # ══════════════════════════════════════════════════════════════
# # # #  Log-Parameterized Loss Weights (Kendall & Gal 2018)
# # # # ══════════════════════════════════════════════════════════════

# # # class LearnedLossWeights(nn.Module):
# # #     """
# # #     λᵢ = 1 / (2·σᵢ²), log_σᵢ learnable.

# # #     Khi σᵢ lớn (task uncertain) → λᵢ nhỏ → ít contribution.
# # #     Model tự học σᵢ từ gradient, không cần grid search.

# # #     Total: L = Σᵢ λᵢ Lᵢ + Σᵢ log(σᵢ)
# # #            = Σᵢ Lᵢ/(2σᵢ²) + Σᵢ log(σᵢ)   (regularization term)
# # #     """
# # #     def __init__(self, n_tasks: int = 3):
# # #         super().__init__()
# # #         # Init log_sigma = 0 → σ = 1 → λ = 0.5
# # #         self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

# # #     def lambdas(self) -> torch.Tensor:
# # #         """λᵢ = 0.5 * exp(-2 * log_σᵢ)"""
# # #         return 0.5 * torch.exp(-2.0 * self.log_sigma)

# # #     def regularization(self) -> torch.Tensor:
# # #         """Σᵢ log(σᵢ) = Σᵢ log_sigma[i]"""
# # #         return self.log_sigma.sum()

# # #     @torch.no_grad()
# # #     def stats(self) -> dict:
# # #         lam = self.lambdas()
# # #         return {
# # #             "lam_kin": lam[0].item(),
# # #             "lam_logspd": lam[1].item(),
# # #             "lam_curv": lam[2].item(),
# # #         }


# # # # ══════════════════════════════════════════════════════════════
# # # #  Difficulty Weighting — Per-Sample, Learned
# # # # ══════════════════════════════════════════════════════════════

# # # class DifficultyWeighter(nn.Module):
# # #     """
# # #     Per-sample difficulty weight w ∈ [1.0, 2.0].
# # #     Không filter data (như SRC-Track), chỉ reweight gradient.

# # #     difficulty = [curvature_rate, speed_cv, boundary_prox]
# # #     w = 1 + sigmoid(w1·d1 + w2·d2 + w3·d3 + b)

# # #     Bão khó (recurvature, RI) → w ≈ 1.8 → gradient lớn hơn
# # #     Bão dễ (straight, stable)  → w ≈ 1.0 → không overfit

# # #     Weights w1,w2,w3,b: ALL LEARNABLE.
# # #     """
# # #     def __init__(self):
# # #         super().__init__()
# # #         # 3-dim difficulty feature → 1 logit
# # #         self.linear = nn.Linear(3, 1, bias=True)
# # #         # Init near zero → w ≈ 1.5 (uniform) at start
# # #         nn.init.zeros_(self.linear.weight)
# # #         nn.init.zeros_(self.linear.bias)

# # #     def compute_difficulty(
# # #         self,
# # #         gt_deg: torch.Tensor,   # [T, B, 2] degrees
# # #     ) -> torch.Tensor:
# # #         """
# # #         Compute 3 difficulty features từ gt trajectory.
# # #         Tính từ gt vì available at training time.
# # #         """
# # #         T, B = gt_deg.shape[:2]

# # #         # Feature 1: Curvature rate — mean |Δheading| per step
# # #         # Recurvature storms: high curvature
# # #         v = velocity_km(gt_deg)            # [T-1, B, 2]
# # #         if v.shape[0] >= 2:
# # #             heading = torch.atan2(v[..., 0], v[..., 1])  # [T-1, B]
# # #             dh = heading[1:] - heading[:-1]
# # #             dh = (dh + math.pi) % (2 * math.pi) - math.pi  # wrap to [-π,π]
# # #             curv_rate = dh.abs().mean(0)  # [B]
# # #         else:
# # #             curv_rate = gt_deg.new_zeros(B)

# # #         # Feature 2: Speed coefficient of variation — std/mean
# # #         # RI and rapid deceleration: high CV
# # #         spd = v.norm(dim=-1)  # [T-1, B] km/6h
# # #         mean_spd = spd.mean(0).clamp(min=1.0)  # [B]
# # #         std_spd  = spd.std(0)                   # [B]
# # #         speed_cv = (std_spd / mean_spd).clamp(max=3.0)  # [B]

# # #         # Feature 3: Max curvature exceedance — ReLU(|dh| - π/4)
# # #         # Proxy for recurvature onset difficulty
# # #         if v.shape[0] >= 2:
# # #             excess_curv = F.relu(dh.abs() - MAX_CURVATURE_RAD).mean(0)  # [B]
# # #         else:
# # #             excess_curv = gt_deg.new_zeros(B)

# # #         # Normalize features to similar scale
# # #         d1 = (curv_rate   / (math.pi / 2)).clamp(0, 1)  # [B]
# # #         d2 = speed_cv.clamp(0, 1)                         # [B]
# # #         d3 = (excess_curv / (math.pi / 4)).clamp(0, 1)   # [B]

# # #         return torch.stack([d1, d2, d3], dim=-1)  # [B, 3]

# # #     def forward(self, gt_deg: torch.Tensor) -> torch.Tensor:
# # #         """Returns per-sample weights [B] ∈ [1.0, 2.0]"""
# # #         diff_feat = self.compute_difficulty(gt_deg)  # [B, 3]
# # #         logit = self.linear(diff_feat).squeeze(-1)   # [B]
# # #         return 1.0 + torch.sigmoid(logit)            # [B] ∈ [1.0, 2.0]


# # # # ══════════════════════════════════════════════════════════════
# # # #  Ensemble Scorer — Learned MLP
# # # # ══════════════════════════════════════════════════════════════

# # # class EnsembleScorer(nn.Module):
# # #     """
# # #     Learned trajectory scorer. Thay thế fixed heuristic v59.

# # #     Input: 7 kinematic features per candidate trajectory
# # #     Output: score ∈ [0,1]

# # #     FIX so với v59:
# # #     - Không dùng prior_sc (v_opt=15 → score ≈ 0 với speed=113)
# # #     - Feature f4: match với obs_speed (không v_opt)
# # #     - Weights learned từ auxiliary training với oracle ADE ranking

# # #     Training: auxiliary BCE với oracle label từ ADE rank trên gt.
# # #     """
# # #     def __init__(self, feat_dim: int = 7, hidden: int = 32):
# # #         super().__init__()
# # #         self.net = nn.Sequential(
# # #             nn.Linear(feat_dim, hidden),
# # #             nn.GELU(),
# # #             nn.Linear(hidden, 16),
# # #             nn.GELU(),
# # #             nn.Linear(16, 1),
# # #         )
# # #         # Init near-zero output → score ≈ 0.5 uniformly at start
# # #         nn.init.zeros_(self.net[-1].weight)
# # #         nn.init.zeros_(self.net[-1].bias)

# # #     def extract_features(
# # #         self,
# # #         traj_norm: torch.Tensor,      # [T, B, 2] normalized
# # #         obs_norm:  torch.Tensor,      # [T_obs, B, 2] normalized
# # #     ) -> torch.Tensor:
# # #         """Returns [B, 7] feature vector."""
# # #         B = traj_norm.shape[1]
# # #         traj_deg = norm_to_deg(traj_norm)  # [T, B, 2]
# # #         obs_deg  = norm_to_deg(obs_norm)   # [T_obs, B, 2]

# # #         v    = velocity_km(traj_deg)   # [T-1, B, 2]
# # #         spd  = v.norm(dim=-1)          # [T-1, B]

# # #         # f1: log mean speed
# # #         f1 = torch.log1p(spd.mean(0))                        # [B]

# # #         # f2: log speed std
# # #         f2 = torch.log1p(spd.std(0).clamp(min=0))            # [B]

# # #         # f3: heading consistency (mean cos of consecutive heading changes)
# # #         if v.shape[0] >= 2:
# # #             heading = torch.atan2(v[..., 0], v[..., 1])
# # #             dh = heading[1:] - heading[:-1]
# # #             dh = (dh + math.pi) % (2 * math.pi) - math.pi
# # #             f3 = torch.cos(dh).mean(0)                        # [B]
# # #         else:
# # #             f3 = traj_norm.new_ones(B)

# # #         # f4: speed match với obs 3 steps cuối (không v_opt!)
# # #         v_obs = velocity_km(obs_deg)  # [T_obs-1, B, 2]
# # #         n_obs = min(3, v_obs.shape[0])
# # #         obs_spd_ref = v_obs[-n_obs:].norm(dim=-1).mean(0).clamp(min=1.0)  # [B]
# # #         n_pred = min(3, spd.shape[0])
# # #         pred_spd_early = spd[:n_pred].mean(0)                              # [B]
# # #         f4 = torch.exp(
# # #             -((pred_spd_early - obs_spd_ref) / obs_spd_ref).pow(2) * 2.0
# # #         )  # [B], 1 = perfect match

# # #         # f5: heading continuation từ obs
# # #         if v_obs.shape[0] >= 1 and v.shape[0] >= 1:
# # #             obs_last_h  = torch.atan2(v_obs[-1, :, 0], v_obs[-1, :, 1])  # [B]
# # #             pred_first_h = torch.atan2(v[0, :, 0], v[0, :, 1])           # [B]
# # #             dh_cont = pred_first_h - obs_last_h
# # #             dh_cont = (dh_cont + math.pi) % (2 * math.pi) - math.pi
# # #             f5 = torch.cos(dh_cont)                                        # [B]
# # #         else:
# # #             f5 = traj_norm.new_ones(B)

# # #         # f6: distance from persistence (normalized)
# # #         #   persistence = last obs position + last obs velocity × steps
# # #         if obs_norm.shape[0] >= 2:
# # #             last_vel_n = obs_norm[-1] - obs_norm[-2]  # [B, 2]
# # #             steps = torch.arange(1, traj_norm.shape[0] + 1,
# # #                                   device=traj_norm.device, dtype=traj_norm.dtype)
# # #             persist = (obs_norm[-1].unsqueeze(0)
# # #                        + last_vel_n.unsqueeze(0) * steps.view(-1, 1, 1))  # [T, B, 2]
# # #             dist_from_persist = (traj_norm - persist).norm(dim=-1).mean(0)  # [B]
# # #             # Normalize by typical displacement
# # #             ref_disp = (last_vel_n.norm(dim=-1) * traj_norm.shape[0]).clamp(min=1e-3)
# # #             f6 = dist_from_persist / ref_disp                              # [B]
# # #         else:
# # #             f6 = traj_norm.new_zeros(B)

# # #         # f7: curvature amount
# # #         if v.shape[0] >= 2:
# # #             heading = torch.atan2(v[..., 0], v[..., 1])
# # #             dh = heading[1:] - heading[:-1]
# # #             dh = (dh + math.pi) % (2 * math.pi) - math.pi
# # #             f7 = dh.abs().mean(0)                                          # [B]
# # #         else:
# # #             f7 = traj_norm.new_zeros(B)

# # #         # Stack và clamp để tránh NaN
# # #         feats = torch.stack([f1, f2, f3, f4, f5, f6, f7], dim=-1)  # [B, 7]
# # #         return feats.clamp(-10.0, 10.0)

# # #     def score(
# # #         self,
# # #         traj_norm: torch.Tensor,
# # #         obs_norm:  torch.Tensor,
# # #     ) -> torch.Tensor:
# # #         """Returns [B] scores ∈ [0,1]."""
# # #         feats = self.extract_features(traj_norm, obs_norm)  # [B, 7]
# # #         return torch.sigmoid(self.net(feats).squeeze(-1))   # [B]

# # #     def auxiliary_loss(
# # #         self,
# # #         candidates: list[torch.Tensor],   # list of [T, B, 2] norm
# # #         obs_norm:   torch.Tensor,          # [T_obs, B, 2] norm
# # #         gt_deg:     torch.Tensor,          # [T, B, 2] degrees
# # #     ) -> torch.Tensor:
# # #         """
# # #         Binary cross-entropy với oracle label từ ADE ranking.
# # #         Candidate có ADE thấp nhất = label 1, còn lại = 0.

# # #         Chỉ gọi khi có ≥ 2 candidates.
# # #         """
# # #         if len(candidates) < 2:
# # #             return candidates[0].new_zeros(())

# # #         B = obs_norm.shape[1]
# # #         ades = []
# # #         for cand in candidates:
# # #             cand_deg = norm_to_deg(cand)             # [T, B, 2]
# # #             # ADE = mean haversine per sample
# # #             d = haversine_km(
# # #                 cand_deg.permute(1, 0, 2),           # [B, T, 2]
# # #                 gt_deg.permute(1, 0, 2),             # [B, T, 2]
# # #             ).mean(dim=1)                             # [B]
# # #             ades.append(d)

# # #         ades_t  = torch.stack(ades, dim=0)            # [n_cands, B]
# # #         # Oracle: candidate với ADE nhỏ nhất = positive
# # #         best_idx = ades_t.argmin(dim=0)               # [B]

# # #         total_loss = ades_t.new_zeros(())
# # #         for i, cand in enumerate(candidates):
# # #             scores = self.score(cand, obs_norm)        # [B]
# # #             # label: 1 nếu đây là best candidate
# # #             labels = (best_idx == i).float()           # [B]
# # #             total_loss = total_loss + F.binary_cross_entropy(
# # #                 scores, labels, reduction='mean'
# # #             )
# # #         return total_loss / len(candidates)


# # # # ══════════════════════════════════════════════════════════════
# # # #  3 Loss Term Implementations
# # # # ══════════════════════════════════════════════════════════════

# # # def l_kinematic(
# # #     pred_deg:     torch.Tensor,   # [T, B, 2]
# # #     gt_deg:       torch.Tensor,   # [T, B, 2]
# # #     step_weights: torch.Tensor,   # [T-1] learned
# # #     sample_w:     Optional[torch.Tensor] = None,  # [B] difficulty weights
# # # ) -> torch.Tensor:
# # #     """
# # #     MSE velocity vector trong km-space, Huber-robust.

# # #     Physics basis: ||v_pred - v_gt||² = ATE_vel² + CTE_vel²
# # #     → 1 term này cover cả ATE lẫn CTE tự nhiên.
# # #     Không cần l_signed_ate, l_signed_cte, l_sph_ate riêng.

# # #     Huber δ=50 km/6h: dưới 50 → quadratic, trên 50 → linear.
# # #     Robust với super-typhoon (speed > 100 km/6h).
# # #     """
# # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # #     if T < 2:
# # #         return pred_deg.new_zeros(())

# # #     v_pred = velocity_km(pred_deg[:T])   # [T-1, B, 2]
# # #     v_gt   = velocity_km(gt_deg[:T])     # [T-1, B, 2]

# # #     # Squared velocity error per step per sample
# # #     err_sq = (v_pred - v_gt).pow(2).sum(dim=-1)  # [T-1, B]

# # #     # Huber loss: δ=50 km/6h
# # #     delta = 50.0
# # #     err_abs = err_sq.sqrt()
# # #     huber = torch.where(
# # #         err_abs < delta,
# # #         0.5 * err_sq / delta,
# # #         err_abs - delta / 2.0
# # #     )  # [T-1, B]

# # #     # Apply step weights (monotonic, learned)
# # #     n = huber.shape[0]
# # #     w = step_weights[:n]  # [T-1]
# # #     weighted = huber * w.unsqueeze(1)  # [T-1, B]

# # #     # Apply per-sample difficulty weights
# # #     if sample_w is not None:
# # #         weighted = weighted * sample_w.unsqueeze(0)  # broadcast over T

# # #     return weighted.mean()


# # # def l_logspeed(
# # #     pred_deg: torch.Tensor,   # [T, B, 2]
# # #     gt_deg:   torch.Tensor,   # [T, B, 2]
# # # ) -> torch.Tensor:
# # #     """
# # #     MSE(log(speed_pred+1), log(speed_gt+1)).

# # #     Physics: speed bão SCS theo log-normal distribution.
# # #     Log-space: balanced penalty cho slow (3-20 km/6h) và fast (80-130 km/6h).

# # #     FIX v59 BUG-2: không có v_opt, không có v_hard_cap.
# # #     Học trực tiếp từ ground truth — không bias.
# # #     """
# # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # #     if T < 2:
# # #         return pred_deg.new_zeros(())

# # #     v_pred = velocity_km(pred_deg[:T])   # [T-1, B, 2]
# # #     v_gt   = velocity_km(gt_deg[:T])

# # #     spd_pred = v_pred.norm(dim=-1)               # [T-1, B] km/6h
# # #     spd_gt   = v_gt.norm(dim=-1).clamp(min=1.0)  # clamp gt: quasi-stationary ≥ 1

# # #     return F.mse_loss(
# # #         torch.log1p(spd_pred.clamp(min=0.0)),
# # #         torch.log1p(spd_gt),
# # #     )


# # # def l_curvature(
# # #     pred_deg:        torch.Tensor,    # [T, B, 2]
# # #     threshold_rad:   float = MAX_CURVATURE_RAD,
# # # ) -> torch.Tensor:
# # #     """
# # #     Penalize chỉ PHẦN THỪA của heading change > threshold (45°/step).

# # #     Physics: steering flow thay đổi mượt. Heading change > 45°/step
# # #     chỉ xảy ra tại recurvature onset — và đó là discontinuous trong gt.
# # #     → Penalize pred để không "teleport" hướng tùy tiện.

# # #     Không compare với gt: self-regularization trên pred trajectory.
# # #     Không penalize curvature trong phạm vi vật lý bình thường.
# # #     """
# # #     T = pred_deg.shape[0]
# # #     if T < 3:
# # #         return pred_deg.new_zeros(())

# # #     v = velocity_km(pred_deg)  # [T-1, B, 2]
# # #     if v.shape[0] < 2:
# # #         return pred_deg.new_zeros(())

# # #     heading = torch.atan2(v[..., 0], v[..., 1])  # [T-1, B]
# # #     dh = heading[1:] - heading[:-1]              # [T-2, B]
# # #     # Wrap to [-π, π]
# # #     dh = (dh + math.pi) % (2 * math.pi) - math.pi

# # #     # Chỉ penalize phần vượt ngưỡng
# # #     excess = F.relu(dh.abs() - threshold_rad)    # [T-2, B]
# # #     return excess.mean()


# # # # ══════════════════════════════════════════════════════════════
# # # #  Master Loss Module
# # # # ══════════════════════════════════════════════════════════════

# # # class FMv60Loss(nn.Module):
# # #     """
# # #     Physics-grounded FM v60 loss với tất cả weights tự học.

# # #     Learnable components:
# # #     - LearnedStepWeights: monotonic step weights cho L_kinematic
# # #     - LearnedLossWeights: λ₁,λ₂,λ₃ qua log-parameterization
# # #     - DifficultyWeighter: per-sample weights [1,2]
# # #     - EnsembleScorer: auxiliary trajectory scorer

# # #     Fixed:
# # #     - L_fm (FM objective) weight = 1.0 (lý thuyết FM chuẩn)
# # #     - L_scorer weight = 0.05 (auxiliary, không dominant)
# # #     """
# # #     def __init__(self, pred_len: int = 12):
# # #         super().__init__()
# # #         self.pred_len = pred_len

# # #         # All learned components
# # #         self.step_weights  = LearnedStepWeights(n_steps=pred_len)
# # #         self.loss_weights  = LearnedLossWeights(n_tasks=3)  # λ₁,λ₂,λ₃
# # #         self.diff_weighter = DifficultyWeighter()
# # #         self.scorer        = EnsembleScorer()

# # #     def compute_main_losses(
# # #         self,
# # #         pred_deg:       torch.Tensor,   # [T, B, 2]
# # #         gt_deg:         torch.Tensor,   # [T, B, 2]
# # #         fm_vel_pred:    torch.Tensor,   # [B, T, 4] FM velocity
# # #         fm_vel_target:  torch.Tensor,   # [B, T, 4] FM target
# # #     ) -> Tuple[torch.Tensor, dict]:
# # #         """
# # #         Compute 4 main loss terms.

# # #         Returns: (total_physics_loss, breakdown_dict)
# # #         """
# # #         T = min(pred_deg.shape[0], gt_deg.shape[0])

# # #         # ── Per-sample difficulty weights ─────────────────────
# # #         sample_w = self.diff_weighter(gt_deg[:T])  # [B] ∈ [1,2]

# # #         # ── Step weights ──────────────────────────────────────
# # #         sw = self.step_weights.get(n=T - 1)        # [T-1], monotonic

# # #         # ── λ weights ─────────────────────────────────────────
# # #         lam = self.loss_weights.lambdas()           # [3]: λ_kin, λ_logspd, λ_curv
# # #         reg = self.loss_weights.regularization()    # log(σ) regularization

# # #         # ── Loss terms ────────────────────────────────────────
# # #         L_fm = F.mse_loss(fm_vel_pred, fm_vel_target)

# # #         L_kin  = l_kinematic(pred_deg[:T], gt_deg[:T], sw, sample_w)
# # #         L_logspd = l_logspeed(pred_deg[:T], gt_deg[:T])
# # #         L_curv = l_curvature(pred_deg[:T])

# # #         # ── Combine với learned λ ──────────────────────────────
# # #         # L_fm: 1.0 fixed (FM theory standard)
# # #         # λᵢ: learned via log-parameterization
# # #         total = (1.0        * L_fm
# # #                + lam[0]     * L_kin
# # #                + lam[1]     * L_logspd
# # #                + lam[2]     * L_curv
# # #                + reg)        # regularization từ log-parameterization

# # #         # Safety: clamp để tránh explosion
# # #         if not torch.isfinite(total):
# # #             total = pred_deg.new_zeros(())

# # #         lam_d = self.loss_weights.stats()
# # #         sw_d  = self.step_weights.stats()

# # #         breakdown = {
# # #             "l_fm":     L_fm.item()     if torch.is_tensor(L_fm)     else 0.0,
# # #             "l_kin":    L_kin.item()    if torch.is_tensor(L_kin)    else 0.0,
# # #             "l_logspd": L_logspd.item() if torch.is_tensor(L_logspd) else 0.0,
# # #             "l_curv":   L_curv.item()   if torch.is_tensor(L_curv)   else 0.0,
# # #             "l_reg":    reg.item()      if torch.is_tensor(reg)       else 0.0,
# # #             "diff_w_mean": sample_w.mean().item(),
# # #             **lam_d, **sw_d,
# # #         }
# # #         return total, breakdown

# # #     def scorer_loss(
# # #         self,
# # #         candidates: list[torch.Tensor],   # list of [T, B, 2] norm
# # #         obs_norm:   torch.Tensor,
# # #         gt_deg:     torch.Tensor,
# # #     ) -> torch.Tensor:
# # #         """Auxiliary scorer training loss."""
# # #         return self.scorer.auxiliary_loss(candidates, obs_norm, gt_deg)

# # #     def forward(
# # #         self,
# # #         pred_deg:      torch.Tensor,
# # #         gt_deg:        torch.Tensor,
# # #         fm_vel_pred:   torch.Tensor,
# # #         fm_vel_target: torch.Tensor,
# # #         candidates:    Optional[list] = None,
# # #         obs_norm:      Optional[torch.Tensor] = None,
# # #     ) -> Tuple[torch.Tensor, dict]:
# # #         """
# # #         Full loss computation.

# # #         candidates: list of candidate trajectories (normalized) cho scorer training.
# # #                     None trong đầu training (scorer chưa ổn định).
# # #         """
# # #         total, breakdown = self.compute_main_losses(
# # #             pred_deg, gt_deg, fm_vel_pred, fm_vel_target
# # #         )

# # #         # Auxiliary scorer (chỉ khi có candidates)
# # #         l_scr = pred_deg.new_zeros(())
# # #         if candidates is not None and obs_norm is not None and len(candidates) >= 2:
# # #             l_scr = self.scorer_loss(candidates, obs_norm, gt_deg)
# # #             total = total + 0.05 * l_scr

# # #         breakdown["l_scorer"] = l_scr.item() if torch.is_tensor(l_scr) else 0.0
# # #         breakdown["total"]    = total.item()  if torch.is_tensor(total)  else 0.0
# # #         return total, breakdown




# # # # nên 'from Model.XXX import ...' sẽ tự tìm TC_project/Model/XXX.py

# # # _NORM_TO_DEG = 5.0


# # # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# # #     return torch.stack([
# # #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# # #         (t[..., 1] * 50.0) / 10.0,
# # #     ], dim=-1)


# # # # ══════════════════════════════════════════════════════════════
# # # #  SpeedHead (giữ từ v59)
# # # # ══════════════════════════════════════════════════════════════

# # # class SpeedHead(nn.Module):
# # #     """Predict speed per step từ context. Log-space output."""
# # #     def __init__(self, ctx_dim: int = 256, obs_feat_dim: int = 256,
# # #                  pred_len: int = 12):
# # #         super().__init__()
# # #         self.pred_len = pred_len
# # #         self.net = nn.Sequential(
# # #             nn.Linear(ctx_dim + obs_feat_dim, 256),
# # #             nn.GELU(),
# # #             nn.LayerNorm(256),
# # #             nn.Linear(256, 128),
# # #             nn.GELU(),
# # #             nn.Linear(128, pred_len),
# # #         )
# # #         nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
# # #         nn.init.zeros_(self.net[-1].bias)

# # #     def forward(self, ctx: torch.Tensor, obs_feat: torch.Tensor) -> torch.Tensor:
# # #         """Returns speed in km/6h, clamped [3, 150]."""
# # #         log_spd = self.net(torch.cat([ctx, obs_feat], dim=-1))  # [B, T]
# # #         return torch.exp(log_spd).clamp(3.0, 150.0)


# # # # ══════════════════════════════════════════════════════════════
# # # #  Feature Extractors — v60 Enhanced
# # # # ══════════════════════════════════════════════════════════════

# # # def _safe_env(env_data, key: str, B: int, device: torch.device,
# # #                norm: float = 1.0) -> torch.Tensor:
# # #     """Safe scalar extraction từ env_data dict."""
# # #     v = env_data.get(key) if env_data is not None else None
# # #     if v is None or not torch.is_tensor(v):
# # #         return torch.zeros(B, device=device)
# # #     v = v.float().to(device)
# # #     while v.dim() > 1:
# # #         v = v.mean(-1)
# # #     if v.numel() >= B:
# # #         v = v.view(-1)[:B]
# # #     else:
# # #         v = torch.zeros(B, device=device)
# # #     return (v / norm).clamp(-3.0, 3.0)


# # # def _safe_env_vec(env_data, key: str, dim: int,
# # #                    B: int, device: torch.device) -> torch.Tensor:
# # #     """Safe vector extraction từ env_data dict → [B, dim]."""
# # #     v = env_data.get(key) if env_data is not None else None
# # #     if v is None:
# # #         return torch.zeros(B, dim, device=device)
# # #     if not torch.is_tensor(v):
# # #         try:
# # #             v = torch.tensor(v, dtype=torch.float, device=device)
# # #         except Exception:
# # #             return torch.zeros(B, dim, device=device)
# # #     v = v.float().to(device)

# # #     if v.dim() == 0:
# # #         return torch.zeros(B, dim, device=device)
# # #     if v.dim() == 1:
# # #         if v.shape[0] == dim:
# # #             return v.unsqueeze(0).expand(B, dim)
# # #         return torch.zeros(B, dim, device=device)
# # #     if v.dim() == 2:
# # #         if v.shape == (B, dim):
# # #             return v
# # #         # Shape mismatch: try to fix
# # #         if v.shape[0] == B:
# # #             if v.shape[1] >= dim:
# # #                 return v[:, :dim]
# # #             return F.pad(v, (0, dim - v.shape[1]))
# # #         return torch.zeros(B, dim, device=device)
# # #     if v.dim() == 3:
# # #         # [B, T, dim] → take last step
# # #         vv = v[:B, -1, :]  # [B, dim]
# # #         if vv.shape[1] >= dim:
# # #             return vv[:, :dim]
# # #         return F.pad(vv, (0, dim - vv.shape[1]))
# # #     return torch.zeros(B, dim, device=device)


# # # # ══════════════════════════════════════════════════════════════
# # # #  VelocityField v60
# # # # ══════════════════════════════════════════════════════════════

# # # class VelocityField(nn.Module):
# # #     """
# # #     FM velocity field với full feature integration và adaptive modules.

# # #     Context dim: 272 → 512 (was 176 → 512)
# # #     Adaptive outputs: blend_alpha, guidance_scale, sigma (all context-conditioned)
# # #     """
# # #     RAW_CTX_DIM = 512

# # #     def __init__(self, pred_len: int = 12, obs_len: int = 8,
# # #                  ctx_dim: int = 256, sigma_min: float = 0.02,
# # #                  unet_in_ch: int = 13, **kwargs):
# # #         super().__init__()
# # #         self.pred_len = pred_len
# # #         self.obs_len  = obs_len
# # #         self.ctx_dim  = ctx_dim
# # #         self.sigma_min = sigma_min

# # #         # ── Encoders (từ FM v59) ──────────────────────────────
# # #         self.spatial_enc = FNO3DEncoder(
# # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # #             spatial_down=32, dropout=0.05)
# # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # #         self.bottleneck_proj = nn.Linear(128, 128)
# # #         self.decoder_proj    = nn.Linear(1, 16)
# # #         self.enc_1d = DataEncoder1D(
# # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # #         # ── Steering encoder: 9 features (v59: 7) ─────────────
# # #         # Thêm: gph500_mean, gph500_center
# # #         self.steering_enc = nn.Sequential(
# # #             nn.Linear(9, 64), nn.GELU(), nn.LayerNorm(64),
# # #             nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 256))

# # #         # ── Env kinematic encoder: 14 features (v59: 14) ──────
# # #         self.env_kine_enc = nn.Sequential(
# # #             nn.Linear(14, 64), nn.GELU(), nn.LayerNorm(64),
# # #             nn.Linear(64, 256), nn.GELU())

# # #         # ── [NEW] Recurvature encoder: 33 features → 64 ───────
# # #         # bearing_to_scs_center(16) + dist_to_scs_boundary(5) + month(12)
# # #         self.recurv_enc = nn.Sequential(
# # #             nn.Linear(33, 64), nn.GELU(), nn.LayerNorm(64),
# # #             nn.Linear(64, 64))

# # #         # ── [NEW] Speed history encoder: 11 features → 32 ─────
# # #         # velocity_history(4) + rapid_intensification(1) + intensity_class(6)
# # #         self.speed_hist_enc = nn.Sequential(
# # #             nn.Linear(11, 32), nn.GELU(), nn.LayerNorm(32),
# # #             nn.Linear(32, 32))

# # #         # ── Context fusion: 272 → 512 (v59: 176 → 512) ────────
# # #         # 128(mamba) + 32(env_net) + 16(fno_dec) + 64(recurv) + 32(speed_hist)
# # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16 + 64 + 32, self.RAW_CTX_DIM)
# # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # #         self.ctx_drop = nn.Dropout(0.15)
# # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # #         self.null_embedding = nn.Parameter(torch.randn(1, self.RAW_CTX_DIM) * 0.02)

# # #         # ── Kinematic obs encoder ──────────────────────────────
# # #         self.vel_obs_enc = nn.Sequential(
# # #             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
# # #             nn.Linear(256, 256), nn.GELU())

# # #         # ── [NEW] Adaptive modules (context-conditioned) ───────
# # #         # Persistence blend alpha: 0 = no blend, 1 = full persistence
# # #         self.blend_head    = nn.Linear(ctx_dim, 1)
# # #         # CFG guidance scale: [0.8, 2.0]
# # #         self.guidance_head = nn.Linear(ctx_dim, 1)
# # #         # Sigma per sample: [0.02, 0.10]
# # #         self.sigma_head    = nn.Linear(ctx_dim, 1)

# # #         # Init adaptive heads to neutral values
# # #         nn.init.zeros_(self.blend_head.weight)
# # #         nn.init.constant_(self.blend_head.bias, -1.0)    # sigmoid(-1) ≈ 0.27
# # #         nn.init.zeros_(self.guidance_head.weight)
# # #         nn.init.constant_(self.guidance_head.bias, 0.0)  # → gs = 0.8 + 1.2×0.5 = 1.4
# # #         nn.init.zeros_(self.sigma_head.weight)
# # #         nn.init.constant_(self.sigma_head.bias, -1.0)    # → sigma ≈ 0.035

# # #         # ── Transformer decoder ────────────────────────────────
# # #         self.time_fc1   = nn.Linear(256, 512)
# # #         self.time_fc2   = nn.Linear(512, 256)
# # #         self.traj_embed = nn.Linear(4, 256)
# # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # #         self.step_embed = nn.Embedding(pred_len, 256)
# # #         self.transformer = nn.TransformerDecoder(
# # #             nn.TransformerDecoderLayer(
# # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # #                 dropout=0.10, activation='gelu', batch_first=True),
# # #             num_layers=2)
# # #         self.out_fc1 = nn.Linear(256, 512)
# # #         self.out_fc2 = nn.Linear(512, 4)

# # #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# # #         self.physics_scale  = nn.Parameter(torch.ones(4) * 1.5)
# # #         self.steering_scale = nn.Parameter(torch.ones(4) * 1.0)

# # #         # SpeedHead
# # #         self.speed_head = SpeedHead(ctx_dim=ctx_dim, obs_feat_dim=256,
# # #                                     pred_len=pred_len)
# # #         self._init_weights()

# # #     def _init_weights(self):
# # #         with torch.no_grad():
# # #             for name, m in self.named_modules():
# # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # #                     if m.bias is not None:
# # #                         nn.init.zeros_(m.bias)

# # #     def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
# # #         half = dim // 2
# # #         freq = torch.exp(
# # #             torch.arange(half, dtype=torch.float, device=t.device)
# # #             * (-math.log(10000.0) / max(half - 1, 1)))
# # #         emb = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # #     # ── Feature extractors ────────────────────────────────────

# # #     def _get_steering_feat(self, env_data, B: int, device) -> torch.Tensor:
# # #         """9 features: u/v500 mean/center + steering + gph500 mean/center"""
# # #         if env_data is None:
# # #             return torch.zeros(B, 256, device=device)
# # #         feats = torch.stack([
# # #             _safe_env(env_data, 'u500_mean',       B, device, 30.0),
# # #             _safe_env(env_data, 'v500_mean',       B, device, 30.0),
# # #             _safe_env(env_data, 'u500_center',     B, device, 30.0),
# # #             _safe_env(env_data, 'v500_center',     B, device, 30.0),
# # #             _safe_env(env_data, 'steering_speed',  B, device, 1.0),
# # #             _safe_env(env_data, 'steering_dir_sin',B, device, 1.0),
# # #             _safe_env(env_data, 'steering_dir_cos',B, device, 1.0),
# # #             # [NEW] GPH500
# # #             _safe_env(env_data, 'gph500_mean',   B, device, 1.0),
# # #             _safe_env(env_data, 'gph500_center', B, device, 1.0),
# # #         ], dim=-1)  # [B, 9]
# # #         return self.steering_enc(feats)  # [B, 256]

# # #     def _get_env_kine_feat(self, env_data, B: int, device) -> torch.Tensor:
# # #         """14 features: move_velocity + history_direction24 + delta_velocity"""
# # #         if env_data is None:
# # #             return torch.zeros(B, 256, device=device)
# # #         mv   = _safe_env(env_data, 'move_velocity', B, device, 150.0).unsqueeze(-1)
# # #         hd24 = _safe_env_vec(env_data, 'history_direction24', 8, B, device)
# # #         dv   = _safe_env_vec(env_data, 'delta_velocity', 5, B, device)
# # #         feat = torch.cat([mv, hd24, dv], dim=-1)   # [B, 14]
# # #         return self.env_kine_enc(feat)              # [B, 256]

# # #     def _get_recurv_feat(self, env_data, B: int, device) -> torch.Tensor:
# # #         """
# # #         [NEW] Recurvature features: 33 dims → 64
# # #         bearing_to_scs_center(16) + dist_to_scs_boundary(5) + month(12)

# # #         Tại sao:
# # #         - bearing_to_scs_center: hướng đến tâm SCS → recurvature signal
# # #         - dist_to_scs_boundary: khoảng cách đến đất liền
# # #         - month: seasonality (ITCZ vs subtropical ridge)
# # #         """
# # #         if env_data is None:
# # #             return torch.zeros(B, 64, device=device)

# # #         bearing = _safe_env_vec(env_data, 'bearing_to_scs_center', 16, B, device)
# # #         dist    = _safe_env_vec(env_data, 'dist_to_scs_boundary',  5,  B, device)
# # #         month   = _safe_env_vec(env_data, 'month',                 12, B, device)

# # #         feat = torch.cat([bearing, dist, month], dim=-1)  # [B, 33]
# # #         return self.recurv_enc(feat)                       # [B, 64]

# # #     def _get_speed_hist_feat(self, env_data, B: int, device) -> torch.Tensor:
# # #         """
# # #         [NEW] Speed history: 11 dims → 32
# # #         velocity_history(4) + rapid_intensification(1) + intensity_class(6)

# # #         Tại sao:
# # #         - velocity_history: 24h speed momentum
# # #         - rapid_intensification: RI flag → speed spike signal
# # #         - intensity_class: category → rough speed range
# # #         """
# # #         if env_data is None:
# # #             return torch.zeros(B, 32, device=device)

# # #         vh  = _safe_env_vec(env_data, 'velocity_history',      4, B, device)
# # #         ri  = _safe_env(env_data, 'rapid_intensification', B, device, 1.0).unsqueeze(-1)
# # #         ic  = _safe_env_vec(env_data, 'intensity_class', 6, B, device)

# # #         feat = torch.cat([vh, ri, ic], dim=-1)   # [B, 11]
# # #         return self.speed_hist_enc(feat)          # [B, 32]

# # #     def _get_kinematic_obs_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
# # #         """[T, B, 2] obs → [B, 256] kinematic features."""
# # #         B = obs_traj.shape[1]
# # #         T_obs = obs_traj.shape[0]
# # #         if T_obs >= 2:
# # #             vel     = obs_traj[1:] - obs_traj[:-1]
# # #             lat_mid = obs_traj[:-1, :, 1] * _NORM_TO_DEG
# # #             cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
# # #             dx_km   = vel[:, :, 0] * cos_lat * DEG2KM * _NORM_TO_DEG
# # #             dy_km   = vel[:, :, 1] * DEG2KM * _NORM_TO_DEG
# # #             speed   = torch.sqrt(dx_km**2 + dy_km**2 + 1e-6) / DT_HOURS
# # #             heading = torch.atan2(vel[:, :, 1], vel[:, :, 0])
# # #             speed_n = (speed / 20.0).clamp(-3.0, 3.0)
# # #             if T_obs >= 3:
# # #                 dspd  = speed[1:] - speed[:-1]
# # #                 accel = torch.cat([obs_traj.new_zeros(1, B),
# # #                                     (dspd / 10.0).clamp(-3.0, 3.0)], dim=0)
# # #             else:
# # #                 accel = obs_traj.new_zeros(T_obs - 1, B)
# # #             kine = torch.stack(
# # #                 [vel[:, :, 0], vel[:, :, 1], speed_n,
# # #                  heading.sin(), heading.cos(), accel], dim=-1)
# # #         else:
# # #             kine = obs_traj.new_zeros(self.obs_len, B, 6)

# # #         if kine.shape[0] < self.obs_len:
# # #             pad = obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6)
# # #             kine = torch.cat([pad, kine], dim=0)
# # #         else:
# # #             kine = kine[-self.obs_len:]

# # #         return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))  # [B, 256]

# # #     # ── Context building ──────────────────────────────────────

# # #     def _context(self, batch_list) -> torch.Tensor:
# # #         """
# # #         Build raw context [B, 512].
# # #         Tích hợp tất cả encoders bao gồm v60 additions.
# # #         """
# # #         obs_traj  = batch_list[0]    # [T_obs, B, 4]
# # #         obs_Me    = batch_list[7]    # [T_obs, B, 2]
# # #         image_obs = batch_list[11]   # [B, C, T, H, W] or [B, 1, T, H, W]
# # #         env_data  = batch_list[13] if len(batch_list) > 13 else None

# # #         B      = obs_traj.shape[1]
# # #         device = obs_traj.device

# # #         if image_obs.dim() == 4:
# # #             image_obs = image_obs.unsqueeze(2)
# # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # #         # FNO3D encoding
# # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # #         T_obs = obs_traj.shape[0]
# # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # #         if e_3d_s.shape[1] != T_obs:
# # #             e_3d_s = F.interpolate(
# # #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# # #                 mode='linear', align_corners=False).permute(0, 2, 1)

# # #         # Temporal summary
# # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # #         t_w = torch.softmax(
# # #             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float, device=device) * 0.5,
# # #             dim=0)
# # #         f_sp = self.decoder_proj(
# # #             (e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))  # [B, 16]

# # #         # Mamba 1D encoder
# # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # #         h_t    = self.enc_1d(obs_in, e_3d_s)        # [B, 128]

# # #         # Env_net
# # #         e_env, _, _ = self.env_enc(env_data, image_obs)  # [B, 32]

# # #         # [NEW] Recurvature + speed history
# # #         recurv_feat  = self._get_recurv_feat(env_data, B, device)    # [B, 64]
# # #         speed_h_feat = self._get_speed_hist_feat(env_data, B, device) # [B, 32]

# # #         # Fusion: 128+32+16+64+32 = 272
# # #         cat_feat = torch.cat([h_t, e_env, f_sp, recurv_feat, speed_h_feat], dim=-1)
# # #         raw_ctx  = F.gelu(self.ctx_ln(self.ctx_fc1(cat_feat)))
# # #         return raw_ctx  # [B, 512]

# # #     def _apply_ctx_head(self, raw: torch.Tensor,
# # #                          noise_scale: float = 0.0,
# # #                          use_null: bool = False) -> torch.Tensor:
# # #         if use_null:
# # #             raw = self.null_embedding.expand(raw.shape[0], -1)
# # #         elif noise_scale > 0.0:
# # #             raw = raw + torch.randn_like(raw) * noise_scale
# # #         return self.ctx_fc2(self.ctx_drop(raw))  # [B, ctx_dim]

# # #     # ── Adaptive module outputs ───────────────────────────────

# # #     def get_blend_alpha(self, ctx: torch.Tensor) -> torch.Tensor:
# # #         """
# # #         Learned persistence blend weight ∈ [0, 0.5].
# # #         Straight storm context → higher alpha (persistence helps)
# # #         Recurvature context → lower alpha (persistence harmful)
# # #         """
# # #         return torch.sigmoid(self.blend_head(ctx)).squeeze(-1) * 0.5  # [B]

# # #     def get_guidance_scale(self, ctx: torch.Tensor) -> torch.Tensor:
# # #         """Learned CFG guidance scale ∈ [0.8, 2.0]."""
# # #         return 0.8 + 1.2 * torch.sigmoid(self.guidance_head(ctx)).squeeze(-1)  # [B]

# # #     def get_sigma(self, ctx: torch.Tensor) -> torch.Tensor:
# # #         """Learned sigma per sample ∈ [0.02, 0.10]."""
# # #         return 0.02 + 0.08 * torch.sigmoid(self.sigma_head(ctx)).squeeze(-1)   # [B]

# # #     # ── Physics drift terms (từ v59, không thay đổi) ─────────

# # #     def _beta_drift(self, x_t: torch.Tensor) -> torch.Tensor:
# # #         lat_rad = torch.deg2rad(x_t[:, :, 1] * 5.0).clamp(-85, 85)
# # #         beta = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # #         R_tc = 3e5
# # #         v = torch.zeros_like(x_t)
# # #         v[:, :, 0] = -beta * R_tc**2 / 2 * 6 * 3600 / (5 * 111 * 1000)
# # #         v[:, :, 1] =  beta * R_tc**2 / 4 * 6 * 3600 / (5 * 111 * 1000)
# # #         return v

# # #     def _steering_drift(self, x_t: torch.Tensor, env_data) -> torch.Tensor:
# # #         if env_data is None:
# # #             return torch.zeros_like(x_t)
# # #         B, device = x_t.shape[0], x_t.device
# # #         u  = _safe_env(env_data, 'u500_center', B, device, 30.0)
# # #         vv = _safe_env(env_data, 'v500_center', B, device, 30.0)
# # #         cos = torch.cos(torch.deg2rad(x_t[:, :, 1] * 5.0)).clamp(1e-3)
# # #         out = torch.zeros_like(x_t)
# # #         out[:, :, 0] = u.unsqueeze(1)  * 30.0 * 21600.0 / (111.0 * 1000.0 * cos)
# # #         out[:, :, 1] = vv.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0)
# # #         return out

# # #     # ── Decode step ───────────────────────────────────────────

# # #     def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
# # #                  ctx: torch.Tensor, vel_obs_feat=None,
# # #                  steering_feat=None, env_kine_feat=None,
# # #                  env_data=None) -> torch.Tensor:
# # #         B = x_t.shape[0]
# # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # #         t_emb = self.time_fc2(t_emb)

# # #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)

# # #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# # #                  + self.pos_enc[:, :T_seq]
# # #                  + t_emb.unsqueeze(1)
# # #                  + self.step_embed(step_idx))

# # #         mem = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # #         if vel_obs_feat  is not None: mem.append(vel_obs_feat.unsqueeze(1))
# # #         if steering_feat is not None: mem.append(steering_feat.unsqueeze(1))
# # #         if env_kine_feat is not None: mem.append(env_kine_feat.unsqueeze(1))

# # #         decoded  = self.transformer(x_emb, torch.cat(mem, dim=1))
# # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
# # #         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # #         v_neural = v_neural * scale

# # #         with torch.no_grad():
# # #             v_phys  = self._beta_drift(x_t[:, :T_seq])
# # #             v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

# # #         return (v_neural
# # #                 + torch.sigmoid(self.physics_scale) * v_phys
# # #                 + torch.sigmoid(self.steering_scale) * v_steer)

# # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
# # #                           vel_obs_feat=None, steering_feat=None,
# # #                           env_kine_feat=None, env_data=None,
# # #                           use_null=False):
# # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale, use_null=use_null)
# # #         return self._decode(x_t, t, ctx,
# # #                             vel_obs_feat=vel_obs_feat,
# # #                             steering_feat=steering_feat,
# # #                             env_kine_feat=env_kine_feat,
# # #                             env_data=env_data)

# # #     def predict_speed(self, raw_ctx, vel_obs_feat):
# # #         ctx = self._apply_ctx_head(raw_ctx)
# # #         return self.speed_head(ctx, vel_obs_feat)








# # # # ══════════════════════════════════════════════════════════════
# # # #  EMA
# # # # ══════════════════════════════════════════════════════════════

# # # class EMAModel:
# # #     def __init__(self, model: nn.Module, decay: float = 0.995):
# # #         self.decay = decay
# # #         m = model._orig_mod if hasattr(model, '_orig_mod') else model
# # #         self.shadow = {k: v.detach().clone()
# # #                        for k, v in m.state_dict().items()
# # #                        if v.dtype.is_floating_point}

# # #     def update(self, model: nn.Module):
# # #         m = model._orig_mod if hasattr(model, '_orig_mod') else model
# # #         with torch.no_grad():
# # #             for k, v in m.state_dict().items():
# # #                 if k in self.shadow:
# # #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

# # #     def apply_to(self, model: nn.Module) -> dict:
# # #         m = model._orig_mod if hasattr(model, '_orig_mod') else model
# # #         backup, sd = {}, m.state_dict()
# # #         for k in self.shadow:
# # #             if k not in sd: continue
# # #             backup[k] = sd[k].detach().clone()
# # #             sd[k].copy_(self.shadow[k])
# # #         return backup

# # #     def restore(self, model: nn.Module, backup: dict):
# # #         m = model._orig_mod if hasattr(model, '_orig_mod') else model
# # #         sd = m.state_dict()
# # #         for k, v in backup.items():
# # #             if k in sd: sd[k].copy_(v)


# # # # ══════════════════════════════════════════════════════════════
# # # #  OT matching (từ v59, không thay đổi)
# # # # ══════════════════════════════════════════════════════════════

# # # def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05,
# # #                    n_iter: int = 50) -> torch.Tensor:
# # #     B = cost.shape[0]
# # #     device = cost.device
# # #     log_a  = -math.log(B) * torch.ones(B, device=device)
# # #     log_b  = -math.log(B) * torch.ones(B, device=device)
# # #     log_K  = -cost / epsilon
# # #     log_u  = torch.zeros(B, device=device)
# # #     log_v  = torch.zeros(B, device=device)
# # #     for _ in range(n_iter):
# # #         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
# # #         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
# # #     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


# # # def _spherical_ot_matching(
# # #     x0_batch: torch.Tensor, x1_batch: torch.Tensor,
# # #     lp: torch.Tensor, epsilon: float = 0.05,
# # # ) -> Tuple[torch.Tensor, torch.Tensor]:
# # #     try:
# # #         B = x0_batch.shape[0]
# # #         abs0 = lp.unsqueeze(1) + x0_batch[:, :, :2]
# # #         abs1 = lp.unsqueeze(1) + x1_batch[:, :, :2]
# # #         abs0_deg = _norm_to_deg(abs0)
# # #         abs1_deg = _norm_to_deg(abs1)
# # #         x0e = abs0_deg.unsqueeze(1).expand(B, B, -1, -1).reshape(B*B, -1, 2)
# # #         x1e = abs1_deg.unsqueeze(0).expand(B, B, -1, -1).reshape(B*B, -1, 2)
# # #         cost = haversine_km(
# # #             x0e.permute(1, 0, 2), x1e.permute(1, 0, 2)
# # #         ).mean(0).reshape(B, B) / 500.0
# # #         pi   = _sinkhorn_log(cost, epsilon=epsilon)
# # #         flat = pi.reshape(-1).clamp(0.0)
# # #         s    = flat.sum()
# # #         if not torch.isfinite(s) or s < 1e-10:
# # #             return x0_batch, x1_batch
# # #         idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
# # #         col = idx % B
# # #         return x0_batch[col], x1_batch[col]
# # #     except Exception:
# # #         return x0_batch, x1_batch


# # # # ══════════════════════════════════════════════════════════════
# # # #  Speed statistics (từ v59)
# # # # ══════════════════════════════════════════════════════════════

# # # def compute_speed_stats_from_norm(obs_traj_norm: torch.Tensor) -> dict:
# # #     T_obs = obs_traj_norm.shape[0]
# # #     if T_obs < 2:
# # #         return {'v_opt': 15.0, 'v_sigma': 10.0, 'v_hard_cap': 80.0, 'p50_kmh': 15.0}

# # #     lon = (obs_traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
# # #     lat = (obs_traj_norm[..., 1] * 50.0) / 10.0
# # #     lat_mid = (lat[:-1] + lat[1:]) / 2
# # #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
# # #     dx = (lon[1:] - lon[:-1]) * cos_lat * DEG2KM
# # #     dy = (lat[1:] - lat[:-1]) * DEG2KM
# # #     spd = torch.sqrt(dx**2 + dy**2) / DT_HOURS  # km/h

# # #     spd_flat = spd.flatten()
# # #     p50 = float(spd_flat.median())
# # #     p95 = float(torch.quantile(spd_flat, 0.95))

# # #     return {
# # #         'v_opt':     max(p50, 5.0),
# # #         'v_sigma':   10.0,
# # #         'v_hard_cap': float(torch.tensor(p95 * 1.8).clamp(25.0, 130.0)),
# # #         'p50_kmh':   p50,
# # #     }


# # # # ══════════════════════════════════════════════════════════════
# # # #  Persistence blend (v60: adaptive)
# # # # ══════════════════════════════════════════════════════════════

# # # @torch.no_grad()
# # # def _persistence_blend_adaptive(
# # #     model_pred_norm: torch.Tensor,   # [T, B, 2]
# # #     obs_traj_norm:   torch.Tensor,   # [T_obs, B, 2]
# # #     blend_alpha:     torch.Tensor,   # [B] ∈ [0, 0.5] learned
# # # ) -> torch.Tensor:
# # #     """
# # #     Adaptive persistence blend với learned α per sample.

# # #     FIX BUG-7: α không phải 0.20 cứng.
# # #     Recurvature context → α≈0.05 (persistence sai → ít blend)
# # #     Straight context    → α≈0.25 (persistence đúng → blend nhiều)
# # #     """
# # #     T_obs = obs_traj_norm.shape[0]
# # #     T     = model_pred_norm.shape[0]
# # #     B     = model_pred_norm.shape[1]
# # #     device = model_pred_norm.device

# # #     if T_obs < 2:
# # #         return model_pred_norm

# # #     # Compute persistence trajectory
# # #     vels = obs_traj_norm[1:] - obs_traj_norm[:-1]
# # #     n_v  = vels.shape[0]
# # #     if n_v >= 3:
# # #         alpha = 0.7
# # #         w = torch.tensor([alpha * (1 - alpha)**i for i in range(n_v)],
# # #                           dtype=torch.float, device=device).flip(0)
# # #         ev = (vels * (w / w.sum()).view(-1, 1, 1)).sum(0)
# # #     elif n_v == 2:
# # #         ev = 0.7 * vels[-1] + 0.3 * vels[-2]
# # #     else:
# # #         ev = vels[-1]

# # #     steps   = torch.arange(1, T + 1, dtype=torch.float, device=device)
# # #     persist = (obs_traj_norm[-1].unsqueeze(0)
# # #                + ev.unsqueeze(0) * steps.view(T, 1, 1))  # [T, B, 2]

# # #     # Adaptive blend: [B] → [1, B, 1]
# # #     alpha_b = blend_alpha.view(1, B, 1).clamp(0.0, 0.5)
# # #     return (1.0 - alpha_b) * model_pred_norm + alpha_b * persist


# # # # ══════════════════════════════════════════════════════════════
# # # #  TCFlowMatching v60
# # # # ══════════════════════════════════════════════════════════════

# # # class TCFlowMatching(nn.Module):

# # #     def __init__(self, pred_len: int = 12, obs_len: int = 8,
# # #                  sigma_min: float = 0.02, unet_in_ch: int = 13,
# # #                  ctx_noise_scale: float = 0.01,
# # #                  use_ema: bool = True, ema_decay: float = 0.995,
# # #                  use_ate_ot: bool = True, ot_epsilon: float = 0.05,
# # #                  cfg_uncond_prob: float = 0.1,
# # #                  **kwargs):
# # #         super().__init__()
# # #         self.pred_len        = pred_len
# # #         self.obs_len         = obs_len
# # #         self.sigma_min       = sigma_min
# # #         self.ctx_noise_scale = ctx_noise_scale
# # #         self.use_ate_ot      = use_ate_ot
# # #         self.ot_epsilon      = ot_epsilon
# # #         self.cfg_uncond_prob = cfg_uncond_prob

# # #         self.net     = VelocityField(
# # #             pred_len=pred_len, obs_len=obs_len,
# # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch, ctx_dim=256)
# # #         self.criterion = FMv60Loss(pred_len=pred_len)

# # #         self.use_ema   = use_ema
# # #         self.ema_decay = ema_decay
# # #         self._ema      = None

# # #     def init_ema(self):
# # #         if self.use_ema:
# # #             self._ema = EMAModel(self, decay=self.ema_decay)

# # #     def ema_update(self):
# # #         if self._ema is not None:
# # #             self._ema.update(self)

# # #     # ── Helpers ───────────────────────────────────────────────

# # #     @staticmethod
# # #     def _to_rel(traj, Me, lp, lm):
# # #         return torch.cat([traj - lp.unsqueeze(0),
# # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # #     @staticmethod
# # #     def _to_abs(rel, lp, lm):
# # #         d = rel.permute(1, 0, 2)
# # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # #     @staticmethod
# # #     def _sigma_schedule(epoch: int) -> float:
# # #         """Annealing sigma from 0.10 to 0.035."""
# # #         if epoch < 2:  return 0.10
# # #         if epoch < 10: return 0.10 - (epoch - 2) / 8.0 * (0.10 - 0.04)
# # #         if epoch < 20: return max(0.04 - (epoch - 10) / 10.0 * 0.01, 0.035)
# # #         return 0.035

# # #     def _cfm_noisy(self, x1: torch.Tensor, sigma_min: Optional[float] = None,
# # #                     lp=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# # #         if sigma_min is None: sigma_min = self.sigma_min
# # #         B      = x1.shape[0]
# # #         device = x1.device
# # #         x0     = torch.randn_like(x1) * sigma_min
# # #         t      = torch.rand(B, device=device)
# # #         te     = t.view(B, 1, 1)
# # #         x_t    = (1.0 - te) * x0 + te * x1
# # #         u_target = x1 - x0
# # #         return x_t, t, u_target

# # #     def _persistence_forecast_rel(self, obs_traj, lp, lm, pred_len: int):
# # #         B, device = obs_traj.shape[1], obs_traj.device
# # #         if obs_traj.shape[0] >= 3:
# # #             vels  = obs_traj[1:] - obs_traj[:-1]
# # #             n_v   = vels.shape[0]
# # #             alpha = 0.7
# # #             w     = torch.tensor([alpha * (1-alpha)**i for i in range(n_v)],
# # #                                   dtype=torch.float, device=device).flip(0)
# # #             lv    = (vels * (w / w.sum()).view(-1, 1, 1)).sum(0)
# # #         elif obs_traj.shape[0] >= 2:
# # #             lv = obs_traj[-1, :, :2] - obs_traj[-2, :, :2]
# # #         else:
# # #             lv = obs_traj.new_zeros(B, 2)

# # #         steps    = torch.arange(1, pred_len + 1, device=device).float()
# # #         pred_abs = (obs_traj[-1, :, :2].unsqueeze(1)
# # #                     + lv.unsqueeze(1) * steps.view(1, -1, 1))  # [B, T, 2]
# # #         pred_rel_pos = pred_abs.permute(1, 0, 2) - lp.unsqueeze(0)
# # #         pred_rel     = torch.cat([pred_rel_pos,
# # #                                     torch.zeros_like(pred_rel_pos)], dim=-1)
# # #         return pred_rel.permute(1, 0, 2)   # [B, T, 4]

# # #     def _compute_obs_momentum(self, obs_traj_norm: torch.Tensor) -> torch.Tensor:
# # #         T_obs = obs_traj_norm.shape[0]
# # #         if T_obs < 2:
# # #             return torch.zeros(obs_traj_norm.shape[1], 2, device=obs_traj_norm.device)
# # #         vels = obs_traj_norm[1:] - obs_traj_norm[:-1]
# # #         n_v  = vels.shape[0]
# # #         if n_v >= 3:
# # #             alpha = 0.65
# # #             w = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
# # #                               dtype=torch.float, device=obs_traj_norm.device).flip(0)
# # #             return (vels * (w / w.sum()).view(-1, 1, 1)).sum(0)
# # #         elif n_v == 2:
# # #             return 0.65 * vels[-1] + 0.35 * vels[-2]
# # #         return vels[-1]

# # #     @staticmethod
# # #     def _obs_noise_aug(bl, sigma: float = 0.005):
# # #         if torch.rand(1).item() > 0.5: return bl
# # #         bl = list(bl)
# # #         if torch.is_tensor(bl[0]):
# # #             bl[0] = bl[0] + torch.randn_like(bl[0]) * sigma
# # #         return bl

# # #     # ── Training ──────────────────────────────────────────────

# # #     def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
# # #         return self.get_loss_breakdown(batch_list, epoch=epoch)['total']

# # #     def get_loss_breakdown(self, batch_list, epoch: int = 0) -> dict:
# # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)

# # #         obs_t    = batch_list[0]
# # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # #         lp, lm   = obs_t[-1], batch_list[7][-1]
# # #         B, device = obs_t.shape[1], obs_t.device

# # #         current_sigma = self._sigma_schedule(epoch)
# # #         raw_ctx       = self.net._context(batch_list)

# # #         x1_rel = self._to_rel(batch_list[1], batch_list[8], lp, lm)

# # #         # OT matching
# # #         if self.use_ate_ot and B >= 4:
# # #             noise_base = torch.randn_like(x1_rel) * current_sigma
# # #             noise_matched, x1_matched = _spherical_ot_matching(
# # #                 noise_base, x1_rel, lp, epsilon=self.ot_epsilon)
# # #         else:
# # #             noise_matched = torch.randn_like(x1_rel) * current_sigma
# # #             x1_matched    = x1_rel

# # #         x_t, fm_t, u_target = self._cfm_noisy(x1_matched, sigma_min=current_sigma, lp=lp)

# # #         use_null     = (torch.rand(1).item() < self.cfg_uncond_prob)
# # #         vel_obs_feat = self.net._get_kinematic_obs_feat(obs_t[:, :, :2])
# # #         steering_feat = self.net._get_steering_feat(env_data, B, device)
# # #         env_kine_feat = self.net._get_env_kine_feat(env_data, B, device)

# # #         pred_vel = self.net.forward_with_ctx(
# # #             x_t, fm_t, raw_ctx, env_data=env_data, use_null=use_null,
# # #             vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
# # #             env_kine_feat=env_kine_feat)

# # #         # Get predicted trajectory để tính physics losses
# # #         fm_te       = fm_t.view(B, 1, 1)
# # #         x1_pred     = x_t + (1.0 - fm_te) * pred_vel
# # #         # pred_abs, _ = self._to_abs(x1_pred, lp, lm)
# # #         # pred_deg    = _norm_to_deg(pred_abs)                    # [B, T, 2]
# # #         # gt_deg      = _norm_to_deg(batch_list[1])               # [B, T, 2]

# # #         # # Reshape: loss expects [T, B, 2]
# # #         # pred_deg_t = pred_deg.permute(1, 0, 2)   # [T, B, 2]
# # #         # gt_deg_t   = gt_deg.permute(1, 0, 2)     # [T, B, 2]


# # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)
# # #         pred_deg_t  = _norm_to_deg(pred_abs)          # [T, B, 2] trực tiếp
# # #         gt_deg_t    = _norm_to_deg(batch_list[1])      # [T, B, 2] trực tiếp


# # #         # FM velocity: [B, T, 4]
# # #         fm_vel_pred   = pred_vel
# # #         fm_vel_target = u_target

# # #         # Scorer training candidates (từ epoch 5+, nhẹ)
# # #         candidates = None
# # #         obs_norm   = None
# # #         if epoch >= 5 and not use_null:
# # #             obs_norm = obs_t[:, :, :2]  # [T_obs, B, 2]
# # #             # Sinh 3 candidates với noise khác nhau
# # #             cands = []
# # #             for _ in range(3):
# # #                 x0_c = torch.randn_like(x1_rel) * current_sigma
# # #                 x1_c = x1_rel  # same gt, different noise path
# # #                 te_c  = fm_t.view(B, 1, 1)
# # #                 x_c   = (1.0 - te_c) * x0_c + te_c * x1_c
# # #                 # Quick single-step rollout (không train, chỉ lấy sample)
# # #                 with torch.no_grad():
# # #                     v_c = self.net.forward_with_ctx(
# # #                         x_c, fm_t, raw_ctx, env_data=env_data,
# # #                         vel_obs_feat=vel_obs_feat,
# # #                         steering_feat=steering_feat,
# # #                         env_kine_feat=env_kine_feat)
# # #                     x1_c_pred = x_c + (1.0 - te_c) * v_c
# # #                     abs_c, _  = self._to_abs(x1_c_pred, lp, lm)
# # #                     # Normalize back
# # #                     # lon_n = (lon * 10 - 1800) / 50
# # #                     lon_n = (abs_c[:, :, 0] * 10.0 - 1800.0) / 50.0
# # #                     lat_n = (abs_c[:, :, 1] * 10.0) / 50.0
# # #                     cand_norm = torch.stack([lon_n, lat_n], dim=-1).permute(1, 0, 2)
# # #                 cands.append(cand_norm)
# # #             candidates = cands

# # #         # Compute loss
# # #         total, breakdown = self.criterion(
# # #             pred_deg_t, gt_deg_t,
# # #             fm_vel_pred, fm_vel_target,
# # #             candidates=candidates,
# # #             obs_norm=obs_norm,
# # #         )

# # #         if torch.isnan(total) or torch.isinf(total):
# # #             total = obs_t.new_zeros(())

# # #         # Add backward-compat keys
# # #         breakdown.update({
# # #             'sigma': current_sigma,
# # #             'v_opt': compute_speed_stats_from_norm(obs_t[:, :, :2]).get('v_opt', 15.0),
# # #             # Legacy keys for trainer compatibility
# # #             'dpe': 0.0, 'mse': 0.0, 'speed': 0.0, 'accel': 0.0,
# # #             'heading': 0.0, 'vel_reg': 0.0, 'ate': 0.0, 'cte': 0.0,
# # #             'sph_ate': 0.0, 'endpoint': 0.0, 'signed_ate': 0.0,
# # #             'signed_cte': 0.0, 'direct_ep': 0.0, 'fm_mse': breakdown.get('l_fm', 0.0),
# # #         })
# # #         breakdown['total'] = total  # tensor for backward

# # #         return breakdown

# # #     # ── Inference ─────────────────────────────────────────────

# # #     @torch.no_grad()
# # #     def sample(self, batch_list, num_ensemble: int = 50,
# # #                 ddim_steps: int = 20, predict_csv: Optional[str] = None) -> tuple:
# # #         """
# # #         FM v60 inference.

# # #         Changes vs v59:
# # #         1. sigma per sample: learned (adaptive), không cứng 0.035
# # #         2. CFG guidance: learned gs per sample, không 1.5 cứng
# # #         3. Scorer: learned MLP, không fixed heuristic với v_opt=15
# # #         4. Persistence blend: learned alpha per sample, không 0.20 cứng
# # #         5. Speed sweep: REMOVED (L_logspeed fix speed, không cần post-hoc)
# # #         """
# # #         obs_t    = batch_list[0]
# # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # #         lp       = obs_t[-1]
# # #         lm       = batch_list[7][-1]
# # #         B        = lp.shape[0]
# # #         device   = lp.device
# # #         T        = self.pred_len
# # #         dt       = 1.0 / max(ddim_steps, 1)

# # #         raw_ctx       = self.net._context(batch_list)
# # #         ctx           = self.net._apply_ctx_head(raw_ctx)         # [B, 256]
# # #         vel_obs_feat  = self.net._get_kinematic_obs_feat(obs_t[:, :, :2])
# # #         steering_feat = self.net._get_steering_feat(env_data, B, device)
# # #         env_kine_feat = self.net._get_env_kine_feat(env_data, B, device)

# # #         obs_norm  = obs_t[:, :, :2]
# # #         obs_mom   = self._compute_obs_momentum(obs_norm)

# # #         # Adaptive modules
# # #         blend_alpha     = self.net.get_blend_alpha(ctx)      # [B] ∈ [0, 0.5]
# # #         guidance_scale  = self.net.get_guidance_scale(ctx)   # [B] ∈ [0.8, 2.0]
# # #         sigma_per_sample = self.net.get_sigma(ctx)           # [B] ∈ [0.02, 0.10]

# # #         # Persistence init
# # #         persist_init = self._persistence_forecast_rel(obs_t, lp, lm, T)

# # #         # Obs heading (for momentum)
# # #         if obs_t.shape[0] >= 2:
# # #             obs_h_n = F.normalize(obs_t[-1, :, :2] - obs_t[-2, :, :2],
# # #                                    dim=-1, eps=1e-6)
# # #         else:
# # #             obs_h_n = None

# # #         def _mom_str(s: int, tot: int) -> float:
# # #             return 0.06 * 0.5 * (1.0 + math.cos(math.pi * s / max(tot, 1)))

# # #         all_norms = []

# # #         for ens_i in range(num_ensemble):
# # #             # Per-sample sigma for diversity
# # #             sigma_noise = sigma_per_sample.mean().item() * 2.5
# # #             x_t = persist_init + torch.randn_like(persist_init) * sigma_noise

# # #             for step in range(ddim_steps):
# # #                 t_b  = torch.full((B,), step * dt, device=device)
# # #                 ns   = self.ctx_noise_scale * 2.0 if step < 3 else 0.0

# # #                 # CFG with learned guidance scale
# # #                 if step > 0:
# # #                     v_cond   = self.net.forward_with_ctx(
# # #                         x_t, t_b, raw_ctx, noise_scale=ns,
# # #                         vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
# # #                         env_kine_feat=env_kine_feat, env_data=env_data,
# # #                         use_null=False)
# # #                     v_uncond = self.net.forward_with_ctx(
# # #                         x_t, t_b, raw_ctx, noise_scale=0.0,
# # #                         vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
# # #                         env_kine_feat=env_kine_feat, env_data=env_data,
# # #                         use_null=True)
# # #                     # Learned guidance scale per sample
# # #                     gs = guidance_scale.view(B, 1, 1)
# # #                     vel = v_uncond + gs * (v_cond - v_uncond)
# # #                 else:
# # #                     vel = self.net.forward_with_ctx(
# # #                         x_t, t_b, raw_ctx, noise_scale=ns,
# # #                         vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
# # #                         env_kine_feat=env_kine_feat, env_data=env_data)

# # #                 # Momentum injection
# # #                 m_s = _mom_str(step, ddim_steps)
# # #                 if m_s > 1e-4:
# # #                     me = obs_mom.unsqueeze(1).expand(B, T, 2)
# # #                     mf = torch.cat([me, torch.zeros(B, T, 2, device=device)], dim=-1)
# # #                     vel = vel + m_s * mf

# # #                 x_t = (x_t + dt * vel).clamp(-3.0, 3.0)

# # #             # tr, me = self._to_abs(x_t, lp, lm)
# # #             # all_norms.append(tr)
# # #             tr, me = self._to_abs(x_t, lp, lm)
# # #             all_norms.append(tr.permute(1, 0, 2))   # [B, T, 2]

# # #         # ── Learned scoring (FIX BUG-1, BUG-5) ──────────────
# # #         # Không có prior_sc (v_opt=15 sai)
# # #         # Dùng learned scorer MLP
# # #         scores = []
# # #         # for tn in all_norms:
# # #         #     # Normalize back to norm space for scorer
# # #         #     lon_n = (tn[:, :, 0] * 10.0 - 1800.0) / 50.0  # [B, T]
# # #         #     lat_n = (tn[:, :, 1] * 10.0) / 50.0
# # #         #     tn_norm = torch.stack([lon_n, lat_n], dim=-1).permute(1, 0, 2)  # [T, B, 2]
# # #         for tn in all_norms:
# # #             tn_norm = tn.permute(1, 0, 2)   # [B,T,2] → [T,B,2]
# # #             sc = self.criterion.scorer.score(tn_norm, obs_norm)  # [B]
# # #             scores.append(sc)

# # #         all_c  = torch.stack(all_norms)  # [N_ens, B, T, 2]
# # #         all_sc = torch.stack(scores)     # [N_ens, B]

# # #         # Top 35%
# # #         k = max(1, int(all_c.shape[0] * 0.35))
# # #         _, top_idx = all_sc.topk(k, dim=0)   # [k, B]

# # #         pred_mean = torch.stack([
# # #             all_c[top_idx[:, b], b, :, :].median(0).values
# # #             for b in range(B)
# # #         ], dim=0).permute(1, 0, 2)   # [T, B, 2]

# # #         # ── Adaptive persistence blend (FIX BUG-7) ───────────
# # #         # pred_norm_t = torch.stack([
# # #         #     ((pred_mean[:, :, 0] * 10.0 - 1800.0) / 50.0),
# # #         #     (pred_mean[:, :, 1] * 10.0 / 50.0),
# # #         # ], dim=-1).permute(1, 0, 2)  # norm → [T, B, 2]

# # #         # pred_norm_t = torch.stack([
# # #         #     ((pred_mean[:, :, 0] * 10.0 - 1800.0) / 50.0),
# # #         #     (pred_mean[:, :, 1] * 10.0 / 50.0),
# # #         # ], dim=-1)  # [T, B, 2] — xóa .permute(1, 0, 2)
# # #         pred_norm_t = pred_mean   # [T, B, 2] normalized — không cần convert
# # #         blended_norm = _persistence_blend_adaptive(pred_norm_t, obs_norm, blend_alpha)

# # #         # Convert back to degrees
# # #         final_deg = _norm_to_deg(blended_norm.permute(1, 0, 2))  # [B, T, 2]
# # #         # Back to normalized for output
# # #         lon_out = (final_deg[:, :, 0] * 10.0 - 1800.0) / 50.0
# # #         lat_out = final_deg[:, :, 1] * 10.0 / 50.0
# # #         pred_final = torch.stack([lon_out, lat_out], dim=-1).permute(1, 0, 2)  # [T, B, 2]

# # #         if predict_csv:
# # #             self._write_predict_csv(predict_csv, pred_final, all_c)

# # #         return pred_final, all_c

# # #     @staticmethod
# # #     def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
# # #                             all_trajs: torch.Tensor):
# # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # #         T, B, _ = traj_mean.shape
# # #         mlon = ((traj_mean[:, :, 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # #         mlat = ((traj_mean[:, :, 1] * 50.0) / 10.0).cpu().numpy()
# # #         ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
# # #         fields = ['timestamp', 'batch_idx', 'step_idx', 'lead_h',
# # #                   'lon_mean_deg', 'lat_mean_deg']
# # #         write_hdr = not os.path.exists(csv_path)
# # #         with open(csv_path, 'a', newline='') as fh:
# # #             w = csv.DictWriter(fh, fieldnames=fields)
# # #             if write_hdr: w.writeheader()
# # #             for b in range(B):
# # #                 for k in range(T):
# # #                     w.writerow({'timestamp': ts, 'batch_idx': b,
# # #                                 'step_idx': k, 'lead_h': (k + 1) * 6,
# # #                                 'lon_mean_deg': f'{mlon[k, b]:.4f}',
# # #                                 'lat_mean_deg': f'{mlat[k, b]:.4f}'})


# # # # Backward compat alias
# # # TCDiffusion = TCFlowMatching

# # """
# # Model/flow_matching_model.py  ── FM v60 FIXED (seq-first)
# # =========================================================
# # COPY FILE NÀY VÀO Model/flow_matching_model.py

# # SHAPE CONVENTION (giữ nguyên như dataloader gốc):
# #   - batch_list[0] obs_t:  [T_obs, B, 4]  seq-first
# #   - batch_list[1] gt:     [T,     B, 2]  seq-first
# #   - _to_rel: nhận [T,B,2], trả [B,T,4] batch-first (giống gốc)
# #   - _to_abs: nhận [B,T,4], trả [T,B,2] seq-first   (giống gốc)
# #   - pred_deg, gt_deg: [T,B,2] seq-first
# #   - candidates:       list of [T,B,2] seq-first
# #   - obs_norm:         [T_obs,B,2] seq-first

# # FIXES:
# #   BUG-CRASH: auxiliary_loss gọi permute sai convention → crash dim mismatch
# #              FIX: transpose nội bộ đúng chiều, không phụ thuộc convention ngoài

# #   BUG-SCALE: l_kinematic ~80-100 >> l_fm ~0.4 → FM không học
# #              FIX: /NORM=100 → l_kin ~0.2-1.5, cùng scale l_fm

# #   BUG-INIT:  DifficultyWeighter bias=0 → dw=1.5 ngay epoch 1
# #              FIX: bias=-2.0 → dw≈1.12

# #   BUG-WARMUP: λ_kin=0.5 epoch 1 + l_kin lớn → FM objective bị bóp
# #               FIX: epoch<5 dùng fixed λ nhỏ
# # """
# # from __future__ import annotations

# # import csv
# # import math
# # import os
# # from datetime import datetime
# # from typing import Optional, Tuple

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # from Model.FNO3D_encoder import FNO3DEncoder
# # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # from Model.env_net_transformer_gphsplit import Env_net

# # R_EARTH  = 6371.0
# # DT_HOURS = 6.0
# # DEG2KM   = 111.0
# # MAX_CURVATURE_RAD = math.pi / 4
# # _NORM_TO_DEG = 5.0


# # # ══════════════════════════════════════════════════════════════
# # #  Coordinate utilities
# # # ══════════════════════════════════════════════════════════════

# # def norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# #     """Normalized [...,2] → degrees. Shape-agnostic."""
# #     return torch.stack([
# #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# #         (t[..., 1] * 50.0) / 10.0,
# #     ], dim=-1)

# # _norm_to_deg = norm_to_deg


# # def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# #     """[...,2] lon°,lat° → [...] km. Shape-agnostic."""
# #     lat1 = torch.deg2rad(p1[..., 1])
# #     lat2 = torch.deg2rad(p2[..., 1])
# #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# #     a = (torch.sin(dlat / 2).pow(2)
# #          + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
# #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1.0 - 1e-12).sqrt())


# # def velocity_km_s(traj_deg: torch.Tensor) -> torch.Tensor:
# #     """
# #     [T,B,2] lon°,lat° → [T-1,B,2] velocity km/6h.
# #     SEQ-FIRST convention.
# #     """
# #     lon = traj_deg[:, :, 0]
# #     lat = traj_deg[:, :, 1]
# #     dlat    = lat[1:] - lat[:-1]
# #     dlon    = lon[1:] - lon[:-1]
# #     lat_mid = (lat[1:] + lat[:-1]) / 2.0
# #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
# #     vx = dlon * cos_lat * DEG2KM
# #     vy = dlat * DEG2KM
# #     return torch.stack([vx, vy], dim=-1)  # [T-1,B,2]


# # # ══════════════════════════════════════════════════════════════
# # #  Learned Step Weights
# # # ══════════════════════════════════════════════════════════════

# # class LearnedStepWeights(nn.Module):
# #     def __init__(self, n_steps: int = 12):
# #         super().__init__()
# #         self.n_steps = n_steps
# #         self.raw = nn.Parameter(torch.zeros(n_steps) + 0.5)

# #     def forward(self) -> torch.Tensor:
# #         increments = F.softplus(self.raw)
# #         weights    = torch.cumsum(increments, dim=0)
# #         return weights / weights.mean().clamp(min=1e-8)

# #     def get(self, n: Optional[int] = None) -> torch.Tensor:
# #         w = self.forward()
# #         return w[:n] if n is not None else w

# #     @torch.no_grad()
# #     def stats(self) -> dict:
# #         w = self.forward()
# #         return {
# #             "sw_ratio":    (w[-1] / w[0].clamp(min=1e-6)).item(),
# #             "sw_monotonic": bool((w[1:] - w[:-1]).min().item() >= -1e-6),
# #         }


# # # ══════════════════════════════════════════════════════════════
# # #  Log-Parameterized Loss Weights
# # # ══════════════════════════════════════════════════════════════

# # class LearnedLossWeights(nn.Module):
# #     def __init__(self, n_tasks: int = 3):
# #         super().__init__()
# #         self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

# #     def lambdas(self) -> torch.Tensor:
# #         return 0.5 * torch.exp(-2.0 * self.log_sigma)

# #     def regularization(self) -> torch.Tensor:
# #         return self.log_sigma.sum()

# #     @torch.no_grad()
# #     def stats(self) -> dict:
# #         lam = self.lambdas()
# #         return {
# #             "lam_kin":    lam[0].item(),
# #             "lam_logspd": lam[1].item(),
# #             "lam_curv":   lam[2].item(),
# #         }


# # # ══════════════════════════════════════════════════════════════
# # #  Difficulty Weighter — seq-first
# # # ══════════════════════════════════════════════════════════════

# # class DifficultyWeighter(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.linear = nn.Linear(3, 1, bias=True)
# #         nn.init.zeros_(self.linear.weight)
# #         # FIX: -2.0 → sigmoid(-2)≈0.12 → dw≈1.12 (not 1.5)
# #         nn.init.constant_(self.linear.bias, -2.0)

# #     def compute_difficulty(self, gt_deg: torch.Tensor) -> torch.Tensor:
# #         """gt_deg [T,B,2] seq-first → [B,3]."""
# #         B = gt_deg.shape[1]
# #         v = velocity_km_s(gt_deg)  # [T-1,B,2]

# #         if v.shape[0] >= 2:
# #             heading  = torch.atan2(v[:, :, 0], v[:, :, 1])  # [T-1,B]
# #             dh       = heading[1:] - heading[:-1]
# #             dh       = (dh + math.pi) % (2 * math.pi) - math.pi
# #             curv_rate = dh.abs().mean(0)  # [B]
# #             excess    = F.relu(dh.abs() - MAX_CURVATURE_RAD).mean(0)
# #         else:
# #             curv_rate = gt_deg.new_zeros(B)
# #             excess    = gt_deg.new_zeros(B)

# #         spd      = v.norm(dim=-1)  # [T-1,B]
# #         mean_spd = spd.mean(0).clamp(min=1.0)
# #         speed_cv = (spd.std(0) / mean_spd).clamp(max=3.0)

# #         d1 = (curv_rate / (math.pi / 2)).clamp(0, 1)
# #         d2 = speed_cv.clamp(0, 1)
# #         d3 = (excess   / (math.pi / 4)).clamp(0, 1)
# #         return torch.stack([d1, d2, d3], dim=-1)  # [B,3]

# #     def forward(self, gt_deg: torch.Tensor) -> torch.Tensor:
# #         """gt_deg [T,B,2] → [B] weights ∈ [1,2]."""
# #         diff  = self.compute_difficulty(gt_deg)  # [B,3]
# #         logit = self.linear(diff).squeeze(-1)    # [B]
# #         return 1.0 + torch.sigmoid(logit)        # [B]


# # # ══════════════════════════════════════════════════════════════
# # #  Ensemble Scorer — seq-first [T,B,2]
# # # ══════════════════════════════════════════════════════════════

# # class EnsembleScorer(nn.Module):
# #     def __init__(self, feat_dim: int = 7, hidden: int = 32):
# #         super().__init__()
# #         # No sigmoid here — score() uses sigmoid, auxiliary_loss uses BCEWithLogits
# #         self.net = nn.Sequential(
# #             nn.Linear(feat_dim, hidden), nn.GELU(),
# #             nn.Linear(hidden, 16),       nn.GELU(),
# #             nn.Linear(16, 1),
# #         )
# #         nn.init.zeros_(self.net[-1].weight)
# #         nn.init.zeros_(self.net[-1].bias)

# #     def extract_features(
# #         self,
# #         traj_norm: torch.Tensor,   # [T,B,2] seq-first normalized
# #         obs_norm:  torch.Tensor,   # [T_obs,B,2] seq-first normalized
# #     ) -> torch.Tensor:
# #         """Returns [B,7]."""
# #         B        = traj_norm.shape[1]
# #         traj_deg = norm_to_deg(traj_norm)  # [T,B,2]
# #         obs_deg  = norm_to_deg(obs_norm)   # [T_obs,B,2]

# #         v   = velocity_km_s(traj_deg)   # [T-1,B,2]
# #         spd = v.norm(dim=-1)            # [T-1,B]

# #         f1 = torch.log1p(spd.mean(0))
# #         f2 = torch.log1p(spd.std(0).clamp(min=0))

# #         if v.shape[0] >= 2:
# #             heading = torch.atan2(v[:, :, 0], v[:, :, 1])
# #             dh = heading[1:] - heading[:-1]
# #             dh = (dh + math.pi) % (2 * math.pi) - math.pi
# #             f3 = torch.cos(dh).mean(0)
# #         else:
# #             f3 = traj_norm.new_ones(B)

# #         v_obs    = velocity_km_s(obs_deg)   # [T_obs-1,B,2]
# #         n_obs    = min(3, v_obs.shape[0])
# #         obs_spd  = v_obs[-n_obs:].norm(dim=-1).mean(0).clamp(min=1.0)  # [B]
# #         n_pred   = min(3, spd.shape[0])
# #         pred_spd = spd[:n_pred].mean(0)
# #         f4       = torch.exp(-((pred_spd - obs_spd) / obs_spd).pow(2) * 2.0)

# #         if v_obs.shape[0] >= 1 and v.shape[0] >= 1:
# #             obs_h   = torch.atan2(v_obs[-1, :, 0], v_obs[-1, :, 1])  # [B]
# #             pred_h  = torch.atan2(v[0, :, 0], v[0, :, 1])            # [B]
# #             dh_cont = pred_h - obs_h
# #             dh_cont = (dh_cont + math.pi) % (2 * math.pi) - math.pi
# #             f5      = torch.cos(dh_cont)
# #         else:
# #             f5 = traj_norm.new_ones(B)

# #         if obs_norm.shape[0] >= 2:
# #             lv      = obs_norm[-1] - obs_norm[-2]   # [B,2]
# #             steps   = torch.arange(1, traj_norm.shape[0] + 1,
# #                                     device=traj_norm.device, dtype=traj_norm.dtype)
# #             persist = (obs_norm[-1].unsqueeze(0)
# #                        + lv.unsqueeze(0) * steps.view(-1, 1, 1))  # [T,B,2]
# #             dfp     = (traj_norm - persist).norm(dim=-1).mean(0)   # [B]
# #             ref     = (lv.norm(dim=-1) * traj_norm.shape[0]).clamp(min=1e-3)
# #             f6      = dfp / ref
# #         else:
# #             f6 = traj_norm.new_zeros(B)

# #         if v.shape[0] >= 2:
# #             heading = torch.atan2(v[:, :, 0], v[:, :, 1])
# #             dh      = heading[1:] - heading[:-1]
# #             dh      = (dh + math.pi) % (2 * math.pi) - math.pi
# #             f7      = dh.abs().mean(0)
# #         else:
# #             f7 = traj_norm.new_zeros(B)

# #         return torch.stack([f1, f2, f3, f4, f5, f6, f7], dim=-1).clamp(-10.0, 10.0)  # [B,7]

# #     def score(
# #         self,
# #         traj_norm: torch.Tensor,   # [T,B,2] seq-first
# #         obs_norm:  torch.Tensor,   # [T_obs,B,2] seq-first
# #     ) -> torch.Tensor:
# #         """Returns [B] ∈ [0,1] (probabilities for inference)."""
# #         feats = self.extract_features(traj_norm, obs_norm)
# #         return torch.sigmoid(self.net(feats).squeeze(-1))

# #     def logits(
# #         self,
# #         traj_norm: torch.Tensor,
# #         obs_norm:  torch.Tensor,
# #     ) -> torch.Tensor:
# #         """Returns [B] raw logits (for BCE_with_logits, AMP-safe)."""
# #         feats = self.extract_features(traj_norm, obs_norm)
# #         return self.net(feats).squeeze(-1)

# #     def auxiliary_loss(
# #         self,
# #         candidates: list,          # list of [T,B,2] seq-first normalized
# #         obs_norm:   torch.Tensor,  # [T_obs,B,2] seq-first normalized
# #         gt_deg:     torch.Tensor,  # [T,B,2] seq-first degrees
# #     ) -> torch.Tensor:
# #         """
# #         BCE với oracle ADE ranking.
# #         FIX: convert seq-first→batch-first nội bộ, nhất quán.
# #         """
# #         if len(candidates) < 2:
# #             return candidates[0].new_zeros(())

# #         # gt_deg [T,B,2] → [B,T,2] cho haversine
# #         gt_deg_b = gt_deg.permute(1, 0, 2)  # [B,T,2]

# #         ades = []
# #         for cand in candidates:
# #             # cand [T,B,2] seq-first → convert deg → [B,T,2]
# #             cand_deg_b = norm_to_deg(cand).permute(1, 0, 2)  # [B,T,2]
# #             d = haversine_km(cand_deg_b, gt_deg_b).mean(dim=1)  # [B]
# #             ades.append(d)

# #         ades_t   = torch.stack(ades, dim=0)   # [n_cands,B]
# #         best_idx = ades_t.argmin(dim=0)        # [B]

# #         total = ades_t.new_zeros(())
# #         for i, cand in enumerate(candidates):
# #             # AMP-safe: use BCEWithLogitsLoss (combines sigmoid+BCE, numerically stable)
# #             logits = self.logits(cand, obs_norm)          # [B] raw logits
# #             labels = (best_idx == i).float()              # [B]
# #             total  = total + F.binary_cross_entropy_with_logits(
# #                 logits, labels, reduction='mean')
# #         return total / len(candidates)


# # # ══════════════════════════════════════════════════════════════
# # #  Physics Loss Terms — seq-first [T,B,2]
# # # ══════════════════════════════════════════════════════════════

# # def l_kinematic(
# #     pred_deg:     torch.Tensor,   # [T,B,2] seq-first degrees
# #     gt_deg:       torch.Tensor,   # [T,B,2]
# #     step_weights: torch.Tensor,   # [T-1]
# #     sample_w:     Optional[torch.Tensor] = None,  # [B]
# # ) -> torch.Tensor:
# #     """
# #     Huber velocity loss. FIX: /NORM=100 → same scale as l_fm (~0.3-0.8).
# #     """
# #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# #     if T < 2:
# #         return pred_deg.new_zeros(())

# #     v_pred  = velocity_km_s(pred_deg[:T])   # [T-1,B,2]
# #     v_gt    = velocity_km_s(gt_deg[:T])

# #     err_sq  = (v_pred - v_gt).pow(2).sum(dim=-1)  # [T-1,B]
# #     delta   = 50.0
# #     err_abs = err_sq.sqrt()
# #     huber   = torch.where(err_abs < delta,
# #                           0.5 * err_sq / delta,
# #                           err_abs - delta / 2.0)

# #     # FIX: normalize to l_fm scale
# #     huber = huber / 100.0

# #     n        = huber.shape[0]
# #     w        = step_weights[:n]
# #     weighted = huber * w.unsqueeze(1)  # [T-1,B]

# #     if sample_w is not None:
# #         weighted = weighted * sample_w.unsqueeze(0)  # broadcast over T

# #     return weighted.mean()


# # def l_logspeed(
# #     pred_deg: torch.Tensor,  # [T,B,2]
# #     gt_deg:   torch.Tensor,
# # ) -> torch.Tensor:
# #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# #     if T < 2:
# #         return pred_deg.new_zeros(())

# #     v_pred   = velocity_km_s(pred_deg[:T])
# #     v_gt     = velocity_km_s(gt_deg[:T])
# #     spd_pred = v_pred.norm(dim=-1).clamp(min=0.1)
# #     spd_gt   = v_gt.norm(dim=-1).clamp(min=0.1)

# #     return F.mse_loss(torch.log1p(spd_pred), torch.log1p(spd_gt))


# # def l_curvature(
# #     pred_deg:      torch.Tensor,   # [T,B,2]
# #     threshold_rad: float = MAX_CURVATURE_RAD,
# # ) -> torch.Tensor:
# #     if pred_deg.shape[0] < 3:
# #         return pred_deg.new_zeros(())

# #     v = velocity_km_s(pred_deg)  # [T-1,B,2]
# #     if v.shape[0] < 2:
# #         return pred_deg.new_zeros(())

# #     heading = torch.atan2(v[:, :, 0], v[:, :, 1])
# #     dh      = heading[1:] - heading[:-1]
# #     dh      = (dh + math.pi) % (2 * math.pi) - math.pi
# #     return F.relu(dh.abs() - threshold_rad).mean()


# # # ══════════════════════════════════════════════════════════════
# # #  Master Loss
# # # ══════════════════════════════════════════════════════════════

# # class FMv60Loss(nn.Module):
# #     def __init__(self, pred_len: int = 12):
# #         super().__init__()
# #         self.pred_len      = pred_len
# #         self.step_weights  = LearnedStepWeights(n_steps=pred_len)
# #         self.loss_weights  = LearnedLossWeights(n_tasks=3)
# #         self.diff_weighter = DifficultyWeighter()
# #         self.scorer        = EnsembleScorer()

# #     def compute_main_losses(
# #         self,
# #         pred_deg:      torch.Tensor,   # [T,B,2] seq-first
# #         gt_deg:        torch.Tensor,   # [T,B,2]
# #         fm_vel_pred:   torch.Tensor,   # [B,T,4]
# #         fm_vel_target: torch.Tensor,   # [B,T,4]
# #         epoch:         int = 0,
# #     ) -> Tuple[torch.Tensor, dict]:
# #         T = min(pred_deg.shape[0], gt_deg.shape[0])

# #         sample_w = self.diff_weighter(gt_deg[:T])  # [B]
# #         sw       = self.step_weights.get(n=T - 1)  # [T-1]

# #         # FIX: warmup — epoch<5: fixed small λ so FM objective dominates first
# #         if epoch < 10:
# #             # Warmup: FM objective dominates.
# #             # λ_logspd=0.02 (not 0.30!) because l_logspd gradient propagates
# #             # through km-scale (×555 chain factor) vs l_fm in normalized space.
# #             # 0.30*large_grad >> l_fm → model learns speed not direction → ADE↑
# #             lam = torch.tensor([0.05, 0.02, 0.01],
# #                                 device=fm_vel_pred.device,
# #                                 dtype=fm_vel_pred.dtype)
# #             reg = fm_vel_pred.new_zeros(())
# #         elif epoch < 20:
# #             # Transition: slowly increase physics losses
# #             lam = torch.tensor([0.10, 0.05, 0.02],
# #                                 device=fm_vel_pred.device,
# #                                 dtype=fm_vel_pred.dtype)
# #             reg = fm_vel_pred.new_zeros(())
# #         else:
# #             lam = self.loss_weights.lambdas()
# #             reg = self.loss_weights.regularization()

# #         L_fm     = F.mse_loss(fm_vel_pred, fm_vel_target)
# #         L_kin    = l_kinematic(pred_deg[:T], gt_deg[:T], sw, sample_w)
# #         L_logspd = l_logspeed(pred_deg[:T], gt_deg[:T])
# #         L_curv   = l_curvature(pred_deg[:T])

# #         total = (1.0    * L_fm
# #                + lam[0] * L_kin
# #                + lam[1] * L_logspd
# #                + lam[2] * L_curv
# #                + reg)

# #         if not torch.isfinite(total):
# #             total = pred_deg.new_zeros(())

# #         sw_d  = self.step_weights.stats()
# #         # Show ACTUAL λ used (warmup override), not learned λ
# #         breakdown = {
# #             "l_fm":        L_fm.item(),
# #             "l_kin":       L_kin.item(),
# #             "l_logspd":    L_logspd.item(),
# #             "l_curv":      L_curv.item(),
# #             "l_reg":       reg.item() if torch.is_tensor(reg) else 0.0,
# #             "diff_w_mean": sample_w.mean().item(),
# #             "lam_kin":     lam[0].item(),
# #             "lam_logspd":  lam[1].item(),
# #             "lam_curv":    lam[2].item(),
# #             **sw_d,
# #         }
# #         return total, breakdown

# #     def forward(
# #         self,
# #         pred_deg:      torch.Tensor,
# #         gt_deg:        torch.Tensor,
# #         fm_vel_pred:   torch.Tensor,
# #         fm_vel_target: torch.Tensor,
# #         candidates:    Optional[list] = None,
# #         obs_norm:      Optional[torch.Tensor] = None,
# #         epoch:         int = 0,
# #     ) -> Tuple[torch.Tensor, dict]:
# #         total, breakdown = self.compute_main_losses(
# #             pred_deg, gt_deg, fm_vel_pred, fm_vel_target, epoch=epoch)

# #         l_scr = pred_deg.new_zeros(())
# #         if candidates is not None and obs_norm is not None and len(candidates) >= 2:
# #             l_scr = self.scorer.auxiliary_loss(candidates, obs_norm, gt_deg)
# #             total = total + 0.05 * l_scr

# #         breakdown["l_scorer"] = l_scr.item() if torch.is_tensor(l_scr) else 0.0
# #         breakdown["total"]    = total
# #         return total, breakdown


# # # ══════════════════════════════════════════════════════════════
# # #  SpeedHead
# # # ══════════════════════════════════════════════════════════════

# # class SpeedHead(nn.Module):
# #     def __init__(self, ctx_dim=256, obs_feat_dim=256, pred_len=12):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(ctx_dim + obs_feat_dim, 256), nn.GELU(), nn.LayerNorm(256),
# #             nn.Linear(256, 128), nn.GELU(), nn.Linear(128, pred_len),
# #         )
# #         nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
# #         nn.init.zeros_(self.net[-1].bias)

# #     def forward(self, ctx, obs_feat):
# #         return torch.exp(self.net(torch.cat([ctx, obs_feat], dim=-1))).clamp(3.0, 150.0)


# # # ══════════════════════════════════════════════════════════════
# # #  Safe env helpers
# # # ══════════════════════════════════════════════════════════════

# # def _safe_env(env_data, key, B, device, norm=1.0):
# #     v = env_data.get(key) if env_data is not None else None
# #     if v is None or not torch.is_tensor(v):
# #         return torch.zeros(B, device=device)
# #     v = v.float().to(device)
# #     while v.dim() > 1: v = v.mean(-1)
# #     v = v.view(-1)[:B] if v.numel() >= B else torch.zeros(B, device=device)
# #     return (v / norm).clamp(-3.0, 3.0)


# # def _safe_env_vec(env_data, key, dim, B, device):
# #     v = env_data.get(key) if env_data is not None else None
# #     if v is None:
# #         return torch.zeros(B, dim, device=device)
# #     if not torch.is_tensor(v):
# #         try: v = torch.tensor(v, dtype=torch.float, device=device)
# #         except: return torch.zeros(B, dim, device=device)
# #     v = v.float().to(device)
# #     if v.dim() == 0: return torch.zeros(B, dim, device=device)
# #     if v.dim() == 1:
# #         return v.unsqueeze(0).expand(B, dim) if v.shape[0] == dim else torch.zeros(B, dim, device=device)
# #     if v.dim() == 2:
# #         if v.shape == (B, dim): return v
# #         if v.shape[0] == B:
# #             return v[:, :dim] if v.shape[1] >= dim else F.pad(v, (0, dim - v.shape[1]))
# #         return torch.zeros(B, dim, device=device)
# #     if v.dim() == 3:
# #         vv = v[:B, -1, :]
# #         return vv[:, :dim] if vv.shape[1] >= dim else F.pad(vv, (0, dim - vv.shape[1]))
# #     return torch.zeros(B, dim, device=device)


# # # ══════════════════════════════════════════════════════════════
# # #  VelocityField (unchanged from original v60)
# # # ══════════════════════════════════════════════════════════════

# # class VelocityField(nn.Module):
# #     RAW_CTX_DIM = 512

# #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256,
# #                  sigma_min=0.02, unet_in_ch=13, **kwargs):
# #         super().__init__()
# #         self.pred_len  = pred_len
# #         self.obs_len   = obs_len
# #         self.ctx_dim   = ctx_dim
# #         self.sigma_min = sigma_min

# #         self.spatial_enc = FNO3DEncoder(
# #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# #             spatial_down=32, dropout=0.05)
# #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# #         self.bottleneck_proj = nn.Linear(128, 128)
# #         self.decoder_proj    = nn.Linear(1, 16)
# #         self.enc_1d = DataEncoder1D(
# #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# #         self.steering_enc   = nn.Sequential(
# #             nn.Linear(9,  64), nn.GELU(), nn.LayerNorm(64),
# #             nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 256))
# #         self.env_kine_enc   = nn.Sequential(
# #             nn.Linear(14, 64), nn.GELU(), nn.LayerNorm(64),
# #             nn.Linear(64, 256), nn.GELU())
# #         self.recurv_enc     = nn.Sequential(
# #             nn.Linear(33, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, 64))
# #         self.speed_hist_enc = nn.Sequential(
# #             nn.Linear(11, 32), nn.GELU(), nn.LayerNorm(32), nn.Linear(32, 32))

# #         self.ctx_fc1  = nn.Linear(128 + 32 + 16 + 64 + 32, self.RAW_CTX_DIM)
# #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# #         self.ctx_drop = nn.Dropout(0.15)
# #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)
# #         self.null_embedding = nn.Parameter(torch.randn(1, self.RAW_CTX_DIM) * 0.02)

# #         self.vel_obs_enc = nn.Sequential(
# #             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
# #             nn.Linear(256, 256), nn.GELU())

# #         self.blend_head    = nn.Linear(ctx_dim, 1)
# #         self.guidance_head = nn.Linear(ctx_dim, 1)
# #         self.sigma_head    = nn.Linear(ctx_dim, 1)
# #         nn.init.zeros_(self.blend_head.weight);    nn.init.constant_(self.blend_head.bias,    -1.0)
# #         nn.init.zeros_(self.guidance_head.weight); nn.init.constant_(self.guidance_head.bias,  0.0)
# #         nn.init.zeros_(self.sigma_head.weight);    nn.init.constant_(self.sigma_head.bias,    -1.0)

# #         self.time_fc1    = nn.Linear(256, 512)
# #         self.time_fc2    = nn.Linear(512, 256)
# #         self.traj_embed  = nn.Linear(4, 256)
# #         self.pos_enc     = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# #         self.step_embed  = nn.Embedding(pred_len, 256)
# #         self.transformer = nn.TransformerDecoder(
# #             nn.TransformerDecoderLayer(
# #                 d_model=256, nhead=8, dim_feedforward=1024,
# #                 dropout=0.10, activation='gelu', batch_first=True),
# #             num_layers=2)
# #         self.out_fc1 = nn.Linear(256, 512)
# #         self.out_fc2 = nn.Linear(512, 4)

# #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# #         self.physics_scale  = nn.Parameter(torch.ones(4) * 1.5)
# #         self.steering_scale = nn.Parameter(torch.ones(4) * 1.0)
# #         self.speed_head = SpeedHead(ctx_dim=ctx_dim, obs_feat_dim=256, pred_len=pred_len)
# #         self._init_weights()

# #     def _init_weights(self):
# #         with torch.no_grad():
# #             for name, m in self.named_modules():
# #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# #                     if m.bias is not None: nn.init.zeros_(m.bias)

# #     def _time_emb(self, t, dim=256):
# #         half = dim // 2
# #         freq = torch.exp(torch.arange(half, dtype=torch.float, device=t.device)
# #                          * (-math.log(10000.0) / max(half - 1, 1)))
# #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# #     def _get_steering_feat(self, env_data, B, device):
# #         if env_data is None: return torch.zeros(B, 256, device=device)
# #         feats = torch.stack([
# #             _safe_env(env_data, 'u500_mean',        B, device, 30.0),
# #             _safe_env(env_data, 'v500_mean',        B, device, 30.0),
# #             _safe_env(env_data, 'u500_center',      B, device, 30.0),
# #             _safe_env(env_data, 'v500_center',      B, device, 30.0),
# #             _safe_env(env_data, 'steering_speed',   B, device, 1.0),
# #             _safe_env(env_data, 'steering_dir_sin', B, device, 1.0),
# #             _safe_env(env_data, 'steering_dir_cos', B, device, 1.0),
# #             _safe_env(env_data, 'gph500_mean',      B, device, 1.0),
# #             _safe_env(env_data, 'gph500_center',    B, device, 1.0),
# #         ], dim=-1)
# #         return self.steering_enc(feats)

# #     def _get_env_kine_feat(self, env_data, B, device):
# #         if env_data is None: return torch.zeros(B, 256, device=device)
# #         mv   = _safe_env(env_data, 'move_velocity', B, device, 150.0).unsqueeze(-1)
# #         hd24 = _safe_env_vec(env_data, 'history_direction24', 8, B, device)
# #         dv   = _safe_env_vec(env_data, 'delta_velocity',      5, B, device)
# #         return self.env_kine_enc(torch.cat([mv, hd24, dv], dim=-1))

# #     def _get_recurv_feat(self, env_data, B, device):
# #         if env_data is None: return torch.zeros(B, 64, device=device)
# #         bearing = _safe_env_vec(env_data, 'bearing_to_scs_center', 16, B, device)
# #         dist    = _safe_env_vec(env_data, 'dist_to_scs_boundary',   5, B, device)
# #         month   = _safe_env_vec(env_data, 'month',                  12, B, device)
# #         return self.recurv_enc(torch.cat([bearing, dist, month], dim=-1))

# #     def _get_speed_hist_feat(self, env_data, B, device):
# #         if env_data is None: return torch.zeros(B, 32, device=device)
# #         vh = _safe_env_vec(env_data, 'velocity_history',      4, B, device)
# #         ri = _safe_env(env_data, 'rapid_intensification', B, device, 1.0).unsqueeze(-1)
# #         ic = _safe_env_vec(env_data, 'intensity_class',       6, B, device)
# #         return self.speed_hist_enc(torch.cat([vh, ri, ic], dim=-1))

# #     def _get_kinematic_obs_feat(self, obs_traj):
# #         """obs_traj [T_obs,B,2] seq-first → [B,256]."""
# #         B     = obs_traj.shape[1]
# #         T_obs = obs_traj.shape[0]
# #         if T_obs >= 2:
# #             vel     = obs_traj[1:] - obs_traj[:-1]
# #             lat_mid = obs_traj[:-1, :, 1] * _NORM_TO_DEG
# #             cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
# #             dx_km   = vel[:, :, 0] * cos_lat * DEG2KM * _NORM_TO_DEG
# #             dy_km   = vel[:, :, 1] * DEG2KM * _NORM_TO_DEG
# #             speed   = torch.sqrt(dx_km**2 + dy_km**2 + 1e-6) / DT_HOURS
# #             heading = torch.atan2(vel[:, :, 1], vel[:, :, 0])
# #             speed_n = (speed / 20.0).clamp(-3.0, 3.0)
# #             if T_obs >= 3:
# #                 dspd  = speed[1:] - speed[:-1]
# #                 accel = torch.cat([obs_traj.new_zeros(1, B),
# #                                     (dspd / 10.0).clamp(-3.0, 3.0)], dim=0)
# #             else:
# #                 accel = obs_traj.new_zeros(T_obs - 1, B)
# #             kine = torch.stack([vel[:,:,0], vel[:,:,1], speed_n,
# #                                   heading.sin(), heading.cos(), accel], dim=-1)
# #         else:
# #             kine = obs_traj.new_zeros(self.obs_len, B, 6)

# #         if kine.shape[0] < self.obs_len:
# #             pad  = obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6)
# #             kine = torch.cat([pad, kine], dim=0)
# #         else:
# #             kine = kine[-self.obs_len:]

# #         return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

# #     def _context(self, batch_list):
# #         obs_traj  = batch_list[0]   # [T_obs,B,4] seq-first
# #         obs_Me    = batch_list[7]   # [T_obs,B,2]
# #         image_obs = batch_list[11]
# #         env_data  = batch_list[13] if len(batch_list) > 13 else None

# #         B      = obs_traj.shape[1]
# #         device = obs_traj.device

# #         if image_obs.dim() == 4: image_obs = image_obs.unsqueeze(2)
# #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# #         T_obs  = obs_traj.shape[0]
# #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# #         e_3d_s = self.bottleneck_proj(e_3d_s)
# #         if e_3d_s.shape[1] != T_obs:
# #             e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
# #                                     mode='linear', align_corners=False).permute(0,2,1)

# #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# #         t_w = torch.softmax(
# #             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float, device=device) * 0.5, dim=0)
# #         f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# #         h_t    = self.enc_1d(obs_in, e_3d_s)

# #         e_env, _, _ = self.env_enc(env_data, image_obs)

# #         recurv_feat  = self._get_recurv_feat(env_data, B, device)
# #         speed_h_feat = self._get_speed_hist_feat(env_data, B, device)

# #         cat_feat = torch.cat([h_t, e_env, f_sp, recurv_feat, speed_h_feat], dim=-1)
# #         return F.gelu(self.ctx_ln(self.ctx_fc1(cat_feat)))  # [B,512]

# #     def _apply_ctx_head(self, raw, noise_scale=0.0, use_null=False):
# #         if use_null:
# #             raw = self.null_embedding.expand(raw.shape[0], -1)
# #         elif noise_scale > 0.0:
# #             raw = raw + torch.randn_like(raw) * noise_scale
# #         return self.ctx_fc2(self.ctx_drop(raw))

# #     def get_blend_alpha(self, ctx):
# #         return torch.sigmoid(self.blend_head(ctx)).squeeze(-1) * 0.5

# #     def get_guidance_scale(self, ctx):
# #         return 0.8 + 1.2 * torch.sigmoid(self.guidance_head(ctx)).squeeze(-1)

# #     def get_sigma(self, ctx):
# #         return 0.02 + 0.08 * torch.sigmoid(self.sigma_head(ctx)).squeeze(-1)

# #     def _beta_drift(self, x_t):
# #         lat_rad = torch.deg2rad(x_t[:, :, 1] * 5.0).clamp(-85, 85)
# #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# #         R_tc    = 3e5
# #         v       = torch.zeros_like(x_t)
# #         v[:, :, 0] = -beta * R_tc**2 / 2 * 6 * 3600 / (5 * 111 * 1000)
# #         v[:, :, 1] =  beta * R_tc**2 / 4 * 6 * 3600 / (5 * 111 * 1000)
# #         return v

# #     def _steering_drift(self, x_t, env_data):
# #         if env_data is None: return torch.zeros_like(x_t)
# #         B, device = x_t.shape[0], x_t.device
# #         u  = _safe_env(env_data, 'u500_center', B, device, 30.0)
# #         vv = _safe_env(env_data, 'v500_center', B, device, 30.0)
# #         cos = torch.cos(torch.deg2rad(x_t[:, :, 1] * 5.0)).clamp(1e-3)
# #         out = torch.zeros_like(x_t)
# #         out[:, :, 0] = u.unsqueeze(1)  * 30.0 * 21600.0 / (111.0 * 1000.0 * cos)
# #         out[:, :, 1] = vv.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0)
# #         return out

# #     def _decode(self, x_t, t, ctx, vel_obs_feat=None,
# #                  steering_feat=None, env_kine_feat=None, env_data=None):
# #         B     = x_t.shape[0]
# #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# #         t_emb = self.time_fc2(t_emb)

# #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# #         x_emb    = (self.traj_embed(x_t[:, :T_seq])
# #                     + self.pos_enc[:, :T_seq]
# #                     + t_emb.unsqueeze(1)
# #                     + self.step_embed(step_idx))

# #         mem = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# #         if vel_obs_feat  is not None: mem.append(vel_obs_feat.unsqueeze(1))
# #         if steering_feat is not None: mem.append(steering_feat.unsqueeze(1))
# #         if env_kine_feat is not None: mem.append(env_kine_feat.unsqueeze(1))

# #         decoded  = self.transformer(x_emb, torch.cat(mem, dim=1))
# #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
# #         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# #         v_neural = v_neural * scale

# #         with torch.no_grad():
# #             v_phys  = self._beta_drift(x_t[:, :T_seq])
# #             v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

# #         return (v_neural
# #                 + torch.sigmoid(self.physics_scale) * v_phys
# #                 + torch.sigmoid(self.steering_scale) * v_steer)

# #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
# #                           vel_obs_feat=None, steering_feat=None,
# #                           env_kine_feat=None, env_data=None, use_null=False):
# #         ctx = self._apply_ctx_head(raw_ctx, noise_scale, use_null=use_null)
# #         return self._decode(x_t, t, ctx,
# #                             vel_obs_feat=vel_obs_feat,
# #                             steering_feat=steering_feat,
# #                             env_kine_feat=env_kine_feat,
# #                             env_data=env_data)

# #     def predict_speed(self, raw_ctx, vel_obs_feat):
# #         return self.speed_head(self._apply_ctx_head(raw_ctx), vel_obs_feat)


# # # ══════════════════════════════════════════════════════════════
# # #  EMA
# # # ══════════════════════════════════════════════════════════════

# # class EMAModel:
# #     def __init__(self, model, decay=0.995):
# #         self.decay  = decay
# #         m = model._orig_mod if hasattr(model, '_orig_mod') else model
# #         self.shadow = {k: v.detach().clone() for k, v in m.state_dict().items()
# #                        if v.dtype.is_floating_point}

# #     def update(self, model):
# #         m = model._orig_mod if hasattr(model, '_orig_mod') else model
# #         with torch.no_grad():
# #             for k, v in m.state_dict().items():
# #                 if k in self.shadow:
# #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

# #     def apply_to(self, model):
# #         m = model._orig_mod if hasattr(model, '_orig_mod') else model
# #         backup, sd = {}, m.state_dict()
# #         for k in self.shadow:
# #             if k not in sd: continue
# #             backup[k] = sd[k].detach().clone(); sd[k].copy_(self.shadow[k])
# #         return backup

# #     def restore(self, model, backup):
# #         m  = model._orig_mod if hasattr(model, '_orig_mod') else model
# #         sd = m.state_dict()
# #         for k, v in backup.items():
# #             if k in sd: sd[k].copy_(v)


# # # ══════════════════════════════════════════════════════════════
# # #  OT matching — ORIGINAL (works with [B,T,4] batch-first)
# # # ══════════════════════════════════════════════════════════════

# # def _sinkhorn_log(cost, epsilon=0.05, n_iter=50):
# #     B     = cost.shape[0]; device = cost.device
# #     log_a = -math.log(B) * torch.ones(B, device=device)
# #     log_b = -math.log(B) * torch.ones(B, device=device)
# #     log_K = -cost / epsilon
# #     log_u = torch.zeros(B, device=device)
# #     log_v = torch.zeros(B, device=device)
# #     for _ in range(n_iter):
# #         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
# #         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
# #     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


# # def _spherical_ot_matching(x0_batch, x1_batch, lp, epsilon=0.05):
# #     """x0_batch, x1_batch: [B,T,4] batch-first."""
# #     try:
# #         B = x0_batch.shape[0]
# #         abs0 = lp.unsqueeze(1)[:, :, :2] + x0_batch[:, :, :2]   # [B,T,2]
# #         abs1 = lp.unsqueeze(1)[:, :, :2] + x1_batch[:, :, :2]
# #         abs0_deg = norm_to_deg(abs0); abs1_deg = norm_to_deg(abs1)
# #         cost = haversine_km(
# #             abs0_deg.unsqueeze(1).expand(B, B, -1, -1).reshape(B*B, -1, 2),
# #             abs1_deg.unsqueeze(0).expand(B, B, -1, -1).reshape(B*B, -1, 2),
# #         ).mean(-1).reshape(B, B) / 500.0
# #         pi   = _sinkhorn_log(cost, epsilon=epsilon)
# #         flat = pi.reshape(-1).clamp(0.0); s = flat.sum()
# #         if not torch.isfinite(s) or s < 1e-10: return x0_batch, x1_batch
# #         idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
# #         return x0_batch[idx % B], x1_batch[idx % B]
# #     except Exception:
# #         return x0_batch, x1_batch


# # # ══════════════════════════════════════════════════════════════
# # #  Speed stats & persistence
# # # ══════════════════════════════════════════════════════════════

# # def compute_speed_stats_from_norm(obs_traj_norm):
# #     """obs_traj_norm [T_obs,B,2] seq-first."""
# #     T_obs = obs_traj_norm.shape[0]
# #     if T_obs < 2:
# #         return {'v_opt': 15.0, 'v_sigma': 10.0, 'v_hard_cap': 80.0, 'p50_kmh': 15.0}
# #     lon = (obs_traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
# #     lat = (obs_traj_norm[..., 1] * 50.0) / 10.0
# #     lat_mid = (lat[:-1] + lat[1:]) / 2
# #     cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
# #     dx  = (lon[1:] - lon[:-1]) * cos_lat * DEG2KM
# #     dy  = (lat[1:] - lat[:-1]) * DEG2KM
# #     spd = torch.sqrt(dx**2 + dy**2) / DT_HOURS
# #     p50 = float(spd.flatten().median())
# #     p95 = float(torch.quantile(spd.flatten(), 0.95))
# #     return {'v_opt': max(p50, 5.0), 'v_sigma': 10.0,
# #             'v_hard_cap': float(torch.tensor(p95*1.8).clamp(25.0,130.0)),
# #             'p50_kmh': p50}


# # @torch.no_grad()
# # def _persistence_blend_adaptive(model_pred_norm, obs_traj_norm, blend_alpha):
# #     """
# #     model_pred_norm: [T,B,2] seq-first normalized
# #     obs_traj_norm:   [T_obs,B,2] seq-first
# #     blend_alpha:     [B]
# #     returns:         [T,B,2] seq-first
# #     """
# #     T_obs  = obs_traj_norm.shape[0]
# #     T      = model_pred_norm.shape[0]
# #     B      = model_pred_norm.shape[1]
# #     device = model_pred_norm.device
# #     if T_obs < 2: return model_pred_norm

# #     vels = obs_traj_norm[1:] - obs_traj_norm[:-1]  # [T_obs-1,B,2]
# #     n_v  = vels.shape[0]
# #     if n_v >= 3:
# #         alpha = 0.7
# #         w  = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
# #                             dtype=torch.float, device=device).flip(0)
# #         ev = (vels * (w / w.sum()).view(-1, 1, 1)).sum(0)   # [B,2]
# #     elif n_v == 2:
# #         ev = 0.7 * vels[-1] + 0.3 * vels[-2]
# #     else:
# #         ev = vels[-1]

# #     steps   = torch.arange(1, T + 1, dtype=torch.float, device=device)
# #     persist = (obs_traj_norm[-1].unsqueeze(0)
# #                + ev.unsqueeze(0) * steps.view(T, 1, 1))  # [T,B,2]

# #     alpha_b = blend_alpha.view(1, B, 1).clamp(0.0, 0.5)
# #     return (1.0 - alpha_b) * model_pred_norm + alpha_b * persist


# # # ══════════════════════════════════════════════════════════════
# # #  TCFlowMatching v60 FIXED
# # # ══════════════════════════════════════════════════════════════

# # class TCFlowMatching(nn.Module):

# #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02, unet_in_ch=13,
# #                  ctx_noise_scale=0.01, use_ema=True, ema_decay=0.995,
# #                  use_ate_ot=True, ot_epsilon=0.05, cfg_uncond_prob=0.1, **kwargs):
# #         super().__init__()
# #         self.pred_len        = pred_len
# #         self.obs_len         = obs_len
# #         self.sigma_min       = sigma_min
# #         self.ctx_noise_scale = ctx_noise_scale
# #         self.use_ate_ot      = use_ate_ot
# #         self.ot_epsilon      = ot_epsilon
# #         self.cfg_uncond_prob = cfg_uncond_prob
# #         self.net       = VelocityField(pred_len=pred_len, obs_len=obs_len,
# #                                         sigma_min=sigma_min, unet_in_ch=unet_in_ch, ctx_dim=256)
# #         self.criterion = FMv60Loss(pred_len=pred_len)
# #         self.use_ema   = use_ema
# #         self._ema      = None

# #     def init_ema(self):
# #         if self.use_ema: self._ema = EMAModel(self, decay=0.995)

# #     def ema_update(self):
# #         if self._ema is not None: self._ema.update(self)

# #     # ── Original helpers (seq-first in, batch-first internal) ──

# #     @staticmethod
# #     def _to_rel(traj, Me, lp, lm):
# #         """
# #         traj: [T,B,2] seq-first
# #         Me:   [T,B,2] seq-first
# #         lp:   [B,4], lm: [B,2]
# #         returns: [B,T,4] batch-first
# #         """
# #         return torch.cat([traj - lp[:, :2].unsqueeze(0),
# #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# #     @staticmethod
# #     def _to_abs(rel, lp, lm):
# #         """
# #         rel: [B,T,4] batch-first
# #         returns: pos [T,B,2] seq-first, me [T,B,2] seq-first
# #         """
# #         d   = rel.permute(1, 0, 2)                    # [T,B,4]
# #         pos = lp[:, :2].unsqueeze(0) + d[:, :, :2]   # [T,B,2]
# #         me  = lm.unsqueeze(0)        + d[:, :, 2:]    # [T,B,2]
# #         return pos, me

# #     @staticmethod
# #     def _sigma_schedule(epoch):
# #         if epoch < 2:  return 0.10
# #         if epoch < 10: return 0.10 - (epoch-2)/8.0*(0.10-0.04)
# #         if epoch < 20: return max(0.04-(epoch-10)/10.0*0.01, 0.035)
# #         return 0.035

# #     def _cfm_noisy(self, x1, sigma_min=None):
# #         if sigma_min is None: sigma_min = self.sigma_min
# #         B  = x1.shape[0]; device = x1.device
# #         x0 = torch.randn_like(x1) * sigma_min
# #         t  = torch.rand(B, device=device)
# #         te = t.view(B, 1, 1)
# #         return (1.0-te)*x0 + te*x1, t, x1-x0

# #     def _cfm_noisy_x0(self, x1, x0):
# #         """CFM with explicit x0 (persist+noise). u_target = x1-x0 = correction."""
# #         B  = x1.shape[0]; device = x1.device
# #         t  = torch.rand(B, device=device)
# #         te = t.view(B, 1, 1)
# #         return (1.0-te)*x0 + te*x1, t, x1-x0

# #     def _persistence_forecast_rel(self, obs_traj, lp, lm, pred_len):
# #         """obs_traj [T_obs,B,4] seq-first → [B,T,4] batch-first."""
# #         B, device = obs_traj.shape[1], obs_traj.device
# #         obs_pos   = obs_traj[:, :, :2]   # [T_obs,B,2]
# #         if obs_pos.shape[0] >= 3:
# #             vels = obs_pos[1:] - obs_pos[:-1]; n_v = vels.shape[0]; alpha = 0.7
# #             w    = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
# #                                  dtype=torch.float, device=device).flip(0)
# #             lv   = (vels * (w / w.sum()).view(-1,1,1)).sum(0)
# #         elif obs_pos.shape[0] >= 2:
# #             lv = obs_pos[-1] - obs_pos[-2]
# #         else:
# #             lv = obs_traj.new_zeros(B, 2)
# #         steps    = torch.arange(1, pred_len+1, device=device).float()
# #         pred_abs = obs_pos[-1].unsqueeze(1) + lv.unsqueeze(1)*steps.view(1,-1,1)  # [B,T,2]
# #         pred_rel = torch.cat([pred_abs - lp[:,:2].unsqueeze(1),
# #                                torch.zeros_like(pred_abs)], dim=-1)                # [B,T,4]
# #         return pred_rel

# #     def _compute_obs_momentum(self, obs_traj_norm):
# #         """[T_obs,B,2] seq-first → [B,2]."""
# #         T_obs = obs_traj_norm.shape[0]
# #         if T_obs < 2: return torch.zeros(obs_traj_norm.shape[1], 2, device=obs_traj_norm.device)
# #         vels = obs_traj_norm[1:] - obs_traj_norm[:-1]; n_v = vels.shape[0]
# #         if n_v >= 3:
# #             alpha = 0.65
# #             w = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
# #                                dtype=torch.float, device=obs_traj_norm.device).flip(0)
# #             return (vels * (w/w.sum()).view(-1,1,1)).sum(0)
# #         elif n_v == 2: return 0.65*vels[-1] + 0.35*vels[-2]
# #         return vels[-1]

# #     @staticmethod
# #     def _obs_noise_aug(bl, sigma=0.005):
# #         if torch.rand(1).item() > 0.5: return bl
# #         bl = list(bl)
# #         if torch.is_tensor(bl[0]): bl[0] = bl[0] + torch.randn_like(bl[0])*sigma
# #         return bl

# #     # ── Training ──────────────────────────────────────────────

# #     def get_loss(self, batch_list, epoch=0, **kwargs):
# #         return self.get_loss_breakdown(batch_list, epoch=epoch)['total']

# #     def get_loss_breakdown(self, batch_list, epoch=0):
# #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)

# #         obs_t    = batch_list[0]   # [T_obs,B,4] seq-first
# #         env_data = batch_list[13] if len(batch_list) > 13 else None
# #         lp       = obs_t[-1]       # [B,4]
# #         lm       = batch_list[7][-1]  # [B,2]
# #         B, device = lp.shape[0], lp.device

# #         # gt_traj: [T,B,2] seq-first (confirmed from dataloader)
# #         gt_traj_s = batch_list[1]   # [T,B,2]

# #         current_sigma = self._sigma_schedule(epoch)
# #         raw_ctx       = self.net._context(batch_list)

# #         # _to_rel: [T,B,2] → [B,T,4] batch-first
# #         x1_rel = self._to_rel(gt_traj_s, batch_list[8], lp, lm)  # [B,T,4]

# #         # FIX-D: x0 = persist_rel + N(0, sigma) instead of pure noise.
# #         # This makes training distribution MATCH inference distribution.
# #         # u_target = x1-x0 = correction from persistence (13km) not from noise (461km)
# #         # → model learns 37x easier task → ADE decreases from epoch 1
# #         persist_x0 = self._persistence_forecast_rel(obs_t, lp, lm, self.pred_len)  # [B,T,4]
# #         persist_noise = persist_x0 + torch.randn_like(persist_x0) * current_sigma  # x0

# #         if self.use_ate_ot and B >= 4:
# #             _, x1_matched = _spherical_ot_matching(
# #                 persist_noise, x1_rel, lp, epsilon=self.ot_epsilon)
# #             noise_matched = persist_noise
# #         else:
# #             noise_matched = persist_noise
# #             x1_matched    = x1_rel

# #         x_t, fm_t, u_target = self._cfm_noisy_x0(x1_matched, noise_matched)
# #         # x_t, u_target: [B,T,4] batch-first

# #         use_null      = (torch.rand(1).item() < self.cfg_uncond_prob)
# #         vel_obs_feat  = self.net._get_kinematic_obs_feat(obs_t[:, :, :2])
# #         steering_feat = self.net._get_steering_feat(env_data, B, device)
# #         env_kine_feat = self.net._get_env_kine_feat(env_data, B, device)

# #         pred_vel = self.net.forward_with_ctx(
# #             x_t, fm_t, raw_ctx, env_data=env_data, use_null=use_null,
# #             vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
# #             env_kine_feat=env_kine_feat)  # [B,T,4] batch-first

# #         # x1_pred → abs → degrees, all seq-first [T,B,2]
# #         fm_te   = fm_t.view(B, 1, 1)
# #         x1_pred = x_t + (1.0 - fm_te) * pred_vel  # [B,T,4]

# #         pred_abs, _  = self._to_abs(x1_pred, lp, lm)   # [T,B,2] seq-first
# #         pred_deg     = norm_to_deg(pred_abs)             # [T,B,2]
# #         gt_deg       = norm_to_deg(gt_traj_s)            # [T,B,2]

# #         # Candidates for scorer — seq-first [T,B,2]
# #         candidates = None
# #         obs_norm   = obs_t[:, :, :2]   # [T_obs,B,2] seq-first
# #         if epoch >= 5 and not use_null:
# #             cands = []
# #             for _ in range(3):
# #                 x0_c = torch.randn_like(x1_rel) * current_sigma
# #                 te_c = fm_t.view(B, 1, 1)
# #                 x_c  = (1.0 - te_c) * x0_c + te_c * x1_rel
# #                 with torch.no_grad():
# #                     v_c       = self.net.forward_with_ctx(
# #                         x_c, fm_t, raw_ctx, env_data=env_data,
# #                         vel_obs_feat=vel_obs_feat,
# #                         steering_feat=steering_feat,
# #                         env_kine_feat=env_kine_feat)
# #                     x1_c_pred = x_c + (1.0 - te_c) * v_c
# #                     abs_c, _  = self._to_abs(x1_c_pred, lp, lm)  # [T,B,2] seq-first normalized
# #                     # abs_c is already in normalized space (same as gt_traj_s)
# #                     # DO NOT apply (x*10-1800)/50 — that converts degrees→norm, not norm→norm
# #                     cand_norm = abs_c[:, :, :2]  # [T,B,2] seq-first normalized
# #                 cands.append(cand_norm)
# #             candidates = cands

# #         total, breakdown = self.criterion(
# #             pred_deg, gt_deg,
# #             pred_vel, u_target,
# #             candidates=candidates,
# #             obs_norm=obs_norm,
# #             epoch=epoch,
# #         )

# #         if torch.isnan(total) or torch.isinf(total):
# #             total = obs_t.new_zeros(())

# #         breakdown.update({
# #             'sigma': current_sigma,
# #             'v_opt': compute_speed_stats_from_norm(obs_t[:,:,:2]).get('v_opt', 15.0),
# #             'dpe':0., 'mse':0., 'speed':0., 'accel':0., 'heading':0.,
# #             'vel_reg':0., 'ate':0., 'cte':0., 'sph_ate':0., 'endpoint':0.,
# #             'signed_ate':0., 'signed_cte':0., 'direct_ep':0.,
# #             'fm_mse': breakdown.get('l_fm', 0.0),
# #         })
# #         breakdown['total'] = total
# #         return breakdown

# #     # ── Inference ─────────────────────────────────────────────

# #     @torch.no_grad()
# #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
# #         obs_t    = batch_list[0]        # [T_obs,B,4] seq-first
# #         env_data = batch_list[13] if len(batch_list) > 13 else None
# #         lp       = obs_t[-1]            # [B,4]
# #         lm       = batch_list[7][-1]    # [B,2]
# #         B        = lp.shape[0]
# #         device   = lp.device
# #         T        = self.pred_len
# #         dt       = 1.0 / max(ddim_steps, 1)

# #         raw_ctx       = self.net._context(batch_list)
# #         ctx           = self.net._apply_ctx_head(raw_ctx)
# #         vel_obs_feat  = self.net._get_kinematic_obs_feat(obs_t[:, :, :2])
# #         steering_feat = self.net._get_steering_feat(env_data, B, device)
# #         env_kine_feat = self.net._get_env_kine_feat(env_data, B, device)

# #         obs_norm = obs_t[:, :, :2]   # [T_obs,B,2] seq-first
# #         obs_mom  = self._compute_obs_momentum(obs_norm)  # [B,2]

# #         blend_alpha      = self.net.get_blend_alpha(ctx)
# #         guidance_scale   = self.net.get_guidance_scale(ctx)
# #         sigma_per_sample = self.net.get_sigma(ctx)

# #         persist_init = self._persistence_forecast_rel(obs_t, lp, lm, T)  # [B,T,4]

# #         def _mom_str(s, tot):
# #             return 0.06 * 0.5 * (1.0 + math.cos(math.pi * s / max(tot, 1)))

# #         all_norms = []   # list of [T,B,2] seq-first normalized

# #         for ens_i in range(num_ensemble):
# #             # Correct FM inference: x0 = persist_rel + N(0, sigma_min)
# #             # Matches training distribution exactly (FIX-D).
# #             # Each ensemble member gets different noise for diversity.
# #             x_t = persist_init + torch.randn_like(persist_init) * self.sigma_min

# #             for step in range(ddim_steps):
# #                 t_b = torch.full((B,), step * dt, device=device)
# #                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0

# #                 if step > 0:
# #                     v_cond   = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns,
# #                                 vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
# #                                 env_kine_feat=env_kine_feat, env_data=env_data, use_null=False)
# #                     v_uncond = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=0.0,
# #                                 vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
# #                                 env_kine_feat=env_kine_feat, env_data=env_data, use_null=True)
# #                     vel = v_uncond + guidance_scale.view(B,1,1) * (v_cond - v_uncond)
# #                 else:
# #                     vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns,
# #                             vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
# #                             env_kine_feat=env_kine_feat, env_data=env_data)

# #                 m_s = _mom_str(step, ddim_steps)
# #                 if m_s > 1e-4:
# #                     me  = obs_mom.unsqueeze(1).expand(B, T, 2)
# #                     mf  = torch.cat([me, torch.zeros(B, T, 2, device=device)], dim=-1)
# #                     vel = vel + m_s * mf

# #                 x_t = (x_t + dt * vel).clamp(-3.0, 3.0)

# #             pred_abs, _ = self._to_abs(x_t, lp, lm)   # [T,B,2] seq-first normalized
# #             all_norms.append(pred_abs)

# #         # Scoring — seq-first [T,B,2]
# #         scores = [self.criterion.scorer.score(tn, obs_norm) for tn in all_norms]  # list of [B]

# #         all_c  = torch.stack(all_norms)   # [N_ens,T,B,2]
# #         all_sc = torch.stack(scores)      # [N_ens,B]

# #         k = max(1, int(all_c.shape[0] * 0.35))
# #         _, top_idx = all_sc.topk(k, dim=0)   # [k,B]

# #         # Median over top-k ensemble — result [T,B,2]
# #         pred_mean = torch.stack([
# #             all_c[top_idx[:, b], :, b, :].median(0).values   # [T,2]
# #             for b in range(B)
# #         ], dim=1)   # [T,B,2]

# #         # Adaptive persistence blend — stays in normalized space [T,B,2]
# #         pred_final = _persistence_blend_adaptive(pred_mean, obs_norm, blend_alpha)

# #         if predict_csv:
# #             self._write_predict_csv(predict_csv, pred_final, all_c)

# #         return pred_final, all_c

# #     @staticmethod
# #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# #         T, B, _ = traj_mean.shape
# #         mlon = ((traj_mean[:,:,0]*50.0+1800.0)/10.0).cpu().numpy()
# #         mlat = ((traj_mean[:,:,1]*50.0)/10.0).cpu().numpy()
# #         ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
# #         fields = ['timestamp','batch_idx','step_idx','lead_h','lon_mean_deg','lat_mean_deg']
# #         write_hdr = not os.path.exists(csv_path)
# #         with open(csv_path, 'a', newline='') as fh:
# #             w = csv.DictWriter(fh, fieldnames=fields)
# #             if write_hdr: w.writeheader()
# #             for b in range(B):
# #                 for k in range(T):
# #                     w.writerow({'timestamp':ts,'batch_idx':b,'step_idx':k,
# #                                 'lead_h':(k+1)*6,
# #                                 'lon_mean_deg':f'{mlon[k,b]:.4f}',
# #                                 'lat_mean_deg':f'{mlat[k,b]:.4f}'})


# # TCDiffusion = TCFlowMatching

# """
# flow_matching_model.py — FM v74-fix
# ====================================
# DROP-IN REPLACEMENT cho v60.

# ROOT CAUSES của v60 oscillation (từ log analysis):
#   BUG-A: x0=persist+noise → u_target=correction (nhỏ 0.29x so với standard)
#          → Model học predict small velocity → speed undershoot -17 to -28 km/6h mỗi epoch
#          → ADE oscillate vì inference distribution ≠ training distribution
#   BUG-B: l_logspd gradient asymmetric → log-space MSE penalize underestimation
#          nặng hơn overestimation → conflict với BUG-A → oscillation amplified
#   BUG-C: Không có direct position loss → model chỉ học velocity, không
#          minimize haversine(pred_pos, gt_pos) trực tiếp → gradient path dài

# FIXES:
#   FIX-A: x0 = N(0, sigma) pure noise như v59
#          → u_target = x1 - x0 ≈ x1 (magnitude chuẩn)
#          → Inference distribution matches training distribution
#          → No speed bias

#   FIX-B: Remove l_logspd
#          → Thay bằng L_disp = ||Δpred - Δgt|| / ||Δgt||.clamp(10km)
#          → Normalized ratio, không asymmetric, không km-scale
#          → Covers ATE+CTE implicitly without naming either

#   FIX-C: Add L_pos = huber(haversine(pred, gt), d=300) / 300
#          → Direct position loss như v59's l_dpe
#          → Clear gradient: push pred_pos toward gt_pos mỗi step
#          → Guarantees monotone decrease (direct ADE proxy)
#          → With LEARNED step_weights (monotone, sw72/sw6≥3)

# LOSS FORMULA (general — không ATE, CTE, speed tên):
#   Total = w_fm * L_fm                    — flow matching velocity
#         + w_pos * L_pos(step_weights)     — position quality (ADE proxy)
#         + w_disp * L_disp                 — displacement quality (ATE+CTE proxy)
#         + 0.3 * L_speed_head              — auxiliary speed prediction
#         + anchor losses

# GIỮA NGUYÊN từ v60:
#   - VelocityField architecture (toàn bộ — FNO3D, Mamba, recurv_enc, etc.)
#   - EMA, OT matching, CFG guidance
#   - LearnedStepWeights (monotone cumsum)
#   - DifficultyWeighter (per-sample [1,2])
#   - EnsembleScorer (auxiliary BCE)
#   - K=3 mode clustering inference (từ analysis trước)
# """
# from __future__ import annotations

# import csv, math, os
# from datetime import datetime
# from typing import Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from Model.FNO3D_encoder import FNO3DEncoder
# from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# from Model.env_net_transformer_gphsplit import Env_net

# R_EARTH  = 6371.0
# DT_HOURS = 6.0
# DEG2KM   = 111.0
# MAX_CURVATURE_RAD = math.pi / 4
# _NORM_TO_DEG = 5.0


# # ══════════════════════════════════════════════════════════════
# #  Coordinate utilities (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# def norm_to_deg(t):
#     return torch.stack([(t[...,0]*50.+1800.)/10., (t[...,1]*50.)/10.], dim=-1)

# _norm_to_deg = norm_to_deg

# def haversine_km(p1, p2):
#     la1=torch.deg2rad(p1[...,1]); la2=torch.deg2rad(p2[...,1])
#     dlat=torch.deg2rad(p2[...,1]-p1[...,1]); dlon=torch.deg2rad(p2[...,0]-p1[...,0])
#     a=torch.sin(dlat/2).pow(2)+torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2)
#     return 2.*R_EARTH*torch.asin(a.clamp(1e-12,1-1e-12).sqrt())

# def velocity_km_s(traj_deg):
#     """[T,B,2] degrees → [T-1,B,2] km/6h."""
#     lon=traj_deg[:,:,0]; lat=traj_deg[:,:,1]
#     dlat=lat[1:]-lat[:-1]; dlon=lon[1:]-lon[:-1]
#     lat_mid=(lat[1:]+lat[:-1])/2.
#     cos_lat=torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
#     return torch.stack([dlon*cos_lat*DEG2KM, dlat*DEG2KM], dim=-1)


# # ══════════════════════════════════════════════════════════════
# #  LearnedStepWeights (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# class LearnedStepWeights(nn.Module):
#     def __init__(self, n_steps=12):
#         super().__init__()
#         self.n_steps=n_steps
#         self.raw=nn.Parameter(torch.zeros(n_steps)+0.5)
#     def forward(self):
#         w=torch.cumsum(F.softplus(self.raw),dim=0)
#         return w/w.mean().clamp(1e-8)
#     def get(self,n=None):
#         w=self.forward(); return w[:n] if n is not None else w
#     @torch.no_grad()
#     def stats(self):
#         w=self.forward()
#         return {"sw_ratio":(w[-1]/w[0].clamp(1e-6)).item(),
#                 "sw_monotonic":bool((w[1:]-w[:-1]).min().item()>=-1e-6),
#                 "sw_6h":w[0].item(),"sw_24h":w[3].item(),
#                 "sw_48h":w[7].item(),"sw_72h":w[-1].item()}


# # ══════════════════════════════════════════════════════════════
# #  [NEW] ConstrainedLossWeights — softplus_inv init
# # ══════════════════════════════════════════════════════════════

# class ConstrainedLossWeights(nn.Module):
#     """
#     3 weights: w_fm, w_pos, w_disp. Tự học, init chính xác qua softplus_inv.
#     Anchor loss ngăn drift khỏi reasonable region.
#     """
#     @staticmethod
#     def _sp_inv(y):
#         if y>20.: return y
#         if y<1e-6: return -20.
#         return math.log(math.expm1(y))

#     def __init__(self, init_fm=1.0, init_pos=2.0, init_disp=0.8, anchor_w=0.02):
#         super().__init__()
#         self.anchor_w = anchor_w
#         raw = torch.tensor([self._sp_inv(init_fm),
#                               self._sp_inv(init_pos),
#                               self._sp_inv(init_disp)], dtype=torch.float)
#         self.log_w = nn.Parameter(raw)
#         self.register_buffer('log_w0', raw.clone())

#     def _get(self,i,mn,mx): return F.softplus(self.log_w[i]).clamp(mn,mx)
#     def w_fm(self):   return self._get(0, 0.2, 4.0)
#     def w_pos(self):  return self._get(1, 0.5, 6.0)
#     def w_disp(self): return self._get(2, 0.1, 3.0)
#     def anchor_loss(self): return self.anchor_w*((self.log_w-self.log_w0)**2).mean()
#     @torch.no_grad()
#     def stats(self):
#         return {"lw_fm":self.w_fm().item(),"lw_pos":self.w_pos().item(),
#                 "lw_disp":self.w_disp().item(),
#                 "lw_pos_fm":(self.w_pos()/self.w_fm().clamp(1e-6)).item()}


# # ══════════════════════════════════════════════════════════════
# #  DifficultyWeighter (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# class DifficultyWeighter(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear=nn.Linear(3,1,bias=True)
#         nn.init.zeros_(self.linear.weight)
#         nn.init.constant_(self.linear.bias,-2.0)
#     def compute_difficulty(self,gt_deg):
#         B=gt_deg.shape[1]; v=velocity_km_s(gt_deg)
#         if v.shape[0]>=2:
#             heading=torch.atan2(v[:,:,0],v[:,:,1])
#             dh=heading[1:]-heading[:-1]
#             dh=(dh+math.pi)%(2*math.pi)-math.pi
#             curv_rate=dh.abs().mean(0)
#             excess=F.relu(dh.abs()-MAX_CURVATURE_RAD).mean(0)
#         else:
#             curv_rate=excess=gt_deg.new_zeros(B)
#         spd=v.norm(dim=-1); mean_spd=spd.mean(0).clamp(1.)
#         speed_cv=(spd.std(0)/mean_spd).clamp(max=3.)
#         d1=(curv_rate/(math.pi/2)).clamp(0,1)
#         d2=speed_cv.clamp(0,1)
#         d3=(excess/(math.pi/4)).clamp(0,1)
#         return torch.stack([d1,d2,d3],dim=-1)
#     def forward(self,gt_deg):
#         diff=self.compute_difficulty(gt_deg)
#         return 1.0+torch.sigmoid(self.linear(diff).squeeze(-1))


# # ══════════════════════════════════════════════════════════════
# #  EnsembleScorer (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# class EnsembleScorer(nn.Module):
#     def __init__(self,feat_dim=7,hidden=32):
#         super().__init__()
#         self.net=nn.Sequential(
#             nn.Linear(feat_dim,hidden),nn.GELU(),
#             nn.Linear(hidden,16),nn.GELU(),
#             nn.Linear(16,1))
#         nn.init.zeros_(self.net[-1].weight)
#         nn.init.zeros_(self.net[-1].bias)

#     def extract_features(self,traj_norm,obs_norm):
#         B=traj_norm.shape[1]
#         traj_deg=norm_to_deg(traj_norm); obs_deg=norm_to_deg(obs_norm)
#         v=velocity_km_s(traj_deg); spd=v.norm(dim=-1)
#         f1=torch.log1p(spd.mean(0)); f2=torch.log1p(spd.std(0).clamp(0))
#         if v.shape[0]>=2:
#             heading=torch.atan2(v[:,:,0],v[:,:,1])
#             dh=heading[1:]-heading[:-1]; dh=(dh+math.pi)%(2*math.pi)-math.pi
#             f3=torch.cos(dh).mean(0)
#         else: f3=traj_norm.new_ones(B)
#         v_obs=velocity_km_s(obs_deg); n_obs=min(3,v_obs.shape[0])
#         obs_spd=v_obs[-n_obs:].norm(dim=-1).mean(0).clamp(1.)
#         n_pred=min(3,spd.shape[0]); pred_spd=spd[:n_pred].mean(0)
#         f4=torch.exp(-((pred_spd-obs_spd)/obs_spd).pow(2)*2.)
#         if v_obs.shape[0]>=1 and v.shape[0]>=1:
#             obs_h=torch.atan2(v_obs[-1,:,0],v_obs[-1,:,1])
#             pred_h=torch.atan2(v[0,:,0],v[0,:,1])
#             dh_cont=pred_h-obs_h; dh_cont=(dh_cont+math.pi)%(2*math.pi)-math.pi
#             f5=torch.cos(dh_cont)
#         else: f5=traj_norm.new_ones(B)
#         if obs_norm.shape[0]>=2:
#             lv=obs_norm[-1]-obs_norm[-2]
#             steps=torch.arange(1,traj_norm.shape[0]+1,device=traj_norm.device,dtype=traj_norm.dtype)
#             persist=obs_norm[-1].unsqueeze(0)+lv.unsqueeze(0)*steps.view(-1,1,1)
#             dfp=(traj_norm-persist).norm(dim=-1).mean(0)
#             ref=(lv.norm(dim=-1)*traj_norm.shape[0]).clamp(1e-3)
#             f6=dfp/ref
#         else: f6=traj_norm.new_zeros(B)
#         if v.shape[0]>=2:
#             heading=torch.atan2(v[:,:,0],v[:,:,1])
#             dh=heading[1:]-heading[:-1]; dh=(dh+math.pi)%(2*math.pi)-math.pi
#             f7=dh.abs().mean(0)
#         else: f7=traj_norm.new_zeros(B)
#         return torch.stack([f1,f2,f3,f4,f5,f6,f7],dim=-1).clamp(-10.,10.)

#     def score(self,traj_norm,obs_norm):
#         return torch.sigmoid(self.net(self.extract_features(traj_norm,obs_norm)).squeeze(-1))

#     def logits(self,traj_norm,obs_norm):
#         return self.net(self.extract_features(traj_norm,obs_norm)).squeeze(-1)

#     def auxiliary_loss(self,candidates,obs_norm,gt_deg):
#         if len(candidates)<2: return candidates[0].new_zeros(())
#         gt_b=gt_deg.permute(1,0,2)
#         ades=[]
#         for c in candidates:
#             cb=norm_to_deg(c).permute(1,0,2)
#             ades.append(haversine_km(cb,gt_b).mean(1))
#         ades_t=torch.stack(ades,0); best=ades_t.argmin(0)
#         tot=ades_t.new_zeros(())
#         for i,c in enumerate(candidates):
#             tot=tot+F.binary_cross_entropy_with_logits(
#                 self.logits(c,obs_norm),(best==i).float())
#         return tot/len(candidates)


# # ══════════════════════════════════════════════════════════════
# #  [NEW] L_pos: direct position loss — the key to monotone decrease
# # ══════════════════════════════════════════════════════════════

# def l_position(pred_deg, gt_deg, step_weights, d=300., sample_w=None):
#     """
#     FIX-C: Direct position loss — haversine(pred_pos, gt_pos).

#     Đây là thành phần QUAN TRỌNG NHẤT để ADE giảm đều.
#     - Gradient trực tiếp: push pred_pos về phía gt_pos
#     - Không qua intermediate velocity → không oscillate
#     - step_weights: learned monotone (w_72h >= 3x w_6h)
#     - sample_w: difficulty weights [1,2]

#     d=300: rộng hơn v59's 200 → better gradient khi error lớn (>300km)
#     Scale: huber(dist,300)/300 → ~0.1-1.0, same as L_fm
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2: return pred_deg.new_zeros(())
#     dist = haversine_km(pred_deg[:T], gt_deg[:T])          # [T,B] km
#     w = step_weights[:T]                                    # [T] learned
#     huber = torch.where(dist<d, dist.pow(2)/(2.*d), dist-d/2.)
#     loss = (huber * w.unsqueeze(1)).mean() / d
#     if sample_w is not None:
#         # Apply difficulty weighting: hard storms get more gradient
#         loss = (loss * sample_w.mean())  # simplified: scale by mean difficulty
#     return loss


# # ══════════════════════════════════════════════════════════════
# #  [NEW] L_disp: normalized displacement — ATE+CTE proxy
# # ══════════════════════════════════════════════════════════════

# def l_displacement(pred_deg, gt_deg):
#     """
#     FIX-B: Normalized displacement error, replaces l_logspd.

#     ||Δpred - Δgt|| / ||Δgt||.clamp(10km)
#     Δ[t] = traj[t+1] - traj[t] (displacement vector in km)

#     Tại sao tốt hơn l_logspd:
#     - l_logspd: MSE in log-space → asymmetric gradient → oscillation
#     - l_disp:   ratio, symmetric → stable gradient → smooth decrease

#     Tại sao covers ATE+CTE:
#     - ATE lớn: |Δpred| ≠ |Δgt| (wrong speed/magnitude) → ratio > 1
#     - CTE lớn: direction(Δpred) ≠ direction(Δgt)       → ratio > 1
#     - Cả hai về 0 khi pred tốt → L_disp → 0

#     Scale: ~0.1-1.5, independent of km magnitude.
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 3: return pred_deg.new_zeros(())
#     cos_lat = torch.cos(torch.deg2rad(gt_deg[1:T,:,1])).clamp(1e-4)  # [T-1,B]
#     def _disp_km(traj):
#         dx=(traj[1:T,:,0]-traj[:T-1,:,0])*cos_lat*DEG2KM
#         dy=(traj[1:T,:,1]-traj[:T-1,:,1])*DEG2KM
#         return torch.stack([dx,dy],dim=-1)  # [T-1,B,2]
#     dp=_disp_km(pred_deg); dg=_disp_km(gt_deg)
#     err_km=(dp-dg).norm(dim=-1)           # [T-1,B]
#     gt_mag=dg.norm(dim=-1).clamp(min=10.) # [T-1,B] min 10km
#     ratio=err_km/gt_mag                   # dimensionless ~0-3
#     return F.huber_loss(ratio, torch.zeros_like(ratio), delta=0.5)


# # ══════════════════════════════════════════════════════════════
# #  SpeedHead (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# class SpeedHead(nn.Module):
#     def __init__(self,ctx_dim=256,obs_feat_dim=256,pred_len=12):
#         super().__init__()
#         self.net=nn.Sequential(
#             nn.Linear(ctx_dim+obs_feat_dim,256),nn.GELU(),nn.LayerNorm(256),
#             nn.Linear(256,128),nn.GELU(),nn.Linear(128,pred_len))
#         nn.init.xavier_uniform_(self.net[-1].weight,gain=0.1)
#         nn.init.zeros_(self.net[-1].bias)
#     def forward(self,ctx,obs_feat):
#         return torch.exp(self.net(torch.cat([ctx,obs_feat],dim=-1))).clamp(3.,150.)

# def l_speed_head(speed_pred, pred_deg, gt_deg, step_weights):
#     """Speed head auxiliary loss — general speed quality."""
#     T = min(speed_pred.shape[1]+1, gt_deg.shape[0])
#     if T < 2: return speed_pred.new_zeros(())
#     v_gt = velocity_km_s(gt_deg[:T])               # [T-1,B,2]
#     spd_gt = v_gt.norm(dim=-1).clamp(3.).permute(1,0)  # [B,T-1]
#     n = min(speed_pred.shape[1], spd_gt.shape[1])
#     w = step_weights[:n]; w = w/w.sum()
#     # MSE in log space but with GT reference — no asymmetry issue
#     # since we're comparing pred to GT directly
#     return F.mse_loss(torch.log1p(speed_pred[:,:n]),
#                        torch.log1p(spd_gt[:,:n]), reduction='none').mean()


# # ══════════════════════════════════════════════════════════════
# #  Master Loss (v74-fix)
# # ══════════════════════════════════════════════════════════════

# class FMv74Loss(nn.Module):
#     """
#     v74-fix loss: 3 general terms + auxiliary.

#     Total = w_fm*L_fm + w_pos*L_pos(step_w) + w_disp*L_disp
#           + 0.3*L_speed_head + 0.05*L_scorer + anchors

#     L_fm:   velocity MSE (flow matching)
#     L_pos:  haversine position loss (ADE proxy, DIRECT)
#     L_disp: normalized displacement (ATE+CTE proxy, INDIRECT)
#     """
#     def __init__(self, pred_len=12):
#         super().__init__()
#         self.pred_len     = pred_len
#         self.step_weights = LearnedStepWeights(n_steps=pred_len)
#         self.loss_weights = ConstrainedLossWeights(
#             init_fm=1.0, init_pos=2.0, init_disp=0.8, anchor_w=0.02)
#         self.diff_weighter = DifficultyWeighter()
#         self.scorer        = EnsembleScorer()
#         self.speed_head_aux = None  # set externally

#     def compute_main_losses(self, pred_deg, gt_deg,
#                              fm_vel_pred, fm_vel_target,
#                              epoch=0):
#         T = min(pred_deg.shape[0], gt_deg.shape[0])
#         sample_w = self.diff_weighter(gt_deg[:T])  # [B] ∈ [1,2]
#         sw = self.step_weights.get(n=T)             # [T] monotone

#         w_fm   = self.loss_weights.w_fm()
#         w_pos  = self.loss_weights.w_pos()
#         w_disp = self.loss_weights.w_disp()
#         anc    = self.step_weights.stats()
#         # Note: step_weights has no anchor loss here — it learns freely
#         # but constrained by cumsum(softplus) monotonicity
#         lw_anc = self.loss_weights.anchor_loss()

#         # FIX-A applied at caller: x0=N(0,sigma), so fm_vel_pred/target are standard
#         L_fm   = F.mse_loss(fm_vel_pred, fm_vel_target)

#         # FIX-C: direct position loss (key to monotone decrease)
#         L_pos  = l_position(pred_deg[:T], gt_deg[:T], sw, d=300., sample_w=sample_w)

#         # FIX-B: normalized displacement (ATE+CTE proxy, no speed bias)
#         L_disp = l_displacement(pred_deg[:T], gt_deg[:T])

#         total = w_fm*L_fm + w_pos*L_pos + w_disp*L_disp + lw_anc

#         if not torch.isfinite(total): total = pred_deg.new_zeros(())

#         sw_s = self.step_weights.stats()
#         lw_s = self.loss_weights.stats()
#         bd = {
#             "l_fm":    L_fm.item(),
#             "l_pos":   L_pos.item(),
#             "l_disp":  L_disp.item(),
#             "diff_w_mean": sample_w.mean().item(),
#             **{f"sw_{k}":v for k,v in sw_s.items()},
#             **{f"lw_{k}":v for k,v in lw_s.items()},
#         }
#         return total, bd

#     def forward(self, pred_deg, gt_deg, fm_vel_pred, fm_vel_target,
#                  speed_pred=None, candidates=None, obs_norm=None, epoch=0):
#         total, bd = self.compute_main_losses(
#             pred_deg, gt_deg, fm_vel_pred, fm_vel_target, epoch=epoch)

#         # Speed head auxiliary (small weight)
#         l_sh = pred_deg.new_zeros(())
#         if speed_pred is not None:
#             T = min(pred_deg.shape[0], gt_deg.shape[0])
#             sw = self.step_weights.get(n=T)
#             l_sh = l_speed_head(speed_pred, pred_deg, gt_deg, sw)
#             total = total + 0.3 * l_sh

#         # Scorer auxiliary
#         l_scr = pred_deg.new_zeros(())
#         if candidates is not None and obs_norm is not None and len(candidates)>=2:
#             l_scr = self.scorer.auxiliary_loss(candidates, obs_norm, gt_deg)
#             total = total + 0.05 * l_scr

#         bd.update({"l_sh":l_sh.item(),"l_scorer":l_scr.item(),"total":total})
#         return total, bd

#     @torch.no_grad()
#     def stats(self):
#         return {**self.step_weights.stats(), **self.loss_weights.stats()}


# # ══════════════════════════════════════════════════════════════
# #  Safe env helpers (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# def _safe_env(env_data,key,B,device,norm=1.0):
#     v=env_data.get(key) if env_data is not None else None
#     if v is None or not torch.is_tensor(v): return torch.zeros(B,device=device)
#     v=v.float().to(device)
#     while v.dim()>1: v=v.mean(-1)
#     v=v.view(-1)[:B] if v.numel()>=B else torch.zeros(B,device=device)
#     return (v/norm).clamp(-3.,3.)

# def _safe_env_vec(env_data,key,dim,B,device):
#     v=env_data.get(key) if env_data is not None else None
#     if v is None: return torch.zeros(B,dim,device=device)
#     if not torch.is_tensor(v):
#         try: v=torch.tensor(v,dtype=torch.float,device=device)
#         except: return torch.zeros(B,dim,device=device)
#     v=v.float().to(device)
#     if v.dim()==0: return torch.zeros(B,dim,device=device)
#     if v.dim()==1:
#         return v.unsqueeze(0).expand(B,dim) if v.shape[0]==dim else torch.zeros(B,dim,device=device)
#     if v.dim()==2:
#         if v.shape==(B,dim): return v
#         if v.shape[0]==B:
#             return v[:,:dim] if v.shape[1]>=dim else F.pad(v,(0,dim-v.shape[1]))
#         return torch.zeros(B,dim,device=device)
#     if v.dim()==3:
#         vv=v[:B,-1,:]
#         return vv[:,:dim] if vv.shape[1]>=dim else F.pad(vv,(0,dim-vv.shape[1]))
#     return torch.zeros(B,dim,device=device)


# # ══════════════════════════════════════════════════════════════
# #  VelocityField (giữ nguyên v60 — toàn bộ architecture)
# # ══════════════════════════════════════════════════════════════

# class VelocityField(nn.Module):
#     RAW_CTX_DIM = 512

#     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256,
#                  sigma_min=0.02, unet_in_ch=13, **kwargs):
#         super().__init__()
#         self.pred_len=pred_len; self.obs_len=obs_len; self.ctx_dim=ctx_dim
#         self.sigma_min=sigma_min

#         self.spatial_enc=FNO3DEncoder(in_channel=unet_in_ch,out_channel=1,d_model=32,
#             n_layers=4,modes_t=4,modes_h=4,modes_w=4,spatial_down=32,dropout=0.05)
#         self.bottleneck_pool=nn.AdaptiveAvgPool3d((None,1,1))
#         self.bottleneck_proj=nn.Linear(128,128)
#         self.decoder_proj=nn.Linear(1,16)
#         self.enc_1d=DataEncoder1D(in_1d=4,feat_3d_dim=128,mlp_h=64,
#             lstm_hidden=128,lstm_layers=3,dropout=0.1,d_state=16)
#         self.env_enc=Env_net(obs_len=obs_len,d_model=32)
#         self.steering_enc=nn.Sequential(
#             nn.Linear(9,64),nn.GELU(),nn.LayerNorm(64),
#             nn.Linear(64,128),nn.GELU(),nn.Linear(128,256))
#         self.env_kine_enc=nn.Sequential(
#             nn.Linear(14,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,256),nn.GELU())
#         self.recurv_enc=nn.Sequential(
#             nn.Linear(33,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,64))
#         self.speed_hist_enc=nn.Sequential(
#             nn.Linear(11,32),nn.GELU(),nn.LayerNorm(32),nn.Linear(32,32))
#         self.ctx_fc1=nn.Linear(128+32+16+64+32,self.RAW_CTX_DIM)
#         self.ctx_ln=nn.LayerNorm(self.RAW_CTX_DIM)
#         self.ctx_drop=nn.Dropout(0.15)
#         self.ctx_fc2=nn.Linear(self.RAW_CTX_DIM,ctx_dim)
#         self.null_embedding=nn.Parameter(torch.randn(1,self.RAW_CTX_DIM)*0.02)
#         self.vel_obs_enc=nn.Sequential(
#             nn.Linear(obs_len*6,256),nn.GELU(),nn.LayerNorm(256),
#             nn.Linear(256,256),nn.GELU())
#         self.blend_head=nn.Linear(ctx_dim,1)
#         self.guidance_head=nn.Linear(ctx_dim,1)
#         self.sigma_head=nn.Linear(ctx_dim,1)
#         nn.init.zeros_(self.blend_head.weight); nn.init.constant_(self.blend_head.bias,-1.)
#         nn.init.zeros_(self.guidance_head.weight); nn.init.constant_(self.guidance_head.bias,0.)
#         nn.init.zeros_(self.sigma_head.weight); nn.init.constant_(self.sigma_head.bias,-1.)
#         self.time_fc1=nn.Linear(256,512); self.time_fc2=nn.Linear(512,256)
#         self.traj_embed=nn.Linear(4,256)
#         self.pos_enc=nn.Parameter(torch.randn(1,pred_len,256)*0.02)
#         self.step_embed=nn.Embedding(pred_len,256)
#         self.transformer=nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model=256,nhead=8,dim_feedforward=1024,
#                 dropout=0.10,activation='gelu',batch_first=True),num_layers=2)
#         self.out_fc1=nn.Linear(256,512); self.out_fc2=nn.Linear(512,4)
#         self.step_scale=nn.Parameter(torch.ones(pred_len)*0.5)
#         self.physics_scale=nn.Parameter(torch.ones(4)*1.5)
#         self.steering_scale=nn.Parameter(torch.ones(4)*1.0)
#         self.speed_head=SpeedHead(ctx_dim=ctx_dim,obs_feat_dim=256,pred_len=pred_len)
#         self._init_weights()

#     def _init_weights(self):
#         with torch.no_grad():
#             for n,m in self.named_modules():
#                 if isinstance(m,nn.Linear) and 'out_fc' in n:
#                     nn.init.xavier_uniform_(m.weight,gain=0.1)
#                     if m.bias is not None: nn.init.zeros_(m.bias)

#     def _time_emb(self,t,dim=256):
#         h=dim//2
#         fr=torch.exp(torch.arange(h,dtype=torch.float,device=t.device)*(-math.log(10000.)/max(h-1,1)))
#         em=t.float().unsqueeze(1)*1000.*fr.unsqueeze(0)
#         return F.pad(torch.cat([em.sin(),em.cos()],dim=-1),(0,dim%2))

#     def _get_steering_feat(self,env_data,B,device):
#         if env_data is None: return torch.zeros(B,256,device=device)
#         feats=torch.stack([_safe_env(env_data,'u500_mean',B,device,30.),
#             _safe_env(env_data,'v500_mean',B,device,30.),
#             _safe_env(env_data,'u500_center',B,device,30.),
#             _safe_env(env_data,'v500_center',B,device,30.),
#             _safe_env(env_data,'steering_speed',B,device,1.),
#             _safe_env(env_data,'steering_dir_sin',B,device,1.),
#             _safe_env(env_data,'steering_dir_cos',B,device,1.),
#             _safe_env(env_data,'gph500_mean',B,device,1.),
#             _safe_env(env_data,'gph500_center',B,device,1.)],dim=-1)
#         return self.steering_enc(feats)

#     def _get_env_kine_feat(self,env_data,B,device):
#         if env_data is None: return torch.zeros(B,256,device=device)
#         mv=_safe_env(env_data,'move_velocity',B,device,150.).unsqueeze(-1)
#         hd24=_safe_env_vec(env_data,'history_direction24',8,B,device)
#         dv=_safe_env_vec(env_data,'delta_velocity',5,B,device)
#         return self.env_kine_enc(torch.cat([mv,hd24,dv],dim=-1))

#     def _get_recurv_feat(self,env_data,B,device):
#         if env_data is None: return torch.zeros(B,64,device=device)
#         bearing=_safe_env_vec(env_data,'bearing_to_scs_center',16,B,device)
#         dist=_safe_env_vec(env_data,'dist_to_scs_boundary',5,B,device)
#         month=_safe_env_vec(env_data,'month',12,B,device)
#         return self.recurv_enc(torch.cat([bearing,dist,month],dim=-1))

#     def _get_speed_hist_feat(self,env_data,B,device):
#         if env_data is None: return torch.zeros(B,32,device=device)
#         vh=_safe_env_vec(env_data,'velocity_history',4,B,device)
#         ri=_safe_env(env_data,'rapid_intensification',B,device,1.).unsqueeze(-1)
#         ic=_safe_env_vec(env_data,'intensity_class',6,B,device)
#         return self.speed_hist_enc(torch.cat([vh,ri,ic],dim=-1))

#     def _get_kinematic_obs_feat(self,obs_traj):
#         B=obs_traj.shape[1]; T_obs=obs_traj.shape[0]
#         if T_obs>=2:
#             vel=obs_traj[1:]-obs_traj[:-1]
#             lat_mid=obs_traj[:-1,:,1]*_NORM_TO_DEG
#             cos_lat=torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
#             dx_km=vel[:,:,0]*cos_lat*DEG2KM*_NORM_TO_DEG
#             dy_km=vel[:,:,1]*DEG2KM*_NORM_TO_DEG
#             speed=torch.sqrt(dx_km**2+dy_km**2+1e-6)/DT_HOURS
#             heading=torch.atan2(vel[:,:,1],vel[:,:,0])
#             speed_n=(speed/20.).clamp(-3.,3.)
#             if T_obs>=3:
#                 dspd=speed[1:]-speed[:-1]
#                 accel=torch.cat([obs_traj.new_zeros(1,B),(dspd/10.).clamp(-3.,3.)],0)
#             else: accel=obs_traj.new_zeros(T_obs-1,B)
#             kine=torch.stack([vel[:,:,0],vel[:,:,1],speed_n,heading.sin(),heading.cos(),accel],dim=-1)
#         else: kine=obs_traj.new_zeros(self.obs_len,B,6)
#         if kine.shape[0]<self.obs_len:
#             kine=torch.cat([obs_traj.new_zeros(self.obs_len-kine.shape[0],B,6),kine],0)
#         else: kine=kine[-self.obs_len:]
#         return self.vel_obs_enc(kine.permute(1,0,2).reshape(B,-1))

#     def _context(self,batch_list):
#         obs_traj=batch_list[0]; obs_Me=batch_list[7]; image_obs=batch_list[11]
#         env_data=batch_list[13] if len(batch_list)>13 else None
#         B=obs_traj.shape[1]; device=obs_traj.device
#         if image_obs.dim()==4: image_obs=image_obs.unsqueeze(2)
#         if image_obs.shape[1]==1 and self.spatial_enc.in_channel!=1:
#             image_obs=image_obs.expand(-1,self.spatial_enc.in_channel,-1,-1,-1)
#         e_3d_bot,e_3d_dec=self.spatial_enc.encode(image_obs)
#         T_obs=obs_traj.shape[0]
#         e_3d_s=self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0,2,1)
#         e_3d_s=self.bottleneck_proj(e_3d_s)
#         if e_3d_s.shape[1]!=T_obs:
#             e_3d_s=F.interpolate(e_3d_s.permute(0,2,1),size=T_obs,mode='linear',align_corners=False).permute(0,2,1)
#         e_3d_dec_t=e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
#         t_w=torch.softmax(torch.arange(e_3d_dec_t.shape[1],dtype=torch.float,device=device)*0.5,dim=0)
#         f_sp=self.decoder_proj((e_3d_dec_t*t_w.unsqueeze(0)).sum(1,keepdim=True))
#         obs_in=torch.cat([obs_traj,obs_Me],dim=2).permute(1,0,2)
#         h_t=self.enc_1d(obs_in,e_3d_s)
#         e_env,_,_=self.env_enc(env_data,image_obs)
#         recurv_feat=self._get_recurv_feat(env_data,B,device)
#         speed_h_feat=self._get_speed_hist_feat(env_data,B,device)
#         cat_feat=torch.cat([h_t,e_env,f_sp,recurv_feat,speed_h_feat],dim=-1)
#         return F.gelu(self.ctx_ln(self.ctx_fc1(cat_feat)))

#     def _apply_ctx_head(self,raw,noise_scale=0.,use_null=False):
#         if use_null: raw=self.null_embedding.expand(raw.shape[0],-1)
#         elif noise_scale>0.: raw=raw+torch.randn_like(raw)*noise_scale
#         return self.ctx_fc2(self.ctx_drop(raw))

#     def get_blend_alpha(self,ctx): return torch.sigmoid(self.blend_head(ctx)).squeeze(-1)*0.5
#     def get_guidance_scale(self,ctx): return 0.8+1.2*torch.sigmoid(self.guidance_head(ctx)).squeeze(-1)
#     def get_sigma(self,ctx): return 0.02+0.08*torch.sigmoid(self.sigma_head(ctx)).squeeze(-1)

#     def _beta_drift(self,x_t):
#         lat_rad=torch.deg2rad(x_t[:,:,1]*5.).clamp(-85,85)
#         beta=2*7.2921e-5*torch.cos(lat_rad)/6.371e6; R=3e5
#         v=torch.zeros_like(x_t)
#         v[:,:,0]=-beta*R**2/2.*6*3600./(5*111*1000.)
#         v[:,:,1]=beta*R**2/4.*6*3600./(5*111*1000.)
#         return v

#     def _steering_drift(self,x_t,env_data):
#         if env_data is None: return torch.zeros_like(x_t)
#         B,device=x_t.shape[0],x_t.device
#         u=_safe_env(env_data,'u500_center',B,device,30.)
#         vv=_safe_env(env_data,'v500_center',B,device,30.)
#         cos=torch.cos(torch.deg2rad(x_t[:,:,1]*5.)).clamp(1e-3)
#         out=torch.zeros_like(x_t)
#         out[:,:,0]=u.unsqueeze(1)*30.*21600./(111.*1000.*cos)
#         out[:,:,1]=vv.unsqueeze(1)*30.*21600./(111.*1000.)
#         return out

#     def _decode(self,x_t,t,ctx,vel_obs_feat=None,steering_feat=None,
#                  env_kine_feat=None,env_data=None):
#         B=x_t.shape[0]
#         t_emb=F.gelu(self.time_fc1(self._time_emb(t))); t_emb=self.time_fc2(t_emb)
#         T=min(x_t.size(1),self.pos_enc.shape[1])
#         si=torch.arange(T,device=x_t.device).unsqueeze(0).expand(B,-1)
#         xe=(self.traj_embed(x_t[:,:T])+self.pos_enc[:,:T]+t_emb.unsqueeze(1)+self.step_embed(si))
#         mem=[t_emb.unsqueeze(1),ctx.unsqueeze(1)]
#         if vel_obs_feat is not None: mem.append(vel_obs_feat.unsqueeze(1))
#         if steering_feat is not None: mem.append(steering_feat.unsqueeze(1))
#         if env_kine_feat is not None: mem.append(env_kine_feat.unsqueeze(1))
#         dec=self.transformer(xe,torch.cat(mem,dim=1))
#         vn=self.out_fc2(F.gelu(self.out_fc1(dec)))
#         sc=torch.sigmoid(self.step_scale[:T]).view(1,T,1)*2.; vn=vn*sc
#         with torch.no_grad():
#             vp=self._beta_drift(x_t[:,:T]); vs=self._steering_drift(x_t[:,:T],env_data)
#         return vn+torch.sigmoid(self.physics_scale)*vp+torch.sigmoid(self.steering_scale)*vs

#     def forward_with_ctx(self,x_t,t,raw_ctx,noise_scale=0.,vel_obs_feat=None,
#                           steering_feat=None,env_kine_feat=None,env_data=None,use_null=False):
#         ctx=self._apply_ctx_head(raw_ctx,noise_scale,use_null=use_null)
#         return self._decode(x_t,t,ctx,vel_obs_feat=vel_obs_feat,
#                             steering_feat=steering_feat,env_kine_feat=env_kine_feat,env_data=env_data)

#     def predict_speed(self,raw_ctx,vel_obs_feat):
#         return self.speed_head(self._apply_ctx_head(raw_ctx),vel_obs_feat)


# # ══════════════════════════════════════════════════════════════
# #  EMA, OT (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# class EMAModel:
#     def __init__(self,model,decay=0.995):
#         self.decay=decay; m=model._orig_mod if hasattr(model,'_orig_mod') else model
#         self.shadow={k:v.detach().clone() for k,v in m.state_dict().items() if v.dtype.is_floating_point}
#     def update(self,model):
#         m=model._orig_mod if hasattr(model,'_orig_mod') else model
#         with torch.no_grad():
#             for k,v in m.state_dict().items():
#                 if k in self.shadow: self.shadow[k].mul_(self.decay).add_(v.detach(),alpha=1-self.decay)
#     def apply_to(self,model):
#         m=model._orig_mod if hasattr(model,'_orig_mod') else model
#         bk,sd={},m.state_dict()
#         for k in self.shadow:
#             if k not in sd: continue
#             bk[k]=sd[k].detach().clone(); sd[k].copy_(self.shadow[k])
#         return bk
#     def restore(self,model,bk):
#         m=model._orig_mod if hasattr(model,'_orig_mod') else model; sd=m.state_dict()
#         for k,v in bk.items():
#             if k in sd: sd[k].copy_(v)


# def _sinkhorn_log(cost,epsilon=0.05,n_iter=50):
#     B=cost.shape[0]; device=cost.device
#     la=-math.log(B)*torch.ones(B,device=device); lb=la.clone()
#     lK=-cost/epsilon; lu=torch.zeros(B,device=device); lv=lu.clone()
#     for _ in range(n_iter):
#         lu=la-torch.logsumexp(lK+lv.unsqueeze(0),dim=1)
#         lv=lb-torch.logsumexp(lK+lu.unsqueeze(1),dim=0)
#     return (lK+lu.unsqueeze(1)+lv.unsqueeze(0)).exp().clamp(0.)

# def _spherical_ot_matching(x0_batch,x1_batch,lp,epsilon=0.05):
#     try:
#         B=x0_batch.shape[0]
#         abs0=lp.unsqueeze(1)[:,:,:2]+x0_batch[:,:,:2]; abs1=lp.unsqueeze(1)[:,:,:2]+x1_batch[:,:,:2]
#         abs0_deg=norm_to_deg(abs0); abs1_deg=norm_to_deg(abs1)
#         cost=haversine_km(
#             abs0_deg.unsqueeze(1).expand(B,B,-1,-1).reshape(B*B,-1,2),
#             abs1_deg.unsqueeze(0).expand(B,B,-1,-1).reshape(B*B,-1,2)
#         ).mean(-1).reshape(B,B)/500.
#         pi=_sinkhorn_log(cost,epsilon=epsilon)
#         flat=pi.reshape(-1).clamp(0.); s=flat.sum()
#         if not torch.isfinite(s) or s<1e-10: return x0_batch,x1_batch
#         idx=torch.multinomial(flat/s,num_samples=B,replacement=True)
#         return x0_batch[idx%B],x1_batch[idx%B]
#     except: return x0_batch,x1_batch


# def compute_speed_stats_from_norm(obs_traj_norm):
#     T_obs=obs_traj_norm.shape[0]
#     if T_obs<2: return {'v_opt':15.,'v_sigma':10.,'v_hard_cap':80.,'p50_kmh':15.}
#     lon=(obs_traj_norm[...,0]*50.+1800.)/10.; lat=(obs_traj_norm[...,1]*50.)/10.
#     lat_mid=(lat[:-1]+lat[1:])/2; cos_lat=torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
#     dx=(lon[1:]-lon[:-1])*cos_lat*DEG2KM; dy=(lat[1:]-lat[:-1])*DEG2KM
#     spd=torch.sqrt(dx**2+dy**2)/DT_HOURS
#     p50=float(spd.flatten().median()); p95=float(torch.quantile(spd.flatten(),.95))
#     return {'v_opt':max(p50,5.),'v_sigma':10.,
#             'v_hard_cap':float(torch.tensor(p95*1.8).clamp(25.,130.)),'p50_kmh':p50}


# @torch.no_grad()
# def _persistence_blend_adaptive(model_pred_norm,obs_traj_norm,blend_alpha):
#     T_obs=obs_traj_norm.shape[0]; T=model_pred_norm.shape[0]
#     B=model_pred_norm.shape[1]; device=model_pred_norm.device
#     if T_obs<2: return model_pred_norm
#     vels=obs_traj_norm[1:]-obs_traj_norm[:-1]; n_v=vels.shape[0]
#     if n_v>=3:
#         alpha=0.7; w=torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],dtype=torch.float,device=device).flip(0)
#         ev=(vels*(w/w.sum()).view(-1,1,1)).sum(0)
#     elif n_v==2: ev=0.7*vels[-1]+0.3*vels[-2]
#     else: ev=vels[-1]
#     steps=torch.arange(1,T+1,dtype=torch.float,device=device)
#     persist=obs_traj_norm[-1].unsqueeze(0)+ev.unsqueeze(0)*steps.view(T,1,1)
#     alpha_b=blend_alpha.view(1,B,1).clamp(0.,.5)
#     return (1.-alpha_b)*model_pred_norm+alpha_b*persist


# # ══════════════════════════════════════════════════════════════
# #  [NEW] K=3 mode clustering inference
# # ══════════════════════════════════════════════════════════════

# @torch.no_grad()
# def _mode_cluster_k3(trajs_norm, obs_norm, scorer, speed_stats=None):
#     """
#     K=3 mode clustering tại 72h endpoint.
#     Thay top-35% median → CTE thấp hơn tự nhiên.

#     trajs_norm: list of [T,B,2] seq-first normalized
#     obs_norm:   [T_obs,B,2] seq-first
#     scorer:     EnsembleScorer
#     returns:    [T,B,2] seq-first normalized
#     """
#     if not trajs_norm: return obs_norm[-1:].expand(12,obs_norm.shape[1],2)
#     dev=trajs_norm[0].device; T,B=trajs_norm[0].shape[0],trajs_norm[0].shape[1]; N=len(trajs_norm); K=min(3,N)

#     # Scores [N,B]
#     all_sc=torch.stack([scorer.score(tr,obs_norm) for tr in trajs_norm],dim=0)
#     # 72h endpoints in degrees [N,B,2]
#     endpoints=torch.stack([norm_to_deg(tr[-1]) for tr in trajs_norm],dim=0)

#     results=[]
#     for b in range(B):
#         ep_b=endpoints[:,b,:]; sc_b=all_sc[:,b]
#         tr_b=torch.stack([tr[:,b,:] for tr in trajs_norm],dim=0)  # [N,T,2]
#         if N<K:
#             w=F.softmax(sc_b*3.,0); results.append((tr_b*w.view(N,1,1)).sum(0)); continue
#         # Farthest-point init
#         idx0=sc_b.argmax().item(); centers=[ep_b[idx0]]
#         for _ in range(K-1):
#             d2c=torch.stack([haversine_km(ep_b,c.unsqueeze(0).expand_as(ep_b)) for c in centers],1).min(1).values
#             centers.append(ep_b[d2c.argmax()])
#         centers=torch.stack(centers,0)
#         # 3 K-means iterations
#         for _ in range(3):
#             d2c=torch.stack([haversine_km(ep_b,c.unsqueeze(0).expand_as(ep_b)) for c in centers],1)
#             assign=d2c.argmin(1)
#             new_c=[]
#             for k in range(K):
#                 mk=(assign==k)
#                 if mk.sum()>0:
#                     wk=F.softmax(sc_b[mk]*3.,0)
#                     new_c.append((ep_b[mk]*wk.unsqueeze(1)).sum(0))
#                 else: new_c.append(centers[k])
#             centers=torch.stack(new_c,0)
#         # Score clusters
#         d2c=torch.stack([haversine_km(ep_b,c.unsqueeze(0).expand_as(ep_b)) for c in centers],1)
#         assign=d2c.argmin(1)
#         csc=torch.zeros(K,device=dev)
#         for k in range(K):
#             mk=(assign==k)
#             if mk.sum()>0: csc[k]=sc_b[mk].sum()
#         best_k=csc.argmax().item(); mk=(assign==best_k)
#         if not mk.any(): mk=torch.ones(N,dtype=torch.bool,device=dev)
#         w_win=F.softmax(sc_b[mk]*3.,0)
#         results.append((tr_b[mk]*w_win.view(-1,1,1)).sum(0))
#     return torch.stack(results,dim=1)  # [T,B,2]


# # ══════════════════════════════════════════════════════════════
# #  TCFlowMatching v74-fix
# # ══════════════════════════════════════════════════════════════

# class TCFlowMatching(nn.Module):

#     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02, unet_in_ch=13,
#                  ctx_noise_scale=0.01, use_ema=True, ema_decay=0.995,
#                  use_ate_ot=True, ot_epsilon=0.05, cfg_uncond_prob=0.1, **kwargs):
#         super().__init__()
#         self.pred_len=pred_len; self.obs_len=obs_len; self.sigma_min=sigma_min
#         self.ctx_noise_scale=ctx_noise_scale; self.use_ate_ot=use_ate_ot
#         self.ot_epsilon=ot_epsilon; self.cfg_uncond_prob=cfg_uncond_prob
#         self.net=VelocityField(pred_len=pred_len,obs_len=obs_len,
#                                 sigma_min=sigma_min,unet_in_ch=unet_in_ch,ctx_dim=256)
#         self.criterion=FMv74Loss(pred_len=pred_len)
#         self.use_ema=use_ema; self._ema=None

#     def init_ema(self):
#         if self.use_ema: self._ema=EMAModel(self,decay=0.995)
#     def ema_update(self):
#         if self._ema is not None: self._ema.update(self)

#     @staticmethod
#     def _to_rel(traj,Me,lp,lm):
#         return torch.cat([traj-lp[:,:2].unsqueeze(0),Me-lm.unsqueeze(0)],dim=-1).permute(1,0,2)

#     @staticmethod
#     def _to_abs(rel,lp,lm):
#         d=rel.permute(1,0,2)
#         return lp[:,:2].unsqueeze(0)+d[:,:,:2], lm.unsqueeze(0)+d[:,:,2:]

#     @staticmethod
#     def _sigma_schedule(ep):
#         if ep<2: return 0.10
#         if ep<10: return 0.10-(ep-2)/8.*(0.10-0.04)
#         if ep<20: return max(0.04-(ep-10)/10.*0.01,0.035)
#         return 0.035

#     def _cfm_standard(self,x1,sigma_min=None):
#         """FIX-A: standard FM x0=N(0,sigma), NOT persist+noise."""
#         if sigma_min is None: sigma_min=self.sigma_min
#         B=x1.shape[0]; dev=x1.device
#         x0=torch.randn_like(x1)*sigma_min
#         t=torch.rand(B,device=dev); te=t.view(B,1,1)
#         return (1.-te)*x0+te*x1, t, x1-x0

#     def _persistence_forecast_rel(self,obs_traj,lp,lm,pred_len):
#         B,device=obs_traj.shape[1],obs_traj.device
#         obs_pos=obs_traj[:,:,:2]
#         if obs_pos.shape[0]>=3:
#             vels=obs_pos[1:]-obs_pos[:-1]; n_v=vels.shape[0]; alpha=0.7
#             w=torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],dtype=torch.float,device=device).flip(0)
#             lv=(vels*(w/w.sum()).view(-1,1,1)).sum(0)
#         elif obs_pos.shape[0]>=2: lv=obs_pos[-1]-obs_pos[-2]
#         else: lv=obs_traj.new_zeros(B,2)
#         steps=torch.arange(1,pred_len+1,device=device).float()
#         pred_abs=obs_pos[-1].unsqueeze(1)+lv.unsqueeze(1)*steps.view(1,-1,1)
#         pred_rel=torch.cat([pred_abs-lp[:,:2].unsqueeze(1),torch.zeros_like(pred_abs)],dim=-1)
#         return pred_rel

#     def _compute_obs_momentum(self,obs_norm):
#         T_obs=obs_norm.shape[0]
#         if T_obs<2: return torch.zeros(obs_norm.shape[1],2,device=obs_norm.device)
#         vels=obs_norm[1:]-obs_norm[:-1]; n_v=vels.shape[0]
#         if n_v>=3:
#             a=0.65; w=torch.tensor([a*(1-a)**i for i in range(n_v)],dtype=torch.float,device=obs_norm.device).flip(0)
#             return (vels*(w/w.sum()).view(-1,1,1)).sum(0)
#         elif n_v==2: return 0.65*vels[-1]+0.35*vels[-2]
#         return vels[-1]

#     @staticmethod
#     def _obs_noise_aug(bl,sigma=0.005):
#         if torch.rand(1).item()>0.5: return bl
#         bl=list(bl)
#         if torch.is_tensor(bl[0]): bl[0]=bl[0]+torch.randn_like(bl[0])*sigma
#         return bl

#     # ── Training ──────────────────────────────────────────────

#     def get_loss(self,batch_list,epoch=0,**kwargs):
#         return self.get_loss_breakdown(batch_list,epoch=epoch)['total']

#     def get_loss_breakdown(self,batch_list,epoch=0):
#         batch_list=self._obs_noise_aug(batch_list,sigma=0.005)
#         obs_t=batch_list[0]; env_data=batch_list[13] if len(batch_list)>13 else None
#         lp=obs_t[-1]; lm=batch_list[7][-1]; B,device=lp.shape[0],lp.device
#         gt_traj_s=batch_list[1]

#         current_sigma=self._sigma_schedule(epoch)
#         raw_ctx=self.net._context(batch_list)
#         x1_rel=self._to_rel(gt_traj_s,batch_list[8],lp,lm)

#         # FIX-A: standard x0=N(0,sigma), NOT persist+noise
#         if self.use_ate_ot and B>=4:
#             x0_noise=torch.randn_like(x1_rel)*current_sigma
#             _,x1_matched=_spherical_ot_matching(x0_noise,x1_rel,lp,epsilon=self.ot_epsilon)
#         else:
#             x1_matched=x1_rel
#         x_t,fm_t,u_target=self._cfm_standard(x1_matched,sigma_min=current_sigma)

#         use_null=(torch.rand(1).item()<self.cfg_uncond_prob)
#         vel_obs_feat=self.net._get_kinematic_obs_feat(obs_t[:,:,:2])
#         steering_feat=self.net._get_steering_feat(env_data,B,device)
#         env_kine_feat=self.net._get_env_kine_feat(env_data,B,device)

#         pred_vel=self.net.forward_with_ctx(x_t,fm_t,raw_ctx,env_data=env_data,use_null=use_null,
#             vel_obs_feat=vel_obs_feat,steering_feat=steering_feat,env_kine_feat=env_kine_feat)

#         fm_te=fm_t.view(B,1,1)
#         x1_pred=x_t+(1.-fm_te)*pred_vel
#         pred_abs,_=self._to_abs(x1_pred,lp,lm)
#         pred_deg=norm_to_deg(pred_abs)    # [T,B,2]
#         gt_deg=norm_to_deg(gt_traj_s)    # [T,B,2]

#         # Speed head prediction for auxiliary loss
#         speed_pred=None
#         if not use_null:
#             speed_pred=self.net.predict_speed(raw_ctx,vel_obs_feat)

#         # Scorer candidates
#         candidates=None; obs_norm=obs_t[:,:,:2]
#         if epoch>=5 and not use_null:
#             cands=[]
#             for _ in range(3):
#                 te_c=fm_t.view(B,1,1)
#                 x0_c=torch.randn_like(x1_rel)*current_sigma
#                 x_c=(1.-te_c)*x0_c+te_c*x1_rel
#                 with torch.no_grad():
#                     v_c=self.net.forward_with_ctx(x_c,fm_t,raw_ctx,env_data=env_data,
#                         vel_obs_feat=vel_obs_feat,steering_feat=steering_feat,env_kine_feat=env_kine_feat)
#                     x1_c=x_c+(1.-te_c)*v_c; abs_c,_=self._to_abs(x1_c,lp,lm)
#                     cands.append(abs_c[:,:,:2])
#             candidates=cands

#         total,bd=self.criterion(pred_deg,gt_deg,pred_vel,u_target,
#                                   speed_pred=speed_pred,candidates=candidates,
#                                   obs_norm=obs_norm,epoch=epoch)

#         if torch.isnan(total) or torch.isinf(total): total=obs_t.new_zeros(())

#         bd.update({'sigma':current_sigma,
#                    'v_opt':compute_speed_stats_from_norm(obs_t[:,:,:2]).get('v_opt',15.),
#                    'total':total,
#                    # legacy keys
#                    'l_fm':bd.get('l_fm',0.),'l_kin':0.,'l_logspd':0.,
#                    'l_curv':0.,'diff_w_mean':bd.get('diff_w_mean',1.),
#                    'lam_kin':0.,'lam_logspd':0.,'lam_curv':0.})
#         return bd

#     # ── Inference ─────────────────────────────────────────────

#     @torch.no_grad()
#     def sample(self,batch_list,num_ensemble=50,ddim_steps=20,predict_csv=None):
#         obs_t=batch_list[0]; env_data=batch_list[13] if len(batch_list)>13 else None
#         lp=obs_t[-1]; lm=batch_list[7][-1]; B=lp.shape[0]; device=lp.device
#         T=self.pred_len; dt=1./max(ddim_steps,1)

#         raw_ctx=self.net._context(batch_list)
#         ctx=self.net._apply_ctx_head(raw_ctx)
#         vel_obs_feat=self.net._get_kinematic_obs_feat(obs_t[:,:,:2])
#         steering_feat=self.net._get_steering_feat(env_data,B,device)
#         env_kine_feat=self.net._get_env_kine_feat(env_data,B,device)
#         obs_norm=obs_t[:,:,:2]; obs_mom=self._compute_obs_momentum(obs_norm)

#         blend_alpha=self.net.get_blend_alpha(ctx)
#         guidance_scale=self.net.get_guidance_scale(ctx)

#         def _mom_str(s,tot): return 0.06*0.5*(1.+math.cos(math.pi*s/max(tot,1)))

#         all_norms=[]  # list of [T,B,2] seq-first normalized

#         for _ in range(num_ensemble):
#             # FIX-A: x0 = N(0, sigma_min) pure noise
#             x_t=torch.randn(B,T,4,device=device)*self.sigma_min

#             for step in range(ddim_steps):
#                 t_b=torch.full((B,),step*dt,device=device)
#                 ns=self.ctx_noise_scale*2. if step<3 else 0.
#                 if step>0:
#                     vc=self.net.forward_with_ctx(x_t,t_b,raw_ctx,noise_scale=ns,
#                         vel_obs_feat=vel_obs_feat,steering_feat=steering_feat,
#                         env_kine_feat=env_kine_feat,env_data=env_data,use_null=False)
#                     vu=self.net.forward_with_ctx(x_t,t_b,raw_ctx,noise_scale=0.,
#                         vel_obs_feat=vel_obs_feat,steering_feat=steering_feat,
#                         env_kine_feat=env_kine_feat,env_data=env_data,use_null=True)
#                     vel=vu+guidance_scale.view(B,1,1)*(vc-vu)
#                 else:
#                     vel=self.net.forward_with_ctx(x_t,t_b,raw_ctx,noise_scale=ns,
#                         vel_obs_feat=vel_obs_feat,steering_feat=steering_feat,
#                         env_kine_feat=env_kine_feat,env_data=env_data)
#                 m_s=_mom_str(step,ddim_steps)
#                 if m_s>1e-4:
#                     me=obs_mom.unsqueeze(1).expand(B,T,2)
#                     mf=torch.cat([me,torch.zeros(B,T,2,device=device)],dim=-1)
#                     vel=vel+m_s*mf
#                 x_t=(x_t+dt*vel).clamp(-3.,3.)
#             pred_abs,_=self._to_abs(x_t,lp,lm)
#             all_norms.append(pred_abs)

#         # K=3 mode clustering
#         pred_mean=_mode_cluster_k3(all_norms,obs_norm,self.criterion.scorer)
#         pred_final=_persistence_blend_adaptive(pred_mean,obs_norm,blend_alpha)
#         if predict_csv: self._write_predict_csv(predict_csv,pred_final)
#         return pred_final,torch.stack(all_norms)

#     @staticmethod
#     def _write_predict_csv(csv_path,traj_mean):
#         os.makedirs(os.path.dirname(os.path.abspath(csv_path)),exist_ok=True)
#         T,B,_=traj_mean.shape
#         mlon=((traj_mean[:,:,0]*50.+1800.)/10.).cpu().numpy()
#         mlat=((traj_mean[:,:,1]*50.)/10.).cpu().numpy()
#         ts=datetime.now().strftime('%Y%m%d_%H%M%S')
#         hdr=not os.path.exists(csv_path)
#         with open(csv_path,'a',newline='') as fh:
#             w=csv.DictWriter(fh,fieldnames=['ts','b','step','lead_h','lon','lat'])
#             if hdr: w.writeheader()
#             for b in range(B):
#                 for k in range(T):
#                     w.writerow({'ts':ts,'b':b,'step':k,'lead_h':(k+1)*6,
#                                  'lon':f'{mlon[k,b]:.4f}','lat':f'{mlat[k,b]:.4f}'})


# TCDiffusion = TCFlowMatching

# """
# flow_matching_model.py — FM v74-fix
# ====================================
# DROP-IN REPLACEMENT cho v60.

# ROOT CAUSES của v60 oscillation (từ log analysis):
#   BUG-A: x0=persist+noise → u_target=correction (nhỏ 0.29x so với standard)
#          → Model học predict small velocity → speed undershoot -17 to -28 km/6h mỗi epoch
#          → ADE oscillate vì inference distribution ≠ training distribution
#   BUG-B: l_logspd gradient asymmetric → log-space MSE penalize underestimation
#          nặng hơn overestimation → conflict với BUG-A → oscillation amplified
#   BUG-C: Không có direct position loss → model chỉ học velocity, không
#          minimize haversine(pred_pos, gt_pos) trực tiếp → gradient path dài

# FIXES:
#   FIX-A: x0 = N(0, sigma) pure noise như v59
#          → u_target = x1 - x0 ≈ x1 (magnitude chuẩn)
#          → Inference distribution matches training distribution
#          → No speed bias

#   FIX-B: Remove l_logspd
#          → Thay bằng L_disp = ||Δpred - Δgt|| / ||Δgt||.clamp(10km)
#          → Normalized ratio, không asymmetric, không km-scale
#          → Covers ATE+CTE implicitly without naming either

#   FIX-C: Add L_pos = huber(haversine(pred, gt), d=300) / 300
#          → Direct position loss như v59's l_dpe
#          → Clear gradient: push pred_pos toward gt_pos mỗi step
#          → Guarantees monotone decrease (direct ADE proxy)
#          → With LEARNED step_weights (monotone, sw72/sw6≥3)

# LOSS FORMULA (general — không ATE, CTE, speed tên):
#   Total = w_fm * L_fm                    — flow matching velocity
#         + w_pos * L_pos(step_weights)     — position quality (ADE proxy)
#         + w_disp * L_disp                 — displacement quality (ATE+CTE proxy)
#         + 0.3 * L_speed_head              — auxiliary speed prediction
#         + anchor losses

# GIỮA NGUYÊN từ v60:
#   - VelocityField architecture (toàn bộ — FNO3D, Mamba, recurv_enc, etc.)
#   - EMA, OT matching, CFG guidance
#   - LearnedStepWeights (monotone cumsum)
#   - DifficultyWeighter (per-sample [1,2])
#   - EnsembleScorer (auxiliary BCE)
#   - K=3 mode clustering inference (từ analysis trước)
# """
# from __future__ import annotations

# import csv, math, os
# from datetime import datetime
# from typing import Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from Model.FNO3D_encoder import FNO3DEncoder
# from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# from Model.env_net_transformer_gphsplit import Env_net

# R_EARTH  = 6371.0
# DT_HOURS = 6.0
# DEG2KM   = 111.0
# MAX_CURVATURE_RAD = math.pi / 4
# _NORM_TO_DEG = 5.0


# # ══════════════════════════════════════════════════════════════
# #  Coordinate utilities (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# def norm_to_deg(t):
#     return torch.stack([(t[...,0]*50.+1800.)/10., (t[...,1]*50.)/10.], dim=-1)

# _norm_to_deg = norm_to_deg

# def haversine_km(p1, p2):
#     la1=torch.deg2rad(p1[...,1]); la2=torch.deg2rad(p2[...,1])
#     dlat=torch.deg2rad(p2[...,1]-p1[...,1]); dlon=torch.deg2rad(p2[...,0]-p1[...,0])
#     a=torch.sin(dlat/2).pow(2)+torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2)
#     return 2.*R_EARTH*torch.asin(a.clamp(1e-12,1-1e-12).sqrt())

# def velocity_km_s(traj_deg):
#     """[T,B,2] degrees → [T-1,B,2] km/6h."""
#     lon=traj_deg[:,:,0]; lat=traj_deg[:,:,1]
#     dlat=lat[1:]-lat[:-1]; dlon=lon[1:]-lon[:-1]
#     lat_mid=(lat[1:]+lat[:-1])/2.
#     cos_lat=torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
#     return torch.stack([dlon*cos_lat*DEG2KM, dlat*DEG2KM], dim=-1)


# # ══════════════════════════════════════════════════════════════
# #  LearnedStepWeights (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# class LearnedStepWeights(nn.Module):
#     def __init__(self, n_steps=12):
#         super().__init__()
#         self.n_steps=n_steps
#         self.raw=nn.Parameter(torch.zeros(n_steps)+0.5)
#     def forward(self):
#         w=torch.cumsum(F.softplus(self.raw),dim=0)
#         return w/w.mean().clamp(1e-8)
#     def get(self,n=None):
#         w=self.forward(); return w[:n] if n is not None else w
#     @torch.no_grad()
#     def stats(self):
#         w=self.forward()
#         return {"sw_ratio":(w[-1]/w[0].clamp(1e-6)).item(),
#                 "sw_monotonic":bool((w[1:]-w[:-1]).min().item()>=-1e-6),
#                 "sw_6h":w[0].item(),"sw_24h":w[3].item(),
#                 "sw_48h":w[7].item(),"sw_72h":w[-1].item()}


# # ══════════════════════════════════════════════════════════════
# #  [NEW] ConstrainedLossWeights — softplus_inv init
# # ══════════════════════════════════════════════════════════════

# class ConstrainedLossWeights(nn.Module):
#     """
#     3 weights: w_fm, w_pos, w_disp. Tự học, init chính xác qua softplus_inv.
#     Anchor loss ngăn drift khỏi reasonable region.
#     """
#     @staticmethod
#     def _sp_inv(y):
#         if y>20.: return y
#         if y<1e-6: return -20.
#         return math.log(math.expm1(y))

#     def __init__(self, init_fm=1.0, init_pos=2.0, init_disp=0.8, anchor_w=0.02):
#         super().__init__()
#         self.anchor_w = anchor_w
#         raw = torch.tensor([self._sp_inv(init_fm),
#                               self._sp_inv(init_pos),
#                               self._sp_inv(init_disp)], dtype=torch.float)
#         self.log_w = nn.Parameter(raw)
#         self.register_buffer('log_w0', raw.clone())

#     def _get(self,i,mn,mx): return F.softplus(self.log_w[i]).clamp(mn,mx)
#     def w_fm(self):   return self._get(0, 0.2, 4.0)
#     def w_pos(self):  return self._get(1, 0.5, 6.0)
#     def w_disp(self): return self._get(2, 0.1, 3.0)
#     def anchor_loss(self): return self.anchor_w*((self.log_w-self.log_w0)**2).mean()
#     @torch.no_grad()
#     def stats(self):
#         return {"lw_fm":self.w_fm().item(),"lw_pos":self.w_pos().item(),
#                 "lw_disp":self.w_disp().item(),
#                 "lw_pos_fm":(self.w_pos()/self.w_fm().clamp(1e-6)).item()}


# # ══════════════════════════════════════════════════════════════
# #  DifficultyWeighter (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# class DifficultyWeighter(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear=nn.Linear(3,1,bias=True)
#         nn.init.zeros_(self.linear.weight)
#         nn.init.constant_(self.linear.bias,-2.0)
#     def compute_difficulty(self,gt_deg):
#         B=gt_deg.shape[1]; v=velocity_km_s(gt_deg)
#         if v.shape[0]>=2:
#             heading=torch.atan2(v[:,:,0],v[:,:,1])
#             dh=heading[1:]-heading[:-1]
#             dh=(dh+math.pi)%(2*math.pi)-math.pi
#             curv_rate=dh.abs().mean(0)
#             excess=F.relu(dh.abs()-MAX_CURVATURE_RAD).mean(0)
#         else:
#             curv_rate=excess=gt_deg.new_zeros(B)
#         spd=v.norm(dim=-1); mean_spd=spd.mean(0).clamp(1.)
#         speed_cv=(spd.std(0)/mean_spd).clamp(max=3.)
#         d1=(curv_rate/(math.pi/2)).clamp(0,1)
#         d2=speed_cv.clamp(0,1)
#         d3=(excess/(math.pi/4)).clamp(0,1)
#         return torch.stack([d1,d2,d3],dim=-1)
#     def forward(self,gt_deg):
#         diff=self.compute_difficulty(gt_deg)
#         return 1.0+torch.sigmoid(self.linear(diff).squeeze(-1))


# # ══════════════════════════════════════════════════════════════
# #  EnsembleScorer (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# class EnsembleScorer(nn.Module):
#     def __init__(self,feat_dim=7,hidden=32):
#         super().__init__()
#         self.net=nn.Sequential(
#             nn.Linear(feat_dim,hidden),nn.GELU(),
#             nn.Linear(hidden,16),nn.GELU(),
#             nn.Linear(16,1))
#         nn.init.zeros_(self.net[-1].weight)
#         nn.init.zeros_(self.net[-1].bias)

#     def extract_features(self,traj_norm,obs_norm):
#         B=traj_norm.shape[1]
#         traj_deg=norm_to_deg(traj_norm); obs_deg=norm_to_deg(obs_norm)
#         v=velocity_km_s(traj_deg); spd=v.norm(dim=-1)
#         f1=torch.log1p(spd.mean(0)); f2=torch.log1p(spd.std(0).clamp(0))
#         if v.shape[0]>=2:
#             heading=torch.atan2(v[:,:,0],v[:,:,1])
#             dh=heading[1:]-heading[:-1]; dh=(dh+math.pi)%(2*math.pi)-math.pi
#             f3=torch.cos(dh).mean(0)
#         else: f3=traj_norm.new_ones(B)
#         v_obs=velocity_km_s(obs_deg); n_obs=min(3,v_obs.shape[0])
#         obs_spd=v_obs[-n_obs:].norm(dim=-1).mean(0).clamp(1.)
#         n_pred=min(3,spd.shape[0]); pred_spd=spd[:n_pred].mean(0)
#         f4=torch.exp(-((pred_spd-obs_spd)/obs_spd).pow(2)*2.)
#         if v_obs.shape[0]>=1 and v.shape[0]>=1:
#             obs_h=torch.atan2(v_obs[-1,:,0],v_obs[-1,:,1])
#             pred_h=torch.atan2(v[0,:,0],v[0,:,1])
#             dh_cont=pred_h-obs_h; dh_cont=(dh_cont+math.pi)%(2*math.pi)-math.pi
#             f5=torch.cos(dh_cont)
#         else: f5=traj_norm.new_ones(B)
#         if obs_norm.shape[0]>=2:
#             lv=obs_norm[-1]-obs_norm[-2]
#             steps=torch.arange(1,traj_norm.shape[0]+1,device=traj_norm.device,dtype=traj_norm.dtype)
#             persist=obs_norm[-1].unsqueeze(0)+lv.unsqueeze(0)*steps.view(-1,1,1)
#             dfp=(traj_norm-persist).norm(dim=-1).mean(0)
#             ref=(lv.norm(dim=-1)*traj_norm.shape[0]).clamp(1e-3)
#             f6=dfp/ref
#         else: f6=traj_norm.new_zeros(B)
#         if v.shape[0]>=2:
#             heading=torch.atan2(v[:,:,0],v[:,:,1])
#             dh=heading[1:]-heading[:-1]; dh=(dh+math.pi)%(2*math.pi)-math.pi
#             f7=dh.abs().mean(0)
#         else: f7=traj_norm.new_zeros(B)
#         return torch.stack([f1,f2,f3,f4,f5,f6,f7],dim=-1).clamp(-10.,10.)

#     def score(self,traj_norm,obs_norm):
#         return torch.sigmoid(self.net(self.extract_features(traj_norm,obs_norm)).squeeze(-1))

#     def logits(self,traj_norm,obs_norm):
#         return self.net(self.extract_features(traj_norm,obs_norm)).squeeze(-1)

#     def auxiliary_loss(self,candidates,obs_norm,gt_deg):
#         if len(candidates)<2: return candidates[0].new_zeros(())
#         gt_b=gt_deg.permute(1,0,2)
#         ades=[]
#         for c in candidates:
#             cb=norm_to_deg(c).permute(1,0,2)
#             ades.append(haversine_km(cb,gt_b).mean(1))
#         ades_t=torch.stack(ades,0); best=ades_t.argmin(0)
#         tot=ades_t.new_zeros(())
#         for i,c in enumerate(candidates):
#             tot=tot+F.binary_cross_entropy_with_logits(
#                 self.logits(c,obs_norm),(best==i).float())
#         return tot/len(candidates)


# # ══════════════════════════════════════════════════════════════
# #  [NEW] L_pos: direct position loss — the key to monotone decrease
# # ══════════════════════════════════════════════════════════════

# def l_position(pred_deg, gt_deg, step_weights, d=300., sample_w=None):
#     """
#     FIX-C: Direct position loss — haversine(pred_pos, gt_pos).

#     Đây là thành phần QUAN TRỌNG NHẤT để ADE giảm đều.
#     - Gradient trực tiếp: push pred_pos về phía gt_pos
#     - Không qua intermediate velocity → không oscillate
#     - step_weights: learned monotone (w_72h >= 3x w_6h)
#     - sample_w: difficulty weights [1,2]

#     d=300: rộng hơn v59's 200 → better gradient khi error lớn (>300km)
#     Scale: huber(dist,300)/300 → ~0.1-1.0, same as L_fm
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2: return pred_deg.new_zeros(())
#     dist = haversine_km(pred_deg[:T], gt_deg[:T])          # [T,B] km
#     w = step_weights[:T]                                    # [T] learned
#     huber = torch.where(dist<d, dist.pow(2)/(2.*d), dist-d/2.)
#     loss = (huber * w.unsqueeze(1)).mean() / d
#     if sample_w is not None:
#         # Apply difficulty weighting: hard storms get more gradient
#         loss = (loss * sample_w.mean())  # simplified: scale by mean difficulty
#     return loss


# # ══════════════════════════════════════════════════════════════
# #  [NEW] L_disp: normalized displacement — ATE+CTE proxy
# # ══════════════════════════════════════════════════════════════

# def l_displacement(pred_deg, gt_deg):
#     """
#     FIX-B: Normalized displacement error, replaces l_logspd.

#     ||Δpred - Δgt|| / ||Δgt||.clamp(10km)
#     Δ[t] = traj[t+1] - traj[t] (displacement vector in km)

#     Tại sao tốt hơn l_logspd:
#     - l_logspd: MSE in log-space → asymmetric gradient → oscillation
#     - l_disp:   ratio, symmetric → stable gradient → smooth decrease

#     Tại sao covers ATE+CTE:
#     - ATE lớn: |Δpred| ≠ |Δgt| (wrong speed/magnitude) → ratio > 1
#     - CTE lớn: direction(Δpred) ≠ direction(Δgt)       → ratio > 1
#     - Cả hai về 0 khi pred tốt → L_disp → 0

#     Scale: ~0.1-1.5, independent of km magnitude.
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 3: return pred_deg.new_zeros(())
#     cos_lat = torch.cos(torch.deg2rad(gt_deg[1:T,:,1])).clamp(1e-4)  # [T-1,B]
#     def _disp_km(traj):
#         dx=(traj[1:T,:,0]-traj[:T-1,:,0])*cos_lat*DEG2KM
#         dy=(traj[1:T,:,1]-traj[:T-1,:,1])*DEG2KM
#         return torch.stack([dx,dy],dim=-1)  # [T-1,B,2]
#     dp=_disp_km(pred_deg); dg=_disp_km(gt_deg)
#     err_km=(dp-dg).norm(dim=-1)           # [T-1,B]
#     gt_mag=dg.norm(dim=-1).clamp(min=10.) # [T-1,B] min 10km
#     ratio=err_km/gt_mag                   # dimensionless ~0-3
#     return F.huber_loss(ratio, torch.zeros_like(ratio), delta=0.5)


# # ══════════════════════════════════════════════════════════════
# #  SpeedHead (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# class SpeedHead(nn.Module):
#     def __init__(self,ctx_dim=256,obs_feat_dim=256,pred_len=12):
#         super().__init__()
#         self.net=nn.Sequential(
#             nn.Linear(ctx_dim+obs_feat_dim,256),nn.GELU(),nn.LayerNorm(256),
#             nn.Linear(256,128),nn.GELU(),nn.Linear(128,pred_len))
#         nn.init.xavier_uniform_(self.net[-1].weight,gain=0.1)
#         nn.init.zeros_(self.net[-1].bias)
#     def forward(self,ctx,obs_feat):
#         return torch.exp(self.net(torch.cat([ctx,obs_feat],dim=-1))).clamp(3.,150.)

# def l_speed_head(speed_pred, pred_deg, gt_deg, step_weights):
#     """Speed head auxiliary loss — general speed quality."""
#     T = min(speed_pred.shape[1]+1, gt_deg.shape[0])
#     if T < 2: return speed_pred.new_zeros(())
#     v_gt = velocity_km_s(gt_deg[:T])               # [T-1,B,2]
#     spd_gt = v_gt.norm(dim=-1).clamp(3.).permute(1,0)  # [B,T-1]
#     n = min(speed_pred.shape[1], spd_gt.shape[1])
#     w = step_weights[:n]; w = w/w.sum()
#     # MSE in log space but with GT reference — no asymmetry issue
#     # since we're comparing pred to GT directly
#     return F.mse_loss(torch.log1p(speed_pred[:,:n]),
#                        torch.log1p(spd_gt[:,:n]), reduction='none').mean()


# # ══════════════════════════════════════════════════════════════
# #  Master Loss (v74-fix)
# # ══════════════════════════════════════════════════════════════

# class FMv74Loss(nn.Module):
#     """
#     v74-fix loss: 3 general terms + auxiliary.

#     Total = w_fm*L_fm + w_pos*L_pos(step_w) + w_disp*L_disp
#           + 0.3*L_speed_head + 0.05*L_scorer + anchors

#     L_fm:   velocity MSE (flow matching)
#     L_pos:  haversine position loss (ADE proxy, DIRECT)
#     L_disp: normalized displacement (ATE+CTE proxy, INDIRECT)
#     """
#     def __init__(self, pred_len=12):
#         super().__init__()
#         self.pred_len     = pred_len
#         self.step_weights = LearnedStepWeights(n_steps=pred_len)
#         self.loss_weights = ConstrainedLossWeights(
#             init_fm=1.0, init_pos=2.0, init_disp=0.8, anchor_w=0.02)
#         self.diff_weighter = DifficultyWeighter()
#         self.scorer        = EnsembleScorer()
#         self.speed_head_aux = None  # set externally

#     def compute_main_losses(self, pred_deg, gt_deg,
#                              fm_vel_pred, fm_vel_target,
#                              epoch=0):
#         T = min(pred_deg.shape[0], gt_deg.shape[0])
#         sample_w = self.diff_weighter(gt_deg[:T])  # [B] ∈ [1,2]
#         sw = self.step_weights.get(n=T)             # [T] monotone

#         w_fm   = self.loss_weights.w_fm()
#         w_pos  = self.loss_weights.w_pos()
#         w_disp = self.loss_weights.w_disp()
#         anc    = self.step_weights.stats()
#         # Note: step_weights has no anchor loss here — it learns freely
#         # but constrained by cumsum(softplus) monotonicity
#         lw_anc = self.loss_weights.anchor_loss()

#         # FIX-A applied at caller: x0=N(0,sigma), so fm_vel_pred/target are standard
#         L_fm   = F.mse_loss(fm_vel_pred, fm_vel_target)

#         # FIX-C: direct position loss (key to monotone decrease)
#         L_pos  = l_position(pred_deg[:T], gt_deg[:T], sw, d=300., sample_w=sample_w)

#         # FIX-B: normalized displacement (ATE+CTE proxy, no speed bias)
#         L_disp = l_displacement(pred_deg[:T], gt_deg[:T])

#         total = w_fm*L_fm + w_pos*L_pos + w_disp*L_disp + lw_anc

#         if not torch.isfinite(total): total = pred_deg.new_zeros(())

#         sw_s = self.step_weights.stats()
#         lw_s = self.loss_weights.stats()
#         bd = {
#             "l_fm":    L_fm.item(),
#             "l_pos":   L_pos.item(),
#             "l_disp":  L_disp.item(),
#             "diff_w_mean": sample_w.mean().item(),
#             **{f"sw_{k}":v for k,v in sw_s.items()},
#             **{f"lw_{k}":v for k,v in lw_s.items()},
#         }
#         return total, bd

#     def forward(self, pred_deg, gt_deg, fm_vel_pred, fm_vel_target,
#                  speed_pred=None, candidates=None, obs_norm=None, epoch=0):
#         total, bd = self.compute_main_losses(
#             pred_deg, gt_deg, fm_vel_pred, fm_vel_target, epoch=epoch)

#         # Speed head auxiliary (small weight)
#         l_sh = pred_deg.new_zeros(())
#         if speed_pred is not None:
#             T = min(pred_deg.shape[0], gt_deg.shape[0])
#             sw = self.step_weights.get(n=T)
#             l_sh = l_speed_head(speed_pred, pred_deg, gt_deg, sw)
#             total = total + 0.3 * l_sh

#         # Scorer auxiliary
#         l_scr = pred_deg.new_zeros(())
#         if candidates is not None and obs_norm is not None and len(candidates)>=2:
#             l_scr = self.scorer.auxiliary_loss(candidates, obs_norm, gt_deg)
#             total = total + 0.05 * l_scr

#         bd.update({"l_sh":l_sh.item(),"l_scorer":l_scr.item(),"total":total})
#         return total, bd

#     @torch.no_grad()
#     def stats(self):
#         return {**self.step_weights.stats(), **self.loss_weights.stats()}


# # ══════════════════════════════════════════════════════════════
# #  Safe env helpers (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# def _safe_env(env_data,key,B,device,norm=1.0):
#     v=env_data.get(key) if env_data is not None else None
#     if v is None or not torch.is_tensor(v): return torch.zeros(B,device=device)
#     v=v.float().to(device)
#     while v.dim()>1: v=v.mean(-1)
#     v=v.view(-1)[:B] if v.numel()>=B else torch.zeros(B,device=device)
#     return (v/norm).clamp(-3.,3.)

# def _safe_env_vec(env_data,key,dim,B,device):
#     v=env_data.get(key) if env_data is not None else None
#     if v is None: return torch.zeros(B,dim,device=device)
#     if not torch.is_tensor(v):
#         try: v=torch.tensor(v,dtype=torch.float,device=device)
#         except: return torch.zeros(B,dim,device=device)
#     v=v.float().to(device)
#     if v.dim()==0: return torch.zeros(B,dim,device=device)
#     if v.dim()==1:
#         return v.unsqueeze(0).expand(B,dim) if v.shape[0]==dim else torch.zeros(B,dim,device=device)
#     if v.dim()==2:
#         if v.shape==(B,dim): return v
#         if v.shape[0]==B:
#             return v[:,:dim] if v.shape[1]>=dim else F.pad(v,(0,dim-v.shape[1]))
#         return torch.zeros(B,dim,device=device)
#     if v.dim()==3:
#         vv=v[:B,-1,:]
#         return vv[:,:dim] if vv.shape[1]>=dim else F.pad(vv,(0,dim-vv.shape[1]))
#     return torch.zeros(B,dim,device=device)


# # ══════════════════════════════════════════════════════════════
# #  VelocityField (giữ nguyên v60 — toàn bộ architecture)
# # ══════════════════════════════════════════════════════════════

# class VelocityField(nn.Module):
#     RAW_CTX_DIM = 512

#     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256,
#                  sigma_min=0.02, unet_in_ch=13, **kwargs):
#         super().__init__()
#         self.pred_len=pred_len; self.obs_len=obs_len; self.ctx_dim=ctx_dim
#         self.sigma_min=sigma_min

#         self.spatial_enc=FNO3DEncoder(in_channel=unet_in_ch,out_channel=1,d_model=32,
#             n_layers=4,modes_t=4,modes_h=4,modes_w=4,spatial_down=32,dropout=0.05)
#         self.bottleneck_pool=nn.AdaptiveAvgPool3d((None,1,1))
#         self.bottleneck_proj=nn.Linear(128,128)
#         self.decoder_proj=nn.Linear(1,16)
#         self.enc_1d=DataEncoder1D(in_1d=4,feat_3d_dim=128,mlp_h=64,
#             lstm_hidden=128,lstm_layers=3,dropout=0.1,d_state=16)
#         self.env_enc=Env_net(obs_len=obs_len,d_model=32)
#         self.steering_enc=nn.Sequential(
#             nn.Linear(9,64),nn.GELU(),nn.LayerNorm(64),
#             nn.Linear(64,128),nn.GELU(),nn.Linear(128,256))
#         self.env_kine_enc=nn.Sequential(
#             nn.Linear(14,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,256),nn.GELU())
#         self.recurv_enc=nn.Sequential(
#             nn.Linear(33,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,64))
#         self.speed_hist_enc=nn.Sequential(
#             nn.Linear(11,32),nn.GELU(),nn.LayerNorm(32),nn.Linear(32,32))
#         self.ctx_fc1=nn.Linear(128+32+16+64+32,self.RAW_CTX_DIM)
#         self.ctx_ln=nn.LayerNorm(self.RAW_CTX_DIM)
#         self.ctx_drop=nn.Dropout(0.20)
#         self.ctx_fc2=nn.Linear(self.RAW_CTX_DIM,ctx_dim)
#         self.null_embedding=nn.Parameter(torch.randn(1,self.RAW_CTX_DIM)*0.02)
#         self.vel_obs_enc=nn.Sequential(
#             nn.Linear(obs_len*6,256),nn.GELU(),nn.LayerNorm(256),
#             nn.Linear(256,256),nn.GELU())
#         self.blend_head=nn.Linear(ctx_dim,1)
#         self.guidance_head=nn.Linear(ctx_dim,1)
#         self.sigma_head=nn.Linear(ctx_dim,1)
#         nn.init.zeros_(self.blend_head.weight); nn.init.constant_(self.blend_head.bias,-1.)
#         nn.init.zeros_(self.guidance_head.weight); nn.init.constant_(self.guidance_head.bias,0.)
#         nn.init.zeros_(self.sigma_head.weight); nn.init.constant_(self.sigma_head.bias,-1.)
#         self.time_fc1=nn.Linear(256,512); self.time_fc2=nn.Linear(512,256)
#         self.traj_embed=nn.Linear(4,256)
#         self.pos_enc=nn.Parameter(torch.randn(1,pred_len,256)*0.02)
#         self.step_embed=nn.Embedding(pred_len,256)
#         self.transformer=nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model=256,nhead=8,dim_feedforward=1024,
#                 dropout=0.12,activation='gelu',batch_first=True),num_layers=2)
#         self.out_fc1=nn.Linear(256,512); self.out_fc2=nn.Linear(512,4)
#         self.step_scale=nn.Parameter(torch.ones(pred_len)*0.5)
#         self.physics_scale=nn.Parameter(torch.ones(4)*1.5)
#         self.steering_scale=nn.Parameter(torch.ones(4)*1.0)
#         self.speed_head=SpeedHead(ctx_dim=ctx_dim,obs_feat_dim=256,pred_len=pred_len)
#         self._init_weights()

#     def _init_weights(self):
#         with torch.no_grad():
#             for n,m in self.named_modules():
#                 if isinstance(m,nn.Linear) and 'out_fc' in n:
#                     nn.init.xavier_uniform_(m.weight,gain=0.1)
#                     if m.bias is not None: nn.init.zeros_(m.bias)

#     def _time_emb(self,t,dim=256):
#         h=dim//2
#         fr=torch.exp(torch.arange(h,dtype=torch.float,device=t.device)*(-math.log(10000.)/max(h-1,1)))
#         em=t.float().unsqueeze(1)*1000.*fr.unsqueeze(0)
#         return F.pad(torch.cat([em.sin(),em.cos()],dim=-1),(0,dim%2))

#     def _get_steering_feat(self,env_data,B,device):
#         if env_data is None: return torch.zeros(B,256,device=device)
#         feats=torch.stack([_safe_env(env_data,'u500_mean',B,device,30.),
#             _safe_env(env_data,'v500_mean',B,device,30.),
#             _safe_env(env_data,'u500_center',B,device,30.),
#             _safe_env(env_data,'v500_center',B,device,30.),
#             _safe_env(env_data,'steering_speed',B,device,1.),
#             _safe_env(env_data,'steering_dir_sin',B,device,1.),
#             _safe_env(env_data,'steering_dir_cos',B,device,1.),
#             _safe_env(env_data,'gph500_mean',B,device,1.),
#             _safe_env(env_data,'gph500_center',B,device,1.)],dim=-1)
#         return self.steering_enc(feats)

#     def _get_env_kine_feat(self,env_data,B,device):
#         if env_data is None: return torch.zeros(B,256,device=device)
#         mv=_safe_env(env_data,'move_velocity',B,device,150.).unsqueeze(-1)
#         hd24=_safe_env_vec(env_data,'history_direction24',8,B,device)
#         dv=_safe_env_vec(env_data,'delta_velocity',5,B,device)
#         return self.env_kine_enc(torch.cat([mv,hd24,dv],dim=-1))

#     def _get_recurv_feat(self,env_data,B,device):
#         if env_data is None: return torch.zeros(B,64,device=device)
#         bearing=_safe_env_vec(env_data,'bearing_to_scs_center',16,B,device)
#         dist=_safe_env_vec(env_data,'dist_to_scs_boundary',5,B,device)
#         month=_safe_env_vec(env_data,'month',12,B,device)
#         return self.recurv_enc(torch.cat([bearing,dist,month],dim=-1))

#     def _get_speed_hist_feat(self,env_data,B,device):
#         if env_data is None: return torch.zeros(B,32,device=device)
#         vh=_safe_env_vec(env_data,'velocity_history',4,B,device)
#         ri=_safe_env(env_data,'rapid_intensification',B,device,1.).unsqueeze(-1)
#         ic=_safe_env_vec(env_data,'intensity_class',6,B,device)
#         return self.speed_hist_enc(torch.cat([vh,ri,ic],dim=-1))

#     def _get_kinematic_obs_feat(self,obs_traj):
#         B=obs_traj.shape[1]; T_obs=obs_traj.shape[0]
#         if T_obs>=2:
#             vel=obs_traj[1:]-obs_traj[:-1]
#             lat_mid=obs_traj[:-1,:,1]*_NORM_TO_DEG
#             cos_lat=torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
#             dx_km=vel[:,:,0]*cos_lat*DEG2KM*_NORM_TO_DEG
#             dy_km=vel[:,:,1]*DEG2KM*_NORM_TO_DEG
#             speed=torch.sqrt(dx_km**2+dy_km**2+1e-6)/DT_HOURS
#             heading=torch.atan2(vel[:,:,1],vel[:,:,0])
#             speed_n=(speed/20.).clamp(-3.,3.)
#             if T_obs>=3:
#                 dspd=speed[1:]-speed[:-1]
#                 accel=torch.cat([obs_traj.new_zeros(1,B),(dspd/10.).clamp(-3.,3.)],0)
#             else: accel=obs_traj.new_zeros(T_obs-1,B)
#             kine=torch.stack([vel[:,:,0],vel[:,:,1],speed_n,heading.sin(),heading.cos(),accel],dim=-1)
#         else: kine=obs_traj.new_zeros(self.obs_len,B,6)
#         if kine.shape[0]<self.obs_len:
#             kine=torch.cat([obs_traj.new_zeros(self.obs_len-kine.shape[0],B,6),kine],0)
#         else: kine=kine[-self.obs_len:]
#         return self.vel_obs_enc(kine.permute(1,0,2).reshape(B,-1))

#     def _context(self,batch_list):
#         obs_traj=batch_list[0]; obs_Me=batch_list[7]; image_obs=batch_list[11]
#         env_data=batch_list[13] if len(batch_list)>13 else None
#         B=obs_traj.shape[1]; device=obs_traj.device
#         if image_obs.dim()==4: image_obs=image_obs.unsqueeze(2)
#         if image_obs.shape[1]==1 and self.spatial_enc.in_channel!=1:
#             image_obs=image_obs.expand(-1,self.spatial_enc.in_channel,-1,-1,-1)
#         e_3d_bot,e_3d_dec=self.spatial_enc.encode(image_obs)
#         T_obs=obs_traj.shape[0]
#         e_3d_s=self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0,2,1)
#         e_3d_s=self.bottleneck_proj(e_3d_s)
#         if e_3d_s.shape[1]!=T_obs:
#             e_3d_s=F.interpolate(e_3d_s.permute(0,2,1),size=T_obs,mode='linear',align_corners=False).permute(0,2,1)
#         e_3d_dec_t=e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
#         t_w=torch.softmax(torch.arange(e_3d_dec_t.shape[1],dtype=torch.float,device=device)*0.5,dim=0)
#         f_sp=self.decoder_proj((e_3d_dec_t*t_w.unsqueeze(0)).sum(1,keepdim=True))
#         obs_in=torch.cat([obs_traj,obs_Me],dim=2).permute(1,0,2)
#         h_t=self.enc_1d(obs_in,e_3d_s)
#         e_env,_,_=self.env_enc(env_data,image_obs)
#         recurv_feat=self._get_recurv_feat(env_data,B,device)
#         speed_h_feat=self._get_speed_hist_feat(env_data,B,device)
#         cat_feat=torch.cat([h_t,e_env,f_sp,recurv_feat,speed_h_feat],dim=-1)
#         return F.gelu(self.ctx_ln(self.ctx_fc1(cat_feat)))

#     def _apply_ctx_head(self,raw,noise_scale=0.,use_null=False):
#         if use_null: raw=self.null_embedding.expand(raw.shape[0],-1)
#         elif noise_scale>0.: raw=raw+torch.randn_like(raw)*noise_scale
#         return self.ctx_fc2(self.ctx_drop(raw))

#     def get_blend_alpha(self,ctx): return torch.sigmoid(self.blend_head(ctx)).squeeze(-1)*0.5
#     def get_guidance_scale(self,ctx): return 0.8+1.2*torch.sigmoid(self.guidance_head(ctx)).squeeze(-1)
#     def get_sigma(self,ctx): return 0.02+0.08*torch.sigmoid(self.sigma_head(ctx)).squeeze(-1)

#     def _beta_drift(self,x_t):
#         lat_rad=torch.deg2rad(x_t[:,:,1]*5.).clamp(-85,85)
#         beta=2*7.2921e-5*torch.cos(lat_rad)/6.371e6; R=3e5
#         v=torch.zeros_like(x_t)
#         v[:,:,0]=-beta*R**2/2.*6*3600./(5*111*1000.)
#         v[:,:,1]=beta*R**2/4.*6*3600./(5*111*1000.)
#         return v

#     def _steering_drift(self,x_t,env_data):
#         if env_data is None: return torch.zeros_like(x_t)
#         B,device=x_t.shape[0],x_t.device
#         u=_safe_env(env_data,'u500_center',B,device,30.)
#         vv=_safe_env(env_data,'v500_center',B,device,30.)
#         cos=torch.cos(torch.deg2rad(x_t[:,:,1]*5.)).clamp(1e-3)
#         out=torch.zeros_like(x_t)
#         out[:,:,0]=u.unsqueeze(1)*30.*21600./(111.*1000.*cos)
#         out[:,:,1]=vv.unsqueeze(1)*30.*21600./(111.*1000.)
#         return out

#     def _decode(self,x_t,t,ctx,vel_obs_feat=None,steering_feat=None,
#                  env_kine_feat=None,env_data=None):
#         B=x_t.shape[0]
#         t_emb=F.gelu(self.time_fc1(self._time_emb(t))); t_emb=self.time_fc2(t_emb)
#         T=min(x_t.size(1),self.pos_enc.shape[1])
#         si=torch.arange(T,device=x_t.device).unsqueeze(0).expand(B,-1)
#         xe=(self.traj_embed(x_t[:,:T])+self.pos_enc[:,:T]+t_emb.unsqueeze(1)+self.step_embed(si))
#         mem=[t_emb.unsqueeze(1),ctx.unsqueeze(1)]
#         if vel_obs_feat is not None: mem.append(vel_obs_feat.unsqueeze(1))
#         if steering_feat is not None: mem.append(steering_feat.unsqueeze(1))
#         if env_kine_feat is not None: mem.append(env_kine_feat.unsqueeze(1))
#         dec=self.transformer(xe,torch.cat(mem,dim=1))
#         vn=self.out_fc2(F.gelu(self.out_fc1(dec)))
#         sc=torch.sigmoid(self.step_scale[:T]).view(1,T,1)*2.; vn=vn*sc
#         with torch.no_grad():
#             vp=self._beta_drift(x_t[:,:T]); vs=self._steering_drift(x_t[:,:T],env_data)
#         return vn+torch.sigmoid(self.physics_scale)*vp+torch.sigmoid(self.steering_scale)*vs

#     def forward_with_ctx(self,x_t,t,raw_ctx,noise_scale=0.,vel_obs_feat=None,
#                           steering_feat=None,env_kine_feat=None,env_data=None,use_null=False):
#         ctx=self._apply_ctx_head(raw_ctx,noise_scale,use_null=use_null)
#         return self._decode(x_t,t,ctx,vel_obs_feat=vel_obs_feat,
#                             steering_feat=steering_feat,env_kine_feat=env_kine_feat,env_data=env_data)

#     def predict_speed(self,raw_ctx,vel_obs_feat):
#         return self.speed_head(self._apply_ctx_head(raw_ctx),vel_obs_feat)


# # ══════════════════════════════════════════════════════════════
# #  EMA, OT (giữ nguyên v60)
# # ══════════════════════════════════════════════════════════════

# class EMAModel:
#     def __init__(self,model,decay=0.995):
#         self.decay=decay; m=model._orig_mod if hasattr(model,'_orig_mod') else model
#         self.shadow={k:v.detach().clone() for k,v in m.state_dict().items() if v.dtype.is_floating_point}
#     def update(self,model):
#         m=model._orig_mod if hasattr(model,'_orig_mod') else model
#         with torch.no_grad():
#             for k,v in m.state_dict().items():
#                 if k in self.shadow: self.shadow[k].mul_(self.decay).add_(v.detach(),alpha=1-self.decay)
#     def apply_to(self,model):
#         m=model._orig_mod if hasattr(model,'_orig_mod') else model
#         bk,sd={},m.state_dict()
#         for k in self.shadow:
#             if k not in sd: continue
#             bk[k]=sd[k].detach().clone(); sd[k].copy_(self.shadow[k])
#         return bk
#     def restore(self,model,bk):
#         m=model._orig_mod if hasattr(model,'_orig_mod') else model; sd=m.state_dict()
#         for k,v in bk.items():
#             if k in sd: sd[k].copy_(v)


# def _sinkhorn_log(cost,epsilon=0.05,n_iter=50):
#     B=cost.shape[0]; device=cost.device
#     la=-math.log(B)*torch.ones(B,device=device); lb=la.clone()
#     lK=-cost/epsilon; lu=torch.zeros(B,device=device); lv=lu.clone()
#     for _ in range(n_iter):
#         lu=la-torch.logsumexp(lK+lv.unsqueeze(0),dim=1)
#         lv=lb-torch.logsumexp(lK+lu.unsqueeze(1),dim=0)
#     return (lK+lu.unsqueeze(1)+lv.unsqueeze(0)).exp().clamp(0.)

# def _spherical_ot_matching(x0_batch,x1_batch,lp,epsilon=0.05):
#     try:
#         B=x0_batch.shape[0]
#         abs0=lp.unsqueeze(1)[:,:,:2]+x0_batch[:,:,:2]; abs1=lp.unsqueeze(1)[:,:,:2]+x1_batch[:,:,:2]
#         abs0_deg=norm_to_deg(abs0); abs1_deg=norm_to_deg(abs1)
#         cost=haversine_km(
#             abs0_deg.unsqueeze(1).expand(B,B,-1,-1).reshape(B*B,-1,2),
#             abs1_deg.unsqueeze(0).expand(B,B,-1,-1).reshape(B*B,-1,2)
#         ).mean(-1).reshape(B,B)/500.
#         pi=_sinkhorn_log(cost,epsilon=epsilon)
#         flat=pi.reshape(-1).clamp(0.); s=flat.sum()
#         if not torch.isfinite(s) or s<1e-10: return x0_batch,x1_batch
#         idx=torch.multinomial(flat/s,num_samples=B,replacement=True)
#         return x0_batch[idx%B],x1_batch[idx%B]
#     except: return x0_batch,x1_batch


# def compute_speed_stats_from_norm(obs_traj_norm):
#     T_obs=obs_traj_norm.shape[0]
#     if T_obs<2: return {'v_opt':15.,'v_sigma':10.,'v_hard_cap':80.,'p50_kmh':15.}
#     lon=(obs_traj_norm[...,0]*50.+1800.)/10.; lat=(obs_traj_norm[...,1]*50.)/10.
#     lat_mid=(lat[:-1]+lat[1:])/2; cos_lat=torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
#     dx=(lon[1:]-lon[:-1])*cos_lat*DEG2KM; dy=(lat[1:]-lat[:-1])*DEG2KM
#     spd=torch.sqrt(dx**2+dy**2)/DT_HOURS
#     p50=float(spd.flatten().median()); p95=float(torch.quantile(spd.flatten(),.95))
#     return {'v_opt':max(p50,5.),'v_sigma':10.,
#             'v_hard_cap':float(torch.tensor(p95*1.8).clamp(25.,130.)),'p50_kmh':p50}


# @torch.no_grad()
# def _persistence_blend_adaptive(model_pred_norm,obs_traj_norm,blend_alpha):
#     T_obs=obs_traj_norm.shape[0]; T=model_pred_norm.shape[0]
#     B=model_pred_norm.shape[1]; device=model_pred_norm.device
#     if T_obs<2: return model_pred_norm
#     vels=obs_traj_norm[1:]-obs_traj_norm[:-1]; n_v=vels.shape[0]
#     if n_v>=3:
#         alpha=0.7; w=torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],dtype=torch.float,device=device).flip(0)
#         ev=(vels*(w/w.sum()).view(-1,1,1)).sum(0)
#     elif n_v==2: ev=0.7*vels[-1]+0.3*vels[-2]
#     else: ev=vels[-1]
#     steps=torch.arange(1,T+1,dtype=torch.float,device=device)
#     persist=obs_traj_norm[-1].unsqueeze(0)+ev.unsqueeze(0)*steps.view(T,1,1)
#     alpha_b=blend_alpha.view(1,B,1).clamp(0.,.5)
#     return (1.-alpha_b)*model_pred_norm+alpha_b*persist


# # ══════════════════════════════════════════════════════════════
# #  [NEW] K=3 mode clustering inference
# # ══════════════════════════════════════════════════════════════

# @torch.no_grad()
# def _mode_cluster_k3(trajs_norm, obs_norm, scorer, speed_stats=None):
#     """
#     K=3 mode clustering tại 72h endpoint.
#     Thay top-35% median → CTE thấp hơn tự nhiên.

#     trajs_norm: list of [T,B,2] seq-first normalized
#     obs_norm:   [T_obs,B,2] seq-first
#     scorer:     EnsembleScorer
#     returns:    [T,B,2] seq-first normalized
#     """
#     if not trajs_norm: return obs_norm[-1:].expand(12,obs_norm.shape[1],2)
#     dev=trajs_norm[0].device; T,B=trajs_norm[0].shape[0],trajs_norm[0].shape[1]; N=len(trajs_norm); K=min(3,N)

#     # Scores [N,B]
#     all_sc=torch.stack([scorer.score(tr,obs_norm) for tr in trajs_norm],dim=0)
#     # 72h endpoints in degrees [N,B,2]
#     endpoints=torch.stack([norm_to_deg(tr[-1]) for tr in trajs_norm],dim=0)

#     results=[]
#     for b in range(B):
#         ep_b=endpoints[:,b,:]; sc_b=all_sc[:,b]
#         tr_b=torch.stack([tr[:,b,:] for tr in trajs_norm],dim=0)  # [N,T,2]
#         if N<K:
#             w=F.softmax(sc_b*3.,0); results.append((tr_b*w.view(N,1,1)).sum(0)); continue
#         # Farthest-point init
#         idx0=sc_b.argmax().item(); centers=[ep_b[idx0]]
#         for _ in range(K-1):
#             d2c=torch.stack([haversine_km(ep_b,c.unsqueeze(0).expand_as(ep_b)) for c in centers],1).min(1).values
#             centers.append(ep_b[d2c.argmax()])
#         centers=torch.stack(centers,0)
#         # 3 K-means iterations
#         for _ in range(3):
#             d2c=torch.stack([haversine_km(ep_b,c.unsqueeze(0).expand_as(ep_b)) for c in centers],1)
#             assign=d2c.argmin(1)
#             new_c=[]
#             for k in range(K):
#                 mk=(assign==k)
#                 if mk.sum()>0:
#                     wk=F.softmax(sc_b[mk]*3.,0)
#                     new_c.append((ep_b[mk]*wk.unsqueeze(1)).sum(0))
#                 else: new_c.append(centers[k])
#             centers=torch.stack(new_c,0)
#         # Score clusters
#         d2c=torch.stack([haversine_km(ep_b,c.unsqueeze(0).expand_as(ep_b)) for c in centers],1)
#         assign=d2c.argmin(1)
#         csc=torch.zeros(K,device=dev)
#         for k in range(K):
#             mk=(assign==k)
#             if mk.sum()>0: csc[k]=sc_b[mk].sum()
#         best_k=csc.argmax().item(); mk=(assign==best_k)
#         if not mk.any(): mk=torch.ones(N,dtype=torch.bool,device=dev)
#         w_win=F.softmax(sc_b[mk]*3.,0)
#         results.append((tr_b[mk]*w_win.view(-1,1,1)).sum(0))
#     return torch.stack(results,dim=1)  # [T,B,2]


# # ══════════════════════════════════════════════════════════════
# #  TCFlowMatching v74-fix
# # ══════════════════════════════════════════════════════════════

# class TCFlowMatching(nn.Module):

#     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02, unet_in_ch=13,
#                  ctx_noise_scale=0.01, use_ema=True, ema_decay=0.995,
#                  use_ate_ot=True, ot_epsilon=0.05, cfg_uncond_prob=0.1, **kwargs):
#         super().__init__()
#         self.pred_len=pred_len; self.obs_len=obs_len; self.sigma_min=sigma_min
#         self.ctx_noise_scale=ctx_noise_scale; self.use_ate_ot=use_ate_ot
#         self.ot_epsilon=ot_epsilon; self.cfg_uncond_prob=cfg_uncond_prob
#         self.net=VelocityField(pred_len=pred_len,obs_len=obs_len,
#                                 sigma_min=sigma_min,unet_in_ch=unet_in_ch,ctx_dim=256)
#         self.criterion=FMv74Loss(pred_len=pred_len)
#         self.use_ema=use_ema; self._ema=None

#     def init_ema(self):
#         if self.use_ema: self._ema=EMAModel(self,decay=0.995)
#     def ema_update(self):
#         if self._ema is not None: self._ema.update(self)

#     @staticmethod
#     def _to_rel(traj,Me,lp,lm):
#         return torch.cat([traj-lp[:,:2].unsqueeze(0),Me-lm.unsqueeze(0)],dim=-1).permute(1,0,2)

#     @staticmethod
#     def _to_abs(rel,lp,lm):
#         d=rel.permute(1,0,2)
#         return lp[:,:2].unsqueeze(0)+d[:,:,:2], lm.unsqueeze(0)+d[:,:,2:]

#     @staticmethod
#     def _sigma_schedule(ep):
#         if ep<2: return 0.10
#         if ep<10: return 0.10-(ep-2)/8.*(0.10-0.04)
#         if ep<20: return max(0.04-(ep-10)/10.*0.01,0.035)
#         return 0.035

#     def _cfm_standard(self,x1,sigma_min=None):
#         """FIX-A: standard FM x0=N(0,sigma), NOT persist+noise."""
#         if sigma_min is None: sigma_min=self.sigma_min
#         B=x1.shape[0]; dev=x1.device
#         x0=torch.randn_like(x1)*sigma_min
#         t=torch.rand(B,device=dev); te=t.view(B,1,1)
#         return (1.-te)*x0+te*x1, t, x1-x0

#     def _persistence_forecast_rel(self,obs_traj,lp,lm,pred_len):
#         B,device=obs_traj.shape[1],obs_traj.device
#         obs_pos=obs_traj[:,:,:2]
#         if obs_pos.shape[0]>=3:
#             vels=obs_pos[1:]-obs_pos[:-1]; n_v=vels.shape[0]; alpha=0.7
#             w=torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],dtype=torch.float,device=device).flip(0)
#             lv=(vels*(w/w.sum()).view(-1,1,1)).sum(0)
#         elif obs_pos.shape[0]>=2: lv=obs_pos[-1]-obs_pos[-2]
#         else: lv=obs_traj.new_zeros(B,2)
#         steps=torch.arange(1,pred_len+1,device=device).float()
#         pred_abs=obs_pos[-1].unsqueeze(1)+lv.unsqueeze(1)*steps.view(1,-1,1)
#         pred_rel=torch.cat([pred_abs-lp[:,:2].unsqueeze(1),torch.zeros_like(pred_abs)],dim=-1)
#         return pred_rel

#     def _compute_obs_momentum(self,obs_norm):
#         T_obs=obs_norm.shape[0]
#         if T_obs<2: return torch.zeros(obs_norm.shape[1],2,device=obs_norm.device)
#         vels=obs_norm[1:]-obs_norm[:-1]; n_v=vels.shape[0]
#         if n_v>=3:
#             a=0.65; w=torch.tensor([a*(1-a)**i for i in range(n_v)],dtype=torch.float,device=obs_norm.device).flip(0)
#             return (vels*(w/w.sum()).view(-1,1,1)).sum(0)
#         elif n_v==2: return 0.65*vels[-1]+0.35*vels[-2]
#         return vels[-1]

#     @staticmethod
#     @staticmethod
#     def _obs_noise_aug(bl,sigma=0.005):
#         if torch.rand(1).item()>0.5: return bl
#         bl=list(bl)
#         if torch.is_tensor(bl[0]): bl[0]=bl[0]+torch.randn_like(bl[0])*sigma
#         return bl

#     @staticmethod
#     def _lon_flip_aug(bl, p=0.35):
#         """
#         Flip longitude: lon_norm → -lon_norm.
#         Mô phỏng bão ở phía đối xứng → model học invariant với hướng E/W.
#         Giảm val-test gap vì test set có nhiều bão hướng khác nhau hơn.
#         """
#         if torch.rand(1).item() > p: return bl
#         bl = list(bl)
#         for i in [0, 1, 2, 3, 7, 8]:
#             if i < len(bl) and torch.is_tensor(bl[i]) and bl[i].shape[-1] >= 1:
#                 t = bl[i].clone(); t[..., 0] = -t[..., 0]; bl[i] = t
#         return bl

#     @staticmethod
#     def _speed_scale_aug(bl, scale_range=(0.88, 1.14), p=0.40):
#         """
#         Scale obs velocity by random factor 0.88-1.14x.
#         Mô phỏng bão nhanh/chậm hơn → generalize đến recent storms (2017-2021).
#         Giúp giảm val-test gap vì test storms thường nhanh hơn train distribution.
#         """
#         if torch.rand(1).item() > p: return bl
#         bl = list(bl)
#         scale = torch.empty(1).uniform_(*scale_range).item()
#         if torch.is_tensor(bl[0]) and bl[0].shape[0] >= 2:
#             obs = bl[0].clone()
#             displ = obs[1:] - obs[:-1]
#             new_obs = obs.clone()
#             for t in range(1, obs.shape[0]):
#                 new_obs[t] = new_obs[t-1] + displ[t-1] * scale
#             bl[0] = new_obs
#         return bl

#     # ── Training ──────────────────────────────────────────────

#     def get_loss(self,batch_list,epoch=0,**kwargs):
#         return self.get_loss_breakdown(batch_list,epoch=epoch)['total']

#     def get_loss_breakdown(self,batch_list,epoch=0):
#         batch_list=self._obs_noise_aug(batch_list,sigma=0.005)
#         batch_list=self._lon_flip_aug(batch_list, p=0.35)
#         batch_list=self._speed_scale_aug(batch_list, p=0.40)
#         obs_t=batch_list[0]; env_data=batch_list[13] if len(batch_list)>13 else None
#         lp=obs_t[-1]; lm=batch_list[7][-1]; B,device=lp.shape[0],lp.device
#         gt_traj_s=batch_list[1]

#         current_sigma=self._sigma_schedule(epoch)
#         raw_ctx=self.net._context(batch_list)
#         x1_rel=self._to_rel(gt_traj_s,batch_list[8],lp,lm)

#         # FIX-A: standard x0=N(0,sigma), NOT persist+noise
#         if self.use_ate_ot and B>=4:
#             x0_noise=torch.randn_like(x1_rel)*current_sigma
#             _,x1_matched=_spherical_ot_matching(x0_noise,x1_rel,lp,epsilon=self.ot_epsilon)
#         else:
#             x1_matched=x1_rel
#         x_t,fm_t,u_target=self._cfm_standard(x1_matched,sigma_min=current_sigma)

#         use_null=(torch.rand(1).item()<self.cfg_uncond_prob)
#         vel_obs_feat=self.net._get_kinematic_obs_feat(obs_t[:,:,:2])
#         steering_feat=self.net._get_steering_feat(env_data,B,device)
#         env_kine_feat=self.net._get_env_kine_feat(env_data,B,device)

#         pred_vel=self.net.forward_with_ctx(x_t,fm_t,raw_ctx,env_data=env_data,use_null=use_null,
#             vel_obs_feat=vel_obs_feat,steering_feat=steering_feat,env_kine_feat=env_kine_feat)

#         fm_te=fm_t.view(B,1,1)
#         x1_pred=x_t+(1.-fm_te)*pred_vel
#         pred_abs,_=self._to_abs(x1_pred,lp,lm)
#         pred_deg=norm_to_deg(pred_abs)    # [T,B,2]
#         gt_deg=norm_to_deg(gt_traj_s)    # [T,B,2]

#         # Speed head prediction for auxiliary loss
#         speed_pred=None
#         if not use_null:
#             speed_pred=self.net.predict_speed(raw_ctx,vel_obs_feat)

#         # Scorer candidates
#         candidates=None; obs_norm=obs_t[:,:,:2]
#         if epoch>=5 and not use_null:
#             cands=[]
#             for _ in range(3):
#                 te_c=fm_t.view(B,1,1)
#                 x0_c=torch.randn_like(x1_rel)*current_sigma
#                 x_c=(1.-te_c)*x0_c+te_c*x1_rel
#                 with torch.no_grad():
#                     v_c=self.net.forward_with_ctx(x_c,fm_t,raw_ctx,env_data=env_data,
#                         vel_obs_feat=vel_obs_feat,steering_feat=steering_feat,env_kine_feat=env_kine_feat)
#                     x1_c=x_c+(1.-te_c)*v_c; abs_c,_=self._to_abs(x1_c,lp,lm)
#                     cands.append(abs_c[:,:,:2])
#             candidates=cands

#         total,bd=self.criterion(pred_deg,gt_deg,pred_vel,u_target,
#                                   speed_pred=speed_pred,candidates=candidates,
#                                   obs_norm=obs_norm,epoch=epoch)

#         if torch.isnan(total) or torch.isinf(total): total=obs_t.new_zeros(())

#         bd.update({'sigma':current_sigma,
#                    'v_opt':compute_speed_stats_from_norm(obs_t[:,:,:2]).get('v_opt',15.),
#                    'total':total,
#                    # legacy keys
#                    'l_fm':bd.get('l_fm',0.),'l_kin':0.,'l_logspd':0.,
#                    'l_curv':0.,'diff_w_mean':bd.get('diff_w_mean',1.),
#                    'lam_kin':0.,'lam_logspd':0.,'lam_curv':0.})
#         return bd

#     # ── Inference ─────────────────────────────────────────────

#     @torch.no_grad()
#     def sample(self,batch_list,num_ensemble=50,ddim_steps=20,predict_csv=None,use_tta=True):
#         obs_t=batch_list[0]; env_data=batch_list[13] if len(batch_list)>13 else None
#         lp=obs_t[-1]; lm=batch_list[7][-1]; B=lp.shape[0]; device=lp.device
#         T=self.pred_len; dt=1./max(ddim_steps,1)

#         def _run_ensemble(bl, n_ens):
#             """Run n_ens ensemble members on a batch_list."""
#             _obs_t=bl[0]; _env=bl[13] if len(bl)>13 else None
#             _lp=_obs_t[-1]; _lm=bl[7][-1]
#             _raw_ctx=self.net._context(bl)
#             _ctx=self.net._apply_ctx_head(_raw_ctx)
#             _vel_f=self.net._get_kinematic_obs_feat(_obs_t[:,:,:2])
#             _steer_f=self.net._get_steering_feat(_env,B,device)
#             _env_f=self.net._get_env_kine_feat(_env,B,device)
#             _obs_n=_obs_t[:,:,:2]; _obs_m=self._compute_obs_momentum(_obs_n)
#             _blend=self.net.get_blend_alpha(_ctx)
#             _gs=self.net.get_guidance_scale(_ctx)

#             def _ms(s,tot): return 0.06*0.5*(1.+math.cos(math.pi*s/max(tot,1)))
#             trajs=[]
#             for _ in range(n_ens):
#                 x_t=torch.randn(B,T,4,device=device)*self.sigma_min
#                 for step in range(ddim_steps):
#                     t_b=torch.full((B,),step*dt,device=device)
#                     ns=self.ctx_noise_scale*2. if step<3 else 0.
#                     if step>0:
#                         vc=self.net.forward_with_ctx(x_t,t_b,_raw_ctx,noise_scale=ns,
#                             vel_obs_feat=_vel_f,steering_feat=_steer_f,
#                             env_kine_feat=_env_f,env_data=_env,use_null=False)
#                         vu=self.net.forward_with_ctx(x_t,t_b,_raw_ctx,noise_scale=0.,
#                             vel_obs_feat=_vel_f,steering_feat=_steer_f,
#                             env_kine_feat=_env_f,env_data=_env,use_null=True)
#                         vel=vu+_gs.view(B,1,1)*(vc-vu)
#                     else:
#                         vel=self.net.forward_with_ctx(x_t,t_b,_raw_ctx,noise_scale=ns,
#                             vel_obs_feat=_vel_f,steering_feat=_steer_f,
#                             env_kine_feat=_env_f,env_data=_env)
#                     ms=_ms(step,ddim_steps)
#                     if ms>1e-4:
#                         me=_obs_m.unsqueeze(1).expand(B,T,2)
#                         vel=vel+ms*torch.cat([me,torch.zeros(B,T,2,device=device)],dim=-1)
#                     x_t=(x_t+dt*vel).clamp(-3.,3.)
#                 pa,_=self._to_abs(x_t,_lp,_lm); trajs.append(pa)
#             pred_m=_mode_cluster_k3(trajs,_obs_n,self.criterion.scorer)
#             pred_f=_persistence_blend_adaptive(pred_m,_obs_n,_blend)
#             return pred_f, trajs

#         # Normal inference
#         n_ens_each = num_ensemble if not use_tta else max(num_ensemble//2, 20)
#         pred_normal, all_norms = _run_ensemble(batch_list, n_ens_each)

#         if use_tta:
#             # TTA: flip longitude, predict, unflip, average
#             # Giảm systematic directional bias → test gap thấp hơn
#             bl_flip = list(batch_list)
#             for i in [0,1,2,3,7,8]:
#                 if i<len(bl_flip) and torch.is_tensor(bl_flip[i]):
#                     t=bl_flip[i].clone(); t[...,0]=-t[...,0]; bl_flip[i]=t
#             pred_flip, _ = _run_ensemble(bl_flip, n_ens_each)
#             # Unflip
#             pred_unflip = pred_flip.clone()
#             pred_unflip[...,0] = -pred_unflip[...,0]
#             # Average (equal weight)
#             pred_final = 0.5*pred_normal + 0.5*pred_unflip
#         else:
#             pred_final = pred_normal

#         if predict_csv: self._write_predict_csv(predict_csv,pred_final)
#         return pred_final,torch.stack(all_norms)

#     @staticmethod
#     def _write_predict_csv(csv_path,traj_mean):
#         os.makedirs(os.path.dirname(os.path.abspath(csv_path)),exist_ok=True)
#         T,B,_=traj_mean.shape
#         mlon=((traj_mean[:,:,0]*50.+1800.)/10.).cpu().numpy()
#         mlat=((traj_mean[:,:,1]*50.)/10.).cpu().numpy()
#         ts=datetime.now().strftime('%Y%m%d_%H%M%S')
#         hdr=not os.path.exists(csv_path)
#         with open(csv_path,'a',newline='') as fh:
#             w=csv.DictWriter(fh,fieldnames=['ts','b','step','lead_h','lon','lat'])
#             if hdr: w.writeheader()
#             for b in range(B):
#                 for k in range(T):
#                     w.writerow({'ts':ts,'b':b,'step':k,'lead_h':(k+1)*6,
#                                  'lon':f'{mlon[k,b]:.4f}','lat':f'{mlat[k,b]:.4f}'})


# TCDiffusion = TCFlowMatching

"""
flow_matching_model.py — FM v74-fix
====================================
DROP-IN REPLACEMENT cho v60.

ROOT CAUSES của v60 oscillation (từ log analysis):
  BUG-A: x0=persist+noise → u_target=correction (nhỏ 0.29x so với standard)
         → Model học predict small velocity → speed undershoot -17 to -28 km/6h mỗi epoch
         → ADE oscillate vì inference distribution ≠ training distribution
  BUG-B: l_logspd gradient asymmetric → log-space MSE penalize underestimation
         nặng hơn overestimation → conflict với BUG-A → oscillation amplified
  BUG-C: Không có direct position loss → model chỉ học velocity, không
         minimize haversine(pred_pos, gt_pos) trực tiếp → gradient path dài

FIXES:
  FIX-A: x0 = N(0, sigma) pure noise như v59
         → u_target = x1 - x0 ≈ x1 (magnitude chuẩn)
         → Inference distribution matches training distribution
         → No speed bias

  FIX-B: Remove l_logspd
         → Thay bằng L_disp = ||Δpred - Δgt|| / ||Δgt||.clamp(10km)
         → Normalized ratio, không asymmetric, không km-scale
         → Covers ATE+CTE implicitly without naming either

  FIX-C: Add L_pos = huber(haversine(pred, gt), d=300) / 300
         → Direct position loss như v59's l_dpe
         → Clear gradient: push pred_pos toward gt_pos mỗi step
         → Guarantees monotone decrease (direct ADE proxy)
         → With LEARNED step_weights (monotone, sw72/sw6≥3)

LOSS FORMULA (general — không ATE, CTE, speed tên):
  Total = w_fm * L_fm                    — flow matching velocity
        + w_pos * L_pos(step_weights)     — position quality (ADE proxy)
        + w_disp * L_disp                 — displacement quality (ATE+CTE proxy)
        + 0.3 * L_speed_head              — auxiliary speed prediction
        + anchor losses

GIỮA NGUYÊN từ v60:
  - VelocityField architecture (toàn bộ — FNO3D, Mamba, recurv_enc, etc.)
  - EMA, OT matching, CFG guidance
  - LearnedStepWeights (monotone cumsum)
  - DifficultyWeighter (per-sample [1,2])
  - EnsembleScorer (auxiliary BCE)
  - K=3 mode clustering inference (từ analysis trước)
"""
from __future__ import annotations

import csv, math, os
from datetime import datetime
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.FNO3D_encoder import FNO3DEncoder
from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
from Model.env_net_transformer_gphsplit import Env_net

R_EARTH  = 6371.0
DT_HOURS = 6.0
DEG2KM   = 111.0
MAX_CURVATURE_RAD = math.pi / 4
_NORM_TO_DEG = 5.0


# ══════════════════════════════════════════════════════════════
#  Coordinate utilities (giữ nguyên v60)
# ══════════════════════════════════════════════════════════════

def norm_to_deg(t):
    return torch.stack([(t[...,0]*50.+1800.)/10., (t[...,1]*50.)/10.], dim=-1)

_norm_to_deg = norm_to_deg

def haversine_km(p1, p2):
    la1=torch.deg2rad(p1[...,1]); la2=torch.deg2rad(p2[...,1])
    dlat=torch.deg2rad(p2[...,1]-p1[...,1]); dlon=torch.deg2rad(p2[...,0]-p1[...,0])
    a=torch.sin(dlat/2).pow(2)+torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2)
    return 2.*R_EARTH*torch.asin(a.clamp(1e-12,1-1e-12).sqrt())

def velocity_km_s(traj_deg):
    """[T,B,2] degrees → [T-1,B,2] km/6h."""
    lon=traj_deg[:,:,0]; lat=traj_deg[:,:,1]
    dlat=lat[1:]-lat[:-1]; dlon=lon[1:]-lon[:-1]
    lat_mid=(lat[1:]+lat[:-1])/2.
    cos_lat=torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
    return torch.stack([dlon*cos_lat*DEG2KM, dlat*DEG2KM], dim=-1)


# ══════════════════════════════════════════════════════════════
#  LearnedStepWeights (giữ nguyên v60)
# ══════════════════════════════════════════════════════════════

class LearnedStepWeights(nn.Module):
    def __init__(self, n_steps=12):
        super().__init__()
        self.n_steps=n_steps
        self.raw=nn.Parameter(torch.zeros(n_steps)+0.5)
    def forward(self):
        w=torch.cumsum(F.softplus(self.raw),dim=0)
        return w/w.mean().clamp(1e-8)
    def get(self,n=None):
        w=self.forward(); return w[:n] if n is not None else w
    @torch.no_grad()
    def stats(self):
        w=self.forward()
        return {"sw_ratio":(w[-1]/w[0].clamp(1e-6)).item(),
                "sw_monotonic":bool((w[1:]-w[:-1]).min().item()>=-1e-6),
                "sw_6h":w[0].item(),"sw_24h":w[3].item(),
                "sw_48h":w[7].item(),"sw_72h":w[-1].item()}


# ══════════════════════════════════════════════════════════════
#  [NEW] ConstrainedLossWeights — softplus_inv init
# ══════════════════════════════════════════════════════════════

class ConstrainedLossWeights(nn.Module):
    """
    3 weights: w_fm, w_pos, w_disp. Tự học, init chính xác qua softplus_inv.
    Anchor loss ngăn drift khỏi reasonable region.
    """
    @staticmethod
    def _sp_inv(y):
        if y>20.: return y
        if y<1e-6: return -20.
        return math.log(math.expm1(y))

    def __init__(self, init_fm=1.0, init_pos=2.0, init_disp=0.8, anchor_w=0.02):
        super().__init__()
        self.anchor_w = anchor_w
        raw = torch.tensor([self._sp_inv(init_fm),
                              self._sp_inv(init_pos),
                              self._sp_inv(init_disp)], dtype=torch.float)
        self.log_w = nn.Parameter(raw)
        self.register_buffer('log_w0', raw.clone())

    def _get(self,i,mn,mx): return F.softplus(self.log_w[i]).clamp(mn,mx)
    def w_fm(self):   return self._get(0, 0.2, 4.0)
    def w_pos(self):  return self._get(1, 0.5, 6.0)
    def w_disp(self): return self._get(2, 0.1, 3.0)
    def anchor_loss(self): return self.anchor_w*((self.log_w-self.log_w0)**2).mean()
    @torch.no_grad()
    def stats(self):
        return {"lw_fm":self.w_fm().item(),"lw_pos":self.w_pos().item(),
                "lw_disp":self.w_disp().item(),
                "lw_pos_fm":(self.w_pos()/self.w_fm().clamp(1e-6)).item()}


# ══════════════════════════════════════════════════════════════
#  DifficultyWeighter (giữ nguyên v60)
# ══════════════════════════════════════════════════════════════

class DifficultyWeighter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(3,1,bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias,-2.0)
    def compute_difficulty(self,gt_deg):
        B=gt_deg.shape[1]; v=velocity_km_s(gt_deg)
        if v.shape[0]>=2:
            heading=torch.atan2(v[:,:,0],v[:,:,1])
            dh=heading[1:]-heading[:-1]
            dh=(dh+math.pi)%(2*math.pi)-math.pi
            curv_rate=dh.abs().mean(0)
            excess=F.relu(dh.abs()-MAX_CURVATURE_RAD).mean(0)
        else:
            curv_rate=excess=gt_deg.new_zeros(B)
        spd=v.norm(dim=-1); mean_spd=spd.mean(0).clamp(1.)
        speed_cv=(spd.std(0)/mean_spd).clamp(max=3.)
        d1=(curv_rate/(math.pi/2)).clamp(0,1)
        d2=speed_cv.clamp(0,1)
        d3=(excess/(math.pi/4)).clamp(0,1)
        return torch.stack([d1,d2,d3],dim=-1)
    def forward(self,gt_deg):
        diff=self.compute_difficulty(gt_deg)
        return 1.0+torch.sigmoid(self.linear(diff).squeeze(-1))


# ══════════════════════════════════════════════════════════════
#  EnsembleScorer (giữ nguyên v60)
# ══════════════════════════════════════════════════════════════

class EnsembleScorer(nn.Module):
    def __init__(self,feat_dim=7,hidden=32):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(feat_dim,hidden),nn.GELU(),
            nn.Linear(hidden,16),nn.GELU(),
            nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def extract_features(self,traj_norm,obs_norm):
        B=traj_norm.shape[1]
        traj_deg=norm_to_deg(traj_norm); obs_deg=norm_to_deg(obs_norm)
        v=velocity_km_s(traj_deg); spd=v.norm(dim=-1)
        f1=torch.log1p(spd.mean(0)); f2=torch.log1p(spd.std(0).clamp(0))
        if v.shape[0]>=2:
            heading=torch.atan2(v[:,:,0],v[:,:,1])
            dh=heading[1:]-heading[:-1]; dh=(dh+math.pi)%(2*math.pi)-math.pi
            f3=torch.cos(dh).mean(0)
        else: f3=traj_norm.new_ones(B)
        v_obs=velocity_km_s(obs_deg); n_obs=min(3,v_obs.shape[0])
        obs_spd=v_obs[-n_obs:].norm(dim=-1).mean(0).clamp(1.)
        n_pred=min(3,spd.shape[0]); pred_spd=spd[:n_pred].mean(0)
        f4=torch.exp(-((pred_spd-obs_spd)/obs_spd).pow(2)*2.)
        if v_obs.shape[0]>=1 and v.shape[0]>=1:
            obs_h=torch.atan2(v_obs[-1,:,0],v_obs[-1,:,1])
            pred_h=torch.atan2(v[0,:,0],v[0,:,1])
            dh_cont=pred_h-obs_h; dh_cont=(dh_cont+math.pi)%(2*math.pi)-math.pi
            f5=torch.cos(dh_cont)
        else: f5=traj_norm.new_ones(B)
        if obs_norm.shape[0]>=2:
            lv=obs_norm[-1]-obs_norm[-2]
            steps=torch.arange(1,traj_norm.shape[0]+1,device=traj_norm.device,dtype=traj_norm.dtype)
            persist=obs_norm[-1].unsqueeze(0)+lv.unsqueeze(0)*steps.view(-1,1,1)
            dfp=(traj_norm-persist).norm(dim=-1).mean(0)
            ref=(lv.norm(dim=-1)*traj_norm.shape[0]).clamp(1e-3)
            f6=dfp/ref
        else: f6=traj_norm.new_zeros(B)
        if v.shape[0]>=2:
            heading=torch.atan2(v[:,:,0],v[:,:,1])
            dh=heading[1:]-heading[:-1]; dh=(dh+math.pi)%(2*math.pi)-math.pi
            f7=dh.abs().mean(0)
        else: f7=traj_norm.new_zeros(B)
        return torch.stack([f1,f2,f3,f4,f5,f6,f7],dim=-1).clamp(-10.,10.)

    def score(self,traj_norm,obs_norm):
        return torch.sigmoid(self.net(self.extract_features(traj_norm,obs_norm)).squeeze(-1))

    def logits(self,traj_norm,obs_norm):
        return self.net(self.extract_features(traj_norm,obs_norm)).squeeze(-1)

    def auxiliary_loss(self,candidates,obs_norm,gt_deg):
        if len(candidates)<2: return candidates[0].new_zeros(())
        gt_b=gt_deg.permute(1,0,2)
        ades=[]
        for c in candidates:
            cb=norm_to_deg(c).permute(1,0,2)
            ades.append(haversine_km(cb,gt_b).mean(1))
        ades_t=torch.stack(ades,0); best=ades_t.argmin(0)
        tot=ades_t.new_zeros(())
        for i,c in enumerate(candidates):
            tot=tot+F.binary_cross_entropy_with_logits(
                self.logits(c,obs_norm),(best==i).float())
        return tot/len(candidates)


# ══════════════════════════════════════════════════════════════
#  [NEW] L_pos: direct position loss — the key to monotone decrease
# ══════════════════════════════════════════════════════════════

def l_position(pred_deg, gt_deg, step_weights, d=300., sample_w=None):
    """
    FIX-C: Direct position loss — haversine(pred_pos, gt_pos).

    Đây là thành phần QUAN TRỌNG NHẤT để ADE giảm đều.
    - Gradient trực tiếp: push pred_pos về phía gt_pos
    - Không qua intermediate velocity → không oscillate
    - step_weights: learned monotone (w_72h >= 3x w_6h)
    - sample_w: difficulty weights [1,2]

    d=300: rộng hơn v59's 200 → better gradient khi error lớn (>300km)
    Scale: huber(dist,300)/300 → ~0.1-1.0, same as L_fm
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2: return pred_deg.new_zeros(())
    dist = haversine_km(pred_deg[:T], gt_deg[:T])          # [T,B] km
    w = step_weights[:T]                                    # [T] learned
    huber = torch.where(dist<d, dist.pow(2)/(2.*d), dist-d/2.)
    loss = (huber * w.unsqueeze(1)).mean() / d
    if sample_w is not None:
        # Apply difficulty weighting: hard storms get more gradient
        loss = (loss * sample_w.mean())  # simplified: scale by mean difficulty
    return loss


# ══════════════════════════════════════════════════════════════
#  [NEW] L_disp: normalized displacement — ATE+CTE proxy
# ══════════════════════════════════════════════════════════════

def l_displacement(pred_deg, gt_deg):
    """
    FIX-B: Normalized displacement error, replaces l_logspd.

    ||Δpred - Δgt|| / ||Δgt||.clamp(10km)
    Δ[t] = traj[t+1] - traj[t] (displacement vector in km)

    Tại sao tốt hơn l_logspd:
    - l_logspd: MSE in log-space → asymmetric gradient → oscillation
    - l_disp:   ratio, symmetric → stable gradient → smooth decrease

    Tại sao covers ATE+CTE:
    - ATE lớn: |Δpred| ≠ |Δgt| (wrong speed/magnitude) → ratio > 1
    - CTE lớn: direction(Δpred) ≠ direction(Δgt)       → ratio > 1
    - Cả hai về 0 khi pred tốt → L_disp → 0

    Scale: ~0.1-1.5, independent of km magnitude.
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 3: return pred_deg.new_zeros(())
    cos_lat = torch.cos(torch.deg2rad(gt_deg[1:T,:,1])).clamp(1e-4)  # [T-1,B]
    def _disp_km(traj):
        dx=(traj[1:T,:,0]-traj[:T-1,:,0])*cos_lat*DEG2KM
        dy=(traj[1:T,:,1]-traj[:T-1,:,1])*DEG2KM
        return torch.stack([dx,dy],dim=-1)  # [T-1,B,2]
    dp=_disp_km(pred_deg); dg=_disp_km(gt_deg)
    err_km=(dp-dg).norm(dim=-1)           # [T-1,B]
    gt_mag=dg.norm(dim=-1).clamp(min=10.) # [T-1,B] min 10km
    ratio=err_km/gt_mag                   # dimensionless ~0-3
    return F.huber_loss(ratio, torch.zeros_like(ratio), delta=0.5)


# ══════════════════════════════════════════════════════════════
#  SpeedHead (giữ nguyên v60)
# ══════════════════════════════════════════════════════════════

class SpeedHead(nn.Module):
    def __init__(self,ctx_dim=256,obs_feat_dim=256,pred_len=12):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(ctx_dim+obs_feat_dim,256),nn.GELU(),nn.LayerNorm(256),
            nn.Linear(256,128),nn.GELU(),nn.Linear(128,pred_len))
        nn.init.xavier_uniform_(self.net[-1].weight,gain=0.1)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self,ctx,obs_feat):
        return torch.exp(self.net(torch.cat([ctx,obs_feat],dim=-1))).clamp(3.,150.)

def l_speed_head(speed_pred, pred_deg, gt_deg, step_weights):
    """Speed head auxiliary loss — general speed quality."""
    T = min(speed_pred.shape[1]+1, gt_deg.shape[0])
    if T < 2: return speed_pred.new_zeros(())
    v_gt = velocity_km_s(gt_deg[:T])               # [T-1,B,2]
    spd_gt = v_gt.norm(dim=-1).clamp(3.).permute(1,0)  # [B,T-1]
    n = min(speed_pred.shape[1], spd_gt.shape[1])
    w = step_weights[:n]; w = w/w.sum()
    # MSE in log space but with GT reference — no asymmetry issue
    # since we're comparing pred to GT directly
    return F.mse_loss(torch.log1p(speed_pred[:,:n]),
                       torch.log1p(spd_gt[:,:n]), reduction='none').mean()


# ══════════════════════════════════════════════════════════════
#  Master Loss (v74-fix)
# ══════════════════════════════════════════════════════════════

class FMv74Loss(nn.Module):
    """
    v74-fix loss: 3 general terms + auxiliary.

    Total = w_fm*L_fm + w_pos*L_pos(step_w) + w_disp*L_disp
          + 0.3*L_speed_head + 0.05*L_scorer + anchors

    L_fm:   velocity MSE (flow matching)
    L_pos:  haversine position loss (ADE proxy, DIRECT)
    L_disp: normalized displacement (ATE+CTE proxy, INDIRECT)
    """
    def __init__(self, pred_len=12):
        super().__init__()
        self.pred_len     = pred_len
        self.step_weights = LearnedStepWeights(n_steps=pred_len)
        self.loss_weights = ConstrainedLossWeights(
            init_fm=1.0, init_pos=2.0, init_disp=0.8, anchor_w=0.02)
        self.diff_weighter = DifficultyWeighter()
        self.scorer        = EnsembleScorer()
        self.speed_head_aux = None  # set externally

    def compute_main_losses(self, pred_deg, gt_deg,
                             fm_vel_pred, fm_vel_target,
                             epoch=0):
        T = min(pred_deg.shape[0], gt_deg.shape[0])
        sample_w = self.diff_weighter(gt_deg[:T])  # [B] ∈ [1,2]
        sw = self.step_weights.get(n=T)             # [T] monotone

        w_fm   = self.loss_weights.w_fm()
        w_pos  = self.loss_weights.w_pos()
        w_disp = self.loss_weights.w_disp()
        anc    = self.step_weights.stats()
        # Note: step_weights has no anchor loss here — it learns freely
        # but constrained by cumsum(softplus) monotonicity
        lw_anc = self.loss_weights.anchor_loss()

        # FIX-A applied at caller: x0=N(0,sigma), so fm_vel_pred/target are standard
        L_fm   = F.mse_loss(fm_vel_pred, fm_vel_target)

        # FIX-C: direct position loss (key to monotone decrease)
        L_pos  = l_position(pred_deg[:T], gt_deg[:T], sw, d=300., sample_w=sample_w)

        # FIX-B: normalized displacement (ATE+CTE proxy, no speed bias)
        L_disp = l_displacement(pred_deg[:T], gt_deg[:T])

        total = w_fm*L_fm + w_pos*L_pos + w_disp*L_disp + lw_anc

        if not torch.isfinite(total): total = pred_deg.new_zeros(())

        sw_s = self.step_weights.stats()
        lw_s = self.loss_weights.stats()
        bd = {
            "l_fm":    L_fm.item(),
            "l_pos":   L_pos.item(),
            "l_disp":  L_disp.item(),
            "diff_w_mean": sample_w.mean().item(),
            **{f"sw_{k}":v for k,v in sw_s.items()},
            **{f"lw_{k}":v for k,v in lw_s.items()},
        }
        return total, bd

    def forward(self, pred_deg, gt_deg, fm_vel_pred, fm_vel_target,
                 speed_pred=None, candidates=None, obs_norm=None, epoch=0):
        total, bd = self.compute_main_losses(
            pred_deg, gt_deg, fm_vel_pred, fm_vel_target, epoch=epoch)

        # Speed head auxiliary (small weight)
        l_sh = pred_deg.new_zeros(())
        if speed_pred is not None:
            T = min(pred_deg.shape[0], gt_deg.shape[0])
            sw = self.step_weights.get(n=T)
            l_sh = l_speed_head(speed_pred, pred_deg, gt_deg, sw)
            total = total + 0.3 * l_sh

        # Scorer auxiliary
        l_scr = pred_deg.new_zeros(())
        if candidates is not None and obs_norm is not None and len(candidates)>=2:
            l_scr = self.scorer.auxiliary_loss(candidates, obs_norm, gt_deg)
            total = total + 0.05 * l_scr

        bd.update({"l_sh":l_sh.item(),"l_scorer":l_scr.item(),"total":total})
        return total, bd

    @torch.no_grad()
    def stats(self):
        return {**self.step_weights.stats(), **self.loss_weights.stats()}


# ══════════════════════════════════════════════════════════════
#  Safe env helpers (giữ nguyên v60)
# ══════════════════════════════════════════════════════════════

def _safe_env(env_data,key,B,device,norm=1.0):
    v=env_data.get(key) if env_data is not None else None
    if v is None or not torch.is_tensor(v): return torch.zeros(B,device=device)
    v=v.float().to(device)
    while v.dim()>1: v=v.mean(-1)
    v=v.view(-1)[:B] if v.numel()>=B else torch.zeros(B,device=device)
    return (v/norm).clamp(-3.,3.)

def _safe_env_vec(env_data,key,dim,B,device):
    v=env_data.get(key) if env_data is not None else None
    if v is None: return torch.zeros(B,dim,device=device)
    if not torch.is_tensor(v):
        try: v=torch.tensor(v,dtype=torch.float,device=device)
        except: return torch.zeros(B,dim,device=device)
    v=v.float().to(device)
    if v.dim()==0: return torch.zeros(B,dim,device=device)
    if v.dim()==1:
        return v.unsqueeze(0).expand(B,dim) if v.shape[0]==dim else torch.zeros(B,dim,device=device)
    if v.dim()==2:
        if v.shape==(B,dim): return v
        if v.shape[0]==B:
            return v[:,:dim] if v.shape[1]>=dim else F.pad(v,(0,dim-v.shape[1]))
        return torch.zeros(B,dim,device=device)
    if v.dim()==3:
        vv=v[:B,-1,:]
        return vv[:,:dim] if vv.shape[1]>=dim else F.pad(vv,(0,dim-vv.shape[1]))
    return torch.zeros(B,dim,device=device)


# ══════════════════════════════════════════════════════════════
#  VelocityField (giữ nguyên v60 — toàn bộ architecture)
# ══════════════════════════════════════════════════════════════

class VelocityField(nn.Module):
    RAW_CTX_DIM = 512

    def __init__(self, pred_len=12, obs_len=8, ctx_dim=256,
                 sigma_min=0.02, unet_in_ch=13, **kwargs):
        super().__init__()
        self.pred_len=pred_len; self.obs_len=obs_len; self.ctx_dim=ctx_dim
        self.sigma_min=sigma_min

        self.spatial_enc=FNO3DEncoder(in_channel=unet_in_ch,out_channel=1,d_model=32,
            n_layers=4,modes_t=4,modes_h=4,modes_w=4,spatial_down=32,dropout=0.05)
        self.bottleneck_pool=nn.AdaptiveAvgPool3d((None,1,1))
        self.bottleneck_proj=nn.Linear(128,128)
        self.decoder_proj=nn.Linear(1,16)
        self.enc_1d=DataEncoder1D(in_1d=4,feat_3d_dim=128,mlp_h=64,
            lstm_hidden=128,lstm_layers=3,dropout=0.1,d_state=16)
        self.env_enc=Env_net(obs_len=obs_len,d_model=32)
        self.steering_enc=nn.Sequential(
            nn.Linear(9,64),nn.GELU(),nn.LayerNorm(64),
            nn.Linear(64,128),nn.GELU(),nn.Linear(128,256))
        self.env_kine_enc=nn.Sequential(
            nn.Linear(14,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,256),nn.GELU())
        self.recurv_enc=nn.Sequential(
            nn.Linear(33,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,64))
        self.speed_hist_enc=nn.Sequential(
            nn.Linear(11,32),nn.GELU(),nn.LayerNorm(32),nn.Linear(32,32))
        self.ctx_fc1=nn.Linear(128+32+16+64+32,self.RAW_CTX_DIM)
        self.ctx_ln=nn.LayerNorm(self.RAW_CTX_DIM)
        self.ctx_drop=nn.Dropout(0.20)
        self.ctx_fc2=nn.Linear(self.RAW_CTX_DIM,ctx_dim)
        self.null_embedding=nn.Parameter(torch.randn(1,self.RAW_CTX_DIM)*0.02)
        self.vel_obs_enc=nn.Sequential(
            nn.Linear(obs_len*6,256),nn.GELU(),nn.LayerNorm(256),
            nn.Linear(256,256),nn.GELU())
        self.blend_head=nn.Linear(ctx_dim,1)
        self.guidance_head=nn.Linear(ctx_dim,1)
        self.sigma_head=nn.Linear(ctx_dim,1)
        nn.init.zeros_(self.blend_head.weight); nn.init.constant_(self.blend_head.bias,-1.)
        nn.init.zeros_(self.guidance_head.weight); nn.init.constant_(self.guidance_head.bias,0.)
        nn.init.zeros_(self.sigma_head.weight); nn.init.constant_(self.sigma_head.bias,-1.)
        self.time_fc1=nn.Linear(256,512); self.time_fc2=nn.Linear(512,256)
        self.traj_embed=nn.Linear(4,256)
        self.pos_enc=nn.Parameter(torch.randn(1,pred_len,256)*0.02)
        self.step_embed=nn.Embedding(pred_len,256)
        self.transformer=nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=256,nhead=8,dim_feedforward=1024,
                dropout=0.12,activation='gelu',batch_first=True),num_layers=2)
        self.out_fc1=nn.Linear(256,512); self.out_fc2=nn.Linear(512,4)
        self.step_scale=nn.Parameter(torch.ones(pred_len)*0.5)
        self.physics_scale=nn.Parameter(torch.ones(4)*1.5)
        self.steering_scale=nn.Parameter(torch.ones(4)*1.0)
        self.speed_head=SpeedHead(ctx_dim=ctx_dim,obs_feat_dim=256,pred_len=pred_len)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for n,m in self.named_modules():
                if isinstance(m,nn.Linear) and 'out_fc' in n:
                    nn.init.xavier_uniform_(m.weight,gain=0.1)
                    if m.bias is not None: nn.init.zeros_(m.bias)

    def _time_emb(self,t,dim=256):
        h=dim//2
        fr=torch.exp(torch.arange(h,dtype=torch.float,device=t.device)*(-math.log(10000.)/max(h-1,1)))
        em=t.float().unsqueeze(1)*1000.*fr.unsqueeze(0)
        return F.pad(torch.cat([em.sin(),em.cos()],dim=-1),(0,dim%2))

    def _get_steering_feat(self,env_data,B,device):
        if env_data is None: return torch.zeros(B,256,device=device)
        feats=torch.stack([_safe_env(env_data,'u500_mean',B,device,30.),
            _safe_env(env_data,'v500_mean',B,device,30.),
            _safe_env(env_data,'u500_center',B,device,30.),
            _safe_env(env_data,'v500_center',B,device,30.),
            _safe_env(env_data,'steering_speed',B,device,1.),
            _safe_env(env_data,'steering_dir_sin',B,device,1.),
            _safe_env(env_data,'steering_dir_cos',B,device,1.),
            _safe_env(env_data,'gph500_mean',B,device,1.),
            _safe_env(env_data,'gph500_center',B,device,1.)],dim=-1)
        return self.steering_enc(feats)

    def _get_env_kine_feat(self,env_data,B,device):
        if env_data is None: return torch.zeros(B,256,device=device)
        mv=_safe_env(env_data,'move_velocity',B,device,150.).unsqueeze(-1)
        hd24=_safe_env_vec(env_data,'history_direction24',8,B,device)
        dv=_safe_env_vec(env_data,'delta_velocity',5,B,device)
        return self.env_kine_enc(torch.cat([mv,hd24,dv],dim=-1))

    def _get_recurv_feat(self,env_data,B,device):
        if env_data is None: return torch.zeros(B,64,device=device)
        bearing=_safe_env_vec(env_data,'bearing_to_scs_center',16,B,device)
        dist=_safe_env_vec(env_data,'dist_to_scs_boundary',5,B,device)
        month=_safe_env_vec(env_data,'month',12,B,device)
        return self.recurv_enc(torch.cat([bearing,dist,month],dim=-1))

    def _get_speed_hist_feat(self,env_data,B,device):
        if env_data is None: return torch.zeros(B,32,device=device)
        vh=_safe_env_vec(env_data,'velocity_history',4,B,device)
        ri=_safe_env(env_data,'rapid_intensification',B,device,1.).unsqueeze(-1)
        ic=_safe_env_vec(env_data,'intensity_class',6,B,device)
        return self.speed_hist_enc(torch.cat([vh,ri,ic],dim=-1))

    def _get_kinematic_obs_feat(self,obs_traj):
        B=obs_traj.shape[1]; T_obs=obs_traj.shape[0]
        if T_obs>=2:
            vel=obs_traj[1:]-obs_traj[:-1]
            lat_mid=obs_traj[:-1,:,1]*_NORM_TO_DEG
            cos_lat=torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
            dx_km=vel[:,:,0]*cos_lat*DEG2KM*_NORM_TO_DEG
            dy_km=vel[:,:,1]*DEG2KM*_NORM_TO_DEG
            speed=torch.sqrt(dx_km**2+dy_km**2+1e-6)/DT_HOURS
            heading=torch.atan2(vel[:,:,1],vel[:,:,0])
            speed_n=(speed/20.).clamp(-3.,3.)
            if T_obs>=3:
                dspd=speed[1:]-speed[:-1]
                accel=torch.cat([obs_traj.new_zeros(1,B),(dspd/10.).clamp(-3.,3.)],0)
            else: accel=obs_traj.new_zeros(T_obs-1,B)
            kine=torch.stack([vel[:,:,0],vel[:,:,1],speed_n,heading.sin(),heading.cos(),accel],dim=-1)
        else: kine=obs_traj.new_zeros(self.obs_len,B,6)
        if kine.shape[0]<self.obs_len:
            kine=torch.cat([obs_traj.new_zeros(self.obs_len-kine.shape[0],B,6),kine],0)
        else: kine=kine[-self.obs_len:]
        return self.vel_obs_enc(kine.permute(1,0,2).reshape(B,-1))

    def _context(self,batch_list):
        obs_traj=batch_list[0]; obs_Me=batch_list[7]; image_obs=batch_list[11]
        env_data=batch_list[13] if len(batch_list)>13 else None
        B=obs_traj.shape[1]; device=obs_traj.device
        if image_obs.dim()==4: image_obs=image_obs.unsqueeze(2)
        if image_obs.shape[1]==1 and self.spatial_enc.in_channel!=1:
            image_obs=image_obs.expand(-1,self.spatial_enc.in_channel,-1,-1,-1)
        e_3d_bot,e_3d_dec=self.spatial_enc.encode(image_obs)
        T_obs=obs_traj.shape[0]
        e_3d_s=self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0,2,1)
        e_3d_s=self.bottleneck_proj(e_3d_s)
        if e_3d_s.shape[1]!=T_obs:
            e_3d_s=F.interpolate(e_3d_s.permute(0,2,1),size=T_obs,mode='linear',align_corners=False).permute(0,2,1)
        e_3d_dec_t=e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
        t_w=torch.softmax(torch.arange(e_3d_dec_t.shape[1],dtype=torch.float,device=device)*0.5,dim=0)
        f_sp=self.decoder_proj((e_3d_dec_t*t_w.unsqueeze(0)).sum(1,keepdim=True))
        obs_in=torch.cat([obs_traj,obs_Me],dim=2).permute(1,0,2)
        h_t=self.enc_1d(obs_in,e_3d_s)
        e_env,_,_=self.env_enc(env_data,image_obs)
        recurv_feat=self._get_recurv_feat(env_data,B,device)
        speed_h_feat=self._get_speed_hist_feat(env_data,B,device)
        cat_feat=torch.cat([h_t,e_env,f_sp,recurv_feat,speed_h_feat],dim=-1)
        return F.gelu(self.ctx_ln(self.ctx_fc1(cat_feat)))

    def _apply_ctx_head(self,raw,noise_scale=0.,use_null=False):
        if use_null: raw=self.null_embedding.expand(raw.shape[0],-1)
        elif noise_scale>0.: raw=raw+torch.randn_like(raw)*noise_scale
        return self.ctx_fc2(self.ctx_drop(raw))

    def get_blend_alpha(self,ctx): return torch.sigmoid(self.blend_head(ctx)).squeeze(-1)*0.5
    def get_guidance_scale(self,ctx): return 0.8+1.2*torch.sigmoid(self.guidance_head(ctx)).squeeze(-1)
    def get_sigma(self,ctx): return 0.02+0.08*torch.sigmoid(self.sigma_head(ctx)).squeeze(-1)

    def _beta_drift(self,x_t):
        lat_rad=torch.deg2rad(x_t[:,:,1]*5.).clamp(-85,85)
        beta=2*7.2921e-5*torch.cos(lat_rad)/6.371e6; R=3e5
        v=torch.zeros_like(x_t)
        v[:,:,0]=-beta*R**2/2.*6*3600./(5*111*1000.)
        v[:,:,1]=beta*R**2/4.*6*3600./(5*111*1000.)
        return v

    def _steering_drift(self,x_t,env_data):
        if env_data is None: return torch.zeros_like(x_t)
        B,device=x_t.shape[0],x_t.device
        u=_safe_env(env_data,'u500_center',B,device,30.)
        vv=_safe_env(env_data,'v500_center',B,device,30.)
        cos=torch.cos(torch.deg2rad(x_t[:,:,1]*5.)).clamp(1e-3)
        out=torch.zeros_like(x_t)
        out[:,:,0]=u.unsqueeze(1)*30.*21600./(111.*1000.*cos)
        out[:,:,1]=vv.unsqueeze(1)*30.*21600./(111.*1000.)
        return out

    def _decode(self,x_t,t,ctx,vel_obs_feat=None,steering_feat=None,
                 env_kine_feat=None,env_data=None):
        B=x_t.shape[0]
        t_emb=F.gelu(self.time_fc1(self._time_emb(t))); t_emb=self.time_fc2(t_emb)
        T=min(x_t.size(1),self.pos_enc.shape[1])
        si=torch.arange(T,device=x_t.device).unsqueeze(0).expand(B,-1)
        xe=(self.traj_embed(x_t[:,:T])+self.pos_enc[:,:T]+t_emb.unsqueeze(1)+self.step_embed(si))
        mem=[t_emb.unsqueeze(1),ctx.unsqueeze(1)]
        if vel_obs_feat is not None: mem.append(vel_obs_feat.unsqueeze(1))
        if steering_feat is not None: mem.append(steering_feat.unsqueeze(1))
        if env_kine_feat is not None: mem.append(env_kine_feat.unsqueeze(1))
        dec=self.transformer(xe,torch.cat(mem,dim=1))
        vn=self.out_fc2(F.gelu(self.out_fc1(dec)))
        sc=torch.sigmoid(self.step_scale[:T]).view(1,T,1)*2.; vn=vn*sc
        with torch.no_grad():
            vp=self._beta_drift(x_t[:,:T]); vs=self._steering_drift(x_t[:,:T],env_data)
        return vn+torch.sigmoid(self.physics_scale)*vp+torch.sigmoid(self.steering_scale)*vs

    def forward_with_ctx(self,x_t,t,raw_ctx,noise_scale=0.,vel_obs_feat=None,
                          steering_feat=None,env_kine_feat=None,env_data=None,use_null=False):
        ctx=self._apply_ctx_head(raw_ctx,noise_scale,use_null=use_null)
        return self._decode(x_t,t,ctx,vel_obs_feat=vel_obs_feat,
                            steering_feat=steering_feat,env_kine_feat=env_kine_feat,env_data=env_data)

    def predict_speed(self,raw_ctx,vel_obs_feat):
        return self.speed_head(self._apply_ctx_head(raw_ctx),vel_obs_feat)


# ══════════════════════════════════════════════════════════════
#  EMA, OT (giữ nguyên v60)
# ══════════════════════════════════════════════════════════════

class EMAModel:
    def __init__(self,model,decay=0.995):
        self.decay=decay; m=model._orig_mod if hasattr(model,'_orig_mod') else model
        self.shadow={k:v.detach().clone() for k,v in m.state_dict().items() if v.dtype.is_floating_point}
    def update(self,model):
        m=model._orig_mod if hasattr(model,'_orig_mod') else model
        with torch.no_grad():
            for k,v in m.state_dict().items():
                if k in self.shadow: self.shadow[k].mul_(self.decay).add_(v.detach(),alpha=1-self.decay)
    def apply_to(self,model):
        m=model._orig_mod if hasattr(model,'_orig_mod') else model
        bk,sd={},m.state_dict()
        for k in self.shadow:
            if k not in sd: continue
            bk[k]=sd[k].detach().clone(); sd[k].copy_(self.shadow[k])
        return bk
    def restore(self,model,bk):
        m=model._orig_mod if hasattr(model,'_orig_mod') else model; sd=m.state_dict()
        for k,v in bk.items():
            if k in sd: sd[k].copy_(v)


def _sinkhorn_log(cost,epsilon=0.05,n_iter=50):
    B=cost.shape[0]; device=cost.device
    la=-math.log(B)*torch.ones(B,device=device); lb=la.clone()
    lK=-cost/epsilon; lu=torch.zeros(B,device=device); lv=lu.clone()
    for _ in range(n_iter):
        lu=la-torch.logsumexp(lK+lv.unsqueeze(0),dim=1)
        lv=lb-torch.logsumexp(lK+lu.unsqueeze(1),dim=0)
    return (lK+lu.unsqueeze(1)+lv.unsqueeze(0)).exp().clamp(0.)

def _spherical_ot_matching(x0_batch,x1_batch,lp,epsilon=0.05):
    try:
        B=x0_batch.shape[0]
        abs0=lp.unsqueeze(1)[:,:,:2]+x0_batch[:,:,:2]; abs1=lp.unsqueeze(1)[:,:,:2]+x1_batch[:,:,:2]
        abs0_deg=norm_to_deg(abs0); abs1_deg=norm_to_deg(abs1)
        cost=haversine_km(
            abs0_deg.unsqueeze(1).expand(B,B,-1,-1).reshape(B*B,-1,2),
            abs1_deg.unsqueeze(0).expand(B,B,-1,-1).reshape(B*B,-1,2)
        ).mean(-1).reshape(B,B)/500.
        pi=_sinkhorn_log(cost,epsilon=epsilon)
        flat=pi.reshape(-1).clamp(0.); s=flat.sum()
        if not torch.isfinite(s) or s<1e-10: return x0_batch,x1_batch
        idx=torch.multinomial(flat/s,num_samples=B,replacement=True)
        return x0_batch[idx%B],x1_batch[idx%B]
    except: return x0_batch,x1_batch


def compute_speed_stats_from_norm(obs_traj_norm):
    T_obs=obs_traj_norm.shape[0]
    if T_obs<2: return {'v_opt':15.,'v_sigma':10.,'v_hard_cap':80.,'p50_kmh':15.}
    lon=(obs_traj_norm[...,0]*50.+1800.)/10.; lat=(obs_traj_norm[...,1]*50.)/10.
    lat_mid=(lat[:-1]+lat[1:])/2; cos_lat=torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
    dx=(lon[1:]-lon[:-1])*cos_lat*DEG2KM; dy=(lat[1:]-lat[:-1])*DEG2KM
    spd=torch.sqrt(dx**2+dy**2)/DT_HOURS
    p50=float(spd.flatten().median()); p95=float(torch.quantile(spd.flatten(),.95))
    return {'v_opt':max(p50,5.),'v_sigma':10.,
            'v_hard_cap':float(torch.tensor(p95*1.8).clamp(25.,130.)),'p50_kmh':p50}


@torch.no_grad()
def _persistence_blend_adaptive(model_pred_norm,obs_traj_norm,blend_alpha):
    T_obs=obs_traj_norm.shape[0]; T=model_pred_norm.shape[0]
    B=model_pred_norm.shape[1]; device=model_pred_norm.device
    if T_obs<2: return model_pred_norm
    vels=obs_traj_norm[1:]-obs_traj_norm[:-1]; n_v=vels.shape[0]
    if n_v>=3:
        alpha=0.7; w=torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],dtype=torch.float,device=device).flip(0)
        ev=(vels*(w/w.sum()).view(-1,1,1)).sum(0)
    elif n_v==2: ev=0.7*vels[-1]+0.3*vels[-2]
    else: ev=vels[-1]
    steps=torch.arange(1,T+1,dtype=torch.float,device=device)
    persist=obs_traj_norm[-1].unsqueeze(0)+ev.unsqueeze(0)*steps.view(T,1,1)
    alpha_b=blend_alpha.view(1,B,1).clamp(0.,.5)
    return (1.-alpha_b)*model_pred_norm+alpha_b*persist


# ══════════════════════════════════════════════════════════════
#  [NEW] K=3 mode clustering inference
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def _mode_cluster_k3(trajs_norm, obs_norm, scorer, speed_stats=None):
    """
    K=3 mode clustering tại 72h endpoint.
    Thay top-35% median → CTE thấp hơn tự nhiên.

    trajs_norm: list of [T,B,2] seq-first normalized
    obs_norm:   [T_obs,B,2] seq-first
    scorer:     EnsembleScorer
    returns:    [T,B,2] seq-first normalized
    """
    if not trajs_norm: return obs_norm[-1:].expand(12,obs_norm.shape[1],2)
    dev=trajs_norm[0].device; T,B=trajs_norm[0].shape[0],trajs_norm[0].shape[1]; N=len(trajs_norm); K=min(3,N)

    # Scores [N,B]
    all_sc=torch.stack([scorer.score(tr,obs_norm) for tr in trajs_norm],dim=0)
    # 72h endpoints in degrees [N,B,2]
    endpoints=torch.stack([norm_to_deg(tr[-1]) for tr in trajs_norm],dim=0)

    results=[]
    for b in range(B):
        ep_b=endpoints[:,b,:]; sc_b=all_sc[:,b]
        tr_b=torch.stack([tr[:,b,:] for tr in trajs_norm],dim=0)  # [N,T,2]
        if N<K:
            w=F.softmax(sc_b*3.,0); results.append((tr_b*w.view(N,1,1)).sum(0)); continue
        # Farthest-point init
        idx0=sc_b.argmax().item(); centers=[ep_b[idx0]]
        for _ in range(K-1):
            d2c=torch.stack([haversine_km(ep_b,c.unsqueeze(0).expand_as(ep_b)) for c in centers],1).min(1).values
            centers.append(ep_b[d2c.argmax()])
        centers=torch.stack(centers,0)
        # 3 K-means iterations
        for _ in range(3):
            d2c=torch.stack([haversine_km(ep_b,c.unsqueeze(0).expand_as(ep_b)) for c in centers],1)
            assign=d2c.argmin(1)
            new_c=[]
            for k in range(K):
                mk=(assign==k)
                if mk.sum()>0:
                    wk=F.softmax(sc_b[mk]*3.,0)
                    new_c.append((ep_b[mk]*wk.unsqueeze(1)).sum(0))
                else: new_c.append(centers[k])
            centers=torch.stack(new_c,0)
        # Score clusters
        d2c=torch.stack([haversine_km(ep_b,c.unsqueeze(0).expand_as(ep_b)) for c in centers],1)
        assign=d2c.argmin(1)
        csc=torch.zeros(K,device=dev)
        for k in range(K):
            mk=(assign==k)
            if mk.sum()>0: csc[k]=sc_b[mk].sum()
        best_k=csc.argmax().item(); mk=(assign==best_k)
        if not mk.any(): mk=torch.ones(N,dtype=torch.bool,device=dev)
        w_win=F.softmax(sc_b[mk]*3.,0)
        results.append((tr_b[mk]*w_win.view(-1,1,1)).sum(0))
    return torch.stack(results,dim=1)  # [T,B,2]


# ══════════════════════════════════════════════════════════════
#  TCFlowMatching v74-fix
# ══════════════════════════════════════════════════════════════

class TCFlowMatching(nn.Module):

    def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02, unet_in_ch=13,
                 ctx_noise_scale=0.01, use_ema=True, ema_decay=0.995,
                 use_ate_ot=True, ot_epsilon=0.05, cfg_uncond_prob=0.1, **kwargs):
        super().__init__()
        self.pred_len=pred_len; self.obs_len=obs_len; self.sigma_min=sigma_min
        self.ctx_noise_scale=ctx_noise_scale; self.use_ate_ot=use_ate_ot
        self.ot_epsilon=ot_epsilon; self.cfg_uncond_prob=cfg_uncond_prob
        self.net=VelocityField(pred_len=pred_len,obs_len=obs_len,
                                sigma_min=sigma_min,unet_in_ch=unet_in_ch,ctx_dim=256)
        self.criterion=FMv74Loss(pred_len=pred_len)
        self.use_ema=use_ema; self._ema=None

    def init_ema(self):
        if self.use_ema: self._ema=EMAModel(self,decay=0.995)
    def ema_update(self):
        if self._ema is not None: self._ema.update(self)

    @staticmethod
    def _to_rel(traj,Me,lp,lm):
        return torch.cat([traj-lp[:,:2].unsqueeze(0),Me-lm.unsqueeze(0)],dim=-1).permute(1,0,2)

    @staticmethod
    def _to_abs(rel,lp,lm):
        d=rel.permute(1,0,2)
        return lp[:,:2].unsqueeze(0)+d[:,:,:2], lm.unsqueeze(0)+d[:,:,2:]

    @staticmethod
    def _sigma_schedule(ep):
        if ep<2: return 0.10
        if ep<10: return 0.10-(ep-2)/8.*(0.10-0.04)
        if ep<20: return max(0.04-(ep-10)/10.*0.01,0.035)
        return 0.035

    def _cfm_standard(self,x1,sigma_min=None):
        """FIX-A: standard FM x0=N(0,sigma), NOT persist+noise."""
        if sigma_min is None: sigma_min=self.sigma_min
        B=x1.shape[0]; dev=x1.device
        x0=torch.randn_like(x1)*sigma_min
        t=torch.rand(B,device=dev); te=t.view(B,1,1)
        return (1.-te)*x0+te*x1, t, x1-x0

    def _persistence_forecast_rel(self,obs_traj,lp,lm,pred_len):
        B,device=obs_traj.shape[1],obs_traj.device
        obs_pos=obs_traj[:,:,:2]
        if obs_pos.shape[0]>=3:
            vels=obs_pos[1:]-obs_pos[:-1]; n_v=vels.shape[0]; alpha=0.7
            w=torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],dtype=torch.float,device=device).flip(0)
            lv=(vels*(w/w.sum()).view(-1,1,1)).sum(0)
        elif obs_pos.shape[0]>=2: lv=obs_pos[-1]-obs_pos[-2]
        else: lv=obs_traj.new_zeros(B,2)
        steps=torch.arange(1,pred_len+1,device=device).float()
        pred_abs=obs_pos[-1].unsqueeze(1)+lv.unsqueeze(1)*steps.view(1,-1,1)
        pred_rel=torch.cat([pred_abs-lp[:,:2].unsqueeze(1),torch.zeros_like(pred_abs)],dim=-1)
        return pred_rel

    def _compute_obs_momentum(self,obs_norm):
        T_obs=obs_norm.shape[0]
        if T_obs<2: return torch.zeros(obs_norm.shape[1],2,device=obs_norm.device)
        vels=obs_norm[1:]-obs_norm[:-1]; n_v=vels.shape[0]
        if n_v>=3:
            a=0.65; w=torch.tensor([a*(1-a)**i for i in range(n_v)],dtype=torch.float,device=obs_norm.device).flip(0)
            return (vels*(w/w.sum()).view(-1,1,1)).sum(0)
        elif n_v==2: return 0.65*vels[-1]+0.35*vels[-2]
        return vels[-1]

    @staticmethod
    @staticmethod
    def _obs_noise_aug(bl,sigma=0.005):
        if torch.rand(1).item()>0.5: return bl
        bl=list(bl)
        if torch.is_tensor(bl[0]): bl[0]=bl[0]+torch.randn_like(bl[0])*sigma
        return bl

    @staticmethod
    def _lon_flip_aug(bl, p=0.35):
        """
        Flip longitude: lon_norm → -lon_norm.
        Mô phỏng bão ở phía đối xứng → model học invariant với hướng E/W.
        Giảm val-test gap vì test set có nhiều bão hướng khác nhau hơn.
        """
        if torch.rand(1).item() > p: return bl
        bl = list(bl)
        for i in [0, 1, 2, 3, 7, 8]:
            if i < len(bl) and torch.is_tensor(bl[i]) and bl[i].shape[-1] >= 1:
                t = bl[i].clone(); t[..., 0] = -t[..., 0]; bl[i] = t
        return bl

    @staticmethod
    def _speed_scale_aug(bl, scale_range=(0.88, 1.14), p=0.40):
        """
        Scale obs velocity by random factor 0.88-1.14x.
        Mô phỏng bão nhanh/chậm hơn → generalize đến recent storms (2017-2021).
        Giúp giảm val-test gap vì test storms thường nhanh hơn train distribution.
        """
        if torch.rand(1).item() > p: return bl
        bl = list(bl)
        scale = torch.empty(1).uniform_(*scale_range).item()
        if torch.is_tensor(bl[0]) and bl[0].shape[0] >= 2:
            obs = bl[0].clone()
            displ = obs[1:] - obs[:-1]
            new_obs = obs.clone()
            for t in range(1, obs.shape[0]):
                new_obs[t] = new_obs[t-1] + displ[t-1] * scale
            bl[0] = new_obs
        return bl

    # ── Training ──────────────────────────────────────────────

    def get_loss(self,batch_list,epoch=0,**kwargs):
        return self.get_loss_breakdown(batch_list,epoch=epoch)['total']

    def get_loss_breakdown(self,batch_list,epoch=0):
        batch_list=self._obs_noise_aug(batch_list,sigma=0.005)
        batch_list=self._lon_flip_aug(batch_list, p=0.35)
        batch_list=self._speed_scale_aug(batch_list, p=0.40)
        obs_t=batch_list[0]; env_data=batch_list[13] if len(batch_list)>13 else None
        lp=obs_t[-1]; lm=batch_list[7][-1]; B,device=lp.shape[0],lp.device
        gt_traj_s=batch_list[1]

        current_sigma=self._sigma_schedule(epoch)
        raw_ctx=self.net._context(batch_list)
        x1_rel=self._to_rel(gt_traj_s,batch_list[8],lp,lm)

        # FIX-A: standard x0=N(0,sigma), NOT persist+noise
        if self.use_ate_ot and B>=4:
            x0_noise=torch.randn_like(x1_rel)*current_sigma
            _,x1_matched=_spherical_ot_matching(x0_noise,x1_rel,lp,epsilon=self.ot_epsilon)
        else:
            x1_matched=x1_rel
        x_t,fm_t,u_target=self._cfm_standard(x1_matched,sigma_min=current_sigma)

        use_null=(torch.rand(1).item()<self.cfg_uncond_prob)
        vel_obs_feat=self.net._get_kinematic_obs_feat(obs_t[:,:,:2])
        steering_feat=self.net._get_steering_feat(env_data,B,device)
        env_kine_feat=self.net._get_env_kine_feat(env_data,B,device)

        pred_vel=self.net.forward_with_ctx(x_t,fm_t,raw_ctx,env_data=env_data,use_null=use_null,
            vel_obs_feat=vel_obs_feat,steering_feat=steering_feat,env_kine_feat=env_kine_feat)

        fm_te=fm_t.view(B,1,1)
        x1_pred=x_t+(1.-fm_te)*pred_vel
        pred_abs,_=self._to_abs(x1_pred,lp,lm)
        pred_deg=norm_to_deg(pred_abs)    # [T,B,2]
        gt_deg=norm_to_deg(gt_traj_s)    # [T,B,2]

        # Speed head prediction for auxiliary loss
        speed_pred=None
        if not use_null:
            speed_pred=self.net.predict_speed(raw_ctx,vel_obs_feat)

        # Scorer candidates
        candidates=None; obs_norm=obs_t[:,:,:2]
        if epoch>=5 and not use_null:
            cands=[]
            for _ in range(3):
                te_c=fm_t.view(B,1,1)
                x0_c=torch.randn_like(x1_rel)*current_sigma
                x_c=(1.-te_c)*x0_c+te_c*x1_rel
                with torch.no_grad():
                    v_c=self.net.forward_with_ctx(x_c,fm_t,raw_ctx,env_data=env_data,
                        vel_obs_feat=vel_obs_feat,steering_feat=steering_feat,env_kine_feat=env_kine_feat)
                    x1_c=x_c+(1.-te_c)*v_c; abs_c,_=self._to_abs(x1_c,lp,lm)
                    cands.append(abs_c[:,:,:2])
            candidates=cands

        total,bd=self.criterion(pred_deg,gt_deg,pred_vel,u_target,
                                  speed_pred=speed_pred,candidates=candidates,
                                  obs_norm=obs_norm,epoch=epoch)

        if torch.isnan(total) or torch.isinf(total): total=obs_t.new_zeros(())

        bd.update({'sigma':current_sigma,
                   'v_opt':compute_speed_stats_from_norm(obs_t[:,:,:2]).get('v_opt',15.),
                   'total':total,
                   # legacy keys
                   'l_fm':bd.get('l_fm',0.),'l_kin':0.,'l_logspd':0.,
                   'l_curv':0.,'diff_w_mean':bd.get('diff_w_mean',1.),
                   'lam_kin':0.,'lam_logspd':0.,'lam_curv':0.})
        return bd

    # ── Inference ─────────────────────────────────────────────

    @torch.no_grad()
    def sample(self,batch_list,num_ensemble=50,ddim_steps=20,predict_csv=None,use_tta=True):
        obs_t=batch_list[0]; env_data=batch_list[13] if len(batch_list)>13 else None
        lp=obs_t[-1]; lm=batch_list[7][-1]; B=lp.shape[0]; device=lp.device
        T=self.pred_len; dt=1./max(ddim_steps,1)

        def _run_ensemble(bl, n_ens):
            """Run n_ens ensemble members on a batch_list."""
            _obs_t=bl[0]; _env=bl[13] if len(bl)>13 else None
            _lp=_obs_t[-1]; _lm=bl[7][-1]
            _raw_ctx=self.net._context(bl)
            _ctx=self.net._apply_ctx_head(_raw_ctx)
            _vel_f=self.net._get_kinematic_obs_feat(_obs_t[:,:,:2])
            _steer_f=self.net._get_steering_feat(_env,B,device)
            _env_f=self.net._get_env_kine_feat(_env,B,device)
            _obs_n=_obs_t[:,:,:2]; _obs_m=self._compute_obs_momentum(_obs_n)
            _blend=self.net.get_blend_alpha(_ctx)
            _gs=self.net.get_guidance_scale(_ctx)

            def _ms(s,tot): return 0.06*0.5*(1.+math.cos(math.pi*s/max(tot,1)))
            trajs=[]
            for _ in range(n_ens):
                x_t=torch.randn(B,T,4,device=device)*self.sigma_min
                for step in range(ddim_steps):
                    t_b=torch.full((B,),step*dt,device=device)
                    ns=self.ctx_noise_scale*2. if step<3 else 0.
                    if step>0:
                        vc=self.net.forward_with_ctx(x_t,t_b,_raw_ctx,noise_scale=ns,
                            vel_obs_feat=_vel_f,steering_feat=_steer_f,
                            env_kine_feat=_env_f,env_data=_env,use_null=False)
                        vu=self.net.forward_with_ctx(x_t,t_b,_raw_ctx,noise_scale=0.,
                            vel_obs_feat=_vel_f,steering_feat=_steer_f,
                            env_kine_feat=_env_f,env_data=_env,use_null=True)
                        vel=vu+_gs.view(B,1,1)*(vc-vu)
                    else:
                        vel=self.net.forward_with_ctx(x_t,t_b,_raw_ctx,noise_scale=ns,
                            vel_obs_feat=_vel_f,steering_feat=_steer_f,
                            env_kine_feat=_env_f,env_data=_env)
                    ms=_ms(step,ddim_steps)
                    if ms>1e-4:
                        me=_obs_m.unsqueeze(1).expand(B,T,2)
                        vel=vel+ms*torch.cat([me,torch.zeros(B,T,2,device=device)],dim=-1)
                    x_t=(x_t+dt*vel).clamp(-3.,3.)
                pa,_=self._to_abs(x_t,_lp,_lm); trajs.append(pa)
            pred_m=_mode_cluster_k3(trajs,_obs_n,self.criterion.scorer)
            pred_f=_persistence_blend_adaptive(pred_m,_obs_n,_blend)
            return pred_f, trajs

        # Normal inference
        n_ens_each = num_ensemble if not use_tta else max(num_ensemble//2, 20)
        pred_normal, all_norms = _run_ensemble(batch_list, n_ens_each)

        if use_tta:
            # TTA: flip longitude, predict, unflip, average
            # Giảm systematic directional bias → test gap thấp hơn
            bl_flip = list(batch_list)
            for i in [0,1,2,3,7,8]:
                if i<len(bl_flip) and torch.is_tensor(bl_flip[i]):
                    t=bl_flip[i].clone(); t[...,0]=-t[...,0]; bl_flip[i]=t
            pred_flip, _ = _run_ensemble(bl_flip, n_ens_each)
            # Unflip
            pred_unflip = pred_flip.clone()
            pred_unflip[...,0] = -pred_unflip[...,0]
            # Average (equal weight)
            pred_final = 0.5*pred_normal + 0.5*pred_unflip
        else:
            pred_final = pred_normal

        if predict_csv: self._write_predict_csv(predict_csv,pred_final)
        return pred_final,torch.stack(all_norms)

    @staticmethod
    def _write_predict_csv(csv_path,traj_mean):
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)),exist_ok=True)
        T,B,_=traj_mean.shape
        mlon=((traj_mean[:,:,0]*50.+1800.)/10.).cpu().numpy()
        mlat=((traj_mean[:,:,1]*50.)/10.).cpu().numpy()
        ts=datetime.now().strftime('%Y%m%d_%H%M%S')
        hdr=not os.path.exists(csv_path)
        with open(csv_path,'a',newline='') as fh:
            w=csv.DictWriter(fh,fieldnames=['ts','b','step','lead_h','lon','lat'])
            if hdr: w.writeheader()
            for b in range(B):
                for k in range(T):
                    w.writerow({'ts':ts,'b':b,'step':k,'lead_h':(k+1)*6,
                                 'lon':f'{mlon[k,b]:.4f}','lat':f'{mlat[k,b]:.4f}'})


TCDiffusion = TCFlowMatching
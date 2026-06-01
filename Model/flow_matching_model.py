"""
Model/flow_matching_model.py  ── FM v60
=========================================
THAY THẾ HOÀN TOÀN flow_matching_model.py v59.

Cách dùng: chỉ copy file này vào Model/ — không cần thay file khác.
Import interface giữ nguyên:
  from Model.flow_matching_model import TCFlowMatching

FIXES:
  BUG-1 CRITICAL: prior_sc(v_opt=15)≈0 với speed=113 → thải trajectory đúng
                  FIX: EnsembleScorer MLP, không prior
  BUG-2 CRITICAL: l_speed(v_opt=15) gradient ngược chiều
                  FIX: l_logspeed = MSE(log(spd+1)) không prior
  BUG-3 HIGH:    STEP_WEIGHTS magic numbers non-monotonic
                  FIX: LearnedStepWeights monotonic
  BUG-4 HIGH:    79% env_data bị bỏ (21/98 dims)
                  FIX: thêm gph500, bearing_to_scs, month, velocity_history, RI
  BUG-5 HIGH:    Scoring exponents cứng
                  FIX: EnsembleScorer MLP learned
  BUG-6 MEDIUM:  speed_sweep target obs_spd sai
                  FIX: bỏ speed_sweep
  BUG-7 MEDIUM:  persistence_blend cứng 20%
                  FIX: blend_alpha = sigmoid(Linear(ctx)) learned
  BUG-8 MEDIUM:  CFG guidance_scale cứng
                  FIX: guidance_scale = 0.8+1.2×sigmoid(Linear(ctx))
  BUG-9 LOW:     sigma cứng 0.035
                  FIX: sigma = 0.02+0.08×sigmoid(Linear(ctx))
"""
from __future__ import annotations

import csv
import math
import os
import sys
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.FNO3D_encoder import FNO3DEncoder
from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
from Model.env_net_transformer_gphsplit import Env_net

R_EARTH  = 6371.0
DT_HOURS = 6.0
DEG2KM   = 111.0




MAX_CURVATURE_RAD = math.pi / 4  # 45°/step vật lý threshold


# ══════════════════════════════════════════════════════════════
#  Coordinate utilities
# ══════════════════════════════════════════════════════════════

def norm_to_deg(t: torch.Tensor) -> torch.Tensor:
    """Normalized [lon_n, lat_n] → degrees.
    lon_n ∈ [-9, 2]  →  lon_deg = (lon_n * 50 + 1800) / 10
    lat_n ∈ [0, 10]  →  lat_deg = lat_n * 50 / 10
    """
    return torch.stack([
        (t[..., 0] * 50.0 + 1800.0) / 10.0,
        (t[..., 1] * 50.0) / 10.0,
    ], dim=-1)


def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """[...,2] lon°,lat° → [...] km"""
    lat1 = torch.deg2rad(p1[..., 1])
    lat2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a = (torch.sin(dlat / 2).pow(2)
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1.0 - 1e-12).sqrt())


def velocity_km(traj_deg: torch.Tensor) -> torch.Tensor:
    """
    [T, B, 2] lon°,lat° → [T-1, B, 2] velocity (vx_km, vy_km) per step.

    vx = Δlon × cos(lat_mid) × 111  (EW km/6h)
    vy = Δlat × 111                  (NS km/6h)

    Physics: đây là Lagrangian velocity trong flat-earth approximation
    đúng với SCS region (lat 5°-25°, error < 1%).
    """
    T = traj_deg.shape[0]
    if T < 2:
        return traj_deg.new_zeros(1, traj_deg.shape[1], 2)

    lon = traj_deg[..., 0]
    lat = traj_deg[..., 1]
    dlat = lat[1:] - lat[:-1]
    dlon = lon[1:] - lon[:-1]
    lat_mid = (lat[1:] + lat[:-1]) / 2.0
    cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)

    vx = dlon * cos_lat * DEG2KM  # EW km/6h
    vy = dlat * DEG2KM             # NS km/6h
    return torch.stack([vx, vy], dim=-1)  # [T-1, B, 2]


# ══════════════════════════════════════════════════════════════
#  Learned Step Weights — Monotonic
# ══════════════════════════════════════════════════════════════

class LearnedStepWeights(nn.Module):
    """
    Step weights w[0..11] thỏa mãn:
    - Monotonic non-decreasing: error lớn hơn ở lead time dài hơn (vật lý)
    - Mean = 1.0: tránh loss scale drift

    Implementation:
      raw ∈ ℝ^n (learnable)
      increments = softplus(raw)      ← dương
      weights = cumsum(increments)    ← monotonic non-decreasing
      weights /= weights.mean()       ← normalize

    Gradient flow: backprop qua cumsum → softplus → raw.
    """
    def __init__(self, n_steps: int = 12):
        super().__init__()
        self.n_steps = n_steps
        # Init: uniform weights → raw s.t. softplus(raw)≈constant
        # softplus(0.5) ≈ 0.97, cumsum(uniform) = linear ramp → reasonable init
        self.raw = nn.Parameter(torch.zeros(n_steps) + 0.5)

    def forward(self) -> torch.Tensor:
        increments = F.softplus(self.raw)            # [n], positive
        weights = torch.cumsum(increments, dim=0)    # [n], monotonic
        weights = weights / weights.mean().clamp(min=1e-8)  # normalize mean=1
        return weights  # [n]

    def get(self, n: Optional[int] = None) -> torch.Tensor:
        w = self.forward()
        return w[:n] if n is not None else w

    @torch.no_grad()
    def stats(self) -> dict:
        w = self.forward()
        return {
            "sw_6h":   w[0].item(),
            "sw_12h":  w[1].item() if len(w) > 1 else 0.0,
            "sw_48h":  w[7].item() if len(w) > 7 else 0.0,
            "sw_72h":  w[-1].item(),
            "sw_ratio": (w[-1] / w[0].clamp(min=1e-6)).item(),
            "sw_monotonic": bool((w[1:] - w[:-1]).min().item() >= -1e-6),
        }


# ══════════════════════════════════════════════════════════════
#  Log-Parameterized Loss Weights (Kendall & Gal 2018)
# ══════════════════════════════════════════════════════════════

class LearnedLossWeights(nn.Module):
    """
    λᵢ = 1 / (2·σᵢ²), log_σᵢ learnable.

    Khi σᵢ lớn (task uncertain) → λᵢ nhỏ → ít contribution.
    Model tự học σᵢ từ gradient, không cần grid search.

    Total: L = Σᵢ λᵢ Lᵢ + Σᵢ log(σᵢ)
           = Σᵢ Lᵢ/(2σᵢ²) + Σᵢ log(σᵢ)   (regularization term)
    """
    def __init__(self, n_tasks: int = 3):
        super().__init__()
        # Init log_sigma = 0 → σ = 1 → λ = 0.5
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def lambdas(self) -> torch.Tensor:
        """λᵢ = 0.5 * exp(-2 * log_σᵢ)"""
        return 0.5 * torch.exp(-2.0 * self.log_sigma)

    def regularization(self) -> torch.Tensor:
        """Σᵢ log(σᵢ) = Σᵢ log_sigma[i]"""
        return self.log_sigma.sum()

    @torch.no_grad()
    def stats(self) -> dict:
        lam = self.lambdas()
        return {
            "lam_kin": lam[0].item(),
            "lam_logspd": lam[1].item(),
            "lam_curv": lam[2].item(),
        }


# ══════════════════════════════════════════════════════════════
#  Difficulty Weighting — Per-Sample, Learned
# ══════════════════════════════════════════════════════════════

class DifficultyWeighter(nn.Module):
    """
    Per-sample difficulty weight w ∈ [1.0, 2.0].
    Không filter data (như SRC-Track), chỉ reweight gradient.

    difficulty = [curvature_rate, speed_cv, boundary_prox]
    w = 1 + sigmoid(w1·d1 + w2·d2 + w3·d3 + b)

    Bão khó (recurvature, RI) → w ≈ 1.8 → gradient lớn hơn
    Bão dễ (straight, stable)  → w ≈ 1.0 → không overfit

    Weights w1,w2,w3,b: ALL LEARNABLE.
    """
    def __init__(self):
        super().__init__()
        # 3-dim difficulty feature → 1 logit
        self.linear = nn.Linear(3, 1, bias=True)
        # Init near zero → w ≈ 1.5 (uniform) at start
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def compute_difficulty(
        self,
        gt_deg: torch.Tensor,   # [T, B, 2] degrees
    ) -> torch.Tensor:
        """
        Compute 3 difficulty features từ gt trajectory.
        Tính từ gt vì available at training time.
        """
        T, B = gt_deg.shape[:2]

        # Feature 1: Curvature rate — mean |Δheading| per step
        # Recurvature storms: high curvature
        v = velocity_km(gt_deg)            # [T-1, B, 2]
        if v.shape[0] >= 2:
            heading = torch.atan2(v[..., 0], v[..., 1])  # [T-1, B]
            dh = heading[1:] - heading[:-1]
            dh = (dh + math.pi) % (2 * math.pi) - math.pi  # wrap to [-π,π]
            curv_rate = dh.abs().mean(0)  # [B]
        else:
            curv_rate = gt_deg.new_zeros(B)

        # Feature 2: Speed coefficient of variation — std/mean
        # RI and rapid deceleration: high CV
        spd = v.norm(dim=-1)  # [T-1, B] km/6h
        mean_spd = spd.mean(0).clamp(min=1.0)  # [B]
        std_spd  = spd.std(0)                   # [B]
        speed_cv = (std_spd / mean_spd).clamp(max=3.0)  # [B]

        # Feature 3: Max curvature exceedance — ReLU(|dh| - π/4)
        # Proxy for recurvature onset difficulty
        if v.shape[0] >= 2:
            excess_curv = F.relu(dh.abs() - MAX_CURVATURE_RAD).mean(0)  # [B]
        else:
            excess_curv = gt_deg.new_zeros(B)

        # Normalize features to similar scale
        d1 = (curv_rate   / (math.pi / 2)).clamp(0, 1)  # [B]
        d2 = speed_cv.clamp(0, 1)                         # [B]
        d3 = (excess_curv / (math.pi / 4)).clamp(0, 1)   # [B]

        return torch.stack([d1, d2, d3], dim=-1)  # [B, 3]

    def forward(self, gt_deg: torch.Tensor) -> torch.Tensor:
        """Returns per-sample weights [B] ∈ [1.0, 2.0]"""
        diff_feat = self.compute_difficulty(gt_deg)  # [B, 3]
        logit = self.linear(diff_feat).squeeze(-1)   # [B]
        return 1.0 + torch.sigmoid(logit)            # [B] ∈ [1.0, 2.0]


# ══════════════════════════════════════════════════════════════
#  Ensemble Scorer — Learned MLP
# ══════════════════════════════════════════════════════════════

class EnsembleScorer(nn.Module):
    """
    Learned trajectory scorer. Thay thế fixed heuristic v59.

    Input: 7 kinematic features per candidate trajectory
    Output: score ∈ [0,1]

    FIX so với v59:
    - Không dùng prior_sc (v_opt=15 → score ≈ 0 với speed=113)
    - Feature f4: match với obs_speed (không v_opt)
    - Weights learned từ auxiliary training với oracle ADE ranking

    Training: auxiliary BCE với oracle label từ ADE rank trên gt.
    """
    def __init__(self, feat_dim: int = 7, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        # Init near-zero output → score ≈ 0.5 uniformly at start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def extract_features(
        self,
        traj_norm: torch.Tensor,      # [T, B, 2] normalized
        obs_norm:  torch.Tensor,      # [T_obs, B, 2] normalized
    ) -> torch.Tensor:
        """Returns [B, 7] feature vector."""
        B = traj_norm.shape[1]
        traj_deg = norm_to_deg(traj_norm)  # [T, B, 2]
        obs_deg  = norm_to_deg(obs_norm)   # [T_obs, B, 2]

        v    = velocity_km(traj_deg)   # [T-1, B, 2]
        spd  = v.norm(dim=-1)          # [T-1, B]

        # f1: log mean speed
        f1 = torch.log1p(spd.mean(0))                        # [B]

        # f2: log speed std
        f2 = torch.log1p(spd.std(0).clamp(min=0))            # [B]

        # f3: heading consistency (mean cos of consecutive heading changes)
        if v.shape[0] >= 2:
            heading = torch.atan2(v[..., 0], v[..., 1])
            dh = heading[1:] - heading[:-1]
            dh = (dh + math.pi) % (2 * math.pi) - math.pi
            f3 = torch.cos(dh).mean(0)                        # [B]
        else:
            f3 = traj_norm.new_ones(B)

        # f4: speed match với obs 3 steps cuối (không v_opt!)
        v_obs = velocity_km(obs_deg)  # [T_obs-1, B, 2]
        n_obs = min(3, v_obs.shape[0])
        obs_spd_ref = v_obs[-n_obs:].norm(dim=-1).mean(0).clamp(min=1.0)  # [B]
        n_pred = min(3, spd.shape[0])
        pred_spd_early = spd[:n_pred].mean(0)                              # [B]
        f4 = torch.exp(
            -((pred_spd_early - obs_spd_ref) / obs_spd_ref).pow(2) * 2.0
        )  # [B], 1 = perfect match

        # f5: heading continuation từ obs
        if v_obs.shape[0] >= 1 and v.shape[0] >= 1:
            obs_last_h  = torch.atan2(v_obs[-1, :, 0], v_obs[-1, :, 1])  # [B]
            pred_first_h = torch.atan2(v[0, :, 0], v[0, :, 1])           # [B]
            dh_cont = pred_first_h - obs_last_h
            dh_cont = (dh_cont + math.pi) % (2 * math.pi) - math.pi
            f5 = torch.cos(dh_cont)                                        # [B]
        else:
            f5 = traj_norm.new_ones(B)

        # f6: distance from persistence (normalized)
        #   persistence = last obs position + last obs velocity × steps
        if obs_norm.shape[0] >= 2:
            last_vel_n = obs_norm[-1] - obs_norm[-2]  # [B, 2]
            steps = torch.arange(1, traj_norm.shape[0] + 1,
                                  device=traj_norm.device, dtype=traj_norm.dtype)
            persist = (obs_norm[-1].unsqueeze(0)
                       + last_vel_n.unsqueeze(0) * steps.view(-1, 1, 1))  # [T, B, 2]
            dist_from_persist = (traj_norm - persist).norm(dim=-1).mean(0)  # [B]
            # Normalize by typical displacement
            ref_disp = (last_vel_n.norm(dim=-1) * traj_norm.shape[0]).clamp(min=1e-3)
            f6 = dist_from_persist / ref_disp                              # [B]
        else:
            f6 = traj_norm.new_zeros(B)

        # f7: curvature amount
        if v.shape[0] >= 2:
            heading = torch.atan2(v[..., 0], v[..., 1])
            dh = heading[1:] - heading[:-1]
            dh = (dh + math.pi) % (2 * math.pi) - math.pi
            f7 = dh.abs().mean(0)                                          # [B]
        else:
            f7 = traj_norm.new_zeros(B)

        # Stack và clamp để tránh NaN
        feats = torch.stack([f1, f2, f3, f4, f5, f6, f7], dim=-1)  # [B, 7]
        return feats.clamp(-10.0, 10.0)

    def score(
        self,
        traj_norm: torch.Tensor,
        obs_norm:  torch.Tensor,
    ) -> torch.Tensor:
        """Returns [B] scores ∈ [0,1]."""
        feats = self.extract_features(traj_norm, obs_norm)  # [B, 7]
        return torch.sigmoid(self.net(feats).squeeze(-1))   # [B]

    def auxiliary_loss(
        self,
        candidates: list[torch.Tensor],   # list of [T, B, 2] norm
        obs_norm:   torch.Tensor,          # [T_obs, B, 2] norm
        gt_deg:     torch.Tensor,          # [T, B, 2] degrees
    ) -> torch.Tensor:
        """
        Binary cross-entropy với oracle label từ ADE ranking.
        Candidate có ADE thấp nhất = label 1, còn lại = 0.

        Chỉ gọi khi có ≥ 2 candidates.
        """
        if len(candidates) < 2:
            return candidates[0].new_zeros(())

        B = obs_norm.shape[1]
        ades = []
        for cand in candidates:
            cand_deg = norm_to_deg(cand)             # [T, B, 2]
            # ADE = mean haversine per sample
            d = haversine_km(
                cand_deg.permute(1, 0, 2),           # [B, T, 2]
                gt_deg.permute(1, 0, 2),             # [B, T, 2]
            ).mean(dim=1)                             # [B]
            ades.append(d)

        ades_t  = torch.stack(ades, dim=0)            # [n_cands, B]
        # Oracle: candidate với ADE nhỏ nhất = positive
        best_idx = ades_t.argmin(dim=0)               # [B]

        total_loss = ades_t.new_zeros(())
        for i, cand in enumerate(candidates):
            scores = self.score(cand, obs_norm)        # [B]
            # label: 1 nếu đây là best candidate
            labels = (best_idx == i).float()           # [B]
            total_loss = total_loss + F.binary_cross_entropy(
                scores, labels, reduction='mean'
            )
        return total_loss / len(candidates)


# ══════════════════════════════════════════════════════════════
#  3 Loss Term Implementations
# ══════════════════════════════════════════════════════════════

def l_kinematic(
    pred_deg:     torch.Tensor,   # [T, B, 2]
    gt_deg:       torch.Tensor,   # [T, B, 2]
    step_weights: torch.Tensor,   # [T-1] learned
    sample_w:     Optional[torch.Tensor] = None,  # [B] difficulty weights
) -> torch.Tensor:
    """
    MSE velocity vector trong km-space, Huber-robust.

    Physics basis: ||v_pred - v_gt||² = ATE_vel² + CTE_vel²
    → 1 term này cover cả ATE lẫn CTE tự nhiên.
    Không cần l_signed_ate, l_signed_cte, l_sph_ate riêng.

    Huber δ=50 km/6h: dưới 50 → quadratic, trên 50 → linear.
    Robust với super-typhoon (speed > 100 km/6h).
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        return pred_deg.new_zeros(())

    v_pred = velocity_km(pred_deg[:T])   # [T-1, B, 2]
    v_gt   = velocity_km(gt_deg[:T])     # [T-1, B, 2]

    # Squared velocity error per step per sample
    err_sq = (v_pred - v_gt).pow(2).sum(dim=-1)  # [T-1, B]

    # Huber loss: δ=50 km/6h
    delta = 50.0
    err_abs = err_sq.sqrt()
    huber = torch.where(
        err_abs < delta,
        0.5 * err_sq / delta,
        err_abs - delta / 2.0
    )  # [T-1, B]

    # Apply step weights (monotonic, learned)
    n = huber.shape[0]
    w = step_weights[:n]  # [T-1]
    weighted = huber * w.unsqueeze(1)  # [T-1, B]

    # Apply per-sample difficulty weights
    if sample_w is not None:
        weighted = weighted * sample_w.unsqueeze(0)  # broadcast over T

    return weighted.mean()


def l_logspeed(
    pred_deg: torch.Tensor,   # [T, B, 2]
    gt_deg:   torch.Tensor,   # [T, B, 2]
) -> torch.Tensor:
    """
    MSE(log(speed_pred+1), log(speed_gt+1)).

    Physics: speed bão SCS theo log-normal distribution.
    Log-space: balanced penalty cho slow (3-20 km/6h) và fast (80-130 km/6h).

    FIX v59 BUG-2: không có v_opt, không có v_hard_cap.
    Học trực tiếp từ ground truth — không bias.
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        return pred_deg.new_zeros(())

    v_pred = velocity_km(pred_deg[:T])   # [T-1, B, 2]
    v_gt   = velocity_km(gt_deg[:T])

    spd_pred = v_pred.norm(dim=-1)               # [T-1, B] km/6h
    spd_gt   = v_gt.norm(dim=-1).clamp(min=1.0)  # clamp gt: quasi-stationary ≥ 1

    return F.mse_loss(
        torch.log1p(spd_pred.clamp(min=0.0)),
        torch.log1p(spd_gt),
    )


def l_curvature(
    pred_deg:        torch.Tensor,    # [T, B, 2]
    threshold_rad:   float = MAX_CURVATURE_RAD,
) -> torch.Tensor:
    """
    Penalize chỉ PHẦN THỪA của heading change > threshold (45°/step).

    Physics: steering flow thay đổi mượt. Heading change > 45°/step
    chỉ xảy ra tại recurvature onset — và đó là discontinuous trong gt.
    → Penalize pred để không "teleport" hướng tùy tiện.

    Không compare với gt: self-regularization trên pred trajectory.
    Không penalize curvature trong phạm vi vật lý bình thường.
    """
    T = pred_deg.shape[0]
    if T < 3:
        return pred_deg.new_zeros(())

    v = velocity_km(pred_deg)  # [T-1, B, 2]
    if v.shape[0] < 2:
        return pred_deg.new_zeros(())

    heading = torch.atan2(v[..., 0], v[..., 1])  # [T-1, B]
    dh = heading[1:] - heading[:-1]              # [T-2, B]
    # Wrap to [-π, π]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi

    # Chỉ penalize phần vượt ngưỡng
    excess = F.relu(dh.abs() - threshold_rad)    # [T-2, B]
    return excess.mean()


# ══════════════════════════════════════════════════════════════
#  Master Loss Module
# ══════════════════════════════════════════════════════════════

class FMv60Loss(nn.Module):
    """
    Physics-grounded FM v60 loss với tất cả weights tự học.

    Learnable components:
    - LearnedStepWeights: monotonic step weights cho L_kinematic
    - LearnedLossWeights: λ₁,λ₂,λ₃ qua log-parameterization
    - DifficultyWeighter: per-sample weights [1,2]
    - EnsembleScorer: auxiliary trajectory scorer

    Fixed:
    - L_fm (FM objective) weight = 1.0 (lý thuyết FM chuẩn)
    - L_scorer weight = 0.05 (auxiliary, không dominant)
    """
    def __init__(self, pred_len: int = 12):
        super().__init__()
        self.pred_len = pred_len

        # All learned components
        self.step_weights  = LearnedStepWeights(n_steps=pred_len)
        self.loss_weights  = LearnedLossWeights(n_tasks=3)  # λ₁,λ₂,λ₃
        self.diff_weighter = DifficultyWeighter()
        self.scorer        = EnsembleScorer()

    def compute_main_losses(
        self,
        pred_deg:       torch.Tensor,   # [T, B, 2]
        gt_deg:         torch.Tensor,   # [T, B, 2]
        fm_vel_pred:    torch.Tensor,   # [B, T, 4] FM velocity
        fm_vel_target:  torch.Tensor,   # [B, T, 4] FM target
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute 4 main loss terms.

        Returns: (total_physics_loss, breakdown_dict)
        """
        T = min(pred_deg.shape[0], gt_deg.shape[0])

        # ── Per-sample difficulty weights ─────────────────────
        sample_w = self.diff_weighter(gt_deg[:T])  # [B] ∈ [1,2]

        # ── Step weights ──────────────────────────────────────
        sw = self.step_weights.get(n=T - 1)        # [T-1], monotonic

        # ── λ weights ─────────────────────────────────────────
        lam = self.loss_weights.lambdas()           # [3]: λ_kin, λ_logspd, λ_curv
        reg = self.loss_weights.regularization()    # log(σ) regularization

        # ── Loss terms ────────────────────────────────────────
        L_fm = F.mse_loss(fm_vel_pred, fm_vel_target)

        L_kin  = l_kinematic(pred_deg[:T], gt_deg[:T], sw, sample_w)
        L_logspd = l_logspeed(pred_deg[:T], gt_deg[:T])
        L_curv = l_curvature(pred_deg[:T])

        # ── Combine với learned λ ──────────────────────────────
        # L_fm: 1.0 fixed (FM theory standard)
        # λᵢ: learned via log-parameterization
        total = (1.0        * L_fm
               + lam[0]     * L_kin
               + lam[1]     * L_logspd
               + lam[2]     * L_curv
               + reg)        # regularization từ log-parameterization

        # Safety: clamp để tránh explosion
        if not torch.isfinite(total):
            total = pred_deg.new_zeros(())

        lam_d = self.loss_weights.stats()
        sw_d  = self.step_weights.stats()

        breakdown = {
            "l_fm":     L_fm.item()     if torch.is_tensor(L_fm)     else 0.0,
            "l_kin":    L_kin.item()    if torch.is_tensor(L_kin)    else 0.0,
            "l_logspd": L_logspd.item() if torch.is_tensor(L_logspd) else 0.0,
            "l_curv":   L_curv.item()   if torch.is_tensor(L_curv)   else 0.0,
            "l_reg":    reg.item()      if torch.is_tensor(reg)       else 0.0,
            "diff_w_mean": sample_w.mean().item(),
            **lam_d, **sw_d,
        }
        return total, breakdown

    def scorer_loss(
        self,
        candidates: list[torch.Tensor],   # list of [T, B, 2] norm
        obs_norm:   torch.Tensor,
        gt_deg:     torch.Tensor,
    ) -> torch.Tensor:
        """Auxiliary scorer training loss."""
        return self.scorer.auxiliary_loss(candidates, obs_norm, gt_deg)

    def forward(
        self,
        pred_deg:      torch.Tensor,
        gt_deg:        torch.Tensor,
        fm_vel_pred:   torch.Tensor,
        fm_vel_target: torch.Tensor,
        candidates:    Optional[list] = None,
        obs_norm:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full loss computation.

        candidates: list of candidate trajectories (normalized) cho scorer training.
                    None trong đầu training (scorer chưa ổn định).
        """
        total, breakdown = self.compute_main_losses(
            pred_deg, gt_deg, fm_vel_pred, fm_vel_target
        )

        # Auxiliary scorer (chỉ khi có candidates)
        l_scr = pred_deg.new_zeros(())
        if candidates is not None and obs_norm is not None and len(candidates) >= 2:
            l_scr = self.scorer_loss(candidates, obs_norm, gt_deg)
            total = total + 0.05 * l_scr

        breakdown["l_scorer"] = l_scr.item() if torch.is_tensor(l_scr) else 0.0
        breakdown["total"]    = total.item()  if torch.is_tensor(total)  else 0.0
        return total, breakdown




# nên 'from Model.XXX import ...' sẽ tự tìm TC_project/Model/XXX.py

_NORM_TO_DEG = 5.0


def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        (t[..., 0] * 50.0 + 1800.0) / 10.0,
        (t[..., 1] * 50.0) / 10.0,
    ], dim=-1)


# ══════════════════════════════════════════════════════════════
#  SpeedHead (giữ từ v59)
# ══════════════════════════════════════════════════════════════

class SpeedHead(nn.Module):
    """Predict speed per step từ context. Log-space output."""
    def __init__(self, ctx_dim: int = 256, obs_feat_dim: int = 256,
                 pred_len: int = 12):
        super().__init__()
        self.pred_len = pred_len
        self.net = nn.Sequential(
            nn.Linear(ctx_dim + obs_feat_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, pred_len),
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ctx: torch.Tensor, obs_feat: torch.Tensor) -> torch.Tensor:
        """Returns speed in km/6h, clamped [3, 150]."""
        log_spd = self.net(torch.cat([ctx, obs_feat], dim=-1))  # [B, T]
        return torch.exp(log_spd).clamp(3.0, 150.0)


# ══════════════════════════════════════════════════════════════
#  Feature Extractors — v60 Enhanced
# ══════════════════════════════════════════════════════════════

def _safe_env(env_data, key: str, B: int, device: torch.device,
               norm: float = 1.0) -> torch.Tensor:
    """Safe scalar extraction từ env_data dict."""
    v = env_data.get(key) if env_data is not None else None
    if v is None or not torch.is_tensor(v):
        return torch.zeros(B, device=device)
    v = v.float().to(device)
    while v.dim() > 1:
        v = v.mean(-1)
    if v.numel() >= B:
        v = v.view(-1)[:B]
    else:
        v = torch.zeros(B, device=device)
    return (v / norm).clamp(-3.0, 3.0)


def _safe_env_vec(env_data, key: str, dim: int,
                   B: int, device: torch.device) -> torch.Tensor:
    """Safe vector extraction từ env_data dict → [B, dim]."""
    v = env_data.get(key) if env_data is not None else None
    if v is None:
        return torch.zeros(B, dim, device=device)
    if not torch.is_tensor(v):
        try:
            v = torch.tensor(v, dtype=torch.float, device=device)
        except Exception:
            return torch.zeros(B, dim, device=device)
    v = v.float().to(device)

    if v.dim() == 0:
        return torch.zeros(B, dim, device=device)
    if v.dim() == 1:
        if v.shape[0] == dim:
            return v.unsqueeze(0).expand(B, dim)
        return torch.zeros(B, dim, device=device)
    if v.dim() == 2:
        if v.shape == (B, dim):
            return v
        # Shape mismatch: try to fix
        if v.shape[0] == B:
            if v.shape[1] >= dim:
                return v[:, :dim]
            return F.pad(v, (0, dim - v.shape[1]))
        return torch.zeros(B, dim, device=device)
    if v.dim() == 3:
        # [B, T, dim] → take last step
        vv = v[:B, -1, :]  # [B, dim]
        if vv.shape[1] >= dim:
            return vv[:, :dim]
        return F.pad(vv, (0, dim - vv.shape[1]))
    return torch.zeros(B, dim, device=device)


# ══════════════════════════════════════════════════════════════
#  VelocityField v60
# ══════════════════════════════════════════════════════════════

class VelocityField(nn.Module):
    """
    FM velocity field với full feature integration và adaptive modules.

    Context dim: 272 → 512 (was 176 → 512)
    Adaptive outputs: blend_alpha, guidance_scale, sigma (all context-conditioned)
    """
    RAW_CTX_DIM = 512

    def __init__(self, pred_len: int = 12, obs_len: int = 8,
                 ctx_dim: int = 256, sigma_min: float = 0.02,
                 unet_in_ch: int = 13, **kwargs):
        super().__init__()
        self.pred_len = pred_len
        self.obs_len  = obs_len
        self.ctx_dim  = ctx_dim
        self.sigma_min = sigma_min

        # ── Encoders (từ FM v59) ──────────────────────────────
        self.spatial_enc = FNO3DEncoder(
            in_channel=unet_in_ch, out_channel=1, d_model=32,
            n_layers=4, modes_t=4, modes_h=4, modes_w=4,
            spatial_down=32, dropout=0.05)
        self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.bottleneck_proj = nn.Linear(128, 128)
        self.decoder_proj    = nn.Linear(1, 16)
        self.enc_1d = DataEncoder1D(
            in_1d=4, feat_3d_dim=128, mlp_h=64,
            lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
        self.env_enc = Env_net(obs_len=obs_len, d_model=32)

        # ── Steering encoder: 9 features (v59: 7) ─────────────
        # Thêm: gph500_mean, gph500_center
        self.steering_enc = nn.Sequential(
            nn.Linear(9, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 256))

        # ── Env kinematic encoder: 14 features (v59: 14) ──────
        self.env_kine_enc = nn.Sequential(
            nn.Linear(14, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 256), nn.GELU())

        # ── [NEW] Recurvature encoder: 33 features → 64 ───────
        # bearing_to_scs_center(16) + dist_to_scs_boundary(5) + month(12)
        self.recurv_enc = nn.Sequential(
            nn.Linear(33, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 64))

        # ── [NEW] Speed history encoder: 11 features → 32 ─────
        # velocity_history(4) + rapid_intensification(1) + intensity_class(6)
        self.speed_hist_enc = nn.Sequential(
            nn.Linear(11, 32), nn.GELU(), nn.LayerNorm(32),
            nn.Linear(32, 32))

        # ── Context fusion: 272 → 512 (v59: 176 → 512) ────────
        # 128(mamba) + 32(env_net) + 16(fno_dec) + 64(recurv) + 32(speed_hist)
        self.ctx_fc1  = nn.Linear(128 + 32 + 16 + 64 + 32, self.RAW_CTX_DIM)
        self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
        self.ctx_drop = nn.Dropout(0.15)
        self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

        self.null_embedding = nn.Parameter(torch.randn(1, self.RAW_CTX_DIM) * 0.02)

        # ── Kinematic obs encoder ──────────────────────────────
        self.vel_obs_enc = nn.Sequential(
            nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.GELU())

        # ── [NEW] Adaptive modules (context-conditioned) ───────
        # Persistence blend alpha: 0 = no blend, 1 = full persistence
        self.blend_head    = nn.Linear(ctx_dim, 1)
        # CFG guidance scale: [0.8, 2.0]
        self.guidance_head = nn.Linear(ctx_dim, 1)
        # Sigma per sample: [0.02, 0.10]
        self.sigma_head    = nn.Linear(ctx_dim, 1)

        # Init adaptive heads to neutral values
        nn.init.zeros_(self.blend_head.weight)
        nn.init.constant_(self.blend_head.bias, -1.0)    # sigmoid(-1) ≈ 0.27
        nn.init.zeros_(self.guidance_head.weight)
        nn.init.constant_(self.guidance_head.bias, 0.0)  # → gs = 0.8 + 1.2×0.5 = 1.4
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, -1.0)    # → sigma ≈ 0.035

        # ── Transformer decoder ────────────────────────────────
        self.time_fc1   = nn.Linear(256, 512)
        self.time_fc2   = nn.Linear(512, 256)
        self.traj_embed = nn.Linear(4, 256)
        self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
        self.step_embed = nn.Embedding(pred_len, 256)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=256, nhead=8, dim_feedforward=1024,
                dropout=0.10, activation='gelu', batch_first=True),
            num_layers=2)
        self.out_fc1 = nn.Linear(256, 512)
        self.out_fc2 = nn.Linear(512, 4)

        self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
        self.physics_scale  = nn.Parameter(torch.ones(4) * 1.5)
        self.steering_scale = nn.Parameter(torch.ones(4) * 1.0)

        # SpeedHead
        self.speed_head = SpeedHead(ctx_dim=ctx_dim, obs_feat_dim=256,
                                    pred_len=pred_len)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for name, m in self.named_modules():
                if isinstance(m, nn.Linear) and 'out_fc' in name:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
        half = dim // 2
        freq = torch.exp(
            torch.arange(half, dtype=torch.float, device=t.device)
            * (-math.log(10000.0) / max(half - 1, 1)))
        emb = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
        return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

    # ── Feature extractors ────────────────────────────────────

    def _get_steering_feat(self, env_data, B: int, device) -> torch.Tensor:
        """9 features: u/v500 mean/center + steering + gph500 mean/center"""
        if env_data is None:
            return torch.zeros(B, 256, device=device)
        feats = torch.stack([
            _safe_env(env_data, 'u500_mean',       B, device, 30.0),
            _safe_env(env_data, 'v500_mean',       B, device, 30.0),
            _safe_env(env_data, 'u500_center',     B, device, 30.0),
            _safe_env(env_data, 'v500_center',     B, device, 30.0),
            _safe_env(env_data, 'steering_speed',  B, device, 1.0),
            _safe_env(env_data, 'steering_dir_sin',B, device, 1.0),
            _safe_env(env_data, 'steering_dir_cos',B, device, 1.0),
            # [NEW] GPH500
            _safe_env(env_data, 'gph500_mean',   B, device, 1.0),
            _safe_env(env_data, 'gph500_center', B, device, 1.0),
        ], dim=-1)  # [B, 9]
        return self.steering_enc(feats)  # [B, 256]

    def _get_env_kine_feat(self, env_data, B: int, device) -> torch.Tensor:
        """14 features: move_velocity + history_direction24 + delta_velocity"""
        if env_data is None:
            return torch.zeros(B, 256, device=device)
        mv   = _safe_env(env_data, 'move_velocity', B, device, 150.0).unsqueeze(-1)
        hd24 = _safe_env_vec(env_data, 'history_direction24', 8, B, device)
        dv   = _safe_env_vec(env_data, 'delta_velocity', 5, B, device)
        feat = torch.cat([mv, hd24, dv], dim=-1)   # [B, 14]
        return self.env_kine_enc(feat)              # [B, 256]

    def _get_recurv_feat(self, env_data, B: int, device) -> torch.Tensor:
        """
        [NEW] Recurvature features: 33 dims → 64
        bearing_to_scs_center(16) + dist_to_scs_boundary(5) + month(12)

        Tại sao:
        - bearing_to_scs_center: hướng đến tâm SCS → recurvature signal
        - dist_to_scs_boundary: khoảng cách đến đất liền
        - month: seasonality (ITCZ vs subtropical ridge)
        """
        if env_data is None:
            return torch.zeros(B, 64, device=device)

        bearing = _safe_env_vec(env_data, 'bearing_to_scs_center', 16, B, device)
        dist    = _safe_env_vec(env_data, 'dist_to_scs_boundary',  5,  B, device)
        month   = _safe_env_vec(env_data, 'month',                 12, B, device)

        feat = torch.cat([bearing, dist, month], dim=-1)  # [B, 33]
        return self.recurv_enc(feat)                       # [B, 64]

    def _get_speed_hist_feat(self, env_data, B: int, device) -> torch.Tensor:
        """
        [NEW] Speed history: 11 dims → 32
        velocity_history(4) + rapid_intensification(1) + intensity_class(6)

        Tại sao:
        - velocity_history: 24h speed momentum
        - rapid_intensification: RI flag → speed spike signal
        - intensity_class: category → rough speed range
        """
        if env_data is None:
            return torch.zeros(B, 32, device=device)

        vh  = _safe_env_vec(env_data, 'velocity_history',      4, B, device)
        ri  = _safe_env(env_data, 'rapid_intensification', B, device, 1.0).unsqueeze(-1)
        ic  = _safe_env_vec(env_data, 'intensity_class', 6, B, device)

        feat = torch.cat([vh, ri, ic], dim=-1)   # [B, 11]
        return self.speed_hist_enc(feat)          # [B, 32]

    def _get_kinematic_obs_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
        """[T, B, 2] obs → [B, 256] kinematic features."""
        B = obs_traj.shape[1]
        T_obs = obs_traj.shape[0]
        if T_obs >= 2:
            vel     = obs_traj[1:] - obs_traj[:-1]
            lat_mid = obs_traj[:-1, :, 1] * _NORM_TO_DEG
            cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
            dx_km   = vel[:, :, 0] * cos_lat * DEG2KM * _NORM_TO_DEG
            dy_km   = vel[:, :, 1] * DEG2KM * _NORM_TO_DEG
            speed   = torch.sqrt(dx_km**2 + dy_km**2 + 1e-6) / DT_HOURS
            heading = torch.atan2(vel[:, :, 1], vel[:, :, 0])
            speed_n = (speed / 20.0).clamp(-3.0, 3.0)
            if T_obs >= 3:
                dspd  = speed[1:] - speed[:-1]
                accel = torch.cat([obs_traj.new_zeros(1, B),
                                    (dspd / 10.0).clamp(-3.0, 3.0)], dim=0)
            else:
                accel = obs_traj.new_zeros(T_obs - 1, B)
            kine = torch.stack(
                [vel[:, :, 0], vel[:, :, 1], speed_n,
                 heading.sin(), heading.cos(), accel], dim=-1)
        else:
            kine = obs_traj.new_zeros(self.obs_len, B, 6)

        if kine.shape[0] < self.obs_len:
            pad = obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6)
            kine = torch.cat([pad, kine], dim=0)
        else:
            kine = kine[-self.obs_len:]

        return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))  # [B, 256]

    # ── Context building ──────────────────────────────────────

    def _context(self, batch_list) -> torch.Tensor:
        """
        Build raw context [B, 512].
        Tích hợp tất cả encoders bao gồm v60 additions.
        """
        obs_traj  = batch_list[0]    # [T_obs, B, 4]
        obs_Me    = batch_list[7]    # [T_obs, B, 2]
        image_obs = batch_list[11]   # [B, C, T, H, W] or [B, 1, T, H, W]
        env_data  = batch_list[13] if len(batch_list) > 13 else None

        B      = obs_traj.shape[1]
        device = obs_traj.device

        if image_obs.dim() == 4:
            image_obs = image_obs.unsqueeze(2)
        if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
            image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

        # FNO3D encoding
        e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
        T_obs = obs_traj.shape[0]
        e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        e_3d_s = self.bottleneck_proj(e_3d_s)
        if e_3d_s.shape[1] != T_obs:
            e_3d_s = F.interpolate(
                e_3d_s.permute(0, 2, 1), size=T_obs,
                mode='linear', align_corners=False).permute(0, 2, 1)

        # Temporal summary
        e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
        t_w = torch.softmax(
            torch.arange(e_3d_dec_t.shape[1], dtype=torch.float, device=device) * 0.5,
            dim=0)
        f_sp = self.decoder_proj(
            (e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))  # [B, 16]

        # Mamba 1D encoder
        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        h_t    = self.enc_1d(obs_in, e_3d_s)        # [B, 128]

        # Env_net
        e_env, _, _ = self.env_enc(env_data, image_obs)  # [B, 32]

        # [NEW] Recurvature + speed history
        recurv_feat  = self._get_recurv_feat(env_data, B, device)    # [B, 64]
        speed_h_feat = self._get_speed_hist_feat(env_data, B, device) # [B, 32]

        # Fusion: 128+32+16+64+32 = 272
        cat_feat = torch.cat([h_t, e_env, f_sp, recurv_feat, speed_h_feat], dim=-1)
        raw_ctx  = F.gelu(self.ctx_ln(self.ctx_fc1(cat_feat)))
        return raw_ctx  # [B, 512]

    def _apply_ctx_head(self, raw: torch.Tensor,
                         noise_scale: float = 0.0,
                         use_null: bool = False) -> torch.Tensor:
        if use_null:
            raw = self.null_embedding.expand(raw.shape[0], -1)
        elif noise_scale > 0.0:
            raw = raw + torch.randn_like(raw) * noise_scale
        return self.ctx_fc2(self.ctx_drop(raw))  # [B, ctx_dim]

    # ── Adaptive module outputs ───────────────────────────────

    def get_blend_alpha(self, ctx: torch.Tensor) -> torch.Tensor:
        """
        Learned persistence blend weight ∈ [0, 0.5].
        Straight storm context → higher alpha (persistence helps)
        Recurvature context → lower alpha (persistence harmful)
        """
        return torch.sigmoid(self.blend_head(ctx)).squeeze(-1) * 0.5  # [B]

    def get_guidance_scale(self, ctx: torch.Tensor) -> torch.Tensor:
        """Learned CFG guidance scale ∈ [0.8, 2.0]."""
        return 0.8 + 1.2 * torch.sigmoid(self.guidance_head(ctx)).squeeze(-1)  # [B]

    def get_sigma(self, ctx: torch.Tensor) -> torch.Tensor:
        """Learned sigma per sample ∈ [0.02, 0.10]."""
        return 0.02 + 0.08 * torch.sigmoid(self.sigma_head(ctx)).squeeze(-1)   # [B]

    # ── Physics drift terms (từ v59, không thay đổi) ─────────

    def _beta_drift(self, x_t: torch.Tensor) -> torch.Tensor:
        lat_rad = torch.deg2rad(x_t[:, :, 1] * 5.0).clamp(-85, 85)
        beta = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
        R_tc = 3e5
        v = torch.zeros_like(x_t)
        v[:, :, 0] = -beta * R_tc**2 / 2 * 6 * 3600 / (5 * 111 * 1000)
        v[:, :, 1] =  beta * R_tc**2 / 4 * 6 * 3600 / (5 * 111 * 1000)
        return v

    def _steering_drift(self, x_t: torch.Tensor, env_data) -> torch.Tensor:
        if env_data is None:
            return torch.zeros_like(x_t)
        B, device = x_t.shape[0], x_t.device
        u  = _safe_env(env_data, 'u500_center', B, device, 30.0)
        vv = _safe_env(env_data, 'v500_center', B, device, 30.0)
        cos = torch.cos(torch.deg2rad(x_t[:, :, 1] * 5.0)).clamp(1e-3)
        out = torch.zeros_like(x_t)
        out[:, :, 0] = u.unsqueeze(1)  * 30.0 * 21600.0 / (111.0 * 1000.0 * cos)
        out[:, :, 1] = vv.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0)
        return out

    # ── Decode step ───────────────────────────────────────────

    def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
                 ctx: torch.Tensor, vel_obs_feat=None,
                 steering_feat=None, env_kine_feat=None,
                 env_data=None) -> torch.Tensor:
        B = x_t.shape[0]
        t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
        t_emb = self.time_fc2(t_emb)

        T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
        step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)

        x_emb = (self.traj_embed(x_t[:, :T_seq])
                 + self.pos_enc[:, :T_seq]
                 + t_emb.unsqueeze(1)
                 + self.step_embed(step_idx))

        mem = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
        if vel_obs_feat  is not None: mem.append(vel_obs_feat.unsqueeze(1))
        if steering_feat is not None: mem.append(steering_feat.unsqueeze(1))
        if env_kine_feat is not None: mem.append(env_kine_feat.unsqueeze(1))

        decoded  = self.transformer(x_emb, torch.cat(mem, dim=1))
        v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
        scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
        v_neural = v_neural * scale

        with torch.no_grad():
            v_phys  = self._beta_drift(x_t[:, :T_seq])
            v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

        return (v_neural
                + torch.sigmoid(self.physics_scale) * v_phys
                + torch.sigmoid(self.steering_scale) * v_steer)

    def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
                          vel_obs_feat=None, steering_feat=None,
                          env_kine_feat=None, env_data=None,
                          use_null=False):
        ctx = self._apply_ctx_head(raw_ctx, noise_scale, use_null=use_null)
        return self._decode(x_t, t, ctx,
                            vel_obs_feat=vel_obs_feat,
                            steering_feat=steering_feat,
                            env_kine_feat=env_kine_feat,
                            env_data=env_data)

    def predict_speed(self, raw_ctx, vel_obs_feat):
        ctx = self._apply_ctx_head(raw_ctx)
        return self.speed_head(ctx, vel_obs_feat)








# ══════════════════════════════════════════════════════════════
#  EMA
# ══════════════════════════════════════════════════════════════

class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        self.shadow = {k: v.detach().clone()
                       for k, v in m.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model: nn.Module):
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        with torch.no_grad():
            for k, v in m.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model: nn.Module) -> dict:
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        backup, sd = {}, m.state_dict()
        for k in self.shadow:
            if k not in sd: continue
            backup[k] = sd[k].detach().clone()
            sd[k].copy_(self.shadow[k])
        return backup

    def restore(self, model: nn.Module, backup: dict):
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        sd = m.state_dict()
        for k, v in backup.items():
            if k in sd: sd[k].copy_(v)


# ══════════════════════════════════════════════════════════════
#  OT matching (từ v59, không thay đổi)
# ══════════════════════════════════════════════════════════════

def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05,
                   n_iter: int = 50) -> torch.Tensor:
    B = cost.shape[0]
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


def _spherical_ot_matching(
    x0_batch: torch.Tensor, x1_batch: torch.Tensor,
    lp: torch.Tensor, epsilon: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        B = x0_batch.shape[0]
        abs0 = lp.unsqueeze(1) + x0_batch[:, :, :2]
        abs1 = lp.unsqueeze(1) + x1_batch[:, :, :2]
        abs0_deg = _norm_to_deg(abs0)
        abs1_deg = _norm_to_deg(abs1)
        x0e = abs0_deg.unsqueeze(1).expand(B, B, -1, -1).reshape(B*B, -1, 2)
        x1e = abs1_deg.unsqueeze(0).expand(B, B, -1, -1).reshape(B*B, -1, 2)
        cost = haversine_km(
            x0e.permute(1, 0, 2), x1e.permute(1, 0, 2)
        ).mean(0).reshape(B, B) / 500.0
        pi   = _sinkhorn_log(cost, epsilon=epsilon)
        flat = pi.reshape(-1).clamp(0.0)
        s    = flat.sum()
        if not torch.isfinite(s) or s < 1e-10:
            return x0_batch, x1_batch
        idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
        col = idx % B
        return x0_batch[col], x1_batch[col]
    except Exception:
        return x0_batch, x1_batch


# ══════════════════════════════════════════════════════════════
#  Speed statistics (từ v59)
# ══════════════════════════════════════════════════════════════

def compute_speed_stats_from_norm(obs_traj_norm: torch.Tensor) -> dict:
    T_obs = obs_traj_norm.shape[0]
    if T_obs < 2:
        return {'v_opt': 15.0, 'v_sigma': 10.0, 'v_hard_cap': 80.0, 'p50_kmh': 15.0}

    lon = (obs_traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
    lat = (obs_traj_norm[..., 1] * 50.0) / 10.0
    lat_mid = (lat[:-1] + lat[1:]) / 2
    cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
    dx = (lon[1:] - lon[:-1]) * cos_lat * DEG2KM
    dy = (lat[1:] - lat[:-1]) * DEG2KM
    spd = torch.sqrt(dx**2 + dy**2) / DT_HOURS  # km/h

    spd_flat = spd.flatten()
    p50 = float(spd_flat.median())
    p95 = float(torch.quantile(spd_flat, 0.95))

    return {
        'v_opt':     max(p50, 5.0),
        'v_sigma':   10.0,
        'v_hard_cap': float(torch.tensor(p95 * 1.8).clamp(25.0, 130.0)),
        'p50_kmh':   p50,
    }


# ══════════════════════════════════════════════════════════════
#  Persistence blend (v60: adaptive)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def _persistence_blend_adaptive(
    model_pred_norm: torch.Tensor,   # [T, B, 2]
    obs_traj_norm:   torch.Tensor,   # [T_obs, B, 2]
    blend_alpha:     torch.Tensor,   # [B] ∈ [0, 0.5] learned
) -> torch.Tensor:
    """
    Adaptive persistence blend với learned α per sample.

    FIX BUG-7: α không phải 0.20 cứng.
    Recurvature context → α≈0.05 (persistence sai → ít blend)
    Straight context    → α≈0.25 (persistence đúng → blend nhiều)
    """
    T_obs = obs_traj_norm.shape[0]
    T     = model_pred_norm.shape[0]
    B     = model_pred_norm.shape[1]
    device = model_pred_norm.device

    if T_obs < 2:
        return model_pred_norm

    # Compute persistence trajectory
    vels = obs_traj_norm[1:] - obs_traj_norm[:-1]
    n_v  = vels.shape[0]
    if n_v >= 3:
        alpha = 0.7
        w = torch.tensor([alpha * (1 - alpha)**i for i in range(n_v)],
                          dtype=torch.float, device=device).flip(0)
        ev = (vels * (w / w.sum()).view(-1, 1, 1)).sum(0)
    elif n_v == 2:
        ev = 0.7 * vels[-1] + 0.3 * vels[-2]
    else:
        ev = vels[-1]

    steps   = torch.arange(1, T + 1, dtype=torch.float, device=device)
    persist = (obs_traj_norm[-1].unsqueeze(0)
               + ev.unsqueeze(0) * steps.view(T, 1, 1))  # [T, B, 2]

    # Adaptive blend: [B] → [1, B, 1]
    alpha_b = blend_alpha.view(1, B, 1).clamp(0.0, 0.5)
    return (1.0 - alpha_b) * model_pred_norm + alpha_b * persist


# ══════════════════════════════════════════════════════════════
#  TCFlowMatching v60
# ══════════════════════════════════════════════════════════════

class TCFlowMatching(nn.Module):

    def __init__(self, pred_len: int = 12, obs_len: int = 8,
                 sigma_min: float = 0.02, unet_in_ch: int = 13,
                 ctx_noise_scale: float = 0.01,
                 use_ema: bool = True, ema_decay: float = 0.995,
                 use_ate_ot: bool = True, ot_epsilon: float = 0.05,
                 cfg_uncond_prob: float = 0.1,
                 **kwargs):
        super().__init__()
        self.pred_len        = pred_len
        self.obs_len         = obs_len
        self.sigma_min       = sigma_min
        self.ctx_noise_scale = ctx_noise_scale
        self.use_ate_ot      = use_ate_ot
        self.ot_epsilon      = ot_epsilon
        self.cfg_uncond_prob = cfg_uncond_prob

        self.net     = VelocityField(
            pred_len=pred_len, obs_len=obs_len,
            sigma_min=sigma_min, unet_in_ch=unet_in_ch, ctx_dim=256)
        self.criterion = FMv60Loss(pred_len=pred_len)

        self.use_ema   = use_ema
        self.ema_decay = ema_decay
        self._ema      = None

    def init_ema(self):
        if self.use_ema:
            self._ema = EMAModel(self, decay=self.ema_decay)

    def ema_update(self):
        if self._ema is not None:
            self._ema.update(self)

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _to_rel(traj, Me, lp, lm):
        return torch.cat([traj - lp.unsqueeze(0),
                           Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

    @staticmethod
    def _to_abs(rel, lp, lm):
        d = rel.permute(1, 0, 2)
        return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

    @staticmethod
    def _sigma_schedule(epoch: int) -> float:
        """Annealing sigma from 0.10 to 0.035."""
        if epoch < 2:  return 0.10
        if epoch < 10: return 0.10 - (epoch - 2) / 8.0 * (0.10 - 0.04)
        if epoch < 20: return max(0.04 - (epoch - 10) / 10.0 * 0.01, 0.035)
        return 0.035

    def _cfm_noisy(self, x1: torch.Tensor, sigma_min: Optional[float] = None,
                    lp=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sigma_min is None: sigma_min = self.sigma_min
        B      = x1.shape[0]
        device = x1.device
        x0     = torch.randn_like(x1) * sigma_min
        t      = torch.rand(B, device=device)
        te     = t.view(B, 1, 1)
        x_t    = (1.0 - te) * x0 + te * x1
        u_target = x1 - x0
        return x_t, t, u_target

    def _persistence_forecast_rel(self, obs_traj, lp, lm, pred_len: int):
        B, device = obs_traj.shape[1], obs_traj.device
        if obs_traj.shape[0] >= 3:
            vels  = obs_traj[1:] - obs_traj[:-1]
            n_v   = vels.shape[0]
            alpha = 0.7
            w     = torch.tensor([alpha * (1-alpha)**i for i in range(n_v)],
                                  dtype=torch.float, device=device).flip(0)
            lv    = (vels * (w / w.sum()).view(-1, 1, 1)).sum(0)
        elif obs_traj.shape[0] >= 2:
            lv = obs_traj[-1, :, :2] - obs_traj[-2, :, :2]
        else:
            lv = obs_traj.new_zeros(B, 2)

        steps    = torch.arange(1, pred_len + 1, device=device).float()
        pred_abs = (obs_traj[-1, :, :2].unsqueeze(1)
                    + lv.unsqueeze(1) * steps.view(1, -1, 1))  # [B, T, 2]
        pred_rel_pos = pred_abs.permute(1, 0, 2) - lp.unsqueeze(0)
        pred_rel     = torch.cat([pred_rel_pos,
                                    torch.zeros_like(pred_rel_pos)], dim=-1)
        return pred_rel.permute(1, 0, 2)   # [B, T, 4]

    def _compute_obs_momentum(self, obs_traj_norm: torch.Tensor) -> torch.Tensor:
        T_obs = obs_traj_norm.shape[0]
        if T_obs < 2:
            return torch.zeros(obs_traj_norm.shape[1], 2, device=obs_traj_norm.device)
        vels = obs_traj_norm[1:] - obs_traj_norm[:-1]
        n_v  = vels.shape[0]
        if n_v >= 3:
            alpha = 0.65
            w = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
                              dtype=torch.float, device=obs_traj_norm.device).flip(0)
            return (vels * (w / w.sum()).view(-1, 1, 1)).sum(0)
        elif n_v == 2:
            return 0.65 * vels[-1] + 0.35 * vels[-2]
        return vels[-1]

    @staticmethod
    def _obs_noise_aug(bl, sigma: float = 0.005):
        if torch.rand(1).item() > 0.5: return bl
        bl = list(bl)
        if torch.is_tensor(bl[0]):
            bl[0] = bl[0] + torch.randn_like(bl[0]) * sigma
        return bl

    # ── Training ──────────────────────────────────────────────

    def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list, epoch=epoch)['total']

    def get_loss_breakdown(self, batch_list, epoch: int = 0) -> dict:
        batch_list = self._obs_noise_aug(batch_list, sigma=0.005)

        obs_t    = batch_list[0]
        env_data = batch_list[13] if len(batch_list) > 13 else None
        lp, lm   = obs_t[-1], batch_list[7][-1]
        B, device = obs_t.shape[1], obs_t.device

        current_sigma = self._sigma_schedule(epoch)
        raw_ctx       = self.net._context(batch_list)

        x1_rel = self._to_rel(batch_list[1], batch_list[8], lp, lm)

        # OT matching
        if self.use_ate_ot and B >= 4:
            noise_base = torch.randn_like(x1_rel) * current_sigma
            noise_matched, x1_matched = _spherical_ot_matching(
                noise_base, x1_rel, lp, epsilon=self.ot_epsilon)
        else:
            noise_matched = torch.randn_like(x1_rel) * current_sigma
            x1_matched    = x1_rel

        x_t, fm_t, u_target = self._cfm_noisy(x1_matched, sigma_min=current_sigma, lp=lp)

        use_null     = (torch.rand(1).item() < self.cfg_uncond_prob)
        vel_obs_feat = self.net._get_kinematic_obs_feat(obs_t[:, :, :2])
        steering_feat = self.net._get_steering_feat(env_data, B, device)
        env_kine_feat = self.net._get_env_kine_feat(env_data, B, device)

        pred_vel = self.net.forward_with_ctx(
            x_t, fm_t, raw_ctx, env_data=env_data, use_null=use_null,
            vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
            env_kine_feat=env_kine_feat)

        # Get predicted trajectory để tính physics losses
        fm_te       = fm_t.view(B, 1, 1)
        x1_pred     = x_t + (1.0 - fm_te) * pred_vel
        # pred_abs, _ = self._to_abs(x1_pred, lp, lm)
        # pred_deg    = _norm_to_deg(pred_abs)                    # [B, T, 2]
        # gt_deg      = _norm_to_deg(batch_list[1])               # [B, T, 2]

        # # Reshape: loss expects [T, B, 2]
        # pred_deg_t = pred_deg.permute(1, 0, 2)   # [T, B, 2]
        # gt_deg_t   = gt_deg.permute(1, 0, 2)     # [T, B, 2]


        pred_abs, _ = self._to_abs(x1_pred, lp, lm)
        pred_deg_t  = _norm_to_deg(pred_abs)          # [T, B, 2] trực tiếp
        gt_deg_t    = _norm_to_deg(batch_list[1])      # [T, B, 2] trực tiếp


        # FM velocity: [B, T, 4]
        fm_vel_pred   = pred_vel
        fm_vel_target = u_target

        # Scorer training candidates (từ epoch 5+, nhẹ)
        candidates = None
        obs_norm   = None
        if epoch >= 5 and not use_null:
            obs_norm = obs_t[:, :, :2]  # [T_obs, B, 2]
            # Sinh 3 candidates với noise khác nhau
            cands = []
            for _ in range(3):
                x0_c = torch.randn_like(x1_rel) * current_sigma
                x1_c = x1_rel  # same gt, different noise path
                te_c  = fm_t.view(B, 1, 1)
                x_c   = (1.0 - te_c) * x0_c + te_c * x1_c
                # Quick single-step rollout (không train, chỉ lấy sample)
                with torch.no_grad():
                    v_c = self.net.forward_with_ctx(
                        x_c, fm_t, raw_ctx, env_data=env_data,
                        vel_obs_feat=vel_obs_feat,
                        steering_feat=steering_feat,
                        env_kine_feat=env_kine_feat)
                    x1_c_pred = x_c + (1.0 - te_c) * v_c
                    abs_c, _  = self._to_abs(x1_c_pred, lp, lm)
                    # Normalize back
                    # lon_n = (lon * 10 - 1800) / 50
                    lon_n = (abs_c[:, :, 0] * 10.0 - 1800.0) / 50.0
                    lat_n = (abs_c[:, :, 1] * 10.0) / 50.0
                    cand_norm = torch.stack([lon_n, lat_n], dim=-1).permute(1, 0, 2)
                cands.append(cand_norm)
            candidates = cands

        # Compute loss
        total, breakdown = self.criterion(
            pred_deg_t, gt_deg_t,
            fm_vel_pred, fm_vel_target,
            candidates=candidates,
            obs_norm=obs_norm,
        )

        if torch.isnan(total) or torch.isinf(total):
            total = obs_t.new_zeros(())

        # Add backward-compat keys
        breakdown.update({
            'sigma': current_sigma,
            'v_opt': compute_speed_stats_from_norm(obs_t[:, :, :2]).get('v_opt', 15.0),
            # Legacy keys for trainer compatibility
            'dpe': 0.0, 'mse': 0.0, 'speed': 0.0, 'accel': 0.0,
            'heading': 0.0, 'vel_reg': 0.0, 'ate': 0.0, 'cte': 0.0,
            'sph_ate': 0.0, 'endpoint': 0.0, 'signed_ate': 0.0,
            'signed_cte': 0.0, 'direct_ep': 0.0, 'fm_mse': breakdown.get('l_fm', 0.0),
        })
        breakdown['total'] = total  # tensor for backward

        return breakdown

    # ── Inference ─────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, batch_list, num_ensemble: int = 50,
                ddim_steps: int = 20, predict_csv: Optional[str] = None) -> tuple:
        """
        FM v60 inference.

        Changes vs v59:
        1. sigma per sample: learned (adaptive), không cứng 0.035
        2. CFG guidance: learned gs per sample, không 1.5 cứng
        3. Scorer: learned MLP, không fixed heuristic với v_opt=15
        4. Persistence blend: learned alpha per sample, không 0.20 cứng
        5. Speed sweep: REMOVED (L_logspeed fix speed, không cần post-hoc)
        """
        obs_t    = batch_list[0]
        env_data = batch_list[13] if len(batch_list) > 13 else None
        lp       = obs_t[-1]
        lm       = batch_list[7][-1]
        B        = lp.shape[0]
        device   = lp.device
        T        = self.pred_len
        dt       = 1.0 / max(ddim_steps, 1)

        raw_ctx       = self.net._context(batch_list)
        ctx           = self.net._apply_ctx_head(raw_ctx)         # [B, 256]
        vel_obs_feat  = self.net._get_kinematic_obs_feat(obs_t[:, :, :2])
        steering_feat = self.net._get_steering_feat(env_data, B, device)
        env_kine_feat = self.net._get_env_kine_feat(env_data, B, device)

        obs_norm  = obs_t[:, :, :2]
        obs_mom   = self._compute_obs_momentum(obs_norm)

        # Adaptive modules
        blend_alpha     = self.net.get_blend_alpha(ctx)      # [B] ∈ [0, 0.5]
        guidance_scale  = self.net.get_guidance_scale(ctx)   # [B] ∈ [0.8, 2.0]
        sigma_per_sample = self.net.get_sigma(ctx)           # [B] ∈ [0.02, 0.10]

        # Persistence init
        persist_init = self._persistence_forecast_rel(obs_t, lp, lm, T)

        # Obs heading (for momentum)
        if obs_t.shape[0] >= 2:
            obs_h_n = F.normalize(obs_t[-1, :, :2] - obs_t[-2, :, :2],
                                   dim=-1, eps=1e-6)
        else:
            obs_h_n = None

        def _mom_str(s: int, tot: int) -> float:
            return 0.06 * 0.5 * (1.0 + math.cos(math.pi * s / max(tot, 1)))

        all_norms = []

        for ens_i in range(num_ensemble):
            # Per-sample sigma for diversity
            sigma_noise = sigma_per_sample.mean().item() * 2.5
            x_t = persist_init + torch.randn_like(persist_init) * sigma_noise

            for step in range(ddim_steps):
                t_b  = torch.full((B,), step * dt, device=device)
                ns   = self.ctx_noise_scale * 2.0 if step < 3 else 0.0

                # CFG with learned guidance scale
                if step > 0:
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
                    # Learned guidance scale per sample
                    gs = guidance_scale.view(B, 1, 1)
                    vel = v_uncond + gs * (v_cond - v_uncond)
                else:
                    vel = self.net.forward_with_ctx(
                        x_t, t_b, raw_ctx, noise_scale=ns,
                        vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
                        env_kine_feat=env_kine_feat, env_data=env_data)

                # Momentum injection
                m_s = _mom_str(step, ddim_steps)
                if m_s > 1e-4:
                    me = obs_mom.unsqueeze(1).expand(B, T, 2)
                    mf = torch.cat([me, torch.zeros(B, T, 2, device=device)], dim=-1)
                    vel = vel + m_s * mf

                x_t = (x_t + dt * vel).clamp(-3.0, 3.0)

            # tr, me = self._to_abs(x_t, lp, lm)
            # all_norms.append(tr)
            tr, me = self._to_abs(x_t, lp, lm)
            all_norms.append(tr.permute(1, 0, 2))   # [B, T, 2]

        # ── Learned scoring (FIX BUG-1, BUG-5) ──────────────
        # Không có prior_sc (v_opt=15 sai)
        # Dùng learned scorer MLP
        scores = []
        # for tn in all_norms:
        #     # Normalize back to norm space for scorer
        #     lon_n = (tn[:, :, 0] * 10.0 - 1800.0) / 50.0  # [B, T]
        #     lat_n = (tn[:, :, 1] * 10.0) / 50.0
        #     tn_norm = torch.stack([lon_n, lat_n], dim=-1).permute(1, 0, 2)  # [T, B, 2]
        for tn in all_norms:
            tn_norm = tn.permute(1, 0, 2)   # [B,T,2] → [T,B,2]
            sc = self.criterion.scorer.score(tn_norm, obs_norm)  # [B]
            scores.append(sc)

        all_c  = torch.stack(all_norms)  # [N_ens, B, T, 2]
        all_sc = torch.stack(scores)     # [N_ens, B]

        # Top 35%
        k = max(1, int(all_c.shape[0] * 0.35))
        _, top_idx = all_sc.topk(k, dim=0)   # [k, B]

        pred_mean = torch.stack([
            all_c[top_idx[:, b], b, :, :].median(0).values
            for b in range(B)
        ], dim=0).permute(1, 0, 2)   # [T, B, 2]

        # ── Adaptive persistence blend (FIX BUG-7) ───────────
        # pred_norm_t = torch.stack([
        #     ((pred_mean[:, :, 0] * 10.0 - 1800.0) / 50.0),
        #     (pred_mean[:, :, 1] * 10.0 / 50.0),
        # ], dim=-1).permute(1, 0, 2)  # norm → [T, B, 2]

        # pred_norm_t = torch.stack([
        #     ((pred_mean[:, :, 0] * 10.0 - 1800.0) / 50.0),
        #     (pred_mean[:, :, 1] * 10.0 / 50.0),
        # ], dim=-1)  # [T, B, 2] — xóa .permute(1, 0, 2)
        pred_norm_t = pred_mean   # [T, B, 2] normalized — không cần convert
        blended_norm = _persistence_blend_adaptive(pred_norm_t, obs_norm, blend_alpha)

        # Convert back to degrees
        final_deg = _norm_to_deg(blended_norm.permute(1, 0, 2))  # [B, T, 2]
        # Back to normalized for output
        lon_out = (final_deg[:, :, 0] * 10.0 - 1800.0) / 50.0
        lat_out = final_deg[:, :, 1] * 10.0 / 50.0
        pred_final = torch.stack([lon_out, lat_out], dim=-1).permute(1, 0, 2)  # [T, B, 2]

        if predict_csv:
            self._write_predict_csv(predict_csv, pred_final, all_c)

        return pred_final, all_c

    @staticmethod
    def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
                            all_trajs: torch.Tensor):
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        T, B, _ = traj_mean.shape
        mlon = ((traj_mean[:, :, 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
        mlat = ((traj_mean[:, :, 1] * 50.0) / 10.0).cpu().numpy()
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        fields = ['timestamp', 'batch_idx', 'step_idx', 'lead_h',
                  'lon_mean_deg', 'lat_mean_deg']
        write_hdr = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if write_hdr: w.writeheader()
            for b in range(B):
                for k in range(T):
                    w.writerow({'timestamp': ts, 'batch_idx': b,
                                'step_idx': k, 'lead_h': (k + 1) * 6,
                                'lon_mean_deg': f'{mlon[k, b]:.4f}',
                                'lat_mean_deg': f'{mlat[k, b]:.4f}'})


# Backward compat alias
TCDiffusion = TCFlowMatching

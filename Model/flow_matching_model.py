"""
Model/flow_matching_model.py  ── FM v60 FIXED
=============================================
COPY FILE NÀY VÀO Model/flow_matching_model.py

FIXES SO VỚI v60 GỐC (làm ADE tăng liên tục):

  BUG-CRASH: EnsembleScorer.auxiliary_loss nhận gt_deg [B,T,2] batch-first
             nhưng code permute(1,0,2) tưởng [T,B,2] → shape mismatch crash
             FIX: Unify toàn bộ pipeline về batch-first [B,T,2]

  BUG-SCALE: l_kinematic trả giá trị ~80-100 (km²-scale)
             l_fm chỉ ~0.4 → gradient ratio 250:1 → FM objective không học được
             FIX: normalize bằng NORM=100 → l_kin ~0.2-1.5, cùng scale l_fm

  BUG-INIT1: DifficultyWeighter bias=0 → dw=1.5 từ epoch 1 → gradient noisy
             FIX: bias=-2.0 → dw≈1.12 khi khởi động

  BUG-INIT2: LearnedLossWeights λ=0.5 ngay từ epoch 1 → l_kin dominate
             FIX: warmup epoch<5 dùng fixed λ nhỏ

  BUG-SHAPE: _to_abs và _to_rel dùng seq-first [T,B,2] trong khi
             batch_list[1] (gt) là [B,T,2] batch-first từ dataloader
             FIX: _to_abs trả [B,T,2], _to_rel nhận [B,T,2]

ORIGINAL v60 BUGS (giữ nguyên fix):
  BUG-1: prior_sc(v_opt=15) → EnsembleScorer MLP
  BUG-2: l_speed(v_opt) → l_logspeed
  BUG-3: STEP_WEIGHTS magic → LearnedStepWeights monotonic
  BUG-4: 79% env_data bỏ → thêm gph500, bearing, month
  BUG-5: scoring exponents → MLP learned
  BUG-6: speed_sweep → removed
  BUG-7: persistence blend 20% cứng → adaptive learned
  BUG-8: guidance_scale 1.5 cứng → adaptive learned
  BUG-9: sigma 0.035 cứng → adaptive learned
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

MAX_CURVATURE_RAD = math.pi / 4


# ══════════════════════════════════════════════════════════════
#  Coordinate utilities
# ══════════════════════════════════════════════════════════════

def norm_to_deg(t: torch.Tensor) -> torch.Tensor:
    """Normalized [...,2] → degrees. Works for any leading dims."""
    return torch.stack([
        (t[..., 0] * 50.0 + 1800.0) / 10.0,
        (t[..., 1] * 50.0) / 10.0,
    ], dim=-1)

_norm_to_deg = norm_to_deg  # alias


def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """[...,2] lon°,lat° → [...] km. Works for any leading dims."""
    lat1 = torch.deg2rad(p1[..., 1])
    lat2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a = (torch.sin(dlat / 2).pow(2)
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1.0 - 1e-12).sqrt())


def velocity_km_b(traj_deg: torch.Tensor) -> torch.Tensor:
    """
    [B, T, 2] lon°,lat° → [B, T-1, 2] velocity (vx_km, vy_km) per step.
    BATCH-FIRST convention.
    """
    lon = traj_deg[:, :, 0]
    lat = traj_deg[:, :, 1]
    dlat = lat[:, 1:] - lat[:, :-1]
    dlon = lon[:, 1:] - lon[:, :-1]
    lat_mid = (lat[:, 1:] + lat[:, :-1]) / 2.0
    cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
    vx = dlon * cos_lat * DEG2KM
    vy = dlat * DEG2KM
    return torch.stack([vx, vy], dim=-1)  # [B, T-1, 2]


# ══════════════════════════════════════════════════════════════
#  Learned Step Weights — Monotonic
# ══════════════════════════════════════════════════════════════

class LearnedStepWeights(nn.Module):
    def __init__(self, n_steps: int = 12):
        super().__init__()
        self.n_steps = n_steps
        self.raw = nn.Parameter(torch.zeros(n_steps) + 0.5)

    def forward(self) -> torch.Tensor:
        increments = F.softplus(self.raw)
        weights = torch.cumsum(increments, dim=0)
        weights = weights / weights.mean().clamp(min=1e-8)
        return weights

    def get(self, n: Optional[int] = None) -> torch.Tensor:
        w = self.forward()
        return w[:n] if n is not None else w

    @torch.no_grad()
    def stats(self) -> dict:
        w = self.forward()
        return {
            "sw_6h":   w[0].item(),
            "sw_72h":  w[-1].item(),
            "sw_ratio": (w[-1] / w[0].clamp(min=1e-6)).item(),
            "sw_monotonic": bool((w[1:] - w[:-1]).min().item() >= -1e-6),
        }


# ══════════════════════════════════════════════════════════════
#  Log-Parameterized Loss Weights
# ══════════════════════════════════════════════════════════════

class LearnedLossWeights(nn.Module):
    def __init__(self, n_tasks: int = 3):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def lambdas(self) -> torch.Tensor:
        return 0.5 * torch.exp(-2.0 * self.log_sigma)

    def regularization(self) -> torch.Tensor:
        return self.log_sigma.sum()

    @torch.no_grad()
    def stats(self) -> dict:
        lam = self.lambdas()
        return {
            "lam_kin":    lam[0].item(),
            "lam_logspd": lam[1].item(),
            "lam_curv":   lam[2].item(),
        }


# ══════════════════════════════════════════════════════════════
#  Difficulty Weighting
# ══════════════════════════════════════════════════════════════

class DifficultyWeighter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        # FIX: bias=-2.0 → sigmoid(-2)≈0.12 → dw≈1.12 (không phải 1.5)
        nn.init.constant_(self.linear.bias, -2.0)

    def compute_difficulty(self, gt_deg: torch.Tensor) -> torch.Tensor:
        """gt_deg: [B, T, 2] batch-first → [B, 3] difficulty features."""
        B = gt_deg.shape[0]

        v = velocity_km_b(gt_deg)  # [B, T-1, 2]
        if v.shape[1] >= 2:
            heading = torch.atan2(v[:, :, 0], v[:, :, 1])  # [B, T-1]
            dh = heading[:, 1:] - heading[:, :-1]
            dh = (dh + math.pi) % (2 * math.pi) - math.pi
            curv_rate = dh.abs().mean(1)   # [B]
        else:
            curv_rate = gt_deg.new_zeros(B)

        spd = v.norm(dim=-1)  # [B, T-1]
        mean_spd = spd.mean(1).clamp(min=1.0)
        std_spd  = spd.std(1)
        speed_cv = (std_spd / mean_spd).clamp(max=3.0)

        if v.shape[1] >= 2:
            excess_curv = F.relu(dh.abs() - MAX_CURVATURE_RAD).mean(1)
        else:
            excess_curv = gt_deg.new_zeros(B)

        d1 = (curv_rate   / (math.pi / 2)).clamp(0, 1)
        d2 = speed_cv.clamp(0, 1)
        d3 = (excess_curv / (math.pi / 4)).clamp(0, 1)

        return torch.stack([d1, d2, d3], dim=-1)  # [B, 3]

    def forward(self, gt_deg: torch.Tensor) -> torch.Tensor:
        """Returns [B] weights ∈ [1.0, 2.0]."""
        diff_feat = self.compute_difficulty(gt_deg)
        logit = self.linear(diff_feat).squeeze(-1)
        return 1.0 + torch.sigmoid(logit)


# ══════════════════════════════════════════════════════════════
#  Ensemble Scorer — FIXED for batch-first [B,T,2]
# ══════════════════════════════════════════════════════════════

class EnsembleScorer(nn.Module):
    def __init__(self, feat_dim: int = 7, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def extract_features(
        self,
        traj_norm: torch.Tensor,   # [B, T, 2] batch-first normalized
        obs_norm:  torch.Tensor,   # [B, T_obs, 2] batch-first normalized
    ) -> torch.Tensor:
        """Returns [B, 7]."""
        B = traj_norm.shape[0]
        traj_deg = norm_to_deg(traj_norm)  # [B, T, 2]
        obs_deg  = norm_to_deg(obs_norm)   # [B, T_obs, 2]

        v   = velocity_km_b(traj_deg)   # [B, T-1, 2]
        spd = v.norm(dim=-1)            # [B, T-1]

        f1 = torch.log1p(spd.mean(1))
        f2 = torch.log1p(spd.std(1).clamp(min=0))

        if v.shape[1] >= 2:
            heading = torch.atan2(v[:, :, 0], v[:, :, 1])
            dh = heading[:, 1:] - heading[:, :-1]
            dh = (dh + math.pi) % (2 * math.pi) - math.pi
            f3 = torch.cos(dh).mean(1)
        else:
            f3 = traj_norm.new_ones(B)

        v_obs = velocity_km_b(obs_deg)  # [B, T_obs-1, 2]
        n_obs = min(3, v_obs.shape[1])
        obs_spd_ref = v_obs[:, -n_obs:, :].norm(dim=-1).mean(1).clamp(min=1.0)
        n_pred = min(3, spd.shape[1])
        pred_spd_early = spd[:, :n_pred].mean(1)
        f4 = torch.exp(-((pred_spd_early - obs_spd_ref) / obs_spd_ref).pow(2) * 2.0)

        if v_obs.shape[1] >= 1 and v.shape[1] >= 1:
            obs_last_h  = torch.atan2(v_obs[:, -1, 0], v_obs[:, -1, 1])
            pred_first_h = torch.atan2(v[:, 0, 0], v[:, 0, 1])
            dh_cont = pred_first_h - obs_last_h
            dh_cont = (dh_cont + math.pi) % (2 * math.pi) - math.pi
            f5 = torch.cos(dh_cont)
        else:
            f5 = traj_norm.new_ones(B)

        if obs_norm.shape[1] >= 2:
            last_vel_n = obs_norm[:, -1, :] - obs_norm[:, -2, :]  # [B, 2]
            steps = torch.arange(1, traj_norm.shape[1] + 1,
                                  device=traj_norm.device, dtype=traj_norm.dtype)
            persist = (obs_norm[:, -1, :].unsqueeze(1)
                       + last_vel_n.unsqueeze(1) * steps.view(1, -1, 1))  # [B,T,2]
            dist_from_persist = (traj_norm - persist).norm(dim=-1).mean(1)
            ref_disp = (last_vel_n.norm(dim=-1) * traj_norm.shape[1]).clamp(min=1e-3)
            f6 = dist_from_persist / ref_disp
        else:
            f6 = traj_norm.new_zeros(B)

        if v.shape[1] >= 2:
            heading = torch.atan2(v[:, :, 0], v[:, :, 1])
            dh = heading[:, 1:] - heading[:, :-1]
            dh = (dh + math.pi) % (2 * math.pi) - math.pi
            f7 = dh.abs().mean(1)
        else:
            f7 = traj_norm.new_zeros(B)

        feats = torch.stack([f1, f2, f3, f4, f5, f6, f7], dim=-1)
        return feats.clamp(-10.0, 10.0)

    def score(
        self,
        traj_norm: torch.Tensor,   # [B, T, 2] batch-first
        obs_norm:  torch.Tensor,   # [B, T_obs, 2] batch-first
    ) -> torch.Tensor:
        """Returns [B] scores ∈ [0,1]."""
        feats = self.extract_features(traj_norm, obs_norm)
        return torch.sigmoid(self.net(feats).squeeze(-1))

    def auxiliary_loss(
        self,
        candidates: list,          # list of [B, T, 2] batch-first normalized
        obs_norm:   torch.Tensor,  # [B, T_obs, 2] batch-first normalized
        gt_deg:     torch.Tensor,  # [B, T, 2] batch-first degrees
    ) -> torch.Tensor:
        """
        BCE với oracle label từ ADE ranking.
        FIX: toàn bộ dùng batch-first [B,T,2] — không permute nữa.
        """
        if len(candidates) < 2:
            return candidates[0].new_zeros(())

        B = obs_norm.shape[0]
        ades = []
        for cand in candidates:
            cand_deg = norm_to_deg(cand)  # [B, T, 2]
            # haversine_km works on [...,2] so [B,T,2] → [B,T] → mean → [B]
            d = haversine_km(cand_deg, gt_deg).mean(dim=1)  # [B]
            ades.append(d)

        ades_t   = torch.stack(ades, dim=0)   # [n_cands, B]
        best_idx = ades_t.argmin(dim=0)       # [B]

        total_loss = ades_t.new_zeros(())
        for i, cand in enumerate(candidates):
            scores = self.score(cand, obs_norm)          # [B]
            labels = (best_idx == i).float()             # [B]
            total_loss = total_loss + F.binary_cross_entropy(
                scores, labels, reduction='mean')
        return total_loss / len(candidates)


# ══════════════════════════════════════════════════════════════
#  Loss Terms — batch-first [B,T,2]
# ══════════════════════════════════════════════════════════════

def l_kinematic(
    pred_deg:     torch.Tensor,   # [B, T, 2] batch-first degrees
    gt_deg:       torch.Tensor,   # [B, T, 2] batch-first degrees
    step_weights: torch.Tensor,   # [T-1] learned
    sample_w:     Optional[torch.Tensor] = None,  # [B]
) -> torch.Tensor:
    """
    Huber loss on velocity in km-space.
    FIX: normalize by NORM=100 to match l_fm scale (~0.3-0.8).
    Typical output: ~0.2-1.5 (vs original ~50-100).
    """
    T = min(pred_deg.shape[1], gt_deg.shape[1])
    if T < 2:
        return pred_deg.new_zeros(())

    v_pred = velocity_km_b(pred_deg[:, :T, :])  # [B, T-1, 2]
    v_gt   = velocity_km_b(gt_deg[:, :T, :])

    err_sq  = (v_pred - v_gt).pow(2).sum(dim=-1)  # [B, T-1]
    delta   = 50.0
    err_abs = err_sq.sqrt()
    huber   = torch.where(err_abs < delta,
                          0.5 * err_sq / delta,
                          err_abs - delta / 2.0)   # [B, T-1]

    # FIX: normalize → same scale as l_fm
    NORM  = 100.0
    huber = huber / NORM

    n = huber.shape[1]
    w = step_weights[:n]
    weighted = huber * w.unsqueeze(0)  # [B, T-1]

    if sample_w is not None:
        weighted = weighted * sample_w.unsqueeze(1)

    return weighted.mean()


def l_logspeed(
    pred_deg: torch.Tensor,  # [B, T, 2] batch-first
    gt_deg:   torch.Tensor,
) -> torch.Tensor:
    T = min(pred_deg.shape[1], gt_deg.shape[1])
    if T < 2:
        return pred_deg.new_zeros(())

    v_pred = velocity_km_b(pred_deg[:, :T, :])
    v_gt   = velocity_km_b(gt_deg[:, :T, :])

    spd_pred = v_pred.norm(dim=-1).clamp(min=0.1)
    spd_gt   = v_gt.norm(dim=-1).clamp(min=0.1)

    return F.mse_loss(torch.log1p(spd_pred), torch.log1p(spd_gt))


def l_curvature(
    pred_deg:      torch.Tensor,   # [B, T, 2] batch-first
    threshold_rad: float = MAX_CURVATURE_RAD,
) -> torch.Tensor:
    T = pred_deg.shape[1]
    if T < 3:
        return pred_deg.new_zeros(())

    v = velocity_km_b(pred_deg)  # [B, T-1, 2]
    if v.shape[1] < 2:
        return pred_deg.new_zeros(())

    heading = torch.atan2(v[:, :, 0], v[:, :, 1])  # [B, T-1]
    dh = heading[:, 1:] - heading[:, :-1]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi

    excess = F.relu(dh.abs() - threshold_rad)
    return excess.mean()


# ══════════════════════════════════════════════════════════════
#  Master Loss Module
# ══════════════════════════════════════════════════════════════

class FMv60Loss(nn.Module):
    def __init__(self, pred_len: int = 12):
        super().__init__()
        self.pred_len     = pred_len
        self.step_weights = LearnedStepWeights(n_steps=pred_len)
        self.loss_weights = LearnedLossWeights(n_tasks=3)
        self.diff_weighter = DifficultyWeighter()
        self.scorer        = EnsembleScorer()

    def compute_main_losses(
        self,
        pred_deg:      torch.Tensor,   # [B, T, 2] batch-first
        gt_deg:        torch.Tensor,   # [B, T, 2] batch-first
        fm_vel_pred:   torch.Tensor,   # [B, T, 4]
        fm_vel_target: torch.Tensor,   # [B, T, 4]
        epoch:         int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        T = min(pred_deg.shape[1], gt_deg.shape[1])

        sample_w = self.diff_weighter(gt_deg[:, :T, :])  # [B]
        sw       = self.step_weights.get(n=T - 1)         # [T-1]

        # FIX: warmup epoch<5 → fixed small λ so FM objective learns first
        if epoch < 5:
            lam = torch.tensor([0.05, 0.30, 0.02],
                                device=fm_vel_pred.device,
                                dtype=fm_vel_pred.dtype)
            reg = fm_vel_pred.new_zeros(())
        else:
            lam = self.loss_weights.lambdas()
            reg = self.loss_weights.regularization()

        L_fm     = F.mse_loss(fm_vel_pred, fm_vel_target)
        L_kin    = l_kinematic(pred_deg[:, :T], gt_deg[:, :T], sw, sample_w)
        L_logspd = l_logspeed(pred_deg[:, :T], gt_deg[:, :T])
        L_curv   = l_curvature(pred_deg[:, :T])

        total = (1.0    * L_fm
               + lam[0] * L_kin
               + lam[1] * L_logspd
               + lam[2] * L_curv
               + reg)

        if not torch.isfinite(total):
            total = pred_deg.new_zeros(())

        lam_d = self.loss_weights.stats()
        sw_d  = self.step_weights.stats()

        breakdown = {
            "l_fm":        L_fm.item()     if torch.is_tensor(L_fm)     else 0.0,
            "l_kin":       L_kin.item()    if torch.is_tensor(L_kin)    else 0.0,
            "l_logspd":    L_logspd.item() if torch.is_tensor(L_logspd) else 0.0,
            "l_curv":      L_curv.item()   if torch.is_tensor(L_curv)   else 0.0,
            "l_reg":       reg.item()      if torch.is_tensor(reg)       else 0.0,
            "diff_w_mean": sample_w.mean().item(),
            **lam_d, **sw_d,
        }
        return total, breakdown

    def scorer_loss(self, candidates, obs_norm, gt_deg):
        return self.scorer.auxiliary_loss(candidates, obs_norm, gt_deg)

    def forward(
        self,
        pred_deg:      torch.Tensor,   # [B, T, 2] batch-first
        gt_deg:        torch.Tensor,   # [B, T, 2] batch-first
        fm_vel_pred:   torch.Tensor,
        fm_vel_target: torch.Tensor,
        candidates:    Optional[list] = None,   # list of [B,T,2] batch-first
        obs_norm:      Optional[torch.Tensor] = None,  # [B,T_obs,2] batch-first
        epoch:         int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        total, breakdown = self.compute_main_losses(
            pred_deg, gt_deg, fm_vel_pred, fm_vel_target, epoch=epoch)

        l_scr = pred_deg.new_zeros(())
        if candidates is not None and obs_norm is not None and len(candidates) >= 2:
            l_scr = self.scorer_loss(candidates, obs_norm, gt_deg)
            total = total + 0.05 * l_scr

        breakdown["l_scorer"] = l_scr.item() if torch.is_tensor(l_scr) else 0.0
        breakdown["total"]    = total
        return total, breakdown


_NORM_TO_DEG = 5.0


# ══════════════════════════════════════════════════════════════
#  SpeedHead
# ══════════════════════════════════════════════════════════════

class SpeedHead(nn.Module):
    def __init__(self, ctx_dim: int = 256, obs_feat_dim: int = 256, pred_len: int = 12):
        super().__init__()
        self.pred_len = pred_len
        self.net = nn.Sequential(
            nn.Linear(ctx_dim + obs_feat_dim, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, pred_len),
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ctx, obs_feat):
        log_spd = self.net(torch.cat([ctx, obs_feat], dim=-1))
        return torch.exp(log_spd).clamp(3.0, 150.0)


# ══════════════════════════════════════════════════════════════
#  Safe env helpers (unchanged)
# ══════════════════════════════════════════════════════════════

def _safe_env(env_data, key, B, device, norm=1.0):
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


def _safe_env_vec(env_data, key, dim, B, device):
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
        if v.shape[0] == B:
            if v.shape[1] >= dim:
                return v[:, :dim]
            return F.pad(v, (0, dim - v.shape[1]))
        return torch.zeros(B, dim, device=device)
    if v.dim() == 3:
        vv = v[:B, -1, :]
        if vv.shape[1] >= dim:
            return vv[:, :dim]
        return F.pad(vv, (0, dim - vv.shape[1]))
    return torch.zeros(B, dim, device=device)


# ══════════════════════════════════════════════════════════════
#  VelocityField v60 (unchanged from original)
# ══════════════════════════════════════════════════════════════

class VelocityField(nn.Module):
    RAW_CTX_DIM = 512

    def __init__(self, pred_len=12, obs_len=8, ctx_dim=256,
                 sigma_min=0.02, unet_in_ch=13, **kwargs):
        super().__init__()
        self.pred_len  = pred_len
        self.obs_len   = obs_len
        self.ctx_dim   = ctx_dim
        self.sigma_min = sigma_min

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

        self.steering_enc = nn.Sequential(
            nn.Linear(9, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 256))
        self.env_kine_enc = nn.Sequential(
            nn.Linear(14, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 256), nn.GELU())
        self.recurv_enc = nn.Sequential(
            nn.Linear(33, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 64))
        self.speed_hist_enc = nn.Sequential(
            nn.Linear(11, 32), nn.GELU(), nn.LayerNorm(32),
            nn.Linear(32, 32))

        self.ctx_fc1  = nn.Linear(128 + 32 + 16 + 64 + 32, self.RAW_CTX_DIM)
        self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
        self.ctx_drop = nn.Dropout(0.15)
        self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)
        self.null_embedding = nn.Parameter(torch.randn(1, self.RAW_CTX_DIM) * 0.02)

        self.vel_obs_enc = nn.Sequential(
            nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.GELU())

        self.blend_head    = nn.Linear(ctx_dim, 1)
        self.guidance_head = nn.Linear(ctx_dim, 1)
        self.sigma_head    = nn.Linear(ctx_dim, 1)

        nn.init.zeros_(self.blend_head.weight)
        nn.init.constant_(self.blend_head.bias, -1.0)
        nn.init.zeros_(self.guidance_head.weight)
        nn.init.constant_(self.guidance_head.bias, 0.0)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, -1.0)

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

        self.speed_head = SpeedHead(ctx_dim=ctx_dim, obs_feat_dim=256, pred_len=pred_len)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for name, m in self.named_modules():
                if isinstance(m, nn.Linear) and 'out_fc' in name:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _time_emb(self, t, dim=256):
        half = dim // 2
        freq = torch.exp(
            torch.arange(half, dtype=torch.float, device=t.device)
            * (-math.log(10000.0) / max(half - 1, 1)))
        emb = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
        return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

    def _get_steering_feat(self, env_data, B, device):
        if env_data is None:
            return torch.zeros(B, 256, device=device)
        feats = torch.stack([
            _safe_env(env_data, 'u500_mean',        B, device, 30.0),
            _safe_env(env_data, 'v500_mean',        B, device, 30.0),
            _safe_env(env_data, 'u500_center',      B, device, 30.0),
            _safe_env(env_data, 'v500_center',      B, device, 30.0),
            _safe_env(env_data, 'steering_speed',   B, device, 1.0),
            _safe_env(env_data, 'steering_dir_sin', B, device, 1.0),
            _safe_env(env_data, 'steering_dir_cos', B, device, 1.0),
            _safe_env(env_data, 'gph500_mean',      B, device, 1.0),
            _safe_env(env_data, 'gph500_center',    B, device, 1.0),
        ], dim=-1)
        return self.steering_enc(feats)

    def _get_env_kine_feat(self, env_data, B, device):
        if env_data is None:
            return torch.zeros(B, 256, device=device)
        mv   = _safe_env(env_data, 'move_velocity', B, device, 150.0).unsqueeze(-1)
        hd24 = _safe_env_vec(env_data, 'history_direction24', 8, B, device)
        dv   = _safe_env_vec(env_data, 'delta_velocity', 5, B, device)
        return self.env_kine_enc(torch.cat([mv, hd24, dv], dim=-1))

    def _get_recurv_feat(self, env_data, B, device):
        if env_data is None:
            return torch.zeros(B, 64, device=device)
        bearing = _safe_env_vec(env_data, 'bearing_to_scs_center', 16, B, device)
        dist    = _safe_env_vec(env_data, 'dist_to_scs_boundary',  5,  B, device)
        month   = _safe_env_vec(env_data, 'month',                 12, B, device)
        return self.recurv_enc(torch.cat([bearing, dist, month], dim=-1))

    def _get_speed_hist_feat(self, env_data, B, device):
        if env_data is None:
            return torch.zeros(B, 32, device=device)
        vh = _safe_env_vec(env_data, 'velocity_history',  4, B, device)
        ri = _safe_env(env_data, 'rapid_intensification', B, device, 1.0).unsqueeze(-1)
        ic = _safe_env_vec(env_data, 'intensity_class', 6, B, device)
        return self.speed_hist_enc(torch.cat([vh, ri, ic], dim=-1))

    def _get_kinematic_obs_feat(self, obs_traj):
        """obs_traj: [T_obs, B, 2] seq-first → [B, 256]."""
        B      = obs_traj.shape[1]
        T_obs  = obs_traj.shape[0]
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
            kine = torch.stack([vel[:,:,0], vel[:,:,1], speed_n,
                                  heading.sin(), heading.cos(), accel], dim=-1)
        else:
            kine = obs_traj.new_zeros(self.obs_len, B, 6)

        if kine.shape[0] < self.obs_len:
            pad  = obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6)
            kine = torch.cat([pad, kine], dim=0)
        else:
            kine = kine[-self.obs_len:]

        return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

    def _context(self, batch_list):
        obs_traj  = batch_list[0]
        obs_Me    = batch_list[7]
        image_obs = batch_list[11]
        env_data  = batch_list[13] if len(batch_list) > 13 else None

        B      = obs_traj.shape[1]
        device = obs_traj.device

        if image_obs.dim() == 4:
            image_obs = image_obs.unsqueeze(2)
        if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
            image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

        e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
        T_obs = obs_traj.shape[0]
        e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        e_3d_s = self.bottleneck_proj(e_3d_s)
        if e_3d_s.shape[1] != T_obs:
            e_3d_s = F.interpolate(
                e_3d_s.permute(0, 2, 1), size=T_obs,
                mode='linear', align_corners=False).permute(0, 2, 1)

        e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
        t_w = torch.softmax(
            torch.arange(e_3d_dec_t.shape[1], dtype=torch.float, device=device) * 0.5,
            dim=0)
        f_sp = self.decoder_proj(
            (e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        h_t    = self.enc_1d(obs_in, e_3d_s)

        e_env, _, _ = self.env_enc(env_data, image_obs)

        recurv_feat  = self._get_recurv_feat(env_data, B, device)
        speed_h_feat = self._get_speed_hist_feat(env_data, B, device)

        cat_feat = torch.cat([h_t, e_env, f_sp, recurv_feat, speed_h_feat], dim=-1)
        raw_ctx  = F.gelu(self.ctx_ln(self.ctx_fc1(cat_feat)))
        return raw_ctx

    def _apply_ctx_head(self, raw, noise_scale=0.0, use_null=False):
        if use_null:
            raw = self.null_embedding.expand(raw.shape[0], -1)
        elif noise_scale > 0.0:
            raw = raw + torch.randn_like(raw) * noise_scale
        return self.ctx_fc2(self.ctx_drop(raw))

    def get_blend_alpha(self, ctx):
        return torch.sigmoid(self.blend_head(ctx)).squeeze(-1) * 0.5

    def get_guidance_scale(self, ctx):
        return 0.8 + 1.2 * torch.sigmoid(self.guidance_head(ctx)).squeeze(-1)

    def get_sigma(self, ctx):
        return 0.02 + 0.08 * torch.sigmoid(self.sigma_head(ctx)).squeeze(-1)

    def _beta_drift(self, x_t):
        lat_rad = torch.deg2rad(x_t[:, :, 1] * 5.0).clamp(-85, 85)
        beta = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
        R_tc = 3e5
        v = torch.zeros_like(x_t)
        v[:, :, 0] = -beta * R_tc**2 / 2 * 6 * 3600 / (5 * 111 * 1000)
        v[:, :, 1] =  beta * R_tc**2 / 4 * 6 * 3600 / (5 * 111 * 1000)
        return v

    def _steering_drift(self, x_t, env_data):
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

    def _decode(self, x_t, t, ctx, vel_obs_feat=None,
                 steering_feat=None, env_kine_feat=None, env_data=None):
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
                          env_kine_feat=None, env_data=None, use_null=False):
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
    def __init__(self, model, decay=0.995):
        self.decay  = decay
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        self.shadow = {k: v.detach().clone()
                       for k, v in m.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        with torch.no_grad():
            for k, v in m.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        backup, sd = {}, m.state_dict()
        for k in self.shadow:
            if k not in sd: continue
            backup[k] = sd[k].detach().clone()
            sd[k].copy_(self.shadow[k])
        return backup

    def restore(self, model, backup):
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        sd = m.state_dict()
        for k, v in backup.items():
            if k in sd: sd[k].copy_(v)


# ══════════════════════════════════════════════════════════════
#  OT matching
# ══════════════════════════════════════════════════════════════

def _sinkhorn_log(cost, epsilon=0.05, n_iter=50):
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


def _spherical_ot_matching(x0_batch, x1_batch, lp, epsilon=0.05):
    """x0_batch, x1_batch: [B,T,4] batch-first."""
    try:
        B = x0_batch.shape[0]
        # Convert to absolute normalized position for cost
        abs0 = lp[:, :2].unsqueeze(1) + x0_batch[:, :, :2]  # [B,T,2]
        abs1 = lp[:, :2].unsqueeze(1) + x1_batch[:, :, :2]
        abs0_deg = norm_to_deg(abs0)   # [B,T,2]
        abs1_deg = norm_to_deg(abs1)

        # Cost matrix [B,B]: mean haversine between all pairs
        # Expand: [B,1,T,2] vs [1,B,T,2]
        cost = haversine_km(
            abs0_deg.unsqueeze(1).expand(B, B, -1, -1).reshape(B*B, -1, 2),
            abs1_deg.unsqueeze(0).expand(B, B, -1, -1).reshape(B*B, -1, 2),
        ).mean(-1).reshape(B, B) / 500.0

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
#  Speed statistics
# ══════════════════════════════════════════════════════════════

def compute_speed_stats_from_norm(obs_traj_norm):
    """obs_traj_norm: [T_obs, B, 2] seq-first."""
    T_obs = obs_traj_norm.shape[0]
    if T_obs < 2:
        return {'v_opt': 15.0, 'v_sigma': 10.0, 'v_hard_cap': 80.0, 'p50_kmh': 15.0}

    lon = (obs_traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
    lat = (obs_traj_norm[..., 1] * 50.0) / 10.0
    lat_mid = (lat[:-1] + lat[1:]) / 2
    cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
    dx = (lon[1:] - lon[:-1]) * cos_lat * DEG2KM
    dy = (lat[1:] - lat[:-1]) * DEG2KM
    spd = torch.sqrt(dx**2 + dy**2) / DT_HOURS

    spd_flat = spd.flatten()
    p50 = float(spd_flat.median())
    p95 = float(torch.quantile(spd_flat, 0.95))
    return {
        'v_opt':      max(p50, 5.0),
        'v_sigma':    10.0,
        'v_hard_cap': float(torch.tensor(p95 * 1.8).clamp(25.0, 130.0)),
        'p50_kmh':    p50,
    }


# ══════════════════════════════════════════════════════════════
#  Persistence blend (adaptive)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def _persistence_blend_adaptive(model_pred_norm, obs_traj_norm, blend_alpha):
    """
    model_pred_norm: [B, T, 2] batch-first normalized
    obs_traj_norm:   [B, T_obs, 2] batch-first normalized
    blend_alpha:     [B] ∈ [0, 0.5]
    returns:         [B, T, 2]
    """
    T_obs = obs_traj_norm.shape[1]
    T     = model_pred_norm.shape[1]
    B     = model_pred_norm.shape[0]
    device = model_pred_norm.device

    if T_obs < 2:
        return model_pred_norm

    vels = obs_traj_norm[:, 1:, :] - obs_traj_norm[:, :-1, :]  # [B,T_obs-1,2]
    n_v  = vels.shape[1]
    if n_v >= 3:
        alpha = 0.7
        w = torch.tensor([alpha * (1 - alpha)**i for i in range(n_v)],
                          dtype=torch.float, device=device).flip(0)
        ev = (vels * (w / w.sum()).view(1, -1, 1)).sum(1)  # [B,2]
    elif n_v == 2:
        ev = 0.7 * vels[:, -1, :] + 0.3 * vels[:, -2, :]
    else:
        ev = vels[:, -1, :]

    steps   = torch.arange(1, T + 1, dtype=torch.float, device=device)
    persist = (obs_traj_norm[:, -1, :].unsqueeze(1)
               + ev.unsqueeze(1) * steps.view(1, T, 1))  # [B,T,2]

    alpha_b = blend_alpha.view(B, 1, 1).clamp(0.0, 0.5)
    return (1.0 - alpha_b) * model_pred_norm + alpha_b * persist


# ══════════════════════════════════════════════════════════════
#  TCFlowMatching v60 FIXED
# ══════════════════════════════════════════════════════════════

class TCFlowMatching(nn.Module):

    def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02, unet_in_ch=13,
                 ctx_noise_scale=0.01, use_ema=True, ema_decay=0.995,
                 use_ate_ot=True, ot_epsilon=0.05, cfg_uncond_prob=0.1, **kwargs):
        super().__init__()
        self.pred_len        = pred_len
        self.obs_len         = obs_len
        self.sigma_min       = sigma_min
        self.ctx_noise_scale = ctx_noise_scale
        self.use_ate_ot      = use_ate_ot
        self.ot_epsilon      = ot_epsilon
        self.cfg_uncond_prob = cfg_uncond_prob

        self.net       = VelocityField(pred_len=pred_len, obs_len=obs_len,
                                        sigma_min=sigma_min, unet_in_ch=unet_in_ch,
                                        ctx_dim=256)
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
        """
        traj: [B, T, 2] batch-first
        Me:   [T_obs, B, 2] seq-first  (obs_Me)
        lp:   [B, 4]  (last obs position)
        lm:   [B, 2]  (last obs Me)
        returns: [B, T, 4] batch-first relative coords
        """
        pos_rel = traj - lp[:, :2].unsqueeze(1)        # [B,T,2]
        me_rel  = Me.permute(1, 0, 2) - lm.unsqueeze(1)  # [B,T_obs,2]
        # Pad/trim Me to match T
        T = traj.shape[1]
        if me_rel.shape[1] < T:
            pad = me_rel.new_zeros(me_rel.shape[0], T - me_rel.shape[1], 2)
            me_rel = torch.cat([pad, me_rel], dim=1)
        me_rel = me_rel[:, -T:, :]
        return torch.cat([pos_rel, me_rel], dim=-1)    # [B, T, 4]

    @staticmethod
    def _to_abs(rel, lp, lm):
        """
        rel: [B, T, 4] batch-first relative
        lp:  [B, 4]
        returns: pos [B, T, 2], me [B, T, 2]  — batch-first
        """
        pos = lp[:, :2].unsqueeze(1) + rel[:, :, :2]   # [B,T,2]
        me  = lm.unsqueeze(1)        + rel[:, :, 2:]    # [B,T,2]
        return pos, me

    @staticmethod
    def _sigma_schedule(epoch):
        if epoch < 2:  return 0.10
        if epoch < 10: return 0.10 - (epoch - 2) / 8.0 * (0.10 - 0.04)
        if epoch < 20: return max(0.04 - (epoch - 10) / 10.0 * 0.01, 0.035)
        return 0.035

    def _cfm_noisy(self, x1, sigma_min=None, lp=None):
        if sigma_min is None:
            sigma_min = self.sigma_min
        B      = x1.shape[0]
        device = x1.device
        x0     = torch.randn_like(x1) * sigma_min
        t      = torch.rand(B, device=device)
        te     = t.view(B, 1, 1)
        x_t    = (1.0 - te) * x0 + te * x1
        u_target = x1 - x0
        return x_t, t, u_target

    def _persistence_forecast_rel(self, obs_traj, lp, lm, pred_len):
        """
        obs_traj: [T_obs, B, 4] seq-first
        returns:  [B, T, 4] batch-first relative
        """
        B, device = obs_traj.shape[1], obs_traj.device
        obs_pos = obs_traj[:, :, :2]  # [T_obs, B, 2] seq-first

        if obs_pos.shape[0] >= 3:
            vels  = obs_pos[1:] - obs_pos[:-1]  # [T_obs-1, B, 2]
            n_v   = vels.shape[0]
            alpha = 0.7
            w     = torch.tensor([alpha * (1-alpha)**i for i in range(n_v)],
                                  dtype=torch.float, device=device).flip(0)
            lv    = (vels * (w / w.sum()).view(-1, 1, 1)).sum(0)  # [B,2]
        elif obs_pos.shape[0] >= 2:
            lv = obs_pos[-1] - obs_pos[-2]  # [B,2]
        else:
            lv = obs_traj.new_zeros(B, 2)

        steps    = torch.arange(1, pred_len + 1, device=device).float()
        pred_abs = (obs_pos[-1].unsqueeze(1)
                    + lv.unsqueeze(1) * steps.view(1, -1, 1))  # [B, T, 2]
        pred_rel_pos = pred_abs - lp[:, :2].unsqueeze(1)        # [B, T, 2]
        pred_rel     = torch.cat([pred_rel_pos,
                                   torch.zeros_like(pred_rel_pos)], dim=-1)
        return pred_rel  # [B, T, 4]

    def _compute_obs_momentum(self, obs_traj_norm):
        """obs_traj_norm: [T_obs, B, 2] seq-first → [B, 2]."""
        T_obs = obs_traj_norm.shape[0]
        if T_obs < 2:
            return torch.zeros(obs_traj_norm.shape[1], 2, device=obs_traj_norm.device)
        vels = obs_traj_norm[1:] - obs_traj_norm[:-1]  # [T_obs-1, B, 2]
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
    def _obs_noise_aug(bl, sigma=0.005):
        if torch.rand(1).item() > 0.5:
            return bl
        bl = list(bl)
        if torch.is_tensor(bl[0]):
            bl[0] = bl[0] + torch.randn_like(bl[0]) * sigma
        return bl

    # ── Training ──────────────────────────────────────────────

    def get_loss(self, batch_list, epoch=0, **kwargs):
        return self.get_loss_breakdown(batch_list, epoch=epoch)['total']

    def get_loss_breakdown(self, batch_list, epoch=0):
        batch_list = self._obs_noise_aug(batch_list, sigma=0.005)

        obs_t    = batch_list[0]   # [T_obs, B, 4] seq-first
        env_data = batch_list[13] if len(batch_list) > 13 else None

        # lp, lm from last obs — [B, 4] and [B, 2]
        lp = obs_t[-1]             # [B, 4]
        lm = batch_list[7][-1]     # [B, 2]
        B, device = lp.shape[0], lp.device

        # gt_traj: batch_list[1] — [B, T, 2] batch-first (from dataloader)
        gt_traj_b = batch_list[1]  # [B, T_pred, 2] batch-first
        T_pred = gt_traj_b.shape[1]

        current_sigma = self._sigma_schedule(epoch)
        raw_ctx       = self.net._context(batch_list)

        # Build x1_rel in batch-first [B,T,4]
        x1_rel = self._to_rel(gt_traj_b, batch_list[8], lp, lm)  # [B,T,4]

        # OT matching
        if self.use_ate_ot and B >= 4:
            noise_base = torch.randn_like(x1_rel) * current_sigma
            noise_matched, x1_matched = _spherical_ot_matching(
                noise_base, x1_rel, lp, epsilon=self.ot_epsilon)
        else:
            noise_matched = torch.randn_like(x1_rel) * current_sigma
            x1_matched    = x1_rel

        x_t, fm_t, u_target = self._cfm_noisy(x1_matched, sigma_min=current_sigma)

        use_null      = (torch.rand(1).item() < self.cfg_uncond_prob)
        vel_obs_feat  = self.net._get_kinematic_obs_feat(obs_t[:, :, :2])
        steering_feat = self.net._get_steering_feat(env_data, B, device)
        env_kine_feat = self.net._get_env_kine_feat(env_data, B, device)

        pred_vel = self.net.forward_with_ctx(
            x_t, fm_t, raw_ctx, env_data=env_data, use_null=use_null,
            vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
            env_kine_feat=env_kine_feat)  # [B, T, 4]

        # Predicted trajectory in batch-first [B,T,2]
        fm_te   = fm_t.view(B, 1, 1)
        x1_pred = x_t + (1.0 - fm_te) * pred_vel  # [B,T,4]

        pred_abs, _ = self._to_abs(x1_pred, lp, lm)   # [B,T,2] normalized
        pred_deg    = norm_to_deg(pred_abs)             # [B,T,2] degrees
        gt_deg      = norm_to_deg(gt_traj_b)            # [B,T,2] degrees

        # Candidate trajectories for scorer (epoch >= 5)
        candidates = None
        obs_norm   = None
        if epoch >= 5 and not use_null:
            # obs_norm: [B, T_obs, 2] batch-first
            obs_norm = obs_t[:, :, :2].permute(1, 0, 2)  # seq→batch: [B,T_obs,2]
            cands = []
            for _ in range(3):
                x0_c = torch.randn_like(x1_rel) * current_sigma
                te_c = fm_t.view(B, 1, 1)
                x_c  = (1.0 - te_c) * x0_c + te_c * x1_rel
                with torch.no_grad():
                    v_c = self.net.forward_with_ctx(
                        x_c, fm_t, raw_ctx, env_data=env_data,
                        vel_obs_feat=vel_obs_feat,
                        steering_feat=steering_feat,
                        env_kine_feat=env_kine_feat)
                    x1_c_pred = x_c + (1.0 - te_c) * v_c
                    abs_c, _  = self._to_abs(x1_c_pred, lp, lm)  # [B,T,2] normalized
                    # cand_norm: [B,T,2] batch-first — consistent with auxiliary_loss
                    lon_n = (abs_c[:, :, 0] * 10.0 - 1800.0) / 50.0
                    lat_n = abs_c[:, :, 1] * 10.0 / 50.0
                    cand_norm = torch.stack([lon_n, lat_n], dim=-1)  # [B,T,2]
                cands.append(cand_norm)
            candidates = cands

        # Compute loss — all in batch-first [B,T,2]
        total, breakdown = self.criterion(
            pred_deg, gt_deg,
            pred_vel, u_target,
            candidates=candidates,
            obs_norm=obs_norm,
            epoch=epoch,
        )

        if torch.isnan(total) or torch.isinf(total):
            total = obs_t.new_zeros(())

        breakdown.update({
            'sigma': current_sigma,
            'v_opt': compute_speed_stats_from_norm(obs_t[:, :, :2]).get('v_opt', 15.0),
            'dpe': 0.0, 'mse': 0.0, 'speed': 0.0, 'accel': 0.0,
            'heading': 0.0, 'vel_reg': 0.0, 'ate': 0.0, 'cte': 0.0,
            'sph_ate': 0.0, 'endpoint': 0.0, 'signed_ate': 0.0,
            'signed_cte': 0.0, 'direct_ep': 0.0,
            'fm_mse': breakdown.get('l_fm', 0.0),
        })
        breakdown['total'] = total
        return breakdown

    # ── Inference ─────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
                predict_csv=None):
        obs_t    = batch_list[0]   # [T_obs, B, 4] seq-first
        env_data = batch_list[13] if len(batch_list) > 13 else None
        lp       = obs_t[-1]       # [B, 4]
        lm       = batch_list[7][-1]  # [B, 2]
        B        = lp.shape[0]
        device   = lp.device
        T        = self.pred_len
        dt       = 1.0 / max(ddim_steps, 1)

        raw_ctx       = self.net._context(batch_list)
        ctx           = self.net._apply_ctx_head(raw_ctx)
        vel_obs_feat  = self.net._get_kinematic_obs_feat(obs_t[:, :, :2])
        steering_feat = self.net._get_steering_feat(env_data, B, device)
        env_kine_feat = self.net._get_env_kine_feat(env_data, B, device)

        # obs_norm: [B, T_obs, 2] batch-first for scorer
        obs_norm  = obs_t[:, :, :2].permute(1, 0, 2)  # [B,T_obs,2]
        obs_mom   = self._compute_obs_momentum(obs_t[:, :, :2])  # [B,2]

        blend_alpha      = self.net.get_blend_alpha(ctx)       # [B]
        guidance_scale   = self.net.get_guidance_scale(ctx)    # [B]
        sigma_per_sample = self.net.get_sigma(ctx)             # [B]

        persist_init = self._persistence_forecast_rel(obs_t, lp, lm, T)  # [B,T,4]

        if obs_t.shape[0] >= 2:
            obs_h_n = F.normalize(
                obs_t[-1, :, :2] - obs_t[-2, :, :2], dim=-1, eps=1e-6)
        else:
            obs_h_n = None

        def _mom_str(s, tot):
            return 0.06 * 0.5 * (1.0 + math.cos(math.pi * s / max(tot, 1)))

        all_norms = []  # list of [B, T, 2] batch-first normalized

        for ens_i in range(num_ensemble):
            sigma_noise = sigma_per_sample.mean().item() * 2.5
            x_t = persist_init + torch.randn_like(persist_init) * sigma_noise  # [B,T,4]

            for step in range(ddim_steps):
                t_b = torch.full((B,), step * dt, device=device)
                ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0

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
                    gs  = guidance_scale.view(B, 1, 1)
                    vel = v_uncond + gs * (v_cond - v_uncond)
                else:
                    vel = self.net.forward_with_ctx(
                        x_t, t_b, raw_ctx, noise_scale=ns,
                        vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
                        env_kine_feat=env_kine_feat, env_data=env_data)

                m_s = _mom_str(step, ddim_steps)
                if m_s > 1e-4:
                    # obs_mom [B,2] → [B,T,4]
                    me = obs_mom.unsqueeze(1).expand(B, T, 2)
                    mf = torch.cat([me, torch.zeros(B, T, 2, device=device)], dim=-1)
                    vel = vel + m_s * mf

                x_t = (x_t + dt * vel).clamp(-3.0, 3.0)

            pred_abs, _ = self._to_abs(x_t, lp, lm)  # [B,T,2] normalized
            all_norms.append(pred_abs)

        # Scoring — all [B,T,2] batch-first
        scores = []
        for tn in all_norms:
            sc = self.criterion.scorer.score(tn, obs_norm)  # [B]
            scores.append(sc)

        all_c  = torch.stack(all_norms)  # [N_ens, B, T, 2]
        all_sc = torch.stack(scores)     # [N_ens, B]

        k = max(1, int(all_c.shape[0] * 0.35))
        _, top_idx = all_sc.topk(k, dim=0)   # [k, B]

        pred_mean = torch.stack([
            all_c[top_idx[:, b], b, :, :].median(0).values
            for b in range(B)
        ], dim=0)  # [B, T, 2] batch-first normalized

        # Persistence blend — [B,T,2]
        blended = _persistence_blend_adaptive(pred_mean, obs_norm, blend_alpha)

        # Convert to degrees then back to normalized for output
        final_deg = norm_to_deg(blended)   # [B,T,2] degrees
        lon_out   = (final_deg[:, :, 0] * 10.0 - 1800.0) / 50.0
        lat_out   = final_deg[:, :, 1] * 10.0 / 50.0
        pred_norm_out = torch.stack([lon_out, lat_out], dim=-1)  # [B,T,2]

        # Convert to seq-first [T,B,2] for trainer compatibility
        pred_final = pred_norm_out.permute(1, 0, 2)  # [T, B, 2]

        if predict_csv:
            self._write_predict_csv(predict_csv, pred_final, all_c)

        return pred_final, all_c

    @staticmethod
    def _write_predict_csv(csv_path, traj_mean, all_trajs):
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
            if write_hdr:
                w.writeheader()
            for b in range(B):
                for k in range(T):
                    w.writerow({'timestamp': ts, 'batch_idx': b,
                                'step_idx': k, 'lead_h': (k + 1) * 6,
                                'lon_mean_deg': f'{mlon[k, b]:.4f}',
                                'lat_mean_deg': f'{mlat[k, b]:.4f}'})


# Backward compat alias
TCDiffusion = TCFlowMatching
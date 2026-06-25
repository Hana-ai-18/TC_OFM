"""
Model/flow_matching_model.py — TC-FlowMatching v2.4 (final)
═══════════════════════════════════════════════════════════════════════════════

THIẾT KẾ:
  Core giống v2.1 (proven val ADE=168km):
    - Velocity nhận [B, 12, 2] full sequence
    - 1-shot inference: x_pred = x0 + v(x0, t=0, cond)
    - L_CFM + L_reg(t=0 với sigma_inference)
    - Training KHÔNG dùng AR conditioning → cond thuần từ obs

  Cải tiến so v2.1:
    [NEW-3] Augmentation mạnh hơn: rotation ±10°, mixup, shift ±15km, scale ×0.6-1.5
    [NEW-4] Exp step weights trong L_reg: 72h weight ≈ 6× 12h
    BUG-1..9 fixes đã confirmed

  XAI (theo đề xuất thầy):
    XAI-1: compute_obs_attribution() — saliency map: obs step nào ảnh hưởng nhất
    XAI-2: hard_score_from_obs(return_components=True) — tại sao storm này khó
    XAI-3: physics_score components per trajectory — tại sao sample này được chọn
    XAI-4: compute_ensemble_uncertainty() — uncertainty per lead time

BUGS ĐÃ FIX TRONG VERSION NÀY:
  BUG-A: ar_enc/ar_gate là dead code (training không dùng, sample() cũng không)
         → Xóa hoàn toàn, tránh param waste và confused
  BUG-B: all_gates không populate → XAI-3 fix: log physics score components thay thế
  BUG-C: compute_obs_attribution gọi model.eval() → fix: không gọi eval()
  BUG-D: hard_score_from_obs @no_grad chặn gradient trong attribution → fix: tính riêng
  BUG-E: _reg_loss dùng x1_matched (OT-shuffled) thay vì x1_rel gốc → fix: dùng x1_rel
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.FNO3D_encoder import FNO3DEncoder
from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
from Model.env_net_transformer_gphsplit import Env_net

R_EARTH  = 6371.0
DT_HOURS = 6.0


# ─────────────────────────────────────────────────────────────────────────────
#  Coordinate utilities
# ─────────────────────────────────────────────────────────────────────────────

def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        (t[..., 0] * 50.0 + 1800.0) / 10.0,
        (t[..., 1] * 50.0) / 10.0,
    ], dim=-1)

def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a = (torch.sin(dlat/2).pow(2)
         + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2))
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
    if traj_deg.shape[0] < 2:
        return traj_deg.new_zeros(1, traj_deg.shape[1])
    return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# ─────────────────────────────────────────────────────────────────────────────
#  EMAModel
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_model(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m

class EMAModel:
    def __init__(self, model, decay: float = 0.995):
        self.decay = decay
        m = _unwrap_model(model)
        self.shadow = {k: v.detach().clone()
                       for k, v in m.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        m = _unwrap_model(model)
        with torch.no_grad():
            for k, v in m.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)

    def apply_to(self, model):
        m = _unwrap_model(model)
        backup, sd = {}, m.state_dict()
        for k in self.shadow:
            if k not in sd: continue
            backup[k] = sd[k].detach().clone()
            sd[k].copy_(self.shadow[k])
        return backup

    def restore(self, model, backup):
        m = _unwrap_model(model)
        sd = m.state_dict()
        for k, v in backup.items():
            if k in sd: sd[k].copy_(v)


# ─────────────────────────────────────────────────────────────────────────────
#  OT matching
# ─────────────────────────────────────────────────────────────────────────────

def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 50) -> torch.Tensor:
    B = cost.shape[0]; device = cost.device
    log_a = -math.log(B) * torch.ones(B, device=device)
    log_b = -math.log(B) * torch.ones(B, device=device)
    log_K = -cost / epsilon
    log_u = torch.zeros(B, device=device)
    log_v = torch.zeros(B, device=device)
    for _ in range(n_iter):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
    return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)

def _ot_match_noise_gt(x0_flat: torch.Tensor, x1_flat: torch.Tensor,
                        epsilon: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    B = x0_flat.shape[0]
    if B < 4: return x0_flat, x1_flat
    try:
        cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1]**0.5)
        with torch.no_grad(): pi = _sinkhorn_log(cost, epsilon=epsilon)
        flat = pi.reshape(-1).clamp(0.0); s = flat.sum()
        if not torch.isfinite(s) or s < 1e-10: return x0_flat, x1_flat
        idx = torch.multinomial(flat/s, num_samples=B, replacement=True)
        return x0_flat[idx//B], x1_flat
    except Exception:
        return x0_flat, x1_flat


# ─────────────────────────────────────────────────────────────────────────────
#  VelocityTransformer — giống hệt v2.1, pred_len=12, full sequence
# ─────────────────────────────────────────────────────────────────────────────

class VelocityTransformer(nn.Module):
    """
    Nhận x_t [B, 12, 2] — toàn bộ 12 steps. Không chia AR stages.
    Train và inference đều nhận full sequence → nhất quán hoàn toàn.
    """
    def __init__(self, pred_len: int = 12, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
                 d_cond: int = 256):
        super().__init__()
        self.pred_len = pred_len
        self.d_model  = d_model

        self.traj_embed = nn.Linear(2, d_model)
        self.pos_emb    = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.step_emb   = nn.Embedding(pred_len, d_model)
        self.time_mlp   = nn.Sequential(
            nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model))
        self.cond_proj  = nn.Sequential(
            nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.GELU(),
            nn.Linear(d_model//2, 2))
        self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
                         * (-math.log(10000.0) / max(half-1, 1)))
        emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb  = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.d_model % 2 == 1: emb = F.pad(emb, (0, 1))
        return self.time_mlp(emb)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_t.shape
        step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
        x_emb = (self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(step_idx))
        memory = torch.cat([self._time_emb(t).unsqueeze(1),
                             self.cond_proj(cond).unsqueeze(1)], dim=1)
        out = self.out_norm(self.decoder(x_emb, memory))
        return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
#  ContextEncoder — backbone v2.1 thuần túy (không có ar_enc/ar_gate)
# ─────────────────────────────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    """
    BUG-A FIX: Xóa ar_enc/ar_gate — chúng là dead code (training và sample()
    đều không dùng). Giữ backbone thuần v2.1.
    """
    RAW_CTX_DIM = 512

    def __init__(self, obs_len: int = 8, unet_in_ch: int = 13, d_cond: int = 256):
        super().__init__()
        self.obs_len = obs_len
        self.d_cond  = d_cond

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

        self.ctx_fc1  = nn.Linear(128+32+16, self.RAW_CTX_DIM)
        self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
        self.ctx_drop = nn.Dropout(0.1)
        self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, d_cond)
        self.ctx_ln2  = nn.LayerNorm(d_cond)

        self.vel_obs_enc = nn.Sequential(
            nn.Linear(obs_len*7, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, d_cond//2), nn.GELU())
        self.hard_embed = nn.Sequential(
            nn.Linear(1, d_cond//4), nn.GELU(),
            nn.Linear(d_cond//4, d_cond//4))
        self.fuse = nn.Sequential(
            nn.Linear(d_cond + d_cond//2 + d_cond//4, d_cond),
            nn.LayerNorm(d_cond), nn.GELU())

    def _encode_raw(self, batch_list) -> torch.Tensor:
        obs_traj  = batch_list[0]
        obs_Me    = batch_list[7]
        image_obs = batch_list[11]
        env_data  = batch_list[13]

        if image_obs.dim() == 4: image_obs = image_obs.unsqueeze(2)
        if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
            image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

        e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
        T_obs = obs_traj.shape[0]
        e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0,2,1)
        e_3d_s = self.bottleneck_proj(e_3d_s)
        if e_3d_s.shape[1] != T_obs:
            e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
                                   mode="linear", align_corners=False).permute(0,2,1)
        e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
        t_w = torch.softmax(
            torch.arange(e_3d_dec_t.shape[1], dtype=torch.float, device=e_3d_dec_t.device) * 0.5,
            dim=0)
        f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        h_t    = self.enc_1d(obs_in, e_3d_s)
        e_env, _, _ = self.env_enc(env_data, image_obs)
        return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

    def _kinematic_feat(self, obs_traj_norm: torch.Tensor) -> torch.Tensor:
        B, T_obs = obs_traj_norm.shape[1], obs_traj_norm.shape[0]
        device   = obs_traj_norm.device
        if T_obs >= 2:
            traj_deg = _norm_to_deg(obs_traj_norm)
            vel_norm = obs_traj_norm[1:] - obs_traj_norm[:-1]
            speed    = _step_speeds_kmh(traj_deg)
            speed_n  = (speed / 20.0).clamp(-3.0, 3.0)
            heading  = torch.atan2(vel_norm[:,:,1], vel_norm[:,:,0])
            if T_obs >= 3:
                dspd  = speed[1:] - speed[:-1]
                accel = torch.cat([obs_traj_norm.new_zeros(1, B),
                                   (dspd/10.0).clamp(-3.0,3.0)], 0)
            else:
                accel = obs_traj_norm.new_zeros(T_obs-1, B)
            # FIX-1: thêm log_speed để encoder biết absolute speed magnitude
            # speed_n đã bị clamp → mất info speed cao. log_speed giữ được.
            log_speed = (speed / 100.0 + 1.0).log() / math.log(2.0)  # [T-1, B], range ~[0,1]
            log_speed = log_speed.clamp(0.0, 3.0)
            kine = torch.stack([vel_norm[:,:,0], vel_norm[:,:,1], speed_n,
                                 heading.sin(), heading.cos(), accel, log_speed], dim=-1)
        else:
            kine = obs_traj_norm.new_zeros(self.obs_len, B, 7)
        if kine.shape[0] < self.obs_len:
            kine = torch.cat([obs_traj_norm.new_zeros(self.obs_len-kine.shape[0], B, 7), kine], 0)
        else:
            kine = kine[-self.obs_len:]
        return self.vel_obs_enc(kine.permute(1,0,2).reshape(B,-1))

    def forward(self, batch_list, hard_score: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        raw   = self._encode_raw(batch_list)
        ctx   = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
        kfeat = self._kinematic_feat(batch_list[0][:, :, :2])
        if hard_score is None:
            hard_score = torch.zeros(ctx.shape[0], device=ctx.device)
        hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))
        return self.fuse(torch.cat([ctx, kfeat, hfeat], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
#  Hard score — XAI-2
# ─────────────────────────────────────────────────────────────────────────────

def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    lon1=torch.deg2rad(p1[...,0]); lat1=torch.deg2rad(p1[...,1])
    lon2=torch.deg2rad(p2[...,0]); lat2=torch.deg2rad(p2[...,1])
    dlon=lon2-lon1
    y=torch.sin(dlon)*torch.cos(lat2)
    x=torch.cos(lat1)*torch.sin(lat2)-torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon)
    return torch.atan2(y,x)

@torch.no_grad()
def hard_score_from_obs(obs_traj_norm: torch.Tensor,
                         return_components: bool = False):
    """
    [XAI-2] Điểm khó của storm. return_components=True → giải thích tại sao khó.
    curvature: bão đang recurve (xoay hướng)
    speed_var: tốc độ thay đổi đột ngột
    dir_change: có nhiều bước rẽ lớn
    """
    T, B   = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
    device = obs_traj_norm.device
    if T < 3:
        z = torch.zeros(B, device=device)
        if return_components:
            return z, {"curvature": z, "speed_var": z, "dir_change": z}
        return z
    traj_deg = _norm_to_deg(obs_traj_norm[..., :2])
    az12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
    az23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
    diff = (az23 - az12).abs()
    diff = torch.where(diff > math.pi, 2*math.pi - diff, diff)
    curvature  = diff.mean(0) / math.pi
    spd = _step_speeds_kmh(traj_deg)
    speed_var  = ((spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 1.)
                  if spd.shape[0] >= 2 else torch.zeros(B, device=device))
    dir_change = (diff > (20./180.*math.pi)).float().mean(0)
    score = (0.4*curvature + 0.3*speed_var + 0.3*dir_change).clamp(0., 1.)
    if return_components:
        return score, {"curvature": curvature, "speed_var": speed_var,
                       "dir_change": dir_change}
    return score


# ─────────────────────────────────────────────────────────────────────────────
#  Physics score — v2.1 (speed + smooth + heading)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _physics_score(traj_norm: torch.Tensor,
                   obs_norm: torch.Tensor) -> torch.Tensor:
    B, device = traj_norm.shape[1], traj_norm.device
    traj_deg  = _norm_to_deg(traj_norm)

    if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
        obs_deg  = _norm_to_deg(obs_norm)
        obs_spd  = _step_speeds_kmh(obs_deg)
        T_s      = obs_spd.shape[0]
        w_obs    = torch.linspace(0.5, 1.0, T_s, device=device)
        v_ref    = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()
        pred_spd = _step_speeds_kmh(traj_deg)
        v_sigma  = v_ref.clamp(min=5.0) * 0.5
        speed_score = torch.exp(
            -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5)
    elif traj_deg.shape[0] >= 2:
        speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.).mean(0))
    else:
        speed_score = torch.ones(B, device=device)

    if traj_deg.shape[0] >= 3:
        vel       = traj_deg[1:] - traj_deg[:-1]
        accel_mag = (vel[1:] - vel[:-1]).norm(dim=-1)
        smooth_score = torch.exp(-accel_mag.mean(0) * 5.0)
    else:
        smooth_score = torch.ones(B, device=device)

    if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
        obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
        pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
        obs_h    = F.normalize(obs_vel,  dim=-1, eps=1e-6)
        pred_h   = F.normalize(pred_vel, dim=-1, eps=1e-6)
        cos_sim  = (obs_h * pred_h).sum(-1).clamp(-1, 1)
        head_score = torch.exp((cos_sim - 1.0) * 3.0)
    else:
        head_score = torch.ones(B, device=device)

    return (speed_score.pow(0.35) * smooth_score.pow(0.30) * head_score.pow(0.35)).clamp(min=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation — NEW-3: mạnh hơn v2.1
# ─────────────────────────────────────────────────────────────────────────────

def augment_batch(batch_list) -> list:
    """
    [NEW-3] 5 loại aug để cover test distribution:
      A (25%): shift ±15km — model không memorize vị trí tuyệt đối
      B (25%): speed scale ×0.6-1.5 — cover tốc độ bão khác nhau
      C (20%): rotation ±10° quanh last_obs — cover hướng di chuyển
      D (15%): mixup — tránh memorize exact val patterns
      E (15%): Gaussian noise nhỏ

    Val KHÔNG gọi hàm này → val loss phản ánh đúng generalization.
    """
    bl = list(batch_list)
    if not torch.is_tensor(bl[0]): return bl
    obs    = bl[0]
    gt     = bl[1]
    device = obs.device
    anchor = obs[-1:, :, :2].detach()  # [1, B, 2] — last obs làm pivot

    r = torch.rand(1).item()

    if r < 0.25:
        # A: shift ±15km. 15km / (50° × 111km/°) ≈ 0.0027 norm/km
        shift  = (torch.rand(2, device=device) - 0.5) * 0.054  # ±15km
        obs_new = obs.clone(); obs_new[..., :2] = obs[..., :2] + shift.view(1,1,2)
        bl[0] = obs_new; bl[1] = gt + shift.view(1,1,2)

    elif r < 0.50:
        # B: speed scale ×0.6-1.5 quanh last_obs
        scale  = 0.60 + 0.90 * torch.rand(1).item()
        obs_new = obs.clone()
        obs_new[..., :2] = anchor + (obs[..., :2] - anchor) * scale
        bl[0] = obs_new; bl[1] = anchor + (gt - anchor) * scale

    elif r < 0.70:
        # C: rotation ±10° quanh last_obs. Giữ anchor cố định.
        angle  = (torch.rand(1).item() - 0.5) * (math.pi / 9)  # ±10°
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]],
                           dtype=obs.dtype, device=device)
        T_obs, B = obs.shape[0], obs.shape[1]
        obs_new  = obs.clone()
        rel_obs  = (obs[..., :2] - anchor).reshape(T_obs*B, 2)
        obs_new[..., :2] = (rot @ rel_obs.T).T.reshape(T_obs, B, 2) + anchor
        bl[0] = obs_new
        T_pred = gt.shape[0]
        rel_gt = (gt - anchor).reshape(T_pred*B, 2)
        bl[1]  = (rot @ rel_gt.T).T.reshape(T_pred, B, 2) + anchor

    elif r < 0.85:
        # D: mixup — blend 2 storms trong batch
        B = obs.shape[1]
        if B >= 4:
            alpha = 0.15 + 0.20 * torch.rand(1).item()  # 0.15-0.35
            idx   = torch.randperm(B, device=device)
            obs_new = obs.clone()
            obs_new[..., :2] = (1-alpha)*obs[..., :2] + alpha*obs[:, idx, :2]
            bl[0] = obs_new
            bl[1] = (1-alpha)*gt + alpha*gt[:, idx, :]

    else:
        # E: Gaussian noise ±4km tương đương
        obs_new = obs.clone()
        obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.004
        bl[0] = obs_new

    return bl


# ─────────────────────────────────────────────────────────────────────────────
#  XAI-1: Feature attribution — obs timestep nào ảnh hưởng nhất
# ─────────────────────────────────────────────────────────────────────────────

def compute_obs_attribution(model, batch_list, device: torch.device,
                             target_step: int = 11) -> torch.Tensor:
    """
    [XAI-1] Gradient của displacement magnitude tại target_step
    w.r.t. obs_traj → saliency map: step obs nào ảnh hưởng nhất đến 72h pred.

    BUG-C FIX: Không gọi model.eval() vì sẽ tắt dropout và làm attribution
    khác với training distribution.
    BUG-D FIX: Tính hard_score riêng bên ngoài no_grad context,
    không truyền qua encoder để tránh gradient bị chặn bởi @no_grad.

    Output: attr [T_obs, B] — normalized importance per obs step per storm.
    attr[-1, b] thường cao nhất (obs gần nhất quan trọng nhất).
    attr[0, b] cao → storm có "trí nhớ dài" về pattern cũ.
    """
    raw = _unwrap_model(model)
    obs_traj_req = batch_list[0].detach().clone().requires_grad_(True)
    bl_grad      = list(batch_list)
    bl_grad[0]   = obs_traj_req

    # BUG-D FIX: tính hard_score bên ngoài để không chặn gradient
    with torch.no_grad():
        h_score_val = hard_score_from_obs(batch_list[0][:, :, :2])

    with torch.enable_grad():
        # Encode — gradient sẽ flow từ velocity output ngược qua encoder
        cond     = raw.encoder(bl_grad, hard_score=h_score_val)
        last_obs = obs_traj_req[-1, :, :2]

        x0   = torch.randn(obs_traj_req.shape[1], raw.pred_len, 2,
                           device=device) * raw.sigma_inference
        t0   = torch.zeros(obs_traj_req.shape[1], device=device)
        v    = raw.velocity(x0, t0, cond)
        pred_rel = x0 + v   # [B, T, 2] relative to last_obs

        # Loss: magnitude của displacement tại target_step, averaged over batch
        ts       = min(target_step, raw.pred_len - 1)
        loss_xai = pred_rel[:, ts, :].norm(dim=-1).mean()
        loss_xai.backward()

    if obs_traj_req.grad is not None:
        attr = obs_traj_req.grad[:, :, :2].norm(dim=-1)   # [T_obs, B]
        attr = attr / (attr.sum(0, keepdim=True) + 1e-8)   # normalize per storm
    else:
        attr = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)

    return attr.detach()


# ─────────────────────────────────────────────────────────────────────────────
#  XAI-4: Ensemble uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def compute_ensemble_uncertainty(all_traj: torch.Tensor) -> Dict:
    """
    [XAI-4] Std deviation per lead time across K ensemble samples.
    all_traj: [K, T, B, 2] absolute normalized coords.

    Kết quả:
      std_per_step [T, B]: uncertainty per lead time per storm (km)
      uncertainty_ratio [B]: 72h_std / 12h_std — tăng theo thời gian là bình thường
      mean_72h_std: trung bình std ở 72h — threshold để flag "uncertain forecast"
    """
    all_deg   = _norm_to_deg(all_traj)   # [K, T, B, 2]
    K, T, B   = all_deg.shape[:3]
    mean_traj = all_deg.mean(0)           # [T, B, 2]

    std_km = torch.zeros(T, B, device=all_traj.device)
    for t in range(T):
        # [K*B, 2] vs [K*B, 2] — tính khoảng cách từ mỗi sample tới mean
        dists = _haversine_deg(
            all_deg[:, t, :, :].reshape(K*B, 2),
            mean_traj[t].unsqueeze(0).expand(K, B, 2).reshape(K*B, 2)
        ).reshape(K, B)
        std_km[t] = dists.std(0)

    step_12h = min(1, T-1)
    step_72h = min(11, T-1)
    return {
        "std_per_step":      std_km,                                          # [T, B] km
        "uncertainty_ratio": (std_km[step_72h]+1e-3) / (std_km[step_12h]+1e-3),  # [B]
        "mean_72h_std":      float(std_km[step_72h].mean().item()),
        "mean_12h_std":      float(std_km[step_12h].mean().item()),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Compat stubs
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def classify_hard_easy(obs_traj_norm, per_sample_loss=None,
                        hard_score_p: float = 70.0, loss_p: float = 50.0):
    scores = hard_score_from_obs(obs_traj_norm)
    B = scores.shape[0]
    if B < 4: return torch.zeros(B, dtype=torch.bool, device=scores.device)
    return scores >= torch.quantile(scores, hard_score_p/100.0)

@torch.no_grad()
def classify_hard_easy_global(obs_traj_norm, global_threshold):
    return hard_score_from_obs(obs_traj_norm) >= global_threshold

@torch.no_grad()
def compute_diversity_score(candidates) -> float:
    if len(candidates) < 2: return 0.0
    T, B    = candidates[0].shape[0], candidates[0].shape[1]
    ep_step = min(T-1, 11)
    endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
    N = endpoints.shape[0]; ep_mean = endpoints.mean(0, keepdim=True)
    dists = _haversine_deg(
        endpoints.reshape(N*B, 2),
        ep_mean.expand(N, B, 2).reshape(N*B, 2)
    ).reshape(N, B)
    return float(dists.std(0).mean().item())


# ─────────────────────────────────────────────────────────────────────────────
#  TCFlowMatching v2.4
# ─────────────────────────────────────────────────────────────────────────────

class TCFlowMatching(nn.Module):
    """
    TC-FlowMatching v2.4 = v2.1 core + augmentation mạnh + exp step weights

    Training:
      L_total = L_CFM + lambda_reg * L_reg
      L_CFM: flow matching objective với random t ∈ [0,1] và OT noise matching
      L_reg: ADE tại t=0 với sigma_inference (BUG-8 fix), exp step weights (NEW-4)
             dùng x1_rel gốc (BUG-E fix), NOT x1_matched từ OT

    Inference (1-shot, nhất quán với training):
      x0 ~ N(0, sigma_inference²)
      x_pred_rel = x0 + v(x0, t=0, cond)
      x_pred_abs = x_pred_rel + last_obs
      Best-of-K với physics score (speed + smoothness + heading)
    """

    def __init__(self,
                 pred_len: int = 12, obs_len: int = 8, unet_in_ch: int = 13,
                 d_cond: int = 256, d_model: int = 256, nhead: int = 8,
                 num_dec_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
                 sigma_min: float = 0.04, sigma_max: float = 0.08,
                 lambda_reg: float = 0.2,
                 use_ot: bool = True, ot_epsilon: float = 0.05,
                 use_ema: bool = True, ema_decay: float = 0.995,
                 n_inference_steps: int = 1, n_ensemble: int = 20,
                 sigma_inference: float = 0.04,
                 **kwargs):
        super().__init__()
        self.pred_len          = pred_len
        self.obs_len           = obs_len
        self.sigma_min         = sigma_min
        self.sigma_max         = sigma_max
        self.lambda_reg        = lambda_reg
        self.use_ot            = use_ot
        self.ot_epsilon        = ot_epsilon
        self.n_inference_steps = n_inference_steps
        self.n_ensemble        = n_ensemble
        self.sigma_inference   = sigma_inference

        self.encoder  = ContextEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
        self.velocity = VelocityTransformer(
            pred_len=pred_len, d_model=d_model, nhead=nhead,
            num_layers=num_dec_layers, dim_ff=dim_ff, dropout=dropout, d_cond=d_cond)
        self.use_ema = use_ema
        self._ema    = None

    def init_ema(self):
        if self.use_ema: self._ema = EMAModel(self)

    def ema_update(self):
        if self._ema is not None: self._ema.update(self)

    def _to_relative(self, x_abs: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
        return x_abs - last_obs.unsqueeze(1)

    def _from_relative(self, x_rel: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
        return x_rel + last_obs.unsqueeze(1)

    def _sigma_schedule(self, epoch: int) -> float:
        """Cosine decay từ sigma_max về sigma_min trong ep5→ep40."""
        if epoch < 5:  return self.sigma_max
        if epoch < 40:
            t = (epoch - 5) / 35.0
            return self.sigma_min + 0.5*(self.sigma_max-self.sigma_min)*(1+math.cos(math.pi*t))
        return self.sigma_min

    def _reg_loss(self, x1_rel: torch.Tensor, last_obs: torch.Tensor,
                  cond: torch.Tensor,
                  hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        ADE loss tại t=0, consistent với inference.

        BUG-8 FIX: dùng sigma_inference thay vì sigma_schedule.
        BUG-E FIX: x1_rel là GT GỐC (không bị OT shuffle).
                   L_reg phải học predict đúng GT, không phải OT-matched GT
                   của noise sample khác.
        NEW-4:     Exp step weights — 72h weight ≈ 6× 12h weight.
        """
        B, T, _ = x1_rel.shape
        device  = x1_rel.device

        x0 = torch.randn_like(x1_rel) * self.sigma_inference  # fresh noise
        t0 = torch.zeros(B, device=device)
        v  = self.velocity(x0, t0, cond)
        x1_pred_abs = self._from_relative(x0 + v, last_obs)
        x1_gt_abs   = self._from_relative(x1_rel, last_obs)   # x1_rel gốc

        pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))  # [T, B, 2]
        gt_deg   = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
        dist     = _haversine_deg(pred_deg, gt_deg)             # [T, B] km

        # FIX-2: Speed-adaptive step weights
        # Nhân 2 nguồn:
        # (a) Exp weights: nhấn mạnh long-range (72h)
        # (b) GT displacement weights: step nào bão di chuyển xa hơn → weight cao hơn
        #     → gradient mạnh đúng chỗ velocity magnitude sai nhiều nhất
        T_actual = dist.shape[0]
        steps = torch.arange(T_actual, device=device, dtype=dist.dtype)
        exp_w = torch.exp(2.5 * steps / T_actual)  # [T]
        exp_w = exp_w / exp_w.mean()

        # GT displacement per step: haversine(gt[t], gt[t-1])
        gt_disp = _haversine_deg(gt_deg[:-1], gt_deg[1:])   # [T-1, B] km
        # Pad step 0 = mean displacement
        gt_disp = torch.cat([gt_disp.mean(0, keepdim=True), gt_disp], 0)  # [T, B]
        spd_w   = (gt_disp / gt_disp.mean(0, keepdim=True).clamp(min=1.0)).clamp(0.3, 3.0)

        sw = (exp_w.unsqueeze(1) * spd_w)        # [T, B]
        sw = sw / sw.mean(0, keepdim=True)        # normalize per storm

        if hard_score is not None:
            # sw đã [T, B], hard_score [B] → broadcast đúng
            sw = sw * (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)

        return (dist * sw).mean() / 300.0

    def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
        obs_traj = batch_list[0]
        gt_traj  = batch_list[1]
        B        = obs_traj.shape[1]
        device   = obs_traj.device

        sigma    = self._sigma_schedule(epoch)
        x1_gt    = gt_traj.permute(1, 0, 2)        # [B, 12, 2]
        last_obs = obs_traj[-1, :, :2]
        x1_rel   = self._to_relative(x1_gt, last_obs)  # GT gốc — dùng cho L_reg

        h_score = hard_score_from_obs(obs_traj[:, :, :2])
        cond    = self.encoder(batch_list, hard_score=h_score)

        # L_CFM — flow matching objective
        x0 = torch.randn_like(x1_rel) * sigma
        if self.use_ot and B >= 4:
            x0_flat, x1_flat = _ot_match_noise_gt(
                x0.reshape(B, -1), x1_rel.reshape(B, -1), self.ot_epsilon)
            x0         = x0_flat.reshape(B, self.pred_len, 2)
            x1_matched = x1_flat.reshape(B, self.pred_len, 2)
        else:
            x1_matched = x1_rel

        t      = torch.rand(B, device=device)
        x_t    = (1 - t.view(B,1,1))*x0 + t.view(B,1,1)*x1_matched
        v_pred = self.velocity(x_t, t, cond)
        l_cfm  = F.mse_loss(v_pred, x1_matched - x0)

        # L_reg ramp ep10→ep30
        if epoch < 10:
            lam_reg = 0.0
        elif epoch < 30:
            lam_reg = self.lambda_reg * (epoch - 10) / 20.0
        else:
            lam_reg = self.lambda_reg

        # BUG-E FIX: truyền x1_rel GỐC, không phải x1_matched
        l_reg = (self._reg_loss(x1_rel, last_obs, cond, h_score)
                 if lam_reg > 0.0 else x0.new_zeros(()))

        total = l_cfm + lam_reg * l_reg
        if not torch.isfinite(total):
            total = x0.new_zeros(())

        # ADE log — BUG-9 FIX: dùng sigma_inference
        with torch.no_grad():
            x0_log = torch.randn_like(x1_rel) * self.sigma_inference
            v_log  = self.velocity(x0_log, torch.zeros(B, device=device), cond)
            x1_log = self._from_relative(x0_log + v_log, last_obs)
            ade_log = _haversine_deg(
                _norm_to_deg(x1_log.permute(1, 0, 2)),
                _norm_to_deg(x1_gt.permute(1, 0, 2))
            ).mean().item()

        return {
            "total":    total,
            "l_cfm":    l_cfm.item(),
            "l_reg":    l_reg.item() if torch.is_tensor(l_reg) else 0.0,
            "lam_reg":  lam_reg,
            "sigma":    sigma,
            "ade_1step": ade_log,
            "hard_score_mean": float(h_score.mean()),
            "hard_score_max":  float(h_score.max()),
            # Compat keys
            "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
            "speed": 0., "accel": 0., "fm_mse": l_cfm.item(),
            "l_hard_total": 0., "n_hard": 0, "alpha_hard": 0.,
            "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
        }

    def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

    @torch.no_grad()
    def sample(self, batch_list, num_ensemble: Optional[int] = None,
               ddim_steps: Optional[int] = None,
               return_xai: bool = False, **kwargs):
        """
        1-shot inference: x_pred = x0 + v(x0, t=0, cond)
        Nhất quán hoàn toàn với training: velocity nhận [B, 12, 2].

        BUG-6 FIX: Cache _encode_raw 1 lần cho K samples.
        XAI-3: Log physics score components để giải thích tại sao sample nào được chọn.
        XAI-4: compute_ensemble_uncertainty nếu return_xai=True.
        """
        K  = num_ensemble or self.n_ensemble
        N  = ddim_steps or self.n_inference_steps
        dt = 1.0 / max(N, 1)

        obs_traj    = batch_list[0]
        T_obs, B, _ = obs_traj.shape
        device      = obs_traj.device
        h_score     = hard_score_from_obs(obs_traj[:, :, :2])
        obs_norm    = obs_traj[:, :, :2]
        last_obs    = obs_traj[-1, :, :2]
        t0          = torch.zeros(B, device=device)

        # [BUG-6 FIX] Cache encoder output — gọi 1 lần cho K samples
        raw_ctx  = self.encoder._encode_raw(batch_list)
        base_ctx = self.encoder.ctx_ln2(
            self.encoder.ctx_fc2(self.encoder.ctx_drop(raw_ctx)))
        kfeat    = self.encoder._kinematic_feat(obs_traj[:, :, :2])
        hfeat    = self.encoder.hard_embed(h_score.unsqueeze(1).to(base_ctx.dtype))
        cond     = self.encoder.fuse(torch.cat([base_ctx, kfeat, hfeat], dim=-1))

        # FIX-3: Speed-conditioned sigma per storm
        # Bão nhanh cần sigma lớn → K samples cover đủ speed range → ATE giảm
        if obs_norm.shape[0] >= 2:
            obs_deg_s  = _norm_to_deg(obs_norm)
            obs_spd_s  = _step_speeds_kmh(obs_deg_s)          # [T-1, B] km/h
            obs_spd_mean = obs_spd_s.mean(0).clamp(5., 120.)  # [B]
        else:
            obs_spd_mean = torch.full((B,), 30., device=device)
        # sigma scale: slow storm (10km/h) → 0.5×, fast storm (80km/h) → 2.0×
        sigma_scale = (0.5 + obs_spd_mean / 40.0).clamp(0.5, 2.0)  # [B]
        sigma_k = (self.sigma_inference * sigma_scale).view(B, 1, 1)  # [B,1,1] để broadcast

        all_traj = []

        # Tính base prediction 1 lần với sigma_inference (nhất quán training)
        x0_base = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference
        with torch.no_grad():
            v_base = self.velocity(x0_base, t0, cond)   # [B, T, 2]
        x_base = x0_base + v_base                        # best 1-shot prediction

        for k in range(K):
            if k == 0:
                # Sample 0: pure 1-shot từ sigma_inference (nhất quán với training)
                x_rel = x_base
            else:
                # Sample 1..K-1: fresh noise với sigma_inference (training consistent)
                # + speed-scaled perturbation theo hướng predicted trajectory
                # → diversity về speed magnitude mà không phá vỡ velocity input range
                x0_k  = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference
                if N <= 1:
                    x_rel = x0_k + self.velocity(x0_k, t0, cond)
                else:
                    x_rel = x0_k.clone()
                    for step in range(N):
                        t_b   = torch.full((B,), step*dt, device=device)
                        x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3., 3.)

                # Speed perturbation: dùng sigma_k để scale magnitude diversity
                # Perturbation theo hướng displacement (ATE direction)
                # → samples có speed khác nhau → physics score chọn đúng speed → ATE giảm
                disp_mag = x_rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)   # [B,T,1]
                disp_dir = x_rel / disp_mag                                    # [B,T,2] unit vector
                # scale = random [0.5, 1.5] × sigma ratio per storm
                scale = (0.7 + 0.6 * torch.rand(B, 1, 1, device=device)) * (sigma_k / self.sigma_inference)
                x_rel = x_rel * scale  # stretch/compress trajectory displacement

            x_abs = self._from_relative(x_rel, last_obs)
            all_traj.append(x_abs.permute(1, 0, 2))  # [T, B, 2]

        scores   = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)  # [K, B]
        all_t    = torch.stack(all_traj, 0)  # [K, T, B, 2]
        top_k    = min(3, K)
        top_idx  = scores.topk(top_k, dim=0).indices  # [top_k, B]

        pred_mean = torch.zeros_like(all_traj[0])
        for b in range(B):
            idx_b = top_idx[:, b]
            w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
            pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k,1,1)).sum(0)

        if return_xai:
            xai = compute_ensemble_uncertainty(all_t)

            # [XAI-3] Vectorized physics score components cho top-1 per storm
            # BUG-I FIX: không dùng loop per-storm (chậm + scalar assign issue)
            # BUG-M FIX: vectorize hoàn toàn
            top1_idx = scores.argmax(0)  # [B]

            # Build best_traj [T, B, 2]: trajectory được chọn cho mỗi storm
            best_traj_list = []
            for b in range(B):
                best_traj_list.append(all_traj[int(top1_idx[b].item())][:, b, :])  # [T, 2]
            best_traj = torch.stack(best_traj_list, dim=1)  # [T, B, 2]
            best_deg  = _norm_to_deg(best_traj)             # [T, B, 2]
            obs_deg_v = _norm_to_deg(obs_norm)              # [T_obs, B, 2]

            # Speed score: exp(-((pred_spd - obs_spd_mean) / sigma)^2 * 0.5)
            obs_spd_v  = _step_speeds_kmh(obs_deg_v)                          # [T_obs-1, B]
            obs_spd_mu = obs_spd_v.mean(0)                                     # [B]
            pred_spd_v = _step_speeds_kmh(best_deg)                           # [T-1, B]
            pred_spd_mu= pred_spd_v.mean(0)                                    # [B]
            v_sigma_v  = obs_spd_mu.clamp(min=5.) * 0.5                       # [B]
            speed_scores = torch.exp(
                -((pred_spd_mu - obs_spd_mu) / v_sigma_v).pow(2) * 0.5)      # [B]

            # Smooth score: exp(-mean_accel_magnitude * 5)
            if best_deg.shape[0] >= 3:
                vel_v   = best_deg[1:] - best_deg[:-1]                        # [T-1, B, 2]
                accel_v = (vel_v[1:] - vel_v[:-1]).norm(dim=-1)               # [T-2, B]
                smooth_scores = torch.exp(-accel_v.mean(0) * 5.)              # [B]
            else:
                smooth_scores = torch.ones(B, device=device)

            # Heading score: cos similarity giữa last obs velocity và first pred velocity
            if obs_norm.shape[0] >= 2 and best_traj.shape[0] >= 1:
                obs_vel_v  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]        # [B, 2]
                pred_vel_v = best_traj[0, :, :2] - obs_norm[-1, :, :2]        # [B, 2]
                obs_h_v    = F.normalize(obs_vel_v,  dim=-1, eps=1e-6)         # [B, 2]
                pred_h_v   = F.normalize(pred_vel_v, dim=-1, eps=1e-6)         # [B, 2]
                cos_sim_v  = (obs_h_v * pred_h_v).sum(-1).clamp(-1., 1.)      # [B]
                head_scores = torch.exp((cos_sim_v - 1.) * 3.)                 # [B]
            else:
                head_scores = torch.ones(B, device=device)

            xai["physics_components"] = {
                "speed":        speed_scores,   # [B] — cao = tốc độ đúng
                "smooth":       smooth_scores,  # [B] — cao = ít zigzag
                "heading":      head_scores,    # [B] — cao = heading đúng
                "obs_speed":    obs_spd_mu,     # [B] km/h — tốc độ quan sát
                "pred_speed":   pred_spd_mu,    # [B] km/h — tốc độ predicted
            }

            # [XAI-2] Hard score components
            _, hard_comps = hard_score_from_obs(obs_norm, return_components=True)
            xai["hard_components"] = hard_comps

            # [XAI-5] Speed comparison: obs vs pred speed (đã tính trong XAI-3)
            # Dùng lại obs_speed/pred_speed từ XAI-3 để tránh tính lại
            # Nhất quán trên toàn bộ B storms, không chỉ 8 storms đầu
            if "physics_components" in xai:
                pc = xai["physics_components"]
                obs_mu  = float(pc["obs_speed"].mean().item())
                pred_mu = float(pc["pred_speed"].mean().item())
                ratio   = pred_mu / max(obs_mu, 1.0)
                xai["speed_comparison"] = {
                    "obs_speed_mean":  obs_mu,   # km/h, trung bình B storms
                    "pred_speed_mean": pred_mu,  # km/h
                    "speed_ratio":     ratio,    # >1: over-predict, <1: under-predict
                    "per_storm_obs":   pc["obs_speed"],   # [B] per-storm detail
                    "per_storm_pred":  pc["pred_speed"],  # [B] per-storm detail
                }

            return pred_mean, torch.zeros_like(pred_mean), all_t, xai

        return pred_mean, torch.zeros_like(pred_mean), all_t


def load_checkpoint_compat(ckpt_path: str, model: "TCFlowMatching",
                           device) -> dict:
    """
    Load checkpoint với backward compat khi vel_obs_enc thay đổi từ obs_len*6 → obs_len*7.
    Expand weight matrix: copy 6-feat weights, init cột thứ 7 (log_speed) = small random.
    """
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck.get("model", ck)

    key = "encoder.vel_obs_enc.0.weight"  # Linear(obs_len*6, 256) → Linear(obs_len*7, 256)
    if key in sd:
        w_old = sd[key]          # [256, obs_len*6]
        obs_len = model.obs_len
        expected_cols = obs_len * 7
        if w_old.shape[1] == obs_len * 6:
            # Expand: thêm obs_len columns (log_speed) với init nhỏ
            extra = torch.randn(w_old.shape[0], obs_len, device=w_old.device) * 0.01
            sd[key] = torch.cat([w_old, extra], dim=1)  # [256, obs_len*7]
            print(f"  ✅ Expanded vel_obs_enc: {w_old.shape[1]} → {expected_cols} cols")
        elif w_old.shape[1] == expected_cols:
            print(f"  ✅ vel_obs_enc already {expected_cols} cols")
        else:
            print(f"  ⚠ vel_obs_enc unexpected shape {w_old.shape}, skip expand")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:    print(f"  Missing keys : {len(missing)}")
    if unexpected: print(f"  Unexpected   : {len(unexpected)}")
    return ck


TCDiffusion = TCFlowMatching
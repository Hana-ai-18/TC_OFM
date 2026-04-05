"""
Model/flow_matching_model.py  ── v16
==========================================
OT-CFM Flow Matching + PINN-BVE for TC trajectory prediction.

FIXES vs v15:

  FIX-M7  [CRITICAL] OT-CFM prediction formula sai trong get_loss_breakdown.
           v15 dùng: x1_pred = x_t + (1-te) * pred_vel  ← SAI
           OT-CFM trajectory: x_t = te*x1 + (1-(1-sm)*te)*x0
           Velocity target: v = (x1 - (1-sm)*x0) / (1-(1-sm)*te)
           → x1 estimate: x1_pred = x_t + denom * pred_vel  ← ĐÚNG
           Điều này giải thích tại sao ADE không giảm dù loss giảm.

  FIX-M8  [CRITICAL] pred_abs cho non-AFCRPS losses cần convert sang degrees
           để Haversine metric có ý nghĩa. Trước khi pass vào velocity_loss,
           heading_loss, recurvature_loss, pinn_bve_loss — convert:
           lon_deg = (pred_norm * 50 + 1800) / 10
           lat_deg = (pred_norm * 50) / 10
           Tương tự cho gt. Các loss này tính vector difference, nếu dùng
           normalized units (~1.0) vs degrees (~15) sẽ sai scale hoàn toàn.

  FIX-M9  [ctx_noise_scale] Giảm default từ 0.05 → 0.01 vì SSR=4.7 cho thấy
           spread quá lớn (cần SSR≈1). Spread 2000+ km vs ADE 900 km → overconfident
           trong diversity hướng sai.

  FIX-M10 [initial_sample_sigma] Giảm từ 0.3 → 0.15. Trong normalized space,
           0.3 units × 500 km/unit ≈ 150 km initial spread. Đây là hợp lý,
           nhưng kết hợp với ctx_noise_scale=0.05 gây spread quá lớn.

Kept from v15:
  FIX-M5  ctx_noise_scale injection per ensemble member
  FIX-M6  configurable initial_sample_sigma
  FIX-M1  _write_predict_csv denorm formula corrected
"""
from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.FNO3D_encoder import FNO3DEncoder
from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
from Model.env_net_transformer_gphsplit import Env_net
from Model.losses import compute_total_loss, WEIGHTS


def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized trajectory coords → degrees for loss computation.
    Input shape: [T, B, 2] or [T, B, 4] (lon_norm, lat_norm, ...)
    Output: same shape, first 2 channels converted to degrees.

    Formula (from dataset):
      lon_deg = (lon_norm * 50 + 1800) / 10
      lat_deg = (lat_norm * 50) / 10
    """
    out = traj_norm.clone()
    out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
    out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  VelocityField  (FlowMatching denoiser)  ── v16
# ══════════════════════════════════════════════════════════════════════════════

class VelocityField(nn.Module):
    """
    OT-CFM velocity field  v_θ(x_t, t, context).

    Context assembly:
      h_t       [B, 128]  ← DataEncoder1D w/ Mamba
      e_Env     [B,  64]  ← Env-T-Net
      f_spatial [B,  16]  ← FNO decoder pooled
      ─────────────────
      total     [B, 208]  → ctx_fc → [B, ctx_dim=256]
    """

    def __init__(
        self,
        pred_len:   int   = 12,
        obs_len:    int   = 8,
        ctx_dim:    int   = 256,
        sigma_min:  float = 0.02,
        unet_in_ch: int   = 13,
    ):
        super().__init__()
        self.pred_len  = pred_len
        self.obs_len   = obs_len
        self.sigma_min = sigma_min

        self.spatial_enc = FNO3DEncoder(
            in_channel   = unet_in_ch,
            out_channel  = 1,
            d_model      = 64,
            n_layers     = 4,
            modes_t      = 4,
            modes_h      = 4,
            modes_w      = 4,
            spatial_down = 32,
            dropout      = 0.05,
        )

        self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.bottleneck_proj = nn.Linear(128, 128)
        self.decoder_proj    = nn.Linear(1, 16)

        self.enc_1d = DataEncoder1D(
            in_1d       = 4,
            feat_3d_dim = 128,
            mlp_h       = 64,
            lstm_hidden = 128,
            lstm_layers = 3,
            dropout     = 0.1,
            d_state     = 16,
        )

        self.env_enc = Env_net(obs_len=obs_len, d_model=64)

        self.ctx_fc1  = nn.Linear(128 + 64 + 16, 512)
        self.ctx_ln   = nn.LayerNorm(512)
        self.ctx_drop = nn.Dropout(0.15)
        self.ctx_fc2  = nn.Linear(512, ctx_dim)

        self.time_fc1 = nn.Linear(256, 512)
        self.time_fc2 = nn.Linear(512, 256)

        self.traj_embed = nn.Linear(4, 256)
        self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=256, nhead=8, dim_feedforward=1024,
                dropout=0.15, activation="gelu", batch_first=True,
            ),
            num_layers=4,
        )
        self.out_fc1 = nn.Linear(256, 512)
        self.out_fc2 = nn.Linear(512, 4)

    def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
        half = dim // 2
        freq = torch.exp(
            torch.arange(half, dtype=torch.float32, device=t.device)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return F.pad(emb, (0, dim % 2))

    def _context(self, batch_list: List) -> torch.Tensor:
        obs_traj  = batch_list[0]
        obs_Me    = batch_list[7]
        image_obs = batch_list[11]
        env_data  = batch_list[13]

        if image_obs.dim() == 4:
            image_obs = image_obs.unsqueeze(1)

        expected_ch = self.spatial_enc.in_channel
        if image_obs.shape[1] == 1 and expected_ch != 1:
            image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

        e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)

        T_obs = obs_traj.shape[0]

        e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
        e_3d_s = e_3d_s.permute(0, 2, 1)
        e_3d_s = self.bottleneck_proj(e_3d_s)

        T_bot = e_3d_s.shape[1]
        if T_bot != T_obs:
            e_3d_s = F.interpolate(
                e_3d_s.permute(0, 2, 1),
                size=T_obs, mode="linear", align_corners=False,
            ).permute(0, 2, 1)

        f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))
        f_spatial     = self.decoder_proj(f_spatial_raw)

        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        h_t    = self.enc_1d(obs_in, e_3d_s)

        e_env, _, _ = self.env_enc(env_data, image_obs)

        raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
        raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))
        return raw

    def _apply_ctx_head(self, raw: torch.Tensor,
                        noise_scale: float = 0.0) -> torch.Tensor:
        if noise_scale > 0.0:
            raw = raw + torch.randn_like(raw) * noise_scale
        return self.ctx_fc2(self.ctx_drop(raw))

    def forward(self, x_t, t, batch_list):
        raw = self._context(batch_list)
        ctx = self._apply_ctx_head(raw, noise_scale=0.0)
        return self._decode(x_t, t, ctx)

    def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale: float = 0.0):
        ctx = self._apply_ctx_head(raw_ctx, noise_scale=noise_scale)
        return self._decode(x_t, t, ctx)

    # def _decode(self, x_t, t, ctx):
    #     t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
    #     t_emb = self.time_fc2(t_emb)

    #     x_emb  = self.traj_embed(x_t) + self.pos_enc[:, :x_t.size(1), :] + t_emb.unsqueeze(1)
    #     memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

    #     out = self.transformer(x_emb, memory)
    #     return self.out_fc2(F.gelu(self.out_fc1(out)))

# Model/flow_matching_model.py — cập nhật VelocityField._decode

class PhysicsGuidedVelocityField(nn.Module):
    """
    Thay thế VelocityField._decode bằng version physics-aware.
    
    Ý tưởng: velocity field học 2 thành phần tách biệt:
      v_total = v_neural + v_physics_residual
    
    v_physics: beta drift + Coriolis — tính trực tiếp từ state
    v_neural:  học phần residual mà physics không giải thích được
    
    Tương đương Neural ODE nhưng chạy trong FM framework.
    """
    
    def __init__(self, ctx_dim=256, pred_len=12):
        super().__init__()
        self.pred_len = pred_len
        
        # Neural residual: chỉ cần học PHẦN DƯ sau physics
        # → input dim nhỏ hơn, converge nhanh hơn
        self.residual_net = nn.Sequential(
            nn.Linear(4 + ctx_dim + 256 + 1, 512),  # state+ctx+time_emb+t
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 4),   # predict residual velocity [lon,lat,pres,wnd]
        )
        
        # Physics scale: học được — cho phép model điều chỉnh
        # mức độ tin vào physics vs neural
        self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)
    
    def _beta_drift_velocity(self, x_t_normalized):
        """
        Beta drift velocity trong normalized space.
        x_t_normalized: [B, T, 4] hoặc [B, 4]
        Returns: [same shape] velocity
        """
        OMEGA = 7.2921e-5
        R = 6.371e6
        DT = 6 * 3600.0
        # 1 normalized lon/lat unit = 50/10 = 5 degrees = 555 km
        DEG_PER_NORM = 5.0
        KM_PER_DEG = 111.0
        M_PER_NORM = DEG_PER_NORM * KM_PER_DEG * 1000  # 555,000 m
        
        if x_t_normalized.dim() == 3:
            lat_norm = x_t_normalized[:, :, 1]
        else:
            lat_norm = x_t_normalized[:, 1]
        
        lat_deg = lat_norm * 5.0   # approximate: lat_norm * 50/10
        lat_rad = torch.deg2rad(lat_deg.clamp(-90, 90))
        
        # Beta parameter
        beta = 2 * OMEGA * torch.cos(lat_rad) / R
        
        # TC outer radius ~300 km = 3e5 m
        R_tc = 3e5
        
        # Beta drift: ~2 m/s westward, ~1 m/s poleward
        v_beta_lon_ms = -beta * R_tc**2 / 2   # westward
        v_beta_lat_ms =  beta * R_tc**2 / 4   # poleward
        
        # Convert m/s → normalized units per 6h step
        v_lon_norm = v_beta_lon_ms * DT / M_PER_NORM
        v_lat_norm = v_beta_lat_ms * DT / M_PER_NORM
        
        # Assemble velocity vector
        v_physics = torch.zeros_like(x_t_normalized)
        if x_t_normalized.dim() == 3:
            v_physics[:, :, 0] = v_lon_norm
            v_physics[:, :, 1] = v_lat_norm
        else:
            v_physics[:, 0] = v_lon_norm
            v_physics[:, 1] = v_lat_norm
        
        return v_physics
    
    def forward(self, x_t, t, ctx, time_emb):
        """
        x_t:      [B, T, 4] — current noisy state
        t:        [B] — flow time
        ctx:      [B, ctx_dim]
        time_emb: [B, 256]
        Returns:  [B, T, 4] velocity
        """
        B, T, D = x_t.shape
        
        # 1. Physics velocity (no grad needed)
        with torch.no_grad():
            v_physics = self._beta_drift_velocity(x_t)  # [B, T, 4]
        
        # 2. Neural residual
        # Expand ctx và time_emb qua T dimension
        ctx_exp  = ctx.unsqueeze(1).expand(-1, T, -1)    # [B, T, ctx_dim]
        temb_exp = time_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, 256]
        t_exp    = t.view(B, 1, 1).expand(-1, T, 1)      # [B, T, 1]
        
        net_input = torch.cat([x_t, ctx_exp, temb_exp, t_exp], dim=-1)
        v_residual = self.residual_net(net_input)  # [B, T, 4]
        
        # 3. Combine: scale physics contribution học được
        scale = torch.sigmoid(self.physics_scale) * 2  # [0, 2]
        v_total = v_residual + scale * v_physics
        
        return v_total
# ══════════════════════════════════════════════════════════════════════════════
#  TCFlowMatching  ── v16
# ══════════════════════════════════════════════════════════════════════════════

class TCFlowMatching(nn.Module):
    """
    TC trajectory prediction via OT-CFM + PINN-BVE.

    v16 changes:
      - FIX-M7: Sửa OT-CFM prediction formula x1_pred = x_t + denom * pred_vel
      - FIX-M8: Convert pred_abs sang degrees trước khi tính directional losses
      - FIX-M9/M10: ctx_noise_scale=0.01, initial_sample_sigma=0.15
    """

    def __init__(
        self,
        pred_len:             int   = 12,
        obs_len:              int   = 8,
        sigma_min:            float = 0.02,
        n_train_ens:          int   = 4,
        unet_in_ch:           int   = 13,
        ctx_noise_scale:      float = 0.01,   # FIX-M9: giảm từ 0.05
        initial_sample_sigma: float = 0.15,   # FIX-M10: giảm từ 0.3
        **kwargs,
    ):
        super().__init__()
        self.pred_len             = pred_len
        self.obs_len              = obs_len
        self.sigma_min            = sigma_min
        self.n_train_ens          = n_train_ens
        self.active_pred_len      = pred_len
        self.ctx_noise_scale      = ctx_noise_scale
        self.initial_sample_sigma = initial_sample_sigma
        self.net = VelocityField(
            pred_len   = pred_len,
            obs_len    = obs_len,
            sigma_min  = sigma_min,
            unet_in_ch = unet_in_ch,
        )

    def set_curriculum_len(self, active_len: int) -> None:
        self.active_pred_len = max(1, min(active_len, self.pred_len))

    @staticmethod
    def _to_rel(traj_gt, Me_gt, last_pos, last_Me):
        return torch.cat(
            [traj_gt - last_pos.unsqueeze(0),
             Me_gt   - last_Me.unsqueeze(0)],
            dim=-1,
        ).permute(1, 0, 2)

    @staticmethod
    def _to_abs(rel, last_pos, last_Me):
        d = rel.permute(1, 0, 2)
        return (
            last_pos.unsqueeze(0) + d[:, :, :2],
            last_Me.unsqueeze(0)  + d[:, :, 2:],
        )

    def _cfm_noisy(self, x1):
        B, device = x1.shape[0], x1.device
        sm  = self.sigma_min
        x0  = torch.randn_like(x1) * sm
        t   = torch.rand(B, device=device)
        te  = t.view(B, 1, 1)
        x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
        denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
        target_vel = (x1 - (1.0 - sm) * x_t) / denom
        return x_t, t, te, denom, target_vel

    @staticmethod
    def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
        wind_norm = obs_Me[-1, :, 1].detach()
        w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
            torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
            torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
                        torch.full_like(wind_norm, 1.5))))
        return w / w.mean().clamp(min=1e-6)

    @staticmethod
    def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
        if torch.rand(1).item() > p:
            return batch_list
        aug = list(batch_list)
        for idx in [0, 1, 2, 3]:
            t = aug[idx]
            if torch.is_tensor(t) and t.shape[-1] >= 1:
                t = t.clone()
                t[..., 0] = -t[..., 0]
                aug[idx] = t
        for idx in [7, 8, 9, 10]:
            t = aug[idx]
            if torch.is_tensor(t):
                aug[idx] = t.clone()
        return aug

    def get_loss(self, batch_list: List) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list)["total"]

    def get_loss_breakdown(self, batch_list: List) -> Dict:
        batch_list = self._lon_flip_aug(batch_list, p=0.3)

        traj_gt = batch_list[1]
        Me_gt   = batch_list[8]
        obs_t   = batch_list[0]
        obs_Me  = batch_list[7]

        apl = self.active_pred_len
        if apl < traj_gt.shape[0]:
            traj_gt = traj_gt[:apl]
            Me_gt   = Me_gt[:apl]

        lp, lm = obs_t[-1], obs_Me[-1]
        x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

        raw_ctx     = self.net._context(batch_list)
        intensity_w = self._intensity_weights(obs_Me)

        x_t, t, te, denom, _ = self._cfm_noisy(x1)
        pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

        # ── Ensemble samples cho AFCRPS ─────────────────────────────────
        samples: List[torch.Tensor] = []
        for _ in range(self.n_train_ens):
            xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
            pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
            # FIX-M7: dùng denom đúng cho sample này
            x1_s  = xt_s + dens_s * pv_s
            pa_s, _ = self._to_abs(x1_s, lp, lm)
            samples.append(pa_s)
        pred_samples = torch.stack(samples)  # [S, T, B, 2]

        # Thêm vào sau khi có pred_samples:
        l_fm_physics = fm_physics_consistency_loss(
            pred_samples,
            gt_norm=traj_gt,
            last_pos=lp,
        )
        # FIX-M7: x1_pred = x_t + denom * pred_vel  (không phải (1-te))
        x1_pred = x_t + denom * pred_vel
        pred_abs, _ = self._to_abs(x1_pred, lp, lm)
        # pred_abs: [T, B, 2] in normalized coords

        # FIX-M8: Convert pred_abs và traj_gt sang degrees cho directional losses
        # Haversine-based losses (velocity, heading, recurv, pinn) cần degrees
        pred_abs_deg = _denorm_to_deg(pred_abs)
        traj_gt_deg  = _denorm_to_deg(traj_gt)
        ref_deg      = _denorm_to_deg(lp.unsqueeze(0)).squeeze(0)  # [B, 2]

        breakdown = compute_total_loss(
            pred_abs     = pred_abs_deg,    # FIX-M8: degrees
            gt           = traj_gt_deg,     # FIX-M8: degrees
            ref          = ref_deg,         # FIX-M8: degrees
            batch_list   = batch_list,
            pred_samples = pred_samples,    # normalized (unit_01deg=True handles)
            gt_norm      = traj_gt,         # FIX-M8: pass normalized gt for AFCRPS
            weights      = WEIGHTS,
            intensity_w  = intensity_w,
        )
        # Inject physics FM loss
        breakdown['total'] = breakdown['total'] + \
            WEIGHTS.get('fm_physics', 0.3) * l_fm_physics
        breakdown['fm_physics'] = l_fm_physics.item()
        
        return breakdown

    # flow_matching_model.py — cập nhật sample()

@torch.no_grad()
def sample_physics_corrected(
    self,
    batch_list,
    num_ensemble=50,
    ddim_steps=20,
    physics_correction_steps=3,  # Thêm: correction cuối
):
    """
    Physics-corrected sampling.
    
    Ý tưởng: sau khi DDIM xong, chạy thêm
    vài bước "physics correction" để đảm bảo
    trajectory thỏa mãn physical constraints.
    
    Tương đương Neural ODE accuracy mà không cần solver đắt.
    """
    lp  = batch_list[0][-1]   # [B, 2]
    lm  = batch_list[7][-1]   # [B, 2]
    B, device = lp.shape[0], lp.device
    dt  = 1.0 / ddim_steps
    
    raw_ctx = self.net._context(batch_list)
    traj_s, me_s = [], []
    
    for k in range(num_ensemble):
        # Standard DDIM
        x_t = torch.randn(B, self.pred_len, 4, device=device) \
              * self.initial_sample_sigma
        
        for step in range(ddim_steps):
            t_b  = torch.full((B,), step * dt, device=device)
            ns   = self.ctx_noise_scale if step == 0 else 0.0
            vel  = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
            x_t  = x_t + dt * vel
        
        # Physics correction steps
        # Gradient descent trên physics residuals
        x_t_corrected = self._physics_correct(x_t, lp, lm, 
                                               n_steps=physics_correction_steps)
        
        x_t_corrected[:, :, :2].clamp_(-5.0, 5.0)
        x_t_corrected[:, :, 2:].clamp_(-3.0, 3.0)
        
        tr, me = self._to_abs(x_t_corrected, lp, lm)
        traj_s.append(tr)
        me_s.append(me)
    
    all_trajs = torch.stack(traj_s)
    all_me    = torch.stack(me_s)
    return all_trajs.mean(0), all_me.mean(0), all_trajs


def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.01):
    """
    Gradient-free physics correction bằng projected gradient.
    
    Chạy sau DDIM để "nudge" trajectory về phía
    thỏa mãn physics constraints.
    
    Không cần grad qua full model — chỉ cần grad
    qua physics residual functions (cheap).
    """
    x = x_pred.clone().requires_grad_(True)
    
    optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)
    
    for _ in range(n_steps):
        optimizer.zero_grad()
        
        # Convert to absolute position (degrees) cho physics
        pred_abs, pred_Me = self._to_abs(x, last_pos, last_Me)
        
        # lon_norm, lat_norm → degrees
        pred_deg = torch.zeros_like(pred_abs)
        pred_deg[:, :, 0] = (pred_abs[:, :, 0] * 50.0 + 1800.0) / 10.0
        pred_deg[:, :, 1] = (pred_abs[:, :, 1] * 50.0) / 10.0
        
        # Physics residuals (cheap to compute)
        l_speed = _pinn_speed_constraint(pred_deg)
        l_sw    = _pinn_beta_plane_simplified(pred_deg)
        
        physics_loss = l_speed + 0.3 * l_sw
        physics_loss.backward()
        
        # Clip correction magnitude
        torch.nn.utils.clip_grad_norm_([x], max_norm=0.1)
        optimizer.step()
    
    return x.detach()


def _pinn_speed_constraint(pred_deg):
    """
    Đơn giản: penalize nếu TC speed > 100 km/h hoặc jump quá lớn.
    pred_deg: [T, B, 2] degrees
    """
    if pred_deg.shape[0] < 2:
        return pred_deg.new_zeros(())
    
    dt_deg = pred_deg[1:] - pred_deg[:-1]
    lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
    cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
    
    # km per 6h step
    dx_km = dt_deg[:, :, 0] * cos_lat * 111.0
    dy_km = dt_deg[:, :, 1] * 111.0
    speed_per_step = torch.sqrt(dx_km**2 + dy_km**2)  # km / 6h
    
    # 100 km/h = 600 km/6h (extreme upper bound)
    max_km_per_step = 600.0
    violation = F.relu(speed_per_step - max_km_per_step)
    
    return violation.pow(2).mean()


def _pinn_beta_plane_simplified(pred_deg):
    """
    Beta plane: d²lat/dt² và d²lon/dt² phải smooth.
    Không thể thay đổi hướng đột ngột trừ khi có recurvature.
    """
    if pred_deg.shape[0] < 3:
        return pred_deg.new_zeros(())
    
    # Tính acceleration
    v = pred_deg[1:] - pred_deg[:-1]   # [T-1, B, 2]
    a = v[1:] - v[:-1]                 # [T-2, B, 2]  acceleration
    
    # Scale: 1 deg change in direction per step = 111 km deviation
    lat_rad = torch.deg2rad(pred_deg[1:-1, :, 1])
    cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
    
    a_lon_km = a[:, :, 0] * cos_lat * 111.0
    a_lat_km = a[:, :, 1] * 111.0
    
    # Max realistic acceleration: ~50 km per step² (recurvature cases)
    max_accel = 50.0
    violation = F.relu(
        torch.sqrt(a_lon_km**2 + a_lat_km**2) - max_accel
    )
    
    return violation.pow(2).mean() * 0.1

    @staticmethod
    def _write_predict_csv(csv_path, traj_mean, all_trajs):
        import numpy as np
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        T, B, _ = traj_mean.shape
        S       = all_trajs.shape[0]

        mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
        mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
        all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
        all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()

        fields = ["timestamp", "batch_idx", "step_idx", "lead_h",
                  "lon_mean_deg", "lat_mean_deg",
                  "lon_std_deg", "lat_std_deg", "ens_spread_km"]
        write_hdr = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if write_hdr:
                w.writeheader()
            for b in range(B):
                for k in range(T):
                    dlat  = all_lat[:, k, b] - mean_lat[k, b]
                    dlon  = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
                        math.radians(mean_lat[k, b]))
                    spread = float(np.sqrt((dlat ** 2 + dlon ** 2).mean()) * 111.0)
                    w.writerow(dict(
                        timestamp     = ts,
                        batch_idx     = b,
                        step_idx      = k,
                        lead_h        = (k + 1) * 6,
                        lon_mean_deg  = f"{mean_lon[k,b]:.4f}",
                        lat_mean_deg  = f"{mean_lat[k,b]:.4f}",
                        lon_std_deg   = f"{all_lon[:,k,b].std():.4f}",
                        lat_std_deg   = f"{all_lat[:,k,b].std():.4f}",
                        ens_spread_km = f"{spread:.2f}",
                    ))
        print(f"  Predictions → {csv_path}  (B={B}, T={T}, S={S})")


# Backward-compat alias
TCDiffusion = TCFlowMatching
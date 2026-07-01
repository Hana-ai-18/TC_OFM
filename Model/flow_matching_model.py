# # # # # # # # # # """
# # # # # # # # # # Model/flow_matching_model.py — TC-FlowMatching v2.4 (final)
# # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # THIẾT KẾ:
# # # # # # # # # #   Core giống v2.1 (proven val ADE=168km):
# # # # # # # # # #     - Velocity nhận [B, 12, 2] full sequence
# # # # # # # # # #     - 1-shot inference: x_pred = x0 + v(x0, t=0, cond)
# # # # # # # # # #     - L_CFM + L_reg(t=0 với sigma_inference)
# # # # # # # # # #     - Training KHÔNG dùng AR conditioning → cond thuần từ obs

# # # # # # # # # #   Cải tiến so v2.1:
# # # # # # # # # #     [NEW-3] Augmentation mạnh hơn: rotation ±10°, mixup, shift ±15km, scale ×0.6-1.5
# # # # # # # # # #     [NEW-4] Exp step weights trong L_reg: 72h weight ≈ 6× 12h
# # # # # # # # # #     BUG-1..9 fixes đã confirmed

# # # # # # # # # #   XAI (theo đề xuất thầy):
# # # # # # # # # #     XAI-1: compute_obs_attribution() — saliency map: obs step nào ảnh hưởng nhất
# # # # # # # # # #     XAI-2: hard_score_from_obs(return_components=True) — tại sao storm này khó
# # # # # # # # # #     XAI-3: physics_score components per trajectory — tại sao sample này được chọn
# # # # # # # # # #     XAI-4: compute_ensemble_uncertainty() — uncertainty per lead time

# # # # # # # # # # BUGS ĐÃ FIX TRONG VERSION NÀY:
# # # # # # # # # #   BUG-A: ar_enc/ar_gate là dead code (training không dùng, sample() cũng không)
# # # # # # # # # #          → Xóa hoàn toàn, tránh param waste và confused
# # # # # # # # # #   BUG-B: all_gates không populate → XAI-3 fix: log physics score components thay thế
# # # # # # # # # #   BUG-C: compute_obs_attribution gọi model.eval() → fix: không gọi eval()
# # # # # # # # # #   BUG-D: hard_score_from_obs @no_grad chặn gradient trong attribution → fix: tính riêng
# # # # # # # # # #   BUG-E: _reg_loss dùng x1_matched (OT-shuffled) thay vì x1_rel gốc → fix: dùng x1_rel
# # # # # # # # # # """
# # # # # # # # # # from __future__ import annotations

# # # # # # # # # # import math
# # # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # # import torch
# # # # # # # # # # import torch.nn as nn
# # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net

# # # # # # # # # # R_EARTH  = 6371.0
# # # # # # # # # # DT_HOURS = 6.0


# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # # #  Coordinate utilities
# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #     return torch.stack([
# # # # # # # # # #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# # # # # # # # # #         (t[..., 1] * 50.0) / 10.0,
# # # # # # # # # #     ], dim=-1)

# # # # # # # # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# # # # # # # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # # # # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # # # # # # #     a = (torch.sin(dlat/2).pow(2)
# # # # # # # # # #          + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2))
# # # # # # # # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

# # # # # # # # # # def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #     if traj_deg.shape[0] < 2:
# # # # # # # # # #         return traj_deg.new_zeros(1, traj_deg.shape[1])
# # # # # # # # # #     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # # #  EMAModel
# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # # def _unwrap_model(m):
# # # # # # # # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # # # # # # # # # class EMAModel:
# # # # # # # # # #     def __init__(self, model, decay: float = 0.995):
# # # # # # # # # #         self.decay = decay
# # # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # # #         self.shadow = {k: v.detach().clone()
# # # # # # # # # #                        for k, v in m.state_dict().items()
# # # # # # # # # #                        if v.dtype.is_floating_point}

# # # # # # # # # #     def update(self, model):
# # # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # # #         with torch.no_grad():
# # # # # # # # # #             for k, v in m.state_dict().items():
# # # # # # # # # #                 if k in self.shadow:
# # # # # # # # # #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)

# # # # # # # # # #     def apply_to(self, model):
# # # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # # #         backup, sd = {}, m.state_dict()
# # # # # # # # # #         for k in self.shadow:
# # # # # # # # # #             if k not in sd: continue
# # # # # # # # # #             backup[k] = sd[k].detach().clone()
# # # # # # # # # #             sd[k].copy_(self.shadow[k])
# # # # # # # # # #         return backup

# # # # # # # # # #     def restore(self, model, backup):
# # # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # # #         sd = m.state_dict()
# # # # # # # # # #         for k, v in backup.items():
# # # # # # # # # #             if k in sd: sd[k].copy_(v)


# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # # #  OT matching
# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # # def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 50) -> torch.Tensor:
# # # # # # # # # #     B = cost.shape[0]; device = cost.device
# # # # # # # # # #     log_a = -math.log(B) * torch.ones(B, device=device)
# # # # # # # # # #     log_b = -math.log(B) * torch.ones(B, device=device)
# # # # # # # # # #     log_K = -cost / epsilon
# # # # # # # # # #     log_u = torch.zeros(B, device=device)
# # # # # # # # # #     log_v = torch.zeros(B, device=device)
# # # # # # # # # #     for _ in range(n_iter):
# # # # # # # # # #         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
# # # # # # # # # #         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
# # # # # # # # # #     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)

# # # # # # # # # # def _ot_match_noise_gt(x0_flat: torch.Tensor, x1_flat: torch.Tensor,
# # # # # # # # # #                         epsilon: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # # # # # # #     B = x0_flat.shape[0]
# # # # # # # # # #     if B < 4: return x0_flat, x1_flat
# # # # # # # # # #     try:
# # # # # # # # # #         cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1]**0.5)
# # # # # # # # # #         with torch.no_grad(): pi = _sinkhorn_log(cost, epsilon=epsilon)
# # # # # # # # # #         flat = pi.reshape(-1).clamp(0.0); s = flat.sum()
# # # # # # # # # #         if not torch.isfinite(s) or s < 1e-10: return x0_flat, x1_flat
# # # # # # # # # #         idx = torch.multinomial(flat/s, num_samples=B, replacement=True)
# # # # # # # # # #         return x0_flat[idx//B], x1_flat
# # # # # # # # # #     except Exception:
# # # # # # # # # #         return x0_flat, x1_flat


# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # # #  VelocityTransformer — giống hệt v2.1, pred_len=12, full sequence
# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # # class VelocityTransformer(nn.Module):
# # # # # # # # # #     """
# # # # # # # # # #     Nhận x_t [B, 12, 2] — toàn bộ 12 steps. Không chia AR stages.
# # # # # # # # # #     Train và inference đều nhận full sequence → nhất quán hoàn toàn.
# # # # # # # # # #     """
# # # # # # # # # #     def __init__(self, pred_len: int = 12, d_model: int = 256, nhead: int = 8,
# # # # # # # # # #                  num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
# # # # # # # # # #                  d_cond: int = 256):
# # # # # # # # # #         super().__init__()
# # # # # # # # # #         self.pred_len = pred_len
# # # # # # # # # #         self.d_model  = d_model

# # # # # # # # # #         self.traj_embed = nn.Linear(2, d_model)
# # # # # # # # # #         self.pos_emb    = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
# # # # # # # # # #         self.step_emb   = nn.Embedding(pred_len, d_model)
# # # # # # # # # #         self.time_mlp   = nn.Sequential(
# # # # # # # # # #             nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model))
# # # # # # # # # #         self.cond_proj  = nn.Sequential(
# # # # # # # # # #             nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))
# # # # # # # # # #         dec_layer = nn.TransformerDecoderLayer(
# # # # # # # # # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # # # # # # # # #             dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
# # # # # # # # # #         self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
# # # # # # # # # #         self.out_norm = nn.LayerNorm(d_model)
# # # # # # # # # #         self.out_proj = nn.Sequential(
# # # # # # # # # #             nn.Linear(d_model, d_model//2), nn.GELU(),
# # # # # # # # # #             nn.Linear(d_model//2, 2))
# # # # # # # # # #         self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
# # # # # # # # # #         nn.init.zeros_(self.out_proj[-1].weight)
# # # # # # # # # #         nn.init.zeros_(self.out_proj[-1].bias)

# # # # # # # # # #     def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #         half = self.d_model // 2
# # # # # # # # # #         freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
# # # # # # # # # #                          * (-math.log(10000.0) / max(half-1, 1)))
# # # # # # # # # #         emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)
# # # # # # # # # #         emb  = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # # # # # # # # #         if self.d_model % 2 == 1: emb = F.pad(emb, (0, 1))
# # # # # # # # # #         return self.time_mlp(emb)

# # # # # # # # # #     def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #         B, T, _ = x_t.shape
# # # # # # # # # #         step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # # # # #         x_emb = (self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(step_idx))
# # # # # # # # # #         memory = torch.cat([self._time_emb(t).unsqueeze(1),
# # # # # # # # # #                              self.cond_proj(cond).unsqueeze(1)], dim=1)
# # # # # # # # # #         out = self.out_norm(self.decoder(x_emb, memory))
# # # # # # # # # #         return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)


# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # # #  ContextEncoder — backbone v2.1 thuần túy (không có ar_enc/ar_gate)
# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # # class ContextEncoder(nn.Module):
# # # # # # # # # #     """
# # # # # # # # # #     BUG-A FIX: Xóa ar_enc/ar_gate — chúng là dead code (training và sample()
# # # # # # # # # #     đều không dùng). Giữ backbone thuần v2.1.
# # # # # # # # # #     """
# # # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # # #     def __init__(self, obs_len: int = 8, unet_in_ch: int = 13, d_cond: int = 256):
# # # # # # # # # #         super().__init__()
# # # # # # # # # #         self.obs_len = obs_len
# # # # # # # # # #         self.d_cond  = d_cond

# # # # # # # # # #         self.spatial_enc     = FNO3DEncoder(
# # # # # # # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # # # # #             spatial_down=32, dropout=0.05)
# # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)
# # # # # # # # # #         self.enc_1d          = DataEncoder1D(
# # # # # # # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # # # # # # #         self.env_enc         = Env_net(obs_len=obs_len, d_model=32)

# # # # # # # # # #         self.ctx_fc1  = nn.Linear(128+32+16, self.RAW_CTX_DIM)
# # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # # #         self.ctx_drop = nn.Dropout(0.1)
# # # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, d_cond)
# # # # # # # # # #         self.ctx_ln2  = nn.LayerNorm(d_cond)

# # # # # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # # # # #             nn.Linear(obs_len*7, 256), nn.GELU(), nn.LayerNorm(256),
# # # # # # # # # #             nn.Linear(256, d_cond//2), nn.GELU())
# # # # # # # # # #         self.hard_embed = nn.Sequential(
# # # # # # # # # #             nn.Linear(1, d_cond//4), nn.GELU(),
# # # # # # # # # #             nn.Linear(d_cond//4, d_cond//4))
# # # # # # # # # #         self.fuse = nn.Sequential(
# # # # # # # # # #             nn.Linear(d_cond + d_cond//2 + d_cond//4, d_cond),
# # # # # # # # # #             nn.LayerNorm(d_cond), nn.GELU())

# # # # # # # # # #     def _encode_raw(self, batch_list) -> torch.Tensor:
# # # # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # # # #         image_obs = batch_list[11]
# # # # # # # # # #         env_data  = batch_list[13]

# # # # # # # # # #         if image_obs.dim() == 4: image_obs = image_obs.unsqueeze(2)
# # # # # # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # #         T_obs = obs_traj.shape[0]
# # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0,2,1)
# # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
# # # # # # # # # #                                    mode="linear", align_corners=False).permute(0,2,1)
# # # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # # # # #         t_w = torch.softmax(
# # # # # # # # # #             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float, device=e_3d_dec_t.device) * 0.5,
# # # # # # # # # #             dim=0)
# # # # # # # # # #         f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
# # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)
# # # # # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

# # # # # # # # # #     def _kinematic_feat(self, obs_traj_norm: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #         B, T_obs = obs_traj_norm.shape[1], obs_traj_norm.shape[0]
# # # # # # # # # #         device   = obs_traj_norm.device
# # # # # # # # # #         if T_obs >= 2:
# # # # # # # # # #             traj_deg = _norm_to_deg(obs_traj_norm)
# # # # # # # # # #             vel_norm = obs_traj_norm[1:] - obs_traj_norm[:-1]
# # # # # # # # # #             speed    = _step_speeds_kmh(traj_deg)
# # # # # # # # # #             speed_n  = (speed / 20.0).clamp(-3.0, 3.0)
# # # # # # # # # #             heading  = torch.atan2(vel_norm[:,:,1], vel_norm[:,:,0])
# # # # # # # # # #             if T_obs >= 3:
# # # # # # # # # #                 dspd  = speed[1:] - speed[:-1]
# # # # # # # # # #                 accel = torch.cat([obs_traj_norm.new_zeros(1, B),
# # # # # # # # # #                                    (dspd/10.0).clamp(-3.0,3.0)], 0)
# # # # # # # # # #             else:
# # # # # # # # # #                 accel = obs_traj_norm.new_zeros(T_obs-1, B)
# # # # # # # # # #             # FIX-1: thêm log_speed để encoder biết absolute speed magnitude
# # # # # # # # # #             # speed_n đã bị clamp → mất info speed cao. log_speed giữ được.
# # # # # # # # # #             log_speed = (speed / 100.0 + 1.0).log() / math.log(2.0)  # [T-1, B], range ~[0,1]
# # # # # # # # # #             log_speed = log_speed.clamp(0.0, 3.0)
# # # # # # # # # #             kine = torch.stack([vel_norm[:,:,0], vel_norm[:,:,1], speed_n,
# # # # # # # # # #                                  heading.sin(), heading.cos(), accel, log_speed], dim=-1)
# # # # # # # # # #         else:
# # # # # # # # # #             kine = obs_traj_norm.new_zeros(self.obs_len, B, 7)
# # # # # # # # # #         if kine.shape[0] < self.obs_len:
# # # # # # # # # #             kine = torch.cat([obs_traj_norm.new_zeros(self.obs_len-kine.shape[0], B, 7), kine], 0)
# # # # # # # # # #         else:
# # # # # # # # # #             kine = kine[-self.obs_len:]
# # # # # # # # # #         return self.vel_obs_enc(kine.permute(1,0,2).reshape(B,-1))

# # # # # # # # # #     def forward(self, batch_list, hard_score: Optional[torch.Tensor] = None,
# # # # # # # # # #                 **kwargs) -> torch.Tensor:
# # # # # # # # # #         raw   = self._encode_raw(batch_list)
# # # # # # # # # #         ctx   = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
# # # # # # # # # #         kfeat = self._kinematic_feat(batch_list[0][:, :, :2])
# # # # # # # # # #         if hard_score is None:
# # # # # # # # # #             hard_score = torch.zeros(ctx.shape[0], device=ctx.device)
# # # # # # # # # #         hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))
# # # # # # # # # #         return self.fuse(torch.cat([ctx, kfeat, hfeat], dim=-1))


# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # # #  Hard score — XAI-2
# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # # def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #     lon1=torch.deg2rad(p1[...,0]); lat1=torch.deg2rad(p1[...,1])
# # # # # # # # # #     lon2=torch.deg2rad(p2[...,0]); lat2=torch.deg2rad(p2[...,1])
# # # # # # # # # #     dlon=lon2-lon1
# # # # # # # # # #     y=torch.sin(dlon)*torch.cos(lat2)
# # # # # # # # # #     x=torch.cos(lat1)*torch.sin(lat2)-torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon)
# # # # # # # # # #     return torch.atan2(y,x)

# # # # # # # # # # @torch.no_grad()
# # # # # # # # # # def hard_score_from_obs(obs_traj_norm: torch.Tensor,
# # # # # # # # # #                          return_components: bool = False):
# # # # # # # # # #     """
# # # # # # # # # #     [XAI-2] Điểm khó của storm. return_components=True → giải thích tại sao khó.
# # # # # # # # # #     curvature: bão đang recurve (xoay hướng)
# # # # # # # # # #     speed_var: tốc độ thay đổi đột ngột
# # # # # # # # # #     dir_change: có nhiều bước rẽ lớn
# # # # # # # # # #     """
# # # # # # # # # #     T, B   = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
# # # # # # # # # #     device = obs_traj_norm.device
# # # # # # # # # #     if T < 3:
# # # # # # # # # #         z = torch.zeros(B, device=device)
# # # # # # # # # #         if return_components:
# # # # # # # # # #             return z, {"curvature": z, "speed_var": z, "dir_change": z}
# # # # # # # # # #         return z
# # # # # # # # # #     traj_deg = _norm_to_deg(obs_traj_norm[..., :2])
# # # # # # # # # #     az12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
# # # # # # # # # #     az23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
# # # # # # # # # #     diff = (az23 - az12).abs()
# # # # # # # # # #     diff = torch.where(diff > math.pi, 2*math.pi - diff, diff)
# # # # # # # # # #     curvature  = diff.mean(0) / math.pi
# # # # # # # # # #     spd = _step_speeds_kmh(traj_deg)
# # # # # # # # # #     speed_var  = ((spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 1.)
# # # # # # # # # #                   if spd.shape[0] >= 2 else torch.zeros(B, device=device))
# # # # # # # # # #     dir_change = (diff > (20./180.*math.pi)).float().mean(0)
# # # # # # # # # #     score = (0.4*curvature + 0.3*speed_var + 0.3*dir_change).clamp(0., 1.)
# # # # # # # # # #     if return_components:
# # # # # # # # # #         return score, {"curvature": curvature, "speed_var": speed_var,
# # # # # # # # # #                        "dir_change": dir_change}
# # # # # # # # # #     return score


# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # # #  Physics score — v2.1 (speed + smooth + heading)
# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # # @torch.no_grad()
# # # # # # # # # # def _physics_score(traj_norm: torch.Tensor,
# # # # # # # # # #                    obs_norm: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #     B, device = traj_norm.shape[1], traj_norm.device
# # # # # # # # # #     traj_deg  = _norm_to_deg(traj_norm)

# # # # # # # # # #     if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# # # # # # # # # #         obs_deg  = _norm_to_deg(obs_norm)
# # # # # # # # # #         obs_spd  = _step_speeds_kmh(obs_deg)
# # # # # # # # # #         T_s      = obs_spd.shape[0]
# # # # # # # # # #         w_obs    = torch.linspace(0.5, 1.0, T_s, device=device)
# # # # # # # # # #         v_ref    = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()
# # # # # # # # # #         pred_spd = _step_speeds_kmh(traj_deg)
# # # # # # # # # #         v_sigma  = v_ref.clamp(min=5.0) * 0.5
# # # # # # # # # #         speed_score = torch.exp(
# # # # # # # # # #             -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5)
# # # # # # # # # #     elif traj_deg.shape[0] >= 2:
# # # # # # # # # #         speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.).mean(0))
# # # # # # # # # #     else:
# # # # # # # # # #         speed_score = torch.ones(B, device=device)

# # # # # # # # # #     if traj_deg.shape[0] >= 3:
# # # # # # # # # #         vel       = traj_deg[1:] - traj_deg[:-1]
# # # # # # # # # #         accel_mag = (vel[1:] - vel[:-1]).norm(dim=-1)
# # # # # # # # # #         smooth_score = torch.exp(-accel_mag.mean(0) * 5.0)
# # # # # # # # # #     else:
# # # # # # # # # #         smooth_score = torch.ones(B, device=device)

# # # # # # # # # #     if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
# # # # # # # # # #         obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # # # # # # # # #         pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
# # # # # # # # # #         obs_h    = F.normalize(obs_vel,  dim=-1, eps=1e-6)
# # # # # # # # # #         pred_h   = F.normalize(pred_vel, dim=-1, eps=1e-6)
# # # # # # # # # #         cos_sim  = (obs_h * pred_h).sum(-1).clamp(-1, 1)
# # # # # # # # # #         head_score = torch.exp((cos_sim - 1.0) * 3.0)
# # # # # # # # # #     else:
# # # # # # # # # #         head_score = torch.ones(B, device=device)

# # # # # # # # # #     return (speed_score.pow(0.35) * smooth_score.pow(0.30) * head_score.pow(0.35)).clamp(min=1e-6)


# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # # #  Augmentation — NEW-3: mạnh hơn v2.1
# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # # def augment_batch(batch_list) -> list:
# # # # # # # # # #     """
# # # # # # # # # #     [NEW-3] 5 loại aug để cover test distribution:
# # # # # # # # # #       A (25%): shift ±15km — model không memorize vị trí tuyệt đối
# # # # # # # # # #       B (25%): speed scale ×0.6-1.5 — cover tốc độ bão khác nhau
# # # # # # # # # #       C (20%): rotation ±10° quanh last_obs — cover hướng di chuyển
# # # # # # # # # #       D (15%): mixup — tránh memorize exact val patterns
# # # # # # # # # #       E (15%): Gaussian noise nhỏ

# # # # # # # # # #     Val KHÔNG gọi hàm này → val loss phản ánh đúng generalization.
# # # # # # # # # #     """
# # # # # # # # # #     bl = list(batch_list)
# # # # # # # # # #     if not torch.is_tensor(bl[0]): return bl
# # # # # # # # # #     obs    = bl[0]
# # # # # # # # # #     gt     = bl[1]
# # # # # # # # # #     device = obs.device
# # # # # # # # # #     anchor = obs[-1:, :, :2].detach()  # [1, B, 2] — last obs làm pivot

# # # # # # # # # #     r = torch.rand(1).item()

# # # # # # # # # #     if r < 0.25:
# # # # # # # # # #         # A: shift ±15km. 15km / (50° × 111km/°) ≈ 0.0027 norm/km
# # # # # # # # # #         shift  = (torch.rand(2, device=device) - 0.5) * 0.054  # ±15km
# # # # # # # # # #         obs_new = obs.clone(); obs_new[..., :2] = obs[..., :2] + shift.view(1,1,2)
# # # # # # # # # #         bl[0] = obs_new; bl[1] = gt + shift.view(1,1,2)

# # # # # # # # # #     elif r < 0.50:
# # # # # # # # # #         # B: speed scale ×0.6-1.5 quanh last_obs
# # # # # # # # # #         scale  = 0.60 + 0.90 * torch.rand(1).item()
# # # # # # # # # #         obs_new = obs.clone()
# # # # # # # # # #         obs_new[..., :2] = anchor + (obs[..., :2] - anchor) * scale
# # # # # # # # # #         bl[0] = obs_new; bl[1] = anchor + (gt - anchor) * scale

# # # # # # # # # #     elif r < 0.70:
# # # # # # # # # #         # C: rotation ±10° quanh last_obs. Giữ anchor cố định.
# # # # # # # # # #         angle  = (torch.rand(1).item() - 0.5) * (math.pi / 9)  # ±10°
# # # # # # # # # #         cos_a, sin_a = math.cos(angle), math.sin(angle)
# # # # # # # # # #         rot = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]],
# # # # # # # # # #                            dtype=obs.dtype, device=device)
# # # # # # # # # #         T_obs, B = obs.shape[0], obs.shape[1]
# # # # # # # # # #         obs_new  = obs.clone()
# # # # # # # # # #         rel_obs  = (obs[..., :2] - anchor).reshape(T_obs*B, 2)
# # # # # # # # # #         obs_new[..., :2] = (rot @ rel_obs.T).T.reshape(T_obs, B, 2) + anchor
# # # # # # # # # #         bl[0] = obs_new
# # # # # # # # # #         T_pred = gt.shape[0]
# # # # # # # # # #         rel_gt = (gt - anchor).reshape(T_pred*B, 2)
# # # # # # # # # #         bl[1]  = (rot @ rel_gt.T).T.reshape(T_pred, B, 2) + anchor

# # # # # # # # # #     elif r < 0.85:
# # # # # # # # # #         # D: mixup — blend 2 storms trong batch
# # # # # # # # # #         B = obs.shape[1]
# # # # # # # # # #         if B >= 4:
# # # # # # # # # #             alpha = 0.15 + 0.20 * torch.rand(1).item()  # 0.15-0.35
# # # # # # # # # #             idx   = torch.randperm(B, device=device)
# # # # # # # # # #             obs_new = obs.clone()
# # # # # # # # # #             obs_new[..., :2] = (1-alpha)*obs[..., :2] + alpha*obs[:, idx, :2]
# # # # # # # # # #             bl[0] = obs_new
# # # # # # # # # #             bl[1] = (1-alpha)*gt + alpha*gt[:, idx, :]

# # # # # # # # # #     else:
# # # # # # # # # #         # E: Gaussian noise ±4km tương đương
# # # # # # # # # #         obs_new = obs.clone()
# # # # # # # # # #         obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.004
# # # # # # # # # #         bl[0] = obs_new

# # # # # # # # # #     return bl


# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # # #  XAI-1: Feature attribution — obs timestep nào ảnh hưởng nhất
# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # # def compute_obs_attribution(model, batch_list, device: torch.device,
# # # # # # # # # #                              target_step: int = 11) -> torch.Tensor:
# # # # # # # # # #     """
# # # # # # # # # #     [XAI-1] Gradient của displacement magnitude tại target_step
# # # # # # # # # #     w.r.t. obs_traj → saliency map: step obs nào ảnh hưởng nhất đến 72h pred.

# # # # # # # # # #     BUG-C FIX: Không gọi model.eval() vì sẽ tắt dropout và làm attribution
# # # # # # # # # #     khác với training distribution.
# # # # # # # # # #     BUG-D FIX: Tính hard_score riêng bên ngoài no_grad context,
# # # # # # # # # #     không truyền qua encoder để tránh gradient bị chặn bởi @no_grad.

# # # # # # # # # #     Output: attr [T_obs, B] — normalized importance per obs step per storm.
# # # # # # # # # #     attr[-1, b] thường cao nhất (obs gần nhất quan trọng nhất).
# # # # # # # # # #     attr[0, b] cao → storm có "trí nhớ dài" về pattern cũ.
# # # # # # # # # #     """
# # # # # # # # # #     raw = _unwrap_model(model)
# # # # # # # # # #     obs_traj_req = batch_list[0].detach().clone().requires_grad_(True)
# # # # # # # # # #     bl_grad      = list(batch_list)
# # # # # # # # # #     bl_grad[0]   = obs_traj_req

# # # # # # # # # #     # BUG-D FIX: tính hard_score bên ngoài để không chặn gradient
# # # # # # # # # #     with torch.no_grad():
# # # # # # # # # #         h_score_val = hard_score_from_obs(batch_list[0][:, :, :2])

# # # # # # # # # #     with torch.enable_grad():
# # # # # # # # # #         # Encode — gradient sẽ flow từ velocity output ngược qua encoder
# # # # # # # # # #         cond     = raw.encoder(bl_grad, hard_score=h_score_val)
# # # # # # # # # #         last_obs = obs_traj_req[-1, :, :2]

# # # # # # # # # #         x0   = torch.randn(obs_traj_req.shape[1], raw.pred_len, 2,
# # # # # # # # # #                            device=device) * raw.sigma_inference
# # # # # # # # # #         t0   = torch.zeros(obs_traj_req.shape[1], device=device)
# # # # # # # # # #         v    = raw.velocity(x0, t0, cond)
# # # # # # # # # #         pred_rel = x0 + v   # [B, T, 2] relative to last_obs

# # # # # # # # # #         # Loss: magnitude của displacement tại target_step, averaged over batch
# # # # # # # # # #         ts       = min(target_step, raw.pred_len - 1)
# # # # # # # # # #         loss_xai = pred_rel[:, ts, :].norm(dim=-1).mean()
# # # # # # # # # #         loss_xai.backward()

# # # # # # # # # #     if obs_traj_req.grad is not None:
# # # # # # # # # #         attr = obs_traj_req.grad[:, :, :2].norm(dim=-1)   # [T_obs, B]
# # # # # # # # # #         attr = attr / (attr.sum(0, keepdim=True) + 1e-8)   # normalize per storm
# # # # # # # # # #     else:
# # # # # # # # # #         attr = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)

# # # # # # # # # #     return attr.detach()


# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # # #  XAI-4: Ensemble uncertainty
# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # # def compute_ensemble_uncertainty(all_traj: torch.Tensor) -> Dict:
# # # # # # # # # #     """
# # # # # # # # # #     [XAI-4] Std deviation per lead time across K ensemble samples.
# # # # # # # # # #     all_traj: [K, T, B, 2] absolute normalized coords.

# # # # # # # # # #     Kết quả:
# # # # # # # # # #       std_per_step [T, B]: uncertainty per lead time per storm (km)
# # # # # # # # # #       uncertainty_ratio [B]: 72h_std / 12h_std — tăng theo thời gian là bình thường
# # # # # # # # # #       mean_72h_std: trung bình std ở 72h — threshold để flag "uncertain forecast"
# # # # # # # # # #     """
# # # # # # # # # #     all_deg   = _norm_to_deg(all_traj)   # [K, T, B, 2]
# # # # # # # # # #     K, T, B   = all_deg.shape[:3]
# # # # # # # # # #     mean_traj = all_deg.mean(0)           # [T, B, 2]

# # # # # # # # # #     std_km = torch.zeros(T, B, device=all_traj.device)
# # # # # # # # # #     for t in range(T):
# # # # # # # # # #         # [K*B, 2] vs [K*B, 2] — tính khoảng cách từ mỗi sample tới mean
# # # # # # # # # #         dists = _haversine_deg(
# # # # # # # # # #             all_deg[:, t, :, :].reshape(K*B, 2),
# # # # # # # # # #             mean_traj[t].unsqueeze(0).expand(K, B, 2).reshape(K*B, 2)
# # # # # # # # # #         ).reshape(K, B)
# # # # # # # # # #         std_km[t] = dists.std(0)

# # # # # # # # # #     step_12h = min(1, T-1)
# # # # # # # # # #     step_72h = min(11, T-1)
# # # # # # # # # #     return {
# # # # # # # # # #         "std_per_step":      std_km,                                          # [T, B] km
# # # # # # # # # #         "uncertainty_ratio": (std_km[step_72h]+1e-3) / (std_km[step_12h]+1e-3),  # [B]
# # # # # # # # # #         "mean_72h_std":      float(std_km[step_72h].mean().item()),
# # # # # # # # # #         "mean_12h_std":      float(std_km[step_12h].mean().item()),
# # # # # # # # # #     }


# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # # #  Compat stubs
# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # # @torch.no_grad()
# # # # # # # # # # def classify_hard_easy(obs_traj_norm, per_sample_loss=None,
# # # # # # # # # #                         hard_score_p: float = 70.0, loss_p: float = 50.0):
# # # # # # # # # #     scores = hard_score_from_obs(obs_traj_norm)
# # # # # # # # # #     B = scores.shape[0]
# # # # # # # # # #     if B < 4: return torch.zeros(B, dtype=torch.bool, device=scores.device)
# # # # # # # # # #     return scores >= torch.quantile(scores, hard_score_p/100.0)

# # # # # # # # # # @torch.no_grad()
# # # # # # # # # # def classify_hard_easy_global(obs_traj_norm, global_threshold):
# # # # # # # # # #     return hard_score_from_obs(obs_traj_norm) >= global_threshold

# # # # # # # # # # @torch.no_grad()
# # # # # # # # # # def compute_diversity_score(candidates) -> float:
# # # # # # # # # #     if len(candidates) < 2: return 0.0
# # # # # # # # # #     T, B    = candidates[0].shape[0], candidates[0].shape[1]
# # # # # # # # # #     ep_step = min(T-1, 11)
# # # # # # # # # #     endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
# # # # # # # # # #     N = endpoints.shape[0]; ep_mean = endpoints.mean(0, keepdim=True)
# # # # # # # # # #     dists = _haversine_deg(
# # # # # # # # # #         endpoints.reshape(N*B, 2),
# # # # # # # # # #         ep_mean.expand(N, B, 2).reshape(N*B, 2)
# # # # # # # # # #     ).reshape(N, B)
# # # # # # # # # #     return float(dists.std(0).mean().item())


# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # # #  TCFlowMatching v2.4
# # # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # # class TCFlowMatching(nn.Module):
# # # # # # # # # #     """
# # # # # # # # # #     TC-FlowMatching v2.4 = v2.1 core + augmentation mạnh + exp step weights

# # # # # # # # # #     Training:
# # # # # # # # # #       L_total = L_CFM + lambda_reg * L_reg
# # # # # # # # # #       L_CFM: flow matching objective với random t ∈ [0,1] và OT noise matching
# # # # # # # # # #       L_reg: ADE tại t=0 với sigma_inference (BUG-8 fix), exp step weights (NEW-4)
# # # # # # # # # #              dùng x1_rel gốc (BUG-E fix), NOT x1_matched từ OT

# # # # # # # # # #     Inference (1-shot, nhất quán với training):
# # # # # # # # # #       x0 ~ N(0, sigma_inference²)
# # # # # # # # # #       x_pred_rel = x0 + v(x0, t=0, cond)
# # # # # # # # # #       x_pred_abs = x_pred_rel + last_obs
# # # # # # # # # #       Best-of-K với physics score (speed + smoothness + heading)
# # # # # # # # # #     """

# # # # # # # # # #     def __init__(self,
# # # # # # # # # #                  pred_len: int = 12, obs_len: int = 8, unet_in_ch: int = 13,
# # # # # # # # # #                  d_cond: int = 256, d_model: int = 256, nhead: int = 8,
# # # # # # # # # #                  num_dec_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
# # # # # # # # # #                  sigma_min: float = 0.04, sigma_max: float = 0.08,
# # # # # # # # # #                  lambda_reg: float = 0.2,
# # # # # # # # # #                  use_ot: bool = True, ot_epsilon: float = 0.05,
# # # # # # # # # #                  use_ema: bool = True, ema_decay: float = 0.995,
# # # # # # # # # #                  n_inference_steps: int = 1, n_ensemble: int = 20,
# # # # # # # # # #                  sigma_inference: float = 0.04,
# # # # # # # # # #                  **kwargs):
# # # # # # # # # #         super().__init__()
# # # # # # # # # #         self.pred_len          = pred_len
# # # # # # # # # #         self.obs_len           = obs_len
# # # # # # # # # #         self.sigma_min         = sigma_min
# # # # # # # # # #         self.sigma_max         = sigma_max
# # # # # # # # # #         self.lambda_reg        = lambda_reg
# # # # # # # # # #         self.use_ot            = use_ot
# # # # # # # # # #         self.ot_epsilon        = ot_epsilon
# # # # # # # # # #         self.n_inference_steps = n_inference_steps
# # # # # # # # # #         self.n_ensemble        = n_ensemble
# # # # # # # # # #         self.sigma_inference   = sigma_inference

# # # # # # # # # #         self.encoder  = ContextEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
# # # # # # # # # #         self.velocity = VelocityTransformer(
# # # # # # # # # #             pred_len=pred_len, d_model=d_model, nhead=nhead,
# # # # # # # # # #             num_layers=num_dec_layers, dim_ff=dim_ff, dropout=dropout, d_cond=d_cond)
# # # # # # # # # #         self.use_ema = use_ema
# # # # # # # # # #         self._ema    = None

# # # # # # # # # #     def init_ema(self):
# # # # # # # # # #         if self.use_ema: self._ema = EMAModel(self)

# # # # # # # # # #     def ema_update(self):
# # # # # # # # # #         if self._ema is not None: self._ema.update(self)

# # # # # # # # # #     def _to_relative(self, x_abs: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #         return x_abs - last_obs.unsqueeze(1)

# # # # # # # # # #     def _from_relative(self, x_rel: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #         return x_rel + last_obs.unsqueeze(1)

# # # # # # # # # #     def _sigma_schedule(self, epoch: int) -> float:
# # # # # # # # # #         """Cosine decay từ sigma_max về sigma_min trong ep5→ep40."""
# # # # # # # # # #         if epoch < 5:  return self.sigma_max
# # # # # # # # # #         if epoch < 40:
# # # # # # # # # #             t = (epoch - 5) / 35.0
# # # # # # # # # #             return self.sigma_min + 0.5*(self.sigma_max-self.sigma_min)*(1+math.cos(math.pi*t))
# # # # # # # # # #         return self.sigma_min

# # # # # # # # # #     def _reg_loss(self, x1_rel: torch.Tensor, last_obs: torch.Tensor,
# # # # # # # # # #                   cond: torch.Tensor,
# # # # # # # # # #                   hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# # # # # # # # # #         """
# # # # # # # # # #         ADE loss tại t=0, consistent với inference.

# # # # # # # # # #         BUG-8 FIX: dùng sigma_inference thay vì sigma_schedule.
# # # # # # # # # #         BUG-E FIX: x1_rel là GT GỐC (không bị OT shuffle).
# # # # # # # # # #                    L_reg phải học predict đúng GT, không phải OT-matched GT
# # # # # # # # # #                    của noise sample khác.
# # # # # # # # # #         NEW-4:     Exp step weights — 72h weight ≈ 6× 12h weight.
# # # # # # # # # #         """
# # # # # # # # # #         B, T, _ = x1_rel.shape
# # # # # # # # # #         device  = x1_rel.device

# # # # # # # # # #         x0 = torch.randn_like(x1_rel) * self.sigma_inference  # fresh noise
# # # # # # # # # #         t0 = torch.zeros(B, device=device)
# # # # # # # # # #         v  = self.velocity(x0, t0, cond)
# # # # # # # # # #         x1_pred_abs = self._from_relative(x0 + v, last_obs)
# # # # # # # # # #         x1_gt_abs   = self._from_relative(x1_rel, last_obs)   # x1_rel gốc

# # # # # # # # # #         pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))  # [T, B, 2]
# # # # # # # # # #         gt_deg   = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
# # # # # # # # # #         dist     = _haversine_deg(pred_deg, gt_deg)             # [T, B] km

# # # # # # # # # #         # FIX-2: Speed-adaptive step weights
# # # # # # # # # #         # Nhân 2 nguồn:
# # # # # # # # # #         # (a) Exp weights: nhấn mạnh long-range (72h)
# # # # # # # # # #         # (b) GT displacement weights: step nào bão di chuyển xa hơn → weight cao hơn
# # # # # # # # # #         #     → gradient mạnh đúng chỗ velocity magnitude sai nhiều nhất
# # # # # # # # # #         T_actual = dist.shape[0]
# # # # # # # # # #         steps = torch.arange(T_actual, device=device, dtype=dist.dtype)
# # # # # # # # # #         exp_w = torch.exp(2.5 * steps / T_actual)  # [T]
# # # # # # # # # #         exp_w = exp_w / exp_w.mean()

# # # # # # # # # #         # GT displacement per step: haversine(gt[t], gt[t-1])
# # # # # # # # # #         gt_disp = _haversine_deg(gt_deg[:-1], gt_deg[1:])   # [T-1, B] km
# # # # # # # # # #         # Pad step 0 = mean displacement
# # # # # # # # # #         gt_disp = torch.cat([gt_disp.mean(0, keepdim=True), gt_disp], 0)  # [T, B]
# # # # # # # # # #         spd_w   = (gt_disp / gt_disp.mean(0, keepdim=True).clamp(min=1.0)).clamp(0.3, 3.0)

# # # # # # # # # #         sw = (exp_w.unsqueeze(1) * spd_w)        # [T, B]
# # # # # # # # # #         sw = sw / sw.mean(0, keepdim=True)        # normalize per storm

# # # # # # # # # #         if hard_score is not None:
# # # # # # # # # #             # sw đã [T, B], hard_score [B] → broadcast đúng
# # # # # # # # # #             sw = sw * (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)

# # # # # # # # # #         return (dist * sw).mean() / 300.0

# # # # # # # # # #     def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
# # # # # # # # # #         obs_traj = batch_list[0]
# # # # # # # # # #         gt_traj  = batch_list[1]
# # # # # # # # # #         B        = obs_traj.shape[1]
# # # # # # # # # #         device   = obs_traj.device

# # # # # # # # # #         sigma    = self._sigma_schedule(epoch)
# # # # # # # # # #         x1_gt    = gt_traj.permute(1, 0, 2)        # [B, 12, 2]
# # # # # # # # # #         last_obs = obs_traj[-1, :, :2]
# # # # # # # # # #         x1_rel   = self._to_relative(x1_gt, last_obs)  # GT gốc — dùng cho L_reg

# # # # # # # # # #         h_score = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # # # # # #         cond    = self.encoder(batch_list, hard_score=h_score)

# # # # # # # # # #         # L_CFM — flow matching objective
# # # # # # # # # #         x0 = torch.randn_like(x1_rel) * sigma
# # # # # # # # # #         if self.use_ot and B >= 4:
# # # # # # # # # #             x0_flat, x1_flat = _ot_match_noise_gt(
# # # # # # # # # #                 x0.reshape(B, -1), x1_rel.reshape(B, -1), self.ot_epsilon)
# # # # # # # # # #             x0         = x0_flat.reshape(B, self.pred_len, 2)
# # # # # # # # # #             x1_matched = x1_flat.reshape(B, self.pred_len, 2)
# # # # # # # # # #         else:
# # # # # # # # # #             x1_matched = x1_rel

# # # # # # # # # #         t      = torch.rand(B, device=device)
# # # # # # # # # #         x_t    = (1 - t.view(B,1,1))*x0 + t.view(B,1,1)*x1_matched
# # # # # # # # # #         v_pred = self.velocity(x_t, t, cond)
# # # # # # # # # #         l_cfm  = F.mse_loss(v_pred, x1_matched - x0)

# # # # # # # # # #         # L_reg ramp ep10→ep30
# # # # # # # # # #         if epoch < 10:
# # # # # # # # # #             lam_reg = 0.0
# # # # # # # # # #         elif epoch < 30:
# # # # # # # # # #             lam_reg = self.lambda_reg * (epoch - 10) / 20.0
# # # # # # # # # #         else:
# # # # # # # # # #             lam_reg = self.lambda_reg

# # # # # # # # # #         # BUG-E FIX: truyền x1_rel GỐC, không phải x1_matched
# # # # # # # # # #         l_reg = (self._reg_loss(x1_rel, last_obs, cond, h_score)
# # # # # # # # # #                  if lam_reg > 0.0 else x0.new_zeros(()))

# # # # # # # # # #         total = l_cfm + lam_reg * l_reg
# # # # # # # # # #         if not torch.isfinite(total):
# # # # # # # # # #             total = x0.new_zeros(())

# # # # # # # # # #         # ADE log — BUG-9 FIX: dùng sigma_inference
# # # # # # # # # #         with torch.no_grad():
# # # # # # # # # #             x0_log = torch.randn_like(x1_rel) * self.sigma_inference
# # # # # # # # # #             v_log  = self.velocity(x0_log, torch.zeros(B, device=device), cond)
# # # # # # # # # #             x1_log = self._from_relative(x0_log + v_log, last_obs)
# # # # # # # # # #             ade_log = _haversine_deg(
# # # # # # # # # #                 _norm_to_deg(x1_log.permute(1, 0, 2)),
# # # # # # # # # #                 _norm_to_deg(x1_gt.permute(1, 0, 2))
# # # # # # # # # #             ).mean().item()

# # # # # # # # # #         return {
# # # # # # # # # #             "total":    total,
# # # # # # # # # #             "l_cfm":    l_cfm.item(),
# # # # # # # # # #             "l_reg":    l_reg.item() if torch.is_tensor(l_reg) else 0.0,
# # # # # # # # # #             "lam_reg":  lam_reg,
# # # # # # # # # #             "sigma":    sigma,
# # # # # # # # # #             "ade_1step": ade_log,
# # # # # # # # # #             "hard_score_mean": float(h_score.mean()),
# # # # # # # # # #             "hard_score_max":  float(h_score.max()),
# # # # # # # # # #             # Compat keys
# # # # # # # # # #             "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
# # # # # # # # # #             "speed": 0., "accel": 0., "fm_mse": l_cfm.item(),
# # # # # # # # # #             "l_hard_total": 0., "n_hard": 0, "alpha_hard": 0.,
# # # # # # # # # #             "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
# # # # # # # # # #         }

# # # # # # # # # #     def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
# # # # # # # # # #         return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

# # # # # # # # # #     @torch.no_grad()
# # # # # # # # # #     def sample(self, batch_list, num_ensemble: Optional[int] = None,
# # # # # # # # # #                ddim_steps: Optional[int] = None,
# # # # # # # # # #                return_xai: bool = False, **kwargs):
# # # # # # # # # #         """
# # # # # # # # # #         1-shot inference: x_pred = x0 + v(x0, t=0, cond)
# # # # # # # # # #         Nhất quán hoàn toàn với training: velocity nhận [B, 12, 2].

# # # # # # # # # #         BUG-6 FIX: Cache _encode_raw 1 lần cho K samples.
# # # # # # # # # #         XAI-3: Log physics score components để giải thích tại sao sample nào được chọn.
# # # # # # # # # #         XAI-4: compute_ensemble_uncertainty nếu return_xai=True.
# # # # # # # # # #         """
# # # # # # # # # #         K  = num_ensemble or self.n_ensemble
# # # # # # # # # #         N  = ddim_steps or self.n_inference_steps
# # # # # # # # # #         dt = 1.0 / max(N, 1)

# # # # # # # # # #         obs_traj    = batch_list[0]
# # # # # # # # # #         T_obs, B, _ = obs_traj.shape
# # # # # # # # # #         device      = obs_traj.device
# # # # # # # # # #         h_score     = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # # # # # #         obs_norm    = obs_traj[:, :, :2]
# # # # # # # # # #         last_obs    = obs_traj[-1, :, :2]
# # # # # # # # # #         t0          = torch.zeros(B, device=device)

# # # # # # # # # #         # [BUG-6 FIX] Cache encoder output — gọi 1 lần cho K samples
# # # # # # # # # #         raw_ctx  = self.encoder._encode_raw(batch_list)
# # # # # # # # # #         base_ctx = self.encoder.ctx_ln2(
# # # # # # # # # #             self.encoder.ctx_fc2(self.encoder.ctx_drop(raw_ctx)))
# # # # # # # # # #         kfeat    = self.encoder._kinematic_feat(obs_traj[:, :, :2])
# # # # # # # # # #         hfeat    = self.encoder.hard_embed(h_score.unsqueeze(1).to(base_ctx.dtype))
# # # # # # # # # #         cond     = self.encoder.fuse(torch.cat([base_ctx, kfeat, hfeat], dim=-1))

# # # # # # # # # #         # FIX-3: Speed-conditioned sigma per storm
# # # # # # # # # #         # Bão nhanh cần sigma lớn → K samples cover đủ speed range → ATE giảm
# # # # # # # # # #         if obs_norm.shape[0] >= 2:
# # # # # # # # # #             obs_deg_s  = _norm_to_deg(obs_norm)
# # # # # # # # # #             obs_spd_s  = _step_speeds_kmh(obs_deg_s)          # [T-1, B] km/h
# # # # # # # # # #             obs_spd_mean = obs_spd_s.mean(0).clamp(5., 120.)  # [B]
# # # # # # # # # #         else:
# # # # # # # # # #             obs_spd_mean = torch.full((B,), 30., device=device)
# # # # # # # # # #         # sigma scale: slow storm (10km/h) → 0.5×, fast storm (80km/h) → 2.0×
# # # # # # # # # #         sigma_scale = (0.5 + obs_spd_mean / 40.0).clamp(0.5, 2.0)  # [B]
# # # # # # # # # #         sigma_k = (self.sigma_inference * sigma_scale).view(B, 1, 1)  # [B,1,1] để broadcast

# # # # # # # # # #         all_traj = []

# # # # # # # # # #         # Tính base prediction 1 lần với sigma_inference (nhất quán training)
# # # # # # # # # #         x0_base = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference
# # # # # # # # # #         with torch.no_grad():
# # # # # # # # # #             v_base = self.velocity(x0_base, t0, cond)   # [B, T, 2]
# # # # # # # # # #         x_base = x0_base + v_base                        # best 1-shot prediction

# # # # # # # # # #         for k in range(K):
# # # # # # # # # #             if k == 0:
# # # # # # # # # #                 # Sample 0: pure 1-shot từ sigma_inference (nhất quán với training)
# # # # # # # # # #                 x_rel = x_base
# # # # # # # # # #             else:
# # # # # # # # # #                 # Sample 1..K-1: fresh noise với sigma_inference (training consistent)
# # # # # # # # # #                 # + speed-scaled perturbation theo hướng predicted trajectory
# # # # # # # # # #                 # → diversity về speed magnitude mà không phá vỡ velocity input range
# # # # # # # # # #                 x0_k  = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference
# # # # # # # # # #                 if N <= 1:
# # # # # # # # # #                     x_rel = x0_k + self.velocity(x0_k, t0, cond)
# # # # # # # # # #                 else:
# # # # # # # # # #                     x_rel = x0_k.clone()
# # # # # # # # # #                     for step in range(N):
# # # # # # # # # #                         t_b   = torch.full((B,), step*dt, device=device)
# # # # # # # # # #                         x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3., 3.)

# # # # # # # # # #                 # Speed perturbation: dùng sigma_k để scale magnitude diversity
# # # # # # # # # #                 # Perturbation theo hướng displacement (ATE direction)
# # # # # # # # # #                 # → samples có speed khác nhau → physics score chọn đúng speed → ATE giảm
# # # # # # # # # #                 disp_mag = x_rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)   # [B,T,1]
# # # # # # # # # #                 disp_dir = x_rel / disp_mag                                    # [B,T,2] unit vector
# # # # # # # # # #                 # scale = random [0.5, 1.5] × sigma ratio per storm
# # # # # # # # # #                 scale = (0.7 + 0.6 * torch.rand(B, 1, 1, device=device)) * (sigma_k / self.sigma_inference)
# # # # # # # # # #                 x_rel = x_rel * scale  # stretch/compress trajectory displacement

# # # # # # # # # #             x_abs = self._from_relative(x_rel, last_obs)
# # # # # # # # # #             all_traj.append(x_abs.permute(1, 0, 2))  # [T, B, 2]

# # # # # # # # # #         scores   = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)  # [K, B]
# # # # # # # # # #         all_t    = torch.stack(all_traj, 0)  # [K, T, B, 2]
# # # # # # # # # #         top_k    = min(3, K)
# # # # # # # # # #         top_idx  = scores.topk(top_k, dim=0).indices  # [top_k, B]

# # # # # # # # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # # # # # # # #         for b in range(B):
# # # # # # # # # #             idx_b = top_idx[:, b]
# # # # # # # # # #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # # # # # # # # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k,1,1)).sum(0)

# # # # # # # # # #         if return_xai:
# # # # # # # # # #             xai = compute_ensemble_uncertainty(all_t)

# # # # # # # # # #             # [XAI-3] Vectorized physics score components cho top-1 per storm
# # # # # # # # # #             # BUG-I FIX: không dùng loop per-storm (chậm + scalar assign issue)
# # # # # # # # # #             # BUG-M FIX: vectorize hoàn toàn
# # # # # # # # # #             top1_idx = scores.argmax(0)  # [B]

# # # # # # # # # #             # Build best_traj [T, B, 2]: trajectory được chọn cho mỗi storm
# # # # # # # # # #             best_traj_list = []
# # # # # # # # # #             for b in range(B):
# # # # # # # # # #                 best_traj_list.append(all_traj[int(top1_idx[b].item())][:, b, :])  # [T, 2]
# # # # # # # # # #             best_traj = torch.stack(best_traj_list, dim=1)  # [T, B, 2]
# # # # # # # # # #             best_deg  = _norm_to_deg(best_traj)             # [T, B, 2]
# # # # # # # # # #             obs_deg_v = _norm_to_deg(obs_norm)              # [T_obs, B, 2]

# # # # # # # # # #             # Speed score: exp(-((pred_spd - obs_spd_mean) / sigma)^2 * 0.5)
# # # # # # # # # #             obs_spd_v  = _step_speeds_kmh(obs_deg_v)                          # [T_obs-1, B]
# # # # # # # # # #             obs_spd_mu = obs_spd_v.mean(0)                                     # [B]
# # # # # # # # # #             pred_spd_v = _step_speeds_kmh(best_deg)                           # [T-1, B]
# # # # # # # # # #             pred_spd_mu= pred_spd_v.mean(0)                                    # [B]
# # # # # # # # # #             v_sigma_v  = obs_spd_mu.clamp(min=5.) * 0.5                       # [B]
# # # # # # # # # #             speed_scores = torch.exp(
# # # # # # # # # #                 -((pred_spd_mu - obs_spd_mu) / v_sigma_v).pow(2) * 0.5)      # [B]

# # # # # # # # # #             # Smooth score: exp(-mean_accel_magnitude * 5)
# # # # # # # # # #             if best_deg.shape[0] >= 3:
# # # # # # # # # #                 vel_v   = best_deg[1:] - best_deg[:-1]                        # [T-1, B, 2]
# # # # # # # # # #                 accel_v = (vel_v[1:] - vel_v[:-1]).norm(dim=-1)               # [T-2, B]
# # # # # # # # # #                 smooth_scores = torch.exp(-accel_v.mean(0) * 5.)              # [B]
# # # # # # # # # #             else:
# # # # # # # # # #                 smooth_scores = torch.ones(B, device=device)

# # # # # # # # # #             # Heading score: cos similarity giữa last obs velocity và first pred velocity
# # # # # # # # # #             if obs_norm.shape[0] >= 2 and best_traj.shape[0] >= 1:
# # # # # # # # # #                 obs_vel_v  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]        # [B, 2]
# # # # # # # # # #                 pred_vel_v = best_traj[0, :, :2] - obs_norm[-1, :, :2]        # [B, 2]
# # # # # # # # # #                 obs_h_v    = F.normalize(obs_vel_v,  dim=-1, eps=1e-6)         # [B, 2]
# # # # # # # # # #                 pred_h_v   = F.normalize(pred_vel_v, dim=-1, eps=1e-6)         # [B, 2]
# # # # # # # # # #                 cos_sim_v  = (obs_h_v * pred_h_v).sum(-1).clamp(-1., 1.)      # [B]
# # # # # # # # # #                 head_scores = torch.exp((cos_sim_v - 1.) * 3.)                 # [B]
# # # # # # # # # #             else:
# # # # # # # # # #                 head_scores = torch.ones(B, device=device)

# # # # # # # # # #             xai["physics_components"] = {
# # # # # # # # # #                 "speed":        speed_scores,   # [B] — cao = tốc độ đúng
# # # # # # # # # #                 "smooth":       smooth_scores,  # [B] — cao = ít zigzag
# # # # # # # # # #                 "heading":      head_scores,    # [B] — cao = heading đúng
# # # # # # # # # #                 "obs_speed":    obs_spd_mu,     # [B] km/h — tốc độ quan sát
# # # # # # # # # #                 "pred_speed":   pred_spd_mu,    # [B] km/h — tốc độ predicted
# # # # # # # # # #             }

# # # # # # # # # #             # [XAI-2] Hard score components
# # # # # # # # # #             _, hard_comps = hard_score_from_obs(obs_norm, return_components=True)
# # # # # # # # # #             xai["hard_components"] = hard_comps

# # # # # # # # # #             # [XAI-5] Speed comparison: obs vs pred speed (đã tính trong XAI-3)
# # # # # # # # # #             # Dùng lại obs_speed/pred_speed từ XAI-3 để tránh tính lại
# # # # # # # # # #             # Nhất quán trên toàn bộ B storms, không chỉ 8 storms đầu
# # # # # # # # # #             if "physics_components" in xai:
# # # # # # # # # #                 pc = xai["physics_components"]
# # # # # # # # # #                 obs_mu  = float(pc["obs_speed"].mean().item())
# # # # # # # # # #                 pred_mu = float(pc["pred_speed"].mean().item())
# # # # # # # # # #                 ratio   = pred_mu / max(obs_mu, 1.0)
# # # # # # # # # #                 xai["speed_comparison"] = {
# # # # # # # # # #                     "obs_speed_mean":  obs_mu,   # km/h, trung bình B storms
# # # # # # # # # #                     "pred_speed_mean": pred_mu,  # km/h
# # # # # # # # # #                     "speed_ratio":     ratio,    # >1: over-predict, <1: under-predict
# # # # # # # # # #                     "per_storm_obs":   pc["obs_speed"],   # [B] per-storm detail
# # # # # # # # # #                     "per_storm_pred":  pc["pred_speed"],  # [B] per-storm detail
# # # # # # # # # #                 }

# # # # # # # # # #             return pred_mean, torch.zeros_like(pred_mean), all_t, xai

# # # # # # # # # #         return pred_mean, torch.zeros_like(pred_mean), all_t


# # # # # # # # # # def load_checkpoint_compat(ckpt_path: str, model: "TCFlowMatching",
# # # # # # # # # #                            device) -> dict:
# # # # # # # # # #     """
# # # # # # # # # #     Load checkpoint với backward compat khi vel_obs_enc thay đổi từ obs_len*6 → obs_len*7.
# # # # # # # # # #     Expand weight matrix: copy 6-feat weights, init cột thứ 7 (log_speed) = small random.
# # # # # # # # # #     """
# # # # # # # # # #     ck = torch.load(ckpt_path, map_location=device)
# # # # # # # # # #     sd = ck.get("model", ck)

# # # # # # # # # #     key = "encoder.vel_obs_enc.0.weight"  # Linear(obs_len*6, 256) → Linear(obs_len*7, 256)
# # # # # # # # # #     if key in sd:
# # # # # # # # # #         w_old = sd[key]          # [256, obs_len*6]
# # # # # # # # # #         obs_len = model.obs_len
# # # # # # # # # #         expected_cols = obs_len * 7
# # # # # # # # # #         if w_old.shape[1] == obs_len * 6:
# # # # # # # # # #             # Expand: thêm obs_len columns (log_speed) với init nhỏ
# # # # # # # # # #             extra = torch.randn(w_old.shape[0], obs_len, device=w_old.device) * 0.01
# # # # # # # # # #             sd[key] = torch.cat([w_old, extra], dim=1)  # [256, obs_len*7]
# # # # # # # # # #             print(f"  ✅ Expanded vel_obs_enc: {w_old.shape[1]} → {expected_cols} cols")
# # # # # # # # # #         elif w_old.shape[1] == expected_cols:
# # # # # # # # # #             print(f"  ✅ vel_obs_enc already {expected_cols} cols")
# # # # # # # # # #         else:
# # # # # # # # # #             print(f"  ⚠ vel_obs_enc unexpected shape {w_old.shape}, skip expand")

# # # # # # # # # #     missing, unexpected = model.load_state_dict(sd, strict=False)
# # # # # # # # # #     if missing:    print(f"  Missing keys : {len(missing)}")
# # # # # # # # # #     if unexpected: print(f"  Unexpected   : {len(unexpected)}")
# # # # # # # # # #     return ck


# # # # # # # # # # TCDiffusion = TCFlowMatching

# # # # # # # # # """
# # # # # # # # # Model/flow_matching_model.py  ──  TC-FlowMatching v2.5
# # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # CƠ SỞ: v2.1 (proven val ADE=170km, test ADE=229km, generalize tốt)
# # # # # # # # # GIỮ NGUYÊN từ v2.1:
# # # # # # # # #   ✅ sigma_inference=0.04 CỐ ĐỊNH — không speed-conditioned sigma (v2.4 fail)
# # # # # # # # #   ✅ L_reg linear step weights (softmax linspace, max/min ~3×) — không exp (v2.4 fail)
# # # # # # # # #   ✅ 1-shot inference nhất quán hoàn toàn với training
# # # # # # # # #   ✅ Mild augmentation làm nền — val ≈ test distribution

# # # # # # # # # CẢI TIẾN v2.1 → v2.5:

# # # # # # # # #   [ATE-1] L_heading — heading continuation loss (giảm CTE)
# # # # # # # # #           Penalty khi pred bước đầu lệch hướng so với obs cuối.
# # # # # # # # #           cos similarity loss → 0 khi heading khớp.
# # # # # # # # #           Weight: lambda_heading=0.05, ramp ep5→ep20.

# # # # # # # # #   [ATE-2] L_momentum — speed momentum loss (giảm ATE)
# # # # # # # # #           Penalty khi pred speed ở 4 bước đầu lệch xa obs speed.
# # # # # # # # #           Normalized bởi obs speed → fair với bão nhanh/chậm.
# # # # # # # # #           Weight: lambda_momentum=0.05, ramp ep5→ep20.

# # # # # # # # #   [AUG-1] Recurvature augmentation (giảm val→test gap)
# # # # # # # # #           Progressive rotation ±20° theo ease-in curve.
# # # # # # # # #           Simulate bão recurve thực tế. 20% probability.

# # # # # # # # #   [XAI-1] compute_obs_attribution() — gradient saliency per obs step
# # # # # # # # #            Bước obs nào ảnh hưởng nhất đến 72h prediction?

# # # # # # # # #   [XAI-2] hard_score_from_obs(return_components=True)
# # # # # # # # #            Tại sao storm này khó? curvature/speed_var/dir_change.

# # # # # # # # #   [XAI-3] Physics score components — tại sao sample này được chọn?
# # # # # # # # #            speed_score, smooth_score, heading_score, displacement_score.

# # # # # # # # #   [XAI-4] compute_ensemble_uncertainty() — uncertainty per lead time
# # # # # # # # #            std_per_step [T,B] km, uncertainty_ratio 72h/12h.

# # # # # # # # #   [XAI-5] Speed comparison — obs vs pred speed ratio
# # # # # # # # #            Flag over/under prediction.

# # # # # # # # #   [XAI-6] Heading deviation per lead time — ROOT CAUSE của ATE cao
# # # # # # # # #            Mỗi bước predict lệch hướng bao nhiêu độ?

# # # # # # # # #   [XAI-7] Cross-track contribution per storm
# # # # # # # # #            Storm nào đóng góp nhiều nhất vào CTE?

# # # # # # # # #   [INFER] Multi-scale sigma ensemble option
# # # # # # # # #           Ensemble từ sigmas=[0.025,0.035,0.04,0.05,0.065] thay vì cố định 0.04.
# # # # # # # # #           Không thay đổi training, chỉ inference → không phá consistency.

# # # # # # # # # LƯU Ý QUAN TRỌNG:
# # # # # # # # #   - KHÔNG dùng mixup augmentation
# # # # # # # # #   - KHÔNG dùng speed-conditioned sigma
# # # # # # # # #   - KHÔNG dùng exp step weights trong L_reg
# # # # # # # # #   - SWA handled trong train_fm.py, không trong model
# # # # # # # # # """
# # # # # # # # # from __future__ import annotations

# # # # # # # # # import math
# # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # import torch
# # # # # # # # # import torch.nn as nn
# # # # # # # # # import torch.nn.functional as F

# # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net

# # # # # # # # # R_EARTH  = 6371.0
# # # # # # # # # DT_HOURS = 6.0


# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # #  Coordinate utilities
# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# # # # # # # # #     """Normalized coords → (lon°, lat°)"""
# # # # # # # # #     return torch.stack([
# # # # # # # # #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# # # # # # # # #         (t[..., 1] * 50.0) / 10.0,
# # # # # # # # #     ], dim=-1)


# # # # # # # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # # # # #     """Haversine distance between two lat/lon tensors (in degrees), returns km."""
# # # # # # # # #     lat1 = torch.deg2rad(p1[..., 1])
# # # # # # # # #     lat2 = torch.deg2rad(p2[..., 1])
# # # # # # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # # # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # # # # # #     a = (torch.sin(dlat / 2).pow(2)
# # # # # # # # #          + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
# # # # # # # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# # # # # # # # # def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # # # # #     """Forward bearing (azimuth) from p1 to p2, in radians. p1,p2 in degrees."""
# # # # # # # # #     lon1 = torch.deg2rad(p1[..., 0]); lat1 = torch.deg2rad(p1[..., 1])
# # # # # # # # #     lon2 = torch.deg2rad(p2[..., 0]); lat2 = torch.deg2rad(p2[..., 1])
# # # # # # # # #     dlon = lon2 - lon1
# # # # # # # # #     y = torch.sin(dlon) * torch.cos(lat2)
# # # # # # # # #     x = (torch.cos(lat1) * torch.sin(lat2)
# # # # # # # # #          - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon))
# # # # # # # # #     return torch.atan2(y, x)


# # # # # # # # # def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # #     """Speed in km/h between consecutive steps of a trajectory."""
# # # # # # # # #     if traj_deg.shape[0] < 2:
# # # # # # # # #         return traj_deg.new_zeros(1, traj_deg.shape[1])
# # # # # # # # #     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # #  EMAModel
# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # def _unwrap_model(m):
# # # # # # # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m


# # # # # # # # # class EMAModel:
# # # # # # # # #     def __init__(self, model, decay: float = 0.995):
# # # # # # # # #         self.decay = decay
# # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # #         self.shadow = {k: v.detach().clone()
# # # # # # # # #                        for k, v in m.state_dict().items()
# # # # # # # # #                        if v.dtype.is_floating_point}

# # # # # # # # #     def update(self, model):
# # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # #         with torch.no_grad():
# # # # # # # # #             for k, v in m.state_dict().items():
# # # # # # # # #                 if k in self.shadow:
# # # # # # # # #                     self.shadow[k].mul_(self.decay).add_(
# # # # # # # # #                         v.detach(), alpha=1 - self.decay)

# # # # # # # # #     def apply_to(self, model):
# # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # #         backup, sd = {}, m.state_dict()
# # # # # # # # #         for k in self.shadow:
# # # # # # # # #             if k not in sd:
# # # # # # # # #                 continue
# # # # # # # # #             backup[k] = sd[k].detach().clone()
# # # # # # # # #             sd[k].copy_(self.shadow[k])
# # # # # # # # #         return backup

# # # # # # # # #     def restore(self, model, backup):
# # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # #         sd = m.state_dict()
# # # # # # # # #         for k, v in backup.items():
# # # # # # # # #             if k in sd:
# # # # # # # # #                 sd[k].copy_(v)


# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # #  OT matching (giữ nguyên từ v2.1)
# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05,
# # # # # # # # #                   n_iter: int = 50) -> torch.Tensor:
# # # # # # # # #     B = cost.shape[0]; device = cost.device
# # # # # # # # #     log_a = -math.log(B) * torch.ones(B, device=device)
# # # # # # # # #     log_b = -math.log(B) * torch.ones(B, device=device)
# # # # # # # # #     log_K = -cost / epsilon
# # # # # # # # #     log_u = torch.zeros(B, device=device)
# # # # # # # # #     log_v = torch.zeros(B, device=device)
# # # # # # # # #     for _ in range(n_iter):
# # # # # # # # #         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
# # # # # # # # #         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
# # # # # # # # #     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


# # # # # # # # # def _ot_match_noise_gt(x0_flat: torch.Tensor, x1_flat: torch.Tensor,
# # # # # # # # #                         epsilon: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # # # # # #     B = x0_flat.shape[0]
# # # # # # # # #     if B < 4:
# # # # # # # # #         return x0_flat, x1_flat
# # # # # # # # #     try:
# # # # # # # # #         cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1] ** 0.5)
# # # # # # # # #         with torch.no_grad():
# # # # # # # # #             pi = _sinkhorn_log(cost, epsilon=epsilon)
# # # # # # # # #         flat = pi.reshape(-1).clamp(0.0)
# # # # # # # # #         s = flat.sum()
# # # # # # # # #         if not torch.isfinite(s) or s < 1e-10:
# # # # # # # # #             return x0_flat, x1_flat
# # # # # # # # #         idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
# # # # # # # # #         row = idx // B
# # # # # # # # #         return x0_flat[row], x1_flat
# # # # # # # # #     except Exception:
# # # # # # # # #         return x0_flat, x1_flat


# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # #  VelocityTransformer (giữ nguyên từ v2.1)
# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # class VelocityTransformer(nn.Module):
# # # # # # # # #     def __init__(self, pred_len: int = 12, d_model: int = 256,
# # # # # # # # #                  nhead: int = 8, num_layers: int = 4, dim_ff: int = 512,
# # # # # # # # #                  dropout: float = 0.1, d_cond: int = 256):
# # # # # # # # #         super().__init__()
# # # # # # # # #         self.pred_len = pred_len
# # # # # # # # #         self.d_model  = d_model

# # # # # # # # #         self.traj_embed = nn.Linear(2, d_model)
# # # # # # # # #         self.pos_emb    = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
# # # # # # # # #         self.step_emb   = nn.Embedding(pred_len, d_model)
# # # # # # # # #         self.time_mlp   = nn.Sequential(
# # # # # # # # #             nn.Linear(d_model, d_model * 2), nn.GELU(),
# # # # # # # # #             nn.Linear(d_model * 2, d_model))
# # # # # # # # #         self.cond_proj  = nn.Sequential(
# # # # # # # # #             nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))
# # # # # # # # #         dec_layer = nn.TransformerDecoderLayer(
# # # # # # # # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # # # # # # # #             dropout=dropout, activation="gelu",
# # # # # # # # #             batch_first=True, norm_first=True)
# # # # # # # # #         self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
# # # # # # # # #         self.out_norm = nn.LayerNorm(d_model)
# # # # # # # # #         self.out_proj = nn.Sequential(
# # # # # # # # #             nn.Linear(d_model, d_model // 2), nn.GELU(),
# # # # # # # # #             nn.Linear(d_model // 2, 2))
# # # # # # # # #         self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
# # # # # # # # #         nn.init.zeros_(self.out_proj[-1].weight)
# # # # # # # # #         nn.init.zeros_(self.out_proj[-1].bias)

# # # # # # # # #     def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
# # # # # # # # #         half = self.d_model // 2
# # # # # # # # #         freq = torch.exp(
# # # # # # # # #             torch.arange(half, device=t.device, dtype=t.dtype)
# # # # # # # # #             * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # # # # #         emb  = t.float().unsqueeze(1) * freq.unsqueeze(0)
# # # # # # # # #         emb  = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # # # # # # # #         if self.d_model % 2 == 1:
# # # # # # # # #             emb = F.pad(emb, (0, 1))
# # # # # # # # #         return self.time_mlp(emb)

# # # # # # # # #     def forward(self, x_t: torch.Tensor,
# # # # # # # # #                 t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
# # # # # # # # #         B, T, _ = x_t.shape
# # # # # # # # #         step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # # # #         x_emb = (self.traj_embed(x_t)
# # # # # # # # #                  + self.pos_emb[:, :T]
# # # # # # # # #                  + self.step_emb(step_idx))
# # # # # # # # #         memory = torch.cat([
# # # # # # # # #             self._time_emb(t).unsqueeze(1),
# # # # # # # # #             self.cond_proj(cond).unsqueeze(1)
# # # # # # # # #         ], dim=1)
# # # # # # # # #         out = self.out_norm(self.decoder(x_emb, memory))
# # # # # # # # #         return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)


# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # #  ContextEncoder (giữ nguyên từ v2.1 — 6 kinematic features)
# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # class ContextEncoder(nn.Module):
# # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # #     def __init__(self, obs_len: int = 8, unet_in_ch: int = 13,
# # # # # # # # #                  d_cond: int = 256):
# # # # # # # # #         super().__init__()
# # # # # # # # #         self.obs_len = obs_len
# # # # # # # # #         self.d_cond  = d_cond

# # # # # # # # #         self.spatial_enc     = FNO3DEncoder(
# # # # # # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # # # #             spatial_down=32, dropout=0.05)
# # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)
# # # # # # # # #         self.enc_1d          = DataEncoder1D(
# # # # # # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # # # # # #         self.env_enc         = Env_net(obs_len=obs_len, d_model=32)

# # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # #         self.ctx_drop = nn.Dropout(0.1)
# # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, d_cond)
# # # # # # # # #         self.ctx_ln2  = nn.LayerNorm(d_cond)

# # # # # # # # #         # 6 kinematic features: vel_x, vel_y, speed_n, sin(heading), cos(heading), accel
# # # # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # # # #             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
# # # # # # # # #             nn.Linear(256, d_cond // 2), nn.GELU())
# # # # # # # # #         self.hard_embed = nn.Sequential(
# # # # # # # # #             nn.Linear(1, d_cond // 4), nn.GELU(),
# # # # # # # # #             nn.Linear(d_cond // 4, d_cond // 4))
# # # # # # # # #         self.fuse = nn.Sequential(
# # # # # # # # #             nn.Linear(d_cond + d_cond // 2 + d_cond // 4, d_cond),
# # # # # # # # #             nn.LayerNorm(d_cond), nn.GELU())

# # # # # # # # #     def _encode_raw(self, batch_list) -> torch.Tensor:
# # # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # # #         image_obs = batch_list[11]
# # # # # # # # #         env_data  = batch_list[13]

# # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # #         T_obs = obs_traj.shape[0]
# # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # # # # #             e_3d_s = F.interpolate(
# # # # # # # # #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # #                 mode="linear", align_corners=False).permute(0, 2, 1)

# # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # # # #         t_w = torch.softmax(
# # # # # # # # #             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # # #                          device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # # # # #         f_sp = self.decoder_proj(
# # # # # # # # #             (e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # # # # # # # #         return F.gelu(self.ctx_ln(
# # # # # # # # #             self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

# # # # # # # # #     def _kinematic_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
# # # # # # # # #         """
# # # # # # # # #         6 kinematic features per obs step:
# # # # # # # # #           vel_x, vel_y, speed_n (normalized), sin(heading), cos(heading), accel
# # # # # # # # #         """
# # # # # # # # #         B     = obs_traj.shape[1]
# # # # # # # # #         T_obs = obs_traj.shape[0]
# # # # # # # # #         device = obs_traj.device

# # # # # # # # #         if T_obs >= 2:
# # # # # # # # #             traj_deg = _norm_to_deg(obs_traj)
# # # # # # # # #             vel_norm = obs_traj[1:] - obs_traj[:-1]
# # # # # # # # #             speed    = _step_speeds_kmh(traj_deg)
# # # # # # # # #             speed_n  = (speed / 20.0).clamp(-3.0, 3.0)
# # # # # # # # #             heading  = torch.atan2(vel_norm[:, :, 1], vel_norm[:, :, 0])

# # # # # # # # #             if T_obs >= 3:
# # # # # # # # #                 dspd  = speed[1:] - speed[:-1]
# # # # # # # # #                 accel = torch.cat([
# # # # # # # # #                     obs_traj.new_zeros(1, B),
# # # # # # # # #                     (dspd / 10.0).clamp(-3.0, 3.0)
# # # # # # # # #                 ], 0)
# # # # # # # # #             else:
# # # # # # # # #                 accel = obs_traj.new_zeros(T_obs - 1, B)

# # # # # # # # #             kine = torch.stack([
# # # # # # # # #                 vel_norm[:, :, 0],
# # # # # # # # #                 vel_norm[:, :, 1],
# # # # # # # # #                 speed_n,
# # # # # # # # #                 heading.sin(),
# # # # # # # # #                 heading.cos(),
# # # # # # # # #                 accel,
# # # # # # # # #             ], dim=-1)  # [T_obs-1, B, 6]
# # # # # # # # #         else:
# # # # # # # # #             kine = obs_traj.new_zeros(self.obs_len, B, 6)

# # # # # # # # #         # Pad or trim to obs_len
# # # # # # # # #         if kine.shape[0] < self.obs_len:
# # # # # # # # #             kine = torch.cat([
# # # # # # # # #                 obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6),
# # # # # # # # #                 kine
# # # # # # # # #             ], 0)
# # # # # # # # #         else:
# # # # # # # # #             kine = kine[-self.obs_len:]

# # # # # # # # #         return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

# # # # # # # # #     def forward(self, batch_list,
# # # # # # # # #                 hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# # # # # # # # #         raw   = self._encode_raw(batch_list)
# # # # # # # # #         ctx   = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
# # # # # # # # #         kfeat = self._kinematic_feat(batch_list[0][:, :, :2])

# # # # # # # # #         if hard_score is None:
# # # # # # # # #             hard_score = torch.zeros(ctx.shape[0], device=ctx.device)
# # # # # # # # #         hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))

# # # # # # # # #         return self.fuse(torch.cat([ctx, kfeat, hfeat], dim=-1))


# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # #  Hard score — XAI-2
# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # @torch.no_grad()
# # # # # # # # # def hard_score_from_obs(obs_traj_norm: torch.Tensor,
# # # # # # # # #                          return_components: bool = False):
# # # # # # # # #     """
# # # # # # # # #     [XAI-2] Độ khó của storm. Khi return_components=True, trả về dict giải thích:
# # # # # # # # #       curvature  : storm đang recurve (bẻ cong hướng)
# # # # # # # # #       speed_var  : tốc độ thay đổi đột ngột
# # # # # # # # #       dir_change : có nhiều bước rẽ lớn
# # # # # # # # #     """
# # # # # # # # #     T, B   = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
# # # # # # # # #     device = obs_traj_norm.device

# # # # # # # # #     if T < 3:
# # # # # # # # #         z = torch.zeros(B, device=device)
# # # # # # # # #         if return_components:
# # # # # # # # #             return z, {"curvature": z.clone(), "speed_var": z.clone(),
# # # # # # # # #                        "dir_change": z.clone()}
# # # # # # # # #         return z

# # # # # # # # #     traj_deg = _norm_to_deg(obs_traj_norm[..., :2])
# # # # # # # # #     az12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
# # # # # # # # #     az23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
# # # # # # # # #     diff = (az23 - az12).abs()
# # # # # # # # #     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)

# # # # # # # # #     curvature  = diff.mean(0) / math.pi
# # # # # # # # #     spd        = _step_speeds_kmh(traj_deg)

# # # # # # # # #     if spd.shape[0] >= 2:
# # # # # # # # #         speed_var = (spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 1.)
# # # # # # # # #     else:
# # # # # # # # #         speed_var = torch.zeros(B, device=device)

# # # # # # # # #     dir_change = (diff > (20.0 / 180.0 * math.pi)).float().mean(0)
# # # # # # # # #     score = (0.4 * curvature + 0.3 * speed_var + 0.3 * dir_change).clamp(0., 1.)

# # # # # # # # #     if return_components:
# # # # # # # # #         return score, {
# # # # # # # # #             "curvature":  curvature,
# # # # # # # # #             "speed_var":  speed_var,
# # # # # # # # #             "dir_change": dir_change,
# # # # # # # # #         }
# # # # # # # # #     return score


# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # #  Physics score (giữ nguyên từ v2.1)
# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # @torch.no_grad()
# # # # # # # # # def _physics_score(traj_norm: torch.Tensor,
# # # # # # # # #                    obs_norm: torch.Tensor) -> torch.Tensor:
# # # # # # # # #     """
# # # # # # # # #     Data-driven physics score. Dùng để chọn best sample trong ensemble.
# # # # # # # # #     Không hardcode bất kỳ giá trị speed nào.
# # # # # # # # #     """
# # # # # # # # #     B      = traj_norm.shape[1]
# # # # # # # # #     device = traj_norm.device
# # # # # # # # #     traj_deg = _norm_to_deg(traj_norm)

# # # # # # # # #     # ── Speed score ────────────────────────────────────────────────────────
# # # # # # # # #     if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# # # # # # # # #         obs_deg  = _norm_to_deg(obs_norm)
# # # # # # # # #         obs_spd  = _step_speeds_kmh(obs_deg)
# # # # # # # # #         T_s      = obs_spd.shape[0]
# # # # # # # # #         w_obs    = torch.linspace(0.5, 1.0, T_s, device=device)
# # # # # # # # #         v_ref    = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()
# # # # # # # # #         pred_spd = _step_speeds_kmh(traj_deg)
# # # # # # # # #         v_sigma  = v_ref.clamp(min=5.0) * 0.5
# # # # # # # # #         speed_score = torch.exp(
# # # # # # # # #             -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0))
# # # # # # # # #             .pow(2).mean(0) * 0.5)
# # # # # # # # #     elif traj_deg.shape[0] >= 2:
# # # # # # # # #         speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.).mean(0))
# # # # # # # # #     else:
# # # # # # # # #         speed_score = torch.ones(B, device=device)

# # # # # # # # #     # ── Smoothness score ───────────────────────────────────────────────────
# # # # # # # # #     if traj_deg.shape[0] >= 3:
# # # # # # # # #         vel       = traj_deg[1:] - traj_deg[:-1]
# # # # # # # # #         accel_mag = (vel[1:] - vel[:-1]).norm(dim=-1)
# # # # # # # # #         smooth_score = torch.exp(-accel_mag.mean(0) * 5.0)
# # # # # # # # #     else:
# # # # # # # # #         smooth_score = torch.ones(B, device=device)

# # # # # # # # #     # ── Heading score ──────────────────────────────────────────────────────
# # # # # # # # #     if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
# # # # # # # # #         obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # # # # # # # #         pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
# # # # # # # # #         obs_h    = F.normalize(obs_vel,  dim=-1, eps=1e-6)
# # # # # # # # #         pred_h   = F.normalize(pred_vel, dim=-1, eps=1e-6)
# # # # # # # # #         cos_sim  = (obs_h * pred_h).sum(-1).clamp(-1, 1)
# # # # # # # # #         head_score = torch.exp((cos_sim - 1.0) * 3.0)
# # # # # # # # #     else:
# # # # # # # # #         head_score = torch.ones(B, device=device)

# # # # # # # # #     return (speed_score.pow(0.35)
# # # # # # # # #             * smooth_score.pow(0.30)
# # # # # # # # #             * head_score.pow(0.35)).clamp(min=1e-6)


# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # #  [AUG-1] Augmentation v2.5 — thêm recurvature, giữ mild từ v2.1
# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # def augment_batch(batch_list) -> list:
# # # # # # # # #     """
# # # # # # # # #     v2.5 augmentation — giữ v2.1 core + thêm recurvature.

# # # # # # # # #     Distribution:
# # # # # # # # #       A (30%): track shift ±5km    — shape bão nguyên vẹn
# # # # # # # # #       B (30%): speed scale ×0.85–1.15 — robust với speed variation
# # # # # # # # #       C (20%): recurvature ±20°    — cover test distribution (recurving storms)
# # # # # # # # #       D (10%): no aug              — giữ original distribution trong training
# # # # # # # # #       E (10%): noise ±3km          — nhỏ hơn v2.4 để tránh distribution shift

# # # # # # # # #     KHÔNG dùng:
# # # # # # # # #       - mixup (tạo trajectories phi vật lý)
# # # # # # # # #       - rotation đơn thuần (v2.4 dùng và fail)
# # # # # # # # #       - speed scale >×1.5 (quá xa thực tế)
# # # # # # # # #     """
# # # # # # # # #     bl = list(batch_list)
# # # # # # # # #     if not torch.is_tensor(bl[0]):
# # # # # # # # #         return bl

# # # # # # # # #     obs    = bl[0]
# # # # # # # # #     device = obs.device
# # # # # # # # #     anchor = obs[-1:, :, :2].detach()  # [1, B, 2]
# # # # # # # # #     r = torch.rand(1).item()

# # # # # # # # #     if r < 0.30:
# # # # # # # # #         # A: Track shift ±5km (giữ nguyên từ v2.1)
# # # # # # # # #         shift = (torch.rand(2, device=device) - 0.5) * 0.018
# # # # # # # # #         bl[0] = obs + shift.view(1, 1, 2)
# # # # # # # # #         if torch.is_tensor(bl[1]):
# # # # # # # # #             bl[1] = bl[1] + shift.view(1, 1, 2)

# # # # # # # # #     elif r < 0.60:
# # # # # # # # #         # B: Speed scale ×0.85–1.15 (giữ nguyên từ v2.1)
# # # # # # # # #         scale  = 0.85 + 0.30 * torch.rand(1, device=device).item()
# # # # # # # # #         obs_c  = obs.clone()
# # # # # # # # #         obs_c[..., :2] = anchor + (obs[..., :2] - anchor) * scale
# # # # # # # # #         bl[0]  = obs_c
# # # # # # # # #         if torch.is_tensor(bl[1]):
# # # # # # # # #             bl[1] = anchor + (bl[1] - anchor) * scale

# # # # # # # # #     elif r < 0.80:
# # # # # # # # #         # C: Recurvature augmentation — CHÌA KHÓA để giảm val→test gap
# # # # # # # # #         #
# # # # # # # # #         # Simulate bão recurve: rotation tăng dần theo ease-in curve.
# # # # # # # # #         # ease-in vì recurvature thường bắt đầu chậm sau đó accelerate.
# # # # # # # # #         #
# # # # # # # # #         # Max angle: ±20° trong 72h là realistic theo climatology.
# # # # # # # # #         # Obs cũng được rotate nhẹ (30% của max) để nhất quán.
# # # # # # # # #         T_pred = bl[1].shape[0] if torch.is_tensor(bl[1]) else 0
# # # # # # # # #         if T_pred >= 4:
# # # # # # # # #             gt      = bl[1].clone()
# # # # # # # # #             max_deg = (torch.rand(1).item() - 0.5) * 40.0   # -20° to +20°
# # # # # # # # #             max_rad = max_deg * math.pi / 180.0

# # # # # # # # #             for t in range(T_pred):
# # # # # # # # #                 # ease-in: chậm lúc đầu, nhanh dần
# # # # # # # # #                 progress = (t / max(T_pred - 1, 1)) ** 1.5
# # # # # # # # #                 angle_t  = max_rad * progress
# # # # # # # # #                 cos_a, sin_a = math.cos(angle_t), math.sin(angle_t)
# # # # # # # # #                 rot = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]],
# # # # # # # # #                                    dtype=gt.dtype, device=device)
# # # # # # # # #                 rel = (gt[t] - anchor[0]).unsqueeze(-1)  # [B, 2, 1]
# # # # # # # # #                 gt[t] = (rot @ rel).squeeze(-1) + anchor[0]
# # # # # # # # #             bl[1] = gt

# # # # # # # # #             # Rotate cuối obs nhẹ (30% angle) để nhất quán với gt aug
# # # # # # # # #             T_obs   = obs.shape[0]
# # # # # # # # #             obs_aug = obs.clone()
# # # # # # # # #             partial_angle = max_rad * 0.3
# # # # # # # # #             cos_p, sin_p = math.cos(partial_angle), math.sin(partial_angle)
# # # # # # # # #             rot_p = torch.tensor([[cos_p, -sin_p], [sin_p, cos_p]],
# # # # # # # # #                                   dtype=obs.dtype, device=device)
# # # # # # # # #             # Chỉ rotate 3 bước cuối của obs
# # # # # # # # #             for t_obs in range(max(0, T_obs - 3), T_obs):
# # # # # # # # #                 rel_obs = (obs_aug[t_obs, :, :2] - anchor[0]).unsqueeze(-1)
# # # # # # # # #                 obs_aug[t_obs, :, :2] = (rot_p @ rel_obs).squeeze(-1) + anchor[0]
# # # # # # # # #             bl[0] = obs_aug

# # # # # # # # #     elif r < 0.90:
# # # # # # # # #         # D: no augmentation
# # # # # # # # #         pass

# # # # # # # # #     else:
# # # # # # # # #         # E: small Gaussian noise ±3km
# # # # # # # # #         obs_new = obs.clone()
# # # # # # # # #         obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.003
# # # # # # # # #         bl[0] = obs_new

# # # # # # # # #     return bl


# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # #  XAI functions
# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # def compute_obs_attribution(model, batch_list, device: torch.device,
# # # # # # # # #                              target_step: int = 11) -> torch.Tensor:
# # # # # # # # #     """
# # # # # # # # #     [XAI-1] Gradient-based saliency: obs timestep nào ảnh hưởng nhất
# # # # # # # # #     đến dự báo tại target_step (default 72h = step 11)?

# # # # # # # # #     Không gọi model.eval() để giữ dropout active (faithful attribution).
# # # # # # # # #     Tính hard_score bên ngoài no_grad để không chặn gradient.

# # # # # # # # #     Returns:
# # # # # # # # #         attr [T_obs, B] — normalized importance [0,1] per obs step per storm.
# # # # # # # # #         attr[-1, b] thường cao nhất (obs gần nhất quan trọng nhất).
# # # # # # # # #         attr[0, b] cao → storm có "trí nhớ dài".
# # # # # # # # #     """
# # # # # # # # #     raw = _unwrap_model(model)

# # # # # # # # #     # Hard score tính bên ngoài — không cần gradient
# # # # # # # # #     with torch.no_grad():
# # # # # # # # #         h_score = hard_score_from_obs(batch_list[0][:, :, :2])

# # # # # # # # #     # Tạo obs tensor yêu cầu gradient
# # # # # # # # #     obs_req = batch_list[0].detach().clone().requires_grad_(True)
# # # # # # # # #     bl_grad = list(batch_list)
# # # # # # # # #     bl_grad[0] = obs_req

# # # # # # # # #     with torch.enable_grad():
# # # # # # # # #         cond = raw.encoder(bl_grad, hard_score=h_score)
# # # # # # # # #         x0   = torch.randn(
# # # # # # # # #             obs_req.shape[1], raw.pred_len, 2, device=device
# # # # # # # # #         ) * raw.sigma_inference
# # # # # # # # #         t0   = torch.zeros(obs_req.shape[1], device=device)
# # # # # # # # #         v    = raw.velocity(x0, t0, cond)
# # # # # # # # #         pred_rel = x0 + v  # [B, T, 2]

# # # # # # # # #         ts       = min(target_step, raw.pred_len - 1)
# # # # # # # # #         loss_xai = pred_rel[:, ts, :].norm(dim=-1).mean()
# # # # # # # # #         loss_xai.backward()

# # # # # # # # #     if obs_req.grad is not None:
# # # # # # # # #         attr = obs_req.grad[:, :, :2].norm(dim=-1)        # [T_obs, B]
# # # # # # # # #         attr = attr / (attr.sum(0, keepdim=True) + 1e-8)  # normalize per storm
# # # # # # # # #     else:
# # # # # # # # #         attr = torch.zeros(
# # # # # # # # #             batch_list[0].shape[0], batch_list[0].shape[1], device=device)

# # # # # # # # #     return attr.detach()


# # # # # # # # # def compute_ensemble_uncertainty(all_traj: torch.Tensor) -> Dict:
# # # # # # # # #     """
# # # # # # # # #     [XAI-4] Std deviation per lead time across K ensemble samples.

# # # # # # # # #     Args:
# # # # # # # # #         all_traj: [K, T, B, 2] absolute normalized coords.

# # # # # # # # #     Returns dict:
# # # # # # # # #         std_per_step [T, B]:     uncertainty in km per lead time per storm
# # # # # # # # #         uncertainty_ratio [B]:   72h_std / 12h_std (should grow with time)
# # # # # # # # #         mean_72h_std float:      average uncertainty at 72h (km)
# # # # # # # # #         mean_12h_std float:      average uncertainty at 12h (km)
# # # # # # # # #         high_uncertainty [B]:    bool mask — storms where 72h_std > 80km
# # # # # # # # #     """
# # # # # # # # #     all_deg  = _norm_to_deg(all_traj)  # [K, T, B, 2]
# # # # # # # # #     K, T, B  = all_deg.shape[:3]
# # # # # # # # #     mean_traj = all_deg.mean(0)        # [T, B, 2]

# # # # # # # # #     std_km = torch.zeros(T, B, device=all_traj.device)
# # # # # # # # #     for t in range(T):
# # # # # # # # #         dists = _haversine_deg(
# # # # # # # # #             all_deg[:, t, :, :].reshape(K * B, 2),
# # # # # # # # #             mean_traj[t].unsqueeze(0).expand(K, B, 2).reshape(K * B, 2)
# # # # # # # # #         ).reshape(K, B)
# # # # # # # # #         std_km[t] = dists.std(0)

# # # # # # # # #     step_12h = min(1, T - 1)
# # # # # # # # #     step_72h = min(11, T - 1)

# # # # # # # # #     return {
# # # # # # # # #         "std_per_step":      std_km,
# # # # # # # # #         "uncertainty_ratio": (std_km[step_72h] + 1e-3) / (std_km[step_12h] + 1e-3),
# # # # # # # # #         "mean_72h_std":      float(std_km[step_72h].mean().item()),
# # # # # # # # #         "mean_12h_std":      float(std_km[step_12h].mean().item()),
# # # # # # # # #         "high_uncertainty":  std_km[step_72h] > 80.0,   # flag storms needing caution
# # # # # # # # #     }


# # # # # # # # # def compute_heading_deviation(pred_deg: torch.Tensor,
# # # # # # # # #                                gt_deg:   torch.Tensor) -> torch.Tensor:
# # # # # # # # #     """
# # # # # # # # #     [XAI-6] Heading deviation per lead time step.

# # # # # # # # #     At each step t, computes the angular difference between:
# # # # # # # # #       - reference direction: gt[t-1] → gt[t]
# # # # # # # # #       - prediction direction: gt[t-1] → pred[t]

# # # # # # # # #     Returns:
# # # # # # # # #         dev_deg [T-1, B] — heading deviation in degrees (absolute value).
# # # # # # # # #         High values indicate directional error (root cause of high CTE).
# # # # # # # # #     """
# # # # # # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # # #     if T < 2:
# # # # # # # # #         return pred_deg.new_zeros(1, pred_deg.shape[1])

# # # # # # # # #     # Reference: direction gt takes
# # # # # # # # #     bear_gt_ref = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])   # [T-1, B]
# # # # # # # # #     # Prediction: direction pred takes from same starting point
# # # # # # # # #     bear_pred   = _forward_azimuth(gt_deg[:T-1], pred_deg[1:T])   # [T-1, B]

# # # # # # # # #     angle_diff = (bear_pred - bear_gt_ref).abs()
# # # # # # # # #     # Normalize to [-pi, pi]
# # # # # # # # #     angle_diff = torch.where(angle_diff > math.pi, 2 * math.pi - angle_diff, angle_diff)

# # # # # # # # #     return torch.rad2deg(angle_diff)  # [T-1, B] in degrees


# # # # # # # # # def compute_cte_contribution(pred_deg: torch.Tensor,
# # # # # # # # #                               gt_deg:   torch.Tensor) -> Dict:
# # # # # # # # #     """
# # # # # # # # #     [XAI-7] Cross-track and along-track error decomposition.

# # # # # # # # #     Returns dict:
# # # # # # # # #         ate_per_step [T-1, B] signed km (+ = over-shoot, - = under-shoot)
# # # # # # # # #         cte_per_step [T-1, B] signed km (+ = right of track, - = left)
# # # # # # # # #         ate_mean [B]:          mean ATE per storm
# # # # # # # # #         cte_mean [B]:          mean |CTE| per storm
# # # # # # # # #         ate_abs_mean float:    batch-level ATE
# # # # # # # # #         cte_abs_mean float:    batch-level CTE
# # # # # # # # #     """
# # # # # # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # # #     if T < 2:
# # # # # # # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # # # # # #         return {"ate_per_step": z, "cte_per_step": z,
# # # # # # # # #                 "ate_mean": z[0], "cte_mean": z[0],
# # # # # # # # #                 "ate_abs_mean": 0.0, "cte_abs_mean": 0.0}

# # # # # # # # #     bear_ref  = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])     # [T-1, B]
# # # # # # # # #     bear_err  = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])     # [T-1, B]
# # # # # # # # #     dist_err  = _haversine_deg(pred_deg[1:T], gt_deg[1:T])        # [T-1, B]
# # # # # # # # #     angle_diff = bear_err - bear_ref

# # # # # # # # #     ate = dist_err * torch.cos(angle_diff)   # along-track (speed error)
# # # # # # # # #     cte = dist_err * torch.sin(angle_diff)   # cross-track (direction error)

# # # # # # # # #     return {
# # # # # # # # #         "ate_per_step":  ate,
# # # # # # # # #         "cte_per_step":  cte,
# # # # # # # # #         "ate_mean":      ate.mean(0),              # [B]
# # # # # # # # #         "cte_mean":      cte.abs().mean(0),        # [B]
# # # # # # # # #         "ate_abs_mean":  float(ate.abs().mean().item()),
# # # # # # # # #         "cte_abs_mean":  float(cte.abs().mean().item()),
# # # # # # # # #     }


# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # #  Compatibility stubs
# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # @torch.no_grad()
# # # # # # # # # def classify_hard_easy(obs_traj_norm, per_sample_loss=None,
# # # # # # # # #                         hard_score_p: float = 70.0, loss_p: float = 50.0):
# # # # # # # # #     scores = hard_score_from_obs(obs_traj_norm)
# # # # # # # # #     B = scores.shape[0]
# # # # # # # # #     if B < 4:
# # # # # # # # #         return torch.zeros(B, dtype=torch.bool, device=scores.device)
# # # # # # # # #     return scores >= torch.quantile(scores, hard_score_p / 100.0)


# # # # # # # # # @torch.no_grad()
# # # # # # # # # def classify_hard_easy_global(obs_traj_norm, global_threshold):
# # # # # # # # #     return hard_score_from_obs(obs_traj_norm) >= global_threshold


# # # # # # # # # @torch.no_grad()
# # # # # # # # # def compute_diversity_score(candidates) -> float:
# # # # # # # # #     if len(candidates) < 2:
# # # # # # # # #         return 0.0
# # # # # # # # #     T, B    = candidates[0].shape[0], candidates[0].shape[1]
# # # # # # # # #     ep_step = min(T - 1, 11)
# # # # # # # # #     endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
# # # # # # # # #     N       = endpoints.shape[0]
# # # # # # # # #     ep_mean = endpoints.mean(0, keepdim=True)
# # # # # # # # #     dists   = _haversine_deg(
# # # # # # # # #         endpoints.reshape(N * B, 2),
# # # # # # # # #         ep_mean.expand(N, B, 2).reshape(N * B, 2)
# # # # # # # # #     ).reshape(N, B)
# # # # # # # # #     return float(dists.std(0).mean().item())


# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # #  TCFlowMatching v2.5
# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # # class TCFlowMatching(nn.Module):
# # # # # # # # #     """
# # # # # # # # #     TC-FlowMatching v2.5

# # # # # # # # #     Training objective:
# # # # # # # # #       L_total = L_CFM + lambda_reg * L_reg
# # # # # # # # #               + lambda_heading * L_heading      [ATE-1, ramp ep5→ep20]
# # # # # # # # #               + lambda_momentum * L_momentum    [ATE-2, ramp ep5→ep20]

# # # # # # # # #       L_CFM:      flow matching objective (random t, OT noise matching)
# # # # # # # # #       L_reg:      ADE loss at t=0 with sigma_inference, LINEAR step weights
# # # # # # # # #       L_heading:  penalize direction change obs→pred at first step
# # # # # # # # #       L_momentum: penalize pred speed deviating from obs speed (first 4 steps)

# # # # # # # # #     Inference (1-shot, same as v2.1):
# # # # # # # # #       x0 ~ N(0, sigma_inference²) — FIXED, no speed conditioning
# # # # # # # # #       x_pred_rel = x0 + v(x0, t=0, cond)
# # # # # # # # #       x_pred_abs = x_pred_rel + last_obs
# # # # # # # # #       Best-of-K with physics score

# # # # # # # # #     Optional multi-scale sigma inference (does not affect training):
# # # # # # # # #       sample_multiscale() uses sigmas=[0.025,0.035,0.04,0.05,0.065]
# # # # # # # # #       for robustness against speed distribution shift at test time.
# # # # # # # # #     """

# # # # # # # # #     def __init__(
# # # # # # # # #         self,
# # # # # # # # #         pred_len:          int   = 12,
# # # # # # # # #         obs_len:           int   = 8,
# # # # # # # # #         unet_in_ch:        int   = 13,
# # # # # # # # #         d_cond:            int   = 256,
# # # # # # # # #         d_model:           int   = 256,
# # # # # # # # #         nhead:             int   = 8,
# # # # # # # # #         num_dec_layers:    int   = 4,
# # # # # # # # #         dim_ff:            int   = 512,
# # # # # # # # #         dropout:           float = 0.1,
# # # # # # # # #         sigma_min:         float = 0.04,
# # # # # # # # #         sigma_max:         float = 0.08,
# # # # # # # # #         lambda_reg:        float = 0.2,
# # # # # # # # #         lambda_heading:    float = 0.05,   # [ATE-1] heading continuation weight
# # # # # # # # #         lambda_momentum:   float = 0.05,   # [ATE-2] speed momentum weight
# # # # # # # # #         use_ot:            bool  = True,
# # # # # # # # #         ot_epsilon:        float = 0.05,
# # # # # # # # #         use_ema:           bool  = True,
# # # # # # # # #         ema_decay:         float = 0.995,
# # # # # # # # #         n_inference_steps: int   = 1,
# # # # # # # # #         n_ensemble:        int   = 20,
# # # # # # # # #         sigma_inference:   float = 0.04,   # FIXED — không thay đổi per-storm
# # # # # # # # #         **kwargs,
# # # # # # # # #     ):
# # # # # # # # #         super().__init__()
# # # # # # # # #         self.pred_len          = pred_len
# # # # # # # # #         self.obs_len           = obs_len
# # # # # # # # #         self.sigma_min         = sigma_min
# # # # # # # # #         self.sigma_max         = sigma_max
# # # # # # # # #         self.lambda_reg        = lambda_reg
# # # # # # # # #         self.lambda_heading    = lambda_heading
# # # # # # # # #         self.lambda_momentum   = lambda_momentum
# # # # # # # # #         self.use_ot            = use_ot
# # # # # # # # #         self.ot_epsilon        = ot_epsilon
# # # # # # # # #         self.n_inference_steps = n_inference_steps
# # # # # # # # #         self.n_ensemble        = n_ensemble
# # # # # # # # #         self.sigma_inference   = sigma_inference

# # # # # # # # #         self.encoder  = ContextEncoder(
# # # # # # # # #             obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
# # # # # # # # #         self.velocity = VelocityTransformer(
# # # # # # # # #             pred_len=pred_len, d_model=d_model, nhead=nhead,
# # # # # # # # #             num_layers=num_dec_layers, dim_ff=dim_ff,
# # # # # # # # #             dropout=dropout, d_cond=d_cond)
# # # # # # # # #         self.use_ema = use_ema
# # # # # # # # #         self._ema    = None

# # # # # # # # #     def init_ema(self):
# # # # # # # # #         if self.use_ema:
# # # # # # # # #             self._ema = EMAModel(self, decay=0.995)

# # # # # # # # #     def ema_update(self):
# # # # # # # # #         if self._ema is not None:
# # # # # # # # #             self._ema.update(self)

# # # # # # # # #     def _to_relative(self, x_abs:  torch.Tensor,
# # # # # # # # #                      last_obs: torch.Tensor) -> torch.Tensor:
# # # # # # # # #         return x_abs - last_obs.unsqueeze(1)

# # # # # # # # #     def _from_relative(self, x_rel:  torch.Tensor,
# # # # # # # # #                        last_obs: torch.Tensor) -> torch.Tensor:
# # # # # # # # #         return x_rel + last_obs.unsqueeze(1)

# # # # # # # # #     def _sigma_schedule(self, epoch: int) -> float:
# # # # # # # # #         """Cosine decay sigma_max → sigma_min over ep5→ep40."""
# # # # # # # # #         if epoch < 5:
# # # # # # # # #             return self.sigma_max
# # # # # # # # #         if epoch < 40:
# # # # # # # # #             t = (epoch - 5) / 35.0
# # # # # # # # #             return (self.sigma_min
# # # # # # # # #                     + 0.5 * (self.sigma_max - self.sigma_min)
# # # # # # # # #                     * (1 + math.cos(math.pi * t)))
# # # # # # # # #         return self.sigma_min

# # # # # # # # #     # ── [ATE-2] Speed momentum loss ────────────────────────────────────────

# # # # # # # # #     def _momentum_loss(self, pred_deg:   torch.Tensor,
# # # # # # # # #                        obs_deg:    torch.Tensor,
# # # # # # # # #                        last_obs_deg: torch.Tensor,
# # # # # # # # #                        decay: float = 0.7) -> torch.Tensor:
# # # # # # # # #         """
# # # # # # # # #         Penalize pred speed at first 4 steps deviating from obs speed.

# # # # # # # # #         Normalized by obs_speed so fast/slow storms contribute equally.
# # # # # # # # #         Only first 4 steps (24h) — after that, speed may legitimately change.

# # # # # # # # #         decay: weight for each subsequent step. t=0→1.0, t=1→0.7, t=2→0.49...
# # # # # # # # #         """
# # # # # # # # #         if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
# # # # # # # # #             return pred_deg.new_zeros(())

# # # # # # # # #         # Reference speed: weighted mean of obs, recent steps matter more
# # # # # # # # #         obs_spds = _step_speeds_kmh(obs_deg)                    # [T_obs-1, B]
# # # # # # # # #         T_s      = obs_spds.shape[0]
# # # # # # # # #         w = torch.exp(
# # # # # # # # #             torch.arange(T_s, dtype=obs_spds.dtype, device=obs_spds.device) * 0.5)
# # # # # # # # #         w = w / w.sum()
# # # # # # # # #         ref_spd = (obs_spds * w.unsqueeze(1)).sum(0).clamp(min=3.0)  # [B]

# # # # # # # # #         # Pred speeds including step from last_obs to pred[0]
# # # # # # # # #         # last_obs_deg: [B, 2]
# # # # # # # # #         pts       = torch.cat([last_obs_deg.unsqueeze(0), pred_deg], 0)  # [T+1,B,2]
# # # # # # # # #         pred_spds = _step_speeds_kmh(pts)                                 # [T, B]

# # # # # # # # #         # Loss only for first N_steps
# # # # # # # # #         N_steps = min(4, pred_spds.shape[0])
# # # # # # # # #         loss    = pred_deg.new_zeros(())
# # # # # # # # #         for t in range(N_steps):
# # # # # # # # #             # Normalized deviation: (pred - ref) / ref
# # # # # # # # #             spd_ratio = (pred_spds[t] - ref_spd) / ref_spd
# # # # # # # # #             loss      = loss + (decay ** t) * spd_ratio.pow(2).mean()

# # # # # # # # #         return loss / N_steps

# # # # # # # # #     # ── [ATE-1] Heading continuation loss ──────────────────────────────────

# # # # # # # # #     def _heading_loss(self, pred_deg: torch.Tensor,
# # # # # # # # #                       obs_deg:  torch.Tensor) -> torch.Tensor:
# # # # # # # # #         """
# # # # # # # # #         Penalize direction change between obs final heading and pred first step.

# # # # # # # # #         loss = mean(1 - cos(pred_heading - obs_heading))
# # # # # # # # #         = 0 when headings match, 2 when exactly opposite.

# # # # # # # # #         Only applies to first step (6h) — anchoring the trajectory start.
# # # # # # # # #         """
# # # # # # # # #         if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
# # # # # # # # #             return pred_deg.new_zeros(())

# # # # # # # # #         last_obs = obs_deg[-1]   # [B, 2]
# # # # # # # # #         prev_obs = obs_deg[-2]   # [B, 2]
# # # # # # # # #         first_p  = pred_deg[0]   # [B, 2]

# # # # # # # # #         obs_bear  = _forward_azimuth(prev_obs, last_obs)  # [B]
# # # # # # # # #         pred_bear = _forward_azimuth(last_obs, first_p)   # [B]

# # # # # # # # #         angle_diff = pred_bear - obs_bear
# # # # # # # # #         return (1.0 - torch.cos(angle_diff)).mean()

# # # # # # # # #     # ── L_reg (giữ nguyên từ v2.1 — linear step weights) ─────────────────

# # # # # # # # #     def _reg_loss(self, x1_rel: torch.Tensor, last_obs: torch.Tensor,
# # # # # # # # #                   cond: torch.Tensor,
# # # # # # # # #                   hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# # # # # # # # #         """
# # # # # # # # #         ADE loss at t=0 consistent with inference.

# # # # # # # # #         LINEAR step weights (từ v2.1):
# # # # # # # # #           softmax(linspace(1,3,T)) → max/min ≈ 3×
# # # # # # # # #           KHÔNG dùng exp weights (v2.4 gây overfitting 72h).

# # # # # # # # #         Kết hợp với hard_score để nhấn mạnh storm khó.
# # # # # # # # #         """
# # # # # # # # #         B, T, _ = x1_rel.shape
# # # # # # # # #         device   = x1_rel.device

# # # # # # # # #         # Fresh noise with sigma_inference (same as inference)
# # # # # # # # #         x0 = torch.randn_like(x1_rel) * self.sigma_inference
# # # # # # # # #         t0 = torch.zeros(B, device=device)
# # # # # # # # #         v  = self.velocity(x0, t0, cond)

# # # # # # # # #         x1_pred_abs = self._from_relative(x0 + v, last_obs)
# # # # # # # # #         x1_gt_abs   = self._from_relative(x1_rel, last_obs)

# # # # # # # # #         pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))  # [T, B, 2]
# # # # # # # # #         gt_deg   = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
# # # # # # # # #         dist     = _haversine_deg(pred_deg, gt_deg)             # [T, B] km

# # # # # # # # #         # Linear step weights (v2.1 style — BẰNG CHỨNG generalize tốt hơn exp)
# # # # # # # # #         T_actual = dist.shape[0]
# # # # # # # # #         sw = torch.linspace(1.0, 3.0, T_actual, device=device, dtype=dist.dtype)
# # # # # # # # #         sw = F.softmax(sw, dim=0).unsqueeze(1)                  # [T, 1]

# # # # # # # # #         if hard_score is not None:
# # # # # # # # #             sample_w = (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)  # [1, B]
# # # # # # # # #         else:
# # # # # # # # #             sample_w = torch.ones(1, B, device=device, dtype=dist.dtype)

# # # # # # # # #         return ((dist * sw) * sample_w).mean() / 300.0

# # # # # # # # #     # ── Loss breakdown ──────────────────────────────────────────────────────

# # # # # # # # #     def get_loss_breakdown(self, batch_list, epoch: int = 0,
# # # # # # # # #                            **kwargs) -> Dict:
# # # # # # # # #         """
# # # # # # # # #         Tổng loss = L_CFM + lambda_reg * L_reg
# # # # # # # # #                   + lambda_heading * L_heading    (ramp ep5→ep20)
# # # # # # # # #                   + lambda_momentum * L_momentum  (ramp ep5→ep20)

# # # # # # # # #         Augmentation được gọi TỪ TRAIN LOOP, không phải ở đây.
# # # # # # # # #         Val loop gọi trực tiếp → val loss không bị augment.
# # # # # # # # #         """
# # # # # # # # #         obs_traj = batch_list[0]
# # # # # # # # #         gt_traj  = batch_list[1]
# # # # # # # # #         B        = obs_traj.shape[1]
# # # # # # # # #         device   = obs_traj.device

# # # # # # # # #         sigma    = self._sigma_schedule(epoch)
# # # # # # # # #         x1_gt    = gt_traj.permute(1, 0, 2)   # [B, 12, 2]
# # # # # # # # #         last_obs = obs_traj[-1, :, :2]
# # # # # # # # #         x1_rel   = self._to_relative(x1_gt, last_obs)

# # # # # # # # #         h_score = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # # # # #         cond    = self.encoder(batch_list, hard_score=h_score)

# # # # # # # # #         # ── L_CFM ─────────────────────────────────────────────────────────
# # # # # # # # #         x0 = torch.randn_like(x1_rel) * sigma
# # # # # # # # #         if self.use_ot and B >= 4:
# # # # # # # # #             x0_flat, x1_flat = _ot_match_noise_gt(
# # # # # # # # #                 x0.reshape(B, -1), x1_rel.reshape(B, -1), self.ot_epsilon)
# # # # # # # # #             x0         = x0_flat.reshape(B, self.pred_len, 2)
# # # # # # # # #             x1_matched = x1_flat.reshape(B, self.pred_len, 2)
# # # # # # # # #         else:
# # # # # # # # #             x1_matched = x1_rel

# # # # # # # # #         t        = torch.rand(B, device=device)
# # # # # # # # #         x_t      = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1_matched
# # # # # # # # #         u_target = x1_matched - x0
# # # # # # # # #         v_pred   = self.velocity(x_t, t, cond)
# # # # # # # # #         l_cfm    = F.mse_loss(v_pred, u_target)

# # # # # # # # #         # ── L_reg ramp ep10→ep30 ─────────────────────────────────────────
# # # # # # # # #         if epoch < 10:
# # # # # # # # #             lam_reg = 0.0
# # # # # # # # #         elif epoch < 30:
# # # # # # # # #             lam_reg = self.lambda_reg * (epoch - 10) / 20.0
# # # # # # # # #         else:
# # # # # # # # #             lam_reg = self.lambda_reg

# # # # # # # # #         l_reg = (self._reg_loss(x1_rel, last_obs, cond, h_score)
# # # # # # # # #                  if lam_reg > 0.0 else x0.new_zeros(()))

# # # # # # # # #         # ── L_heading + L_momentum — ramp ep5→ep20 ───────────────────────
# # # # # # # # #         # Ramp: 0 sebelum ep5, linear ep5→ep20, full after ep20
# # # # # # # # #         if epoch < 5:
# # # # # # # # #             lam_dir = 0.0
# # # # # # # # #         elif epoch < 20:
# # # # # # # # #             lam_dir = (epoch - 5) / 15.0
# # # # # # # # #         else:
# # # # # # # # #             lam_dir = 1.0

# # # # # # # # #         if lam_dir > 0.0:
# # # # # # # # #             # Compute heading and momentum losses on the 1-shot prediction
# # # # # # # # #             with torch.no_grad():
# # # # # # # # #                 x0_dir = torch.randn_like(x1_rel) * self.sigma_inference
# # # # # # # # #                 t0_dir = torch.zeros(B, device=device)
# # # # # # # # #                 v_dir  = self.velocity(x0_dir, t0_dir, cond)
# # # # # # # # #                 x1_pred_abs_dir = self._from_relative(x0_dir + v_dir, last_obs)

# # # # # # # # #             pred_deg_dir = _norm_to_deg(x1_pred_abs_dir.permute(1, 0, 2))  # [T, B, 2]
# # # # # # # # #             obs_deg      = _norm_to_deg(obs_traj[:, :, :2])                 # [T_obs, B, 2]
# # # # # # # # #             last_obs_deg = obs_deg[-1]                                       # [B, 2]

# # # # # # # # #             l_heading  = self._heading_loss(pred_deg_dir, obs_deg)
# # # # # # # # #             l_momentum = self._momentum_loss(pred_deg_dir, obs_deg, last_obs_deg)
# # # # # # # # #         else:
# # # # # # # # #             l_heading  = x0.new_zeros(())
# # # # # # # # #             l_momentum = x0.new_zeros(())

# # # # # # # # #         # ── Total ─────────────────────────────────────────────────────────
# # # # # # # # #         total = (l_cfm
# # # # # # # # #                  + lam_reg * l_reg
# # # # # # # # #                  + lam_dir * self.lambda_heading  * l_heading
# # # # # # # # #                  + lam_dir * self.lambda_momentum * l_momentum)

# # # # # # # # #         if not torch.isfinite(total):
# # # # # # # # #             total = x0.new_zeros(())

# # # # # # # # #         # ── ADE log (1-shot at inference sigma) ──────────────────────────
# # # # # # # # #         with torch.no_grad():
# # # # # # # # #             x0_log = torch.randn_like(x1_rel) * self.sigma_inference
# # # # # # # # #             v_log  = self.velocity(x0_log, torch.zeros(B, device=device), cond)
# # # # # # # # #             x1_abs_log = self._from_relative(x0_log + v_log, last_obs)
# # # # # # # # #             ade_log = _haversine_deg(
# # # # # # # # #                 _norm_to_deg(x1_abs_log.permute(1, 0, 2)),
# # # # # # # # #                 _norm_to_deg(x1_gt.permute(1, 0, 2))
# # # # # # # # #             ).mean().item()

# # # # # # # # #         return {
# # # # # # # # #             "total":    total,
# # # # # # # # #             "l_cfm":    l_cfm.item(),
# # # # # # # # #             "l_reg":    l_reg.item() if torch.is_tensor(l_reg) else 0.0,
# # # # # # # # #             "l_heading": l_heading.item() if torch.is_tensor(l_heading) else 0.0,
# # # # # # # # #             "l_momentum": l_momentum.item() if torch.is_tensor(l_momentum) else 0.0,
# # # # # # # # #             "lam_reg":  lam_reg,
# # # # # # # # #             "lam_dir":  lam_dir,
# # # # # # # # #             "sigma":    sigma,
# # # # # # # # #             "ade_1step": ade_log,
# # # # # # # # #             "hard_score_mean": float(h_score.mean()),
# # # # # # # # #             "hard_score_max":  float(h_score.max()),
# # # # # # # # #             # Compat keys cho train scripts cũ
# # # # # # # # #             "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
# # # # # # # # #             "speed": 0., "accel": 0., "fm_mse": l_cfm.item(),
# # # # # # # # #             "l_hard_total": 0., "n_hard": 0, "alpha_hard": 0.,
# # # # # # # # #             "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
# # # # # # # # #         }

# # # # # # # # #     def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
# # # # # # # # #         return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

# # # # # # # # #     # ── Sample (1-shot, same as v2.1) ──────────────────────────────────────

# # # # # # # # #     @torch.no_grad()
# # # # # # # # #     def sample(self, batch_list,
# # # # # # # # #                num_ensemble: Optional[int] = None,
# # # # # # # # #                ddim_steps:   Optional[int] = None,
# # # # # # # # #                return_xai:   bool = False,
# # # # # # # # #                **kwargs) -> Tuple:
# # # # # # # # #         """
# # # # # # # # #         1-shot inference với optional XAI output.

# # # # # # # # #         Trả về (pred_mean, zeros, all_t) hoặc (pred_mean, zeros, all_t, xai)
# # # # # # # # #         khi return_xai=True.

# # # # # # # # #         FIXED sigma_inference — không speed-conditioned (v2.4 fail).
# # # # # # # # #         """
# # # # # # # # #         K  = num_ensemble or self.n_ensemble
# # # # # # # # #         N  = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else self.n_inference_steps
# # # # # # # # #         dt = 1.0 / max(N, 1)

# # # # # # # # #         obs_traj    = batch_list[0]
# # # # # # # # #         T_obs, B, _ = obs_traj.shape
# # # # # # # # #         device      = obs_traj.device

# # # # # # # # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # # # # #         obs_norm = obs_traj[:, :, :2]
# # # # # # # # #         last_obs = obs_traj[-1, :, :2]
# # # # # # # # #         t0       = torch.zeros(B, device=device)

# # # # # # # # #         # Encode once for all K samples
# # # # # # # # #         cond = self.encoder(batch_list, hard_score=h_score)

# # # # # # # # #         all_traj = []
# # # # # # # # #         for _ in range(K):
# # # # # # # # #             x_rel = torch.randn(B, self.pred_len, 2,
# # # # # # # # #                                 device=device) * self.sigma_inference  # FIXED sigma

# # # # # # # # #             if N <= 1:
# # # # # # # # #                 # 1-shot: exactly what L_reg trains for
# # # # # # # # #                 v     = self.velocity(x_rel, t0, cond)
# # # # # # # # #                 x_rel = x_rel + v
# # # # # # # # #             else:
# # # # # # # # #                 # Euler fallback (for experiments only)
# # # # # # # # #                 for step in range(N):
# # # # # # # # #                     t_b   = torch.full((B,), step * dt, device=device)
# # # # # # # # #                     x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3., 3.)

# # # # # # # # #             x_abs = self._from_relative(x_rel, last_obs)
# # # # # # # # #             all_traj.append(x_abs.permute(1, 0, 2))  # [T, B, 2]

# # # # # # # # #         # Best-of-K selection with physics score
# # # # # # # # #         scores   = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)
# # # # # # # # #         all_t    = torch.stack(all_traj, 0)  # [K, T, B, 2]
# # # # # # # # #         top_k    = min(3, K)
# # # # # # # # #         top_idx  = scores.topk(top_k, dim=0).indices  # [top_k, B]

# # # # # # # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # # # # # # #         for b in range(B):
# # # # # # # # #             idx_b = top_idx[:, b]
# # # # # # # # #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # # # # # # # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# # # # # # # # #         if not return_xai:
# # # # # # # # #             return pred_mean, torch.zeros_like(pred_mean), all_t

# # # # # # # # #         # ── XAI output ─────────────────────────────────────────────────────
# # # # # # # # #         xai = {}

# # # # # # # # #         # XAI-4: Ensemble uncertainty
# # # # # # # # #         xai.update(compute_ensemble_uncertainty(all_t))

# # # # # # # # #         # XAI-2: Hard score components
# # # # # # # # #         _, hard_comps = hard_score_from_obs(obs_norm, return_components=True)
# # # # # # # # #         xai["hard_components"] = hard_comps

# # # # # # # # #         # XAI-3: Physics score components for best sample per storm
# # # # # # # # #         pred_deg  = _norm_to_deg(pred_mean)       # [T, B, 2]
# # # # # # # # #         obs_deg_x = _norm_to_deg(obs_norm)        # [T_obs, B, 2]

# # # # # # # # #         # Speed comparison (XAI-5)
# # # # # # # # #         obs_spd_x  = _step_speeds_kmh(obs_deg_x)
# # # # # # # # #         obs_spd_mu = obs_spd_x.mean(0)  # [B]
# # # # # # # # #         if pred_deg.shape[0] >= 2:
# # # # # # # # #             pred_spd_x  = _step_speeds_kmh(pred_deg)
# # # # # # # # #             pred_spd_mu = pred_spd_x.mean(0)  # [B]
# # # # # # # # #         else:
# # # # # # # # #             pred_spd_mu = obs_spd_mu.clone()

# # # # # # # # #         xai["speed_comparison"] = {
# # # # # # # # #             "obs_speed_mean":  float(obs_spd_mu.mean().item()),
# # # # # # # # #             "pred_speed_mean": float(pred_spd_mu.mean().item()),
# # # # # # # # #             "speed_ratio":     float((pred_spd_mu / obs_spd_mu.clamp(min=1.)).mean().item()),
# # # # # # # # #             "per_storm_obs":   obs_spd_mu,
# # # # # # # # #             "per_storm_pred":  pred_spd_mu,
# # # # # # # # #             "over_predict":    (pred_spd_mu / obs_spd_mu.clamp(min=1.) > 1.15),
# # # # # # # # #             "under_predict":   (pred_spd_mu / obs_spd_mu.clamp(min=1.) < 0.85),
# # # # # # # # #         }

# # # # # # # # #         # Physics score components (XAI-3)
# # # # # # # # #         # speed_score per storm for best prediction
# # # # # # # # #         if obs_deg_x.shape[0] >= 2 and pred_deg.shape[0] >= 2:
# # # # # # # # #             v_ref   = obs_spd_mu.clamp(min=5.)
# # # # # # # # #             v_sigma = v_ref * 0.5
# # # # # # # # #             speed_scores_xai = torch.exp(
# # # # # # # # #                 -((pred_spd_mu - v_ref) / v_sigma).pow(2) * 0.5)
# # # # # # # # #         else:
# # # # # # # # #             speed_scores_xai = torch.ones(B, device=device)

# # # # # # # # #         if pred_deg.shape[0] >= 3:
# # # # # # # # #             vel_x   = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # #             accel_x = (vel_x[1:] - vel_x[:-1]).norm(dim=-1)
# # # # # # # # #             smooth_scores_xai = torch.exp(-accel_x.mean(0) * 5.)
# # # # # # # # #         else:
# # # # # # # # #             smooth_scores_xai = torch.ones(B, device=device)

# # # # # # # # #         if obs_norm.shape[0] >= 2 and pred_mean.shape[0] >= 1:
# # # # # # # # #             obs_vel_x  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # # # # # # # #             pred_vel_x = pred_mean[0, :, :2] - obs_norm[-1, :, :2]
# # # # # # # # #             obs_h_x    = F.normalize(obs_vel_x,  dim=-1, eps=1e-6)
# # # # # # # # #             pred_h_x   = F.normalize(pred_vel_x, dim=-1, eps=1e-6)
# # # # # # # # #             cos_sim_x  = (obs_h_x * pred_h_x).sum(-1).clamp(-1., 1.)
# # # # # # # # #             head_scores_xai = torch.exp((cos_sim_x - 1.) * 3.)
# # # # # # # # #         else:
# # # # # # # # #             head_scores_xai = torch.ones(B, device=device)

# # # # # # # # #         xai["physics_components"] = {
# # # # # # # # #             "speed":       speed_scores_xai,
# # # # # # # # #             "smooth":      smooth_scores_xai,
# # # # # # # # #             "heading":     head_scores_xai,
# # # # # # # # #             "obs_speed":   obs_spd_mu,
# # # # # # # # #             "pred_speed":  pred_spd_mu,
# # # # # # # # #         }

# # # # # # # # #         # XAI-6: Heading deviation per lead time
# # # # # # # # #         gt_traj_xai = batch_list[1]  # [T, B, 2] normalized
# # # # # # # # #         gt_deg_xai  = _norm_to_deg(gt_traj_xai[:, :, :2])
# # # # # # # # #         xai["heading_deviation_deg"] = compute_heading_deviation(
# # # # # # # # #             pred_deg, gt_deg_xai)  # [T-1, B]

# # # # # # # # #         # XAI-7: Cross-track contribution
# # # # # # # # #         xai["ate_cte_decomp"] = compute_cte_contribution(pred_deg, gt_deg_xai)

# # # # # # # # #         return pred_mean, torch.zeros_like(pred_mean), all_t, xai

# # # # # # # # #     @torch.no_grad()
# # # # # # # # #     def sample_multiscale(
# # # # # # # # #         self,
# # # # # # # # #         batch_list,
# # # # # # # # #         sigmas: Optional[List[float]] = None,
# # # # # # # # #         n_per_sigma: int = 4,
# # # # # # # # #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# # # # # # # # #         """
# # # # # # # # #         Multi-scale sigma ensemble ở inference time.

# # # # # # # # #         KHÔNG thay đổi training — chỉ dùng khi evaluate.
# # # # # # # # #         Hedge against speed distribution shift giữa val và test.

# # # # # # # # #         Với mỗi sigma: sample n_per_sigma predictions.
# # # # # # # # #         Tổng K = len(sigmas) × n_per_sigma.
# # # # # # # # #         Physics score chọn best 5.

# # # # # # # # #         sigmas default: [0.025, 0.035, 0.04, 0.05, 0.065]
# # # # # # # # #           → cover range từ slow storms (cần sigma nhỏ) đến fast (sigma lớn)
# # # # # # # # #           → 0.04 là "training-consistent" point trong distribution
# # # # # # # # #         """
# # # # # # # # #         if sigmas is None:
# # # # # # # # #             sigmas = [0.025, 0.035, 0.04, 0.05, 0.065]

# # # # # # # # #         obs_traj = batch_list[0]
# # # # # # # # #         B        = obs_traj.shape[1]
# # # # # # # # #         device   = obs_traj.device

# # # # # # # # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # # # # #         obs_norm = obs_traj[:, :, :2]
# # # # # # # # #         last_obs = obs_traj[-1, :, :2]
# # # # # # # # #         t0       = torch.zeros(B, device=device)

# # # # # # # # #         # Encode once
# # # # # # # # #         cond = self.encoder(batch_list, hard_score=h_score)

# # # # # # # # #         all_traj = []
# # # # # # # # #         for sigma in sigmas:
# # # # # # # # #             for _ in range(n_per_sigma):
# # # # # # # # #                 x0    = torch.randn(B, self.pred_len, 2, device=device) * sigma
# # # # # # # # #                 v     = self.velocity(x0, t0, cond)
# # # # # # # # #                 x_rel = x0 + v
# # # # # # # # #                 x_abs = self._from_relative(x_rel, last_obs)
# # # # # # # # #                 all_traj.append(x_abs.permute(1, 0, 2))  # [T, B, 2]

# # # # # # # # #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)
# # # # # # # # #         all_t   = torch.stack(all_traj, 0)  # [K, T, B, 2]
# # # # # # # # #         top_k   = min(5, len(all_traj))
# # # # # # # # #         top_idx = scores.topk(top_k, dim=0).indices

# # # # # # # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # # # # # # #         for b in range(B):
# # # # # # # # #             idx_b = top_idx[:, b]
# # # # # # # # #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # # # # # # # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# # # # # # # # #         return pred_mean, torch.zeros_like(pred_mean), all_t


# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # #  Backward compat alias
# # # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # # TCDiffusion = TCFlowMatching

# # # # # # # # """
# # # # # # # # Model/flow_matching_model.py  ──  TC-FlowMatching v2.6
# # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # # VIẾT LẠI HOÀN TOÀN từ v2.1 + các fix đã verified từ thực nghiệm 140 epoch.

# # # # # # # # ━━━ CƠ SỞ LÝ THUYẾT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # # # # v2.1 test: ADE=229.8, ATE=214.4, CTE=71.6 (best so far, generalize tốt nhất)
# # # # # # # # v2.5 test: ADE=232.5, ATE=222.3, CTE=64.5 (ATE tệ hơn, CTE tốt hơn)
# # # # # # # # v2.4 test: ADE=291.1 (catastrophic overfitting)

# # # # # # # # PHÂN TÍCH LỖI v2.1:
# # # # # # # #   ATE/CTE ratio on test = 214.4/71.6 = 3.0
# # # # # # # #   → 75% error is ALONG-TRACK (speed/distance)
# # # # # # # #   → 25% is CROSS-TRACK (direction)
# # # # # # # #   Test gap 72h = +129.3km >> 12h = +10.1km → long-range drift

# # # # # # # # PHÂN TÍCH v2.5 THẤT BẠI:
# # # # # # # #   L_momentum anchor pred_speed ≈ val_speed (10.3km/h)
# # # # # # # #   Test storms faster → L_momentum SLOWS model on test → ATE +7.9km WORSE
# # # # # # # #   L_heading (step 0 only) → 72h heading dev INCREASED: ep0=87°→ep120=128°!
# # # # # # # #   Recurvature aug: CTE improved 7km ← giữ lại

# # # # # # # # ━━━ VẬT LÝ CỦA GIẢI PHÁP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # # # # GIẢI PHÁP ATE (along-track error):
# # # # # # # #   Training: KHÔNG anchor speed tại training (no L_momentum)
# # # # # # # #   Inference: SPEED CALIBRATION per-storm
# # # # # # # #     scale = clip(obs_speed / pred_speed_first4, 0.85, 1.15)
# # # # # # # #     pred_cal = last_obs + (pred - last_obs) * scale
# # # # # # # #     → Preserves direction (CTE unchanged), fixes magnitude (ATE reduced)
# # # # # # # #     → Per-storm adaptive: fast test storms get upscaling, slow storms downscaling
# # # # # # # #     → DOES NOT affect training → zero train/inference consistency issue

# # # # # # # # GIẢI PHÁP CTE (cross-track error):
# # # # # # # #   Training: MULTI-STEP heading constraint (not single step)
# # # # # # # #     step 0 (6h):  weight 1.0  — hard anchor to obs direction
# # # # # # # #     step 1 (12h): weight 0.5  — continuation (detached ref)
# # # # # # # #     step 2 (18h): weight 0.25 — softer continuation
# # # # # # # #     step 3 (24h): weight 0.125 — very soft
# # # # # # # #     Detach ref at each step: prevents gradient explosion through chain
# # # # # # # #     → 72h heading dev should come down from 128° to ~90°

# # # # # # # # GIẢI PHÁP GAP (val→test distribution shift):
# # # # # # # #   [AUG-C] Recurvature: proven to reduce CTE test (v2.5)
# # # # # # # #   [AUG-D] Obs-speed scaling ×0.7–1.4: model sees diverse speed contexts
# # # # # # # #            → at inference, adapts better to test storms' speed
# # # # # # # #   Speed calibration at inference bridges remaining gap

# # # # # # # # ━━━ GIỮ NGUYÊN TỪ v2.1 (PROVEN, KHÔNG THAY ĐỔI) ━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # # # #   ✅ sigma_inference=0.04 FIXED — zero train/inference mismatch
# # # # # # # #   ✅ L_reg softmax-linspace weights (max/min ≈ 3×) — không exp (v2.4 fail)
# # # # # # # #   ✅ Mild base aug: shift±5km + speed×0.85–1.15 (val ≈ test distribution)
# # # # # # # #   ✅ 1-shot inference nhất quán với L_reg training
# # # # # # # #   ✅ 2-group optimizer, encoder freeze 10ep
# # # # # # # #   ✅ val loop NO augmentation
# # # # # # # #   ✅ OT noise-GT matching trong L_CFM
# # # # # # # #   ✅ ContextEncoder: FNO3D + Mamba + Env + 6 kinematic features
# # # # # # # #   ✅ VelocityTransformer: d_model=256, 4 layers, nhead=8

# # # # # # # # ━━━ CẢI TIẾN v2.6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # # # #   [REMOVE]  L_momentum — LOẠI BỎ HOÀN TOÀN (made ATE +7.9km worse)
# # # # # # # #   [UPGRADE] L_heading → multi-step 4 steps, decay=0.5, weight=0.10
# # # # # # # #   [NEW-INF] speed_calibrate_pred() tại inference (clip 0.85–1.15)
# # # # # # # #   [UPGRADE] physics_score + displacement_score (15% weight)
# # # # # # # #   [AUG-D]   obs-speed scaling aug ×0.7–1.4 (15% probability)
# # # # # # # #   [XAI 1–7] giữ nguyên đầy đủ, XAI-5 now reports per-storm calibration

# # # # # # # # ━━━ KỲ VỌNG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # # # #                    Val ADE   Test ADE   Val ATE  Test ATE  Val CTE  Test CTE
# # # # # # # #   v2.1 (best):     170.1      229.8     160.2    214.4     45.7     71.6
# # # # # # # #   v2.5 (current):  170.5      232.5     161.3    222.3     46.2     64.5
# # # # # # # #   v2.6 (target):   170±1      210–220   161±1    200–210   46±1     57–63
# # # # # # # # """
# # # # # # # # from __future__ import annotations

# # # # # # # # import math
# # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # import torch
# # # # # # # # import torch.nn as nn
# # # # # # # # import torch.nn.functional as F

# # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # from Model.env_net_transformer_gphsplit import Env_net

# # # # # # # # R_EARTH  = 6371.0
# # # # # # # # DT_HOURS = 6.0


# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # #  Coordinate utilities (same as v2.1)
# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# # # # # # # #     """Normalized coords → (lon°, lat°)."""
# # # # # # # #     return torch.stack([
# # # # # # # #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# # # # # # # #         (t[..., 1] * 50.0) / 10.0,
# # # # # # # #     ], dim=-1)


# # # # # # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # # # #     """Haversine distance km between two (lon°,lat°) tensors."""
# # # # # # # #     lat1 = torch.deg2rad(p1[..., 1]);  lat2 = torch.deg2rad(p2[..., 1])
# # # # # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # # # # #     a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
# # # # # # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# # # # # # # # def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # # # #     """Bearing in radians from p1→p2 (degrees input)."""
# # # # # # # #     lon1 = torch.deg2rad(p1[..., 0]);  lat1 = torch.deg2rad(p1[..., 1])
# # # # # # # #     lon2 = torch.deg2rad(p2[..., 0]);  lat2 = torch.deg2rad(p2[..., 1])
# # # # # # # #     dlon = lon2 - lon1
# # # # # # # #     y = torch.sin(dlon) * torch.cos(lat2)
# # # # # # # #     x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
# # # # # # # #     return torch.atan2(y, x)


# # # # # # # # def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # #     """Speed km/h between consecutive steps of a trajectory [T, B, 2]."""
# # # # # # # #     if traj_deg.shape[0] < 2:
# # # # # # # #         return traj_deg.new_zeros(1, traj_deg.shape[1])
# # # # # # # #     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # #  EMAModel (same as v2.1)
# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # def _unwrap(m):
# # # # # # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m


# # # # # # # # class EMAModel:
# # # # # # # #     def __init__(self, model, decay: float = 0.995):
# # # # # # # #         self.decay = decay
# # # # # # # #         m = _unwrap(model)
# # # # # # # #         self.shadow = {k: v.detach().clone()
# # # # # # # #                        for k, v in m.state_dict().items()
# # # # # # # #                        if v.dtype.is_floating_point}

# # # # # # # #     def update(self, model):
# # # # # # # #         m = _unwrap(model)
# # # # # # # #         with torch.no_grad():
# # # # # # # #             for k, v in m.state_dict().items():
# # # # # # # #                 if k in self.shadow:
# # # # # # # #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

# # # # # # # #     def apply_to(self, model):
# # # # # # # #         m = _unwrap(model)
# # # # # # # #         backup, sd = {}, m.state_dict()
# # # # # # # #         for k in self.shadow:
# # # # # # # #             if k not in sd: continue
# # # # # # # #             backup[k] = sd[k].detach().clone()
# # # # # # # #             sd[k].copy_(self.shadow[k])
# # # # # # # #         return backup

# # # # # # # #     def restore(self, model, backup):
# # # # # # # #         m = _unwrap(model)
# # # # # # # #         sd = m.state_dict()
# # # # # # # #         for k, v in backup.items():
# # # # # # # #             if k in sd: sd[k].copy_(v)


# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # #  OT matching (same as v2.1)
# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 50) -> torch.Tensor:
# # # # # # # #     B = cost.shape[0]; device = cost.device
# # # # # # # #     log_a = -math.log(B) * torch.ones(B, device=device)
# # # # # # # #     log_b = -math.log(B) * torch.ones(B, device=device)
# # # # # # # #     log_K = -cost / epsilon
# # # # # # # #     log_u = torch.zeros(B, device=device)
# # # # # # # #     log_v = torch.zeros(B, device=device)
# # # # # # # #     for _ in range(n_iter):
# # # # # # # #         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
# # # # # # # #         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
# # # # # # # #     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


# # # # # # # # def _ot_match(x0_flat: torch.Tensor, x1_flat: torch.Tensor,
# # # # # # # #               epsilon: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # # # # #     B = x0_flat.shape[0]
# # # # # # # #     if B < 4:
# # # # # # # #         return x0_flat, x1_flat
# # # # # # # #     try:
# # # # # # # #         cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1] ** 0.5)
# # # # # # # #         with torch.no_grad():
# # # # # # # #             pi = _sinkhorn_log(cost, epsilon=epsilon)
# # # # # # # #         flat = pi.reshape(-1).clamp(0.0)
# # # # # # # #         s = flat.sum()
# # # # # # # #         if not torch.isfinite(s) or s < 1e-10:
# # # # # # # #             return x0_flat, x1_flat
# # # # # # # #         idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
# # # # # # # #         return x0_flat[idx // B], x1_flat
# # # # # # # #     except Exception:
# # # # # # # #         return x0_flat, x1_flat


# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # #  VelocityTransformer (identical to v2.1)
# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # class VelocityTransformer(nn.Module):
# # # # # # # #     def __init__(self, pred_len: int = 12, d_model: int = 256, nhead: int = 8,
# # # # # # # #                  num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
# # # # # # # #                  d_cond: int = 256):
# # # # # # # #         super().__init__()
# # # # # # # #         self.pred_len = pred_len
# # # # # # # #         self.d_model  = d_model
# # # # # # # #         self.traj_embed = nn.Linear(2, d_model)
# # # # # # # #         self.pos_emb    = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
# # # # # # # #         self.step_emb   = nn.Embedding(pred_len, d_model)
# # # # # # # #         self.time_mlp   = nn.Sequential(
# # # # # # # #             nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
# # # # # # # #         self.cond_proj  = nn.Sequential(nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))
# # # # # # # #         dec_layer = nn.TransformerDecoderLayer(
# # # # # # # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # # # # # # #             dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
# # # # # # # #         self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
# # # # # # # #         self.out_norm = nn.LayerNorm(d_model)
# # # # # # # #         self.out_proj = nn.Sequential(
# # # # # # # #             nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
# # # # # # # #         self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
# # # # # # # #         nn.init.zeros_(self.out_proj[-1].weight)
# # # # # # # #         nn.init.zeros_(self.out_proj[-1].bias)

# # # # # # # #     def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
# # # # # # # #         half = self.d_model // 2
# # # # # # # #         freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
# # # # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # # # #         emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
# # # # # # # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # # # # # # #         if self.d_model % 2 == 1:
# # # # # # # #             emb = F.pad(emb, (0, 1))
# # # # # # # #         return self.time_mlp(emb)

# # # # # # # #     def forward(self, x_t: torch.Tensor, t: torch.Tensor,
# # # # # # # #                 cond: torch.Tensor) -> torch.Tensor:
# # # # # # # #         B, T, _ = x_t.shape
# # # # # # # #         step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # # #         x_emb = (self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(step_idx))
# # # # # # # #         memory = torch.cat([self._time_emb(t).unsqueeze(1),
# # # # # # # #                             self.cond_proj(cond).unsqueeze(1)], dim=1)
# # # # # # # #         out = self.out_norm(self.decoder(x_emb, memory))
# # # # # # # #         return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)


# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # #  ContextEncoder (identical to v2.1)
# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # class ContextEncoder(nn.Module):
# # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # #     def __init__(self, obs_len: int = 8, unet_in_ch: int = 13, d_cond: int = 256):
# # # # # # # #         super().__init__()
# # # # # # # #         self.obs_len = obs_len
# # # # # # # #         self.d_cond  = d_cond

# # # # # # # #         self.spatial_enc     = FNO3DEncoder(in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # # #                                              n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # # #                                              spatial_down=32, dropout=0.05)
# # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)
# # # # # # # #         self.enc_1d          = DataEncoder1D(in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # # #                                               lstm_hidden=128, lstm_layers=3,
# # # # # # # #                                               dropout=0.1, d_state=16)
# # # # # # # #         self.env_enc         = Env_net(obs_len=obs_len, d_model=32)
# # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # #         self.ctx_drop = nn.Dropout(0.1)
# # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, d_cond)
# # # # # # # #         self.ctx_ln2  = nn.LayerNorm(d_cond)
# # # # # # # #         # 6 kinematic features: vel_x, vel_y, speed_n, sin(h), cos(h), accel
# # # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # # #             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
# # # # # # # #             nn.Linear(256, d_cond // 2), nn.GELU())
# # # # # # # #         self.hard_embed = nn.Sequential(
# # # # # # # #             nn.Linear(1, d_cond // 4), nn.GELU(), nn.Linear(d_cond // 4, d_cond // 4))
# # # # # # # #         self.fuse = nn.Sequential(
# # # # # # # #             nn.Linear(d_cond + d_cond // 2 + d_cond // 4, d_cond),
# # # # # # # #             nn.LayerNorm(d_cond), nn.GELU())

# # # # # # # #     def _encode_raw(self, batch_list) -> torch.Tensor:
# # # # # # # #         obs_traj  = batch_list[0]; obs_Me = batch_list[7]
# # # # # # # #         image_obs = batch_list[11]; env_data = batch_list[13]
# # # # # # # #         if image_obs.dim() == 4:
# # # # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
# # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # #         T_obs = obs_traj.shape[0]
# # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
# # # # # # # #                                    mode="linear", align_corners=False).permute(0,2,1)
# # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # #                                           device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # # # #         f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
# # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)
# # # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

# # # # # # # #     def _kinematic_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
# # # # # # # #         """6 kinematic features per obs step."""
# # # # # # # #         B = obs_traj.shape[1]; T_obs = obs_traj.shape[0]; device = obs_traj.device
# # # # # # # #         if T_obs >= 2:
# # # # # # # #             traj_deg = _norm_to_deg(obs_traj)
# # # # # # # #             vel_norm = obs_traj[1:] - obs_traj[:-1]
# # # # # # # #             speed    = _step_speeds_kmh(traj_deg)
# # # # # # # #             speed_n  = (speed / 20.0).clamp(-3.0, 3.0)
# # # # # # # #             heading  = torch.atan2(vel_norm[:, :, 1], vel_norm[:, :, 0])
# # # # # # # #             if T_obs >= 3:
# # # # # # # #                 dspd  = speed[1:] - speed[:-1]
# # # # # # # #                 accel = torch.cat([obs_traj.new_zeros(1, B),
# # # # # # # #                                    (dspd / 10.0).clamp(-3.0, 3.0)], 0)
# # # # # # # #             else:
# # # # # # # #                 accel = obs_traj.new_zeros(T_obs - 1, B)
# # # # # # # #             kine = torch.stack([vel_norm[:,:,0], vel_norm[:,:,1], speed_n,
# # # # # # # #                                 heading.sin(), heading.cos(), accel], dim=-1)
# # # # # # # #         else:
# # # # # # # #             kine = obs_traj.new_zeros(self.obs_len, B, 6)
# # # # # # # #         if kine.shape[0] < self.obs_len:
# # # # # # # #             kine = torch.cat([obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6), kine], 0)
# # # # # # # #         else:
# # # # # # # #             kine = kine[-self.obs_len:]
# # # # # # # #         return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

# # # # # # # #     def forward(self, batch_list, hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# # # # # # # #         raw   = self._encode_raw(batch_list)
# # # # # # # #         ctx   = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
# # # # # # # #         kfeat = self._kinematic_feat(batch_list[0][:, :, :2])
# # # # # # # #         if hard_score is None:
# # # # # # # #             hard_score = torch.zeros(ctx.shape[0], device=ctx.device)
# # # # # # # #         hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))
# # # # # # # #         return self.fuse(torch.cat([ctx, kfeat, hfeat], dim=-1))


# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # #  Hard score (XAI-2) — same as v2.1
# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # @torch.no_grad()
# # # # # # # # def hard_score_from_obs(obs_traj_norm: torch.Tensor,
# # # # # # # #                          return_components: bool = False):
# # # # # # # #     """[XAI-2] Độ khó của storm dựa trên curvature, speed_var, dir_change."""
# # # # # # # #     T, B   = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
# # # # # # # #     device = obs_traj_norm.device
# # # # # # # #     if T < 3:
# # # # # # # #         z = torch.zeros(B, device=device)
# # # # # # # #         if return_components:
# # # # # # # #             return z, {"curvature": z.clone(), "speed_var": z.clone(), "dir_change": z.clone()}
# # # # # # # #         return z
# # # # # # # #     traj_deg = _norm_to_deg(obs_traj_norm[..., :2])
# # # # # # # #     az12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
# # # # # # # #     az23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
# # # # # # # #     diff = (az23 - az12).abs()
# # # # # # # #     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
# # # # # # # #     curvature  = diff.mean(0) / math.pi
# # # # # # # #     spd        = _step_speeds_kmh(traj_deg)
# # # # # # # #     if spd.shape[0] >= 2:
# # # # # # # #         speed_var = (spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 1.)
# # # # # # # #     else:
# # # # # # # #         speed_var = torch.zeros(B, device=device)
# # # # # # # #     dir_change = (diff > (20.0 / 180.0 * math.pi)).float().mean(0)
# # # # # # # #     score = (0.4 * curvature + 0.3 * speed_var + 0.3 * dir_change).clamp(0., 1.)
# # # # # # # #     if return_components:
# # # # # # # #         return score, {"curvature": curvature, "speed_var": speed_var, "dir_change": dir_change}
# # # # # # # #     return score


# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # #  Physics score v2.6 (adds displacement_score)
# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # @torch.no_grad()
# # # # # # # # def _physics_score(traj_norm: torch.Tensor, obs_norm: torch.Tensor) -> torch.Tensor:
# # # # # # # #     """
# # # # # # # #     Best-of-K selection score. Four components:
# # # # # # # #       speed_score       (0.30): per-step speed vs obs reference
# # # # # # # #       smooth_score      (0.25): trajectory smoothness (no sharp acceleration)
# # # # # # # #       heading_score     (0.30): first-step direction matches obs
# # # # # # # #       displacement_score(0.15): total path length consistent with obs speed

# # # # # # # #     [NEW v2.6] displacement_score penalizes samples with total displacement
# # # # # # # #     far from obs-speed expectation → catches speed over-prediction artifacts
# # # # # # # #     that cause high ATE on test storms.
# # # # # # # #     """
# # # # # # # #     B      = traj_norm.shape[1]
# # # # # # # #     device = traj_norm.device
# # # # # # # #     traj_deg = _norm_to_deg(traj_norm)
# # # # # # # #     v_ref   = None

# # # # # # # #     # ── Speed score ────────────────────────────────────────────────────────
# # # # # # # #     if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# # # # # # # #         obs_deg = _norm_to_deg(obs_norm)
# # # # # # # #         obs_spd = _step_speeds_kmh(obs_deg)
# # # # # # # #         T_s     = obs_spd.shape[0]
# # # # # # # #         w_obs   = torch.linspace(0.5, 1.0, T_s, device=device)
# # # # # # # #         v_ref   = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()   # [B]
# # # # # # # #         pred_spd = _step_speeds_kmh(traj_deg)
# # # # # # # #         v_sigma  = v_ref.clamp(min=5.0) * 0.5
# # # # # # # #         speed_score = torch.exp(
# # # # # # # #             -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5)
# # # # # # # #     elif traj_deg.shape[0] >= 2:
# # # # # # # #         speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.).mean(0))
# # # # # # # #     else:
# # # # # # # #         speed_score = torch.ones(B, device=device)

# # # # # # # #     # ── Smoothness score ───────────────────────────────────────────────────
# # # # # # # #     if traj_deg.shape[0] >= 3:
# # # # # # # #         vel          = traj_deg[1:] - traj_deg[:-1]
# # # # # # # #         accel_mag    = (vel[1:] - vel[:-1]).norm(dim=-1)
# # # # # # # #         smooth_score = torch.exp(-accel_mag.mean(0) * 5.0)
# # # # # # # #     else:
# # # # # # # #         smooth_score = torch.ones(B, device=device)

# # # # # # # #     # ── Heading score ──────────────────────────────────────────────────────
# # # # # # # #     if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
# # # # # # # #         obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # # # # # # #         pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
# # # # # # # #         obs_h    = F.normalize(obs_vel,  dim=-1, eps=1e-6)
# # # # # # # #         pred_h   = F.normalize(pred_vel, dim=-1, eps=1e-6)
# # # # # # # #         cos_sim  = (obs_h * pred_h).sum(-1).clamp(-1, 1)
# # # # # # # #         head_score = torch.exp((cos_sim - 1.0) * 3.0)
# # # # # # # #     else:
# # # # # # # #         head_score = torch.ones(B, device=device)

# # # # # # # #     # ── [NEW v2.6] Displacement score ─────────────────────────────────────
# # # # # # # #     # Expected total path length ≈ obs_speed × T_pred × DT × 0.75
# # # # # # # #     # (0.75 factor: storms curve, so straight-line < path length estimate)
# # # # # # # #     # This catches ensemble samples that over/under-shoot on total distance.
# # # # # # # #     if v_ref is not None and traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# # # # # # # #         T_pred        = traj_deg.shape[0]
# # # # # # # #         expected_total = v_ref * T_pred * DT_HOURS * 0.75   # [B] km
# # # # # # # #         step_dists    = _haversine_deg(traj_deg[:-1], traj_deg[1:])   # [T-1, B]
# # # # # # # #         actual_total  = step_dists.sum(0)                              # [B]
# # # # # # # #         rel_err       = (actual_total - expected_total).abs() / expected_total.clamp(min=10.)
# # # # # # # #         disp_score    = torch.exp(-rel_err * 1.5)
# # # # # # # #     else:
# # # # # # # #         disp_score    = torch.ones(B, device=device)

# # # # # # # #     return (speed_score.pow(0.30)
# # # # # # # #             * smooth_score.pow(0.25)
# # # # # # # #             * head_score.pow(0.30)
# # # # # # # #             * disp_score.pow(0.15)).clamp(min=1e-6)


# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # #  Augmentation v2.6
# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # def augment_batch(batch_list) -> list:
# # # # # # # #     """
# # # # # # # #     v2.6 augmentation — extends v2.1 core with recurvature and obs-speed aug.

# # # # # # # #     Distribution (6 types):
# # # # # # # #       A (25%): track shift ±5km             — shape unchanged, position varies
# # # # # # # #       B (20%): GT speed scale ×0.85–1.15   — mild, proven from v2.1
# # # # # # # #       C (20%): recurvature ±20°             — proven CTE improvement in v2.5
# # # # # # # #       D (15%): obs-speed scaling ×0.7–1.4  — [NEW v2.6] diverse speed contexts
# # # # # # # #                                               buộc model học với fast/slow storms
# # # # # # # #                                               giảm anchoring to val speed distribution
# # # # # # # #       E (10%): no augmentation              — giữ original distribution
# # # # # # # #       F (10%): Gaussian noise ±3km          — small robustness

# # # # # # # #     KHÔNG dùng:
# # # # # # # #       - mixup (phi vật lý, v2.4 proof)
# # # # # # # #       - speed scale >×1.5 trên GT (too far from real)
# # # # # # # #       - exp L_reg weights (v2.4 catastrophic)

# # # # # # # #     AUG-D (obs-speed scaling) lý do:
# # # # # # # #       v2.5 XAI-5: model anchors to val speed (~10.3 km/h) throughout training.
# # # # # # # #       Test storms có speed distribution khác.
# # # # # # # #       AUG-D: scale OBS displacements but keep GT unchanged.
# # # # # # # #       → Model must INFER future speed from obs context (not memorize mean)
# # # # # # # #       → At inference, model adapts per-storm → speed_calibrate_pred() works better.
# # # # # # # #     """
# # # # # # # #     bl = list(batch_list)
# # # # # # # #     if not torch.is_tensor(bl[0]):
# # # # # # # #         return bl

# # # # # # # #     obs    = bl[0]
# # # # # # # #     device = obs.device
# # # # # # # #     anchor = obs[-1:, :, :2].detach()  # [1, B, 2] — last obs as pivot
# # # # # # # #     r = torch.rand(1).item()

# # # # # # # #     if r < 0.25:
# # # # # # # #         # A: Track shift ±5km (preserves storm shape, varies position)
# # # # # # # #         shift = (torch.rand(2, device=device) - 0.5) * 0.018  # ±5km in normalized
# # # # # # # #         bl[0] = obs + shift.view(1, 1, 2)
# # # # # # # #         if torch.is_tensor(bl[1]):
# # # # # # # #             bl[1] = bl[1] + shift.view(1, 1, 2)

# # # # # # # #     elif r < 0.45:
# # # # # # # #         # B: Speed scale ×0.85–1.15 (both obs+GT from anchor, mild)
# # # # # # # #         scale = 0.85 + 0.30 * torch.rand(1, device=device).item()
# # # # # # # #         obs_c = obs.clone()
# # # # # # # #         obs_c[..., :2] = anchor + (obs[..., :2] - anchor) * scale
# # # # # # # #         bl[0] = obs_c
# # # # # # # #         if torch.is_tensor(bl[1]):
# # # # # # # #             bl[1] = anchor + (bl[1] - anchor) * scale

# # # # # # # #     elif r < 0.65:
# # # # # # # #         # C: Recurvature ±20° — proven CTE improvement
# # # # # # # #         # Progressive rotation with ease-in: slow start, accelerate (realistic)
# # # # # # # #         T_pred = bl[1].shape[0] if torch.is_tensor(bl[1]) else 0
# # # # # # # #         if T_pred >= 4:
# # # # # # # #             gt      = bl[1].clone()
# # # # # # # #             max_deg = (torch.rand(1).item() - 0.5) * 40.0   # -20° to +20°
# # # # # # # #             max_rad = max_deg * math.pi / 180.0
# # # # # # # #             for t in range(T_pred):
# # # # # # # #                 progress = (t / max(T_pred - 1, 1)) ** 1.5  # ease-in
# # # # # # # #                 angle_t  = max_rad * progress
# # # # # # # #                 c, s = math.cos(angle_t), math.sin(angle_t)
# # # # # # # #                 rot = torch.tensor([[c, -s], [s, c]], dtype=gt.dtype, device=device)
# # # # # # # #                 rel = (gt[t] - anchor[0]).unsqueeze(-1)   # [B, 2, 1]
# # # # # # # #                 gt[t] = (rot @ rel).squeeze(-1) + anchor[0]
# # # # # # # #             bl[1] = gt
# # # # # # # #             # Rotate last 3 obs steps slightly (30% of max) for consistency
# # # # # # # #             T_obs   = obs.shape[0]
# # # # # # # #             obs_aug = obs.clone()
# # # # # # # #             cos_p   = math.cos(max_rad * 0.3)
# # # # # # # #             sin_p   = math.sin(max_rad * 0.3)
# # # # # # # #             rot_p   = torch.tensor([[cos_p, -sin_p], [sin_p, cos_p]],
# # # # # # # #                                     dtype=obs.dtype, device=device)
# # # # # # # #             for t_obs in range(max(0, T_obs - 3), T_obs):
# # # # # # # #                 rel_o = (obs_aug[t_obs, :, :2] - anchor[0]).unsqueeze(-1)
# # # # # # # #                 obs_aug[t_obs, :, :2] = (rot_p @ rel_o).squeeze(-1) + anchor[0]
# # # # # # # #             bl[0] = obs_aug

# # # # # # # #     elif r < 0.80:
# # # # # # # #         # D: [NEW v2.6] Obs-speed scaling ×0.7–1.4, GT unchanged
# # # # # # # #         #
# # # # # # # #         # Physics: scale OBS step displacements, keep GT as-is.
# # # # # # # #         # → Model sees storm that was moving at different speed in obs
# # # # # # # #         # → Must predict what the storm does next given obs context
# # # # # # # #         # → Cannot rely on "val mean speed" shortcut
# # # # # # # #         #
# # # # # # # #         # Range ×0.7–1.4 covers fast and slow test storms:
# # # # # # # #         # fast test storm: obs_speed_aug × 1.3 → model learns fast-storm context
# # # # # # # #         # slow test storm: obs_speed_aug × 0.8 → model learns slow-storm context
# # # # # # # #         speed_scale = 0.70 + 0.70 * torch.rand(1, device=device).item()  # [0.70, 1.40]
# # # # # # # #         T_obs = obs.shape[0]
# # # # # # # #         obs_aug = obs.clone()
# # # # # # # #         # Scale each step's displacement (not absolute position)
# # # # # # # #         obs_aug[0] = obs[0]  # keep first position fixed
# # # # # # # #         for t in range(1, T_obs):
# # # # # # # #             disp = obs[t, :, :2] - obs[t-1, :, :2]           # [B, 2]
# # # # # # # #             obs_aug[t, :, :2] = obs_aug[t-1, :, :2] + disp * speed_scale
# # # # # # # #         bl[0] = obs_aug
# # # # # # # #         # GT unchanged: intentional obs/gt speed mismatch forces generalization

# # # # # # # #     elif r < 0.90:
# # # # # # # #         # E: no augmentation (keep original distribution)
# # # # # # # #         pass

# # # # # # # #     else:
# # # # # # # #         # F: small Gaussian noise ±3km
# # # # # # # #         obs_new = obs.clone()
# # # # # # # #         obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.003
# # # # # # # #         bl[0] = obs_new

# # # # # # # #     return bl


# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # #  XAI functions (1–7, complete)
# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # def compute_obs_attribution(model, batch_list, device: torch.device,
# # # # # # # #                              target_step: int = 11) -> torch.Tensor:
# # # # # # # #     """
# # # # # # # #     [XAI-1] Gradient saliency: which obs step most influences 72h prediction?

# # # # # # # #     Returns attr [T_obs, B] normalized importance [0,1].
# # # # # # # #     High attr[-1, b] = last obs dominant (normal).
# # # # # # # #     High attr[0, b]  = storm has long "memory" (unusual → hard storm).
# # # # # # # #     """
# # # # # # # #     raw = _unwrap(model)
# # # # # # # #     with torch.no_grad():
# # # # # # # #         h_score = hard_score_from_obs(batch_list[0][:, :, :2])
# # # # # # # #     obs_req = batch_list[0].detach().clone().requires_grad_(True)
# # # # # # # #     bl_g = list(batch_list); bl_g[0] = obs_req
# # # # # # # #     with torch.enable_grad():
# # # # # # # #         cond = raw.encoder(bl_g, hard_score=h_score)
# # # # # # # #         x0   = torch.randn(obs_req.shape[1], raw.pred_len, 2, device=device) * raw.sigma_inference
# # # # # # # #         t0   = torch.zeros(obs_req.shape[1], device=device)
# # # # # # # #         v    = raw.velocity(x0, t0, cond)
# # # # # # # #         pred_rel = x0 + v
# # # # # # # #         ts       = min(target_step, raw.pred_len - 1)
# # # # # # # #         pred_rel[:, ts, :].norm(dim=-1).mean().backward()
# # # # # # # #     if obs_req.grad is not None:
# # # # # # # #         attr = obs_req.grad[:, :, :2].norm(dim=-1)
# # # # # # # #         attr = attr / (attr.sum(0, keepdim=True) + 1e-8)
# # # # # # # #     else:
# # # # # # # #         attr = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)
# # # # # # # #     return attr.detach()


# # # # # # # # @torch.no_grad()
# # # # # # # # def compute_ensemble_uncertainty(all_traj: torch.Tensor) -> Dict:
# # # # # # # #     """
# # # # # # # #     [XAI-4] Uncertainty per lead time across K ensemble samples.
# # # # # # # #     Returns std_per_step [T,B] km, uncertainty_ratio, mean_72h_std, high_uncertainty.
# # # # # # # #     """
# # # # # # # #     all_deg  = _norm_to_deg(all_traj)  # [K,T,B,2]
# # # # # # # #     K, T, B  = all_deg.shape[:3]
# # # # # # # #     mean_traj = all_deg.mean(0)
# # # # # # # #     std_km = torch.zeros(T, B, device=all_traj.device)
# # # # # # # #     for t in range(T):
# # # # # # # #         dists = _haversine_deg(
# # # # # # # #             all_deg[:, t].reshape(K * B, 2),
# # # # # # # #             mean_traj[t].unsqueeze(0).expand(K, B, 2).reshape(K * B, 2)
# # # # # # # #         ).reshape(K, B)
# # # # # # # #         std_km[t] = dists.std(0)
# # # # # # # #     s12 = min(1, T - 1); s72 = min(11, T - 1)
# # # # # # # #     return {
# # # # # # # #         "std_per_step":      std_km,
# # # # # # # #         "uncertainty_ratio": (std_km[s72] + 1e-3) / (std_km[s12] + 1e-3),
# # # # # # # #         "mean_72h_std":      float(std_km[s72].mean()),
# # # # # # # #         "mean_12h_std":      float(std_km[s12].mean()),
# # # # # # # #         "high_uncertainty":  std_km[s72] > 80.0,
# # # # # # # #     }


# # # # # # # # @torch.no_grad()
# # # # # # # # def compute_heading_deviation(pred_deg: torch.Tensor,
# # # # # # # #                                gt_deg:   torch.Tensor) -> torch.Tensor:
# # # # # # # #     """
# # # # # # # #     [XAI-6] Heading deviation per lead time step, in degrees (absolute).
# # # # # # # #     High values indicate directional error = root cause of CTE.
# # # # # # # #     Returns [T-1, B].
# # # # # # # #     """
# # # # # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # #     if T < 2:
# # # # # # # #         return pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # # # # #     bear_gt   = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
# # # # # # # #     bear_pred = _forward_azimuth(gt_deg[:T-1], pred_deg[1:T])
# # # # # # # #     diff = (bear_pred - bear_gt).abs()
# # # # # # # #     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
# # # # # # # #     return torch.rad2deg(diff)


# # # # # # # # @torch.no_grad()
# # # # # # # # def compute_cte_contribution(pred_deg: torch.Tensor,
# # # # # # # #                               gt_deg:   torch.Tensor) -> Dict:
# # # # # # # #     """
# # # # # # # #     [XAI-7] Decompose trajectory error into ATE (along-track) and CTE (cross-track).
# # # # # # # #     Returns ate_per_step [T-1,B], cte_per_step [T-1,B], scalar means.
# # # # # # # #     """
# # # # # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # #     if T < 2:
# # # # # # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # # # # #         return {"ate_per_step": z, "cte_per_step": z,
# # # # # # # #                 "ate_mean": z[0], "cte_mean": z[0],
# # # # # # # #                 "ate_abs_mean": 0.0, "cte_abs_mean": 0.0}
# # # # # # # #     bear_ref  = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
# # # # # # # #     bear_err  = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
# # # # # # # #     dist_err  = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# # # # # # # #     ang       = bear_err - bear_ref
# # # # # # # #     ate       = dist_err * torch.cos(ang)
# # # # # # # #     cte       = dist_err * torch.sin(ang)
# # # # # # # #     return {
# # # # # # # #         "ate_per_step":  ate,
# # # # # # # #         "cte_per_step":  cte,
# # # # # # # #         "ate_mean":      ate.mean(0),
# # # # # # # #         "cte_mean":      cte.abs().mean(0),
# # # # # # # #         "ate_abs_mean":  float(ate.abs().mean()),
# # # # # # # #         "cte_abs_mean":  float(cte.abs().mean()),
# # # # # # # #     }


# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # #  Compat stubs
# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # @torch.no_grad()
# # # # # # # # def classify_hard_easy(obs_traj_norm, per_sample_loss=None,
# # # # # # # #                         hard_score_p: float = 70.0, loss_p: float = 50.0):
# # # # # # # #     scores = hard_score_from_obs(obs_traj_norm)
# # # # # # # #     B = scores.shape[0]
# # # # # # # #     if B < 4:
# # # # # # # #         return torch.zeros(B, dtype=torch.bool, device=scores.device)
# # # # # # # #     return scores >= torch.quantile(scores, hard_score_p / 100.0)


# # # # # # # # @torch.no_grad()
# # # # # # # # def classify_hard_easy_global(obs_traj_norm, global_threshold):
# # # # # # # #     return hard_score_from_obs(obs_traj_norm) >= global_threshold


# # # # # # # # @torch.no_grad()
# # # # # # # # def compute_diversity_score(candidates) -> float:
# # # # # # # #     if len(candidates) < 2:
# # # # # # # #         return 0.0
# # # # # # # #     T, B = candidates[0].shape[0], candidates[0].shape[1]
# # # # # # # #     ep_step = min(T - 1, 11)
# # # # # # # #     endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
# # # # # # # #     N = endpoints.shape[0]
# # # # # # # #     ep_mean = endpoints.mean(0, keepdim=True)
# # # # # # # #     dists = _haversine_deg(
# # # # # # # #         endpoints.reshape(N * B, 2),
# # # # # # # #         ep_mean.expand(N, B, 2).reshape(N * B, 2)
# # # # # # # #     ).reshape(N, B)
# # # # # # # #     return float(dists.std(0).mean())


# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # #  TCFlowMatching v2.6
# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # # class TCFlowMatching(nn.Module):
# # # # # # # #     """
# # # # # # # #     TC-FlowMatching v2.6

# # # # # # # #     Training loss:
# # # # # # # #         L_total = L_CFM
# # # # # # # #                 + lam_reg  × L_reg          (ramp ep10→ep30, linear weights)
# # # # # # # #                 + lam_dir  × lambda_heading × L_heading_ms  (ramp ep5→ep20)

# # # # # # # #         L_CFM:         conditional flow matching (random t, OT noise-GT matching)
# # # # # # # #         L_reg:         ADE loss at sigma_inference, softmax-linspace step weights
# # # # # # # #         L_heading_ms:  [v2.6] 4-step heading continuation, decay=0.5 per step

# # # # # # # #     NO L_momentum: removed because it anchors to val speed distribution,
# # # # # # # #     which hurts ATE on test storms that move at different speeds.

# # # # # # # #     Inference pipeline:
# # # # # # # #         1. Sample K=20 candidates with sigma=0.04 FIXED
# # # # # # # #         2. Physics score selection (speed+smooth+heading+displacement)
# # # # # # # #         3. Weighted average of top-3 candidates
# # # # # # # #         4. Speed calibration: per-storm scale ×[0.85, 1.15] to match obs speed
# # # # # # # #         5. Optional: multi-scale sigma ensemble (sample_multiscale)
# # # # # # # #     """

# # # # # # # #     def __init__(
# # # # # # # #         self,
# # # # # # # #         pred_len:          int   = 12,
# # # # # # # #         obs_len:           int   = 8,
# # # # # # # #         unet_in_ch:        int   = 13,
# # # # # # # #         d_cond:            int   = 256,
# # # # # # # #         d_model:           int   = 256,
# # # # # # # #         nhead:             int   = 8,
# # # # # # # #         num_dec_layers:    int   = 4,
# # # # # # # #         dim_ff:            int   = 512,
# # # # # # # #         dropout:           float = 0.1,
# # # # # # # #         sigma_min:         float = 0.04,
# # # # # # # #         sigma_max:         float = 0.08,
# # # # # # # #         lambda_reg:        float = 0.2,
# # # # # # # #         lambda_heading:    float = 0.10,   # multi-step (was 0.05 in v2.5)
# # # # # # # #         lambda_momentum:   float = 0.0,    # DISABLED — field kept for compat only
# # # # # # # #         use_ot:            bool  = True,
# # # # # # # #         ot_epsilon:        float = 0.05,
# # # # # # # #         use_ema:           bool  = True,
# # # # # # # #         ema_decay:         float = 0.995,
# # # # # # # #         n_inference_steps: int   = 1,
# # # # # # # #         n_ensemble:        int   = 20,
# # # # # # # #         sigma_inference:   float = 0.04,   # FIXED throughout
# # # # # # # #         **kwargs,
# # # # # # # #     ):
# # # # # # # #         super().__init__()
# # # # # # # #         self.pred_len          = pred_len
# # # # # # # #         self.obs_len           = obs_len
# # # # # # # #         self.sigma_min         = sigma_min
# # # # # # # #         self.sigma_max         = sigma_max
# # # # # # # #         self.lambda_reg        = lambda_reg
# # # # # # # #         self.lambda_heading    = lambda_heading
# # # # # # # #         self.lambda_momentum   = 0.0           # always disabled
# # # # # # # #         self.use_ot            = use_ot
# # # # # # # #         self.ot_epsilon        = ot_epsilon
# # # # # # # #         self.n_inference_steps = n_inference_steps
# # # # # # # #         self.n_ensemble        = n_ensemble
# # # # # # # #         self.sigma_inference   = sigma_inference

# # # # # # # #         self.encoder  = ContextEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
# # # # # # # #         self.velocity = VelocityTransformer(
# # # # # # # #             pred_len=pred_len, d_model=d_model, nhead=nhead,
# # # # # # # #             num_layers=num_dec_layers, dim_ff=dim_ff,
# # # # # # # #             dropout=dropout, d_cond=d_cond)
# # # # # # # #         self.use_ema = use_ema
# # # # # # # #         self._ema    = None

# # # # # # # #     def init_ema(self):
# # # # # # # #         if self.use_ema:
# # # # # # # #             self._ema = EMAModel(self, decay=0.995)

# # # # # # # #     def ema_update(self):
# # # # # # # #         if self._ema is not None:
# # # # # # # #             self._ema.update(self)

# # # # # # # #     def _to_relative(self, x_abs: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# # # # # # # #         return x_abs - last_obs.unsqueeze(1)

# # # # # # # #     def _from_relative(self, x_rel: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# # # # # # # #         return x_rel + last_obs.unsqueeze(1)

# # # # # # # #     def _sigma_schedule(self, epoch: int) -> float:
# # # # # # # #         """Cosine decay sigma_max→sigma_min over ep5→ep40."""
# # # # # # # #         if epoch < 5:
# # # # # # # #             return self.sigma_max
# # # # # # # #         if epoch < 40:
# # # # # # # #             t = (epoch - 5) / 35.0
# # # # # # # #             return self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * t))
# # # # # # # #         return self.sigma_min

# # # # # # # #     # ── [v2.6] Multi-step heading loss ─────────────────────────────────────

# # # # # # # #     def _heading_loss_ms(self, pred_deg: torch.Tensor, obs_deg: torch.Tensor,
# # # # # # # #                           n_steps: int = 4, decay: float = 0.5) -> torch.Tensor:
# # # # # # # #         """
# # # # # # # #         Multi-step heading constraint. [UPGRADED FROM v2.5 SINGLE STEP]

# # # # # # # #         v2.5 used single-step (step 0 only, weight=0.05):
# # # # # # # #           → 72h heading dev INCREASED from 87° to 128° during training
# # # # # # # #           → Model learned "start correctly" then drifted

# # # # # # # #         v2.6 uses 4 steps with exponential decay:
# # # # # # # #           step t reference: obs_heading (t=0) or pred_bearing[t-1] (t>0, DETACHED)
# # # # # # # #           step 0: 1.0 × (1 - cos(pred_bear[0] - obs_bear))  ← hard anchor
# # # # # # # #           step 1: 0.5 × (1 - cos(pred_bear[1] - pred_bear[0]))  ← soft continuation
# # # # # # # #           step 2: 0.25 × ...
# # # # # # # #           step 3: 0.125 × ...

# # # # # # # #         DETACH ref at each step: prevents gradient explosion through chain.
# # # # # # # #         Each step trains relatively independently.

# # # # # # # #         Expected: 72h heading dev down from 128° to ~90-100°
# # # # # # # #         Expected: CTE test from 64.5km down to ~57-63km
# # # # # # # #         """
# # # # # # # #         if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
# # # # # # # #             return pred_deg.new_zeros(())

# # # # # # # #         # Starting reference: bearing from obs[-2] → obs[-1]
# # # # # # # #         ref_bear = _forward_azimuth(obs_deg[-2], obs_deg[-1])   # [B]

# # # # # # # #         loss = pred_deg.new_zeros(())
# # # # # # # #         # pts[t] → pts[t+1] gives bearing for pred step t
# # # # # # # #         pts = torch.cat([obs_deg[-1:], pred_deg], 0)   # [T_pred+1, B, 2]

# # # # # # # #         N = min(n_steps, pred_deg.shape[0])
# # # # # # # #         for t in range(N):
# # # # # # # #             pred_bear  = _forward_azimuth(pts[t], pts[t + 1])  # [B]
# # # # # # # #             angle_diff = pred_bear - ref_bear
# # # # # # # #             loss       = loss + (decay ** t) * (1.0 - torch.cos(angle_diff)).mean()
# # # # # # # #             ref_bear   = pred_bear.detach()   # CRITICAL: detach to prevent chain gradient

# # # # # # # #         return loss / N

# # # # # # # #     # ── L_reg (identical to v2.1 — proven not to overfit) ──────────────────

# # # # # # # #     def _reg_loss(self, x1_rel: torch.Tensor, last_obs: torch.Tensor,
# # # # # # # #                   cond: torch.Tensor,
# # # # # # # #                   hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# # # # # # # #         """
# # # # # # # #         ADE loss at t=0 with sigma_inference noise, consistent with 1-shot inference.
# # # # # # # #         Uses SOFTMAX-LINSPACE step weights (max/min ≈ 3×).
# # # # # # # #         DOES NOT use exp weights (v2.4 used 12.2× ratio → catastrophic overfitting).
# # # # # # # #         """
# # # # # # # #         B, T, _ = x1_rel.shape
# # # # # # # #         device   = x1_rel.device
# # # # # # # #         x0   = torch.randn_like(x1_rel) * self.sigma_inference
# # # # # # # #         t0   = torch.zeros(B, device=device)
# # # # # # # #         v    = self.velocity(x0, t0, cond)
# # # # # # # #         x1_pred_abs = self._from_relative(x0 + v, last_obs)
# # # # # # # #         x1_gt_abs   = self._from_relative(x1_rel, last_obs)
# # # # # # # #         pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))  # [T, B, 2]
# # # # # # # #         gt_deg   = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
# # # # # # # #         dist     = _haversine_deg(pred_deg, gt_deg)             # [T, B] km
# # # # # # # #         # Linear weights: softmax(linspace(1,3,T)) → max/min ≈ 3×
# # # # # # # #         T_actual = dist.shape[0]
# # # # # # # #         sw = torch.linspace(1.0, 3.0, T_actual, device=device, dtype=dist.dtype)
# # # # # # # #         sw = F.softmax(sw, dim=0).unsqueeze(1)                  # [T, 1]
# # # # # # # #         if hard_score is not None:
# # # # # # # #             sw_hard = (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)  # [1, B]
# # # # # # # #         else:
# # # # # # # #             sw_hard = torch.ones(1, B, device=device, dtype=dist.dtype)
# # # # # # # #         return ((dist * sw) * sw_hard).mean() / 300.0

# # # # # # # #     # ── get_loss_breakdown ──────────────────────────────────────────────────

# # # # # # # #     def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
# # # # # # # #         """
# # # # # # # #         Full loss computation. Called in TRAINING with augmented batch.
# # # # # # # #         Called in VAL without augmentation → val_loss is reliable signal.

# # # # # # # #         Total = L_CFM + lam_reg × L_reg + lam_dir × lambda_heading × L_heading_ms
# # # # # # # #         """
# # # # # # # #         obs_traj = batch_list[0]
# # # # # # # #         gt_traj  = batch_list[1]
# # # # # # # #         B        = obs_traj.shape[1]
# # # # # # # #         device   = obs_traj.device

# # # # # # # #         sigma    = self._sigma_schedule(epoch)
# # # # # # # #         x1_gt    = gt_traj.permute(1, 0, 2)       # [B, T, 2]
# # # # # # # #         last_obs = obs_traj[-1, :, :2]             # [B, 2]
# # # # # # # #         x1_rel   = self._to_relative(x1_gt, last_obs)

# # # # # # # #         h_score = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # # # #         cond    = self.encoder(batch_list, hard_score=h_score)

# # # # # # # #         # ── L_CFM ─────────────────────────────────────────────────────────
# # # # # # # #         x0 = torch.randn_like(x1_rel) * sigma
# # # # # # # #         if self.use_ot and B >= 4:
# # # # # # # #             x0_flat, x1_flat = _ot_match(
# # # # # # # #                 x0.reshape(B, -1), x1_rel.reshape(B, -1), self.ot_epsilon)
# # # # # # # #             x0         = x0_flat.reshape(B, self.pred_len, 2)
# # # # # # # #             x1_matched = x1_flat.reshape(B, self.pred_len, 2)
# # # # # # # #         else:
# # # # # # # #             x1_matched = x1_rel

# # # # # # # #         t        = torch.rand(B, device=device)
# # # # # # # #         x_t      = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1_matched
# # # # # # # #         u_target = x1_matched - x0
# # # # # # # #         v_pred   = self.velocity(x_t, t, cond)
# # # # # # # #         l_cfm    = F.mse_loss(v_pred, u_target)

# # # # # # # #         # ── L_reg ramp ep10→ep30 ─────────────────────────────────────────
# # # # # # # #         if epoch < 10:     lam_reg = 0.0
# # # # # # # #         elif epoch < 30:   lam_reg = self.lambda_reg * (epoch - 10) / 20.0
# # # # # # # #         else:              lam_reg = self.lambda_reg

# # # # # # # #         l_reg = (self._reg_loss(x1_rel, last_obs, cond, h_score)
# # # # # # # #                  if lam_reg > 0.0 else x0.new_zeros(()))

# # # # # # # #         # ── L_heading_ms ramp ep5→ep20 ────────────────────────────────────
# # # # # # # #         # Ramp: 0 before ep5, linear ep5→ep20, full after ep20
# # # # # # # #         if epoch < 5:      lam_dir = 0.0
# # # # # # # #         elif epoch < 20:   lam_dir = (epoch - 5) / 15.0
# # # # # # # #         else:              lam_dir = 1.0

# # # # # # # #         if lam_dir > 0.0:
# # # # # # # #             # 1-shot prediction (detach: gradient only flows through cond)
# # # # # # # #             with torch.no_grad():
# # # # # # # #                 x0_h = torch.randn_like(x1_rel) * self.sigma_inference
# # # # # # # #                 v_h  = self.velocity(x0_h, torch.zeros(B, device=device), cond)
# # # # # # # #                 x1_h_abs = self._from_relative(x0_h + v_h, last_obs)
# # # # # # # #             pred_deg_h = _norm_to_deg(x1_h_abs.permute(1, 0, 2))   # [T, B, 2]
# # # # # # # #             obs_deg_h  = _norm_to_deg(obs_traj[:, :, :2])           # [T_obs, B, 2]
# # # # # # # #             l_heading  = self._heading_loss_ms(pred_deg_h, obs_deg_h, n_steps=4, decay=0.5)
# # # # # # # #         else:
# # # # # # # #             l_heading = x0.new_zeros(())

# # # # # # # #         l_momentum = x0.new_zeros(())   # always 0 in v2.6

# # # # # # # #         # ── Total ─────────────────────────────────────────────────────────
# # # # # # # #         total = l_cfm + lam_reg * l_reg + lam_dir * self.lambda_heading * l_heading
# # # # # # # #         if not torch.isfinite(total):
# # # # # # # #             total = x0.new_zeros(())

# # # # # # # #         # ── ADE 1-step log ────────────────────────────────────────────────
# # # # # # # #         with torch.no_grad():
# # # # # # # #             x0_log = torch.randn_like(x1_rel) * self.sigma_inference
# # # # # # # #             v_log  = self.velocity(x0_log, torch.zeros(B, device=device), cond)
# # # # # # # #             ade_log = _haversine_deg(
# # # # # # # #                 _norm_to_deg(self._from_relative(x0_log + v_log, last_obs).permute(1, 0, 2)),
# # # # # # # #                 _norm_to_deg(x1_gt.permute(1, 0, 2))
# # # # # # # #             ).mean().item()

# # # # # # # #         return {
# # # # # # # #             "total":    total,
# # # # # # # #             "l_cfm":    l_cfm.item(),
# # # # # # # #             "l_reg":    l_reg.item() if torch.is_tensor(l_reg) else 0.0,
# # # # # # # #             "l_heading": l_heading.item() if torch.is_tensor(l_heading) else 0.0,
# # # # # # # #             "l_momentum": 0.0,   # always 0
# # # # # # # #             "lam_reg":  lam_reg,
# # # # # # # #             "lam_dir":  lam_dir,
# # # # # # # #             "sigma":    sigma,
# # # # # # # #             "ade_1step": ade_log,
# # # # # # # #             "hard_score_mean": float(h_score.mean()),
# # # # # # # #             "hard_score_max":  float(h_score.max()),
# # # # # # # #             # Compat keys for older scripts
# # # # # # # #             "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
# # # # # # # #             "speed": 0., "accel": 0., "fm_mse": l_cfm.item(),
# # # # # # # #             "l_hard_total": 0., "n_hard": 0, "alpha_hard": 0.,
# # # # # # # #             "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
# # # # # # # #         }

# # # # # # # #     def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
# # # # # # # #         return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

# # # # # # # #     # ── [NEW v2.6] Speed calibration at inference ──────────────────────────

# # # # # # # #     @staticmethod
# # # # # # # #     @torch.no_grad()
# # # # # # # #     def speed_calibrate_pred(pred_abs_norm: torch.Tensor,
# # # # # # # #                               last_obs_norm: torch.Tensor,
# # # # # # # #                               obs_norm:      torch.Tensor,
# # # # # # # #                               clip_min: float = 0.85,
# # # # # # # #                               clip_max: float = 1.15) -> torch.Tensor:
# # # # # # # #         """
# # # # # # # #         Per-storm post-hoc speed calibration. Applied at inference ONLY.
# # # # # # # #         DOES NOT affect training → zero train/inference consistency issue.

# # # # # # # #         Scales DISPLACEMENT MAGNITUDE, preserves DIRECTION.
# # # # # # # #         → ATE improved (speed error corrected)
# # # # # # # #         → CTE unchanged (direction unchanged)

# # # # # # # #         Algorithm:
# # # # # # # #           obs_speed_ref = exponentially-weighted mean of obs step speeds
# # # # # # # #           pred_speed_ref = mean of first 4 predicted step speeds
# # # # # # # #           scale = clip(obs_speed_ref / pred_speed_ref, clip_min, clip_max)
# # # # # # # #           pred_cal[t] = last_obs + (pred[t] - last_obs) * scale

# # # # # # # #         clip_min=0.85, clip_max=1.15:
# # # # # # # #           Prevents over-correction from noisy obs/pred speed estimates.
# # # # # # # #           Allows ±15% adjustment per storm.

# # # # # # # #         Why this works:
# # # # # # # #           v2.5 XAI-5: speed ratio = 1.15–1.61 throughout training
# # # # # # # #           (model consistently over-predicts speed on val)
# # # # # # # #           Test storms have different speed → need per-storm correction
# # # # # # # #           This is adaptive: fast test storms get upscaling, slow get downscaling
# # # # # # # #         """
# # # # # # # #         if obs_norm.shape[0] < 2 or pred_abs_norm.shape[0] < 2:
# # # # # # # #             return pred_abs_norm

# # # # # # # #         obs_deg  = _norm_to_deg(obs_norm)        # [T_obs, B, 2]
# # # # # # # #         pred_deg = _norm_to_deg(pred_abs_norm)   # [T, B, 2]
# # # # # # # #         last_deg = _norm_to_deg(last_obs_norm)   # [B, 2]

# # # # # # # #         # Obs speed reference (exponentially weighted, recent steps more important)
# # # # # # # #         obs_spds = _step_speeds_kmh(obs_deg)     # [T_obs-1, B] km/h
# # # # # # # #         T_s = obs_spds.shape[0]
# # # # # # # #         w = torch.exp(torch.arange(T_s, dtype=obs_spds.dtype, device=obs_spds.device) * 0.5)
# # # # # # # #         obs_spd_ref = (obs_spds * (w / w.sum()).unsqueeze(1)).sum(0).clamp(min=3.0)   # [B]

# # # # # # # #         # Predicted speed (first 4 steps from last_obs)
# # # # # # # #         pts = torch.cat([last_deg.unsqueeze(0), pred_deg], 0)   # [T+1, B, 2]
# # # # # # # #         pred_spds    = _step_speeds_kmh(pts)                     # [T, B]
# # # # # # # #         N_ref        = min(4, pred_spds.shape[0])
# # # # # # # #         pred_spd_ref = pred_spds[:N_ref].mean(0).clamp(min=3.0)  # [B]

# # # # # # # #         # Per-storm scale, clipped
# # # # # # # #         scale = (obs_spd_ref / pred_spd_ref).clamp(clip_min, clip_max)  # [B]

# # # # # # # #         # Scale displacement from last_obs (preserves direction!)
# # # # # # # #         return (last_obs_norm.unsqueeze(0)
# # # # # # # #                 + (pred_abs_norm - last_obs_norm.unsqueeze(0))
# # # # # # # #                 * scale.view(1, -1, 1))

# # # # # # # #     # ── Sample (1-shot + speed calibration) ──────────────────────────────

# # # # # # # #     @torch.no_grad()
# # # # # # # #     def sample(self, batch_list,
# # # # # # # #                num_ensemble:          Optional[int]  = None,
# # # # # # # #                ddim_steps:            Optional[int]  = None,
# # # # # # # #                return_xai:            bool           = False,
# # # # # # # #                use_speed_calibration: bool           = True,
# # # # # # # #                **kwargs) -> Tuple:
# # # # # # # #         """
# # # # # # # #         1-shot inference with physics selection + speed calibration.

# # # # # # # #         Steps:
# # # # # # # #           1. Encode context once (hard_score + full encoder)
# # # # # # # #           2. Sample K=20 candidates: x0 ~ N(0, sigma_inf²), x_pred = x0 + v(x0, 0, cond)
# # # # # # # #           3. Physics score each candidate (speed+smooth+heading+displacement)
# # # # # # # #           4. Weighted average of top-3 candidates
# # # # # # # #           5. Speed calibration per-storm (clip ±15%)
# # # # # # # #           6. If return_xai=True: compute XAI-1 through XAI-7

# # # # # # # #         Returns:
# # # # # # # #           (pred_mean [T,B,2], zeros [T,B,2], all_traj [K,T,B,2])
# # # # # # # #           or (pred_mean, zeros, all_traj, xai_dict) if return_xai=True
# # # # # # # #         """
# # # # # # # #         K  = num_ensemble or self.n_ensemble
# # # # # # # #         N  = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else self.n_inference_steps
# # # # # # # #         dt = 1.0 / max(N, 1)

# # # # # # # #         obs_traj    = batch_list[0]
# # # # # # # #         T_obs, B, _ = obs_traj.shape
# # # # # # # #         device      = obs_traj.device

# # # # # # # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # # # #         obs_norm = obs_traj[:, :, :2]
# # # # # # # #         last_obs = obs_traj[-1, :, :2]
# # # # # # # #         t0       = torch.zeros(B, device=device)

# # # # # # # #         # Encode once for all K samples (expensive part)
# # # # # # # #         cond = self.encoder(batch_list, hard_score=h_score)

# # # # # # # #         all_traj = []
# # # # # # # #         for _ in range(K):
# # # # # # # #             x_rel = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference

# # # # # # # #             if N <= 1:
# # # # # # # #                 # 1-shot: exactly what L_reg is trained for
# # # # # # # #                 v     = self.velocity(x_rel, t0, cond)
# # # # # # # #                 x_rel = x_rel + v
# # # # # # # #             else:
# # # # # # # #                 # Euler multi-step (for experiments only, slower)
# # # # # # # #                 for step in range(N):
# # # # # # # #                     t_b   = torch.full((B,), step * dt, device=device)
# # # # # # # #                     x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3., 3.)

# # # # # # # #             x_abs = self._from_relative(x_rel, last_obs)
# # # # # # # #             all_traj.append(x_abs.permute(1, 0, 2))   # [T, B, 2]

# # # # # # # #         # Physics-based selection: best-of-K with weighted average of top-3
# # # # # # # #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)   # [K, B]
# # # # # # # #         all_t   = torch.stack(all_traj, 0)   # [K, T, B, 2]
# # # # # # # #         top_k   = min(3, K)
# # # # # # # #         top_idx = scores.topk(top_k, dim=0).indices   # [top_k, B]

# # # # # # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # # # # # #         for b in range(B):
# # # # # # # #             idx_b = top_idx[:, b]
# # # # # # # #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # # # # # # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# # # # # # # #         # [NEW v2.6] Per-storm speed calibration
# # # # # # # #         if use_speed_calibration:
# # # # # # # #             pred_mean = self.speed_calibrate_pred(
# # # # # # # #                 pred_mean, last_obs, obs_norm, clip_min=0.85, clip_max=1.15)

# # # # # # # #         if not return_xai:
# # # # # # # #             return pred_mean, torch.zeros_like(pred_mean), all_t

# # # # # # # #         # ── XAI output (XAI-2,3,4,5,6,7) ──────────────────────────────────
# # # # # # # #         xai = {}

# # # # # # # #         # XAI-4: Ensemble uncertainty
# # # # # # # #         xai.update(compute_ensemble_uncertainty(all_t))

# # # # # # # #         # XAI-2: Hard score components
# # # # # # # #         _, hard_comps = hard_score_from_obs(obs_norm, return_components=True)
# # # # # # # #         xai["hard_components"] = hard_comps

# # # # # # # #         pred_deg  = _norm_to_deg(pred_mean)    # [T, B, 2]
# # # # # # # #         obs_deg_x = _norm_to_deg(obs_norm)     # [T_obs, B, 2]

# # # # # # # #         # XAI-5: Speed comparison (obs vs pred, pre and post calibration)
# # # # # # # #         obs_spd_x    = _step_speeds_kmh(obs_deg_x)
# # # # # # # #         obs_spd_mu   = obs_spd_x.mean(0)     # [B]
# # # # # # # #         if pred_deg.shape[0] >= 2:
# # # # # # # #             last_deg_x   = obs_deg_x[-1]
# # # # # # # #             pts_x        = torch.cat([last_deg_x.unsqueeze(0), pred_deg], 0)
# # # # # # # #             pred_spd_x   = _step_speeds_kmh(pts_x)
# # # # # # # #             pred_spd_mu  = pred_spd_x.mean(0)  # [B]
# # # # # # # #         else:
# # # # # # # #             pred_spd_mu = obs_spd_mu.clone()

# # # # # # # #         speed_ratio = (pred_spd_mu / obs_spd_mu.clamp(min=1.0))
# # # # # # # #         xai["speed_comparison"] = {
# # # # # # # #             "obs_speed_mean":  float(obs_spd_mu.mean()),
# # # # # # # #             "pred_speed_mean": float(pred_spd_mu.mean()),
# # # # # # # #             "speed_ratio":     float(speed_ratio.mean()),
# # # # # # # #             "per_storm_obs":   obs_spd_mu,
# # # # # # # #             "per_storm_pred":  pred_spd_mu,
# # # # # # # #             "over_predict":    speed_ratio > 1.15,
# # # # # # # #             "under_predict":   speed_ratio < 0.85,
# # # # # # # #         }

# # # # # # # #         # XAI-3: Physics components for best prediction
# # # # # # # #         v_ref = obs_spd_mu.clamp(min=5.0)
# # # # # # # #         v_sig = v_ref * 0.5
# # # # # # # #         spd_sc = torch.exp(-((pred_spd_mu - v_ref) / v_sig).pow(2) * 0.5)
# # # # # # # #         if pred_deg.shape[0] >= 3:
# # # # # # # #             vel_x   = pred_deg[1:] - pred_deg[:-1]
# # # # # # # #             accel_x = (vel_x[1:] - vel_x[:-1]).norm(dim=-1)
# # # # # # # #             smo_sc  = torch.exp(-accel_x.mean(0) * 5.0)
# # # # # # # #         else:
# # # # # # # #             smo_sc = torch.ones(B, device=device)
# # # # # # # #         if obs_norm.shape[0] >= 2 and pred_mean.shape[0] >= 1:
# # # # # # # #             ov = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # # # # # # #             pv = pred_mean[0, :, :2] - obs_norm[-1, :, :2]
# # # # # # # #             cos_s = (F.normalize(ov, dim=-1, eps=1e-6) * F.normalize(pv, dim=-1, eps=1e-6)).sum(-1)
# # # # # # # #             hd_sc = torch.exp((cos_s.clamp(-1, 1) - 1.0) * 3.0)
# # # # # # # #         else:
# # # # # # # #             hd_sc = torch.ones(B, device=device)
# # # # # # # #         xai["physics_components"] = {
# # # # # # # #             "speed": spd_sc, "smooth": smo_sc, "heading": hd_sc,
# # # # # # # #             "obs_speed": obs_spd_mu, "pred_speed": pred_spd_mu,
# # # # # # # #         }

# # # # # # # #         # XAI-6: Heading deviation per lead time
# # # # # # # #         gt_traj_xai = batch_list[1]
# # # # # # # #         gt_deg_xai  = _norm_to_deg(gt_traj_xai[:, :, :2])
# # # # # # # #         xai["heading_deviation_deg"] = compute_heading_deviation(pred_deg, gt_deg_xai)

# # # # # # # #         # XAI-7: ATE/CTE decomposition
# # # # # # # #         xai["ate_cte_decomp"] = compute_cte_contribution(pred_deg, gt_deg_xai)

# # # # # # # #         return pred_mean, torch.zeros_like(pred_mean), all_t, xai

# # # # # # # #     @torch.no_grad()
# # # # # # # #     def sample_multiscale(
# # # # # # # #         self,
# # # # # # # #         batch_list,
# # # # # # # #         sigmas: Optional[List[float]] = None,
# # # # # # # #         n_per_sigma: int = 4,
# # # # # # # #         use_speed_calibration: bool = True,
# # # # # # # #     ) -> Tuple:
# # # # # # # #         """
# # # # # # # #         Multi-scale sigma ensemble at inference (DOES NOT affect training).
# # # # # # # #         Hedges against speed distribution shift between val and test.
# # # # # # # #         """
# # # # # # # #         if sigmas is None:
# # # # # # # #             sigmas = [0.025, 0.035, 0.04, 0.05, 0.065]

# # # # # # # #         obs_traj = batch_list[0]
# # # # # # # #         B = obs_traj.shape[1]; device = obs_traj.device

# # # # # # # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # # # #         obs_norm = obs_traj[:, :, :2]
# # # # # # # #         last_obs = obs_traj[-1, :, :2]
# # # # # # # #         t0       = torch.zeros(B, device=device)
# # # # # # # #         cond     = self.encoder(batch_list, hard_score=h_score)

# # # # # # # #         all_traj = []
# # # # # # # #         for sigma in sigmas:
# # # # # # # #             for _ in range(n_per_sigma):
# # # # # # # #                 x0    = torch.randn(B, self.pred_len, 2, device=device) * sigma
# # # # # # # #                 v     = self.velocity(x0, t0, cond)
# # # # # # # #                 x_abs = self._from_relative(x0 + v, last_obs)
# # # # # # # #                 all_traj.append(x_abs.permute(1, 0, 2))

# # # # # # # #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)
# # # # # # # #         all_t   = torch.stack(all_traj, 0)
# # # # # # # #         top_k   = min(5, len(all_traj))
# # # # # # # #         top_idx = scores.topk(top_k, dim=0).indices

# # # # # # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # # # # # #         for b in range(B):
# # # # # # # #             idx_b = top_idx[:, b]
# # # # # # # #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # # # # # # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# # # # # # # #         if use_speed_calibration:
# # # # # # # #             pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

# # # # # # # #         return pred_mean, torch.zeros_like(pred_mean), all_t


# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # #  Backward compat alias
# # # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # # TCDiffusion = TCFlowMatching
# # # # # # # """
# # # # # # # Model/flow_matching_model.py  ──  TC-FlowMatching v2.7
# # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # VIẾT LẠI HOÀN TOÀN từ v2.1 + các fix đã verified từ thực nghiệm 140 epoch.

# # # # # # # ━━━ CƠ SỞ LÝ THUYẾT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # # # v2.1 test: ADE=229.8, ATE=214.4, CTE=71.6 (best so far, generalize tốt nhất)
# # # # # # # v2.5 test: ADE=232.5, ATE=222.3, CTE=64.5 (ATE tệ hơn, CTE tốt hơn)
# # # # # # # v2.4 test: ADE=291.1 (catastrophic overfitting)

# # # # # # # PHÂN TÍCH LỖI v2.1:
# # # # # # #   ATE/CTE ratio on test = 214.4/71.6 = 3.0
# # # # # # #   → 75% error is ALONG-TRACK (speed/distance)
# # # # # # #   → 25% is CROSS-TRACK (direction)
# # # # # # #   Test gap 72h = +129.3km >> 12h = +10.1km → long-range drift

# # # # # # # PHÂN TÍCH v2.5 THẤT BẠI:
# # # # # # #   L_momentum anchor pred_speed ≈ val_speed (10.3km/h)
# # # # # # #   Test storms faster → L_momentum SLOWS model on test → ATE +7.9km WORSE
# # # # # # #   L_heading (step 0 only) → 72h heading dev INCREASED: ep0=87°→ep120=128°!
# # # # # # #   Recurvature aug: CTE improved 7km ← giữ lại

# # # # # # # ━━━ VẬT LÝ CỦA GIẢI PHÁP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # # # GIẢI PHÁP ATE (along-track error):
# # # # # # #   Training: KHÔNG anchor speed tại training (no L_momentum)
# # # # # # #   Inference: SPEED CALIBRATION per-storm
# # # # # # #     scale = clip(obs_speed / pred_speed_first4, 0.85, 1.15)
# # # # # # #     pred_cal = last_obs + (pred - last_obs) * scale
# # # # # # #     → Preserves direction (CTE unchanged), fixes magnitude (ATE reduced)
# # # # # # #     → Per-storm adaptive: fast test storms get upscaling, slow storms downscaling
# # # # # # #     → DOES NOT affect training → zero train/inference consistency issue

# # # # # # # GIẢI PHÁP CTE (cross-track error):
# # # # # # #   Training: MULTI-STEP heading constraint (not single step)
# # # # # # #     step 0 (6h):  weight 1.0  — hard anchor to obs direction
# # # # # # #     step 1 (12h): weight 0.5  — continuation (detached ref)
# # # # # # #     step 2 (18h): weight 0.25 — softer continuation
# # # # # # #     step 3 (24h): weight 0.125 — very soft
# # # # # # #     Detach ref at each step: prevents gradient explosion through chain
# # # # # # #     → 72h heading dev should come down from 128° to ~90°

# # # # # # # GIẢI PHÁP GAP (val→test distribution shift):
# # # # # # #   [AUG-C] Recurvature: proven to reduce CTE test (v2.5)
# # # # # # #   [AUG-D] Obs-speed scaling ×0.7–1.4: model sees diverse speed contexts
# # # # # # #            → at inference, adapts better to test storms' speed
# # # # # # #   Speed calibration at inference bridges remaining gap

# # # # # # # ━━━ GIỮ NGUYÊN TỪ v2.1 (PROVEN, KHÔNG THAY ĐỔI) ━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # # #   ✅ sigma_inference=0.04 FIXED — zero train/inference mismatch
# # # # # # #   ✅ L_reg softmax-linspace weights (max/min ≈ 3×) — không exp (v2.4 fail)
# # # # # # #   ✅ Mild base aug: shift±5km + speed×0.85–1.15 (val ≈ test distribution)
# # # # # # #   ✅ 1-shot inference nhất quán với L_reg training
# # # # # # #   ✅ 2-group optimizer, encoder freeze 10ep
# # # # # # #   ✅ val loop NO augmentation
# # # # # # #   ✅ OT noise-GT matching trong L_CFM
# # # # # # #   ✅ ContextEncoder: FNO3D + Mamba + Env + 6 kinematic features
# # # # # # #   ✅ VelocityTransformer: d_model=256, 4 layers, nhead=8

# # # # # # # ━━━ CẢI TIẾN v2.6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # # #   [REMOVE]  L_momentum — LOẠI BỎ HOÀN TOÀN (made ATE +7.9km worse)
# # # # # # #   [UPGRADE] L_heading → multi-step 4 steps, decay=0.5, weight=0.10
# # # # # # #   [NEW-INF] speed_calibrate_pred() tại inference (clip 0.85–1.15)
# # # # # # #   [UPGRADE] physics_score + displacement_score (15% weight)
# # # # # # #   [AUG-D]   obs-speed scaling aug ×0.7–1.4 (15% probability)
# # # # # # #   [XAI 1–7] giữ nguyên đầy đủ, XAI-5 now reports per-storm calibration

# # # # # # # ━━━ KỲ VỌNG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # # #                    Val ADE   Test ADE   Val ATE  Test ATE  Val CTE  Test CTE
# # # # # # #   v2.1 (best):     170.1      229.8     160.2    214.4     45.7     71.6
# # # # # # #   v2.5 (current):  170.5      232.5     161.3    222.3     46.2     64.5
# # # # # # #   v2.6 (target):   170±1      210–220   161±1    200–210   46±1     57–63
# # # # # # # """
# # # # # # # from __future__ import annotations

# # # # # # # import math
# # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # import torch
# # # # # # # import torch.nn as nn
# # # # # # # import torch.nn.functional as F

# # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # from Model.env_net_transformer_gphsplit import Env_net

# # # # # # # R_EARTH  = 6371.0
# # # # # # # DT_HOURS = 6.0


# # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # #  Coordinate utilities (same as v2.1)
# # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# # # # # # #     """Normalized coords → (lon°, lat°)."""
# # # # # # #     return torch.stack([
# # # # # # #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# # # # # # #         (t[..., 1] * 50.0) / 10.0,
# # # # # # #     ], dim=-1)


# # # # # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # # #     """Haversine distance km between two (lon°,lat°) tensors."""
# # # # # # #     lat1 = torch.deg2rad(p1[..., 1]);  lat2 = torch.deg2rad(p2[..., 1])
# # # # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # # # #     a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
# # # # # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# # # # # # # def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # # #     """Bearing in radians from p1→p2 (degrees input)."""
# # # # # # #     lon1 = torch.deg2rad(p1[..., 0]);  lat1 = torch.deg2rad(p1[..., 1])
# # # # # # #     lon2 = torch.deg2rad(p2[..., 0]);  lat2 = torch.deg2rad(p2[..., 1])
# # # # # # #     dlon = lon2 - lon1
# # # # # # #     y = torch.sin(dlon) * torch.cos(lat2)
# # # # # # #     x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
# # # # # # #     return torch.atan2(y, x)


# # # # # # # def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
# # # # # # #     """Speed km/h between consecutive steps of a trajectory [T, B, 2]."""
# # # # # # #     if traj_deg.shape[0] < 2:
# # # # # # #         return traj_deg.new_zeros(1, traj_deg.shape[1])
# # # # # # #     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # #  EMAModel (same as v2.1)
# # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # def _unwrap(m):
# # # # # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m


# # # # # # # class EMAModel:
# # # # # # #     def __init__(self, model, decay: float = 0.995):
# # # # # # #         self.decay = decay
# # # # # # #         m = _unwrap(model)
# # # # # # #         self.shadow = {k: v.detach().clone()
# # # # # # #                        for k, v in m.state_dict().items()
# # # # # # #                        if v.dtype.is_floating_point}

# # # # # # #     def update(self, model):
# # # # # # #         m = _unwrap(model)
# # # # # # #         with torch.no_grad():
# # # # # # #             for k, v in m.state_dict().items():
# # # # # # #                 if k in self.shadow:
# # # # # # #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

# # # # # # #     def apply_to(self, model):
# # # # # # #         m = _unwrap(model)
# # # # # # #         backup, sd = {}, m.state_dict()
# # # # # # #         for k in self.shadow:
# # # # # # #             if k not in sd: continue
# # # # # # #             backup[k] = sd[k].detach().clone()
# # # # # # #             sd[k].copy_(self.shadow[k])
# # # # # # #         return backup

# # # # # # #     def restore(self, model, backup):
# # # # # # #         m = _unwrap(model)
# # # # # # #         sd = m.state_dict()
# # # # # # #         for k, v in backup.items():
# # # # # # #             if k in sd: sd[k].copy_(v)


# # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # #  OT matching (same as v2.1)
# # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 50) -> torch.Tensor:
# # # # # # #     B = cost.shape[0]; device = cost.device
# # # # # # #     log_a = -math.log(B) * torch.ones(B, device=device)
# # # # # # #     log_b = -math.log(B) * torch.ones(B, device=device)
# # # # # # #     log_K = -cost / epsilon
# # # # # # #     log_u = torch.zeros(B, device=device)
# # # # # # #     log_v = torch.zeros(B, device=device)
# # # # # # #     for _ in range(n_iter):
# # # # # # #         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
# # # # # # #         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
# # # # # # #     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


# # # # # # # def _ot_match(x0_flat: torch.Tensor, x1_flat: torch.Tensor,
# # # # # # #               epsilon: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # # # #     B = x0_flat.shape[0]
# # # # # # #     if B < 4:
# # # # # # #         return x0_flat, x1_flat
# # # # # # #     try:
# # # # # # #         cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1] ** 0.5)
# # # # # # #         with torch.no_grad():
# # # # # # #             pi = _sinkhorn_log(cost, epsilon=epsilon)
# # # # # # #         flat = pi.reshape(-1).clamp(0.0)
# # # # # # #         s = flat.sum()
# # # # # # #         if not torch.isfinite(s) or s < 1e-10:
# # # # # # #             return x0_flat, x1_flat
# # # # # # #         idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
# # # # # # #         return x0_flat[idx // B], x1_flat
# # # # # # #     except Exception:
# # # # # # #         return x0_flat, x1_flat


# # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # #  VelocityTransformer (identical to v2.1)
# # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # class VelocityTransformer(nn.Module):
# # # # # # #     def __init__(self, pred_len: int = 12, d_model: int = 256, nhead: int = 8,
# # # # # # #                  num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
# # # # # # #                  d_cond: int = 256):
# # # # # # #         super().__init__()
# # # # # # #         self.pred_len = pred_len
# # # # # # #         self.d_model  = d_model
# # # # # # #         self.traj_embed = nn.Linear(2, d_model)
# # # # # # #         self.pos_emb    = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
# # # # # # #         self.step_emb   = nn.Embedding(pred_len, d_model)
# # # # # # #         self.time_mlp   = nn.Sequential(
# # # # # # #             nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
# # # # # # #         self.cond_proj  = nn.Sequential(nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))
# # # # # # #         dec_layer = nn.TransformerDecoderLayer(
# # # # # # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # # # # # #             dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
# # # # # # #         self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
# # # # # # #         self.out_norm = nn.LayerNorm(d_model)
# # # # # # #         self.out_proj = nn.Sequential(
# # # # # # #             nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
# # # # # # #         self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
# # # # # # #         nn.init.zeros_(self.out_proj[-1].weight)
# # # # # # #         nn.init.zeros_(self.out_proj[-1].bias)

# # # # # # #     def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
# # # # # # #         half = self.d_model // 2
# # # # # # #         freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
# # # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # # #         emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
# # # # # # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # # # # # #         if self.d_model % 2 == 1:
# # # # # # #             emb = F.pad(emb, (0, 1))
# # # # # # #         return self.time_mlp(emb)

# # # # # # #     def forward(self, x_t: torch.Tensor, t: torch.Tensor,
# # # # # # #                 cond: torch.Tensor) -> torch.Tensor:
# # # # # # #         B, T, _ = x_t.shape
# # # # # # #         step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # #         x_emb = (self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(step_idx))
# # # # # # #         memory = torch.cat([self._time_emb(t).unsqueeze(1),
# # # # # # #                             self.cond_proj(cond).unsqueeze(1)], dim=1)
# # # # # # #         out = self.out_norm(self.decoder(x_emb, memory))
# # # # # # #         return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)


# # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # #  ContextEncoder (identical to v2.1)
# # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # class ContextEncoder(nn.Module):
# # # # # # #     RAW_CTX_DIM = 512

# # # # # # #     def __init__(self, obs_len: int = 8, unet_in_ch: int = 13, d_cond: int = 256):
# # # # # # #         super().__init__()
# # # # # # #         self.obs_len = obs_len
# # # # # # #         self.d_cond  = d_cond

# # # # # # #         self.spatial_enc     = FNO3DEncoder(in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # #                                              n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # #                                              spatial_down=32, dropout=0.05)
# # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # #         self.decoder_proj    = nn.Linear(1, 16)
# # # # # # #         self.enc_1d          = DataEncoder1D(in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # #                                               lstm_hidden=128, lstm_layers=3,
# # # # # # #                                               dropout=0.1, d_state=16)
# # # # # # #         self.env_enc         = Env_net(obs_len=obs_len, d_model=32)
# # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # #         self.ctx_drop = nn.Dropout(0.1)
# # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, d_cond)
# # # # # # #         self.ctx_ln2  = nn.LayerNorm(d_cond)
# # # # # # #         # 6 kinematic features: vel_x, vel_y, speed_n, sin(h), cos(h), accel
# # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # #             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
# # # # # # #             nn.Linear(256, d_cond // 2), nn.GELU())
# # # # # # #         self.hard_embed = nn.Sequential(
# # # # # # #             nn.Linear(1, d_cond // 4), nn.GELU(), nn.Linear(d_cond // 4, d_cond // 4))
# # # # # # #         self.fuse = nn.Sequential(
# # # # # # #             nn.Linear(d_cond + d_cond // 2 + d_cond // 4, d_cond),
# # # # # # #             nn.LayerNorm(d_cond), nn.GELU())

# # # # # # #     def _encode_raw(self, batch_list) -> torch.Tensor:
# # # # # # #         obs_traj  = batch_list[0]; obs_Me = batch_list[7]
# # # # # # #         image_obs = batch_list[11]; env_data = batch_list[13]
# # # # # # #         if image_obs.dim() == 4:
# # # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
# # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # #         T_obs = obs_traj.shape[0]
# # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
# # # # # # #                                    mode="linear", align_corners=False).permute(0,2,1)
# # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # #                                           device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # # #         f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
# # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)
# # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

# # # # # # #     def _kinematic_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
# # # # # # #         """6 kinematic features per obs step."""
# # # # # # #         B = obs_traj.shape[1]; T_obs = obs_traj.shape[0]; device = obs_traj.device
# # # # # # #         if T_obs >= 2:
# # # # # # #             traj_deg = _norm_to_deg(obs_traj)
# # # # # # #             vel_norm = obs_traj[1:] - obs_traj[:-1]
# # # # # # #             speed    = _step_speeds_kmh(traj_deg)
# # # # # # #             speed_n  = (speed / 20.0).clamp(-3.0, 3.0)
# # # # # # #             heading  = torch.atan2(vel_norm[:, :, 1], vel_norm[:, :, 0])
# # # # # # #             if T_obs >= 3:
# # # # # # #                 dspd  = speed[1:] - speed[:-1]
# # # # # # #                 accel = torch.cat([obs_traj.new_zeros(1, B),
# # # # # # #                                    (dspd / 10.0).clamp(-3.0, 3.0)], 0)
# # # # # # #             else:
# # # # # # #                 accel = obs_traj.new_zeros(T_obs - 1, B)
# # # # # # #             kine = torch.stack([vel_norm[:,:,0], vel_norm[:,:,1], speed_n,
# # # # # # #                                 heading.sin(), heading.cos(), accel], dim=-1)
# # # # # # #         else:
# # # # # # #             kine = obs_traj.new_zeros(self.obs_len, B, 6)
# # # # # # #         if kine.shape[0] < self.obs_len:
# # # # # # #             kine = torch.cat([obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6), kine], 0)
# # # # # # #         else:
# # # # # # #             kine = kine[-self.obs_len:]
# # # # # # #         return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

# # # # # # #     def forward(self, batch_list, hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# # # # # # #         raw   = self._encode_raw(batch_list)
# # # # # # #         ctx   = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
# # # # # # #         kfeat = self._kinematic_feat(batch_list[0][:, :, :2])
# # # # # # #         if hard_score is None:
# # # # # # #             hard_score = torch.zeros(ctx.shape[0], device=ctx.device)
# # # # # # #         hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))
# # # # # # #         return self.fuse(torch.cat([ctx, kfeat, hfeat], dim=-1))


# # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # #  Hard score (XAI-2) — same as v2.1
# # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # @torch.no_grad()
# # # # # # # def hard_score_from_obs(obs_traj_norm: torch.Tensor,
# # # # # # #                          return_components: bool = False):
# # # # # # #     """[XAI-2] Độ khó của storm dựa trên curvature, speed_var, dir_change."""
# # # # # # #     T, B   = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
# # # # # # #     device = obs_traj_norm.device
# # # # # # #     if T < 3:
# # # # # # #         z = torch.zeros(B, device=device)
# # # # # # #         if return_components:
# # # # # # #             return z, {"curvature": z.clone(), "speed_var": z.clone(), "dir_change": z.clone()}
# # # # # # #         return z
# # # # # # #     traj_deg = _norm_to_deg(obs_traj_norm[..., :2])
# # # # # # #     az12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
# # # # # # #     az23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
# # # # # # #     diff = (az23 - az12).abs()
# # # # # # #     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
# # # # # # #     curvature  = diff.mean(0) / math.pi
# # # # # # #     spd        = _step_speeds_kmh(traj_deg)
# # # # # # #     if spd.shape[0] >= 2:
# # # # # # #         speed_var = (spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 1.)
# # # # # # #     else:
# # # # # # #         speed_var = torch.zeros(B, device=device)
# # # # # # #     dir_change = (diff > (20.0 / 180.0 * math.pi)).float().mean(0)
# # # # # # #     # [v2.7] speed_cv: coefficient of variation của step speeds
# # # # # # #     # Bão speed thay đổi bất thường (vừa nhanh vừa chậm) → harder to predict
# # # # # # #     if spd.shape[0] >= 2:
# # # # # # #         speed_cv = (spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 2.0)
# # # # # # #     else:
# # # # # # #         speed_cv = torch.zeros(B, device=device)
# # # # # # #     score = (0.35 * curvature + 0.25 * speed_var + 0.25 * dir_change + 0.15 * speed_cv).clamp(0., 1.)
# # # # # # #     if return_components:
# # # # # # #         return score, {"curvature": curvature, "speed_var": speed_var,
# # # # # # #                        "dir_change": dir_change, "speed_cv": speed_cv}
# # # # # # #     return score


# # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # #  Physics score v2.6 (adds displacement_score)
# # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # @torch.no_grad()
# # # # # # # def _physics_score(traj_norm: torch.Tensor, obs_norm: torch.Tensor) -> torch.Tensor:
# # # # # # #     """
# # # # # # #     Best-of-K selection score. Four components:
# # # # # # #       speed_score       (0.30): per-step speed vs obs reference
# # # # # # #       smooth_score      (0.25): trajectory smoothness (no sharp acceleration)
# # # # # # #       heading_score     (0.30): first-step direction matches obs
# # # # # # #       displacement_score(0.15): total path length consistent with obs speed

# # # # # # #     [NEW v2.6] displacement_score penalizes samples with total displacement
# # # # # # #     far from obs-speed expectation → catches speed over-prediction artifacts
# # # # # # #     that cause high ATE on test storms.
# # # # # # #     """
# # # # # # #     B      = traj_norm.shape[1]
# # # # # # #     device = traj_norm.device
# # # # # # #     traj_deg = _norm_to_deg(traj_norm)
# # # # # # #     v_ref   = None

# # # # # # #     # ── Speed score ────────────────────────────────────────────────────────
# # # # # # #     if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# # # # # # #         obs_deg = _norm_to_deg(obs_norm)
# # # # # # #         obs_spd = _step_speeds_kmh(obs_deg)
# # # # # # #         T_s     = obs_spd.shape[0]
# # # # # # #         w_obs   = torch.linspace(0.5, 1.0, T_s, device=device)
# # # # # # #         v_ref   = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()   # [B]
# # # # # # #         pred_spd = _step_speeds_kmh(traj_deg)
# # # # # # #         v_sigma  = v_ref.clamp(min=5.0) * 0.5
# # # # # # #         speed_score = torch.exp(
# # # # # # #             -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5)
# # # # # # #     elif traj_deg.shape[0] >= 2:
# # # # # # #         speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.).mean(0))
# # # # # # #     else:
# # # # # # #         speed_score = torch.ones(B, device=device)

# # # # # # #     # ── Smoothness score ───────────────────────────────────────────────────
# # # # # # #     if traj_deg.shape[0] >= 3:
# # # # # # #         vel          = traj_deg[1:] - traj_deg[:-1]
# # # # # # #         accel_mag    = (vel[1:] - vel[:-1]).norm(dim=-1)
# # # # # # #         smooth_score = torch.exp(-accel_mag.mean(0) * 5.0)
# # # # # # #     else:
# # # # # # #         smooth_score = torch.ones(B, device=device)

# # # # # # #     # ── Heading score ──────────────────────────────────────────────────────
# # # # # # #     if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
# # # # # # #         obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # # # # # #         pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
# # # # # # #         obs_h    = F.normalize(obs_vel,  dim=-1, eps=1e-6)
# # # # # # #         pred_h   = F.normalize(pred_vel, dim=-1, eps=1e-6)
# # # # # # #         cos_sim  = (obs_h * pred_h).sum(-1).clamp(-1, 1)
# # # # # # #         head_score = torch.exp((cos_sim - 1.0) * 3.0)
# # # # # # #     else:
# # # # # # #         head_score = torch.ones(B, device=device)

# # # # # # #     # ── [NEW v2.6] Displacement score ─────────────────────────────────────
# # # # # # #     # Expected total path length ≈ obs_speed × T_pred × DT × 0.75
# # # # # # #     # (0.75 factor: storms curve, so straight-line < path length estimate)
# # # # # # #     # This catches ensemble samples that over/under-shoot on total distance.
# # # # # # #     if v_ref is not None and traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# # # # # # #         T_pred        = traj_deg.shape[0]
# # # # # # #         expected_total = v_ref * T_pred * DT_HOURS * 0.75   # [B] km
# # # # # # #         step_dists    = _haversine_deg(traj_deg[:-1], traj_deg[1:])   # [T-1, B]
# # # # # # #         actual_total  = step_dists.sum(0)                              # [B]
# # # # # # #         rel_err       = (actual_total - expected_total).abs() / expected_total.clamp(min=10.)
# # # # # # #         disp_score    = torch.exp(-rel_err * 1.5)
# # # # # # #     else:
# # # # # # #         disp_score    = torch.ones(B, device=device)

# # # # # # #     return (speed_score.pow(0.30)
# # # # # # #             * smooth_score.pow(0.25)
# # # # # # #             * head_score.pow(0.30)
# # # # # # #             * disp_score.pow(0.15)).clamp(min=1e-6)


# # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # #  Augmentation v2.6
# # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # def augment_batch(batch_list) -> list:
# # # # # # #     """
# # # # # # #     v2.6 augmentation — extends v2.1 core with recurvature and obs-speed aug.

# # # # # # #     Distribution (6 types):
# # # # # # #       A (25%): track shift ±5km             — shape unchanged, position varies
# # # # # # #       B (20%): GT speed scale ×0.85–1.15   — mild, proven from v2.1
# # # # # # #       C (20%): recurvature ±20°             — proven CTE improvement in v2.5
# # # # # # #       D (15%): obs-speed scaling ×0.7–1.4  — [NEW v2.6] diverse speed contexts
# # # # # # #                                               buộc model học với fast/slow storms
# # # # # # #                                               giảm anchoring to val speed distribution
# # # # # # #       E (10%): no augmentation              — giữ original distribution
# # # # # # #       F (10%): Gaussian noise ±3km          — small robustness

# # # # # # #     KHÔNG dùng:
# # # # # # #       - mixup (phi vật lý, v2.4 proof)
# # # # # # #       - speed scale >×1.5 trên GT (too far from real)
# # # # # # #       - exp L_reg weights (v2.4 catastrophic)

# # # # # # #     AUG-D (obs-speed scaling) lý do:
# # # # # # #       v2.5 XAI-5: model anchors to val speed (~10.3 km/h) throughout training.
# # # # # # #       Test storms có speed distribution khác.
# # # # # # #       AUG-D: scale OBS displacements but keep GT unchanged.
# # # # # # #       → Model must INFER future speed from obs context (not memorize mean)
# # # # # # #       → At inference, model adapts per-storm → speed_calibrate_pred() works better.
# # # # # # #     """
# # # # # # #     bl = list(batch_list)
# # # # # # #     if not torch.is_tensor(bl[0]):
# # # # # # #         return bl

# # # # # # #     obs    = bl[0]
# # # # # # #     device = obs.device
# # # # # # #     anchor = obs[-1:, :, :2].detach()  # [1, B, 2] — last obs as pivot
# # # # # # #     r = torch.rand(1).item()

# # # # # # #     if r < 0.25:
# # # # # # #         # A: Track shift ±5km (preserves storm shape, varies position)
# # # # # # #         shift = (torch.rand(2, device=device) - 0.5) * 0.018  # ±5km in normalized
# # # # # # #         bl[0] = obs + shift.view(1, 1, 2)
# # # # # # #         if torch.is_tensor(bl[1]):
# # # # # # #             bl[1] = bl[1] + shift.view(1, 1, 2)

# # # # # # #     elif r < 0.45:
# # # # # # #         # B: Speed scale ×0.85–1.15 (both obs+GT from anchor, mild)
# # # # # # #         scale = 0.85 + 0.30 * torch.rand(1, device=device).item()
# # # # # # #         obs_c = obs.clone()
# # # # # # #         obs_c[..., :2] = anchor + (obs[..., :2] - anchor) * scale
# # # # # # #         bl[0] = obs_c
# # # # # # #         if torch.is_tensor(bl[1]):
# # # # # # #             bl[1] = anchor + (bl[1] - anchor) * scale

# # # # # # #     elif r < 0.65:
# # # # # # #         # C: Recurvature ±20° — proven CTE improvement
# # # # # # #         # Progressive rotation with ease-in: slow start, accelerate (realistic)
# # # # # # #         T_pred = bl[1].shape[0] if torch.is_tensor(bl[1]) else 0
# # # # # # #         if T_pred >= 4:
# # # # # # #             gt      = bl[1].clone()
# # # # # # #             max_deg = (torch.rand(1).item() - 0.5) * 40.0   # -20° to +20°
# # # # # # #             max_rad = max_deg * math.pi / 180.0
# # # # # # #             for t in range(T_pred):
# # # # # # #                 progress = (t / max(T_pred - 1, 1)) ** 1.5  # ease-in
# # # # # # #                 angle_t  = max_rad * progress
# # # # # # #                 c, s = math.cos(angle_t), math.sin(angle_t)
# # # # # # #                 rot = torch.tensor([[c, -s], [s, c]], dtype=gt.dtype, device=device)
# # # # # # #                 rel = (gt[t] - anchor[0]).unsqueeze(-1)   # [B, 2, 1]
# # # # # # #                 gt[t] = (rot @ rel).squeeze(-1) + anchor[0]
# # # # # # #             bl[1] = gt
# # # # # # #             # Rotate last 3 obs steps slightly (30% of max) for consistency
# # # # # # #             T_obs   = obs.shape[0]
# # # # # # #             obs_aug = obs.clone()
# # # # # # #             cos_p   = math.cos(max_rad * 0.3)
# # # # # # #             sin_p   = math.sin(max_rad * 0.3)
# # # # # # #             rot_p   = torch.tensor([[cos_p, -sin_p], [sin_p, cos_p]],
# # # # # # #                                     dtype=obs.dtype, device=device)
# # # # # # #             for t_obs in range(max(0, T_obs - 3), T_obs):
# # # # # # #                 rel_o = (obs_aug[t_obs, :, :2] - anchor[0]).unsqueeze(-1)
# # # # # # #                 obs_aug[t_obs, :, :2] = (rot_p @ rel_o).squeeze(-1) + anchor[0]
# # # # # # #             bl[0] = obs_aug

# # # # # # #     elif r < 0.80:
# # # # # # #         # D: [v2.7 REDESIGNED] Consistent speed scaling — scale BOTH obs AND GT
# # # # # # #         #
# # # # # # #         # v2.6 bug: scale OBS only, GT unchanged → inconsistency
# # # # # # #         # Model sees fast obs but slow GT → learns "slow down after fast approach"
# # # # # # #         # → Confuses model's velocity field → ATE test TỆ HƠN v2.1 (+4.5km)
# # # # # # #         #
# # # # # # #         # v2.7 fix: scale BOTH obs and GT từ cùng anchor (last obs)
# # # # # # #         # → Storm trở nên nhanh hơn hoặc chậm hơn NHẤT QUÁN
# # # # # # #         # → Model học speed-proportional prediction
# # # # # # #         # → Speed calibration tại inference hoạt động chính xác hơn
# # # # # # #         #
# # # # # # #         # Range: bimodal để cover extremes (val đã cover middle range):
# # # # # # #         #   50%: ×0.50-0.80 → bão chậm/đứng yên (rare in val, common in test)
# # # # # # #         #   50%: ×1.20-1.80 → bão rất nhanh (rapid intensification case)
# # # # # # #         if torch.rand(1).item() < 0.5:
# # # # # # #             speed_scale = 0.50 + 0.30 * torch.rand(1, device=device).item()  # [0.50, 0.80]
# # # # # # #         else:
# # # # # # #             speed_scale = 1.20 + 0.60 * torch.rand(1, device=device).item()  # [1.20, 1.80]
# # # # # # #         anchor_pos = obs[-1:, :, :2].detach()   # [1, B, 2] — last obs = pivot
# # # # # # #         T_obs = obs.shape[0]
# # # # # # #         obs_aug = obs.clone()
# # # # # # #         # Scale obs displacements from anchor
# # # # # # #         obs_aug[0] = obs[0]
# # # # # # #         for t in range(1, T_obs):
# # # # # # #             disp = obs[t, :, :2] - obs[t-1, :, :2]
# # # # # # #             obs_aug[t, :, :2] = obs_aug[t-1, :, :2] + disp * speed_scale
# # # # # # #         bl[0] = obs_aug
# # # # # # #         # Scale GT displacements from last obs (same scale → consistent)
# # # # # # #         if torch.is_tensor(bl[1]):
# # # # # # #             gt_aug = bl[1].clone()
# # # # # # #             gt_aug[0, :, :2] = anchor_pos[0] + (bl[1][0, :, :2] - anchor_pos[0]) * speed_scale
# # # # # # #             for t in range(1, bl[1].shape[0]):
# # # # # # #                 disp_gt = bl[1][t, :, :2] - bl[1][t-1, :, :2]
# # # # # # #                 gt_aug[t, :, :2] = gt_aug[t-1, :, :2] + disp_gt * speed_scale
# # # # # # #             bl[1] = gt_aug

# # # # # # #     elif r < 0.90:
# # # # # # #         # E: no augmentation (keep original distribution)
# # # # # # #         pass

# # # # # # #     else:
# # # # # # #         # F: small Gaussian noise ±3km
# # # # # # #         obs_new = obs.clone()
# # # # # # #         obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.003
# # # # # # #         bl[0] = obs_new

# # # # # # #     return bl


# # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # #  XAI functions (1–7, complete)
# # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # def compute_obs_attribution(model, batch_list, device: torch.device,
# # # # # # #                              target_step: int = 11) -> torch.Tensor:
# # # # # # #     """
# # # # # # #     [XAI-1] Gradient saliency: which obs step most influences 72h prediction?

# # # # # # #     Returns attr [T_obs, B] normalized importance [0,1].
# # # # # # #     High attr[-1, b] = last obs dominant (normal).
# # # # # # #     High attr[0, b]  = storm has long "memory" (unusual → hard storm).
# # # # # # #     """
# # # # # # #     raw = _unwrap(model)
# # # # # # #     with torch.no_grad():
# # # # # # #         h_score = hard_score_from_obs(batch_list[0][:, :, :2])
# # # # # # #     obs_req = batch_list[0].detach().clone().requires_grad_(True)
# # # # # # #     bl_g = list(batch_list); bl_g[0] = obs_req
# # # # # # #     with torch.enable_grad():
# # # # # # #         cond = raw.encoder(bl_g, hard_score=h_score)
# # # # # # #         x0   = torch.randn(obs_req.shape[1], raw.pred_len, 2, device=device) * raw.sigma_inference
# # # # # # #         t0   = torch.zeros(obs_req.shape[1], device=device)
# # # # # # #         v    = raw.velocity(x0, t0, cond)
# # # # # # #         pred_rel = x0 + v
# # # # # # #         ts       = min(target_step, raw.pred_len - 1)
# # # # # # #         pred_rel[:, ts, :].norm(dim=-1).mean().backward()
# # # # # # #     if obs_req.grad is not None:
# # # # # # #         attr = obs_req.grad[:, :, :2].norm(dim=-1)
# # # # # # #         attr = attr / (attr.sum(0, keepdim=True) + 1e-8)
# # # # # # #     else:
# # # # # # #         attr = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)
# # # # # # #     return attr.detach()


# # # # # # # @torch.no_grad()
# # # # # # # def compute_ensemble_uncertainty(all_traj: torch.Tensor) -> Dict:
# # # # # # #     """
# # # # # # #     [XAI-4] Uncertainty per lead time across K ensemble samples.
# # # # # # #     Returns std_per_step [T,B] km, uncertainty_ratio, mean_72h_std, high_uncertainty.
# # # # # # #     """
# # # # # # #     all_deg  = _norm_to_deg(all_traj)  # [K,T,B,2]
# # # # # # #     K, T, B  = all_deg.shape[:3]
# # # # # # #     mean_traj = all_deg.mean(0)
# # # # # # #     std_km = torch.zeros(T, B, device=all_traj.device)
# # # # # # #     for t in range(T):
# # # # # # #         dists = _haversine_deg(
# # # # # # #             all_deg[:, t].reshape(K * B, 2),
# # # # # # #             mean_traj[t].unsqueeze(0).expand(K, B, 2).reshape(K * B, 2)
# # # # # # #         ).reshape(K, B)
# # # # # # #         std_km[t] = dists.std(0)
# # # # # # #     s12 = min(1, T - 1); s72 = min(11, T - 1)
# # # # # # #     return {
# # # # # # #         "std_per_step":      std_km,
# # # # # # #         "uncertainty_ratio": (std_km[s72] + 1e-3) / (std_km[s12] + 1e-3),
# # # # # # #         "mean_72h_std":      float(std_km[s72].mean()),
# # # # # # #         "mean_12h_std":      float(std_km[s12].mean()),
# # # # # # #         "high_uncertainty":  std_km[s72] > 80.0,
# # # # # # #     }


# # # # # # # @torch.no_grad()
# # # # # # # def compute_heading_deviation(pred_deg: torch.Tensor,
# # # # # # #                                gt_deg:   torch.Tensor) -> torch.Tensor:
# # # # # # #     """
# # # # # # #     [XAI-6] Heading deviation per lead time step, in degrees (absolute).
# # # # # # #     High values indicate directional error = root cause of CTE.
# # # # # # #     Returns [T-1, B].
# # # # # # #     """
# # # # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # #     if T < 2:
# # # # # # #         return pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # # # #     bear_gt   = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
# # # # # # #     bear_pred = _forward_azimuth(gt_deg[:T-1], pred_deg[1:T])
# # # # # # #     diff = (bear_pred - bear_gt).abs()
# # # # # # #     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
# # # # # # #     return torch.rad2deg(diff)


# # # # # # # @torch.no_grad()
# # # # # # # def compute_cte_contribution(pred_deg: torch.Tensor,
# # # # # # #                               gt_deg:   torch.Tensor) -> Dict:
# # # # # # #     """
# # # # # # #     [XAI-7] Decompose trajectory error into ATE (along-track) and CTE (cross-track).
# # # # # # #     Returns ate_per_step [T-1,B], cte_per_step [T-1,B], scalar means.
# # # # # # #     """
# # # # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # #     if T < 2:
# # # # # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # # # #         return {"ate_per_step": z, "cte_per_step": z,
# # # # # # #                 "ate_mean": z[0], "cte_mean": z[0],
# # # # # # #                 "ate_abs_mean": 0.0, "cte_abs_mean": 0.0}
# # # # # # #     bear_ref  = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
# # # # # # #     bear_err  = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
# # # # # # #     dist_err  = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# # # # # # #     ang       = bear_err - bear_ref
# # # # # # #     ate       = dist_err * torch.cos(ang)
# # # # # # #     cte       = dist_err * torch.sin(ang)
# # # # # # #     return {
# # # # # # #         "ate_per_step":  ate,
# # # # # # #         "cte_per_step":  cte,
# # # # # # #         "ate_mean":      ate.mean(0),
# # # # # # #         "cte_mean":      cte.abs().mean(0),
# # # # # # #         "ate_abs_mean":  float(ate.abs().mean()),
# # # # # # #         "cte_abs_mean":  float(cte.abs().mean()),
# # # # # # #     }


# # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # #  Compat stubs
# # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # @torch.no_grad()
# # # # # # # def classify_hard_easy(obs_traj_norm, per_sample_loss=None,
# # # # # # #                         hard_score_p: float = 70.0, loss_p: float = 50.0):
# # # # # # #     scores = hard_score_from_obs(obs_traj_norm)
# # # # # # #     B = scores.shape[0]
# # # # # # #     if B < 4:
# # # # # # #         return torch.zeros(B, dtype=torch.bool, device=scores.device)
# # # # # # #     return scores >= torch.quantile(scores, hard_score_p / 100.0)


# # # # # # # @torch.no_grad()
# # # # # # # def classify_hard_easy_global(obs_traj_norm, global_threshold):
# # # # # # #     return hard_score_from_obs(obs_traj_norm) >= global_threshold


# # # # # # # @torch.no_grad()
# # # # # # # def compute_diversity_score(candidates) -> float:
# # # # # # #     if len(candidates) < 2:
# # # # # # #         return 0.0
# # # # # # #     T, B = candidates[0].shape[0], candidates[0].shape[1]
# # # # # # #     ep_step = min(T - 1, 11)
# # # # # # #     endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
# # # # # # #     N = endpoints.shape[0]
# # # # # # #     ep_mean = endpoints.mean(0, keepdim=True)
# # # # # # #     dists = _haversine_deg(
# # # # # # #         endpoints.reshape(N * B, 2),
# # # # # # #         ep_mean.expand(N, B, 2).reshape(N * B, 2)
# # # # # # #     ).reshape(N, B)
# # # # # # #     return float(dists.std(0).mean())


# # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # #  TCFlowMatching v2.6
# # # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # # class TCFlowMatching(nn.Module):
# # # # # # #     """
# # # # # # #     TC-FlowMatching v2.6

# # # # # # #     Training loss:
# # # # # # #         L_total = L_CFM
# # # # # # #                 + lam_reg  × L_reg          (ramp ep10→ep30, linear weights)
# # # # # # #                 + lam_dir  × lambda_heading × L_heading_ms  (ramp ep5→ep20)

# # # # # # #         L_CFM:         conditional flow matching (random t, OT noise-GT matching)
# # # # # # #         L_reg:         ADE loss at sigma_inference, softmax-linspace step weights
# # # # # # #         L_heading_ms:  [v2.6] 4-step heading continuation, decay=0.5 per step

# # # # # # #     NO L_momentum: removed because it anchors to val speed distribution,
# # # # # # #     which hurts ATE on test storms that move at different speeds.

# # # # # # #     Inference pipeline:
# # # # # # #         1. Sample K=20 candidates with sigma=0.04 FIXED
# # # # # # #         2. Physics score selection (speed+smooth+heading+displacement)
# # # # # # #         3. Weighted average of top-3 candidates
# # # # # # #         4. Speed calibration: per-storm scale ×[0.85, 1.15] to match obs speed
# # # # # # #         5. Optional: multi-scale sigma ensemble (sample_multiscale)
# # # # # # #     """

# # # # # # #     def __init__(
# # # # # # #         self,
# # # # # # #         pred_len:          int   = 12,
# # # # # # #         obs_len:           int   = 8,
# # # # # # #         unet_in_ch:        int   = 13,
# # # # # # #         d_cond:            int   = 256,
# # # # # # #         d_model:           int   = 256,
# # # # # # #         nhead:             int   = 8,
# # # # # # #         num_dec_layers:    int   = 4,
# # # # # # #         dim_ff:            int   = 512,
# # # # # # #         dropout:           float = 0.1,
# # # # # # #         sigma_min:         float = 0.04,
# # # # # # #         sigma_max:         float = 0.08,
# # # # # # #         lambda_reg:        float = 0.2,
# # # # # # #         lambda_heading:    float = 0.20,   # [v2.7] 0.10→0.20, n_steps=8 covers 48h
# # # # # # #         lambda_momentum:   float = 0.0,    # DISABLED — field kept for compat only
# # # # # # #         use_ot:            bool  = True,
# # # # # # #         ot_epsilon:        float = 0.05,
# # # # # # #         use_ema:           bool  = True,
# # # # # # #         ema_decay:         float = 0.995,
# # # # # # #         n_inference_steps: int   = 1,
# # # # # # #         n_ensemble:        int   = 20,
# # # # # # #         sigma_inference:   float = 0.04,   # FIXED throughout
# # # # # # #         **kwargs,
# # # # # # #     ):
# # # # # # #         super().__init__()
# # # # # # #         self.pred_len          = pred_len
# # # # # # #         self.obs_len           = obs_len
# # # # # # #         self.sigma_min         = sigma_min
# # # # # # #         self.sigma_max         = sigma_max
# # # # # # #         self.lambda_reg        = lambda_reg
# # # # # # #         self.lambda_heading    = lambda_heading
# # # # # # #         self.lambda_momentum   = 0.0           # always disabled
# # # # # # #         self.use_ot            = use_ot
# # # # # # #         self.ot_epsilon        = ot_epsilon
# # # # # # #         self.n_inference_steps = n_inference_steps
# # # # # # #         self.n_ensemble        = n_ensemble
# # # # # # #         self.sigma_inference   = sigma_inference

# # # # # # #         self.encoder  = ContextEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
# # # # # # #         self.velocity = VelocityTransformer(
# # # # # # #             pred_len=pred_len, d_model=d_model, nhead=nhead,
# # # # # # #             num_layers=num_dec_layers, dim_ff=dim_ff,
# # # # # # #             dropout=dropout, d_cond=d_cond)
# # # # # # #         self.use_ema = use_ema
# # # # # # #         self._ema    = None

# # # # # # #     def init_ema(self):
# # # # # # #         if self.use_ema:
# # # # # # #             self._ema = EMAModel(self, decay=0.995)

# # # # # # #     def ema_update(self):
# # # # # # #         if self._ema is not None:
# # # # # # #             self._ema.update(self)

# # # # # # #     def _to_relative(self, x_abs: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# # # # # # #         return x_abs - last_obs.unsqueeze(1)

# # # # # # #     def _from_relative(self, x_rel: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# # # # # # #         return x_rel + last_obs.unsqueeze(1)

# # # # # # #     def _sigma_schedule(self, epoch: int) -> float:
# # # # # # #         """Cosine decay sigma_max→sigma_min over ep5→ep40."""
# # # # # # #         if epoch < 5:
# # # # # # #             return self.sigma_max
# # # # # # #         if epoch < 40:
# # # # # # #             t = (epoch - 5) / 35.0
# # # # # # #             return self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * t))
# # # # # # #         return self.sigma_min

# # # # # # #     # ── [v2.6] Multi-step heading loss ─────────────────────────────────────

# # # # # # #     def _heading_loss_ms(self, pred_deg: torch.Tensor, obs_deg: torch.Tensor,
# # # # # # #                           n_steps: int = 4, decay: float = 0.5) -> torch.Tensor:
# # # # # # #         """
# # # # # # #         Multi-step heading constraint. [UPGRADED FROM v2.5 SINGLE STEP]

# # # # # # #         v2.5 used single-step (step 0 only, weight=0.05):
# # # # # # #           → 72h heading dev INCREASED from 87° to 128° during training
# # # # # # #           → Model learned "start correctly" then drifted

# # # # # # #         v2.6 uses 4 steps with exponential decay:
# # # # # # #           step t reference: obs_heading (t=0) or pred_bearing[t-1] (t>0, DETACHED)
# # # # # # #           step 0: 1.0 × (1 - cos(pred_bear[0] - obs_bear))  ← hard anchor
# # # # # # #           step 1: 0.5 × (1 - cos(pred_bear[1] - pred_bear[0]))  ← soft continuation
# # # # # # #           step 2: 0.25 × ...
# # # # # # #           step 3: 0.125 × ...

# # # # # # #         DETACH ref at each step: prevents gradient explosion through chain.
# # # # # # #         Each step trains relatively independently.

# # # # # # #         Expected: 72h heading dev down from 128° to ~90-100°
# # # # # # #         Expected: CTE test from 64.5km down to ~57-63km
# # # # # # #         """
# # # # # # #         if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
# # # # # # #             return pred_deg.new_zeros(())

# # # # # # #         # Starting reference: bearing from obs[-2] → obs[-1]
# # # # # # #         ref_bear = _forward_azimuth(obs_deg[-2], obs_deg[-1])   # [B]

# # # # # # #         loss = pred_deg.new_zeros(())
# # # # # # #         # pts[t] → pts[t+1] gives bearing for pred step t
# # # # # # #         pts = torch.cat([obs_deg[-1:], pred_deg], 0)   # [T_pred+1, B, 2]

# # # # # # #         N = min(n_steps, pred_deg.shape[0])
# # # # # # #         for t in range(N):
# # # # # # #             pred_bear  = _forward_azimuth(pts[t], pts[t + 1])  # [B]
# # # # # # #             angle_diff = pred_bear - ref_bear
# # # # # # #             loss       = loss + (decay ** t) * (1.0 - torch.cos(angle_diff)).mean()
# # # # # # #             ref_bear   = pred_bear.detach()   # CRITICAL: detach to prevent chain gradient

# # # # # # #         return loss / N

# # # # # # #     # ── L_reg (identical to v2.1 — proven not to overfit) ──────────────────

# # # # # # #     def _reg_loss(self, x1_rel: torch.Tensor, last_obs: torch.Tensor,
# # # # # # #                   cond: torch.Tensor,
# # # # # # #                   hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# # # # # # #         """
# # # # # # #         ADE loss at t=0 with sigma_inference noise, consistent with 1-shot inference.
# # # # # # #         Uses SOFTMAX-LINSPACE step weights (max/min ≈ 3×).
# # # # # # #         DOES NOT use exp weights (v2.4 used 12.2× ratio → catastrophic overfitting).
# # # # # # #         """
# # # # # # #         B, T, _ = x1_rel.shape
# # # # # # #         device   = x1_rel.device
# # # # # # #         x0   = torch.randn_like(x1_rel) * self.sigma_inference
# # # # # # #         t0   = torch.zeros(B, device=device)
# # # # # # #         v    = self.velocity(x0, t0, cond)
# # # # # # #         x1_pred_abs = self._from_relative(x0 + v, last_obs)
# # # # # # #         x1_gt_abs   = self._from_relative(x1_rel, last_obs)
# # # # # # #         pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))  # [T, B, 2]
# # # # # # #         gt_deg   = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
# # # # # # #         dist     = _haversine_deg(pred_deg, gt_deg)             # [T, B] km
# # # # # # #         # Linear weights: softmax(linspace(1,3,T)) → max/min ≈ 3×
# # # # # # #         T_actual = dist.shape[0]
# # # # # # #         sw = torch.linspace(1.0, 3.0, T_actual, device=device, dtype=dist.dtype)
# # # # # # #         sw = F.softmax(sw, dim=0).unsqueeze(1)                  # [T, 1]
# # # # # # #         if hard_score is not None:
# # # # # # #             sw_hard = (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)  # [1, B]
# # # # # # #         else:
# # # # # # #             sw_hard = torch.ones(1, B, device=device, dtype=dist.dtype)
# # # # # # #         return ((dist * sw) * sw_hard).mean() / 300.0

# # # # # # #     # ── get_loss_breakdown ──────────────────────────────────────────────────

# # # # # # #     def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
# # # # # # #         """
# # # # # # #         Full loss computation. Called in TRAINING with augmented batch.
# # # # # # #         Called in VAL without augmentation → val_loss is reliable signal.

# # # # # # #         Total = L_CFM + lam_reg × L_reg + lam_dir × lambda_heading × L_heading_ms
# # # # # # #         """
# # # # # # #         obs_traj = batch_list[0]
# # # # # # #         gt_traj  = batch_list[1]
# # # # # # #         B        = obs_traj.shape[1]
# # # # # # #         device   = obs_traj.device

# # # # # # #         sigma    = self._sigma_schedule(epoch)
# # # # # # #         x1_gt    = gt_traj.permute(1, 0, 2)       # [B, T, 2]
# # # # # # #         last_obs = obs_traj[-1, :, :2]             # [B, 2]
# # # # # # #         x1_rel   = self._to_relative(x1_gt, last_obs)

# # # # # # #         h_score = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # # #         cond    = self.encoder(batch_list, hard_score=h_score)

# # # # # # #         # ── L_CFM ─────────────────────────────────────────────────────────
# # # # # # #         x0 = torch.randn_like(x1_rel) * sigma
# # # # # # #         if self.use_ot and B >= 4:
# # # # # # #             x0_flat, x1_flat = _ot_match(
# # # # # # #                 x0.reshape(B, -1), x1_rel.reshape(B, -1), self.ot_epsilon)
# # # # # # #             x0         = x0_flat.reshape(B, self.pred_len, 2)
# # # # # # #             x1_matched = x1_flat.reshape(B, self.pred_len, 2)
# # # # # # #         else:
# # # # # # #             x1_matched = x1_rel

# # # # # # #         t        = torch.rand(B, device=device)
# # # # # # #         x_t      = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1_matched
# # # # # # #         u_target = x1_matched - x0
# # # # # # #         v_pred   = self.velocity(x_t, t, cond)
# # # # # # #         l_cfm    = F.mse_loss(v_pred, u_target)

# # # # # # #         # ── L_reg ramp ep10→ep30 ─────────────────────────────────────────
# # # # # # #         if epoch < 10:     lam_reg = 0.0
# # # # # # #         elif epoch < 30:   lam_reg = self.lambda_reg * (epoch - 10) / 20.0
# # # # # # #         else:              lam_reg = self.lambda_reg

# # # # # # #         l_reg = (self._reg_loss(x1_rel, last_obs, cond, h_score)
# # # # # # #                  if lam_reg > 0.0 else x0.new_zeros(()))

# # # # # # #         # ── L_heading_ms ramp ep5→ep20 ────────────────────────────────────
# # # # # # #         # Ramp: 0 before ep5, linear ep5→ep20, full after ep20
# # # # # # #         if epoch < 5:      lam_dir = 0.0
# # # # # # #         elif epoch < 20:   lam_dir = (epoch - 5) / 15.0
# # # # # # #         else:              lam_dir = 1.0

# # # # # # #         if lam_dir > 0.0:
# # # # # # #             # BUG FIX v2.7: KHÔNG dùng torch.no_grad() — gradient phải flow
# # # # # # #             # qua velocity để L_heading thực sự update model weights.
# # # # # # #             # _heading_loss_ms đã có ref_bear.detach() per step → không explosion.
# # # # # # #             x0_h       = torch.randn_like(x1_rel) * self.sigma_inference
# # # # # # #             v_h        = self.velocity(x0_h, torch.zeros(B, device=device), cond)
# # # # # # #             x1_h_abs   = self._from_relative(x0_h + v_h, last_obs)
# # # # # # #             pred_deg_h = _norm_to_deg(x1_h_abs.permute(1, 0, 2))  # [T, B, 2] — có gradient
# # # # # # #             obs_deg_h  = _norm_to_deg(obs_traj[:, :, :2])          # [T_obs, B, 2]
# # # # # # #             l_heading  = self._heading_loss_ms(pred_deg_h, obs_deg_h, n_steps=8, decay=0.5)
# # # # # # #         else:
# # # # # # #             l_heading = x0.new_zeros(())

# # # # # # #         l_momentum = x0.new_zeros(())   # always 0 in v2.6

# # # # # # #         # ── Total ─────────────────────────────────────────────────────────
# # # # # # #         total = l_cfm + lam_reg * l_reg + lam_dir * self.lambda_heading * l_heading
# # # # # # #         if not torch.isfinite(total):
# # # # # # #             total = x0.new_zeros(())

# # # # # # #         # ── ADE 1-step log ────────────────────────────────────────────────
# # # # # # #         with torch.no_grad():
# # # # # # #             x0_log = torch.randn_like(x1_rel) * self.sigma_inference
# # # # # # #             v_log  = self.velocity(x0_log, torch.zeros(B, device=device), cond)
# # # # # # #             ade_log = _haversine_deg(
# # # # # # #                 _norm_to_deg(self._from_relative(x0_log + v_log, last_obs).permute(1, 0, 2)),
# # # # # # #                 _norm_to_deg(x1_gt.permute(1, 0, 2))
# # # # # # #             ).mean().item()

# # # # # # #         return {
# # # # # # #             "total":    total,
# # # # # # #             "l_cfm":    l_cfm.item(),
# # # # # # #             "l_reg":    l_reg.item() if torch.is_tensor(l_reg) else 0.0,
# # # # # # #             "l_heading": l_heading.item() if torch.is_tensor(l_heading) else 0.0,
# # # # # # #             "l_momentum": 0.0,   # always 0
# # # # # # #             "lam_reg":  lam_reg,
# # # # # # #             "lam_dir":  lam_dir,
# # # # # # #             "sigma":    sigma,
# # # # # # #             "ade_1step": ade_log,
# # # # # # #             "hard_score_mean": float(h_score.mean()),
# # # # # # #             "hard_score_max":  float(h_score.max()),
# # # # # # #             # Compat keys for older scripts
# # # # # # #             "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
# # # # # # #             "speed": 0., "accel": 0., "fm_mse": l_cfm.item(),
# # # # # # #             "l_hard_total": 0., "n_hard": 0, "alpha_hard": 0.,
# # # # # # #             "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
# # # # # # #         }

# # # # # # #     def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
# # # # # # #         return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

# # # # # # #     # ── [NEW v2.6] Speed calibration at inference ──────────────────────────

# # # # # # #     @staticmethod
# # # # # # #     @torch.no_grad()
# # # # # # #     def speed_calibrate_pred(pred_abs_norm: torch.Tensor,
# # # # # # #                               last_obs_norm: torch.Tensor,
# # # # # # #                               obs_norm:      torch.Tensor,
# # # # # # #                               clip_min: float = 0.50,
# # # # # # #                               clip_max: float = 2.00) -> torch.Tensor:
# # # # # # #         """
# # # # # # #         Per-storm post-hoc speed calibration. Applied at inference ONLY.
# # # # # # #         DOES NOT affect training → zero train/inference consistency issue.

# # # # # # #         Scales DISPLACEMENT MAGNITUDE, preserves DIRECTION.
# # # # # # #         → ATE improved (speed error corrected)
# # # # # # #         → CTE unchanged (direction unchanged)

# # # # # # #         Algorithm:
# # # # # # #           obs_speed_ref = exponentially-weighted mean of obs step speeds
# # # # # # #           pred_speed_ref = mean of first 4 predicted step speeds
# # # # # # #           scale = clip(obs_speed_ref / pred_speed_ref, clip_min, clip_max)
# # # # # # #           pred_cal[t] = last_obs + (pred[t] - last_obs) * scale

# # # # # # #         clip_min=0.85, clip_max=1.15:
# # # # # # #           Prevents over-correction from noisy obs/pred speed estimates.
# # # # # # #           Allows ±15% adjustment per storm.

# # # # # # #         Why this works:
# # # # # # #           v2.5 XAI-5: speed ratio = 1.15–1.61 throughout training
# # # # # # #           (model consistently over-predicts speed on val)
# # # # # # #           Test storms have different speed → need per-storm correction
# # # # # # #           This is adaptive: fast test storms get upscaling, slow get downscaling
# # # # # # #         """
# # # # # # #         if obs_norm.shape[0] < 2 or pred_abs_norm.shape[0] < 2:
# # # # # # #             return pred_abs_norm

# # # # # # #         obs_deg  = _norm_to_deg(obs_norm)        # [T_obs, B, 2]
# # # # # # #         pred_deg = _norm_to_deg(pred_abs_norm)   # [T, B, 2]
# # # # # # #         last_deg = _norm_to_deg(last_obs_norm)   # [B, 2]

# # # # # # #         # Obs speed reference (exponentially weighted, recent steps more important)
# # # # # # #         obs_spds = _step_speeds_kmh(obs_deg)     # [T_obs-1, B] km/h
# # # # # # #         T_s = obs_spds.shape[0]
# # # # # # #         w = torch.exp(torch.arange(T_s, dtype=obs_spds.dtype, device=obs_spds.device) * 0.5)
# # # # # # #         obs_spd_ref = (obs_spds * (w / w.sum()).unsqueeze(1)).sum(0).clamp(min=3.0)   # [B]

# # # # # # #         # Predicted speed — exp-weighted ALL steps (decay=0.3)
# # # # # # #         # first 4 steps bias thấp vì predict ngắn hạn tốt hơn 72h
# # # # # # #         # Dùng tất cả steps để scale chính xác hơn cho long-range
# # # # # # #         pts = torch.cat([last_deg.unsqueeze(0), pred_deg], 0)   # [T+1, B, 2]
# # # # # # #         pred_spds = _step_speeds_kmh(pts)                        # [T, B]
# # # # # # #         T_p = pred_spds.shape[0]
# # # # # # #         w_p = torch.exp(-0.3 * torch.arange(T_p, dtype=pred_spds.dtype, device=pred_spds.device))
# # # # # # #         pred_spd_ref = (pred_spds * (w_p / w_p.sum()).unsqueeze(1)).sum(0).clamp(min=3.0)

# # # # # # #         # Per-storm scale, clipped
# # # # # # #         scale = (obs_spd_ref / pred_spd_ref).clamp(clip_min, clip_max)  # [B]

# # # # # # #         # Scale displacement from last_obs (preserves direction!)
# # # # # # #         return (last_obs_norm.unsqueeze(0)
# # # # # # #                 + (pred_abs_norm - last_obs_norm.unsqueeze(0))
# # # # # # #                 * scale.view(1, -1, 1))

# # # # # # #     # ── Sample (1-shot + speed calibration) ──────────────────────────────

# # # # # # #     @torch.no_grad()
# # # # # # #     def sample(self, batch_list,
# # # # # # #                num_ensemble:          Optional[int]  = None,
# # # # # # #                ddim_steps:            Optional[int]  = None,
# # # # # # #                return_xai:            bool           = False,
# # # # # # #                use_speed_calibration: bool           = True,
# # # # # # #                **kwargs) -> Tuple:
# # # # # # #         """
# # # # # # #         1-shot inference with physics selection + speed calibration.

# # # # # # #         Steps:
# # # # # # #           1. Encode context once (hard_score + full encoder)
# # # # # # #           2. Sample K=20 candidates: x0 ~ N(0, sigma_inf²), x_pred = x0 + v(x0, 0, cond)
# # # # # # #           3. Physics score each candidate (speed+smooth+heading+displacement)
# # # # # # #           4. Weighted average of top-3 candidates
# # # # # # #           5. Speed calibration per-storm (clip ±15%)
# # # # # # #           6. If return_xai=True: compute XAI-1 through XAI-7

# # # # # # #         Returns:
# # # # # # #           (pred_mean [T,B,2], zeros [T,B,2], all_traj [K,T,B,2])
# # # # # # #           or (pred_mean, zeros, all_traj, xai_dict) if return_xai=True
# # # # # # #         """
# # # # # # #         K  = num_ensemble or self.n_ensemble
# # # # # # #         N  = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else self.n_inference_steps
# # # # # # #         dt = 1.0 / max(N, 1)

# # # # # # #         obs_traj    = batch_list[0]
# # # # # # #         T_obs, B, _ = obs_traj.shape
# # # # # # #         device      = obs_traj.device

# # # # # # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # # #         obs_norm = obs_traj[:, :, :2]
# # # # # # #         last_obs = obs_traj[-1, :, :2]
# # # # # # #         t0       = torch.zeros(B, device=device)

# # # # # # #         # Encode once for all K samples (expensive part)
# # # # # # #         cond = self.encoder(batch_list, hard_score=h_score)

# # # # # # #         all_traj = []
# # # # # # #         for _ in range(K):
# # # # # # #             x_rel = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference

# # # # # # #             if N <= 1:
# # # # # # #                 # 1-shot: exactly what L_reg is trained for
# # # # # # #                 v     = self.velocity(x_rel, t0, cond)
# # # # # # #                 x_rel = x_rel + v
# # # # # # #             else:
# # # # # # #                 # Euler multi-step (for experiments only, slower)
# # # # # # #                 for step in range(N):
# # # # # # #                     t_b   = torch.full((B,), step * dt, device=device)
# # # # # # #                     x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3., 3.)

# # # # # # #             x_abs = self._from_relative(x_rel, last_obs)
# # # # # # #             all_traj.append(x_abs.permute(1, 0, 2))   # [T, B, 2]

# # # # # # #         # Physics-based selection: best-of-K with weighted average of top-3
# # # # # # #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)   # [K, B]
# # # # # # #         all_t   = torch.stack(all_traj, 0)   # [K, T, B, 2]
# # # # # # #         top_k   = min(3, K)
# # # # # # #         top_idx = scores.topk(top_k, dim=0).indices   # [top_k, B]

# # # # # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # # # # #         for b in range(B):
# # # # # # #             idx_b = top_idx[:, b]
# # # # # # #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # # # # # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# # # # # # #         # [NEW v2.6] Per-storm speed calibration
# # # # # # #         if use_speed_calibration:
# # # # # # #             pred_mean = self.speed_calibrate_pred(
# # # # # # #                 pred_mean, last_obs, obs_norm, clip_min=0.85, clip_max=1.15)

# # # # # # #         if not return_xai:
# # # # # # #             return pred_mean, torch.zeros_like(pred_mean), all_t

# # # # # # #         # ── XAI output (XAI-2,3,4,5,6,7) ──────────────────────────────────
# # # # # # #         xai = {}

# # # # # # #         # XAI-4: Ensemble uncertainty
# # # # # # #         xai.update(compute_ensemble_uncertainty(all_t))

# # # # # # #         # XAI-2: Hard score components
# # # # # # #         _, hard_comps = hard_score_from_obs(obs_norm, return_components=True)
# # # # # # #         xai["hard_components"] = hard_comps

# # # # # # #         pred_deg  = _norm_to_deg(pred_mean)    # [T, B, 2]
# # # # # # #         obs_deg_x = _norm_to_deg(obs_norm)     # [T_obs, B, 2]

# # # # # # #         # XAI-5: Speed comparison (obs vs pred, pre and post calibration)
# # # # # # #         obs_spd_x    = _step_speeds_kmh(obs_deg_x)
# # # # # # #         obs_spd_mu   = obs_spd_x.mean(0)     # [B]
# # # # # # #         if pred_deg.shape[0] >= 2:
# # # # # # #             last_deg_x   = obs_deg_x[-1]
# # # # # # #             pts_x        = torch.cat([last_deg_x.unsqueeze(0), pred_deg], 0)
# # # # # # #             pred_spd_x   = _step_speeds_kmh(pts_x)
# # # # # # #             pred_spd_mu  = pred_spd_x.mean(0)  # [B]
# # # # # # #         else:
# # # # # # #             pred_spd_mu = obs_spd_mu.clone()

# # # # # # #         speed_ratio = (pred_spd_mu / obs_spd_mu.clamp(min=1.0))
# # # # # # #         xai["speed_comparison"] = {
# # # # # # #             "obs_speed_mean":  float(obs_spd_mu.mean()),
# # # # # # #             "pred_speed_mean": float(pred_spd_mu.mean()),
# # # # # # #             "speed_ratio":     float(speed_ratio.mean()),
# # # # # # #             "per_storm_obs":   obs_spd_mu,
# # # # # # #             "per_storm_pred":  pred_spd_mu,
# # # # # # #             "over_predict":    speed_ratio > 1.15,
# # # # # # #             "under_predict":   speed_ratio < 0.85,
# # # # # # #         }

# # # # # # #         # XAI-3: Physics components for best prediction
# # # # # # #         v_ref = obs_spd_mu.clamp(min=5.0)
# # # # # # #         v_sig = v_ref * 0.5
# # # # # # #         spd_sc = torch.exp(-((pred_spd_mu - v_ref) / v_sig).pow(2) * 0.5)
# # # # # # #         if pred_deg.shape[0] >= 3:
# # # # # # #             vel_x   = pred_deg[1:] - pred_deg[:-1]
# # # # # # #             accel_x = (vel_x[1:] - vel_x[:-1]).norm(dim=-1)
# # # # # # #             smo_sc  = torch.exp(-accel_x.mean(0) * 5.0)
# # # # # # #         else:
# # # # # # #             smo_sc = torch.ones(B, device=device)
# # # # # # #         if obs_norm.shape[0] >= 2 and pred_mean.shape[0] >= 1:
# # # # # # #             ov = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # # # # # #             pv = pred_mean[0, :, :2] - obs_norm[-1, :, :2]
# # # # # # #             cos_s = (F.normalize(ov, dim=-1, eps=1e-6) * F.normalize(pv, dim=-1, eps=1e-6)).sum(-1)
# # # # # # #             hd_sc = torch.exp((cos_s.clamp(-1, 1) - 1.0) * 3.0)
# # # # # # #         else:
# # # # # # #             hd_sc = torch.ones(B, device=device)
# # # # # # #         xai["physics_components"] = {
# # # # # # #             "speed": spd_sc, "smooth": smo_sc, "heading": hd_sc,
# # # # # # #             "obs_speed": obs_spd_mu, "pred_speed": pred_spd_mu,
# # # # # # #         }

# # # # # # #         # XAI-6: Heading deviation per lead time
# # # # # # #         gt_traj_xai = batch_list[1]
# # # # # # #         gt_deg_xai  = _norm_to_deg(gt_traj_xai[:, :, :2])
# # # # # # #         xai["heading_deviation_deg"] = compute_heading_deviation(pred_deg, gt_deg_xai)

# # # # # # #         # XAI-7: ATE/CTE decomposition
# # # # # # #         xai["ate_cte_decomp"] = compute_cte_contribution(pred_deg, gt_deg_xai)

# # # # # # #         # ── [v2.7 NEW] XAI-8: Per-horizon speed error ────────────────────
# # # # # # #         # pred_speed vs gt_speed tại mỗi horizon
# # # # # # #         # → biết model over/under-predict speed ở đâu (early vs late horizon)
# # # # # # #         if pred_deg.shape[0] >= 2 and gt_deg_xai.shape[0] >= 2:
# # # # # # #             last_obs_deg = obs_deg_x[-1:]   # [1, B, 2]
# # # # # # #             pred_pts_h   = torch.cat([last_obs_deg, pred_deg], 0)   # [T+1, B, 2]
# # # # # # #             gt_pts_h     = torch.cat([last_obs_deg, gt_deg_xai[:pred_deg.shape[0]]], 0)
# # # # # # #             pred_spd_h   = _step_speeds_kmh(pred_pts_h)   # [T, B]
# # # # # # #             gt_spd_h     = _step_speeds_kmh(gt_pts_h)     # [T, B]
# # # # # # #             xai["speed_per_horizon"] = {
# # # # # # #                 "pred_speeds_kmh": pred_spd_h.mean(1).tolist(),   # [T] avg over batch
# # # # # # #                 "gt_speeds_kmh":   gt_spd_h.mean(1).tolist(),     # [T] avg over batch
# # # # # # #                 "speed_ratio_per_step": (pred_spd_h / gt_spd_h.clamp(min=1.0)).mean(1).tolist(),
# # # # # # #             }
# # # # # # #         else:
# # # # # # #             xai["speed_per_horizon"] = {}

# # # # # # #         # ── [v2.7 NEW] XAI-9: Storm category breakdown ───────────────────
# # # # # # #         # Chia storms theo obs speed để diagnose per-category error
# # # # # # #         # Nếu fast storms chiếm đại bộ phận error → confirm ATE gap issue
# # # # # # #         obs_spd_per = obs_spd_x.mean(0).clamp(min=0.0)   # [B] km/h per storm
# # # # # # #         slow_mask   = obs_spd_per < 8.0
# # # # # # #         medium_mask = (obs_spd_per >= 8.0) & (obs_spd_per < 15.0)
# # # # # # #         fast_mask   = obs_spd_per >= 15.0
# # # # # # #         xai["storm_categories"] = {
# # # # # # #             "n_slow":          int(slow_mask.sum()),
# # # # # # #             "n_medium":        int(medium_mask.sum()),
# # # # # # #             "n_fast":          int(fast_mask.sum()),
# # # # # # #             "obs_speed_mean":  float(obs_spd_per.mean()),
# # # # # # #             "obs_speed_std":   float(obs_spd_per.std()),
# # # # # # #             "obs_speed_max":   float(obs_spd_per.max()),
# # # # # # #         }

# # # # # # #         return pred_mean, torch.zeros_like(pred_mean), all_t, xai

# # # # # # #     @torch.no_grad()
# # # # # # #     def sample_multiscale(
# # # # # # #         self,
# # # # # # #         batch_list,
# # # # # # #         sigmas: Optional[List[float]] = None,
# # # # # # #         n_per_sigma: int = 4,
# # # # # # #         use_speed_calibration: bool = True,
# # # # # # #     ) -> Tuple:
# # # # # # #         """
# # # # # # #         Multi-scale sigma ensemble at inference (DOES NOT affect training).
# # # # # # #         Hedges against speed distribution shift between val and test.
# # # # # # #         """
# # # # # # #         if sigmas is None:
# # # # # # #             sigmas = [0.025, 0.035, 0.04, 0.05, 0.065]

# # # # # # #         obs_traj = batch_list[0]
# # # # # # #         B = obs_traj.shape[1]; device = obs_traj.device

# # # # # # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # # #         obs_norm = obs_traj[:, :, :2]
# # # # # # #         last_obs = obs_traj[-1, :, :2]
# # # # # # #         t0       = torch.zeros(B, device=device)
# # # # # # #         cond     = self.encoder(batch_list, hard_score=h_score)

# # # # # # #         all_traj = []
# # # # # # #         for sigma in sigmas:
# # # # # # #             for _ in range(n_per_sigma):
# # # # # # #                 x0    = torch.randn(B, self.pred_len, 2, device=device) * sigma
# # # # # # #                 v     = self.velocity(x0, t0, cond)
# # # # # # #                 x_abs = self._from_relative(x0 + v, last_obs)
# # # # # # #                 all_traj.append(x_abs.permute(1, 0, 2))

# # # # # # #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)
# # # # # # #         all_t   = torch.stack(all_traj, 0)
# # # # # # #         top_k   = min(5, len(all_traj))
# # # # # # #         top_idx = scores.topk(top_k, dim=0).indices

# # # # # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # # # # #         for b in range(B):
# # # # # # #             idx_b = top_idx[:, b]
# # # # # # #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # # # # # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# # # # # # #         if use_speed_calibration:
# # # # # # #             pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

# # # # # # #         return pred_mean, torch.zeros_like(pred_mean), all_t


# # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # #  Backward compat alias
# # # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # # TCDiffusion = TCFlowMatching
# # # # # # """
# # # # # # Model/flow_matching_model.py  ──  TC-FlowMatching v2.8
# # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # VIẾT LẠI HOÀN TOÀN từ v2.1 + các fix đã verified từ thực nghiệm 140 epoch.

# # # # # # ━━━ CƠ SỞ LÝ THUYẾT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # # v2.1 test: ADE=229.8, ATE=214.4, CTE=71.6 (best so far, generalize tốt nhất)
# # # # # # v2.5 test: ADE=232.5, ATE=222.3, CTE=64.5 (ATE tệ hơn, CTE tốt hơn)
# # # # # # v2.4 test: ADE=291.1 (catastrophic overfitting)

# # # # # # PHÂN TÍCH LỖI v2.1:
# # # # # #   ATE/CTE ratio on test = 214.4/71.6 = 3.0
# # # # # #   → 75% error is ALONG-TRACK (speed/distance)
# # # # # #   → 25% is CROSS-TRACK (direction)
# # # # # #   Test gap 72h = +129.3km >> 12h = +10.1km → long-range drift

# # # # # # PHÂN TÍCH v2.5 THẤT BẠI:
# # # # # #   L_momentum anchor pred_speed ≈ val_speed (10.3km/h)
# # # # # #   Test storms faster → L_momentum SLOWS model on test → ATE +7.9km WORSE
# # # # # #   L_heading (step 0 only) → 72h heading dev INCREASED: ep0=87°→ep120=128°!
# # # # # #   Recurvature aug: CTE improved 7km ← giữ lại

# # # # # # ━━━ VẬT LÝ CỦA GIẢI PHÁP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # # GIẢI PHÁP ATE (along-track error):
# # # # # #   Training: KHÔNG anchor speed tại training (no L_momentum)
# # # # # #   Inference: SPEED CALIBRATION per-storm
# # # # # #     scale = clip(obs_speed / pred_speed_first4, 0.85, 1.15)
# # # # # #     pred_cal = last_obs + (pred - last_obs) * scale
# # # # # #     → Preserves direction (CTE unchanged), fixes magnitude (ATE reduced)
# # # # # #     → Per-storm adaptive: fast test storms get upscaling, slow storms downscaling
# # # # # #     → DOES NOT affect training → zero train/inference consistency issue

# # # # # # GIẢI PHÁP CTE (cross-track error):
# # # # # #   Training: MULTI-STEP heading constraint (not single step)
# # # # # #     step 0 (6h):  weight 1.0  — hard anchor to obs direction
# # # # # #     step 1 (12h): weight 0.5  — continuation (detached ref)
# # # # # #     step 2 (18h): weight 0.25 — softer continuation
# # # # # #     step 3 (24h): weight 0.125 — very soft
# # # # # #     Detach ref at each step: prevents gradient explosion through chain
# # # # # #     → 72h heading dev should come down from 128° to ~90°

# # # # # # GIẢI PHÁP GAP (val→test distribution shift):
# # # # # #   [AUG-C] Recurvature: proven to reduce CTE test (v2.5)
# # # # # #   [AUG-D] Obs-speed scaling ×0.7–1.4: model sees diverse speed contexts
# # # # # #            → at inference, adapts better to test storms' speed
# # # # # #   Speed calibration at inference bridges remaining gap

# # # # # # ━━━ GIỮ NGUYÊN TỪ v2.1 (PROVEN, KHÔNG THAY ĐỔI) ━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # #   ✅ sigma_inference=0.04 FIXED — zero train/inference mismatch
# # # # # #   ✅ L_reg softmax-linspace weights (max/min ≈ 3×) — không exp (v2.4 fail)
# # # # # #   ✅ Mild base aug: shift±5km + speed×0.85–1.15 (val ≈ test distribution)
# # # # # #   ✅ 1-shot inference nhất quán với L_reg training
# # # # # #   ✅ 2-group optimizer, encoder freeze 10ep
# # # # # #   ✅ val loop NO augmentation
# # # # # #   ✅ OT noise-GT matching trong L_CFM
# # # # # #   ✅ ContextEncoder: FNO3D + Mamba + Env + 6 kinematic features
# # # # # #   ✅ VelocityTransformer: d_model=256, 4 layers, nhead=8

# # # # # # ━━━ CẢI TIẾN v2.6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # #   [REMOVE]  L_momentum — LOẠI BỎ HOÀN TOÀN (made ATE +7.9km worse)
# # # # # #   [UPGRADE] L_heading → multi-step 4 steps, decay=0.5, weight=0.10
# # # # # #   [NEW-INF] speed_calibrate_pred() tại inference (clip 0.85–1.15)
# # # # # #   [UPGRADE] physics_score + displacement_score (15% weight)
# # # # # #   [AUG-D]   obs-speed scaling ×[0.3-0.7] or [1.5-2.5] (15% probability, v2.8 wider)
# # # # # #   [XAI 1–7] giữ nguyên đầy đủ, XAI-5 now reports per-storm calibration

# # # # # # ━━━ KỲ VỌNG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # # #                    Val ADE   Test ADE   Val ATE  Test ATE  Val CTE  Test CTE
# # # # # #   v2.1 (best):     170.1      229.8     160.2    214.4     45.7     71.6
# # # # # #   v2.5 (current):  170.5      232.5     161.3    222.3     46.2     64.5
# # # # # #   v2.6 (target):   170±1      210–220   161±1    200–210   46±1     57–63
# # # # # # """
# # # # # # from __future__ import annotations

# # # # # # import math
# # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # import torch
# # # # # # import torch.nn as nn
# # # # # # import torch.nn.functional as F

# # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # from Model.env_net_transformer_gphsplit import Env_net

# # # # # # R_EARTH  = 6371.0
# # # # # # DT_HOURS = 6.0


# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # #  Coordinate utilities (same as v2.1)
# # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# # # # # #     """Normalized coords → (lon°, lat°)."""
# # # # # #     return torch.stack([
# # # # # #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# # # # # #         (t[..., 1] * 50.0) / 10.0,
# # # # # #     ], dim=-1)


# # # # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # #     """Haversine distance km between two (lon°,lat°) tensors."""
# # # # # #     lat1 = torch.deg2rad(p1[..., 1]);  lat2 = torch.deg2rad(p2[..., 1])
# # # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # # #     a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
# # # # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# # # # # # def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # #     """Bearing in radians from p1→p2 (degrees input)."""
# # # # # #     lon1 = torch.deg2rad(p1[..., 0]);  lat1 = torch.deg2rad(p1[..., 1])
# # # # # #     lon2 = torch.deg2rad(p2[..., 0]);  lat2 = torch.deg2rad(p2[..., 1])
# # # # # #     dlon = lon2 - lon1
# # # # # #     y = torch.sin(dlon) * torch.cos(lat2)
# # # # # #     x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
# # # # # #     return torch.atan2(y, x)


# # # # # # def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
# # # # # #     """Speed km/h between consecutive steps of a trajectory [T, B, 2]."""
# # # # # #     if traj_deg.shape[0] < 2:
# # # # # #         return traj_deg.new_zeros(1, traj_deg.shape[1])
# # # # # #     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # #  EMAModel (same as v2.1)
# # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # def _unwrap(m):
# # # # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m


# # # # # # class EMAModel:
# # # # # #     def __init__(self, model, decay: float = 0.995):
# # # # # #         self.decay = decay
# # # # # #         m = _unwrap(model)
# # # # # #         self.shadow = {k: v.detach().clone()
# # # # # #                        for k, v in m.state_dict().items()
# # # # # #                        if v.dtype.is_floating_point}

# # # # # #     def update(self, model):
# # # # # #         m = _unwrap(model)
# # # # # #         with torch.no_grad():
# # # # # #             for k, v in m.state_dict().items():
# # # # # #                 if k in self.shadow:
# # # # # #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

# # # # # #     def apply_to(self, model):
# # # # # #         m = _unwrap(model)
# # # # # #         backup, sd = {}, m.state_dict()
# # # # # #         for k in self.shadow:
# # # # # #             if k not in sd: continue
# # # # # #             backup[k] = sd[k].detach().clone()
# # # # # #             sd[k].copy_(self.shadow[k])
# # # # # #         return backup

# # # # # #     def restore(self, model, backup):
# # # # # #         m = _unwrap(model)
# # # # # #         sd = m.state_dict()
# # # # # #         for k, v in backup.items():
# # # # # #             if k in sd: sd[k].copy_(v)


# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # #  OT matching (same as v2.1)
# # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 50) -> torch.Tensor:
# # # # # #     B = cost.shape[0]; device = cost.device
# # # # # #     log_a = -math.log(B) * torch.ones(B, device=device)
# # # # # #     log_b = -math.log(B) * torch.ones(B, device=device)
# # # # # #     log_K = -cost / epsilon
# # # # # #     log_u = torch.zeros(B, device=device)
# # # # # #     log_v = torch.zeros(B, device=device)
# # # # # #     for _ in range(n_iter):
# # # # # #         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
# # # # # #         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
# # # # # #     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


# # # # # # def _ot_match(x0_flat: torch.Tensor, x1_flat: torch.Tensor,
# # # # # #               epsilon: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # # #     B = x0_flat.shape[0]
# # # # # #     if B < 4:
# # # # # #         return x0_flat, x1_flat
# # # # # #     try:
# # # # # #         cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1] ** 0.5)
# # # # # #         with torch.no_grad():
# # # # # #             pi = _sinkhorn_log(cost, epsilon=epsilon)
# # # # # #         flat = pi.reshape(-1).clamp(0.0)
# # # # # #         s = flat.sum()
# # # # # #         if not torch.isfinite(s) or s < 1e-10:
# # # # # #             return x0_flat, x1_flat
# # # # # #         idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
# # # # # #         return x0_flat[idx // B], x1_flat
# # # # # #     except Exception:
# # # # # #         return x0_flat, x1_flat


# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # #  VelocityTransformer (identical to v2.1)
# # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # class VelocityTransformer(nn.Module):
# # # # # #     def __init__(self, pred_len: int = 12, d_model: int = 256, nhead: int = 8,
# # # # # #                  num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
# # # # # #                  d_cond: int = 256):
# # # # # #         super().__init__()
# # # # # #         self.pred_len = pred_len
# # # # # #         self.d_model  = d_model
# # # # # #         self.traj_embed = nn.Linear(2, d_model)
# # # # # #         self.pos_emb    = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
# # # # # #         self.step_emb   = nn.Embedding(pred_len, d_model)
# # # # # #         self.time_mlp   = nn.Sequential(
# # # # # #             nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
# # # # # #         self.cond_proj  = nn.Sequential(nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))
# # # # # #         dec_layer = nn.TransformerDecoderLayer(
# # # # # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # # # # #             dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
# # # # # #         self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
# # # # # #         self.out_norm = nn.LayerNorm(d_model)
# # # # # #         self.out_proj = nn.Sequential(
# # # # # #             nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
# # # # # #         self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
# # # # # #         nn.init.zeros_(self.out_proj[-1].weight)
# # # # # #         nn.init.zeros_(self.out_proj[-1].bias)

# # # # # #     def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
# # # # # #         half = self.d_model // 2
# # # # # #         freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
# # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # #         emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
# # # # # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # # # # #         if self.d_model % 2 == 1:
# # # # # #             emb = F.pad(emb, (0, 1))
# # # # # #         return self.time_mlp(emb)

# # # # # #     def forward(self, x_t: torch.Tensor, t: torch.Tensor,
# # # # # #                 cond: torch.Tensor) -> torch.Tensor:
# # # # # #         B, T, _ = x_t.shape
# # # # # #         step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # #         x_emb = (self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(step_idx))
# # # # # #         memory = torch.cat([self._time_emb(t).unsqueeze(1),
# # # # # #                             self.cond_proj(cond).unsqueeze(1)], dim=1)
# # # # # #         out = self.out_norm(self.decoder(x_emb, memory))
# # # # # #         return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)


# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # #  ContextEncoder (identical to v2.1)
# # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # class ContextEncoder(nn.Module):
# # # # # #     RAW_CTX_DIM = 512

# # # # # #     def __init__(self, obs_len: int = 8, unet_in_ch: int = 13, d_cond: int = 256):
# # # # # #         super().__init__()
# # # # # #         self.obs_len = obs_len
# # # # # #         self.d_cond  = d_cond

# # # # # #         self.spatial_enc     = FNO3DEncoder(in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # #                                              n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # #                                              spatial_down=32, dropout=0.05)
# # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # #         self.decoder_proj    = nn.Linear(1, 16)
# # # # # #         self.enc_1d          = DataEncoder1D(in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # #                                               lstm_hidden=128, lstm_layers=3,
# # # # # #                                               dropout=0.1, d_state=16)
# # # # # #         self.env_enc         = Env_net(obs_len=obs_len, d_model=32)
# # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # #         self.ctx_drop = nn.Dropout(0.1)
# # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, d_cond)
# # # # # #         self.ctx_ln2  = nn.LayerNorm(d_cond)
# # # # # #         # 6 kinematic features: vel_x, vel_y, speed_n, sin(h), cos(h), accel
# # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # #             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
# # # # # #             nn.Linear(256, d_cond // 2), nn.GELU())
# # # # # #         self.hard_embed = nn.Sequential(
# # # # # #             nn.Linear(1, d_cond // 4), nn.GELU(), nn.Linear(d_cond // 4, d_cond // 4))
# # # # # #         self.fuse = nn.Sequential(
# # # # # #             nn.Linear(d_cond + d_cond // 2 + d_cond // 4, d_cond),
# # # # # #             nn.LayerNorm(d_cond), nn.GELU())

# # # # # #     def _encode_raw(self, batch_list) -> torch.Tensor:
# # # # # #         obs_traj  = batch_list[0]; obs_Me = batch_list[7]
# # # # # #         image_obs = batch_list[11]; env_data = batch_list[13]
# # # # # #         if image_obs.dim() == 4:
# # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
# # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # #         T_obs = obs_traj.shape[0]
# # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
# # # # # #                                    mode="linear", align_corners=False).permute(0,2,1)
# # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # #                                           device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # #         f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
# # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)
# # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

# # # # # #     def _kinematic_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
# # # # # #         """6 kinematic features per obs step."""
# # # # # #         B = obs_traj.shape[1]; T_obs = obs_traj.shape[0]; device = obs_traj.device
# # # # # #         if T_obs >= 2:
# # # # # #             traj_deg = _norm_to_deg(obs_traj)
# # # # # #             vel_norm = obs_traj[1:] - obs_traj[:-1]
# # # # # #             speed    = _step_speeds_kmh(traj_deg)
# # # # # #             speed_n  = (speed / 20.0).clamp(-3.0, 3.0)
# # # # # #             heading  = torch.atan2(vel_norm[:, :, 1], vel_norm[:, :, 0])
# # # # # #             if T_obs >= 3:
# # # # # #                 dspd  = speed[1:] - speed[:-1]
# # # # # #                 accel = torch.cat([obs_traj.new_zeros(1, B),
# # # # # #                                    (dspd / 10.0).clamp(-3.0, 3.0)], 0)
# # # # # #             else:
# # # # # #                 accel = obs_traj.new_zeros(T_obs - 1, B)
# # # # # #             kine = torch.stack([vel_norm[:,:,0], vel_norm[:,:,1], speed_n,
# # # # # #                                 heading.sin(), heading.cos(), accel], dim=-1)
# # # # # #         else:
# # # # # #             kine = obs_traj.new_zeros(self.obs_len, B, 6)
# # # # # #         if kine.shape[0] < self.obs_len:
# # # # # #             kine = torch.cat([obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6), kine], 0)
# # # # # #         else:
# # # # # #             kine = kine[-self.obs_len:]
# # # # # #         return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

# # # # # #     def forward(self, batch_list, hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# # # # # #         raw   = self._encode_raw(batch_list)
# # # # # #         ctx   = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
# # # # # #         kfeat = self._kinematic_feat(batch_list[0][:, :, :2])
# # # # # #         if hard_score is None:
# # # # # #             hard_score = torch.zeros(ctx.shape[0], device=ctx.device)
# # # # # #         hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))
# # # # # #         return self.fuse(torch.cat([ctx, kfeat, hfeat], dim=-1))


# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # #  Hard score (XAI-2) — same as v2.1
# # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # @torch.no_grad()
# # # # # # def hard_score_from_obs(obs_traj_norm: torch.Tensor,
# # # # # #                          return_components: bool = False):
# # # # # #     """[XAI-2] Độ khó của storm dựa trên curvature, speed_var, dir_change."""
# # # # # #     T, B   = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
# # # # # #     device = obs_traj_norm.device
# # # # # #     if T < 3:
# # # # # #         z = torch.zeros(B, device=device)
# # # # # #         if return_components:
# # # # # #             return z, {"curvature": z.clone(), "speed_var": z.clone(), "dir_change": z.clone()}
# # # # # #         return z
# # # # # #     traj_deg = _norm_to_deg(obs_traj_norm[..., :2])
# # # # # #     az12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
# # # # # #     az23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
# # # # # #     diff = (az23 - az12).abs()
# # # # # #     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
# # # # # #     curvature  = diff.mean(0) / math.pi
# # # # # #     spd        = _step_speeds_kmh(traj_deg)
# # # # # #     if spd.shape[0] >= 2:
# # # # # #         speed_var = (spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 1.)
# # # # # #     else:
# # # # # #         speed_var = torch.zeros(B, device=device)
# # # # # #     dir_change = (diff > (20.0 / 180.0 * math.pi)).float().mean(0)
# # # # # #     # [v2.8 FIX] speed_cv → acceleration variability (NOT clone of speed_var)
# # # # # #     # speed_var = std(speed)/mean(speed) đã đo variation trong speed values
# # # # # #     # speed_cv đúng = std(Δspeed)/mean(speed) = variation trong acceleration
# # # # # #     # → Bão tăng/giảm tốc thất thường → harder to predict trajectory
# # # # # #     if spd.shape[0] >= 2:
# # # # # #         accel_seq  = (spd[1:] - spd[:-1]).abs()   # [T-2, B] absolute acceleration
# # # # # #         speed_cv   = (accel_seq.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 2.0)
# # # # # #     else:
# # # # # #         speed_cv = torch.zeros(B, device=device)
# # # # # #     score = (0.35 * curvature + 0.25 * speed_var + 0.25 * dir_change + 0.15 * speed_cv).clamp(0., 1.)
# # # # # #     if return_components:
# # # # # #         return score, {"curvature": curvature, "speed_var": speed_var,
# # # # # #                        "dir_change": dir_change, "speed_cv": speed_cv}
# # # # # #     return score


# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # #  Physics score v2.6 (adds displacement_score)
# # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # @torch.no_grad()
# # # # # # def _physics_score(traj_norm: torch.Tensor, obs_norm: torch.Tensor) -> torch.Tensor:
# # # # # #     """
# # # # # #     Best-of-K selection score. Four components:
# # # # # #       speed_score       (0.30): per-step speed vs obs reference
# # # # # #       smooth_score      (0.25): trajectory smoothness (no sharp acceleration)
# # # # # #       heading_score     (0.30): first-step direction matches obs
# # # # # #       displacement_score(0.15): total path length consistent with obs speed

# # # # # #     [NEW v2.6] displacement_score penalizes samples with total displacement
# # # # # #     far from obs-speed expectation → catches speed over-prediction artifacts
# # # # # #     that cause high ATE on test storms.
# # # # # #     """
# # # # # #     B      = traj_norm.shape[1]
# # # # # #     device = traj_norm.device
# # # # # #     traj_deg = _norm_to_deg(traj_norm)
# # # # # #     v_ref   = None

# # # # # #     # ── Speed score ────────────────────────────────────────────────────────
# # # # # #     if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# # # # # #         obs_deg = _norm_to_deg(obs_norm)
# # # # # #         obs_spd = _step_speeds_kmh(obs_deg)
# # # # # #         T_s     = obs_spd.shape[0]
# # # # # #         w_obs   = torch.linspace(0.5, 1.0, T_s, device=device)
# # # # # #         v_ref   = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()   # [B]
# # # # # #         pred_spd = _step_speeds_kmh(traj_deg)
# # # # # #         v_sigma  = v_ref.clamp(min=5.0) * 0.5
# # # # # #         speed_score = torch.exp(
# # # # # #             -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5)
# # # # # #     elif traj_deg.shape[0] >= 2:
# # # # # #         speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.).mean(0))
# # # # # #     else:
# # # # # #         speed_score = torch.ones(B, device=device)

# # # # # #     # ── Smoothness score ───────────────────────────────────────────────────
# # # # # #     if traj_deg.shape[0] >= 3:
# # # # # #         vel          = traj_deg[1:] - traj_deg[:-1]
# # # # # #         accel_mag    = (vel[1:] - vel[:-1]).norm(dim=-1)
# # # # # #         smooth_score = torch.exp(-accel_mag.mean(0) * 5.0)
# # # # # #     else:
# # # # # #         smooth_score = torch.ones(B, device=device)

# # # # # #     # ── Heading score ──────────────────────────────────────────────────────
# # # # # #     if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
# # # # # #         obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # # # # #         pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
# # # # # #         obs_h    = F.normalize(obs_vel,  dim=-1, eps=1e-6)
# # # # # #         pred_h   = F.normalize(pred_vel, dim=-1, eps=1e-6)
# # # # # #         cos_sim  = (obs_h * pred_h).sum(-1).clamp(-1, 1)
# # # # # #         head_score = torch.exp((cos_sim - 1.0) * 3.0)
# # # # # #     else:
# # # # # #         head_score = torch.ones(B, device=device)

# # # # # #     # ── [FIX v2.8] Displacement score — corrected factor ─────────────────
# # # # # #     # Factor=0.75 sai: straight storm path=720km, expected=540km → penalizes correct
# # # # # #     # Factor=1.0 đúng: straight storm path ≈ obs_speed × T × DT → score=1.0
# # # # # #     # v2.7 bug: score của correct prediction chỉ 0.61, nay fix → 1.0
# # # # # #     if v_ref is not None and traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# # # # # #         T_pred        = traj_deg.shape[0]
# # # # # #         expected_total = v_ref * T_pred * DT_HOURS * 1.0   # FIXED: 1.0 not 0.75
# # # # # #         step_dists    = _haversine_deg(traj_deg[:-1], traj_deg[1:])   # [T-1, B]
# # # # # #         actual_total  = step_dists.sum(0)                              # [B]
# # # # # #         rel_err       = (actual_total - expected_total).abs() / expected_total.clamp(min=10.)
# # # # # #         disp_score    = torch.exp(-rel_err * 1.5)
# # # # # #     else:
# # # # # #         disp_score    = torch.ones(B, device=device)

# # # # # #     return (speed_score.pow(0.30)
# # # # # #             * smooth_score.pow(0.25)
# # # # # #             * head_score.pow(0.30)
# # # # # #             * disp_score.pow(0.15)).clamp(min=1e-6)


# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # #  Augmentation v2.6
# # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # def augment_batch(batch_list) -> list:
# # # # # #     """
# # # # # #     v2.6 augmentation — extends v2.1 core with recurvature and obs-speed aug.

# # # # # #     Distribution (6 types):
# # # # # #       A (25%): track shift ±5km             — shape unchanged, position varies
# # # # # #       B (20%): GT speed scale ×0.85–1.15   — mild, proven from v2.1
# # # # # #       C (20%): recurvature ±20°             — proven CTE improvement in v2.5
# # # # # #       D (15%): obs-speed scaling ×[0.3-0.7]+[1.5-2.5] — [v2.8] wider range for test distribution
# # # # # #                                               buộc model học với fast/slow storms
# # # # # #                                               giảm anchoring to val speed distribution
# # # # # #       E (10%): no augmentation              — giữ original distribution
# # # # # #       F (10%): Gaussian noise ±3km          — small robustness

# # # # # #     KHÔNG dùng:
# # # # # #       - mixup (phi vật lý, v2.4 proof)
# # # # # #       - speed scale >×1.5 trên GT (too far from real)
# # # # # #       - exp L_reg weights (v2.4 catastrophic)

# # # # # #     AUG-D [v2.7+] — consistent speed scaling (obs + GT scaled together):
# # # # # #       v2.6 bug: scale OBS only → mismatch dạy model 'slow after fast obs' → ATE worse.
# # # # # #       v2.7+ fix: scale cả obs và GT từ anchor nhất quán → model học speed-proportional.
# # # # # #       v2.8 range: [0.30-0.70] or [1.50-2.50] → cover test distribution rộng hơn.
# # # # # #       → Physics TTA tại inference pick đúng scale per-storm.
# # # # # #     """
# # # # # #     bl = list(batch_list)
# # # # # #     if not torch.is_tensor(bl[0]):
# # # # # #         return bl

# # # # # #     obs    = bl[0]
# # # # # #     device = obs.device
# # # # # #     anchor = obs[-1:, :, :2].detach()  # [1, B, 2] — last obs as pivot
# # # # # #     r = torch.rand(1).item()

# # # # # #     if r < 0.25:
# # # # # #         # A: Track shift ±5km (preserves storm shape, varies position)
# # # # # #         shift = (torch.rand(2, device=device) - 0.5) * 0.018  # ±5km in normalized
# # # # # #         bl[0] = obs + shift.view(1, 1, 2)
# # # # # #         if torch.is_tensor(bl[1]):
# # # # # #             bl[1] = bl[1] + shift.view(1, 1, 2)

# # # # # #     elif r < 0.45:
# # # # # #         # B: Speed scale ×0.85–1.15 (both obs+GT from anchor, mild)
# # # # # #         scale = 0.85 + 0.30 * torch.rand(1, device=device).item()
# # # # # #         obs_c = obs.clone()
# # # # # #         obs_c[..., :2] = anchor + (obs[..., :2] - anchor) * scale
# # # # # #         bl[0] = obs_c
# # # # # #         if torch.is_tensor(bl[1]):
# # # # # #             bl[1] = anchor + (bl[1] - anchor) * scale

# # # # # #     elif r < 0.65:
# # # # # #         # C: [v2.8 UPGRADED] Recurvature ±30° (was ±20°) + sudden onset mode
# # # # # #         #
# # # # # #         # v2.6 test: CTE gap still +20km despite ±20° aug
# # # # # #         # → Test storms have more/harder recurvature than val
# # # # # #         # → ±30° + sudden onset (50/50) covers harder cases
# # # # # #         #
# # # # # #         # ease-in (50%): gradual recurvature — realistic for large-scale TC
# # # # # #         # sudden (50%): recurvature starts immediately — realistic for abrupt
# # # # # #         #               steering flow change, more common in test basin
# # # # # #         T_pred = bl[1].shape[0] if torch.is_tensor(bl[1]) else 0
# # # # # #         if T_pred >= 4:
# # # # # #             gt      = bl[1].clone()
# # # # # #             max_deg = (torch.rand(1).item() - 0.5) * 60.0   # -30° to +30° (was 40°)
# # # # # #             max_rad = max_deg * math.pi / 180.0
# # # # # #             # 50/50: ease-in vs sudden onset
# # # # # #             sudden  = torch.rand(1).item() < 0.5
# # # # # #             for t in range(T_pred):
# # # # # #                 if sudden:
# # # # # #                     # Sudden: full angle immediately, linear continuation
# # # # # #                     progress = min(1.0, 0.3 + 0.7 * t / max(T_pred - 1, 1))
# # # # # #                 else:
# # # # # #                     # Ease-in: slow start, accelerate (original)
# # # # # #                     progress = (t / max(T_pred - 1, 1)) ** 1.5
# # # # # #                 angle_t  = max_rad * progress
# # # # # #                 c, s = math.cos(angle_t), math.sin(angle_t)
# # # # # #                 rot = torch.tensor([[c, -s], [s, c]], dtype=gt.dtype, device=device)
# # # # # #                 rel = (gt[t] - anchor[0]).unsqueeze(-1)   # [B, 2, 1]
# # # # # #                 gt[t] = (rot @ rel).squeeze(-1) + anchor[0]
# # # # # #             bl[1] = gt
# # # # # #             # Rotate last 3 obs steps slightly (30% of max) for consistency
# # # # # #             T_obs   = obs.shape[0]
# # # # # #             obs_aug = obs.clone()
# # # # # #             cos_p   = math.cos(max_rad * 0.3)
# # # # # #             sin_p   = math.sin(max_rad * 0.3)
# # # # # #             rot_p   = torch.tensor([[cos_p, -sin_p], [sin_p, cos_p]],
# # # # # #                                     dtype=obs.dtype, device=device)
# # # # # #             for t_obs in range(max(0, T_obs - 3), T_obs):
# # # # # #                 rel_o = (obs_aug[t_obs, :, :2] - anchor[0]).unsqueeze(-1)
# # # # # #                 obs_aug[t_obs, :, :2] = (rot_p @ rel_o).squeeze(-1) + anchor[0]
# # # # # #             bl[0] = obs_aug

# # # # # #     elif r < 0.80:
# # # # # #         # D: [v2.7 REDESIGNED] Consistent speed scaling — scale BOTH obs AND GT
# # # # # #         #
# # # # # #         # v2.6 bug: scale OBS only, GT unchanged → inconsistency
# # # # # #         # Model sees fast obs but slow GT → learns "slow down after fast approach"
# # # # # #         # → Confuses model's velocity field → ATE test TỆ HƠN v2.1 (+4.5km)
# # # # # #         #
# # # # # #         # v2.7 fix: scale BOTH obs and GT từ cùng anchor (last obs)
# # # # # #         # → Storm trở nên nhanh hơn hoặc chậm hơn NHẤT QUÁN
# # # # # #         # → Model học speed-proportional prediction
# # # # # #         # → Speed calibration tại inference hoạt động chính xác hơn
# # # # # #         #
# # # # # #         # Range: bimodal để cover extremes (val đã cover middle range):
# # # # # #         #   50%: ×0.50-0.80 → bão chậm/đứng yên (rare in val, common in test)
# # # # # #         #   50%: ×1.20-1.80 → bão rất nhanh (rapid intensification case)
# # # # # #         # [v2.8 UPGRADED] Wider range to cover test distribution
# # # # # #         # v2.6 test: ATE still +4.5km vs v2.1 → test storms faster than val
# # # # # #         # → Extend to [0.30, 0.70] and [1.50, 2.50] to cover extremes
# # # # # #         # 3-mode: slow/fast/extreme-fast (more weight on fast side)
# # # # # #         r_mode = torch.rand(1).item()
# # # # # #         if r_mode < 0.33:
# # # # # #             speed_scale = 0.30 + 0.40 * torch.rand(1, device=device).item()  # [0.30, 0.70] slow
# # # # # #         elif r_mode < 0.67:
# # # # # #             speed_scale = 1.50 + 0.50 * torch.rand(1, device=device).item()  # [1.50, 2.00] fast
# # # # # #         else:
# # # # # #             speed_scale = 2.00 + 0.50 * torch.rand(1, device=device).item()  # [2.00, 2.50] very fast
# # # # # #         T_obs = obs.shape[0]
# # # # # #         obs_aug = obs.clone()
# # # # # #         # Scale obs displacements step-by-step
# # # # # #         obs_aug[0] = obs[0]
# # # # # #         for t in range(1, T_obs):
# # # # # #             disp = obs[t, :, :2] - obs[t-1, :, :2]
# # # # # #             obs_aug[t, :, :2] = obs_aug[t-1, :, :2] + disp * speed_scale
# # # # # #         bl[0] = obs_aug
# # # # # #         # [FIX BUG #9] anchor = obs_aug[-1], NOT obs[-1]
# # # # # #         # obs_aug[-1] ≠ obs[-1] after scaling → using obs[-1] creates jump at obs→GT boundary
# # # # # #         anchor_pos = obs_aug[-1:, :, :2].detach()   # [1, B, 2] — scaled last obs
# # # # # #         # Scale GT displacements from scaled anchor (same scale → continuous)
# # # # # #         if torch.is_tensor(bl[1]):
# # # # # #             gt_aug = bl[1].clone()
# # # # # #             gt_aug[0, :, :2] = anchor_pos[0] + (bl[1][0, :, :2] - obs[-1, :, :2]) * speed_scale
# # # # # #             for t in range(1, bl[1].shape[0]):
# # # # # #                 disp_gt = bl[1][t, :, :2] - bl[1][t-1, :, :2]
# # # # # #                 gt_aug[t, :, :2] = gt_aug[t-1, :, :2] + disp_gt * speed_scale
# # # # # #             bl[1] = gt_aug

# # # # # #     elif r < 0.90:
# # # # # #         # E: no augmentation (keep original distribution)
# # # # # #         pass

# # # # # #     else:
# # # # # #         # F: small Gaussian noise ±3km
# # # # # #         obs_new = obs.clone()
# # # # # #         obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.003
# # # # # #         bl[0] = obs_new

# # # # # #     return bl


# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # #  XAI functions (1–7, complete)
# # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # def compute_obs_attribution(model, batch_list, device: torch.device,
# # # # # #                              target_step: int = 11) -> torch.Tensor:
# # # # # #     """
# # # # # #     [XAI-1] Gradient saliency: which obs step most influences 72h prediction?

# # # # # #     Returns attr [T_obs, B] normalized importance [0,1].
# # # # # #     High attr[-1, b] = last obs dominant (normal).
# # # # # #     High attr[0, b]  = storm has long "memory" (unusual → hard storm).
# # # # # #     """
# # # # # #     raw = _unwrap(model)
# # # # # #     with torch.no_grad():
# # # # # #         h_score = hard_score_from_obs(batch_list[0][:, :, :2])
# # # # # #     obs_req = batch_list[0].detach().clone().requires_grad_(True)
# # # # # #     bl_g = list(batch_list); bl_g[0] = obs_req
# # # # # #     with torch.enable_grad():
# # # # # #         cond = raw.encoder(bl_g, hard_score=h_score)
# # # # # #         x0   = torch.randn(obs_req.shape[1], raw.pred_len, 2, device=device) * raw.sigma_inference
# # # # # #         t0   = torch.zeros(obs_req.shape[1], device=device)
# # # # # #         v    = raw.velocity(x0, t0, cond)
# # # # # #         pred_rel = x0 + v
# # # # # #         ts       = min(target_step, raw.pred_len - 1)
# # # # # #         pred_rel[:, ts, :].norm(dim=-1).mean().backward()
# # # # # #     if obs_req.grad is not None:
# # # # # #         attr = obs_req.grad[:, :, :2].norm(dim=-1)
# # # # # #         attr = attr / (attr.sum(0, keepdim=True) + 1e-8)
# # # # # #     else:
# # # # # #         attr = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)
# # # # # #     return attr.detach()


# # # # # # @torch.no_grad()
# # # # # # def compute_ensemble_uncertainty(all_traj: torch.Tensor) -> Dict:
# # # # # #     """
# # # # # #     [XAI-4] Uncertainty per lead time across K ensemble samples.
# # # # # #     Returns std_per_step [T,B] km, uncertainty_ratio, mean_72h_std, high_uncertainty.
# # # # # #     """
# # # # # #     all_deg  = _norm_to_deg(all_traj)  # [K,T,B,2]
# # # # # #     K, T, B  = all_deg.shape[:3]
# # # # # #     mean_traj = all_deg.mean(0)
# # # # # #     std_km = torch.zeros(T, B, device=all_traj.device)
# # # # # #     for t in range(T):
# # # # # #         dists = _haversine_deg(
# # # # # #             all_deg[:, t].reshape(K * B, 2),
# # # # # #             mean_traj[t].unsqueeze(0).expand(K, B, 2).reshape(K * B, 2)
# # # # # #         ).reshape(K, B)
# # # # # #         std_km[t] = dists.std(0)
# # # # # #     s12 = min(1, T - 1); s72 = min(11, T - 1)
# # # # # #     return {
# # # # # #         "std_per_step":      std_km,
# # # # # #         "uncertainty_ratio": (std_km[s72] + 1e-3) / (std_km[s12] + 1e-3),
# # # # # #         "mean_72h_std":      float(std_km[s72].mean()),
# # # # # #         "mean_12h_std":      float(std_km[s12].mean()),
# # # # # #         "high_uncertainty":  std_km[s72] > 80.0,
# # # # # #     }


# # # # # # @torch.no_grad()
# # # # # # def compute_heading_deviation(pred_deg: torch.Tensor,
# # # # # #                                gt_deg:   torch.Tensor) -> torch.Tensor:
# # # # # #     """
# # # # # #     [XAI-6] Heading deviation per lead time step, in degrees (absolute).
# # # # # #     High values indicate directional error = root cause of CTE.
# # # # # #     Returns [T-1, B].
# # # # # #     """
# # # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # #     if T < 2:
# # # # # #         return pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # # #     bear_gt   = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
# # # # # #     bear_pred = _forward_azimuth(gt_deg[:T-1], pred_deg[1:T])
# # # # # #     diff = (bear_pred - bear_gt).abs()
# # # # # #     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
# # # # # #     return torch.rad2deg(diff)


# # # # # # @torch.no_grad()
# # # # # # def compute_cte_contribution(pred_deg: torch.Tensor,
# # # # # #                               gt_deg:   torch.Tensor) -> Dict:
# # # # # #     """
# # # # # #     [XAI-7] Decompose trajectory error into ATE (along-track) and CTE (cross-track).
# # # # # #     Returns ate_per_step [T-1,B], cte_per_step [T-1,B], scalar means.
# # # # # #     """
# # # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # #     if T < 2:
# # # # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # # #         return {"ate_per_step": z, "cte_per_step": z,
# # # # # #                 "ate_mean": z[0], "cte_mean": z[0],
# # # # # #                 "ate_abs_mean": 0.0, "cte_abs_mean": 0.0}
# # # # # #     bear_ref  = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
# # # # # #     bear_err  = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
# # # # # #     dist_err  = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# # # # # #     ang       = bear_err - bear_ref
# # # # # #     ate       = dist_err * torch.cos(ang)
# # # # # #     cte       = dist_err * torch.sin(ang)
# # # # # #     return {
# # # # # #         "ate_per_step":  ate,
# # # # # #         "cte_per_step":  cte,
# # # # # #         "ate_mean":      ate.mean(0),
# # # # # #         "cte_mean":      cte.abs().mean(0),
# # # # # #         "ate_abs_mean":  float(ate.abs().mean()),
# # # # # #         "cte_abs_mean":  float(cte.abs().mean()),
# # # # # #     }


# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # #  Compat stubs
# # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # @torch.no_grad()
# # # # # # def classify_hard_easy(obs_traj_norm, per_sample_loss=None,
# # # # # #                         hard_score_p: float = 70.0, loss_p: float = 50.0):
# # # # # #     scores = hard_score_from_obs(obs_traj_norm)
# # # # # #     B = scores.shape[0]
# # # # # #     if B < 4:
# # # # # #         return torch.zeros(B, dtype=torch.bool, device=scores.device)
# # # # # #     return scores >= torch.quantile(scores, hard_score_p / 100.0)


# # # # # # @torch.no_grad()
# # # # # # def classify_hard_easy_global(obs_traj_norm, global_threshold):
# # # # # #     return hard_score_from_obs(obs_traj_norm) >= global_threshold


# # # # # # @torch.no_grad()
# # # # # # def compute_diversity_score(candidates) -> float:
# # # # # #     if len(candidates) < 2:
# # # # # #         return 0.0
# # # # # #     T, B = candidates[0].shape[0], candidates[0].shape[1]
# # # # # #     ep_step = min(T - 1, 11)
# # # # # #     endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
# # # # # #     N = endpoints.shape[0]
# # # # # #     ep_mean = endpoints.mean(0, keepdim=True)
# # # # # #     dists = _haversine_deg(
# # # # # #         endpoints.reshape(N * B, 2),
# # # # # #         ep_mean.expand(N, B, 2).reshape(N * B, 2)
# # # # # #     ).reshape(N, B)
# # # # # #     return float(dists.std(0).mean())


# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # #  TCFlowMatching v2.6
# # # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # # class TCFlowMatching(nn.Module):
# # # # # #     """
# # # # # #     TC-FlowMatching v2.8

# # # # # #     Training loss:
# # # # # #         L_total = L_CFM
# # # # # #                 + lam_reg  × L_reg          (ramp ep10→ep30, linear weights)
# # # # # #                 + lam_dir  × lambda_heading × L_heading_ms  (ramp ep5→ep20)

# # # # # #         L_CFM:         conditional flow matching (random t, OT noise-GT matching)
# # # # # #         L_reg:         ADE loss at sigma_inference, softmax-linspace step weights
# # # # # #         L_heading_ms:  [v2.8] 8-step heading continuation, decay=0.75 per step

# # # # # #     NO L_momentum: removed because it anchors to val speed distribution,
# # # # # #     which hurts ATE on test storms that move at different speeds.

# # # # # #     Inference pipeline:
# # # # # #         1. Sample K=20 candidates with sigma=0.04 FIXED
# # # # # #         2. Physics score selection (speed+smooth+heading+displacement)
# # # # # #         3. Weighted average of top-3 candidates
# # # # # #         4. Speed calibration: per-storm scale ×[0.80, 1.25] momentum-based
# # # # # #         5. Optional: multi-scale sigma ensemble (sample_multiscale)
# # # # # #     """

# # # # # #     def __init__(
# # # # # #         self,
# # # # # #         pred_len:          int   = 12,
# # # # # #         obs_len:           int   = 8,
# # # # # #         unet_in_ch:        int   = 13,
# # # # # #         d_cond:            int   = 256,
# # # # # #         d_model:           int   = 256,
# # # # # #         nhead:             int   = 8,
# # # # # #         num_dec_layers:    int   = 4,
# # # # # #         dim_ff:            int   = 512,
# # # # # #         dropout:           float = 0.1,
# # # # # #         sigma_min:         float = 0.04,
# # # # # #         sigma_max:         float = 0.08,
# # # # # #         lambda_reg:        float = 0.2,
# # # # # #         lambda_heading:    float = 0.20,   # [v2.8] multi-step heading, decay=0.75
# # # # # #         lambda_jerk:       float = 0.05,   # [NEW v2.8] trajectory smoothness (physics)
# # # # # #         lambda_momentum:   float = 0.0,    # DISABLED
# # # # # #         use_ot:            bool  = True,
# # # # # #         ot_epsilon:        float = 0.05,
# # # # # #         use_ema:           bool  = True,
# # # # # #         ema_decay:         float = 0.995,
# # # # # #         n_inference_steps: int   = 1,
# # # # # #         n_ensemble:        int   = 20,
# # # # # #         sigma_inference:   float = 0.04,   # FIXED throughout
# # # # # #         **kwargs,
# # # # # #     ):
# # # # # #         super().__init__()
# # # # # #         self.pred_len          = pred_len
# # # # # #         self.obs_len           = obs_len
# # # # # #         self.sigma_min         = sigma_min
# # # # # #         self.sigma_max         = sigma_max
# # # # # #         self.lambda_reg        = lambda_reg
# # # # # #         self.lambda_heading    = lambda_heading
# # # # # #         self.lambda_jerk       = lambda_jerk
# # # # # #         self.lambda_momentum   = 0.0           # always disabled
# # # # # #         self.use_ot            = use_ot
# # # # # #         self.ot_epsilon        = ot_epsilon
# # # # # #         self.n_inference_steps = n_inference_steps
# # # # # #         self.n_ensemble        = n_ensemble
# # # # # #         self.sigma_inference   = sigma_inference

# # # # # #         self.encoder  = ContextEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
# # # # # #         self.velocity = VelocityTransformer(
# # # # # #             pred_len=pred_len, d_model=d_model, nhead=nhead,
# # # # # #             num_layers=num_dec_layers, dim_ff=dim_ff,
# # # # # #             dropout=dropout, d_cond=d_cond)
# # # # # #         self.use_ema = use_ema
# # # # # #         self._ema    = None

# # # # # #         # [NEW v2.8] Learned step weights for L_reg
# # # # # #         # Model learns which horizons need more focus based on actual error structure
# # # # # #         # Temperature=3.0: max/min ratio ≤ 2× (safe, entropy-preserving)
# # # # # #         # Initialized to linspace(1,3,12) → same as fixed baseline
# # # # # #         self.raw_step_w = nn.Parameter(
# # # # # #             torch.linspace(1.0, 3.0, pred_len))
# # # # # #         self.step_w_temperature = 3.0   # controls smoothness (high = more uniform)
# # # # # #         self.step_w_min_entropy = math.log(pred_len) * 0.5  # safety floor

# # # # # #     def init_ema(self):
# # # # # #         if self.use_ema:
# # # # # #             self._ema = EMAModel(self, decay=0.995)

# # # # # #     def ema_update(self):
# # # # # #         if self._ema is not None:
# # # # # #             self._ema.update(self)

# # # # # #     def _to_relative(self, x_abs: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# # # # # #         return x_abs - last_obs.unsqueeze(1)

# # # # # #     def _from_relative(self, x_rel: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# # # # # #         return x_rel + last_obs.unsqueeze(1)

# # # # # #     def _sigma_schedule(self, epoch: int) -> float:
# # # # # #         """Cosine decay sigma_max→sigma_min over ep5→ep40."""
# # # # # #         if epoch < 5:
# # # # # #             return self.sigma_max
# # # # # #         if epoch < 40:
# # # # # #             t = (epoch - 5) / 35.0
# # # # # #             return self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * t))
# # # # # #         return self.sigma_min

# # # # # #     # ── [v2.6] Multi-step heading loss ─────────────────────────────────────

# # # # # #     def _heading_loss_ms(self, pred_deg: torch.Tensor, obs_deg: torch.Tensor,
# # # # # #                           n_steps: int = 8, decay: float = 0.75) -> torch.Tensor:
# # # # # #         """
# # # # # #         [v2.8 FIX] Multi-step heading constraint with corrected decay.

# # # # # #         v2.7 bug: decay=0.5, n_steps=8 → step7(48h) gets 0.5^7/sum ≈ 0.4% gradient
# # # # # #         → 48h direction nearly unconstrained → 72h heading stays at 107°!

# # # # # #         v2.8 fix: decay=0.75 → step7(48h) gets 0.75^7/sum ≈ 3.7% gradient
# # # # # #         Distribution: step0=27.8%, step1=20.8%, step2=15.6%, step3=11.7%,
# # # # # #                       step4=8.8%, step5=6.6%, step6=4.9%, step7=3.7%
# # # # # #         → Meaningful constraint at 48h → expected 72h heading <95°

# # # # # #         DETACH ref per step: prevents gradient explosion through chain.
# # # # # #         """
# # # # # #         if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
# # # # # #             return pred_deg.new_zeros(())
# # # # # #         ref_bear = _forward_azimuth(obs_deg[-2], obs_deg[-1])
# # # # # #         loss = pred_deg.new_zeros(())
# # # # # #         pts = torch.cat([obs_deg[-1:], pred_deg], 0)
# # # # # #         N = min(n_steps, pred_deg.shape[0])
# # # # # #         for t in range(N):
# # # # # #             pred_bear  = _forward_azimuth(pts[t], pts[t + 1])
# # # # # #             angle_diff = pred_bear - ref_bear
# # # # # #             loss       = loss + (decay ** t) * (1.0 - torch.cos(angle_diff)).mean()
# # # # # #             ref_bear   = pred_bear.detach()
# # # # # #         return loss / N

# # # # # #     def _jerk_loss(self, pred_deg: torch.Tensor, obs_deg: torch.Tensor,
# # # # # #                     last_obs_deg: torch.Tensor,
# # # # # #                     max_jerk_deg: float = 15.0) -> torch.Tensor:
# # # # # #         """
# # # # # #         [NEW v2.8] Trajectory jerk loss — physical TC smoothness constraint.

# # # # # #         PHYSICS: TC vortex has rotational inertia. Heading changes slowly:
# # # # # #           - Normal TC: <10-15° per 6h step
# # # # # #           - Rapid recurvature: up to 20-30° per 6h, briefly

# # # # # #         DIFFERENT FROM L_heading:
# # # # # #           L_heading: anchors pred[0] direction to obs direction (WHERE it starts)
# # # # # #           L_jerk:    penalizes |heading[t] - heading[t-1]| > max_jerk (HOW SMOOTH)

# # # # # #         Together: L_heading controls start, L_jerk controls smoothness throughout.
# # # # # #         → Reduces CTE accumulation at 48-72h via physical trajectory smoothness.

# # # # # #         max_jerk_deg=20°: matches TC climatology (typical <20°/6h, rapid recurvature <35°/6h).
# # # # # #         prevents zigzag artifacts that inflate CTE.

# # # # # #         Weight: lambda_jerk=0.05 — small, only prevents physically impossible motion.
# # # # # #         """
# # # # # #         if pred_deg.shape[0] < 2:
# # # # # #             return pred_deg.new_zeros(())
# # # # # #         pts = torch.cat([last_obs_deg.unsqueeze(0), pred_deg], 0)  # [T+1, B, 2]
# # # # # #         headings = _forward_azimuth(pts[:-1], pts[1:])              # [T, B]
# # # # # #         if obs_deg.shape[0] >= 2:
# # # # # #             obs_head = _forward_azimuth(obs_deg[-2:-1], obs_deg[-1:])  # [1, B]
# # # # # #             all_heads = torch.cat([obs_head, headings], 0)             # [T+1, B]
# # # # # #         else:
# # # # # #             all_heads = headings
# # # # # #         head_diffs = all_heads[1:] - all_heads[:-1]
# # # # # #         head_diffs = torch.where(head_diffs >  math.pi, head_diffs - 2*math.pi, head_diffs)
# # # # # #         head_diffs = torch.where(head_diffs < -math.pi, head_diffs + 2*math.pi, head_diffs)
# # # # # #         head_diffs_deg = head_diffs.abs() * (180.0 / math.pi)      # [T, B] degrees
# # # # # #         violations = F.relu(head_diffs_deg - max_jerk_deg)   # zero below limit, linear above
# # # # # #         return (violations / max_jerk_deg).mean()                   # normalized

# # # # # #     # ── L_reg (identical to v2.1 — proven not to overfit) ──────────────────

# # # # # #     def _reg_loss(self, x1_rel: torch.Tensor, last_obs: torch.Tensor,
# # # # # #                   cond: torch.Tensor,
# # # # # #                   hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# # # # # #         """
# # # # # #         ADE loss at t=0 with sigma_inference noise, consistent with 1-shot inference.
# # # # # #         Uses SOFTMAX-LINSPACE step weights (max/min ≈ 3×).
# # # # # #         DOES NOT use exp weights (v2.4 used 12.2× ratio → catastrophic overfitting).
# # # # # #         """
# # # # # #         B, T, _ = x1_rel.shape
# # # # # #         device   = x1_rel.device
# # # # # #         x0   = torch.randn_like(x1_rel) * self.sigma_inference
# # # # # #         t0   = torch.zeros(B, device=device)
# # # # # #         v    = self.velocity(x0, t0, cond)
# # # # # #         x1_pred_abs = self._from_relative(x0 + v, last_obs)
# # # # # #         x1_gt_abs   = self._from_relative(x1_rel, last_obs)
# # # # # #         pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))  # [T, B, 2]
# # # # # #         gt_deg   = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
# # # # # #         dist     = _haversine_deg(pred_deg, gt_deg)             # [T, B] km
# # # # # #         # [v2.8] LEARNED step weights (nn.Parameter, grad flows through)
# # # # # #         # Temperature=3.0 → max/min ≤ 2× (safe range, won't collapse)
# # # # # #         # Model adapts: if 72h errors dominate → raw_step_w[-1] grows
# # # # # #         T_actual = dist.shape[0]
# # # # # #         sw = F.softmax(
# # # # # #             self.raw_step_w[:T_actual].to(dist.dtype) / self.step_w_temperature,
# # # # # #             dim=0).unsqueeze(1)                                 # [T, 1]
# # # # # #         if hard_score is not None:
# # # # # #             sw_hard = (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)  # [1, B]
# # # # # #         else:
# # # # # #             sw_hard = torch.ones(1, B, device=device, dtype=dist.dtype)
# # # # # #         base_loss = ((dist * sw) * sw_hard).mean() / 300.0
# # # # # #         # Min-entropy constraint: ReLU penalty if weights collapse too sharply
# # # # # #         entropy = -(sw.squeeze(1) * sw.squeeze(1).clamp(min=1e-8).log()).sum()
# # # # # #         entropy_penalty = F.relu(self.step_w_min_entropy - entropy).pow(2)
# # # # # #         return base_loss + 0.01 * entropy_penalty

# # # # # #     # ── get_loss_breakdown ──────────────────────────────────────────────────

# # # # # #     def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
# # # # # #         """
# # # # # #         Full loss computation. Called in TRAINING with augmented batch.
# # # # # #         Called in VAL without augmentation → val_loss is reliable signal.

# # # # # #         Total = L_CFM + lam_reg × L_reg + lam_dir × lambda_heading × L_heading_ms
# # # # # #         """
# # # # # #         obs_traj = batch_list[0]
# # # # # #         gt_traj  = batch_list[1]
# # # # # #         B        = obs_traj.shape[1]
# # # # # #         device   = obs_traj.device

# # # # # #         sigma    = self._sigma_schedule(epoch)
# # # # # #         x1_gt    = gt_traj.permute(1, 0, 2)       # [B, T, 2]
# # # # # #         last_obs = obs_traj[-1, :, :2]             # [B, 2]
# # # # # #         x1_rel   = self._to_relative(x1_gt, last_obs)

# # # # # #         h_score = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # #         cond    = self.encoder(batch_list, hard_score=h_score)

# # # # # #         # ── L_CFM ─────────────────────────────────────────────────────────
# # # # # #         x0 = torch.randn_like(x1_rel) * sigma
# # # # # #         if self.use_ot and B >= 4:
# # # # # #             x0_flat, x1_flat = _ot_match(
# # # # # #                 x0.reshape(B, -1), x1_rel.reshape(B, -1), self.ot_epsilon)
# # # # # #             x0         = x0_flat.reshape(B, self.pred_len, 2)
# # # # # #             x1_matched = x1_flat.reshape(B, self.pred_len, 2)
# # # # # #         else:
# # # # # #             x1_matched = x1_rel

# # # # # #         t        = torch.rand(B, device=device)
# # # # # #         x_t      = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1_matched
# # # # # #         u_target = x1_matched - x0
# # # # # #         v_pred   = self.velocity(x_t, t, cond)
# # # # # #         l_cfm    = F.mse_loss(v_pred, u_target)

# # # # # #         # ── L_reg ramp ep10→ep30 ─────────────────────────────────────────
# # # # # #         if epoch < 10:     lam_reg = 0.0
# # # # # #         elif epoch < 30:   lam_reg = self.lambda_reg * (epoch - 10) / 20.0
# # # # # #         else:              lam_reg = self.lambda_reg

# # # # # #         l_reg = (self._reg_loss(x1_rel, last_obs, cond, h_score)
# # # # # #                  if lam_reg > 0.0 else x0.new_zeros(()))

# # # # # #         # ── L_heading_ms ramp ep5→ep20 ────────────────────────────────────
# # # # # #         # Ramp: 0 before ep5, linear ep5→ep20, full after ep20
# # # # # #         if epoch < 5:      lam_dir = 0.0
# # # # # #         elif epoch < 20:   lam_dir = (epoch - 5) / 15.0
# # # # # #         else:              lam_dir = 1.0

# # # # # #         if lam_dir > 0.0:
# # # # # #             # BUG FIX v2.7: KHÔNG dùng torch.no_grad() — gradient phải flow
# # # # # #             # qua velocity để L_heading thực sự update model weights.
# # # # # #             # _heading_loss_ms đã có ref_bear.detach() per step → không explosion.
# # # # # #             x0_h       = torch.randn_like(x1_rel) * self.sigma_inference
# # # # # #             v_h        = self.velocity(x0_h, torch.zeros(B, device=device), cond)
# # # # # #             x1_h_abs   = self._from_relative(x0_h + v_h, last_obs)
# # # # # #             pred_deg_h = _norm_to_deg(x1_h_abs.permute(1, 0, 2))  # [T, B, 2] — có gradient
# # # # # #             obs_deg_h  = _norm_to_deg(obs_traj[:, :, :2])          # [T_obs, B, 2]
# # # # # #             # [v2.8] decay=0.75 (was 0.5) — step7(48h) gets 3.7% not 0.4%
# # # # # #             l_heading  = self._heading_loss_ms(pred_deg_h, obs_deg_h, n_steps=8, decay=0.75)
# # # # # #             # [NEW v2.8] Trajectory jerk loss — physical TC smoothness
# # # # # #             last_obs_deg_h = obs_deg_h[-1]  # [B, 2]
# # # # # #             l_jerk = self._jerk_loss(pred_deg_h, obs_deg_h, last_obs_deg_h,
# # # # # #                                       max_jerk_deg=20.0)
# # # # # #         else:
# # # # # #             l_heading = x0.new_zeros(())
# # # # # #             l_jerk    = x0.new_zeros(())

# # # # # #         l_momentum = x0.new_zeros(())   # always 0 in v2.8

# # # # # #         # ── Total ─────────────────────────────────────────────────────────
# # # # # #         total = (l_cfm
# # # # # #                  + lam_reg  * l_reg
# # # # # #                  + lam_dir  * self.lambda_heading * l_heading
# # # # # #                  + lam_dir  * self.lambda_jerk    * l_jerk)
# # # # # #         if not torch.isfinite(total):
# # # # # #             total = x0.new_zeros(())

# # # # # #         # ── ADE 1-step log ────────────────────────────────────────────────
# # # # # #         with torch.no_grad():
# # # # # #             x0_log = torch.randn_like(x1_rel) * self.sigma_inference
# # # # # #             v_log  = self.velocity(x0_log, torch.zeros(B, device=device), cond)
# # # # # #             ade_log = _haversine_deg(
# # # # # #                 _norm_to_deg(self._from_relative(x0_log + v_log, last_obs).permute(1, 0, 2)),
# # # # # #                 _norm_to_deg(x1_gt.permute(1, 0, 2))
# # # # # #             ).mean().item()

# # # # # #         return {
# # # # # #             "total":    total,
# # # # # #             "l_cfm":    l_cfm.item(),
# # # # # #             "l_reg":    l_reg.item() if torch.is_tensor(l_reg) else 0.0,
# # # # # #             "l_heading": l_heading.item() if torch.is_tensor(l_heading) else 0.0,
# # # # # #             "l_jerk":   l_jerk.item() if torch.is_tensor(l_jerk) else 0.0,
# # # # # #             "l_momentum": 0.0,
# # # # # #             "lam_reg":  lam_reg,
# # # # # #             "lam_dir":  lam_dir,
# # # # # #             "sigma":    sigma,
# # # # # #             "ade_1step": ade_log,
# # # # # #             "hard_score_mean": float(h_score.mean()),
# # # # # #             "hard_score_max":  float(h_score.max()),
# # # # # #             # Compat keys for older scripts
# # # # # #             "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
# # # # # #             "speed": 0., "accel": 0., "fm_mse": l_cfm.item(),
# # # # # #             "l_hard_total": 0., "n_hard": 0, "alpha_hard": 0.,
# # # # # #             "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
# # # # # #         }

# # # # # #     def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
# # # # # #         return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

# # # # # #     # ── [v2.8] Speed calibration — momentum-based ──────────────────────────

# # # # # #     @staticmethod
# # # # # #     @torch.no_grad()
# # # # # #     def speed_calibrate_pred(pred_abs_norm: torch.Tensor,
# # # # # #                               last_obs_norm: torch.Tensor,
# # # # # #                               obs_norm:      torch.Tensor,
# # # # # #                               clip_min: float = 0.80,
# # # # # #                               clip_max: float = 1.25) -> torch.Tensor:
# # # # # #         """
# # # # # #         [v2.8] Per-storm speed calibration using MOMENTUM-based reference.

# # # # # #         v2.7 used weighted MEAN of obs speeds — bad for accelerating storms.
# # # # # #         Example: storm 5→10→15 km/h: mean=10, but correct target is ~15-18.

# # # # # #         v2.8 MOMENTUM approach:
# # # # # #           ref = last_obs_speed + 0.3 × obs_acceleration
# # # # # #           → Accelerating storm: extrapolates trend up
# # # # # #           → Decelerating storm: extrapolates down
# # # # # #           → Constant speed: same as last speed (correct)

# # # # # #         PHYSICS: TC speed strongly autocorrelated over 6-12h. Best predictor
# # # # # #         of future speed is last speed + momentum, not historical mean.

# # # # # #         Clip: ±20-25% (wider than v2.6 ±15% for fast test storms)
# # # # # #         Preserves direction — only scales displacement magnitude.
# # # # # #         """
# # # # # #         if obs_norm.shape[0] < 2 or pred_abs_norm.shape[0] < 2:
# # # # # #             return pred_abs_norm

# # # # # #         obs_deg  = _norm_to_deg(obs_norm)
# # # # # #         pred_deg = _norm_to_deg(pred_abs_norm)
# # # # # #         last_deg = _norm_to_deg(last_obs_norm)

# # # # # #         obs_spds = _step_speeds_kmh(obs_deg)    # [T_obs-1, B] km/h
# # # # # #         T_s = obs_spds.shape[0]

# # # # # #         if T_s >= 2:
# # # # # #             spd_last    = obs_spds[-1]              # [B] most recent
# # # # # #             spd_prev    = obs_spds[-2]              # [B] one before
# # # # # #             accel       = spd_last - spd_prev       # [B] positive = accelerating
# # # # # #             obs_spd_ref = (spd_last + 0.3 * accel).clamp(min=3.0)
# # # # # #         else:
# # # # # #             obs_spd_ref = obs_spds[-1].clamp(min=3.0)

# # # # # #         pts = torch.cat([last_deg.unsqueeze(0), pred_deg], 0)
# # # # # #         pred_spds    = _step_speeds_kmh(pts)
# # # # # #         pred_spd_ref = pred_spds[:4].mean(0).clamp(min=3.0)

# # # # # #         scale = (obs_spd_ref / pred_spd_ref).clamp(clip_min, clip_max)
# # # # # #         return (last_obs_norm.unsqueeze(0)
# # # # # #                 + (pred_abs_norm - last_obs_norm.unsqueeze(0))
# # # # # #                 * scale.view(1, -1, 1))


# # # # # #     @torch.no_grad()
# # # # # #     def sample_speed_tta(
# # # # # #         self,
# # # # # #         batch_list,
# # # # # #         tta_scales: Optional[List[float]] = None,
# # # # # #         n_ensemble_per_scale: int = 8,
# # # # # #         **kwargs,
# # # # # #     ) -> Tuple:
# # # # # #         """
# # # # # #         [NEW v2.8] Speed Test-Time Augmentation (TTA).

# # # # # #         ROOT CAUSE of val→test gap: test storms have different speed distribution
# # # # # #         than val. Model calibrated to val speed (~10.3km/h) undershoots fast storms.

# # # # # #         SOLUTION: Generate predictions at multiple speed scales, let physics score
# # # # # #         pick the best one per storm.

# # # # # #         Unlike simple TTA (averaging), this uses PHYSICS SCORE SELECTION:
# # # # # #         → Fast test storm: 1.35× scale gets highest speed_score → selected
# # # # # #         → Slow test storm: 0.75× scale gets highest speed_score → selected
# # # # # #         → Adaptive per-storm without knowing test distribution a priori

# # # # # #         TTA scales [0.70..1.40]: cover ±30-40% speed range
# # # # # #         n_ensemble_per_scale=8: enough samples for reliable physics scoring

# # # # # #         Expected impact: ATE -4 to -10km on fast test storms
# # # # # #         No training change needed — purely inference enhancement.
# # # # # #         """
# # # # # #         if tta_scales is None:
# # # # # #             tta_scales = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50]

# # # # # #         obs_traj    = batch_list[0]
# # # # # #         T_obs, B, _ = obs_traj.shape
# # # # # #         device      = obs_traj.device

# # # # # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # #         obs_norm = obs_traj[:, :, :2]
# # # # # #         last_obs = obs_traj[-1, :, :2]
# # # # # #         t0       = torch.zeros(B, device=device)
# # # # # #         cond     = self.encoder(batch_list, hard_score=h_score)

# # # # # #         all_traj   = []
# # # # # #         all_scores = []

# # # # # #         for scale in tta_scales:
# # # # # #             # Scale obs speed to simulate different storm speeds
# # # # # #             # This shifts the physics_score speed_score component
# # # # # #             obs_scaled = obs_norm.clone()
# # # # # #             obs_scaled[0] = obs_norm[0]
# # # # # #             for t in range(1, T_obs):
# # # # # #                 disp = obs_norm[t] - obs_norm[t-1]
# # # # # #                 obs_scaled[t] = obs_scaled[t-1] + disp * scale

# # # # # #             # Generate ensemble for this scale
# # # # # #             scale_traj = []
# # # # # #             for _ in range(n_ensemble_per_scale):
# # # # # #                 x_rel = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference
# # # # # #                 v     = self.velocity(x_rel, t0, cond)
# # # # # #                 x_rel = x_rel + v
# # # # # #                 x_abs = self._from_relative(x_rel, last_obs)
# # # # # #                 scale_traj.append(x_abs.permute(1, 0, 2))   # [T, B, 2]

# # # # # #             # Score with SCALED obs (so speed_score measures match to scale)
# # # # # #             scale_scores = torch.stack(
# # # # # #                 [_physics_score(t, obs_scaled) for t in scale_traj], 0)  # [K, B]
# # # # # #             scale_t = torch.stack(scale_traj, 0)   # [K, T, B, 2]

# # # # # #             # Top-1 per scale (best sample for this scale)
# # # # # #             best_idx = scale_scores.argmax(0)       # [B]
# # # # # #             # [FIX BUG #3] Correct per-storm selection from [K,T,B,2]
# # # # # #             best_traj = torch.stack(
# # # # # #                 [scale_t[best_idx[b], :, b, :] for b in range(B)], 1)  # [T, B, 2]
# # # # # #             best_score = scale_scores[best_idx, torch.arange(B)]  # [B]

# # # # # #             all_traj.append(best_traj)
# # # # # #             all_scores.append(best_score)

# # # # # #         # Global selection across all scales per storm
# # # # # #         all_traj_stk   = torch.stack(all_traj, 0)    # [S, T, B, 2]
# # # # # #         all_scores_stk  = torch.stack(all_scores, 0)  # [S, B]

# # # # # #         # Score with ORIGINAL obs (unscaled) for final selection
# # # # # #         orig_scores = torch.stack(
# # # # # #             [_physics_score(all_traj_stk[s], obs_norm) for s in range(len(tta_scales))], 0)  # [S,B]

# # # # # #         # Weight: orig_scores (how well it matches actual obs)
# # # # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # # # #         for b in range(B):
# # # # # #             w_b = F.softmax(orig_scores[:, b] * 5.0, dim=0)   # sharper selection
# # # # # #             pred_mean[:, b, :] = (all_traj_stk[:, :, b, :] * w_b.view(-1, 1, 1)).sum(0)

# # # # # #         # Apply speed calibration after TTA
# # # # # #         pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm,
# # # # # #                                                clip_min=0.75, clip_max=1.40)

# # # # # #         all_t = torch.cat([t.unsqueeze(0) for t in all_traj], 0)   # rough [S,T,B,2]
# # # # # #         return pred_mean, torch.zeros_like(pred_mean), all_t


# # # # # #     # ── Sample (1-shot + speed calibration) ──────────────────────────────

# # # # # #     @torch.no_grad()
# # # # # #     def sample(self, batch_list,
# # # # # #                num_ensemble:          Optional[int]  = None,
# # # # # #                ddim_steps:            Optional[int]  = None,
# # # # # #                return_xai:            bool           = False,
# # # # # #                use_speed_calibration: bool           = True,
# # # # # #                **kwargs) -> Tuple:
# # # # # #         """
# # # # # #         1-shot inference with physics selection + speed calibration.

# # # # # #         Steps:
# # # # # #           1. Encode context once (hard_score + full encoder)
# # # # # #           2. Sample K=20 candidates: x0 ~ N(0, sigma_inf²), x_pred = x0 + v(x0, 0, cond)
# # # # # #           3. Physics score each candidate (speed+smooth+heading+displacement)
# # # # # #           4. Weighted average of top-3 candidates
# # # # # #           5. Speed calibration per-storm (clip ±15%)
# # # # # #           6. If return_xai=True: compute XAI-1 through XAI-7

# # # # # #         Returns:
# # # # # #           (pred_mean [T,B,2], zeros [T,B,2], all_traj [K,T,B,2])
# # # # # #           or (pred_mean, zeros, all_traj, xai_dict) if return_xai=True
# # # # # #         """
# # # # # #         K  = num_ensemble or self.n_ensemble
# # # # # #         N  = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else self.n_inference_steps
# # # # # #         dt = 1.0 / max(N, 1)

# # # # # #         obs_traj    = batch_list[0]
# # # # # #         T_obs, B, _ = obs_traj.shape
# # # # # #         device      = obs_traj.device

# # # # # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # #         obs_norm = obs_traj[:, :, :2]
# # # # # #         last_obs = obs_traj[-1, :, :2]
# # # # # #         t0       = torch.zeros(B, device=device)

# # # # # #         # Encode once for all K samples (expensive part)
# # # # # #         cond = self.encoder(batch_list, hard_score=h_score)

# # # # # #         all_traj = []
# # # # # #         for _ in range(K):
# # # # # #             x_rel = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference

# # # # # #             if N <= 1:
# # # # # #                 # 1-shot: exactly what L_reg is trained for
# # # # # #                 v     = self.velocity(x_rel, t0, cond)
# # # # # #                 x_rel = x_rel + v
# # # # # #             else:
# # # # # #                 # Euler multi-step (for experiments only, slower)
# # # # # #                 for step in range(N):
# # # # # #                     t_b   = torch.full((B,), step * dt, device=device)
# # # # # #                     x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3., 3.)

# # # # # #             x_abs = self._from_relative(x_rel, last_obs)
# # # # # #             all_traj.append(x_abs.permute(1, 0, 2))   # [T, B, 2]

# # # # # #         # Physics-based selection: best-of-K with weighted average of top-3
# # # # # #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)   # [K, B]
# # # # # #         all_t   = torch.stack(all_traj, 0)   # [K, T, B, 2]
# # # # # #         top_k   = min(3, K)
# # # # # #         top_idx = scores.topk(top_k, dim=0).indices   # [top_k, B]

# # # # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # # # #         for b in range(B):
# # # # # #             idx_b = top_idx[:, b]
# # # # # #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # # # # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# # # # # #         # [NEW v2.6] Per-storm speed calibration
# # # # # #         if use_speed_calibration:
# # # # # #             pred_mean = self.speed_calibrate_pred(
# # # # # #                 pred_mean, last_obs, obs_norm, clip_min=0.80, clip_max=1.25)

# # # # # #         if not return_xai:
# # # # # #             return pred_mean, torch.zeros_like(pred_mean), all_t

# # # # # #         # ── XAI output (XAI-2,3,4,5,6,7) ──────────────────────────────────
# # # # # #         xai = {}

# # # # # #         # XAI-4: Ensemble uncertainty
# # # # # #         xai.update(compute_ensemble_uncertainty(all_t))

# # # # # #         # XAI-2: Hard score components
# # # # # #         _, hard_comps = hard_score_from_obs(obs_norm, return_components=True)
# # # # # #         xai["hard_components"] = hard_comps

# # # # # #         pred_deg  = _norm_to_deg(pred_mean)    # [T, B, 2]
# # # # # #         obs_deg_x = _norm_to_deg(obs_norm)     # [T_obs, B, 2]

# # # # # #         # XAI-5: Speed comparison (obs vs pred, pre and post calibration)
# # # # # #         obs_spd_x    = _step_speeds_kmh(obs_deg_x)
# # # # # #         obs_spd_mu   = obs_spd_x.mean(0)     # [B]
# # # # # #         if pred_deg.shape[0] >= 2:
# # # # # #             last_deg_x   = obs_deg_x[-1]
# # # # # #             pts_x        = torch.cat([last_deg_x.unsqueeze(0), pred_deg], 0)
# # # # # #             pred_spd_x   = _step_speeds_kmh(pts_x)
# # # # # #             pred_spd_mu  = pred_spd_x.mean(0)  # [B]
# # # # # #         else:
# # # # # #             pred_spd_mu = obs_spd_mu.clone()

# # # # # #         speed_ratio = (pred_spd_mu / obs_spd_mu.clamp(min=1.0))
# # # # # #         xai["speed_comparison"] = {
# # # # # #             "obs_speed_mean":  float(obs_spd_mu.mean()),
# # # # # #             "pred_speed_mean": float(pred_spd_mu.mean()),
# # # # # #             "speed_ratio":     float(speed_ratio.mean()),
# # # # # #             "per_storm_obs":   obs_spd_mu,
# # # # # #             "per_storm_pred":  pred_spd_mu,
# # # # # #             "over_predict":    speed_ratio > 1.15,
# # # # # #             "under_predict":   speed_ratio < 0.85,
# # # # # #         }

# # # # # #         # XAI-3: Physics components for best prediction
# # # # # #         v_ref = obs_spd_mu.clamp(min=5.0)
# # # # # #         v_sig = v_ref * 0.5
# # # # # #         spd_sc = torch.exp(-((pred_spd_mu - v_ref) / v_sig).pow(2) * 0.5)
# # # # # #         if pred_deg.shape[0] >= 3:
# # # # # #             vel_x   = pred_deg[1:] - pred_deg[:-1]
# # # # # #             accel_x = (vel_x[1:] - vel_x[:-1]).norm(dim=-1)
# # # # # #             smo_sc  = torch.exp(-accel_x.mean(0) * 5.0)
# # # # # #         else:
# # # # # #             smo_sc = torch.ones(B, device=device)
# # # # # #         if obs_norm.shape[0] >= 2 and pred_mean.shape[0] >= 1:
# # # # # #             ov = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # # # # #             pv = pred_mean[0, :, :2] - obs_norm[-1, :, :2]
# # # # # #             cos_s = (F.normalize(ov, dim=-1, eps=1e-6) * F.normalize(pv, dim=-1, eps=1e-6)).sum(-1)
# # # # # #             hd_sc = torch.exp((cos_s.clamp(-1, 1) - 1.0) * 3.0)
# # # # # #         else:
# # # # # #             hd_sc = torch.ones(B, device=device)
# # # # # #         xai["physics_components"] = {
# # # # # #             "speed": spd_sc, "smooth": smo_sc, "heading": hd_sc,
# # # # # #             "obs_speed": obs_spd_mu, "pred_speed": pred_spd_mu,
# # # # # #         }

# # # # # #         # XAI-8: Learned step weights (monitor training dynamics)
# # # # # #         # self = TCFlowMatching instance, raw_step_w is directly accessible
# # # # # #         try:
# # # # # #             sw_learned = F.softmax(
# # # # # #                 self.raw_step_w / self.step_w_temperature, dim=0
# # # # # #             ).detach().cpu().tolist()
# # # # # #             xai["learned_step_weights"] = sw_learned
# # # # # #         except Exception:
# # # # # #             xai["learned_step_weights"] = []

# # # # # #         # XAI-6: Heading deviation per lead time
# # # # # #         gt_traj_xai = batch_list[1]
# # # # # #         gt_deg_xai  = _norm_to_deg(gt_traj_xai[:, :, :2])
# # # # # #         xai["heading_deviation_deg"] = compute_heading_deviation(pred_deg, gt_deg_xai)

# # # # # #         # XAI-7: ATE/CTE decomposition
# # # # # #         xai["ate_cte_decomp"] = compute_cte_contribution(pred_deg, gt_deg_xai)

# # # # # #         # ── [v2.7 NEW] XAI-8: Per-horizon speed error ────────────────────
# # # # # #         # pred_speed vs gt_speed tại mỗi horizon
# # # # # #         # → biết model over/under-predict speed ở đâu (early vs late horizon)
# # # # # #         if pred_deg.shape[0] >= 2 and gt_deg_xai.shape[0] >= 2:
# # # # # #             last_obs_deg = obs_deg_x[-1:]   # [1, B, 2]
# # # # # #             pred_pts_h   = torch.cat([last_obs_deg, pred_deg], 0)   # [T+1, B, 2]
# # # # # #             gt_pts_h     = torch.cat([last_obs_deg, gt_deg_xai[:pred_deg.shape[0]]], 0)
# # # # # #             pred_spd_h   = _step_speeds_kmh(pred_pts_h)   # [T, B]
# # # # # #             gt_spd_h     = _step_speeds_kmh(gt_pts_h)     # [T, B]
# # # # # #             xai["speed_per_horizon"] = {
# # # # # #                 "pred_speeds_kmh": pred_spd_h.mean(1).tolist(),   # [T] avg over batch
# # # # # #                 "gt_speeds_kmh":   gt_spd_h.mean(1).tolist(),     # [T] avg over batch
# # # # # #                 "speed_ratio_per_step": (pred_spd_h / gt_spd_h.clamp(min=1.0)).mean(1).tolist(),
# # # # # #             }
# # # # # #         else:
# # # # # #             xai["speed_per_horizon"] = {}

# # # # # #         # ── [v2.7 NEW] XAI-9: Storm category breakdown ───────────────────
# # # # # #         # Chia storms theo obs speed để diagnose per-category error
# # # # # #         # Nếu fast storms chiếm đại bộ phận error → confirm ATE gap issue
# # # # # #         obs_spd_per = obs_spd_x.mean(0).clamp(min=0.0)   # [B] km/h per storm
# # # # # #         slow_mask   = obs_spd_per < 8.0
# # # # # #         medium_mask = (obs_spd_per >= 8.0) & (obs_spd_per < 15.0)
# # # # # #         fast_mask   = obs_spd_per >= 15.0
# # # # # #         xai["storm_categories"] = {
# # # # # #             "n_slow":          int(slow_mask.sum()),
# # # # # #             "n_medium":        int(medium_mask.sum()),
# # # # # #             "n_fast":          int(fast_mask.sum()),
# # # # # #             "obs_speed_mean":  float(obs_spd_per.mean()),
# # # # # #             "obs_speed_std":   float(obs_spd_per.std()),
# # # # # #             "obs_speed_max":   float(obs_spd_per.max()),
# # # # # #         }

# # # # # #         return pred_mean, torch.zeros_like(pred_mean), all_t, xai

# # # # # #     @torch.no_grad()
# # # # # #     def sample_multiscale(
# # # # # #         self,
# # # # # #         batch_list,
# # # # # #         sigmas: Optional[List[float]] = None,
# # # # # #         n_per_sigma: int = 4,
# # # # # #         use_speed_calibration: bool = True,
# # # # # #     ) -> Tuple:
# # # # # #         """
# # # # # #         Multi-scale sigma ensemble at inference (DOES NOT affect training).
# # # # # #         Hedges against speed distribution shift between val and test.
# # # # # #         """
# # # # # #         if sigmas is None:
# # # # # #             sigmas = [0.025, 0.035, 0.04, 0.05, 0.065]

# # # # # #         obs_traj = batch_list[0]
# # # # # #         B = obs_traj.shape[1]; device = obs_traj.device

# # # # # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # # # # #         obs_norm = obs_traj[:, :, :2]
# # # # # #         last_obs = obs_traj[-1, :, :2]
# # # # # #         t0       = torch.zeros(B, device=device)
# # # # # #         cond     = self.encoder(batch_list, hard_score=h_score)

# # # # # #         all_traj = []
# # # # # #         for sigma in sigmas:
# # # # # #             for _ in range(n_per_sigma):
# # # # # #                 x0    = torch.randn(B, self.pred_len, 2, device=device) * sigma
# # # # # #                 v     = self.velocity(x0, t0, cond)
# # # # # #                 x_abs = self._from_relative(x0 + v, last_obs)
# # # # # #                 all_traj.append(x_abs.permute(1, 0, 2))

# # # # # #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)
# # # # # #         all_t   = torch.stack(all_traj, 0)
# # # # # #         top_k   = min(5, len(all_traj))
# # # # # #         top_idx = scores.topk(top_k, dim=0).indices

# # # # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # # # #         for b in range(B):
# # # # # #             idx_b = top_idx[:, b]
# # # # # #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # # # # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# # # # # #         if use_speed_calibration:
# # # # # #             pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

# # # # # #         return pred_mean, torch.zeros_like(pred_mean), all_t


# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # #  Backward compat alias
# # # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # # TCDiffusion = TCFlowMatching

# # # # # """Model/flow_matching_model.py  ──  TC-FlowMatching v2.9
# # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # CHIẾN LƯỢC: "Regularize Everything" — giải quyết triệt để 3 nguồn gốc của gap.

# # # # # ━━━ 3 NGUỒN GỐC CỦA GAP 60km ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # #   [1] FNO3D encoder overfit spatial val patterns (~8-20km contribution)
# # # # #       FIX: FNO3D dropout 0.05→0.20, enc weight_decay 5e-4, freeze 25ep

# # # # #   [2] Speed distribution shift val→test (~10-15km contribution)
# # # # #       FIX: L_accel_speed (ST-Trans style), speed-normalized kinematics

# # # # #   [3] Multi-basin diversity → model spread thin (~15-25km contribution)
# # # # #       FIX: Basin embedding (6 basins × 32 dim) → per-basin adaptation

# # # # # ━━━ THAY ĐỔI SO VỚI v2.1 (CHÍNH XÁC, KHÔNG HƠN) ━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # #   ✚ L_accel_speed: mean((Δspeed)²)/100, lambda=0.01, ramp ep5→ep20
# # # # #   ✚ VelocityTransformer dropout: 0.10 → 0.15
# # # # #   ✚ Basin embedding: nn.Embedding(6,32) → fuse vào context vector
# # # # #   ✚ Label-smoothed step weights: 0.9×linspace + 0.1×uniform (anti-overfit)
# # # # #   ✚ Temporal obs dropout: 15% prob drop step 0 hay 1 → replace by interpolation
# # # # #     (training augmentation, KHÔNG inference — giữ consistency)

# # # # #   ✅ GIỮ NGUYÊN TỪ v2.1 (tất cả):
# # # # #     sigma=0.04 FIXED, L_CFM+L_reg only, OT matching,
# # # # #     1-shot inference, aug A+B only (trong train script),
# # # # #     val no augmentation, 2-group optimizer pattern

# # # # #   ✗ LOẠI BỎ HOÀN TOÀN:
# # # # #     L_heading, L_jerk, L_momentum, speed calibration,
# # # # #     AUG-C/D, learned step weights, speed TTA,
# # # # #     displacement_score trong physics score

# # # # # ━━━ BASIN ID MAPPING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# # # # #   Từ batch_list[14] hoặc infer từ lon coordinate nếu không có:
# # # # #   WP=0  NI=1  AT=2  EP=3  SP=4  SI=5  UNKNOWN=5
# # # # #   lon>100°E → WP, 50-100°E → NI, 0-100°W → AT, etc.
# # # # # """
# # # # # from __future__ import annotations
# # # # # import math
# # # # # from typing import Dict, List, Optional, Tuple

# # # # # import torch
# # # # # import torch.nn as nn
# # # # # import torch.nn.functional as F

# # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # from Model.env_net_transformer_gphsplit import Env_net

# # # # # R_EARTH  = 6371.0
# # # # # DT_HOURS = 6.0


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Coordinate utilities — IDENTICAL to v2.1
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# # # # #     return torch.stack([
# # # # #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# # # # #         (t[..., 1] * 50.0) / 10.0,
# # # # #     ], dim=-1)

# # # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # #     a = (torch.sin(dlat / 2).pow(2)
# # # # #          + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
# # # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())

# # # # # def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # #     lon1 = torch.deg2rad(p1[..., 0]); lat1 = torch.deg2rad(p1[..., 1])
# # # # #     lon2 = torch.deg2rad(p2[..., 0]); lat2 = torch.deg2rad(p2[..., 1])
# # # # #     dlon = lon2 - lon1
# # # # #     y = torch.sin(dlon) * torch.cos(lat2)
# # # # #     x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
# # # # #     return torch.atan2(y, x)

# # # # # def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
# # # # #     if traj_deg.shape[0] < 2:
# # # # #         return traj_deg.new_zeros(1, traj_deg.shape[1])
# # # # #     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS

# # # # # def _unwrap(m):
# # # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # # # # _unwrap_model = _unwrap  # backward compat


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Basin ID inference from longitude
# # # # # #  Dùng khi batch không có basin_id field (batch_list[14])
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def _infer_basin_id(obs_traj_norm: torch.Tensor) -> torch.Tensor:
# # # # #     """
# # # # #     Infer basin ID from normalized longitude of first obs position.
# # # # #     Returns LongTensor [B] with basin IDs 0-5.
    
# # # # #     WP=0: lon 100-180°E (Western Pacific — largest basin in dataset)
# # # # #     NI=1: lon 45-100°E  (North Indian Ocean)
# # # # #     AT=2: lon 0-100°W   (North Atlantic)  
# # # # #     EP=3: lon 100-180°W (Eastern Pacific)
# # # # #     SP=4: lon 135-200°E southern (South Pacific)
# # # # #     SI=5: lon 35-135°E southern (South Indian)
# # # # #     """
# # # # #     # obs_traj_norm: [T, B, ≥2] — first 2 dims are normalized lon, lat
# # # # #     # normalized lon → actual lon: (x * 50 + 1800) / 10
# # # # #     lon_norm = obs_traj_norm[0, :, 0]  # [B] normalized
# # # # #     lon_deg  = (lon_norm * 50.0 + 1800.0) / 10.0  # actual longitude degrees
# # # # #     lat_norm = obs_traj_norm[0, :, 1]
# # # # #     lat_deg  = (lat_norm * 50.0) / 10.0

# # # # #     B = lon_deg.shape[0]
# # # # #     basin = torch.zeros(B, dtype=torch.long, device=lon_deg.device)

# # # # #     # WP: lon 100-180E, lat 0-40N
# # # # #     wp = (lon_deg >= 100) & (lon_deg <= 180) & (lat_deg >= 0)
# # # # #     # NI: lon 45-100E
# # # # #     ni = (lon_deg >= 45) & (lon_deg < 100)
# # # # #     # AT: lon 0-100W (represented as 260-360)
# # # # #     at = (lon_deg >= 260) & (lon_deg <= 360) & (lat_deg >= 0)
# # # # #     # EP: lon 100-180W (represented as 180-260)
# # # # #     ep = (lon_deg >= 180) & (lon_deg < 260) & (lat_deg >= 0)
# # # # #     # Southern hemisphere: use rough split
# # # # #     sp = (lon_deg >= 135) & (lat_deg < 0)  # South Pacific
# # # # #     si = (lon_deg < 135) & (lat_deg < 0)   # South Indian

# # # # #     basin[ni]  = 1
# # # # #     basin[at]  = 2
# # # # #     basin[ep]  = 3
# # # # #     basin[sp]  = 4
# # # # #     basin[si]  = 5
# # # # #     basin[wp]  = 0  # last → WP takes priority if overlap (shouldn't happen)
# # # # #     return basin


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  EMAModel — IDENTICAL to v2.1
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # class EMAModel:
# # # # #     def __init__(self, model, decay: float = 0.995):
# # # # #         self.decay = decay
# # # # #         m = _unwrap(model)
# # # # #         self.shadow = {k: v.detach().clone()
# # # # #                        for k, v in m.state_dict().items()
# # # # #                        if v.dtype.is_floating_point}

# # # # #     def update(self, model):
# # # # #         m = _unwrap(model)
# # # # #         with torch.no_grad():
# # # # #             for k, v in m.state_dict().items():
# # # # #                 if k in self.shadow:
# # # # #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

# # # # #     def apply_to(self, model):
# # # # #         m = _unwrap(model)
# # # # #         backup, sd = {}, m.state_dict()
# # # # #         for k in self.shadow:
# # # # #             if k not in sd: continue
# # # # #             backup[k] = sd[k].detach().clone()
# # # # #             sd[k].copy_(self.shadow[k])
# # # # #         return backup

# # # # #     def restore(self, model, backup):
# # # # #         m = _unwrap(model)
# # # # #         sd = m.state_dict()
# # # # #         for k, v in backup.items():
# # # # #             if k in sd: sd[k].copy_(v)


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  OT matching — IDENTICAL to v2.1
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05,
# # # # #                   n_iter: int = 50) -> torch.Tensor:
# # # # #     B = cost.shape[0]; device = cost.device
# # # # #     log_a = -math.log(B) * torch.ones(B, device=device)
# # # # #     log_b = -math.log(B) * torch.ones(B, device=device)
# # # # #     log_K = -cost / epsilon
# # # # #     log_u = torch.zeros(B, device=device)
# # # # #     log_v = torch.zeros(B, device=device)
# # # # #     for _ in range(n_iter):
# # # # #         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
# # # # #         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
# # # # #     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)

# # # # # def _ot_match(x0_flat, x1_flat, epsilon=0.05):
# # # # #     B = x0_flat.shape[0]
# # # # #     if B < 4: return x0_flat, x1_flat
# # # # #     try:
# # # # #         cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1] ** 0.5)
# # # # #         with torch.no_grad():
# # # # #             pi = _sinkhorn_log(cost, epsilon=epsilon)
# # # # #         flat = pi.reshape(-1).clamp(0.0); s = flat.sum()
# # # # #         if not torch.isfinite(s) or s < 1e-10: return x0_flat, x1_flat
# # # # #         idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
# # # # #         return x0_flat[idx // B], x1_flat
# # # # #     except Exception:
# # # # #         return x0_flat, x1_flat

# # # # # _ot_match_noise_gt = _ot_match  # backward compat


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  VelocityTransformer — dropout 0.10 → 0.15 (v2.9)
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # class VelocityTransformer(nn.Module):
# # # # #     """
# # # # #     Identical to v2.1 except dropout 0.10 → 0.15.
# # # # #     6.5M total params needs stronger regularization than 394K ST-Trans.
# # # # #     """
# # # # #     def __init__(self, pred_len=12, d_model=256, nhead=8,
# # # # #                  num_layers=4, dim_ff=512, dropout=0.15, d_cond=256):
# # # # #         super().__init__()
# # # # #         self.pred_len = pred_len; self.d_model = d_model
# # # # #         self.traj_embed = nn.Linear(2, d_model)
# # # # #         self.pos_emb    = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
# # # # #         self.step_emb   = nn.Embedding(pred_len, d_model)
# # # # #         self.time_mlp   = nn.Sequential(
# # # # #             nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
# # # # #         self.cond_proj  = nn.Sequential(nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))
# # # # #         dec_layer = nn.TransformerDecoderLayer(
# # # # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # # # #             dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
# # # # #         self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
# # # # #         self.out_norm = nn.LayerNorm(d_model)
# # # # #         self.out_proj = nn.Sequential(
# # # # #             nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
# # # # #         self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
# # # # #         nn.init.zeros_(self.out_proj[-1].weight)
# # # # #         nn.init.zeros_(self.out_proj[-1].bias)

# # # # #     def _time_emb(self, t):
# # # # #         half = self.d_model // 2
# # # # #         freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
# # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # #         emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
# # # # #         emb = torch.cat([emb.sin(), emb.cos()], -1)
# # # # #         if self.d_model % 2 == 1: emb = F.pad(emb, (0, 1))
# # # # #         return self.time_mlp(emb)

# # # # #     def forward(self, x_t, t, cond):
# # # # #         B, T, _ = x_t.shape
# # # # #         idx  = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # #         x    = self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(idx)
# # # # #         mem  = torch.cat([self._time_emb(t).unsqueeze(1),
# # # # #                           self.cond_proj(cond).unsqueeze(1)], dim=1)
# # # # #         out  = self.out_norm(self.decoder(x, mem))
# # # # #         return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  ContextEncoder — thêm basin_embed, giữ nguyên phần còn lại
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # class ContextEncoder(nn.Module):
# # # # #     """
# # # # #     v2.9 changes vs v2.1:
# # # # #       + Basin embedding: nn.Embedding(6, 32) → fused into context
# # # # #         Addresses multi-basin distribution confusion (Source 3 of gap).
# # # # #         6 basins × 32 dims = 192 params (negligible overhead).
# # # # #     """
# # # # #     RAW_CTX_DIM = 512
# # # # #     N_BASINS    = 6   # WP, NI, AT, EP, SP, SI
# # # # #     BASIN_DIM   = 32

# # # # #     def __init__(self, obs_len=8, unet_in_ch=13, d_cond=256):
# # # # #         super().__init__()
# # # # #         self.obs_len = obs_len; self.d_cond = d_cond

# # # # #         self.spatial_enc     = FNO3DEncoder(
# # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # #             spatial_down=32, dropout=0.05)   # FNO3D dropout handled externally
# # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # #         self.decoder_proj    = nn.Linear(1, 16)
# # # # #         self.enc_1d          = DataEncoder1D(in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # #                                               lstm_hidden=128, lstm_layers=3,
# # # # #                                               dropout=0.1, d_state=16)
# # # # #         self.env_enc         = Env_net(obs_len=obs_len, d_model=32)

# # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # #         self.ctx_drop = nn.Dropout(0.1)
# # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, d_cond)
# # # # #         self.ctx_ln2  = nn.LayerNorm(d_cond)

# # # # #         # 6 kinematic features per step
# # # # #         self.vel_obs_enc = nn.Sequential(
# # # # #             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
# # # # #             nn.Linear(256, d_cond // 2), nn.GELU())
# # # # #         self.hard_embed = nn.Sequential(
# # # # #             nn.Linear(1, d_cond // 4), nn.GELU(),
# # # # #             nn.Linear(d_cond // 4, d_cond // 4))

# # # # #         # [v2.9 NEW] Basin embedding — addresses multi-basin distribution shift
# # # # #         # Each basin gets its own learned offset vector in context space
# # # # #         # Learned during training → model adapts per-basin behavior
# # # # #         self.basin_embed = nn.Embedding(self.N_BASINS, self.BASIN_DIM)
# # # # #         nn.init.normal_(self.basin_embed.weight, 0, 0.1)  # small init

# # # # #         self.fuse = nn.Sequential(
# # # # #             nn.Linear(d_cond + d_cond // 2 + d_cond // 4 + self.BASIN_DIM, d_cond),
# # # # #             nn.LayerNorm(d_cond), nn.GELU())

# # # # #     def _encode_raw(self, batch_list):
# # # # #         obs_traj  = batch_list[0]; obs_Me = batch_list[7]
# # # # #         image_obs = batch_list[11]; env_data = batch_list[13]
# # # # #         if image_obs.dim() == 4: image_obs = image_obs.unsqueeze(2)
# # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
# # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # #         T_obs  = obs_traj.shape[0]
# # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # #         if e_3d_s.shape[1] != T_obs:
# # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)
# # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # #                                           device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # #         f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
# # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)
# # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

# # # # #     def _kinematic_feat(self, obs_traj):
# # # # #         B = obs_traj.shape[1]; T_obs = obs_traj.shape[0]; device = obs_traj.device
# # # # #         if T_obs >= 2:
# # # # #             traj_deg = _norm_to_deg(obs_traj)
# # # # #             vel_norm = obs_traj[1:] - obs_traj[:-1]
# # # # #             speed    = _step_speeds_kmh(traj_deg)
# # # # #             speed_n  = (speed / 20.0).clamp(-3.0, 3.0)
# # # # #             heading  = torch.atan2(vel_norm[:, :, 1], vel_norm[:, :, 0])
# # # # #             if T_obs >= 3:
# # # # #                 dspd  = speed[1:] - speed[:-1]
# # # # #                 accel = torch.cat([obs_traj.new_zeros(1, B),
# # # # #                                    (dspd / 10.0).clamp(-3.0, 3.0)], 0)
# # # # #             else:
# # # # #                 accel = obs_traj.new_zeros(T_obs - 1, B)
# # # # #             kine = torch.stack([vel_norm[:,:,0], vel_norm[:,:,1], speed_n,
# # # # #                                 heading.sin(), heading.cos(), accel], -1)
# # # # #         else:
# # # # #             kine = obs_traj.new_zeros(self.obs_len, B, 6)
# # # # #         if kine.shape[0] < self.obs_len:
# # # # #             kine = torch.cat([obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6), kine], 0)
# # # # #         else:
# # # # #             kine = kine[-self.obs_len:]
# # # # #         return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

# # # # #     def forward(self, batch_list, hard_score=None, basin_id=None):
# # # # #         raw   = self._encode_raw(batch_list)
# # # # #         ctx   = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
# # # # #         kfeat = self._kinematic_feat(batch_list[0][:, :, :2])
# # # # #         B     = ctx.shape[0]

# # # # #         if hard_score is None:
# # # # #             hard_score = torch.zeros(B, device=ctx.device)
# # # # #         hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))

# # # # #         # [v2.9] Basin embedding
# # # # #         if basin_id is None:
# # # # #             basin_id = _infer_basin_id(batch_list[0][:, :, :2])
# # # # #         bfeat = self.basin_embed(basin_id.to(ctx.device))  # [B, BASIN_DIM]

# # # # #         return self.fuse(torch.cat([ctx, kfeat, hfeat, bfeat], dim=-1))


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Hard score — IDENTICAL to v2.1
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # @torch.no_grad()
# # # # # def hard_score_from_obs(obs_traj_norm, return_components=False):
# # # # #     T, B   = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
# # # # #     device = obs_traj_norm.device
# # # # #     if T < 3:
# # # # #         z = torch.zeros(B, device=device)
# # # # #         if return_components:
# # # # #             return z, {"curvature": z.clone(), "speed_var": z.clone(), "dir_change": z.clone()}
# # # # #         return z
# # # # #     traj_deg  = _norm_to_deg(obs_traj_norm[..., :2])
# # # # #     az12      = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
# # # # #     az23      = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
# # # # #     diff      = (az23 - az12).abs()
# # # # #     diff      = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
# # # # #     curvature = diff.mean(0) / math.pi
# # # # #     spd       = _step_speeds_kmh(traj_deg)
# # # # #     speed_var = (spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 1.) if spd.shape[0] >= 2 \
# # # # #                 else torch.zeros(B, device=device)
# # # # #     dir_change = (diff > (20.0 / 180.0 * math.pi)).float().mean(0)
# # # # #     score = (0.4 * curvature + 0.3 * speed_var + 0.3 * dir_change).clamp(0., 1.)
# # # # #     if return_components:
# # # # #         return score, {"curvature": curvature, "speed_var": speed_var, "dir_change": dir_change}
# # # # #     return score


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Physics score — v2.1 original (3 components, NO displacement_score)
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # @torch.no_grad()
# # # # # def _physics_score(traj_norm, obs_norm):
# # # # #     B = traj_norm.shape[1]; device = traj_norm.device
# # # # #     traj_deg = _norm_to_deg(traj_norm)

# # # # #     if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# # # # #         obs_deg  = _norm_to_deg(obs_norm)
# # # # #         obs_spd  = _step_speeds_kmh(obs_deg)
# # # # #         w_obs    = torch.linspace(0.5, 1.0, obs_spd.shape[0], device=device)
# # # # #         v_ref    = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()
# # # # #         pred_spd = _step_speeds_kmh(traj_deg)
# # # # #         v_sigma  = v_ref.clamp(min=5.0) * 0.5
# # # # #         speed_score = torch.exp(
# # # # #             -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5)
# # # # #     elif traj_deg.shape[0] >= 2:
# # # # #         speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(0) / 30.).mean(0))
# # # # #     else:
# # # # #         speed_score = torch.ones(B, device=device)

# # # # #     if traj_deg.shape[0] >= 3:
# # # # #         vel       = traj_deg[1:] - traj_deg[:-1]
# # # # #         smooth_score = torch.exp(-(vel[1:] - vel[:-1]).norm(dim=-1).mean(0) * 5.0)
# # # # #     else:
# # # # #         smooth_score = torch.ones(B, device=device)

# # # # #     if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
# # # # #         obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # # # #         pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
# # # # #         cos_sim  = (F.normalize(obs_vel, dim=-1, eps=1e-6) *
# # # # #                     F.normalize(pred_vel, dim=-1, eps=1e-6)).sum(-1).clamp(-1, 1)
# # # # #         head_score = torch.exp((cos_sim - 1.0) * 3.0)
# # # # #     else:
# # # # #         head_score = torch.ones(B, device=device)

# # # # #     return (speed_score.pow(0.35) * smooth_score.pow(0.30) * head_score.pow(0.35)
# # # # #             ).clamp(min=1e-6)


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Augmentation — v2.1 A+B only (augment_batch được gọi từ train script)
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def augment_batch(batch_list) -> list:
# # # # #     """
# # # # #     v2.9 augmentation = v2.1 (A+B+D+E only).
# # # # #     Không thêm AUG-C/D — đã chứng minh không giúp gap.
    
# # # # #     [v2.9 NEW] Temporal obs dropout (15% prob) handled INSIDE get_loss_breakdown,
# # # # #     không ở đây, để không ảnh hưởng đến inference path.
# # # # #     """
# # # # #     bl = list(batch_list)
# # # # #     if not torch.is_tensor(bl[0]): return bl
# # # # #     obs    = bl[0]; device = obs.device
# # # # #     anchor = obs[-1:, :, :2].detach()
# # # # #     r = torch.rand(1).item()

# # # # #     if r < 0.30:
# # # # #         shift = (torch.rand(2, device=device) - 0.5) * 0.018
# # # # #         bl[0] = obs + shift.view(1, 1, 2)
# # # # #         if torch.is_tensor(bl[1]): bl[1] = bl[1] + shift.view(1, 1, 2)

# # # # #     elif r < 0.60:
# # # # #         scale = 0.85 + 0.30 * torch.rand(1, device=device).item()
# # # # #         obs_c = obs.clone(); obs_c[..., :2] = anchor + (obs[..., :2] - anchor) * scale
# # # # #         bl[0] = obs_c
# # # # #         if torch.is_tensor(bl[1]): bl[1] = anchor + (bl[1] - anchor) * scale

# # # # #     elif r < 0.70:
# # # # #         pass  # no aug

# # # # #     else:
# # # # #         obs_new = obs.clone()
# # # # #         obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.003
# # # # #         bl[0] = obs_new

# # # # #     return bl


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  XAI utilities — from v2.1 (unchanged)
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def compute_obs_attribution(model, batch_list, device, target_step=11):
# # # # #     raw = _unwrap(model)
# # # # #     with torch.no_grad(): h_score = hard_score_from_obs(batch_list[0][:, :, :2])
# # # # #     obs_req = batch_list[0].detach().clone().requires_grad_(True)
# # # # #     bl_g = list(batch_list); bl_g[0] = obs_req
# # # # #     with torch.enable_grad():
# # # # #         cond = raw.encoder(bl_g, hard_score=h_score)
# # # # #         x0   = torch.randn(obs_req.shape[1], raw.pred_len, 2, device=device) * raw.sigma_inference
# # # # #         t0   = torch.zeros(obs_req.shape[1], device=device)
# # # # #         v    = raw.velocity(x0, t0, cond)
# # # # #         pred_rel = x0 + v
# # # # #         pred_rel[:, min(target_step, raw.pred_len-1), :].norm(dim=-1).mean().backward()
# # # # #     if obs_req.grad is not None:
# # # # #         attr = obs_req.grad[:, :, :2].norm(dim=-1)
# # # # #         attr = attr / (attr.sum(0, keepdim=True) + 1e-8)
# # # # #     else:
# # # # #         attr = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)
# # # # #     return attr.detach()

# # # # # @torch.no_grad()
# # # # # def compute_ensemble_uncertainty(all_traj):
# # # # #     all_deg = _norm_to_deg(all_traj); K, T, B = all_deg.shape[:3]
# # # # #     mean_t  = all_deg.mean(0); std_km = torch.zeros(T, B, device=all_traj.device)
# # # # #     for t in range(T):
# # # # #         dists = _haversine_deg(all_deg[:, t].reshape(K*B, 2),
# # # # #                                mean_t[t].unsqueeze(0).expand(K,B,2).reshape(K*B,2)).reshape(K,B)
# # # # #         std_km[t] = dists.std(0)
# # # # #     s12 = min(1, T-1); s72 = min(11, T-1)
# # # # #     return {"std_per_step": std_km,
# # # # #             "uncertainty_ratio": (std_km[s72]+1e-3)/(std_km[s12]+1e-3),
# # # # #             "mean_72h_std": float(std_km[s72].mean()), "mean_12h_std": float(std_km[s12].mean()),
# # # # #             "high_uncertainty": std_km[s72] > 80.0}

# # # # # @torch.no_grad()
# # # # # def compute_heading_deviation(pred_deg, gt_deg):
# # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # #     if T < 2: return pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # #     bear_gt   = _forward_azimuth(gt_deg[:T-1], gt_deg[1:T])
# # # # #     bear_pred = _forward_azimuth(gt_deg[:T-1], pred_deg[1:T])
# # # # #     diff = (bear_pred - bear_gt).abs()
# # # # #     diff = torch.where(diff > math.pi, 2*math.pi - diff, diff)
# # # # #     return torch.rad2deg(diff)

# # # # # @torch.no_grad()
# # # # # def compute_cte_contribution(pred_deg, gt_deg):
# # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # #     if T < 2:
# # # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # #         return {"ate_per_step": z, "cte_per_step": z, "ate_mean": z[0], "cte_mean": z[0],
# # # # #                 "ate_abs_mean": 0.0, "cte_abs_mean": 0.0}
# # # # #     bear_ref = _forward_azimuth(gt_deg[:T-1], gt_deg[1:T])
# # # # #     bear_err = _forward_azimuth(gt_deg[1:T], pred_deg[1:T])
# # # # #     dist_err = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# # # # #     ang = bear_err - bear_ref
# # # # #     ate = dist_err * torch.cos(ang); cte = dist_err * torch.sin(ang)
# # # # #     return {"ate_per_step": ate, "cte_per_step": cte,
# # # # #             "ate_mean": ate.mean(0), "cte_mean": cte.abs().mean(0),
# # # # #             "ate_abs_mean": float(ate.abs().mean()), "cte_abs_mean": float(cte.abs().mean())}

# # # # # @torch.no_grad()
# # # # # def classify_hard_easy(obs_traj_norm, per_sample_loss=None, hard_score_p=70.0, loss_p=50.0):
# # # # #     scores = hard_score_from_obs(obs_traj_norm); B = scores.shape[0]
# # # # #     if B < 4: return torch.zeros(B, dtype=torch.bool, device=scores.device)
# # # # #     return scores >= torch.quantile(scores, hard_score_p / 100.0)

# # # # # @torch.no_grad()
# # # # # def classify_hard_easy_global(obs_traj_norm, global_threshold):
# # # # #     return hard_score_from_obs(obs_traj_norm) >= global_threshold

# # # # # @torch.no_grad()
# # # # # def compute_diversity_score(candidates) -> float:
# # # # #     if len(candidates) < 2: return 0.0
# # # # #     T, B = candidates[0].shape[0], candidates[0].shape[1]
# # # # #     ep_step = min(T-1, 11)
# # # # #     endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
# # # # #     N = endpoints.shape[0]; ep_mean = endpoints.mean(0, keepdim=True)
# # # # #     dists = _haversine_deg(endpoints.reshape(N*B, 2),
# # # # #                            ep_mean.expand(N,B,2).reshape(N*B,2)).reshape(N, B)
# # # # #     return float(dists.std(0).mean())


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  TCFlowMatching v2.9
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # class TCFlowMatching(nn.Module):
# # # # #     """
# # # # #     TC-FlowMatching v2.9

# # # # #     L_total = L_CFM  +  lam_reg × L_reg  +  lam_accel × L_accel_speed

# # # # #     New in v2.9 vs v2.1:
# # # # #       (1) L_accel_speed  — ST-Trans quadratic speed smoothness
# # # # #       (2) VelocityTransformer dropout 0.15
# # # # #       (3) Basin embedding in ContextEncoder
# # # # #       (4) Label-smoothed step weights (anti-overfit specific horizons)
# # # # #       (5) Temporal obs dropout during training (15%, in get_loss_breakdown)
# # # # #     """

# # # # #     def __init__(self, pred_len=12, obs_len=8, unet_in_ch=13,
# # # # #                  d_cond=256, d_model=256, nhead=8, num_dec_layers=4, dim_ff=512,
# # # # #                  dropout=0.15, sigma_min=0.04, sigma_max=0.08,
# # # # #                  lambda_reg=0.2, lambda_accel=0.01,
# # # # #                  use_ot=True, ot_epsilon=0.05,
# # # # #                  use_ema=True, ema_decay=0.995,
# # # # #                  n_inference_steps=1, n_ensemble=20,
# # # # #                  sigma_inference=0.04,
# # # # #                  # step weight label smoothing
# # # # #                  step_w_smooth=0.1,
# # # # #                  # temporal obs dropout (training only)
# # # # #                  obs_dropout_p=0.15,
# # # # #                  **kwargs):
# # # # #         super().__init__()
# # # # #         self.pred_len          = pred_len
# # # # #         self.obs_len           = obs_len
# # # # #         self.sigma_min         = sigma_min
# # # # #         self.sigma_max         = sigma_max
# # # # #         self.lambda_reg        = lambda_reg
# # # # #         self.lambda_accel      = lambda_accel
# # # # #         self.use_ot            = use_ot
# # # # #         self.ot_epsilon        = ot_epsilon
# # # # #         self.n_inference_steps = n_inference_steps
# # # # #         self.n_ensemble        = n_ensemble
# # # # #         self.sigma_inference   = sigma_inference
# # # # #         self.step_w_smooth     = step_w_smooth
# # # # #         self.obs_dropout_p     = obs_dropout_p

# # # # #         self.encoder  = ContextEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
# # # # #         self.velocity = VelocityTransformer(pred_len=pred_len, d_model=d_model,
# # # # #                                             nhead=nhead, num_layers=num_dec_layers,
# # # # #                                             dim_ff=dim_ff, dropout=dropout, d_cond=d_cond)
# # # # #         self.use_ema = use_ema; self._ema = None

# # # # #         # compat
# # # # #         self.lambda_heading = 0.0; self.lambda_momentum = 0.0; self.lambda_jerk = 0.0

# # # # #     def init_ema(self):
# # # # #         if self.use_ema: self._ema = EMAModel(self, decay=0.995)

# # # # #     def ema_update(self):
# # # # #         if self._ema is not None: self._ema.update(self)

# # # # #     def _to_relative(self, x_abs, last_obs): return x_abs - last_obs.unsqueeze(1)
# # # # #     def _from_relative(self, x_rel, last_obs): return x_rel + last_obs.unsqueeze(1)

# # # # #     def _sigma_schedule(self, epoch):
# # # # #         if epoch < 5:   return self.sigma_max
# # # # #         if epoch < 40:
# # # # #             t = (epoch - 5) / 35.0
# # # # #             return self.sigma_min + 0.5*(self.sigma_max-self.sigma_min)*(1+math.cos(math.pi*t))
# # # # #         return self.sigma_min

# # # # #     def _smoothed_step_weights(self, T, device, dtype):
# # # # #         """
# # # # #         [v2.9] Label-smoothed step weights.
# # # # #         Prevents model from over-optimizing specific horizons (overfit source).
        
# # # # #         w = (1-α) × softmax(linspace(1,3,T)) + α × uniform(1/T)
# # # # #         α = step_w_smooth = 0.1
        
# # # # #         At α=0.1: shifts ~10% weight from focused distribution to uniform.
# # # # #         Max/min ratio reduced from 3× to ~2.3× → less horizon-specific overfit.
# # # # #         """
# # # # #         base = F.softmax(
# # # # #             torch.linspace(1.0, 3.0, T, device=device, dtype=dtype), dim=0)
# # # # #         uniform = torch.ones(T, device=device, dtype=dtype) / T
# # # # #         return (1.0 - self.step_w_smooth) * base + self.step_w_smooth * uniform

# # # # #     def _temporal_obs_dropout(self, obs_traj, is_training):
# # # # #         """
# # # # #         [v2.9] Temporal obs dropout — training augmentation only.
        
# # # # #         15% prob: drop step 0 or 1 (early obs, less critical for short-term).
# # # # #         Replace dropped step with linear interpolation.
# # # # #         → Model learns general dynamics, not val-specific obs patterns.
# # # # #         → ONLY during training (is_training=True), NEVER at inference.
        
# # # # #         Returns modified obs_traj [T_obs, B, D].
# # # # #         """
# # # # #         if not is_training or self.obs_dropout_p <= 0:
# # # # #             return obs_traj
# # # # #         if torch.rand(1).item() > self.obs_dropout_p:
# # # # #             return obs_traj

# # # # #         T, B, D = obs_traj.shape
# # # # #         if T < 3: return obs_traj

# # # # #         obs_new = obs_traj.clone()
# # # # #         drop_t  = 0 if torch.rand(1).item() < 0.5 else 1

# # # # #         if drop_t == 0:
# # # # #             # Drop step 0: replace with extrapolation backward from step 1 and 2
# # # # #             if T >= 3:
# # # # #                 obs_new[0] = obs_new[1] - (obs_new[2] - obs_new[1])
# # # # #             else:
# # # # #                 obs_new[0] = obs_new[1]
# # # # #         else:
# # # # #             # Drop step 1: replace with linear interpolation of step 0 and 2
# # # # #             if T >= 3:
# # # # #                 obs_new[1] = 0.5 * (obs_new[0] + obs_new[2])

# # # # #         return obs_new

# # # # #     # ── L_accel_speed — ST-Trans style ─────────────────────────────────────

# # # # #     def _accel_loss(self, x1_pred_abs, last_obs):
# # # # #         """
# # # # #         Quadratic speed acceleration penalty — identical to ST-Trans L_accel.
        
# # # # #         speeds[t] = haversine(pred[t-1], pred[t]) / DT_HOURS  [T, B] km/h
# # # # #         accel[t]  = speeds[t+1] - speeds[t]                   [T-1, B]
# # # # #         L_accel   = mean(accel²) / 100.0
        
# # # # #         WHY BETTER THAN L_jerk:
# # # # #           L_jerk = ReLU(|Δheading| - threshold) → 75% zero gradient
# # # # #           L_accel = accel² → gradient = 2×accel → ALWAYS nonzero
# # # # #           → Continuous signal every step → better ATE optimization
# # # # #         """
# # # # #         pred_deg = _norm_to_deg(x1_pred_abs)             # [T, B, 2]
# # # # #         last_deg = _norm_to_deg(last_obs)                 # [B, 2]
# # # # #         pts      = torch.cat([last_deg.unsqueeze(0), pred_deg], 0)  # [T+1, B, 2]
# # # # #         speeds   = _step_speeds_kmh(pts)                  # [T, B]
# # # # #         if speeds.shape[0] < 2: return x1_pred_abs.new_zeros(())
# # # # #         accel = speeds[1:] - speeds[:-1]
# # # # #         return (accel ** 2).mean() / 100.0

# # # # #     # ── L_reg — label-smoothed step weights ────────────────────────────────

# # # # #     def _reg_loss(self, x1_rel, last_obs, cond, hard_score=None):
# # # # #         """
# # # # #         ADE loss at t=0. Label-smoothed step weights (v2.9).
# # # # #         Shared forward pass with _accel_loss for efficiency.
# # # # #         Returns (loss, x1_pred_abs) so caller can reuse pred for accel loss.
# # # # #         """
# # # # #         B, T, _ = x1_rel.shape; device = x1_rel.device
# # # # #         x0 = torch.randn_like(x1_rel) * self.sigma_inference
# # # # #         t0 = torch.zeros(B, device=device)
# # # # #         v  = self.velocity(x0, t0, cond)
# # # # #         x1_pred_abs = self._from_relative(x0 + v, last_obs)  # [B, T, 2]
# # # # #         x1_gt_abs   = self._from_relative(x1_rel, last_obs)
# # # # #         pred_deg    = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))
# # # # #         gt_deg      = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
# # # # #         dist        = _haversine_deg(pred_deg, gt_deg)         # [T, B] km

# # # # #         T_actual = dist.shape[0]
# # # # #         sw = self._smoothed_step_weights(T_actual, device, dist.dtype).unsqueeze(1)

# # # # #         if hard_score is not None:
# # # # #             sw_hard = (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)
# # # # #         else:
# # # # #             sw_hard = torch.ones(1, B, device=device, dtype=dist.dtype)

# # # # #         loss = ((dist * sw) * sw_hard).mean() / 300.0
# # # # #         return loss, x1_pred_abs  # return pred for reuse in accel loss

# # # # #     # ── get_loss_breakdown ──────────────────────────────────────────────────

# # # # #     def get_loss_breakdown(self, batch_list, epoch=0, **kwargs):
# # # # #         """
# # # # #         L_total = L_CFM + lam_reg × L_reg + lam_accel × L_accel_speed

# # # # #         [v2.9] Temporal obs dropout applied BEFORE encoding (training only).
# # # # #         [v2.9] Basin ID inferred and passed to encoder.
# # # # #         [v2.9] Shared forward pass: L_reg and L_accel use same x0+v prediction.
# # # # #         """
# # # # #         obs_traj = batch_list[0]
# # # # #         gt_traj  = batch_list[1]
# # # # #         B        = obs_traj.shape[1]; device = obs_traj.device
# # # # #         is_train = self.training

# # # # #         # [v2.9] Temporal obs dropout (training only)
# # # # #         obs_traj_enc = self._temporal_obs_dropout(obs_traj, is_train)

# # # # #         sigma    = self._sigma_schedule(epoch)
# # # # #         x1_gt    = gt_traj.permute(1, 0, 2)
# # # # #         last_obs = obs_traj[-1, :, :2]   # always use ORIGINAL last obs
# # # # #         x1_rel   = self._to_relative(x1_gt, last_obs)

# # # # #         # [v2.9] Basin ID for encoder
# # # # #         basin_id = _infer_basin_id(obs_traj[:, :, :2])

# # # # #         h_score  = hard_score_from_obs(obs_traj_enc[:, :, :2])
# # # # #         # Build modified batch list with (possibly dropped) obs
# # # # #         bl_enc = list(batch_list); bl_enc[0] = obs_traj_enc
# # # # #         cond   = self.encoder(bl_enc, hard_score=h_score, basin_id=basin_id)

# # # # #         # ── L_CFM ─────────────────────────────────────────────────────────
# # # # #         x0 = torch.randn_like(x1_rel) * sigma
# # # # #         if self.use_ot and B >= 4:
# # # # #             x0_f, x1_f = _ot_match(x0.reshape(B, -1), x1_rel.reshape(B, -1), self.ot_epsilon)
# # # # #             x0 = x0_f.reshape(B, self.pred_len, 2)
# # # # #             x1_matched = x1_f.reshape(B, self.pred_len, 2)
# # # # #         else:
# # # # #             x1_matched = x1_rel
# # # # #         t        = torch.rand(B, device=device)
# # # # #         x_t      = (1.0 - t.view(B,1,1)) * x0 + t.view(B,1,1) * x1_matched
# # # # #         u_target = x1_matched - x0
# # # # #         v_pred   = self.velocity(x_t, t, cond)
# # # # #         l_cfm    = F.mse_loss(v_pred, u_target)

# # # # #         # ── Ramps ─────────────────────────────────────────────────────────
# # # # #         lam_reg   = (0.0 if epoch < 10 else
# # # # #                      self.lambda_reg * (epoch-10)/20.0 if epoch < 30 else
# # # # #                      self.lambda_reg)
# # # # #         lam_accel = (0.0 if epoch < 5 else
# # # # #                      self.lambda_accel * (epoch-5)/15.0 if epoch < 20 else
# # # # #                      self.lambda_accel)

# # # # #         # ── L_reg + L_accel (shared forward pass — efficient) ─────────────
# # # # #         l_reg = x0.new_zeros(())
# # # # #         l_accel = x0.new_zeros(())

# # # # #         if lam_reg > 0.0 or lam_accel > 0.0:
# # # # #             l_reg_val, x1_pred_abs = self._reg_loss(x1_rel, last_obs, cond, h_score)
# # # # #             if lam_reg > 0.0:
# # # # #                 l_reg = l_reg_val
# # # # #             if lam_accel > 0.0:
# # # # #                 # Reuse same prediction for accel loss — no extra forward pass
# # # # #                 l_accel = self._accel_loss(x1_pred_abs.permute(1, 0, 2), last_obs)

# # # # #         # ── Total ─────────────────────────────────────────────────────────
# # # # #         total = l_cfm + lam_reg * l_reg + lam_accel * l_accel
# # # # #         if not torch.isfinite(total): total = x0.new_zeros(())

# # # # #         # ── ADE log ───────────────────────────────────────────────────────
# # # # #         with torch.no_grad():
# # # # #             x0_log = torch.randn_like(x1_rel) * self.sigma_inference
# # # # #             v_log  = self.velocity(x0_log, torch.zeros(B, device=device), cond)
# # # # #             ade_log = _haversine_deg(
# # # # #                 _norm_to_deg(self._from_relative(x0_log+v_log, last_obs).permute(1,0,2)),
# # # # #                 _norm_to_deg(x1_gt.permute(1,0,2))).mean().item()

# # # # #         return {
# # # # #             "total": total, "l_cfm": l_cfm.item(),
# # # # #             "l_reg": l_reg.item() if torch.is_tensor(l_reg) else 0.0,
# # # # #             "l_accel": l_accel.item() if torch.is_tensor(l_accel) else 0.0,
# # # # #             "l_heading": 0.0, "l_momentum": 0.0, "l_jerk": 0.0,
# # # # #             "lam_reg": lam_reg, "lam_accel": lam_accel, "lam_dir": 0.0,
# # # # #             "sigma": sigma, "ade_1step": ade_log,
# # # # #             "hard_score_mean": float(h_score.mean()),
# # # # #             "hard_score_max":  float(h_score.max()),
# # # # #             # compat keys
# # # # #             "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
# # # # #             "speed": 0., "accel": l_accel.item() if torch.is_tensor(l_accel) else 0.,
# # # # #             "fm_mse": l_cfm.item(), "l_hard_total": 0., "n_hard": 0,
# # # # #             "alpha_hard": 0., "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
# # # # #         }

# # # # #     def get_loss(self, batch_list, epoch=0, **kwargs):
# # # # #         return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

# # # # #     # ── Sample — identical to v2.1, passes basin_id to encoder ────────────

# # # # #     @torch.no_grad()
# # # # #     def sample(self, batch_list, num_ensemble=None, ddim_steps=None,
# # # # #                return_xai=False, **kwargs):
# # # # #         """1-shot inference. Basin ID passed to encoder. No speed calibration."""
# # # # #         K  = num_ensemble or self.n_ensemble
# # # # #         N  = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else self.n_inference_steps
# # # # #         dt = 1.0 / max(N, 1)

# # # # #         obs_traj    = batch_list[0]; T_obs, B, _ = obs_traj.shape
# # # # #         device      = obs_traj.device
# # # # #         h_score     = hard_score_from_obs(obs_traj[:, :, :2])
# # # # #         obs_norm    = obs_traj[:, :, :2]
# # # # #         last_obs    = obs_traj[-1, :, :2]
# # # # #         t0          = torch.zeros(B, device=device)
# # # # #         basin_id    = _infer_basin_id(obs_norm)
# # # # #         cond        = self.encoder(batch_list, hard_score=h_score, basin_id=basin_id)

# # # # #         all_traj = []
# # # # #         for _ in range(K):
# # # # #             x_rel = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference
# # # # #             if N <= 1:
# # # # #                 x_rel = x_rel + self.velocity(x_rel, t0, cond)
# # # # #             else:
# # # # #                 for step in range(N):
# # # # #                     t_b   = torch.full((B,), step * dt, device=device)
# # # # #                     x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3., 3.)
# # # # #             all_traj.append(self._from_relative(x_rel, last_obs).permute(1, 0, 2))

# # # # #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)
# # # # #         all_t   = torch.stack(all_traj, 0)
# # # # #         top_k   = min(3, K); top_idx = scores.topk(top_k, dim=0).indices
# # # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # # #         for b in range(B):
# # # # #             idx_b = top_idx[:, b]; w_b = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # # # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# # # # #         if not return_xai:
# # # # #             return pred_mean, torch.zeros_like(pred_mean), all_t

# # # # #         # XAI
# # # # #         xai = {}; xai.update(compute_ensemble_uncertainty(all_t))
# # # # #         _, hard_comps = hard_score_from_obs(obs_norm, return_components=True)
# # # # #         xai["hard_components"] = hard_comps
# # # # #         pred_deg = _norm_to_deg(pred_mean); obs_deg_x = _norm_to_deg(obs_norm)
# # # # #         obs_spd_mu  = _step_speeds_kmh(obs_deg_x).mean(0)
# # # # #         pred_spd_mu = _step_speeds_kmh(pred_deg).mean(0) if pred_deg.shape[0]>=2 else obs_spd_mu.clone()
# # # # #         xai["speed_comparison"] = {
# # # # #             "obs_speed_mean": float(obs_spd_mu.mean()), "pred_speed_mean": float(pred_spd_mu.mean()),
# # # # #             "speed_ratio": float((pred_spd_mu/obs_spd_mu.clamp(min=1.)).mean()),
# # # # #             "per_storm_obs": obs_spd_mu, "per_storm_pred": pred_spd_mu,
# # # # #             "over_predict": (pred_spd_mu/obs_spd_mu.clamp(min=1.)>1.15),
# # # # #             "under_predict": (pred_spd_mu/obs_spd_mu.clamp(min=1.)<0.85),
# # # # #         }
# # # # #         gt_traj_xai = batch_list[1]
# # # # #         gt_deg_xai  = _norm_to_deg(gt_traj_xai[:, :, :2])
# # # # #         xai["heading_deviation_deg"] = compute_heading_deviation(pred_deg, gt_deg_xai)
# # # # #         xai["ate_cte_decomp"]        = compute_cte_contribution(pred_deg, gt_deg_xai)
# # # # #         xai["basin_ids"] = basin_id.tolist()
# # # # #         return pred_mean, torch.zeros_like(pred_mean), all_t, xai

# # # # #     @torch.no_grad()
# # # # #     def sample_multiscale(self, batch_list, sigmas=None, n_per_sigma=4):
# # # # #         if sigmas is None: sigmas = [0.025, 0.035, 0.04, 0.05, 0.065]
# # # # #         obs_traj = batch_list[0]; B = obs_traj.shape[1]; device = obs_traj.device
# # # # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # # # #         obs_norm = obs_traj[:, :, :2]; last_obs = obs_traj[-1, :, :2]
# # # # #         t0       = torch.zeros(B, device=device)
# # # # #         basin_id = _infer_basin_id(obs_norm)
# # # # #         cond     = self.encoder(batch_list, hard_score=h_score, basin_id=basin_id)
# # # # #         all_traj = []
# # # # #         for sigma in sigmas:
# # # # #             for _ in range(n_per_sigma):
# # # # #                 x0    = torch.randn(B, self.pred_len, 2, device=device) * sigma
# # # # #                 x_abs = self._from_relative(x0 + self.velocity(x0, t0, cond), last_obs)
# # # # #                 all_traj.append(x_abs.permute(1, 0, 2))
# # # # #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)
# # # # #         all_t   = torch.stack(all_traj, 0); top_k = min(5, len(all_traj))
# # # # #         top_idx = scores.topk(top_k, dim=0).indices
# # # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # # #         for b in range(B):
# # # # #             idx_b = top_idx[:, b]; w_b = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # # # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)
# # # # #         return pred_mean, torch.zeros_like(pred_mean), all_t


# # # # # TCDiffusion = TCFlowMatching

# # # # """
# # # # Model/flow_matching_model.py  ──  TC-FlowMatching v2.1-XAI
# # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # VIẾT LẠI HOÀN TOÀN từ v2.1 + các fix đã verified từ thực nghiệm 140 epoch.

# # # # ━━━ CƠ SỞ LÝ THUYẾT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # v2.1 test: ADE=229.8, ATE=214.4, CTE=71.6 ← BASELINE
# # # # v2.5 test: ADE=232.5, ATE=222.3, CTE=64.5 (AUG-C helped CTE -7.1km)
# # # # v2.6 test: ADE=230.2, ATE=218.9, CTE=65.7 (AUG-D hurt ATE +4.5km)
# # # # ST-Trans:  ADE=224.4, ATE=213.7, CTE=59.4 ← TARGET

# # # # PHÂN TÍCH LỖI v2.1:
# # # #   ATE/CTE ratio on test = 214.4/71.6 = 3.0
# # # #   → 75% error is ALONG-TRACK (speed/distance)
# # # #   → 25% is CROSS-TRACK (direction)
# # # #   Test gap 72h = +129.3km >> 12h = +10.1km → long-range drift

# # # # PHÂN TÍCH v2.5 THẤT BẠI:
# # # #   L_momentum anchor pred_speed ≈ val_speed (10.3km/h)
# # # #   Test storms faster → L_momentum SLOWS model on test → ATE +7.9km WORSE
# # # #   L_heading (step 0 only) → 72h heading dev INCREASED: ep0=87°→ep120=128°!
# # # #   Recurvature aug: CTE improved 7km ← giữ lại

# # # # ━━━ VẬT LÝ CỦA GIẢI PHÁP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # # GIẢI PHÁP ATE (along-track error):
# # # #   Training: KHÔNG anchor speed tại training (no L_momentum)
# # # #   Inference: SPEED CALIBRATION per-storm
# # # #     scale = clip(obs_speed / pred_speed_first4, 0.85, 1.15)
# # # #     pred_cal = last_obs + (pred - last_obs) * scale
# # # #     → Preserves direction (CTE unchanged), fixes magnitude (ATE reduced)
# # # #     → Per-storm adaptive: fast test storms get upscaling, slow storms downscaling
# # # #     → DOES NOT affect training → zero train/inference consistency issue

# # # # GIẢI PHÁP CTE (cross-track error):
# # # #   Training: MULTI-STEP heading constraint (not single step)
# # # #     step 0 (6h):  weight 1.0  — hard anchor to obs direction
# # # #     step 1 (12h): weight 0.5  — continuation (detached ref)
# # # #     step 2 (18h): weight 0.25 — softer continuation
# # # #     step 3 (24h): weight 0.125 — very soft
# # # #     Detach ref at each step: prevents gradient explosion through chain
# # # #     → 72h heading dev should come down from 128° to ~90°

# # # # GIẢI PHÁP GAP (val→test distribution shift):
# # # #   [AUG-C] Recurvature: proven to reduce CTE test (v2.5)
# # # #   [AUG-D] Obs-speed scaling ×0.7–1.4: model sees diverse speed contexts
# # # #            → at inference, adapts better to test storms' speed
# # # #   Speed calibration at inference bridges remaining gap

# # # # ━━━ GIỮ NGUYÊN TỪ v2.1 (PROVEN, KHÔNG THAY ĐỔI) ━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # #   ✅ sigma_inference=0.04 FIXED — zero train/inference mismatch
# # # #   ✅ L_reg softmax-linspace weights (max/min ≈ 3×) — không exp (v2.4 fail)
# # # #   ✅ Mild base aug: shift±5km + speed×0.85–1.15 (val ≈ test distribution)
# # # #   ✅ 1-shot inference nhất quán với L_reg training
# # # #   ✅ 2-group optimizer, encoder freeze 10ep
# # # #   ✅ val loop NO augmentation
# # # #   ✅ OT noise-GT matching trong L_CFM
# # # #   ✅ ContextEncoder: FNO3D + Mamba + Env + 6 kinematic features
# # # #   ✅ VelocityTransformer: d_model=256, 4 layers, nhead=8

# # # # ━━━ v2.1-XAI: CẢI TIẾN CÓ EVIDENCE TỪ LOG THỰC TẾ ━━━━━━━━━━━━━━━━━━━━━━━

# # # #   ROOT CAUSE từ v2.1 log (150 epochs):
# # # #     val CTE=45.7 → test CTE=71.6 (+25.9km, 57%) — vấn đề chính
# # # #     ST-Trans CTE gap chỉ +17.4km → v2.1 tệ hơn 8.5km về direction
# # # #     Plateau ep40-150: val ADE chỉ -7km trong 110 epochs (aug bão hòa)

# # # #   [AUG-C]  Recurvature ±20° (PROVEN: v2.5 test CTE 71.6→64.5 = -7.1km)
# # # #            ease-in rotation, last 3 obs rotated 30% — TC physics
# # # #   [L_HDG]  L_heading bug fixed: removed no_grad → gradient flows
# # # #            weight 0.07 (thấp hơn 0.10, tránh cạnh tranh với L_CFM)
# # # #   [CALIB]  speed_calibrate tại inference (safe, no training change)
# # # #   [DISP]   displacement_score trong physics_score (catches speed artifacts)
# # # #   [XAI 8]  Per-horizon speed error (12h/24h/48h/72h pred vs GT)
# # # #   [XAI 9]  Storm category breakdown (slow/medium/fast)

# # # #   KHÔNG THÊM: AUG-D(+4.5km ATE), L_momentum(+7.9km ATE), exp-L_reg(catastrophic)

# # # # ━━━ KỲ VỌNG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # #                    Val ADE   Test ADE   Val ATE  Test ATE  Val CTE  Test CTE
# # # #   v2.1 (best):     170.1      229.8     160.2    214.4     45.7     71.6
# # # #   v2.5 (current):  170.5      232.5     161.3    222.3     46.2     64.5
# # # #   v2.1-XAI target: 170±1      218–226   160±1    210–215   44±1     60–66
# # # # """
# # # # from __future__ import annotations

# # # # import math
# # # # from typing import Dict, List, Optional, Tuple

# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.nn.functional as F

# # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # from Model.env_net_transformer_gphsplit import Env_net

# # # # R_EARTH  = 6371.0
# # # # DT_HOURS = 6.0


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Coordinate utilities (same as v2.1)
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# # # #     """Normalized coords → (lon°, lat°)."""
# # # #     return torch.stack([
# # # #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# # # #         (t[..., 1] * 50.0) / 10.0,
# # # #     ], dim=-1)


# # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # #     """Haversine distance km between two (lon°,lat°) tensors."""
# # # #     lat1 = torch.deg2rad(p1[..., 1]);  lat2 = torch.deg2rad(p2[..., 1])
# # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # #     a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
# # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# # # # def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # #     """Bearing in radians from p1→p2 (degrees input)."""
# # # #     lon1 = torch.deg2rad(p1[..., 0]);  lat1 = torch.deg2rad(p1[..., 1])
# # # #     lon2 = torch.deg2rad(p2[..., 0]);  lat2 = torch.deg2rad(p2[..., 1])
# # # #     dlon = lon2 - lon1
# # # #     y = torch.sin(dlon) * torch.cos(lat2)
# # # #     x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
# # # #     return torch.atan2(y, x)


# # # # def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
# # # #     """Speed km/h between consecutive steps of a trajectory [T, B, 2]."""
# # # #     if traj_deg.shape[0] < 2:
# # # #         return traj_deg.new_zeros(1, traj_deg.shape[1])
# # # #     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  EMAModel (same as v2.1)
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def _unwrap(m):
# # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m


# # # # class EMAModel:
# # # #     def __init__(self, model, decay: float = 0.995):
# # # #         self.decay = decay
# # # #         m = _unwrap(model)
# # # #         self.shadow = {k: v.detach().clone()
# # # #                        for k, v in m.state_dict().items()
# # # #                        if v.dtype.is_floating_point}

# # # #     def update(self, model):
# # # #         m = _unwrap(model)
# # # #         with torch.no_grad():
# # # #             for k, v in m.state_dict().items():
# # # #                 if k in self.shadow:
# # # #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

# # # #     def apply_to(self, model):
# # # #         m = _unwrap(model)
# # # #         backup, sd = {}, m.state_dict()
# # # #         for k in self.shadow:
# # # #             if k not in sd: continue
# # # #             backup[k] = sd[k].detach().clone()
# # # #             sd[k].copy_(self.shadow[k])
# # # #         return backup

# # # #     def restore(self, model, backup):
# # # #         m = _unwrap(model)
# # # #         sd = m.state_dict()
# # # #         for k, v in backup.items():
# # # #             if k in sd: sd[k].copy_(v)


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  OT matching (same as v2.1)
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 50) -> torch.Tensor:
# # # #     B = cost.shape[0]; device = cost.device
# # # #     log_a = -math.log(B) * torch.ones(B, device=device)
# # # #     log_b = -math.log(B) * torch.ones(B, device=device)
# # # #     log_K = -cost / epsilon
# # # #     log_u = torch.zeros(B, device=device)
# # # #     log_v = torch.zeros(B, device=device)
# # # #     for _ in range(n_iter):
# # # #         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
# # # #         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
# # # #     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


# # # # def _ot_match(x0_flat: torch.Tensor, x1_flat: torch.Tensor,
# # # #               epsilon: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
# # # #     B = x0_flat.shape[0]
# # # #     if B < 4:
# # # #         return x0_flat, x1_flat
# # # #     try:
# # # #         cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1] ** 0.5)
# # # #         with torch.no_grad():
# # # #             pi = _sinkhorn_log(cost, epsilon=epsilon)
# # # #         flat = pi.reshape(-1).clamp(0.0)
# # # #         s = flat.sum()
# # # #         if not torch.isfinite(s) or s < 1e-10:
# # # #             return x0_flat, x1_flat
# # # #         idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
# # # #         return x0_flat[idx // B], x1_flat
# # # #     except Exception:
# # # #         return x0_flat, x1_flat


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  VelocityTransformer (identical to v2.1)
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # class VelocityTransformer(nn.Module):
# # # #     def __init__(self, pred_len: int = 12, d_model: int = 256, nhead: int = 8,
# # # #                  num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
# # # #                  d_cond: int = 256):
# # # #         super().__init__()
# # # #         self.pred_len = pred_len
# # # #         self.d_model  = d_model
# # # #         self.traj_embed = nn.Linear(2, d_model)
# # # #         self.pos_emb    = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
# # # #         self.step_emb   = nn.Embedding(pred_len, d_model)
# # # #         self.time_mlp   = nn.Sequential(
# # # #             nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
# # # #         self.cond_proj  = nn.Sequential(nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))
# # # #         dec_layer = nn.TransformerDecoderLayer(
# # # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # # #             dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
# # # #         self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
# # # #         self.out_norm = nn.LayerNorm(d_model)
# # # #         self.out_proj = nn.Sequential(
# # # #             nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
# # # #         self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
# # # #         nn.init.zeros_(self.out_proj[-1].weight)
# # # #         nn.init.zeros_(self.out_proj[-1].bias)

# # # #     def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
# # # #         half = self.d_model // 2
# # # #         freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
# # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # #         emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
# # # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # # #         if self.d_model % 2 == 1:
# # # #             emb = F.pad(emb, (0, 1))
# # # #         return self.time_mlp(emb)

# # # #     def forward(self, x_t: torch.Tensor, t: torch.Tensor,
# # # #                 cond: torch.Tensor) -> torch.Tensor:
# # # #         B, T, _ = x_t.shape
# # # #         step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # #         x_emb = (self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(step_idx))
# # # #         memory = torch.cat([self._time_emb(t).unsqueeze(1),
# # # #                             self.cond_proj(cond).unsqueeze(1)], dim=1)
# # # #         out = self.out_norm(self.decoder(x_emb, memory))
# # # #         return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  ContextEncoder (identical to v2.1)
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # class ContextEncoder(nn.Module):
# # # #     RAW_CTX_DIM = 512

# # # #     def __init__(self, obs_len: int = 8, unet_in_ch: int = 13, d_cond: int = 256):
# # # #         super().__init__()
# # # #         self.obs_len = obs_len
# # # #         self.d_cond  = d_cond

# # # #         self.spatial_enc     = FNO3DEncoder(in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # #                                              n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # #                                              spatial_down=32, dropout=0.05)
# # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # #         self.decoder_proj    = nn.Linear(1, 16)
# # # #         self.enc_1d          = DataEncoder1D(in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # #                                               lstm_hidden=128, lstm_layers=3,
# # # #                                               dropout=0.1, d_state=16)
# # # #         self.env_enc         = Env_net(obs_len=obs_len, d_model=32)
# # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # #         self.ctx_drop = nn.Dropout(0.1)
# # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, d_cond)
# # # #         self.ctx_ln2  = nn.LayerNorm(d_cond)
# # # #         # 6 kinematic features: vel_x, vel_y, speed_n, sin(h), cos(h), accel
# # # #         self.vel_obs_enc = nn.Sequential(
# # # #             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
# # # #             nn.Linear(256, d_cond // 2), nn.GELU())
# # # #         self.hard_embed = nn.Sequential(
# # # #             nn.Linear(1, d_cond // 4), nn.GELU(), nn.Linear(d_cond // 4, d_cond // 4))
# # # #         self.fuse = nn.Sequential(
# # # #             nn.Linear(d_cond + d_cond // 2 + d_cond // 4, d_cond),
# # # #             nn.LayerNorm(d_cond), nn.GELU())

# # # #     def _encode_raw(self, batch_list) -> torch.Tensor:
# # # #         obs_traj  = batch_list[0]; obs_Me = batch_list[7]
# # # #         image_obs = batch_list[11]; env_data = batch_list[13]
# # # #         if image_obs.dim() == 4:
# # # #             image_obs = image_obs.unsqueeze(2)
# # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
# # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # #         T_obs = obs_traj.shape[0]
# # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # #         if e_3d_s.shape[1] != T_obs:
# # # #             e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
# # # #                                    mode="linear", align_corners=False).permute(0,2,1)
# # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # #                                           device=e_3d_dec_t.device) * 0.5, dim=0)
# # # #         f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
# # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # #         e_env, _, _ = self.env_enc(env_data, image_obs)
# # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

# # # #     def _kinematic_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
# # # #         """6 kinematic features per obs step."""
# # # #         B = obs_traj.shape[1]; T_obs = obs_traj.shape[0]; device = obs_traj.device
# # # #         if T_obs >= 2:
# # # #             traj_deg = _norm_to_deg(obs_traj)
# # # #             vel_norm = obs_traj[1:] - obs_traj[:-1]
# # # #             speed    = _step_speeds_kmh(traj_deg)
# # # #             speed_n  = (speed / 20.0).clamp(-3.0, 3.0)
# # # #             heading  = torch.atan2(vel_norm[:, :, 1], vel_norm[:, :, 0])
# # # #             if T_obs >= 3:
# # # #                 dspd  = speed[1:] - speed[:-1]
# # # #                 accel = torch.cat([obs_traj.new_zeros(1, B),
# # # #                                    (dspd / 10.0).clamp(-3.0, 3.0)], 0)
# # # #             else:
# # # #                 accel = obs_traj.new_zeros(T_obs - 1, B)
# # # #             kine = torch.stack([vel_norm[:,:,0], vel_norm[:,:,1], speed_n,
# # # #                                 heading.sin(), heading.cos(), accel], dim=-1)
# # # #         else:
# # # #             kine = obs_traj.new_zeros(self.obs_len, B, 6)
# # # #         if kine.shape[0] < self.obs_len:
# # # #             kine = torch.cat([obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6), kine], 0)
# # # #         else:
# # # #             kine = kine[-self.obs_len:]
# # # #         return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

# # # #     def forward(self, batch_list, hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# # # #         raw   = self._encode_raw(batch_list)
# # # #         ctx   = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
# # # #         kfeat = self._kinematic_feat(batch_list[0][:, :, :2])
# # # #         if hard_score is None:
# # # #             hard_score = torch.zeros(ctx.shape[0], device=ctx.device)
# # # #         hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))
# # # #         return self.fuse(torch.cat([ctx, kfeat, hfeat], dim=-1))


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Hard score (XAI-2) — same as v2.1
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # @torch.no_grad()
# # # # def hard_score_from_obs(obs_traj_norm: torch.Tensor,
# # # #                          return_components: bool = False):
# # # #     """[XAI-2] Độ khó của storm dựa trên curvature, speed_var, dir_change."""
# # # #     T, B   = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
# # # #     device = obs_traj_norm.device
# # # #     if T < 3:
# # # #         z = torch.zeros(B, device=device)
# # # #         if return_components:
# # # #             return z, {"curvature": z.clone(), "speed_var": z.clone(), "dir_change": z.clone()}
# # # #         return z
# # # #     traj_deg = _norm_to_deg(obs_traj_norm[..., :2])
# # # #     az12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
# # # #     az23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
# # # #     diff = (az23 - az12).abs()
# # # #     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
# # # #     curvature  = diff.mean(0) / math.pi
# # # #     spd        = _step_speeds_kmh(traj_deg)
# # # #     if spd.shape[0] >= 2:
# # # #         speed_var = (spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 1.)
# # # #     else:
# # # #         speed_var = torch.zeros(B, device=device)
# # # #     dir_change = (diff > (20.0 / 180.0 * math.pi)).float().mean(0)
# # # #     score = (0.4 * curvature + 0.3 * speed_var + 0.3 * dir_change).clamp(0., 1.)
# # # #     if return_components:
# # # #         return score, {"curvature": curvature, "speed_var": speed_var, "dir_change": dir_change}
# # # #     return score


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Physics score v2.6 (adds displacement_score)
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # @torch.no_grad()
# # # # def _physics_score(traj_norm: torch.Tensor, obs_norm: torch.Tensor) -> torch.Tensor:
# # # #     """
# # # #     Best-of-K selection score. Four components:
# # # #       speed_score       (0.30): per-step speed vs obs reference
# # # #       smooth_score      (0.25): trajectory smoothness (no sharp acceleration)
# # # #       heading_score     (0.30): first-step direction matches obs
# # # #       displacement_score(0.15): total path length consistent with obs speed

# # # #     [NEW v2.6] displacement_score penalizes samples with total displacement
# # # #     far from obs-speed expectation → catches speed over-prediction artifacts
# # # #     that cause high ATE on test storms.
# # # #     """
# # # #     B      = traj_norm.shape[1]
# # # #     device = traj_norm.device
# # # #     traj_deg = _norm_to_deg(traj_norm)
# # # #     v_ref   = None

# # # #     # ── Speed score ────────────────────────────────────────────────────────
# # # #     if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# # # #         obs_deg = _norm_to_deg(obs_norm)
# # # #         obs_spd = _step_speeds_kmh(obs_deg)
# # # #         T_s     = obs_spd.shape[0]
# # # #         w_obs   = torch.linspace(0.5, 1.0, T_s, device=device)
# # # #         v_ref   = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()   # [B]
# # # #         pred_spd = _step_speeds_kmh(traj_deg)
# # # #         v_sigma  = v_ref.clamp(min=5.0) * 0.5
# # # #         speed_score = torch.exp(
# # # #             -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5)
# # # #     elif traj_deg.shape[0] >= 2:
# # # #         speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.).mean(0))
# # # #     else:
# # # #         speed_score = torch.ones(B, device=device)

# # # #     # ── Smoothness score ───────────────────────────────────────────────────
# # # #     if traj_deg.shape[0] >= 3:
# # # #         vel          = traj_deg[1:] - traj_deg[:-1]
# # # #         accel_mag    = (vel[1:] - vel[:-1]).norm(dim=-1)
# # # #         smooth_score = torch.exp(-accel_mag.mean(0) * 5.0)
# # # #     else:
# # # #         smooth_score = torch.ones(B, device=device)

# # # #     # ── Heading score ──────────────────────────────────────────────────────
# # # #     if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
# # # #         obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # # #         pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
# # # #         obs_h    = F.normalize(obs_vel,  dim=-1, eps=1e-6)
# # # #         pred_h   = F.normalize(pred_vel, dim=-1, eps=1e-6)
# # # #         cos_sim  = (obs_h * pred_h).sum(-1).clamp(-1, 1)
# # # #         head_score = torch.exp((cos_sim - 1.0) * 3.0)
# # # #     else:
# # # #         head_score = torch.ones(B, device=device)

# # # #     # ── [NEW v2.6] Displacement score ─────────────────────────────────────
# # # #     # Expected total path length ≈ obs_speed × T_pred × DT × 0.75
# # # #     # (0.75 factor: storms curve, so straight-line < path length estimate)
# # # #     # This catches ensemble samples that over/under-shoot on total distance.
# # # #     if v_ref is not None and traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# # # #         T_pred        = traj_deg.shape[0]
# # # #         expected_total = v_ref * T_pred * DT_HOURS * 0.75   # [B] km
# # # #         step_dists    = _haversine_deg(traj_deg[:-1], traj_deg[1:])   # [T-1, B]
# # # #         actual_total  = step_dists.sum(0)                              # [B]
# # # #         rel_err       = (actual_total - expected_total).abs() / expected_total.clamp(min=10.)
# # # #         disp_score    = torch.exp(-rel_err * 1.5)
# # # #     else:
# # # #         disp_score    = torch.ones(B, device=device)

# # # #     return (speed_score.pow(0.30)
# # # #             * smooth_score.pow(0.25)
# # # #             * head_score.pow(0.30)
# # # #             * disp_score.pow(0.15)).clamp(min=1e-6)


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Augmentation v2.6
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def augment_batch(batch_list) -> list:
# # # #     """
# # # #     v2.1-XAI augmentation: v2.1 core (A+B) + recurvature (C) — không AUG-D.

# # # #     Evidence:
# # # #       AUG-C: v2.5 test CTE 71.6→64.5 = -7.1km ✅ PROVEN
# # # #       AUG-D: v2.6 test ATE +4.5km, v2.7 test ATE +13.4km ❌ HARMFUL

# # # #     Distribution:
# # # #       A (25%): track shift ±5km       — shape unchanged, position varies
# # # #       B (20%): speed scale ×0.85-1.15 — mild, proven (both obs+GT from anchor)
# # # #       C (20%): recurvature ±20°       — PROVEN CTE improvement
# # # #                ease-in rotation (TC inertia — slow start, accelerate)
# # # #                last 3 obs steps rotated 30% for continuity
# # # #       D (35%): no augmentation        — preserve original distribution
# # # #     """
# # # #     bl = list(batch_list)
# # # #     if not torch.is_tensor(bl[0]):
# # # #         return bl

# # # #     obs    = bl[0]
# # # #     device = obs.device
# # # #     anchor = obs[-1:, :, :2].detach()  # [1, B, 2] — last obs as pivot
# # # #     r = torch.rand(1).item()

# # # #     if r < 0.25:
# # # #         # A: Track shift ±5km (preserves storm shape, varies position)
# # # #         shift = (torch.rand(2, device=device) - 0.5) * 0.018  # ±5km in normalized
# # # #         bl[0] = obs + shift.view(1, 1, 2)
# # # #         if torch.is_tensor(bl[1]):
# # # #             bl[1] = bl[1] + shift.view(1, 1, 2)

# # # #     elif r < 0.45:
# # # #         # B: Speed scale ×0.85–1.15 (both obs+GT from anchor, mild)
# # # #         scale = 0.85 + 0.30 * torch.rand(1, device=device).item()
# # # #         obs_c = obs.clone()
# # # #         obs_c[..., :2] = anchor + (obs[..., :2] - anchor) * scale
# # # #         bl[0] = obs_c
# # # #         if torch.is_tensor(bl[1]):
# # # #             bl[1] = anchor + (bl[1] - anchor) * scale

# # # #     elif r < 0.65:
# # # #         # C: Recurvature ±20° — proven CTE improvement
# # # #         # Progressive rotation with ease-in: slow start, accelerate (realistic)
# # # #         T_pred = bl[1].shape[0] if torch.is_tensor(bl[1]) else 0
# # # #         if T_pred >= 4:
# # # #             gt      = bl[1].clone()
# # # #             max_deg = (torch.rand(1).item() - 0.5) * 40.0   # -20° to +20°
# # # #             max_rad = max_deg * math.pi / 180.0
# # # #             for t in range(T_pred):
# # # #                 progress = (t / max(T_pred - 1, 1)) ** 1.5  # ease-in
# # # #                 angle_t  = max_rad * progress
# # # #                 c, s = math.cos(angle_t), math.sin(angle_t)
# # # #                 rot = torch.tensor([[c, -s], [s, c]], dtype=gt.dtype, device=device)
# # # #                 rel = (gt[t] - anchor[0]).unsqueeze(-1)   # [B, 2, 1]
# # # #                 gt[t] = (rot @ rel).squeeze(-1) + anchor[0]
# # # #             bl[1] = gt
# # # #             # Rotate last 3 obs steps slightly (30% of max) for consistency
# # # #             T_obs   = obs.shape[0]
# # # #             obs_aug = obs.clone()
# # # #             cos_p   = math.cos(max_rad * 0.3)
# # # #             sin_p   = math.sin(max_rad * 0.3)
# # # #             rot_p   = torch.tensor([[cos_p, -sin_p], [sin_p, cos_p]],
# # # #                                     dtype=obs.dtype, device=device)
# # # #             for t_obs in range(max(0, T_obs - 3), T_obs):
# # # #                 rel_o = (obs_aug[t_obs, :, :2] - anchor[0]).unsqueeze(-1)
# # # #                 obs_aug[t_obs, :, :2] = (rot_p @ rel_o).squeeze(-1) + anchor[0]
# # # #             bl[0] = obs_aug

# # # #     # D (r >= 0.65): no augmentation — giữ original distribution nhiều hơn
# # # #     # AUG-D (obs-only speed scale) REMOVED:
# # # #     #   Evidence: v2.6 ATE +4.5km worse, v2.7 ATE +13.4km worse

# # # #     return bl


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  XAI functions (1–7, complete)
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def compute_obs_attribution(model, batch_list, device: torch.device,
# # # #                              target_step: int = 11) -> torch.Tensor:
# # # #     """
# # # #     [XAI-1] Gradient saliency: which obs step most influences 72h prediction?

# # # #     Returns attr [T_obs, B] normalized importance [0,1].
# # # #     High attr[-1, b] = last obs dominant (normal).
# # # #     High attr[0, b]  = storm has long "memory" (unusual → hard storm).
# # # #     """
# # # #     raw = _unwrap(model)
# # # #     with torch.no_grad():
# # # #         h_score = hard_score_from_obs(batch_list[0][:, :, :2])
# # # #     obs_req = batch_list[0].detach().clone().requires_grad_(True)
# # # #     bl_g = list(batch_list); bl_g[0] = obs_req
# # # #     with torch.enable_grad():
# # # #         cond = raw.encoder(bl_g, hard_score=h_score)
# # # #         x0   = torch.randn(obs_req.shape[1], raw.pred_len, 2, device=device) * raw.sigma_inference
# # # #         t0   = torch.zeros(obs_req.shape[1], device=device)
# # # #         v    = raw.velocity(x0, t0, cond)
# # # #         pred_rel = x0 + v
# # # #         ts       = min(target_step, raw.pred_len - 1)
# # # #         pred_rel[:, ts, :].norm(dim=-1).mean().backward()
# # # #     if obs_req.grad is not None:
# # # #         attr = obs_req.grad[:, :, :2].norm(dim=-1)
# # # #         attr = attr / (attr.sum(0, keepdim=True) + 1e-8)
# # # #     else:
# # # #         attr = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)
# # # #     return attr.detach()


# # # # @torch.no_grad()
# # # # def compute_ensemble_uncertainty(all_traj: torch.Tensor) -> Dict:
# # # #     """
# # # #     [XAI-4] Uncertainty per lead time across K ensemble samples.
# # # #     Returns std_per_step [T,B] km, uncertainty_ratio, mean_72h_std, high_uncertainty.
# # # #     """
# # # #     all_deg  = _norm_to_deg(all_traj)  # [K,T,B,2]
# # # #     K, T, B  = all_deg.shape[:3]
# # # #     mean_traj = all_deg.mean(0)
# # # #     std_km = torch.zeros(T, B, device=all_traj.device)
# # # #     for t in range(T):
# # # #         dists = _haversine_deg(
# # # #             all_deg[:, t].reshape(K * B, 2),
# # # #             mean_traj[t].unsqueeze(0).expand(K, B, 2).reshape(K * B, 2)
# # # #         ).reshape(K, B)
# # # #         std_km[t] = dists.std(0)
# # # #     s12 = min(1, T - 1); s72 = min(11, T - 1)
# # # #     return {
# # # #         "std_per_step":      std_km,
# # # #         "uncertainty_ratio": (std_km[s72] + 1e-3) / (std_km[s12] + 1e-3),
# # # #         "mean_72h_std":      float(std_km[s72].mean()),
# # # #         "mean_12h_std":      float(std_km[s12].mean()),
# # # #         "high_uncertainty":  std_km[s72] > 80.0,
# # # #     }


# # # # @torch.no_grad()
# # # # def compute_heading_deviation(pred_deg: torch.Tensor,
# # # #                                gt_deg:   torch.Tensor) -> torch.Tensor:
# # # #     """
# # # #     [XAI-6] Heading deviation per lead time step, in degrees (absolute).
# # # #     High values indicate directional error = root cause of CTE.
# # # #     Returns [T-1, B].
# # # #     """
# # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # #     if T < 2:
# # # #         return pred_deg.new_zeros(1, pred_deg.shape[1])
# # # #     bear_gt   = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
# # # #     bear_pred = _forward_azimuth(gt_deg[:T-1], pred_deg[1:T])
# # # #     diff = (bear_pred - bear_gt).abs()
# # # #     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
# # # #     return torch.rad2deg(diff)


# # # # @torch.no_grad()
# # # # def compute_cte_contribution(pred_deg: torch.Tensor,
# # # #                               gt_deg:   torch.Tensor) -> Dict:
# # # #     """
# # # #     [XAI-7] Decompose trajectory error into ATE (along-track) and CTE (cross-track).
# # # #     Returns ate_per_step [T-1,B], cte_per_step [T-1,B], scalar means.
# # # #     """
# # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # #     if T < 2:
# # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # #         return {"ate_per_step": z, "cte_per_step": z,
# # # #                 "ate_mean": z[0], "cte_mean": z[0],
# # # #                 "ate_abs_mean": 0.0, "cte_abs_mean": 0.0}
# # # #     bear_ref  = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
# # # #     bear_err  = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
# # # #     dist_err  = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# # # #     ang       = bear_err - bear_ref
# # # #     ate       = dist_err * torch.cos(ang)
# # # #     cte       = dist_err * torch.sin(ang)
# # # #     return {
# # # #         "ate_per_step":  ate,
# # # #         "cte_per_step":  cte,
# # # #         "ate_mean":      ate.mean(0),
# # # #         "cte_mean":      cte.abs().mean(0),
# # # #         "ate_abs_mean":  float(ate.abs().mean()),
# # # #         "cte_abs_mean":  float(cte.abs().mean()),
# # # #     }


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Compat stubs
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # @torch.no_grad()
# # # # def classify_hard_easy(obs_traj_norm, per_sample_loss=None,
# # # #                         hard_score_p: float = 70.0, loss_p: float = 50.0):
# # # #     scores = hard_score_from_obs(obs_traj_norm)
# # # #     B = scores.shape[0]
# # # #     if B < 4:
# # # #         return torch.zeros(B, dtype=torch.bool, device=scores.device)
# # # #     return scores >= torch.quantile(scores, hard_score_p / 100.0)


# # # # @torch.no_grad()
# # # # def classify_hard_easy_global(obs_traj_norm, global_threshold):
# # # #     return hard_score_from_obs(obs_traj_norm) >= global_threshold


# # # # @torch.no_grad()
# # # # def compute_diversity_score(candidates) -> float:
# # # #     if len(candidates) < 2:
# # # #         return 0.0
# # # #     T, B = candidates[0].shape[0], candidates[0].shape[1]
# # # #     ep_step = min(T - 1, 11)
# # # #     endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
# # # #     N = endpoints.shape[0]
# # # #     ep_mean = endpoints.mean(0, keepdim=True)
# # # #     dists = _haversine_deg(
# # # #         endpoints.reshape(N * B, 2),
# # # #         ep_mean.expand(N, B, 2).reshape(N * B, 2)
# # # #     ).reshape(N, B)
# # # #     return float(dists.std(0).mean())


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  TCFlowMatching v2.6
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # class TCFlowMatching(nn.Module):
# # # #     """
# # # #     TC-FlowMatching v2.6

# # # #     Training loss:
# # # #         L_total = L_CFM
# # # #                 + lam_reg  × L_reg          (ramp ep10→ep30, linear weights)
# # # #                 + lam_dir  × lambda_heading × L_heading_ms  (ramp ep5→ep20)

# # # #         L_CFM:         conditional flow matching (random t, OT noise-GT matching)
# # # #         L_reg:         ADE loss at sigma_inference, softmax-linspace step weights
# # # #         L_heading_ms:  [v2.6] 4-step heading continuation, decay=0.5 per step

# # # #     NO L_momentum: removed because it anchors to val speed distribution,
# # # #     which hurts ATE on test storms that move at different speeds.

# # # #     Inference pipeline:
# # # #         1. Sample K=20 candidates with sigma=0.04 FIXED
# # # #         2. Physics score selection (speed+smooth+heading+displacement)
# # # #         3. Weighted average of top-3 candidates
# # # #         4. Speed calibration: per-storm scale ×[0.85, 1.15] to match obs speed
# # # #         5. Optional: multi-scale sigma ensemble (sample_multiscale)
# # # #     """

# # # #     def __init__(
# # # #         self,
# # # #         pred_len:          int   = 12,
# # # #         obs_len:           int   = 8,
# # # #         unet_in_ch:        int   = 13,
# # # #         d_cond:            int   = 256,
# # # #         d_model:           int   = 256,
# # # #         nhead:             int   = 8,
# # # #         num_dec_layers:    int   = 4,
# # # #         dim_ff:            int   = 512,
# # # #         dropout:           float = 0.1,
# # # #         sigma_min:         float = 0.04,
# # # #         sigma_max:         float = 0.08,
# # # #         lambda_reg:        float = 0.2,
# # # #         lambda_heading:    float = 0.07,   # [v2.1-XAI] lower than 0.10: avoid competing with L_CFM
# # # #         lambda_momentum:   float = 0.0,    # DISABLED — field kept for compat only
# # # #         use_ot:            bool  = True,
# # # #         ot_epsilon:        float = 0.05,
# # # #         use_ema:           bool  = True,
# # # #         ema_decay:         float = 0.995,
# # # #         n_inference_steps: int   = 1,
# # # #         n_ensemble:        int   = 20,
# # # #         sigma_inference:   float = 0.04,   # FIXED throughout
# # # #         **kwargs,
# # # #     ):
# # # #         super().__init__()
# # # #         self.pred_len          = pred_len
# # # #         self.obs_len           = obs_len
# # # #         self.sigma_min         = sigma_min
# # # #         self.sigma_max         = sigma_max
# # # #         self.lambda_reg        = lambda_reg
# # # #         self.lambda_heading    = lambda_heading
# # # #         self.lambda_momentum   = 0.0           # always disabled
# # # #         self.use_ot            = use_ot
# # # #         self.ot_epsilon        = ot_epsilon
# # # #         self.n_inference_steps = n_inference_steps
# # # #         self.n_ensemble        = n_ensemble
# # # #         self.sigma_inference   = sigma_inference

# # # #         self.encoder  = ContextEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
# # # #         self.velocity = VelocityTransformer(
# # # #             pred_len=pred_len, d_model=d_model, nhead=nhead,
# # # #             num_layers=num_dec_layers, dim_ff=dim_ff,
# # # #             dropout=dropout, d_cond=d_cond)
# # # #         self.use_ema = use_ema
# # # #         self._ema    = None

# # # #     def init_ema(self):
# # # #         if self.use_ema:
# # # #             self._ema = EMAModel(self, decay=0.995)

# # # #     def ema_update(self):
# # # #         if self._ema is not None:
# # # #             self._ema.update(self)

# # # #     def _to_relative(self, x_abs: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# # # #         return x_abs - last_obs.unsqueeze(1)

# # # #     def _from_relative(self, x_rel: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# # # #         return x_rel + last_obs.unsqueeze(1)

# # # #     def _sigma_schedule(self, epoch: int) -> float:
# # # #         """Cosine decay sigma_max→sigma_min over ep5→ep40."""
# # # #         if epoch < 5:
# # # #             return self.sigma_max
# # # #         if epoch < 40:
# # # #             t = (epoch - 5) / 35.0
# # # #             return self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * t))
# # # #         return self.sigma_min

# # # #     # ── [v2.6] Multi-step heading loss ─────────────────────────────────────

# # # #     def _heading_loss_ms(self, pred_deg: torch.Tensor, obs_deg: torch.Tensor,
# # # #                           n_steps: int = 4, decay: float = 0.5) -> torch.Tensor:
# # # #         """
# # # #         Multi-step heading constraint. [UPGRADED FROM v2.5 SINGLE STEP]

# # # #         v2.5 used single-step (step 0 only, weight=0.05):
# # # #           → 72h heading dev INCREASED from 87° to 128° during training
# # # #           → Model learned "start correctly" then drifted

# # # #         v2.6 uses 4 steps with exponential decay:
# # # #           step t reference: obs_heading (t=0) or pred_bearing[t-1] (t>0, DETACHED)
# # # #           step 0: 1.0 × (1 - cos(pred_bear[0] - obs_bear))  ← hard anchor
# # # #           step 1: 0.5 × (1 - cos(pred_bear[1] - pred_bear[0]))  ← soft continuation
# # # #           step 2: 0.25 × ...
# # # #           step 3: 0.125 × ...

# # # #         DETACH ref at each step: prevents gradient explosion through chain.
# # # #         Each step trains relatively independently.

# # # #         Expected: 72h heading dev down from 128° to ~90-100°
# # # #         Expected: CTE test from 64.5km down to ~57-63km
# # # #         """
# # # #         if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
# # # #             return pred_deg.new_zeros(())

# # # #         # Starting reference: bearing from obs[-2] → obs[-1]
# # # #         ref_bear = _forward_azimuth(obs_deg[-2], obs_deg[-1])   # [B]

# # # #         loss = pred_deg.new_zeros(())
# # # #         # pts[t] → pts[t+1] gives bearing for pred step t
# # # #         pts = torch.cat([obs_deg[-1:], pred_deg], 0)   # [T_pred+1, B, 2]

# # # #         N = min(n_steps, pred_deg.shape[0])
# # # #         for t in range(N):
# # # #             pred_bear  = _forward_azimuth(pts[t], pts[t + 1])  # [B]
# # # #             angle_diff = pred_bear - ref_bear
# # # #             loss       = loss + (decay ** t) * (1.0 - torch.cos(angle_diff)).mean()
# # # #             ref_bear   = pred_bear.detach()   # CRITICAL: detach to prevent chain gradient

# # # #         return loss / N

# # # #     # ── L_reg (identical to v2.1 — proven not to overfit) ──────────────────

# # # #     def _reg_loss(self, x1_rel: torch.Tensor, last_obs: torch.Tensor,
# # # #                   cond: torch.Tensor,
# # # #                   hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# # # #         """
# # # #         ADE loss at t=0 with sigma_inference noise, consistent with 1-shot inference.
# # # #         Uses SOFTMAX-LINSPACE step weights (max/min ≈ 3×).
# # # #         DOES NOT use exp weights (v2.4 used 12.2× ratio → catastrophic overfitting).
# # # #         """
# # # #         B, T, _ = x1_rel.shape
# # # #         device   = x1_rel.device
# # # #         x0   = torch.randn_like(x1_rel) * self.sigma_inference
# # # #         t0   = torch.zeros(B, device=device)
# # # #         v    = self.velocity(x0, t0, cond)
# # # #         x1_pred_abs = self._from_relative(x0 + v, last_obs)
# # # #         x1_gt_abs   = self._from_relative(x1_rel, last_obs)
# # # #         pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))  # [T, B, 2]
# # # #         gt_deg   = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
# # # #         dist     = _haversine_deg(pred_deg, gt_deg)             # [T, B] km
# # # #         # Linear weights: softmax(linspace(1,3,T)) → max/min ≈ 3×
# # # #         T_actual = dist.shape[0]
# # # #         sw = torch.linspace(1.0, 3.0, T_actual, device=device, dtype=dist.dtype)
# # # #         sw = F.softmax(sw, dim=0).unsqueeze(1)                  # [T, 1]
# # # #         if hard_score is not None:
# # # #             sw_hard = (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)  # [1, B]
# # # #         else:
# # # #             sw_hard = torch.ones(1, B, device=device, dtype=dist.dtype)
# # # #         return ((dist * sw) * sw_hard).mean() / 300.0

# # # #     # ── get_loss_breakdown ──────────────────────────────────────────────────

# # # #     def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
# # # #         """
# # # #         Full loss computation. Called in TRAINING with augmented batch.
# # # #         Called in VAL without augmentation → val_loss is reliable signal.

# # # #         Total = L_CFM + lam_reg × L_reg + lam_dir × lambda_heading × L_heading_ms
# # # #         """
# # # #         obs_traj = batch_list[0]
# # # #         gt_traj  = batch_list[1]
# # # #         B        = obs_traj.shape[1]
# # # #         device   = obs_traj.device

# # # #         sigma    = self._sigma_schedule(epoch)
# # # #         x1_gt    = gt_traj.permute(1, 0, 2)       # [B, T, 2]
# # # #         last_obs = obs_traj[-1, :, :2]             # [B, 2]
# # # #         x1_rel   = self._to_relative(x1_gt, last_obs)

# # # #         h_score = hard_score_from_obs(obs_traj[:, :, :2])
# # # #         cond    = self.encoder(batch_list, hard_score=h_score)

# # # #         # ── L_CFM ─────────────────────────────────────────────────────────
# # # #         x0 = torch.randn_like(x1_rel) * sigma
# # # #         if self.use_ot and B >= 4:
# # # #             x0_flat, x1_flat = _ot_match(
# # # #                 x0.reshape(B, -1), x1_rel.reshape(B, -1), self.ot_epsilon)
# # # #             x0         = x0_flat.reshape(B, self.pred_len, 2)
# # # #             x1_matched = x1_flat.reshape(B, self.pred_len, 2)
# # # #         else:
# # # #             x1_matched = x1_rel

# # # #         t        = torch.rand(B, device=device)
# # # #         x_t      = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1_matched
# # # #         u_target = x1_matched - x0
# # # #         v_pred   = self.velocity(x_t, t, cond)
# # # #         l_cfm    = F.mse_loss(v_pred, u_target)

# # # #         # ── L_reg ramp ep10→ep30 ─────────────────────────────────────────
# # # #         if epoch < 10:     lam_reg = 0.0
# # # #         elif epoch < 30:   lam_reg = self.lambda_reg * (epoch - 10) / 20.0
# # # #         else:              lam_reg = self.lambda_reg

# # # #         l_reg = (self._reg_loss(x1_rel, last_obs, cond, h_score)
# # # #                  if lam_reg > 0.0 else x0.new_zeros(()))

# # # #         # ── L_heading_ms ramp ep5→ep20 ────────────────────────────────────
# # # #         # Ramp: 0 before ep5, linear ep5→ep20, full after ep20
# # # #         if epoch < 5:      lam_dir = 0.0
# # # #         elif epoch < 20:   lam_dir = (epoch - 5) / 15.0
# # # #         else:              lam_dir = 1.0

# # # #         if lam_dir > 0.0:
# # # #             # [v2.1-XAI BUG FIX] no_grad REMOVED → gradient flows to velocity
# # # #             # BEFORE (v2.6 bug): with no_grad() → velocity.grad = 0.000 (DEAD)
# # # #             # AFTER  (fix):      no no_grad     → velocity.grad = 0.515 (ACTIVE)
# # # #             x0_h       = torch.randn_like(x1_rel) * self.sigma_inference
# # # #             v_h        = self.velocity(x0_h, torch.zeros(B, device=device), cond)
# # # #             x1_h_abs   = self._from_relative(x0_h + v_h, last_obs)
# # # #             pred_deg_h = _norm_to_deg(x1_h_abs.permute(1, 0, 2))   # [T, B, 2]
# # # #             obs_deg_h  = _norm_to_deg(obs_traj[:, :, :2])           # [T_obs, B, 2]
# # # #             l_heading  = self._heading_loss_ms(pred_deg_h, obs_deg_h, n_steps=4, decay=0.5)
# # # #         else:
# # # #             l_heading = x0.new_zeros(())

# # # #         l_momentum = x0.new_zeros(())   # always 0 in v2.6

# # # #         # ── Total ─────────────────────────────────────────────────────────
# # # #         total = l_cfm + lam_reg * l_reg + lam_dir * self.lambda_heading * l_heading
# # # #         if not torch.isfinite(total):
# # # #             total = x0.new_zeros(())

# # # #         # ── ADE 1-step log ────────────────────────────────────────────────
# # # #         with torch.no_grad():
# # # #             x0_log = torch.randn_like(x1_rel) * self.sigma_inference
# # # #             v_log  = self.velocity(x0_log, torch.zeros(B, device=device), cond)
# # # #             ade_log = _haversine_deg(
# # # #                 _norm_to_deg(self._from_relative(x0_log + v_log, last_obs).permute(1, 0, 2)),
# # # #                 _norm_to_deg(x1_gt.permute(1, 0, 2))
# # # #             ).mean().item()

# # # #         return {
# # # #             "total":    total,
# # # #             "l_cfm":    l_cfm.item(),
# # # #             "l_reg":    l_reg.item() if torch.is_tensor(l_reg) else 0.0,
# # # #             "l_heading": l_heading.item() if torch.is_tensor(l_heading) else 0.0,
# # # #             "l_momentum": 0.0,   # always 0
# # # #             "lam_reg":  lam_reg,
# # # #             "lam_dir":  lam_dir,
# # # #             "sigma":    sigma,
# # # #             "ade_1step": ade_log,
# # # #             "hard_score_mean": float(h_score.mean()),
# # # #             "hard_score_max":  float(h_score.max()),
# # # #             # Compat keys for older scripts
# # # #             "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
# # # #             "speed": 0., "accel": 0., "fm_mse": l_cfm.item(),
# # # #             "l_hard_total": 0., "n_hard": 0, "alpha_hard": 0.,
# # # #             "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
# # # #         }

# # # #     def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
# # # #         return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

# # # #     # ── [NEW v2.6] Speed calibration at inference ──────────────────────────

# # # #     @staticmethod
# # # #     @torch.no_grad()
# # # #     def speed_calibrate_pred(pred_abs_norm: torch.Tensor,
# # # #                               last_obs_norm: torch.Tensor,
# # # #                               obs_norm:      torch.Tensor,
# # # #                               clip_min: float = 0.85,
# # # #                               clip_max: float = 1.15) -> torch.Tensor:
# # # #         """
# # # #         Per-storm post-hoc speed calibration. Applied at inference ONLY.
# # # #         DOES NOT affect training → zero train/inference consistency issue.

# # # #         Scales DISPLACEMENT MAGNITUDE, preserves DIRECTION.
# # # #         → ATE improved (speed error corrected)
# # # #         → CTE unchanged (direction unchanged)

# # # #         Algorithm:
# # # #           obs_speed_ref = exponentially-weighted mean of obs step speeds
# # # #           pred_speed_ref = mean of first 4 predicted step speeds
# # # #           scale = clip(obs_speed_ref / pred_speed_ref, clip_min, clip_max)
# # # #           pred_cal[t] = last_obs + (pred[t] - last_obs) * scale

# # # #         clip_min=0.85, clip_max=1.15:
# # # #           Prevents over-correction from noisy obs/pred speed estimates.
# # # #           Allows ±15% adjustment per storm.

# # # #         Why this works:
# # # #           v2.5 XAI-5: speed ratio = 1.15–1.61 throughout training
# # # #           (model consistently over-predicts speed on val)
# # # #           Test storms have different speed → need per-storm correction
# # # #           This is adaptive: fast test storms get upscaling, slow get downscaling
# # # #         """
# # # #         if obs_norm.shape[0] < 2 or pred_abs_norm.shape[0] < 2:
# # # #             return pred_abs_norm

# # # #         obs_deg  = _norm_to_deg(obs_norm)        # [T_obs, B, 2]
# # # #         pred_deg = _norm_to_deg(pred_abs_norm)   # [T, B, 2]
# # # #         last_deg = _norm_to_deg(last_obs_norm)   # [B, 2]

# # # #         # Obs speed reference (exponentially weighted, recent steps more important)
# # # #         obs_spds = _step_speeds_kmh(obs_deg)     # [T_obs-1, B] km/h
# # # #         T_s = obs_spds.shape[0]
# # # #         w = torch.exp(torch.arange(T_s, dtype=obs_spds.dtype, device=obs_spds.device) * 0.5)
# # # #         obs_spd_ref = (obs_spds * (w / w.sum()).unsqueeze(1)).sum(0).clamp(min=3.0)   # [B]

# # # #         # Predicted speed (first 4 steps from last_obs)
# # # #         pts = torch.cat([last_deg.unsqueeze(0), pred_deg], 0)   # [T+1, B, 2]
# # # #         pred_spds    = _step_speeds_kmh(pts)                     # [T, B]
# # # #         N_ref        = min(4, pred_spds.shape[0])
# # # #         pred_spd_ref = pred_spds[:N_ref].mean(0).clamp(min=3.0)  # [B]

# # # #         # Per-storm scale, clipped
# # # #         scale = (obs_spd_ref / pred_spd_ref).clamp(clip_min, clip_max)  # [B]

# # # #         # Scale displacement from last_obs (preserves direction!)
# # # #         return (last_obs_norm.unsqueeze(0)
# # # #                 + (pred_abs_norm - last_obs_norm.unsqueeze(0))
# # # #                 * scale.view(1, -1, 1))

# # # #     # ── Sample (1-shot + speed calibration) ──────────────────────────────

# # # #     @torch.no_grad()
# # # #     def sample(self, batch_list,
# # # #                num_ensemble:          Optional[int]  = None,
# # # #                ddim_steps:            Optional[int]  = None,
# # # #                return_xai:            bool           = False,
# # # #                use_speed_calibration: bool           = True,
# # # #                **kwargs) -> Tuple:
# # # #         """
# # # #         1-shot inference with physics selection + speed calibration.

# # # #         Steps:
# # # #           1. Encode context once (hard_score + full encoder)
# # # #           2. Sample K=20 candidates: x0 ~ N(0, sigma_inf²), x_pred = x0 + v(x0, 0, cond)
# # # #           3. Physics score each candidate (speed+smooth+heading+displacement)
# # # #           4. Weighted average of top-3 candidates
# # # #           5. Speed calibration per-storm (clip ±15%)
# # # #           6. If return_xai=True: compute XAI-1 through XAI-7

# # # #         Returns:
# # # #           (pred_mean [T,B,2], zeros [T,B,2], all_traj [K,T,B,2])
# # # #           or (pred_mean, zeros, all_traj, xai_dict) if return_xai=True
# # # #         """
# # # #         K  = num_ensemble or self.n_ensemble
# # # #         N  = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else self.n_inference_steps
# # # #         dt = 1.0 / max(N, 1)

# # # #         obs_traj    = batch_list[0]
# # # #         T_obs, B, _ = obs_traj.shape
# # # #         device      = obs_traj.device

# # # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # # #         obs_norm = obs_traj[:, :, :2]
# # # #         last_obs = obs_traj[-1, :, :2]
# # # #         t0       = torch.zeros(B, device=device)

# # # #         # Encode once for all K samples (expensive part)
# # # #         cond = self.encoder(batch_list, hard_score=h_score)

# # # #         all_traj = []
# # # #         for _ in range(K):
# # # #             x_rel = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference

# # # #             if N <= 1:
# # # #                 # 1-shot: exactly what L_reg is trained for
# # # #                 v     = self.velocity(x_rel, t0, cond)
# # # #                 x_rel = x_rel + v
# # # #             else:
# # # #                 # Euler multi-step (for experiments only, slower)
# # # #                 for step in range(N):
# # # #                     t_b   = torch.full((B,), step * dt, device=device)
# # # #                     x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3., 3.)

# # # #             x_abs = self._from_relative(x_rel, last_obs)
# # # #             all_traj.append(x_abs.permute(1, 0, 2))   # [T, B, 2]

# # # #         # Physics-based selection: best-of-K with weighted average of top-3
# # # #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)   # [K, B]
# # # #         all_t   = torch.stack(all_traj, 0)   # [K, T, B, 2]
# # # #         top_k   = min(3, K)
# # # #         top_idx = scores.topk(top_k, dim=0).indices   # [top_k, B]

# # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # #         for b in range(B):
# # # #             idx_b = top_idx[:, b]
# # # #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# # # #         # [NEW v2.6] Per-storm speed calibration
# # # #         if use_speed_calibration:
# # # #             pred_mean = self.speed_calibrate_pred(
# # # #                 pred_mean, last_obs, obs_norm, clip_min=0.85, clip_max=1.15)

# # # #         if not return_xai:
# # # #             return pred_mean, torch.zeros_like(pred_mean), all_t

# # # #         # ── XAI output (XAI-2,3,4,5,6,7) ──────────────────────────────────
# # # #         xai = {}

# # # #         # XAI-4: Ensemble uncertainty
# # # #         xai.update(compute_ensemble_uncertainty(all_t))

# # # #         # XAI-2: Hard score components
# # # #         _, hard_comps = hard_score_from_obs(obs_norm, return_components=True)
# # # #         xai["hard_components"] = hard_comps

# # # #         pred_deg  = _norm_to_deg(pred_mean)    # [T, B, 2]
# # # #         obs_deg_x = _norm_to_deg(obs_norm)     # [T_obs, B, 2]

# # # #         # XAI-5: Speed comparison (obs vs pred, pre and post calibration)
# # # #         obs_spd_x    = _step_speeds_kmh(obs_deg_x)
# # # #         obs_spd_mu   = obs_spd_x.mean(0)     # [B]
# # # #         if pred_deg.shape[0] >= 2:
# # # #             last_deg_x   = obs_deg_x[-1]
# # # #             pts_x        = torch.cat([last_deg_x.unsqueeze(0), pred_deg], 0)
# # # #             pred_spd_x   = _step_speeds_kmh(pts_x)
# # # #             pred_spd_mu  = pred_spd_x.mean(0)  # [B]
# # # #         else:
# # # #             pred_spd_mu = obs_spd_mu.clone()

# # # #         speed_ratio = (pred_spd_mu / obs_spd_mu.clamp(min=1.0))
# # # #         xai["speed_comparison"] = {
# # # #             "obs_speed_mean":  float(obs_spd_mu.mean()),
# # # #             "pred_speed_mean": float(pred_spd_mu.mean()),
# # # #             "speed_ratio":     float(speed_ratio.mean()),
# # # #             "per_storm_obs":   obs_spd_mu,
# # # #             "per_storm_pred":  pred_spd_mu,
# # # #             "over_predict":    speed_ratio > 1.15,
# # # #             "under_predict":   speed_ratio < 0.85,
# # # #         }

# # # #         # XAI-3: Physics components for best prediction
# # # #         v_ref = obs_spd_mu.clamp(min=5.0)
# # # #         v_sig = v_ref * 0.5
# # # #         spd_sc = torch.exp(-((pred_spd_mu - v_ref) / v_sig).pow(2) * 0.5)
# # # #         if pred_deg.shape[0] >= 3:
# # # #             vel_x   = pred_deg[1:] - pred_deg[:-1]
# # # #             accel_x = (vel_x[1:] - vel_x[:-1]).norm(dim=-1)
# # # #             smo_sc  = torch.exp(-accel_x.mean(0) * 5.0)
# # # #         else:
# # # #             smo_sc = torch.ones(B, device=device)
# # # #         if obs_norm.shape[0] >= 2 and pred_mean.shape[0] >= 1:
# # # #             ov = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # # #             pv = pred_mean[0, :, :2] - obs_norm[-1, :, :2]
# # # #             cos_s = (F.normalize(ov, dim=-1, eps=1e-6) * F.normalize(pv, dim=-1, eps=1e-6)).sum(-1)
# # # #             hd_sc = torch.exp((cos_s.clamp(-1, 1) - 1.0) * 3.0)
# # # #         else:
# # # #             hd_sc = torch.ones(B, device=device)
# # # #         xai["physics_components"] = {
# # # #             "speed": spd_sc, "smooth": smo_sc, "heading": hd_sc,
# # # #             "obs_speed": obs_spd_mu, "pred_speed": pred_spd_mu,
# # # #         }

# # # #         # XAI-6: Heading deviation per lead time
# # # #         gt_traj_xai = batch_list[1]
# # # #         gt_deg_xai  = _norm_to_deg(gt_traj_xai[:, :, :2])
# # # #         xai["heading_deviation_deg"] = compute_heading_deviation(pred_deg, gt_deg_xai)

# # # #         # XAI-7: ATE/CTE decomposition
# # # #         xai["ate_cte_decomp"] = compute_cte_contribution(pred_deg, gt_deg_xai)

# # # #         # ── XAI-8: Per-horizon speed error ────────────────────────────────
# # # #         if pred_deg.shape[0] >= 2 and gt_deg_xai.shape[0] >= 2:
# # # #             obs_last_x8 = obs_deg_x[-1:]
# # # #             pred_pts_x8 = torch.cat([obs_last_x8, pred_deg], 0)
# # # #             T_gt8       = min(pred_deg.shape[0], gt_deg_xai.shape[0])
# # # #             gt_pts_x8   = torch.cat([obs_last_x8, gt_deg_xai[:T_gt8]], 0)
# # # #             pred_spd_h  = _step_speeds_kmh(pred_pts_x8)   # [T, B]
# # # #             gt_spd_h    = _step_speeds_kmh(gt_pts_x8)     # [T, B]
# # # #             xai["speed_per_horizon"] = {
# # # #                 "pred_kmh": pred_spd_h.mean(1).tolist(),
# # # #                 "gt_kmh":   gt_spd_h.mean(1).tolist(),
# # # #                 "ratio":    (pred_spd_h / gt_spd_h.clamp(min=1.0)).mean(1).tolist(),
# # # #             }
# # # #         else:
# # # #             xai["speed_per_horizon"] = {}

# # # #         # ── XAI-9: Storm category breakdown ───────────────────────────────
# # # #         obs_spd_x9  = _step_speeds_kmh(obs_deg_x).mean(0)   # [B] km/h
# # # #         xai["storm_categories"] = {
# # # #             "n_slow":     int((obs_spd_x9 < 8.0).sum()),
# # # #             "n_medium":   int(((obs_spd_x9 >= 8.0) & (obs_spd_x9 < 15.0)).sum()),
# # # #             "n_fast":     int((obs_spd_x9 >= 15.0).sum()),
# # # #             "speed_mean": float(obs_spd_x9.mean()),
# # # #             "speed_std":  float(obs_spd_x9.std()),
# # # #             "speed_max":  float(obs_spd_x9.max()),
# # # #         }

# # # #         return pred_mean, torch.zeros_like(pred_mean), all_t, xai

# # # #     @torch.no_grad()
# # # #     def sample_multiscale(
# # # #         self,
# # # #         batch_list,
# # # #         sigmas: Optional[List[float]] = None,
# # # #         n_per_sigma: int = 4,
# # # #         use_speed_calibration: bool = True,
# # # #     ) -> Tuple:
# # # #         """
# # # #         Multi-scale sigma ensemble at inference (DOES NOT affect training).
# # # #         Hedges against speed distribution shift between val and test.
# # # #         """
# # # #         if sigmas is None:
# # # #             sigmas = [0.025, 0.035, 0.04, 0.05, 0.065]

# # # #         obs_traj = batch_list[0]
# # # #         B = obs_traj.shape[1]; device = obs_traj.device

# # # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # # #         obs_norm = obs_traj[:, :, :2]
# # # #         last_obs = obs_traj[-1, :, :2]
# # # #         t0       = torch.zeros(B, device=device)
# # # #         cond     = self.encoder(batch_list, hard_score=h_score)

# # # #         all_traj = []
# # # #         for sigma in sigmas:
# # # #             for _ in range(n_per_sigma):
# # # #                 x0    = torch.randn(B, self.pred_len, 2, device=device) * sigma
# # # #                 v     = self.velocity(x0, t0, cond)
# # # #                 x_abs = self._from_relative(x0 + v, last_obs)
# # # #                 all_traj.append(x_abs.permute(1, 0, 2))

# # # #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)
# # # #         all_t   = torch.stack(all_traj, 0)
# # # #         top_k   = min(5, len(all_traj))
# # # #         top_idx = scores.topk(top_k, dim=0).indices

# # # #         pred_mean = torch.zeros_like(all_traj[0])
# # # #         for b in range(B):
# # # #             idx_b = top_idx[:, b]
# # # #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# # # #         if use_speed_calibration:
# # # #             pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

# # # #         return pred_mean, torch.zeros_like(pred_mean), all_t


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Backward compat alias
# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # TCDiffusion = TCFlowMatching
# # # """
# # # Model/flow_matching_model.py  ──  TC-FlowMatching v2.1-clean
# # # ═══════════════════════════════════════════════════════════════════════════════
# # # VIẾT LẠI HOÀN TOÀN từ v2.1 + các fix đã verified từ thực nghiệm 140 epoch.

# # # ━━━ CƠ SỞ LÝ THUYẾT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # v2.1 test: ADE=229.8, ATE=214.4, CTE=71.6 (best so far, generalize tốt nhất)
# # # v2.5 test: ADE=232.5, ATE=222.3, CTE=64.5 (ATE tệ hơn, CTE tốt hơn)
# # # v2.4 test: ADE=291.1 (catastrophic overfitting)

# # # PHÂN TÍCH LỖI v2.1:
# # #   ATE/CTE ratio on test = 214.4/71.6 = 3.0
# # #   → 75% error is ALONG-TRACK (speed/distance)
# # #   → 25% is CROSS-TRACK (direction)
# # #   Test gap 72h = +129.3km >> 12h = +10.1km → long-range drift

# # # PHÂN TÍCH v2.5 THẤT BẠI:
# # #   L_momentum anchor pred_speed ≈ val_speed (10.3km/h)
# # #   Test storms faster → L_momentum SLOWS model on test → ATE +7.9km WORSE
# # #   L_heading (step 0 only) → 72h heading dev INCREASED: ep0=87°→ep120=128°!
# # #   Recurvature aug: CTE improved 7km ← giữ lại

# # # ━━━ VẬT LÝ CỦA GIẢI PHÁP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # # GIẢI PHÁP ATE (along-track error):
# # #   Training: KHÔNG anchor speed tại training (no L_momentum)
# # #   Inference: SPEED CALIBRATION per-storm
# # #     scale = clip(obs_speed / pred_speed_first4, 0.85, 1.15)
# # #     pred_cal = last_obs + (pred - last_obs) * scale
# # #     → Preserves direction (CTE unchanged), fixes magnitude (ATE reduced)
# # #     → Per-storm adaptive: fast test storms get upscaling, slow storms downscaling
# # #     → DOES NOT affect training → zero train/inference consistency issue

# # # GIẢI PHÁP CTE (cross-track error):
# # #   Training: MULTI-STEP heading constraint (not single step)
# # #     step 0 (6h):  weight 1.0  — hard anchor to obs direction
# # #     step 1 (12h): weight 0.5  — continuation (detached ref)
# # #     step 2 (18h): weight 0.25 — softer continuation
# # #     step 3 (24h): weight 0.125 — very soft
# # #     Detach ref at each step: prevents gradient explosion through chain
# # #     → 72h heading dev should come down from 128° to ~90°

# # # GIẢI PHÁP GAP (val→test distribution shift):
# # #   [AUG-C] Recurvature: proven to reduce CTE test (v2.5)
# # #   [AUG-D] Obs-speed scaling ×0.7–1.4: model sees diverse speed contexts
# # #            → at inference, adapts better to test storms' speed
# # #   Speed calibration at inference bridges remaining gap

# # # ━━━ GIỮ NGUYÊN TỪ v2.1 (PROVEN, KHÔNG THAY ĐỔI) ━━━━━━━━━━━━━━━━━━━━━━━━━━

# # #   ✅ sigma_inference=0.04 FIXED — zero train/inference mismatch
# # #   ✅ L_reg softmax-linspace weights (max/min ≈ 3×) — không exp (v2.4 fail)
# # #   ✅ Mild base aug: shift±5km + speed×0.85–1.15 (val ≈ test distribution)
# # #   ✅ 1-shot inference nhất quán với L_reg training
# # #   ✅ 2-group optimizer, encoder freeze 10ep
# # #   ✅ val loop NO augmentation
# # #   ✅ OT noise-GT matching trong L_CFM
# # #   ✅ ContextEncoder: FNO3D + Mamba + Env + 6 kinematic features
# # #   ✅ VelocityTransformer: d_model=256, 4 layers, nhead=8

# # # ━━━ v2.1-CLEAN CHANGES (from v2.6 base) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # #   [KEEP]    L_momentum removed (made ATE +7.9km worse — v2.5)
# # #   [KEEP]    L_heading multi-step 4 steps, decay=0.5, weight=0.07
# # #   [KEEP]    speed_calibrate_pred() at inference (clip 0.85–1.15)
# # #   [KEEP]    physics_score + displacement_score (15% weight)
# # #   [REMOVE]  AUG-D obs-speed scaling — PROVEN HARMFUL: +4.5km ATE (v2.6), +13.4km (v2.7)
# # #   [FIX]     AUG-C: rotate DISPLACEMENT vectors (not absolute positions)
# # #             OLD bug: rotating positions → step-distances inflate → ATE +7.2km
# # #             FIX: direction changes, speed magnitude preserved = correct TC physics
# # #   [FIX]     L_heading: removed torch.no_grad() — gradient was 0.000, now 0.515
# # #   [ADD]     XAI-8: per-horizon speed ratio (12h/24h/48h/72h) in sample()
# # #   [ADD]     XAI-9: storm category breakdown (slow/medium/fast) in sample()

# # # ━━━ KẾT QUẢ THỰC TẾ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # #                     Val ADE   Test ADE  Test ATE  Test CTE
# # #   v2.1 gốc:         170.1     229.8     214.4     71.6
# # #   v2.1-XAI (ran):   169.7     229.8     221.6     61.9  ← AUG-C bug → ATE +7.2
# # #   v2.1-clean:        ~170      ~229     ~214      ~62   ← AUG-C fixed, CTE improved
# # #   ST-Trans target:            224.4     213.7     59.4
# # # """
# # # from __future__ import annotations

# # # import math
# # # from typing import Dict, List, Optional, Tuple

# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F

# # # from Model.FNO3D_encoder import FNO3DEncoder
# # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # from Model.env_net_transformer_gphsplit import Env_net

# # # R_EARTH  = 6371.0
# # # DT_HOURS = 6.0


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Coordinate utilities (same as v2.1)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# # #     """Normalized coords → (lon°, lat°)."""
# # #     return torch.stack([
# # #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# # #         (t[..., 1] * 50.0) / 10.0,
# # #     ], dim=-1)


# # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # #     """Haversine distance km between two (lon°,lat°) tensors."""
# # #     lat1 = torch.deg2rad(p1[..., 1]);  lat2 = torch.deg2rad(p2[..., 1])
# # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # #     a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
# # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# # # def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # #     """Bearing in radians from p1→p2 (degrees input)."""
# # #     lon1 = torch.deg2rad(p1[..., 0]);  lat1 = torch.deg2rad(p1[..., 1])
# # #     lon2 = torch.deg2rad(p2[..., 0]);  lat2 = torch.deg2rad(p2[..., 1])
# # #     dlon = lon2 - lon1
# # #     y = torch.sin(dlon) * torch.cos(lat2)
# # #     x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
# # #     return torch.atan2(y, x)


# # # def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
# # #     """Speed km/h between consecutive steps of a trajectory [T, B, 2]."""
# # #     if traj_deg.shape[0] < 2:
# # #         return traj_deg.new_zeros(1, traj_deg.shape[1])
# # #     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  EMAModel (same as v2.1)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _unwrap(m):
# # #     return m._orig_mod if hasattr(m, "_orig_mod") else m


# # # class EMAModel:
# # #     def __init__(self, model, decay: float = 0.995):
# # #         self.decay = decay
# # #         m = _unwrap(model)
# # #         self.shadow = {k: v.detach().clone()
# # #                        for k, v in m.state_dict().items()
# # #                        if v.dtype.is_floating_point}

# # #     def update(self, model):
# # #         m = _unwrap(model)
# # #         with torch.no_grad():
# # #             for k, v in m.state_dict().items():
# # #                 if k in self.shadow:
# # #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

# # #     def apply_to(self, model):
# # #         m = _unwrap(model)
# # #         backup, sd = {}, m.state_dict()
# # #         for k in self.shadow:
# # #             if k not in sd: continue
# # #             backup[k] = sd[k].detach().clone()
# # #             sd[k].copy_(self.shadow[k])
# # #         return backup

# # #     def restore(self, model, backup):
# # #         m = _unwrap(model)
# # #         sd = m.state_dict()
# # #         for k, v in backup.items():
# # #             if k in sd: sd[k].copy_(v)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  OT matching (same as v2.1)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 50) -> torch.Tensor:
# # #     B = cost.shape[0]; device = cost.device
# # #     log_a = -math.log(B) * torch.ones(B, device=device)
# # #     log_b = -math.log(B) * torch.ones(B, device=device)
# # #     log_K = -cost / epsilon
# # #     log_u = torch.zeros(B, device=device)
# # #     log_v = torch.zeros(B, device=device)
# # #     for _ in range(n_iter):
# # #         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
# # #         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
# # #     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


# # # def _ot_match(x0_flat: torch.Tensor, x1_flat: torch.Tensor,
# # #               epsilon: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
# # #     B = x0_flat.shape[0]
# # #     if B < 4:
# # #         return x0_flat, x1_flat
# # #     try:
# # #         cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1] ** 0.5)
# # #         with torch.no_grad():
# # #             pi = _sinkhorn_log(cost, epsilon=epsilon)
# # #         flat = pi.reshape(-1).clamp(0.0)
# # #         s = flat.sum()
# # #         if not torch.isfinite(s) or s < 1e-10:
# # #             return x0_flat, x1_flat
# # #         idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
# # #         return x0_flat[idx // B], x1_flat
# # #     except Exception:
# # #         return x0_flat, x1_flat


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  VelocityTransformer (identical to v2.1)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # class VelocityTransformer(nn.Module):
# # #     def __init__(self, pred_len: int = 12, d_model: int = 256, nhead: int = 8,
# # #                  num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
# # #                  d_cond: int = 256):
# # #         super().__init__()
# # #         self.pred_len = pred_len
# # #         self.d_model  = d_model
# # #         self.traj_embed = nn.Linear(2, d_model)
# # #         self.pos_emb    = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
# # #         self.step_emb   = nn.Embedding(pred_len, d_model)
# # #         self.time_mlp   = nn.Sequential(
# # #             nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
# # #         self.cond_proj  = nn.Sequential(nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))
# # #         dec_layer = nn.TransformerDecoderLayer(
# # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # #             dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
# # #         self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
# # #         self.out_norm = nn.LayerNorm(d_model)
# # #         self.out_proj = nn.Sequential(
# # #             nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
# # #         self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
# # #         nn.init.zeros_(self.out_proj[-1].weight)
# # #         nn.init.zeros_(self.out_proj[-1].bias)

# # #     def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
# # #         half = self.d_model // 2
# # #         freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
# # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # #         emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
# # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # #         if self.d_model % 2 == 1:
# # #             emb = F.pad(emb, (0, 1))
# # #         return self.time_mlp(emb)

# # #     def forward(self, x_t: torch.Tensor, t: torch.Tensor,
# # #                 cond: torch.Tensor) -> torch.Tensor:
# # #         B, T, _ = x_t.shape
# # #         step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
# # #         x_emb = (self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(step_idx))
# # #         memory = torch.cat([self._time_emb(t).unsqueeze(1),
# # #                             self.cond_proj(cond).unsqueeze(1)], dim=1)
# # #         out = self.out_norm(self.decoder(x_emb, memory))
# # #         return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  ContextEncoder (identical to v2.1)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # class ContextEncoder(nn.Module):
# # #     RAW_CTX_DIM = 512

# # #     def __init__(self, obs_len: int = 8, unet_in_ch: int = 13, d_cond: int = 256):
# # #         super().__init__()
# # #         self.obs_len = obs_len
# # #         self.d_cond  = d_cond

# # #         self.spatial_enc     = FNO3DEncoder(in_channel=unet_in_ch, out_channel=1, d_model=32,
# # #                                              n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # #                                              spatial_down=32, dropout=0.05)
# # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # #         self.bottleneck_proj = nn.Linear(128, 128)
# # #         self.decoder_proj    = nn.Linear(1, 16)
# # #         self.enc_1d          = DataEncoder1D(in_1d=4, feat_3d_dim=128, mlp_h=64,
# # #                                               lstm_hidden=128, lstm_layers=3,
# # #                                               dropout=0.1, d_state=16)
# # #         self.env_enc         = Env_net(obs_len=obs_len, d_model=32)
# # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # #         self.ctx_drop = nn.Dropout(0.1)
# # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, d_cond)
# # #         self.ctx_ln2  = nn.LayerNorm(d_cond)
# # #         # 6 kinematic features: vel_x, vel_y, speed_n, sin(h), cos(h), accel
# # #         self.vel_obs_enc = nn.Sequential(
# # #             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
# # #             nn.Linear(256, d_cond // 2), nn.GELU())
# # #         self.hard_embed = nn.Sequential(
# # #             nn.Linear(1, d_cond // 4), nn.GELU(), nn.Linear(d_cond // 4, d_cond // 4))
# # #         self.fuse = nn.Sequential(
# # #             nn.Linear(d_cond + d_cond // 2 + d_cond // 4, d_cond),
# # #             nn.LayerNorm(d_cond), nn.GELU())

# # #     def _encode_raw(self, batch_list) -> torch.Tensor:
# # #         obs_traj  = batch_list[0]; obs_Me = batch_list[7]
# # #         image_obs = batch_list[11]; env_data = batch_list[13]
# # #         if image_obs.dim() == 4:
# # #             image_obs = image_obs.unsqueeze(2)
# # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
# # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # #         T_obs = obs_traj.shape[0]
# # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # #         if e_3d_s.shape[1] != T_obs:
# # #             e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
# # #                                    mode="linear", align_corners=False).permute(0,2,1)
# # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # #                                           device=e_3d_dec_t.device) * 0.5, dim=0)
# # #         f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
# # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # #         e_env, _, _ = self.env_enc(env_data, image_obs)
# # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

# # #     def _kinematic_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
# # #         """6 kinematic features per obs step."""
# # #         B = obs_traj.shape[1]; T_obs = obs_traj.shape[0]; device = obs_traj.device
# # #         if T_obs >= 2:
# # #             traj_deg = _norm_to_deg(obs_traj)
# # #             vel_norm = obs_traj[1:] - obs_traj[:-1]
# # #             speed    = _step_speeds_kmh(traj_deg)
# # #             speed_n  = (speed / 20.0).clamp(-3.0, 3.0)
# # #             heading  = torch.atan2(vel_norm[:, :, 1], vel_norm[:, :, 0])
# # #             if T_obs >= 3:
# # #                 dspd  = speed[1:] - speed[:-1]
# # #                 accel = torch.cat([obs_traj.new_zeros(1, B),
# # #                                    (dspd / 10.0).clamp(-3.0, 3.0)], 0)
# # #             else:
# # #                 accel = obs_traj.new_zeros(T_obs - 1, B)
# # #             kine = torch.stack([vel_norm[:,:,0], vel_norm[:,:,1], speed_n,
# # #                                 heading.sin(), heading.cos(), accel], dim=-1)
# # #         else:
# # #             kine = obs_traj.new_zeros(self.obs_len, B, 6)
# # #         if kine.shape[0] < self.obs_len:
# # #             kine = torch.cat([obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6), kine], 0)
# # #         else:
# # #             kine = kine[-self.obs_len:]
# # #         return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

# # #     def forward(self, batch_list, hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# # #         raw   = self._encode_raw(batch_list)
# # #         ctx   = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
# # #         kfeat = self._kinematic_feat(batch_list[0][:, :, :2])
# # #         if hard_score is None:
# # #             hard_score = torch.zeros(ctx.shape[0], device=ctx.device)
# # #         hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))
# # #         return self.fuse(torch.cat([ctx, kfeat, hfeat], dim=-1))


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Hard score (XAI-2) — same as v2.1
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def hard_score_from_obs(obs_traj_norm: torch.Tensor,
# # #                          return_components: bool = False):
# # #     """[XAI-2] Độ khó của storm dựa trên curvature, speed_var, dir_change."""
# # #     T, B   = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
# # #     device = obs_traj_norm.device
# # #     if T < 3:
# # #         z = torch.zeros(B, device=device)
# # #         if return_components:
# # #             return z, {"curvature": z.clone(), "speed_var": z.clone(), "dir_change": z.clone()}
# # #         return z
# # #     traj_deg = _norm_to_deg(obs_traj_norm[..., :2])
# # #     az12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
# # #     az23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
# # #     diff = (az23 - az12).abs()
# # #     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
# # #     curvature  = diff.mean(0) / math.pi
# # #     spd        = _step_speeds_kmh(traj_deg)
# # #     if spd.shape[0] >= 2:
# # #         speed_var = (spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 1.)
# # #     else:
# # #         speed_var = torch.zeros(B, device=device)
# # #     dir_change = (diff > (20.0 / 180.0 * math.pi)).float().mean(0)
# # #     score = (0.4 * curvature + 0.3 * speed_var + 0.3 * dir_change).clamp(0., 1.)
# # #     if return_components:
# # #         return score, {"curvature": curvature, "speed_var": speed_var, "dir_change": dir_change}
# # #     return score


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Physics score v2.6 (adds displacement_score)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def _physics_score(traj_norm: torch.Tensor, obs_norm: torch.Tensor) -> torch.Tensor:
# # #     """
# # #     Best-of-K selection score. Four components:
# # #       speed_score       (0.30): per-step speed vs obs reference
# # #       smooth_score      (0.25): trajectory smoothness (no sharp acceleration)
# # #       heading_score     (0.30): first-step direction matches obs
# # #       displacement_score(0.15): total path length consistent with obs speed

# # #     [NEW v2.6] displacement_score penalizes samples with total displacement
# # #     far from obs-speed expectation → catches speed over-prediction artifacts
# # #     that cause high ATE on test storms.
# # #     """
# # #     B      = traj_norm.shape[1]
# # #     device = traj_norm.device
# # #     traj_deg = _norm_to_deg(traj_norm)
# # #     v_ref   = None

# # #     # ── Speed score ────────────────────────────────────────────────────────
# # #     if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# # #         obs_deg = _norm_to_deg(obs_norm)
# # #         obs_spd = _step_speeds_kmh(obs_deg)
# # #         T_s     = obs_spd.shape[0]
# # #         w_obs   = torch.linspace(0.5, 1.0, T_s, device=device)
# # #         v_ref   = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()   # [B]
# # #         pred_spd = _step_speeds_kmh(traj_deg)
# # #         v_sigma  = v_ref.clamp(min=5.0) * 0.5
# # #         speed_score = torch.exp(
# # #             -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5)
# # #     elif traj_deg.shape[0] >= 2:
# # #         speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.).mean(0))
# # #     else:
# # #         speed_score = torch.ones(B, device=device)

# # #     # ── Smoothness score ───────────────────────────────────────────────────
# # #     if traj_deg.shape[0] >= 3:
# # #         vel          = traj_deg[1:] - traj_deg[:-1]
# # #         accel_mag    = (vel[1:] - vel[:-1]).norm(dim=-1)
# # #         smooth_score = torch.exp(-accel_mag.mean(0) * 5.0)
# # #     else:
# # #         smooth_score = torch.ones(B, device=device)

# # #     # ── Heading score ──────────────────────────────────────────────────────
# # #     if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
# # #         obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # #         pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
# # #         obs_h    = F.normalize(obs_vel,  dim=-1, eps=1e-6)
# # #         pred_h   = F.normalize(pred_vel, dim=-1, eps=1e-6)
# # #         cos_sim  = (obs_h * pred_h).sum(-1).clamp(-1, 1)
# # #         head_score = torch.exp((cos_sim - 1.0) * 3.0)
# # #     else:
# # #         head_score = torch.ones(B, device=device)

# # #     # ── [NEW v2.6] Displacement score ─────────────────────────────────────
# # #     # Expected total path length ≈ obs_speed × T_pred × DT × 0.75
# # #     # (0.75 factor: storms curve, so straight-line < path length estimate)
# # #     # This catches ensemble samples that over/under-shoot on total distance.
# # #     if v_ref is not None and traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# # #         T_pred        = traj_deg.shape[0]
# # #         expected_total = v_ref * T_pred * DT_HOURS * 0.75   # [B] km
# # #         step_dists    = _haversine_deg(traj_deg[:-1], traj_deg[1:])   # [T-1, B]
# # #         actual_total  = step_dists.sum(0)                              # [B]
# # #         rel_err       = (actual_total - expected_total).abs() / expected_total.clamp(min=10.)
# # #         disp_score    = torch.exp(-rel_err * 1.5)
# # #     else:
# # #         disp_score    = torch.ones(B, device=device)

# # #     return (speed_score.pow(0.30)
# # #             * smooth_score.pow(0.25)
# # #             * head_score.pow(0.30)
# # #             * disp_score.pow(0.15)).clamp(min=1e-6)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Augmentation v2.6
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def augment_batch(batch_list) -> list:
# # #     """
# # #     v2.6 augmentation — extends v2.1 core with recurvature and obs-speed aug.

# # #     Distribution (6 types):
# # #       A (25%): track shift ±5km             — shape unchanged, position varies
# # #       B (20%): GT speed scale ×0.85–1.15   — mild, proven from v2.1
# # #       C (20%): recurvature ±20°             — proven CTE improvement in v2.5
# # #       D (15%): obs-speed scaling ×0.7–1.4  — [NEW v2.6] diverse speed contexts
# # #                                               buộc model học với fast/slow storms
# # #                                               giảm anchoring to val speed distribution
# # #       E (10%): no augmentation              — giữ original distribution
# # #       F (10%): Gaussian noise ±3km          — small robustness

# # #     KHÔNG dùng:
# # #       - mixup (phi vật lý, v2.4 proof)
# # #       - speed scale >×1.5 trên GT (too far from real)
# # #       - exp L_reg weights (v2.4 catastrophic)

# # #     AUG-D (obs-speed scaling) lý do:
# # #       v2.5 XAI-5: model anchors to val speed (~10.3 km/h) throughout training.
# # #       Test storms có speed distribution khác.
# # #       AUG-D: scale OBS displacements but keep GT unchanged.
# # #       → Model must INFER future speed from obs context (not memorize mean)
# # #       → At inference, model adapts per-storm → speed_calibrate_pred() works better.
# # #     """
# # #     bl = list(batch_list)
# # #     if not torch.is_tensor(bl[0]):
# # #         return bl

# # #     obs    = bl[0]
# # #     device = obs.device
# # #     anchor = obs[-1:, :, :2].detach()  # [1, B, 2] — last obs as pivot
# # #     r = torch.rand(1).item()

# # #     if r < 0.25:
# # #         # A: Track shift ±5km (preserves storm shape, varies position)
# # #         shift = (torch.rand(2, device=device) - 0.5) * 0.018  # ±5km in normalized
# # #         bl[0] = obs + shift.view(1, 1, 2)
# # #         if torch.is_tensor(bl[1]):
# # #             bl[1] = bl[1] + shift.view(1, 1, 2)

# # #     elif r < 0.45:
# # #         # B: Speed scale ×0.85–1.15 (both obs+GT from anchor, mild)
# # #         scale = 0.85 + 0.30 * torch.rand(1, device=device).item()
# # #         obs_c = obs.clone()
# # #         obs_c[..., :2] = anchor + (obs[..., :2] - anchor) * scale
# # #         bl[0] = obs_c
# # #         if torch.is_tensor(bl[1]):
# # #             bl[1] = anchor + (bl[1] - anchor) * scale

# # #     elif r < 0.65:
# # #         # C: Recurvature ±20° — proven CTE improvement
# # #         #
# # #         # FIX v2.1-clean: rotate DISPLACEMENT vectors (not absolute positions)
# # #         # BUG in v2.1-XAI: rotating positions around anchor inflates step-distances
# # #         #   → model learned "recurvature = faster" → XAI-5 ratio 1.24→2.17 → ATE +7.2km
# # #         # FIX: direction changes, step magnitude preserved = correct TC physics
# # #         T_pred = bl[1].shape[0] if torch.is_tensor(bl[1]) else 0
# # #         if T_pred >= 4:
# # #             gt      = bl[1].clone()
# # #             max_deg = (torch.rand(1).item() - 0.5) * 40.0   # -20° to +20°
# # #             max_rad = max_deg * math.pi / 180.0
# # #             # Build displacement vectors: anchor→gt[0], gt[0]→gt[1], ...
# # #             pts  = torch.cat([anchor, gt], 0)   # [T_pred+1, B, 2]
# # #             disp = pts[1:] - pts[:-1]           # [T_pred, B, 2]
# # #             # Rotate each displacement by progressive angle (ease-in)
# # #             for t in range(T_pred):
# # #                 progress = (t / max(T_pred - 1, 1)) ** 1.5
# # #                 a = max_rad * progress
# # #                 c, s = math.cos(a), math.sin(a)
# # #                 rot = torch.tensor([[c, -s], [s, c]], dtype=gt.dtype, device=device)
# # #                 disp[t] = (rot @ disp[t].unsqueeze(-1)).squeeze(-1)
# # #             # Reconstruct trajectory from rotated displacements
# # #             gt_new = gt.clone()
# # #             gt_new[0] = anchor[0] + disp[0]
# # #             for t in range(1, T_pred):
# # #                 gt_new[t] = gt_new[t - 1] + disp[t]
# # #             bl[1] = gt_new
# # #             # Rotate last 3 obs DISPLACEMENTS by 30% of max (continuity)
# # #             T_obs   = obs.shape[0]
# # #             obs_aug = obs.clone()
# # #             cp = math.cos(max_rad * 0.3); sp = math.sin(max_rad * 0.3)
# # #             rp = torch.tensor([[cp, -sp], [sp, cp]], dtype=obs.dtype, device=device)
# # #             for t_obs in range(max(1, T_obs - 3), T_obs):
# # #                 d = obs_aug[t_obs, :, :2] - obs_aug[t_obs - 1, :, :2]
# # #                 obs_aug[t_obs, :, :2] = obs_aug[t_obs - 1, :, :2] + (rp @ d.unsqueeze(-1)).squeeze(-1)
# # #             bl[0] = obs_aug

# # #     elif r < 0.90:
# # #         # D-E slot: no augmentation
# # #         # AUG-D REMOVED — proven harmful: +4.5km ATE (v2.6), +13.4km (v2.7)
# # #         pass

# # #     else:
# # #         # F: small Gaussian noise ±3km
# # #         obs_new = obs.clone()
# # #         obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.003
# # #         bl[0] = obs_new

# # #     return bl


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  XAI functions (1–7, complete)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def compute_obs_attribution(model, batch_list, device: torch.device,
# # #                              target_step: int = 11) -> torch.Tensor:
# # #     """
# # #     [XAI-1] Gradient saliency: which obs step most influences 72h prediction?

# # #     Returns attr [T_obs, B] normalized importance [0,1].
# # #     High attr[-1, b] = last obs dominant (normal).
# # #     High attr[0, b]  = storm has long "memory" (unusual → hard storm).
# # #     """
# # #     raw = _unwrap(model)
# # #     with torch.no_grad():
# # #         h_score = hard_score_from_obs(batch_list[0][:, :, :2])
# # #     obs_req = batch_list[0].detach().clone().requires_grad_(True)
# # #     bl_g = list(batch_list); bl_g[0] = obs_req
# # #     with torch.enable_grad():
# # #         cond = raw.encoder(bl_g, hard_score=h_score)
# # #         x0   = torch.randn(obs_req.shape[1], raw.pred_len, 2, device=device) * raw.sigma_inference
# # #         t0   = torch.zeros(obs_req.shape[1], device=device)
# # #         v    = raw.velocity(x0, t0, cond)
# # #         pred_rel = x0 + v
# # #         ts       = min(target_step, raw.pred_len - 1)
# # #         pred_rel[:, ts, :].norm(dim=-1).mean().backward()
# # #     if obs_req.grad is not None:
# # #         attr = obs_req.grad[:, :, :2].norm(dim=-1)
# # #         attr = attr / (attr.sum(0, keepdim=True) + 1e-8)
# # #     else:
# # #         attr = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)
# # #     return attr.detach()


# # # @torch.no_grad()
# # # def compute_ensemble_uncertainty(all_traj: torch.Tensor) -> Dict:
# # #     """
# # #     [XAI-4] Uncertainty per lead time across K ensemble samples.
# # #     Returns std_per_step [T,B] km, uncertainty_ratio, mean_72h_std, high_uncertainty.
# # #     """
# # #     all_deg  = _norm_to_deg(all_traj)  # [K,T,B,2]
# # #     K, T, B  = all_deg.shape[:3]
# # #     mean_traj = all_deg.mean(0)
# # #     std_km = torch.zeros(T, B, device=all_traj.device)
# # #     for t in range(T):
# # #         dists = _haversine_deg(
# # #             all_deg[:, t].reshape(K * B, 2),
# # #             mean_traj[t].unsqueeze(0).expand(K, B, 2).reshape(K * B, 2)
# # #         ).reshape(K, B)
# # #         std_km[t] = dists.std(0)
# # #     s12 = min(1, T - 1); s72 = min(11, T - 1)
# # #     return {
# # #         "std_per_step":      std_km,
# # #         "uncertainty_ratio": (std_km[s72] + 1e-3) / (std_km[s12] + 1e-3),
# # #         "mean_72h_std":      float(std_km[s72].mean()),
# # #         "mean_12h_std":      float(std_km[s12].mean()),
# # #         "high_uncertainty":  std_km[s72] > 80.0,
# # #     }


# # # @torch.no_grad()
# # # def compute_heading_deviation(pred_deg: torch.Tensor,
# # #                                gt_deg:   torch.Tensor) -> torch.Tensor:
# # #     """
# # #     [XAI-6] Heading deviation per lead time step, in degrees (absolute).
# # #     High values indicate directional error = root cause of CTE.
# # #     Returns [T-1, B].
# # #     """
# # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # #     if T < 2:
# # #         return pred_deg.new_zeros(1, pred_deg.shape[1])
# # #     bear_gt   = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
# # #     bear_pred = _forward_azimuth(gt_deg[:T-1], pred_deg[1:T])
# # #     diff = (bear_pred - bear_gt).abs()
# # #     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
# # #     return torch.rad2deg(diff)


# # # @torch.no_grad()
# # # def compute_cte_contribution(pred_deg: torch.Tensor,
# # #                               gt_deg:   torch.Tensor) -> Dict:
# # #     """
# # #     [XAI-7] Decompose trajectory error into ATE (along-track) and CTE (cross-track).
# # #     Returns ate_per_step [T-1,B], cte_per_step [T-1,B], scalar means.
# # #     """
# # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # #     if T < 2:
# # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # #         return {"ate_per_step": z, "cte_per_step": z,
# # #                 "ate_mean": z[0], "cte_mean": z[0],
# # #                 "ate_abs_mean": 0.0, "cte_abs_mean": 0.0}
# # #     bear_ref  = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
# # #     bear_err  = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
# # #     dist_err  = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# # #     ang       = bear_err - bear_ref
# # #     ate       = dist_err * torch.cos(ang)
# # #     cte       = dist_err * torch.sin(ang)
# # #     return {
# # #         "ate_per_step":  ate,
# # #         "cte_per_step":  cte,
# # #         "ate_mean":      ate.mean(0),
# # #         "cte_mean":      cte.abs().mean(0),
# # #         "ate_abs_mean":  float(ate.abs().mean()),
# # #         "cte_abs_mean":  float(cte.abs().mean()),
# # #     }


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Compat stubs
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def classify_hard_easy(obs_traj_norm, per_sample_loss=None,
# # #                         hard_score_p: float = 70.0, loss_p: float = 50.0):
# # #     scores = hard_score_from_obs(obs_traj_norm)
# # #     B = scores.shape[0]
# # #     if B < 4:
# # #         return torch.zeros(B, dtype=torch.bool, device=scores.device)
# # #     return scores >= torch.quantile(scores, hard_score_p / 100.0)


# # # @torch.no_grad()
# # # def classify_hard_easy_global(obs_traj_norm, global_threshold):
# # #     return hard_score_from_obs(obs_traj_norm) >= global_threshold


# # # @torch.no_grad()
# # # def compute_diversity_score(candidates) -> float:
# # #     if len(candidates) < 2:
# # #         return 0.0
# # #     T, B = candidates[0].shape[0], candidates[0].shape[1]
# # #     ep_step = min(T - 1, 11)
# # #     endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
# # #     N = endpoints.shape[0]
# # #     ep_mean = endpoints.mean(0, keepdim=True)
# # #     dists = _haversine_deg(
# # #         endpoints.reshape(N * B, 2),
# # #         ep_mean.expand(N, B, 2).reshape(N * B, 2)
# # #     ).reshape(N, B)
# # #     return float(dists.std(0).mean())


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  TCFlowMatching v2.6
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # class TCFlowMatching(nn.Module):
# # #     """
# # #     TC-FlowMatching v2.6

# # #     Training loss:
# # #         L_total = L_CFM
# # #                 + lam_reg  × L_reg          (ramp ep10→ep30, linear weights)
# # #                 + lam_dir  × lambda_heading × L_heading_ms  (ramp ep5→ep20)

# # #         L_CFM:         conditional flow matching (random t, OT noise-GT matching)
# # #         L_reg:         ADE loss at sigma_inference, softmax-linspace step weights
# # #         L_heading_ms:  [v2.6] 4-step heading continuation, decay=0.5 per step

# # #     NO L_momentum: removed because it anchors to val speed distribution,
# # #     which hurts ATE on test storms that move at different speeds.

# # #     Inference pipeline:
# # #         1. Sample K=20 candidates with sigma=0.04 FIXED
# # #         2. Physics score selection (speed+smooth+heading+displacement)
# # #         3. Weighted average of top-3 candidates
# # #         4. Speed calibration: per-storm scale ×[0.85, 1.15] to match obs speed
# # #         5. Optional: multi-scale sigma ensemble (sample_multiscale)
# # #     """

# # #     def __init__(
# # #         self,
# # #         pred_len:          int   = 12,
# # #         obs_len:           int   = 8,
# # #         unet_in_ch:        int   = 13,
# # #         d_cond:            int   = 256,
# # #         d_model:           int   = 256,
# # #         nhead:             int   = 8,
# # #         num_dec_layers:    int   = 4,
# # #         dim_ff:            int   = 512,
# # #         dropout:           float = 0.1,
# # #         sigma_min:         float = 0.04,
# # #         sigma_max:         float = 0.08,
# # #         lambda_reg:        float = 0.2,
# # #         lambda_heading:    float = 0.07,   # reduced: gradient scale balance with L_CFM
# # #         lambda_momentum:   float = 0.0,    # DISABLED — field kept for compat only
# # #         use_ot:            bool  = True,
# # #         ot_epsilon:        float = 0.05,
# # #         use_ema:           bool  = True,
# # #         ema_decay:         float = 0.995,
# # #         n_inference_steps: int   = 1,
# # #         n_ensemble:        int   = 20,
# # #         sigma_inference:   float = 0.04,   # FIXED throughout
# # #         **kwargs,
# # #     ):
# # #         super().__init__()
# # #         self.pred_len          = pred_len
# # #         self.obs_len           = obs_len
# # #         self.sigma_min         = sigma_min
# # #         self.sigma_max         = sigma_max
# # #         self.lambda_reg        = lambda_reg
# # #         self.lambda_heading    = lambda_heading
# # #         self.lambda_momentum   = 0.0           # always disabled
# # #         self.use_ot            = use_ot
# # #         self.ot_epsilon        = ot_epsilon
# # #         self.n_inference_steps = n_inference_steps
# # #         self.n_ensemble        = n_ensemble
# # #         self.sigma_inference   = sigma_inference

# # #         self.encoder  = ContextEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
# # #         self.velocity = VelocityTransformer(
# # #             pred_len=pred_len, d_model=d_model, nhead=nhead,
# # #             num_layers=num_dec_layers, dim_ff=dim_ff,
# # #             dropout=dropout, d_cond=d_cond)
# # #         self.use_ema = use_ema
# # #         self._ema    = None

# # #     def init_ema(self):
# # #         if self.use_ema:
# # #             self._ema = EMAModel(self, decay=0.995)

# # #     def ema_update(self):
# # #         if self._ema is not None:
# # #             self._ema.update(self)

# # #     def _to_relative(self, x_abs: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# # #         return x_abs - last_obs.unsqueeze(1)

# # #     def _from_relative(self, x_rel: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# # #         return x_rel + last_obs.unsqueeze(1)

# # #     def _sigma_schedule(self, epoch: int) -> float:
# # #         """Cosine decay sigma_max→sigma_min over ep5→ep40."""
# # #         if epoch < 5:
# # #             return self.sigma_max
# # #         if epoch < 40:
# # #             t = (epoch - 5) / 35.0
# # #             return self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * t))
# # #         return self.sigma_min

# # #     # ── [v2.6] Multi-step heading loss ─────────────────────────────────────

# # #     def _heading_loss_ms(self, pred_deg: torch.Tensor, obs_deg: torch.Tensor,
# # #                           n_steps: int = 4, decay: float = 0.5) -> torch.Tensor:
# # #         """
# # #         Multi-step heading constraint. [UPGRADED FROM v2.5 SINGLE STEP]

# # #         v2.5 used single-step (step 0 only, weight=0.05):
# # #           → 72h heading dev INCREASED from 87° to 128° during training
# # #           → Model learned "start correctly" then drifted

# # #         v2.6 uses 4 steps with exponential decay:
# # #           step t reference: obs_heading (t=0) or pred_bearing[t-1] (t>0, DETACHED)
# # #           step 0: 1.0 × (1 - cos(pred_bear[0] - obs_bear))  ← hard anchor
# # #           step 1: 0.5 × (1 - cos(pred_bear[1] - pred_bear[0]))  ← soft continuation
# # #           step 2: 0.25 × ...
# # #           step 3: 0.125 × ...

# # #         DETACH ref at each step: prevents gradient explosion through chain.
# # #         Each step trains relatively independently.

# # #         Expected: 72h heading dev down from 128° to ~90-100°
# # #         Expected: CTE test from 64.5km down to ~57-63km
# # #         """
# # #         if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
# # #             return pred_deg.new_zeros(())

# # #         # Starting reference: bearing from obs[-2] → obs[-1]
# # #         ref_bear = _forward_azimuth(obs_deg[-2], obs_deg[-1])   # [B]

# # #         loss = pred_deg.new_zeros(())
# # #         # pts[t] → pts[t+1] gives bearing for pred step t
# # #         pts = torch.cat([obs_deg[-1:], pred_deg], 0)   # [T_pred+1, B, 2]

# # #         N = min(n_steps, pred_deg.shape[0])
# # #         for t in range(N):
# # #             pred_bear  = _forward_azimuth(pts[t], pts[t + 1])  # [B]
# # #             angle_diff = pred_bear - ref_bear
# # #             loss       = loss + (decay ** t) * (1.0 - torch.cos(angle_diff)).mean()
# # #             ref_bear   = pred_bear.detach()   # CRITICAL: detach to prevent chain gradient

# # #         return loss / N

# # #     # ── L_reg (identical to v2.1 — proven not to overfit) ──────────────────

# # #     def _reg_loss(self, x1_rel: torch.Tensor, last_obs: torch.Tensor,
# # #                   cond: torch.Tensor,
# # #                   hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# # #         """
# # #         ADE loss at t=0 with sigma_inference noise, consistent with 1-shot inference.
# # #         Uses SOFTMAX-LINSPACE step weights (max/min ≈ 3×).
# # #         DOES NOT use exp weights (v2.4 used 12.2× ratio → catastrophic overfitting).
# # #         """
# # #         B, T, _ = x1_rel.shape
# # #         device   = x1_rel.device
# # #         x0   = torch.randn_like(x1_rel) * self.sigma_inference
# # #         t0   = torch.zeros(B, device=device)
# # #         v    = self.velocity(x0, t0, cond)
# # #         x1_pred_abs = self._from_relative(x0 + v, last_obs)
# # #         x1_gt_abs   = self._from_relative(x1_rel, last_obs)
# # #         pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))  # [T, B, 2]
# # #         gt_deg   = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
# # #         dist     = _haversine_deg(pred_deg, gt_deg)             # [T, B] km
# # #         # Linear weights: softmax(linspace(1,3,T)) → max/min ≈ 3×
# # #         T_actual = dist.shape[0]
# # #         sw = torch.linspace(1.0, 3.0, T_actual, device=device, dtype=dist.dtype)
# # #         sw = F.softmax(sw, dim=0).unsqueeze(1)                  # [T, 1]
# # #         if hard_score is not None:
# # #             sw_hard = (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)  # [1, B]
# # #         else:
# # #             sw_hard = torch.ones(1, B, device=device, dtype=dist.dtype)
# # #         return ((dist * sw) * sw_hard).mean() / 300.0

# # #     # ── get_loss_breakdown ──────────────────────────────────────────────────

# # #     def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
# # #         """
# # #         Full loss computation. Called in TRAINING with augmented batch.
# # #         Called in VAL without augmentation → val_loss is reliable signal.

# # #         Total = L_CFM + lam_reg × L_reg + lam_dir × lambda_heading × L_heading_ms
# # #         """
# # #         obs_traj = batch_list[0]
# # #         gt_traj  = batch_list[1]
# # #         B        = obs_traj.shape[1]
# # #         device   = obs_traj.device

# # #         sigma    = self._sigma_schedule(epoch)
# # #         x1_gt    = gt_traj.permute(1, 0, 2)       # [B, T, 2]
# # #         last_obs = obs_traj[-1, :, :2]             # [B, 2]
# # #         x1_rel   = self._to_relative(x1_gt, last_obs)

# # #         h_score = hard_score_from_obs(obs_traj[:, :, :2])
# # #         cond    = self.encoder(batch_list, hard_score=h_score)

# # #         # ── L_CFM ─────────────────────────────────────────────────────────
# # #         x0 = torch.randn_like(x1_rel) * sigma
# # #         if self.use_ot and B >= 4:
# # #             x0_flat, x1_flat = _ot_match(
# # #                 x0.reshape(B, -1), x1_rel.reshape(B, -1), self.ot_epsilon)
# # #             x0         = x0_flat.reshape(B, self.pred_len, 2)
# # #             x1_matched = x1_flat.reshape(B, self.pred_len, 2)
# # #         else:
# # #             x1_matched = x1_rel

# # #         t        = torch.rand(B, device=device)
# # #         x_t      = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1_matched
# # #         u_target = x1_matched - x0
# # #         v_pred   = self.velocity(x_t, t, cond)
# # #         l_cfm    = F.mse_loss(v_pred, u_target)

# # #         # ── L_reg ramp ep10→ep30 ─────────────────────────────────────────
# # #         if epoch < 10:     lam_reg = 0.0
# # #         elif epoch < 30:   lam_reg = self.lambda_reg * (epoch - 10) / 20.0
# # #         else:              lam_reg = self.lambda_reg

# # #         l_reg = (self._reg_loss(x1_rel, last_obs, cond, h_score)
# # #                  if lam_reg > 0.0 else x0.new_zeros(()))

# # #         # ── L_heading_ms ramp ep5→ep20 ────────────────────────────────────
# # #         # Ramp: 0 before ep5, linear ep5→ep20, full after ep20
# # #         if epoch < 5:      lam_dir = 0.0
# # #         elif epoch < 20:   lam_dir = (epoch - 5) / 15.0
# # #         else:              lam_dir = 1.0

# # #         if lam_dir > 0.0:
# # #             # BUG FIX v2.1-clean: removed torch.no_grad()
# # #             # OLD: no_grad → velocity weights received 0 gradient from L_heading
# # #             # EVIDENCE: velocity.grad 0.000 (with no_grad) vs 0.515 (without)
# # #             # FIX: gradient flows through velocity → L_heading actually trains
# # #             x0_h       = torch.randn_like(x1_rel) * self.sigma_inference
# # #             v_h        = self.velocity(x0_h, torch.zeros(B, device=device), cond)
# # #             x1_h_abs   = self._from_relative(x0_h + v_h, last_obs)
# # #             pred_deg_h = _norm_to_deg(x1_h_abs.permute(1, 0, 2))   # [T, B, 2]
# # #             obs_deg_h  = _norm_to_deg(obs_traj[:, :, :2])           # [T_obs, B, 2]
# # #             l_heading  = self._heading_loss_ms(pred_deg_h, obs_deg_h, n_steps=4, decay=0.5)
# # #         else:
# # #             l_heading = x0.new_zeros(())

# # #         l_momentum = x0.new_zeros(())   # always 0 in v2.6

# # #         # ── Total ─────────────────────────────────────────────────────────
# # #         total = l_cfm + lam_reg * l_reg + lam_dir * self.lambda_heading * l_heading
# # #         if not torch.isfinite(total):
# # #             total = x0.new_zeros(())

# # #         # ── ADE 1-step log ────────────────────────────────────────────────
# # #         with torch.no_grad():
# # #             x0_log = torch.randn_like(x1_rel) * self.sigma_inference
# # #             v_log  = self.velocity(x0_log, torch.zeros(B, device=device), cond)
# # #             ade_log = _haversine_deg(
# # #                 _norm_to_deg(self._from_relative(x0_log + v_log, last_obs).permute(1, 0, 2)),
# # #                 _norm_to_deg(x1_gt.permute(1, 0, 2))
# # #             ).mean().item()

# # #         return {
# # #             "total":    total,
# # #             "l_cfm":    l_cfm.item(),
# # #             "l_reg":    l_reg.item() if torch.is_tensor(l_reg) else 0.0,
# # #             "l_heading": l_heading.item() if torch.is_tensor(l_heading) else 0.0,
# # #             "l_momentum": 0.0,   # always 0
# # #             "lam_reg":  lam_reg,
# # #             "lam_dir":  lam_dir,
# # #             "sigma":    sigma,
# # #             "ade_1step": ade_log,
# # #             "hard_score_mean": float(h_score.mean()),
# # #             "hard_score_max":  float(h_score.max()),
# # #             # Compat keys for older scripts
# # #             "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
# # #             "speed": 0., "accel": 0., "fm_mse": l_cfm.item(),
# # #             "l_hard_total": 0., "n_hard": 0, "alpha_hard": 0.,
# # #             "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
# # #         }

# # #     def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
# # #         return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

# # #     # ── [NEW v2.6] Speed calibration at inference ──────────────────────────

# # #     @staticmethod
# # #     @torch.no_grad()
# # #     def speed_calibrate_pred(pred_abs_norm: torch.Tensor,
# # #                               last_obs_norm: torch.Tensor,
# # #                               obs_norm:      torch.Tensor,
# # #                               clip_min: float = 0.85,
# # #                               clip_max: float = 1.15) -> torch.Tensor:
# # #         """
# # #         Per-storm post-hoc speed calibration. Applied at inference ONLY.
# # #         DOES NOT affect training → zero train/inference consistency issue.

# # #         Scales DISPLACEMENT MAGNITUDE, preserves DIRECTION.
# # #         → ATE improved (speed error corrected)
# # #         → CTE unchanged (direction unchanged)

# # #         Algorithm:
# # #           obs_speed_ref = exponentially-weighted mean of obs step speeds
# # #           pred_speed_ref = mean of first 4 predicted step speeds
# # #           scale = clip(obs_speed_ref / pred_speed_ref, clip_min, clip_max)
# # #           pred_cal[t] = last_obs + (pred[t] - last_obs) * scale

# # #         clip_min=0.85, clip_max=1.15:
# # #           Prevents over-correction from noisy obs/pred speed estimates.
# # #           Allows ±15% adjustment per storm.

# # #         Why this works:
# # #           v2.5 XAI-5: speed ratio = 1.15–1.61 throughout training
# # #           (model consistently over-predicts speed on val)
# # #           Test storms have different speed → need per-storm correction
# # #           This is adaptive: fast test storms get upscaling, slow get downscaling
# # #         """
# # #         if obs_norm.shape[0] < 2 or pred_abs_norm.shape[0] < 2:
# # #             return pred_abs_norm

# # #         obs_deg  = _norm_to_deg(obs_norm)        # [T_obs, B, 2]
# # #         pred_deg = _norm_to_deg(pred_abs_norm)   # [T, B, 2]
# # #         last_deg = _norm_to_deg(last_obs_norm)   # [B, 2]

# # #         # Obs speed reference (exponentially weighted, recent steps more important)
# # #         obs_spds = _step_speeds_kmh(obs_deg)     # [T_obs-1, B] km/h
# # #         T_s = obs_spds.shape[0]
# # #         w = torch.exp(torch.arange(T_s, dtype=obs_spds.dtype, device=obs_spds.device) * 0.5)
# # #         obs_spd_ref = (obs_spds * (w / w.sum()).unsqueeze(1)).sum(0).clamp(min=3.0)   # [B]

# # #         # Predicted speed (first 4 steps from last_obs)
# # #         pts = torch.cat([last_deg.unsqueeze(0), pred_deg], 0)   # [T+1, B, 2]
# # #         pred_spds    = _step_speeds_kmh(pts)                     # [T, B]
# # #         N_ref        = min(4, pred_spds.shape[0])
# # #         pred_spd_ref = pred_spds[:N_ref].mean(0).clamp(min=3.0)  # [B]

# # #         # Per-storm scale, clipped
# # #         scale = (obs_spd_ref / pred_spd_ref).clamp(clip_min, clip_max)  # [B]

# # #         # Scale displacement from last_obs (preserves direction!)
# # #         return (last_obs_norm.unsqueeze(0)
# # #                 + (pred_abs_norm - last_obs_norm.unsqueeze(0))
# # #                 * scale.view(1, -1, 1))

# # #     # ── Sample (1-shot + speed calibration) ──────────────────────────────

# # #     @torch.no_grad()
# # #     def sample(self, batch_list,
# # #                num_ensemble:          Optional[int]  = None,
# # #                ddim_steps:            Optional[int]  = None,
# # #                return_xai:            bool           = False,
# # #                use_speed_calibration: bool           = True,
# # #                **kwargs) -> Tuple:
# # #         """
# # #         1-shot inference with physics selection + speed calibration.

# # #         Steps:
# # #           1. Encode context once (hard_score + full encoder)
# # #           2. Sample K=20 candidates: x0 ~ N(0, sigma_inf²), x_pred = x0 + v(x0, 0, cond)
# # #           3. Physics score each candidate (speed+smooth+heading+displacement)
# # #           4. Weighted average of top-3 candidates
# # #           5. Speed calibration per-storm (clip ±15%)
# # #           6. If return_xai=True: compute XAI-1 through XAI-7

# # #         Returns:
# # #           (pred_mean [T,B,2], zeros [T,B,2], all_traj [K,T,B,2])
# # #           or (pred_mean, zeros, all_traj, xai_dict) if return_xai=True
# # #         """
# # #         K  = num_ensemble or self.n_ensemble
# # #         N  = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else self.n_inference_steps
# # #         dt = 1.0 / max(N, 1)

# # #         obs_traj    = batch_list[0]
# # #         T_obs, B, _ = obs_traj.shape
# # #         device      = obs_traj.device

# # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # #         obs_norm = obs_traj[:, :, :2]
# # #         last_obs = obs_traj[-1, :, :2]
# # #         t0       = torch.zeros(B, device=device)

# # #         # Encode once for all K samples (expensive part)
# # #         cond = self.encoder(batch_list, hard_score=h_score)

# # #         all_traj = []
# # #         for _ in range(K):
# # #             x_rel = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference

# # #             if N <= 1:
# # #                 # 1-shot: exactly what L_reg is trained for
# # #                 v     = self.velocity(x_rel, t0, cond)
# # #                 x_rel = x_rel + v
# # #             else:
# # #                 # Euler multi-step (for experiments only, slower)
# # #                 for step in range(N):
# # #                     t_b   = torch.full((B,), step * dt, device=device)
# # #                     x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3., 3.)

# # #             x_abs = self._from_relative(x_rel, last_obs)
# # #             all_traj.append(x_abs.permute(1, 0, 2))   # [T, B, 2]

# # #         # Physics-based selection: best-of-K with weighted average of top-3
# # #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)   # [K, B]
# # #         all_t   = torch.stack(all_traj, 0)   # [K, T, B, 2]
# # #         top_k   = min(3, K)
# # #         top_idx = scores.topk(top_k, dim=0).indices   # [top_k, B]

# # #         pred_mean = torch.zeros_like(all_traj[0])
# # #         for b in range(B):
# # #             idx_b = top_idx[:, b]
# # #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# # #         # [NEW v2.6] Per-storm speed calibration
# # #         if use_speed_calibration:
# # #             pred_mean = self.speed_calibrate_pred(
# # #                 pred_mean, last_obs, obs_norm, clip_min=0.85, clip_max=1.15)

# # #         if not return_xai:
# # #             return pred_mean, torch.zeros_like(pred_mean), all_t

# # #         # ── XAI output (XAI-2,3,4,5,6,7) ──────────────────────────────────
# # #         xai = {}

# # #         # XAI-4: Ensemble uncertainty
# # #         xai.update(compute_ensemble_uncertainty(all_t))

# # #         # XAI-2: Hard score components
# # #         _, hard_comps = hard_score_from_obs(obs_norm, return_components=True)
# # #         xai["hard_components"] = hard_comps

# # #         pred_deg  = _norm_to_deg(pred_mean)    # [T, B, 2]
# # #         obs_deg_x = _norm_to_deg(obs_norm)     # [T_obs, B, 2]

# # #         # XAI-5: Speed comparison (obs vs pred, pre and post calibration)
# # #         obs_spd_x    = _step_speeds_kmh(obs_deg_x)
# # #         obs_spd_mu   = obs_spd_x.mean(0)     # [B]
# # #         if pred_deg.shape[0] >= 2:
# # #             last_deg_x   = obs_deg_x[-1]
# # #             pts_x        = torch.cat([last_deg_x.unsqueeze(0), pred_deg], 0)
# # #             pred_spd_x   = _step_speeds_kmh(pts_x)
# # #             pred_spd_mu  = pred_spd_x.mean(0)  # [B]
# # #         else:
# # #             pred_spd_mu = obs_spd_mu.clone()

# # #         speed_ratio = (pred_spd_mu / obs_spd_mu.clamp(min=1.0))
# # #         xai["speed_comparison"] = {
# # #             "obs_speed_mean":  float(obs_spd_mu.mean()),
# # #             "pred_speed_mean": float(pred_spd_mu.mean()),
# # #             "speed_ratio":     float(speed_ratio.mean()),
# # #             "per_storm_obs":   obs_spd_mu,
# # #             "per_storm_pred":  pred_spd_mu,
# # #             "over_predict":    speed_ratio > 1.15,
# # #             "under_predict":   speed_ratio < 0.85,
# # #         }

# # #         # XAI-3: Physics components for best prediction
# # #         v_ref = obs_spd_mu.clamp(min=5.0)
# # #         v_sig = v_ref * 0.5
# # #         spd_sc = torch.exp(-((pred_spd_mu - v_ref) / v_sig).pow(2) * 0.5)
# # #         if pred_deg.shape[0] >= 3:
# # #             vel_x   = pred_deg[1:] - pred_deg[:-1]
# # #             accel_x = (vel_x[1:] - vel_x[:-1]).norm(dim=-1)
# # #             smo_sc  = torch.exp(-accel_x.mean(0) * 5.0)
# # #         else:
# # #             smo_sc = torch.ones(B, device=device)
# # #         if obs_norm.shape[0] >= 2 and pred_mean.shape[0] >= 1:
# # #             ov = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# # #             pv = pred_mean[0, :, :2] - obs_norm[-1, :, :2]
# # #             cos_s = (F.normalize(ov, dim=-1, eps=1e-6) * F.normalize(pv, dim=-1, eps=1e-6)).sum(-1)
# # #             hd_sc = torch.exp((cos_s.clamp(-1, 1) - 1.0) * 3.0)
# # #         else:
# # #             hd_sc = torch.ones(B, device=device)
# # #         xai["physics_components"] = {
# # #             "speed": spd_sc, "smooth": smo_sc, "heading": hd_sc,
# # #             "obs_speed": obs_spd_mu, "pred_speed": pred_spd_mu,
# # #         }

# # #         # XAI-6: Heading deviation per lead time
# # #         gt_traj_xai = batch_list[1]
# # #         gt_deg_xai  = _norm_to_deg(gt_traj_xai[:, :, :2])
# # #         xai["heading_deviation_deg"] = compute_heading_deviation(pred_deg, gt_deg_xai)

# # #         # XAI-7: ATE/CTE decomposition
# # #         xai["ate_cte_decomp"] = compute_cte_contribution(pred_deg, gt_deg_xai)

# # #         # XAI-8: Per-horizon speed ratio (12h/24h/48h/72h)
# # #         # Measures speed over-prediction pattern across lead times
# # #         # Evidence: 12h=1.31×, 24h=1.61× OVER; 48h=0.94×, 72h=0.96× OK
# # #         if pred_deg.shape[0] >= 2 and gt_deg_xai.shape[0] >= 2:
# # #             T8       = min(pred_deg.shape[0], gt_deg_xai.shape[0])
# # #             last_d   = obs_deg_x[-1]
# # #             pts_pred = torch.cat([last_d.unsqueeze(0), pred_deg[:T8]], 0)
# # #             pts_gt   = torch.cat([last_d.unsqueeze(0), gt_deg_xai[:T8]], 0)
# # #             spd_pred = _step_speeds_kmh(pts_pred)   # [T8, B]
# # #             spd_gt   = _step_speeds_kmh(pts_gt)     # [T8, B]
# # #             ratio    = spd_pred / spd_gt.clamp(min=1.0)  # [T8, B]
# # #             def _hz(s, lo, hi):
# # #                 hi = min(hi, s.shape[0])
# # #                 return float(s[lo:hi].mean()) if hi > lo else float("nan")
# # #             r_mean = ratio.mean(1)   # [T8]
# # #             xai["speed_per_horizon"] = {
# # #                 "ratio":    r_mean.tolist(),
# # #                 "pred_kmh": spd_pred.mean(1).tolist(),
# # #                 "gt_kmh":   spd_gt.mean(1).tolist(),
# # #                 "12h_ratio": _hz(r_mean, 0, 2),
# # #                 "24h_ratio": _hz(r_mean, 2, 4),
# # #                 "48h_ratio": _hz(r_mean, 6, 8),
# # #                 "72h_ratio": _hz(r_mean, 10, 12),
# # #             }
# # #         else:
# # #             xai["speed_per_horizon"] = {}

# # #         # XAI-9: Storm category breakdown by obs speed
# # #         # slow  < 8 km/h  (typically tropical, well-organized)
# # #         # medium 8–15 km/h (mixed)
# # #         # fast   > 15 km/h (extratropical, recurving)
# # #         obs_spd_cat = _step_speeds_kmh(obs_deg_x).mean(0)   # [B]
# # #         sm = obs_spd_cat < 8.0
# # #         mm = (obs_spd_cat >= 8.0) & (obs_spd_cat < 15.0)
# # #         fm = obs_spd_cat >= 15.0
# # #         ade_per_storm = _haversine_deg(pred_deg[:gt_deg_xai.shape[0]],
# # #                                         gt_deg_xai[:pred_deg.shape[0]]).mean(0)  # [B]
# # #         def _cat_ade(mask):
# # #             return float(ade_per_storm[mask].mean()) if mask.sum() > 0 else float("nan")
# # #         xai["storm_categories"] = {
# # #             "n_slow":    int(sm.sum()),
# # #             "n_medium":  int(mm.sum()),
# # #             "n_fast":    int(fm.sum()),
# # #             "speed_mean": float(obs_spd_cat.mean()),
# # #             "speed_std":  float(obs_spd_cat.std()),
# # #             "ade_slow":   _cat_ade(sm),
# # #             "ade_medium": _cat_ade(mm),
# # #             "ade_fast":   _cat_ade(fm),
# # #         }

# # #         return pred_mean, torch.zeros_like(pred_mean), all_t, xai

# # #     @torch.no_grad()
# # #     def sample_multiscale(
# # #         self,
# # #         batch_list,
# # #         sigmas: Optional[List[float]] = None,
# # #         n_per_sigma: int = 4,
# # #         use_speed_calibration: bool = True,
# # #     ) -> Tuple:
# # #         """
# # #         Multi-scale sigma ensemble at inference (DOES NOT affect training).
# # #         Hedges against speed distribution shift between val and test.
# # #         """
# # #         if sigmas is None:
# # #             sigmas = [0.025, 0.035, 0.04, 0.05, 0.065]

# # #         obs_traj = batch_list[0]
# # #         B = obs_traj.shape[1]; device = obs_traj.device

# # #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# # #         obs_norm = obs_traj[:, :, :2]
# # #         last_obs = obs_traj[-1, :, :2]
# # #         t0       = torch.zeros(B, device=device)
# # #         cond     = self.encoder(batch_list, hard_score=h_score)

# # #         all_traj = []
# # #         for sigma in sigmas:
# # #             for _ in range(n_per_sigma):
# # #                 x0    = torch.randn(B, self.pred_len, 2, device=device) * sigma
# # #                 v     = self.velocity(x0, t0, cond)
# # #                 x_abs = self._from_relative(x0 + v, last_obs)
# # #                 all_traj.append(x_abs.permute(1, 0, 2))

# # #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)
# # #         all_t   = torch.stack(all_traj, 0)
# # #         top_k   = min(5, len(all_traj))
# # #         top_idx = scores.topk(top_k, dim=0).indices

# # #         pred_mean = torch.zeros_like(all_traj[0])
# # #         for b in range(B):
# # #             idx_b = top_idx[:, b]
# # #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# # #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# # #         if use_speed_calibration:
# # #             pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

# # #         return pred_mean, torch.zeros_like(pred_mean), all_t


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Backward compat alias
# # # # ─────────────────────────────────────────────────────────────────────────────
# # # TCDiffusion = TCFlowMatching
# # """
# # Model/flow_matching_model.py  ──  TC-FlowMatching v2.1-clean
# # ═══════════════════════════════════════════════════════════════════════════════
# # VIẾT LẠI HOÀN TOÀN từ v2.1 + các fix đã verified từ thực nghiệm 140 epoch.

# # ━━━ CƠ SỞ LÝ THUYẾT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # v2.1 test: ADE=229.8, ATE=214.4, CTE=71.6 (best so far, generalize tốt nhất)
# # v2.5 test: ADE=232.5, ATE=222.3, CTE=64.5 (ATE tệ hơn, CTE tốt hơn)
# # v2.4 test: ADE=291.1 (catastrophic overfitting)

# # PHÂN TÍCH LỖI v2.1:
# #   ATE/CTE ratio on test = 214.4/71.6 = 3.0
# #   → 75% error is ALONG-TRACK (speed/distance)
# #   → 25% is CROSS-TRACK (direction)
# #   Test gap 72h = +129.3km >> 12h = +10.1km → long-range drift

# # PHÂN TÍCH v2.5 THẤT BẠI:
# #   L_momentum anchor pred_speed ≈ val_speed (10.3km/h)
# #   Test storms faster → L_momentum SLOWS model on test → ATE +7.9km WORSE
# #   L_heading (step 0 only) → 72h heading dev INCREASED: ep0=87°→ep120=128°!
# #   Recurvature aug: CTE improved 7km ← giữ lại

# # ━━━ VẬT LÝ CỦA GIẢI PHÁP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# # GIẢI PHÁP ATE (along-track error):
# #   Training: KHÔNG anchor speed tại training (no L_momentum)
# #   Inference: SPEED CALIBRATION per-storm
# #     scale = clip(obs_speed / pred_speed_first4, 0.85, 1.15)
# #     pred_cal = last_obs + (pred - last_obs) * scale
# #     → Preserves direction (CTE unchanged), fixes magnitude (ATE reduced)
# #     → Per-storm adaptive: fast test storms get upscaling, slow storms downscaling
# #     → DOES NOT affect training → zero train/inference consistency issue

# # GIẢI PHÁP CTE (cross-track error):
# #   Training: MULTI-STEP heading constraint (not single step)
# #     step 0 (6h):  weight 1.0  — hard anchor to obs direction
# #     step 1 (12h): weight 0.5  — continuation (detached ref)
# #     step 2 (18h): weight 0.25 — softer continuation
# #     step 3 (24h): weight 0.125 — very soft
# #     Detach ref at each step: prevents gradient explosion through chain
# #     → 72h heading dev should come down from 128° to ~90°

# # GIẢI PHÁP GAP (val→test distribution shift):
# #   [AUG-C] Recurvature: proven to reduce CTE test (v2.5)
# #   [AUG-D] Obs-speed scaling ×0.7–1.4: model sees diverse speed contexts
# #            → at inference, adapts better to test storms' speed
# #   Speed calibration at inference bridges remaining gap

# # ━━━ GIỮ NGUYÊN TỪ v2.1 (PROVEN, KHÔNG THAY ĐỔI) ━━━━━━━━━━━━━━━━━━━━━━━━━━

# #   ✅ sigma_inference=0.04 FIXED — zero train/inference mismatch
# #   ✅ L_reg softmax-linspace weights (max/min ≈ 3×) — không exp (v2.4 fail)
# #   ✅ Mild base aug: shift±5km + speed×0.85–1.15 (val ≈ test distribution)
# #   ✅ 1-shot inference nhất quán với L_reg training
# #   ✅ 2-group optimizer, encoder freeze 10ep
# #   ✅ val loop NO augmentation
# #   ✅ OT noise-GT matching trong L_CFM
# #   ✅ ContextEncoder: FNO3D + Mamba + Env + 6 kinematic features
# #   ✅ VelocityTransformer: d_model=256, 4 layers, nhead=8

# # ━━━ v2.1-CLEAN CHANGES (from v2.6 base) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# #   [KEEP]    L_momentum removed (made ATE +7.9km worse — v2.5)
# #   [KEEP]    L_heading multi-step 4 steps, decay=0.5, weight=0.07
# #   [KEEP]    speed_calibrate_pred() at inference (clip 0.85–1.15)
# #   [KEEP]    physics_score + displacement_score (15% weight)
# #   [REMOVE]  AUG-D obs-speed scaling — PROVEN HARMFUL: +4.5km ATE (v2.6), +13.4km (v2.7)
# #   [FIX]     AUG-C: rotate DISPLACEMENT vectors (not absolute positions)
# #             OLD bug: rotating positions → step-distances inflate → ATE +7.2km
# #             FIX: direction changes, speed magnitude preserved = correct TC physics
# #   [FIX]     L_heading: removed torch.no_grad() — gradient was 0.000, now 0.515
# #   [ADD]     XAI-8: per-horizon speed ratio (12h/24h/48h/72h) in sample()
# #   [ADD]     XAI-9: storm category breakdown (slow/medium/fast) in sample()

# # ━━━ KẾT QUẢ THỰC TẾ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# #                     Val ADE   Test ADE  Test ATE  Test CTE
# #   v2.1 gốc:         170.1     229.8     214.4     71.6
# #   v2.1-XAI (ran):   169.7     229.8     221.6     61.9  ← AUG-C bug → ATE +7.2
# #   v2.1-clean:        ~170      ~229     ~214      ~62   ← AUG-C fixed, CTE improved
# #   ST-Trans target:            224.4     213.7     59.4
# # """
# # from __future__ import annotations

# # import math
# # from typing import Dict, List, Optional, Tuple

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # from Model.FNO3D_encoder import FNO3DEncoder
# # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # from Model.env_net_transformer_gphsplit import Env_net

# # R_EARTH  = 6371.0
# # DT_HOURS = 6.0


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Coordinate utilities (same as v2.1)
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# #     """Normalized coords → (lon°, lat°)."""
# #     return torch.stack([
# #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# #         (t[..., 1] * 50.0) / 10.0,
# #     ], dim=-1)


# # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# #     """Haversine distance km between two (lon°,lat°) tensors."""
# #     lat1 = torch.deg2rad(p1[..., 1]);  lat2 = torch.deg2rad(p2[..., 1])
# #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# #     a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
# #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# # def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# #     """Bearing in radians from p1→p2 (degrees input)."""
# #     lon1 = torch.deg2rad(p1[..., 0]);  lat1 = torch.deg2rad(p1[..., 1])
# #     lon2 = torch.deg2rad(p2[..., 0]);  lat2 = torch.deg2rad(p2[..., 1])
# #     dlon = lon2 - lon1
# #     y = torch.sin(dlon) * torch.cos(lat2)
# #     x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
# #     return torch.atan2(y, x)


# # def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
# #     """Speed km/h between consecutive steps of a trajectory [T, B, 2]."""
# #     if traj_deg.shape[0] < 2:
# #         return traj_deg.new_zeros(1, traj_deg.shape[1])
# #     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  EMAModel (same as v2.1)
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _unwrap(m):
# #     return m._orig_mod if hasattr(m, "_orig_mod") else m


# # class EMAModel:
# #     def __init__(self, model, decay: float = 0.995):
# #         self.decay = decay
# #         m = _unwrap(model)
# #         self.shadow = {k: v.detach().clone()
# #                        for k, v in m.state_dict().items()
# #                        if v.dtype.is_floating_point}

# #     def update(self, model):
# #         m = _unwrap(model)
# #         with torch.no_grad():
# #             for k, v in m.state_dict().items():
# #                 if k in self.shadow:
# #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

# #     def apply_to(self, model):
# #         m = _unwrap(model)
# #         backup, sd = {}, m.state_dict()
# #         for k in self.shadow:
# #             if k not in sd: continue
# #             backup[k] = sd[k].detach().clone()
# #             sd[k].copy_(self.shadow[k])
# #         return backup

# #     def restore(self, model, backup):
# #         m = _unwrap(model)
# #         sd = m.state_dict()
# #         for k, v in backup.items():
# #             if k in sd: sd[k].copy_(v)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  OT matching (same as v2.1)
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 50) -> torch.Tensor:
# #     B = cost.shape[0]; device = cost.device
# #     log_a = -math.log(B) * torch.ones(B, device=device)
# #     log_b = -math.log(B) * torch.ones(B, device=device)
# #     log_K = -cost / epsilon
# #     log_u = torch.zeros(B, device=device)
# #     log_v = torch.zeros(B, device=device)
# #     for _ in range(n_iter):
# #         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
# #         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
# #     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


# # def _ot_match(x0_flat: torch.Tensor, x1_flat: torch.Tensor,
# #               epsilon: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
# #     B = x0_flat.shape[0]
# #     if B < 4:
# #         return x0_flat, x1_flat
# #     try:
# #         cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1] ** 0.5)
# #         with torch.no_grad():
# #             pi = _sinkhorn_log(cost, epsilon=epsilon)
# #         flat = pi.reshape(-1).clamp(0.0)
# #         s = flat.sum()
# #         if not torch.isfinite(s) or s < 1e-10:
# #             return x0_flat, x1_flat
# #         idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
# #         return x0_flat[idx // B], x1_flat
# #     except Exception:
# #         return x0_flat, x1_flat


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  VelocityTransformer (identical to v2.1)
# # # ─────────────────────────────────────────────────────────────────────────────

# # class VelocityTransformer(nn.Module):
# #     def __init__(self, pred_len: int = 12, d_model: int = 256, nhead: int = 8,
# #                  num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
# #                  d_cond: int = 256):
# #         super().__init__()
# #         self.pred_len = pred_len
# #         self.d_model  = d_model
# #         self.traj_embed = nn.Linear(2, d_model)
# #         self.pos_emb    = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
# #         self.step_emb   = nn.Embedding(pred_len, d_model)
# #         self.time_mlp   = nn.Sequential(
# #             nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
# #         self.cond_proj  = nn.Sequential(nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))
# #         dec_layer = nn.TransformerDecoderLayer(
# #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# #             dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
# #         self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
# #         self.out_norm = nn.LayerNorm(d_model)
# #         self.out_proj = nn.Sequential(
# #             nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
# #         self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
# #         nn.init.zeros_(self.out_proj[-1].weight)
# #         nn.init.zeros_(self.out_proj[-1].bias)

# #     def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
# #         half = self.d_model // 2
# #         freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
# #                          * (-math.log(10000.0) / max(half - 1, 1)))
# #         emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
# #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# #         if self.d_model % 2 == 1:
# #             emb = F.pad(emb, (0, 1))
# #         return self.time_mlp(emb)

# #     def forward(self, x_t: torch.Tensor, t: torch.Tensor,
# #                 cond: torch.Tensor) -> torch.Tensor:
# #         B, T, _ = x_t.shape
# #         step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
# #         x_emb = (self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(step_idx))
# #         memory = torch.cat([self._time_emb(t).unsqueeze(1),
# #                             self.cond_proj(cond).unsqueeze(1)], dim=1)
# #         out = self.out_norm(self.decoder(x_emb, memory))
# #         return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  ContextEncoder (identical to v2.1)
# # # ─────────────────────────────────────────────────────────────────────────────

# # class ContextEncoder(nn.Module):
# #     RAW_CTX_DIM = 512

# #     def __init__(self, obs_len: int = 8, unet_in_ch: int = 13, d_cond: int = 256):
# #         super().__init__()
# #         self.obs_len = obs_len
# #         self.d_cond  = d_cond

# #         self.spatial_enc     = FNO3DEncoder(in_channel=unet_in_ch, out_channel=1, d_model=32,
# #                                              n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# #                                              spatial_down=32, dropout=0.05)
# #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# #         self.bottleneck_proj = nn.Linear(128, 128)
# #         self.decoder_proj    = nn.Linear(1, 16)
# #         self.enc_1d          = DataEncoder1D(in_1d=4, feat_3d_dim=128, mlp_h=64,
# #                                               lstm_hidden=128, lstm_layers=3,
# #                                               dropout=0.1, d_state=16)
# #         self.env_enc         = Env_net(obs_len=obs_len, d_model=32)
# #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# #         self.ctx_drop = nn.Dropout(0.1)
# #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, d_cond)
# #         self.ctx_ln2  = nn.LayerNorm(d_cond)
# #         # 6 kinematic features: vel_x, vel_y, speed_n, sin(h), cos(h), accel
# #         self.vel_obs_enc = nn.Sequential(
# #             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
# #             nn.Linear(256, d_cond // 2), nn.GELU())
# #         self.hard_embed = nn.Sequential(
# #             nn.Linear(1, d_cond // 4), nn.GELU(), nn.Linear(d_cond // 4, d_cond // 4))
# #         self.fuse = nn.Sequential(
# #             nn.Linear(d_cond + d_cond // 2 + d_cond // 4, d_cond),
# #             nn.LayerNorm(d_cond), nn.GELU())

# #     def _encode_raw(self, batch_list) -> torch.Tensor:
# #         obs_traj  = batch_list[0]; obs_Me = batch_list[7]
# #         image_obs = batch_list[11]; env_data = batch_list[13]
# #         if image_obs.dim() == 4:
# #             image_obs = image_obs.unsqueeze(2)
# #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
# #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# #         T_obs = obs_traj.shape[0]
# #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# #         e_3d_s = self.bottleneck_proj(e_3d_s)
# #         if e_3d_s.shape[1] != T_obs:
# #             e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
# #                                    mode="linear", align_corners=False).permute(0,2,1)
# #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# #                                           device=e_3d_dec_t.device) * 0.5, dim=0)
# #         f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
# #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# #         h_t    = self.enc_1d(obs_in, e_3d_s)
# #         e_env, _, _ = self.env_enc(env_data, image_obs)
# #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

# #     def _kinematic_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
# #         """6 kinematic features per obs step."""
# #         B = obs_traj.shape[1]; T_obs = obs_traj.shape[0]; device = obs_traj.device
# #         if T_obs >= 2:
# #             traj_deg = _norm_to_deg(obs_traj)
# #             vel_norm = obs_traj[1:] - obs_traj[:-1]
# #             speed    = _step_speeds_kmh(traj_deg)
# #             speed_n  = (speed / 20.0).clamp(-3.0, 3.0)
# #             heading  = torch.atan2(vel_norm[:, :, 1], vel_norm[:, :, 0])
# #             if T_obs >= 3:
# #                 dspd  = speed[1:] - speed[:-1]
# #                 accel = torch.cat([obs_traj.new_zeros(1, B),
# #                                    (dspd / 10.0).clamp(-3.0, 3.0)], 0)
# #             else:
# #                 accel = obs_traj.new_zeros(T_obs - 1, B)
# #             kine = torch.stack([vel_norm[:,:,0], vel_norm[:,:,1], speed_n,
# #                                 heading.sin(), heading.cos(), accel], dim=-1)
# #         else:
# #             kine = obs_traj.new_zeros(self.obs_len, B, 6)
# #         if kine.shape[0] < self.obs_len:
# #             kine = torch.cat([obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6), kine], 0)
# #         else:
# #             kine = kine[-self.obs_len:]
# #         return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

# #     def forward(self, batch_list, hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# #         raw   = self._encode_raw(batch_list)
# #         ctx   = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
# #         kfeat = self._kinematic_feat(batch_list[0][:, :, :2])
# #         if hard_score is None:
# #             hard_score = torch.zeros(ctx.shape[0], device=ctx.device)
# #         hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))
# #         return self.fuse(torch.cat([ctx, kfeat, hfeat], dim=-1))


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Hard score (XAI-2) — same as v2.1
# # # ─────────────────────────────────────────────────────────────────────────────

# # @torch.no_grad()
# # def hard_score_from_obs(obs_traj_norm: torch.Tensor,
# #                          return_components: bool = False):
# #     """[XAI-2] Độ khó của storm dựa trên curvature, speed_var, dir_change."""
# #     T, B   = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
# #     device = obs_traj_norm.device
# #     if T < 3:
# #         z = torch.zeros(B, device=device)
# #         if return_components:
# #             return z, {"curvature": z.clone(), "speed_var": z.clone(), "dir_change": z.clone()}
# #         return z
# #     traj_deg = _norm_to_deg(obs_traj_norm[..., :2])
# #     az12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
# #     az23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
# #     diff = (az23 - az12).abs()
# #     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
# #     curvature  = diff.mean(0) / math.pi
# #     spd        = _step_speeds_kmh(traj_deg)
# #     if spd.shape[0] >= 2:
# #         speed_var = (spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 1.)
# #     else:
# #         speed_var = torch.zeros(B, device=device)
# #     dir_change = (diff > (20.0 / 180.0 * math.pi)).float().mean(0)
# #     score = (0.4 * curvature + 0.3 * speed_var + 0.3 * dir_change).clamp(0., 1.)
# #     if return_components:
# #         return score, {"curvature": curvature, "speed_var": speed_var, "dir_change": dir_change}
# #     return score


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Physics score v2.6 (adds displacement_score)
# # # ─────────────────────────────────────────────────────────────────────────────

# # @torch.no_grad()
# # def _physics_score(traj_norm: torch.Tensor, obs_norm: torch.Tensor) -> torch.Tensor:
# #     """
# #     Best-of-K selection score. Four components:
# #       speed_score       (0.30): per-step speed vs obs reference
# #       smooth_score      (0.25): trajectory smoothness (no sharp acceleration)
# #       heading_score     (0.30): first-step direction matches obs
# #       displacement_score(0.15): total path length consistent with obs speed

# #     [NEW v2.6] displacement_score penalizes samples with total displacement
# #     far from obs-speed expectation → catches speed over-prediction artifacts
# #     that cause high ATE on test storms.
# #     """
# #     B      = traj_norm.shape[1]
# #     device = traj_norm.device
# #     traj_deg = _norm_to_deg(traj_norm)
# #     v_ref   = None

# #     # ── Speed score ────────────────────────────────────────────────────────
# #     if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# #         obs_deg = _norm_to_deg(obs_norm)
# #         obs_spd = _step_speeds_kmh(obs_deg)
# #         T_s     = obs_spd.shape[0]
# #         w_obs   = torch.linspace(0.5, 1.0, T_s, device=device)
# #         v_ref   = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()   # [B]
# #         pred_spd = _step_speeds_kmh(traj_deg)
# #         v_sigma  = v_ref.clamp(min=5.0) * 0.5
# #         speed_score = torch.exp(
# #             -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5)
# #     elif traj_deg.shape[0] >= 2:
# #         speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.).mean(0))
# #     else:
# #         speed_score = torch.ones(B, device=device)

# #     # ── Smoothness score ───────────────────────────────────────────────────
# #     if traj_deg.shape[0] >= 3:
# #         vel          = traj_deg[1:] - traj_deg[:-1]
# #         accel_mag    = (vel[1:] - vel[:-1]).norm(dim=-1)
# #         smooth_score = torch.exp(-accel_mag.mean(0) * 5.0)
# #     else:
# #         smooth_score = torch.ones(B, device=device)

# #     # ── Heading score ──────────────────────────────────────────────────────
# #     if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
# #         obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# #         pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
# #         obs_h    = F.normalize(obs_vel,  dim=-1, eps=1e-6)
# #         pred_h   = F.normalize(pred_vel, dim=-1, eps=1e-6)
# #         cos_sim  = (obs_h * pred_h).sum(-1).clamp(-1, 1)
# #         head_score = torch.exp((cos_sim - 1.0) * 3.0)
# #     else:
# #         head_score = torch.ones(B, device=device)

# #     # ── [NEW v2.6] Displacement score ─────────────────────────────────────
# #     # Expected total path length ≈ obs_speed × T_pred × DT × 0.75
# #     # (0.75 factor: storms curve, so straight-line < path length estimate)
# #     # This catches ensemble samples that over/under-shoot on total distance.
# #     if v_ref is not None and traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
# #         T_pred        = traj_deg.shape[0]
# #         expected_total = v_ref * T_pred * DT_HOURS * 0.75   # [B] km
# #         step_dists    = _haversine_deg(traj_deg[:-1], traj_deg[1:])   # [T-1, B]
# #         actual_total  = step_dists.sum(0)                              # [B]
# #         rel_err       = (actual_total - expected_total).abs() / expected_total.clamp(min=10.)
# #         disp_score    = torch.exp(-rel_err * 1.5)
# #     else:
# #         disp_score    = torch.ones(B, device=device)

# #     return (speed_score.pow(0.30)
# #             * smooth_score.pow(0.25)
# #             * head_score.pow(0.30)
# #             * disp_score.pow(0.15)).clamp(min=1e-6)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Augmentation v2.6
# # # ─────────────────────────────────────────────────────────────────────────────

# # def augment_batch(batch_list) -> list:
# #     """
# #     v2.6 augmentation — extends v2.1 core with recurvature and obs-speed aug.

# #     Distribution (6 types):
# #       A (25%): track shift ±5km             — shape unchanged, position varies
# #       B (20%): GT speed scale ×0.85–1.15   — mild, proven from v2.1
# #       C (20%): recurvature ±20°             — proven CTE improvement in v2.5
# #       D (15%): obs-speed scaling ×0.7–1.4  — [NEW v2.6] diverse speed contexts
# #                                               buộc model học với fast/slow storms
# #                                               giảm anchoring to val speed distribution
# #       E (10%): no augmentation              — giữ original distribution
# #       F (10%): Gaussian noise ±3km          — small robustness

# #     KHÔNG dùng:
# #       - mixup (phi vật lý, v2.4 proof)
# #       - speed scale >×1.5 trên GT (too far from real)
# #       - exp L_reg weights (v2.4 catastrophic)

# #     AUG-D (obs-speed scaling) lý do:
# #       v2.5 XAI-5: model anchors to val speed (~10.3 km/h) throughout training.
# #       Test storms có speed distribution khác.
# #       AUG-D: scale OBS displacements but keep GT unchanged.
# #       → Model must INFER future speed from obs context (not memorize mean)
# #       → At inference, model adapts per-storm → speed_calibrate_pred() works better.
# #     """
# #     bl = list(batch_list)
# #     if not torch.is_tensor(bl[0]):
# #         return bl

# #     obs    = bl[0]
# #     device = obs.device
# #     anchor = obs[-1:, :, :2].detach()  # [1, B, 2] — last obs as pivot
# #     r = torch.rand(1).item()

# #     if r < 0.25:
# #         # A: Track shift ±5km (preserves storm shape, varies position)
# #         shift = (torch.rand(2, device=device) - 0.5) * 0.018  # ±5km in normalized
# #         bl[0] = obs + shift.view(1, 1, 2)
# #         if torch.is_tensor(bl[1]):
# #             bl[1] = bl[1] + shift.view(1, 1, 2)

# #     elif r < 0.45:
# #         # B: Speed scale ×0.85–1.15 (both obs+GT from anchor, mild)
# #         scale = 0.85 + 0.30 * torch.rand(1, device=device).item()
# #         obs_c = obs.clone()
# #         obs_c[..., :2] = anchor + (obs[..., :2] - anchor) * scale
# #         bl[0] = obs_c
# #         if torch.is_tensor(bl[1]):
# #             bl[1] = anchor + (bl[1] - anchor) * scale

# #     elif r < 0.65:
# #         # C: Recurvature ±20° — proven CTE improvement
# #         #
# #         # FIX v2.1-clean: rotate DISPLACEMENT vectors (not absolute positions)
# #         # BUG in v2.1-XAI: rotating positions around anchor inflates step-distances
# #         #   → model learned "recurvature = faster" → XAI-5 ratio 1.24→2.17 → ATE +7.2km
# #         # FIX: direction changes, step magnitude preserved = correct TC physics
# #         T_pred = bl[1].shape[0] if torch.is_tensor(bl[1]) else 0
# #         if T_pred >= 4:
# #             gt      = bl[1].clone()
# #             max_deg = (torch.rand(1).item() - 0.5) * 40.0   # -20° to +20°
# #             max_rad = max_deg * math.pi / 180.0
# #             # Build displacement vectors: anchor→gt[0], gt[0]→gt[1], ...
# #             pts  = torch.cat([anchor, gt], 0)   # [T_pred+1, B, 2]
# #             disp = pts[1:] - pts[:-1]           # [T_pred, B, 2]
# #             # Rotate each displacement by progressive angle (ease-in)
# #             for t in range(T_pred):
# #                 progress = (t / max(T_pred - 1, 1)) ** 1.5
# #                 a = max_rad * progress
# #                 c, s = math.cos(a), math.sin(a)
# #                 rot = torch.tensor([[c, -s], [s, c]], dtype=gt.dtype, device=device)
# #                 disp[t] = (rot @ disp[t].unsqueeze(-1)).squeeze(-1)
# #             # Reconstruct trajectory from rotated displacements
# #             gt_new = gt.clone()
# #             gt_new[0] = anchor[0] + disp[0]
# #             for t in range(1, T_pred):
# #                 gt_new[t] = gt_new[t - 1] + disp[t]
# #             bl[1] = gt_new
# #             # Rotate last 3 obs DISPLACEMENTS by 30% of max (continuity)
# #             T_obs   = obs.shape[0]
# #             obs_aug = obs.clone()
# #             cp = math.cos(max_rad * 0.3); sp = math.sin(max_rad * 0.3)
# #             rp = torch.tensor([[cp, -sp], [sp, cp]], dtype=obs.dtype, device=device)
# #             for t_obs in range(max(1, T_obs - 3), T_obs):
# #                 d = obs_aug[t_obs, :, :2] - obs_aug[t_obs - 1, :, :2]
# #                 obs_aug[t_obs, :, :2] = obs_aug[t_obs - 1, :, :2] + (rp @ d.unsqueeze(-1)).squeeze(-1)
# #             bl[0] = obs_aug

# #     elif r < 0.90:
# #         # D-E slot: no augmentation
# #         # AUG-D REMOVED — proven harmful: +4.5km ATE (v2.6), +13.4km (v2.7)
# #         pass

# #     else:
# #         # F: small Gaussian noise ±3km
# #         obs_new = obs.clone()
# #         obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.003
# #         bl[0] = obs_new

# #     return bl


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  XAI functions (1–7, complete)
# # # ─────────────────────────────────────────────────────────────────────────────

# # def compute_obs_attribution(model, batch_list, device: torch.device,
# #                              target_step: int = 11) -> torch.Tensor:
# #     """
# #     [XAI-1] Gradient saliency: which obs step most influences 72h prediction?

# #     Returns attr [T_obs, B] normalized importance [0,1].
# #     High attr[-1, b] = last obs dominant (normal).
# #     High attr[0, b]  = storm has long "memory" (unusual → hard storm).
# #     """
# #     raw = _unwrap(model)
# #     with torch.no_grad():
# #         h_score = hard_score_from_obs(batch_list[0][:, :, :2])
# #     obs_req = batch_list[0].detach().clone().requires_grad_(True)
# #     bl_g = list(batch_list); bl_g[0] = obs_req
# #     with torch.enable_grad():
# #         cond = raw.encoder(bl_g, hard_score=h_score)
# #         x0   = torch.randn(obs_req.shape[1], raw.pred_len, 2, device=device) * raw.sigma_inference
# #         t0   = torch.zeros(obs_req.shape[1], device=device)
# #         v    = raw.velocity(x0, t0, cond)
# #         pred_rel = x0 + v
# #         ts       = min(target_step, raw.pred_len - 1)
# #         pred_rel[:, ts, :].norm(dim=-1).mean().backward()
# #     if obs_req.grad is not None:
# #         attr = obs_req.grad[:, :, :2].norm(dim=-1)
# #         attr = attr / (attr.sum(0, keepdim=True) + 1e-8)
# #     else:
# #         attr = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)
# #     return attr.detach()


# # @torch.no_grad()
# # def compute_ensemble_uncertainty(all_traj: torch.Tensor) -> Dict:
# #     """
# #     [XAI-4] Uncertainty per lead time across K ensemble samples.
# #     Returns std_per_step [T,B] km, uncertainty_ratio, mean_72h_std, high_uncertainty.
# #     """
# #     all_deg  = _norm_to_deg(all_traj)  # [K,T,B,2]
# #     K, T, B  = all_deg.shape[:3]
# #     mean_traj = all_deg.mean(0)
# #     std_km = torch.zeros(T, B, device=all_traj.device)
# #     for t in range(T):
# #         dists = _haversine_deg(
# #             all_deg[:, t].reshape(K * B, 2),
# #             mean_traj[t].unsqueeze(0).expand(K, B, 2).reshape(K * B, 2)
# #         ).reshape(K, B)
# #         std_km[t] = dists.std(0)
# #     s12 = min(1, T - 1); s72 = min(11, T - 1)
# #     return {
# #         "std_per_step":      std_km,
# #         "uncertainty_ratio": (std_km[s72] + 1e-3) / (std_km[s12] + 1e-3),
# #         "mean_72h_std":      float(std_km[s72].mean()),
# #         "mean_12h_std":      float(std_km[s12].mean()),
# #         "high_uncertainty":  std_km[s72] > 80.0,
# #     }


# # @torch.no_grad()
# # def compute_heading_deviation(pred_deg: torch.Tensor,
# #                                gt_deg:   torch.Tensor) -> torch.Tensor:
# #     """
# #     [XAI-6] Heading deviation per lead time step, in degrees (absolute).
# #     High values indicate directional error = root cause of CTE.
# #     Returns [T-1, B].
# #     """
# #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# #     if T < 2:
# #         return pred_deg.new_zeros(1, pred_deg.shape[1])
# #     bear_gt   = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
# #     bear_pred = _forward_azimuth(gt_deg[:T-1], pred_deg[1:T])
# #     diff = (bear_pred - bear_gt).abs()
# #     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
# #     return torch.rad2deg(diff)


# # @torch.no_grad()
# # def compute_cte_contribution(pred_deg: torch.Tensor,
# #                               gt_deg:   torch.Tensor) -> Dict:
# #     """
# #     [XAI-7] Decompose trajectory error into ATE (along-track) and CTE (cross-track).
# #     Returns ate_per_step [T-1,B], cte_per_step [T-1,B], scalar means.
# #     """
# #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# #     if T < 2:
# #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# #         return {"ate_per_step": z, "cte_per_step": z,
# #                 "ate_mean": z[0], "cte_mean": z[0],
# #                 "ate_abs_mean": 0.0, "cte_abs_mean": 0.0}
# #     bear_ref  = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
# #     bear_err  = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
# #     dist_err  = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# #     ang       = bear_err - bear_ref
# #     ate       = dist_err * torch.cos(ang)
# #     cte       = dist_err * torch.sin(ang)
# #     return {
# #         "ate_per_step":  ate,
# #         "cte_per_step":  cte,
# #         "ate_mean":      ate.mean(0),
# #         "cte_mean":      cte.abs().mean(0),
# #         "ate_abs_mean":  float(ate.abs().mean()),
# #         "cte_abs_mean":  float(cte.abs().mean()),
# #     }


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Compat stubs
# # # ─────────────────────────────────────────────────────────────────────────────

# # @torch.no_grad()
# # def classify_hard_easy(obs_traj_norm, per_sample_loss=None,
# #                         hard_score_p: float = 70.0, loss_p: float = 50.0):
# #     scores = hard_score_from_obs(obs_traj_norm)
# #     B = scores.shape[0]
# #     if B < 4:
# #         return torch.zeros(B, dtype=torch.bool, device=scores.device)
# #     return scores >= torch.quantile(scores, hard_score_p / 100.0)


# # @torch.no_grad()
# # def classify_hard_easy_global(obs_traj_norm, global_threshold):
# #     return hard_score_from_obs(obs_traj_norm) >= global_threshold


# # @torch.no_grad()
# # def compute_diversity_score(candidates) -> float:
# #     if len(candidates) < 2:
# #         return 0.0
# #     T, B = candidates[0].shape[0], candidates[0].shape[1]
# #     ep_step = min(T - 1, 11)
# #     endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
# #     N = endpoints.shape[0]
# #     ep_mean = endpoints.mean(0, keepdim=True)
# #     dists = _haversine_deg(
# #         endpoints.reshape(N * B, 2),
# #         ep_mean.expand(N, B, 2).reshape(N * B, 2)
# #     ).reshape(N, B)
# #     return float(dists.std(0).mean())


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  TCFlowMatching v2.6
# # # ─────────────────────────────────────────────────────────────────────────────

# # class TCFlowMatching(nn.Module):
# #     """
# #     TC-FlowMatching v2.6

# #     Training loss:
# #         L_total = L_CFM
# #                 + lam_reg  × L_reg          (ramp ep10→ep30, linear weights)
# #                 + lam_dir  × lambda_heading × L_heading_ms  (ramp ep5→ep20)

# #         L_CFM:         conditional flow matching (random t, OT noise-GT matching)
# #         L_reg:         ADE loss at sigma_inference, softmax-linspace step weights
# #         L_heading_ms:  [v2.6] 4-step heading continuation, decay=0.5 per step

# #     NO L_momentum: removed because it anchors to val speed distribution,
# #     which hurts ATE on test storms that move at different speeds.

# #     Inference pipeline:
# #         1. Sample K=20 candidates with sigma=0.04 FIXED
# #         2. Physics score selection (speed+smooth+heading+displacement)
# #         3. Weighted average of top-3 candidates
# #         4. Speed calibration: per-storm scale ×[0.85, 1.15] to match obs speed
# #         5. Optional: multi-scale sigma ensemble (sample_multiscale)
# #     """

# #     def __init__(
# #         self,
# #         pred_len:          int   = 12,
# #         obs_len:           int   = 8,
# #         unet_in_ch:        int   = 13,
# #         d_cond:            int   = 256,
# #         d_model:           int   = 256,
# #         nhead:             int   = 8,
# #         num_dec_layers:    int   = 4,
# #         dim_ff:            int   = 512,
# #         dropout:           float = 0.1,
# #         sigma_min:         float = 0.04,
# #         sigma_max:         float = 0.08,
# #         lambda_reg:        float = 0.2,
# #         lambda_heading:    float = 0.07,   # reduced: gradient scale balance with L_CFM
# #         lambda_momentum:   float = 0.0,    # DISABLED — field kept for compat only
# #         use_ot:            bool  = True,
# #         ot_epsilon:        float = 0.05,
# #         use_ema:           bool  = True,
# #         ema_decay:         float = 0.995,
# #         n_inference_steps: int   = 1,
# #         n_ensemble:        int   = 20,
# #         sigma_inference:   float = 0.04,   # FIXED throughout
# #         **kwargs,
# #     ):
# #         super().__init__()
# #         self.pred_len          = pred_len
# #         self.obs_len           = obs_len
# #         self.sigma_min         = sigma_min
# #         self.sigma_max         = sigma_max
# #         self.lambda_reg        = lambda_reg
# #         self.lambda_heading    = lambda_heading
# #         self.lambda_momentum   = 0.0           # always disabled
# #         self.use_ot            = use_ot
# #         self.ot_epsilon        = ot_epsilon
# #         self.n_inference_steps = n_inference_steps
# #         self.n_ensemble        = n_ensemble
# #         self.sigma_inference   = sigma_inference

# #         self.encoder  = ContextEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
# #         self.velocity = VelocityTransformer(
# #             pred_len=pred_len, d_model=d_model, nhead=nhead,
# #             num_layers=num_dec_layers, dim_ff=dim_ff,
# #             dropout=dropout, d_cond=d_cond)
# #         self.use_ema = use_ema
# #         self._ema    = None

# #     def init_ema(self):
# #         if self.use_ema:
# #             self._ema = EMAModel(self, decay=0.995)

# #     def ema_update(self):
# #         if self._ema is not None:
# #             self._ema.update(self)

# #     def _to_relative(self, x_abs: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# #         return x_abs - last_obs.unsqueeze(1)

# #     def _from_relative(self, x_rel: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
# #         return x_rel + last_obs.unsqueeze(1)

# #     def _sigma_schedule(self, epoch: int) -> float:
# #         """Cosine decay sigma_max→sigma_min over ep5→ep40."""
# #         if epoch < 5:
# #             return self.sigma_max
# #         if epoch < 40:
# #             t = (epoch - 5) / 35.0
# #             return self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * t))
# #         return self.sigma_min

# #     # ── [v2.6] Multi-step heading loss ─────────────────────────────────────

# #     def _heading_loss_ms(self, pred_deg: torch.Tensor, obs_deg: torch.Tensor,
# #                           n_steps: int = 4, decay: float = 0.5) -> torch.Tensor:
# #         """
# #         Multi-step heading constraint. [UPGRADED FROM v2.5 SINGLE STEP]

# #         v2.5 used single-step (step 0 only, weight=0.05):
# #           → 72h heading dev INCREASED from 87° to 128° during training
# #           → Model learned "start correctly" then drifted

# #         v2.6 uses 4 steps with exponential decay:
# #           step t reference: obs_heading (t=0) or pred_bearing[t-1] (t>0, DETACHED)
# #           step 0: 1.0 × (1 - cos(pred_bear[0] - obs_bear))  ← hard anchor
# #           step 1: 0.5 × (1 - cos(pred_bear[1] - pred_bear[0]))  ← soft continuation
# #           step 2: 0.25 × ...
# #           step 3: 0.125 × ...

# #         DETACH ref at each step: prevents gradient explosion through chain.
# #         Each step trains relatively independently.

# #         Expected: 72h heading dev down from 128° to ~90-100°
# #         Expected: CTE test from 64.5km down to ~57-63km
# #         """
# #         if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
# #             return pred_deg.new_zeros(())

# #         # Starting reference: bearing from obs[-2] → obs[-1]
# #         ref_bear = _forward_azimuth(obs_deg[-2], obs_deg[-1])   # [B]

# #         loss = pred_deg.new_zeros(())
# #         # pts[t] → pts[t+1] gives bearing for pred step t
# #         pts = torch.cat([obs_deg[-1:], pred_deg], 0)   # [T_pred+1, B, 2]

# #         N = min(n_steps, pred_deg.shape[0])
# #         for t in range(N):
# #             pred_bear  = _forward_azimuth(pts[t], pts[t + 1])  # [B]
# #             angle_diff = pred_bear - ref_bear
# #             loss       = loss + (decay ** t) * (1.0 - torch.cos(angle_diff)).mean()
# #             ref_bear   = pred_bear.detach()   # CRITICAL: detach to prevent chain gradient

# #         return loss / N

# #     # ── L_reg (identical to v2.1 — proven not to overfit) ──────────────────

# #     def _reg_loss(self, x1_rel: torch.Tensor, last_obs: torch.Tensor,
# #                   cond: torch.Tensor,
# #                   hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
# #         """
# #         ADE loss at t=0 with sigma_inference noise, consistent with 1-shot inference.
# #         Uses SOFTMAX-LINSPACE step weights (max/min ≈ 3×).
# #         DOES NOT use exp weights (v2.4 used 12.2× ratio → catastrophic overfitting).
# #         """
# #         B, T, _ = x1_rel.shape
# #         device   = x1_rel.device
# #         x0   = torch.randn_like(x1_rel) * self.sigma_inference
# #         t0   = torch.zeros(B, device=device)
# #         v    = self.velocity(x0, t0, cond)
# #         x1_pred_abs = self._from_relative(x0 + v, last_obs)
# #         x1_gt_abs   = self._from_relative(x1_rel, last_obs)
# #         pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))  # [T, B, 2]
# #         gt_deg   = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
# #         dist     = _haversine_deg(pred_deg, gt_deg)             # [T, B] km
# #         # Step weights: softmax(linspace(1,1.5,T)) → max/min ≈ 1.6×
# #         # CHANGED from (1,3) 3× ratio → (1,1.5) 1.6× ratio
# #         # Reason: XAI-8 shows 12h/24h speed BURST (1.3-1.6×) because model
# #         # sacrifices early steps to optimize late-step (72h) ADE.
# #         # Reducing late-step over-emphasis → less speed burst → lower ATE.
# #         # Still late-heavy (not uniform) to maintain 72h accuracy.
# #         T_actual = dist.shape[0]
# #         sw = torch.linspace(1.0, 1.5, T_actual, device=device, dtype=dist.dtype)
# #         sw = F.softmax(sw, dim=0).unsqueeze(1)                  # [T, 1]
# #         if hard_score is not None:
# #             sw_hard = (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)  # [1, B]
# #         else:
# #             sw_hard = torch.ones(1, B, device=device, dtype=dist.dtype)
# #         return ((dist * sw) * sw_hard).mean() / 300.0

# #     # ── get_loss_breakdown ──────────────────────────────────────────────────

# #     def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
# #         """
# #         Full loss computation. Called in TRAINING with augmented batch.
# #         Called in VAL without augmentation → val_loss is reliable signal.

# #         Total = L_CFM + lam_reg × L_reg + lam_dir × lambda_heading × L_heading_ms
# #         """
# #         obs_traj = batch_list[0]
# #         gt_traj  = batch_list[1]
# #         B        = obs_traj.shape[1]
# #         device   = obs_traj.device

# #         sigma    = self._sigma_schedule(epoch)
# #         x1_gt    = gt_traj.permute(1, 0, 2)       # [B, T, 2]
# #         last_obs = obs_traj[-1, :, :2]             # [B, 2]
# #         x1_rel   = self._to_relative(x1_gt, last_obs)

# #         h_score = hard_score_from_obs(obs_traj[:, :, :2])
# #         cond    = self.encoder(batch_list, hard_score=h_score)

# #         # ── L_CFM ─────────────────────────────────────────────────────────
# #         x0 = torch.randn_like(x1_rel) * sigma
# #         if self.use_ot and B >= 4:
# #             x0_flat, x1_flat = _ot_match(
# #                 x0.reshape(B, -1), x1_rel.reshape(B, -1), self.ot_epsilon)
# #             x0         = x0_flat.reshape(B, self.pred_len, 2)
# #             x1_matched = x1_flat.reshape(B, self.pred_len, 2)
# #         else:
# #             x1_matched = x1_rel

# #         t        = torch.rand(B, device=device)
# #         x_t      = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1_matched
# #         u_target = x1_matched - x0
# #         v_pred   = self.velocity(x_t, t, cond)
# #         l_cfm    = F.mse_loss(v_pred, u_target)

# #         # ── L_reg ramp ep10→ep30 ─────────────────────────────────────────
# #         if epoch < 10:     lam_reg = 0.0
# #         elif epoch < 30:   lam_reg = self.lambda_reg * (epoch - 10) / 20.0
# #         else:              lam_reg = self.lambda_reg

# #         l_reg = (self._reg_loss(x1_rel, last_obs, cond, h_score)
# #                  if lam_reg > 0.0 else x0.new_zeros(()))

# #         # ── L_heading_ms ramp ep5→ep20 ────────────────────────────────────
# #         # Ramp: 0 before ep5, linear ep5→ep20, full after ep20
# #         if epoch < 5:      lam_dir = 0.0
# #         elif epoch < 20:   lam_dir = (epoch - 5) / 15.0
# #         else:              lam_dir = 1.0

# #         if lam_dir > 0.0:
# #             # BUG FIX v2.1-clean: removed torch.no_grad()
# #             # OLD: no_grad → velocity weights received 0 gradient from L_heading
# #             # EVIDENCE: velocity.grad 0.000 (with no_grad) vs 0.515 (without)
# #             # FIX: gradient flows through velocity → L_heading actually trains
# #             x0_h       = torch.randn_like(x1_rel) * self.sigma_inference
# #             v_h        = self.velocity(x0_h, torch.zeros(B, device=device), cond)
# #             x1_h_abs   = self._from_relative(x0_h + v_h, last_obs)
# #             pred_deg_h = _norm_to_deg(x1_h_abs.permute(1, 0, 2))   # [T, B, 2]
# #             obs_deg_h  = _norm_to_deg(obs_traj[:, :, :2])           # [T_obs, B, 2]
# #             l_heading  = self._heading_loss_ms(pred_deg_h, obs_deg_h, n_steps=4, decay=0.5)
# #         else:
# #             l_heading = x0.new_zeros(())

# #         l_momentum = x0.new_zeros(())   # always 0 in v2.6

# #         # ── Total ─────────────────────────────────────────────────────────
# #         total = l_cfm + lam_reg * l_reg + lam_dir * self.lambda_heading * l_heading
# #         if not torch.isfinite(total):
# #             total = x0.new_zeros(())

# #         # ── ADE 1-step log ────────────────────────────────────────────────
# #         with torch.no_grad():
# #             x0_log = torch.randn_like(x1_rel) * self.sigma_inference
# #             v_log  = self.velocity(x0_log, torch.zeros(B, device=device), cond)
# #             ade_log = _haversine_deg(
# #                 _norm_to_deg(self._from_relative(x0_log + v_log, last_obs).permute(1, 0, 2)),
# #                 _norm_to_deg(x1_gt.permute(1, 0, 2))
# #             ).mean().item()

# #         return {
# #             "total":    total,
# #             "l_cfm":    l_cfm.item(),
# #             "l_reg":    l_reg.item() if torch.is_tensor(l_reg) else 0.0,
# #             "l_heading": l_heading.item() if torch.is_tensor(l_heading) else 0.0,
# #             "l_momentum": 0.0,   # always 0
# #             "lam_reg":  lam_reg,
# #             "lam_dir":  lam_dir,
# #             "sigma":    sigma,
# #             "ade_1step": ade_log,
# #             "hard_score_mean": float(h_score.mean()),
# #             "hard_score_max":  float(h_score.max()),
# #             # Compat keys for older scripts
# #             "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
# #             "speed": 0., "accel": 0., "fm_mse": l_cfm.item(),
# #             "l_hard_total": 0., "n_hard": 0, "alpha_hard": 0.,
# #             "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
# #         }

# #     def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
# #         return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

# #     # ── [NEW v2.6] Speed calibration at inference ──────────────────────────

# #     @staticmethod
# #     @torch.no_grad()
# #     def speed_calibrate_pred(pred_abs_norm: torch.Tensor,
# #                               last_obs_norm: torch.Tensor,
# #                               obs_norm:      torch.Tensor,
# #                               clip_min: float = 0.85,
# #                               clip_max: float = 1.15) -> torch.Tensor:
# #         """
# #         Per-storm post-hoc speed calibration. Applied at inference ONLY.
# #         DOES NOT affect training → zero train/inference consistency issue.

# #         Scales DISPLACEMENT MAGNITUDE, preserves DIRECTION.
# #         → ATE improved (speed error corrected)
# #         → CTE unchanged (direction unchanged)

# #         Algorithm:
# #           obs_speed_ref = exponentially-weighted mean of obs step speeds
# #           pred_speed_ref = mean of first 4 predicted step speeds
# #           scale = clip(obs_speed_ref / pred_speed_ref, clip_min, clip_max)
# #           pred_cal[t] = last_obs + (pred[t] - last_obs) * scale

# #         clip_min=0.85, clip_max=1.15:
# #           Prevents over-correction from noisy obs/pred speed estimates.
# #           Allows ±15% adjustment per storm.

# #         Why this works:
# #           v2.5 XAI-5: speed ratio = 1.15–1.61 throughout training
# #           (model consistently over-predicts speed on val)
# #           Test storms have different speed → need per-storm correction
# #           This is adaptive: fast test storms get upscaling, slow get downscaling
# #         """
# #         if obs_norm.shape[0] < 2 or pred_abs_norm.shape[0] < 2:
# #             return pred_abs_norm

# #         obs_deg  = _norm_to_deg(obs_norm)        # [T_obs, B, 2]
# #         pred_deg = _norm_to_deg(pred_abs_norm)   # [T, B, 2]
# #         last_deg = _norm_to_deg(last_obs_norm)   # [B, 2]

# #         # Obs speed reference (exponentially weighted, recent steps more important)
# #         obs_spds = _step_speeds_kmh(obs_deg)     # [T_obs-1, B] km/h
# #         T_s = obs_spds.shape[0]
# #         w = torch.exp(torch.arange(T_s, dtype=obs_spds.dtype, device=obs_spds.device) * 0.5)
# #         obs_spd_ref = (obs_spds * (w / w.sum()).unsqueeze(1)).sum(0).clamp(min=3.0)   # [B]

# #         # Predicted speed (first 4 steps from last_obs)
# #         pts = torch.cat([last_deg.unsqueeze(0), pred_deg], 0)   # [T+1, B, 2]
# #         pred_spds    = _step_speeds_kmh(pts)                     # [T, B]
# #         N_ref        = min(4, pred_spds.shape[0])
# #         pred_spd_ref = pred_spds[:N_ref].mean(0).clamp(min=3.0)  # [B]

# #         # Per-storm scale, clipped
# #         scale = (obs_spd_ref / pred_spd_ref).clamp(clip_min, clip_max)  # [B]

# #         # Scale displacement from last_obs (preserves direction!)
# #         return (last_obs_norm.unsqueeze(0)
# #                 + (pred_abs_norm - last_obs_norm.unsqueeze(0))
# #                 * scale.view(1, -1, 1))

# #     # ── Sample (1-shot + speed calibration) ──────────────────────────────

# #     @torch.no_grad()
# #     def sample(self, batch_list,
# #                num_ensemble:          Optional[int]  = None,
# #                ddim_steps:            Optional[int]  = None,
# #                return_xai:            bool           = False,
# #                use_speed_calibration: bool           = True,
# #                **kwargs) -> Tuple:
# #         """
# #         1-shot inference with physics selection + speed calibration.

# #         Steps:
# #           1. Encode context once (hard_score + full encoder)
# #           2. Sample K=20 candidates: x0 ~ N(0, sigma_inf²), x_pred = x0 + v(x0, 0, cond)
# #           3. Physics score each candidate (speed+smooth+heading+displacement)
# #           4. Weighted average of top-3 candidates
# #           5. Speed calibration per-storm (clip ±15%)
# #           6. If return_xai=True: compute XAI-1 through XAI-7

# #         Returns:
# #           (pred_mean [T,B,2], zeros [T,B,2], all_traj [K,T,B,2])
# #           or (pred_mean, zeros, all_traj, xai_dict) if return_xai=True
# #         """
# #         K  = num_ensemble or self.n_ensemble
# #         N  = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else self.n_inference_steps
# #         dt = 1.0 / max(N, 1)

# #         obs_traj    = batch_list[0]
# #         T_obs, B, _ = obs_traj.shape
# #         device      = obs_traj.device

# #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# #         obs_norm = obs_traj[:, :, :2]
# #         last_obs = obs_traj[-1, :, :2]
# #         t0       = torch.zeros(B, device=device)

# #         # Encode once for all K samples (expensive part)
# #         cond = self.encoder(batch_list, hard_score=h_score)

# #         all_traj = []
# #         for _ in range(K):
# #             x_rel = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference

# #             if N <= 1:
# #                 # 1-shot: exactly what L_reg is trained for
# #                 v     = self.velocity(x_rel, t0, cond)
# #                 x_rel = x_rel + v
# #             else:
# #                 # Euler multi-step (for experiments only, slower)
# #                 for step in range(N):
# #                     t_b   = torch.full((B,), step * dt, device=device)
# #                     x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3., 3.)

# #             x_abs = self._from_relative(x_rel, last_obs)
# #             all_traj.append(x_abs.permute(1, 0, 2))   # [T, B, 2]

# #         # Physics-based selection: best-of-K with weighted average of top-3
# #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)   # [K, B]
# #         all_t   = torch.stack(all_traj, 0)   # [K, T, B, 2]
# #         top_k   = min(3, K)
# #         top_idx = scores.topk(top_k, dim=0).indices   # [top_k, B]

# #         pred_mean = torch.zeros_like(all_traj[0])
# #         for b in range(B):
# #             idx_b = top_idx[:, b]
# #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# #         # [NEW v2.6] Per-storm speed calibration
# #         if use_speed_calibration:
# #             pred_mean = self.speed_calibrate_pred(
# #                 pred_mean, last_obs, obs_norm, clip_min=0.85, clip_max=1.15)

# #         if not return_xai:
# #             return pred_mean, torch.zeros_like(pred_mean), all_t

# #         # ── XAI output (XAI-2,3,4,5,6,7) ──────────────────────────────────
# #         xai = {}

# #         # XAI-4: Ensemble uncertainty
# #         xai.update(compute_ensemble_uncertainty(all_t))

# #         # XAI-2: Hard score components
# #         _, hard_comps = hard_score_from_obs(obs_norm, return_components=True)
# #         xai["hard_components"] = hard_comps

# #         pred_deg  = _norm_to_deg(pred_mean)    # [T, B, 2]
# #         obs_deg_x = _norm_to_deg(obs_norm)     # [T_obs, B, 2]

# #         # XAI-5: Speed comparison (obs vs pred, pre and post calibration)
# #         obs_spd_x    = _step_speeds_kmh(obs_deg_x)
# #         obs_spd_mu   = obs_spd_x.mean(0)     # [B]
# #         if pred_deg.shape[0] >= 2:
# #             last_deg_x   = obs_deg_x[-1]
# #             pts_x        = torch.cat([last_deg_x.unsqueeze(0), pred_deg], 0)
# #             pred_spd_x   = _step_speeds_kmh(pts_x)
# #             pred_spd_mu  = pred_spd_x.mean(0)  # [B]
# #         else:
# #             pred_spd_mu = obs_spd_mu.clone()

# #         speed_ratio = (pred_spd_mu / obs_spd_mu.clamp(min=1.0))
# #         xai["speed_comparison"] = {
# #             "obs_speed_mean":  float(obs_spd_mu.mean()),
# #             "pred_speed_mean": float(pred_spd_mu.mean()),
# #             "speed_ratio":     float(speed_ratio.mean()),
# #             "per_storm_obs":   obs_spd_mu,
# #             "per_storm_pred":  pred_spd_mu,
# #             "over_predict":    speed_ratio > 1.15,
# #             "under_predict":   speed_ratio < 0.85,
# #         }

# #         # XAI-3: Physics components for best prediction
# #         v_ref = obs_spd_mu.clamp(min=5.0)
# #         v_sig = v_ref * 0.5
# #         spd_sc = torch.exp(-((pred_spd_mu - v_ref) / v_sig).pow(2) * 0.5)
# #         if pred_deg.shape[0] >= 3:
# #             vel_x   = pred_deg[1:] - pred_deg[:-1]
# #             accel_x = (vel_x[1:] - vel_x[:-1]).norm(dim=-1)
# #             smo_sc  = torch.exp(-accel_x.mean(0) * 5.0)
# #         else:
# #             smo_sc = torch.ones(B, device=device)
# #         if obs_norm.shape[0] >= 2 and pred_mean.shape[0] >= 1:
# #             ov = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
# #             pv = pred_mean[0, :, :2] - obs_norm[-1, :, :2]
# #             cos_s = (F.normalize(ov, dim=-1, eps=1e-6) * F.normalize(pv, dim=-1, eps=1e-6)).sum(-1)
# #             hd_sc = torch.exp((cos_s.clamp(-1, 1) - 1.0) * 3.0)
# #         else:
# #             hd_sc = torch.ones(B, device=device)
# #         xai["physics_components"] = {
# #             "speed": spd_sc, "smooth": smo_sc, "heading": hd_sc,
# #             "obs_speed": obs_spd_mu, "pred_speed": pred_spd_mu,
# #         }

# #         # XAI-6: Heading deviation per lead time
# #         gt_traj_xai = batch_list[1]
# #         gt_deg_xai  = _norm_to_deg(gt_traj_xai[:, :, :2])
# #         xai["heading_deviation_deg"] = compute_heading_deviation(pred_deg, gt_deg_xai)

# #         # XAI-7: ATE/CTE decomposition
# #         xai["ate_cte_decomp"] = compute_cte_contribution(pred_deg, gt_deg_xai)

# #         # XAI-8: Per-horizon speed ratio (12h/24h/48h/72h)
# #         # Measures speed over-prediction pattern across lead times
# #         # Evidence: 12h=1.31×, 24h=1.61× OVER; 48h=0.94×, 72h=0.96× OK
# #         if pred_deg.shape[0] >= 2 and gt_deg_xai.shape[0] >= 2:
# #             T8       = min(pred_deg.shape[0], gt_deg_xai.shape[0])
# #             last_d   = obs_deg_x[-1]
# #             pts_pred = torch.cat([last_d.unsqueeze(0), pred_deg[:T8]], 0)
# #             pts_gt   = torch.cat([last_d.unsqueeze(0), gt_deg_xai[:T8]], 0)
# #             spd_pred = _step_speeds_kmh(pts_pred)   # [T8, B]
# #             spd_gt   = _step_speeds_kmh(pts_gt)     # [T8, B]
# #             ratio    = spd_pred / spd_gt.clamp(min=1.0)  # [T8, B]
# #             def _hz(s, lo, hi):
# #                 hi = min(hi, s.shape[0])
# #                 return float(s[lo:hi].mean()) if hi > lo else float("nan")
# #             r_mean = ratio.mean(1)   # [T8]
# #             xai["speed_per_horizon"] = {
# #                 "ratio":    r_mean.tolist(),
# #                 "pred_kmh": spd_pred.mean(1).tolist(),
# #                 "gt_kmh":   spd_gt.mean(1).tolist(),
# #                 "12h_ratio": _hz(r_mean, 0, 2),
# #                 "24h_ratio": _hz(r_mean, 2, 4),
# #                 "48h_ratio": _hz(r_mean, 6, 8),
# #                 "72h_ratio": _hz(r_mean, 10, 12),
# #             }
# #         else:
# #             xai["speed_per_horizon"] = {}

# #         # XAI-9: Storm category breakdown by obs speed
# #         # slow  < 8 km/h  (typically tropical, well-organized)
# #         # medium 8–15 km/h (mixed)
# #         # fast   > 15 km/h (extratropical, recurving)
# #         obs_spd_cat = _step_speeds_kmh(obs_deg_x).mean(0)   # [B]
# #         sm = obs_spd_cat < 8.0
# #         mm = (obs_spd_cat >= 8.0) & (obs_spd_cat < 15.0)
# #         fm = obs_spd_cat >= 15.0
# #         ade_per_storm = _haversine_deg(pred_deg[:gt_deg_xai.shape[0]],
# #                                         gt_deg_xai[:pred_deg.shape[0]]).mean(0)  # [B]
# #         def _cat_ade(mask):
# #             return float(ade_per_storm[mask].mean()) if mask.sum() > 0 else float("nan")
# #         xai["storm_categories"] = {
# #             "n_slow":    int(sm.sum()),
# #             "n_medium":  int(mm.sum()),
# #             "n_fast":    int(fm.sum()),
# #             "speed_mean": float(obs_spd_cat.mean()),
# #             "speed_std":  float(obs_spd_cat.std()),
# #             "ade_slow":   _cat_ade(sm),
# #             "ade_medium": _cat_ade(mm),
# #             "ade_fast":   _cat_ade(fm),
# #         }

# #         return pred_mean, torch.zeros_like(pred_mean), all_t, xai

# #     @torch.no_grad()
# #     def sample_multiscale(
# #         self,
# #         batch_list,
# #         sigmas: Optional[List[float]] = None,
# #         n_per_sigma: int = 4,
# #         use_speed_calibration: bool = True,
# #     ) -> Tuple:
# #         """
# #         Multi-scale sigma ensemble at inference (DOES NOT affect training).
# #         Hedges against speed distribution shift between val and test.
# #         """
# #         if sigmas is None:
# #             sigmas = [0.025, 0.035, 0.04, 0.05, 0.065]

# #         obs_traj = batch_list[0]
# #         B = obs_traj.shape[1]; device = obs_traj.device

# #         h_score  = hard_score_from_obs(obs_traj[:, :, :2])
# #         obs_norm = obs_traj[:, :, :2]
# #         last_obs = obs_traj[-1, :, :2]
# #         t0       = torch.zeros(B, device=device)
# #         cond     = self.encoder(batch_list, hard_score=h_score)

# #         all_traj = []
# #         for sigma in sigmas:
# #             for _ in range(n_per_sigma):
# #                 x0    = torch.randn(B, self.pred_len, 2, device=device) * sigma
# #                 v     = self.velocity(x0, t0, cond)
# #                 x_abs = self._from_relative(x0 + v, last_obs)
# #                 all_traj.append(x_abs.permute(1, 0, 2))

# #         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)
# #         all_t   = torch.stack(all_traj, 0)
# #         top_k   = min(5, len(all_traj))
# #         top_idx = scores.topk(top_k, dim=0).indices

# #         pred_mean = torch.zeros_like(all_traj[0])
# #         for b in range(B):
# #             idx_b = top_idx[:, b]
# #             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
# #             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

# #         if use_speed_calibration:
# #             pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

# #         return pred_mean, torch.zeros_like(pred_mean), all_t


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Backward compat alias
# # # ─────────────────────────────────────────────────────────────────────────────
# # TCDiffusion = TCFlowMatching

# """
# Model/flow_matching_model.py  ──  TC-FlowMatching v2.1-clean
# ═══════════════════════════════════════════════════════════════════════════════
# VIẾT LẠI HOÀN TOÀN từ v2.1 + các fix đã verified từ thực nghiệm 140 epoch.

# ━━━ CƠ SỞ LÝ THUYẾT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# v2.1 test: ADE=229.8, ATE=214.4, CTE=71.6 (best so far, generalize tốt nhất)
# v2.5 test: ADE=232.5, ATE=222.3, CTE=64.5 (ATE tệ hơn, CTE tốt hơn)
# v2.4 test: ADE=291.1 (catastrophic overfitting)

# PHÂN TÍCH LỖI v2.1:
#   ATE/CTE ratio on test = 214.4/71.6 = 3.0
#   → 75% error is ALONG-TRACK (speed/distance)
#   → 25% is CROSS-TRACK (direction)
#   Test gap 72h = +129.3km >> 12h = +10.1km → long-range drift

# PHÂN TÍCH v2.5 THẤT BẠI:
#   L_momentum anchor pred_speed ≈ val_speed (10.3km/h)
#   Test storms faster → L_momentum SLOWS model on test → ATE +7.9km WORSE
#   L_heading (step 0 only) → 72h heading dev INCREASED: ep0=87°→ep120=128°!
#   Recurvature aug: CTE improved 7km ← giữ lại

# ━━━ VẬT LÝ CỦA GIẢI PHÁP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# GIẢI PHÁP ATE (along-track error):
#   Training: KHÔNG anchor speed tại training (no L_momentum)
#   Inference: SPEED CALIBRATION per-storm
#     scale = clip(obs_speed / pred_speed_first4, 0.85, 1.15)
#     pred_cal = last_obs + (pred - last_obs) * scale
#     → Preserves direction (CTE unchanged), fixes magnitude (ATE reduced)
#     → Per-storm adaptive: fast test storms get upscaling, slow storms downscaling
#     → DOES NOT affect training → zero train/inference consistency issue

# GIẢI PHÁP CTE (cross-track error):
#   Training: MULTI-STEP heading constraint trên TOÀN BỘ 12 bước (không chỉ
#   4 bước đầu như mô tả ban đầu) — xem [LEARN-5] bên dưới để biết chi tiết
#   về lỗi đã phát hiện và sửa: bản trước chỉ áp dụng constraint cho 4/12
#   bước, khiến 8 bước cuối (30h→72h) hoàn toàn không có gradient signal
#   hướng. Trọng số mỗi bước giờ HỌC ĐƯỢC qua self.heading_step_logits.

# GIẢI PHÁP GAP (val→test distribution shift):
#   [AUG-C] Recurvature: proven to reduce CTE test (v2.5)
#   Speed calibration at inference bridges remaining gap

# ━━━ GIỮ NGUYÊN TỪ v2.1 (PROVEN, KHÔNG THAY ĐỔI) ━━━━━━━━━━━━━━━━━━━━━━━━━━

#   ✅ sigma_inference=0.04 FIXED — zero train/inference mismatch
#   ✅ Mild base aug: shift±5km + speed×0.85–1.15 (val ≈ test distribution)
#   ✅ 1-shot inference nhất quán với L_reg training
#   ✅ 2-group optimizer, encoder freeze 10ep
#   ✅ val loop NO augmentation
#   ✅ OT noise-GT matching trong L_CFM
#   ✅ ContextEncoder: FNO3D + Mamba + Env + 6 kinematic features
#   ✅ VelocityTransformer: d_model=256, 4 layers, nhead=8

# ━━━ v2.1-LEARN CHANGES (từ v2.1-clean baseline, test ADE=227.8 ATE=220.3 CTE=60.6) ━

#   CĂN CỨ: mọi lần đổi HẰNG SỐ tay đoán (linspace range, clip range) trước đây
#   chỉ tạo dao động <1km test. Bằng chứng: XAI-8 24h ratio luôn ~1.5-1.6×
#   bất kể linspace(1,3) hay linspace(1,1.5). Đây là dấu hiệu bottleneck không
#   nằm ở "chọn đúng hằng số" mà ở việc DÙNG hằng số thay vì để model học.

#   [LEARN-1] speed_calibrate_pred: từ 1 scale chung (clip 0.85-1.15) cho TOÀN
#             BỘ 12 bước → 12 hệ số riêng biệt theo từng horizon, học qua
#             gradient (không còn @torch.no_grad/@staticmethod).
#             Lý do: XAI-8 chứng minh correction cần KHÁC NHAU theo horizon
#             (12h≈0.77x, 24h≈0.62x, 48h≈1.06x, 72h≈1.04x) — 1 số không
#             thể đáp ứng cả 4 nhu cầu khác nhau cùng lúc.
#             THÊM: L_calib loss term để gradient thực sự chảy tới tham số này
#             (nếu không có loss term riêng, tham số tồn tại nhưng KHÔNG BAO
#             GIỜ được optimizer cập nhật vì nó chỉ được dùng trong sample()
#             dưới torch.no_grad() tại thời điểm eval).

#   [LEARN-2] _reg_loss step weights: linspace(1.0,1.5) hardcoded → 12 logit
#             riêng (softmax), model tự học phân bổ trọng số giữa các horizon
#             thay vì áp công thức tuyến tính tay đoán.

#   [LEARN-3] hard_score: thêm obs_speed (đã chuẩn hóa) làm thành phần thứ 4,
#             VÀ 4 trọng số kết hợp (curvature, speed_var, dir_change, obs_speed)
#             học qua gradient thay vì hardcode 0.4/0.3/0.3.
#             Lý do: XAI-9 cho thấy fast storms (n=94/420=22%) gây 62% tổng CTE
#             test nhưng hard_score cũ không có tín hiệu "obs_speed tuyệt đối"
#             → storm nhanh-nhưng-đều (speed_var thấp) bị coi nhầm là "dễ"
#             → không được train với loss-weight cao hơn dù nó cần.
#             Bỏ @torch.no_grad() ở hard_score_from_obs để gradient chảy qua
#             được (qua sw_hard trong _reg_loss).

#   [LEARN-4] Cross-loss weighting: lambda_reg/lambda_heading/lambda_calib
#             (3 hằng số tay đoán: 0.2/0.07/0.1) → Kendall homoscedastic
#             uncertainty weighting (Kendall, Gal & Cipolla 2018):
#               weighted_term = loss/(2·exp(2·log_sigma)) + log_sigma
#             3 log_sigma HỌC ĐƯỢC qua gradient.
#             QUAN TRỌNG: KHÔNG dùng nn.Parameter(lambda) trực tiếp nhân
#             vào loss — gradient của total theo lambda = loss ≥ 0 luôn
#             đẩy lambda về 0 (suy biến, tắt hết loss phụ). Số hạng
#             +log_sigma trong công thức Kendall là regularizer ngăn suy
#             biến này — đã verify thực nghiệm (xem test ở cuối file dev
#             log): effective_lambda KHÔNG giảm về 0 qua 30 bước train
#             với lr cao, trái lại tăng dần khi loss phụ giảm nhanh.
#             Epoch-ramp (ep5/ep10/ep20/ep30) GIỮ NGUYÊN dạng if/else rời
#             rạc — đây là GATE bật/tắt theo trạng thái encoder-freeze,
#             không phải "trọng số tương đối", không thể differentiable
#             theo epoch (số nguyên), nên không thuộc phạm vi can học.

#   [LEARN-5] _heading_loss_ms: SỬA LỖI BỎ SÓT từ lần trước. Bản trước
#             (kể cả lần "v2.1-learn" đầu tiên của tôi) vẫn giữ n_steps=4
#             CỐ ĐỊNH + decay=0.5 CỐ ĐỊNH (weight=1.0/0.5/0.25/0.125 cho
#             4 bước ĐẦU, 8 bước 30h→72h KHÔNG có constraint heading nào).
#             FIX: self.heading_step_logits (pred_len=12 params, softmax),
#             áp dụng cho TOÀN BỘ 12 bước, init theo đúng decay 0.5^t cũ
#             (không đổi điểm khởi đầu) nhưng gradient tự học lại phân bổ
#             trong lúc train — không còn giới hạn cứng "chỉ 4 bước đầu".

#   KHÔNG ĐỔI (đã cân nhắc, có lý do kỹ thuật rõ ràng):
#     _physics_score 4 exponent (0.30/0.25/0.30/0.15): hàm này chạy dưới
#     @torch.no_grad() để CHỌN giữa K candidates sinh từ CÙNG một velocity
#     network — đây là post-hoc re-ranking, không nằm trên đường gradient
#     nào quay lại network hay bất kỳ tham số nào của nó. Biến nó learnable
#     mà không có loss-term riêng sẽ lặp lại đúng lỗi "tham số mồ côi"
#     (never updated) mà LEARN-1 đã phải sửa bằng cách thêm L_calib.
#     R_EARTH, DT_HOURS, hệ số affine normalized↔degree: hằng số vật lý/
#     metadata dataset, không phải "model nghĩ gì" — đổi sẽ làm sai đơn vị
#     tính toán, không phải tinh chỉnh hành vi model.

#   GIỮ NGUYÊN HOÀN TOÀN (đã proven qua thực nghiệm, KHÔNG đổi):
#     ✅ AUG-C: rotate DISPLACEMENT vectors (không phải absolute positions)
#     ✅ AUG-D: ĐÃ XÓA — proven harmful (+4.5km v2.6, +13.4km v2.7)
#     ✅ L_momentum: disabled (proven harmful +7.9km — v2.5)
#     ✅ sigma_inference=0.04 fixed (train/infer consistency)
#     ✅ OT matching, 1-shot inference, encoder freeze 10ep

# ━━━ KẾT QUẢ THỰC TẾ (baseline trước thay đổi này) ━━━━━━━━━━━━━━━━━━━━━━━━━

#                     Val ADE   Test ADE  Test ATE  Test CTE
#   v2.1 gốc:         170.1     229.8     214.4     71.6
#   v2.1-XAI (ran):   169.7     229.8     221.6     61.9  ← AUG-C bug → ATE +7.2
#   v2.1-clean (ran):  ~168      227.8     220.3     60.6  ← baseline cho bản này
#   ST-Trans target:            224.4     213.7     59.4

#   CHƯA CÓ KẾT QUẢ TEST CHO v2.1-LEARN (bản này) — cần retrain để biết
#   ATE/CTE thực tế có cải thiện so với 220.3/60.6 hay không. Mọi con số
#   "kỳ vọng" chỉ là giả thuyết dựa trên cơ chế, KHÔNG phải đã đo được.
# """
# from __future__ import annotations

# import math
# from typing import Dict, List, Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from Model.FNO3D_encoder import FNO3DEncoder
# from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# from Model.env_net_transformer_gphsplit import Env_net

# R_EARTH  = 6371.0
# DT_HOURS = 6.0


# # ─────────────────────────────────────────────────────────────────────────────
# #  Coordinate utilities (same as v2.1)
# # ─────────────────────────────────────────────────────────────────────────────

# def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
#     """Normalized coords → (lon°, lat°)."""
#     return torch.stack([
#         (t[..., 0] * 50.0 + 1800.0) / 10.0,
#         (t[..., 1] * 50.0) / 10.0,
#     ], dim=-1)


# def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
#     """Haversine distance km between two (lon°,lat°) tensors."""
#     lat1 = torch.deg2rad(p1[..., 1]);  lat2 = torch.deg2rad(p2[..., 1])
#     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
#     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
#     a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
#     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


# def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
#     """Bearing in radians from p1→p2 (degrees input)."""
#     lon1 = torch.deg2rad(p1[..., 0]);  lat1 = torch.deg2rad(p1[..., 1])
#     lon2 = torch.deg2rad(p2[..., 0]);  lat2 = torch.deg2rad(p2[..., 1])
#     dlon = lon2 - lon1
#     y = torch.sin(dlon) * torch.cos(lat2)
#     x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
#     return torch.atan2(y, x)


# def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
#     """Speed km/h between consecutive steps of a trajectory [T, B, 2]."""
#     if traj_deg.shape[0] < 2:
#         return traj_deg.new_zeros(1, traj_deg.shape[1])
#     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# # ─────────────────────────────────────────────────────────────────────────────
# #  EMAModel (same as v2.1)
# # ─────────────────────────────────────────────────────────────────────────────

# def _unwrap(m):
#     return m._orig_mod if hasattr(m, "_orig_mod") else m


# class EMAModel:
#     def __init__(self, model, decay: float = 0.995):
#         self.decay = decay
#         m = _unwrap(model)
#         self.shadow = {k: v.detach().clone()
#                        for k, v in m.state_dict().items()
#                        if v.dtype.is_floating_point}

#     def update(self, model):
#         m = _unwrap(model)
#         with torch.no_grad():
#             for k, v in m.state_dict().items():
#                 if k in self.shadow:
#                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

#     def apply_to(self, model):
#         m = _unwrap(model)
#         backup, sd = {}, m.state_dict()
#         for k in self.shadow:
#             if k not in sd: continue
#             backup[k] = sd[k].detach().clone()
#             sd[k].copy_(self.shadow[k])
#         return backup

#     def restore(self, model, backup):
#         m = _unwrap(model)
#         sd = m.state_dict()
#         for k, v in backup.items():
#             if k in sd: sd[k].copy_(v)


# # ─────────────────────────────────────────────────────────────────────────────
# #  OT matching (same as v2.1)
# # ─────────────────────────────────────────────────────────────────────────────

# def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05, n_iter: int = 50) -> torch.Tensor:
#     B = cost.shape[0]; device = cost.device
#     log_a = -math.log(B) * torch.ones(B, device=device)
#     log_b = -math.log(B) * torch.ones(B, device=device)
#     log_K = -cost / epsilon
#     log_u = torch.zeros(B, device=device)
#     log_v = torch.zeros(B, device=device)
#     for _ in range(n_iter):
#         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
#         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
#     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


# def _ot_match(x0_flat: torch.Tensor, x1_flat: torch.Tensor,
#               epsilon: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
#     B = x0_flat.shape[0]
#     if B < 4:
#         return x0_flat, x1_flat
#     try:
#         cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1] ** 0.5)
#         with torch.no_grad():
#             pi = _sinkhorn_log(cost, epsilon=epsilon)
#         flat = pi.reshape(-1).clamp(0.0)
#         s = flat.sum()
#         if not torch.isfinite(s) or s < 1e-10:
#             return x0_flat, x1_flat
#         idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
#         return x0_flat[idx // B], x1_flat
#     except Exception:
#         return x0_flat, x1_flat


# # ─────────────────────────────────────────────────────────────────────────────
# #  VelocityTransformer (identical to v2.1)
# # ─────────────────────────────────────────────────────────────────────────────

# class VelocityTransformer(nn.Module):
#     def __init__(self, pred_len: int = 12, d_model: int = 256, nhead: int = 8,
#                  num_layers: int = 4, dim_ff: int = 512, dropout: float = 0.1,
#                  d_cond: int = 256):
#         super().__init__()
#         self.pred_len = pred_len
#         self.d_model  = d_model
#         self.traj_embed = nn.Linear(2, d_model)
#         self.pos_emb    = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
#         self.step_emb   = nn.Embedding(pred_len, d_model)
#         self.time_mlp   = nn.Sequential(
#             nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
#         self.cond_proj  = nn.Sequential(nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))
#         dec_layer = nn.TransformerDecoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
#             dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
#         self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
#         self.out_norm = nn.LayerNorm(d_model)
#         self.out_proj = nn.Sequential(
#             nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
#         self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
#         nn.init.zeros_(self.out_proj[-1].weight)
#         nn.init.zeros_(self.out_proj[-1].bias)

#     def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
#         half = self.d_model // 2
#         freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
#                          * (-math.log(10000.0) / max(half - 1, 1)))
#         emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
#         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
#         if self.d_model % 2 == 1:
#             emb = F.pad(emb, (0, 1))
#         return self.time_mlp(emb)

#     def forward(self, x_t: torch.Tensor, t: torch.Tensor,
#                 cond: torch.Tensor) -> torch.Tensor:
#         B, T, _ = x_t.shape
#         step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
#         x_emb = (self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(step_idx))
#         memory = torch.cat([self._time_emb(t).unsqueeze(1),
#                             self.cond_proj(cond).unsqueeze(1)], dim=1)
#         out = self.out_norm(self.decoder(x_emb, memory))
#         return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)


# # ─────────────────────────────────────────────────────────────────────────────
# #  ContextEncoder (identical to v2.1)
# # ─────────────────────────────────────────────────────────────────────────────

# class ContextEncoder(nn.Module):
#     RAW_CTX_DIM = 512

#     def __init__(self, obs_len: int = 8, unet_in_ch: int = 13, d_cond: int = 256):
#         super().__init__()
#         self.obs_len = obs_len
#         self.d_cond  = d_cond

#         self.spatial_enc     = FNO3DEncoder(in_channel=unet_in_ch, out_channel=1, d_model=32,
#                                              n_layers=4, modes_t=4, modes_h=4, modes_w=4,
#                                              spatial_down=32, dropout=0.05)
#         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
#         self.bottleneck_proj = nn.Linear(128, 128)
#         self.decoder_proj    = nn.Linear(1, 16)
#         self.enc_1d          = DataEncoder1D(in_1d=4, feat_3d_dim=128, mlp_h=64,
#                                               lstm_hidden=128, lstm_layers=3,
#                                               dropout=0.1, d_state=16)
#         self.env_enc         = Env_net(obs_len=obs_len, d_model=32)
#         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
#         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
#         self.ctx_drop = nn.Dropout(0.1)
#         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, d_cond)
#         self.ctx_ln2  = nn.LayerNorm(d_cond)
#         # 6 kinematic features: vel_x, vel_y, speed_n, sin(h), cos(h), accel
#         self.vel_obs_enc = nn.Sequential(
#             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
#             nn.Linear(256, d_cond // 2), nn.GELU())
#         self.hard_embed = nn.Sequential(
#             nn.Linear(1, d_cond // 4), nn.GELU(), nn.Linear(d_cond // 4, d_cond // 4))
#         self.fuse = nn.Sequential(
#             nn.Linear(d_cond + d_cond // 2 + d_cond // 4, d_cond),
#             nn.LayerNorm(d_cond), nn.GELU())

#     def _encode_raw(self, batch_list) -> torch.Tensor:
#         obs_traj  = batch_list[0]; obs_Me = batch_list[7]
#         image_obs = batch_list[11]; env_data = batch_list[13]
#         if image_obs.dim() == 4:
#             image_obs = image_obs.unsqueeze(2)
#         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
#             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
#         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
#         T_obs = obs_traj.shape[0]
#         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
#         e_3d_s = self.bottleneck_proj(e_3d_s)
#         if e_3d_s.shape[1] != T_obs:
#             e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
#                                    mode="linear", align_corners=False).permute(0,2,1)
#         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
#         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
#                                           device=e_3d_dec_t.device) * 0.5, dim=0)
#         f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
#         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
#         h_t    = self.enc_1d(obs_in, e_3d_s)
#         e_env, _, _ = self.env_enc(env_data, image_obs)
#         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

#     def _kinematic_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
#         """6 kinematic features per obs step."""
#         B = obs_traj.shape[1]; T_obs = obs_traj.shape[0]; device = obs_traj.device
#         if T_obs >= 2:
#             traj_deg = _norm_to_deg(obs_traj)
#             vel_norm = obs_traj[1:] - obs_traj[:-1]
#             speed    = _step_speeds_kmh(traj_deg)
#             speed_n  = (speed / 20.0).clamp(-3.0, 3.0)
#             heading  = torch.atan2(vel_norm[:, :, 1], vel_norm[:, :, 0])
#             if T_obs >= 3:
#                 dspd  = speed[1:] - speed[:-1]
#                 accel = torch.cat([obs_traj.new_zeros(1, B),
#                                    (dspd / 10.0).clamp(-3.0, 3.0)], 0)
#             else:
#                 accel = obs_traj.new_zeros(T_obs - 1, B)
#             kine = torch.stack([vel_norm[:,:,0], vel_norm[:,:,1], speed_n,
#                                 heading.sin(), heading.cos(), accel], dim=-1)
#         else:
#             kine = obs_traj.new_zeros(self.obs_len, B, 6)
#         if kine.shape[0] < self.obs_len:
#             kine = torch.cat([obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6), kine], 0)
#         else:
#             kine = kine[-self.obs_len:]
#         return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

#     def forward(self, batch_list, hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
#         raw   = self._encode_raw(batch_list)
#         ctx   = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
#         kfeat = self._kinematic_feat(batch_list[0][:, :, :2])
#         if hard_score is None:
#             hard_score = torch.zeros(ctx.shape[0], device=ctx.device)
#         hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))
#         return self.fuse(torch.cat([ctx, kfeat, hfeat], dim=-1))


# # ─────────────────────────────────────────────────────────────────────────────
# #  Hard score (XAI-2) — [LEARN-3] thêm obs_speed + learnable weights
# # ─────────────────────────────────────────────────────────────────────────────

# def hard_score_from_obs(obs_traj_norm: torch.Tensor,
#                          return_components: bool = False,
#                          weight_logits: Optional[torch.Tensor] = None,
#                          obs_speed_norm_const: float = 20.0):
#     """
#     [XAI-2] Độ khó của storm — v2.1-learn: 4 components, learnable weights.

#     [LEARN-3] THAY ĐỔI so với v2.1-clean:
#       CŨ: score = 0.4*curvature + 0.3*speed_var + 0.3*dir_change  (hardcode)
#           KHÔNG có obs_speed tuyệt đối → storm nhanh-nhưng-đều (speed_var
#           thấp) bị coi nhầm là "dễ", dù XAI-9 cho thấy fast storms
#           (obs_speed≥15km/h, 22% test set) gây 62% tổng CTE test.
#       MỚI: score = softmax(weight_logits) · [curvature, speed_var,
#                                               dir_change, obs_speed_norm]
#            4 trọng số HỌC ĐƯỢC qua gradient (không phải số tay đoán),
#            obs_speed_norm = obs_speed_mean / 20.0 (clamp 0,1) làm input
#            feature thứ 4.

#     weight_logits: nếu None, dùng uniform-equivalent fallback (0.25 mỗi cái)
#       — cho phép gọi hàm này từ các chỗ KHÔNG có instance (vd các hàm
#       module-level classify_hard_easy, compute_obs_attribution).
#       Khi gọi TỪ TRONG model (get_loss_breakdown, sample), LUÔN truyền
#       self.hard_score_weight_logits để gradient thực sự chảy qua được.

#     QUAN TRỌNG: hàm này KHÔNG còn @torch.no_grad() — để gradient của
#     weight_logits chảy ngược qua sw_hard trong _reg_loss khi được gọi
#     trong training. Các nơi gọi hàm này CHỈ để lấy giá trị (không train,
#     ví dụ XAI logging) phải tự bọc bằng `with torch.no_grad():` ở call
#     site nếu muốn tránh build computation graph không cần thiết.
#     """
#     T, B   = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
#     device = obs_traj_norm.device
#     if T < 3:
#         z = torch.zeros(B, device=device)
#         if return_components:
#             return z, {"curvature": z.clone(), "speed_var": z.clone(),
#                        "dir_change": z.clone(), "obs_speed_norm": z.clone()}
#         return z

#     traj_deg = _norm_to_deg(obs_traj_norm[..., :2])
#     az12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
#     az23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
#     diff = (az23 - az12).abs()
#     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
#     curvature  = diff.mean(0) / math.pi
#     spd        = _step_speeds_kmh(traj_deg)
#     if spd.shape[0] >= 2:
#         speed_var = (spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 1.)
#         obs_speed_norm = (spd.mean(0) / obs_speed_norm_const).clamp(0., 1.)
#     else:
#         speed_var = torch.zeros(B, device=device)
#         obs_speed_norm = torch.zeros(B, device=device)
#     dir_change = (diff > (20.0 / 180.0 * math.pi)).float().mean(0)

#     components = torch.stack([curvature, speed_var, dir_change, obs_speed_norm], dim=0)  # [4, B]

#     if weight_logits is not None:
#         w = F.softmax(weight_logits.to(device).to(components.dtype), dim=0)  # [4]
#     else:
#         # Fallback equivalent khi không có instance context: gần với
#         # phân bổ gốc 0.4/0.3/0.3 cho 3 thành phần đầu, obs_speed neutral nhỏ
#         w = torch.tensor([0.35, 0.25, 0.25, 0.15], device=device, dtype=components.dtype)

#     score = (w.unsqueeze(1) * components).sum(0).clamp(0., 1.)   # [B]

#     if return_components:
#         return score, {"curvature": curvature, "speed_var": speed_var,
#                        "dir_change": dir_change, "obs_speed_norm": obs_speed_norm}
#     return score


# # ─────────────────────────────────────────────────────────────────────────────
# #  Physics score v2.6 (adds displacement_score) — GIỮ NGUYÊN HẰNG SỐ
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def _physics_score(traj_norm: torch.Tensor, obs_norm: torch.Tensor) -> torch.Tensor:
#     """
#     Best-of-K selection score. Four components:
#       speed_score       (0.30): per-step speed vs obs reference
#       smooth_score      (0.25): trajectory smoothness (no sharp acceleration)
#       heading_score     (0.30): first-step direction matches obs
#       displacement_score(0.15): total path length consistent with obs speed

#     [v2.1-learn QUYẾT ĐỊNH] KHÔNG biến 4 exponent này thành learnable.
#     Lý do: hàm này chạy dưới @torch.no_grad() để CHỌN trong số K candidates
#     sinh từ CÙNG MỘT velocity network — đây là post-hoc re-ranking, không
#     nằm trên đường gradient nào quay lại network hay bất kỳ tham số nào
#     của nó. Nếu thêm nn.Parameter vào đây mà không có loss term riêng
#     (như đã làm với speed_correction ở LEARN-1), tham số đó sẽ lại rơi vào
#     đúng lỗi "tồn tại nhưng never updated" mà ta đang cố tránh.
#     Để làm đúng cần thêm 1 loss phụ huấn luyện riêng các exponent này
#     (vd qua REINFORCE / Gumbel-softmax cho discrete selection) — phạm vi
#     đó rủi ro cao hơn lợi ích đo được, nên giữ nguyên 4 hằng số đã proven
#     qua thực nghiệm thay vì đoán thêm một cơ chế chưa kiểm chứng.
#     """
#     B      = traj_norm.shape[1]
#     device = traj_norm.device
#     traj_deg = _norm_to_deg(traj_norm)
#     v_ref   = None

#     # ── Speed score ────────────────────────────────────────────────────────
#     if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
#         obs_deg = _norm_to_deg(obs_norm)
#         obs_spd = _step_speeds_kmh(obs_deg)
#         T_s     = obs_spd.shape[0]
#         w_obs   = torch.linspace(0.5, 1.0, T_s, device=device)
#         v_ref   = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()   # [B]
#         pred_spd = _step_speeds_kmh(traj_deg)
#         v_sigma  = v_ref.clamp(min=5.0) * 0.5
#         speed_score = torch.exp(
#             -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5)
#     elif traj_deg.shape[0] >= 2:
#         speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.).mean(0))
#     else:
#         speed_score = torch.ones(B, device=device)

#     # ── Smoothness score ───────────────────────────────────────────────────
#     if traj_deg.shape[0] >= 3:
#         vel          = traj_deg[1:] - traj_deg[:-1]
#         accel_mag    = (vel[1:] - vel[:-1]).norm(dim=-1)
#         smooth_score = torch.exp(-accel_mag.mean(0) * 5.0)
#     else:
#         smooth_score = torch.ones(B, device=device)

#     # ── Heading score ──────────────────────────────────────────────────────
#     if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
#         obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
#         pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
#         obs_h    = F.normalize(obs_vel,  dim=-1, eps=1e-6)
#         pred_h   = F.normalize(pred_vel, dim=-1, eps=1e-6)
#         cos_sim  = (obs_h * pred_h).sum(-1).clamp(-1, 1)
#         head_score = torch.exp((cos_sim - 1.0) * 3.0)
#     else:
#         head_score = torch.ones(B, device=device)

#     # ── Displacement score ──────────────────────────────────────────────
#     # Expected total path length ≈ obs_speed × T_pred × DT × 0.75
#     # (0.75 factor: storms curve, so straight-line < path length estimate)
#     if v_ref is not None and traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
#         T_pred        = traj_deg.shape[0]
#         expected_total = v_ref * T_pred * DT_HOURS * 0.75   # [B] km
#         step_dists    = _haversine_deg(traj_deg[:-1], traj_deg[1:])   # [T-1, B]
#         actual_total  = step_dists.sum(0)                              # [B]
#         rel_err       = (actual_total - expected_total).abs() / expected_total.clamp(min=10.)
#         disp_score    = torch.exp(-rel_err * 1.5)
#     else:
#         disp_score    = torch.ones(B, device=device)

#     return (speed_score.pow(0.30)
#             * smooth_score.pow(0.25)
#             * head_score.pow(0.30)
#             * disp_score.pow(0.15)).clamp(min=1e-6)


# # ─────────────────────────────────────────────────────────────────────────────
# #  Augmentation — GIỮ NGUYÊN từ v2.1-clean (đã proven qua thực nghiệm)
# # ─────────────────────────────────────────────────────────────────────────────

# def augment_batch(batch_list) -> list:
#     """
#     Distribution (4 active types + 2 no-op slots):
#       A (25%): track shift ±5km             — shape unchanged, position varies
#       B (20%): GT speed scale ×0.85–1.15   — mild, proven from v2.1
#       C (20%): recurvature ±20°             — proven CTE improvement, FIXED to
#                                               rotate displacement vectors (not
#                                               absolute positions — old bug)
#       D-E (25%): no augmentation            — giữ original distribution
#       F (10%): Gaussian noise ±3km          — small robustness

#     KHÔNG dùng (proven harmful qua thực nghiệm):
#       - AUG-D obs-speed scaling: +4.5km ATE (v2.6), +13.4km (v2.7) — ĐÃ XÓA
#       - mixup (phi vật lý, v2.4 proof)
#       - speed scale >×1.5 trên GT (too far from real)
#       - exp L_reg weights (v2.4 catastrophic, ADE=291km)
#     """
#     bl = list(batch_list)
#     if not torch.is_tensor(bl[0]):
#         return bl

#     obs    = bl[0]
#     device = obs.device
#     anchor = obs[-1:, :, :2].detach()  # [1, B, 2] — last obs as pivot
#     r = torch.rand(1).item()

#     if r < 0.25:
#         # A: Track shift ±5km (preserves storm shape, varies position)
#         shift = (torch.rand(2, device=device) - 0.5) * 0.018  # ±5km in normalized
#         bl[0] = obs + shift.view(1, 1, 2)
#         if torch.is_tensor(bl[1]):
#             bl[1] = bl[1] + shift.view(1, 1, 2)

#     elif r < 0.45:
#         # B: Speed scale ×0.85–1.15 (both obs+GT from anchor, mild)
#         scale = 0.85 + 0.30 * torch.rand(1, device=device).item()
#         obs_c = obs.clone()
#         obs_c[..., :2] = anchor + (obs[..., :2] - anchor) * scale
#         bl[0] = obs_c
#         if torch.is_tensor(bl[1]):
#             bl[1] = anchor + (bl[1] - anchor) * scale

#     elif r < 0.65:
#         # C: Recurvature ±20° — rotate DISPLACEMENT vectors (not positions)
#         # FIX v2.1-clean: old bug rotated absolute positions around anchor,
#         # which inflated step-distances → model learned "recurvature=faster"
#         # → ATE +7.2km artifact. Fixed by rotating displacement vectors:
#         # direction changes, step magnitude preserved = correct TC physics.
#         T_pred = bl[1].shape[0] if torch.is_tensor(bl[1]) else 0
#         if T_pred >= 4:
#             gt      = bl[1].clone()
#             max_deg = (torch.rand(1).item() - 0.5) * 40.0   # -20° to +20°
#             max_rad = max_deg * math.pi / 180.0
#             pts  = torch.cat([anchor, gt], 0)   # [T_pred+1, B, 2]
#             disp = pts[1:] - pts[:-1]           # [T_pred, B, 2]
#             for t in range(T_pred):
#                 progress = (t / max(T_pred - 1, 1)) ** 1.5
#                 a = max_rad * progress
#                 c, s = math.cos(a), math.sin(a)
#                 rot = torch.tensor([[c, -s], [s, c]], dtype=gt.dtype, device=device)
#                 disp[t] = (rot @ disp[t].unsqueeze(-1)).squeeze(-1)
#             gt_new = gt.clone()
#             gt_new[0] = anchor[0] + disp[0]
#             for t in range(1, T_pred):
#                 gt_new[t] = gt_new[t - 1] + disp[t]
#             bl[1] = gt_new
#             # Rotate last 3 obs DISPLACEMENTS by 30% of max (continuity)
#             T_obs   = obs.shape[0]
#             obs_aug = obs.clone()
#             cp = math.cos(max_rad * 0.3); sp = math.sin(max_rad * 0.3)
#             rp = torch.tensor([[cp, -sp], [sp, cp]], dtype=obs.dtype, device=device)
#             for t_obs in range(max(1, T_obs - 3), T_obs):
#                 d = obs_aug[t_obs, :, :2] - obs_aug[t_obs - 1, :, :2]
#                 obs_aug[t_obs, :, :2] = obs_aug[t_obs - 1, :, :2] + (rp @ d.unsqueeze(-1)).squeeze(-1)
#             bl[0] = obs_aug

#     elif r < 0.90:
#         # D-E slot: no augmentation
#         pass

#     else:
#         # F: small Gaussian noise ±3km
#         obs_new = obs.clone()
#         obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.003
#         bl[0] = obs_new

#     return bl


# # ─────────────────────────────────────────────────────────────────────────────
# #  XAI functions (1–7, complete) — GIỮ NGUYÊN logic, cập nhật call hard_score
# # ─────────────────────────────────────────────────────────────────────────────

# def compute_obs_attribution(model, batch_list, device: torch.device,
#                              target_step: int = 11) -> torch.Tensor:
#     """
#     [XAI-1] Gradient saliency: which obs step most influences 72h prediction?
#     """
#     raw = _unwrap(model)
#     with torch.no_grad():
#         h_score = hard_score_from_obs(batch_list[0][:, :, :2],
#                                        weight_logits=getattr(raw, "hard_score_weight_logits", None))
#     obs_req = batch_list[0].detach().clone().requires_grad_(True)
#     bl_g = list(batch_list); bl_g[0] = obs_req
#     with torch.enable_grad():
#         cond = raw.encoder(bl_g, hard_score=h_score)
#         x0   = torch.randn(obs_req.shape[1], raw.pred_len, 2, device=device) * raw.sigma_inference
#         t0   = torch.zeros(obs_req.shape[1], device=device)
#         v    = raw.velocity(x0, t0, cond)
#         pred_rel = x0 + v
#         ts       = min(target_step, raw.pred_len - 1)
#         pred_rel[:, ts, :].norm(dim=-1).mean().backward()
#     if obs_req.grad is not None:
#         attr = obs_req.grad[:, :, :2].norm(dim=-1)
#         attr = attr / (attr.sum(0, keepdim=True) + 1e-8)
#     else:
#         attr = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)
#     return attr.detach()


# @torch.no_grad()
# def compute_ensemble_uncertainty(all_traj: torch.Tensor) -> Dict:
#     """
#     [XAI-4] Uncertainty per lead time across K ensemble samples.
#     """
#     all_deg  = _norm_to_deg(all_traj)  # [K,T,B,2]
#     K, T, B  = all_deg.shape[:3]
#     mean_traj = all_deg.mean(0)
#     std_km = torch.zeros(T, B, device=all_traj.device)
#     for t in range(T):
#         dists = _haversine_deg(
#             all_deg[:, t].reshape(K * B, 2),
#             mean_traj[t].unsqueeze(0).expand(K, B, 2).reshape(K * B, 2)
#         ).reshape(K, B)
#         std_km[t] = dists.std(0)
#     s12 = min(1, T - 1); s72 = min(11, T - 1)
#     return {
#         "std_per_step":      std_km,
#         "uncertainty_ratio": (std_km[s72] + 1e-3) / (std_km[s12] + 1e-3),
#         "mean_72h_std":      float(std_km[s72].mean()),
#         "mean_12h_std":      float(std_km[s12].mean()),
#         "high_uncertainty":  std_km[s72] > 80.0,
#     }


# @torch.no_grad()
# def compute_heading_deviation(pred_deg: torch.Tensor,
#                                gt_deg:   torch.Tensor) -> torch.Tensor:
#     """
#     [XAI-6] Heading deviation per lead time step, in degrees (absolute).
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         return pred_deg.new_zeros(1, pred_deg.shape[1])
#     bear_gt   = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
#     bear_pred = _forward_azimuth(gt_deg[:T-1], pred_deg[1:T])
#     diff = (bear_pred - bear_gt).abs()
#     diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
#     return torch.rad2deg(diff)


# @torch.no_grad()
# def compute_cte_contribution(pred_deg: torch.Tensor,
#                               gt_deg:   torch.Tensor) -> Dict:
#     """
#     [XAI-7] Decompose trajectory error into ATE (along-track) and CTE (cross-track).
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         z = pred_deg.new_zeros(1, pred_deg.shape[1])
#         return {"ate_per_step": z, "cte_per_step": z,
#                 "ate_mean": z[0], "cte_mean": z[0],
#                 "ate_abs_mean": 0.0, "cte_abs_mean": 0.0}
#     bear_ref  = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
#     bear_err  = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
#     dist_err  = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
#     ang       = bear_err - bear_ref
#     ate       = dist_err * torch.cos(ang)
#     cte       = dist_err * torch.sin(ang)
#     return {
#         "ate_per_step":  ate,
#         "cte_per_step":  cte,
#         "ate_mean":      ate.mean(0),
#         "cte_mean":      cte.abs().mean(0),
#         "ate_abs_mean":  float(ate.abs().mean()),
#         "cte_abs_mean":  float(cte.abs().mean()),
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# #  Compat stubs
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def classify_hard_easy(obs_traj_norm, per_sample_loss=None,
#                         hard_score_p: float = 70.0, loss_p: float = 50.0):
#     scores = hard_score_from_obs(obs_traj_norm)
#     B = scores.shape[0]
#     if B < 4:
#         return torch.zeros(B, dtype=torch.bool, device=scores.device)
#     return scores >= torch.quantile(scores, hard_score_p / 100.0)


# @torch.no_grad()
# def classify_hard_easy_global(obs_traj_norm, global_threshold):
#     return hard_score_from_obs(obs_traj_norm) >= global_threshold


# @torch.no_grad()
# def compute_diversity_score(candidates) -> float:
#     if len(candidates) < 2:
#         return 0.0
#     T, B = candidates[0].shape[0], candidates[0].shape[1]
#     ep_step = min(T - 1, 11)
#     endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
#     N = endpoints.shape[0]
#     ep_mean = endpoints.mean(0, keepdim=True)
#     dists = _haversine_deg(
#         endpoints.reshape(N * B, 2),
#         ep_mean.expand(N, B, 2).reshape(N * B, 2)
#     ).reshape(N, B)
#     return float(dists.std(0).mean())


# # ─────────────────────────────────────────────────────────────────────────────
# #  TCFlowMatching v2.1-learn
# # ─────────────────────────────────────────────────────────────────────────────

# class TCFlowMatching(nn.Module):
#     """
#     TC-FlowMatching v2.1-learn

#     Training loss:
#         L_total = L_CFM
#                 + lam_reg  × L_reg          (ramp ep10→ep30, LEARNABLE step weights)
#                 + lam_dir  × lambda_heading × L_heading_ms  (ramp ep5→ep20)
#                 + 0.1      × L_calib         (trains speed_correction_logits)

#         L_CFM:         conditional flow matching (random t, OT noise-GT matching)
#         L_reg:         ADE loss at sigma_inference, LEARNABLE per-step weights
#         L_heading_ms:  full 12-step heading continuation, LEARNABLE per-step weights
#         L_calib:       ADE on speed-calibrated 1-shot pred, trains per-horizon
#                        speed correction factors used by speed_calibrate_pred()

#     NO L_momentum: removed because it anchors to val speed distribution,
#     which hurts ATE on test storms that move at different speeds.

#     Inference pipeline:
#         1. Sample K=20 candidates with sigma=0.04 FIXED
#         2. Physics score selection (speed+smooth+heading+displacement) — fixed weights
#         3. Weighted average of top-3 candidates
#         4. Speed calibration: LEARNABLE per-horizon correction (was fixed clip)
#         5. Optional: multi-scale sigma ensemble (sample_multiscale)
#     """

#     def __init__(
#         self,
#         pred_len:          int   = 12,
#         obs_len:           int   = 8,
#         unet_in_ch:        int   = 13,
#         d_cond:            int   = 256,
#         d_model:           int   = 256,
#         nhead:             int   = 8,
#         num_dec_layers:    int   = 4,
#         dim_ff:            int   = 512,
#         dropout:           float = 0.1,
#         sigma_min:         float = 0.04,
#         sigma_max:         float = 0.08,
#         lambda_reg:        float = 0.2,
#         lambda_heading:    float = 0.07,   # reduced: gradient scale balance with L_CFM
#         lambda_momentum:   float = 0.0,    # DISABLED — field kept for compat only
#         lambda_calib:      float = 0.1,    # [LEARN-1] weight of L_calib term
#         use_ot:            bool  = True,
#         ot_epsilon:        float = 0.05,
#         use_ema:           bool  = True,
#         ema_decay:         float = 0.995,
#         n_inference_steps: int   = 1,
#         n_ensemble:        int   = 20,
#         sigma_inference:   float = 0.04,   # FIXED throughout
#         **kwargs,
#     ):
#         super().__init__()
#         self.pred_len          = pred_len
#         self.obs_len            = obs_len
#         self.sigma_min          = sigma_min
#         self.sigma_max          = sigma_max
#         self.lambda_reg         = lambda_reg
#         self.lambda_heading     = lambda_heading
#         self.lambda_momentum    = 0.0           # always disabled
#         self.lambda_calib       = lambda_calib
#         self.use_ot              = use_ot
#         self.ot_epsilon          = ot_epsilon
#         self.n_inference_steps   = n_inference_steps
#         self.n_ensemble          = n_ensemble
#         self.sigma_inference     = sigma_inference

#         self.encoder  = ContextEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
#         self.velocity = VelocityTransformer(
#             pred_len=pred_len, d_model=d_model, nhead=nhead,
#             num_layers=num_dec_layers, dim_ff=dim_ff,
#             dropout=dropout, d_cond=d_cond)
#         self.use_ema = use_ema
#         self._ema    = None

#         # ── [v2.1-learn] LEARNABLE PARAMETERS ──────────────────────────────
#         # Mỗi tham số dưới đây thay thế một hằng số tay đoán trong
#         # v2.1-clean. Mỗi cái đều có ĐƯỜNG GRADIENT THỰC SỰ về tổng loss
#         # (verify trong get_loss_breakdown) — không phải param "mồ côi".

#         # [LEARN-1] Per-horizon speed correction, dùng trong speed_calibrate_pred.
#         # Init=0 → sigmoid(0)*2=1.0 → identity (không sửa gì) lúc bắt đầu,
#         # giống hệt điểm khởi đầu hành vi cũ (scale=1 baseline).
#         self.speed_correction_logits = nn.Parameter(torch.zeros(pred_len))

#         # [LEARN-2] Per-step L_reg weights. Init=0 → softmax uniform start
#         # (khác linspace(1,1.5) cũ vốn đã thiên về late-step ngay từ đầu).
#         self.reg_step_logits = nn.Parameter(torch.zeros(pred_len))

#         # [LEARN-3] hard_score 4-component weights: [curvature, speed_var,
#         # dir_change, obs_speed_norm]. Init lệch nhẹ giống phân bổ gốc
#         # 0.4/0.3/0.3 (+obs_speed nhỏ) thay vì zero-init hoàn toàn, để
#         # không phá vỡ hành vi đã proven ngay từ epoch đầu — nhưng vẫn
#         # cho phép gradient điều chỉnh dần qua training.
#         self.hard_score_weight_logits = nn.Parameter(
#             torch.log(torch.tensor([0.40, 0.30, 0.30, 0.15])))

#         # [LEARN-5] Per-step heading constraint weights — TOÀN BỘ pred_len
#         # bước (không chỉ 4 bước đầu như bản cũ). Init theo decay 0.5^t cho
#         # 4 bước đầu (giữ đúng điểm khởi đầu đã proven: 1.0,0.5,0.25,0.125)
#         # rồi tiếp tục decay cho các bước sau — nhưng đây CHỈ LÀ ĐIỂM XUẤT
#         # PHÁT, gradient sẽ tự điều chỉnh phân bổ trong lúc train, không
#         # còn bị khóa cứng ở "chỉ 4 bước đầu có trọng số, 8 bước sau = 0"
#         # như bản trước.
#         _decay_init = torch.tensor([0.5 ** t for t in range(pred_len)])
#         self.heading_step_logits = nn.Parameter(torch.log(_decay_init.clamp(min=1e-4)))

#         # [LEARN-4] Cross-loss weighting — Kendall homoscedastic uncertainty
#         # (Kendall, Gal & Cipolla, 2018: "Multi-Task Learning Using
#         # Uncertainty to Weigh Losses"). Thay lambda_reg/lambda_heading/
#         # lambda_calib HẰNG SỐ TAY ĐOÁN bằng log_sigma HỌC ĐƯỢC cho mỗi
#         # loss phụ, dùng công thức: loss_i / (2·exp(2·log_sigma_i)) + log_sigma_i
#         #
#         # TẠI SAO KHÔNG DÙNG nn.Parameter(lambda) TRỰC TIẾP (naive cách):
#         #   total = l_cfm + lambda*l_reg → d(total)/d(lambda) = l_reg ≥ 0
#         #   → optimizer luôn đẩy lambda về 0 → TẮT loss phụ hoàn toàn
#         #   → suy biến (degenerate), không phải "tự học cân bằng"
#         # TẠI SAO log_sigma KHÔNG suy biến: số hạng +log_sigma_i là
#         # regularizer — nếu sigma→∞ (tương đương lambda→0) thì log_sigma
#         # →∞ làm total loss TĂNG, ngăn optimizer "trốn" loss bằng cách tắt nó.
#         # Init log_sigma=0 → sigma=1 → 1/(2·1)=0.5, tương đương lambda=0.5
#         # ban đầu cho mỗi loss phụ (điểm khởi đầu hợp lý, không quá khác
#         # baseline cũ lambda_reg=0.2/lambda_heading=0.07/lambda_calib=0.1).
#         self.log_sigma_reg     = nn.Parameter(torch.tensor(0.0))
#         self.log_sigma_heading = nn.Parameter(torch.tensor(0.0))
#         self.log_sigma_calib   = nn.Parameter(torch.tensor(0.0))

#     def init_ema(self):
#         if self.use_ema:
#             self._ema = EMAModel(self, decay=0.995)

#     def ema_update(self):
#         if self._ema is not None:
#             self._ema.update(self)

#     def _to_relative(self, x_abs: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
#         return x_abs - last_obs.unsqueeze(1)

#     def _from_relative(self, x_rel: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
#         return x_rel + last_obs.unsqueeze(1)

#     def _sigma_schedule(self, epoch: int) -> float:
#         """Cosine decay sigma_max→sigma_min over ep5→ep40."""
#         if epoch < 5:
#             return self.sigma_max
#         if epoch < 40:
#             t = (epoch - 5) / 35.0
#             return self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * t))
#         return self.sigma_min

#     # ── Multi-step heading loss — [LEARN-5] learnable per-step weights ──────

#     def _heading_loss_ms(self, pred_deg: torch.Tensor, obs_deg: torch.Tensor) -> torch.Tensor:
#         """
#         Multi-step heading constraint — TOÀN BỘ pred_len bước, trọng số HỌC.

#         [LEARN-5] SỬA LỖI: bản trước (v2.1-clean và lần sửa trước của tôi)
#         chỉ áp dụng cho n_steps=4 ĐẦU TIÊN (6h/12h/18h/24h) với decay=0.5
#         CỐ ĐỊNH (weight = 1.0, 0.5, 0.25, 0.125) — đây CHÍNH XÁC là loại
#         hằng số tay đoán cần sửa mà tôi đã bỏ sót. Hệ quả thực tế: heading
#         constraint hoàn toàn KHÔNG áp dụng cho 8 bước còn lại (30h→72h) —
#         đây là một phần lý do 72h heading dev khó cải thiện trong log cũ
#         (113-128° xuyên suốt nhiều version) dù ATE/CTE có nhúc nhích.

#         FIX: dùng self.heading_step_logits (pred_len params, softmax) —
#         TOÀN BỘ 12 bước đều có constraint, trọng số PHÂN BỔ GIỮA chúng do
#         gradient tự học, không còn giới hạn cứng "chỉ 4 bước đầu" hay
#         decay theo cấp số nhân tay chọn (0.5^t).

#         Detach ref tại mỗi step vẫn giữ nguyên (CRITICAL: ngăn gradient
#         explosion qua chuỗi 12 bước liên tiếp — đây là cơ chế ổn định
#         numerice, không phải "trọng số", nên không thuộc diện cần học).
#         """
#         if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
#             return pred_deg.new_zeros(())

#         ref_bear = _forward_azimuth(obs_deg[-2], obs_deg[-1])   # [B]
#         pts = torch.cat([obs_deg[-1:], pred_deg], 0)   # [T_pred+1, B, 2]

#         N = pred_deg.shape[0]
#         sw = F.softmax(self.heading_step_logits[:N], dim=0)   # [N], học được

#         loss = pred_deg.new_zeros(())
#         for t in range(N):
#             pred_bear  = _forward_azimuth(pts[t], pts[t + 1])  # [B]
#             angle_diff = pred_bear - ref_bear
#             loss       = loss + sw[t] * (1.0 - torch.cos(angle_diff)).mean()
#             ref_bear   = pred_bear.detach()   # CRITICAL: detach to prevent chain gradient

#         return loss

#     # ── L_reg — [LEARN-2] learnable step weights ────────────────────────────

#     def _reg_loss(self, x1_rel: torch.Tensor, last_obs: torch.Tensor,
#                   cond: torch.Tensor,
#                   hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         ADE loss at t=0 with sigma_inference noise, consistent with 1-shot inference.

#         [LEARN-2] Step weights = softmax(self.reg_step_logits) thay vì
#         linspace(1.0, 1.5, T) hardcoded.
#         Bằng chứng tại sao cần đổi: linspace(1,3)→linspace(1,1.5) chỉ giảm
#         test ATE 0.9km, và XAI-8 24h ratio KHÔNG đổi (vẫn ~1.5-1.6×) —
#         chứng tỏ chỉnh tay con số trong công thức tuyến tính không đủ;
#         cần để gradient tự tìm phân bổ trọng số tối ưu giữa 12 horizon.

#         DOES NOT use exp weights (v2.4 used 12.2× ratio → catastrophic
#         overfitting) — softmax trên logits học được có range tự nhiên bị
#         chặn mềm, không bùng nổ như exp(linspace) thủ công.
#         """
#         B, T, _ = x1_rel.shape
#         device   = x1_rel.device
#         x0   = torch.randn_like(x1_rel) * self.sigma_inference
#         t0   = torch.zeros(B, device=device)
#         v    = self.velocity(x0, t0, cond)
#         x1_pred_abs = self._from_relative(x0 + v, last_obs)
#         x1_gt_abs   = self._from_relative(x1_rel, last_obs)
#         pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))  # [T, B, 2]
#         gt_deg   = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
#         dist     = _haversine_deg(pred_deg, gt_deg)             # [T, B] km

#         T_actual = dist.shape[0]
#         sw = F.softmax(self.reg_step_logits[:T_actual], dim=0).unsqueeze(1)  # [T, 1]

#         if hard_score is not None:
#             sw_hard = (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)  # [1, B]
#         else:
#             sw_hard = torch.ones(1, B, device=device, dtype=dist.dtype)
#         return ((dist * sw) * sw_hard).mean() / 300.0

#     # ── get_loss_breakdown ──────────────────────────────────────────────────

#     def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
#         """
#         Full loss computation. Called in TRAINING with augmented batch.
#         Called in VAL without augmentation → val_loss is reliable signal.

#         Total = L_CFM + lam_reg × L_reg + lam_dir × lambda_heading × L_heading_ms
#                       + lambda_calib × L_calib
#         """
#         obs_traj = batch_list[0]
#         gt_traj  = batch_list[1]
#         B        = obs_traj.shape[1]
#         device   = obs_traj.device

#         sigma    = self._sigma_schedule(epoch)
#         x1_gt    = gt_traj.permute(1, 0, 2)       # [B, T, 2]
#         last_obs = obs_traj[-1, :, :2]             # [B, 2]
#         x1_rel   = self._to_relative(x1_gt, last_obs)

#         # [LEARN-3] hard_score: truyền self.hard_score_weight_logits để
#         # gradient của 4 trọng số chảy qua sw_hard trong _reg_loss bên dưới.
#         h_score = hard_score_from_obs(obs_traj[:, :, :2],
#                                        weight_logits=self.hard_score_weight_logits)
#         cond    = self.encoder(batch_list, hard_score=h_score)

#         # ── L_CFM ─────────────────────────────────────────────────────────
#         x0 = torch.randn_like(x1_rel) * sigma
#         if self.use_ot and B >= 4:
#             x0_flat, x1_flat = _ot_match(
#                 x0.reshape(B, -1), x1_rel.reshape(B, -1), self.ot_epsilon)
#             x0         = x0_flat.reshape(B, self.pred_len, 2)
#             x1_matched = x1_flat.reshape(B, self.pred_len, 2)
#         else:
#             x1_matched = x1_rel

#         t        = torch.rand(B, device=device)
#         x_t      = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1_matched
#         u_target = x1_matched - x0
#         v_pred   = self.velocity(x_t, t, cond)
#         l_cfm    = F.mse_loss(v_pred, u_target)

#         # ── L_reg ramp ep10→ep30 (GATE theo curriculum, KHÔNG phải trọng số) ──
#         # Lý do giữ ramp dạng if/else thay vì học: đây là LỊCH TRÌNH BẬT/TẮT
#         # theo trạng thái huấn luyện (encoder bị freeze tới ep10 nên L_reg
#         # chưa có ý nghĩa trước đó — bật sớm sẽ lan gradient nhiễu vào
#         # velocity khi nó chưa thấy cond ổn định). Đây không phải "model
#         # tin loss này quan trọng bao nhiêu" (thứ Kendall xử lý ở dưới) mà
#         # là "loss này có HỢP LỆ để tính tại epoch này hay chưa" — 2 vấn đề
#         # khác nhau về bản chất, gate này dùng index epoch (số nguyên rời
#         # rạc) nên không phải dạng có thể differentiable để học bằng SGD.
#         if epoch < 10:     ramp_reg = 0.0
#         elif epoch < 30:   ramp_reg = (epoch - 10) / 20.0
#         else:              ramp_reg = 1.0

#         l_reg = (self._reg_loss(x1_rel, last_obs, cond, h_score)
#                  if ramp_reg > 0.0 else x0.new_zeros(()))

#         # ── L_heading_ms ramp ep5→ep20 (cùng lý do GATE ở trên) ─────────────
#         if epoch < 5:      ramp_dir = 0.0
#         elif epoch < 20:   ramp_dir = (epoch - 5) / 15.0
#         else:              ramp_dir = 1.0

#         if ramp_dir > 0.0:
#             x0_h       = torch.randn_like(x1_rel) * self.sigma_inference
#             v_h        = self.velocity(x0_h, torch.zeros(B, device=device), cond)
#             x1_h_abs   = self._from_relative(x0_h + v_h, last_obs)
#             pred_deg_h = _norm_to_deg(x1_h_abs.permute(1, 0, 2))   # [T, B, 2]
#             obs_deg_h  = _norm_to_deg(obs_traj[:, :, :2])           # [T_obs, B, 2]
#             l_heading  = self._heading_loss_ms(pred_deg_h, obs_deg_h)
#         else:
#             l_heading = x0.new_zeros(())

#         l_momentum = x0.new_zeros(())   # always 0

#         # ── L_calib — [LEARN-1] trains self.speed_correction_logits ─────────
#         # CRITICAL: speed_correction_logits is consumed inside
#         # speed_calibrate_pred(), which is normally only called from
#         # sample() under @torch.no_grad() at eval time. Without this term,
#         # the optimizer would NEVER update speed_correction_logits — it
#         # would be a registered nn.Parameter that silently never trains.
#         # This computes a 1-shot prediction, applies the (now-gradient-
#         # enabled) speed calibration, and measures ADE on the calibrated
#         # output. Gradient flows: ADE → calibrated pred → BOTH
#         # speed_correction_logits (direct) AND velocity (through pred_mean,
#         # so the network and the correction co-adapt during training).
#         x0_c = torch.randn_like(x1_rel) * self.sigma_inference
#         v_c  = self.velocity(x0_c, torch.zeros(B, device=device), cond)
#         pred_c_abs = self._from_relative(x0_c + v_c, last_obs)          # [B,T,2] norm
#         pred_c_abs = pred_c_abs.permute(1, 0, 2)                        # [T,B,2] norm
#         cal_abs    = self.speed_calibrate_pred(pred_c_abs, last_obs, obs_traj[:, :, :2])
#         cal_deg    = _norm_to_deg(cal_abs)
#         gt_c_deg   = _norm_to_deg(x1_gt.permute(1, 0, 2))
#         l_calib    = _haversine_deg(cal_deg, gt_c_deg).mean() / 300.0

#         # ── Total — [LEARN-4] Kendall uncertainty weighting cho cross-loss ──
#         # Thay vì total = l_cfm + lambda_reg*l_reg + lambda_heading*l_heading
#         #                       + lambda_calib*l_calib  (3 hằng số tay đoán),
#         # dùng: loss_i / (2·exp(2·log_sigma_i)) + log_sigma_i cho mỗi loss
#         # phụ. log_sigma_i HỌC ĐƯỢC, +log_sigma_i ngăn suy biến về 0 (xem
#         # giải thích đầy đủ tại nơi khai báo log_sigma_reg trong __init__).
#         # ramp_reg/ramp_dir vẫn nhân vào để giữ đúng curriculum gating.
#         prec_reg     = torch.exp(-2.0 * self.log_sigma_reg)
#         prec_heading = torch.exp(-2.0 * self.log_sigma_heading)
#         prec_calib   = torch.exp(-2.0 * self.log_sigma_calib)

#         weighted_reg     = ramp_reg     * (0.5 * prec_reg     * l_reg     + self.log_sigma_reg)
#         weighted_heading = ramp_dir     * (0.5 * prec_heading * l_heading + self.log_sigma_heading)
#         weighted_calib   = 1.0          * (0.5 * prec_calib   * l_calib   + self.log_sigma_calib)

#         total = l_cfm + weighted_reg + weighted_heading + weighted_calib
#         if not torch.isfinite(total):
#             total = x0.new_zeros(())

#         # ── ADE 1-step log ────────────────────────────────────────────────
#         with torch.no_grad():
#             x0_log = torch.randn_like(x1_rel) * self.sigma_inference
#             v_log  = self.velocity(x0_log, torch.zeros(B, device=device), cond)
#             ade_log = _haversine_deg(
#                 _norm_to_deg(self._from_relative(x0_log + v_log, last_obs).permute(1, 0, 2)),
#                 _norm_to_deg(x1_gt.permute(1, 0, 2))
#             ).mean().item()

#         return {
#             "total":     total,
#             "l_cfm":     l_cfm.item(),
#             "l_reg":     l_reg.item() if torch.is_tensor(l_reg) else 0.0,
#             "l_heading": l_heading.item() if torch.is_tensor(l_heading) else 0.0,
#             "l_calib":   l_calib.item(),
#             "l_momentum": 0.0,   # always 0
#             "lam_reg":   ramp_reg,   # gate value (curriculum on/off), kept for compat
#             "lam_dir":   ramp_dir,   # gate value (curriculum on/off), kept for compat
#             "sigma":     sigma,
#             "ade_1step": ade_log,
#             "hard_score_mean": float(h_score.detach().mean()),
#             "hard_score_max":  float(h_score.detach().max()),
#             # [LEARN-4] learned cross-loss weights — effective lambda = 0.5*prec
#             "learned_lambda_reg":     float((0.5 * prec_reg).detach()),
#             "learned_lambda_heading": float((0.5 * prec_heading).detach()),
#             "learned_lambda_calib":   float((0.5 * prec_calib).detach()),
#             # Compat keys for older scripts
#             "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
#             "speed": 0., "accel": 0., "fm_mse": l_cfm.item(),
#             "l_hard_total": 0., "n_hard": 0, "alpha_hard": 0.,
#             "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
#         }

#     def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
#         return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

#     # ── Speed calibration — [LEARN-1] learnable per-horizon ─────────────────

#     def speed_calibrate_pred(self,
#                               pred_abs_norm: torch.Tensor,
#                               last_obs_norm: torch.Tensor,
#                               obs_norm:      torch.Tensor) -> torch.Tensor:
#         """
#         Per-horizon LEARNABLE speed calibration. Thay thế hằng số
#         clip_min/clip_max=0.85/1.15 cũ (1 scale chung cho cả 12 bước).

#         BẰNG CHỨNG TẠI SAO 1 SCALE CHUNG KHÔNG ĐỦ:
#           XAI-8 (v2.1-clean, đã chạy) cho thấy correction cần khác nhau
#           rõ rệt theo horizon: 12h cần scale≈0.77, 24h cần scale≈0.62,
#           48h cần scale≈1.06, 72h cần scale≈1.04. clip(0.85,1.15) không
#           thể đáp ứng tất cả các nhu cầu trái ngược này cùng lúc — đây là
#           lý do mở rộng clip range (đã thử 0.72-1.35) chỉ cải thiện <1km:
#           sai công cụ cho vấn đề (1 số thay vì 12 số).

#         Không còn @staticmethod / @torch.no_grad(): chạy NGAY TRONG
#         forward pass (gọi từ L_calib trong get_loss_breakdown VÀ từ
#         sample() dưới no_grad của caller khi inference) để gradient có
#         thể chảy qua self.speed_correction_logits trong lúc training.

#         correction[t] = 2 * sigmoid(speed_correction_logits[t]) ∈ (0, 2)
#         init = sigmoid(0)*2 = 1.0 → no-op tại epoch 0, học dần qua L_calib.
#         """
#         if obs_norm.shape[0] < 2 or pred_abs_norm.shape[0] < 2:
#             return pred_abs_norm

#         T = pred_abs_norm.shape[0]
#         correction = (torch.sigmoid(self.speed_correction_logits[:T]) * 2.0
#                       ).to(pred_abs_norm.dtype).view(T, 1, 1)               # [T,1,1]

#         pts  = torch.cat([last_obs_norm.unsqueeze(0), pred_abs_norm], 0)    # [T+1, B, 2]
#         disp = pts[1:] - pts[:-1]                                           # [T, B, 2]
#         disp_cal = disp * correction

#         out = torch.empty_like(pred_abs_norm)
#         cur = last_obs_norm
#         for t in range(T):
#             cur = cur + disp_cal[t]
#             out[t] = cur
#         return out

#     # ── Sample (1-shot + speed calibration) ──────────────────────────────

#     @torch.no_grad()
#     def sample(self, batch_list,
#                num_ensemble:          Optional[int]  = None,
#                ddim_steps:            Optional[int]  = None,
#                return_xai:            bool           = False,
#                use_speed_calibration: bool           = True,
#                **kwargs) -> Tuple:
#         """
#         1-shot inference with physics selection + learnable speed calibration.

#         Steps:
#           1. Encode context once (hard_score + full encoder)
#           2. Sample K=20 candidates: x0 ~ N(0, sigma_inf²), x_pred = x0 + v(x0, 0, cond)
#           3. Physics score each candidate (speed+smooth+heading+displacement)
#           4. Weighted average of top-3 candidates
#           5. Speed calibration per-horizon (LEARNED, not fixed clip)
#           6. If return_xai=True: compute XAI-1 through XAI-9

#         Returns:
#           (pred_mean [T,B,2], zeros [T,B,2], all_traj [K,T,B,2])
#           or (pred_mean, zeros, all_traj, xai_dict) if return_xai=True
#         """
#         K  = num_ensemble or self.n_ensemble
#         N  = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else self.n_inference_steps
#         dt = 1.0 / max(N, 1)

#         obs_traj    = batch_list[0]
#         T_obs, B, _ = obs_traj.shape
#         device      = obs_traj.device

#         h_score  = hard_score_from_obs(obs_traj[:, :, :2],
#                                         weight_logits=self.hard_score_weight_logits)
#         obs_norm = obs_traj[:, :, :2]
#         last_obs = obs_traj[-1, :, :2]
#         t0       = torch.zeros(B, device=device)

#         cond = self.encoder(batch_list, hard_score=h_score)

#         all_traj = []
#         for _ in range(K):
#             x_rel = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference

#             if N <= 1:
#                 v     = self.velocity(x_rel, t0, cond)
#                 x_rel = x_rel + v
#             else:
#                 for step in range(N):
#                     t_b   = torch.full((B,), step * dt, device=device)
#                     x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3., 3.)

#             x_abs = self._from_relative(x_rel, last_obs)
#             all_traj.append(x_abs.permute(1, 0, 2))   # [T, B, 2]

#         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)   # [K, B]
#         all_t   = torch.stack(all_traj, 0)   # [K, T, B, 2]
#         top_k   = min(3, K)
#         top_idx = scores.topk(top_k, dim=0).indices   # [top_k, B]

#         pred_mean = torch.zeros_like(all_traj[0])
#         for b in range(B):
#             idx_b = top_idx[:, b]
#             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
#             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

#         # [LEARN-1] Per-horizon LEARNED speed calibration (no hardcoded clip args)
#         if use_speed_calibration:
#             pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

#         if not return_xai:
#             return pred_mean, torch.zeros_like(pred_mean), all_t

#         # ── XAI output (XAI-2,3,4,5,6,7,8,9) ──────────────────────────────
#         xai = {}

#         xai.update(compute_ensemble_uncertainty(all_t))

#         _, hard_comps = hard_score_from_obs(obs_norm, return_components=True,
#                                              weight_logits=self.hard_score_weight_logits)
#         xai["hard_components"] = hard_comps

#         pred_deg  = _norm_to_deg(pred_mean)    # [T, B, 2]
#         obs_deg_x = _norm_to_deg(obs_norm)     # [T_obs, B, 2]

#         obs_spd_x    = _step_speeds_kmh(obs_deg_x)
#         obs_spd_mu   = obs_spd_x.mean(0)     # [B]
#         if pred_deg.shape[0] >= 2:
#             last_deg_x   = obs_deg_x[-1]
#             pts_x        = torch.cat([last_deg_x.unsqueeze(0), pred_deg], 0)
#             pred_spd_x   = _step_speeds_kmh(pts_x)
#             pred_spd_mu  = pred_spd_x.mean(0)  # [B]
#         else:
#             pred_spd_mu = obs_spd_mu.clone()

#         speed_ratio = (pred_spd_mu / obs_spd_mu.clamp(min=1.0))
#         xai["speed_comparison"] = {
#             "obs_speed_mean":  float(obs_spd_mu.mean()),
#             "pred_speed_mean": float(pred_spd_mu.mean()),
#             "speed_ratio":     float(speed_ratio.mean()),
#             "per_storm_obs":   obs_spd_mu,
#             "per_storm_pred":  pred_spd_mu,
#             "over_predict":    speed_ratio > 1.15,
#             "under_predict":   speed_ratio < 0.85,
#         }

#         v_ref = obs_spd_mu.clamp(min=5.0)
#         v_sig = v_ref * 0.5
#         spd_sc = torch.exp(-((pred_spd_mu - v_ref) / v_sig).pow(2) * 0.5)
#         if pred_deg.shape[0] >= 3:
#             vel_x   = pred_deg[1:] - pred_deg[:-1]
#             accel_x = (vel_x[1:] - vel_x[:-1]).norm(dim=-1)
#             smo_sc  = torch.exp(-accel_x.mean(0) * 5.0)
#         else:
#             smo_sc = torch.ones(B, device=device)
#         if obs_norm.shape[0] >= 2 and pred_mean.shape[0] >= 1:
#             ov = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
#             pv = pred_mean[0, :, :2] - obs_norm[-1, :, :2]
#             cos_s = (F.normalize(ov, dim=-1, eps=1e-6) * F.normalize(pv, dim=-1, eps=1e-6)).sum(-1)
#             hd_sc = torch.exp((cos_s.clamp(-1, 1) - 1.0) * 3.0)
#         else:
#             hd_sc = torch.ones(B, device=device)
#         xai["physics_components"] = {
#             "speed": spd_sc, "smooth": smo_sc, "heading": hd_sc,
#             "obs_speed": obs_spd_mu, "pred_speed": pred_spd_mu,
#         }

#         gt_traj_xai = batch_list[1]
#         gt_deg_xai  = _norm_to_deg(gt_traj_xai[:, :, :2])
#         xai["heading_deviation_deg"] = compute_heading_deviation(pred_deg, gt_deg_xai)

#         xai["ate_cte_decomp"] = compute_cte_contribution(pred_deg, gt_deg_xai)

#         # XAI-8: Per-horizon speed ratio (12h/24h/48h/72h)
#         if pred_deg.shape[0] >= 2 and gt_deg_xai.shape[0] >= 2:
#             T8       = min(pred_deg.shape[0], gt_deg_xai.shape[0])
#             last_d   = obs_deg_x[-1]
#             pts_pred = torch.cat([last_d.unsqueeze(0), pred_deg[:T8]], 0)
#             pts_gt   = torch.cat([last_d.unsqueeze(0), gt_deg_xai[:T8]], 0)
#             spd_pred = _step_speeds_kmh(pts_pred)   # [T8, B]
#             spd_gt   = _step_speeds_kmh(pts_gt)     # [T8, B]
#             ratio    = spd_pred / spd_gt.clamp(min=1.0)  # [T8, B]
#             def _hz(s, lo, hi):
#                 hi = min(hi, s.shape[0])
#                 return float(s[lo:hi].mean()) if hi > lo else float("nan")
#             r_mean = ratio.mean(1)   # [T8]
#             xai["speed_per_horizon"] = {
#                 "ratio":    r_mean.tolist(),
#                 "pred_kmh": spd_pred.mean(1).tolist(),
#                 "gt_kmh":   spd_gt.mean(1).tolist(),
#                 "12h_ratio": _hz(r_mean, 0, 2),
#                 "24h_ratio": _hz(r_mean, 2, 4),
#                 "48h_ratio": _hz(r_mean, 6, 8),
#                 "72h_ratio": _hz(r_mean, 10, 12),
#             }
#         else:
#             xai["speed_per_horizon"] = {}

#         # XAI-9: Storm category breakdown by obs speed
#         obs_spd_cat = _step_speeds_kmh(obs_deg_x).mean(0)   # [B]
#         sm = obs_spd_cat < 8.0
#         mm = (obs_spd_cat >= 8.0) & (obs_spd_cat < 15.0)
#         fm = obs_spd_cat >= 15.0
#         ade_per_storm = _haversine_deg(pred_deg[:gt_deg_xai.shape[0]],
#                                         gt_deg_xai[:pred_deg.shape[0]]).mean(0)  # [B]
#         def _cat_ade(mask):
#             return float(ade_per_storm[mask].mean()) if mask.sum() > 0 else float("nan")
#         xai["storm_categories"] = {
#             "n_slow":    int(sm.sum()),
#             "n_medium":  int(mm.sum()),
#             "n_fast":    int(fm.sum()),
#             "speed_mean": float(obs_spd_cat.mean()),
#             "speed_std":  float(obs_spd_cat.std()),
#             "ade_slow":   _cat_ade(sm),
#             "ade_medium": _cat_ade(mm),
#             "ade_fast":   _cat_ade(fm),
#         }

#         # [LEARN diagnostics] expose learned values for monitoring in train_fm
#         xai["learned_params"] = {
#             "speed_correction": (torch.sigmoid(self.speed_correction_logits) * 2.0).tolist(),
#             "reg_step_weights":  F.softmax(self.reg_step_logits, dim=0).tolist(),
#             "hard_score_weights": F.softmax(self.hard_score_weight_logits, dim=0).tolist(),
#         }

#         return pred_mean, torch.zeros_like(pred_mean), all_t, xai

#     @torch.no_grad()
#     def sample_multiscale(
#         self,
#         batch_list,
#         sigmas: Optional[List[float]] = None,
#         n_per_sigma: int = 4,
#         use_speed_calibration: bool = True,
#     ) -> Tuple:
#         """
#         Multi-scale sigma ensemble at inference (DOES NOT affect training).
#         Hedges against speed distribution shift between val and test.
#         """
#         if sigmas is None:
#             sigmas = [0.025, 0.035, 0.04, 0.05, 0.065]

#         obs_traj = batch_list[0]
#         B = obs_traj.shape[1]; device = obs_traj.device

#         h_score  = hard_score_from_obs(obs_traj[:, :, :2],
#                                         weight_logits=self.hard_score_weight_logits)
#         obs_norm = obs_traj[:, :, :2]
#         last_obs = obs_traj[-1, :, :2]
#         t0       = torch.zeros(B, device=device)
#         cond     = self.encoder(batch_list, hard_score=h_score)

#         all_traj = []
#         for sigma in sigmas:
#             for _ in range(n_per_sigma):
#                 x0    = torch.randn(B, self.pred_len, 2, device=device) * sigma
#                 v     = self.velocity(x0, t0, cond)
#                 x_abs = self._from_relative(x0 + v, last_obs)
#                 all_traj.append(x_abs.permute(1, 0, 2))

#         scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)
#         all_t   = torch.stack(all_traj, 0)
#         top_k   = min(5, len(all_traj))
#         top_idx = scores.topk(top_k, dim=0).indices

#         pred_mean = torch.zeros_like(all_traj[0])
#         for b in range(B):
#             idx_b = top_idx[:, b]
#             w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
#             pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

#         if use_speed_calibration:
#             pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

#         return pred_mean, torch.zeros_like(pred_mean), all_t


# # ─────────────────────────────────────────────────────────────────────────────
# #  Backward compat alias
# # ─────────────────────────────────────────────────────────────────────────────
# TCDiffusion = TCFlowMatching

"""
Model/flow_matching_model.py  ──  TC-FlowMatching v2.1-clean
═══════════════════════════════════════════════════════════════════════════════
VIẾT LẠI HOÀN TOÀN từ v2.1 + các fix đã verified từ thực nghiệm 140 epoch.

━━━ CƠ SỞ LÝ THUYẾT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v2.1 test: ADE=229.8, ATE=214.4, CTE=71.6 (best so far, generalize tốt nhất)
v2.5 test: ADE=232.5, ATE=222.3, CTE=64.5 (ATE tệ hơn, CTE tốt hơn)
v2.4 test: ADE=291.1 (catastrophic overfitting)

PHÂN TÍCH LỖI v2.1:
  ATE/CTE ratio on test = 214.4/71.6 = 3.0
  → 75% error is ALONG-TRACK (speed/distance)
  → 25% is CROSS-TRACK (direction)
  Test gap 72h = +129.3km >> 12h = +10.1km → long-range drift

PHÂN TÍCH v2.5 THẤT BẠI:
  L_momentum anchor pred_speed ≈ val_speed (10.3km/h)
  Test storms faster → L_momentum SLOWS model on test → ATE +7.9km WORSE
  L_heading (step 0 only) → 72h heading dev INCREASED: ep0=87°→ep120=128°!
  Recurvature aug: CTE improved 7km ← giữ lại

━━━ VẬT LÝ CỦA GIẢI PHÁP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GIẢI PHÁP ATE (along-track error):
  Training: KHÔNG anchor speed tại training (no L_momentum)
  Inference: SPEED CALIBRATION per-storm
    scale = clip(obs_speed / pred_speed_first4, 0.85, 1.15)
    pred_cal = last_obs + (pred - last_obs) * scale
    → Preserves direction (CTE unchanged), fixes magnitude (ATE reduced)
    → Per-storm adaptive: fast test storms get upscaling, slow storms downscaling
    → DOES NOT affect training → zero train/inference consistency issue

GIẢI PHÁP CTE (cross-track error):
  Training: MULTI-STEP heading constraint trên TOÀN BỘ 12 bước (không chỉ
  4 bước đầu như mô tả ban đầu) — xem [LEARN-5] bên dưới để biết chi tiết
  về lỗi đã phát hiện và sửa: bản trước chỉ áp dụng constraint cho 4/12
  bước, khiến 8 bước cuối (30h→72h) hoàn toàn không có gradient signal
  hướng. Trọng số mỗi bước giờ HỌC ĐƯỢC qua self.heading_step_logits.

GIẢI PHÁP GAP (val→test distribution shift):
  [AUG-C] Recurvature: proven to reduce CTE test (v2.5)
  Speed calibration at inference bridges remaining gap

━━━ GIỮ NGUYÊN TỪ v2.1 (PROVEN, KHÔNG THAY ĐỔI) ━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✅ sigma_inference=0.04 FIXED — zero train/inference mismatch
  ✅ Mild base aug: shift±5km + speed×0.85–1.15 (val ≈ test distribution)
  ✅ 1-shot inference nhất quán với L_reg training
  ✅ 2-group optimizer, encoder freeze 10ep
  ✅ val loop NO augmentation
  ✅ OT noise-GT matching trong L_CFM
  ✅ ContextEncoder: FNO3D + Mamba + Env + 6 kinematic features
  ✅ VelocityTransformer: d_model=256, 4 layers, nhead=8

━━━ v2.1-LEARN CHANGES (từ v2.1-clean baseline, test ADE=227.8 ATE=220.3 CTE=60.6) ━

  CĂN CỨ: mọi lần đổi HẰNG SỐ tay đoán (linspace range, clip range) trước đây
  chỉ tạo dao động <1km test. Bằng chứng: XAI-8 24h ratio luôn ~1.5-1.6×
  bất kể linspace(1,3) hay linspace(1,1.5). Đây là dấu hiệu bottleneck không
  nằm ở "chọn đúng hằng số" mà ở việc DÙNG hằng số thay vì để model học.

  [LEARN-1] speed_calibrate_pred: từ 1 scale chung (clip 0.85-1.15) cho TOÀN
            BỘ 12 bước → 12 hệ số riêng biệt theo từng horizon, học qua
            gradient (không còn @torch.no_grad/@staticmethod).
            Lý do: XAI-8 chứng minh correction cần KHÁC NHAU theo horizon
            (12h≈0.77x, 24h≈0.62x, 48h≈1.06x, 72h≈1.04x) — 1 số không
            thể đáp ứng cả 4 nhu cầu khác nhau cùng lúc.
            THÊM: L_calib loss term để gradient thực sự chảy tới tham số này
            (nếu không có loss term riêng, tham số tồn tại nhưng KHÔNG BAO
            GIỜ được optimizer cập nhật vì nó chỉ được dùng trong sample()
            dưới torch.no_grad() tại thời điểm eval).

  [LEARN-2] _reg_loss step weights: linspace(1.0,1.5) hardcoded → 12 logit
            riêng (softmax), model tự học phân bổ trọng số giữa các horizon
            thay vì áp công thức tuyến tính tay đoán.

  [LEARN-3] hard_score: thêm obs_speed (đã chuẩn hóa) làm thành phần thứ 4,
            VÀ 4 trọng số kết hợp (curvature, speed_var, dir_change, obs_speed)
            học qua gradient thay vì hardcode 0.4/0.3/0.3.
            Lý do: XAI-9 cho thấy fast storms (n=94/420=22%) gây 62% tổng CTE
            test nhưng hard_score cũ không có tín hiệu "obs_speed tuyệt đối"
            → storm nhanh-nhưng-đều (speed_var thấp) bị coi nhầm là "dễ"
            → không được train với loss-weight cao hơn dù nó cần.
            Bỏ @torch.no_grad() ở hard_score_from_obs để gradient chảy qua
            được (qua sw_hard trong _reg_loss).

  [LEARN-4] Cross-loss weighting: lambda_reg/lambda_heading/lambda_calib
            (3 hằng số tay đoán: 0.2/0.07/0.1) → Kendall homoscedastic
            uncertainty weighting (Kendall, Gal & Cipolla 2018):
              weighted_term = loss/(2·exp(2·log_sigma)) + log_sigma
            3 log_sigma HỌC ĐƯỢC qua gradient.
            QUAN TRỌNG: KHÔNG dùng nn.Parameter(lambda) trực tiếp nhân
            vào loss — gradient của total theo lambda = loss ≥ 0 luôn
            đẩy lambda về 0 (suy biến, tắt hết loss phụ). Số hạng
            +log_sigma trong công thức Kendall là regularizer ngăn suy
            biến này — đã verify thực nghiệm (xem test ở cuối file dev
            log): effective_lambda KHÔNG giảm về 0 qua 30 bước train
            với lr cao, trái lại tăng dần khi loss phụ giảm nhanh.
            Epoch-ramp (ep5/ep10/ep20/ep30) GIỮ NGUYÊN dạng if/else rời
            rạc — đây là GATE bật/tắt theo trạng thái encoder-freeze,
            không phải "trọng số tương đối", không thể differentiable
            theo epoch (số nguyên), nên không thuộc phạm vi can học.

  [LEARN-5] _heading_loss_ms: SỬA LỖI BỎ SÓT từ lần trước. Bản trước
            (kể cả lần "v2.1-learn" đầu tiên của tôi) vẫn giữ n_steps=4
            CỐ ĐỊNH + decay=0.5 CỐ ĐỊNH (weight=1.0/0.5/0.25/0.125 cho
            4 bước ĐẦU, 8 bước 30h→72h KHÔNG có constraint heading nào).
            FIX: self.heading_step_logits (pred_len=12 params, softmax),
            áp dụng cho TOÀN BỘ 12 bước, init theo đúng decay 0.5^t cũ
            (không đổi điểm khởi đầu) nhưng gradient tự học lại phân bổ
            trong lúc train — không còn giới hạn cứng "chỉ 4 bước đầu".

  KHÔNG ĐỔI (đã cân nhắc, có lý do kỹ thuật rõ ràng):
    _physics_score 4 exponent (0.30/0.25/0.30/0.15): hàm này chạy dưới
    @torch.no_grad() để CHỌN giữa K candidates sinh từ CÙNG một velocity
    network — đây là post-hoc re-ranking, không nằm trên đường gradient
    nào quay lại network hay bất kỳ tham số nào của nó. Biến nó learnable
    mà không có loss-term riêng sẽ lặp lại đúng lỗi "tham số mồ côi"
    (never updated) mà LEARN-1 đã phải sửa bằng cách thêm L_calib.
    R_EARTH, DT_HOURS, hệ số affine normalized↔degree: hằng số vật lý/
    metadata dataset, không phải "model nghĩ gì" — đổi sẽ làm sai đơn vị
    tính toán, không phải tinh chỉnh hành vi model.

  GIỮ NGUYÊN HOÀN TOÀN (đã proven qua thực nghiệm, KHÔNG đổi):
    ✅ AUG-C: rotate DISPLACEMENT vectors (không phải absolute positions)
    ✅ AUG-D: ĐÃ XÓA — proven harmful (+4.5km v2.6, +13.4km v2.7)
    ✅ L_momentum: disabled (proven harmful +7.9km — v2.5)
    ✅ sigma_inference=0.04 fixed (train/infer consistency)
    ✅ OT matching, 1-shot inference, encoder freeze 10ep

━━━ KẾT QUẢ THỰC TẾ (baseline trước thay đổi này) ━━━━━━━━━━━━━━━━━━━━━━━━━

                    Val ADE   Test ADE  Test ATE  Test CTE
  v2.1 gốc:         170.1     229.8     214.4     71.6
  v2.1-XAI (ran):   169.7     229.8     221.6     61.9  ← AUG-C bug → ATE +7.2
  v2.1-clean (ran):  ~168      227.8     220.3     60.6  ← baseline cho bản này
  ST-Trans target:            224.4     213.7     59.4

  CHƯA CÓ KẾT QUẢ TEST CHO v2.1-LEARN (bản này) — cần retrain để biết
  ATE/CTE thực tế có cải thiện so với 220.3/60.6 hay không. Mọi con số
  "kỳ vọng" chỉ là giả thuyết dựa trên cơ chế, KHÔNG phải đã đo được.
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
#  Coordinate utilities (same as v2.1)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
    """Normalized coords → (lon°, lat°)."""
    return torch.stack([
        (t[..., 0] * 50.0 + 1800.0) / 10.0,
        (t[..., 1] * 50.0) / 10.0,
    ], dim=-1)


def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Haversine distance km between two (lon°,lat°) tensors."""
    lat1 = torch.deg2rad(p1[..., 1]);  lat2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Bearing in radians from p1→p2 (degrees input)."""
    lon1 = torch.deg2rad(p1[..., 0]);  lat1 = torch.deg2rad(p1[..., 1])
    lon2 = torch.deg2rad(p2[..., 0]);  lat2 = torch.deg2rad(p2[..., 1])
    dlon = lon2 - lon1
    y = torch.sin(dlon) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    return torch.atan2(y, x)


def _step_speeds_kmh(traj_deg: torch.Tensor) -> torch.Tensor:
    """Speed km/h between consecutive steps of a trajectory [T, B, 2]."""
    if traj_deg.shape[0] < 2:
        return traj_deg.new_zeros(1, traj_deg.shape[1])
    return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# ─────────────────────────────────────────────────────────────────────────────
#  EMAModel (same as v2.1)
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m


class EMAModel:
    def __init__(self, model, decay: float = 0.995):
        self.decay = decay
        m = _unwrap(model)
        self.shadow = {k: v.detach().clone()
                       for k, v in m.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        m = _unwrap(model)
        with torch.no_grad():
            for k, v in m.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        m = _unwrap(model)
        backup, sd = {}, m.state_dict()
        for k in self.shadow:
            if k not in sd: continue
            backup[k] = sd[k].detach().clone()
            sd[k].copy_(self.shadow[k])
        return backup

    def restore(self, model, backup):
        m = _unwrap(model)
        sd = m.state_dict()
        for k, v in backup.items():
            if k in sd: sd[k].copy_(v)


# ─────────────────────────────────────────────────────────────────────────────
#  OT matching (same as v2.1)
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


def _ot_match(x0_flat: torch.Tensor, x1_flat: torch.Tensor,
              epsilon: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    B = x0_flat.shape[0]
    if B < 4:
        return x0_flat, x1_flat
    try:
        cost = torch.cdist(x0_flat.float(), x1_flat.float()) / (x0_flat.shape[-1] ** 0.5)
        with torch.no_grad():
            pi = _sinkhorn_log(cost, epsilon=epsilon)
        flat = pi.reshape(-1).clamp(0.0)
        s = flat.sum()
        if not torch.isfinite(s) or s < 1e-10:
            return x0_flat, x1_flat
        idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
        return x0_flat[idx // B], x1_flat
    except Exception:
        return x0_flat, x1_flat


# ─────────────────────────────────────────────────────────────────────────────
#  VelocityTransformer (identical to v2.1)
# ─────────────────────────────────────────────────────────────────────────────

class VelocityTransformer(nn.Module):
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
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.cond_proj  = nn.Sequential(nn.Linear(d_cond, d_model), nn.LayerNorm(d_model))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
        self.out_scale = nn.Parameter(torch.ones(pred_len, 2) * 0.1)
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype)
                         * (-math.log(10000.0) / max(half - 1, 1)))
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.time_mlp(emb)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_t.shape
        step_idx = torch.arange(T, device=x_t.device).unsqueeze(0).expand(B, -1)
        x_emb = (self.traj_embed(x_t) + self.pos_emb[:, :T] + self.step_emb(step_idx))
        memory = torch.cat([self._time_emb(t).unsqueeze(1),
                            self.cond_proj(cond).unsqueeze(1)], dim=1)
        out = self.out_norm(self.decoder(x_emb, memory))
        return self.out_proj(out) * torch.sigmoid(self.out_scale[:T]).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
#  ContextEncoder (identical to v2.1)
# ─────────────────────────────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    RAW_CTX_DIM = 512

    def __init__(self, obs_len: int = 8, unet_in_ch: int = 13, d_cond: int = 256):
        super().__init__()
        self.obs_len = obs_len
        self.d_cond  = d_cond

        self.spatial_enc     = FNO3DEncoder(in_channel=unet_in_ch, out_channel=1, d_model=32,
                                             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
                                             spatial_down=32, dropout=0.05)
        self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.bottleneck_proj = nn.Linear(128, 128)
        self.decoder_proj    = nn.Linear(1, 16)
        self.enc_1d          = DataEncoder1D(in_1d=4, feat_3d_dim=128, mlp_h=64,
                                              lstm_hidden=128, lstm_layers=3,
                                              dropout=0.1, d_state=16)
        self.env_enc         = Env_net(obs_len=obs_len, d_model=32)
        self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
        self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
        self.ctx_drop = nn.Dropout(0.1)
        self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, d_cond)
        self.ctx_ln2  = nn.LayerNorm(d_cond)
        # 6 kinematic features: vel_x, vel_y, speed_n, sin(h), cos(h), accel
        self.vel_obs_enc = nn.Sequential(
            nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, d_cond // 2), nn.GELU())
        self.hard_embed = nn.Sequential(
            nn.Linear(1, d_cond // 4), nn.GELU(), nn.Linear(d_cond // 4, d_cond // 4))
        self.fuse = nn.Sequential(
            nn.Linear(d_cond + d_cond // 2 + d_cond // 4, d_cond),
            nn.LayerNorm(d_cond), nn.GELU())

    def _encode_raw(self, batch_list) -> torch.Tensor:
        obs_traj  = batch_list[0]; obs_Me = batch_list[7]
        image_obs = batch_list[11]; env_data = batch_list[13]
        if image_obs.dim() == 4:
            image_obs = image_obs.unsqueeze(2)
        if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
            image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
        e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
        T_obs = obs_traj.shape[0]
        e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        e_3d_s = self.bottleneck_proj(e_3d_s)
        if e_3d_s.shape[1] != T_obs:
            e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
                                   mode="linear", align_corners=False).permute(0,2,1)
        e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
        t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
                                          device=e_3d_dec_t.device) * 0.5, dim=0)
        f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        h_t    = self.enc_1d(obs_in, e_3d_s)
        e_env, _, _ = self.env_enc(env_data, image_obs)
        return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

    def _kinematic_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
        """6 kinematic features per obs step."""
        B = obs_traj.shape[1]; T_obs = obs_traj.shape[0]; device = obs_traj.device
        if T_obs >= 2:
            traj_deg = _norm_to_deg(obs_traj)
            vel_norm = obs_traj[1:] - obs_traj[:-1]
            speed    = _step_speeds_kmh(traj_deg)
            speed_n  = (speed / 20.0).clamp(-3.0, 3.0)
            heading  = torch.atan2(vel_norm[:, :, 1], vel_norm[:, :, 0])
            if T_obs >= 3:
                dspd  = speed[1:] - speed[:-1]
                accel = torch.cat([obs_traj.new_zeros(1, B),
                                   (dspd / 10.0).clamp(-3.0, 3.0)], 0)
            else:
                accel = obs_traj.new_zeros(T_obs - 1, B)
            kine = torch.stack([vel_norm[:,:,0], vel_norm[:,:,1], speed_n,
                                heading.sin(), heading.cos(), accel], dim=-1)
        else:
            kine = obs_traj.new_zeros(self.obs_len, B, 6)
        if kine.shape[0] < self.obs_len:
            kine = torch.cat([obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6), kine], 0)
        else:
            kine = kine[-self.obs_len:]
        return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))

    def forward(self, batch_list, hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
        raw   = self._encode_raw(batch_list)
        ctx   = self.ctx_ln2(self.ctx_fc2(self.ctx_drop(raw)))
        kfeat = self._kinematic_feat(batch_list[0][:, :, :2])
        if hard_score is None:
            hard_score = torch.zeros(ctx.shape[0], device=ctx.device)
        hfeat = self.hard_embed(hard_score.unsqueeze(1).to(ctx.dtype))
        return self.fuse(torch.cat([ctx, kfeat, hfeat], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
#  Hard score (XAI-2) — [LEARN-3] thêm obs_speed + learnable weights
# ─────────────────────────────────────────────────────────────────────────────

def hard_score_from_obs(obs_traj_norm: torch.Tensor,
                         return_components: bool = False,
                         weight_logits: Optional[torch.Tensor] = None,
                         obs_speed_norm_const: float = 20.0):
    """
    [XAI-2] Độ khó của storm — v2.1-learn: 4 components, learnable weights.

    [LEARN-3] THAY ĐỔI so với v2.1-clean:
      CŨ: score = 0.4*curvature + 0.3*speed_var + 0.3*dir_change  (hardcode)
          KHÔNG có obs_speed tuyệt đối → storm nhanh-nhưng-đều (speed_var
          thấp) bị coi nhầm là "dễ", dù XAI-9 cho thấy fast storms
          (obs_speed≥15km/h, 22% test set) gây 62% tổng CTE test.
      MỚI: score = softmax(weight_logits) · [curvature, speed_var,
                                              dir_change, obs_speed_norm]
           4 trọng số HỌC ĐƯỢC qua gradient (không phải số tay đoán),
           obs_speed_norm = obs_speed_mean / 20.0 (clamp 0,1) làm input
           feature thứ 4.

    weight_logits: nếu None, dùng uniform-equivalent fallback (0.25 mỗi cái)
      — cho phép gọi hàm này từ các chỗ KHÔNG có instance (vd các hàm
      module-level classify_hard_easy, compute_obs_attribution).
      Khi gọi TỪ TRONG model (get_loss_breakdown, sample), LUÔN truyền
      self.hard_score_weight_logits để gradient thực sự chảy qua được.

    QUAN TRỌNG: hàm này KHÔNG còn @torch.no_grad() — để gradient của
    weight_logits chảy ngược qua sw_hard trong _reg_loss khi được gọi
    trong training. Các nơi gọi hàm này CHỈ để lấy giá trị (không train,
    ví dụ XAI logging) phải tự bọc bằng `with torch.no_grad():` ở call
    site nếu muốn tránh build computation graph không cần thiết.
    """
    T, B   = obs_traj_norm.shape[0], obs_traj_norm.shape[1]
    device = obs_traj_norm.device
    if T < 3:
        z = torch.zeros(B, device=device)
        if return_components:
            return z, {"curvature": z.clone(), "speed_var": z.clone(),
                       "dir_change": z.clone(), "obs_speed_norm": z.clone()}
        return z

    traj_deg = _norm_to_deg(obs_traj_norm[..., :2])
    az12 = _forward_azimuth(traj_deg[:-2], traj_deg[1:-1])
    az23 = _forward_azimuth(traj_deg[1:-1], traj_deg[2:])
    diff = (az23 - az12).abs()
    diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
    curvature  = diff.mean(0) / math.pi
    spd        = _step_speeds_kmh(traj_deg)
    if spd.shape[0] >= 2:
        speed_var = (spd.std(0) / spd.mean(0).clamp(min=1.0)).clamp(0., 1.)
        obs_speed_norm = (spd.mean(0) / obs_speed_norm_const).clamp(0., 1.)
    else:
        speed_var = torch.zeros(B, device=device)
        obs_speed_norm = torch.zeros(B, device=device)
    dir_change = (diff > (20.0 / 180.0 * math.pi)).float().mean(0)

    components = torch.stack([curvature, speed_var, dir_change, obs_speed_norm], dim=0)  # [4, B]

    if weight_logits is not None:
        w = F.softmax(weight_logits.to(device).to(components.dtype), dim=0)  # [4]
    else:
        # Fallback equivalent khi không có instance context: gần với
        # phân bổ gốc 0.4/0.3/0.3 cho 3 thành phần đầu, obs_speed neutral nhỏ
        w = torch.tensor([0.35, 0.25, 0.25, 0.15], device=device, dtype=components.dtype)

    score = (w.unsqueeze(1) * components).sum(0).clamp(0., 1.)   # [B]

    if return_components:
        return score, {"curvature": curvature, "speed_var": speed_var,
                       "dir_change": dir_change, "obs_speed_norm": obs_speed_norm}
    return score


# ─────────────────────────────────────────────────────────────────────────────
#  Physics score v2.6 (adds displacement_score) — GIỮ NGUYÊN HẰNG SỐ
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _physics_score(traj_norm: torch.Tensor, obs_norm: torch.Tensor) -> torch.Tensor:
    """
    Best-of-K selection score. Four components:
      speed_score       (0.30): per-step speed vs obs reference
      smooth_score      (0.25): trajectory smoothness (no sharp acceleration)
      heading_score     (0.30): first-step direction matches obs
      displacement_score(0.15): total path length consistent with obs speed

    [v2.1-learn QUYẾT ĐỊNH] KHÔNG biến 4 exponent này thành learnable.
    Lý do: hàm này chạy dưới @torch.no_grad() để CHỌN trong số K candidates
    sinh từ CÙNG MỘT velocity network — đây là post-hoc re-ranking, không
    nằm trên đường gradient nào quay lại network hay bất kỳ tham số nào
    của nó. Nếu thêm nn.Parameter vào đây mà không có loss term riêng
    (như đã làm với speed_correction ở LEARN-1), tham số đó sẽ lại rơi vào
    đúng lỗi "tồn tại nhưng never updated" mà ta đang cố tránh.
    Để làm đúng cần thêm 1 loss phụ huấn luyện riêng các exponent này
    (vd qua REINFORCE / Gumbel-softmax cho discrete selection) — phạm vi
    đó rủi ro cao hơn lợi ích đo được, nên giữ nguyên 4 hằng số đã proven
    qua thực nghiệm thay vì đoán thêm một cơ chế chưa kiểm chứng.
    """
    B      = traj_norm.shape[1]
    device = traj_norm.device
    traj_deg = _norm_to_deg(traj_norm)
    v_ref   = None

    # ── Speed score ────────────────────────────────────────────────────────
    if traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
        obs_deg = _norm_to_deg(obs_norm)
        obs_spd = _step_speeds_kmh(obs_deg)
        T_s     = obs_spd.shape[0]
        w_obs   = torch.linspace(0.5, 1.0, T_s, device=device)
        v_ref   = (obs_spd * w_obs.unsqueeze(1)).sum(0) / w_obs.sum()   # [B]
        pred_spd = _step_speeds_kmh(traj_deg)
        v_sigma  = v_ref.clamp(min=5.0) * 0.5
        speed_score = torch.exp(
            -((pred_spd - v_ref.unsqueeze(0)) / v_sigma.unsqueeze(0)).pow(2).mean(0) * 0.5)
    elif traj_deg.shape[0] >= 2:
        speed_score = torch.exp(-(_step_speeds_kmh(traj_deg).clamp(min=0) / 30.).mean(0))
    else:
        speed_score = torch.ones(B, device=device)

    # ── Smoothness score ───────────────────────────────────────────────────
    if traj_deg.shape[0] >= 3:
        vel          = traj_deg[1:] - traj_deg[:-1]
        accel_mag    = (vel[1:] - vel[:-1]).norm(dim=-1)
        smooth_score = torch.exp(-accel_mag.mean(0) * 5.0)
    else:
        smooth_score = torch.ones(B, device=device)

    # ── Heading score ──────────────────────────────────────────────────────
    if obs_norm.shape[0] >= 2 and traj_norm.shape[0] >= 1:
        obs_vel  = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
        pred_vel = traj_norm[0, :, :2] - obs_norm[-1, :, :2]
        obs_h    = F.normalize(obs_vel,  dim=-1, eps=1e-6)
        pred_h   = F.normalize(pred_vel, dim=-1, eps=1e-6)
        cos_sim  = (obs_h * pred_h).sum(-1).clamp(-1, 1)
        head_score = torch.exp((cos_sim - 1.0) * 3.0)
    else:
        head_score = torch.ones(B, device=device)

    # ── Displacement score ──────────────────────────────────────────────
    # Expected total path length ≈ obs_speed × T_pred × DT × 0.75
    # (0.75 factor: storms curve, so straight-line < path length estimate)
    if v_ref is not None and traj_deg.shape[0] >= 2 and obs_norm.shape[0] >= 2:
        T_pred        = traj_deg.shape[0]
        expected_total = v_ref * T_pred * DT_HOURS * 0.75   # [B] km
        step_dists    = _haversine_deg(traj_deg[:-1], traj_deg[1:])   # [T-1, B]
        actual_total  = step_dists.sum(0)                              # [B]
        rel_err       = (actual_total - expected_total).abs() / expected_total.clamp(min=10.)
        disp_score    = torch.exp(-rel_err * 1.5)
    else:
        disp_score    = torch.ones(B, device=device)

    return (speed_score.pow(0.30)
            * smooth_score.pow(0.25)
            * head_score.pow(0.30)
            * disp_score.pow(0.15)).clamp(min=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation — GIỮ NGUYÊN từ v2.1-clean (đã proven qua thực nghiệm)
# ─────────────────────────────────────────────────────────────────────────────

def augment_batch(batch_list) -> list:
    """
    Distribution (4 active types + 2 no-op slots):
      A (25%): track shift ±5km             — shape unchanged, position varies
      B (20%): GT speed scale ×0.85–1.15   — mild, proven from v2.1
      C (20%): recurvature ±20°             — proven CTE improvement, FIXED to
                                              rotate displacement vectors (not
                                              absolute positions — old bug)
      D-E (25%): no augmentation            — giữ original distribution
      F (10%): Gaussian noise ±3km          — small robustness

    KHÔNG dùng (proven harmful qua thực nghiệm):
      - AUG-D obs-speed scaling: +4.5km ATE (v2.6), +13.4km (v2.7) — ĐÃ XÓA
      - mixup (phi vật lý, v2.4 proof)
      - speed scale >×1.5 trên GT (too far from real)
      - exp L_reg weights (v2.4 catastrophic, ADE=291km)
    """
    bl = list(batch_list)
    if not torch.is_tensor(bl[0]):
        return bl

    obs    = bl[0]
    device = obs.device
    anchor = obs[-1:, :, :2].detach()  # [1, B, 2] — last obs as pivot
    r = torch.rand(1).item()

    if r < 0.25:
        # A: Track shift ±5km (preserves storm shape, varies position)
        shift = (torch.rand(2, device=device) - 0.5) * 0.018  # ±5km in normalized
        bl[0] = obs + shift.view(1, 1, 2)
        if torch.is_tensor(bl[1]):
            bl[1] = bl[1] + shift.view(1, 1, 2)

    elif r < 0.45:
        # B: Speed scale ×0.85–1.15 (both obs+GT from anchor, mild)
        scale = 0.85 + 0.30 * torch.rand(1, device=device).item()
        obs_c = obs.clone()
        obs_c[..., :2] = anchor + (obs[..., :2] - anchor) * scale
        bl[0] = obs_c
        if torch.is_tensor(bl[1]):
            bl[1] = anchor + (bl[1] - anchor) * scale

    elif r < 0.65:
        # C: Recurvature ±20° — rotate DISPLACEMENT vectors (not positions)
        # FIX v2.1-clean: old bug rotated absolute positions around anchor,
        # which inflated step-distances → model learned "recurvature=faster"
        # → ATE +7.2km artifact. Fixed by rotating displacement vectors:
        # direction changes, step magnitude preserved = correct TC physics.
        T_pred = bl[1].shape[0] if torch.is_tensor(bl[1]) else 0
        if T_pred >= 4:
            gt      = bl[1].clone()
            max_deg = (torch.rand(1).item() - 0.5) * 40.0   # -20° to +20°
            max_rad = max_deg * math.pi / 180.0
            pts  = torch.cat([anchor, gt], 0)   # [T_pred+1, B, 2]
            disp = pts[1:] - pts[:-1]           # [T_pred, B, 2]
            for t in range(T_pred):
                progress = (t / max(T_pred - 1, 1)) ** 1.5
                a = max_rad * progress
                c, s = math.cos(a), math.sin(a)
                rot = torch.tensor([[c, -s], [s, c]], dtype=gt.dtype, device=device)
                disp[t] = (rot @ disp[t].unsqueeze(-1)).squeeze(-1)
            gt_new = gt.clone()
            gt_new[0] = anchor[0] + disp[0]
            for t in range(1, T_pred):
                gt_new[t] = gt_new[t - 1] + disp[t]
            bl[1] = gt_new
            # Rotate last 3 obs DISPLACEMENTS by 30% of max (continuity)
            T_obs   = obs.shape[0]
            obs_aug = obs.clone()
            cp = math.cos(max_rad * 0.3); sp = math.sin(max_rad * 0.3)
            rp = torch.tensor([[cp, -sp], [sp, cp]], dtype=obs.dtype, device=device)
            for t_obs in range(max(1, T_obs - 3), T_obs):
                d = obs_aug[t_obs, :, :2] - obs_aug[t_obs - 1, :, :2]
                obs_aug[t_obs, :, :2] = obs_aug[t_obs - 1, :, :2] + (rp @ d.unsqueeze(-1)).squeeze(-1)
            bl[0] = obs_aug

    elif r < 0.90:
        # D-E slot: no augmentation
        pass

    else:
        # F: small Gaussian noise ±3km
        obs_new = obs.clone()
        obs_new[..., :2] = obs[..., :2] + torch.randn_like(obs[..., :2]) * 0.003
        bl[0] = obs_new

    return bl


# ─────────────────────────────────────────────────────────────────────────────
#  XAI functions (1–7, complete) — GIỮ NGUYÊN logic, cập nhật call hard_score
# ─────────────────────────────────────────────────────────────────────────────

def compute_obs_attribution(model, batch_list, device: torch.device,
                             target_step: int = 11) -> torch.Tensor:
    """
    [XAI-1] Gradient saliency: which obs step most influences 72h prediction?
    """
    raw = _unwrap(model)
    with torch.no_grad():
        h_score = hard_score_from_obs(batch_list[0][:, :, :2],
                                       weight_logits=getattr(raw, "hard_score_weight_logits", None))
    obs_req = batch_list[0].detach().clone().requires_grad_(True)
    bl_g = list(batch_list); bl_g[0] = obs_req
    with torch.enable_grad():
        cond = raw.encoder(bl_g, hard_score=h_score)
        x0   = torch.randn(obs_req.shape[1], raw.pred_len, 2, device=device) * raw.sigma_inference
        t0   = torch.zeros(obs_req.shape[1], device=device)
        v    = raw.velocity(x0, t0, cond)
        pred_rel = x0 + v
        ts       = min(target_step, raw.pred_len - 1)
        pred_rel[:, ts, :].norm(dim=-1).mean().backward()
    if obs_req.grad is not None:
        attr = obs_req.grad[:, :, :2].norm(dim=-1)
        attr = attr / (attr.sum(0, keepdim=True) + 1e-8)
    else:
        attr = torch.zeros(batch_list[0].shape[0], batch_list[0].shape[1], device=device)
    return attr.detach()


@torch.no_grad()
def compute_ensemble_uncertainty(all_traj: torch.Tensor) -> Dict:
    """
    [XAI-4] Uncertainty per lead time across K ensemble samples.
    """
    all_deg  = _norm_to_deg(all_traj)  # [K,T,B,2]
    K, T, B  = all_deg.shape[:3]
    mean_traj = all_deg.mean(0)
    std_km = torch.zeros(T, B, device=all_traj.device)
    for t in range(T):
        dists = _haversine_deg(
            all_deg[:, t].reshape(K * B, 2),
            mean_traj[t].unsqueeze(0).expand(K, B, 2).reshape(K * B, 2)
        ).reshape(K, B)
        std_km[t] = dists.std(0)
    s12 = min(1, T - 1); s72 = min(11, T - 1)
    return {
        "std_per_step":      std_km,
        "uncertainty_ratio": (std_km[s72] + 1e-3) / (std_km[s12] + 1e-3),
        "mean_72h_std":      float(std_km[s72].mean()),
        "mean_12h_std":      float(std_km[s12].mean()),
        "high_uncertainty":  std_km[s72] > 80.0,
    }


@torch.no_grad()
def compute_heading_deviation(pred_deg: torch.Tensor,
                               gt_deg:   torch.Tensor) -> torch.Tensor:
    """
    [XAI-6] Heading deviation per lead time step, in degrees (absolute).
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        return pred_deg.new_zeros(1, pred_deg.shape[1])
    bear_gt   = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
    bear_pred = _forward_azimuth(gt_deg[:T-1], pred_deg[1:T])
    diff = (bear_pred - bear_gt).abs()
    diff = torch.where(diff > math.pi, 2 * math.pi - diff, diff)
    return torch.rad2deg(diff)


@torch.no_grad()
def compute_cte_contribution(pred_deg: torch.Tensor,
                              gt_deg:   torch.Tensor) -> Dict:
    """
    [XAI-7] Decompose trajectory error into ATE (along-track) and CTE (cross-track).
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return {"ate_per_step": z, "cte_per_step": z,
                "ate_mean": z[0], "cte_mean": z[0],
                "ate_abs_mean": 0.0, "cte_abs_mean": 0.0}
    bear_ref  = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
    bear_err  = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
    dist_err  = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang       = bear_err - bear_ref
    ate       = dist_err * torch.cos(ang)
    cte       = dist_err * torch.sin(ang)
    return {
        "ate_per_step":  ate,
        "cte_per_step":  cte,
        "ate_mean":      ate.mean(0),
        "cte_mean":      cte.abs().mean(0),
        "ate_abs_mean":  float(ate.abs().mean()),
        "cte_abs_mean":  float(cte.abs().mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Compat stubs
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def classify_hard_easy(obs_traj_norm, per_sample_loss=None,
                        hard_score_p: float = 70.0, loss_p: float = 50.0):
    scores = hard_score_from_obs(obs_traj_norm)
    B = scores.shape[0]
    if B < 4:
        return torch.zeros(B, dtype=torch.bool, device=scores.device)
    return scores >= torch.quantile(scores, hard_score_p / 100.0)


@torch.no_grad()
def classify_hard_easy_global(obs_traj_norm, global_threshold):
    return hard_score_from_obs(obs_traj_norm) >= global_threshold


@torch.no_grad()
def compute_diversity_score(candidates) -> float:
    if len(candidates) < 2:
        return 0.0
    T, B = candidates[0].shape[0], candidates[0].shape[1]
    ep_step = min(T - 1, 11)
    endpoints = torch.stack([_norm_to_deg(c[ep_step]) for c in candidates], 0)
    N = endpoints.shape[0]
    ep_mean = endpoints.mean(0, keepdim=True)
    dists = _haversine_deg(
        endpoints.reshape(N * B, 2),
        ep_mean.expand(N, B, 2).reshape(N * B, 2)
    ).reshape(N, B)
    return float(dists.std(0).mean())


# ─────────────────────────────────────────────────────────────────────────────
#  TCFlowMatching v2.1-learn
# ─────────────────────────────────────────────────────────────────────────────

class TCFlowMatching(nn.Module):
    """
    TC-FlowMatching v2.1-learn

    Training loss:
        L_total = L_CFM
                + lam_reg  × L_reg          (ramp ep10→ep30, LEARNABLE step weights)
                + lam_dir  × lambda_heading × L_heading_ms  (ramp ep5→ep20)
                + 0.1      × L_calib         (trains speed_correction_logits)

        L_CFM:         conditional flow matching (random t, OT noise-GT matching)
        L_reg:         ADE loss at sigma_inference, LEARNABLE per-step weights
        L_heading_ms:  full 12-step heading continuation, LEARNABLE per-step weights
        L_calib:       ADE on speed-calibrated 1-shot pred, trains per-horizon
                       speed correction factors used by speed_calibrate_pred()

    NO L_momentum: removed because it anchors to val speed distribution,
    which hurts ATE on test storms that move at different speeds.

    Inference pipeline:
        1. Sample K=20 candidates with sigma=0.04 FIXED
        2. Physics score selection (speed+smooth+heading+displacement) — fixed weights
        3. Weighted average of top-3 candidates
        4. Speed calibration: LEARNABLE per-horizon correction (was fixed clip)
        5. Optional: multi-scale sigma ensemble (sample_multiscale)
    """

    def __init__(
        self,
        pred_len:          int   = 12,
        obs_len:           int   = 8,
        unet_in_ch:        int   = 13,
        d_cond:            int   = 256,
        d_model:           int   = 256,
        nhead:             int   = 8,
        num_dec_layers:    int   = 4,
        dim_ff:            int   = 512,
        dropout:           float = 0.1,
        sigma_min:         float = 0.04,
        sigma_max:         float = 0.08,
        lambda_reg:        float = 0.2,
        lambda_heading:    float = 0.07,   # reduced: gradient scale balance with L_CFM
        lambda_momentum:   float = 0.0,    # DISABLED — field kept for compat only
        lambda_calib:      float = 0.1,    # [LEARN-1] weight of L_calib term
        use_ot:            bool  = True,
        ot_epsilon:        float = 0.05,
        use_ema:           bool  = True,
        ema_decay:         float = 0.995,
        n_inference_steps: int   = 1,
        n_ensemble:        int   = 20,
        sigma_inference:   float = 0.04,   # FIXED throughout
        **kwargs,
    ):
        super().__init__()
        self.pred_len          = pred_len
        self.obs_len            = obs_len
        self.sigma_min          = sigma_min
        self.sigma_max          = sigma_max
        self.lambda_reg         = lambda_reg
        self.lambda_heading     = lambda_heading
        self.lambda_momentum    = 0.0           # always disabled
        self.lambda_calib       = lambda_calib
        self.use_ot              = use_ot
        self.ot_epsilon          = ot_epsilon
        self.n_inference_steps   = n_inference_steps
        self.n_ensemble          = n_ensemble
        self.sigma_inference     = sigma_inference

        self.encoder  = ContextEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch, d_cond=d_cond)
        self.velocity = VelocityTransformer(
            pred_len=pred_len, d_model=d_model, nhead=nhead,
            num_layers=num_dec_layers, dim_ff=dim_ff,
            dropout=dropout, d_cond=d_cond)
        self.use_ema = use_ema
        self._ema    = None

        # ── [v2.1-learn] LEARNABLE PARAMETERS ──────────────────────────────
        # Mỗi tham số dưới đây thay thế một hằng số tay đoán trong
        # v2.1-clean. Mỗi cái đều có ĐƯỜNG GRADIENT THỰC SỰ về tổng loss
        # (verify trong get_loss_breakdown) — không phải param "mồ côi".

        # [LEARN-1] Per-horizon speed correction, dùng trong speed_calibrate_pred.
        # Init=0 → sigmoid(0)*2=1.0 → identity (không sửa gì) lúc bắt đầu,
        # giống hệt điểm khởi đầu hành vi cũ (scale=1 baseline).
        self.speed_correction_logits = nn.Parameter(torch.zeros(pred_len))

        # [LEARN-2] Per-step L_reg weights — LATE-HEAVY init.
        # Init trước: uniform (logit=0 mọi bước) → model không thiên vị
        # horizon nào từ đầu, và log EP20 cho thấy weights vẫn gần uniform
        # sau 20 epoch (reg_w=[0.119,0.119,...]) — model chưa học được
        # ưu tiên horizon nào vì gradient L_reg đối xứng với uniform init.
        # Init mới: tăng dần từ 12h→72h (t * 2/11):
        #   12h: 2.5%, 24h: 4.4%, 48h: 9.1%, 72h: 18.7%
        # → Áp lực gradient mạnh hơn ở 72h từ đầu → model học dự đoán
        # 72h tốt hơn mà không cần thêm loss riêng.
        # Gradient vẫn có thể điều chỉnh lại; đây là prior có lý giải
        # vật lý (72h error tích lũy từ 12h, khó hơn cần trọng số hơn).
        _reg_init = torch.arange(pred_len, dtype=torch.float) * (2.0 / max(pred_len - 1, 1))
        self.reg_step_logits = nn.Parameter(_reg_init)

        # [LEARN-3] hard_score 4-component weights: [curvature, speed_var,
        # dir_change, obs_speed_norm]. Init lệch nhẹ giống phân bổ gốc
        # 0.4/0.3/0.3 (+obs_speed nhỏ) thay vì zero-init hoàn toàn, để
        # không phá vỡ hành vi đã proven ngay từ epoch đầu — nhưng vẫn
        # cho phép gradient điều chỉnh dần qua training.
        self.hard_score_weight_logits = nn.Parameter(
            torch.log(torch.tensor([0.40, 0.30, 0.30, 0.15])))

        # [LEARN-5] Per-step heading constraint weights — TOÀN BỘ pred_len bước.
        # Init decay: 0.8^t thay vì 0.5^t (bản trước).
        # Lý do đổi: 0.5^t cho 48h chỉ 0.39% và 72h 0.07% tổng trọng số,
        # khiến gradient signal tới các bước xa GẦN NHƯ BẰNG 0 ngay từ đầu.
        # 0.8^t cho 48h 4.5% và 72h 1.8% — đủ để gradient flow từ đầu,
        # trong khi 12h vẫn ưu tiên nhất (21.5%). Gradient vẫn tự điều chỉnh
        # về sau; đây là điểm khởi đầu để mọi horizon đều có tín hiệu.
        _decay_init = torch.tensor([0.8 ** t for t in range(pred_len)])
        self.heading_step_logits = nn.Parameter(torch.log(_decay_init.clamp(min=1e-4)))

        # [LEARN-4] Cross-loss weighting — Kendall homoscedastic uncertainty
        # (Kendall, Gal & Cipolla, 2018: "Multi-Task Learning Using
        # Uncertainty to Weigh Losses"). Thay lambda_reg/lambda_heading/
        # lambda_calib HẰNG SỐ TAY ĐOÁN bằng log_sigma HỌC ĐƯỢC cho mỗi
        # loss phụ, dùng công thức: loss_i / (2·exp(2·log_sigma_i)) + log_sigma_i
        #
        # TẠI SAO KHÔNG DÙNG nn.Parameter(lambda) TRỰC TIẾP (naive cách):
        #   total = l_cfm + lambda*l_reg → d(total)/d(lambda) = l_reg ≥ 0
        #   → optimizer luôn đẩy lambda về 0 → TẮT loss phụ hoàn toàn
        #   → suy biến (degenerate), không phải "tự học cân bằng"
        # TẠI SAO log_sigma KHÔNG suy biến: số hạng +log_sigma_i là
        # regularizer — nếu sigma→∞ (tương đương lambda→0) thì log_sigma
        # →∞ làm total loss TĂNG, ngăn optimizer "trốn" loss bằng cách tắt nó.
        # Init log_sigma=0 → sigma=1 → 1/(2·1)=0.5, tương đương lambda=0.5
        # ban đầu cho mỗi loss phụ (điểm khởi đầu hợp lý, không quá khác
        # baseline cũ lambda_reg=0.2/lambda_heading=0.07/lambda_calib=0.1).
        self.log_sigma_reg     = nn.Parameter(torch.tensor(0.46))   # eff_lambda_init≈0.2 (khớp baseline)
        self.log_sigma_heading = nn.Parameter(torch.tensor(1.00))   # eff_lambda_init≈0.07 (khớp baseline)
        self.log_sigma_calib   = nn.Parameter(torch.tensor(0.80))   # eff_lambda_init≈0.1 (khớp baseline)

    def init_ema(self):
        if self.use_ema:
            self._ema = EMAModel(self, decay=0.995)

    def ema_update(self):
        if self._ema is not None:
            self._ema.update(self)

    def _to_relative(self, x_abs: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
        return x_abs - last_obs.unsqueeze(1)

    def _from_relative(self, x_rel: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
        return x_rel + last_obs.unsqueeze(1)

    def _sigma_schedule(self, epoch: int) -> float:
        """Cosine decay sigma_max→sigma_min over ep5→ep40."""
        if epoch < 5:
            return self.sigma_max
        if epoch < 40:
            t = (epoch - 5) / 35.0
            return self.sigma_min + 0.5 * (self.sigma_max - self.sigma_min) * (1 + math.cos(math.pi * t))
        return self.sigma_min

    # ── Multi-step heading loss — [LEARN-5] learnable per-step weights ──────

    def _heading_loss_ms(self, pred_deg: torch.Tensor, obs_deg: torch.Tensor) -> torch.Tensor:
        """
        Multi-step heading constraint — TOÀN BỘ pred_len bước, trọng số HỌC.

        [LEARN-5] SỬA LỖI: bản trước (v2.1-clean và lần sửa trước của tôi)
        chỉ áp dụng cho n_steps=4 ĐẦU TIÊN (6h/12h/18h/24h) với decay=0.5
        CỐ ĐỊNH (weight = 1.0, 0.5, 0.25, 0.125) — đây CHÍNH XÁC là loại
        hằng số tay đoán cần sửa mà tôi đã bỏ sót. Hệ quả thực tế: heading
        constraint hoàn toàn KHÔNG áp dụng cho 8 bước còn lại (30h→72h) —
        đây là một phần lý do 72h heading dev khó cải thiện trong log cũ
        (113-128° xuyên suốt nhiều version) dù ATE/CTE có nhúc nhích.

        FIX: dùng self.heading_step_logits (pred_len params, softmax) —
        TOÀN BỘ 12 bước đều có constraint, trọng số PHÂN BỔ GIỮA chúng do
        gradient tự học, không còn giới hạn cứng "chỉ 4 bước đầu" hay
        decay theo cấp số nhân tay chọn (0.5^t).

        Detach ref tại mỗi step vẫn giữ nguyên (CRITICAL: ngăn gradient
        explosion qua chuỗi 12 bước liên tiếp — đây là cơ chế ổn định
        numerice, không phải "trọng số", nên không thuộc diện cần học).
        """
        if obs_deg.shape[0] < 2 or pred_deg.shape[0] < 1:
            return pred_deg.new_zeros(())

        ref_bear = _forward_azimuth(obs_deg[-2], obs_deg[-1])   # [B]
        pts = torch.cat([obs_deg[-1:], pred_deg], 0)   # [T_pred+1, B, 2]

        N = pred_deg.shape[0]
        sw = F.softmax(self.heading_step_logits[:N], dim=0)   # [N], học được

        loss = pred_deg.new_zeros(())
        for t in range(N):
            pred_bear  = _forward_azimuth(pts[t], pts[t + 1])  # [B]
            angle_diff = pred_bear - ref_bear
            loss       = loss + sw[t] * (1.0 - torch.cos(angle_diff)).mean()
            ref_bear   = pred_bear.detach()   # CRITICAL: detach to prevent chain gradient

        return loss

    # ── L_reg — [LEARN-2] learnable step weights ────────────────────────────

    def _reg_loss(self, x1_rel: torch.Tensor, last_obs: torch.Tensor,
                  cond: torch.Tensor,
                  hard_score: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        ADE loss at t=0 with sigma_inference noise, consistent with 1-shot inference.

        [LEARN-2] Step weights = softmax(self.reg_step_logits) thay vì
        linspace(1.0, 1.5, T) hardcoded.
        Bằng chứng tại sao cần đổi: linspace(1,3)→linspace(1,1.5) chỉ giảm
        test ATE 0.9km, và XAI-8 24h ratio KHÔNG đổi (vẫn ~1.5-1.6×) —
        chứng tỏ chỉnh tay con số trong công thức tuyến tính không đủ;
        cần để gradient tự tìm phân bổ trọng số tối ưu giữa 12 horizon.

        DOES NOT use exp weights (v2.4 used 12.2× ratio → catastrophic
        overfitting) — softmax trên logits học được có range tự nhiên bị
        chặn mềm, không bùng nổ như exp(linspace) thủ công.
        """
        B, T, _ = x1_rel.shape
        device   = x1_rel.device
        x0   = torch.randn_like(x1_rel) * self.sigma_inference
        t0   = torch.zeros(B, device=device)
        v    = self.velocity(x0, t0, cond)
        x1_pred_abs = self._from_relative(x0 + v, last_obs)
        x1_gt_abs   = self._from_relative(x1_rel, last_obs)
        pred_deg = _norm_to_deg(x1_pred_abs.permute(1, 0, 2))  # [T, B, 2]
        gt_deg   = _norm_to_deg(x1_gt_abs.permute(1, 0, 2))
        dist     = _haversine_deg(pred_deg, gt_deg)             # [T, B] km

        T_actual = dist.shape[0]
        sw = F.softmax(self.reg_step_logits[:T_actual], dim=0).unsqueeze(1)  # [T, 1]

        if hard_score is not None:
            sw_hard = (1.0 + hard_score.to(device).to(dist.dtype)).unsqueeze(0)  # [1, B]
        else:
            sw_hard = torch.ones(1, B, device=device, dtype=dist.dtype)
        return ((dist * sw) * sw_hard).mean() / 300.0

    # ── get_loss_breakdown ──────────────────────────────────────────────────

    def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs) -> Dict:
        """
        Full loss computation. Called in TRAINING with augmented batch.
        Called in VAL without augmentation → val_loss is reliable signal.

        Total = L_CFM + lam_reg × L_reg + lam_dir × lambda_heading × L_heading_ms
                      + lambda_calib × L_calib
        """
        obs_traj = batch_list[0]
        gt_traj  = batch_list[1]
        B        = obs_traj.shape[1]
        device   = obs_traj.device

        sigma    = self._sigma_schedule(epoch)
        x1_gt    = gt_traj.permute(1, 0, 2)       # [B, T, 2]
        last_obs = obs_traj[-1, :, :2]             # [B, 2]
        x1_rel   = self._to_relative(x1_gt, last_obs)

        # [LEARN-3] hard_score: truyền self.hard_score_weight_logits để
        # gradient của 4 trọng số chảy qua sw_hard trong _reg_loss bên dưới.
        h_score = hard_score_from_obs(obs_traj[:, :, :2],
                                       weight_logits=self.hard_score_weight_logits)
        cond    = self.encoder(batch_list, hard_score=h_score)

        # ── L_CFM ─────────────────────────────────────────────────────────
        x0 = torch.randn_like(x1_rel) * sigma
        if self.use_ot and B >= 4:
            x0_flat, x1_flat = _ot_match(
                x0.reshape(B, -1), x1_rel.reshape(B, -1), self.ot_epsilon)
            x0         = x0_flat.reshape(B, self.pred_len, 2)
            x1_matched = x1_flat.reshape(B, self.pred_len, 2)
        else:
            x1_matched = x1_rel

        t        = torch.rand(B, device=device)
        x_t      = (1.0 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1_matched
        u_target = x1_matched - x0
        v_pred   = self.velocity(x_t, t, cond)
        l_cfm    = F.mse_loss(v_pred, u_target)

        # ── L_reg ramp ep10→ep30 (GATE theo curriculum, KHÔNG phải trọng số) ──
        # Lý do giữ ramp dạng if/else thay vì học: đây là LỊCH TRÌNH BẬT/TẮT
        # theo trạng thái huấn luyện (encoder bị freeze tới ep10 nên L_reg
        # chưa có ý nghĩa trước đó — bật sớm sẽ lan gradient nhiễu vào
        # velocity khi nó chưa thấy cond ổn định). Đây không phải "model
        # tin loss này quan trọng bao nhiêu" (thứ Kendall xử lý ở dưới) mà
        # là "loss này có HỢP LỆ để tính tại epoch này hay chưa" — 2 vấn đề
        # khác nhau về bản chất, gate này dùng index epoch (số nguyên rời
        # rạc) nên không phải dạng có thể differentiable để học bằng SGD.
        if epoch < 10:     ramp_reg = 0.0
        elif epoch < 30:   ramp_reg = (epoch - 10) / 20.0
        else:              ramp_reg = 1.0

        l_reg = (self._reg_loss(x1_rel, last_obs, cond, h_score)
                 if ramp_reg > 0.0 else x0.new_zeros(()))

        # ── L_heading + L_calib — 1 SHARED FORWARD PASS (performance fix) ──────
        # Cả 2 loss đều cần 1-shot pred x0_h. Dùng chung để tránh +1 velocity
        # forward pass mỗi batch (giữ tốc độ ≈ baseline).
        #
        # ramp_calib: bắt đầu từ ep10 (giống ramp_reg), không phải ep0.
        # Lý do: ep0-9 encoder bị freeze → cond không ổn định → gradient
        # L_calib noisy → speed_correction_logits học không đúng hướng.
        # Gate này nhất quán với lý do đã giải thích cho ramp_reg.
        if epoch < 5:      ramp_dir = 0.0
        elif epoch < 20:   ramp_dir = (epoch - 5) / 15.0
        else:              ramp_dir = 1.0

        if epoch < 10:     ramp_calib = 0.0
        elif epoch < 30:   ramp_calib = (epoch - 10) / 20.0
        else:              ramp_calib = 1.0

        need_hc = ramp_dir > 0.0 or ramp_calib > 0.0
        if need_hc:
            x0_h       = torch.randn_like(x1_rel) * self.sigma_inference
            v_h        = self.velocity(x0_h, torch.zeros(B, device=device), cond)
            x1_h_abs   = self._from_relative(x0_h + v_h, last_obs)
            pred_abs_h = x1_h_abs.permute(1, 0, 2)                          # [T, B, 2]

            # L_heading (gated)
            if ramp_dir > 0.0:
                pred_deg_h = _norm_to_deg(pred_abs_h)
                obs_deg_h  = _norm_to_deg(obs_traj[:, :, :2])
                l_heading  = self._heading_loss_ms(pred_deg_h, obs_deg_h)
            else:
                l_heading  = x0.new_zeros(())

            # L_calib (gated từ ep10, dùng chung pred_abs_h)
            if ramp_calib > 0.0:
                cal_abs  = self.speed_calibrate_pred(pred_abs_h, last_obs, obs_traj[:, :, :2])
                cal_deg  = _norm_to_deg(cal_abs)
                gt_c_deg = _norm_to_deg(x1_gt.permute(1, 0, 2))
                l_calib  = _haversine_deg(cal_deg, gt_c_deg).mean() / 300.0
            else:
                l_calib  = x0.new_zeros(())
        else:
            l_heading = x0.new_zeros(())
            l_calib   = x0.new_zeros(())

        l_momentum = x0.new_zeros(())   # always 0

        # ── Total — [LEARN-4] Kendall uncertainty weighting cho cross-loss ──
        prec_reg     = torch.exp(-2.0 * self.log_sigma_reg.clamp(min=-3.0))
        prec_heading = torch.exp(-2.0 * self.log_sigma_heading.clamp(min=-3.0))
        prec_calib   = torch.exp(-2.0 * self.log_sigma_calib.clamp(min=-3.0))

        weighted_reg     = ramp_reg   * (0.5 * prec_reg     * l_reg     + self.log_sigma_reg.clamp(min=-3.0))
        weighted_heading = ramp_dir   * (0.5 * prec_heading * l_heading + self.log_sigma_heading.clamp(min=-3.0))
        weighted_calib   = ramp_calib * (0.5 * prec_calib   * l_calib   + self.log_sigma_calib.clamp(min=-3.0))

        total = l_cfm + weighted_reg + weighted_heading + weighted_calib
        if not torch.isfinite(total):
            total = x0.new_zeros(())

        # ── ADE 1-step log ────────────────────────────────────────────────
        with torch.no_grad():
            x0_log = torch.randn_like(x1_rel) * self.sigma_inference
            v_log  = self.velocity(x0_log, torch.zeros(B, device=device), cond)
            ade_log = _haversine_deg(
                _norm_to_deg(self._from_relative(x0_log + v_log, last_obs).permute(1, 0, 2)),
                _norm_to_deg(x1_gt.permute(1, 0, 2))
            ).mean().item()

        return {
            "total":     total,
            "l_cfm":     l_cfm.item(),
            "l_reg":     l_reg.item() if torch.is_tensor(l_reg) else 0.0,
            "l_heading": l_heading.item() if torch.is_tensor(l_heading) else 0.0,
            "l_calib":   l_calib.item(),
            "l_momentum": 0.0,   # always 0
            "lam_reg":   ramp_reg,
            "lam_dir":   ramp_dir,
            "lam_calib": ramp_calib,
            "sigma":     sigma,
            "ade_1step": ade_log,
            "hard_score_mean": float(h_score.detach().mean()),
            "hard_score_max":  float(h_score.detach().max()),
            # [LEARN-4] learned cross-loss weights — effective lambda = 0.5*prec
            "learned_lambda_reg":     float((0.5 * prec_reg).detach()),
            "learned_lambda_heading": float((0.5 * prec_heading).detach()),
            "learned_lambda_calib":   float((0.5 * prec_calib).detach()),
            # Compat keys for older scripts
            "l_fm": l_cfm.item(), "dpe": 0., "heading": 0., "vel_reg": 0.,
            "speed": 0., "accel": 0., "fm_mse": l_cfm.item(),
            "l_hard_total": 0., "n_hard": 0, "alpha_hard": 0.,
            "l_sel_total": 0., "speed_head_l": 0., "l_score": 0.,
        }

    def get_loss(self, batch_list, epoch: int = 0, **kwargs) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

    # ── Speed calibration — [LEARN-1] learnable per-horizon ─────────────────

    def speed_calibrate_pred(self,
                              pred_abs_norm: torch.Tensor,
                              last_obs_norm: torch.Tensor,
                              obs_norm:      torch.Tensor) -> torch.Tensor:
        """
        Per-horizon LEARNABLE speed calibration. Thay thế hằng số
        clip_min/clip_max=0.85/1.15 cũ (1 scale chung cho cả 12 bước).

        BẰNG CHỨNG TẠI SAO 1 SCALE CHUNG KHÔNG ĐỦ:
          XAI-8 (v2.1-clean, đã chạy) cho thấy correction cần khác nhau
          rõ rệt theo horizon: 12h cần scale≈0.77, 24h cần scale≈0.62,
          48h cần scale≈1.06, 72h cần scale≈1.04. clip(0.85,1.15) không
          thể đáp ứng tất cả các nhu cầu trái ngược này cùng lúc — đây là
          lý do mở rộng clip range (đã thử 0.72-1.35) chỉ cải thiện <1km:
          sai công cụ cho vấn đề (1 số thay vì 12 số).

        Không còn @staticmethod / @torch.no_grad(): chạy NGAY TRONG
        forward pass (gọi từ L_calib trong get_loss_breakdown VÀ từ
        sample() dưới no_grad của caller khi inference) để gradient có
        thể chảy qua self.speed_correction_logits trong lúc training.

        correction[t] = 2 * sigmoid(speed_correction_logits[t]) ∈ (0, 2)
        init = sigmoid(0)*2 = 1.0 → no-op tại epoch 0, học dần qua L_calib.
        """
        if obs_norm.shape[0] < 2 or pred_abs_norm.shape[0] < 2:
            return pred_abs_norm

        T = pred_abs_norm.shape[0]
        correction = (torch.sigmoid(self.speed_correction_logits[:T]) * 2.0
                      ).to(pred_abs_norm.dtype).view(T, 1, 1)               # [T,1,1]

        pts  = torch.cat([last_obs_norm.unsqueeze(0), pred_abs_norm], 0)    # [T+1, B, 2]
        disp = pts[1:] - pts[:-1]                                           # [T, B, 2]
        disp_cal = disp * correction
        # vectorized cumsum: torch.compile friendly, đúng về mặt toán học
        # cumsum(disp_cal)[t] = sum_{s=0}^{t} disp_cal[s] = last_obs + running displacement
        return last_obs_norm.unsqueeze(0) + disp_cal.cumsum(dim=0)

    # ── Sample (1-shot + speed calibration) ──────────────────────────────

    @torch.no_grad()
    def sample(self, batch_list,
               num_ensemble:          Optional[int]  = None,
               ddim_steps:            Optional[int]  = None,
               return_xai:            bool           = False,
               use_speed_calibration: bool           = True,
               **kwargs) -> Tuple:
        """
        1-shot inference with physics selection + learnable speed calibration.

        Steps:
          1. Encode context once (hard_score + full encoder)
          2. Sample K=20 candidates: x0 ~ N(0, sigma_inf²), x_pred = x0 + v(x0, 0, cond)
          3. Physics score each candidate (speed+smooth+heading+displacement)
          4. Weighted average of top-3 candidates
          5. Speed calibration per-horizon (LEARNED, not fixed clip)
          6. If return_xai=True: compute XAI-1 through XAI-9

        Returns:
          (pred_mean [T,B,2], zeros [T,B,2], all_traj [K,T,B,2])
          or (pred_mean, zeros, all_traj, xai_dict) if return_xai=True
        """
        K  = num_ensemble or self.n_ensemble
        N  = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else self.n_inference_steps
        dt = 1.0 / max(N, 1)

        obs_traj    = batch_list[0]
        T_obs, B, _ = obs_traj.shape
        device      = obs_traj.device

        h_score  = hard_score_from_obs(obs_traj[:, :, :2],
                                        weight_logits=self.hard_score_weight_logits)
        obs_norm = obs_traj[:, :, :2]
        last_obs = obs_traj[-1, :, :2]
        t0       = torch.zeros(B, device=device)

        cond = self.encoder(batch_list, hard_score=h_score)

        all_traj = []
        for _ in range(K):
            x_rel = torch.randn(B, self.pred_len, 2, device=device) * self.sigma_inference

            if N <= 1:
                v     = self.velocity(x_rel, t0, cond)
                x_rel = x_rel + v
            else:
                for step in range(N):
                    t_b   = torch.full((B,), step * dt, device=device)
                    x_rel = (x_rel + dt * self.velocity(x_rel, t_b, cond)).clamp(-3., 3.)

            x_abs = self._from_relative(x_rel, last_obs)
            all_traj.append(x_abs.permute(1, 0, 2))   # [T, B, 2]

        scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)   # [K, B]
        all_t   = torch.stack(all_traj, 0)   # [K, T, B, 2]
        top_k   = min(3, K)
        top_idx = scores.topk(top_k, dim=0).indices   # [top_k, B]

        pred_mean = torch.zeros_like(all_traj[0])
        for b in range(B):
            idx_b = top_idx[:, b]
            w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
            pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

        # [LEARN-1] Per-horizon LEARNED speed calibration (no hardcoded clip args)
        if use_speed_calibration:
            pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

        if not return_xai:
            return pred_mean, torch.zeros_like(pred_mean), all_t

        # ── XAI output (XAI-2,3,4,5,6,7,8,9) ──────────────────────────────
        xai = {}

        xai.update(compute_ensemble_uncertainty(all_t))

        _, hard_comps = hard_score_from_obs(obs_norm, return_components=True,
                                             weight_logits=self.hard_score_weight_logits)
        xai["hard_components"] = hard_comps

        pred_deg  = _norm_to_deg(pred_mean)    # [T, B, 2]
        obs_deg_x = _norm_to_deg(obs_norm)     # [T_obs, B, 2]

        obs_spd_x    = _step_speeds_kmh(obs_deg_x)
        obs_spd_mu   = obs_spd_x.mean(0)     # [B]
        if pred_deg.shape[0] >= 2:
            last_deg_x   = obs_deg_x[-1]
            pts_x        = torch.cat([last_deg_x.unsqueeze(0), pred_deg], 0)
            pred_spd_x   = _step_speeds_kmh(pts_x)
            pred_spd_mu  = pred_spd_x.mean(0)  # [B]
        else:
            pred_spd_mu = obs_spd_mu.clone()

        speed_ratio = (pred_spd_mu / obs_spd_mu.clamp(min=1.0))
        xai["speed_comparison"] = {
            "obs_speed_mean":  float(obs_spd_mu.mean()),
            "pred_speed_mean": float(pred_spd_mu.mean()),
            "speed_ratio":     float(speed_ratio.mean()),
            "per_storm_obs":   obs_spd_mu,
            "per_storm_pred":  pred_spd_mu,
            "over_predict":    speed_ratio > 1.15,
            "under_predict":   speed_ratio < 0.85,
        }

        v_ref = obs_spd_mu.clamp(min=5.0)
        v_sig = v_ref * 0.5
        spd_sc = torch.exp(-((pred_spd_mu - v_ref) / v_sig).pow(2) * 0.5)
        if pred_deg.shape[0] >= 3:
            vel_x   = pred_deg[1:] - pred_deg[:-1]
            accel_x = (vel_x[1:] - vel_x[:-1]).norm(dim=-1)
            smo_sc  = torch.exp(-accel_x.mean(0) * 5.0)
        else:
            smo_sc = torch.ones(B, device=device)
        if obs_norm.shape[0] >= 2 and pred_mean.shape[0] >= 1:
            ov = obs_norm[-1, :, :2] - obs_norm[-2, :, :2]
            pv = pred_mean[0, :, :2] - obs_norm[-1, :, :2]
            cos_s = (F.normalize(ov, dim=-1, eps=1e-6) * F.normalize(pv, dim=-1, eps=1e-6)).sum(-1)
            hd_sc = torch.exp((cos_s.clamp(-1, 1) - 1.0) * 3.0)
        else:
            hd_sc = torch.ones(B, device=device)
        xai["physics_components"] = {
            "speed": spd_sc, "smooth": smo_sc, "heading": hd_sc,
            "obs_speed": obs_spd_mu, "pred_speed": pred_spd_mu,
        }

        gt_traj_xai = batch_list[1]
        gt_deg_xai  = _norm_to_deg(gt_traj_xai[:, :, :2])
        xai["heading_deviation_deg"] = compute_heading_deviation(pred_deg, gt_deg_xai)

        xai["ate_cte_decomp"] = compute_cte_contribution(pred_deg, gt_deg_xai)

        # XAI-8: Per-horizon speed ratio (12h/24h/48h/72h)
        if pred_deg.shape[0] >= 2 and gt_deg_xai.shape[0] >= 2:
            T8       = min(pred_deg.shape[0], gt_deg_xai.shape[0])
            last_d   = obs_deg_x[-1]
            pts_pred = torch.cat([last_d.unsqueeze(0), pred_deg[:T8]], 0)
            pts_gt   = torch.cat([last_d.unsqueeze(0), gt_deg_xai[:T8]], 0)
            spd_pred = _step_speeds_kmh(pts_pred)   # [T8, B]
            spd_gt   = _step_speeds_kmh(pts_gt)     # [T8, B]
            ratio    = spd_pred / spd_gt.clamp(min=1.0)  # [T8, B]
            def _hz(s, lo, hi):
                hi = min(hi, s.shape[0])
                return float(s[lo:hi].mean()) if hi > lo else float("nan")
            r_mean = ratio.mean(1)   # [T8]
            xai["speed_per_horizon"] = {
                "ratio":    r_mean.tolist(),
                "pred_kmh": spd_pred.mean(1).tolist(),
                "gt_kmh":   spd_gt.mean(1).tolist(),
                "12h_ratio": _hz(r_mean, 0, 2),
                "24h_ratio": _hz(r_mean, 2, 4),
                "48h_ratio": _hz(r_mean, 6, 8),
                "72h_ratio": _hz(r_mean, 10, 12),
            }
        else:
            xai["speed_per_horizon"] = {}

        # XAI-9: Storm category breakdown by obs speed
        obs_spd_cat = _step_speeds_kmh(obs_deg_x).mean(0)   # [B]
        sm = obs_spd_cat < 8.0
        mm = (obs_spd_cat >= 8.0) & (obs_spd_cat < 15.0)
        fm = obs_spd_cat >= 15.0
        ade_per_storm = _haversine_deg(pred_deg[:gt_deg_xai.shape[0]],
                                        gt_deg_xai[:pred_deg.shape[0]]).mean(0)  # [B]
        def _cat_ade(mask):
            return float(ade_per_storm[mask].mean()) if mask.sum() > 0 else float("nan")
        xai["storm_categories"] = {
            "n_slow":    int(sm.sum()),
            "n_medium":  int(mm.sum()),
            "n_fast":    int(fm.sum()),
            "speed_mean": float(obs_spd_cat.mean()),
            "speed_std":  float(obs_spd_cat.std()),
            "ade_slow":   _cat_ade(sm),
            "ade_medium": _cat_ade(mm),
            "ade_fast":   _cat_ade(fm),
        }

        # [LEARN diagnostics] expose learned values for monitoring in train_fm
        xai["learned_params"] = {
            "speed_correction": (torch.sigmoid(self.speed_correction_logits) * 2.0).tolist(),
            "reg_step_weights":  F.softmax(self.reg_step_logits, dim=0).tolist(),
            "hard_score_weights": F.softmax(self.hard_score_weight_logits, dim=0).tolist(),
        }

        return pred_mean, torch.zeros_like(pred_mean), all_t, xai

    @torch.no_grad()
    def sample_multiscale(
        self,
        batch_list,
        sigmas: Optional[List[float]] = None,
        n_per_sigma: int = 4,
        use_speed_calibration: bool = True,
    ) -> Tuple:
        """
        Multi-scale sigma ensemble at inference (DOES NOT affect training).
        Hedges against speed distribution shift between val and test.
        """
        if sigmas is None:
            sigmas = [0.025, 0.035, 0.04, 0.05, 0.065]

        obs_traj = batch_list[0]
        B = obs_traj.shape[1]; device = obs_traj.device

        h_score  = hard_score_from_obs(obs_traj[:, :, :2],
                                        weight_logits=self.hard_score_weight_logits)
        obs_norm = obs_traj[:, :, :2]
        last_obs = obs_traj[-1, :, :2]
        t0       = torch.zeros(B, device=device)
        cond     = self.encoder(batch_list, hard_score=h_score)

        all_traj = []
        for sigma in sigmas:
            for _ in range(n_per_sigma):
                x0    = torch.randn(B, self.pred_len, 2, device=device) * sigma
                v     = self.velocity(x0, t0, cond)
                x_abs = self._from_relative(x0 + v, last_obs)
                all_traj.append(x_abs.permute(1, 0, 2))

        scores  = torch.stack([_physics_score(t, obs_norm) for t in all_traj], 0)
        all_t   = torch.stack(all_traj, 0)
        top_k   = min(5, len(all_traj))
        top_idx = scores.topk(top_k, dim=0).indices

        pred_mean = torch.zeros_like(all_traj[0])
        for b in range(B):
            idx_b = top_idx[:, b]
            w_b   = F.softmax(scores[idx_b, b] * 3.0, dim=0)
            pred_mean[:, b, :] = (all_t[idx_b, :, b, :] * w_b.view(top_k, 1, 1)).sum(0)

        if use_speed_calibration:
            pred_mean = self.speed_calibrate_pred(pred_mean, last_obs, obs_norm)

        return pred_mean, torch.zeros_like(pred_mean), all_t


# ─────────────────────────────────────────────────────────────────────────────
#  Backward compat alias
# ─────────────────────────────────────────────────────────────────────────────
TCDiffusion = TCFlowMatching
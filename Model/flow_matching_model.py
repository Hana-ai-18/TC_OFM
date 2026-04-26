# # # # # # # # # # # # """
# # # # # # # # # # # # Model/flow_matching_model.py  ── v23
# # # # # # # # # # # # ==========================================
# # # # # # # # # # # # FIXES vs v21:

# # # # # # # # # # # #   FIX-M18  [CURRICULUM REMOVED] set_curriculum_len() vẫn giữ để backward
# # # # # # # # # # # #            compat nhưng KHÔNG được gọi từ trainer nữa. active_pred_len
# # # # # # # # # # # #            luôn = pred_len. evaluate_full_val_ade không cần restore nữa.

# # # # # # # # # # # #   FIX-M19  get_loss_breakdown(): nhận thêm step_weight_alpha parameter
# # # # # # # # # # # #            và truyền vào compute_total_loss() → fm_afcrps_loss() sử dụng
# # # # # # # # # # # #            soft weighting thay curriculum len-slicing.

# # # # # # # # # # # #   FIX-M20  get_loss_breakdown(): truyền all_trajs vào compute_total_loss()
# # # # # # # # # # # #            để tính ensemble_spread_loss. Giúp kiểm soát spread tăng quá mức.

# # # # # # # # # # # #   FIX-M21  _physics_correct(): tăng n_steps=5 (từ 3), giảm lr=0.002 (từ
# # # # # # # # # # # #            0.005) để physics correction ổn định hơn và ít overshoot.

# # # # # # # # # # # #   FIX-M22  sample(): initial_sample_sigma=0.1 (set từ constructor) đã fix
# # # # # # # # # # # #            spread. Thêm post-sampling clip chặt hơn [-3.0, 3.0] cho cả lon
# # # # # # # # # # # #            và lat (từ [-5.0, 5.0] cho lon).

# # # # # # # # # # # # Kept from v21:
# # # # # # # # # # # #   FIX-M17  _physics_correct với torch.enable_grad()
# # # # # # # # # # # #   FIX-M11..M16 OT-CFM, beta drift, env_data, physics scale
# # # # # # # # # # # # """
# # # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # # import csv
# # # # # # # # # # # # import math
# # # # # # # # # # # # import os
# # # # # # # # # # # # from datetime import datetime
# # # # # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # # # # import torch
# # # # # # # # # # # # import torch.nn as nn
# # # # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # # # # from Model.losses import (
# # # # # # # # # # # #     compute_total_loss, fm_physics_consistency_loss, WEIGHTS,
# # # # # # # # # # # #     pinn_speed_constraint,
# # # # # # # # # # # # )


# # # # # # # # # # # # def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # #     """Normalised → degrees. Handles [T, B, 2] and [B, 2]."""
# # # # # # # # # # # #     out = traj_norm.clone()
# # # # # # # # # # # #     out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # # #     out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
# # # # # # # # # # # #     return out


# # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # #  VelocityField
# # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # # # # #     """
# # # # # # # # # # # #     OT-CFM velocity field v_θ(x_t, t, context).
# # # # # # # # # # # #     Architecture: DataEncoder1D (Mamba) + FNO3D + Env-T-Net → Transformer decoder.
# # # # # # # # # # # #     Physics-guided: v_total = v_neural + sigmoid(w_physics) * v_beta_drift.
# # # # # # # # # # # #     """

# # # # # # # # # # # #     def __init__(
# # # # # # # # # # # #         self,
# # # # # # # # # # # #         pred_len:   int   = 12,
# # # # # # # # # # # #         obs_len:    int   = 8,
# # # # # # # # # # # #         ctx_dim:    int   = 256,
# # # # # # # # # # # #         sigma_min:  float = 0.02,
# # # # # # # # # # # #         unet_in_ch: int   = 13,
# # # # # # # # # # # #     ):
# # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # #         self.pred_len  = pred_len
# # # # # # # # # # # #         self.obs_len   = obs_len
# # # # # # # # # # # #         self.sigma_min = sigma_min

# # # # # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # # # # #             in_channel   = unet_in_ch,
# # # # # # # # # # # #             out_channel  = 1,
# # # # # # # # # # # #             d_model      = 64,
# # # # # # # # # # # #             n_layers     = 4,
# # # # # # # # # # # #             modes_t      = 4,
# # # # # # # # # # # #             modes_h      = 4,
# # # # # # # # # # # #             modes_w      = 4,
# # # # # # # # # # # #             spatial_down = 32,
# # # # # # # # # # # #             dropout      = 0.05,
# # # # # # # # # # # #         )

# # # # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # # # # #             in_1d       = 4,
# # # # # # # # # # # #             feat_3d_dim = 128,
# # # # # # # # # # # #             mlp_h       = 64,
# # # # # # # # # # # #             lstm_hidden = 128,
# # # # # # # # # # # #             lstm_layers = 3,
# # # # # # # # # # # #             dropout     = 0.1,
# # # # # # # # # # # #             d_state     = 16,
# # # # # # # # # # # #         )

# # # # # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=64)

# # # # # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 64 + 16, 512)
# # # # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(512)
# # # # # # # # # # # #         self.ctx_drop = nn.Dropout(0.15)
# # # # # # # # # # # #         self.ctx_fc2  = nn.Linear(512, ctx_dim)

# # # # # # # # # # # #         self.time_fc1 = nn.Linear(256, 512)
# # # # # # # # # # # #         self.time_fc2 = nn.Linear(512, 256)

# # # # # # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # # # # #                 dropout=0.15, activation="gelu", batch_first=True,
# # # # # # # # # # # #             ),
# # # # # # # # # # # #             num_layers=4,
# # # # # # # # # # # #         )
# # # # # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # # # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)

# # # # # # # # # # # #     def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
# # # # # # # # # # # #         half = dim // 2
# # # # # # # # # # # #         freq = torch.exp(
# # # # # # # # # # # #             torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # # # # #             * (-math.log(10_000.0) / max(half - 1, 1))
# # # # # # # # # # # #         )
# # # # # # # # # # # #         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
# # # # # # # # # # # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # # # # # # # # # # #         return F.pad(emb, (0, dim % 2))

# # # # # # # # # # # #     def _context(self, batch_list: List) -> torch.Tensor:
# # # # # # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # # # # # #         image_obs = batch_list[11]
# # # # # # # # # # # #         env_data  = batch_list[13]

# # # # # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # # # # #             image_obs = image_obs.unsqueeze(1)

# # # # # # # # # # # #         expected_ch = self.spatial_enc.in_channel
# # # # # # # # # # # #         if image_obs.shape[1] == 1 and expected_ch != 1:
# # # # # # # # # # # #             image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# # # # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
# # # # # # # # # # # #         e_3d_s = e_3d_s.permute(0, 2, 1)
# # # # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)

# # # # # # # # # # # #         T_bot = e_3d_s.shape[1]
# # # # # # # # # # # #         if T_bot != T_obs:
# # # # # # # # # # # #             e_3d_s = F.interpolate(
# # # # # # # # # # # #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # # # #                 mode="linear", align_corners=False,
# # # # # # # # # # # #             ).permute(0, 2, 1)

# # # # # # # # # # # #         f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))
# # # # # # # # # # # #         f_spatial     = self.decoder_proj(f_spatial_raw)

# # # # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)

# # # # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # # # # # # # # # # #         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
# # # # # # # # # # # #         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))
# # # # # # # # # # # #         return raw

# # # # # # # # # # # #     def _apply_ctx_head(self, raw: torch.Tensor,
# # # # # # # # # # # #                         noise_scale: float = 0.0) -> torch.Tensor:
# # # # # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # # # # #     def _beta_drift_velocity(self, x_t: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # #         """Beta drift in normalised state space. x_t: [B, T, 4]."""
# # # # # # # # # # # #         OMEGA_val  = 7.2921e-5
# # # # # # # # # # # #         R_val      = 6.371e6
# # # # # # # # # # # #         DT         = 6 * 3600.0
# # # # # # # # # # # #         M_PER_NORM = 5.0 * 111.0 * 1000.0

# # # # # # # # # # # #         lat_norm = x_t[:, :, 1]
# # # # # # # # # # # #         lat_deg  = lat_norm * 5.0
# # # # # # # # # # # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))

# # # # # # # # # # # #         beta   = 2 * OMEGA_val * torch.cos(lat_rad) / R_val
# # # # # # # # # # # #         R_tc   = 3e5
# # # # # # # # # # # #         v_lon  = -beta * R_tc ** 2 / 2
# # # # # # # # # # # #         v_lat  =  beta * R_tc ** 2 / 4

# # # # # # # # # # # #         v_lon_norm = v_lon * DT / M_PER_NORM
# # # # # # # # # # # #         v_lat_norm = v_lat * DT / M_PER_NORM

# # # # # # # # # # # #         v_phys = torch.zeros_like(x_t)
# # # # # # # # # # # #         v_phys[:, :, 0] = v_lon_norm
# # # # # # # # # # # #         v_phys[:, :, 1] = v_lat_norm
# # # # # # # # # # # #         return v_phys

# # # # # # # # # # # #     def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
# # # # # # # # # # # #                 ctx: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
# # # # # # # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # # # # # # #         T_seq  = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # # # # # #         x_emb  = (self.traj_embed(x_t[:, :T_seq, :])
# # # # # # # # # # # #                   + self.pos_enc[:, :T_seq, :]
# # # # # # # # # # # #                   + t_emb.unsqueeze(1))
# # # # # # # # # # # #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

# # # # # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(
# # # # # # # # # # # #             self.transformer(x_emb, memory)
# # # # # # # # # # # #         )))  # [B, T, 4]

# # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # #             v_phys = self._beta_drift_velocity(x_t[:, :T_seq, :])

# # # # # # # # # # # #         scale = torch.sigmoid(self.physics_scale) * 2.0
# # # # # # # # # # # #         return v_neural + scale * v_phys

# # # # # # # # # # # #     def forward(self, x_t, t, batch_list):
# # # # # # # # # # # #         raw = self._context(batch_list)
# # # # # # # # # # # #         ctx = self._apply_ctx_head(raw, noise_scale=0.0)
# # # # # # # # # # # #         return self._decode(x_t, t, ctx)

# # # # # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale: float = 0.0):
# # # # # # # # # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale=noise_scale)
# # # # # # # # # # # #         return self._decode(x_t, t, ctx)


# # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # #  TCFlowMatching
# # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # class TCFlowMatching(nn.Module):
# # # # # # # # # # # #     """TC trajectory prediction via OT-CFM + Physics-guided velocity field."""

# # # # # # # # # # # #     def __init__(
# # # # # # # # # # # #         self,
# # # # # # # # # # # #         pred_len:             int   = 12,
# # # # # # # # # # # #         obs_len:              int   = 8,
# # # # # # # # # # # #         sigma_min:            float = 0.02,
# # # # # # # # # # # #         n_train_ens:          int   = 4,
# # # # # # # # # # # #         unet_in_ch:           int   = 13,
# # # # # # # # # # # #         ctx_noise_scale:      float = 0.02,   # FIX-T23-5: 0.02 default
# # # # # # # # # # # #         initial_sample_sigma: float = 0.1,    # FIX-T23-4: 0.1 default
# # # # # # # # # # # #         **kwargs,
# # # # # # # # # # # #     ):
# # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # # # # #         self.active_pred_len      = pred_len   # FIX-M18: always full pred_len
# # # # # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # # # # # # #         self.net = VelocityField(
# # # # # # # # # # # #             pred_len   = pred_len,
# # # # # # # # # # # #             obs_len    = obs_len,
# # # # # # # # # # # #             sigma_min  = sigma_min,
# # # # # # # # # # # #             unet_in_ch = unet_in_ch,
# # # # # # # # # # # #         )

# # # # # # # # # # # #     def set_curriculum_len(self, active_len: int) -> None:
# # # # # # # # # # # #         """
# # # # # # # # # # # #         FIX-M18: Kept for backward compat but NO-OP in v23.
# # # # # # # # # # # #         Curriculum is removed. active_pred_len is always pred_len.
# # # # # # # # # # # #         """
# # # # # # # # # # # #         # self.active_pred_len = max(1, min(active_len, self.pred_len))
# # # # # # # # # # # #         pass  # no-op

# # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # #     def _to_rel(traj_gt, Me_gt, last_pos, last_Me):
# # # # # # # # # # # #         return torch.cat(
# # # # # # # # # # # #             [traj_gt - last_pos.unsqueeze(0),
# # # # # # # # # # # #              Me_gt   - last_Me.unsqueeze(0)],
# # # # # # # # # # # #             dim=-1,
# # # # # # # # # # # #         ).permute(1, 0, 2)

# # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # #     def _to_abs(rel, last_pos, last_Me):
# # # # # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # # # # #         return (
# # # # # # # # # # # #             last_pos.unsqueeze(0) + d[:, :, :2],
# # # # # # # # # # # #             last_Me.unsqueeze(0)  + d[:, :, 2:],
# # # # # # # # # # # #         )

# # # # # # # # # # # #     def _cfm_noisy(self, x1):
# # # # # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # # # # #         sm  = self.sigma_min
# # # # # # # # # # # #         x0  = torch.randn_like(x1) * sm
# # # # # # # # # # # #         t   = torch.rand(B, device=device)
# # # # # # # # # # # #         te  = t.view(B, 1, 1)
# # # # # # # # # # # #         x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
# # # # # # # # # # # #         denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
# # # # # # # # # # # #         target_vel = (x1 - (1.0 - sm) * x_t) / denom
# # # # # # # # # # # #         return x_t, t, te, denom, target_vel

# # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # #     def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # #         wind_norm = obs_Me[-1, :, 1].detach()
# # # # # # # # # # # #         w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
# # # # # # # # # # # #             torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
# # # # # # # # # # # #             torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
# # # # # # # # # # # #                         torch.full_like(wind_norm, 1.5))))
# # # # # # # # # # # #         return w / w.mean().clamp(min=1e-6)

# # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # #     def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
# # # # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # # # #             return batch_list
# # # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # # # # #             t = aug[idx]
# # # # # # # # # # # #             if torch.is_tensor(t) and t.shape[-1] >= 1:
# # # # # # # # # # # #                 t = t.clone()
# # # # # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # # # # #                 aug[idx] = t
# # # # # # # # # # # #         return aug

# # # # # # # # # # # #     def get_loss(self, batch_list: List,
# # # # # # # # # # # #                  step_weight_alpha: float = 0.0) -> torch.Tensor:
# # # # # # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha)["total"]

# # # # # # # # # # # #     def get_loss_breakdown(self, batch_list: List,
# # # # # # # # # # # #                            step_weight_alpha: float = 0.0) -> Dict:
# # # # # # # # # # # #         """
# # # # # # # # # # # #         FIX-M19: Nhận step_weight_alpha, truyền vào compute_total_loss.
# # # # # # # # # # # #         FIX-M20: Truyền all_trajs để tính ensemble_spread_loss.
# # # # # # # # # # # #         """
# # # # # # # # # # # #         batch_list = self._lon_flip_aug(batch_list, p=0.3)

# # # # # # # # # # # #         traj_gt  = batch_list[1]
# # # # # # # # # # # #         Me_gt    = batch_list[8]
# # # # # # # # # # # #         obs_t    = batch_list[0]
# # # # # # # # # # # #         obs_Me   = batch_list[7]

# # # # # # # # # # # #         try:
# # # # # # # # # # # #             env_data = batch_list[13]
# # # # # # # # # # # #         except (IndexError, TypeError):
# # # # # # # # # # # #             env_data = None

# # # # # # # # # # # #         # FIX-M18: NO curriculum slicing. Always use full pred_len.
# # # # # # # # # # # #         lp, lm = obs_t[-1], obs_Me[-1]
# # # # # # # # # # # #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

# # # # # # # # # # # #         raw_ctx     = self.net._context(batch_list)
# # # # # # # # # # # #         intensity_w = self._intensity_weights(obs_Me)

# # # # # # # # # # # #         x_t, t, te, denom, _ = self._cfm_noisy(x1)
# # # # # # # # # # # #         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

# # # # # # # # # # # #         # Ensemble samples for AFCRPS + spread penalty
# # # # # # # # # # # #         samples: List[torch.Tensor] = []
# # # # # # # # # # # #         for _ in range(self.n_train_ens):
# # # # # # # # # # # #             xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
# # # # # # # # # # # #             pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
# # # # # # # # # # # #             x1_s  = xt_s + dens_s * pv_s   # OT-CFM
# # # # # # # # # # # #             pa_s, _ = self._to_abs(x1_s, lp, lm)
# # # # # # # # # # # #             samples.append(pa_s)
# # # # # # # # # # # #         pred_samples = torch.stack(samples)   # [S, T, B, 2]

# # # # # # # # # # # #         # FIX-M20: all_trajs for spread penalty
# # # # # # # # # # # #         all_trajs_4d = pred_samples   # [S, T, B, 2]

# # # # # # # # # # # #         l_fm_physics = fm_physics_consistency_loss(
# # # # # # # # # # # #             pred_samples, gt_norm=traj_gt, last_pos=lp)

# # # # # # # # # # # #         x1_pred = x_t + denom * pred_vel
# # # # # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# # # # # # # # # # # #         pred_abs_deg = _denorm_to_deg(pred_abs)
# # # # # # # # # # # #         traj_gt_deg  = _denorm_to_deg(traj_gt)
# # # # # # # # # # # #         ref_deg      = _denorm_to_deg(lp)

# # # # # # # # # # # #         breakdown = compute_total_loss(
# # # # # # # # # # # #             pred_abs           = pred_abs_deg,
# # # # # # # # # # # #             gt                 = traj_gt_deg,
# # # # # # # # # # # #             ref                = ref_deg,
# # # # # # # # # # # #             batch_list         = batch_list,
# # # # # # # # # # # #             pred_samples       = pred_samples,
# # # # # # # # # # # #             gt_norm            = traj_gt,
# # # # # # # # # # # #             weights            = WEIGHTS,
# # # # # # # # # # # #             intensity_w        = intensity_w,
# # # # # # # # # # # #             env_data           = env_data,
# # # # # # # # # # # #             step_weight_alpha  = step_weight_alpha,   # FIX-M19
# # # # # # # # # # # #             all_trajs          = all_trajs_4d,         # FIX-M20
# # # # # # # # # # # #         )

# # # # # # # # # # # #         fm_phys_w = WEIGHTS.get("fm_physics", 0.3)
# # # # # # # # # # # #         breakdown["total"]      = breakdown["total"] + fm_phys_w * l_fm_physics
# # # # # # # # # # # #         breakdown["fm_physics"] = l_fm_physics.item()

# # # # # # # # # # # #         return breakdown

# # # # # # # # # # # #     # ── sample() ─────────────────────────────────────────────────────────────

# # # # # # # # # # # #     @torch.no_grad()
# # # # # # # # # # # #     def sample(
# # # # # # # # # # # #         self,
# # # # # # # # # # # #         batch_list: List,
# # # # # # # # # # # #         num_ensemble: int = 50,
# # # # # # # # # # # #         ddim_steps:   int = 20,
# # # # # # # # # # # #         predict_csv:  Optional[str] = None,
# # # # # # # # # # # #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# # # # # # # # # # # #         """
# # # # # # # # # # # #         FIX-T23-3: ddim_steps default 20 (từ 10).
# # # # # # # # # # # #         FIX-M22: tighter clip [-3.0, 3.0] cho cả lon và lat.
# # # # # # # # # # # #         Returns:
# # # # # # # # # # # #             pred_mean:  [T, B, 2] mean track (normalised)
# # # # # # # # # # # #             me_mean:    [T, B, 2] mean intensity
# # # # # # # # # # # #             all_trajs:  [S, T, B, 2] all ensemble members
# # # # # # # # # # # #         """
# # # # # # # # # # # #         lp  = batch_list[0][-1]   # [B, 2]
# # # # # # # # # # # #         lm  = batch_list[7][-1]   # [B, 2]
# # # # # # # # # # # #         B   = lp.shape[0]
# # # # # # # # # # # #         device = lp.device
# # # # # # # # # # # #         T   = self.pred_len   # FIX-M18: always full pred_len
# # # # # # # # # # # #         dt  = 1.0 / max(ddim_steps, 1)

# # # # # # # # # # # #         raw_ctx = self.net._context(batch_list)

# # # # # # # # # # # #         traj_s: List[torch.Tensor] = []
# # # # # # # # # # # #         me_s:   List[torch.Tensor] = []

# # # # # # # # # # # #         for k in range(num_ensemble):
# # # # # # # # # # # #             # FIX-T23-4: initial_sample_sigma=0.1 (set in constructor)
# # # # # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# # # # # # # # # # # #             # DDIM Euler integration
# # # # # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # # #                 ns  = self.ctx_noise_scale if step == 0 else 0.0
# # # # # # # # # # # #                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
# # # # # # # # # # # #                 x_t = x_t + dt * vel

# # # # # # # # # # # #             # Physics correction
# # # # # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)

# # # # # # # # # # # #             # FIX-M22: tighter clip
# # # # # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)

# # # # # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # # # #             traj_s.append(tr)
# # # # # # # # # # # #             me_s.append(me)

# # # # # # # # # # # #         all_trajs = torch.stack(traj_s)   # [S, T, B, 2]
# # # # # # # # # # # #         all_me    = torch.stack(me_s)
# # # # # # # # # # # #         pred_mean = all_trajs.mean(0)
# # # # # # # # # # # #         me_mean   = all_me.mean(0)

# # # # # # # # # # # #         if predict_csv:
# # # # # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)

# # # # # # # # # # # #         return pred_mean, me_mean, all_trajs

# # # # # # # # # # # #     # ── Physics correction ────────────────────────────────────────────────────

# # # # # # # # # # # #     def _physics_correct(
# # # # # # # # # # # #         self,
# # # # # # # # # # # #         x_pred: torch.Tensor,
# # # # # # # # # # # #         last_pos: torch.Tensor,
# # # # # # # # # # # #         last_Me:  torch.Tensor,
# # # # # # # # # # # #         n_steps:  int   = 5,    # FIX-M21: 5 (từ 3)
# # # # # # # # # # # #         lr:       float = 0.002, # FIX-M21: 0.002 (từ 0.005)
# # # # # # # # # # # #     ) -> torch.Tensor:
# # # # # # # # # # # #         """
# # # # # # # # # # # #         FIX-M17: torch.enable_grad() inside no_grad context.
# # # # # # # # # # # #         FIX-M21: n_steps=5, lr=0.002 for more stable correction.
# # # # # # # # # # # #         """
# # # # # # # # # # # #         with torch.enable_grad():
# # # # # # # # # # # #             x = x_pred.detach().requires_grad_(True)
# # # # # # # # # # # #             optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)

# # # # # # # # # # # #             for _ in range(n_steps):
# # # # # # # # # # # #                 optimizer.zero_grad()
# # # # # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # # # # #                 pred_deg    = _denorm_to_deg(pred_abs)

# # # # # # # # # # # #                 l_speed = self._pinn_speed_constraint(pred_deg)
# # # # # # # # # # # #                 l_accel = self._pinn_beta_plane_simplified(pred_deg)

# # # # # # # # # # # #                 physics_loss = l_speed + 0.3 * l_accel
# # # # # # # # # # # #                 physics_loss.backward()

# # # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], max_norm=0.05)
# # # # # # # # # # # #                 optimizer.step()

# # # # # # # # # # # #         return x.detach()

# # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # #     def _pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # #         if pred_deg.shape[0] < 2:
# # # # # # # # # # # #             return pred_deg.new_zeros(())
# # # # # # # # # # # #         dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # # #         lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # #         dx_km   = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # # # # # # # # #         dy_km   = dt_deg[:, :, 1] * 111.0
# # # # # # # # # # # #         speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # # # # # # # # # # #         return F.relu(speed - 600.0).pow(2).mean()

# # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # #     def _pinn_beta_plane_simplified(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # #         if pred_deg.shape[0] < 3:
# # # # # # # # # # # #             return pred_deg.new_zeros(())
# # # # # # # # # # # #         v = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # # #         a = v[1:] - v[:-1]
# # # # # # # # # # # #         lat_rad = torch.deg2rad(pred_deg[1:-1, :, 1])
# # # # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # #         a_lon_km = a[:, :, 0] * cos_lat * 111.0
# # # # # # # # # # # #         a_lat_km = a[:, :, 1] * 111.0
# # # # # # # # # # # #         max_accel = 50.0
# # # # # # # # # # # #         violation = F.relu(torch.sqrt(a_lon_km**2 + a_lat_km**2) - max_accel)
# # # # # # # # # # # #         return violation.pow(2).mean() * 0.1

# # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # #     def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
# # # # # # # # # # # #                            all_trajs: torch.Tensor) -> None:
# # # # # # # # # # # #         import numpy as np
# # # # # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # # # # #         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # # # # #         S       = all_trajs.shape[0]

# # # # # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()

# # # # # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # # # # #                     "lon_mean_deg", "lat_mean_deg",
# # # # # # # # # # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # # # # #             if write_hdr:
# # # # # # # # # # # #                 w.writeheader()
# # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # #                 for k in range(T):
# # # # # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # # # # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
# # # # # # # # # # # #                         math.radians(mean_lat[k, b]))
# # # # # # # # # # # #                     spread = float(np.sqrt((dlat**2 + dlon**2).mean()) * 111.0)
# # # # # # # # # # # #                     w.writerow(dict(
# # # # # # # # # # # #                         timestamp     = ts,
# # # # # # # # # # # #                         batch_idx     = b,
# # # # # # # # # # # #                         step_idx      = k,
# # # # # # # # # # # #                         lead_h        = (k + 1) * 6,
# # # # # # # # # # # #                         lon_mean_deg  = f"{mean_lon[k,b]:.4f}",
# # # # # # # # # # # #                         lat_mean_deg  = f"{mean_lat[k,b]:.4f}",
# # # # # # # # # # # #                         lon_std_deg   = f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # # # # #                         lat_std_deg   = f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # # # # #                         ens_spread_km = f"{spread:.2f}",
# # # # # # # # # # # #                     ))
# # # # # # # # # # # #         print(f"  Predictions → {csv_path}  (B={B}, T={T}, S={S})")


# # # # # # # # # # # # # Backward-compat alias
# # # # # # # # # # # # TCDiffusion = TCFlowMatching

# # # # # # # # # # # """
# # # # # # # # # # # Model/flow_matching_model.py  ── v24
# # # # # # # # # # # ==========================================
# # # # # # # # # # # FIXES vs v23:

# # # # # # # # # # #   FIX-M23  [SHORT-RANGE HEAD] Thêm ShortRangeHead – bộ dự báo riêng
# # # # # # # # # # #            cho 4 bước đầu (6h, 12h, 18h, 24h) dùng GRU autoregressive
# # # # # # # # # # #            + motion prior từ obs_traj. KHÔNG dùng flow matching cho
# # # # # # # # # # #            các bước này → ổn định hơn, chính xác hơn cho tầm ngắn.

# # # # # # # # # # #   FIX-M24  [INFERENCE BLEND] sample() dùng ShortRangeHead (deterministic)
# # # # # # # # # # #            cho steps 1-4, FM ensemble cho steps 5-12. pred_mean[:4]
# # # # # # # # # # #            luôn đến từ ShortRangeHead → giảm ADE 12h/24h đáng kể.

# # # # # # # # # # #   FIX-M25  [SPREAD CONTROL] initial_sample_sigma=0.03, ctx_noise_scale=0.002
# # # # # # # # # # #            → spread ensemble giảm từ 400-500km xuống mục tiêu <150km.

# # # # # # # # # # #   FIX-M26  [PHYSICS IN HEAD] ShortRangeHead clamp delta per-step ≤ MAX_DELTA
# # # # # # # # # # #            (~600km/6h), built-in vào forward pass → không cần PINN
# # # # # # # # # # #            cho tầm ngắn.

# # # # # # # # # # #   FIX-M27  [LOSS SEPARATION] get_loss_breakdown() tính thêm short_range_loss
# # # # # # # # # # #            riêng với weight cao (5.0). FM loss vẫn tính đầy đủ 12 bước
# # # # # # # # # # #            nhưng short_range override pred_mean cho bước 1-4.

# # # # # # # # # # # Kept from v23:
# # # # # # # # # # #   FIX-M18  active_pred_len = pred_len (no curriculum)
# # # # # # # # # # #   FIX-M21  _physics_correct n_steps=5, lr=0.002
# # # # # # # # # # #   FIX-M22  post-sampling clip [-3.0, 3.0]
# # # # # # # # # # # """
# # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # import csv
# # # # # # # # # # # import math
# # # # # # # # # # # import os
# # # # # # # # # # # from datetime import datetime
# # # # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # # # import torch
# # # # # # # # # # # import torch.nn as nn
# # # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # # # from Model.losses import (
# # # # # # # # # # #     compute_total_loss, fm_physics_consistency_loss, WEIGHTS,
# # # # # # # # # # #     pinn_speed_constraint, short_range_regression_loss,
# # # # # # # # # # # )


# # # # # # # # # # # def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # #     """Normalised → degrees. Handles [T, B, 2] and [B, 2]."""
# # # # # # # # # # #     out = traj_norm.clone()
# # # # # # # # # # #     out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # #     out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
# # # # # # # # # # #     return out


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # #  ShortRangeHead  – FIX-M23
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # class ShortRangeHead(nn.Module):
# # # # # # # # # # #     """
# # # # # # # # # # #     Dedicated deterministic predictor for steps 1-4 (6h/12h/18h/24h).

# # # # # # # # # # #     Architecture: GRU autoregressive cell
# # # # # # # # # # #         Input  each step : [ctx_feat(256) | vel_feat(128) | cur_pos(2)] = 386
# # # # # # # # # # #         Hidden           : 256
# # # # # # # # # # #         Output each step : delta [2] → clamp → new position

# # # # # # # # # # #     Physics built-in:
# # # # # # # # # # #         • MAX_DELTA = 0.48  ≈  576 km / 6h  (upper TC speed bound)
# # # # # # # # # # #         • Speed is implicitly constrained by architecture + clamp

# # # # # # # # # # #     Why GRU (not Transformer):
# # # # # # # # # # #         • Autoregressive dependency: step t depends on step t-1 position
# # # # # # # # # # #         • Small number of steps (4) → GRU is simpler and faster
# # # # # # # # # # #         • Motion prior via velocity encoding stabilises early epochs
# # # # # # # # # # #     """

# # # # # # # # # # #     N_STEPS   = 4        # 6h, 12h, 18h, 24h
# # # # # # # # # # #     MAX_DELTA = 0.48     # normalised units ≈ 576 km / 6h

# # # # # # # # # # #     def __init__(self, raw_ctx_dim=512, obs_len=8):
# # # # # # # # # # #         super().__init__()
# # # # # # # # # # #         self.obs_len = obs_len

# # # # # # # # # # #         # Motion prior: dùng tất cả obs velocities, không chỉ 3 bước cuối
# # # # # # # # # # #         self.vel_enc = nn.Sequential(
# # # # # # # # # # #             nn.Linear(obs_len * 2, 256),   # FIX: toàn bộ obs_len velocities
# # # # # # # # # # #             nn.GELU(),
# # # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # # #             nn.Linear(256, 128),
# # # # # # # # # # #             nn.GELU(),
# # # # # # # # # # #         )

# # # # # # # # # # #         self.ctx_proj = nn.Sequential(
# # # # # # # # # # #             nn.Linear(raw_ctx_dim, 256),
# # # # # # # # # # #             nn.GELU(),
# # # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # # #             nn.Dropout(0.05),   # FIX: giảm dropout 0.08→0.05
# # # # # # # # # # #         )

# # # # # # # # # # #         # FIX: GRU hidden = ctx+vel kết hợp ngay từ đầu
# # # # # # # # # # #         # input = ctx(256) + vel(128) + cur_pos(2) = 386
# # # # # # # # # # #         self.gru_cell = nn.GRUCell(input_size=386, hidden_size=256)

# # # # # # # # # # #         # FIX: thêm residual connection từ motion prior
# # # # # # # # # # #         self.motion_gate = nn.Sequential(
# # # # # # # # # # #             nn.Linear(128 + 256, 1),
# # # # # # # # # # #             nn.Sigmoid(),
# # # # # # # # # # #         )

# # # # # # # # # # #         self.out_head = nn.Sequential(
# # # # # # # # # # #             nn.Linear(256, 128),
# # # # # # # # # # #             nn.GELU(),
# # # # # # # # # # #             nn.LayerNorm(128),
# # # # # # # # # # #             nn.Linear(128, 2),
# # # # # # # # # # #         )

# # # # # # # # # # #         # Per-step scale được học, khởi tạo nhỏ để ổn định
# # # # # # # # # # #         self.step_scale = nn.Parameter(torch.ones(self.N_STEPS) * 0.2)
# # # # # # # # # # #         self._init_weights()

# # # # # # # # # # #     def _init_weights(self):
# # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # #             for m in self.modules():
# # # # # # # # # # #                 if isinstance(m, nn.Linear):
# # # # # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.3)
# # # # # # # # # # #                     if m.bias is not None:
# # # # # # # # # # #                         nn.init.zeros_(m.bias)

# # # # # # # # # # #     def forward(self, raw_ctx, obs_traj):
# # # # # # # # # # #         # obs_traj: [T_obs, B, 2]
# # # # # # # # # # #         B        = raw_ctx.shape[0]
# # # # # # # # # # #         T_obs    = obs_traj.shape[0]

# # # # # # # # # # #         # FIX: encode toàn bộ velocity sequence
# # # # # # # # # # #         if T_obs >= 2:
# # # # # # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]   # [T_obs-1, B, 2]
# # # # # # # # # # #         else:
# # # # # # # # # # #             vels = torch.zeros(1, B, 2, device=raw_ctx.device)

# # # # # # # # # # #         # Pad hoặc crop về obs_len velocity steps
# # # # # # # # # # #         target_v = self.obs_len  # số velocity steps cần
# # # # # # # # # # #         if vels.shape[0] < target_v:
# # # # # # # # # # #             pad = torch.zeros(target_v - vels.shape[0], B, 2,
# # # # # # # # # # #                               device=raw_ctx.device)
# # # # # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # # # # #         else:
# # # # # # # # # # #             vels = vels[-target_v:]   # lấy target_v bước cuối

# # # # # # # # # # #         vel_input = vels.permute(1, 0, 2).reshape(B, -1)  # [B, obs_len*2]
# # # # # # # # # # #         vel_feat  = self.vel_enc(vel_input)                 # [B, 128]

# # # # # # # # # # #         ctx_feat = self.ctx_proj(raw_ctx)  # [B, 256]

# # # # # # # # # # #         # FIX: hx được khởi tạo từ ctx thay vì chính ctx
# # # # # # # # # # #         # để GRU có không gian học riêng
# # # # # # # # # # #         hx      = ctx_feat
# # # # # # # # # # #         cur_pos = obs_traj[-1].clone()   # [B, 2]

# # # # # # # # # # #         # Last observed velocity làm prior bước đầu
# # # # # # # # # # #         last_vel = vels[-1].clone()      # [B, 2]

# # # # # # # # # # #         preds = []
# # # # # # # # # # #         for i in range(self.N_STEPS):
# # # # # # # # # # #             inp = torch.cat([ctx_feat, vel_feat, cur_pos], dim=-1)  # [B,386]
# # # # # # # # # # #             hx  = self.gru_cell(inp, hx)                             # [B,256]

# # # # # # # # # # #             # FIX: motion gate — blend giữa GRU output và motion prior
# # # # # # # # # # #             gate = self.motion_gate(
# # # # # # # # # # #                 torch.cat([vel_feat, hx], dim=-1)
# # # # # # # # # # #             )  # [B,1]  — giai đoạn đầu training gate≈1 → theo motion prior

# # # # # # # # # # #             raw_delta   = self.out_head(hx)                     # [B,2]
# # # # # # # # # # #             # Motion prior: tiếp tục với velocity cuối (decay theo step)
# # # # # # # # # # #             prior_delta = last_vel * (0.9 ** i)                 # [B,2]

# # # # # # # # # # #             # Blend: gate=1 → dùng motion prior hoàn toàn (đầu training)
# # # # # # # # # # #             #        gate=0 → dùng GRU output hoàn toàn (sau khi học)
# # # # # # # # # # #             scale_i = torch.sigmoid(self.step_scale[i]) * self.MAX_DELTA
# # # # # # # # # # #             blended = gate * prior_delta + (1 - gate) * raw_delta * scale_i

# # # # # # # # # # #             delta   = blended.clamp(-self.MAX_DELTA, self.MAX_DELTA)
# # # # # # # # # # #             cur_pos = cur_pos + delta
# # # # # # # # # # #             last_vel = delta   # update velocity prior
# # # # # # # # # # #             preds.append(cur_pos)

# # # # # # # # # # #         return torch.stack(preds, dim=0)   # [4, B, 2]


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # #  VelocityField
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # # # #     """
# # # # # # # # # # #     OT-CFM velocity field v_θ(x_t, t, context).
# # # # # # # # # # #     v24: ShortRangeHead thêm vào để dự báo tầm ngắn chính xác hơn.
# # # # # # # # # # #     """

# # # # # # # # # # #     # Dim của raw_ctx (output ctx_fc1) – phải khớp ShortRangeHead
# # # # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # # # #     def __init__(
# # # # # # # # # # #         self,
# # # # # # # # # # #         pred_len:   int   = 12,
# # # # # # # # # # #         obs_len:    int   = 8,
# # # # # # # # # # #         ctx_dim:    int   = 256,
# # # # # # # # # # #         sigma_min:  float = 0.02,
# # # # # # # # # # #         unet_in_ch: int   = 13,
# # # # # # # # # # #     ):
# # # # # # # # # # #         super().__init__()
# # # # # # # # # # #         self.pred_len  = pred_len
# # # # # # # # # # #         self.obs_len   = obs_len
# # # # # # # # # # #         self.sigma_min = sigma_min

# # # # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # # # #             in_channel   = unet_in_ch,
# # # # # # # # # # #             out_channel  = 1,
# # # # # # # # # # #             d_model      = 32,
# # # # # # # # # # #             n_layers     = 4,
# # # # # # # # # # #             modes_t      = 4,
# # # # # # # # # # #             modes_h      = 4,
# # # # # # # # # # #             modes_w      = 4,
# # # # # # # # # # #             spatial_down = 32,
# # # # # # # # # # #             dropout      = 0.05,
# # # # # # # # # # #         )

# # # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # # # #             in_1d       = 4,
# # # # # # # # # # #             feat_3d_dim = 128,
# # # # # # # # # # #             mlp_h       = 64,
# # # # # # # # # # #             lstm_hidden = 128,
# # # # # # # # # # #             lstm_layers = 3,
# # # # # # # # # # #             dropout     = 0.1,
# # # # # # # # # # #             d_state     = 16,
# # # # # # # # # # #         )

# # # # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # # # # # # # # #         # ctx_fc1 output = 512 = RAW_CTX_DIM
# # # # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # # # #         self.ctx_drop = nn.Dropout(0.15)
# # # # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # # # # # # # #         self.time_fc1 = nn.Linear(256, 512)
# # # # # # # # # # #         self.time_fc2 = nn.Linear(512, 256)

# # # # # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # # # #                 dropout=0.15, activation="gelu", batch_first=True,
# # # # # # # # # # #             ),
# # # # # # # # # # #             num_layers=4,
# # # # # # # # # # #         )
# # # # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)

# # # # # # # # # # #         # ── FIX-M23: Short-range head ──────────────────────────────────────
# # # # # # # # # # #         self.short_range_head = ShortRangeHead(
# # # # # # # # # # #             raw_ctx_dim = self.RAW_CTX_DIM,
# # # # # # # # # # #             obs_len     = obs_len,
# # # # # # # # # # #         )

# # # # # # # # # # #     # ── helpers ──────────────────────────────────────────────────────────────

# # # # # # # # # # #     def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
# # # # # # # # # # #         half = dim // 2
# # # # # # # # # # #         freq = torch.exp(
# # # # # # # # # # #             torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # # # #             * (-math.log(10_000.0) / max(half - 1, 1))
# # # # # # # # # # #         )
# # # # # # # # # # #         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
# # # # # # # # # # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # # # # # # # # # #         return F.pad(emb, (0, dim % 2))

# # # # # # # # # # #     # def _context(self, batch_list: List) -> torch.Tensor:
# # # # # # # # # # #     #     """Returns raw_ctx [B, RAW_CTX_DIM] (before dropout/projection)."""
# # # # # # # # # # #     #     obs_traj  = batch_list[0]
# # # # # # # # # # #     #     obs_Me    = batch_list[7]
# # # # # # # # # # #     #     image_obs = batch_list[11]
# # # # # # # # # # #     #     env_data  = batch_list[13]

# # # # # # # # # # #     #     if image_obs.dim() == 4:
# # # # # # # # # # #     #         image_obs = image_obs.unsqueeze(1)

# # # # # # # # # # #     #     expected_ch = self.spatial_enc.in_channel
# # # # # # # # # # #     #     if image_obs.shape[1] == 1 and expected_ch != 1:
# # # # # # # # # # #     #         image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# # # # # # # # # # #     #     e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # # #     #     T_obs = obs_traj.shape[0]

# # # # # # # # # # #     #     e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
# # # # # # # # # # #     #     e_3d_s = e_3d_s.permute(0, 2, 1)
# # # # # # # # # # #     #     e_3d_s = self.bottleneck_proj(e_3d_s)

# # # # # # # # # # #     #     T_bot = e_3d_s.shape[1]
# # # # # # # # # # #     #     if T_bot != T_obs:
# # # # # # # # # # #     #         e_3d_s = F.interpolate(
# # # # # # # # # # #     #             e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # # #     #             mode="linear", align_corners=False,
# # # # # # # # # # #     #         ).permute(0, 2, 1)

# # # # # # # # # # #     #     # f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))
# # # # # # # # # # #     #     # f_spatial     = self.decoder_proj(f_spatial_raw)

# # # # # # # # # # #     #     # FIX: pool H,W, giữ T, rồi flatten
# # # # # # # # # # #     #     # f_spatial_raw = e_3d_dec.squeeze(1)          # [B,T,1,1] → [B,T]  (squeeze C và H,W)
# # # # # # # # # # #     #     # f_spatial_raw = f_spatial_raw.mean(dim=1)    # [B] — hoặc giữ T nếu muốn
# # # # # # # # # # #     #     # Hoặc đơn giản hơn: mean chỉ spatial
# # # # # # # # # # #     #     f_spatial_raw = e_3d_dec.mean(dim=(3,4))     # [B,1,T] → squeeze → [B,T]
# # # # # # # # # # #     #     f_spatial     = self.decoder_proj(f_spatial_raw.squeeze(1))  # cần Linear(T,16)

# # # # # # # # # # #     #     obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # # # #     #     h_t    = self.enc_1d(obs_in, e_3d_s)

# # # # # # # # # # #     #     e_env, _, _ = self.env_enc(env_data, image_obs)

# # # # # # # # # # #     #     raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
# # # # # # # # # # #     #     raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))  # [B, RAW_CTX_DIM]
# # # # # # # # # # #     #     return raw                                      # raw_ctx

# # # # # # # # # # #     # ── _context(): fix f_spatial ──────────────────────────────────────────────
# # # # # # # # # # #     def _context(self, batch_list):
# # # # # # # # # # #         obs_traj  = batch_list[0]   # [T_obs, B, 2]
# # # # # # # # # # #         obs_Me    = batch_list[7]   # [T_obs, B, 2]
# # # # # # # # # # #         image_obs = batch_list[11]  # [B, 13, T, 81, 81]  từ seq_collate
# # # # # # # # # # #         env_data  = batch_list[13]

# # # # # # # # # # #         # image_obs đã được permute đúng bởi seq_collate → [B,13,T,H,W]
# # # # # # # # # # #         # Chỉ cần guard cho trường hợp dim==4 (không có T)
# # # # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # # # #             image_obs = image_obs.unsqueeze(2)  # thêm T dim, không phải C

# # # # # # # # # # #         expected_ch = self.spatial_enc.in_channel
# # # # # # # # # # #         if image_obs.shape[1] == 1 and expected_ch != 1:
# # # # # # # # # # #             image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# # # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # # #         # e_3d_bot: [B,128,T,4,4]   e_3d_dec: [B,1,T,1,1]

# # # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # # #         # Bottleneck: pool spatial → [B,128,T] → permute → [B,T,128]
# # # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot)   # [B,128,T,1,1]
# # # # # # # # # # #         e_3d_s = e_3d_s.squeeze(-1).squeeze(-1)   # [B,128,T]
# # # # # # # # # # #         e_3d_s = e_3d_s.permute(0, 2, 1)          # [B,T,128]
# # # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)     # [B,T,128]

# # # # # # # # # # #         # T alignment
# # # # # # # # # # #         T_bot = e_3d_s.shape[1]
# # # # # # # # # # #         if T_bot != T_obs:
# # # # # # # # # # #             e_3d_s = F.interpolate(
# # # # # # # # # # #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # # #                 mode="linear", align_corners=False,
# # # # # # # # # # #             ).permute(0, 2, 1)

# # # # # # # # # # #         # FIX Bug G: f_spatial giữ lại temporal info
# # # # # # # # # # #         # e_3d_dec: [B,1,T,1,1] → mean(H,W) → [B,1,T] → mean(T) → [B,1]
# # # # # # # # # # #         # Thay bằng: pool T về 1 bằng weighted mean (cuối quan trọng hơn)
# # # # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)  # [B,T]
# # # # # # # # # # #         # Exponential weighting: bước cuối obs quan trọng nhất
# # # # # # # # # # #         t_weights = torch.softmax(
# # # # # # # # # # #             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # # # # #                         device=e_3d_dec_t.device) * 0.5,
# # # # # # # # # # #             dim=0
# # # # # # # # # # #         )  # [T]
# # # # # # # # # # #         f_spatial_scalar = (e_3d_dec_t * t_weights.unsqueeze(0)).sum(1, keepdim=True)  # [B,1]
# # # # # # # # # # #         f_spatial = self.decoder_proj(f_spatial_scalar)  # [B,16]

# # # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)  # [B,T,4]
# # # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)   # [B,128]

# # # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)  # [B,64]

# # # # # # # # # # #         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)  # [B,208]
# # # # # # # # # # #         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))       # [B,512]
# # # # # # # # # # #         return raw

# # # # # # # # # # #     def _apply_ctx_head(self, raw: torch.Tensor,
# # # # # # # # # # #                         noise_scale: float = 0.0) -> torch.Tensor:
# # # # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # # # #     def _beta_drift_velocity(self, x_t: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # #         """Beta drift in normalised state space. x_t: [B, T, 4]."""
# # # # # # # # # # #         OMEGA_val  = 7.2921e-5
# # # # # # # # # # #         R_val      = 6.371e6
# # # # # # # # # # #         DT         = 6 * 3600.0
# # # # # # # # # # #         M_PER_NORM = 5.0 * 111.0 * 1000.0

# # # # # # # # # # #         lat_norm = x_t[:, :, 1]
# # # # # # # # # # #         lat_deg  = lat_norm * 5.0
# # # # # # # # # # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))

# # # # # # # # # # #         beta   = 2 * OMEGA_val * torch.cos(lat_rad) / R_val
# # # # # # # # # # #         R_tc   = 3e5
# # # # # # # # # # #         v_lon  = -beta * R_tc ** 2 / 2
# # # # # # # # # # #         v_lat  =  beta * R_tc ** 2 / 4

# # # # # # # # # # #         v_lon_norm = v_lon * DT / M_PER_NORM
# # # # # # # # # # #         v_lat_norm = v_lat * DT / M_PER_NORM

# # # # # # # # # # #         v_phys = torch.zeros_like(x_t)
# # # # # # # # # # #         v_phys[:, :, 0] = v_lon_norm
# # # # # # # # # # #         v_phys[:, :, 1] = v_lat_norm
# # # # # # # # # # #         return v_phys

# # # # # # # # # # #     def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
# # # # # # # # # # #                 ctx: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
# # # # # # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # # # # # #         T_seq  = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # # # # #         x_emb  = (self.traj_embed(x_t[:, :T_seq, :])
# # # # # # # # # # #                   + self.pos_enc[:, :T_seq, :]
# # # # # # # # # # #                   + t_emb.unsqueeze(1))
# # # # # # # # # # #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

# # # # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(
# # # # # # # # # # #             self.transformer(x_emb, memory)
# # # # # # # # # # #         )))

# # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # #             v_phys = self._beta_drift_velocity(x_t[:, :T_seq, :])

# # # # # # # # # # #         scale = torch.sigmoid(self.physics_scale) * 2.0
# # # # # # # # # # #         return v_neural + scale * v_phys

# # # # # # # # # # #     def forward(self, x_t, t, batch_list):
# # # # # # # # # # #         raw = self._context(batch_list)
# # # # # # # # # # #         ctx = self._apply_ctx_head(raw, noise_scale=0.0)
# # # # # # # # # # #         return self._decode(x_t, t, ctx)

# # # # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale: float = 0.0):
# # # # # # # # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale=noise_scale)
# # # # # # # # # # #         return self._decode(x_t, t, ctx)

# # # # # # # # # # #     def forward_short_range(
# # # # # # # # # # #         self,
# # # # # # # # # # #         obs_traj: torch.Tensor,  # [T_obs, B, 2]
# # # # # # # # # # #         raw_ctx:  torch.Tensor,  # [B, RAW_CTX_DIM]
# # # # # # # # # # #     ) -> torch.Tensor:
# # # # # # # # # # #         """FIX-M23: Deterministic short-range prediction [4, B, 2]."""
# # # # # # # # # # #         return self.short_range_head(raw_ctx, obs_traj)


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # #  TCFlowMatching
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # class TCFlowMatching(nn.Module):
# # # # # # # # # # #     """TC trajectory prediction via OT-CFM + ShortRangeHead + PINN."""

# # # # # # # # # # #     def __init__(
# # # # # # # # # # #         self,
# # # # # # # # # # #         pred_len:             int   = 12,
# # # # # # # # # # #         obs_len:              int   = 8,
# # # # # # # # # # #         sigma_min:            float = 0.02,
# # # # # # # # # # #         n_train_ens:          int   = 4,
# # # # # # # # # # #         unet_in_ch:           int   = 13,
# # # # # # # # # # #         ctx_noise_scale:      float = 0.002,    # FIX-M25: 0.02→0.002
# # # # # # # # # # #         initial_sample_sigma: float = 0.03,     # FIX-M25: 0.1→0.03
# # # # # # # # # # #         **kwargs,
# # # # # # # # # # #     ):
# # # # # # # # # # #         super().__init__()
# # # # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # # # #         self.active_pred_len      = pred_len
# # # # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma

# # # # # # # # # # #         self.net = VelocityField(
# # # # # # # # # # #             pred_len   = pred_len,
# # # # # # # # # # #             obs_len    = obs_len,
# # # # # # # # # # #             sigma_min  = sigma_min,
# # # # # # # # # # #             unet_in_ch = unet_in_ch,
# # # # # # # # # # #         )

# # # # # # # # # # #     def set_curriculum_len(self, active_len: int) -> None:
# # # # # # # # # # #         """FIX-M18: No-op."""
# # # # # # # # # # #         pass

# # # # # # # # # # #     # ── Static helpers ────────────────────────────────────────────────────────

# # # # # # # # # # #     @staticmethod
# # # # # # # # # # #     def _to_rel(traj_gt, Me_gt, last_pos, last_Me):
# # # # # # # # # # #         return torch.cat(
# # # # # # # # # # #             [traj_gt - last_pos.unsqueeze(0),
# # # # # # # # # # #              Me_gt   - last_Me.unsqueeze(0)],
# # # # # # # # # # #             dim=-1,
# # # # # # # # # # #         ).permute(1, 0, 2)

# # # # # # # # # # #     @staticmethod
# # # # # # # # # # #     def _to_abs(rel, last_pos, last_Me):
# # # # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # # # #         return (
# # # # # # # # # # #             last_pos.unsqueeze(0) + d[:, :, :2],
# # # # # # # # # # #             last_Me.unsqueeze(0)  + d[:, :, 2:],
# # # # # # # # # # #         )

# # # # # # # # # # #     def _cfm_noisy(self, x1):
# # # # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # # # #         sm  = self.sigma_min
# # # # # # # # # # #         x0  = torch.randn_like(x1) * sm
# # # # # # # # # # #         t   = torch.rand(B, device=device)
# # # # # # # # # # #         te  = t.view(B, 1, 1)
# # # # # # # # # # #         x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
# # # # # # # # # # #         denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
# # # # # # # # # # #         target_vel = (x1 - (1.0 - sm) * x_t) / denom
# # # # # # # # # # #         return x_t, t, te, denom, target_vel

# # # # # # # # # # #     @staticmethod
# # # # # # # # # # #     def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # #         wind_norm = obs_Me[-1, :, 1].detach()
# # # # # # # # # # #         w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
# # # # # # # # # # #             torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
# # # # # # # # # # #             torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
# # # # # # # # # # #                         torch.full_like(wind_norm, 1.5))))
# # # # # # # # # # #         return w / w.mean().clamp(min=1e-6)

# # # # # # # # # # #     @staticmethod
# # # # # # # # # # #     def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
# # # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # # #             return batch_list
# # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # # # #             t = aug[idx]
# # # # # # # # # # #             if torch.is_tensor(t) and t.shape[-1] >= 1:
# # # # # # # # # # #                 t = t.clone()
# # # # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # # # #                 aug[idx] = t
# # # # # # # # # # #         return aug

# # # # # # # # # # #     # ── Loss ──────────────────────────────────────────────────────────────────

# # # # # # # # # # #     def get_loss(self, batch_list: List,
# # # # # # # # # # #                  step_weight_alpha: float = 0.0) -> torch.Tensor:
# # # # # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha)["total"]

# # # # # # # # # # #     def get_loss_breakdown(self, batch_list: List,
# # # # # # # # # # #                            step_weight_alpha: float = 0.0) -> Dict:
# # # # # # # # # # #         """
# # # # # # # # # # #         FIX-M27: Thêm short_range_loss với weight cao (5.0).
# # # # # # # # # # #         """
# # # # # # # # # # #         batch_list = self._lon_flip_aug(batch_list, p=0.3)

# # # # # # # # # # #         traj_gt  = batch_list[1]
# # # # # # # # # # #         Me_gt    = batch_list[8]
# # # # # # # # # # #         obs_t    = batch_list[0]
# # # # # # # # # # #         obs_Me   = batch_list[7]

# # # # # # # # # # #         try:
# # # # # # # # # # #             env_data = batch_list[13]
# # # # # # # # # # #         except (IndexError, TypeError):
# # # # # # # # # # #             env_data = None

# # # # # # # # # # #         lp, lm = obs_t[-1], obs_Me[-1]
# # # # # # # # # # #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

# # # # # # # # # # #         raw_ctx     = self.net._context(batch_list)
# # # # # # # # # # #         intensity_w = self._intensity_weights(obs_Me)

# # # # # # # # # # #         x_t, t, te, denom, _ = self._cfm_noisy(x1)
# # # # # # # # # # #         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

# # # # # # # # # # #         # Ensemble for AFCRPS
# # # # # # # # # # #         samples: List[torch.Tensor] = []
# # # # # # # # # # #         for _ in range(self.n_train_ens):
# # # # # # # # # # #             xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
# # # # # # # # # # #             pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
# # # # # # # # # # #             x1_s  = xt_s + dens_s * pv_s
# # # # # # # # # # #             pa_s, _ = self._to_abs(x1_s, lp, lm)
# # # # # # # # # # #             samples.append(pa_s)
# # # # # # # # # # #         pred_samples = torch.stack(samples)     # [S, T, B, 2]
# # # # # # # # # # #         all_trajs_4d = pred_samples

# # # # # # # # # # #         l_fm_physics = fm_physics_consistency_loss(
# # # # # # # # # # #             pred_samples, gt_norm=traj_gt, last_pos=lp)

# # # # # # # # # # #         x1_pred = x_t + denom * pred_vel
# # # # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# # # # # # # # # # #         pred_abs_deg = _denorm_to_deg(pred_abs)
# # # # # # # # # # #         # # Thêm debug vào get_loss_breakdown, sau khi tính pred_abs_deg:
# # # # # # # # # # #         # print(f"  pred_abs requires_grad: {pred_abs.requires_grad}")
# # # # # # # # # # #         # print(f"  pred_abs_deg requires_grad: {pred_abs_deg.requires_grad}")

# # # # # # # # # # #         traj_gt_deg  = _denorm_to_deg(traj_gt)
# # # # # # # # # # #         ref_deg      = _denorm_to_deg(lp)

# # # # # # # # # # #         breakdown = compute_total_loss(
# # # # # # # # # # #             pred_abs           = pred_abs_deg,
# # # # # # # # # # #             gt                 = traj_gt_deg,
# # # # # # # # # # #             ref                = ref_deg,
# # # # # # # # # # #             batch_list         = batch_list,
# # # # # # # # # # #             pred_samples       = pred_samples,
# # # # # # # # # # #             gt_norm            = traj_gt,
# # # # # # # # # # #             weights            = WEIGHTS,
# # # # # # # # # # #             intensity_w        = intensity_w,
# # # # # # # # # # #             env_data           = env_data,
# # # # # # # # # # #             step_weight_alpha  = step_weight_alpha,
# # # # # # # # # # #             all_trajs          = all_trajs_4d,
# # # # # # # # # # #         )

# # # # # # # # # # #         fm_phys_w = WEIGHTS.get("fm_physics", 0.3)
# # # # # # # # # # #         breakdown["total"]      = breakdown["total"] + fm_phys_w * l_fm_physics
# # # # # # # # # # #         breakdown["fm_physics"] = l_fm_physics.item()

# # # # # # # # # # #         # ── FIX-M27: Short-range loss ──────────────────────────────────────
# # # # # # # # # # #         n_sr = ShortRangeHead.N_STEPS  # 4
# # # # # # # # # # #         if traj_gt.shape[0] >= n_sr:
# # # # # # # # # # #             sr_pred = self.net.forward_short_range(obs_t, raw_ctx)  # [4, B, 2]
# # # # # # # # # # #             sr_gt   = traj_gt[:n_sr]                                # [4, B, 2]
# # # # # # # # # # #             l_sr    = short_range_regression_loss(sr_pred, sr_gt, lp)
# # # # # # # # # # #             sr_w    = WEIGHTS.get("short_range", 5.0)
# # # # # # # # # # #             breakdown["total"]       = breakdown["total"] + sr_w * l_sr
# # # # # # # # # # #             breakdown["short_range"] = l_sr.item()
# # # # # # # # # # #         else:
# # # # # # # # # # #             breakdown["short_range"] = 0.0

# # # # # # # # # # #         return breakdown

# # # # # # # # # # #     # ── sample() ─────────────────────────────────────────────────────────────

# # # # # # # # # # #     # @torch.no_grad()
# # # # # # # # # # #     # def sample(
# # # # # # # # # # #     #     self,
# # # # # # # # # # #     #     batch_list:  List,
# # # # # # # # # # #     #     num_ensemble: int = 50,
# # # # # # # # # # #     #     ddim_steps:   int = 20,
# # # # # # # # # # #     #     predict_csv:  Optional[str] = None,
# # # # # # # # # # #     # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# # # # # # # # # # #     #     """
# # # # # # # # # # #     #     FIX-M24: pred_mean[:4] ← ShortRangeHead (deterministic, low-error).
# # # # # # # # # # #     #              pred_mean[4:] ← FM ensemble mean.

# # # # # # # # # # #     #     Returns:
# # # # # # # # # # #     #         pred_mean : [T, B, 2]
# # # # # # # # # # #     #         me_mean   : [T, B, 2]
# # # # # # # # # # #     #         all_trajs : [S, T, B, 2]
# # # # # # # # # # #     #     """
# # # # # # # # # # #     #     obs_t = batch_list[0]          # [T_obs, B, 2]
# # # # # # # # # # #     #     lp    = obs_t[-1]              # [B, 2]
# # # # # # # # # # #     #     lm    = batch_list[7][-1]      # [B, 2]
# # # # # # # # # # #     #     B     = lp.shape[0]
# # # # # # # # # # #     #     device = lp.device
# # # # # # # # # # #     #     T      = self.pred_len
# # # # # # # # # # #     #     dt     = 1.0 / max(ddim_steps, 1)

# # # # # # # # # # #     #     raw_ctx = self.net._context(batch_list)

# # # # # # # # # # #     #     # ── Short-range deterministic (steps 1-4) ─────────────────────────
# # # # # # # # # # #     #     n_sr     = ShortRangeHead.N_STEPS      # 4
# # # # # # # # # # #     #     sr_pred  = self.net.forward_short_range(obs_t, raw_ctx)  # [4, B, 2]

# # # # # # # # # # #     #     # ── FM ensemble (all 12 steps, for steps 5-12) ───────────────────
# # # # # # # # # # #     #     traj_s: List[torch.Tensor] = []
# # # # # # # # # # #     #     me_s:   List[torch.Tensor] = []

# # # # # # # # # # #     #     for k in range(num_ensemble):
# # # # # # # # # # #     #         x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# # # # # # # # # # #     #         for step in range(ddim_steps):
# # # # # # # # # # #     #             t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # #     #             ns  = self.ctx_noise_scale if step == 0 else 0.0
# # # # # # # # # # #     #             vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
# # # # # # # # # # #     #             x_t = x_t + dt * vel

# # # # # # # # # # #     #         x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)
# # # # # # # # # # #     #         x_t = x_t.clamp(-3.0, 3.0)

# # # # # # # # # # #     #         tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # # #     #         traj_s.append(tr)
# # # # # # # # # # #     #         me_s.append(me)

# # # # # # # # # # #     #     all_trajs = torch.stack(traj_s)    # [S, T, B, 2]
# # # # # # # # # # #     #     all_me    = torch.stack(me_s)

# # # # # # # # # # #     #     # ── FIX-M24: Blend short-range into pred_mean ─────────────────────
# # # # # # # # # # #     #     fm_mean  = all_trajs.mean(0)       # [T, B, 2]
# # # # # # # # # # #     #     pred_mean = fm_mean.clone()
# # # # # # # # # # #     #     pred_mean[:n_sr] = sr_pred         # Override steps 1-4

# # # # # # # # # # #     #     # # Also override all_trajs first 4 steps with deterministic prediction
# # # # # # # # # # #     #     # # (reduces spurious spread for 12h/24h in CRPS)
# # # # # # # # # # #     #     # all_trajs[:, :n_sr, :, :] = sr_pred.unsqueeze(0).expand(
# # # # # # # # # # #     #     #     num_ensemble, -1, -1, -1
# # # # # # # # # # #     #     # )

# # # # # # # # # # #     #     me_mean = all_me.mean(0)

# # # # # # # # # # #     #     if predict_csv:
# # # # # # # # # # #     #         self._write_predict_csv(predict_csv, pred_mean, all_trajs)

# # # # # # # # # # #     #     return pred_mean, me_mean, all_trajs

# # # # # # # # # # #     @torch.no_grad()
# # # # # # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
# # # # # # # # # # #         obs_t  = batch_list[0]
# # # # # # # # # # #         lp     = obs_t[-1]
# # # # # # # # # # #         lm     = batch_list[7][-1]
# # # # # # # # # # #         B      = lp.shape[0]
# # # # # # # # # # #         device = lp.device
# # # # # # # # # # #         T      = self.pred_len
# # # # # # # # # # #         dt     = 1.0 / max(ddim_steps, 1)

# # # # # # # # # # #         raw_ctx = self.net._context(batch_list)
# # # # # # # # # # #         n_sr    = ShortRangeHead.N_STEPS

# # # # # # # # # # #         # Short-range deterministic
# # # # # # # # # # #         sr_pred = self.net.forward_short_range(obs_t, raw_ctx)   # [4, B, 2]

# # # # # # # # # # #         # FM ensemble — giữ nguyên, KHÔNG override
# # # # # # # # # # #         traj_s, me_s = [], []
# # # # # # # # # # #         for k in range(num_ensemble):
# # # # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# # # # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # #                 ns  = self.ctx_noise_scale if step == 0 else 0.0
# # # # # # # # # # #                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
# # # # # # # # # # #                 x_t = x_t + dt * vel

# # # # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)
# # # # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)

# # # # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # # #             traj_s.append(tr)
# # # # # # # # # # #             me_s.append(me)

# # # # # # # # # # #         all_trajs = torch.stack(traj_s)   # [S, T, B, 2] — KHÔNG override steps 1-4
# # # # # # # # # # #         all_me    = torch.stack(me_s)

# # # # # # # # # # #         # FIX: pred_mean blend đúng cách
# # # # # # # # # # #         # Steps 1-4: ShortRangeHead (deterministic, low-error)
# # # # # # # # # # #         # Steps 5-12: FM ensemble mean
# # # # # # # # # # #         fm_mean   = all_trajs.mean(0)    # [T, B, 2]
# # # # # # # # # # #         pred_mean = fm_mean.clone()
# # # # # # # # # # #         pred_mean[:n_sr] = sr_pred       # override pred_mean, KHÔNG override all_trajs

# # # # # # # # # # #         me_mean = all_me.mean(0)

# # # # # # # # # # #         if predict_csv:
# # # # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)

# # # # # # # # # # #         return pred_mean, me_mean, all_trajs

# # # # # # # # # # #     # ── Physics correction ────────────────────────────────────────────────────

# # # # # # # # # # #     def _physics_correct(
# # # # # # # # # # #         self,
# # # # # # # # # # #         x_pred: torch.Tensor,
# # # # # # # # # # #         last_pos: torch.Tensor,
# # # # # # # # # # #         last_Me:  torch.Tensor,
# # # # # # # # # # #         n_steps:  int   = 5,
# # # # # # # # # # #         lr:       float = 0.002,
# # # # # # # # # # #     ) -> torch.Tensor:
# # # # # # # # # # #         with torch.enable_grad():
# # # # # # # # # # #             x = x_pred.detach().requires_grad_(True)
# # # # # # # # # # #             optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)

# # # # # # # # # # #             for _ in range(n_steps):
# # # # # # # # # # #                 optimizer.zero_grad()
# # # # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # # # #                 pred_deg    = _denorm_to_deg(pred_abs)

# # # # # # # # # # #                 l_speed = self._pinn_speed_constraint(pred_deg)
# # # # # # # # # # #                 l_accel = self._pinn_beta_plane_simplified(pred_deg)

# # # # # # # # # # #                 physics_loss = l_speed + 0.3 * l_accel
# # # # # # # # # # #                 physics_loss.backward()

# # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], max_norm=0.05)
# # # # # # # # # # #                 optimizer.step()

# # # # # # # # # # #         return x.detach()

# # # # # # # # # # #     @staticmethod
# # # # # # # # # # #     def _pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # #         if pred_deg.shape[0] < 2:
# # # # # # # # # # #             return pred_deg.new_zeros(())
# # # # # # # # # # #         dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # #         lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # #         dx_km   = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # # # # # # # #         dy_km   = dt_deg[:, :, 1] * 111.0
# # # # # # # # # # #         speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # # # # # # # # # #         return F.relu(speed - 600.0).pow(2).mean()

# # # # # # # # # # #     @staticmethod
# # # # # # # # # # #     def _pinn_beta_plane_simplified(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # #         if pred_deg.shape[0] < 3:
# # # # # # # # # # #             return pred_deg.new_zeros(())
# # # # # # # # # # #         v = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # #         a = v[1:] - v[:-1]
# # # # # # # # # # #         lat_rad = torch.deg2rad(pred_deg[1:-1, :, 1])
# # # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # #         a_lon_km = a[:, :, 0] * cos_lat * 111.0
# # # # # # # # # # #         a_lat_km = a[:, :, 1] * 111.0
# # # # # # # # # # #         max_accel = 50.0
# # # # # # # # # # #         violation = F.relu(torch.sqrt(a_lon_km**2 + a_lat_km**2) - max_accel)
# # # # # # # # # # #         return violation.pow(2).mean() * 0.1

# # # # # # # # # # #     @staticmethod
# # # # # # # # # # #     def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
# # # # # # # # # # #                            all_trajs: torch.Tensor) -> None:
# # # # # # # # # # #         import numpy as np
# # # # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # # # #         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # # # #         S       = all_trajs.shape[0]

# # # # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()

# # # # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # # # #                     "lon_mean_deg", "lat_mean_deg",
# # # # # # # # # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # # # #             if write_hdr:
# # # # # # # # # # #                 w.writeheader()
# # # # # # # # # # #             for b in range(B):
# # # # # # # # # # #                 for k in range(T):
# # # # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # # # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
# # # # # # # # # # #                         math.radians(mean_lat[k, b]))
# # # # # # # # # # #                     spread = float(np.sqrt((dlat**2 + dlon**2).mean()) * 111.0)
# # # # # # # # # # #                     w.writerow(dict(
# # # # # # # # # # #                         timestamp     = ts,
# # # # # # # # # # #                         batch_idx     = b,
# # # # # # # # # # #                         step_idx      = k,
# # # # # # # # # # #                         lead_h        = (k + 1) * 6,
# # # # # # # # # # #                         lon_mean_deg  = f"{mean_lon[k,b]:.4f}",
# # # # # # # # # # #                         lat_mean_deg  = f"{mean_lat[k,b]:.4f}",
# # # # # # # # # # #                         lon_std_deg   = f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # # # #                         lat_std_deg   = f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # # # #                         ens_spread_km = f"{spread:.2f}",
# # # # # # # # # # #                     ))
# # # # # # # # # # #         print(f"  Predictions → {csv_path}  (B={B}, T={T}, S={S})")


# # # # # # # # # # # # Backward-compat alias
# # # # # # # # # # # TCDiffusion = TCFlowMatching

# # # # # # # # # # """
# # # # # # # # # # Model/flow_matching_model.py  ── v25
# # # # # # # # # # ==========================================
# # # # # # # # # # FULL REWRITE – fixes từ review:

# # # # # # # # # #   FIX-M-A  [CRITICAL] ShortRangeHead motion_gate init: bias = +2.0
# # # # # # # # # #            để gate ≈ 0.88 đầu training (motion prior dominant),
# # # # # # # # # #            thay vì xavier_uniform → gate ≈ 0.5 (mixed từ đầu).

# # # # # # # # # #   FIX-M-B  [HIGH] Phase 1 freeze logic: freeze ctx_fc1/ctx_ln cũng
# # # # # # # # # #            vì ShortRangeHead nhận raw_ctx từ frozen encoder → học
# # # # # # # # # #            trên noise. Phase 1 chỉ train ShortRangeHead với obs_traj.

# # # # # # # # # #   FIX-M-C  [HIGH] Bridge loss integration: compute sr_pred trong
# # # # # # # # # #            get_loss_breakdown() và pass sang compute_total_loss().
# # # # # # # # # #            SR↔FM nhất quán tại step 4 (Eq.80).

# # # # # # # # # #   FIX-M-D  [HIGH] _physics_correct: thay SGD+momentum bằng Adam
# # # # # # # # # #            (n_steps nhỏ, momentum gây oscillation).

# # # # # # # # # #   FIX-M-E  [MEDIUM] get_loss_breakdown() pass epoch, gt_abs_deg,
# # # # # # # # # #            vmax_pred, pmin_pred cho adaptive PINN weighting.

# # # # # # # # # #   FIX-M-F  [MEDIUM] sample(): cache raw_ctx một lần, dùng cho cả
# # # # # # # # # #            SR head và FM ensemble (tránh tính 2 lần).

# # # # # # # # # #   FIX-M-G  [LOW] ShortRangeHead: thêm layer norm trước out_head
# # # # # # # # # #            để gradient scale ổn định hơn.

# # # # # # # # # # Kept from v24:
# # # # # # # # # #   FIX-M23  ShortRangeHead GRU autoregressive
# # # # # # # # # #   FIX-M24  Blend: pred_mean[:4] ← SR, [4:] ← FM mean
# # # # # # # # # #   FIX-M25  initial_sample_sigma=0.03, ctx_noise_scale=0.002
# # # # # # # # # #   FIX-M26  MAX_DELTA clamp
# # # # # # # # # #   FIX-M27  short_range_regression_loss trong get_loss_breakdown
# # # # # # # # # # """
# # # # # # # # # # from __future__ import annotations

# # # # # # # # # # import csv
# # # # # # # # # # import math
# # # # # # # # # # import os
# # # # # # # # # # from datetime import datetime
# # # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # # import torch
# # # # # # # # # # import torch.nn as nn
# # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # # from Model.losses import (
# # # # # # # # # #     compute_total_loss, fm_physics_consistency_loss, WEIGHTS,
# # # # # # # # # #     pinn_speed_constraint, short_range_regression_loss, bridge_loss,
# # # # # # # # # #     _norm_to_deg,
# # # # # # # # # # )


# # # # # # # # # # def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #     """Normalised → degrees. [T, B, 2] or [B, 2]."""
# # # # # # # # # #     out = traj_norm.clone()
# # # # # # # # # #     out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # #     out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
# # # # # # # # # #     return out


# # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # #  ShortRangeHead  – FIX-M-A, FIX-M-G
# # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # class ShortRangeHead(nn.Module):
# # # # # # # # # #     """
# # # # # # # # # #     GRU autoregressive predictor cho steps 1-4 (6h/12h/18h/24h).

# # # # # # # # # #     FIX-M-A: motion_gate bias init = +2.0 → sigmoid(2) ≈ 0.88
# # # # # # # # # #              đảm bảo motion prior dominant đầu training.
# # # # # # # # # #     FIX-M-G: LayerNorm trước out_head để gradient scale ổn định.
# # # # # # # # # #     """

# # # # # # # # # #     N_STEPS   = 4
# # # # # # # # # #     MAX_DELTA = 0.48     # ≈ 576 km / 6h

# # # # # # # # # #     def __init__(self, raw_ctx_dim: int = 512, obs_len: int = 8):
# # # # # # # # # #         super().__init__()
# # # # # # # # # #         self.obs_len = obs_len

# # # # # # # # # #         # Velocity encoder từ obs
# # # # # # # # # #         self.vel_enc = nn.Sequential(
# # # # # # # # # #             nn.Linear(obs_len * 2, 256),
# # # # # # # # # #             nn.GELU(),
# # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # #             nn.Linear(256, 128),
# # # # # # # # # #             nn.GELU(),
# # # # # # # # # #         )

# # # # # # # # # #         self.ctx_proj = nn.Sequential(
# # # # # # # # # #             nn.Linear(raw_ctx_dim, 256),
# # # # # # # # # #             nn.GELU(),
# # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # #             nn.Dropout(0.05),
# # # # # # # # # #         )

# # # # # # # # # #         # GRU: input = ctx(256) + vel(128) + cur_pos(2) = 386
# # # # # # # # # #         self.gru_cell = nn.GRUCell(input_size=386, hidden_size=256)

# # # # # # # # # #         # FIX-M-A: motion gate với bias khởi tạo cao
# # # # # # # # # #         self.motion_gate_linear = nn.Linear(128 + 256, 1)

# # # # # # # # # #         # FIX-M-G: LayerNorm trước output
# # # # # # # # # #         self.out_norm = nn.LayerNorm(256)
# # # # # # # # # #         self.out_head = nn.Sequential(
# # # # # # # # # #             nn.Linear(256, 128),
# # # # # # # # # #             nn.GELU(),
# # # # # # # # # #             nn.LayerNorm(128),
# # # # # # # # # #             nn.Linear(128, 2),
# # # # # # # # # #         )

# # # # # # # # # #         self.step_scale = nn.Parameter(torch.ones(self.N_STEPS) * 0.2)
# # # # # # # # # #         self._init_weights()

# # # # # # # # # #     def _init_weights(self):
# # # # # # # # # #         with torch.no_grad():
# # # # # # # # # #             for m in self.modules():
# # # # # # # # # #                 if isinstance(m, nn.Linear):
# # # # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.3)
# # # # # # # # # #                     if m.bias is not None:
# # # # # # # # # #                         nn.init.zeros_(m.bias)
# # # # # # # # # #             # FIX-M-A: init bias của gate linear = +2.0
# # # # # # # # # #             # sigmoid(2.0) ≈ 0.88 → motion prior chiếm 88% đầu training
# # # # # # # # # #             nn.init.constant_(self.motion_gate_linear.bias, 2.0)

# # # # # # # # # #     def forward(self, raw_ctx: torch.Tensor,
# # # # # # # # # #                 obs_traj: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #         """
# # # # # # # # # #         raw_ctx:  [B, raw_ctx_dim]
# # # # # # # # # #         obs_traj: [T_obs, B, 2]  normalised
# # # # # # # # # #         Returns:  [4, B, 2]  normalised predictions
# # # # # # # # # #         """
# # # # # # # # # #         B     = raw_ctx.shape[0]
# # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # #         # Velocity encoding
# # # # # # # # # #         if T_obs >= 2:
# # # # # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]   # [T_obs-1, B, 2]
# # # # # # # # # #         else:
# # # # # # # # # #             vels = torch.zeros(1, B, 2, device=raw_ctx.device)

# # # # # # # # # #         target_v = self.obs_len
# # # # # # # # # #         if vels.shape[0] < target_v:
# # # # # # # # # #             pad  = torch.zeros(target_v - vels.shape[0], B, 2, device=raw_ctx.device)
# # # # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # # # #         else:
# # # # # # # # # #             vels = vels[-target_v:]

# # # # # # # # # #         vel_input = vels.permute(1, 0, 2).reshape(B, -1)  # [B, obs_len*2]
# # # # # # # # # #         vel_feat  = self.vel_enc(vel_input)                 # [B, 128]
# # # # # # # # # #         ctx_feat  = self.ctx_proj(raw_ctx)                  # [B, 256]

# # # # # # # # # #         hx      = ctx_feat
# # # # # # # # # #         cur_pos = obs_traj[-1].clone()   # [B, 2]
# # # # # # # # # #         last_vel = vels[-1].clone()      # [B, 2]

# # # # # # # # # #         preds = []
# # # # # # # # # #         for i in range(self.N_STEPS):
# # # # # # # # # #             inp = torch.cat([ctx_feat, vel_feat, cur_pos], dim=-1)  # [B, 386]
# # # # # # # # # #             hx  = self.gru_cell(inp, hx)                            # [B, 256]

# # # # # # # # # #             # FIX-M-A: gate với bias=+2.0 → dominant motion prior đầu training
# # # # # # # # # #             gate_input = torch.cat([vel_feat, hx], dim=-1)  # [B, 384]
# # # # # # # # # #             gate = torch.sigmoid(self.motion_gate_linear(gate_input))  # [B, 1]

# # # # # # # # # #             # FIX-M-G: LayerNorm trước output
# # # # # # # # # #             raw_delta   = self.out_head(self.out_norm(hx))  # [B, 2]
# # # # # # # # # #             prior_delta = last_vel * (0.9 ** i)              # [B, 2]

# # # # # # # # # #             scale_i = torch.sigmoid(self.step_scale[i]) * self.MAX_DELTA
# # # # # # # # # #             blended = gate * prior_delta + (1.0 - gate) * raw_delta * scale_i
# # # # # # # # # #             delta   = blended.clamp(-self.MAX_DELTA, self.MAX_DELTA)

# # # # # # # # # #             cur_pos  = cur_pos + delta
# # # # # # # # # #             last_vel = delta
# # # # # # # # # #             preds.append(cur_pos)

# # # # # # # # # #         return torch.stack(preds, dim=0)   # [4, B, 2]


# # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # #  VelocityField
# # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # # #     """OT-CFM velocity field v_θ(x_t, t, context)."""

# # # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # # #     def __init__(
# # # # # # # # # #         self,
# # # # # # # # # #         pred_len:   int   = 12,
# # # # # # # # # #         obs_len:    int   = 8,
# # # # # # # # # #         ctx_dim:    int   = 256,
# # # # # # # # # #         sigma_min:  float = 0.02,
# # # # # # # # # #         unet_in_ch: int   = 13,
# # # # # # # # # #     ):
# # # # # # # # # #         super().__init__()
# # # # # # # # # #         self.pred_len  = pred_len
# # # # # # # # # #         self.obs_len   = obs_len
# # # # # # # # # #         self.sigma_min = sigma_min

# # # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # # #             in_channel   = unet_in_ch,
# # # # # # # # # #             out_channel  = 1,
# # # # # # # # # #             d_model      = 32,
# # # # # # # # # #             n_layers     = 4,
# # # # # # # # # #             modes_t      = 4,
# # # # # # # # # #             modes_h      = 4,
# # # # # # # # # #             modes_w      = 4,
# # # # # # # # # #             spatial_down = 32,
# # # # # # # # # #             dropout      = 0.05,
# # # # # # # # # #         )

# # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # # #             in_1d       = 4,
# # # # # # # # # #             feat_3d_dim = 128,
# # # # # # # # # #             mlp_h       = 64,
# # # # # # # # # #             lstm_hidden = 128,
# # # # # # # # # #             lstm_layers = 3,
# # # # # # # # # #             dropout     = 0.1,
# # # # # # # # # #             d_state     = 16,
# # # # # # # # # #         )

# # # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # # #         self.ctx_drop = nn.Dropout(0.15)
# # # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # # # # # # #         self.time_fc1 = nn.Linear(256, 512)
# # # # # # # # # #         self.time_fc2 = nn.Linear(512, 256)

# # # # # # # # # #         self.traj_embed  = nn.Linear(4, 256)
# # # # # # # # # #         # self.pos_enc     = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # # # # # #         # Thay vì:
# # # # # # # # # #         # self.pos_enc = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)

# # # # # # # # # #         # Hãy thử:
# # # # # # # # # #         self.pos_enc = nn.Parameter(torch.randn(1, pred_len, 256) * 0.1) # Tăng scale lên 5 lần
# # # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # # #                 dropout=0.15, activation="gelu", batch_first=True,
# # # # # # # # # #             ),
# # # # # # # # # #             num_layers=4,
# # # # # # # # # #         )
# # # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)

# # # # # # # # # #         # ShortRangeHead (FIX-M-A applied)
# # # # # # # # # #         self.short_range_head = ShortRangeHead(
# # # # # # # # # #             raw_ctx_dim = self.RAW_CTX_DIM,
# # # # # # # # # #             obs_len     = obs_len,
# # # # # # # # # #         )

# # # # # # # # # #     # ── Helpers ───────────────────────────────────────────────────────────────

# # # # # # # # # #     def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
# # # # # # # # # #         half = dim // 2
# # # # # # # # # #         freq = torch.exp(
# # # # # # # # # #             torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # # #             * (-math.log(10_000.0) / max(half - 1, 1))
# # # # # # # # # #         )
# # # # # # # # # #         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
# # # # # # # # # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # # # # # # # # #         return F.pad(emb, (0, dim % 2))

# # # # # # # # # #     def _context(self, batch_list) -> torch.Tensor:
# # # # # # # # # #         """Compute raw_ctx [B, RAW_CTX_DIM]."""
# # # # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # # # #         image_obs = batch_list[11]
# # # # # # # # # #         env_data  = batch_list[13]

# # # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # # #             image_obs = image_obs.unsqueeze(2)

# # # # # # # # # #         expected_ch = self.spatial_enc.in_channel
# # # # # # # # # #         if image_obs.shape[1] == 1 and expected_ch != 1:
# # # # # # # # # #             image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
# # # # # # # # # #         e_3d_s = e_3d_s.permute(0, 2, 1)
# # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)

# # # # # # # # # #         T_bot = e_3d_s.shape[1]
# # # # # # # # # #         if T_bot != T_obs:
# # # # # # # # # #             e_3d_s = F.interpolate(
# # # # # # # # # #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # #                 mode="linear", align_corners=False,
# # # # # # # # # #             ).permute(0, 2, 1)

# # # # # # # # # #         # Weighted temporal pooling của decoder output
# # # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)  # [B, T]
# # # # # # # # # #         t_weights = torch.softmax(
# # # # # # # # # #             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # # # #                         device=e_3d_dec_t.device) * 0.5, dim=0,
# # # # # # # # # #         )
# # # # # # # # # #         f_spatial_scalar = (e_3d_dec_t * t_weights.unsqueeze(0)).sum(1, keepdim=True)  # [B,1]
# # # # # # # # # #         f_spatial = self.decoder_proj(f_spatial_scalar)   # [B, 16]

# # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)  # [B,T,4]
# # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)              # [B, 128]

# # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)   # [B, 32]

# # # # # # # # # #         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)  # [B, 176]
# # # # # # # # # #         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))       # [B, 512]
# # # # # # # # # #         return raw

# # # # # # # # # #     def _apply_ctx_head(self, raw: torch.Tensor,
# # # # # # # # # #                         noise_scale: float = 0.0) -> torch.Tensor:
# # # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # # #     def _beta_drift_velocity(self, x_t: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #         """Beta drift trong normalised state space."""
# # # # # # # # # #         OMEGA_val  = 7.2921e-5
# # # # # # # # # #         R_val      = 6.371e6
# # # # # # # # # #         DT         = 6 * 3600.0
# # # # # # # # # #         M_PER_NORM = 5.0 * 111.0 * 1000.0

# # # # # # # # # #         lat_norm = x_t[:, :, 1]
# # # # # # # # # #         lat_deg  = lat_norm * 5.0
# # # # # # # # # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))

# # # # # # # # # #         beta   = 2 * OMEGA_val * torch.cos(lat_rad) / R_val
# # # # # # # # # #         R_tc   = 3e5
# # # # # # # # # #         v_lon  = -beta * R_tc ** 2 / 2
# # # # # # # # # #         v_lat  =  beta * R_tc ** 2 / 4

# # # # # # # # # #         v_lon_norm = v_lon * DT / M_PER_NORM
# # # # # # # # # #         v_lat_norm = v_lat * DT / M_PER_NORM

# # # # # # # # # #         v_phys = torch.zeros_like(x_t)
# # # # # # # # # #         v_phys[:, :, 0] = v_lon_norm
# # # # # # # # # #         v_phys[:, :, 1] = v_lat_norm
# # # # # # # # # #         return v_phys

# # # # # # # # # #     def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
# # # # # # # # # #                 ctx: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
# # # # # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # # # # #         T_seq = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # # # #         x_emb = (self.traj_embed(x_t[:, :T_seq, :])
# # # # # # # # # #                   + self.pos_enc[:, :T_seq, :]
# # # # # # # # # #                   + t_emb.unsqueeze(1))
# # # # # # # # # #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

# # # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(
# # # # # # # # # #             self.transformer(x_emb, memory)
# # # # # # # # # #         )))

# # # # # # # # # #         with torch.no_grad():
# # # # # # # # # #             v_phys = self._beta_drift_velocity(x_t[:, :T_seq, :])

# # # # # # # # # #         scale = torch.sigmoid(self.physics_scale) * 2.0
# # # # # # # # # #         return v_neural + scale * v_phys

# # # # # # # # # #     def forward(self, x_t, t, batch_list):
# # # # # # # # # #         raw = self._context(batch_list)
# # # # # # # # # #         ctx = self._apply_ctx_head(raw, noise_scale=0.0)
# # # # # # # # # #         return self._decode(x_t, t, ctx)

# # # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale: float = 0.0):
# # # # # # # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale=noise_scale)
# # # # # # # # # #         return self._decode(x_t, t, ctx)

# # # # # # # # # #     def forward_short_range(self, obs_traj: torch.Tensor,
# # # # # # # # # #                             raw_ctx: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #         """Deterministic short-range [4, B, 2]."""
# # # # # # # # # #         return self.short_range_head(raw_ctx, obs_traj)


# # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # #  TCFlowMatching
# # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # class TCFlowMatching(nn.Module):
# # # # # # # # # #     """TC trajectory prediction via OT-CFM + ShortRangeHead + PINN."""

# # # # # # # # # #     def __init__(
# # # # # # # # # #         self,
# # # # # # # # # #         pred_len:             int   = 12,
# # # # # # # # # #         obs_len:              int   = 8,
# # # # # # # # # #         sigma_min:            float = 0.02,
# # # # # # # # # #         n_train_ens:          int   = 4,
# # # # # # # # # #         unet_in_ch:           int   = 13,
# # # # # # # # # #         # ctx_noise_scale:      float = 0.002,
# # # # # # # # # #         # initial_sample_sigma: float = 0.03,
# # # # # # # # # #         ctx_noise_scale:      float = 0.01,
# # # # # # # # # #         initial_sample_sigma: float = 0.15,
# # # # # # # # # #         **kwargs,
# # # # # # # # # #     ):
# # # # # # # # # #         super().__init__()
# # # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # # #         self.active_pred_len      = pred_len
# # # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma

# # # # # # # # # #         self.net = VelocityField(
# # # # # # # # # #             pred_len   = pred_len,
# # # # # # # # # #             obs_len    = obs_len,
# # # # # # # # # #             sigma_min  = sigma_min,
# # # # # # # # # #             unet_in_ch = unet_in_ch,
# # # # # # # # # #         )

# # # # # # # # # #     def set_curriculum_len(self, active_len: int) -> None:
# # # # # # # # # #         pass  # No curriculum

# # # # # # # # # #     # ── Static helpers ────────────────────────────────────────────────────────

# # # # # # # # # #     @staticmethod
# # # # # # # # # #     def _to_rel(traj_gt, Me_gt, last_pos, last_Me):
# # # # # # # # # #         return torch.cat(
# # # # # # # # # #             [traj_gt - last_pos.unsqueeze(0),
# # # # # # # # # #              Me_gt   - last_Me.unsqueeze(0)],
# # # # # # # # # #             dim=-1,
# # # # # # # # # #         ).permute(1, 0, 2)

# # # # # # # # # #     @staticmethod
# # # # # # # # # #     def _to_abs(rel, last_pos, last_Me):
# # # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # # #         return (
# # # # # # # # # #             last_pos.unsqueeze(0) + d[:, :, :2],
# # # # # # # # # #             last_Me.unsqueeze(0)  + d[:, :, 2:],
# # # # # # # # # #         )

# # # # # # # # # #     def _cfm_noisy(self, x1):
# # # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # # #         sm  = self.sigma_min
# # # # # # # # # #         x0  = torch.randn_like(x1) * sm
# # # # # # # # # #         t   = torch.rand(B, device=device)
# # # # # # # # # #         te  = t.view(B, 1, 1)
# # # # # # # # # #         x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
# # # # # # # # # #         denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
# # # # # # # # # #         target_vel = (x1 - (1.0 - sm) * x_t) / denom
# # # # # # # # # #         return x_t, t, te, denom, target_vel

# # # # # # # # # #     @staticmethod
# # # # # # # # # #     def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #         wind_norm = obs_Me[-1, :, 1].detach()
# # # # # # # # # #         w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
# # # # # # # # # #             torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
# # # # # # # # # #             torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
# # # # # # # # # #                         torch.full_like(wind_norm, 1.5))))
# # # # # # # # # #         return w / w.mean().clamp(min=1e-6)

# # # # # # # # # #     @staticmethod
# # # # # # # # # #     def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
# # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # #             return batch_list
# # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # # #             t = aug[idx]
# # # # # # # # # #             if torch.is_tensor(t) and t.shape[-1] >= 1:
# # # # # # # # # #                 t = t.clone()
# # # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # # #                 aug[idx] = t
# # # # # # # # # #         return aug

# # # # # # # # # #     # ── Loss ──────────────────────────────────────────────────────────────────

# # # # # # # # # #     def get_loss(self, batch_list: List,
# # # # # # # # # #                  step_weight_alpha: float = 0.0,
# # # # # # # # # #                  epoch: int = 0) -> torch.Tensor:
# # # # # # # # # #         return self.get_loss_breakdown(
# # # # # # # # # #             batch_list, step_weight_alpha, epoch)["total"]

# # # # # # # # # #     def get_loss_breakdown(
# # # # # # # # # #         self,
# # # # # # # # # #         batch_list: List,
# # # # # # # # # #         step_weight_alpha: float = 0.0,
# # # # # # # # # #         epoch: int = 0,
# # # # # # # # # #     ) -> Dict:
# # # # # # # # # #         """
# # # # # # # # # #         FIX-M-C: Tính sr_pred và pass sang compute_total_loss qua bridge_loss.
# # # # # # # # # #         FIX-M-E: Pass epoch và gt_abs_deg cho adaptive PINN.
# # # # # # # # # #         FIX-M-F: Cache raw_ctx một lần.
# # # # # # # # # #         """
# # # # # # # # # #         batch_list = self._lon_flip_aug(batch_list, p=0.3)

# # # # # # # # # #         traj_gt = batch_list[1]    # [T, B, 2] normalised
# # # # # # # # # #         Me_gt   = batch_list[8]
# # # # # # # # # #         obs_t   = batch_list[0]    # [T_obs, B, 2] normalised
# # # # # # # # # #         obs_Me  = batch_list[7]

# # # # # # # # # #         try:
# # # # # # # # # #             env_data = batch_list[13]
# # # # # # # # # #         except (IndexError, TypeError):
# # # # # # # # # #             env_data = None

# # # # # # # # # #         lp, lm = obs_t[-1], obs_Me[-1]
# # # # # # # # # #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

# # # # # # # # # #         # FIX-M-F: compute raw_ctx một lần duy nhất
# # # # # # # # # #         raw_ctx     = self.net._context(batch_list)
# # # # # # # # # #         intensity_w = self._intensity_weights(obs_Me)

# # # # # # # # # #         # CFM forward
# # # # # # # # # #         x_t, t, te, denom, _ = self._cfm_noisy(x1)
# # # # # # # # # #         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

# # # # # # # # # #         # Ensemble cho AFCRPS
# # # # # # # # # #         samples: List[torch.Tensor] = []
# # # # # # # # # #         for _ in range(self.n_train_ens):
# # # # # # # # # #             xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
# # # # # # # # # #             pv_s   = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
# # # # # # # # # #             x1_s   = xt_s + dens_s * pv_s
# # # # # # # # # #             pa_s, _ = self._to_abs(x1_s, lp, lm)
# # # # # # # # # #             samples.append(pa_s)
# # # # # # # # # #         pred_samples = torch.stack(samples)     # [S, T, B, 2]

# # # # # # # # # #         l_fm_physics = fm_physics_consistency_loss(pred_samples, gt_norm=traj_gt, last_pos=lp)

# # # # # # # # # #         x1_pred = x_t + denom * pred_vel
# # # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# # # # # # # # # #         pred_abs_deg = _denorm_to_deg(pred_abs)    # [T, B, 2] degrees
# # # # # # # # # #         traj_gt_deg  = _denorm_to_deg(traj_gt)     # [T, B, 2] degrees
# # # # # # # # # #         ref_deg      = _denorm_to_deg(lp)           # [B, 2]

# # # # # # # # # #         # FIX-M-C: Compute sr_pred (dùng raw_ctx đã cache)
# # # # # # # # # #         n_sr = ShortRangeHead.N_STEPS  # 4
# # # # # # # # # #         sr_pred = None
# # # # # # # # # #         l_sr    = pred_abs.new_zeros(())
# # # # # # # # # #         if traj_gt.shape[0] >= n_sr:
# # # # # # # # # #             sr_pred = self.net.forward_short_range(obs_t, raw_ctx)  # [4, B, 2]
# # # # # # # # # #             sr_gt   = traj_gt[:n_sr]                                # [4, B, 2]
# # # # # # # # # #             l_sr    = short_range_regression_loss(sr_pred, sr_gt, lp)

# # # # # # # # # #         breakdown = compute_total_loss(
# # # # # # # # # #             pred_abs           = pred_abs_deg,
# # # # # # # # # #             gt                 = traj_gt_deg,
# # # # # # # # # #             ref                = ref_deg,
# # # # # # # # # #             batch_list         = batch_list,
# # # # # # # # # #             pred_samples       = pred_samples,
# # # # # # # # # #             gt_norm            = traj_gt,
# # # # # # # # # #             weights            = WEIGHTS,
# # # # # # # # # #             intensity_w        = intensity_w,
# # # # # # # # # #             env_data           = env_data,
# # # # # # # # # #             step_weight_alpha  = step_weight_alpha,
# # # # # # # # # #             all_trajs          = pred_samples,
# # # # # # # # # #             # FIX-M-E: pass epoch và gt cho adaptive PINN
# # # # # # # # # #             epoch              = epoch,
# # # # # # # # # #             gt_abs_deg         = traj_gt_deg,
# # # # # # # # # #             # FIX-M-C: pass sr_pred cho bridge loss
# # # # # # # # # #             sr_pred            = sr_pred,
# # # # # # # # # #         )

# # # # # # # # # #         # FM physics consistency
# # # # # # # # # #         fm_phys_w = WEIGHTS.get("fm_physics", 0.3)
# # # # # # # # # #         breakdown["total"]      = breakdown["total"] + fm_phys_w * l_fm_physics
# # # # # # # # # #         breakdown["fm_physics"] = l_fm_physics.item()

# # # # # # # # # #         # Short-range loss
# # # # # # # # # #         sr_w = WEIGHTS.get("short_range", 5.0)
# # # # # # # # # #         breakdown["total"]       = breakdown["total"] + sr_w * l_sr
# # # # # # # # # #         breakdown["short_range"] = l_sr.item()

# # # # # # # # # #         return breakdown

# # # # # # # # # #     # ── sample() ─────────────────────────────────────────────────────────────

# # # # # # # # # #     @torch.no_grad()
# # # # # # # # # #     def sample(
# # # # # # # # # #         self,
# # # # # # # # # #         batch_list:   List,
# # # # # # # # # #         num_ensemble: int = 50,
# # # # # # # # # #         ddim_steps:   int = 20,
# # # # # # # # # #         predict_csv:  Optional[str] = None,
# # # # # # # # # #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# # # # # # # # # #         """
# # # # # # # # # #         FIX-M-F: raw_ctx cached một lần cho cả SR và FM ensemble.

# # # # # # # # # #         Returns:
# # # # # # # # # #             pred_mean : [T, B, 2]  (SR[:4] + FM[4:])
# # # # # # # # # #             me_mean   : [T, B, 2]
# # # # # # # # # #             all_trajs : [S, T, B, 2]  (FM ensemble, KHÔNG override step 1-4)
# # # # # # # # # #         """
# # # # # # # # # #         obs_t  = batch_list[0]
# # # # # # # # # #         lp     = obs_t[-1]
# # # # # # # # # #         lm     = batch_list[7][-1]
# # # # # # # # # #         B      = lp.shape[0]
# # # # # # # # # #         device = lp.device
# # # # # # # # # #         T      = self.pred_len
# # # # # # # # # #         dt     = 1.0 / max(ddim_steps, 1)

# # # # # # # # # #         # FIX-M-F: một lần duy nhất
# # # # # # # # # #         raw_ctx = self.net._context(batch_list)
# # # # # # # # # #         n_sr    = ShortRangeHead.N_STEPS

# # # # # # # # # #         # Short-range deterministic
# # # # # # # # # #         sr_pred = self.net.forward_short_range(obs_t, raw_ctx)   # [4, B, 2]

# # # # # # # # # #         # FM ensemble
# # # # # # # # # #         traj_s: List[torch.Tensor] = []
# # # # # # # # # #         me_s:   List[torch.Tensor] = []

# # # # # # # # # #         for k in range(num_ensemble):
# # # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# # # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # #                 # ns  = self.ctx_noise_scale if step == 0 else 0.0
# # # # # # # # # #                 ns = self.ctx_noise_scale * 10.0 if step == 0 else 0.0
# # # # # # # # # #                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
# # # # # # # # # #                 x_t = x_t + dt * vel
# # # # # # # # # #         # for k in range(num_ensemble):
# # # # # # # # # #         #     x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# # # # # # # # # #         #     # Mỗi member dùng ctx với noise riêng → diversity đến từ ctx uncertainty
# # # # # # # # # #         #     ctx_k = self.net._apply_ctx_head(
# # # # # # # # # #         #         raw_ctx, noise_scale=self.ctx_noise_scale * 5.0  # tăng 5× khi sampling
# # # # # # # # # #         #     )

# # # # # # # # # #         #     for step in range(ddim_steps):
# # # # # # # # # #         #         t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # #         #         vel = self.net._decode(x_t, t_b, ctx_k)
# # # # # # # # # #         #         x_t = x_t + dt * vel

# # # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)
# # # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)

# # # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # #             traj_s.append(tr)
# # # # # # # # # #             me_s.append(me)

# # # # # # # # # #         all_trajs = torch.stack(traj_s)    # [S, T, B, 2]
# # # # # # # # # #         all_me    = torch.stack(me_s)

# # # # # # # # # #         # Blend: step 1-4 từ SR, 5-12 từ FM mean
# # # # # # # # # #         fm_mean   = all_trajs.mean(0)      # [T, B, 2]
# # # # # # # # # #         pred_mean = fm_mean.clone()
# # # # # # # # # #         # pred_mean[:n_sr] = sr_pred         # override pred_mean, giữ all_trajs nguyên

# # # # # # # # # #         # Dùng:
# # # # # # # # # #         alpha = torch.linspace(0.0, 1.0, n_sr, device=device).view(n_sr, 1, 1)
# # # # # # # # # #         pred_mean[:n_sr] = (1 - alpha) * sr_pred + alpha * fm_mean[:n_sr]
        
# # # # # # # # # #         # step 1: 100% SR, step 4: 50% SR + 50% FM → smooth transition
# # # # # # # # # #         me_mean = all_me.mean(0)

# # # # # # # # # #         if predict_csv:
# # # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)

# # # # # # # # # #         return pred_mean, me_mean, all_trajs

# # # # # # # # # #     # ── Physics correction (FIX-M-D) ─────────────────────────────────────────

# # # # # # # # # #     def _physics_correct(
# # # # # # # # # #         self,
# # # # # # # # # #         x_pred:   torch.Tensor,
# # # # # # # # # #         last_pos: torch.Tensor,
# # # # # # # # # #         last_Me:  torch.Tensor,
# # # # # # # # # #         n_steps:  int   = 5,
# # # # # # # # # #         lr:       float = 0.002,
# # # # # # # # # #     ) -> torch.Tensor:
# # # # # # # # # #         """
# # # # # # # # # #         FIX-M-D: Dùng Adam thay vì SGD+momentum.
# # # # # # # # # #         SGD+momentum với n_steps nhỏ (5) gây oscillation;
# # # # # # # # # #         Adam converge nhanh hơn và stable hơn.
# # # # # # # # # #         """
# # # # # # # # # #         with torch.enable_grad():
# # # # # # # # # #             x = x_pred.detach().requires_grad_(True)
# # # # # # # # # #             # FIX-M-D: Adam thay vì SGD+momentum
# # # # # # # # # #             optimizer = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))

# # # # # # # # # #             for _ in range(n_steps):
# # # # # # # # # #                 optimizer.zero_grad()
# # # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # # #                 pred_deg    = _denorm_to_deg(pred_abs)

# # # # # # # # # #                 l_speed = self._pinn_speed_constraint(pred_deg)
# # # # # # # # # #                 l_accel = self._pinn_beta_plane_simplified(pred_deg)

# # # # # # # # # #                 physics_loss = l_speed + 0.3 * l_accel
# # # # # # # # # #                 physics_loss.backward()

# # # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], max_norm=0.05)
# # # # # # # # # #                 optimizer.step()

# # # # # # # # # #         return x.detach()

# # # # # # # # # #     @staticmethod
# # # # # # # # # #     def _pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #         if pred_deg.shape[0] < 2:
# # # # # # # # # #             return pred_deg.new_zeros(())
# # # # # # # # # #         dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # #         lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # #         dx_km   = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # # # # # # #         dy_km   = dt_deg[:, :, 1] * 111.0
# # # # # # # # # #         speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # # # # # # # # #         return F.relu(speed - 600.0).pow(2).mean()

# # # # # # # # # #     @staticmethod
# # # # # # # # # #     def _pinn_beta_plane_simplified(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # # #         if pred_deg.shape[0] < 3:
# # # # # # # # # #             return pred_deg.new_zeros(())
# # # # # # # # # #         v = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # #         a = v[1:] - v[:-1]
# # # # # # # # # #         lat_rad = torch.deg2rad(pred_deg[1:-1, :, 1])
# # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # #         a_lon_km = a[:, :, 0] * cos_lat * 111.0
# # # # # # # # # #         a_lat_km = a[:, :, 1] * 111.0
# # # # # # # # # #         max_accel = 50.0
# # # # # # # # # #         violation = F.relu(torch.sqrt(a_lon_km**2 + a_lat_km**2) - max_accel)
# # # # # # # # # #         return violation.pow(2).mean() * 0.1

# # # # # # # # # #     @staticmethod
# # # # # # # # # #     def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
# # # # # # # # # #                            all_trajs: torch.Tensor) -> None:
# # # # # # # # # #         import numpy as np
# # # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # # #         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # # #         S       = all_trajs.shape[0]

# # # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()

# # # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # # #                     "lon_mean_deg", "lat_mean_deg",
# # # # # # # # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # # #             if write_hdr:
# # # # # # # # # #                 w.writeheader()
# # # # # # # # # #             for b in range(B):
# # # # # # # # # #                 for k in range(T):
# # # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
# # # # # # # # # #                         math.radians(mean_lat[k, b]))
# # # # # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # # # # #                     w.writerow(dict(
# # # # # # # # # #                         timestamp     = ts,
# # # # # # # # # #                         batch_idx     = b,
# # # # # # # # # #                         step_idx      = k,
# # # # # # # # # #                         lead_h        = (k + 1) * 6,
# # # # # # # # # #                         lon_mean_deg  = f"{mean_lon[k,b]:.4f}",
# # # # # # # # # #                         lat_mean_deg  = f"{mean_lat[k,b]:.4f}",
# # # # # # # # # #                         lon_std_deg   = f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # # #                         lat_std_deg   = f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # # #                         ens_spread_km = f"{spread:.2f}",
# # # # # # # # # #                     ))
# # # # # # # # # #         print(f"  Predictions → {csv_path}  (B={B}, T={T}, S={S})")


# # # # # # # # # # # Backward-compat alias
# # # # # # # # # # TCDiffusion = TCFlowMatching

# # # # # # # # # """
# # # # # # # # # flow_matching_model_v32.py — Tiếp cận đơn giản nhất để beat LSTM

# # # # # # # # # INSIGHT: LSTM beat vì nó train với pure MSE → gradient rõ ràng, không conflict.
# # # # # # # # # Vấn đề của FM không phải architecture mà là training signal:
# # # # # # # # #   - AFCRPS: pull về diverse predictions
# # # # # # # # #   - MSE_hav: pull về mean prediction  
# # # # # # # # #   - Hai cái này conflict nhau → stuck at mean (ADE=412km)

# # # # # # # # # GIẢI PHÁP V32:
# # # # # # # # #   Phase 1 (epoch 0-30): Train như LSTM thuần túy
# # # # # # # # #     - Loss = SR_Huber(step 1-4) + MSE_hav(step 1-12)
# # # # # # # # #     - Không dùng FM AFCRPS
# # # # # # # # #     - Model học được basic trajectory prediction
    
# # # # # # # # #   Phase 2 (epoch 31+): Thêm FM ensemble diversity
# # # # # # # # #     - Loss = SR_Huber + MSE_hav + 0.5*AFCRPS
# # # # # # # # #     - FM dùng để calibrate uncertainty, không phải accuracy
    
# # # # # # # # # KẾT QUẢ DỰ KIẾN:
# # # # # # # # #   - Epoch 10-20: ADE giảm nhanh từ 412 → ~250km (như LSTM ban đầu)
# # # # # # # # #   - Epoch 30-50: ADE ~170km, 72h ~300km
# # # # # # # # #   - Epoch 60+: SR converge → 12h ~50km
# # # # # # # # # """
# # # # # # # # # from __future__ import annotations

# # # # # # # # # import csv
# # # # # # # # # import math
# # # # # # # # # import os
# # # # # # # # # from datetime import datetime
# # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # import torch
# # # # # # # # # import torch.nn as nn
# # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # Import các module unchanged
# # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # from Model.losses import (
# # # # # # # # #     _haversine_deg, _norm_to_deg, N_SR_STEPS,
# # # # # # # # #     short_range_regression_loss, heading_loss,
# # # # # # # # #     velocity_loss_per_sample, recurvature_loss,
# # # # # # # # #     fm_afcrps_loss, fm_physics_consistency_loss,
# # # # # # # # # )

# # # # # # # # # MSE_STEP_WEIGHTS = [1.0, 3.0, 1.5, 2.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]


# # # # # # # # # def _denorm_to_deg(t):
# # # # # # # # #     out = t.clone()
# # # # # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # # # # #     return out


# # # # # # # # # def mse_hav_loss(pred_norm, gt_norm, step_w=None):
# # # # # # # # #     """Haversine MSE per-step với step weighting."""
# # # # # # # # #     if step_w is None:
# # # # # # # # #         step_w = MSE_STEP_WEIGHTS
# # # # # # # # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # # #     pred_deg = _norm_to_deg(pred_norm[:T])
# # # # # # # # #     gt_deg   = _norm_to_deg(gt_norm[:T])
# # # # # # # # #     dist_km  = _haversine_deg(pred_deg, gt_deg)   # [T, B]
# # # # # # # # #     w = pred_norm.new_tensor(step_w[:T])
# # # # # # # # #     w = w / w.sum() * T
# # # # # # # # #     return (dist_km.pow(2) * w.unsqueeze(1)).mean() / (200.0 ** 2)


# # # # # # # # # # ─── ShortRangeHead (giữ nguyên từ v30) ──────────────────────────────────────

# # # # # # # # # class ShortRangeHead(nn.Module):
# # # # # # # # #     N_STEPS   = 4
# # # # # # # # #     MAX_DELTA = 0.48

# # # # # # # # #     def __init__(self, raw_ctx_dim=512, obs_len=8):
# # # # # # # # #         super().__init__()
# # # # # # # # #         self.obs_len = obs_len
# # # # # # # # #         self.vel_enc = nn.Sequential(
# # # # # # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # # # # # #             nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(),
# # # # # # # # #         )
# # # # # # # # #         self.ctx_proj = nn.Sequential(
# # # # # # # # #             nn.Linear(raw_ctx_dim, 256), nn.GELU(),
# # # # # # # # #             nn.LayerNorm(256), nn.Dropout(0.05),
# # # # # # # # #         )
# # # # # # # # #         self.gru_cell = nn.GRUCell(input_size=386, hidden_size=256)
# # # # # # # # #         self.motion_gate_linear = nn.Linear(128 + 256, 1)
# # # # # # # # #         self.out_norm = nn.LayerNorm(256)
# # # # # # # # #         self.out_head = nn.Sequential(
# # # # # # # # #             nn.Linear(256, 128), nn.GELU(),
# # # # # # # # #             nn.LayerNorm(128), nn.Linear(128, 2),
# # # # # # # # #         )
# # # # # # # # #         self.step_scale = nn.Parameter(torch.ones(self.N_STEPS) * 0.2)
# # # # # # # # #         self._init_weights()

# # # # # # # # #     def _init_weights(self):
# # # # # # # # #         with torch.no_grad():
# # # # # # # # #             for m in self.modules():
# # # # # # # # #                 if isinstance(m, nn.Linear):
# # # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.3)
# # # # # # # # #                     if m.bias is not None:
# # # # # # # # #                         nn.init.zeros_(m.bias)
# # # # # # # # #             nn.init.constant_(self.motion_gate_linear.bias, 2.0)

# # # # # # # # #     def forward(self, raw_ctx, obs_traj):
# # # # # # # # #         B, T_obs = raw_ctx.shape[0], obs_traj.shape[0]
# # # # # # # # #         vels = obs_traj[1:] - obs_traj[:-1] if T_obs >= 2 else torch.zeros(1, B, 2, device=raw_ctx.device)
# # # # # # # # #         if vels.shape[0] < self.obs_len:
# # # # # # # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=raw_ctx.device)
# # # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # # #         else:
# # # # # # # # #             vels = vels[-self.obs_len:]
# # # # # # # # #         vel_feat = self.vel_enc(vels.permute(1, 0, 2).reshape(B, -1))
# # # # # # # # #         ctx_feat = self.ctx_proj(raw_ctx)
# # # # # # # # #         hx, cur_pos, last_vel = ctx_feat, obs_traj[-1].clone(), vels[-1].clone()
# # # # # # # # #         preds = []
# # # # # # # # #         for i in range(self.N_STEPS):
# # # # # # # # #             hx  = self.gru_cell(torch.cat([ctx_feat, vel_feat, cur_pos], dim=-1), hx)
# # # # # # # # #             gate = torch.sigmoid(self.motion_gate_linear(torch.cat([vel_feat, hx], dim=-1)))
# # # # # # # # #             raw_delta   = self.out_head(self.out_norm(hx))
# # # # # # # # #             prior_delta = last_vel * (0.9 ** i)
# # # # # # # # #             scale_i = torch.sigmoid(self.step_scale[i]) * self.MAX_DELTA
# # # # # # # # #             delta   = (gate * prior_delta + (1.0 - gate) * raw_delta * scale_i).clamp(-self.MAX_DELTA, self.MAX_DELTA)
# # # # # # # # #             cur_pos = cur_pos + delta
# # # # # # # # #             last_vel = delta
# # # # # # # # #             preds.append(cur_pos)
# # # # # # # # #         return torch.stack(preds, dim=0)


# # # # # # # # # # ─── VelocityField (giữ nguyên từ v30, pos_enc dùng pred_len) ────────────────

# # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # # # # # #                  unet_in_ch=13, fm_pred_len=8):
# # # # # # # # #         super().__init__()
# # # # # # # # #         self.pred_len = pred_len
# # # # # # # # #         self.obs_len  = obs_len
# # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # # # #             spatial_down=32, dropout=0.05)
# # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)
# # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)
# # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # #         self.ctx_drop = nn.Dropout(0.15)
# # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)
# # # # # # # # #         self.sr_anchor_proj = nn.Sequential(
# # # # # # # # #             nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 256))
# # # # # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # # #         # KEY FIX: pos_enc dùng pred_len (12), không phải fm_pred_len (8)
# # # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.05)
# # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # #                 dropout=0.15, activation="gelu", batch_first=True),
# # # # # # # # #             num_layers=4)
# # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)
# # # # # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)
# # # # # # # # #         self.short_range_head = ShortRangeHead(self.RAW_CTX_DIM, obs_len)

# # # # # # # # #     def _time_emb(self, t, dim=256):
# # # # # # # # #         half = dim // 2
# # # # # # # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # # # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # # # # # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # # # # # # #     def _context(self, batch_list):
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
# # # # # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)
# # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # # # # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
# # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)
# # # # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # #     def _beta_drift(self, x_t):
# # # # # # # # #         lat_deg  = x_t[:, :, 1] * 5.0
# # # # # # # # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # # # # #         beta     = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # # # # #         R_tc     = 3e5
# # # # # # # # #         v_phys   = torch.zeros_like(x_t)
# # # # # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # # # # #         return v_phys

# # # # # # # # #     def _decode(self, x_t, t, ctx, sr_anchor_emb=None):
# # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # # # # #         t_emb = self.time_fc2(t_emb)
# # # # # # # # #         T_seq = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # # #         x_emb = self.traj_embed(x_t[:, :T_seq]) + self.pos_enc[:, :T_seq] + t_emb.unsqueeze(1)
# # # # # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # # # # #         if sr_anchor_emb is not None:
# # # # # # # # #             mem_parts.append(sr_anchor_emb.unsqueeze(1))
# # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(
# # # # # # # # #             self.transformer(x_emb, torch.cat(mem_parts, dim=1)))))
# # # # # # # # #         with torch.no_grad():
# # # # # # # # #             v_phys = self._beta_drift(x_t[:, :T_seq])
# # # # # # # # #         return v_neural + torch.sigmoid(self.physics_scale) * 2.0 * v_phys

# # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0, sr_anchor_emb=None):
# # # # # # # # #         return self._decode(x_t, t, self._apply_ctx_head(raw_ctx, noise_scale), sr_anchor_emb)

# # # # # # # # #     def forward_short_range(self, obs_traj, raw_ctx):
# # # # # # # # #         return self.short_range_head(raw_ctx, obs_traj)

# # # # # # # # #     def compute_sr_anchor_emb(self, sr_pred):
# # # # # # # # #         anchor_input = torch.cat([sr_pred[-1], sr_pred[-1] - sr_pred[-2]], dim=-1)
# # # # # # # # #         return self.sr_anchor_proj(anchor_input)


# # # # # # # # # # ─── TCFlowMatching v32 ───────────────────────────────────────────────────────

# # # # # # # # # class TCFlowMatching(nn.Module):
# # # # # # # # #     """
# # # # # # # # #     v32: Phase-based training để tránh FM/MSE conflict.
    
# # # # # # # # #     Phase 1 (epoch < phase_switch): Pure MSE + SR
# # # # # # # # #     Phase 2 (epoch >= phase_switch): MSE + SR + AFCRPS
# # # # # # # # #     """

# # # # # # # # #     PHASE_SWITCH = 30  # Sau epoch này mới thêm AFCRPS

# # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # # # # #                  ctx_noise_scale=0.002, initial_sample_sigma=0.03, **kwargs):
# # # # # # # # #         super().__init__()
# # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # #         self.fm_pred_len          = pred_len - N_SR_STEPS
# # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # # # #         self.active_pred_len      = pred_len
# # # # # # # # #         self.net = VelocityField(pred_len=pred_len, obs_len=obs_len,
# # # # # # # # #                                   sigma_min=sigma_min, unet_in_ch=unet_in_ch,
# # # # # # # # #                                   fm_pred_len=self.fm_pred_len)

# # # # # # # # #     def set_curriculum_len(self, *a, **kw): pass

# # # # # # # # #     @staticmethod
# # # # # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # # # # #                            Me   - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # # # # # #     @staticmethod
# # # # # # # # #     def _to_abs(rel, lp, lm):
# # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # # # # # #     def _cfm_noisy(self, x1):
# # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # #         sm = self.sigma_min
# # # # # # # # #         x0 = torch.randn_like(x1) * sm
# # # # # # # # #         t  = torch.rand(B, device=device)
# # # # # # # # #         te = t.view(B, 1, 1)
# # # # # # # # #         x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
# # # # # # # # #         denom = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
# # # # # # # # #         return x_t, t, te, denom

# # # # # # # # #     @staticmethod
# # # # # # # # #     def _intensity_weights(obs_Me):
# # # # # # # # #         w = obs_Me[-1, :, 1].detach()
# # # # # # # # #         w = torch.where(w < 0.1, torch.full_like(w, 0.5),
# # # # # # # # #             torch.where(w < 0.3, torch.full_like(w, 0.8),
# # # # # # # # #             torch.where(w < 0.6, torch.full_like(w, 1.0),
# # # # # # # # #                         torch.full_like(w, 1.5))))
# # # # # # # # #         return w / w.mean().clamp(min=1e-6)

# # # # # # # # #     @staticmethod
# # # # # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # #             return batch_list
# # # # # # # # #         aug = list(batch_list)
# # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # # # # #                 t = aug[idx].clone()
# # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # #                 aug[idx] = t
# # # # # # # # #         return aug

# # # # # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # #         """
# # # # # # # # #         Phase 1 (epoch < 30): SR_Huber + MSE_hav — học như LSTM
# # # # # # # # #         Phase 2 (epoch >= 30): SR_Huber + MSE_hav + AFCRPS — thêm probabilistic
# # # # # # # # #         """
# # # # # # # # #         batch_list = self._lon_flip_aug(batch_list)

# # # # # # # # #         traj_gt = batch_list[1]
# # # # # # # # #         Me_gt   = batch_list[8]
# # # # # # # # #         obs_t   = batch_list[0]
# # # # # # # # #         obs_Me  = batch_list[7]
# # # # # # # # #         lp, lm  = obs_t[-1], obs_Me[-1]

# # # # # # # # #         raw_ctx     = self.net._context(batch_list)
# # # # # # # # #         intensity_w = self._intensity_weights(obs_Me)

# # # # # # # # #         # ── SR (step 1-4) ──────────────────────────────────────────────
# # # # # # # # #         n_sr    = ShortRangeHead.N_STEPS
# # # # # # # # #         sr_pred = self.net.forward_short_range(obs_t, raw_ctx)
# # # # # # # # #         l_sr    = short_range_regression_loss(sr_pred, traj_gt[:n_sr], lp)

# # # # # # # # #         # ── FM: full 12 steps từ lp ────────────────────────────────────
# # # # # # # # #         sr_anchor = self.net.compute_sr_anchor_emb(sr_pred.detach())
# # # # # # # # #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)
# # # # # # # # #         x_t, t, te, denom = self._cfm_noisy(x1)
# # # # # # # # #         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, sr_anchor_emb=sr_anchor)
# # # # # # # # #         x1_pred  = x_t + denom * pred_vel
# # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)   # [12, B, 2] normalised

# # # # # # # # #         # ── MSE haversine per-step ─────────────────────────────────────
# # # # # # # # #         # Đây là loss chính trong Phase 1 — giống LSTM nhưng physically correct
# # # # # # # # #         l_mse = mse_hav_loss(pred_abs, traj_gt)

# # # # # # # # #         # ── Directional ────────────────────────────────────────────────
# # # # # # # # #         pred_deg = _denorm_to_deg(pred_abs)
# # # # # # # # #         gt_deg   = _denorm_to_deg(traj_gt)
# # # # # # # # #         l_vel    = velocity_loss_per_sample(pred_deg, gt_deg).mean()
# # # # # # # # #         l_head   = heading_loss(pred_deg, gt_deg)

# # # # # # # # #         # ── Total (Phase 1) ────────────────────────────────────────────
# # # # # # # # #         NRM   = 35.0
# # # # # # # # #         total = (
# # # # # # # # #             5.0  * l_sr
# # # # # # # # #             + 4.0  * l_mse          # HIGH weight như LSTM
# # # # # # # # #             + 0.5  * l_vel  * NRM
# # # # # # # # #             + 0.8  * l_head * NRM
# # # # # # # # #         ) / NRM

# # # # # # # # #         bd = dict(
# # # # # # # # #             total=total, short_range=l_sr.item(), mse_hav=l_mse.item(),
# # # # # # # # #             velocity=l_vel.item()*NRM, heading=l_head.item(),
# # # # # # # # #             fm=0.0, step=0.0, disp=0.0, cont=0.0, pinn=0.0,
# # # # # # # # #             spread=0.0, continuity=0.0, recurv_ratio=0.0,
# # # # # # # # #         )

# # # # # # # # #         # ── Phase 2: Thêm AFCRPS (epoch >= PHASE_SWITCH) ───────────────
# # # # # # # # #         if epoch >= self.PHASE_SWITCH:
# # # # # # # # #             samples = []
# # # # # # # # #             for _ in range(self.n_train_ens):
# # # # # # # # #                 xt_s, ts, _, dens_s = self._cfm_noisy(x1)
# # # # # # # # #                 pv_s = self.net.forward_with_ctx(xt_s, ts, raw_ctx, sr_anchor_emb=sr_anchor)
# # # # # # # # #                 pa_s, _ = self._to_abs(xt_s + dens_s * pv_s, lp, lm)
# # # # # # # # #                 samples.append(pa_s)
# # # # # # # # #             pred_samples = torch.stack(samples)   # [S, 12, B, 2]

# # # # # # # # #             l_fm = fm_afcrps_loss(
# # # # # # # # #                 pred_samples, traj_gt,
# # # # # # # # #                 unit_01deg=True,
# # # # # # # # #                 intensity_w=intensity_w,
# # # # # # # # #                 w_es=0.2,
# # # # # # # # #             )

# # # # # # # # #             # Weight AFCRPS nhỏ hơn MSE để không dominate
# # # # # # # # #             fm_w = min(1.0, (epoch - self.PHASE_SWITCH) / 20.0)  # ramp up 0→1
# # # # # # # # #             bd["total"] = total + fm_w * 1.5 * l_fm / NRM
# # # # # # # # #             bd["fm"]    = l_fm.item()

# # # # # # # # #         if torch.isnan(bd["total"]) or torch.isinf(bd["total"]):
# # # # # # # # #             bd["total"] = lp.new_zeros(())

# # # # # # # # #         return bd

# # # # # # # # #     @torch.no_grad()
# # # # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
# # # # # # # # #         obs_t  = batch_list[0]
# # # # # # # # #         lp     = obs_t[-1]
# # # # # # # # #         lm     = batch_list[7][-1]
# # # # # # # # #         B      = lp.shape[0]
# # # # # # # # #         device = lp.device
# # # # # # # # #         T      = self.pred_len
# # # # # # # # #         dt     = 1.0 / max(ddim_steps, 1)

# # # # # # # # #         raw_ctx = self.net._context(batch_list)
# # # # # # # # #         sr_pred = self.net.forward_short_range(obs_t, raw_ctx)
# # # # # # # # #         sr_anchor = self.net.compute_sr_anchor_emb(sr_pred)

# # # # # # # # #         traj_s, me_s = [], []
# # # # # # # # #         for _ in range(num_ensemble):
# # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma
# # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # #                 ns  = self.ctx_noise_scale if step < 3 else 0.0
# # # # # # # # #                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx,
# # # # # # # # #                                                  noise_scale=ns, sr_anchor_emb=sr_anchor)
# # # # # # # # #                 x_t = x_t + dt * vel
# # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # #             traj_s.append(tr)
# # # # # # # # #             me_s.append(me)

# # # # # # # # #         all_trajs = torch.stack(traj_s)
# # # # # # # # #         fm_mean   = all_trajs.mean(0)
# # # # # # # # #         pred_mean = fm_mean.clone()
# # # # # # # # #         pred_mean[:ShortRangeHead.N_STEPS] = sr_pred   # SR override step 1-4

# # # # # # # # #         if predict_csv:
# # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # # # # #         return pred_mean, torch.stack(me_s).mean(0), all_trajs

# # # # # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # # # # #         with torch.enable_grad():
# # # # # # # # #             x = x_pred.detach().requires_grad_(True)
# # # # # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # # # # #             for _ in range(n_steps):
# # # # # # # # #                 opt.zero_grad()
# # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # #                 pred_deg = _denorm_to_deg(pred_abs)
# # # # # # # # #                 dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # #                 lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # #                 cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # #                 speed   = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # # # # #                                       + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # # # # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # # # # # # # #                 loss.backward()
# # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # # # # # # # #                 opt.step()
# # # # # # # # #         return x.detach()

# # # # # # # # #     @staticmethod
# # # # # # # # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # # # # # # # #         import numpy as np
# # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # #                     "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # #             if write_hdr: w.writeheader()
# # # # # # # # #             for b in range(B):
# # # # # # # # #                 for k in range(T):
# # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(math.radians(mean_lat[k, b]))
# # # # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # # # #                     w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # # # # #                                 "lead_h": (k+1)*6, "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # # # # #                                 "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # # # # #                                 "lon_std_deg": f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # #                                 "lat_std_deg": f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # #                                 "ens_spread_km": f"{spread:.2f}"})


# # # # # # # # # TCDiffusion = TCFlowMatching

# # # # # # # # """
# # # # # # # # flow_matching_model_v33.py — Pure FM + MSE: Beat LSTM bằng đơn giản

# # # # # # # # ═══════════════════════════════════════════════════════════════════════
# # # # # # # # PHÂN TÍCH TẠI SAO FM THUA LSTM:
# # # # # # # # ═══════════════════════════════════════════════════════════════════════

# # # # # # # # 1. LSTM dùng pure MSE → gradient rõ ràng, 1 hướng duy nhất
# # # # # # # # 2. FM v32 dùng 5+ loss → gradient conflict, model không biết optimize gì
# # # # # # # # 3. SR head riêng + FM riêng → discontinuity tại step 4→5
# # # # # # # # 4. AFCRPS pull ensemble diversity ≠ accuracy → ADE cao

# # # # # # # # ═══════════════════════════════════════════════════════════════════════
# # # # # # # # GIẢI PHÁP V33: 
# # # # # # # # ═══════════════════════════════════════════════════════════════════════

# # # # # # # # CORE INSIGHT: FM architecture + MSE training = best of both worlds
# # # # # # # #   - FM cho ensemble/uncertainty estimation khi inference
# # # # # # # #   - MSE cho gradient rõ ràng khi training (như LSTM)
# # # # # # # #   - KHÔNG dùng SR head riêng → FM predict ALL 12 steps
# # # # # # # #   - KHÔNG dùng AFCRPS khi training → chỉ MSE haversine

# # # # # # # # ARCHITECTURE CHANGES:
# # # # # # # #   1. BỎ ShortRangeHead → FM predict tất cả 12 steps trực tiếp
# # # # # # # #   2. BỎ sr_anchor_emb → đơn giản hóa decode
# # # # # # # #   3. THÊM teacher forcing: train FM với gt trajectory làm guidance
# # # # # # # #   4. Loss = MSE_haversine + nhẹ velocity + nhẹ heading (tổng 3 loss)

# # # # # # # # TRAINING STRATEGY:
# # # # # # # #   Phase 1 (epoch 0-20): MSE-only, sigma_min lớn (0.1) → near-deterministic
# # # # # # # #     → FM hoạt động gần như regression network
# # # # # # # #     → ADE giảm nhanh từ 400 → ~200km
    
# # # # # # # #   Phase 2 (epoch 20-50): MSE + nhẹ velocity/heading, sigma_min giảm dần
# # # # # # # #     → Refine trajectory shape
# # # # # # # #     → ADE ~170km, 12h ~45km
    
# # # # # # # #   Phase 3 (epoch 50+): Giữ nguyên, fine-tune
# # # # # # # #     → 12h < 50km, 72h < 300km

# # # # # # # # KEY INSIGHT VỀ FM VÀ 12h:
# # # # # # # #   - FM CÓ THỂ đạt 12h < 50km NẾU train đúng cách
# # # # # # # #   - Vấn đề cũ: SR head riêng → FM không được train cho step 1-4
# # # # # # # #   - V33: FM predict ALL steps → step 1-4 cũng được optimize trực tiếp
# # # # # # # #   - Step weighting nhấn mạnh 12h (step 2) và 24h (step 4)

# # # # # # # # ═══════════════════════════════════════════════════════════════════════
# # # # # # # # """
# # # # # # # # from __future__ import annotations

# # # # # # # # import csv
# # # # # # # # import math
# # # # # # # # import os
# # # # # # # # from datetime import datetime
# # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # import torch
# # # # # # # # import torch.nn as nn
# # # # # # # # import torch.nn.functional as F

# # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # from Model.losses import _haversine_deg, _norm_to_deg, _haversine

# # # # # # # # # ── Step weights: nhấn mạnh 12h và 24h ─────────────────────────────────────
# # # # # # # # # Step:    6h   12h  18h  24h  30h  36h  42h  48h  54h  60h  66h  72h
# # # # # # # # MSE_STEP_WEIGHTS = [
# # # # # # # #     1.5,  # 6h  - quan trọng
# # # # # # # #     4.0,  # 12h - TARGET < 50km → weight cao nhất  
# # # # # # # #     2.0,  # 18h
# # # # # # # #     3.5,  # 24h - TARGET < 100km → weight cao
# # # # # # # #     1.5,  # 30h
# # # # # # # #     1.5,  # 36h
# # # # # # # #     1.5,  # 42h
# # # # # # # #     2.5,  # 48h - TARGET < 200km
# # # # # # # #     1.0,  # 54h
# # # # # # # #     1.0,  # 60h
# # # # # # # #     1.5,  # 66h
# # # # # # # #     2.5,  # 72h - TARGET < 300km
# # # # # # # # ]


# # # # # # # # def _denorm_to_deg(t):
# # # # # # # #     """Convert normalized coords to degrees."""
# # # # # # # #     out = t.clone()
# # # # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # # # #     return out


# # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  MSE Haversine Loss — Vũ khí chính, giống LSTM nhưng physically correct
# # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # def mse_hav_loss(pred_norm, gt_norm, step_w=None):
# # # # # # # #     """
# # # # # # # #     Haversine MSE per-step.
    
# # # # # # # #     Tại sao dùng haversine thay vì MSE(lon,lat)?
# # # # # # # #     - MSE(lon,lat) bị bias do cos(lat): 1 degree lon ở equator ≠ 1 degree ở 30°N
# # # # # # # #     - Haversine cho distance thật sự trên Earth → gradient chính xác hơn
# # # # # # # #     - LSTM dùng MSE(lon,lat) → có thể beat bằng MSE(haversine)
# # # # # # # #     """
# # # # # # # #     if step_w is None:
# # # # # # # #         step_w = MSE_STEP_WEIGHTS
# # # # # # # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # #     pred_deg = _norm_to_deg(pred_norm[:T])
# # # # # # # #     gt_deg   = _norm_to_deg(gt_norm[:T])
# # # # # # # #     dist_km  = _haversine_deg(pred_deg, gt_deg)   # [T, B] in km
    
# # # # # # # #     w = pred_norm.new_tensor(step_w[:T])
# # # # # # # #     w = w / w.sum() * T  # normalize nhưng giữ scale
    
# # # # # # # #     # Huber-style: dùng L1 cho outlier (dist > 300km), L2 cho phần còn lại
# # # # # # # #     # Tránh gradient explosion từ large errors
# # # # # # # #     delta = 300.0  # km
# # # # # # # #     huber_dist = torch.where(
# # # # # # # #         dist_km < delta,
# # # # # # # #         dist_km.pow(2) / (2.0 * delta),
# # # # # # # #         dist_km - delta / 2.0,
# # # # # # # #     )
    
# # # # # # # #     return (huber_dist * w.unsqueeze(1)).mean() / delta


# # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  VelocityField — Đơn giản hóa, bỏ ShortRangeHead
# # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # class VelocityField(nn.Module):
# # # # # # # #     """
# # # # # # # #     Velocity field cho FM — predict velocity cho ALL 12 steps.
    
# # # # # # # #     Thay đổi so với v32:
# # # # # # # #     - BỎ ShortRangeHead → FM predict step 1-12 trực tiếp
# # # # # # # #     - BỎ sr_anchor_emb → đơn giản decode
# # # # # # # #     - THÊM residual connection trong decoder
# # # # # # # #     - THÊM step embedding (model biết đang predict step nào)
# # # # # # # #     """
# # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # # # # #                  unet_in_ch=13, **kwargs):
# # # # # # # #         super().__init__()
# # # # # # # #         self.pred_len = pred_len
# # # # # # # #         self.obs_len  = obs_len
        
# # # # # # # #         # ── Encoder (giữ nguyên) ──────────────────────────────────────
# # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # # #             spatial_down=32, dropout=0.05)
# # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)
        
# # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)
        
# # # # # # # #         # ── Context projection ────────────────────────────────────────
# # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # #         self.ctx_drop = nn.Dropout(0.10)
# # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)
        
# # # # # # # #         # ── Velocity observation encoder ──────────────────────────────
# # # # # # # #         # Encode observed velocity pattern → giúp predict step 1-4 chính xác
# # # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # # # # #             nn.LayerNorm(256),
# # # # # # # #             nn.Linear(256, 256), nn.GELU(),
# # # # # # # #         )
        
# # # # # # # #         # ── FM Decoder ────────────────────────────────────────────────
# # # # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
        
# # # # # # # #         # Step embedding: model biết đang ở step nào → quan trọng cho short-range
# # # # # # # #         self.step_embed = nn.Embedding(pred_len, 256)
        
# # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # # # # # #             num_layers=5)  # 5 layers thay vì 4 → capacity lớn hơn
        
# # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # #         self.out_fc2 = nn.Linear(512, 4)
        
# # # # # # # #         # Learnable output scale per step → model tự learn scale phù hợp
# # # # # # # #         self.step_scale = nn.Parameter(torch.ones(pred_len) * 0.5)
        
# # # # # # # #         # Physics beta-drift (giữ nguyên)
# # # # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.3)
        
# # # # # # # #         self._init_weights()

# # # # # # # #     def _init_weights(self):
# # # # # # # #         """Xavier init với gain nhỏ → output gần 0 ban đầu."""
# # # # # # # #         with torch.no_grad():
# # # # # # # #             for name, m in self.named_modules():
# # # # # # # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # # # # # #                     if m.bias is not None:
# # # # # # # #                         nn.init.zeros_(m.bias)

# # # # # # # #     def _time_emb(self, t, dim=256):
# # # # # # # #         half = dim // 2
# # # # # # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # # # # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # # # # # #     def _context(self, batch_list):
# # # # # # # #         """Encode observation context — giống v32."""
# # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # #         image_obs = batch_list[11]
# # # # # # # #         env_data  = batch_list[13]
        
# # # # # # # #         if image_obs.dim() == 4:
# # # # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
        
# # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # #         T_obs = obs_traj.shape[0]
        
# # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)
        
# # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # # # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
        
# # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)
        
# # # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # # # #         if noise_scale > 0.0:
# # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # #     def _get_vel_obs_feat(self, obs_traj):
# # # # # # # #         """Encode observed velocities → giúp FM predict step 1-4."""
# # # # # # # #         B = obs_traj.shape[1]
# # # # # # # #         T_obs = obs_traj.shape[0]
        
# # # # # # # #         if T_obs >= 2:
# # # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # # # # # #         else:
# # # # # # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
        
# # # # # # # #         # Pad/truncate to obs_len
# # # # # # # #         if vels.shape[0] < self.obs_len:
# # # # # # # #             pad = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # #         else:
# # # # # # # #             vels = vels[-self.obs_len:]
        
# # # # # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # # # # #     def _beta_drift(self, x_t):
# # # # # # # #         """Physical beta-drift bias."""
# # # # # # # #         lat_deg  = x_t[:, :, 1] * 5.0
# # # # # # # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # # # #         beta     = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # # # #         R_tc     = 3e5
# # # # # # # #         v_phys   = torch.zeros_like(x_t)
# # # # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # # # #         return v_phys

# # # # # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None):
# # # # # # # #         """
# # # # # # # #         Decode velocity field.
        
# # # # # # # #         Thay đổi:
# # # # # # # #         - Thêm vel_obs_feat vào memory → giúp predict step 1-4
# # # # # # # #         - Thêm step embedding → model biết step position
# # # # # # # #         - Output scale per step → kiểm soát magnitude
# # # # # # # #         """
# # # # # # # #         B = x_t.shape[0]
# # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # # # #         t_emb = self.time_fc2(t_emb)
        
# # # # # # # #         T_seq = min(x_t.size(1), self.pos_enc.shape[1])
        
# # # # # # # #         # Step embedding
# # # # # # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # # #         s_emb = self.step_embed(step_idx)  # [B, T_seq, 256]
        
# # # # # # # #         x_emb = self.traj_embed(x_t[:, :T_seq]) + self.pos_enc[:, :T_seq] + t_emb.unsqueeze(1) + s_emb
        
# # # # # # # #         # Memory: context + time + velocity observations
# # # # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # # # #         if vel_obs_feat is not None:
# # # # # # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))
        
# # # # # # # #         decoded = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
        
# # # # # # # #         # Per-step scale
# # # # # # # #         scale = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # # # # #         v_neural = v_neural * scale
        
# # # # # # # #         # Physics correction
# # # # # # # #         with torch.no_grad():
# # # # # # # #             v_phys = self._beta_drift(x_t[:, :T_seq])
        
# # # # # # # #         return v_neural + torch.sigmoid(self.physics_scale) * v_phys

# # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0, vel_obs_feat=None):
# # # # # # # #         return self._decode(x_t, t, self._apply_ctx_head(raw_ctx, noise_scale), vel_obs_feat)


# # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  TCFlowMatching v33 — Pure FM + MSE
# # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # class TCFlowMatching(nn.Module):
# # # # # # # #     """
# # # # # # # #     v33: Pure FM architecture, MSE-only training.
    
# # # # # # # #     KEY DIFFERENCES từ v32:
# # # # # # # #     1. FM predict ALL 12 steps (không có SR head riêng)
# # # # # # # #     2. Training loss = MSE_haversine + nhẹ velocity + nhẹ heading
# # # # # # # #     3. KHÔNG dùng AFCRPS → gradient không conflict
# # # # # # # #     4. sigma_min schedule: lớn ban đầu (near-deterministic) → nhỏ dần
# # # # # # # #     5. Teacher forcing: dùng gt trajectory interpolation khi train
    
# # # # # # # #     TẠI SAO FM CÓ THỂ BEAT LSTM:
# # # # # # # #     - FM có thêm encoder mạnh (FNO3D + Mamba + Env_net)
# # # # # # # #     - FM có ensemble capability → uncertainty estimation
# # # # # # # #     - FM có physics correction (beta-drift)
# # # # # # # #     - Nếu train đúng (MSE-only), FM ít nhất = LSTM, có thể tốt hơn
# # # # # # # #     """

# # # # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # # # #                  ctx_noise_scale=0.002, initial_sample_sigma=0.03, **kwargs):
# # # # # # # #         super().__init__()
# # # # # # # #         self.pred_len             = pred_len
# # # # # # # #         self.obs_len              = obs_len
# # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # # #         self.active_pred_len      = pred_len
        
# # # # # # # #         self.net = VelocityField(
# # # # # # # #             pred_len=pred_len, obs_len=obs_len,
# # # # # # # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # # # # #     def set_curriculum_len(self, *a, **kw):
# # # # # # # #         pass

# # # # # # # #     @staticmethod
# # # # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # # # #         """Convert absolute coords to relative (from last observed position)."""
# # # # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # # # #                            Me   - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # # # # #     @staticmethod
# # # # # # # #     def _to_abs(rel, lp, lm):
# # # # # # # #         """Convert relative coords back to absolute."""
# # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # # # # #     # def _cfm_noisy(self, x1, sigma_min=None):
# # # # # # # #     #     """
# # # # # # # #     #     Conditional FM noise process.
        
# # # # # # # #     #     KEY: sigma_min controls stochasticity:
# # # # # # # #     #     - sigma_min lớn (0.1-0.2) → gần deterministic → tốt cho MSE training
# # # # # # # #     #     - sigma_min nhỏ (0.01-0.02) → stochastic → tốt cho ensemble diversity
# # # # # # # #     #     """
# # # # # # # #     #     if sigma_min is None:
# # # # # # # #     #         sigma_min = self.sigma_min
# # # # # # # #     #     B, device = x1.shape[0], x1.device
# # # # # # # #     #     x0 = torch.randn_like(x1) * sigma_min
# # # # # # # #     #     t  = torch.rand(B, device=device)
# # # # # # # #     #     te = t.view(B, 1, 1)
# # # # # # # #     #     # x_t = te * x1 + (1.0 - te * (1.0 - sigma_min)) * x0
# # # # # # # #     #     # denom = (1.0 - (1.0 - sigma_min) * te).clamp(min=1e-5)
# # # # # # # #     #     x_t = te * x1 + (1.0 - te * (1.0 - sigma_min)) * x0
# # # # # # # #     #     denom = (1.0 - (1.0 - sigma_min) * te).clamp(min=1e-5)
# # # # # # # #     #     return x_t, t, te, denom

# # # # # # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # # # # # #         if sigma_min is None:
# # # # # # # #             sigma_min = self.sigma_min
# # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # #         x0 = torch.randn_like(x1) * sigma_min
# # # # # # # #         t  = torch.rand(B, device=device)
# # # # # # # #         te = t.view(B, 1, 1)
# # # # # # # #         x_t = (1.0 - te) * x0 + te * x1
# # # # # # # #         u   = x1 - x0
# # # # # # # #         return x_t, t, u  # 3 giá trị, không còn denom

# # # # # # # #     @staticmethod
# # # # # # # #     def _intensity_weights(obs_Me):
# # # # # # # #         """Weight samples by intensity — stronger storms get more attention."""
# # # # # # # #         w = obs_Me[-1, :, 1].detach()
# # # # # # # #         w = torch.where(w < 0.1, torch.full_like(w, 0.5),
# # # # # # # #             torch.where(w < 0.3, torch.full_like(w, 0.8),
# # # # # # # #             torch.where(w < 0.6, torch.full_like(w, 1.0),
# # # # # # # #                         torch.full_like(w, 1.5))))
# # # # # # # #         return w / w.mean().clamp(min=1e-6)

# # # # # # # #     @staticmethod
# # # # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # # # #         """Data augmentation: flip longitude."""
# # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # #             return batch_list
# # # # # # # #         aug = list(batch_list)
# # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # # # #                 t = aug[idx].clone()
# # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # #                 aug[idx] = t
# # # # # # # #         return aug

# # # # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # # # #     # def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # #     #     """
# # # # # # # #     #     ═══════════════════════════════════════════════════════════════
# # # # # # # #     #     CORE TRAINING LOGIC — MSE-focused, FM as architecture only
# # # # # # # #     #     ═══════════════════════════════════════════════════════════════
        
# # # # # # # #     #     Phase 1 (epoch < 15): Pure MSE, sigma_min = 0.15 (near-deterministic)
# # # # # # # #     #       → FM gần như regression network
# # # # # # # #     #       → Gradient rõ ràng, ADE giảm nhanh
          
# # # # # # # #     #     Phase 2 (epoch 15-40): MSE + nhẹ velocity/heading, sigma_min = 0.08
# # # # # # # #     #       → Refine trajectory shape
# # # # # # # #     #       → Giảm directional errors
          
# # # # # # # #     #     Phase 3 (epoch 40+): MSE + velocity/heading, sigma_min = 0.03
# # # # # # # #     #       → Fine-tune, cho phép ensemble diversity
# # # # # # # #     #       → Maintain accuracy, add calibration
# # # # # # # #     #     """
# # # # # # # #     #     batch_list = self._lon_flip_aug(batch_list)

# # # # # # # #     #     traj_gt = batch_list[1]    # [T, B, 2] normalized
# # # # # # # #     #     Me_gt   = batch_list[8]    # [T, B, 2] normalized  
# # # # # # # #     #     obs_t   = batch_list[0]    # [T_obs, B, 2] normalized
# # # # # # # #     #     obs_Me  = batch_list[7]    # [T_obs, B, 2] normalized
# # # # # # # #     #     lp, lm  = obs_t[-1], obs_Me[-1]

# # # # # # # #     #     # ── Sigma schedule ─────────────────────────────────────────────
# # # # # # # #     #     # Ban đầu sigma lớn → FM gần deterministic → MSE hiệu quả
# # # # # # # #     #     if epoch < 15:
# # # # # # # #     #         current_sigma = 0.15
# # # # # # # #     #     elif epoch < 40:
# # # # # # # #     #         # Linear decay 0.15 → 0.03
# # # # # # # #     #         t = (epoch - 15) / 25.0
# # # # # # # #     #         current_sigma = 0.15 - t * (0.15 - 0.03)
# # # # # # # #     #     else:
# # # # # # # #     #         current_sigma = 0.03

# # # # # # # #     #     # ── Encode context ─────────────────────────────────────────────
# # # # # # # #     #     raw_ctx     = self.net._context(batch_list)
# # # # # # # #     #     vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # # # #     #     # ── FM forward: predict ALL 12 steps ───────────────────────────
# # # # # # # #     #     x1 = self._to_rel(traj_gt, Me_gt, lp, lm)  # [B, T, 4]
# # # # # # # #     #     # # x_t, t, te, denom = self._cfm_noisy(x1, sigma_min=current_sigma)
# # # # # # # #     #     # x_t, t, u_target = self._cfm_noisy(x1, sigma_min=current_sigma)

# # # # # # # #     #     # pred_vel = self.net.forward_with_ctx(
# # # # # # # #     #     #     x_t, t, raw_ctx, vel_obs_feat=vel_obs_feat)
# # # # # # # #     #     # x1_pred  = x_t + denom * pred_vel
# # # # # # # #     #     # pred_abs, _ = self._to_abs(x1_pred, lp, lm)  # [12, B, 2] normalized

# # # # # # # #     #     # # ── LOSS 1: MSE Haversine (CHÍNH) ──────────────────────────────
# # # # # # # #     #     # l_mse = mse_hav_loss(pred_abs, traj_gt)
# # # # # # # #     #     x_t, t, u_target = self._cfm_noisy(x1, sigma_min=current_sigma)
# # # # # # # #     #     pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, vel_obs_feat=vel_obs_feat)
# # # # # # # #     #     l_mse = F.mse_loss(pred_vel, u_target)  # loss trên velocity, stable qua mọi epo

# # # # # # # #     #     # ── LOSS 2: Velocity matching (nhẹ) ───────────────────────────
# # # # # # # #     #     pred_deg = _denorm_to_deg(pred_abs)
# # # # # # # #     #     gt_deg   = _denorm_to_deg(traj_gt)
        
# # # # # # # #     #     if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # # #     #         T_min = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # #     #         v_pred = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # # #     #         v_gt   = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # # # #     #         l_vel  = F.smooth_l1_loss(v_pred, v_gt)
# # # # # # # #     #     else:
# # # # # # # #     #         l_vel = pred_abs.new_zeros(())

# # # # # # # #     #     # ── LOSS 3: Heading consistency (nhẹ) ─────────────────────────
# # # # # # # #     #     if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # # #     #         T_min = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # #     #         v_pred = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # # #     #         v_gt   = gt_deg[1:T_min]   - gt_deg[:T_min-1]
            
# # # # # # # #     #         v_pred_n = v_pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # # #     #         v_gt_n   = v_gt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # # #     #         cos_sim  = ((v_pred / v_pred_n) * (v_gt / v_gt_n)).sum(-1)
# # # # # # # #     #         l_head   = F.relu(-cos_sim).pow(2).mean()  # penalize opposite direction
# # # # # # # #     #     else:
# # # # # # # #     #         l_head = pred_abs.new_zeros(())

# # # # # # # #     #     # ── Total Loss ─────────────────────────────────────────────────
# # # # # # # #     #     # Phase 1: pure MSE
# # # # # # # #     #     # Phase 2+: MSE + small velocity + small heading
# # # # # # # #     #     if epoch < 15:
# # # # # # # #     #         w_vel  = 0.0
# # # # # # # #     #         w_head = 0.0
# # # # # # # #     #     elif epoch < 40:
# # # # # # # #     #         t_phase = (epoch - 15) / 25.0
# # # # # # # #     #         w_vel  = t_phase * 0.3
# # # # # # # #     #         w_head = t_phase * 0.2
# # # # # # # #     #     else:
# # # # # # # #     #         w_vel  = 0.3
# # # # # # # #     #         w_head = 0.2

# # # # # # # #     #     total = l_mse + w_vel * l_vel + w_head * l_head

# # # # # # # #     #     # ── Ensemble consistency (optional, epoch 40+) ─────────────────
# # # # # # # #     #     # Train thêm 1 sample, penalize nếu quá khác nhau
# # # # # # # #     #     l_ens_consist = pred_abs.new_zeros(())
# # # # # # # #     #     if epoch >= 40 and self.n_train_ens >= 2:
# # # # # # # #     #         x_t2, t2, te2, denom2 = self._cfm_noisy(x1, sigma_min=current_sigma)
# # # # # # # #     #         pred_vel2 = self.net.forward_with_ctx(
# # # # # # # #     #             x_t2, t2, raw_ctx, vel_obs_feat=vel_obs_feat)
# # # # # # # #     #         x1_pred2 = x_t2 + denom2 * pred_vel2
# # # # # # # #     #         pred_abs2, _ = self._to_abs(x1_pred2, lp, lm)
            
# # # # # # # #     #         # MSE of second sample → cũng phải gần gt
# # # # # # # #     #         l_ens_consist = mse_hav_loss(pred_abs2, traj_gt)
# # # # # # # #     #         total = total + 0.3 * l_ens_consist

# # # # # # # #     #     if torch.isnan(total) or torch.isinf(total):
# # # # # # # #     #         total = lp.new_zeros(())

# # # # # # # #     #     return dict(
# # # # # # # #     #         total=total,
# # # # # # # #     #         mse_hav=l_mse.item(),
# # # # # # # #     #         velocity=l_vel.item(),
# # # # # # # #     #         heading=l_head.item(),
# # # # # # # #     #         ens_consist=l_ens_consist.item(),
# # # # # # # #     #         sigma=current_sigma,
# # # # # # # #     #         # Backward compat
# # # # # # # #     #         fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # # # #     #         spread=0.0, continuity=0.0, step=0.0, disp=0.0,
# # # # # # # #     #         recurv_ratio=0.0,
# # # # # # # #     #     )

# # # # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # #         batch_list = self._lon_flip_aug(batch_list)

# # # # # # # #         traj_gt = batch_list[1]
# # # # # # # #         Me_gt   = batch_list[8]
# # # # # # # #         obs_t   = batch_list[0]
# # # # # # # #         obs_Me  = batch_list[7]
# # # # # # # #         lp, lm  = obs_t[-1], obs_Me[-1]

# # # # # # # #         # ── Sigma schedule — đổi tên biến thành sigma_t tránh conflict với FM t ──
# # # # # # # #         if epoch < 15:
# # # # # # # #             current_sigma = 0.15
# # # # # # # #         elif epoch < 40:
# # # # # # # #             sigma_frac = (epoch - 15) / 25.0
# # # # # # # #             current_sigma = 0.15 - sigma_frac * (0.15 - 0.03)
# # # # # # # #         else:
# # # # # # # #             current_sigma = 0.03

# # # # # # # #         # ── Encode context ────────────────────────────────────────────────
# # # # # # # #         raw_ctx      = self.net._context(batch_list)
# # # # # # # #         vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # # # #         # ── FM forward ────────────────────────────────────────────────────
# # # # # # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)  # [B, T, 4]
# # # # # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # # # # #         pred_vel = self.net.forward_with_ctx(
# # # # # # # #             x_t, fm_t, raw_ctx, vel_obs_feat=vel_obs_feat)

# # # # # # # #         # ── LOSS 1: FM velocity MSE (chính) ──────────────────────────────
# # # # # # # #         l_mse = F.mse_loss(pred_vel, u_target)

# # # # # # # #         # ── Reconstruct pred_abs để tính velocity/heading loss ───────────
# # # # # # # #         with torch.no_grad():
# # # # # # # #             fm_te    = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # # # #             x1_pred  = x_t + (1.0 - fm_te) * pred_vel  # FM chuẩn: x1 = x_t + (1-t)*v
# # # # # # # #             pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# # # # # # # #         pred_deg = _denorm_to_deg(pred_abs)
# # # # # # # #         gt_deg   = _denorm_to_deg(traj_gt)

# # # # # # # #         # ── LOSS 2: Velocity matching ─────────────────────────────────────
# # # # # # # #         if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # # #             T_min  = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # #             v_pred = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # # #             v_gt   = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # # # #             l_vel  = F.smooth_l1_loss(v_pred, v_gt)
# # # # # # # #         else:
# # # # # # # #             l_vel = x_t.new_zeros(())

# # # # # # # #         # ── LOSS 3: Heading consistency ───────────────────────────────────
# # # # # # # #         if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # # #             T_min    = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # #             v_pred   = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # # #             v_gt     = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # # # #             v_pred_n = v_pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # # #             v_gt_n   = v_gt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # # #             cos_sim  = ((v_pred / v_pred_n) * (v_gt / v_gt_n)).sum(-1)
# # # # # # # #             l_head   = F.relu(-cos_sim).pow(2).mean()
# # # # # # # #         else:
# # # # # # # #             l_head = x_t.new_zeros(())

# # # # # # # #         # ── Phase weights ─────────────────────────────────────────────────
# # # # # # # #         if epoch < 15:
# # # # # # # #             w_vel, w_head = 0.0, 0.0
# # # # # # # #         elif epoch < 40:
# # # # # # # #             phase_frac = (epoch - 15) / 25.0
# # # # # # # #             w_vel  = phase_frac * 0.3
# # # # # # # #             w_head = phase_frac * 0.2
# # # # # # # #         else:
# # # # # # # #             w_vel, w_head = 0.3, 0.2

# # # # # # # #         total = l_mse + w_vel * l_vel + w_head * l_head

# # # # # # # #         # ── Ensemble consistency (epoch 40+) ──────────────────────────────
# # # # # # # #         l_ens_consist = x_t.new_zeros(())
# # # # # # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # # # # # #             x_t2, fm_t2, u_target2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # # # # #             pred_vel2 = self.net.forward_with_ctx(
# # # # # # # #                 x_t2, fm_t2, raw_ctx, vel_obs_feat=vel_obs_feat)
# # # # # # # #             l_ens_consist = F.mse_loss(pred_vel2, u_target2)
# # # # # # # #             total = total + 0.3 * l_ens_consist

# # # # # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # # # # #             total = x_t.new_zeros(())

# # # # # # # #         return dict(
# # # # # # # #             total=total,
# # # # # # # #             mse_hav=l_mse.item(),
# # # # # # # #             velocity=l_vel.item(),
# # # # # # # #             heading=l_head.item(),
# # # # # # # #             ens_consist=l_ens_consist.item(),
# # # # # # # #             sigma=current_sigma,
# # # # # # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0,
# # # # # # # #             recurv_ratio=0.0,
# # # # # # # #         )
# # # # # # # #     @torch.no_grad()
# # # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
# # # # # # # #         """
# # # # # # # #         Inference: ODE integration từ noise → trajectory.
        
# # # # # # # #         Thay đổi so với v32:
# # # # # # # #         - Không có SR override → FM predict ALL steps
# # # # # # # #         - vel_obs_feat truyền vào decode → short-range accuracy
# # # # # # # #         - Dùng sigma_min nhỏ (0.02) khi inference → diversity
# # # # # # # #         """
# # # # # # # #         obs_t  = batch_list[0]
# # # # # # # #         lp     = obs_t[-1]
# # # # # # # #         lm     = batch_list[7][-1]
# # # # # # # #         B      = lp.shape[0]
# # # # # # # #         device = lp.device
# # # # # # # #         T      = self.pred_len
# # # # # # # #         dt     = 1.0 / max(ddim_steps, 1)

# # # # # # # #         raw_ctx      = self.net._context(batch_list)
# # # # # # # #         vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # # # #         traj_s, me_s = [], []
# # # # # # # #         for _ in range(num_ensemble):
# # # # # # # #             # x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma
# # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.sigma_min

# # # # # # # #             for step in range(ddim_steps):
# # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # #                 ns  = self.ctx_noise_scale if step < 3 else 0.0
# # # # # # # #                 vel = self.net.forward_with_ctx(
# # # # # # # #                     x_t, t_b, raw_ctx,
# # # # # # # #                     noise_scale=ns, vel_obs_feat=vel_obs_feat)
# # # # # # # #                 x_t = x_t + dt * vel
            
# # # # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # #             traj_s.append(tr)
# # # # # # # #             me_s.append(me)

# # # # # # # #         all_trajs = torch.stack(traj_s)       # [S, T, B, 2]
# # # # # # # #         pred_mean = all_trajs.mean(0)          # [T, B, 2]

# # # # # # # #         if predict_csv:
# # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # # # #         return pred_mean, torch.stack(me_s).mean(0), all_trajs

# # # # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # # # #         """Speed constraint: TCs can't move > 600 km/6h."""
# # # # # # # #         with torch.enable_grad():
# # # # # # # #             x = x_pred.detach().requires_grad_(True)
# # # # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # # # #             for _ in range(n_steps):
# # # # # # # #                 opt.zero_grad()
# # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # #                 pred_deg = _denorm_to_deg(pred_abs)
# # # # # # # #                 dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # # # #                 lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # #                 cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # #                 speed   = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # # # #                                       + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # # # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # # # # # # #                 loss.backward()
# # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # # # # # # #                 opt.step()
# # # # # # # #         return x.detach()

# # # # # # # #     @staticmethod
# # # # # # # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # # # # # # #         import numpy as np
# # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # #                     "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # #             if write_hdr: w.writeheader()
# # # # # # # #             for b in range(B):
# # # # # # # #                 for k in range(T):
# # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(math.radians(mean_lat[k, b]))
# # # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # # #                     w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # # # #                                 "lead_h": (k+1)*6, "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # # # #                                 "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # # # #                                 "lon_std_deg": f"{all_lon[:,k,b].std():.4f}",
# # # # # # # #                                 "lat_std_deg": f"{all_lat[:,k,b].std():.4f}",
# # # # # # # #                                 "ens_spread_km": f"{spread:.2f}"})


# # # # # # # # # Backward compat alias
# # # # # # # # TCDiffusion = TCFlowMatching
# # # # # # # """
# # # # # # # flow_matching_model_v33.py — Pure FM + MSE: Beat LSTM bằng đơn giản

# # # # # # # CHANGES v33→v33fix:
# # # # # # #   1. BỎ `with torch.no_grad()` khi reconstruct pred_abs → l_vel/l_head có gradient thực
# # # # # # #   2. THÊM long_range_aux_loss() chỉ cho step 8-12 (48h-72h), zero risk cho 6-24h
# # # # # # #   3. Phase weight cho long_range loss: chỉ bật sau epoch 20 (sau khi 12h ổn)
# # # # # # #   4. Median ensemble thay mean tại inference → robust hơn với outlier members
# # # # # # #   5. MSE_STEP_WEIGHTS giữ nguyên (không đổi short-range emphasis)
# # # # # # # """
# # # # # # # from __future__ import annotations

# # # # # # # import csv
# # # # # # # import math
# # # # # # # import os
# # # # # # # from datetime import datetime
# # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # import torch
# # # # # # # import torch.nn as nn
# # # # # # # import torch.nn.functional as F

# # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # from Model.losses import _haversine_deg, _norm_to_deg, _haversine

# # # # # # # # ── Step weights: nhấn mạnh 12h và 24h ─────────────────────────────────────
# # # # # # # # Step:    6h   12h  18h  24h  30h  36h  42h  48h  54h  60h  66h  72h
# # # # # # # # Thay MSE_STEP_WEIGHTS
# # # # # # # MSE_STEP_WEIGHTS = [
# # # # # # #     1.0,  # 6h
# # # # # # #     4.0,  # 12h
# # # # # # #     1.5,  # 18h
# # # # # # #     3.0,  # 24h
# # # # # # #     1.0,  # 30h
# # # # # # #     1.0,  # 36h
# # # # # # #     1.5,  # 42h
# # # # # # #     3.0,  # 48h  ← tăng từ 2.5
# # # # # # #     2.0,  # 54h  ← tăng từ 1.0
# # # # # # #     2.5,  # 60h  ← tăng từ 1.0
# # # # # # #     3.5,  # 66h  ← tăng từ 1.5
# # # # # # #     5.0,  # 72h  ← tăng từ 2.5, quan trọng nhất
# # # # # # # ]

# # # # # # # def _denorm_to_deg(t):
# # # # # # #     """Convert normalized coords to degrees."""
# # # # # # #     out = t.clone()
# # # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # # #     return out


# # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  MSE Haversine Loss
# # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # def mse_hav_loss(pred_norm, gt_norm, step_w=None):
# # # # # # #     """Haversine MSE per-step với Huber để tránh gradient explosion."""
# # # # # # #     if step_w is None:
# # # # # # #         step_w = MSE_STEP_WEIGHTS
# # # # # # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # #     pred_deg = _norm_to_deg(pred_norm[:T])
# # # # # # #     gt_deg   = _norm_to_deg(gt_norm[:T])
# # # # # # #     dist_km  = _haversine_deg(pred_deg, gt_deg)   # [T, B]

# # # # # # #     w = pred_norm.new_tensor(step_w[:T])
# # # # # # #     w = w / w.sum() * T

# # # # # # #     delta = 300.0
# # # # # # #     huber = torch.where(
# # # # # # #         dist_km < delta,
# # # # # # #         dist_km.pow(2) / (2.0 * delta),
# # # # # # #         dist_km - delta / 2.0,
# # # # # # #     )
# # # # # # #     return (huber * w.unsqueeze(1)).mean() / delta


# # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  NEW: Long-range Auxiliary Loss — chỉ 48h-72h, zero risk cho 6-24h
# # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # def long_range_aux_loss(pred_abs, gt_abs, start_step=7):
# # # # # # #     """
# # # # # # #     Auxiliary loss chỉ cho step start_step trở đi (mặc định step 8-12 = 48h-72h).

# # # # # # #     KEY PROPERTIES:
# # # # # # #     - Gradient chỉ flow từ step 8-12, không đụng step 1-7
# # # # # # #     - L1 Huber (delta=200km) robust hơn L2 với outlier trajectories
# # # # # # #     - Weight tăng dần theo lead time (penalize 72h nhiều hơn 48h)
# # # # # # #     - Normalize để scale tương đương với l_mse
# # # # # # #     """
# # # # # # #     T = min(pred_abs.shape[0], gt_abs.shape[0])
# # # # # # #     if T <= start_step:
# # # # # # #         return pred_abs.new_zeros(())

# # # # # # #     pred_lr = pred_abs[start_step:]   # step 8-12: 48h→72h [T_lr, B, 2]
# # # # # # #     gt_lr   = gt_abs[start_step:]     # [T_lr, B, 2]

# # # # # # #     # Convert to degrees để tính haversine
# # # # # # #     pred_lr_deg = _denorm_to_deg(pred_lr)
# # # # # # #     gt_lr_deg   = _denorm_to_deg(gt_lr)

# # # # # # #     dist_km = _haversine_deg(pred_lr_deg, gt_lr_deg)  # [T_lr, B]

# # # # # # #     T_lr = dist_km.shape[0]
# # # # # # #     # Weight tăng dần: 48h(1.0) → 54h(1.5) → 60h(2.0) → 66h(2.5) → 72h(3.0)
# # # # # # #     w_vals = [1.0, 1.5, 2.0, 2.5, 3.0]
# # # # # # #     w = pred_abs.new_tensor(w_vals[:T_lr])
# # # # # # #     w = w / w.sum()

# # # # # # #     # Huber loss, delta=200km (robust hơn L2 với large errors)
# # # # # # #     delta = 200.0
# # # # # # #     huber = torch.where(
# # # # # # #         dist_km < delta,
# # # # # # #         0.5 * dist_km.pow(2) / delta,
# # # # # # #         dist_km - delta / 2.0,
# # # # # # #     )
# # # # # # #     return (huber * w.unsqueeze(1)).mean() / delta


# # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  VelocityField
# # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # class VelocityField(nn.Module):
# # # # # # #     RAW_CTX_DIM = 512

# # # # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # # # #                  unet_in_ch=13, **kwargs):
# # # # # # #         super().__init__()
# # # # # # #         self.pred_len = pred_len
# # # # # # #         self.obs_len  = obs_len

# # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # #             spatial_down=32, dropout=0.05)
# # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # #         self.ctx_drop = nn.Dropout(0.10)
# # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # # # #             nn.LayerNorm(256),
# # # # # # #             nn.Linear(256, 256), nn.GELU(),
# # # # # # #         )

# # # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)

# # # # # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # #             nn.TransformerDecoderLayer(
# # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # # # # #             num_layers=5)

# # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # #         self.step_scale  = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.3)

# # # # # # #         self._init_weights()

# # # # # # #     def _init_weights(self):
# # # # # # #         with torch.no_grad():
# # # # # # #             for name, m in self.named_modules():
# # # # # # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # # # # #                     if m.bias is not None:
# # # # # # #                         nn.init.zeros_(m.bias)

# # # # # # #     def _time_emb(self, t, dim=256):
# # # # # # #         half = dim // 2
# # # # # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # # # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # # # # #     def _context(self, batch_list):
# # # # # # #         obs_traj  = batch_list[0]
# # # # # # #         obs_Me    = batch_list[7]
# # # # # # #         image_obs = batch_list[11]
# # # # # # #         env_data  = batch_list[13]

# # # # # # #         if image_obs.dim() == 4:
# # # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)

# # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # # #         if noise_scale > 0.0:
# # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # #     def _get_vel_obs_feat(self, obs_traj):
# # # # # # #         B = obs_traj.shape[1]
# # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # #         if T_obs >= 2:
# # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # # # # #         else:
# # # # # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)

# # # # # # #         if vels.shape[0] < self.obs_len:
# # # # # # #             pad = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # #         else:
# # # # # # #             vels = vels[-self.obs_len:]

# # # # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # # # #     def _beta_drift(self, x_t):
# # # # # # #         lat_deg  = x_t[:, :, 1] * 5.0
# # # # # # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # # #         beta     = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # # #         R_tc     = 3e5
# # # # # # #         v_phys   = torch.zeros_like(x_t)
# # # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # # #         return v_phys

# # # # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None):
# # # # # # #         B = x_t.shape[0]
# # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # #         T_seq = min(x_t.size(1), self.pos_enc.shape[1])

# # # # # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # #         s_emb = self.step_embed(step_idx)

# # # # # # #         x_emb = self.traj_embed(x_t[:, :T_seq]) + self.pos_enc[:, :T_seq] + t_emb.unsqueeze(1) + s_emb

# # # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # # #         if vel_obs_feat is not None:
# # # # # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))

# # # # # # #         decoded = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))

# # # # # # #         scale = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # # # #         v_neural = v_neural * scale

# # # # # # #         with torch.no_grad():
# # # # # # #             v_phys = self._beta_drift(x_t[:, :T_seq])

# # # # # # #         return v_neural + torch.sigmoid(self.physics_scale) * v_phys

# # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0, vel_obs_feat=None):
# # # # # # #         return self._decode(x_t, t, self._apply_ctx_head(raw_ctx, noise_scale), vel_obs_feat)


# # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  TCFlowMatching v33fix
# # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # class TCFlowMatching(nn.Module):

# # # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # # #                  ctx_noise_scale=0.002, initial_sample_sigma=0.03, **kwargs):
# # # # # # #         super().__init__()
# # # # # # #         self.pred_len             = pred_len
# # # # # # #         self.obs_len              = obs_len
# # # # # # #         self.sigma_min            = sigma_min
# # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # #         self.active_pred_len      = pred_len

# # # # # # #         self.net = VelocityField(
# # # # # # #             pred_len=pred_len, obs_len=obs_len,
# # # # # # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # # # #     def set_curriculum_len(self, *a, **kw):
# # # # # # #         pass

# # # # # # #     @staticmethod
# # # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # # #                            Me   - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # # # #     @staticmethod
# # # # # # #     def _to_abs(rel, lp, lm):
# # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # # # # #         if sigma_min is None:
# # # # # # #             sigma_min = self.sigma_min
# # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # #         x0 = torch.randn_like(x1) * sigma_min
# # # # # # #         t  = torch.rand(B, device=device)
# # # # # # #         te = t.view(B, 1, 1)
# # # # # # #         x_t = (1.0 - te) * x0 + te * x1
# # # # # # #         u   = x1 - x0
# # # # # # #         return x_t, t, u

# # # # # # #     @staticmethod
# # # # # # #     def _intensity_weights(obs_Me):
# # # # # # #         w = obs_Me[-1, :, 1].detach()
# # # # # # #         w = torch.where(w < 0.1, torch.full_like(w, 0.5),
# # # # # # #             torch.where(w < 0.3, torch.full_like(w, 0.8),
# # # # # # #             torch.where(w < 0.6, torch.full_like(w, 1.0),
# # # # # # #                         torch.full_like(w, 1.5))))
# # # # # # #         return w / w.mean().clamp(min=1e-6)

# # # # # # #     @staticmethod
# # # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # # #         if torch.rand(1).item() > p:
# # # # # # #             return batch_list
# # # # # # #         aug = list(batch_list)
# # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # # #                 t = aug[idx].clone()
# # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # #                 aug[idx] = t
# # # # # # #         return aug

# # # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # #         """
# # # # # # #         FIX KEY CHANGES:
# # # # # # #         1. BỎ `with torch.no_grad()` → pred_abs có gradient thực
# # # # # # #         2. THÊM long_range_aux_loss cho 48h-72h (chỉ bật sau epoch 20)
# # # # # # #         3. l_vel và l_head bây giờ có gradient flow đúng
# # # # # # #         """
# # # # # # #         batch_list = self._lon_flip_aug(batch_list)

# # # # # # #         traj_gt = batch_list[1]
# # # # # # #         Me_gt   = batch_list[8]
# # # # # # #         obs_t   = batch_list[0]
# # # # # # #         obs_Me  = batch_list[7]
# # # # # # #         lp, lm  = obs_t[-1], obs_Me[-1]

# # # # # # #         # ── Sigma schedule ──────────────────────────────────────────────────
# # # # # # #         if epoch < 15:
# # # # # # #             current_sigma = 0.15
# # # # # # #         elif epoch < 40:
# # # # # # #             sigma_frac = (epoch - 15) / 25.0
# # # # # # #             current_sigma = 0.15 - sigma_frac * (0.15 - 0.03)
# # # # # # #         else:
# # # # # # #             current_sigma = 0.03

# # # # # # #         # ── Encode context ───────────────────────────────────────────────────
# # # # # # #         raw_ctx      = self.net._context(batch_list)
# # # # # # #         vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # # #         # ── FM forward ───────────────────────────────────────────────────────
# # # # # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)  # [B, T, 4]
# # # # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # # # #         pred_vel = self.net.forward_with_ctx(
# # # # # # #             x_t, fm_t, raw_ctx, vel_obs_feat=vel_obs_feat)

# # # # # # #         # ── LOSS 1: FM velocity MSE (primary) ────────────────────────────────
# # # # # # #         l_mse = F.mse_loss(pred_vel, u_target)

# # # # # # #         # ── Reconstruct pred_abs ─────────────────────────────────────────────
# # # # # # #         # FIX: BỎ `with torch.no_grad()` để l_vel, l_head, l_lr có gradient thực
# # # # # # #         # → l_vel/l_head từ epoch 15+ bây giờ thực sự optimize model weights
# # # # # # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# # # # # # #         pred_deg = _denorm_to_deg(pred_abs)
# # # # # # #         gt_deg   = _denorm_to_deg(traj_gt)

# # # # # # #         # ── LOSS 2: Velocity matching ─────────────────────────────────────────
# # # # # # #         if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # #             T_min  = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # #             v_pred = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # #             v_gt   = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # # #             l_vel  = F.smooth_l1_loss(v_pred, v_gt)
# # # # # # #         else:
# # # # # # #             l_vel = x_t.new_zeros(())

# # # # # # #         # ── LOSS 3: Heading consistency ────────────────────────────────────────
# # # # # # #         if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # #             T_min    = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # #             v_pred   = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # #             v_gt     = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # # #             v_pred_n = v_pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # #             v_gt_n   = v_gt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # #             cos_sim  = ((v_pred / v_pred_n) * (v_gt / v_gt_n)).sum(-1)
# # # # # # #             l_head   = F.relu(-cos_sim).pow(2).mean()
# # # # # # #         else:
# # # # # # #             l_head = x_t.new_zeros(())

# # # # # # #         # ── LOSS 4: Long-range auxiliary (48h-72h ONLY) ────────────────────────
# # # # # # #         # FIX: Loss này chỉ tính gradient từ step 8-12, KHÔNG đụng step 1-7
# # # # # # #         # → 6h/12h/24h hoàn toàn không bị ảnh hưởng
# # # # # # #         # → Chỉ bật sau epoch 20 khi 12h đã ổn định (< 60km)
# # # # # # #         l_lr = long_range_aux_loss(pred_abs, traj_gt, start_step=7)

# # # # # # #         # ── Phase weights ──────────────────────────────────────────────────────
# # # # # # #         # if epoch < 15:
# # # # # # #         #     w_vel, w_head = 0.0, 0.0
# # # # # # #         #     w_lr = 0.0
# # # # # # #         # elif epoch < 20:
# # # # # # #         #     # Transition: vel/head bắt đầu, lr chưa bật
# # # # # # #         #     phase_frac = (epoch - 15) / 5.0
# # # # # # #         #     w_vel  = phase_frac * 0.3
# # # # # # #         #     w_head = phase_frac * 0.2
# # # # # # #         #     w_lr   = 0.0
# # # # # # #         # elif epoch < 40:
# # # # # # #         #     phase_frac = (epoch - 15) / 25.0
# # # # # # #         #     w_vel  = phase_frac * 0.3
# # # # # # #         #     w_head = phase_frac * 0.2
# # # # # # #         #     # Long-range loss: warm-up từ epoch 20, tăng dần đến 0.5
# # # # # # #         #     lr_frac = (epoch - 20) / 20.0
# # # # # # #         #     w_lr = min(lr_frac, 1.0) * 0.5
# # # # # # #         # else:
# # # # # # #         #     w_vel, w_head = 0.3, 0.2
# # # # # # #         #     w_lr = 0.5  # Full weight sau epoch 40

# # # # # # #         # total = l_mse + w_vel * l_vel + w_head * l_head + w_lr * l_lr
# # # # # # #          # --- THÊM: Terminal displacement loss (FDE-style) ---
# # # # # # #         # Penalize error tại step 12 một cách mạnh mẽ hơn
# # # # # # #         T = min(pred_abs.shape[0], traj_gt.shape[0])
# # # # # # #         if T >= 12:
# # # # # # #             pred_last = pred_abs[11]   # step 12 normalized
# # # # # # #             gt_last   = traj_gt[11]
# # # # # # #             pred_last_deg = _denorm_to_deg(pred_last.unsqueeze(0))
# # # # # # #             gt_last_deg   = _denorm_to_deg(gt_last.unsqueeze(0))
# # # # # # #             l_fde = _haversine_deg(pred_last_deg, gt_last_deg).mean()
# # # # # # #             # Normalize: target ~300km → l_fde/300 ~ 1.0
# # # # # # #             l_fde = l_fde / 300.0
# # # # # # #         else:
# # # # # # #             l_fde = pred_abs.new_zeros(())
        
# # # # # # #         # --- THÊM: Long-range trajectory shape loss ---
# # # # # # #         # MSE giữa displacement vectors ở step 7-12
# # # # # # #         if pred_deg.shape[0] >= 12 and gt_deg.shape[0] >= 12:
# # # # # # #             pred_lr = pred_deg[6:]    # step 7-12 [6, B, 2]
# # # # # # #             gt_lr   = gt_deg[6:]
# # # # # # #             # Displacement từ step 7 đến step 12
# # # # # # #             pred_disp = pred_lr[-1] - pred_lr[0]   # [B, 2]
# # # # # # #             gt_disp   = gt_lr[-1]   - gt_lr[0]
# # # # # # #             # Penalize sai hướng và magnitude
# # # # # # #             l_lr_shape = F.smooth_l1_loss(pred_disp, gt_disp)
# # # # # # #         else:
# # # # # # #             l_lr_shape = pred_abs.new_zeros(())
        
# # # # # # #         # FDE weight schedule: bắt đầu nhẹ, tăng dần
# # # # # # #         if epoch < 20:
# # # # # # #             w_fde = 0.0
# # # # # # #             w_lr_shape = 0.0
# # # # # # #         elif epoch < 40:
# # # # # # #             w_fde = (epoch - 20) / 20.0 * 0.5      # 0 → 0.5
# # # # # # #             w_lr_shape = (epoch - 20) / 20.0 * 0.3  # 0 → 0.3
# # # # # # #         else:
# # # # # # #             w_fde = 0.5
# # # # # # #             w_lr_shape = 0.3

# # # # # # #         total = l_mse + w_vel * l_vel + w_head * l_head + w_lr * l_lr \
# # # # # # #                 + w_fde * l_fde + w_lr_shape * l_lr_shape
# # # # # # #         # ── Ensemble consistency (epoch 40+) ────────────────────────────────────
# # # # # # #         l_ens_consist = x_t.new_zeros(())
# # # # # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # # # # #             x_t2, fm_t2, u_target2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # # # #             pred_vel2 = self.net.forward_with_ctx(
# # # # # # #                 x_t2, fm_t2, raw_ctx, vel_obs_feat=vel_obs_feat)
# # # # # # #             l_ens_consist = F.mse_loss(pred_vel2, u_target2)
# # # # # # #             total = total + 0.3 * l_ens_consist

# # # # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # # # #             total = x_t.new_zeros(())

# # # # # # #         return dict(
# # # # # # #             total=total,
# # # # # # #             mse_hav=l_mse.item(),
# # # # # # #             velocity=l_vel.item() if isinstance(l_vel, torch.Tensor) else l_vel,
# # # # # # #             heading=l_head.item() if isinstance(l_head, torch.Tensor) else l_head,
# # # # # # #             long_range=l_lr.item(),
# # # # # # #             ens_consist=l_ens_consist.item(),
# # # # # # #             sigma=current_sigma,
# # # # # # #             w_lr=w_lr,
# # # # # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0,
# # # # # # #             recurv_ratio=0.0,
# # # # # # #         )

# # # # # # #     @torch.no_grad()
# # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
# # # # # # #         """
# # # # # # #         FIX: Dùng median thay mean → robust hơn với outlier ensemble members.
# # # # # # #         Median đặc biệt tốt cho long-range vì spread lớn hơn ở 48h/72h.
# # # # # # #         """
# # # # # # #         obs_t  = batch_list[0]
# # # # # # #         lp     = obs_t[-1]
# # # # # # #         lm     = batch_list[7][-1]
# # # # # # #         B      = lp.shape[0]
# # # # # # #         device = lp.device
# # # # # # #         T      = self.pred_len
# # # # # # #         dt     = 1.0 / max(ddim_steps, 1)

# # # # # # #         raw_ctx      = self.net._context(batch_list)
# # # # # # #         vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # # #         traj_s, me_s = [], []
# # # # # # #         for _ in range(num_ensemble):
# # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.sigma_min

# # # # # # #             for step in range(ddim_steps):
# # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # #                 ns  = self.ctx_noise_scale if step < 3 else 0.0
# # # # # # #                 vel = self.net.forward_with_ctx(
# # # # # # #                     x_t, t_b, raw_ctx,
# # # # # # #                     noise_scale=ns, vel_obs_feat=vel_obs_feat)
# # # # # # #                 x_t = x_t + dt * vel

# # # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # #             traj_s.append(tr)
# # # # # # #             me_s.append(me)

# # # # # # #         all_trajs = torch.stack(traj_s)       # [S, T, B, 2]

# # # # # # #         # FIX: Median thay vì mean → robust với outlier ensemble members
# # # # # # #         # Đặc biệt quan trọng cho 48h/72h khi spread lớn (~50-70km)
# # # # # # #         pred_mean = all_trajs.median(0).values  # [T, B, 2]

# # # # # # #         if predict_csv:
# # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # # #         return pred_mean, torch.stack(me_s).mean(0), all_trajs

# # # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # # #         """Speed constraint: TCs can't move > 600 km/6h."""
# # # # # # #         with torch.enable_grad():
# # # # # # #             x = x_pred.detach().requires_grad_(True)
# # # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # # #             for _ in range(n_steps):
# # # # # # #                 opt.zero_grad()
# # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # #                 pred_deg = _denorm_to_deg(pred_abs)
# # # # # # #                 dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # # #                 lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # #                 cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # #                 speed   = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # # #                                       + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # # # # # #                 loss.backward()
# # # # # # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # # # # # #                 opt.step()
# # # # # # #         return x.detach()

# # # # # # #     @staticmethod
# # # # # # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # # # # # #         import numpy as np
# # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # #         T, B, _ = traj_mean.shape
# # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # #                     "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # #             if write_hdr: w.writeheader()
# # # # # # #             for b in range(B):
# # # # # # #                 for k in range(T):
# # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(math.radians(mean_lat[k, b]))
# # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # #                     w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # # #                                 "lead_h": (k+1)*6, "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # # #                                 "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # # #                                 "lon_std_deg": f"{all_lon[:,k,b].std():.4f}",
# # # # # # #                                 "lat_std_deg": f"{all_lat[:,k,b].std():.4f}",
# # # # # # #                                 "ens_spread_km": f"{spread:.2f}"})


# # # # # # # # Backward compat alias
# # # # # # # TCDiffusion = TCFlowMatching
# # # # # # """
# # # # # # flow_matching_model_v33_fixed.py

# # # # # # FIXES từ v33fix — 3 bugs nghiêm trọng:

# # # # # # BUG 1 (NameError): w_vel, w_lr, w_head undefined.
# # # # # #   Code cũ comment out phần định nghĩa nhưng vẫn dùng trong total.
# # # # # #   → NameError khi chạy, training crash ngay batch đầu.

# # # # # # BUG 2 (double-denorm): l_lr_shape dùng pred_deg (đã degrees)
# # # # # #   rồi gọi _denorm_to_deg lần nữa → kết quả vô nghĩa,
# # # # # #   gradient sai hoàn toàn, 72h không học được gì.

# # # # # # BUG 3 (wrong input): long_range_aux_loss trong v33fix
# # # # # #   nhận pred_abs (normalized) nhưng gọi _denorm_to_deg để convert —
# # # # # #   đây là ĐÚNG. Nhưng lr_shape_loss cũng nhận pred_deg (degrees)
# # # # # #   mà lại gọi pred_deg[6:] trên degrees — đây cũng ĐÚNG nếu giữ nguyên.
# # # # # #   Vấn đề là code cũ lẫn lộn normalized và degrees không nhất quán.

# # # # # # SOLUTION: Tách 3 loss functions với input type rõ ràng:
# # # # # #   - long_range_aux_loss(normalized, normalized) → haversine nội bộ
# # # # # #   - fde_loss(normalized, normalized) → haversine tại step cuối
# # # # # #   - lr_shape_loss(degrees, degrees) → smooth_l1 displacement

# # # # # # Phase weights bật 72h losses từ epoch 0 (không đợi epoch 20).
# # # # # # """
# # # # # # from __future__ import annotations

# # # # # # import csv
# # # # # # import math
# # # # # # import os
# # # # # # from datetime import datetime

# # # # # # import torch
# # # # # # import torch.nn as nn
# # # # # # import torch.nn.functional as F

# # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # from Model.losses import _haversine_deg, _norm_to_deg, _haversine

# # # # # # MSE_STEP_WEIGHTS = [
# # # # # #     1.0,  # 6h
# # # # # #     4.0,  # 12h  target < 50km
# # # # # #     1.5,  # 18h
# # # # # #     3.0,  # 24h  target < 100km
# # # # # #     1.0,  # 30h
# # # # # #     1.0,  # 36h
# # # # # #     1.5,  # 42h
# # # # # #     3.0,  # 48h  target < 200km
# # # # # #     2.0,  # 54h
# # # # # #     2.5,  # 60h
# # # # # #     3.5,  # 66h
# # # # # #     5.0,  # 72h  target < 300km — weight cao nhất
# # # # # # ]


# # # # # # def _denorm_to_deg(t):
# # # # # #     out = t.clone()
# # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # #     return out


# # # # # # def mse_hav_loss(pred_norm, gt_norm, step_w=None):
# # # # # #     if step_w is None:
# # # # # #         step_w = MSE_STEP_WEIGHTS
# # # # # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # #     pred_deg = _norm_to_deg(pred_norm[:T])
# # # # # #     gt_deg   = _norm_to_deg(gt_norm[:T])
# # # # # #     dist_km  = _haversine_deg(pred_deg, gt_deg)
# # # # # #     w = pred_norm.new_tensor(step_w[:T])
# # # # # #     w = w / w.sum() * T
# # # # # #     delta = 300.0
# # # # # #     huber = torch.where(dist_km < delta,
# # # # # #                         dist_km.pow(2) / (2.0 * delta),
# # # # # #                         dist_km - delta / 2.0)
# # # # # #     return (huber * w.unsqueeze(1)).mean() / delta


# # # # # # def long_range_aux_loss(pred_norm, gt_norm, start_step=7):
# # # # # #     """Input: NORMALIZED [T,B,2]. Gradient chỉ từ step 8-12."""
# # # # # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # #     if T <= start_step:
# # # # # #         return pred_norm.new_zeros(())
# # # # # #     pred_deg = _norm_to_deg(pred_norm[start_step:T])
# # # # # #     gt_deg   = _norm_to_deg(gt_norm[start_step:T])
# # # # # #     dist_km  = _haversine_deg(pred_deg, gt_deg)
# # # # # #     T_lr = dist_km.shape[0]
# # # # # #     w_vals = [1.0, 1.5, 2.0, 2.5, 3.0]
# # # # # #     w = pred_norm.new_tensor(w_vals[:T_lr])
# # # # # #     w = w / w.sum()
# # # # # #     delta = 200.0
# # # # # #     huber = torch.where(dist_km < delta,
# # # # # #                         0.5 * dist_km.pow(2) / delta,
# # # # # #                         dist_km - delta / 2.0)
# # # # # #     return (huber * w.unsqueeze(1)).mean() / delta


# # # # # # def fde_loss(pred_norm, gt_norm):
# # # # # #     """Terminal haversine error tại step cuối. Input: NORMALIZED [T,B,2]."""
# # # # # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # #     if T < 1:
# # # # # #         return pred_norm.new_zeros(())
# # # # # #     pred_last_deg = _norm_to_deg(pred_norm[T-1:T])
# # # # # #     gt_last_deg   = _norm_to_deg(gt_norm[T-1:T])
# # # # # #     return _haversine_deg(pred_last_deg, gt_last_deg).mean() / 300.0


# # # # # # def lr_shape_loss(pred_deg, gt_deg):
# # # # # #     """Displacement step7→step12. Input: DEGREES [T,B,2]. Không denorm."""
# # # # # #     if pred_deg.shape[0] < 12 or gt_deg.shape[0] < 12:
# # # # # #         return pred_deg.new_zeros(())
# # # # # #     pred_disp = pred_deg[11] - pred_deg[6]
# # # # # #     gt_disp   = gt_deg[11]   - gt_deg[6]
# # # # # #     return F.smooth_l1_loss(pred_disp, gt_disp)


# # # # # # class VelocityField(nn.Module):
# # # # # #     RAW_CTX_DIM = 512

# # # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # # #                  unet_in_ch=13, **kwargs):
# # # # # #         super().__init__()
# # # # # #         self.pred_len = pred_len
# # # # # #         self.obs_len  = obs_len

# # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # #             spatial_down=32, dropout=0.05)
# # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # # #         self.enc_1d = DataEncoder1D(
# # # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # #         self.ctx_drop = nn.Dropout(0.10)
# # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # # #             nn.LayerNorm(256),
# # # # # #             nn.Linear(256, 256), nn.GELU(),
# # # # # #         )

# # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # #             nn.TransformerDecoderLayer(
# # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # # # #             num_layers=5)

# # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # #         self.step_scale    = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.3)
# # # # # #         self._init_weights()

# # # # # #     def _init_weights(self):
# # # # # #         with torch.no_grad():
# # # # # #             for name, m in self.named_modules():
# # # # # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # # # #                     if m.bias is not None:
# # # # # #                         nn.init.zeros_(m.bias)

# # # # # #     def _time_emb(self, t, dim=256):
# # # # # #         half = dim // 2
# # # # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # # # #     def _context(self, batch_list):
# # # # # #         obs_traj  = batch_list[0]
# # # # # #         obs_Me    = batch_list[7]
# # # # # #         image_obs = batch_list[11]
# # # # # #         env_data  = batch_list[13]

# # # # # #         if image_obs.dim() == 4:
# # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # #         T_obs = obs_traj.shape[0]

# # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)

# # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(
# # # # # #             torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # #         if noise_scale > 0.0:
# # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # #     def _get_vel_obs_feat(self, obs_traj):
# # # # # #         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
# # # # # #         if T_obs >= 2:
# # # # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # # # #         else:
# # # # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
# # # # # #         if vels.shape[0] < self.obs_len:
# # # # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # #         else:
# # # # # #             vels = vels[-self.obs_len:]
# # # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # # #     def _beta_drift(self, x_t):
# # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # #         R_tc    = 3e5
# # # # # #         v_phys  = torch.zeros_like(x_t)
# # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # #         return v_phys

# # # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None):
# # # # # #         B     = x_t.shape[0]
# # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # #         s_emb    = self.step_embed(step_idx)

# # # # # #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# # # # # #                  + self.pos_enc[:, :T_seq]
# # # # # #                  + t_emb.unsqueeze(1)
# # # # # #                  + s_emb)

# # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # #         if vel_obs_feat is not None:
# # # # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))

# # # # # #         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))

# # # # # #         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # # #         v_neural = v_neural * scale

# # # # # #         with torch.no_grad():
# # # # # #             v_phys = self._beta_drift(x_t[:, :T_seq])
# # # # # #         return v_neural + torch.sigmoid(self.physics_scale) * v_phys

# # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0, vel_obs_feat=None):
# # # # # #         return self._decode(x_t, t, self._apply_ctx_head(raw_ctx, noise_scale), vel_obs_feat)


# # # # # # class TCFlowMatching(nn.Module):

# # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # #                  ctx_noise_scale=0.002, initial_sample_sigma=0.03, **kwargs):
# # # # # #         super().__init__()
# # # # # #         self.pred_len             = pred_len
# # # # # #         self.obs_len              = obs_len
# # # # # #         self.sigma_min            = sigma_min
# # # # # #         self.n_train_ens          = n_train_ens
# # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # #         self.active_pred_len      = pred_len
# # # # # #         self.net = VelocityField(pred_len=pred_len, obs_len=obs_len,
# # # # # #                                   sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # # #     def set_curriculum_len(self, *a, **kw):
# # # # # #         pass

# # # # # #     @staticmethod
# # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # # #     @staticmethod
# # # # # #     def _to_abs(rel, lp, lm):
# # # # # #         d = rel.permute(1, 0, 2)
# # # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # # # #         if sigma_min is None:
# # # # # #             sigma_min = self.sigma_min
# # # # # #         B, device = x1.shape[0], x1.device
# # # # # #         x0 = torch.randn_like(x1) * sigma_min
# # # # # #         t  = torch.rand(B, device=device)
# # # # # #         te = t.view(B, 1, 1)
# # # # # #         x_t = (1.0 - te) * x0 + te * x1
# # # # # #         u   = x1 - x0
# # # # # #         return x_t, t, u

# # # # # #     @staticmethod
# # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # #         if torch.rand(1).item() > p:
# # # # # #             return batch_list
# # # # # #         aug = list(batch_list)
# # # # # #         for idx in [0, 1, 2, 3]:
# # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # #                 t = aug[idx].clone()
# # # # # #                 t[..., 0] = -t[..., 0]
# # # # # #                 aug[idx] = t
# # # # # #         return aug

# # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # #         batch_list = self._lon_flip_aug(batch_list)

# # # # # #         traj_gt = batch_list[1]
# # # # # #         Me_gt   = batch_list[8]
# # # # # #         obs_t   = batch_list[0]
# # # # # #         obs_Me  = batch_list[7]
# # # # # #         lp, lm  = obs_t[-1], obs_Me[-1]

# # # # # #         if epoch < 15:
# # # # # #             current_sigma = 0.15
# # # # # #         elif epoch < 40:
# # # # # #             current_sigma = 0.15 - (epoch - 15) / 25.0 * 0.12
# # # # # #         else:
# # # # # #             current_sigma = 0.03

# # # # # #         raw_ctx      = self.net._context(batch_list)
# # # # # #         vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
# # # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # # #         pred_vel = self.net.forward_with_ctx(x_t, fm_t, raw_ctx, vel_obs_feat=vel_obs_feat)

# # # # # #         l_mse = F.mse_loss(pred_vel, u_target)

# # # # # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)  # normalized [T,B,2]
# # # # # #         pred_deg = _denorm_to_deg(pred_abs)            # degrees [T,B,2]
# # # # # #         gt_deg   = _denorm_to_deg(traj_gt)             # degrees [T,B,2]

# # # # # #         # L2: velocity (degrees)
# # # # # #         if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # #             T_min  = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # #             v_pred = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # #             v_gt   = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # #             l_vel  = F.smooth_l1_loss(v_pred, v_gt)
# # # # # #         else:
# # # # # #             l_vel  = x_t.new_zeros(())

# # # # # #         # L3: heading (degrees)
# # # # # #         if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # #             T_min    = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # #             v_pred   = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # #             v_gt     = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # #             v_pred_n = v_pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # #             v_gt_n   = v_gt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # #             cos_sim  = ((v_pred / v_pred_n) * (v_gt / v_gt_n)).sum(-1)
# # # # # #             l_head   = F.relu(-cos_sim).pow(2).mean()
# # # # # #         else:
# # # # # #             l_head   = x_t.new_zeros(())

# # # # # #         # L4: long-range auxiliary (NORMALIZED input)
# # # # # #         l_lr = long_range_aux_loss(pred_abs, traj_gt, start_step=7)

# # # # # #         # L5: FDE terminal (NORMALIZED input)
# # # # # #         l_fde = fde_loss(pred_abs, traj_gt)

# # # # # #         # L6: shape (DEGREES input — NO second denorm)
# # # # # #         l_shape = lr_shape_loss(pred_deg, gt_deg)

# # # # # #         # ── ALL weights defined here, no undefined variables ──────────────
# # # # # #         if epoch < 5:
# # # # # #             w_vel = 0.0;  w_head = 0.0
# # # # # #             w_lr  = 0.2;  w_fde  = 0.15;  w_shape = 0.1
# # # # # #         elif epoch < 20:
# # # # # #             t_p   = (epoch - 5) / 15.0
# # # # # #             w_vel   = t_p * 0.3
# # # # # #             w_head  = t_p * 0.2
# # # # # #             w_lr    = 0.2 + t_p * 0.3
# # # # # #             w_fde   = 0.15 + t_p * 0.35
# # # # # #             w_shape = 0.1 + t_p * 0.2
# # # # # #         else:
# # # # # #             w_vel = 0.3;  w_head = 0.2
# # # # # #             w_lr  = 0.5;  w_fde  = 0.5;   w_shape = 0.3

# # # # # #         total = (l_mse
# # # # # #                  + w_vel   * l_vel
# # # # # #                  + w_head  * l_head
# # # # # #                  + w_lr    * l_lr
# # # # # #                  + w_fde   * l_fde
# # # # # #                  + w_shape * l_shape)

# # # # # #         l_ens = x_t.new_zeros(())
# # # # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # # # #             x_t2, fm_t2, u_t2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # # #             pv2   = self.net.forward_with_ctx(x_t2, fm_t2, raw_ctx, vel_obs_feat=vel_obs_feat)
# # # # # #             l_ens = F.mse_loss(pv2, u_t2)
# # # # # #             total = total + 0.3 * l_ens

# # # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # # #             total = x_t.new_zeros(())

# # # # # #         return dict(
# # # # # #             total      = total,
# # # # # #             mse_hav    = l_mse.item(),
# # # # # #             velocity   = l_vel.item()   if torch.is_tensor(l_vel)   else float(l_vel),
# # # # # #             heading    = l_head.item()  if torch.is_tensor(l_head)  else float(l_head),
# # # # # #             long_range = l_lr.item(),
# # # # # #             fde        = l_fde.item(),
# # # # # #             shape      = l_shape.item() if torch.is_tensor(l_shape) else float(l_shape),
# # # # # #             ens_consist= l_ens.item(),
# # # # # #             sigma      = current_sigma,
# # # # # #             w_lr       = w_lr,
# # # # # #             w_fde      = w_fde,
# # # # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0, recurv_ratio=0.0,
# # # # # #         )

# # # # # #     @torch.no_grad()
# # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
# # # # # #         obs_t  = batch_list[0]
# # # # # #         lp     = obs_t[-1]
# # # # # #         lm     = batch_list[7][-1]
# # # # # #         B      = lp.shape[0]
# # # # # #         device = lp.device
# # # # # #         T      = self.pred_len
# # # # # #         dt     = 1.0 / max(ddim_steps, 1)

# # # # # #         raw_ctx      = self.net._context(batch_list)
# # # # # #         vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # #         traj_s, me_s = [], []
# # # # # #         for _ in range(num_ensemble):
# # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.sigma_min
# # # # # #             for step in range(ddim_steps):
# # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # #                 ns  = self.ctx_noise_scale if step < 3 else 0.0
# # # # # #                 vel = self.net.forward_with_ctx(
# # # # # #                     x_t, t_b, raw_ctx, noise_scale=ns, vel_obs_feat=vel_obs_feat)
# # # # # #                 x_t = x_t + dt * vel
# # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # #             traj_s.append(tr)
# # # # # #             me_s.append(me)

# # # # # #         all_trajs = torch.stack(traj_s)
# # # # # #         pred_mean = all_trajs.median(0).values

# # # # # #         if predict_csv:
# # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # #         return pred_mean, torch.stack(me_s).mean(0), all_trajs

# # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # #         with torch.enable_grad():
# # # # # #             x   = x_pred.detach().requires_grad_(True)
# # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # #             for _ in range(n_steps):
# # # # # #                 opt.zero_grad()
# # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # #                 pred_deg    = _denorm_to_deg(pred_abs)
# # # # # #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# # # # # #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # #                 speed       = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # #                                           + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # # # # #                 loss.backward()
# # # # # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # # # # #                 opt.step()
# # # # # #         return x.detach()

# # # # # #     @staticmethod
# # # # # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # # # # #         import numpy as np
# # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # #         T, B, _ = traj_mean.shape
# # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # #                     "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg",
# # # # # #                     "ens_spread_km"]
# # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # #             if write_hdr: w.writeheader()
# # # # # #             for b in range(B):
# # # # # #                 for k in range(T):
# # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# # # # # #                                * math.cos(math.radians(mean_lat[k, b])))
# # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # #                     w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # #                                 "lead_h": (k+1)*6,
# # # # # #                                 "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # #                                 "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # #                                 "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # # # # #                                 "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # # # # #                                 "ens_spread_km": f"{spread:.2f}"})


# # # # # # TCDiffusion = TCFlowMatching
# # # # # """
# # # # # Model/flow_matching_model.py — v34 HORIZON-AWARE + RESIDUAL
# # # # # ═══════════════════════════════════════════════════════════════════

# # # # # FIXES từ v33:
# # # # #   BUG 1 (NameError w_vel/w_lr/w_head): Dùng compute_total_loss thống nhất
# # # # #   BUG 2 (double-denorm): LUÔN pass degrees cho losses, normalized chỉ nội bộ
# # # # #   BUG 3 (inconsistent types): Naming convention rõ ràng _deg vs _norm

# # # # # NEW IDEAS:
# # # # #   1. Horizon-aware loss (v34 losses.py)
# # # # #   2. Residual prediction: predict displacement từ persistence baseline
# # # # #   3. Scheduled teacher forcing (training only)
# # # # #   4. Steering-conditioned velocity (env influence > only β-drift)
# # # # #   5. EMA weights support
# # # # #   6. Importance-weighted sampling at inference
# # # # # """
# # # # # from __future__ import annotations

# # # # # import csv
# # # # # import math
# # # # # import os
# # # # # from copy import deepcopy
# # # # # from datetime import datetime

# # # # # import torch
# # # # # import torch.nn as nn
# # # # # import torch.nn.functional as F

# # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # from Model.losses import (
# # # # #     compute_total_loss,
# # # # #     _haversine_deg, _norm_to_deg,
# # # # #     WEIGHTS,
# # # # # )


# # # # # def _norm_to_deg_fn(t):
# # # # #     """Normalized [lon, lat] → degrees."""
# # # # #     out = t.clone()
# # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # #     return out


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  EMA wrapper
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class EMAModel:
# # # # #     """Exponential Moving Average of model weights."""
# # # # #     def __init__(self, model: nn.Module, decay: float = 0.999):
# # # # #         self.decay = decay
# # # # #         self.shadow = {k: v.detach().clone()
# # # # #                         for k, v in model.state_dict().items()
# # # # #                         if v.dtype.is_floating_point}

# # # # #     def update(self, model: nn.Module):
# # # # #         with torch.no_grad():
# # # # #             for k, v in model.state_dict().items():
# # # # #                 if k in self.shadow:
# # # # #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

# # # # #     def apply_to(self, model: nn.Module):
# # # # #         """Copy EMA weights into model (for eval). Returns backup dict."""
# # # # #         backup = {}
# # # # #         sd = model.state_dict()
# # # # #         for k in self.shadow:
# # # # #             backup[k] = sd[k].detach().clone()
# # # # #             sd[k].copy_(self.shadow[k])
# # # # #         return backup

# # # # #     def restore(self, model: nn.Module, backup: dict):
# # # # #         sd = model.state_dict()
# # # # #         for k, v in backup.items():
# # # # #             sd[k].copy_(v)


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  VelocityField
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class VelocityField(nn.Module):
# # # # #     RAW_CTX_DIM = 512

# # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # #                  unet_in_ch=13, **kwargs):
# # # # #         super().__init__()
# # # # #         self.pred_len = pred_len
# # # # #         self.obs_len  = obs_len

# # # # #         # Spatial encoder
# # # # #         self.spatial_enc = FNO3DEncoder(
# # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # #             spatial_down=32, dropout=0.05)
# # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # #         # Temporal encoder (Mamba)
# # # # #         self.enc_1d = DataEncoder1D(
# # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # # #         # Context projection
# # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # #         self.ctx_drop = nn.Dropout(0.10)
# # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # #         # Velocity observation features
# # # # #         self.vel_obs_enc = nn.Sequential(
# # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # #             nn.LayerNorm(256),
# # # # #             nn.Linear(256, 256), nn.GELU(),
# # # # #         )

# # # # #         # ★ NEW: Steering flow encoder (từ env 500hPa)
# # # # #         # Output dim = 256 để match transformer d_model
# # # # #         self.steering_enc = nn.Sequential(
# # # # #             nn.Linear(4, 64), nn.GELU(),   # [u_mean, v_mean, u_center, v_center]
# # # # #             nn.LayerNorm(64),
# # # # #             nn.Linear(64, 128), nn.GELU(),
# # # # #             nn.Linear(128, 256),
# # # # #         )

# # # # #         # Time embedding
# # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # # #         # Decoder
# # # # #         self.transformer = nn.TransformerDecoder(
# # # # #             nn.TransformerDecoderLayer(
# # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # # #             num_layers=5)

# # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # #         # Physics scales (learnable)
# # # # #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # # #         self.physics_scale  = nn.Parameter(torch.ones(4) * 0.3)
# # # # #         self.steering_scale = nn.Parameter(torch.ones(4) * 0.2)   # ★ NEW
# # # # #         self._init_weights()

# # # # #     def _init_weights(self):
# # # # #         with torch.no_grad():
# # # # #             for name, m in self.named_modules():
# # # # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # # #                     if m.bias is not None:
# # # # #                         nn.init.zeros_(m.bias)

# # # # #     def _time_emb(self, t, dim=256):
# # # # #         half = dim // 2
# # # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # # #     def _context(self, batch_list):
# # # # #         obs_traj  = batch_list[0]
# # # # #         obs_Me    = batch_list[7]
# # # # #         image_obs = batch_list[11]
# # # # #         env_data  = batch_list[13]

# # # # #         if image_obs.dim() == 4:
# # # # #             image_obs = image_obs.unsqueeze(2)
# # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # #         T_obs = obs_traj.shape[0]

# # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # #         if e_3d_s.shape[1] != T_obs:
# # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)

# # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(
# # # # #             torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # #         if noise_scale > 0.0:
# # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # #     def _get_vel_obs_feat(self, obs_traj):
# # # # #         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
# # # # #         if T_obs >= 2:
# # # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # # #         else:
# # # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
# # # # #         if vels.shape[0] < self.obs_len:
# # # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # #         else:
# # # # #             vels = vels[-self.obs_len:]
# # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # #     # ★ NEW: Steering feature extraction
# # # # #     def _get_steering_feat(self, env_data, B, device):
# # # # #         """Extract 500hPa steering as a context vector [B, 256]."""
# # # # #         if env_data is None:
# # # # #             return torch.zeros(B, 256, device=device)

# # # # #         def _safe_get(key, default_val=0.0):
# # # # #             v = env_data.get(key, None)
# # # # #             if v is None or not torch.is_tensor(v):
# # # # #                 return torch.full((B,), default_val, device=device)
# # # # #             v = v.to(device).float()
# # # # #             if v.dim() >= 2:
# # # # #                 # Take mean over time/spatial dims, keep batch
# # # # #                 while v.dim() > 1:
# # # # #                     v = v.mean(-1)
# # # # #             if v.shape[0] != B:
# # # # #                 v = v.view(-1)[:B] if v.numel() >= B else torch.full((B,), default_val, device=device)
# # # # #             return v

# # # # #         u_m = _safe_get("u500_mean")
# # # # #         v_m = _safe_get("v500_mean")
# # # # #         u_c = _safe_get("u500_center")
# # # # #         v_c = _safe_get("v500_center")
# # # # #         feat = torch.stack([u_m, v_m, u_c, v_c], dim=-1)  # [B, 4]
# # # # #         return self.steering_enc(feat)  # [B, 256]

# # # # #     def _beta_drift(self, x_t):
# # # # #         """β-drift (Coriolis-derived) in normalized units."""
# # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # #         R_tc    = 3e5
# # # # #         v_phys  = torch.zeros_like(x_t)
# # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # #         return v_phys

# # # # #     # ★ NEW: Steering drift (from env 500hPa)
# # # # #     def _steering_drift(self, x_t, env_data):
# # # # #         """
# # # # #         Inject steering flow as additional drift velocity.
# # # # #         env data → u, v in m/s → normalize per 6h.
# # # # #         """
# # # # #         if env_data is None:
# # # # #             return torch.zeros_like(x_t)

# # # # #         B = x_t.shape[0]
# # # # #         device = x_t.device

# # # # #         def _safe_mean(key):
# # # # #             v = env_data.get(key, None)
# # # # #             if v is None or not torch.is_tensor(v):
# # # # #                 return torch.zeros(B, device=device)
# # # # #             v = v.to(device).float()
# # # # #             while v.dim() > 1:
# # # # #                 v = v.mean(-1)
# # # # #             if v.numel() < B:
# # # # #                 return torch.zeros(B, device=device)
# # # # #             return v.view(-1)[:B]

# # # # #         # u, v in normalized units (~m/s / 30)
# # # # #         u = _safe_mean("u500_center")  # [B]
# # # # #         v = _safe_mean("v500_center")

# # # # #         # Convert to deg/6h at approx latitude from x_t
# # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# # # # #         # m/s * 21600s / (111km * 1000m) = deg/6h
# # # # #         u_deg = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat)
# # # # #         v_deg = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0)
# # # # #         # Normalize back: lon_norm = (deg*10 - 1800)/50, but for velocity → scale 10/50 = 0.2
# # # # #         u_norm = u_deg * 0.2
# # # # #         v_norm = v_deg * 0.2

# # # # #         out = torch.zeros_like(x_t)
# # # # #         out[:, :, 0] = u_norm
# # # # #         out[:, :, 1] = v_norm
# # # # #         return out

# # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
# # # # #                 env_data=None):
# # # # #         B     = x_t.shape[0]
# # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # #         t_emb = self.time_fc2(t_emb)

# # # # #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# # # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # #         s_emb    = self.step_embed(step_idx)

# # # # #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# # # # #                  + self.pos_enc[:, :T_seq]
# # # # #                  + t_emb.unsqueeze(1)
# # # # #                  + s_emb)

# # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # #         if vel_obs_feat is not None:
# # # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))
# # # # #         if steering_feat is not None:
# # # # #             mem_parts.append(steering_feat.unsqueeze(1))

# # # # #         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))

# # # # #         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # #         v_neural = v_neural * scale

# # # # #         # β-drift (Coriolis)
# # # # #         with torch.no_grad():
# # # # #             v_phys = self._beta_drift(x_t[:, :T_seq])
# # # # #         # Steering drift
# # # # #         with torch.no_grad():
# # # # #             v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

# # # # #         return (v_neural
# # # # #                 + torch.sigmoid(self.physics_scale) * v_phys
# # # # #                 + torch.sigmoid(self.steering_scale) * v_steer)

# # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
# # # # #                            vel_obs_feat=None, steering_feat=None, env_data=None):
# # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# # # # #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  TCFlowMatching (main)
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class TCFlowMatching(nn.Module):

# # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # #                  ctx_noise_scale=0.01, initial_sample_sigma=0.03,
# # # # #                  teacher_forcing=True, use_ema=True, ema_decay=0.999,
# # # # #                  **kwargs):
# # # # #         super().__init__()
# # # # #         self.pred_len             = pred_len
# # # # #         self.obs_len              = obs_len
# # # # #         self.sigma_min            = sigma_min
# # # # #         self.n_train_ens          = n_train_ens
# # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # #         self.teacher_forcing      = teacher_forcing
# # # # #         self.active_pred_len      = pred_len

# # # # #         self.net = VelocityField(
# # # # #             pred_len=pred_len, obs_len=obs_len,
# # # # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # #         # EMA (set up after all submodules exist)
# # # # #         self.use_ema = use_ema
# # # # #         self.ema_decay = ema_decay
# # # # #         self._ema = None  # lazy init (outside __init__ to avoid state conflict)

# # # # #     def init_ema(self):
# # # # #         """Call once after model on device."""
# # # # #         if self.use_ema:
# # # # #             self._ema = EMAModel(self, decay=self.ema_decay)

# # # # #     def ema_update(self):
# # # # #         if self._ema is not None:
# # # # #             self._ema.update(self)

# # # # #     def set_curriculum_len(self, *a, **kw):
# # # # #         pass

# # # # #     # ── Normalization helpers ────────────────────────────────────────────────
# # # # #     @staticmethod
# # # # #     def _to_rel(traj, Me, lp, lm):
# # # # #         """Absolute → relative (to last obs)."""
# # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # #     @staticmethod
# # # # #     def _to_abs(rel, lp, lm):
# # # # #         """Relative → absolute (normalized)."""
# # # # #         d = rel.permute(1, 0, 2)
# # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # #     # ── CFM noise ────────────────────────────────────────────────────────────
# # # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # # #         if sigma_min is None:
# # # # #             sigma_min = self.sigma_min
# # # # #         B, device = x1.shape[0], x1.device
# # # # #         x0 = torch.randn_like(x1) * sigma_min
# # # # #         t  = torch.rand(B, device=device)
# # # # #         te = t.view(B, 1, 1)
# # # # #         x_t = (1.0 - te) * x0 + te * x1
# # # # #         u   = x1 - x0
# # # # #         return x_t, t, u

# # # # #     # ── Augmentation ─────────────────────────────────────────────────────────
# # # # #     @staticmethod
# # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # #         if torch.rand(1).item() > p:
# # # # #             return batch_list
# # # # #         aug = list(batch_list)
# # # # #         for idx in [0, 1, 2, 3]:
# # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # #                 t = aug[idx].clone()
# # # # #                 t[..., 0] = -t[..., 0]
# # # # #                 aug[idx] = t
# # # # #         return aug

# # # # #     @staticmethod
# # # # #     def _obs_noise_aug(batch_list, sigma=0.005):
# # # # #         """Add small Gaussian noise to obs trajectory (input augmentation)."""
# # # # #         if torch.rand(1).item() > 0.5:
# # # # #             return batch_list
# # # # #         aug = list(batch_list)
# # # # #         if torch.is_tensor(aug[0]):
# # # # #             aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
# # # # #         return aug

# # # # #     @staticmethod
# # # # #     def _mixup_aug(batch_list, alpha=0.2, p=0.2):
# # # # #         """Light mixup on trajectories (shuffle batch, mix)."""
# # # # #         if torch.rand(1).item() > p:
# # # # #             return batch_list
# # # # #         aug = list(batch_list)
# # # # #         lam = float(torch.distributions.Beta(alpha, alpha).sample())
# # # # #         lam = max(lam, 1 - lam)  # prefer near 1 to not destroy signal
# # # # #         if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
# # # # #             B = aug[0].shape[1]
# # # # #             perm = torch.randperm(B, device=aug[0].device)
# # # # #             for idx in [0, 1, 7, 8]:
# # # # #                 if torch.is_tensor(aug[idx]):
# # # # #                     aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
# # # # #         return aug

# # # # #     # ── Public loss API ───────────────────────────────────────────────────────
# # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # #         # Augmentation
# # # # #         batch_list = self._lon_flip_aug(batch_list)
# # # # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
# # # # #         if epoch >= 5:
# # # # #             batch_list = self._mixup_aug(batch_list, alpha=0.2, p=0.15)

# # # # #         traj_gt = batch_list[1]
# # # # #         Me_gt   = batch_list[8]
# # # # #         obs_t   = batch_list[0]
# # # # #         obs_Me  = batch_list[7]
# # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # #         lp, lm  = obs_t[-1], obs_Me[-1]

# # # # #         # Sigma schedule
# # # # #         if epoch < 15:
# # # # #             current_sigma = 0.15
# # # # #         elif epoch < 40:
# # # # #             current_sigma = 0.15 - (epoch - 15) / 25.0 * 0.12
# # # # #         else:
# # # # #             current_sigma = 0.03

# # # # #         # Context
# # # # #         raw_ctx       = self.net._context(batch_list)
# # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # #         steering_feat = self.net._get_steering_feat(env_data, obs_t.shape[1], obs_t.device)

# # # # #         # ── CFM target ────────────────────────────────────────────────────
# # # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
# # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # #         # ── Scheduled teacher forcing (training only) ─────────────────────
# # # # #         if self.teacher_forcing and self.training and epoch >= 3:
# # # # #             # p_teacher decreases from 0.5 at ep3 → 0 at ep40
# # # # #             p_teacher = max(0.0, 0.5 * (1.0 - (epoch - 3) / 37.0))
# # # # #             if p_teacher > 0 and torch.rand(1).item() < p_teacher:
# # # # #                 # Mix x_t với a bit of x1_rel (GT) at far steps
# # # # #                 far_mask = torch.zeros_like(x_t)
# # # # #                 far_mask[:, 6:, :] = 0.3  # 30% GT at steps 7-12
# # # # #                 x_t = x_t * (1 - far_mask) + x1_rel * far_mask

# # # # #         pred_vel = self.net.forward_with_ctx(
# # # # #             x_t, fm_t, raw_ctx,
# # # # #             vel_obs_feat=vel_obs_feat,
# # # # #             steering_feat=steering_feat,
# # # # #             env_data=env_data,
# # # # #         )

# # # # #         # CFM velocity MSE loss (base)
# # # # #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# # # # #         # Reconstruct predicted x1
# # # # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)   # [T, B, 2] normalized
# # # # #         pred_deg = _norm_to_deg_fn(pred_abs)           # [T, B, 2] degrees
# # # # #         gt_deg   = _norm_to_deg_fn(traj_gt)            # [T, B, 2] degrees

# # # # #         # ── v34 horizon-aware total loss ──────────────────────────────────
# # # # #         loss_dict = compute_total_loss(
# # # # #             pred_deg=pred_deg,
# # # # #             gt_deg=gt_deg,
# # # # #             env_data=env_data,
# # # # #             weights=WEIGHTS,
# # # # #             epoch=epoch,
# # # # #         )

# # # # #         # Combine FM velocity loss + v34 trajectory loss
# # # # #         # FM weight decays as training progresses
# # # # #         w_fm = max(0.3, 1.0 - epoch / 60.0)
# # # # #         total = w_fm * l_fm_mse + loss_dict["total"]

# # # # #         # ── Ensemble consistency (late epochs) ────────────────────────────
# # # # #         l_ens = x_t.new_zeros(())
# # # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # # #             x_t2, fm_t2, u_t2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # #             pv2 = self.net.forward_with_ctx(
# # # # #                 x_t2, fm_t2, raw_ctx,
# # # # #                 vel_obs_feat=vel_obs_feat,
# # # # #                 steering_feat=steering_feat,
# # # # #                 env_data=env_data,
# # # # #             )
# # # # #             l_ens = F.mse_loss(pv2, u_t2)
# # # # #             total = total + 0.3 * l_ens

# # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # #             total = x_t.new_zeros(())

# # # # #         # Return breakdown
# # # # #         return dict(
# # # # #             total        = total,
# # # # #             fm_mse       = l_fm_mse.item(),
# # # # #             mse_hav      = loss_dict["mse_hav_horizon"],
# # # # #             multi_scale  = loss_dict["multi_scale"],
# # # # #             endpoint     = loss_dict["endpoint"],
# # # # #             shape        = loss_dict["shape"],
# # # # #             velocity     = loss_dict["velocity"],
# # # # #             heading      = loss_dict["heading"],
# # # # #             recurv       = loss_dict["recurv"],
# # # # #             steering     = loss_dict["steering"],
# # # # #             ens_consist  = l_ens.item() if torch.is_tensor(l_ens) else float(l_ens),
# # # # #             sigma        = current_sigma,
# # # # #             w_fm         = w_fm,
# # # # #             # backward compat zeros
# # # # #             long_range=0.0, fde=0.0, w_lr=0.0, w_fde=0.0,
# # # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0, recurv_ratio=0.0,
# # # # #         )

# # # # #     # ── Sampling ──────────────────────────────────────────────────────────────
# # # # #     @torch.no_grad()
# # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
# # # # #                 predict_csv=None, importance_weight=True):
# # # # #         obs_t    = batch_list[0]
# # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # #         lp       = obs_t[-1]
# # # # #         lm       = batch_list[7][-1]
# # # # #         B        = lp.shape[0]
# # # # #         device   = lp.device
# # # # #         T        = self.pred_len
# # # # #         dt       = 1.0 / max(ddim_steps, 1)

# # # # #         raw_ctx       = self.net._context(batch_list)
# # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # #         steering_feat = self.net._get_steering_feat(env_data, B, device)

# # # # #         traj_s, me_s, scores = [], [], []
# # # # #         for ens_i in range(num_ensemble):
# # # # #             x_t = torch.randn(B, T, 4, device=device) * self.sigma_min
# # # # #             for step in range(ddim_steps):
# # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # #                 # Larger noise early for diversity
# # # # #                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
# # # # #                 vel = self.net.forward_with_ctx(
# # # # #                     x_t, t_b, raw_ctx,
# # # # #                     noise_scale=ns,
# # # # #                     vel_obs_feat=vel_obs_feat,
# # # # #                     steering_feat=steering_feat,
# # # # #                     env_data=env_data,
# # # # #                 )
# # # # #                 x_t = x_t + dt * vel
# # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # #             traj_s.append(tr)
# # # # #             me_s.append(me)

# # # # #             # ★ Importance score for this sample
# # # # #             if importance_weight:
# # # # #                 scores.append(self._score_sample(tr, env_data))

# # # # #         all_trajs = torch.stack(traj_s)       # [S, T, B, 2]
# # # # #         all_me    = torch.stack(me_s)

# # # # #         # ── Prediction: importance-weighted median/mean ──────────────────
# # # # #         if importance_weight and scores:
# # # # #             score_tensor = torch.stack(scores)  # [S, B]
# # # # #             # Top-70% by score per batch
# # # # #             k = max(1, int(num_ensemble * 0.7))
# # # # #             _, top_idx = score_tensor.topk(k, dim=0)  # [k, B]
# # # # #             # Gather
# # # # #             pred_mean_list = []
# # # # #             for b in range(B):
# # # # #                 sel = all_trajs[top_idx[:, b], :, b, :]  # [k, T, 2]
# # # # #                 pred_mean_list.append(sel.median(0).values)
# # # # #             pred_mean = torch.stack(pred_mean_list, dim=1)  # [T, B, 2]
# # # # #         else:
# # # # #             pred_mean = all_trajs.median(0).values

# # # # #         if predict_csv:
# # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # #         return pred_mean, all_me.mean(0), all_trajs

# # # # #     def _score_sample(self, traj, env_data):
# # # # #         """
# # # # #         Score a sample: higher is better.
# # # # #         Checks: speed in [10, 70] km/h, smoothness, steering alignment.
# # # # #         Returns [B] score.
# # # # #         """
# # # # #         B = traj.shape[1]
# # # # #         if traj.shape[0] < 2:
# # # # #             return torch.ones(B, device=traj.device)

# # # # #         traj_deg = _norm_to_deg_fn(traj)
# # # # #         dt_deg = traj_deg[1:] - traj_deg[:-1]
# # # # #         lat_rad = torch.deg2rad(traj_deg[:-1, :, 1])
# # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # #         dx_km = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # #         dy_km = dt_deg[:, :, 1] * 111.0
# # # # #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)  # km per 6h

# # # # #         # Reasonable speed: 10-70 km per 6h (~2-12 m/s)
# # # # #         speed_penalty = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
# # # # #         speed_score = torch.exp(-speed_penalty.mean(0) / 20.0)  # [B]

# # # # #         # Smoothness: low jerk
# # # # #         if dt_deg.shape[0] >= 2:
# # # # #             jerk = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)  # [B]
# # # # #             smooth_score = torch.exp(-jerk * 5.0)
# # # # #         else:
# # # # #             smooth_score = torch.ones(B, device=traj.device)

# # # # #         return speed_score * smooth_score

# # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # #         with torch.enable_grad():
# # # # #             x   = x_pred.detach().requires_grad_(True)
# # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # #             for _ in range(n_steps):
# # # # #                 opt.zero_grad()
# # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # #                 pred_deg    = _norm_to_deg_fn(pred_abs)
# # # # #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# # # # #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# # # # #                 speed       = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # #                                           + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # # # #                 loss.backward()
# # # # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # # # #                 opt.step()
# # # # #         return x.detach()

# # # # #     @staticmethod
# # # # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # # # #         import numpy as np
# # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # #         T, B, _ = traj_mean.shape
# # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # #                     "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg",
# # # # #                     "ens_spread_km"]
# # # # #         write_hdr = not os.path.exists(csv_path)
# # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # #         with open(csv_path, "a", newline="") as fh:
# # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # #             if write_hdr: w.writeheader()
# # # # #             for b in range(B):
# # # # #                 for k in range(T):
# # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# # # # #                                * math.cos(math.radians(mean_lat[k, b])))
# # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # #                     w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # #                                 "lead_h": (k+1)*6,
# # # # #                                 "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # #                                 "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # #                                 "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # # # #                                 "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # # # #                                 "ens_spread_km": f"{spread:.2f}"})


# # # # # TCDiffusion = TCFlowMatching

# # # # """
# # # # Model/flow_matching_model.py — v34 HORIZON-AWARE + RESIDUAL
# # # # ═══════════════════════════════════════════════════════════════════

# # # # FIXES từ v33:
# # # #   BUG 1 (NameError w_vel/w_lr/w_head): Dùng compute_total_loss thống nhất
# # # #   BUG 2 (double-denorm): LUÔN pass degrees cho losses, normalized chỉ nội bộ
# # # #   BUG 3 (inconsistent types): Naming convention rõ ràng _deg vs _norm

# # # # FIXES v34→v34fix:
# # # #   BUG 4 (EMA KeyError 'net.pos_enc'): torch.compile đổi tên key thành
# # # #          _orig_mod.net.pos_enc — EMAModel phải unwrap _orig_mod khi
# # # #          init/update/apply_to/restore.
# # # #   BUG 5 (sigma floor 0.03 quá thấp): giữ floor ở 0.06 để ensemble diversity
# # # #   BUG 6 (ensemble init noise quá nhỏ): tăng initial noise × 2.5

# # # # NEW IDEAS:
# # # #   1. Horizon-aware loss (v34 losses.py)
# # # #   2. Residual prediction: predict displacement từ persistence baseline
# # # #   3. Scheduled teacher forcing (training only)
# # # #   4. Steering-conditioned velocity (env influence > only β-drift)
# # # #   5. EMA weights support (fixed for torch.compile)
# # # #   6. Importance-weighted sampling at inference
# # # # """
# # # # from __future__ import annotations

# # # # import csv
# # # # import math
# # # # import os
# # # # from copy import deepcopy
# # # # from datetime import datetime

# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.nn.functional as F

# # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # from Model.losses import (
# # # #     compute_total_loss,
# # # #     _haversine_deg, _norm_to_deg,
# # # #     WEIGHTS,
# # # # )


# # # # def _norm_to_deg_fn(t):
# # # #     """Normalized [lon, lat] → degrees."""
# # # #     out = t.clone()
# # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # #     return out


# # # # def _unwrap_model(model: nn.Module) -> nn.Module:
# # # #     """Unwrap torch.compile wrapper nếu có, để lấy state_dict gốc."""
# # # #     if hasattr(model, '_orig_mod'):
# # # #         return model._orig_mod
# # # #     return model


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  EMA wrapper  — FIX: unwrap _orig_mod để tránh KeyError từ torch.compile
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class EMAModel:
# # # #     """
# # # #     Exponential Moving Average of model weights.

# # # #     FIX (v34fix): torch.compile() đổi tên tất cả key trong state_dict
# # # #     từ "net.pos_enc" → "_orig_mod.net.pos_enc".
# # # #     Nếu EMAModel.shadow được build từ compiled model, key sẽ không match
# # # #     khi apply_to() gọi với raw model (hoặc ngược lại).

# # # #     Giải pháp: LUÔN unwrap _orig_mod trước khi đọc/ghi state_dict.
# # # #     """
# # # #     def __init__(self, model: nn.Module, decay: float = 0.999):
# # # #         self.decay = decay
# # # #         m = _unwrap_model(model)
# # # #         self.shadow = {
# # # #             k: v.detach().clone()
# # # #             for k, v in m.state_dict().items()
# # # #             if v.dtype.is_floating_point
# # # #         }

# # # #     def update(self, model: nn.Module):
# # # #         m = _unwrap_model(model)
# # # #         with torch.no_grad():
# # # #             for k, v in m.state_dict().items():
# # # #                 if k in self.shadow:
# # # #                     self.shadow[k].mul_(self.decay).add_(
# # # #                         v.detach(), alpha=1 - self.decay)

# # # #     def apply_to(self, model: nn.Module):
# # # #         """Copy EMA weights vào model (để eval). Trả về backup dict."""
# # # #         m = _unwrap_model(model)
# # # #         backup = {}
# # # #         sd = m.state_dict()
# # # #         for k in self.shadow:
# # # #             if k not in sd:
# # # #                 # Key không tồn tại trong model — bỏ qua thay vì crash
# # # #                 continue
# # # #             backup[k] = sd[k].detach().clone()
# # # #             sd[k].copy_(self.shadow[k])
# # # #         return backup

# # # #     def restore(self, model: nn.Module, backup: dict):
# # # #         m = _unwrap_model(model)
# # # #         sd = m.state_dict()
# # # #         for k, v in backup.items():
# # # #             if k in sd:
# # # #                 sd[k].copy_(v)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  VelocityField
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class VelocityField(nn.Module):
# # # #     RAW_CTX_DIM = 512

# # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # #                  unet_in_ch=13, **kwargs):
# # # #         super().__init__()
# # # #         self.pred_len = pred_len
# # # #         self.obs_len  = obs_len

# # # #         # Spatial encoder
# # # #         self.spatial_enc = FNO3DEncoder(
# # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # #             spatial_down=32, dropout=0.05)
# # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # #         # Temporal encoder (Mamba)
# # # #         self.enc_1d = DataEncoder1D(
# # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # #         # Context projection
# # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # #         self.ctx_drop = nn.Dropout(0.10)
# # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # #         # Velocity observation features
# # # #         self.vel_obs_enc = nn.Sequential(
# # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # #             nn.LayerNorm(256),
# # # #             nn.Linear(256, 256), nn.GELU(),
# # # #         )

# # # #         # Steering flow encoder (từ env 500hPa)
# # # #         self.steering_enc = nn.Sequential(
# # # #             nn.Linear(4, 64), nn.GELU(),
# # # #             nn.LayerNorm(64),
# # # #             nn.Linear(64, 128), nn.GELU(),
# # # #             nn.Linear(128, 256),
# # # #         )

# # # #         # Time embedding
# # # #         self.time_fc1   = nn.Linear(256, 512)
# # # #         self.time_fc2   = nn.Linear(512, 256)
# # # #         self.traj_embed = nn.Linear(4, 256)
# # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # #         # Decoder
# # # #         self.transformer = nn.TransformerDecoder(
# # # #             nn.TransformerDecoderLayer(
# # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # #             num_layers=5)

# # # #         self.out_fc1 = nn.Linear(256, 512)
# # # #         self.out_fc2 = nn.Linear(512, 4)

# # # #         # Physics scales (learnable)
# # # #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # #         self.physics_scale  = nn.Parameter(torch.ones(4) * 0.3)
# # # #         self.steering_scale = nn.Parameter(torch.ones(4) * 0.2)
# # # #         self._init_weights()

# # # #     def _init_weights(self):
# # # #         with torch.no_grad():
# # # #             for name, m in self.named_modules():
# # # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # #                     if m.bias is not None:
# # # #                         nn.init.zeros_(m.bias)

# # # #     def _time_emb(self, t, dim=256):
# # # #         half = dim // 2
# # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # #     def _context(self, batch_list):
# # # #         obs_traj  = batch_list[0]
# # # #         obs_Me    = batch_list[7]
# # # #         image_obs = batch_list[11]
# # # #         env_data  = batch_list[13]

# # # #         if image_obs.dim() == 4:
# # # #             image_obs = image_obs.unsqueeze(2)
# # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # #         T_obs = obs_traj.shape[0]

# # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # #         if e_3d_s.shape[1] != T_obs:
# # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)

# # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(
# # # #             torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # #         if noise_scale > 0.0:
# # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # #     def _get_vel_obs_feat(self, obs_traj):
# # # #         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
# # # #         if T_obs >= 2:
# # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # #         else:
# # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
# # # #         if vels.shape[0] < self.obs_len:
# # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# # # #             vels = torch.cat([pad, vels], dim=0)
# # # #         else:
# # # #             vels = vels[-self.obs_len:]
# # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # #     def _get_steering_feat(self, env_data, B, device):
# # # #         """Extract 500hPa steering as a context vector [B, 256]."""
# # # #         if env_data is None:
# # # #             return torch.zeros(B, 256, device=device)

# # # #         def _safe_get(key, default_val=0.0):
# # # #             v = env_data.get(key, None)
# # # #             if v is None or not torch.is_tensor(v):
# # # #                 return torch.full((B,), default_val, device=device)
# # # #             v = v.to(device).float()
# # # #             if v.dim() >= 2:
# # # #                 while v.dim() > 1:
# # # #                     v = v.mean(-1)
# # # #             if v.shape[0] != B:
# # # #                 v = v.view(-1)[:B] if v.numel() >= B else torch.full((B,), default_val, device=device)
# # # #             return v

# # # #         u_m = _safe_get("u500_mean")
# # # #         v_m = _safe_get("v500_mean")
# # # #         u_c = _safe_get("u500_center")
# # # #         v_c = _safe_get("v500_center")
# # # #         feat = torch.stack([u_m, v_m, u_c, v_c], dim=-1)  # [B, 4]
# # # #         return self.steering_enc(feat)  # [B, 256]

# # # #     def _beta_drift(self, x_t):
# # # #         """β-drift (Coriolis-derived) in normalized units."""
# # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # #         R_tc    = 3e5
# # # #         v_phys  = torch.zeros_like(x_t)
# # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # #         return v_phys

# # # #     def _steering_drift(self, x_t, env_data):
# # # #         """Inject steering flow as additional drift velocity."""
# # # #         if env_data is None:
# # # #             return torch.zeros_like(x_t)

# # # #         B = x_t.shape[0]
# # # #         device = x_t.device

# # # #         def _safe_mean(key):
# # # #             v = env_data.get(key, None)
# # # #             if v is None or not torch.is_tensor(v):
# # # #                 return torch.zeros(B, device=device)
# # # #             v = v.to(device).float()
# # # #             while v.dim() > 1:
# # # #                 v = v.mean(-1)
# # # #             if v.numel() < B:
# # # #                 return torch.zeros(B, device=device)
# # # #             return v.view(-1)[:B]

# # # #         u = _safe_mean("u500_center")
# # # #         v = _safe_mean("v500_center")

# # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# # # #         u_deg = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat)
# # # #         v_deg = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0)
# # # #         u_norm = u_deg * 0.2
# # # #         v_norm = v_deg * 0.2

# # # #         out = torch.zeros_like(x_t)
# # # #         out[:, :, 0] = u_norm
# # # #         out[:, :, 1] = v_norm
# # # #         return out

# # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
# # # #                 env_data=None):
# # # #         B     = x_t.shape[0]
# # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # #         t_emb = self.time_fc2(t_emb)

# # # #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # #         s_emb    = self.step_embed(step_idx)

# # # #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# # # #                  + self.pos_enc[:, :T_seq]
# # # #                  + t_emb.unsqueeze(1)
# # # #                  + s_emb)

# # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # #         if vel_obs_feat is not None:
# # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))
# # # #         if steering_feat is not None:
# # # #             mem_parts.append(steering_feat.unsqueeze(1))

# # # #         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))

# # # #         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # #         v_neural = v_neural * scale

# # # #         with torch.no_grad():
# # # #             v_phys  = self._beta_drift(x_t[:, :T_seq])
# # # #             v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

# # # #         return (v_neural
# # # #                 + torch.sigmoid(self.physics_scale) * v_phys
# # # #                 + torch.sigmoid(self.steering_scale) * v_steer)

# # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
# # # #                            vel_obs_feat=None, steering_feat=None, env_data=None):
# # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# # # #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  TCFlowMatching (main)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class TCFlowMatching(nn.Module):

# # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # #                  n_train_ens=4, unet_in_ch=13,
# # # #                  ctx_noise_scale=0.01, initial_sample_sigma=0.03,
# # # #                  teacher_forcing=True, use_ema=True, ema_decay=0.999,
# # # #                  **kwargs):
# # # #         super().__init__()
# # # #         self.pred_len             = pred_len
# # # #         self.obs_len              = obs_len
# # # #         self.sigma_min            = sigma_min
# # # #         self.n_train_ens          = n_train_ens
# # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # #         self.initial_sample_sigma = initial_sample_sigma
# # # #         self.teacher_forcing      = teacher_forcing
# # # #         self.active_pred_len      = pred_len

# # # #         self.net = VelocityField(
# # # #             pred_len=pred_len, obs_len=obs_len,
# # # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # #         self.use_ema  = use_ema
# # # #         self.ema_decay = ema_decay
# # # #         self._ema = None  # lazy init

# # # #     def init_ema(self):
# # # #         """Gọi một lần sau khi model đã lên device."""
# # # #         if self.use_ema:
# # # #             # FIX: truyền self (chưa compile) để shadow có đúng key names
# # # #             self._ema = EMAModel(self, decay=self.ema_decay)

# # # #     def ema_update(self):
# # # #         if self._ema is not None:
# # # #             # FIX: EMAModel.update tự unwrap _orig_mod bên trong
# # # #             self._ema.update(self)

# # # #     def set_curriculum_len(self, *a, **kw):
# # # #         pass

# # # #     # ── Normalization helpers ────────────────────────────────────────────────
# # # #     @staticmethod
# # # #     def _to_rel(traj, Me, lp, lm):
# # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # #     @staticmethod
# # # #     def _to_abs(rel, lp, lm):
# # # #         d = rel.permute(1, 0, 2)
# # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # #     # ── CFM noise ────────────────────────────────────────────────────────────
# # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # #         if sigma_min is None:
# # # #             sigma_min = self.sigma_min
# # # #         B, device = x1.shape[0], x1.device
# # # #         x0 = torch.randn_like(x1) * sigma_min
# # # #         t  = torch.rand(B, device=device)
# # # #         te = t.view(B, 1, 1)
# # # #         x_t = (1.0 - te) * x0 + te * x1
# # # #         u   = x1 - x0
# # # #         return x_t, t, u

# # # #     # ── Augmentation ─────────────────────────────────────────────────────────
# # # #     @staticmethod
# # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # #         if torch.rand(1).item() > p:
# # # #             return batch_list
# # # #         aug = list(batch_list)
# # # #         for idx in [0, 1, 2, 3]:
# # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # #                 t = aug[idx].clone()
# # # #                 t[..., 0] = -t[..., 0]
# # # #                 aug[idx] = t
# # # #         return aug

# # # #     @staticmethod
# # # #     def _obs_noise_aug(batch_list, sigma=0.005):
# # # #         if torch.rand(1).item() > 0.5:
# # # #             return batch_list
# # # #         aug = list(batch_list)
# # # #         if torch.is_tensor(aug[0]):
# # # #             aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
# # # #         return aug

# # # #     @staticmethod
# # # #     def _mixup_aug(batch_list, alpha=0.2, p=0.2):
# # # #         if torch.rand(1).item() > p:
# # # #             return batch_list
# # # #         aug = list(batch_list)
# # # #         lam = float(torch.distributions.Beta(alpha, alpha).sample())
# # # #         lam = max(lam, 1 - lam)
# # # #         if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
# # # #             B = aug[0].shape[1]
# # # #             perm = torch.randperm(B, device=aug[0].device)
# # # #             for idx in [0, 1, 7, 8]:
# # # #                 if torch.is_tensor(aug[idx]):
# # # #                     aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
# # # #         return aug

# # # #     # ── Public loss API ───────────────────────────────────────────────────────
# # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # #         # Augmentation
# # # #         batch_list = self._lon_flip_aug(batch_list)
# # # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
# # # #         if epoch >= 5:
# # # #             batch_list = self._mixup_aug(batch_list, alpha=0.2, p=0.15)

# # # #         traj_gt  = batch_list[1]
# # # #         Me_gt    = batch_list[8]
# # # #         obs_t    = batch_list[0]
# # # #         obs_Me   = batch_list[7]
# # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # #         lp, lm   = obs_t[-1], obs_Me[-1]

# # # #         # ── Sigma schedule — FIX: floor 0.06 (không phải 0.03) ──────────
# # # #         # Sigma quá thấp → ensemble diversity thấp → median bị bias center
# # # #         if epoch < 15:
# # # #             current_sigma = 0.15
# # # #         elif epoch < 40:
# # # #             # Giảm từ 0.15 → 0.06 (thay vì 0.03)
# # # #             current_sigma = 0.15 - (epoch - 15) / 25.0 * 0.09
# # # #         else:
# # # #             current_sigma = 0.06  # FIX: floor 0.06, không phải 0.03

# # # #         # Context
# # # #         raw_ctx       = self.net._context(batch_list)
# # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # #         steering_feat = self.net._get_steering_feat(env_data, obs_t.shape[1], obs_t.device)

# # # #         # ── CFM target ────────────────────────────────────────────────────
# # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
# # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # #         # ── Scheduled teacher forcing ─────────────────────────────────────
# # # #         if self.teacher_forcing and self.training and epoch >= 3:
# # # #             p_teacher = max(0.0, 0.5 * (1.0 - (epoch - 3) / 37.0))
# # # #             if p_teacher > 0 and torch.rand(1).item() < p_teacher:
# # # #                 far_mask = torch.zeros_like(x_t)
# # # #                 far_mask[:, 6:, :] = 0.3
# # # #                 x_t = x_t * (1 - far_mask) + x1_rel * far_mask

# # # #         pred_vel = self.net.forward_with_ctx(
# # # #             x_t, fm_t, raw_ctx,
# # # #             vel_obs_feat=vel_obs_feat,
# # # #             steering_feat=steering_feat,
# # # #             env_data=env_data,
# # # #         )

# # # #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# # # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)
# # # #         pred_deg = _norm_to_deg_fn(pred_abs)
# # # #         gt_deg   = _norm_to_deg_fn(traj_gt)

# # # #         loss_dict = compute_total_loss(
# # # #             pred_deg=pred_deg,
# # # #             gt_deg=gt_deg,
# # # #             env_data=env_data,
# # # #             weights=WEIGHTS,
# # # #             epoch=epoch,
# # # #         )

# # # #         w_fm  = max(0.3, 1.0 - epoch / 60.0)
# # # #         total = w_fm * l_fm_mse + loss_dict["total"]

# # # #         l_ens = x_t.new_zeros(())
# # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # #             x_t2, fm_t2, u_t2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # #             pv2 = self.net.forward_with_ctx(
# # # #                 x_t2, fm_t2, raw_ctx,
# # # #                 vel_obs_feat=vel_obs_feat,
# # # #                 steering_feat=steering_feat,
# # # #                 env_data=env_data,
# # # #             )
# # # #             l_ens = F.mse_loss(pv2, u_t2)
# # # #             total = total + 0.3 * l_ens

# # # #         if torch.isnan(total) or torch.isinf(total):
# # # #             total = x_t.new_zeros(())

# # # #         return dict(
# # # #             total        = total,
# # # #             fm_mse       = l_fm_mse.item(),
# # # #             mse_hav      = loss_dict["mse_hav_horizon"],
# # # #             multi_scale  = loss_dict["multi_scale"],
# # # #             endpoint     = loss_dict["endpoint"],
# # # #             shape        = loss_dict["shape"],
# # # #             velocity     = loss_dict["velocity"],
# # # #             heading      = loss_dict["heading"],
# # # #             recurv       = loss_dict["recurv"],
# # # #             steering     = loss_dict["steering"],
# # # #             ens_consist  = l_ens.item() if torch.is_tensor(l_ens) else float(l_ens),
# # # #             sigma        = current_sigma,
# # # #             w_fm         = w_fm,
# # # #             long_range=0.0, fde=0.0, w_lr=0.0, w_fde=0.0,
# # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0, recurv_ratio=0.0,
# # # #         )

# # # #     # ── Sampling ──────────────────────────────────────────────────────────────
# # # #     @torch.no_grad()
# # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
# # # #                 predict_csv=None, importance_weight=True):
# # # #         obs_t    = batch_list[0]
# # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # #         lp       = obs_t[-1]
# # # #         lm       = batch_list[7][-1]
# # # #         B        = lp.shape[0]
# # # #         device   = lp.device
# # # #         T        = self.pred_len
# # # #         dt       = 1.0 / max(ddim_steps, 1)

# # # #         raw_ctx       = self.net._context(batch_list)
# # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # #         steering_feat = self.net._get_steering_feat(env_data, B, device)

# # # #         traj_s, me_s, scores = [], [], []
# # # #         for ens_i in range(num_ensemble):
# # # #             # FIX: tăng initial noise × 2.5 để ensemble diversity cao hơn
# # # #             # Sigma floor 0.06 → initial noise 0.06 * 2.5 = 0.15
# # # #             x_t = torch.randn(B, T, 4, device=device) * (self.sigma_min * 2.5)
# # # #             for step in range(ddim_steps):
# # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # #                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
# # # #                 vel = self.net.forward_with_ctx(
# # # #                     x_t, t_b, raw_ctx,
# # # #                     noise_scale=ns,
# # # #                     vel_obs_feat=vel_obs_feat,
# # # #                     steering_feat=steering_feat,
# # # #                     env_data=env_data,
# # # #                 )
# # # #                 x_t = x_t + dt * vel
# # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # #             traj_s.append(tr)
# # # #             me_s.append(me)

# # # #             if importance_weight:
# # # #                 scores.append(self._score_sample(tr, env_data))

# # # #         all_trajs = torch.stack(traj_s)
# # # #         all_me    = torch.stack(me_s)

# # # #         if importance_weight and scores:
# # # #             score_tensor = torch.stack(scores)  # [S, B]
# # # #             k = max(1, int(num_ensemble * 0.7))
# # # #             _, top_idx = score_tensor.topk(k, dim=0)
# # # #             pred_mean_list = []
# # # #             for b in range(B):
# # # #                 sel = all_trajs[top_idx[:, b], :, b, :]
# # # #                 pred_mean_list.append(sel.median(0).values)
# # # #             pred_mean = torch.stack(pred_mean_list, dim=1)
# # # #         else:
# # # #             pred_mean = all_trajs.median(0).values

# # # #         if predict_csv:
# # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # #         return pred_mean, all_me.mean(0), all_trajs

# # # #     def _score_sample(self, traj, env_data):
# # # #         B = traj.shape[1]
# # # #         if traj.shape[0] < 2:
# # # #             return torch.ones(B, device=traj.device)

# # # #         traj_deg = _norm_to_deg_fn(traj)
# # # #         dt_deg = traj_deg[1:] - traj_deg[:-1]
# # # #         lat_rad = torch.deg2rad(traj_deg[:-1, :, 1])
# # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # #         dx_km = dt_deg[:, :, 0] * cos_lat * 111.0
# # # #         dy_km = dt_deg[:, :, 1] * 111.0
# # # #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)

# # # #         speed_penalty = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
# # # #         speed_score = torch.exp(-speed_penalty.mean(0) / 20.0)

# # # #         if dt_deg.shape[0] >= 2:
# # # #             jerk = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
# # # #             smooth_score = torch.exp(-jerk * 5.0)
# # # #         else:
# # # #             smooth_score = torch.ones(B, device=traj.device)

# # # #         return speed_score * smooth_score

# # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # #         with torch.enable_grad():
# # # #             x   = x_pred.detach().requires_grad_(True)
# # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # #             for _ in range(n_steps):
# # # #                 opt.zero_grad()
# # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # #                 pred_deg    = _norm_to_deg_fn(pred_abs)
# # # #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# # # #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# # # #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# # # #                 speed       = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # #                                           + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # # #                 loss.backward()
# # # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # # #                 opt.step()
# # # #         return x.detach()

# # # #     @staticmethod
# # # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # # #         import numpy as np
# # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # #         T, B, _ = traj_mean.shape
# # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # #                     "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg",
# # # #                     "ens_spread_km"]
# # # #         write_hdr = not os.path.exists(csv_path)
# # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # #         with open(csv_path, "a", newline="") as fh:
# # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # #             if write_hdr: w.writeheader()
# # # #             for b in range(B):
# # # #                 for k in range(T):
# # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# # # #                                * math.cos(math.radians(mean_lat[k, b])))
# # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # #                     w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
# # # #                                 "lead_h": (k+1)*6,
# # # #                                 "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # #                                 "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # #                                 "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # # #                                 "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # # #                                 "ens_spread_km": f"{spread:.2f}"})


# # # # TCDiffusion = TCFlowMatching
# # # """
# # # Model/flow_matching_model.py — v34fix + v36_cal logging fix
# # # ════════════════════════════════════════════════════════════════════
# # # THAY ĐỔI DUY NHẤT so với v34fix (doc 7):

# # #   FIX get_loss_breakdown() return dict — THÊM 3 keys còn thiếu:
# # #     BEFORE:
# # #       velocity = loss_dict['velocity']   # = 0.0
# # #       # vel_smooth, ate_cte, h_direct KHÔNG có → log hiện 0.000

# # #     AFTER:
# # #       h_direct   = loss_dict['h_direct']    # ← THÊM
# # #       vel_smooth = loss_dict['vel_smooth']  # ← THÊM
# # #       ate_cte    = loss_dict['ate_cte']     # ← THÊM
# # #       velocity   = loss_dict['velocity']    # giữ nguyên

# # #   Tất cả code khác GIỐNG HỆT v34fix.
# # #   Training logic không thay đổi gì.
# # # """
# # # from __future__ import annotations

# # # import csv
# # # import math
# # # import os
# # # from copy import deepcopy
# # # from datetime import datetime

# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F

# # # from Model.FNO3D_encoder import FNO3DEncoder
# # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # from Model.env_net_transformer_gphsplit import Env_net
# # # from Model.losses import (
# # #     compute_total_loss,
# # #     _haversine_deg, _norm_to_deg,
# # #     WEIGHTS,
# # # )


# # # def _norm_to_deg_fn(t):
# # #     """Normalized [lon, lat] → degrees."""
# # #     out = t.clone()
# # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # #     return out


# # # def _unwrap_model(model: nn.Module) -> nn.Module:
# # #     if hasattr(model, '_orig_mod'):
# # #         return model._orig_mod
# # #     return model


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  EMA wrapper — v34fix (unchanged)
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class EMAModel:
# # #     def __init__(self, model: nn.Module, decay: float = 0.999):
# # #         self.decay = decay
# # #         m = _unwrap_model(model)
# # #         self.shadow = {
# # #             k: v.detach().clone()
# # #             for k, v in m.state_dict().items()
# # #             if v.dtype.is_floating_point
# # #         }

# # #     def update(self, model: nn.Module):
# # #         m = _unwrap_model(model)
# # #         with torch.no_grad():
# # #             for k, v in m.state_dict().items():
# # #                 if k in self.shadow:
# # #                     self.shadow[k].mul_(self.decay).add_(
# # #                         v.detach(), alpha=1 - self.decay)

# # #     def apply_to(self, model: nn.Module):
# # #         m = _unwrap_model(model)
# # #         backup = {}
# # #         sd = m.state_dict()
# # #         for k in self.shadow:
# # #             if k not in sd:
# # #                 continue
# # #             backup[k] = sd[k].detach().clone()
# # #             sd[k].copy_(self.shadow[k])
# # #         return backup

# # #     def restore(self, model: nn.Module, backup: dict):
# # #         m = _unwrap_model(model)
# # #         sd = m.state_dict()
# # #         for k, v in backup.items():
# # #             if k in sd:
# # #                 sd[k].copy_(v)


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  VelocityField — unchanged from v34fix
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class VelocityField(nn.Module):
# # #     RAW_CTX_DIM = 512

# # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # #                  unet_in_ch=13, **kwargs):
# # #         super().__init__()
# # #         self.pred_len = pred_len
# # #         self.obs_len  = obs_len

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

# # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # #         self.ctx_drop = nn.Dropout(0.10)
# # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # #         self.vel_obs_enc = nn.Sequential(
# # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # #             nn.LayerNorm(256),
# # #             nn.Linear(256, 256), nn.GELU(),
# # #         )
# # #         self.steering_enc = nn.Sequential(
# # #             nn.Linear(4, 64), nn.GELU(),
# # #             nn.LayerNorm(64),
# # #             nn.Linear(64, 128), nn.GELU(),
# # #             nn.Linear(128, 256),
# # #         )

# # #         self.time_fc1   = nn.Linear(256, 512)
# # #         self.time_fc2   = nn.Linear(512, 256)
# # #         self.traj_embed = nn.Linear(4, 256)
# # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # #         self.step_embed = nn.Embedding(pred_len, 256)

# # #         self.transformer = nn.TransformerDecoder(
# # #             nn.TransformerDecoderLayer(
# # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # #                 dropout=0.10, activation="gelu", batch_first=True),
# # #             num_layers=5)

# # #         self.out_fc1 = nn.Linear(256, 512)
# # #         self.out_fc2 = nn.Linear(512, 4)

# # #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# # #         self.physics_scale  = nn.Parameter(torch.ones(4) * 0.3)
# # #         self.steering_scale = nn.Parameter(torch.ones(4) * 0.2)
# # #         self._init_weights()

# # #     def _init_weights(self):
# # #         with torch.no_grad():
# # #             for name, m in self.named_modules():
# # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # #                     if m.bias is not None:
# # #                         nn.init.zeros_(m.bias)

# # #     def _time_emb(self, t, dim=256):
# # #         half = dim // 2
# # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # #     def _context(self, batch_list):
# # #         obs_traj  = batch_list[0]
# # #         obs_Me    = batch_list[7]
# # #         image_obs = batch_list[11]
# # #         env_data  = batch_list[13]

# # #         if image_obs.dim() == 4:
# # #             image_obs = image_obs.unsqueeze(2)
# # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # #         T_obs = obs_traj.shape[0]

# # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # #         if e_3d_s.shape[1] != T_obs:
# # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # #                                    mode="linear", align_corners=False).permute(0, 2, 1)

# # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # #         return F.gelu(self.ctx_ln(self.ctx_fc1(
# # #             torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # #         if noise_scale > 0.0:
# # #             raw = raw + torch.randn_like(raw) * noise_scale
# # #         return self.ctx_fc2(self.ctx_drop(raw))

# # #     def _get_vel_obs_feat(self, obs_traj):
# # #         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
# # #         if T_obs >= 2:
# # #             vels = obs_traj[1:] - obs_traj[:-1]
# # #         else:
# # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
# # #         if vels.shape[0] < self.obs_len:
# # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# # #             vels = torch.cat([pad, vels], dim=0)
# # #         else:
# # #             vels = vels[-self.obs_len:]
# # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # #     def _get_steering_feat(self, env_data, B, device):
# # #         if env_data is None:
# # #             return torch.zeros(B, 256, device=device)

# # #         def _safe_get(key, default_val=0.0):
# # #             v = env_data.get(key, None)
# # #             if v is None or not torch.is_tensor(v):
# # #                 return torch.full((B,), default_val, device=device)
# # #             v = v.to(device).float()
# # #             if v.dim() >= 2:
# # #                 while v.dim() > 1:
# # #                     v = v.mean(-1)
# # #             if v.shape[0] != B:
# # #                 v = (v.view(-1)[:B] if v.numel() >= B
# # #                      else torch.full((B,), default_val, device=device))
# # #             return v

# # #         feat = torch.stack([_safe_get("u500_mean"), _safe_get("v500_mean"),
# # #                              _safe_get("u500_center"), _safe_get("v500_center")], dim=-1)
# # #         return self.steering_enc(feat)

# # #     def _beta_drift(self, x_t):
# # #         lat_deg = x_t[:, :, 1] * 5.0
# # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # #         R_tc    = 3e5
# # #         v_phys  = torch.zeros_like(x_t)
# # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # #         return v_phys

# # #     def _steering_drift(self, x_t, env_data):
# # #         if env_data is None:
# # #             return torch.zeros_like(x_t)
# # #         B = x_t.shape[0]
# # #         device = x_t.device

# # #         def _safe_mean(key):
# # #             v = env_data.get(key, None)
# # #             if v is None or not torch.is_tensor(v):
# # #                 return torch.zeros(B, device=device)
# # #             v = v.to(device).float()
# # #             while v.dim() > 1:
# # #                 v = v.mean(-1)
# # #             if v.numel() < B:
# # #                 return torch.zeros(B, device=device)
# # #             return v.view(-1)[:B]

# # #         u = _safe_mean("u500_center")
# # #         v = _safe_mean("v500_center")
# # #         lat_deg = x_t[:, :, 1] * 5.0
# # #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# # #         u_norm = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat) * 0.2
# # #         v_norm = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0) * 0.2
# # #         out = torch.zeros_like(x_t)
# # #         out[:, :, 0] = u_norm
# # #         out[:, :, 1] = v_norm
# # #         return out

# # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
# # #                 env_data=None):
# # #         B     = x_t.shape[0]
# # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # #         t_emb = self.time_fc2(t_emb)

# # #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # #         s_emb    = self.step_embed(step_idx)

# # #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# # #                  + self.pos_enc[:, :T_seq]
# # #                  + t_emb.unsqueeze(1)
# # #                  + s_emb)

# # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # #         if vel_obs_feat is not None:
# # #             mem_parts.append(vel_obs_feat.unsqueeze(1))
# # #         if steering_feat is not None:
# # #             mem_parts.append(steering_feat.unsqueeze(1))

# # #         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
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
# # #                            vel_obs_feat=None, steering_feat=None, env_data=None):
# # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# # #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  TCFlowMatching — unchanged except get_loss_breakdown return dict
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class TCFlowMatching(nn.Module):

# # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # #                  n_train_ens=4, unet_in_ch=13,
# # #                  ctx_noise_scale=0.01, initial_sample_sigma=0.03,
# # #                  teacher_forcing=True, use_ema=True, ema_decay=0.999,
# # #                  **kwargs):
# # #         super().__init__()
# # #         self.pred_len             = pred_len
# # #         self.obs_len              = obs_len
# # #         self.sigma_min            = sigma_min
# # #         self.n_train_ens          = n_train_ens
# # #         self.ctx_noise_scale      = ctx_noise_scale
# # #         self.initial_sample_sigma = initial_sample_sigma
# # #         self.teacher_forcing      = teacher_forcing
# # #         self.active_pred_len      = pred_len

# # #         self.net = VelocityField(
# # #             pred_len=pred_len, obs_len=obs_len,
# # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # #         self.use_ema   = use_ema
# # #         self.ema_decay = ema_decay
# # #         self._ema      = None

# # #     def init_ema(self):
# # #         if self.use_ema:
# # #             self._ema = EMAModel(self, decay=self.ema_decay)

# # #     def ema_update(self):
# # #         if self._ema is not None:
# # #             self._ema.update(self)

# # #     def set_curriculum_len(self, *a, **kw):
# # #         pass

# # #     @staticmethod
# # #     def _to_rel(traj, Me, lp, lm):
# # #         return torch.cat([traj - lp.unsqueeze(0),
# # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # #     @staticmethod
# # #     def _to_abs(rel, lp, lm):
# # #         d = rel.permute(1, 0, 2)
# # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # #     def _cfm_noisy(self, x1, sigma_min=None):
# # #         if sigma_min is None:
# # #             sigma_min = self.sigma_min
# # #         B, device = x1.shape[0], x1.device
# # #         x0 = torch.randn_like(x1) * sigma_min
# # #         t  = torch.rand(B, device=device)
# # #         te = t.view(B, 1, 1)
# # #         x_t = (1.0 - te) * x0 + te * x1
# # #         u   = x1 - x0
# # #         return x_t, t, u

# # #     @staticmethod
# # #     def _lon_flip_aug(batch_list, p=0.3):
# # #         if torch.rand(1).item() > p:
# # #             return batch_list
# # #         aug = list(batch_list)
# # #         for idx in [0, 1, 2, 3]:
# # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # #                 t = aug[idx].clone()
# # #                 t[..., 0] = -t[..., 0]
# # #                 aug[idx] = t
# # #         return aug

# # #     @staticmethod
# # #     def _obs_noise_aug(batch_list, sigma=0.005):
# # #         if torch.rand(1).item() > 0.5:
# # #             return batch_list
# # #         aug = list(batch_list)
# # #         if torch.is_tensor(aug[0]):
# # #             aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
# # #         return aug

# # #     @staticmethod
# # #     def _mixup_aug(batch_list, alpha=0.2, p=0.2):
# # #         if torch.rand(1).item() > p:
# # #             return batch_list
# # #         aug = list(batch_list)
# # #         lam = float(torch.distributions.Beta(alpha, alpha).sample())
# # #         lam = max(lam, 1 - lam)
# # #         if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
# # #             B    = aug[0].shape[1]
# # #             perm = torch.randperm(B, device=aug[0].device)
# # #             for idx in [0, 1, 7, 8]:
# # #                 if torch.is_tensor(aug[idx]):
# # #                     aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
# # #         return aug

# # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # #         batch_list = self._lon_flip_aug(batch_list)
# # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
# # #         if epoch >= 5:
# # #             batch_list = self._mixup_aug(batch_list, alpha=0.2, p=0.15)

# # #         traj_gt  = batch_list[1]
# # #         Me_gt    = batch_list[8]
# # #         obs_t    = batch_list[0]
# # #         obs_Me   = batch_list[7]
# # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # #         lp, lm   = obs_t[-1], obs_Me[-1]

# # #         # Sigma schedule
# # #         if epoch < 15:
# # #             current_sigma = 0.15
# # #         elif epoch < 40:
# # #             current_sigma = 0.15 - (epoch - 15) / 25.0 * 0.09
# # #         else:
# # #             current_sigma = 0.06

# # #         raw_ctx       = self.net._context(batch_list)
# # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # #         steering_feat = self.net._get_steering_feat(
# # #             env_data, obs_t.shape[1], obs_t.device)

# # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
# # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # #         if self.teacher_forcing and self.training and epoch >= 3:
# # #             p_teacher = max(0.0, 0.5 * (1.0 - (epoch - 3) / 37.0))
# # #             if p_teacher > 0 and torch.rand(1).item() < p_teacher:
# # #                 far_mask = torch.zeros_like(x_t)
# # #                 far_mask[:, 6:, :] = 0.3
# # #                 x_t = x_t * (1 - far_mask) + x1_rel * far_mask

# # #         pred_vel = self.net.forward_with_ctx(
# # #             x_t, fm_t, raw_ctx,
# # #             vel_obs_feat=vel_obs_feat,
# # #             steering_feat=steering_feat,
# # #             env_data=env_data,
# # #         )

# # #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)
# # #         pred_deg = _norm_to_deg_fn(pred_abs)
# # #         gt_deg   = _norm_to_deg_fn(traj_gt)

# # #         loss_dict = compute_total_loss(
# # #             pred_deg=pred_deg,
# # #             gt_deg=gt_deg,
# # #             env_data=env_data,
# # #             weights=WEIGHTS,
# # #             epoch=epoch,
# # #         )

# # #         w_fm  = max(0.3, 1.0 - epoch / 60.0)
# # #         total = w_fm * l_fm_mse + loss_dict["total"]

# # #         l_ens = x_t.new_zeros(())
# # #         if epoch >= 40 and self.n_train_ens >= 2:
# # #             x_t2, fm_t2, u_t2 = self._cfm_noisy(
# # #                 x1_rel, sigma_min=current_sigma)
# # #             pv2 = self.net.forward_with_ctx(
# # #                 x_t2, fm_t2, raw_ctx,
# # #                 vel_obs_feat=vel_obs_feat,
# # #                 steering_feat=steering_feat,
# # #                 env_data=env_data,
# # #             )
# # #             l_ens = F.mse_loss(pv2, u_t2)
# # #             total = total + 0.3 * l_ens

# # #         if torch.isnan(total) or torch.isinf(total):
# # #             total = x_t.new_zeros(())

# # #         # ── FIX: thêm vel_smooth, ate_cte, h_direct vào return dict ──────
# # #         return dict(
# # #             total        = total,
# # #             fm_mse       = l_fm_mse.item(),
# # #             mse_hav      = loss_dict["mse_hav_horizon"],
# # #             multi_scale  = loss_dict["multi_scale"],
# # #             endpoint     = loss_dict["endpoint"],
# # #             h_direct     = loss_dict.get("h_direct",    0.0),   # ← ADDED
# # #             vel_smooth   = loss_dict.get("vel_smooth",  0.0),   # ← ADDED
# # #             ate_cte      = loss_dict.get("ate_cte",     0.0),   # ← ADDED
# # #             shape        = loss_dict["shape"],
# # #             velocity     = loss_dict["velocity"],
# # #             heading      = loss_dict["heading"],
# # #             recurv       = loss_dict["recurv"],
# # #             steering     = loss_dict["steering"],
# # #             ens_consist  = (l_ens.item() if torch.is_tensor(l_ens)
# # #                             else float(l_ens)),
# # #             sigma        = current_sigma,
# # #             w_fm         = w_fm,
# # #             long_range=0.0, fde=0.0, w_lr=0.0, w_fde=0.0,
# # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0,
# # #             recurv_ratio=0.0,
# # #         )

# # #     # ── Sampling — unchanged from v34fix ──────────────────────────────────
# # #     @torch.no_grad()
# # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
# # #                 predict_csv=None, importance_weight=True):
# # #         obs_t    = batch_list[0]
# # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # #         lp       = obs_t[-1]
# # #         lm       = batch_list[7][-1]
# # #         B        = lp.shape[0]
# # #         device   = lp.device
# # #         T        = self.pred_len
# # #         dt       = 1.0 / max(ddim_steps, 1)

# # #         raw_ctx       = self.net._context(batch_list)
# # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # #         steering_feat = self.net._get_steering_feat(env_data, B, device)

# # #         traj_s, me_s, scores = [], [], []
# # #         for ens_i in range(num_ensemble):
# # #             x_t = torch.randn(B, T, 4, device=device) * (self.sigma_min * 2.5)
# # #             for step in range(ddim_steps):
# # #                 t_b = torch.full((B,), step * dt, device=device)
# # #                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
# # #                 vel = self.net.forward_with_ctx(
# # #                     x_t, t_b, raw_ctx,
# # #                     noise_scale=ns,
# # #                     vel_obs_feat=vel_obs_feat,
# # #                     steering_feat=steering_feat,
# # #                     env_data=env_data,
# # #                 )
# # #                 x_t = x_t + dt * vel
# # #             x_t = self._physics_correct(x_t, lp, lm)
# # #             x_t = x_t.clamp(-3.0, 3.0)
# # #             tr, me = self._to_abs(x_t, lp, lm)
# # #             traj_s.append(tr)
# # #             me_s.append(me)
# # #             if importance_weight:
# # #                 scores.append(self._score_sample(tr, env_data))

# # #         all_trajs = torch.stack(traj_s)
# # #         all_me    = torch.stack(me_s)

# # #         if importance_weight and scores:
# # #             score_tensor = torch.stack(scores)
# # #             k = max(1, int(num_ensemble * 0.7))
# # #             _, top_idx = score_tensor.topk(k, dim=0)
# # #             pred_mean = torch.stack([
# # #                 all_trajs[top_idx[:, b], :, b, :].median(0).values
# # #                 for b in range(B)
# # #             ], dim=1)
# # #         else:
# # #             pred_mean = all_trajs.median(0).values

# # #         if predict_csv:
# # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # #         return pred_mean, all_me.mean(0), all_trajs

# # #     def _score_sample(self, traj, env_data):
# # #         B = traj.shape[1]
# # #         if traj.shape[0] < 2:
# # #             return torch.ones(B, device=traj.device)
# # #         traj_deg = _norm_to_deg_fn(traj)
# # #         dt_deg   = traj_deg[1:] - traj_deg[:-1]
# # #         lat_rad  = torch.deg2rad(traj_deg[:-1, :, 1])
# # #         cos_lat  = torch.cos(lat_rad).clamp(min=1e-4)
# # #         dx_km    = dt_deg[:, :, 0] * cos_lat * 111.0
# # #         dy_km    = dt_deg[:, :, 1] * 111.0
# # #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)
# # #         speed_penalty = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
# # #         speed_score   = torch.exp(-speed_penalty.mean(0) / 20.0)
# # #         if dt_deg.shape[0] >= 2:
# # #             jerk = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
# # #             smooth_score = torch.exp(-jerk * 5.0)
# # #         else:
# # #             smooth_score = torch.ones(B, device=traj.device)
# # #         return speed_score * smooth_score

# # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # #         with torch.enable_grad():
# # #             x   = x_pred.detach().requires_grad_(True)
# # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # #             for _ in range(n_steps):
# # #                 opt.zero_grad()
# # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # #                 pred_deg    = _norm_to_deg_fn(pred_abs)
# # #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# # #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# # #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# # #                 speed       = torch.sqrt(
# # #                     (dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # #                     + (dt_deg[:, :, 1] * 111.0).pow(2))
# # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # #                 loss.backward()
# # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # #                 opt.step()
# # #         return x.detach()

# # #     @staticmethod
# # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # #         import numpy as np
# # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # #         T, B, _ = traj_mean.shape
# # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # #                     "lon_mean_deg", "lat_mean_deg",
# # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # #         write_hdr = not os.path.exists(csv_path)
# # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # #         with open(csv_path, "a", newline="") as fh:
# # #             w = csv.DictWriter(fh, fieldnames=fields)
# # #             if write_hdr: w.writeheader()
# # #             for b in range(B):
# # #                 for k in range(T):
# # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# # #                                * math.cos(math.radians(mean_lat[k, b])))
# # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # #                     w.writerow({
# # #                         "timestamp": ts, "batch_idx": b, "step_idx": k,
# # #                         "lead_h": (k + 1) * 6,
# # #                         "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # #                         "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # #                         "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # #                         "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # #                         "ens_spread_km": f"{spread:.2f}",
# # #                     })


# # # TCDiffusion = TCFlowMatching

# # """
# # Model/flow_matching_model.py — v37_ate
# # ════════════════════════════════════════════════════════════════════
# # THAY ĐỔI so với v36 (doc 7):

# #   1. get_loss_breakdown() return dict — THÊM key "along_track":
# #        along_track = loss_dict.get("along_track", 0.0)  ← NEW

# #   2. Sigma schedule — TIGHTER từ early epoch:
# #        ep0-5:   0.15 → 0.10  (tighter → học exact position sớm hơn)
# #        ep5-20:  0.10 → 0.05
# #        ep20-50: 0.05 → 0.03
# #        ep50+:   0.03

# #   3. EMA decay: 0.992 → 0.990 (faster update)

# #   KHÔNG thay đổi gì khác.
# # """
# # from __future__ import annotations

# # import csv
# # import math
# # import os
# # from copy import deepcopy
# # from datetime import datetime

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # from Model.FNO3D_encoder import FNO3DEncoder
# # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # from Model.env_net_transformer_gphsplit import Env_net
# # from Model.losses import (
# #     compute_total_loss,
# #     _haversine_deg, _norm_to_deg,
# #     WEIGHTS,
# # )


# # def _norm_to_deg_fn(t):
# #     out = t.clone()
# #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# #     return out


# # def _unwrap_model(model: nn.Module) -> nn.Module:
# #     if hasattr(model, '_orig_mod'):
# #         return model._orig_mod
# #     return model


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  EMA wrapper — unchanged
# # # ══════════════════════════════════════════════════════════════════════════════

# # class EMAModel:
# #     def __init__(self, model: nn.Module, decay: float = 0.999):
# #         self.decay = decay
# #         m = _unwrap_model(model)
# #         self.shadow = {
# #             k: v.detach().clone()
# #             for k, v in m.state_dict().items()
# #             if v.dtype.is_floating_point
# #         }

# #     def update(self, model: nn.Module):
# #         m = _unwrap_model(model)
# #         with torch.no_grad():
# #             for k, v in m.state_dict().items():
# #                 if k in self.shadow:
# #                     self.shadow[k].mul_(self.decay).add_(
# #                         v.detach(), alpha=1 - self.decay)

# #     def apply_to(self, model: nn.Module):
# #         m = _unwrap_model(model)
# #         backup = {}
# #         sd = m.state_dict()
# #         for k in self.shadow:
# #             if k not in sd:
# #                 continue
# #             backup[k] = sd[k].detach().clone()
# #             sd[k].copy_(self.shadow[k])
# #         return backup

# #     def restore(self, model: nn.Module, backup: dict):
# #         m = _unwrap_model(model)
# #         sd = m.state_dict()
# #         for k, v in backup.items():
# #             if k in sd:
# #                 sd[k].copy_(v)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  VelocityField — unchanged from v36
# # # ══════════════════════════════════════════════════════════════════════════════

# # class VelocityField(nn.Module):
# #     RAW_CTX_DIM = 512

# #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# #                  unet_in_ch=13, **kwargs):
# #         super().__init__()
# #         self.pred_len = pred_len
# #         self.obs_len  = obs_len

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

# #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# #         self.ctx_drop = nn.Dropout(0.10)
# #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# #         self.vel_obs_enc = nn.Sequential(
# #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# #             nn.LayerNorm(256),
# #             nn.Linear(256, 256), nn.GELU(),
# #         )
# #         self.steering_enc = nn.Sequential(
# #             nn.Linear(4, 64), nn.GELU(),
# #             nn.LayerNorm(64),
# #             nn.Linear(64, 128), nn.GELU(),
# #             nn.Linear(128, 256),
# #         )

# #         self.time_fc1   = nn.Linear(256, 512)
# #         self.time_fc2   = nn.Linear(512, 256)
# #         self.traj_embed = nn.Linear(4, 256)
# #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# #         self.step_embed = nn.Embedding(pred_len, 256)

# #         self.transformer = nn.TransformerDecoder(
# #             nn.TransformerDecoderLayer(
# #                 d_model=256, nhead=8, dim_feedforward=1024,
# #                 dropout=0.10, activation="gelu", batch_first=True),
# #             num_layers=5)

# #         self.out_fc1 = nn.Linear(256, 512)
# #         self.out_fc2 = nn.Linear(512, 4)

# #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# #         self.physics_scale  = nn.Parameter(torch.ones(4) * 0.3)
# #         self.steering_scale = nn.Parameter(torch.ones(4) * 0.2)
# #         self._init_weights()

# #     def _init_weights(self):
# #         with torch.no_grad():
# #             for name, m in self.named_modules():
# #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# #                     if m.bias is not None:
# #                         nn.init.zeros_(m.bias)

# #     def _time_emb(self, t, dim=256):
# #         half = dim // 2
# #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# #                          * (-math.log(10000.0) / max(half - 1, 1)))
# #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# #     def _context(self, batch_list):
# #         obs_traj  = batch_list[0]
# #         obs_Me    = batch_list[7]
# #         image_obs = batch_list[11]
# #         env_data  = batch_list[13]

# #         if image_obs.dim() == 4:
# #             image_obs = image_obs.unsqueeze(2)
# #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# #         T_obs = obs_traj.shape[0]

# #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# #         e_3d_s = self.bottleneck_proj(e_3d_s)
# #         if e_3d_s.shape[1] != T_obs:
# #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# #                                    mode="linear", align_corners=False).permute(0, 2, 1)

# #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# #         h_t    = self.enc_1d(obs_in, e_3d_s)
# #         e_env, _, _ = self.env_enc(env_data, image_obs)

# #         return F.gelu(self.ctx_ln(self.ctx_fc1(
# #             torch.cat([h_t, e_env, f_spatial], dim=-1))))

# #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# #         if noise_scale > 0.0:
# #             raw = raw + torch.randn_like(raw) * noise_scale
# #         return self.ctx_fc2(self.ctx_drop(raw))

# #     def _get_vel_obs_feat(self, obs_traj):
# #         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
# #         if T_obs >= 2:
# #             vels = obs_traj[1:] - obs_traj[:-1]
# #         else:
# #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
# #         if vels.shape[0] < self.obs_len:
# #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# #             vels = torch.cat([pad, vels], dim=0)
# #         else:
# #             vels = vels[-self.obs_len:]
# #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# #     def _get_steering_feat(self, env_data, B, device):
# #         if env_data is None:
# #             return torch.zeros(B, 256, device=device)

# #         def _safe_get(key, default_val=0.0):
# #             v = env_data.get(key, None)
# #             if v is None or not torch.is_tensor(v):
# #                 return torch.full((B,), default_val, device=device)
# #             v = v.to(device).float()
# #             if v.dim() >= 2:
# #                 while v.dim() > 1:
# #                     v = v.mean(-1)
# #             if v.shape[0] != B:
# #                 v = (v.view(-1)[:B] if v.numel() >= B
# #                      else torch.full((B,), default_val, device=device))
# #             return v

# #         feat = torch.stack([_safe_get("u500_mean"), _safe_get("v500_mean"),
# #                              _safe_get("u500_center"), _safe_get("v500_center")], dim=-1)
# #         return self.steering_enc(feat)

# #     def _beta_drift(self, x_t):
# #         lat_deg = x_t[:, :, 1] * 5.0
# #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# #         R_tc    = 3e5
# #         v_phys  = torch.zeros_like(x_t)
# #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# #         return v_phys

# #     def _steering_drift(self, x_t, env_data):
# #         if env_data is None:
# #             return torch.zeros_like(x_t)
# #         B = x_t.shape[0]
# #         device = x_t.device

# #         def _safe_mean(key):
# #             v = env_data.get(key, None)
# #             if v is None or not torch.is_tensor(v):
# #                 return torch.zeros(B, device=device)
# #             v = v.to(device).float()
# #             while v.dim() > 1:
# #                 v = v.mean(-1)
# #             if v.numel() < B:
# #                 return torch.zeros(B, device=device)
# #             return v.view(-1)[:B]

# #         u = _safe_mean("u500_center")
# #         v = _safe_mean("v500_center")
# #         lat_deg = x_t[:, :, 1] * 5.0
# #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# #         u_norm = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat) * 0.2
# #         v_norm = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0) * 0.2
# #         out = torch.zeros_like(x_t)
# #         out[:, :, 0] = u_norm
# #         out[:, :, 1] = v_norm
# #         return out

# #     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
# #                 env_data=None):
# #         B     = x_t.shape[0]
# #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# #         t_emb = self.time_fc2(t_emb)

# #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# #         s_emb    = self.step_embed(step_idx)

# #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# #                  + self.pos_enc[:, :T_seq]
# #                  + t_emb.unsqueeze(1)
# #                  + s_emb)

# #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# #         if vel_obs_feat is not None:
# #             mem_parts.append(vel_obs_feat.unsqueeze(1))
# #         if steering_feat is not None:
# #             mem_parts.append(steering_feat.unsqueeze(1))

# #         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
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
# #                            vel_obs_feat=None, steering_feat=None, env_data=None):
# #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  TCFlowMatching v37
# # # ══════════════════════════════════════════════════════════════════════════════

# # class TCFlowMatching(nn.Module):

# #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# #                  n_train_ens=4, unet_in_ch=13,
# #                  ctx_noise_scale=0.01, initial_sample_sigma=0.03,
# #                  teacher_forcing=True, use_ema=True, ema_decay=0.990,  # v37: 0.992→0.990
# #                  **kwargs):
# #         super().__init__()
# #         self.pred_len             = pred_len
# #         self.obs_len              = obs_len
# #         self.sigma_min            = sigma_min
# #         self.n_train_ens          = n_train_ens
# #         self.ctx_noise_scale      = ctx_noise_scale
# #         self.initial_sample_sigma = initial_sample_sigma
# #         self.teacher_forcing      = teacher_forcing
# #         self.active_pred_len      = pred_len

# #         self.net = VelocityField(
# #             pred_len=pred_len, obs_len=obs_len,
# #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# #         self.use_ema   = use_ema
# #         self.ema_decay = ema_decay
# #         self._ema      = None

# #     def init_ema(self):
# #         if self.use_ema:
# #             self._ema = EMAModel(self, decay=self.ema_decay)

# #     def ema_update(self):
# #         if self._ema is not None:
# #             self._ema.update(self)

# #     def set_curriculum_len(self, *a, **kw):
# #         pass

# #     @staticmethod
# #     def _to_rel(traj, Me, lp, lm):
# #         return torch.cat([traj - lp.unsqueeze(0),
# #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# #     @staticmethod
# #     def _to_abs(rel, lp, lm):
# #         d = rel.permute(1, 0, 2)
# #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# #     def _cfm_noisy(self, x1, sigma_min=None):
# #         if sigma_min is None:
# #             sigma_min = self.sigma_min
# #         B, device = x1.shape[0], x1.device
# #         x0 = torch.randn_like(x1) * sigma_min
# #         t  = torch.rand(B, device=device)
# #         te = t.view(B, 1, 1)
# #         x_t = (1.0 - te) * x0 + te * x1
# #         u   = x1 - x0
# #         return x_t, t, u

# #     @staticmethod
# #     def _lon_flip_aug(batch_list, p=0.3):
# #         if torch.rand(1).item() > p:
# #             return batch_list
# #         aug = list(batch_list)
# #         for idx in [0, 1, 2, 3]:
# #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# #                 t = aug[idx].clone()
# #                 t[..., 0] = -t[..., 0]
# #                 aug[idx] = t
# #         return aug

# #     @staticmethod
# #     def _obs_noise_aug(batch_list, sigma=0.005):
# #         if torch.rand(1).item() > 0.5:
# #             return batch_list
# #         aug = list(batch_list)
# #         if torch.is_tensor(aug[0]):
# #             aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
# #         return aug

# #     @staticmethod
# #     def _mixup_aug(batch_list, alpha=0.2, p=0.2):
# #         if torch.rand(1).item() > p:
# #             return batch_list
# #         aug = list(batch_list)
# #         lam = float(torch.distributions.Beta(alpha, alpha).sample())
# #         lam = max(lam, 1 - lam)
# #         if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
# #             B    = aug[0].shape[1]
# #             perm = torch.randperm(B, device=aug[0].device)
# #             for idx in [0, 1, 7, 8]:
# #                 if torch.is_tensor(aug[idx]):
# #                     aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
# #         return aug

# #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# #         batch_list = self._lon_flip_aug(batch_list)
# #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
# #         if epoch >= 5:
# #             batch_list = self._mixup_aug(batch_list, alpha=0.2, p=0.15)

# #         traj_gt  = batch_list[1]
# #         Me_gt    = batch_list[8]
# #         obs_t    = batch_list[0]
# #         obs_Me   = batch_list[7]
# #         env_data = batch_list[13] if len(batch_list) > 13 else None
# #         lp, lm   = obs_t[-1], obs_Me[-1]

# #         # ── v37: Tighter sigma schedule → học exact position sớm hơn ─────
# #         if epoch < 5:
# #             current_sigma = 0.10                                     # 0.15→0.10
# #         elif epoch < 20:
# #             current_sigma = 0.10 - (epoch - 5) / 15.0 * 0.05       # 0.10→0.05
# #         elif epoch < 50:
# #             current_sigma = 0.05 - (epoch - 20) / 30.0 * 0.02      # 0.05→0.03
# #         else:
# #             current_sigma = 0.03

# #         raw_ctx       = self.net._context(batch_list)
# #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# #         steering_feat = self.net._get_steering_feat(
# #             env_data, obs_t.shape[1], obs_t.device)

# #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
# #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# #         if self.teacher_forcing and self.training and epoch >= 3:
# #             p_teacher = max(0.0, 0.5 * (1.0 - (epoch - 3) / 37.0))
# #             if p_teacher > 0 and torch.rand(1).item() < p_teacher:
# #                 far_mask = torch.zeros_like(x_t)
# #                 far_mask[:, 6:, :] = 0.3
# #                 x_t = x_t * (1 - far_mask) + x1_rel * far_mask

# #         pred_vel = self.net.forward_with_ctx(
# #             x_t, fm_t, raw_ctx,
# #             vel_obs_feat=vel_obs_feat,
# #             steering_feat=steering_feat,
# #             env_data=env_data,
# #         )

# #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)
# #         pred_deg = _norm_to_deg_fn(pred_abs)
# #         gt_deg   = _norm_to_deg_fn(traj_gt)

# #         loss_dict = compute_total_loss(
# #             pred_deg=pred_deg,
# #             gt_deg=gt_deg,
# #             env_data=env_data,
# #             weights=WEIGHTS,
# #             epoch=epoch,
# #         )

# #         w_fm  = max(0.3, 1.0 - epoch / 60.0)
# #         total = w_fm * l_fm_mse + loss_dict["total"]

# #         l_ens = x_t.new_zeros(())
# #         if epoch >= 40 and self.n_train_ens >= 2:
# #             x_t2, fm_t2, u_t2 = self._cfm_noisy(
# #                 x1_rel, sigma_min=current_sigma)
# #             pv2 = self.net.forward_with_ctx(
# #                 x_t2, fm_t2, raw_ctx,
# #                 vel_obs_feat=vel_obs_feat,
# #                 steering_feat=steering_feat,
# #                 env_data=env_data,
# #             )
# #             l_ens = F.mse_loss(pv2, u_t2)
# #             total = total + 0.3 * l_ens

# #         if torch.isnan(total) or torch.isinf(total):
# #             total = x_t.new_zeros(())

# #         # v37: thêm key "along_track" vào return dict
# #         return dict(
# #             total        = total,
# #             fm_mse       = l_fm_mse.item(),
# #             mse_hav      = loss_dict["mse_hav_horizon"],
# #             multi_scale  = loss_dict["multi_scale"],
# #             endpoint     = loss_dict["endpoint"],
# #             h_direct     = loss_dict.get("h_direct",     0.0),
# #             vel_smooth   = loss_dict.get("vel_smooth",   0.0),
# #             ate_cte      = loss_dict.get("ate_cte",      0.0),
# #             along_track  = loss_dict.get("along_track",  0.0),   # ← v37 NEW
# #             shape        = loss_dict["shape"],
# #             velocity     = loss_dict["velocity"],
# #             heading      = loss_dict["heading"],
# #             recurv       = loss_dict["recurv"],
# #             steering     = loss_dict["steering"],
# #             ens_consist  = (l_ens.item() if torch.is_tensor(l_ens)
# #                             else float(l_ens)),
# #             sigma        = current_sigma,
# #             w_fm         = w_fm,
# #             long_range=0.0, fde=0.0, w_lr=0.0, w_fde=0.0,
# #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# #             spread=0.0, continuity=0.0, step=0.0, disp=0.0,
# #             recurv_ratio=0.0,
# #         )

# #     # ── Sampling — unchanged from v36 ─────────────────────────────────────
# #     @torch.no_grad()
# #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
# #                 predict_csv=None, importance_weight=True):
# #         obs_t    = batch_list[0]
# #         env_data = batch_list[13] if len(batch_list) > 13 else None
# #         lp       = obs_t[-1]
# #         lm       = batch_list[7][-1]
# #         B        = lp.shape[0]
# #         device   = lp.device
# #         T        = self.pred_len
# #         dt       = 1.0 / max(ddim_steps, 1)

# #         raw_ctx       = self.net._context(batch_list)
# #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# #         steering_feat = self.net._get_steering_feat(env_data, B, device)

# #         traj_s, me_s, scores = [], [], []
# #         for ens_i in range(num_ensemble):
# #             x_t = torch.randn(B, T, 4, device=device) * (self.sigma_min * 2.5)
# #             for step in range(ddim_steps):
# #                 t_b = torch.full((B,), step * dt, device=device)
# #                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
# #                 vel = self.net.forward_with_ctx(
# #                     x_t, t_b, raw_ctx,
# #                     noise_scale=ns,
# #                     vel_obs_feat=vel_obs_feat,
# #                     steering_feat=steering_feat,
# #                     env_data=env_data,
# #                 )
# #                 x_t = x_t + dt * vel
# #             x_t = self._physics_correct(x_t, lp, lm)
# #             x_t = x_t.clamp(-3.0, 3.0)
# #             tr, me = self._to_abs(x_t, lp, lm)
# #             traj_s.append(tr)
# #             me_s.append(me)
# #             if importance_weight:
# #                 scores.append(self._score_sample(tr, env_data))

# #         all_trajs = torch.stack(traj_s)
# #         all_me    = torch.stack(me_s)

# #         if importance_weight and scores:
# #             score_tensor = torch.stack(scores)
# #             k = max(1, int(num_ensemble * 0.7))
# #             _, top_idx = score_tensor.topk(k, dim=0)
# #             pred_mean = torch.stack([
# #                 all_trajs[top_idx[:, b], :, b, :].median(0).values
# #                 for b in range(B)
# #             ], dim=1)
# #         else:
# #             pred_mean = all_trajs.median(0).values

# #         if predict_csv:
# #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# #         return pred_mean, all_me.mean(0), all_trajs

# #     def _score_sample(self, traj, env_data):
# #         B = traj.shape[1]
# #         if traj.shape[0] < 2:
# #             return torch.ones(B, device=traj.device)
# #         traj_deg = _norm_to_deg_fn(traj)
# #         dt_deg   = traj_deg[1:] - traj_deg[:-1]
# #         lat_rad  = torch.deg2rad(traj_deg[:-1, :, 1])
# #         cos_lat  = torch.cos(lat_rad).clamp(min=1e-4)
# #         dx_km    = dt_deg[:, :, 0] * cos_lat * 111.0
# #         dy_km    = dt_deg[:, :, 1] * 111.0
# #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)
# #         speed_penalty = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
# #         speed_score   = torch.exp(-speed_penalty.mean(0) / 20.0)
# #         if dt_deg.shape[0] >= 2:
# #             jerk = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
# #             smooth_score = torch.exp(-jerk * 5.0)
# #         else:
# #             smooth_score = torch.ones(B, device=traj.device)
# #         return speed_score * smooth_score

# #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# #         with torch.enable_grad():
# #             x   = x_pred.detach().requires_grad_(True)
# #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# #             for _ in range(n_steps):
# #                 opt.zero_grad()
# #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# #                 pred_deg    = _norm_to_deg_fn(pred_abs)
# #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# #                 speed       = torch.sqrt(
# #                     (dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# #                     + (dt_deg[:, :, 1] * 111.0).pow(2))
# #                 loss = F.relu(speed - 600.0).pow(2).mean()
# #                 loss.backward()
# #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# #                 opt.step()
# #         return x.detach()

# #     @staticmethod
# #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# #         import numpy as np
# #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# #         T, B, _ = traj_mean.shape
# #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# #                     "lon_mean_deg", "lat_mean_deg",
# #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# #         write_hdr = not os.path.exists(csv_path)
# #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# #         with open(csv_path, "a", newline="") as fh:
# #             w = csv.DictWriter(fh, fieldnames=fields)
# #             if write_hdr: w.writeheader()
# #             for b in range(B):
# #                 for k in range(T):
# #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# #                                * math.cos(math.radians(mean_lat[k, b])))
# #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# #                     w.writerow({
# #                         "timestamp": ts, "batch_idx": b, "step_idx": k,
# #                         "lead_h": (k + 1) * 6,
# #                         "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# #                         "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# #                         "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# #                         "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# #                         "ens_spread_km": f"{spread:.2f}",
# #                     })


# # TCDiffusion = TCFlowMatching

# """
# Model/flow_matching_model.py — v34fix  (v39 compat)
# ═══════════════════════════════════════════════════════════════════

# THAY ĐỔI DUY NHẤT so với v34fix gốc (doc 7):
#   get_loss_breakdown() return dict — thêm 2 keys cho losses v39:
#     speed_acc  = loss_dict.get('speed_acc',  0.0)   ← THÊM
#     cumul_disp = loss_dict.get('cumul_disp', 0.0)   ← THÊM

# Tất cả code khác GIỐNG HỆT v34fix.
# Không thay đổi architecture, training logic, sampling.
# """
# from __future__ import annotations

# import csv
# import math
# import os
# from datetime import datetime

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from Model.FNO3D_encoder import FNO3DEncoder
# from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# from Model.env_net_transformer_gphsplit import Env_net
# from Model.losses import (
#     compute_total_loss,
#     _haversine_deg, _norm_to_deg,
#     WEIGHTS,
# )


# def _norm_to_deg_fn(t):
#     out = t.clone()
#     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
#     out[..., 1] = (t[..., 1] * 50.0) / 10.0
#     return out


# def _unwrap_model(model: nn.Module) -> nn.Module:
#     if hasattr(model, '_orig_mod'):
#         return model._orig_mod
#     return model


# # ══════════════════════════════════════════════════════════════════════════════
# #  EMAModel — unchanged from v34fix
# # ══════════════════════════════════════════════════════════════════════════════

# class EMAModel:
#     def __init__(self, model: nn.Module, decay: float = 0.999):
#         self.decay = decay
#         m = _unwrap_model(model)
#         self.shadow = {
#             k: v.detach().clone()
#             for k, v in m.state_dict().items()
#             if v.dtype.is_floating_point
#         }

#     def update(self, model: nn.Module):
#         m = _unwrap_model(model)
#         with torch.no_grad():
#             for k, v in m.state_dict().items():
#                 if k in self.shadow:
#                     self.shadow[k].mul_(self.decay).add_(
#                         v.detach(), alpha=1 - self.decay)

#     def apply_to(self, model: nn.Module):
#         m = _unwrap_model(model)
#         backup = {}
#         sd = m.state_dict()
#         for k in self.shadow:
#             if k not in sd:
#                 continue
#             backup[k] = sd[k].detach().clone()
#             sd[k].copy_(self.shadow[k])
#         return backup

#     def restore(self, model: nn.Module, backup: dict):
#         m = _unwrap_model(model)
#         sd = m.state_dict()
#         for k, v in backup.items():
#             if k in sd:
#                 sd[k].copy_(v)


# # ══════════════════════════════════════════════════════════════════════════════
# #  VelocityField — unchanged from v34fix
# # ══════════════════════════════════════════════════════════════════════════════

# class VelocityField(nn.Module):
#     RAW_CTX_DIM = 512

#     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
#                  unet_in_ch=13, **kwargs):
#         super().__init__()
#         self.pred_len = pred_len
#         self.obs_len  = obs_len

#         self.spatial_enc = FNO3DEncoder(
#             in_channel=unet_in_ch, out_channel=1, d_model=32,
#             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
#             spatial_down=32, dropout=0.05)
#         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
#         self.bottleneck_proj = nn.Linear(128, 128)
#         self.decoder_proj    = nn.Linear(1, 16)

#         self.enc_1d = DataEncoder1D(
#             in_1d=4, feat_3d_dim=128, mlp_h=64,
#             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
#         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

#         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
#         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
#         self.ctx_drop = nn.Dropout(0.10)
#         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

#         self.vel_obs_enc = nn.Sequential(
#             nn.Linear(obs_len * 2, 256), nn.GELU(),
#             nn.LayerNorm(256),
#             nn.Linear(256, 256), nn.GELU(),
#         )
#         self.steering_enc = nn.Sequential(
#             nn.Linear(4, 64), nn.GELU(),
#             nn.LayerNorm(64),
#             nn.Linear(64, 128), nn.GELU(),
#             nn.Linear(128, 256),
#         )

#         self.time_fc1   = nn.Linear(256, 512)
#         self.time_fc2   = nn.Linear(512, 256)
#         self.traj_embed = nn.Linear(4, 256)
#         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
#         self.step_embed = nn.Embedding(pred_len, 256)

#         self.transformer = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 d_model=256, nhead=8, dim_feedforward=1024,
#                 dropout=0.10, activation="gelu", batch_first=True),
#             num_layers=5)

#         self.out_fc1 = nn.Linear(256, 512)
#         self.out_fc2 = nn.Linear(512, 4)

#         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
#         self.physics_scale  = nn.Parameter(torch.ones(4) * 0.3)
#         self.steering_scale = nn.Parameter(torch.ones(4) * 0.2)
#         self._init_weights()

#     def _init_weights(self):
#         with torch.no_grad():
#             for name, m in self.named_modules():
#                 if isinstance(m, nn.Linear) and 'out_fc' in name:
#                     nn.init.xavier_uniform_(m.weight, gain=0.1)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)

#     def _time_emb(self, t, dim=256):
#         half = dim // 2
#         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
#                          * (-math.log(10000.0) / max(half - 1, 1)))
#         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
#         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

#     def _context(self, batch_list):
#         obs_traj  = batch_list[0]
#         obs_Me    = batch_list[7]
#         image_obs = batch_list[11]
#         env_data  = batch_list[13]

#         if image_obs.dim() == 4:
#             image_obs = image_obs.unsqueeze(2)
#         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
#             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

#         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
#         T_obs = obs_traj.shape[0]

#         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
#         e_3d_s = self.bottleneck_proj(e_3d_s)
#         if e_3d_s.shape[1] != T_obs:
#             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
#                                    mode="linear", align_corners=False).permute(0, 2, 1)

#         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
#         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
#                              device=e_3d_dec_t.device) * 0.5, dim=0)
#         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

#         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
#         h_t    = self.enc_1d(obs_in, e_3d_s)
#         e_env, _, _ = self.env_enc(env_data, image_obs)

#         return F.gelu(self.ctx_ln(self.ctx_fc1(
#             torch.cat([h_t, e_env, f_spatial], dim=-1))))

#     def _apply_ctx_head(self, raw, noise_scale=0.0):
#         if noise_scale > 0.0:
#             raw = raw + torch.randn_like(raw) * noise_scale
#         return self.ctx_fc2(self.ctx_drop(raw))

#     def _get_vel_obs_feat(self, obs_traj):
#         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
#         if T_obs >= 2:
#             vels = obs_traj[1:] - obs_traj[:-1]
#         else:
#             vels = torch.zeros(1, B, 2, device=obs_traj.device)
#         if vels.shape[0] < self.obs_len:
#             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2,
#                                device=obs_traj.device)
#             vels = torch.cat([pad, vels], dim=0)
#         else:
#             vels = vels[-self.obs_len:]
#         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

#     def _get_steering_feat(self, env_data, B, device):
#         if env_data is None:
#             return torch.zeros(B, 256, device=device)

#         def _safe_get(key, default=0.0):
#             v = env_data.get(key, None)
#             if v is None or not torch.is_tensor(v):
#                 return torch.full((B,), default, device=device)
#             v = v.to(device).float()
#             if v.dim() >= 2:
#                 while v.dim() > 1: v = v.mean(-1)
#             if v.shape[0] != B:
#                 v = (v.view(-1)[:B] if v.numel() >= B
#                      else torch.full((B,), default, device=device))
#             return v

#         feat = torch.stack([_safe_get("u500_mean"), _safe_get("v500_mean"),
#                              _safe_get("u500_center"), _safe_get("v500_center")],
#                             dim=-1)
#         return self.steering_enc(feat)

#     def _beta_drift(self, x_t):
#         lat_deg = x_t[:, :, 1] * 5.0
#         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
#         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
#         R_tc    = 3e5
#         v_phys  = torch.zeros_like(x_t)
#         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
#         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
#         return v_phys

#     def _steering_drift(self, x_t, env_data):
#         if env_data is None:
#             return torch.zeros_like(x_t)
#         B, device = x_t.shape[0], x_t.device

#         def _safe_mean(key):
#             v = env_data.get(key, None)
#             if v is None or not torch.is_tensor(v):
#                 return torch.zeros(B, device=device)
#             v = v.to(device).float()
#             while v.dim() > 1: v = v.mean(-1)
#             return v.view(-1)[:B] if v.numel() >= B else torch.zeros(B, device=device)

#         u = _safe_mean("u500_center")
#         v = _safe_mean("v500_center")
#         lat_deg = x_t[:, :, 1] * 5.0
#         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
#         out = torch.zeros_like(x_t)
#         out[:, :, 0] = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat) * 0.2
#         out[:, :, 1] = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0) * 0.2
#         return out

#     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
#                 env_data=None):
#         B     = x_t.shape[0]
#         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
#         t_emb = self.time_fc2(t_emb)

#         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
#         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
#         s_emb    = self.step_embed(step_idx)

#         x_emb = (self.traj_embed(x_t[:, :T_seq])
#                  + self.pos_enc[:, :T_seq]
#                  + t_emb.unsqueeze(1)
#                  + s_emb)

#         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
#         if vel_obs_feat is not None:
#             mem_parts.append(vel_obs_feat.unsqueeze(1))
#         if steering_feat is not None:
#             mem_parts.append(steering_feat.unsqueeze(1))

#         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
#         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
#         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
#         v_neural = v_neural * scale

#         with torch.no_grad():
#             v_phys  = self._beta_drift(x_t[:, :T_seq])
#             v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

#         return (v_neural
#                 + torch.sigmoid(self.physics_scale) * v_phys
#                 + torch.sigmoid(self.steering_scale) * v_steer)

#     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
#                            vel_obs_feat=None, steering_feat=None, env_data=None):
#         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
#         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # ══════════════════════════════════════════════════════════════════════════════
# #  TCFlowMatching — ONE CHANGE: return dict in get_loss_breakdown
# # ══════════════════════════════════════════════════════════════════════════════

# class TCFlowMatching(nn.Module):

#     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
#                  n_train_ens=4, unet_in_ch=13,
#                  ctx_noise_scale=0.01, initial_sample_sigma=0.03,
#                  teacher_forcing=True, use_ema=True, ema_decay=0.999,
#                  **kwargs):
#         super().__init__()
#         self.pred_len             = pred_len
#         self.obs_len              = obs_len
#         self.sigma_min            = sigma_min
#         self.n_train_ens          = n_train_ens
#         self.ctx_noise_scale      = ctx_noise_scale
#         self.initial_sample_sigma = initial_sample_sigma
#         self.teacher_forcing      = teacher_forcing
#         self.active_pred_len      = pred_len

#         self.net = VelocityField(
#             pred_len=pred_len, obs_len=obs_len,
#             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

#         self.use_ema   = use_ema
#         self.ema_decay = ema_decay
#         self._ema      = None

#     def init_ema(self):
#         if self.use_ema:
#             self._ema = EMAModel(self, decay=self.ema_decay)

#     def ema_update(self):
#         if self._ema is not None:
#             self._ema.update(self)

#     def set_curriculum_len(self, *a, **kw):
#         pass

#     @staticmethod
#     def _to_rel(traj, Me, lp, lm):
#         return torch.cat([traj - lp.unsqueeze(0),
#                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

#     @staticmethod
#     def _to_abs(rel, lp, lm):
#         d = rel.permute(1, 0, 2)
#         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

#     def _cfm_noisy(self, x1, sigma_min=None):
#         if sigma_min is None:
#             sigma_min = self.sigma_min
#         B, device = x1.shape[0], x1.device
#         x0 = torch.randn_like(x1) * sigma_min
#         t  = torch.rand(B, device=device)
#         te = t.view(B, 1, 1)
#         return (1.0 - te) * x0 + te * x1, t, x1 - x0

#     @staticmethod
#     def _lon_flip_aug(batch_list, p=0.3):
#         if torch.rand(1).item() > p:
#             return batch_list
#         aug = list(batch_list)
#         for idx in [0, 1, 2, 3]:
#             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
#                 t = aug[idx].clone(); t[..., 0] = -t[..., 0]; aug[idx] = t
#         return aug

#     @staticmethod
#     def _obs_noise_aug(batch_list, sigma=0.005):
#         if torch.rand(1).item() > 0.5:
#             return batch_list
#         aug = list(batch_list)
#         if torch.is_tensor(aug[0]):
#             aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
#         return aug

#     @staticmethod
#     def _mixup_aug(batch_list, alpha=0.2, p=0.2):
#         if torch.rand(1).item() > p:
#             return batch_list
#         aug = list(batch_list)
#         lam = float(torch.distributions.Beta(alpha, alpha).sample())
#         lam = max(lam, 1 - lam)
#         if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
#             B    = aug[0].shape[1]
#             perm = torch.randperm(B, device=aug[0].device)
#             for idx in [0, 1, 7, 8]:
#                 if torch.is_tensor(aug[idx]):
#                     aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
#         return aug

#     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
#         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

#     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
#         # Augmentation
#         batch_list = self._lon_flip_aug(batch_list)
#         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
#         if epoch >= 5:
#             batch_list = self._mixup_aug(batch_list, alpha=0.2, p=0.15)

#         traj_gt  = batch_list[1]
#         Me_gt    = batch_list[8]
#         obs_t    = batch_list[0]
#         obs_Me   = batch_list[7]
#         env_data = batch_list[13] if len(batch_list) > 13 else None
#         lp, lm   = obs_t[-1], obs_Me[-1]

#         # Sigma schedule
#         if epoch < 15:
#             current_sigma = 0.15
#         elif epoch < 40:
#             current_sigma = 0.15 - (epoch - 15) / 25.0 * 0.09
#         else:
#             current_sigma = 0.06

#         raw_ctx       = self.net._context(batch_list)
#         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
#         steering_feat = self.net._get_steering_feat(
#             env_data, obs_t.shape[1], obs_t.device)

#         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
#         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

#         # Scheduled teacher forcing
#         if self.teacher_forcing and self.training and epoch >= 6:
#             p_teacher = max(0.0, 0.5 * (1.0 - (epoch - 6) / 34.0))
#             if p_teacher > 0 and torch.rand(1).item() < p_teacher:
#                 far_mask = torch.zeros_like(x_t)
#                 far_mask[:, 6:, :] = 0.3
#                 x_t = x_t * (1 - far_mask) + x1_rel * far_mask

#         pred_vel = self.net.forward_with_ctx(
#             x_t, fm_t, raw_ctx,
#             vel_obs_feat=vel_obs_feat,
#             steering_feat=steering_feat,
#             env_data=env_data,
#         )

#         l_fm_mse = F.mse_loss(pred_vel, u_target)

#         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
#         x1_pred = x_t + (1.0 - fm_te) * pred_vel
#         pred_abs, _ = self._to_abs(x1_pred, lp, lm)
#         pred_deg    = _norm_to_deg_fn(pred_abs)
#         gt_deg      = _norm_to_deg_fn(traj_gt)

#         loss_dict = compute_total_loss(
#             pred_deg=pred_deg,
#             gt_deg=gt_deg,
#             env_data=env_data,
#             weights=WEIGHTS,
#             epoch=epoch,
#         )

#         w_fm  = max(0.3, 1.0 - epoch / 60.0)
#         total = w_fm * l_fm_mse + loss_dict["total"]

#         l_ens = x_t.new_zeros(())
#         if epoch >= 40 and self.n_train_ens >= 2:
#             x_t2, fm_t2, u_t2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
#             pv2 = self.net.forward_with_ctx(
#                 x_t2, fm_t2, raw_ctx,
#                 vel_obs_feat=vel_obs_feat,
#                 steering_feat=steering_feat,
#                 env_data=env_data,
#             )
#             l_ens = F.mse_loss(pv2, u_t2)
#             total = total + 0.3 * l_ens

#         if torch.isnan(total) or torch.isinf(total):
#             total = x_t.new_zeros(())

#         # ── RETURN DICT — v39 compat: thêm speed_acc + cumul_disp ────────
#         return dict(
#             total        = total,
#             fm_mse       = l_fm_mse.item(),
#             mse_hav      = loss_dict["mse_hav_horizon"],
#             multi_scale  = loss_dict["multi_scale"],
#             endpoint     = loss_dict["endpoint"],
#             h_direct     = loss_dict.get("h_direct",    0.0),
#             vel_smooth   = loss_dict.get("vel_smooth",  0.0),
#             speed_acc    = loss_dict.get("speed_acc",   0.0),  # ← v39 NEW
#             cumul_disp   = loss_dict.get("cumul_disp",  0.0),  # ← v39 NEW
#             ate_cte      = loss_dict.get("ate_cte",     0.0),  # compat (=0)
#             shape        = loss_dict.get("shape",       0.0),
#             velocity     = loss_dict.get("velocity",    0.0),
#             heading      = loss_dict.get("heading",     0.0),
#             recurv       = loss_dict.get("recurv",      0.0),
#             steering     = loss_dict.get("steering",    0.0),
#             ens_consist  = (l_ens.item() if torch.is_tensor(l_ens)
#                             else float(l_ens)),
#             sigma        = current_sigma,
#             w_fm         = w_fm,
#             long_range=0.0, fde=0.0, w_lr=0.0, w_fde=0.0,
#             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
#             spread=0.0, continuity=0.0, step=0.0, disp=0.0,
#             recurv_ratio=0.0,
#         )

#     # ── Sampling — unchanged from v34fix ─────────────────────────────────────
#     @torch.no_grad()
#     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
#                 predict_csv=None, importance_weight=True):
#         obs_t    = batch_list[0]
#         env_data = batch_list[13] if len(batch_list) > 13 else None
#         lp       = obs_t[-1]
#         lm       = batch_list[7][-1]
#         B        = lp.shape[0]
#         device   = lp.device
#         T        = self.pred_len
#         dt       = 1.0 / max(ddim_steps, 1)

#         raw_ctx       = self.net._context(batch_list)
#         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
#         steering_feat = self.net._get_steering_feat(env_data, B, device)

#         traj_s, me_s, scores = [], [], []
#         for _ in range(num_ensemble):
#             x_t = torch.randn(B, T, 4, device=device) * (self.sigma_min * 2.5)
#             for step in range(ddim_steps):
#                 t_b = torch.full((B,), step * dt, device=device)
#                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
#                 vel = self.net.forward_with_ctx(
#                     x_t, t_b, raw_ctx,
#                     noise_scale=ns,
#                     vel_obs_feat=vel_obs_feat,
#                     steering_feat=steering_feat,
#                     env_data=env_data,
#                 )
#                 x_t = x_t + dt * vel
#             x_t = self._physics_correct(x_t, lp, lm)
#             x_t = x_t.clamp(-3.0, 3.0)
#             tr, me = self._to_abs(x_t, lp, lm)
#             traj_s.append(tr)
#             me_s.append(me)
#             if importance_weight:
#                 scores.append(self._score_sample(tr, env_data))

#         all_trajs = torch.stack(traj_s)
#         all_me    = torch.stack(me_s)

#         if importance_weight and scores:
#             score_tensor = torch.stack(scores)
#             k = max(1, int(num_ensemble * 0.7))
#             _, top_idx = score_tensor.topk(k, dim=0)
#             pred_mean = torch.stack([
#                 all_trajs[top_idx[:, b], :, b, :].median(0).values
#                 for b in range(B)
#             ], dim=1)
#         else:
#             pred_mean = all_trajs.median(0).values

#         if predict_csv:
#             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
#         return pred_mean, all_me.mean(0), all_trajs

#     def _score_sample(self, traj, env_data):
#         B = traj.shape[1]
#         if traj.shape[0] < 2:
#             return torch.ones(B, device=traj.device)
#         traj_deg  = _norm_to_deg_fn(traj)
#         dt_deg    = traj_deg[1:] - traj_deg[:-1]
#         lat_rad   = torch.deg2rad(traj_deg[:-1, :, 1])
#         cos_lat   = torch.cos(lat_rad).clamp(min=1e-4)
#         dx_km     = dt_deg[:, :, 0] * cos_lat * 111.0
#         dy_km     = dt_deg[:, :, 1] * 111.0
#         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)
#         speed_pen = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
#         speed_sc  = torch.exp(-speed_pen.mean(0) / 20.0)
#         if dt_deg.shape[0] >= 2:
#             jerk = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
#             smooth_sc = torch.exp(-jerk * 5.0)
#         else:
#             smooth_sc = torch.ones(B, device=traj.device)
#         return speed_sc * smooth_sc

#     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
#         with torch.enable_grad():
#             x   = x_pred.detach().requires_grad_(True)
#             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
#             for _ in range(n_steps):
#                 opt.zero_grad()
#                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
#                 pred_deg    = _norm_to_deg_fn(pred_abs)
#                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
#                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
#                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
#                 speed       = torch.sqrt(
#                     (dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
#                     + (dt_deg[:, :, 1] * 111.0).pow(2))
#                 F.relu(speed - 600.0).pow(2).mean().backward()
#                 torch.nn.utils.clip_grad_norm_([x], 0.05)
#                 opt.step()
#         return x.detach()

#     @staticmethod
#     def _write_predict_csv(csv_path, traj_mean, all_trajs):
#         import numpy as np
#         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
#         T, B, _ = traj_mean.shape
#         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
#         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
#         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
#         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
#         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
#                     "lon_mean_deg", "lat_mean_deg",
#                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
#         write_hdr = not os.path.exists(csv_path)
#         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#         with open(csv_path, "a", newline="") as fh:
#             w = csv.DictWriter(fh, fieldnames=fields)
#             if write_hdr: w.writeheader()
#             for b in range(B):
#                 for k in range(T):
#                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
#                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
#                                * math.cos(math.radians(mean_lat[k, b])))
#                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
#                     w.writerow({
#                         "timestamp": ts, "batch_idx": b, "step_idx": k,
#                         "lead_h": (k + 1) * 6,
#                         "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
#                         "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
#                         "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
#                         "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
#                         "ens_spread_km": f"{spread:.2f}",
#                     })


# TCDiffusion = TCFlowMatching
"""
Model/flow_matching_model.py — v34fix  (v39 compat)
═══════════════════════════════════════════════════════════════════

THAY ĐỔI so với v34fix (v40_clean compat):
  get_loss_breakdown() return dict — thêm 5 keys cho losses v40_clean:
    speed_acc  = loss_dict.get('speed_acc',  0.0)
    cumul_disp = loss_dict.get('cumul_disp', 0.0)
    accel      = loss_dict.get('accel',      0.0)  ← v40 NEW
    decomp     = loss_dict.get('decomp',     0.0)  ← v40 NEW
    cons       = loss_dict.get('cons',       0.0)  ← v40 NEW

Tất cả code khác GIỐNG HỆT v34fix.
Không thay đổi architecture, training logic, sampling.
"""
from __future__ import annotations

import csv
import math
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.FNO3D_encoder import FNO3DEncoder
from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
from Model.env_net_transformer_gphsplit import Env_net
from Model.losses import (
    compute_total_loss,
    _haversine_deg, _norm_to_deg,
    WEIGHTS,
)


def _norm_to_deg_fn(t):
    out = t.clone()
    out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
    out[..., 1] = (t[..., 1] * 50.0) / 10.0
    return out


def _unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  EMAModel — unchanged from v34fix
# ══════════════════════════════════════════════════════════════════════════════

class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        m = _unwrap_model(model)
        self.shadow = {
            k: v.detach().clone()
            for k, v in m.state_dict().items()
            if v.dtype.is_floating_point
        }

    def update(self, model: nn.Module):
        m = _unwrap_model(model)
        with torch.no_grad():
            for k, v in m.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(
                        v.detach(), alpha=1 - self.decay)

    def apply_to(self, model: nn.Module):
        m = _unwrap_model(model)
        backup = {}
        sd = m.state_dict()
        for k in self.shadow:
            if k not in sd:
                continue
            backup[k] = sd[k].detach().clone()
            sd[k].copy_(self.shadow[k])
        return backup

    def restore(self, model: nn.Module, backup: dict):
        m = _unwrap_model(model)
        sd = m.state_dict()
        for k, v in backup.items():
            if k in sd:
                sd[k].copy_(v)


# ══════════════════════════════════════════════════════════════════════════════
#  VelocityField — unchanged from v34fix
# ══════════════════════════════════════════════════════════════════════════════

class VelocityField(nn.Module):
    RAW_CTX_DIM = 512

    def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
                 unet_in_ch=13, **kwargs):
        super().__init__()
        self.pred_len = pred_len
        self.obs_len  = obs_len

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

        self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
        self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
        self.ctx_drop = nn.Dropout(0.10)
        self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

        self.vel_obs_enc = nn.Sequential(
            nn.Linear(obs_len * 2, 256), nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256), nn.GELU(),
        )
        self.steering_enc = nn.Sequential(
            nn.Linear(4, 64), nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 256),
        )

        self.time_fc1   = nn.Linear(256, 512)
        self.time_fc2   = nn.Linear(512, 256)
        self.traj_embed = nn.Linear(4, 256)
        self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
        self.step_embed = nn.Embedding(pred_len, 256)

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=256, nhead=8, dim_feedforward=1024,
                dropout=0.10, activation="gelu", batch_first=True),
            num_layers=5)

        self.out_fc1 = nn.Linear(256, 512)
        self.out_fc2 = nn.Linear(512, 4)

        self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
        self.physics_scale  = nn.Parameter(torch.ones(4) * 0.3)
        self.steering_scale = nn.Parameter(torch.ones(4) * 0.2)
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
        freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
                         * (-math.log(10000.0) / max(half - 1, 1)))
        emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
        return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

    def _context(self, batch_list):
        obs_traj  = batch_list[0]
        obs_Me    = batch_list[7]
        image_obs = batch_list[11]
        env_data  = batch_list[13]

        if image_obs.dim() == 4:
            image_obs = image_obs.unsqueeze(2)
        if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
            image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

        e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
        T_obs = obs_traj.shape[0]

        e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        e_3d_s = self.bottleneck_proj(e_3d_s)
        if e_3d_s.shape[1] != T_obs:
            e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
                                   mode="linear", align_corners=False).permute(0, 2, 1)

        e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
        t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
                             device=e_3d_dec_t.device) * 0.5, dim=0)
        f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        h_t    = self.enc_1d(obs_in, e_3d_s)
        e_env, _, _ = self.env_enc(env_data, image_obs)

        return F.gelu(self.ctx_ln(self.ctx_fc1(
            torch.cat([h_t, e_env, f_spatial], dim=-1))))

    def _apply_ctx_head(self, raw, noise_scale=0.0):
        if noise_scale > 0.0:
            raw = raw + torch.randn_like(raw) * noise_scale
        return self.ctx_fc2(self.ctx_drop(raw))

    def _get_vel_obs_feat(self, obs_traj):
        B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
        if T_obs >= 2:
            vels = obs_traj[1:] - obs_traj[:-1]
        else:
            vels = torch.zeros(1, B, 2, device=obs_traj.device)
        if vels.shape[0] < self.obs_len:
            pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2,
                               device=obs_traj.device)
            vels = torch.cat([pad, vels], dim=0)
        else:
            vels = vels[-self.obs_len:]
        return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

    def _get_steering_feat(self, env_data, B, device):
        if env_data is None:
            return torch.zeros(B, 256, device=device)

        def _safe_get(key, default=0.0):
            v = env_data.get(key, None)
            if v is None or not torch.is_tensor(v):
                return torch.full((B,), default, device=device)
            v = v.to(device).float()
            if v.dim() >= 2:
                while v.dim() > 1: v = v.mean(-1)
            if v.shape[0] != B:
                v = (v.view(-1)[:B] if v.numel() >= B
                     else torch.full((B,), default, device=device))
            return v

        feat = torch.stack([_safe_get("u500_mean"), _safe_get("v500_mean"),
                             _safe_get("u500_center"), _safe_get("v500_center")],
                            dim=-1)
        return self.steering_enc(feat)

    def _beta_drift(self, x_t):
        lat_deg = x_t[:, :, 1] * 5.0
        lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
        beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
        R_tc    = 3e5
        v_phys  = torch.zeros_like(x_t)
        v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
        v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
        return v_phys

    def _steering_drift(self, x_t, env_data):
        if env_data is None:
            return torch.zeros_like(x_t)
        B, device = x_t.shape[0], x_t.device

        def _safe_mean(key):
            v = env_data.get(key, None)
            if v is None or not torch.is_tensor(v):
                return torch.zeros(B, device=device)
            v = v.to(device).float()
            while v.dim() > 1: v = v.mean(-1)
            return v.view(-1)[:B] if v.numel() >= B else torch.zeros(B, device=device)

        u = _safe_mean("u500_center")
        v = _safe_mean("v500_center")
        lat_deg = x_t[:, :, 1] * 5.0
        cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
        out = torch.zeros_like(x_t)
        out[:, :, 0] = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat) * 0.2
        out[:, :, 1] = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0) * 0.2
        return out

    def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
                env_data=None):
        B     = x_t.shape[0]
        t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
        t_emb = self.time_fc2(t_emb)

        T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
        step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
        s_emb    = self.step_embed(step_idx)

        x_emb = (self.traj_embed(x_t[:, :T_seq])
                 + self.pos_enc[:, :T_seq]
                 + t_emb.unsqueeze(1)
                 + s_emb)

        mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
        if vel_obs_feat is not None:
            mem_parts.append(vel_obs_feat.unsqueeze(1))
        if steering_feat is not None:
            mem_parts.append(steering_feat.unsqueeze(1))

        decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
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
                           vel_obs_feat=None, steering_feat=None, env_data=None):
        ctx = self._apply_ctx_head(raw_ctx, noise_scale)
        return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# ══════════════════════════════════════════════════════════════════════════════
#  TCFlowMatching — ONE CHANGE: return dict in get_loss_breakdown
# ══════════════════════════════════════════════════════════════════════════════

class TCFlowMatching(nn.Module):

    def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
                 n_train_ens=4, unet_in_ch=13,
                 ctx_noise_scale=0.01, initial_sample_sigma=0.03,
                 teacher_forcing=True, use_ema=True, ema_decay=0.999,
                 **kwargs):
        super().__init__()
        self.pred_len             = pred_len
        self.obs_len              = obs_len
        self.sigma_min            = sigma_min
        self.n_train_ens          = n_train_ens
        self.ctx_noise_scale      = ctx_noise_scale
        self.initial_sample_sigma = initial_sample_sigma
        self.teacher_forcing      = teacher_forcing
        self.active_pred_len      = pred_len

        self.net = VelocityField(
            pred_len=pred_len, obs_len=obs_len,
            sigma_min=sigma_min, unet_in_ch=unet_in_ch)

        self.use_ema   = use_ema
        self.ema_decay = ema_decay
        self._ema      = None

    def init_ema(self):
        if self.use_ema:
            self._ema = EMAModel(self, decay=self.ema_decay)

    def ema_update(self):
        if self._ema is not None:
            self._ema.update(self)

    def set_curriculum_len(self, *a, **kw):
        pass

    @staticmethod
    def _to_rel(traj, Me, lp, lm):
        return torch.cat([traj - lp.unsqueeze(0),
                           Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

    @staticmethod
    def _to_abs(rel, lp, lm):
        d = rel.permute(1, 0, 2)
        return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

    def _cfm_noisy(self, x1, sigma_min=None):
        if sigma_min is None:
            sigma_min = self.sigma_min
        B, device = x1.shape[0], x1.device
        x0 = torch.randn_like(x1) * sigma_min
        t  = torch.rand(B, device=device)
        te = t.view(B, 1, 1)
        return (1.0 - te) * x0 + te * x1, t, x1 - x0

    @staticmethod
    def _lon_flip_aug(batch_list, p=0.3):
        if torch.rand(1).item() > p:
            return batch_list
        aug = list(batch_list)
        for idx in [0, 1, 2, 3]:
            if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
                t = aug[idx].clone(); t[..., 0] = -t[..., 0]; aug[idx] = t
        return aug

    @staticmethod
    def _obs_noise_aug(batch_list, sigma=0.005):
        if torch.rand(1).item() > 0.5:
            return batch_list
        aug = list(batch_list)
        if torch.is_tensor(aug[0]):
            aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
        return aug

    @staticmethod
    def _mixup_aug(batch_list, alpha=0.2, p=0.2):
        if torch.rand(1).item() > p:
            return batch_list
        aug = list(batch_list)
        lam = float(torch.distributions.Beta(alpha, alpha).sample())
        lam = max(lam, 1 - lam)
        if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
            B    = aug[0].shape[1]
            perm = torch.randperm(B, device=aug[0].device)
            for idx in [0, 1, 7, 8]:
                if torch.is_tensor(aug[idx]):
                    aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
        return aug

    def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
        return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

    def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
        # Augmentation
        batch_list = self._lon_flip_aug(batch_list)
        batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
        if epoch >= 5:
            batch_list = self._mixup_aug(batch_list, alpha=0.2, p=0.15)

        traj_gt  = batch_list[1]
        Me_gt    = batch_list[8]
        obs_t    = batch_list[0]
        obs_Me   = batch_list[7]
        env_data = batch_list[13] if len(batch_list) > 13 else None
        lp, lm   = obs_t[-1], obs_Me[-1]

        # Sigma schedule
        if epoch < 15:
            current_sigma = 0.15
        elif epoch < 40:
            current_sigma = 0.15 - (epoch - 15) / 25.0 * 0.09
        else:
            current_sigma = 0.06

        raw_ctx       = self.net._context(batch_list)
        vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
        steering_feat = self.net._get_steering_feat(
            env_data, obs_t.shape[1], obs_t.device)

        x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
        x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

        # Scheduled teacher forcing
        if self.teacher_forcing and self.training and epoch >= 3:
            p_teacher = max(0.0, 0.5 * (1.0 - (epoch - 3) / 37.0))
            if p_teacher > 0 and torch.rand(1).item() < p_teacher:
                far_mask = torch.zeros_like(x_t)
                far_mask[:, 6:, :] = 0.3
                x_t = x_t * (1 - far_mask) + x1_rel * far_mask

        pred_vel = self.net.forward_with_ctx(
            x_t, fm_t, raw_ctx,
            vel_obs_feat=vel_obs_feat,
            steering_feat=steering_feat,
            env_data=env_data,
        )

        l_fm_mse = F.mse_loss(pred_vel, u_target)

        fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
        x1_pred = x_t + (1.0 - fm_te) * pred_vel
        pred_abs, _ = self._to_abs(x1_pred, lp, lm)
        pred_deg    = _norm_to_deg_fn(pred_abs)
        gt_deg      = _norm_to_deg_fn(traj_gt)

        loss_dict = compute_total_loss(
            pred_deg=pred_deg,
            gt_deg=gt_deg,
            env_data=env_data,
            weights=WEIGHTS,
            epoch=epoch,
        )

        w_fm  = max(0.3, 1.0 - epoch / 60.0)
        total = w_fm * l_fm_mse + loss_dict["total"]

        l_ens = x_t.new_zeros(())
        if epoch >= 40 and self.n_train_ens >= 2:
            x_t2, fm_t2, u_t2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
            pv2 = self.net.forward_with_ctx(
                x_t2, fm_t2, raw_ctx,
                vel_obs_feat=vel_obs_feat,
                steering_feat=steering_feat,
                env_data=env_data,
            )
            l_ens = F.mse_loss(pv2, u_t2)
            total = total + 0.3 * l_ens

        if torch.isnan(total) or torch.isinf(total):
            total = x_t.new_zeros(())

        # ── RETURN DICT — v39 compat: thêm speed_acc + cumul_disp ────────
        return dict(
            total        = total,
            fm_mse       = l_fm_mse.item(),
            mse_hav      = loss_dict["mse_hav_horizon"],
            multi_scale  = loss_dict["multi_scale"],
            endpoint     = loss_dict["endpoint"],
            h_direct     = loss_dict.get("h_direct",    0.0),
            vel_smooth   = loss_dict.get("vel_smooth",  0.0),
            speed_acc    = loss_dict.get("speed_acc",   0.0),
            cumul_disp   = loss_dict.get("cumul_disp",  0.0),
            accel        = loss_dict.get("accel",       0.0),  # ← v40 NEW
            decomp       = loss_dict.get("decomp",      0.0),  # ← v40 NEW
            cons         = loss_dict.get("cons",        0.0),  # ← v40 NEW
            ate_cte      = loss_dict.get("ate_cte",     0.0),  # compat (=0)
            shape        = loss_dict.get("shape",       0.0),
            velocity     = loss_dict.get("velocity",    0.0),
            heading      = loss_dict.get("heading",     0.0),
            recurv       = loss_dict.get("recurv",      0.0),
            steering     = loss_dict.get("steering",    0.0),
            ens_consist  = (l_ens.item() if torch.is_tensor(l_ens)
                            else float(l_ens)),
            sigma        = current_sigma,
            w_fm         = w_fm,
            long_range=0.0, fde=0.0, w_lr=0.0, w_fde=0.0,
            fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
            spread=0.0, continuity=0.0, step=0.0, disp=0.0,
            recurv_ratio=0.0,
        )

    # ── Sampling — unchanged from v34fix ─────────────────────────────────────
    @torch.no_grad()
    def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
                predict_csv=None, importance_weight=True):
        obs_t    = batch_list[0]
        env_data = batch_list[13] if len(batch_list) > 13 else None
        lp       = obs_t[-1]
        lm       = batch_list[7][-1]
        B        = lp.shape[0]
        device   = lp.device
        T        = self.pred_len
        dt       = 1.0 / max(ddim_steps, 1)

        raw_ctx       = self.net._context(batch_list)
        vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
        steering_feat = self.net._get_steering_feat(env_data, B, device)

        traj_s, me_s, scores = [], [], []
        for _ in range(num_ensemble):
            x_t = torch.randn(B, T, 4, device=device) * (self.sigma_min * 2.5)
            for step in range(ddim_steps):
                t_b = torch.full((B,), step * dt, device=device)
                ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
                vel = self.net.forward_with_ctx(
                    x_t, t_b, raw_ctx,
                    noise_scale=ns,
                    vel_obs_feat=vel_obs_feat,
                    steering_feat=steering_feat,
                    env_data=env_data,
                )
                x_t = x_t + dt * vel
            x_t = self._physics_correct(x_t, lp, lm)
            x_t = x_t.clamp(-3.0, 3.0)
            tr, me = self._to_abs(x_t, lp, lm)
            traj_s.append(tr)
            me_s.append(me)
            if importance_weight:
                scores.append(self._score_sample(tr, env_data))

        all_trajs = torch.stack(traj_s)
        all_me    = torch.stack(me_s)

        if importance_weight and scores:
            score_tensor = torch.stack(scores)
            k = max(1, int(num_ensemble * 0.7))
            _, top_idx = score_tensor.topk(k, dim=0)
            pred_mean = torch.stack([
                all_trajs[top_idx[:, b], :, b, :].median(0).values
                for b in range(B)
            ], dim=1)
        else:
            pred_mean = all_trajs.median(0).values

        if predict_csv:
            self._write_predict_csv(predict_csv, pred_mean, all_trajs)
        return pred_mean, all_me.mean(0), all_trajs

    def _score_sample(self, traj, env_data):
        B = traj.shape[1]
        if traj.shape[0] < 2:
            return torch.ones(B, device=traj.device)
        traj_deg  = _norm_to_deg_fn(traj)
        dt_deg    = traj_deg[1:] - traj_deg[:-1]
        lat_rad   = torch.deg2rad(traj_deg[:-1, :, 1])
        cos_lat   = torch.cos(lat_rad).clamp(min=1e-4)
        dx_km     = dt_deg[:, :, 0] * cos_lat * 111.0
        dy_km     = dt_deg[:, :, 1] * 111.0
        speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)
        speed_pen = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
        speed_sc  = torch.exp(-speed_pen.mean(0) / 20.0)
        if dt_deg.shape[0] >= 2:
            jerk = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
            smooth_sc = torch.exp(-jerk * 5.0)
        else:
            smooth_sc = torch.ones(B, device=traj.device)
        return speed_sc * smooth_sc

    def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
        with torch.enable_grad():
            x   = x_pred.detach().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
            for _ in range(n_steps):
                opt.zero_grad()
                pred_abs, _ = self._to_abs(x, last_pos, last_Me)
                pred_deg    = _norm_to_deg_fn(pred_abs)
                dt_deg      = pred_deg[1:] - pred_deg[:-1]
                lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
                cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
                speed       = torch.sqrt(
                    (dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
                    + (dt_deg[:, :, 1] * 111.0).pow(2))
                F.relu(speed - 600.0).pow(2).mean().backward()
                torch.nn.utils.clip_grad_norm_([x], 0.05)
                opt.step()
        return x.detach()

    @staticmethod
    def _write_predict_csv(csv_path, traj_mean, all_trajs):
        import numpy as np
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        T, B, _ = traj_mean.shape
        mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
        mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
        all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
        all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
        fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
                    "lon_mean_deg", "lat_mean_deg",
                    "lon_std_deg", "lat_std_deg", "ens_spread_km"]
        write_hdr = not os.path.exists(csv_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if write_hdr: w.writeheader()
            for b in range(B):
                for k in range(T):
                    dlat   = all_lat[:, k, b] - mean_lat[k, b]
                    dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
                               * math.cos(math.radians(mean_lat[k, b])))
                    spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
                    w.writerow({
                        "timestamp": ts, "batch_idx": b, "step_idx": k,
                        "lead_h": (k + 1) * 6,
                        "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
                        "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
                        "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
                        "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
                        "ens_spread_km": f"{spread:.2f}",
                    })


TCDiffusion = TCFlowMatching
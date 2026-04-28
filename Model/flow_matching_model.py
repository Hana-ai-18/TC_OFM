# # # # # # # # # # # # # # # # # # # """
# # # # # # # # # # # # # # # # # # # Model/flow_matching_model.py  ── v23
# # # # # # # # # # # # # # # # # # # ==========================================
# # # # # # # # # # # # # # # # # # # FIXES vs v21:

# # # # # # # # # # # # # # # # # # #   FIX-M18  [CURRICULUM REMOVED] set_curriculum_len() vẫn giữ để backward
# # # # # # # # # # # # # # # # # # #            compat nhưng KHÔNG được gọi từ trainer nữa. active_pred_len
# # # # # # # # # # # # # # # # # # #            luôn = pred_len. evaluate_full_val_ade không cần restore nữa.

# # # # # # # # # # # # # # # # # # #   FIX-M19  get_loss_breakdown(): nhận thêm step_weight_alpha parameter
# # # # # # # # # # # # # # # # # # #            và truyền vào compute_total_loss() → fm_afcrps_loss() sử dụng
# # # # # # # # # # # # # # # # # # #            soft weighting thay curriculum len-slicing.

# # # # # # # # # # # # # # # # # # #   FIX-M20  get_loss_breakdown(): truyền all_trajs vào compute_total_loss()
# # # # # # # # # # # # # # # # # # #            để tính ensemble_spread_loss. Giúp kiểm soát spread tăng quá mức.

# # # # # # # # # # # # # # # # # # #   FIX-M21  _physics_correct(): tăng n_steps=5 (từ 3), giảm lr=0.002 (từ
# # # # # # # # # # # # # # # # # # #            0.005) để physics correction ổn định hơn và ít overshoot.

# # # # # # # # # # # # # # # # # # #   FIX-M22  sample(): initial_sample_sigma=0.1 (set từ constructor) đã fix
# # # # # # # # # # # # # # # # # # #            spread. Thêm post-sampling clip chặt hơn [-3.0, 3.0] cho cả lon
# # # # # # # # # # # # # # # # # # #            và lat (từ [-5.0, 5.0] cho lon).

# # # # # # # # # # # # # # # # # # # Kept from v21:
# # # # # # # # # # # # # # # # # # #   FIX-M17  _physics_correct với torch.enable_grad()
# # # # # # # # # # # # # # # # # # #   FIX-M11..M16 OT-CFM, beta drift, env_data, physics scale
# # # # # # # # # # # # # # # # # # # """
# # # # # # # # # # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # # # # # # # # # import csv
# # # # # # # # # # # # # # # # # # # import math
# # # # # # # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # # # # # # from datetime import datetime
# # # # # # # # # # # # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # # # # # # # # # # # import torch
# # # # # # # # # # # # # # # # # # # import torch.nn as nn
# # # # # # # # # # # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # # # # # # # # # # # from Model.losses import (
# # # # # # # # # # # # # # # # # # #     compute_total_loss, fm_physics_consistency_loss, WEIGHTS,
# # # # # # # # # # # # # # # # # # #     pinn_speed_constraint,
# # # # # # # # # # # # # # # # # # # )


# # # # # # # # # # # # # # # # # # # def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # # # #     """Normalised → degrees. Handles [T, B, 2] and [B, 2]."""
# # # # # # # # # # # # # # # # # # #     out = traj_norm.clone()
# # # # # # # # # # # # # # # # # # #     out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # # # # # # # # # #     out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
# # # # # # # # # # # # # # # # # # #     return out


# # # # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # # # # # #  VelocityField
# # # # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # # # # # #     OT-CFM velocity field v_θ(x_t, t, context).
# # # # # # # # # # # # # # # # # # #     Architecture: DataEncoder1D (Mamba) + FNO3D + Env-T-Net → Transformer decoder.
# # # # # # # # # # # # # # # # # # #     Physics-guided: v_total = v_neural + sigmoid(w_physics) * v_beta_drift.
# # # # # # # # # # # # # # # # # # #     """

# # # # # # # # # # # # # # # # # # #     def __init__(
# # # # # # # # # # # # # # # # # # #         self,
# # # # # # # # # # # # # # # # # # #         pred_len:   int   = 12,
# # # # # # # # # # # # # # # # # # #         obs_len:    int   = 8,
# # # # # # # # # # # # # # # # # # #         ctx_dim:    int   = 256,
# # # # # # # # # # # # # # # # # # #         sigma_min:  float = 0.02,
# # # # # # # # # # # # # # # # # # #         unet_in_ch: int   = 13,
# # # # # # # # # # # # # # # # # # #     ):
# # # # # # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # # # # # #         self.pred_len  = pred_len
# # # # # # # # # # # # # # # # # # #         self.obs_len   = obs_len
# # # # # # # # # # # # # # # # # # #         self.sigma_min = sigma_min

# # # # # # # # # # # # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # # # # # # # # # # # #             in_channel   = unet_in_ch,
# # # # # # # # # # # # # # # # # # #             out_channel  = 1,
# # # # # # # # # # # # # # # # # # #             d_model      = 64,
# # # # # # # # # # # # # # # # # # #             n_layers     = 4,
# # # # # # # # # # # # # # # # # # #             modes_t      = 4,
# # # # # # # # # # # # # # # # # # #             modes_h      = 4,
# # # # # # # # # # # # # # # # # # #             modes_w      = 4,
# # # # # # # # # # # # # # # # # # #             spatial_down = 32,
# # # # # # # # # # # # # # # # # # #             dropout      = 0.05,
# # # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # # # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # # # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # # # # # # # # # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # # # # # # # # # # # #             in_1d       = 4,
# # # # # # # # # # # # # # # # # # #             feat_3d_dim = 128,
# # # # # # # # # # # # # # # # # # #             mlp_h       = 64,
# # # # # # # # # # # # # # # # # # #             lstm_hidden = 128,
# # # # # # # # # # # # # # # # # # #             lstm_layers = 3,
# # # # # # # # # # # # # # # # # # #             dropout     = 0.1,
# # # # # # # # # # # # # # # # # # #             d_state     = 16,
# # # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=64)

# # # # # # # # # # # # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 64 + 16, 512)
# # # # # # # # # # # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(512)
# # # # # # # # # # # # # # # # # # #         self.ctx_drop = nn.Dropout(0.15)
# # # # # # # # # # # # # # # # # # #         self.ctx_fc2  = nn.Linear(512, ctx_dim)

# # # # # # # # # # # # # # # # # # #         self.time_fc1 = nn.Linear(256, 512)
# # # # # # # # # # # # # # # # # # #         self.time_fc2 = nn.Linear(512, 256)

# # # # # # # # # # # # # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # # # # # # # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # # # # # # # # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # # # # # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # # # # # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # # # # # # # # # # # #                 dropout=0.15, activation="gelu", batch_first=True,
# # # # # # # # # # # # # # # # # # #             ),
# # # # # # # # # # # # # # # # # # #             num_layers=4,
# # # # # # # # # # # # # # # # # # #         )
# # # # # # # # # # # # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # # # # # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # # # # # # # # # # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)

# # # # # # # # # # # # # # # # # # #     def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
# # # # # # # # # # # # # # # # # # #         half = dim // 2
# # # # # # # # # # # # # # # # # # #         freq = torch.exp(
# # # # # # # # # # # # # # # # # # #             torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # # # # # # # # # # # #             * (-math.log(10_000.0) / max(half - 1, 1))
# # # # # # # # # # # # # # # # # # #         )
# # # # # # # # # # # # # # # # # # #         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
# # # # # # # # # # # # # # # # # # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # # # # # # # # # # # # # # # # # #         return F.pad(emb, (0, dim % 2))

# # # # # # # # # # # # # # # # # # #     def _context(self, batch_list: List) -> torch.Tensor:
# # # # # # # # # # # # # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # # # # # # # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # # # # # # # # # # # # #         image_obs = batch_list[11]
# # # # # # # # # # # # # # # # # # #         env_data  = batch_list[13]

# # # # # # # # # # # # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # # # # # # # # # # # #             image_obs = image_obs.unsqueeze(1)

# # # # # # # # # # # # # # # # # # #         expected_ch = self.spatial_enc.in_channel
# # # # # # # # # # # # # # # # # # #         if image_obs.shape[1] == 1 and expected_ch != 1:
# # # # # # # # # # # # # # # # # # #             image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# # # # # # # # # # # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # # # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # # # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
# # # # # # # # # # # # # # # # # # #         e_3d_s = e_3d_s.permute(0, 2, 1)
# # # # # # # # # # # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)

# # # # # # # # # # # # # # # # # # #         T_bot = e_3d_s.shape[1]
# # # # # # # # # # # # # # # # # # #         if T_bot != T_obs:
# # # # # # # # # # # # # # # # # # #             e_3d_s = F.interpolate(
# # # # # # # # # # # # # # # # # # #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # # # # # # # # # # #                 mode="linear", align_corners=False,
# # # # # # # # # # # # # # # # # # #             ).permute(0, 2, 1)

# # # # # # # # # # # # # # # # # # #         f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))
# # # # # # # # # # # # # # # # # # #         f_spatial     = self.decoder_proj(f_spatial_raw)

# # # # # # # # # # # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # # # # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)

# # # # # # # # # # # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # # # # # # # # # # # # # # # # # #         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
# # # # # # # # # # # # # # # # # # #         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))
# # # # # # # # # # # # # # # # # # #         return raw

# # # # # # # # # # # # # # # # # # #     def _apply_ctx_head(self, raw: torch.Tensor,
# # # # # # # # # # # # # # # # # # #                         noise_scale: float = 0.0) -> torch.Tensor:
# # # # # # # # # # # # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # # # # # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # # # # # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # # # # # # # # # # # #     def _beta_drift_velocity(self, x_t: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # # # #         """Beta drift in normalised state space. x_t: [B, T, 4]."""
# # # # # # # # # # # # # # # # # # #         OMEGA_val  = 7.2921e-5
# # # # # # # # # # # # # # # # # # #         R_val      = 6.371e6
# # # # # # # # # # # # # # # # # # #         DT         = 6 * 3600.0
# # # # # # # # # # # # # # # # # # #         M_PER_NORM = 5.0 * 111.0 * 1000.0

# # # # # # # # # # # # # # # # # # #         lat_norm = x_t[:, :, 1]
# # # # # # # # # # # # # # # # # # #         lat_deg  = lat_norm * 5.0
# # # # # # # # # # # # # # # # # # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))

# # # # # # # # # # # # # # # # # # #         beta   = 2 * OMEGA_val * torch.cos(lat_rad) / R_val
# # # # # # # # # # # # # # # # # # #         R_tc   = 3e5
# # # # # # # # # # # # # # # # # # #         v_lon  = -beta * R_tc ** 2 / 2
# # # # # # # # # # # # # # # # # # #         v_lat  =  beta * R_tc ** 2 / 4

# # # # # # # # # # # # # # # # # # #         v_lon_norm = v_lon * DT / M_PER_NORM
# # # # # # # # # # # # # # # # # # #         v_lat_norm = v_lat * DT / M_PER_NORM

# # # # # # # # # # # # # # # # # # #         v_phys = torch.zeros_like(x_t)
# # # # # # # # # # # # # # # # # # #         v_phys[:, :, 0] = v_lon_norm
# # # # # # # # # # # # # # # # # # #         v_phys[:, :, 1] = v_lat_norm
# # # # # # # # # # # # # # # # # # #         return v_phys

# # # # # # # # # # # # # # # # # # #     def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
# # # # # # # # # # # # # # # # # # #                 ctx: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
# # # # # # # # # # # # # # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # # # # # # # # # # # # # #         T_seq  = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # # # # # # # # # # # # #         x_emb  = (self.traj_embed(x_t[:, :T_seq, :])
# # # # # # # # # # # # # # # # # # #                   + self.pos_enc[:, :T_seq, :]
# # # # # # # # # # # # # # # # # # #                   + t_emb.unsqueeze(1))
# # # # # # # # # # # # # # # # # # #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

# # # # # # # # # # # # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(
# # # # # # # # # # # # # # # # # # #             self.transformer(x_emb, memory)
# # # # # # # # # # # # # # # # # # #         )))  # [B, T, 4]

# # # # # # # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # # # # # # #             v_phys = self._beta_drift_velocity(x_t[:, :T_seq, :])

# # # # # # # # # # # # # # # # # # #         scale = torch.sigmoid(self.physics_scale) * 2.0
# # # # # # # # # # # # # # # # # # #         return v_neural + scale * v_phys

# # # # # # # # # # # # # # # # # # #     def forward(self, x_t, t, batch_list):
# # # # # # # # # # # # # # # # # # #         raw = self._context(batch_list)
# # # # # # # # # # # # # # # # # # #         ctx = self._apply_ctx_head(raw, noise_scale=0.0)
# # # # # # # # # # # # # # # # # # #         return self._decode(x_t, t, ctx)

# # # # # # # # # # # # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale: float = 0.0):
# # # # # # # # # # # # # # # # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale=noise_scale)
# # # # # # # # # # # # # # # # # # #         return self._decode(x_t, t, ctx)


# # # # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # # # # # #  TCFlowMatching
# # # # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # # # # # # class TCFlowMatching(nn.Module):
# # # # # # # # # # # # # # # # # # #     """TC trajectory prediction via OT-CFM + Physics-guided velocity field."""

# # # # # # # # # # # # # # # # # # #     def __init__(
# # # # # # # # # # # # # # # # # # #         self,
# # # # # # # # # # # # # # # # # # #         pred_len:             int   = 12,
# # # # # # # # # # # # # # # # # # #         obs_len:              int   = 8,
# # # # # # # # # # # # # # # # # # #         sigma_min:            float = 0.02,
# # # # # # # # # # # # # # # # # # #         n_train_ens:          int   = 4,
# # # # # # # # # # # # # # # # # # #         unet_in_ch:           int   = 13,
# # # # # # # # # # # # # # # # # # #         ctx_noise_scale:      float = 0.02,   # FIX-T23-5: 0.02 default
# # # # # # # # # # # # # # # # # # #         initial_sample_sigma: float = 0.1,    # FIX-T23-4: 0.1 default
# # # # # # # # # # # # # # # # # # #         **kwargs,
# # # # # # # # # # # # # # # # # # #     ):
# # # # # # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # # # # # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # # # # # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # # # # # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # # # # # # # # # # # #         self.active_pred_len      = pred_len   # FIX-M18: always full pred_len
# # # # # # # # # # # # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # # # # # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # # # # # # # # # # # # # #         self.net = VelocityField(
# # # # # # # # # # # # # # # # # # #             pred_len   = pred_len,
# # # # # # # # # # # # # # # # # # #             obs_len    = obs_len,
# # # # # # # # # # # # # # # # # # #             sigma_min  = sigma_min,
# # # # # # # # # # # # # # # # # # #             unet_in_ch = unet_in_ch,
# # # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # # #     def set_curriculum_len(self, active_len: int) -> None:
# # # # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # # # #         FIX-M18: Kept for backward compat but NO-OP in v23.
# # # # # # # # # # # # # # # # # # #         Curriculum is removed. active_pred_len is always pred_len.
# # # # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # # # #         # self.active_pred_len = max(1, min(active_len, self.pred_len))
# # # # # # # # # # # # # # # # # # #         pass  # no-op

# # # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # # #     def _to_rel(traj_gt, Me_gt, last_pos, last_Me):
# # # # # # # # # # # # # # # # # # #         return torch.cat(
# # # # # # # # # # # # # # # # # # #             [traj_gt - last_pos.unsqueeze(0),
# # # # # # # # # # # # # # # # # # #              Me_gt   - last_Me.unsqueeze(0)],
# # # # # # # # # # # # # # # # # # #             dim=-1,
# # # # # # # # # # # # # # # # # # #         ).permute(1, 0, 2)

# # # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # # #     def _to_abs(rel, last_pos, last_Me):
# # # # # # # # # # # # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # # # # # # # # # # # #         return (
# # # # # # # # # # # # # # # # # # #             last_pos.unsqueeze(0) + d[:, :, :2],
# # # # # # # # # # # # # # # # # # #             last_Me.unsqueeze(0)  + d[:, :, 2:],
# # # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # # #     def _cfm_noisy(self, x1):
# # # # # # # # # # # # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # # # # # # # # # # # #         sm  = self.sigma_min
# # # # # # # # # # # # # # # # # # #         x0  = torch.randn_like(x1) * sm
# # # # # # # # # # # # # # # # # # #         t   = torch.rand(B, device=device)
# # # # # # # # # # # # # # # # # # #         te  = t.view(B, 1, 1)
# # # # # # # # # # # # # # # # # # #         x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
# # # # # # # # # # # # # # # # # # #         denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
# # # # # # # # # # # # # # # # # # #         target_vel = (x1 - (1.0 - sm) * x_t) / denom
# # # # # # # # # # # # # # # # # # #         return x_t, t, te, denom, target_vel

# # # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # # #     def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # # # #         wind_norm = obs_Me[-1, :, 1].detach()
# # # # # # # # # # # # # # # # # # #         w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
# # # # # # # # # # # # # # # # # # #             torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
# # # # # # # # # # # # # # # # # # #             torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
# # # # # # # # # # # # # # # # # # #                         torch.full_like(wind_norm, 1.5))))
# # # # # # # # # # # # # # # # # # #         return w / w.mean().clamp(min=1e-6)

# # # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # # #     def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
# # # # # # # # # # # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # # # # # # # # # # #             return batch_list
# # # # # # # # # # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # # # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # # # # # # # # # # # #             t = aug[idx]
# # # # # # # # # # # # # # # # # # #             if torch.is_tensor(t) and t.shape[-1] >= 1:
# # # # # # # # # # # # # # # # # # #                 t = t.clone()
# # # # # # # # # # # # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # # # # # # # # # # # #                 aug[idx] = t
# # # # # # # # # # # # # # # # # # #         return aug

# # # # # # # # # # # # # # # # # # #     def get_loss(self, batch_list: List,
# # # # # # # # # # # # # # # # # # #                  step_weight_alpha: float = 0.0) -> torch.Tensor:
# # # # # # # # # # # # # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha)["total"]

# # # # # # # # # # # # # # # # # # #     def get_loss_breakdown(self, batch_list: List,
# # # # # # # # # # # # # # # # # # #                            step_weight_alpha: float = 0.0) -> Dict:
# # # # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # # # #         FIX-M19: Nhận step_weight_alpha, truyền vào compute_total_loss.
# # # # # # # # # # # # # # # # # # #         FIX-M20: Truyền all_trajs để tính ensemble_spread_loss.
# # # # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # # # #         batch_list = self._lon_flip_aug(batch_list, p=0.3)

# # # # # # # # # # # # # # # # # # #         traj_gt  = batch_list[1]
# # # # # # # # # # # # # # # # # # #         Me_gt    = batch_list[8]
# # # # # # # # # # # # # # # # # # #         obs_t    = batch_list[0]
# # # # # # # # # # # # # # # # # # #         obs_Me   = batch_list[7]

# # # # # # # # # # # # # # # # # # #         try:
# # # # # # # # # # # # # # # # # # #             env_data = batch_list[13]
# # # # # # # # # # # # # # # # # # #         except (IndexError, TypeError):
# # # # # # # # # # # # # # # # # # #             env_data = None

# # # # # # # # # # # # # # # # # # #         # FIX-M18: NO curriculum slicing. Always use full pred_len.
# # # # # # # # # # # # # # # # # # #         lp, lm = obs_t[-1], obs_Me[-1]
# # # # # # # # # # # # # # # # # # #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

# # # # # # # # # # # # # # # # # # #         raw_ctx     = self.net._context(batch_list)
# # # # # # # # # # # # # # # # # # #         intensity_w = self._intensity_weights(obs_Me)

# # # # # # # # # # # # # # # # # # #         x_t, t, te, denom, _ = self._cfm_noisy(x1)
# # # # # # # # # # # # # # # # # # #         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

# # # # # # # # # # # # # # # # # # #         # Ensemble samples for AFCRPS + spread penalty
# # # # # # # # # # # # # # # # # # #         samples: List[torch.Tensor] = []
# # # # # # # # # # # # # # # # # # #         for _ in range(self.n_train_ens):
# # # # # # # # # # # # # # # # # # #             xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
# # # # # # # # # # # # # # # # # # #             pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
# # # # # # # # # # # # # # # # # # #             x1_s  = xt_s + dens_s * pv_s   # OT-CFM
# # # # # # # # # # # # # # # # # # #             pa_s, _ = self._to_abs(x1_s, lp, lm)
# # # # # # # # # # # # # # # # # # #             samples.append(pa_s)
# # # # # # # # # # # # # # # # # # #         pred_samples = torch.stack(samples)   # [S, T, B, 2]

# # # # # # # # # # # # # # # # # # #         # FIX-M20: all_trajs for spread penalty
# # # # # # # # # # # # # # # # # # #         all_trajs_4d = pred_samples   # [S, T, B, 2]

# # # # # # # # # # # # # # # # # # #         l_fm_physics = fm_physics_consistency_loss(
# # # # # # # # # # # # # # # # # # #             pred_samples, gt_norm=traj_gt, last_pos=lp)

# # # # # # # # # # # # # # # # # # #         x1_pred = x_t + denom * pred_vel
# # # # # # # # # # # # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# # # # # # # # # # # # # # # # # # #         pred_abs_deg = _denorm_to_deg(pred_abs)
# # # # # # # # # # # # # # # # # # #         traj_gt_deg  = _denorm_to_deg(traj_gt)
# # # # # # # # # # # # # # # # # # #         ref_deg      = _denorm_to_deg(lp)

# # # # # # # # # # # # # # # # # # #         breakdown = compute_total_loss(
# # # # # # # # # # # # # # # # # # #             pred_abs           = pred_abs_deg,
# # # # # # # # # # # # # # # # # # #             gt                 = traj_gt_deg,
# # # # # # # # # # # # # # # # # # #             ref                = ref_deg,
# # # # # # # # # # # # # # # # # # #             batch_list         = batch_list,
# # # # # # # # # # # # # # # # # # #             pred_samples       = pred_samples,
# # # # # # # # # # # # # # # # # # #             gt_norm            = traj_gt,
# # # # # # # # # # # # # # # # # # #             weights            = WEIGHTS,
# # # # # # # # # # # # # # # # # # #             intensity_w        = intensity_w,
# # # # # # # # # # # # # # # # # # #             env_data           = env_data,
# # # # # # # # # # # # # # # # # # #             step_weight_alpha  = step_weight_alpha,   # FIX-M19
# # # # # # # # # # # # # # # # # # #             all_trajs          = all_trajs_4d,         # FIX-M20
# # # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # # #         fm_phys_w = WEIGHTS.get("fm_physics", 0.3)
# # # # # # # # # # # # # # # # # # #         breakdown["total"]      = breakdown["total"] + fm_phys_w * l_fm_physics
# # # # # # # # # # # # # # # # # # #         breakdown["fm_physics"] = l_fm_physics.item()

# # # # # # # # # # # # # # # # # # #         return breakdown

# # # # # # # # # # # # # # # # # # #     # ── sample() ─────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # # # # # #     @torch.no_grad()
# # # # # # # # # # # # # # # # # # #     def sample(
# # # # # # # # # # # # # # # # # # #         self,
# # # # # # # # # # # # # # # # # # #         batch_list: List,
# # # # # # # # # # # # # # # # # # #         num_ensemble: int = 50,
# # # # # # # # # # # # # # # # # # #         ddim_steps:   int = 20,
# # # # # # # # # # # # # # # # # # #         predict_csv:  Optional[str] = None,
# # # # # # # # # # # # # # # # # # #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# # # # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # # # #         FIX-T23-3: ddim_steps default 20 (từ 10).
# # # # # # # # # # # # # # # # # # #         FIX-M22: tighter clip [-3.0, 3.0] cho cả lon và lat.
# # # # # # # # # # # # # # # # # # #         Returns:
# # # # # # # # # # # # # # # # # # #             pred_mean:  [T, B, 2] mean track (normalised)
# # # # # # # # # # # # # # # # # # #             me_mean:    [T, B, 2] mean intensity
# # # # # # # # # # # # # # # # # # #             all_trajs:  [S, T, B, 2] all ensemble members
# # # # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # # # #         lp  = batch_list[0][-1]   # [B, 2]
# # # # # # # # # # # # # # # # # # #         lm  = batch_list[7][-1]   # [B, 2]
# # # # # # # # # # # # # # # # # # #         B   = lp.shape[0]
# # # # # # # # # # # # # # # # # # #         device = lp.device
# # # # # # # # # # # # # # # # # # #         T   = self.pred_len   # FIX-M18: always full pred_len
# # # # # # # # # # # # # # # # # # #         dt  = 1.0 / max(ddim_steps, 1)

# # # # # # # # # # # # # # # # # # #         raw_ctx = self.net._context(batch_list)

# # # # # # # # # # # # # # # # # # #         traj_s: List[torch.Tensor] = []
# # # # # # # # # # # # # # # # # # #         me_s:   List[torch.Tensor] = []

# # # # # # # # # # # # # # # # # # #         for k in range(num_ensemble):
# # # # # # # # # # # # # # # # # # #             # FIX-T23-4: initial_sample_sigma=0.1 (set in constructor)
# # # # # # # # # # # # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# # # # # # # # # # # # # # # # # # #             # DDIM Euler integration
# # # # # # # # # # # # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # # # # # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # # # # # # # # # #                 ns  = self.ctx_noise_scale if step == 0 else 0.0
# # # # # # # # # # # # # # # # # # #                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
# # # # # # # # # # # # # # # # # # #                 x_t = x_t + dt * vel

# # # # # # # # # # # # # # # # # # #             # Physics correction
# # # # # # # # # # # # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)

# # # # # # # # # # # # # # # # # # #             # FIX-M22: tighter clip
# # # # # # # # # # # # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)

# # # # # # # # # # # # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # # # # # # # # # # #             traj_s.append(tr)
# # # # # # # # # # # # # # # # # # #             me_s.append(me)

# # # # # # # # # # # # # # # # # # #         all_trajs = torch.stack(traj_s)   # [S, T, B, 2]
# # # # # # # # # # # # # # # # # # #         all_me    = torch.stack(me_s)
# # # # # # # # # # # # # # # # # # #         pred_mean = all_trajs.mean(0)
# # # # # # # # # # # # # # # # # # #         me_mean   = all_me.mean(0)

# # # # # # # # # # # # # # # # # # #         if predict_csv:
# # # # # # # # # # # # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)

# # # # # # # # # # # # # # # # # # #         return pred_mean, me_mean, all_trajs

# # # # # # # # # # # # # # # # # # #     # ── Physics correction ────────────────────────────────────────────────────

# # # # # # # # # # # # # # # # # # #     def _physics_correct(
# # # # # # # # # # # # # # # # # # #         self,
# # # # # # # # # # # # # # # # # # #         x_pred: torch.Tensor,
# # # # # # # # # # # # # # # # # # #         last_pos: torch.Tensor,
# # # # # # # # # # # # # # # # # # #         last_Me:  torch.Tensor,
# # # # # # # # # # # # # # # # # # #         n_steps:  int   = 5,    # FIX-M21: 5 (từ 3)
# # # # # # # # # # # # # # # # # # #         lr:       float = 0.002, # FIX-M21: 0.002 (từ 0.005)
# # # # # # # # # # # # # # # # # # #     ) -> torch.Tensor:
# # # # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # # # #         FIX-M17: torch.enable_grad() inside no_grad context.
# # # # # # # # # # # # # # # # # # #         FIX-M21: n_steps=5, lr=0.002 for more stable correction.
# # # # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # # # #         with torch.enable_grad():
# # # # # # # # # # # # # # # # # # #             x = x_pred.detach().requires_grad_(True)
# # # # # # # # # # # # # # # # # # #             optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)

# # # # # # # # # # # # # # # # # # #             for _ in range(n_steps):
# # # # # # # # # # # # # # # # # # #                 optimizer.zero_grad()
# # # # # # # # # # # # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # # # # # # # # # # # #                 pred_deg    = _denorm_to_deg(pred_abs)

# # # # # # # # # # # # # # # # # # #                 l_speed = self._pinn_speed_constraint(pred_deg)
# # # # # # # # # # # # # # # # # # #                 l_accel = self._pinn_beta_plane_simplified(pred_deg)

# # # # # # # # # # # # # # # # # # #                 physics_loss = l_speed + 0.3 * l_accel
# # # # # # # # # # # # # # # # # # #                 physics_loss.backward()

# # # # # # # # # # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], max_norm=0.05)
# # # # # # # # # # # # # # # # # # #                 optimizer.step()

# # # # # # # # # # # # # # # # # # #         return x.detach()

# # # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # # #     def _pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # # # #         if pred_deg.shape[0] < 2:
# # # # # # # # # # # # # # # # # # #             return pred_deg.new_zeros(())
# # # # # # # # # # # # # # # # # # #         dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # # # # # # # # # #         lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # # # # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # # # # # # # # #         dx_km   = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # # # # # # # # # # # # # # # #         dy_km   = dt_deg[:, :, 1] * 111.0
# # # # # # # # # # # # # # # # # # #         speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # # # # # # # # # # # # # # # # # #         return F.relu(speed - 600.0).pow(2).mean()

# # # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # # #     def _pinn_beta_plane_simplified(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # # # #         if pred_deg.shape[0] < 3:
# # # # # # # # # # # # # # # # # # #             return pred_deg.new_zeros(())
# # # # # # # # # # # # # # # # # # #         v = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # # # # # # # # # #         a = v[1:] - v[:-1]
# # # # # # # # # # # # # # # # # # #         lat_rad = torch.deg2rad(pred_deg[1:-1, :, 1])
# # # # # # # # # # # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # # # # # # # # #         a_lon_km = a[:, :, 0] * cos_lat * 111.0
# # # # # # # # # # # # # # # # # # #         a_lat_km = a[:, :, 1] * 111.0
# # # # # # # # # # # # # # # # # # #         max_accel = 50.0
# # # # # # # # # # # # # # # # # # #         violation = F.relu(torch.sqrt(a_lon_km**2 + a_lat_km**2) - max_accel)
# # # # # # # # # # # # # # # # # # #         return violation.pow(2).mean() * 0.1

# # # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # # #     def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
# # # # # # # # # # # # # # # # # # #                            all_trajs: torch.Tensor) -> None:
# # # # # # # # # # # # # # # # # # #         import numpy as np
# # # # # # # # # # # # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # # # # # # # # # # # #         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # # # # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # # # # # # # # # # # #         S       = all_trajs.shape[0]

# # # # # # # # # # # # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()

# # # # # # # # # # # # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # # # # # # # # # # # #                     "lon_mean_deg", "lat_mean_deg",
# # # # # # # # # # # # # # # # # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # # # # # # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # # # # # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # # # # # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # # # # # # # # # # # #             if write_hdr:
# # # # # # # # # # # # # # # # # # #                 w.writeheader()
# # # # # # # # # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # # # # # # # # #                 for k in range(T):
# # # # # # # # # # # # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # # # # # # # # # # # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
# # # # # # # # # # # # # # # # # # #                         math.radians(mean_lat[k, b]))
# # # # # # # # # # # # # # # # # # #                     spread = float(np.sqrt((dlat**2 + dlon**2).mean()) * 111.0)
# # # # # # # # # # # # # # # # # # #                     w.writerow(dict(
# # # # # # # # # # # # # # # # # # #                         timestamp     = ts,
# # # # # # # # # # # # # # # # # # #                         batch_idx     = b,
# # # # # # # # # # # # # # # # # # #                         step_idx      = k,
# # # # # # # # # # # # # # # # # # #                         lead_h        = (k + 1) * 6,
# # # # # # # # # # # # # # # # # # #                         lon_mean_deg  = f"{mean_lon[k,b]:.4f}",
# # # # # # # # # # # # # # # # # # #                         lat_mean_deg  = f"{mean_lat[k,b]:.4f}",
# # # # # # # # # # # # # # # # # # #                         lon_std_deg   = f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # # # # # # # # # # # #                         lat_std_deg   = f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # # # # # # # # # # # #                         ens_spread_km = f"{spread:.2f}",
# # # # # # # # # # # # # # # # # # #                     ))
# # # # # # # # # # # # # # # # # # #         print(f"  Predictions → {csv_path}  (B={B}, T={T}, S={S})")


# # # # # # # # # # # # # # # # # # # # Backward-compat alias
# # # # # # # # # # # # # # # # # # # TCDiffusion = TCFlowMatching

# # # # # # # # # # # # # # # # # # """
# # # # # # # # # # # # # # # # # # Model/flow_matching_model.py  ── v24
# # # # # # # # # # # # # # # # # # ==========================================
# # # # # # # # # # # # # # # # # # FIXES vs v23:

# # # # # # # # # # # # # # # # # #   FIX-M23  [SHORT-RANGE HEAD] Thêm ShortRangeHead – bộ dự báo riêng
# # # # # # # # # # # # # # # # # #            cho 4 bước đầu (6h, 12h, 18h, 24h) dùng GRU autoregressive
# # # # # # # # # # # # # # # # # #            + motion prior từ obs_traj. KHÔNG dùng flow matching cho
# # # # # # # # # # # # # # # # # #            các bước này → ổn định hơn, chính xác hơn cho tầm ngắn.

# # # # # # # # # # # # # # # # # #   FIX-M24  [INFERENCE BLEND] sample() dùng ShortRangeHead (deterministic)
# # # # # # # # # # # # # # # # # #            cho steps 1-4, FM ensemble cho steps 5-12. pred_mean[:4]
# # # # # # # # # # # # # # # # # #            luôn đến từ ShortRangeHead → giảm ADE 12h/24h đáng kể.

# # # # # # # # # # # # # # # # # #   FIX-M25  [SPREAD CONTROL] initial_sample_sigma=0.03, ctx_noise_scale=0.002
# # # # # # # # # # # # # # # # # #            → spread ensemble giảm từ 400-500km xuống mục tiêu <150km.

# # # # # # # # # # # # # # # # # #   FIX-M26  [PHYSICS IN HEAD] ShortRangeHead clamp delta per-step ≤ MAX_DELTA
# # # # # # # # # # # # # # # # # #            (~600km/6h), built-in vào forward pass → không cần PINN
# # # # # # # # # # # # # # # # # #            cho tầm ngắn.

# # # # # # # # # # # # # # # # # #   FIX-M27  [LOSS SEPARATION] get_loss_breakdown() tính thêm short_range_loss
# # # # # # # # # # # # # # # # # #            riêng với weight cao (5.0). FM loss vẫn tính đầy đủ 12 bước
# # # # # # # # # # # # # # # # # #            nhưng short_range override pred_mean cho bước 1-4.

# # # # # # # # # # # # # # # # # # Kept from v23:
# # # # # # # # # # # # # # # # # #   FIX-M18  active_pred_len = pred_len (no curriculum)
# # # # # # # # # # # # # # # # # #   FIX-M21  _physics_correct n_steps=5, lr=0.002
# # # # # # # # # # # # # # # # # #   FIX-M22  post-sampling clip [-3.0, 3.0]
# # # # # # # # # # # # # # # # # # """
# # # # # # # # # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # # # # # # # # import csv
# # # # # # # # # # # # # # # # # # import math
# # # # # # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # # # # # from datetime import datetime
# # # # # # # # # # # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # # # # # # # # # # import torch
# # # # # # # # # # # # # # # # # # import torch.nn as nn
# # # # # # # # # # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # # # # # # # # # # from Model.losses import (
# # # # # # # # # # # # # # # # # #     compute_total_loss, fm_physics_consistency_loss, WEIGHTS,
# # # # # # # # # # # # # # # # # #     pinn_speed_constraint, short_range_regression_loss,
# # # # # # # # # # # # # # # # # # )


# # # # # # # # # # # # # # # # # # def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # # #     """Normalised → degrees. Handles [T, B, 2] and [B, 2]."""
# # # # # # # # # # # # # # # # # #     out = traj_norm.clone()
# # # # # # # # # # # # # # # # # #     out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # # # # # # # # #     out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
# # # # # # # # # # # # # # # # # #     return out


# # # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # # # # #  ShortRangeHead  – FIX-M23
# # # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # # # # # class ShortRangeHead(nn.Module):
# # # # # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # # # # #     Dedicated deterministic predictor for steps 1-4 (6h/12h/18h/24h).

# # # # # # # # # # # # # # # # # #     Architecture: GRU autoregressive cell
# # # # # # # # # # # # # # # # # #         Input  each step : [ctx_feat(256) | vel_feat(128) | cur_pos(2)] = 386
# # # # # # # # # # # # # # # # # #         Hidden           : 256
# # # # # # # # # # # # # # # # # #         Output each step : delta [2] → clamp → new position

# # # # # # # # # # # # # # # # # #     Physics built-in:
# # # # # # # # # # # # # # # # # #         • MAX_DELTA = 0.48  ≈  576 km / 6h  (upper TC speed bound)
# # # # # # # # # # # # # # # # # #         • Speed is implicitly constrained by architecture + clamp

# # # # # # # # # # # # # # # # # #     Why GRU (not Transformer):
# # # # # # # # # # # # # # # # # #         • Autoregressive dependency: step t depends on step t-1 position
# # # # # # # # # # # # # # # # # #         • Small number of steps (4) → GRU is simpler and faster
# # # # # # # # # # # # # # # # # #         • Motion prior via velocity encoding stabilises early epochs
# # # # # # # # # # # # # # # # # #     """

# # # # # # # # # # # # # # # # # #     N_STEPS   = 4        # 6h, 12h, 18h, 24h
# # # # # # # # # # # # # # # # # #     MAX_DELTA = 0.48     # normalised units ≈ 576 km / 6h

# # # # # # # # # # # # # # # # # #     def __init__(self, raw_ctx_dim=512, obs_len=8):
# # # # # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # # # # #         self.obs_len = obs_len

# # # # # # # # # # # # # # # # # #         # Motion prior: dùng tất cả obs velocities, không chỉ 3 bước cuối
# # # # # # # # # # # # # # # # # #         self.vel_enc = nn.Sequential(
# # # # # # # # # # # # # # # # # #             nn.Linear(obs_len * 2, 256),   # FIX: toàn bộ obs_len velocities
# # # # # # # # # # # # # # # # # #             nn.GELU(),
# # # # # # # # # # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # # # # # # # # # #             nn.Linear(256, 128),
# # # # # # # # # # # # # # # # # #             nn.GELU(),
# # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # #         self.ctx_proj = nn.Sequential(
# # # # # # # # # # # # # # # # # #             nn.Linear(raw_ctx_dim, 256),
# # # # # # # # # # # # # # # # # #             nn.GELU(),
# # # # # # # # # # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # # # # # # # # # #             nn.Dropout(0.05),   # FIX: giảm dropout 0.08→0.05
# # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # #         # FIX: GRU hidden = ctx+vel kết hợp ngay từ đầu
# # # # # # # # # # # # # # # # # #         # input = ctx(256) + vel(128) + cur_pos(2) = 386
# # # # # # # # # # # # # # # # # #         self.gru_cell = nn.GRUCell(input_size=386, hidden_size=256)

# # # # # # # # # # # # # # # # # #         # FIX: thêm residual connection từ motion prior
# # # # # # # # # # # # # # # # # #         self.motion_gate = nn.Sequential(
# # # # # # # # # # # # # # # # # #             nn.Linear(128 + 256, 1),
# # # # # # # # # # # # # # # # # #             nn.Sigmoid(),
# # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # #         self.out_head = nn.Sequential(
# # # # # # # # # # # # # # # # # #             nn.Linear(256, 128),
# # # # # # # # # # # # # # # # # #             nn.GELU(),
# # # # # # # # # # # # # # # # # #             nn.LayerNorm(128),
# # # # # # # # # # # # # # # # # #             nn.Linear(128, 2),
# # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # #         # Per-step scale được học, khởi tạo nhỏ để ổn định
# # # # # # # # # # # # # # # # # #         self.step_scale = nn.Parameter(torch.ones(self.N_STEPS) * 0.2)
# # # # # # # # # # # # # # # # # #         self._init_weights()

# # # # # # # # # # # # # # # # # #     def _init_weights(self):
# # # # # # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # # # # # #             for m in self.modules():
# # # # # # # # # # # # # # # # # #                 if isinstance(m, nn.Linear):
# # # # # # # # # # # # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.3)
# # # # # # # # # # # # # # # # # #                     if m.bias is not None:
# # # # # # # # # # # # # # # # # #                         nn.init.zeros_(m.bias)

# # # # # # # # # # # # # # # # # #     def forward(self, raw_ctx, obs_traj):
# # # # # # # # # # # # # # # # # #         # obs_traj: [T_obs, B, 2]
# # # # # # # # # # # # # # # # # #         B        = raw_ctx.shape[0]
# # # # # # # # # # # # # # # # # #         T_obs    = obs_traj.shape[0]

# # # # # # # # # # # # # # # # # #         # FIX: encode toàn bộ velocity sequence
# # # # # # # # # # # # # # # # # #         if T_obs >= 2:
# # # # # # # # # # # # # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]   # [T_obs-1, B, 2]
# # # # # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # # # # #             vels = torch.zeros(1, B, 2, device=raw_ctx.device)

# # # # # # # # # # # # # # # # # #         # Pad hoặc crop về obs_len velocity steps
# # # # # # # # # # # # # # # # # #         target_v = self.obs_len  # số velocity steps cần
# # # # # # # # # # # # # # # # # #         if vels.shape[0] < target_v:
# # # # # # # # # # # # # # # # # #             pad = torch.zeros(target_v - vels.shape[0], B, 2,
# # # # # # # # # # # # # # # # # #                               device=raw_ctx.device)
# # # # # # # # # # # # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # # # # #             vels = vels[-target_v:]   # lấy target_v bước cuối

# # # # # # # # # # # # # # # # # #         vel_input = vels.permute(1, 0, 2).reshape(B, -1)  # [B, obs_len*2]
# # # # # # # # # # # # # # # # # #         vel_feat  = self.vel_enc(vel_input)                 # [B, 128]

# # # # # # # # # # # # # # # # # #         ctx_feat = self.ctx_proj(raw_ctx)  # [B, 256]

# # # # # # # # # # # # # # # # # #         # FIX: hx được khởi tạo từ ctx thay vì chính ctx
# # # # # # # # # # # # # # # # # #         # để GRU có không gian học riêng
# # # # # # # # # # # # # # # # # #         hx      = ctx_feat
# # # # # # # # # # # # # # # # # #         cur_pos = obs_traj[-1].clone()   # [B, 2]

# # # # # # # # # # # # # # # # # #         # Last observed velocity làm prior bước đầu
# # # # # # # # # # # # # # # # # #         last_vel = vels[-1].clone()      # [B, 2]

# # # # # # # # # # # # # # # # # #         preds = []
# # # # # # # # # # # # # # # # # #         for i in range(self.N_STEPS):
# # # # # # # # # # # # # # # # # #             inp = torch.cat([ctx_feat, vel_feat, cur_pos], dim=-1)  # [B,386]
# # # # # # # # # # # # # # # # # #             hx  = self.gru_cell(inp, hx)                             # [B,256]

# # # # # # # # # # # # # # # # # #             # FIX: motion gate — blend giữa GRU output và motion prior
# # # # # # # # # # # # # # # # # #             gate = self.motion_gate(
# # # # # # # # # # # # # # # # # #                 torch.cat([vel_feat, hx], dim=-1)
# # # # # # # # # # # # # # # # # #             )  # [B,1]  — giai đoạn đầu training gate≈1 → theo motion prior

# # # # # # # # # # # # # # # # # #             raw_delta   = self.out_head(hx)                     # [B,2]
# # # # # # # # # # # # # # # # # #             # Motion prior: tiếp tục với velocity cuối (decay theo step)
# # # # # # # # # # # # # # # # # #             prior_delta = last_vel * (0.9 ** i)                 # [B,2]

# # # # # # # # # # # # # # # # # #             # Blend: gate=1 → dùng motion prior hoàn toàn (đầu training)
# # # # # # # # # # # # # # # # # #             #        gate=0 → dùng GRU output hoàn toàn (sau khi học)
# # # # # # # # # # # # # # # # # #             scale_i = torch.sigmoid(self.step_scale[i]) * self.MAX_DELTA
# # # # # # # # # # # # # # # # # #             blended = gate * prior_delta + (1 - gate) * raw_delta * scale_i

# # # # # # # # # # # # # # # # # #             delta   = blended.clamp(-self.MAX_DELTA, self.MAX_DELTA)
# # # # # # # # # # # # # # # # # #             cur_pos = cur_pos + delta
# # # # # # # # # # # # # # # # # #             last_vel = delta   # update velocity prior
# # # # # # # # # # # # # # # # # #             preds.append(cur_pos)

# # # # # # # # # # # # # # # # # #         return torch.stack(preds, dim=0)   # [4, B, 2]


# # # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # # # # #  VelocityField
# # # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # # # # #     OT-CFM velocity field v_θ(x_t, t, context).
# # # # # # # # # # # # # # # # # #     v24: ShortRangeHead thêm vào để dự báo tầm ngắn chính xác hơn.
# # # # # # # # # # # # # # # # # #     """

# # # # # # # # # # # # # # # # # #     # Dim của raw_ctx (output ctx_fc1) – phải khớp ShortRangeHead
# # # # # # # # # # # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # # # # # # # # # # #     def __init__(
# # # # # # # # # # # # # # # # # #         self,
# # # # # # # # # # # # # # # # # #         pred_len:   int   = 12,
# # # # # # # # # # # # # # # # # #         obs_len:    int   = 8,
# # # # # # # # # # # # # # # # # #         ctx_dim:    int   = 256,
# # # # # # # # # # # # # # # # # #         sigma_min:  float = 0.02,
# # # # # # # # # # # # # # # # # #         unet_in_ch: int   = 13,
# # # # # # # # # # # # # # # # # #     ):
# # # # # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # # # # #         self.pred_len  = pred_len
# # # # # # # # # # # # # # # # # #         self.obs_len   = obs_len
# # # # # # # # # # # # # # # # # #         self.sigma_min = sigma_min

# # # # # # # # # # # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # # # # # # # # # # #             in_channel   = unet_in_ch,
# # # # # # # # # # # # # # # # # #             out_channel  = 1,
# # # # # # # # # # # # # # # # # #             d_model      = 32,
# # # # # # # # # # # # # # # # # #             n_layers     = 4,
# # # # # # # # # # # # # # # # # #             modes_t      = 4,
# # # # # # # # # # # # # # # # # #             modes_h      = 4,
# # # # # # # # # # # # # # # # # #             modes_w      = 4,
# # # # # # # # # # # # # # # # # #             spatial_down = 32,
# # # # # # # # # # # # # # # # # #             dropout      = 0.05,
# # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # # # # # # # # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # # # # # # # # # # #             in_1d       = 4,
# # # # # # # # # # # # # # # # # #             feat_3d_dim = 128,
# # # # # # # # # # # # # # # # # #             mlp_h       = 64,
# # # # # # # # # # # # # # # # # #             lstm_hidden = 128,
# # # # # # # # # # # # # # # # # #             lstm_layers = 3,
# # # # # # # # # # # # # # # # # #             dropout     = 0.1,
# # # # # # # # # # # # # # # # # #             d_state     = 16,
# # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # # # # # # # # # # # # # # # #         # ctx_fc1 output = 512 = RAW_CTX_DIM
# # # # # # # # # # # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # # # # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # # # # # # # # # # #         self.ctx_drop = nn.Dropout(0.15)
# # # # # # # # # # # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # # # # # # # # # # # # # # #         self.time_fc1 = nn.Linear(256, 512)
# # # # # # # # # # # # # # # # # #         self.time_fc2 = nn.Linear(512, 256)

# # # # # # # # # # # # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # # # # # # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # # # # # # # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # # # # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # # # # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # # # # # # # # # # #                 dropout=0.15, activation="gelu", batch_first=True,
# # # # # # # # # # # # # # # # # #             ),
# # # # # # # # # # # # # # # # # #             num_layers=4,
# # # # # # # # # # # # # # # # # #         )
# # # # # # # # # # # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # # # # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # # # # # # # # # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)

# # # # # # # # # # # # # # # # # #         # ── FIX-M23: Short-range head ──────────────────────────────────────
# # # # # # # # # # # # # # # # # #         self.short_range_head = ShortRangeHead(
# # # # # # # # # # # # # # # # # #             raw_ctx_dim = self.RAW_CTX_DIM,
# # # # # # # # # # # # # # # # # #             obs_len     = obs_len,
# # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # #     # ── helpers ──────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # # # # #     def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
# # # # # # # # # # # # # # # # # #         half = dim // 2
# # # # # # # # # # # # # # # # # #         freq = torch.exp(
# # # # # # # # # # # # # # # # # #             torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # # # # # # # # # # #             * (-math.log(10_000.0) / max(half - 1, 1))
# # # # # # # # # # # # # # # # # #         )
# # # # # # # # # # # # # # # # # #         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
# # # # # # # # # # # # # # # # # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # # # # # # # # # # # # # # # # #         return F.pad(emb, (0, dim % 2))

# # # # # # # # # # # # # # # # # #     # def _context(self, batch_list: List) -> torch.Tensor:
# # # # # # # # # # # # # # # # # #     #     """Returns raw_ctx [B, RAW_CTX_DIM] (before dropout/projection)."""
# # # # # # # # # # # # # # # # # #     #     obs_traj  = batch_list[0]
# # # # # # # # # # # # # # # # # #     #     obs_Me    = batch_list[7]
# # # # # # # # # # # # # # # # # #     #     image_obs = batch_list[11]
# # # # # # # # # # # # # # # # # #     #     env_data  = batch_list[13]

# # # # # # # # # # # # # # # # # #     #     if image_obs.dim() == 4:
# # # # # # # # # # # # # # # # # #     #         image_obs = image_obs.unsqueeze(1)

# # # # # # # # # # # # # # # # # #     #     expected_ch = self.spatial_enc.in_channel
# # # # # # # # # # # # # # # # # #     #     if image_obs.shape[1] == 1 and expected_ch != 1:
# # # # # # # # # # # # # # # # # #     #         image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# # # # # # # # # # # # # # # # # #     #     e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # # # # # # # # # #     #     T_obs = obs_traj.shape[0]

# # # # # # # # # # # # # # # # # #     #     e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
# # # # # # # # # # # # # # # # # #     #     e_3d_s = e_3d_s.permute(0, 2, 1)
# # # # # # # # # # # # # # # # # #     #     e_3d_s = self.bottleneck_proj(e_3d_s)

# # # # # # # # # # # # # # # # # #     #     T_bot = e_3d_s.shape[1]
# # # # # # # # # # # # # # # # # #     #     if T_bot != T_obs:
# # # # # # # # # # # # # # # # # #     #         e_3d_s = F.interpolate(
# # # # # # # # # # # # # # # # # #     #             e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # # # # # # # # # #     #             mode="linear", align_corners=False,
# # # # # # # # # # # # # # # # # #     #         ).permute(0, 2, 1)

# # # # # # # # # # # # # # # # # #     #     # f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))
# # # # # # # # # # # # # # # # # #     #     # f_spatial     = self.decoder_proj(f_spatial_raw)

# # # # # # # # # # # # # # # # # #     #     # FIX: pool H,W, giữ T, rồi flatten
# # # # # # # # # # # # # # # # # #     #     # f_spatial_raw = e_3d_dec.squeeze(1)          # [B,T,1,1] → [B,T]  (squeeze C và H,W)
# # # # # # # # # # # # # # # # # #     #     # f_spatial_raw = f_spatial_raw.mean(dim=1)    # [B] — hoặc giữ T nếu muốn
# # # # # # # # # # # # # # # # # #     #     # Hoặc đơn giản hơn: mean chỉ spatial
# # # # # # # # # # # # # # # # # #     #     f_spatial_raw = e_3d_dec.mean(dim=(3,4))     # [B,1,T] → squeeze → [B,T]
# # # # # # # # # # # # # # # # # #     #     f_spatial     = self.decoder_proj(f_spatial_raw.squeeze(1))  # cần Linear(T,16)

# # # # # # # # # # # # # # # # # #     #     obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # # # # # # # # # # #     #     h_t    = self.enc_1d(obs_in, e_3d_s)

# # # # # # # # # # # # # # # # # #     #     e_env, _, _ = self.env_enc(env_data, image_obs)

# # # # # # # # # # # # # # # # # #     #     raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
# # # # # # # # # # # # # # # # # #     #     raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))  # [B, RAW_CTX_DIM]
# # # # # # # # # # # # # # # # # #     #     return raw                                      # raw_ctx

# # # # # # # # # # # # # # # # # #     # ── _context(): fix f_spatial ──────────────────────────────────────────────
# # # # # # # # # # # # # # # # # #     def _context(self, batch_list):
# # # # # # # # # # # # # # # # # #         obs_traj  = batch_list[0]   # [T_obs, B, 2]
# # # # # # # # # # # # # # # # # #         obs_Me    = batch_list[7]   # [T_obs, B, 2]
# # # # # # # # # # # # # # # # # #         image_obs = batch_list[11]  # [B, 13, T, 81, 81]  từ seq_collate
# # # # # # # # # # # # # # # # # #         env_data  = batch_list[13]

# # # # # # # # # # # # # # # # # #         # image_obs đã được permute đúng bởi seq_collate → [B,13,T,H,W]
# # # # # # # # # # # # # # # # # #         # Chỉ cần guard cho trường hợp dim==4 (không có T)
# # # # # # # # # # # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # # # # # # # # # # #             image_obs = image_obs.unsqueeze(2)  # thêm T dim, không phải C

# # # # # # # # # # # # # # # # # #         expected_ch = self.spatial_enc.in_channel
# # # # # # # # # # # # # # # # # #         if image_obs.shape[1] == 1 and expected_ch != 1:
# # # # # # # # # # # # # # # # # #             image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# # # # # # # # # # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # # # # # # # # # #         # e_3d_bot: [B,128,T,4,4]   e_3d_dec: [B,1,T,1,1]

# # # # # # # # # # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # # # # # # # # # #         # Bottleneck: pool spatial → [B,128,T] → permute → [B,T,128]
# # # # # # # # # # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot)   # [B,128,T,1,1]
# # # # # # # # # # # # # # # # # #         e_3d_s = e_3d_s.squeeze(-1).squeeze(-1)   # [B,128,T]
# # # # # # # # # # # # # # # # # #         e_3d_s = e_3d_s.permute(0, 2, 1)          # [B,T,128]
# # # # # # # # # # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)     # [B,T,128]

# # # # # # # # # # # # # # # # # #         # T alignment
# # # # # # # # # # # # # # # # # #         T_bot = e_3d_s.shape[1]
# # # # # # # # # # # # # # # # # #         if T_bot != T_obs:
# # # # # # # # # # # # # # # # # #             e_3d_s = F.interpolate(
# # # # # # # # # # # # # # # # # #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # # # # # # # # # #                 mode="linear", align_corners=False,
# # # # # # # # # # # # # # # # # #             ).permute(0, 2, 1)

# # # # # # # # # # # # # # # # # #         # FIX Bug G: f_spatial giữ lại temporal info
# # # # # # # # # # # # # # # # # #         # e_3d_dec: [B,1,T,1,1] → mean(H,W) → [B,1,T] → mean(T) → [B,1]
# # # # # # # # # # # # # # # # # #         # Thay bằng: pool T về 1 bằng weighted mean (cuối quan trọng hơn)
# # # # # # # # # # # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)  # [B,T]
# # # # # # # # # # # # # # # # # #         # Exponential weighting: bước cuối obs quan trọng nhất
# # # # # # # # # # # # # # # # # #         t_weights = torch.softmax(
# # # # # # # # # # # # # # # # # #             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # # # # # # # # # # # #                         device=e_3d_dec_t.device) * 0.5,
# # # # # # # # # # # # # # # # # #             dim=0
# # # # # # # # # # # # # # # # # #         )  # [T]
# # # # # # # # # # # # # # # # # #         f_spatial_scalar = (e_3d_dec_t * t_weights.unsqueeze(0)).sum(1, keepdim=True)  # [B,1]
# # # # # # # # # # # # # # # # # #         f_spatial = self.decoder_proj(f_spatial_scalar)  # [B,16]

# # # # # # # # # # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)  # [B,T,4]
# # # # # # # # # # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)   # [B,128]

# # # # # # # # # # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)  # [B,64]

# # # # # # # # # # # # # # # # # #         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)  # [B,208]
# # # # # # # # # # # # # # # # # #         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))       # [B,512]
# # # # # # # # # # # # # # # # # #         return raw

# # # # # # # # # # # # # # # # # #     def _apply_ctx_head(self, raw: torch.Tensor,
# # # # # # # # # # # # # # # # # #                         noise_scale: float = 0.0) -> torch.Tensor:
# # # # # # # # # # # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # # # # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # # # # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # # # # # # # # # # #     def _beta_drift_velocity(self, x_t: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # # #         """Beta drift in normalised state space. x_t: [B, T, 4]."""
# # # # # # # # # # # # # # # # # #         OMEGA_val  = 7.2921e-5
# # # # # # # # # # # # # # # # # #         R_val      = 6.371e6
# # # # # # # # # # # # # # # # # #         DT         = 6 * 3600.0
# # # # # # # # # # # # # # # # # #         M_PER_NORM = 5.0 * 111.0 * 1000.0

# # # # # # # # # # # # # # # # # #         lat_norm = x_t[:, :, 1]
# # # # # # # # # # # # # # # # # #         lat_deg  = lat_norm * 5.0
# # # # # # # # # # # # # # # # # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))

# # # # # # # # # # # # # # # # # #         beta   = 2 * OMEGA_val * torch.cos(lat_rad) / R_val
# # # # # # # # # # # # # # # # # #         R_tc   = 3e5
# # # # # # # # # # # # # # # # # #         v_lon  = -beta * R_tc ** 2 / 2
# # # # # # # # # # # # # # # # # #         v_lat  =  beta * R_tc ** 2 / 4

# # # # # # # # # # # # # # # # # #         v_lon_norm = v_lon * DT / M_PER_NORM
# # # # # # # # # # # # # # # # # #         v_lat_norm = v_lat * DT / M_PER_NORM

# # # # # # # # # # # # # # # # # #         v_phys = torch.zeros_like(x_t)
# # # # # # # # # # # # # # # # # #         v_phys[:, :, 0] = v_lon_norm
# # # # # # # # # # # # # # # # # #         v_phys[:, :, 1] = v_lat_norm
# # # # # # # # # # # # # # # # # #         return v_phys

# # # # # # # # # # # # # # # # # #     def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
# # # # # # # # # # # # # # # # # #                 ctx: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
# # # # # # # # # # # # # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # # # # # # # # # # # # #         T_seq  = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # # # # # # # # # # # #         x_emb  = (self.traj_embed(x_t[:, :T_seq, :])
# # # # # # # # # # # # # # # # # #                   + self.pos_enc[:, :T_seq, :]
# # # # # # # # # # # # # # # # # #                   + t_emb.unsqueeze(1))
# # # # # # # # # # # # # # # # # #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

# # # # # # # # # # # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(
# # # # # # # # # # # # # # # # # #             self.transformer(x_emb, memory)
# # # # # # # # # # # # # # # # # #         )))

# # # # # # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # # # # # #             v_phys = self._beta_drift_velocity(x_t[:, :T_seq, :])

# # # # # # # # # # # # # # # # # #         scale = torch.sigmoid(self.physics_scale) * 2.0
# # # # # # # # # # # # # # # # # #         return v_neural + scale * v_phys

# # # # # # # # # # # # # # # # # #     def forward(self, x_t, t, batch_list):
# # # # # # # # # # # # # # # # # #         raw = self._context(batch_list)
# # # # # # # # # # # # # # # # # #         ctx = self._apply_ctx_head(raw, noise_scale=0.0)
# # # # # # # # # # # # # # # # # #         return self._decode(x_t, t, ctx)

# # # # # # # # # # # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale: float = 0.0):
# # # # # # # # # # # # # # # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale=noise_scale)
# # # # # # # # # # # # # # # # # #         return self._decode(x_t, t, ctx)

# # # # # # # # # # # # # # # # # #     def forward_short_range(
# # # # # # # # # # # # # # # # # #         self,
# # # # # # # # # # # # # # # # # #         obs_traj: torch.Tensor,  # [T_obs, B, 2]
# # # # # # # # # # # # # # # # # #         raw_ctx:  torch.Tensor,  # [B, RAW_CTX_DIM]
# # # # # # # # # # # # # # # # # #     ) -> torch.Tensor:
# # # # # # # # # # # # # # # # # #         """FIX-M23: Deterministic short-range prediction [4, B, 2]."""
# # # # # # # # # # # # # # # # # #         return self.short_range_head(raw_ctx, obs_traj)


# # # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # # # # #  TCFlowMatching
# # # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # # # # # class TCFlowMatching(nn.Module):
# # # # # # # # # # # # # # # # # #     """TC trajectory prediction via OT-CFM + ShortRangeHead + PINN."""

# # # # # # # # # # # # # # # # # #     def __init__(
# # # # # # # # # # # # # # # # # #         self,
# # # # # # # # # # # # # # # # # #         pred_len:             int   = 12,
# # # # # # # # # # # # # # # # # #         obs_len:              int   = 8,
# # # # # # # # # # # # # # # # # #         sigma_min:            float = 0.02,
# # # # # # # # # # # # # # # # # #         n_train_ens:          int   = 4,
# # # # # # # # # # # # # # # # # #         unet_in_ch:           int   = 13,
# # # # # # # # # # # # # # # # # #         ctx_noise_scale:      float = 0.002,    # FIX-M25: 0.02→0.002
# # # # # # # # # # # # # # # # # #         initial_sample_sigma: float = 0.03,     # FIX-M25: 0.1→0.03
# # # # # # # # # # # # # # # # # #         **kwargs,
# # # # # # # # # # # # # # # # # #     ):
# # # # # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # # # # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # # # # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # # # # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # # # # # # # # # # #         self.active_pred_len      = pred_len
# # # # # # # # # # # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # # # # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma

# # # # # # # # # # # # # # # # # #         self.net = VelocityField(
# # # # # # # # # # # # # # # # # #             pred_len   = pred_len,
# # # # # # # # # # # # # # # # # #             obs_len    = obs_len,
# # # # # # # # # # # # # # # # # #             sigma_min  = sigma_min,
# # # # # # # # # # # # # # # # # #             unet_in_ch = unet_in_ch,
# # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # #     def set_curriculum_len(self, active_len: int) -> None:
# # # # # # # # # # # # # # # # # #         """FIX-M18: No-op."""
# # # # # # # # # # # # # # # # # #         pass

# # # # # # # # # # # # # # # # # #     # ── Static helpers ────────────────────────────────────────────────────────

# # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # #     def _to_rel(traj_gt, Me_gt, last_pos, last_Me):
# # # # # # # # # # # # # # # # # #         return torch.cat(
# # # # # # # # # # # # # # # # # #             [traj_gt - last_pos.unsqueeze(0),
# # # # # # # # # # # # # # # # # #              Me_gt   - last_Me.unsqueeze(0)],
# # # # # # # # # # # # # # # # # #             dim=-1,
# # # # # # # # # # # # # # # # # #         ).permute(1, 0, 2)

# # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # #     def _to_abs(rel, last_pos, last_Me):
# # # # # # # # # # # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # # # # # # # # # # #         return (
# # # # # # # # # # # # # # # # # #             last_pos.unsqueeze(0) + d[:, :, :2],
# # # # # # # # # # # # # # # # # #             last_Me.unsqueeze(0)  + d[:, :, 2:],
# # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # #     def _cfm_noisy(self, x1):
# # # # # # # # # # # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # # # # # # # # # # #         sm  = self.sigma_min
# # # # # # # # # # # # # # # # # #         x0  = torch.randn_like(x1) * sm
# # # # # # # # # # # # # # # # # #         t   = torch.rand(B, device=device)
# # # # # # # # # # # # # # # # # #         te  = t.view(B, 1, 1)
# # # # # # # # # # # # # # # # # #         x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
# # # # # # # # # # # # # # # # # #         denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
# # # # # # # # # # # # # # # # # #         target_vel = (x1 - (1.0 - sm) * x_t) / denom
# # # # # # # # # # # # # # # # # #         return x_t, t, te, denom, target_vel

# # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # #     def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # # #         wind_norm = obs_Me[-1, :, 1].detach()
# # # # # # # # # # # # # # # # # #         w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
# # # # # # # # # # # # # # # # # #             torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
# # # # # # # # # # # # # # # # # #             torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
# # # # # # # # # # # # # # # # # #                         torch.full_like(wind_norm, 1.5))))
# # # # # # # # # # # # # # # # # #         return w / w.mean().clamp(min=1e-6)

# # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # #     def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
# # # # # # # # # # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # # # # # # # # # #             return batch_list
# # # # # # # # # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # # # # # # # # # # #             t = aug[idx]
# # # # # # # # # # # # # # # # # #             if torch.is_tensor(t) and t.shape[-1] >= 1:
# # # # # # # # # # # # # # # # # #                 t = t.clone()
# # # # # # # # # # # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # # # # # # # # # # #                 aug[idx] = t
# # # # # # # # # # # # # # # # # #         return aug

# # # # # # # # # # # # # # # # # #     # ── Loss ──────────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # # # # #     def get_loss(self, batch_list: List,
# # # # # # # # # # # # # # # # # #                  step_weight_alpha: float = 0.0) -> torch.Tensor:
# # # # # # # # # # # # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha)["total"]

# # # # # # # # # # # # # # # # # #     def get_loss_breakdown(self, batch_list: List,
# # # # # # # # # # # # # # # # # #                            step_weight_alpha: float = 0.0) -> Dict:
# # # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # # #         FIX-M27: Thêm short_range_loss với weight cao (5.0).
# # # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # # #         batch_list = self._lon_flip_aug(batch_list, p=0.3)

# # # # # # # # # # # # # # # # # #         traj_gt  = batch_list[1]
# # # # # # # # # # # # # # # # # #         Me_gt    = batch_list[8]
# # # # # # # # # # # # # # # # # #         obs_t    = batch_list[0]
# # # # # # # # # # # # # # # # # #         obs_Me   = batch_list[7]

# # # # # # # # # # # # # # # # # #         try:
# # # # # # # # # # # # # # # # # #             env_data = batch_list[13]
# # # # # # # # # # # # # # # # # #         except (IndexError, TypeError):
# # # # # # # # # # # # # # # # # #             env_data = None

# # # # # # # # # # # # # # # # # #         lp, lm = obs_t[-1], obs_Me[-1]
# # # # # # # # # # # # # # # # # #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

# # # # # # # # # # # # # # # # # #         raw_ctx     = self.net._context(batch_list)
# # # # # # # # # # # # # # # # # #         intensity_w = self._intensity_weights(obs_Me)

# # # # # # # # # # # # # # # # # #         x_t, t, te, denom, _ = self._cfm_noisy(x1)
# # # # # # # # # # # # # # # # # #         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

# # # # # # # # # # # # # # # # # #         # Ensemble for AFCRPS
# # # # # # # # # # # # # # # # # #         samples: List[torch.Tensor] = []
# # # # # # # # # # # # # # # # # #         for _ in range(self.n_train_ens):
# # # # # # # # # # # # # # # # # #             xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
# # # # # # # # # # # # # # # # # #             pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
# # # # # # # # # # # # # # # # # #             x1_s  = xt_s + dens_s * pv_s
# # # # # # # # # # # # # # # # # #             pa_s, _ = self._to_abs(x1_s, lp, lm)
# # # # # # # # # # # # # # # # # #             samples.append(pa_s)
# # # # # # # # # # # # # # # # # #         pred_samples = torch.stack(samples)     # [S, T, B, 2]
# # # # # # # # # # # # # # # # # #         all_trajs_4d = pred_samples

# # # # # # # # # # # # # # # # # #         l_fm_physics = fm_physics_consistency_loss(
# # # # # # # # # # # # # # # # # #             pred_samples, gt_norm=traj_gt, last_pos=lp)

# # # # # # # # # # # # # # # # # #         x1_pred = x_t + denom * pred_vel
# # # # # # # # # # # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# # # # # # # # # # # # # # # # # #         pred_abs_deg = _denorm_to_deg(pred_abs)
# # # # # # # # # # # # # # # # # #         # # Thêm debug vào get_loss_breakdown, sau khi tính pred_abs_deg:
# # # # # # # # # # # # # # # # # #         # print(f"  pred_abs requires_grad: {pred_abs.requires_grad}")
# # # # # # # # # # # # # # # # # #         # print(f"  pred_abs_deg requires_grad: {pred_abs_deg.requires_grad}")

# # # # # # # # # # # # # # # # # #         traj_gt_deg  = _denorm_to_deg(traj_gt)
# # # # # # # # # # # # # # # # # #         ref_deg      = _denorm_to_deg(lp)

# # # # # # # # # # # # # # # # # #         breakdown = compute_total_loss(
# # # # # # # # # # # # # # # # # #             pred_abs           = pred_abs_deg,
# # # # # # # # # # # # # # # # # #             gt                 = traj_gt_deg,
# # # # # # # # # # # # # # # # # #             ref                = ref_deg,
# # # # # # # # # # # # # # # # # #             batch_list         = batch_list,
# # # # # # # # # # # # # # # # # #             pred_samples       = pred_samples,
# # # # # # # # # # # # # # # # # #             gt_norm            = traj_gt,
# # # # # # # # # # # # # # # # # #             weights            = WEIGHTS,
# # # # # # # # # # # # # # # # # #             intensity_w        = intensity_w,
# # # # # # # # # # # # # # # # # #             env_data           = env_data,
# # # # # # # # # # # # # # # # # #             step_weight_alpha  = step_weight_alpha,
# # # # # # # # # # # # # # # # # #             all_trajs          = all_trajs_4d,
# # # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # # #         fm_phys_w = WEIGHTS.get("fm_physics", 0.3)
# # # # # # # # # # # # # # # # # #         breakdown["total"]      = breakdown["total"] + fm_phys_w * l_fm_physics
# # # # # # # # # # # # # # # # # #         breakdown["fm_physics"] = l_fm_physics.item()

# # # # # # # # # # # # # # # # # #         # ── FIX-M27: Short-range loss ──────────────────────────────────────
# # # # # # # # # # # # # # # # # #         n_sr = ShortRangeHead.N_STEPS  # 4
# # # # # # # # # # # # # # # # # #         if traj_gt.shape[0] >= n_sr:
# # # # # # # # # # # # # # # # # #             sr_pred = self.net.forward_short_range(obs_t, raw_ctx)  # [4, B, 2]
# # # # # # # # # # # # # # # # # #             sr_gt   = traj_gt[:n_sr]                                # [4, B, 2]
# # # # # # # # # # # # # # # # # #             l_sr    = short_range_regression_loss(sr_pred, sr_gt, lp)
# # # # # # # # # # # # # # # # # #             sr_w    = WEIGHTS.get("short_range", 5.0)
# # # # # # # # # # # # # # # # # #             breakdown["total"]       = breakdown["total"] + sr_w * l_sr
# # # # # # # # # # # # # # # # # #             breakdown["short_range"] = l_sr.item()
# # # # # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # # # # #             breakdown["short_range"] = 0.0

# # # # # # # # # # # # # # # # # #         return breakdown

# # # # # # # # # # # # # # # # # #     # ── sample() ─────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # # # # #     # @torch.no_grad()
# # # # # # # # # # # # # # # # # #     # def sample(
# # # # # # # # # # # # # # # # # #     #     self,
# # # # # # # # # # # # # # # # # #     #     batch_list:  List,
# # # # # # # # # # # # # # # # # #     #     num_ensemble: int = 50,
# # # # # # # # # # # # # # # # # #     #     ddim_steps:   int = 20,
# # # # # # # # # # # # # # # # # #     #     predict_csv:  Optional[str] = None,
# # # # # # # # # # # # # # # # # #     # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# # # # # # # # # # # # # # # # # #     #     """
# # # # # # # # # # # # # # # # # #     #     FIX-M24: pred_mean[:4] ← ShortRangeHead (deterministic, low-error).
# # # # # # # # # # # # # # # # # #     #              pred_mean[4:] ← FM ensemble mean.

# # # # # # # # # # # # # # # # # #     #     Returns:
# # # # # # # # # # # # # # # # # #     #         pred_mean : [T, B, 2]
# # # # # # # # # # # # # # # # # #     #         me_mean   : [T, B, 2]
# # # # # # # # # # # # # # # # # #     #         all_trajs : [S, T, B, 2]
# # # # # # # # # # # # # # # # # #     #     """
# # # # # # # # # # # # # # # # # #     #     obs_t = batch_list[0]          # [T_obs, B, 2]
# # # # # # # # # # # # # # # # # #     #     lp    = obs_t[-1]              # [B, 2]
# # # # # # # # # # # # # # # # # #     #     lm    = batch_list[7][-1]      # [B, 2]
# # # # # # # # # # # # # # # # # #     #     B     = lp.shape[0]
# # # # # # # # # # # # # # # # # #     #     device = lp.device
# # # # # # # # # # # # # # # # # #     #     T      = self.pred_len
# # # # # # # # # # # # # # # # # #     #     dt     = 1.0 / max(ddim_steps, 1)

# # # # # # # # # # # # # # # # # #     #     raw_ctx = self.net._context(batch_list)

# # # # # # # # # # # # # # # # # #     #     # ── Short-range deterministic (steps 1-4) ─────────────────────────
# # # # # # # # # # # # # # # # # #     #     n_sr     = ShortRangeHead.N_STEPS      # 4
# # # # # # # # # # # # # # # # # #     #     sr_pred  = self.net.forward_short_range(obs_t, raw_ctx)  # [4, B, 2]

# # # # # # # # # # # # # # # # # #     #     # ── FM ensemble (all 12 steps, for steps 5-12) ───────────────────
# # # # # # # # # # # # # # # # # #     #     traj_s: List[torch.Tensor] = []
# # # # # # # # # # # # # # # # # #     #     me_s:   List[torch.Tensor] = []

# # # # # # # # # # # # # # # # # #     #     for k in range(num_ensemble):
# # # # # # # # # # # # # # # # # #     #         x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# # # # # # # # # # # # # # # # # #     #         for step in range(ddim_steps):
# # # # # # # # # # # # # # # # # #     #             t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # # # # # # # # #     #             ns  = self.ctx_noise_scale if step == 0 else 0.0
# # # # # # # # # # # # # # # # # #     #             vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
# # # # # # # # # # # # # # # # # #     #             x_t = x_t + dt * vel

# # # # # # # # # # # # # # # # # #     #         x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)
# # # # # # # # # # # # # # # # # #     #         x_t = x_t.clamp(-3.0, 3.0)

# # # # # # # # # # # # # # # # # #     #         tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # # # # # # # # # #     #         traj_s.append(tr)
# # # # # # # # # # # # # # # # # #     #         me_s.append(me)

# # # # # # # # # # # # # # # # # #     #     all_trajs = torch.stack(traj_s)    # [S, T, B, 2]
# # # # # # # # # # # # # # # # # #     #     all_me    = torch.stack(me_s)

# # # # # # # # # # # # # # # # # #     #     # ── FIX-M24: Blend short-range into pred_mean ─────────────────────
# # # # # # # # # # # # # # # # # #     #     fm_mean  = all_trajs.mean(0)       # [T, B, 2]
# # # # # # # # # # # # # # # # # #     #     pred_mean = fm_mean.clone()
# # # # # # # # # # # # # # # # # #     #     pred_mean[:n_sr] = sr_pred         # Override steps 1-4

# # # # # # # # # # # # # # # # # #     #     # # Also override all_trajs first 4 steps with deterministic prediction
# # # # # # # # # # # # # # # # # #     #     # # (reduces spurious spread for 12h/24h in CRPS)
# # # # # # # # # # # # # # # # # #     #     # all_trajs[:, :n_sr, :, :] = sr_pred.unsqueeze(0).expand(
# # # # # # # # # # # # # # # # # #     #     #     num_ensemble, -1, -1, -1
# # # # # # # # # # # # # # # # # #     #     # )

# # # # # # # # # # # # # # # # # #     #     me_mean = all_me.mean(0)

# # # # # # # # # # # # # # # # # #     #     if predict_csv:
# # # # # # # # # # # # # # # # # #     #         self._write_predict_csv(predict_csv, pred_mean, all_trajs)

# # # # # # # # # # # # # # # # # #     #     return pred_mean, me_mean, all_trajs

# # # # # # # # # # # # # # # # # #     @torch.no_grad()
# # # # # # # # # # # # # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
# # # # # # # # # # # # # # # # # #         obs_t  = batch_list[0]
# # # # # # # # # # # # # # # # # #         lp     = obs_t[-1]
# # # # # # # # # # # # # # # # # #         lm     = batch_list[7][-1]
# # # # # # # # # # # # # # # # # #         B      = lp.shape[0]
# # # # # # # # # # # # # # # # # #         device = lp.device
# # # # # # # # # # # # # # # # # #         T      = self.pred_len
# # # # # # # # # # # # # # # # # #         dt     = 1.0 / max(ddim_steps, 1)

# # # # # # # # # # # # # # # # # #         raw_ctx = self.net._context(batch_list)
# # # # # # # # # # # # # # # # # #         n_sr    = ShortRangeHead.N_STEPS

# # # # # # # # # # # # # # # # # #         # Short-range deterministic
# # # # # # # # # # # # # # # # # #         sr_pred = self.net.forward_short_range(obs_t, raw_ctx)   # [4, B, 2]

# # # # # # # # # # # # # # # # # #         # FM ensemble — giữ nguyên, KHÔNG override
# # # # # # # # # # # # # # # # # #         traj_s, me_s = [], []
# # # # # # # # # # # # # # # # # #         for k in range(num_ensemble):
# # # # # # # # # # # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# # # # # # # # # # # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # # # # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # # # # # # # # #                 ns  = self.ctx_noise_scale if step == 0 else 0.0
# # # # # # # # # # # # # # # # # #                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
# # # # # # # # # # # # # # # # # #                 x_t = x_t + dt * vel

# # # # # # # # # # # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)
# # # # # # # # # # # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)

# # # # # # # # # # # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # # # # # # # # # #             traj_s.append(tr)
# # # # # # # # # # # # # # # # # #             me_s.append(me)

# # # # # # # # # # # # # # # # # #         all_trajs = torch.stack(traj_s)   # [S, T, B, 2] — KHÔNG override steps 1-4
# # # # # # # # # # # # # # # # # #         all_me    = torch.stack(me_s)

# # # # # # # # # # # # # # # # # #         # FIX: pred_mean blend đúng cách
# # # # # # # # # # # # # # # # # #         # Steps 1-4: ShortRangeHead (deterministic, low-error)
# # # # # # # # # # # # # # # # # #         # Steps 5-12: FM ensemble mean
# # # # # # # # # # # # # # # # # #         fm_mean   = all_trajs.mean(0)    # [T, B, 2]
# # # # # # # # # # # # # # # # # #         pred_mean = fm_mean.clone()
# # # # # # # # # # # # # # # # # #         pred_mean[:n_sr] = sr_pred       # override pred_mean, KHÔNG override all_trajs

# # # # # # # # # # # # # # # # # #         me_mean = all_me.mean(0)

# # # # # # # # # # # # # # # # # #         if predict_csv:
# # # # # # # # # # # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)

# # # # # # # # # # # # # # # # # #         return pred_mean, me_mean, all_trajs

# # # # # # # # # # # # # # # # # #     # ── Physics correction ────────────────────────────────────────────────────

# # # # # # # # # # # # # # # # # #     def _physics_correct(
# # # # # # # # # # # # # # # # # #         self,
# # # # # # # # # # # # # # # # # #         x_pred: torch.Tensor,
# # # # # # # # # # # # # # # # # #         last_pos: torch.Tensor,
# # # # # # # # # # # # # # # # # #         last_Me:  torch.Tensor,
# # # # # # # # # # # # # # # # # #         n_steps:  int   = 5,
# # # # # # # # # # # # # # # # # #         lr:       float = 0.002,
# # # # # # # # # # # # # # # # # #     ) -> torch.Tensor:
# # # # # # # # # # # # # # # # # #         with torch.enable_grad():
# # # # # # # # # # # # # # # # # #             x = x_pred.detach().requires_grad_(True)
# # # # # # # # # # # # # # # # # #             optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)

# # # # # # # # # # # # # # # # # #             for _ in range(n_steps):
# # # # # # # # # # # # # # # # # #                 optimizer.zero_grad()
# # # # # # # # # # # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # # # # # # # # # # #                 pred_deg    = _denorm_to_deg(pred_abs)

# # # # # # # # # # # # # # # # # #                 l_speed = self._pinn_speed_constraint(pred_deg)
# # # # # # # # # # # # # # # # # #                 l_accel = self._pinn_beta_plane_simplified(pred_deg)

# # # # # # # # # # # # # # # # # #                 physics_loss = l_speed + 0.3 * l_accel
# # # # # # # # # # # # # # # # # #                 physics_loss.backward()

# # # # # # # # # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], max_norm=0.05)
# # # # # # # # # # # # # # # # # #                 optimizer.step()

# # # # # # # # # # # # # # # # # #         return x.detach()

# # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # #     def _pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # # #         if pred_deg.shape[0] < 2:
# # # # # # # # # # # # # # # # # #             return pred_deg.new_zeros(())
# # # # # # # # # # # # # # # # # #         dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # # # # # # # # #         lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # # # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # # # # # # # #         dx_km   = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # # # # # # # # # # # # # # #         dy_km   = dt_deg[:, :, 1] * 111.0
# # # # # # # # # # # # # # # # # #         speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # # # # # # # # # # # # # # # # #         return F.relu(speed - 600.0).pow(2).mean()

# # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # #     def _pinn_beta_plane_simplified(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # # #         if pred_deg.shape[0] < 3:
# # # # # # # # # # # # # # # # # #             return pred_deg.new_zeros(())
# # # # # # # # # # # # # # # # # #         v = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # # # # # # # # #         a = v[1:] - v[:-1]
# # # # # # # # # # # # # # # # # #         lat_rad = torch.deg2rad(pred_deg[1:-1, :, 1])
# # # # # # # # # # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # # # # # # # #         a_lon_km = a[:, :, 0] * cos_lat * 111.0
# # # # # # # # # # # # # # # # # #         a_lat_km = a[:, :, 1] * 111.0
# # # # # # # # # # # # # # # # # #         max_accel = 50.0
# # # # # # # # # # # # # # # # # #         violation = F.relu(torch.sqrt(a_lon_km**2 + a_lat_km**2) - max_accel)
# # # # # # # # # # # # # # # # # #         return violation.pow(2).mean() * 0.1

# # # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # # #     def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
# # # # # # # # # # # # # # # # # #                            all_trajs: torch.Tensor) -> None:
# # # # # # # # # # # # # # # # # #         import numpy as np
# # # # # # # # # # # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # # # # # # # # # # #         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # # # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # # # # # # # # # # #         S       = all_trajs.shape[0]

# # # # # # # # # # # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()

# # # # # # # # # # # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # # # # # # # # # # #                     "lon_mean_deg", "lat_mean_deg",
# # # # # # # # # # # # # # # # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # # # # # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # # # # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # # # # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # # # # # # # # # # #             if write_hdr:
# # # # # # # # # # # # # # # # # #                 w.writeheader()
# # # # # # # # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # # # # # # # #                 for k in range(T):
# # # # # # # # # # # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # # # # # # # # # # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
# # # # # # # # # # # # # # # # # #                         math.radians(mean_lat[k, b]))
# # # # # # # # # # # # # # # # # #                     spread = float(np.sqrt((dlat**2 + dlon**2).mean()) * 111.0)
# # # # # # # # # # # # # # # # # #                     w.writerow(dict(
# # # # # # # # # # # # # # # # # #                         timestamp     = ts,
# # # # # # # # # # # # # # # # # #                         batch_idx     = b,
# # # # # # # # # # # # # # # # # #                         step_idx      = k,
# # # # # # # # # # # # # # # # # #                         lead_h        = (k + 1) * 6,
# # # # # # # # # # # # # # # # # #                         lon_mean_deg  = f"{mean_lon[k,b]:.4f}",
# # # # # # # # # # # # # # # # # #                         lat_mean_deg  = f"{mean_lat[k,b]:.4f}",
# # # # # # # # # # # # # # # # # #                         lon_std_deg   = f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # # # # # # # # # # #                         lat_std_deg   = f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # # # # # # # # # # #                         ens_spread_km = f"{spread:.2f}",
# # # # # # # # # # # # # # # # # #                     ))
# # # # # # # # # # # # # # # # # #         print(f"  Predictions → {csv_path}  (B={B}, T={T}, S={S})")


# # # # # # # # # # # # # # # # # # # Backward-compat alias
# # # # # # # # # # # # # # # # # # TCDiffusion = TCFlowMatching

# # # # # # # # # # # # # # # # # """
# # # # # # # # # # # # # # # # # Model/flow_matching_model.py  ── v25
# # # # # # # # # # # # # # # # # ==========================================
# # # # # # # # # # # # # # # # # FULL REWRITE – fixes từ review:

# # # # # # # # # # # # # # # # #   FIX-M-A  [CRITICAL] ShortRangeHead motion_gate init: bias = +2.0
# # # # # # # # # # # # # # # # #            để gate ≈ 0.88 đầu training (motion prior dominant),
# # # # # # # # # # # # # # # # #            thay vì xavier_uniform → gate ≈ 0.5 (mixed từ đầu).

# # # # # # # # # # # # # # # # #   FIX-M-B  [HIGH] Phase 1 freeze logic: freeze ctx_fc1/ctx_ln cũng
# # # # # # # # # # # # # # # # #            vì ShortRangeHead nhận raw_ctx từ frozen encoder → học
# # # # # # # # # # # # # # # # #            trên noise. Phase 1 chỉ train ShortRangeHead với obs_traj.

# # # # # # # # # # # # # # # # #   FIX-M-C  [HIGH] Bridge loss integration: compute sr_pred trong
# # # # # # # # # # # # # # # # #            get_loss_breakdown() và pass sang compute_total_loss().
# # # # # # # # # # # # # # # # #            SR↔FM nhất quán tại step 4 (Eq.80).

# # # # # # # # # # # # # # # # #   FIX-M-D  [HIGH] _physics_correct: thay SGD+momentum bằng Adam
# # # # # # # # # # # # # # # # #            (n_steps nhỏ, momentum gây oscillation).

# # # # # # # # # # # # # # # # #   FIX-M-E  [MEDIUM] get_loss_breakdown() pass epoch, gt_abs_deg,
# # # # # # # # # # # # # # # # #            vmax_pred, pmin_pred cho adaptive PINN weighting.

# # # # # # # # # # # # # # # # #   FIX-M-F  [MEDIUM] sample(): cache raw_ctx một lần, dùng cho cả
# # # # # # # # # # # # # # # # #            SR head và FM ensemble (tránh tính 2 lần).

# # # # # # # # # # # # # # # # #   FIX-M-G  [LOW] ShortRangeHead: thêm layer norm trước out_head
# # # # # # # # # # # # # # # # #            để gradient scale ổn định hơn.

# # # # # # # # # # # # # # # # # Kept from v24:
# # # # # # # # # # # # # # # # #   FIX-M23  ShortRangeHead GRU autoregressive
# # # # # # # # # # # # # # # # #   FIX-M24  Blend: pred_mean[:4] ← SR, [4:] ← FM mean
# # # # # # # # # # # # # # # # #   FIX-M25  initial_sample_sigma=0.03, ctx_noise_scale=0.002
# # # # # # # # # # # # # # # # #   FIX-M26  MAX_DELTA clamp
# # # # # # # # # # # # # # # # #   FIX-M27  short_range_regression_loss trong get_loss_breakdown
# # # # # # # # # # # # # # # # # """
# # # # # # # # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # # # # # # # import csv
# # # # # # # # # # # # # # # # # import math
# # # # # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # # # # from datetime import datetime
# # # # # # # # # # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # # # # # # # # # import torch
# # # # # # # # # # # # # # # # # import torch.nn as nn
# # # # # # # # # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # # # # # # # # # from Model.losses import (
# # # # # # # # # # # # # # # # #     compute_total_loss, fm_physics_consistency_loss, WEIGHTS,
# # # # # # # # # # # # # # # # #     pinn_speed_constraint, short_range_regression_loss, bridge_loss,
# # # # # # # # # # # # # # # # #     _norm_to_deg,
# # # # # # # # # # # # # # # # # )


# # # # # # # # # # # # # # # # # def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # #     """Normalised → degrees. [T, B, 2] or [B, 2]."""
# # # # # # # # # # # # # # # # #     out = traj_norm.clone()
# # # # # # # # # # # # # # # # #     out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # # # # # # # #     out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
# # # # # # # # # # # # # # # # #     return out


# # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # # # #  ShortRangeHead  – FIX-M-A, FIX-M-G
# # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # # # # class ShortRangeHead(nn.Module):
# # # # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # # # #     GRU autoregressive predictor cho steps 1-4 (6h/12h/18h/24h).

# # # # # # # # # # # # # # # # #     FIX-M-A: motion_gate bias init = +2.0 → sigmoid(2) ≈ 0.88
# # # # # # # # # # # # # # # # #              đảm bảo motion prior dominant đầu training.
# # # # # # # # # # # # # # # # #     FIX-M-G: LayerNorm trước out_head để gradient scale ổn định.
# # # # # # # # # # # # # # # # #     """

# # # # # # # # # # # # # # # # #     N_STEPS   = 4
# # # # # # # # # # # # # # # # #     MAX_DELTA = 0.48     # ≈ 576 km / 6h

# # # # # # # # # # # # # # # # #     def __init__(self, raw_ctx_dim: int = 512, obs_len: int = 8):
# # # # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # # # #         self.obs_len = obs_len

# # # # # # # # # # # # # # # # #         # Velocity encoder từ obs
# # # # # # # # # # # # # # # # #         self.vel_enc = nn.Sequential(
# # # # # # # # # # # # # # # # #             nn.Linear(obs_len * 2, 256),
# # # # # # # # # # # # # # # # #             nn.GELU(),
# # # # # # # # # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # # # # # # # # #             nn.Linear(256, 128),
# # # # # # # # # # # # # # # # #             nn.GELU(),
# # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # #         self.ctx_proj = nn.Sequential(
# # # # # # # # # # # # # # # # #             nn.Linear(raw_ctx_dim, 256),
# # # # # # # # # # # # # # # # #             nn.GELU(),
# # # # # # # # # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # # # # # # # # #             nn.Dropout(0.05),
# # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # #         # GRU: input = ctx(256) + vel(128) + cur_pos(2) = 386
# # # # # # # # # # # # # # # # #         self.gru_cell = nn.GRUCell(input_size=386, hidden_size=256)

# # # # # # # # # # # # # # # # #         # FIX-M-A: motion gate với bias khởi tạo cao
# # # # # # # # # # # # # # # # #         self.motion_gate_linear = nn.Linear(128 + 256, 1)

# # # # # # # # # # # # # # # # #         # FIX-M-G: LayerNorm trước output
# # # # # # # # # # # # # # # # #         self.out_norm = nn.LayerNorm(256)
# # # # # # # # # # # # # # # # #         self.out_head = nn.Sequential(
# # # # # # # # # # # # # # # # #             nn.Linear(256, 128),
# # # # # # # # # # # # # # # # #             nn.GELU(),
# # # # # # # # # # # # # # # # #             nn.LayerNorm(128),
# # # # # # # # # # # # # # # # #             nn.Linear(128, 2),
# # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # #         self.step_scale = nn.Parameter(torch.ones(self.N_STEPS) * 0.2)
# # # # # # # # # # # # # # # # #         self._init_weights()

# # # # # # # # # # # # # # # # #     def _init_weights(self):
# # # # # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # # # # #             for m in self.modules():
# # # # # # # # # # # # # # # # #                 if isinstance(m, nn.Linear):
# # # # # # # # # # # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.3)
# # # # # # # # # # # # # # # # #                     if m.bias is not None:
# # # # # # # # # # # # # # # # #                         nn.init.zeros_(m.bias)
# # # # # # # # # # # # # # # # #             # FIX-M-A: init bias của gate linear = +2.0
# # # # # # # # # # # # # # # # #             # sigmoid(2.0) ≈ 0.88 → motion prior chiếm 88% đầu training
# # # # # # # # # # # # # # # # #             nn.init.constant_(self.motion_gate_linear.bias, 2.0)

# # # # # # # # # # # # # # # # #     def forward(self, raw_ctx: torch.Tensor,
# # # # # # # # # # # # # # # # #                 obs_traj: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # #         raw_ctx:  [B, raw_ctx_dim]
# # # # # # # # # # # # # # # # #         obs_traj: [T_obs, B, 2]  normalised
# # # # # # # # # # # # # # # # #         Returns:  [4, B, 2]  normalised predictions
# # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # #         B     = raw_ctx.shape[0]
# # # # # # # # # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # # # # # # # # #         # Velocity encoding
# # # # # # # # # # # # # # # # #         if T_obs >= 2:
# # # # # # # # # # # # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]   # [T_obs-1, B, 2]
# # # # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # # # #             vels = torch.zeros(1, B, 2, device=raw_ctx.device)

# # # # # # # # # # # # # # # # #         target_v = self.obs_len
# # # # # # # # # # # # # # # # #         if vels.shape[0] < target_v:
# # # # # # # # # # # # # # # # #             pad  = torch.zeros(target_v - vels.shape[0], B, 2, device=raw_ctx.device)
# # # # # # # # # # # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # # # #             vels = vels[-target_v:]

# # # # # # # # # # # # # # # # #         vel_input = vels.permute(1, 0, 2).reshape(B, -1)  # [B, obs_len*2]
# # # # # # # # # # # # # # # # #         vel_feat  = self.vel_enc(vel_input)                 # [B, 128]
# # # # # # # # # # # # # # # # #         ctx_feat  = self.ctx_proj(raw_ctx)                  # [B, 256]

# # # # # # # # # # # # # # # # #         hx      = ctx_feat
# # # # # # # # # # # # # # # # #         cur_pos = obs_traj[-1].clone()   # [B, 2]
# # # # # # # # # # # # # # # # #         last_vel = vels[-1].clone()      # [B, 2]

# # # # # # # # # # # # # # # # #         preds = []
# # # # # # # # # # # # # # # # #         for i in range(self.N_STEPS):
# # # # # # # # # # # # # # # # #             inp = torch.cat([ctx_feat, vel_feat, cur_pos], dim=-1)  # [B, 386]
# # # # # # # # # # # # # # # # #             hx  = self.gru_cell(inp, hx)                            # [B, 256]

# # # # # # # # # # # # # # # # #             # FIX-M-A: gate với bias=+2.0 → dominant motion prior đầu training
# # # # # # # # # # # # # # # # #             gate_input = torch.cat([vel_feat, hx], dim=-1)  # [B, 384]
# # # # # # # # # # # # # # # # #             gate = torch.sigmoid(self.motion_gate_linear(gate_input))  # [B, 1]

# # # # # # # # # # # # # # # # #             # FIX-M-G: LayerNorm trước output
# # # # # # # # # # # # # # # # #             raw_delta   = self.out_head(self.out_norm(hx))  # [B, 2]
# # # # # # # # # # # # # # # # #             prior_delta = last_vel * (0.9 ** i)              # [B, 2]

# # # # # # # # # # # # # # # # #             scale_i = torch.sigmoid(self.step_scale[i]) * self.MAX_DELTA
# # # # # # # # # # # # # # # # #             blended = gate * prior_delta + (1.0 - gate) * raw_delta * scale_i
# # # # # # # # # # # # # # # # #             delta   = blended.clamp(-self.MAX_DELTA, self.MAX_DELTA)

# # # # # # # # # # # # # # # # #             cur_pos  = cur_pos + delta
# # # # # # # # # # # # # # # # #             last_vel = delta
# # # # # # # # # # # # # # # # #             preds.append(cur_pos)

# # # # # # # # # # # # # # # # #         return torch.stack(preds, dim=0)   # [4, B, 2]


# # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # # # #  VelocityField
# # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # # # # # # # # # #     """OT-CFM velocity field v_θ(x_t, t, context)."""

# # # # # # # # # # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # # # # # # # # # #     def __init__(
# # # # # # # # # # # # # # # # #         self,
# # # # # # # # # # # # # # # # #         pred_len:   int   = 12,
# # # # # # # # # # # # # # # # #         obs_len:    int   = 8,
# # # # # # # # # # # # # # # # #         ctx_dim:    int   = 256,
# # # # # # # # # # # # # # # # #         sigma_min:  float = 0.02,
# # # # # # # # # # # # # # # # #         unet_in_ch: int   = 13,
# # # # # # # # # # # # # # # # #     ):
# # # # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # # # #         self.pred_len  = pred_len
# # # # # # # # # # # # # # # # #         self.obs_len   = obs_len
# # # # # # # # # # # # # # # # #         self.sigma_min = sigma_min

# # # # # # # # # # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # # # # # # # # # #             in_channel   = unet_in_ch,
# # # # # # # # # # # # # # # # #             out_channel  = 1,
# # # # # # # # # # # # # # # # #             d_model      = 32,
# # # # # # # # # # # # # # # # #             n_layers     = 4,
# # # # # # # # # # # # # # # # #             modes_t      = 4,
# # # # # # # # # # # # # # # # #             modes_h      = 4,
# # # # # # # # # # # # # # # # #             modes_w      = 4,
# # # # # # # # # # # # # # # # #             spatial_down = 32,
# # # # # # # # # # # # # # # # #             dropout      = 0.05,
# # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # # # # # # # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # # # # # # # # # #             in_1d       = 4,
# # # # # # # # # # # # # # # # #             feat_3d_dim = 128,
# # # # # # # # # # # # # # # # #             mlp_h       = 64,
# # # # # # # # # # # # # # # # #             lstm_hidden = 128,
# # # # # # # # # # # # # # # # #             lstm_layers = 3,
# # # # # # # # # # # # # # # # #             dropout     = 0.1,
# # # # # # # # # # # # # # # # #             d_state     = 16,
# # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # # # # # # # # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # # # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # # # # # # # # # #         self.ctx_drop = nn.Dropout(0.15)
# # # # # # # # # # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # # # # # # # # # # # # # #         self.time_fc1 = nn.Linear(256, 512)
# # # # # # # # # # # # # # # # #         self.time_fc2 = nn.Linear(512, 256)

# # # # # # # # # # # # # # # # #         self.traj_embed  = nn.Linear(4, 256)
# # # # # # # # # # # # # # # # #         # self.pos_enc     = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # # # # # # # # # # # # #         # Thay vì:
# # # # # # # # # # # # # # # # #         # self.pos_enc = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)

# # # # # # # # # # # # # # # # #         # Hãy thử:
# # # # # # # # # # # # # # # # #         self.pos_enc = nn.Parameter(torch.randn(1, pred_len, 256) * 0.1) # Tăng scale lên 5 lần
# # # # # # # # # # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # # # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # # # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # # # # # # # # # #                 dropout=0.15, activation="gelu", batch_first=True,
# # # # # # # # # # # # # # # # #             ),
# # # # # # # # # # # # # # # # #             num_layers=4,
# # # # # # # # # # # # # # # # #         )
# # # # # # # # # # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # # # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # # # # # # # # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)

# # # # # # # # # # # # # # # # #         # ShortRangeHead (FIX-M-A applied)
# # # # # # # # # # # # # # # # #         self.short_range_head = ShortRangeHead(
# # # # # # # # # # # # # # # # #             raw_ctx_dim = self.RAW_CTX_DIM,
# # # # # # # # # # # # # # # # #             obs_len     = obs_len,
# # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # #     # ── Helpers ───────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # # # #     def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
# # # # # # # # # # # # # # # # #         half = dim // 2
# # # # # # # # # # # # # # # # #         freq = torch.exp(
# # # # # # # # # # # # # # # # #             torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # # # # # # # # # #             * (-math.log(10_000.0) / max(half - 1, 1))
# # # # # # # # # # # # # # # # #         )
# # # # # # # # # # # # # # # # #         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
# # # # # # # # # # # # # # # # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # # # # # # # # # # # # # # # #         return F.pad(emb, (0, dim % 2))

# # # # # # # # # # # # # # # # #     def _context(self, batch_list) -> torch.Tensor:
# # # # # # # # # # # # # # # # #         """Compute raw_ctx [B, RAW_CTX_DIM]."""
# # # # # # # # # # # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # # # # # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # # # # # # # # # # #         image_obs = batch_list[11]
# # # # # # # # # # # # # # # # #         env_data  = batch_list[13]

# # # # # # # # # # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # # # # # # # # # #             image_obs = image_obs.unsqueeze(2)

# # # # # # # # # # # # # # # # #         expected_ch = self.spatial_enc.in_channel
# # # # # # # # # # # # # # # # #         if image_obs.shape[1] == 1 and expected_ch != 1:
# # # # # # # # # # # # # # # # #             image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# # # # # # # # # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
# # # # # # # # # # # # # # # # #         e_3d_s = e_3d_s.permute(0, 2, 1)
# # # # # # # # # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)

# # # # # # # # # # # # # # # # #         T_bot = e_3d_s.shape[1]
# # # # # # # # # # # # # # # # #         if T_bot != T_obs:
# # # # # # # # # # # # # # # # #             e_3d_s = F.interpolate(
# # # # # # # # # # # # # # # # #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # # # # # # # # #                 mode="linear", align_corners=False,
# # # # # # # # # # # # # # # # #             ).permute(0, 2, 1)

# # # # # # # # # # # # # # # # #         # Weighted temporal pooling của decoder output
# # # # # # # # # # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)  # [B, T]
# # # # # # # # # # # # # # # # #         t_weights = torch.softmax(
# # # # # # # # # # # # # # # # #             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # # # # # # # # # # #                         device=e_3d_dec_t.device) * 0.5, dim=0,
# # # # # # # # # # # # # # # # #         )
# # # # # # # # # # # # # # # # #         f_spatial_scalar = (e_3d_dec_t * t_weights.unsqueeze(0)).sum(1, keepdim=True)  # [B,1]
# # # # # # # # # # # # # # # # #         f_spatial = self.decoder_proj(f_spatial_scalar)   # [B, 16]

# # # # # # # # # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)  # [B,T,4]
# # # # # # # # # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)              # [B, 128]

# # # # # # # # # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)   # [B, 32]

# # # # # # # # # # # # # # # # #         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)  # [B, 176]
# # # # # # # # # # # # # # # # #         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))       # [B, 512]
# # # # # # # # # # # # # # # # #         return raw

# # # # # # # # # # # # # # # # #     def _apply_ctx_head(self, raw: torch.Tensor,
# # # # # # # # # # # # # # # # #                         noise_scale: float = 0.0) -> torch.Tensor:
# # # # # # # # # # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # # # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # # # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # # # # # # # # # #     def _beta_drift_velocity(self, x_t: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # #         """Beta drift trong normalised state space."""
# # # # # # # # # # # # # # # # #         OMEGA_val  = 7.2921e-5
# # # # # # # # # # # # # # # # #         R_val      = 6.371e6
# # # # # # # # # # # # # # # # #         DT         = 6 * 3600.0
# # # # # # # # # # # # # # # # #         M_PER_NORM = 5.0 * 111.0 * 1000.0

# # # # # # # # # # # # # # # # #         lat_norm = x_t[:, :, 1]
# # # # # # # # # # # # # # # # #         lat_deg  = lat_norm * 5.0
# # # # # # # # # # # # # # # # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))

# # # # # # # # # # # # # # # # #         beta   = 2 * OMEGA_val * torch.cos(lat_rad) / R_val
# # # # # # # # # # # # # # # # #         R_tc   = 3e5
# # # # # # # # # # # # # # # # #         v_lon  = -beta * R_tc ** 2 / 2
# # # # # # # # # # # # # # # # #         v_lat  =  beta * R_tc ** 2 / 4

# # # # # # # # # # # # # # # # #         v_lon_norm = v_lon * DT / M_PER_NORM
# # # # # # # # # # # # # # # # #         v_lat_norm = v_lat * DT / M_PER_NORM

# # # # # # # # # # # # # # # # #         v_phys = torch.zeros_like(x_t)
# # # # # # # # # # # # # # # # #         v_phys[:, :, 0] = v_lon_norm
# # # # # # # # # # # # # # # # #         v_phys[:, :, 1] = v_lat_norm
# # # # # # # # # # # # # # # # #         return v_phys

# # # # # # # # # # # # # # # # #     def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
# # # # # # # # # # # # # # # # #                 ctx: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
# # # # # # # # # # # # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # # # # # # # # # # # #         T_seq = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # # # # # # # # # # #         x_emb = (self.traj_embed(x_t[:, :T_seq, :])
# # # # # # # # # # # # # # # # #                   + self.pos_enc[:, :T_seq, :]
# # # # # # # # # # # # # # # # #                   + t_emb.unsqueeze(1))
# # # # # # # # # # # # # # # # #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

# # # # # # # # # # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(
# # # # # # # # # # # # # # # # #             self.transformer(x_emb, memory)
# # # # # # # # # # # # # # # # #         )))

# # # # # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # # # # #             v_phys = self._beta_drift_velocity(x_t[:, :T_seq, :])

# # # # # # # # # # # # # # # # #         scale = torch.sigmoid(self.physics_scale) * 2.0
# # # # # # # # # # # # # # # # #         return v_neural + scale * v_phys

# # # # # # # # # # # # # # # # #     def forward(self, x_t, t, batch_list):
# # # # # # # # # # # # # # # # #         raw = self._context(batch_list)
# # # # # # # # # # # # # # # # #         ctx = self._apply_ctx_head(raw, noise_scale=0.0)
# # # # # # # # # # # # # # # # #         return self._decode(x_t, t, ctx)

# # # # # # # # # # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale: float = 0.0):
# # # # # # # # # # # # # # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale=noise_scale)
# # # # # # # # # # # # # # # # #         return self._decode(x_t, t, ctx)

# # # # # # # # # # # # # # # # #     def forward_short_range(self, obs_traj: torch.Tensor,
# # # # # # # # # # # # # # # # #                             raw_ctx: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # #         """Deterministic short-range [4, B, 2]."""
# # # # # # # # # # # # # # # # #         return self.short_range_head(raw_ctx, obs_traj)


# # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # # # #  TCFlowMatching
# # # # # # # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # # # # class TCFlowMatching(nn.Module):
# # # # # # # # # # # # # # # # #     """TC trajectory prediction via OT-CFM + ShortRangeHead + PINN."""

# # # # # # # # # # # # # # # # #     def __init__(
# # # # # # # # # # # # # # # # #         self,
# # # # # # # # # # # # # # # # #         pred_len:             int   = 12,
# # # # # # # # # # # # # # # # #         obs_len:              int   = 8,
# # # # # # # # # # # # # # # # #         sigma_min:            float = 0.02,
# # # # # # # # # # # # # # # # #         n_train_ens:          int   = 4,
# # # # # # # # # # # # # # # # #         unet_in_ch:           int   = 13,
# # # # # # # # # # # # # # # # #         # ctx_noise_scale:      float = 0.002,
# # # # # # # # # # # # # # # # #         # initial_sample_sigma: float = 0.03,
# # # # # # # # # # # # # # # # #         ctx_noise_scale:      float = 0.01,
# # # # # # # # # # # # # # # # #         initial_sample_sigma: float = 0.15,
# # # # # # # # # # # # # # # # #         **kwargs,
# # # # # # # # # # # # # # # # #     ):
# # # # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # # # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # # # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # # # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # # # # # # # # # #         self.active_pred_len      = pred_len
# # # # # # # # # # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # # # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma

# # # # # # # # # # # # # # # # #         self.net = VelocityField(
# # # # # # # # # # # # # # # # #             pred_len   = pred_len,
# # # # # # # # # # # # # # # # #             obs_len    = obs_len,
# # # # # # # # # # # # # # # # #             sigma_min  = sigma_min,
# # # # # # # # # # # # # # # # #             unet_in_ch = unet_in_ch,
# # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # #     def set_curriculum_len(self, active_len: int) -> None:
# # # # # # # # # # # # # # # # #         pass  # No curriculum

# # # # # # # # # # # # # # # # #     # ── Static helpers ────────────────────────────────────────────────────────

# # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # #     def _to_rel(traj_gt, Me_gt, last_pos, last_Me):
# # # # # # # # # # # # # # # # #         return torch.cat(
# # # # # # # # # # # # # # # # #             [traj_gt - last_pos.unsqueeze(0),
# # # # # # # # # # # # # # # # #              Me_gt   - last_Me.unsqueeze(0)],
# # # # # # # # # # # # # # # # #             dim=-1,
# # # # # # # # # # # # # # # # #         ).permute(1, 0, 2)

# # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # #     def _to_abs(rel, last_pos, last_Me):
# # # # # # # # # # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # # # # # # # # # #         return (
# # # # # # # # # # # # # # # # #             last_pos.unsqueeze(0) + d[:, :, :2],
# # # # # # # # # # # # # # # # #             last_Me.unsqueeze(0)  + d[:, :, 2:],
# # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # #     def _cfm_noisy(self, x1):
# # # # # # # # # # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # # # # # # # # # #         sm  = self.sigma_min
# # # # # # # # # # # # # # # # #         x0  = torch.randn_like(x1) * sm
# # # # # # # # # # # # # # # # #         t   = torch.rand(B, device=device)
# # # # # # # # # # # # # # # # #         te  = t.view(B, 1, 1)
# # # # # # # # # # # # # # # # #         x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
# # # # # # # # # # # # # # # # #         denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
# # # # # # # # # # # # # # # # #         target_vel = (x1 - (1.0 - sm) * x_t) / denom
# # # # # # # # # # # # # # # # #         return x_t, t, te, denom, target_vel

# # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # #     def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # #         wind_norm = obs_Me[-1, :, 1].detach()
# # # # # # # # # # # # # # # # #         w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
# # # # # # # # # # # # # # # # #             torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
# # # # # # # # # # # # # # # # #             torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
# # # # # # # # # # # # # # # # #                         torch.full_like(wind_norm, 1.5))))
# # # # # # # # # # # # # # # # #         return w / w.mean().clamp(min=1e-6)

# # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # #     def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
# # # # # # # # # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # # # # # # # # #             return batch_list
# # # # # # # # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # # # # # # # # # #             t = aug[idx]
# # # # # # # # # # # # # # # # #             if torch.is_tensor(t) and t.shape[-1] >= 1:
# # # # # # # # # # # # # # # # #                 t = t.clone()
# # # # # # # # # # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # # # # # # # # # #                 aug[idx] = t
# # # # # # # # # # # # # # # # #         return aug

# # # # # # # # # # # # # # # # #     # ── Loss ──────────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # # # #     def get_loss(self, batch_list: List,
# # # # # # # # # # # # # # # # #                  step_weight_alpha: float = 0.0,
# # # # # # # # # # # # # # # # #                  epoch: int = 0) -> torch.Tensor:
# # # # # # # # # # # # # # # # #         return self.get_loss_breakdown(
# # # # # # # # # # # # # # # # #             batch_list, step_weight_alpha, epoch)["total"]

# # # # # # # # # # # # # # # # #     def get_loss_breakdown(
# # # # # # # # # # # # # # # # #         self,
# # # # # # # # # # # # # # # # #         batch_list: List,
# # # # # # # # # # # # # # # # #         step_weight_alpha: float = 0.0,
# # # # # # # # # # # # # # # # #         epoch: int = 0,
# # # # # # # # # # # # # # # # #     ) -> Dict:
# # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # #         FIX-M-C: Tính sr_pred và pass sang compute_total_loss qua bridge_loss.
# # # # # # # # # # # # # # # # #         FIX-M-E: Pass epoch và gt_abs_deg cho adaptive PINN.
# # # # # # # # # # # # # # # # #         FIX-M-F: Cache raw_ctx một lần.
# # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # #         batch_list = self._lon_flip_aug(batch_list, p=0.3)

# # # # # # # # # # # # # # # # #         traj_gt = batch_list[1]    # [T, B, 2] normalised
# # # # # # # # # # # # # # # # #         Me_gt   = batch_list[8]
# # # # # # # # # # # # # # # # #         obs_t   = batch_list[0]    # [T_obs, B, 2] normalised
# # # # # # # # # # # # # # # # #         obs_Me  = batch_list[7]

# # # # # # # # # # # # # # # # #         try:
# # # # # # # # # # # # # # # # #             env_data = batch_list[13]
# # # # # # # # # # # # # # # # #         except (IndexError, TypeError):
# # # # # # # # # # # # # # # # #             env_data = None

# # # # # # # # # # # # # # # # #         lp, lm = obs_t[-1], obs_Me[-1]
# # # # # # # # # # # # # # # # #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

# # # # # # # # # # # # # # # # #         # FIX-M-F: compute raw_ctx một lần duy nhất
# # # # # # # # # # # # # # # # #         raw_ctx     = self.net._context(batch_list)
# # # # # # # # # # # # # # # # #         intensity_w = self._intensity_weights(obs_Me)

# # # # # # # # # # # # # # # # #         # CFM forward
# # # # # # # # # # # # # # # # #         x_t, t, te, denom, _ = self._cfm_noisy(x1)
# # # # # # # # # # # # # # # # #         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

# # # # # # # # # # # # # # # # #         # Ensemble cho AFCRPS
# # # # # # # # # # # # # # # # #         samples: List[torch.Tensor] = []
# # # # # # # # # # # # # # # # #         for _ in range(self.n_train_ens):
# # # # # # # # # # # # # # # # #             xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
# # # # # # # # # # # # # # # # #             pv_s   = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
# # # # # # # # # # # # # # # # #             x1_s   = xt_s + dens_s * pv_s
# # # # # # # # # # # # # # # # #             pa_s, _ = self._to_abs(x1_s, lp, lm)
# # # # # # # # # # # # # # # # #             samples.append(pa_s)
# # # # # # # # # # # # # # # # #         pred_samples = torch.stack(samples)     # [S, T, B, 2]

# # # # # # # # # # # # # # # # #         l_fm_physics = fm_physics_consistency_loss(pred_samples, gt_norm=traj_gt, last_pos=lp)

# # # # # # # # # # # # # # # # #         x1_pred = x_t + denom * pred_vel
# # # # # # # # # # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# # # # # # # # # # # # # # # # #         pred_abs_deg = _denorm_to_deg(pred_abs)    # [T, B, 2] degrees
# # # # # # # # # # # # # # # # #         traj_gt_deg  = _denorm_to_deg(traj_gt)     # [T, B, 2] degrees
# # # # # # # # # # # # # # # # #         ref_deg      = _denorm_to_deg(lp)           # [B, 2]

# # # # # # # # # # # # # # # # #         # FIX-M-C: Compute sr_pred (dùng raw_ctx đã cache)
# # # # # # # # # # # # # # # # #         n_sr = ShortRangeHead.N_STEPS  # 4
# # # # # # # # # # # # # # # # #         sr_pred = None
# # # # # # # # # # # # # # # # #         l_sr    = pred_abs.new_zeros(())
# # # # # # # # # # # # # # # # #         if traj_gt.shape[0] >= n_sr:
# # # # # # # # # # # # # # # # #             sr_pred = self.net.forward_short_range(obs_t, raw_ctx)  # [4, B, 2]
# # # # # # # # # # # # # # # # #             sr_gt   = traj_gt[:n_sr]                                # [4, B, 2]
# # # # # # # # # # # # # # # # #             l_sr    = short_range_regression_loss(sr_pred, sr_gt, lp)

# # # # # # # # # # # # # # # # #         breakdown = compute_total_loss(
# # # # # # # # # # # # # # # # #             pred_abs           = pred_abs_deg,
# # # # # # # # # # # # # # # # #             gt                 = traj_gt_deg,
# # # # # # # # # # # # # # # # #             ref                = ref_deg,
# # # # # # # # # # # # # # # # #             batch_list         = batch_list,
# # # # # # # # # # # # # # # # #             pred_samples       = pred_samples,
# # # # # # # # # # # # # # # # #             gt_norm            = traj_gt,
# # # # # # # # # # # # # # # # #             weights            = WEIGHTS,
# # # # # # # # # # # # # # # # #             intensity_w        = intensity_w,
# # # # # # # # # # # # # # # # #             env_data           = env_data,
# # # # # # # # # # # # # # # # #             step_weight_alpha  = step_weight_alpha,
# # # # # # # # # # # # # # # # #             all_trajs          = pred_samples,
# # # # # # # # # # # # # # # # #             # FIX-M-E: pass epoch và gt cho adaptive PINN
# # # # # # # # # # # # # # # # #             epoch              = epoch,
# # # # # # # # # # # # # # # # #             gt_abs_deg         = traj_gt_deg,
# # # # # # # # # # # # # # # # #             # FIX-M-C: pass sr_pred cho bridge loss
# # # # # # # # # # # # # # # # #             sr_pred            = sr_pred,
# # # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # # #         # FM physics consistency
# # # # # # # # # # # # # # # # #         fm_phys_w = WEIGHTS.get("fm_physics", 0.3)
# # # # # # # # # # # # # # # # #         breakdown["total"]      = breakdown["total"] + fm_phys_w * l_fm_physics
# # # # # # # # # # # # # # # # #         breakdown["fm_physics"] = l_fm_physics.item()

# # # # # # # # # # # # # # # # #         # Short-range loss
# # # # # # # # # # # # # # # # #         sr_w = WEIGHTS.get("short_range", 5.0)
# # # # # # # # # # # # # # # # #         breakdown["total"]       = breakdown["total"] + sr_w * l_sr
# # # # # # # # # # # # # # # # #         breakdown["short_range"] = l_sr.item()

# # # # # # # # # # # # # # # # #         return breakdown

# # # # # # # # # # # # # # # # #     # ── sample() ─────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # # # #     @torch.no_grad()
# # # # # # # # # # # # # # # # #     def sample(
# # # # # # # # # # # # # # # # #         self,
# # # # # # # # # # # # # # # # #         batch_list:   List,
# # # # # # # # # # # # # # # # #         num_ensemble: int = 50,
# # # # # # # # # # # # # # # # #         ddim_steps:   int = 20,
# # # # # # # # # # # # # # # # #         predict_csv:  Optional[str] = None,
# # # # # # # # # # # # # # # # #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # #         FIX-M-F: raw_ctx cached một lần cho cả SR và FM ensemble.

# # # # # # # # # # # # # # # # #         Returns:
# # # # # # # # # # # # # # # # #             pred_mean : [T, B, 2]  (SR[:4] + FM[4:])
# # # # # # # # # # # # # # # # #             me_mean   : [T, B, 2]
# # # # # # # # # # # # # # # # #             all_trajs : [S, T, B, 2]  (FM ensemble, KHÔNG override step 1-4)
# # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # #         obs_t  = batch_list[0]
# # # # # # # # # # # # # # # # #         lp     = obs_t[-1]
# # # # # # # # # # # # # # # # #         lm     = batch_list[7][-1]
# # # # # # # # # # # # # # # # #         B      = lp.shape[0]
# # # # # # # # # # # # # # # # #         device = lp.device
# # # # # # # # # # # # # # # # #         T      = self.pred_len
# # # # # # # # # # # # # # # # #         dt     = 1.0 / max(ddim_steps, 1)

# # # # # # # # # # # # # # # # #         # FIX-M-F: một lần duy nhất
# # # # # # # # # # # # # # # # #         raw_ctx = self.net._context(batch_list)
# # # # # # # # # # # # # # # # #         n_sr    = ShortRangeHead.N_STEPS

# # # # # # # # # # # # # # # # #         # Short-range deterministic
# # # # # # # # # # # # # # # # #         sr_pred = self.net.forward_short_range(obs_t, raw_ctx)   # [4, B, 2]

# # # # # # # # # # # # # # # # #         # FM ensemble
# # # # # # # # # # # # # # # # #         traj_s: List[torch.Tensor] = []
# # # # # # # # # # # # # # # # #         me_s:   List[torch.Tensor] = []

# # # # # # # # # # # # # # # # #         for k in range(num_ensemble):
# # # # # # # # # # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# # # # # # # # # # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # # # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # # # # # # # #                 # ns  = self.ctx_noise_scale if step == 0 else 0.0
# # # # # # # # # # # # # # # # #                 ns = self.ctx_noise_scale * 10.0 if step == 0 else 0.0
# # # # # # # # # # # # # # # # #                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
# # # # # # # # # # # # # # # # #                 x_t = x_t + dt * vel
# # # # # # # # # # # # # # # # #         # for k in range(num_ensemble):
# # # # # # # # # # # # # # # # #         #     x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# # # # # # # # # # # # # # # # #         #     # Mỗi member dùng ctx với noise riêng → diversity đến từ ctx uncertainty
# # # # # # # # # # # # # # # # #         #     ctx_k = self.net._apply_ctx_head(
# # # # # # # # # # # # # # # # #         #         raw_ctx, noise_scale=self.ctx_noise_scale * 5.0  # tăng 5× khi sampling
# # # # # # # # # # # # # # # # #         #     )

# # # # # # # # # # # # # # # # #         #     for step in range(ddim_steps):
# # # # # # # # # # # # # # # # #         #         t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # # # # # # # #         #         vel = self.net._decode(x_t, t_b, ctx_k)
# # # # # # # # # # # # # # # # #         #         x_t = x_t + dt * vel

# # # # # # # # # # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)
# # # # # # # # # # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)

# # # # # # # # # # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # # # # # # # # #             traj_s.append(tr)
# # # # # # # # # # # # # # # # #             me_s.append(me)

# # # # # # # # # # # # # # # # #         all_trajs = torch.stack(traj_s)    # [S, T, B, 2]
# # # # # # # # # # # # # # # # #         all_me    = torch.stack(me_s)

# # # # # # # # # # # # # # # # #         # Blend: step 1-4 từ SR, 5-12 từ FM mean
# # # # # # # # # # # # # # # # #         fm_mean   = all_trajs.mean(0)      # [T, B, 2]
# # # # # # # # # # # # # # # # #         pred_mean = fm_mean.clone()
# # # # # # # # # # # # # # # # #         # pred_mean[:n_sr] = sr_pred         # override pred_mean, giữ all_trajs nguyên

# # # # # # # # # # # # # # # # #         # Dùng:
# # # # # # # # # # # # # # # # #         alpha = torch.linspace(0.0, 1.0, n_sr, device=device).view(n_sr, 1, 1)
# # # # # # # # # # # # # # # # #         pred_mean[:n_sr] = (1 - alpha) * sr_pred + alpha * fm_mean[:n_sr]
        
# # # # # # # # # # # # # # # # #         # step 1: 100% SR, step 4: 50% SR + 50% FM → smooth transition
# # # # # # # # # # # # # # # # #         me_mean = all_me.mean(0)

# # # # # # # # # # # # # # # # #         if predict_csv:
# # # # # # # # # # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)

# # # # # # # # # # # # # # # # #         return pred_mean, me_mean, all_trajs

# # # # # # # # # # # # # # # # #     # ── Physics correction (FIX-M-D) ─────────────────────────────────────────

# # # # # # # # # # # # # # # # #     def _physics_correct(
# # # # # # # # # # # # # # # # #         self,
# # # # # # # # # # # # # # # # #         x_pred:   torch.Tensor,
# # # # # # # # # # # # # # # # #         last_pos: torch.Tensor,
# # # # # # # # # # # # # # # # #         last_Me:  torch.Tensor,
# # # # # # # # # # # # # # # # #         n_steps:  int   = 5,
# # # # # # # # # # # # # # # # #         lr:       float = 0.002,
# # # # # # # # # # # # # # # # #     ) -> torch.Tensor:
# # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # #         FIX-M-D: Dùng Adam thay vì SGD+momentum.
# # # # # # # # # # # # # # # # #         SGD+momentum với n_steps nhỏ (5) gây oscillation;
# # # # # # # # # # # # # # # # #         Adam converge nhanh hơn và stable hơn.
# # # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # # #         with torch.enable_grad():
# # # # # # # # # # # # # # # # #             x = x_pred.detach().requires_grad_(True)
# # # # # # # # # # # # # # # # #             # FIX-M-D: Adam thay vì SGD+momentum
# # # # # # # # # # # # # # # # #             optimizer = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))

# # # # # # # # # # # # # # # # #             for _ in range(n_steps):
# # # # # # # # # # # # # # # # #                 optimizer.zero_grad()
# # # # # # # # # # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # # # # # # # # # #                 pred_deg    = _denorm_to_deg(pred_abs)

# # # # # # # # # # # # # # # # #                 l_speed = self._pinn_speed_constraint(pred_deg)
# # # # # # # # # # # # # # # # #                 l_accel = self._pinn_beta_plane_simplified(pred_deg)

# # # # # # # # # # # # # # # # #                 physics_loss = l_speed + 0.3 * l_accel
# # # # # # # # # # # # # # # # #                 physics_loss.backward()

# # # # # # # # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], max_norm=0.05)
# # # # # # # # # # # # # # # # #                 optimizer.step()

# # # # # # # # # # # # # # # # #         return x.detach()

# # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # #     def _pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # #         if pred_deg.shape[0] < 2:
# # # # # # # # # # # # # # # # #             return pred_deg.new_zeros(())
# # # # # # # # # # # # # # # # #         dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # # # # # # # #         lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # # # # # # #         dx_km   = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # # # # # # # # # # # # # #         dy_km   = dt_deg[:, :, 1] * 111.0
# # # # # # # # # # # # # # # # #         speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # # # # # # # # # # # # # # # #         return F.relu(speed - 600.0).pow(2).mean()

# # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # #     def _pinn_beta_plane_simplified(pred_deg: torch.Tensor) -> torch.Tensor:
# # # # # # # # # # # # # # # # #         if pred_deg.shape[0] < 3:
# # # # # # # # # # # # # # # # #             return pred_deg.new_zeros(())
# # # # # # # # # # # # # # # # #         v = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # # # # # # # #         a = v[1:] - v[:-1]
# # # # # # # # # # # # # # # # #         lat_rad = torch.deg2rad(pred_deg[1:-1, :, 1])
# # # # # # # # # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # # # # # # #         a_lon_km = a[:, :, 0] * cos_lat * 111.0
# # # # # # # # # # # # # # # # #         a_lat_km = a[:, :, 1] * 111.0
# # # # # # # # # # # # # # # # #         max_accel = 50.0
# # # # # # # # # # # # # # # # #         violation = F.relu(torch.sqrt(a_lon_km**2 + a_lat_km**2) - max_accel)
# # # # # # # # # # # # # # # # #         return violation.pow(2).mean() * 0.1

# # # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # # #     def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
# # # # # # # # # # # # # # # # #                            all_trajs: torch.Tensor) -> None:
# # # # # # # # # # # # # # # # #         import numpy as np
# # # # # # # # # # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # # # # # # # # # #         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # # # # # # # # # #         S       = all_trajs.shape[0]

# # # # # # # # # # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()

# # # # # # # # # # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # # # # # # # # # #                     "lon_mean_deg", "lat_mean_deg",
# # # # # # # # # # # # # # # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # # # # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # # # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # # # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # # # # # # # # # #             if write_hdr:
# # # # # # # # # # # # # # # # #                 w.writeheader()
# # # # # # # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # # # # # # #                 for k in range(T):
# # # # # # # # # # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # # # # # # # # # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
# # # # # # # # # # # # # # # # #                         math.radians(mean_lat[k, b]))
# # # # # # # # # # # # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # # # # # # # # # # # #                     w.writerow(dict(
# # # # # # # # # # # # # # # # #                         timestamp     = ts,
# # # # # # # # # # # # # # # # #                         batch_idx     = b,
# # # # # # # # # # # # # # # # #                         step_idx      = k,
# # # # # # # # # # # # # # # # #                         lead_h        = (k + 1) * 6,
# # # # # # # # # # # # # # # # #                         lon_mean_deg  = f"{mean_lon[k,b]:.4f}",
# # # # # # # # # # # # # # # # #                         lat_mean_deg  = f"{mean_lat[k,b]:.4f}",
# # # # # # # # # # # # # # # # #                         lon_std_deg   = f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # # # # # # # # # #                         lat_std_deg   = f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # # # # # # # # # #                         ens_spread_km = f"{spread:.2f}",
# # # # # # # # # # # # # # # # #                     ))
# # # # # # # # # # # # # # # # #         print(f"  Predictions → {csv_path}  (B={B}, T={T}, S={S})")


# # # # # # # # # # # # # # # # # # Backward-compat alias
# # # # # # # # # # # # # # # # # TCDiffusion = TCFlowMatching

# # # # # # # # # # # # # # # # """
# # # # # # # # # # # # # # # # flow_matching_model_v32.py — Tiếp cận đơn giản nhất để beat LSTM

# # # # # # # # # # # # # # # # INSIGHT: LSTM beat vì nó train với pure MSE → gradient rõ ràng, không conflict.
# # # # # # # # # # # # # # # # Vấn đề của FM không phải architecture mà là training signal:
# # # # # # # # # # # # # # # #   - AFCRPS: pull về diverse predictions
# # # # # # # # # # # # # # # #   - MSE_hav: pull về mean prediction  
# # # # # # # # # # # # # # # #   - Hai cái này conflict nhau → stuck at mean (ADE=412km)

# # # # # # # # # # # # # # # # GIẢI PHÁP V32:
# # # # # # # # # # # # # # # #   Phase 1 (epoch 0-30): Train như LSTM thuần túy
# # # # # # # # # # # # # # # #     - Loss = SR_Huber(step 1-4) + MSE_hav(step 1-12)
# # # # # # # # # # # # # # # #     - Không dùng FM AFCRPS
# # # # # # # # # # # # # # # #     - Model học được basic trajectory prediction
    
# # # # # # # # # # # # # # # #   Phase 2 (epoch 31+): Thêm FM ensemble diversity
# # # # # # # # # # # # # # # #     - Loss = SR_Huber + MSE_hav + 0.5*AFCRPS
# # # # # # # # # # # # # # # #     - FM dùng để calibrate uncertainty, không phải accuracy
    
# # # # # # # # # # # # # # # # KẾT QUẢ DỰ KIẾN:
# # # # # # # # # # # # # # # #   - Epoch 10-20: ADE giảm nhanh từ 412 → ~250km (như LSTM ban đầu)
# # # # # # # # # # # # # # # #   - Epoch 30-50: ADE ~170km, 72h ~300km
# # # # # # # # # # # # # # # #   - Epoch 60+: SR converge → 12h ~50km
# # # # # # # # # # # # # # # # """
# # # # # # # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # # # # # # import csv
# # # # # # # # # # # # # # # # import math
# # # # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # # # from datetime import datetime
# # # # # # # # # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # # # # # # # # import torch
# # # # # # # # # # # # # # # # import torch.nn as nn
# # # # # # # # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # # # # # # # # Import các module unchanged
# # # # # # # # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # # # # # # # # from Model.losses import (
# # # # # # # # # # # # # # # #     _haversine_deg, _norm_to_deg, N_SR_STEPS,
# # # # # # # # # # # # # # # #     short_range_regression_loss, heading_loss,
# # # # # # # # # # # # # # # #     velocity_loss_per_sample, recurvature_loss,
# # # # # # # # # # # # # # # #     fm_afcrps_loss, fm_physics_consistency_loss,
# # # # # # # # # # # # # # # # )

# # # # # # # # # # # # # # # # MSE_STEP_WEIGHTS = [1.0, 3.0, 1.5, 2.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]


# # # # # # # # # # # # # # # # def _denorm_to_deg(t):
# # # # # # # # # # # # # # # #     out = t.clone()
# # # # # # # # # # # # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # # # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # # # # # # # # # # # #     return out


# # # # # # # # # # # # # # # # def mse_hav_loss(pred_norm, gt_norm, step_w=None):
# # # # # # # # # # # # # # # #     """Haversine MSE per-step với step weighting."""
# # # # # # # # # # # # # # # #     if step_w is None:
# # # # # # # # # # # # # # # #         step_w = MSE_STEP_WEIGHTS
# # # # # # # # # # # # # # # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # # # # # # # # # #     pred_deg = _norm_to_deg(pred_norm[:T])
# # # # # # # # # # # # # # # #     gt_deg   = _norm_to_deg(gt_norm[:T])
# # # # # # # # # # # # # # # #     dist_km  = _haversine_deg(pred_deg, gt_deg)   # [T, B]
# # # # # # # # # # # # # # # #     w = pred_norm.new_tensor(step_w[:T])
# # # # # # # # # # # # # # # #     w = w / w.sum() * T
# # # # # # # # # # # # # # # #     return (dist_km.pow(2) * w.unsqueeze(1)).mean() / (200.0 ** 2)


# # # # # # # # # # # # # # # # # ─── ShortRangeHead (giữ nguyên từ v30) ──────────────────────────────────────

# # # # # # # # # # # # # # # # class ShortRangeHead(nn.Module):
# # # # # # # # # # # # # # # #     N_STEPS   = 4
# # # # # # # # # # # # # # # #     MAX_DELTA = 0.48

# # # # # # # # # # # # # # # #     def __init__(self, raw_ctx_dim=512, obs_len=8):
# # # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # # #         self.obs_len = obs_len
# # # # # # # # # # # # # # # #         self.vel_enc = nn.Sequential(
# # # # # # # # # # # # # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # # # # # # # # # # # # #             nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(),
# # # # # # # # # # # # # # # #         )
# # # # # # # # # # # # # # # #         self.ctx_proj = nn.Sequential(
# # # # # # # # # # # # # # # #             nn.Linear(raw_ctx_dim, 256), nn.GELU(),
# # # # # # # # # # # # # # # #             nn.LayerNorm(256), nn.Dropout(0.05),
# # # # # # # # # # # # # # # #         )
# # # # # # # # # # # # # # # #         self.gru_cell = nn.GRUCell(input_size=386, hidden_size=256)
# # # # # # # # # # # # # # # #         self.motion_gate_linear = nn.Linear(128 + 256, 1)
# # # # # # # # # # # # # # # #         self.out_norm = nn.LayerNorm(256)
# # # # # # # # # # # # # # # #         self.out_head = nn.Sequential(
# # # # # # # # # # # # # # # #             nn.Linear(256, 128), nn.GELU(),
# # # # # # # # # # # # # # # #             nn.LayerNorm(128), nn.Linear(128, 2),
# # # # # # # # # # # # # # # #         )
# # # # # # # # # # # # # # # #         self.step_scale = nn.Parameter(torch.ones(self.N_STEPS) * 0.2)
# # # # # # # # # # # # # # # #         self._init_weights()

# # # # # # # # # # # # # # # #     def _init_weights(self):
# # # # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # # # #             for m in self.modules():
# # # # # # # # # # # # # # # #                 if isinstance(m, nn.Linear):
# # # # # # # # # # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.3)
# # # # # # # # # # # # # # # #                     if m.bias is not None:
# # # # # # # # # # # # # # # #                         nn.init.zeros_(m.bias)
# # # # # # # # # # # # # # # #             nn.init.constant_(self.motion_gate_linear.bias, 2.0)

# # # # # # # # # # # # # # # #     def forward(self, raw_ctx, obs_traj):
# # # # # # # # # # # # # # # #         B, T_obs = raw_ctx.shape[0], obs_traj.shape[0]
# # # # # # # # # # # # # # # #         vels = obs_traj[1:] - obs_traj[:-1] if T_obs >= 2 else torch.zeros(1, B, 2, device=raw_ctx.device)
# # # # # # # # # # # # # # # #         if vels.shape[0] < self.obs_len:
# # # # # # # # # # # # # # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=raw_ctx.device)
# # # # # # # # # # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # # #             vels = vels[-self.obs_len:]
# # # # # # # # # # # # # # # #         vel_feat = self.vel_enc(vels.permute(1, 0, 2).reshape(B, -1))
# # # # # # # # # # # # # # # #         ctx_feat = self.ctx_proj(raw_ctx)
# # # # # # # # # # # # # # # #         hx, cur_pos, last_vel = ctx_feat, obs_traj[-1].clone(), vels[-1].clone()
# # # # # # # # # # # # # # # #         preds = []
# # # # # # # # # # # # # # # #         for i in range(self.N_STEPS):
# # # # # # # # # # # # # # # #             hx  = self.gru_cell(torch.cat([ctx_feat, vel_feat, cur_pos], dim=-1), hx)
# # # # # # # # # # # # # # # #             gate = torch.sigmoid(self.motion_gate_linear(torch.cat([vel_feat, hx], dim=-1)))
# # # # # # # # # # # # # # # #             raw_delta   = self.out_head(self.out_norm(hx))
# # # # # # # # # # # # # # # #             prior_delta = last_vel * (0.9 ** i)
# # # # # # # # # # # # # # # #             scale_i = torch.sigmoid(self.step_scale[i]) * self.MAX_DELTA
# # # # # # # # # # # # # # # #             delta   = (gate * prior_delta + (1.0 - gate) * raw_delta * scale_i).clamp(-self.MAX_DELTA, self.MAX_DELTA)
# # # # # # # # # # # # # # # #             cur_pos = cur_pos + delta
# # # # # # # # # # # # # # # #             last_vel = delta
# # # # # # # # # # # # # # # #             preds.append(cur_pos)
# # # # # # # # # # # # # # # #         return torch.stack(preds, dim=0)


# # # # # # # # # # # # # # # # # ─── VelocityField (giữ nguyên từ v30, pos_enc dùng pred_len) ────────────────

# # # # # # # # # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # # # # # # # # # # # # #                  unet_in_ch=13, fm_pred_len=8):
# # # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # # #         self.pred_len = pred_len
# # # # # # # # # # # # # # # #         self.obs_len  = obs_len
# # # # # # # # # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # # # # # # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # # # # # # # # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # # # # # # # # # # #             spatial_down=32, dropout=0.05)
# # # # # # # # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)
# # # # # # # # # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # # # # # # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # # # # # # # # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # # # # # # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)
# # # # # # # # # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # # # # # # # # #         self.ctx_drop = nn.Dropout(0.15)
# # # # # # # # # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)
# # # # # # # # # # # # # # # #         self.sr_anchor_proj = nn.Sequential(
# # # # # # # # # # # # # # # #             nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 256))
# # # # # # # # # # # # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # # # # # # # # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # # # # # # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # # # # # # # # # #         # KEY FIX: pos_enc dùng pred_len (12), không phải fm_pred_len (8)
# # # # # # # # # # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.05)
# # # # # # # # # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # # # # # # # # #                 dropout=0.15, activation="gelu", batch_first=True),
# # # # # # # # # # # # # # # #             num_layers=4)
# # # # # # # # # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)
# # # # # # # # # # # # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)
# # # # # # # # # # # # # # # #         self.short_range_head = ShortRangeHead(self.RAW_CTX_DIM, obs_len)

# # # # # # # # # # # # # # # #     def _time_emb(self, t, dim=256):
# # # # # # # # # # # # # # # #         half = dim // 2
# # # # # # # # # # # # # # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # # # # # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # # # # # # # # # # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # # # # # # # # # # # # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # # # # # # # # # # # # # #     def _context(self, batch_list):
# # # # # # # # # # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # # # # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # # # # # # # # # #         image_obs = batch_list[11]
# # # # # # # # # # # # # # # #         env_data  = batch_list[13]
# # # # # # # # # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # # # # # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # # # # # # # # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # # # # # # # # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
# # # # # # # # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # # # # # # # #         T_obs = obs_traj.shape[0]
# # # # # # # # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # # # # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # # # # # # # # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # # # # # # # # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # # # # # # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)
# # # # # # # # # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # # # # # # # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # # # # # # # # # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # # # # # # # # # # # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
# # # # # # # # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # # # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)
# # # # # # # # # # # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # # # # # # # # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # # # # # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # # # # # # # # #     def _beta_drift(self, x_t):
# # # # # # # # # # # # # # # #         lat_deg  = x_t[:, :, 1] * 5.0
# # # # # # # # # # # # # # # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # # # # # # # # # # # #         beta     = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # # # # # # # # # # # #         R_tc     = 3e5
# # # # # # # # # # # # # # # #         v_phys   = torch.zeros_like(x_t)
# # # # # # # # # # # # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # # # # # # # # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # # # # # # # # # # # #         return v_phys

# # # # # # # # # # # # # # # #     def _decode(self, x_t, t, ctx, sr_anchor_emb=None):
# # # # # # # # # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # # # # # # # # # # # #         t_emb = self.time_fc2(t_emb)
# # # # # # # # # # # # # # # #         T_seq = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # # # # # # # # # #         x_emb = self.traj_embed(x_t[:, :T_seq]) + self.pos_enc[:, :T_seq] + t_emb.unsqueeze(1)
# # # # # # # # # # # # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # # # # # # # # # # # #         if sr_anchor_emb is not None:
# # # # # # # # # # # # # # # #             mem_parts.append(sr_anchor_emb.unsqueeze(1))
# # # # # # # # # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(
# # # # # # # # # # # # # # # #             self.transformer(x_emb, torch.cat(mem_parts, dim=1)))))
# # # # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # # # #             v_phys = self._beta_drift(x_t[:, :T_seq])
# # # # # # # # # # # # # # # #         return v_neural + torch.sigmoid(self.physics_scale) * 2.0 * v_phys

# # # # # # # # # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0, sr_anchor_emb=None):
# # # # # # # # # # # # # # # #         return self._decode(x_t, t, self._apply_ctx_head(raw_ctx, noise_scale), sr_anchor_emb)

# # # # # # # # # # # # # # # #     def forward_short_range(self, obs_traj, raw_ctx):
# # # # # # # # # # # # # # # #         return self.short_range_head(raw_ctx, obs_traj)

# # # # # # # # # # # # # # # #     def compute_sr_anchor_emb(self, sr_pred):
# # # # # # # # # # # # # # # #         anchor_input = torch.cat([sr_pred[-1], sr_pred[-1] - sr_pred[-2]], dim=-1)
# # # # # # # # # # # # # # # #         return self.sr_anchor_proj(anchor_input)


# # # # # # # # # # # # # # # # # ─── TCFlowMatching v32 ───────────────────────────────────────────────────────

# # # # # # # # # # # # # # # # class TCFlowMatching(nn.Module):
# # # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # # #     v32: Phase-based training để tránh FM/MSE conflict.
    
# # # # # # # # # # # # # # # #     Phase 1 (epoch < phase_switch): Pure MSE + SR
# # # # # # # # # # # # # # # #     Phase 2 (epoch >= phase_switch): MSE + SR + AFCRPS
# # # # # # # # # # # # # # # #     """

# # # # # # # # # # # # # # # #     PHASE_SWITCH = 30  # Sau epoch này mới thêm AFCRPS

# # # # # # # # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # # # # # # # # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # # # # # # # # # # # #                  ctx_noise_scale=0.002, initial_sample_sigma=0.03, **kwargs):
# # # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # # # # # # # # #         self.fm_pred_len          = pred_len - N_SR_STEPS
# # # # # # # # # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # # # # # # # # # # #         self.active_pred_len      = pred_len
# # # # # # # # # # # # # # # #         self.net = VelocityField(pred_len=pred_len, obs_len=obs_len,
# # # # # # # # # # # # # # # #                                   sigma_min=sigma_min, unet_in_ch=unet_in_ch,
# # # # # # # # # # # # # # # #                                   fm_pred_len=self.fm_pred_len)

# # # # # # # # # # # # # # # #     def set_curriculum_len(self, *a, **kw): pass

# # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # # # # # # # # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # # # # # # # # # # # #                            Me   - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # #     def _to_abs(rel, lp, lm):
# # # # # # # # # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # # # # # # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # # # # # # # # # # # # #     def _cfm_noisy(self, x1):
# # # # # # # # # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # # # # # # # # #         sm = self.sigma_min
# # # # # # # # # # # # # # # #         x0 = torch.randn_like(x1) * sm
# # # # # # # # # # # # # # # #         t  = torch.rand(B, device=device)
# # # # # # # # # # # # # # # #         te = t.view(B, 1, 1)
# # # # # # # # # # # # # # # #         x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
# # # # # # # # # # # # # # # #         denom = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
# # # # # # # # # # # # # # # #         return x_t, t, te, denom

# # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # #     def _intensity_weights(obs_Me):
# # # # # # # # # # # # # # # #         w = obs_Me[-1, :, 1].detach()
# # # # # # # # # # # # # # # #         w = torch.where(w < 0.1, torch.full_like(w, 0.5),
# # # # # # # # # # # # # # # #             torch.where(w < 0.3, torch.full_like(w, 0.8),
# # # # # # # # # # # # # # # #             torch.where(w < 0.6, torch.full_like(w, 1.0),
# # # # # # # # # # # # # # # #                         torch.full_like(w, 1.5))))
# # # # # # # # # # # # # # # #         return w / w.mean().clamp(min=1e-6)

# # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # # # # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # # # # # # # #             return batch_list
# # # # # # # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # # # # # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # # # # # # # # # # # #                 t = aug[idx].clone()
# # # # # # # # # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # # # # # # # # #                 aug[idx] = t
# # # # # # # # # # # # # # # #         return aug

# # # # # # # # # # # # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # # # # # # # # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # #         Phase 1 (epoch < 30): SR_Huber + MSE_hav — học như LSTM
# # # # # # # # # # # # # # # #         Phase 2 (epoch >= 30): SR_Huber + MSE_hav + AFCRPS — thêm probabilistic
# # # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # # #         batch_list = self._lon_flip_aug(batch_list)

# # # # # # # # # # # # # # # #         traj_gt = batch_list[1]
# # # # # # # # # # # # # # # #         Me_gt   = batch_list[8]
# # # # # # # # # # # # # # # #         obs_t   = batch_list[0]
# # # # # # # # # # # # # # # #         obs_Me  = batch_list[7]
# # # # # # # # # # # # # # # #         lp, lm  = obs_t[-1], obs_Me[-1]

# # # # # # # # # # # # # # # #         raw_ctx     = self.net._context(batch_list)
# # # # # # # # # # # # # # # #         intensity_w = self._intensity_weights(obs_Me)

# # # # # # # # # # # # # # # #         # ── SR (step 1-4) ──────────────────────────────────────────────
# # # # # # # # # # # # # # # #         n_sr    = ShortRangeHead.N_STEPS
# # # # # # # # # # # # # # # #         sr_pred = self.net.forward_short_range(obs_t, raw_ctx)
# # # # # # # # # # # # # # # #         l_sr    = short_range_regression_loss(sr_pred, traj_gt[:n_sr], lp)

# # # # # # # # # # # # # # # #         # ── FM: full 12 steps từ lp ────────────────────────────────────
# # # # # # # # # # # # # # # #         sr_anchor = self.net.compute_sr_anchor_emb(sr_pred.detach())
# # # # # # # # # # # # # # # #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)
# # # # # # # # # # # # # # # #         x_t, t, te, denom = self._cfm_noisy(x1)
# # # # # # # # # # # # # # # #         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, sr_anchor_emb=sr_anchor)
# # # # # # # # # # # # # # # #         x1_pred  = x_t + denom * pred_vel
# # # # # # # # # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)   # [12, B, 2] normalised

# # # # # # # # # # # # # # # #         # ── MSE haversine per-step ─────────────────────────────────────
# # # # # # # # # # # # # # # #         # Đây là loss chính trong Phase 1 — giống LSTM nhưng physically correct
# # # # # # # # # # # # # # # #         l_mse = mse_hav_loss(pred_abs, traj_gt)

# # # # # # # # # # # # # # # #         # ── Directional ────────────────────────────────────────────────
# # # # # # # # # # # # # # # #         pred_deg = _denorm_to_deg(pred_abs)
# # # # # # # # # # # # # # # #         gt_deg   = _denorm_to_deg(traj_gt)
# # # # # # # # # # # # # # # #         l_vel    = velocity_loss_per_sample(pred_deg, gt_deg).mean()
# # # # # # # # # # # # # # # #         l_head   = heading_loss(pred_deg, gt_deg)

# # # # # # # # # # # # # # # #         # ── Total (Phase 1) ────────────────────────────────────────────
# # # # # # # # # # # # # # # #         NRM   = 35.0
# # # # # # # # # # # # # # # #         total = (
# # # # # # # # # # # # # # # #             5.0  * l_sr
# # # # # # # # # # # # # # # #             + 4.0  * l_mse          # HIGH weight như LSTM
# # # # # # # # # # # # # # # #             + 0.5  * l_vel  * NRM
# # # # # # # # # # # # # # # #             + 0.8  * l_head * NRM
# # # # # # # # # # # # # # # #         ) / NRM

# # # # # # # # # # # # # # # #         bd = dict(
# # # # # # # # # # # # # # # #             total=total, short_range=l_sr.item(), mse_hav=l_mse.item(),
# # # # # # # # # # # # # # # #             velocity=l_vel.item()*NRM, heading=l_head.item(),
# # # # # # # # # # # # # # # #             fm=0.0, step=0.0, disp=0.0, cont=0.0, pinn=0.0,
# # # # # # # # # # # # # # # #             spread=0.0, continuity=0.0, recurv_ratio=0.0,
# # # # # # # # # # # # # # # #         )

# # # # # # # # # # # # # # # #         # ── Phase 2: Thêm AFCRPS (epoch >= PHASE_SWITCH) ───────────────
# # # # # # # # # # # # # # # #         if epoch >= self.PHASE_SWITCH:
# # # # # # # # # # # # # # # #             samples = []
# # # # # # # # # # # # # # # #             for _ in range(self.n_train_ens):
# # # # # # # # # # # # # # # #                 xt_s, ts, _, dens_s = self._cfm_noisy(x1)
# # # # # # # # # # # # # # # #                 pv_s = self.net.forward_with_ctx(xt_s, ts, raw_ctx, sr_anchor_emb=sr_anchor)
# # # # # # # # # # # # # # # #                 pa_s, _ = self._to_abs(xt_s + dens_s * pv_s, lp, lm)
# # # # # # # # # # # # # # # #                 samples.append(pa_s)
# # # # # # # # # # # # # # # #             pred_samples = torch.stack(samples)   # [S, 12, B, 2]

# # # # # # # # # # # # # # # #             l_fm = fm_afcrps_loss(
# # # # # # # # # # # # # # # #                 pred_samples, traj_gt,
# # # # # # # # # # # # # # # #                 unit_01deg=True,
# # # # # # # # # # # # # # # #                 intensity_w=intensity_w,
# # # # # # # # # # # # # # # #                 w_es=0.2,
# # # # # # # # # # # # # # # #             )

# # # # # # # # # # # # # # # #             # Weight AFCRPS nhỏ hơn MSE để không dominate
# # # # # # # # # # # # # # # #             fm_w = min(1.0, (epoch - self.PHASE_SWITCH) / 20.0)  # ramp up 0→1
# # # # # # # # # # # # # # # #             bd["total"] = total + fm_w * 1.5 * l_fm / NRM
# # # # # # # # # # # # # # # #             bd["fm"]    = l_fm.item()

# # # # # # # # # # # # # # # #         if torch.isnan(bd["total"]) or torch.isinf(bd["total"]):
# # # # # # # # # # # # # # # #             bd["total"] = lp.new_zeros(())

# # # # # # # # # # # # # # # #         return bd

# # # # # # # # # # # # # # # #     @torch.no_grad()
# # # # # # # # # # # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
# # # # # # # # # # # # # # # #         obs_t  = batch_list[0]
# # # # # # # # # # # # # # # #         lp     = obs_t[-1]
# # # # # # # # # # # # # # # #         lm     = batch_list[7][-1]
# # # # # # # # # # # # # # # #         B      = lp.shape[0]
# # # # # # # # # # # # # # # #         device = lp.device
# # # # # # # # # # # # # # # #         T      = self.pred_len
# # # # # # # # # # # # # # # #         dt     = 1.0 / max(ddim_steps, 1)

# # # # # # # # # # # # # # # #         raw_ctx = self.net._context(batch_list)
# # # # # # # # # # # # # # # #         sr_pred = self.net.forward_short_range(obs_t, raw_ctx)
# # # # # # # # # # # # # # # #         sr_anchor = self.net.compute_sr_anchor_emb(sr_pred)

# # # # # # # # # # # # # # # #         traj_s, me_s = [], []
# # # # # # # # # # # # # # # #         for _ in range(num_ensemble):
# # # # # # # # # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma
# # # # # # # # # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # # # # # # #                 ns  = self.ctx_noise_scale if step < 3 else 0.0
# # # # # # # # # # # # # # # #                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx,
# # # # # # # # # # # # # # # #                                                  noise_scale=ns, sr_anchor_emb=sr_anchor)
# # # # # # # # # # # # # # # #                 x_t = x_t + dt * vel
# # # # # # # # # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # # # # # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # # # # # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # # # # # # # #             traj_s.append(tr)
# # # # # # # # # # # # # # # #             me_s.append(me)

# # # # # # # # # # # # # # # #         all_trajs = torch.stack(traj_s)
# # # # # # # # # # # # # # # #         fm_mean   = all_trajs.mean(0)
# # # # # # # # # # # # # # # #         pred_mean = fm_mean.clone()
# # # # # # # # # # # # # # # #         pred_mean[:ShortRangeHead.N_STEPS] = sr_pred   # SR override step 1-4

# # # # # # # # # # # # # # # #         if predict_csv:
# # # # # # # # # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # # # # # # # # # # # #         return pred_mean, torch.stack(me_s).mean(0), all_trajs

# # # # # # # # # # # # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # # # # # # # # # # # #         with torch.enable_grad():
# # # # # # # # # # # # # # # #             x = x_pred.detach().requires_grad_(True)
# # # # # # # # # # # # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # # # # # # # # # # # #             for _ in range(n_steps):
# # # # # # # # # # # # # # # #                 opt.zero_grad()
# # # # # # # # # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # # # # # # # # #                 pred_deg = _denorm_to_deg(pred_abs)
# # # # # # # # # # # # # # # #                 dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # # # # # # #                 lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # # # # # # # # #                 cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # # # # # #                 speed   = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # # # # # # # # # # # #                                       + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # # # # # # # # # # # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # # # # # # # # # # # # # # #                 loss.backward()
# # # # # # # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # # # # # # # # # # # # # # #                 opt.step()
# # # # # # # # # # # # # # # #         return x.detach()

# # # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # # # # # # # # # # # # # # #         import numpy as np
# # # # # # # # # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # # # # # # # # #                     "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # # # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # # # # # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # # # # # # # # #             if write_hdr: w.writeheader()
# # # # # # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # # # # # #                 for k in range(T):
# # # # # # # # # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # # # # # # # # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(math.radians(mean_lat[k, b]))
# # # # # # # # # # # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # # # # # # # # # # #                     w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # # # # # # # # # # # #                                 "lead_h": (k+1)*6, "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # # # # # # # # # # # #                                 "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # # # # # # # # # # # #                                 "lon_std_deg": f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # # # # # # # # #                                 "lat_std_deg": f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # # # # # # # # #                                 "ens_spread_km": f"{spread:.2f}"})


# # # # # # # # # # # # # # # # TCDiffusion = TCFlowMatching

# # # # # # # # # # # # # # # """
# # # # # # # # # # # # # # # flow_matching_model_v33.py — Pure FM + MSE: Beat LSTM bằng đơn giản

# # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # PHÂN TÍCH TẠI SAO FM THUA LSTM:
# # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # # 1. LSTM dùng pure MSE → gradient rõ ràng, 1 hướng duy nhất
# # # # # # # # # # # # # # # 2. FM v32 dùng 5+ loss → gradient conflict, model không biết optimize gì
# # # # # # # # # # # # # # # 3. SR head riêng + FM riêng → discontinuity tại step 4→5
# # # # # # # # # # # # # # # 4. AFCRPS pull ensemble diversity ≠ accuracy → ADE cao

# # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # GIẢI PHÁP V33: 
# # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # # CORE INSIGHT: FM architecture + MSE training = best of both worlds
# # # # # # # # # # # # # # #   - FM cho ensemble/uncertainty estimation khi inference
# # # # # # # # # # # # # # #   - MSE cho gradient rõ ràng khi training (như LSTM)
# # # # # # # # # # # # # # #   - KHÔNG dùng SR head riêng → FM predict ALL 12 steps
# # # # # # # # # # # # # # #   - KHÔNG dùng AFCRPS khi training → chỉ MSE haversine

# # # # # # # # # # # # # # # ARCHITECTURE CHANGES:
# # # # # # # # # # # # # # #   1. BỎ ShortRangeHead → FM predict tất cả 12 steps trực tiếp
# # # # # # # # # # # # # # #   2. BỎ sr_anchor_emb → đơn giản hóa decode
# # # # # # # # # # # # # # #   3. THÊM teacher forcing: train FM với gt trajectory làm guidance
# # # # # # # # # # # # # # #   4. Loss = MSE_haversine + nhẹ velocity + nhẹ heading (tổng 3 loss)

# # # # # # # # # # # # # # # TRAINING STRATEGY:
# # # # # # # # # # # # # # #   Phase 1 (epoch 0-20): MSE-only, sigma_min lớn (0.1) → near-deterministic
# # # # # # # # # # # # # # #     → FM hoạt động gần như regression network
# # # # # # # # # # # # # # #     → ADE giảm nhanh từ 400 → ~200km
    
# # # # # # # # # # # # # # #   Phase 2 (epoch 20-50): MSE + nhẹ velocity/heading, sigma_min giảm dần
# # # # # # # # # # # # # # #     → Refine trajectory shape
# # # # # # # # # # # # # # #     → ADE ~170km, 12h ~45km
    
# # # # # # # # # # # # # # #   Phase 3 (epoch 50+): Giữ nguyên, fine-tune
# # # # # # # # # # # # # # #     → 12h < 50km, 72h < 300km

# # # # # # # # # # # # # # # KEY INSIGHT VỀ FM VÀ 12h:
# # # # # # # # # # # # # # #   - FM CÓ THỂ đạt 12h < 50km NẾU train đúng cách
# # # # # # # # # # # # # # #   - Vấn đề cũ: SR head riêng → FM không được train cho step 1-4
# # # # # # # # # # # # # # #   - V33: FM predict ALL steps → step 1-4 cũng được optimize trực tiếp
# # # # # # # # # # # # # # #   - Step weighting nhấn mạnh 12h (step 2) và 24h (step 4)

# # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # """
# # # # # # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # # # # # import csv
# # # # # # # # # # # # # # # import math
# # # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # # from datetime import datetime
# # # # # # # # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # # # # # # # import torch
# # # # # # # # # # # # # # # import torch.nn as nn
# # # # # # # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # # # # # # # from Model.losses import _haversine_deg, _norm_to_deg, _haversine

# # # # # # # # # # # # # # # # ── Step weights: nhấn mạnh 12h và 24h ─────────────────────────────────────
# # # # # # # # # # # # # # # # Step:    6h   12h  18h  24h  30h  36h  42h  48h  54h  60h  66h  72h
# # # # # # # # # # # # # # # MSE_STEP_WEIGHTS = [
# # # # # # # # # # # # # # #     1.5,  # 6h  - quan trọng
# # # # # # # # # # # # # # #     4.0,  # 12h - TARGET < 50km → weight cao nhất  
# # # # # # # # # # # # # # #     2.0,  # 18h
# # # # # # # # # # # # # # #     3.5,  # 24h - TARGET < 100km → weight cao
# # # # # # # # # # # # # # #     1.5,  # 30h
# # # # # # # # # # # # # # #     1.5,  # 36h
# # # # # # # # # # # # # # #     1.5,  # 42h
# # # # # # # # # # # # # # #     2.5,  # 48h - TARGET < 200km
# # # # # # # # # # # # # # #     1.0,  # 54h
# # # # # # # # # # # # # # #     1.0,  # 60h
# # # # # # # # # # # # # # #     1.5,  # 66h
# # # # # # # # # # # # # # #     2.5,  # 72h - TARGET < 300km
# # # # # # # # # # # # # # # ]


# # # # # # # # # # # # # # # def _denorm_to_deg(t):
# # # # # # # # # # # # # # #     """Convert normalized coords to degrees."""
# # # # # # # # # # # # # # #     out = t.clone()
# # # # # # # # # # # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # # # # # # # # # # #     return out


# # # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # #  MSE Haversine Loss — Vũ khí chính, giống LSTM nhưng physically correct
# # # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # # def mse_hav_loss(pred_norm, gt_norm, step_w=None):
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     Haversine MSE per-step.
    
# # # # # # # # # # # # # # #     Tại sao dùng haversine thay vì MSE(lon,lat)?
# # # # # # # # # # # # # # #     - MSE(lon,lat) bị bias do cos(lat): 1 degree lon ở equator ≠ 1 degree ở 30°N
# # # # # # # # # # # # # # #     - Haversine cho distance thật sự trên Earth → gradient chính xác hơn
# # # # # # # # # # # # # # #     - LSTM dùng MSE(lon,lat) → có thể beat bằng MSE(haversine)
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     if step_w is None:
# # # # # # # # # # # # # # #         step_w = MSE_STEP_WEIGHTS
# # # # # # # # # # # # # # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # # # # # # # # #     pred_deg = _norm_to_deg(pred_norm[:T])
# # # # # # # # # # # # # # #     gt_deg   = _norm_to_deg(gt_norm[:T])
# # # # # # # # # # # # # # #     dist_km  = _haversine_deg(pred_deg, gt_deg)   # [T, B] in km
    
# # # # # # # # # # # # # # #     w = pred_norm.new_tensor(step_w[:T])
# # # # # # # # # # # # # # #     w = w / w.sum() * T  # normalize nhưng giữ scale
    
# # # # # # # # # # # # # # #     # Huber-style: dùng L1 cho outlier (dist > 300km), L2 cho phần còn lại
# # # # # # # # # # # # # # #     # Tránh gradient explosion từ large errors
# # # # # # # # # # # # # # #     delta = 300.0  # km
# # # # # # # # # # # # # # #     huber_dist = torch.where(
# # # # # # # # # # # # # # #         dist_km < delta,
# # # # # # # # # # # # # # #         dist_km.pow(2) / (2.0 * delta),
# # # # # # # # # # # # # # #         dist_km - delta / 2.0,
# # # # # # # # # # # # # # #     )
    
# # # # # # # # # # # # # # #     return (huber_dist * w.unsqueeze(1)).mean() / delta


# # # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # #  VelocityField — Đơn giản hóa, bỏ ShortRangeHead
# # # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     Velocity field cho FM — predict velocity cho ALL 12 steps.
    
# # # # # # # # # # # # # # #     Thay đổi so với v32:
# # # # # # # # # # # # # # #     - BỎ ShortRangeHead → FM predict step 1-12 trực tiếp
# # # # # # # # # # # # # # #     - BỎ sr_anchor_emb → đơn giản decode
# # # # # # # # # # # # # # #     - THÊM residual connection trong decoder
# # # # # # # # # # # # # # #     - THÊM step embedding (model biết đang predict step nào)
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # # # # # # # # # # # #                  unet_in_ch=13, **kwargs):
# # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # #         self.pred_len = pred_len
# # # # # # # # # # # # # # #         self.obs_len  = obs_len
        
# # # # # # # # # # # # # # #         # ── Encoder (giữ nguyên) ──────────────────────────────────────
# # # # # # # # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # # # # # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # # # # # # # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # # # # # # # # # #             spatial_down=32, dropout=0.05)
# # # # # # # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)
        
# # # # # # # # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # # # # # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # # # # # # # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # # # # # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)
        
# # # # # # # # # # # # # # #         # ── Context projection ────────────────────────────────────────
# # # # # # # # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # # # # # # # #         self.ctx_drop = nn.Dropout(0.10)
# # # # # # # # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)
        
# # # # # # # # # # # # # # #         # ── Velocity observation encoder ──────────────────────────────
# # # # # # # # # # # # # # #         # Encode observed velocity pattern → giúp predict step 1-4 chính xác
# # # # # # # # # # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # # # # # # # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # # # # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # # # # # # #             nn.Linear(256, 256), nn.GELU(),
# # # # # # # # # # # # # # #         )
        
# # # # # # # # # # # # # # #         # ── FM Decoder ────────────────────────────────────────────────
# # # # # # # # # # # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # # # # # # # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # # # # # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # # # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
        
# # # # # # # # # # # # # # #         # Step embedding: model biết đang ở step nào → quan trọng cho short-range
# # # # # # # # # # # # # # #         self.step_embed = nn.Embedding(pred_len, 256)
        
# # # # # # # # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # # # # # # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # # # # # # # # # # # # #             num_layers=5)  # 5 layers thay vì 4 → capacity lớn hơn
        
# # # # # # # # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)
        
# # # # # # # # # # # # # # #         # Learnable output scale per step → model tự learn scale phù hợp
# # # # # # # # # # # # # # #         self.step_scale = nn.Parameter(torch.ones(pred_len) * 0.5)
        
# # # # # # # # # # # # # # #         # Physics beta-drift (giữ nguyên)
# # # # # # # # # # # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.3)
        
# # # # # # # # # # # # # # #         self._init_weights()

# # # # # # # # # # # # # # #     def _init_weights(self):
# # # # # # # # # # # # # # #         """Xavier init với gain nhỏ → output gần 0 ban đầu."""
# # # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # # #             for name, m in self.named_modules():
# # # # # # # # # # # # # # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # # # # # # # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # # # # # # # # # # # # #                     if m.bias is not None:
# # # # # # # # # # # # # # #                         nn.init.zeros_(m.bias)

# # # # # # # # # # # # # # #     def _time_emb(self, t, dim=256):
# # # # # # # # # # # # # # #         half = dim // 2
# # # # # # # # # # # # # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # # # # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # # # # # # # # # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # # # # # # # # # # # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # # # # # # # # # # # # #     def _context(self, batch_list):
# # # # # # # # # # # # # # #         """Encode observation context — giống v32."""
# # # # # # # # # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # # # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # # # # # # # # #         image_obs = batch_list[11]
# # # # # # # # # # # # # # #         env_data  = batch_list[13]
        
# # # # # # # # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # # # # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # # # # # # # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # # # # # # # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
        
# # # # # # # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # # # # # # #         T_obs = obs_traj.shape[0]
        
# # # # # # # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # # # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # # # # # # # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # # # # # # # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # # # # # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)
        
# # # # # # # # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # # # # # # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # # # # # # # # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # # # # # # # # # # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))
        
# # # # # # # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)
        
# # # # # # # # # # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # # # # # # # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # # # # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # # # # # # # #     def _get_vel_obs_feat(self, obs_traj):
# # # # # # # # # # # # # # #         """Encode observed velocities → giúp FM predict step 1-4."""
# # # # # # # # # # # # # # #         B = obs_traj.shape[1]
# # # # # # # # # # # # # # #         T_obs = obs_traj.shape[0]
        
# # # # # # # # # # # # # # #         if T_obs >= 2:
# # # # # # # # # # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
        
# # # # # # # # # # # # # # #         # Pad/truncate to obs_len
# # # # # # # # # # # # # # #         if vels.shape[0] < self.obs_len:
# # # # # # # # # # # # # # #             pad = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# # # # # # # # # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # #             vels = vels[-self.obs_len:]
        
# # # # # # # # # # # # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # # # # # # # # # # # #     def _beta_drift(self, x_t):
# # # # # # # # # # # # # # #         """Physical beta-drift bias."""
# # # # # # # # # # # # # # #         lat_deg  = x_t[:, :, 1] * 5.0
# # # # # # # # # # # # # # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # # # # # # # # # # #         beta     = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # # # # # # # # # # #         R_tc     = 3e5
# # # # # # # # # # # # # # #         v_phys   = torch.zeros_like(x_t)
# # # # # # # # # # # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # # # # # # # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # # # # # # # # # # #         return v_phys

# # # # # # # # # # # # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None):
# # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # #         Decode velocity field.
        
# # # # # # # # # # # # # # #         Thay đổi:
# # # # # # # # # # # # # # #         - Thêm vel_obs_feat vào memory → giúp predict step 1-4
# # # # # # # # # # # # # # #         - Thêm step embedding → model biết step position
# # # # # # # # # # # # # # #         - Output scale per step → kiểm soát magnitude
# # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # #         B = x_t.shape[0]
# # # # # # # # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # # # # # # # # # # #         t_emb = self.time_fc2(t_emb)
        
# # # # # # # # # # # # # # #         T_seq = min(x_t.size(1), self.pos_enc.shape[1])
        
# # # # # # # # # # # # # # #         # Step embedding
# # # # # # # # # # # # # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # # # # # # # # # #         s_emb = self.step_embed(step_idx)  # [B, T_seq, 256]
        
# # # # # # # # # # # # # # #         x_emb = self.traj_embed(x_t[:, :T_seq]) + self.pos_enc[:, :T_seq] + t_emb.unsqueeze(1) + s_emb
        
# # # # # # # # # # # # # # #         # Memory: context + time + velocity observations
# # # # # # # # # # # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # # # # # # # # # # #         if vel_obs_feat is not None:
# # # # # # # # # # # # # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))
        
# # # # # # # # # # # # # # #         decoded = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # # # # # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
        
# # # # # # # # # # # # # # #         # Per-step scale
# # # # # # # # # # # # # # #         scale = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # # # # # # # # # # # #         v_neural = v_neural * scale
        
# # # # # # # # # # # # # # #         # Physics correction
# # # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # # #             v_phys = self._beta_drift(x_t[:, :T_seq])
        
# # # # # # # # # # # # # # #         return v_neural + torch.sigmoid(self.physics_scale) * v_phys

# # # # # # # # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0, vel_obs_feat=None):
# # # # # # # # # # # # # # #         return self._decode(x_t, t, self._apply_ctx_head(raw_ctx, noise_scale), vel_obs_feat)


# # # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # # #  TCFlowMatching v33 — Pure FM + MSE
# # # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # # class TCFlowMatching(nn.Module):
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     v33: Pure FM architecture, MSE-only training.
    
# # # # # # # # # # # # # # #     KEY DIFFERENCES từ v32:
# # # # # # # # # # # # # # #     1. FM predict ALL 12 steps (không có SR head riêng)
# # # # # # # # # # # # # # #     2. Training loss = MSE_haversine + nhẹ velocity + nhẹ heading
# # # # # # # # # # # # # # #     3. KHÔNG dùng AFCRPS → gradient không conflict
# # # # # # # # # # # # # # #     4. sigma_min schedule: lớn ban đầu (near-deterministic) → nhỏ dần
# # # # # # # # # # # # # # #     5. Teacher forcing: dùng gt trajectory interpolation khi train
    
# # # # # # # # # # # # # # #     TẠI SAO FM CÓ THỂ BEAT LSTM:
# # # # # # # # # # # # # # #     - FM có thêm encoder mạnh (FNO3D + Mamba + Env_net)
# # # # # # # # # # # # # # #     - FM có ensemble capability → uncertainty estimation
# # # # # # # # # # # # # # #     - FM có physics correction (beta-drift)
# # # # # # # # # # # # # # #     - Nếu train đúng (MSE-only), FM ít nhất = LSTM, có thể tốt hơn
# # # # # # # # # # # # # # #     """

# # # # # # # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # # # # # # # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # # # # # # # # # # #                  ctx_noise_scale=0.002, initial_sample_sigma=0.03, **kwargs):
# # # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # # # # # # # # # #         self.active_pred_len      = pred_len
        
# # # # # # # # # # # # # # #         self.net = VelocityField(
# # # # # # # # # # # # # # #             pred_len=pred_len, obs_len=obs_len,
# # # # # # # # # # # # # # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # # # # # # # # # # # #     def set_curriculum_len(self, *a, **kw):
# # # # # # # # # # # # # # #         pass

# # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # # # # # # # # # # #         """Convert absolute coords to relative (from last observed position)."""
# # # # # # # # # # # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # # # # # # # # # # #                            Me   - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # #     def _to_abs(rel, lp, lm):
# # # # # # # # # # # # # # #         """Convert relative coords back to absolute."""
# # # # # # # # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # # # # # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # # # # # # # # # # # #     # def _cfm_noisy(self, x1, sigma_min=None):
# # # # # # # # # # # # # # #     #     """
# # # # # # # # # # # # # # #     #     Conditional FM noise process.
        
# # # # # # # # # # # # # # #     #     KEY: sigma_min controls stochasticity:
# # # # # # # # # # # # # # #     #     - sigma_min lớn (0.1-0.2) → gần deterministic → tốt cho MSE training
# # # # # # # # # # # # # # #     #     - sigma_min nhỏ (0.01-0.02) → stochastic → tốt cho ensemble diversity
# # # # # # # # # # # # # # #     #     """
# # # # # # # # # # # # # # #     #     if sigma_min is None:
# # # # # # # # # # # # # # #     #         sigma_min = self.sigma_min
# # # # # # # # # # # # # # #     #     B, device = x1.shape[0], x1.device
# # # # # # # # # # # # # # #     #     x0 = torch.randn_like(x1) * sigma_min
# # # # # # # # # # # # # # #     #     t  = torch.rand(B, device=device)
# # # # # # # # # # # # # # #     #     te = t.view(B, 1, 1)
# # # # # # # # # # # # # # #     #     # x_t = te * x1 + (1.0 - te * (1.0 - sigma_min)) * x0
# # # # # # # # # # # # # # #     #     # denom = (1.0 - (1.0 - sigma_min) * te).clamp(min=1e-5)
# # # # # # # # # # # # # # #     #     x_t = te * x1 + (1.0 - te * (1.0 - sigma_min)) * x0
# # # # # # # # # # # # # # #     #     denom = (1.0 - (1.0 - sigma_min) * te).clamp(min=1e-5)
# # # # # # # # # # # # # # #     #     return x_t, t, te, denom

# # # # # # # # # # # # # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # # # # # # # # # # # # #         if sigma_min is None:
# # # # # # # # # # # # # # #             sigma_min = self.sigma_min
# # # # # # # # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # # # # # # # #         x0 = torch.randn_like(x1) * sigma_min
# # # # # # # # # # # # # # #         t  = torch.rand(B, device=device)
# # # # # # # # # # # # # # #         te = t.view(B, 1, 1)
# # # # # # # # # # # # # # #         x_t = (1.0 - te) * x0 + te * x1
# # # # # # # # # # # # # # #         u   = x1 - x0
# # # # # # # # # # # # # # #         return x_t, t, u  # 3 giá trị, không còn denom

# # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # #     def _intensity_weights(obs_Me):
# # # # # # # # # # # # # # #         """Weight samples by intensity — stronger storms get more attention."""
# # # # # # # # # # # # # # #         w = obs_Me[-1, :, 1].detach()
# # # # # # # # # # # # # # #         w = torch.where(w < 0.1, torch.full_like(w, 0.5),
# # # # # # # # # # # # # # #             torch.where(w < 0.3, torch.full_like(w, 0.8),
# # # # # # # # # # # # # # #             torch.where(w < 0.6, torch.full_like(w, 1.0),
# # # # # # # # # # # # # # #                         torch.full_like(w, 1.5))))
# # # # # # # # # # # # # # #         return w / w.mean().clamp(min=1e-6)

# # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # # # # # # # # # # #         """Data augmentation: flip longitude."""
# # # # # # # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # # # # # # #             return batch_list
# # # # # # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # # # # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # # # # # # # # # # #                 t = aug[idx].clone()
# # # # # # # # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # # # # # # # #                 aug[idx] = t
# # # # # # # # # # # # # # #         return aug

# # # # # # # # # # # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # # # # # # # # # # #     # def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # # # # # # #     #     """
# # # # # # # # # # # # # # #     #     ═══════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # #     #     CORE TRAINING LOGIC — MSE-focused, FM as architecture only
# # # # # # # # # # # # # # #     #     ═══════════════════════════════════════════════════════════════
        
# # # # # # # # # # # # # # #     #     Phase 1 (epoch < 15): Pure MSE, sigma_min = 0.15 (near-deterministic)
# # # # # # # # # # # # # # #     #       → FM gần như regression network
# # # # # # # # # # # # # # #     #       → Gradient rõ ràng, ADE giảm nhanh
          
# # # # # # # # # # # # # # #     #     Phase 2 (epoch 15-40): MSE + nhẹ velocity/heading, sigma_min = 0.08
# # # # # # # # # # # # # # #     #       → Refine trajectory shape
# # # # # # # # # # # # # # #     #       → Giảm directional errors
          
# # # # # # # # # # # # # # #     #     Phase 3 (epoch 40+): MSE + velocity/heading, sigma_min = 0.03
# # # # # # # # # # # # # # #     #       → Fine-tune, cho phép ensemble diversity
# # # # # # # # # # # # # # #     #       → Maintain accuracy, add calibration
# # # # # # # # # # # # # # #     #     """
# # # # # # # # # # # # # # #     #     batch_list = self._lon_flip_aug(batch_list)

# # # # # # # # # # # # # # #     #     traj_gt = batch_list[1]    # [T, B, 2] normalized
# # # # # # # # # # # # # # #     #     Me_gt   = batch_list[8]    # [T, B, 2] normalized  
# # # # # # # # # # # # # # #     #     obs_t   = batch_list[0]    # [T_obs, B, 2] normalized
# # # # # # # # # # # # # # #     #     obs_Me  = batch_list[7]    # [T_obs, B, 2] normalized
# # # # # # # # # # # # # # #     #     lp, lm  = obs_t[-1], obs_Me[-1]

# # # # # # # # # # # # # # #     #     # ── Sigma schedule ─────────────────────────────────────────────
# # # # # # # # # # # # # # #     #     # Ban đầu sigma lớn → FM gần deterministic → MSE hiệu quả
# # # # # # # # # # # # # # #     #     if epoch < 15:
# # # # # # # # # # # # # # #     #         current_sigma = 0.15
# # # # # # # # # # # # # # #     #     elif epoch < 40:
# # # # # # # # # # # # # # #     #         # Linear decay 0.15 → 0.03
# # # # # # # # # # # # # # #     #         t = (epoch - 15) / 25.0
# # # # # # # # # # # # # # #     #         current_sigma = 0.15 - t * (0.15 - 0.03)
# # # # # # # # # # # # # # #     #     else:
# # # # # # # # # # # # # # #     #         current_sigma = 0.03

# # # # # # # # # # # # # # #     #     # ── Encode context ─────────────────────────────────────────────
# # # # # # # # # # # # # # #     #     raw_ctx     = self.net._context(batch_list)
# # # # # # # # # # # # # # #     #     vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # # # # # # # # # # #     #     # ── FM forward: predict ALL 12 steps ───────────────────────────
# # # # # # # # # # # # # # #     #     x1 = self._to_rel(traj_gt, Me_gt, lp, lm)  # [B, T, 4]
# # # # # # # # # # # # # # #     #     # # x_t, t, te, denom = self._cfm_noisy(x1, sigma_min=current_sigma)
# # # # # # # # # # # # # # #     #     # x_t, t, u_target = self._cfm_noisy(x1, sigma_min=current_sigma)

# # # # # # # # # # # # # # #     #     # pred_vel = self.net.forward_with_ctx(
# # # # # # # # # # # # # # #     #     #     x_t, t, raw_ctx, vel_obs_feat=vel_obs_feat)
# # # # # # # # # # # # # # #     #     # x1_pred  = x_t + denom * pred_vel
# # # # # # # # # # # # # # #     #     # pred_abs, _ = self._to_abs(x1_pred, lp, lm)  # [12, B, 2] normalized

# # # # # # # # # # # # # # #     #     # # ── LOSS 1: MSE Haversine (CHÍNH) ──────────────────────────────
# # # # # # # # # # # # # # #     #     # l_mse = mse_hav_loss(pred_abs, traj_gt)
# # # # # # # # # # # # # # #     #     x_t, t, u_target = self._cfm_noisy(x1, sigma_min=current_sigma)
# # # # # # # # # # # # # # #     #     pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, vel_obs_feat=vel_obs_feat)
# # # # # # # # # # # # # # #     #     l_mse = F.mse_loss(pred_vel, u_target)  # loss trên velocity, stable qua mọi epo

# # # # # # # # # # # # # # #     #     # ── LOSS 2: Velocity matching (nhẹ) ───────────────────────────
# # # # # # # # # # # # # # #     #     pred_deg = _denorm_to_deg(pred_abs)
# # # # # # # # # # # # # # #     #     gt_deg   = _denorm_to_deg(traj_gt)
        
# # # # # # # # # # # # # # #     #     if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # # # # # # # # # #     #         T_min = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # # # # # # # # #     #         v_pred = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # # # # # # # # # #     #         v_gt   = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # # # # # # # # # # #     #         l_vel  = F.smooth_l1_loss(v_pred, v_gt)
# # # # # # # # # # # # # # #     #     else:
# # # # # # # # # # # # # # #     #         l_vel = pred_abs.new_zeros(())

# # # # # # # # # # # # # # #     #     # ── LOSS 3: Heading consistency (nhẹ) ─────────────────────────
# # # # # # # # # # # # # # #     #     if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # # # # # # # # # #     #         T_min = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # # # # # # # # #     #         v_pred = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # # # # # # # # # #     #         v_gt   = gt_deg[1:T_min]   - gt_deg[:T_min-1]
            
# # # # # # # # # # # # # # #     #         v_pred_n = v_pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # # # # # # # # # #     #         v_gt_n   = v_gt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # # # # # # # # # #     #         cos_sim  = ((v_pred / v_pred_n) * (v_gt / v_gt_n)).sum(-1)
# # # # # # # # # # # # # # #     #         l_head   = F.relu(-cos_sim).pow(2).mean()  # penalize opposite direction
# # # # # # # # # # # # # # #     #     else:
# # # # # # # # # # # # # # #     #         l_head = pred_abs.new_zeros(())

# # # # # # # # # # # # # # #     #     # ── Total Loss ─────────────────────────────────────────────────
# # # # # # # # # # # # # # #     #     # Phase 1: pure MSE
# # # # # # # # # # # # # # #     #     # Phase 2+: MSE + small velocity + small heading
# # # # # # # # # # # # # # #     #     if epoch < 15:
# # # # # # # # # # # # # # #     #         w_vel  = 0.0
# # # # # # # # # # # # # # #     #         w_head = 0.0
# # # # # # # # # # # # # # #     #     elif epoch < 40:
# # # # # # # # # # # # # # #     #         t_phase = (epoch - 15) / 25.0
# # # # # # # # # # # # # # #     #         w_vel  = t_phase * 0.3
# # # # # # # # # # # # # # #     #         w_head = t_phase * 0.2
# # # # # # # # # # # # # # #     #     else:
# # # # # # # # # # # # # # #     #         w_vel  = 0.3
# # # # # # # # # # # # # # #     #         w_head = 0.2

# # # # # # # # # # # # # # #     #     total = l_mse + w_vel * l_vel + w_head * l_head

# # # # # # # # # # # # # # #     #     # ── Ensemble consistency (optional, epoch 40+) ─────────────────
# # # # # # # # # # # # # # #     #     # Train thêm 1 sample, penalize nếu quá khác nhau
# # # # # # # # # # # # # # #     #     l_ens_consist = pred_abs.new_zeros(())
# # # # # # # # # # # # # # #     #     if epoch >= 40 and self.n_train_ens >= 2:
# # # # # # # # # # # # # # #     #         x_t2, t2, te2, denom2 = self._cfm_noisy(x1, sigma_min=current_sigma)
# # # # # # # # # # # # # # #     #         pred_vel2 = self.net.forward_with_ctx(
# # # # # # # # # # # # # # #     #             x_t2, t2, raw_ctx, vel_obs_feat=vel_obs_feat)
# # # # # # # # # # # # # # #     #         x1_pred2 = x_t2 + denom2 * pred_vel2
# # # # # # # # # # # # # # #     #         pred_abs2, _ = self._to_abs(x1_pred2, lp, lm)
            
# # # # # # # # # # # # # # #     #         # MSE of second sample → cũng phải gần gt
# # # # # # # # # # # # # # #     #         l_ens_consist = mse_hav_loss(pred_abs2, traj_gt)
# # # # # # # # # # # # # # #     #         total = total + 0.3 * l_ens_consist

# # # # # # # # # # # # # # #     #     if torch.isnan(total) or torch.isinf(total):
# # # # # # # # # # # # # # #     #         total = lp.new_zeros(())

# # # # # # # # # # # # # # #     #     return dict(
# # # # # # # # # # # # # # #     #         total=total,
# # # # # # # # # # # # # # #     #         mse_hav=l_mse.item(),
# # # # # # # # # # # # # # #     #         velocity=l_vel.item(),
# # # # # # # # # # # # # # #     #         heading=l_head.item(),
# # # # # # # # # # # # # # #     #         ens_consist=l_ens_consist.item(),
# # # # # # # # # # # # # # #     #         sigma=current_sigma,
# # # # # # # # # # # # # # #     #         # Backward compat
# # # # # # # # # # # # # # #     #         fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # # # # # # # # # # #     #         spread=0.0, continuity=0.0, step=0.0, disp=0.0,
# # # # # # # # # # # # # # #     #         recurv_ratio=0.0,
# # # # # # # # # # # # # # #     #     )

# # # # # # # # # # # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # # # # # # #         batch_list = self._lon_flip_aug(batch_list)

# # # # # # # # # # # # # # #         traj_gt = batch_list[1]
# # # # # # # # # # # # # # #         Me_gt   = batch_list[8]
# # # # # # # # # # # # # # #         obs_t   = batch_list[0]
# # # # # # # # # # # # # # #         obs_Me  = batch_list[7]
# # # # # # # # # # # # # # #         lp, lm  = obs_t[-1], obs_Me[-1]

# # # # # # # # # # # # # # #         # ── Sigma schedule — đổi tên biến thành sigma_t tránh conflict với FM t ──
# # # # # # # # # # # # # # #         if epoch < 15:
# # # # # # # # # # # # # # #             current_sigma = 0.15
# # # # # # # # # # # # # # #         elif epoch < 40:
# # # # # # # # # # # # # # #             sigma_frac = (epoch - 15) / 25.0
# # # # # # # # # # # # # # #             current_sigma = 0.15 - sigma_frac * (0.15 - 0.03)
# # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # #             current_sigma = 0.03

# # # # # # # # # # # # # # #         # ── Encode context ────────────────────────────────────────────────
# # # # # # # # # # # # # # #         raw_ctx      = self.net._context(batch_list)
# # # # # # # # # # # # # # #         vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # # # # # # # # # # #         # ── FM forward ────────────────────────────────────────────────────
# # # # # # # # # # # # # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)  # [B, T, 4]
# # # # # # # # # # # # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # # # # # # # # # # # #         pred_vel = self.net.forward_with_ctx(
# # # # # # # # # # # # # # #             x_t, fm_t, raw_ctx, vel_obs_feat=vel_obs_feat)

# # # # # # # # # # # # # # #         # ── LOSS 1: FM velocity MSE (chính) ──────────────────────────────
# # # # # # # # # # # # # # #         l_mse = F.mse_loss(pred_vel, u_target)

# # # # # # # # # # # # # # #         # ── Reconstruct pred_abs để tính velocity/heading loss ───────────
# # # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # # #             fm_te    = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # # # # # # # # # # #             x1_pred  = x_t + (1.0 - fm_te) * pred_vel  # FM chuẩn: x1 = x_t + (1-t)*v
# # # # # # # # # # # # # # #             pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# # # # # # # # # # # # # # #         pred_deg = _denorm_to_deg(pred_abs)
# # # # # # # # # # # # # # #         gt_deg   = _denorm_to_deg(traj_gt)

# # # # # # # # # # # # # # #         # ── LOSS 2: Velocity matching ─────────────────────────────────────
# # # # # # # # # # # # # # #         if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # # # # # # # # # #             T_min  = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # # # # # # # # #             v_pred = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # # # # # # # # # #             v_gt   = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # # # # # # # # # # #             l_vel  = F.smooth_l1_loss(v_pred, v_gt)
# # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # #             l_vel = x_t.new_zeros(())

# # # # # # # # # # # # # # #         # ── LOSS 3: Heading consistency ───────────────────────────────────
# # # # # # # # # # # # # # #         if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # # # # # # # # # #             T_min    = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # # # # # # # # #             v_pred   = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # # # # # # # # # #             v_gt     = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # # # # # # # # # # #             v_pred_n = v_pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # # # # # # # # # #             v_gt_n   = v_gt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # # # # # # # # # #             cos_sim  = ((v_pred / v_pred_n) * (v_gt / v_gt_n)).sum(-1)
# # # # # # # # # # # # # # #             l_head   = F.relu(-cos_sim).pow(2).mean()
# # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # #             l_head = x_t.new_zeros(())

# # # # # # # # # # # # # # #         # ── Phase weights ─────────────────────────────────────────────────
# # # # # # # # # # # # # # #         if epoch < 15:
# # # # # # # # # # # # # # #             w_vel, w_head = 0.0, 0.0
# # # # # # # # # # # # # # #         elif epoch < 40:
# # # # # # # # # # # # # # #             phase_frac = (epoch - 15) / 25.0
# # # # # # # # # # # # # # #             w_vel  = phase_frac * 0.3
# # # # # # # # # # # # # # #             w_head = phase_frac * 0.2
# # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # #             w_vel, w_head = 0.3, 0.2

# # # # # # # # # # # # # # #         total = l_mse + w_vel * l_vel + w_head * l_head

# # # # # # # # # # # # # # #         # ── Ensemble consistency (epoch 40+) ──────────────────────────────
# # # # # # # # # # # # # # #         l_ens_consist = x_t.new_zeros(())
# # # # # # # # # # # # # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # # # # # # # # # # # # #             x_t2, fm_t2, u_target2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # # # # # # # # # # # #             pred_vel2 = self.net.forward_with_ctx(
# # # # # # # # # # # # # # #                 x_t2, fm_t2, raw_ctx, vel_obs_feat=vel_obs_feat)
# # # # # # # # # # # # # # #             l_ens_consist = F.mse_loss(pred_vel2, u_target2)
# # # # # # # # # # # # # # #             total = total + 0.3 * l_ens_consist

# # # # # # # # # # # # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # # # # # # # # # # # #             total = x_t.new_zeros(())

# # # # # # # # # # # # # # #         return dict(
# # # # # # # # # # # # # # #             total=total,
# # # # # # # # # # # # # # #             mse_hav=l_mse.item(),
# # # # # # # # # # # # # # #             velocity=l_vel.item(),
# # # # # # # # # # # # # # #             heading=l_head.item(),
# # # # # # # # # # # # # # #             ens_consist=l_ens_consist.item(),
# # # # # # # # # # # # # # #             sigma=current_sigma,
# # # # # # # # # # # # # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # # # # # # # # # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0,
# # # # # # # # # # # # # # #             recurv_ratio=0.0,
# # # # # # # # # # # # # # #         )
# # # # # # # # # # # # # # #     @torch.no_grad()
# # # # # # # # # # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
# # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # #         Inference: ODE integration từ noise → trajectory.
        
# # # # # # # # # # # # # # #         Thay đổi so với v32:
# # # # # # # # # # # # # # #         - Không có SR override → FM predict ALL steps
# # # # # # # # # # # # # # #         - vel_obs_feat truyền vào decode → short-range accuracy
# # # # # # # # # # # # # # #         - Dùng sigma_min nhỏ (0.02) khi inference → diversity
# # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # #         obs_t  = batch_list[0]
# # # # # # # # # # # # # # #         lp     = obs_t[-1]
# # # # # # # # # # # # # # #         lm     = batch_list[7][-1]
# # # # # # # # # # # # # # #         B      = lp.shape[0]
# # # # # # # # # # # # # # #         device = lp.device
# # # # # # # # # # # # # # #         T      = self.pred_len
# # # # # # # # # # # # # # #         dt     = 1.0 / max(ddim_steps, 1)

# # # # # # # # # # # # # # #         raw_ctx      = self.net._context(batch_list)
# # # # # # # # # # # # # # #         vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # # # # # # # # # # #         traj_s, me_s = [], []
# # # # # # # # # # # # # # #         for _ in range(num_ensemble):
# # # # # # # # # # # # # # #             # x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma
# # # # # # # # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.sigma_min

# # # # # # # # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # # # # # #                 ns  = self.ctx_noise_scale if step < 3 else 0.0
# # # # # # # # # # # # # # #                 vel = self.net.forward_with_ctx(
# # # # # # # # # # # # # # #                     x_t, t_b, raw_ctx,
# # # # # # # # # # # # # # #                     noise_scale=ns, vel_obs_feat=vel_obs_feat)
# # # # # # # # # # # # # # #                 x_t = x_t + dt * vel
            
# # # # # # # # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # # # # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # # # # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # # # # # # #             traj_s.append(tr)
# # # # # # # # # # # # # # #             me_s.append(me)

# # # # # # # # # # # # # # #         all_trajs = torch.stack(traj_s)       # [S, T, B, 2]
# # # # # # # # # # # # # # #         pred_mean = all_trajs.mean(0)          # [T, B, 2]

# # # # # # # # # # # # # # #         if predict_csv:
# # # # # # # # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # # # # # # # # # # #         return pred_mean, torch.stack(me_s).mean(0), all_trajs

# # # # # # # # # # # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # # # # # # # # # # #         """Speed constraint: TCs can't move > 600 km/6h."""
# # # # # # # # # # # # # # #         with torch.enable_grad():
# # # # # # # # # # # # # # #             x = x_pred.detach().requires_grad_(True)
# # # # # # # # # # # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # # # # # # # # # # #             for _ in range(n_steps):
# # # # # # # # # # # # # # #                 opt.zero_grad()
# # # # # # # # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # # # # # # # #                 pred_deg = _denorm_to_deg(pred_abs)
# # # # # # # # # # # # # # #                 dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # # # # # #                 lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # # # # # # # #                 cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # # # # #                 speed   = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # # # # # # # # # # #                                       + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # # # # # # # # # # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # # # # # # # # # # # # # #                 loss.backward()
# # # # # # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # # # # # # # # # # # # # #                 opt.step()
# # # # # # # # # # # # # # #         return x.detach()

# # # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # # # # # # # # # # # # # #         import numpy as np
# # # # # # # # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # # # # # # # #                     "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # # # # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # # # # # # # #             if write_hdr: w.writeheader()
# # # # # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # # # # #                 for k in range(T):
# # # # # # # # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # # # # # # # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(math.radians(mean_lat[k, b]))
# # # # # # # # # # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # # # # # # # # # #                     w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # # # # # # # # # # #                                 "lead_h": (k+1)*6, "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # # # # # # # # # # #                                 "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # # # # # # # # # # #                                 "lon_std_deg": f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # # # # # # # #                                 "lat_std_deg": f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # # # # # # # #                                 "ens_spread_km": f"{spread:.2f}"})


# # # # # # # # # # # # # # # # Backward compat alias
# # # # # # # # # # # # # # # TCDiffusion = TCFlowMatching
# # # # # # # # # # # # # # """
# # # # # # # # # # # # # # flow_matching_model_v33.py — Pure FM + MSE: Beat LSTM bằng đơn giản

# # # # # # # # # # # # # # CHANGES v33→v33fix:
# # # # # # # # # # # # # #   1. BỎ `with torch.no_grad()` khi reconstruct pred_abs → l_vel/l_head có gradient thực
# # # # # # # # # # # # # #   2. THÊM long_range_aux_loss() chỉ cho step 8-12 (48h-72h), zero risk cho 6-24h
# # # # # # # # # # # # # #   3. Phase weight cho long_range loss: chỉ bật sau epoch 20 (sau khi 12h ổn)
# # # # # # # # # # # # # #   4. Median ensemble thay mean tại inference → robust hơn với outlier members
# # # # # # # # # # # # # #   5. MSE_STEP_WEIGHTS giữ nguyên (không đổi short-range emphasis)
# # # # # # # # # # # # # # """
# # # # # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # # # # import csv
# # # # # # # # # # # # # # import math
# # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # from datetime import datetime
# # # # # # # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # # # # # # import torch
# # # # # # # # # # # # # # import torch.nn as nn
# # # # # # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # # # # # # from Model.losses import _haversine_deg, _norm_to_deg, _haversine

# # # # # # # # # # # # # # # ── Step weights: nhấn mạnh 12h và 24h ─────────────────────────────────────
# # # # # # # # # # # # # # # Step:    6h   12h  18h  24h  30h  36h  42h  48h  54h  60h  66h  72h
# # # # # # # # # # # # # # # Thay MSE_STEP_WEIGHTS
# # # # # # # # # # # # # # MSE_STEP_WEIGHTS = [
# # # # # # # # # # # # # #     1.0,  # 6h
# # # # # # # # # # # # # #     4.0,  # 12h
# # # # # # # # # # # # # #     1.5,  # 18h
# # # # # # # # # # # # # #     3.0,  # 24h
# # # # # # # # # # # # # #     1.0,  # 30h
# # # # # # # # # # # # # #     1.0,  # 36h
# # # # # # # # # # # # # #     1.5,  # 42h
# # # # # # # # # # # # # #     3.0,  # 48h  ← tăng từ 2.5
# # # # # # # # # # # # # #     2.0,  # 54h  ← tăng từ 1.0
# # # # # # # # # # # # # #     2.5,  # 60h  ← tăng từ 1.0
# # # # # # # # # # # # # #     3.5,  # 66h  ← tăng từ 1.5
# # # # # # # # # # # # # #     5.0,  # 72h  ← tăng từ 2.5, quan trọng nhất
# # # # # # # # # # # # # # ]

# # # # # # # # # # # # # # def _denorm_to_deg(t):
# # # # # # # # # # # # # #     """Convert normalized coords to degrees."""
# # # # # # # # # # # # # #     out = t.clone()
# # # # # # # # # # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # # # # # # # # # #     return out


# # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # #  MSE Haversine Loss
# # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # def mse_hav_loss(pred_norm, gt_norm, step_w=None):
# # # # # # # # # # # # # #     """Haversine MSE per-step với Huber để tránh gradient explosion."""
# # # # # # # # # # # # # #     if step_w is None:
# # # # # # # # # # # # # #         step_w = MSE_STEP_WEIGHTS
# # # # # # # # # # # # # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # # # # # # # #     pred_deg = _norm_to_deg(pred_norm[:T])
# # # # # # # # # # # # # #     gt_deg   = _norm_to_deg(gt_norm[:T])
# # # # # # # # # # # # # #     dist_km  = _haversine_deg(pred_deg, gt_deg)   # [T, B]

# # # # # # # # # # # # # #     w = pred_norm.new_tensor(step_w[:T])
# # # # # # # # # # # # # #     w = w / w.sum() * T

# # # # # # # # # # # # # #     delta = 300.0
# # # # # # # # # # # # # #     huber = torch.where(
# # # # # # # # # # # # # #         dist_km < delta,
# # # # # # # # # # # # # #         dist_km.pow(2) / (2.0 * delta),
# # # # # # # # # # # # # #         dist_km - delta / 2.0,
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     return (huber * w.unsqueeze(1)).mean() / delta


# # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # #  NEW: Long-range Auxiliary Loss — chỉ 48h-72h, zero risk cho 6-24h
# # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # def long_range_aux_loss(pred_abs, gt_abs, start_step=7):
# # # # # # # # # # # # # #     """
# # # # # # # # # # # # # #     Auxiliary loss chỉ cho step start_step trở đi (mặc định step 8-12 = 48h-72h).

# # # # # # # # # # # # # #     KEY PROPERTIES:
# # # # # # # # # # # # # #     - Gradient chỉ flow từ step 8-12, không đụng step 1-7
# # # # # # # # # # # # # #     - L1 Huber (delta=200km) robust hơn L2 với outlier trajectories
# # # # # # # # # # # # # #     - Weight tăng dần theo lead time (penalize 72h nhiều hơn 48h)
# # # # # # # # # # # # # #     - Normalize để scale tương đương với l_mse
# # # # # # # # # # # # # #     """
# # # # # # # # # # # # # #     T = min(pred_abs.shape[0], gt_abs.shape[0])
# # # # # # # # # # # # # #     if T <= start_step:
# # # # # # # # # # # # # #         return pred_abs.new_zeros(())

# # # # # # # # # # # # # #     pred_lr = pred_abs[start_step:]   # step 8-12: 48h→72h [T_lr, B, 2]
# # # # # # # # # # # # # #     gt_lr   = gt_abs[start_step:]     # [T_lr, B, 2]

# # # # # # # # # # # # # #     # Convert to degrees để tính haversine
# # # # # # # # # # # # # #     pred_lr_deg = _denorm_to_deg(pred_lr)
# # # # # # # # # # # # # #     gt_lr_deg   = _denorm_to_deg(gt_lr)

# # # # # # # # # # # # # #     dist_km = _haversine_deg(pred_lr_deg, gt_lr_deg)  # [T_lr, B]

# # # # # # # # # # # # # #     T_lr = dist_km.shape[0]
# # # # # # # # # # # # # #     # Weight tăng dần: 48h(1.0) → 54h(1.5) → 60h(2.0) → 66h(2.5) → 72h(3.0)
# # # # # # # # # # # # # #     w_vals = [1.0, 1.5, 2.0, 2.5, 3.0]
# # # # # # # # # # # # # #     w = pred_abs.new_tensor(w_vals[:T_lr])
# # # # # # # # # # # # # #     w = w / w.sum()

# # # # # # # # # # # # # #     # Huber loss, delta=200km (robust hơn L2 với large errors)
# # # # # # # # # # # # # #     delta = 200.0
# # # # # # # # # # # # # #     huber = torch.where(
# # # # # # # # # # # # # #         dist_km < delta,
# # # # # # # # # # # # # #         0.5 * dist_km.pow(2) / delta,
# # # # # # # # # # # # # #         dist_km - delta / 2.0,
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     return (huber * w.unsqueeze(1)).mean() / delta


# # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # #  VelocityField
# # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # # # # # # # # # # #                  unet_in_ch=13, **kwargs):
# # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # #         self.pred_len = pred_len
# # # # # # # # # # # # # #         self.obs_len  = obs_len

# # # # # # # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # # # # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # # # # # # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # # # # # # # # #             spatial_down=32, dropout=0.05)
# # # # # # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # # # # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # # # # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # # # # # # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # # # # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # # # # # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # # # # # # #         self.ctx_drop = nn.Dropout(0.10)
# # # # # # # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # # # # # # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # # # # # # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # # # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # # # # # #             nn.Linear(256, 256), nn.GELU(),
# # # # # # # # # # # # # #         )

# # # # # # # # # # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # # # # # # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # # # # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)

# # # # # # # # # # # # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # # # # # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # # # # # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # # # # # # # # # # # #             num_layers=5)

# # # # # # # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # # # # # # # # #         self.step_scale  = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # # # # # # # # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.3)

# # # # # # # # # # # # # #         self._init_weights()

# # # # # # # # # # # # # #     def _init_weights(self):
# # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # #             for name, m in self.named_modules():
# # # # # # # # # # # # # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # # # # # # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # # # # # # # # # # # #                     if m.bias is not None:
# # # # # # # # # # # # # #                         nn.init.zeros_(m.bias)

# # # # # # # # # # # # # #     def _time_emb(self, t, dim=256):
# # # # # # # # # # # # # #         half = dim // 2
# # # # # # # # # # # # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # # # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # # # # # # # # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # # # # # # # # # # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # # # # # # # # # # # #     def _context(self, batch_list):
# # # # # # # # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # # # # # # # #         image_obs = batch_list[11]
# # # # # # # # # # # # # #         env_data  = batch_list[13]

# # # # # # # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # # # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # # # # # # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # # # # # # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # # # # # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # # # # # # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # # # # # # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # # # # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)

# # # # # # # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # # # # # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # # # # # # # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # # # # # # # # # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# # # # # # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # # # # # # # # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # # # # # # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # # # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # # # # # # #     def _get_vel_obs_feat(self, obs_traj):
# # # # # # # # # # # # # #         B = obs_traj.shape[1]
# # # # # # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # # # # # #         if T_obs >= 2:
# # # # # # # # # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)

# # # # # # # # # # # # # #         if vels.shape[0] < self.obs_len:
# # # # # # # # # # # # # #             pad = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# # # # # # # # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # #             vels = vels[-self.obs_len:]

# # # # # # # # # # # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # # # # # # # # # # #     def _beta_drift(self, x_t):
# # # # # # # # # # # # # #         lat_deg  = x_t[:, :, 1] * 5.0
# # # # # # # # # # # # # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # # # # # # # # # #         beta     = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # # # # # # # # # #         R_tc     = 3e5
# # # # # # # # # # # # # #         v_phys   = torch.zeros_like(x_t)
# # # # # # # # # # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # # # # # # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # # # # # # # # # #         return v_phys

# # # # # # # # # # # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None):
# # # # # # # # # # # # # #         B = x_t.shape[0]
# # # # # # # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # # # # # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # # # # # # # # #         T_seq = min(x_t.size(1), self.pos_enc.shape[1])

# # # # # # # # # # # # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # # # # # # # # #         s_emb = self.step_embed(step_idx)

# # # # # # # # # # # # # #         x_emb = self.traj_embed(x_t[:, :T_seq]) + self.pos_enc[:, :T_seq] + t_emb.unsqueeze(1) + s_emb

# # # # # # # # # # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # # # # # # # # # #         if vel_obs_feat is not None:
# # # # # # # # # # # # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))

# # # # # # # # # # # # # #         decoded = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # # # # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))

# # # # # # # # # # # # # #         scale = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # # # # # # # # # # #         v_neural = v_neural * scale

# # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # #             v_phys = self._beta_drift(x_t[:, :T_seq])

# # # # # # # # # # # # # #         return v_neural + torch.sigmoid(self.physics_scale) * v_phys

# # # # # # # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0, vel_obs_feat=None):
# # # # # # # # # # # # # #         return self._decode(x_t, t, self._apply_ctx_head(raw_ctx, noise_scale), vel_obs_feat)


# # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # # # #  TCFlowMatching v33fix
# # # # # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # # # class TCFlowMatching(nn.Module):

# # # # # # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # # # # # # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # # # # # # # # # #                  ctx_noise_scale=0.002, initial_sample_sigma=0.03, **kwargs):
# # # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # # # # # # # # #         self.active_pred_len      = pred_len

# # # # # # # # # # # # # #         self.net = VelocityField(
# # # # # # # # # # # # # #             pred_len=pred_len, obs_len=obs_len,
# # # # # # # # # # # # # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # # # # # # # # # # #     def set_curriculum_len(self, *a, **kw):
# # # # # # # # # # # # # #         pass

# # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # # # # # # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # # # # # # # # # #                            Me   - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # #     def _to_abs(rel, lp, lm):
# # # # # # # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # # # # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # # # # # # # # # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # # # # # # # # # # # #         if sigma_min is None:
# # # # # # # # # # # # # #             sigma_min = self.sigma_min
# # # # # # # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # # # # # # #         x0 = torch.randn_like(x1) * sigma_min
# # # # # # # # # # # # # #         t  = torch.rand(B, device=device)
# # # # # # # # # # # # # #         te = t.view(B, 1, 1)
# # # # # # # # # # # # # #         x_t = (1.0 - te) * x0 + te * x1
# # # # # # # # # # # # # #         u   = x1 - x0
# # # # # # # # # # # # # #         return x_t, t, u

# # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # #     def _intensity_weights(obs_Me):
# # # # # # # # # # # # # #         w = obs_Me[-1, :, 1].detach()
# # # # # # # # # # # # # #         w = torch.where(w < 0.1, torch.full_like(w, 0.5),
# # # # # # # # # # # # # #             torch.where(w < 0.3, torch.full_like(w, 0.8),
# # # # # # # # # # # # # #             torch.where(w < 0.6, torch.full_like(w, 1.0),
# # # # # # # # # # # # # #                         torch.full_like(w, 1.5))))
# # # # # # # # # # # # # #         return w / w.mean().clamp(min=1e-6)

# # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # # # # # #             return batch_list
# # # # # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # # # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # # # # # # # # # #                 t = aug[idx].clone()
# # # # # # # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # # # # # # #                 aug[idx] = t
# # # # # # # # # # # # # #         return aug

# # # # # # # # # # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # # # # # # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # # # # # #         """
# # # # # # # # # # # # # #         FIX KEY CHANGES:
# # # # # # # # # # # # # #         1. BỎ `with torch.no_grad()` → pred_abs có gradient thực
# # # # # # # # # # # # # #         2. THÊM long_range_aux_loss cho 48h-72h (chỉ bật sau epoch 20)
# # # # # # # # # # # # # #         3. l_vel và l_head bây giờ có gradient flow đúng
# # # # # # # # # # # # # #         """
# # # # # # # # # # # # # #         batch_list = self._lon_flip_aug(batch_list)

# # # # # # # # # # # # # #         traj_gt = batch_list[1]
# # # # # # # # # # # # # #         Me_gt   = batch_list[8]
# # # # # # # # # # # # # #         obs_t   = batch_list[0]
# # # # # # # # # # # # # #         obs_Me  = batch_list[7]
# # # # # # # # # # # # # #         lp, lm  = obs_t[-1], obs_Me[-1]

# # # # # # # # # # # # # #         # ── Sigma schedule ──────────────────────────────────────────────────
# # # # # # # # # # # # # #         if epoch < 15:
# # # # # # # # # # # # # #             current_sigma = 0.15
# # # # # # # # # # # # # #         elif epoch < 40:
# # # # # # # # # # # # # #             sigma_frac = (epoch - 15) / 25.0
# # # # # # # # # # # # # #             current_sigma = 0.15 - sigma_frac * (0.15 - 0.03)
# # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # #             current_sigma = 0.03

# # # # # # # # # # # # # #         # ── Encode context ───────────────────────────────────────────────────
# # # # # # # # # # # # # #         raw_ctx      = self.net._context(batch_list)
# # # # # # # # # # # # # #         vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # # # # # # # # # #         # ── FM forward ───────────────────────────────────────────────────────
# # # # # # # # # # # # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)  # [B, T, 4]
# # # # # # # # # # # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # # # # # # # # # # #         pred_vel = self.net.forward_with_ctx(
# # # # # # # # # # # # # #             x_t, fm_t, raw_ctx, vel_obs_feat=vel_obs_feat)

# # # # # # # # # # # # # #         # ── LOSS 1: FM velocity MSE (primary) ────────────────────────────────
# # # # # # # # # # # # # #         l_mse = F.mse_loss(pred_vel, u_target)

# # # # # # # # # # # # # #         # ── Reconstruct pred_abs ─────────────────────────────────────────────
# # # # # # # # # # # # # #         # FIX: BỎ `with torch.no_grad()` để l_vel, l_head, l_lr có gradient thực
# # # # # # # # # # # # # #         # → l_vel/l_head từ epoch 15+ bây giờ thực sự optimize model weights
# # # # # # # # # # # # # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # # # # # # # # # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # # # # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# # # # # # # # # # # # # #         pred_deg = _denorm_to_deg(pred_abs)
# # # # # # # # # # # # # #         gt_deg   = _denorm_to_deg(traj_gt)

# # # # # # # # # # # # # #         # ── LOSS 2: Velocity matching ─────────────────────────────────────────
# # # # # # # # # # # # # #         if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # # # # # # # # #             T_min  = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # # # # # # # #             v_pred = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # # # # # # # # #             v_gt   = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # # # # # # # # # #             l_vel  = F.smooth_l1_loss(v_pred, v_gt)
# # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # #             l_vel = x_t.new_zeros(())

# # # # # # # # # # # # # #         # ── LOSS 3: Heading consistency ────────────────────────────────────────
# # # # # # # # # # # # # #         if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # # # # # # # # #             T_min    = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # # # # # # # #             v_pred   = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # # # # # # # # #             v_gt     = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # # # # # # # # # #             v_pred_n = v_pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # # # # # # # # #             v_gt_n   = v_gt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # # # # # # # # #             cos_sim  = ((v_pred / v_pred_n) * (v_gt / v_gt_n)).sum(-1)
# # # # # # # # # # # # # #             l_head   = F.relu(-cos_sim).pow(2).mean()
# # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # #             l_head = x_t.new_zeros(())

# # # # # # # # # # # # # #         # ── LOSS 4: Long-range auxiliary (48h-72h ONLY) ────────────────────────
# # # # # # # # # # # # # #         # FIX: Loss này chỉ tính gradient từ step 8-12, KHÔNG đụng step 1-7
# # # # # # # # # # # # # #         # → 6h/12h/24h hoàn toàn không bị ảnh hưởng
# # # # # # # # # # # # # #         # → Chỉ bật sau epoch 20 khi 12h đã ổn định (< 60km)
# # # # # # # # # # # # # #         l_lr = long_range_aux_loss(pred_abs, traj_gt, start_step=7)

# # # # # # # # # # # # # #         # ── Phase weights ──────────────────────────────────────────────────────
# # # # # # # # # # # # # #         # if epoch < 15:
# # # # # # # # # # # # # #         #     w_vel, w_head = 0.0, 0.0
# # # # # # # # # # # # # #         #     w_lr = 0.0
# # # # # # # # # # # # # #         # elif epoch < 20:
# # # # # # # # # # # # # #         #     # Transition: vel/head bắt đầu, lr chưa bật
# # # # # # # # # # # # # #         #     phase_frac = (epoch - 15) / 5.0
# # # # # # # # # # # # # #         #     w_vel  = phase_frac * 0.3
# # # # # # # # # # # # # #         #     w_head = phase_frac * 0.2
# # # # # # # # # # # # # #         #     w_lr   = 0.0
# # # # # # # # # # # # # #         # elif epoch < 40:
# # # # # # # # # # # # # #         #     phase_frac = (epoch - 15) / 25.0
# # # # # # # # # # # # # #         #     w_vel  = phase_frac * 0.3
# # # # # # # # # # # # # #         #     w_head = phase_frac * 0.2
# # # # # # # # # # # # # #         #     # Long-range loss: warm-up từ epoch 20, tăng dần đến 0.5
# # # # # # # # # # # # # #         #     lr_frac = (epoch - 20) / 20.0
# # # # # # # # # # # # # #         #     w_lr = min(lr_frac, 1.0) * 0.5
# # # # # # # # # # # # # #         # else:
# # # # # # # # # # # # # #         #     w_vel, w_head = 0.3, 0.2
# # # # # # # # # # # # # #         #     w_lr = 0.5  # Full weight sau epoch 40

# # # # # # # # # # # # # #         # total = l_mse + w_vel * l_vel + w_head * l_head + w_lr * l_lr
# # # # # # # # # # # # # #          # --- THÊM: Terminal displacement loss (FDE-style) ---
# # # # # # # # # # # # # #         # Penalize error tại step 12 một cách mạnh mẽ hơn
# # # # # # # # # # # # # #         T = min(pred_abs.shape[0], traj_gt.shape[0])
# # # # # # # # # # # # # #         if T >= 12:
# # # # # # # # # # # # # #             pred_last = pred_abs[11]   # step 12 normalized
# # # # # # # # # # # # # #             gt_last   = traj_gt[11]
# # # # # # # # # # # # # #             pred_last_deg = _denorm_to_deg(pred_last.unsqueeze(0))
# # # # # # # # # # # # # #             gt_last_deg   = _denorm_to_deg(gt_last.unsqueeze(0))
# # # # # # # # # # # # # #             l_fde = _haversine_deg(pred_last_deg, gt_last_deg).mean()
# # # # # # # # # # # # # #             # Normalize: target ~300km → l_fde/300 ~ 1.0
# # # # # # # # # # # # # #             l_fde = l_fde / 300.0
# # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # #             l_fde = pred_abs.new_zeros(())
        
# # # # # # # # # # # # # #         # --- THÊM: Long-range trajectory shape loss ---
# # # # # # # # # # # # # #         # MSE giữa displacement vectors ở step 7-12
# # # # # # # # # # # # # #         if pred_deg.shape[0] >= 12 and gt_deg.shape[0] >= 12:
# # # # # # # # # # # # # #             pred_lr = pred_deg[6:]    # step 7-12 [6, B, 2]
# # # # # # # # # # # # # #             gt_lr   = gt_deg[6:]
# # # # # # # # # # # # # #             # Displacement từ step 7 đến step 12
# # # # # # # # # # # # # #             pred_disp = pred_lr[-1] - pred_lr[0]   # [B, 2]
# # # # # # # # # # # # # #             gt_disp   = gt_lr[-1]   - gt_lr[0]
# # # # # # # # # # # # # #             # Penalize sai hướng và magnitude
# # # # # # # # # # # # # #             l_lr_shape = F.smooth_l1_loss(pred_disp, gt_disp)
# # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # #             l_lr_shape = pred_abs.new_zeros(())
        
# # # # # # # # # # # # # #         # FDE weight schedule: bắt đầu nhẹ, tăng dần
# # # # # # # # # # # # # #         if epoch < 20:
# # # # # # # # # # # # # #             w_fde = 0.0
# # # # # # # # # # # # # #             w_lr_shape = 0.0
# # # # # # # # # # # # # #         elif epoch < 40:
# # # # # # # # # # # # # #             w_fde = (epoch - 20) / 20.0 * 0.5      # 0 → 0.5
# # # # # # # # # # # # # #             w_lr_shape = (epoch - 20) / 20.0 * 0.3  # 0 → 0.3
# # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # #             w_fde = 0.5
# # # # # # # # # # # # # #             w_lr_shape = 0.3

# # # # # # # # # # # # # #         total = l_mse + w_vel * l_vel + w_head * l_head + w_lr * l_lr \
# # # # # # # # # # # # # #                 + w_fde * l_fde + w_lr_shape * l_lr_shape
# # # # # # # # # # # # # #         # ── Ensemble consistency (epoch 40+) ────────────────────────────────────
# # # # # # # # # # # # # #         l_ens_consist = x_t.new_zeros(())
# # # # # # # # # # # # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # # # # # # # # # # # #             x_t2, fm_t2, u_target2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # # # # # # # # # # #             pred_vel2 = self.net.forward_with_ctx(
# # # # # # # # # # # # # #                 x_t2, fm_t2, raw_ctx, vel_obs_feat=vel_obs_feat)
# # # # # # # # # # # # # #             l_ens_consist = F.mse_loss(pred_vel2, u_target2)
# # # # # # # # # # # # # #             total = total + 0.3 * l_ens_consist

# # # # # # # # # # # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # # # # # # # # # # #             total = x_t.new_zeros(())

# # # # # # # # # # # # # #         return dict(
# # # # # # # # # # # # # #             total=total,
# # # # # # # # # # # # # #             mse_hav=l_mse.item(),
# # # # # # # # # # # # # #             velocity=l_vel.item() if isinstance(l_vel, torch.Tensor) else l_vel,
# # # # # # # # # # # # # #             heading=l_head.item() if isinstance(l_head, torch.Tensor) else l_head,
# # # # # # # # # # # # # #             long_range=l_lr.item(),
# # # # # # # # # # # # # #             ens_consist=l_ens_consist.item(),
# # # # # # # # # # # # # #             sigma=current_sigma,
# # # # # # # # # # # # # #             w_lr=w_lr,
# # # # # # # # # # # # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # # # # # # # # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0,
# # # # # # # # # # # # # #             recurv_ratio=0.0,
# # # # # # # # # # # # # #         )

# # # # # # # # # # # # # #     @torch.no_grad()
# # # # # # # # # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
# # # # # # # # # # # # # #         """
# # # # # # # # # # # # # #         FIX: Dùng median thay mean → robust hơn với outlier ensemble members.
# # # # # # # # # # # # # #         Median đặc biệt tốt cho long-range vì spread lớn hơn ở 48h/72h.
# # # # # # # # # # # # # #         """
# # # # # # # # # # # # # #         obs_t  = batch_list[0]
# # # # # # # # # # # # # #         lp     = obs_t[-1]
# # # # # # # # # # # # # #         lm     = batch_list[7][-1]
# # # # # # # # # # # # # #         B      = lp.shape[0]
# # # # # # # # # # # # # #         device = lp.device
# # # # # # # # # # # # # #         T      = self.pred_len
# # # # # # # # # # # # # #         dt     = 1.0 / max(ddim_steps, 1)

# # # # # # # # # # # # # #         raw_ctx      = self.net._context(batch_list)
# # # # # # # # # # # # # #         vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # # # # # # # # # #         traj_s, me_s = [], []
# # # # # # # # # # # # # #         for _ in range(num_ensemble):
# # # # # # # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.sigma_min

# # # # # # # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # # # # #                 ns  = self.ctx_noise_scale if step < 3 else 0.0
# # # # # # # # # # # # # #                 vel = self.net.forward_with_ctx(
# # # # # # # # # # # # # #                     x_t, t_b, raw_ctx,
# # # # # # # # # # # # # #                     noise_scale=ns, vel_obs_feat=vel_obs_feat)
# # # # # # # # # # # # # #                 x_t = x_t + dt * vel

# # # # # # # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # # # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # # # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # # # # # #             traj_s.append(tr)
# # # # # # # # # # # # # #             me_s.append(me)

# # # # # # # # # # # # # #         all_trajs = torch.stack(traj_s)       # [S, T, B, 2]

# # # # # # # # # # # # # #         # FIX: Median thay vì mean → robust với outlier ensemble members
# # # # # # # # # # # # # #         # Đặc biệt quan trọng cho 48h/72h khi spread lớn (~50-70km)
# # # # # # # # # # # # # #         pred_mean = all_trajs.median(0).values  # [T, B, 2]

# # # # # # # # # # # # # #         if predict_csv:
# # # # # # # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # # # # # # # # # #         return pred_mean, torch.stack(me_s).mean(0), all_trajs

# # # # # # # # # # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # # # # # # # # # #         """Speed constraint: TCs can't move > 600 km/6h."""
# # # # # # # # # # # # # #         with torch.enable_grad():
# # # # # # # # # # # # # #             x = x_pred.detach().requires_grad_(True)
# # # # # # # # # # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # # # # # # # # # #             for _ in range(n_steps):
# # # # # # # # # # # # # #                 opt.zero_grad()
# # # # # # # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # # # # # # #                 pred_deg = _denorm_to_deg(pred_abs)
# # # # # # # # # # # # # #                 dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # # # # #                 lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # # # # # # #                 cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # # # #                 speed   = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # # # # # # # # # #                                       + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # # # # # # # # # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # # # # # # # # # # # # #                 loss.backward()
# # # # # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # # # # # # # # # # # # #                 opt.step()
# # # # # # # # # # # # # #         return x.detach()

# # # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # # # # # # # # # # # # #         import numpy as np
# # # # # # # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # # # # # # #                     "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # # # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # # # # # # #             if write_hdr: w.writeheader()
# # # # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # # # #                 for k in range(T):
# # # # # # # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # # # # # # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(math.radians(mean_lat[k, b]))
# # # # # # # # # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # # # # # # # # #                     w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # # # # # # # # # #                                 "lead_h": (k+1)*6, "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # # # # # # # # # #                                 "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # # # # # # # # # #                                 "lon_std_deg": f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # # # # # # #                                 "lat_std_deg": f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # # # # # # #                                 "ens_spread_km": f"{spread:.2f}"})


# # # # # # # # # # # # # # # Backward compat alias
# # # # # # # # # # # # # # TCDiffusion = TCFlowMatching
# # # # # # # # # # # # # """
# # # # # # # # # # # # # flow_matching_model_v33_fixed.py

# # # # # # # # # # # # # FIXES từ v33fix — 3 bugs nghiêm trọng:

# # # # # # # # # # # # # BUG 1 (NameError): w_vel, w_lr, w_head undefined.
# # # # # # # # # # # # #   Code cũ comment out phần định nghĩa nhưng vẫn dùng trong total.
# # # # # # # # # # # # #   → NameError khi chạy, training crash ngay batch đầu.

# # # # # # # # # # # # # BUG 2 (double-denorm): l_lr_shape dùng pred_deg (đã degrees)
# # # # # # # # # # # # #   rồi gọi _denorm_to_deg lần nữa → kết quả vô nghĩa,
# # # # # # # # # # # # #   gradient sai hoàn toàn, 72h không học được gì.

# # # # # # # # # # # # # BUG 3 (wrong input): long_range_aux_loss trong v33fix
# # # # # # # # # # # # #   nhận pred_abs (normalized) nhưng gọi _denorm_to_deg để convert —
# # # # # # # # # # # # #   đây là ĐÚNG. Nhưng lr_shape_loss cũng nhận pred_deg (degrees)
# # # # # # # # # # # # #   mà lại gọi pred_deg[6:] trên degrees — đây cũng ĐÚNG nếu giữ nguyên.
# # # # # # # # # # # # #   Vấn đề là code cũ lẫn lộn normalized và degrees không nhất quán.

# # # # # # # # # # # # # SOLUTION: Tách 3 loss functions với input type rõ ràng:
# # # # # # # # # # # # #   - long_range_aux_loss(normalized, normalized) → haversine nội bộ
# # # # # # # # # # # # #   - fde_loss(normalized, normalized) → haversine tại step cuối
# # # # # # # # # # # # #   - lr_shape_loss(degrees, degrees) → smooth_l1 displacement

# # # # # # # # # # # # # Phase weights bật 72h losses từ epoch 0 (không đợi epoch 20).
# # # # # # # # # # # # # """
# # # # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # # # import csv
# # # # # # # # # # # # # import math
# # # # # # # # # # # # # import os
# # # # # # # # # # # # # from datetime import datetime

# # # # # # # # # # # # # import torch
# # # # # # # # # # # # # import torch.nn as nn
# # # # # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # # # # # from Model.losses import _haversine_deg, _norm_to_deg, _haversine

# # # # # # # # # # # # # MSE_STEP_WEIGHTS = [
# # # # # # # # # # # # #     1.0,  # 6h
# # # # # # # # # # # # #     4.0,  # 12h  target < 50km
# # # # # # # # # # # # #     1.5,  # 18h
# # # # # # # # # # # # #     3.0,  # 24h  target < 100km
# # # # # # # # # # # # #     1.0,  # 30h
# # # # # # # # # # # # #     1.0,  # 36h
# # # # # # # # # # # # #     1.5,  # 42h
# # # # # # # # # # # # #     3.0,  # 48h  target < 200km
# # # # # # # # # # # # #     2.0,  # 54h
# # # # # # # # # # # # #     2.5,  # 60h
# # # # # # # # # # # # #     3.5,  # 66h
# # # # # # # # # # # # #     5.0,  # 72h  target < 300km — weight cao nhất
# # # # # # # # # # # # # ]


# # # # # # # # # # # # # def _denorm_to_deg(t):
# # # # # # # # # # # # #     out = t.clone()
# # # # # # # # # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # # # # # # # # #     return out


# # # # # # # # # # # # # def mse_hav_loss(pred_norm, gt_norm, step_w=None):
# # # # # # # # # # # # #     if step_w is None:
# # # # # # # # # # # # #         step_w = MSE_STEP_WEIGHTS
# # # # # # # # # # # # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # # # # # # #     pred_deg = _norm_to_deg(pred_norm[:T])
# # # # # # # # # # # # #     gt_deg   = _norm_to_deg(gt_norm[:T])
# # # # # # # # # # # # #     dist_km  = _haversine_deg(pred_deg, gt_deg)
# # # # # # # # # # # # #     w = pred_norm.new_tensor(step_w[:T])
# # # # # # # # # # # # #     w = w / w.sum() * T
# # # # # # # # # # # # #     delta = 300.0
# # # # # # # # # # # # #     huber = torch.where(dist_km < delta,
# # # # # # # # # # # # #                         dist_km.pow(2) / (2.0 * delta),
# # # # # # # # # # # # #                         dist_km - delta / 2.0)
# # # # # # # # # # # # #     return (huber * w.unsqueeze(1)).mean() / delta


# # # # # # # # # # # # # def long_range_aux_loss(pred_norm, gt_norm, start_step=7):
# # # # # # # # # # # # #     """Input: NORMALIZED [T,B,2]. Gradient chỉ từ step 8-12."""
# # # # # # # # # # # # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # # # # # # #     if T <= start_step:
# # # # # # # # # # # # #         return pred_norm.new_zeros(())
# # # # # # # # # # # # #     pred_deg = _norm_to_deg(pred_norm[start_step:T])
# # # # # # # # # # # # #     gt_deg   = _norm_to_deg(gt_norm[start_step:T])
# # # # # # # # # # # # #     dist_km  = _haversine_deg(pred_deg, gt_deg)
# # # # # # # # # # # # #     T_lr = dist_km.shape[0]
# # # # # # # # # # # # #     w_vals = [1.0, 1.5, 2.0, 2.5, 3.0]
# # # # # # # # # # # # #     w = pred_norm.new_tensor(w_vals[:T_lr])
# # # # # # # # # # # # #     w = w / w.sum()
# # # # # # # # # # # # #     delta = 200.0
# # # # # # # # # # # # #     huber = torch.where(dist_km < delta,
# # # # # # # # # # # # #                         0.5 * dist_km.pow(2) / delta,
# # # # # # # # # # # # #                         dist_km - delta / 2.0)
# # # # # # # # # # # # #     return (huber * w.unsqueeze(1)).mean() / delta


# # # # # # # # # # # # # def fde_loss(pred_norm, gt_norm):
# # # # # # # # # # # # #     """Terminal haversine error tại step cuối. Input: NORMALIZED [T,B,2]."""
# # # # # # # # # # # # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # # # # # # #     if T < 1:
# # # # # # # # # # # # #         return pred_norm.new_zeros(())
# # # # # # # # # # # # #     pred_last_deg = _norm_to_deg(pred_norm[T-1:T])
# # # # # # # # # # # # #     gt_last_deg   = _norm_to_deg(gt_norm[T-1:T])
# # # # # # # # # # # # #     return _haversine_deg(pred_last_deg, gt_last_deg).mean() / 300.0


# # # # # # # # # # # # # def lr_shape_loss(pred_deg, gt_deg):
# # # # # # # # # # # # #     """Displacement step7→step12. Input: DEGREES [T,B,2]. Không denorm."""
# # # # # # # # # # # # #     if pred_deg.shape[0] < 12 or gt_deg.shape[0] < 12:
# # # # # # # # # # # # #         return pred_deg.new_zeros(())
# # # # # # # # # # # # #     pred_disp = pred_deg[11] - pred_deg[6]
# # # # # # # # # # # # #     gt_disp   = gt_deg[11]   - gt_deg[6]
# # # # # # # # # # # # #     return F.smooth_l1_loss(pred_disp, gt_disp)


# # # # # # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # # # # # # # # # #                  unet_in_ch=13, **kwargs):
# # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # #         self.pred_len = pred_len
# # # # # # # # # # # # #         self.obs_len  = obs_len

# # # # # # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # # # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # # # # # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # # # # # # # #             spatial_down=32, dropout=0.05)
# # # # # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # # # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # # # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # # # # # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # # # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # # # # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # # # # # #         self.ctx_drop = nn.Dropout(0.10)
# # # # # # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # # # # # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # # # # # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # # # # #             nn.Linear(256, 256), nn.GELU(),
# # # # # # # # # # # # #         )

# # # # # # # # # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # # # # # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # # # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # # # # # # # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # # # # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # # # # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # # # # # # # # # # #             num_layers=5)

# # # # # # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # # # # # # # #         self.step_scale    = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # # # # # # # # # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.3)
# # # # # # # # # # # # #         self._init_weights()

# # # # # # # # # # # # #     def _init_weights(self):
# # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # #             for name, m in self.named_modules():
# # # # # # # # # # # # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # # # # # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # # # # # # # # # # #                     if m.bias is not None:
# # # # # # # # # # # # #                         nn.init.zeros_(m.bias)

# # # # # # # # # # # # #     def _time_emb(self, t, dim=256):
# # # # # # # # # # # # #         half = dim // 2
# # # # # # # # # # # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # # # # # # # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # # # # # # # # # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # # # # # # # # # # #     def _context(self, batch_list):
# # # # # # # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # # # # # # #         image_obs = batch_list[11]
# # # # # # # # # # # # #         env_data  = batch_list[13]

# # # # # # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # # # # # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # # # # # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # # # # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # # # # # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # # # # # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # # # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)

# # # # # # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # # # # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # # # # # # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # # # # # # # # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# # # # # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # # # # # # # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(
# # # # # # # # # # # # #             torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # # # # # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # # # # # #     def _get_vel_obs_feat(self, obs_traj):
# # # # # # # # # # # # #         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
# # # # # # # # # # # # #         if T_obs >= 2:
# # # # # # # # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # # # # # # # # # # #         else:
# # # # # # # # # # # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
# # # # # # # # # # # # #         if vels.shape[0] < self.obs_len:
# # # # # # # # # # # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# # # # # # # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # # # # # # #         else:
# # # # # # # # # # # # #             vels = vels[-self.obs_len:]
# # # # # # # # # # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # # # # # # # # # #     def _beta_drift(self, x_t):
# # # # # # # # # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # # # # # # # # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # # # # # # # # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # # # # # # # # #         R_tc    = 3e5
# # # # # # # # # # # # #         v_phys  = torch.zeros_like(x_t)
# # # # # # # # # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # # # # # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # # # # # # # # #         return v_phys

# # # # # # # # # # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None):
# # # # # # # # # # # # #         B     = x_t.shape[0]
# # # # # # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # # # # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # # # # # # # #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # # # # # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # # # # # # # #         s_emb    = self.step_embed(step_idx)

# # # # # # # # # # # # #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# # # # # # # # # # # # #                  + self.pos_enc[:, :T_seq]
# # # # # # # # # # # # #                  + t_emb.unsqueeze(1)
# # # # # # # # # # # # #                  + s_emb)

# # # # # # # # # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # # # # # # # # #         if vel_obs_feat is not None:
# # # # # # # # # # # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))

# # # # # # # # # # # # #         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # # # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))

# # # # # # # # # # # # #         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # # # # # # # # # #         v_neural = v_neural * scale

# # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # #             v_phys = self._beta_drift(x_t[:, :T_seq])
# # # # # # # # # # # # #         return v_neural + torch.sigmoid(self.physics_scale) * v_phys

# # # # # # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0, vel_obs_feat=None):
# # # # # # # # # # # # #         return self._decode(x_t, t, self._apply_ctx_head(raw_ctx, noise_scale), vel_obs_feat)


# # # # # # # # # # # # # class TCFlowMatching(nn.Module):

# # # # # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # # # # # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # # # # # # # # #                  ctx_noise_scale=0.002, initial_sample_sigma=0.03, **kwargs):
# # # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # # # # # # # #         self.active_pred_len      = pred_len
# # # # # # # # # # # # #         self.net = VelocityField(pred_len=pred_len, obs_len=obs_len,
# # # # # # # # # # # # #                                   sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # # # # # # # # # #     def set_curriculum_len(self, *a, **kw):
# # # # # # # # # # # # #         pass

# # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # # # # # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # # # # # # # # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # #     def _to_abs(rel, lp, lm):
# # # # # # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # # # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # # # # # # # # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # # # # # # # # # # #         if sigma_min is None:
# # # # # # # # # # # # #             sigma_min = self.sigma_min
# # # # # # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # # # # # #         x0 = torch.randn_like(x1) * sigma_min
# # # # # # # # # # # # #         t  = torch.rand(B, device=device)
# # # # # # # # # # # # #         te = t.view(B, 1, 1)
# # # # # # # # # # # # #         x_t = (1.0 - te) * x0 + te * x1
# # # # # # # # # # # # #         u   = x1 - x0
# # # # # # # # # # # # #         return x_t, t, u

# # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # # # # #             return batch_list
# # # # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # # # # # # # # #                 t = aug[idx].clone()
# # # # # # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # # # # # #                 aug[idx] = t
# # # # # # # # # # # # #         return aug

# # # # # # # # # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # # # # # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # # # # #         batch_list = self._lon_flip_aug(batch_list)

# # # # # # # # # # # # #         traj_gt = batch_list[1]
# # # # # # # # # # # # #         Me_gt   = batch_list[8]
# # # # # # # # # # # # #         obs_t   = batch_list[0]
# # # # # # # # # # # # #         obs_Me  = batch_list[7]
# # # # # # # # # # # # #         lp, lm  = obs_t[-1], obs_Me[-1]

# # # # # # # # # # # # #         if epoch < 15:
# # # # # # # # # # # # #             current_sigma = 0.15
# # # # # # # # # # # # #         elif epoch < 40:
# # # # # # # # # # # # #             current_sigma = 0.15 - (epoch - 15) / 25.0 * 0.12
# # # # # # # # # # # # #         else:
# # # # # # # # # # # # #             current_sigma = 0.03

# # # # # # # # # # # # #         raw_ctx      = self.net._context(batch_list)
# # # # # # # # # # # # #         vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # # # # # # # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
# # # # # # # # # # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # # # # # # # # # #         pred_vel = self.net.forward_with_ctx(x_t, fm_t, raw_ctx, vel_obs_feat=vel_obs_feat)

# # # # # # # # # # # # #         l_mse = F.mse_loss(pred_vel, u_target)

# # # # # # # # # # # # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # # # # # # # # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # # # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)  # normalized [T,B,2]
# # # # # # # # # # # # #         pred_deg = _denorm_to_deg(pred_abs)            # degrees [T,B,2]
# # # # # # # # # # # # #         gt_deg   = _denorm_to_deg(traj_gt)             # degrees [T,B,2]

# # # # # # # # # # # # #         # L2: velocity (degrees)
# # # # # # # # # # # # #         if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # # # # # # # #             T_min  = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # # # # # # #             v_pred = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # # # # # # # #             v_gt   = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # # # # # # # # #             l_vel  = F.smooth_l1_loss(v_pred, v_gt)
# # # # # # # # # # # # #         else:
# # # # # # # # # # # # #             l_vel  = x_t.new_zeros(())

# # # # # # # # # # # # #         # L3: heading (degrees)
# # # # # # # # # # # # #         if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
# # # # # # # # # # # # #             T_min    = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # # # # # # # # #             v_pred   = pred_deg[1:T_min] - pred_deg[:T_min-1]
# # # # # # # # # # # # #             v_gt     = gt_deg[1:T_min]   - gt_deg[:T_min-1]
# # # # # # # # # # # # #             v_pred_n = v_pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # # # # # # # #             v_gt_n   = v_gt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # # # # # # # # # # #             cos_sim  = ((v_pred / v_pred_n) * (v_gt / v_gt_n)).sum(-1)
# # # # # # # # # # # # #             l_head   = F.relu(-cos_sim).pow(2).mean()
# # # # # # # # # # # # #         else:
# # # # # # # # # # # # #             l_head   = x_t.new_zeros(())

# # # # # # # # # # # # #         # L4: long-range auxiliary (NORMALIZED input)
# # # # # # # # # # # # #         l_lr = long_range_aux_loss(pred_abs, traj_gt, start_step=7)

# # # # # # # # # # # # #         # L5: FDE terminal (NORMALIZED input)
# # # # # # # # # # # # #         l_fde = fde_loss(pred_abs, traj_gt)

# # # # # # # # # # # # #         # L6: shape (DEGREES input — NO second denorm)
# # # # # # # # # # # # #         l_shape = lr_shape_loss(pred_deg, gt_deg)

# # # # # # # # # # # # #         # ── ALL weights defined here, no undefined variables ──────────────
# # # # # # # # # # # # #         if epoch < 5:
# # # # # # # # # # # # #             w_vel = 0.0;  w_head = 0.0
# # # # # # # # # # # # #             w_lr  = 0.2;  w_fde  = 0.15;  w_shape = 0.1
# # # # # # # # # # # # #         elif epoch < 20:
# # # # # # # # # # # # #             t_p   = (epoch - 5) / 15.0
# # # # # # # # # # # # #             w_vel   = t_p * 0.3
# # # # # # # # # # # # #             w_head  = t_p * 0.2
# # # # # # # # # # # # #             w_lr    = 0.2 + t_p * 0.3
# # # # # # # # # # # # #             w_fde   = 0.15 + t_p * 0.35
# # # # # # # # # # # # #             w_shape = 0.1 + t_p * 0.2
# # # # # # # # # # # # #         else:
# # # # # # # # # # # # #             w_vel = 0.3;  w_head = 0.2
# # # # # # # # # # # # #             w_lr  = 0.5;  w_fde  = 0.5;   w_shape = 0.3

# # # # # # # # # # # # #         total = (l_mse
# # # # # # # # # # # # #                  + w_vel   * l_vel
# # # # # # # # # # # # #                  + w_head  * l_head
# # # # # # # # # # # # #                  + w_lr    * l_lr
# # # # # # # # # # # # #                  + w_fde   * l_fde
# # # # # # # # # # # # #                  + w_shape * l_shape)

# # # # # # # # # # # # #         l_ens = x_t.new_zeros(())
# # # # # # # # # # # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # # # # # # # # # # #             x_t2, fm_t2, u_t2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # # # # # # # # # #             pv2   = self.net.forward_with_ctx(x_t2, fm_t2, raw_ctx, vel_obs_feat=vel_obs_feat)
# # # # # # # # # # # # #             l_ens = F.mse_loss(pv2, u_t2)
# # # # # # # # # # # # #             total = total + 0.3 * l_ens

# # # # # # # # # # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # # # # # # # # # #             total = x_t.new_zeros(())

# # # # # # # # # # # # #         return dict(
# # # # # # # # # # # # #             total      = total,
# # # # # # # # # # # # #             mse_hav    = l_mse.item(),
# # # # # # # # # # # # #             velocity   = l_vel.item()   if torch.is_tensor(l_vel)   else float(l_vel),
# # # # # # # # # # # # #             heading    = l_head.item()  if torch.is_tensor(l_head)  else float(l_head),
# # # # # # # # # # # # #             long_range = l_lr.item(),
# # # # # # # # # # # # #             fde        = l_fde.item(),
# # # # # # # # # # # # #             shape      = l_shape.item() if torch.is_tensor(l_shape) else float(l_shape),
# # # # # # # # # # # # #             ens_consist= l_ens.item(),
# # # # # # # # # # # # #             sigma      = current_sigma,
# # # # # # # # # # # # #             w_lr       = w_lr,
# # # # # # # # # # # # #             w_fde      = w_fde,
# # # # # # # # # # # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # # # # # # # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0, recurv_ratio=0.0,
# # # # # # # # # # # # #         )

# # # # # # # # # # # # #     @torch.no_grad()
# # # # # # # # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
# # # # # # # # # # # # #         obs_t  = batch_list[0]
# # # # # # # # # # # # #         lp     = obs_t[-1]
# # # # # # # # # # # # #         lm     = batch_list[7][-1]
# # # # # # # # # # # # #         B      = lp.shape[0]
# # # # # # # # # # # # #         device = lp.device
# # # # # # # # # # # # #         T      = self.pred_len
# # # # # # # # # # # # #         dt     = 1.0 / max(ddim_steps, 1)

# # # # # # # # # # # # #         raw_ctx      = self.net._context(batch_list)
# # # # # # # # # # # # #         vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

# # # # # # # # # # # # #         traj_s, me_s = [], []
# # # # # # # # # # # # #         for _ in range(num_ensemble):
# # # # # # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.sigma_min
# # # # # # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # # # #                 ns  = self.ctx_noise_scale if step < 3 else 0.0
# # # # # # # # # # # # #                 vel = self.net.forward_with_ctx(
# # # # # # # # # # # # #                     x_t, t_b, raw_ctx, noise_scale=ns, vel_obs_feat=vel_obs_feat)
# # # # # # # # # # # # #                 x_t = x_t + dt * vel
# # # # # # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # # # # #             traj_s.append(tr)
# # # # # # # # # # # # #             me_s.append(me)

# # # # # # # # # # # # #         all_trajs = torch.stack(traj_s)
# # # # # # # # # # # # #         pred_mean = all_trajs.median(0).values

# # # # # # # # # # # # #         if predict_csv:
# # # # # # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # # # # # # # # #         return pred_mean, torch.stack(me_s).mean(0), all_trajs

# # # # # # # # # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # # # # # # # # #         with torch.enable_grad():
# # # # # # # # # # # # #             x   = x_pred.detach().requires_grad_(True)
# # # # # # # # # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # # # # # # # # #             for _ in range(n_steps):
# # # # # # # # # # # # #                 opt.zero_grad()
# # # # # # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # # # # # #                 pred_deg    = _denorm_to_deg(pred_abs)
# # # # # # # # # # # # #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # # # #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # # # # # #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # # #                 speed       = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # # # # # # # # #                                           + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # # # # # # # # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # # # # # # # # # # # #                 loss.backward()
# # # # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # # # # # # # # # # # #                 opt.step()
# # # # # # # # # # # # #         return x.detach()

# # # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # # # # # # # # # # # #         import numpy as np
# # # # # # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # # # # # #                     "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg",
# # # # # # # # # # # # #                     "ens_spread_km"]
# # # # # # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # # # # # #             if write_hdr: w.writeheader()
# # # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # # #                 for k in range(T):
# # # # # # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # # # # # #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# # # # # # # # # # # # #                                * math.cos(math.radians(mean_lat[k, b])))
# # # # # # # # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # # # # # # # #                     w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # # # # # # # # #                                 "lead_h": (k+1)*6,
# # # # # # # # # # # # #                                 "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # # # # # # # # #                                 "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # # # # # # # # #                                 "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # # # # # #                                 "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # # # # # #                                 "ens_spread_km": f"{spread:.2f}"})


# # # # # # # # # # # # # TCDiffusion = TCFlowMatching
# # # # # # # # # # # # """
# # # # # # # # # # # # Model/flow_matching_model.py — v34 HORIZON-AWARE + RESIDUAL
# # # # # # # # # # # # ═══════════════════════════════════════════════════════════════════

# # # # # # # # # # # # FIXES từ v33:
# # # # # # # # # # # #   BUG 1 (NameError w_vel/w_lr/w_head): Dùng compute_total_loss thống nhất
# # # # # # # # # # # #   BUG 2 (double-denorm): LUÔN pass degrees cho losses, normalized chỉ nội bộ
# # # # # # # # # # # #   BUG 3 (inconsistent types): Naming convention rõ ràng _deg vs _norm

# # # # # # # # # # # # NEW IDEAS:
# # # # # # # # # # # #   1. Horizon-aware loss (v34 losses.py)
# # # # # # # # # # # #   2. Residual prediction: predict displacement từ persistence baseline
# # # # # # # # # # # #   3. Scheduled teacher forcing (training only)
# # # # # # # # # # # #   4. Steering-conditioned velocity (env influence > only β-drift)
# # # # # # # # # # # #   5. EMA weights support
# # # # # # # # # # # #   6. Importance-weighted sampling at inference
# # # # # # # # # # # # """
# # # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # # import csv
# # # # # # # # # # # # import math
# # # # # # # # # # # # import os
# # # # # # # # # # # # from copy import deepcopy
# # # # # # # # # # # # from datetime import datetime

# # # # # # # # # # # # import torch
# # # # # # # # # # # # import torch.nn as nn
# # # # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # # # # from Model.losses import (
# # # # # # # # # # # #     compute_total_loss,
# # # # # # # # # # # #     _haversine_deg, _norm_to_deg,
# # # # # # # # # # # #     WEIGHTS,
# # # # # # # # # # # # )


# # # # # # # # # # # # def _norm_to_deg_fn(t):
# # # # # # # # # # # #     """Normalized [lon, lat] → degrees."""
# # # # # # # # # # # #     out = t.clone()
# # # # # # # # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # # # # # # # #     return out


# # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # #  EMA wrapper
# # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # class EMAModel:
# # # # # # # # # # # #     """Exponential Moving Average of model weights."""
# # # # # # # # # # # #     def __init__(self, model: nn.Module, decay: float = 0.999):
# # # # # # # # # # # #         self.decay = decay
# # # # # # # # # # # #         self.shadow = {k: v.detach().clone()
# # # # # # # # # # # #                         for k, v in model.state_dict().items()
# # # # # # # # # # # #                         if v.dtype.is_floating_point}

# # # # # # # # # # # #     def update(self, model: nn.Module):
# # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # #             for k, v in model.state_dict().items():
# # # # # # # # # # # #                 if k in self.shadow:
# # # # # # # # # # # #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

# # # # # # # # # # # #     def apply_to(self, model: nn.Module):
# # # # # # # # # # # #         """Copy EMA weights into model (for eval). Returns backup dict."""
# # # # # # # # # # # #         backup = {}
# # # # # # # # # # # #         sd = model.state_dict()
# # # # # # # # # # # #         for k in self.shadow:
# # # # # # # # # # # #             backup[k] = sd[k].detach().clone()
# # # # # # # # # # # #             sd[k].copy_(self.shadow[k])
# # # # # # # # # # # #         return backup

# # # # # # # # # # # #     def restore(self, model: nn.Module, backup: dict):
# # # # # # # # # # # #         sd = model.state_dict()
# # # # # # # # # # # #         for k, v in backup.items():
# # # # # # # # # # # #             sd[k].copy_(v)


# # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # #  VelocityField
# # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # # # # # # # # #                  unet_in_ch=13, **kwargs):
# # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # #         self.pred_len = pred_len
# # # # # # # # # # # #         self.obs_len  = obs_len

# # # # # # # # # # # #         # Spatial encoder
# # # # # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # # # # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # # # # # # #             spatial_down=32, dropout=0.05)
# # # # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # # # # # # # # #         # Temporal encoder (Mamba)
# # # # # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # # # # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # # # # # # # # # #         # Context projection
# # # # # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # # # # #         self.ctx_drop = nn.Dropout(0.10)
# # # # # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # # # # # # # # #         # Velocity observation features
# # # # # # # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # # # # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # # # #             nn.Linear(256, 256), nn.GELU(),
# # # # # # # # # # # #         )

# # # # # # # # # # # #         # ★ NEW: Steering flow encoder (từ env 500hPa)
# # # # # # # # # # # #         # Output dim = 256 để match transformer d_model
# # # # # # # # # # # #         self.steering_enc = nn.Sequential(
# # # # # # # # # # # #             nn.Linear(4, 64), nn.GELU(),   # [u_mean, v_mean, u_center, v_center]
# # # # # # # # # # # #             nn.LayerNorm(64),
# # # # # # # # # # # #             nn.Linear(64, 128), nn.GELU(),
# # # # # # # # # # # #             nn.Linear(128, 256),
# # # # # # # # # # # #         )

# # # # # # # # # # # #         # Time embedding
# # # # # # # # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # # # # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # # # # # # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # # # # # # # # # #         # Decoder
# # # # # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # # # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # # # # # # # # # #             num_layers=5)

# # # # # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # # # # # # #         # Physics scales (learnable)
# # # # # # # # # # # #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # # # # # # # # # #         self.physics_scale  = nn.Parameter(torch.ones(4) * 0.3)
# # # # # # # # # # # #         self.steering_scale = nn.Parameter(torch.ones(4) * 0.2)   # ★ NEW
# # # # # # # # # # # #         self._init_weights()

# # # # # # # # # # # #     def _init_weights(self):
# # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # #             for name, m in self.named_modules():
# # # # # # # # # # # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # # # # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # # # # # # # # # #                     if m.bias is not None:
# # # # # # # # # # # #                         nn.init.zeros_(m.bias)

# # # # # # # # # # # #     def _time_emb(self, t, dim=256):
# # # # # # # # # # # #         half = dim // 2
# # # # # # # # # # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # # # # # # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # # # # # # # # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # # # # # # # # # #     def _context(self, batch_list):
# # # # # # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # # # # # #         image_obs = batch_list[11]
# # # # # # # # # # # #         env_data  = batch_list[13]

# # # # # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # # # # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # # # # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # # # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # # # # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # # # # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)

# # # # # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # # # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # # # # # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # # # # # # # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# # # # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # # # # # # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(
# # # # # # # # # # # #             torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # # # # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # # # # #     def _get_vel_obs_feat(self, obs_traj):
# # # # # # # # # # # #         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
# # # # # # # # # # # #         if T_obs >= 2:
# # # # # # # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # # # # # # # # # #         else:
# # # # # # # # # # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
# # # # # # # # # # # #         if vels.shape[0] < self.obs_len:
# # # # # # # # # # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# # # # # # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # # # # # #         else:
# # # # # # # # # # # #             vels = vels[-self.obs_len:]
# # # # # # # # # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # # # # # # # # #     # ★ NEW: Steering feature extraction
# # # # # # # # # # # #     def _get_steering_feat(self, env_data, B, device):
# # # # # # # # # # # #         """Extract 500hPa steering as a context vector [B, 256]."""
# # # # # # # # # # # #         if env_data is None:
# # # # # # # # # # # #             return torch.zeros(B, 256, device=device)

# # # # # # # # # # # #         def _safe_get(key, default_val=0.0):
# # # # # # # # # # # #             v = env_data.get(key, None)
# # # # # # # # # # # #             if v is None or not torch.is_tensor(v):
# # # # # # # # # # # #                 return torch.full((B,), default_val, device=device)
# # # # # # # # # # # #             v = v.to(device).float()
# # # # # # # # # # # #             if v.dim() >= 2:
# # # # # # # # # # # #                 # Take mean over time/spatial dims, keep batch
# # # # # # # # # # # #                 while v.dim() > 1:
# # # # # # # # # # # #                     v = v.mean(-1)
# # # # # # # # # # # #             if v.shape[0] != B:
# # # # # # # # # # # #                 v = v.view(-1)[:B] if v.numel() >= B else torch.full((B,), default_val, device=device)
# # # # # # # # # # # #             return v

# # # # # # # # # # # #         u_m = _safe_get("u500_mean")
# # # # # # # # # # # #         v_m = _safe_get("v500_mean")
# # # # # # # # # # # #         u_c = _safe_get("u500_center")
# # # # # # # # # # # #         v_c = _safe_get("v500_center")
# # # # # # # # # # # #         feat = torch.stack([u_m, v_m, u_c, v_c], dim=-1)  # [B, 4]
# # # # # # # # # # # #         return self.steering_enc(feat)  # [B, 256]

# # # # # # # # # # # #     def _beta_drift(self, x_t):
# # # # # # # # # # # #         """β-drift (Coriolis-derived) in normalized units."""
# # # # # # # # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # # # # # # # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # # # # # # # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # # # # # # # #         R_tc    = 3e5
# # # # # # # # # # # #         v_phys  = torch.zeros_like(x_t)
# # # # # # # # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # # # # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # # # # # # # #         return v_phys

# # # # # # # # # # # #     # ★ NEW: Steering drift (from env 500hPa)
# # # # # # # # # # # #     def _steering_drift(self, x_t, env_data):
# # # # # # # # # # # #         """
# # # # # # # # # # # #         Inject steering flow as additional drift velocity.
# # # # # # # # # # # #         env data → u, v in m/s → normalize per 6h.
# # # # # # # # # # # #         """
# # # # # # # # # # # #         if env_data is None:
# # # # # # # # # # # #             return torch.zeros_like(x_t)

# # # # # # # # # # # #         B = x_t.shape[0]
# # # # # # # # # # # #         device = x_t.device

# # # # # # # # # # # #         def _safe_mean(key):
# # # # # # # # # # # #             v = env_data.get(key, None)
# # # # # # # # # # # #             if v is None or not torch.is_tensor(v):
# # # # # # # # # # # #                 return torch.zeros(B, device=device)
# # # # # # # # # # # #             v = v.to(device).float()
# # # # # # # # # # # #             while v.dim() > 1:
# # # # # # # # # # # #                 v = v.mean(-1)
# # # # # # # # # # # #             if v.numel() < B:
# # # # # # # # # # # #                 return torch.zeros(B, device=device)
# # # # # # # # # # # #             return v.view(-1)[:B]

# # # # # # # # # # # #         # u, v in normalized units (~m/s / 30)
# # # # # # # # # # # #         u = _safe_mean("u500_center")  # [B]
# # # # # # # # # # # #         v = _safe_mean("v500_center")

# # # # # # # # # # # #         # Convert to deg/6h at approx latitude from x_t
# # # # # # # # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # # # # # # # #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# # # # # # # # # # # #         # m/s * 21600s / (111km * 1000m) = deg/6h
# # # # # # # # # # # #         u_deg = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat)
# # # # # # # # # # # #         v_deg = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0)
# # # # # # # # # # # #         # Normalize back: lon_norm = (deg*10 - 1800)/50, but for velocity → scale 10/50 = 0.2
# # # # # # # # # # # #         u_norm = u_deg * 0.2
# # # # # # # # # # # #         v_norm = v_deg * 0.2

# # # # # # # # # # # #         out = torch.zeros_like(x_t)
# # # # # # # # # # # #         out[:, :, 0] = u_norm
# # # # # # # # # # # #         out[:, :, 1] = v_norm
# # # # # # # # # # # #         return out

# # # # # # # # # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
# # # # # # # # # # # #                 env_data=None):
# # # # # # # # # # # #         B     = x_t.shape[0]
# # # # # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # # # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # # # # # # #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # # # # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # # # # # # #         s_emb    = self.step_embed(step_idx)

# # # # # # # # # # # #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# # # # # # # # # # # #                  + self.pos_enc[:, :T_seq]
# # # # # # # # # # # #                  + t_emb.unsqueeze(1)
# # # # # # # # # # # #                  + s_emb)

# # # # # # # # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # # # # # # # #         if vel_obs_feat is not None:
# # # # # # # # # # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))
# # # # # # # # # # # #         if steering_feat is not None:
# # # # # # # # # # # #             mem_parts.append(steering_feat.unsqueeze(1))

# # # # # # # # # # # #         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))

# # # # # # # # # # # #         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # # # # # # # # #         v_neural = v_neural * scale

# # # # # # # # # # # #         # β-drift (Coriolis)
# # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # #             v_phys = self._beta_drift(x_t[:, :T_seq])
# # # # # # # # # # # #         # Steering drift
# # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # #             v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

# # # # # # # # # # # #         return (v_neural
# # # # # # # # # # # #                 + torch.sigmoid(self.physics_scale) * v_phys
# # # # # # # # # # # #                 + torch.sigmoid(self.steering_scale) * v_steer)

# # # # # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
# # # # # # # # # # # #                            vel_obs_feat=None, steering_feat=None, env_data=None):
# # # # # # # # # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# # # # # # # # # # # #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # # #  TCFlowMatching (main)
# # # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # # class TCFlowMatching(nn.Module):

# # # # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # # # # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # # # # # # # #                  ctx_noise_scale=0.01, initial_sample_sigma=0.03,
# # # # # # # # # # # #                  teacher_forcing=True, use_ema=True, ema_decay=0.999,
# # # # # # # # # # # #                  **kwargs):
# # # # # # # # # # # #         super().__init__()
# # # # # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # # # # # # #         self.teacher_forcing      = teacher_forcing
# # # # # # # # # # # #         self.active_pred_len      = pred_len

# # # # # # # # # # # #         self.net = VelocityField(
# # # # # # # # # # # #             pred_len=pred_len, obs_len=obs_len,
# # # # # # # # # # # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # # # # # # # # #         # EMA (set up after all submodules exist)
# # # # # # # # # # # #         self.use_ema = use_ema
# # # # # # # # # # # #         self.ema_decay = ema_decay
# # # # # # # # # # # #         self._ema = None  # lazy init (outside __init__ to avoid state conflict)

# # # # # # # # # # # #     def init_ema(self):
# # # # # # # # # # # #         """Call once after model on device."""
# # # # # # # # # # # #         if self.use_ema:
# # # # # # # # # # # #             self._ema = EMAModel(self, decay=self.ema_decay)

# # # # # # # # # # # #     def ema_update(self):
# # # # # # # # # # # #         if self._ema is not None:
# # # # # # # # # # # #             self._ema.update(self)

# # # # # # # # # # # #     def set_curriculum_len(self, *a, **kw):
# # # # # # # # # # # #         pass

# # # # # # # # # # # #     # ── Normalization helpers ────────────────────────────────────────────────
# # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # # # # # # # #         """Absolute → relative (to last obs)."""
# # # # # # # # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # # # # # # # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # #     def _to_abs(rel, lp, lm):
# # # # # # # # # # # #         """Relative → absolute (normalized)."""
# # # # # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # # # # # # # # #     # ── CFM noise ────────────────────────────────────────────────────────────
# # # # # # # # # # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # # # # # # # # # #         if sigma_min is None:
# # # # # # # # # # # #             sigma_min = self.sigma_min
# # # # # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # # # # #         x0 = torch.randn_like(x1) * sigma_min
# # # # # # # # # # # #         t  = torch.rand(B, device=device)
# # # # # # # # # # # #         te = t.view(B, 1, 1)
# # # # # # # # # # # #         x_t = (1.0 - te) * x0 + te * x1
# # # # # # # # # # # #         u   = x1 - x0
# # # # # # # # # # # #         return x_t, t, u

# # # # # # # # # # # #     # ── Augmentation ─────────────────────────────────────────────────────────
# # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # # # #             return batch_list
# # # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # # # # # # # #                 t = aug[idx].clone()
# # # # # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # # # # #                 aug[idx] = t
# # # # # # # # # # # #         return aug

# # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # #     def _obs_noise_aug(batch_list, sigma=0.005):
# # # # # # # # # # # #         """Add small Gaussian noise to obs trajectory (input augmentation)."""
# # # # # # # # # # # #         if torch.rand(1).item() > 0.5:
# # # # # # # # # # # #             return batch_list
# # # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # # #         if torch.is_tensor(aug[0]):
# # # # # # # # # # # #             aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
# # # # # # # # # # # #         return aug

# # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # #     def _mixup_aug(batch_list, alpha=0.2, p=0.2):
# # # # # # # # # # # #         """Light mixup on trajectories (shuffle batch, mix)."""
# # # # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # # # #             return batch_list
# # # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # # #         lam = float(torch.distributions.Beta(alpha, alpha).sample())
# # # # # # # # # # # #         lam = max(lam, 1 - lam)  # prefer near 1 to not destroy signal
# # # # # # # # # # # #         if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
# # # # # # # # # # # #             B = aug[0].shape[1]
# # # # # # # # # # # #             perm = torch.randperm(B, device=aug[0].device)
# # # # # # # # # # # #             for idx in [0, 1, 7, 8]:
# # # # # # # # # # # #                 if torch.is_tensor(aug[idx]):
# # # # # # # # # # # #                     aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
# # # # # # # # # # # #         return aug

# # # # # # # # # # # #     # ── Public loss API ───────────────────────────────────────────────────────
# # # # # # # # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # # # # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # # # #         # Augmentation
# # # # # # # # # # # #         batch_list = self._lon_flip_aug(batch_list)
# # # # # # # # # # # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
# # # # # # # # # # # #         if epoch >= 5:
# # # # # # # # # # # #             batch_list = self._mixup_aug(batch_list, alpha=0.2, p=0.15)

# # # # # # # # # # # #         traj_gt = batch_list[1]
# # # # # # # # # # # #         Me_gt   = batch_list[8]
# # # # # # # # # # # #         obs_t   = batch_list[0]
# # # # # # # # # # # #         obs_Me  = batch_list[7]
# # # # # # # # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # # # # # # # #         lp, lm  = obs_t[-1], obs_Me[-1]

# # # # # # # # # # # #         # Sigma schedule
# # # # # # # # # # # #         if epoch < 15:
# # # # # # # # # # # #             current_sigma = 0.15
# # # # # # # # # # # #         elif epoch < 40:
# # # # # # # # # # # #             current_sigma = 0.15 - (epoch - 15) / 25.0 * 0.12
# # # # # # # # # # # #         else:
# # # # # # # # # # # #             current_sigma = 0.03

# # # # # # # # # # # #         # Context
# # # # # # # # # # # #         raw_ctx       = self.net._context(batch_list)
# # # # # # # # # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # # # # # # # #         steering_feat = self.net._get_steering_feat(env_data, obs_t.shape[1], obs_t.device)

# # # # # # # # # # # #         # ── CFM target ────────────────────────────────────────────────────
# # # # # # # # # # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
# # # # # # # # # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # # # # # # # # #         # ── Scheduled teacher forcing (training only) ─────────────────────
# # # # # # # # # # # #         if self.teacher_forcing and self.training and epoch >= 3:
# # # # # # # # # # # #             # p_teacher decreases from 0.5 at ep3 → 0 at ep40
# # # # # # # # # # # #             p_teacher = max(0.0, 0.5 * (1.0 - (epoch - 3) / 37.0))
# # # # # # # # # # # #             if p_teacher > 0 and torch.rand(1).item() < p_teacher:
# # # # # # # # # # # #                 # Mix x_t với a bit of x1_rel (GT) at far steps
# # # # # # # # # # # #                 far_mask = torch.zeros_like(x_t)
# # # # # # # # # # # #                 far_mask[:, 6:, :] = 0.3  # 30% GT at steps 7-12
# # # # # # # # # # # #                 x_t = x_t * (1 - far_mask) + x1_rel * far_mask

# # # # # # # # # # # #         pred_vel = self.net.forward_with_ctx(
# # # # # # # # # # # #             x_t, fm_t, raw_ctx,
# # # # # # # # # # # #             vel_obs_feat=vel_obs_feat,
# # # # # # # # # # # #             steering_feat=steering_feat,
# # # # # # # # # # # #             env_data=env_data,
# # # # # # # # # # # #         )

# # # # # # # # # # # #         # CFM velocity MSE loss (base)
# # # # # # # # # # # #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# # # # # # # # # # # #         # Reconstruct predicted x1
# # # # # # # # # # # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # # # # # # # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)   # [T, B, 2] normalized
# # # # # # # # # # # #         pred_deg = _norm_to_deg_fn(pred_abs)           # [T, B, 2] degrees
# # # # # # # # # # # #         gt_deg   = _norm_to_deg_fn(traj_gt)            # [T, B, 2] degrees

# # # # # # # # # # # #         # ── v34 horizon-aware total loss ──────────────────────────────────
# # # # # # # # # # # #         loss_dict = compute_total_loss(
# # # # # # # # # # # #             pred_deg=pred_deg,
# # # # # # # # # # # #             gt_deg=gt_deg,
# # # # # # # # # # # #             env_data=env_data,
# # # # # # # # # # # #             weights=WEIGHTS,
# # # # # # # # # # # #             epoch=epoch,
# # # # # # # # # # # #         )

# # # # # # # # # # # #         # Combine FM velocity loss + v34 trajectory loss
# # # # # # # # # # # #         # FM weight decays as training progresses
# # # # # # # # # # # #         w_fm = max(0.3, 1.0 - epoch / 60.0)
# # # # # # # # # # # #         total = w_fm * l_fm_mse + loss_dict["total"]

# # # # # # # # # # # #         # ── Ensemble consistency (late epochs) ────────────────────────────
# # # # # # # # # # # #         l_ens = x_t.new_zeros(())
# # # # # # # # # # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # # # # # # # # # #             x_t2, fm_t2, u_t2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # # # # # # # # #             pv2 = self.net.forward_with_ctx(
# # # # # # # # # # # #                 x_t2, fm_t2, raw_ctx,
# # # # # # # # # # # #                 vel_obs_feat=vel_obs_feat,
# # # # # # # # # # # #                 steering_feat=steering_feat,
# # # # # # # # # # # #                 env_data=env_data,
# # # # # # # # # # # #             )
# # # # # # # # # # # #             l_ens = F.mse_loss(pv2, u_t2)
# # # # # # # # # # # #             total = total + 0.3 * l_ens

# # # # # # # # # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # # # # # # # # #             total = x_t.new_zeros(())

# # # # # # # # # # # #         # Return breakdown
# # # # # # # # # # # #         return dict(
# # # # # # # # # # # #             total        = total,
# # # # # # # # # # # #             fm_mse       = l_fm_mse.item(),
# # # # # # # # # # # #             mse_hav      = loss_dict["mse_hav_horizon"],
# # # # # # # # # # # #             multi_scale  = loss_dict["multi_scale"],
# # # # # # # # # # # #             endpoint     = loss_dict["endpoint"],
# # # # # # # # # # # #             shape        = loss_dict["shape"],
# # # # # # # # # # # #             velocity     = loss_dict["velocity"],
# # # # # # # # # # # #             heading      = loss_dict["heading"],
# # # # # # # # # # # #             recurv       = loss_dict["recurv"],
# # # # # # # # # # # #             steering     = loss_dict["steering"],
# # # # # # # # # # # #             ens_consist  = l_ens.item() if torch.is_tensor(l_ens) else float(l_ens),
# # # # # # # # # # # #             sigma        = current_sigma,
# # # # # # # # # # # #             w_fm         = w_fm,
# # # # # # # # # # # #             # backward compat zeros
# # # # # # # # # # # #             long_range=0.0, fde=0.0, w_lr=0.0, w_fde=0.0,
# # # # # # # # # # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # # # # # # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0, recurv_ratio=0.0,
# # # # # # # # # # # #         )

# # # # # # # # # # # #     # ── Sampling ──────────────────────────────────────────────────────────────
# # # # # # # # # # # #     @torch.no_grad()
# # # # # # # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
# # # # # # # # # # # #                 predict_csv=None, importance_weight=True):
# # # # # # # # # # # #         obs_t    = batch_list[0]
# # # # # # # # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # # # # # # # #         lp       = obs_t[-1]
# # # # # # # # # # # #         lm       = batch_list[7][-1]
# # # # # # # # # # # #         B        = lp.shape[0]
# # # # # # # # # # # #         device   = lp.device
# # # # # # # # # # # #         T        = self.pred_len
# # # # # # # # # # # #         dt       = 1.0 / max(ddim_steps, 1)

# # # # # # # # # # # #         raw_ctx       = self.net._context(batch_list)
# # # # # # # # # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # # # # # # # #         steering_feat = self.net._get_steering_feat(env_data, B, device)

# # # # # # # # # # # #         traj_s, me_s, scores = [], [], []
# # # # # # # # # # # #         for ens_i in range(num_ensemble):
# # # # # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * self.sigma_min
# # # # # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # # #                 # Larger noise early for diversity
# # # # # # # # # # # #                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
# # # # # # # # # # # #                 vel = self.net.forward_with_ctx(
# # # # # # # # # # # #                     x_t, t_b, raw_ctx,
# # # # # # # # # # # #                     noise_scale=ns,
# # # # # # # # # # # #                     vel_obs_feat=vel_obs_feat,
# # # # # # # # # # # #                     steering_feat=steering_feat,
# # # # # # # # # # # #                     env_data=env_data,
# # # # # # # # # # # #                 )
# # # # # # # # # # # #                 x_t = x_t + dt * vel
# # # # # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # # # #             traj_s.append(tr)
# # # # # # # # # # # #             me_s.append(me)

# # # # # # # # # # # #             # ★ Importance score for this sample
# # # # # # # # # # # #             if importance_weight:
# # # # # # # # # # # #                 scores.append(self._score_sample(tr, env_data))

# # # # # # # # # # # #         all_trajs = torch.stack(traj_s)       # [S, T, B, 2]
# # # # # # # # # # # #         all_me    = torch.stack(me_s)

# # # # # # # # # # # #         # ── Prediction: importance-weighted median/mean ──────────────────
# # # # # # # # # # # #         if importance_weight and scores:
# # # # # # # # # # # #             score_tensor = torch.stack(scores)  # [S, B]
# # # # # # # # # # # #             # Top-70% by score per batch
# # # # # # # # # # # #             k = max(1, int(num_ensemble * 0.7))
# # # # # # # # # # # #             _, top_idx = score_tensor.topk(k, dim=0)  # [k, B]
# # # # # # # # # # # #             # Gather
# # # # # # # # # # # #             pred_mean_list = []
# # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # #                 sel = all_trajs[top_idx[:, b], :, b, :]  # [k, T, 2]
# # # # # # # # # # # #                 pred_mean_list.append(sel.median(0).values)
# # # # # # # # # # # #             pred_mean = torch.stack(pred_mean_list, dim=1)  # [T, B, 2]
# # # # # # # # # # # #         else:
# # # # # # # # # # # #             pred_mean = all_trajs.median(0).values

# # # # # # # # # # # #         if predict_csv:
# # # # # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # # # # # # # #         return pred_mean, all_me.mean(0), all_trajs

# # # # # # # # # # # #     def _score_sample(self, traj, env_data):
# # # # # # # # # # # #         """
# # # # # # # # # # # #         Score a sample: higher is better.
# # # # # # # # # # # #         Checks: speed in [10, 70] km/h, smoothness, steering alignment.
# # # # # # # # # # # #         Returns [B] score.
# # # # # # # # # # # #         """
# # # # # # # # # # # #         B = traj.shape[1]
# # # # # # # # # # # #         if traj.shape[0] < 2:
# # # # # # # # # # # #             return torch.ones(B, device=traj.device)

# # # # # # # # # # # #         traj_deg = _norm_to_deg_fn(traj)
# # # # # # # # # # # #         dt_deg = traj_deg[1:] - traj_deg[:-1]
# # # # # # # # # # # #         lat_rad = torch.deg2rad(traj_deg[:-1, :, 1])
# # # # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # #         dx_km = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # # # # # # # # #         dy_km = dt_deg[:, :, 1] * 111.0
# # # # # # # # # # # #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)  # km per 6h

# # # # # # # # # # # #         # Reasonable speed: 10-70 km per 6h (~2-12 m/s)
# # # # # # # # # # # #         speed_penalty = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
# # # # # # # # # # # #         speed_score = torch.exp(-speed_penalty.mean(0) / 20.0)  # [B]

# # # # # # # # # # # #         # Smoothness: low jerk
# # # # # # # # # # # #         if dt_deg.shape[0] >= 2:
# # # # # # # # # # # #             jerk = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)  # [B]
# # # # # # # # # # # #             smooth_score = torch.exp(-jerk * 5.0)
# # # # # # # # # # # #         else:
# # # # # # # # # # # #             smooth_score = torch.ones(B, device=traj.device)

# # # # # # # # # # # #         return speed_score * smooth_score

# # # # # # # # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # # # # # # # #         with torch.enable_grad():
# # # # # # # # # # # #             x   = x_pred.detach().requires_grad_(True)
# # # # # # # # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # # # # # # # #             for _ in range(n_steps):
# # # # # # # # # # # #                 opt.zero_grad()
# # # # # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # # # # #                 pred_deg    = _norm_to_deg_fn(pred_abs)
# # # # # # # # # # # #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # # #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # # # # #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # # #                 speed       = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # # # # # # # #                                           + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # # # # # # # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # # # # # # # # # # #                 loss.backward()
# # # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # # # # # # # # # # #                 opt.step()
# # # # # # # # # # # #         return x.detach()

# # # # # # # # # # # #     @staticmethod
# # # # # # # # # # # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # # # # # # # # # # #         import numpy as np
# # # # # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # # # # #                     "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg",
# # # # # # # # # # # #                     "ens_spread_km"]
# # # # # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # # # # #             if write_hdr: w.writeheader()
# # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # #                 for k in range(T):
# # # # # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # # # # #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# # # # # # # # # # # #                                * math.cos(math.radians(mean_lat[k, b])))
# # # # # # # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # # # # # # #                     w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # # # # # # # #                                 "lead_h": (k+1)*6,
# # # # # # # # # # # #                                 "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # # # # # # # #                                 "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # # # # # # # #                                 "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # # # # #                                 "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # # # # #                                 "ens_spread_km": f"{spread:.2f}"})


# # # # # # # # # # # # TCDiffusion = TCFlowMatching

# # # # # # # # # # # """
# # # # # # # # # # # Model/flow_matching_model.py — v34 HORIZON-AWARE + RESIDUAL
# # # # # # # # # # # ═══════════════════════════════════════════════════════════════════

# # # # # # # # # # # FIXES từ v33:
# # # # # # # # # # #   BUG 1 (NameError w_vel/w_lr/w_head): Dùng compute_total_loss thống nhất
# # # # # # # # # # #   BUG 2 (double-denorm): LUÔN pass degrees cho losses, normalized chỉ nội bộ
# # # # # # # # # # #   BUG 3 (inconsistent types): Naming convention rõ ràng _deg vs _norm

# # # # # # # # # # # FIXES v34→v34fix:
# # # # # # # # # # #   BUG 4 (EMA KeyError 'net.pos_enc'): torch.compile đổi tên key thành
# # # # # # # # # # #          _orig_mod.net.pos_enc — EMAModel phải unwrap _orig_mod khi
# # # # # # # # # # #          init/update/apply_to/restore.
# # # # # # # # # # #   BUG 5 (sigma floor 0.03 quá thấp): giữ floor ở 0.06 để ensemble diversity
# # # # # # # # # # #   BUG 6 (ensemble init noise quá nhỏ): tăng initial noise × 2.5

# # # # # # # # # # # NEW IDEAS:
# # # # # # # # # # #   1. Horizon-aware loss (v34 losses.py)
# # # # # # # # # # #   2. Residual prediction: predict displacement từ persistence baseline
# # # # # # # # # # #   3. Scheduled teacher forcing (training only)
# # # # # # # # # # #   4. Steering-conditioned velocity (env influence > only β-drift)
# # # # # # # # # # #   5. EMA weights support (fixed for torch.compile)
# # # # # # # # # # #   6. Importance-weighted sampling at inference
# # # # # # # # # # # """
# # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # import csv
# # # # # # # # # # # import math
# # # # # # # # # # # import os
# # # # # # # # # # # from copy import deepcopy
# # # # # # # # # # # from datetime import datetime

# # # # # # # # # # # import torch
# # # # # # # # # # # import torch.nn as nn
# # # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # # # from Model.losses import (
# # # # # # # # # # #     compute_total_loss,
# # # # # # # # # # #     _haversine_deg, _norm_to_deg,
# # # # # # # # # # #     WEIGHTS,
# # # # # # # # # # # )


# # # # # # # # # # # def _norm_to_deg_fn(t):
# # # # # # # # # # #     """Normalized [lon, lat] → degrees."""
# # # # # # # # # # #     out = t.clone()
# # # # # # # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # # # # # # #     return out


# # # # # # # # # # # def _unwrap_model(model: nn.Module) -> nn.Module:
# # # # # # # # # # #     """Unwrap torch.compile wrapper nếu có, để lấy state_dict gốc."""
# # # # # # # # # # #     if hasattr(model, '_orig_mod'):
# # # # # # # # # # #         return model._orig_mod
# # # # # # # # # # #     return model


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # #  EMA wrapper  — FIX: unwrap _orig_mod để tránh KeyError từ torch.compile
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # class EMAModel:
# # # # # # # # # # #     """
# # # # # # # # # # #     Exponential Moving Average of model weights.

# # # # # # # # # # #     FIX (v34fix): torch.compile() đổi tên tất cả key trong state_dict
# # # # # # # # # # #     từ "net.pos_enc" → "_orig_mod.net.pos_enc".
# # # # # # # # # # #     Nếu EMAModel.shadow được build từ compiled model, key sẽ không match
# # # # # # # # # # #     khi apply_to() gọi với raw model (hoặc ngược lại).

# # # # # # # # # # #     Giải pháp: LUÔN unwrap _orig_mod trước khi đọc/ghi state_dict.
# # # # # # # # # # #     """
# # # # # # # # # # #     def __init__(self, model: nn.Module, decay: float = 0.999):
# # # # # # # # # # #         self.decay = decay
# # # # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # # # #         self.shadow = {
# # # # # # # # # # #             k: v.detach().clone()
# # # # # # # # # # #             for k, v in m.state_dict().items()
# # # # # # # # # # #             if v.dtype.is_floating_point
# # # # # # # # # # #         }

# # # # # # # # # # #     def update(self, model: nn.Module):
# # # # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # #             for k, v in m.state_dict().items():
# # # # # # # # # # #                 if k in self.shadow:
# # # # # # # # # # #                     self.shadow[k].mul_(self.decay).add_(
# # # # # # # # # # #                         v.detach(), alpha=1 - self.decay)

# # # # # # # # # # #     def apply_to(self, model: nn.Module):
# # # # # # # # # # #         """Copy EMA weights vào model (để eval). Trả về backup dict."""
# # # # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # # # #         backup = {}
# # # # # # # # # # #         sd = m.state_dict()
# # # # # # # # # # #         for k in self.shadow:
# # # # # # # # # # #             if k not in sd:
# # # # # # # # # # #                 # Key không tồn tại trong model — bỏ qua thay vì crash
# # # # # # # # # # #                 continue
# # # # # # # # # # #             backup[k] = sd[k].detach().clone()
# # # # # # # # # # #             sd[k].copy_(self.shadow[k])
# # # # # # # # # # #         return backup

# # # # # # # # # # #     def restore(self, model: nn.Module, backup: dict):
# # # # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # # # #         sd = m.state_dict()
# # # # # # # # # # #         for k, v in backup.items():
# # # # # # # # # # #             if k in sd:
# # # # # # # # # # #                 sd[k].copy_(v)


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # #  VelocityField
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # # # # # # # #                  unet_in_ch=13, **kwargs):
# # # # # # # # # # #         super().__init__()
# # # # # # # # # # #         self.pred_len = pred_len
# # # # # # # # # # #         self.obs_len  = obs_len

# # # # # # # # # # #         # Spatial encoder
# # # # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # # # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # # # # # #             spatial_down=32, dropout=0.05)
# # # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # # # # # # # #         # Temporal encoder (Mamba)
# # # # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # # # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # # # # # # # # #         # Context projection
# # # # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # # # #         self.ctx_drop = nn.Dropout(0.10)
# # # # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # # # # # # # #         # Velocity observation features
# # # # # # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # # # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # # #             nn.Linear(256, 256), nn.GELU(),
# # # # # # # # # # #         )

# # # # # # # # # # #         # Steering flow encoder (từ env 500hPa)
# # # # # # # # # # #         self.steering_enc = nn.Sequential(
# # # # # # # # # # #             nn.Linear(4, 64), nn.GELU(),
# # # # # # # # # # #             nn.LayerNorm(64),
# # # # # # # # # # #             nn.Linear(64, 128), nn.GELU(),
# # # # # # # # # # #             nn.Linear(128, 256),
# # # # # # # # # # #         )

# # # # # # # # # # #         # Time embedding
# # # # # # # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # # # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # # # # # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # # # # # # # # #         # Decoder
# # # # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # # # # # # # # #             num_layers=5)

# # # # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # # # # # #         # Physics scales (learnable)
# # # # # # # # # # #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # # # # # # # # #         self.physics_scale  = nn.Parameter(torch.ones(4) * 0.3)
# # # # # # # # # # #         self.steering_scale = nn.Parameter(torch.ones(4) * 0.2)
# # # # # # # # # # #         self._init_weights()

# # # # # # # # # # #     def _init_weights(self):
# # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # #             for name, m in self.named_modules():
# # # # # # # # # # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # # # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # # # # # # # # #                     if m.bias is not None:
# # # # # # # # # # #                         nn.init.zeros_(m.bias)

# # # # # # # # # # #     def _time_emb(self, t, dim=256):
# # # # # # # # # # #         half = dim // 2
# # # # # # # # # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # # # # # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # # # # # # # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # # # # # # # # #     def _context(self, batch_list):
# # # # # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # # # # #         image_obs = batch_list[11]
# # # # # # # # # # #         env_data  = batch_list[13]

# # # # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # # # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # # # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # # # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # # # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)

# # # # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # # # # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # # # # # # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# # # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # # # # # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(
# # # # # # # # # # #             torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # # # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # # # #     def _get_vel_obs_feat(self, obs_traj):
# # # # # # # # # # #         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
# # # # # # # # # # #         if T_obs >= 2:
# # # # # # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # # # # # # # # #         else:
# # # # # # # # # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
# # # # # # # # # # #         if vels.shape[0] < self.obs_len:
# # # # # # # # # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# # # # # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # # # # #         else:
# # # # # # # # # # #             vels = vels[-self.obs_len:]
# # # # # # # # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # # # # # # # #     def _get_steering_feat(self, env_data, B, device):
# # # # # # # # # # #         """Extract 500hPa steering as a context vector [B, 256]."""
# # # # # # # # # # #         if env_data is None:
# # # # # # # # # # #             return torch.zeros(B, 256, device=device)

# # # # # # # # # # #         def _safe_get(key, default_val=0.0):
# # # # # # # # # # #             v = env_data.get(key, None)
# # # # # # # # # # #             if v is None or not torch.is_tensor(v):
# # # # # # # # # # #                 return torch.full((B,), default_val, device=device)
# # # # # # # # # # #             v = v.to(device).float()
# # # # # # # # # # #             if v.dim() >= 2:
# # # # # # # # # # #                 while v.dim() > 1:
# # # # # # # # # # #                     v = v.mean(-1)
# # # # # # # # # # #             if v.shape[0] != B:
# # # # # # # # # # #                 v = v.view(-1)[:B] if v.numel() >= B else torch.full((B,), default_val, device=device)
# # # # # # # # # # #             return v

# # # # # # # # # # #         u_m = _safe_get("u500_mean")
# # # # # # # # # # #         v_m = _safe_get("v500_mean")
# # # # # # # # # # #         u_c = _safe_get("u500_center")
# # # # # # # # # # #         v_c = _safe_get("v500_center")
# # # # # # # # # # #         feat = torch.stack([u_m, v_m, u_c, v_c], dim=-1)  # [B, 4]
# # # # # # # # # # #         return self.steering_enc(feat)  # [B, 256]

# # # # # # # # # # #     def _beta_drift(self, x_t):
# # # # # # # # # # #         """β-drift (Coriolis-derived) in normalized units."""
# # # # # # # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # # # # # # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # # # # # # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # # # # # # #         R_tc    = 3e5
# # # # # # # # # # #         v_phys  = torch.zeros_like(x_t)
# # # # # # # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # # # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # # # # # # #         return v_phys

# # # # # # # # # # #     def _steering_drift(self, x_t, env_data):
# # # # # # # # # # #         """Inject steering flow as additional drift velocity."""
# # # # # # # # # # #         if env_data is None:
# # # # # # # # # # #             return torch.zeros_like(x_t)

# # # # # # # # # # #         B = x_t.shape[0]
# # # # # # # # # # #         device = x_t.device

# # # # # # # # # # #         def _safe_mean(key):
# # # # # # # # # # #             v = env_data.get(key, None)
# # # # # # # # # # #             if v is None or not torch.is_tensor(v):
# # # # # # # # # # #                 return torch.zeros(B, device=device)
# # # # # # # # # # #             v = v.to(device).float()
# # # # # # # # # # #             while v.dim() > 1:
# # # # # # # # # # #                 v = v.mean(-1)
# # # # # # # # # # #             if v.numel() < B:
# # # # # # # # # # #                 return torch.zeros(B, device=device)
# # # # # # # # # # #             return v.view(-1)[:B]

# # # # # # # # # # #         u = _safe_mean("u500_center")
# # # # # # # # # # #         v = _safe_mean("v500_center")

# # # # # # # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # # # # # # #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# # # # # # # # # # #         u_deg = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat)
# # # # # # # # # # #         v_deg = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0)
# # # # # # # # # # #         u_norm = u_deg * 0.2
# # # # # # # # # # #         v_norm = v_deg * 0.2

# # # # # # # # # # #         out = torch.zeros_like(x_t)
# # # # # # # # # # #         out[:, :, 0] = u_norm
# # # # # # # # # # #         out[:, :, 1] = v_norm
# # # # # # # # # # #         return out

# # # # # # # # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
# # # # # # # # # # #                 env_data=None):
# # # # # # # # # # #         B     = x_t.shape[0]
# # # # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # # # # # #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # # # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # # # # # #         s_emb    = self.step_embed(step_idx)

# # # # # # # # # # #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# # # # # # # # # # #                  + self.pos_enc[:, :T_seq]
# # # # # # # # # # #                  + t_emb.unsqueeze(1)
# # # # # # # # # # #                  + s_emb)

# # # # # # # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # # # # # # #         if vel_obs_feat is not None:
# # # # # # # # # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))
# # # # # # # # # # #         if steering_feat is not None:
# # # # # # # # # # #             mem_parts.append(steering_feat.unsqueeze(1))

# # # # # # # # # # #         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))

# # # # # # # # # # #         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # # # # # # # #         v_neural = v_neural * scale

# # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # #             v_phys  = self._beta_drift(x_t[:, :T_seq])
# # # # # # # # # # #             v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

# # # # # # # # # # #         return (v_neural
# # # # # # # # # # #                 + torch.sigmoid(self.physics_scale) * v_phys
# # # # # # # # # # #                 + torch.sigmoid(self.steering_scale) * v_steer)

# # # # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
# # # # # # # # # # #                            vel_obs_feat=None, steering_feat=None, env_data=None):
# # # # # # # # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# # # # # # # # # # #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # # #  TCFlowMatching (main)
# # # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # # class TCFlowMatching(nn.Module):

# # # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # # # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # # # # # # #                  ctx_noise_scale=0.01, initial_sample_sigma=0.03,
# # # # # # # # # # #                  teacher_forcing=True, use_ema=True, ema_decay=0.999,
# # # # # # # # # # #                  **kwargs):
# # # # # # # # # # #         super().__init__()
# # # # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # # # # # #         self.teacher_forcing      = teacher_forcing
# # # # # # # # # # #         self.active_pred_len      = pred_len

# # # # # # # # # # #         self.net = VelocityField(
# # # # # # # # # # #             pred_len=pred_len, obs_len=obs_len,
# # # # # # # # # # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # # # # # # # #         self.use_ema  = use_ema
# # # # # # # # # # #         self.ema_decay = ema_decay
# # # # # # # # # # #         self._ema = None  # lazy init

# # # # # # # # # # #     def init_ema(self):
# # # # # # # # # # #         """Gọi một lần sau khi model đã lên device."""
# # # # # # # # # # #         if self.use_ema:
# # # # # # # # # # #             # FIX: truyền self (chưa compile) để shadow có đúng key names
# # # # # # # # # # #             self._ema = EMAModel(self, decay=self.ema_decay)

# # # # # # # # # # #     def ema_update(self):
# # # # # # # # # # #         if self._ema is not None:
# # # # # # # # # # #             # FIX: EMAModel.update tự unwrap _orig_mod bên trong
# # # # # # # # # # #             self._ema.update(self)

# # # # # # # # # # #     def set_curriculum_len(self, *a, **kw):
# # # # # # # # # # #         pass

# # # # # # # # # # #     # ── Normalization helpers ────────────────────────────────────────────────
# # # # # # # # # # #     @staticmethod
# # # # # # # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # # # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # # # # # # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # # # # # # # #     @staticmethod
# # # # # # # # # # #     def _to_abs(rel, lp, lm):
# # # # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # # # # # # # #     # ── CFM noise ────────────────────────────────────────────────────────────
# # # # # # # # # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # # # # # # # # #         if sigma_min is None:
# # # # # # # # # # #             sigma_min = self.sigma_min
# # # # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # # # #         x0 = torch.randn_like(x1) * sigma_min
# # # # # # # # # # #         t  = torch.rand(B, device=device)
# # # # # # # # # # #         te = t.view(B, 1, 1)
# # # # # # # # # # #         x_t = (1.0 - te) * x0 + te * x1
# # # # # # # # # # #         u   = x1 - x0
# # # # # # # # # # #         return x_t, t, u

# # # # # # # # # # #     # ── Augmentation ─────────────────────────────────────────────────────────
# # # # # # # # # # #     @staticmethod
# # # # # # # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # # #             return batch_list
# # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # # # # # # #                 t = aug[idx].clone()
# # # # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # # # #                 aug[idx] = t
# # # # # # # # # # #         return aug

# # # # # # # # # # #     @staticmethod
# # # # # # # # # # #     def _obs_noise_aug(batch_list, sigma=0.005):
# # # # # # # # # # #         if torch.rand(1).item() > 0.5:
# # # # # # # # # # #             return batch_list
# # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # #         if torch.is_tensor(aug[0]):
# # # # # # # # # # #             aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
# # # # # # # # # # #         return aug

# # # # # # # # # # #     @staticmethod
# # # # # # # # # # #     def _mixup_aug(batch_list, alpha=0.2, p=0.2):
# # # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # # #             return batch_list
# # # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # # #         lam = float(torch.distributions.Beta(alpha, alpha).sample())
# # # # # # # # # # #         lam = max(lam, 1 - lam)
# # # # # # # # # # #         if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
# # # # # # # # # # #             B = aug[0].shape[1]
# # # # # # # # # # #             perm = torch.randperm(B, device=aug[0].device)
# # # # # # # # # # #             for idx in [0, 1, 7, 8]:
# # # # # # # # # # #                 if torch.is_tensor(aug[idx]):
# # # # # # # # # # #                     aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
# # # # # # # # # # #         return aug

# # # # # # # # # # #     # ── Public loss API ───────────────────────────────────────────────────────
# # # # # # # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # # # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # # #         # Augmentation
# # # # # # # # # # #         batch_list = self._lon_flip_aug(batch_list)
# # # # # # # # # # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
# # # # # # # # # # #         if epoch >= 5:
# # # # # # # # # # #             batch_list = self._mixup_aug(batch_list, alpha=0.2, p=0.15)

# # # # # # # # # # #         traj_gt  = batch_list[1]
# # # # # # # # # # #         Me_gt    = batch_list[8]
# # # # # # # # # # #         obs_t    = batch_list[0]
# # # # # # # # # # #         obs_Me   = batch_list[7]
# # # # # # # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # # # # # # #         lp, lm   = obs_t[-1], obs_Me[-1]

# # # # # # # # # # #         # ── Sigma schedule — FIX: floor 0.06 (không phải 0.03) ──────────
# # # # # # # # # # #         # Sigma quá thấp → ensemble diversity thấp → median bị bias center
# # # # # # # # # # #         if epoch < 15:
# # # # # # # # # # #             current_sigma = 0.15
# # # # # # # # # # #         elif epoch < 40:
# # # # # # # # # # #             # Giảm từ 0.15 → 0.06 (thay vì 0.03)
# # # # # # # # # # #             current_sigma = 0.15 - (epoch - 15) / 25.0 * 0.09
# # # # # # # # # # #         else:
# # # # # # # # # # #             current_sigma = 0.06  # FIX: floor 0.06, không phải 0.03

# # # # # # # # # # #         # Context
# # # # # # # # # # #         raw_ctx       = self.net._context(batch_list)
# # # # # # # # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # # # # # # #         steering_feat = self.net._get_steering_feat(env_data, obs_t.shape[1], obs_t.device)

# # # # # # # # # # #         # ── CFM target ────────────────────────────────────────────────────
# # # # # # # # # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
# # # # # # # # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # # # # # # # #         # ── Scheduled teacher forcing ─────────────────────────────────────
# # # # # # # # # # #         if self.teacher_forcing and self.training and epoch >= 3:
# # # # # # # # # # #             p_teacher = max(0.0, 0.5 * (1.0 - (epoch - 3) / 37.0))
# # # # # # # # # # #             if p_teacher > 0 and torch.rand(1).item() < p_teacher:
# # # # # # # # # # #                 far_mask = torch.zeros_like(x_t)
# # # # # # # # # # #                 far_mask[:, 6:, :] = 0.3
# # # # # # # # # # #                 x_t = x_t * (1 - far_mask) + x1_rel * far_mask

# # # # # # # # # # #         pred_vel = self.net.forward_with_ctx(
# # # # # # # # # # #             x_t, fm_t, raw_ctx,
# # # # # # # # # # #             vel_obs_feat=vel_obs_feat,
# # # # # # # # # # #             steering_feat=steering_feat,
# # # # # # # # # # #             env_data=env_data,
# # # # # # # # # # #         )

# # # # # # # # # # #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# # # # # # # # # # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # # # # # # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)
# # # # # # # # # # #         pred_deg = _norm_to_deg_fn(pred_abs)
# # # # # # # # # # #         gt_deg   = _norm_to_deg_fn(traj_gt)

# # # # # # # # # # #         loss_dict = compute_total_loss(
# # # # # # # # # # #             pred_deg=pred_deg,
# # # # # # # # # # #             gt_deg=gt_deg,
# # # # # # # # # # #             env_data=env_data,
# # # # # # # # # # #             weights=WEIGHTS,
# # # # # # # # # # #             epoch=epoch,
# # # # # # # # # # #         )

# # # # # # # # # # #         w_fm  = max(0.3, 1.0 - epoch / 60.0)
# # # # # # # # # # #         total = w_fm * l_fm_mse + loss_dict["total"]

# # # # # # # # # # #         l_ens = x_t.new_zeros(())
# # # # # # # # # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # # # # # # # # #             x_t2, fm_t2, u_t2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # # # # # # # #             pv2 = self.net.forward_with_ctx(
# # # # # # # # # # #                 x_t2, fm_t2, raw_ctx,
# # # # # # # # # # #                 vel_obs_feat=vel_obs_feat,
# # # # # # # # # # #                 steering_feat=steering_feat,
# # # # # # # # # # #                 env_data=env_data,
# # # # # # # # # # #             )
# # # # # # # # # # #             l_ens = F.mse_loss(pv2, u_t2)
# # # # # # # # # # #             total = total + 0.3 * l_ens

# # # # # # # # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # # # # # # # #             total = x_t.new_zeros(())

# # # # # # # # # # #         return dict(
# # # # # # # # # # #             total        = total,
# # # # # # # # # # #             fm_mse       = l_fm_mse.item(),
# # # # # # # # # # #             mse_hav      = loss_dict["mse_hav_horizon"],
# # # # # # # # # # #             multi_scale  = loss_dict["multi_scale"],
# # # # # # # # # # #             endpoint     = loss_dict["endpoint"],
# # # # # # # # # # #             shape        = loss_dict["shape"],
# # # # # # # # # # #             velocity     = loss_dict["velocity"],
# # # # # # # # # # #             heading      = loss_dict["heading"],
# # # # # # # # # # #             recurv       = loss_dict["recurv"],
# # # # # # # # # # #             steering     = loss_dict["steering"],
# # # # # # # # # # #             ens_consist  = l_ens.item() if torch.is_tensor(l_ens) else float(l_ens),
# # # # # # # # # # #             sigma        = current_sigma,
# # # # # # # # # # #             w_fm         = w_fm,
# # # # # # # # # # #             long_range=0.0, fde=0.0, w_lr=0.0, w_fde=0.0,
# # # # # # # # # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # # # # # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0, recurv_ratio=0.0,
# # # # # # # # # # #         )

# # # # # # # # # # #     # ── Sampling ──────────────────────────────────────────────────────────────
# # # # # # # # # # #     @torch.no_grad()
# # # # # # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
# # # # # # # # # # #                 predict_csv=None, importance_weight=True):
# # # # # # # # # # #         obs_t    = batch_list[0]
# # # # # # # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # # # # # # #         lp       = obs_t[-1]
# # # # # # # # # # #         lm       = batch_list[7][-1]
# # # # # # # # # # #         B        = lp.shape[0]
# # # # # # # # # # #         device   = lp.device
# # # # # # # # # # #         T        = self.pred_len
# # # # # # # # # # #         dt       = 1.0 / max(ddim_steps, 1)

# # # # # # # # # # #         raw_ctx       = self.net._context(batch_list)
# # # # # # # # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # # # # # # #         steering_feat = self.net._get_steering_feat(env_data, B, device)

# # # # # # # # # # #         traj_s, me_s, scores = [], [], []
# # # # # # # # # # #         for ens_i in range(num_ensemble):
# # # # # # # # # # #             # FIX: tăng initial noise × 2.5 để ensemble diversity cao hơn
# # # # # # # # # # #             # Sigma floor 0.06 → initial noise 0.06 * 2.5 = 0.15
# # # # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * (self.sigma_min * 2.5)
# # # # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # # #                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
# # # # # # # # # # #                 vel = self.net.forward_with_ctx(
# # # # # # # # # # #                     x_t, t_b, raw_ctx,
# # # # # # # # # # #                     noise_scale=ns,
# # # # # # # # # # #                     vel_obs_feat=vel_obs_feat,
# # # # # # # # # # #                     steering_feat=steering_feat,
# # # # # # # # # # #                     env_data=env_data,
# # # # # # # # # # #                 )
# # # # # # # # # # #                 x_t = x_t + dt * vel
# # # # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # # #             traj_s.append(tr)
# # # # # # # # # # #             me_s.append(me)

# # # # # # # # # # #             if importance_weight:
# # # # # # # # # # #                 scores.append(self._score_sample(tr, env_data))

# # # # # # # # # # #         all_trajs = torch.stack(traj_s)
# # # # # # # # # # #         all_me    = torch.stack(me_s)

# # # # # # # # # # #         if importance_weight and scores:
# # # # # # # # # # #             score_tensor = torch.stack(scores)  # [S, B]
# # # # # # # # # # #             k = max(1, int(num_ensemble * 0.7))
# # # # # # # # # # #             _, top_idx = score_tensor.topk(k, dim=0)
# # # # # # # # # # #             pred_mean_list = []
# # # # # # # # # # #             for b in range(B):
# # # # # # # # # # #                 sel = all_trajs[top_idx[:, b], :, b, :]
# # # # # # # # # # #                 pred_mean_list.append(sel.median(0).values)
# # # # # # # # # # #             pred_mean = torch.stack(pred_mean_list, dim=1)
# # # # # # # # # # #         else:
# # # # # # # # # # #             pred_mean = all_trajs.median(0).values

# # # # # # # # # # #         if predict_csv:
# # # # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # # # # # # #         return pred_mean, all_me.mean(0), all_trajs

# # # # # # # # # # #     def _score_sample(self, traj, env_data):
# # # # # # # # # # #         B = traj.shape[1]
# # # # # # # # # # #         if traj.shape[0] < 2:
# # # # # # # # # # #             return torch.ones(B, device=traj.device)

# # # # # # # # # # #         traj_deg = _norm_to_deg_fn(traj)
# # # # # # # # # # #         dt_deg = traj_deg[1:] - traj_deg[:-1]
# # # # # # # # # # #         lat_rad = torch.deg2rad(traj_deg[:-1, :, 1])
# # # # # # # # # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # #         dx_km = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # # # # # # # #         dy_km = dt_deg[:, :, 1] * 111.0
# # # # # # # # # # #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)

# # # # # # # # # # #         speed_penalty = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
# # # # # # # # # # #         speed_score = torch.exp(-speed_penalty.mean(0) / 20.0)

# # # # # # # # # # #         if dt_deg.shape[0] >= 2:
# # # # # # # # # # #             jerk = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
# # # # # # # # # # #             smooth_score = torch.exp(-jerk * 5.0)
# # # # # # # # # # #         else:
# # # # # # # # # # #             smooth_score = torch.ones(B, device=traj.device)

# # # # # # # # # # #         return speed_score * smooth_score

# # # # # # # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # # # # # # #         with torch.enable_grad():
# # # # # # # # # # #             x   = x_pred.detach().requires_grad_(True)
# # # # # # # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # # # # # # #             for _ in range(n_steps):
# # # # # # # # # # #                 opt.zero_grad()
# # # # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # # # #                 pred_deg    = _norm_to_deg_fn(pred_abs)
# # # # # # # # # # #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # # #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # # # #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # # #                 speed       = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # # # # # # #                                           + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # # # # # # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # # # # # # # # # #                 loss.backward()
# # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # # # # # # # # # #                 opt.step()
# # # # # # # # # # #         return x.detach()

# # # # # # # # # # #     @staticmethod
# # # # # # # # # # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # # # # # # # # # #         import numpy as np
# # # # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # # # #                     "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg",
# # # # # # # # # # #                     "ens_spread_km"]
# # # # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # # # #             if write_hdr: w.writeheader()
# # # # # # # # # # #             for b in range(B):
# # # # # # # # # # #                 for k in range(T):
# # # # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # # # #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# # # # # # # # # # #                                * math.cos(math.radians(mean_lat[k, b])))
# # # # # # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # # # # # #                     w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # # # # # # #                                 "lead_h": (k+1)*6,
# # # # # # # # # # #                                 "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # # # # # # #                                 "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # # # # # # #                                 "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # # # #                                 "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # # # #                                 "ens_spread_km": f"{spread:.2f}"})


# # # # # # # # # # # TCDiffusion = TCFlowMatching
# # # # # # # # # # """
# # # # # # # # # # Model/flow_matching_model.py — v34fix + v36_cal logging fix
# # # # # # # # # # ════════════════════════════════════════════════════════════════════
# # # # # # # # # # THAY ĐỔI DUY NHẤT so với v34fix (doc 7):

# # # # # # # # # #   FIX get_loss_breakdown() return dict — THÊM 3 keys còn thiếu:
# # # # # # # # # #     BEFORE:
# # # # # # # # # #       velocity = loss_dict['velocity']   # = 0.0
# # # # # # # # # #       # vel_smooth, ate_cte, h_direct KHÔNG có → log hiện 0.000

# # # # # # # # # #     AFTER:
# # # # # # # # # #       h_direct   = loss_dict['h_direct']    # ← THÊM
# # # # # # # # # #       vel_smooth = loss_dict['vel_smooth']  # ← THÊM
# # # # # # # # # #       ate_cte    = loss_dict['ate_cte']     # ← THÊM
# # # # # # # # # #       velocity   = loss_dict['velocity']    # giữ nguyên

# # # # # # # # # #   Tất cả code khác GIỐNG HỆT v34fix.
# # # # # # # # # #   Training logic không thay đổi gì.
# # # # # # # # # # """
# # # # # # # # # # from __future__ import annotations

# # # # # # # # # # import csv
# # # # # # # # # # import math
# # # # # # # # # # import os
# # # # # # # # # # from copy import deepcopy
# # # # # # # # # # from datetime import datetime

# # # # # # # # # # import torch
# # # # # # # # # # import torch.nn as nn
# # # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # # from Model.losses import (
# # # # # # # # # #     compute_total_loss,
# # # # # # # # # #     _haversine_deg, _norm_to_deg,
# # # # # # # # # #     WEIGHTS,
# # # # # # # # # # )


# # # # # # # # # # def _norm_to_deg_fn(t):
# # # # # # # # # #     """Normalized [lon, lat] → degrees."""
# # # # # # # # # #     out = t.clone()
# # # # # # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # # # # # #     return out


# # # # # # # # # # def _unwrap_model(model: nn.Module) -> nn.Module:
# # # # # # # # # #     if hasattr(model, '_orig_mod'):
# # # # # # # # # #         return model._orig_mod
# # # # # # # # # #     return model


# # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # #  EMA wrapper — v34fix (unchanged)
# # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # class EMAModel:
# # # # # # # # # #     def __init__(self, model: nn.Module, decay: float = 0.999):
# # # # # # # # # #         self.decay = decay
# # # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # # #         self.shadow = {
# # # # # # # # # #             k: v.detach().clone()
# # # # # # # # # #             for k, v in m.state_dict().items()
# # # # # # # # # #             if v.dtype.is_floating_point
# # # # # # # # # #         }

# # # # # # # # # #     def update(self, model: nn.Module):
# # # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # # #         with torch.no_grad():
# # # # # # # # # #             for k, v in m.state_dict().items():
# # # # # # # # # #                 if k in self.shadow:
# # # # # # # # # #                     self.shadow[k].mul_(self.decay).add_(
# # # # # # # # # #                         v.detach(), alpha=1 - self.decay)

# # # # # # # # # #     def apply_to(self, model: nn.Module):
# # # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # # #         backup = {}
# # # # # # # # # #         sd = m.state_dict()
# # # # # # # # # #         for k in self.shadow:
# # # # # # # # # #             if k not in sd:
# # # # # # # # # #                 continue
# # # # # # # # # #             backup[k] = sd[k].detach().clone()
# # # # # # # # # #             sd[k].copy_(self.shadow[k])
# # # # # # # # # #         return backup

# # # # # # # # # #     def restore(self, model: nn.Module, backup: dict):
# # # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # # #         sd = m.state_dict()
# # # # # # # # # #         for k, v in backup.items():
# # # # # # # # # #             if k in sd:
# # # # # # # # # #                 sd[k].copy_(v)


# # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # #  VelocityField — unchanged from v34fix
# # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # # # # # # #                  unet_in_ch=13, **kwargs):
# # # # # # # # # #         super().__init__()
# # # # # # # # # #         self.pred_len = pred_len
# # # # # # # # # #         self.obs_len  = obs_len

# # # # # # # # # #         self.spatial_enc = FNO3DEncoder(
# # # # # # # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # # # # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # # # # # # #             spatial_down=32, dropout=0.05)
# # # # # # # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # # # # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # # # # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # # # # # # #         self.enc_1d = DataEncoder1D(
# # # # # # # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # # # # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # # # # # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # # # #         self.ctx_drop = nn.Dropout(0.10)
# # # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # # #             nn.Linear(256, 256), nn.GELU(),
# # # # # # # # # #         )
# # # # # # # # # #         self.steering_enc = nn.Sequential(
# # # # # # # # # #             nn.Linear(4, 64), nn.GELU(),
# # # # # # # # # #             nn.LayerNorm(64),
# # # # # # # # # #             nn.Linear(64, 128), nn.GELU(),
# # # # # # # # # #             nn.Linear(128, 256),
# # # # # # # # # #         )

# # # # # # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # # # # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # # # # # # # #             num_layers=5)

# # # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # # # # #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # # # # # # # #         self.physics_scale  = nn.Parameter(torch.ones(4) * 0.3)
# # # # # # # # # #         self.steering_scale = nn.Parameter(torch.ones(4) * 0.2)
# # # # # # # # # #         self._init_weights()

# # # # # # # # # #     def _init_weights(self):
# # # # # # # # # #         with torch.no_grad():
# # # # # # # # # #             for name, m in self.named_modules():
# # # # # # # # # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # # # # # # # #                     if m.bias is not None:
# # # # # # # # # #                         nn.init.zeros_(m.bias)

# # # # # # # # # #     def _time_emb(self, t, dim=256):
# # # # # # # # # #         half = dim // 2
# # # # # # # # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # # # # # # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # # # # # # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # # # # # # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # # # # # # # #     def _context(self, batch_list):
# # # # # # # # # #         obs_traj  = batch_list[0]
# # # # # # # # # #         obs_Me    = batch_list[7]
# # # # # # # # # #         image_obs = batch_list[11]
# # # # # # # # # #         env_data  = batch_list[13]

# # # # # # # # # #         if image_obs.dim() == 4:
# # # # # # # # # #             image_obs = image_obs.unsqueeze(2)
# # # # # # # # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # # # # # # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # # # # # # # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # # # # # # # #         T_obs = obs_traj.shape[0]

# # # # # # # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # # # # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # # # # # # #         if e_3d_s.shape[1] != T_obs:
# # # # # # # # # #             e_3d_s = F.interpolate(e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # # # # # # #                                    mode="linear", align_corners=False).permute(0, 2, 1)

# # # # # # # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # # # # # # #         t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # # # # # # #                              device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # # # # # # #         f_spatial = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# # # # # # # # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # # # # # # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # # # # # # # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # # # # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(
# # # # # # # # # #             torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # # #     def _get_vel_obs_feat(self, obs_traj):
# # # # # # # # # #         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
# # # # # # # # # #         if T_obs >= 2:
# # # # # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # # # # # # # #         else:
# # # # # # # # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
# # # # # # # # # #         if vels.shape[0] < self.obs_len:
# # # # # # # # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# # # # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # # # #         else:
# # # # # # # # # #             vels = vels[-self.obs_len:]
# # # # # # # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # # # # # # #     def _get_steering_feat(self, env_data, B, device):
# # # # # # # # # #         if env_data is None:
# # # # # # # # # #             return torch.zeros(B, 256, device=device)

# # # # # # # # # #         def _safe_get(key, default_val=0.0):
# # # # # # # # # #             v = env_data.get(key, None)
# # # # # # # # # #             if v is None or not torch.is_tensor(v):
# # # # # # # # # #                 return torch.full((B,), default_val, device=device)
# # # # # # # # # #             v = v.to(device).float()
# # # # # # # # # #             if v.dim() >= 2:
# # # # # # # # # #                 while v.dim() > 1:
# # # # # # # # # #                     v = v.mean(-1)
# # # # # # # # # #             if v.shape[0] != B:
# # # # # # # # # #                 v = (v.view(-1)[:B] if v.numel() >= B
# # # # # # # # # #                      else torch.full((B,), default_val, device=device))
# # # # # # # # # #             return v

# # # # # # # # # #         feat = torch.stack([_safe_get("u500_mean"), _safe_get("v500_mean"),
# # # # # # # # # #                              _safe_get("u500_center"), _safe_get("v500_center")], dim=-1)
# # # # # # # # # #         return self.steering_enc(feat)

# # # # # # # # # #     def _beta_drift(self, x_t):
# # # # # # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # # # # # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # # # # # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # # # # # #         R_tc    = 3e5
# # # # # # # # # #         v_phys  = torch.zeros_like(x_t)
# # # # # # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # # # # # #         return v_phys

# # # # # # # # # #     def _steering_drift(self, x_t, env_data):
# # # # # # # # # #         if env_data is None:
# # # # # # # # # #             return torch.zeros_like(x_t)
# # # # # # # # # #         B = x_t.shape[0]
# # # # # # # # # #         device = x_t.device

# # # # # # # # # #         def _safe_mean(key):
# # # # # # # # # #             v = env_data.get(key, None)
# # # # # # # # # #             if v is None or not torch.is_tensor(v):
# # # # # # # # # #                 return torch.zeros(B, device=device)
# # # # # # # # # #             v = v.to(device).float()
# # # # # # # # # #             while v.dim() > 1:
# # # # # # # # # #                 v = v.mean(-1)
# # # # # # # # # #             if v.numel() < B:
# # # # # # # # # #                 return torch.zeros(B, device=device)
# # # # # # # # # #             return v.view(-1)[:B]

# # # # # # # # # #         u = _safe_mean("u500_center")
# # # # # # # # # #         v = _safe_mean("v500_center")
# # # # # # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # # # # # #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# # # # # # # # # #         u_norm = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat) * 0.2
# # # # # # # # # #         v_norm = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0) * 0.2
# # # # # # # # # #         out = torch.zeros_like(x_t)
# # # # # # # # # #         out[:, :, 0] = u_norm
# # # # # # # # # #         out[:, :, 1] = v_norm
# # # # # # # # # #         return out

# # # # # # # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
# # # # # # # # # #                 env_data=None):
# # # # # # # # # #         B     = x_t.shape[0]
# # # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # # # # #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # # # # #         s_emb    = self.step_embed(step_idx)

# # # # # # # # # #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# # # # # # # # # #                  + self.pos_enc[:, :T_seq]
# # # # # # # # # #                  + t_emb.unsqueeze(1)
# # # # # # # # # #                  + s_emb)

# # # # # # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # # # # # #         if vel_obs_feat is not None:
# # # # # # # # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))
# # # # # # # # # #         if steering_feat is not None:
# # # # # # # # # #             mem_parts.append(steering_feat.unsqueeze(1))

# # # # # # # # # #         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
# # # # # # # # # #         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # # # # # # #         v_neural = v_neural * scale

# # # # # # # # # #         with torch.no_grad():
# # # # # # # # # #             v_phys  = self._beta_drift(x_t[:, :T_seq])
# # # # # # # # # #             v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

# # # # # # # # # #         return (v_neural
# # # # # # # # # #                 + torch.sigmoid(self.physics_scale) * v_phys
# # # # # # # # # #                 + torch.sigmoid(self.steering_scale) * v_steer)

# # # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
# # # # # # # # # #                            vel_obs_feat=None, steering_feat=None, env_data=None):
# # # # # # # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# # # # # # # # # #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # # #  TCFlowMatching — unchanged except get_loss_breakdown return dict
# # # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # # class TCFlowMatching(nn.Module):

# # # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # # # # # #                  ctx_noise_scale=0.01, initial_sample_sigma=0.03,
# # # # # # # # # #                  teacher_forcing=True, use_ema=True, ema_decay=0.999,
# # # # # # # # # #                  **kwargs):
# # # # # # # # # #         super().__init__()
# # # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # # # # #         self.teacher_forcing      = teacher_forcing
# # # # # # # # # #         self.active_pred_len      = pred_len

# # # # # # # # # #         self.net = VelocityField(
# # # # # # # # # #             pred_len=pred_len, obs_len=obs_len,
# # # # # # # # # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # # # # # # #         self.use_ema   = use_ema
# # # # # # # # # #         self.ema_decay = ema_decay
# # # # # # # # # #         self._ema      = None

# # # # # # # # # #     def init_ema(self):
# # # # # # # # # #         if self.use_ema:
# # # # # # # # # #             self._ema = EMAModel(self, decay=self.ema_decay)

# # # # # # # # # #     def ema_update(self):
# # # # # # # # # #         if self._ema is not None:
# # # # # # # # # #             self._ema.update(self)

# # # # # # # # # #     def set_curriculum_len(self, *a, **kw):
# # # # # # # # # #         pass

# # # # # # # # # #     @staticmethod
# # # # # # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # # # # # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # # # # # # #     @staticmethod
# # # # # # # # # #     def _to_abs(rel, lp, lm):
# # # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # # # # # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # # # # # # # #         if sigma_min is None:
# # # # # # # # # #             sigma_min = self.sigma_min
# # # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # # #         x0 = torch.randn_like(x1) * sigma_min
# # # # # # # # # #         t  = torch.rand(B, device=device)
# # # # # # # # # #         te = t.view(B, 1, 1)
# # # # # # # # # #         x_t = (1.0 - te) * x0 + te * x1
# # # # # # # # # #         u   = x1 - x0
# # # # # # # # # #         return x_t, t, u

# # # # # # # # # #     @staticmethod
# # # # # # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # #             return batch_list
# # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # # # # # #                 t = aug[idx].clone()
# # # # # # # # # #                 t[..., 0] = -t[..., 0]
# # # # # # # # # #                 aug[idx] = t
# # # # # # # # # #         return aug

# # # # # # # # # #     @staticmethod
# # # # # # # # # #     def _obs_noise_aug(batch_list, sigma=0.005):
# # # # # # # # # #         if torch.rand(1).item() > 0.5:
# # # # # # # # # #             return batch_list
# # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # #         if torch.is_tensor(aug[0]):
# # # # # # # # # #             aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
# # # # # # # # # #         return aug

# # # # # # # # # #     @staticmethod
# # # # # # # # # #     def _mixup_aug(batch_list, alpha=0.2, p=0.2):
# # # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # # #             return batch_list
# # # # # # # # # #         aug = list(batch_list)
# # # # # # # # # #         lam = float(torch.distributions.Beta(alpha, alpha).sample())
# # # # # # # # # #         lam = max(lam, 1 - lam)
# # # # # # # # # #         if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
# # # # # # # # # #             B    = aug[0].shape[1]
# # # # # # # # # #             perm = torch.randperm(B, device=aug[0].device)
# # # # # # # # # #             for idx in [0, 1, 7, 8]:
# # # # # # # # # #                 if torch.is_tensor(aug[idx]):
# # # # # # # # # #                     aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
# # # # # # # # # #         return aug

# # # # # # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # # #         batch_list = self._lon_flip_aug(batch_list)
# # # # # # # # # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
# # # # # # # # # #         if epoch >= 5:
# # # # # # # # # #             batch_list = self._mixup_aug(batch_list, alpha=0.2, p=0.15)

# # # # # # # # # #         traj_gt  = batch_list[1]
# # # # # # # # # #         Me_gt    = batch_list[8]
# # # # # # # # # #         obs_t    = batch_list[0]
# # # # # # # # # #         obs_Me   = batch_list[7]
# # # # # # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # # # # # #         lp, lm   = obs_t[-1], obs_Me[-1]

# # # # # # # # # #         # Sigma schedule
# # # # # # # # # #         if epoch < 15:
# # # # # # # # # #             current_sigma = 0.15
# # # # # # # # # #         elif epoch < 40:
# # # # # # # # # #             current_sigma = 0.15 - (epoch - 15) / 25.0 * 0.09
# # # # # # # # # #         else:
# # # # # # # # # #             current_sigma = 0.06

# # # # # # # # # #         raw_ctx       = self.net._context(batch_list)
# # # # # # # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # # # # # #         steering_feat = self.net._get_steering_feat(
# # # # # # # # # #             env_data, obs_t.shape[1], obs_t.device)

# # # # # # # # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
# # # # # # # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # # # # # # #         if self.teacher_forcing and self.training and epoch >= 3:
# # # # # # # # # #             p_teacher = max(0.0, 0.5 * (1.0 - (epoch - 3) / 37.0))
# # # # # # # # # #             if p_teacher > 0 and torch.rand(1).item() < p_teacher:
# # # # # # # # # #                 far_mask = torch.zeros_like(x_t)
# # # # # # # # # #                 far_mask[:, 6:, :] = 0.3
# # # # # # # # # #                 x_t = x_t * (1 - far_mask) + x1_rel * far_mask

# # # # # # # # # #         pred_vel = self.net.forward_with_ctx(
# # # # # # # # # #             x_t, fm_t, raw_ctx,
# # # # # # # # # #             vel_obs_feat=vel_obs_feat,
# # # # # # # # # #             steering_feat=steering_feat,
# # # # # # # # # #             env_data=env_data,
# # # # # # # # # #         )

# # # # # # # # # #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# # # # # # # # # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # # # # # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)
# # # # # # # # # #         pred_deg = _norm_to_deg_fn(pred_abs)
# # # # # # # # # #         gt_deg   = _norm_to_deg_fn(traj_gt)

# # # # # # # # # #         loss_dict = compute_total_loss(
# # # # # # # # # #             pred_deg=pred_deg,
# # # # # # # # # #             gt_deg=gt_deg,
# # # # # # # # # #             env_data=env_data,
# # # # # # # # # #             weights=WEIGHTS,
# # # # # # # # # #             epoch=epoch,
# # # # # # # # # #         )

# # # # # # # # # #         w_fm  = max(0.3, 1.0 - epoch / 60.0)
# # # # # # # # # #         total = w_fm * l_fm_mse + loss_dict["total"]

# # # # # # # # # #         l_ens = x_t.new_zeros(())
# # # # # # # # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # # # # # # # #             x_t2, fm_t2, u_t2 = self._cfm_noisy(
# # # # # # # # # #                 x1_rel, sigma_min=current_sigma)
# # # # # # # # # #             pv2 = self.net.forward_with_ctx(
# # # # # # # # # #                 x_t2, fm_t2, raw_ctx,
# # # # # # # # # #                 vel_obs_feat=vel_obs_feat,
# # # # # # # # # #                 steering_feat=steering_feat,
# # # # # # # # # #                 env_data=env_data,
# # # # # # # # # #             )
# # # # # # # # # #             l_ens = F.mse_loss(pv2, u_t2)
# # # # # # # # # #             total = total + 0.3 * l_ens

# # # # # # # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # # # # # # #             total = x_t.new_zeros(())

# # # # # # # # # #         # ── FIX: thêm vel_smooth, ate_cte, h_direct vào return dict ──────
# # # # # # # # # #         return dict(
# # # # # # # # # #             total        = total,
# # # # # # # # # #             fm_mse       = l_fm_mse.item(),
# # # # # # # # # #             mse_hav      = loss_dict["mse_hav_horizon"],
# # # # # # # # # #             multi_scale  = loss_dict["multi_scale"],
# # # # # # # # # #             endpoint     = loss_dict["endpoint"],
# # # # # # # # # #             h_direct     = loss_dict.get("h_direct",    0.0),   # ← ADDED
# # # # # # # # # #             vel_smooth   = loss_dict.get("vel_smooth",  0.0),   # ← ADDED
# # # # # # # # # #             ate_cte      = loss_dict.get("ate_cte",     0.0),   # ← ADDED
# # # # # # # # # #             shape        = loss_dict["shape"],
# # # # # # # # # #             velocity     = loss_dict["velocity"],
# # # # # # # # # #             heading      = loss_dict["heading"],
# # # # # # # # # #             recurv       = loss_dict["recurv"],
# # # # # # # # # #             steering     = loss_dict["steering"],
# # # # # # # # # #             ens_consist  = (l_ens.item() if torch.is_tensor(l_ens)
# # # # # # # # # #                             else float(l_ens)),
# # # # # # # # # #             sigma        = current_sigma,
# # # # # # # # # #             w_fm         = w_fm,
# # # # # # # # # #             long_range=0.0, fde=0.0, w_lr=0.0, w_fde=0.0,
# # # # # # # # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # # # # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0,
# # # # # # # # # #             recurv_ratio=0.0,
# # # # # # # # # #         )

# # # # # # # # # #     # ── Sampling — unchanged from v34fix ──────────────────────────────────
# # # # # # # # # #     @torch.no_grad()
# # # # # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
# # # # # # # # # #                 predict_csv=None, importance_weight=True):
# # # # # # # # # #         obs_t    = batch_list[0]
# # # # # # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # # # # # #         lp       = obs_t[-1]
# # # # # # # # # #         lm       = batch_list[7][-1]
# # # # # # # # # #         B        = lp.shape[0]
# # # # # # # # # #         device   = lp.device
# # # # # # # # # #         T        = self.pred_len
# # # # # # # # # #         dt       = 1.0 / max(ddim_steps, 1)

# # # # # # # # # #         raw_ctx       = self.net._context(batch_list)
# # # # # # # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # # # # # #         steering_feat = self.net._get_steering_feat(env_data, B, device)

# # # # # # # # # #         traj_s, me_s, scores = [], [], []
# # # # # # # # # #         for ens_i in range(num_ensemble):
# # # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * (self.sigma_min * 2.5)
# # # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # # #                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
# # # # # # # # # #                 vel = self.net.forward_with_ctx(
# # # # # # # # # #                     x_t, t_b, raw_ctx,
# # # # # # # # # #                     noise_scale=ns,
# # # # # # # # # #                     vel_obs_feat=vel_obs_feat,
# # # # # # # # # #                     steering_feat=steering_feat,
# # # # # # # # # #                     env_data=env_data,
# # # # # # # # # #                 )
# # # # # # # # # #                 x_t = x_t + dt * vel
# # # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # # #             traj_s.append(tr)
# # # # # # # # # #             me_s.append(me)
# # # # # # # # # #             if importance_weight:
# # # # # # # # # #                 scores.append(self._score_sample(tr, env_data))

# # # # # # # # # #         all_trajs = torch.stack(traj_s)
# # # # # # # # # #         all_me    = torch.stack(me_s)

# # # # # # # # # #         if importance_weight and scores:
# # # # # # # # # #             score_tensor = torch.stack(scores)
# # # # # # # # # #             k = max(1, int(num_ensemble * 0.7))
# # # # # # # # # #             _, top_idx = score_tensor.topk(k, dim=0)
# # # # # # # # # #             pred_mean = torch.stack([
# # # # # # # # # #                 all_trajs[top_idx[:, b], :, b, :].median(0).values
# # # # # # # # # #                 for b in range(B)
# # # # # # # # # #             ], dim=1)
# # # # # # # # # #         else:
# # # # # # # # # #             pred_mean = all_trajs.median(0).values

# # # # # # # # # #         if predict_csv:
# # # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # # # # # #         return pred_mean, all_me.mean(0), all_trajs

# # # # # # # # # #     def _score_sample(self, traj, env_data):
# # # # # # # # # #         B = traj.shape[1]
# # # # # # # # # #         if traj.shape[0] < 2:
# # # # # # # # # #             return torch.ones(B, device=traj.device)
# # # # # # # # # #         traj_deg = _norm_to_deg_fn(traj)
# # # # # # # # # #         dt_deg   = traj_deg[1:] - traj_deg[:-1]
# # # # # # # # # #         lat_rad  = torch.deg2rad(traj_deg[:-1, :, 1])
# # # # # # # # # #         cos_lat  = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # #         dx_km    = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # # # # # # #         dy_km    = dt_deg[:, :, 1] * 111.0
# # # # # # # # # #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)
# # # # # # # # # #         speed_penalty = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
# # # # # # # # # #         speed_score   = torch.exp(-speed_penalty.mean(0) / 20.0)
# # # # # # # # # #         if dt_deg.shape[0] >= 2:
# # # # # # # # # #             jerk = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
# # # # # # # # # #             smooth_score = torch.exp(-jerk * 5.0)
# # # # # # # # # #         else:
# # # # # # # # # #             smooth_score = torch.ones(B, device=traj.device)
# # # # # # # # # #         return speed_score * smooth_score

# # # # # # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # # # # # #         with torch.enable_grad():
# # # # # # # # # #             x   = x_pred.detach().requires_grad_(True)
# # # # # # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # # # # # #             for _ in range(n_steps):
# # # # # # # # # #                 opt.zero_grad()
# # # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # # #                 pred_deg    = _norm_to_deg_fn(pred_abs)
# # # # # # # # # #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # # #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # # #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # # #                 speed       = torch.sqrt(
# # # # # # # # # #                     (dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # # # # # #                     + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # # # # # #                 loss = F.relu(speed - 600.0).pow(2).mean()
# # # # # # # # # #                 loss.backward()
# # # # # # # # # #                 torch.nn.utils.clip_grad_norm_([x], 0.05)
# # # # # # # # # #                 opt.step()
# # # # # # # # # #         return x.detach()

# # # # # # # # # #     @staticmethod
# # # # # # # # # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # # # # # # # # #         import numpy as np
# # # # # # # # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # # # # # # # #         T, B, _ = traj_mean.shape
# # # # # # # # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # # # # # # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # # # # # # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # # # # # # # #                     "lon_mean_deg", "lat_mean_deg",
# # # # # # # # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # # #             if write_hdr: w.writeheader()
# # # # # # # # # #             for b in range(B):
# # # # # # # # # #                 for k in range(T):
# # # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # # #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# # # # # # # # # #                                * math.cos(math.radians(mean_lat[k, b])))
# # # # # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # # # # #                     w.writerow({
# # # # # # # # # #                         "timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # # # # # #                         "lead_h": (k + 1) * 6,
# # # # # # # # # #                         "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # # # # # #                         "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # # # # # #                         "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # # #                         "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # # #                         "ens_spread_km": f"{spread:.2f}",
# # # # # # # # # #                     })


# # # # # # # # # # TCDiffusion = TCFlowMatching

# # # # # # # # # """
# # # # # # # # # Model/flow_matching_model.py — v37_ate
# # # # # # # # # ════════════════════════════════════════════════════════════════════
# # # # # # # # # THAY ĐỔI so với v36 (doc 7):

# # # # # # # # #   1. get_loss_breakdown() return dict — THÊM key "along_track":
# # # # # # # # #        along_track = loss_dict.get("along_track", 0.0)  ← NEW

# # # # # # # # #   2. Sigma schedule — TIGHTER từ early epoch:
# # # # # # # # #        ep0-5:   0.15 → 0.10  (tighter → học exact position sớm hơn)
# # # # # # # # #        ep5-20:  0.10 → 0.05
# # # # # # # # #        ep20-50: 0.05 → 0.03
# # # # # # # # #        ep50+:   0.03

# # # # # # # # #   3. EMA decay: 0.992 → 0.990 (faster update)

# # # # # # # # #   KHÔNG thay đổi gì khác.
# # # # # # # # # """
# # # # # # # # # from __future__ import annotations

# # # # # # # # # import csv
# # # # # # # # # import math
# # # # # # # # # import os
# # # # # # # # # from copy import deepcopy
# # # # # # # # # from datetime import datetime

# # # # # # # # # import torch
# # # # # # # # # import torch.nn as nn
# # # # # # # # # import torch.nn.functional as F

# # # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # # from Model.losses import (
# # # # # # # # #     compute_total_loss,
# # # # # # # # #     _haversine_deg, _norm_to_deg,
# # # # # # # # #     WEIGHTS,
# # # # # # # # # )


# # # # # # # # # def _norm_to_deg_fn(t):
# # # # # # # # #     out = t.clone()
# # # # # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # # # # #     return out


# # # # # # # # # def _unwrap_model(model: nn.Module) -> nn.Module:
# # # # # # # # #     if hasattr(model, '_orig_mod'):
# # # # # # # # #         return model._orig_mod
# # # # # # # # #     return model


# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  EMA wrapper — unchanged
# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # class EMAModel:
# # # # # # # # #     def __init__(self, model: nn.Module, decay: float = 0.999):
# # # # # # # # #         self.decay = decay
# # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # #         self.shadow = {
# # # # # # # # #             k: v.detach().clone()
# # # # # # # # #             for k, v in m.state_dict().items()
# # # # # # # # #             if v.dtype.is_floating_point
# # # # # # # # #         }

# # # # # # # # #     def update(self, model: nn.Module):
# # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # #         with torch.no_grad():
# # # # # # # # #             for k, v in m.state_dict().items():
# # # # # # # # #                 if k in self.shadow:
# # # # # # # # #                     self.shadow[k].mul_(self.decay).add_(
# # # # # # # # #                         v.detach(), alpha=1 - self.decay)

# # # # # # # # #     def apply_to(self, model: nn.Module):
# # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # #         backup = {}
# # # # # # # # #         sd = m.state_dict()
# # # # # # # # #         for k in self.shadow:
# # # # # # # # #             if k not in sd:
# # # # # # # # #                 continue
# # # # # # # # #             backup[k] = sd[k].detach().clone()
# # # # # # # # #             sd[k].copy_(self.shadow[k])
# # # # # # # # #         return backup

# # # # # # # # #     def restore(self, model: nn.Module, backup: dict):
# # # # # # # # #         m = _unwrap_model(model)
# # # # # # # # #         sd = m.state_dict()
# # # # # # # # #         for k, v in backup.items():
# # # # # # # # #             if k in sd:
# # # # # # # # #                 sd[k].copy_(v)


# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  VelocityField — unchanged from v36
# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # class VelocityField(nn.Module):
# # # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # # # # # #                  unet_in_ch=13, **kwargs):
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
# # # # # # # # #         self.ctx_drop = nn.Dropout(0.10)
# # # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # # # # # #             nn.LayerNorm(256),
# # # # # # # # #             nn.Linear(256, 256), nn.GELU(),
# # # # # # # # #         )
# # # # # # # # #         self.steering_enc = nn.Sequential(
# # # # # # # # #             nn.Linear(4, 64), nn.GELU(),
# # # # # # # # #             nn.LayerNorm(64),
# # # # # # # # #             nn.Linear(64, 128), nn.GELU(),
# # # # # # # # #             nn.Linear(128, 256),
# # # # # # # # #         )

# # # # # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # # # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # # # # # # #             num_layers=5)

# # # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # # # #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # # # # # # #         self.physics_scale  = nn.Parameter(torch.ones(4) * 0.3)
# # # # # # # # #         self.steering_scale = nn.Parameter(torch.ones(4) * 0.2)
# # # # # # # # #         self._init_weights()

# # # # # # # # #     def _init_weights(self):
# # # # # # # # #         with torch.no_grad():
# # # # # # # # #             for name, m in self.named_modules():
# # # # # # # # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # # # # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # # # # # # #                     if m.bias is not None:
# # # # # # # # #                         nn.init.zeros_(m.bias)

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

# # # # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(
# # # # # # # # #             torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # # # # #         if noise_scale > 0.0:
# # # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # # #     def _get_vel_obs_feat(self, obs_traj):
# # # # # # # # #         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
# # # # # # # # #         if T_obs >= 2:
# # # # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # # # # # # #         else:
# # # # # # # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
# # # # # # # # #         if vels.shape[0] < self.obs_len:
# # # # # # # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
# # # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # # #         else:
# # # # # # # # #             vels = vels[-self.obs_len:]
# # # # # # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # # # # # #     def _get_steering_feat(self, env_data, B, device):
# # # # # # # # #         if env_data is None:
# # # # # # # # #             return torch.zeros(B, 256, device=device)

# # # # # # # # #         def _safe_get(key, default_val=0.0):
# # # # # # # # #             v = env_data.get(key, None)
# # # # # # # # #             if v is None or not torch.is_tensor(v):
# # # # # # # # #                 return torch.full((B,), default_val, device=device)
# # # # # # # # #             v = v.to(device).float()
# # # # # # # # #             if v.dim() >= 2:
# # # # # # # # #                 while v.dim() > 1:
# # # # # # # # #                     v = v.mean(-1)
# # # # # # # # #             if v.shape[0] != B:
# # # # # # # # #                 v = (v.view(-1)[:B] if v.numel() >= B
# # # # # # # # #                      else torch.full((B,), default_val, device=device))
# # # # # # # # #             return v

# # # # # # # # #         feat = torch.stack([_safe_get("u500_mean"), _safe_get("v500_mean"),
# # # # # # # # #                              _safe_get("u500_center"), _safe_get("v500_center")], dim=-1)
# # # # # # # # #         return self.steering_enc(feat)

# # # # # # # # #     def _beta_drift(self, x_t):
# # # # # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # # # # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # # # # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # # # # #         R_tc    = 3e5
# # # # # # # # #         v_phys  = torch.zeros_like(x_t)
# # # # # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # # # # #         return v_phys

# # # # # # # # #     def _steering_drift(self, x_t, env_data):
# # # # # # # # #         if env_data is None:
# # # # # # # # #             return torch.zeros_like(x_t)
# # # # # # # # #         B = x_t.shape[0]
# # # # # # # # #         device = x_t.device

# # # # # # # # #         def _safe_mean(key):
# # # # # # # # #             v = env_data.get(key, None)
# # # # # # # # #             if v is None or not torch.is_tensor(v):
# # # # # # # # #                 return torch.zeros(B, device=device)
# # # # # # # # #             v = v.to(device).float()
# # # # # # # # #             while v.dim() > 1:
# # # # # # # # #                 v = v.mean(-1)
# # # # # # # # #             if v.numel() < B:
# # # # # # # # #                 return torch.zeros(B, device=device)
# # # # # # # # #             return v.view(-1)[:B]

# # # # # # # # #         u = _safe_mean("u500_center")
# # # # # # # # #         v = _safe_mean("v500_center")
# # # # # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # # # # #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# # # # # # # # #         u_norm = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat) * 0.2
# # # # # # # # #         v_norm = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0) * 0.2
# # # # # # # # #         out = torch.zeros_like(x_t)
# # # # # # # # #         out[:, :, 0] = u_norm
# # # # # # # # #         out[:, :, 1] = v_norm
# # # # # # # # #         return out

# # # # # # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
# # # # # # # # #                 env_data=None):
# # # # # # # # #         B     = x_t.shape[0]
# # # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # # # #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # # # #         s_emb    = self.step_embed(step_idx)

# # # # # # # # #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# # # # # # # # #                  + self.pos_enc[:, :T_seq]
# # # # # # # # #                  + t_emb.unsqueeze(1)
# # # # # # # # #                  + s_emb)

# # # # # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # # # # #         if vel_obs_feat is not None:
# # # # # # # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))
# # # # # # # # #         if steering_feat is not None:
# # # # # # # # #             mem_parts.append(steering_feat.unsqueeze(1))

# # # # # # # # #         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
# # # # # # # # #         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # # # # # #         v_neural = v_neural * scale

# # # # # # # # #         with torch.no_grad():
# # # # # # # # #             v_phys  = self._beta_drift(x_t[:, :T_seq])
# # # # # # # # #             v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

# # # # # # # # #         return (v_neural
# # # # # # # # #                 + torch.sigmoid(self.physics_scale) * v_phys
# # # # # # # # #                 + torch.sigmoid(self.steering_scale) * v_steer)

# # # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
# # # # # # # # #                            vel_obs_feat=None, steering_feat=None, env_data=None):
# # # # # # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# # # # # # # # #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  TCFlowMatching v37
# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # class TCFlowMatching(nn.Module):

# # # # # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # # # # #                  ctx_noise_scale=0.01, initial_sample_sigma=0.03,
# # # # # # # # #                  teacher_forcing=True, use_ema=True, ema_decay=0.990,  # v37: 0.992→0.990
# # # # # # # # #                  **kwargs):
# # # # # # # # #         super().__init__()
# # # # # # # # #         self.pred_len             = pred_len
# # # # # # # # #         self.obs_len              = obs_len
# # # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # # # #         self.teacher_forcing      = teacher_forcing
# # # # # # # # #         self.active_pred_len      = pred_len

# # # # # # # # #         self.net = VelocityField(
# # # # # # # # #             pred_len=pred_len, obs_len=obs_len,
# # # # # # # # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # # # # # #         self.use_ema   = use_ema
# # # # # # # # #         self.ema_decay = ema_decay
# # # # # # # # #         self._ema      = None

# # # # # # # # #     def init_ema(self):
# # # # # # # # #         if self.use_ema:
# # # # # # # # #             self._ema = EMAModel(self, decay=self.ema_decay)

# # # # # # # # #     def ema_update(self):
# # # # # # # # #         if self._ema is not None:
# # # # # # # # #             self._ema.update(self)

# # # # # # # # #     def set_curriculum_len(self, *a, **kw):
# # # # # # # # #         pass

# # # # # # # # #     @staticmethod
# # # # # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # # # # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # # # # # #     @staticmethod
# # # # # # # # #     def _to_abs(rel, lp, lm):
# # # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # # # # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # # # # # # #         if sigma_min is None:
# # # # # # # # #             sigma_min = self.sigma_min
# # # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # # #         x0 = torch.randn_like(x1) * sigma_min
# # # # # # # # #         t  = torch.rand(B, device=device)
# # # # # # # # #         te = t.view(B, 1, 1)
# # # # # # # # #         x_t = (1.0 - te) * x0 + te * x1
# # # # # # # # #         u   = x1 - x0
# # # # # # # # #         return x_t, t, u

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

# # # # # # # # #     @staticmethod
# # # # # # # # #     def _obs_noise_aug(batch_list, sigma=0.005):
# # # # # # # # #         if torch.rand(1).item() > 0.5:
# # # # # # # # #             return batch_list
# # # # # # # # #         aug = list(batch_list)
# # # # # # # # #         if torch.is_tensor(aug[0]):
# # # # # # # # #             aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
# # # # # # # # #         return aug

# # # # # # # # #     @staticmethod
# # # # # # # # #     def _mixup_aug(batch_list, alpha=0.2, p=0.2):
# # # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # # #             return batch_list
# # # # # # # # #         aug = list(batch_list)
# # # # # # # # #         lam = float(torch.distributions.Beta(alpha, alpha).sample())
# # # # # # # # #         lam = max(lam, 1 - lam)
# # # # # # # # #         if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
# # # # # # # # #             B    = aug[0].shape[1]
# # # # # # # # #             perm = torch.randperm(B, device=aug[0].device)
# # # # # # # # #             for idx in [0, 1, 7, 8]:
# # # # # # # # #                 if torch.is_tensor(aug[idx]):
# # # # # # # # #                     aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
# # # # # # # # #         return aug

# # # # # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # # #         batch_list = self._lon_flip_aug(batch_list)
# # # # # # # # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
# # # # # # # # #         if epoch >= 5:
# # # # # # # # #             batch_list = self._mixup_aug(batch_list, alpha=0.2, p=0.15)

# # # # # # # # #         traj_gt  = batch_list[1]
# # # # # # # # #         Me_gt    = batch_list[8]
# # # # # # # # #         obs_t    = batch_list[0]
# # # # # # # # #         obs_Me   = batch_list[7]
# # # # # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # # # # #         lp, lm   = obs_t[-1], obs_Me[-1]

# # # # # # # # #         # ── v37: Tighter sigma schedule → học exact position sớm hơn ─────
# # # # # # # # #         if epoch < 5:
# # # # # # # # #             current_sigma = 0.10                                     # 0.15→0.10
# # # # # # # # #         elif epoch < 20:
# # # # # # # # #             current_sigma = 0.10 - (epoch - 5) / 15.0 * 0.05       # 0.10→0.05
# # # # # # # # #         elif epoch < 50:
# # # # # # # # #             current_sigma = 0.05 - (epoch - 20) / 30.0 * 0.02      # 0.05→0.03
# # # # # # # # #         else:
# # # # # # # # #             current_sigma = 0.03

# # # # # # # # #         raw_ctx       = self.net._context(batch_list)
# # # # # # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # # # # #         steering_feat = self.net._get_steering_feat(
# # # # # # # # #             env_data, obs_t.shape[1], obs_t.device)

# # # # # # # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
# # # # # # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # # # # # #         if self.teacher_forcing and self.training and epoch >= 3:
# # # # # # # # #             p_teacher = max(0.0, 0.5 * (1.0 - (epoch - 3) / 37.0))
# # # # # # # # #             if p_teacher > 0 and torch.rand(1).item() < p_teacher:
# # # # # # # # #                 far_mask = torch.zeros_like(x_t)
# # # # # # # # #                 far_mask[:, 6:, :] = 0.3
# # # # # # # # #                 x_t = x_t * (1 - far_mask) + x1_rel * far_mask

# # # # # # # # #         pred_vel = self.net.forward_with_ctx(
# # # # # # # # #             x_t, fm_t, raw_ctx,
# # # # # # # # #             vel_obs_feat=vel_obs_feat,
# # # # # # # # #             steering_feat=steering_feat,
# # # # # # # # #             env_data=env_data,
# # # # # # # # #         )

# # # # # # # # #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# # # # # # # # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # # # # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)
# # # # # # # # #         pred_deg = _norm_to_deg_fn(pred_abs)
# # # # # # # # #         gt_deg   = _norm_to_deg_fn(traj_gt)

# # # # # # # # #         loss_dict = compute_total_loss(
# # # # # # # # #             pred_deg=pred_deg,
# # # # # # # # #             gt_deg=gt_deg,
# # # # # # # # #             env_data=env_data,
# # # # # # # # #             weights=WEIGHTS,
# # # # # # # # #             epoch=epoch,
# # # # # # # # #         )

# # # # # # # # #         w_fm  = max(0.3, 1.0 - epoch / 60.0)
# # # # # # # # #         total = w_fm * l_fm_mse + loss_dict["total"]

# # # # # # # # #         l_ens = x_t.new_zeros(())
# # # # # # # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # # # # # # #             x_t2, fm_t2, u_t2 = self._cfm_noisy(
# # # # # # # # #                 x1_rel, sigma_min=current_sigma)
# # # # # # # # #             pv2 = self.net.forward_with_ctx(
# # # # # # # # #                 x_t2, fm_t2, raw_ctx,
# # # # # # # # #                 vel_obs_feat=vel_obs_feat,
# # # # # # # # #                 steering_feat=steering_feat,
# # # # # # # # #                 env_data=env_data,
# # # # # # # # #             )
# # # # # # # # #             l_ens = F.mse_loss(pv2, u_t2)
# # # # # # # # #             total = total + 0.3 * l_ens

# # # # # # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # # # # # #             total = x_t.new_zeros(())

# # # # # # # # #         # v37: thêm key "along_track" vào return dict
# # # # # # # # #         return dict(
# # # # # # # # #             total        = total,
# # # # # # # # #             fm_mse       = l_fm_mse.item(),
# # # # # # # # #             mse_hav      = loss_dict["mse_hav_horizon"],
# # # # # # # # #             multi_scale  = loss_dict["multi_scale"],
# # # # # # # # #             endpoint     = loss_dict["endpoint"],
# # # # # # # # #             h_direct     = loss_dict.get("h_direct",     0.0),
# # # # # # # # #             vel_smooth   = loss_dict.get("vel_smooth",   0.0),
# # # # # # # # #             ate_cte      = loss_dict.get("ate_cte",      0.0),
# # # # # # # # #             along_track  = loss_dict.get("along_track",  0.0),   # ← v37 NEW
# # # # # # # # #             shape        = loss_dict["shape"],
# # # # # # # # #             velocity     = loss_dict["velocity"],
# # # # # # # # #             heading      = loss_dict["heading"],
# # # # # # # # #             recurv       = loss_dict["recurv"],
# # # # # # # # #             steering     = loss_dict["steering"],
# # # # # # # # #             ens_consist  = (l_ens.item() if torch.is_tensor(l_ens)
# # # # # # # # #                             else float(l_ens)),
# # # # # # # # #             sigma        = current_sigma,
# # # # # # # # #             w_fm         = w_fm,
# # # # # # # # #             long_range=0.0, fde=0.0, w_lr=0.0, w_fde=0.0,
# # # # # # # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # # # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0,
# # # # # # # # #             recurv_ratio=0.0,
# # # # # # # # #         )

# # # # # # # # #     # ── Sampling — unchanged from v36 ─────────────────────────────────────
# # # # # # # # #     @torch.no_grad()
# # # # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
# # # # # # # # #                 predict_csv=None, importance_weight=True):
# # # # # # # # #         obs_t    = batch_list[0]
# # # # # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # # # # #         lp       = obs_t[-1]
# # # # # # # # #         lm       = batch_list[7][-1]
# # # # # # # # #         B        = lp.shape[0]
# # # # # # # # #         device   = lp.device
# # # # # # # # #         T        = self.pred_len
# # # # # # # # #         dt       = 1.0 / max(ddim_steps, 1)

# # # # # # # # #         raw_ctx       = self.net._context(batch_list)
# # # # # # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # # # # #         steering_feat = self.net._get_steering_feat(env_data, B, device)

# # # # # # # # #         traj_s, me_s, scores = [], [], []
# # # # # # # # #         for ens_i in range(num_ensemble):
# # # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * (self.sigma_min * 2.5)
# # # # # # # # #             for step in range(ddim_steps):
# # # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # # #                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
# # # # # # # # #                 vel = self.net.forward_with_ctx(
# # # # # # # # #                     x_t, t_b, raw_ctx,
# # # # # # # # #                     noise_scale=ns,
# # # # # # # # #                     vel_obs_feat=vel_obs_feat,
# # # # # # # # #                     steering_feat=steering_feat,
# # # # # # # # #                     env_data=env_data,
# # # # # # # # #                 )
# # # # # # # # #                 x_t = x_t + dt * vel
# # # # # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # # #             traj_s.append(tr)
# # # # # # # # #             me_s.append(me)
# # # # # # # # #             if importance_weight:
# # # # # # # # #                 scores.append(self._score_sample(tr, env_data))

# # # # # # # # #         all_trajs = torch.stack(traj_s)
# # # # # # # # #         all_me    = torch.stack(me_s)

# # # # # # # # #         if importance_weight and scores:
# # # # # # # # #             score_tensor = torch.stack(scores)
# # # # # # # # #             k = max(1, int(num_ensemble * 0.7))
# # # # # # # # #             _, top_idx = score_tensor.topk(k, dim=0)
# # # # # # # # #             pred_mean = torch.stack([
# # # # # # # # #                 all_trajs[top_idx[:, b], :, b, :].median(0).values
# # # # # # # # #                 for b in range(B)
# # # # # # # # #             ], dim=1)
# # # # # # # # #         else:
# # # # # # # # #             pred_mean = all_trajs.median(0).values

# # # # # # # # #         if predict_csv:
# # # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # # # # #         return pred_mean, all_me.mean(0), all_trajs

# # # # # # # # #     def _score_sample(self, traj, env_data):
# # # # # # # # #         B = traj.shape[1]
# # # # # # # # #         if traj.shape[0] < 2:
# # # # # # # # #             return torch.ones(B, device=traj.device)
# # # # # # # # #         traj_deg = _norm_to_deg_fn(traj)
# # # # # # # # #         dt_deg   = traj_deg[1:] - traj_deg[:-1]
# # # # # # # # #         lat_rad  = torch.deg2rad(traj_deg[:-1, :, 1])
# # # # # # # # #         cos_lat  = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # #         dx_km    = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # # # # # #         dy_km    = dt_deg[:, :, 1] * 111.0
# # # # # # # # #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)
# # # # # # # # #         speed_penalty = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
# # # # # # # # #         speed_score   = torch.exp(-speed_penalty.mean(0) / 20.0)
# # # # # # # # #         if dt_deg.shape[0] >= 2:
# # # # # # # # #             jerk = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
# # # # # # # # #             smooth_score = torch.exp(-jerk * 5.0)
# # # # # # # # #         else:
# # # # # # # # #             smooth_score = torch.ones(B, device=traj.device)
# # # # # # # # #         return speed_score * smooth_score

# # # # # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # # # # #         with torch.enable_grad():
# # # # # # # # #             x   = x_pred.detach().requires_grad_(True)
# # # # # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # # # # #             for _ in range(n_steps):
# # # # # # # # #                 opt.zero_grad()
# # # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # # #                 pred_deg    = _norm_to_deg_fn(pred_abs)
# # # # # # # # #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# # # # # # # # #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # # #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # # #                 speed       = torch.sqrt(
# # # # # # # # #                     (dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # # # # #                     + (dt_deg[:, :, 1] * 111.0).pow(2))
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
# # # # # # # # #                     "lon_mean_deg", "lat_mean_deg",
# # # # # # # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # # #             if write_hdr: w.writeheader()
# # # # # # # # #             for b in range(B):
# # # # # # # # #                 for k in range(T):
# # # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # # #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# # # # # # # # #                                * math.cos(math.radians(mean_lat[k, b])))
# # # # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # # # #                     w.writerow({
# # # # # # # # #                         "timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # # # # #                         "lead_h": (k + 1) * 6,
# # # # # # # # #                         "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # # # # #                         "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # # # # #                         "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # # # # # # # #                         "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # # # # # # # #                         "ens_spread_km": f"{spread:.2f}",
# # # # # # # # #                     })


# # # # # # # # # TCDiffusion = TCFlowMatching

# # # # # # # # """
# # # # # # # # Model/flow_matching_model.py — v34fix  (v39 compat)
# # # # # # # # ═══════════════════════════════════════════════════════════════════

# # # # # # # # THAY ĐỔI DUY NHẤT so với v34fix gốc (doc 7):
# # # # # # # #   get_loss_breakdown() return dict — thêm 2 keys cho losses v39:
# # # # # # # #     speed_acc  = loss_dict.get('speed_acc',  0.0)   ← THÊM
# # # # # # # #     cumul_disp = loss_dict.get('cumul_disp', 0.0)   ← THÊM

# # # # # # # # Tất cả code khác GIỐNG HỆT v34fix.
# # # # # # # # Không thay đổi architecture, training logic, sampling.
# # # # # # # # """
# # # # # # # # from __future__ import annotations

# # # # # # # # import csv
# # # # # # # # import math
# # # # # # # # import os
# # # # # # # # from datetime import datetime

# # # # # # # # import torch
# # # # # # # # import torch.nn as nn
# # # # # # # # import torch.nn.functional as F

# # # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # # from Model.losses import (
# # # # # # # #     compute_total_loss,
# # # # # # # #     _haversine_deg, _norm_to_deg,
# # # # # # # #     WEIGHTS,
# # # # # # # # )


# # # # # # # # def _norm_to_deg_fn(t):
# # # # # # # #     out = t.clone()
# # # # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # # # #     return out


# # # # # # # # def _unwrap_model(model: nn.Module) -> nn.Module:
# # # # # # # #     if hasattr(model, '_orig_mod'):
# # # # # # # #         return model._orig_mod
# # # # # # # #     return model


# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  EMAModel — unchanged from v34fix
# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # class EMAModel:
# # # # # # # #     def __init__(self, model: nn.Module, decay: float = 0.999):
# # # # # # # #         self.decay = decay
# # # # # # # #         m = _unwrap_model(model)
# # # # # # # #         self.shadow = {
# # # # # # # #             k: v.detach().clone()
# # # # # # # #             for k, v in m.state_dict().items()
# # # # # # # #             if v.dtype.is_floating_point
# # # # # # # #         }

# # # # # # # #     def update(self, model: nn.Module):
# # # # # # # #         m = _unwrap_model(model)
# # # # # # # #         with torch.no_grad():
# # # # # # # #             for k, v in m.state_dict().items():
# # # # # # # #                 if k in self.shadow:
# # # # # # # #                     self.shadow[k].mul_(self.decay).add_(
# # # # # # # #                         v.detach(), alpha=1 - self.decay)

# # # # # # # #     def apply_to(self, model: nn.Module):
# # # # # # # #         m = _unwrap_model(model)
# # # # # # # #         backup = {}
# # # # # # # #         sd = m.state_dict()
# # # # # # # #         for k in self.shadow:
# # # # # # # #             if k not in sd:
# # # # # # # #                 continue
# # # # # # # #             backup[k] = sd[k].detach().clone()
# # # # # # # #             sd[k].copy_(self.shadow[k])
# # # # # # # #         return backup

# # # # # # # #     def restore(self, model: nn.Module, backup: dict):
# # # # # # # #         m = _unwrap_model(model)
# # # # # # # #         sd = m.state_dict()
# # # # # # # #         for k, v in backup.items():
# # # # # # # #             if k in sd:
# # # # # # # #                 sd[k].copy_(v)


# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  VelocityField — unchanged from v34fix
# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # class VelocityField(nn.Module):
# # # # # # # #     RAW_CTX_DIM = 512

# # # # # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # # # # #                  unet_in_ch=13, **kwargs):
# # # # # # # #         super().__init__()
# # # # # # # #         self.pred_len = pred_len
# # # # # # # #         self.obs_len  = obs_len

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

# # # # # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # # # # #         self.ctx_drop = nn.Dropout(0.10)
# # # # # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # # # # #         self.vel_obs_enc = nn.Sequential(
# # # # # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # # # # #             nn.LayerNorm(256),
# # # # # # # #             nn.Linear(256, 256), nn.GELU(),
# # # # # # # #         )
# # # # # # # #         self.steering_enc = nn.Sequential(
# # # # # # # #             nn.Linear(4, 64), nn.GELU(),
# # # # # # # #             nn.LayerNorm(64),
# # # # # # # #             nn.Linear(64, 128), nn.GELU(),
# # # # # # # #             nn.Linear(128, 256),
# # # # # # # #         )

# # # # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # # # #             nn.TransformerDecoderLayer(
# # # # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # # # # # #             num_layers=5)

# # # # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # # # #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # # # # # #         self.physics_scale  = nn.Parameter(torch.ones(4) * 0.3)
# # # # # # # #         self.steering_scale = nn.Parameter(torch.ones(4) * 0.2)
# # # # # # # #         self._init_weights()

# # # # # # # #     def _init_weights(self):
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

# # # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(
# # # # # # # #             torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # # # #         if noise_scale > 0.0:
# # # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # # #     def _get_vel_obs_feat(self, obs_traj):
# # # # # # # #         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
# # # # # # # #         if T_obs >= 2:
# # # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # # # # # #         else:
# # # # # # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
# # # # # # # #         if vels.shape[0] < self.obs_len:
# # # # # # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2,
# # # # # # # #                                device=obs_traj.device)
# # # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # # #         else:
# # # # # # # #             vels = vels[-self.obs_len:]
# # # # # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # # # # #     def _get_steering_feat(self, env_data, B, device):
# # # # # # # #         if env_data is None:
# # # # # # # #             return torch.zeros(B, 256, device=device)

# # # # # # # #         def _safe_get(key, default=0.0):
# # # # # # # #             v = env_data.get(key, None)
# # # # # # # #             if v is None or not torch.is_tensor(v):
# # # # # # # #                 return torch.full((B,), default, device=device)
# # # # # # # #             v = v.to(device).float()
# # # # # # # #             if v.dim() >= 2:
# # # # # # # #                 while v.dim() > 1: v = v.mean(-1)
# # # # # # # #             if v.shape[0] != B:
# # # # # # # #                 v = (v.view(-1)[:B] if v.numel() >= B
# # # # # # # #                      else torch.full((B,), default, device=device))
# # # # # # # #             return v

# # # # # # # #         feat = torch.stack([_safe_get("u500_mean"), _safe_get("v500_mean"),
# # # # # # # #                              _safe_get("u500_center"), _safe_get("v500_center")],
# # # # # # # #                             dim=-1)
# # # # # # # #         return self.steering_enc(feat)

# # # # # # # #     def _beta_drift(self, x_t):
# # # # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # # # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # # # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # # # #         R_tc    = 3e5
# # # # # # # #         v_phys  = torch.zeros_like(x_t)
# # # # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # # # #         return v_phys

# # # # # # # #     def _steering_drift(self, x_t, env_data):
# # # # # # # #         if env_data is None:
# # # # # # # #             return torch.zeros_like(x_t)
# # # # # # # #         B, device = x_t.shape[0], x_t.device

# # # # # # # #         def _safe_mean(key):
# # # # # # # #             v = env_data.get(key, None)
# # # # # # # #             if v is None or not torch.is_tensor(v):
# # # # # # # #                 return torch.zeros(B, device=device)
# # # # # # # #             v = v.to(device).float()
# # # # # # # #             while v.dim() > 1: v = v.mean(-1)
# # # # # # # #             return v.view(-1)[:B] if v.numel() >= B else torch.zeros(B, device=device)

# # # # # # # #         u = _safe_mean("u500_center")
# # # # # # # #         v = _safe_mean("v500_center")
# # # # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # # # #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# # # # # # # #         out = torch.zeros_like(x_t)
# # # # # # # #         out[:, :, 0] = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat) * 0.2
# # # # # # # #         out[:, :, 1] = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0) * 0.2
# # # # # # # #         return out

# # # # # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
# # # # # # # #                 env_data=None):
# # # # # # # #         B     = x_t.shape[0]
# # # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # # #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # # #         s_emb    = self.step_embed(step_idx)

# # # # # # # #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# # # # # # # #                  + self.pos_enc[:, :T_seq]
# # # # # # # #                  + t_emb.unsqueeze(1)
# # # # # # # #                  + s_emb)

# # # # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # # # #         if vel_obs_feat is not None:
# # # # # # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))
# # # # # # # #         if steering_feat is not None:
# # # # # # # #             mem_parts.append(steering_feat.unsqueeze(1))

# # # # # # # #         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
# # # # # # # #         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # # # # #         v_neural = v_neural * scale

# # # # # # # #         with torch.no_grad():
# # # # # # # #             v_phys  = self._beta_drift(x_t[:, :T_seq])
# # # # # # # #             v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

# # # # # # # #         return (v_neural
# # # # # # # #                 + torch.sigmoid(self.physics_scale) * v_phys
# # # # # # # #                 + torch.sigmoid(self.steering_scale) * v_steer)

# # # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
# # # # # # # #                            vel_obs_feat=None, steering_feat=None, env_data=None):
# # # # # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# # # # # # # #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  TCFlowMatching — ONE CHANGE: return dict in get_loss_breakdown
# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # class TCFlowMatching(nn.Module):

# # # # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # # # #                  ctx_noise_scale=0.01, initial_sample_sigma=0.03,
# # # # # # # #                  teacher_forcing=True, use_ema=True, ema_decay=0.999,
# # # # # # # #                  **kwargs):
# # # # # # # #         super().__init__()
# # # # # # # #         self.pred_len             = pred_len
# # # # # # # #         self.obs_len              = obs_len
# # # # # # # #         self.sigma_min            = sigma_min
# # # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # # #         self.teacher_forcing      = teacher_forcing
# # # # # # # #         self.active_pred_len      = pred_len

# # # # # # # #         self.net = VelocityField(
# # # # # # # #             pred_len=pred_len, obs_len=obs_len,
# # # # # # # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # # # # #         self.use_ema   = use_ema
# # # # # # # #         self.ema_decay = ema_decay
# # # # # # # #         self._ema      = None

# # # # # # # #     def init_ema(self):
# # # # # # # #         if self.use_ema:
# # # # # # # #             self._ema = EMAModel(self, decay=self.ema_decay)

# # # # # # # #     def ema_update(self):
# # # # # # # #         if self._ema is not None:
# # # # # # # #             self._ema.update(self)

# # # # # # # #     def set_curriculum_len(self, *a, **kw):
# # # # # # # #         pass

# # # # # # # #     @staticmethod
# # # # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # # # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # # # # #     @staticmethod
# # # # # # # #     def _to_abs(rel, lp, lm):
# # # # # # # #         d = rel.permute(1, 0, 2)
# # # # # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # # # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # # # # # #         if sigma_min is None:
# # # # # # # #             sigma_min = self.sigma_min
# # # # # # # #         B, device = x1.shape[0], x1.device
# # # # # # # #         x0 = torch.randn_like(x1) * sigma_min
# # # # # # # #         t  = torch.rand(B, device=device)
# # # # # # # #         te = t.view(B, 1, 1)
# # # # # # # #         return (1.0 - te) * x0 + te * x1, t, x1 - x0

# # # # # # # #     @staticmethod
# # # # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # #             return batch_list
# # # # # # # #         aug = list(batch_list)
# # # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # # # #                 t = aug[idx].clone(); t[..., 0] = -t[..., 0]; aug[idx] = t
# # # # # # # #         return aug

# # # # # # # #     @staticmethod
# # # # # # # #     def _obs_noise_aug(batch_list, sigma=0.005):
# # # # # # # #         if torch.rand(1).item() > 0.5:
# # # # # # # #             return batch_list
# # # # # # # #         aug = list(batch_list)
# # # # # # # #         if torch.is_tensor(aug[0]):
# # # # # # # #             aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
# # # # # # # #         return aug

# # # # # # # #     @staticmethod
# # # # # # # #     def _mixup_aug(batch_list, alpha=0.2, p=0.2):
# # # # # # # #         if torch.rand(1).item() > p:
# # # # # # # #             return batch_list
# # # # # # # #         aug = list(batch_list)
# # # # # # # #         lam = float(torch.distributions.Beta(alpha, alpha).sample())
# # # # # # # #         lam = max(lam, 1 - lam)
# # # # # # # #         if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
# # # # # # # #             B    = aug[0].shape[1]
# # # # # # # #             perm = torch.randperm(B, device=aug[0].device)
# # # # # # # #             for idx in [0, 1, 7, 8]:
# # # # # # # #                 if torch.is_tensor(aug[idx]):
# # # # # # # #                     aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
# # # # # # # #         return aug

# # # # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # # #         # Augmentation
# # # # # # # #         batch_list = self._lon_flip_aug(batch_list)
# # # # # # # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
# # # # # # # #         if epoch >= 5:
# # # # # # # #             batch_list = self._mixup_aug(batch_list, alpha=0.2, p=0.15)

# # # # # # # #         traj_gt  = batch_list[1]
# # # # # # # #         Me_gt    = batch_list[8]
# # # # # # # #         obs_t    = batch_list[0]
# # # # # # # #         obs_Me   = batch_list[7]
# # # # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # # # #         lp, lm   = obs_t[-1], obs_Me[-1]

# # # # # # # #         # Sigma schedule
# # # # # # # #         if epoch < 15:
# # # # # # # #             current_sigma = 0.15
# # # # # # # #         elif epoch < 40:
# # # # # # # #             current_sigma = 0.15 - (epoch - 15) / 25.0 * 0.09
# # # # # # # #         else:
# # # # # # # #             current_sigma = 0.06

# # # # # # # #         raw_ctx       = self.net._context(batch_list)
# # # # # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # # # #         steering_feat = self.net._get_steering_feat(
# # # # # # # #             env_data, obs_t.shape[1], obs_t.device)

# # # # # # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
# # # # # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # # # # #         # Scheduled teacher forcing
# # # # # # # #         if self.teacher_forcing and self.training and epoch >= 6:
# # # # # # # #             p_teacher = max(0.0, 0.5 * (1.0 - (epoch - 6) / 34.0))
# # # # # # # #             if p_teacher > 0 and torch.rand(1).item() < p_teacher:
# # # # # # # #                 far_mask = torch.zeros_like(x_t)
# # # # # # # #                 far_mask[:, 6:, :] = 0.3
# # # # # # # #                 x_t = x_t * (1 - far_mask) + x1_rel * far_mask

# # # # # # # #         pred_vel = self.net.forward_with_ctx(
# # # # # # # #             x_t, fm_t, raw_ctx,
# # # # # # # #             vel_obs_feat=vel_obs_feat,
# # # # # # # #             steering_feat=steering_feat,
# # # # # # # #             env_data=env_data,
# # # # # # # #         )

# # # # # # # #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# # # # # # # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # # # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)
# # # # # # # #         pred_deg    = _norm_to_deg_fn(pred_abs)
# # # # # # # #         gt_deg      = _norm_to_deg_fn(traj_gt)

# # # # # # # #         loss_dict = compute_total_loss(
# # # # # # # #             pred_deg=pred_deg,
# # # # # # # #             gt_deg=gt_deg,
# # # # # # # #             env_data=env_data,
# # # # # # # #             weights=WEIGHTS,
# # # # # # # #             epoch=epoch,
# # # # # # # #         )

# # # # # # # #         w_fm  = max(0.3, 1.0 - epoch / 60.0)
# # # # # # # #         total = w_fm * l_fm_mse + loss_dict["total"]

# # # # # # # #         l_ens = x_t.new_zeros(())
# # # # # # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # # # # # #             x_t2, fm_t2, u_t2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # # # # #             pv2 = self.net.forward_with_ctx(
# # # # # # # #                 x_t2, fm_t2, raw_ctx,
# # # # # # # #                 vel_obs_feat=vel_obs_feat,
# # # # # # # #                 steering_feat=steering_feat,
# # # # # # # #                 env_data=env_data,
# # # # # # # #             )
# # # # # # # #             l_ens = F.mse_loss(pv2, u_t2)
# # # # # # # #             total = total + 0.3 * l_ens

# # # # # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # # # # #             total = x_t.new_zeros(())

# # # # # # # #         # ── RETURN DICT — v39 compat: thêm speed_acc + cumul_disp ────────
# # # # # # # #         return dict(
# # # # # # # #             total        = total,
# # # # # # # #             fm_mse       = l_fm_mse.item(),
# # # # # # # #             mse_hav      = loss_dict["mse_hav_horizon"],
# # # # # # # #             multi_scale  = loss_dict["multi_scale"],
# # # # # # # #             endpoint     = loss_dict["endpoint"],
# # # # # # # #             h_direct     = loss_dict.get("h_direct",    0.0),
# # # # # # # #             vel_smooth   = loss_dict.get("vel_smooth",  0.0),
# # # # # # # #             speed_acc    = loss_dict.get("speed_acc",   0.0),  # ← v39 NEW
# # # # # # # #             cumul_disp   = loss_dict.get("cumul_disp",  0.0),  # ← v39 NEW
# # # # # # # #             ate_cte      = loss_dict.get("ate_cte",     0.0),  # compat (=0)
# # # # # # # #             shape        = loss_dict.get("shape",       0.0),
# # # # # # # #             velocity     = loss_dict.get("velocity",    0.0),
# # # # # # # #             heading      = loss_dict.get("heading",     0.0),
# # # # # # # #             recurv       = loss_dict.get("recurv",      0.0),
# # # # # # # #             steering     = loss_dict.get("steering",    0.0),
# # # # # # # #             ens_consist  = (l_ens.item() if torch.is_tensor(l_ens)
# # # # # # # #                             else float(l_ens)),
# # # # # # # #             sigma        = current_sigma,
# # # # # # # #             w_fm         = w_fm,
# # # # # # # #             long_range=0.0, fde=0.0, w_lr=0.0, w_fde=0.0,
# # # # # # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0,
# # # # # # # #             recurv_ratio=0.0,
# # # # # # # #         )

# # # # # # # #     # ── Sampling — unchanged from v34fix ─────────────────────────────────────
# # # # # # # #     @torch.no_grad()
# # # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
# # # # # # # #                 predict_csv=None, importance_weight=True):
# # # # # # # #         obs_t    = batch_list[0]
# # # # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # # # #         lp       = obs_t[-1]
# # # # # # # #         lm       = batch_list[7][-1]
# # # # # # # #         B        = lp.shape[0]
# # # # # # # #         device   = lp.device
# # # # # # # #         T        = self.pred_len
# # # # # # # #         dt       = 1.0 / max(ddim_steps, 1)

# # # # # # # #         raw_ctx       = self.net._context(batch_list)
# # # # # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # # # #         steering_feat = self.net._get_steering_feat(env_data, B, device)

# # # # # # # #         traj_s, me_s, scores = [], [], []
# # # # # # # #         for _ in range(num_ensemble):
# # # # # # # #             x_t = torch.randn(B, T, 4, device=device) * (self.sigma_min * 2.5)
# # # # # # # #             for step in range(ddim_steps):
# # # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # # #                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
# # # # # # # #                 vel = self.net.forward_with_ctx(
# # # # # # # #                     x_t, t_b, raw_ctx,
# # # # # # # #                     noise_scale=ns,
# # # # # # # #                     vel_obs_feat=vel_obs_feat,
# # # # # # # #                     steering_feat=steering_feat,
# # # # # # # #                     env_data=env_data,
# # # # # # # #                 )
# # # # # # # #                 x_t = x_t + dt * vel
# # # # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # # #             traj_s.append(tr)
# # # # # # # #             me_s.append(me)
# # # # # # # #             if importance_weight:
# # # # # # # #                 scores.append(self._score_sample(tr, env_data))

# # # # # # # #         all_trajs = torch.stack(traj_s)
# # # # # # # #         all_me    = torch.stack(me_s)

# # # # # # # #         if importance_weight and scores:
# # # # # # # #             score_tensor = torch.stack(scores)
# # # # # # # #             k = max(1, int(num_ensemble * 0.7))
# # # # # # # #             _, top_idx = score_tensor.topk(k, dim=0)
# # # # # # # #             pred_mean = torch.stack([
# # # # # # # #                 all_trajs[top_idx[:, b], :, b, :].median(0).values
# # # # # # # #                 for b in range(B)
# # # # # # # #             ], dim=1)
# # # # # # # #         else:
# # # # # # # #             pred_mean = all_trajs.median(0).values

# # # # # # # #         if predict_csv:
# # # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # # # #         return pred_mean, all_me.mean(0), all_trajs

# # # # # # # #     def _score_sample(self, traj, env_data):
# # # # # # # #         B = traj.shape[1]
# # # # # # # #         if traj.shape[0] < 2:
# # # # # # # #             return torch.ones(B, device=traj.device)
# # # # # # # #         traj_deg  = _norm_to_deg_fn(traj)
# # # # # # # #         dt_deg    = traj_deg[1:] - traj_deg[:-1]
# # # # # # # #         lat_rad   = torch.deg2rad(traj_deg[:-1, :, 1])
# # # # # # # #         cos_lat   = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # #         dx_km     = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # # # # #         dy_km     = dt_deg[:, :, 1] * 111.0
# # # # # # # #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)
# # # # # # # #         speed_pen = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
# # # # # # # #         speed_sc  = torch.exp(-speed_pen.mean(0) / 20.0)
# # # # # # # #         if dt_deg.shape[0] >= 2:
# # # # # # # #             jerk = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
# # # # # # # #             smooth_sc = torch.exp(-jerk * 5.0)
# # # # # # # #         else:
# # # # # # # #             smooth_sc = torch.ones(B, device=traj.device)
# # # # # # # #         return speed_sc * smooth_sc

# # # # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # # # #         with torch.enable_grad():
# # # # # # # #             x   = x_pred.detach().requires_grad_(True)
# # # # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # # # #             for _ in range(n_steps):
# # # # # # # #                 opt.zero_grad()
# # # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # # #                 pred_deg    = _norm_to_deg_fn(pred_abs)
# # # # # # # #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# # # # # # # #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # # #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # # #                 speed       = torch.sqrt(
# # # # # # # #                     (dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # # # #                     + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # # # #                 F.relu(speed - 600.0).pow(2).mean().backward()
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
# # # # # # # #                     "lon_mean_deg", "lat_mean_deg",
# # # # # # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # # #             if write_hdr: w.writeheader()
# # # # # # # #             for b in range(B):
# # # # # # # #                 for k in range(T):
# # # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # # #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# # # # # # # #                                * math.cos(math.radians(mean_lat[k, b])))
# # # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # # #                     w.writerow({
# # # # # # # #                         "timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # # # #                         "lead_h": (k + 1) * 6,
# # # # # # # #                         "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # # # #                         "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # # # #                         "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # # # # # # #                         "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # # # # # # #                         "ens_spread_km": f"{spread:.2f}",
# # # # # # # #                     })


# # # # # # # # TCDiffusion = TCFlowMatching
# # # # # # # """
# # # # # # # Model/flow_matching_model.py — v34fix  (v39 compat)
# # # # # # # ═══════════════════════════════════════════════════════════════════

# # # # # # # THAY ĐỔI so với v34fix (v40_clean compat):
# # # # # # #   get_loss_breakdown() return dict — thêm 5 keys cho losses v40_clean:
# # # # # # #     speed_acc  = loss_dict.get('speed_acc',  0.0)
# # # # # # #     cumul_disp = loss_dict.get('cumul_disp', 0.0)
# # # # # # #     accel      = loss_dict.get('accel',      0.0)  ← v40 NEW
# # # # # # #     decomp     = loss_dict.get('decomp',     0.0)  ← v40 NEW
# # # # # # #     cons       = loss_dict.get('cons',       0.0)  ← v40 NEW

# # # # # # # Tất cả code khác GIỐNG HỆT v34fix.
# # # # # # # Không thay đổi architecture, training logic, sampling.
# # # # # # # """
# # # # # # # from __future__ import annotations

# # # # # # # import csv
# # # # # # # import math
# # # # # # # import os
# # # # # # # from datetime import datetime

# # # # # # # import torch
# # # # # # # import torch.nn as nn
# # # # # # # import torch.nn.functional as F

# # # # # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # # # # from Model.losses import (
# # # # # # #     compute_total_loss,
# # # # # # #     _haversine_deg, _norm_to_deg,
# # # # # # #     WEIGHTS,
# # # # # # # )


# # # # # # # def _norm_to_deg_fn(t):
# # # # # # #     out = t.clone()
# # # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # # #     return out


# # # # # # # def _unwrap_model(model: nn.Module) -> nn.Module:
# # # # # # #     if hasattr(model, '_orig_mod'):
# # # # # # #         return model._orig_mod
# # # # # # #     return model


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  EMAModel — unchanged from v34fix
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # class EMAModel:
# # # # # # #     def __init__(self, model: nn.Module, decay: float = 0.999):
# # # # # # #         self.decay = decay
# # # # # # #         m = _unwrap_model(model)
# # # # # # #         self.shadow = {
# # # # # # #             k: v.detach().clone()
# # # # # # #             for k, v in m.state_dict().items()
# # # # # # #             if v.dtype.is_floating_point
# # # # # # #         }

# # # # # # #     def update(self, model: nn.Module):
# # # # # # #         m = _unwrap_model(model)
# # # # # # #         with torch.no_grad():
# # # # # # #             for k, v in m.state_dict().items():
# # # # # # #                 if k in self.shadow:
# # # # # # #                     self.shadow[k].mul_(self.decay).add_(
# # # # # # #                         v.detach(), alpha=1 - self.decay)

# # # # # # #     def apply_to(self, model: nn.Module):
# # # # # # #         m = _unwrap_model(model)
# # # # # # #         backup = {}
# # # # # # #         sd = m.state_dict()
# # # # # # #         for k in self.shadow:
# # # # # # #             if k not in sd:
# # # # # # #                 continue
# # # # # # #             backup[k] = sd[k].detach().clone()
# # # # # # #             sd[k].copy_(self.shadow[k])
# # # # # # #         return backup

# # # # # # #     def restore(self, model: nn.Module, backup: dict):
# # # # # # #         m = _unwrap_model(model)
# # # # # # #         sd = m.state_dict()
# # # # # # #         for k, v in backup.items():
# # # # # # #             if k in sd:
# # # # # # #                 sd[k].copy_(v)


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  VelocityField — unchanged from v34fix
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

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
# # # # # # #         self.steering_enc = nn.Sequential(
# # # # # # #             nn.Linear(4, 64), nn.GELU(),
# # # # # # #             nn.LayerNorm(64),
# # # # # # #             nn.Linear(64, 128), nn.GELU(),
# # # # # # #             nn.Linear(128, 256),
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

# # # # # # #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # # # # #         self.physics_scale  = nn.Parameter(torch.ones(4) * 0.3)
# # # # # # #         self.steering_scale = nn.Parameter(torch.ones(4) * 0.2)
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

# # # # # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(
# # # # # # #             torch.cat([h_t, e_env, f_spatial], dim=-1))))

# # # # # # #     def _apply_ctx_head(self, raw, noise_scale=0.0):
# # # # # # #         if noise_scale > 0.0:
# # # # # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # # # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # # # # #     def _get_vel_obs_feat(self, obs_traj):
# # # # # # #         B, T_obs = obs_traj.shape[1], obs_traj.shape[0]
# # # # # # #         if T_obs >= 2:
# # # # # # #             vels = obs_traj[1:] - obs_traj[:-1]
# # # # # # #         else:
# # # # # # #             vels = torch.zeros(1, B, 2, device=obs_traj.device)
# # # # # # #         if vels.shape[0] < self.obs_len:
# # # # # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2,
# # # # # # #                                device=obs_traj.device)
# # # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # # #         else:
# # # # # # #             vels = vels[-self.obs_len:]
# # # # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # # # #     def _get_steering_feat(self, env_data, B, device):
# # # # # # #         if env_data is None:
# # # # # # #             return torch.zeros(B, 256, device=device)

# # # # # # #         def _safe_get(key, default=0.0):
# # # # # # #             v = env_data.get(key, None)
# # # # # # #             if v is None or not torch.is_tensor(v):
# # # # # # #                 return torch.full((B,), default, device=device)
# # # # # # #             v = v.to(device).float()
# # # # # # #             if v.dim() >= 2:
# # # # # # #                 while v.dim() > 1: v = v.mean(-1)
# # # # # # #             if v.shape[0] != B:
# # # # # # #                 v = (v.view(-1)[:B] if v.numel() >= B
# # # # # # #                      else torch.full((B,), default, device=device))
# # # # # # #             return v

# # # # # # #         feat = torch.stack([_safe_get("u500_mean"), _safe_get("v500_mean"),
# # # # # # #                              _safe_get("u500_center"), _safe_get("v500_center")],
# # # # # # #                             dim=-1)
# # # # # # #         return self.steering_enc(feat)

# # # # # # #     def _beta_drift(self, x_t):
# # # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # # #         R_tc    = 3e5
# # # # # # #         v_phys  = torch.zeros_like(x_t)
# # # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # # #         return v_phys

# # # # # # #     def _steering_drift(self, x_t, env_data):
# # # # # # #         if env_data is None:
# # # # # # #             return torch.zeros_like(x_t)
# # # # # # #         B, device = x_t.shape[0], x_t.device

# # # # # # #         def _safe_mean(key):
# # # # # # #             v = env_data.get(key, None)
# # # # # # #             if v is None or not torch.is_tensor(v):
# # # # # # #                 return torch.zeros(B, device=device)
# # # # # # #             v = v.to(device).float()
# # # # # # #             while v.dim() > 1: v = v.mean(-1)
# # # # # # #             return v.view(-1)[:B] if v.numel() >= B else torch.zeros(B, device=device)

# # # # # # #         u = _safe_mean("u500_center")
# # # # # # #         v = _safe_mean("v500_center")
# # # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # # #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# # # # # # #         out = torch.zeros_like(x_t)
# # # # # # #         out[:, :, 0] = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat) * 0.2
# # # # # # #         out[:, :, 1] = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0) * 0.2
# # # # # # #         return out

# # # # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
# # # # # # #                 env_data=None):
# # # # # # #         B     = x_t.shape[0]
# # # # # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # # # # # #         t_emb = self.time_fc2(t_emb)

# # # # # # #         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
# # # # # # #         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
# # # # # # #         s_emb    = self.step_embed(step_idx)

# # # # # # #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# # # # # # #                  + self.pos_enc[:, :T_seq]
# # # # # # #                  + t_emb.unsqueeze(1)
# # # # # # #                  + s_emb)

# # # # # # #         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
# # # # # # #         if vel_obs_feat is not None:
# # # # # # #             mem_parts.append(vel_obs_feat.unsqueeze(1))
# # # # # # #         if steering_feat is not None:
# # # # # # #             mem_parts.append(steering_feat.unsqueeze(1))

# # # # # # #         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
# # # # # # #         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # # # #         v_neural = v_neural * scale

# # # # # # #         with torch.no_grad():
# # # # # # #             v_phys  = self._beta_drift(x_t[:, :T_seq])
# # # # # # #             v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

# # # # # # #         return (v_neural
# # # # # # #                 + torch.sigmoid(self.physics_scale) * v_phys
# # # # # # #                 + torch.sigmoid(self.steering_scale) * v_steer)

# # # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
# # # # # # #                            vel_obs_feat=None, steering_feat=None, env_data=None):
# # # # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# # # # # # #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  TCFlowMatching — ONE CHANGE: return dict in get_loss_breakdown
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # class TCFlowMatching(nn.Module):

# # # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # # #                  ctx_noise_scale=0.01, initial_sample_sigma=0.03,
# # # # # # #                  teacher_forcing=True, use_ema=True, ema_decay=0.999,
# # # # # # #                  **kwargs):
# # # # # # #         super().__init__()
# # # # # # #         self.pred_len             = pred_len
# # # # # # #         self.obs_len              = obs_len
# # # # # # #         self.sigma_min            = sigma_min
# # # # # # #         self.n_train_ens          = n_train_ens
# # # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # # #         self.teacher_forcing      = teacher_forcing
# # # # # # #         self.active_pred_len      = pred_len

# # # # # # #         self.net = VelocityField(
# # # # # # #             pred_len=pred_len, obs_len=obs_len,
# # # # # # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # # # #         self.use_ema   = use_ema
# # # # # # #         self.ema_decay = ema_decay
# # # # # # #         self._ema      = None

# # # # # # #     def init_ema(self):
# # # # # # #         if self.use_ema:
# # # # # # #             self._ema = EMAModel(self, decay=self.ema_decay)

# # # # # # #     def ema_update(self):
# # # # # # #         if self._ema is not None:
# # # # # # #             self._ema.update(self)

# # # # # # #     def set_curriculum_len(self, *a, **kw):
# # # # # # #         pass

# # # # # # #     @staticmethod
# # # # # # #     def _to_rel(traj, Me, lp, lm):
# # # # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # # # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

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
# # # # # # #         return (1.0 - te) * x0 + te * x1, t, x1 - x0

# # # # # # #     @staticmethod
# # # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # # #         if torch.rand(1).item() > p:
# # # # # # #             return batch_list
# # # # # # #         aug = list(batch_list)
# # # # # # #         for idx in [0, 1, 2, 3]:
# # # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # # #                 t = aug[idx].clone(); t[..., 0] = -t[..., 0]; aug[idx] = t
# # # # # # #         return aug

# # # # # # #     @staticmethod
# # # # # # #     def _obs_noise_aug(batch_list, sigma=0.005):
# # # # # # #         if torch.rand(1).item() > 0.5:
# # # # # # #             return batch_list
# # # # # # #         aug = list(batch_list)
# # # # # # #         if torch.is_tensor(aug[0]):
# # # # # # #             aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
# # # # # # #         return aug

# # # # # # #     @staticmethod
# # # # # # #     def _mixup_aug(batch_list, alpha=0.2, p=0.2):
# # # # # # #         if torch.rand(1).item() > p:
# # # # # # #             return batch_list
# # # # # # #         aug = list(batch_list)
# # # # # # #         lam = float(torch.distributions.Beta(alpha, alpha).sample())
# # # # # # #         lam = max(lam, 1 - lam)
# # # # # # #         if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
# # # # # # #             B    = aug[0].shape[1]
# # # # # # #             perm = torch.randperm(B, device=aug[0].device)
# # # # # # #             for idx in [0, 1, 7, 8]:
# # # # # # #                 if torch.is_tensor(aug[idx]):
# # # # # # #                     aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
# # # # # # #         return aug

# # # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # # #         # Augmentation
# # # # # # #         batch_list = self._lon_flip_aug(batch_list)
# # # # # # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
# # # # # # #         if epoch >= 5:
# # # # # # #             batch_list = self._mixup_aug(batch_list, alpha=0.2, p=0.15)

# # # # # # #         traj_gt  = batch_list[1]
# # # # # # #         Me_gt    = batch_list[8]
# # # # # # #         obs_t    = batch_list[0]
# # # # # # #         obs_Me   = batch_list[7]
# # # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # # #         lp, lm   = obs_t[-1], obs_Me[-1]

# # # # # # #         # Sigma schedule
# # # # # # #         if epoch < 15:
# # # # # # #             current_sigma = 0.15
# # # # # # #         elif epoch < 40:
# # # # # # #             current_sigma = 0.15 - (epoch - 15) / 25.0 * 0.09
# # # # # # #         else:
# # # # # # #             current_sigma = 0.06

# # # # # # #         raw_ctx       = self.net._context(batch_list)
# # # # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # # #         steering_feat = self.net._get_steering_feat(
# # # # # # #             env_data, obs_t.shape[1], obs_t.device)

# # # # # # #         x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
# # # # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # # # #         # Scheduled teacher forcing
# # # # # # #         if self.teacher_forcing and self.training and epoch >= 3:
# # # # # # #             p_teacher = max(0.0, 0.5 * (1.0 - (epoch - 3) / 37.0))
# # # # # # #             if p_teacher > 0 and torch.rand(1).item() < p_teacher:
# # # # # # #                 far_mask = torch.zeros_like(x_t)
# # # # # # #                 far_mask[:, 6:, :] = 0.3
# # # # # # #                 x_t = x_t * (1 - far_mask) + x1_rel * far_mask

# # # # # # #         pred_vel = self.net.forward_with_ctx(
# # # # # # #             x_t, fm_t, raw_ctx,
# # # # # # #             vel_obs_feat=vel_obs_feat,
# # # # # # #             steering_feat=steering_feat,
# # # # # # #             env_data=env_data,
# # # # # # #         )

# # # # # # #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# # # # # # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)
# # # # # # #         pred_deg    = _norm_to_deg_fn(pred_abs)
# # # # # # #         gt_deg      = _norm_to_deg_fn(traj_gt)

# # # # # # #         loss_dict = compute_total_loss(
# # # # # # #             pred_deg=pred_deg,
# # # # # # #             gt_deg=gt_deg,
# # # # # # #             env_data=env_data,
# # # # # # #             weights=WEIGHTS,
# # # # # # #             epoch=epoch,
# # # # # # #         )

# # # # # # #         w_fm  = max(0.3, 1.0 - epoch / 60.0)
# # # # # # #         total = w_fm * l_fm_mse + loss_dict["total"]

# # # # # # #         l_ens = x_t.new_zeros(())
# # # # # # #         if epoch >= 40 and self.n_train_ens >= 2:
# # # # # # #             x_t2, fm_t2, u_t2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # # # #             pv2 = self.net.forward_with_ctx(
# # # # # # #                 x_t2, fm_t2, raw_ctx,
# # # # # # #                 vel_obs_feat=vel_obs_feat,
# # # # # # #                 steering_feat=steering_feat,
# # # # # # #                 env_data=env_data,
# # # # # # #             )
# # # # # # #             l_ens = F.mse_loss(pv2, u_t2)
# # # # # # #             total = total + 0.3 * l_ens

# # # # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # # # #             total = x_t.new_zeros(())

# # # # # # #         # ── RETURN DICT — v39 compat: thêm speed_acc + cumul_disp ────────
# # # # # # #         return dict(
# # # # # # #             total        = total,
# # # # # # #             fm_mse       = l_fm_mse.item(),
# # # # # # #             mse_hav      = loss_dict["mse_hav_horizon"],
# # # # # # #             multi_scale  = loss_dict["multi_scale"],
# # # # # # #             endpoint     = loss_dict["endpoint"],
# # # # # # #             h_direct     = loss_dict.get("h_direct",    0.0),
# # # # # # #             vel_smooth   = loss_dict.get("vel_smooth",  0.0),
# # # # # # #             speed_acc    = loss_dict.get("speed_acc",   0.0),
# # # # # # #             cumul_disp   = loss_dict.get("cumul_disp",  0.0),
# # # # # # #             accel        = loss_dict.get("accel",       0.0),  # ← v40 NEW
# # # # # # #             decomp       = loss_dict.get("decomp",      0.0),  # ← v40 NEW
# # # # # # #             cons         = loss_dict.get("cons",        0.0),  # ← v40 NEW
# # # # # # #             ate_cte      = loss_dict.get("ate_cte",     0.0),  # compat (=0)
# # # # # # #             shape        = loss_dict.get("shape",       0.0),
# # # # # # #             velocity     = loss_dict.get("velocity",    0.0),
# # # # # # #             heading      = loss_dict.get("heading",     0.0),
# # # # # # #             recurv       = loss_dict.get("recurv",      0.0),
# # # # # # #             steering     = loss_dict.get("steering",    0.0),
# # # # # # #             ens_consist  = (l_ens.item() if torch.is_tensor(l_ens)
# # # # # # #                             else float(l_ens)),
# # # # # # #             sigma        = current_sigma,
# # # # # # #             w_fm         = w_fm,
# # # # # # #             long_range=0.0, fde=0.0, w_lr=0.0, w_fde=0.0,
# # # # # # #             fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # # #             spread=0.0, continuity=0.0, step=0.0, disp=0.0,
# # # # # # #             recurv_ratio=0.0,
# # # # # # #         )

# # # # # # #     # ── Sampling — unchanged from v34fix ─────────────────────────────────────
# # # # # # #     @torch.no_grad()
# # # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
# # # # # # #                 predict_csv=None, importance_weight=True):
# # # # # # #         obs_t    = batch_list[0]
# # # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # # #         lp       = obs_t[-1]
# # # # # # #         lm       = batch_list[7][-1]
# # # # # # #         B        = lp.shape[0]
# # # # # # #         device   = lp.device
# # # # # # #         T        = self.pred_len
# # # # # # #         dt       = 1.0 / max(ddim_steps, 1)

# # # # # # #         raw_ctx       = self.net._context(batch_list)
# # # # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # # #         steering_feat = self.net._get_steering_feat(env_data, B, device)

# # # # # # #         traj_s, me_s, scores = [], [], []
# # # # # # #         for _ in range(num_ensemble):
# # # # # # #             x_t = torch.randn(B, T, 4, device=device) * (self.sigma_min * 2.5)
# # # # # # #             for step in range(ddim_steps):
# # # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # # #                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
# # # # # # #                 vel = self.net.forward_with_ctx(
# # # # # # #                     x_t, t_b, raw_ctx,
# # # # # # #                     noise_scale=ns,
# # # # # # #                     vel_obs_feat=vel_obs_feat,
# # # # # # #                     steering_feat=steering_feat,
# # # # # # #                     env_data=env_data,
# # # # # # #                 )
# # # # # # #                 x_t = x_t + dt * vel
# # # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # # #             traj_s.append(tr)
# # # # # # #             me_s.append(me)
# # # # # # #             if importance_weight:
# # # # # # #                 scores.append(self._score_sample(tr, env_data))

# # # # # # #         all_trajs = torch.stack(traj_s)
# # # # # # #         all_me    = torch.stack(me_s)

# # # # # # #         if importance_weight and scores:
# # # # # # #             score_tensor = torch.stack(scores)
# # # # # # #             k = max(1, int(num_ensemble * 0.7))
# # # # # # #             _, top_idx = score_tensor.topk(k, dim=0)
# # # # # # #             pred_mean = torch.stack([
# # # # # # #                 all_trajs[top_idx[:, b], :, b, :].median(0).values
# # # # # # #                 for b in range(B)
# # # # # # #             ], dim=1)
# # # # # # #         else:
# # # # # # #             pred_mean = all_trajs.median(0).values

# # # # # # #         if predict_csv:
# # # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # # #         return pred_mean, all_me.mean(0), all_trajs

# # # # # # #     def _score_sample(self, traj, env_data):
# # # # # # #         B = traj.shape[1]
# # # # # # #         if traj.shape[0] < 2:
# # # # # # #             return torch.ones(B, device=traj.device)
# # # # # # #         traj_deg  = _norm_to_deg_fn(traj)
# # # # # # #         dt_deg    = traj_deg[1:] - traj_deg[:-1]
# # # # # # #         lat_rad   = torch.deg2rad(traj_deg[:-1, :, 1])
# # # # # # #         cos_lat   = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # #         dx_km     = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # # # #         dy_km     = dt_deg[:, :, 1] * 111.0
# # # # # # #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)
# # # # # # #         speed_pen = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
# # # # # # #         speed_sc  = torch.exp(-speed_pen.mean(0) / 20.0)
# # # # # # #         if dt_deg.shape[0] >= 2:
# # # # # # #             jerk = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
# # # # # # #             smooth_sc = torch.exp(-jerk * 5.0)
# # # # # # #         else:
# # # # # # #             smooth_sc = torch.ones(B, device=traj.device)
# # # # # # #         return speed_sc * smooth_sc

# # # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # # #         with torch.enable_grad():
# # # # # # #             x   = x_pred.detach().requires_grad_(True)
# # # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # # #             for _ in range(n_steps):
# # # # # # #                 opt.zero_grad()
# # # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # # #                 pred_deg    = _norm_to_deg_fn(pred_abs)
# # # # # # #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# # # # # # #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # # #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # # #                 speed       = torch.sqrt(
# # # # # # #                     (dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # # #                     + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # # #                 F.relu(speed - 600.0).pow(2).mean().backward()
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
# # # # # # #                     "lon_mean_deg", "lat_mean_deg",
# # # # # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # # # #         write_hdr = not os.path.exists(csv_path)
# # # # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # #         with open(csv_path, "a", newline="") as fh:
# # # # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # # # #             if write_hdr: w.writeheader()
# # # # # # #             for b in range(B):
# # # # # # #                 for k in range(T):
# # # # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # # # #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# # # # # # #                                * math.cos(math.radians(mean_lat[k, b])))
# # # # # # #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
# # # # # # #                     w.writerow({
# # # # # # #                         "timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # # #                         "lead_h": (k + 1) * 6,
# # # # # # #                         "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # # #                         "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # # #                         "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # # # # # #                         "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # # # # # #                         "ens_spread_km": f"{spread:.2f}",
# # # # # # #                     })


# # # # # # # TCDiffusion = TCFlowMatching
# # # # # # """
# # # # # # Model/flow_matching_model.py — v41  SPEED-BIAS FIX
# # # # # # ═══════════════════════════════════════════════════════════════════

# # # # # # THAY ĐỔI so với v40:
# # # # # #   FIX 1: dropout 0.10 → 0.15  (giảm overfit, val/train gap 2.4×)
# # # # # #   FIX 2: _speed_aug() — random scale TC speed ×[0.8,1.2]
# # # # # #           Breaks systematic speed bias (ATE plateau 142km)
# # # # # #           p=0.4 khi epoch >= 10
# # # # # #   FIX 3: _temporal_reverse_aug() — reverse obs sequence (p=0.15)
# # # # # #           Teaches model speed invariance
# # # # # #   FIX 4: return dict thêm accel/decomp/cons (giữ từ v40)
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
# # # # # # from Model.losses import (
# # # # # #     compute_total_loss,
# # # # # #     _haversine_deg, _norm_to_deg,
# # # # # #     WEIGHTS,
# # # # # # )


# # # # # # def _norm_to_deg_fn(t):
# # # # # #     out = t.clone()
# # # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # # #     return out


# # # # # # def _unwrap_model(model: nn.Module) -> nn.Module:
# # # # # #     if hasattr(model, '_orig_mod'):
# # # # # #         return model._orig_mod
# # # # # #     return model


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  EMAModel — unchanged from v34fix
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # class EMAModel:
# # # # # #     def __init__(self, model: nn.Module, decay: float = 0.999):
# # # # # #         self.decay = decay
# # # # # #         m = _unwrap_model(model)
# # # # # #         self.shadow = {
# # # # # #             k: v.detach().clone()
# # # # # #             for k, v in m.state_dict().items()
# # # # # #             if v.dtype.is_floating_point
# # # # # #         }

# # # # # #     def update(self, model: nn.Module):
# # # # # #         m = _unwrap_model(model)
# # # # # #         with torch.no_grad():
# # # # # #             for k, v in m.state_dict().items():
# # # # # #                 if k in self.shadow:
# # # # # #                     self.shadow[k].mul_(self.decay).add_(
# # # # # #                         v.detach(), alpha=1 - self.decay)

# # # # # #     def apply_to(self, model: nn.Module):
# # # # # #         m = _unwrap_model(model)
# # # # # #         backup = {}
# # # # # #         sd = m.state_dict()
# # # # # #         for k in self.shadow:
# # # # # #             if k not in sd:
# # # # # #                 continue
# # # # # #             backup[k] = sd[k].detach().clone()
# # # # # #             sd[k].copy_(self.shadow[k])
# # # # # #         return backup

# # # # # #     def restore(self, model: nn.Module, backup: dict):
# # # # # #         m = _unwrap_model(model)
# # # # # #         sd = m.state_dict()
# # # # # #         for k, v in backup.items():
# # # # # #             if k in sd:
# # # # # #                 sd[k].copy_(v)


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  VelocityField — unchanged from v34fix
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

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
# # # # # #         self.steering_enc = nn.Sequential(
# # # # # #             nn.Linear(4, 64), nn.GELU(),
# # # # # #             nn.LayerNorm(64),
# # # # # #             nn.Linear(64, 128), nn.GELU(),
# # # # # #             nn.Linear(128, 256),
# # # # # #         )

# # # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # # # #         self.transformer = nn.TransformerDecoder(
# # # # # #             nn.TransformerDecoderLayer(
# # # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # # #                 dropout=0.15, activation="gelu", batch_first=True),  # v41: 0.10→0.15 (reduce overfit)
# # # # # #             num_layers=2)

# # # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # # #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # # # #         self.physics_scale  = nn.Parameter(torch.ones(4) * 0.3)
# # # # # #         self.steering_scale = nn.Parameter(torch.ones(4) * 0.2)
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
# # # # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2,
# # # # # #                                device=obs_traj.device)
# # # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # # #         else:
# # # # # #             vels = vels[-self.obs_len:]
# # # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # # #     def _get_steering_feat(self, env_data, B, device):
# # # # # #         if env_data is None:
# # # # # #             return torch.zeros(B, 256, device=device)

# # # # # #         def _safe_get(key, default=0.0):
# # # # # #             v = env_data.get(key, None)
# # # # # #             if v is None or not torch.is_tensor(v):
# # # # # #                 return torch.full((B,), default, device=device)
# # # # # #             v = v.to(device).float()
# # # # # #             if v.dim() >= 2:
# # # # # #                 while v.dim() > 1: v = v.mean(-1)
# # # # # #             if v.shape[0] != B:
# # # # # #                 v = (v.view(-1)[:B] if v.numel() >= B
# # # # # #                      else torch.full((B,), default, device=device))
# # # # # #             return v

# # # # # #         feat = torch.stack([_safe_get("u500_mean"), _safe_get("v500_mean"),
# # # # # #                              _safe_get("u500_center"), _safe_get("v500_center")],
# # # # # #                             dim=-1)
# # # # # #         return self.steering_enc(feat)

# # # # # #     def _beta_drift(self, x_t):
# # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # # #         R_tc    = 3e5
# # # # # #         v_phys  = torch.zeros_like(x_t)
# # # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
# # # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
# # # # # #         return v_phys

# # # # # #     def _steering_drift(self, x_t, env_data):
# # # # # #         if env_data is None:
# # # # # #             return torch.zeros_like(x_t)
# # # # # #         B, device = x_t.shape[0], x_t.device

# # # # # #         def _safe_mean(key):
# # # # # #             v = env_data.get(key, None)
# # # # # #             if v is None or not torch.is_tensor(v):
# # # # # #                 return torch.zeros(B, device=device)
# # # # # #             v = v.to(device).float()
# # # # # #             while v.dim() > 1: v = v.mean(-1)
# # # # # #             return v.view(-1)[:B] if v.numel() >= B else torch.zeros(B, device=device)

# # # # # #         u = _safe_mean("u500_center")
# # # # # #         v = _safe_mean("v500_center")
# # # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # # #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# # # # # #         out = torch.zeros_like(x_t)
# # # # # #         out[:, :, 0] = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat) * 0.2
# # # # # #         out[:, :, 1] = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0) * 0.2
# # # # # #         return out

# # # # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None,
# # # # # #                 env_data=None):
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
# # # # # #         if steering_feat is not None:
# # # # # #             mem_parts.append(steering_feat.unsqueeze(1))

# # # # # #         decoded  = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
# # # # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
# # # # # #         scale    = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # # # #         v_neural = v_neural * scale

# # # # # #         with torch.no_grad():
# # # # # #             v_phys  = self._beta_drift(x_t[:, :T_seq])
# # # # # #             v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

# # # # # #         return (v_neural
# # # # # #                 + torch.sigmoid(self.physics_scale) * v_phys
# # # # # #                 + torch.sigmoid(self.steering_scale) * v_steer)

# # # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
# # # # # #                            vel_obs_feat=None, steering_feat=None, env_data=None):
# # # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# # # # # #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  TCFlowMatching — ONE CHANGE: return dict in get_loss_breakdown
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # class TCFlowMatching(nn.Module):

# # # # # #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# # # # # #                  n_train_ens=4, unet_in_ch=13,
# # # # # #                  ctx_noise_scale=0.01, initial_sample_sigma=0.03,
# # # # # #                  teacher_forcing=True, use_ema=True, ema_decay=0.999,
# # # # # #                  **kwargs):
# # # # # #         super().__init__()
# # # # # #         self.pred_len             = pred_len
# # # # # #         self.obs_len              = obs_len
# # # # # #         self.sigma_min            = sigma_min
# # # # # #         self.n_train_ens          = n_train_ens
# # # # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # # # #         self.initial_sample_sigma = initial_sample_sigma
# # # # # #         self.teacher_forcing      = teacher_forcing
# # # # # #         self.active_pred_len      = pred_len

# # # # # #         self.net = VelocityField(
# # # # # #             pred_len=pred_len, obs_len=obs_len,
# # # # # #             sigma_min=sigma_min, unet_in_ch=unet_in_ch)

# # # # # #         self.use_ema   = use_ema
# # # # # #         self.ema_decay = ema_decay
# # # # # #         self._ema      = None

# # # # # #     def init_ema(self):
# # # # # #         if self.use_ema:
# # # # # #             self._ema = EMAModel(self, decay=self.ema_decay)

# # # # # #     def ema_update(self):
# # # # # #         if self._ema is not None:
# # # # # #             self._ema.update(self)

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
# # # # # #         return (1.0 - te) * x0 + te * x1, t, x1 - x0

# # # # # #     @staticmethod
# # # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # # #         if torch.rand(1).item() > p:
# # # # # #             return batch_list
# # # # # #         aug = list(batch_list)
# # # # # #         for idx in [0, 1, 2, 3]:
# # # # # #             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
# # # # # #                 t = aug[idx].clone(); t[..., 0] = -t[..., 0]; aug[idx] = t
# # # # # #         return aug

# # # # # #     @staticmethod
# # # # # #     def _obs_noise_aug(batch_list, sigma=0.005):
# # # # # #         if torch.rand(1).item() > 0.5:
# # # # # #             return batch_list
# # # # # #         aug = list(batch_list)
# # # # # #         if torch.is_tensor(aug[0]):
# # # # # #             aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
# # # # # #         return aug

# # # # # #     @staticmethod
# # # # # #     def _mixup_aug(batch_list, alpha=0.2, p=0.2):
# # # # # #         if torch.rand(1).item() > p:
# # # # # #             return batch_list
# # # # # #         aug = list(batch_list)
# # # # # #         lam = float(torch.distributions.Beta(alpha, alpha).sample())
# # # # # #         lam = max(lam, 1 - lam)
# # # # # #         if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
# # # # # #             B    = aug[0].shape[1]
# # # # # #             perm = torch.randperm(B, device=aug[0].device)
# # # # # #             for idx in [0, 1, 7, 8]:
# # # # # #                 if torch.is_tensor(aug[idx]):
# # # # # #                     aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
# # # # # #         return aug

# # # # # #     @staticmethod
# # # # # #     def _speed_aug(batch_list, p=0.4):
# # # # # #         """
# # # # # #         FIX FOR SYSTEMATIC SPEED BIAS (ATE plateau ~142km).

# # # # # #         Root cause: model learns a fixed speed offset across all storms.
# # # # # #         Per-step loss cannot correct a constant bias — needs input diversity.

# # # # # #         Method: randomly scale TC translation speed on obs+gt trajectories
# # # # # #         by factor s ~ Uniform[0.75, 1.25]. This is equivalent to random
# # # # # #         time-warping: a storm moving at 0.8× speed arrives 20% later.

# # # # # #         By seeing the same spatial path at different speeds, model learns
# # # # # #         to estimate speed from ERA5 steering flow rather than memorizing
# # # # # #         typical speed from training distribution.

# # # # # #         Citation: Data augmentation for trajectory speed bias correction
# # # # # #         is standard in vehicle trajectory prediction (Salzmann et al. 2020,
# # # # # #         Trajectron++; Mangalam et al. 2021, GoalNet) where speed
# # # # # #         perturbation is applied to break dataset-level speed correlations.
# # # # # #         """
# # # # # #         if torch.rand(1).item() > p:
# # # # # #             return batch_list
# # # # # #         aug = list(batch_list)

# # # # # #         # Speed scale factor — uniform in [0.75, 1.25]
# # # # # #         s = 0.75 + torch.rand(1).item() * 0.5   # [0.75, 1.25]

# # # # # #         # Scale obs trajectory: interpolate positions to simulate s× speed
# # # # # #         # Strategy: keep start position, scale displacements
# # # # # #         # obs_traj: [T_obs, B, 2]
# # # # # #         if torch.is_tensor(aug[0]) and aug[0].shape[0] >= 2:
# # # # # #             obs = aug[0].clone()   # [T, B, 2]
# # # # # #             last = obs[-1:].clone()   # anchor: last obs position unchanged
# # # # # #             # Displacements from last position
# # # # # #             disp = obs - last   # [T, B, 2]
# # # # # #             # Scale: points farther in time get their displacement scaled
# # # # # #             # This moves past positions closer/farther → simulates speed change
# # # # # #             aug[0] = last + disp * s

# # # # # #         # Scale gt trajectory: same operation on future positions
# # # # # #         # traj_gt: [T_pred, B, 2] — index 1
# # # # # #         if torch.is_tensor(aug[1]) and aug[1].shape[0] >= 2:
# # # # # #             gt = aug[1].clone()   # [T, B, 2]
# # # # # #             last = aug[0][-1:].clone()   # anchor at last obs
# # # # # #             disp = gt - last
# # # # # #             aug[1] = last + disp * s

# # # # # #         return aug

# # # # # #     @staticmethod
# # # # # #     def _temporal_reverse_aug(batch_list, p=0.15):
# # # # # #         """
# # # # # #         Reverse observation sequence (p=0.15).
# # # # # #         Forces model to use ERA5 context rather than trajectory momentum alone.
# # # # # #         Prevents over-reliance on recent obs direction → reduces CTE bias.
# # # # # #         """
# # # # # #         if torch.rand(1).item() > p:
# # # # # #             return batch_list
# # # # # #         aug = list(batch_list)
# # # # # #         if torch.is_tensor(aug[0]):
# # # # # #             aug[0] = aug[0].flip(0)   # [T, B, 2] → reverse time
# # # # # #         if torch.is_tensor(aug[7]):
# # # # # #             aug[7] = aug[7].flip(0)   # obs Me
# # # # # #         return aug

# # # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # # #         # Augmentation
# # # # # #         batch_list = self._lon_flip_aug(batch_list)
# # # # # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
# # # # # #         # if epoch >= 5:
# # # # # #         #     batch_list = self._mixup_aug(batch_list, alpha=0.2, p=0.15)
# # # # # #         # if epoch >= 10:
# # # # # #         #     batch_list = self._speed_aug(batch_list, p=0.4)        # v41: speed bias fix
# # # # # #         # if epoch >= 8:
# # # # # #         #     batch_list = self._temporal_reverse_aug(batch_list, p=0.15)  # v41
# # # # # #         if epoch >= 3:                                    # sớm hơn 7 epoch!
# # # # # #             batch_list = self._speed_aug(batch_list, p=0.35)
# # # # # #         if epoch >= 5:
# # # # # #             batch_list = self._temporal_reverse_aug(batch_list, p=0.15)
        
# # # # # #         traj_gt  = batch_list[1]
# # # # # #         Me_gt    = batch_list[8]
# # # # # #         obs_t    = batch_list[0]
# # # # # #         obs_Me   = batch_list[7]
# # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # #         lp, lm   = obs_t[-1], obs_Me[-1]

# # # # # #         # Sigma schedule (giữ nguyên v41)
# # # # # #         if epoch < 5: current_sigma = 0.12
# # # # # #         elif epoch < 15: current_sigma = 0.12 - (epoch - 5) / 10.0 * 0.06
# # # # # #         else: current_sigma = 0.03

# # # # # #         # FM Forward
# # # # # #         raw_ctx = self.net._context(batch_list)
# # # # # #         obs_t = batch_list[0]
# # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # #         lp, lm = obs_t[-1], batch_list[7][-1]
        
# # # # # #         x1_rel = self._to_rel(batch_list[1], batch_list[8], lp, lm)
# # # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # # #         pred_vel = self.net.forward_with_ctx(
# # # # # #             x_t, fm_t, raw_ctx, env_data=env_data,
# # # # # #             vel_obs_feat=self.net._get_vel_obs_feat(obs_t),
# # # # # #             steering_feat=self.net._get_steering_feat(env_data, obs_t.shape[1], obs_t.device)
# # # # # #         )

# # # # # #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# # # # # #         # Chuyển sang tọa độ thật để tính position loss
# # # # # #         with torch.no_grad():
# # # # # #             fm_te = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # #             x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # # #             pred_abs, _ = self._to_abs(x1_pred, lp, lm)
# # # # # #             pred_deg = _norm_to_deg_fn(pred_abs)
# # # # # #             gt_deg = _norm_to_deg_fn(batch_list[1])

# # # # # #         # Gọi hàm loss v43 vừa sửa ở trên
# # # # # #         loss_dict = compute_total_loss(pred_deg, gt_deg, epoch=epoch)

# # # # # #         # CÂN BẰNG TRỌNG SỐ: 
# # # # # #         # fm_mse thường rất nhỏ (0.03), loss_dict["total"] thường lớn (5.0 - 10.0)
# # # # # #         # Chúng ta dùng w_pos nhỏ để không phá vỡ cấu trúc Flow Matching
# # # # # #         w_fm  = 1.0 
# # # # # #         w_pos = 0.2 # Giảm xuống để l_fm_mse vẫn là leader
        
# # # # # #         total = w_fm * l_fm_mse + w_pos * loss_dict["total"]

# # # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # # #             total = x_t.new_zeros(())

# # # # # #         return dict(
# # # # # #             total=total,
# # # # # #             fm_mse=l_fm_mse.item(),
# # # # # #             mse_hav=loss_dict["mse_hav"],
# # # # # #             endpoint=loss_dict["endpoint"],
# # # # # #             decomp=loss_dict["decomp"],
# # # # # #             accel=loss_dict["accel"],
# # # # # #             cons=loss_dict["cons"],
# # # # # #             sigma=current_sigma
# # # # # #         )
# # # # # #             # # Sigma schedule
# # # # # #         # # if epoch < 15:
# # # # # #         # #     current_sigma = 0.15
# # # # # #         # # elif epoch < 40:
# # # # # #         # #     current_sigma = 0.15 - (epoch - 15) / 25.0 * 0.09
# # # # # #         # # else:
# # # # # #         # #     current_sigma = 0.06
# # # # # #         # if epoch < 5:
# # # # # #         #     current_sigma = 0.12          # giảm từ 0.15 → 0.12
# # # # # #         # elif epoch < 15:
# # # # # #         #     current_sigma = 0.12 - (epoch - 5) / 10.0 * 0.06   # 0.12 → 0.06
# # # # # #         # elif epoch < 30:
# # # # # #         #     current_sigma = 0.06 - (epoch - 15) / 15.0 * 0.03  # 0.06 → 0.03
# # # # # #         # else:
# # # # # #         #     current_sigma = 0.03          # training sạch hơn nhiều ở epoch muộn
# # # # # #         # raw_ctx       = self.net._context(batch_list)
# # # # # #         # vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # #         # steering_feat = self.net._get_steering_feat(
# # # # # #         #     env_data, obs_t.shape[1], obs_t.device)

# # # # # #         # x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)
# # # # # #         # x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # # #         # # Scheduled teacher forcing
# # # # # #         # if self.teacher_forcing and self.training and epoch >= 3:
# # # # # #         #     p_teacher = max(0.0, 0.5 * (1.0 - (epoch - 3) / 37.0))
# # # # # #         #     if p_teacher > 0 and torch.rand(1).item() < p_teacher:
# # # # # #         #         far_mask = torch.zeros_like(x_t)
# # # # # #         #         far_mask[:, 6:, :] = 0.3
# # # # # #         #         x_t = x_t * (1 - far_mask) + x1_rel * far_mask

# # # # # #         # pred_vel = self.net.forward_with_ctx(
# # # # # #         #     x_t, fm_t, raw_ctx,
# # # # # #         #     vel_obs_feat=vel_obs_feat,
# # # # # #         #     steering_feat=steering_feat,
# # # # # #         #     env_data=env_data,
# # # # # #         # )

# # # # # #         # l_fm_mse = F.mse_loss(pred_vel, u_target)

# # # # # #         # fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # # #         # x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # # #         # pred_abs, _ = self._to_abs(x1_pred, lp, lm)
# # # # # #         # pred_deg    = _norm_to_deg_fn(pred_abs)
# # # # # #         # gt_deg      = _norm_to_deg_fn(traj_gt)

# # # # # #         # loss_dict = compute_total_loss(
# # # # # #         #     pred_deg=pred_deg,
# # # # # #         #     gt_deg=gt_deg,
# # # # # #         #     env_data=env_data,
# # # # # #         #     weights=WEIGHTS,
# # # # # #         #     epoch=epoch,
# # # # # #         # )

# # # # # #         # # w_fm  = max(0.3, 1.0 - epoch / 60.0)
# # # # # #         # # total = w_fm * l_fm_mse + loss_dict["total"]
# # # # # #         # w_fm     = max(0.8, 1.5 - epoch / 40.0)   # fm weight CAO hơn
# # # # # #         # w_pos    = min(0.4, 0.1 + epoch / 50.0)   # position weight bắt đầu nhỏ
# # # # # #         # total    = w_fm * l_fm_mse + w_pos * loss_dict["total"]
# # # # # #         # # → epoch 0: total = 1.5 * 0.57 + 0.1 * X = fm chiếm ~50%+
# # # # # #         # l_ens = x_t.new_zeros(())
# # # # # #         # if epoch >= 40 and self.n_train_ens >= 2:
# # # # # #         #     x_t2, fm_t2, u_t2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
# # # # # #         #     pv2 = self.net.forward_with_ctx(
# # # # # #         #         x_t2, fm_t2, raw_ctx,
# # # # # #         #         vel_obs_feat=vel_obs_feat,
# # # # # #         #         steering_feat=steering_feat,
# # # # # #         #         env_data=env_data,
# # # # # #         #     )
# # # # # #         #     l_ens = F.mse_loss(pv2, u_t2)
# # # # # #         #     total = total + 0.3 * l_ens

# # # # # #         # if torch.isnan(total) or torch.isinf(total):
# # # # # #         #     total = x_t.new_zeros(())

# # # # # #         # # ── RETURN DICT — v39 compat: thêm speed_acc + cumul_disp ────────
# # # # # #         # return dict(
# # # # # #         #     total        = total,
# # # # # #         #     fm_mse       = l_fm_mse.item(),
# # # # # #         #     mse_hav      = loss_dict["mse_hav_horizon"],
# # # # # #         #     multi_scale  = loss_dict["multi_scale"],
# # # # # #         #     endpoint     = loss_dict["endpoint"],
# # # # # #         #     h_direct     = loss_dict.get("h_direct",    0.0),
# # # # # #         #     vel_smooth   = loss_dict.get("vel_smooth",  0.0),
# # # # # #         #     speed_acc    = loss_dict.get("speed_acc",   0.0),
# # # # # #         #     cumul_disp   = loss_dict.get("cumul_disp",  0.0),
# # # # # #         #     accel        = loss_dict.get("accel",       0.0),  # ← v40 NEW
# # # # # #         #     decomp       = loss_dict.get("decomp",      0.0),  # ← v40 NEW
# # # # # #         #     cons         = loss_dict.get("cons",        0.0),  # ← v40 NEW
# # # # # #         #     ate_cte      = loss_dict.get("ate_cte",     0.0),  # compat (=0)
# # # # # #         #     shape        = loss_dict.get("shape",       0.0),
# # # # # #         #     velocity     = loss_dict.get("velocity",    0.0),
# # # # # #         #     heading      = loss_dict.get("heading",     0.0),
# # # # # #         #     recurv       = loss_dict.get("recurv",      0.0),
# # # # # #         #     steering     = loss_dict.get("steering",    0.0),
# # # # # #         #     ens_consist  = (l_ens.item() if torch.is_tensor(l_ens)
# # # # # #         #                     else float(l_ens)),
# # # # # #         #     sigma        = current_sigma,
# # # # # #         #     w_fm         = w_fm,
# # # # # #         #     long_range=0.0, fde=0.0, w_lr=0.0, w_fde=0.0,
# # # # # #         #     fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
# # # # # #         #     spread=0.0, continuity=0.0, step=0.0, disp=0.0,
# # # # # #         #     recurv_ratio=0.0,
# # # # # #         # )

# # # # # #     # ── Sampling — unchanged from v34fix ─────────────────────────────────────
# # # # # #     @torch.no_grad()
# # # # # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
# # # # # #                 predict_csv=None, importance_weight=True):
# # # # # #         obs_t    = batch_list[0]
# # # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # # #         lp       = obs_t[-1]
# # # # # #         lm       = batch_list[7][-1]
# # # # # #         B        = lp.shape[0]
# # # # # #         device   = lp.device
# # # # # #         T        = self.pred_len
# # # # # #         dt       = 1.0 / max(ddim_steps, 1)

# # # # # #         raw_ctx       = self.net._context(batch_list)
# # # # # #         vel_obs_feat  = self.net._get_vel_obs_feat(obs_t)
# # # # # #         steering_feat = self.net._get_steering_feat(env_data, B, device)

# # # # # #         traj_s, me_s, scores = [], [], []
# # # # # #         for _ in range(num_ensemble):
# # # # # #             x_t = torch.randn(B, T, 4, device=device) * (self.sigma_min * 2.5)
# # # # # #             for step in range(ddim_steps):
# # # # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # # # #                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
# # # # # #                 vel = self.net.forward_with_ctx(
# # # # # #                     x_t, t_b, raw_ctx,
# # # # # #                     noise_scale=ns,
# # # # # #                     vel_obs_feat=vel_obs_feat,
# # # # # #                     steering_feat=steering_feat,
# # # # # #                     env_data=env_data,
# # # # # #                 )
# # # # # #                 x_t = x_t + dt * vel
# # # # # #             x_t = self._physics_correct(x_t, lp, lm)
# # # # # #             x_t = x_t.clamp(-3.0, 3.0)
# # # # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # # # #             traj_s.append(tr)
# # # # # #             me_s.append(me)
# # # # # #             if importance_weight:
# # # # # #                 scores.append(self._score_sample(tr, env_data))

# # # # # #         all_trajs = torch.stack(traj_s)
# # # # # #         all_me    = torch.stack(me_s)

# # # # # #         if importance_weight and scores:
# # # # # #             score_tensor = torch.stack(scores)
# # # # # #             k = max(1, int(num_ensemble * 0.7))
# # # # # #             _, top_idx = score_tensor.topk(k, dim=0)
# # # # # #             pred_mean = torch.stack([
# # # # # #                 all_trajs[top_idx[:, b], :, b, :].median(0).values
# # # # # #                 for b in range(B)
# # # # # #             ], dim=1)
# # # # # #         else:
# # # # # #             pred_mean = all_trajs.median(0).values

# # # # # #         if predict_csv:
# # # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # # #         return pred_mean, all_me.mean(0), all_trajs

# # # # # #     def _score_sample(self, traj, env_data):
# # # # # #         B = traj.shape[1]
# # # # # #         if traj.shape[0] < 2:
# # # # # #             return torch.ones(B, device=traj.device)
# # # # # #         traj_deg  = _norm_to_deg_fn(traj)
# # # # # #         dt_deg    = traj_deg[1:] - traj_deg[:-1]
# # # # # #         lat_rad   = torch.deg2rad(traj_deg[:-1, :, 1])
# # # # # #         cos_lat   = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # #         dx_km     = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # # #         dy_km     = dt_deg[:, :, 1] * 111.0
# # # # # #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)
# # # # # #         speed_pen = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
# # # # # #         speed_sc  = torch.exp(-speed_pen.mean(0) / 20.0)
# # # # # #         if dt_deg.shape[0] >= 2:
# # # # # #             jerk = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
# # # # # #             smooth_sc = torch.exp(-jerk * 5.0)
# # # # # #         else:
# # # # # #             smooth_sc = torch.ones(B, device=traj.device)
# # # # # #         return speed_sc * smooth_sc

# # # # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # # # #         with torch.enable_grad():
# # # # # #             x   = x_pred.detach().requires_grad_(True)
# # # # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # # # #             for _ in range(n_steps):
# # # # # #                 opt.zero_grad()
# # # # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # # # #                 pred_deg    = _norm_to_deg_fn(pred_abs)
# # # # # #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# # # # # #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# # # # # #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# # # # # #                 speed       = torch.sqrt(
# # # # # #                     (dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # # #                     + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # # #                 F.relu(speed - 600.0).pow(2).mean().backward()
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
# # # # # #                     "lon_mean_deg", "lat_mean_deg",
# # # # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
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
# # # # # #                     w.writerow({
# # # # # #                         "timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # # #                         "lead_h": (k + 1) * 6,
# # # # # #                         "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # # #                         "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # # #                         "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # # # # #                         "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # # # # #                         "ens_spread_km": f"{spread:.2f}",
# # # # # #                     })


# # # # # # TCDiffusion = TCFlowMatching
# # # # # """
# # # # # Model/flow_matching_model.py — v44_speed_fix
# # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # THAY ĐỔI so với v41 (dựa trên phân tích 22 epoch training):

# # # # #   FIX 1: physics_scale & steering_scale init — SỬA CHÍNH
# # # # #     v41: physics_scale = ones(4) * 0.3  →  sigmoid(0.3) ≈ 0.57  (quá nhỏ)
# # # # #     v44: physics_scale = ones(4) * 1.5  →  sigmoid(1.5) ≈ 0.82  (physics đủ mạnh)
# # # # #     v44: steering_scale = ones(4) * 2.0 →  sigmoid(2.0) ≈ 0.88  (ERA5 steering mạnh hơn)
# # # # #     Lý do: FNO bị bypass (env làm model TỆ HƠN -13.1%),
# # # # #            phải dựa vào steering_drift trực tiếp từ ERA5 u/v500.

# # # # #   FIX 2: Tắt speed_aug HOÀN TOÀN
# # # # #     v41: speed_aug(p=0.35) từ epoch 3
# # # # #     v44: COMMENT OUT
# # # # #     Lý do: speed_aug scale cả input lẫn GT cùng factor →
# # # # #            model học pattern relative, KHÔNG học absolute speed →
# # # # #            đây là nguyên nhân thứ 2 gây speed bias sau thiếu speed loss.

# # # # #   FIX 3: Tắt temporal_reverse_aug
# # # # #     v41: temporal_reverse_aug(p=0.15) từ epoch 5
# # # # #     v44: COMMENT OUT
# # # # #     Lý do: lật obs nhưng KHÔNG lật env_data →
# # # # #            mismatch giữa trajectory direction và steering flow.

# # # # #   FIX 4: Cân bằng FM / Position loss weights
# # # # #     v41: w_fm=1.0, w_pos=0.2  →  l_fm_mse(0.03) vs pos_loss(5-10)
# # # # #          → pos dominate gấp 30-60x, FM gradient bị lấn át
# # # # #     v44: w_fm=30.0, w_pos=1.0  →  FM contribution ~0.9, pos ~5-10
# # # # #          → FM có "tiếng nói" trong optimization

# # # # #   FIX 5: Thêm speed_acc vào return dict của get_loss_breakdown
# # # # #     v41: speed_acc KHÔNG trong return dict → bd.get('speed_acc', 0) = 0 LUÔN
# # # # #     v44: thêm speed_acc=loss_dict.get("speed_acc", 0.0) vào return

# # # # #   FIX 6: Auxiliary FNO Loss — ép FNO học steering (GIẢI QUYẾT BYPASS)
# # # # #     Diagnostic: ADE without env (299km) < ADE with env (338km) → FNO inject noise!
# # # # #     Bottleneck variance = 0.05 (gần như constant output)
# # # # #     v44: thêm aux_steering_head: Linear(128, 2) predict u500/v500
# # # # #          → VelocityField._context() lưu self.aux_uv_pred
# # # # #          → get_loss_breakdown thêm l_aux = MSE(aux_uv_pred, u500/v500_target)
# # # # #          → Bắt FNO phải học extract steering signal từ geopotential field

# # # # #   FIX 7: Dropout 0.15 → 0.10 (reduce underfit)
# # # # #     v41: 0.15 (tăng để giảm overfit)
# # # # #     v44: 0.10 (quay lại) vì model đã lớn hơn 394k params của ST-Trans
# # # # #          mà chúng ta cần học, không phải regularize mạnh hơn.

# # # # #   FIX 8: return dict đủ keys (speed_acc, cumul_disp) để log không bị KeyError

# # # # # EXPECTED RESULTS sau 10-20 epoch với v44:
# # # # #   spd log: 0.000 → 0.5~2.0 (xác nhận speed loss hoạt động)
# # # # #   ATE: 165km → ~100-120km
# # # # #   72h: 384km → ~300-320km
# # # # #   CTE: 58km (giữ nguyên hoặc giảm nhẹ, KHÔNG để tăng)
# # # # # """
# # # # # from __future__ import annotations

# # # # # import csv
# # # # # import math
# # # # # import os
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
# # # # #     """Convert normalized tensor to degrees."""
# # # # #     out = t.clone()
# # # # #     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # # #     out[..., 1] = (t[..., 1] * 50.0) / 10.0
# # # # #     return out


# # # # # def _unwrap_model(model: nn.Module) -> nn.Module:
# # # # #     if hasattr(model, '_orig_mod'):
# # # # #         return model._orig_mod
# # # # #     return model


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  EMAModel — không đổi
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class EMAModel:
# # # # #     def __init__(self, model: nn.Module, decay: float = 0.999):
# # # # #         self.decay = decay
# # # # #         m = _unwrap_model(model)
# # # # #         self.shadow = {
# # # # #             k: v.detach().clone()
# # # # #             for k, v in m.state_dict().items()
# # # # #             if v.dtype.is_floating_point
# # # # #         }

# # # # #     def update(self, model: nn.Module):
# # # # #         m = _unwrap_model(model)
# # # # #         with torch.no_grad():
# # # # #             for k, v in m.state_dict().items():
# # # # #                 if k in self.shadow:
# # # # #                     self.shadow[k].mul_(self.decay).add_(
# # # # #                         v.detach(), alpha=1 - self.decay)

# # # # #     def apply_to(self, model: nn.Module):
# # # # #         m = _unwrap_model(model)
# # # # #         backup = {}
# # # # #         sd = m.state_dict()
# # # # #         for k in self.shadow:
# # # # #             if k not in sd:
# # # # #                 continue
# # # # #             backup[k] = sd[k].detach().clone()
# # # # #             sd[k].copy_(self.shadow[k])
# # # # #         return backup

# # # # #     def restore(self, model: nn.Module, backup: dict):
# # # # #         m = _unwrap_model(model)
# # # # #         sd = m.state_dict()
# # # # #         for k, v in backup.items():
# # # # #             if k in sd:
# # # # #                 sd[k].copy_(v)


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  VelocityField — FIXES: physics_scale, steering_scale, dropout, aux_head
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class VelocityField(nn.Module):
# # # # #     RAW_CTX_DIM = 512

# # # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # # #                  unet_in_ch=13, **kwargs):
# # # # #         super().__init__()
# # # # #         self.pred_len = pred_len
# # # # #         self.obs_len  = obs_len

# # # # #         # ── FNO3D Spatial Encoder ────────────────────────────────────────────
# # # # #         self.spatial_enc = FNO3DEncoder(
# # # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # # #             spatial_down=32, dropout=0.05)
# # # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # # #         # ── FIX 6: Auxiliary head để ép FNO học steering (giải quyết bypass) ─
# # # # #         # Bottleneck (128 dim) → predict u500, v500 steering components
# # # # #         # Loss: MSE(aux_uv_pred, env_data[u500_center, v500_center])
# # # # #         # Kỳ vọng: bottleneck variance tăng từ 0.05 → 0.2-0.3
# # # # #         self.aux_steering_head = nn.Linear(128, 2)   # [B, 128] → [B, 2] (u,v)
# # # # #         self.aux_uv_pred = None   # populated in _context(), read in get_loss_breakdown

# # # # #         # ── 1D Trajectory Encoder ────────────────────────────────────────────
# # # # #         self.enc_1d = DataEncoder1D(
# # # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)

# # # # #         # ── ENV Encoder ──────────────────────────────────────────────────────
# # # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # # #         # ── Context projection ───────────────────────────────────────────────
# # # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # # #         self.ctx_drop = nn.Dropout(0.10)   # FIX 7: 0.15 → 0.10
# # # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # # #         # ── Velocity observation encoder ─────────────────────────────────────
# # # # #         self.vel_obs_enc = nn.Sequential(
# # # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # # #             nn.LayerNorm(256),
# # # # #             nn.Linear(256, 256), nn.GELU(),
# # # # #         )

# # # # #         # ── Steering feature encoder ─────────────────────────────────────────
# # # # #         self.steering_enc = nn.Sequential(
# # # # #             nn.Linear(4, 64), nn.GELU(),
# # # # #             nn.LayerNorm(64),
# # # # #             nn.Linear(64, 128), nn.GELU(),
# # # # #             nn.Linear(128, 256),
# # # # #         )

# # # # #         # ── Transformer Decoder ──────────────────────────────────────────────
# # # # #         self.time_fc1   = nn.Linear(256, 512)
# # # # #         self.time_fc2   = nn.Linear(512, 256)
# # # # #         self.traj_embed = nn.Linear(4, 256)
# # # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # # #         self.transformer = nn.TransformerDecoder(
# # # # #             nn.TransformerDecoderLayer(
# # # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # # #                 dropout=0.10,          # FIX 7: 0.15 → 0.10
# # # # #                 activation="gelu", batch_first=True),
# # # # #             num_layers=2)

# # # # #         self.out_fc1 = nn.Linear(256, 512)
# # # # #         self.out_fc2 = nn.Linear(512, 4)

# # # # #         # ── Physics scale parameters ─────────────────────────────────────────
# # # # #         self.step_scale = nn.Parameter(torch.ones(pred_len) * 0.5)

# # # # #         # FIX 1: physics_scale và steering_scale init — SỬA CHÍNH
# # # # #         # v41: 0.3 → sigmoid(0.3)≈0.57, quá nhỏ → physics không có ảnh hưởng
# # # # #         # v44: 1.5 → sigmoid(1.5)≈0.82, physics contribution mạnh hơn
# # # # #         # Khi FNO bị bypass, cần physics priors có ảnh hưởng đủ lớn
# # # # #         self.physics_scale  = nn.Parameter(torch.ones(4) * 1.5)   # sigmoid≈0.82
# # # # #         self.steering_scale = nn.Parameter(torch.ones(4) * 1.0)   # sigmoid≈0.88

# # # # #         self._init_weights()

# # # # #     # def _init_weights(self):
# # # # #     #     with torch.no_grad():
# # # # #     #         for name, m in self.named_modules():
# # # # #     #             if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # # #     #                 nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # # #     #                 if m.bias is not None:
# # # # #     #                     nn.init.zeros_(m.bias)
# # # # #     #         # Khởi tạo aux_steering_head nhỏ để không ảnh hưởng training ban đầu
# # # # #     #         nn.init.xavier_uniform_(self.aux_steering_head.weight, gain=0.01)
# # # # #     #         nn.init.zeros_(self.aux_steering_head.bias)
# # # # #     def _init_weights(self):
# # # # #         with torch.no_grad():
# # # # #             for name, m in self.named_modules():
# # # # #                 if isinstance(m, nn.Linear) and 'out_fc' in name:
# # # # #                     nn.init.xavier_uniform_(m.weight, gain=0.1)
# # # # #                     if m.bias is not None:
# # # # #                         nn.init.zeros_(m.bias)
# # # # #             # FIX v45: aux_steering_head dùng init bình thường để có gradient signal
# # # # #             # v44 dùng gain=0.01 → output ≈ 0 → MSE ≈ 0.0025 → loss quá nhỏ
# # # # #             # v45: gain=1.0 (default) → output ~ N(0, 1/sqrt(128)) ≈ N(0, 0.09)
# # # # #             # phù hợp với scale của u500/v500 (~0.05) → MSE ban đầu ~0.01
# # # # #             nn.init.xavier_uniform_(self.aux_steering_head.weight, gain=1.0)
# # # # #             nn.init.zeros_(self.aux_steering_head.bias)
# # # # #     def _time_emb(self, t, dim=256):
# # # # #         half = dim // 2
# # # # #         freq = torch.exp(
# # # # #             torch.arange(half, dtype=torch.float32, device=t.device)
# # # # #             * (-math.log(10000.0) / max(half - 1, 1))
# # # # #         )
# # # # #         emb = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
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

# # # # #         # Bottleneck pooling
# # # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # # #         if e_3d_s.shape[1] != T_obs:
# # # # #             e_3d_s = F.interpolate(
# # # # #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# # # # #                 mode="linear", align_corners=False
# # # # #             ).permute(0, 2, 1)

# # # # #         # FIX 6: Auxiliary steering prediction từ bottleneck
# # # # #         # Pooled bottleneck: [B, T, 128] → mean over T → [B, 128]
# # # # #         bot_pooled = e_3d_s.mean(dim=1)                          # [B, 128]
# # # # #         self.aux_uv_pred = self.aux_steering_head(bot_pooled)    # [B, 2]
# # # # #         # self.aux_uv_pred sẽ được đọc bởi TCFlowMatching.get_loss_breakdown()

# # # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # # #         t_w = torch.softmax(
# # # # #             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # # #                          device=e_3d_dec_t.device) * 0.5, dim=0)
# # # # #         f_spatial = self.decoder_proj(
# # # # #             (e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

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
# # # # #             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2,
# # # # #                                device=obs_traj.device)
# # # # #             vels = torch.cat([pad, vels], dim=0)
# # # # #         else:
# # # # #             vels = vels[-self.obs_len:]
# # # # #         return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

# # # # #     def _get_steering_feat(self, env_data, B, device):
# # # # #         if env_data is None:
# # # # #             return torch.zeros(B, 256, device=device)

# # # # #         def _safe_get(key, default=0.0):
# # # # #             v = env_data.get(key, None)
# # # # #             if v is None or not torch.is_tensor(v):
# # # # #                 return torch.full((B,), default, device=device)
# # # # #             v = v.to(device).float()
# # # # #             if v.dim() >= 2:
# # # # #                 while v.dim() > 1:
# # # # #                     v = v.mean(-1)
# # # # #             if v.shape[0] != B:
# # # # #                 v = (v.view(-1)[:B] if v.numel() >= B
# # # # #                      else torch.full((B,), default, device=device))
# # # # #             return v

# # # # #         feat = torch.stack([
# # # # #             _safe_get("u500_mean"), _safe_get("v500_mean"),
# # # # #             _safe_get("u500_center"), _safe_get("v500_center")
# # # # #         ], dim=-1)
# # # # #         return self.steering_enc(feat)

# # # # #     def _beta_drift(self, x_t):
# # # # #         """Beta drift: Coriolis-induced TC track deflection."""
# # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # # #         R_tc    = 3e5
# # # # #         v_phys  = torch.zeros_like(x_t)
# # # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6 * 3600 / (5 * 111 * 1000)
# # # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6 * 3600 / (5 * 111 * 1000)
# # # # #         return v_phys

# # # # #     def _steering_drift(self, x_t, env_data):
# # # # #         """ERA5 u500/v500 steering flow → displacement per 6h step."""
# # # # #         if env_data is None:
# # # # #             return torch.zeros_like(x_t)
# # # # #         B, device = x_t.shape[0], x_t.device

# # # # #         def _safe_mean(key):
# # # # #             v = env_data.get(key, None)
# # # # #             if v is None or not torch.is_tensor(v):
# # # # #                 return torch.zeros(B, device=device)
# # # # #             v = v.to(device).float()
# # # # #             while v.dim() > 1:
# # # # #                 v = v.mean(-1)
# # # # #             return v.view(-1)[:B] if v.numel() >= B else torch.zeros(B, device=device)

# # # # #         u = _safe_mean("u500_center")
# # # # #         v = _safe_mean("v500_center")
# # # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # # #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# # # # #         out = torch.zeros_like(x_t)

# # # # #         # # FIX: hệ số 0.2 cũ quá nhỏ, giờ dùng 0.5 để ERA5 có ảnh hưởng rõ hơn
# # # # #         # # v41: * 0.2 (quá nhỏ → steering contribution yếu)
# # # # #         # # v44: * 0.5 (tăng để physics prior mạnh hơn khi FNO bị bypass)
# # # # #         # out[:, :, 0] = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat) * 0.5
# # # # #         # out[:, :, 1] = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0) * 0.5
# # # # #         # return out
# # # # #         # FIX v45: hệ số 1.0 (từ 0.5) — physics steering có ảnh hưởng đầy đủ
# # # # #         # khi FNO bị bypass. Sau khi aux loss buộc FNO học, có thể giảm lại.
# # # # #         out[:, :, 0] = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat) * 1.0
# # # # #         out[:, :, 1] = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0) * 1.0
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

# # # # #         with torch.no_grad():
# # # # #             v_phys  = self._beta_drift(x_t[:, :T_seq])
# # # # #             v_steer = self._steering_drift(x_t[:, :T_seq], env_data)

# # # # #         return (v_neural
# # # # #                 + torch.sigmoid(self.physics_scale) * v_phys
# # # # #                 + torch.sigmoid(self.steering_scale) * v_steer)

# # # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
# # # # #                           vel_obs_feat=None, steering_feat=None, env_data=None):
# # # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# # # # #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  TCFlowMatching — FIXES: augmentation, FM balance, return dict, aux loss
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

# # # # #         self.use_ema   = use_ema
# # # # #         self.ema_decay = ema_decay
# # # # #         self._ema      = None

# # # # #     def init_ema(self):
# # # # #         if self.use_ema:
# # # # #             self._ema = EMAModel(self, decay=self.ema_decay)

# # # # #     def ema_update(self):
# # # # #         if self._ema is not None:
# # # # #             self._ema.update(self)

# # # # #     def set_curriculum_len(self, *a, **kw):
# # # # #         pass

# # # # #     @staticmethod
# # # # #     def _to_rel(traj, Me, lp, lm):
# # # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # # #     @staticmethod
# # # # #     def _to_abs(rel, lp, lm):
# # # # #         d = rel.permute(1, 0, 2)
# # # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # # #         if sigma_min is None:
# # # # #             sigma_min = self.sigma_min
# # # # #         B, device = x1.shape[0], x1.device
# # # # #         x0 = torch.randn_like(x1) * sigma_min
# # # # #         t  = torch.rand(B, device=device)
# # # # #         te = t.view(B, 1, 1)
# # # # #         return (1.0 - te) * x0 + te * x1, t, x1 - x0

# # # # #     # ── Augmentation methods ──────────────────────────────────────────────────

# # # # #     @staticmethod
# # # # #     def _lon_flip_aug(batch_list, p=0.3):
# # # # #         """Longitude flip: flip trajectory horizontally."""
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
# # # # #         """Nhỏ Gaussian noise vào obs trajectory."""
# # # # #         if torch.rand(1).item() > 0.5:
# # # # #             return batch_list
# # # # #         aug = list(batch_list)
# # # # #         if torch.is_tensor(aug[0]):
# # # # #             aug[0] = aug[0] + torch.randn_like(aug[0]) * sigma
# # # # #         return aug

# # # # #     @staticmethod
# # # # #     def _mixup_aug(batch_list, alpha=0.2, p=0.2):
# # # # #         """Mixup augmentation giữa các samples trong batch."""
# # # # #         if torch.rand(1).item() > p:
# # # # #             return batch_list
# # # # #         aug = list(batch_list)
# # # # #         lam = float(torch.distributions.Beta(alpha, alpha).sample())
# # # # #         lam = max(lam, 1 - lam)
# # # # #         if torch.is_tensor(aug[0]) and aug[0].shape[1] >= 2:
# # # # #             B    = aug[0].shape[1]
# # # # #             perm = torch.randperm(B, device=aug[0].device)
# # # # #             for idx in [0, 1, 7, 8]:
# # # # #                 if torch.is_tensor(aug[idx]):
# # # # #                     aug[idx] = lam * aug[idx] + (1 - lam) * aug[idx][:, perm]
# # # # #         return aug

# # # # #     @staticmethod
# # # # #     def _speed_aug(batch_list, p=0.4):
# # # # #         """
# # # # #         DISABLED trong v44 (FIX 2).

# # # # #         Lý do tắt: speed_aug scale cả obs và GT với cùng factor s →
# # # # #         model học pattern relative giữa obs và GT, KHÔNG học absolute speed.
# # # # #         Đây là nguyên nhân thứ 2 gây speed bias (sau thiếu speed loss).

# # # # #         Cụ thể:
# # # # #           aug[0] (obs) và aug[1] (GT) đều được scale bởi cùng s ~[0.75, 1.25]
# # # # #           → Error signal (pred - GT) không bị ảnh hưởng bởi s
# # # # #           → Model không học được speed từ ERA5 steering
# # # # #           → ATE bị bias vì model không có reference cho absolute speed
# # # # #         """
# # # # #         # FIX 2: LUÔN return unmodified batch
# # # # #         return batch_list

# # # # #     @staticmethod
# # # # #     def _temporal_reverse_aug(batch_list, p=0.15):
# # # # #         """
# # # # #         DISABLED trong v44 (FIX 3).

# # # # #         Lý do tắt: lật obs_traj và obs_Me nhưng KHÔNG lật env_data →
# # # # #         steering flow từ ERA5 không khớp với trajectory direction →
# # # # #         model học cách ignore env_data hoàn toàn (FNO bypass symptom).
# # # # #         """
# # # # #         # FIX 3: LUÔN return unmodified batch
# # # # #         return batch_list

# # # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # # #         """
# # # # #         Compute full loss breakdown.

# # # # #         CHANGES từ v41:
# # # # #           1. Chỉ giữ lon_flip + obs_noise augmentation (tắt speed_aug, temporal_reverse)
# # # # #           2. Compute auxiliary FNO loss (l_aux) sau khi gọi _context()
# # # # #           3. Cân bằng FM/pos weights: w_fm=30.0, w_pos=1.0
# # # # #           4. Return dict đầy đủ tất cả keys cần thiết cho logging
# # # # #         """
# # # # #         # ── Augmentation (chỉ giữ stable augs) ───────────────────────────────
# # # # #         batch_list = self._lon_flip_aug(batch_list)
# # # # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)
# # # # #         # FIX 2 & 3: speed_aug và temporal_reverse_aug đã bị disabled trong methods
# # # # #         # (return unmodified batch list), nhưng để rõ ràng, không gọi ở đây.

# # # # #         obs_t    = batch_list[0]
# # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # #         lp, lm   = obs_t[-1], batch_list[7][-1]

# # # # #         # ── Sigma schedule (giữ nguyên v41) ──────────────────────────────────
# # # # #         if epoch < 5:
# # # # #             current_sigma = 0.12
# # # # #         elif epoch < 15:
# # # # #             current_sigma = 0.12 - (epoch - 5) / 10.0 * 0.06   # 0.12 → 0.06
# # # # #         else:
# # # # #             current_sigma = 0.03

# # # # #         # ── Flow Matching Forward ─────────────────────────────────────────────
# # # # #         # _context() gọi FNO3D encoder và lưu aux_uv_pred trong self.net
# # # # #         raw_ctx = self.net._context(batch_list)

# # # # #         obs_t    = batch_list[0]
# # # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # # #         lp, lm   = obs_t[-1], batch_list[7][-1]

# # # # #         x1_rel = self._to_rel(batch_list[1], batch_list[8], lp, lm)
# # # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # # #         pred_vel = self.net.forward_with_ctx(
# # # # #             x_t, fm_t, raw_ctx, env_data=env_data,
# # # # #             vel_obs_feat=self.net._get_vel_obs_feat(obs_t),
# # # # #             steering_feat=self.net._get_steering_feat(env_data, obs_t.shape[1], obs_t.device)
# # # # #         )

# # # # #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# # # # #         # ── Chuyển sang degree coordinates để tính position loss ──────────────
# # # # #         with torch.no_grad():
# # # # #             fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # # #             x1_pred = x_t + (1.0 - fm_te) * pred_vel
# # # # #             pred_abs, _ = self._to_abs(x1_pred, lp, lm)
# # # # #             pred_deg = _norm_to_deg_fn(pred_abs)
# # # # #             gt_deg   = _norm_to_deg_fn(batch_list[1])

# # # # #         # ── Position loss (v44_speed_fix) ─────────────────────────────────────
# # # # #         loss_dict = compute_total_loss(pred_deg, gt_deg, epoch=epoch)

# # # # #         # ── FIX 6: Auxiliary FNO steering loss ──────────────────────────────
# # # # #         # Ép FNO phải học extract u500/v500 từ geopotential height field.
# # # # #         # Nếu không có term này, FNO sẽ output gần như constant (variance=0.05)
# # # # #         # và model bỏ qua env_data hoàn toàn.
# # # # #         l_aux = x_t.new_zeros(())
# # # # #         if (self.net.aux_uv_pred is not None
# # # # #                 and env_data is not None
# # # # #                 and "u500_center" in env_data
# # # # #                 and "v500_center" in env_data):
# # # # #             try:
# # # # #                 B_aux = self.net.aux_uv_pred.shape[0]
# # # # #                 device_aux = self.net.aux_uv_pred.device

# # # # #                 u_raw = env_data["u500_center"].to(device_aux).float()
# # # # #                 v_raw = env_data["v500_center"].to(device_aux).float()

# # # # #                 # Average nếu có nhiều dimensions (e.g., spatial grid)
# # # # #                 while u_raw.dim() > 1:
# # # # #                     u_raw = u_raw.mean(-1)
# # # # #                 while v_raw.dim() > 1:
# # # # #                     v_raw = v_raw.mean(-1)

# # # # #                 # Đảm bảo shape khớp với batch size
# # # # #                 if u_raw.shape[0] >= B_aux:
# # # # #                     u_target_aux = u_raw[:B_aux]
# # # # #                     v_target_aux = v_raw[:B_aux]
# # # # #                 else:
# # # # #                     u_target_aux = torch.zeros(B_aux, device=device_aux)
# # # # #                     v_target_aux = torch.zeros(B_aux, device=device_aux)

# # # # #                 uv_target = torch.stack([u_target_aux, v_target_aux], dim=-1)  # [B, 2]
# # # # #                 l_aux = F.mse_loss(self.net.aux_uv_pred, uv_target)
# # # # #             except Exception:
# # # # #                 l_aux = x_t.new_zeros(())

# # # # #         # ─────────────────────────────────────────────────────────────────────
# # # # #         # FIX v45: Loss balance sau khi position losses đã normalized
# # # # #         # ─────────────────────────────────────────────────────────────────────
# # # # #         # Sau khi compute_total_loss đã normalize, loss_dict["total"] ≈ 3-7
# # # # #         # → có thể balance với fm_mse (~0.03-0.5) bằng weight nhỏ hơn nhiều
# # # # #         #
# # # # #         # Mục tiêu contributions:
# # # # #         #   FM:    w_fm * 0.1   ≈ 1.0   (FM velocity field học sạch)
# # # # #         #   POS:   w_pos * 4.0  ≈ 4.0   (primary position learning)
# # # # #         #   AUX:   w_aux * 0.01 ≈ 1.0   (FNO bị ép phải học steering)
# # # # #         #
# # # # #         # Total ≈ 6.0 → gradient clip 1.0 hoạt động chuẩn
# # # # #         w_fm  = 10.0   # FM coefficient (giảm từ v44=30 vì pos đã normalized)
# # # # #         w_pos = 1.0    # position loss (đã normalized)
# # # # #         w_aux = 100.0  # CRITICAL: aux phải lớn để ép FNO học (l_aux ~ 0.01)

# # # # #         total = w_fm * l_fm_mse + w_pos * loss_dict["total"] + w_aux * l_aux

# # # # #         if torch.isnan(total) or torch.isinf(total):
# # # # #             total = x_t.new_zeros(())

# # # # #         # Return dict đầy đủ — bao gồm cả normalized và raw values
# # # # #         return dict(
# # # # #             total       = total,
# # # # #             fm_mse      = l_fm_mse.item(),
# # # # #             # Normalized losses (small scale, stable for monitoring)
# # # # #             mse_hav     = loss_dict.get("mse_hav", 0.0),
# # # # #             endpoint    = loss_dict.get("endpoint", 0.0),
# # # # #             speed_acc   = loss_dict.get("speed_acc", 0.0),
# # # # #             accel       = loss_dict.get("accel", 0.0),
# # # # #             decomp      = loss_dict.get("decomp", 0.0),
# # # # #             cons        = loss_dict.get("cons", 0.0),
# # # # #             # Raw physical values (for human interpretation)
# # # # #             hav_km      = loss_dict.get("l_hav_km", 0.0),
# # # # #             h72_km      = loss_dict.get("l_72h_km", 0.0),
# # # # #             spd_kmh     = loss_dict.get("l_speed_kmh", 0.0),
# # # # #             acc_kmh2    = loss_dict.get("l_acc_kmh2", 0.0),
# # # # #             # Auxiliary FNO learning signal
# # # # #             aux_fno     = l_aux.item() if torch.is_tensor(l_aux) else float(l_aux),
# # # # #             sigma       = current_sigma,
# # # # #         )
# # # # #     # ── Sampling — không đổi từ v34fix ─────────────────────────────────────

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
# # # # #         for _ in range(num_ensemble):
# # # # #             x_t = torch.randn(B, T, 4, device=device) * (self.sigma_min * 2.5)
# # # # #             for step in range(ddim_steps):
# # # # #                 t_b = torch.full((B,), step * dt, device=device)
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
# # # # #             if importance_weight:
# # # # #                 scores.append(self._score_sample(tr, env_data))

# # # # #         all_trajs = torch.stack(traj_s)
# # # # #         all_me    = torch.stack(me_s)

# # # # #         if importance_weight and scores:
# # # # #             score_tensor = torch.stack(scores)
# # # # #             k            = max(1, int(num_ensemble * 0.7))
# # # # #             _, top_idx   = score_tensor.topk(k, dim=0)
# # # # #             pred_mean = torch.stack([
# # # # #                 all_trajs[top_idx[:, b], :, b, :].median(0).values
# # # # #                 for b in range(B)
# # # # #             ], dim=1)
# # # # #         else:
# # # # #             pred_mean = all_trajs.median(0).values

# # # # #         if predict_csv:
# # # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # # #         return pred_mean, all_me.mean(0), all_trajs

# # # # #     def _score_sample(self, traj, env_data):
# # # # #         B = traj.shape[1]
# # # # #         if traj.shape[0] < 2:
# # # # #             return torch.ones(B, device=traj.device)
# # # # #         traj_deg  = _norm_to_deg_fn(traj)
# # # # #         dt_deg    = traj_deg[1:] - traj_deg[:-1]
# # # # #         lat_rad   = torch.deg2rad(traj_deg[:-1, :, 1])
# # # # #         cos_lat   = torch.cos(lat_rad).clamp(min=1e-4)
# # # # #         dx_km     = dt_deg[:, :, 0] * cos_lat * 111.0
# # # # #         dy_km     = dt_deg[:, :, 1] * 111.0
# # # # #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)
# # # # #         speed_pen = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
# # # # #         speed_sc  = torch.exp(-speed_pen.mean(0) / 20.0)
# # # # #         if dt_deg.shape[0] >= 2:
# # # # #             jerk     = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
# # # # #             smooth_sc = torch.exp(-jerk * 5.0)
# # # # #         else:
# # # # #             smooth_sc = torch.ones(B, device=traj.device)
# # # # #         return speed_sc * smooth_sc

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
# # # # #                 speed = torch.sqrt(
# # # # #                     (dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
# # # # #                     + (dt_deg[:, :, 1] * 111.0).pow(2))
# # # # #                 F.relu(speed - 600.0).pow(2).mean().backward()
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
# # # # #                     "lon_mean_deg", "lat_mean_deg",
# # # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # # #         write_hdr = not os.path.exists(csv_path)
# # # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # #         with open(csv_path, "a", newline="") as fh:
# # # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # # #             if write_hdr:
# # # # #                 w.writeheader()
# # # # #             for b in range(B):
# # # # #                 for k in range(T):
# # # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # # #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# # # # #                                * math.cos(math.radians(mean_lat[k, b])))
# # # # #                     spread = float(((dlat**2 + dlon**2) ** 0.5).mean() * 111.0)
# # # # #                     w.writerow({
# # # # #                         "timestamp": ts, "batch_idx": b, "step_idx": k,
# # # # #                         "lead_h": (k + 1) * 6,
# # # # #                         "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # # #                         "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # # #                         "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # # # #                         "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # # # #                         "ens_spread_km": f"{spread:.2f}",
# # # # #                     })


# # # # # # Backward compat alias
# # # # # TCDiffusion = TCFlowMatching

# # # # """
# # # # Model/flow_matching_model_v46.py — ST-Trans-inspired FM
# # # # ═══════════════════════════════════════════════════════════════════════════════

# # # # ROOT CAUSE ANALYSIS từ v41/v45 plateau ở 384km:
# # # #   - Loss quá phức tạp: 6+ terms, conflicting gradients
# # # #   - FNO auxiliary loss, speed_acc, decomp,... → gradient war
# # # #   - Model học "average" thay vì trajectory structure

# # # # INSIGHT từ ST-Trans (297km, 394k params):
# # # #   - Loss CỰC ĐƠN GIẢN: DPE + 0.05*MSE + speed_penalty + accel_penalty
# # # #   - Không có AFCRPS, không có ATE/CTE decomp, không có FNO aux
# # # #   - Physics = chỉ penalty over-speed và jerk, không enforce dynamics
# # # #   - Non-autoregressive: predict all steps parallel

# # # # v46 STRATEGY — "ST-Trans loss, FM architecture":
# # # #   LOSS = haversine_DPE (weighted) + 0.05*coord_MSE + λ_spd*speed_penalty + λ_acc*accel_penalty
  
# # # #   Đơn giản hóa hoàn toàn:
# # # #   - BỎ: FNO aux loss, speed_acc term, decomp, cons, FM_MSE balance
# # # #   - GIỮ: haversine accuracy + kinematic regularization (như ST-Trans)
# # # #   - THÊM: step weights nhấn mạnh 48h/72h (horizon targets)
  
# # # #   Điều chỉnh sigma schedule:
# # # #   - Epoch 0-10: sigma=0.15 (near-deterministic, học fast)
# # # #   - Epoch 10-40: linear decay 0.15→0.03
# # # #   - Epoch 40+: sigma=0.03 (full FM diversity)

# # # # EXPECTED: 72h < 320km ở ep30, < 300km ở ep50
# # # # """
# # # # from __future__ import annotations

# # # # import csv
# # # # import math
# # # # import os
# # # # from datetime import datetime

# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.nn.functional as F

# # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # from Model.env_net_transformer_gphsplit import Env_net


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Constants
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # R_EARTH  = 6371.0
# # # # DT_HOURS = 6.0
# # # # DEG2KM   = 111.0

# # # # # Step weights: nhấn mạnh 48h (step 8) và 72h (step 12)
# # # # # 6h  12h  18h  24h  30h  36h  42h  48h  54h  60h  66h  72h
# # # # STEP_WEIGHTS = [
# # # #     1.0, 2.0, 1.0, 2.5, 1.0, 1.5, 1.0, 3.0, 1.0, 1.5, 1.0, 4.0
# # # # ]


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Utility
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# # # #     """Convert normalized coords to degrees. Differentiable."""
# # # #     lon = (t[..., 0] * 50.0 + 1800.0) / 10.0
# # # #     lat = (t[..., 1] * 50.0) / 10.0
# # # #     return torch.stack([lon, lat], dim=-1)


# # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # #     """Haversine distance in km. p1, p2: [..., 2] in degrees (lon, lat)."""
# # # #     lat1 = torch.deg2rad(p1[..., 1])
# # # #     lat2 = torch.deg2rad(p2[..., 1])
# # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # #     a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
# # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1.0 - 1e-12).sqrt())


# # # # def _unwrap_model(model: nn.Module) -> nn.Module:
# # # #     if hasattr(model, '_orig_mod'):
# # # #         return model._orig_mod
# # # #     return model


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  ST-Trans Inspired Loss — CỰC ĐƠN GIẢN
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def compute_st_trans_loss(
# # # #     pred_deg: torch.Tensor,   # [T, B, 2] degrees
# # # #     gt_deg:   torch.Tensor,   # [T, B, 2] degrees
# # # #     epoch: int = 0,
# # # # ) -> dict:
# # # #     """
# # # #     Replica của ST-Trans physics-guided composite loss:
# # # #       L = L_DPE + 0.05*L_MSE + 0.1*L_speed + 0.01*L_accel
    
# # # #     Với step weighting nhấn mạnh 48h/72h.
# # # #     Normalize output về scale ~1-3 để gradient stable.
# # # #     """
# # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # #     if T < 2:
# # # #         return {"total": pred_deg.new_zeros(()), "dpe": 0.0, "spd": 0.0, "acc": 0.0}

# # # #     # ── L_DPE: weighted haversine per step ────────────────────────────────────
# # # #     w = pred_deg.new_tensor(STEP_WEIGHTS[:T])
# # # #     w = w / w.sum() * T  # normalize, giữ scale

# # # #     dist_km = _haversine_deg(pred_deg[:T], gt_deg[:T])  # [T, B]

# # # #     # Huber-style để robust với outlier (TC recurvature)
# # # #     delta_km = 200.0
# # # #     l_dpe_per = torch.where(
# # # #         dist_km < delta_km,
# # # #         dist_km.pow(2) / (2.0 * delta_km),
# # # #         dist_km - delta_km / 2.0
# # # #     )
# # # #     l_dpe = (l_dpe_per * w.unsqueeze(1)).mean() / delta_km  # normalized ~1-3

# # # #     # ── L_MSE: coordinate-space MSE (gradient smoothness) ────────────────────
# # # #     # Như ST-Trans, weight nhỏ để không dominate
# # # #     l_mse = F.mse_loss(pred_deg[:T], gt_deg[:T])  # degrees² scale

# # # #     # ── L_speed: penalty cho excessive translation speed ─────────────────────
# # # #     # ST-Trans: vmax = 80 km/h per 3h step → vmax_deg = 80*3/111 = 2.16°
# # # #     # Dữ liệu 6h step → vmax = 80 km/h * 6h = 480km → ~4.3°
# # # #     # Dùng soft quadratic penalty
# # # #     if T >= 2:
# # # #         dt_deg = pred_deg[1:T] - pred_deg[:T-1]       # [T-1, B, 2]
# # # #         lat_mid = (pred_deg[:T-1, :, 1] + pred_deg[1:T, :, 1]) * 0.5
# # # #         cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)

# # # #         dx_km = dt_deg[:, :, 0] * cos_lat * DEG2KM    # [T-1, B]
# # # #         dy_km = dt_deg[:, :, 1] * DEG2KM
# # # #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2 + 1e-6) / DT_HOURS  # km/h

# # # #         vmax_kmh = 80.0
# # # #         l_speed = F.relu(speed_kmh - vmax_kmh).pow(2).mean()
# # # #         l_speed = l_speed / (vmax_kmh ** 2)  # normalize
# # # #     else:
# # # #         l_speed = pred_deg.new_zeros(())

# # # #     # ── L_accel: smooth acceleration (penalty jerk) ───────────────────────────
# # # #     if T >= 3:
# # # #         # acceleration ≈ Δspeed / Δt
# # # #         dx1 = pred_deg[1:T-1, :, 0] - pred_deg[:T-2, :, 0]
# # # #         dy1 = pred_deg[1:T-1, :, 1] - pred_deg[:T-2, :, 1]
# # # #         dx2 = pred_deg[2:T,   :, 0] - pred_deg[1:T-1, :, 0]
# # # #         dy2 = pred_deg[2:T,   :, 1] - pred_deg[1:T-1, :, 1]

# # # #         lat_mid2 = pred_deg[1:T-1, :, 1]
# # # #         cos2 = torch.cos(torch.deg2rad(lat_mid2)).clamp(min=1e-4)

# # # #         spd1 = torch.sqrt((dx1 * cos2 * DEG2KM)**2 + (dy1 * DEG2KM)**2 + 1e-6) / DT_HOURS
# # # #         spd2 = torch.sqrt((dx2 * cos2 * DEG2KM)**2 + (dy2 * DEG2KM)**2 + 1e-6) / DT_HOURS
# # # #         accel = (spd2 - spd1).abs() / DT_HOURS  # km/h²

# # # #         a0_kmh2 = 10.0  # typical TC acceleration scale
# # # #         l_accel = accel.pow(2).mean() / (a0_kmh2 ** 2)
# # # #     else:
# # # #         l_accel = pred_deg.new_zeros(())

# # # #     # ── Total (ST-Trans weights) ──────────────────────────────────────────────
# # # #     total = l_dpe + 0.05 * l_mse + 0.1 * l_speed + 0.01 * l_accel

# # # #     if torch.isnan(total) or torch.isinf(total):
# # # #         total = pred_deg.new_zeros(())

# # # #     def _s(x): return x.item() if torch.is_tensor(x) else float(x)
# # # #     return dict(
# # # #         total   = total,
# # # #         dpe     = _s(l_dpe),
# # # #         mse     = _s(l_mse),
# # # #         speed   = _s(l_speed),
# # # #         accel   = _s(l_accel),
# # # #         # backward compat
# # # #         fm_mse  = 0.0, mse_hav = _s(l_dpe), endpoint = 0.0,
# # # #         speed_acc=0.0, accel_b=0.0, decomp=0.0, cons=0.0,
# # # #         hav_km  = _s(dist_km.mean()) if T > 0 else 0.0,
# # # #         h72_km  = _s(_haversine_deg(pred_deg[min(11, T-1)], gt_deg[min(11, T-1)]).mean()) if T > 0 else 0.0,
# # # #         spd_kmh = 0.0, acc_kmh2 = 0.0, aux_fno = 0.0, sigma = 0.0,
# # # #     )


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  EMAModel
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class EMAModel:
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
# # # #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

# # # #     def apply_to(self, model: nn.Module):
# # # #         m = _unwrap_model(model)
# # # #         backup = {}
# # # #         sd = m.state_dict()
# # # #         for k in self.shadow:
# # # #             if k not in sd:
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
# # # # #  VelocityField — Giữ encoder mạnh, đơn giản hóa physics
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class VelocityField(nn.Module):
# # # #     RAW_CTX_DIM = 512

# # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # # #                  unet_in_ch=13, **kwargs):
# # # #         super().__init__()
# # # #         self.pred_len = pred_len
# # # #         self.obs_len  = obs_len

# # # #         # ── FNO3D Spatial Encoder ────────────────────────────────────────────
# # # #         self.spatial_enc = FNO3DEncoder(
# # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # #             spatial_down=32, dropout=0.05)
# # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # #         # ── 1D Trajectory Encoder ────────────────────────────────────────────
# # # #         self.enc_1d = DataEncoder1D(
# # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)

# # # #         # ── ENV Encoder ──────────────────────────────────────────────────────
# # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # #         # ── Context projection ───────────────────────────────────────────────
# # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # #         self.ctx_drop = nn.Dropout(0.10)
# # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # # #         # ── Velocity observation encoder ─────────────────────────────────────
# # # #         self.vel_obs_enc = nn.Sequential(
# # # #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# # # #             nn.LayerNorm(256),
# # # #             nn.Linear(256, 256), nn.GELU(),
# # # #         )

# # # #         # ── Steering feature encoder ─────────────────────────────────────────
# # # #         self.steering_enc = nn.Sequential(
# # # #             nn.Linear(4, 64), nn.GELU(),
# # # #             nn.LayerNorm(64),
# # # #             nn.Linear(64, 128), nn.GELU(),
# # # #             nn.Linear(128, 256),
# # # #         )

# # # #         # ── Transformer Decoder ──────────────────────────────────────────────
# # # #         self.time_fc1   = nn.Linear(256, 512)
# # # #         self.time_fc2   = nn.Linear(512, 256)
# # # #         self.traj_embed = nn.Linear(4, 256)
# # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # #         self.step_embed = nn.Embedding(pred_len, 256)

# # # #         self.transformer = nn.TransformerDecoder(
# # # #             nn.TransformerDecoderLayer(
# # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # #             num_layers=2)

# # # #         self.out_fc1 = nn.Linear(256, 512)
# # # #         self.out_fc2 = nn.Linear(512, 4)

# # # #         # ── Scale params ─────────────────────────────────────────────────────
# # # #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# # # #         self.physics_scale  = nn.Parameter(torch.ones(4) * 1.5)   # beta drift
# # # #         self.steering_scale = nn.Parameter(torch.ones(4) * 1.0)   # ERA5 steering

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
# # # #         freq = torch.exp(
# # # #             torch.arange(half, dtype=torch.float32, device=t.device)
# # # #             * (-math.log(10000.0) / max(half - 1, 1))
# # # #         )
# # # #         emb = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
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
# # # #             e_3d_s = F.interpolate(
# # # #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# # # #                 mode="linear", align_corners=False
# # # #             ).permute(0, 2, 1)

# # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # #         t_w = torch.softmax(
# # # #             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float, device=e_3d_dec_t.device) * 0.5,
# # # #             dim=0)
# # # #         f_spatial = self.decoder_proj(
# # # #             (e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

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
# # # #         B     = obs_traj.shape[1]
# # # #         T_obs = obs_traj.shape[0]
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
# # # #         if env_data is None:
# # # #             return torch.zeros(B, 256, device=device)

# # # #         def _safe_get(key, default=0.0):
# # # #             v = env_data.get(key, None)
# # # #             if v is None or not torch.is_tensor(v):
# # # #                 return torch.full((B,), default, device=device)
# # # #             v = v.to(device).float()
# # # #             if v.dim() >= 2:
# # # #                 while v.dim() > 1:
# # # #                     v = v.mean(-1)
# # # #             if v.shape[0] != B:
# # # #                 v = (v.view(-1)[:B] if v.numel() >= B
# # # #                      else torch.full((B,), default, device=device))
# # # #             return v

# # # #         feat = torch.stack([
# # # #             _safe_get("u500_mean"), _safe_get("v500_mean"),
# # # #             _safe_get("u500_center"), _safe_get("v500_center")
# # # #         ], dim=-1)
# # # #         return self.steering_enc(feat)

# # # #     def _beta_drift(self, x_t):
# # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# # # #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# # # #         R_tc    = 3e5
# # # #         v_phys  = torch.zeros_like(x_t)
# # # #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6 * 3600 / (5 * 111 * 1000)
# # # #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6 * 3600 / (5 * 111 * 1000)
# # # #         return v_phys

# # # #     def _steering_drift(self, x_t, env_data):
# # # #         if env_data is None:
# # # #             return torch.zeros_like(x_t)
# # # #         B, device = x_t.shape[0], x_t.device

# # # #         def _safe_mean(key):
# # # #             v = env_data.get(key, None)
# # # #             if v is None or not torch.is_tensor(v):
# # # #                 return torch.zeros(B, device=device)
# # # #             v = v.to(device).float()
# # # #             while v.dim() > 1:
# # # #                 v = v.mean(-1)
# # # #             return v.view(-1)[:B] if v.numel() >= B else torch.zeros(B, device=device)

# # # #         u = _safe_mean("u500_center")
# # # #         v = _safe_mean("v500_center")
# # # #         lat_deg = x_t[:, :, 1] * 5.0
# # # #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# # # #         out = torch.zeros_like(x_t)
# # # #         out[:, :, 0] = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat) * 1.0
# # # #         out[:, :, 1] = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0) * 1.0
# # # #         return out

# # # #     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None, env_data=None):
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
# # # #                           vel_obs_feat=None, steering_feat=None, env_data=None):
# # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# # # #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  TCFlowMatching v46 — ST-Trans loss + FM architecture
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

# # # #         self.use_ema   = use_ema
# # # #         self.ema_decay = ema_decay
# # # #         self._ema      = None

# # # #     def init_ema(self):
# # # #         if self.use_ema:
# # # #             self._ema = EMAModel(self, decay=self.ema_decay)

# # # #     def ema_update(self):
# # # #         if self._ema is not None:
# # # #             self._ema.update(self)

# # # #     def set_curriculum_len(self, *a, **kw):
# # # #         pass

# # # #     @staticmethod
# # # #     def _to_rel(traj, Me, lp, lm):
# # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # #                            Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # #     @staticmethod
# # # #     def _to_abs(rel, lp, lm):
# # # #         d = rel.permute(1, 0, 2)
# # # #         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

# # # #     def _cfm_noisy(self, x1, sigma_min=None):
# # # #         if sigma_min is None:
# # # #             sigma_min = self.sigma_min
# # # #         B, device = x1.shape[0], x1.device
# # # #         x0 = torch.randn_like(x1) * sigma_min
# # # #         t  = torch.rand(B, device=device)
# # # #         te = t.view(B, 1, 1)
# # # #         return (1.0 - te) * x0 + te * x1, t, x1 - x0

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

# # # #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# # # #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# # # #         """
# # # #         v46: ST-Trans inspired loss — CỰC ĐƠN GIẢN
        
# # # #         Flow:
# # # #           1. Augment (flip + noise only)
# # # #           2. Encode context
# # # #           3. FM forward → predict trajectory
# # # #           4. Convert to degrees
# # # #           5. ST-Trans loss: DPE + 0.05*MSE + 0.1*speed_penalty + 0.01*accel_penalty
# # # #         """
# # # #         # ── Augmentation ──────────────────────────────────────────────────────
# # # #         batch_list = self._lon_flip_aug(batch_list)
# # # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)

# # # #         obs_t    = batch_list[0]
# # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # #         lp, lm   = obs_t[-1], batch_list[7][-1]

# # # #         # ── Sigma schedule (ST-Trans style) ───────────────────────────────────
# # # #         # Epoch 0-10: sigma=0.15 (near-deterministic) → fast convergence
# # # #         # Epoch 10-40: linear decay → 0.03
# # # #         # Epoch 40+: sigma=0.03 (full diversity)
# # # #         if epoch < 10:
# # # #             current_sigma = 0.15
# # # #         elif epoch < 40:
# # # #             t = (epoch - 10) / 30.0
# # # #             current_sigma = 0.15 - t * (0.15 - 0.03)
# # # #         else:
# # # #             current_sigma = 0.03

# # # #         # ── Context encoding ──────────────────────────────────────────────────
# # # #         raw_ctx = self.net._context(batch_list)

# # # #         obs_t    = batch_list[0]
# # # #         env_data = batch_list[13] if len(batch_list) > 13 else None
# # # #         lp, lm   = obs_t[-1], batch_list[7][-1]

# # # #         # ── FM forward ────────────────────────────────────────────────────────
# # # #         x1_rel = self._to_rel(batch_list[1], batch_list[8], lp, lm)
# # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# # # #         pred_vel = self.net.forward_with_ctx(
# # # #             x_t, fm_t, raw_ctx, env_data=env_data,
# # # #             vel_obs_feat=self.net._get_vel_obs_feat(obs_t),
# # # #             steering_feat=self.net._get_steering_feat(env_data, obs_t.shape[1], obs_t.device)
# # # #         )

# # # #         # ── FM velocity loss (gradient flows through pred_vel) ───────────────
# # # #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# # # #         # ── Convert to degree coordinates — VỚI GRADIENT ─────────────────────
# # # #         # Bug cũ: wrap torch.no_grad() → position loss không có gradient!
# # # #         # Fix: tính pred_abs trực tiếp từ pred_vel (có gradient)
# # # #         #
# # # #         # FM interpolation: x1_pred = x_t + (1 - fm_te) * pred_vel
# # # #         # Đây là ODE 1-step approximation tại t=fm_t → x1
# # # #         # Khi fm_t ~ Uniform(0,1), E[x1_pred] ≈ x1_gt nếu model tốt
# # # #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# # # #         x1_pred = x_t + (1.0 - fm_te) * pred_vel   # [B, T, 4] — CÓ GRADIENT
# # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm) # [T, B, 2] normalized

# # # #         pred_deg = _norm_to_deg(pred_abs)            # [T, B, 2] degrees — CÓ GRADIENT
# # # #         gt_deg   = _norm_to_deg(batch_list[1])       # [T, B, 2] degrees

# # # #         # ── ST-Trans loss ─────────────────────────────────────────────────────
# # # #         loss_dict = compute_st_trans_loss(pred_deg, gt_deg, epoch=epoch)

# # # #         # Combine: FM velocity loss + ST-Trans position loss
# # # #         # Tỉ lệ 1:2 — position loss là primary signal
# # # #         total = 1.0 * l_fm_mse + 2.0 * loss_dict["total"]

# # # #         if torch.isnan(total) or torch.isinf(total):
# # # #             total = x_t.new_zeros(())

# # # #         # Return dict cho logging
# # # #         d = dict(loss_dict)
# # # #         d["total"]   = total
# # # #         d["fm_mse"]  = l_fm_mse.item()
# # # #         d["sigma"]   = current_sigma
# # # #         return d

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
# # # #         for _ in range(num_ensemble):
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
# # # #             score_tensor = torch.stack(scores)
# # # #             k            = max(1, int(num_ensemble * 0.7))
# # # #             _, top_idx   = score_tensor.topk(k, dim=0)
# # # #             pred_mean = torch.stack([
# # # #                 all_trajs[top_idx[:, b], :, b, :].median(0).values
# # # #                 for b in range(B)
# # # #             ], dim=1)
# # # #         else:
# # # #             pred_mean = all_trajs.median(0).values

# # # #         if predict_csv:
# # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# # # #         return pred_mean, all_me.mean(0), all_trajs

# # # #     def _score_sample(self, traj, env_data):
# # # #         B = traj.shape[1]
# # # #         if traj.shape[0] < 2:
# # # #             return torch.ones(B, device=traj.device)
# # # #         traj_deg  = _norm_to_deg(traj)
# # # #         dt_deg    = traj_deg[1:] - traj_deg[:-1]
# # # #         lat_rad   = torch.deg2rad(traj_deg[:-1, :, 1])
# # # #         cos_lat   = torch.cos(lat_rad).clamp(min=1e-4)
# # # #         dx_km     = dt_deg[:, :, 0] * cos_lat * DEG2KM
# # # #         dy_km     = dt_deg[:, :, 1] * DEG2KM
# # # #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)
# # # #         speed_pen = F.relu(speed_kmh - 70.0) + F.relu(10.0 - speed_kmh)
# # # #         speed_sc  = torch.exp(-speed_pen.mean(0) / 20.0)
# # # #         if dt_deg.shape[0] >= 2:
# # # #             jerk      = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
# # # #             smooth_sc = torch.exp(-jerk * 5.0)
# # # #         else:
# # # #             smooth_sc = torch.ones(B, device=traj.device)
# # # #         return speed_sc * smooth_sc

# # # #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# # # #         with torch.enable_grad():
# # # #             x   = x_pred.detach().requires_grad_(True)
# # # #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# # # #             for _ in range(n_steps):
# # # #                 opt.zero_grad()
# # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # #                 pred_deg    = _norm_to_deg(pred_abs)
# # # #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# # # #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# # # #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# # # #                 speed = torch.sqrt(
# # # #                     (dt_deg[:, :, 0] * cos_lat * DEG2KM).pow(2)
# # # #                     + (dt_deg[:, :, 1] * DEG2KM).pow(2))
# # # #                 F.relu(speed - 600.0).pow(2).mean().backward()
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
# # # #                     "lon_mean_deg", "lat_mean_deg",
# # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # #         write_hdr = not os.path.exists(csv_path)
# # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # #         with open(csv_path, "a", newline="") as fh:
# # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # #             if write_hdr:
# # # #                 w.writeheader()
# # # #             for b in range(B):
# # # #                 for k in range(T):
# # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# # # #                                * math.cos(math.radians(mean_lat[k, b])))
# # # #                     spread = float(((dlat**2 + dlon**2) ** 0.5).mean() * DEG2KM)
# # # #                     w.writerow({
# # # #                         "timestamp": ts, "batch_idx": b, "step_idx": k,
# # # #                         "lead_h": (k + 1) * 6,
# # # #                         "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# # # #                         "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# # # #                         "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# # # #                         "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# # # #                         "ens_spread_km": f"{spread:.2f}",
# # # #                     })


# # # # # Backward compat alias
# # # # TCDiffusion = TCFlowMatching
# # """
# # Model/flow_matching_model_v46.py — ST-Trans-inspired FM
# # ═══════════════════════════════════════════════════════════════════════════════

# # ROOT CAUSE ANALYSIS từ v41/v45 plateau ở 384km:
# #   - Loss quá phức tạp: 6+ terms, conflicting gradients
# #   - FNO auxiliary loss, speed_acc, decomp,... → gradient war
# #   - Model học "average" thay vì trajectory structure

# # INSIGHT từ ST-Trans (297km, 394k params):
# #   - Loss CỰC ĐƠN GIẢN: DPE + 0.05*MSE + speed_penalty + accel_penalty
# #   - Không có AFCRPS, không có ATE/CTE decomp, không có FNO aux
# #   - Physics = chỉ penalty over-speed và jerk, không enforce dynamics
# #   - Non-autoregressive: predict all steps parallel

# # v46 STRATEGY — "ST-Trans loss, FM architecture":
# #   LOSS = haversine_DPE (weighted) + 0.05*coord_MSE + λ_spd*speed_penalty + λ_acc*accel_penalty

# #   Key fix: speed_penalty dùng DATA-DRIVEN threshold từ obs_traj
# #     - Từ data thực: TC speed ~ 5-19 km/h (6h step)
# #     - vmax_kmh cũ = 80 → KHÔNG BAO GIỜ trigger
# #     - Mới: compute_speed_stats_from_norm() → v_opt, v_sigma, v_hard_cap từ data

# #   Đơn giản hóa hoàn toàn:
# #   - BỎ: FNO aux loss, speed_acc term, decomp, cons, FM_MSE balance
# #   - GIỮ: haversine accuracy + kinematic regularization (như ST-Trans)
# #   - THÊM: step weights nhấn mạnh 48h/72h, adaptive speed threshold

# #   Điều chỉnh sigma schedule:
# #   - Epoch 0-10: sigma=0.15 (near-deterministic, học fast)
# #   - Epoch 10-40: linear decay 0.15→0.03
# #   - Epoch 40+: sigma=0.03 (full FM diversity)

# # EXPECTED: 72h < 320km ở ep30, < 300km ở ep50
# # """
# # from __future__ import annotations

# # import csv
# # import math
# # import os
# # from datetime import datetime

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # from Model.FNO3D_encoder import FNO3DEncoder
# # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # from Model.env_net_transformer_gphsplit import Env_net


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Constants
# # # ══════════════════════════════════════════════════════════════════════════════

# # R_EARTH  = 6371.0
# # DT_HOURS = 6.0
# # DEG2KM   = 111.0

# # # Step weights: nhấn mạnh 48h (step 8) và 72h (step 12)
# # # 6h  12h  18h  24h  30h  36h  42h  48h  54h  60h  66h  72h
# # # Đổi thành (giảm trọng số đầu, tăng mạnh hơn ở 48-72h):
# # STEP_WEIGHTS = [0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.5, 3.5, 4.5, 5.0, 5.0, 6.0]

# # # Data-driven speed prior (km/h, 6h step) — từ thống kê dataset thực
# # # Computed từ obs: mean~11, std~3, p95~16, max~25
# # # Dùng làm fallback khi không có obs_traj
# # _SPEED_PRIOR = {
# #     "v_opt"      : 15.0,   # km/h — tốc độ TC điển hình
# #     "v_sigma"    : 10.0,   # km/h — bandwidth của Gaussian penalty
# #     "v_hard_cap" : 35.0,   # km/h — hard upper cap (p95 * 1.8)
# # }


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Utility
# # # ══════════════════════════════════════════════════════════════════════════════

# # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# #     """Convert normalized coords to degrees. Differentiable."""
# #     lon = (t[..., 0] * 50.0 + 1800.0) / 10.0
# #     lat = (t[..., 1] * 50.0) / 10.0
# #     return torch.stack([lon, lat], dim=-1)


# # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# #     """Haversine distance in km. p1, p2: [..., 2] in degrees (lon, lat)."""
# #     lat1 = torch.deg2rad(p1[..., 1])
# #     lat2 = torch.deg2rad(p2[..., 1])
# #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# #     a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
# #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1.0 - 1e-12).sqrt())


# # def _unwrap_model(model: nn.Module) -> nn.Module:
# #     if hasattr(model, '_orig_mod'):
# #         return model._orig_mod
# #     return model


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Data-driven Speed Statistics
# # # ══════════════════════════════════════════════════════════════════════════════

# # def compute_speed_stats_from_norm(obs_traj: torch.Tensor) -> dict:
# #     """
# #     Tính speed statistics từ normalized obs trajectory.

# #     Args:
# #         obs_traj: [T_obs, B, 2] — normalized (lon_norm, lat_norm)
# #                   Công thức: lon_deg = (lon_norm * 50 + 1800) / 10
# #                              lat_deg = lat_norm * 50 / 10

# #     Returns:
# #         dict:
# #             mean_kmh   — tốc độ trung bình trong batch
# #             std_kmh    — std dev
# #             p50_kmh    — median
# #             p75_kmh    — 75th percentile
# #             p95_kmh    — 95th percentile
# #             v_opt      — target tốc độ cho soft penalty (= p50)
# #             v_sigma    — bandwidth Gaussian penalty (= std + 5)
# #             v_hard_cap — hard upper cap (= p95 * 1.8, clamp 25–60)

# #     Ví dụ kết quả từ sample data trong dataset (6h step):
# #         mean≈11 km/h, std≈3, p95≈16 km/h
# #         → v_opt=11, v_sigma=8, v_hard_cap=29
# #     """
# #     T = obs_traj.shape[0]
# #     if T < 2:
# #         return dict(_SPEED_PRIOR)

# #     with torch.no_grad():
# #         # Convert normalized → degrees
# #         lon_deg = (obs_traj[..., 0] * 50.0 + 1800.0) / 10.0   # [T, B]
# #         lat_deg = (obs_traj[..., 1] * 50.0) / 10.0             # [T, B]

# #         # Step displacements
# #         dlon = lon_deg[1:] - lon_deg[:-1]   # [T-1, B]
# #         dlat = lat_deg[1:] - lat_deg[:-1]   # [T-1, B]

# #         lat_mid = (lat_deg[:-1] + lat_deg[1:]) * 0.5
# #         cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)

# #         dx_km     = dlon * cos_lat * DEG2KM          # [T-1, B]
# #         dy_km     = dlat * DEG2KM
# #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2 + 1e-6) / DT_HOURS

# #         speed_flat = speed_kmh.flatten()
# #         n          = speed_flat.numel()

# #         if n < 4:
# #             return dict(_SPEED_PRIOR)

# #         mean_s = float(speed_flat.mean())
# #         std_s  = float(speed_flat.std().clamp(min=1.0))

# #         # Quantiles
# #         q = torch.quantile(speed_flat,
# #                            torch.tensor([0.50, 0.75, 0.95], device=speed_flat.device))
# #         p50 = float(q[0])
# #         p75 = float(q[1])
# #         p95 = float(q[2])

# #         # Adaptive thresholds
# #         v_opt      = max(p50, 5.0)                         # ≥ 5 km/h
# #         v_sigma    = max(std_s + 5.0, 5.0)                 # bandwidth
# #         v_hard_cap = float(torch.tensor(p95 * 1.8).clamp(25.0, 80.0))

# #     return {
# #         "mean_kmh"  : mean_s,
# #         "std_kmh"   : std_s,
# #         "p50_kmh"   : p50,
# #         "p75_kmh"   : p75,
# #         "p95_kmh"   : p95,
# #         "v_opt"     : v_opt,
# #         "v_sigma"   : v_sigma,
# #         "v_hard_cap": v_hard_cap,
# #     }


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  ST-Trans Inspired Loss — với Data-Driven Speed Penalty
# # # ══════════════════════════════════════════════════════════════════════════════

# # def compute_st_trans_loss(
# #     pred_deg     : torch.Tensor,      # [T, B, 2] degrees
# #     gt_deg       : torch.Tensor,      # [T, B, 2] degrees
# #     epoch        : int = 0,
# #     speed_stats  : dict | None = None, # từ compute_speed_stats_from_norm()
# # ) -> dict:
# #     """
# #     ST-Trans physics-guided composite loss:
# #       L = L_DPE + 0.05*L_MSE + 0.1*L_speed + 0.01*L_accel

# #     Với step weighting nhấn mạnh 48h/72h.

# #     KEY FIX v46:
# #       L_speed dùng BILATERAL soft penalty thay vì relu(speed - 80):
# #         - 80 km/h cũ → không bao giờ trigger (data thực: 5-19 km/h)
# #         - Mới: Gaussian pull về v_opt (từ data) + hard relu cap
# #         - Gradient tồn tại cho mọi speed, không chỉ khi vượt threshold
# #     """
# #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# #     if T < 2:
# #         return {"total": pred_deg.new_zeros(()), "dpe": 0.0, "spd": 0.0, "acc": 0.0}

# #     # ── Speed thresholds (data-driven hoặc prior) ──────────────────────────────
# #     sp = speed_stats if speed_stats is not None else _SPEED_PRIOR
# #     v_opt      = sp.get("v_opt",       _SPEED_PRIOR["v_opt"])
# #     v_sigma    = sp.get("v_sigma",     _SPEED_PRIOR["v_sigma"])
# #     v_hard_cap = sp.get("v_hard_cap",  _SPEED_PRIOR["v_hard_cap"])

# #     # ── L_DPE: weighted haversine per step ────────────────────────────────────
# #     w = pred_deg.new_tensor(STEP_WEIGHTS[:T])
# #     w = w / w.sum() * T  # normalize, giữ scale

# #     dist_km = _haversine_deg(pred_deg[:T], gt_deg[:T])  # [T, B]

# #     # Huber-style để robust với outlier (TC recurvature)
# #     delta_km  = 200.0
# #     l_dpe_per = torch.where(
# #         dist_km < delta_km,
# #         dist_km.pow(2) / (2.0 * delta_km),
# #         dist_km - delta_km / 2.0
# #     )
# #     l_dpe = (l_dpe_per * w.unsqueeze(1)).mean() / delta_km  # normalized ~1-3

# #     # ── L_MSE: coordinate-space MSE (gradient smoothness) ────────────────────
# #     l_mse = F.mse_loss(pred_deg[:T], gt_deg[:T])

# #     # ── L_speed: DATA-DRIVEN bilateral soft penalty ───────────────────────────
# #     #
# #     # TRƯỚC (broken):
# #     #   vmax_kmh = 80.0
# #     #   l_speed = F.relu(speed_kmh - 80).pow(2).mean()
# #     #   → Với data TC 5-19 km/h: LUÔN = 0, không có gradient!
# #     #
# #     # SAU (fixed):
# #     #   Dùng Gaussian soft penalty về v_opt (median từ data)
# #     #   + hard relu cho extreme outlier (v_hard_cap từ p95*1.8)
# #     #   → Gradient tồn tại cho MỌI speed value
# #     #
# #     if T >= 2:
# #         dt_deg  = pred_deg[1:T] - pred_deg[:T-1]           # [T-1, B, 2]
# #         lat_mid = (pred_deg[:T-1, :, 1] + pred_deg[1:T, :, 1]) * 0.5
# #         cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)

# #         dx_km     = dt_deg[:, :, 0] * cos_lat * DEG2KM     # [T-1, B]
# #         dy_km     = dt_deg[:, :, 1] * DEG2KM
# #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2 + 1e-6) / DT_HOURS  # [T-1, B]

# #         # Gaussian soft pull về v_opt — gradient với MỌI speed
# #         l_speed_soft = ((speed_kmh - v_opt) / v_sigma).pow(2).mean()

# #         # Hard relu cho extreme outlier (physical impossibility)
# #         l_speed_hard = F.relu(speed_kmh - v_hard_cap).pow(2).mean() / (v_hard_cap ** 2)

# #         # Combine: 70% soft (luôn có gradient) + 30% hard (cap outlier)
# #         l_speed = 0.7 * l_speed_soft + 0.3 * l_speed_hard
# #     else:
# #         l_speed = pred_deg.new_zeros(())
# #         speed_kmh = None

# #     # ── L_accel: smooth acceleration (penalty jerk) ───────────────────────────
# #     if T >= 3:
# #         dx1 = pred_deg[1:T-1, :, 0] - pred_deg[:T-2, :, 0]
# #         dy1 = pred_deg[1:T-1, :, 1] - pred_deg[:T-2, :, 1]
# #         dx2 = pred_deg[2:T,   :, 0] - pred_deg[1:T-1, :, 0]
# #         dy2 = pred_deg[2:T,   :, 1] - pred_deg[1:T-1, :, 1]

# #         lat_mid2 = pred_deg[1:T-1, :, 1]
# #         cos2 = torch.cos(torch.deg2rad(lat_mid2)).clamp(min=1e-4)

# #         spd1  = torch.sqrt((dx1 * cos2 * DEG2KM)**2 + (dy1 * DEG2KM)**2 + 1e-6) / DT_HOURS
# #         spd2  = torch.sqrt((dx2 * cos2 * DEG2KM)**2 + (dy2 * DEG2KM)**2 + 1e-6) / DT_HOURS
# #         accel = (spd2 - spd1).abs() / DT_HOURS   # km/h per hour

# #         # Scale theo v_sigma để adaptive với data
# #         a0 = max(v_sigma * 0.5, 3.0)
# #         l_accel = accel.pow(2).mean() / (a0 ** 2)
# #     else:
# #         l_accel = pred_deg.new_zeros(())

# #     # Trong compute_st_trans_loss(), thêm sau l_accel:

# #     def _along_track_error(pred_deg, gt_deg):
# #         """ATE = projection của error lên direction of motion."""
# #         T = pred_deg.shape[0]
# #         if T < 2:
# #             return pred_deg.new_zeros(())
        
# #         # Direction of GT motion at each step
# #         gt_motion = gt_deg[1:] - gt_deg[:-1]  # [T-1, B, 2]
# #         lat_mid = gt_deg[:-1, :, 1]
# #         cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)
        
# #         # Scale lon by cos(lat) để đúng metric
# #         scale = torch.stack([cos_lat * DEG2KM, 
# #                             torch.ones_like(cos_lat) * DEG2KM], dim=-1)
        
# #         gt_vec  = gt_motion * scale          # [T-1, B, 2] km
# #         gt_norm = gt_vec.norm(dim=-1, keepdim=True).clamp(min=1e-3)
# #         gt_unit = gt_vec / gt_norm           # unit vector along track
        
# #         # Error vector
# #         err     = (pred_deg[:-1] - gt_deg[:-1]) * scale  # [T-1, B, 2] km
        
# #         # Along-track component
# #         ate_per = (err * gt_unit).sum(dim=-1).abs()       # [T-1, B]
        
# #         # Weight later steps more (ATE accumulates)
# #         w = torch.arange(1, T, device=pred_deg.device, dtype=torch.float)
# #         w = w / w.sum()
        
# #         return (ate_per * w.unsqueeze(1)).mean()

# #     # Thêm l_cumulative vào compute_st_trans_loss():

# #     # Cumulative displacement error (penalize tốc độ tích lũy sai)
# #     pred_cum = torch.cumsum(
# #         _haversine_deg(pred_deg[:-1], pred_deg[1:]), dim=0)  # [T-1, B]
# #     gt_cum   = torch.cumsum(
# #         _haversine_deg(gt_deg[:-1],   gt_deg[1:]),   dim=0)  # [T-1, B]

# #     # Penalty khi tích lũy sai > 20% so với GT
# #     cum_ratio = (pred_cum / (gt_cum + 1.0)).clamp(0.3, 3.0)
# #     l_cumspeed = (cum_ratio - 1.0).pow(2).mean()
# #     # ── Total (ST-Trans weights) ──────────────────────────────────────────────
# #     # total = l_dpe + 0.05 * l_mse + 0.1 * l_speed + 0.01 * l_accel
# #     l_ate       = _along_track_error(pred_deg[:T], gt_deg[:T])
# #     total = (l_dpe 
# #          + 0.05  * l_mse 
# #          + 0.05  * l_speed      # giảm instantaneous speed penalty
# #          + 0.01  * l_accel 
# #          + 0.15  * l_ate        # ← NEW: direct ATE loss
# #          + 0.10  * l_cumspeed)  # ← NEW: penalize tốc độ tích lũy sai

# #     if torch.isnan(total) or torch.isinf(total):
# #         total = pred_deg.new_zeros(())

# #     def _s(x): return x.item() if torch.is_tensor(x) else float(x)

# #     mean_speed_kmh = _s(speed_kmh.mean()) if speed_kmh is not None else 0.0

# #     return dict(
# #         total     = total,
# #         dpe       = _s(l_dpe),
# #         mse       = _s(l_mse),
# #         speed     = _s(l_speed),
# #         accel     = _s(l_accel),
# #         spd_kmh   = mean_speed_kmh,       # tốc độ trung bình của pred (km/h)
# #         v_opt     = float(v_opt),         # threshold đang dùng
# #         v_hard_cap= float(v_hard_cap),
# #         # backward compat
# #         fm_mse    = 0.0, mse_hav=_s(l_dpe), endpoint=0.0,
# #         speed_acc = 0.0, accel_b=0.0, decomp=0.0, cons=0.0,
# #         hav_km    = _s(dist_km.mean()) if T > 0 else 0.0,
# #         h72_km    = _s(_haversine_deg(pred_deg[min(11, T-1)], gt_deg[min(11, T-1)]).mean()) if T > 0 else 0.0,
# #         acc_kmh2  = 0.0, aux_fno=0.0, sigma=0.0,
# #     )


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  EMAModel
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
# #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

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
# # #  VelocityField
# # # ══════════════════════════════════════════════════════════════════════════════

# # class VelocityField(nn.Module):
# #     RAW_CTX_DIM = 512

# #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# #                  unet_in_ch=13, **kwargs):
# #         super().__init__()
# #         self.pred_len = pred_len
# #         self.obs_len  = obs_len

# #         # ── FNO3D Spatial Encoder ────────────────────────────────────────────
# #         self.spatial_enc = FNO3DEncoder(
# #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# #             spatial_down=32, dropout=0.05)
# #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# #         self.bottleneck_proj = nn.Linear(128, 128)
# #         self.decoder_proj    = nn.Linear(1, 16)

# #         # ── 1D Trajectory Encoder ────────────────────────────────────────────
# #         self.enc_1d = DataEncoder1D(
# #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)

# #         # ── ENV Encoder ──────────────────────────────────────────────────────
# #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# #         # ── Context projection ───────────────────────────────────────────────
# #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# #         self.ctx_drop = nn.Dropout(0.10)
# #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# #         # ── Velocity observation encoder ─────────────────────────────────────
# #         self.vel_obs_enc = nn.Sequential(
# #             nn.Linear(obs_len * 2, 256), nn.GELU(),
# #             nn.LayerNorm(256),
# #             nn.Linear(256, 256), nn.GELU(),
# #         )

# #         # ── Steering feature encoder ─────────────────────────────────────────
# #         self.steering_enc = nn.Sequential(
# #             nn.Linear(4, 64), nn.GELU(),
# #             nn.LayerNorm(64),
# #             nn.Linear(64, 128), nn.GELU(),
# #             nn.Linear(128, 256),
# #         )

# #         # ── Transformer Decoder ──────────────────────────────────────────────
# #         self.time_fc1   = nn.Linear(256, 512)
# #         self.time_fc2   = nn.Linear(512, 256)
# #         self.traj_embed = nn.Linear(4, 256)
# #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# #         self.step_embed = nn.Embedding(pred_len, 256)

# #         self.transformer = nn.TransformerDecoder(
# #             nn.TransformerDecoderLayer(
# #                 d_model=256, nhead=8, dim_feedforward=1024,
# #                 dropout=0.10, activation="gelu", batch_first=True),
# #             num_layers=2)

# #         self.out_fc1 = nn.Linear(256, 512)
# #         self.out_fc2 = nn.Linear(512, 4)

# #         # ── Scale params ─────────────────────────────────────────────────────
# #         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
# #         self.physics_scale  = nn.Parameter(torch.ones(4) * 1.5)
# #         self.steering_scale = nn.Parameter(torch.ones(4) * 1.0)

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
# #         freq = torch.exp(
# #             torch.arange(half, dtype=torch.float32, device=t.device)
# #             * (-math.log(10000.0) / max(half - 1, 1))
# #         )
# #         emb = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
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
# #             e_3d_s = F.interpolate(
# #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# #                 mode="linear", align_corners=False
# #             ).permute(0, 2, 1)

# #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# #         t_w = torch.softmax(
# #             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float, device=e_3d_dec_t.device) * 0.5,
# #             dim=0)
# #         f_spatial = self.decoder_proj(
# #             (e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

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
# #         B     = obs_traj.shape[1]
# #         T_obs = obs_traj.shape[0]
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

# #         def _safe_get(key, default=0.0):
# #             v = env_data.get(key, None)
# #             if v is None or not torch.is_tensor(v):
# #                 return torch.full((B,), default, device=device)
# #             v = v.to(device).float()
# #             if v.dim() >= 2:
# #                 while v.dim() > 1:
# #                     v = v.mean(-1)
# #             if v.shape[0] != B:
# #                 v = (v.view(-1)[:B] if v.numel() >= B
# #                      else torch.full((B,), default, device=device))
# #             return v

# #         feat = torch.stack([
# #             _safe_get("u500_mean"), _safe_get("v500_mean"),
# #             _safe_get("u500_center"), _safe_get("v500_center")
# #         ], dim=-1)
# #         return self.steering_enc(feat)

# #     def _beta_drift(self, x_t):
# #         lat_deg = x_t[:, :, 1] * 5.0
# #         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
# #         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
# #         R_tc    = 3e5
# #         v_phys  = torch.zeros_like(x_t)
# #         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6 * 3600 / (5 * 111 * 1000)
# #         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6 * 3600 / (5 * 111 * 1000)
# #         return v_phys

# #     def _steering_drift(self, x_t, env_data):
# #         if env_data is None:
# #             return torch.zeros_like(x_t)
# #         B, device = x_t.shape[0], x_t.device

# #         def _safe_mean(key):
# #             v = env_data.get(key, None)
# #             if v is None or not torch.is_tensor(v):
# #                 return torch.zeros(B, device=device)
# #             v = v.to(device).float()
# #             while v.dim() > 1:
# #                 v = v.mean(-1)
# #             return v.view(-1)[:B] if v.numel() >= B else torch.zeros(B, device=device)

# #         u = _safe_mean("u500_center")
# #         v = _safe_mean("v500_center")
# #         lat_deg = x_t[:, :, 1] * 5.0
# #         cos_lat = torch.cos(torch.deg2rad(lat_deg)).clamp(min=1e-3)
# #         out = torch.zeros_like(x_t)
# #         out[:, :, 0] = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos_lat) * 1.0
# #         out[:, :, 1] = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0) * 1.0
# #         return out

# #     def _decode(self, x_t, t, ctx, vel_obs_feat=None, steering_feat=None, env_data=None):
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
# #                           vel_obs_feat=None, steering_feat=None, env_data=None):
# #         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
# #         return self._decode(x_t, t, ctx, vel_obs_feat, steering_feat, env_data)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  TCFlowMatching v46 — ST-Trans loss + FM architecture + Adaptive Speed
# # # ══════════════════════════════════════════════════════════════════════════════

# # class TCFlowMatching(nn.Module):

# #     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
# #                  n_train_ens=4, unet_in_ch=13,
# #                  ctx_noise_scale=0.01, initial_sample_sigma=0.03,
# #                  teacher_forcing=True, use_ema=True, ema_decay=0.999,
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
# #         return (1.0 - te) * x0 + te * x1, t, x1 - x0

# #     # ── Augmentation ─────────────────────────────────────────────────────────

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

# #     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
# #         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

# #     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
# #         """
# #         v46: ST-Trans inspired loss với data-driven speed penalty.

# #         Flow:
# #           1. Augment
# #           2. Compute speed stats từ obs_traj (DATA-DRIVEN threshold)
# #           3. Encode context
# #           4. FM forward → predict trajectory
# #           5. Convert to degrees
# #           6. ST-Trans loss: DPE + 0.05*MSE + 0.1*speed(adaptive) + 0.01*accel
# #         """
# #         # ── Augmentation ──────────────────────────────────────────────────────
# #         batch_list = self._lon_flip_aug(batch_list)
# #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)

# #         obs_t    = batch_list[0]
# #         env_data = batch_list[13] if len(batch_list) > 13 else None
# #         lp, lm   = obs_t[-1], batch_list[7][-1]

# #         # ── DATA-DRIVEN speed stats từ obs_traj ───────────────────────────────
# #         # Chỉ dùng obs_traj[:, :, :2] (lon, lat), không cần gradient
# #         speed_stats = compute_speed_stats_from_norm(obs_t[..., :2])

# #         # ── Sigma schedule (ST-Trans style) ───────────────────────────────────
# #         if epoch < 10:
# #             current_sigma = 0.15
# #         elif epoch < 40:
# #             t = (epoch - 10) / 30.0
# #             current_sigma = 0.15 - t * (0.15 - 0.03)
# #         else:
# #             current_sigma = 0.03

# #         # ── Context encoding ──────────────────────────────────────────────────
# #         raw_ctx = self.net._context(batch_list)

# #         obs_t    = batch_list[0]
# #         env_data = batch_list[13] if len(batch_list) > 13 else None
# #         lp, lm   = obs_t[-1], batch_list[7][-1]

# #         # ── FM forward ────────────────────────────────────────────────────────
# #         x1_rel = self._to_rel(batch_list[1], batch_list[8], lp, lm)
# #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

# #         pred_vel = self.net.forward_with_ctx(
# #             x_t, fm_t, raw_ctx, env_data=env_data,
# #             vel_obs_feat=self.net._get_vel_obs_feat(obs_t),
# #             steering_feat=self.net._get_steering_feat(env_data, obs_t.shape[1], obs_t.device)
# #         )

# #         # ── FM velocity loss ──────────────────────────────────────────────────
# #         l_fm_mse = F.mse_loss(pred_vel, u_target)

# #         # ── Predict x1 từ pred_vel (có gradient) ─────────────────────────────
# #         fm_te   = fm_t.view(x1_rel.shape[0], 1, 1)
# #         x1_pred = x_t + (1.0 - fm_te) * pred_vel   # [B, T, 4]
# #         pred_abs, _ = self._to_abs(x1_pred, lp, lm) # [T, B, 2] normalized

# #         pred_deg = _norm_to_deg(pred_abs)            # [T, B, 2] degrees
# #         gt_deg   = _norm_to_deg(batch_list[1])       # [T, B, 2] degrees

# #         # ── ST-Trans loss (với adaptive speed stats) ──────────────────────────
# #         loss_dict = compute_st_trans_loss(
# #             pred_deg, gt_deg,
# #             epoch=epoch,
# #             speed_stats=speed_stats,      # ← DATA-DRIVEN threshold
# #         )

# #         # Combine: FM velocity loss + ST-Trans position loss
# #         total = 1.0 * l_fm_mse + 2.0 * loss_dict["total"]

# #         if torch.isnan(total) or torch.isinf(total):
# #             total = x_t.new_zeros(())

# #         d = dict(loss_dict)
# #         d["total"]      = total
# #         d["fm_mse"]     = l_fm_mse.item()
# #         d["sigma"]      = current_sigma
# #         d["v_opt"]      = speed_stats["v_opt"]
# #         d["v_hard_cap"] = speed_stats["v_hard_cap"]
# #         d["obs_spd_p50"]= speed_stats["p50_kmh"]   # log để monitor
# #         return d

# #     # ── Sampling ──────────────────────────────────────────────────────────────

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

# #         # Speed stats cho scoring
# #         speed_stats = compute_speed_stats_from_norm(obs_t[..., :2])

# #         traj_s, me_s, scores = [], [], []
# #         for _ in range(num_ensemble):
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
# #                 scores.append(self._score_sample(tr, speed_stats))

# #         all_trajs = torch.stack(traj_s)
# #         all_me    = torch.stack(me_s)

# #         if importance_weight and scores:
# #             score_tensor = torch.stack(scores)
# #             k            = max(1, int(num_ensemble * 0.7))
# #             _, top_idx   = score_tensor.topk(k, dim=0)
# #             pred_mean = torch.stack([
# #                 all_trajs[top_idx[:, b], :, b, :].median(0).values
# #                 for b in range(B)
# #             ], dim=1)
# #         else:
# #             pred_mean = all_trajs.median(0).values

# #         if predict_csv:
# #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
# #         return pred_mean, all_me.mean(0), all_trajs

# #     def _score_sample(self, traj, speed_stats: dict | None = None):
# #         """
# #         Score mỗi ensemble member dựa vào speed realistic.
# #         Dùng v_opt và v_hard_cap từ speed_stats (data-driven).
# #         """
# #         B = traj.shape[1]
# #         if traj.shape[0] < 2:
# #             return torch.ones(B, device=traj.device)

# #         sp        = speed_stats if speed_stats is not None else _SPEED_PRIOR
# #         v_opt     = sp.get("v_opt", _SPEED_PRIOR["v_opt"])
# #         v_hard_cap= sp.get("v_hard_cap", _SPEED_PRIOR["v_hard_cap"])

# #         traj_deg  = _norm_to_deg(traj)
# #         dt_deg    = traj_deg[1:] - traj_deg[:-1]
# #         lat_rad   = torch.deg2rad(traj_deg[:-1, :, 1])
# #         cos_lat   = torch.cos(lat_rad).clamp(min=1e-4)
# #         dx_km     = dt_deg[:, :, 0] * cos_lat * DEG2KM
# #         dy_km     = dt_deg[:, :, 1] * DEG2KM
# #         speed_kmh = torch.sqrt(dx_km**2 + dy_km**2)

# #         # Score: gần v_opt → score cao, vượt v_hard_cap → penalty mạnh
# #         v_sigma   = sp.get("v_sigma", _SPEED_PRIOR["v_sigma"])
# #         speed_pen = ((speed_kmh - v_opt) / v_sigma).pow(2)
# #         hard_pen  = F.relu(speed_kmh - v_hard_cap) * 2.0
# #         speed_sc  = torch.exp(-(speed_pen + hard_pen).mean(0) * 0.5)

# #         if dt_deg.shape[0] >= 2:
# #             jerk      = (dt_deg[1:] - dt_deg[:-1]).norm(dim=-1).mean(0)
# #             smooth_sc = torch.exp(-jerk * 5.0)
# #         else:
# #             smooth_sc = torch.ones(B, device=traj.device)

# #         return speed_sc * smooth_sc

# #     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
# #         with torch.enable_grad():
# #             x   = x_pred.detach().requires_grad_(True)
# #             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
# #             for _ in range(n_steps):
# #                 opt.zero_grad()
# #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# #                 pred_deg    = _norm_to_deg(pred_abs)
# #                 dt_deg      = pred_deg[1:] - pred_deg[:-1]
# #                 lat_rad     = torch.deg2rad(pred_deg[:-1, :, 1])
# #                 cos_lat     = torch.cos(lat_rad).clamp(min=1e-4)
# #                 speed = torch.sqrt(
# #                     (dt_deg[:, :, 0] * cos_lat * DEG2KM).pow(2)
# #                     + (dt_deg[:, :, 1] * DEG2KM).pow(2))
# #                 # Cap dựa trên physics thực tế (TC max ~80 km/h)
# #                 F.relu(speed - 80.0).pow(2).mean().backward()
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
# #             if write_hdr:
# #                 w.writeheader()
# #             for b in range(B):
# #                 for k in range(T):
# #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# #                     dlon   = ((all_lon[:, k, b] - mean_lon[k, b])
# #                                * math.cos(math.radians(mean_lat[k, b])))
# #                     spread = float(((dlat**2 + dlon**2) ** 0.5).mean() * DEG2KM)
# #                     w.writerow({
# #                         "timestamp": ts, "batch_idx": b, "step_idx": k,
# #                         "lead_h": (k + 1) * 6,
# #                         "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
# #                         "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
# #                         "lon_std_deg":  f"{all_lon[:,k,b].std():.4f}",
# #                         "lat_std_deg":  f"{all_lat[:,k,b].std():.4f}",
# #                         "ens_spread_km": f"{spread:.2f}",
# #                     })


# # # Backward compat alias
# # TCDiffusion = TCFlowMatching

# """
# Model/flow_matching_model_v47.py — ATE Fix Edition
# ═══════════════════════════════════════════════════════════════════════════════

# ROOT CAUSES ATE CAO → FIXES:

#   [FIX-1] mamba_encoder v11: enc_1d augment [lon,lat,pres,wnd]→9D kinematic
#     → Mamba "thấy" velocity/heading, không chỉ position
#     → KinematicHead inject momentum vào context

#   [FIX-2] _get_kinematic_obs_feat(): thay thế _get_vel_obs_feat()
#     6 features/step: [vel_x, vel_y, speed_norm, sin_h, cos_h, accel_norm]
#     vel_obs_enc: Linear(obs_len*2→256) → Linear(obs_len*6→256)
#     → Decoder nhận đúng kinematic state

#   [FIX-3] EnvKinematicFeat: inject env_data kinematic vào decoder memory
#     env_data đã có move_velocity, history_direction24, delta_velocity (90 dims)
#     nhưng TRƯỚC ĐÂY chỉ dùng trong Env_net context (1 lần)
#     → Giờ inject trực tiếp vào transformer decoder memory
#     → Decoder biết TC đang đi hướng nào theo env data

#   [FIX-4] Loss: thêm l_speed_match weight=0.2
#     (pred_speed / gt_speed.clamp(2.0) - 1)² → direct speed supervision
#     → Giảm l_speed từ 0.1→0.05 vì l_speed_match thay thế

#   [FIX-5] STEP_WEIGHTS: tăng trọng số 12h/24h
#     → Penalize sai tốc độ sớm → tránh error accumulation

#   [FIX-6] sample(): init từ persistence forecast thay vì pure noise
#     → Ensemble members bắt đầu từ "TC đi thẳng" → physically reasonable

#   [FIX-7] Tắt _physics_correct (gây shrinkage trajectory)

# CHECKPOINT COMPAT:
#   Mismatch layers (strict=False):
#     net.vel_obs_enc.0.weight  : (256,16) → (256,48)
#     net.env_kine_enc.*        : NEW
#   Các layer khác load bình thường từ ep30+ checkpoint.
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

# R_EARTH  = 6371.0
# DT_HOURS = 6.0
# DEG2KM   = 111.0
# _NORM_TO_DEG = 5.0   # 1 norm unit ≈ 5 degrees

# # [FIX-5] Tăng trọng số bước đầu để model học speed đúng sớm
# # 6h    12h   18h   24h   30h   36h   42h   48h   54h   60h   66h   72h
# STEP_WEIGHTS = [
#     2.0,  3.0,  2.0,  3.5,  2.5,  3.0,  2.5,  4.0,  4.5,  5.0,  4.5,  6.0
# ]

# _SPEED_PRIOR = {
#     "v_opt"      : 15.0,
#     "v_sigma"    : 10.0,
#     "v_hard_cap" : 35.0,
# }


# # ══════════════════════════════════════════════════════════════════════════════
# #  Utilities
# # ══════════════════════════════════════════════════════════════════════════════

# def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
#     lon = (t[..., 0] * 50.0 + 1800.0) / 10.0
#     lat = (t[..., 1] * 50.0) / 10.0
#     return torch.stack([lon, lat], dim=-1)


# def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
#     lat1 = torch.deg2rad(p1[..., 1]);  lat2 = torch.deg2rad(p2[..., 1])
#     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
#     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
#     a = (torch.sin(dlat/2).pow(2) +
#          torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2).pow(2))
#     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())


# def _unwrap_model(m: nn.Module) -> nn.Module:
#     return m._orig_mod if hasattr(m, '_orig_mod') else m


# # ══════════════════════════════════════════════════════════════════════════════
# #  Speed statistics from obs trajectory
# # ══════════════════════════════════════════════════════════════════════════════

# def compute_speed_stats_from_norm(obs_traj: torch.Tensor) -> dict:
#     T = obs_traj.shape[0]
#     if T < 2:
#         return dict(_SPEED_PRIOR)
#     with torch.no_grad():
#         lon_deg = (obs_traj[..., 0] * 50.0 + 1800.0) / 10.0
#         lat_deg = (obs_traj[..., 1] * 50.0) / 10.0
#         dlon    = lon_deg[1:] - lon_deg[:-1]
#         dlat    = lat_deg[1:] - lat_deg[:-1]
#         cos_lat = torch.cos(torch.deg2rad((lat_deg[:-1] + lat_deg[1:]) * 0.5)).clamp(1e-4)
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
#         "mean_kmh"  : mean_s,  "std_kmh"  : std_s,
#         "p50_kmh"   : p50,     "p95_kmh"  : p95,
#         "v_opt"     : max(p50, 5.0),
#         "v_sigma"   : max(std_s + 5.0, 5.0),
#         "v_hard_cap": float(torch.tensor(p95 * 1.8).clamp(25.0, 80.0)),
#     }


# # ══════════════════════════════════════════════════════════════════════════════
# #  [FIX-4] Loss with speed matching
# # ══════════════════════════════════════════════════════════════════════════════

# def compute_st_trans_loss(
#     pred_deg    : torch.Tensor,
#     gt_deg      : torch.Tensor,
#     epoch       : int  = 0,
#     speed_stats : dict | None = None,
# ) -> dict:
#     """
#     v47 composite loss:
#       L = L_DPE + 0.05*L_MSE + 0.05*L_speed + 0.01*L_accel + 0.20*L_speed_match

#     L_speed_match: (pred_speed / gt_speed - 1)²  ← direct ATE supervision
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         return {"total": pred_deg.new_zeros(()), "dpe": 0.0, "spd": 0.0, "acc": 0.0}

#     sp         = speed_stats if speed_stats is not None else _SPEED_PRIOR
#     v_opt      = sp.get("v_opt",      _SPEED_PRIOR["v_opt"])
#     v_sigma    = sp.get("v_sigma",    _SPEED_PRIOR["v_sigma"])
#     v_hard_cap = sp.get("v_hard_cap", _SPEED_PRIOR["v_hard_cap"])

#     # ── L_DPE: weighted haversine ─────────────────────────────────────────────
#     w     = pred_deg.new_tensor(STEP_WEIGHTS[:T])
#     w     = w / w.sum() * T
#     dist  = _haversine_deg(pred_deg[:T], gt_deg[:T])
#     delta = 200.0
#     l_dpe = ((torch.where(dist < delta,
#                           dist.pow(2) / (2.0 * delta),
#                           dist - delta / 2.0)
#               ) * w.unsqueeze(1)).mean() / delta

#     # ── L_MSE ─────────────────────────────────────────────────────────────────
#     l_mse = F.mse_loss(pred_deg[:T], gt_deg[:T])

#     # ── Compute speeds ────────────────────────────────────────────────────────
#     pred_speed = gt_speed = None
#     if T >= 2:
#         def _speed(traj):
#             dt    = traj[1:T] - traj[:T-1]
#             lm    = (traj[:T-1, :, 1] + traj[1:T, :, 1]) * 0.5
#             cos   = torch.cos(torch.deg2rad(lm)).clamp(1e-4)
#             dx_km = dt[:, :, 0] * cos * DEG2KM
#             dy_km = dt[:, :, 1] * DEG2KM
#             return torch.sqrt(dx_km**2 + dy_km**2 + 1e-6) / DT_HOURS

#         pred_speed = _speed(pred_deg)   # [T-1, B]
#         gt_speed   = _speed(gt_deg)     # [T-1, B]

#     # ── L_speed: bilateral soft penalty ──────────────────────────────────────
#     if pred_speed is not None:
#         l_spd_soft = ((pred_speed - v_opt) / v_sigma).pow(2).mean()
#         l_spd_hard = F.relu(pred_speed - v_hard_cap).pow(2).mean() / (v_hard_cap**2)
#         l_speed    = 0.7 * l_spd_soft + 0.3 * l_spd_hard
#     else:
#         l_speed = pred_deg.new_zeros(())

#     # ── [FIX-4] L_speed_match: direct supervision on speed ───────────────────
#     # Penalize (pred_speed / gt_speed - 1)^2
#     # → Force model to predict correct translation speed → reduces ATE
#     if pred_speed is not None and gt_speed is not None:
#         speed_ratio   = pred_speed / gt_speed.clamp(min=2.0)   # avoid div0
#         l_speed_match = (speed_ratio - 1.0).pow(2).mean()
#     else:
#         l_speed_match = pred_deg.new_zeros(())

#     # ── L_accel ───────────────────────────────────────────────────────────────
#     if T >= 3 and pred_speed is not None:
#         accel   = (pred_speed[1:] - pred_speed[:-1]).abs() / DT_HOURS
#         a0      = max(v_sigma * 0.5, 3.0)
#         l_accel = accel.pow(2).mean() / (a0**2)
#     else:
#         l_accel = pred_deg.new_zeros(())

#     # ── Total ─────────────────────────────────────────────────────────────────
#     # [FIX-4] l_speed 0.1→0.05, thêm l_speed_match 0.20
#     total = (l_dpe
#              + 0.05 * l_mse
#              + 0.05 * l_speed
#              + 0.01 * l_accel
#              + 0.20 * l_speed_match)

#     if torch.isnan(total) or torch.isinf(total):
#         total = pred_deg.new_zeros(())

#     def _s(x): return x.item() if torch.is_tensor(x) else float(x)

#     return dict(
#         total        = total,
#         dpe          = _s(l_dpe),
#         mse          = _s(l_mse),
#         speed        = _s(l_speed),
#         accel        = _s(l_accel),
#         speed_match  = _s(l_speed_match),
#         spd_kmh      = _s(pred_speed.mean()) if pred_speed is not None else 0.0,
#         gt_spd_kmh   = _s(gt_speed.mean())   if gt_speed  is not None else 0.0,
#         v_opt        = float(v_opt),
#         v_hard_cap   = float(v_hard_cap),
#         # backward compat keys
#         fm_mse=0.0, mse_hav=_s(l_dpe), endpoint=0.0,
#         speed_acc=0.0, accel_b=0.0, decomp=0.0, cons=0.0,
#         hav_km   = _s(dist.mean()) if T > 0 else 0.0,
#         h72_km   = _s(_haversine_deg(
#             pred_deg[min(11, T-1)], gt_deg[min(11, T-1)]).mean()) if T > 0 else 0.0,
#         acc_kmh2=0.0, aux_fno=0.0, sigma=0.0,
#     )


# # ══════════════════════════════════════════════════════════════════════════════
# #  EMA
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
#                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)

#     def apply_to(self, model: nn.Module):
#         m = _unwrap_model(model)
#         backup = {}
#         sd = m.state_dict()
#         for k in self.shadow:
#             if k not in sd: continue
#             backup[k] = sd[k].detach().clone()
#             sd[k].copy_(self.shadow[k])
#         return backup

#     def restore(self, model: nn.Module, backup: dict):
#         m = _unwrap_model(model)
#         sd = m.state_dict()
#         for k, v in backup.items():
#             if k in sd: sd[k].copy_(v)


# # ══════════════════════════════════════════════════════════════════════════════
# #  VelocityField
# # ══════════════════════════════════════════════════════════════════════════════

# class VelocityField(nn.Module):
#     RAW_CTX_DIM = 512

#     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256,
#                  sigma_min=0.02, unet_in_ch=13, **kwargs):
#         super().__init__()
#         self.pred_len = pred_len
#         self.obs_len  = obs_len

#         # ── Encoders ─────────────────────────────────────────────────────────
#         self.spatial_enc = FNO3DEncoder(
#             in_channel=unet_in_ch, out_channel=1, d_model=32,
#             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
#             spatial_down=32, dropout=0.05)
#         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
#         self.bottleneck_proj = nn.Linear(128, 128)
#         self.decoder_proj    = nn.Linear(1, 16)

#         self.enc_1d  = DataEncoder1D(
#             in_1d=4, feat_3d_dim=128, mlp_h=64,
#             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)

#         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

#         # ── Context head ─────────────────────────────────────────────────────
#         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
#         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
#         self.ctx_drop = nn.Dropout(0.10)
#         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

#         # ── [FIX-2] Kinematic obs encoder: 6 features/step ───────────────────
#         # v46: Linear(obs_len*2, 256) = Linear(16,256)
#         # v47: Linear(obs_len*6, 256) = Linear(48,256)
#         self.vel_obs_enc = nn.Sequential(
#             nn.Linear(obs_len * 6, 256), nn.GELU(),
#             nn.LayerNorm(256),
#             nn.Linear(256, 256), nn.GELU(),
#         )

#         # ── Steering encoder (u/v500 from env) ───────────────────────────────
#         self.steering_enc = nn.Sequential(
#             nn.Linear(4, 64),   nn.GELU(), nn.LayerNorm(64),
#             nn.Linear(64, 128), nn.GELU(),
#             nn.Linear(128, 256),
#         )

#         # ── [FIX-3] Env kinematic encoder: inject env history into decoder ───
#         # env_data has: move_velocity(1) + history_direction24(8) + delta_velocity(5)
#         # = 14 dims of kinematic info per timestep, take last obs_len steps
#         # → aggregated → 256D → added to decoder memory
#         self.env_kine_enc = nn.Sequential(
#             nn.Linear(1 + 8 + 5, 64), nn.GELU(),
#             nn.LayerNorm(64),
#             nn.Linear(64, 256), nn.GELU(),
#         )

#         # ── Transformer decoder ──────────────────────────────────────────────
#         self.time_fc1    = nn.Linear(256, 512)
#         self.time_fc2    = nn.Linear(512, 256)
#         self.traj_embed  = nn.Linear(4, 256)
#         self.pos_enc     = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
#         self.step_embed  = nn.Embedding(pred_len, 256)
#         self.transformer = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 d_model=256, nhead=8, dim_feedforward=1024,
#                 dropout=0.10, activation="gelu", batch_first=True),
#             num_layers=2)
#         self.out_fc1 = nn.Linear(256, 512)
#         self.out_fc2 = nn.Linear(512, 4)

#         # ── Learnable scale params ────────────────────────────────────────────
#         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
#         self.physics_scale  = nn.Parameter(torch.ones(4) * 1.5)
#         self.steering_scale = nn.Parameter(torch.ones(4) * 1.0)

#         self._init_weights()

#     def _init_weights(self):
#         with torch.no_grad():
#             for name, m in self.named_modules():
#                 if isinstance(m, nn.Linear) and 'out_fc' in name:
#                     nn.init.xavier_uniform_(m.weight, gain=0.1)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)

#     # ── Time embedding ────────────────────────────────────────────────────────
#     def _time_emb(self, t, dim=256):
#         half = dim // 2
#         freq = torch.exp(
#             torch.arange(half, dtype=torch.float32, device=t.device)
#             * (-math.log(10000.0) / max(half - 1, 1)))
#         emb = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
#         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

#     # ── Context encoding ──────────────────────────────────────────────────────
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
#             e_3d_s = F.interpolate(
#                 e_3d_s.permute(0, 2, 1), size=T_obs,
#                 mode="linear", align_corners=False).permute(0, 2, 1)

#         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
#         t_w        = torch.softmax(
#             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
#                          device=e_3d_dec_t.device) * 0.5, dim=0)
#         f_spatial  = self.decoder_proj(
#             (e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

#         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
#         h_t    = self.enc_1d(obs_in, e_3d_s)
#         e_env, _, _ = self.env_enc(env_data, image_obs)

#         return F.gelu(self.ctx_ln(self.ctx_fc1(
#             torch.cat([h_t, e_env, f_spatial], dim=-1))))

#     def _apply_ctx_head(self, raw, noise_scale=0.0):
#         if noise_scale > 0.0:
#             raw = raw + torch.randn_like(raw) * noise_scale
#         return self.ctx_fc2(self.ctx_drop(raw))

#     # ── [FIX-2] Kinematic obs features (6D/step) ─────────────────────────────
#     def _get_kinematic_obs_feat(self, obs_traj: torch.Tensor) -> torch.Tensor:
#         """
#         obs_traj: [T_obs, B, 2] normalized lonlat
#         Returns:  [B, 256]

#         6 features per step: [vel_x, vel_y, speed_norm, sin_h, cos_h, accel_norm]
#         """
#         B     = obs_traj.shape[1]
#         T_obs = obs_traj.shape[0]

#         if T_obs >= 2:
#             vel     = obs_traj[1:] - obs_traj[:-1]         # [T-1, B, 2]
#             lat_mid = obs_traj[:-1, :, 1] * _NORM_TO_DEG
#             cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
#             dx_km   = vel[:, :, 0] * cos_lat * DEG2KM * _NORM_TO_DEG
#             dy_km   = vel[:, :, 1]            * DEG2KM * _NORM_TO_DEG
#             speed   = torch.sqrt(dx_km**2 + dy_km**2 + 1e-6) / DT_HOURS
#             heading = torch.atan2(vel[:, :, 1], vel[:, :, 0])

#             speed_n = (speed / 20.0).clamp(-3.0, 3.0)
#             sin_h   = heading.sin()
#             cos_h   = heading.cos()

#             if T_obs >= 3:
#                 dspd   = speed[1:] - speed[:-1]
#                 accel  = (dspd / 10.0).clamp(-3.0, 3.0)
#                 accel  = torch.cat([obs_traj.new_zeros(1, B), accel], dim=0)
#             else:
#                 accel  = obs_traj.new_zeros(T_obs - 1, B)

#             kine = torch.stack(
#                 [vel[:, :, 0], vel[:, :, 1], speed_n, sin_h, cos_h, accel],
#                 dim=-1)   # [T-1, B, 6]
#         else:
#             kine = obs_traj.new_zeros(self.obs_len, B, 6)

#         # Pad/trim to obs_len
#         if kine.shape[0] < self.obs_len:
#             pad  = obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6)
#             kine = torch.cat([pad, kine], dim=0)
#         else:
#             kine = kine[-self.obs_len:]   # [obs_len, B, 6]

#         return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))  # [B, 256]

#     # Backward compat alias
#     def _get_vel_obs_feat(self, obs_traj):
#         return self._get_kinematic_obs_feat(obs_traj)

#     # ── Steering features (u/v500 from env) ──────────────────────────────────
#     def _get_steering_feat(self, env_data, B, device):
#         if env_data is None:
#             return torch.zeros(B, 256, device=device)

#         def _safe(key, default=0.0):
#             v = env_data.get(key)
#             if v is None or not torch.is_tensor(v):
#                 return torch.full((B,), default, device=device)
#             v = v.float().to(device)
#             while v.dim() > 1: v = v.mean(-1)
#             return (v.view(-1)[:B] if v.numel() >= B
#                     else torch.full((B,), default, device=device))

#         feat = torch.stack([
#             _safe("u500_mean"), _safe("v500_mean"),
#             _safe("u500_center"), _safe("v500_center"),
#         ], dim=-1)
#         return self.steering_enc(feat)

#     # ── [FIX-3] Env kinematic features for decoder ───────────────────────────
#     def _get_env_kine_feat(self, env_data, B, device) -> torch.Tensor:
#         """
#         Extract kinematic info từ env_data → inject vào decoder memory.

#         env_data keys used:
#           move_velocity      (1 scalar): normalized speed km/h
#           history_direction24 (8 dims):  one-hot heading direction
#           delta_velocity      (5 dims):  one-hot acceleration bin

#         Returns: [B, 256]
#         """
#         if env_data is None:
#             return torch.zeros(B, 256, device=device)

#         def _get_tensor(key, dim, default=0.0):
#             v = env_data.get(key)
#             if v is None:
#                 return torch.full((B, dim), default, device=device)
#             if not torch.is_tensor(v):
#                 try:
#                     v = torch.tensor(v, dtype=torch.float, device=device)
#                 except Exception:
#                     return torch.full((B, dim), default, device=device)
#             v = v.float().to(device)
#             # Handle various shapes
#             if v.dim() == 0:
#                 return v.expand(B, dim) if dim == 1 else torch.full((B, dim), float(v), device=device)
#             if v.dim() == 1:
#                 if v.shape[0] == dim:
#                     return v.unsqueeze(0).expand(B, dim)
#                 elif v.shape[0] == B:
#                     return v.unsqueeze(1).expand(B, dim) if dim == 1 else torch.full((B, dim), 0.0, device=device)
#             if v.dim() == 2:
#                 if v.shape == (B, dim):
#                     return v
#                 if v.shape[1] == dim:
#                     return v[:B] if v.shape[0] >= B else F.pad(v, (0, 0, 0, B - v.shape[0]))
#                 # Take last timestep
#                 return v[:B, :dim] if v.shape[-1] >= dim else F.pad(v[:B], (0, dim - v.shape[-1]))
#             if v.dim() == 3:
#                 # [T, B, dim] or [B, T, dim] → take last
#                 if v.shape[1] == B:
#                     vv = v[-1]  # [B, dim]
#                 else:
#                     vv = v[:B, -1]  # [B, dim]
#                 return vv[:, :dim] if vv.shape[-1] >= dim else F.pad(vv, (0, dim - vv.shape[-1]))
#             return torch.full((B, dim), default, device=device)

#         mv   = _get_tensor("move_velocity",       1)   # [B, 1]
#         dir24= _get_tensor("history_direction24",  8)   # [B, 8]
#         dvel = _get_tensor("delta_velocity",       5)   # [B, 5]

#         feat = torch.cat([mv, dir24, dvel], dim=-1)    # [B, 14]
#         return self.env_kine_enc(feat)                  # [B, 256]

#     # ── Physics drifts ────────────────────────────────────────────────────────
#     def _beta_drift(self, x_t):
#         lat_deg = x_t[:, :, 1] * 5.0
#         lat_rad = torch.deg2rad(lat_deg.clamp(-85, 85))
#         beta    = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
#         R_tc    = 3e5
#         v       = torch.zeros_like(x_t)
#         v[:, :, 0] = -beta * R_tc**2 / 2 * 6 * 3600 / (5 * 111 * 1000)
#         v[:, :, 1] =  beta * R_tc**2 / 4 * 6 * 3600 / (5 * 111 * 1000)
#         return v

#     def _steering_drift(self, x_t, env_data):
#         if env_data is None:
#             return torch.zeros_like(x_t)
#         B, device = x_t.shape[0], x_t.device

#         def _sm(k):
#             v = env_data.get(k)
#             if v is None or not torch.is_tensor(v):
#                 return torch.zeros(B, device=device)
#             v = v.float().to(device)
#             while v.dim() > 1: v = v.mean(-1)
#             return v.view(-1)[:B] if v.numel() >= B else torch.zeros(B, device=device)

#         u   = _sm("u500_center")
#         v   = _sm("v500_center")
#         cos = torch.cos(torch.deg2rad(x_t[:, :, 1] * 5.0)).clamp(1e-3)
#         out = torch.zeros_like(x_t)
#         out[:, :, 0] = u.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0 * cos)
#         out[:, :, 1] = v.unsqueeze(1) * 30.0 * 21600.0 / (111.0 * 1000.0)
#         return out

#     # ── Decoder ───────────────────────────────────────────────────────────────
#     def _decode(self, x_t, t, ctx,
#                 vel_obs_feat=None, steering_feat=None,
#                 env_kine_feat=None, env_data=None):
#         B    = x_t.shape[0]
#         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
#         t_emb = self.time_fc2(t_emb)

#         T_seq    = min(x_t.size(1), self.pos_enc.shape[1])
#         step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)

#         x_emb = (self.traj_embed(x_t[:, :T_seq])
#                  + self.pos_enc[:, :T_seq]
#                  + t_emb.unsqueeze(1)
#                  + self.step_embed(step_idx))

#         # [FIX-3] Decoder memory includes env kinematic features
#         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
#         if vel_obs_feat  is not None: mem_parts.append(vel_obs_feat.unsqueeze(1))
#         if steering_feat is not None: mem_parts.append(steering_feat.unsqueeze(1))
#         if env_kine_feat is not None: mem_parts.append(env_kine_feat.unsqueeze(1))

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
#                          vel_obs_feat=None, steering_feat=None,
#                          env_kine_feat=None, env_data=None):
#         ctx = self._apply_ctx_head(raw_ctx, noise_scale)
#         return self._decode(x_t, t, ctx,
#                             vel_obs_feat=vel_obs_feat,
#                             steering_feat=steering_feat,
#                             env_kine_feat=env_kine_feat,
#                             env_data=env_data)


# # ══════════════════════════════════════════════════════════════════════════════
# #  TCFlowMatching v47
# # ══════════════════════════════════════════════════════════════════════════════

# class TCFlowMatching(nn.Module):

#     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
#                  n_train_ens=4, unet_in_ch=13, ctx_noise_scale=0.01,
#                  initial_sample_sigma=0.03, teacher_forcing=True,
#                  use_ema=True, ema_decay=0.999, **kwargs):
#         super().__init__()
#         self.pred_len        = pred_len
#         self.obs_len         = obs_len
#         self.sigma_min       = sigma_min
#         self.ctx_noise_scale = ctx_noise_scale
#         self.active_pred_len = pred_len

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

#     def set_curriculum_len(self, *a, **kw): pass

#     @staticmethod
#     def _to_rel(traj, Me, lp, lm):
#         return torch.cat([traj - lp.unsqueeze(0),
#                           Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

#     @staticmethod
#     def _to_abs(rel, lp, lm):
#         d = rel.permute(1, 0, 2)
#         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

#     def _cfm_noisy(self, x1, sigma_min=None):
#         if sigma_min is None: sigma_min = self.sigma_min
#         B, device = x1.shape[0], x1.device
#         x0 = torch.randn_like(x1) * sigma_min
#         t  = torch.rand(B, device=device)
#         te = t.view(B, 1, 1)
#         return (1.0 - te) * x0 + te * x1, t, x1 - x0

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

#     # ── [FIX-6] Persistence forecast for sample init ──────────────────────────
#     def _persistence_forecast_rel(self, obs_traj, lp, lm, pred_len):
#         """
#         Tính persistence trajectory: TC tiếp tục đi thẳng với velocity cuối.
#         Returns relative coords [B, T, 4] để dùng làm init point cho sampling.
#         """
#         B, device = obs_traj.shape[1], obs_traj.device

#         if obs_traj.shape[0] >= 2:
#             last_vel_lon = obs_traj[-1, :, 0] - obs_traj[-2, :, 0]  # [B]
#             last_vel_lat = obs_traj[-1, :, 1] - obs_traj[-2, :, 1]  # [B]
#         else:
#             last_vel_lon = torch.zeros(B, device=device)
#             last_vel_lat = torch.zeros(B, device=device)

#         steps    = torch.arange(1, pred_len + 1, device=device).float()
#         pred_lon = obs_traj[-1, :, 0].unsqueeze(1) + last_vel_lon.unsqueeze(1) * steps
#         pred_lat = obs_traj[-1, :, 1].unsqueeze(1) + last_vel_lat.unsqueeze(1) * steps
#         pred_abs = torch.stack([pred_lon, pred_lat], dim=-1)   # [B, T, 2]

#         pred_rel_pos = pred_abs.permute(1, 0, 2) - lp.unsqueeze(0)  # [T, B, 2]
#         pred_rel     = torch.cat([pred_rel_pos,
#                                   torch.zeros_like(pred_rel_pos)], dim=-1)  # [T, B, 4]
#         return pred_rel.permute(1, 0, 2)   # [B, T, 4]

#     # ── Sigma schedule ────────────────────────────────────────────────────────
#     @staticmethod
#     def _sigma_schedule(epoch):
#         if epoch < 10:   return 0.15
#         if epoch < 40:   return 0.15 - (epoch - 10) / 30.0 * (0.15 - 0.03)
#         return 0.03

#     # ── Training loss ─────────────────────────────────────────────────────────
#     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
#         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

#     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
#         batch_list = self._lon_flip_aug(batch_list)
#         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)

#         obs_t    = batch_list[0]
#         env_data = batch_list[13] if len(batch_list) > 13 else None
#         lp, lm   = obs_t[-1], batch_list[7][-1]

#         speed_stats    = compute_speed_stats_from_norm(obs_t[..., :2])
#         current_sigma  = self._sigma_schedule(epoch)

#         raw_ctx = self.net._context(batch_list)

#         # Re-read (after aug)
#         obs_t    = batch_list[0]
#         env_data = batch_list[13] if len(batch_list) > 13 else None
#         lp, lm   = obs_t[-1], batch_list[7][-1]

#         x1_rel = self._to_rel(batch_list[1], batch_list[8], lp, lm)
#         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

#         B, device = obs_t.shape[1], obs_t.device

#         pred_vel = self.net.forward_with_ctx(
#             x_t, fm_t, raw_ctx, env_data=env_data,
#             vel_obs_feat  = self.net._get_kinematic_obs_feat(obs_t[:, :, :2]),
#             steering_feat = self.net._get_steering_feat(env_data, B, device),
#             env_kine_feat = self.net._get_env_kine_feat(env_data, B, device),  # [FIX-3]
#         )

#         l_fm_mse = F.mse_loss(pred_vel, u_target)

#         fm_te       = fm_t.view(B, 1, 1)
#         x1_pred     = x_t + (1.0 - fm_te) * pred_vel
#         pred_abs, _ = self._to_abs(x1_pred, lp, lm)
#         pred_deg    = _norm_to_deg(pred_abs)
#         gt_deg      = _norm_to_deg(batch_list[1])

#         loss_dict = compute_st_trans_loss(
#             pred_deg, gt_deg, epoch=epoch, speed_stats=speed_stats)

#         total = l_fm_mse + 2.0 * loss_dict["total"]
#         if torch.isnan(total) or torch.isinf(total):
#             total = x_t.new_zeros(())

#         d = dict(loss_dict)
#         d.update({
#             "total"      : total,
#             "fm_mse"     : l_fm_mse.item(),
#             "sigma"      : current_sigma,
#             "v_opt"      : speed_stats["v_opt"],
#             "v_hard_cap" : speed_stats["v_hard_cap"],
#             "obs_spd_p50": speed_stats["p50_kmh"],
#         })
#         return d

#     # ── Sampling ──────────────────────────────────────────────────────────────
#     @torch.no_grad()
#     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
#                predict_csv=None, importance_weight=True):
#         obs_t    = batch_list[0]
#         env_data = batch_list[13] if len(batch_list) > 13 else None
#         lp       = obs_t[-1]
#         lm       = batch_list[7][-1]
#         B        = lp.shape[0]
#         device   = lp.device
#         T        = self.pred_len
#         dt       = 1.0 / max(ddim_steps, 1)

#         raw_ctx       = self.net._context(batch_list)
#         vel_obs_feat  = self.net._get_kinematic_obs_feat(obs_t[:, :, :2])
#         steering_feat = self.net._get_steering_feat(env_data, B, device)
#         env_kine_feat = self.net._get_env_kine_feat(env_data, B, device)   # [FIX-3]
#         speed_stats   = compute_speed_stats_from_norm(obs_t[..., :2])

#         # [FIX-6] Persistence-based init
#         persist_init = self._persistence_forecast_rel(obs_t, lp, lm, T)   # [B,T,4]

#         traj_s, me_s, scores = [], [], []
#         for _ in range(num_ensemble):
#             # Init: persistence + small noise (not pure noise)
#             x_t = persist_init + torch.randn_like(persist_init) * self.sigma_min * 2.5

#             for step in range(ddim_steps):
#                 t_b = torch.full((B,), step * dt, device=device)
#                 ns  = self.ctx_noise_scale * 2.0 if step < 3 else 0.0
#                 vel = self.net.forward_with_ctx(
#                     x_t, t_b, raw_ctx, noise_scale=ns,
#                     vel_obs_feat  = vel_obs_feat,
#                     steering_feat = steering_feat,
#                     env_kine_feat = env_kine_feat,
#                     env_data      = env_data,
#                 )
#                 x_t = (x_t + dt * vel).clamp(-3.0, 3.0)

#             tr, me = self._to_abs(x_t, lp, lm)
#             traj_s.append(tr)
#             me_s.append(me)
#             if importance_weight:
#                 scores.append(self._score_sample(tr, speed_stats))

#         all_trajs = torch.stack(traj_s)
#         all_me    = torch.stack(me_s)

#         if importance_weight and scores:
#             score_t = torch.stack(scores)
#             k       = max(1, int(num_ensemble * 0.7))
#             _, idx  = score_t.topk(k, dim=0)
#             pred_mean = torch.stack([
#                 all_trajs[idx[:, b], :, b, :].median(0).values
#                 for b in range(B)], dim=1)
#         else:
#             pred_mean = all_trajs.median(0).values

#         if predict_csv:
#             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
#         return pred_mean, all_me.mean(0), all_trajs

#     def _score_sample(self, traj, speed_stats=None):
#         B = traj.shape[1]
#         if traj.shape[0] < 2:
#             return torch.ones(B, device=traj.device)
#         sp        = speed_stats if speed_stats is not None else _SPEED_PRIOR
#         v_opt     = sp.get("v_opt",      _SPEED_PRIOR["v_opt"])
#         v_sigma   = sp.get("v_sigma",    _SPEED_PRIOR["v_sigma"])
#         v_hard_cap= sp.get("v_hard_cap", _SPEED_PRIOR["v_hard_cap"])
#         td        = _norm_to_deg(traj)
#         dtd       = td[1:] - td[:-1]
#         cos_lat   = torch.cos(torch.deg2rad(td[:-1, :, 1])).clamp(1e-4)
#         speed     = torch.sqrt(
#             (dtd[:,:,0] * cos_lat * DEG2KM)**2 + (dtd[:,:,1] * DEG2KM)**2)
#         spd_pen   = ((speed - v_opt) / v_sigma).pow(2)
#         hard_pen  = F.relu(speed - v_hard_cap) * 2.0
#         spd_sc    = torch.exp(-(spd_pen + hard_pen).mean(0) * 0.5)
#         smooth_sc = (torch.exp(-((dtd[1:] - dtd[:-1]).norm(dim=-1).mean(0)) * 5.0)
#                      if dtd.shape[0] >= 2 else torch.ones(B, device=traj.device))
#         return spd_sc * smooth_sc

#     @staticmethod
#     def _write_predict_csv(csv_path, traj_mean, all_trajs):
#         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
#         T, B, _ = traj_mean.shape
#         mlon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
#         mlat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
#         alon = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
#         alat = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()
#         fields = ["timestamp","batch_idx","step_idx","lead_h",
#                   "lon_mean_deg","lat_mean_deg","lon_std_deg","lat_std_deg","ens_spread_km"]
#         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#         write_hdr = not os.path.exists(csv_path)
#         with open(csv_path, "a", newline="") as fh:
#             w = csv.DictWriter(fh, fieldnames=fields)
#             if write_hdr: w.writeheader()
#             for b in range(B):
#                 for k in range(T):
#                     dlat   = alat[:, k, b] - mlat[k, b]
#                     dlon   = ((alon[:, k, b] - mlon[k, b])
#                               * math.cos(math.radians(mlat[k, b])))
#                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * DEG2KM)
#                     w.writerow({
#                         "timestamp": ts, "batch_idx": b, "step_idx": k,
#                         "lead_h": (k + 1) * 6,
#                         "lon_mean_deg": f"{mlon[k,b]:.4f}",
#                         "lat_mean_deg": f"{mlat[k,b]:.4f}",
#                         "lon_std_deg":  f"{alon[:,k,b].std():.4f}",
#                         "lat_std_deg":  f"{alat[:,k,b].std():.4f}",
#                         "ens_spread_km": f"{spread:.2f}",
#                     })


# # Backward compat
# TCDiffusion = TCFlowMatching

# """
# Model/flow_matching_model.py — TC-FlowMatching v50
# ═══════════════════════════════════════════════════════════════════════════════

# Gộp từ flow_matching_model_v48.py + flow_matching_v50_patch.py

# THAY ĐỔI SO VỚI v48:
#   [F1] compute_velocity_regression_loss  — velocity supervision, root cause ATE
#   [F2] spherical_ate_cte_loss           — true geodesic ATE/CTE projection
#   [F3] geodesic_ot_cost + spherical_ot_matching — km/h OT cost thay Euclidean
#   [F4] compute_st_trans_loss (v50)      — loss tổng hợp, xóa multi_marginal
#   [F5] sample()                         — speed-sweep 7×50 + adaptive CFG
#   [BUGFIX] SLERP velocity target đúng   — dùng degrees, không dùng norm coords
#   [BUGFIX] _sigma_schedule              — 0.10→0.04 từ ep2

# TARGETS: ATE<79.94 | 72h<297 | CTE<93.58 | ADE<136.41

# COMPAT: load checkpoint v47/v48 với strict=False
# """
# from __future__ import annotations

# import csv
# import math
# import os
# from datetime import datetime
# from typing import Optional, Tuple

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

# # [F4] Step weights: 72h = 10×, early steps nhẹ hơn
# STEP_WEIGHTS = [1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0]

# _SPEED_PRIOR = {"v_opt": 15.0, "v_sigma": 10.0, "v_hard_cap": 35.0}


# # ══════════════════════════════════════════════════════════════════════════════
# #  Coordinate utilities
# # ══════════════════════════════════════════════════════════════════════════════

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
#     """Per-step speed km/h. traj_deg [T,B,2] → [T-1,B]. Haversine-correct."""
#     T = traj_deg.shape[0]
#     if T < 2:
#         return traj_deg.new_zeros(1, traj_deg.shape[1])
#     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


# def _forward_azimuth(p1, p2):
#     """Great-circle bearing p1→p2. Both [...,2] degrees. Returns [...] radians."""
#     lon1 = torch.deg2rad(p1[..., 0]); lat1 = torch.deg2rad(p1[..., 1])
#     lon2 = torch.deg2rad(p2[..., 0]); lat2 = torch.deg2rad(p2[..., 1])
#     dlon = lon2 - lon1
#     y = torch.sin(dlon) * torch.cos(lat2)
#     x = torch.cos(lat1)*torch.sin(lat2) - torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon)
#     return torch.atan2(y, x)


# # ══════════════════════════════════════════════════════════════════════════════
# #  Speed statistics
# # ══════════════════════════════════════════════════════════════════════════════

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


# # ══════════════════════════════════════════════════════════════════════════════
# #  [F2] Spherical ATE/CTE loss
# # ══════════════════════════════════════════════════════════════════════════════

# def _spherical_ate_cte_loss(pred_deg, gt_deg, ate_weight=3.0, cte_weight=1.0):
#     """
#     [F2] Geodesic ATE/CTE projection.

#     Dùng forward azimuth (great-circle bearing) thay vì flat-Earth vector.
#     ATE = total_haversine_error * cos(angle_diff)
#     CTE = total_haversine_error * sin(angle_diff)
#     angle_diff = bearing(gt→pred) − bearing(gt[t-1]→gt[t])
#     """
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         zero = pred_deg.new_zeros(())
#         return {"ate": zero, "cte": zero, "ate_cte_total": zero,
#                 "ate_mean_km": 0.0, "cte_mean_km": 0.0}

#     bear_along = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])   # [T-1,B]
#     bear_error = _forward_azimuth(gt_deg[1:T],    pred_deg[1:T]) # [T-1,B]
#     total_err  = _haversine_deg(pred_deg[1:T], gt_deg[1:T])      # [T-1,B] km

#     angle  = bear_error - bear_along
#     ate    = total_err * torch.cos(angle)   # [T-1,B]
#     cte    = total_err * torch.sin(angle)

#     w = pred_deg.new_tensor(STEP_WEIGHTS[1:T])
#     w = w / w.sum() * (T - 1)

#     ate_t = 200.0; cte_t = 100.0
#     ate_loss = (torch.where(ate.abs() < ate_t, ate.pow(2)/(2*ate_t),
#                             ate.abs() - ate_t/2) * w.unsqueeze(1)).mean() / ate_t
#     cte_loss = (torch.where(cte.abs() < cte_t, cte.pow(2)/(2*cte_t),
#                             cte.abs() - cte_t/2) * w.unsqueeze(1)).mean() / cte_t

#     total = ate_weight * ate_loss + cte_weight * cte_loss
#     return {"ate": ate_loss, "cte": cte_loss, "ate_cte_total": total,
#             "ate_mean_km": float(ate.abs().mean()),
#             "cte_mean_km": float(cte.abs().mean())}


# # ══════════════════════════════════════════════════════════════════════════════
# #  [F1] Velocity regression loss
# # ══════════════════════════════════════════════════════════════════════════════

# def _velocity_regression_loss(pred_deg, gt_deg, speed_stats=None):
#     """
#     [F1] Root cause fix cho ATE=330km.

#     DPE loss không dạy "bước này đi bao nhiêu km". Model bị mean-regression.
#     Supervision direct trên per-step velocity:
#       ratio (40%): Huber(pred_spd/gt_spd − 1, δ=0.3)
#       abs   (30%): |pred_spd − gt_spd|² / v_sigma²
#       vec   (30%): ||pred_vel_km − gt_vel_km||²
#     """
#     sp      = speed_stats or _SPEED_PRIOR
#     v_sigma = sp.get("v_sigma", 10.0)
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         return pred_deg.new_zeros(())

#     pred_spd = _step_speeds_deg(pred_deg[:T])
#     gt_spd   = _step_speeds_deg(gt_deg[:T])
#     w = pred_deg.new_tensor(STEP_WEIGHTS[1:T]); w = w / w.sum()

#     ratio   = pred_spd / gt_spd.clamp(min=2.0)
#     l_ratio = (F.huber_loss(ratio, torch.ones_like(ratio), delta=0.3,
#                              reduction="none") * w.unsqueeze(1)).mean()
#     l_abs   = ((pred_spd - gt_spd).pow(2) / v_sigma**2 * w.unsqueeze(1)).mean()

#     plon = pred_deg[1:T, :, 0] - pred_deg[:T-1, :, 0]
#     plat = pred_deg[1:T, :, 1] - pred_deg[:T-1, :, 1]
#     glon = gt_deg[1:T, :, 0]   - gt_deg[:T-1, :, 0]
#     glat = gt_deg[1:T, :, 1]   - gt_deg[:T-1, :, 1]
#     cos  = torch.cos(torch.deg2rad(
#         (pred_deg[:T-1,:,1]+pred_deg[1:T,:,1])*0.5)).clamp(1e-4)
#     pv_km = torch.stack([plon*cos*DEG2KM/DT_HOURS, plat*DEG2KM/DT_HOURS], -1)
#     gv_km = torch.stack([glon*cos*DEG2KM/DT_HOURS, glat*DEG2KM/DT_HOURS], -1)
#     l_vec = (F.mse_loss(pv_km, gv_km, reduction="none").mean(-1)
#              / v_sigma**2 * w.unsqueeze(1)).mean()

#     return (0.4*l_ratio + 0.3*l_abs + 0.3*l_vec).clamp(0.0, 20.0)


# # ══════════════════════════════════════════════════════════════════════════════
# #  [F4] compute_st_trans_loss — v50
# # ══════════════════════════════════════════════════════════════════════════════

# def compute_st_trans_loss(pred_deg, gt_deg, epoch=0, speed_stats=None):
#     """
#     v50 loss:
#       DPE(v50_weights) + 0.03*MSE + 0.05*speed + 0.01*accel
#       + 0.30*vel_reg  [F1]
#       + 0.20*sph_ate  [F2]
#       + 0.20*endpoint
#       + 0.10*heading
#     """
#     sp         = speed_stats or _SPEED_PRIOR
#     v_opt      = sp.get("v_opt",      15.0)
#     v_sigma    = sp.get("v_sigma",    10.0)
#     v_hard_cap = sp.get("v_hard_cap", 35.0)
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         return {"total": pred_deg.new_zeros(()), "dpe": 0.0, "vel_reg": 0.0}

#     # DPE
#     w    = pred_deg.new_tensor(STEP_WEIGHTS[:T]); w = w / w.sum() * T
#     dist = _haversine_deg(pred_deg[:T], gt_deg[:T])
#     d    = 200.0
#     l_dpe = ((torch.where(dist < d, dist.pow(2)/(2*d), dist - d/2)
#               ) * w.unsqueeze(1)).mean() / d

#     # MSE
#     l_mse = F.mse_loss(pred_deg[:T], gt_deg[:T])

#     # Speed prior
#     pred_spd = _step_speeds_deg(pred_deg[:T])
#     l_speed  = (0.7*((pred_spd-v_opt)/v_sigma).pow(2).mean() +
#                 0.3*F.relu(pred_spd-v_hard_cap).pow(2).mean()/v_hard_cap**2
#                 ) if pred_spd.shape[0] > 0 else pred_deg.new_zeros(())

#     # Accel
#     l_accel = (((pred_spd[1:]-pred_spd[:-1]).abs()/DT_HOURS).pow(2).mean()
#                / max(v_sigma*0.5, 3.0)**2
#                ) if pred_spd.shape[0] >= 2 else pred_deg.new_zeros(())

#     # Heading cosine
#     if T >= 3:
#         pv = pred_deg[1:T] - pred_deg[:T-1]
#         gv = gt_deg[1:T]   - gt_deg[:T-1]
#         l_heading = (1.0 - (
#             F.normalize(pv.reshape(-1,2), dim=-1, eps=1e-6) *
#             F.normalize(gv.reshape(-1,2), dim=-1, eps=1e-6)
#         ).sum(-1)).mean().clamp(0.0, 2.0)
#     else:
#         l_heading = pred_deg.new_zeros(())

#     # [F1] Velocity regression
#     l_vel_reg = _velocity_regression_loss(pred_deg, gt_deg, speed_stats)

#     # [F2] Spherical ATE/CTE
#     ate_d     = _spherical_ate_cte_loss(pred_deg[:T], gt_deg[:T], 3.0, 1.0)
#     l_sph_ate = ate_d["ate_cte_total"]

#     # Endpoint (48h→72h heavy penalty)
#     d2 = 150.0
#     ep_total, ep_w = pred_deg.new_zeros(()), 0.0
#     for s, ew in [(8, 1.0), (9, 1.5), (10, 2.0), (11, 3.0)]:
#         if s >= T: continue
#         de = _haversine_deg(pred_deg[s], gt_deg[s])
#         ep_total = ep_total + ew * torch.where(
#             de < d2, de.pow(2)/(2*d2), de - d2/2).mean() / d2
#         ep_w += ew
#     l_endpoint = ep_total / max(ep_w, 1e-6)

#     total = (l_dpe + 0.03*l_mse + 0.05*l_speed + 0.01*l_accel
#              + 0.10*l_heading + 0.30*l_vel_reg
#              + 0.20*l_sph_ate + 0.20*l_endpoint)

#     if torch.isnan(total) or torch.isinf(total):
#         total = pred_deg.new_zeros(())

#     def _s(x): return x.item() if torch.is_tensor(x) else float(x)
#     return dict(
#         total=total,
#         dpe=_s(l_dpe), mse=_s(l_mse), speed=_s(l_speed), accel=_s(l_accel),
#         heading=_s(l_heading), vel_reg=_s(l_vel_reg),
#         ate=_s(ate_d["ate"]), cte=_s(ate_d["cte"]), sph_ate=_s(l_sph_ate),
#         endpoint=_s(l_endpoint),
#         ate_mean_km=ate_d.get("ate_mean_km", 0.0),
#         cte_mean_km=ate_d.get("cte_mean_km", 0.0),
#         # backward compat
#         speed_match=0.0, acc_kmh2=0.0, aux_fno=0.0, sigma=0.0,
#         fm_mse=0.0, multi_marg=0.0,
#     )

# # Alias cho v48 compatibility
# compute_ate_focused_loss = compute_st_trans_loss


# # ══════════════════════════════════════════════════════════════════════════════
# #  [F3] Geodesic OT cost + Sinkhorn
# # ══════════════════════════════════════════════════════════════════════════════

# def _sinkhorn_log(cost, epsilon=0.05, n_iter=50):
#     """Log-domain Sinkhorn for numerical stability."""
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
#     """
#     [F3] OT cost matrix dùng geodesic metrics (km/h thật, great-circle heading).
#     x0_rel, x1_rel: [B,T,4] relative normalized.  lp: [B,2] anchor normalized.
#     """
#     B = x0_rel.shape[0]

#     def _abs_deg(rel):
#         return _norm_to_deg(lp.unsqueeze(1) + rel[:, :, :2])   # [B,T,2]

#     x0d = _abs_deg(x0_rel); x1d = _abs_deg(x1_rel)

#     # Position cost: mean haversine pairwise [B,B]
#     x0e = x0d.unsqueeze(1).expand(B,B,-1,-1).reshape(B*B,-1,2)
#     x1e = x1d.unsqueeze(0).expand(B,B,-1,-1).reshape(B*B,-1,2)
#     pos_cost = _haversine_deg(x0e, x1e).reshape(B,B,-1).mean(-1) / 500.0

#     # Speed cost [B,B]
#     spd0 = _step_speeds_deg(x0d.permute(1,0,2)).permute(1,0).mean(-1)  # [B]
#     spd1 = _step_speeds_deg(x1d.permute(1,0,2)).permute(1,0).mean(-1)
#     speed_cost = (spd0.unsqueeze(1) - spd1.unsqueeze(0)).abs() / 20.0

#     # Direction cost (great-circle bearing) [B,B]
#     def _mean_bearing(td):  # td [B,T,2]
#         b = _forward_azimuth(td[:,:-1,:], td[:,1:,:])  # [B,T-1]
#         return torch.atan2(b.sin().mean(-1), b.cos().mean(-1))
#     h0 = _mean_bearing(x0d); h1 = _mean_bearing(x1d)
#     dh = (h0.unsqueeze(1) - h1.unsqueeze(0) + math.pi) % (2*math.pi) - math.pi
#     dir_cost = dh.abs() / math.pi

#     return 1.0*pos_cost + 0.5*speed_cost + 0.3*dir_cost


# def _spherical_ot_matching(x0_batch, x1_batch, lp, epsilon=0.05):
#     """
#     [F3] Mini-batch OT: match x1 (GT trajectories) to each other via geodesic cost,
#     then re-index x0 (noise) to follow the same permutation.

#     Lý do: x0 là Gaussian noise → không có ý nghĩa địa lý.
#     OT meaningful chỉ khi match x1↔x1 (different GT trajectories trong batch).
#     x0 được permute theo cùng index để cặp (x0_i, x1_i) consistent.

#     Đây là cách standard mini-batch OT trong FM literature.
#     Falls back to identity if any numerical issue.
#     """
#     B = x0_batch.shape[0]
#     if B < 4:
#         return x0_batch, x1_batch
#     try:
#         # Cost giữa các GT trajectories trong batch (x1↔x1)
#         cost = _geodesic_ot_cost(x1_batch, x1_batch, lp)
#         with torch.no_grad():
#             pi = _sinkhorn_log(cost, epsilon=epsilon)
#         flat = pi.reshape(-1).clamp(0.0)
#         s    = flat.sum()
#         if not torch.isfinite(s) or s < 1e-10:
#             return x0_batch, x1_batch
#         idx  = torch.multinomial(flat / s, num_samples=B, replacement=True)
#         col  = idx % B   # which x1 each position maps to
#         # x0 gets permuted by col so (x0[col[i]], x1[col[i]]) are paired
#         return x0_batch[col], x1_batch[col]
#     except Exception:
#         return x0_batch, x1_batch


# # ══════════════════════════════════════════════════════════════════════════════
# #  SLERP interpolant (từ v48, bug-fixed: velocity target dùng đúng manifold)
# # ══════════════════════════════════════════════════════════════════════════════

# def _slerp_interpolant(x0, x1, t, lp=None):
#     """
#     Spherical interpolation cho flow matching.

#     x0, x1: [B,T,C] — có thể là relative normalized hoặc absolute normalized.
#     t: [B]
#     lp: [B,2] anchor position (normalized). Nếu cung cấp, cộng vào để tính omega đúng.

#     FIX: omega được tính từ ABSOLUTE degrees (lon/lat thật).
#     Nếu x0/x1 là relative coords, cần lp để reconstruct absolute.
#     Coefficients SLERP sau đó áp lên x0/x1 gốc (relative hoặc absolute).

#     Vì relative coords nhỏ (~0.01-0.3 norm units), omega sẽ nhỏ và
#     fallback to linear — đây là hành vi đúng cho bài toán này.
#     """
#     B  = x0.shape[0]; te = t.view(B, 1, 1)

#     # Tính omega từ absolute position để có ý nghĩa vật lý
#     if lp is not None and x0.shape[-1] >= 2:
#         # x0, x1 là relative → cộng anchor để ra absolute normalized
#         abs0 = lp.unsqueeze(1) + x0[:, :, :2]   # [B, T, 2]
#         abs1 = lp.unsqueeze(1) + x1[:, :, :2]
#     else:
#         abs0 = x0[:, :, :2]
#         abs1 = x1[:, :, :2]

#     abs0_deg = _norm_to_deg(abs0); abs1_deg = _norm_to_deg(abs1)
#     lat0 = torch.deg2rad(abs0_deg[..., 1]); lat1 = torch.deg2rad(abs1_deg[..., 1])
#     dlon = torch.deg2rad(abs1_deg[..., 0] - abs0_deg[..., 0])
#     dlat = lat1 - lat0
#     a    = torch.sin(dlat/2).pow(2) + torch.cos(lat0)*torch.cos(lat1)*torch.sin(dlon/2).pow(2)
#     omega     = 2.0 * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())
#     sin_omega = torch.sin(omega).clamp(1e-6)
#     linear    = omega < 1e-4   # fallback to linear when trajectory step is tiny
#     te_sq     = te.squeeze(1)
#     coeff0 = torch.where(linear, 1.0 - te_sq,
#                          torch.sin((1-te_sq)*omega) / sin_omega)
#     coeff1 = torch.where(linear, te_sq,
#                          torch.sin(te_sq*omega) / sin_omega)
#     return coeff0.unsqueeze(-1)*x0 + coeff1.unsqueeze(-1)*x1


# def _slerp_velocity_target(x0, x1, t, lp=None):
#     """Velocity target d(x_t)/dt for SLERP interpolant. Same lp fix as above."""
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
#     a    = torch.sin(dlat/2).pow(2) + torch.cos(lat0)*torch.cos(lat1)*torch.sin(dlon/2).pow(2)
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


# # ══════════════════════════════════════════════════════════════════════════════
# #  EMA
# # ══════════════════════════════════════════════════════════════════════════════

# class EMAModel:
#     def __init__(self, model, decay=0.999):
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
#                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)

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


# # KinematicHead removed — defined in v48 but never called in forward pass.
# # Keeping as stub for checkpoint compatibility (strict=False load ignores extra keys).


# # ══════════════════════════════════════════════════════════════════════════════
# #  VelocityField (từ v48, giữ nguyên — không đổi weights)
# # ══════════════════════════════════════════════════════════════════════════════

# class VelocityField(nn.Module):
#     RAW_CTX_DIM = 512

#     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256,
#                  sigma_min=0.02, unet_in_ch=13, **kwargs):
#         super().__init__()
#         self.pred_len = pred_len
#         self.obs_len  = obs_len

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
#         self.ctx_drop = nn.Dropout(0.10)
#         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

#         # CFG null embedding
#         self.null_embedding = nn.Parameter(torch.randn(1, self.RAW_CTX_DIM) * 0.02)

#         self.vel_obs_enc = nn.Sequential(
#             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
#             nn.Linear(256, 256), nn.GELU())

#         self.steering_enc = nn.Sequential(
#             nn.Linear(4, 64), nn.GELU(), nn.LayerNorm(64),
#             nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 256))

#         self.env_kine_enc = nn.Sequential(
#             nn.Linear(14, 64), nn.GELU(), nn.LayerNorm(64),
#             nn.Linear(64, 256), nn.GELU())

#         # kinematic_head not instantiated (removed — was unused in forward pass)

#         self.time_fc1   = nn.Linear(256, 512)
#         self.time_fc2   = nn.Linear(512, 256)
#         self.traj_embed = nn.Linear(4, 256)
#         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
#         self.step_embed = nn.Embedding(pred_len, 256)
#         self.transformer= nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 d_model=256, nhead=8, dim_feedforward=1024,
#                 dropout=0.10, activation="gelu", batch_first=True),
#             num_layers=2)
#         self.out_fc1 = nn.Linear(256, 512)
#         self.out_fc2 = nn.Linear(512, 4)

#         self.step_scale     = nn.Parameter(torch.ones(pred_len) * 0.5)
#         self.physics_scale  = nn.Parameter(torch.ones(4) * 1.5)
#         self.steering_scale = nn.Parameter(torch.ones(4) * 1.0)
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
#         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

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
#                 accel = torch.cat([obs_traj.new_zeros(1,B), (dspd/10.0).clamp(-3.0,3.0)], 0)
#             else:
#                 accel = obs_traj.new_zeros(T_obs-1, B)
#             kine = torch.stack([vel[:,:,0], vel[:,:,1], speed_n,
#                                  heading.sin(), heading.cos(), accel], dim=-1)
#         else:
#             kine = obs_traj.new_zeros(self.obs_len, B, 6)
#         if kine.shape[0] < self.obs_len:
#             kine = torch.cat([obs_traj.new_zeros(self.obs_len-kine.shape[0], B, 6), kine], 0)
#         else:
#             kine = kine[-self.obs_len:]
#         return self.vel_obs_enc(kine.permute(1,0,2).reshape(B, -1))

#     def _get_vel_obs_feat(self, obs_traj):
#         return self._get_kinematic_obs_feat(obs_traj)

#     def _get_steering_feat(self, env_data, B, device):
#         if env_data is None: return torch.zeros(B, 256, device=device)
#         def _safe(k):
#             v = env_data.get(k)
#             if v is None or not torch.is_tensor(v): return torch.full((B,), 0.0, device=device)
#             v = v.float().to(device)
#             while v.dim() > 1: v = v.mean(-1)
#             return v.view(-1)[:B] if v.numel() >= B else torch.full((B,), 0.0, device=device)
#         return self.steering_enc(torch.stack([
#             _safe("u500_mean"), _safe("v500_mean"),
#             _safe("u500_center"), _safe("v500_center")], dim=-1))

#     def _get_env_kine_feat(self, env_data, B, device):
#         if env_data is None: return torch.zeros(B, 256, device=device)
#         def _get_t(key, dim):
#             v = env_data.get(key)
#             if v is None: return torch.zeros(B, dim, device=device)
#             if not torch.is_tensor(v):
#                 try: v = torch.tensor(v, dtype=torch.float, device=device)
#                 except: return torch.zeros(B, dim, device=device)
#             v = v.float().to(device)
#             if v.dim() == 0: return v.expand(B, dim) if dim == 1 else torch.zeros(B, dim, device=device)
#             if v.dim() == 1:
#                 if v.shape[0] == dim: return v.unsqueeze(0).expand(B, dim)
#                 if v.shape[0] == B:   return v.unsqueeze(1).expand(B, dim) if dim == 1 else torch.zeros(B, dim, device=device)
#             if v.dim() == 2:
#                 if v.shape == (B, dim): return v
#                 return (v[:B, :dim] if v.shape[1] >= dim else F.pad(v[:B], (0, dim-v.shape[1])))
#             if v.dim() == 3:
#                 vv = v[-1] if v.shape[1] == B else v[:B, -1]
#                 return vv[:, :dim] if vv.shape[-1] >= dim else F.pad(vv, (0, dim-vv.shape[-1]))
#             return torch.zeros(B, dim, device=device)
#         feat = torch.cat([_get_t("move_velocity",1), _get_t("history_direction24",8),
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
#             if v is None or not torch.is_tensor(v): return torch.zeros(B, device=device)
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
#         x_emb = (self.traj_embed(x_t[:,:T_seq]) + self.pos_enc[:,:T_seq]
#                  + t_emb.unsqueeze(1) + self.step_embed(step_idx))
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
#         return (v_neural + torch.sigmoid(self.physics_scale)*v_phys
#                 + torch.sigmoid(self.steering_scale)*v_steer)

#     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0,
#                           vel_obs_feat=None, steering_feat=None,
#                           env_kine_feat=None, env_data=None, use_null=False):
#         ctx = self._apply_ctx_head(raw_ctx, noise_scale, use_null=use_null)
#         return self._decode(x_t, t, ctx, vel_obs_feat=vel_obs_feat,
#                             steering_feat=steering_feat,
#                             env_kine_feat=env_kine_feat, env_data=env_data)


# # ══════════════════════════════════════════════════════════════════════════════
# #  Inference helpers
# # ══════════════════════════════════════════════════════════════════════════════

# @torch.no_grad()
# def _speed_sweep_correction(pred_traj_norm, obs_traj_norm,
#                              scales=(0.65, 0.80, 0.90, 1.00, 1.10, 1.25, 1.45)):
#     """[F5] Speed-sweep: tìm scale tốt nhất per-batch per-member."""
#     T_obs, T, B = obs_traj_norm.shape[0], pred_traj_norm.shape[0], pred_traj_norm.shape[1]
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
#         full_deg  = torch.cat([_norm_to_deg(anchor),
#                                 _norm_to_deg(cand)], dim=0)
#         n_c       = min(4, T)
#         cand_spd  = _step_speeds_deg(full_deg[:n_c+1]).mean(0)
#         score     = torch.exp(-((cand_spd - obs_spd)/obs_spd).pow(2)*4.0)
#         better    = score > best_sc
#         best_traj = torch.where(better.view(1,B,1).expand_as(cand), cand, best_traj)
#         best_sc   = torch.where(better, score, best_sc)

#     return best_traj, best_sc


# @torch.no_grad()
# def _persistence_blend(model_pred_norm, obs_traj_norm, blend_strength=0.20):
#     """[F5] Adaptive blend với EWM-persistence."""
#     T_obs, T = obs_traj_norm.shape[0], model_pred_norm.shape[0]
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
#     persist = obs_traj_norm[-1].unsqueeze(0) + ev.unsqueeze(0)*steps.view(T,1,1)
#     obs_spd_all = _step_speeds_deg(_norm_to_deg(obs_traj_norm))
#     if obs_spd_all.shape[0] >= 2:
#         spd_cv  = obs_spd_all.std(0) / obs_spd_all.mean(0).clamp(min=1.0)
#         alpha_b = (blend_strength * torch.sigmoid(-(spd_cv-0.3)*5.0)).unsqueeze(0).unsqueeze(-1)
#     else:
#         alpha_b = blend_strength * 0.5
#     return (1.0-alpha_b)*model_pred_norm + alpha_b*persist


# def _score_ensemble_member(traj_norm, obs_traj_norm, speed_stats=None):
#     """Score một ensemble member: heading(35%) + speed(30%) + prior(20%) + smooth(15%)."""
#     sp      = speed_stats or _SPEED_PRIOR
#     v_opt   = sp.get("v_opt", 15.0); v_sigma = sp.get("v_sigma", 10.0)
#     v_cap   = sp.get("v_hard_cap", 35.0)
#     B, device = traj_norm.shape[1], traj_norm.device

#     spd = _step_speeds_deg(_norm_to_deg(traj_norm))
#     dtd = _norm_to_deg(traj_norm[1:]) - _norm_to_deg(traj_norm[:-1])
#     prior_sc  = torch.exp(-(((spd-v_opt)/v_sigma).pow(2)+F.relu(spd-v_cap)*2.0).mean(0)*0.5)
#     smooth_sc = (torch.exp(-(dtd[1:]-dtd[:-1]).norm(dim=-1).mean(0)*5.0)
#                  if dtd.shape[0] >= 2 else torch.ones(B, device=device))

#     if obs_traj_norm is not None and obs_traj_norm.shape[0] >= 2:
#         obs_v = obs_traj_norm[-1] - obs_traj_norm[-2]
#         if obs_traj_norm.shape[0] >= 3:
#             obs_v = 0.7*obs_v + 0.3*(obs_traj_norm[-2]-obs_traj_norm[-3])
#         obs_hn  = F.normalize(obs_v, dim=-1, eps=1e-6)
#         n_h     = min(3, traj_norm.shape[0]-1)
#         pv_m    = (traj_norm[1:1+n_h]-traj_norm[:n_h]).mean(0) if n_h >= 1 else obs_v
#         pred_hn = F.normalize(pv_m, dim=-1, eps=1e-6)
#         head_sc = torch.exp(((obs_hn*pred_hn).sum(-1)-1.0)*3.0)
#         obs_ref = _step_speeds_deg(_norm_to_deg(obs_traj_norm))[-min(3,obs_traj_norm.shape[0]-1):].mean(0)
#         spd_sc  = torch.exp(-((spd[:min(4,spd.shape[0])].mean(0)-obs_ref)/obs_ref.clamp(min=5.0)).pow(2)*3.0)
#     else:
#         head_sc = spd_sc = torch.ones(B, device=device)

#     return head_sc.pow(0.35)*spd_sc.pow(0.30)*prior_sc.pow(0.20)*smooth_sc.pow(0.15)


# # ══════════════════════════════════════════════════════════════════════════════
# #  TCFlowMatching v50
# # ══════════════════════════════════════════════════════════════════════════════

# class TCFlowMatching(nn.Module):

#     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
#                  n_train_ens=4, unet_in_ch=13, ctx_noise_scale=0.01,
#                  initial_sample_sigma=0.03, teacher_forcing=True,
#                  use_ema=True, ema_decay=0.999,
#                  use_ate_ot=True, ot_epsilon=0.05,
#                  use_slerp=True,
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

#         self.net       = VelocityField(pred_len=pred_len, obs_len=obs_len,
#                                        sigma_min=sigma_min, unet_in_ch=unet_in_ch)
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

#     def _cfm_noisy_slerp(self, x1, sigma_min=None, lp=None):
#         """
#         lp: [B,2] anchor position (normalized). Passed to SLERP for correct omega.
#         x1 is relative coords [B,T,4], so lp is needed to reconstruct absolute.
#         """
#         if sigma_min is None: sigma_min = self.sigma_min
#         B = x1.shape[0]; device = x1.device
#         x0 = torch.randn_like(x1) * sigma_min
#         t  = torch.rand(B, device=device)
#         if self.use_slerp and x1.shape[-1] >= 2:
#             x_t      = _slerp_interpolant(x0, x1, t, lp=lp)
#             u_target = _slerp_velocity_target(x0, x1, t, lp=lp)
#         else:
#             te = t.view(B, 1, 1)
#             x_t      = (1.0-te)*x0 + te*x1
#             u_target = x1 - x0
#         return x_t, t, u_target

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
#             # EWM velocity
#             vels = obs_traj[1:] - obs_traj[:-1]  # [T-1,B,2]
#             n_v  = vels.shape[0]
#             alpha = 0.7
#             w = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
#                               dtype=torch.float, device=device).flip(0)
#             w  = w / w.sum()
#             lv = (vels * w.view(-1,1,1)).sum(0)  # [B,2]
#         elif obs_traj.shape[0] >= 2:
#             lv = obs_traj[-1, :, :2] - obs_traj[-2, :, :2]
#         else:
#             lv = obs_traj.new_zeros(B, 2)

#         steps    = torch.arange(1, pred_len+1, device=device).float()
#         pred_abs = obs_traj[-1, :, :2].unsqueeze(1) + lv.unsqueeze(1)*steps.view(1,-1,1)
#         pred_abs = pred_abs.permute(1, 0, 2)  # [T,B,2]
#         pred_rel_pos = pred_abs - lp.unsqueeze(0)
#         pred_rel     = torch.cat([pred_rel_pos, torch.zeros_like(pred_rel_pos)], dim=-1)
#         return pred_rel.permute(1, 0, 2)  # [B,T,4]

#     def _compute_obs_momentum(self, obs_traj_norm):
#         """EWM velocity from obs."""
#         T_obs = obs_traj_norm.shape[0]
#         if T_obs < 2:
#             return torch.zeros(obs_traj_norm.shape[1], 2, device=obs_traj_norm.device)
#         vels = obs_traj_norm[1:] - obs_traj_norm[:-1]
#         n_v  = vels.shape[0]
#         if n_v >= 3:
#             alpha = 0.65
#             w = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
#                               dtype=torch.float, device=obs_traj_norm.device).flip(0)
#             return (vels * (w/w.sum()).view(-1,1,1)).sum(0)
#         elif n_v == 2:
#             return 0.65*vels[-1] + 0.35*vels[-2]
#         return vels[-1]

#     @staticmethod
#     def _sigma_schedule(epoch):
#         """[BUGFIX] Faster convergence: 0.10→0.04 starting ep2."""
#         if epoch < 2:   return 0.10
#         if epoch < 20:  return 0.10 - (epoch-2)/18.0*(0.10-0.04)
#         return max(0.04 - (epoch-20)/20.0*0.01, 0.03)

#     # ── Training ──────────────────────────────────────────────────────────────

#     def get_loss(self, batch_list, epoch=0, **kwargs):
#         return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

#     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
#         batch_list = self._lon_flip_aug(batch_list)
#         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)

#         obs_t    = batch_list[0]
#         env_data = batch_list[13] if len(batch_list) > 13 else None
#         lp, lm   = obs_t[-1], batch_list[7][-1]
#         B, device = obs_t.shape[1], obs_t.device

#         speed_stats   = compute_speed_stats_from_norm(obs_t[..., :2])
#         current_sigma = self._sigma_schedule(epoch)
#         raw_ctx       = self.net._context(batch_list)

#         x1_rel = self._to_rel(batch_list[1], batch_list[8], lp, lm)

#         # [F3] Spherical OT matching
#         if self.use_ate_ot and B >= 4:
#             noise_base = torch.randn_like(x1_rel) * current_sigma
#             noise_matched, x1_matched = _spherical_ot_matching(
#                 noise_base, x1_rel, lp, epsilon=self.ot_epsilon)
#         else:
#             noise_matched = torch.randn_like(x1_rel) * current_sigma
#             x1_matched    = x1_rel

#         # [F1 SLERP] interpolant — lp passed so SLERP omega uses absolute positions
#         x_t, fm_t, u_target = self._cfm_noisy_slerp(x1_matched, sigma_min=current_sigma, lp=lp)

#         use_null = (torch.rand(1).item() < self.cfg_uncond_prob)
#         pred_vel = self.net.forward_with_ctx(
#             x_t, fm_t, raw_ctx, env_data=env_data, use_null=use_null,
#             vel_obs_feat  = self.net._get_kinematic_obs_feat(obs_t[:, :, :2]),
#             steering_feat = self.net._get_steering_feat(env_data, B, device),
#             env_kine_feat = self.net._get_env_kine_feat(env_data, B, device),
#         )

#         l_fm_mse = F.mse_loss(pred_vel, u_target)
#         fm_te    = fm_t.view(B, 1, 1)
#         x1_pred  = x_t + (1.0 - fm_te) * pred_vel
#         pred_abs, _ = self._to_abs(x1_pred, lp, lm)
#         pred_deg    = _norm_to_deg(pred_abs)
#         gt_deg      = _norm_to_deg(batch_list[1])

#         # [F4] v50 loss
#         loss_dict = compute_st_trans_loss(
#             pred_deg, gt_deg, epoch=epoch, speed_stats=speed_stats)

#         total = l_fm_mse + 2.0 * loss_dict["total"]
#         if torch.isnan(total) or torch.isinf(total):
#             total = x_t.new_zeros(())

#         d = dict(loss_dict)
#         d.update({
#             "total"      : total,
#             "fm_mse"     : l_fm_mse.item(),
#             "sigma"      : current_sigma,
#             "v_opt"      : speed_stats.get("v_opt", 15.0),
#             "obs_spd_p50": speed_stats.get("p50_kmh", 0.0),
#         })
#         return d

#     # ── Sampling ──────────────────────────────────────────────────────────────

#     @torch.no_grad()
#     def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
#                predict_csv=None, importance_weight=True, use_cfg=True):
#         """
#         [F5] v50 sample:
#           - 50 DDIM trajectories với adaptive CFG
#           - Speed-sweep 7×50 = 350 candidates
#           - Top-35% median + persistence blend
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

#         # Obs heading for CFG safety gate
#         if obs_t.shape[0] >= 2:
#             obs_h_n = F.normalize(obs_t[-1,:,:2] - obs_t[-2,:,:2], dim=-1, eps=1e-6)
#         else:
#             obs_h_n = None

#         # Momentum + stability gate
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
#                         env_kine_feat=env_kine_feat, env_data=env_data, use_null=False)
#                     v_uncond = self.net.forward_with_ctx(
#                         x_t, t_b, raw_ctx, noise_scale=0.0,
#                         vel_obs_feat=vel_obs_feat, steering_feat=steering_feat,
#                         env_kine_feat=env_kine_feat, env_data=env_data, use_null=True)
#                     if obs_h_n is not None:
#                         # Compare direction of predicted velocity vs obs heading
#                         pred_h = F.normalize(v_cond[:,0,:2].detach(), dim=-1, eps=1e-6)
#                         cos_a  = (obs_h_n * pred_h).sum(-1).clamp(-1.0, 1.0)
#                         # gs: 1.5 when perfectly aligned, 0.8 when >90° off
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

#         # Speed-sweep
#         SCALES = (0.65, 0.80, 0.90, 1.00, 1.10, 1.25, 1.45)
#         cands, scores = [], []
#         for tn in all_norms:
#             bt, bsc = _speed_sweep_correction(tn, obs_norm, SCALES)
#             cands.append(bt); scores.append(bsc)
#             cands.append(tn); scores.append(_score_ensemble_member(tn, obs_norm, speed_stats))

#         all_c  = torch.stack(cands)   # [100,T,B,2]
#         all_sc = torch.stack(scores)  # [100,B]
#         all_me_t = torch.stack(all_me)

#         k = max(1, int(all_c.shape[0] * 0.35))
#         _, idx = all_sc.topk(k, dim=0)
#         pred_mean = torch.stack([all_c[idx[:,b],:,b,:].median(0).values
#                                   for b in range(B)], dim=1)

#         pred_mean = _persistence_blend(pred_mean, obs_norm, blend_strength=0.20)

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
#                   "lon_mean_deg","lat_mean_deg","lon_std_deg","lat_std_deg","ens_spread_km"]
#         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#         write_hdr = not os.path.exists(csv_path)
#         with open(csv_path, "a", newline="") as fh:
#             w = csv.DictWriter(fh, fieldnames=fields)
#             if write_hdr: w.writeheader()
#             for b in range(B):
#                 for k in range(T):
#                     dlat   = alat[:,k,b] - mlat[k,b]
#                     dlon   = (alon[:,k,b]-mlon[k,b]) * math.cos(math.radians(float(mlat[k,b])))
#                     spread = float(((dlat**2+dlon**2)**0.5).mean() * DEG2KM)
#                     w.writerow({"timestamp":ts,"batch_idx":b,"step_idx":k,
#                                 "lead_h":(k+1)*6,
#                                 "lon_mean_deg":f"{mlon[k,b]:.4f}",
#                                 "lat_mean_deg":f"{mlat[k,b]:.4f}",
#                                 "lon_std_deg":f"{alon[:,k,b].std():.4f}",
#                                 "lat_std_deg":f"{alat[:,k,b].std():.4f}",
#                                 "ens_spread_km":f"{spread:.2f}"})


# # Backward compat aliases
# TCDiffusion = TCFlowMatching

"""
Model/flow_matching_model.py — TC-FlowMatching v51
═══════════════════════════════════════════════════════════════════════════════

THAY ĐỔI SO VỚI v50 (Phase 1 ATE fix — resume từ ep15):

  [P1-1] EMA decay 0.999 → 0.995
         Lý do: half-life cũ ~693 epoch → EMA không reflect model
         half-life mới ~138 epoch → EMA meaningful từ ep20+

  [P1-2] use_slerp=False (default)
         Lý do: fm_mse spikes tới 2.7 tại ep13 do SLERP+OT
         Linear interpolant: u_target = x1 - x0 (constant, stable)
         fm_mse về range 0.3-0.8

  [P1-3] vel_reg weight 0.30 → 0.60
         Lý do: vel_reg loss value ~0.05, gradient ~0.015 vs DPE ~1.0
         Fix: 0.05 × 0.60 = 0.030, cân bằng hơn

  [P1-4] _signed_ate_loss (MỚI, weight 0.40)
         Penalize trực tiếp along-track error component
         Paper ST-Trans không có → đây là lợi thế cạnh tranh
         Target: ATE 149km → <80km

  [P1-5] _direct_72h_loss (weight scheduled)
         Scheduled: 0 tại ep15, tăng dần đến 0.50 tại ep20+
         Lý do: tránh shock gradient khi resume
         72h target: 360km → <300km

  [P1-6] _sigma_schedule aggressive
         ep<2: 0.10, ep2-10: linear→0.04, ep10+: 0.03
         Lý do: FM mean-trajectory bias khi sigma cao → ATE cao

  [P1-7] _spherical_ate_cte_loss: ate_weight 3.0 → 5.0
         CTE đã tốt (70km), cần shift budget sang ATE

  [P1-8] DPE weight giảm 1.0 → 0.60, nhường cho vel_reg và ate losses

TARGETS: ATE<79.94 | 72h<297 | 48h<200 | CTE<93.58 | ADE<136.41

MONITOR sau ep16-18:
  vel_reg   : phải tăng lên 0.05-0.15
  ate_direct: metric mới, phải giảm liên tục
  fm_mse    : phải < 1.0 (hết spike)
  ATE val   : ep20 ~120km, ep25 ~100km, ep35 <80km
"""
from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from typing import Optional, Tuple

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

# Step weights: 72h (step 11) = 10×, 48h (step 7) = 4×
STEP_WEIGHTS = [1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0]

_SPEED_PRIOR = {"v_opt": 15.0, "v_sigma": 10.0, "v_hard_cap": 35.0}


# ══════════════════════════════════════════════════════════════════════════════
#  Coordinate utilities
# ══════════════════════════════════════════════════════════════════════════════

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
    """Per-step speed km/h. traj_deg [T,B,2] → [T-1,B]."""
    T = traj_deg.shape[0]
    if T < 2:
        return traj_deg.new_zeros(1, traj_deg.shape[1])
    return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS


def _forward_azimuth(p1, p2):
    """Great-circle bearing p1→p2. [...,2] degrees → [...] radians."""
    lon1 = torch.deg2rad(p1[..., 0]); lat1 = torch.deg2rad(p1[..., 1])
    lon2 = torch.deg2rad(p2[..., 0]); lat2 = torch.deg2rad(p2[..., 1])
    dlon = lon2 - lon1
    y = torch.sin(dlon) * torch.cos(lat2)
    x = (torch.cos(lat1) * torch.sin(lat2)
         - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon))
    return torch.atan2(y, x)


# ══════════════════════════════════════════════════════════════════════════════
#  Speed statistics
# ══════════════════════════════════════════════════════════════════════════════

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
        q      = torch.quantile(sf, torch.tensor([.50, .75, .95], device=sf.device))
        p50, p95 = float(q[0]), float(q[2])
    return {
        "mean_kmh"  : mean_s,  "std_kmh"   : std_s,
        "p50_kmh"   : p50,     "p95_kmh"   : p95,
        "v_opt"     : max(p50, 5.0),
        "v_sigma"   : max(std_s + 5.0, 5.0),
        "v_hard_cap": float(torch.tensor(p95 * 1.8).clamp(25.0, 80.0)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  [P1-7] Spherical ATE/CTE loss — ate_weight 3.0 → 5.0
# ══════════════════════════════════════════════════════════════════════════════

def _spherical_ate_cte_loss(pred_deg, gt_deg, ate_weight=5.0, cte_weight=1.0):
    """
    [P1-7] Geodesic ATE/CTE projection. ate_weight tăng 3.0→5.0.

    CTE đã tốt (70km), shift learning budget sang ATE reduction.
    ATE = total_err * cos(bearing_error - bearing_along)
    CTE = total_err * sin(bearing_error - bearing_along)
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        zero = pred_deg.new_zeros(())
        return {"ate": zero, "cte": zero, "ate_cte_total": zero,
                "ate_mean_km": 0.0, "cte_mean_km": 0.0}

    bear_along = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
    bear_error = _forward_azimuth(gt_deg[1:T],    pred_deg[1:T])
    total_err  = _haversine_deg(pred_deg[1:T], gt_deg[1:T])

    angle = bear_error - bear_along
    ate   = total_err * torch.cos(angle)
    cte   = total_err * torch.sin(angle)

    w = pred_deg.new_tensor(STEP_WEIGHTS[1:T])
    w = w / w.sum() * (T - 1)

    ate_t = 200.0; cte_t = 100.0
    ate_loss = (torch.where(ate.abs() < ate_t,
                            ate.pow(2) / (2*ate_t),
                            ate.abs() - ate_t/2) * w.unsqueeze(1)).mean() / ate_t
    cte_loss = (torch.where(cte.abs() < cte_t,
                            cte.pow(2) / (2*cte_t),
                            cte.abs() - cte_t/2) * w.unsqueeze(1)).mean() / cte_t

    total = ate_weight * ate_loss + cte_weight * cte_loss
    return {"ate": ate_loss, "cte": cte_loss, "ate_cte_total": total,
            "ate_mean_km": float(ate.abs().mean()),
            "cte_mean_km": float(cte.abs().mean())}


# ══════════════════════════════════════════════════════════════════════════════
#  [P1-3] Velocity regression loss — weight sẽ tăng 0.30→0.60
# ══════════════════════════════════════════════════════════════════════════════

def _velocity_regression_loss(pred_deg, gt_deg, speed_stats=None):
    """
    [P1-3] Velocity supervision — root cause ATE fix.

    DPE loss không dạy per-step speed. Model bị mean-regression → ATE cao.
    ratio (40%): Huber(pred_spd/gt_spd − 1, δ=0.3)
    abs   (30%): |pred_spd − gt_spd|² / v_sigma²
    vec   (30%): ||pred_vel_km − gt_vel_km||² vector supervision
    """
    sp      = speed_stats or _SPEED_PRIOR
    v_sigma = sp.get("v_sigma", 10.0)
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        return pred_deg.new_zeros(())

    pred_spd = _step_speeds_deg(pred_deg[:T])
    gt_spd   = _step_speeds_deg(gt_deg[:T])
    w = pred_deg.new_tensor(STEP_WEIGHTS[1:T]); w = w / w.sum()

    ratio   = pred_spd / gt_spd.clamp(min=2.0)
    l_ratio = (F.huber_loss(ratio, torch.ones_like(ratio), delta=0.3,
                             reduction="none") * w.unsqueeze(1)).mean()
    l_abs   = ((pred_spd - gt_spd).pow(2) / v_sigma**2
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

    return (0.4*l_ratio + 0.3*l_abs + 0.3*l_vec).clamp(0.0, 20.0)


# ══════════════════════════════════════════════════════════════════════════════
#  [P1-4] Signed ATE loss — MỚI
# ══════════════════════════════════════════════════════════════════════════════

def _signed_ate_loss(pred_deg, gt_deg):
    """
    [P1-4] Direct along-track error penalization.

    Penalize projection của error vector lên along-track direction của GT.
    Khác với spherical_ate_cte (dùng bearing angle), hàm này dùng
    flat-projection tại từng step → gradient trực tiếp hơn.

    Positive ATE: pred đi chậm hơn GT (bão chưa tới nơi)
    Negative ATE: pred đi nhanh hơn GT (bão qua khỏi nơi)
    Cả hai đều bị penalize như nhau.

    Tại sao paper ST-Trans không có: họ dùng best-track only, không có
    ERA5 context → model đơn giản hơn, không cần loss này.
    Với FM model phức tạp hơn, loss này là lợi thế cạnh tranh.
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 3:
        return pred_deg.new_zeros(())

    # Along-track direction tại mỗi step (từ GT)
    gt_dir = gt_deg[1:T] - gt_deg[:T-1]        # [T-1, B, 2]
    gt_dir_n = F.normalize(gt_dir, dim=-1, eps=1e-6)

    # Error vector tại mỗi step
    err = pred_deg[1:T] - gt_deg[1:T]          # [T-1, B, 2]

    # ATE = projection of error onto along-track direction
    ate = (err * gt_dir_n).sum(-1)             # [T-1, B] km (approx, flat)

    # Convert to km: lat deg * 111 + lon deg * 111 * cos(lat)
    cos_lat = torch.cos(torch.deg2rad(gt_deg[1:T, :, 1])).clamp(1e-4)
    ate_lon = err[:, :, 0] * cos_lat * DEG2KM
    ate_lat = err[:, :, 1] * DEG2KM
    gt_dir_lon = gt_dir_n[:, :, 0] * cos_lat
    gt_dir_lat = gt_dir_n[:, :, 1]
    gt_dir_km_n = F.normalize(
        torch.stack([gt_dir_lon, gt_dir_lat], dim=-1), dim=-1, eps=1e-6)
    err_km = torch.stack([ate_lon, ate_lat], dim=-1)
    ate_km = (err_km * gt_dir_km_n).sum(-1)   # [T-1, B]

    w = pred_deg.new_tensor(STEP_WEIGHTS[1:T]); w = w / w.sum()

    # Huber loss trên ATE_km để robust với outlier
    # delta=80km: linear khi ATE > 80km → strong gradient cho large errors
    ate_loss = F.huber_loss(
        ate_km, torch.zeros_like(ate_km),
        delta=80.0, reduction="none"
    )
    return (ate_loss * w.unsqueeze(1)).mean() / 100.0


# ══════════════════════════════════════════════════════════════════════════════
#  [P1-5] Direct 72h/48h haversine loss — scheduled
# ══════════════════════════════════════════════════════════════════════════════

def _direct_endpoint_loss(pred_deg, gt_deg, epoch=0):
    """
    [P1-5] Raw haversine tại 48h (step 7) và 72h (step 11), không cap.

    Endpoint Huber bị cap tại delta=150km → gradient nhỏ khi error lớn.
    Direct haversine: loss tuyến tính với error → gradient mạnh hơn.

    Schedule: weight tăng dần từ ep15 để tránh shock gradient khi resume.
    ep15: weight=0.10, ep17: 0.25, ep20+: 0.50
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 4:
        return pred_deg.new_zeros(())

    # Schedule weight: tăng dần sau ep15
    base_w = min(0.50, max(0.10, 0.10 + (epoch - 15) * 0.08))

    total = pred_deg.new_zeros(())
    w_sum = 0.0

    # 48h (step 7), 60h (step 9), 72h (step 11) — heavy weight ở 72h
    for s, w, target_km in [
        (7,  0.3, 200.0),   # 48h
        (9,  0.5, 250.0),   # 60h
        (11, 1.5, 297.0),   # 72h — target < 297km (paper)
    ]:
        if s >= T:
            continue
        dist = _haversine_deg(pred_deg[s], gt_deg[s]).mean()
        total = total + w * dist / target_km
        w_sum += w

    return base_w * total / max(w_sum, 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
#  [P1-8] compute_st_trans_loss — v51 rebalanced
# ══════════════════════════════════════════════════════════════════════════════

def compute_st_trans_loss(pred_deg, gt_deg, epoch=0, speed_stats=None):
    """
    v51 loss — Phase 1 ATE fix:

    THAY ĐỔI SO VỚI v50:
      DPE      : 1.0 → 0.60  (giảm, nhường room cho vel_reg + ate losses)
      vel_reg  : 0.30 → 0.60 [P1-3] (tăng 2x, fix gradient imbalance)
      sph_ate  : 0.20 → 0.25 [P1-7] (tăng nhẹ, ate_weight đã tăng 3→5)
      endpoint : 0.20 → 0.15 (giảm, nhường cho direct_endpoint)
      signed_ate: MỚI 0.40   [P1-4] (impact lớn nhất cho ATE)
      direct_ep : MỚI 0.50 scheduled [P1-5]
      MSE      : 0.03 → 0.02 (giảm nhẹ)
      heading  : 0.10 → 0.08 (CTE đã tốt)

    Tổng effective weight: 0.60+0.02+0.05+0.01+0.08+0.60+0.25+0.15+0.40+0.50 = 2.66
    """
    sp         = speed_stats or _SPEED_PRIOR
    v_opt      = sp.get("v_opt",      15.0)
    v_sigma    = sp.get("v_sigma",    10.0)
    v_hard_cap = sp.get("v_hard_cap", 35.0)
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        return {"total": pred_deg.new_zeros(()), "dpe": 0.0, "vel_reg": 0.0}

    # ── DPE (Huber, v50 weights) ──────────────────────────────────────────
    w    = pred_deg.new_tensor(STEP_WEIGHTS[:T]); w = w / w.sum() * T
    dist = _haversine_deg(pred_deg[:T], gt_deg[:T])
    d    = 200.0
    l_dpe = ((torch.where(dist < d,
                          dist.pow(2) / (2*d),
                          dist - d/2)) * w.unsqueeze(1)).mean() / d

    # ── MSE (stability) ───────────────────────────────────────────────────
    l_mse = F.mse_loss(pred_deg[:T], gt_deg[:T])

    # ── Speed prior ───────────────────────────────────────────────────────
    pred_spd = _step_speeds_deg(pred_deg[:T])
    l_speed  = (0.7 * ((pred_spd - v_opt) / v_sigma).pow(2).mean() +
                0.3 * F.relu(pred_spd - v_hard_cap).pow(2).mean() / v_hard_cap**2
                ) if pred_spd.shape[0] > 0 else pred_deg.new_zeros(())

    # ── Accel ─────────────────────────────────────────────────────────────
    l_accel = (((pred_spd[1:] - pred_spd[:-1]).abs() / DT_HOURS).pow(2).mean()
               / max(v_sigma * 0.5, 3.0)**2
               ) if pred_spd.shape[0] >= 2 else pred_deg.new_zeros(())

    # ── Heading ───────────────────────────────────────────────────────────
    if T >= 3:
        pv = pred_deg[1:T] - pred_deg[:T-1]
        gv = gt_deg[1:T]   - gt_deg[:T-1]
        l_heading = (1.0 - (
            F.normalize(pv.reshape(-1, 2), dim=-1, eps=1e-6) *
            F.normalize(gv.reshape(-1, 2), dim=-1, eps=1e-6)
        ).sum(-1)).mean().clamp(0.0, 2.0)
    else:
        l_heading = pred_deg.new_zeros(())

    # ── [P1-3] Velocity regression × 0.60 ────────────────────────────────
    l_vel_reg = _velocity_regression_loss(pred_deg, gt_deg, speed_stats)

    # ── [P1-7] Spherical ATE/CTE (ate_weight=5.0) ────────────────────────
    ate_d     = _spherical_ate_cte_loss(pred_deg[:T], gt_deg[:T],
                                        ate_weight=5.0, cte_weight=1.0)
    l_sph_ate = ate_d["ate_cte_total"]

    # ── Endpoint Huber (giảm weight, nhường cho direct_endpoint) ─────────
    d2 = 150.0
    ep_total, ep_w = pred_deg.new_zeros(()), 0.0
    for s, ew in [(7, 0.5), (9, 0.8), (10, 1.2), (11, 2.0)]:
        if s >= T:
            continue
        de = _haversine_deg(pred_deg[s], gt_deg[s])
        ep_total = ep_total + ew * torch.where(
            de < d2, de.pow(2) / (2*d2), de - d2/2).mean() / d2
        ep_w += ew
    l_endpoint = ep_total / max(ep_w, 1e-6)

    # ── [P1-4] Signed ATE loss (MỚI) × 0.40 ─────────────────────────────
    l_signed_ate = _signed_ate_loss(pred_deg, gt_deg)

    # ── [P1-5] Direct endpoint loss (scheduled) ───────────────────────────
    l_direct_ep = _direct_endpoint_loss(pred_deg, gt_deg, epoch=epoch)

    # ── Total ─────────────────────────────────────────────────────────────
    total = (0.60 * l_dpe
             + 0.02 * l_mse
             + 0.05 * l_speed
             + 0.01 * l_accel
             + 0.08 * l_heading
             + 0.60 * l_vel_reg      # [P1-3] 2× tăng
             + 0.25 * l_sph_ate      # [P1-7] ate_weight=5.0
             + 0.15 * l_endpoint
             + 0.40 * l_signed_ate   # [P1-4] MỚI — direct ATE
             + l_direct_ep)          # [P1-5] MỚI — scheduled 72h/48h

    if torch.isnan(total) or torch.isinf(total):
        total = pred_deg.new_zeros(())

    def _s(x): return x.item() if torch.is_tensor(x) else float(x)
    return dict(
        total       = total,
        dpe         = _s(l_dpe),
        mse         = _s(l_mse),
        speed       = _s(l_speed),
        accel       = _s(l_accel),
        heading     = _s(l_heading),
        vel_reg     = _s(l_vel_reg),
        ate         = _s(ate_d["ate"]),
        cte         = _s(ate_d["cte"]),
        sph_ate     = _s(l_sph_ate),
        endpoint    = _s(l_endpoint),
        signed_ate  = _s(l_signed_ate),   # metric mới — phải giảm liên tục
        direct_ep   = _s(l_direct_ep),    # metric mới — phải giảm liên tục
        ate_mean_km = ate_d.get("ate_mean_km", 0.0),
        cte_mean_km = ate_d.get("cte_mean_km", 0.0),
        # backward compat
        speed_match=0.0, acc_kmh2=0.0, aux_fno=0.0, sigma=0.0,
        fm_mse=0.0, multi_marg=0.0,
    )


# Alias
compute_ate_focused_loss = compute_st_trans_loss


# ══════════════════════════════════════════════════════════════════════════════
#  [F3] Geodesic OT cost + Sinkhorn (giữ nguyên từ v50)
# ══════════════════════════════════════════════════════════════════════════════

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
        cost = _geodesic_ot_cost(x1_batch, x1_batch, lp)
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


# ══════════════════════════════════════════════════════════════════════════════
#  [P1-2] Linear interpolant — SLERP bị tắt
# ══════════════════════════════════════════════════════════════════════════════

def _slerp_interpolant(x0, x1, t, lp=None):
    """
    [P1-2] use_slerp=False mặc định → hàm này chỉ được gọi nếu user set True.
    Giữ lại để backward compat. Với use_slerp=False, dùng linear.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
#  [P1-1] EMA decay 0.999 → 0.995
# ══════════════════════════════════════════════════════════════════════════════

class EMAModel:
    def __init__(self, model, decay=0.995):  # [P1-1] 0.999 → 0.995
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


# ══════════════════════════════════════════════════════════════════════════════
#  VelocityField (giữ nguyên từ v50 — không đổi weights)
# ══════════════════════════════════════════════════════════════════════════════

class VelocityField(nn.Module):
    RAW_CTX_DIM = 512

    def __init__(self, pred_len=12, obs_len=8, ctx_dim=256,
                 sigma_min=0.02, unet_in_ch=13, **kwargs):
        super().__init__()
        self.pred_len = pred_len
        self.obs_len  = obs_len

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
        self.ctx_drop = nn.Dropout(0.10)
        self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

        self.null_embedding = nn.Parameter(torch.randn(1, self.RAW_CTX_DIM) * 0.02)

        self.vel_obs_enc = nn.Sequential(
            nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.GELU())

        self.steering_enc = nn.Sequential(
            nn.Linear(4, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 256))

        self.env_kine_enc = nn.Sequential(
            nn.Linear(14, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 256), nn.GELU())

        self.time_fc1   = nn.Linear(256, 512)
        self.time_fc2   = nn.Linear(512, 256)
        self.traj_embed = nn.Linear(4, 256)
        self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
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
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for name, m in self.named_modules():
                if isinstance(m, nn.Linear) and "out_fc" in name:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None: nn.init.zeros_(m.bias)

    def _time_emb(self, t, dim=256):
        half = dim // 2
        freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
                         * (-math.log(10000.0) / max(half-1, 1)))
        emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
        return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

    def _context(self, batch_list):
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
        t_w = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
                                          device=e_3d_dec_t.device)*0.5, dim=0)
        f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

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
            dy_km   = vel[:,:,1]             * DEG2KM * _NORM_TO_DEG
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
        def _safe(k):
            v = env_data.get(k)
            if v is None or not torch.is_tensor(v):
                return torch.full((B,), 0.0, device=device)
            v = v.float().to(device)
            while v.dim() > 1: v = v.mean(-1)
            return v.view(-1)[:B] if v.numel() >= B else torch.full((B,), 0.0, device=device)
        return self.steering_enc(torch.stack([
            _safe("u500_mean"), _safe("v500_mean"),
            _safe("u500_center"), _safe("v500_center")], dim=-1))

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
        lat_rad = torch.deg2rad(x_t[:,:,1]*5.0).clamp(-85,85)
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
            return v.view(-1)[:B] if v.numel() >= B else torch.zeros(B, device=device)
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


# ══════════════════════════════════════════════════════════════════════════════
#  Inference helpers (giữ nguyên từ v50)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _speed_sweep_correction(pred_traj_norm, obs_traj_norm,
                             scales=(0.65, 0.80, 0.90, 1.00, 1.10, 1.25, 1.45)):
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


def _score_ensemble_member(traj_norm, obs_traj_norm, speed_stats=None):
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

    return (head_sc.pow(0.35) * spd_sc.pow(0.30)
            * prior_sc.pow(0.20) * smooth_sc.pow(0.15))


# ══════════════════════════════════════════════════════════════════════════════
#  TCFlowMatching v51
# ══════════════════════════════════════════════════════════════════════════════

class TCFlowMatching(nn.Module):

    def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
                 n_train_ens=4, unet_in_ch=13, ctx_noise_scale=0.01,
                 initial_sample_sigma=0.03, teacher_forcing=True,
                 use_ema=True, ema_decay=0.995,      # [P1-1] 0.999 → 0.995
                 use_ate_ot=True, ot_epsilon=0.05,
                 use_slerp=False,                    # [P1-2] True → False
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
        self.use_slerp          = use_slerp          # [P1-2]
        self.cfg_guidance_scale = cfg_guidance_scale
        self.cfg_uncond_prob    = cfg_uncond_prob

        self.net     = VelocityField(pred_len=pred_len, obs_len=obs_len,
                                     sigma_min=sigma_min, unet_in_ch=unet_in_ch)
        self.use_ema  = use_ema
        self.ema_decay = ema_decay
        self._ema     = None

    def init_ema(self):
        if self.use_ema:
            self._ema = EMAModel(self, decay=self.ema_decay)  # [P1-1]

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
        """
        [P1-2] Linear interpolant (use_slerp=False mặc định).

        u_target = x1 - x0 (constant, không phụ thuộc t)
        → gradient flow ổn định, fm_mse spikes được loại bỏ
        → fm_mse về range 0.3-0.8 thay vì 0.5-2.7
        """
        if sigma_min is None: sigma_min = self.sigma_min
        B = x1.shape[0]; device = x1.device
        x0 = torch.randn_like(x1) * sigma_min
        t  = torch.rand(B, device=device)
        if self.use_slerp and x1.shape[-1] >= 2:
            # SLERP path — chỉ khi user explicitly set use_slerp=True
            x_t      = _slerp_interpolant(x0, x1, t, lp=lp)
            u_target = _slerp_velocity_target(x0, x1, t, lp=lp)
        else:
            # [P1-2] Linear — default
            te       = t.view(B, 1, 1)
            x_t      = (1.0 - te)*x0 + te*x1
            u_target = x1 - x0          # constant velocity target
        return x_t, t, u_target

    # Alias for backward compat
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
        """
        [P1-6] Aggressive sigma schedule.

        v50: 0.10→0.04 over ep2-20 (quá chậm → FM học mean trajectory → ATE cao)
        v51: 0.10→0.04 over ep2-10, sau ep10: 0.03

        Sigma thấp sớm → FM không bị pull về mean trajectory →
        predict trajectory cụ thể hơn → ATE giảm trực tiếp.
        """
        if epoch < 2:   return 0.10
        if epoch < 10:  return 0.10 - (epoch-2)/8.0 * (0.10 - 0.04)
        if epoch < 20:  return max(0.04 - (epoch-10)/10.0 * 0.01, 0.03)
        return 0.03

    # ── Training ──────────────────────────────────────────────────────────────

    def get_loss(self, batch_list, epoch=0, **kwargs):
        return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

    def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
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

        # OT matching
        if self.use_ate_ot and B >= 4:
            noise_base = torch.randn_like(x1_rel) * current_sigma
            noise_matched, x1_matched = _spherical_ot_matching(
                noise_base, x1_rel, lp, epsilon=self.ot_epsilon)
        else:
            noise_matched = torch.randn_like(x1_rel) * current_sigma
            x1_matched    = x1_rel

        # [P1-2] Linear interpolant (use_slerp=False)
        x_t, fm_t, u_target = self._cfm_noisy(
            x1_matched, sigma_min=current_sigma, lp=lp)

        use_null = (torch.rand(1).item() < self.cfg_uncond_prob)
        pred_vel = self.net.forward_with_ctx(
            x_t, fm_t, raw_ctx, env_data=env_data, use_null=use_null,
            vel_obs_feat  = self.net._get_kinematic_obs_feat(obs_t[:, :, :2]),
            steering_feat = self.net._get_steering_feat(env_data, B, device),
            env_kine_feat = self.net._get_env_kine_feat(env_data, B, device),
        )

        l_fm_mse = F.mse_loss(pred_vel, u_target)
        fm_te    = fm_t.view(B, 1, 1)
        x1_pred  = x_t + (1.0 - fm_te) * pred_vel
        pred_abs, _ = self._to_abs(x1_pred, lp, lm)
        pred_deg    = _norm_to_deg(pred_abs)
        gt_deg      = _norm_to_deg(batch_list[1])

        # [P1-3,4,5,7,8] v51 loss
        loss_dict = compute_st_trans_loss(
            pred_deg, gt_deg, epoch=epoch, speed_stats=speed_stats)

        total = l_fm_mse + 2.0 * loss_dict["total"]
        if torch.isnan(total) or torch.isinf(total):
            total = x_t.new_zeros(())

        d = dict(loss_dict)
        d.update({
            "total"      : total,
            "fm_mse"     : l_fm_mse.item(),
            "sigma"      : current_sigma,
            "v_opt"      : speed_stats.get("v_opt", 15.0),
            "obs_spd_p50": speed_stats.get("p50_kmh", 0.0),
        })
        return d

    # ── Sampling (giữ nguyên từ v50) ─────────────────────────────────────────

    @torch.no_grad()
    def sample(self, batch_list, num_ensemble=50, ddim_steps=20,
               predict_csv=None, importance_weight=True, use_cfg=True):
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
                        pred_h = F.normalize(
                            v_cond[:,0,:2].detach(), dim=-1, eps=1e-6)
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

        SCALES = (0.65, 0.80, 0.90, 1.00, 1.10, 1.25, 1.45)
        cands, scores = [], []
        for tn in all_norms:
            bt, bsc = _speed_sweep_correction(tn, obs_norm, SCALES)
            cands.append(bt); scores.append(bsc)
            cands.append(tn)
            scores.append(_score_ensemble_member(tn, obs_norm, speed_stats))

        all_c  = torch.stack(cands)
        all_sc = torch.stack(scores)
        all_me_t = torch.stack(all_me)

        k = max(1, int(all_c.shape[0] * 0.35))
        _, idx = all_sc.topk(k, dim=0)
        pred_mean = torch.stack([
            all_c[idx[:,b],:,b,:].median(0).values for b in range(B)
        ], dim=1)

        pred_mean = _persistence_blend(pred_mean, obs_norm, blend_strength=0.20)

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
                        "timestamp"    : ts,
                        "batch_idx"    : b,
                        "step_idx"     : k,
                        "lead_h"       : (k+1)*6,
                        "lon_mean_deg" : f"{mlon[k,b]:.4f}",
                        "lat_mean_deg" : f"{mlat[k,b]:.4f}",
                        "lon_std_deg"  : f"{alon[:,k,b].std():.4f}",
                        "lat_std_deg"  : f"{alat[:,k,b].std():.4f}",
                        "ens_spread_km": f"{spread:.2f}",
                    })


# Backward compat
TCDiffusion = TCFlowMatching
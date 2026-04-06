# # """
# # Model/flow_matching_model.py  ── v23
# # ==========================================
# # FIXES vs v21:

# #   FIX-M18  [CURRICULUM REMOVED] set_curriculum_len() vẫn giữ để backward
# #            compat nhưng KHÔNG được gọi từ trainer nữa. active_pred_len
# #            luôn = pred_len. evaluate_full_val_ade không cần restore nữa.

# #   FIX-M19  get_loss_breakdown(): nhận thêm step_weight_alpha parameter
# #            và truyền vào compute_total_loss() → fm_afcrps_loss() sử dụng
# #            soft weighting thay curriculum len-slicing.

# #   FIX-M20  get_loss_breakdown(): truyền all_trajs vào compute_total_loss()
# #            để tính ensemble_spread_loss. Giúp kiểm soát spread tăng quá mức.

# #   FIX-M21  _physics_correct(): tăng n_steps=5 (từ 3), giảm lr=0.002 (từ
# #            0.005) để physics correction ổn định hơn và ít overshoot.

# #   FIX-M22  sample(): initial_sample_sigma=0.1 (set từ constructor) đã fix
# #            spread. Thêm post-sampling clip chặt hơn [-3.0, 3.0] cho cả lon
# #            và lat (từ [-5.0, 5.0] cho lon).

# # Kept from v21:
# #   FIX-M17  _physics_correct với torch.enable_grad()
# #   FIX-M11..M16 OT-CFM, beta drift, env_data, physics scale
# # """
# # from __future__ import annotations

# # import csv
# # import math
# # import os
# # from datetime import datetime
# # from typing import Dict, List, Optional, Tuple

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # from Model.FNO3D_encoder import FNO3DEncoder
# # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # from Model.env_net_transformer_gphsplit import Env_net
# # from Model.losses import (
# #     compute_total_loss, fm_physics_consistency_loss, WEIGHTS,
# #     pinn_speed_constraint,
# # )


# # def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
# #     """Normalised → degrees. Handles [T, B, 2] and [B, 2]."""
# #     out = traj_norm.clone()
# #     out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
# #     out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
# #     return out


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  VelocityField
# # # ══════════════════════════════════════════════════════════════════════════════

# # class VelocityField(nn.Module):
# #     """
# #     OT-CFM velocity field v_θ(x_t, t, context).
# #     Architecture: DataEncoder1D (Mamba) + FNO3D + Env-T-Net → Transformer decoder.
# #     Physics-guided: v_total = v_neural + sigmoid(w_physics) * v_beta_drift.
# #     """

# #     def __init__(
# #         self,
# #         pred_len:   int   = 12,
# #         obs_len:    int   = 8,
# #         ctx_dim:    int   = 256,
# #         sigma_min:  float = 0.02,
# #         unet_in_ch: int   = 13,
# #     ):
# #         super().__init__()
# #         self.pred_len  = pred_len
# #         self.obs_len   = obs_len
# #         self.sigma_min = sigma_min

# #         self.spatial_enc = FNO3DEncoder(
# #             in_channel   = unet_in_ch,
# #             out_channel  = 1,
# #             d_model      = 64,
# #             n_layers     = 4,
# #             modes_t      = 4,
# #             modes_h      = 4,
# #             modes_w      = 4,
# #             spatial_down = 32,
# #             dropout      = 0.05,
# #         )

# #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# #         self.bottleneck_proj = nn.Linear(128, 128)
# #         self.decoder_proj    = nn.Linear(1, 16)

# #         self.enc_1d = DataEncoder1D(
# #             in_1d       = 4,
# #             feat_3d_dim = 128,
# #             mlp_h       = 64,
# #             lstm_hidden = 128,
# #             lstm_layers = 3,
# #             dropout     = 0.1,
# #             d_state     = 16,
# #         )

# #         self.env_enc = Env_net(obs_len=obs_len, d_model=64)

# #         self.ctx_fc1  = nn.Linear(128 + 64 + 16, 512)
# #         self.ctx_ln   = nn.LayerNorm(512)
# #         self.ctx_drop = nn.Dropout(0.15)
# #         self.ctx_fc2  = nn.Linear(512, ctx_dim)

# #         self.time_fc1 = nn.Linear(256, 512)
# #         self.time_fc2 = nn.Linear(512, 256)

# #         self.traj_embed = nn.Linear(4, 256)
# #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# #         self.transformer = nn.TransformerDecoder(
# #             nn.TransformerDecoderLayer(
# #                 d_model=256, nhead=8, dim_feedforward=1024,
# #                 dropout=0.15, activation="gelu", batch_first=True,
# #             ),
# #             num_layers=4,
# #         )
# #         self.out_fc1 = nn.Linear(256, 512)
# #         self.out_fc2 = nn.Linear(512, 4)

# #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)

# #     def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
# #         half = dim // 2
# #         freq = torch.exp(
# #             torch.arange(half, dtype=torch.float32, device=t.device)
# #             * (-math.log(10_000.0) / max(half - 1, 1))
# #         )
# #         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
# #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# #         return F.pad(emb, (0, dim % 2))

# #     def _context(self, batch_list: List) -> torch.Tensor:
# #         obs_traj  = batch_list[0]
# #         obs_Me    = batch_list[7]
# #         image_obs = batch_list[11]
# #         env_data  = batch_list[13]

# #         if image_obs.dim() == 4:
# #             image_obs = image_obs.unsqueeze(1)

# #         expected_ch = self.spatial_enc.in_channel
# #         if image_obs.shape[1] == 1 and expected_ch != 1:
# #             image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# #         T_obs = obs_traj.shape[0]

# #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
# #         e_3d_s = e_3d_s.permute(0, 2, 1)
# #         e_3d_s = self.bottleneck_proj(e_3d_s)

# #         T_bot = e_3d_s.shape[1]
# #         if T_bot != T_obs:
# #             e_3d_s = F.interpolate(
# #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# #                 mode="linear", align_corners=False,
# #             ).permute(0, 2, 1)

# #         f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))
# #         f_spatial     = self.decoder_proj(f_spatial_raw)

# #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# #         h_t    = self.enc_1d(obs_in, e_3d_s)

# #         e_env, _, _ = self.env_enc(env_data, image_obs)

# #         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
# #         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))
# #         return raw

# #     def _apply_ctx_head(self, raw: torch.Tensor,
# #                         noise_scale: float = 0.0) -> torch.Tensor:
# #         if noise_scale > 0.0:
# #             raw = raw + torch.randn_like(raw) * noise_scale
# #         return self.ctx_fc2(self.ctx_drop(raw))

# #     def _beta_drift_velocity(self, x_t: torch.Tensor) -> torch.Tensor:
# #         """Beta drift in normalised state space. x_t: [B, T, 4]."""
# #         OMEGA_val  = 7.2921e-5
# #         R_val      = 6.371e6
# #         DT         = 6 * 3600.0
# #         M_PER_NORM = 5.0 * 111.0 * 1000.0

# #         lat_norm = x_t[:, :, 1]
# #         lat_deg  = lat_norm * 5.0
# #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))

# #         beta   = 2 * OMEGA_val * torch.cos(lat_rad) / R_val
# #         R_tc   = 3e5
# #         v_lon  = -beta * R_tc ** 2 / 2
# #         v_lat  =  beta * R_tc ** 2 / 4

# #         v_lon_norm = v_lon * DT / M_PER_NORM
# #         v_lat_norm = v_lat * DT / M_PER_NORM

# #         v_phys = torch.zeros_like(x_t)
# #         v_phys[:, :, 0] = v_lon_norm
# #         v_phys[:, :, 1] = v_lat_norm
# #         return v_phys

# #     def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
# #                 ctx: torch.Tensor) -> torch.Tensor:
# #         t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
# #         t_emb = self.time_fc2(t_emb)

# #         T_seq  = min(x_t.size(1), self.pos_enc.shape[1])
# #         x_emb  = (self.traj_embed(x_t[:, :T_seq, :])
# #                   + self.pos_enc[:, :T_seq, :]
# #                   + t_emb.unsqueeze(1))
# #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

# #         v_neural = self.out_fc2(F.gelu(self.out_fc1(
# #             self.transformer(x_emb, memory)
# #         )))  # [B, T, 4]

# #         with torch.no_grad():
# #             v_phys = self._beta_drift_velocity(x_t[:, :T_seq, :])

# #         scale = torch.sigmoid(self.physics_scale) * 2.0
# #         return v_neural + scale * v_phys

# #     def forward(self, x_t, t, batch_list):
# #         raw = self._context(batch_list)
# #         ctx = self._apply_ctx_head(raw, noise_scale=0.0)
# #         return self._decode(x_t, t, ctx)

# #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale: float = 0.0):
# #         ctx = self._apply_ctx_head(raw_ctx, noise_scale=noise_scale)
# #         return self._decode(x_t, t, ctx)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  TCFlowMatching
# # # ══════════════════════════════════════════════════════════════════════════════

# # class TCFlowMatching(nn.Module):
# #     """TC trajectory prediction via OT-CFM + Physics-guided velocity field."""

# #     def __init__(
# #         self,
# #         pred_len:             int   = 12,
# #         obs_len:              int   = 8,
# #         sigma_min:            float = 0.02,
# #         n_train_ens:          int   = 4,
# #         unet_in_ch:           int   = 13,
# #         ctx_noise_scale:      float = 0.02,   # FIX-T23-5: 0.02 default
# #         initial_sample_sigma: float = 0.1,    # FIX-T23-4: 0.1 default
# #         **kwargs,
# #     ):
# #         super().__init__()
# #         self.pred_len             = pred_len
# #         self.obs_len              = obs_len
# #         self.sigma_min            = sigma_min
# #         self.n_train_ens          = n_train_ens
# #         self.active_pred_len      = pred_len   # FIX-M18: always full pred_len
# #         self.ctx_noise_scale      = ctx_noise_scale
# #         self.initial_sample_sigma = initial_sample_sigma
# #         self.net = VelocityField(
# #             pred_len   = pred_len,
# #             obs_len    = obs_len,
# #             sigma_min  = sigma_min,
# #             unet_in_ch = unet_in_ch,
# #         )

# #     def set_curriculum_len(self, active_len: int) -> None:
# #         """
# #         FIX-M18: Kept for backward compat but NO-OP in v23.
# #         Curriculum is removed. active_pred_len is always pred_len.
# #         """
# #         # self.active_pred_len = max(1, min(active_len, self.pred_len))
# #         pass  # no-op

# #     @staticmethod
# #     def _to_rel(traj_gt, Me_gt, last_pos, last_Me):
# #         return torch.cat(
# #             [traj_gt - last_pos.unsqueeze(0),
# #              Me_gt   - last_Me.unsqueeze(0)],
# #             dim=-1,
# #         ).permute(1, 0, 2)

# #     @staticmethod
# #     def _to_abs(rel, last_pos, last_Me):
# #         d = rel.permute(1, 0, 2)
# #         return (
# #             last_pos.unsqueeze(0) + d[:, :, :2],
# #             last_Me.unsqueeze(0)  + d[:, :, 2:],
# #         )

# #     def _cfm_noisy(self, x1):
# #         B, device = x1.shape[0], x1.device
# #         sm  = self.sigma_min
# #         x0  = torch.randn_like(x1) * sm
# #         t   = torch.rand(B, device=device)
# #         te  = t.view(B, 1, 1)
# #         x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
# #         denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
# #         target_vel = (x1 - (1.0 - sm) * x_t) / denom
# #         return x_t, t, te, denom, target_vel

# #     @staticmethod
# #     def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
# #         wind_norm = obs_Me[-1, :, 1].detach()
# #         w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
# #             torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
# #             torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
# #                         torch.full_like(wind_norm, 1.5))))
# #         return w / w.mean().clamp(min=1e-6)

# #     @staticmethod
# #     def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
# #         if torch.rand(1).item() > p:
# #             return batch_list
# #         aug = list(batch_list)
# #         for idx in [0, 1, 2, 3]:
# #             t = aug[idx]
# #             if torch.is_tensor(t) and t.shape[-1] >= 1:
# #                 t = t.clone()
# #                 t[..., 0] = -t[..., 0]
# #                 aug[idx] = t
# #         return aug

# #     def get_loss(self, batch_list: List,
# #                  step_weight_alpha: float = 0.0) -> torch.Tensor:
# #         return self.get_loss_breakdown(batch_list, step_weight_alpha)["total"]

# #     def get_loss_breakdown(self, batch_list: List,
# #                            step_weight_alpha: float = 0.0) -> Dict:
# #         """
# #         FIX-M19: Nhận step_weight_alpha, truyền vào compute_total_loss.
# #         FIX-M20: Truyền all_trajs để tính ensemble_spread_loss.
# #         """
# #         batch_list = self._lon_flip_aug(batch_list, p=0.3)

# #         traj_gt  = batch_list[1]
# #         Me_gt    = batch_list[8]
# #         obs_t    = batch_list[0]
# #         obs_Me   = batch_list[7]

# #         try:
# #             env_data = batch_list[13]
# #         except (IndexError, TypeError):
# #             env_data = None

# #         # FIX-M18: NO curriculum slicing. Always use full pred_len.
# #         lp, lm = obs_t[-1], obs_Me[-1]
# #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

# #         raw_ctx     = self.net._context(batch_list)
# #         intensity_w = self._intensity_weights(obs_Me)

# #         x_t, t, te, denom, _ = self._cfm_noisy(x1)
# #         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

# #         # Ensemble samples for AFCRPS + spread penalty
# #         samples: List[torch.Tensor] = []
# #         for _ in range(self.n_train_ens):
# #             xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
# #             pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
# #             x1_s  = xt_s + dens_s * pv_s   # OT-CFM
# #             pa_s, _ = self._to_abs(x1_s, lp, lm)
# #             samples.append(pa_s)
# #         pred_samples = torch.stack(samples)   # [S, T, B, 2]

# #         # FIX-M20: all_trajs for spread penalty
# #         all_trajs_4d = pred_samples   # [S, T, B, 2]

# #         l_fm_physics = fm_physics_consistency_loss(
# #             pred_samples, gt_norm=traj_gt, last_pos=lp)

# #         x1_pred = x_t + denom * pred_vel
# #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# #         pred_abs_deg = _denorm_to_deg(pred_abs)
# #         traj_gt_deg  = _denorm_to_deg(traj_gt)
# #         ref_deg      = _denorm_to_deg(lp)

# #         breakdown = compute_total_loss(
# #             pred_abs           = pred_abs_deg,
# #             gt                 = traj_gt_deg,
# #             ref                = ref_deg,
# #             batch_list         = batch_list,
# #             pred_samples       = pred_samples,
# #             gt_norm            = traj_gt,
# #             weights            = WEIGHTS,
# #             intensity_w        = intensity_w,
# #             env_data           = env_data,
# #             step_weight_alpha  = step_weight_alpha,   # FIX-M19
# #             all_trajs          = all_trajs_4d,         # FIX-M20
# #         )

# #         fm_phys_w = WEIGHTS.get("fm_physics", 0.3)
# #         breakdown["total"]      = breakdown["total"] + fm_phys_w * l_fm_physics
# #         breakdown["fm_physics"] = l_fm_physics.item()

# #         return breakdown

# #     # ── sample() ─────────────────────────────────────────────────────────────

# #     @torch.no_grad()
# #     def sample(
# #         self,
# #         batch_list: List,
# #         num_ensemble: int = 50,
# #         ddim_steps:   int = 20,
# #         predict_csv:  Optional[str] = None,
# #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# #         """
# #         FIX-T23-3: ddim_steps default 20 (từ 10).
# #         FIX-M22: tighter clip [-3.0, 3.0] cho cả lon và lat.
# #         Returns:
# #             pred_mean:  [T, B, 2] mean track (normalised)
# #             me_mean:    [T, B, 2] mean intensity
# #             all_trajs:  [S, T, B, 2] all ensemble members
# #         """
# #         lp  = batch_list[0][-1]   # [B, 2]
# #         lm  = batch_list[7][-1]   # [B, 2]
# #         B   = lp.shape[0]
# #         device = lp.device
# #         T   = self.pred_len   # FIX-M18: always full pred_len
# #         dt  = 1.0 / max(ddim_steps, 1)

# #         raw_ctx = self.net._context(batch_list)

# #         traj_s: List[torch.Tensor] = []
# #         me_s:   List[torch.Tensor] = []

# #         for k in range(num_ensemble):
# #             # FIX-T23-4: initial_sample_sigma=0.1 (set in constructor)
# #             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# #             # DDIM Euler integration
# #             for step in range(ddim_steps):
# #                 t_b = torch.full((B,), step * dt, device=device)
# #                 ns  = self.ctx_noise_scale if step == 0 else 0.0
# #                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
# #                 x_t = x_t + dt * vel

# #             # Physics correction
# #             x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)

# #             # FIX-M22: tighter clip
# #             x_t = x_t.clamp(-3.0, 3.0)

# #             tr, me = self._to_abs(x_t, lp, lm)
# #             traj_s.append(tr)
# #             me_s.append(me)

# #         all_trajs = torch.stack(traj_s)   # [S, T, B, 2]
# #         all_me    = torch.stack(me_s)
# #         pred_mean = all_trajs.mean(0)
# #         me_mean   = all_me.mean(0)

# #         if predict_csv:
# #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)

# #         return pred_mean, me_mean, all_trajs

# #     # ── Physics correction ────────────────────────────────────────────────────

# #     def _physics_correct(
# #         self,
# #         x_pred: torch.Tensor,
# #         last_pos: torch.Tensor,
# #         last_Me:  torch.Tensor,
# #         n_steps:  int   = 5,    # FIX-M21: 5 (từ 3)
# #         lr:       float = 0.002, # FIX-M21: 0.002 (từ 0.005)
# #     ) -> torch.Tensor:
# #         """
# #         FIX-M17: torch.enable_grad() inside no_grad context.
# #         FIX-M21: n_steps=5, lr=0.002 for more stable correction.
# #         """
# #         with torch.enable_grad():
# #             x = x_pred.detach().requires_grad_(True)
# #             optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)

# #             for _ in range(n_steps):
# #                 optimizer.zero_grad()
# #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# #                 pred_deg    = _denorm_to_deg(pred_abs)

# #                 l_speed = self._pinn_speed_constraint(pred_deg)
# #                 l_accel = self._pinn_beta_plane_simplified(pred_deg)

# #                 physics_loss = l_speed + 0.3 * l_accel
# #                 physics_loss.backward()

# #                 torch.nn.utils.clip_grad_norm_([x], max_norm=0.05)
# #                 optimizer.step()

# #         return x.detach()

# #     @staticmethod
# #     def _pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# #         if pred_deg.shape[0] < 2:
# #             return pred_deg.new_zeros(())
# #         dt_deg  = pred_deg[1:] - pred_deg[:-1]
# #         lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# #         dx_km   = dt_deg[:, :, 0] * cos_lat * 111.0
# #         dy_km   = dt_deg[:, :, 1] * 111.0
# #         speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# #         return F.relu(speed - 600.0).pow(2).mean()

# #     @staticmethod
# #     def _pinn_beta_plane_simplified(pred_deg: torch.Tensor) -> torch.Tensor:
# #         if pred_deg.shape[0] < 3:
# #             return pred_deg.new_zeros(())
# #         v = pred_deg[1:] - pred_deg[:-1]
# #         a = v[1:] - v[:-1]
# #         lat_rad = torch.deg2rad(pred_deg[1:-1, :, 1])
# #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# #         a_lon_km = a[:, :, 0] * cos_lat * 111.0
# #         a_lat_km = a[:, :, 1] * 111.0
# #         max_accel = 50.0
# #         violation = F.relu(torch.sqrt(a_lon_km**2 + a_lat_km**2) - max_accel)
# #         return violation.pow(2).mean() * 0.1

# #     @staticmethod
# #     def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
# #                            all_trajs: torch.Tensor) -> None:
# #         import numpy as np
# #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# #         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
# #         T, B, _ = traj_mean.shape
# #         S       = all_trajs.shape[0]

# #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()

# #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# #                     "lon_mean_deg", "lat_mean_deg",
# #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# #         write_hdr = not os.path.exists(csv_path)
# #         with open(csv_path, "a", newline="") as fh:
# #             w = csv.DictWriter(fh, fieldnames=fields)
# #             if write_hdr:
# #                 w.writeheader()
# #             for b in range(B):
# #                 for k in range(T):
# #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
# #                         math.radians(mean_lat[k, b]))
# #                     spread = float(np.sqrt((dlat**2 + dlon**2).mean()) * 111.0)
# #                     w.writerow(dict(
# #                         timestamp     = ts,
# #                         batch_idx     = b,
# #                         step_idx      = k,
# #                         lead_h        = (k + 1) * 6,
# #                         lon_mean_deg  = f"{mean_lon[k,b]:.4f}",
# #                         lat_mean_deg  = f"{mean_lat[k,b]:.4f}",
# #                         lon_std_deg   = f"{all_lon[:,k,b].std():.4f}",
# #                         lat_std_deg   = f"{all_lat[:,k,b].std():.4f}",
# #                         ens_spread_km = f"{spread:.2f}",
# #                     ))
# #         print(f"  Predictions → {csv_path}  (B={B}, T={T}, S={S})")


# # # Backward-compat alias
# # TCDiffusion = TCFlowMatching

# """
# Model/flow_matching_model.py  ── v24
# ==========================================
# FIXES vs v23:

#   FIX-M23  [P0-CRITICAL] get_loss() nhận step_weight_alpha parameter.
#            v23: get_loss() luôn gọi get_loss_breakdown() với alpha=0.0
#            → val loss và train loss dùng khác objective.
#            Fix: get_loss(batch_list, step_weight_alpha=0.0) forward alpha.

#   FIX-M24  [P3] _lat_shift_aug(): thêm latitude shift augmentation.
#            Tropical cyclones di chuyển trong latitudinal range rộng.
#            Shift nhỏ ±5° latitude trong normalised space giúp model
#            học được TC behaviour tại các latitudes khác nhau mà không
#            cần thêm data.
#            p=0.2 (nhẹ để không distort quá nhiều)

#   FIX-M25  [P3] _jitter_aug(): thêm jitter nhỏ vào trajectory.
#            Gaussian noise trên obs và pred trajectory (std=0.002 normalised
#            ~ 1.1 km) để prevent overfitting trên ít sequences.

#   FIX-M26  sample() trả về đúng normalised coords cho denorm_torch.
#            Không có bug nhưng thêm assertion để dễ debug.

# Kept from v23:
#   FIX-M17..M22 (physics_correct, initial_sample_sigma, etc.)
# """
# from __future__ import annotations

# import csv
# import math
# import os
# from datetime import datetime
# from typing import Dict, List, Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from Model.FNO3D_encoder import FNO3DEncoder
# from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# from Model.env_net_transformer_gphsplit import Env_net
# from Model.losses import (
#     compute_total_loss, fm_physics_consistency_loss, WEIGHTS,
#     pinn_speed_constraint,
# )


# def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
#     """Normalised → degrees. Handles [T, B, 2] and [B, 2]."""
#     out = traj_norm.clone()
#     out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
#     out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
#     return out


# # ══════════════════════════════════════════════════════════════════════════════
# #  VelocityField
# # ══════════════════════════════════════════════════════════════════════════════

# class VelocityField(nn.Module):
#     def __init__(
#         self,
#         pred_len:   int   = 12,
#         obs_len:    int   = 8,
#         ctx_dim:    int   = 256,
#         sigma_min:  float = 0.02,
#         unet_in_ch: int   = 13,
#     ):
#         super().__init__()
#         self.pred_len  = pred_len
#         self.obs_len   = obs_len
#         self.sigma_min = sigma_min

#         self.spatial_enc = FNO3DEncoder(
#             in_channel   = unet_in_ch,
#             out_channel  = 1,
#             d_model      = 64,
#             n_layers     = 4,
#             modes_t      = 4,
#             modes_h      = 4,
#             modes_w      = 4,
#             spatial_down = 32,
#             dropout      = 0.05,
#         )

#         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
#         self.bottleneck_proj = nn.Linear(128, 128)
#         self.decoder_proj    = nn.Linear(1, 16)

#         self.enc_1d = DataEncoder1D(
#             in_1d       = 4,
#             feat_3d_dim = 128,
#             mlp_h       = 64,
#             lstm_hidden = 128,
#             lstm_layers = 3,
#             dropout     = 0.1,
#             d_state     = 16,
#         )

#         self.env_enc = Env_net(obs_len=obs_len, d_model=64)

#         self.ctx_fc1  = nn.Linear(128 + 64 + 16, 512)
#         self.ctx_ln   = nn.LayerNorm(512)
#         self.ctx_drop = nn.Dropout(0.15)
#         self.ctx_fc2  = nn.Linear(512, ctx_dim)

#         self.time_fc1 = nn.Linear(256, 512)
#         self.time_fc2 = nn.Linear(512, 256)

#         self.traj_embed = nn.Linear(4, 256)
#         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
#         self.transformer = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 d_model=256, nhead=8, dim_feedforward=1024,
#                 dropout=0.15, activation="gelu", batch_first=True,
#             ),
#             num_layers=4,
#         )
#         self.out_fc1 = nn.Linear(256, 512)
#         self.out_fc2 = nn.Linear(512, 4)

#         self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)

#     def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
#         half = dim // 2
#         freq = torch.exp(
#             torch.arange(half, dtype=torch.float32, device=t.device)
#             * (-math.log(10_000.0) / max(half - 1, 1))
#         )
#         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
#         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
#         return F.pad(emb, (0, dim % 2))

#     def _context(self, batch_list: List) -> torch.Tensor:
#         obs_traj  = batch_list[0]
#         obs_Me    = batch_list[7]
#         image_obs = batch_list[11]
#         env_data  = batch_list[13]

#         if image_obs.dim() == 4:
#             image_obs = image_obs.unsqueeze(1)

#         expected_ch = self.spatial_enc.in_channel
#         if image_obs.shape[1] == 1 and expected_ch != 1:
#             image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

#         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
#         T_obs = obs_traj.shape[0]

#         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
#         e_3d_s = e_3d_s.permute(0, 2, 1)
#         e_3d_s = self.bottleneck_proj(e_3d_s)

#         T_bot = e_3d_s.shape[1]
#         if T_bot != T_obs:
#             e_3d_s = F.interpolate(
#                 e_3d_s.permute(0, 2, 1), size=T_obs,
#                 mode="linear", align_corners=False,
#             ).permute(0, 2, 1)

#         f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))
#         f_spatial     = self.decoder_proj(f_spatial_raw)

#         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
#         h_t    = self.enc_1d(obs_in, e_3d_s)

#         e_env, _, _ = self.env_enc(env_data, image_obs)

#         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
#         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))
#         return raw

#     def _apply_ctx_head(self, raw: torch.Tensor,
#                         noise_scale: float = 0.0) -> torch.Tensor:
#         if noise_scale > 0.0:
#             raw = raw + torch.randn_like(raw) * noise_scale
#         return self.ctx_fc2(self.ctx_drop(raw))

#     def _beta_drift_velocity(self, x_t: torch.Tensor) -> torch.Tensor:
#         OMEGA_val  = 7.2921e-5
#         R_val      = 6.371e6
#         DT         = 6 * 3600.0
#         M_PER_NORM = 5.0 * 111.0 * 1000.0

#         lat_norm = x_t[:, :, 1]
#         lat_deg  = lat_norm * 5.0
#         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))

#         beta   = 2 * OMEGA_val * torch.cos(lat_rad) / R_val
#         R_tc   = 3e5
#         v_lon  = -beta * R_tc ** 2 / 2
#         v_lat  =  beta * R_tc ** 2 / 4

#         v_lon_norm = v_lon * DT / M_PER_NORM
#         v_lat_norm = v_lat * DT / M_PER_NORM

#         v_phys = torch.zeros_like(x_t)
#         v_phys[:, :, 0] = v_lon_norm
#         v_phys[:, :, 1] = v_lat_norm
#         return v_phys

#     def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
#                 ctx: torch.Tensor) -> torch.Tensor:
#         t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
#         t_emb = self.time_fc2(t_emb)

#         T_seq  = min(x_t.size(1), self.pos_enc.shape[1])
#         x_emb  = (self.traj_embed(x_t[:, :T_seq, :])
#                   + self.pos_enc[:, :T_seq, :]
#                   + t_emb.unsqueeze(1))
#         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

#         v_neural = self.out_fc2(F.gelu(self.out_fc1(
#             self.transformer(x_emb, memory)
#         )))  # [B, T, 4]

#         with torch.no_grad():
#             v_phys = self._beta_drift_velocity(x_t[:, :T_seq, :])

#         scale = torch.sigmoid(self.physics_scale) * 2.0
#         return v_neural + scale * v_phys

#     def forward(self, x_t, t, batch_list):
#         raw = self._context(batch_list)
#         ctx = self._apply_ctx_head(raw, noise_scale=0.0)
#         return self._decode(x_t, t, ctx)

#     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale: float = 0.0):
#         ctx = self._apply_ctx_head(raw_ctx, noise_scale=noise_scale)
#         return self._decode(x_t, t, ctx)


# # ══════════════════════════════════════════════════════════════════════════════
# #  TCFlowMatching
# # ══════════════════════════════════════════════════════════════════════════════

# class TCFlowMatching(nn.Module):

#     def __init__(
#         self,
#         pred_len:             int   = 12,
#         obs_len:              int   = 8,
#         sigma_min:            float = 0.02,
#         n_train_ens:          int   = 4,
#         unet_in_ch:           int   = 13,
#         ctx_noise_scale:      float = 0.02,
#         initial_sample_sigma: float = 0.1,
#         **kwargs,
#     ):
#         super().__init__()
#         self.pred_len             = pred_len
#         self.obs_len              = obs_len
#         self.sigma_min            = sigma_min
#         self.n_train_ens          = n_train_ens
#         self.active_pred_len      = pred_len
#         self.ctx_noise_scale      = ctx_noise_scale
#         self.initial_sample_sigma = initial_sample_sigma
#         self.net = VelocityField(
#             pred_len   = pred_len,
#             obs_len    = obs_len,
#             sigma_min  = sigma_min,
#             unet_in_ch = unet_in_ch,
#         )

#     def set_curriculum_len(self, active_len: int) -> None:
#         """No-op — curriculum removed."""
#         pass

#     @staticmethod
#     def _to_rel(traj_gt, Me_gt, last_pos, last_Me):
#         return torch.cat(
#             [traj_gt - last_pos.unsqueeze(0),
#              Me_gt   - last_Me.unsqueeze(0)],
#             dim=-1,
#         ).permute(1, 0, 2)

#     @staticmethod
#     def _to_abs(rel, last_pos, last_Me):
#         d = rel.permute(1, 0, 2)
#         return (
#             last_pos.unsqueeze(0) + d[:, :, :2],
#             last_Me.unsqueeze(0)  + d[:, :, 2:],
#         )

#     def _cfm_noisy(self, x1):
#         B, device = x1.shape[0], x1.device
#         sm  = self.sigma_min
#         x0  = torch.randn_like(x1) * sm
#         t   = torch.rand(B, device=device)
#         te  = t.view(B, 1, 1)
#         x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
#         denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
#         target_vel = (x1 - (1.0 - sm) * x_t) / denom
#         return x_t, t, te, denom, target_vel

#     @staticmethod
#     def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
#         wind_norm = obs_Me[-1, :, 1].detach()
#         w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
#             torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
#             torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
#                         torch.full_like(wind_norm, 1.5))))
#         return w / w.mean().clamp(min=1e-6)

#     @staticmethod
#     def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
#         if torch.rand(1).item() > p:
#             return batch_list
#         aug = list(batch_list)
#         for idx in [0, 1, 2, 3]:
#             t = aug[idx]
#             if torch.is_tensor(t) and t.shape[-1] >= 1:
#                 t = t.clone()
#                 t[..., 0] = -t[..., 0]
#                 aug[idx] = t
#         return aug

#     @staticmethod
#     def _lat_shift_aug(batch_list: List, p: float = 0.2,
#                        max_shift: float = 0.1) -> List:
#         """
#         FIX-M24: Latitude shift augmentation.
#         Shift toàn bộ trajectory (obs + pred) cùng một lượng nhỏ theo latitude.
#         max_shift=0.1 normalised ~ 0.1 * 50/10 * 1° = 0.5° latitude ~ 55 km.
#         Không shift longitude để giữ seasonal/basin patterns.
#         """
#         if torch.rand(1).item() > p:
#             return batch_list
#         aug = list(batch_list)
#         device = batch_list[0].device if torch.is_tensor(batch_list[0]) else None
#         if device is None:
#             return batch_list
#         shift = (torch.rand(1, device=device) * 2 - 1) * max_shift
#         for idx in [0, 1, 2, 3]:
#             t = aug[idx]
#             if torch.is_tensor(t) and t.shape[-1] >= 2:
#                 t = t.clone()
#                 t[..., 1] = t[..., 1] + shift
#                 aug[idx] = t
#         return aug

#     @staticmethod
#     def _jitter_aug(batch_list: List, p: float = 0.3,
#                     std: float = 0.002) -> List:
#         """
#         FIX-M25: Jitter augmentation on trajectory.
#         std=0.002 normalised ~ 0.002 * 50/10 * 111 = 1.11 km.
#         Chỉ jitter obs và pred traj (idx 0,1), không jitter intensities (2,3).
#         """
#         if torch.rand(1).item() > p:
#             return batch_list
#         aug = list(batch_list)
#         for idx in [0, 1]:
#             t = aug[idx]
#             if torch.is_tensor(t) and t.shape[-1] >= 2:
#                 noise = torch.randn_like(t[..., :2]) * std
#                 t = t.clone()
#                 t[..., :2] = t[..., :2] + noise
#                 aug[idx] = t
#         return aug

#     def get_loss(self, batch_list: List,
#                  step_weight_alpha: float = 0.0) -> torch.Tensor:
#         """
#         FIX-M23: Nhận step_weight_alpha để val loss aligned với train loss.
#         """
#         return self.get_loss_breakdown(batch_list, step_weight_alpha)["total"]

#     def get_loss_breakdown(self, batch_list: List,
#                            step_weight_alpha: float = 0.0) -> Dict:
#         # Apply augmentations (chỉ trong training - model.training flag)
#         if self.training:
#             batch_list = self._lon_flip_aug(batch_list, p=0.3)
#             batch_list = self._lat_shift_aug(batch_list, p=0.2)  # FIX-M24
#             batch_list = self._jitter_aug(batch_list, p=0.3)     # FIX-M25

#         traj_gt  = batch_list[1]
#         Me_gt    = batch_list[8]
#         obs_t    = batch_list[0]
#         obs_Me   = batch_list[7]

#         try:
#             env_data = batch_list[13]
#         except (IndexError, TypeError):
#             env_data = None

#         lp, lm = obs_t[-1], obs_Me[-1]
#         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

#         raw_ctx     = self.net._context(batch_list)
#         intensity_w = self._intensity_weights(obs_Me)

#         x_t, t, te, denom, _ = self._cfm_noisy(x1)
#         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

#         samples: List[torch.Tensor] = []
#         for _ in range(self.n_train_ens):
#             xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
#             pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
#             x1_s  = xt_s + dens_s * pv_s
#             pa_s, _ = self._to_abs(x1_s, lp, lm)
#             samples.append(pa_s)
#         pred_samples = torch.stack(samples)

#         all_trajs_4d = pred_samples

#         l_fm_physics = fm_physics_consistency_loss(
#             pred_samples, gt_norm=traj_gt, last_pos=lp)

#         x1_pred = x_t + denom * pred_vel
#         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

#         pred_abs_deg = _denorm_to_deg(pred_abs)
#         traj_gt_deg  = _denorm_to_deg(traj_gt)
#         ref_deg      = _denorm_to_deg(lp)

#         breakdown = compute_total_loss(
#             pred_abs           = pred_abs_deg,
#             gt                 = traj_gt_deg,
#             ref                = ref_deg,
#             batch_list         = batch_list,
#             pred_samples       = pred_samples,
#             gt_norm            = traj_gt,
#             weights            = WEIGHTS,
#             intensity_w        = intensity_w,
#             env_data           = env_data,
#             step_weight_alpha  = step_weight_alpha,
#             all_trajs          = all_trajs_4d,
#         )

#         fm_phys_w = WEIGHTS.get("fm_physics", 0.3)
#         breakdown["total"]      = breakdown["total"] + fm_phys_w * l_fm_physics
#         breakdown["fm_physics"] = l_fm_physics.item()

#         return breakdown

#     # ── sample() ─────────────────────────────────────────────────────────────

#     @torch.no_grad()
#     def sample(
#         self,
#         batch_list: List,
#         num_ensemble: int = 50,
#         ddim_steps:   int = 20,
#         predict_csv:  Optional[str] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         lp  = batch_list[0][-1]
#         lm  = batch_list[7][-1]
#         B   = lp.shape[0]
#         device = lp.device
#         T   = self.pred_len
#         dt  = 1.0 / max(ddim_steps, 1)

#         raw_ctx = self.net._context(batch_list)

#         traj_s: List[torch.Tensor] = []
#         me_s:   List[torch.Tensor] = []

#         for k in range(num_ensemble):
#             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

#             for step in range(ddim_steps):
#                 t_b = torch.full((B,), step * dt, device=device)
#                 ns  = self.ctx_noise_scale if step == 0 else 0.0
#                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
#                 x_t = x_t + dt * vel

#             x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)
#             x_t = x_t.clamp(-3.0, 3.0)

#             tr, me = self._to_abs(x_t, lp, lm)
#             traj_s.append(tr)
#             me_s.append(me)

#         all_trajs = torch.stack(traj_s)
#         all_me    = torch.stack(me_s)
#         pred_mean = all_trajs.mean(0)
#         me_mean   = all_me.mean(0)

#         if predict_csv:
#             self._write_predict_csv(predict_csv, pred_mean, all_trajs)

#         return pred_mean, me_mean, all_trajs

#     # ── Physics correction ────────────────────────────────────────────────────

#     def _physics_correct(
#         self,
#         x_pred: torch.Tensor,
#         last_pos: torch.Tensor,
#         last_Me:  torch.Tensor,
#         n_steps:  int   = 5,
#         lr:       float = 0.002,
#     ) -> torch.Tensor:
#         with torch.enable_grad():
#             x = x_pred.detach().requires_grad_(True)
#             optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)

#             for _ in range(n_steps):
#                 optimizer.zero_grad()
#                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
#                 pred_deg    = _denorm_to_deg(pred_abs)

#                 l_speed = self._pinn_speed_constraint(pred_deg)
#                 l_accel = self._pinn_beta_plane_simplified(pred_deg)

#                 physics_loss = l_speed + 0.3 * l_accel
#                 physics_loss.backward()

#                 torch.nn.utils.clip_grad_norm_([x], max_norm=0.05)
#                 optimizer.step()

#         return x.detach()

#     @staticmethod
#     def _pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
#         if pred_deg.shape[0] < 2:
#             return pred_deg.new_zeros(())
#         dt_deg  = pred_deg[1:] - pred_deg[:-1]
#         lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
#         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
#         dx_km   = dt_deg[:, :, 0] * cos_lat * 111.0
#         dy_km   = dt_deg[:, :, 1] * 111.0
#         speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
#         return F.relu(speed - 600.0).pow(2).mean()

#     @staticmethod
#     def _pinn_beta_plane_simplified(pred_deg: torch.Tensor) -> torch.Tensor:
#         if pred_deg.shape[0] < 3:
#             return pred_deg.new_zeros(())
#         v = pred_deg[1:] - pred_deg[:-1]
#         a = v[1:] - v[:-1]
#         lat_rad = torch.deg2rad(pred_deg[1:-1, :, 1])
#         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
#         a_lon_km = a[:, :, 0] * cos_lat * 111.0
#         a_lat_km = a[:, :, 1] * 111.0
#         max_accel = 50.0
#         violation = F.relu(torch.sqrt(a_lon_km**2 + a_lat_km**2) - max_accel)
#         return violation.pow(2).mean() * 0.1

#     @staticmethod
#     def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
#                            all_trajs: torch.Tensor) -> None:
#         import numpy as np
#         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
#         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
#         T, B, _ = traj_mean.shape
#         S       = all_trajs.shape[0]

#         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
#         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
#         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
#         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()

#         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
#                     "lon_mean_deg", "lat_mean_deg",
#                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
#         write_hdr = not os.path.exists(csv_path)
#         with open(csv_path, "a", newline="") as fh:
#             w = csv.DictWriter(fh, fieldnames=fields)
#             if write_hdr:
#                 w.writeheader()
#             for b in range(B):
#                 for k in range(T):
#                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
#                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
#                         math.radians(mean_lat[k, b]))
#                     spread = float(np.sqrt((dlat**2 + dlon**2).mean()) * 111.0)
#                     w.writerow(dict(
#                         timestamp     = ts,
#                         batch_idx     = b,
#                         step_idx      = k,
#                         lead_h        = (k + 1) * 6,
#                         lon_mean_deg  = f"{mean_lon[k,b]:.4f}",
#                         lat_mean_deg  = f"{mean_lat[k,b]:.4f}",
#                         lon_std_deg   = f"{all_lon[:,k,b].std():.4f}",
#                         lat_std_deg   = f"{all_lat[:,k,b].std():.4f}",
#                         ens_spread_km = f"{spread:.2f}",
#                     ))
#         print(f"  Predictions → {csv_path}  (B={B}, T={T}, S={S})")


# # Backward-compat alias
# TCDiffusion = TCFlowMatching

"""
Model/flow_matching_model.py  ── v25
==========================================
ROOT CAUSE FIXES:

  FIX-M27  [P0-CRITICAL] _context(): khi img_obs toàn zeros (Data3d không
           load được), FNO3D vẫn chạy nhưng produce garbage bottleneck.
           Thêm img_obs_valid_flag để detect và zero-out spatial features
           khi input không có data, tránh poisoning context vector.
           Đồng thời add fallback: nếu spatial features = 0, tăng weight
           của 1D encoder để vẫn có gradient flow.

  FIX-M28  [P0-CRITICAL] n_train_ens: không dùng progressive schedule
           từ script. Model set n_train_ens=args.n_train_ens ngay từ đầu.
           ens=2 → AFCRPS chỉ có 1 pair → loss noisy → model không học
           distribution. Fix: trong get_loss_breakdown, dùng n_samples
           tối thiểu 4 bất kể n_train_ens setting.

  FIX-M29  [P1] _cfm_noisy(): x0 sigma từ sigma_min → 0.5.
           sigma_min=0.02 quá nhỏ → noise floor quá thấp → flow quá
           deterministic → ensemble sau khi sample bị collapse.
           initial_sample_sigma=0.1 trong sample() cũng quá nhỏ → samples
           bắt đầu quá gần nhau.
           Fix: x0 = randn * 0.5 (base noise level cao hơn).
           Trong sample(), initial_sample_sigma → 0.3.

  FIX-M30  [P1] _physics_correct(): n_steps=5, lr=0.002 → quá aggressive
           khi model chưa train. Physics correction distort trajectory
           trong early training. Fix: disable physics_correct khi sample,
           chỉ dùng clip thôi. Có thể re-enable sau ep=50.

  FIX-M31  [P2] sample(): thêm temperature parameter.
           temperature > 1.0 tăng diversity của samples.
           Default temperature=1.2 để ensemble có spread hợp lý.

  FIX-M32  [P2] get_loss_breakdown(): logging rõ ràng hơn về spatial
           feature quality để debug.

Kept:
  FIX-M23..M26 (val alignment, lat_shift_aug, jitter_aug, etc.)
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
from Model.losses import (
    compute_total_loss, fm_physics_consistency_loss, WEIGHTS,
    pinn_speed_constraint,
)

# Minimum ensemble size để AFCRPS có ý nghĩa
_MIN_TRAIN_ENS = 4


def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
    """Normalised → degrees. Handles [T, B, 2] and [B, 2]."""
    out = traj_norm.clone()
    out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
    out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
    return out


def _img_obs_valid_ratio(image_obs: torch.Tensor) -> float:
    """Check ratio of nonzero pixels để detect Data3d missing."""
    return (image_obs.abs() > 1e-6).float().mean().item()


# ══════════════════════════════════════════════════════════════════════════════
#  VelocityField
# ══════════════════════════════════════════════════════════════════════════════

class VelocityField(nn.Module):
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

        # FIX-M27: fallback projection khi spatial features = 0
        self.ctx_fc1  = nn.Linear(128 + 64 + 16, 512)
        self.ctx_ln   = nn.LayerNorm(512)
        self.ctx_drop = nn.Dropout(0.15)
        self.ctx_fc2  = nn.Linear(512, ctx_dim)

        # FIX-M27: 1D-only fallback path khi img_obs toàn zeros
        self.ctx_fc1_1d  = nn.Linear(128 + 64, 512)
        self.ctx_ln_1d   = nn.LayerNorm(512)
        self.ctx_fc2_1d  = nn.Linear(512, ctx_dim)

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

        self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)

    def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
        half = dim // 2
        freq = torch.exp(
            torch.arange(half, dtype=torch.float32, device=t.device)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return F.pad(emb, (0, dim % 2))

    def _context(self, batch_list: List) -> Tuple[torch.Tensor, bool]:
        """
        FIX-M27: Returns (raw_ctx, spatial_valid).
        spatial_valid=False khi img_obs toàn zeros → dùng 1D-only path.
        """
        obs_traj  = batch_list[0]
        obs_Me    = batch_list[7]
        image_obs = batch_list[11]
        env_data  = batch_list[13]

        if image_obs.dim() == 4:
            image_obs = image_obs.unsqueeze(1)

        expected_ch = self.spatial_enc.in_channel
        if image_obs.shape[1] == 1 and expected_ch != 1:
            image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

        # FIX-M27: check spatial data quality
        spatial_valid = _img_obs_valid_ratio(image_obs) > 0.05

        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)

        if spatial_valid:
            e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
            T_obs = obs_traj.shape[0]

            e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
            e_3d_s = e_3d_s.permute(0, 2, 1)
            e_3d_s = self.bottleneck_proj(e_3d_s)

            T_bot = e_3d_s.shape[1]
            if T_bot != T_obs:
                e_3d_s = F.interpolate(
                    e_3d_s.permute(0, 2, 1), size=T_obs,
                    mode="linear", align_corners=False,
                ).permute(0, 2, 1)

            f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))
            f_spatial     = self.decoder_proj(f_spatial_raw)

            h_t = self.enc_1d(obs_in, e_3d_s)
            e_env, _, _ = self.env_enc(env_data, image_obs)

            raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
            raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))
        else:
            # FIX-M27: 1D-only fallback — spatial zeros không contribute
            B = obs_traj.shape[1]
            device = obs_traj.device
            e_3d_s_dummy = torch.zeros(B, obs_traj.shape[0], 128, device=device)
            h_t = self.enc_1d(obs_in, e_3d_s_dummy)
            e_env, _, _ = self.env_enc(env_data, image_obs)
            raw = torch.cat([h_t, e_env], dim=-1)
            raw = F.gelu(self.ctx_ln_1d(self.ctx_fc1_1d(raw)))

        return raw, spatial_valid

    def _apply_ctx_head(self, raw: torch.Tensor, use_spatial: bool = True,
                        noise_scale: float = 0.0) -> torch.Tensor:
        if noise_scale > 0.0:
            raw = raw + torch.randn_like(raw) * noise_scale
        if use_spatial:
            return self.ctx_fc2(self.ctx_drop(raw))
        else:
            return self.ctx_fc2_1d(self.ctx_drop(raw))

    def _beta_drift_velocity(self, x_t: torch.Tensor) -> torch.Tensor:
        OMEGA_val  = 7.2921e-5
        R_val      = 6.371e6
        DT         = 6 * 3600.0
        M_PER_NORM = 5.0 * 111.0 * 1000.0

        lat_norm = x_t[:, :, 1]
        lat_deg  = lat_norm * 5.0
        lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))

        beta   = 2 * OMEGA_val * torch.cos(lat_rad) / R_val
        R_tc   = 3e5
        v_lon  = -beta * R_tc ** 2 / 2
        v_lat  =  beta * R_tc ** 2 / 4

        v_lon_norm = v_lon * DT / M_PER_NORM
        v_lat_norm = v_lat * DT / M_PER_NORM

        v_phys = torch.zeros_like(x_t)
        v_phys[:, :, 0] = v_lon_norm
        v_phys[:, :, 1] = v_lat_norm
        return v_phys

    def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
                ctx: torch.Tensor) -> torch.Tensor:
        t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
        t_emb = self.time_fc2(t_emb)

        T_seq  = min(x_t.size(1), self.pos_enc.shape[1])
        x_emb  = (self.traj_embed(x_t[:, :T_seq, :])
                  + self.pos_enc[:, :T_seq, :]
                  + t_emb.unsqueeze(1))
        memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

        v_neural = self.out_fc2(F.gelu(self.out_fc1(
            self.transformer(x_emb, memory)
        )))

        with torch.no_grad():
            v_phys = self._beta_drift_velocity(x_t[:, :T_seq, :])

        scale = torch.sigmoid(self.physics_scale) * 2.0
        return v_neural + scale * v_phys

    def forward(self, x_t, t, batch_list):
        raw, spatial_valid = self._context(batch_list)
        ctx = self._apply_ctx_head(raw, use_spatial=spatial_valid, noise_scale=0.0)
        return self._decode(x_t, t, ctx)

    def forward_with_ctx(self, x_t, t, raw_ctx, spatial_valid: bool = True,
                         noise_scale: float = 0.0):
        ctx = self._apply_ctx_head(raw_ctx, use_spatial=spatial_valid,
                                   noise_scale=noise_scale)
        return self._decode(x_t, t, ctx)


# ══════════════════════════════════════════════════════════════════════════════
#  TCFlowMatching
# ══════════════════════════════════════════════════════════════════════════════

class TCFlowMatching(nn.Module):

    def __init__(
        self,
        pred_len:             int   = 12,
        obs_len:              int   = 8,
        sigma_min:            float = 0.02,
        n_train_ens:          int   = 6,
        unet_in_ch:           int   = 13,
        ctx_noise_scale:      float = 0.02,
        initial_sample_sigma: float = 0.3,   # FIX-M29: 0.1 → 0.3
        **kwargs,
    ):
        super().__init__()
        self.pred_len             = pred_len
        self.obs_len              = obs_len
        self.sigma_min            = sigma_min
        self.n_train_ens          = max(n_train_ens, _MIN_TRAIN_ENS)
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
        pass

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
        """
        FIX-M29: x0 noise level 0.5 thay vì sigma_min.
        Cao hơn → ensemble có spread tốt hơn sau reverse process.
        """
        B, device = x1.shape[0], x1.device
        sm  = self.sigma_min
        # FIX-M29: base noise 0.5 thay vì sigma_min=0.02
        x0  = torch.randn_like(x1) * 0.5
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
        return aug

    @staticmethod
    def _lat_shift_aug(batch_list: List, p: float = 0.2,
                       max_shift: float = 0.1) -> List:
        if torch.rand(1).item() > p:
            return batch_list
        aug = list(batch_list)
        device = batch_list[0].device if torch.is_tensor(batch_list[0]) else None
        if device is None:
            return batch_list
        shift = (torch.rand(1, device=device) * 2 - 1) * max_shift
        for idx in [0, 1, 2, 3]:
            t = aug[idx]
            if torch.is_tensor(t) and t.shape[-1] >= 2:
                t = t.clone()
                t[..., 1] = t[..., 1] + shift
                aug[idx] = t
        return aug

    @staticmethod
    def _jitter_aug(batch_list: List, p: float = 0.3,
                    std: float = 0.002) -> List:
        if torch.rand(1).item() > p:
            return batch_list
        aug = list(batch_list)
        for idx in [0, 1]:
            t = aug[idx]
            if torch.is_tensor(t) and t.shape[-1] >= 2:
                noise = torch.randn_like(t[..., :2]) * std
                t = t.clone()
                t[..., :2] = t[..., :2] + noise
                aug[idx] = t
        return aug

    def get_loss(self, batch_list: List,
                 step_weight_alpha: float = 0.0) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list, step_weight_alpha)["total"]

    def get_loss_breakdown(self, batch_list: List,
                           step_weight_alpha: float = 0.0) -> Dict:
        if self.training:
            batch_list = self._lon_flip_aug(batch_list, p=0.3)
            batch_list = self._lat_shift_aug(batch_list, p=0.2)
            batch_list = self._jitter_aug(batch_list, p=0.3)

        traj_gt  = batch_list[1]
        Me_gt    = batch_list[8]
        obs_t    = batch_list[0]
        obs_Me   = batch_list[7]

        try:
            env_data = batch_list[13]
        except (IndexError, TypeError):
            env_data = None

        lp, lm = obs_t[-1], obs_Me[-1]
        x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

        # FIX-M27: get context with spatial validity flag
        raw_ctx, spatial_valid = self.net._context(batch_list)
        intensity_w = self._intensity_weights(obs_Me)

        x_t, t, te, denom, _ = self._cfm_noisy(x1)
        pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx,
                                             spatial_valid=spatial_valid,
                                             noise_scale=0.0)

        # FIX-M28: guarantee minimum ensemble size
        ens_size = max(self.n_train_ens, _MIN_TRAIN_ENS)

        samples: List[torch.Tensor] = []
        for _ in range(ens_size):
            xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
            pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx,
                                              spatial_valid=spatial_valid,
                                              noise_scale=0.0)
            x1_s  = xt_s + dens_s * pv_s
            pa_s, _ = self._to_abs(x1_s, lp, lm)
            samples.append(pa_s)
        pred_samples = torch.stack(samples)  # [ens_size, T, B, 2]

        all_trajs_4d = pred_samples

        l_fm_physics = fm_physics_consistency_loss(
            pred_samples, gt_norm=traj_gt, last_pos=lp)

        x1_pred = x_t + denom * pred_vel
        pred_abs, _ = self._to_abs(x1_pred, lp, lm)

        pred_abs_deg = _denorm_to_deg(pred_abs)
        traj_gt_deg  = _denorm_to_deg(traj_gt)
        ref_deg      = _denorm_to_deg(lp)

        breakdown = compute_total_loss(
            pred_abs           = pred_abs_deg,
            gt                 = traj_gt_deg,
            ref                = ref_deg,
            batch_list         = batch_list,
            pred_samples       = pred_samples,
            gt_norm            = traj_gt,
            weights            = WEIGHTS,
            intensity_w        = intensity_w,
            env_data           = env_data,
            step_weight_alpha  = step_weight_alpha,
            all_trajs          = all_trajs_4d,
        )

        fm_phys_w = WEIGHTS.get("fm_physics", 0.1)
        breakdown["total"]      = breakdown["total"] + fm_phys_w * l_fm_physics
        breakdown["fm_physics"] = l_fm_physics.item()
        breakdown["spatial_valid"] = float(spatial_valid)

        return breakdown

    # ── sample() ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        batch_list: List,
        num_ensemble: int = 50,
        ddim_steps:   int = 20,
        predict_csv:  Optional[str] = None,
        temperature:  float = 1.2,   # FIX-M31: diversity control
        epoch:        int   = 0,     # FIX-M30: physics correction từ ep50
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lp  = batch_list[0][-1]
        lm  = batch_list[7][-1]
        B   = lp.shape[0]
        device = lp.device
        T   = self.pred_len
        dt  = 1.0 / max(ddim_steps, 1)

        raw_ctx, spatial_valid = self.net._context(batch_list)

        traj_s: List[torch.Tensor] = []
        me_s:   List[torch.Tensor] = []

        for k in range(num_ensemble):
            # FIX-M29 + M31: higher initial sigma * temperature
            init_sigma = self.initial_sample_sigma * temperature
            x_t = torch.randn(B, T, 4, device=device) * init_sigma

            for step in range(ddim_steps):
                t_b = torch.full((B,), step * dt, device=device)
                ns  = self.ctx_noise_scale if step == 0 else 0.0
                vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx,
                                                spatial_valid=spatial_valid,
                                                noise_scale=ns)
                x_t = x_t + dt * vel

            # FIX-M30: physics correction chỉ sau epoch 50
            if epoch >= 50:
                x_t = self._physics_correct(x_t, lp, lm, n_steps=3, lr=0.001)

            x_t = x_t.clamp(-3.0, 3.0)

            tr, me = self._to_abs(x_t, lp, lm)
            traj_s.append(tr)
            me_s.append(me)

        all_trajs = torch.stack(traj_s)
        all_me    = torch.stack(me_s)
        pred_mean = all_trajs.mean(0)
        me_mean   = all_me.mean(0)

        if predict_csv:
            self._write_predict_csv(predict_csv, pred_mean, all_trajs)

        return pred_mean, me_mean, all_trajs

    # ── Physics correction ────────────────────────────────────────────────────

    def _physics_correct(
        self,
        x_pred: torch.Tensor,
        last_pos: torch.Tensor,
        last_Me:  torch.Tensor,
        n_steps:  int   = 3,
        lr:       float = 0.001,
    ) -> torch.Tensor:
        with torch.enable_grad():
            x = x_pred.detach().requires_grad_(True)
            optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)
            for _ in range(n_steps):
                optimizer.zero_grad()
                pred_abs, _ = self._to_abs(x, last_pos, last_Me)
                pred_deg    = _denorm_to_deg(pred_abs)
                l_speed = self._pinn_speed_constraint(pred_deg)
                l_accel = self._pinn_beta_plane_simplified(pred_deg)
                physics_loss = l_speed + 0.3 * l_accel
                physics_loss.backward()
                torch.nn.utils.clip_grad_norm_([x], max_norm=0.05)
                optimizer.step()
        return x.detach()

    @staticmethod
    def _pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
        if pred_deg.shape[0] < 2:
            return pred_deg.new_zeros(())
        dt_deg  = pred_deg[1:] - pred_deg[:-1]
        lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
        cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
        dx_km   = dt_deg[:, :, 0] * cos_lat * 111.0
        dy_km   = dt_deg[:, :, 1] * 111.0
        speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
        return F.relu(speed - 450.0).pow(2).mean()

    @staticmethod
    def _pinn_beta_plane_simplified(pred_deg: torch.Tensor) -> torch.Tensor:
        if pred_deg.shape[0] < 3:
            return pred_deg.new_zeros(())
        v = pred_deg[1:] - pred_deg[:-1]
        a = v[1:] - v[:-1]
        lat_rad = torch.deg2rad(pred_deg[1:-1, :, 1])
        cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
        a_lon_km = a[:, :, 0] * cos_lat * 111.0
        a_lat_km = a[:, :, 1] * 111.0
        max_accel = 50.0
        violation = F.relu(torch.sqrt(a_lon_km**2 + a_lat_km**2) - max_accel)
        return violation.pow(2).mean() * 0.1

    @staticmethod
    def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
                           all_trajs: torch.Tensor) -> None:
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
                    dlat   = all_lat[:, k, b] - mean_lat[k, b]
                    dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
                        math.radians(mean_lat[k, b]))
                    spread = float(np.sqrt((dlat**2 + dlon**2).mean()) * 111.0)
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


# Backward-compat alias
TCDiffusion = TCFlowMatching
# # # # """
# # # # Model/flow_matching_model.py  ── v23
# # # # ==========================================
# # # # FIXES vs v21:

# # # #   FIX-M18  [CURRICULUM REMOVED] set_curriculum_len() vẫn giữ để backward
# # # #            compat nhưng KHÔNG được gọi từ trainer nữa. active_pred_len
# # # #            luôn = pred_len. evaluate_full_val_ade không cần restore nữa.

# # # #   FIX-M19  get_loss_breakdown(): nhận thêm step_weight_alpha parameter
# # # #            và truyền vào compute_total_loss() → fm_afcrps_loss() sử dụng
# # # #            soft weighting thay curriculum len-slicing.

# # # #   FIX-M20  get_loss_breakdown(): truyền all_trajs vào compute_total_loss()
# # # #            để tính ensemble_spread_loss. Giúp kiểm soát spread tăng quá mức.

# # # #   FIX-M21  _physics_correct(): tăng n_steps=5 (từ 3), giảm lr=0.002 (từ
# # # #            0.005) để physics correction ổn định hơn và ít overshoot.

# # # #   FIX-M22  sample(): initial_sample_sigma=0.1 (set từ constructor) đã fix
# # # #            spread. Thêm post-sampling clip chặt hơn [-3.0, 3.0] cho cả lon
# # # #            và lat (từ [-5.0, 5.0] cho lon).

# # # # Kept from v21:
# # # #   FIX-M17  _physics_correct với torch.enable_grad()
# # # #   FIX-M11..M16 OT-CFM, beta drift, env_data, physics scale
# # # # """
# # # # from __future__ import annotations

# # # # import csv
# # # # import math
# # # # import os
# # # # from datetime import datetime
# # # # from typing import Dict, List, Optional, Tuple

# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.nn.functional as F

# # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # from Model.env_net_transformer_gphsplit import Env_net
# # # # from Model.losses import (
# # # #     compute_total_loss, fm_physics_consistency_loss, WEIGHTS,
# # # #     pinn_speed_constraint,
# # # # )


# # # # def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
# # # #     """Normalised → degrees. Handles [T, B, 2] and [B, 2]."""
# # # #     out = traj_norm.clone()
# # # #     out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
# # # #     out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
# # # #     return out


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  VelocityField
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class VelocityField(nn.Module):
# # # #     """
# # # #     OT-CFM velocity field v_θ(x_t, t, context).
# # # #     Architecture: DataEncoder1D (Mamba) + FNO3D + Env-T-Net → Transformer decoder.
# # # #     Physics-guided: v_total = v_neural + sigmoid(w_physics) * v_beta_drift.
# # # #     """

# # # #     def __init__(
# # # #         self,
# # # #         pred_len:   int   = 12,
# # # #         obs_len:    int   = 8,
# # # #         ctx_dim:    int   = 256,
# # # #         sigma_min:  float = 0.02,
# # # #         unet_in_ch: int   = 13,
# # # #     ):
# # # #         super().__init__()
# # # #         self.pred_len  = pred_len
# # # #         self.obs_len   = obs_len
# # # #         self.sigma_min = sigma_min

# # # #         self.spatial_enc = FNO3DEncoder(
# # # #             in_channel   = unet_in_ch,
# # # #             out_channel  = 1,
# # # #             d_model      = 64,
# # # #             n_layers     = 4,
# # # #             modes_t      = 4,
# # # #             modes_h      = 4,
# # # #             modes_w      = 4,
# # # #             spatial_down = 32,
# # # #             dropout      = 0.05,
# # # #         )

# # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # #         self.decoder_proj    = nn.Linear(1, 16)

# # # #         self.enc_1d = DataEncoder1D(
# # # #             in_1d       = 4,
# # # #             feat_3d_dim = 128,
# # # #             mlp_h       = 64,
# # # #             lstm_hidden = 128,
# # # #             lstm_layers = 3,
# # # #             dropout     = 0.1,
# # # #             d_state     = 16,
# # # #         )

# # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=64)

# # # #         self.ctx_fc1  = nn.Linear(128 + 64 + 16, 512)
# # # #         self.ctx_ln   = nn.LayerNorm(512)
# # # #         self.ctx_drop = nn.Dropout(0.15)
# # # #         self.ctx_fc2  = nn.Linear(512, ctx_dim)

# # # #         self.time_fc1 = nn.Linear(256, 512)
# # # #         self.time_fc2 = nn.Linear(512, 256)

# # # #         self.traj_embed = nn.Linear(4, 256)
# # # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # # #         self.transformer = nn.TransformerDecoder(
# # # #             nn.TransformerDecoderLayer(
# # # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # # #                 dropout=0.15, activation="gelu", batch_first=True,
# # # #             ),
# # # #             num_layers=4,
# # # #         )
# # # #         self.out_fc1 = nn.Linear(256, 512)
# # # #         self.out_fc2 = nn.Linear(512, 4)

# # # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)

# # # #     def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
# # # #         half = dim // 2
# # # #         freq = torch.exp(
# # # #             torch.arange(half, dtype=torch.float32, device=t.device)
# # # #             * (-math.log(10_000.0) / max(half - 1, 1))
# # # #         )
# # # #         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
# # # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # # #         return F.pad(emb, (0, dim % 2))

# # # #     def _context(self, batch_list: List) -> torch.Tensor:
# # # #         obs_traj  = batch_list[0]
# # # #         obs_Me    = batch_list[7]
# # # #         image_obs = batch_list[11]
# # # #         env_data  = batch_list[13]

# # # #         if image_obs.dim() == 4:
# # # #             image_obs = image_obs.unsqueeze(1)

# # # #         expected_ch = self.spatial_enc.in_channel
# # # #         if image_obs.shape[1] == 1 and expected_ch != 1:
# # # #             image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # #         T_obs = obs_traj.shape[0]

# # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
# # # #         e_3d_s = e_3d_s.permute(0, 2, 1)
# # # #         e_3d_s = self.bottleneck_proj(e_3d_s)

# # # #         T_bot = e_3d_s.shape[1]
# # # #         if T_bot != T_obs:
# # # #             e_3d_s = F.interpolate(
# # # #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# # # #                 mode="linear", align_corners=False,
# # # #             ).permute(0, 2, 1)

# # # #         f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))
# # # #         f_spatial     = self.decoder_proj(f_spatial_raw)

# # # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # # #         h_t    = self.enc_1d(obs_in, e_3d_s)

# # # #         e_env, _, _ = self.env_enc(env_data, image_obs)

# # # #         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
# # # #         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))
# # # #         return raw

# # # #     def _apply_ctx_head(self, raw: torch.Tensor,
# # # #                         noise_scale: float = 0.0) -> torch.Tensor:
# # # #         if noise_scale > 0.0:
# # # #             raw = raw + torch.randn_like(raw) * noise_scale
# # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # #     def _beta_drift_velocity(self, x_t: torch.Tensor) -> torch.Tensor:
# # # #         """Beta drift in normalised state space. x_t: [B, T, 4]."""
# # # #         OMEGA_val  = 7.2921e-5
# # # #         R_val      = 6.371e6
# # # #         DT         = 6 * 3600.0
# # # #         M_PER_NORM = 5.0 * 111.0 * 1000.0

# # # #         lat_norm = x_t[:, :, 1]
# # # #         lat_deg  = lat_norm * 5.0
# # # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))

# # # #         beta   = 2 * OMEGA_val * torch.cos(lat_rad) / R_val
# # # #         R_tc   = 3e5
# # # #         v_lon  = -beta * R_tc ** 2 / 2
# # # #         v_lat  =  beta * R_tc ** 2 / 4

# # # #         v_lon_norm = v_lon * DT / M_PER_NORM
# # # #         v_lat_norm = v_lat * DT / M_PER_NORM

# # # #         v_phys = torch.zeros_like(x_t)
# # # #         v_phys[:, :, 0] = v_lon_norm
# # # #         v_phys[:, :, 1] = v_lat_norm
# # # #         return v_phys

# # # #     def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
# # # #                 ctx: torch.Tensor) -> torch.Tensor:
# # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
# # # #         t_emb = self.time_fc2(t_emb)

# # # #         T_seq  = min(x_t.size(1), self.pos_enc.shape[1])
# # # #         x_emb  = (self.traj_embed(x_t[:, :T_seq, :])
# # # #                   + self.pos_enc[:, :T_seq, :]
# # # #                   + t_emb.unsqueeze(1))
# # # #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

# # # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(
# # # #             self.transformer(x_emb, memory)
# # # #         )))  # [B, T, 4]

# # # #         with torch.no_grad():
# # # #             v_phys = self._beta_drift_velocity(x_t[:, :T_seq, :])

# # # #         scale = torch.sigmoid(self.physics_scale) * 2.0
# # # #         return v_neural + scale * v_phys

# # # #     def forward(self, x_t, t, batch_list):
# # # #         raw = self._context(batch_list)
# # # #         ctx = self._apply_ctx_head(raw, noise_scale=0.0)
# # # #         return self._decode(x_t, t, ctx)

# # # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale: float = 0.0):
# # # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale=noise_scale)
# # # #         return self._decode(x_t, t, ctx)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  TCFlowMatching
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class TCFlowMatching(nn.Module):
# # # #     """TC trajectory prediction via OT-CFM + Physics-guided velocity field."""

# # # #     def __init__(
# # # #         self,
# # # #         pred_len:             int   = 12,
# # # #         obs_len:              int   = 8,
# # # #         sigma_min:            float = 0.02,
# # # #         n_train_ens:          int   = 4,
# # # #         unet_in_ch:           int   = 13,
# # # #         ctx_noise_scale:      float = 0.02,   # FIX-T23-5: 0.02 default
# # # #         initial_sample_sigma: float = 0.1,    # FIX-T23-4: 0.1 default
# # # #         **kwargs,
# # # #     ):
# # # #         super().__init__()
# # # #         self.pred_len             = pred_len
# # # #         self.obs_len              = obs_len
# # # #         self.sigma_min            = sigma_min
# # # #         self.n_train_ens          = n_train_ens
# # # #         self.active_pred_len      = pred_len   # FIX-M18: always full pred_len
# # # #         self.ctx_noise_scale      = ctx_noise_scale
# # # #         self.initial_sample_sigma = initial_sample_sigma
# # # #         self.net = VelocityField(
# # # #             pred_len   = pred_len,
# # # #             obs_len    = obs_len,
# # # #             sigma_min  = sigma_min,
# # # #             unet_in_ch = unet_in_ch,
# # # #         )

# # # #     def set_curriculum_len(self, active_len: int) -> None:
# # # #         """
# # # #         FIX-M18: Kept for backward compat but NO-OP in v23.
# # # #         Curriculum is removed. active_pred_len is always pred_len.
# # # #         """
# # # #         # self.active_pred_len = max(1, min(active_len, self.pred_len))
# # # #         pass  # no-op

# # # #     @staticmethod
# # # #     def _to_rel(traj_gt, Me_gt, last_pos, last_Me):
# # # #         return torch.cat(
# # # #             [traj_gt - last_pos.unsqueeze(0),
# # # #              Me_gt   - last_Me.unsqueeze(0)],
# # # #             dim=-1,
# # # #         ).permute(1, 0, 2)

# # # #     @staticmethod
# # # #     def _to_abs(rel, last_pos, last_Me):
# # # #         d = rel.permute(1, 0, 2)
# # # #         return (
# # # #             last_pos.unsqueeze(0) + d[:, :, :2],
# # # #             last_Me.unsqueeze(0)  + d[:, :, 2:],
# # # #         )

# # # #     def _cfm_noisy(self, x1):
# # # #         B, device = x1.shape[0], x1.device
# # # #         sm  = self.sigma_min
# # # #         x0  = torch.randn_like(x1) * sm
# # # #         t   = torch.rand(B, device=device)
# # # #         te  = t.view(B, 1, 1)
# # # #         x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
# # # #         denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
# # # #         target_vel = (x1 - (1.0 - sm) * x_t) / denom
# # # #         return x_t, t, te, denom, target_vel

# # # #     @staticmethod
# # # #     def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
# # # #         wind_norm = obs_Me[-1, :, 1].detach()
# # # #         w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
# # # #             torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
# # # #             torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
# # # #                         torch.full_like(wind_norm, 1.5))))
# # # #         return w / w.mean().clamp(min=1e-6)

# # # #     @staticmethod
# # # #     def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
# # # #         if torch.rand(1).item() > p:
# # # #             return batch_list
# # # #         aug = list(batch_list)
# # # #         for idx in [0, 1, 2, 3]:
# # # #             t = aug[idx]
# # # #             if torch.is_tensor(t) and t.shape[-1] >= 1:
# # # #                 t = t.clone()
# # # #                 t[..., 0] = -t[..., 0]
# # # #                 aug[idx] = t
# # # #         return aug

# # # #     def get_loss(self, batch_list: List,
# # # #                  step_weight_alpha: float = 0.0) -> torch.Tensor:
# # # #         return self.get_loss_breakdown(batch_list, step_weight_alpha)["total"]

# # # #     def get_loss_breakdown(self, batch_list: List,
# # # #                            step_weight_alpha: float = 0.0) -> Dict:
# # # #         """
# # # #         FIX-M19: Nhận step_weight_alpha, truyền vào compute_total_loss.
# # # #         FIX-M20: Truyền all_trajs để tính ensemble_spread_loss.
# # # #         """
# # # #         batch_list = self._lon_flip_aug(batch_list, p=0.3)

# # # #         traj_gt  = batch_list[1]
# # # #         Me_gt    = batch_list[8]
# # # #         obs_t    = batch_list[0]
# # # #         obs_Me   = batch_list[7]

# # # #         try:
# # # #             env_data = batch_list[13]
# # # #         except (IndexError, TypeError):
# # # #             env_data = None

# # # #         # FIX-M18: NO curriculum slicing. Always use full pred_len.
# # # #         lp, lm = obs_t[-1], obs_Me[-1]
# # # #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

# # # #         raw_ctx     = self.net._context(batch_list)
# # # #         intensity_w = self._intensity_weights(obs_Me)

# # # #         x_t, t, te, denom, _ = self._cfm_noisy(x1)
# # # #         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

# # # #         # Ensemble samples for AFCRPS + spread penalty
# # # #         samples: List[torch.Tensor] = []
# # # #         for _ in range(self.n_train_ens):
# # # #             xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
# # # #             pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
# # # #             x1_s  = xt_s + dens_s * pv_s   # OT-CFM
# # # #             pa_s, _ = self._to_abs(x1_s, lp, lm)
# # # #             samples.append(pa_s)
# # # #         pred_samples = torch.stack(samples)   # [S, T, B, 2]

# # # #         # FIX-M20: all_trajs for spread penalty
# # # #         all_trajs_4d = pred_samples   # [S, T, B, 2]

# # # #         l_fm_physics = fm_physics_consistency_loss(
# # # #             pred_samples, gt_norm=traj_gt, last_pos=lp)

# # # #         x1_pred = x_t + denom * pred_vel
# # # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# # # #         pred_abs_deg = _denorm_to_deg(pred_abs)
# # # #         traj_gt_deg  = _denorm_to_deg(traj_gt)
# # # #         ref_deg      = _denorm_to_deg(lp)

# # # #         breakdown = compute_total_loss(
# # # #             pred_abs           = pred_abs_deg,
# # # #             gt                 = traj_gt_deg,
# # # #             ref                = ref_deg,
# # # #             batch_list         = batch_list,
# # # #             pred_samples       = pred_samples,
# # # #             gt_norm            = traj_gt,
# # # #             weights            = WEIGHTS,
# # # #             intensity_w        = intensity_w,
# # # #             env_data           = env_data,
# # # #             step_weight_alpha  = step_weight_alpha,   # FIX-M19
# # # #             all_trajs          = all_trajs_4d,         # FIX-M20
# # # #         )

# # # #         fm_phys_w = WEIGHTS.get("fm_physics", 0.3)
# # # #         breakdown["total"]      = breakdown["total"] + fm_phys_w * l_fm_physics
# # # #         breakdown["fm_physics"] = l_fm_physics.item()

# # # #         return breakdown

# # # #     # ── sample() ─────────────────────────────────────────────────────────────

# # # #     @torch.no_grad()
# # # #     def sample(
# # # #         self,
# # # #         batch_list: List,
# # # #         num_ensemble: int = 50,
# # # #         ddim_steps:   int = 20,
# # # #         predict_csv:  Optional[str] = None,
# # # #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# # # #         """
# # # #         FIX-T23-3: ddim_steps default 20 (từ 10).
# # # #         FIX-M22: tighter clip [-3.0, 3.0] cho cả lon và lat.
# # # #         Returns:
# # # #             pred_mean:  [T, B, 2] mean track (normalised)
# # # #             me_mean:    [T, B, 2] mean intensity
# # # #             all_trajs:  [S, T, B, 2] all ensemble members
# # # #         """
# # # #         lp  = batch_list[0][-1]   # [B, 2]
# # # #         lm  = batch_list[7][-1]   # [B, 2]
# # # #         B   = lp.shape[0]
# # # #         device = lp.device
# # # #         T   = self.pred_len   # FIX-M18: always full pred_len
# # # #         dt  = 1.0 / max(ddim_steps, 1)

# # # #         raw_ctx = self.net._context(batch_list)

# # # #         traj_s: List[torch.Tensor] = []
# # # #         me_s:   List[torch.Tensor] = []

# # # #         for k in range(num_ensemble):
# # # #             # FIX-T23-4: initial_sample_sigma=0.1 (set in constructor)
# # # #             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# # # #             # DDIM Euler integration
# # # #             for step in range(ddim_steps):
# # # #                 t_b = torch.full((B,), step * dt, device=device)
# # # #                 ns  = self.ctx_noise_scale if step == 0 else 0.0
# # # #                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
# # # #                 x_t = x_t + dt * vel

# # # #             # Physics correction
# # # #             x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)

# # # #             # FIX-M22: tighter clip
# # # #             x_t = x_t.clamp(-3.0, 3.0)

# # # #             tr, me = self._to_abs(x_t, lp, lm)
# # # #             traj_s.append(tr)
# # # #             me_s.append(me)

# # # #         all_trajs = torch.stack(traj_s)   # [S, T, B, 2]
# # # #         all_me    = torch.stack(me_s)
# # # #         pred_mean = all_trajs.mean(0)
# # # #         me_mean   = all_me.mean(0)

# # # #         if predict_csv:
# # # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)

# # # #         return pred_mean, me_mean, all_trajs

# # # #     # ── Physics correction ────────────────────────────────────────────────────

# # # #     def _physics_correct(
# # # #         self,
# # # #         x_pred: torch.Tensor,
# # # #         last_pos: torch.Tensor,
# # # #         last_Me:  torch.Tensor,
# # # #         n_steps:  int   = 5,    # FIX-M21: 5 (từ 3)
# # # #         lr:       float = 0.002, # FIX-M21: 0.002 (từ 0.005)
# # # #     ) -> torch.Tensor:
# # # #         """
# # # #         FIX-M17: torch.enable_grad() inside no_grad context.
# # # #         FIX-M21: n_steps=5, lr=0.002 for more stable correction.
# # # #         """
# # # #         with torch.enable_grad():
# # # #             x = x_pred.detach().requires_grad_(True)
# # # #             optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)

# # # #             for _ in range(n_steps):
# # # #                 optimizer.zero_grad()
# # # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # # #                 pred_deg    = _denorm_to_deg(pred_abs)

# # # #                 l_speed = self._pinn_speed_constraint(pred_deg)
# # # #                 l_accel = self._pinn_beta_plane_simplified(pred_deg)

# # # #                 physics_loss = l_speed + 0.3 * l_accel
# # # #                 physics_loss.backward()

# # # #                 torch.nn.utils.clip_grad_norm_([x], max_norm=0.05)
# # # #                 optimizer.step()

# # # #         return x.detach()

# # # #     @staticmethod
# # # #     def _pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # # #         if pred_deg.shape[0] < 2:
# # # #             return pred_deg.new_zeros(())
# # # #         dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # # #         lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # #         dx_km   = dt_deg[:, :, 0] * cos_lat * 111.0
# # # #         dy_km   = dt_deg[:, :, 1] * 111.0
# # # #         speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # # #         return F.relu(speed - 600.0).pow(2).mean()

# # # #     @staticmethod
# # # #     def _pinn_beta_plane_simplified(pred_deg: torch.Tensor) -> torch.Tensor:
# # # #         if pred_deg.shape[0] < 3:
# # # #             return pred_deg.new_zeros(())
# # # #         v = pred_deg[1:] - pred_deg[:-1]
# # # #         a = v[1:] - v[:-1]
# # # #         lat_rad = torch.deg2rad(pred_deg[1:-1, :, 1])
# # # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # # #         a_lon_km = a[:, :, 0] * cos_lat * 111.0
# # # #         a_lat_km = a[:, :, 1] * 111.0
# # # #         max_accel = 50.0
# # # #         violation = F.relu(torch.sqrt(a_lon_km**2 + a_lat_km**2) - max_accel)
# # # #         return violation.pow(2).mean() * 0.1

# # # #     @staticmethod
# # # #     def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
# # # #                            all_trajs: torch.Tensor) -> None:
# # # #         import numpy as np
# # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # #         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # #         T, B, _ = traj_mean.shape
# # # #         S       = all_trajs.shape[0]

# # # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()

# # # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # # #                     "lon_mean_deg", "lat_mean_deg",
# # # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # # #         write_hdr = not os.path.exists(csv_path)
# # # #         with open(csv_path, "a", newline="") as fh:
# # # #             w = csv.DictWriter(fh, fieldnames=fields)
# # # #             if write_hdr:
# # # #                 w.writeheader()
# # # #             for b in range(B):
# # # #                 for k in range(T):
# # # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
# # # #                         math.radians(mean_lat[k, b]))
# # # #                     spread = float(np.sqrt((dlat**2 + dlon**2).mean()) * 111.0)
# # # #                     w.writerow(dict(
# # # #                         timestamp     = ts,
# # # #                         batch_idx     = b,
# # # #                         step_idx      = k,
# # # #                         lead_h        = (k + 1) * 6,
# # # #                         lon_mean_deg  = f"{mean_lon[k,b]:.4f}",
# # # #                         lat_mean_deg  = f"{mean_lat[k,b]:.4f}",
# # # #                         lon_std_deg   = f"{all_lon[:,k,b].std():.4f}",
# # # #                         lat_std_deg   = f"{all_lat[:,k,b].std():.4f}",
# # # #                         ens_spread_km = f"{spread:.2f}",
# # # #                     ))
# # # #         print(f"  Predictions → {csv_path}  (B={B}, T={T}, S={S})")


# # # # # Backward-compat alias
# # # # TCDiffusion = TCFlowMatching

# # # """
# # # Model/flow_matching_model.py  ── v24
# # # ==========================================
# # # FIXES vs v23:

# # #   FIX-M23  [SHORT-RANGE HEAD] Thêm ShortRangeHead – bộ dự báo riêng
# # #            cho 4 bước đầu (6h, 12h, 18h, 24h) dùng GRU autoregressive
# # #            + motion prior từ obs_traj. KHÔNG dùng flow matching cho
# # #            các bước này → ổn định hơn, chính xác hơn cho tầm ngắn.

# # #   FIX-M24  [INFERENCE BLEND] sample() dùng ShortRangeHead (deterministic)
# # #            cho steps 1-4, FM ensemble cho steps 5-12. pred_mean[:4]
# # #            luôn đến từ ShortRangeHead → giảm ADE 12h/24h đáng kể.

# # #   FIX-M25  [SPREAD CONTROL] initial_sample_sigma=0.03, ctx_noise_scale=0.002
# # #            → spread ensemble giảm từ 400-500km xuống mục tiêu <150km.

# # #   FIX-M26  [PHYSICS IN HEAD] ShortRangeHead clamp delta per-step ≤ MAX_DELTA
# # #            (~600km/6h), built-in vào forward pass → không cần PINN
# # #            cho tầm ngắn.

# # #   FIX-M27  [LOSS SEPARATION] get_loss_breakdown() tính thêm short_range_loss
# # #            riêng với weight cao (5.0). FM loss vẫn tính đầy đủ 12 bước
# # #            nhưng short_range override pred_mean cho bước 1-4.

# # # Kept from v23:
# # #   FIX-M18  active_pred_len = pred_len (no curriculum)
# # #   FIX-M21  _physics_correct n_steps=5, lr=0.002
# # #   FIX-M22  post-sampling clip [-3.0, 3.0]
# # # """
# # # from __future__ import annotations

# # # import csv
# # # import math
# # # import os
# # # from datetime import datetime
# # # from typing import Dict, List, Optional, Tuple

# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F

# # # from Model.FNO3D_encoder import FNO3DEncoder
# # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # from Model.env_net_transformer_gphsplit import Env_net
# # # from Model.losses import (
# # #     compute_total_loss, fm_physics_consistency_loss, WEIGHTS,
# # #     pinn_speed_constraint, short_range_regression_loss,
# # # )


# # # def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
# # #     """Normalised → degrees. Handles [T, B, 2] and [B, 2]."""
# # #     out = traj_norm.clone()
# # #     out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
# # #     out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
# # #     return out


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  ShortRangeHead  – FIX-M23
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class ShortRangeHead(nn.Module):
# # #     """
# # #     Dedicated deterministic predictor for steps 1-4 (6h/12h/18h/24h).

# # #     Architecture: GRU autoregressive cell
# # #         Input  each step : [ctx_feat(256) | vel_feat(128) | cur_pos(2)] = 386
# # #         Hidden           : 256
# # #         Output each step : delta [2] → clamp → new position

# # #     Physics built-in:
# # #         • MAX_DELTA = 0.48  ≈  576 km / 6h  (upper TC speed bound)
# # #         • Speed is implicitly constrained by architecture + clamp

# # #     Why GRU (not Transformer):
# # #         • Autoregressive dependency: step t depends on step t-1 position
# # #         • Small number of steps (4) → GRU is simpler and faster
# # #         • Motion prior via velocity encoding stabilises early epochs
# # #     """

# # #     N_STEPS   = 4        # 6h, 12h, 18h, 24h
# # #     MAX_DELTA = 0.48     # normalised units ≈ 576 km / 6h

# # #     def __init__(self, raw_ctx_dim=512, obs_len=8):
# # #         super().__init__()
# # #         self.obs_len = obs_len

# # #         # Motion prior: dùng tất cả obs velocities, không chỉ 3 bước cuối
# # #         self.vel_enc = nn.Sequential(
# # #             nn.Linear(obs_len * 2, 256),   # FIX: toàn bộ obs_len velocities
# # #             nn.GELU(),
# # #             nn.LayerNorm(256),
# # #             nn.Linear(256, 128),
# # #             nn.GELU(),
# # #         )

# # #         self.ctx_proj = nn.Sequential(
# # #             nn.Linear(raw_ctx_dim, 256),
# # #             nn.GELU(),
# # #             nn.LayerNorm(256),
# # #             nn.Dropout(0.05),   # FIX: giảm dropout 0.08→0.05
# # #         )

# # #         # FIX: GRU hidden = ctx+vel kết hợp ngay từ đầu
# # #         # input = ctx(256) + vel(128) + cur_pos(2) = 386
# # #         self.gru_cell = nn.GRUCell(input_size=386, hidden_size=256)

# # #         # FIX: thêm residual connection từ motion prior
# # #         self.motion_gate = nn.Sequential(
# # #             nn.Linear(128 + 256, 1),
# # #             nn.Sigmoid(),
# # #         )

# # #         self.out_head = nn.Sequential(
# # #             nn.Linear(256, 128),
# # #             nn.GELU(),
# # #             nn.LayerNorm(128),
# # #             nn.Linear(128, 2),
# # #         )

# # #         # Per-step scale được học, khởi tạo nhỏ để ổn định
# # #         self.step_scale = nn.Parameter(torch.ones(self.N_STEPS) * 0.2)
# # #         self._init_weights()

# # #     def _init_weights(self):
# # #         with torch.no_grad():
# # #             for m in self.modules():
# # #                 if isinstance(m, nn.Linear):
# # #                     nn.init.xavier_uniform_(m.weight, gain=0.3)
# # #                     if m.bias is not None:
# # #                         nn.init.zeros_(m.bias)

# # #     def forward(self, raw_ctx, obs_traj):
# # #         # obs_traj: [T_obs, B, 2]
# # #         B        = raw_ctx.shape[0]
# # #         T_obs    = obs_traj.shape[0]

# # #         # FIX: encode toàn bộ velocity sequence
# # #         if T_obs >= 2:
# # #             vels = obs_traj[1:] - obs_traj[:-1]   # [T_obs-1, B, 2]
# # #         else:
# # #             vels = torch.zeros(1, B, 2, device=raw_ctx.device)

# # #         # Pad hoặc crop về obs_len velocity steps
# # #         target_v = self.obs_len  # số velocity steps cần
# # #         if vels.shape[0] < target_v:
# # #             pad = torch.zeros(target_v - vels.shape[0], B, 2,
# # #                               device=raw_ctx.device)
# # #             vels = torch.cat([pad, vels], dim=0)
# # #         else:
# # #             vels = vels[-target_v:]   # lấy target_v bước cuối

# # #         vel_input = vels.permute(1, 0, 2).reshape(B, -1)  # [B, obs_len*2]
# # #         vel_feat  = self.vel_enc(vel_input)                 # [B, 128]

# # #         ctx_feat = self.ctx_proj(raw_ctx)  # [B, 256]

# # #         # FIX: hx được khởi tạo từ ctx thay vì chính ctx
# # #         # để GRU có không gian học riêng
# # #         hx      = ctx_feat
# # #         cur_pos = obs_traj[-1].clone()   # [B, 2]

# # #         # Last observed velocity làm prior bước đầu
# # #         last_vel = vels[-1].clone()      # [B, 2]

# # #         preds = []
# # #         for i in range(self.N_STEPS):
# # #             inp = torch.cat([ctx_feat, vel_feat, cur_pos], dim=-1)  # [B,386]
# # #             hx  = self.gru_cell(inp, hx)                             # [B,256]

# # #             # FIX: motion gate — blend giữa GRU output và motion prior
# # #             gate = self.motion_gate(
# # #                 torch.cat([vel_feat, hx], dim=-1)
# # #             )  # [B,1]  — giai đoạn đầu training gate≈1 → theo motion prior

# # #             raw_delta   = self.out_head(hx)                     # [B,2]
# # #             # Motion prior: tiếp tục với velocity cuối (decay theo step)
# # #             prior_delta = last_vel * (0.9 ** i)                 # [B,2]

# # #             # Blend: gate=1 → dùng motion prior hoàn toàn (đầu training)
# # #             #        gate=0 → dùng GRU output hoàn toàn (sau khi học)
# # #             scale_i = torch.sigmoid(self.step_scale[i]) * self.MAX_DELTA
# # #             blended = gate * prior_delta + (1 - gate) * raw_delta * scale_i

# # #             delta   = blended.clamp(-self.MAX_DELTA, self.MAX_DELTA)
# # #             cur_pos = cur_pos + delta
# # #             last_vel = delta   # update velocity prior
# # #             preds.append(cur_pos)

# # #         return torch.stack(preds, dim=0)   # [4, B, 2]


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  VelocityField
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class VelocityField(nn.Module):
# # #     """
# # #     OT-CFM velocity field v_θ(x_t, t, context).
# # #     v24: ShortRangeHead thêm vào để dự báo tầm ngắn chính xác hơn.
# # #     """

# # #     # Dim của raw_ctx (output ctx_fc1) – phải khớp ShortRangeHead
# # #     RAW_CTX_DIM = 512

# # #     def __init__(
# # #         self,
# # #         pred_len:   int   = 12,
# # #         obs_len:    int   = 8,
# # #         ctx_dim:    int   = 256,
# # #         sigma_min:  float = 0.02,
# # #         unet_in_ch: int   = 13,
# # #     ):
# # #         super().__init__()
# # #         self.pred_len  = pred_len
# # #         self.obs_len   = obs_len
# # #         self.sigma_min = sigma_min

# # #         self.spatial_enc = FNO3DEncoder(
# # #             in_channel   = unet_in_ch,
# # #             out_channel  = 1,
# # #             d_model      = 32,
# # #             n_layers     = 4,
# # #             modes_t      = 4,
# # #             modes_h      = 4,
# # #             modes_w      = 4,
# # #             spatial_down = 32,
# # #             dropout      = 0.05,
# # #         )

# # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # #         self.bottleneck_proj = nn.Linear(128, 128)
# # #         self.decoder_proj    = nn.Linear(1, 16)

# # #         self.enc_1d = DataEncoder1D(
# # #             in_1d       = 4,
# # #             feat_3d_dim = 128,
# # #             mlp_h       = 64,
# # #             lstm_hidden = 128,
# # #             lstm_layers = 3,
# # #             dropout     = 0.1,
# # #             d_state     = 16,
# # #         )

# # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # #         # ctx_fc1 output = 512 = RAW_CTX_DIM
# # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # #         self.ctx_drop = nn.Dropout(0.15)
# # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# # #         self.time_fc1 = nn.Linear(256, 512)
# # #         self.time_fc2 = nn.Linear(512, 256)

# # #         self.traj_embed = nn.Linear(4, 256)
# # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# # #         self.transformer = nn.TransformerDecoder(
# # #             nn.TransformerDecoderLayer(
# # #                 d_model=256, nhead=8, dim_feedforward=1024,
# # #                 dropout=0.15, activation="gelu", batch_first=True,
# # #             ),
# # #             num_layers=4,
# # #         )
# # #         self.out_fc1 = nn.Linear(256, 512)
# # #         self.out_fc2 = nn.Linear(512, 4)

# # #         self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)

# # #         # ── FIX-M23: Short-range head ──────────────────────────────────────
# # #         self.short_range_head = ShortRangeHead(
# # #             raw_ctx_dim = self.RAW_CTX_DIM,
# # #             obs_len     = obs_len,
# # #         )

# # #     # ── helpers ──────────────────────────────────────────────────────────────

# # #     def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
# # #         half = dim // 2
# # #         freq = torch.exp(
# # #             torch.arange(half, dtype=torch.float32, device=t.device)
# # #             * (-math.log(10_000.0) / max(half - 1, 1))
# # #         )
# # #         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
# # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # #         return F.pad(emb, (0, dim % 2))

# # #     # def _context(self, batch_list: List) -> torch.Tensor:
# # #     #     """Returns raw_ctx [B, RAW_CTX_DIM] (before dropout/projection)."""
# # #     #     obs_traj  = batch_list[0]
# # #     #     obs_Me    = batch_list[7]
# # #     #     image_obs = batch_list[11]
# # #     #     env_data  = batch_list[13]

# # #     #     if image_obs.dim() == 4:
# # #     #         image_obs = image_obs.unsqueeze(1)

# # #     #     expected_ch = self.spatial_enc.in_channel
# # #     #     if image_obs.shape[1] == 1 and expected_ch != 1:
# # #     #         image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# # #     #     e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # #     #     T_obs = obs_traj.shape[0]

# # #     #     e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
# # #     #     e_3d_s = e_3d_s.permute(0, 2, 1)
# # #     #     e_3d_s = self.bottleneck_proj(e_3d_s)

# # #     #     T_bot = e_3d_s.shape[1]
# # #     #     if T_bot != T_obs:
# # #     #         e_3d_s = F.interpolate(
# # #     #             e_3d_s.permute(0, 2, 1), size=T_obs,
# # #     #             mode="linear", align_corners=False,
# # #     #         ).permute(0, 2, 1)

# # #     #     # f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))
# # #     #     # f_spatial     = self.decoder_proj(f_spatial_raw)

# # #     #     # FIX: pool H,W, giữ T, rồi flatten
# # #     #     # f_spatial_raw = e_3d_dec.squeeze(1)          # [B,T,1,1] → [B,T]  (squeeze C và H,W)
# # #     #     # f_spatial_raw = f_spatial_raw.mean(dim=1)    # [B] — hoặc giữ T nếu muốn
# # #     #     # Hoặc đơn giản hơn: mean chỉ spatial
# # #     #     f_spatial_raw = e_3d_dec.mean(dim=(3,4))     # [B,1,T] → squeeze → [B,T]
# # #     #     f_spatial     = self.decoder_proj(f_spatial_raw.squeeze(1))  # cần Linear(T,16)

# # #     #     obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# # #     #     h_t    = self.enc_1d(obs_in, e_3d_s)

# # #     #     e_env, _, _ = self.env_enc(env_data, image_obs)

# # #     #     raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
# # #     #     raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))  # [B, RAW_CTX_DIM]
# # #     #     return raw                                      # raw_ctx

# # #     # ── _context(): fix f_spatial ──────────────────────────────────────────────
# # #     def _context(self, batch_list):
# # #         obs_traj  = batch_list[0]   # [T_obs, B, 2]
# # #         obs_Me    = batch_list[7]   # [T_obs, B, 2]
# # #         image_obs = batch_list[11]  # [B, 13, T, 81, 81]  từ seq_collate
# # #         env_data  = batch_list[13]

# # #         # image_obs đã được permute đúng bởi seq_collate → [B,13,T,H,W]
# # #         # Chỉ cần guard cho trường hợp dim==4 (không có T)
# # #         if image_obs.dim() == 4:
# # #             image_obs = image_obs.unsqueeze(2)  # thêm T dim, không phải C

# # #         expected_ch = self.spatial_enc.in_channel
# # #         if image_obs.shape[1] == 1 and expected_ch != 1:
# # #             image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # #         # e_3d_bot: [B,128,T,4,4]   e_3d_dec: [B,1,T,1,1]

# # #         T_obs = obs_traj.shape[0]

# # #         # Bottleneck: pool spatial → [B,128,T] → permute → [B,T,128]
# # #         e_3d_s = self.bottleneck_pool(e_3d_bot)   # [B,128,T,1,1]
# # #         e_3d_s = e_3d_s.squeeze(-1).squeeze(-1)   # [B,128,T]
# # #         e_3d_s = e_3d_s.permute(0, 2, 1)          # [B,T,128]
# # #         e_3d_s = self.bottleneck_proj(e_3d_s)     # [B,T,128]

# # #         # T alignment
# # #         T_bot = e_3d_s.shape[1]
# # #         if T_bot != T_obs:
# # #             e_3d_s = F.interpolate(
# # #                 e_3d_s.permute(0, 2, 1), size=T_obs,
# # #                 mode="linear", align_corners=False,
# # #             ).permute(0, 2, 1)

# # #         # FIX Bug G: f_spatial giữ lại temporal info
# # #         # e_3d_dec: [B,1,T,1,1] → mean(H,W) → [B,1,T] → mean(T) → [B,1]
# # #         # Thay bằng: pool T về 1 bằng weighted mean (cuối quan trọng hơn)
# # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)  # [B,T]
# # #         # Exponential weighting: bước cuối obs quan trọng nhất
# # #         t_weights = torch.softmax(
# # #             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # #                         device=e_3d_dec_t.device) * 0.5,
# # #             dim=0
# # #         )  # [T]
# # #         f_spatial_scalar = (e_3d_dec_t * t_weights.unsqueeze(0)).sum(1, keepdim=True)  # [B,1]
# # #         f_spatial = self.decoder_proj(f_spatial_scalar)  # [B,16]

# # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)  # [B,T,4]
# # #         h_t    = self.enc_1d(obs_in, e_3d_s)   # [B,128]

# # #         e_env, _, _ = self.env_enc(env_data, image_obs)  # [B,64]

# # #         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)  # [B,208]
# # #         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))       # [B,512]
# # #         return raw

# # #     def _apply_ctx_head(self, raw: torch.Tensor,
# # #                         noise_scale: float = 0.0) -> torch.Tensor:
# # #         if noise_scale > 0.0:
# # #             raw = raw + torch.randn_like(raw) * noise_scale
# # #         return self.ctx_fc2(self.ctx_drop(raw))

# # #     def _beta_drift_velocity(self, x_t: torch.Tensor) -> torch.Tensor:
# # #         """Beta drift in normalised state space. x_t: [B, T, 4]."""
# # #         OMEGA_val  = 7.2921e-5
# # #         R_val      = 6.371e6
# # #         DT         = 6 * 3600.0
# # #         M_PER_NORM = 5.0 * 111.0 * 1000.0

# # #         lat_norm = x_t[:, :, 1]
# # #         lat_deg  = lat_norm * 5.0
# # #         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))

# # #         beta   = 2 * OMEGA_val * torch.cos(lat_rad) / R_val
# # #         R_tc   = 3e5
# # #         v_lon  = -beta * R_tc ** 2 / 2
# # #         v_lat  =  beta * R_tc ** 2 / 4

# # #         v_lon_norm = v_lon * DT / M_PER_NORM
# # #         v_lat_norm = v_lat * DT / M_PER_NORM

# # #         v_phys = torch.zeros_like(x_t)
# # #         v_phys[:, :, 0] = v_lon_norm
# # #         v_phys[:, :, 1] = v_lat_norm
# # #         return v_phys

# # #     def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
# # #                 ctx: torch.Tensor) -> torch.Tensor:
# # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
# # #         t_emb = self.time_fc2(t_emb)

# # #         T_seq  = min(x_t.size(1), self.pos_enc.shape[1])
# # #         x_emb  = (self.traj_embed(x_t[:, :T_seq, :])
# # #                   + self.pos_enc[:, :T_seq, :]
# # #                   + t_emb.unsqueeze(1))
# # #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

# # #         v_neural = self.out_fc2(F.gelu(self.out_fc1(
# # #             self.transformer(x_emb, memory)
# # #         )))

# # #         with torch.no_grad():
# # #             v_phys = self._beta_drift_velocity(x_t[:, :T_seq, :])

# # #         scale = torch.sigmoid(self.physics_scale) * 2.0
# # #         return v_neural + scale * v_phys

# # #     def forward(self, x_t, t, batch_list):
# # #         raw = self._context(batch_list)
# # #         ctx = self._apply_ctx_head(raw, noise_scale=0.0)
# # #         return self._decode(x_t, t, ctx)

# # #     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale: float = 0.0):
# # #         ctx = self._apply_ctx_head(raw_ctx, noise_scale=noise_scale)
# # #         return self._decode(x_t, t, ctx)

# # #     def forward_short_range(
# # #         self,
# # #         obs_traj: torch.Tensor,  # [T_obs, B, 2]
# # #         raw_ctx:  torch.Tensor,  # [B, RAW_CTX_DIM]
# # #     ) -> torch.Tensor:
# # #         """FIX-M23: Deterministic short-range prediction [4, B, 2]."""
# # #         return self.short_range_head(raw_ctx, obs_traj)


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  TCFlowMatching
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class TCFlowMatching(nn.Module):
# # #     """TC trajectory prediction via OT-CFM + ShortRangeHead + PINN."""

# # #     def __init__(
# # #         self,
# # #         pred_len:             int   = 12,
# # #         obs_len:              int   = 8,
# # #         sigma_min:            float = 0.02,
# # #         n_train_ens:          int   = 4,
# # #         unet_in_ch:           int   = 13,
# # #         ctx_noise_scale:      float = 0.002,    # FIX-M25: 0.02→0.002
# # #         initial_sample_sigma: float = 0.03,     # FIX-M25: 0.1→0.03
# # #         **kwargs,
# # #     ):
# # #         super().__init__()
# # #         self.pred_len             = pred_len
# # #         self.obs_len              = obs_len
# # #         self.sigma_min            = sigma_min
# # #         self.n_train_ens          = n_train_ens
# # #         self.active_pred_len      = pred_len
# # #         self.ctx_noise_scale      = ctx_noise_scale
# # #         self.initial_sample_sigma = initial_sample_sigma

# # #         self.net = VelocityField(
# # #             pred_len   = pred_len,
# # #             obs_len    = obs_len,
# # #             sigma_min  = sigma_min,
# # #             unet_in_ch = unet_in_ch,
# # #         )

# # #     def set_curriculum_len(self, active_len: int) -> None:
# # #         """FIX-M18: No-op."""
# # #         pass

# # #     # ── Static helpers ────────────────────────────────────────────────────────

# # #     @staticmethod
# # #     def _to_rel(traj_gt, Me_gt, last_pos, last_Me):
# # #         return torch.cat(
# # #             [traj_gt - last_pos.unsqueeze(0),
# # #              Me_gt   - last_Me.unsqueeze(0)],
# # #             dim=-1,
# # #         ).permute(1, 0, 2)

# # #     @staticmethod
# # #     def _to_abs(rel, last_pos, last_Me):
# # #         d = rel.permute(1, 0, 2)
# # #         return (
# # #             last_pos.unsqueeze(0) + d[:, :, :2],
# # #             last_Me.unsqueeze(0)  + d[:, :, 2:],
# # #         )

# # #     def _cfm_noisy(self, x1):
# # #         B, device = x1.shape[0], x1.device
# # #         sm  = self.sigma_min
# # #         x0  = torch.randn_like(x1) * sm
# # #         t   = torch.rand(B, device=device)
# # #         te  = t.view(B, 1, 1)
# # #         x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
# # #         denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
# # #         target_vel = (x1 - (1.0 - sm) * x_t) / denom
# # #         return x_t, t, te, denom, target_vel

# # #     @staticmethod
# # #     def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
# # #         wind_norm = obs_Me[-1, :, 1].detach()
# # #         w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
# # #             torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
# # #             torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
# # #                         torch.full_like(wind_norm, 1.5))))
# # #         return w / w.mean().clamp(min=1e-6)

# # #     @staticmethod
# # #     def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
# # #         if torch.rand(1).item() > p:
# # #             return batch_list
# # #         aug = list(batch_list)
# # #         for idx in [0, 1, 2, 3]:
# # #             t = aug[idx]
# # #             if torch.is_tensor(t) and t.shape[-1] >= 1:
# # #                 t = t.clone()
# # #                 t[..., 0] = -t[..., 0]
# # #                 aug[idx] = t
# # #         return aug

# # #     # ── Loss ──────────────────────────────────────────────────────────────────

# # #     def get_loss(self, batch_list: List,
# # #                  step_weight_alpha: float = 0.0) -> torch.Tensor:
# # #         return self.get_loss_breakdown(batch_list, step_weight_alpha)["total"]

# # #     def get_loss_breakdown(self, batch_list: List,
# # #                            step_weight_alpha: float = 0.0) -> Dict:
# # #         """
# # #         FIX-M27: Thêm short_range_loss với weight cao (5.0).
# # #         """
# # #         batch_list = self._lon_flip_aug(batch_list, p=0.3)

# # #         traj_gt  = batch_list[1]
# # #         Me_gt    = batch_list[8]
# # #         obs_t    = batch_list[0]
# # #         obs_Me   = batch_list[7]

# # #         try:
# # #             env_data = batch_list[13]
# # #         except (IndexError, TypeError):
# # #             env_data = None

# # #         lp, lm = obs_t[-1], obs_Me[-1]
# # #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

# # #         raw_ctx     = self.net._context(batch_list)
# # #         intensity_w = self._intensity_weights(obs_Me)

# # #         x_t, t, te, denom, _ = self._cfm_noisy(x1)
# # #         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

# # #         # Ensemble for AFCRPS
# # #         samples: List[torch.Tensor] = []
# # #         for _ in range(self.n_train_ens):
# # #             xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
# # #             pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
# # #             x1_s  = xt_s + dens_s * pv_s
# # #             pa_s, _ = self._to_abs(x1_s, lp, lm)
# # #             samples.append(pa_s)
# # #         pred_samples = torch.stack(samples)     # [S, T, B, 2]
# # #         all_trajs_4d = pred_samples

# # #         l_fm_physics = fm_physics_consistency_loss(
# # #             pred_samples, gt_norm=traj_gt, last_pos=lp)

# # #         x1_pred = x_t + denom * pred_vel
# # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# # #         pred_abs_deg = _denorm_to_deg(pred_abs)
# # #         # # Thêm debug vào get_loss_breakdown, sau khi tính pred_abs_deg:
# # #         # print(f"  pred_abs requires_grad: {pred_abs.requires_grad}")
# # #         # print(f"  pred_abs_deg requires_grad: {pred_abs_deg.requires_grad}")

# # #         traj_gt_deg  = _denorm_to_deg(traj_gt)
# # #         ref_deg      = _denorm_to_deg(lp)

# # #         breakdown = compute_total_loss(
# # #             pred_abs           = pred_abs_deg,
# # #             gt                 = traj_gt_deg,
# # #             ref                = ref_deg,
# # #             batch_list         = batch_list,
# # #             pred_samples       = pred_samples,
# # #             gt_norm            = traj_gt,
# # #             weights            = WEIGHTS,
# # #             intensity_w        = intensity_w,
# # #             env_data           = env_data,
# # #             step_weight_alpha  = step_weight_alpha,
# # #             all_trajs          = all_trajs_4d,
# # #         )

# # #         fm_phys_w = WEIGHTS.get("fm_physics", 0.3)
# # #         breakdown["total"]      = breakdown["total"] + fm_phys_w * l_fm_physics
# # #         breakdown["fm_physics"] = l_fm_physics.item()

# # #         # ── FIX-M27: Short-range loss ──────────────────────────────────────
# # #         n_sr = ShortRangeHead.N_STEPS  # 4
# # #         if traj_gt.shape[0] >= n_sr:
# # #             sr_pred = self.net.forward_short_range(obs_t, raw_ctx)  # [4, B, 2]
# # #             sr_gt   = traj_gt[:n_sr]                                # [4, B, 2]
# # #             l_sr    = short_range_regression_loss(sr_pred, sr_gt, lp)
# # #             sr_w    = WEIGHTS.get("short_range", 5.0)
# # #             breakdown["total"]       = breakdown["total"] + sr_w * l_sr
# # #             breakdown["short_range"] = l_sr.item()
# # #         else:
# # #             breakdown["short_range"] = 0.0

# # #         return breakdown

# # #     # ── sample() ─────────────────────────────────────────────────────────────

# # #     # @torch.no_grad()
# # #     # def sample(
# # #     #     self,
# # #     #     batch_list:  List,
# # #     #     num_ensemble: int = 50,
# # #     #     ddim_steps:   int = 20,
# # #     #     predict_csv:  Optional[str] = None,
# # #     # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# # #     #     """
# # #     #     FIX-M24: pred_mean[:4] ← ShortRangeHead (deterministic, low-error).
# # #     #              pred_mean[4:] ← FM ensemble mean.

# # #     #     Returns:
# # #     #         pred_mean : [T, B, 2]
# # #     #         me_mean   : [T, B, 2]
# # #     #         all_trajs : [S, T, B, 2]
# # #     #     """
# # #     #     obs_t = batch_list[0]          # [T_obs, B, 2]
# # #     #     lp    = obs_t[-1]              # [B, 2]
# # #     #     lm    = batch_list[7][-1]      # [B, 2]
# # #     #     B     = lp.shape[0]
# # #     #     device = lp.device
# # #     #     T      = self.pred_len
# # #     #     dt     = 1.0 / max(ddim_steps, 1)

# # #     #     raw_ctx = self.net._context(batch_list)

# # #     #     # ── Short-range deterministic (steps 1-4) ─────────────────────────
# # #     #     n_sr     = ShortRangeHead.N_STEPS      # 4
# # #     #     sr_pred  = self.net.forward_short_range(obs_t, raw_ctx)  # [4, B, 2]

# # #     #     # ── FM ensemble (all 12 steps, for steps 5-12) ───────────────────
# # #     #     traj_s: List[torch.Tensor] = []
# # #     #     me_s:   List[torch.Tensor] = []

# # #     #     for k in range(num_ensemble):
# # #     #         x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# # #     #         for step in range(ddim_steps):
# # #     #             t_b = torch.full((B,), step * dt, device=device)
# # #     #             ns  = self.ctx_noise_scale if step == 0 else 0.0
# # #     #             vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
# # #     #             x_t = x_t + dt * vel

# # #     #         x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)
# # #     #         x_t = x_t.clamp(-3.0, 3.0)

# # #     #         tr, me = self._to_abs(x_t, lp, lm)
# # #     #         traj_s.append(tr)
# # #     #         me_s.append(me)

# # #     #     all_trajs = torch.stack(traj_s)    # [S, T, B, 2]
# # #     #     all_me    = torch.stack(me_s)

# # #     #     # ── FIX-M24: Blend short-range into pred_mean ─────────────────────
# # #     #     fm_mean  = all_trajs.mean(0)       # [T, B, 2]
# # #     #     pred_mean = fm_mean.clone()
# # #     #     pred_mean[:n_sr] = sr_pred         # Override steps 1-4

# # #     #     # # Also override all_trajs first 4 steps with deterministic prediction
# # #     #     # # (reduces spurious spread for 12h/24h in CRPS)
# # #     #     # all_trajs[:, :n_sr, :, :] = sr_pred.unsqueeze(0).expand(
# # #     #     #     num_ensemble, -1, -1, -1
# # #     #     # )

# # #     #     me_mean = all_me.mean(0)

# # #     #     if predict_csv:
# # #     #         self._write_predict_csv(predict_csv, pred_mean, all_trajs)

# # #     #     return pred_mean, me_mean, all_trajs

# # #     @torch.no_grad()
# # #     def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
# # #         obs_t  = batch_list[0]
# # #         lp     = obs_t[-1]
# # #         lm     = batch_list[7][-1]
# # #         B      = lp.shape[0]
# # #         device = lp.device
# # #         T      = self.pred_len
# # #         dt     = 1.0 / max(ddim_steps, 1)

# # #         raw_ctx = self.net._context(batch_list)
# # #         n_sr    = ShortRangeHead.N_STEPS

# # #         # Short-range deterministic
# # #         sr_pred = self.net.forward_short_range(obs_t, raw_ctx)   # [4, B, 2]

# # #         # FM ensemble — giữ nguyên, KHÔNG override
# # #         traj_s, me_s = [], []
# # #         for k in range(num_ensemble):
# # #             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# # #             for step in range(ddim_steps):
# # #                 t_b = torch.full((B,), step * dt, device=device)
# # #                 ns  = self.ctx_noise_scale if step == 0 else 0.0
# # #                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
# # #                 x_t = x_t + dt * vel

# # #             x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)
# # #             x_t = x_t.clamp(-3.0, 3.0)

# # #             tr, me = self._to_abs(x_t, lp, lm)
# # #             traj_s.append(tr)
# # #             me_s.append(me)

# # #         all_trajs = torch.stack(traj_s)   # [S, T, B, 2] — KHÔNG override steps 1-4
# # #         all_me    = torch.stack(me_s)

# # #         # FIX: pred_mean blend đúng cách
# # #         # Steps 1-4: ShortRangeHead (deterministic, low-error)
# # #         # Steps 5-12: FM ensemble mean
# # #         fm_mean   = all_trajs.mean(0)    # [T, B, 2]
# # #         pred_mean = fm_mean.clone()
# # #         pred_mean[:n_sr] = sr_pred       # override pred_mean, KHÔNG override all_trajs

# # #         me_mean = all_me.mean(0)

# # #         if predict_csv:
# # #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)

# # #         return pred_mean, me_mean, all_trajs

# # #     # ── Physics correction ────────────────────────────────────────────────────

# # #     def _physics_correct(
# # #         self,
# # #         x_pred: torch.Tensor,
# # #         last_pos: torch.Tensor,
# # #         last_Me:  torch.Tensor,
# # #         n_steps:  int   = 5,
# # #         lr:       float = 0.002,
# # #     ) -> torch.Tensor:
# # #         with torch.enable_grad():
# # #             x = x_pred.detach().requires_grad_(True)
# # #             optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)

# # #             for _ in range(n_steps):
# # #                 optimizer.zero_grad()
# # #                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
# # #                 pred_deg    = _denorm_to_deg(pred_abs)

# # #                 l_speed = self._pinn_speed_constraint(pred_deg)
# # #                 l_accel = self._pinn_beta_plane_simplified(pred_deg)

# # #                 physics_loss = l_speed + 0.3 * l_accel
# # #                 physics_loss.backward()

# # #                 torch.nn.utils.clip_grad_norm_([x], max_norm=0.05)
# # #                 optimizer.step()

# # #         return x.detach()

# # #     @staticmethod
# # #     def _pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
# # #         if pred_deg.shape[0] < 2:
# # #             return pred_deg.new_zeros(())
# # #         dt_deg  = pred_deg[1:] - pred_deg[:-1]
# # #         lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
# # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # #         dx_km   = dt_deg[:, :, 0] * cos_lat * 111.0
# # #         dy_km   = dt_deg[:, :, 1] * 111.0
# # #         speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
# # #         return F.relu(speed - 600.0).pow(2).mean()

# # #     @staticmethod
# # #     def _pinn_beta_plane_simplified(pred_deg: torch.Tensor) -> torch.Tensor:
# # #         if pred_deg.shape[0] < 3:
# # #             return pred_deg.new_zeros(())
# # #         v = pred_deg[1:] - pred_deg[:-1]
# # #         a = v[1:] - v[:-1]
# # #         lat_rad = torch.deg2rad(pred_deg[1:-1, :, 1])
# # #         cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
# # #         a_lon_km = a[:, :, 0] * cos_lat * 111.0
# # #         a_lat_km = a[:, :, 1] * 111.0
# # #         max_accel = 50.0
# # #         violation = F.relu(torch.sqrt(a_lon_km**2 + a_lat_km**2) - max_accel)
# # #         return violation.pow(2).mean() * 0.1

# # #     @staticmethod
# # #     def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
# # #                            all_trajs: torch.Tensor) -> None:
# # #         import numpy as np
# # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # #         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
# # #         T, B, _ = traj_mean.shape
# # #         S       = all_trajs.shape[0]

# # #         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # #         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
# # #         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
# # #         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()

# # #         fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # #                     "lon_mean_deg", "lat_mean_deg",
# # #                     "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # #         write_hdr = not os.path.exists(csv_path)
# # #         with open(csv_path, "a", newline="") as fh:
# # #             w = csv.DictWriter(fh, fieldnames=fields)
# # #             if write_hdr:
# # #                 w.writeheader()
# # #             for b in range(B):
# # #                 for k in range(T):
# # #                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
# # #                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
# # #                         math.radians(mean_lat[k, b]))
# # #                     spread = float(np.sqrt((dlat**2 + dlon**2).mean()) * 111.0)
# # #                     w.writerow(dict(
# # #                         timestamp     = ts,
# # #                         batch_idx     = b,
# # #                         step_idx      = k,
# # #                         lead_h        = (k + 1) * 6,
# # #                         lon_mean_deg  = f"{mean_lon[k,b]:.4f}",
# # #                         lat_mean_deg  = f"{mean_lat[k,b]:.4f}",
# # #                         lon_std_deg   = f"{all_lon[:,k,b].std():.4f}",
# # #                         lat_std_deg   = f"{all_lat[:,k,b].std():.4f}",
# # #                         ens_spread_km = f"{spread:.2f}",
# # #                     ))
# # #         print(f"  Predictions → {csv_path}  (B={B}, T={T}, S={S})")


# # # # Backward-compat alias
# # # TCDiffusion = TCFlowMatching

# # """
# # Model/flow_matching_model.py  ── v25
# # ==========================================
# # FULL REWRITE – fixes từ review:

# #   FIX-M-A  [CRITICAL] ShortRangeHead motion_gate init: bias = +2.0
# #            để gate ≈ 0.88 đầu training (motion prior dominant),
# #            thay vì xavier_uniform → gate ≈ 0.5 (mixed từ đầu).

# #   FIX-M-B  [HIGH] Phase 1 freeze logic: freeze ctx_fc1/ctx_ln cũng
# #            vì ShortRangeHead nhận raw_ctx từ frozen encoder → học
# #            trên noise. Phase 1 chỉ train ShortRangeHead với obs_traj.

# #   FIX-M-C  [HIGH] Bridge loss integration: compute sr_pred trong
# #            get_loss_breakdown() và pass sang compute_total_loss().
# #            SR↔FM nhất quán tại step 4 (Eq.80).

# #   FIX-M-D  [HIGH] _physics_correct: thay SGD+momentum bằng Adam
# #            (n_steps nhỏ, momentum gây oscillation).

# #   FIX-M-E  [MEDIUM] get_loss_breakdown() pass epoch, gt_abs_deg,
# #            vmax_pred, pmin_pred cho adaptive PINN weighting.

# #   FIX-M-F  [MEDIUM] sample(): cache raw_ctx một lần, dùng cho cả
# #            SR head và FM ensemble (tránh tính 2 lần).

# #   FIX-M-G  [LOW] ShortRangeHead: thêm layer norm trước out_head
# #            để gradient scale ổn định hơn.

# # Kept from v24:
# #   FIX-M23  ShortRangeHead GRU autoregressive
# #   FIX-M24  Blend: pred_mean[:4] ← SR, [4:] ← FM mean
# #   FIX-M25  initial_sample_sigma=0.03, ctx_noise_scale=0.002
# #   FIX-M26  MAX_DELTA clamp
# #   FIX-M27  short_range_regression_loss trong get_loss_breakdown
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
# #     pinn_speed_constraint, short_range_regression_loss, bridge_loss,
# #     _norm_to_deg,
# # )


# # def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
# #     """Normalised → degrees. [T, B, 2] or [B, 2]."""
# #     out = traj_norm.clone()
# #     out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
# #     out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
# #     return out


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  ShortRangeHead  – FIX-M-A, FIX-M-G
# # # ══════════════════════════════════════════════════════════════════════════════

# # class ShortRangeHead(nn.Module):
# #     """
# #     GRU autoregressive predictor cho steps 1-4 (6h/12h/18h/24h).

# #     FIX-M-A: motion_gate bias init = +2.0 → sigmoid(2) ≈ 0.88
# #              đảm bảo motion prior dominant đầu training.
# #     FIX-M-G: LayerNorm trước out_head để gradient scale ổn định.
# #     """

# #     N_STEPS   = 4
# #     MAX_DELTA = 0.48     # ≈ 576 km / 6h

# #     def __init__(self, raw_ctx_dim: int = 512, obs_len: int = 8):
# #         super().__init__()
# #         self.obs_len = obs_len

# #         # Velocity encoder từ obs
# #         self.vel_enc = nn.Sequential(
# #             nn.Linear(obs_len * 2, 256),
# #             nn.GELU(),
# #             nn.LayerNorm(256),
# #             nn.Linear(256, 128),
# #             nn.GELU(),
# #         )

# #         self.ctx_proj = nn.Sequential(
# #             nn.Linear(raw_ctx_dim, 256),
# #             nn.GELU(),
# #             nn.LayerNorm(256),
# #             nn.Dropout(0.05),
# #         )

# #         # GRU: input = ctx(256) + vel(128) + cur_pos(2) = 386
# #         self.gru_cell = nn.GRUCell(input_size=386, hidden_size=256)

# #         # FIX-M-A: motion gate với bias khởi tạo cao
# #         self.motion_gate_linear = nn.Linear(128 + 256, 1)

# #         # FIX-M-G: LayerNorm trước output
# #         self.out_norm = nn.LayerNorm(256)
# #         self.out_head = nn.Sequential(
# #             nn.Linear(256, 128),
# #             nn.GELU(),
# #             nn.LayerNorm(128),
# #             nn.Linear(128, 2),
# #         )

# #         self.step_scale = nn.Parameter(torch.ones(self.N_STEPS) * 0.2)
# #         self._init_weights()

# #     def _init_weights(self):
# #         with torch.no_grad():
# #             for m in self.modules():
# #                 if isinstance(m, nn.Linear):
# #                     nn.init.xavier_uniform_(m.weight, gain=0.3)
# #                     if m.bias is not None:
# #                         nn.init.zeros_(m.bias)
# #             # FIX-M-A: init bias của gate linear = +2.0
# #             # sigmoid(2.0) ≈ 0.88 → motion prior chiếm 88% đầu training
# #             nn.init.constant_(self.motion_gate_linear.bias, 2.0)

# #     def forward(self, raw_ctx: torch.Tensor,
# #                 obs_traj: torch.Tensor) -> torch.Tensor:
# #         """
# #         raw_ctx:  [B, raw_ctx_dim]
# #         obs_traj: [T_obs, B, 2]  normalised
# #         Returns:  [4, B, 2]  normalised predictions
# #         """
# #         B     = raw_ctx.shape[0]
# #         T_obs = obs_traj.shape[0]

# #         # Velocity encoding
# #         if T_obs >= 2:
# #             vels = obs_traj[1:] - obs_traj[:-1]   # [T_obs-1, B, 2]
# #         else:
# #             vels = torch.zeros(1, B, 2, device=raw_ctx.device)

# #         target_v = self.obs_len
# #         if vels.shape[0] < target_v:
# #             pad  = torch.zeros(target_v - vels.shape[0], B, 2, device=raw_ctx.device)
# #             vels = torch.cat([pad, vels], dim=0)
# #         else:
# #             vels = vels[-target_v:]

# #         vel_input = vels.permute(1, 0, 2).reshape(B, -1)  # [B, obs_len*2]
# #         vel_feat  = self.vel_enc(vel_input)                 # [B, 128]
# #         ctx_feat  = self.ctx_proj(raw_ctx)                  # [B, 256]

# #         hx      = ctx_feat
# #         cur_pos = obs_traj[-1].clone()   # [B, 2]
# #         last_vel = vels[-1].clone()      # [B, 2]

# #         preds = []
# #         for i in range(self.N_STEPS):
# #             inp = torch.cat([ctx_feat, vel_feat, cur_pos], dim=-1)  # [B, 386]
# #             hx  = self.gru_cell(inp, hx)                            # [B, 256]

# #             # FIX-M-A: gate với bias=+2.0 → dominant motion prior đầu training
# #             gate_input = torch.cat([vel_feat, hx], dim=-1)  # [B, 384]
# #             gate = torch.sigmoid(self.motion_gate_linear(gate_input))  # [B, 1]

# #             # FIX-M-G: LayerNorm trước output
# #             raw_delta   = self.out_head(self.out_norm(hx))  # [B, 2]
# #             prior_delta = last_vel * (0.9 ** i)              # [B, 2]

# #             scale_i = torch.sigmoid(self.step_scale[i]) * self.MAX_DELTA
# #             blended = gate * prior_delta + (1.0 - gate) * raw_delta * scale_i
# #             delta   = blended.clamp(-self.MAX_DELTA, self.MAX_DELTA)

# #             cur_pos  = cur_pos + delta
# #             last_vel = delta
# #             preds.append(cur_pos)

# #         return torch.stack(preds, dim=0)   # [4, B, 2]


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  VelocityField
# # # ══════════════════════════════════════════════════════════════════════════════

# # class VelocityField(nn.Module):
# #     """OT-CFM velocity field v_θ(x_t, t, context)."""

# #     RAW_CTX_DIM = 512

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
# #             d_model      = 32,
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

# #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# #         self.ctx_drop = nn.Dropout(0.15)
# #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

# #         self.time_fc1 = nn.Linear(256, 512)
# #         self.time_fc2 = nn.Linear(512, 256)

# #         self.traj_embed  = nn.Linear(4, 256)
# #         # self.pos_enc     = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
# #         # Thay vì:
# #         # self.pos_enc = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)

# #         # Hãy thử:
# #         self.pos_enc = nn.Parameter(torch.randn(1, pred_len, 256) * 0.1) # Tăng scale lên 5 lần
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

# #         # ShortRangeHead (FIX-M-A applied)
# #         self.short_range_head = ShortRangeHead(
# #             raw_ctx_dim = self.RAW_CTX_DIM,
# #             obs_len     = obs_len,
# #         )

# #     # ── Helpers ───────────────────────────────────────────────────────────────

# #     def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
# #         half = dim // 2
# #         freq = torch.exp(
# #             torch.arange(half, dtype=torch.float32, device=t.device)
# #             * (-math.log(10_000.0) / max(half - 1, 1))
# #         )
# #         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
# #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# #         return F.pad(emb, (0, dim % 2))

# #     def _context(self, batch_list) -> torch.Tensor:
# #         """Compute raw_ctx [B, RAW_CTX_DIM]."""
# #         obs_traj  = batch_list[0]
# #         obs_Me    = batch_list[7]
# #         image_obs = batch_list[11]
# #         env_data  = batch_list[13]

# #         if image_obs.dim() == 4:
# #             image_obs = image_obs.unsqueeze(2)

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

# #         # Weighted temporal pooling của decoder output
# #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)  # [B, T]
# #         t_weights = torch.softmax(
# #             torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# #                         device=e_3d_dec_t.device) * 0.5, dim=0,
# #         )
# #         f_spatial_scalar = (e_3d_dec_t * t_weights.unsqueeze(0)).sum(1, keepdim=True)  # [B,1]
# #         f_spatial = self.decoder_proj(f_spatial_scalar)   # [B, 16]

# #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)  # [B,T,4]
# #         h_t    = self.enc_1d(obs_in, e_3d_s)              # [B, 128]

# #         e_env, _, _ = self.env_enc(env_data, image_obs)   # [B, 32]

# #         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)  # [B, 176]
# #         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))       # [B, 512]
# #         return raw

# #     def _apply_ctx_head(self, raw: torch.Tensor,
# #                         noise_scale: float = 0.0) -> torch.Tensor:
# #         if noise_scale > 0.0:
# #             raw = raw + torch.randn_like(raw) * noise_scale
# #         return self.ctx_fc2(self.ctx_drop(raw))

# #     def _beta_drift_velocity(self, x_t: torch.Tensor) -> torch.Tensor:
# #         """Beta drift trong normalised state space."""
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

# #         T_seq = min(x_t.size(1), self.pos_enc.shape[1])
# #         x_emb = (self.traj_embed(x_t[:, :T_seq, :])
# #                   + self.pos_enc[:, :T_seq, :]
# #                   + t_emb.unsqueeze(1))
# #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

# #         v_neural = self.out_fc2(F.gelu(self.out_fc1(
# #             self.transformer(x_emb, memory)
# #         )))

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

# #     def forward_short_range(self, obs_traj: torch.Tensor,
# #                             raw_ctx: torch.Tensor) -> torch.Tensor:
# #         """Deterministic short-range [4, B, 2]."""
# #         return self.short_range_head(raw_ctx, obs_traj)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  TCFlowMatching
# # # ══════════════════════════════════════════════════════════════════════════════

# # class TCFlowMatching(nn.Module):
# #     """TC trajectory prediction via OT-CFM + ShortRangeHead + PINN."""

# #     def __init__(
# #         self,
# #         pred_len:             int   = 12,
# #         obs_len:              int   = 8,
# #         sigma_min:            float = 0.02,
# #         n_train_ens:          int   = 4,
# #         unet_in_ch:           int   = 13,
# #         # ctx_noise_scale:      float = 0.002,
# #         # initial_sample_sigma: float = 0.03,
# #         ctx_noise_scale:      float = 0.01,
# #         initial_sample_sigma: float = 0.15,
# #         **kwargs,
# #     ):
# #         super().__init__()
# #         self.pred_len             = pred_len
# #         self.obs_len              = obs_len
# #         self.sigma_min            = sigma_min
# #         self.n_train_ens          = n_train_ens
# #         self.active_pred_len      = pred_len
# #         self.ctx_noise_scale      = ctx_noise_scale
# #         self.initial_sample_sigma = initial_sample_sigma

# #         self.net = VelocityField(
# #             pred_len   = pred_len,
# #             obs_len    = obs_len,
# #             sigma_min  = sigma_min,
# #             unet_in_ch = unet_in_ch,
# #         )

# #     def set_curriculum_len(self, active_len: int) -> None:
# #         pass  # No curriculum

# #     # ── Static helpers ────────────────────────────────────────────────────────

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

# #     # ── Loss ──────────────────────────────────────────────────────────────────

# #     def get_loss(self, batch_list: List,
# #                  step_weight_alpha: float = 0.0,
# #                  epoch: int = 0) -> torch.Tensor:
# #         return self.get_loss_breakdown(
# #             batch_list, step_weight_alpha, epoch)["total"]

# #     def get_loss_breakdown(
# #         self,
# #         batch_list: List,
# #         step_weight_alpha: float = 0.0,
# #         epoch: int = 0,
# #     ) -> Dict:
# #         """
# #         FIX-M-C: Tính sr_pred và pass sang compute_total_loss qua bridge_loss.
# #         FIX-M-E: Pass epoch và gt_abs_deg cho adaptive PINN.
# #         FIX-M-F: Cache raw_ctx một lần.
# #         """
# #         batch_list = self._lon_flip_aug(batch_list, p=0.3)

# #         traj_gt = batch_list[1]    # [T, B, 2] normalised
# #         Me_gt   = batch_list[8]
# #         obs_t   = batch_list[0]    # [T_obs, B, 2] normalised
# #         obs_Me  = batch_list[7]

# #         try:
# #             env_data = batch_list[13]
# #         except (IndexError, TypeError):
# #             env_data = None

# #         lp, lm = obs_t[-1], obs_Me[-1]
# #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

# #         # FIX-M-F: compute raw_ctx một lần duy nhất
# #         raw_ctx     = self.net._context(batch_list)
# #         intensity_w = self._intensity_weights(obs_Me)

# #         # CFM forward
# #         x_t, t, te, denom, _ = self._cfm_noisy(x1)
# #         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

# #         # Ensemble cho AFCRPS
# #         samples: List[torch.Tensor] = []
# #         for _ in range(self.n_train_ens):
# #             xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
# #             pv_s   = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
# #             x1_s   = xt_s + dens_s * pv_s
# #             pa_s, _ = self._to_abs(x1_s, lp, lm)
# #             samples.append(pa_s)
# #         pred_samples = torch.stack(samples)     # [S, T, B, 2]

# #         l_fm_physics = fm_physics_consistency_loss(pred_samples, gt_norm=traj_gt, last_pos=lp)

# #         x1_pred = x_t + denom * pred_vel
# #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# #         pred_abs_deg = _denorm_to_deg(pred_abs)    # [T, B, 2] degrees
# #         traj_gt_deg  = _denorm_to_deg(traj_gt)     # [T, B, 2] degrees
# #         ref_deg      = _denorm_to_deg(lp)           # [B, 2]

# #         # FIX-M-C: Compute sr_pred (dùng raw_ctx đã cache)
# #         n_sr = ShortRangeHead.N_STEPS  # 4
# #         sr_pred = None
# #         l_sr    = pred_abs.new_zeros(())
# #         if traj_gt.shape[0] >= n_sr:
# #             sr_pred = self.net.forward_short_range(obs_t, raw_ctx)  # [4, B, 2]
# #             sr_gt   = traj_gt[:n_sr]                                # [4, B, 2]
# #             l_sr    = short_range_regression_loss(sr_pred, sr_gt, lp)

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
# #             step_weight_alpha  = step_weight_alpha,
# #             all_trajs          = pred_samples,
# #             # FIX-M-E: pass epoch và gt cho adaptive PINN
# #             epoch              = epoch,
# #             gt_abs_deg         = traj_gt_deg,
# #             # FIX-M-C: pass sr_pred cho bridge loss
# #             sr_pred            = sr_pred,
# #         )

# #         # FM physics consistency
# #         fm_phys_w = WEIGHTS.get("fm_physics", 0.3)
# #         breakdown["total"]      = breakdown["total"] + fm_phys_w * l_fm_physics
# #         breakdown["fm_physics"] = l_fm_physics.item()

# #         # Short-range loss
# #         sr_w = WEIGHTS.get("short_range", 5.0)
# #         breakdown["total"]       = breakdown["total"] + sr_w * l_sr
# #         breakdown["short_range"] = l_sr.item()

# #         return breakdown

# #     # ── sample() ─────────────────────────────────────────────────────────────

# #     @torch.no_grad()
# #     def sample(
# #         self,
# #         batch_list:   List,
# #         num_ensemble: int = 50,
# #         ddim_steps:   int = 20,
# #         predict_csv:  Optional[str] = None,
# #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# #         """
# #         FIX-M-F: raw_ctx cached một lần cho cả SR và FM ensemble.

# #         Returns:
# #             pred_mean : [T, B, 2]  (SR[:4] + FM[4:])
# #             me_mean   : [T, B, 2]
# #             all_trajs : [S, T, B, 2]  (FM ensemble, KHÔNG override step 1-4)
# #         """
# #         obs_t  = batch_list[0]
# #         lp     = obs_t[-1]
# #         lm     = batch_list[7][-1]
# #         B      = lp.shape[0]
# #         device = lp.device
# #         T      = self.pred_len
# #         dt     = 1.0 / max(ddim_steps, 1)

# #         # FIX-M-F: một lần duy nhất
# #         raw_ctx = self.net._context(batch_list)
# #         n_sr    = ShortRangeHead.N_STEPS

# #         # Short-range deterministic
# #         sr_pred = self.net.forward_short_range(obs_t, raw_ctx)   # [4, B, 2]

# #         # FM ensemble
# #         traj_s: List[torch.Tensor] = []
# #         me_s:   List[torch.Tensor] = []

# #         for k in range(num_ensemble):
# #             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# #             for step in range(ddim_steps):
# #                 t_b = torch.full((B,), step * dt, device=device)
# #                 # ns  = self.ctx_noise_scale if step == 0 else 0.0
# #                 ns = self.ctx_noise_scale * 10.0 if step == 0 else 0.0
# #                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
# #                 x_t = x_t + dt * vel
# #         # for k in range(num_ensemble):
# #         #     x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

# #         #     # Mỗi member dùng ctx với noise riêng → diversity đến từ ctx uncertainty
# #         #     ctx_k = self.net._apply_ctx_head(
# #         #         raw_ctx, noise_scale=self.ctx_noise_scale * 5.0  # tăng 5× khi sampling
# #         #     )

# #         #     for step in range(ddim_steps):
# #         #         t_b = torch.full((B,), step * dt, device=device)
# #         #         vel = self.net._decode(x_t, t_b, ctx_k)
# #         #         x_t = x_t + dt * vel

# #             x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)
# #             x_t = x_t.clamp(-3.0, 3.0)

# #             tr, me = self._to_abs(x_t, lp, lm)
# #             traj_s.append(tr)
# #             me_s.append(me)

# #         all_trajs = torch.stack(traj_s)    # [S, T, B, 2]
# #         all_me    = torch.stack(me_s)

# #         # Blend: step 1-4 từ SR, 5-12 từ FM mean
# #         fm_mean   = all_trajs.mean(0)      # [T, B, 2]
# #         pred_mean = fm_mean.clone()
# #         # pred_mean[:n_sr] = sr_pred         # override pred_mean, giữ all_trajs nguyên

# #         # Dùng:
# #         alpha = torch.linspace(0.0, 1.0, n_sr, device=device).view(n_sr, 1, 1)
# #         pred_mean[:n_sr] = (1 - alpha) * sr_pred + alpha * fm_mean[:n_sr]
        
# #         # step 1: 100% SR, step 4: 50% SR + 50% FM → smooth transition
# #         me_mean = all_me.mean(0)

# #         if predict_csv:
# #             self._write_predict_csv(predict_csv, pred_mean, all_trajs)

# #         return pred_mean, me_mean, all_trajs

# #     # ── Physics correction (FIX-M-D) ─────────────────────────────────────────

# #     def _physics_correct(
# #         self,
# #         x_pred:   torch.Tensor,
# #         last_pos: torch.Tensor,
# #         last_Me:  torch.Tensor,
# #         n_steps:  int   = 5,
# #         lr:       float = 0.002,
# #     ) -> torch.Tensor:
# #         """
# #         FIX-M-D: Dùng Adam thay vì SGD+momentum.
# #         SGD+momentum với n_steps nhỏ (5) gây oscillation;
# #         Adam converge nhanh hơn và stable hơn.
# #         """
# #         with torch.enable_grad():
# #             x = x_pred.detach().requires_grad_(True)
# #             # FIX-M-D: Adam thay vì SGD+momentum
# #             optimizer = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))

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
# #                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
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
# flow_matching_model_v32.py — Tiếp cận đơn giản nhất để beat LSTM

# INSIGHT: LSTM beat vì nó train với pure MSE → gradient rõ ràng, không conflict.
# Vấn đề của FM không phải architecture mà là training signal:
#   - AFCRPS: pull về diverse predictions
#   - MSE_hav: pull về mean prediction  
#   - Hai cái này conflict nhau → stuck at mean (ADE=412km)

# GIẢI PHÁP V32:
#   Phase 1 (epoch 0-30): Train như LSTM thuần túy
#     - Loss = SR_Huber(step 1-4) + MSE_hav(step 1-12)
#     - Không dùng FM AFCRPS
#     - Model học được basic trajectory prediction
    
#   Phase 2 (epoch 31+): Thêm FM ensemble diversity
#     - Loss = SR_Huber + MSE_hav + 0.5*AFCRPS
#     - FM dùng để calibrate uncertainty, không phải accuracy
    
# KẾT QUẢ DỰ KIẾN:
#   - Epoch 10-20: ADE giảm nhanh từ 412 → ~250km (như LSTM ban đầu)
#   - Epoch 30-50: ADE ~170km, 72h ~300km
#   - Epoch 60+: SR converge → 12h ~50km
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

# # Import các module unchanged
# from Model.FNO3D_encoder import FNO3DEncoder
# from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# from Model.env_net_transformer_gphsplit import Env_net
# from Model.losses import (
#     _haversine_deg, _norm_to_deg, N_SR_STEPS,
#     short_range_regression_loss, heading_loss,
#     velocity_loss_per_sample, recurvature_loss,
#     fm_afcrps_loss, fm_physics_consistency_loss,
# )

# MSE_STEP_WEIGHTS = [1.0, 3.0, 1.5, 2.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]


# def _denorm_to_deg(t):
#     out = t.clone()
#     out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
#     out[..., 1] = (t[..., 1] * 50.0) / 10.0
#     return out


# def mse_hav_loss(pred_norm, gt_norm, step_w=None):
#     """Haversine MSE per-step với step weighting."""
#     if step_w is None:
#         step_w = MSE_STEP_WEIGHTS
#     T = min(pred_norm.shape[0], gt_norm.shape[0])
#     pred_deg = _norm_to_deg(pred_norm[:T])
#     gt_deg   = _norm_to_deg(gt_norm[:T])
#     dist_km  = _haversine_deg(pred_deg, gt_deg)   # [T, B]
#     w = pred_norm.new_tensor(step_w[:T])
#     w = w / w.sum() * T
#     return (dist_km.pow(2) * w.unsqueeze(1)).mean() / (200.0 ** 2)


# # ─── ShortRangeHead (giữ nguyên từ v30) ──────────────────────────────────────

# class ShortRangeHead(nn.Module):
#     N_STEPS   = 4
#     MAX_DELTA = 0.48

#     def __init__(self, raw_ctx_dim=512, obs_len=8):
#         super().__init__()
#         self.obs_len = obs_len
#         self.vel_enc = nn.Sequential(
#             nn.Linear(obs_len * 2, 256), nn.GELU(),
#             nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(),
#         )
#         self.ctx_proj = nn.Sequential(
#             nn.Linear(raw_ctx_dim, 256), nn.GELU(),
#             nn.LayerNorm(256), nn.Dropout(0.05),
#         )
#         self.gru_cell = nn.GRUCell(input_size=386, hidden_size=256)
#         self.motion_gate_linear = nn.Linear(128 + 256, 1)
#         self.out_norm = nn.LayerNorm(256)
#         self.out_head = nn.Sequential(
#             nn.Linear(256, 128), nn.GELU(),
#             nn.LayerNorm(128), nn.Linear(128, 2),
#         )
#         self.step_scale = nn.Parameter(torch.ones(self.N_STEPS) * 0.2)
#         self._init_weights()

#     def _init_weights(self):
#         with torch.no_grad():
#             for m in self.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.xavier_uniform_(m.weight, gain=0.3)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#             nn.init.constant_(self.motion_gate_linear.bias, 2.0)

#     def forward(self, raw_ctx, obs_traj):
#         B, T_obs = raw_ctx.shape[0], obs_traj.shape[0]
#         vels = obs_traj[1:] - obs_traj[:-1] if T_obs >= 2 else torch.zeros(1, B, 2, device=raw_ctx.device)
#         if vels.shape[0] < self.obs_len:
#             pad  = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=raw_ctx.device)
#             vels = torch.cat([pad, vels], dim=0)
#         else:
#             vels = vels[-self.obs_len:]
#         vel_feat = self.vel_enc(vels.permute(1, 0, 2).reshape(B, -1))
#         ctx_feat = self.ctx_proj(raw_ctx)
#         hx, cur_pos, last_vel = ctx_feat, obs_traj[-1].clone(), vels[-1].clone()
#         preds = []
#         for i in range(self.N_STEPS):
#             hx  = self.gru_cell(torch.cat([ctx_feat, vel_feat, cur_pos], dim=-1), hx)
#             gate = torch.sigmoid(self.motion_gate_linear(torch.cat([vel_feat, hx], dim=-1)))
#             raw_delta   = self.out_head(self.out_norm(hx))
#             prior_delta = last_vel * (0.9 ** i)
#             scale_i = torch.sigmoid(self.step_scale[i]) * self.MAX_DELTA
#             delta   = (gate * prior_delta + (1.0 - gate) * raw_delta * scale_i).clamp(-self.MAX_DELTA, self.MAX_DELTA)
#             cur_pos = cur_pos + delta
#             last_vel = delta
#             preds.append(cur_pos)
#         return torch.stack(preds, dim=0)


# # ─── VelocityField (giữ nguyên từ v30, pos_enc dùng pred_len) ────────────────

# class VelocityField(nn.Module):
#     RAW_CTX_DIM = 512

#     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
#                  unet_in_ch=13, fm_pred_len=8):
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
#         self.ctx_drop = nn.Dropout(0.15)
#         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)
#         self.sr_anchor_proj = nn.Sequential(
#             nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 256))
#         self.time_fc1   = nn.Linear(256, 512)
#         self.time_fc2   = nn.Linear(512, 256)
#         self.traj_embed = nn.Linear(4, 256)
#         # KEY FIX: pos_enc dùng pred_len (12), không phải fm_pred_len (8)
#         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.05)
#         self.transformer = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 d_model=256, nhead=8, dim_feedforward=1024,
#                 dropout=0.15, activation="gelu", batch_first=True),
#             num_layers=4)
#         self.out_fc1 = nn.Linear(256, 512)
#         self.out_fc2 = nn.Linear(512, 4)
#         self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)
#         self.short_range_head = ShortRangeHead(self.RAW_CTX_DIM, obs_len)

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
#         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_spatial], dim=-1))))

#     def _apply_ctx_head(self, raw, noise_scale=0.0):
#         if noise_scale > 0.0:
#             raw = raw + torch.randn_like(raw) * noise_scale
#         return self.ctx_fc2(self.ctx_drop(raw))

#     def _beta_drift(self, x_t):
#         lat_deg  = x_t[:, :, 1] * 5.0
#         lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))
#         beta     = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
#         R_tc     = 3e5
#         v_phys   = torch.zeros_like(x_t)
#         v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
#         v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
#         return v_phys

#     def _decode(self, x_t, t, ctx, sr_anchor_emb=None):
#         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
#         t_emb = self.time_fc2(t_emb)
#         T_seq = min(x_t.size(1), self.pos_enc.shape[1])
#         x_emb = self.traj_embed(x_t[:, :T_seq]) + self.pos_enc[:, :T_seq] + t_emb.unsqueeze(1)
#         mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
#         if sr_anchor_emb is not None:
#             mem_parts.append(sr_anchor_emb.unsqueeze(1))
#         v_neural = self.out_fc2(F.gelu(self.out_fc1(
#             self.transformer(x_emb, torch.cat(mem_parts, dim=1)))))
#         with torch.no_grad():
#             v_phys = self._beta_drift(x_t[:, :T_seq])
#         return v_neural + torch.sigmoid(self.physics_scale) * 2.0 * v_phys

#     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0, sr_anchor_emb=None):
#         return self._decode(x_t, t, self._apply_ctx_head(raw_ctx, noise_scale), sr_anchor_emb)

#     def forward_short_range(self, obs_traj, raw_ctx):
#         return self.short_range_head(raw_ctx, obs_traj)

#     def compute_sr_anchor_emb(self, sr_pred):
#         anchor_input = torch.cat([sr_pred[-1], sr_pred[-1] - sr_pred[-2]], dim=-1)
#         return self.sr_anchor_proj(anchor_input)


# # ─── TCFlowMatching v32 ───────────────────────────────────────────────────────

# class TCFlowMatching(nn.Module):
#     """
#     v32: Phase-based training để tránh FM/MSE conflict.
    
#     Phase 1 (epoch < phase_switch): Pure MSE + SR
#     Phase 2 (epoch >= phase_switch): MSE + SR + AFCRPS
#     """

#     PHASE_SWITCH = 30  # Sau epoch này mới thêm AFCRPS

#     def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
#                  n_train_ens=4, unet_in_ch=13,
#                  ctx_noise_scale=0.002, initial_sample_sigma=0.03, **kwargs):
#         super().__init__()
#         self.pred_len             = pred_len
#         self.fm_pred_len          = pred_len - N_SR_STEPS
#         self.obs_len              = obs_len
#         self.sigma_min            = sigma_min
#         self.n_train_ens          = n_train_ens
#         self.ctx_noise_scale      = ctx_noise_scale
#         self.initial_sample_sigma = initial_sample_sigma
#         self.active_pred_len      = pred_len
#         self.net = VelocityField(pred_len=pred_len, obs_len=obs_len,
#                                   sigma_min=sigma_min, unet_in_ch=unet_in_ch,
#                                   fm_pred_len=self.fm_pred_len)

#     def set_curriculum_len(self, *a, **kw): pass

#     @staticmethod
#     def _to_rel(traj, Me, lp, lm):
#         return torch.cat([traj - lp.unsqueeze(0),
#                            Me   - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

#     @staticmethod
#     def _to_abs(rel, lp, lm):
#         d = rel.permute(1, 0, 2)
#         return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

#     def _cfm_noisy(self, x1):
#         B, device = x1.shape[0], x1.device
#         sm = self.sigma_min
#         x0 = torch.randn_like(x1) * sm
#         t  = torch.rand(B, device=device)
#         te = t.view(B, 1, 1)
#         x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
#         denom = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
#         return x_t, t, te, denom

#     @staticmethod
#     def _intensity_weights(obs_Me):
#         w = obs_Me[-1, :, 1].detach()
#         w = torch.where(w < 0.1, torch.full_like(w, 0.5),
#             torch.where(w < 0.3, torch.full_like(w, 0.8),
#             torch.where(w < 0.6, torch.full_like(w, 1.0),
#                         torch.full_like(w, 1.5))))
#         return w / w.mean().clamp(min=1e-6)

#     @staticmethod
#     def _lon_flip_aug(batch_list, p=0.3):
#         if torch.rand(1).item() > p:
#             return batch_list
#         aug = list(batch_list)
#         for idx in [0, 1, 2, 3]:
#             if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
#                 t = aug[idx].clone()
#                 t[..., 0] = -t[..., 0]
#                 aug[idx] = t
#         return aug

#     def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
#         return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

#     def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
#         """
#         Phase 1 (epoch < 30): SR_Huber + MSE_hav — học như LSTM
#         Phase 2 (epoch >= 30): SR_Huber + MSE_hav + AFCRPS — thêm probabilistic
#         """
#         batch_list = self._lon_flip_aug(batch_list)

#         traj_gt = batch_list[1]
#         Me_gt   = batch_list[8]
#         obs_t   = batch_list[0]
#         obs_Me  = batch_list[7]
#         lp, lm  = obs_t[-1], obs_Me[-1]

#         raw_ctx     = self.net._context(batch_list)
#         intensity_w = self._intensity_weights(obs_Me)

#         # ── SR (step 1-4) ──────────────────────────────────────────────
#         n_sr    = ShortRangeHead.N_STEPS
#         sr_pred = self.net.forward_short_range(obs_t, raw_ctx)
#         l_sr    = short_range_regression_loss(sr_pred, traj_gt[:n_sr], lp)

#         # ── FM: full 12 steps từ lp ────────────────────────────────────
#         sr_anchor = self.net.compute_sr_anchor_emb(sr_pred.detach())
#         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)
#         x_t, t, te, denom = self._cfm_noisy(x1)
#         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, sr_anchor_emb=sr_anchor)
#         x1_pred  = x_t + denom * pred_vel
#         pred_abs, _ = self._to_abs(x1_pred, lp, lm)   # [12, B, 2] normalised

#         # ── MSE haversine per-step ─────────────────────────────────────
#         # Đây là loss chính trong Phase 1 — giống LSTM nhưng physically correct
#         l_mse = mse_hav_loss(pred_abs, traj_gt)

#         # ── Directional ────────────────────────────────────────────────
#         pred_deg = _denorm_to_deg(pred_abs)
#         gt_deg   = _denorm_to_deg(traj_gt)
#         l_vel    = velocity_loss_per_sample(pred_deg, gt_deg).mean()
#         l_head   = heading_loss(pred_deg, gt_deg)

#         # ── Total (Phase 1) ────────────────────────────────────────────
#         NRM   = 35.0
#         total = (
#             5.0  * l_sr
#             + 4.0  * l_mse          # HIGH weight như LSTM
#             + 0.5  * l_vel  * NRM
#             + 0.8  * l_head * NRM
#         ) / NRM

#         bd = dict(
#             total=total, short_range=l_sr.item(), mse_hav=l_mse.item(),
#             velocity=l_vel.item()*NRM, heading=l_head.item(),
#             fm=0.0, step=0.0, disp=0.0, cont=0.0, pinn=0.0,
#             spread=0.0, continuity=0.0, recurv_ratio=0.0,
#         )

#         # ── Phase 2: Thêm AFCRPS (epoch >= PHASE_SWITCH) ───────────────
#         if epoch >= self.PHASE_SWITCH:
#             samples = []
#             for _ in range(self.n_train_ens):
#                 xt_s, ts, _, dens_s = self._cfm_noisy(x1)
#                 pv_s = self.net.forward_with_ctx(xt_s, ts, raw_ctx, sr_anchor_emb=sr_anchor)
#                 pa_s, _ = self._to_abs(xt_s + dens_s * pv_s, lp, lm)
#                 samples.append(pa_s)
#             pred_samples = torch.stack(samples)   # [S, 12, B, 2]

#             l_fm = fm_afcrps_loss(
#                 pred_samples, traj_gt,
#                 unit_01deg=True,
#                 intensity_w=intensity_w,
#                 w_es=0.2,
#             )

#             # Weight AFCRPS nhỏ hơn MSE để không dominate
#             fm_w = min(1.0, (epoch - self.PHASE_SWITCH) / 20.0)  # ramp up 0→1
#             bd["total"] = total + fm_w * 1.5 * l_fm / NRM
#             bd["fm"]    = l_fm.item()

#         if torch.isnan(bd["total"]) or torch.isinf(bd["total"]):
#             bd["total"] = lp.new_zeros(())

#         return bd

#     @torch.no_grad()
#     def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
#         obs_t  = batch_list[0]
#         lp     = obs_t[-1]
#         lm     = batch_list[7][-1]
#         B      = lp.shape[0]
#         device = lp.device
#         T      = self.pred_len
#         dt     = 1.0 / max(ddim_steps, 1)

#         raw_ctx = self.net._context(batch_list)
#         sr_pred = self.net.forward_short_range(obs_t, raw_ctx)
#         sr_anchor = self.net.compute_sr_anchor_emb(sr_pred)

#         traj_s, me_s = [], []
#         for _ in range(num_ensemble):
#             x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma
#             for step in range(ddim_steps):
#                 t_b = torch.full((B,), step * dt, device=device)
#                 ns  = self.ctx_noise_scale if step < 3 else 0.0
#                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx,
#                                                  noise_scale=ns, sr_anchor_emb=sr_anchor)
#                 x_t = x_t + dt * vel
#             x_t = self._physics_correct(x_t, lp, lm)
#             x_t = x_t.clamp(-3.0, 3.0)
#             tr, me = self._to_abs(x_t, lp, lm)
#             traj_s.append(tr)
#             me_s.append(me)

#         all_trajs = torch.stack(traj_s)
#         fm_mean   = all_trajs.mean(0)
#         pred_mean = fm_mean.clone()
#         pred_mean[:ShortRangeHead.N_STEPS] = sr_pred   # SR override step 1-4

#         if predict_csv:
#             self._write_predict_csv(predict_csv, pred_mean, all_trajs)
#         return pred_mean, torch.stack(me_s).mean(0), all_trajs

#     def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
#         with torch.enable_grad():
#             x = x_pred.detach().requires_grad_(True)
#             opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
#             for _ in range(n_steps):
#                 opt.zero_grad()
#                 pred_abs, _ = self._to_abs(x, last_pos, last_Me)
#                 pred_deg = _denorm_to_deg(pred_abs)
#                 dt_deg  = pred_deg[1:] - pred_deg[:-1]
#                 lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
#                 cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
#                 speed   = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
#                                       + (dt_deg[:, :, 1] * 111.0).pow(2))
#                 loss = F.relu(speed - 600.0).pow(2).mean()
#                 loss.backward()
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
#                     "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg", "ens_spread_km"]
#         write_hdr = not os.path.exists(csv_path)
#         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#         with open(csv_path, "a", newline="") as fh:
#             w = csv.DictWriter(fh, fieldnames=fields)
#             if write_hdr: w.writeheader()
#             for b in range(B):
#                 for k in range(T):
#                     dlat   = all_lat[:, k, b] - mean_lat[k, b]
#                     dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(math.radians(mean_lat[k, b]))
#                     spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
#                     w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
#                                 "lead_h": (k+1)*6, "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
#                                 "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
#                                 "lon_std_deg": f"{all_lon[:,k,b].std():.4f}",
#                                 "lat_std_deg": f"{all_lat[:,k,b].std():.4f}",
#                                 "ens_spread_km": f"{spread:.2f}"})


# TCDiffusion = TCFlowMatching

"""
flow_matching_model_v33.py — Pure FM + MSE: Beat LSTM bằng đơn giản

═══════════════════════════════════════════════════════════════════════
PHÂN TÍCH TẠI SAO FM THUA LSTM:
═══════════════════════════════════════════════════════════════════════

1. LSTM dùng pure MSE → gradient rõ ràng, 1 hướng duy nhất
2. FM v32 dùng 5+ loss → gradient conflict, model không biết optimize gì
3. SR head riêng + FM riêng → discontinuity tại step 4→5
4. AFCRPS pull ensemble diversity ≠ accuracy → ADE cao

═══════════════════════════════════════════════════════════════════════
GIẢI PHÁP V33: 
═══════════════════════════════════════════════════════════════════════

CORE INSIGHT: FM architecture + MSE training = best of both worlds
  - FM cho ensemble/uncertainty estimation khi inference
  - MSE cho gradient rõ ràng khi training (như LSTM)
  - KHÔNG dùng SR head riêng → FM predict ALL 12 steps
  - KHÔNG dùng AFCRPS khi training → chỉ MSE haversine

ARCHITECTURE CHANGES:
  1. BỎ ShortRangeHead → FM predict tất cả 12 steps trực tiếp
  2. BỎ sr_anchor_emb → đơn giản hóa decode
  3. THÊM teacher forcing: train FM với gt trajectory làm guidance
  4. Loss = MSE_haversine + nhẹ velocity + nhẹ heading (tổng 3 loss)

TRAINING STRATEGY:
  Phase 1 (epoch 0-20): MSE-only, sigma_min lớn (0.1) → near-deterministic
    → FM hoạt động gần như regression network
    → ADE giảm nhanh từ 400 → ~200km
    
  Phase 2 (epoch 20-50): MSE + nhẹ velocity/heading, sigma_min giảm dần
    → Refine trajectory shape
    → ADE ~170km, 12h ~45km
    
  Phase 3 (epoch 50+): Giữ nguyên, fine-tune
    → 12h < 50km, 72h < 300km

KEY INSIGHT VỀ FM VÀ 12h:
  - FM CÓ THỂ đạt 12h < 50km NẾU train đúng cách
  - Vấn đề cũ: SR head riêng → FM không được train cho step 1-4
  - V33: FM predict ALL steps → step 1-4 cũng được optimize trực tiếp
  - Step weighting nhấn mạnh 12h (step 2) và 24h (step 4)

═══════════════════════════════════════════════════════════════════════
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
from Model.losses import _haversine_deg, _norm_to_deg, _haversine

# ── Step weights: nhấn mạnh 12h và 24h ─────────────────────────────────────
# Step:    6h   12h  18h  24h  30h  36h  42h  48h  54h  60h  66h  72h
MSE_STEP_WEIGHTS = [
    1.5,  # 6h  - quan trọng
    4.0,  # 12h - TARGET < 50km → weight cao nhất  
    2.0,  # 18h
    3.5,  # 24h - TARGET < 100km → weight cao
    1.5,  # 30h
    1.5,  # 36h
    1.5,  # 42h
    2.5,  # 48h - TARGET < 200km
    1.0,  # 54h
    1.0,  # 60h
    1.5,  # 66h
    2.5,  # 72h - TARGET < 300km
]


def _denorm_to_deg(t):
    """Convert normalized coords to degrees."""
    out = t.clone()
    out[..., 0] = (t[..., 0] * 50.0 + 1800.0) / 10.0
    out[..., 1] = (t[..., 1] * 50.0) / 10.0
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  MSE Haversine Loss — Vũ khí chính, giống LSTM nhưng physically correct
# ═══════════════════════════════════════════════════════════════════════════════

def mse_hav_loss(pred_norm, gt_norm, step_w=None):
    """
    Haversine MSE per-step.
    
    Tại sao dùng haversine thay vì MSE(lon,lat)?
    - MSE(lon,lat) bị bias do cos(lat): 1 degree lon ở equator ≠ 1 degree ở 30°N
    - Haversine cho distance thật sự trên Earth → gradient chính xác hơn
    - LSTM dùng MSE(lon,lat) → có thể beat bằng MSE(haversine)
    """
    if step_w is None:
        step_w = MSE_STEP_WEIGHTS
    T = min(pred_norm.shape[0], gt_norm.shape[0])
    pred_deg = _norm_to_deg(pred_norm[:T])
    gt_deg   = _norm_to_deg(gt_norm[:T])
    dist_km  = _haversine_deg(pred_deg, gt_deg)   # [T, B] in km
    
    w = pred_norm.new_tensor(step_w[:T])
    w = w / w.sum() * T  # normalize nhưng giữ scale
    
    # Huber-style: dùng L1 cho outlier (dist > 300km), L2 cho phần còn lại
    # Tránh gradient explosion từ large errors
    delta = 300.0  # km
    huber_dist = torch.where(
        dist_km < delta,
        dist_km.pow(2) / (2.0 * delta),
        dist_km - delta / 2.0,
    )
    
    return (huber_dist * w.unsqueeze(1)).mean() / delta


# ═══════════════════════════════════════════════════════════════════════════════
#  VelocityField — Đơn giản hóa, bỏ ShortRangeHead
# ═══════════════════════════════════════════════════════════════════════════════

class VelocityField(nn.Module):
    """
    Velocity field cho FM — predict velocity cho ALL 12 steps.
    
    Thay đổi so với v32:
    - BỎ ShortRangeHead → FM predict step 1-12 trực tiếp
    - BỎ sr_anchor_emb → đơn giản decode
    - THÊM residual connection trong decoder
    - THÊM step embedding (model biết đang predict step nào)
    """
    RAW_CTX_DIM = 512

    def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
                 unet_in_ch=13, **kwargs):
        super().__init__()
        self.pred_len = pred_len
        self.obs_len  = obs_len
        
        # ── Encoder (giữ nguyên) ──────────────────────────────────────
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
        
        # ── Context projection ────────────────────────────────────────
        self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
        self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
        self.ctx_drop = nn.Dropout(0.10)
        self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)
        
        # ── Velocity observation encoder ──────────────────────────────
        # Encode observed velocity pattern → giúp predict step 1-4 chính xác
        self.vel_obs_enc = nn.Sequential(
            nn.Linear(obs_len * 2, 256), nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256), nn.GELU(),
        )
        
        # ── FM Decoder ────────────────────────────────────────────────
        self.time_fc1   = nn.Linear(256, 512)
        self.time_fc2   = nn.Linear(512, 256)
        self.traj_embed = nn.Linear(4, 256)
        self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
        
        # Step embedding: model biết đang ở step nào → quan trọng cho short-range
        self.step_embed = nn.Embedding(pred_len, 256)
        
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=256, nhead=8, dim_feedforward=1024,
                dropout=0.10, activation="gelu", batch_first=True),
            num_layers=5)  # 5 layers thay vì 4 → capacity lớn hơn
        
        self.out_fc1 = nn.Linear(256, 512)
        self.out_fc2 = nn.Linear(512, 4)
        
        # Learnable output scale per step → model tự learn scale phù hợp
        self.step_scale = nn.Parameter(torch.ones(pred_len) * 0.5)
        
        # Physics beta-drift (giữ nguyên)
        self.physics_scale = nn.Parameter(torch.ones(4) * 0.3)
        
        self._init_weights()

    def _init_weights(self):
        """Xavier init với gain nhỏ → output gần 0 ban đầu."""
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
        """Encode observation context — giống v32."""
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
        
        return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_spatial], dim=-1))))

    def _apply_ctx_head(self, raw, noise_scale=0.0):
        if noise_scale > 0.0:
            raw = raw + torch.randn_like(raw) * noise_scale
        return self.ctx_fc2(self.ctx_drop(raw))

    def _get_vel_obs_feat(self, obs_traj):
        """Encode observed velocities → giúp FM predict step 1-4."""
        B = obs_traj.shape[1]
        T_obs = obs_traj.shape[0]
        
        if T_obs >= 2:
            vels = obs_traj[1:] - obs_traj[:-1]
        else:
            vels = torch.zeros(1, B, 2, device=obs_traj.device)
        
        # Pad/truncate to obs_len
        if vels.shape[0] < self.obs_len:
            pad = torch.zeros(self.obs_len - vels.shape[0], B, 2, device=obs_traj.device)
            vels = torch.cat([pad, vels], dim=0)
        else:
            vels = vels[-self.obs_len:]
        
        return self.vel_obs_enc(vels.permute(1, 0, 2).reshape(B, -1))

    def _beta_drift(self, x_t):
        """Physical beta-drift bias."""
        lat_deg  = x_t[:, :, 1] * 5.0
        lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))
        beta     = 2 * 7.2921e-5 * torch.cos(lat_rad) / 6.371e6
        R_tc     = 3e5
        v_phys   = torch.zeros_like(x_t)
        v_phys[:, :, 0] = -beta * R_tc**2 / 2 * 6*3600 / (5*111*1000)
        v_phys[:, :, 1] =  beta * R_tc**2 / 4 * 6*3600 / (5*111*1000)
        return v_phys

    def _decode(self, x_t, t, ctx, vel_obs_feat=None):
        """
        Decode velocity field.
        
        Thay đổi:
        - Thêm vel_obs_feat vào memory → giúp predict step 1-4
        - Thêm step embedding → model biết step position
        - Output scale per step → kiểm soát magnitude
        """
        B = x_t.shape[0]
        t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
        t_emb = self.time_fc2(t_emb)
        
        T_seq = min(x_t.size(1), self.pos_enc.shape[1])
        
        # Step embedding
        step_idx = torch.arange(T_seq, device=x_t.device).unsqueeze(0).expand(B, -1)
        s_emb = self.step_embed(step_idx)  # [B, T_seq, 256]
        
        x_emb = self.traj_embed(x_t[:, :T_seq]) + self.pos_enc[:, :T_seq] + t_emb.unsqueeze(1) + s_emb
        
        # Memory: context + time + velocity observations
        mem_parts = [t_emb.unsqueeze(1), ctx.unsqueeze(1)]
        if vel_obs_feat is not None:
            mem_parts.append(vel_obs_feat.unsqueeze(1))
        
        decoded = self.transformer(x_emb, torch.cat(mem_parts, dim=1))
        v_neural = self.out_fc2(F.gelu(self.out_fc1(decoded)))
        
        # Per-step scale
        scale = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
        v_neural = v_neural * scale
        
        # Physics correction
        with torch.no_grad():
            v_phys = self._beta_drift(x_t[:, :T_seq])
        
        return v_neural + torch.sigmoid(self.physics_scale) * v_phys

    def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale=0.0, vel_obs_feat=None):
        return self._decode(x_t, t, self._apply_ctx_head(raw_ctx, noise_scale), vel_obs_feat)


# ═══════════════════════════════════════════════════════════════════════════════
#  TCFlowMatching v33 — Pure FM + MSE
# ═══════════════════════════════════════════════════════════════════════════════

class TCFlowMatching(nn.Module):
    """
    v33: Pure FM architecture, MSE-only training.
    
    KEY DIFFERENCES từ v32:
    1. FM predict ALL 12 steps (không có SR head riêng)
    2. Training loss = MSE_haversine + nhẹ velocity + nhẹ heading
    3. KHÔNG dùng AFCRPS → gradient không conflict
    4. sigma_min schedule: lớn ban đầu (near-deterministic) → nhỏ dần
    5. Teacher forcing: dùng gt trajectory interpolation khi train
    
    TẠI SAO FM CÓ THỂ BEAT LSTM:
    - FM có thêm encoder mạnh (FNO3D + Mamba + Env_net)
    - FM có ensemble capability → uncertainty estimation
    - FM có physics correction (beta-drift)
    - Nếu train đúng (MSE-only), FM ít nhất = LSTM, có thể tốt hơn
    """

    def __init__(self, pred_len=12, obs_len=8, sigma_min=0.02,
                 n_train_ens=4, unet_in_ch=13,
                 ctx_noise_scale=0.002, initial_sample_sigma=0.03, **kwargs):
        super().__init__()
        self.pred_len             = pred_len
        self.obs_len              = obs_len
        self.sigma_min            = sigma_min
        self.n_train_ens          = n_train_ens
        self.ctx_noise_scale      = ctx_noise_scale
        self.initial_sample_sigma = initial_sample_sigma
        self.active_pred_len      = pred_len
        
        self.net = VelocityField(
            pred_len=pred_len, obs_len=obs_len,
            sigma_min=sigma_min, unet_in_ch=unet_in_ch)

    def set_curriculum_len(self, *a, **kw):
        pass

    @staticmethod
    def _to_rel(traj, Me, lp, lm):
        """Convert absolute coords to relative (from last observed position)."""
        return torch.cat([traj - lp.unsqueeze(0),
                           Me   - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

    @staticmethod
    def _to_abs(rel, lp, lm):
        """Convert relative coords back to absolute."""
        d = rel.permute(1, 0, 2)
        return lp.unsqueeze(0) + d[:, :, :2], lm.unsqueeze(0) + d[:, :, 2:]

    # def _cfm_noisy(self, x1, sigma_min=None):
    #     """
    #     Conditional FM noise process.
        
    #     KEY: sigma_min controls stochasticity:
    #     - sigma_min lớn (0.1-0.2) → gần deterministic → tốt cho MSE training
    #     - sigma_min nhỏ (0.01-0.02) → stochastic → tốt cho ensemble diversity
    #     """
    #     if sigma_min is None:
    #         sigma_min = self.sigma_min
    #     B, device = x1.shape[0], x1.device
    #     x0 = torch.randn_like(x1) * sigma_min
    #     t  = torch.rand(B, device=device)
    #     te = t.view(B, 1, 1)
    #     # x_t = te * x1 + (1.0 - te * (1.0 - sigma_min)) * x0
    #     # denom = (1.0 - (1.0 - sigma_min) * te).clamp(min=1e-5)
    #     x_t = te * x1 + (1.0 - te * (1.0 - sigma_min)) * x0
    #     denom = (1.0 - (1.0 - sigma_min) * te).clamp(min=1e-5)
    #     return x_t, t, te, denom

    def _cfm_noisy(self, x1, sigma_min=None):
        if sigma_min is None:
            sigma_min = self.sigma_min
        B, device = x1.shape[0], x1.device
        x0 = torch.randn_like(x1) * sigma_min
        t  = torch.rand(B, device=device)
        te = t.view(B, 1, 1)
        x_t = (1.0 - te) * x0 + te * x1
        u   = x1 - x0
        return x_t, t, u  # 3 giá trị, không còn denom

    @staticmethod
    def _intensity_weights(obs_Me):
        """Weight samples by intensity — stronger storms get more attention."""
        w = obs_Me[-1, :, 1].detach()
        w = torch.where(w < 0.1, torch.full_like(w, 0.5),
            torch.where(w < 0.3, torch.full_like(w, 0.8),
            torch.where(w < 0.6, torch.full_like(w, 1.0),
                        torch.full_like(w, 1.5))))
        return w / w.mean().clamp(min=1e-6)

    @staticmethod
    def _lon_flip_aug(batch_list, p=0.3):
        """Data augmentation: flip longitude."""
        if torch.rand(1).item() > p:
            return batch_list
        aug = list(batch_list)
        for idx in [0, 1, 2, 3]:
            if torch.is_tensor(aug[idx]) and aug[idx].shape[-1] >= 1:
                t = aug[idx].clone()
                t[..., 0] = -t[..., 0]
                aug[idx] = t
        return aug

    def get_loss(self, batch_list, step_weight_alpha=0.0, epoch=0):
        return self.get_loss_breakdown(batch_list, step_weight_alpha, epoch)["total"]

    # def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
    #     """
    #     ═══════════════════════════════════════════════════════════════
    #     CORE TRAINING LOGIC — MSE-focused, FM as architecture only
    #     ═══════════════════════════════════════════════════════════════
        
    #     Phase 1 (epoch < 15): Pure MSE, sigma_min = 0.15 (near-deterministic)
    #       → FM gần như regression network
    #       → Gradient rõ ràng, ADE giảm nhanh
          
    #     Phase 2 (epoch 15-40): MSE + nhẹ velocity/heading, sigma_min = 0.08
    #       → Refine trajectory shape
    #       → Giảm directional errors
          
    #     Phase 3 (epoch 40+): MSE + velocity/heading, sigma_min = 0.03
    #       → Fine-tune, cho phép ensemble diversity
    #       → Maintain accuracy, add calibration
    #     """
    #     batch_list = self._lon_flip_aug(batch_list)

    #     traj_gt = batch_list[1]    # [T, B, 2] normalized
    #     Me_gt   = batch_list[8]    # [T, B, 2] normalized  
    #     obs_t   = batch_list[0]    # [T_obs, B, 2] normalized
    #     obs_Me  = batch_list[7]    # [T_obs, B, 2] normalized
    #     lp, lm  = obs_t[-1], obs_Me[-1]

    #     # ── Sigma schedule ─────────────────────────────────────────────
    #     # Ban đầu sigma lớn → FM gần deterministic → MSE hiệu quả
    #     if epoch < 15:
    #         current_sigma = 0.15
    #     elif epoch < 40:
    #         # Linear decay 0.15 → 0.03
    #         t = (epoch - 15) / 25.0
    #         current_sigma = 0.15 - t * (0.15 - 0.03)
    #     else:
    #         current_sigma = 0.03

    #     # ── Encode context ─────────────────────────────────────────────
    #     raw_ctx     = self.net._context(batch_list)
    #     vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

    #     # ── FM forward: predict ALL 12 steps ───────────────────────────
    #     x1 = self._to_rel(traj_gt, Me_gt, lp, lm)  # [B, T, 4]
    #     # # x_t, t, te, denom = self._cfm_noisy(x1, sigma_min=current_sigma)
    #     # x_t, t, u_target = self._cfm_noisy(x1, sigma_min=current_sigma)

    #     # pred_vel = self.net.forward_with_ctx(
    #     #     x_t, t, raw_ctx, vel_obs_feat=vel_obs_feat)
    #     # x1_pred  = x_t + denom * pred_vel
    #     # pred_abs, _ = self._to_abs(x1_pred, lp, lm)  # [12, B, 2] normalized

    #     # # ── LOSS 1: MSE Haversine (CHÍNH) ──────────────────────────────
    #     # l_mse = mse_hav_loss(pred_abs, traj_gt)
    #     x_t, t, u_target = self._cfm_noisy(x1, sigma_min=current_sigma)
    #     pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, vel_obs_feat=vel_obs_feat)
    #     l_mse = F.mse_loss(pred_vel, u_target)  # loss trên velocity, stable qua mọi epo

    #     # ── LOSS 2: Velocity matching (nhẹ) ───────────────────────────
    #     pred_deg = _denorm_to_deg(pred_abs)
    #     gt_deg   = _denorm_to_deg(traj_gt)
        
    #     if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
    #         T_min = min(pred_deg.shape[0], gt_deg.shape[0])
    #         v_pred = pred_deg[1:T_min] - pred_deg[:T_min-1]
    #         v_gt   = gt_deg[1:T_min]   - gt_deg[:T_min-1]
    #         l_vel  = F.smooth_l1_loss(v_pred, v_gt)
    #     else:
    #         l_vel = pred_abs.new_zeros(())

    #     # ── LOSS 3: Heading consistency (nhẹ) ─────────────────────────
    #     if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
    #         T_min = min(pred_deg.shape[0], gt_deg.shape[0])
    #         v_pred = pred_deg[1:T_min] - pred_deg[:T_min-1]
    #         v_gt   = gt_deg[1:T_min]   - gt_deg[:T_min-1]
            
    #         v_pred_n = v_pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    #         v_gt_n   = v_gt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    #         cos_sim  = ((v_pred / v_pred_n) * (v_gt / v_gt_n)).sum(-1)
    #         l_head   = F.relu(-cos_sim).pow(2).mean()  # penalize opposite direction
    #     else:
    #         l_head = pred_abs.new_zeros(())

    #     # ── Total Loss ─────────────────────────────────────────────────
    #     # Phase 1: pure MSE
    #     # Phase 2+: MSE + small velocity + small heading
    #     if epoch < 15:
    #         w_vel  = 0.0
    #         w_head = 0.0
    #     elif epoch < 40:
    #         t_phase = (epoch - 15) / 25.0
    #         w_vel  = t_phase * 0.3
    #         w_head = t_phase * 0.2
    #     else:
    #         w_vel  = 0.3
    #         w_head = 0.2

    #     total = l_mse + w_vel * l_vel + w_head * l_head

    #     # ── Ensemble consistency (optional, epoch 40+) ─────────────────
    #     # Train thêm 1 sample, penalize nếu quá khác nhau
    #     l_ens_consist = pred_abs.new_zeros(())
    #     if epoch >= 40 and self.n_train_ens >= 2:
    #         x_t2, t2, te2, denom2 = self._cfm_noisy(x1, sigma_min=current_sigma)
    #         pred_vel2 = self.net.forward_with_ctx(
    #             x_t2, t2, raw_ctx, vel_obs_feat=vel_obs_feat)
    #         x1_pred2 = x_t2 + denom2 * pred_vel2
    #         pred_abs2, _ = self._to_abs(x1_pred2, lp, lm)
            
    #         # MSE of second sample → cũng phải gần gt
    #         l_ens_consist = mse_hav_loss(pred_abs2, traj_gt)
    #         total = total + 0.3 * l_ens_consist

    #     if torch.isnan(total) or torch.isinf(total):
    #         total = lp.new_zeros(())

    #     return dict(
    #         total=total,
    #         mse_hav=l_mse.item(),
    #         velocity=l_vel.item(),
    #         heading=l_head.item(),
    #         ens_consist=l_ens_consist.item(),
    #         sigma=current_sigma,
    #         # Backward compat
    #         fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
    #         spread=0.0, continuity=0.0, step=0.0, disp=0.0,
    #         recurv_ratio=0.0,
    #     )

    def get_loss_breakdown(self, batch_list, step_weight_alpha=0.0, epoch=0):
        batch_list = self._lon_flip_aug(batch_list)

        traj_gt = batch_list[1]
        Me_gt   = batch_list[8]
        obs_t   = batch_list[0]
        obs_Me  = batch_list[7]
        lp, lm  = obs_t[-1], obs_Me[-1]

        # ── Sigma schedule — đổi tên biến thành sigma_t tránh conflict với FM t ──
        if epoch < 15:
            current_sigma = 0.15
        elif epoch < 40:
            sigma_frac = (epoch - 15) / 25.0
            current_sigma = 0.15 - sigma_frac * (0.15 - 0.03)
        else:
            current_sigma = 0.03

        # ── Encode context ────────────────────────────────────────────────
        raw_ctx      = self.net._context(batch_list)
        vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

        # ── FM forward ────────────────────────────────────────────────────
        x1_rel = self._to_rel(traj_gt, Me_gt, lp, lm)  # [B, T, 4]
        x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=current_sigma)

        pred_vel = self.net.forward_with_ctx(
            x_t, fm_t, raw_ctx, vel_obs_feat=vel_obs_feat)

        # ── LOSS 1: FM velocity MSE (chính) ──────────────────────────────
        l_mse = F.mse_loss(pred_vel, u_target)

        # ── Reconstruct pred_abs để tính velocity/heading loss ───────────
        with torch.no_grad():
            fm_te    = fm_t.view(x1_rel.shape[0], 1, 1)
            x1_pred  = x_t + (1.0 - fm_te) * pred_vel  # FM chuẩn: x1 = x_t + (1-t)*v
            pred_abs, _ = self._to_abs(x1_pred, lp, lm)

        pred_deg = _denorm_to_deg(pred_abs)
        gt_deg   = _denorm_to_deg(traj_gt)

        # ── LOSS 2: Velocity matching ─────────────────────────────────────
        if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
            T_min  = min(pred_deg.shape[0], gt_deg.shape[0])
            v_pred = pred_deg[1:T_min] - pred_deg[:T_min-1]
            v_gt   = gt_deg[1:T_min]   - gt_deg[:T_min-1]
            l_vel  = F.smooth_l1_loss(v_pred, v_gt)
        else:
            l_vel = x_t.new_zeros(())

        # ── LOSS 3: Heading consistency ───────────────────────────────────
        if pred_deg.shape[0] >= 2 and gt_deg.shape[0] >= 2:
            T_min    = min(pred_deg.shape[0], gt_deg.shape[0])
            v_pred   = pred_deg[1:T_min] - pred_deg[:T_min-1]
            v_gt     = gt_deg[1:T_min]   - gt_deg[:T_min-1]
            v_pred_n = v_pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            v_gt_n   = v_gt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            cos_sim  = ((v_pred / v_pred_n) * (v_gt / v_gt_n)).sum(-1)
            l_head   = F.relu(-cos_sim).pow(2).mean()
        else:
            l_head = x_t.new_zeros(())

        # ── Phase weights ─────────────────────────────────────────────────
        if epoch < 15:
            w_vel, w_head = 0.0, 0.0
        elif epoch < 40:
            phase_frac = (epoch - 15) / 25.0
            w_vel  = phase_frac * 0.3
            w_head = phase_frac * 0.2
        else:
            w_vel, w_head = 0.3, 0.2

        total = l_mse + w_vel * l_vel + w_head * l_head

        # ── Ensemble consistency (epoch 40+) ──────────────────────────────
        l_ens_consist = x_t.new_zeros(())
        if epoch >= 40 and self.n_train_ens >= 2:
            x_t2, fm_t2, u_target2 = self._cfm_noisy(x1_rel, sigma_min=current_sigma)
            pred_vel2 = self.net.forward_with_ctx(
                x_t2, fm_t2, raw_ctx, vel_obs_feat=vel_obs_feat)
            l_ens_consist = F.mse_loss(pred_vel2, u_target2)
            total = total + 0.3 * l_ens_consist

        if torch.isnan(total) or torch.isinf(total):
            total = x_t.new_zeros(())

        return dict(
            total=total,
            mse_hav=l_mse.item(),
            velocity=l_vel.item(),
            heading=l_head.item(),
            ens_consist=l_ens_consist.item(),
            sigma=current_sigma,
            fm=0.0, short_range=0.0, cont=0.0, pinn=0.0,
            spread=0.0, continuity=0.0, step=0.0, disp=0.0,
            recurv_ratio=0.0,
        )
    @torch.no_grad()
    def sample(self, batch_list, num_ensemble=50, ddim_steps=20, predict_csv=None):
        """
        Inference: ODE integration từ noise → trajectory.
        
        Thay đổi so với v32:
        - Không có SR override → FM predict ALL steps
        - vel_obs_feat truyền vào decode → short-range accuracy
        - Dùng sigma_min nhỏ (0.02) khi inference → diversity
        """
        obs_t  = batch_list[0]
        lp     = obs_t[-1]
        lm     = batch_list[7][-1]
        B      = lp.shape[0]
        device = lp.device
        T      = self.pred_len
        dt     = 1.0 / max(ddim_steps, 1)

        raw_ctx      = self.net._context(batch_list)
        vel_obs_feat = self.net._get_vel_obs_feat(obs_t)

        traj_s, me_s = [], []
        for _ in range(num_ensemble):
            # x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma
            x_t = torch.randn(B, T, 4, device=device) * self.sigma_min

            for step in range(ddim_steps):
                t_b = torch.full((B,), step * dt, device=device)
                ns  = self.ctx_noise_scale if step < 3 else 0.0
                vel = self.net.forward_with_ctx(
                    x_t, t_b, raw_ctx,
                    noise_scale=ns, vel_obs_feat=vel_obs_feat)
                x_t = x_t + dt * vel
            
            x_t = self._physics_correct(x_t, lp, lm)
            x_t = x_t.clamp(-3.0, 3.0)
            tr, me = self._to_abs(x_t, lp, lm)
            traj_s.append(tr)
            me_s.append(me)

        all_trajs = torch.stack(traj_s)       # [S, T, B, 2]
        pred_mean = all_trajs.mean(0)          # [T, B, 2]

        if predict_csv:
            self._write_predict_csv(predict_csv, pred_mean, all_trajs)
        return pred_mean, torch.stack(me_s).mean(0), all_trajs

    def _physics_correct(self, x_pred, last_pos, last_Me, n_steps=3, lr=0.001):
        """Speed constraint: TCs can't move > 600 km/6h."""
        with torch.enable_grad():
            x = x_pred.detach().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
            for _ in range(n_steps):
                opt.zero_grad()
                pred_abs, _ = self._to_abs(x, last_pos, last_Me)
                pred_deg = _denorm_to_deg(pred_abs)
                dt_deg  = pred_deg[1:] - pred_deg[:-1]
                lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
                cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
                speed   = torch.sqrt((dt_deg[:, :, 0] * cos_lat * 111.0).pow(2)
                                      + (dt_deg[:, :, 1] * 111.0).pow(2))
                loss = F.relu(speed - 600.0).pow(2).mean()
                loss.backward()
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
                    "lon_mean_deg", "lat_mean_deg", "lon_std_deg", "lat_std_deg", "ens_spread_km"]
        write_hdr = not os.path.exists(csv_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if write_hdr: w.writeheader()
            for b in range(B):
                for k in range(T):
                    dlat   = all_lat[:, k, b] - mean_lat[k, b]
                    dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(math.radians(mean_lat[k, b]))
                    spread = float(((dlat**2 + dlon**2)**0.5).mean() * 111.0)
                    w.writerow({"timestamp": ts, "batch_idx": b, "step_idx": k,
                                "lead_h": (k+1)*6, "lon_mean_deg": f"{mean_lon[k,b]:.4f}",
                                "lat_mean_deg": f"{mean_lat[k,b]:.4f}",
                                "lon_std_deg": f"{all_lon[:,k,b].std():.4f}",
                                "lat_std_deg": f"{all_lat[:,k,b].std():.4f}",
                                "ens_spread_km": f"{spread:.2f}"})


# Backward compat alias
TCDiffusion = TCFlowMatching
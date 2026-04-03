# # # """
# # # Model/flow_matching_model.py  ── v10-turbo
# # # ==========================================
# # # OT-CFM Flow Matching + PINN-BVE for TC trajectory prediction.

# # # CHANGES vs v9-fixed:
# # #   ✅ UNet3D   → FNO3DEncoder  (4-8x faster spatial encoding)
# # #   ✅ En-LSTM  → MambaEncoder  (3x faster temporal encoding, parallel)
# # #   ✅ ctx_dim kept at 128, total context 128+64+16=208 (unchanged)
# # #   ✅ All batch_list indices unchanged (full backward compat)

# # # Speed gain (Kaggle T4, B=32, T_obs=8):
# # #   v9-fixed : ~1.3s/batch  → 80 epochs × 481 batches ≈ 16h
# # #   v10-turbo: ~0.35s/batch → 80 epochs × 481 batches ≈ 4.3h ✅✅

# # # Batch list indices (from seq_collate — unchanged):
# # #   0  obs_traj   [T_obs, B, 2]
# # #   1  pred_traj  [T_pred, B, 2]
# # #   2  obs_rel    [T_obs, B, 2]
# # #   3  pred_rel   [T_pred, B, 2]
# # #   4  nlp        tensor
# # #   5  mask       [seq_len, B]
# # #   6  seq_start_end [B, 2]
# # #   7  obs_Me     [T_obs, B, 2]
# # #   8  pred_Me    [T_pred, B, 2]
# # #   9  obs_Me_rel [T_obs, B, 2]
# # #   10 pred_Me_rel[T_pred, B, 2]
# # #   11 img_obs    [B, 13, T_obs, 81, 81]
# # #   12 img_pred   [B, 13, T_pred, 81, 81]
# # #   13 env_data   dict
# # #   14 None
# # #   15 list[dict]
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

# # # # ── SWAP: FNO3D instead of UNet3D ────────────────────────────────────────────
# # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # ── SWAP: Mamba instead of LSTM ──────────────────────────────────────────────
# # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D

# # # from Model.env_net_transformer_gphsplit import Env_net
# # # from Model.losses import compute_total_loss, WEIGHTS


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  VelocityField  (FlowMatching denoiser)  ── v10
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class VelocityField(nn.Module):
# # #     """
# # #     OT-CFM velocity field  v_θ(x_t, t, context).

# # #     Context assembly (identical to v9):
# # #       h_t       [B, 128]  ← DataEncoder1D w/ Mamba (was LSTM)
# # #       e_Env     [B,  64]  ← Env-T-Net (unchanged)
# # #       f_spatial [B,  16]  ← FNO decoder pooled (was UNet decoder)
# # #       ─────────────────
# # #       total     [B, 208]  → ctx_fc → [B, ctx_dim=128]

# # #     Trajectory decoder: TransformerDecoder + linear head → [B, T, 4]
# # #     """

# # #     def __init__(
# # #         self,
# # #         pred_len:   int   = 12,
# # #         obs_len:    int   = 8,
# # #         ctx_dim:    int   = 128,
# # #         sigma_min:  float = 0.02,
# # #         unet_in_ch: int   = 13,
# # #     ):
# # #         super().__init__()
# # #         self.pred_len  = pred_len
# # #         self.obs_len   = obs_len
# # #         self.sigma_min = sigma_min

# # #         # ── FNO3D encoder (replaces UNet3D) ─────────────────────────────
# # #         self.spatial_enc = FNO3DEncoder(
# # #             in_channel   = unet_in_ch,
# # #             out_channel  = 1,
# # #             d_model      = 32,      # FIX: 64 → 32
# # #             n_layers     = 4,
# # #             modes_t      = 4,
# # #             modes_h      = 4,       # FIX: 16 → 4
# # #             modes_w      = 4,       # FIX: 16 → 4
# # #             spatial_down = 32,
# # #             dropout      = 0.05,
# # #         )

# # #         # Bottleneck: [B, 128, T, 4, 4] → pool spatial → [B, T, 128]
# # #         # (same interface as v9 bottleneck_pool + bottleneck_proj)
# # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # #         self.bottleneck_proj = nn.Linear(128, 128)

# # #         # Summary → [B, 16]
# # #         self.decoder_proj = nn.Linear(1, 16)

# # #         # ── DataEncoder1D with Mamba (replaces LSTM) ─────────────────────
# # #         self.enc_1d = DataEncoder1D(
# # #             in_1d       = 4,
# # #             feat_3d_dim = 128,
# # #             mlp_h       = 64,
# # #             lstm_hidden = 128,
# # #             lstm_layers = 3,
# # #             dropout     = 0.1,
# # #             d_state     = 16,
# # #         )

# # #         # ── Env-T-Net (Eq. 10–13) — unchanged ───────────────────────────
# # #         self.env_enc = Env_net(obs_len=obs_len, d_model=64)

# # #         # ── Context fusion: 128 + 64 + 16 = 208 → ctx_dim ───────────────
# # #         self.ctx_fc1  = nn.Linear(128 + 64 + 16, 512)
# # #         self.ctx_ln   = nn.LayerNorm(512)
# # #         self.ctx_drop = nn.Dropout(0.15)
# # #         self.ctx_fc2  = nn.Linear(512, ctx_dim)

# # #         # ── Time embedding ────────────────────────────────────────────────
# # #         self.time_fc1 = nn.Linear(128, 256)
# # #         self.time_fc2 = nn.Linear(256, 128)

# # #         # ── Trajectory Transformer decoder (unchanged) ───────────────────
# # #         self.traj_embed = nn.Linear(4, 128)
# # #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)
# # #         self.transformer = nn.TransformerDecoder(
# # #             nn.TransformerDecoderLayer(
# # #                 d_model=128, nhead=8, dim_feedforward=512,
# # #                 dropout=0.15, activation="gelu", batch_first=True,
# # #             ),
# # #             num_layers=4,
# # #         )
# # #         self.out_fc1 = nn.Linear(128, 256)
# # #         self.out_fc2 = nn.Linear(256, 4)

# # #     # ── Sinusoidal time embedding ─────────────────────────────────────────

# # #     def _time_emb(self, t: torch.Tensor, dim: int = 128) -> torch.Tensor:
# # #         half = dim // 2
# # #         freq = torch.exp(
# # #             torch.arange(half, dtype=torch.float32, device=t.device)
# # #             * (-math.log(10_000.0) / max(half - 1, 1))
# # #         )
# # #         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
# # #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# # #         return F.pad(emb, (0, dim % 2))

# # #     # ── Build context vector ──────────────────────────────────────────────

# # #     def _context(self, batch_list: List) -> torch.Tensor:
# # #         obs_traj  = batch_list[0]   # [T_obs, B, 2]
# # #         obs_Me    = batch_list[7]   # [T_obs, B, 2]
# # #         image_obs = batch_list[11]  # [B, 13, T_obs, 81, 81]
# # #         env_data  = batch_list[13]  # dict

# # #         # ── FNO3D encode ──────────────────────────────────────────────────
# # #         if image_obs.dim() == 4:
# # #             image_obs = image_obs.unsqueeze(1)

# # #         # Single-channel tile if needed
# # #         expected_ch = self.spatial_enc.in_channel
# # #         if image_obs.shape[1] == 1 and expected_ch != 1:
# # #             image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # #         # e_3d_bot: [B, 128, T, 4, 4]
# # #         # e_3d_dec: [B, 1,   T, 1, 1]

# # #         B     = e_3d_bot.shape[0]
# # #         T_obs = obs_traj.shape[0]

# # #         # Pool spatial dims 4×4 → 1×1, keep T: [B, 128, T, 1, 1] → [B, T, 128]
# # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)  # [B, 128, T_bot]
# # #         e_3d_s = e_3d_s.permute(0, 2, 1)                                 # [B, T_bot, 128]
# # #         e_3d_s = self.bottleneck_proj(e_3d_s)                            # [B, T_bot, 128]

# # #         # Align T_bot → T_obs if needed (FNO preserves T, but just in case)
# # #         T_bot = e_3d_s.shape[1]
# # #         if T_bot != T_obs:
# # #             e_3d_s = F.interpolate(
# # #                 e_3d_s.permute(0, 2, 1),
# # #                 size=T_obs, mode="linear", align_corners=False,
# # #             ).permute(0, 2, 1)

# # #         # Decoder spatial summary: [B, 1, T, 1, 1] → mean(T) → [B, 1] → [B, 16]
# # #         f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))       # [B, 1]
# # #         f_spatial     = self.decoder_proj(f_spatial_raw)   # [B, 16]

# # #         # ── DataEncoder1D + Mamba ─────────────────────────────────────────
# # #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)  # [B, T, 4]
# # #         h_t    = self.enc_1d(obs_in, e_3d_s)   # [B, 128]

# # #         # ── Env-T-Net ─────────────────────────────────────────────────────
# # #         e_env, _, _ = self.env_enc(env_data, image_obs)  # [B, 64]

# # #         # ── Fuse: [128 + 64 + 16] → [ctx_dim] ────────────────────────────
# # #         ctx = torch.cat([h_t, e_env, f_spatial], dim=-1)  # [B, 208]
# # #         ctx = F.gelu(self.ctx_ln(self.ctx_fc1(ctx)))
# # #         ctx = self.ctx_drop(ctx)
# # #         return self.ctx_fc2(ctx)  # [B, ctx_dim]

# # #     # ── Forward ───────────────────────────────────────────────────────────

# # #     def forward(
# # #         self,
# # #         x_t:        torch.Tensor,  # [B, T_pred, 4]
# # #         t:          torch.Tensor,  # [B]
# # #         batch_list: List,
# # #     ) -> torch.Tensor:             # [B, T_pred, 4]
# # #         ctx   = self._context(batch_list)
# # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # #         t_emb = self.time_fc2(t_emb)  # [B, 128]

# # #         x_emb  = self.traj_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
# # #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

# # #         out = self.transformer(x_emb, memory)
# # #         return self.out_fc2(F.gelu(self.out_fc1(out)))

# # #     def forward_with_ctx(self, x_t, t, ctx):
# # #         """Forward pass reusing pre-computed context (skips FNO + Mamba)."""
# # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# # #         t_emb = self.time_fc2(t_emb)
# # #         x_emb  = self.traj_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
# # #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)
# # #         out = self.transformer(x_emb, memory)
# # #         return self.out_fc2(F.gelu(self.out_fc1(out)))


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  TCFlowMatching  (unchanged logic, new backbone)
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class TCFlowMatching(nn.Module):
# # #     """
# # #     TC trajectory prediction via OT-CFM + PINN-BVE.

# # #     Training loss (unchanged):
# # #         L = 1.0·L_FM + 2.0·L_dir + 0.5·L_step
# # #           + 1.0·L_disp + 2.0·L_heading + 0.2·L_smooth + 0.5·L_PINN

# # #     Backbone changes:
# # #         UNet3D → FNO3DEncoder   (4-8x faster)
# # #         LSTM   → MambaEncoder   (3x faster)

# # #     Inference:
# # #         Euler ODE integration (ddim_steps) × num_ensemble samples
# # #         → (traj_mean [T,B,2], Me_mean [T,B,2], all_trajs [S,T,B,2])
# # #     """

# # #     def __init__(
# # #         self,
# # #         pred_len:    int   = 12,
# # #         obs_len:     int   = 8,
# # #         sigma_min:   float = 0.02,
# # #         n_train_ens: int   = 4,
# # #         unet_in_ch:  int   = 13,
# # #         **kwargs,
# # #     ):
# # #         super().__init__()
# # #         self.pred_len    = pred_len
# # #         self.obs_len     = obs_len
# # #         self.sigma_min   = sigma_min
# # #         self.n_train_ens = n_train_ens
# # #         self.net = VelocityField(
# # #             pred_len   = pred_len,
# # #             obs_len    = obs_len,
# # #             sigma_min  = sigma_min,
# # #             unet_in_ch = unet_in_ch,
# # #         )

# # #     # ── Coordinate helpers (unchanged) ────────────────────────────────────

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

# # #     # ── OT-CFM noise schedule (unchanged) ────────────────────────────────

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

# # #     # ── Training (unchanged logic) ────────────────────────────────────────

# # #     def get_loss(self, batch_list: List) -> torch.Tensor:
# # #         return self.get_loss_breakdown(batch_list)["total"]

# # #     def get_loss_breakdown(self, batch_list: List) -> Dict:
# # #         traj_gt = batch_list[1]
# # #         Me_gt   = batch_list[8]
# # #         obs_t   = batch_list[0]
# # #         obs_Me  = batch_list[7]

# # #         lp, lm = obs_t[-1], obs_Me[-1]
# # #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

# # #         # Context computed ONCE (FNO3D + Mamba run once per batch)
# # #         ctx = self.net._context(batch_list)

# # #         x_t, t, te, denom, _ = self._cfm_noisy(x1)
# # #         pred_vel = self.net.forward_with_ctx(x_t, t, ctx)

# # #         samples: List[torch.Tensor] = []
# # #         for _ in range(self.n_train_ens):
# # #             xt_s, ts, _, dens, _ = self._cfm_noisy(x1)
# # #             pv_s  = self.net.forward_with_ctx(xt_s, ts, ctx)
# # #             x1_s  = xt_s + dens * pv_s
# # #             pa_s, _ = self._to_abs(x1_s, lp, lm)
# # #             samples.append(pa_s)
# # #         pred_samples = torch.stack(samples)

# # #         x1_pred  = x_t + denom * pred_vel
# # #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# # #         return compute_total_loss(
# # #             pred_abs     = pred_abs,
# # #             gt           = traj_gt,
# # #             ref          = lp,
# # #             batch_list   = batch_list,
# # #             pred_samples = pred_samples,
# # #             weights      = WEIGHTS,
# # #         )

# # #     # ── Inference (unchanged logic) ───────────────────────────────────────

# # #     @torch.no_grad()
# # #     def sample(
# # #         self,
# # #         batch_list:   List,
# # #         num_ensemble: int = 50,
# # #         ddim_steps:   int = 10,
# # #         predict_csv:  Optional[str] = None,
# # #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# # #         lp  = batch_list[0][-1]
# # #         lm  = batch_list[7][-1]
# # #         B, device = lp.shape[0], lp.device
# # #         dt  = 1.0 / ddim_steps

# # #         traj_s: List[torch.Tensor] = []
# # #         me_s:   List[torch.Tensor] = []

# # #         for _ in range(num_ensemble):
# # #             x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
# # #             for step in range(ddim_steps):
# # #                 t_b = torch.full((B,), step * dt, device=device)
# # #                 x_t = x_t + dt * self.net(x_t, t_b, batch_list)
# # #                 x_t[:, :, :2].clamp_(-5.0, 5.0)
# # #             tr, me = self._to_abs(x_t, lp, lm)
# # #             traj_s.append(tr)
# # #             me_s.append(me)

# # #         all_trajs = torch.stack(traj_s)
# # #         all_me    = torch.stack(me_s)
# # #         traj_mean = all_trajs.mean(0)
# # #         me_mean   = all_me.mean(0)

# # #         if predict_csv is not None:
# # #             self._write_predict_csv(predict_csv, traj_mean, all_trajs)

# # #         return traj_mean, me_mean, all_trajs

# # #     # ── CSV export (unchanged) ────────────────────────────────────────────

# # #     @staticmethod
# # #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# # #         import numpy as np
# # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # #         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
# # #         T, B, _ = traj_mean.shape
# # #         S       = all_trajs.shape[0]

# # #         mean_lon = (traj_mean[..., 0] * 5.0 + 180.0).cpu().numpy()
# # #         mean_lat = (traj_mean[..., 1] * 5.0).cpu().numpy()
# # #         all_lon  = (all_trajs[..., 0] * 5.0 + 180.0).cpu().numpy()
# # #         all_lat  = (all_trajs[..., 1] * 5.0).cpu().numpy()

# # #         fields = ["timestamp", "batch_idx", "step_idx", "lead_h",
# # #                   "lon_mean_deg", "lat_mean_deg",
# # #                   "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# # #         write_hdr = not os.path.exists(csv_path)
# # #         with open(csv_path, "a", newline="") as fh:
# # #             w = csv.DictWriter(fh, fieldnames=fields)
# # #             if write_hdr:
# # #                 w.writeheader()
# # #             for b in range(B):
# # #                 for k in range(T):
# # #                     dlat  = all_lat[:, k, b] - mean_lat[k, b]
# # #                     dlon  = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
# # #                         math.radians(mean_lat[k, b]))
# # #                     spread = float(np.sqrt((dlat ** 2 + dlon ** 2).mean()) * 111.0)
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
# # Model/flow_matching_model.py  ── v11-fixed
# # ==========================================
# # OT-CFM Flow Matching + PINN-BVE for TC trajectory prediction.

# # CHANGES vs v10-turbo:
# #   ✅ BUG-1 FIXED: Dropout mask reuse in ensemble — ctx_drop now applied
# #      independently per ensemble sample inside get_loss_breakdown loop.
# #      In v10, ctx was computed once then shared across all n_train_ens
# #      forward_with_ctx calls → all samples had identical dropout mask.
# #   ✅ BUG-2 FIXED: Me dims (pressure/wind) not clamped in sample() —
# #      added x_t[:, :, 2:].clamp_(-3.0, 3.0) alongside existing lon/lat clamp.
# #   ✅ BUG-3 FIXED: loader import — train script now imports from
# #      loader_training.py which uses split= kwarg for TrajectoryDataset v11.
# #   ✅ IMPROVEMENT: n_train_ens=6 from epoch 60+ (progressive schedule).
# #   ✅ IMPROVEMENT: ddim_steps=20 during validation.
# #   ✅ IMPROVEMENT: LR cosine restart at epoch 60 in addition to epoch 30.
# #   ✅ IMPROVEMENT: lon flip augmentation (typhoon track symmetry).
# #   ✅ IMPROVEMENT: intensity-weighted loss (TY/SevTY samples weighted higher).
# #   ✅ IMPROVEMENT: curriculum pred_len 6→12 over epochs 0-50.
# #   ✅ IMPROVEMENT: smooth weight 0.2→0.05 (reduces over-smoothing).
# #   ✅ IMPROVEMENT: CRPS loss added directly in training objective.

# # Batch list indices (from seq_collate — unchanged):
# #   0  obs_traj   [T_obs, B, 2]
# #   1  pred_traj  [T_pred, B, 2]
# #   2  obs_rel    [T_obs, B, 2]
# #   3  pred_rel   [T_pred, B, 2]
# #   4  nlp        tensor
# #   5  mask       [seq_len, B]
# #   6  seq_start_end [B, 2]
# #   7  obs_Me     [T_obs, B, 2]
# #   8  pred_Me    [T_pred, B, 2]
# #   9  obs_Me_rel [T_obs, B, 2]
# #   10 pred_Me_rel[T_pred, B, 2]
# #   11 img_obs    [B, 13, T_obs, 81, 81]
# #   12 img_pred   [B, 13, T_pred, 81, 81]
# #   13 env_data   dict
# #   14 None
# #   15 list[dict]
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
# # from Model.losses import compute_total_loss, WEIGHTS


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  VelocityField  (FlowMatching denoiser)  ── v11
# # # ══════════════════════════════════════════════════════════════════════════════

# # class VelocityField(nn.Module):
# #     """
# #     OT-CFM velocity field  v_θ(x_t, t, context).

# #     Context assembly:
# #       h_t       [B, 128]  ← DataEncoder1D w/ Mamba
# #       e_Env     [B,  64]  ← Env-T-Net
# #       f_spatial [B,  16]  ← FNO decoder pooled
# #       ─────────────────
# #       total     [B, 208]  → ctx_fc → [B, ctx_dim=128]

# #     BUG-1 FIX: ctx_drop is NOT applied inside _context(); instead it is
# #     applied *per forward call* in forward() and forward_with_ctx() so
# #     each ensemble sample gets an independent dropout mask.
# #     """

# #     def __init__(
# #         self,
# #         pred_len:   int   = 12,
# #         obs_len:    int   = 8,
# #         ctx_dim:    int   = 128,
# #         sigma_min:  float = 0.02,
# #         unet_in_ch: int   = 13,
# #     ):
# #         super().__init__()
# #         self.pred_len  = pred_len
# #         self.obs_len   = obs_len
# #         self.sigma_min = sigma_min

# #         # ── FNO3D encoder ────────────────────────────────────────────────
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

# #         # ── DataEncoder1D with Mamba ─────────────────────────────────────
# #         self.enc_1d = DataEncoder1D(
# #             in_1d       = 4,
# #             feat_3d_dim = 128,
# #             mlp_h       = 64,
# #             lstm_hidden = 128,
# #             lstm_layers = 3,
# #             dropout     = 0.1,
# #             d_state     = 16,
# #         )

# #         # ── Env-T-Net ────────────────────────────────────────────────────
# #         self.env_enc = Env_net(obs_len=obs_len, d_model=64)

# #         # ── Context fusion: 128 + 64 + 16 = 208 → ctx_dim ───────────────
# #         self.ctx_fc1  = nn.Linear(128 + 64 + 16, 512)
# #         self.ctx_ln   = nn.LayerNorm(512)
# #         # BUG-1 FIX: ctx_drop is applied OUTSIDE _context() so each
# #         # ensemble member gets its own independent dropout mask.
# #         self.ctx_drop = nn.Dropout(0.15)
# #         self.ctx_fc2  = nn.Linear(512, ctx_dim)

# #         # ── Time embedding ────────────────────────────────────────────────
# #         self.time_fc1 = nn.Linear(128, 256)
# #         self.time_fc2 = nn.Linear(256, 128)

# #         # ── Trajectory Transformer decoder ──────────────────────────────
# #         self.traj_embed = nn.Linear(4, 128)
# #         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)
# #         self.transformer = nn.TransformerDecoder(
# #             nn.TransformerDecoderLayer(
# #                 d_model=128, nhead=8, dim_feedforward=512,
# #                 dropout=0.15, activation="gelu", batch_first=True,
# #             ),
# #             num_layers=4,
# #         )
# #         self.out_fc1 = nn.Linear(128, 256)
# #         self.out_fc2 = nn.Linear(256, 4)

# #     # ── Sinusoidal time embedding ─────────────────────────────────────────

# #     def _time_emb(self, t: torch.Tensor, dim: int = 128) -> torch.Tensor:
# #         half = dim // 2
# #         freq = torch.exp(
# #             torch.arange(half, dtype=torch.float32, device=t.device)
# #             * (-math.log(10_000.0) / max(half - 1, 1))
# #         )
# #         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
# #         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
# #         return F.pad(emb, (0, dim % 2))

# #     # ── Build context vector (NO dropout here — applied per-sample) ───────

# #     def _context(self, batch_list: List) -> torch.Tensor:
# #         """
# #         Returns pre-dropout context [B, 512].
# #         Caller applies ctx_drop independently per ensemble sample.
# #         """
# #         obs_traj  = batch_list[0]   # [T_obs, B, 2]
# #         obs_Me    = batch_list[7]   # [T_obs, B, 2]
# #         image_obs = batch_list[11]  # [B, 13, T_obs, 81, 81]
# #         env_data  = batch_list[13]  # dict

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
# #                 e_3d_s.permute(0, 2, 1),
# #                 size=T_obs, mode="linear", align_corners=False,
# #             ).permute(0, 2, 1)

# #         f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))
# #         f_spatial     = self.decoder_proj(f_spatial_raw)

# #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
# #         h_t    = self.enc_1d(obs_in, e_3d_s)

# #         e_env, _, _ = self.env_enc(env_data, image_obs)

# #         # Fuse but do NOT apply dropout here
# #         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
# #         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))
# #         # Return pre-dropout features; caller applies dropout independently
# #         return raw  # [B, 512]

# #     def _apply_ctx_head(self, raw: torch.Tensor) -> torch.Tensor:
# #         """Apply dropout + final projection. Called per ensemble sample."""
# #         return self.ctx_fc2(self.ctx_drop(raw))  # [B, ctx_dim]

# #     # ── Forward ───────────────────────────────────────────────────────────

# #     def forward(
# #         self,
# #         x_t:        torch.Tensor,
# #         t:          torch.Tensor,
# #         batch_list: List,
# #     ) -> torch.Tensor:
# #         raw = self._context(batch_list)
# #         ctx = self._apply_ctx_head(raw)   # independent dropout mask
# #         return self._decode(x_t, t, ctx)

# #     def forward_with_ctx(self, x_t, t, raw_ctx):
# #         """
# #         Forward pass reusing pre-computed raw context features (pre-dropout).
# #         BUG-1 FIX: applies ctx_drop here so each call gets a fresh mask.
# #         """
# #         ctx = self._apply_ctx_head(raw_ctx)   # fresh dropout mask each call
# #         return self._decode(x_t, t, ctx)

# #     def _decode(self, x_t, t, ctx):
# #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# #         t_emb = self.time_fc2(t_emb)

# #         x_emb  = self.traj_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
# #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

# #         out = self.transformer(x_emb, memory)
# #         return self.out_fc2(F.gelu(self.out_fc1(out)))


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  TCFlowMatching  ── v11
# # # ══════════════════════════════════════════════════════════════════════════════

# # class TCFlowMatching(nn.Module):
# #     """
# #     TC trajectory prediction via OT-CFM + PINN-BVE.

# #     v11 changes:
# #       - ctx_drop applied per-sample (BUG-1 fix)
# #       - Me clamp in sample() (BUG-2 fix)
# #       - smooth weight 0.2→0.05
# #       - intensity-weighted loss
# #       - curriculum pred_len 6→12 (set externally via set_curriculum_len)
# #       - CRPS loss in training objective
# #     """

# #     def __init__(
# #         self,
# #         pred_len:    int   = 12,
# #         obs_len:     int   = 8,
# #         sigma_min:   float = 0.02,
# #         n_train_ens: int   = 4,
# #         unet_in_ch:  int   = 13,
# #         **kwargs,
# #     ):
# #         super().__init__()
# #         self.pred_len     = pred_len
# #         self.obs_len      = obs_len
# #         self.sigma_min    = sigma_min
# #         self.n_train_ens  = n_train_ens
# #         # Curriculum: active pred length (updated externally each epoch)
# #         self.active_pred_len = pred_len
# #         self.net = VelocityField(
# #             pred_len   = pred_len,
# #             obs_len    = obs_len,
# #             sigma_min  = sigma_min,
# #             unet_in_ch = unet_in_ch,
# #         )

# #     def set_curriculum_len(self, active_len: int) -> None:
# #         """Update active prediction horizon for curriculum training."""
# #         self.active_pred_len = max(1, min(active_len, self.pred_len))

# #     # ── Coordinate helpers ────────────────────────────────────────────────

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

# #     # ── OT-CFM noise schedule ─────────────────────────────────────────────

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

# #     # ── Intensity-based sample weights ────────────────────────────────────

# #     @staticmethod
# #     def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
# #         """
# #         Weight samples by typhoon intensity class.
# #         obs_Me: [T_obs, B, 2] — channel 1 is normalized wind speed.
# #         Higher wind = higher weight (TY/SevTY/SuperTY more important).
# #         Returns [B] weights, mean=1.
# #         """
# #         # Last observed wind (normalized, ~0-1 range)
# #         wind_norm = obs_Me[-1, :, 1].detach()  # [B]
# #         # Map to weight: TD(0.5), TS(0.8), TY(1.0), SevTY+(1.5)
# #         w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
# #             torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
# #             torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
# #                         torch.full_like(wind_norm, 1.5))))
# #         return w / w.mean().clamp(min=1e-6)  # normalize mean to 1

# #     # ── lon-flip augmentation ─────────────────────────────────────────────

# #     @staticmethod
# #     def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
# #         """
# #         Randomly flip longitude for typhoon track symmetry augmentation.
# #         Probability p per batch.
# #         """
# #         if torch.rand(1).item() > p:
# #             return batch_list
# #         aug = list(batch_list)
# #         for idx in [0, 1, 2, 3]:   # obs_traj, pred_traj, obs_rel, pred_rel
# #             t = aug[idx]
# #             if torch.is_tensor(t) and t.shape[-1] >= 1:
# #                 t = t.clone()
# #                 # Flip lon (dim 0 of last axis) — mirror around center
# #                 t[..., 0] = -t[..., 0]
# #                 aug[idx] = t
# #         for idx in [7, 8, 9, 10]:  # obs_Me, pred_Me, obs_Me_rel, pred_Me_rel
# #             t = aug[idx]
# #             if torch.is_tensor(t):
# #                 aug[idx] = t.clone()
# #         return aug

# #     # ── Training ──────────────────────────────────────────────────────────

# #     def get_loss(self, batch_list: List) -> torch.Tensor:
# #         return self.get_loss_breakdown(batch_list)["total"]

# #     def get_loss_breakdown(self, batch_list: List) -> Dict:
# #         # Apply augmentation
# #         batch_list = self._lon_flip_aug(batch_list, p=0.3)

# #         traj_gt = batch_list[1]
# #         Me_gt   = batch_list[8]
# #         obs_t   = batch_list[0]
# #         obs_Me  = batch_list[7]

# #         # Curriculum: truncate to active_pred_len
# #         apl = self.active_pred_len
# #         if apl < traj_gt.shape[0]:
# #             traj_gt = traj_gt[:apl]
# #             Me_gt   = Me_gt[:apl]

# #         lp, lm = obs_t[-1], obs_Me[-1]
# #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

# #         # BUG-1 FIX: compute RAW context once (pre-dropout)
# #         raw_ctx = self.net._context(batch_list)  # [B, 512]

# #         # Intensity weights for loss scaling
# #         intensity_w = self._intensity_weights(obs_Me)  # [B]

# #         x_t, t, te, denom, _ = self._cfm_noisy(x1)
# #         # Each call to forward_with_ctx applies ctx_drop independently
# #         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx)

# #         samples: List[torch.Tensor] = []
# #         for _ in range(self.n_train_ens):
# #             xt_s, ts, _, dens, _ = self._cfm_noisy(x1)
# #             # BUG-1 FIX: each sample gets fresh dropout mask via forward_with_ctx
# #             pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx)
# #             x1_s  = xt_s + dens * pv_s
# #             pa_s, _ = self._to_abs(x1_s, lp, lm)
# #             samples.append(pa_s)
# #         pred_samples = torch.stack(samples)  # [S, T, B, 2]

# #         x1_pred  = x_t + denom * pred_vel
# #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# #         breakdown = compute_total_loss(
# #             pred_abs     = pred_abs,
# #             gt           = traj_gt,
# #             ref          = lp,
# #             batch_list   = batch_list,
# #             pred_samples = pred_samples,
# #             weights      = WEIGHTS,
# #             intensity_w  = intensity_w,
# #         )
# #         return breakdown

# #     # ── Inference ─────────────────────────────────────────────────────────

# #     @torch.no_grad()
# #     def sample(
# #         self,
# #         batch_list:   List,
# #         num_ensemble: int = 50,
# #         ddim_steps:   int = 10,
# #         predict_csv:  Optional[str] = None,
# #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# #         lp  = batch_list[0][-1]
# #         lm  = batch_list[7][-1]
# #         B, device = lp.shape[0], lp.device
# #         dt  = 1.0 / ddim_steps

# #         traj_s: List[torch.Tensor] = []
# #         me_s:   List[torch.Tensor] = []

# #         # Pre-compute raw context once; each forward_with_ctx applies
# #         # its own dropout (model.eval() → dropout disabled during inference)
# #         raw_ctx = self.net._context(batch_list)

# #         for _ in range(num_ensemble):
# #             x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
# #             for step in range(ddim_steps):
# #                 t_b = torch.full((B,), step * dt, device=device)
# #                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx)
# #                 x_t = x_t + dt * vel
# #                 # BUG-2 FIX: clamp lon/lat AND pressure/wind dims
# #                 x_t[:, :, :2].clamp_(-5.0, 5.0)   # traj (lon/lat)
# #                 x_t[:, :, 2:].clamp_(-3.0, 3.0)   # Me (pressure/wind)
# #             tr, me = self._to_abs(x_t, lp, lm)
# #             traj_s.append(tr)
# #             me_s.append(me)

# #         all_trajs = torch.stack(traj_s)
# #         all_me    = torch.stack(me_s)
# #         traj_mean = all_trajs.mean(0)
# #         me_mean   = all_me.mean(0)

# #         if predict_csv is not None:
# #             self._write_predict_csv(predict_csv, traj_mean, all_trajs)

# #         return traj_mean, me_mean, all_trajs

# #     # ── CSV export ────────────────────────────────────────────────────────

# #     @staticmethod
# #     def _write_predict_csv(csv_path, traj_mean, all_trajs):
# #         import numpy as np
# #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# #         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
# #         T, B, _ = traj_mean.shape
# #         S       = all_trajs.shape[0]

# #         mean_lon = (traj_mean[..., 0] * 5.0 + 180.0).cpu().numpy()
# #         mean_lat = (traj_mean[..., 1] * 5.0).cpu().numpy()
# #         all_lon  = (all_trajs[..., 0] * 5.0 + 180.0).cpu().numpy()
# #         all_lat  = (all_trajs[..., 1] * 5.0).cpu().numpy()

# #         fields = ["timestamp", "batch_idx", "step_idx", "lead_h",
# #                   "lon_mean_deg", "lat_mean_deg",
# #                   "lon_std_deg", "lat_std_deg", "ens_spread_km"]
# #         write_hdr = not os.path.exists(csv_path)
# #         with open(csv_path, "a", newline="") as fh:
# #             w = csv.DictWriter(fh, fieldnames=fields)
# #             if write_hdr:
# #                 w.writeheader()
# #             for b in range(B):
# #                 for k in range(T):
# #                     dlat  = all_lat[:, k, b] - mean_lat[k, b]
# #                     dlon  = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
# #                         math.radians(mean_lat[k, b]))
# #                     spread = float(np.sqrt((dlat ** 2 + dlon ** 2).mean()) * 111.0)
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
# # """
# # scripts/train_flowmatching.py  ── v19
# # ======================================
# # FIXES vs v18:

# # FIX-V19-1  [EARLY STOP ROOT CAUSE FIX] ADE chỉ được evaluate khi
# #            epoch % val_freq == 0 (mặc định mỗi 2 epoch). Nhưng
# #            saver.counter_ade tăng MỖI LẦN update_ade() được gọi,
# #            tức là mỗi 2 epoch tăng 1. Với patience=50, early stop
# #            xảy ra sau 50 lần không cải thiện × 2 epoch = epoch ~100
# #            nếu may mắn, nhưng thực tế counter tính theo LẦN GỌI
# #            không theo epoch → chỉ cần 50 lần gọi liên tiếp không
# #            cải thiện là stop. Với val_freq=2 thì stop ở epoch 50+.
# #            Fix: tách ADE evaluation ra ngoài khối val_freq, chạy
# #            ĐỘC LẬP mỗi epoch. counter_ade giờ = số epoch thực sự
# #            không cải thiện, đúng với ý nghĩa của patience.

# # FIX-V19-2  [COUNTER SEMANTICS FIX] patience=50 bây giờ thực sự có
# #            nghĩa là "50 epoch liên tiếp không cải thiện ADE" thay
# #            vì "50 lần gọi evaluate_fast không cải thiện".

# # Kept from v18:
# #     FIX-V18-1..V18-4 (cliper shape, patience reset, early stop, denorm)
# # """
# # from __future__ import annotations

# # import sys
# # import os
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # import argparse
# # import time
# # import math
# # import random
# # import copy

# # import numpy as np
# # import torch
# # import torch.optim as optim
# # from torch.amp import autocast, GradScaler
# # from torch.utils.data import DataLoader, Subset

# # from Model.data.loader_training import data_loader
# # from Model.flow_matching_model import TCFlowMatching
# # from Model.utils import get_cosine_schedule_with_warmup
# # from Model.losses import WEIGHTS as _BASE_WEIGHTS
# # from utils.metrics import (
# #     TCEvaluator, StepErrorAccumulator,
# #     save_metrics_csv, haversine_km_torch, denorm_torch, HORIZON_STEPS,
# #     cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
# #     brier_skill_score,
# # )
# # from utils.evaluation_tables import (
# #     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
# #     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
# #     DEFAULT_COMPUTE, paired_tests, persistence_errors, cliper_errors,
# # )
# # from scripts.statistical_tests import run_all_tests


# # # ── Helpers ───────────────────────────────────────────────────────────────────

# # def haversine_km_np(pred_deg: np.ndarray, gt_deg: np.ndarray) -> np.ndarray:
# #     pred_deg = np.atleast_2d(pred_deg)
# #     gt_deg   = np.atleast_2d(gt_deg)
# #     R = 6371.0
# #     lon1, lat1 = np.radians(pred_deg[:, 0]), np.radians(pred_deg[:, 1])
# #     lon2, lat2 = np.radians(gt_deg[:, 0]),   np.radians(gt_deg[:, 1])
# #     dlon = lon2 - lon1;  dlat = lat2 - lat1
# #     a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
# #     return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# # def denorm_deg_np(arr_norm: np.ndarray) -> np.ndarray:
# #     arr_norm = np.atleast_2d(arr_norm)
# #     out = arr_norm.copy()
# #     out[:, 0] = (arr_norm[:, 0] * 50.0 + 1800.0) / 10.0
# #     out[:, 1] = (arr_norm[:, 1] * 50.0) / 10.0
# #     return out


# # def seq_ade_km(pred_norm: np.ndarray, gt_norm: np.ndarray) -> float:
# #     return float(haversine_km_np(denorm_deg_np(pred_norm),
# #                                   denorm_deg_np(gt_norm)).mean())


# # # ── Adaptive weight schedules ─────────────────────────────────────────────────

# # def get_pinn_weight(epoch, warmup_epochs=30, w_start=0.01, w_end=0.1):
# #     if epoch >= warmup_epochs:
# #         return w_end
# #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# # def get_velocity_weight(epoch, warmup_epochs=20, w_start=0.5, w_end=1.5):
# #     if epoch >= warmup_epochs:
# #         return w_end
# #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# # def get_recurv_weight(epoch, warmup_epochs=10, w_start=0.3, w_end=1.0):
# #     if epoch >= warmup_epochs:
# #         return w_end
# #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# # def get_grad_clip(epoch, warmup_epochs=20, clip_start=2.0, clip_end=1.0):
# #     if epoch >= warmup_epochs:
# #         return clip_end
# #     return clip_start - (epoch / max(warmup_epochs-1, 1)) * (clip_start - clip_end)


# # def get_args():
# #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# #     p.add_argument("--dataset_root",    default="TCND_vn",      type=str)
# #     p.add_argument("--obs_len",         default=8,              type=int)
# #     p.add_argument("--pred_len",        default=12,             type=int)
# #     p.add_argument("--test_year",       default=None,           type=int)
# #     p.add_argument("--batch_size",      default=32,             type=int)
# #     p.add_argument("--num_epochs",      default=200,            type=int)
# #     p.add_argument("--g_learning_rate", default=2e-4,           type=float)
# #     p.add_argument("--weight_decay",    default=1e-4,           type=float)
# #     p.add_argument("--warmup_epochs",   default=3,              type=int)
# #     p.add_argument("--grad_clip",       default=2.0,            type=float)
# #     p.add_argument("--grad_accum",      default=2,              type=int)
# #     p.add_argument("--patience",        default=50,             type=int)
# #     p.add_argument("--min_epochs",      default=80,             type=int)
# #     p.add_argument("--n_train_ens",     default=6,              type=int)
# #     p.add_argument("--use_amp",         action="store_true")
# #     p.add_argument("--num_workers",     default=2,              type=int)
# #     p.add_argument("--sigma_min",       default=0.05,           type=float)
# #     p.add_argument("--ode_steps_train", default=20,             type=int)
# #     p.add_argument("--ode_steps_val",   default=30,             type=int)
# #     p.add_argument("--ode_steps_test",  default=50,             type=int)
# #     p.add_argument("--ode_steps",       default=None,           type=int)
# #     p.add_argument("--val_ensemble",    default=30,             type=int)
# #     p.add_argument("--fast_ensemble",   default=8,              type=int)
# #     p.add_argument("--fno_modes_h",     default=4,              type=int)
# #     p.add_argument("--fno_modes_t",     default=4,              type=int)
# #     p.add_argument("--fno_layers",      default=4,              type=int)
# #     p.add_argument("--fno_d_model",     default=32,             type=int)
# #     p.add_argument("--fno_spatial_down",default=32,             type=int)
# #     p.add_argument("--mamba_d_state",   default=16,             type=int)
# #     p.add_argument("--val_loss_freq",   default=2,              type=int)
# #     p.add_argument("--val_freq",        default=2,              type=int)
# #     p.add_argument("--full_eval_freq",  default=10,             type=int)
# #     p.add_argument("--val_subset_size", default=600,            type=int)
# #     p.add_argument("--output_dir",      default="runs/v19",     type=str)
# #     p.add_argument("--save_interval",   default=10,             type=int)
# #     p.add_argument("--metrics_csv",     default="metrics.csv",        type=str)
# #     p.add_argument("--predict_csv",     default="predictions.csv",    type=str)
# #     p.add_argument("--lstm_errors_npy",      default=None, type=str)
# #     p.add_argument("--diffusion_errors_npy", default=None, type=str)
# #     p.add_argument("--gpu_num",         default="0",            type=str)
# #     p.add_argument("--delim",           default=" ")
# #     p.add_argument("--skip",            default=1,              type=int)
# #     p.add_argument("--min_ped",         default=1,              type=int)
# #     p.add_argument("--threshold",       default=0.002,          type=float)
# #     p.add_argument("--other_modal",     default="gph")
# #     p.add_argument("--curriculum",      default=True,
# #                    type=lambda x: x.lower() != 'false')
# #     p.add_argument("--curriculum_start_len", default=4,         type=int)
# #     p.add_argument("--curriculum_end_epoch", default=40,        type=int)
# #     p.add_argument("--lon_flip_prob",   default=0.3,            type=float)
# #     p.add_argument("--pinn_warmup_epochs", default=30,          type=int)
# #     p.add_argument("--pinn_w_start",    default=0.01,           type=float)
# #     p.add_argument("--pinn_w_end",      default=0.1,            type=float)
# #     p.add_argument("--vel_warmup_epochs",  default=20,          type=int)
# #     p.add_argument("--vel_w_start",        default=0.5,         type=float)
# #     p.add_argument("--vel_w_end",          default=1.5,         type=float)
# #     p.add_argument("--recurv_warmup_epochs", default=10,        type=int)
# #     p.add_argument("--recurv_w_start",       default=0.3,       type=float)
# #     p.add_argument("--recurv_w_end",         default=1.0,       type=float)
# #     return p.parse_args()


# # def _resolve_ode_steps(args):
# #     if args.ode_steps is not None:
# #         return args.ode_steps, args.ode_steps, args.ode_steps
# #     return args.ode_steps_train, args.ode_steps_val, args.ode_steps_test


# # def move(batch, device):
# #     out = list(batch)
# #     for i, x in enumerate(out):
# #         if torch.is_tensor(x):
# #             out[i] = x.to(device)
# #         elif isinstance(x, dict):
# #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# #                       for k, v in x.items()}
# #     return out


# # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# #                            collate_fn, num_workers):
# #     n   = len(val_dataset)
# #     rng = random.Random(42)
# #     idx = rng.sample(range(n), min(subset_size, n))
# #     return DataLoader(Subset(val_dataset, idx),
# #                       batch_size=batch_size, shuffle=False,
# #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # def get_curriculum_len(epoch, args) -> int:
# #     if not args.curriculum:
# #         return args.pred_len
# #     if epoch >= args.curriculum_end_epoch:
# #         return args.pred_len
# #     frac = epoch / max(args.curriculum_end_epoch, 1)
# #     return int(args.curriculum_start_len
# #                + frac * (args.pred_len - args.curriculum_start_len))


# # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
# #     model.eval()
# #     acc = StepErrorAccumulator(pred_len)
# #     t0  = time.perf_counter()
# #     n   = 0
# #     with torch.no_grad():
# #         for batch in loader:
# #             bl = move(list(batch), device)
# #             pred, _, _ = model.sample(bl, num_ensemble=fast_ensemble,
# #                                       ddim_steps=ode_steps)
# #             acc.update(haversine_km_torch(denorm_torch(pred), denorm_torch(bl[1])))
# #             n += 1
# #     r = acc.compute()
# #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# #     return r


# # def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
# #                   metrics_csv, tag="", predict_csv=""):
# #     model.eval()
# #     cliper_step_errors = []
# #     ev = TCEvaluator(pred_len=pred_len, compute_dtw=True)
# #     obs_seqs_01 = []; gt_seqs_01 = []; pred_seqs_01 = []; ens_seqs_01 = []

# #     with torch.no_grad():
# #         for batch in loader:
# #             bl  = move(list(batch), device)
# #             gt  = bl[1];  obs = bl[0]
# #             pred_mean, _, all_trajs = model.sample(
# #                 bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
# #                 predict_csv=predict_csv if predict_csv else None)

# #             pd_np = denorm_torch(pred_mean).cpu().numpy()
# #             gd_np = denorm_torch(gt).cpu().numpy()
# #             od_np = denorm_torch(obs).cpu().numpy()
# #             ed_np = denorm_torch(all_trajs).cpu().numpy()

# #             B = pd_np.shape[1]
# #             for b in range(B):
# #                 ens_b = ed_np[:, :, b, :]
# #                 ev.update(pd_np[:, b, :], gd_np[:, b, :], pred_ens=ens_b)
# #                 obs_seqs_01.append(od_np[:, b, :])
# #                 gt_seqs_01.append(gd_np[:, b, :])
# #                 pred_seqs_01.append(pd_np[:, b, :])
# #                 ens_seqs_01.append(ens_b)
# #                 obs_b = od_np[:, b, :]

# #                 cliper_errors_b = np.zeros(pred_len)
# #                 for h in range(pred_len):
# #                     pred_cliper_01  = cliper_forecast(obs_b, h + 1)
# #                     pred_cliper_deg = denorm_deg_np(pred_cliper_01[np.newaxis, :])
# #                     gt_point        = gd_np[h, b, :][np.newaxis, :]
# #                     gt_deg          = denorm_deg_np(gt_point)
# #                     cliper_errors_b[h] = float(haversine_km_np(pred_cliper_deg, gt_deg)[0])

# #                 cliper_step_errors.append(cliper_errors_b)

# #     if cliper_step_errors:
# #         cliper_mat = np.stack(cliper_step_errors)
# #         cliper_ugde_dict = {h: float(cliper_mat[:, s].mean())
# #                             for h, s in HORIZON_STEPS.items()
# #                             if s < cliper_mat.shape[1]}
# #         ev.cliper_ugde = cliper_ugde_dict
# #         print(f"  [CLIPER UGDE] 72h={cliper_ugde_dict.get(72, float('nan')):.1f} km")

# #     dm = ev.compute(tag=tag)

# #     try:
# #         if LANDFALL_TARGETS and ens_seqs_01:
# #             bss_vals = []
# #             step_72  = HORIZON_STEPS.get(72, pred_len - 1)
# #             for tname, t_lon, t_lat in LANDFALL_TARGETS:
# #                 bv = brier_skill_score(
# #                     [e.transpose(1,0,2) if e.ndim==3 else e for e in ens_seqs_01],
# #                     gt_seqs_01, min(step_72, pred_len-1),
# #                     (t_lon, t_lat), LANDFALL_RADIUS_KM)
# #                 if not math.isnan(bv):
# #                     bss_vals.append(bv)
# #             if bss_vals:
# #                 dm.bss_mean = float(np.mean(bss_vals))
# #                 print(f"  [BSS] mean={dm.bss_mean:.4f}")
# #     except Exception as e:
# #         print(f"  ⚠  BSS failed: {e}")

# #     save_metrics_csv(dm, metrics_csv, tag=tag)
# #     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# # class BestModelSaver:
# #     def __init__(self, patience=50, ade_tol=5.0):
# #         self.patience      = patience
# #         self.ade_tol       = ade_tol
# #         self.best_ade      = float("inf")
# #         self.best_val_loss = float("inf")
# #         self.counter_ade   = 0
# #         self.counter_loss  = 0
# #         self.early_stop    = False

# #     def reset_counters(self, reason=""):
# #         self.counter_ade  = 0
# #         self.counter_loss = 0
# #         if reason:
# #             print(f"  [SAVER] Patience counters reset: {reason}")

# #     def update_val_loss(self, val_loss, model, out_dir, epoch, optimizer, tl):
# #         if val_loss < self.best_val_loss - 1e-4:
# #             self.best_val_loss = val_loss;  self.counter_loss = 0
# #             torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
# #                             optimizer_state=optimizer.state_dict(),
# #                             train_loss=tl, val_loss=val_loss,
# #                             model_version="v19-valloss"),
# #                        os.path.join(out_dir, "best_model_valloss.pth"))
# #         else:
# #             self.counter_loss += 1

# #     def update_ade(self, ade, model, out_dir, epoch, optimizer, tl, vl,
# #                    min_epochs=80):
# #         """
# #         FIX-V19-1: Được gọi MỖI EPOCH nên counter_ade = số epoch thực sự
# #         không cải thiện. patience=50 giờ đúng nghĩa 50 epoch liên tiếp.
# #         """
# #         if ade < self.best_ade - self.ade_tol:
# #             self.best_ade = ade;  self.counter_ade = 0
# #             torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
# #                             optimizer_state=optimizer.state_dict(),
# #                             train_loss=tl, val_loss=vl, val_ade_km=ade,
# #                             model_version="v19-FNO-Mamba-recurv"),
# #                        os.path.join(out_dir, "best_model.pth"))
# #             print(f"  ✅ Best ADE {ade:.1f} km  (epoch {epoch})")
# #         else:
# #             self.counter_ade += 1
# #             print(f"  No ADE improvement {self.counter_ade}/{self.patience}"
# #                   f"  (Δ={self.best_ade-ade:.1f} km < tol={self.ade_tol} km)"
# #                   f"  | Loss counter {self.counter_loss}/{self.patience}")

# #         if epoch >= min_epochs:
# #             if self.counter_ade >= self.patience and self.counter_loss >= self.patience:
# #                 self.early_stop = True
# #                 print(f"  ⛔ Early stop @ epoch {epoch}")
# #         else:
# #             if self.counter_ade >= self.patience and self.counter_loss >= self.patience:
# #                 print(f"  ⚠  Would stop but min_epochs={min_epochs} not reached. Continuing...")
# #                 self.counter_ade = 0;  self.counter_loss = 0


# # def _load_baseline_errors(path, name):
# #     if path is None:
# #         print(f"\n  ⚠  WARNING: --{name.lower().replace(' ','_')}_errors_npy not provided.")
# #         print(f"     Statistical comparison vs {name} will be SKIPPED.\n")
# #         return None
# #     if not os.path.exists(path):
# #         print(f"\n  ⚠  {path} not found — {name} skipped.\n")
# #         return None
# #     arr = np.load(path)
# #     print(f"  ✓  Loaded {name}: {arr.shape}")
# #     return arr


# # def main(args):
# #     if torch.cuda.is_available():
# #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     os.makedirs(args.output_dir, exist_ok=True)

# #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# #     predict_csv = os.path.join(args.output_dir, args.predict_csv)
# #     tables_dir  = os.path.join(args.output_dir, "tables")
# #     stat_dir    = os.path.join(tables_dir, "stat_tests")
# #     os.makedirs(tables_dir, exist_ok=True)
# #     os.makedirs(stat_dir,   exist_ok=True)

# #     ode_train, ode_val, ode_test = _resolve_ode_steps(args)

# #     print("=" * 68)
# #     print("  TC-FlowMatching v19  |  FNO3D + Mamba + OT-CFM + PINN")
# #     print("  v19 FIX: ADE evaluated EVERY epoch (counter = real epochs)")
# #     print("=" * 68)
# #     print(f"  device          : {device}")
# #     print(f"  sigma_min       : {args.sigma_min}")
# #     print(f"  ode_steps       : train={ode_train}  val={ode_val}  test={ode_test}")
# #     print(f"  val_ensemble    : {args.val_ensemble}")
# #     print(f"  patience        : {args.patience} epochs  (min_epochs={args.min_epochs})")
# #     print()

# #     train_dataset, train_loader = data_loader(
# #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# #     val_dataset, val_loader = data_loader(
# #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# #     val_subset_loader = make_val_subset_loader(
# #         val_dataset, args.val_subset_size, args.batch_size,
# #         seq_collate, args.num_workers)

# #     test_loader = None
# #     try:
# #         _, test_loader = data_loader(
# #             args, {"root": args.dataset_root, "type": "test"},
# #             test=True, test_year=None)
# #     except Exception as e:
# #         print(f"  Warning: test loader: {e}")

# #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# #     print(f"  val   : {len(val_dataset)} seq")
# #     if test_loader:
# #         print(f"  test  : {len(test_loader.dataset)} seq")

# #     model = TCFlowMatching(
# #         pred_len    = args.pred_len,
# #         obs_len     = args.obs_len,
# #         sigma_min   = args.sigma_min,
# #         n_train_ens = args.n_train_ens,
# #     ).to(device)

# #     if (args.fno_spatial_down != 32 or args.fno_modes_h != 4
# #             or args.fno_layers != 4 or args.fno_d_model != 32):
# #         from Model.FNO3D_encoder import FNO3DEncoder
# #         model.net.spatial_enc = FNO3DEncoder(
# #             in_channel=13, out_channel=1,
# #             d_model=args.fno_d_model, n_layers=args.fno_layers,
# #             modes_t=args.fno_modes_t, modes_h=args.fno_modes_h,
# #             modes_w=args.fno_modes_h, spatial_down=args.fno_spatial_down,
# #             dropout=0.05).to(device)

# #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #     print(f"  params  : {n_params:,}")

# #     try:
# #         model = torch.compile(model, mode="reduce-overhead")
# #         print("  torch.compile: enabled")
# #     except Exception:
# #         pass

# #     optimizer = optim.AdamW(model.parameters(),
# #                              lr=args.g_learning_rate,
# #                              weight_decay=args.weight_decay)
# #     steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
# #     total_steps     = steps_per_epoch * args.num_epochs
# #     warmup          = steps_per_epoch * args.warmup_epochs
# #     scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
# #     saver  = BestModelSaver(patience=args.patience, ade_tol=5.0)
# #     scaler = GradScaler('cuda', enabled=args.use_amp)

# #     print("=" * 68)
# #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch)")
# #     print("=" * 68)

# #     epoch_times   = []
# #     train_start   = time.perf_counter()
# #     last_val_loss = float("inf")
# #     _lr_ep30_done = False
# #     _lr_ep60_done = False
# #     _prev_ens     = 1

# #     import Model.losses as _losses_mod

# #     for epoch in range(args.num_epochs):
# #         # Progressive ensemble
# #         current_ens = 1 if epoch < 30 else (2 if epoch < 60 else args.n_train_ens)
# #         model.n_train_ens = current_ens
# #         effective_fast_ens = min(args.fast_ensemble, max(current_ens*2, args.fast_ensemble))

# #         # Reset patience counters when ensemble size increases
# #         if current_ens != _prev_ens:
# #             saver.reset_counters(
# #                 f"n_train_ens {_prev_ens}→{current_ens} at epoch {epoch}")
# #             _prev_ens = current_ens

# #         curr_len = get_curriculum_len(epoch, args)
# #         if hasattr(model, "set_curriculum_len"):
# #             model.set_curriculum_len(curr_len)

# #         epoch_weights = copy.copy(_BASE_WEIGHTS)
# #         epoch_weights["pinn"]     = get_pinn_weight(epoch, args.pinn_warmup_epochs,
# #                                                     args.pinn_w_start, args.pinn_w_end)
# #         epoch_weights["velocity"] = get_velocity_weight(epoch, args.vel_warmup_epochs,
# #                                                         args.vel_w_start, args.vel_w_end)
# #         epoch_weights["recurv"]   = get_recurv_weight(epoch, args.recurv_warmup_epochs,
# #                                                       args.recurv_w_start, args.recurv_w_end)
# #         _losses_mod.WEIGHTS.update(epoch_weights)
# #         if hasattr(model, 'weights'):
# #             model.weights = epoch_weights
# #         _losses_mod.WEIGHTS["pinn"]     = epoch_weights["pinn"]
# #         _losses_mod.WEIGHTS["velocity"] = epoch_weights["velocity"]
# #         _losses_mod.WEIGHTS["recurv"]   = epoch_weights["recurv"]

# #         current_clip = get_grad_clip(epoch, warmup_epochs=20,
# #                                      clip_start=args.grad_clip, clip_end=1.0)

# #         # LR restarts
# #         if epoch == 30 and not _lr_ep30_done:
# #             _lr_ep30_done = True
# #             warmup_steps = steps_per_epoch * 1
# #             scheduler = get_cosine_schedule_with_warmup(
# #                 optimizer, warmup_steps,
# #                 steps_per_epoch * (args.num_epochs - 30),
# #                 min_lr=5e-6)
# #             saver.reset_counters("LR warm restart at epoch 30")
# #             print(f"  ↺  Warm Restart LR at epoch 30")

# #         if epoch == 60 and not _lr_ep60_done:
# #             _lr_ep60_done = True
# #             warmup_steps = steps_per_epoch * 1
# #             scheduler = get_cosine_schedule_with_warmup(
# #                 optimizer, warmup_steps,
# #                 steps_per_epoch * (args.num_epochs - 60),
# #                 min_lr=1e-6)
# #             saver.reset_counters("LR warm restart at epoch 60")
# #             print(f"  ↺  Warm Restart LR at epoch 60")

# #         # ── Training loop ─────────────────────────────────────────────────────
# #         model.train()
# #         sum_loss = 0.0
# #         t0 = time.perf_counter()
# #         optimizer.zero_grad()
# #         recurv_ratio_buf = []

# #         for i, batch in enumerate(train_loader):
# #             bl = move(list(batch), device)

# #             if epoch == 0 and i == 0:
# #                 test_env = bl[13]
# #                 if test_env is not None and "gph500_mean" in test_env:
# #                     gph_val = test_env["gph500_mean"]
# #                     if torch.all(gph_val == 0):
# #                         print("\n" + "!"*60)
# #                         print("  ⚠️  GPH500 đang bị triệt tiêu về 0!")
# #                         print("!"*60 + "\n")
# #                     else:
# #                         print(f"  ✅ Data Check: GPH500 OK (Mean: {gph_val.mean().item():.4f})")

# #             with autocast(device_type='cuda', enabled=args.use_amp):
# #                 bd = model.get_loss_breakdown(bl)

# #             loss_to_backpass = bd["total"] / max(args.grad_accum, 1)
# #             scaler.scale(loss_to_backpass).backward()

# #             if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
# #                 scaler.unscale_(optimizer)
# #                 torch.nn.utils.clip_grad_norm_(model.parameters(), current_clip)
# #                 scaler.step(optimizer)
# #                 scaler.update()
# #                 scheduler.step()
# #                 optimizer.zero_grad()

# #             sum_loss += bd["total"].item()

# #             if "recurv_ratio" in bd:
# #                 recurv_ratio_buf.append(bd["recurv_ratio"])

# #             if i % 20 == 0:
# #                 lr  = optimizer.param_groups[0]["lr"]
# #                 rr  = bd.get("recurv_ratio", 0.0)
# #                 elapsed = time.perf_counter() - t0
# #                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# #                       f"  loss={bd['total'].item():.3f}"
# #                       f"  fm={bd.get('fm',0):.2f}"
# #                       f"  vel={bd.get('velocity',0):.6f}"
# #                       f"  pinn={bd.get('pinn', 0):.6f}"
# #                       f"  recurv={bd.get('recurv',0):.3f}"
# #                       f"  rr={rr:.2f}"
# #                       f"  pinn_w={epoch_weights['pinn']:.3f}"
# #                       f"  clip={current_clip:.1f}"
# #                       f"  ens={current_ens}  len={curr_len}"
# #                       f"  lr={lr:.2e}  t={elapsed:.0f}s")

# #         ep_s  = time.perf_counter() - t0
# #         epoch_times.append(ep_s)
# #         avg_t = sum_loss / len(train_loader)
# #         mean_rr = float(np.mean(recurv_ratio_buf)) if recurv_ratio_buf else 0.0

# #         # ── Val loss (mỗi val_freq epoch) ─────────────────────────────────────
# #         if epoch % args.val_freq == 0:
# #             model.eval()
# #             val_loss = 0.0
# #             t_val = time.perf_counter()
# #             with torch.no_grad():
# #                 for batch in val_loader:
# #                     bl_v = move(list(batch), device)
# #                     with autocast(device_type='cuda', enabled=args.use_amp):
# #                         val_loss += model.get_loss(bl_v).item()
# #             last_val_loss = val_loss / len(val_loader)
# #             t_val_s = time.perf_counter() - t_val
# #             saver.update_val_loss(last_val_loss, model, args.output_dir,
# #                                   epoch, optimizer, avg_t)
# #             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
# #                   f"  rr={mean_rr:.2f}"
# #                   f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
# #                   f"  ens={current_ens}  len={curr_len}"
# #                   f"  recurv_w={epoch_weights['recurv']:.2f}")
# #         else:
# #             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
# #                   f"  val={last_val_loss:.3f}(cached)"
# #                   f"  rr={mean_rr:.2f}  t={ep_s:.0f}s")

# #         # ── ADE evaluation MỖI EPOCH (FIX-V19-1) ──────────────────────────────
# #         # Tách độc lập khỏi val_freq để counter_ade luôn = số epoch thực sự
# #         # không cải thiện. patience=50 → đúng nghĩa 50 epoch liên tiếp.
# #         t_ade = time.perf_counter()
# #         m = evaluate_fast(model, val_subset_loader, device,
# #                           ode_train, args.pred_len, effective_fast_ens)
# #         t_ade_s = time.perf_counter() - t_ade
# #         print(f"  [ADE ep{epoch} {t_ade_s:.0f}s]"
# #               f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km"
# #               f"  12h={m.get('12h',0):.0f}  24h={m.get('24h',0):.0f}"
# #               f"  72h={m.get('72h',0):.0f} km"
# #               f"  (ens={effective_fast_ens}, steps={ode_train})"
# #               f"  counter={saver.counter_ade}/{args.patience}")
# #         saver.update_ade(m["ADE"], model, args.output_dir, epoch,
# #                          optimizer, avg_t, last_val_loss,
# #                          min_epochs=args.min_epochs)

# #         # ── Full eval (mỗi full_eval_freq epoch) ──────────────────────────────
# #         if epoch % args.full_eval_freq == 0 and epoch > 0:
# #             print(f"  [Full eval epoch {epoch}, ode_steps={ode_val}]")
# #             try:
# #                 dm, _, _, _ = evaluate_full(
# #                     model, val_loader, device,
# #                     ode_val, args.pred_len, args.val_ensemble,
# #                     metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}")
# #                 print(dm.summary())
# #             except Exception as e:
# #                 print(f"  ⚠  full_eval failed at epoch {epoch}: {e}")
# #                 import traceback; traceback.print_exc()

# #         if (epoch+1) % args.save_interval == 0:
# #             torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
# #                        os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))

# #         if saver.early_stop:
# #             print(f"  Early stopping @ epoch {epoch}")
# #             break

# #         if epoch % 5 == 4:
# #             avg_ep    = sum(epoch_times) / len(epoch_times)
# #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# #             elapsed_h = (time.perf_counter() - train_start) / 3600
# #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
# #                   f"  (avg {avg_ep:.0f}s/epoch)")

# #     _losses_mod.WEIGHTS["pinn"]     = args.pinn_w_end
# #     _losses_mod.WEIGHTS["velocity"] = args.vel_w_end
# #     _losses_mod.WEIGHTS["recurv"]   = args.recurv_w_end

# #     total_train_h = (time.perf_counter() - train_start) / 3600

# #     # ── Final test eval ───────────────────────────────────────────────────────
# #     print(f"\n{'='*68}  FINAL TEST (ode_steps={ode_test})")
# #     all_results = []

# #     if test_loader:
# #         best_path = os.path.join(args.output_dir, "best_model.pth")
# #         if not os.path.exists(best_path):
# #             best_path = os.path.join(args.output_dir, "best_model_valloss.pth")
# #         if os.path.exists(best_path):
# #             ck = torch.load(best_path, map_location=device)
# #             try:
# #                 model.load_state_dict(ck["model_state_dict"])
# #             except Exception:
# #                 model.load_state_dict(ck["model_state_dict"], strict=False)
# #             print(f"  Loaded best @ epoch {ck.get('epoch','?')}"
# #                   f"  ADE={ck.get('val_ade_km','?')}")

# #         final_ens = max(args.val_ensemble, 50)
# #         dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
# #             model, test_loader, device,
# #             ode_test, args.pred_len, final_ens,
# #             metrics_csv=metrics_csv, tag="test_final",
# #             predict_csv=predict_csv)
# #         print(dm_test.summary())

# #         all_results.append(ModelResult(
# #             model_name   = "FM+PINN-v19",
# #             split        = "test",
# #             ADE          = dm_test.ade,
# #             FDE          = dm_test.fde,
# #             ADE_str      = dm_test.ade_str,
# #             ADE_rec      = dm_test.ade_rec,
# #             delta_rec    = dm_test.pr,
# #             CRPS_mean    = dm_test.crps_mean,
# #             CRPS_72h     = dm_test.crps_72h,
# #             SSR          = dm_test.ssr_mean,
# #             TSS_72h      = dm_test.tss_72h,
# #             OYR          = dm_test.oyr_mean,
# #             DTW          = dm_test.dtw_mean,
# #             ATE_abs      = dm_test.ate_abs_mean,
# #             CTE_abs      = dm_test.cte_abs_mean,
# #             n_total      = dm_test.n_total,
# #             n_recurv     = dm_test.n_rec,
# #             train_time_h = total_train_h,
# #             params_M     = sum(p.numel() for p in model.parameters()) / 1e6,
# #         ))

# #         _, cliper_errs = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
# #         persist_errs   = persistence_errors(obs_seqs, gt_seqs, args.pred_len)
# #         fmpinn_per_seq = np.array([seq_ade_km(np.array(pp), np.array(g))
# #                                     for pp, g in zip(pred_seqs, gt_seqs)])

# #         np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
# #         np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
# #         np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))

# #         lstm_per_seq      = _load_baseline_errors(args.lstm_errors_npy,      "LSTM")
# #         diffusion_per_seq = _load_baseline_errors(args.diffusion_errors_npy, "Diffusion")
# #         if lstm_per_seq is not None:
# #             np.save(os.path.join(stat_dir, "lstm.npy"), lstm_per_seq)
# #         if diffusion_per_seq is not None:
# #             np.save(os.path.join(stat_dir, "diffusion.npy"), diffusion_per_seq)

# #         _dummy = np.array([float("nan")])
# #         run_all_tests(
# #             fmpinn_ade    = fmpinn_per_seq,
# #             cliper_ade    = cliper_errs.mean(1),
# #             lstm_ade      = lstm_per_seq if lstm_per_seq is not None else _dummy,
# #             diffusion_ade = diffusion_per_seq if diffusion_per_seq is not None else _dummy,
# #             persist_ade   = persist_errs.mean(1),
# #             out_dir       = stat_dir)

# #         all_results += [
# #             ModelResult("CLIPER", "test",
# #                         ADE=float(cliper_errs.mean()),
# #                         FDE=float(cliper_errs[:, -1].mean()),
# #                         n_total=len(gt_seqs)),
# #             ModelResult("Persistence", "test",
# #                         ADE=float(persist_errs.mean()),
# #                         FDE=float(persist_errs[:, -1].mean()),
# #                         n_total=len(gt_seqs)),
# #         ]

# #         stat_rows = [
# #             paired_tests(fmpinn_per_seq, cliper_errs.mean(1),  "FM+PINN vs CLIPER",      5),
# #             paired_tests(fmpinn_per_seq, persist_errs.mean(1), "FM+PINN vs Persistence", 5),
# #         ]
# #         if lstm_per_seq is not None:
# #             stat_rows.append(paired_tests(fmpinn_per_seq, lstm_per_seq, "FM+PINN vs LSTM", 5))
# #         if diffusion_per_seq is not None:
# #             stat_rows.append(paired_tests(fmpinn_per_seq, diffusion_per_seq, "FM+PINN vs Diffusion", 5))

# #         compute_rows = DEFAULT_COMPUTE
# #         try:
# #             sb = next(iter(test_loader))
# #             sb = move(list(sb), device)
# #             from utils.evaluation_tables import profile_model_components
# #             compute_rows = profile_model_components(model, sb, device)
# #         except Exception as e:
# #             print(f"  Profiling skipped: {e}")

# #         export_all_tables(
# #             results=all_results, ablation_rows=DEFAULT_ABLATION,
# #             stat_rows=stat_rows, pinn_sens_rows=DEFAULT_PINN_SENSITIVITY,
# #             compute_rows=compute_rows, out_dir=tables_dir)

# #         with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
# #             fh.write(dm_test.summary())
# #             fh.write(f"\n\nmodel_version    : FM+PINN v19\n")
# #             fh.write(f"sigma_min        : {args.sigma_min}\n")
# #             fh.write(f"ode_steps_test   : {ode_test}\n")
# #             fh.write(f"eval_ensemble    : {final_ens}\n")
# #             fh.write(f"train_time_h     : {total_train_h:.2f}\n")
# #             fh.write(f"n_params_M       : "
# #                      f"{sum(p.numel() for p in model.parameters())/1e6:.2f}\n")

# #     avg_ep = sum(epoch_times)/len(epoch_times) if epoch_times else 0
# #     print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
# #     print(f"  Best val loss  : {saver.best_val_loss:.4f}")
# #     print(f"  Avg epoch time : {avg_ep:.0f}s")
# #     print(f"  Total training : {total_train_h:.2f}h")
# #     print(f"  Tables dir     : {tables_dir}")
# #     print("=" * 68)


# # if __name__ == "__main__":
# #     args = get_args()
# #     np.random.seed(42);  torch.manual_seed(42)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(42)
# #     main(args)

# """
# Model/flow_matching_model.py  ── v15
# ==========================================
# OT-CFM Flow Matching + PINN-BVE for TC trajectory prediction.

# FIXES vs v14:

#   FIX-M5  [ENSEMBLE COLLAPSE FIX] sample() dùng model.eval() nên dropout
#            TẮT hoàn toàn. raw_ctx được tính 1 lần và share cho tất cả
#            ensemble members → mọi sample nhận cùng ctx → velocity field
#            giống hệt nhau → 50 trajectories converge về 1 điểm.
#            Root cause: diversity chỉ đến từ initial noise randn * sigma_min
#            (0.05) — quá nhỏ để tạo spread có ý nghĩa.

#            Fix 3 tầng:
#            (a) ctx_noise_scale=0.05: Inject Gaussian noise vào raw_ctx
#                trước khi project → mỗi member có ctx khác nhau
#            (b) initial_sigma=max(sigma_min, 0.3): Tăng spread của noise
#                ban đầu trong sample() — sigma_min=0.05 quá nhỏ
#            (c) Bỏ clamp trong ODE loop: clamp_(-10,10) trong mỗi step
#                làm triệt tiêu divergence tự nhiên giữa các trajectories.
#                Chỉ clamp 1 lần sau khi ODE hoàn thành.

#   FIX-M6  [TRAINING STABILITY] Thêm args --ctx_noise_scale và
#            --initial_sample_sigma để dễ tune mà không cần sửa code.
#            Default: ctx_noise_scale=0.05, initial_sample_sigma=0.3

# Kept from v14:
#   FIX-M1  _write_predict_csv denorm formula corrected
#   FIX-M2  clarified inference stochasticity source
#   FIX-M3  lon_flip augmentation consistency confirmed
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
# from Model.losses import compute_total_loss, WEIGHTS


# # ══════════════════════════════════════════════════════════════════════════════
# #  VelocityField  (FlowMatching denoiser)  ── v15
# # ══════════════════════════════════════════════════════════════════════════════

# class VelocityField(nn.Module):
#     """
#     OT-CFM velocity field  v_θ(x_t, t, context).

#     Context assembly:
#       h_t       [B, 128]  ← DataEncoder1D w/ Mamba
#       e_Env     [B,  64]  ← Env-T-Net
#       f_spatial [B,  16]  ← FNO decoder pooled
#       ─────────────────
#       total     [B, 208]  → ctx_fc → [B, ctx_dim=256]

#     FIX-M5: ctx_drop vẫn giữ nguyên nhưng diversity tại inference
#     giờ đến từ ctx_noise_scale injection thay vì dropout.
#     """

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

#         # ── FNO3D encoder ────────────────────────────────────────────────
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

#         # ── DataEncoder1D with Mamba ─────────────────────────────────────
#         self.enc_1d = DataEncoder1D(
#             in_1d       = 4,
#             feat_3d_dim = 128,
#             mlp_h       = 64,
#             lstm_hidden = 128,
#             lstm_layers = 3,
#             dropout     = 0.1,
#             d_state     = 16,
#         )

#         # ── Env-T-Net ────────────────────────────────────────────────────
#         self.env_enc = Env_net(obs_len=obs_len, d_model=64)

#         # ── Context fusion: 128 + 64 + 16 = 208 → ctx_dim ───────────────
#         self.ctx_fc1  = nn.Linear(128 + 64 + 16, 512)
#         self.ctx_ln   = nn.LayerNorm(512)
#         self.ctx_drop = nn.Dropout(0.15)
#         self.ctx_fc2  = nn.Linear(512, ctx_dim)

#         # ── Time embedding ────────────────────────────────────────────────
#         self.time_fc1 = nn.Linear(256, 512)
#         self.time_fc2 = nn.Linear(512, 256)

#         # ── Trajectory Transformer decoder ──────────────────────────────
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
#         """
#         Returns pre-dropout context [B, 512].
#         Dropout applied independently per ensemble sample via _apply_ctx_head.
#         """
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
#                 e_3d_s.permute(0, 2, 1),
#                 size=T_obs, mode="linear", align_corners=False,
#             ).permute(0, 2, 1)

#         f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))
#         f_spatial     = self.decoder_proj(f_spatial_raw)

#         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
#         h_t    = self.enc_1d(obs_in, e_3d_s)

#         e_env, _, _ = self.env_enc(env_data, image_obs)

#         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
#         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))
#         return raw  # [B, 512] — pre-dropout, pre-projection

#     def _apply_ctx_head(self, raw: torch.Tensor,
#                         noise_scale: float = 0.0) -> torch.Tensor:
#         """
#         Apply optional noise injection + dropout + final projection.

#         FIX-M5: noise_scale > 0 injects Gaussian noise vào raw_ctx trước
#         khi project. Mỗi ensemble member gọi hàm này với noise khác nhau
#         → ctx diversity ngay cả khi dropout=off (eval mode).
#         """
#         if noise_scale > 0.0:
#             raw = raw + torch.randn_like(raw) * noise_scale
#         return self.ctx_fc2(self.ctx_drop(raw))  # [B, ctx_dim]

#     def forward(self, x_t, t, batch_list):
#         raw = self._context(batch_list)
#         ctx = self._apply_ctx_head(raw, noise_scale=0.0)  # train: dropout handles diversity
#         return self._decode(x_t, t, ctx)

#     def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale: float = 0.0):
#         """
#         Forward pass reusing pre-computed raw context (pre-dropout).

#         FIX-M5: noise_scale được pass vào _apply_ctx_head để tạo ctx
#         diversity tại inference thay vì dựa vào dropout (bị tắt ở eval).
#         """
#         ctx = self._apply_ctx_head(raw_ctx, noise_scale=noise_scale)
#         return self._decode(x_t, t, ctx)

#     def _decode(self, x_t, t, ctx):
#         t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
#         t_emb = self.time_fc2(t_emb)

#         x_emb  = self.traj_embed(x_t) + self.pos_enc[:, :x_t.size(1), :] + t_emb.unsqueeze(1)
#         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

#         out = self.transformer(x_emb, memory)
#         return self.out_fc2(F.gelu(self.out_fc1(out)))


# # ══════════════════════════════════════════════════════════════════════════════
# #  TCFlowMatching  ── v15
# # ══════════════════════════════════════════════════════════════════════════════

# class TCFlowMatching(nn.Module):
#     """
#     TC trajectory prediction via OT-CFM + PINN-BVE.

#     v15 changes:
#       - FIX-M5: Ensemble collapse fix — ctx noise injection + larger initial sigma
#       - FIX-M6: ctx_noise_scale và initial_sample_sigma configurable
#     """

#     def __init__(
#         self,
#         pred_len:             int   = 12,
#         obs_len:              int   = 8,
#         sigma_min:            float = 0.02,
#         n_train_ens:          int   = 4,
#         unet_in_ch:           int   = 13,
#         ctx_noise_scale:      float = 0.05,   # FIX-M5: ctx diversity at inference
#         initial_sample_sigma: float = 0.3,    # FIX-M5: larger initial noise spread
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
#         self.active_pred_len = max(1, min(active_len, self.pred_len))

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
#         for idx in [7, 8, 9, 10]:
#             t = aug[idx]
#             if torch.is_tensor(t):
#                 aug[idx] = t.clone()
#         return aug

#     def get_loss(self, batch_list: List) -> torch.Tensor:
#         return self.get_loss_breakdown(batch_list)["total"]

#     def get_loss_breakdown(self, batch_list: List) -> Dict:
#         batch_list = self._lon_flip_aug(batch_list, p=0.3)

#         traj_gt = batch_list[1]
#         Me_gt   = batch_list[8]
#         obs_t   = batch_list[0]
#         obs_Me  = batch_list[7]

#         apl = self.active_pred_len
#         if apl < traj_gt.shape[0]:
#             traj_gt = traj_gt[:apl]
#             Me_gt   = Me_gt[:apl]

#         lp, lm = obs_t[-1], obs_Me[-1]
#         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

#         # Training: dropout is ON → ctx_drop provides diversity naturally
#         # noise_scale=0.0 during training (dropout handles stochasticity)
#         raw_ctx     = self.net._context(batch_list)
#         intensity_w = self._intensity_weights(obs_Me)

#         x_t, t, te, denom, _ = self._cfm_noisy(x1)
#         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

#         samples: List[torch.Tensor] = []
#         for _ in range(self.n_train_ens):
#             xt_s, ts, _, dens, _ = self._cfm_noisy(x1)
#             pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
#             x1_s  = xt_s + dens * pv_s
#             pa_s, _ = self._to_abs(x1_s, lp, lm)
#             samples.append(pa_s)
#         pred_samples = torch.stack(samples)  # [S, T, B, 2]

#         x1_pred = x_t + (1.0 - te) * pred_vel
#         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

#         breakdown = compute_total_loss(
#             pred_abs     = pred_abs,
#             gt           = traj_gt,
#             ref          = lp,
#             batch_list   = batch_list,
#             pred_samples = pred_samples,
#             weights      = WEIGHTS,
#             intensity_w  = intensity_w,
#         )
#         return breakdown

#     @torch.no_grad()
#     def sample(
#         self,
#         batch_list:   List,
#         num_ensemble: int = 50,
#         ddim_steps:   int = 20,
#         predict_csv:  Optional[str] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         FIX-M5: Ensemble collapse fix.

#         Vấn đề cũ:
#           - model.eval() tắt dropout → raw_ctx deterministic với noise_scale=0
#           - initial noise = randn * sigma_min = 0.05 → quá nhỏ
#           - clamp_(-10,10) trong mỗi ODE step → triệt tiêu divergence

#         Fix:
#           (a) Tính raw_ctx 1 lần (tiết kiệm compute FNO/Mamba/Env)
#           (b) Inject ctx_noise_scale noise vào raw_ctx PER MEMBER trước khi project
#           (c) Dùng initial_sample_sigma (default 0.3) thay vì sigma_min (0.05)
#           (d) Bỏ clamp trong ODE loop, chỉ clamp sau khi ODE xong
#         """
#         lp  = batch_list[0][-1]
#         lm  = batch_list[7][-1]
#         B, device = lp.shape[0], lp.device
#         dt  = 1.0 / ddim_steps

#         traj_s: List[torch.Tensor] = []
#         me_s:   List[torch.Tensor] = []

#         # Tính raw_ctx 1 lần (heavy encoders: FNO3D, Mamba, Env-T)
#         # Diversity sẽ đến từ (a) ctx noise injection và (b) initial noise
#         raw_ctx = self.net._context(batch_list)

#         for k in range(num_ensemble):
#             # FIX-M5(c): Dùng initial_sample_sigma >> sigma_min để có spread ban đầu
#             x_t = torch.randn(B, self.pred_len, 4, device=device) * self.initial_sample_sigma

#             # FIX-M5(b): Mỗi member nhận ctx có noise riêng → velocity field khác nhau
#             # ctx_noise_scale=0.05 là small perturbation, không làm mất semantic context
#             for step in range(ddim_steps):
#                 t_b = torch.full((B,), step * dt, device=device)
#                 # noise_scale chỉ inject ở step đầu để tránh accumulate quá nhiều noise
#                 ns = self.ctx_noise_scale if step == 0 else 0.0
#                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)

#                 # FIX-M5(d): KHÔNG clamp trong ODE loop — giữ natural divergence
#                 x_t = x_t + dt * vel

#             # Chỉ clamp sau khi ODE hoàn thành để đảm bảo physical validity
#             x_t[:, :, :2].clamp_(-5.0, 5.0)   # lon/lat relative displacement
#             x_t[:, :, 2:].clamp_(-3.0, 3.0)   # pressure/wind relative change

#             tr, me = self._to_abs(x_t, lp, lm)
#             traj_s.append(tr)
#             me_s.append(me)

#         all_trajs = torch.stack(traj_s)
#         all_me    = torch.stack(me_s)
#         traj_mean = all_trajs.mean(0)
#         me_mean   = all_me.mean(0)

#         if predict_csv is not None:
#             self._write_predict_csv(predict_csv, traj_mean, all_trajs)

#         return traj_mean, me_mean, all_trajs

#     @staticmethod
#     def _write_predict_csv(csv_path, traj_mean, all_trajs):
#         """
#         FIX-M1: Correct denorm formula.
#         lon_deg = (lon_norm * 50 + 1800) / 10
#         lat_deg = (lat_norm * 50) / 10
#         """
#         import numpy as np
#         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
#         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
#         T, B, _ = traj_mean.shape
#         S       = all_trajs.shape[0]

#         mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
#         mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
#         all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
#         all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()

#         fields = ["timestamp", "batch_idx", "step_idx", "lead_h",
#                   "lon_mean_deg", "lat_mean_deg",
#                   "lon_std_deg", "lat_std_deg", "ens_spread_km"]
#         write_hdr = not os.path.exists(csv_path)
#         with open(csv_path, "a", newline="") as fh:
#             w = csv.DictWriter(fh, fieldnames=fields)
#             if write_hdr:
#                 w.writeheader()
#             for b in range(B):
#                 for k in range(T):
#                     dlat  = all_lat[:, k, b] - mean_lat[k, b]
#                     dlon  = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
#                         math.radians(mean_lat[k, b]))
#                     spread = float(np.sqrt((dlat ** 2 + dlon ** 2).mean()) * 111.0)
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

    def _decode(self, x_t, t, ctx):
        t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
        t_emb = self.time_fc2(t_emb)

        x_emb  = self.traj_embed(x_t) + self.pos_enc[:, :x_t.size(1), :] + t_emb.unsqueeze(1)
        memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

        out = self.transformer(x_emb, memory)
        return self.out_fc2(F.gelu(self.out_fc1(out)))


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
        return breakdown

    @torch.no_grad()
    def sample(
        self,
        batch_list:   List,
        num_ensemble: int = 50,
        ddim_steps:   int = 20,
        predict_csv:  Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        FIX-M9/M10: Reduced ctx_noise_scale and initial_sample_sigma
        để SSR converge về 1.
        """
        lp  = batch_list[0][-1]
        lm  = batch_list[7][-1]
        B, device = lp.shape[0], lp.device
        dt  = 1.0 / ddim_steps

        traj_s: List[torch.Tensor] = []
        me_s:   List[torch.Tensor] = []

        raw_ctx = self.net._context(batch_list)

        for k in range(num_ensemble):
            x_t = torch.randn(B, self.pred_len, 4, device=device) * self.initial_sample_sigma

            for step in range(ddim_steps):
                t_b = torch.full((B,), step * dt, device=device)
                ns = self.ctx_noise_scale if step == 0 else 0.0
                vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
                x_t = x_t + dt * vel

            x_t[:, :, :2].clamp_(-5.0, 5.0)
            x_t[:, :, 2:].clamp_(-3.0, 3.0)

            tr, me = self._to_abs(x_t, lp, lm)
            traj_s.append(tr)
            me_s.append(me)

        all_trajs = torch.stack(traj_s)
        all_me    = torch.stack(me_s)
        traj_mean = all_trajs.mean(0)
        me_mean   = all_me.mean(0)

        if predict_csv is not None:
            self._write_predict_csv(predict_csv, traj_mean, all_trajs)

        return traj_mean, me_mean, all_trajs

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
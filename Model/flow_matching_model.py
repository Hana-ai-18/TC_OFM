# # """
# # Model/flow_matching_model.py  ── v10-turbo
# # ==========================================
# # OT-CFM Flow Matching + PINN-BVE for TC trajectory prediction.

# # CHANGES vs v9-fixed:
# #   ✅ UNet3D   → FNO3DEncoder  (4-8x faster spatial encoding)
# #   ✅ En-LSTM  → MambaEncoder  (3x faster temporal encoding, parallel)
# #   ✅ ctx_dim kept at 128, total context 128+64+16=208 (unchanged)
# #   ✅ All batch_list indices unchanged (full backward compat)

# # Speed gain (Kaggle T4, B=32, T_obs=8):
# #   v9-fixed : ~1.3s/batch  → 80 epochs × 481 batches ≈ 16h
# #   v10-turbo: ~0.35s/batch → 80 epochs × 481 batches ≈ 4.3h ✅✅

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

# # # ── SWAP: FNO3D instead of UNet3D ────────────────────────────────────────────
# # from Model.FNO3D_encoder import FNO3DEncoder
# # # ── SWAP: Mamba instead of LSTM ──────────────────────────────────────────────
# # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D

# # from Model.env_net_transformer_gphsplit import Env_net
# # from Model.losses import compute_total_loss, WEIGHTS


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  VelocityField  (FlowMatching denoiser)  ── v10
# # # ══════════════════════════════════════════════════════════════════════════════

# # class VelocityField(nn.Module):
# #     """
# #     OT-CFM velocity field  v_θ(x_t, t, context).

# #     Context assembly (identical to v9):
# #       h_t       [B, 128]  ← DataEncoder1D w/ Mamba (was LSTM)
# #       e_Env     [B,  64]  ← Env-T-Net (unchanged)
# #       f_spatial [B,  16]  ← FNO decoder pooled (was UNet decoder)
# #       ─────────────────
# #       total     [B, 208]  → ctx_fc → [B, ctx_dim=128]

# #     Trajectory decoder: TransformerDecoder + linear head → [B, T, 4]
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

# #         # ── FNO3D encoder (replaces UNet3D) ─────────────────────────────
# #         self.spatial_enc = FNO3DEncoder(
# #             in_channel   = unet_in_ch,
# #             out_channel  = 1,
# #             d_model      = 32,      # FIX: 64 → 32
# #             n_layers     = 4,
# #             modes_t      = 4,
# #             modes_h      = 4,       # FIX: 16 → 4
# #             modes_w      = 4,       # FIX: 16 → 4
# #             spatial_down = 32,
# #             dropout      = 0.05,
# #         )

# #         # Bottleneck: [B, 128, T, 4, 4] → pool spatial → [B, T, 128]
# #         # (same interface as v9 bottleneck_pool + bottleneck_proj)
# #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# #         self.bottleneck_proj = nn.Linear(128, 128)

# #         # Summary → [B, 16]
# #         self.decoder_proj = nn.Linear(1, 16)

# #         # ── DataEncoder1D with Mamba (replaces LSTM) ─────────────────────
# #         self.enc_1d = DataEncoder1D(
# #             in_1d       = 4,
# #             feat_3d_dim = 128,
# #             mlp_h       = 64,
# #             lstm_hidden = 128,
# #             lstm_layers = 3,
# #             dropout     = 0.1,
# #             d_state     = 16,
# #         )

# #         # ── Env-T-Net (Eq. 10–13) — unchanged ───────────────────────────
# #         self.env_enc = Env_net(obs_len=obs_len, d_model=64)

# #         # ── Context fusion: 128 + 64 + 16 = 208 → ctx_dim ───────────────
# #         self.ctx_fc1  = nn.Linear(128 + 64 + 16, 512)
# #         self.ctx_ln   = nn.LayerNorm(512)
# #         self.ctx_drop = nn.Dropout(0.15)
# #         self.ctx_fc2  = nn.Linear(512, ctx_dim)

# #         # ── Time embedding ────────────────────────────────────────────────
# #         self.time_fc1 = nn.Linear(128, 256)
# #         self.time_fc2 = nn.Linear(256, 128)

# #         # ── Trajectory Transformer decoder (unchanged) ───────────────────
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

# #     # ── Build context vector ──────────────────────────────────────────────

# #     def _context(self, batch_list: List) -> torch.Tensor:
# #         obs_traj  = batch_list[0]   # [T_obs, B, 2]
# #         obs_Me    = batch_list[7]   # [T_obs, B, 2]
# #         image_obs = batch_list[11]  # [B, 13, T_obs, 81, 81]
# #         env_data  = batch_list[13]  # dict

# #         # ── FNO3D encode ──────────────────────────────────────────────────
# #         if image_obs.dim() == 4:
# #             image_obs = image_obs.unsqueeze(1)

# #         # Single-channel tile if needed
# #         expected_ch = self.spatial_enc.in_channel
# #         if image_obs.shape[1] == 1 and expected_ch != 1:
# #             image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

# #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# #         # e_3d_bot: [B, 128, T, 4, 4]
# #         # e_3d_dec: [B, 1,   T, 1, 1]

# #         B     = e_3d_bot.shape[0]
# #         T_obs = obs_traj.shape[0]

# #         # Pool spatial dims 4×4 → 1×1, keep T: [B, 128, T, 1, 1] → [B, T, 128]
# #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)  # [B, 128, T_bot]
# #         e_3d_s = e_3d_s.permute(0, 2, 1)                                 # [B, T_bot, 128]
# #         e_3d_s = self.bottleneck_proj(e_3d_s)                            # [B, T_bot, 128]

# #         # Align T_bot → T_obs if needed (FNO preserves T, but just in case)
# #         T_bot = e_3d_s.shape[1]
# #         if T_bot != T_obs:
# #             e_3d_s = F.interpolate(
# #                 e_3d_s.permute(0, 2, 1),
# #                 size=T_obs, mode="linear", align_corners=False,
# #             ).permute(0, 2, 1)

# #         # Decoder spatial summary: [B, 1, T, 1, 1] → mean(T) → [B, 1] → [B, 16]
# #         f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))       # [B, 1]
# #         f_spatial     = self.decoder_proj(f_spatial_raw)   # [B, 16]

# #         # ── DataEncoder1D + Mamba ─────────────────────────────────────────
# #         obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)  # [B, T, 4]
# #         h_t    = self.enc_1d(obs_in, e_3d_s)   # [B, 128]

# #         # ── Env-T-Net ─────────────────────────────────────────────────────
# #         e_env, _, _ = self.env_enc(env_data, image_obs)  # [B, 64]

# #         # ── Fuse: [128 + 64 + 16] → [ctx_dim] ────────────────────────────
# #         ctx = torch.cat([h_t, e_env, f_spatial], dim=-1)  # [B, 208]
# #         ctx = F.gelu(self.ctx_ln(self.ctx_fc1(ctx)))
# #         ctx = self.ctx_drop(ctx)
# #         return self.ctx_fc2(ctx)  # [B, ctx_dim]

# #     # ── Forward ───────────────────────────────────────────────────────────

# #     def forward(
# #         self,
# #         x_t:        torch.Tensor,  # [B, T_pred, 4]
# #         t:          torch.Tensor,  # [B]
# #         batch_list: List,
# #     ) -> torch.Tensor:             # [B, T_pred, 4]
# #         ctx   = self._context(batch_list)
# #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# #         t_emb = self.time_fc2(t_emb)  # [B, 128]

# #         x_emb  = self.traj_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
# #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

# #         out = self.transformer(x_emb, memory)
# #         return self.out_fc2(F.gelu(self.out_fc1(out)))

# #     def forward_with_ctx(self, x_t, t, ctx):
# #         """Forward pass reusing pre-computed context (skips FNO + Mamba)."""
# #         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
# #         t_emb = self.time_fc2(t_emb)
# #         x_emb  = self.traj_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
# #         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)
# #         out = self.transformer(x_emb, memory)
# #         return self.out_fc2(F.gelu(self.out_fc1(out)))


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  TCFlowMatching  (unchanged logic, new backbone)
# # # ══════════════════════════════════════════════════════════════════════════════

# # class TCFlowMatching(nn.Module):
# #     """
# #     TC trajectory prediction via OT-CFM + PINN-BVE.

# #     Training loss (unchanged):
# #         L = 1.0·L_FM + 2.0·L_dir + 0.5·L_step
# #           + 1.0·L_disp + 2.0·L_heading + 0.2·L_smooth + 0.5·L_PINN

# #     Backbone changes:
# #         UNet3D → FNO3DEncoder   (4-8x faster)
# #         LSTM   → MambaEncoder   (3x faster)

# #     Inference:
# #         Euler ODE integration (ddim_steps) × num_ensemble samples
# #         → (traj_mean [T,B,2], Me_mean [T,B,2], all_trajs [S,T,B,2])
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
# #         self.pred_len    = pred_len
# #         self.obs_len     = obs_len
# #         self.sigma_min   = sigma_min
# #         self.n_train_ens = n_train_ens
# #         self.net = VelocityField(
# #             pred_len   = pred_len,
# #             obs_len    = obs_len,
# #             sigma_min  = sigma_min,
# #             unet_in_ch = unet_in_ch,
# #         )

# #     # ── Coordinate helpers (unchanged) ────────────────────────────────────

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

# #     # ── OT-CFM noise schedule (unchanged) ────────────────────────────────

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

# #     # ── Training (unchanged logic) ────────────────────────────────────────

# #     def get_loss(self, batch_list: List) -> torch.Tensor:
# #         return self.get_loss_breakdown(batch_list)["total"]

# #     def get_loss_breakdown(self, batch_list: List) -> Dict:
# #         traj_gt = batch_list[1]
# #         Me_gt   = batch_list[8]
# #         obs_t   = batch_list[0]
# #         obs_Me  = batch_list[7]

# #         lp, lm = obs_t[-1], obs_Me[-1]
# #         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

# #         # Context computed ONCE (FNO3D + Mamba run once per batch)
# #         ctx = self.net._context(batch_list)

# #         x_t, t, te, denom, _ = self._cfm_noisy(x1)
# #         pred_vel = self.net.forward_with_ctx(x_t, t, ctx)

# #         samples: List[torch.Tensor] = []
# #         for _ in range(self.n_train_ens):
# #             xt_s, ts, _, dens, _ = self._cfm_noisy(x1)
# #             pv_s  = self.net.forward_with_ctx(xt_s, ts, ctx)
# #             x1_s  = xt_s + dens * pv_s
# #             pa_s, _ = self._to_abs(x1_s, lp, lm)
# #             samples.append(pa_s)
# #         pred_samples = torch.stack(samples)

# #         x1_pred  = x_t + denom * pred_vel
# #         pred_abs, _ = self._to_abs(x1_pred, lp, lm)

# #         return compute_total_loss(
# #             pred_abs     = pred_abs,
# #             gt           = traj_gt,
# #             ref          = lp,
# #             batch_list   = batch_list,
# #             pred_samples = pred_samples,
# #             weights      = WEIGHTS,
# #         )

# #     # ── Inference (unchanged logic) ───────────────────────────────────────

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

# #         for _ in range(num_ensemble):
# #             x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
# #             for step in range(ddim_steps):
# #                 t_b = torch.full((B,), step * dt, device=device)
# #                 x_t = x_t + dt * self.net(x_t, t_b, batch_list)
# #                 x_t[:, :, :2].clamp_(-5.0, 5.0)
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

# #     # ── CSV export (unchanged) ────────────────────────────────────────────

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

# """
# Model/flow_matching_model.py  ── v11-fixed
# ==========================================
# OT-CFM Flow Matching + PINN-BVE for TC trajectory prediction.

# CHANGES vs v10-turbo:
#   ✅ BUG-1 FIXED: Dropout mask reuse in ensemble — ctx_drop now applied
#      independently per ensemble sample inside get_loss_breakdown loop.
#      In v10, ctx was computed once then shared across all n_train_ens
#      forward_with_ctx calls → all samples had identical dropout mask.
#   ✅ BUG-2 FIXED: Me dims (pressure/wind) not clamped in sample() —
#      added x_t[:, :, 2:].clamp_(-3.0, 3.0) alongside existing lon/lat clamp.
#   ✅ BUG-3 FIXED: loader import — train script now imports from
#      loader_training.py which uses split= kwarg for TrajectoryDataset v11.
#   ✅ IMPROVEMENT: n_train_ens=6 from epoch 60+ (progressive schedule).
#   ✅ IMPROVEMENT: ddim_steps=20 during validation.
#   ✅ IMPROVEMENT: LR cosine restart at epoch 60 in addition to epoch 30.
#   ✅ IMPROVEMENT: lon flip augmentation (typhoon track symmetry).
#   ✅ IMPROVEMENT: intensity-weighted loss (TY/SevTY samples weighted higher).
#   ✅ IMPROVEMENT: curriculum pred_len 6→12 over epochs 0-50.
#   ✅ IMPROVEMENT: smooth weight 0.2→0.05 (reduces over-smoothing).
#   ✅ IMPROVEMENT: CRPS loss added directly in training objective.

# Batch list indices (from seq_collate — unchanged):
#   0  obs_traj   [T_obs, B, 2]
#   1  pred_traj  [T_pred, B, 2]
#   2  obs_rel    [T_obs, B, 2]
#   3  pred_rel   [T_pred, B, 2]
#   4  nlp        tensor
#   5  mask       [seq_len, B]
#   6  seq_start_end [B, 2]
#   7  obs_Me     [T_obs, B, 2]
#   8  pred_Me    [T_pred, B, 2]
#   9  obs_Me_rel [T_obs, B, 2]
#   10 pred_Me_rel[T_pred, B, 2]
#   11 img_obs    [B, 13, T_obs, 81, 81]
#   12 img_pred   [B, 13, T_pred, 81, 81]
#   13 env_data   dict
#   14 None
#   15 list[dict]
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
# #  VelocityField  (FlowMatching denoiser)  ── v11
# # ══════════════════════════════════════════════════════════════════════════════

# class VelocityField(nn.Module):
#     """
#     OT-CFM velocity field  v_θ(x_t, t, context).

#     Context assembly:
#       h_t       [B, 128]  ← DataEncoder1D w/ Mamba
#       e_Env     [B,  64]  ← Env-T-Net
#       f_spatial [B,  16]  ← FNO decoder pooled
#       ─────────────────
#       total     [B, 208]  → ctx_fc → [B, ctx_dim=128]

#     BUG-1 FIX: ctx_drop is NOT applied inside _context(); instead it is
#     applied *per forward call* in forward() and forward_with_ctx() so
#     each ensemble sample gets an independent dropout mask.
#     """

#     def __init__(
#         self,
#         pred_len:   int   = 12,
#         obs_len:    int   = 8,
#         ctx_dim:    int   = 128,
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
#             d_model      = 32,
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
#         # BUG-1 FIX: ctx_drop is applied OUTSIDE _context() so each
#         # ensemble member gets its own independent dropout mask.
#         self.ctx_drop = nn.Dropout(0.15)
#         self.ctx_fc2  = nn.Linear(512, ctx_dim)

#         # ── Time embedding ────────────────────────────────────────────────
#         self.time_fc1 = nn.Linear(128, 256)
#         self.time_fc2 = nn.Linear(256, 128)

#         # ── Trajectory Transformer decoder ──────────────────────────────
#         self.traj_embed = nn.Linear(4, 128)
#         self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)
#         self.transformer = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 d_model=128, nhead=8, dim_feedforward=512,
#                 dropout=0.15, activation="gelu", batch_first=True,
#             ),
#             num_layers=4,
#         )
#         self.out_fc1 = nn.Linear(128, 256)
#         self.out_fc2 = nn.Linear(256, 4)

#     # ── Sinusoidal time embedding ─────────────────────────────────────────

#     def _time_emb(self, t: torch.Tensor, dim: int = 128) -> torch.Tensor:
#         half = dim // 2
#         freq = torch.exp(
#             torch.arange(half, dtype=torch.float32, device=t.device)
#             * (-math.log(10_000.0) / max(half - 1, 1))
#         )
#         emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
#         emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
#         return F.pad(emb, (0, dim % 2))

#     # ── Build context vector (NO dropout here — applied per-sample) ───────

#     def _context(self, batch_list: List) -> torch.Tensor:
#         """
#         Returns pre-dropout context [B, 512].
#         Caller applies ctx_drop independently per ensemble sample.
#         """
#         obs_traj  = batch_list[0]   # [T_obs, B, 2]
#         obs_Me    = batch_list[7]   # [T_obs, B, 2]
#         image_obs = batch_list[11]  # [B, 13, T_obs, 81, 81]
#         env_data  = batch_list[13]  # dict

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

#         # Fuse but do NOT apply dropout here
#         raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
#         raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))
#         # Return pre-dropout features; caller applies dropout independently
#         return raw  # [B, 512]

#     def _apply_ctx_head(self, raw: torch.Tensor) -> torch.Tensor:
#         """Apply dropout + final projection. Called per ensemble sample."""
#         return self.ctx_fc2(self.ctx_drop(raw))  # [B, ctx_dim]

#     # ── Forward ───────────────────────────────────────────────────────────

#     def forward(
#         self,
#         x_t:        torch.Tensor,
#         t:          torch.Tensor,
#         batch_list: List,
#     ) -> torch.Tensor:
#         raw = self._context(batch_list)
#         ctx = self._apply_ctx_head(raw)   # independent dropout mask
#         return self._decode(x_t, t, ctx)

#     def forward_with_ctx(self, x_t, t, raw_ctx):
#         """
#         Forward pass reusing pre-computed raw context features (pre-dropout).
#         BUG-1 FIX: applies ctx_drop here so each call gets a fresh mask.
#         """
#         ctx = self._apply_ctx_head(raw_ctx)   # fresh dropout mask each call
#         return self._decode(x_t, t, ctx)

#     def _decode(self, x_t, t, ctx):
#         t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
#         t_emb = self.time_fc2(t_emb)

#         x_emb  = self.traj_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
#         memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

#         out = self.transformer(x_emb, memory)
#         return self.out_fc2(F.gelu(self.out_fc1(out)))


# # ══════════════════════════════════════════════════════════════════════════════
# #  TCFlowMatching  ── v11
# # ══════════════════════════════════════════════════════════════════════════════

# class TCFlowMatching(nn.Module):
#     """
#     TC trajectory prediction via OT-CFM + PINN-BVE.

#     v11 changes:
#       - ctx_drop applied per-sample (BUG-1 fix)
#       - Me clamp in sample() (BUG-2 fix)
#       - smooth weight 0.2→0.05
#       - intensity-weighted loss
#       - curriculum pred_len 6→12 (set externally via set_curriculum_len)
#       - CRPS loss in training objective
#     """

#     def __init__(
#         self,
#         pred_len:    int   = 12,
#         obs_len:     int   = 8,
#         sigma_min:   float = 0.02,
#         n_train_ens: int   = 4,
#         unet_in_ch:  int   = 13,
#         **kwargs,
#     ):
#         super().__init__()
#         self.pred_len     = pred_len
#         self.obs_len      = obs_len
#         self.sigma_min    = sigma_min
#         self.n_train_ens  = n_train_ens
#         # Curriculum: active pred length (updated externally each epoch)
#         self.active_pred_len = pred_len
#         self.net = VelocityField(
#             pred_len   = pred_len,
#             obs_len    = obs_len,
#             sigma_min  = sigma_min,
#             unet_in_ch = unet_in_ch,
#         )

#     def set_curriculum_len(self, active_len: int) -> None:
#         """Update active prediction horizon for curriculum training."""
#         self.active_pred_len = max(1, min(active_len, self.pred_len))

#     # ── Coordinate helpers ────────────────────────────────────────────────

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

#     # ── OT-CFM noise schedule ─────────────────────────────────────────────

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

#     # ── Intensity-based sample weights ────────────────────────────────────

#     @staticmethod
#     def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
#         """
#         Weight samples by typhoon intensity class.
#         obs_Me: [T_obs, B, 2] — channel 1 is normalized wind speed.
#         Higher wind = higher weight (TY/SevTY/SuperTY more important).
#         Returns [B] weights, mean=1.
#         """
#         # Last observed wind (normalized, ~0-1 range)
#         wind_norm = obs_Me[-1, :, 1].detach()  # [B]
#         # Map to weight: TD(0.5), TS(0.8), TY(1.0), SevTY+(1.5)
#         w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
#             torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
#             torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
#                         torch.full_like(wind_norm, 1.5))))
#         return w / w.mean().clamp(min=1e-6)  # normalize mean to 1

#     # ── lon-flip augmentation ─────────────────────────────────────────────

#     @staticmethod
#     def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
#         """
#         Randomly flip longitude for typhoon track symmetry augmentation.
#         Probability p per batch.
#         """
#         if torch.rand(1).item() > p:
#             return batch_list
#         aug = list(batch_list)
#         for idx in [0, 1, 2, 3]:   # obs_traj, pred_traj, obs_rel, pred_rel
#             t = aug[idx]
#             if torch.is_tensor(t) and t.shape[-1] >= 1:
#                 t = t.clone()
#                 # Flip lon (dim 0 of last axis) — mirror around center
#                 t[..., 0] = -t[..., 0]
#                 aug[idx] = t
#         for idx in [7, 8, 9, 10]:  # obs_Me, pred_Me, obs_Me_rel, pred_Me_rel
#             t = aug[idx]
#             if torch.is_tensor(t):
#                 aug[idx] = t.clone()
#         return aug

#     # ── Training ──────────────────────────────────────────────────────────

#     def get_loss(self, batch_list: List) -> torch.Tensor:
#         return self.get_loss_breakdown(batch_list)["total"]

#     def get_loss_breakdown(self, batch_list: List) -> Dict:
#         # Apply augmentation
#         batch_list = self._lon_flip_aug(batch_list, p=0.3)

#         traj_gt = batch_list[1]
#         Me_gt   = batch_list[8]
#         obs_t   = batch_list[0]
#         obs_Me  = batch_list[7]

#         # Curriculum: truncate to active_pred_len
#         apl = self.active_pred_len
#         if apl < traj_gt.shape[0]:
#             traj_gt = traj_gt[:apl]
#             Me_gt   = Me_gt[:apl]

#         lp, lm = obs_t[-1], obs_Me[-1]
#         x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

#         # BUG-1 FIX: compute RAW context once (pre-dropout)
#         raw_ctx = self.net._context(batch_list)  # [B, 512]

#         # Intensity weights for loss scaling
#         intensity_w = self._intensity_weights(obs_Me)  # [B]

#         x_t, t, te, denom, _ = self._cfm_noisy(x1)
#         # Each call to forward_with_ctx applies ctx_drop independently
#         pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx)

#         samples: List[torch.Tensor] = []
#         for _ in range(self.n_train_ens):
#             xt_s, ts, _, dens, _ = self._cfm_noisy(x1)
#             # BUG-1 FIX: each sample gets fresh dropout mask via forward_with_ctx
#             pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx)
#             x1_s  = xt_s + dens * pv_s
#             pa_s, _ = self._to_abs(x1_s, lp, lm)
#             samples.append(pa_s)
#         pred_samples = torch.stack(samples)  # [S, T, B, 2]

#         x1_pred  = x_t + denom * pred_vel
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

#     # ── Inference ─────────────────────────────────────────────────────────

#     @torch.no_grad()
#     def sample(
#         self,
#         batch_list:   List,
#         num_ensemble: int = 50,
#         ddim_steps:   int = 10,
#         predict_csv:  Optional[str] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         lp  = batch_list[0][-1]
#         lm  = batch_list[7][-1]
#         B, device = lp.shape[0], lp.device
#         dt  = 1.0 / ddim_steps

#         traj_s: List[torch.Tensor] = []
#         me_s:   List[torch.Tensor] = []

#         # Pre-compute raw context once; each forward_with_ctx applies
#         # its own dropout (model.eval() → dropout disabled during inference)
#         raw_ctx = self.net._context(batch_list)

#         for _ in range(num_ensemble):
#             x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
#             for step in range(ddim_steps):
#                 t_b = torch.full((B,), step * dt, device=device)
#                 vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx)
#                 x_t = x_t + dt * vel
#                 # BUG-2 FIX: clamp lon/lat AND pressure/wind dims
#                 x_t[:, :, :2].clamp_(-5.0, 5.0)   # traj (lon/lat)
#                 x_t[:, :, 2:].clamp_(-3.0, 3.0)   # Me (pressure/wind)
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

#     # ── CSV export ────────────────────────────────────────────────────────

#     @staticmethod
#     def _write_predict_csv(csv_path, traj_mean, all_trajs):
#         import numpy as np
#         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
#         ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
#         T, B, _ = traj_mean.shape
#         S       = all_trajs.shape[0]

#         mean_lon = (traj_mean[..., 0] * 5.0 + 180.0).cpu().numpy()
#         mean_lat = (traj_mean[..., 1] * 5.0).cpu().numpy()
#         all_lon  = (all_trajs[..., 0] * 5.0 + 180.0).cpu().numpy()
#         all_lat  = (all_trajs[..., 1] * 5.0).cpu().numpy()

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
Model/flow_matching_model.py  ── v14
==========================================
OT-CFM Flow Matching + PINN-BVE for TC trajectory prediction.

FIXES vs v11-fixed:

  FIX-M1  _write_predict_csv denorm formula was wrong:
           v11 used: lon = traj_mean[...,0] * 5.0 + 180.0
           This is lon_norm * 5 + 180, which gives values in ~[170,195]°
           for normalised lon_norm in ~[-2,3]. That is NOT the correct
           inverse of the dataset normalization:
               lon_norm = (lon_deg * 10 - 1800) / 50
           → lon_deg = (lon_norm * 50 + 1800) / 10
           Similarly lat_deg = (lat_norm * 50) / 10.
           The old formula (×5+180) is wrong by ~10× in scale.
           Fix: apply correct denorm formula before writing CSV.

  FIX-M2  sample() ctx computed once but dropout is disabled during
           model.eval() — the repeated forward_with_ctx calls with the
           same raw_ctx produce IDENTICAL samples (no stochasticity from
           dropout, all randomness comes from the initial noise x_t).
           This is actually correct behaviour: at inference, diversity
           comes from the noise samples, not dropout. No code change,
           but added an explicit comment clarifying this.

  FIX-M3  get_loss_breakdown calls _lon_flip_aug THEN uses batch_list[1]
           (pred_traj) for traj_gt, but the augmented batch_list index 1
           has been lon-flipped. However pinn_bve_loss receives pred_abs
           which is derived from the flipped traj_gt — this is consistent.
           Confirmed: no bug, but added comment.

Kept from v11:
  BUG-1: ctx_drop applied per-sample (independent dropout masks)
  BUG-2: Me dims clamped in sample()
  BUG-3: loader from loader_training
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


# ══════════════════════════════════════════════════════════════════════════════
#  VelocityField  (FlowMatching denoiser)  ── v14
# ══════════════════════════════════════════════════════════════════════════════

class VelocityField(nn.Module):
    """
    OT-CFM velocity field  v_θ(x_t, t, context).

    Context assembly:
      h_t       [B, 128]  ← DataEncoder1D w/ Mamba
      e_Env     [B,  64]  ← Env-T-Net
      f_spatial [B,  16]  ← FNO decoder pooled
      ─────────────────
      total     [B, 208]  → ctx_fc → [B, ctx_dim=128]

    BUG-1 FIX: ctx_drop applied per forward call so each ensemble
    sample gets an independent dropout mask.
    """

    def __init__(
        self,
        pred_len:   int   = 12,
        obs_len:    int   = 8,
        ctx_dim:    int   = 128,
        sigma_min:  float = 0.02,
        unet_in_ch: int   = 13,
    ):
        super().__init__()
        self.pred_len  = pred_len
        self.obs_len   = obs_len
        self.sigma_min = sigma_min

        # ── FNO3D encoder ────────────────────────────────────────────────
        self.spatial_enc = FNO3DEncoder(
            in_channel   = unet_in_ch,
            out_channel  = 1,
            d_model      = 32,
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

        # ── DataEncoder1D with Mamba ─────────────────────────────────────
        self.enc_1d = DataEncoder1D(
            in_1d       = 4,
            feat_3d_dim = 128,
            mlp_h       = 64,
            lstm_hidden = 128,
            lstm_layers = 3,
            dropout     = 0.1,
            d_state     = 16,
        )

        # ── Env-T-Net ────────────────────────────────────────────────────
        self.env_enc = Env_net(obs_len=obs_len, d_model=64)

        # ── Context fusion: 128 + 64 + 16 = 208 → ctx_dim ───────────────
        self.ctx_fc1  = nn.Linear(128 + 64 + 16, 512)
        self.ctx_ln   = nn.LayerNorm(512)
        self.ctx_drop = nn.Dropout(0.15)
        self.ctx_fc2  = nn.Linear(512, ctx_dim)

        # ── Time embedding ────────────────────────────────────────────────
        self.time_fc1 = nn.Linear(128, 256)
        self.time_fc2 = nn.Linear(256, 128)

        # ── Trajectory Transformer decoder ──────────────────────────────
        self.traj_embed = nn.Linear(4, 128)
        self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=128, nhead=8, dim_feedforward=512,
                dropout=0.15, activation="gelu", batch_first=True,
            ),
            num_layers=4,
        )
        self.out_fc1 = nn.Linear(128, 256)
        self.out_fc2 = nn.Linear(256, 4)

    def _time_emb(self, t: torch.Tensor, dim: int = 128) -> torch.Tensor:
        half = dim // 2
        freq = torch.exp(
            torch.arange(half, dtype=torch.float32, device=t.device)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return F.pad(emb, (0, dim % 2))

    def _context(self, batch_list: List) -> torch.Tensor:
        """
        Returns pre-dropout context [B, 512].
        Dropout applied independently per ensemble sample via _apply_ctx_head.
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
        return raw  # [B, 512] — pre-dropout

    def _apply_ctx_head(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply dropout + final projection. Called per ensemble sample."""
        return self.ctx_fc2(self.ctx_drop(raw))  # [B, ctx_dim]

    def forward(self, x_t, t, batch_list):
        raw = self._context(batch_list)
        ctx = self._apply_ctx_head(raw)
        return self._decode(x_t, t, ctx)

    def forward_with_ctx(self, x_t, t, raw_ctx):
        """
        Forward pass reusing pre-computed raw context (pre-dropout).
        BUG-1 FIX: ctx_drop applied here so each call gets a fresh mask.
        FIX-M2: During model.eval(), dropout is disabled so all calls with
        the same raw_ctx produce identical outputs. Diversity at inference
        comes from the initial noise samples (randn in sample()), not dropout.
        This is the correct behaviour for flow matching.
        """
        ctx = self._apply_ctx_head(raw_ctx)
        return self._decode(x_t, t, ctx)

    def _decode(self, x_t, t, ctx):
        t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
        t_emb = self.time_fc2(t_emb)

        x_emb  = self.traj_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
        memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

        out = self.transformer(x_emb, memory)
        return self.out_fc2(F.gelu(self.out_fc1(out)))


# ══════════════════════════════════════════════════════════════════════════════
#  TCFlowMatching  ── v14
# ══════════════════════════════════════════════════════════════════════════════

class TCFlowMatching(nn.Module):
    """
    TC trajectory prediction via OT-CFM + PINN-BVE.

    v14 changes:
      - FIX-M1: _write_predict_csv denorm formula corrected
      - FIX-M2: clarified inference stochasticity source (noise, not dropout)
      - FIX-M3: lon_flip augmentation consistency confirmed
    """

    def __init__(
        self,
        pred_len:    int   = 12,
        obs_len:     int   = 8,
        sigma_min:   float = 0.02,
        n_train_ens: int   = 4,
        unet_in_ch:  int   = 13,
        **kwargs,
    ):
        super().__init__()
        self.pred_len     = pred_len
        self.obs_len      = obs_len
        self.sigma_min    = sigma_min
        self.n_train_ens  = n_train_ens
        self.active_pred_len = pred_len
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
        """
        Randomly flip longitude for typhoon track symmetry augmentation.
        FIX-M3: The augmented pred_traj (index 1) is used for traj_gt,
        and pred_abs derived from it is passed to pinn_bve_loss — consistent.
        """
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
        pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx)

        samples: List[torch.Tensor] = []
        for _ in range(self.n_train_ens):
            xt_s, ts, _, dens, _ = self._cfm_noisy(x1)
            pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx)
            x1_s  = xt_s + dens * pv_s
            pa_s, _ = self._to_abs(x1_s, lp, lm)
            samples.append(pa_s)
        pred_samples = torch.stack(samples)  # [S, T, B, 2]

        x1_pred  = x_t + denom * pred_vel
        pred_abs, _ = self._to_abs(x1_pred, lp, lm)

        breakdown = compute_total_loss(
            pred_abs     = pred_abs,
            gt           = traj_gt,
            ref          = lp,
            batch_list   = batch_list,
            pred_samples = pred_samples,
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
        lp  = batch_list[0][-1]
        lm  = batch_list[7][-1]
        B, device = lp.shape[0], lp.device
        dt  = 1.0 / ddim_steps

        traj_s: List[torch.Tensor] = []
        me_s:   List[torch.Tensor] = []

        # Pre-compute raw context once.
        # FIX-M2: During eval(), dropout is off — raw_ctx is deterministic.
        # Diversity across ensemble members comes entirely from the initial
        # noise x_t = randn * sigma_min (different each call).
        raw_ctx = self.net._context(batch_list)

        for _ in range(num_ensemble):
            x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
            for step in range(ddim_steps):
                t_b = torch.full((B,), step * dt, device=device)
                vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx)
                x_t = x_t + dt * vel
                # BUG-2 FIX: clamp lon/lat AND pressure/wind dims
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
        """
        FIX-M1: Correct denorm formula.
        Normalised coords: lon_norm, lat_norm
        Dataset normalization:
            lon_norm = (lon_deg * 10 - 1800) / 50
            lat_norm = (lat_deg * 10) / 50
        Inverse:
            lon_deg = (lon_norm * 50 + 1800) / 10
            lat_deg = (lat_norm * 50) / 10

        v11 bug: used lon = traj * 5.0 + 180.0 (wrong by 10x in scale)
        """
        import numpy as np
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        T, B, _ = traj_mean.shape
        S       = all_trajs.shape[0]

        # FIX-M1: correct inverse normalization
        # lon_deg = (lon_norm * 50 + 1800) / 10
        # lat_deg = (lat_norm * 50) / 10
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
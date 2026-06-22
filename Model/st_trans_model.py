# # # # # # # # # """
# # # # # # # # # Model/st_trans_model.py  ── ST-Trans Baseline
# # # # # # # # # ===============================================
# # # # # # # # # THUẬT TOÁN: Faiaz et al. (2026)
# # # # # # # # # "Physics-guided non-autoregressive transformer for lightweight
# # # # # # # # # cyclone track prediction in the Bay of Bengal"
# # # # # # # # # Expert Systems With Applications 317 (2026) 131972

# # # # # # # # # CHIẾN LƯỢC (theo paper §3):
# # # # # # # # #   - Input : obs_traj cuối (obs_len bước) → encode 8D features
# # # # # # # # #   - CNN   : reshape → [S, 1, 2, 4] grid → 2×2 conv → 1×2 conv → 64D
# # # # # # # # #   - Encoder: Transformer encoder (self-attention over S steps)
# # # # # # # # #   - Decoder: Non-autoregressive, learned horizon queries [H×dmodel]
# # # # # # # # #   - Output : toàn bộ H bước predict song song (không autoregressive)

# # # # # # # # # LOSS (§3.5.1 - Physics-guided composite):
# # # # # # # # #   L = L_DPE + 0.05*L_MSE + λ_speed*L_speed + λ_accel*L_accel
# # # # # # # # #   λ_speed = 0.1, λ_accel = 0.01, v_max = 80 km/h

# # # # # # # # # ADAPTED cho TCND_vn:
# # # # # # # # #   - Input features: lat, lon, speed, heading, year, month, day, hour
# # # # # # # # #     (giống 8D của paper, extract từ obs_traj + metadata)
# # # # # # # # #   - Dùng chung DataLoader của bạn
# # # # # # # # #   - Output: [pred_len, B, 2] normalised (cùng format với bài của bạn)
# # # # # # # # #   - Eval bằng cùng haversine ADE metrics (12h/24h/48h/72h)

# # # # # # # # # NOTE:
# # # # # # # # #   - Paper dùng S=3 (9h), H=16 (48h) ở resolution 3h
# # # # # # # # #   - Bạn dùng S=8 (obs_len), H=12 (pred_len) ở resolution 6h → 72h
# # # # # # # # #   - Điều chỉnh: S=obs_len, H=pred_len, giữ nguyên architecture
# # # # # # # # # """
# # # # # # # # # from __future__ import annotations

# # # # # # # # # import math
# # # # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # # # import numpy as np
# # # # # # # # # import torch
# # # # # # # # # import torch.nn as nn
# # # # # # # # # import torch.nn.functional as F

# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  Haversine helpers  (cùng convention với bài của bạn)
# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
# # # # # # # # #     """Denormalise từ normalised space → degrees."""
# # # # # # # # #     out = arr.clone()
# # # # # # # # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0   # lon
# # # # # # # # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0              # lat
# # # # # # # # #     return out


# # # # # # # # # def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # # # # # # #     """Haversine distance [km]. p1, p2: [..., 2] degrees (lon, lat)."""
# # # # # # # # #     lon1 = torch.deg2rad(p1[..., 0]); lat1 = torch.deg2rad(p1[..., 1])
# # # # # # # # #     lon2 = torch.deg2rad(p2[..., 0]); lat2 = torch.deg2rad(p2[..., 1])
# # # # # # # # #     dlat = lat2 - lat1; dlon = lon2 - lon1
# # # # # # # # #     a = torch.sin(dlat/2).pow(2) + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2)
# # # # # # # # #     return 2.0 * 6371.0 * torch.asin(a.clamp(1e-12, 1.0).sqrt())


# # # # # # # # # HORIZON_STEPS: Dict[int, int] = {12: 1, 24: 3, 48: 7, 72: 11}


# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  Feature extractor  ── lấy 8D features từ obs_traj
# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # class ObsTrajFeatureExtractor(nn.Module):
# # # # # # # # #     """
# # # # # # # # #     Extract 8D features từ obs_traj [T_obs, B, 2] (normalised lat/lon).

# # # # # # # # #     Paper features:
# # # # # # # # #       1. lat_norm, lon_norm     (position)
# # # # # # # # #       2. speed_norm             (translation speed estimate)
# # # # # # # # #       3. heading_norm           (motion direction)
# # # # # # # # #       4. year_norm, month_norm, day_norm, hour_norm  (temporal)

# # # # # # # # #     Vì bạn không có temporal metadata trong batch, ta dùng:
# # # # # # # # #       1-2: lat, lon (từ obs_traj)
# # # # # # # # #       3-4: delta_lat, delta_lon (velocity)
# # # # # # # # #       5-6: delta2_lat, delta2_lon (acceleration)
# # # # # # # # #       7-8: step index (normalized), cumulative distance

# # # # # # # # #     → 8D vẫn đủ để encode kinematic history
# # # # # # # # #     """

# # # # # # # # #     def __init__(self, feat_dim: int = 8):
# # # # # # # # #         super().__init__()
# # # # # # # # #         self.feat_dim = feat_dim

# # # # # # # # #     def forward(self, obs_traj: torch.Tensor) -> torch.Tensor:
# # # # # # # # #         """
# # # # # # # # #         obs_traj: [T_obs, B, 2] normalised (lon, lat)
# # # # # # # # #         Returns:  [B, T_obs, 8]
# # # # # # # # #         """
# # # # # # # # #         T, B, _ = obs_traj.shape
# # # # # # # # #         device   = obs_traj.device

# # # # # # # # #         # Position
# # # # # # # # #         lon = obs_traj[:, :, 0]   # [T, B]
# # # # # # # # #         lat = obs_traj[:, :, 1]   # [T, B]

# # # # # # # # #         # Velocity (delta per step)
# # # # # # # # #         if T >= 2:
# # # # # # # # #             d_lon = torch.cat([obs_traj[1:, :, 0] - obs_traj[:-1, :, 0],
# # # # # # # # #                                torch.zeros(1, B, device=device)], dim=0)
# # # # # # # # #             d_lat = torch.cat([obs_traj[1:, :, 1] - obs_traj[:-1, :, 1],
# # # # # # # # #                                torch.zeros(1, B, device=device)], dim=0)
# # # # # # # # #         else:
# # # # # # # # #             d_lon = torch.zeros(T, B, device=device)
# # # # # # # # #             d_lat = torch.zeros(T, B, device=device)

# # # # # # # # #         # Acceleration (delta of delta)
# # # # # # # # #         if T >= 3:
# # # # # # # # #             dd_lon = torch.cat([d_lon[1:] - d_lon[:-1],
# # # # # # # # #                                 torch.zeros(1, B, device=device)], dim=0)
# # # # # # # # #             dd_lat = torch.cat([d_lat[1:] - d_lat[:-1],
# # # # # # # # #                                 torch.zeros(1, B, device=device)], dim=0)
# # # # # # # # #         else:
# # # # # # # # #             dd_lon = torch.zeros(T, B, device=device)
# # # # # # # # #             dd_lat = torch.zeros(T, B, device=device)

# # # # # # # # #         # Step index (0→1 normalized)
# # # # # # # # #         step_idx = torch.linspace(0, 1, T, device=device).unsqueeze(1).expand(T, B)

# # # # # # # # #         # Speed magnitude
# # # # # # # # #         speed = torch.sqrt(d_lon.pow(2) + d_lat.pow(2)).clamp(min=0)

# # # # # # # # #         # Stack: [T, B, 8]
# # # # # # # # #         features = torch.stack([lon, lat, d_lon, d_lat,
# # # # # # # # #                                  dd_lon, dd_lat, step_idx, speed], dim=-1)
# # # # # # # # #         return features.permute(1, 0, 2)   # [B, T_obs, 8]


# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  CNN State Encoder  ── §3.3.2
# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # class CNNStateEncoder(nn.Module):
# # # # # # # # #     """
# # # # # # # # #     Per-timestep CNN encoder.
# # # # # # # # #     Input: 8D feature vector → reshape [1, 2, 4] → conv → 64D.

# # # # # # # # #     Paper §3.3.2:
# # # # # # # # #       - Reshape 8D → [1, 2, 4] synthetic grid
# # # # # # # # #       - Conv1: 2×2 kernel, 1→32 channels, ReLU
# # # # # # # # #       - Conv2: 1×2 kernel, 32→32 channels, ReLU
# # # # # # # # #       - Flatten → 64D
# # # # # # # # #     """

# # # # # # # # #     def __init__(self, feat_dim: int = 8, out_dim: int = 64):
# # # # # # # # #         super().__init__()
# # # # # # # # #         self.feat_dim = feat_dim
# # # # # # # # #         self.out_dim  = out_dim

# # # # # # # # #         self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, 2), stride=1, padding=0)
# # # # # # # # #         self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=1, padding=0)

# # # # # # # # #         # After conv1: [B, 32, 1, 3], after conv2: [B, 32, 1, 2] → flatten=64
# # # # # # # # #         self.proj = nn.Linear(64, out_dim)

# # # # # # # # #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# # # # # # # # #         """
# # # # # # # # #         x: [B, S, 8]
# # # # # # # # #         Returns: [B, S, out_dim]
# # # # # # # # #         """
# # # # # # # # #         B, S, _ = x.shape

# # # # # # # # #         # Process each timestep independently
# # # # # # # # #         x_flat = x.reshape(B * S, 1, 2, 4)   # [B*S, 1, 2, 4]

# # # # # # # # #         h = F.relu(self.conv1(x_flat))         # [B*S, 32, 1, 3]
# # # # # # # # #         h = F.relu(self.conv2(h))              # [B*S, 32, 1, 2]
# # # # # # # # #         h = h.flatten(1)                       # [B*S, 64]
# # # # # # # # #         h = self.proj(h)                       # [B*S, out_dim]

# # # # # # # # #         return h.reshape(B, S, self.out_dim)


# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  Sinusoidal Positional Encoding
# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # class SinusoidalPE(nn.Module):
# # # # # # # # #     def __init__(self, d_model: int, max_len: int = 200):
# # # # # # # # #         super().__init__()
# # # # # # # # #         pe = torch.zeros(max_len, d_model)
# # # # # # # # #         pos = torch.arange(max_len).unsqueeze(1).float()
# # # # # # # # #         div = torch.exp(torch.arange(0, d_model, 2).float() *
# # # # # # # # #                         (-math.log(10000.0) / d_model))
# # # # # # # # #         pe[:, 0::2] = torch.sin(pos * div)
# # # # # # # # #         pe[:, 1::2] = torch.cos(pos * div)
# # # # # # # # #         self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

# # # # # # # # #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# # # # # # # # #         return x + self.pe[:, :x.size(1), :]


# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  ST-Trans Main Model  ── §3.3.1
# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # class STTrans(nn.Module):
# # # # # # # # #     """
# # # # # # # # #     Physics-guided Non-Autoregressive Transformer for TC track prediction.

# # # # # # # # #     Architecture (paper §3.3):
# # # # # # # # #       1. Feature extraction: obs_traj → 8D features
# # # # # # # # #       2. CNN state encoder: 8D → 64D per timestep
# # # # # # # # #       3. Linear projection: 64D → dmodel
# # # # # # # # #       4. Transformer encoder: self-attention over S history steps
# # # # # # # # #       5. Horizon queries: H learned queries [H, dmodel]
# # # # # # # # #       6. Transformer decoder: cross-attention → [H, dmodel]
# # # # # # # # #       7. Regression head: [H, dmodel] → [H, 2]

# # # # # # # # #     Physics loss (§3.5.1):
# # # # # # # # #       L = L_DPE + 0.05*L_MSE + 0.1*L_speed + 0.01*L_accel
# # # # # # # # #     """

# # # # # # # # #     def __init__(
# # # # # # # # #         self,
# # # # # # # # #         obs_len:    int   = 8,
# # # # # # # # #         pred_len:   int   = 12,
# # # # # # # # #         feat_dim:   int   = 8,
# # # # # # # # #         d_model:    int   = 64,
# # # # # # # # #         nhead:      int   = 4,
# # # # # # # # #         num_enc_layers: int = 1,
# # # # # # # # #         num_dec_layers: int = 3,
# # # # # # # # #         dim_ff:     int   = 512,
# # # # # # # # #         dropout:    float = 0.1,
# # # # # # # # #         # Physics loss weights (paper §3.5.1)
# # # # # # # # #         lambda_speed: float = 0.1,
# # # # # # # # #         lambda_accel: float = 0.01,
# # # # # # # # #         w_mse:        float = 0.05,
# # # # # # # # #         v_max_kmh:    float = 80.0,   # paper default
# # # # # # # # #         dt_h:         float = 6.0,    # 6h per step (your dataset)
# # # # # # # # #     ):
# # # # # # # # #         super().__init__()
# # # # # # # # #         self.obs_len    = obs_len
# # # # # # # # #         self.pred_len   = pred_len
# # # # # # # # #         self.d_model    = d_model
# # # # # # # # #         self.lambda_speed = lambda_speed
# # # # # # # # #         self.lambda_accel = lambda_accel
# # # # # # # # #         self.w_mse        = w_mse
# # # # # # # # #         # v_max in normalised units: 80 km/h * 6h = 480 km ≈ 480/111 ≈ 4.3 deg
# # # # # # # # #         # normalised: 4.3 / 5.0 ≈ 0.86 per step
# # # # # # # # #         self.v_max_norm = v_max_kmh * dt_h / (111.0 * 50.0)

# # # # # # # # #         # ── Feature extraction ────────────────────────────────────────────
# # # # # # # # #         self.feat_extractor = ObsTrajFeatureExtractor(feat_dim)
# # # # # # # # #         self.cnn_enc        = CNNStateEncoder(feat_dim, d_model)
# # # # # # # # #         self.input_proj     = nn.Linear(d_model, d_model)
# # # # # # # # #         self.enc_pe         = SinusoidalPE(d_model, max_len=obs_len + 10)

# # # # # # # # #         # ── Transformer encoder ───────────────────────────────────────────
# # # # # # # # #         enc_layer = nn.TransformerEncoderLayer(
# # # # # # # # #             d_model=d_model, nhead=nhead,
# # # # # # # # #             dim_feedforward=dim_ff, dropout=dropout,
# # # # # # # # #             activation="relu", batch_first=True,
# # # # # # # # #         )
# # # # # # # # #         self.transformer_enc = nn.TransformerEncoder(enc_layer,
# # # # # # # # #                                                       num_layers=num_enc_layers)

# # # # # # # # #         # ── Horizon queries (learned, paper §3.3.4) ───────────────────────
# # # # # # # # #         self.horizon_queries = nn.Parameter(
# # # # # # # # #             torch.randn(1, pred_len, d_model) * 0.02
# # # # # # # # #         )
# # # # # # # # #         self.dec_pe = SinusoidalPE(d_model, max_len=pred_len + 10)

# # # # # # # # #         # ── Transformer decoder ───────────────────────────────────────────
# # # # # # # # #         dec_layer = nn.TransformerDecoderLayer(
# # # # # # # # #             d_model=d_model, nhead=nhead,
# # # # # # # # #             dim_feedforward=dim_ff, dropout=dropout,
# # # # # # # # #             activation="relu", batch_first=True,
# # # # # # # # #         )
# # # # # # # # #         self.transformer_dec = nn.TransformerDecoder(dec_layer,
# # # # # # # # #                                                       num_layers=num_dec_layers)

# # # # # # # # #         # ── Regression head (paper §3.3.4: g(dτ) = W2·σ(W1·dτ+b1)+b2) ──
# # # # # # # # #         self.reg_head = nn.Sequential(
# # # # # # # # #             nn.Linear(d_model, d_model),
# # # # # # # # #             nn.ReLU(),
# # # # # # # # #             nn.Linear(d_model, 2),
# # # # # # # # #         )

# # # # # # # # #         self._init_weights()

# # # # # # # # #     def _init_weights(self):
# # # # # # # # #         for m in self.modules():
# # # # # # # # #             if isinstance(m, nn.Linear):
# # # # # # # # #                 nn.init.xavier_uniform_(m.weight, gain=0.5)
# # # # # # # # #                 if m.bias is not None:
# # # # # # # # #                     nn.init.zeros_(m.bias)

# # # # # # # # #     # ── Forward ───────────────────────────────────────────────────────────

# # # # # # # # #     def forward(self, obs_traj: torch.Tensor) -> torch.Tensor:
# # # # # # # # #         """
# # # # # # # # #         obs_traj: [T_obs, B, 2] normalised
# # # # # # # # #         Returns:  [pred_len, B, 2] normalised predictions
# # # # # # # # #         """
# # # # # # # # #         B = obs_traj.shape[1]

# # # # # # # # #         # 1. Extract 8D features
# # # # # # # # #         feat = self.feat_extractor(obs_traj)   # [B, S, 8]

# # # # # # # # #         # 2. CNN state encoder
# # # # # # # # #         h = self.cnn_enc(feat)                  # [B, S, d_model]
# # # # # # # # #         h = self.input_proj(h)                  # [B, S, d_model]
# # # # # # # # #         h = self.enc_pe(h)                      # [B, S, d_model]

# # # # # # # # #         # 3. Transformer encoder → memory
# # # # # # # # #         memory = self.transformer_enc(h)        # [B, S, d_model]

# # # # # # # # #         # 4. Horizon queries with positional encoding
# # # # # # # # #         Q = self.horizon_queries.expand(B, -1, -1)   # [B, H, d_model]
# # # # # # # # #         Q = self.dec_pe(Q)                            # [B, H, d_model]

# # # # # # # # #         # 5. Transformer decoder (non-autoregressive cross-attention)
# # # # # # # # #         D = self.transformer_dec(Q, memory)     # [B, H, d_model]

# # # # # # # # #         # 6. Regression head → [B, H, 2]
# # # # # # # # #         out = self.reg_head(D)                  # [B, H, 2]

# # # # # # # # #         # Return [pred_len, B, 2] (same format as your model)
# # # # # # # # #         return out.permute(1, 0, 2)

# # # # # # # # #     # ── Physics-guided loss (§3.5.1) ─────────────────────────────────────

# # # # # # # # #     def physics_loss(
# # # # # # # # #         self,
# # # # # # # # #         pred_norm: torch.Tensor,   # [T, B, 2] normalised
# # # # # # # # #         gt_norm:   torch.Tensor,   # [T, B, 2] normalised
# # # # # # # # #     ) -> Dict:
# # # # # # # # #         T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # # #         pred = pred_norm[:T]
# # # # # # # # #         gt   = gt_norm[:T]

# # # # # # # # #         # Convert to degrees for haversine
# # # # # # # # #         pred_deg = _norm_to_deg(pred)
# # # # # # # # #         gt_deg   = _norm_to_deg(gt)

# # # # # # # # #         # L_DPE: mean great-circle distance (eq. 16)
# # # # # # # # #         dist = haversine_km(pred_deg, gt_deg)   # [T, B]
# # # # # # # # #         l_dpe = dist.mean()

# # # # # # # # #         # L_MSE: coordinate space MSE (eq. 17)
# # # # # # # # #         l_mse = F.mse_loss(pred, gt)

# # # # # # # # #         # L_speed: penalize speeds > v_max (eq. 18-20)
# # # # # # # # #         if T >= 2:
# # # # # # # # #             step_dist = torch.sqrt(
# # # # # # # # #                 (pred[1:, :, 0] - pred[:-1, :, 0]).pow(2) +
# # # # # # # # #                 (pred[1:, :, 1] - pred[:-1, :, 1]).pow(2)
# # # # # # # # #             )   # [T-1, B] in normalised units
# # # # # # # # #             excess_speed = F.relu(step_dist - self.v_max_norm)
# # # # # # # # #             l_speed = excess_speed.pow(2).mean()
# # # # # # # # #         else:
# # # # # # # # #             l_speed = pred_norm.new_zeros(())

# # # # # # # # #         # L_accel: penalize acceleration changes (eq. 21-22)
# # # # # # # # #         if T >= 3:
# # # # # # # # #             vel = pred[1:] - pred[:-1]        # [T-1, B, 2]
# # # # # # # # #             spd = vel.norm(dim=-1)             # [T-1, B]
# # # # # # # # #             accel = (spd[1:] - spd[:-1]).pow(2).mean()
# # # # # # # # #             l_accel = accel
# # # # # # # # #         else:
# # # # # # # # #             l_accel = pred_norm.new_zeros(())

# # # # # # # # #         # Total physics-guided loss (eq. 15)
# # # # # # # # #         total = (l_dpe
# # # # # # # # #                  + self.w_mse        * l_mse
# # # # # # # # #                  + self.lambda_speed * l_speed
# # # # # # # # #                  + self.lambda_accel * l_accel)

# # # # # # # # #         return dict(
# # # # # # # # #             total=total,
# # # # # # # # #             dpe=l_dpe.item(),
# # # # # # # # #             mse=l_mse.item(),
# # # # # # # # #             speed=l_speed.item(),
# # # # # # # # #             accel=l_accel.item(),
# # # # # # # # #         )

# # # # # # # # #     # ── Interface matching TCFlowMatching ─────────────────────────────────

# # # # # # # # #     def get_loss(self, batch_list) -> torch.Tensor:
# # # # # # # # #         return self.get_loss_breakdown(batch_list)["total"]

# # # # # # # # #     def get_loss_breakdown(self, batch_list) -> Dict:
# # # # # # # # #         obs_traj = batch_list[0]   # [T_obs, B, 2]
# # # # # # # # #         traj_gt  = batch_list[1]   # [T_gt, B, 2]

# # # # # # # # #         pred = self.forward(obs_traj)
# # # # # # # # #         return self.physics_loss(pred, traj_gt)

# # # # # # # # #     @torch.no_grad()
# # # # # # # # #     def sample(
# # # # # # # # #         self,
# # # # # # # # #         batch_list,
# # # # # # # # #         num_ensemble: int = 1,
# # # # # # # # #         **kwargs,
# # # # # # # # #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# # # # # # # # #         """
# # # # # # # # #         Returns:
# # # # # # # # #             pred_mean : [T, B, 2] normalised
# # # # # # # # #             me_mean   : [T, B, 2] zeros
# # # # # # # # #             all_trajs : [1, T, B, 2]
# # # # # # # # #         """
# # # # # # # # #         obs_traj = batch_list[0]
# # # # # # # # #         pred = self.forward(obs_traj)
# # # # # # # # #         T, B, _ = pred.shape
# # # # # # # # #         me_mean   = torch.zeros(T, B, 2, device=pred.device)
# # # # # # # # #         all_trajs = pred.unsqueeze(0)
# # # # # # # # #         return pred, me_mean, all_trajs


# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  Autoregressive variant  ── ST-Trans-AR (paper §3.4)
# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # class STTransAR(nn.Module):
# # # # # # # # #     """
# # # # # # # # #     Autoregressive variant of ST-Trans (baseline trong paper §3.4).
# # # # # # # # #     Cùng encoder architecture, nhưng decoder predict từng bước một.
# # # # # # # # #     Dùng để so sánh với non-AR trong ablation study.
# # # # # # # # #     """

# # # # # # # # #     def __init__(
# # # # # # # # #         self,
# # # # # # # # #         obs_len:    int   = 8,
# # # # # # # # #         pred_len:   int   = 12,
# # # # # # # # #         feat_dim:   int   = 8,
# # # # # # # # #         d_model:    int   = 64,
# # # # # # # # #         nhead:      int   = 4,
# # # # # # # # #         num_enc_layers: int = 1,
# # # # # # # # #         dim_ff:     int   = 512,
# # # # # # # # #         dropout:    float = 0.1,
# # # # # # # # #         lambda_speed: float = 0.1,
# # # # # # # # #         lambda_accel: float = 0.01,
# # # # # # # # #         w_mse:        float = 0.05,
# # # # # # # # #         v_max_kmh:    float = 80.0,
# # # # # # # # #         dt_h:         float = 6.0,
# # # # # # # # #     ):
# # # # # # # # #         super().__init__()
# # # # # # # # #         self.obs_len    = obs_len
# # # # # # # # #         self.pred_len   = pred_len
# # # # # # # # #         self.d_model    = d_model
# # # # # # # # #         self.lambda_speed = lambda_speed
# # # # # # # # #         self.lambda_accel = lambda_accel
# # # # # # # # #         self.w_mse        = w_mse
# # # # # # # # #         self.v_max_norm = v_max_kmh * dt_h / (111.0 * 50.0)

# # # # # # # # #         self.feat_extractor = ObsTrajFeatureExtractor(feat_dim)
# # # # # # # # #         self.cnn_enc        = CNNStateEncoder(feat_dim, d_model)
# # # # # # # # #         self.input_proj     = nn.Linear(d_model, d_model)
# # # # # # # # #         self.enc_pe         = SinusoidalPE(d_model, max_len=obs_len + 10)

# # # # # # # # #         enc_layer = nn.TransformerEncoderLayer(
# # # # # # # # #             d_model=d_model, nhead=nhead,
# # # # # # # # #             dim_feedforward=dim_ff, dropout=dropout,
# # # # # # # # #             activation="relu", batch_first=True,
# # # # # # # # #         )
# # # # # # # # #         self.transformer_enc = nn.TransformerEncoder(enc_layer,
# # # # # # # # #                                                       num_layers=num_enc_layers)

# # # # # # # # #         # AR GRU decoder (simpler than full transformer decoder for AR)
# # # # # # # # #         self.ar_gru   = nn.GRUCell(2 + d_model, d_model)
# # # # # # # # #         self.reg_head = nn.Sequential(
# # # # # # # # #             nn.Linear(d_model, d_model),
# # # # # # # # #             nn.ReLU(),
# # # # # # # # #             nn.Linear(d_model, 2),
# # # # # # # # #         )

# # # # # # # # #         self._init_weights()

# # # # # # # # #     def _init_weights(self):
# # # # # # # # #         for m in self.modules():
# # # # # # # # #             if isinstance(m, nn.Linear):
# # # # # # # # #                 nn.init.xavier_uniform_(m.weight, gain=0.5)
# # # # # # # # #                 if m.bias is not None:
# # # # # # # # #                     nn.init.zeros_(m.bias)

# # # # # # # # #     def forward(self, obs_traj: torch.Tensor,
# # # # # # # # #                 gt_traj: Optional[torch.Tensor] = None,
# # # # # # # # #                 teacher_forcing: bool = True) -> torch.Tensor:
# # # # # # # # #         B = obs_traj.shape[1]
# # # # # # # # #         feat   = self.feat_extractor(obs_traj)
# # # # # # # # #         h      = self.cnn_enc(feat)
# # # # # # # # #         h      = self.input_proj(h)
# # # # # # # # #         h      = self.enc_pe(h)
# # # # # # # # #         memory = self.transformer_enc(h)            # [B, S, d_model]
# # # # # # # # #         ctx    = memory.mean(dim=1)                  # [B, d_model]

# # # # # # # # #         cur_pos = obs_traj[-1].clone()
# # # # # # # # #         hx      = ctx
# # # # # # # # #         preds   = []

# # # # # # # # #         for i in range(self.pred_len):
# # # # # # # # #             inp  = torch.cat([cur_pos, ctx], dim=-1)
# # # # # # # # #             hx   = self.ar_gru(inp, hx)
# # # # # # # # #             out  = self.reg_head(hx)
# # # # # # # # #             preds.append(out)

# # # # # # # # #             if teacher_forcing and gt_traj is not None and i < gt_traj.shape[0]:
# # # # # # # # #                 cur_pos = gt_traj[i]
# # # # # # # # #             else:
# # # # # # # # #                 cur_pos = out.detach()

# # # # # # # # #         return torch.stack(preds, dim=0)   # [pred_len, B, 2]

# # # # # # # # #     def physics_loss(self, pred_norm, gt_norm):
# # # # # # # # #         T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # # #         pred = pred_norm[:T]; gt = gt_norm[:T]
# # # # # # # # #         pred_deg = _norm_to_deg(pred); gt_deg = _norm_to_deg(gt)
# # # # # # # # #         l_dpe   = haversine_km(pred_deg, gt_deg).mean()
# # # # # # # # #         l_mse   = F.mse_loss(pred, gt)
# # # # # # # # #         if T >= 2:
# # # # # # # # #             step_dist = (pred[1:] - pred[:-1]).norm(dim=-1)
# # # # # # # # #             l_speed = F.relu(step_dist - self.v_max_norm).pow(2).mean()
# # # # # # # # #         else:
# # # # # # # # #             l_speed = pred_norm.new_zeros(())
# # # # # # # # #         if T >= 3:
# # # # # # # # #             vel = pred[1:] - pred[:-1]
# # # # # # # # #             l_accel = (vel[1:].norm(dim=-1) - vel[:-1].norm(dim=-1)).pow(2).mean()
# # # # # # # # #         else:
# # # # # # # # #             l_accel = pred_norm.new_zeros(())
# # # # # # # # #         total = l_dpe + self.w_mse*l_mse + self.lambda_speed*l_speed + self.lambda_accel*l_accel
# # # # # # # # #         return dict(total=total, dpe=l_dpe.item(), mse=l_mse.item(),
# # # # # # # # #                     speed=l_speed.item(), accel=l_accel.item())

# # # # # # # # #     def get_loss(self, batch_list):
# # # # # # # # #         return self.get_loss_breakdown(batch_list)["total"]

# # # # # # # # #     def get_loss_breakdown(self, batch_list):
# # # # # # # # #         obs_traj = batch_list[0]; traj_gt = batch_list[1]
# # # # # # # # #         pred = self.forward(obs_traj, traj_gt, teacher_forcing=True)
# # # # # # # # #         return self.physics_loss(pred, traj_gt)

# # # # # # # # #     @torch.no_grad()
# # # # # # # # #     def sample(self, batch_list, **kwargs):
# # # # # # # # #         obs_traj = batch_list[0]
# # # # # # # # #         pred = self.forward(obs_traj, teacher_forcing=False)
# # # # # # # # #         T, B, _ = pred.shape
# # # # # # # # #         me_mean = torch.zeros(T, B, 2, device=pred.device)
# # # # # # # # #         return pred, me_mean, pred.unsqueeze(0)


# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # # #  ADE metrics  (cùng format với bài của bạn)
# # # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # # def compute_ade_per_horizon(
# # # # # # # # #     pred_norm: torch.Tensor,
# # # # # # # # #     gt_norm:   torch.Tensor,
# # # # # # # # # ) -> Dict[str, float]:
# # # # # # # # #     T = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # # #     pred_deg = _norm_to_deg(pred_norm[:T])
# # # # # # # # #     gt_deg   = _norm_to_deg(gt_norm[:T])
# # # # # # # # #     dist = haversine_km(pred_deg, gt_deg)
# # # # # # # # #     result = dict(ADE=float(dist.mean()), FDE=float(dist[-1].mean()))
# # # # # # # # #     for h, s in HORIZON_STEPS.items():
# # # # # # # # #         result[f"{h}h"] = float(dist[s].mean()) if s < T else float("nan")
# # # # # # # # #     return result

# # # # # # # # """
# # # # # # # # Model/st_trans_model.py  ── ST-Trans Baseline
# # # # # # # # ===============================================
# # # # # # # # THUẬT TOÁN: Faiaz et al. (2026)
# # # # # # # # "Physics-guided non-autoregressive transformer for lightweight
# # # # # # # # cyclone track prediction in the Bay of Bengal"
# # # # # # # # Expert Systems With Applications 317 (2026) 131972

# # # # # # # # THAY ĐỔI SO VỚI PHIÊN BẢN TRƯỚC:
# # # # # # # #   ✅ Dùng cùng PaperEncoder (FNO3D + Mamba + Env_net) với paper baseline
# # # # # # # #      thay vì chỉ encode obs_traj qua CNN đơn giản.
# # # # # # # #   ✅ Thêm ATE / CTE metrics (Along-Track / Cross-Track Error)
# # # # # # # #   ✅ Interface nhất quán: forward(batch_list) thay vì forward(obs_traj)

# # # # # # # # KIẾN TRÚC MỚI:
# # # # # # # #   PaperEncoder(batch_list)   → raw_ctx [B, 512]          (context phong phú)
# # # # # # # #   obs_traj features          → obs_memory [B, S, d_model] (temporal structure)
# # # # # # # #   full_memory = cat([raw_ctx_token, obs_memory], dim=1)   [B, S+1, d_model]
# # # # # # # #   Transformer decoder (learned horizon queries) → [B, H, d_model]
# # # # # # # #   Regression head  → [H, B, 2]

# # # # # # # # LOSS (§3.5.1 - Physics-guided composite):
# # # # # # # #   L = L_DPE + 0.05*L_MSE + λ_speed*L_speed + λ_accel*L_accel
# # # # # # # # """
# # # # # # # # from __future__ import annotations

# # # # # # # # import math
# # # # # # # # from typing import Dict, Optional, Tuple

# # # # # # # # import torch
# # # # # # # # import torch.nn as nn
# # # # # # # # import torch.nn.functional as F

# # # # # # # # # ── Import shared encoder và metric helpers ──────────────────────────────────
# # # # # # # # from Model.paper_baseline_model import (
# # # # # # # #     PaperEncoder,
# # # # # # # #     _norm_to_deg,
# # # # # # # #     _ate_cte_tensors,
# # # # # # # #     haversine_km,
# # # # # # # #     compute_ade_per_horizon,
# # # # # # # #     compute_ate_cte_per_horizon,
# # # # # # # #     compute_full_metrics,
# # # # # # # #     HORIZON_STEPS,
# # # # # # # # )


# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Sinusoidal Positional Encoding
# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # class SinusoidalPE(nn.Module):
# # # # # # # #     def __init__(self, d_model: int, max_len: int = 300):
# # # # # # # #         super().__init__()
# # # # # # # #         pe  = torch.zeros(max_len, d_model)
# # # # # # # #         pos = torch.arange(max_len).unsqueeze(1).float()
# # # # # # # #         div = torch.exp(torch.arange(0, d_model, 2).float() *
# # # # # # # #                         (-math.log(10000.0) / d_model))
# # # # # # # #         pe[:, 0::2] = torch.sin(pos * div)
# # # # # # # #         pe[:, 1::2] = torch.cos(pos * div)
# # # # # # # #         self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

# # # # # # # #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# # # # # # # #         return x + self.pe[:, :x.size(1), :]


# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Lightweight obs-traj encoder  (thay CNNStateEncoder cũ)
# # # # # # # # #  8D kinematic features → [B, S, d_model] sequence memory
# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # class ObsKinematicEncoder(nn.Module):
# # # # # # # #     """
# # # # # # # #     Encode obs_traj [T_obs, B, 2] → sequence memory [B, T_obs, d_model].

# # # # # # # #     Tính 8 kinematic features (position, velocity, acceleration, speed, step-idx)
# # # # # # # #     rồi project qua MLP → transformer encoder để capture temporal dependencies.
# # # # # # # #     """

# # # # # # # #     FEAT_DIM = 8

# # # # # # # #     def __init__(self, d_model: int = 64, nhead: int = 4,
# # # # # # # #                  num_layers: int = 1, dim_ff: int = 256, dropout: float = 0.1):
# # # # # # # #         super().__init__()
# # # # # # # #         # 8 → d_model projection
# # # # # # # #         self.proj = nn.Sequential(
# # # # # # # #             nn.Linear(self.FEAT_DIM, d_model),
# # # # # # # #             nn.ReLU(),
# # # # # # # #             nn.Linear(d_model, d_model),
# # # # # # # #         )
# # # # # # # #         self.pe = SinusoidalPE(d_model, max_len=64)
# # # # # # # #         enc_layer = nn.TransformerEncoderLayer(
# # # # # # # #             d_model=d_model, nhead=nhead,
# # # # # # # #             dim_feedforward=dim_ff, dropout=dropout,
# # # # # # # #             activation="relu", batch_first=True,
# # # # # # # #         )
# # # # # # # #         self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

# # # # # # # #     @staticmethod
# # # # # # # #     def _extract_features(obs_traj: torch.Tensor) -> torch.Tensor:
# # # # # # # #         """obs_traj [T, B, 2] → features [B, T, 8]."""
# # # # # # # #         T, B, _ = obs_traj.shape
# # # # # # # #         device  = obs_traj.device

# # # # # # # #         lon = obs_traj[:, :, 0]
# # # # # # # #         lat = obs_traj[:, :, 1]

# # # # # # # #         if T >= 2:
# # # # # # # #             d_lon = torch.cat([obs_traj[1:, :, 0] - obs_traj[:-1, :, 0],
# # # # # # # #                                torch.zeros(1, B, device=device)], dim=0)
# # # # # # # #             d_lat = torch.cat([obs_traj[1:, :, 1] - obs_traj[:-1, :, 1],
# # # # # # # #                                torch.zeros(1, B, device=device)], dim=0)
# # # # # # # #         else:
# # # # # # # #             d_lon = torch.zeros(T, B, device=device)
# # # # # # # #             d_lat = torch.zeros(T, B, device=device)

# # # # # # # #         if T >= 3:
# # # # # # # #             dd_lon = torch.cat([d_lon[1:] - d_lon[:-1],
# # # # # # # #                                 torch.zeros(1, B, device=device)], dim=0)
# # # # # # # #             dd_lat = torch.cat([d_lat[1:] - d_lat[:-1],
# # # # # # # #                                 torch.zeros(1, B, device=device)], dim=0)
# # # # # # # #         else:
# # # # # # # #             dd_lon = torch.zeros(T, B, device=device)
# # # # # # # #             dd_lat = torch.zeros(T, B, device=device)

# # # # # # # #         step_idx = torch.linspace(0, 1, T, device=device).unsqueeze(1).expand(T, B)
# # # # # # # #         speed    = (d_lon.pow(2) + d_lat.pow(2)).sqrt()

# # # # # # # #         feat = torch.stack([lon, lat, d_lon, d_lat,
# # # # # # # #                             dd_lon, dd_lat, step_idx, speed], dim=-1)
# # # # # # # #         return feat.permute(1, 0, 2)   # [B, T, 8]

# # # # # # # #     def forward(self, obs_traj: torch.Tensor) -> torch.Tensor:
# # # # # # # #         """→ [B, T_obs, d_model]"""
# # # # # # # #         feat = self._extract_features(obs_traj)   # [B, T, 8]
# # # # # # # #         h    = self.proj(feat)                     # [B, T, d_model]
# # # # # # # #         h    = self.pe(h)
# # # # # # # #         return self.enc(h)                         # [B, T, d_model]


# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  ST-Trans Main Model
# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # class STTrans(nn.Module):
# # # # # # # #     """
# # # # # # # #     Physics-guided Non-Autoregressive Transformer for TC track prediction.

# # # # # # # #     Kiến trúc (sau khi sửa để dùng cùng encoder với PaperBaseline):
# # # # # # # #       1. PaperEncoder(batch_list)  → raw_ctx [B, 512]
# # # # # # # #       2. Project raw_ctx → ctx_token [B, 1, d_model]  (global context token)
# # # # # # # #       3. ObsKinematicEncoder(obs_traj) → obs_memory [B, S, d_model]
# # # # # # # #       4. full_memory = cat([ctx_token, obs_memory], dim=1)  [B, S+1, d_model]
# # # # # # # #       5. Learned horizon queries [B, H, d_model] + dec_pe
# # # # # # # #       6. Transformer decoder (cross-attention) → [B, H, d_model]
# # # # # # # #       7. Regression head → [H, B, 2]

# # # # # # # #     Loss: Physics-guided composite (§3.5.1)
# # # # # # # #       L = L_DPE + 0.05*L_MSE + 0.1*L_speed + 0.01*L_accel
# # # # # # # #     """

# # # # # # # #     def __init__(
# # # # # # # #         self,
# # # # # # # #         obs_len:        int   = 8,
# # # # # # # #         pred_len:       int   = 12,
# # # # # # # #         unet_in_ch:     int   = 13,
# # # # # # # #         d_model:        int   = 64,
# # # # # # # #         nhead:          int   = 4,
# # # # # # # #         num_enc_layers: int   = 1,
# # # # # # # #         num_dec_layers: int   = 3,
# # # # # # # #         dim_ff:         int   = 512,
# # # # # # # #         dropout:        float = 0.1,
# # # # # # # #         # Physics loss weights
# # # # # # # #         lambda_speed:   float = 0.1,
# # # # # # # #         lambda_accel:   float = 0.01,
# # # # # # # #         w_mse:          float = 0.05,
# # # # # # # #         v_max_kmh:      float = 80.0,
# # # # # # # #         dt_h:           float = 6.0,
# # # # # # # #     ):
# # # # # # # #         super().__init__()
# # # # # # # #         self.obs_len      = obs_len
# # # # # # # #         self.pred_len     = pred_len
# # # # # # # #         self.d_model      = d_model
# # # # # # # #         self.lambda_speed = lambda_speed
# # # # # # # #         self.lambda_accel = lambda_accel
# # # # # # # #         self.w_mse        = w_mse
# # # # # # # #         # v_max trong normalised units
# # # # # # # #         self.v_max_norm   = v_max_kmh * dt_h / (111.0 * 50.0)

# # # # # # # #         # ── Shared encoder (cùng với PaperBaseline) ───────────────────────
# # # # # # # #         self.encoder = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)

# # # # # # # #         # Project raw_ctx [B, 512] → context token [B, 1, d_model]
# # # # # # # #         self.ctx_proj = nn.Sequential(
# # # # # # # #             nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
# # # # # # # #             nn.LayerNorm(d_model),
# # # # # # # #         )

# # # # # # # #         # ── Obs trajectory kinematic encoder → temporal memory ────────────
# # # # # # # #         self.obs_enc = ObsKinematicEncoder(
# # # # # # # #             d_model=d_model, nhead=nhead,
# # # # # # # #             num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout,
# # # # # # # #         )

# # # # # # # #         # ── Learned horizon queries (paper §3.3.4) ────────────────────────
# # # # # # # #         self.horizon_queries = nn.Parameter(
# # # # # # # #             torch.randn(1, pred_len, d_model) * 0.02
# # # # # # # #         )
# # # # # # # #         self.dec_pe = SinusoidalPE(d_model, max_len=pred_len + 10)

# # # # # # # #         # ── Transformer decoder ───────────────────────────────────────────
# # # # # # # #         dec_layer = nn.TransformerDecoderLayer(
# # # # # # # #             d_model=d_model, nhead=nhead,
# # # # # # # #             dim_feedforward=dim_ff, dropout=dropout,
# # # # # # # #             activation="relu", batch_first=True,
# # # # # # # #         )
# # # # # # # #         self.transformer_dec = nn.TransformerDecoder(
# # # # # # # #             dec_layer, num_layers=num_dec_layers)

# # # # # # # #         # ── Regression head (paper §3.3.4) ───────────────────────────────
# # # # # # # #         self.reg_head = nn.Sequential(
# # # # # # # #             nn.Linear(d_model, d_model),
# # # # # # # #             nn.ReLU(),
# # # # # # # #             nn.Linear(d_model, 2),
# # # # # # # #         )

# # # # # # # #         self._init_weights()

# # # # # # # #     def _init_weights(self):
# # # # # # # #         for m in self.modules():
# # # # # # # #             if isinstance(m, nn.Linear):
# # # # # # # #                 nn.init.xavier_uniform_(m.weight, gain=0.5)
# # # # # # # #                 if m.bias is not None:
# # # # # # # #                     nn.init.zeros_(m.bias)

# # # # # # # #     # ── Forward ───────────────────────────────────────────────────────────

# # # # # # # #     def forward(self, batch_list) -> torch.Tensor:
# # # # # # # #         """
# # # # # # # #         batch_list: cùng format với PaperBaseline (full batch với ảnh, env, ...)
# # # # # # # #         → pred [pred_len, B, 2] normalised
# # # # # # # #         """
# # # # # # # #         obs_traj = batch_list[0]     # [T_obs, B, 2]
# # # # # # # #         B        = obs_traj.shape[1]

# # # # # # # #         # 1. Rich context từ full encoder
# # # # # # # #         raw_ctx    = self.encoder(batch_list)          # [B, 512]
# # # # # # # #         ctx_token  = self.ctx_proj(raw_ctx).unsqueeze(1)  # [B, 1, d_model]

# # # # # # # #         # 2. Temporal obs memory
# # # # # # # #         obs_memory = self.obs_enc(obs_traj)            # [B, S, d_model]

# # # # # # # #         # 3. Kết hợp context token + temporal memory
# # # # # # # #         full_memory = torch.cat([ctx_token, obs_memory], dim=1)  # [B, S+1, d_model]

# # # # # # # #         # 4. Horizon queries
# # # # # # # #         Q = self.horizon_queries.expand(B, -1, -1)    # [B, H, d_model]
# # # # # # # #         Q = self.dec_pe(Q)

# # # # # # # #         # 5. Non-autoregressive decoder
# # # # # # # #         D   = self.transformer_dec(Q, full_memory)     # [B, H, d_model]
# # # # # # # #         out = self.reg_head(D)                         # [B, H, 2]

# # # # # # # #         return out.permute(1, 0, 2)                    # [pred_len, B, 2]

# # # # # # # #     # ── Physics-guided loss (§3.5.1) ─────────────────────────────────────

# # # # # # # #     def physics_loss(
# # # # # # # #         self,
# # # # # # # #         pred_norm: torch.Tensor,
# # # # # # # #         gt_norm:   torch.Tensor,
# # # # # # # #     ) -> Dict:
# # # # # # # #         T    = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # #         pred = pred_norm[:T]
# # # # # # # #         gt   = gt_norm[:T]

# # # # # # # #         pred_deg = _norm_to_deg(pred)
# # # # # # # #         gt_deg   = _norm_to_deg(gt)

# # # # # # # #         # L_DPE: mean great-circle distance (eq. 16)
# # # # # # # #         l_dpe = haversine_km(pred_deg, gt_deg).mean()

# # # # # # # #         # L_MSE: coordinate MSE (eq. 17)
# # # # # # # #         l_mse = F.mse_loss(pred, gt)

# # # # # # # #         # L_speed: penalise speeds > v_max (eq. 18-20)
# # # # # # # #         if T >= 2:
# # # # # # # #             step_dist = (pred[1:] - pred[:-1]).norm(dim=-1)      # [T-1, B]
# # # # # # # #             l_speed   = F.relu(step_dist - self.v_max_norm).pow(2).mean()
# # # # # # # #         else:
# # # # # # # #             l_speed = pred_norm.new_zeros(())

# # # # # # # #         # L_accel: penalise acceleration changes (eq. 21-22)
# # # # # # # #         if T >= 3:
# # # # # # # #             vel     = pred[1:] - pred[:-1]                        # [T-1, B, 2]
# # # # # # # #             l_accel = (vel[1:].norm(dim=-1) - vel[:-1].norm(dim=-1)).pow(2).mean()
# # # # # # # #         else:
# # # # # # # #             l_accel = pred_norm.new_zeros(())

# # # # # # # #         total = (l_dpe
# # # # # # # #                  + self.w_mse        * l_mse
# # # # # # # #                  + self.lambda_speed * l_speed
# # # # # # # #                  + self.lambda_accel * l_accel)

# # # # # # # #         return dict(
# # # # # # # #             total=total,
# # # # # # # #             dpe=l_dpe.item(),
# # # # # # # #             mse=l_mse.item(),
# # # # # # # #             speed=l_speed.item(),
# # # # # # # #             accel=l_accel.item(),
# # # # # # # #         )

# # # # # # # #     # ── Training / inference interface (nhất quán với PaperBaseline) ─────

# # # # # # # #     def get_loss(self, batch_list) -> torch.Tensor:
# # # # # # # #         return self.get_loss_breakdown(batch_list)["total"]

# # # # # # # #     def get_loss_breakdown(self, batch_list) -> Dict:
# # # # # # # #         traj_gt = batch_list[1]
# # # # # # # #         pred    = self.forward(batch_list)
# # # # # # # #         bd      = self.physics_loss(pred, traj_gt)

# # # # # # # #         with torch.no_grad():
# # # # # # # #             ade_m = compute_ade_per_horizon(pred.detach(), traj_gt)
# # # # # # # #             atc_m = compute_ate_cte_per_horizon(pred.detach(), traj_gt)

# # # # # # # #         bd.update(ade_m)
# # # # # # # #         bd.update(atc_m)
# # # # # # # #         return bd

# # # # # # # #     @torch.no_grad()
# # # # # # # #     def sample(
# # # # # # # #         self,
# # # # # # # #         batch_list,
# # # # # # # #         num_ensemble: int = 1,
# # # # # # # #         **kwargs,
# # # # # # # #     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# # # # # # # #         pred     = self.forward(batch_list)
# # # # # # # #         T, B, _  = pred.shape
# # # # # # # #         me_mean  = torch.zeros(T, B, 2, device=pred.device)
# # # # # # # #         return pred, me_mean, pred.unsqueeze(0)


# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  ST-Trans-AR  ── Autoregressive variant (paper §3.4)
# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # class STTransAR(nn.Module):
# # # # # # # #     """
# # # # # # # #     Autoregressive ST-Trans.
# # # # # # # #     Cùng encoder backbone với STTrans và PaperBaseline, decoder AR-GRU.
# # # # # # # #     Dùng để ablation so sánh với non-AR.
# # # # # # # #     """

# # # # # # # #     def __init__(
# # # # # # # #         self,
# # # # # # # #         obs_len:        int   = 8,
# # # # # # # #         pred_len:       int   = 12,
# # # # # # # #         unet_in_ch:     int   = 13,
# # # # # # # #         d_model:        int   = 64,
# # # # # # # #         nhead:          int   = 4,
# # # # # # # #         num_enc_layers: int   = 1,
# # # # # # # #         dim_ff:         int   = 512,
# # # # # # # #         dropout:        float = 0.1,
# # # # # # # #         lambda_speed:   float = 0.1,
# # # # # # # #         lambda_accel:   float = 0.01,
# # # # # # # #         w_mse:          float = 0.05,
# # # # # # # #         v_max_kmh:      float = 80.0,
# # # # # # # #         dt_h:           float = 6.0,
# # # # # # # #     ):
# # # # # # # #         super().__init__()
# # # # # # # #         self.obs_len      = obs_len
# # # # # # # #         self.pred_len     = pred_len
# # # # # # # #         self.d_model      = d_model
# # # # # # # #         self.lambda_speed = lambda_speed
# # # # # # # #         self.lambda_accel = lambda_accel
# # # # # # # #         self.w_mse        = w_mse
# # # # # # # #         self.v_max_norm   = v_max_kmh * dt_h / (111.0 * 50.0)

# # # # # # # #         # ── Shared encoder ────────────────────────────────────────────────
# # # # # # # #         self.encoder  = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
# # # # # # # #         self.ctx_proj = nn.Sequential(
# # # # # # # #             nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
# # # # # # # #             nn.LayerNorm(d_model),
# # # # # # # #         )

# # # # # # # #         self.obs_enc = ObsKinematicEncoder(
# # # # # # # #             d_model=d_model, nhead=nhead,
# # # # # # # #             num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout,
# # # # # # # #         )

# # # # # # # #         # ── AR-GRU decoder ────────────────────────────────────────────────
# # # # # # # #         # Input: cur_pos(2) + pooled_memory(d_model)
# # # # # # # #         self.ar_gru   = nn.GRUCell(2 + d_model, d_model)
# # # # # # # #         self.reg_head = nn.Sequential(
# # # # # # # #             nn.Linear(d_model, d_model),
# # # # # # # #             nn.ReLU(),
# # # # # # # #             nn.Linear(d_model, 2),
# # # # # # # #         )
# # # # # # # #         self._init_weights()

# # # # # # # #     def _init_weights(self):
# # # # # # # #         for m in self.modules():
# # # # # # # #             if isinstance(m, nn.Linear):
# # # # # # # #                 nn.init.xavier_uniform_(m.weight, gain=0.5)
# # # # # # # #                 if m.bias is not None:
# # # # # # # #                     nn.init.zeros_(m.bias)

# # # # # # # #     def forward(
# # # # # # # #         self,
# # # # # # # #         batch_list,
# # # # # # # #         gt_traj:         Optional[torch.Tensor] = None,
# # # # # # # #         teacher_forcing: bool = True,
# # # # # # # #     ) -> torch.Tensor:
# # # # # # # #         obs_traj = batch_list[0]
# # # # # # # #         B        = obs_traj.shape[1]

# # # # # # # #         raw_ctx    = self.encoder(batch_list)               # [B, 512]
# # # # # # # #         ctx_token  = self.ctx_proj(raw_ctx).unsqueeze(1)    # [B, 1, d_model]
# # # # # # # #         obs_memory = self.obs_enc(obs_traj)                 # [B, S, d_model]
# # # # # # # #         full_mem   = torch.cat([ctx_token, obs_memory], dim=1)  # [B, S+1, d_model]

# # # # # # # #         # Pooled context for AR decoder
# # # # # # # #         ctx = full_mem.mean(dim=1)     # [B, d_model]
# # # # # # # #         cur_pos = obs_traj[-1].clone()
# # # # # # # #         hx      = ctx
# # # # # # # #         preds   = []

# # # # # # # #         for i in range(self.pred_len):
# # # # # # # #             inp = torch.cat([cur_pos, ctx], dim=-1)
# # # # # # # #             hx  = self.ar_gru(inp, hx)
# # # # # # # #             out = self.reg_head(hx)
# # # # # # # #             preds.append(out)
# # # # # # # #             if teacher_forcing and gt_traj is not None and i < gt_traj.shape[0]:
# # # # # # # #                 cur_pos = gt_traj[i]
# # # # # # # #             else:
# # # # # # # #                 cur_pos = out.detach()

# # # # # # # #         return torch.stack(preds, dim=0)   # [pred_len, B, 2]

# # # # # # # #     def _physics_loss(self, pred_norm, gt_norm):
# # # # # # # #         T    = min(pred_norm.shape[0], gt_norm.shape[0])
# # # # # # # #         pred = pred_norm[:T]; gt = gt_norm[:T]
# # # # # # # #         pred_deg = _norm_to_deg(pred); gt_deg = _norm_to_deg(gt)
# # # # # # # #         l_dpe    = haversine_km(pred_deg, gt_deg).mean()
# # # # # # # #         l_mse    = F.mse_loss(pred, gt)
# # # # # # # #         if T >= 2:
# # # # # # # #             step_dist = (pred[1:] - pred[:-1]).norm(dim=-1)
# # # # # # # #             l_speed   = F.relu(step_dist - self.v_max_norm).pow(2).mean()
# # # # # # # #         else:
# # # # # # # #             l_speed   = pred_norm.new_zeros(())
# # # # # # # #         if T >= 3:
# # # # # # # #             vel     = pred[1:] - pred[:-1]
# # # # # # # #             l_accel = (vel[1:].norm(dim=-1) - vel[:-1].norm(dim=-1)).pow(2).mean()
# # # # # # # #         else:
# # # # # # # #             l_accel = pred_norm.new_zeros(())
# # # # # # # #         total = (l_dpe + self.w_mse * l_mse
# # # # # # # #                  + self.lambda_speed * l_speed + self.lambda_accel * l_accel)
# # # # # # # #         return dict(total=total, dpe=l_dpe.item(), mse=l_mse.item(),
# # # # # # # #                     speed=l_speed.item(), accel=l_accel.item())

# # # # # # # #     def get_loss(self, batch_list) -> torch.Tensor:
# # # # # # # #         return self.get_loss_breakdown(batch_list)["total"]

# # # # # # # #     def get_loss_breakdown(self, batch_list) -> Dict:
# # # # # # # #         traj_gt = batch_list[1]
# # # # # # # #         pred    = self.forward(batch_list, traj_gt, teacher_forcing=True)
# # # # # # # #         bd      = self._physics_loss(pred, traj_gt)
# # # # # # # #         with torch.no_grad():
# # # # # # # #             bd.update(compute_ade_per_horizon(pred.detach(), traj_gt))
# # # # # # # #             bd.update(compute_ate_cte_per_horizon(pred.detach(), traj_gt))
# # # # # # # #         return bd

# # # # # # # #     @torch.no_grad()
# # # # # # # #     def sample(self, batch_list, **kwargs):
# # # # # # # #         pred    = self.forward(batch_list, teacher_forcing=False)
# # # # # # # #         T, B, _ = pred.shape
# # # # # # # #         me_mean = torch.zeros(T, B, 2, device=pred.device)
# # # # # # # #         return pred, me_mean, pred.unsqueeze(0)
# # # # # # # """
# # # # # # # Model/st_trans_v2_model.py  ── ST-Trans v2 (Physics-Steering-Gate + Easy/Hard Split)
# # # # # # # ======================================================================================
# # # # # # # Mở rộng từ STTrans (Faiaz et al., 2026) với thiết kế loss hai nhánh:

# # # # # # #   EASY samples (bão thẳng, ~68% data):
# # # # # # #       Loss giữ nguyên 100% như ST-Trans gốc:
# # # # # # #         L_easy = L_DPE + 0.05*L_MSE + λ_speed*L_speed + λ_accel*L_accel

# # # # # # #   HARD samples (bão recurvature, ~32% data):
# # # # # # #       Loss mở rộng với 3 cải tiến:
# # # # # # #         L_hard = L_DPE_weighted   ← step-weighted, tập trung bước xa
# # # # # # #                + λ_speed*L_speed + λ_accel*L_accel
# # # # # # #                + w_heading*L_heading   ← cosine hướng, tổng quát
# # # # # # #                + w_recurv*L_recurv     ← auxiliary timing head
# # # # # # #                + w_gate_reg*L_gate_reg ← regularize gate không collapse

# # # # # # #   PHYSICS STEERING GATE (chỉ active với hard samples):
# # # # # # #       pred_final[t] = α[t]*pred_learned[t] + (1−α[t])*pred_physics[t]
# # # # # # #       α[t] = sigmoid(gate(ctx, steer_t, step_t))
# # # # # # #       Không có error accumulation. Không thay đổi inference structure.

# # # # # # # EASY/HARD CLASSIFICATION:
# # # # # # #   Từ obs_traj (8 bước quan sát), tính:
# # # # # # #     curvature_index = mean(|angle_change[t]|) trên T-2 bước
# # # # # # #     speed_variance  = std(step_speed) / (mean(step_speed) + eps)
# # # # # # #   hard nếu curvature_index > threshold_curv
# # # # # # #        hoặc speed_variance  > threshold_spd
# # # # # # #   Threshold tính 1 lần trên train set (p70) và truyền vào model.

# # # # # # # INTERFACE: Tương thích 100% với STTrans gốc.
# # # # # # #   forward(batch_list)        → [pred_len, B, 2]
# # # # # # #   get_loss(batch_list)       → scalar tensor
# # # # # # #   get_loss_breakdown(batch_list) → dict
# # # # # # #   sample(batch_list)         → (pred, me_mean, pred.unsqueeze(0))

# # # # # # # BATCH_LIST:
# # # # # # #   [0] obs_traj  [T_obs, B, 2]   normalised lon/lat
# # # # # # #   [1] traj_gt   [T_pred, B, 2]  normalised lon/lat
# # # # # # #   [2] data3d    [B, C, H, W]    ERA5 patches (13 channels, 81×81)
# # # # # # #                                  u500 @ channel _U500_CH, v500 @ _V500_CH
# # # # # # #   [3+] env_data ...              PaperEncoder handles all

# # # # # # # NOTE về units:
# # # # # # #   Normalised space: ±1 span 50°, tức 1 unit ≈ 5556 km tại equator
# # # # # # #   u500 ERA5 đơn vị m/s → convert qua _MS_TO_NORM_PER_6H
# # # # # # #   Nếu u500/v500 không ở channel 0/1, đổi _U500_CH và _V500_CH
# # # # # # # """
# # # # # # # from __future__ import annotations

# # # # # # # import math
# # # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # # import torch
# # # # # # # import torch.nn as nn
# # # # # # # import torch.nn.functional as F

# # # # # # # from Model.paper_baseline_model import (
# # # # # # #     PaperEncoder,
# # # # # # #     _norm_to_deg,
# # # # # # #     haversine_km,
# # # # # # #     compute_ade_per_horizon,
# # # # # # #     compute_ate_cte_per_horizon,
# # # # # # #     HORIZON_STEPS,
# # # # # # # )

# # # # # # # # ── ERA5 channel config ───────────────────────────────────────────────────────
# # # # # # # _U500_CH = 0          # channel index của u500 trong data3d [B,C,H,W]
# # # # # # # _V500_CH = 1          # channel index của v500
# # # # # # # _DEG_SCALE = 25.0     # half-span: 1 norm unit = 25 degree  (50°/2)
# # # # # # # _MS_TO_NORM_PER_6H = (6.0 * 3600.0) / (111320.0 * _DEG_SCALE)
# # # # # # # # = (m/s) → (deg/6h) / 25 = normalised displacement per step


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Shared modules (copy từ STTrans gốc, không thay đổi)
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # class SinusoidalPE(nn.Module):
# # # # # # #     def __init__(self, d_model: int, max_len: int = 300):
# # # # # # #         super().__init__()
# # # # # # #         pe  = torch.zeros(max_len, d_model)
# # # # # # #         pos = torch.arange(max_len).unsqueeze(1).float()
# # # # # # #         div = torch.exp(torch.arange(0, d_model, 2).float() *
# # # # # # #                         (-math.log(10000.0) / d_model))
# # # # # # #         pe[:, 0::2] = torch.sin(pos * div)
# # # # # # #         pe[:, 1::2] = torch.cos(pos * div)
# # # # # # #         self.register_buffer("pe", pe.unsqueeze(0))

# # # # # # #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# # # # # # #         return x + self.pe[:, :x.size(1), :]


# # # # # # # class ObsKinematicEncoder(nn.Module):
# # # # # # #     """Encode obs_traj [T_obs,B,2] → [B,T,d_model]. Giống STTrans gốc."""
# # # # # # #     FEAT_DIM = 8

# # # # # # #     def __init__(self, d_model=64, nhead=4, num_layers=1,
# # # # # # #                  dim_ff=256, dropout=0.1):
# # # # # # #         super().__init__()
# # # # # # #         self.proj = nn.Sequential(
# # # # # # #             nn.Linear(self.FEAT_DIM, d_model), nn.ReLU(),
# # # # # # #             nn.Linear(d_model, d_model))
# # # # # # #         self.pe  = SinusoidalPE(d_model, max_len=64)
# # # # # # #         enc      = nn.TransformerEncoderLayer(
# # # # # # #             d_model=d_model, nhead=nhead,
# # # # # # #             dim_feedforward=dim_ff, dropout=dropout,
# # # # # # #             activation="relu", batch_first=True)
# # # # # # #         self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)

# # # # # # #     @staticmethod
# # # # # # #     def _feats(obs: torch.Tensor) -> torch.Tensor:
# # # # # # #         """[T,B,2] → [B,T,8]"""
# # # # # # #         T, B, _ = obs.shape
# # # # # # #         dev = obs.device
# # # # # # #         lon, lat = obs[:,:,0], obs[:,:,1]
# # # # # # #         if T >= 2:
# # # # # # #             dl = torch.cat([obs[1:,:,0]-obs[:-1,:,0], torch.zeros(1,B,device=dev)], 0)
# # # # # # #             dt = torch.cat([obs[1:,:,1]-obs[:-1,:,1], torch.zeros(1,B,device=dev)], 0)
# # # # # # #         else:
# # # # # # #             dl = dt = torch.zeros(T, B, device=dev)
# # # # # # #         if T >= 3:
# # # # # # #             ddl = torch.cat([dl[1:]-dl[:-1], torch.zeros(1,B,device=dev)], 0)
# # # # # # #             ddt = torch.cat([dt[1:]-dt[:-1], torch.zeros(1,B,device=dev)], 0)
# # # # # # #         else:
# # # # # # #             ddl = ddt = torch.zeros(T, B, device=dev)
# # # # # # #         si  = torch.linspace(0,1,T,device=dev).unsqueeze(1).expand(T,B)
# # # # # # #         spd = (dl**2 + dt**2).sqrt()
# # # # # # #         return torch.stack([lon,lat,dl,dt,ddl,ddt,si,spd],-1).permute(1,0,2)

# # # # # # #     def forward(self, obs: torch.Tensor) -> torch.Tensor:
# # # # # # #         h = self.proj(self._feats(obs))
# # # # # # #         return self.enc(self.pe(h))


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Easy/Hard classifier (từ obs_traj, không cần gt)
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # def classify_hard_obs(
# # # # # # #     obs_traj: torch.Tensor,
# # # # # # #     threshold_curv: float,
# # # # # # #     threshold_spd:  float,
# # # # # # # ) -> torch.Tensor:
# # # # # # #     """
# # # # # # #     Classify từng sample trong batch là easy(0) hay hard(1).

# # # # # # #     obs_traj: [T_obs, B, 2] normalised

# # # # # # #     Tiêu chí hard (hoặc):
# # # # # # #       1. curvature_index > threshold_curv
# # # # # # #          curvature_index = mean(|angle_change|) trên các bước obs
# # # # # # #          angle_change[t] = góc giữa velocity[t] và velocity[t-1] (degrees)

# # # # # # #       2. speed_cv > threshold_spd
# # # # # # #          speed_cv = std(step_speed) / (mean(step_speed) + 1e-6)
# # # # # # #          Coefficient of Variation của tốc độ

# # # # # # #     Trả về: [B] bool tensor (True = hard)

# # # # # # #     NOTE: Hàm này chạy với no_grad trong cả train và eval.
# # # # # # #     Không có gradient chảy qua đây.
# # # # # # #     """
# # # # # # #     T, B, _ = obs_traj.shape
# # # # # # #     device   = obs_traj.device

# # # # # # #     with torch.no_grad():
# # # # # # #         # Step velocities [T-1, B, 2]
# # # # # # #         if T < 2:
# # # # # # #             return torch.zeros(B, dtype=torch.bool, device=device)

# # # # # # #         vel = obs_traj[1:] - obs_traj[:-1]           # [T-1, B, 2]
# # # # # # #         spd = vel.norm(dim=-1)                         # [T-1, B]  speed per step

# # # # # # #         # Speed coefficient of variation [B]
# # # # # # #         spd_mean = spd.mean(0)                         # [B]
# # # # # # #         spd_std  = spd.std(0)                          # [B]
# # # # # # #         speed_cv = spd_std / (spd_mean + 1e-6)         # [B]

# # # # # # #         # Curvature index: mean angle change between consecutive velocity vectors
# # # # # # #         if T >= 3:
# # # # # # #             vel_n  = F.normalize(vel, dim=-1, eps=1e-8)  # [T-1, B, 2]
# # # # # # #             cos_a  = (vel_n[1:] * vel_n[:-1]).sum(-1)    # [T-2, B]
# # # # # # #             cos_a  = cos_a.clamp(-1.0, 1.0)
# # # # # # #             ang    = torch.acos(cos_a) * (180.0 / math.pi)  # [T-2, B] degrees
# # # # # # #             curv   = ang.mean(0)                             # [B]
# # # # # # #         else:
# # # # # # #             curv = torch.zeros(B, device=device)

# # # # # # #         # Hard nếu EITHER tiêu chí vượt ngưỡng
# # # # # # #         is_hard = (curv > threshold_curv) | (speed_cv > threshold_spd)

# # # # # # #     return is_hard  # [B] bool


# # # # # # # def compute_hard_thresholds(
# # # # # # #     train_loader,
# # # # # # #     device: torch.device,
# # # # # # #     percentile: float = 70.0,
# # # # # # # ) -> Tuple[float, float]:
# # # # # # #     """
# # # # # # #     Tính threshold curvature và speed_cv trên toàn train set.
# # # # # # #     Gọi 1 lần trước khi train, cache kết quả.

# # # # # # #     Returns: (threshold_curv, threshold_spd)
# # # # # # #     """
# # # # # # #     all_curv = []
# # # # # # #     all_spd  = []

# # # # # # #     with torch.no_grad():
# # # # # # #         for batch in train_loader:
# # # # # # #             obs = batch[0].to(device)   # [T_obs, B, 2]
# # # # # # #             T, B, _ = obs.shape

# # # # # # #             if T < 2:
# # # # # # #                 continue

# # # # # # #             vel = obs[1:] - obs[:-1]
# # # # # # #             spd = vel.norm(dim=-1)

# # # # # # #             spd_mean = spd.mean(0)
# # # # # # #             spd_std  = spd.std(0)
# # # # # # #             speed_cv = (spd_std / (spd_mean + 1e-6)).cpu().tolist()
# # # # # # #             all_spd.extend(speed_cv)

# # # # # # #             if T >= 3:
# # # # # # #                 vel_n = F.normalize(vel, dim=-1, eps=1e-8)
# # # # # # #                 cos_a = (vel_n[1:] * vel_n[:-1]).sum(-1).clamp(-1,1)
# # # # # # #                 ang   = torch.acos(cos_a) * (180.0 / math.pi)
# # # # # # #                 curv  = ang.mean(0).cpu().tolist()
# # # # # # #                 all_curv.extend(curv)

# # # # # # #     if not all_curv:
# # # # # # #         return 15.0, 0.5   # fallback defaults

# # # # # # #     import numpy as np
# # # # # # #     thr_c = float(np.percentile(all_curv, percentile))
# # # # # # #     thr_s = float(np.percentile(all_spd,  percentile))
# # # # # # #     return thr_c, thr_s


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Physics Steering Gate
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # class PhysicsSteeringGate(nn.Module):
# # # # # # #     """
# # # # # # #     Tính α[t] ∈ (0,1) cho từng bước của HARD samples.

# # # # # # #     pred_final[t] = α[t] * pred_learned[t] + (1−α[t]) * pred_physics[t]

# # # # # # #     α ≈ 1: tin model, α ≈ 0: tin physics (steering ERA5)

# # # # # # #     Khởi tạo bias = 2.0 → sigmoid(2) ≈ 0.88 → đầu tiên thiên về model.
# # # # # # #     Gate sẽ học giảm α khi nhận thấy recurvature pattern.

# # # # # # #     Input mỗi bước:
# # # # # # #       ctx:     [B_hard, d_model]  — pooled context của hard samples
# # # # # # #       steer:   [B_hard, 2]        — (u_norm, v_norm) tại current pos
# # # # # # #       step_t:  int                — index bước (0..pred_len-1)

# # # # # # #     Không có error accumulation: steer lấy từ ERA5 grid, không từ pred trước.
# # # # # # #     """
# # # # # # #     def __init__(self, ctx_dim: int = 64, steer_dim: int = 2,
# # # # # # #                  hidden: int = 32, pred_len: int = 12):
# # # # # # #         super().__init__()
# # # # # # #         self.step_emb = nn.Embedding(pred_len, 16)
# # # # # # #         self.gate_net = nn.Sequential(
# # # # # # #             nn.Linear(ctx_dim + 16 + steer_dim, hidden),
# # # # # # #             nn.ReLU(),
# # # # # # #             nn.Linear(hidden, 1),
# # # # # # #         )
# # # # # # #         # Bias init: sigmoid(2.0) ≈ 0.88 → thiên về learned prediction
# # # # # # #         nn.init.constant_(self.gate_net[-1].bias, 2.0)
# # # # # # #         nn.init.zeros_(self.gate_net[-1].weight)

# # # # # # #     def forward(self, ctx: torch.Tensor, steer: torch.Tensor,
# # # # # # #                 step_t: int) -> torch.Tensor:
# # # # # # #         """→ α: [B_hard, 1]"""
# # # # # # #         B    = ctx.shape[0]
# # # # # # #         dev  = ctx.device
# # # # # # #         s    = torch.tensor(step_t, dtype=torch.long, device=dev)
# # # # # # #         semb = self.step_emb(s).unsqueeze(0).expand(B, -1)
# # # # # # #         return torch.sigmoid(self.gate_net(torch.cat([ctx, semb, steer], -1)))


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Recurvature Timing Head (auxiliary cho hard samples)
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # class RecurvatureTimingHead(nn.Module):
# # # # # # #     """
# # # # # # #     Auxiliary classifier: predict bước recurvature (đổi hướng > threshold_deg).

# # # # # # #     Output logits [B_hard, pred_len+1]:
# # # # # # #       class 0     = không recurve
# # # # # # #       class k+1   = recurve tại step k (0-indexed)

# # # # # # #     Loss weight nhỏ (w_recurv=0.05) → chỉ là supervision phụ,
# # # # # # #     không ảnh hưởng đến loss chính.
# # # # # # #     """
# # # # # # #     def __init__(self, ctx_dim: int = 64, pred_len: int = 12, hidden: int = 64):
# # # # # # #         super().__init__()
# # # # # # #         self.pred_len = pred_len
# # # # # # #         self.clf = nn.Sequential(
# # # # # # #             nn.Linear(ctx_dim, hidden),
# # # # # # #             nn.GELU(),
# # # # # # #             nn.Dropout(0.1),
# # # # # # #             nn.Linear(hidden, pred_len + 1),
# # # # # # #         )

# # # # # # #     def forward(self, ctx: torch.Tensor) -> torch.Tensor:
# # # # # # #         """ctx [B,d_model] → logits [B, pred_len+1]"""
# # # # # # #         return self.clf(ctx)

# # # # # # #     @staticmethod
# # # # # # #     def make_label(gt_norm: torch.Tensor, obs_norm: torch.Tensor,
# # # # # # #                    threshold_deg: float = 45.0) -> torch.Tensor:
# # # # # # #         """
# # # # # # #         gt_norm:  [T_pred, B, 2]
# # # # # # #         obs_norm: [T_obs,  B, 2]
# # # # # # #         → label [B] long  (0=no recurve, k+1=recurve at step k)
# # # # # # #         """
# # # # # # #         T_pred, B, _ = gt_norm.shape
# # # # # # #         dev   = gt_norm.device
# # # # # # #         label = torch.zeros(B, dtype=torch.long, device=dev)

# # # # # # #         # Ghép 1 bước cuối obs với gt để tính angle tại step 0
# # # # # # #         prev  = obs_norm[-1:]          # [1, B, 2]
# # # # # # #         full  = torch.cat([prev, gt_norm], 0)   # [T_pred+1, B, 2]

# # # # # # #         with torch.no_grad():
# # # # # # #             for t in range(T_pred - 1):
# # # # # # #                 d_in  = full[t+1] - full[t]
# # # # # # #                 d_out = full[t+2] - full[t+1]
# # # # # # #                 n_in  = F.normalize(d_in,  dim=-1, eps=1e-8)
# # # # # # #                 n_out = F.normalize(d_out, dim=-1, eps=1e-8)
# # # # # # #                 cos   = (n_in * n_out).sum(-1).clamp(-1, 1)
# # # # # # #                 ang   = torch.acos(cos) * (180.0 / math.pi)
# # # # # # #                 mask  = (ang > threshold_deg) & (label == 0)
# # # # # # #                 label[mask] = t + 1   # 1-indexed, 0=no recurve

# # # # # # #         return label


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Helper: steering từ ERA5 data3d
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # def _get_steering(data3d: Optional[torch.Tensor],
# # # # # # #                   obs_traj: torch.Tensor) -> Optional[torch.Tensor]:
# # # # # # #     """
# # # # # # #     Trả về (u_norm, v_norm) tại tâm ERA5 patch [B, 2] hoặc None.

# # # # # # #     u_norm, v_norm: normalised displacement per step (cùng đơn vị obs_traj).
# # # # # # #     Dùng center pixel của patch (vị trí bão là tâm patch).
# # # # # # #     """
# # # # # # #     if data3d is None or not isinstance(data3d, torch.Tensor):
# # # # # # #         return None
# # # # # # #     if data3d.dim() != 4:
# # # # # # #         return None

# # # # # # #     B, C, H, W = data3d.shape
# # # # # # #     if C <= max(_U500_CH, _V500_CH):
# # # # # # #         return None   # không đủ channels

# # # # # # #     cy, cx = H // 2, W // 2
# # # # # # #     u_ms   = data3d[:, _U500_CH, cy, cx]   # [B] m/s
# # # # # # #     v_ms   = data3d[:, _V500_CH, cy, cx]   # [B] m/s

# # # # # # #     # Latitude correction cho u (zonal)
# # # # # # #     lat_norm = obs_traj[-1, :, 1]          # [B] normalised lat
# # # # # # #     lat_deg  = lat_norm * _DEG_SCALE        # rough degree
# # # # # # #     cos_lat  = torch.cos(lat_deg * (math.pi / 180.0)).clamp(0.1, 1.0)

# # # # # # #     u_norm = (u_ms / cos_lat) * _MS_TO_NORM_PER_6H   # [B]
# # # # # # #     v_norm = v_ms * _MS_TO_NORM_PER_6H               # [B]

# # # # # # #     return torch.stack([u_norm, v_norm], dim=-1)      # [B, 2]


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  STTransV2 — Main Model
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # class STTransV2(nn.Module):
# # # # # # #     """
# # # # # # #     ST-Trans v2: Easy/Hard split loss + Physics Steering Gate.

# # # # # # #     EASY loss = ST-Trans gốc (unchanged):
# # # # # # #         L_DPE + 0.05*L_MSE + λ_speed*L_speed + λ_accel*L_accel

# # # # # # #     HARD loss = Extended:
# # # # # # #         L_DPE_weighted + λ_speed*L_speed + λ_accel*L_accel
# # # # # # #         + w_heading * L_heading
# # # # # # #         + w_recurv  * L_recurv  (auxiliary)
# # # # # # #         + w_gate    * L_gate_reg

# # # # # # #     Tổng: L = w_easy_frac*L_easy(easy_batch) + w_hard_frac*L_hard(hard_batch)
# # # # # # #     w_easy_frac, w_hard_frac không phải hyperparameter — đơn giản là
# # # # # # #     average losses trên 2 subsets. Không cần tune.
# # # # # # #     """

# # # # # # #     def __init__(
# # # # # # #         self,
# # # # # # #         obs_len:           int   = 8,
# # # # # # #         pred_len:          int   = 12,
# # # # # # #         unet_in_ch:        int   = 13,
# # # # # # #         d_model:           int   = 64,
# # # # # # #         nhead:             int   = 4,
# # # # # # #         num_enc_layers:    int   = 1,
# # # # # # #         num_dec_layers:    int   = 3,
# # # # # # #         dim_ff:            int   = 512,
# # # # # # #         dropout:           float = 0.1,
# # # # # # #         # Easy loss weights (ST-Trans gốc)
# # # # # # #         lambda_speed:      float = 0.1,
# # # # # # #         lambda_accel:      float = 0.01,
# # # # # # #         w_mse:             float = 0.05,
# # # # # # #         v_max_kmh:         float = 80.0,
# # # # # # #         dt_h:              float = 6.0,
# # # # # # #         # Hard loss extras
# # # # # # #         w_heading:         float = 0.3,
# # # # # # #         w_recurv:          float = 0.05,
# # # # # # #         w_gate_reg:        float = 0.01,
# # # # # # #         step_weight_slope: float = 0.1,
# # # # # # #         recurv_threshold:  float = 45.0,
# # # # # # #         # Easy/hard thresholds (set after compute_hard_thresholds)
# # # # # # #         threshold_curv:    float = 15.0,
# # # # # # #         threshold_spd:     float = 0.5,
# # # # # # #         # Gate config
# # # # # # #         gate_hidden:       int   = 32,
# # # # # # #         recurv_hidden:     int   = 64,
# # # # # # #     ):
# # # # # # #         super().__init__()
# # # # # # #         self.obs_len          = obs_len
# # # # # # #         self.pred_len         = pred_len
# # # # # # #         self.d_model          = d_model
# # # # # # #         self.lambda_speed     = lambda_speed
# # # # # # #         self.lambda_accel     = lambda_accel
# # # # # # #         self.w_mse            = w_mse
# # # # # # #         self.v_max_norm       = v_max_kmh * dt_h / (111.0 * _DEG_SCALE)
# # # # # # #         self.w_heading        = w_heading
# # # # # # #         self.w_recurv         = w_recurv
# # # # # # #         self.w_gate_reg       = w_gate_reg
# # # # # # #         self.recurv_threshold = recurv_threshold
# # # # # # #         self.threshold_curv   = threshold_curv
# # # # # # #         self.threshold_spd    = threshold_spd

# # # # # # #         # ── Encoder (giống STTrans gốc hoàn toàn) ────────────────────────
# # # # # # #         self.encoder  = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
# # # # # # #         self.ctx_proj = nn.Sequential(
# # # # # # #             nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
# # # # # # #             nn.LayerNorm(d_model),
# # # # # # #         )
# # # # # # #         self.obs_enc = ObsKinematicEncoder(
# # # # # # #             d_model=d_model, nhead=nhead,
# # # # # # #             num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout)

# # # # # # #         # ── Decoder (giống STTrans gốc hoàn toàn) ────────────────────────
# # # # # # #         self.horizon_queries = nn.Parameter(
# # # # # # #             torch.randn(1, pred_len, d_model) * 0.02)
# # # # # # #         self.dec_pe = SinusoidalPE(d_model, max_len=pred_len + 10)
# # # # # # #         dec_layer = nn.TransformerDecoderLayer(
# # # # # # #             d_model=d_model, nhead=nhead,
# # # # # # #             dim_feedforward=dim_ff, dropout=dropout,
# # # # # # #             activation="relu", batch_first=True)
# # # # # # #         self.transformer_dec = nn.TransformerDecoder(dec_layer,
# # # # # # #                                                      num_layers=num_dec_layers)
# # # # # # #         self.reg_head = nn.Sequential(
# # # # # # #             nn.Linear(d_model, d_model), nn.ReLU(),
# # # # # # #             nn.Linear(d_model, 2))

# # # # # # #         # ── Hard-only modules ─────────────────────────────────────────────
# # # # # # #         self.steering_gate = PhysicsSteeringGate(
# # # # # # #             ctx_dim=d_model, steer_dim=2,
# # # # # # #             hidden=gate_hidden, pred_len=pred_len)

# # # # # # #         self.recurv_head = RecurvatureTimingHead(
# # # # # # #             ctx_dim=d_model, pred_len=pred_len, hidden=recurv_hidden)

# # # # # # #         # Learnable step weights cho hard L_DPE_weighted
# # # # # # #         # w_t = softplus(raw_w[t]), init: 1.0 + t*slope
# # # # # # #         init_w   = torch.tensor(
# # # # # # #             [1.0 + t * step_weight_slope for t in range(pred_len)])
# # # # # # #         init_raw = torch.log(torch.expm1(init_w.clamp(min=1e-3)))
# # # # # # #         self.raw_step_weights = nn.Parameter(init_raw)

# # # # # # #         self._init_weights()

# # # # # # #     def _init_weights(self):
# # # # # # #         for m in self.modules():
# # # # # # #             if isinstance(m, nn.Linear):
# # # # # # #                 nn.init.xavier_uniform_(m.weight, gain=0.5)
# # # # # # #                 if m.bias is not None:
# # # # # # #                     nn.init.zeros_(m.bias)
# # # # # # #         # Re-set gate bias sau _init_weights
# # # # # # #         nn.init.constant_(self.steering_gate.gate_net[-1].bias, 2.0)
# # # # # # #         nn.init.zeros_(self.steering_gate.gate_net[-1].weight)

# # # # # # #     @property
# # # # # # #     def step_weights(self) -> torch.Tensor:
# # # # # # #         """[pred_len] dương, không bao giờ âm."""
# # # # # # #         return F.softplus(self.raw_step_weights)

# # # # # # #     def set_thresholds(self, threshold_curv: float, threshold_spd: float):
# # # # # # #         """Cập nhật threshold sau khi tính từ train set."""
# # # # # # #         self.threshold_curv = threshold_curv
# # # # # # #         self.threshold_spd  = threshold_spd

# # # # # # #     # ── Encode: shared giữa easy và hard ─────────────────────────────────

# # # # # # #     def _encode(self, batch_list) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # # # #         """
# # # # # # #         Chạy encoder 1 lần, trả về:
# # # # # # #           learned_pred: [B, pred_len, 2]  — output của reg_head
# # # # # # #           ctx_pooled:   [B, d_model]      — pooled context cho gate/recurv
# # # # # # #         """
# # # # # # #         obs_traj = batch_list[0]
# # # # # # #         B        = obs_traj.shape[1]

# # # # # # #         raw_ctx     = self.encoder(batch_list)
# # # # # # #         ctx_token   = self.ctx_proj(raw_ctx).unsqueeze(1)   # [B,1,d]
# # # # # # #         obs_memory  = self.obs_enc(obs_traj)                 # [B,T,d]
# # # # # # #         full_memory = torch.cat([ctx_token, obs_memory], 1)  # [B,T+1,d]
# # # # # # #         ctx_pooled  = full_memory.mean(1)                    # [B,d]

# # # # # # #         Q   = self.horizon_queries.expand(B, -1, -1)
# # # # # # #         Q   = self.dec_pe(Q)
# # # # # # #         D   = self.transformer_dec(Q, full_memory)           # [B,H,d]
# # # # # # #         lp  = self.reg_head(D)                              # [B,H,2]
# # # # # # #         return lp, ctx_pooled

# # # # # # #     # ── Apply gate (chỉ dùng cho hard samples) ───────────────────────────

# # # # # # #     def _apply_gate(
# # # # # # #         self,
# # # # # # #         learned_pred: torch.Tensor,   # [B_h, pred_len, 2]
# # # # # # #         ctx_pooled:   torch.Tensor,   # [B_h, d_model]
# # # # # # #         steer:        torch.Tensor,   # [B_h, 2]  hoặc None
# # # # # # #         obs_traj:     torch.Tensor,   # [T_obs, B_h, 2]
# # # # # # #     ) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # # # #         """
# # # # # # #         Blend learned_pred với physics prediction bằng gate.
# # # # # # #         Returns:
# # # # # # #           final_pred: [B_h, pred_len, 2]
# # # # # # #           alpha_mean: [1] scalar — trung bình alpha để log
# # # # # # #         """
# # # # # # #         B_h = learned_pred.shape[0]
# # # # # # #         dev = learned_pred.device

# # # # # # #         if steer is None:
# # # # # # #             # Không có ERA5 → dùng cuối obs velocity làm physics
# # # # # # #             if obs_traj.shape[0] >= 2:
# # # # # # #                 steer = obs_traj[-1] - obs_traj[-2]   # [B_h, 2]
# # # # # # #             else:
# # # # # # #                 steer = torch.zeros(B_h, 2, device=dev)

# # # # # # #         # Physics trajectory: persistence of steering
# # # # # # #         physics_steps = []
# # # # # # #         cur = obs_traj[-1].clone()   # [B_h, 2]
# # # # # # #         for _ in range(self.pred_len):
# # # # # # #             cur = cur + steer
# # # # # # #             physics_steps.append(cur.clone())
# # # # # # #         physics = torch.stack(physics_steps, dim=1)  # [B_h, pred_len, 2]

# # # # # # #         # Gate per step
# # # # # # #         final_steps = []
# # # # # # #         alphas      = []
# # # # # # #         for t in range(self.pred_len):
# # # # # # #             alpha = self.steering_gate(ctx_pooled, steer, t)  # [B_h, 1]
# # # # # # #             blended = alpha * learned_pred[:, t] + (1.0 - alpha) * physics[:, t]
# # # # # # #             final_steps.append(blended)
# # # # # # #             alphas.append(alpha.mean())

# # # # # # #         final_pred = torch.stack(final_steps, dim=1)        # [B_h, pred_len, 2]
# # # # # # #         alpha_mean = torch.stack(alphas).mean()
# # # # # # #         return final_pred, alpha_mean

# # # # # # #     # ── Forward (full batch, không phân easy/hard) ────────────────────────

# # # # # # #     def forward(self, batch_list) -> torch.Tensor:
# # # # # # #         """
# # # # # # #         Inference: không có easy/hard split, không dùng gate.
# # # # # # #         Gate chỉ active trong training loss để tránh inference overhead.
# # # # # # #         → pred [pred_len, B, 2] normalised

# # # # # # #         Lý do không dùng gate trong inference:
# # # # # # #         Gate nhỏ và thêm noise nếu không cần. Sau khi train,
# # # # # # #         learned_pred đã học được kiến thức của gate vì gate
# # # # # # #         backprop qua learned_pred trong hard loss.
# # # # # # #         """
# # # # # # #         lp, _ = self._encode(batch_list)         # [B, pred_len, 2]
# # # # # # #         return lp.permute(1, 0, 2)               # [pred_len, B, 2]

# # # # # # #     # ── Loss easy: giữ nguyên 100% ST-Trans gốc ──────────────────────────

# # # # # # #     def _loss_easy(self, pred: torch.Tensor, gt: torch.Tensor) -> Dict:
# # # # # # #         """
# # # # # # #         pred, gt: [T, B_easy, 2] normalised — chỉ easy samples
# # # # # # #         Công thức: L_DPE + 0.05*L_MSE + λ_speed*L_speed + λ_accel*L_accel
# # # # # # #         Không thêm gì — giữ nguyên ST-Trans gốc.
# # # # # # #         """
# # # # # # #         T  = min(pred.shape[0], gt.shape[0])
# # # # # # #         p  = pred[:T]
# # # # # # #         g  = gt[:T]

# # # # # # #         pd = _norm_to_deg(p)
# # # # # # #         gd = _norm_to_deg(g)

# # # # # # #         l_dpe  = haversine_km(pd, gd).mean()
# # # # # # #         l_mse  = F.mse_loss(p, g)

# # # # # # #         if T >= 2:
# # # # # # #             sd      = (p[1:] - p[:-1]).norm(dim=-1)
# # # # # # #             l_speed = F.relu(sd - self.v_max_norm).pow(2).mean()
# # # # # # #         else:
# # # # # # #             l_speed = p.new_zeros(())

# # # # # # #         if T >= 3:
# # # # # # #             v       = p[1:] - p[:-1]
# # # # # # #             l_accel = (v[1:].norm(-1) - v[:-1].norm(-1)).pow(2).mean()
# # # # # # #         else:
# # # # # # #             l_accel = p.new_zeros(())

# # # # # # #         total = (l_dpe
# # # # # # #                  + self.w_mse        * l_mse
# # # # # # #                  + self.lambda_speed * l_speed
# # # # # # #                  + self.lambda_accel * l_accel)

# # # # # # #         return dict(
# # # # # # #             loss_easy=total,
# # # # # # #             easy_dpe=l_dpe.item(),
# # # # # # #             easy_mse=l_mse.item(),
# # # # # # #             easy_speed=l_speed.item(),
# # # # # # #             easy_accel=l_accel.item(),
# # # # # # #         )

# # # # # # #     # ── Loss hard: extended với gate + heading + recurv ───────────────────

# # # # # # #     def _loss_hard(
# # # # # # #         self,
# # # # # # #         learned_pred: torch.Tensor,   # [B_hard, pred_len, 2] — trước gate
# # # # # # #         final_pred:   torch.Tensor,   # [B_hard, pred_len, 2] — sau gate
# # # # # # #         gt:           torch.Tensor,   # [T_pred, B_hard, 2]
# # # # # # #         ctx_pooled:   torch.Tensor,   # [B_hard, d_model]
# # # # # # #         obs_norm:     torch.Tensor,   # [T_obs, B_hard, 2]
# # # # # # #         alpha_mean:   torch.Tensor,   # scalar
# # # # # # #     ) -> Dict:
# # # # # # #         """
# # # # # # #         Hard loss:
# # # # # # #           L_DPE_weighted  ← step-weighted haversine (tập trung bước xa)
# # # # # # #           L_speed         ← như gốc
# # # # # # #           L_accel         ← như gốc
# # # # # # #           L_heading       ← cosine direction similarity
# # # # # # #           L_recurv        ← auxiliary cross-entropy (timing classifier)
# # # # # # #           L_gate_reg      ← regularize alpha không về 0 hoặc 1 cực đoan
# # # # # # #         """
# # # # # # #         T  = min(final_pred.shape[1], gt.shape[0])
# # # # # # #         fp = final_pred[:, :T]          # [B_h, T, 2]  — sau gate
# # # # # # #         lp = learned_pred[:, :T]        # [B_h, T, 2]  — trước gate
# # # # # # #         g  = gt[:T].permute(1, 0, 2)   # [B_h, T, 2]

# # # # # # #         # Permute sang [T, B_h, 2] cho tất cả loss calculations
# # # # # # #         fp_perm_dpe = fp.permute(1,0,2)   # [T,B_h,2]
# # # # # # #         gt_perm     = g.permute(1,0,2)    # [T,B_h,2]
# # # # # # #         fpd = _norm_to_deg(fp_perm_dpe)
# # # # # # #         gd  = _norm_to_deg(gt_perm)

# # # # # # #         # ── L_DPE step-weighted ──────────────────────────────────────────
# # # # # # #         sw = self.step_weights[:T]               # [T]
# # # # # # #         sw = sw / sw.sum() * T                   # normalize
# # # # # # #         hav = haversine_km(fpd, gd)             # [T, B_h]
# # # # # # #         l_dpe_w = (hav * sw.unsqueeze(1)).mean()

# # # # # # #         # ── L_speed (trên final_pred) ─────────────────────────────────────
# # # # # # #         fp_perm = fp.permute(1,0,2)   # [T, B_h, 2]
# # # # # # #         if T >= 2:
# # # # # # #             sd       = (fp_perm[1:] - fp_perm[:-1]).norm(-1)
# # # # # # #             l_speed  = F.relu(sd - self.v_max_norm).pow(2).mean()
# # # # # # #         else:
# # # # # # #             l_speed  = fp.new_zeros(())

# # # # # # #         # ── L_accel (trên final_pred) ─────────────────────────────────────
# # # # # # #         if T >= 3:
# # # # # # #             vel     = fp_perm[1:] - fp_perm[:-1]
# # # # # # #             l_accel = (vel[1:].norm(-1) - vel[:-1].norm(-1)).pow(2).mean()
# # # # # # #         else:
# # # # # # #             l_accel = fp.new_zeros(())

# # # # # # #         # ── L_heading: cosine direction similarity ────────────────────────
# # # # # # #         # Không dùng gt_dir làm trục tham chiếu cố định (tránh overfit hướng)
# # # # # # #         # Chỉ so cosine giữa pred_dir và gt_dir → tổng quát mọi hướng
# # # # # # #         if T >= 2:
# # # # # # #             pred_dir = fp_perm[1:] - fp_perm[:-1]   # [T-1, B_h, 2]
# # # # # # #             gt_dir   = gt_perm[1:] - gt_perm[:-1]
# # # # # # #             pdn = F.normalize(pred_dir, dim=-1, eps=1e-8)
# # # # # # #             gdn = F.normalize(gt_dir,   dim=-1, eps=1e-8)
# # # # # # #             cos_sim   = (pdn * gdn).sum(-1).clamp(-1, 1)   # [T-1, B_h]
# # # # # # #             l_heading = ((1.0 - cos_sim) / 2.0).mean()     # in [0,1]
# # # # # # #         else:
# # # # # # #             l_heading = fp.new_zeros(())

# # # # # # #         # ── L_recurv: auxiliary timing prediction ─────────────────────────
# # # # # # #         recurv_logits = self.recurv_head(ctx_pooled)   # [B_h, pred_len+1]
# # # # # # #         try:
# # # # # # #             label = RecurvatureTimingHead.make_label(
# # # # # # #                 gt,  # [T_pred, B_h, 2] — đúng shape cho make_label
# # # # # # #                 obs_norm,  # [T_obs, B_h, 2] — đúng shape cho make_label
# # # # # # #                 threshold_deg=self.recurv_threshold,
# # # # # # #             )
# # # # # # #             # Đảm bảo label shape match
# # # # # # #             if label.shape[0] == recurv_logits.shape[0]:
# # # # # # #                 l_recurv = F.cross_entropy(recurv_logits, label)
# # # # # # #             else:
# # # # # # #                 l_recurv = fp.new_zeros(())
# # # # # # #         except Exception:
# # # # # # #             l_recurv = fp.new_zeros(())

# # # # # # #         # ── L_gate_reg: giữ alpha trong vùng [0.2, 0.95] ─────────────────
# # # # # # #         # Tránh gate collapse hoàn toàn về 0 hoặc 1
# # # # # # #         # alpha_mean là mean alpha qua tất cả steps và samples (từ _apply_gate)
# # # # # # #         l_gate_reg = (F.relu(alpha_mean - 0.95) +
# # # # # # #                       F.relu(0.2 - alpha_mean))

# # # # # # #         # ── Total hard loss ───────────────────────────────────────────────
# # # # # # #         total = (l_dpe_w
# # # # # # #                  + self.lambda_speed * l_speed
# # # # # # #                  + self.lambda_accel * l_accel
# # # # # # #                  + self.w_heading    * l_heading
# # # # # # #                  + self.w_recurv     * l_recurv
# # # # # # #                  + self.w_gate_reg   * l_gate_reg)

# # # # # # #         return dict(
# # # # # # #             loss_hard    = total,
# # # # # # #             hard_dpe     = l_dpe_w.item(),
# # # # # # #             hard_speed   = l_speed.item(),
# # # # # # #             hard_accel   = l_accel.item(),
# # # # # # #             hard_heading = l_heading.item()
# # # # # # #                            if isinstance(l_heading, torch.Tensor)
# # # # # # #                            else float(l_heading),
# # # # # # #             hard_recurv  = l_recurv.item()
# # # # # # #                            if isinstance(l_recurv, torch.Tensor)
# # # # # # #                            else float(l_recurv),
# # # # # # #             hard_gate    = l_gate_reg.item()
# # # # # # #                            if isinstance(l_gate_reg, torch.Tensor)
# # # # # # #                            else 0.0,
# # # # # # #             alpha_mean   = alpha_mean.item()
# # # # # # #                            if isinstance(alpha_mean, torch.Tensor)
# # # # # # #                            else float(alpha_mean),
# # # # # # #             step_w_72h   = self.step_weights[-1].item(),
# # # # # # #         )

# # # # # # #     # ── get_loss_breakdown: main training entry point ─────────────────────

# # # # # # #     def get_loss_breakdown(self, batch_list) -> Dict:
# # # # # # #         """
# # # # # # #         1. Classify easy/hard từ obs_traj
# # # # # # #         2. Encode toàn bộ batch 1 lần
# # # # # # #         3. Tính L_easy trên easy subset
# # # # # # #         4. Apply gate + tính L_hard trên hard subset
# # # # # # #         5. Trả về combined loss và detailed breakdown
# # # # # # #         """
# # # # # # #         obs_traj = batch_list[0]   # [T_obs, B, 2]
# # # # # # #         traj_gt  = batch_list[1]   # [T_pred, B, 2]
# # # # # # #         B        = obs_traj.shape[1]
# # # # # # #         device   = obs_traj.device

# # # # # # #         # ── 1. Classify ───────────────────────────────────────────────────
# # # # # # #         is_hard = classify_hard_obs(
# # # # # # #             obs_traj, self.threshold_curv, self.threshold_spd)   # [B] bool
# # # # # # #         is_easy = ~is_hard

# # # # # # #         n_easy = is_easy.sum().item()
# # # # # # #         n_hard = is_hard.sum().item()

# # # # # # #         # ── 2. Encode toàn bộ batch 1 lần ────────────────────────────────
# # # # # # #         learned_pred, ctx_pooled = self._encode(batch_list)
# # # # # # #         # learned_pred: [B, pred_len, 2]
# # # # # # #         # ctx_pooled:   [B, d_model]

# # # # # # #         # ── 3. Easy loss ──────────────────────────────────────────────────
# # # # # # #         loss_easy_total = None
# # # # # # #         easy_details    = {}

# # # # # # #         if n_easy > 0:
# # # # # # #             # Slice easy samples
# # # # # # #             lp_easy = learned_pred[is_easy]          # [B_e, pred_len, 2]
# # # # # # #             gt_easy = traj_gt[:, is_easy, :]         # [T, B_e, 2]

# # # # # # #             # Convert [B_e, pred_len, 2] → [pred_len, B_e, 2] cho loss
# # # # # # #             pred_easy_perm = lp_easy.permute(1, 0, 2)

# # # # # # #             easy_res = self._loss_easy(pred_easy_perm, gt_easy)
# # # # # # #             loss_easy_total = easy_res["loss_easy"]
# # # # # # #             easy_details    = easy_res
# # # # # # #         else:
# # # # # # #             # Không có easy samples trong batch — dùng hard loss làm total
# # # # # # #             loss_easy_total = learned_pred.new_zeros((), requires_grad=False)

# # # # # # #         # ── 4. Hard loss ──────────────────────────────────────────────────
# # # # # # #         loss_hard_total = None
# # # # # # #         hard_details    = {}

# # # # # # #         if n_hard > 0:
# # # # # # #             lp_hard  = learned_pred[is_hard]          # [B_h, pred_len, 2]
# # # # # # #             gt_hard  = traj_gt[:, is_hard, :]         # [T, B_h, 2]
# # # # # # #             obs_hard = obs_traj[:, is_hard, :]        # [T_obs, B_h, 2]
# # # # # # #             ctx_hard = ctx_pooled[is_hard]            # [B_h, d_model]

# # # # # # #             # Get steering từ ERA5
# # # # # # #             data3d = None
# # # # # # #             if len(batch_list) > 2 and isinstance(batch_list[2], torch.Tensor):
# # # # # # #                 d = batch_list[2]
# # # # # # #                 if d.dim() == 4:
# # # # # # #                     data3d = d[is_hard]               # [B_h, C, H, W]

# # # # # # #             steer = _get_steering(data3d, obs_hard)  # [B_h, 2] or None

# # # # # # #             # Apply gate → final prediction
# # # # # # #             final_hard, alpha_mean = self._apply_gate(
# # # # # # #                 lp_hard, ctx_hard, steer, obs_hard)  # [B_h, pred_len, 2]

# # # # # # #             hard_res = self._loss_hard(
# # # # # # #                 learned_pred = lp_hard,
# # # # # # #                 final_pred   = final_hard,
# # # # # # #                 gt           = gt_hard,
# # # # # # #                 ctx_pooled   = ctx_hard,
# # # # # # #                 obs_norm     = obs_hard,
# # # # # # #                 alpha_mean   = alpha_mean,
# # # # # # #             )
# # # # # # #             loss_hard_total = hard_res["loss_hard"]
# # # # # # #             hard_details    = hard_res
# # # # # # #         else:
# # # # # # #             loss_hard_total = learned_pred.new_zeros((), requires_grad=False)

# # # # # # #         # ── 5. Combine ────────────────────────────────────────────────────
# # # # # # #         # Đơn giản average 2 losses — không cần tune weight
# # # # # # #         # Nếu batch có cả easy và hard, contribution tự nhiên
# # # # # # #         # proportional với số lượng sample
# # # # # # #         if n_easy > 0 and n_hard > 0:
# # # # # # #             # Weighted average theo số samples
# # # # # # #             w_e = n_easy / B
# # # # # # #             w_h = n_hard / B
# # # # # # #             total = w_e * loss_easy_total + w_h * loss_hard_total
# # # # # # #         elif n_easy > 0:
# # # # # # #             total = loss_easy_total
# # # # # # #         else:
# # # # # # #             total = loss_hard_total

# # # # # # #         # ── 6. ADE metrics (no grad) ──────────────────────────────────────
# # # # # # #         # Dùng learned_pred (không có gate) để eval → inference-consistent
# # # # # # #         lp_perm = learned_pred.permute(1, 0, 2)   # [pred_len, B, 2]
# # # # # # #         with torch.no_grad():
# # # # # # #             ade_m = compute_ade_per_horizon(lp_perm.detach(), traj_gt)
# # # # # # #             atc_m = compute_ate_cte_per_horizon(lp_perm.detach(), traj_gt)

# # # # # # #         result = dict(
# # # # # # #             total   = total,
# # # # # # #             n_easy  = n_easy,
# # # # # # #             n_hard  = n_hard,
# # # # # # #             **easy_details,
# # # # # # #             **hard_details,
# # # # # # #             **ade_m,
# # # # # # #             **atc_m,
# # # # # # #         )
# # # # # # #         return result

# # # # # # #     def get_loss(self, batch_list) -> torch.Tensor:
# # # # # # #         return self.get_loss_breakdown(batch_list)["total"]

# # # # # # #     @torch.no_grad()
# # # # # # #     def sample(self, batch_list, num_ensemble: int = 1, **kwargs):
# # # # # # #         pred    = self.forward(batch_list)
# # # # # # #         T, B, _ = pred.shape
# # # # # # #         me_mean = torch.zeros(T, B, 2, device=pred.device)
# # # # # # #         return pred, me_mean, pred.unsqueeze(0)


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Factory
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # def build_st_trans_v2(args, threshold_curv=15.0, threshold_spd=0.5) -> STTransV2:
# # # # # # #     return STTransV2(
# # # # # # #         obs_len           = getattr(args, 'obs_len',           8),
# # # # # # #         pred_len          = getattr(args, 'pred_len',          12),
# # # # # # #         unet_in_ch        = getattr(args, 'unet_in_ch',        13),
# # # # # # #         d_model           = getattr(args, 'd_model',           64),
# # # # # # #         nhead             = getattr(args, 'nhead',             4),
# # # # # # #         num_enc_layers    = getattr(args, 'num_enc_layers',    1),
# # # # # # #         num_dec_layers    = getattr(args, 'num_dec_layers',    3),
# # # # # # #         dim_ff            = getattr(args, 'dim_ff',            512),
# # # # # # #         dropout           = getattr(args, 'dropout',           0.1),
# # # # # # #         lambda_speed      = getattr(args, 'lambda_speed',      0.1),
# # # # # # #         lambda_accel      = getattr(args, 'lambda_accel',      0.01),
# # # # # # #         w_mse             = getattr(args, 'w_mse',             0.05),
# # # # # # #         v_max_kmh         = getattr(args, 'v_max_kmh',         80.0),
# # # # # # #         w_heading         = getattr(args, 'w_heading',         0.3),
# # # # # # #         w_recurv          = getattr(args, 'w_recurv',          0.05),
# # # # # # #         w_gate_reg        = getattr(args, 'w_gate_reg',        0.01),
# # # # # # #         step_weight_slope = getattr(args, 'step_weight_slope', 0.1),
# # # # # # #         recurv_threshold  = getattr(args, 'recurv_threshold',  45.0),
# # # # # # #         threshold_curv    = threshold_curv,
# # # # # # #         threshold_spd     = threshold_spd,
# # # # # # #         gate_hidden       = getattr(args, 'gate_hidden',       32),
# # # # # # #         recurv_hidden     = getattr(args, 'recurv_hidden',     64),
# # # # # # #     )

# # # # # # """ Model/st_trans_v2_model.py  ── ST-Trans v2 (GradNorm + Easy/Hard Split)
# # # # # # =========================================================================
# # # # # # Mở rộng STTrans (Faiaz et al., 2026) với:

# # # # # #   [1] EASY/HARD LOSS SPLIT
# # # # # #       Easy (bão thẳng ~68%):  L_DPE + λ_mse*L_MSE + λ_speed*L_speed + λ_accel*L_accel
# # # # # #       Hard (recurvature ~32%): L_DPE_weighted + λ_speed*L_speed + λ_accel*L_accel
# # # # # #                                 + λ_heading*L_heading + λ_recurv*L_recurv

# # # # # #   [2] GRADNORM (Chen et al. 2018) — tất cả λ tự học
# # # # # #       Không hand-tune bất kỳ weight nào. GradNorm cân bằng gradient norms
# # # # # #       của từng loss term tự động trong quá trình training.

# # # # # #       λ_i(t) ← proportional đến r_i(t)^α × G_bar(t)
# # # # # #       r_i(t) = L_i(t) / L_i(0)  ← relative loss drop

# # # # # #       Loss terms được balance:
# # # # # #         EASY:   λ_mse, λ_speed, λ_accel
# # # # # #         HARD:   λ_heading, λ_recurv
# # # # # #         SHARED: λ_speed, λ_accel (cùng parameter)

# # # # # #       Anchor losses (không balance):
# # # # # #         L_DPE (easy), L_DPE_weighted (hard)

# # # # # #   [3] PHYSICS STEERING GATE (hard samples only)
# # # # # #       pred[t] = α[t]*pred_learned[t] + (1−α[t])*pred_physics[t]
# # # # # #       Không error accumulation. Gate không vào inference.

# # # # # #   [4] LEARNABLE STEP WEIGHTS
# # # # # #       w_t = softplus(raw_w[t]) cho L_DPE_weighted (hard)

# # # # # # INTERFACE: 100% tương thích STTrans gốc.
# # # # # #   forward(), get_loss(), get_loss_breakdown(), sample()

# # # # # # GRADNORM USAGE trong train script:
# # # # # #   bd = model.get_loss_breakdown(bl)
# # # # # #   loss = bd["total"]

# # # # # #   # Backward main loss (update model weights)
# # # # # #   optimizer.zero_grad()
# # # # # #   loss.backward(retain_graph=True)   # retain_graph cho GradNorm
# # # # # #   clip_grad_norm_(model.model_params(), grad_clip)
# # # # # #   optimizer.step()

# # # # # #   # Backward GradNorm loss (update λ weights)
# # # # # #   gn_loss = model.gradnorm_loss(bd)
# # # # # #   gn_optimizer.zero_grad()
# # # # # #   gn_loss.backward()
# # # # # #   gn_optimizer.step()
# # # # # #   model.renorm_lambdas()   # normalize λ về sum = n_tasks * λ_0
# # # # # # """
# # # # # # from __future__ import annotations

# # # # # # import math
# # # # # # from typing import Dict, List, Optional, Tuple

# # # # # # import torch
# # # # # # import torch.nn as nn
# # # # # # import torch.nn.functional as F

# # # # # # from Model.paper_baseline_model import (
# # # # # #     PaperEncoder,
# # # # # #     _norm_to_deg,
# # # # # #     haversine_km,
# # # # # #     compute_ade_per_horizon,
# # # # # #     compute_ate_cte_per_horizon,
# # # # # #     HORIZON_STEPS,
# # # # # # )

# # # # # # # ── ERA5 channel config ───────────────────────────────────────────────────────
# # # # # # _U500_CH   = 0          # channel index của u500 trong data3d [B,C,H,W]
# # # # # # _V500_CH   = 1          # channel index của v500
# # # # # # _DEG_SCALE = 25.0       # half-span: 1 norm unit = 25°  (50°/2)
# # # # # # _MS_TO_NORM_PER_6H = (6.0 * 3600.0) / (111320.0 * _DEG_SCALE)


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Shared modules (giống STTrans gốc)
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # class SinusoidalPE(nn.Module):
# # # # # #     def __init__(self, d_model: int, max_len: int = 300):
# # # # # #         super().__init__()
# # # # # #         pe  = torch.zeros(max_len, d_model)
# # # # # #         pos = torch.arange(max_len).unsqueeze(1).float()
# # # # # #         div = torch.exp(torch.arange(0, d_model, 2).float() *
# # # # # #                         (-math.log(10000.0) / d_model))
# # # # # #         pe[:, 0::2] = torch.sin(pos * div)
# # # # # #         pe[:, 1::2] = torch.cos(pos * div)
# # # # # #         self.register_buffer("pe", pe.unsqueeze(0))

# # # # # #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# # # # # #         return x + self.pe[:, :x.size(1), :]


# # # # # # class ObsKinematicEncoder(nn.Module):
# # # # # #     """obs_traj [T_obs,B,2] → [B,T,d_model]. Giống STTrans gốc."""
# # # # # #     FEAT_DIM = 8

# # # # # #     def __init__(self, d_model=64, nhead=4, num_layers=1,
# # # # # #                  dim_ff=256, dropout=0.1):
# # # # # #         super().__init__()
# # # # # #         self.proj = nn.Sequential(
# # # # # #             nn.Linear(self.FEAT_DIM, d_model), nn.ReLU(),
# # # # # #             nn.Linear(d_model, d_model))
# # # # # #         self.pe  = SinusoidalPE(d_model, max_len=64)
# # # # # #         enc      = nn.TransformerEncoderLayer(
# # # # # #             d_model=d_model, nhead=nhead,
# # # # # #             dim_feedforward=dim_ff, dropout=dropout,
# # # # # #             activation="relu", batch_first=True)
# # # # # #         self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)

# # # # # #     @staticmethod
# # # # # #     def _feats(obs: torch.Tensor) -> torch.Tensor:
# # # # # #         T, B, _ = obs.shape
# # # # # #         dev = obs.device
# # # # # #         lon, lat = obs[:,:,0], obs[:,:,1]
# # # # # #         if T >= 2:
# # # # # #             dl = torch.cat([obs[1:,:,0]-obs[:-1,:,0], torch.zeros(1,B,device=dev)], 0)
# # # # # #             dt = torch.cat([obs[1:,:,1]-obs[:-1,:,1], torch.zeros(1,B,device=dev)], 0)
# # # # # #         else:
# # # # # #             dl = dt = torch.zeros(T, B, device=dev)
# # # # # #         if T >= 3:
# # # # # #             ddl = torch.cat([dl[1:]-dl[:-1], torch.zeros(1,B,device=dev)], 0)
# # # # # #             ddt = torch.cat([dt[1:]-dt[:-1], torch.zeros(1,B,device=dev)], 0)
# # # # # #         else:
# # # # # #             ddl = ddt = torch.zeros(T, B, device=dev)
# # # # # #         si  = torch.linspace(0,1,T,device=dev).unsqueeze(1).expand(T,B)
# # # # # #         spd = (dl**2 + dt**2).sqrt()
# # # # # #         return torch.stack([lon,lat,dl,dt,ddl,ddt,si,spd],-1).permute(1,0,2)

# # # # # #     def forward(self, obs: torch.Tensor) -> torch.Tensor:
# # # # # #         return self.enc(self.pe(self.proj(self._feats(obs))))


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Easy/Hard classifier
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def classify_hard_obs(obs_traj: torch.Tensor,
# # # # # #                       threshold_curv: float,
# # # # # #                       threshold_spd: float) -> torch.Tensor:
# # # # # #     """
# # # # # #     obs_traj: [T_obs, B, 2] → [B] bool (True=hard)
# # # # # #     Hard nếu:
# # # # # #       curvature_index > threshold_curv  (mean angle change in degrees)
# # # # # #       OR speed_cv > threshold_spd       (std/mean của step speed)
# # # # # #     """
# # # # # #     T, B, _ = obs_traj.shape
# # # # # #     device   = obs_traj.device
# # # # # #     with torch.no_grad():
# # # # # #         if T < 2:
# # # # # #             return torch.zeros(B, dtype=torch.bool, device=device)
# # # # # #         vel      = obs_traj[1:] - obs_traj[:-1]       # [T-1,B,2]
# # # # # #         spd      = vel.norm(dim=-1)                     # [T-1,B]
# # # # # #         speed_cv = spd.std(0) / (spd.mean(0) + 1e-6)
# # # # # #         if T >= 3:
# # # # # #             vn   = F.normalize(vel, dim=-1, eps=1e-8)
# # # # # #             cos  = (vn[1:] * vn[:-1]).sum(-1).clamp(-1,1)
# # # # # #             curv = (torch.acos(cos) * (180.0/math.pi)).mean(0)
# # # # # #         else:
# # # # # #             curv = torch.zeros(B, device=device)
# # # # # #         return (curv > threshold_curv) | (speed_cv > threshold_spd)


# # # # # # def compute_hard_thresholds(train_loader, device: torch.device,
# # # # # #                              percentile: float = 70.0) -> Tuple[float, float]:
# # # # # #     """1 pass qua train set → (threshold_curv, threshold_spd)."""
# # # # # #     all_curv, all_spd = [], []
# # # # # #     with torch.no_grad():
# # # # # #         for batch in train_loader:
# # # # # #             obs = batch[0].to(device)
# # # # # #             T, B, _ = obs.shape
# # # # # #             if T < 2:
# # # # # #                 continue
# # # # # #             vel  = obs[1:] - obs[:-1]
# # # # # #             spd  = vel.norm(dim=-1)
# # # # # #             all_spd.extend((spd.std(0)/(spd.mean(0)+1e-6)).cpu().tolist())
# # # # # #             if T >= 3:
# # # # # #                 vn  = F.normalize(vel, dim=-1, eps=1e-8)
# # # # # #                 cos = (vn[1:]*vn[:-1]).sum(-1).clamp(-1,1)
# # # # # #                 all_curv.extend(
# # # # # #                     (torch.acos(cos)*(180/math.pi)).mean(0).cpu().tolist())
# # # # # #     if not all_curv:
# # # # # #         return 15.0, 0.5
# # # # # #     import numpy as np
# # # # # #     return (float(np.percentile(all_curv, percentile)),
# # # # # #             float(np.percentile(all_spd,  percentile)))


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Physics Steering Gate
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # class PhysicsSteeringGate(nn.Module):
# # # # # #     """
# # # # # #     α[t] = sigmoid(gate(ctx, steer, step_t)) ∈ (0,1)
# # # # # #     pred_final[t] = α[t]*pred_learned[t] + (1−α[t])*pred_physics[t]
# # # # # #     Khởi tạo bias=2.0 → α≈0.88 ban đầu (thiên về learned).
# # # # # #     """
# # # # # #     def __init__(self, ctx_dim=64, steer_dim=2, hidden=32, pred_len=12):
# # # # # #         super().__init__()
# # # # # #         self.step_emb = nn.Embedding(pred_len, 16)
# # # # # #         self.gate_net = nn.Sequential(
# # # # # #             nn.Linear(ctx_dim + 16 + steer_dim, hidden),
# # # # # #             nn.ReLU(),
# # # # # #             nn.Linear(hidden, 1))
# # # # # #         nn.init.constant_(self.gate_net[-1].bias, 2.0)
# # # # # #         nn.init.zeros_(self.gate_net[-1].weight)

# # # # # #     def forward(self, ctx: torch.Tensor, steer: torch.Tensor,
# # # # # #                 step_t: int) -> torch.Tensor:
# # # # # #         B   = ctx.shape[0]
# # # # # #         dev = ctx.device
# # # # # #         s   = torch.tensor(step_t, dtype=torch.long, device=dev)
# # # # # #         emb = self.step_emb(s).unsqueeze(0).expand(B, -1)
# # # # # #         return torch.sigmoid(self.gate_net(torch.cat([ctx, emb, steer], -1)))


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Recurvature Timing Head
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # class RecurvatureTimingHead(nn.Module):
# # # # # #     """
# # # # # #     Auxiliary: predict bước nào bão recurve.
# # # # # #     logits [B, pred_len+1]: class 0=no recurve, k+1=recurve at step k
# # # # # #     """
# # # # # #     def __init__(self, ctx_dim=64, pred_len=12, hidden=64):
# # # # # #         super().__init__()
# # # # # #         self.clf = nn.Sequential(
# # # # # #             nn.Linear(ctx_dim, hidden), nn.GELU(),
# # # # # #             nn.Dropout(0.1),
# # # # # #             nn.Linear(hidden, pred_len + 1))

# # # # # #     def forward(self, ctx: torch.Tensor) -> torch.Tensor:
# # # # # #         return self.clf(ctx)

# # # # # #     @staticmethod
# # # # # #     def make_label(gt_norm: torch.Tensor, obs_norm: torch.Tensor,
# # # # # #                    threshold_deg: float = 45.0) -> torch.Tensor:
# # # # # #         """
# # # # # #         gt_norm:  [T_pred, B, 2]
# # # # # #         obs_norm: [T_obs,  B, 2]
# # # # # #         → label [B] long
# # # # # #         """
# # # # # #         T_pred, B, _ = gt_norm.shape
# # # # # #         dev   = gt_norm.device
# # # # # #         label = torch.zeros(B, dtype=torch.long, device=dev)
# # # # # #         prev  = obs_norm[-1:]
# # # # # #         full  = torch.cat([prev, gt_norm], 0)   # [T_pred+1, B, 2]
# # # # # #         with torch.no_grad():
# # # # # #             for t in range(T_pred - 1):
# # # # # #                 d_in  = full[t+1] - full[t]
# # # # # #                 d_out = full[t+2] - full[t+1]
# # # # # #                 ni = F.normalize(d_in,  dim=-1, eps=1e-8)
# # # # # #                 no = F.normalize(d_out, dim=-1, eps=1e-8)
# # # # # #                 ang  = torch.acos((ni*no).sum(-1).clamp(-1,1)) * (180/math.pi)
# # # # # #                 mask = (ang > threshold_deg) & (label == 0)
# # # # # #                 label[mask] = t + 1
# # # # # #         return label


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Steering helper
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def _get_steering(data3d: Optional[torch.Tensor],
# # # # # #                   obs_traj: torch.Tensor) -> Optional[torch.Tensor]:
# # # # # #     """ERA5 center pixel → [B,2] normalised displacement/step. None nếu không có."""
# # # # # #     if data3d is None or not isinstance(data3d, torch.Tensor) or data3d.dim() != 4:
# # # # # #         return None
# # # # # #     B, C, H, W = data3d.shape
# # # # # #     if C <= max(_U500_CH, _V500_CH):
# # # # # #         return None
# # # # # #     cy, cx = H // 2, W // 2
# # # # # #     u_ms   = data3d[:, _U500_CH, cy, cx]
# # # # # #     v_ms   = data3d[:, _V500_CH, cy, cx]
# # # # # #     lat_deg = obs_traj[-1, :, 1] * _DEG_SCALE
# # # # # #     cos_lat = torch.cos(lat_deg * (math.pi/180.0)).clamp(0.1, 1.0)
# # # # # #     return torch.stack([
# # # # # #         (u_ms / cos_lat) * _MS_TO_NORM_PER_6H,
# # # # # #         v_ms * _MS_TO_NORM_PER_6H,
# # # # # #     ], dim=-1)


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  GradNorm Module
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # class GradNormController(nn.Module):
# # # # # #     """
# # # # # #     GradNorm (Chen et al. 2018) tự động cân bằng loss weights.

# # # # # #     Quản lý N learnable log-weights:
# # # # # #       λ_i = exp(log_λ_i)   (luôn dương)

# # # # # #     Loss terms được balance:
# # # # # #       easy: mse, speed, accel         (3 terms)
# # # # # #       hard: heading, recurv            (2 terms)
# # # # # #       shared: speed, accel có thể dùng chung param với easy
# # # # # #               → 5 terms độc lập

# # # # # #     Shared layer để tính gradient: reg_head cuối của model.

# # # # # #     Usage:
# # # # # #       # init
# # # # # #       gn = GradNormController(task_names=[...], alpha=1.5)

# # # # # #       # trong train loop:
# # # # # #       total, weighted_losses = gn.weighted_sum(raw_losses)
# # # # # #       total.backward(retain_graph=True)
# # # # # #       # update model params...

# # # # # #       gn_loss = gn.gradnorm_loss(raw_losses, shared_layer)
# # # # # #       gn_loss.backward()
# # # # # #       # update gn params...
# # # # # #       gn.renormalize()   # giữ sum(λ) = n_tasks
# # # # # #     """

# # # # # #     def __init__(self,
# # # # # #                  task_names: List[str],
# # # # # #                  alpha: float = 1.5,
# # # # # #                  init_val: float = 1.0):
# # # # # #         """
# # # # # #         task_names: danh sách tên loss terms cần balance
# # # # # #         alpha: GradNorm hyperparameter (1.0-2.0, thường 1.5)
# # # # # #         init_val: giá trị khởi đầu cho mỗi λ
# # # # # #         """
# # # # # #         super().__init__()
# # # # # #         self.task_names = task_names
# # # # # #         self.n_tasks    = len(task_names)
# # # # # #         self.alpha      = alpha
# # # # # #         # Log-space parameters: λ_i = softplus(raw_i) để luôn dương
# # # # # #         # Dùng softplus thay exp để tránh exploding
# # # # # #         init_raw = torch.tensor(
# # # # # #             [math.log(math.exp(init_val) - 1.0)] * self.n_tasks)
# # # # # #         self.raw_lambdas = nn.Parameter(init_raw)

# # # # # #         # L_i(0) — initial loss values, set sau epoch đầu tiên
# # # # # #         self.register_buffer("L0",
# # # # # #             torch.ones(self.n_tasks) * float("nan"))
# # # # # #         self._L0_set = False

# # # # # #     @property
# # # # # #     def lambdas(self) -> torch.Tensor:
# # # # # #         """[n_tasks] — luôn dương, giá trị thực của weights."""
# # # # # #         return F.softplus(self.raw_lambdas)

# # # # # #     def set_initial_losses(self, loss_dict: Dict[str, float]):
# # # # # #         """
# # # # # #         Gọi 1 lần sau epoch đầu tiên để set L_i(0).
# # # # # #         loss_dict: {task_name: loss_value}
# # # # # #         """
# # # # # #         for i, name in enumerate(self.task_names):
# # # # # #             if name in loss_dict and not math.isnan(loss_dict[name]):
# # # # # #                 self.L0[i] = loss_dict[name]
# # # # # #         self._L0_set = True

# # # # # #     def weighted_sum(self,
# # # # # #                      raw_losses: Dict[str, torch.Tensor],
# # # # # #                      anchor_losses: Optional[Dict[str, torch.Tensor]] = None,
# # # # # #                      ) -> Tuple[torch.Tensor, Dict[str, float]]:
# # # # # #         """
# # # # # #         Tính tổng loss có weight:
# # # # # #           total = sum_anchor + sum_i λ_i * L_i

# # # # # #         raw_losses:    {name: tensor} — losses được balance bởi GradNorm
# # # # # #         anchor_losses: {name: tensor} — losses KHÔNG balance (DPE)

# # # # # #         Returns:
# # # # # #           (total_loss, lambda_log_dict)
# # # # # #         """
# # # # # #         lam = self.lambdas  # [n_tasks]
# # # # # #         total = raw_losses[self.task_names[0]].new_zeros(())

# # # # # #         # Anchor losses (không multiply λ)
# # # # # #         if anchor_losses:
# # # # # #             for v in anchor_losses.values():
# # # # # #                 total = total + v

# # # # # #         # Balanced losses
# # # # # #         lam_log = {}
# # # # # #         for i, name in enumerate(self.task_names):
# # # # # #             if name in raw_losses:
# # # # # #                 total = total + lam[i] * raw_losses[name]
# # # # # #                 lam_log[f"λ_{name}"] = lam[i].item()

# # # # # #         return total, lam_log

# # # # # #     def gradnorm_loss(self,
# # # # # #                       raw_losses: Dict[str, torch.Tensor],
# # # # # #                       shared_param: torch.Tensor) -> torch.Tensor:
# # # # # #         """
# # # # # #         Tính GradNorm loss để update λ_i.

# # # # # #         GradNorm loss = sum_i |G_i(t) - G_bar(t) * r_i(t)^α|_1
# # # # # #           G_i(t)   = ||∇_{W_shared} (λ_i * L_i)||_2
# # # # # #           G_bar(t) = mean_i(G_i(t))
# # # # # #           r_i(t)   = (L_i(t) / L_i(0)) / mean_j(L_j(t)/L_j(0))

# # # # # #         shared_param: parameter tensor của shared layer (reg_head[-1].weight)
# # # # # #         Gradient của shared_param phải còn trong graph (retain_graph=True).
# # # # # #         """
# # # # # #         lam      = self.lambdas
# # # # # #         n        = self.n_tasks
# # # # # #         device   = shared_param.device

# # # # # #         G_norms = []
# # # # # #         L_cur   = []

# # # # # #         for i, name in enumerate(self.task_names):
# # # # # #             if name not in raw_losses:
# # # # # #                 G_norms.append(torch.zeros(1, device=device))
# # # # # #                 L_cur.append(1.0)
# # # # # #                 continue

# # # # # #             Li     = raw_losses[name]
# # # # # #             # Gradient của λ_i * L_i đối với shared param
# # # # # #             grads  = torch.autograd.grad(
# # # # # #                 lam[i] * Li,
# # # # # #                 shared_param,
# # # # # #                 retain_graph=True,
# # # # # #                 create_graph=True,   # cần để backward qua GradNorm loss
# # # # # #                 allow_unused=True,
# # # # # #             )
# # # # # #             if grads[0] is None:
# # # # # #                 G_norms.append(torch.zeros(1, device=device))
# # # # # #             else:
# # # # # #                 G_norms.append(grads[0].norm(2).unsqueeze(0))
# # # # # #             L_cur.append(Li.detach().item())

# # # # # #         G_norms = torch.cat(G_norms)    # [n_tasks]
# # # # # #         G_bar   = G_norms.mean()         # scalar

# # # # # #         # Relative inverse training rates r_i(t)
# # # # # #         L_cur_t  = torch.tensor(L_cur, device=device)

# # # # # #         if self._L0_set and not torch.isnan(self.L0).any():
# # # # # #             L0_safe  = self.L0.to(device).clamp(min=1e-8)
# # # # # #             loss_ratio = L_cur_t / L0_safe
# # # # # #             r_i = loss_ratio / (loss_ratio.mean() + 1e-8)
# # # # # #         else:
# # # # # #             # Trước khi set L0: tất cả r_i = 1 → equal weighting
# # # # # #             r_i = torch.ones(n, device=device)

# # # # # #         # Target gradient norms
# # # # # #         target_G = (G_bar * r_i.pow(self.alpha)).detach()

# # # # # #         # GradNorm loss = L1 distance
# # # # # #         gn_loss = (G_norms - target_G).abs().sum()
# # # # # #         return gn_loss

# # # # # #     def renormalize(self):
# # # # # #         """
# # # # # #         Normalize λ về sum = n_tasks sau mỗi update.
# # # # # #         Giữ relative magnitudes nhưng ngăn toàn bộ λ drift về 0 hoặc ∞.
# # # # # #         """
# # # # # #         with torch.no_grad():
# # # # # #             lam     = self.lambdas                          # [n_tasks]
# # # # # #             lam_sum = lam.sum().clamp(min=1e-8)
# # # # # #             target  = lam * (self.n_tasks / lam_sum)
# # # # # #             # Inverse softplus để set raw_lambdas
# # # # # #             # softplus(x) = log(1 + exp(x)) → x = log(exp(y) - 1)
# # # # # #             target_clamped = target.clamp(min=1e-3)
# # # # # #             self.raw_lambdas.data = torch.log(
# # # # # #                 torch.expm1(target_clamped).clamp(min=1e-8))

# # # # # #     def lambda_dict(self) -> Dict[str, float]:
# # # # # #         """Trả về dict {task_name: λ_value} để log."""
# # # # # #         lam = self.lambdas.detach()
# # # # # #         return {f"λ_{name}": lam[i].item()
# # # # # #                 for i, name in enumerate(self.task_names)}


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  STTransV2 — Main Model
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # Loss terms được GradNorm balance
# # # # # # _EASY_TASKS = ["mse", "speed_easy", "accel_easy"]
# # # # # # _HARD_TASKS = ["heading", "recurv"]
# # # # # # _ALL_TASKS  = _EASY_TASKS + _HARD_TASKS   # 5 tasks


# # # # # # class STTransV2(nn.Module):
# # # # # #     """
# # # # # #     ST-Trans v2 với GradNorm tự học tất cả loss weights.

# # # # # #     EASY samples: L_DPE [anchor] + λ_mse*L_MSE + λ_speed*L_speed + λ_accel*L_accel
# # # # # #     HARD samples: L_DPE_weighted [anchor] + λ_speed*L_speed + λ_accel*L_accel
# # # # # #                   + λ_heading*L_heading + λ_recurv*L_recurv

# # # # # #     Tất cả λ do GradNormController học tự động.
# # # # # #     step_weights trong L_DPE_weighted cũng learned.
# # # # # #     """

# # # # # #     def __init__(
# # # # # #         self,
# # # # # #         obs_len:           int   = 8,
# # # # # #         pred_len:          int   = 12,
# # # # # #         unet_in_ch:        int   = 13,
# # # # # #         d_model:           int   = 64,
# # # # # #         nhead:             int   = 4,
# # # # # #         num_enc_layers:    int   = 1,
# # # # # #         num_dec_layers:    int   = 3,
# # # # # #         dim_ff:            int   = 512,
# # # # # #         dropout:           float = 0.1,
# # # # # #         v_max_kmh:         float = 80.0,
# # # # # #         dt_h:              float = 6.0,
# # # # # #         # GradNorm
# # # # # #         gradnorm_alpha:    float = 1.5,
# # # # # #         # Hard extras
# # # # # #         step_weight_slope: float = 0.1,
# # # # # #         recurv_threshold:  float = 45.0,
# # # # # #         gate_hidden:       int   = 32,
# # # # # #         recurv_hidden:     int   = 64,
# # # # # #         # Easy/hard thresholds (set sau compute_hard_thresholds)
# # # # # #         threshold_curv:    float = 15.0,
# # # # # #         threshold_spd:     float = 0.5,
# # # # # #     ):
# # # # # #         super().__init__()
# # # # # #         self.obs_len         = obs_len
# # # # # #         self.pred_len        = pred_len
# # # # # #         self.d_model         = d_model
# # # # # #         self.v_max_norm      = v_max_kmh * dt_h / (111.0 * _DEG_SCALE)
# # # # # #         self.recurv_threshold= recurv_threshold
# # # # # #         self.threshold_curv  = threshold_curv
# # # # # #         self.threshold_spd   = threshold_spd

# # # # # #         # ── Encoder (giống STTrans gốc hoàn toàn) ────────────────────────
# # # # # #         self.encoder  = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
# # # # # #         self.ctx_proj = nn.Sequential(
# # # # # #             nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
# # # # # #             nn.LayerNorm(d_model))
# # # # # #         self.obs_enc = ObsKinematicEncoder(
# # # # # #             d_model=d_model, nhead=nhead,
# # # # # #             num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout)

# # # # # #         # ── Decoder (giống STTrans gốc hoàn toàn) ────────────────────────
# # # # # #         self.horizon_queries = nn.Parameter(
# # # # # #             torch.randn(1, pred_len, d_model) * 0.02)
# # # # # #         self.dec_pe = SinusoidalPE(d_model, max_len=pred_len + 10)
# # # # # #         dec_layer = nn.TransformerDecoderLayer(
# # # # # #             d_model=d_model, nhead=nhead,
# # # # # #             dim_feedforward=dim_ff, dropout=dropout,
# # # # # #             activation="relu", batch_first=True)
# # # # # #         self.transformer_dec = nn.TransformerDecoder(
# # # # # #             dec_layer, num_layers=num_dec_layers)
# # # # # #         self.reg_head = nn.Sequential(
# # # # # #             nn.Linear(d_model, d_model), nn.ReLU(),
# # # # # #             nn.Linear(d_model, 2))

# # # # # #         # ── Hard-only modules ─────────────────────────────────────────────
# # # # # #         self.steering_gate = PhysicsSteeringGate(
# # # # # #             ctx_dim=d_model, steer_dim=2,
# # # # # #             hidden=gate_hidden, pred_len=pred_len)
# # # # # #         self.recurv_head = RecurvatureTimingHead(
# # # # # #             ctx_dim=d_model, pred_len=pred_len, hidden=recurv_hidden)

# # # # # #         # Learnable step weights cho L_DPE_weighted (hard)
# # # # # #         init_w   = torch.tensor([1.0 + t*step_weight_slope
# # # # # #                                   for t in range(pred_len)])
# # # # # #         init_raw = torch.log(torch.expm1(init_w.clamp(min=1e-3)))
# # # # # #         self.raw_step_weights = nn.Parameter(init_raw)

# # # # # #         # ── GradNorm Controller ───────────────────────────────────────────
# # # # # #         self.gradnorm = GradNormController(
# # # # # #             task_names = _ALL_TASKS,
# # # # # #             alpha      = gradnorm_alpha,
# # # # # #             init_val   = 1.0,
# # # # # #         )

# # # # # #         self._init_weights()

# # # # # #     def _init_weights(self):
# # # # # #         for m in self.modules():
# # # # # #             if isinstance(m, nn.Linear):
# # # # # #                 nn.init.xavier_uniform_(m.weight, gain=0.5)
# # # # # #                 if m.bias is not None:
# # # # # #                     nn.init.zeros_(m.bias)
# # # # # #         nn.init.constant_(self.steering_gate.gate_net[-1].bias, 2.0)
# # # # # #         nn.init.zeros_(self.steering_gate.gate_net[-1].weight)

# # # # # #     @property
# # # # # #     def step_weights(self) -> torch.Tensor:
# # # # # #         return F.softplus(self.raw_step_weights)

# # # # # #     def set_thresholds(self, threshold_curv: float, threshold_spd: float):
# # # # # #         self.threshold_curv = threshold_curv
# # # # # #         self.threshold_spd  = threshold_spd

# # # # # #     def model_params(self):
# # # # # #         """Parameters của model (không bao gồm GradNorm λ)."""
# # # # # #         gn_ids = set(id(p) for p in self.gradnorm.parameters())
# # # # # #         return [p for p in self.parameters() if id(p) not in gn_ids]

# # # # # #     def gradnorm_params(self):
# # # # # #         """Chỉ GradNorm λ parameters."""
# # # # # #         return list(self.gradnorm.parameters())

# # # # # #     def shared_layer_param(self) -> torch.Tensor:
# # # # # #         """
# # # # # #         Shared layer parameter dùng để tính gradient trong GradNorm.
# # # # # #         Dùng weight của linear layer cuối cùng trong reg_head.
# # # # # #         Đây là điểm mà TẤT CẢ loss terms (easy và hard) đều đi qua.
# # # # # #         """
# # # # # #         return self.reg_head[-1].weight

# # # # # #     # ── Encode: 1 lần duy nhất ───────────────────────────────────────────

# # # # # #     def _encode(self, batch_list) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # # #         """→ (learned_pred [B,H,2], ctx_pooled [B,d])"""
# # # # # #         obs_traj = batch_list[0]
# # # # # #         B        = obs_traj.shape[1]
# # # # # #         raw_ctx     = self.encoder(batch_list)
# # # # # #         ctx_token   = self.ctx_proj(raw_ctx).unsqueeze(1)
# # # # # #         obs_memory  = self.obs_enc(obs_traj)
# # # # # #         full_memory = torch.cat([ctx_token, obs_memory], 1)
# # # # # #         ctx_pooled  = full_memory.mean(1)
# # # # # #         Q  = self.dec_pe(self.horizon_queries.expand(B, -1, -1))
# # # # # #         D  = self.transformer_dec(Q, full_memory)
# # # # # #         lp = self.reg_head(D)              # [B, H, 2]
# # # # # #         return lp, ctx_pooled

# # # # # #     # ── Gate application (hard only) ─────────────────────────────────────

# # # # # #     def _apply_gate(self, learned_pred: torch.Tensor,
# # # # # #                     ctx_pooled: torch.Tensor,
# # # # # #                     steer: Optional[torch.Tensor],
# # # # # #                     obs_traj: torch.Tensor
# # # # # #                     ) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # # #         """→ (final_pred [B_h,H,2], alpha_mean scalar)"""
# # # # # #         B_h = learned_pred.shape[0]
# # # # # #         dev = learned_pred.device
# # # # # #         if steer is None:
# # # # # #             steer = (obs_traj[-1] - obs_traj[-2]
# # # # # #                      if obs_traj.shape[0] >= 2
# # # # # #                      else torch.zeros(B_h, 2, device=dev))

# # # # # #         # Physics: persistence of steering
# # # # # #         cur = obs_traj[-1].clone()
# # # # # #         physics_steps = []
# # # # # #         for _ in range(self.pred_len):
# # # # # #             cur = cur + steer
# # # # # #             physics_steps.append(cur.clone())
# # # # # #         physics = torch.stack(physics_steps, dim=1)  # [B_h, H, 2]

# # # # # #         finals, alphas = [], []
# # # # # #         for t in range(self.pred_len):
# # # # # #             alpha = self.steering_gate(ctx_pooled, steer, t)  # [B_h,1]
# # # # # #             finals.append(alpha * learned_pred[:,t] + (1-alpha) * physics[:,t])
# # # # # #             alphas.append(alpha.mean())

# # # # # #         return torch.stack(finals, dim=1), torch.stack(alphas).mean()

# # # # # #     # ── Compute individual loss terms (không apply λ) ─────────────────────

# # # # # #     def _compute_easy_raw(self,
# # # # # #                           pred: torch.Tensor,
# # # # # #                           gt:   torch.Tensor,
# # # # # #                           ) -> Dict[str, torch.Tensor]:
# # # # # #         """
# # # # # #         pred: [T, B_e, 2]  gt: [T, B_e, 2]
# # # # # #         Returns raw losses (chưa multiply λ) cho easy samples.
# # # # # #         anchor: L_DPE
# # # # # #         balanced: mse, speed_easy, accel_easy
# # # # # #         """
# # # # # #         T  = min(pred.shape[0], gt.shape[0])
# # # # # #         p, g = pred[:T], gt[:T]
# # # # # #         pd = _norm_to_deg(p)
# # # # # #         gd = _norm_to_deg(g)

# # # # # #         l_dpe = haversine_km(pd, gd).mean()
# # # # # #         l_mse = F.mse_loss(p, g)

# # # # # #         if T >= 2:
# # # # # #             sd      = (p[1:] - p[:-1]).norm(-1)
# # # # # #             l_speed = F.relu(sd - self.v_max_norm).pow(2).mean()
# # # # # #         else:
# # # # # #             l_speed = p.new_zeros(())

# # # # # #         if T >= 3:
# # # # # #             vel     = p[1:] - p[:-1]
# # # # # #             l_accel = (vel[1:].norm(-1) - vel[:-1].norm(-1)).pow(2).mean()
# # # # # #         else:
# # # # # #             l_accel = p.new_zeros(())

# # # # # #         return {
# # # # # #             "anchor_easy": l_dpe,      # không balance
# # # # # #             "mse":         l_mse,      # balance
# # # # # #             "speed_easy":  l_speed,    # balance
# # # # # #             "accel_easy":  l_accel,    # balance
# # # # # #         }

# # # # # #     def _compute_hard_raw(self,
# # # # # #                           final_pred:   torch.Tensor,   # [B_h, H, 2] sau gate
# # # # # #                           gt:           torch.Tensor,   # [T, B_h, 2]
# # # # # #                           ctx_pooled:   torch.Tensor,   # [B_h, d]
# # # # # #                           obs_norm:     torch.Tensor,   # [T_obs, B_h, 2]
# # # # # #                           alpha_mean:   torch.Tensor,
# # # # # #                           ) -> Dict[str, torch.Tensor]:
# # # # # #         """
# # # # # #         Returns raw losses cho hard samples.
# # # # # #         anchor: L_DPE_weighted
# # # # # #         balanced: speed_hard, accel_hard, heading, recurv
# # # # # #         NOTE: speed và accel dùng param RIÊNG cho hard (speed_hard, accel_hard)
# # # # # #               vì scale khác với easy.
# # # # # #         """
# # # # # #         T  = min(final_pred.shape[1], gt.shape[0])
# # # # # #         fp = final_pred[:, :T]                     # [B_h, T, 2]
# # # # # #         g  = gt[:T].permute(1, 0, 2)              # [B_h, T, 2]
# # # # # #         fp_t = fp.permute(1, 0, 2)                # [T, B_h, 2]
# # # # # #         gt_t = g.permute(1, 0, 2)                 # [T, B_h, 2]

# # # # # #         # L_DPE_weighted (anchor — không balance)
# # # # # #         sw  = self.step_weights[:T]
# # # # # #         sw  = sw / sw.sum() * T
# # # # # #         hav = haversine_km(_norm_to_deg(fp_t), _norm_to_deg(gt_t))   # [T,B_h]
# # # # # #         l_dpe_w = (hav * sw.unsqueeze(1)).mean()

# # # # # #         # L_speed, L_accel (hard) — balanced riêng
# # # # # #         if T >= 2:
# # # # # #             sd      = (fp_t[1:] - fp_t[:-1]).norm(-1)
# # # # # #             l_speed = F.relu(sd - self.v_max_norm).pow(2).mean()
# # # # # #         else:
# # # # # #             l_speed = fp.new_zeros(())

# # # # # #         if T >= 3:
# # # # # #             vel     = fp_t[1:] - fp_t[:-1]
# # # # # #             l_accel = (vel[1:].norm(-1) - vel[:-1].norm(-1)).pow(2).mean()
# # # # # #         else:
# # # # # #             l_accel = fp.new_zeros(())

# # # # # #         # L_heading: cosine similarity pred_dir vs gt_dir
# # # # # #         if T >= 2:
# # # # # #             pred_dir = fp_t[1:] - fp_t[:-1]      # [T-1, B_h, 2]
# # # # # #             gt_dir   = gt_t[1:] - gt_t[:-1]
# # # # # #             pdn = F.normalize(pred_dir, dim=-1, eps=1e-8)
# # # # # #             gdn = F.normalize(gt_dir,   dim=-1, eps=1e-8)
# # # # # #             l_heading = ((1.0 - (pdn*gdn).sum(-1).clamp(-1,1)) / 2.0).mean()
# # # # # #         else:
# # # # # #             l_heading = fp.new_zeros(())

# # # # # #         # L_recurv: auxiliary cross-entropy
# # # # # #         recurv_logits = self.recurv_head(ctx_pooled)  # [B_h, pred_len+1]
# # # # # #         try:
# # # # # #             label    = RecurvatureTimingHead.make_label(
# # # # # #                 gt, obs_norm, self.recurv_threshold)   # [B_h]
# # # # # #             if label.shape[0] == recurv_logits.shape[0]:
# # # # # #                 l_recurv = F.cross_entropy(recurv_logits, label)
# # # # # #             else:
# # # # # #                 l_recurv = fp.new_zeros(())
# # # # # #         except Exception:
# # # # # #             l_recurv = fp.new_zeros(())

# # # # # #         return {
# # # # # #             "anchor_hard": l_dpe_w,    # không balance
# # # # # #             "speed_hard":  l_speed,    # balance
# # # # # #             "accel_hard":  l_accel,    # balance
# # # # # #             "heading":     l_heading,  # balance
# # # # # #             "recurv":      l_recurv,   # balance
# # # # # #             "alpha_mean":  alpha_mean,
# # # # # #             "step_w_72h":  self.step_weights[-1],
# # # # # #         }

# # # # # #     # ── get_loss_breakdown ────────────────────────────────────────────────

# # # # # #     def get_loss_breakdown(self, batch_list) -> Dict:
# # # # # #         """
# # # # # #         1. Classify easy/hard
# # # # # #         2. Encode 1 lần
# # # # # #         3. Tính raw losses riêng cho easy và hard
# # # # # #         4. Apply GradNorm weighted sum
# # # # # #         5. Trả về dict gồm:
# # # # # #            - "total": loss chính để backward
# # # # # #            - "raw_easy", "raw_hard": dict của raw losses (cho GradNorm)
# # # # # #            - tất cả individual loss values để log
# # # # # #         """
# # # # # #         obs_traj = batch_list[0]
# # # # # #         traj_gt  = batch_list[1]
# # # # # #         B        = obs_traj.shape[1]
# # # # # #         device   = obs_traj.device

# # # # # #         # ── 1. Classify ───────────────────────────────────────────────────
# # # # # #         is_hard  = classify_hard_obs(obs_traj, self.threshold_curv,
# # # # # #                                      self.threshold_spd)
# # # # # #         is_easy  = ~is_hard
# # # # # #         n_easy   = int(is_easy.sum().item())
# # # # # #         n_hard   = int(is_hard.sum().item())

# # # # # #         # ── 2. Encode ─────────────────────────────────────────────────────
# # # # # #         learned_pred, ctx_pooled = self._encode(batch_list)
# # # # # #         # learned_pred: [B, H, 2]  ctx_pooled: [B, d]

# # # # # #         # ── 3. Easy raw losses ────────────────────────────────────────────
# # # # # #         easy_raw   = {}
# # # # # #         easy_info  = {}

# # # # # #         if n_easy > 0:
# # # # # #             lp_e  = learned_pred[is_easy]              # [B_e, H, 2]
# # # # # #             gt_e  = traj_gt[:, is_easy, :]             # [T, B_e, 2]
# # # # # #             pred_e_perm = lp_e.permute(1, 0, 2)        # [T, B_e, 2]
# # # # # #             raw_e = self._compute_easy_raw(pred_e_perm, gt_e)
# # # # # #             # Tách anchor và balanced
# # # # # #             easy_anchor = {"dpe_easy": raw_e.pop("anchor_easy")}
# # # # # #             easy_raw    = raw_e   # {mse, speed_easy, accel_easy}
# # # # # #             easy_info   = {f"easy_{k}": v.item() for k,v in easy_raw.items()}
# # # # # #             easy_info["easy_dpe"] = easy_anchor["dpe_easy"].item()
# # # # # #         else:
# # # # # #             easy_anchor = {}

# # # # # #         # ── 4. Hard raw losses ────────────────────────────────────────────
# # # # # #         hard_raw   = {}
# # # # # #         hard_info  = {}

# # # # # #         if n_hard > 0:
# # # # # #             lp_h  = learned_pred[is_hard]              # [B_h, H, 2]
# # # # # #             gt_h  = traj_gt[:, is_hard, :]             # [T, B_h, 2]
# # # # # #             obs_h = obs_traj[:, is_hard, :]            # [T_obs, B_h, 2]
# # # # # #             ctx_h = ctx_pooled[is_hard]                # [B_h, d]

# # # # # #             data3d = None
# # # # # #             if len(batch_list) > 2 and isinstance(batch_list[2], torch.Tensor):
# # # # # #                 d = batch_list[2]
# # # # # #                 if d.dim() == 4:
# # # # # #                     data3d = d[is_hard]

# # # # # #             steer = _get_steering(data3d, obs_h)
# # # # # #             final_h, alpha_mean = self._apply_gate(lp_h, ctx_h, steer, obs_h)

# # # # # #             raw_h = self._compute_hard_raw(final_h, gt_h, ctx_h, obs_h,
# # # # # #                                            alpha_mean)

# # # # # #             # Tách anchor và non-loss info
# # # # # #             hard_anchor = {"dpe_hard": raw_h.pop("anchor_hard")}
# # # # # #             alpha_val   = raw_h.pop("alpha_mean")
# # # # # #             sw72_val    = raw_h.pop("step_w_72h")

# # # # # #             # speed_hard và accel_hard: không trong _ALL_TASKS,
# # # # # #             # thêm vào anchor để không bị GradNorm balance
# # # # # #             # (giữ đơn giản: chỉ balance heading và recurv cho hard)
# # # # # #             speed_hard_val = raw_h.pop("speed_hard")
# # # # # #             accel_hard_val = raw_h.pop("accel_hard")
# # # # # #             hard_anchor["speed_hard"] = speed_hard_val
# # # # # #             hard_anchor["accel_hard"] = accel_hard_val

# # # # # #             hard_raw   = raw_h   # {heading, recurv}
# # # # # #             hard_info  = {f"hard_{k}": v.item() for k,v in hard_raw.items()}
# # # # # #             hard_info["hard_dpe"]    = hard_anchor["dpe_hard"].item()
# # # # # #             hard_info["alpha_mean"]  = alpha_val.item() \
# # # # # #                                        if isinstance(alpha_val, torch.Tensor) \
# # # # # #                                        else float(alpha_val)
# # # # # #             hard_info["step_w_72h"] = sw72_val.item()
# # # # # #         else:
# # # # # #             hard_anchor = {}
# # # # # #             hard_info   = {}

# # # # # #         # ── 5. GradNorm weighted sum ──────────────────────────────────────
# # # # # #         # Merge raw losses: easy và hard share cùng GradNorm
# # # # # #         # Tasks: [mse, speed_easy, accel_easy, heading, recurv]
# # # # # #         # Nếu batch thiếu easy hoặc hard, loss term = 0 (không có gradient)
# # # # # #         all_raw = {}
# # # # # #         for task in _EASY_TASKS:
# # # # # #             if task in easy_raw:
# # # # # #                 all_raw[task] = easy_raw[task]
# # # # # #             else:
# # # # # #                 # Không có easy samples → zero tensor (không contributes)
# # # # # #                 all_raw[task] = learned_pred.new_zeros((), requires_grad=False)

# # # # # #         for task in _HARD_TASKS:
# # # # # #             if task in hard_raw:
# # # # # #                 all_raw[task] = hard_raw[task]
# # # # # #             else:
# # # # # #                 all_raw[task] = learned_pred.new_zeros((), requires_grad=False)

# # # # # #         # Anchor losses: DPE easy + DPE hard + speed_hard + accel_hard
# # # # # #         all_anchors = {}
# # # # # #         all_anchors.update(easy_anchor)
# # # # # #         all_anchors.update(hard_anchor)

# # # # # #         total, lam_log = self.gradnorm.weighted_sum(all_raw, all_anchors)

# # # # # #         # ── 6. ADE metrics (no grad) ──────────────────────────────────────
# # # # # #         lp_perm = learned_pred.permute(1, 0, 2)   # [H, B, 2]
# # # # # #         with torch.no_grad():
# # # # # #             ade_m = compute_ade_per_horizon(lp_perm.detach(), traj_gt)
# # # # # #             atc_m = compute_ate_cte_per_horizon(lp_perm.detach(), traj_gt)

# # # # # #         result = {
# # # # # #             "total":   total,
# # # # # #             "n_easy":  n_easy,
# # # # # #             "n_hard":  n_hard,
# # # # # #             # Lưu raw losses để train script gọi gradnorm_loss()
# # # # # #             "_raw_losses": all_raw,
# # # # # #         }
# # # # # #         result.update(easy_info)
# # # # # #         result.update(hard_info)
# # # # # #         result.update(lam_log)
# # # # # #         result.update(ade_m)
# # # # # #         result.update(atc_m)
# # # # # #         return result

# # # # # #     def gradnorm_loss(self, bd: Dict) -> torch.Tensor:
# # # # # #         """
# # # # # #         Tính GradNorm loss từ breakdown dict.
# # # # # #         Gọi sau khi đã backward total loss với retain_graph=True.
# # # # # #         """
# # # # # #         raw = bd.get("_raw_losses", {})
# # # # # #         if not raw:
# # # # # #             return self.gradnorm.raw_lambdas.sum() * 0.0

# # # # # #         # Lọc chỉ lấy tasks có actual gradient (tensor từ real computation)
# # # # # #         real_raw = {k: v for k, v in raw.items()
# # # # # #                     if isinstance(v, torch.Tensor) and v.requires_grad}
# # # # # #         if not real_raw:
# # # # # #             return self.gradnorm.raw_lambdas.sum() * 0.0

# # # # # #         shared = self.shared_layer_param()
# # # # # #         return self.gradnorm.gradnorm_loss(real_raw, shared)

# # # # # #     def get_loss(self, batch_list) -> torch.Tensor:
# # # # # #         return self.get_loss_breakdown(batch_list)["total"]

# # # # # #     def forward(self, batch_list) -> torch.Tensor:
# # # # # #         """Inference: không có gate, không GradNorm."""
# # # # # #         lp, _ = self._encode(batch_list)
# # # # # #         return lp.permute(1, 0, 2)   # [H, B, 2]

# # # # # #     @torch.no_grad()
# # # # # #     def sample(self, batch_list, **kwargs):
# # # # # #         pred    = self.forward(batch_list)
# # # # # #         T, B, _ = pred.shape
# # # # # #         return pred, torch.zeros(T, B, 2, device=pred.device), pred.unsqueeze(0)


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Factory
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def build_st_trans_v2(args,
# # # # # #                       threshold_curv: float = 15.0,
# # # # # #                       threshold_spd:  float = 0.5) -> STTransV2:
# # # # # #     return STTransV2(
# # # # # #         obs_len           = getattr(args, 'obs_len',           8),
# # # # # #         pred_len          = getattr(args, 'pred_len',          12),
# # # # # #         unet_in_ch        = getattr(args, 'unet_in_ch',        13),
# # # # # #         d_model           = getattr(args, 'd_model',           64),
# # # # # #         nhead             = getattr(args, 'nhead',             4),
# # # # # #         num_enc_layers    = getattr(args, 'num_enc_layers',    1),
# # # # # #         num_dec_layers    = getattr(args, 'num_dec_layers',    3),
# # # # # #         dim_ff            = getattr(args, 'dim_ff',            512),
# # # # # #         dropout           = getattr(args, 'dropout',           0.1),
# # # # # #         v_max_kmh         = getattr(args, 'v_max_kmh',         80.0),
# # # # # #         gradnorm_alpha    = getattr(args, 'gradnorm_alpha',    1.5),
# # # # # #         step_weight_slope = getattr(args, 'step_weight_slope', 0.1),
# # # # # #         recurv_threshold  = getattr(args, 'recurv_threshold',  45.0),
# # # # # #         gate_hidden       = getattr(args, 'gate_hidden',       32),
# # # # # #         recurv_hidden     = getattr(args, 'recurv_hidden',     64),
# # # # # #         threshold_curv    = threshold_curv,
# # # # # #         threshold_spd     = threshold_spd,
# # # # # #     )
# # # # # """
# # # # # Model/st_trans_v2_model.py  ── ST-Trans v2 (Clean, All Weights Self-Learned)
# # # # # =============================================================================
# # # # # Tất cả weights TỰ HỌC qua:
# # # # #   [A] UncertaintyWeighting (Kendall et al. 2018) — 7 loss terms
# # # # #   [B] Softmax step weights — learnable, không collapse

# # # # # Gradient flow được bảo vệ:
# # # # #   Easy path → Decoder + Encoder          (gradient đầy đủ như ST-Trans)
# # # # #   Hard path → Gate + Encoder only        (decoder PROTECTED bởi lp_h.detach())

# # # # # Fixes so với version trước:
# # # # #   FIX1: step_weights dùng softmax (không collapse về 0)
# # # # #   FIX2: lp_h.detach() bảo vệ decoder khỏi hard gradient
# # # # #   FIX3: Tất cả loss terms qua UW (kể cả DPE)
# # # # #   FIX4: Xóa 'step_w' task khỏi UW (nó không được dùng — dead param)
# # # # #   FIX5: easy/hard balance học qua UW tự động (bỏ w_easy_boost fixed)
# # # # #   FIX6: Tách σ_speed_easy và σ_speed_hard (khác scale)
# # # # # """
# # # # # from __future__ import annotations
# # # # # import math
# # # # # from typing import Dict, List, Optional, Tuple

# # # # # import torch
# # # # # import torch.nn as nn
# # # # # import torch.nn.functional as F

# # # # # from Model.paper_baseline_model import (
# # # # #     PaperEncoder,
# # # # #     _norm_to_deg,
# # # # #     haversine_km,
# # # # #     compute_ade_per_horizon,
# # # # #     compute_ate_cte_per_horizon,
# # # # #     HORIZON_STEPS,
# # # # # )

# # # # # _U500_CH          = 0
# # # # # _V500_CH          = 1
# # # # # _DEG_SCALE        = 25.0
# # # # # _MS_TO_NORM_PER_6H = (6.0 * 3600.0) / (111320.0 * _DEG_SCALE)


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Uncertainty Weighting (Kendall et al. 2018)
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class UncertaintyWeighting(nn.Module):
# # # # #     """
# # # # #     L_weighted_i = L_i / (2σ_i²) + log(σ_i)

# # # # #     σ_i lớn → weight nhỏ  (loss ít quan trọng / khó cân bằng)
# # # # #     σ_i nhỏ → weight lớn  (loss quan trọng)

# # # # #     Gradient tự động: loss lớn → σ tăng → weight giảm
# # # # #                       loss nhỏ → σ giảm → weight tăng
# # # # #     Một backward() pass duy nhất, không cần retain_graph.
# # # # #     """
# # # # #     def __init__(self, task_names: List[str]):
# # # # #         super().__init__()
# # # # #         self.task_names = task_names
# # # # #         # log_σ = 0 → σ = 1 → effective weight = 0.5 ban đầu
# # # # #         self.log_sigma  = nn.Parameter(torch.zeros(len(task_names)))
# # # # #         self._idx       = {n: i for i, n in enumerate(task_names)}

# # # # #     def weight(self, name: str, loss: torch.Tensor) -> torch.Tensor:
# # # # #         i     = self._idx[name]
# # # # #         log_s = self.log_sigma[i]
# # # # #         return torch.exp(-2.0 * log_s) * loss / 2.0 + log_s

# # # # #     def sigma_dict(self) -> Dict[str, float]:
# # # # #         s = torch.exp(self.log_sigma).detach()
# # # # #         return {f"σ_{n}": s[i].item() for i, n in enumerate(self.task_names)}


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Shared modules (giống STTrans gốc)
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class SinusoidalPE(nn.Module):
# # # # #     def __init__(self, d_model: int, max_len: int = 300):
# # # # #         super().__init__()
# # # # #         pe  = torch.zeros(max_len, d_model)
# # # # #         pos = torch.arange(max_len).unsqueeze(1).float()
# # # # #         div = torch.exp(torch.arange(0, d_model, 2).float() *
# # # # #                         (-math.log(10000.0) / d_model))
# # # # #         pe[:, 0::2] = torch.sin(pos * div)
# # # # #         pe[:, 1::2] = torch.cos(pos * div)
# # # # #         self.register_buffer("pe", pe.unsqueeze(0))

# # # # #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# # # # #         return x + self.pe[:, :x.size(1), :]


# # # # # class ObsKinematicEncoder(nn.Module):
# # # # #     FEAT_DIM = 8

# # # # #     def __init__(self, d_model=64, nhead=4, num_layers=1,
# # # # #                  dim_ff=256, dropout=0.1):
# # # # #         super().__init__()
# # # # #         self.proj = nn.Sequential(
# # # # #             nn.Linear(self.FEAT_DIM, d_model), nn.ReLU(),
# # # # #             nn.Linear(d_model, d_model))
# # # # #         self.pe  = SinusoidalPE(d_model, max_len=64)
# # # # #         enc      = nn.TransformerEncoderLayer(
# # # # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # # # #             dropout=dropout, activation="relu", batch_first=True)
# # # # #         self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)

# # # # #     @staticmethod
# # # # #     def _feats(obs: torch.Tensor) -> torch.Tensor:
# # # # #         T, B, _ = obs.shape
# # # # #         dev = obs.device
# # # # #         lon, lat = obs[:,:,0], obs[:,:,1]
# # # # #         if T >= 2:
# # # # #             dl = torch.cat([obs[1:,:,0]-obs[:-1,:,0], torch.zeros(1,B,device=dev)],0)
# # # # #             dt = torch.cat([obs[1:,:,1]-obs[:-1,:,1], torch.zeros(1,B,device=dev)],0)
# # # # #         else:
# # # # #             dl = dt = torch.zeros(T, B, device=dev)
# # # # #         if T >= 3:
# # # # #             ddl = torch.cat([dl[1:]-dl[:-1], torch.zeros(1,B,device=dev)],0)
# # # # #             ddt = torch.cat([dt[1:]-dt[:-1], torch.zeros(1,B,device=dev)],0)
# # # # #         else:
# # # # #             ddl = ddt = torch.zeros(T, B, device=dev)
# # # # #         si  = torch.linspace(0,1,T,device=dev).unsqueeze(1).expand(T,B)
# # # # #         spd = (dl**2 + dt**2).sqrt()
# # # # #         return torch.stack([lon,lat,dl,dt,ddl,ddt,si,spd],-1).permute(1,0,2)

# # # # #     def forward(self, obs: torch.Tensor) -> torch.Tensor:
# # # # #         return self.enc(self.pe(self.proj(self._feats(obs))))


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Easy/Hard classifier
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def classify_hard_obs(obs_traj: torch.Tensor,
# # # # #                       threshold_curv: float,
# # # # #                       threshold_spd:  float) -> torch.Tensor:
# # # # #     """[T_obs, B, 2] → [B] bool (True=hard), no_grad"""
# # # # #     T, B, _ = obs_traj.shape
# # # # #     device   = obs_traj.device
# # # # #     with torch.no_grad():
# # # # #         if T < 2:
# # # # #             return torch.zeros(B, dtype=torch.bool, device=device)
# # # # #         vel      = obs_traj[1:] - obs_traj[:-1]
# # # # #         spd      = vel.norm(dim=-1)
# # # # #         speed_cv = spd.std(0) / (spd.mean(0) + 1e-6)
# # # # #         if T >= 3:
# # # # #             vn   = F.normalize(vel, dim=-1, eps=1e-8)
# # # # #             cos  = (vn[1:] * vn[:-1]).sum(-1).clamp(-1, 1)
# # # # #             curv = (torch.acos(cos) * (180.0/math.pi)).mean(0)
# # # # #         else:
# # # # #             curv = torch.zeros(B, device=device)
# # # # #         return (curv > threshold_curv) | (speed_cv > threshold_spd)


# # # # # def compute_hard_thresholds(train_loader, device,
# # # # #                              percentile: float = 70.0) -> Tuple[float, float]:
# # # # #     all_curv, all_spd = [], []
# # # # #     with torch.no_grad():
# # # # #         for batch in train_loader:
# # # # #             obs = batch[0].to(device)
# # # # #             T, B, _ = obs.shape
# # # # #             if T < 2:
# # # # #                 continue
# # # # #             vel = obs[1:] - obs[:-1]
# # # # #             spd = vel.norm(dim=-1)
# # # # #             all_spd.extend((spd.std(0)/(spd.mean(0)+1e-6)).cpu().tolist())
# # # # #             if T >= 3:
# # # # #                 vn  = F.normalize(vel, dim=-1, eps=1e-8)
# # # # #                 cos = (vn[1:]*vn[:-1]).sum(-1).clamp(-1,1)
# # # # #                 all_curv.extend(
# # # # #                     (torch.acos(cos)*(180/math.pi)).mean(0).cpu().tolist())
# # # # #     if not all_curv:
# # # # #         return 15.0, 0.5
# # # # #     import numpy as np
# # # # #     return (float(np.percentile(all_curv, percentile)),
# # # # #             float(np.percentile(all_spd,  percentile)))


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Physics Steering Gate
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class PhysicsSteeringGate(nn.Module):
# # # # #     """
# # # # #     α[t] = sigmoid(gate(ctx, steer, step_t)) ∈ (0, 1)
# # # # #     pred_final[t] = α[t] * learned[t] + (1-α[t]) * physics[t]

# # # # #     Khởi tạo bias=2.0 → α≈0.88 ban đầu (thiên về learned prediction).
# # # # #     """
# # # # #     def __init__(self, ctx_dim=64, steer_dim=2, hidden=32, pred_len=12):
# # # # #         super().__init__()
# # # # #         self.step_emb = nn.Embedding(pred_len, 16)
# # # # #         self.gate_net = nn.Sequential(
# # # # #             nn.Linear(ctx_dim + 16 + steer_dim, hidden),
# # # # #             nn.ReLU(),
# # # # #             nn.Linear(hidden, 1))
# # # # #         nn.init.constant_(self.gate_net[-1].bias, 2.0)
# # # # #         nn.init.zeros_(self.gate_net[-1].weight)

# # # # #     def forward(self, ctx: torch.Tensor, steer: torch.Tensor,
# # # # #                 step_t: int) -> torch.Tensor:
# # # # #         B   = ctx.shape[0]
# # # # #         dev = ctx.device
# # # # #         emb = self.step_emb(
# # # # #             torch.tensor(step_t, dtype=torch.long, device=dev)
# # # # #         ).unsqueeze(0).expand(B, -1)
# # # # #         return torch.sigmoid(self.gate_net(torch.cat([ctx, emb, steer], -1)))


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Recurvature Timing Head
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class RecurvatureTimingHead(nn.Module):
# # # # #     """
# # # # #     Predict bước nào bão recurve: logits [B, pred_len+1]
# # # # #     class 0 = không recurve, class k+1 = recurve tại step k
# # # # #     """
# # # # #     def __init__(self, ctx_dim=64, pred_len=12, hidden=64):
# # # # #         super().__init__()
# # # # #         self.clf = nn.Sequential(
# # # # #             nn.Linear(ctx_dim, hidden), nn.GELU(),
# # # # #             nn.Dropout(0.1),
# # # # #             nn.Linear(hidden, pred_len + 1))

# # # # #     def forward(self, ctx: torch.Tensor) -> torch.Tensor:
# # # # #         return self.clf(ctx)

# # # # #     @staticmethod
# # # # #     def make_label(gt_norm: torch.Tensor, obs_norm: torch.Tensor,
# # # # #                    threshold_deg: float = 45.0) -> torch.Tensor:
# # # # #         """
# # # # #         gt_norm:  [T_pred, B, 2]
# # # # #         obs_norm: [T_obs,  B, 2]
# # # # #         → label [B] long
# # # # #         """
# # # # #         T_pred, B, _ = gt_norm.shape
# # # # #         dev   = gt_norm.device
# # # # #         label = torch.zeros(B, dtype=torch.long, device=dev)
# # # # #         full  = torch.cat([obs_norm[-1:], gt_norm], 0)  # [T_pred+1, B, 2]
# # # # #         with torch.no_grad():
# # # # #             for t in range(T_pred - 1):
# # # # #                 d_in  = full[t+1] - full[t]
# # # # #                 d_out = full[t+2] - full[t+1]
# # # # #                 ni = F.normalize(d_in,  dim=-1, eps=1e-8)
# # # # #                 no = F.normalize(d_out, dim=-1, eps=1e-8)
# # # # #                 ang  = torch.acos((ni*no).sum(-1).clamp(-1,1)) * (180/math.pi)
# # # # #                 mask = (ang > threshold_deg) & (label == 0)
# # # # #                 label[mask] = t + 1
# # # # #         return label


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Steering helper
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def _get_steering(data3d: Optional[torch.Tensor],
# # # # #                   obs_traj: torch.Tensor) -> Optional[torch.Tensor]:
# # # # #     """ERA5 center pixel → [B, 2] normalised displacement/step."""
# # # # #     if (data3d is None or not isinstance(data3d, torch.Tensor)
# # # # #             or data3d.dim() != 4):
# # # # #         return None
# # # # #     B, C, H, W = data3d.shape
# # # # #     if C <= max(_U500_CH, _V500_CH):
# # # # #         return None
# # # # #     cy, cx  = H // 2, W // 2
# # # # #     u_ms    = data3d[:, _U500_CH, cy, cx]
# # # # #     v_ms    = data3d[:, _V500_CH, cy, cx]
# # # # #     lat_deg = obs_traj[-1, :, 1] * _DEG_SCALE
# # # # #     cos_lat = torch.cos(lat_deg * (math.pi / 180.0)).clamp(0.1, 1.0)
# # # # #     return torch.stack([
# # # # #         (u_ms / cos_lat) * _MS_TO_NORM_PER_6H,
# # # # #         v_ms * _MS_TO_NORM_PER_6H,
# # # # #     ], dim=-1)


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  UW task names  (7 tasks — tất cả tự học)
# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # _UW_TASKS = [
# # # # #     "dpe_easy",     # L_DPE easy
# # # # #     "mse",          # L_MSE easy
# # # # #     "speed_easy",   # L_speed easy  (FIX6: tách riêng)
# # # # #     "accel_easy",   # L_accel easy  (FIX6: tách riêng)
# # # # #     "dpe_hard",     # L_DPE_weighted hard
# # # # #     "speed_hard",   # L_speed hard  (FIX6: tách riêng)
# # # # #     "accel_hard",   # L_accel hard
# # # # #     "heading",      # L_heading hard
# # # # #     "recurv",       # L_recurv hard (auxiliary)
# # # # # ]


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  STTransV2 — Main Model
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class STTransV2(nn.Module):
# # # # #     """
# # # # #     ST-Trans v2 — All weights self-learned, decoder protected.

# # # # #     GRADIENT FLOW:
# # # # #       Easy:  L_easy  → learned_pred → decoder → encoder   ✓
# # # # #       Hard:  L_hard  → final_pred  → gate(α) → encoder   ✓
# # # # #                      → lp_h.detach() → STOP (decoder protected)

# # # # #     LEARNABLE WEIGHTS:
# # # # #       UW: 9 σ params (dpe_easy, mse, speed_easy, accel_easy,
# # # # #                        dpe_hard, speed_hard, accel_hard, heading, recurv)
# # # # #       Step weights: softmax(raw_w) * T  (không collapse, tự học phân phối)
# # # # #     """

# # # # #     def __init__(
# # # # #         self,
# # # # #         obs_len:           int   = 8,
# # # # #         pred_len:          int   = 12,
# # # # #         unet_in_ch:        int   = 13,
# # # # #         d_model:           int   = 64,
# # # # #         nhead:             int   = 4,
# # # # #         num_enc_layers:    int   = 1,
# # # # #         num_dec_layers:    int   = 3,
# # # # #         dim_ff:            int   = 512,
# # # # #         dropout:           float = 0.1,
# # # # #         v_max_kmh:         float = 80.0,
# # # # #         dt_h:              float = 6.0,
# # # # #         # Recurvature
# # # # #         recurv_threshold:  float = 45.0,
# # # # #         gate_hidden:       int   = 32,
# # # # #         recurv_hidden:     int   = 64,
# # # # #         # Thresholds (set sau compute_hard_thresholds)
# # # # #         threshold_curv:    float = 15.0,
# # # # #         threshold_spd:     float = 0.5,
# # # # #     ):
# # # # #         super().__init__()
# # # # #         self.obs_len          = obs_len
# # # # #         self.pred_len         = pred_len
# # # # #         self.d_model          = d_model
# # # # #         self.v_max_norm       = v_max_kmh * dt_h / (111.0 * _DEG_SCALE)
# # # # #         self.recurv_threshold = recurv_threshold
# # # # #         self.threshold_curv   = threshold_curv
# # # # #         self.threshold_spd    = threshold_spd

# # # # #         # ── Step weights: SOFTMAX — không collapse (FIX1) ─────────────────
# # # # #         # sw = softmax(raw_w) * T  → sum(sw) = T luôn
# # # # #         # Optimizer tối ưu phân phối weights, không scale
# # # # #         # Init uniform: raw_w = 0 → softmax = 1/T → sw = 1 (mọi bước bằng nhau)
# # # # #         # Model sẽ học tập trung vào bước nào khó hơn
# # # # #         self.raw_step_weights = nn.Parameter(torch.zeros(pred_len))

# # # # #         # ── Encoder (giống STTrans gốc) ───────────────────────────────────
# # # # #         self.encoder  = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
# # # # #         self.ctx_proj = nn.Sequential(
# # # # #             nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
# # # # #             nn.LayerNorm(d_model))
# # # # #         self.obs_enc  = ObsKinematicEncoder(
# # # # #             d_model=d_model, nhead=nhead,
# # # # #             num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout)

# # # # #         # ── Decoder (giống STTrans gốc) ───────────────────────────────────
# # # # #         self.horizon_queries = nn.Parameter(
# # # # #             torch.randn(1, pred_len, d_model) * 0.02)
# # # # #         self.dec_pe = SinusoidalPE(d_model, max_len=pred_len + 10)
# # # # #         dec_layer = nn.TransformerDecoderLayer(
# # # # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # # # #             dropout=dropout, activation="relu", batch_first=True)
# # # # #         self.transformer_dec = nn.TransformerDecoder(
# # # # #             dec_layer, num_layers=num_dec_layers)
# # # # #         self.reg_head = nn.Sequential(
# # # # #             nn.Linear(d_model, d_model), nn.ReLU(),
# # # # #             nn.Linear(d_model, 2))

# # # # #         # ── Hard-only modules ──────────────────────────────────────────────
# # # # #         self.steering_gate = PhysicsSteeringGate(
# # # # #             ctx_dim=d_model, steer_dim=2,
# # # # #             hidden=gate_hidden, pred_len=pred_len)
# # # # #         self.recurv_head = RecurvatureTimingHead(
# # # # #             ctx_dim=d_model, pred_len=pred_len, hidden=recurv_hidden)

# # # # #         # ── Uncertainty Weighting — 9 tasks tự học (FIX3, FIX4, FIX5, FIX6) ─
# # # # #         self.uw = UncertaintyWeighting(_UW_TASKS)

# # # # #         self._init_weights()

# # # # #     def _init_weights(self):
# # # # #         for m in self.modules():
# # # # #             if isinstance(m, nn.Linear):
# # # # #                 nn.init.xavier_uniform_(m.weight, gain=0.5)
# # # # #                 if m.bias is not None:
# # # # #                     nn.init.zeros_(m.bias)
# # # # #         # Gate: khởi đầu thiên về learned prediction (α ≈ 0.88)
# # # # #         nn.init.constant_(self.steering_gate.gate_net[-1].bias, 2.0)
# # # # #         nn.init.zeros_(self.steering_gate.gate_net[-1].weight)

# # # # #     @property
# # # # #     def step_weights(self) -> torch.Tensor:
# # # # #         """
# # # # #         [pred_len] — SOFTMAX (FIX1: không collapse)
# # # # #         sum(sw) = T luôn luôn
# # # # #         Optimizer học PHÂN PHỐI importance qua các bước, không scale tuyệt đối.
# # # # #         """
# # # # #         return F.softmax(self.raw_step_weights, dim=0) * self.pred_len

# # # # #     def set_thresholds(self, threshold_curv: float, threshold_spd: float):
# # # # #         self.threshold_curv = threshold_curv
# # # # #         self.threshold_spd  = threshold_spd

# # # # #     # ── Encode: 1 lần duy nhất ───────────────────────────────────────────

# # # # #     def _encode(self, batch_list) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # #         """→ (learned_pred [B,H,2], ctx_pooled [B,d])"""
# # # # #         obs  = batch_list[0]
# # # # #         B    = obs.shape[1]
# # # # #         raw  = self.encoder(batch_list)
# # # # #         ctok = self.ctx_proj(raw).unsqueeze(1)
# # # # #         omem = self.obs_enc(obs)
# # # # #         fmem = torch.cat([ctok, omem], 1)
# # # # #         ctx  = fmem.mean(1)
# # # # #         Q    = self.dec_pe(self.horizon_queries.expand(B, -1, -1))
# # # # #         D    = self.transformer_dec(Q, fmem)
# # # # #         lp   = self.reg_head(D)   # [B, H, 2]
# # # # #         return lp, ctx

# # # # #     # ── Gate: chỉ nhận learned_pred đã DETACH (FIX2) ────────────────────

# # # # #     def _apply_gate(self,
# # # # #                     lp_detached: torch.Tensor,   # [B_h, H, 2] — ĐÃ DETACH
# # # # #                     ctx_h:       torch.Tensor,   # [B_h, d]
# # # # #                     steer:       Optional[torch.Tensor],
# # # # #                     obs_h:       torch.Tensor    # [T_obs, B_h, 2]
# # # # #                     ) -> Tuple[torch.Tensor, torch.Tensor]:
# # # # #         """
# # # # #         Gradient từ L_hard chỉ đi: final_pred → α → gate → ctx_h → encoder
# # # # #         KHÔNG đi vào decoder vì lp_detached đã detach.
# # # # #         """
# # # # #         B_h = lp_detached.shape[0]
# # # # #         dev = lp_detached.device

# # # # #         if steer is None:
# # # # #             steer = (obs_h[-1] - obs_h[-2]
# # # # #                      if obs_h.shape[0] >= 2
# # # # #                      else torch.zeros(B_h, 2, device=dev))

# # # # #         # Physics trajectory: persistence of steering
# # # # #         cur   = obs_h[-1].clone()
# # # # #         phys  = []
# # # # #         for _ in range(self.pred_len):
# # # # #             cur = cur + steer
# # # # #             phys.append(cur.clone())
# # # # #         physics = torch.stack(phys, dim=1)   # [B_h, H, 2]

# # # # #         # Blend per step
# # # # #         finals, alphas = [], []
# # # # #         for t in range(self.pred_len):
# # # # #             a = self.steering_gate(ctx_h, steer, t)   # [B_h, 1]
# # # # #             # Gradient: a → gate → ctx_h → encoder ✓
# # # # #             # lp_detached[:,t] không có gradient → decoder protected ✓
# # # # #             finals.append(a * lp_detached[:, t] + (1 - a) * physics[:, t])
# # # # #             alphas.append(a.mean())

# # # # #         return torch.stack(finals, dim=1), torch.stack(alphas).mean()

# # # # #     # ── Easy raw losses ───────────────────────────────────────────────────

# # # # #     def _easy_losses(self, pred: torch.Tensor,
# # # # #                      gt: torch.Tensor) -> Dict[str, torch.Tensor]:
# # # # #         """pred, gt: [T, B_e, 2] → raw losses (chưa UW-weighted)"""
# # # # #         T = min(pred.shape[0], gt.shape[0])
# # # # #         p, g = pred[:T], gt[:T]

# # # # #         l_dpe = haversine_km(_norm_to_deg(p), _norm_to_deg(g)).mean()
# # # # #         l_mse = F.mse_loss(p, g)

# # # # #         if T >= 2:
# # # # #             sd      = (p[1:] - p[:-1]).norm(-1)
# # # # #             l_speed = F.relu(sd - self.v_max_norm).pow(2).mean()
# # # # #         else:
# # # # #             l_speed = p.new_zeros(())

# # # # #         if T >= 3:
# # # # #             v       = p[1:] - p[:-1]
# # # # #             l_accel = (v[1:].norm(-1) - v[:-1].norm(-1)).pow(2).mean()
# # # # #         else:
# # # # #             l_accel = p.new_zeros(())

# # # # #         return {"dpe": l_dpe, "mse": l_mse,
# # # # #                 "speed": l_speed, "accel": l_accel}

# # # # #     # ── Hard raw losses ───────────────────────────────────────────────────

# # # # #     def _hard_losses(self,
# # # # #                      final_pred: torch.Tensor,   # [B_h, H, 2]
# # # # #                      gt:         torch.Tensor,   # [T_pred, B_h, 2]
# # # # #                      ctx_h:      torch.Tensor,   # [B_h, d]
# # # # #                      obs_h:      torch.Tensor,   # [T_obs, B_h, 2]
# # # # #                      alpha_mean: torch.Tensor,
# # # # #                      ) -> Dict[str, torch.Tensor]:
# # # # #         """→ raw losses cho hard samples"""
# # # # #         T    = min(final_pred.shape[1], gt.shape[0])
# # # # #         fp   = final_pred[:, :T]           # [B_h, T, 2]
# # # # #         g    = gt[:T].permute(1, 0, 2)    # [B_h, T, 2]
# # # # #         fp_t = fp.permute(1, 0, 2)        # [T, B_h, 2]
# # # # #         gt_t = g.permute(1, 0, 2)         # [T, B_h, 2]

# # # # #         # L_DPE_weighted với step weights SOFTMAX (FIX1: không collapse)
# # # # #         sw      = self.step_weights[:T]        # [T], sum = T
# # # # #         hav     = haversine_km(_norm_to_deg(fp_t), _norm_to_deg(gt_t))  # [T,B_h]
# # # # #         l_dpe_w = (hav * sw.unsqueeze(1)).mean()

# # # # #         # L_speed, L_accel
# # # # #         if T >= 2:
# # # # #             sd      = (fp_t[1:] - fp_t[:-1]).norm(-1)
# # # # #             l_speed = F.relu(sd - self.v_max_norm).pow(2).mean()
# # # # #         else:
# # # # #             l_speed = fp.new_zeros(())

# # # # #         if T >= 3:
# # # # #             v       = fp_t[1:] - fp_t[:-1]
# # # # #             l_accel = (v[1:].norm(-1) - v[:-1].norm(-1)).pow(2).mean()
# # # # #         else:
# # # # #             l_accel = fp.new_zeros(())

# # # # #         # L_heading: cosine similarity giữa pred_dir và gt_dir
# # # # #         if T >= 2:
# # # # #             pd = fp_t[1:] - fp_t[:-1]     # [T-1, B_h, 2]
# # # # #             gd = gt_t[1:] - gt_t[:-1]
# # # # #             pn = F.normalize(pd, dim=-1, eps=1e-8)
# # # # #             gn = F.normalize(gd, dim=-1, eps=1e-8)
# # # # #             l_heading = ((1.0 - (pn * gn).sum(-1).clamp(-1, 1)) / 2.0).mean()
# # # # #         else:
# # # # #             l_heading = fp.new_zeros(())

# # # # #         # L_recurv: auxiliary cross-entropy trên timing head
# # # # #         recurv_logits = self.recurv_head(ctx_h)   # [B_h, pred_len+1]
# # # # #         try:
# # # # #             # gt:   [T_pred, B_h, 2] — đúng shape cho make_label
# # # # #             # obs_h: [T_obs,  B_h, 2] — đúng shape cho make_label
# # # # #             label = RecurvatureTimingHead.make_label(
# # # # #                 gt, obs_h, self.recurv_threshold)
# # # # #             l_recurv = (F.cross_entropy(recurv_logits, label)
# # # # #                         if label.shape[0] == recurv_logits.shape[0]
# # # # #                         else fp.new_zeros(()))
# # # # #         except Exception:
# # # # #             l_recurv = fp.new_zeros(())

# # # # #         return {
# # # # #             "dpe_w":   l_dpe_w,
# # # # #             "speed":   l_speed,
# # # # #             "accel":   l_accel,
# # # # #             "heading": l_heading,
# # # # #             "recurv":  l_recurv,
# # # # #             "alpha":   alpha_mean,
# # # # #         }

# # # # #     # ── get_loss_breakdown ────────────────────────────────────────────────

# # # # #     def get_loss_breakdown(self, batch_list) -> Dict:
# # # # #         obs_traj = batch_list[0]   # [T_obs, B, 2]
# # # # #         traj_gt  = batch_list[1]   # [T_pred, B, 2]
# # # # #         B        = obs_traj.shape[1]

# # # # #         # 1. Classify easy/hard (no grad)
# # # # #         is_hard = classify_hard_obs(obs_traj, self.threshold_curv,
# # # # #                                     self.threshold_spd)
# # # # #         is_easy = ~is_hard
# # # # #         n_easy  = int(is_easy.sum().item())
# # # # #         n_hard  = int(is_hard.sum().item())

# # # # #         # 2. Encode 1 lần — shared
# # # # #         learned_pred, ctx_pooled = self._encode(batch_list)
# # # # #         # learned_pred: [B, H, 2] có gradient đến decoder

# # # # #         result = {"n_easy": n_easy, "n_hard": n_hard}

# # # # #         # ── 3. EASY LOSS ─────────────────────────────────────────────────
# # # # #         # Gradient đi bình thường: L_easy → learned_pred → decoder + encoder
# # # # #         L_easy = None
# # # # #         if n_easy > 0:
# # # # #             lp_e = learned_pred[is_easy].permute(1, 0, 2)  # [T, B_e, 2]
# # # # #             gt_e = traj_gt[:, is_easy, :]                   # [T, B_e, 2]

# # # # #             el = self._easy_losses(lp_e, gt_e)

# # # # #             # TẤT CẢ qua UW — không có anchor cứng (FIX3)
# # # # #             L_easy = (
# # # # #                 self.uw.weight("dpe_easy",  el["dpe"])
# # # # #                 + self.uw.weight("mse",     el["mse"])
# # # # #                 + self.uw.weight("speed_easy", el["speed"])
# # # # #                 + self.uw.weight("accel_easy", el["accel"])
# # # # #             )
# # # # #             result.update({
# # # # #                 "easy_dpe":   el["dpe"].item(),
# # # # #                 "easy_mse":   el["mse"].item(),
# # # # #                 "easy_speed": el["speed"].item(),
# # # # #                 "easy_accel": el["accel"].item(),
# # # # #             })

# # # # #         # ── 4. HARD LOSS ──────────────────────────────────────────────────
# # # # #         # FIX2: lp_h.detach() → decoder KHÔNG nhận gradient từ L_hard
# # # # #         # Gradient: L_hard → final_pred → gate(α) → ctx_h → encoder
# # # # #         L_hard = None
# # # # #         if n_hard > 0:
# # # # #             lp_h  = learned_pred[is_hard]        # [B_h, H, 2]
# # # # #             gt_h  = traj_gt[:, is_hard, :]       # [T_pred, B_h, 2]
# # # # #             obs_h = obs_traj[:, is_hard, :]      # [T_obs, B_h, 2]
# # # # #             ctx_h = ctx_pooled[is_hard]          # [B_h, d]

# # # # #             data3d = None
# # # # #             if (len(batch_list) > 2
# # # # #                     and isinstance(batch_list[2], torch.Tensor)
# # # # #                     and batch_list[2].dim() == 4):
# # # # #                 data3d = batch_list[2][is_hard]

# # # # #             steer = _get_steering(data3d, obs_h)

# # # # #             # KEY: lp_h.detach() bảo vệ decoder
# # # # #             final_h, alpha_mean = self._apply_gate(
# # # # #                 lp_h.detach(), ctx_h, steer, obs_h)

# # # # #             hl = self._hard_losses(final_h, gt_h, ctx_h, obs_h, alpha_mean)

# # # # #             # TẤT CẢ qua UW (FIX3), σ tách riêng easy/hard (FIX6)
# # # # #             L_hard = (
# # # # #                 self.uw.weight("dpe_hard",   hl["dpe_w"])
# # # # #                 + self.uw.weight("speed_hard", hl["speed"])
# # # # #                 + self.uw.weight("accel_hard", hl["accel"])
# # # # #                 + self.uw.weight("heading",    hl["heading"])
# # # # #                 + self.uw.weight("recurv",     hl["recurv"])
# # # # #             )
# # # # #             result.update({
# # # # #                 "hard_dpe":    hl["dpe_w"].item(),
# # # # #                 "hard_heading": (hl["heading"].item()
# # # # #                                  if isinstance(hl["heading"], torch.Tensor)
# # # # #                                  else 0.0),
# # # # #                 "hard_recurv":  (hl["recurv"].item()
# # # # #                                  if isinstance(hl["recurv"], torch.Tensor)
# # # # #                                  else 0.0),
# # # # #                 "alpha_mean":   (hl["alpha"].item()
# # # # #                                  if isinstance(hl["alpha"], torch.Tensor)
# # # # #                                  else 0.0),
# # # # #                 "sw_min": self.step_weights.min().item(),
# # # # #                 "sw_max": self.step_weights.max().item(),
# # # # #                 "sw72":   self.step_weights[-1].item(),
# # # # #             })

# # # # #         # 5. Combine: UW tự cân bằng easy/hard (FIX5: bỏ w_easy_boost)
# # # # #         # Không cần boost cứng vì UW σ_dpe_easy sẽ tự điều chỉnh
# # # # #         if L_easy is not None and L_hard is not None:
# # # # #             total = L_easy + L_hard
# # # # #         elif L_easy is not None:
# # # # #             total = L_easy
# # # # #         else:
# # # # #             total = L_hard

# # # # #         result["total"] = total

# # # # #         # Log σ để monitor
# # # # #         result.update(self.uw.sigma_dict())

# # # # #         # 6. ADE metrics (no grad)
# # # # #         lp_perm = learned_pred.permute(1, 0, 2)   # [H, B, 2]
# # # # #         with torch.no_grad():
# # # # #             result.update(compute_ade_per_horizon(lp_perm.detach(), traj_gt))
# # # # #             result.update(compute_ate_cte_per_horizon(lp_perm.detach(), traj_gt))

# # # # #         return result

# # # # #     def get_loss(self, batch_list) -> torch.Tensor:
# # # # #         return self.get_loss_breakdown(batch_list)["total"]

# # # # #     def forward(self, batch_list) -> torch.Tensor:
# # # # #         """Inference: decoder trực tiếp, không gate"""
# # # # #         lp, _ = self._encode(batch_list)
# # # # #         return lp.permute(1, 0, 2)   # [H, B, 2]

# # # # #     @torch.no_grad()
# # # # #     def sample(self, batch_list, **kwargs):
# # # # #         pred    = self.forward(batch_list)
# # # # #         T, B, _ = pred.shape
# # # # #         return pred, torch.zeros(T, B, 2, device=pred.device), pred.unsqueeze(0)


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Factory
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def build_st_trans_v2(args, threshold_curv=15.0, threshold_spd=0.5):
# # # # #     return STTransV2(
# # # # #         obs_len          = getattr(args, 'obs_len',          8),
# # # # #         pred_len         = getattr(args, 'pred_len',         12),
# # # # #         unet_in_ch       = getattr(args, 'unet_in_ch',       13),
# # # # #         d_model          = getattr(args, 'd_model',          64),
# # # # #         nhead            = getattr(args, 'nhead',            4),
# # # # #         num_enc_layers   = getattr(args, 'num_enc_layers',   1),
# # # # #         num_dec_layers   = getattr(args, 'num_dec_layers',   3),
# # # # #         dim_ff           = getattr(args, 'dim_ff',           512),
# # # # #         dropout          = getattr(args, 'dropout',          0.1),
# # # # #         v_max_kmh        = getattr(args, 'v_max_kmh',        80.0),
# # # # #         recurv_threshold = getattr(args, 'recurv_threshold', 45.0),
# # # # #         gate_hidden      = getattr(args, 'gate_hidden',      32),
# # # # #         recurv_hidden    = getattr(args, 'recurv_hidden',    64),
# # # # #         threshold_curv   = threshold_curv,
# # # # #         threshold_spd    = threshold_spd,
# # # # #     )

# # # # """
# # # # Model/st_trans_v2_model.py  ── ST-Trans v2 (Clean, All Weights Self-Learned)
# # # # =============================================================================
# # # # Tất cả weights TỰ HỌC qua:
# # # #   [A] UncertaintyWeighting (Kendall et al. 2018) — 7 loss terms
# # # #   [B] Softmax step weights — learnable, không collapse

# # # # Gradient flow được bảo vệ:
# # # #   Easy path → Decoder + Encoder          (gradient đầy đủ như ST-Trans)
# # # #   Hard path → Gate + Encoder only        (decoder PROTECTED bởi lp_h.detach())

# # # # Fixes so với version trước:
# # # #   FIX1: step_weights dùng softmax (không collapse về 0)
# # # #   FIX2: lp_h.detach() bảo vệ decoder khỏi hard gradient
# # # #   FIX3: Tất cả loss terms qua UW (kể cả DPE)
# # # #   FIX4: Xóa 'step_w' task khỏi UW (nó không được dùng — dead param)
# # # #   FIX5: easy/hard balance học qua UW tự động (bỏ w_easy_boost fixed)
# # # #   FIX6: Tách σ_speed_easy và σ_speed_hard (khác scale)
# # # # """
# # # # from __future__ import annotations
# # # # import math
# # # # from typing import Dict, List, Optional, Tuple

# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.nn.functional as F

# # # # from Model.paper_baseline_model import (
# # # #     PaperEncoder,
# # # #     _norm_to_deg,
# # # #     haversine_km,
# # # #     compute_ade_per_horizon,
# # # #     compute_ate_cte_per_horizon,
# # # #     HORIZON_STEPS,
# # # # )

# # # # _U500_CH          = 0
# # # # _V500_CH          = 1
# # # # _DEG_SCALE        = 25.0
# # # # _MS_TO_NORM_PER_6H = (6.0 * 3600.0) / (111320.0 * _DEG_SCALE)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Uncertainty Weighting (Kendall et al. 2018)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class UncertaintyWeighting(nn.Module):
# # # #     """
# # # #     L_weighted_i = L_i / (2σ_i²) + log(σ_i)

# # # #     σ_i lớn → weight nhỏ  (loss ít quan trọng / khó cân bằng)
# # # #     σ_i nhỏ → weight lớn  (loss quan trọng)

# # # #     Gradient tự động: loss lớn → σ tăng → weight giảm
# # # #                       loss nhỏ → σ giảm → weight tăng
# # # #     Một backward() pass duy nhất, không cần retain_graph.
# # # #     """
# # # #     def __init__(self, task_names: List[str],
# # # #                  init_log_sigmas=None):
# # # #         super().__init__()
# # # #         self.task_names = task_names
# # # #         inits = torch.zeros(len(task_names))
# # # #         if init_log_sigmas:
# # # #             for i, n in enumerate(task_names):
# # # #                 if n in init_log_sigmas:
# # # #                     inits[i] = init_log_sigmas[n]
# # # #         self.log_sigma = nn.Parameter(inits)
# # # #         self._idx      = {n: i for i, n in enumerate(task_names)}
# # # #     def weight(self, name: str, loss: torch.Tensor) -> torch.Tensor:
# # # #         i     = self._idx[name]
# # # #         log_s = self.log_sigma[i]
# # # #         return torch.exp(-2.0 * log_s) * loss / 2.0 + log_s

# # # #     def sigma_dict(self) -> Dict[str, float]:
# # # #         s = torch.exp(self.log_sigma).detach()
# # # #         return {f"σ_{n}": s[i].item() for i, n in enumerate(self.task_names)}


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Shared modules (giống STTrans gốc)
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class SinusoidalPE(nn.Module):
# # # #     def __init__(self, d_model: int, max_len: int = 300):
# # # #         super().__init__()
# # # #         pe  = torch.zeros(max_len, d_model)
# # # #         pos = torch.arange(max_len).unsqueeze(1).float()
# # # #         div = torch.exp(torch.arange(0, d_model, 2).float() *
# # # #                         (-math.log(10000.0) / d_model))
# # # #         pe[:, 0::2] = torch.sin(pos * div)
# # # #         pe[:, 1::2] = torch.cos(pos * div)
# # # #         self.register_buffer("pe", pe.unsqueeze(0))

# # # #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# # # #         return x + self.pe[:, :x.size(1), :]


# # # # class ObsKinematicEncoder(nn.Module):
# # # #     FEAT_DIM = 8

# # # #     def __init__(self, d_model=64, nhead=4, num_layers=1,
# # # #                  dim_ff=256, dropout=0.1):
# # # #         super().__init__()
# # # #         self.proj = nn.Sequential(
# # # #             nn.Linear(self.FEAT_DIM, d_model), nn.ReLU(),
# # # #             nn.Linear(d_model, d_model))
# # # #         self.pe  = SinusoidalPE(d_model, max_len=64)
# # # #         enc      = nn.TransformerEncoderLayer(
# # # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # # #             dropout=dropout, activation="relu", batch_first=True)
# # # #         self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)

# # # #     @staticmethod
# # # #     def _feats(obs: torch.Tensor) -> torch.Tensor:
# # # #         T, B, _ = obs.shape
# # # #         dev = obs.device
# # # #         lon, lat = obs[:,:,0], obs[:,:,1]
# # # #         if T >= 2:
# # # #             dl = torch.cat([obs[1:,:,0]-obs[:-1,:,0], torch.zeros(1,B,device=dev)],0)
# # # #             dt = torch.cat([obs[1:,:,1]-obs[:-1,:,1], torch.zeros(1,B,device=dev)],0)
# # # #         else:
# # # #             dl = dt = torch.zeros(T, B, device=dev)
# # # #         if T >= 3:
# # # #             ddl = torch.cat([dl[1:]-dl[:-1], torch.zeros(1,B,device=dev)],0)
# # # #             ddt = torch.cat([dt[1:]-dt[:-1], torch.zeros(1,B,device=dev)],0)
# # # #         else:
# # # #             ddl = ddt = torch.zeros(T, B, device=dev)
# # # #         si  = torch.linspace(0,1,T,device=dev).unsqueeze(1).expand(T,B)
# # # #         spd = (dl**2 + dt**2).sqrt()
# # # #         return torch.stack([lon,lat,dl,dt,ddl,ddt,si,spd],-1).permute(1,0,2)

# # # #     def forward(self, obs: torch.Tensor) -> torch.Tensor:
# # # #         return self.enc(self.pe(self.proj(self._feats(obs))))


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Easy/Hard classifier
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def classify_hard_obs(obs_traj: torch.Tensor,
# # # #                       threshold_curv: float,
# # # #                       threshold_spd:  float) -> torch.Tensor:
# # # #     """[T_obs, B, 2] → [B] bool (True=hard), no_grad"""
# # # #     T, B, _ = obs_traj.shape
# # # #     device   = obs_traj.device
# # # #     with torch.no_grad():
# # # #         if T < 2:
# # # #             return torch.zeros(B, dtype=torch.bool, device=device)
# # # #         vel      = obs_traj[1:] - obs_traj[:-1]
# # # #         spd      = vel.norm(dim=-1)
# # # #         speed_cv = spd.std(0) / (spd.mean(0) + 1e-6)
# # # #         if T >= 3:
# # # #             vn   = F.normalize(vel, dim=-1, eps=1e-8)
# # # #             cos  = (vn[1:] * vn[:-1]).sum(-1).clamp(-1, 1)
# # # #             curv = (torch.acos(cos) * (180.0/math.pi)).mean(0)
# # # #         else:
# # # #             curv = torch.zeros(B, device=device)
# # # #         return (curv > threshold_curv) | (speed_cv > threshold_spd)


# # # # def compute_hard_thresholds(train_loader, device,
# # # #                              percentile: float = 70.0) -> Tuple[float, float]:
# # # #     all_curv, all_spd = [], []
# # # #     with torch.no_grad():
# # # #         for batch in train_loader:
# # # #             obs = batch[0].to(device)
# # # #             T, B, _ = obs.shape
# # # #             if T < 2:
# # # #                 continue
# # # #             vel = obs[1:] - obs[:-1]
# # # #             spd = vel.norm(dim=-1)
# # # #             all_spd.extend((spd.std(0)/(spd.mean(0)+1e-6)).cpu().tolist())
# # # #             if T >= 3:
# # # #                 vn  = F.normalize(vel, dim=-1, eps=1e-8)
# # # #                 cos = (vn[1:]*vn[:-1]).sum(-1).clamp(-1,1)
# # # #                 all_curv.extend(
# # # #                     (torch.acos(cos)*(180/math.pi)).mean(0).cpu().tolist())
# # # #     if not all_curv:
# # # #         return 15.0, 0.5
# # # #     import numpy as np
# # # #     return (float(np.percentile(all_curv, percentile)),
# # # #             float(np.percentile(all_spd,  percentile)))


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Physics Steering Gate
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class PhysicsSteeringGate(nn.Module):
# # # #     """
# # # #     α[t] = sigmoid(gate(ctx, steer, step_t)) ∈ (0, 1)
# # # #     pred_final[t] = α[t] * learned[t] + (1-α[t]) * physics[t]

# # # #     Khởi tạo bias=2.0 → α≈0.88 ban đầu (thiên về learned prediction).
# # # #     """
# # # #     def __init__(self, ctx_dim=64, steer_dim=2, hidden=32, pred_len=12):
# # # #         super().__init__()
# # # #         self.step_emb = nn.Embedding(pred_len, 16)
# # # #         self.gate_net = nn.Sequential(
# # # #             nn.Linear(ctx_dim + 16 + steer_dim, hidden),
# # # #             nn.ReLU(),
# # # #             nn.Linear(hidden, 1))
# # # #         nn.init.constant_(self.gate_net[-1].bias, 2.0)
# # # #         nn.init.zeros_(self.gate_net[-1].weight)

# # # #     def forward(self, ctx: torch.Tensor, steer: torch.Tensor,
# # # #                 step_t: int) -> torch.Tensor:
# # # #         B   = ctx.shape[0]
# # # #         dev = ctx.device
# # # #         emb = self.step_emb(
# # # #             torch.tensor(step_t, dtype=torch.long, device=dev)
# # # #         ).unsqueeze(0).expand(B, -1)
# # # #         return torch.sigmoid(self.gate_net(torch.cat([ctx, emb, steer], -1)))


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Recurvature Timing Head
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class RecurvatureTimingHead(nn.Module):
# # # #     """
# # # #     Predict bước nào bão recurve: logits [B, pred_len+1]
# # # #     class 0 = không recurve, class k+1 = recurve tại step k
# # # #     """
# # # #     def __init__(self, ctx_dim=64, pred_len=12, hidden=64):
# # # #         super().__init__()
# # # #         self.clf = nn.Sequential(
# # # #             nn.Linear(ctx_dim, hidden), nn.GELU(),
# # # #             nn.Dropout(0.1),
# # # #             nn.Linear(hidden, pred_len + 1))

# # # #     def forward(self, ctx: torch.Tensor) -> torch.Tensor:
# # # #         return self.clf(ctx)

# # # #     @staticmethod
# # # #     def make_label(gt_norm: torch.Tensor, obs_norm: torch.Tensor,
# # # #                    threshold_deg: float = 45.0) -> torch.Tensor:
# # # #         """
# # # #         gt_norm:  [T_pred, B, 2]
# # # #         obs_norm: [T_obs,  B, 2]
# # # #         → label [B] long
# # # #         """
# # # #         T_pred, B, _ = gt_norm.shape
# # # #         dev   = gt_norm.device
# # # #         label = torch.zeros(B, dtype=torch.long, device=dev)
# # # #         full  = torch.cat([obs_norm[-1:], gt_norm], 0)  # [T_pred+1, B, 2]
# # # #         with torch.no_grad():
# # # #             for t in range(T_pred - 1):
# # # #                 d_in  = full[t+1] - full[t]
# # # #                 d_out = full[t+2] - full[t+1]
# # # #                 ni = F.normalize(d_in,  dim=-1, eps=1e-8)
# # # #                 no = F.normalize(d_out, dim=-1, eps=1e-8)
# # # #                 ang  = torch.acos((ni*no).sum(-1).clamp(-1,1)) * (180/math.pi)
# # # #                 mask = (ang > threshold_deg) & (label == 0)
# # # #                 label[mask] = t + 1
# # # #         return label


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Steering helper
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def _get_steering(data3d: Optional[torch.Tensor],
# # # #                   obs_traj: torch.Tensor) -> Optional[torch.Tensor]:
# # # #     """ERA5 center pixel → [B, 2] normalised displacement/step."""
# # # #     if (data3d is None or not isinstance(data3d, torch.Tensor)
# # # #             or data3d.dim() != 4):
# # # #         return None
# # # #     B, C, H, W = data3d.shape
# # # #     if C <= max(_U500_CH, _V500_CH):
# # # #         return None
# # # #     cy, cx  = H // 2, W // 2
# # # #     u_ms    = data3d[:, _U500_CH, cy, cx]
# # # #     v_ms    = data3d[:, _V500_CH, cy, cx]
# # # #     lat_deg = obs_traj[-1, :, 1] * _DEG_SCALE
# # # #     cos_lat = torch.cos(lat_deg * (math.pi / 180.0)).clamp(0.1, 1.0)
# # # #     return torch.stack([
# # # #         (u_ms / cos_lat) * _MS_TO_NORM_PER_6H,
# # # #         v_ms * _MS_TO_NORM_PER_6H,
# # # #     ], dim=-1)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  UW task names  (7 tasks — tất cả tự học)
# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # _UW_TASKS = [
# # # #     "dpe_easy",  # L_DPE easy — chính (quan trọng nhất)
# # # #     "dpe_hard",  # L_DPE_weighted hard — chính
# # # #     "mse",       # L_MSE easy — regularization
# # # #     "speed",     # L_speed shared easy+hard
# # # #     "accel",     # L_accel shared easy+hard
# # # #     "heading",   # L_heading hard — auxiliary
# # # #     "recurv",    # L_recurv hard — auxiliary (ít nhất)
# # # # ]

# # # # # Init log_σ: nhỏ hơn = weight lớn hơn = task quan trọng hơn
# # # # # UW weight = 1/(2σ²), σ = exp(log_σ)
# # # # _UW_INIT_LOG_SIGMA = {
# # # #     "dpe_easy": -0.5,   # σ≈0.61, w≈1.34 — task chính
# # # #     "dpe_hard": -0.5,   # σ≈0.61, w≈1.34 — task chính
# # # #     "mse":       0.0,   # σ=1.00, w=0.50
# # # #     "speed":     0.0,   # σ=1.00, w=0.50
# # # #     "accel":     0.0,   # σ=1.00, w=0.50
# # # #     "heading":   0.3,   # σ≈1.35, w≈0.27 — direction
# # # #     "recurv":    0.7,   # σ≈2.01, w≈0.12 — auxiliary
# # # # }


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  STTransV2 — Main Model
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class STTransV2(nn.Module):
# # # #     """
# # # #     ST-Trans v2 — All weights self-learned, decoder protected.

# # # #     GRADIENT FLOW:
# # # #       Easy:  L_easy  → learned_pred → decoder → encoder   ✓
# # # #       Hard:  L_hard  → final_pred  → gate(α) → encoder   ✓
# # # #                      → lp_h.detach() → STOP (decoder protected)

# # # #     LEARNABLE WEIGHTS:
# # # #       UW: 9 σ params (dpe_easy, mse, speed_easy, accel_easy,
# # # #                        dpe_hard, speed_hard, accel_hard, heading, recurv)
# # # #       Step weights: softmax(raw_w) * T  (không collapse, tự học phân phối)
# # # #     """

# # # #     def __init__(
# # # #         self,
# # # #         obs_len:           int   = 8,
# # # #         pred_len:          int   = 12,
# # # #         unet_in_ch:        int   = 13,
# # # #         d_model:           int   = 64,
# # # #         nhead:             int   = 4,
# # # #         num_enc_layers:    int   = 1,
# # # #         num_dec_layers:    int   = 3,
# # # #         dim_ff:            int   = 512,
# # # #         dropout:           float = 0.1,
# # # #         v_max_kmh:         float = 80.0,
# # # #         dt_h:              float = 6.0,
# # # #         # Recurvature
# # # #         recurv_threshold:  float = 45.0,
# # # #         gate_hidden:       int   = 32,
# # # #         recurv_hidden:     int   = 64,
# # # #         # Thresholds (set sau compute_hard_thresholds)
# # # #         threshold_curv:    float = 15.0,
# # # #         threshold_spd:     float = 0.5,
# # # #     ):
# # # #         super().__init__()
# # # #         self.obs_len          = obs_len
# # # #         self.pred_len         = pred_len
# # # #         self.d_model          = d_model
# # # #         self.v_max_norm       = v_max_kmh * dt_h / (111.0 * _DEG_SCALE)
# # # #         self.recurv_threshold = recurv_threshold
# # # #         self.threshold_curv   = threshold_curv
# # # #         self.threshold_spd    = threshold_spd

# # # #         # ── Step weights: SOFTMAX — không collapse (FIX1) ─────────────────
# # # #         # sw = softmax(raw_w) * T  → sum(sw) = T luôn
# # # #         # Optimizer tối ưu phân phối weights, không scale
# # # #         # Init uniform: raw_w = 0 → softmax = 1/T → sw = 1 (mọi bước bằng nhau)
# # # #         # Model sẽ học tập trung vào bước nào khó hơn
# # # #         self.raw_step_weights = nn.Parameter(torch.zeros(pred_len))

# # # #         # ── Encoder (giống STTrans gốc) ───────────────────────────────────
# # # #         self.encoder  = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
# # # #         self.ctx_proj = nn.Sequential(
# # # #             nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
# # # #             nn.LayerNorm(d_model))
# # # #         self.obs_enc  = ObsKinematicEncoder(
# # # #             d_model=d_model, nhead=nhead,
# # # #             num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout)

# # # #         # ── Decoder (giống STTrans gốc) ───────────────────────────────────
# # # #         self.horizon_queries = nn.Parameter(
# # # #             torch.randn(1, pred_len, d_model) * 0.02)
# # # #         self.dec_pe = SinusoidalPE(d_model, max_len=pred_len + 10)
# # # #         dec_layer = nn.TransformerDecoderLayer(
# # # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # # #             dropout=dropout, activation="relu", batch_first=True)
# # # #         self.transformer_dec = nn.TransformerDecoder(
# # # #             dec_layer, num_layers=num_dec_layers)
# # # #         self.reg_head = nn.Sequential(
# # # #             nn.Linear(d_model, d_model), nn.ReLU(),
# # # #             nn.Linear(d_model, 2))

# # # #         # ── Hard-only modules ──────────────────────────────────────────────
# # # #         self.steering_gate = PhysicsSteeringGate(
# # # #             ctx_dim=d_model, steer_dim=2,
# # # #             hidden=gate_hidden, pred_len=pred_len)
# # # #         self.recurv_head = RecurvatureTimingHead(
# # # #             ctx_dim=d_model, pred_len=pred_len, hidden=recurv_hidden)

# # # #         # ── Uncertainty Weighting — 9 tasks tự học (FIX3, FIX4, FIX5, FIX6) ─
# # # #         self.uw = UncertaintyWeighting(_UW_TASKS, _UW_INIT_LOG_SIGMA)

# # # #         self._init_weights()

# # # #     def _init_weights(self):
# # # #         for m in self.modules():
# # # #             if isinstance(m, nn.Linear):
# # # #                 nn.init.xavier_uniform_(m.weight, gain=0.5)
# # # #                 if m.bias is not None:
# # # #                     nn.init.zeros_(m.bias)
# # # #         # Gate: khởi đầu thiên về learned prediction (α ≈ 0.88)
# # # #         nn.init.constant_(self.steering_gate.gate_net[-1].bias, 2.0)
# # # #         nn.init.zeros_(self.steering_gate.gate_net[-1].weight)

# # # #     @property
# # # #     def step_weights(self) -> torch.Tensor:
# # # #         """
# # # #         [pred_len] — SOFTMAX (FIX1: không collapse)
# # # #         sum(sw) = T luôn luôn
# # # #         Optimizer học PHÂN PHỐI importance qua các bước, không scale tuyệt đối.
# # # #         """
# # # #         return F.softmax(self.raw_step_weights, dim=0)  # sum=1

# # # #     def set_thresholds(self, threshold_curv: float, threshold_spd: float):
# # # #         self.threshold_curv = threshold_curv
# # # #         self.threshold_spd  = threshold_spd

# # # #     # ── Encode: 1 lần duy nhất ───────────────────────────────────────────

# # # #     def _encode(self, batch_list) -> Tuple[torch.Tensor, torch.Tensor]:
# # # #         """→ (learned_pred [B,H,2], ctx_pooled [B,d])"""
# # # #         obs  = batch_list[0]
# # # #         B    = obs.shape[1]
# # # #         raw  = self.encoder(batch_list)
# # # #         ctok = self.ctx_proj(raw).unsqueeze(1)
# # # #         omem = self.obs_enc(obs)
# # # #         fmem = torch.cat([ctok, omem], 1)
# # # #         ctx  = fmem.mean(1)
# # # #         Q    = self.dec_pe(self.horizon_queries.expand(B, -1, -1))
# # # #         D    = self.transformer_dec(Q, fmem)
# # # #         lp   = self.reg_head(D)   # [B, H, 2]
# # # #         return lp, ctx

# # # #     # ── Gate: chỉ nhận learned_pred đã DETACH (FIX2) ────────────────────

# # # #     def _apply_gate(self,
# # # #                     lp_detached: torch.Tensor,   # [B_h, H, 2] — ĐÃ DETACH
# # # #                     ctx_h:       torch.Tensor,   # [B_h, d]
# # # #                     steer:       Optional[torch.Tensor],
# # # #                     obs_h:       torch.Tensor    # [T_obs, B_h, 2]
# # # #                     ) -> Tuple[torch.Tensor, torch.Tensor]:
# # # #         """
# # # #         Gradient từ L_hard chỉ đi: final_pred → α → gate → ctx_h → encoder
# # # #         KHÔNG đi vào decoder vì lp_detached đã detach.
# # # #         """
# # # #         B_h = lp_detached.shape[0]
# # # #         dev = lp_detached.device

# # # #         if steer is None:
# # # #             steer = (obs_h[-1] - obs_h[-2]
# # # #                      if obs_h.shape[0] >= 2
# # # #                      else torch.zeros(B_h, 2, device=dev))

# # # #         # Physics trajectory: persistence of steering
# # # #         cur   = obs_h[-1].clone()
# # # #         phys  = []
# # # #         for _ in range(self.pred_len):
# # # #             cur = cur + steer
# # # #             phys.append(cur.clone())
# # # #         physics = torch.stack(phys, dim=1)   # [B_h, H, 2]

# # # #         # Blend per step
# # # #         finals, alphas = [], []
# # # #         for t in range(self.pred_len):
# # # #             a = self.steering_gate(ctx_h, steer, t)   # [B_h, 1]
# # # #             # Gradient: a → gate → ctx_h → encoder ✓
# # # #             # lp_detached[:,t] không có gradient → decoder protected ✓
# # # #             finals.append(a * lp_detached[:, t] + (1 - a) * physics[:, t])
# # # #             alphas.append(a.mean())

# # # #         return torch.stack(finals, dim=1), torch.stack(alphas).mean()

# # # #     # ── Easy raw losses ───────────────────────────────────────────────────

# # # #     def _easy_losses(self, pred: torch.Tensor,
# # # #                      gt: torch.Tensor) -> Dict[str, torch.Tensor]:
# # # #         """pred, gt: [T, B_e, 2] → raw losses (chưa UW-weighted)"""
# # # #         T = min(pred.shape[0], gt.shape[0])
# # # #         p, g = pred[:T], gt[:T]

# # # #         l_dpe = haversine_km(_norm_to_deg(p), _norm_to_deg(g)).mean()
# # # #         l_mse = F.mse_loss(p, g)

# # # #         if T >= 2:
# # # #             sd      = (p[1:] - p[:-1]).norm(-1)
# # # #             l_speed = F.relu(sd - self.v_max_norm).pow(2).mean()
# # # #         else:
# # # #             l_speed = p.new_zeros(())

# # # #         if T >= 3:
# # # #             v       = p[1:] - p[:-1]
# # # #             l_accel = (v[1:].norm(-1) - v[:-1].norm(-1)).pow(2).mean()
# # # #         else:
# # # #             l_accel = p.new_zeros(())

# # # #         return {"dpe": l_dpe, "mse": l_mse,
# # # #                 "speed": l_speed, "accel": l_accel}

# # # #     # ── Hard raw losses ───────────────────────────────────────────────────

# # # #     def _hard_losses(self,
# # # #                      final_pred: torch.Tensor,   # [B_h, H, 2]
# # # #                      gt:         torch.Tensor,   # [T_pred, B_h, 2]
# # # #                      ctx_h:      torch.Tensor,   # [B_h, d]
# # # #                      obs_h:      torch.Tensor,   # [T_obs, B_h, 2]
# # # #                      alpha_mean: torch.Tensor,
# # # #                      ) -> Dict[str, torch.Tensor]:
# # # #         """→ raw losses cho hard samples"""
# # # #         T    = min(final_pred.shape[1], gt.shape[0])
# # # #         fp   = final_pred[:, :T]           # [B_h, T, 2]
# # # #         g    = gt[:T].permute(1, 0, 2)    # [B_h, T, 2]
# # # #         fp_t = fp.permute(1, 0, 2)        # [T, B_h, 2]
# # # #         gt_t = g.permute(1, 0, 2)         # [T, B_h, 2]

# # # #         # L_DPE_weighted: dot(sw_norm, hav_per_step)
# # # #         # Gradient: raw_w[t] += sw[t] * (hav_mean[t] - l_dpe_w)
# # # #         # → sw[t] tăng khi hav_mean[t] > weighted_avg → attention đúng
# # # #         hav          = haversine_km(_norm_to_deg(fp_t), _norm_to_deg(gt_t))  # [T, B_h]
# # # #         hav_per_step = hav.mean(dim=1)               # [T] — avg qua batch
# # # #         sw_norm      = self.step_weights[:T]         # [T], softmax sum<=1
# # # #         sw_norm      = sw_norm / (sw_norm.sum() + 1e-8)  # renorm nếu T<pred_len
# # # #         l_dpe_w      = (sw_norm * hav_per_step).sum()    # dot product

# # # #         # L_speed, L_accel
# # # #         if T >= 2:
# # # #             sd      = (fp_t[1:] - fp_t[:-1]).norm(-1)
# # # #             l_speed = F.relu(sd - self.v_max_norm).pow(2).mean()
# # # #         else:
# # # #             l_speed = fp.new_zeros(())

# # # #         if T >= 3:
# # # #             v       = fp_t[1:] - fp_t[:-1]
# # # #             l_accel = (v[1:].norm(-1) - v[:-1].norm(-1)).pow(2).mean()
# # # #         else:
# # # #             l_accel = fp.new_zeros(())

# # # #         # L_heading: cosine similarity giữa pred_dir và gt_dir
# # # #         if T >= 2:
# # # #             pd = fp_t[1:] - fp_t[:-1]     # [T-1, B_h, 2]
# # # #             gd = gt_t[1:] - gt_t[:-1]
# # # #             pn = F.normalize(pd, dim=-1, eps=1e-8)
# # # #             gn = F.normalize(gd, dim=-1, eps=1e-8)
# # # #             l_heading = ((1.0 - (pn * gn).sum(-1).clamp(-1, 1)) / 2.0).mean()
# # # #         else:
# # # #             l_heading = fp.new_zeros(())

# # # #         # L_recurv: auxiliary cross-entropy trên timing head
# # # #         recurv_logits = self.recurv_head(ctx_h)   # [B_h, pred_len+1]
# # # #         try:
# # # #             # gt:   [T_pred, B_h, 2] — đúng shape cho make_label
# # # #             # obs_h: [T_obs,  B_h, 2] — đúng shape cho make_label
# # # #             label = RecurvatureTimingHead.make_label(
# # # #                 gt, obs_h, self.recurv_threshold)
# # # #             l_recurv = (F.cross_entropy(recurv_logits, label)
# # # #                         if label.shape[0] == recurv_logits.shape[0]
# # # #                         else fp.new_zeros(()))
# # # #         except Exception:
# # # #             l_recurv = fp.new_zeros(())

# # # #         return {
# # # #             "dpe_w":   l_dpe_w,
# # # #             "speed":   l_speed,
# # # #             "accel":   l_accel,
# # # #             "heading": l_heading,
# # # #             "recurv":  l_recurv,
# # # #             "alpha":   alpha_mean,
# # # #         }

# # # #     # ── get_loss_breakdown ────────────────────────────────────────────────

# # # #     def get_loss_breakdown(self, batch_list) -> Dict:
# # # #         obs_traj = batch_list[0]   # [T_obs, B, 2]
# # # #         traj_gt  = batch_list[1]   # [T_pred, B, 2]
# # # #         B        = obs_traj.shape[1]

# # # #         # 1. Classify easy/hard (no grad)
# # # #         is_hard = classify_hard_obs(obs_traj, self.threshold_curv,
# # # #                                     self.threshold_spd)
# # # #         is_easy = ~is_hard
# # # #         n_easy  = int(is_easy.sum().item())
# # # #         n_hard  = int(is_hard.sum().item())

# # # #         # 2. Encode 1 lần — shared
# # # #         learned_pred, ctx_pooled = self._encode(batch_list)
# # # #         # learned_pred: [B, H, 2] có gradient đến decoder

# # # #         result = {"n_easy": n_easy, "n_hard": n_hard}

# # # #         # ── 3. EASY LOSS ─────────────────────────────────────────────────
# # # #         # Gradient đi bình thường: L_easy → learned_pred → decoder + encoder
# # # #         L_easy = None
# # # #         if n_easy > 0:
# # # #             lp_e = learned_pred[is_easy].permute(1, 0, 2)  # [T, B_e, 2]
# # # #             gt_e = traj_gt[:, is_easy, :]                   # [T, B_e, 2]

# # # #             el = self._easy_losses(lp_e, gt_e)

# # # #             # TẤT CẢ qua UW — không có anchor cứng (FIX3)
# # # #             L_easy = (
# # # #                 self.uw.weight("dpe_easy",  el["dpe"])
# # # #                 + self.uw.weight("mse",     el["mse"])
# # # #                 + self.uw.weight("speed",  el["speed"])
# # # #                 + self.uw.weight("accel",  el["accel"])
# # # #             )
# # # #             result.update({
# # # #                 "easy_dpe":   el["dpe"].item(),
# # # #                 "easy_mse":   el["mse"].item(),
# # # #                 "easy_speed": el["speed"].item(),
# # # #                 "easy_accel": el["accel"].item(),
# # # #             })

# # # #         # ── 4. HARD LOSS ──────────────────────────────────────────────────
# # # #         # FIX2: lp_h.detach() → decoder KHÔNG nhận gradient từ L_hard
# # # #         # Gradient: L_hard → final_pred → gate(α) → ctx_h → encoder
# # # #         L_hard = None
# # # #         if n_hard > 0:
# # # #             lp_h  = learned_pred[is_hard]        # [B_h, H, 2]
# # # #             gt_h  = traj_gt[:, is_hard, :]       # [T_pred, B_h, 2]
# # # #             obs_h = obs_traj[:, is_hard, :]      # [T_obs, B_h, 2]
# # # #             ctx_h = ctx_pooled[is_hard]          # [B_h, d]

# # # #             data3d = None
# # # #             if (len(batch_list) > 2
# # # #                     and isinstance(batch_list[2], torch.Tensor)
# # # #                     and batch_list[2].dim() == 4):
# # # #                 data3d = batch_list[2][is_hard]

# # # #             steer = _get_steering(data3d, obs_h)

# # # #             # KEY: lp_h.detach() bảo vệ decoder
# # # #             final_h, alpha_mean = self._apply_gate(
# # # #                 lp_h.detach(), ctx_h, steer, obs_h)

# # # #             hl = self._hard_losses(final_h, gt_h, ctx_h, obs_h, alpha_mean)

# # # #             # TẤT CẢ qua UW (FIX3), σ tách riêng easy/hard (FIX6)
# # # #             L_hard = (
# # # #                 self.uw.weight("dpe_hard",   hl["dpe_w"])
# # # #                 + self.uw.weight("speed",  hl["speed"])
# # # #                 + self.uw.weight("accel",  hl["accel"])
# # # #                 + self.uw.weight("heading",    hl["heading"])
# # # #                 + self.uw.weight("recurv",     hl["recurv"])
# # # #             )
# # # #             result.update({
# # # #                 "hard_dpe":    hl["dpe_w"].item(),
# # # #                 "hard_heading": (hl["heading"].item()
# # # #                                  if isinstance(hl["heading"], torch.Tensor)
# # # #                                  else 0.0),
# # # #                 "hard_recurv":  (hl["recurv"].item()
# # # #                                  if isinstance(hl["recurv"], torch.Tensor)
# # # #                                  else 0.0),
# # # #                 "alpha_mean":   (hl["alpha"].item()
# # # #                                  if isinstance(hl["alpha"], torch.Tensor)
# # # #                                  else 0.0),
# # # #                 "sw_min": self.step_weights.min().item(),
# # # #                 "sw_max": self.step_weights.max().item(),
# # # #                 "sw72":   self.step_weights[-1].item(),
# # # #             })

# # # #         # 5. Combine: UW tự cân bằng easy/hard (FIX5: bỏ w_easy_boost)
# # # #         # Không cần boost cứng vì UW σ_dpe_easy sẽ tự điều chỉnh
# # # #         if L_easy is not None and L_hard is not None:
# # # #             total = L_easy + L_hard
# # # #         elif L_easy is not None:
# # # #             total = L_easy
# # # #         else:
# # # #             total = L_hard

# # # #         result["total"] = total

# # # #         # Log σ để monitor
# # # #         result.update(self.uw.sigma_dict())

# # # #         # 6. ADE metrics (no grad)
# # # #         lp_perm = learned_pred.permute(1, 0, 2)   # [H, B, 2]
# # # #         with torch.no_grad():
# # # #             result.update(compute_ade_per_horizon(lp_perm.detach(), traj_gt))
# # # #             result.update(compute_ate_cte_per_horizon(lp_perm.detach(), traj_gt))

# # # #         return result

# # # #     def get_loss(self, batch_list) -> torch.Tensor:
# # # #         return self.get_loss_breakdown(batch_list)["total"]

# # # #     def forward(self, batch_list) -> torch.Tensor:
# # # #         """Inference: decoder trực tiếp, không gate"""
# # # #         lp, _ = self._encode(batch_list)
# # # #         return lp.permute(1, 0, 2)   # [H, B, 2]

# # # #     @torch.no_grad()
# # # #     def sample(self, batch_list, **kwargs):
# # # #         pred    = self.forward(batch_list)
# # # #         T, B, _ = pred.shape
# # # #         return pred, torch.zeros(T, B, 2, device=pred.device), pred.unsqueeze(0)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Factory
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def build_st_trans_v2(args, threshold_curv=15.0, threshold_spd=0.5):
# # # #     return STTransV2(
# # # #         obs_len          = getattr(args, 'obs_len',          8),
# # # #         pred_len         = getattr(args, 'pred_len',         12),
# # # #         unet_in_ch       = getattr(args, 'unet_in_ch',       13),
# # # #         d_model          = getattr(args, 'd_model',          64),
# # # #         nhead            = getattr(args, 'nhead',            4),
# # # #         num_enc_layers   = getattr(args, 'num_enc_layers',   1),
# # # #         num_dec_layers   = getattr(args, 'num_dec_layers',   3),
# # # #         dim_ff           = getattr(args, 'dim_ff',           512),
# # # #         dropout          = getattr(args, 'dropout',          0.1),
# # # #         v_max_kmh        = getattr(args, 'v_max_kmh',        80.0),
# # # #         recurv_threshold = getattr(args, 'recurv_threshold', 45.0),
# # # #         gate_hidden      = getattr(args, 'gate_hidden',      32),
# # # #         recurv_hidden    = getattr(args, 'recurv_hidden',    64),
# # # #         threshold_curv   = threshold_curv,
# # # #         threshold_spd    = threshold_spd,
# # # #     )
# # # """
# # # Model/st_trans_v2_model.py  —— ST-Trans v2
# # # ============================================
# # # Mở rộng ST-Trans (Faiaz 2026) với:

# # #   [1] EASY/HARD CLASSIFICATION
# # #       Easy (~61%): bão thẳng, không recurvature
# # #       Hard (~39%): bão recurvature, speed biến động mạnh

# # #   [2] THIẾT KẾ TRAINING:
# # #       Easy path = ST-Trans gốc HOÀN TOÀN
# # #         L_easy = L_DPE + w_mse*MSE + w_speed*speed + w_accel*accel
# # #         Gradient: L_easy → decoder + encoder (đầy đủ, không bị chia)

# # #       Hard path = Physics Steering Gate
# # #         final_h = α*lp_h + (1-α)*physics_persistence
# # #         L_hard  = L_DPE(final_h) + λ_head*L_heading(final_h)
# # #                 + λ_rec*L_recurv(ctx_h)
# # #         lp_h.detach()  → decoder KHÔNG nhận gradient từ L_hard
# # #         ctx_det.detach() trong gate → encoder KHÔNG bị gate gradient
# # #         ctx_h (gốc) vào recurv_head → encoder học classify recurvature

# # #   [3] INFERENCE (KEY):
# # #       Easy samples → learned_pred trực tiếp (= ST-Trans gốc)
# # #       Hard samples → gate-blended prediction (tốt hơn ST-Trans)
# # #       → easy_ADE ≈ ST-Trans gốc, hard_ADE tốt hơn nhờ gate

# # #   [4] GRADIENT FLOW:
# # #       Decoder: chỉ từ easy  (decoder học như ST-Trans gốc)
# # #       Encoder: easy (dominant) + recurv classification (auxiliary)
# # #       Gate:    học từ ctx.detach() + steer + step_emb

# # #   [5] UW (Kendall 2018): 4 tasks — dpe_easy, dpe_hard, heading, recurv
# # #       MSE/speed/accel: fixed weights như ST-Trans gốc
# # # """
# # # from __future__ import annotations
# # # import math
# # # from typing import Dict, List, Optional, Tuple

# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F

# # # from Model.paper_baseline_model import (
# # #     PaperEncoder,
# # #     _norm_to_deg,
# # #     haversine_km,
# # #     compute_ade_per_horizon,
# # #     compute_ate_cte_per_horizon,
# # #     HORIZON_STEPS,
# # # )

# # # # ERA5 channel indices
# # # _U500_CH         = 0
# # # _V500_CH         = 1
# # # _DEG_SCALE       = 25.0
# # # _MS_TO_NORM_PER_6H = (6.0 * 3600.0) / (111320.0 * _DEG_SCALE)


# # # # ══════════════════════════════════════════════════════════════════════
# # # #  Uncertainty Weighting (Kendall et al. 2018)
# # # # ══════════════════════════════════════════════════════════════════════

# # # class UncertaintyWeighting(nn.Module):
# # #     """
# # #     L_i_weighted = L_i / (2σ_i²) + log(σ_i)
# # #     σ_i = exp(log_σ_i) — learnable, luôn dương

# # #     4 tasks: dpe_easy, dpe_hard, heading, recurv
# # #     MSE/speed/accel không qua UW (dùng fixed weights như ST-Trans gốc)
# # #     """
# # #     TASKS = ["dpe_easy", "dpe_hard", "heading", "recurv"]
# # #     INIT  = {"dpe_easy": -0.5, "dpe_hard": -0.5,
# # #               "heading": 0.3,  "recurv": 0.7}

# # #     def __init__(self):
# # #         super().__init__()
# # #         inits = torch.tensor([self.INIT[t] for t in self.TASKS])
# # #         self.log_sigma = nn.Parameter(inits)
# # #         self._idx = {t: i for i, t in enumerate(self.TASKS)}

# # #     def weight(self, name: str, loss: torch.Tensor) -> torch.Tensor:
# # #         s = self.log_sigma[self._idx[name]]
# # #         return torch.exp(-2.0 * s) * loss / 2.0 + s

# # #     def sigma_dict(self) -> Dict[str, float]:
# # #         s = torch.exp(self.log_sigma).detach()
# # #         return {f"σ_{t}": s[i].item() for i, t in enumerate(self.TASKS)}


# # # # ══════════════════════════════════════════════════════════════════════
# # # #  Shared sub-modules (giống ST-Trans gốc)
# # # # ══════════════════════════════════════════════════════════════════════

# # # class SinusoidalPE(nn.Module):
# # #     def __init__(self, d_model: int, max_len: int = 300):
# # #         super().__init__()
# # #         pe  = torch.zeros(max_len, d_model)
# # #         pos = torch.arange(max_len).unsqueeze(1).float()
# # #         div = torch.exp(torch.arange(0, d_model, 2).float() *
# # #                         (-math.log(10000.0) / d_model))
# # #         pe[:, 0::2] = torch.sin(pos * div)
# # #         pe[:, 1::2] = torch.cos(pos * div)
# # #         self.register_buffer("pe", pe.unsqueeze(0))

# # #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# # #         return x + self.pe[:, :x.size(1)]


# # # class ObsKinematicEncoder(nn.Module):
# # #     """obs_traj [T,B,2] → [B,T,d_model]. Giống ST-Trans gốc."""
# # #     FEAT_DIM = 8

# # #     def __init__(self, d_model=64, nhead=4, num_layers=1,
# # #                  dim_ff=256, dropout=0.1):
# # #         super().__init__()
# # #         self.proj = nn.Sequential(
# # #             nn.Linear(self.FEAT_DIM, d_model), nn.ReLU(),
# # #             nn.Linear(d_model, d_model))
# # #         self.pe  = SinusoidalPE(d_model, max_len=64)
# # #         enc      = nn.TransformerEncoderLayer(
# # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # #             dropout=dropout, activation="relu", batch_first=True)
# # #         self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)

# # #     @staticmethod
# # #     def _feats(obs: torch.Tensor) -> torch.Tensor:
# # #         T, B, _ = obs.shape
# # #         dev = obs.device
# # #         lon, lat = obs[:, :, 0], obs[:, :, 1]
# # #         dl = torch.cat([obs[1:,:,0] - obs[:-1,:,0],
# # #                         torch.zeros(1, B, device=dev)], 0)
# # #         dt = torch.cat([obs[1:,:,1] - obs[:-1,:,1],
# # #                         torch.zeros(1, B, device=dev)], 0) if T >= 2 \
# # #              else torch.zeros_like(lon)
# # #         ddl = torch.cat([dl[1:] - dl[:-1], torch.zeros(1, B, device=dev)], 0) \
# # #               if T >= 3 else torch.zeros_like(lon)
# # #         ddt = torch.cat([dt[1:] - dt[:-1], torch.zeros(1, B, device=dev)], 0) \
# # #               if T >= 3 else torch.zeros_like(lon)
# # #         si  = torch.linspace(0, 1, T, device=dev).unsqueeze(1).expand(T, B)
# # #         spd = (dl**2 + dt**2).sqrt()
# # #         return torch.stack([lon, lat, dl, dt, ddl, ddt, si, spd], -1).permute(1, 0, 2)

# # #     def forward(self, obs: torch.Tensor) -> torch.Tensor:
# # #         return self.enc(self.pe(self.proj(self._feats(obs))))


# # # # ══════════════════════════════════════════════════════════════════════
# # # #  Easy/Hard classifier
# # # # ══════════════════════════════════════════════════════════════════════

# # # def classify_hard_obs(obs_traj: torch.Tensor,
# # #                       threshold_curv: float,
# # #                       threshold_spd: float) -> torch.Tensor:
# # #     """[T_obs, B, 2] → [B] bool. Hard = recurvature hoặc erratic speed."""
# # #     T, B, _ = obs_traj.shape
# # #     device  = obs_traj.device
# # #     with torch.no_grad():
# # #         if T < 2:
# # #             return torch.zeros(B, dtype=torch.bool, device=device)
# # #         vel     = obs_traj[1:] - obs_traj[:-1]
# # #         spd     = vel.norm(dim=-1)
# # #         spd_cv  = spd.std(0) / (spd.mean(0) + 1e-6)
# # #         if T >= 3:
# # #             vn   = F.normalize(vel, dim=-1, eps=1e-8)
# # #             cos  = (vn[1:] * vn[:-1]).sum(-1).clamp(-1, 1)
# # #             curv = (torch.acos(cos) * (180 / math.pi)).mean(0)
# # #         else:
# # #             curv = torch.zeros(B, device=device)
# # #         return (curv > threshold_curv) | (spd_cv > threshold_spd)


# # # def compute_hard_thresholds(train_loader, device,
# # #                              percentile: float = 70.0) -> Tuple[float, float]:
# # #     """1 pass qua train set → (threshold_curv°, threshold_spd)."""
# # #     all_curv, all_spd = [], []
# # #     with torch.no_grad():
# # #         for batch in train_loader:
# # #             obs = batch[0].to(device)
# # #             T, B, _ = obs.shape
# # #             if T < 2:
# # #                 continue
# # #             vel = obs[1:] - obs[:-1]
# # #             spd = vel.norm(dim=-1)
# # #             all_spd.extend((spd.std(0) / (spd.mean(0) + 1e-6)).cpu().tolist())
# # #             if T >= 3:
# # #                 vn  = F.normalize(vel, dim=-1, eps=1e-8)
# # #                 cos = (vn[1:] * vn[:-1]).sum(-1).clamp(-1, 1)
# # #                 all_curv.extend(
# # #                     (torch.acos(cos) * (180 / math.pi)).mean(0).cpu().tolist())
# # #     if not all_curv:
# # #         return 15.0, 0.5
# # #     import numpy as np
# # #     return (float(np.percentile(all_curv, percentile)),
# # #             float(np.percentile(all_spd,  percentile)))


# # # # ══════════════════════════════════════════════════════════════════════
# # # #  Physics Steering Gate
# # # # ══════════════════════════════════════════════════════════════════════

# # # class PhysicsSteeringGate(nn.Module):
# # #     """
# # #     α[t] = sigmoid(gate_net([ctx, step_emb[t], steer]))
# # #     final[t] = α[t]*learned[t] + (1-α[t])*physics[t]

# # #     Vectorized: 1 forward pass cho tất cả T bước.
# # #     Init bias=2.0 → α≈0.88 ban đầu (lean về learned).
# # #     """
# # #     def __init__(self, ctx_dim=64, steer_dim=2, hidden=32, pred_len=12):
# # #         super().__init__()
# # #         self.pred_len = pred_len
# # #         self.step_emb = nn.Embedding(pred_len, 16)
# # #         self.gate_net = nn.Sequential(
# # #             nn.Linear(ctx_dim + 16 + steer_dim, hidden),
# # #             nn.ReLU(),
# # #             nn.Linear(hidden, 1))
# # #         nn.init.constant_(self.gate_net[-1].bias, 2.0)
# # #         nn.init.zeros_(self.gate_net[-1].weight)


# # # class RecurvatureTimingHead(nn.Module):
# # #     """Auxiliary head: predict bước nào bão recurve. [B,H] → [B, H+1]."""
# # #     def __init__(self, ctx_dim=64, pred_len=12, hidden=64):
# # #         super().__init__()
# # #         self.clf = nn.Sequential(
# # #             nn.Linear(ctx_dim, hidden), nn.GELU(),
# # #             nn.Dropout(0.1),
# # #             nn.Linear(hidden, pred_len + 1))

# # #     def forward(self, ctx: torch.Tensor) -> torch.Tensor:
# # #         return self.clf(ctx)

# # #     @staticmethod
# # #     def make_label(gt: torch.Tensor, obs: torch.Tensor,
# # #                    threshold_deg: float = 45.0) -> torch.Tensor:
# # #         """gt [T,B,2], obs [T_obs,B,2] → label [B] long."""
# # #         T, B, _ = gt.shape
# # #         dev   = gt.device
# # #         label = torch.zeros(B, dtype=torch.long, device=dev)
# # #         full  = torch.cat([obs[-1:], gt], 0)
# # #         with torch.no_grad():
# # #             for t in range(T - 1):
# # #                 d_in  = full[t+1] - full[t]
# # #                 d_out = full[t+2] - full[t+1]
# # #                 ni = F.normalize(d_in,  dim=-1, eps=1e-8)
# # #                 no = F.normalize(d_out, dim=-1, eps=1e-8)
# # #                 ang  = torch.acos((ni*no).sum(-1).clamp(-1,1)) * (180/math.pi)
# # #                 mask = (ang > threshold_deg) & (label == 0)
# # #                 label[mask] = t + 1
# # #         return label


# # # def _get_steering(data3d: Optional[torch.Tensor],
# # #                   obs_h: torch.Tensor) -> Optional[torch.Tensor]:
# # #     """ERA5 center pixel → [B,2] normalised displacement. None nếu không có."""
# # #     if data3d is None or not isinstance(data3d, torch.Tensor) \
# # #             or data3d.dim() != 4:
# # #         return None
# # #     B, C, H, W = data3d.shape
# # #     if C <= max(_U500_CH, _V500_CH):
# # #         return None
# # #     cy, cx  = H // 2, W // 2
# # #     u_ms    = data3d[:, _U500_CH, cy, cx]
# # #     v_ms    = data3d[:, _V500_CH, cy, cx]
# # #     lat_deg = obs_h[-1, :, 1] * _DEG_SCALE
# # #     cos_lat = torch.cos(lat_deg * (math.pi / 180)).clamp(0.1, 1.0)
# # #     return torch.stack([(u_ms / cos_lat) * _MS_TO_NORM_PER_6H,
# # #                         v_ms * _MS_TO_NORM_PER_6H], dim=-1)


# # # # ══════════════════════════════════════════════════════════════════════
# # # #  STTransV2 — Main Model
# # # # ══════════════════════════════════════════════════════════════════════

# # # class STTransV2(nn.Module):
# # #     """
# # #     ST-Trans v2: Easy/Hard split + Physics Gate at inference.

# # #     Training:
# # #       Easy: L_DPE + L_MSE + L_speed + L_accel (= ST-Trans gốc)
# # #       Hard: L_DPE(gate_out) + L_heading(gate_out) + L_recurv(ctx)
# # #             lp_h.detach() → decoder protected
# # #             ctx.detach() in gate → encoder protected from gate gradient

# # #     Inference:
# # #       Easy: decoder output trực tiếp (= ST-Trans gốc)
# # #       Hard: gate-blended output → ADE tốt hơn
# # #     """

# # #     def __init__(
# # #         self,
# # #         obs_len=8, pred_len=12, unet_in_ch=13,
# # #         d_model=64, nhead=4,
# # #         num_enc_layers=1, num_dec_layers=3,
# # #         dim_ff=512, dropout=0.1,
# # #         # Physics loss weights (như ST-Trans gốc)
# # #         w_mse=0.05, lambda_speed=0.1, lambda_accel=0.01,
# # #         v_max_kmh=80.0, dt_h=6.0,
# # #         # Hard extras
# # #         recurv_threshold=45.0,
# # #         gate_hidden=32, recurv_hidden=64,
# # #         # Thresholds (set sau compute_hard_thresholds)
# # #         threshold_curv=15.0, threshold_spd=0.5,
# # #     ):
# # #         super().__init__()
# # #         self.obs_len          = obs_len
# # #         self.pred_len         = pred_len
# # #         self.w_mse            = w_mse
# # #         self.lambda_speed     = lambda_speed
# # #         self.lambda_accel     = lambda_accel
# # #         self.v_max_norm       = v_max_kmh * dt_h / (111.0 * _DEG_SCALE)
# # #         self.recurv_threshold = recurv_threshold
# # #         self.threshold_curv   = threshold_curv
# # #         self.threshold_spd    = threshold_spd

# # #         # ── Encoder (= ST-Trans gốc) ──────────────────────────────────
# # #         self.encoder  = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
# # #         self.ctx_proj = nn.Sequential(
# # #             nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
# # #             nn.LayerNorm(d_model))
# # #         self.obs_enc  = ObsKinematicEncoder(
# # #             d_model=d_model, nhead=nhead,
# # #             num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout)

# # #         # ── Decoder (= ST-Trans gốc) ──────────────────────────────────
# # #         self.horizon_queries = nn.Parameter(
# # #             torch.randn(1, pred_len, d_model) * 0.02)
# # #         self.dec_pe = SinusoidalPE(d_model, max_len=pred_len + 10)
# # #         dec_layer = nn.TransformerDecoderLayer(
# # #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# # #             dropout=dropout, activation="relu", batch_first=True)
# # #         self.transformer_dec = nn.TransformerDecoder(
# # #             dec_layer, num_layers=num_dec_layers)
# # #         self.reg_head = nn.Sequential(
# # #             nn.Linear(d_model, d_model), nn.ReLU(),
# # #             nn.Linear(d_model, 2))

# # #         # ── Hard-only modules ─────────────────────────────────────────
# # #         self.steering_gate = PhysicsSteeringGate(
# # #             ctx_dim=d_model, steer_dim=2,
# # #             hidden=gate_hidden, pred_len=pred_len)
# # #         self.recurv_head   = RecurvatureTimingHead(
# # #             ctx_dim=d_model, pred_len=pred_len, hidden=recurv_hidden)

# # #         # ── UW (4 tasks only) ─────────────────────────────────────────
# # #         self.uw = UncertaintyWeighting()

# # #         self._init_weights()

# # #     def _init_weights(self):
# # #         for m in self.modules():
# # #             if isinstance(m, nn.Linear):
# # #                 nn.init.xavier_uniform_(m.weight, gain=0.5)
# # #                 if m.bias is not None:
# # #                     nn.init.zeros_(m.bias)
# # #         nn.init.constant_(self.steering_gate.gate_net[-1].bias, 2.0)
# # #         nn.init.zeros_(self.steering_gate.gate_net[-1].weight)

# # #     def set_thresholds(self, curv: float, spd: float):
# # #         self.threshold_curv = curv
# # #         self.threshold_spd  = spd

# # #     # ── Encode (shared, 1 lần) ────────────────────────────────────────

# # #     def _encode(self, batch_list) -> Tuple[torch.Tensor, torch.Tensor]:
# # #         """→ (learned_pred [B,H,2], ctx_pooled [B,d])"""
# # #         obs  = batch_list[0]
# # #         B    = obs.shape[1]
# # #         raw  = self.encoder(batch_list)
# # #         ctok = self.ctx_proj(raw).unsqueeze(1)          # [B,1,d]
# # #         omem = self.obs_enc(obs)                         # [B,T_obs,d]
# # #         fmem = torch.cat([ctok, omem], 1)               # [B,1+T_obs,d]
# # #         ctx  = fmem.mean(1)                              # [B,d]
# # #         Q    = self.dec_pe(self.horizon_queries.expand(B, -1, -1))
# # #         D    = self.transformer_dec(Q, fmem)             # [B,H,d]
# # #         lp   = self.reg_head(D)                          # [B,H,2]
# # #         return lp, ctx

# # #     # ── Gate blend (hard samples) ─────────────────────────────────────

# # #     def _gate_blend(self,
# # #                     lp_h:   torch.Tensor,          # [B_h, H, 2]
# # #                     ctx_h:  torch.Tensor,          # [B_h, d]
# # #                     steer:  Optional[torch.Tensor],
# # #                     obs_h:  torch.Tensor,          # [T_obs, B_h, 2]
# # #                     detach_ctx: bool = True,
# # #                     ) -> Tuple[torch.Tensor, torch.Tensor]:
# # #         """
# # #         Physics persistence blend:
# # #           physics[t] = obs_h[-1] + (t+1)*steer
# # #           alpha[t]   = sigmoid(gate([ctx, step_emb[t], steer]))
# # #           final[t]   = alpha[t]*lp_h[t] + (1-alpha[t])*physics[t]

# # #         detach_ctx=True  khi training → gate không update encoder
# # #         detach_ctx=False khi inference → alpha phản ánh đúng quality
# # #         """
# # #         B_h = lp_h.shape[0]
# # #         T   = self.pred_len
# # #         dev = lp_h.device

# # #         # Steer fallback
# # #         if steer is None:
# # #             steer = (obs_h[-1] - obs_h[-2]
# # #                      if obs_h.shape[0] >= 2
# # #                      else torch.zeros(B_h, 2, device=dev))

# # #         # Physics: [B_h, T, 2]
# # #         offsets = steer.unsqueeze(1) * torch.arange(
# # #             1, T+1, device=dev, dtype=steer.dtype).view(1, T, 1)
# # #         physics = obs_h[-1].unsqueeze(1) + offsets

# # #         # Gate (vectorized over T steps)
# # #         ctx_in  = ctx_h.detach() if detach_ctx else ctx_h   # STOP/PASS grad
# # #         t_idx   = torch.arange(T, device=dev)
# # #         s_emb   = self.steering_gate.step_emb(t_idx)        # [T, 16]
# # #         ctx_e   = ctx_in.unsqueeze(0).expand(T, -1, -1)     # [T, B_h, d]
# # #         steer_e = steer.unsqueeze(0).expand(T, -1, -1)      # [T, B_h, 2]
# # #         s_emb_e = s_emb.unsqueeze(1).expand(T, B_h, -1)    # [T, B_h, 16]
# # #         gate_in = torch.cat([ctx_e, s_emb_e, steer_e], -1) # [T, B_h, d+18]
# # #         alpha   = torch.sigmoid(
# # #             self.steering_gate.gate_net(gate_in))            # [T, B_h, 1]

# # #         # Blend → [B_h, T, 2]
# # #         alpha_t = alpha.permute(1, 0, 2)                    # [B_h, T, 1]
# # #         final_h = alpha_t * lp_h + (1 - alpha_t) * physics

# # #         return final_h, alpha.mean()

# # #     # ── Easy losses (= ST-Trans gốc) ─────────────────────────────────

# # #     def _easy_losses(self, pred: torch.Tensor,
# # #                      gt: torch.Tensor) -> Dict[str, torch.Tensor]:
# # #         """
# # #         pred, gt: [T, B_e, 2]
# # #         Giống hệt ST-Trans gốc: DPE uniform + MSE + speed + accel
# # #         """
# # #         T = min(pred.shape[0], gt.shape[0])
# # #         p, g = pred[:T], gt[:T]

# # #         # DPE: uniform mean (không sw, không attention)
# # #         l_dpe = haversine_km(_norm_to_deg(p), _norm_to_deg(g)).mean()
# # #         l_mse = F.mse_loss(p, g)

# # #         if T >= 2:
# # #             sd      = (p[1:] - p[:-1]).norm(-1)
# # #             l_speed = F.relu(sd - self.v_max_norm).pow(2).mean()
# # #         else:
# # #             l_speed = p.new_zeros(())

# # #         if T >= 3:
# # #             v       = p[1:] - p[:-1]
# # #             l_accel = (v[1:].norm(-1) - v[:-1].norm(-1)).pow(2).mean()
# # #         else:
# # #             l_accel = p.new_zeros(())

# # #         return {"dpe": l_dpe, "mse": l_mse,
# # #                 "speed": l_speed, "accel": l_accel}

# # #     # ── Hard losses (gate-corrected) ──────────────────────────────────

# # #     def _hard_losses(self,
# # #                      final_h:  torch.Tensor,   # [B_h, H, 2]
# # #                      gt_h:     torch.Tensor,   # [T_pred, B_h, 2]
# # #                      ctx_h:    torch.Tensor,   # [B_h, d] — gốc có gradient
# # #                      obs_h:    torch.Tensor,   # [T_obs, B_h, 2]
# # #                      ) -> Dict[str, torch.Tensor]:
# # #         """
# # #         final_h: gate-blended prediction (lp_h.detach() đã được apply trong gate)
# # #         ctx_h:   encoder context gốc (có gradient → recurv head update encoder)
# # #         """
# # #         T    = min(final_h.shape[1], gt_h.shape[0])
# # #         fp   = final_h[:, :T]                      # [B_h, T, 2]
# # #         gt   = gt_h[:T].permute(1, 0, 2)           # [B_h, T, 2]
# # #         fp_t = fp.permute(1, 0, 2)                 # [T, B_h, 2]
# # #         gt_t = gt.permute(1, 0, 2)                 # [T, B_h, 2]

# # #         # L_DPE: uniform mean (không sw)
# # #         l_dpe = haversine_km(
# # #             _norm_to_deg(fp_t), _norm_to_deg(gt_t)).mean()

# # #         # L_heading: cosine similarity của prediction direction
# # #         if T >= 2:
# # #             pd  = fp_t[1:] - fp_t[:-1]             # [T-1, B_h, 2]
# # #             gd  = gt_t[1:] - gt_t[:-1]
# # #             pn  = F.normalize(pd, dim=-1, eps=1e-8)
# # #             gn  = F.normalize(gd, dim=-1, eps=1e-8)
# # #             l_heading = ((1.0 - (pn*gn).sum(-1).clamp(-1,1)) / 2.0).mean()
# # #         else:
# # #             l_heading = fp.new_zeros(())

# # #         # L_recurv: ctx_h gốc → encoder học classify recurvature
# # #         recurv_logits = self.recurv_head(ctx_h)     # [B_h, pred_len+1]
# # #         try:
# # #             label    = RecurvatureTimingHead.make_label(
# # #                 gt_h, obs_h, self.recurv_threshold)
# # #             l_recurv = (F.cross_entropy(recurv_logits, label)
# # #                         if label.shape[0] == recurv_logits.shape[0]
# # #                         else fp.new_zeros(()))
# # #         except Exception:
# # #             l_recurv = fp.new_zeros(())

# # #         return {"dpe": l_dpe, "heading": l_heading, "recurv": l_recurv}

# # #     # ── get_loss_breakdown ────────────────────────────────────────────

# # #     def get_loss_breakdown(self, batch_list) -> Dict:
# # #         obs_traj = batch_list[0]   # [T_obs, B, 2]
# # #         traj_gt  = batch_list[1]   # [T_pred, B, 2]

# # #         # 1. Classify
# # #         is_hard = classify_hard_obs(obs_traj,
# # #                                     self.threshold_curv, self.threshold_spd)
# # #         is_easy = ~is_hard
# # #         n_easy  = int(is_easy.sum())
# # #         n_hard  = int(is_hard.sum())

# # #         # 2. Encode 1 lần — shared encoder + decoder
# # #         lp, ctx = self._encode(batch_list)  # [B,H,2], [B,d]

# # #         result = {"n_easy": n_easy, "n_hard": n_hard}

# # #         # ── 3. EASY LOSS = ST-TRANS GỐC ──────────────────────────────
# # #         # Gradient đầy đủ: L_easy → lp_e → reg_head → transformer_dec → encoder
# # #         L_easy = None
# # #         if n_easy > 0:
# # #             lp_e = lp[is_easy].permute(1, 0, 2)    # [T, B_e, 2]
# # #             gt_e = traj_gt[:, is_easy]              # [T, B_e, 2]
# # #             el   = self._easy_losses(lp_e, gt_e)

# # #             # DPE qua UW, MSE/speed/accel dùng fixed weights như ST-Trans gốc
# # #             L_easy = (
# # #                 self.uw.weight("dpe_easy", el["dpe"])
# # #                 + self.w_mse         * el["mse"]
# # #                 + self.lambda_speed  * el["speed"]
# # #                 + self.lambda_accel  * el["accel"]
# # #             )
# # #             result.update({
# # #                 "easy_dpe":   el["dpe"].item(),
# # #                 "easy_mse":   el["mse"].item(),
# # #             })

# # #         # ── 4. HARD LOSS = GATE-CORRECTED ────────────────────────────
# # #         # lp_h.detach()   → decoder KHÔNG nhận gradient từ L_hard
# # #         # ctx_h.detach() trong gate → encoder KHÔNG bị gate gradient
# # #         # ctx_h (gốc)    → recurv_head UPDATE encoder (auxiliary)
# # #         L_hard = None
# # #         if n_hard > 0:
# # #             lp_h  = lp[is_hard]                     # [B_h, H, 2]
# # #             gt_h  = traj_gt[:, is_hard]             # [T, B_h, 2]
# # #             obs_h = obs_traj[:, is_hard]            # [T_obs, B_h, 2]
# # #             ctx_h = ctx[is_hard]                    # [B_h, d]

# # #             # Lấy steering từ ERA5 nếu có
# # #             data3d = None
# # #             if (len(batch_list) > 2
# # #                     and isinstance(batch_list[2], torch.Tensor)
# # #                     and batch_list[2].dim() == 4):
# # #                 data3d = batch_list[2][is_hard]
# # #             steer = _get_steering(data3d, obs_h)

# # #             # Gate blend với lp_h.detach() + ctx_h.detach() trong gate
# # #             final_h, alpha_m = self._gate_blend(
# # #                 lp_h.detach(),      # decoder STOP
# # #                 ctx_h,              # _gate_blend sẽ detach ctx bên trong
# # #                 steer, obs_h,
# # #                 detach_ctx=True)    # encoder STOP from gate gradient

# # #             hl = self._hard_losses(final_h, gt_h, ctx_h, obs_h)

# # #             L_hard = (
# # #                 self.uw.weight("dpe_hard", hl["dpe"])
# # #                 + self.uw.weight("heading", hl["heading"])
# # #                 + self.uw.weight("recurv",  hl["recurv"])
# # #             )
# # #             result.update({
# # #                 "hard_dpe":    hl["dpe"].item(),
# # #                 "hard_heading":(hl["heading"].item()
# # #                                 if isinstance(hl["heading"], torch.Tensor)
# # #                                 else 0.0),
# # #                 "alpha_mean":  alpha_m.item()
# # #                                if isinstance(alpha_m, torch.Tensor)
# # #                                else 0.0,
# # #             })

# # #         # 5. Combine
# # #         if L_easy is not None and L_hard is not None:
# # #             total = L_easy + L_hard
# # #         elif L_easy is not None:
# # #             total = L_easy
# # #         else:
# # #             total = L_hard

# # #         result["total"] = total
# # #         result.update(self.uw.sigma_dict())

# # #         # 6. ADE/ATE/CTE metrics (no grad) — dùng learned_pred
# # #         lp_perm = lp.permute(1, 0, 2)   # [T, B, 2]
# # #         with torch.no_grad():
# # #             result.update(
# # #                 compute_ade_per_horizon(lp_perm.detach(), traj_gt))
# # #             result.update(
# # #                 compute_ate_cte_per_horizon(lp_perm.detach(), traj_gt))

# # #         return result

# # #     def get_loss(self, batch_list) -> torch.Tensor:
# # #         return self.get_loss_breakdown(batch_list)["total"]

# # #     # ── INFERENCE (KEY: gate applied to hard samples) ─────────────────

# # #     def forward(self, batch_list) -> torch.Tensor:
# # #         """
# # #         Easy samples → learned_pred trực tiếp (= ST-Trans gốc)
# # #         Hard samples → gate-blended (tốt hơn ST-Trans cho recurvature)

# # #         detach_ctx=False khi inference vì không cần backward
# # #         """
# # #         lp, ctx = self._encode(batch_list)          # [B,H,2], [B,d]
# # #         obs     = batch_list[0]                     # [T_obs, B, 2]

# # #         # Start với learned_pred cho tất cả
# # #         pred    = lp.permute(1, 0, 2).clone()       # [T, B, 2]

# # #         # Hard samples: override bằng gate-blended prediction
# # #         is_hard = classify_hard_obs(obs, self.threshold_curv,
# # #                                     self.threshold_spd)
# # #         if is_hard.any():
# # #             obs_h  = obs[:, is_hard]
# # #             ctx_h  = ctx[is_hard]
# # #             lp_h   = lp[is_hard]                   # [B_h, H, 2]

# # #             data3d = None
# # #             if (len(batch_list) > 2
# # #                     and isinstance(batch_list[2], torch.Tensor)
# # #                     and batch_list[2].dim() == 4):
# # #                 data3d = batch_list[2][is_hard]
# # #             steer = _get_steering(data3d, obs_h)

# # #             # No detach needed at inference (no backward)
# # #             final_h, _ = self._gate_blend(
# # #                 lp_h, ctx_h, steer, obs_h,
# # #                 detach_ctx=False)                   # pass full ctx

# # #             # [B_h, T, 2] → [T, B_h, 2]
# # #             pred[:, is_hard] = final_h.permute(1, 0, 2)

# # #         return pred   # [T, B, 2]

# # #     @torch.no_grad()
# # #     def sample(self, batch_list, **kwargs):
# # #         pred    = self.forward(batch_list)          # [T, B, 2]
# # #         T, B, _ = pred.shape
# # #         return pred, torch.zeros(T, B, 2, device=pred.device), pred.unsqueeze(0)


# # # # ══════════════════════════════════════════════════════════════════════
# # # #  Factory
# # # # ══════════════════════════════════════════════════════════════════════

# # # def build_st_trans_v2(args, threshold_curv=15.0, threshold_spd=0.5):
# # #     return STTransV2(
# # #         obs_len          = getattr(args, "obs_len",          8),
# # #         pred_len         = getattr(args, "pred_len",         12),
# # #         unet_in_ch       = getattr(args, "unet_in_ch",       13),
# # #         d_model          = getattr(args, "d_model",          64),
# # #         nhead            = getattr(args, "nhead",            4),
# # #         num_enc_layers   = getattr(args, "num_enc_layers",   1),
# # #         num_dec_layers   = getattr(args, "num_dec_layers",   3),
# # #         dim_ff           = getattr(args, "dim_ff",           512),
# # #         dropout          = getattr(args, "dropout",          0.1),
# # #         w_mse            = getattr(args, "w_mse",            0.05),
# # #         lambda_speed     = getattr(args, "lambda_speed",     0.1),
# # #         lambda_accel     = getattr(args, "lambda_accel",     0.01),
# # #         v_max_kmh        = getattr(args, "v_max_kmh",        80.0),
# # #         recurv_threshold = getattr(args, "recurv_threshold", 45.0),
# # #         gate_hidden      = getattr(args, "gate_hidden",      32),
# # #         recurv_hidden    = getattr(args, "recurv_hidden",    64),
# # #         threshold_curv   = threshold_curv,
# # #         threshold_spd    = threshold_spd,
# # #     )

# # """
# # Model/st_trans_v2_model.py  —— ST-Trans v2
# # ============================================
# # Mở rộng ST-Trans (Faiaz 2026) với Physics Steering Gate.

# # THIẾT KẾ:
# #   Easy samples (non-recurvature, ~61%):
# #     Training: L_DPE + w_mse*MSE + w_speed + w_accel  (= ST-Trans gốc)
# #     Inference: decoder output trực tiếp               (= ST-Trans gốc)
# #     → easy_ADE ≈ ST-Trans gốc (~155 km)

# #   Hard samples (recurvature/erratic, ~39%):
# #     Training:  final_h = α*lp_detach + (1-α)*physics
# #                L_hard = UW(DPE) + UW(heading) + UW(recurv_classify)
# #     Inference: final_h = α*lp + (1-α)*physics  (gate applied!)
# #     → hard_ADE tốt hơn ST-Trans nhờ physics correction

# # GRADIENT FLOW:
# #   Decoder: CHỈ từ easy  (lp_h.detach() trong training)
# #   Encoder: easy dominant + recurv auxiliary  (gate không update encoder)
# #   Gate:    học từ ctx.detach() + steer  (không ảnh hưởng encoder)

# # KEY: Gate apply ở INFERENCE cho hard → hard_ADE thực sự giảm mạnh
# # """
# # from __future__ import annotations
# # import math
# # from typing import Dict, Optional, Tuple

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # from Model.paper_baseline_model import (
# #     PaperEncoder,
# #     _norm_to_deg,
# #     haversine_km,
# #     compute_ade_per_horizon,
# #     compute_ate_cte_per_horizon,
# #     HORIZON_STEPS,
# # )

# # _U500_CH           = 0
# # _V500_CH           = 1
# # _DEG_SCALE         = 25.0
# # _MS_TO_NORM_PER_6H = (6.0 * 3600.0) / (111320.0 * _DEG_SCALE)


# # # ══════════════════════════════════════════════════════════════════════
# # #  Uncertainty Weighting — chỉ cho hard extras
# # # ══════════════════════════════════════════════════════════════════════

# # class UncertaintyWeighting(nn.Module):
# #     """
# #     L_weighted = L / (2σ²) + log(σ)   — Kendall et al. 2018

# #     CHỈ 3 tasks: dpe_hard, heading, recurv
# #     Easy loss dùng FIXED weights (= ST-Trans gốc) — không UW
# #     """
# #     TASKS = ["dpe_hard", "heading", "recurv"]
# #     INIT  = {
# #         "dpe_hard": -0.5,   # σ≈0.61, weight≈1.34 — quan trọng
# #         "heading":   0.3,   # σ≈1.35, weight≈0.27 — direction aux
# #         "recurv":    1.0,   # σ≈2.72, weight≈0.07 — classify aux (nhẹ)
# #     }

# #     def __init__(self):
# #         super().__init__()
# #         inits = torch.tensor([self.INIT[t] for t in self.TASKS])
# #         self.log_sigma = nn.Parameter(inits)
# #         self._idx = {t: i for i, t in enumerate(self.TASKS)}

# #     def weight(self, name: str, loss: torch.Tensor) -> torch.Tensor:
# #         s = self.log_sigma[self._idx[name]]
# #         return torch.exp(-2.0 * s) * loss / 2.0 + s

# #     def sigma_dict(self) -> Dict[str, float]:
# #         s = torch.exp(self.log_sigma).detach()
# #         return {f"σ_{t}": s[i].item() for i, t in enumerate(self.TASKS)}


# # # ══════════════════════════════════════════════════════════════════════
# # #  Sub-modules (giống ST-Trans gốc)
# # # ══════════════════════════════════════════════════════════════════════

# # class SinusoidalPE(nn.Module):
# #     def __init__(self, d_model: int, max_len: int = 300):
# #         super().__init__()
# #         pe  = torch.zeros(max_len, d_model)
# #         pos = torch.arange(max_len).unsqueeze(1).float()
# #         div = torch.exp(torch.arange(0, d_model, 2).float() *
# #                         (-math.log(10000.0) / d_model))
# #         pe[:, 0::2] = torch.sin(pos * div)
# #         pe[:, 1::2] = torch.cos(pos * div)
# #         self.register_buffer("pe", pe.unsqueeze(0))

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         return x + self.pe[:, :x.size(1)]


# # class ObsKinematicEncoder(nn.Module):
# #     """[T_obs, B, 2] → [B, T_obs, d_model]. Giống ST-Trans gốc."""
# #     FEAT_DIM = 8

# #     def __init__(self, d_model=64, nhead=4, num_layers=1,
# #                  dim_ff=256, dropout=0.1):
# #         super().__init__()
# #         self.proj = nn.Sequential(
# #             nn.Linear(self.FEAT_DIM, d_model), nn.ReLU(),
# #             nn.Linear(d_model, d_model))
# #         self.pe  = SinusoidalPE(d_model, max_len=64)
# #         enc      = nn.TransformerEncoderLayer(
# #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# #             dropout=dropout, activation="relu", batch_first=True)
# #         self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)

# #     @staticmethod
# #     def _feats(obs: torch.Tensor) -> torch.Tensor:
# #         T, B, _ = obs.shape
# #         dev = obs.device
# #         zeros1 = torch.zeros(1, B, device=dev)
# #         lon, lat = obs[:, :, 0], obs[:, :, 1]
# #         dl = torch.cat([obs[1:,:,0]-obs[:-1,:,0], zeros1], 0) if T>=2 \
# #              else torch.zeros_like(lon)
# #         dt = torch.cat([obs[1:,:,1]-obs[:-1,:,1], zeros1], 0) if T>=2 \
# #              else torch.zeros_like(lon)
# #         ddl = torch.cat([dl[1:]-dl[:-1], zeros1], 0) if T>=3 \
# #               else torch.zeros_like(lon)
# #         ddt = torch.cat([dt[1:]-dt[:-1], zeros1], 0) if T>=3 \
# #               else torch.zeros_like(lon)
# #         si  = torch.linspace(0,1,T,device=dev).unsqueeze(1).expand(T,B)
# #         spd = (dl**2 + dt**2).sqrt()
# #         return torch.stack([lon,lat,dl,dt,ddl,ddt,si,spd],-1).permute(1,0,2)

# #     def forward(self, obs: torch.Tensor) -> torch.Tensor:
# #         return self.enc(self.pe(self.proj(self._feats(obs))))


# # # ══════════════════════════════════════════════════════════════════════
# # #  Easy/Hard classifier
# # # ══════════════════════════════════════════════════════════════════════

# # def classify_hard_obs(obs_traj: torch.Tensor,
# #                       threshold_curv: float,
# #                       threshold_spd:  float) -> torch.Tensor:
# #     """[T_obs, B, 2] → [B] bool. True = hard (recurvature/erratic)."""
# #     T, B, _ = obs_traj.shape
# #     device  = obs_traj.device
# #     with torch.no_grad():
# #         if T < 2:
# #             return torch.zeros(B, dtype=torch.bool, device=device)
# #         vel     = obs_traj[1:] - obs_traj[:-1]
# #         spd     = vel.norm(dim=-1)                          # [T-1, B]
# #         spd_cv  = spd.std(0) / (spd.mean(0) + 1e-6)       # [B]
# #         if T >= 3:
# #             vn   = F.normalize(vel, dim=-1, eps=1e-8)
# #             cos  = (vn[1:] * vn[:-1]).sum(-1).clamp(-1, 1)
# #             curv = (torch.acos(cos) * (180/math.pi)).mean(0)  # [B]
# #         else:
# #             curv = torch.zeros(B, device=device)
# #         return (curv > threshold_curv) | (spd_cv > threshold_spd)


# # def compute_hard_thresholds(train_loader, device,
# #                              percentile: float = 70.0) -> Tuple[float, float]:
# #     all_curv, all_spd = [], []
# #     with torch.no_grad():
# #         for batch in train_loader:
# #             obs = batch[0].to(device)
# #             T, B, _ = obs.shape
# #             if T < 2:
# #                 continue
# #             vel = obs[1:] - obs[:-1]
# #             spd = vel.norm(dim=-1)
# #             all_spd.extend((spd.std(0)/(spd.mean(0)+1e-6)).cpu().tolist())
# #             if T >= 3:
# #                 vn  = F.normalize(vel, dim=-1, eps=1e-8)
# #                 cos = (vn[1:]*vn[:-1]).sum(-1).clamp(-1,1)
# #                 all_curv.extend(
# #                     (torch.acos(cos)*(180/math.pi)).mean(0).cpu().tolist())
# #     if not all_curv:
# #         return 15.0, 0.5
# #     import numpy as np
# #     return (float(np.percentile(all_curv, percentile)),
# #             float(np.percentile(all_spd,  percentile)))


# # # ══════════════════════════════════════════════════════════════════════
# # #  Physics Steering Gate
# # # ══════════════════════════════════════════════════════════════════════

# # class PhysicsSteeringGate(nn.Module):
# #     """
# #     Vectorized gate: 1 forward pass cho tất cả T bước.
# #     α[t] = sigmoid(W·[ctx, step_emb[t], steer] + b)
# #     Init b=2.0 → α≈0.88 (lean về learned ban đầu)
# #     """
# #     def __init__(self, ctx_dim=64, steer_dim=2, hidden=32, pred_len=12):
# #         super().__init__()
# #         self.pred_len = pred_len
# #         self.step_emb = nn.Embedding(pred_len, 16)
# #         self.gate_net = nn.Sequential(
# #             nn.Linear(ctx_dim + 16 + steer_dim, hidden),
# #             nn.ReLU(),
# #             nn.Linear(hidden, 1))
# #         nn.init.constant_(self.gate_net[-1].bias, 2.0)
# #         nn.init.zeros_(self.gate_net[-1].weight)


# # class RecurvatureTimingHead(nn.Module):
# #     """
# #     Auxiliary classifier: predict bước recurvature.
# #     Output: logits [B, pred_len+1]  (class 0 = không recurve)
# #     Gradient vào encoder → encoder học phân biệt recurvature pattern
# #     Weight nhỏ (σ_recurv=1.0) → encoder không bị dominated
# #     """
# #     def __init__(self, ctx_dim=64, pred_len=12, hidden=64):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(ctx_dim, hidden), nn.GELU(),
# #             nn.Dropout(0.1),
# #             nn.Linear(hidden, pred_len + 1))

# #     def forward(self, ctx: torch.Tensor) -> torch.Tensor:
# #         return self.net(ctx)

# #     @staticmethod
# #     def make_label(gt: torch.Tensor, obs: torch.Tensor,
# #                    threshold_deg: float = 45.0) -> torch.Tensor:
# #         """gt [T,B,2], obs [T_obs,B,2] → label [B] long."""
# #         T, B, _ = gt.shape
# #         dev   = gt.device
# #         label = torch.zeros(B, dtype=torch.long, device=dev)
# #         full  = torch.cat([obs[-1:], gt], 0)   # [T+1, B, 2]
# #         with torch.no_grad():
# #             for t in range(T - 1):
# #                 d_in  = full[t+1] - full[t]
# #                 d_out = full[t+2] - full[t+1]
# #                 ni = F.normalize(d_in,  dim=-1, eps=1e-8)
# #                 no = F.normalize(d_out, dim=-1, eps=1e-8)
# #                 ang  = torch.acos((ni*no).sum(-1).clamp(-1,1)) * (180/math.pi)
# #                 mask = (ang > threshold_deg) & (label == 0)
# #                 label[mask] = t + 1
# #         return label


# # def _get_steering(data3d: Optional[torch.Tensor],
# #                   obs_h:  torch.Tensor) -> Optional[torch.Tensor]:
# #     """ERA5 center-pixel wind → [B_h, 2] normalised displacement."""
# #     if (data3d is None or not isinstance(data3d, torch.Tensor)
# #             or data3d.dim() != 4):
# #         return None
# #     B, C, H, W = data3d.shape
# #     if C <= max(_U500_CH, _V500_CH):
# #         return None
# #     cy, cx  = H // 2, W // 2
# #     u_ms    = data3d[:, _U500_CH, cy, cx]
# #     v_ms    = data3d[:, _V500_CH, cy, cx]
# #     lat_deg = obs_h[-1, :, 1] * _DEG_SCALE
# #     cos_lat = torch.cos(lat_deg * (math.pi/180)).clamp(0.1, 1.0)
# #     return torch.stack([(u_ms/cos_lat) * _MS_TO_NORM_PER_6H,
# #                         v_ms * _MS_TO_NORM_PER_6H], dim=-1)


# # # ══════════════════════════════════════════════════════════════════════
# # #  STTransV2
# # # ══════════════════════════════════════════════════════════════════════

# # class STTransV2(nn.Module):

# #     def __init__(
# #         self,
# #         obs_len=8, pred_len=12, unet_in_ch=13,
# #         d_model=64, nhead=4, num_enc_layers=1, num_dec_layers=3,
# #         dim_ff=512, dropout=0.1,
# #         # Easy loss weights — FIXED như ST-Trans gốc
# #         w_mse=0.05, lambda_speed=0.1, lambda_accel=0.01,
# #         v_max_kmh=80.0, dt_h=6.0,
# #         # Hard extras
# #         recurv_threshold=45.0, gate_hidden=32, recurv_hidden=64,
# #         # Easy/hard thresholds
# #         threshold_curv=15.0, threshold_spd=0.5,
# #     ):
# #         super().__init__()
# #         self.obs_len          = obs_len
# #         self.pred_len         = pred_len
# #         self.w_mse            = w_mse
# #         self.lambda_speed     = lambda_speed
# #         self.lambda_accel     = lambda_accel
# #         self.v_max_norm       = v_max_kmh * dt_h / (111.0 * _DEG_SCALE)
# #         self.recurv_threshold = recurv_threshold
# #         self.threshold_curv   = threshold_curv
# #         self.threshold_spd    = threshold_spd

# #         # ── Encoder (= ST-Trans gốc) ──────────────────────────────────
# #         self.encoder  = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
# #         self.ctx_proj = nn.Sequential(
# #             nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
# #             nn.LayerNorm(d_model))
# #         self.obs_enc  = ObsKinematicEncoder(
# #             d_model=d_model, nhead=nhead,
# #             num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout)

# #         # ── Decoder (= ST-Trans gốc) ──────────────────────────────────
# #         self.horizon_queries = nn.Parameter(
# #             torch.randn(1, pred_len, d_model) * 0.02)
# #         self.dec_pe = SinusoidalPE(d_model, max_len=pred_len + 10)
# #         dec_layer = nn.TransformerDecoderLayer(
# #             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
# #             dropout=dropout, activation="relu", batch_first=True)
# #         self.transformer_dec = nn.TransformerDecoder(
# #             dec_layer, num_layers=num_dec_layers)
# #         self.reg_head = nn.Sequential(
# #             nn.Linear(d_model, d_model), nn.ReLU(),
# #             nn.Linear(d_model, 2))

# #         # ── Hard-only modules ─────────────────────────────────────────
# #         self.steering_gate = PhysicsSteeringGate(
# #             ctx_dim=d_model, steer_dim=2,
# #             hidden=gate_hidden, pred_len=pred_len)
# #         self.recurv_head   = RecurvatureTimingHead(
# #             ctx_dim=d_model, pred_len=pred_len, hidden=recurv_hidden)

# #         # ── UW: chỉ 3 tasks cho hard ─────────────────────────────────
# #         self.uw = UncertaintyWeighting()

# #         self._init_weights()

# #     def _init_weights(self):
# #         for m in self.modules():
# #             if isinstance(m, nn.Linear):
# #                 nn.init.xavier_uniform_(m.weight, gain=0.5)
# #                 if m.bias is not None:
# #                     nn.init.zeros_(m.bias)
# #         # Gate bias = 2.0 → alpha≈0.88 ban đầu (lean về learned)
# #         nn.init.constant_(self.steering_gate.gate_net[-1].bias, 2.0)
# #         nn.init.zeros_(self.steering_gate.gate_net[-1].weight)

# #     def set_thresholds(self, curv: float, spd: float):
# #         self.threshold_curv = curv
# #         self.threshold_spd  = spd

# #     # ── Encode (1 lần cho toàn batch) ────────────────────────────────

# #     def _encode(self, batch_list) -> Tuple[torch.Tensor, torch.Tensor]:
# #         """→ (lp [B,H,2], ctx [B,d])"""
# #         obs  = batch_list[0]                            # [T_obs, B, 2]
# #         B    = obs.shape[1]
# #         raw  = self.encoder(batch_list)                 # [B, RAW_CTX_DIM]
# #         ctok = self.ctx_proj(raw).unsqueeze(1)          # [B, 1, d]
# #         omem = self.obs_enc(obs)                        # [B, T_obs, d]
# #         fmem = torch.cat([ctok, omem], 1)              # [B, 1+T_obs, d]
# #         ctx  = fmem.mean(1)                             # [B, d]
# #         Q    = self.dec_pe(self.horizon_queries.expand(B, -1, -1))
# #         D    = self.transformer_dec(Q, fmem)            # [B, H, d]
# #         lp   = self.reg_head(D)                         # [B, H, 2]
# #         return lp, ctx

# #     # ── Gate blend ────────────────────────────────────────────────────

# #     def _gate_blend(self,
# #                     lp_h:       torch.Tensor,          # [B_h, H, 2]
# #                     ctx_h:      torch.Tensor,          # [B_h, d]
# #                     steer:      Optional[torch.Tensor],# [B_h, 2] hoặc None
# #                     obs_h:      torch.Tensor,          # [T_obs, B_h, 2]
# #                     stop_ctx_grad: bool = True,
# #                     ) -> Tuple[torch.Tensor, torch.Tensor]:
# #         """
# #         Physics persistence blend:
# #           physics[t] = obs_h[-1] + (t+1)*steer        [B_h, 2]
# #           alpha[t]   = sigmoid(gate([ctx, step_emb, steer]))
# #           final[t]   = alpha[t]*lp_h[t] + (1-alpha[t])*physics[t]

# #         stop_ctx_grad=True  (training):
# #           → gate KHÔNG update encoder qua ctx gradient
# #           → encoder gradient chỉ từ easy + recurv_head
# #         stop_ctx_grad=False (inference):
# #           → full forward, không cần gradient
# #         """
# #         B_h = lp_h.shape[0]
# #         T   = self.pred_len
# #         dev = lp_h.device

# #         # Steer fallback: last observed velocity
# #         if steer is None:
# #             steer = (obs_h[-1] - obs_h[-2]
# #                      if obs_h.shape[0] >= 2
# #                      else torch.zeros(B_h, 2, device=dev))

# #         # Physics trajectory: [B_h, T, 2]
# #         steps   = torch.arange(1, T+1, device=dev, dtype=steer.dtype)
# #         offsets = steer.unsqueeze(1) * steps.view(1, T, 1)
# #         physics = obs_h[-1].unsqueeze(1) + offsets

# #         # Gate (vectorized, 1 forward pass)
# #         ctx_in  = ctx_h.detach() if stop_ctx_grad else ctx_h
# #         t_idx   = torch.arange(T, device=dev)
# #         s_emb   = self.steering_gate.step_emb(t_idx)        # [T, 16]
# #         ctx_e   = ctx_in.unsqueeze(0).expand(T, -1, -1)     # [T, B_h, d]
# #         steer_e = steer.unsqueeze(0).expand(T, -1, -1)      # [T, B_h, 2]
# #         s_emb_e = s_emb.unsqueeze(1).expand(T, B_h, -1)    # [T, B_h, 16]
# #         gate_in = torch.cat([ctx_e, s_emb_e, steer_e], -1) # [T, B_h, d+18]
# #         alpha   = torch.sigmoid(
# #             self.steering_gate.gate_net(gate_in))            # [T, B_h, 1]

# #         # Blend: [B_h, T, 2]
# #         alpha_t = alpha.permute(1, 0, 2)                    # [B_h, T, 1]
# #         final_h = alpha_t * lp_h + (1 - alpha_t) * physics

# #         return final_h, alpha.mean()

# #     # ── Easy losses: GIỐNG HỆT ST-TRANS GỐC ─────────────────────────

# #     def _easy_losses(self, pred: torch.Tensor,
# #                      gt: torch.Tensor) -> Dict[str, torch.Tensor]:
# #         """
# #         pred, gt: [T, B_e, 2]
# #         Fixed weights, không UW — giống ST-Trans gốc.
# #         Gradient đầy đủ → decoder + encoder.
# #         """
# #         T = min(pred.shape[0], gt.shape[0])
# #         p, g = pred[:T], gt[:T]

# #         # DPE: uniform haversine mean (không step_weights)
# #         l_dpe = haversine_km(_norm_to_deg(p), _norm_to_deg(g)).mean()
# #         l_mse = F.mse_loss(p, g)

# #         l_speed = p.new_zeros(())
# #         if T >= 2:
# #             sd      = (p[1:] - p[:-1]).norm(-1)
# #             l_speed = F.relu(sd - self.v_max_norm).pow(2).mean()

# #         l_accel = p.new_zeros(())
# #         if T >= 3:
# #             v       = p[1:] - p[:-1]
# #             l_accel = (v[1:].norm(-1) - v[:-1].norm(-1)).pow(2).mean()

# #         return {"dpe": l_dpe, "mse": l_mse,
# #                 "speed": l_speed, "accel": l_accel}

# #     # ── Hard losses: GATE-CORRECTED ───────────────────────────────────

# #     def _hard_losses(self,
# #                      final_h: torch.Tensor,  # [B_h, H, 2] gate output
# #                      gt_h:    torch.Tensor,  # [T_pred, B_h, 2]
# #                      ctx_h:   torch.Tensor,  # [B_h, d] gốc (có gradient)
# #                      obs_h:   torch.Tensor,  # [T_obs, B_h, 2]
# #                      ) -> Dict[str, torch.Tensor]:
# #         """
# #         final_h: lp_h.detach() đã apply trước khi gọi gate_blend
# #           → gradient từ L_dpe, L_heading đi: final_h → alpha → gate
# #           → gate KHÔNG update encoder (stop_ctx_grad=True trong gate)

# #         ctx_h (gốc): được dùng bởi recurv_head
# #           → gradient từ L_recurv đi: ctx_h → encoder
# #           → encoder học classify recurvature (auxiliary, weight nhỏ)
# #         """
# #         T   = min(final_h.shape[1], gt_h.shape[0])
# #         fp  = final_h[:, :T]                        # [B_h, T, 2]
# #         g   = gt_h[:T].permute(1, 0, 2)             # [B_h, T, 2]
# #         fpt = fp.permute(1, 0, 2)                   # [T, B_h, 2]
# #         gt  = g.permute(1, 0, 2)                    # [T, B_h, 2]

# #         # L_DPE: uniform mean trên gate output
# #         l_dpe = haversine_km(_norm_to_deg(fpt), _norm_to_deg(gt)).mean()

# #         # L_heading: cosine similarity giữa pred và gt direction
# #         l_heading = fp.new_zeros(())
# #         if T >= 2:
# #             pd = fpt[1:] - fpt[:-1]                 # [T-1, B_h, 2]
# #             gd = gt[1:]  - gt[:-1]
# #             pn = F.normalize(pd, dim=-1, eps=1e-8)
# #             gn = F.normalize(gd, dim=-1, eps=1e-8)
# #             l_heading = ((1.0 - (pn*gn).sum(-1).clamp(-1,1)) / 2.0).mean()

# #         # L_recurv: auxiliary CE — ctx_h GỐC (có gradient → encoder)
# #         l_recurv   = fp.new_zeros(())
# #         recurv_log = self.recurv_head(ctx_h)         # [B_h, pred_len+1]
# #         try:
# #             label = RecurvatureTimingHead.make_label(
# #                 gt_h, obs_h, self.recurv_threshold)
# #             if label.shape[0] == recurv_log.shape[0]:
# #                 l_recurv = F.cross_entropy(recurv_log, label)
# #         except Exception:
# #             pass

# #         return {"dpe": l_dpe, "heading": l_heading, "recurv": l_recurv}

# #     # ── Training loss ─────────────────────────────────────────────────

# #     def get_loss_breakdown(self, batch_list) -> Dict:
# #         obs_traj = batch_list[0]   # [T_obs, B, 2]
# #         traj_gt  = batch_list[1]   # [T_pred, B, 2]

# #         # 1. Classify easy / hard
# #         is_hard = classify_hard_obs(
# #             obs_traj, self.threshold_curv, self.threshold_spd)
# #         is_easy = ~is_hard
# #         n_easy  = int(is_easy.sum())
# #         n_hard  = int(is_hard.sum())

# #         # 2. Encode: 1 shared forward pass
# #         lp, ctx = self._encode(batch_list)   # [B,H,2], [B,d]

# #         result = {"n_easy": n_easy, "n_hard": n_hard}

# #         # ── EASY: = ST-Trans gốc ─────────────────────────────────────
# #         # Gradient đầy đủ: L_easy → lp_e → decoder + encoder
# #         L_easy = None
# #         if n_easy > 0:
# #             lp_e = lp[is_easy].permute(1, 0, 2)   # [T, B_e, 2]
# #             gt_e = traj_gt[:, is_easy]             # [T, B_e, 2]
# #             el   = self._easy_losses(lp_e, gt_e)

# #             # FIXED weights — không UW (như ST-Trans gốc)
# #             L_easy = (
# #                 el["dpe"]
# #                 + self.w_mse        * el["mse"]
# #                 + self.lambda_speed * el["speed"]
# #                 + self.lambda_accel * el["accel"]
# #             )
# #             result.update({
# #                 "easy_dpe":   el["dpe"].item(),
# #                 "easy_mse":   el["mse"].item(),
# #             })

# #         # ── HARD: gate-corrected ─────────────────────────────────────
# #         # lp_h.detach()      → decoder KHÔNG nhận gradient từ L_hard
# #         # stop_ctx_grad=True → encoder KHÔNG nhận gradient từ gate
# #         # ctx_h (gốc) vào recurv_head → encoder học classify (auxiliary)
# #         L_hard = None
# #         if n_hard > 0:
# #             lp_h  = lp[is_hard]               # [B_h, H, 2]
# #             gt_h  = traj_gt[:, is_hard]       # [T, B_h, 2]
# #             obs_h = obs_traj[:, is_hard]      # [T_obs, B_h, 2]
# #             ctx_h = ctx[is_hard]              # [B_h, d]

# #             data3d = None
# #             if (len(batch_list) > 2
# #                     and isinstance(batch_list[2], torch.Tensor)
# #                     and batch_list[2].dim() == 4):
# #                 data3d = batch_list[2][is_hard]
# #             steer = _get_steering(data3d, obs_h)

# #             # Gate với lp_h.detach() + stop_ctx_grad → bảo vệ decoder + encoder
# #             final_h, alpha_m = self._gate_blend(
# #                 lp_h.detach(), ctx_h, steer, obs_h,
# #                 stop_ctx_grad=True)

# #             hl = self._hard_losses(final_h, gt_h, ctx_h, obs_h)

# #             # UW cho hard losses
# #             L_hard = (
# #                 self.uw.weight("dpe_hard", hl["dpe"])
# #                 + self.uw.weight("heading", hl["heading"])
# #                 + self.uw.weight("recurv",  hl["recurv"])
# #             )
# #             result.update({
# #                 "hard_dpe":    hl["dpe"].item(),
# #                 "hard_heading":(hl["heading"].item()
# #                                 if isinstance(hl["heading"], torch.Tensor)
# #                                 else 0.0),
# #                 "alpha_mean":  (alpha_m.item()
# #                                 if isinstance(alpha_m, torch.Tensor)
# #                                 else 0.0),
# #             })

# #         # 3. Combine
# #         if L_easy is not None and L_hard is not None:
# #             total = L_easy + L_hard
# #         elif L_easy is not None:
# #             total = L_easy
# #         else:
# #             total = L_hard

# #         result["total"] = total
# #         result.update(self.uw.sigma_dict())

# #         # 4. Raw ADE metrics (decoder output, no gate) — để monitor training
# #         lp_t = lp.permute(1, 0, 2)   # [T, B, 2]
# #         with torch.no_grad():
# #             result.update(compute_ade_per_horizon(lp_t.detach(), traj_gt))
# #             result.update(compute_ate_cte_per_horizon(lp_t.detach(), traj_gt))

# #         return result

# #     def get_loss(self, batch_list) -> torch.Tensor:
# #         return self.get_loss_breakdown(batch_list)["total"]

# #     # ── Inference ─────────────────────────────────────────────────────

# #     def forward(self, batch_list) -> torch.Tensor:
# #         """
# #         Easy  → decoder output (= ST-Trans gốc)
# #         Hard  → gate-blended   (tốt hơn nhờ physics correction)
# #         Return: [T, B, 2]
# #         """
# #         lp, ctx = self._encode(batch_list)
# #         obs     = batch_list[0]                        # [T_obs, B, 2]

# #         # Default: decoder output cho tất cả
# #         pred = lp.permute(1, 0, 2).clone()             # [T, B, 2]

# #         # Hard: override bằng gate-blended
# #         is_hard = classify_hard_obs(
# #             obs, self.threshold_curv, self.threshold_spd)

# #         if is_hard.any():
# #             obs_h = obs[:, is_hard]
# #             ctx_h = ctx[is_hard]
# #             lp_h  = lp[is_hard]                        # [B_h, H, 2]

# #             data3d = None
# #             if (len(batch_list) > 2
# #                     and isinstance(batch_list[2], torch.Tensor)
# #                     and batch_list[2].dim() == 4):
# #                 data3d = batch_list[2][is_hard]
# #             steer = _get_steering(data3d, obs_h)

# #             # Inference: không cần stop_ctx_grad (không backward)
# #             final_h, _ = self._gate_blend(
# #                 lp_h, ctx_h, steer, obs_h,
# #                 stop_ctx_grad=False)

# #             pred[:, is_hard] = final_h.permute(1, 0, 2)  # [T, B_h, 2]

# #         return pred   # [T, B, 2]

# #     @torch.no_grad()
# #     def sample(self, batch_list, **kwargs):
# #         pred    = self.forward(batch_list)   # [T, B, 2]
# #         T, B, _ = pred.shape
# #         return pred, torch.zeros(T, B, 2, device=pred.device), pred.unsqueeze(0)


# # # ══════════════════════════════════════════════════════════════════════
# # #  Factory
# # # ══════════════════════════════════════════════════════════════════════

# # def build_st_trans_v2(args, threshold_curv=15.0, threshold_spd=0.5):
# #     return STTransV2(
# #         obs_len          = getattr(args, "obs_len",          8),
# #         pred_len         = getattr(args, "pred_len",         12),
# #         unet_in_ch       = getattr(args, "unet_in_ch",       13),
# #         d_model          = getattr(args, "d_model",          64),
# #         nhead            = getattr(args, "nhead",            4),
# #         num_enc_layers   = getattr(args, "num_enc_layers",   1),
# #         num_dec_layers   = getattr(args, "num_dec_layers",   3),
# #         dim_ff           = getattr(args, "dim_ff",           512),
# #         dropout          = getattr(args, "dropout",          0.1),
# #         w_mse            = getattr(args, "w_mse",            0.05),
# #         lambda_speed     = getattr(args, "lambda_speed",     0.1),
# #         lambda_accel     = getattr(args, "lambda_accel",     0.01),
# #         v_max_kmh        = getattr(args, "v_max_kmh",        80.0),
# #         recurv_threshold = getattr(args, "recurv_threshold", 45.0),
# #         gate_hidden      = getattr(args, "gate_hidden",      32),
# #         recurv_hidden    = getattr(args, "recurv_hidden",    64),
# #         threshold_curv   = threshold_curv,
# #         threshold_spd    = threshold_spd,
# #     )

# """
# Model/st_trans_v2_model.py  —— ST-Trans v2
# ============================================
# Mở rộng ST-Trans (Faiaz 2026) với Physics Steering Gate.

# THIẾT KẾ:
#   Easy samples (non-recurvature, ~61%):
#     Training: L_DPE + w_mse*MSE + w_speed + w_accel  (= ST-Trans gốc)
#     Inference: decoder output trực tiếp               (= ST-Trans gốc)
#     → easy_ADE ≈ ST-Trans gốc (~155 km)

#   Hard samples (recurvature/erratic, ~39%):
#     Training:  final_h = α*lp_detach + (1-α)*physics
#                L_hard = UW(DPE) + UW(heading) + UW(recurv_classify)
#     Inference: final_h = α*lp + (1-α)*physics  (gate applied!)
#     → hard_ADE tốt hơn ST-Trans nhờ physics correction

# GRADIENT FLOW:
#   Decoder: CHỈ từ easy  (lp_h.detach() trong training)
#   Encoder: easy dominant + recurv auxiliary  (gate không update encoder)
#   Gate:    học từ ctx.detach() + steer  (không ảnh hưởng encoder)

# KEY: Gate apply ở INFERENCE cho hard → hard_ADE thực sự giảm mạnh
# """
# from __future__ import annotations
# import math
# from typing import Dict, Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from Model.paper_baseline_model import (
#     PaperEncoder,
#     _norm_to_deg,
#     haversine_km,
#     compute_ade_per_horizon,
#     compute_ate_cte_per_horizon,
#     HORIZON_STEPS,
# )

# _U500_CH           = 0
# _V500_CH           = 1
# _DEG_SCALE         = 25.0
# _MS_TO_NORM_PER_6H = (6.0 * 3600.0) / (111320.0 * _DEG_SCALE)


# # ══════════════════════════════════════════════════════════════════════
# #  Uncertainty Weighting — chỉ cho hard extras
# # ══════════════════════════════════════════════════════════════════════

# class UncertaintyWeighting(nn.Module):
#     """
#     L_weighted = L / (2σ²) + log(σ)   — Kendall et al. 2018

#     CHỈ 3 tasks: dpe_hard, heading, recurv
#     Easy loss dùng FIXED weights (= ST-Trans gốc) — không UW
#     """
#     TASKS = ["dpe_hard", "heading", "recurv"]
#     INIT  = {
#         "dpe_hard": -0.5,   # σ≈0.61, weight≈1.34 — quan trọng
#         "heading":   0.3,   # σ≈1.35, weight≈0.27 — direction aux
#         "recurv":    1.0,   # σ≈2.72, weight≈0.07 — classify aux (nhẹ)
#     }

#     def __init__(self):
#         super().__init__()
#         inits = torch.tensor([self.INIT[t] for t in self.TASKS])
#         self.log_sigma = nn.Parameter(inits)
#         self._idx = {t: i for i, t in enumerate(self.TASKS)}

#     def weight(self, name: str, loss: torch.Tensor) -> torch.Tensor:
#         s = self.log_sigma[self._idx[name]]
#         return torch.exp(-2.0 * s) * loss / 2.0 + s

#     def sigma_dict(self) -> Dict[str, float]:
#         s = torch.exp(self.log_sigma).detach()
#         return {f"σ_{t}": s[i].item() for i, t in enumerate(self.TASKS)}


# # ══════════════════════════════════════════════════════════════════════
# #  Sub-modules (giống ST-Trans gốc)
# # ══════════════════════════════════════════════════════════════════════

# class SinusoidalPE(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 300):
#         super().__init__()
#         pe  = torch.zeros(max_len, d_model)
#         pos = torch.arange(max_len).unsqueeze(1).float()
#         div = torch.exp(torch.arange(0, d_model, 2).float() *
#                         (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(pos * div)
#         pe[:, 1::2] = torch.cos(pos * div)
#         self.register_buffer("pe", pe.unsqueeze(0))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x + self.pe[:, :x.size(1)]


# class ObsKinematicEncoder(nn.Module):
#     """[T_obs, B, 2] → [B, T_obs, d_model]. Giống ST-Trans gốc."""
#     FEAT_DIM = 8

#     def __init__(self, d_model=64, nhead=4, num_layers=1,
#                  dim_ff=256, dropout=0.1):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Linear(self.FEAT_DIM, d_model), nn.ReLU(),
#             nn.Linear(d_model, d_model))
#         self.pe  = SinusoidalPE(d_model, max_len=64)
#         enc      = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
#             dropout=dropout, activation="relu", batch_first=True)
#         self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)

#     @staticmethod
#     def _feats(obs: torch.Tensor) -> torch.Tensor:
#         T, B, _ = obs.shape
#         dev = obs.device
#         zeros1 = torch.zeros(1, B, device=dev)
#         lon, lat = obs[:, :, 0], obs[:, :, 1]
#         dl = torch.cat([obs[1:,:,0]-obs[:-1,:,0], zeros1], 0) if T>=2 \
#              else torch.zeros_like(lon)
#         dt = torch.cat([obs[1:,:,1]-obs[:-1,:,1], zeros1], 0) if T>=2 \
#              else torch.zeros_like(lon)
#         ddl = torch.cat([dl[1:]-dl[:-1], zeros1], 0) if T>=3 \
#               else torch.zeros_like(lon)
#         ddt = torch.cat([dt[1:]-dt[:-1], zeros1], 0) if T>=3 \
#               else torch.zeros_like(lon)
#         si  = torch.linspace(0,1,T,device=dev).unsqueeze(1).expand(T,B)
#         spd = (dl**2 + dt**2).sqrt()
#         return torch.stack([lon,lat,dl,dt,ddl,ddt,si,spd],-1).permute(1,0,2)

#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         return self.enc(self.pe(self.proj(self._feats(obs))))


# # ══════════════════════════════════════════════════════════════════════
# #  Easy/Hard classifier
# # ══════════════════════════════════════════════════════════════════════

# def classify_hard_obs(obs_traj: torch.Tensor,
#                       threshold_curv: float,
#                       threshold_spd:  float) -> torch.Tensor:
#     """[T_obs, B, 2] → [B] bool. True = hard (recurvature/erratic)."""
#     T, B, _ = obs_traj.shape
#     device  = obs_traj.device
#     with torch.no_grad():
#         if T < 2:
#             return torch.zeros(B, dtype=torch.bool, device=device)
#         vel     = obs_traj[1:] - obs_traj[:-1]
#         spd     = vel.norm(dim=-1)                          # [T-1, B]
#         spd_cv  = spd.std(0) / (spd.mean(0) + 1e-6)       # [B]
#         if T >= 3:
#             vn   = F.normalize(vel, dim=-1, eps=1e-8)
#             cos  = (vn[1:] * vn[:-1]).sum(-1).clamp(-1, 1)
#             curv = (torch.acos(cos) * (180/math.pi)).mean(0)  # [B]
#         else:
#             curv = torch.zeros(B, device=device)
#         return (curv > threshold_curv) | (spd_cv > threshold_spd)


# def compute_hard_thresholds(train_loader, device,
#                              percentile: float = 70.0) -> Tuple[float, float]:
#     all_curv, all_spd = [], []
#     with torch.no_grad():
#         for batch in train_loader:
#             obs = batch[0].to(device)
#             T, B, _ = obs.shape
#             if T < 2:
#                 continue
#             vel = obs[1:] - obs[:-1]
#             spd = vel.norm(dim=-1)
#             all_spd.extend((spd.std(0)/(spd.mean(0)+1e-6)).cpu().tolist())
#             if T >= 3:
#                 vn  = F.normalize(vel, dim=-1, eps=1e-8)
#                 cos = (vn[1:]*vn[:-1]).sum(-1).clamp(-1,1)
#                 all_curv.extend(
#                     (torch.acos(cos)*(180/math.pi)).mean(0).cpu().tolist())
#     if not all_curv:
#         return 15.0, 0.5
#     import numpy as np
#     return (float(np.percentile(all_curv, percentile)),
#             float(np.percentile(all_spd,  percentile)))


# # ══════════════════════════════════════════════════════════════════════
# #  Physics Steering Gate
# # ══════════════════════════════════════════════════════════════════════

# class PhysicsSteeringGate(nn.Module):
#     """
#     Vectorized gate: 1 forward pass cho tất cả T bước.
#     α[t] = sigmoid(W·[ctx, step_emb[t], steer] + b)
#     Init b=2.0 → α≈0.88 (lean về learned ban đầu)
#     """
#     def __init__(self, ctx_dim=64, steer_dim=2, hidden=32, pred_len=12):
#         super().__init__()
#         self.pred_len = pred_len
#         self.step_emb = nn.Embedding(pred_len, 16)
#         self.gate_net = nn.Sequential(
#             nn.Linear(ctx_dim + 16 + steer_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, 1))
#         nn.init.constant_(self.gate_net[-1].bias, 2.0)
#         nn.init.zeros_(self.gate_net[-1].weight)


# class RecurvatureTimingHead(nn.Module):
#     """
#     Auxiliary classifier: predict bước recurvature.
#     Output: logits [B, pred_len+1]  (class 0 = không recurve)
#     Gradient vào encoder → encoder học phân biệt recurvature pattern
#     Weight nhỏ (σ_recurv=1.0) → encoder không bị dominated
#     """
#     def __init__(self, ctx_dim=64, pred_len=12, hidden=64):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(ctx_dim, hidden), nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden, pred_len + 1))

#     def forward(self, ctx: torch.Tensor) -> torch.Tensor:
#         return self.net(ctx)

#     @staticmethod
#     def make_label(gt: torch.Tensor, obs: torch.Tensor,
#                    threshold_deg: float = 45.0) -> torch.Tensor:
#         """gt [T,B,2], obs [T_obs,B,2] → label [B] long."""
#         T, B, _ = gt.shape
#         dev   = gt.device
#         label = torch.zeros(B, dtype=torch.long, device=dev)
#         full  = torch.cat([obs[-1:], gt], 0)   # [T+1, B, 2]
#         with torch.no_grad():
#             for t in range(T - 1):
#                 d_in  = full[t+1] - full[t]
#                 d_out = full[t+2] - full[t+1]
#                 ni = F.normalize(d_in,  dim=-1, eps=1e-8)
#                 no = F.normalize(d_out, dim=-1, eps=1e-8)
#                 ang  = torch.acos((ni*no).sum(-1).clamp(-1,1)) * (180/math.pi)
#                 mask = (ang > threshold_deg) & (label == 0)
#                 label[mask] = t + 1
#         return label


# def _get_steering(data3d: Optional[torch.Tensor],
#                   obs_h:  torch.Tensor) -> Optional[torch.Tensor]:
#     """ERA5 center-pixel wind → [B_h, 2] normalised displacement."""
#     if (data3d is None or not isinstance(data3d, torch.Tensor)
#             or data3d.dim() != 4):
#         return None
#     B, C, H, W = data3d.shape
#     if C <= max(_U500_CH, _V500_CH):
#         return None
#     cy, cx  = H // 2, W // 2
#     u_ms    = data3d[:, _U500_CH, cy, cx]
#     v_ms    = data3d[:, _V500_CH, cy, cx]
#     lat_deg = obs_h[-1, :, 1] * _DEG_SCALE
#     cos_lat = torch.cos(lat_deg * (math.pi/180)).clamp(0.1, 1.0)
#     return torch.stack([(u_ms/cos_lat) * _MS_TO_NORM_PER_6H,
#                         v_ms * _MS_TO_NORM_PER_6H], dim=-1)


# # ══════════════════════════════════════════════════════════════════════
# #  STTransV2
# # ══════════════════════════════════════════════════════════════════════

# class STTransV2(nn.Module):

#     def __init__(
#         self,
#         obs_len=8, pred_len=12, unet_in_ch=13,
#         d_model=64, nhead=4, num_enc_layers=1, num_dec_layers=3,
#         dim_ff=512, dropout=0.1,
#         # Easy loss weights — FIXED như ST-Trans gốc
#         w_mse=0.05, lambda_speed=0.1, lambda_accel=0.01,
#         v_max_kmh=80.0, dt_h=6.0,
#         # Hard extras
#         recurv_threshold=45.0, gate_hidden=32, recurv_hidden=64,
#         # Easy/hard thresholds
#         threshold_curv=15.0, threshold_spd=0.5,
#     ):
#         super().__init__()
#         self.obs_len          = obs_len
#         self.pred_len         = pred_len
#         self.w_mse            = w_mse
#         self.lambda_speed     = lambda_speed
#         self.lambda_accel     = lambda_accel
#         self.v_max_norm       = v_max_kmh * dt_h / (111.0 * _DEG_SCALE)
#         self.recurv_threshold = recurv_threshold
#         self.threshold_curv   = threshold_curv
#         self.threshold_spd    = threshold_spd

#         # ── Encoder (= ST-Trans gốc) ──────────────────────────────────
#         self.encoder  = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
#         self.ctx_proj = nn.Sequential(
#             nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
#             nn.LayerNorm(d_model))
#         self.obs_enc  = ObsKinematicEncoder(
#             d_model=d_model, nhead=nhead,
#             num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout)

#         # ── Decoder (= ST-Trans gốc) ──────────────────────────────────
#         self.horizon_queries = nn.Parameter(
#             torch.randn(1, pred_len, d_model) * 0.02)
#         self.dec_pe = SinusoidalPE(d_model, max_len=pred_len + 10)
#         dec_layer = nn.TransformerDecoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
#             dropout=dropout, activation="relu", batch_first=True)
#         self.transformer_dec = nn.TransformerDecoder(
#             dec_layer, num_layers=num_dec_layers)
#         self.reg_head = nn.Sequential(
#             nn.Linear(d_model, d_model), nn.ReLU(),
#             nn.Linear(d_model, 2))

#         # ── Hard-only modules ─────────────────────────────────────────
#         self.steering_gate = PhysicsSteeringGate(
#             ctx_dim=d_model, steer_dim=2,
#             hidden=gate_hidden, pred_len=pred_len)
#         self.recurv_head   = RecurvatureTimingHead(
#             ctx_dim=d_model, pred_len=pred_len, hidden=recurv_hidden)

#         # ── UW: chỉ 3 tasks cho hard ─────────────────────────────────
#         self.uw = UncertaintyWeighting()

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=0.5)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#         # Gate bias = 2.0 → alpha≈0.88 ban đầu (lean về learned)
#         nn.init.constant_(self.steering_gate.gate_net[-1].bias, 2.0)
#         nn.init.zeros_(self.steering_gate.gate_net[-1].weight)

#     def set_thresholds(self, curv: float, spd: float):
#         self.threshold_curv = curv
#         self.threshold_spd  = spd

#     # ── Encode (1 lần cho toàn batch) ────────────────────────────────

#     def _encode(self, batch_list) -> Tuple[torch.Tensor, torch.Tensor]:
#         """→ (lp [B,H,2], ctx [B,d])"""
#         obs  = batch_list[0]                            # [T_obs, B, 2]
#         B    = obs.shape[1]
#         raw  = self.encoder(batch_list)                 # [B, RAW_CTX_DIM]
#         ctok = self.ctx_proj(raw).unsqueeze(1)          # [B, 1, d]
#         omem = self.obs_enc(obs)                        # [B, T_obs, d]
#         fmem = torch.cat([ctok, omem], 1)              # [B, 1+T_obs, d]
#         ctx  = fmem.mean(1)                             # [B, d]
#         Q    = self.dec_pe(self.horizon_queries.expand(B, -1, -1))
#         D    = self.transformer_dec(Q, fmem)            # [B, H, d]
#         lp   = self.reg_head(D)                         # [B, H, 2]
#         return lp, ctx

#     # ── Gate blend ────────────────────────────────────────────────────

#     def _gate_blend(self,
#                     lp_h:       torch.Tensor,          # [B_h, H, 2]
#                     ctx_h:      torch.Tensor,          # [B_h, d]
#                     steer:      Optional[torch.Tensor],# [B_h, 2] hoặc None
#                     obs_h:      torch.Tensor,          # [T_obs, B_h, 2]
#                     stop_ctx_grad: bool = True,
#                     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Physics persistence blend:
#           physics[t] = obs_h[-1] + (t+1)*steer        [B_h, 2]
#           alpha[t]   = sigmoid(gate([ctx, step_emb, steer]))
#           final[t]   = alpha[t]*lp_h[t] + (1-alpha[t])*physics[t]

#         stop_ctx_grad=True  (training):
#           → gate KHÔNG update encoder qua ctx gradient
#           → encoder gradient chỉ từ easy + recurv_head
#         stop_ctx_grad=False (inference):
#           → full forward, không cần gradient
#         """
#         B_h = lp_h.shape[0]
#         T   = self.pred_len
#         dev = lp_h.device

#         # Steer fallback: last observed velocity
#         if steer is None:
#             steer = (obs_h[-1] - obs_h[-2]
#                      if obs_h.shape[0] >= 2
#                      else torch.zeros(B_h, 2, device=dev))

#         # Physics trajectory: [B_h, T, 2]
#         steps   = torch.arange(1, T+1, device=dev, dtype=steer.dtype)
#         offsets = steer.unsqueeze(1) * steps.view(1, T, 1)
#         physics = obs_h[-1].unsqueeze(1) + offsets

#         # Gate (vectorized, 1 forward pass)
#         ctx_in  = ctx_h.detach() if stop_ctx_grad else ctx_h
#         t_idx   = torch.arange(T, device=dev)
#         s_emb   = self.steering_gate.step_emb(t_idx)        # [T, 16]
#         ctx_e   = ctx_in.unsqueeze(0).expand(T, -1, -1)     # [T, B_h, d]
#         steer_e = steer.unsqueeze(0).expand(T, -1, -1)      # [T, B_h, 2]
#         s_emb_e = s_emb.unsqueeze(1).expand(T, B_h, -1)    # [T, B_h, 16]
#         gate_in = torch.cat([ctx_e, s_emb_e, steer_e], -1) # [T, B_h, d+18]
#         alpha   = torch.sigmoid(
#             self.steering_gate.gate_net(gate_in))            # [T, B_h, 1]

#         # Blend: [B_h, T, 2]
#         alpha_t = alpha.permute(1, 0, 2)                    # [B_h, T, 1]
#         final_h = alpha_t * lp_h + (1 - alpha_t) * physics

#         return final_h, alpha.mean()

#     # ── Easy losses: GIỐNG HỆT ST-TRANS GỐC ─────────────────────────

#     def _easy_losses(self, pred: torch.Tensor,
#                      gt: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """
#         pred, gt: [T, B_e, 2]
#         Fixed weights, không UW — giống ST-Trans gốc.
#         Gradient đầy đủ → decoder + encoder.
#         """
#         T = min(pred.shape[0], gt.shape[0])
#         p, g = pred[:T], gt[:T]

#         # DPE: uniform haversine mean (không step_weights)
#         l_dpe = haversine_km(_norm_to_deg(p), _norm_to_deg(g)).mean()
#         l_mse = F.mse_loss(p, g)

#         l_speed = p.new_zeros(())
#         if T >= 2:
#             sd      = (p[1:] - p[:-1]).norm(-1)
#             l_speed = F.relu(sd - self.v_max_norm).pow(2).mean()

#         l_accel = p.new_zeros(())
#         if T >= 3:
#             v       = p[1:] - p[:-1]
#             l_accel = (v[1:].norm(-1) - v[:-1].norm(-1)).pow(2).mean()

#         return {"dpe": l_dpe, "mse": l_mse,
#                 "speed": l_speed, "accel": l_accel}

#     # ── Hard losses: GATE-CORRECTED ───────────────────────────────────

#     def _hard_losses(self,
#                      final_h: torch.Tensor,  # [B_h, H, 2] gate output
#                      gt_h:    torch.Tensor,  # [T_pred, B_h, 2]
#                      ctx_h:   torch.Tensor,  # [B_h, d] gốc (có gradient)
#                      obs_h:   torch.Tensor,  # [T_obs, B_h, 2]
#                      ) -> Dict[str, torch.Tensor]:
#         """
#         final_h: lp_h.detach() đã apply trước khi gọi gate_blend
#           → gradient từ L_dpe, L_heading đi: final_h → alpha → gate
#           → gate KHÔNG update encoder (stop_ctx_grad=True trong gate)

#         ctx_h (gốc): được dùng bởi recurv_head
#           → gradient từ L_recurv đi: ctx_h → encoder
#           → encoder học classify recurvature (auxiliary, weight nhỏ)
#         """
#         T   = min(final_h.shape[1], gt_h.shape[0])
#         fp  = final_h[:, :T]                        # [B_h, T, 2]
#         g   = gt_h[:T].permute(1, 0, 2)             # [B_h, T, 2]
#         fpt = fp.permute(1, 0, 2)                   # [T, B_h, 2]
#         gt  = g.permute(1, 0, 2)                    # [T, B_h, 2]

#         # L_DPE: uniform mean trên gate output
#         l_dpe = haversine_km(_norm_to_deg(fpt), _norm_to_deg(gt)).mean()

#         # L_heading: cosine similarity giữa pred và gt direction
#         l_heading = fp.new_zeros(())
#         if T >= 2:
#             pd = fpt[1:] - fpt[:-1]                 # [T-1, B_h, 2]
#             gd = gt[1:]  - gt[:-1]
#             pn = F.normalize(pd, dim=-1, eps=1e-8)
#             gn = F.normalize(gd, dim=-1, eps=1e-8)
#             l_heading = ((1.0 - (pn*gn).sum(-1).clamp(-1,1)) / 2.0).mean()

#         # L_recurv: auxiliary CE — ctx_h GỐC (có gradient → encoder)
#         l_recurv   = fp.new_zeros(())
#         recurv_log = self.recurv_head(ctx_h)         # [B_h, pred_len+1]
#         try:
#             label = RecurvatureTimingHead.make_label(
#                 gt_h, obs_h, self.recurv_threshold)
#             if label.shape[0] == recurv_log.shape[0]:
#                 l_recurv = F.cross_entropy(recurv_log, label)
#         except Exception:
#             pass

#         return {"dpe": l_dpe, "heading": l_heading, "recurv": l_recurv}

#     # ── Training loss ─────────────────────────────────────────────────

#     def get_loss_breakdown(self, batch_list) -> Dict:
#         obs_traj = batch_list[0]   # [T_obs, B, 2]
#         traj_gt  = batch_list[1]   # [T_pred, B, 2]

#         # 1. Classify easy / hard
#         is_hard = classify_hard_obs(
#             obs_traj, self.threshold_curv, self.threshold_spd)
#         is_easy = ~is_hard
#         n_easy  = int(is_easy.sum())
#         n_hard  = int(is_hard.sum())

#         # 2. Encode: 1 shared forward pass
#         lp, ctx = self._encode(batch_list)   # [B,H,2], [B,d]

#         result = {"n_easy": n_easy, "n_hard": n_hard}

#         # ── EASY: = ST-Trans gốc ─────────────────────────────────────
#         # Gradient đầy đủ: L_easy → lp_e → decoder + encoder
#         L_easy = None
#         if n_easy > 0:
#             lp_e = lp[is_easy].permute(1, 0, 2)   # [T, B_e, 2]
#             gt_e = traj_gt[:, is_easy]             # [T, B_e, 2]
#             el   = self._easy_losses(lp_e, gt_e)

#             # FIXED weights — không UW (như ST-Trans gốc)
#             L_easy = (
#                 el["dpe"]
#                 + self.w_mse        * el["mse"]
#                 + self.lambda_speed * el["speed"]
#                 + self.lambda_accel * el["accel"]
#             )
#             result.update({
#                 "easy_dpe":   el["dpe"].item(),
#                 "easy_mse":   el["mse"].item(),
#             })

#         # ── HARD: gate-corrected ─────────────────────────────────────
#         # lp_h.detach()      → decoder KHÔNG nhận gradient từ L_hard
#         # stop_ctx_grad=True → encoder KHÔNG nhận gradient từ gate
#         # ctx_h (gốc) vào recurv_head → encoder học classify (auxiliary)
#         L_hard = None
#         if n_hard > 0:
#             lp_h  = lp[is_hard]               # [B_h, H, 2]
#             gt_h  = traj_gt[:, is_hard]       # [T, B_h, 2]
#             obs_h = obs_traj[:, is_hard]      # [T_obs, B_h, 2]
#             ctx_h = ctx[is_hard]              # [B_h, d]

#             data3d = None
#             if (len(batch_list) > 2
#                     and isinstance(batch_list[2], torch.Tensor)
#                     and batch_list[2].dim() == 4):
#                 data3d = batch_list[2][is_hard]
#             steer = _get_steering(data3d, obs_h)

#             # Gate với lp_h.detach() + stop_ctx_grad → bảo vệ decoder + encoder
#             final_h, alpha_m = self._gate_blend(
#                 lp_h.detach(), ctx_h, steer, obs_h,
#                 stop_ctx_grad=True)

#             hl = self._hard_losses(final_h, gt_h, ctx_h, obs_h)

#             # UW cho hard losses
#             L_hard = (
#                 self.uw.weight("dpe_hard", hl["dpe"])
#                 + self.uw.weight("heading", hl["heading"])
#                 + self.uw.weight("recurv",  hl["recurv"])
#             )
#             result.update({
#                 "hard_dpe":    hl["dpe"].item(),
#                 "hard_heading":(hl["heading"].item()
#                                 if isinstance(hl["heading"], torch.Tensor)
#                                 else 0.0),
#                 "alpha_mean":  (alpha_m.item()
#                                 if isinstance(alpha_m, torch.Tensor)
#                                 else 0.0),
#             })

#         # 3. Combine
#         if L_easy is not None and L_hard is not None:
#             total = L_easy + L_hard
#         elif L_easy is not None:
#             total = L_easy
#         else:
#             total = L_hard

#         result["total"] = total
#         result.update(self.uw.sigma_dict())

#         return result

#     def get_loss(self, batch_list) -> torch.Tensor:
#         return self.get_loss_breakdown(batch_list)["total"]

#     # ── Inference ─────────────────────────────────────────────────────

#     def forward(self, batch_list) -> torch.Tensor:
#         """
#         Easy  → decoder output (= ST-Trans gốc)
#         Hard  → gate-blended   (tốt hơn nhờ physics correction)
#         Return: [T, B, 2]
#         """
#         lp, ctx = self._encode(batch_list)
#         obs     = batch_list[0]                        # [T_obs, B, 2]

#         # Default: decoder output cho tất cả
#         pred = lp.permute(1, 0, 2).clone()             # [T, B, 2]

#         # Hard: override bằng gate-blended
#         is_hard = classify_hard_obs(
#             obs, self.threshold_curv, self.threshold_spd)

#         if is_hard.any():
#             obs_h = obs[:, is_hard]
#             ctx_h = ctx[is_hard]
#             lp_h  = lp[is_hard]                        # [B_h, H, 2]

#             data3d = None
#             if (len(batch_list) > 2
#                     and isinstance(batch_list[2], torch.Tensor)
#                     and batch_list[2].dim() == 4):
#                 data3d = batch_list[2][is_hard]
#             steer = _get_steering(data3d, obs_h)

#             # Inference: không cần stop_ctx_grad (không backward)
#             final_h, _ = self._gate_blend(
#                 lp_h, ctx_h, steer, obs_h,
#                 stop_ctx_grad=False)

#             pred[:, is_hard] = final_h.permute(1, 0, 2)  # [T, B_h, 2]

#         return pred   # [T, B, 2]

#     @torch.no_grad()
#     def sample(self, batch_list, **kwargs):
#         pred    = self.forward(batch_list)   # [T, B, 2]
#         T, B, _ = pred.shape
#         return pred, torch.zeros(T, B, 2, device=pred.device), pred.unsqueeze(0)


# # ══════════════════════════════════════════════════════════════════════
# #  Factory
# # ══════════════════════════════════════════════════════════════════════

# def build_st_trans_v2(args, threshold_curv=15.0, threshold_spd=0.5):
#     return STTransV2(
#         obs_len          = getattr(args, "obs_len",          8),
#         pred_len         = getattr(args, "pred_len",         12),
#         unet_in_ch       = getattr(args, "unet_in_ch",       13),
#         d_model          = getattr(args, "d_model",          64),
#         nhead            = getattr(args, "nhead",            4),
#         num_enc_layers   = getattr(args, "num_enc_layers",   1),
#         num_dec_layers   = getattr(args, "num_dec_layers",   3),
#         dim_ff           = getattr(args, "dim_ff",           512),
#         dropout          = getattr(args, "dropout",          0.1),
#         w_mse            = getattr(args, "w_mse",            0.05),
#         lambda_speed     = getattr(args, "lambda_speed",     0.1),
#         lambda_accel     = getattr(args, "lambda_accel",     0.01),
#         v_max_kmh        = getattr(args, "v_max_kmh",        80.0),
#         recurv_threshold = getattr(args, "recurv_threshold", 45.0),
#         gate_hidden      = getattr(args, "gate_hidden",      32),
#         recurv_hidden    = getattr(args, "recurv_hidden",    64),
#         threshold_curv   = threshold_curv,
#         threshold_spd    = threshold_spd,
#     )

"""
Model/st_trans_v3_model.py  —— ST-Trans v3
============================================
Mở rộng ST-Trans v2 với 3 chiến lược dài hạn để đạt < 160 km ADE:

CHIẾN LƯỢC 1 — ERA5 Wind Steering cho TẤT CẢ samples (không chỉ hard)
  v2: steer chỉ được dùng trong gate của hard samples
  v3: steer inject vào full_memory qua SteeringContextEncoder (2-layer MLP)
      → tất cả easy + hard samples đều "biết" hướng gió lúc encode
  Lý do: ERA5 500hPa steering wind chi phối hướng di chuyển của cả bão không
  recurvature lẫn bão recurvature. Bỏ đi là bỏ đi signal vật lý quan trọng nhất.

CHIẾN LƯỢC 2 — d_model 64 → 128 (decoder capacity)
  v2: d_model=64, decoder ~400K params → decoder underfit (easy_ADE 220 km)
  v3: d_model=128, decoder ~1.6M params → +4x capacity cho attention cross-attn
  Encoder (FNO3D+Mamba): vẫn output RAW_CTX_DIM=512, được project xuống 128
  Không thay đổi encoder backbone → params encoder không tăng nhiều
  Tổng params ước tính: ~4.5M (tăng ~1.2M so với v2)

CHIẾN LƯỢC 3 — RoPE thay Sinusoidal PE cho horizon queries
  v2: SinusoidalPE cộng vào horizon queries → thông tin vị trí không phân biệt
      được phụ thuộc tương đối giữa các bước dự báo (bước 6h vs 12h vs 72h)
  v3: RoPEHorizonPE dùng Rotary Position Embedding: rotate Q và K trong
      attention theo góc phụ thuộc vị trí tương đối → cross-attention biết
      "bước dự báo này cách bước kia bao nhiêu" → tốt hơn cho long-horizon
  Implement đúng: apply RoPE vào Q,K của TỪNG decoder layer (monkey-patch)

BONUS FIX — Easy sample gradient protection
  v2 bug tiềm ẩn: khi batch size nhỏ và hard fraction cao, easy path chỉ có
  ~55 samples → gradient noisy → easy_ADE cao (220 km vs target 155 km)
  v3 fix: StepWeightedDPE với increasing horizon weights để force decoder học
  tốt hơn ở long horizon thay vì chỉ tối ưu average

SO SÁNH PARAMS:
  v2: d_model=64   → decoder ~400K  → total ~3.28M
  v3: d_model=128  → decoder ~1.6M  → total ~4.5M  (ước tính)

GRADIENT FLOW (giữ nguyên design v2):
  Encoder:  easy dominant + recurv auxiliary
  Decoder:  CHỈ từ easy (lp_h.detach() cho hard)
  Gate:     ctx.detach() + steer → KHÔNG update encoder
  Steering: gradient FULL qua SteeringContextEncoder (easy + hard đều benefit)
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.paper_baseline_model import (
    PaperEncoder,
    _norm_to_deg,
    haversine_km,
    compute_ade_per_horizon,
    compute_ate_cte_per_horizon,
    HORIZON_STEPS,
)

# ── Constants ────────────────────────────────────────────────────────────────
_U500_CH           = 0          # channel index của U-wind 500hPa trong data3d
_V500_CH           = 1          # channel index của V-wind 500hPa
_DEG_SCALE         = 25.0       # norm → degree scale factor
_MS_TO_NORM_PER_6H = (6.0 * 3600.0) / (111320.0 * _DEG_SCALE)


# ══════════════════════════════════════════════════════════════════════════════
#  CHIẾN LƯỢC 1 — ERA5 Steering Context Encoder (cho TẤT CẢ samples)
# ══════════════════════════════════════════════════════════════════════════════

class SteeringContextEncoder(nn.Module):
    """
    Encode ERA5 wind steering vector → context token thêm vào full_memory.

    Input:  steer [B, 2]  (u_norm, v_norm) — normalised displacement per step
            steer_magnitude [B, 1] — tổng speed
            steer_direction [B, 2] — sin/cos của hướng gió
    Output: steer_token [B, 1, d_model]  — thêm vào memory sequence

    Tại sao dùng token riêng thay vì cộng vào ctx?
    - Decoder cross-attention có thể attend selectively vào steer_token
    - Không bị "trung bình hoá" với obs signal trong ctx pooling
    - Cho phép decoder học: "ở bước t xa, tôi cần attend nhiều hơn vào steer"

    Fallback: nếu không có ERA5 data → steer_token = zero token
    (tức là model vẫn chạy được với data không có wind field)
    """

    STEER_FEAT_DIM = 5  # [u_norm, v_norm, magnitude, sin_dir, cos_dir]

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.encoder = nn.Sequential(
            nn.Linear(self.STEER_FEAT_DIM, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )
        # Gate để model học khi nào cần trust steering signal
        # Nếu steer = 0 (fallback) → gate → ~0 → token không contribute
        self.gate = nn.Sequential(
            nn.Linear(self.STEER_FEAT_DIM, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def extract_steer(data3d: Optional[torch.Tensor],
                      obs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Trích xuất ERA5 steering vector từ center pixel của patch.
        data3d: [B, C, H, W] — ERA5 patch
        obs:    [T_obs, B, 2] — để lấy latitude cho cos correction

        Returns: [B, 2] normalised displacement, hoặc None nếu không có data
        """
        if (data3d is None
                or not isinstance(data3d, torch.Tensor)
                or data3d.dim() != 4):
            return None
        B, C, H, W = data3d.shape
        if C <= max(_U500_CH, _V500_CH):
            return None
        cy, cx  = H // 2, W // 2
        u_ms    = data3d[:, _U500_CH, cy, cx]   # [B]
        v_ms    = data3d[:, _V500_CH, cy, cx]   # [B]
        lat_deg = obs[-1, :, 1] * _DEG_SCALE    # [B] — dùng step cuối obs
        cos_lat = torch.cos(lat_deg * (math.pi / 180)).clamp(0.1, 1.0)
        u_norm  = (u_ms / cos_lat) * _MS_TO_NORM_PER_6H
        v_norm  = v_ms * _MS_TO_NORM_PER_6H
        return torch.stack([u_norm, v_norm], dim=-1)  # [B, 2]

    @staticmethod
    def steer_to_features(steer: Optional[torch.Tensor],
                          B: int,
                          device: torch.device,
                          dtype: torch.dtype) -> torch.Tensor:
        """
        Chuyển steer [B, 2] → rich features [B, 5].
        Nếu steer = None → zero features (fallback an toàn).
        """
        if steer is None:
            return torch.zeros(B, SteeringContextEncoder.STEER_FEAT_DIM,
                               device=device, dtype=dtype)
        u, v   = steer[:, 0], steer[:, 1]
        mag    = (u.pow(2) + v.pow(2)).sqrt()           # magnitude
        eps    = 1e-8
        sin_d  = v / (mag + eps)                        # sin of direction
        cos_d  = u / (mag + eps)                        # cos of direction
        return torch.stack([u, v, mag, sin_d, cos_d], dim=-1)  # [B, 5]

    def forward(self,
                steer: Optional[torch.Tensor],
                B: int,
                device: torch.device,
                dtype: torch.dtype) -> torch.Tensor:
        """→ steer_token [B, 1, d_model]"""
        feat = self.steer_to_features(steer, B, device, dtype)  # [B, 5]
        g    = self.gate(feat)                                    # [B, 1]
        emb  = self.encoder(feat)                                 # [B, d_model]
        # Gate: nếu steer ≈ 0 → emb * gate ≈ 0 → zero token (safe fallback)
        return (emb * g).unsqueeze(1)                             # [B, 1, d_model]


# ══════════════════════════════════════════════════════════════════════════════
#  CHIẾN LƯỢC 3 — RoPE cho Horizon Queries
# ══════════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (Su et al. 2021, RoFormer).

    Thay vì cộng PE vào vector (additive như Sinusoidal),
    RoPE ROTATE Q và K theo góc phụ thuộc position → cross-attention
    tự động encode RELATIVE position giữa query[i] và key[j].

    Tại sao tốt hơn cho horizon queries?
    - Horizon query tại step t=6h cần biết "tôi cách step t=12h bao nhiêu"
    - Additive PE chỉ encode absolute position → decoder không distinguish
      tốt relative distance trong attention
    - RoPE: cos(t·θ_i) và sin(t·θ_i) trực tiếp modulate Q,K →
      dot product Q[t]·K[s] ∝ f(t-s) → relative distance được encode

    Implementation: trả về (cos_emb, sin_emb) để apply trong attention.
    """

    def __init__(self, dim: int, max_seq: int = 64, base: float = 100.0):
        """
        base=100 (thay vì 10000 mặc định của RoPE gốc).

        Lý do: pred_len=12 steps là chuỗi rất ngắn.
        Với base=10000: inv_freq[0]=1.0 → freq[t=12]=12.0 → cos(12)≈0.84,
        sin(12)≈-0.54 → tần số quá cao, không smooth cho 12 steps.
        Với base=100:  inv_freq[0]≈0.01 → freq[t=12]=0.12 → cos,sin smooth
        → relative position signal rõ ràng hơn trong short-horizon attention.
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE dim phải chẵn"
        # θ_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq = max_seq
        self._build_cache(max_seq)

    def _build_cache(self, max_seq: int):
        t     = torch.arange(max_seq, dtype=self.inv_freq.dtype,
                             device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [max_seq, dim/2]
        emb   = torch.cat([freqs, freqs], dim=-1)           # [max_seq, dim]
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0))  # [1, S, dim]
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0))  # [1, S, dim]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """[-x1, x0] interleaved pattern."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*H, seq_len, d_head]
        Returns: x với RoPE applied

        Lazy rebuild nếu cache device không match x.device (BUG 1 fix).
        Điều này xảy ra khi model.to(cuda) sau khi __init__ trên CPU.
        register_buffer tự di chuyển với .to() nhưng _build_cache()
        cần được gọi lại một lần nếu cache bị stale.
        """
        S   = x.size(1)
        # Rebuild cache nếu device mismatch (an toàn, chỉ xảy ra 1 lần)
        if self.cos_cached.device != x.device or S > self.cos_cached.size(1):
            self._build_cache(max(S + 4, self.max_seq))
        cos = self.cos_cached[:, :S, :].to(x.dtype)
        sin = self.sin_cached[:, :S, :].to(x.dtype)
        return x * cos + self._rotate_half(x) * sin


class RoPETransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer với RoPE applied vào Q,K của self-attention
    VÀ cross-attention (cho cả horizon position và memory position).

    Thay thế nn.TransformerDecoderLayer built-in để inject RoPE vào
    đúng vị trí trong attention computation.

    Architecture giống hệt PyTorch TransformerDecoderLayer:
    - Self-attention (masked) với RoPE trên Q,K của horizon queries
    - Cross-attention với memory (RoPE trên horizon Q, memory K vẫn sinusoidal)
    - FFN với dim_feedforward, dropout, LayerNorm
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, max_horizon: int = 24):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model    = d_model
        self.nhead      = nhead
        self.d_head     = d_model // nhead
        self.scale      = self.d_head ** -0.5

        # Self-attention projections
        self.sa_q = nn.Linear(d_model, d_model, bias=False)
        self.sa_k = nn.Linear(d_model, d_model, bias=False)
        self.sa_v = nn.Linear(d_model, d_model, bias=False)
        self.sa_o = nn.Linear(d_model, d_model)

        # Cross-attention projections
        self.ca_q = nn.Linear(d_model, d_model, bias=False)
        self.ca_k = nn.Linear(d_model, d_model, bias=False)
        self.ca_v = nn.Linear(d_model, d_model, bias=False)
        self.ca_o = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

        # RoPE — dim = d_head (apply per head)
        self.rope = RotaryEmbedding(dim=self.d_head, max_seq=max_horizon + 10)

        self._init_weights()

    def _init_weights(self):
        for m in [self.sa_q, self.sa_k, self.sa_v, self.sa_o,
                  self.ca_q, self.ca_k, self.ca_v, self.ca_o]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, D] → [B, H, S, d_head]"""
        B, S, D = x.shape
        return x.view(B, S, self.nhead, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, S, d_head] → [B, S, D]"""
        B, H, S, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, H * d)

    def _apply_rope_per_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, S, d_head]
        Apply RoPE theo S dimension cho mỗi head.
        RoPE được define trên d_head → cần reshape.
        """
        B, H, S, d = x.shape
        # Reshape thành [B*H, S, d_head] để apply RoPE
        x_2d = x.reshape(B * H, S, d)
        x_rope = self.rope(x_2d)           # [B*H, S, d_head]
        return x_rope.view(B, H, S, d)

    def forward(self,
                tgt: torch.Tensor,           # [B, H_q, d_model] horizon queries
                memory: torch.Tensor,        # [B, S_mem, d_model] encoder output
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        tgt:    [B, pred_len, d_model]  — horizon queries
        memory: [B, S_mem, d_model]     — full_memory từ encoder
        Returns: [B, pred_len, d_model]
        """
        # ── Self-attention (horizon queries attend to each other) ──────────
        # RoPE apply vào Q, K để encode relative position giữa các horizon steps
        sa_q = self._split_heads(self.sa_q(tgt))   # [B, H, P, d_head]
        sa_k = self._split_heads(self.sa_k(tgt))
        sa_v = self._split_heads(self.sa_v(tgt))

        sa_q = self._apply_rope_per_head(sa_q)     # RoPE trên Q
        sa_k = self._apply_rope_per_head(sa_k)     # RoPE trên K

        attn = torch.matmul(sa_q, sa_k.transpose(-2, -1)) * self.scale
        if tgt_mask is not None:
            attn = attn + tgt_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        sa_out = self._merge_heads(torch.matmul(attn, sa_v))
        tgt = self.norm1(tgt + self.drop(self.sa_o(sa_out)))

        # ── Cross-attention (horizon queries attend to encoder memory) ─────
        # RoPE trên Q (horizon): relative position trong prediction sequence
        # K (memory): không apply RoPE vì memory không có temporal ordering
        #             theo cùng không gian với horizon
        ca_q = self._split_heads(self.ca_q(tgt))       # [B, H, P, d_head]
        ca_k = self._split_heads(self.ca_k(memory))    # [B, H, S_mem, d_head]
        ca_v = self._split_heads(self.ca_v(memory))

        ca_q = self._apply_rope_per_head(ca_q)         # RoPE chỉ trên Q

        cross_attn = torch.matmul(ca_q, ca_k.transpose(-2, -1)) * self.scale
        cross_attn = F.softmax(cross_attn, dim=-1)
        cross_attn = self.drop(cross_attn)

        ca_out = self._merge_heads(torch.matmul(cross_attn, ca_v))
        tgt = self.norm2(tgt + self.drop(self.ca_o(ca_out)))

        # ── FFN ───────────────────────────────────────────────────────────
        tgt = self.norm3(tgt + self.drop(self.ffn(tgt)))

        return tgt


class RoPETransformerDecoder(nn.Module):
    """Stack of RoPETransformerDecoderLayer."""

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float, max_horizon: int):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPETransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, max_horizon=max_horizon)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask)
        return self.norm(tgt)


# ══════════════════════════════════════════════════════════════════════════════
#  Sinusoidal PE (giữ cho obs encoder — không cần thay)
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 300):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ══════════════════════════════════════════════════════════════════════════════
#  Obs Kinematic Encoder (CHIẾN LƯỢC 2: cập nhật d_model=128)
# ══════════════════════════════════════════════════════════════════════════════

class ObsKinematicEncoder(nn.Module):
    """
    [T_obs, B, 2] → [B, T_obs, d_model].

    Tăng d_model từ 64 → 128 (CHIẾN LƯỢC 2).
    Thêm 2 features mới: bearing (hướng tuyệt đối so với north) và
    turning_rate (tốc độ thay đổi bearing) → 10D thay vì 8D.
    """
    FEAT_DIM = 10  # lon, lat, d_lon, d_lat, dd_lon, dd_lat, step_idx, speed, bearing, turn_rate

    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 2,
                 dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(self.FEAT_DIM, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.pe  = SinusoidalPE(d_model, max_len=64)
        enc      = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)  # Pre-LN: ổn định hơn khi train
        self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)

    @staticmethod
    def _feats(obs: torch.Tensor) -> torch.Tensor:
        """obs [T, B, 2] → features [B, T, 10]"""
        T, B, _ = obs.shape
        dev = obs.device
        zeros1 = torch.zeros(1, B, device=dev, dtype=obs.dtype)

        lon, lat = obs[:, :, 0], obs[:, :, 1]

        # Velocity
        if T >= 2:
            dl = torch.cat([obs[1:,:,0] - obs[:-1,:,0], zeros1], 0)
            dt = torch.cat([obs[1:,:,1] - obs[:-1,:,1], zeros1], 0)
        else:
            dl = torch.zeros_like(lon)
            dt = torch.zeros_like(lon)

        # Acceleration
        if T >= 3:
            ddl = torch.cat([dl[1:] - dl[:-1], zeros1], 0)
            ddt = torch.cat([dt[1:] - dt[:-1], zeros1], 0)
        else:
            ddl = torch.zeros_like(lon)
            ddt = torch.zeros_like(lon)

        step_idx = torch.linspace(0, 1, T, device=dev, dtype=obs.dtype).unsqueeze(1).expand(T, B)
        speed    = (dl.pow(2) + dt.pow(2)).sqrt()

        # Bearing: atan2(d_lon, d_lat) — hướng di chuyển so với north
        # Normalize về [-1, 1] bằng cách chia cho π
        bearing  = torch.atan2(dl, dt + 1e-8) / math.pi  # [-1, 1]

        # Turning rate: thay đổi bearing giữa các bước
        if T >= 2:
            # Circular difference (wrap around ±π)
            b_diff = bearing[1:] - bearing[:-1]
            # Wrap: đưa về [-1, 1]
            b_diff = (b_diff + 1.0) % 2.0 - 1.0
            turn_rate = torch.cat([b_diff, zeros1], 0)
        else:
            turn_rate = torch.zeros_like(lon)

        feat = torch.stack([lon, lat, dl, dt, ddl, ddt,
                            step_idx, speed, bearing, turn_rate], dim=-1)
        return feat.permute(1, 0, 2)   # [B, T, 10]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.enc(self.pe(self.proj(self._feats(obs))))


# ══════════════════════════════════════════════════════════════════════════════
#  Easy/Hard Classifier (giữ từ v2)
# ══════════════════════════════════════════════════════════════════════════════

def classify_hard_obs(obs_traj: torch.Tensor,
                      threshold_curv: float,
                      threshold_spd: float) -> torch.Tensor:
    """[T_obs, B, 2] → [B] bool. True = hard (recurvature/erratic)."""
    T, B, _ = obs_traj.shape
    device = obs_traj.device
    with torch.no_grad():
        if T < 2:
            return torch.zeros(B, dtype=torch.bool, device=device)
        vel    = obs_traj[1:] - obs_traj[:-1]
        spd    = vel.norm(dim=-1)
        spd_cv = spd.std(0) / (spd.mean(0) + 1e-6)
        if T >= 3:
            vn   = F.normalize(vel, dim=-1, eps=1e-8)
            cos  = (vn[1:] * vn[:-1]).sum(-1).clamp(-1, 1)
            curv = (torch.acos(cos) * (180 / math.pi)).mean(0)
        else:
            curv = torch.zeros(B, device=device)
        return (curv > threshold_curv) | (spd_cv > threshold_spd)


def compute_hard_thresholds(train_loader, device,
                             percentile: float = 70.0) -> Tuple[float, float]:
    import numpy as np
    all_curv, all_spd = [], []
    with torch.no_grad():
        for batch in train_loader:
            obs = batch[0].to(device)
            T, B, _ = obs.shape
            if T < 2:
                continue
            vel = obs[1:] - obs[:-1]
            spd = vel.norm(dim=-1)
            all_spd.extend((spd.std(0) / (spd.mean(0) + 1e-6)).cpu().tolist())
            if T >= 3:
                vn  = F.normalize(vel, dim=-1, eps=1e-8)
                cos = (vn[1:] * vn[:-1]).sum(-1).clamp(-1, 1)
                all_curv.extend((torch.acos(cos) * (180 / math.pi)).mean(0).cpu().tolist())
    if not all_curv:
        return 15.0, 0.5
    return (float(np.percentile(all_curv, percentile)),
            float(np.percentile(all_spd, percentile)))


# ══════════════════════════════════════════════════════════════════════════════
#  Uncertainty Weighting (giữ từ v2)
# ══════════════════════════════════════════════════════════════════════════════

class UncertaintyWeighting(nn.Module):
    """
    Kendall et al. 2018: L_weighted = L / (2σ²) + log(σ)
    Chỉ 3 tasks cho hard path. Easy path dùng fixed weights.
    """
    TASKS = ["dpe_hard", "heading", "recurv"]
    INIT  = {
        "dpe_hard": -0.5,
        "heading":   0.3,
        "recurv":    1.0,
    }

    def __init__(self):
        super().__init__()
        inits = torch.tensor([self.INIT[t] for t in self.TASKS])
        self.log_sigma = nn.Parameter(inits)
        self._idx = {t: i for i, t in enumerate(self.TASKS)}

    def weight(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        s = self.log_sigma[self._idx[name]]
        return torch.exp(-2.0 * s) * loss / 2.0 + s

    def sigma_dict(self) -> Dict[str, float]:
        s = torch.exp(self.log_sigma).detach()
        return {f"σ_{t}": s[i].item() for i, t in enumerate(self.TASKS)}


# ══════════════════════════════════════════════════════════════════════════════
#  Physics Steering Gate (nâng cấp: steer input mở rộng)
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsSteeringGate(nn.Module):
    """
    Vectorized gate: 1 forward pass cho tất cả T bước.
    α[t] = sigmoid(W·[ctx, step_emb[t], steer_feat] + b)
    Init b=2.0 → α≈0.88 (lean về learned ban đầu)

    v3 upgrade: steer input là 5D features (thay vì 2D raw)
    để gate có thêm magnitude và direction info.
    """
    def __init__(self, ctx_dim: int = 128, steer_feat_dim: int = 5,
                 hidden: int = 64, pred_len: int = 12):
        super().__init__()
        self.pred_len = pred_len
        self.step_emb = nn.Embedding(pred_len, 16)
        self.gate_net = nn.Sequential(
            nn.Linear(ctx_dim + 16 + steer_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1))
        nn.init.constant_(self.gate_net[-1].bias, 2.0)
        nn.init.zeros_(self.gate_net[-1].weight)


class RecurvatureTimingHead(nn.Module):
    """Auxiliary classifier: predict bước recurvature. Giống v2."""
    def __init__(self, ctx_dim: int = 128, pred_len: int = 12, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, pred_len + 1))

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(ctx)

    @staticmethod
    def make_label(gt: torch.Tensor, obs: torch.Tensor,
                   threshold_deg: float = 45.0) -> torch.Tensor:
        T, B, _ = gt.shape
        dev   = gt.device
        label = torch.zeros(B, dtype=torch.long, device=dev)
        full  = torch.cat([obs[-1:], gt], 0)
        with torch.no_grad():
            for t in range(T - 1):
                d_in  = full[t + 1] - full[t]
                d_out = full[t + 2] - full[t + 1]
                ni    = F.normalize(d_in,  dim=-1, eps=1e-8)
                no    = F.normalize(d_out, dim=-1, eps=1e-8)
                ang   = torch.acos((ni * no).sum(-1).clamp(-1, 1)) * (180 / math.pi)
                mask  = (ang > threshold_deg) & (label == 0)
                label[mask] = t + 1
        return label


# ══════════════════════════════════════════════════════════════════════════════
#  BONUS — Step-Weighted DPE Loss
# ══════════════════════════════════════════════════════════════════════════════

def step_weighted_dpe(pred_deg: torch.Tensor,
                      gt_deg: torch.Tensor,
                      ramp: float = 2.0) -> torch.Tensor:
    """
    DPE với weights tăng dần theo horizon step.

    Lý do: ADE = mean haversine → đối xử bước 6h và 72h như nhau.
    Nhưng lỗi tại 72h nặng hơn nhiều về mặt ứng dụng (và ADE metric),
    vì nó có trị số lớn hơn và dominant hơn trong trung bình.

    weights[t] = softmax([1, 1+ramp/(T-1), ..., 1+ramp])
    → bước cuối (72h) có weight ~exp(ramp) lần bước đầu (6h)

    pred_deg, gt_deg: [T, B, 2] in degrees
    Returns: scalar weighted mean haversine distance
    """
    T = pred_deg.shape[0]
    dist = haversine_km(pred_deg, gt_deg)  # [T, B]

    # Tạo weights tăng tuyến tính rồi softmax → sum = 1
    raw_w = torch.linspace(1.0, 1.0 + ramp, T,
                           device=pred_deg.device, dtype=pred_deg.dtype)
    w = F.softmax(raw_w, dim=0).unsqueeze(1)  # [T, 1]

    return (dist * w).sum(0).mean()  # scalar


# ══════════════════════════════════════════════════════════════════════════════
#  STTransV3 — Main Model
# ══════════════════════════════════════════════════════════════════════════════

class STTransV3(nn.Module):
    """
    ST-Trans v3: Physics-guided Non-Autoregressive Transformer.

    Thêm 3 chiến lược so với v2:

    1. SteeringContextEncoder: ERA5 wind token → full_memory cho TẤT CẢ samples
       Memory sequence: [ctx_token, steer_token, obs_memory_0, ..., obs_memory_{T-1}]
       Length: 1 + 1 + T_obs = T_obs + 2

    2. d_model = 128 (tăng từ 64): decoder capacity tăng 4x
       ObsKinematicEncoder: 10D features + 2 layers + nhead=8

    3. RoPETransformerDecoder: thay nn.TransformerDecoder
       RoPE apply vào Q,K của self-attn và cross-attn Q
       → relative position giữa horizon steps được encode chính xác

    4. StepWeightedDPE: easy loss dùng horizon-weighted DPE (bonus fix)
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        unet_in_ch: int = 13,
        # CHIẾN LƯỢC 2: d_model tăng lên 128
        d_model: int = 128,
        nhead: int = 8,
        num_enc_layers: int = 2,
        num_dec_layers: int = 3,
        dim_ff: int = 512,
        dropout: float = 0.1,
        # Physics loss weights — giữ như v2
        w_mse: float = 0.05,
        lambda_speed: float = 0.1,
        lambda_accel: float = 0.01,
        v_max_kmh: float = 80.0,
        dt_h: float = 6.0,
        # Hard extras
        recurv_threshold: float = 45.0,
        gate_hidden: int = 64,
        recurv_hidden: int = 128,
        # Step-weighted DPE ramp
        dpe_ramp: float = 2.0,
        # Easy/hard thresholds
        threshold_curv: float = 15.0,
        threshold_spd: float = 0.5,
    ):
        super().__init__()
        self.obs_len          = obs_len
        self.pred_len         = pred_len
        self.d_model          = d_model
        self.w_mse            = w_mse
        self.lambda_speed     = lambda_speed
        self.lambda_accel     = lambda_accel
        self.v_max_norm       = v_max_kmh * dt_h / (111.0 * _DEG_SCALE)
        self.recurv_threshold = recurv_threshold
        self.threshold_curv   = threshold_curv
        self.threshold_spd    = threshold_spd
        self.dpe_ramp         = dpe_ramp

        # ── Shared Encoder (backbone giữ nguyên) ─────────────────────────
        self.encoder  = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)
        # Project RAW_CTX_DIM (512) → d_model (128)
        self.ctx_proj = nn.Sequential(
            nn.Linear(PaperEncoder.RAW_CTX_DIM, d_model),
            nn.LayerNorm(d_model),
        )

        # ── CHIẾN LƯỢC 2: ObsKinematicEncoder với d_model=128 ────────────
        self.obs_enc = ObsKinematicEncoder(
            d_model=d_model, nhead=nhead,
            num_layers=num_enc_layers, dim_ff=dim_ff, dropout=dropout,
        )

        # ── CHIẾN LƯỢC 1: Steering Context Encoder ───────────────────────
        self.steer_enc = SteeringContextEncoder(d_model=d_model)

        # ── CHIẾN LƯỢC 3: Horizon Queries + Learned Position Bias ───────
        # RoPE encode RELATIVE position trong attention (không cần additive PE)
        # Nhưng vẫn cần initial diversity trong queries để cross-attn có diff signal
        #
        # BUG 3 FIX: std=0.01 quá nhỏ → queries ≈ 0 → cross-attn output đồng đều
        # → gradients nhỏ → slow convergence đặc biệt với d_model=128
        # Giải pháp kép:
        #   1. std=0.02 (như v1 gốc) cho horizon_queries magnitude hợp lý
        #   2. Thêm learned_pos_bias [1, pred_len, d_model]: absolute position info
        #      được học riêng biệt với queries → không conflict với RoPE relative info
        #      (RoPE encode relative trong Q*K; bias encode absolute trong Q trước RoPE)
        self.horizon_queries   = nn.Parameter(
            torch.randn(1, pred_len, d_model) * 0.02
        )
        # Learned position bias: cung cấp absolute position identity cho mỗi step
        # Được cộng vào horizon_queries TRƯỚC khi đưa vào decoder
        # (RoPE sau đó chuyển hoá thành relative via Q*K rotation)
        self.horizon_pos_bias  = nn.Parameter(
            torch.zeros(1, pred_len, d_model)   # init=0: không ảnh hưởng ban đầu
        )

        # ── CHIẾN LƯỢC 3: RoPE Decoder thay Sinusoidal+TransformerDecoder ─
        self.transformer_dec = RoPETransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_dec_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            max_horizon=pred_len,
        )

        # ── Regression Head (d_model=128 → 2) ────────────────────────────
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, 2),
        )

        # ── Hard-only modules ─────────────────────────────────────────────
        self.steering_gate = PhysicsSteeringGate(
            ctx_dim=d_model,
            steer_feat_dim=SteeringContextEncoder.STEER_FEAT_DIM,
            hidden=gate_hidden,
            pred_len=pred_len,
        )
        self.recurv_head = RecurvatureTimingHead(
            ctx_dim=d_model, pred_len=pred_len, hidden=recurv_hidden,
        )

        # ── UW: giữ 3 tasks cho hard ─────────────────────────────────────
        self.uw = UncertaintyWeighting()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Gate init: bias=2.0 → alpha≈0.88 ban đầu (lean về learned decoder)
        nn.init.constant_(self.steering_gate.gate_net[-1].bias, 2.0)
        nn.init.zeros_(self.steering_gate.gate_net[-1].weight)
        # horizon_pos_bias: init với small sinusoidal để cung cấp diversity ban đầu
        # (giúp early convergence tốt hơn so với init=0)
        with torch.no_grad():
            for pos in range(self.pred_len):
                for i in range(0, self.d_model, 2):
                    div = math.exp(-i * math.log(100.0) / self.d_model)
                    self.horizon_pos_bias[0, pos, i]     = 0.1 * math.sin(pos * div)
                    if i + 1 < self.d_model:
                        self.horizon_pos_bias[0, pos, i+1] = 0.1 * math.cos(pos * div)

    def set_thresholds(self, curv: float, spd: float):
        self.threshold_curv = curv
        self.threshold_spd  = spd

    # ── Encode (1 lần cho toàn batch) ────────────────────────────────────

    def _get_data3d(self, batch_list) -> Optional[torch.Tensor]:
        """Trích xuất data3d từ batch_list nếu có."""
        if (len(batch_list) > 2
                and isinstance(batch_list[2], torch.Tensor)
                and batch_list[2].dim() == 4):
            return batch_list[2]
        return None

    def _encode(self, batch_list) -> Tuple[torch.Tensor, torch.Tensor,
                                           torch.Tensor, Optional[torch.Tensor]]:
        """
        Full encode với ERA5 steering token (CHIẾN LƯỢC 1).

        Returns:
            lp:     [B, pred_len, 2]  — decoder output
            ctx:    [B, d_model]      — pooled context
            fmem:   [B, T_obs+2, d_model] — full memory (ctx_tok + steer_tok + obs)
            steer:  [B, 2] hoặc None — steering vector để reuse trong gate
        """
        obs    = batch_list[0]     # [T_obs, B, 2]
        B      = obs.shape[1]
        device = obs.device
        dtype  = obs.dtype

        # 1. Backbone encoder → ctx_token
        raw   = self.encoder(batch_list)         # [B, 512]
        ctok  = self.ctx_proj(raw).unsqueeze(1)  # [B, 1, d_model]

        # 2. ERA5 steering → steer_token (CHIẾN LƯỢC 1)
        data3d = self._get_data3d(batch_list)
        steer  = SteeringContextEncoder.extract_steer(data3d, obs)  # [B, 2] or None
        stok   = self.steer_enc(steer, B, device, dtype)            # [B, 1, d_model]

        # 3. Obs kinematic encoder
        omem  = self.obs_enc(obs)                # [B, T_obs, d_model]

        # 4. Full memory: [ctx_token | steer_token | obs_memory]
        #    Shape: [B, 1+1+T_obs, d_model] = [B, T_obs+2, d_model]
        fmem  = torch.cat([ctok, stok, omem], dim=1)

        # 5. Pooled ctx từ full memory (bao gồm steer info)
        ctx   = fmem.mean(dim=1)                 # [B, d_model]

        # 6. CHIẾN LƯỢC 3: Horizon queries + Learned Position Bias
        #    horizon_queries: content queries (WHAT to predict)
        #    horizon_pos_bias: absolute position identity (WHICH step)
        #    RoPE trong decoder: relative position (HOW FAR apart)
        #    Ba nguồn info bổ sung nhau, không conflict
        Q = (self.horizon_queries + self.horizon_pos_bias).expand(B, -1, -1)  # [B, pred_len, d_model]

        # 7. RoPE Decoder (CHIẾN LƯỢC 3)
        D   = self.transformer_dec(Q, fmem)      # [B, pred_len, d_model]
        lp  = self.reg_head(D)                   # [B, pred_len, 2]

        return lp, ctx, fmem, steer

    # ── Gate Blend (nâng cấp: steer_feat 5D) ─────────────────────────────

    def _gate_blend(self,
                    lp_h: torch.Tensor,              # [B_h, H, 2]
                    ctx_h: torch.Tensor,             # [B_h, d_model]
                    steer_h: Optional[torch.Tensor], # [B_h, 2] or None
                    obs_h: torch.Tensor,             # [T_obs, B_h, 2]
                    stop_ctx_grad: bool = True,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Blend learned prediction với physics persistence.
        Nâng cấp từ v2: steer input là 5D features (magnitude + direction)
        để gate có context phong phú hơn khi quyết định α.

        Physics fallback: nếu steer = None → dùng last velocity
        """
        B_h = lp_h.shape[0]
        T   = self.pred_len
        dev = lp_h.device
        dtype = lp_h.dtype

        # Steer fallback: last observed velocity
        if steer_h is None:
            steer_h = (obs_h[-1] - obs_h[-2]
                       if obs_h.shape[0] >= 2
                       else torch.zeros(B_h, 2, device=dev, dtype=dtype))

        # Physics trajectory: constant velocity extrapolation
        steps   = torch.arange(1, T + 1, device=dev, dtype=dtype)
        offsets = steer_h.unsqueeze(1) * steps.view(1, T, 1)   # [B_h, T, 2]
        physics = obs_h[-1].unsqueeze(1) + offsets              # [B_h, T, 2]

        # Gate: dùng 5D steer features thay vì 2D raw
        steer_feat = SteeringContextEncoder.steer_to_features(
            steer_h, B_h, dev, dtype)  # [B_h, 5]

        ctx_in  = ctx_h.detach() if stop_ctx_grad else ctx_h
        t_idx   = torch.arange(T, device=dev)
        s_emb   = self.steering_gate.step_emb(t_idx)        # [T, 16]

        ctx_e    = ctx_in.unsqueeze(0).expand(T, -1, -1)    # [T, B_h, d]
        steer_e  = steer_feat.unsqueeze(0).expand(T, -1, -1)# [T, B_h, 5]
        s_emb_e  = s_emb.unsqueeze(1).expand(T, B_h, -1)   # [T, B_h, 16]

        gate_in  = torch.cat([ctx_e, s_emb_e, steer_e], -1) # [T, B_h, d+21]
        alpha    = torch.sigmoid(
            self.steering_gate.gate_net(gate_in))            # [T, B_h, 1]

        alpha_t  = alpha.permute(1, 0, 2)                   # [B_h, T, 1]
        final_h  = alpha_t * lp_h + (1 - alpha_t) * physics # [B_h, T, 2]

        return final_h, alpha.mean()

    # ── Easy losses (BONUS: step-weighted DPE) ───────────────────────────

    def _easy_losses(self, pred: torch.Tensor,
                     gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        pred, gt: [T, B_e, 2]  (normalised)
        BONUS FIX: step-weighted DPE + step-weighted MSE (BUG 4 fix).

        BUG 4: dùng step_weighted_dpe cho DPE nhưng MSE vẫn uniform
        → contradicting gradient signal: DPE push model để cải thiện step 12,
          nhưng MSE pull ngược lại vì uniform → model không converge tốt
        Fix: MSE cũng dùng cùng step weights để consistent.
        """
        T    = min(pred.shape[0], gt.shape[0])
        p, g = pred[:T], gt[:T]

        # Step-weighted DPE
        l_dpe = step_weighted_dpe(_norm_to_deg(p), _norm_to_deg(g),
                                  ramp=self.dpe_ramp)

        # Step-weighted MSE: cùng weights với DPE → consistent gradient direction
        raw_w = torch.linspace(1.0, 1.0 + self.dpe_ramp, T,
                               device=p.device, dtype=p.dtype)
        w = F.softmax(raw_w, dim=0).view(T, 1, 1)   # [T, 1, 1]
        l_mse = (w * (p - g).pow(2)).sum(0).mean()   # weighted MSE

        l_speed = p.new_zeros(())
        if T >= 2:
            sd      = (p[1:] - p[:-1]).norm(-1)
            l_speed = F.relu(sd - self.v_max_norm).pow(2).mean()

        l_accel = p.new_zeros(())
        if T >= 3:
            v       = p[1:] - p[:-1]
            l_accel = (v[1:].norm(-1) - v[:-1].norm(-1)).pow(2).mean()

        return {"dpe": l_dpe, "mse": l_mse,
                "speed": l_speed, "accel": l_accel}

    # ── Hard losses ───────────────────────────────────────────────────────

    def _hard_losses(self,
                     final_h: torch.Tensor,  # [B_h, H, 2]
                     gt_h: torch.Tensor,     # [T, B_h, 2]
                     ctx_h: torch.Tensor,    # [B_h, d] gốc
                     obs_h: torch.Tensor,    # [T_obs, B_h, 2]
                     ) -> Dict[str, torch.Tensor]:
        T   = min(final_h.shape[1], gt_h.shape[0])
        fp  = final_h[:, :T]                       # [B_h, T, 2]
        g   = gt_h[:T].permute(1, 0, 2)            # [B_h, T, 2]
        fpt = fp.permute(1, 0, 2)                  # [T, B_h, 2]
        gt  = g.permute(1, 0, 2)                   # [T, B_h, 2]

        # Step-weighted DPE cho hard cũng
        l_dpe = step_weighted_dpe(_norm_to_deg(fpt), _norm_to_deg(gt),
                                  ramp=self.dpe_ramp)

        # Heading loss
        l_heading = fp.new_zeros(())
        if T >= 2:
            pd    = fpt[1:] - fpt[:-1]
            gd    = gt[1:]  - gt[:-1]
            pn    = F.normalize(pd, dim=-1, eps=1e-8)
            gn    = F.normalize(gd, dim=-1, eps=1e-8)
            l_heading = ((1.0 - (pn * gn).sum(-1).clamp(-1, 1)) / 2.0).mean()

        # Recurvature auxiliary loss
        l_recurv   = fp.new_zeros(())
        recurv_log = self.recurv_head(ctx_h)
        try:
            label = RecurvatureTimingHead.make_label(
                gt_h, obs_h, self.recurv_threshold)
            if label.shape[0] == recurv_log.shape[0]:
                l_recurv = F.cross_entropy(recurv_log, label)
        except Exception:
            pass

        return {"dpe": l_dpe, "heading": l_heading, "recurv": l_recurv}

    # ── Training loss ─────────────────────────────────────────────────────

    def get_loss_breakdown(self, batch_list) -> Dict:
        obs_traj = batch_list[0]   # [T_obs, B, 2]
        traj_gt  = batch_list[1]   # [T_pred, B, 2]

        # Classify easy / hard
        is_hard = classify_hard_obs(obs_traj, self.threshold_curv, self.threshold_spd)
        is_easy = ~is_hard
        n_easy  = int(is_easy.sum())
        n_hard  = int(is_hard.sum())

        # 1 forward pass cho toàn batch
        lp, ctx, fmem, steer = self._encode(batch_list)   # lp [B, H, 2]

        result = {"n_easy": n_easy, "n_hard": n_hard}

        # ── EASY: step-weighted DPE (BONUS FIX) ──────────────────────────
        # Gradient đầy đủ → decoder + encoder + steer_enc
        L_easy = None
        if n_easy > 0:
            lp_e = lp[is_easy].permute(1, 0, 2)   # [T, B_e, 2]
            gt_e = traj_gt[:, is_easy]             # [T, B_e, 2]
            el   = self._easy_losses(lp_e, gt_e)

            L_easy = (
                el["dpe"]
                + self.w_mse        * el["mse"]
                + self.lambda_speed * el["speed"]
                + self.lambda_accel * el["accel"]
            )
            result.update({
                "easy_dpe": el["dpe"].item(),
                "easy_mse": el["mse"].item(),
            })

        # ── HARD: gate-corrected ─────────────────────────────────────────
        # lp_h.detach() → decoder KHÔNG nhận gradient từ L_hard
        # stop_ctx_grad=True → encoder KHÔNG nhận gradient từ gate
        L_hard = None
        if n_hard > 0:
            lp_h  = lp[is_hard]           # [B_h, H, 2]
            gt_h  = traj_gt[:, is_hard]   # [T, B_h, 2]
            obs_h = obs_traj[:, is_hard]  # [T_obs, B_h, 2]
            ctx_h = ctx[is_hard]          # [B_h, d]

            # Steer subset cho hard
            steer_h = steer[is_hard] if steer is not None else None

            final_h, alpha_m = self._gate_blend(
                lp_h.detach(), ctx_h, steer_h, obs_h,
                stop_ctx_grad=True)

            hl = self._hard_losses(final_h, gt_h, ctx_h, obs_h)

            L_hard = (
                self.uw.weight("dpe_hard", hl["dpe"])
                + self.uw.weight("heading", hl["heading"])
                + self.uw.weight("recurv",  hl["recurv"])
            )
            result.update({
                "hard_dpe":    hl["dpe"].item(),
                "hard_heading": (hl["heading"].item()
                                 if isinstance(hl["heading"], torch.Tensor)
                                 else 0.0),
                "alpha_mean":  (alpha_m.item()
                                if isinstance(alpha_m, torch.Tensor)
                                else 0.0),
            })

        # Combine
        if L_easy is not None and L_hard is not None:
            total = L_easy + L_hard
        elif L_easy is not None:
            total = L_easy
        else:
            total = L_hard

        result["total"] = total
        result.update(self.uw.sigma_dict())
        return result

    def get_loss(self, batch_list) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list)["total"]

    # ── Inference ─────────────────────────────────────────────────────────

    def forward(self, batch_list) -> torch.Tensor:
        """
        Easy  → decoder output với ERA5 context (CHIẾN LƯỢC 1 benefit)
        Hard  → gate-blended với steer 5D
        Returns: [T, B, 2]
        """
        lp, ctx, fmem, steer = self._encode(batch_list)   # [B, H, 2]
        obs    = batch_list[0]                             # [T_obs, B, 2]

        pred = lp.permute(1, 0, 2).clone()                # [T, B, 2]

        is_hard = classify_hard_obs(obs, self.threshold_curv, self.threshold_spd)

        if is_hard.any():
            obs_h   = obs[:, is_hard]
            ctx_h   = ctx[is_hard]
            lp_h    = lp[is_hard]                         # [B_h, H, 2]
            steer_h = steer[is_hard] if steer is not None else None

            final_h, _ = self._gate_blend(
                lp_h, ctx_h, steer_h, obs_h,
                stop_ctx_grad=False)

            pred[:, is_hard] = final_h.permute(1, 0, 2)

        return pred   # [T, B, 2]

    @torch.no_grad()
    def sample(self, batch_list, **kwargs):
        pred    = self.forward(batch_list)
        T, B, _ = pred.shape
        return pred, torch.zeros(T, B, 2, device=pred.device), pred.unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_st_trans_v3(args, threshold_curv: float = 15.0,
                      threshold_spd: float = 0.5) -> STTransV3:
    return STTransV3(
        obs_len          = getattr(args, "obs_len",          8),
        pred_len         = getattr(args, "pred_len",         12),
        unet_in_ch       = getattr(args, "unet_in_ch",       13),
        d_model          = getattr(args, "d_model",          128),    # v3 default
        nhead            = getattr(args, "nhead",            8),      # v3 default
        num_enc_layers   = getattr(args, "num_enc_layers",   2),      # v3 default
        num_dec_layers   = getattr(args, "num_dec_layers",   3),
        dim_ff           = getattr(args, "dim_ff",           512),
        dropout          = getattr(args, "dropout",          0.1),
        w_mse            = getattr(args, "w_mse",            0.05),
        lambda_speed     = getattr(args, "lambda_speed",     0.1),
        lambda_accel     = getattr(args, "lambda_accel",     0.01),
        v_max_kmh        = getattr(args, "v_max_kmh",        80.0),
        recurv_threshold = getattr(args, "recurv_threshold", 45.0),
        gate_hidden      = getattr(args, "gate_hidden",      64),
        recurv_hidden    = getattr(args, "recurv_hidden",    128),
        dpe_ramp         = getattr(args, "dpe_ramp",         2.0),
        threshold_curv   = threshold_curv,
        threshold_spd    = threshold_spd,
    )
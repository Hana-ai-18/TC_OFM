"""
Model/st_trans_model.py  ── ST-Trans Baseline
===============================================
THUẬT TOÁN: Faiaz et al. (2026)
"Physics-guided non-autoregressive transformer for lightweight
cyclone track prediction in the Bay of Bengal"
Expert Systems With Applications 317 (2026) 131972

CHIẾN LƯỢC (theo paper §3):
  - Input : obs_traj cuối (obs_len bước) → encode 8D features
  - CNN   : reshape → [S, 1, 2, 4] grid → 2×2 conv → 1×2 conv → 64D
  - Encoder: Transformer encoder (self-attention over S steps)
  - Decoder: Non-autoregressive, learned horizon queries [H×dmodel]
  - Output : toàn bộ H bước predict song song (không autoregressive)

LOSS (§3.5.1 - Physics-guided composite):
  L = L_DPE + 0.05*L_MSE + λ_speed*L_speed + λ_accel*L_accel
  λ_speed = 0.1, λ_accel = 0.01, v_max = 80 km/h

ADAPTED cho TCND_vn:
  - Input features: lat, lon, speed, heading, year, month, day, hour
    (giống 8D của paper, extract từ obs_traj + metadata)
  - Dùng chung DataLoader của bạn
  - Output: [pred_len, B, 2] normalised (cùng format với bài của bạn)
  - Eval bằng cùng haversine ADE metrics (12h/24h/48h/72h)

NOTE:
  - Paper dùng S=3 (9h), H=16 (48h) ở resolution 3h
  - Bạn dùng S=8 (obs_len), H=12 (pred_len) ở resolution 6h → 72h
  - Điều chỉnh: S=obs_len, H=pred_len, giữ nguyên architecture
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
#  Haversine helpers  (cùng convention với bài của bạn)
# ══════════════════════════════════════════════════════════════════════════════

def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
    """Denormalise từ normalised space → degrees."""
    out = arr.clone()
    out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0   # lon
    out[..., 1] = (arr[..., 1] * 50.0) / 10.0              # lat
    return out


def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Haversine distance [km]. p1, p2: [..., 2] degrees (lon, lat)."""
    lon1 = torch.deg2rad(p1[..., 0]); lat1 = torch.deg2rad(p1[..., 1])
    lon2 = torch.deg2rad(p2[..., 0]); lat2 = torch.deg2rad(p2[..., 1])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = torch.sin(dlat/2).pow(2) + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2)
    return 2.0 * 6371.0 * torch.asin(a.clamp(1e-12, 1.0).sqrt())


HORIZON_STEPS: Dict[int, int] = {12: 1, 24: 3, 48: 7, 72: 11}


# ══════════════════════════════════════════════════════════════════════════════
#  Feature extractor  ── lấy 8D features từ obs_traj
# ══════════════════════════════════════════════════════════════════════════════

class ObsTrajFeatureExtractor(nn.Module):
    """
    Extract 8D features từ obs_traj [T_obs, B, 2] (normalised lat/lon).

    Paper features:
      1. lat_norm, lon_norm     (position)
      2. speed_norm             (translation speed estimate)
      3. heading_norm           (motion direction)
      4. year_norm, month_norm, day_norm, hour_norm  (temporal)

    Vì bạn không có temporal metadata trong batch, ta dùng:
      1-2: lat, lon (từ obs_traj)
      3-4: delta_lat, delta_lon (velocity)
      5-6: delta2_lat, delta2_lon (acceleration)
      7-8: step index (normalized), cumulative distance

    → 8D vẫn đủ để encode kinematic history
    """

    def __init__(self, feat_dim: int = 8):
        super().__init__()
        self.feat_dim = feat_dim

    def forward(self, obs_traj: torch.Tensor) -> torch.Tensor:
        """
        obs_traj: [T_obs, B, 2] normalised (lon, lat)
        Returns:  [B, T_obs, 8]
        """
        T, B, _ = obs_traj.shape
        device   = obs_traj.device

        # Position
        lon = obs_traj[:, :, 0]   # [T, B]
        lat = obs_traj[:, :, 1]   # [T, B]

        # Velocity (delta per step)
        if T >= 2:
            d_lon = torch.cat([obs_traj[1:, :, 0] - obs_traj[:-1, :, 0],
                               torch.zeros(1, B, device=device)], dim=0)
            d_lat = torch.cat([obs_traj[1:, :, 1] - obs_traj[:-1, :, 1],
                               torch.zeros(1, B, device=device)], dim=0)
        else:
            d_lon = torch.zeros(T, B, device=device)
            d_lat = torch.zeros(T, B, device=device)

        # Acceleration (delta of delta)
        if T >= 3:
            dd_lon = torch.cat([d_lon[1:] - d_lon[:-1],
                                torch.zeros(1, B, device=device)], dim=0)
            dd_lat = torch.cat([d_lat[1:] - d_lat[:-1],
                                torch.zeros(1, B, device=device)], dim=0)
        else:
            dd_lon = torch.zeros(T, B, device=device)
            dd_lat = torch.zeros(T, B, device=device)

        # Step index (0→1 normalized)
        step_idx = torch.linspace(0, 1, T, device=device).unsqueeze(1).expand(T, B)

        # Speed magnitude
        speed = torch.sqrt(d_lon.pow(2) + d_lat.pow(2)).clamp(min=0)

        # Stack: [T, B, 8]
        features = torch.stack([lon, lat, d_lon, d_lat,
                                 dd_lon, dd_lat, step_idx, speed], dim=-1)
        return features.permute(1, 0, 2)   # [B, T_obs, 8]


# ══════════════════════════════════════════════════════════════════════════════
#  CNN State Encoder  ── §3.3.2
# ══════════════════════════════════════════════════════════════════════════════

class CNNStateEncoder(nn.Module):
    """
    Per-timestep CNN encoder.
    Input: 8D feature vector → reshape [1, 2, 4] → conv → 64D.

    Paper §3.3.2:
      - Reshape 8D → [1, 2, 4] synthetic grid
      - Conv1: 2×2 kernel, 1→32 channels, ReLU
      - Conv2: 1×2 kernel, 32→32 channels, ReLU
      - Flatten → 64D
    """

    def __init__(self, feat_dim: int = 8, out_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.out_dim  = out_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, 2), stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=1, padding=0)

        # After conv1: [B, 32, 1, 3], after conv2: [B, 32, 1, 2] → flatten=64
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, 8]
        Returns: [B, S, out_dim]
        """
        B, S, _ = x.shape

        # Process each timestep independently
        x_flat = x.reshape(B * S, 1, 2, 4)   # [B*S, 1, 2, 4]

        h = F.relu(self.conv1(x_flat))         # [B*S, 32, 1, 3]
        h = F.relu(self.conv2(h))              # [B*S, 32, 1, 2]
        h = h.flatten(1)                       # [B*S, 64]
        h = self.proj(h)                       # [B*S, out_dim]

        return h.reshape(B, S, self.out_dim)


# ══════════════════════════════════════════════════════════════════════════════
#  Sinusoidal Positional Encoding
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ══════════════════════════════════════════════════════════════════════════════
#  ST-Trans Main Model  ── §3.3.1
# ══════════════════════════════════════════════════════════════════════════════

class STTrans(nn.Module):
    """
    Physics-guided Non-Autoregressive Transformer for TC track prediction.

    Architecture (paper §3.3):
      1. Feature extraction: obs_traj → 8D features
      2. CNN state encoder: 8D → 64D per timestep
      3. Linear projection: 64D → dmodel
      4. Transformer encoder: self-attention over S history steps
      5. Horizon queries: H learned queries [H, dmodel]
      6. Transformer decoder: cross-attention → [H, dmodel]
      7. Regression head: [H, dmodel] → [H, 2]

    Physics loss (§3.5.1):
      L = L_DPE + 0.05*L_MSE + 0.1*L_speed + 0.01*L_accel
    """

    def __init__(
        self,
        obs_len:    int   = 8,
        pred_len:   int   = 12,
        feat_dim:   int   = 8,
        d_model:    int   = 64,
        nhead:      int   = 4,
        num_enc_layers: int = 1,
        num_dec_layers: int = 3,
        dim_ff:     int   = 512,
        dropout:    float = 0.1,
        # Physics loss weights (paper §3.5.1)
        lambda_speed: float = 0.1,
        lambda_accel: float = 0.01,
        w_mse:        float = 0.05,
        v_max_kmh:    float = 80.0,   # paper default
        dt_h:         float = 6.0,    # 6h per step (your dataset)
    ):
        super().__init__()
        self.obs_len    = obs_len
        self.pred_len   = pred_len
        self.d_model    = d_model
        self.lambda_speed = lambda_speed
        self.lambda_accel = lambda_accel
        self.w_mse        = w_mse
        # v_max in normalised units: 80 km/h * 6h = 480 km ≈ 480/111 ≈ 4.3 deg
        # normalised: 4.3 / 5.0 ≈ 0.86 per step
        self.v_max_norm = v_max_kmh * dt_h / (111.0 * 50.0)

        # ── Feature extraction ────────────────────────────────────────────
        self.feat_extractor = ObsTrajFeatureExtractor(feat_dim)
        self.cnn_enc        = CNNStateEncoder(feat_dim, d_model)
        self.input_proj     = nn.Linear(d_model, d_model)
        self.enc_pe         = SinusoidalPE(d_model, max_len=obs_len + 10)

        # ── Transformer encoder ───────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            activation="relu", batch_first=True,
        )
        self.transformer_enc = nn.TransformerEncoder(enc_layer,
                                                      num_layers=num_enc_layers)

        # ── Horizon queries (learned, paper §3.3.4) ───────────────────────
        self.horizon_queries = nn.Parameter(
            torch.randn(1, pred_len, d_model) * 0.02
        )
        self.dec_pe = SinusoidalPE(d_model, max_len=pred_len + 10)

        # ── Transformer decoder ───────────────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            activation="relu", batch_first=True,
        )
        self.transformer_dec = nn.TransformerDecoder(dec_layer,
                                                      num_layers=num_dec_layers)

        # ── Regression head (paper §3.3.4: g(dτ) = W2·σ(W1·dτ+b1)+b2) ──
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, obs_traj: torch.Tensor) -> torch.Tensor:
        """
        obs_traj: [T_obs, B, 2] normalised
        Returns:  [pred_len, B, 2] normalised predictions
        """
        B = obs_traj.shape[1]

        # 1. Extract 8D features
        feat = self.feat_extractor(obs_traj)   # [B, S, 8]

        # 2. CNN state encoder
        h = self.cnn_enc(feat)                  # [B, S, d_model]
        h = self.input_proj(h)                  # [B, S, d_model]
        h = self.enc_pe(h)                      # [B, S, d_model]

        # 3. Transformer encoder → memory
        memory = self.transformer_enc(h)        # [B, S, d_model]

        # 4. Horizon queries with positional encoding
        Q = self.horizon_queries.expand(B, -1, -1)   # [B, H, d_model]
        Q = self.dec_pe(Q)                            # [B, H, d_model]

        # 5. Transformer decoder (non-autoregressive cross-attention)
        D = self.transformer_dec(Q, memory)     # [B, H, d_model]

        # 6. Regression head → [B, H, 2]
        out = self.reg_head(D)                  # [B, H, 2]

        # Return [pred_len, B, 2] (same format as your model)
        return out.permute(1, 0, 2)

    # ── Physics-guided loss (§3.5.1) ─────────────────────────────────────

    def physics_loss(
        self,
        pred_norm: torch.Tensor,   # [T, B, 2] normalised
        gt_norm:   torch.Tensor,   # [T, B, 2] normalised
    ) -> Dict:
        T = min(pred_norm.shape[0], gt_norm.shape[0])
        pred = pred_norm[:T]
        gt   = gt_norm[:T]

        # Convert to degrees for haversine
        pred_deg = _norm_to_deg(pred)
        gt_deg   = _norm_to_deg(gt)

        # L_DPE: mean great-circle distance (eq. 16)
        dist = haversine_km(pred_deg, gt_deg)   # [T, B]
        l_dpe = dist.mean()

        # L_MSE: coordinate space MSE (eq. 17)
        l_mse = F.mse_loss(pred, gt)

        # L_speed: penalize speeds > v_max (eq. 18-20)
        if T >= 2:
            step_dist = torch.sqrt(
                (pred[1:, :, 0] - pred[:-1, :, 0]).pow(2) +
                (pred[1:, :, 1] - pred[:-1, :, 1]).pow(2)
            )   # [T-1, B] in normalised units
            excess_speed = F.relu(step_dist - self.v_max_norm)
            l_speed = excess_speed.pow(2).mean()
        else:
            l_speed = pred_norm.new_zeros(())

        # L_accel: penalize acceleration changes (eq. 21-22)
        if T >= 3:
            vel = pred[1:] - pred[:-1]        # [T-1, B, 2]
            spd = vel.norm(dim=-1)             # [T-1, B]
            accel = (spd[1:] - spd[:-1]).pow(2).mean()
            l_accel = accel
        else:
            l_accel = pred_norm.new_zeros(())

        # Total physics-guided loss (eq. 15)
        total = (l_dpe
                 + self.w_mse        * l_mse
                 + self.lambda_speed * l_speed
                 + self.lambda_accel * l_accel)

        return dict(
            total=total,
            dpe=l_dpe.item(),
            mse=l_mse.item(),
            speed=l_speed.item(),
            accel=l_accel.item(),
        )

    # ── Interface matching TCFlowMatching ─────────────────────────────────

    def get_loss(self, batch_list) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list)["total"]

    def get_loss_breakdown(self, batch_list) -> Dict:
        obs_traj = batch_list[0]   # [T_obs, B, 2]
        traj_gt  = batch_list[1]   # [T_gt, B, 2]

        pred = self.forward(obs_traj)
        return self.physics_loss(pred, traj_gt)

    @torch.no_grad()
    def sample(
        self,
        batch_list,
        num_ensemble: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_mean : [T, B, 2] normalised
            me_mean   : [T, B, 2] zeros
            all_trajs : [1, T, B, 2]
        """
        obs_traj = batch_list[0]
        pred = self.forward(obs_traj)
        T, B, _ = pred.shape
        me_mean   = torch.zeros(T, B, 2, device=pred.device)
        all_trajs = pred.unsqueeze(0)
        return pred, me_mean, all_trajs


# ══════════════════════════════════════════════════════════════════════════════
#  Autoregressive variant  ── ST-Trans-AR (paper §3.4)
# ══════════════════════════════════════════════════════════════════════════════

class STTransAR(nn.Module):
    """
    Autoregressive variant of ST-Trans (baseline trong paper §3.4).
    Cùng encoder architecture, nhưng decoder predict từng bước một.
    Dùng để so sánh với non-AR trong ablation study.
    """

    def __init__(
        self,
        obs_len:    int   = 8,
        pred_len:   int   = 12,
        feat_dim:   int   = 8,
        d_model:    int   = 64,
        nhead:      int   = 4,
        num_enc_layers: int = 1,
        dim_ff:     int   = 512,
        dropout:    float = 0.1,
        lambda_speed: float = 0.1,
        lambda_accel: float = 0.01,
        w_mse:        float = 0.05,
        v_max_kmh:    float = 80.0,
        dt_h:         float = 6.0,
    ):
        super().__init__()
        self.obs_len    = obs_len
        self.pred_len   = pred_len
        self.d_model    = d_model
        self.lambda_speed = lambda_speed
        self.lambda_accel = lambda_accel
        self.w_mse        = w_mse
        self.v_max_norm = v_max_kmh * dt_h / (111.0 * 50.0)

        self.feat_extractor = ObsTrajFeatureExtractor(feat_dim)
        self.cnn_enc        = CNNStateEncoder(feat_dim, d_model)
        self.input_proj     = nn.Linear(d_model, d_model)
        self.enc_pe         = SinusoidalPE(d_model, max_len=obs_len + 10)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            activation="relu", batch_first=True,
        )
        self.transformer_enc = nn.TransformerEncoder(enc_layer,
                                                      num_layers=num_enc_layers)

        # AR GRU decoder (simpler than full transformer decoder for AR)
        self.ar_gru   = nn.GRUCell(2 + d_model, d_model)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs_traj: torch.Tensor,
                gt_traj: Optional[torch.Tensor] = None,
                teacher_forcing: bool = True) -> torch.Tensor:
        B = obs_traj.shape[1]
        feat   = self.feat_extractor(obs_traj)
        h      = self.cnn_enc(feat)
        h      = self.input_proj(h)
        h      = self.enc_pe(h)
        memory = self.transformer_enc(h)            # [B, S, d_model]
        ctx    = memory.mean(dim=1)                  # [B, d_model]

        cur_pos = obs_traj[-1].clone()
        hx      = ctx
        preds   = []

        for i in range(self.pred_len):
            inp  = torch.cat([cur_pos, ctx], dim=-1)
            hx   = self.ar_gru(inp, hx)
            out  = self.reg_head(hx)
            preds.append(out)

            if teacher_forcing and gt_traj is not None and i < gt_traj.shape[0]:
                cur_pos = gt_traj[i]
            else:
                cur_pos = out.detach()

        return torch.stack(preds, dim=0)   # [pred_len, B, 2]

    def physics_loss(self, pred_norm, gt_norm):
        T = min(pred_norm.shape[0], gt_norm.shape[0])
        pred = pred_norm[:T]; gt = gt_norm[:T]
        pred_deg = _norm_to_deg(pred); gt_deg = _norm_to_deg(gt)
        l_dpe   = haversine_km(pred_deg, gt_deg).mean()
        l_mse   = F.mse_loss(pred, gt)
        if T >= 2:
            step_dist = (pred[1:] - pred[:-1]).norm(dim=-1)
            l_speed = F.relu(step_dist - self.v_max_norm).pow(2).mean()
        else:
            l_speed = pred_norm.new_zeros(())
        if T >= 3:
            vel = pred[1:] - pred[:-1]
            l_accel = (vel[1:].norm(dim=-1) - vel[:-1].norm(dim=-1)).pow(2).mean()
        else:
            l_accel = pred_norm.new_zeros(())
        total = l_dpe + self.w_mse*l_mse + self.lambda_speed*l_speed + self.lambda_accel*l_accel
        return dict(total=total, dpe=l_dpe.item(), mse=l_mse.item(),
                    speed=l_speed.item(), accel=l_accel.item())

    def get_loss(self, batch_list):
        return self.get_loss_breakdown(batch_list)["total"]

    def get_loss_breakdown(self, batch_list):
        obs_traj = batch_list[0]; traj_gt = batch_list[1]
        pred = self.forward(obs_traj, traj_gt, teacher_forcing=True)
        return self.physics_loss(pred, traj_gt)

    @torch.no_grad()
    def sample(self, batch_list, **kwargs):
        obs_traj = batch_list[0]
        pred = self.forward(obs_traj, teacher_forcing=False)
        T, B, _ = pred.shape
        me_mean = torch.zeros(T, B, 2, device=pred.device)
        return pred, me_mean, pred.unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
#  ADE metrics  (cùng format với bài của bạn)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ade_per_horizon(
    pred_norm: torch.Tensor,
    gt_norm:   torch.Tensor,
) -> Dict[str, float]:
    T = min(pred_norm.shape[0], gt_norm.shape[0])
    pred_deg = _norm_to_deg(pred_norm[:T])
    gt_deg   = _norm_to_deg(gt_norm[:T])
    dist = haversine_km(pred_deg, gt_deg)
    result = dict(ADE=float(dist.mean()), FDE=float(dist[-1].mean()))
    for h, s in HORIZON_STEPS.items():
        result[f"{h}h"] = float(dist[s].mean()) if s < T else float("nan")
    return result
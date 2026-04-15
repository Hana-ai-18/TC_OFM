"""
Model/paper_baseline_model.py  ── PAPER BASELINE
==================================================
THUẬT TOÁN GỐC: Rahman et al. (2025)
"Tropical cyclone track prediction harnessing deep learning algorithms"
Results in Engineering 26, 105009

CHIẾN LƯỢC:
  - Giữ nguyên encoder pipeline của bạn (FNO3D + Mamba + Env_net → raw_ctx [B, 512])
  - Thay toàn bộ prediction head bằng paper algorithm:
      * LSTM_Model  : LSTMCell stepwise (Algorithm 3)
      * GRU_Model   : GRUCell  stepwise với dropout=0.2 (Algorithm 4)
      * RNN_Model   : RNNCell  stepwise (Algorithm 2)
  - Loss: MSE (paper §2.7: "Mean Squared Error loss function")
  - Optimizer: Adam lr=0.001 (paper Table 2)
  - Epochs: 7000 (paper Table 2) → dùng early stopping trong train script
  - LR Scheduler: ReduceLROnPlateau (paper §2.8)

GIỮ NGUYÊN:
  - FNO3DEncoder, DataEncoder1D_Mamba, Env_net
  - _context() và toàn bộ encoder pipeline
  - haversine ADE metric qua các mốc thời gian

PAPER REFERENCE:
  Algorithm 2: Forward Pass in RNN Model
  Algorithm 3: LSTM_Model (LSTMCell)
  Algorithm 4: Forward Pass in GRU Model (GRUCell + dropout)
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


# ══════════════════════════════════════════════════════════════════════════════
#  Haversine ADE helpers  ── giữ lại để so sánh vs bài của bạn
# ══════════════════════════════════════════════════════════════════════════════

def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
    """Denormalise từ [0,1]-ish space về degrees."""
    out = arr.clone()
    out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0   # lon
    out[..., 1] = (arr[..., 1] * 50.0) / 10.0              # lat
    return out


def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    Haversine distance [km] giữa p1, p2 dạng degrees.
    p1, p2: [..., 2] (lon, lat)
    """
    lon1 = torch.deg2rad(p1[..., 0]);  lat1 = torch.deg2rad(p1[..., 1])
    lon2 = torch.deg2rad(p2[..., 0]);  lat2 = torch.deg2rad(p2[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2)
    return 2.0 * 6371.0 * torch.asin(a.clamp(1e-12, 1.0).sqrt())


# Horizon steps (6h per step): 12h=step1, 24h=step3, 48h=step7, 72h=step11
HORIZON_STEPS: Dict[int, int] = {12: 1, 24: 3, 48: 7, 72: 11}


def compute_ade_per_horizon(
    pred_norm: torch.Tensor,   # [T, B, 2] normalised
    gt_norm:   torch.Tensor,   # [T, B, 2] normalised
) -> Dict[str, float]:
    """
    Tính ADE (km) tại các mốc 12h, 24h, 48h, 72h + overall ADE/FDE.
    Dùng để so sánh trực tiếp với bài của bạn.
    """
    T = min(pred_norm.shape[0], gt_norm.shape[0])
    pred_deg = _norm_to_deg(pred_norm[:T])
    gt_deg   = _norm_to_deg(gt_norm[:T])

    dist = haversine_km(pred_deg, gt_deg)   # [T, B]

    result: Dict[str, float] = {}
    result["ADE"] = float(dist.mean().item())
    result["FDE"] = float(dist[-1].mean().item())

    for h_label, step_idx in HORIZON_STEPS.items():
        if step_idx < T:
            result[f"{h_label}h"] = float(dist[step_idx].mean().item())
        else:
            result[f"{h_label}h"] = float("nan")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Paper RNN_Model  ── Algorithm 2
# ══════════════════════════════════════════════════════════════════════════════

class PaperRNNHead(nn.Module):
    """
    RNN stepwise prediction (Algorithm 2 của paper).

    Paper §2.4:
      ht = σ(Wh*xt + Uh*ht-1 + bh)
      yt = fc(ht)   [regression thay softmax vì predict lat/lon]

    hidden_dim=128, layer_dim=3, feed_forward=True (paper default)
    """

    def __init__(
        self,
        input_dim:  int = 512 + 2,   # ctx concat cur_pos
        hidden_dim: int = 128,
        n_layers:   int = 3,
        pred_len:   int = 12,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.pred_len   = pred_len

        # RNNCell stack (paper Algorithm 2 line 9: ht = rnn(input_t, ht.detach()))
        self.cells = nn.ModuleList()
        self.cells.append(nn.RNNCell(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.cells.append(nn.RNNCell(hidden_dim, hidden_dim))

        # Readout (paper Algorithm 2 line 10: output = fc(ht))
        self.fc = nn.Linear(hidden_dim, 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        raw_ctx:  torch.Tensor,   # [B, 512]
        obs_traj: torch.Tensor,   # [T_obs, B, 2]
        feed_forward: bool = True,
    ) -> torch.Tensor:            # [pred_len, B, 2]
        B      = raw_ctx.shape[0]
        device = raw_ctx.device

        # Paper Algorithm 2 line 3: init hidden = zeros
        h_list = [torch.zeros(B, self.hidden_dim, device=device)
                  for _ in range(self.n_layers)]

        cur_pos = obs_traj[-1].clone()   # [B, 2]
        preds   = []

        for i in range(self.pred_len):
            # Paper Algorithm 2 line 5-7: input = [ctx, cur_pos]
            inp = torch.cat([raw_ctx, cur_pos], dim=-1)   # feed_forward

            # Layer stack
            h_list[0] = self.cells[0](inp, h_list[0].detach())
            h = h_list[0]
            for j in range(1, self.n_layers):
                h_list[j] = self.cells[j](h, h_list[j].detach())
                h = h_list[j]

            # Readout (paper Algorithm 2 line 10)
            output = self.fc(h)          # [B, 2] — absolute position
            preds.append(output)

            if feed_forward:             # paper Algorithm 2 line 6-8
                cur_pos = output.detach()

        return torch.stack(preds, dim=0)  # [pred_len, B, 2]


# ══════════════════════════════════════════════════════════════════════════════
#  Paper LSTM_Model  ── Algorithm 3
# ══════════════════════════════════════════════════════════════════════════════

class PaperLSTMHead(nn.Module):
    """
    LSTMCell stepwise prediction (Algorithm 3 của paper).

    Paper §2.5:
      i(t) = σ(W(i)x(t) + U(i)h(t-1) + b(i))
      f(t) = σ(W(f)x(t) + U(f)h(t-1) + b(f))
      c(t) = i(t) ⊙ tanh(W(c)x(t) + U(c)h(t-1) + b(c)) + f(t) ⊙ c(t-1)
      o(t) = σ(W(o)x(t) + U(o)h(t-1) + b(o))
      h(t) = o(t) ⊙ tanh(c(t))
      output = fc(h_t)

    Paper Table 2: hidden_dim=28, layer_dim=3
    (Chúng ta dùng hidden_dim lớn hơn vì input ctx=512 phức tạp hơn paper)
    """

    def __init__(
        self,
        input_dim:  int = 512 + 2,   # ctx concat cur_pos
        hidden_dim: int = 256,        # paper=28, ta dùng 256 vì ctx phức tạp hơn
        n_layers:   int = 3,
        pred_len:   int = 12,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.pred_len   = pred_len

        # LSTMCell stack (paper Algorithm 3 line 6: lstm = nn.LSTMCell)
        self.cells = nn.ModuleList()
        self.cells.append(nn.LSTMCell(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.cells.append(nn.LSTMCell(hidden_dim, hidden_dim))

        # Readout (paper Algorithm 3 line 7: fc = nn.Linear(hidden_dim, output_dim))
        self.fc = nn.Linear(hidden_dim, 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        raw_ctx:  torch.Tensor,   # [B, 512]
        obs_traj: torch.Tensor,   # [T_obs, B, 2]
        feed_forward: bool = True,
    ) -> torch.Tensor:            # [pred_len, B, 2]
        B      = raw_ctx.shape[0]
        device = raw_ctx.device

        # Paper Algorithm 3 line 11-12: h_t=zeros, c_t=zeros
        h_list = [torch.zeros(B, self.hidden_dim, device=device)
                  for _ in range(self.n_layers)]
        c_list = [torch.zeros(B, self.hidden_dim, device=device)
                  for _ in range(self.n_layers)]

        cur_pos = obs_traj[-1].clone()   # [B, 2]
        preds   = []

        for i in range(self.pred_len):
            # Paper Algorithm 3 line 14: input_t = [ctx, cur_pos]
            inp = torch.cat([raw_ctx, cur_pos], dim=-1)

            # Paper Algorithm 3 line 18: h_t, c_t = lstm(input_t, (h_t.detach(), c_t.detach()))
            h_list[0], c_list[0] = self.cells[0](
                inp, (h_list[0].detach(), c_list[0].detach()))
            h = h_list[0]
            for j in range(1, self.n_layers):
                h_list[j], c_list[j] = self.cells[j](
                    h, (h_list[j].detach(), c_list[j].detach()))
                h = h_list[j]

            # Paper Algorithm 3 line 19: output = fc(h_t)
            output = self.fc(h)          # [B, 2]
            preds.append(output)

            if feed_forward:             # paper Algorithm 3 line 15-17
                cur_pos = output.detach()

        return torch.stack(preds, dim=0)  # [pred_len, B, 2]


# ══════════════════════════════════════════════════════════════════════════════
#  Paper GRU_Model  ── Algorithm 4
# ══════════════════════════════════════════════════════════════════════════════

class PaperGRUHead(nn.Module):
    """
    GRUCell stepwise prediction với dropout (Algorithm 4 của paper).

    Paper §2.6 + §2.8:
      z_t = σ(Wz*xt + Uz*ht-1 + bz)
      r_t = σ(Wr*xt + Ur*ht-1 + br)
      h̃_t = tanh(Wh*xt + rt ⊙ Uh*ht-1 + bh)
      h_t = (1-z_t) ⊙ ht-1 + z_t ⊙ h̃_t
      ht  = dropout(ht)          [paper Algorithm 4 line 12]
      output = fc(ht)            [paper Algorithm 4 line 13]

    Paper Table 2: hidden_dim=64, layer_dim=3, dropout=0.2
    """

    def __init__(
        self,
        input_dim:  int   = 512 + 2,
        hidden_dim: int   = 256,       # paper=64, ta dùng 256
        n_layers:   int   = 3,
        pred_len:   int   = 12,
        dropout:    float = 0.20,      # paper §2.8: dropout rate 0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.pred_len   = pred_len

        # GRUCell stack (paper Algorithm 4 line 11)
        self.cells = nn.ModuleList()
        self.cells.append(nn.GRUCell(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.cells.append(nn.GRUCell(hidden_dim, hidden_dim))

        # Dropout (paper Algorithm 4 line 12: ht = dropout(ht))
        self.dropout = nn.Dropout(dropout)

        # Readout (paper Algorithm 4 line 13: output = fc(ht))
        self.fc = nn.Linear(hidden_dim, 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        raw_ctx:  torch.Tensor,   # [B, 512]
        obs_traj: torch.Tensor,   # [T_obs, B, 2]
        feed_forward: bool = True,
    ) -> torch.Tensor:            # [pred_len, B, 2]
        B      = raw_ctx.shape[0]
        device = raw_ctx.device

        # Paper Algorithm 4 line 5: init hidden = zeros
        h_list = [torch.zeros(B, self.hidden_dim, device=device)
                  for _ in range(self.n_layers)]

        cur_pos = obs_traj[-1].clone()   # [B, 2]
        preds   = []

        for i in range(self.pred_len):
            # Paper Algorithm 4 line 7-10: input = [ctx, cur_pos]
            inp = torch.cat([raw_ctx, cur_pos], dim=-1)

            # Paper Algorithm 4 line 11: ht = gru(input_t, ht.detach())
            h_list[0] = self.cells[0](inp, h_list[0].detach())
            # Paper Algorithm 4 line 12: ht = dropout(ht)
            h = self.dropout(h_list[0])
            for j in range(1, self.n_layers):
                h_list[j] = self.cells[j](h, h_list[j].detach())
                h = self.dropout(h_list[j])

            # Paper Algorithm 4 line 13: output = fc(ht)
            output = self.fc(h)          # [B, 2]
            preds.append(output)

            if feed_forward:             # paper Algorithm 4 line 8-9
                cur_pos = output.detach()

        return torch.stack(preds, dim=0)  # [pred_len, B, 2]


# ══════════════════════════════════════════════════════════════════════════════
#  Encoder Pipeline  ── GIỮ NGUYÊN từ bạn (VelocityField._context)
# ══════════════════════════════════════════════════════════════════════════════

class PaperEncoder(nn.Module):
    """
    Encoder pipeline GIỮ NGUYÊN từ flow_matching_model.py.
    FNO3D + Mamba + Env_net → raw_ctx [B, 512]
    """
    RAW_CTX_DIM = 512

    def __init__(
        self,
        obs_len:    int = 8,
        unet_in_ch: int = 13,
        ctx_dim:    int = 256,
    ):
        super().__init__()
        self.obs_len = obs_len

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

        self.enc_1d = DataEncoder1D(
            in_1d       = 4,
            feat_3d_dim = 128,
            mlp_h       = 64,
            lstm_hidden = 128,
            lstm_layers = 3,
            dropout     = 0.1,
            d_state     = 16,
        )

        self.env_enc = Env_net(obs_len=obs_len, d_model=32)

        self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
        self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
        self.ctx_drop = nn.Dropout(0.15)
        self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)

    def forward(self, batch_list) -> torch.Tensor:
        """Giữ nguyên _context() từ VelocityField → raw_ctx [B, 512]"""
        obs_traj  = batch_list[0]
        obs_Me    = batch_list[7]
        image_obs = batch_list[11]
        env_data  = batch_list[13]

        if image_obs.dim() == 4:
            image_obs = image_obs.unsqueeze(2)

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
                e_3d_s.permute(0, 2, 1), size=T_obs,
                mode="linear", align_corners=False,
            ).permute(0, 2, 1)

        e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
        t_weights  = torch.softmax(
            torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
                         device=e_3d_dec_t.device) * 0.5, dim=0,
        )
        f_spatial_scalar = (e_3d_dec_t * t_weights.unsqueeze(0)).sum(1, keepdim=True)
        f_spatial = self.decoder_proj(f_spatial_scalar)

        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        h_t    = self.enc_1d(obs_in, e_3d_s)

        e_env, _, _ = self.env_enc(env_data, image_obs)

        raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
        raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))
        return raw   # [B, 512]


# ══════════════════════════════════════════════════════════════════════════════
#  Paper Baseline Model  ── Wrapper chính
# ══════════════════════════════════════════════════════════════════════════════

MODEL_TYPES = ("lstm", "gru", "rnn")


class PaperBaseline(nn.Module):
    """
    Paper baseline model theo Rahman et al. (2025).

    Encoder: FNO3D + Mamba + Env_net (GIỮ NGUYÊN từ bạn)
    Prediction head: LSTMCell / GRUCell / RNNCell (paper algorithms)
    Loss: MSE (paper §2.7)
    """

    def __init__(
        self,
        model_type:  str = "lstm",    # "lstm" | "gru" | "rnn"
        pred_len:    int = 12,
        obs_len:     int = 8,
        hidden_dim:  int = 256,
        n_layers:    int = 3,
        unet_in_ch:  int = 13,
        dropout:     float = 0.20,    # cho GRU
    ):
        super().__init__()
        assert model_type in MODEL_TYPES, f"model_type phải là {MODEL_TYPES}"
        self.model_type = model_type
        self.pred_len   = pred_len
        self.obs_len    = obs_len

        # Encoder (giữ nguyên)
        self.encoder = PaperEncoder(obs_len=obs_len, unet_in_ch=unet_in_ch)

        # Input dim: ctx(512) + cur_pos(2)
        input_dim = PaperEncoder.RAW_CTX_DIM + 2

        # Prediction head theo paper
        if model_type == "lstm":
            self.head = PaperLSTMHead(
                input_dim=input_dim, hidden_dim=hidden_dim,
                n_layers=n_layers, pred_len=pred_len)
        elif model_type == "gru":
            self.head = PaperGRUHead(
                input_dim=input_dim, hidden_dim=hidden_dim,
                n_layers=n_layers, pred_len=pred_len, dropout=dropout)
        else:
            self.head = PaperRNNHead(
                input_dim=input_dim, hidden_dim=hidden_dim,
                n_layers=n_layers, pred_len=pred_len)

    # ── MSE Loss (paper §2.7) ─────────────────────────────────────────────────

    def mse_loss(
        self,
        pred_norm: torch.Tensor,   # [T, B, 2] normalised
        gt_norm:   torch.Tensor,   # [T, B, 2] normalised
    ) -> torch.Tensor:
        """
        Paper §2.7: MSE loss function.
        θ* = argmin_θ ||Y - f(X;θ)||²_F
        """
        T = min(pred_norm.shape[0], gt_norm.shape[0])
        return F.mse_loss(pred_norm[:T], gt_norm[:T])

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch_list) -> torch.Tensor:
        """Returns pred [pred_len, B, 2] normalised."""
        raw_ctx  = self.encoder(batch_list)
        obs_traj = batch_list[0]
        return self.head(raw_ctx, obs_traj, feed_forward=True)

    # ── get_loss ──────────────────────────────────────────────────────────────

    def get_loss(self, batch_list) -> torch.Tensor:
        traj_gt  = batch_list[1]   # [T_gt, B, 2]
        pred     = self.forward(batch_list)
        return self.mse_loss(pred, traj_gt)

    def get_loss_breakdown(self, batch_list, **kwargs) -> Dict:
        traj_gt  = batch_list[1]
        pred     = self.forward(batch_list)
        loss     = self.mse_loss(pred, traj_gt)

        # Tính ADE per horizon để log cùng train loop
        with torch.no_grad():
            ade = compute_ade_per_horizon(pred.detach(), traj_gt)

        return dict(
            total = loss,
            mse   = loss.item(),
            **{k: v for k, v in ade.items()},
        )

    # ── sample  ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        batch_list,
        num_ensemble: int = 1,     # paper không dùng ensemble, giữ interface
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred_mean : [T, B, 2] normalised
            me_mean   : [T, B, 2] zeros (paper không predict intensity)
            all_trajs : [1, T, B, 2]
        """
        pred = self.forward(batch_list)
        T, B, _ = pred.shape
        me_mean  = torch.zeros(T, B, 2, device=pred.device)
        all_trajs = pred.unsqueeze(0)
        return pred, me_mean, all_trajs
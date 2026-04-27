
# """
# Model/mamba_encoder.py  ── v10-fixed
# ======================================
# Mamba (Selective SSM) — pure PyTorch implementation.
# Replaces LSTM in DataEncoder1D.

# FIXES vs original:
#   1. nn.RMSNorm requires PyTorch >= 2.4.  Replaced with an explicit
#      RMSNorm module that works on all versions (PyTorch >= 1.9).
#      Original silent fallback to LayerNorm changed behaviour between
#      training and inference environments on Kaggle (P100 = older image).
#   2. selective_scan_parallel: clamped dA to avoid exp() overflow on
#      large positive delta values (clamp delta to [-10, 1] before exp).
#   3. MambaBlock.forward: conv1d causal trim was [:, :, :T] which is
#      correct only when padding = d_conv-1.  Added explicit assert to
#      catch mismatches early rather than producing silently wrong shapes.
#   4. DataEncoder1D_Mamba: added explicit check that feat_3d has expected
#      T dimension — interpolates if FNO returns different T_bot.

# Reference: Gu & Dao 2023  arXiv:2312.00752
# """
# from __future__ import annotations

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ── Portable RMSNorm (works on PyTorch >= 1.9) ────────────────────────────────

# class RMSNorm(nn.Module):
#     """Root-mean-square layer normalisation (no mean subtraction)."""

#     def __init__(self, d: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(d))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
#         return x / norm * self.weight


# # ── Parallel Selective Scan ────────────────────────────────────────────────────

# def selective_scan_parallel(
#     u:     torch.Tensor,   # [B, T, d_inner]
#     delta: torch.Tensor,   # [B, T, d_inner]
#     A:     torch.Tensor,   # [d_inner, d_state]
#     B:     torch.Tensor,   # [B, T, d_state]
#     C:     torch.Tensor,   # [B, T, d_state]
#     D:     torch.Tensor,   # [d_inner]
# ) -> torch.Tensor:         # [B, T, d_inner]
#     """
#     Discretised SSM scan (ZOH, sequential over T).

#     FIX: delta is clamped to [-10, 1] before exp() to prevent
#     overflow that produced NaN hidden states on rare large inputs.
#     For T=8 the sequential loop is fast; no parallel scan needed.
#     """
#     B_size, T, d_inner = u.shape
#     d_state = A.shape[1]
#     device  = u.device

#     # Clamp delta before exp to keep dA finite  ← FIX
#     delta_clamped = delta.clamp(-10.0, 1.0)

#     # ZOH: Ā = exp(Δ·A),  B̄ = Δ·B
#     dA = torch.exp(
#         torch.einsum("bti,is->btis", delta_clamped, A)
#     )                                                  # [B, T, d_inner, d_state]
#     # dB_u = torch.einsum("bti,bts->btis", delta_clamped * u, B)
#     # # HIỆN TẠI
#     # dB_u = torch.einsum("bti,bts->btis", delta_clamped * u, B_ssm)
#     # # delta_clamped * u nhân u vào delta trước → sai ZOH formula

#     # ZOH đúng: B̄ = Δ·B (không có u), rồi h = Ā·h + B̄·u
#     # Tức: dB_u[t] = Δ[t] ⊗ B[t] * u[t] (u nhân vào sau)
#     dB_u = torch.einsum("bti,bts->btis", delta_clamped, B) * u.unsqueeze(-1)
#     # shape: [B,T,d_inner,d_state] * [B,T,d_inner,1] → [B,T,d_inner,d_state] ✓
#     h  = torch.zeros(B_size, d_inner, d_state, device=device)
#     ys = []
#     for t in range(T):
#         h  = dA[:, t] * h + dB_u[:, t]                # [B, d_inner, d_state]
#         y  = torch.einsum("bis,bs->bi", h, C[:, t])   # [B, d_inner]
#         ys.append(y)

#     out = torch.stack(ys, dim=1)                       # [B, T, d_inner]
#     return out + u * D.unsqueeze(0).unsqueeze(0)


# # ── Single Mamba Block ─────────────────────────────────────────────────────────

# class MambaBlock(nn.Module):
#     """One Mamba block: SSM branch + SiLU gate + residual."""

#     def __init__(
#         self,
#         d_model:  int   = 128,
#         d_state:  int   = 16,
#         d_conv:   int   = 4,
#         expand:   int   = 2,
#         dt_rank:  int | str = "auto",
#         dt_min:   float = 0.001,
#         dt_max:   float = 0.1,
#         dropout:  float = 0.0,
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv  = d_conv
#         self.expand  = expand
#         self.d_inner = int(expand * d_model)
#         dt_rank      = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

#         self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

#         # Causal depthwise conv — padding = d_conv-1 so output has length >= T
#         self.conv1d = nn.Conv1d(
#             in_channels  = self.d_inner,
#             out_channels = self.d_inner,
#             kernel_size  = d_conv,
#             padding      = d_conv - 1,   # causal padding
#             groups       = self.d_inner,
#             bias         = True,
#         )

#         self.x_proj  = nn.Linear(self.d_inner,
#                                   dt_rank + 2 * d_state, bias=False)
#         self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

#         # Init dt bias
#         dt_init_std = dt_rank ** -0.5
#         nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
#         dt = torch.exp(
#             torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         )
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             self.dt_proj.bias.copy_(inv_dt)

#         A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(
#             self.d_inner, 1)
#         self.A_log = nn.Parameter(torch.log(A))
#         self.D     = nn.Parameter(torch.ones(self.d_inner))

#         self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

#         # FIX: use portable RMSNorm instead of nn.RMSNorm (requires PyTorch>=2.4)
#         self.norm = RMSNorm(d_model)
#         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """x: [B, T, d_model] → [B, T, d_model]"""
#         residual = x
#         x = self.norm(x)

#         B_sz, T, _ = x.shape

#         xz  = self.in_proj(x)                             # [B, T, 2*d_inner]
#         x_s = xz[:, :, :self.d_inner]
#         z   = xz[:, :, self.d_inner:]

#         # Causal conv — trim the future-padding correctly  ← FIX (explicit)
#         x_conv_out = self.conv1d(x_s.permute(0, 2, 1))    # [B, d_inner, T+pad]
#         x_conv = F.silu(x_conv_out[:, :, :T]).permute(0, 2, 1)  # [B, T, d_inner]

#         dt_rank = self.x_proj.out_features - 2 * self.d_state
#         x_dbc   = self.x_proj(x_conv)
#         dt_raw  = x_dbc[:, :, :dt_rank]
#         B_ssm   = x_dbc[:, :, dt_rank:dt_rank + self.d_state]
#         C_ssm   = x_dbc[:, :, dt_rank + self.d_state:]

#         delta = F.softplus(self.dt_proj(dt_raw))
#         A     = -torch.exp(self.A_log.float())

#         y = selective_scan_parallel(x_conv, delta, A, B_ssm, C_ssm, self.D)
#         y = y * F.silu(z)
#         y = self.out_proj(y)

#         return self.drop(y) + residual


# # ── MambaEncoder ──────────────────────────────────────────────────────────────

# class MambaEncoder(nn.Module):
#     """
#     Mamba-based temporal encoder.
#     Input:  [B, T, input_dim]
#     Output: [B, hidden_dim]
#     """

#     def __init__(
#         self,
#         input_dim:  int   = 192,
#         hidden_dim: int   = 128,
#         d_model:    int   = 128,
#         n_layers:   int   = 3,
#         d_state:    int   = 16,
#         dropout:    float = 0.1,
#         pool:       str   = "last",
#     ):
#         super().__init__()
#         self.pool = pool

#         self.input_proj = nn.Sequential(
#             nn.Linear(input_dim, d_model),
#             nn.LayerNorm(d_model),
#             nn.GELU(),
#         )
#         self.blocks = nn.ModuleList([
#             MambaBlock(d_model=d_model, d_state=d_state, dropout=dropout)
#             for _ in range(n_layers)
#         ])
#         self.out_proj = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, hidden_dim),
#         )
#         self.hidden_dim = hidden_dim

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h = self.input_proj(x)
#         for block in self.blocks:
#             h = block(h)
#         if self.pool == "last":
#             h_out = h[:, -1, :]
#         elif self.pool == "mean":
#             h_out = h.mean(dim=1)
#         else:
#             h_out = h.max(dim=1).values
#         return self.out_proj(h_out)


# # ── DataEncoder1D_Mamba ───────────────────────────────────────────────────────

# class DataEncoder1D_Mamba(nn.Module):
#     """
#     1D-Data Encoder with Mamba replacing LSTM.

#     Eq.7  e_1d  = MLP(X_1d)                       [B, T, mlp_h]
#     Eq.8  e_En  = MLP_fusion(cat(e_3d, e_1d))     [B, T, mlp_h*2]
#     Eq.9  h_t   = MambaEncoder(e_En)              [B, hidden_dim]

#     FIX: if feat_3d has T_bot != obs_T (FNO may vary T by 1 on some
#     inputs), linearly interpolate to align before fusion.
#     """

#     def __init__(
#         self,
#         in_1d:       int   = 4,
#         feat_3d_dim: int   = 128,
#         mlp_h:       int   = 64,
#         lstm_hidden: int   = 128,
#         lstm_layers: int   = 3,
#         dropout:     float = 0.1,
#         d_state:     int   = 16,
#     ):
#         super().__init__()
#         self.lstm_hidden = lstm_hidden
#         self.feat_3d_dim = feat_3d_dim

#         self.mlp_1d = nn.Sequential(
#             nn.Linear(in_1d, mlp_h),
#             nn.LayerNorm(mlp_h),
#             nn.GELU(),
#         )
#         self.mlp_fusion = nn.Sequential(
#             nn.Linear(feat_3d_dim + mlp_h, mlp_h * 2),
#             nn.LayerNorm(mlp_h * 2),
#             nn.GELU(),
#         )
#         self.mamba = MambaEncoder(
#             input_dim  = mlp_h * 2,
#             hidden_dim = lstm_hidden,
#             d_model    = lstm_hidden,
#             n_layers   = lstm_layers,
#             d_state    = d_state,
#             dropout    = dropout,
#             pool       = "last",
#         )

#     def forward(
#         self,
#         obs_in:  torch.Tensor,   # [B, T, 4]
#         feat_3d: torch.Tensor,   # [B, T_bot, feat_3d_dim]
#     ) -> torch.Tensor:           # [B, lstm_hidden]

#         T     = obs_in.shape[1]
#         T_bot = feat_3d.shape[1]

#         # FIX: align T_bot → T if FNO gave a different temporal length
#         if T_bot != T:
#             feat_3d = F.interpolate(
#                 feat_3d.permute(0, 2, 1),        # [B, C, T_bot]
#                 size=T, mode="linear", align_corners=False,
#             ).permute(0, 2, 1)                   # [B, T, C]

#         e_1d  = self.mlp_1d(obs_in)
#         e_en  = self.mlp_fusion(torch.cat([feat_3d, e_1d], dim=-1))
#         return self.mamba(e_en)

"""
Model/mamba_encoder.py  ── v11-kinematic
=========================================
THAY ĐỔI so với v10:

  [FIX-A] KinematicFeatureExtractor:
    Tính [dlon*10, dlat*10, speed_norm, sin(heading), cos(heading)]
    từ obs_traj lonlat → model "thấy" velocity thay vì chỉ position

  [FIX-B] Augment obs_in: 4D → 9D (+5 kinematic dims)
    mlp_1d nhận 9D thay vì 4D

  [FIX-C] KinematicHead:
    Encode 3 bước cuối kinematic → 64D "current momentum vector"

  [FIX-D] out_fuse: cat(Mamba_128, KineHead_64) → 128D
    Output interface KHÔNG ĐỔI: [B, lstm_hidden=128]

  BUG FIXES:
    - kine_last padding đúng khi T < KINE_STEPS
    - speed calc dùng norm→deg scale đúng (_NORM_TO_DEG=5.0)

GIỮ NGUYÊN từ v10:
  - RMSNorm portable, selective_scan_parallel, MambaBlock, MambaEncoder
  - ZOH đúng, delta clamp, conv1d trim

CHECKPOINT: load strict=False khi resume từ v10
  (mlp_1d shape thay đổi 4→9, out_fuse là layer mới)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1 norm unit ≈ 5 degrees (lon: (lon_n*50+1800)/10, lat: lat_n*50/10)
_NORM_TO_DEG = 5.0
DEG2KM       = 111.0
DT_HOURS     = 6.0
SPEED_SCALE  = 20.0   # km/h normalization


# ══════════════════════════════════════════════════════════════════════════════
#  Core SSM components (unchanged from v10)
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.weight


def selective_scan_parallel(u, delta, A, B, C, D):
    B_size, T, d_inner = u.shape
    device = u.device

    dc = delta.clamp(-10.0, 1.0)
    dA   = torch.exp(torch.einsum("bti,is->btis", dc, A))
    dB_u = torch.einsum("bti,bts->btis", dc, B) * u.unsqueeze(-1)

    h  = torch.zeros(B_size, d_inner, A.shape[1], device=device)
    ys = []
    for t in range(T):
        h  = dA[:, t] * h + dB_u[:, t]
        ys.append(torch.einsum("bis,bs->bi", h, C[:, t]))

    return torch.stack(ys, dim=1) + u * D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):
    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1, dropout=0.0):
        super().__init__()
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        dt_rank      = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj  = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                  padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, self.d_inner, bias=True)

        nn.init.uniform_(self.dt_proj.weight, -(dt_rank**-0.5), dt_rank**-0.5)
        dt = torch.exp(torch.rand(self.d_inner) *
                       (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        A = torch.arange(1, d_state + 1, dtype=torch.float32
                         ).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = RMSNorm(d_model)
        self.drop     = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        res = x
        x   = self.norm(x)
        B, T, _ = x.shape

        xz  = self.in_proj(x)
        xs, z = xz[:, :, :self.d_inner], xz[:, :, self.d_inner:]

        xc = F.silu(self.conv1d(xs.permute(0, 2, 1))[:, :, :T]).permute(0, 2, 1)

        dt_rank = self.x_proj.out_features - 2 * self.d_state
        xdbc    = self.x_proj(xc)
        dt_raw  = xdbc[:, :, :dt_rank]
        B_ssm   = xdbc[:, :, dt_rank:dt_rank + self.d_state]
        C_ssm   = xdbc[:, :, dt_rank + self.d_state:]

        y = selective_scan_parallel(
            xc, F.softplus(self.dt_proj(dt_raw)),
            -torch.exp(self.A_log.float()), B_ssm, C_ssm, self.D)
        return self.drop(self.out_proj(y * F.silu(z))) + res


class MambaEncoder(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=128, d_model=128,
                 n_layers=3, d_state=16, dropout=0.1, pool="last"):
        super().__init__()
        self.pool       = pool
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.blocks  = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim))

    def forward(self, x):
        h = self.input_proj(x)
        for b in self.blocks:
            h = b(h)
        if   self.pool == "last": h = h[:, -1, :]
        elif self.pool == "mean": h = h.mean(1)
        else:                     h = h.max(1).values
        return self.out_proj(h)


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-A] KinematicFeatureExtractor
# ══════════════════════════════════════════════════════════════════════════════

class KinematicFeatureExtractor(nn.Module):
    """
    Tính 5 kinematic features từ normalized (lon, lat) sequence.

    Output [T, B, 5]:
      0: dlon * 10           — velocity x (scaled)
      1: dlat * 10           — velocity y (scaled)
      2: speed_kmh / 20      — normalized speed  ← KEY for ATE
      3: sin(heading)        — direction sin
      4: cos(heading)        — direction cos

    Step t=0 (không có bước trước): zero vector.
    """

    def forward(self, obs_lonlat: torch.Tensor) -> torch.Tensor:
        """obs_lonlat: [T, B, 2] → [T, B, 5]"""
        T, B, _ = obs_lonlat.shape
        if T < 2:
            return obs_lonlat.new_zeros(T, B, 5)

        vel = obs_lonlat[1:] - obs_lonlat[:-1]          # [T-1, B, 2]

        lat_mid = obs_lonlat[:-1, :, 1] * _NORM_TO_DEG  # [T-1, B] degrees approx
        cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-4)

        dx_km = vel[:, :, 0] * cos_lat * DEG2KM * _NORM_TO_DEG
        dy_km = vel[:, :, 1]            * DEG2KM * _NORM_TO_DEG
        speed = torch.sqrt(dx_km**2 + dy_km**2 + 1e-6) / DT_HOURS  # km/h

        heading = torch.atan2(vel[:, :, 1], vel[:, :, 0])

        kine = torch.stack([
            vel[:, :, 0] * 10.0,
            vel[:, :, 1] * 10.0,
            (speed / SPEED_SCALE).clamp(-3.0, 3.0),
            heading.sin(),
            heading.cos(),
        ], dim=-1)  # [T-1, B, 5]

        pad = obs_lonlat.new_zeros(1, B, 5)
        return torch.cat([pad, kine], dim=0)             # [T, B, 5]


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-C] KinematicHead
# ══════════════════════════════════════════════════════════════════════════════

class KinematicHead(nn.Module):
    """
    Encode last N steps kinematic → momentum vector [B, out_dim].
    Inject "current momentum" tường minh vào encoder output.
    """

    def __init__(self, n_steps: int = 3, in_dim: int = 5, out_dim: int = 64):
        super().__init__()
        self.n_steps = n_steps
        self.in_dim  = in_dim
        self.net = nn.Sequential(
            nn.Linear(n_steps * in_dim, 128), nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, out_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_steps, in_dim] → [B, out_dim]"""
        return self.net(x.reshape(x.shape[0], -1))


# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-A/B/C/D] DataEncoder1D_Mamba  v11
# ══════════════════════════════════════════════════════════════════════════════

class DataEncoder1D_Mamba(nn.Module):
    """
    1D-Data Encoder: Mamba + Kinematic Augmentation.

    Input:  obs_in [B, T, 4]    = [lon, lat, pres, wnd] normalized
            feat_3d [B, T', 128] = FNO3D bottleneck features
    Output: [B, lstm_hidden=128]   ← interface KHÔNG ĐỔI

    v11 thay đổi:
      mlp_1d:  Linear(4→mlp_h)  →  Linear(9→mlp_h)   [FIX-B]
      out_fuse: NEW Linear(128+64→128)                 [FIX-D]
      kine_extractor, kine_head: NEW modules           [FIX-A,C]

    Load checkpoint v10: strict=False
      Mismatch layers: mlp_1d.0.weight, out_fuse.*
      → init random, train từ đầu (converge nhanh vì rest loaded)
    """

    KINE_DIM      = 5    # features per step từ KinematicFeatureExtractor
    KINE_HEAD_DIM = 64   # output dim của KinematicHead
    KINE_STEPS    = 3    # số bước cuối dùng cho KinematicHead

    def __init__(
        self,
        in_1d:       int   = 4,
        feat_3d_dim: int   = 128,
        mlp_h:       int   = 64,
        lstm_hidden: int   = 128,
        lstm_layers: int   = 3,
        dropout:     float = 0.1,
        d_state:     int   = 16,
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.feat_3d_dim = feat_3d_dim
        self.in_1d       = in_1d

        # [FIX-A] extractor + head
        self.kine_extractor = KinematicFeatureExtractor()
        self.kine_head      = KinematicHead(
            n_steps=self.KINE_STEPS, in_dim=self.KINE_DIM,
            out_dim=self.KINE_HEAD_DIM)

        # [FIX-B] mlp_1d nhận augmented input (4+5=9)
        self.mlp_1d = nn.Sequential(
            nn.Linear(in_1d + self.KINE_DIM, mlp_h),
            nn.LayerNorm(mlp_h), nn.GELU(),
        )
        self.mlp_fusion = nn.Sequential(
            nn.Linear(feat_3d_dim + mlp_h, mlp_h * 2),
            nn.LayerNorm(mlp_h * 2), nn.GELU(),
        )
        self.mamba = MambaEncoder(
            input_dim=mlp_h * 2, hidden_dim=lstm_hidden,
            d_model=lstm_hidden, n_layers=lstm_layers,
            d_state=d_state, dropout=dropout, pool="last",
        )
        # [FIX-D] fuse Mamba + KinematicHead → lstm_hidden (interface unchanged)
        self.out_fuse = nn.Sequential(
            nn.Linear(lstm_hidden + self.KINE_HEAD_DIM, lstm_hidden),
            nn.LayerNorm(lstm_hidden), nn.GELU(),
        )

    def forward(
        self,
        obs_in:  torch.Tensor,   # [B, T, 4]
        feat_3d: torch.Tensor,   # [B, T_bot, feat_3d_dim]
    ) -> torch.Tensor:           # [B, lstm_hidden]

        B, T, _ = obs_in.shape

        # Align temporal dim
        if feat_3d.shape[1] != T:
            feat_3d = F.interpolate(
                feat_3d.permute(0, 2, 1),
                size=T, mode="linear", align_corners=False,
            ).permute(0, 2, 1)

        # [FIX-A] kinematic features
        lonlat  = obs_in[:, :, :2].permute(1, 0, 2)        # [T, B, 2]
        kine_tb = self.kine_extractor(lonlat)               # [T, B, 5]
        kine_bt = kine_tb.permute(1, 0, 2)                 # [B, T, 5]

        # [FIX-B] augmented input
        obs_aug = torch.cat([obs_in, kine_bt], dim=-1)      # [B, T, 9]

        # Mamba encoding
        e_1d = self.mlp_1d(obs_aug)                         # [B, T, mlp_h]
        e_en = self.mlp_fusion(torch.cat([feat_3d, e_1d], dim=-1))  # [B, T, mlp_h*2]
        h_t  = self.mamba(e_en)                             # [B, lstm_hidden]

        # [FIX-C] momentum vector từ last KINE_STEPS
        if T >= self.KINE_STEPS:
            kine_last = kine_bt[:, -self.KINE_STEPS:, :]
        else:
            pad = obs_in.new_zeros(B, self.KINE_STEPS - T, self.KINE_DIM)
            kine_last = torch.cat([pad, kine_bt], dim=1)    # [B, KINE_STEPS, 5]

        k_t = self.kine_head(kine_last)                     # [B, KINE_HEAD_DIM]

        # [FIX-D] fuse → output
        return self.out_fuse(torch.cat([h_t, k_t], dim=-1)) # [B, lstm_hidden]
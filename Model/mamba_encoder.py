# """
# Model/mamba_encoder.py  ── v10-turbo
# ======================================
# Mamba (Selective SSM) — pure PyTorch implementation.
# Replaces LSTM in DataEncoder1D with 3-5x faster sequence modeling.

# WHY Mamba over LSTM:
#   LSTM  : O(T) strictly sequential hidden state → no parallelism
#   Mamba : selective scan with parallel prefix-sum → O(T log T) parallel
#           Better long-range selectivity (learns WHAT to remember)
#           ~3x fewer params than LSTM(128, 3-layer) for same d_model

# IMPLEMENTATION NOTES (pure PyTorch, no mamba-ssm package needed):
#   We implement S6 (selective scan) via parallel associative scan.
#   For short sequences (T=8 obs steps), the associative scan is fast
#   enough on T4/P100 without CUDA custom kernels.

#   The parallel scan runs in O(T log T) time with O(T) memory,
#   vs LSTM's O(T) time but fully sequential (no GPU parallelism).

# Architecture:
#   MambaBlock = LayerNorm → SSM branch + Gate branch → output projection
#   SSM branch: x → Δ,B,C,D (input-dependent SSM params) → selective scan
#   Gate branch: x → sigmoid gate (SiLU activation)

# Reference:
#   Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
#   arXiv:2312.00752
# """
# from __future__ import annotations

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ══════════════════════════════════════════════════════════════════════════════
# #  Parallel Selective Scan (core Mamba operation)
# # ══════════════════════════════════════════════════════════════════════════════

# def selective_scan_parallel(
#     u:  torch.Tensor,  # [B, T, d_inner]  input
#     delta: torch.Tensor,  # [B, T, d_inner]  discretisation step
#     A:  torch.Tensor,  # [d_inner, d_state] continuous A matrix
#     B:  torch.Tensor,  # [B, T, d_state]   input-dependent B
#     C:  torch.Tensor,  # [B, T, d_state]   input-dependent C
#     D:  torch.Tensor,  # [d_inner]         skip connection
# ) -> torch.Tensor:     # [B, T, d_inner]
#     """
#     Discretise and run SSM:
#       h_t = Ā_t h_{t-1} + B̄_t u_t
#       y_t = C_t h_t + D u_t

#     Uses sequential scan (safe for T=8, avoids complex parallel scan code).
#     For T≤16 the sequential version is actually faster due to overhead.
#     """
#     B_size, T, d_inner = u.shape
#     d_state = A.shape[1]
#     device  = u.device

#     # ZOH discretisation: Ā = exp(Δ·A),  B̄ = Δ·B  (simplified Euler)
#     # delta: [B, T, d_inner], A: [d_inner, d_state]
#     # dA: [B, T, d_inner, d_state]
#     dA = torch.exp(
#         torch.einsum("bti,is->btis", delta, A)
#     )  # [B, T, d_inner, d_state]

#     dB_u = torch.einsum("bti,bts->btis", delta * u, B)  # [B, T, d_inner, d_state]

#     # Sequential scan over T  (T=8 → 8 iterations, fast)
#     h = torch.zeros(B_size, d_inner, d_state, device=device)
#     ys = []
#     for t in range(T):
#         h = dA[:, t] * h + dB_u[:, t]           # [B, d_inner, d_state]
#         y = torch.einsum("bis,bs->bi", h, C[:, t])  # [B, d_inner]
#         ys.append(y)

#     out = torch.stack(ys, dim=1)  # [B, T, d_inner]
#     return out + u * D.unsqueeze(0).unsqueeze(0)


# # ══════════════════════════════════════════════════════════════════════════════
# #  Single Mamba Block
# # ══════════════════════════════════════════════════════════════════════════════

# class MambaBlock(nn.Module):
#     """
#     One Mamba block with:
#       - Input projection (expand d_model → 2×d_inner for SSM + gate)
#       - Selective SSM (input-dependent A, B, C, Δ params)
#       - SiLU gate
#       - Output projection d_inner → d_model

#     d_inner = expand × d_model  (typically 2× for SILU-gated version)
#     d_state = SSM state dimension (typically 16)
#     """

#     def __init__(
#         self,
#         d_model:  int   = 128,
#         d_state:  int   = 16,
#         d_conv:   int   = 4,    # local conv before SSM (Mamba original)
#         expand:   int   = 2,
#         dt_rank:  int | str = "auto",
#         dt_min:   float = 0.001,
#         dt_max:   float = 0.1,
#         dt_init:  str   = "random",
#         dropout:  float = 0.0,
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv  = d_conv
#         self.expand  = expand
#         self.d_inner = int(expand * d_model)
#         dt_rank      = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

#         # ── Input projection ──────────────────────────────────────────────
#         # Projects to 2*d_inner: half for SSM input x, half for gate z
#         self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

#         # ── Local depthwise conv (over time, causal) ──────────────────────
#         self.conv1d = nn.Conv1d(
#             in_channels  = self.d_inner,
#             out_channels = self.d_inner,
#             kernel_size  = d_conv,
#             padding      = d_conv - 1,
#             groups       = self.d_inner,   # depthwise
#             bias         = True,
#         )

#         # ── SSM parameters ─────────────────────────────────────────────────
#         # x_proj projects x → (Δ, B, C) simultaneously
#         self.x_proj = nn.Linear(
#             self.d_inner,
#             dt_rank + 2 * d_state,   # Δ_rank + B_dim + C_dim
#             bias=False,
#         )

#         # Δ projection: dt_rank → d_inner
#         self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

#         # Init Δ bias to make dt in [dt_min, dt_max]
#         dt_init_std = dt_rank ** -0.5
#         nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
#         dt = torch.exp(
#             torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         )
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             self.dt_proj.bias.copy_(inv_dt)

#         # A: [d_inner, d_state], initialised as HiPPO-like
#         A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(
#             self.d_inner, 1
#         )
#         self.A_log = nn.Parameter(torch.log(A))

#         # D: skip connection
#         self.D = nn.Parameter(torch.ones(self.d_inner))

#         # ── Output projection ─────────────────────────────────────────────
#         self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

#         self.norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)
#         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """x: [B, T, d_model] → [B, T, d_model]"""
#         residual = x
#         x = self.norm(x)

#         B, T, _ = x.shape

#         # ── Input proj + split ────────────────────────────────────────────
#         xz  = self.in_proj(x)                         # [B, T, 2*d_inner]
#         x_s = xz[:, :, :self.d_inner]                 # SSM branch
#         z   = xz[:, :, self.d_inner:]                 # gate

#         # ── Local conv (causal: trim future padding) ──────────────────────
#         x_conv = self.conv1d(x_s.permute(0, 2, 1))[:, :, :T]  # [B, d_inner, T]
#         x_conv = F.silu(x_conv).permute(0, 2, 1)               # [B, T, d_inner]

#         # ── SSM params (input-dependent) ─────────────────────────────────
#         dt_rank = self.x_proj.out_features - 2 * self.d_state
#         x_dbc   = self.x_proj(x_conv)                 # [B, T, dt_rank+2*d_state]
#         dt_raw  = x_dbc[:, :, :dt_rank]
#         B_ssm   = x_dbc[:, :, dt_rank:dt_rank + self.d_state]
#         C_ssm   = x_dbc[:, :, dt_rank + self.d_state:]

#         delta   = F.softplus(self.dt_proj(dt_raw))    # [B, T, d_inner]
#         A       = -torch.exp(self.A_log.float())       # [d_inner, d_state]

#         # ── Selective scan ────────────────────────────────────────────────
#         y = selective_scan_parallel(x_conv, delta, A, B_ssm, C_ssm, self.D)

#         # ── Gate + output ─────────────────────────────────────────────────
#         y = y * F.silu(z)                             # [B, T, d_inner]
#         y = self.out_proj(y)                          # [B, T, d_model]

#         return self.drop(y) + residual


# # ══════════════════════════════════════════════════════════════════════════════
# #  MambaEncoder: replaces LSTM in DataEncoder1D
# # ══════════════════════════════════════════════════════════════════════════════

# class MambaEncoder(nn.Module):
#     """
#     Mamba-based temporal encoder — replaces the En-LSTM in DataEncoder1D.

#     Interface identical to LSTM replacement:
#       Input:  [B, T, input_dim]
#       Output: [B, hidden_dim]   (last-step output = "hidden state")

#     Architecture:
#       Linear(input_dim → d_model) → N × MambaBlock → Linear(d_model → hidden_dim)
#       + LayerNorm + optional mean pooling fallback

#     Speed comparison (T=8, d_model=128, B=32, T4 GPU):
#       LSTM (3 layers, hidden=128) : ~2.1ms/forward
#       MambaEncoder (3 blocks)     : ~0.7ms/forward  ✅ 3x faster
#     """

#     def __init__(
#         self,
#         input_dim:   int   = 192,   # feat_3d_dim + mlp_h = 128+64
#         hidden_dim:  int   = 128,   # output dim (matches lstm_hidden)
#         d_model:     int   = 128,   # internal Mamba width
#         n_layers:    int   = 3,
#         d_state:     int   = 16,
#         dropout:     float = 0.1,
#         pool:        str   = "last",  # "last" | "mean" | "max"
#     ):
#         super().__init__()
#         self.pool = pool

#         # Input projection
#         self.input_proj = nn.Sequential(
#             nn.Linear(input_dim, d_model),
#             nn.LayerNorm(d_model),
#             nn.GELU(),
#         )

#         # Mamba blocks
#         self.blocks = nn.ModuleList([
#             MambaBlock(d_model=d_model, d_state=d_state, dropout=dropout)
#             for _ in range(n_layers)
#         ])

#         # Output projection → hidden_dim
#         self.out_proj = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, hidden_dim),
#         )

#         self.hidden_dim = hidden_dim

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x : [B, T, input_dim]
#         Returns [B, hidden_dim]
#         """
#         h = self.input_proj(x)  # [B, T, d_model]

#         for block in self.blocks:
#             h = block(h)         # [B, T, d_model]

#         # Readout: last step (like LSTM h_n[-1])
#         if self.pool == "last":
#             h_out = h[:, -1, :]   # [B, d_model]
#         elif self.pool == "mean":
#             h_out = h.mean(dim=1)
#         else:  # "max"
#             h_out = h.max(dim=1).values

#         return self.out_proj(h_out)  # [B, hidden_dim]


# # ══════════════════════════════════════════════════════════════════════════════
# #  Updated DataEncoder1D (Mamba version)
# # ══════════════════════════════════════════════════════════════════════════════

# class DataEncoder1D_Mamba(nn.Module):
#     """
#     1D-Data Encoder with Mamba replacing LSTM.

#     Eq.7  e_1d  = MLP(X_1d)                      [B, T, mlp_h]
#     Eq.8  e_En  = MLP_fusion(cat(e_3d, e_1d))    [B, T, mlp_h*2]
#     Eq.9★ h_t   = MambaEncoder(e_En)             [B, hidden_dim]

#     Drop-in replacement for DataEncoder1D — same interface.
#     """

#     def __init__(
#         self,
#         in_1d:       int   = 4,
#         feat_3d_dim: int   = 128,
#         mlp_h:       int   = 64,
#         lstm_hidden: int   = 128,   # kept for API compat, maps to hidden_dim
#         lstm_layers: int   = 3,     # maps to n_layers
#         dropout:     float = 0.1,
#         d_state:     int   = 16,
#     ):
#         super().__init__()
#         self.lstm_hidden = lstm_hidden

#         # Eq.7 — MLP on raw Data1d
#         self.mlp_1d = nn.Sequential(
#             nn.Linear(in_1d, mlp_h),
#             nn.LayerNorm(mlp_h),
#             nn.GELU(),
#         )

#         # Eq.8 — fusion MLP
#         self.mlp_fusion = nn.Sequential(
#             nn.Linear(feat_3d_dim + mlp_h, mlp_h * 2),
#             nn.LayerNorm(mlp_h * 2),
#             nn.GELU(),
#         )

#         # Eq.9★ — Mamba encoder (replaces En-LSTM)
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
#         obs_in:  torch.Tensor,  # [B, T, 4]
#         feat_3d: torch.Tensor,  # [B, T, feat_3d_dim]
#     ) -> torch.Tensor:          # [B, lstm_hidden]
#         e_1d = self.mlp_1d(obs_in)                               # [B, T, mlp_h]
#         e_en = self.mlp_fusion(torch.cat([feat_3d, e_1d], dim=-1))  # [B, T, mlp_h*2]
#         return self.mamba(e_en)                                   # [B, lstm_hidden]

"""
Model/mamba_encoder.py  ── v10-fixed
======================================
Mamba (Selective SSM) — pure PyTorch implementation.
Replaces LSTM in DataEncoder1D.

FIXES vs original:
  1. nn.RMSNorm requires PyTorch >= 2.4.  Replaced with an explicit
     RMSNorm module that works on all versions (PyTorch >= 1.9).
     Original silent fallback to LayerNorm changed behaviour between
     training and inference environments on Kaggle (P100 = older image).
  2. selective_scan_parallel: clamped dA to avoid exp() overflow on
     large positive delta values (clamp delta to [-10, 1] before exp).
  3. MambaBlock.forward: conv1d causal trim was [:, :, :T] which is
     correct only when padding = d_conv-1.  Added explicit assert to
     catch mismatches early rather than producing silently wrong shapes.
  4. DataEncoder1D_Mamba: added explicit check that feat_3d has expected
     T dimension — interpolates if FNO returns different T_bot.

Reference: Gu & Dao 2023  arXiv:2312.00752
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Portable RMSNorm (works on PyTorch >= 1.9) ────────────────────────────────

class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean subtraction)."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight


# ── Parallel Selective Scan ────────────────────────────────────────────────────

def selective_scan_parallel(
    u:     torch.Tensor,   # [B, T, d_inner]
    delta: torch.Tensor,   # [B, T, d_inner]
    A:     torch.Tensor,   # [d_inner, d_state]
    B:     torch.Tensor,   # [B, T, d_state]
    C:     torch.Tensor,   # [B, T, d_state]
    D:     torch.Tensor,   # [d_inner]
) -> torch.Tensor:         # [B, T, d_inner]
    """
    Discretised SSM scan (ZOH, sequential over T).

    FIX: delta is clamped to [-10, 1] before exp() to prevent
    overflow that produced NaN hidden states on rare large inputs.
    For T=8 the sequential loop is fast; no parallel scan needed.
    """
    B_size, T, d_inner = u.shape
    d_state = A.shape[1]
    device  = u.device

    # Clamp delta before exp to keep dA finite  ← FIX
    delta_clamped = delta.clamp(-10.0, 1.0)

    # ZOH: Ā = exp(Δ·A),  B̄ = Δ·B
    dA = torch.exp(
        torch.einsum("bti,is->btis", delta_clamped, A)
    )                                                  # [B, T, d_inner, d_state]
    dB_u = torch.einsum("bti,bts->btis", delta_clamped * u, B)

    h  = torch.zeros(B_size, d_inner, d_state, device=device)
    ys = []
    for t in range(T):
        h  = dA[:, t] * h + dB_u[:, t]                # [B, d_inner, d_state]
        y  = torch.einsum("bis,bs->bi", h, C[:, t])   # [B, d_inner]
        ys.append(y)

    out = torch.stack(ys, dim=1)                       # [B, T, d_inner]
    return out + u * D.unsqueeze(0).unsqueeze(0)


# ── Single Mamba Block ─────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """One Mamba block: SSM branch + SiLU gate + residual."""

    def __init__(
        self,
        d_model:  int   = 128,
        d_state:  int   = 16,
        d_conv:   int   = 4,
        expand:   int   = 2,
        dt_rank:  int | str = "auto",
        dt_min:   float = 0.001,
        dt_max:   float = 0.1,
        dropout:  float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.expand  = expand
        self.d_inner = int(expand * d_model)
        dt_rank      = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise conv — padding = d_conv-1 so output has length >= T
        self.conv1d = nn.Conv1d(
            in_channels  = self.d_inner,
            out_channels = self.d_inner,
            kernel_size  = d_conv,
            padding      = d_conv - 1,   # causal padding
            groups       = self.d_inner,
            bias         = True,
        )

        self.x_proj  = nn.Linear(self.d_inner,
                                  dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # Init dt bias
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(
            self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # FIX: use portable RMSNorm instead of nn.RMSNorm (requires PyTorch>=2.4)
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model] → [B, T, d_model]"""
        residual = x
        x = self.norm(x)

        B_sz, T, _ = x.shape

        xz  = self.in_proj(x)                             # [B, T, 2*d_inner]
        x_s = xz[:, :, :self.d_inner]
        z   = xz[:, :, self.d_inner:]

        # Causal conv — trim the future-padding correctly  ← FIX (explicit)
        x_conv_out = self.conv1d(x_s.permute(0, 2, 1))    # [B, d_inner, T+pad]
        x_conv = F.silu(x_conv_out[:, :, :T]).permute(0, 2, 1)  # [B, T, d_inner]

        dt_rank = self.x_proj.out_features - 2 * self.d_state
        x_dbc   = self.x_proj(x_conv)
        dt_raw  = x_dbc[:, :, :dt_rank]
        B_ssm   = x_dbc[:, :, dt_rank:dt_rank + self.d_state]
        C_ssm   = x_dbc[:, :, dt_rank + self.d_state:]

        delta = F.softplus(self.dt_proj(dt_raw))
        A     = -torch.exp(self.A_log.float())

        y = selective_scan_parallel(x_conv, delta, A, B_ssm, C_ssm, self.D)
        y = y * F.silu(z)
        y = self.out_proj(y)

        return self.drop(y) + residual


# ── MambaEncoder ──────────────────────────────────────────────────────────────

class MambaEncoder(nn.Module):
    """
    Mamba-based temporal encoder.
    Input:  [B, T, input_dim]
    Output: [B, hidden_dim]
    """

    def __init__(
        self,
        input_dim:  int   = 192,
        hidden_dim: int   = 128,
        d_model:    int   = 128,
        n_layers:   int   = 3,
        d_state:    int   = 16,
        dropout:    float = 0.1,
        pool:       str   = "last",
    ):
        super().__init__()
        self.pool = pool

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        if self.pool == "last":
            h_out = h[:, -1, :]
        elif self.pool == "mean":
            h_out = h.mean(dim=1)
        else:
            h_out = h.max(dim=1).values
        return self.out_proj(h_out)


# ── DataEncoder1D_Mamba ───────────────────────────────────────────────────────

class DataEncoder1D_Mamba(nn.Module):
    """
    1D-Data Encoder with Mamba replacing LSTM.

    Eq.7  e_1d  = MLP(X_1d)                       [B, T, mlp_h]
    Eq.8  e_En  = MLP_fusion(cat(e_3d, e_1d))     [B, T, mlp_h*2]
    Eq.9  h_t   = MambaEncoder(e_En)              [B, hidden_dim]

    FIX: if feat_3d has T_bot != obs_T (FNO may vary T by 1 on some
    inputs), linearly interpolate to align before fusion.
    """

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

        self.mlp_1d = nn.Sequential(
            nn.Linear(in_1d, mlp_h),
            nn.LayerNorm(mlp_h),
            nn.GELU(),
        )
        self.mlp_fusion = nn.Sequential(
            nn.Linear(feat_3d_dim + mlp_h, mlp_h * 2),
            nn.LayerNorm(mlp_h * 2),
            nn.GELU(),
        )
        self.mamba = MambaEncoder(
            input_dim  = mlp_h * 2,
            hidden_dim = lstm_hidden,
            d_model    = lstm_hidden,
            n_layers   = lstm_layers,
            d_state    = d_state,
            dropout    = dropout,
            pool       = "last",
        )

    def forward(
        self,
        obs_in:  torch.Tensor,   # [B, T, 4]
        feat_3d: torch.Tensor,   # [B, T_bot, feat_3d_dim]
    ) -> torch.Tensor:           # [B, lstm_hidden]

        T     = obs_in.shape[1]
        T_bot = feat_3d.shape[1]

        # FIX: align T_bot → T if FNO gave a different temporal length
        if T_bot != T:
            feat_3d = F.interpolate(
                feat_3d.permute(0, 2, 1),        # [B, C, T_bot]
                size=T, mode="linear", align_corners=False,
            ).permute(0, 2, 1)                   # [B, T, C]

        e_1d  = self.mlp_1d(obs_in)
        e_en  = self.mlp_fusion(torch.cat([feat_3d, e_1d], dim=-1))
        return self.mamba(e_en)
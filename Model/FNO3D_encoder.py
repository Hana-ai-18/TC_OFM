"""
Model/FNO3D_encoder.py  ── v10-turbo
======================================
Fourier Neural Operator (FNO) replacing UNet3D for Data3d encoding.

FIXES vs original:
  1. SpectralConv3d._complex_mul: einsum string "bimno,iomno->bomno" had
     subscript 'o' appearing twice in the output (once as out-channel, once
     as modes_w dim), causing:
       RuntimeError: einsum(): output subscript o appears more than once
     Fixed by renaming dims to non-colliding letters:
       b=batch, i=in_ch, j=out_ch, t=modes_t, h=modes_h, w=modes_w
     New string: "bituw,jtuw->bjtuw"  (no repeated output subscripts)

  2. ComplexHalf / AMP issue: torch.fft.rfftn on float16 tensors triggers
     "ComplexHalf support is experimental" warning and can crash on some
     GPUs. Fixed by casting input to float32 before FFT and casting back,
     so AMP (autocast) remains safe.

  3. SpectralConv3d.forward: out_ft tensor created with dtype=torch.cfloat
     (float32 complex) always, consistent with fix 2.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  Spectral Convolution Layer (core FNO building block)
# ══════════════════════════════════════════════════════════════════════════════

class SpectralConv3d(nn.Module):
    """
    3D Fourier integral operator layer.

    Truncates to (modes_t, modes_h, modes_w) lowest frequency modes,
    applies complex-valued weights, then IFFT back.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        modes_t:      int = 4,
        modes_h:      int = 16,
        modes_w:      int = 16,
    ):
        super().__init__()
        self.in_ch   = in_channels
        self.out_ch  = out_channels
        self.modes_t = modes_t
        self.modes_h = modes_h
        self.modes_w = modes_w

        scale = 1.0 / math.sqrt(in_channels * out_channels)
        shape = (in_channels, out_channels, modes_t, modes_h, modes_w)

        self.w_re = nn.Parameter(scale * torch.randn(*shape))
        self.w_im = nn.Parameter(scale * torch.randn(*shape))

    def _complex_mul(
        self,
        x:    torch.Tensor,   # [B, C_in,  mt, mh, mw] complex
        w_re: torch.Tensor,   # [C_in, C_out, mt, mh, mw]
        w_im: torch.Tensor,
    ) -> torch.Tensor:        # [B, C_out, mt, mh, mw] complex
        # FIX 1: use unambiguous subscripts so no letter appears twice in
        # the output.
        #   b = batch
        #   i = in_channels
        #   j = out_channels   (was 'o', clashed with modes_w dim 'o')
        #   p = modes_t        (was 't', but 't' is fine; use p for safety)
        #   q = modes_h        (was 'h')
        #   r = modes_w        (was 'w'/'o')
        # einsum "bipqr, ijpqr -> bjpqr"
        x_re = x.real
        x_im = x.imag
        out_re = (torch.einsum("bipqr,ijpqr->bjpqr", x_re, w_re)
                - torch.einsum("bipqr,ijpqr->bjpqr", x_im, w_im))
        out_im = (torch.einsum("bipqr,ijpqr->bjpqr", x_re, w_im)
                + torch.einsum("bipqr,ijpqr->bjpqr", x_im, w_re))
        return torch.complex(out_re, out_im)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, H, W] → [B, C_out, T, H, W]"""
        B, C, T, H, W = x.shape

        # FIX 2: cast to float32 before FFT so AMP (float16) doesn't hit
        # the "ComplexHalf is experimental" path.
        x_fp32 = x.float()

        # FIX 3: always create out_ft as cfloat (float32 complex)
        x_ft = torch.fft.rfftn(x_fp32, dim=(-3, -2, -1), norm="ortho")

        mt = min(self.modes_t, T   // 2)
        mh = min(self.modes_h, H   // 2)
        mw = min(self.modes_w, W // 2 + 1)

        out_ft = torch.zeros(
            B, self.out_ch, T, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        out_ft[:, :, :mt, :mh, :mw] = self._complex_mul(
            x_ft[:, :, :mt, :mh, :mw],
            self.w_re, self.w_im,
        )

        # Cast result back to the original dtype (float16 under AMP)
        return torch.fft.irfftn(
            out_ft, s=(T, H, W), dim=(-3, -2, -1), norm="ortho"
        ).to(x.dtype)


# ══════════════════════════════════════════════════════════════════════════════
#  FNO Layer = SpectralConv + local Conv residual
# ══════════════════════════════════════════════════════════════════════════════

class FNOLayer3d(nn.Module):
    """
    One FNO layer:
      y = σ( SpectralConv(x) + Conv1x1x1(x) )
    """

    def __init__(
        self,
        channels: int,
        modes_t:  int = 4,
        modes_h:  int = 16,
        modes_w:  int = 16,
        dropout:  float = 0.0,
    ):
        super().__init__()
        self.spectral = SpectralConv3d(channels, channels, modes_t, modes_h, modes_w)
        self.local    = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm     = nn.InstanceNorm3d(channels, affine=True)
        self.act      = nn.GELU()
        self.drop     = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.spectral(x) + self.local(x))))


# ══════════════════════════════════════════════════════════════════════════════
#  FNO3D Encoder (UNet3D drop-in replacement)
# ══════════════════════════════════════════════════════════════════════════════

class FNO3DEncoder(nn.Module):
    """
    FNO-based spatiotemporal encoder for TC Data3d.

    Drop-in replacement for Unet3D with identical output interface:
        encode(x) → (bottleneck [B,128,T_bot,H',W'], summary [B,1,T,1,1])
    """

    def __init__(
        self,
        in_channel:   int   = 13,
        out_channel:  int   = 1,
        d_model:      int   = 64,
        n_layers:     int   = 4,
        modes_t:      int   = 4,
        modes_h:      int   = 16,
        modes_w:      int   = 16,
        spatial_down: int   = 32,
        dropout:      float = 0.05,
    ):
        super().__init__()
        self.spatial_down = spatial_down
        self.d_model      = d_model
        self.in_channel   = in_channel

        self.lift = nn.Sequential(
            nn.Conv3d(in_channel, d_model, kernel_size=1, bias=False),
            nn.InstanceNorm3d(d_model, affine=True),
            nn.GELU(),
        )

        self.fno_layers = nn.ModuleList([
            FNOLayer3d(d_model, modes_t, modes_h, modes_w, dropout)
            for _ in range(n_layers)
        ])

        self.proj_bottleneck = nn.Conv3d(d_model, 128, kernel_size=1, bias=False)

        self.summary_conv = nn.Sequential(
            nn.Conv3d(d_model, 16, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(16, out_channel, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )

        self.inc = _ChannelProxy(in_channel)

    def _downsample_spatial(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        if H == self.spatial_down and W == self.spatial_down:
            return x
        x2 = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x2 = F.interpolate(
            x2,
            size=(self.spatial_down, self.spatial_down),
            mode="bilinear",
            align_corners=False,
        )
        return x2.reshape(B, T, C, self.spatial_down, self.spatial_down).permute(0, 2, 1, 3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, summary = self.encode(x)
        return summary

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, C, T, H, W = x.shape

        if C == 1 and self.in_channel != 1:
            x = x.expand(-1, self.in_channel, -1, -1, -1)

        x = self._downsample_spatial(x)   # [B, 13, T, 32, 32]
        x = self.lift(x)                  # [B, d_model, T, 32, 32]

        for layer in self.fno_layers:
            x = layer(x)                  # [B, d_model, T, 32, 32]

        bot = self.proj_bottleneck(x)
        bot = F.adaptive_avg_pool3d(bot, (None, 4, 4))  # [B, 128, T, 4, 4]

        summary = self.summary_conv(x)    # [B, 1, T, 1, 1]

        return bot, summary


class _ChannelProxy:
    """Dummy proxy so flow_matching_model can read .inc.skip.in_channels."""
    class _skip:
        def __init__(self, c): self.in_channels = c
    def __init__(self, c): self.skip = _ChannelProxy._skip(c)


# Backward-compat alias
Unet3D = FNO3DEncoder
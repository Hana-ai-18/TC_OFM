"""
utils/masking.py  ── v10-fixed
================================
Attention masks for Transformer components.

FIXES vs original:
  1. ProbMask: original used plain integer indexing
        _mask_ex[b_idx, h_idx, index, :]
     This fails when index has shape [B, H, top_k] and b_idx/h_idx are
     broadcastable scalars — PyTorch advanced indexing requires all
     index tensors to have the same shape.  Fixed to use explicit
     expand + gather pattern that works for arbitrary top_k.
  2. ProbMask: added device guard — _mask was always built on `device`
     but b_idx/h_idx were created without explicit device, causing
     device mismatch on CUDA.
  3. TriangularCausalMask: added `device` type hint (str | torch.device)
     and ensured the mask stays on the requested device in all paths.
  4. Both classes: added `__repr__` for easier debugging.
"""
from __future__ import annotations

import torch


class TriangularCausalMask:
    """
    Upper-triangular causal mask — prevents attending to future positions.

    Shape : [B, 1, L, L]   (True = masked / ignored by nn.MultiheadAttention)

    Usage
    -----
    mask = TriangularCausalMask(B, L, device=x.device)
    attn_out = attn(q, k, v, attn_mask=mask.mask)
    """

    def __init__(self, B: int, L: int,
                 device: str | torch.device = "cpu"):
        with torch.no_grad():
            self._mask: torch.Tensor = torch.triu(
                torch.ones([B, 1, L, L],
                           dtype=torch.bool, device=device),
                diagonal=1,
            )

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    def __repr__(self) -> str:
        B, _, L, _ = self._mask.shape
        return f"TriangularCausalMask(B={B}, L={L}, device={self._mask.device})"


class ProbMask:
    """
    ProbSparse attention mask for Informer-style Transformers.

    Only the top-k selected queries need masks.

    Args
    ----
    B      : batch size
    H      : number of heads
    L      : full query length
    index  : selected query indices  [B, H, top_k]
    scores : sparse score tensor     [B, H, top_k, L_kv]
    device : target device
    """

    def __init__(
        self,
        B:      int,
        H:      int,
        L:      int,
        index:  torch.Tensor,
        scores: torch.Tensor,
        device: str | torch.device = "cpu",
    ):
        L_kv = scores.shape[-1]
        top_k = index.shape[-1]

        # Full upper-triangular causal mask over (L, L_kv)
        _mask    = torch.ones(L, L_kv, dtype=torch.bool, device=device).triu(1)
        # Expand to [B, H, L, L_kv]
        _mask_ex = _mask.unsqueeze(0).unsqueeze(0).expand(B, H, L, L_kv)

        # FIX: gather selected rows using valid advanced indexing
        # index: [B, H, top_k] → expand to [B, H, top_k, L_kv] for gather
        idx_expanded = index.unsqueeze(-1).expand(B, H, top_k, L_kv)  # [B,H,k,Lkv]
        indicator    = _mask_ex.gather(dim=2, index=idx_expanded)      # [B,H,k,Lkv]

        self._mask: torch.Tensor = indicator.view(scores.shape)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    def __repr__(self) -> str:
        return (f"ProbMask(shape={list(self._mask.shape)}, "
                f"device={self._mask.device})")
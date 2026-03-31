
"""
utils/masking.py
=====================
Attention masks for Transformer components.

Classes
-------
TriangularCausalMask  — standard upper-triangular causal mask
ProbMask              — ProbSparse mask for Informer-style attention
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

    def __init__(self, B: int, L: int, device: str | torch.device = "cpu"):
        with torch.no_grad():
            self._mask: torch.Tensor = torch.triu(
                torch.ones([B, 1, L, L], dtype=torch.bool, device=device),
                diagonal=1,
            )

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class ProbMask:
    """
    ProbSparse attention mask used in Informer-style Transformers.

    Only the top-k selected queries need masks; the rest are discarded.
    The result matches the shape of the sparse score tensor so it can be
    applied directly as an additive mask.

    Args
    ----
    B      : batch size
    H      : number of heads
    L      : full query length  (before top-k selection)
    index  : selected query indices   shape [B, H, top_k]
    scores : sparse score tensor      shape [B, H, top_k, L_kv]
    device : target device string or torch.device
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

        # Full causal mask [L, L_kv]
        _mask    = torch.ones(L, L_kv, dtype=torch.bool, device=device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, L_kv)

        # Select rows for the top-k queries
        b_idx = torch.arange(B, device=device)[:, None, None]
        h_idx = torch.arange(H, device=device)[None, :, None]
        indicator = _mask_ex[b_idx, h_idx, index, :]   # [B, H, top_k, L_kv]

        self._mask: torch.Tensor = indicator.view(scores.shape)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask
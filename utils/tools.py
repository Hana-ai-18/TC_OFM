"""
utils/tools.py  ── v10-fixed
==============================
Utility functions.

FIXES vs original:
  1. EarlyStopping._save_checkpoint: os.makedirs was called with path
     as a file path (e.g. "runs/v10/checkpoint.pth") — makedirs on a
     .pth filename creates a *directory* named "checkpoint.pth".  Fixed
     to call makedirs on os.path.dirname(path) and save to path directly
     (matching the call site in Model/utils.py which already appends the
     filename).
  2. EarlyStopping: self.val_loss_min was np.inf which on NumPy >= 2.0
     triggers a DeprecationWarning ("use math.inf").  Fixed to use
     float("inf").
  3. dic2cuda: original silently skipped non-tensor values.  Non-tensor
     numeric values (int, float, list, np.ndarray) are now converted to
     float32 tensors before moving to device, consistent with how
     env_data is consumed downstream.
  4. StandardScaler: mean/std tensors were registered as plain attributes,
     not buffers — they were not moved with .to(device) and .cuda().
     Fixed by registering as buffers so they follow the module device.
     Note: StandardScaler is not an nn.Module in the original, so the
     fix uses explicit .to() calls with device tracking instead.
  5. relative_to_abs: added input validation — if start_pos is 1-D
     (single sample, no batch dim) it is unsqueezed automatically.
  6. adjust_learning_rate: lradj='cosine' variant added (used by some
     training scripts but missing from the original).
"""
from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager

import numpy as np
import torch


def int_tuple(s: str):
    return tuple(int(i) for i in s.split(","))


def bool_flag(s: str) -> bool:
    if s in ("1", "true", "True", "yes"):
        return True
    if s in ("0", "false", "False", "no"):
        return False
    raise ValueError(f"Expected bool flag, got: {s!r}")


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def dic2cuda(env_data: dict, device=None) -> dict:
    """
    Move every value in env_data to device.

    FIX: non-tensor numeric types (list, np.ndarray, int, float) are
    converted to float32 tensors before being moved to the device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {}
    for key, val in env_data.items():
        if torch.is_tensor(val):
            result[key] = val.float().to(device)
        elif isinstance(val, np.ndarray):
            result[key] = torch.from_numpy(val.astype(np.float32)).to(device)
        elif isinstance(val, (list, tuple)):
            try:
                result[key] = torch.tensor(val, dtype=torch.float32,
                                            device=device)
            except (ValueError, TypeError):
                result[key] = val   # leave non-numeric lists as-is
        elif isinstance(val, (int, float)):
            result[key] = torch.tensor([val], dtype=torch.float32,
                                        device=device)
        else:
            result[key] = val       # unknown type — leave unchanged
    return result


class EarlyStopping:
    """
    Early stopping with checkpoint saving.

    FIX 1: makedirs on dirname(path) not on path itself.
    FIX 2: val_loss_min uses float("inf") (no NumPy deprecation).
    """

    def __init__(self, patience: int = 7, verbose: bool = False,
                 delta: float = 0.0):
        self.patience     = patience
        self.verbose      = verbose
        self.delta        = delta
        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
        self.val_loss_min = float("inf")   # FIX: not np.inf

    def __call__(self, val_loss: float, model_state: dict, path: str):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model_state, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model_state, path)
            self.counter = 0

    def _save_checkpoint(self, val_loss: float, model_state: dict,
                         path: str):
        if self.verbose:
            print(f"Val loss decreased "
                  f"({self.val_loss_min:.6f} → {val_loss:.6f}). Saving…")
        # FIX: makedirs on the *directory*, not the file path
        save_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model_state, path)
        self.val_loss_min = val_loss


class StandardScaler:
    """
    Normalise / denormalise TC data (lon, lat, pres, wnd).

    FIX: mean/std stored as plain tensors but .to(device) is now called
    explicitly in transform/inverse_transform rather than relying on
    accidental device inference.
    """

    def __init__(self):
        self.mean = torch.tensor([1316.42, 218.44, 979.47,  28.18])
        self.std  = torch.tensor([ 145.29,  88.04,  23.42,  13.26])

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        m = self.mean.to(data.device)
        s = self.std.to(data.device)
        return (data - m) / s

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        m = self.mean.to(data.device)
        s = self.std.to(data.device)
        return data * s + m


def relative_to_abs(rel_traj: torch.Tensor,
                    start_pos: torch.Tensor) -> torch.Tensor:
    """
    Convert displacement sequence to absolute coordinates.

    rel_traj  : [Time, Batch, 2]
    start_pos : [Batch, 2]  or  [2]  (FIX: auto-unsqueeze if 1-D)
    """
    if start_pos.dim() == 1:
        start_pos = start_pos.unsqueeze(0)   # → [1, 2]
    return torch.cumsum(rel_traj, dim=0) + start_pos.unsqueeze(0)


def adjust_learning_rate(optimizer, epoch: int, args) -> None:
    """Adjust LR by epoch according to args.lradj strategy."""
    lradj = getattr(args, "lradj", "type1")

    if lradj == "type1":
        lr = args.g_learning_rate * (0.5 ** ((epoch - 1) // 1))
        lr_adjust = {epoch: lr}
    elif lradj == "type2":
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8,
        }
    elif lradj == "cosine":
        # FIX: cosine decay variant (used by some run scripts)
        total = getattr(args, "num_epochs", 80)
        lr = args.g_learning_rate * (
            1 + math.cos(math.pi * epoch / total)) / 2
        lr_adjust = {epoch: max(lr, 1e-8)}
    else:
        lr_adjust = {}

    if epoch in lr_adjust:
        new_lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        print(f"  LR → {new_lr:.2e}  (epoch {epoch})")


@contextmanager
def timeit(msg: str, should_time: bool = True):
    if should_time and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    yield
    if should_time:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"{msg}: {(time.time() - t0) * 1000.0:.2f} ms")


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
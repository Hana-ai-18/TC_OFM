# """
# Model/data/loader.py  ── v10
# ============================
# Smart data loader với persistent_workers fix cho Kaggle.
# """
# from __future__ import annotations

# import os
# import sys

# _here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# if _here not in sys.path:
#     sys.path.insert(0, _here)

# from torch.utils.data import DataLoader
# from Model.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate


# def _find_tcnd_root(path: str) -> str:
#     path  = os.path.abspath(path)
#     check = path
#     for _ in range(6):
#         if os.path.exists(os.path.join(check, "Data1d")):
#             return check
#         parent = os.path.dirname(check)
#         if parent == check:
#             break
#         check = parent
#     for sub in ("TCND_vn", "tcnd_vn", "data", "TCND"):
#         candidate = os.path.join(path, sub)
#         if os.path.exists(os.path.join(candidate, "Data1d")):
#             return candidate
#     return path


# def data_loader(args, path_config, test=False, test_year=None, batch_size=None):
#     if isinstance(path_config, dict):
#         raw_path  = path_config.get("root", "")
#         dset_type = path_config.get("type", "test" if test else "train")
#     else:
#         raw_path  = str(path_config)
#         dset_type = "test" if test else "train"

#     root = _find_tcnd_root(raw_path)
#     print(f"DataLoader | root={root} | type={dset_type} | year={test_year}")

#     dataset = TrajectoryDataset(
#         data_dir=root, obs_len=args.obs_len, pred_len=args.pred_len,
#         skip=getattr(args, "skip", 1), threshold=getattr(args, "threshold", 0.002),
#         min_ped=getattr(args, "min_ped", 1), delim=getattr(args, "delim", " "),
#         other_modal=getattr(args, "other_modal", "gph"),
#         test_year=test_year, type=dset_type, is_test=test,
#     )

#     num_workers      = getattr(args, "num_workers", 0)
#     use_persistent   = num_workers > 0
#     prefetch         = 2 if num_workers > 0 else None

#     loader = DataLoader(
#         dataset,
#         batch_size         = batch_size or args.batch_size,
#         shuffle            = not test,
#         collate_fn         = seq_collate,
#         num_workers        = num_workers,
#         persistent_workers = use_persistent,
#         prefetch_factor    = prefetch,
#         drop_last          = False,
#         pin_memory         = _cuda_available() and num_workers > 0,
#     )
#     print(f"  {len(dataset)} sequences  (workers={num_workers})")
#     return dataset, loader


# def _cuda_available():
#     try:
#         import torch
#         return torch.cuda.is_available()
#     except ImportError:
#         return False

"""
Model/data/loader.py  ── v11
============================
Data loader cho inference / test.

BUG-3 FIX: import TrajectoryDataset từ trajectoriesWithMe_unet_training
(v11) thay vì từ file cũ. Dùng split= kwarg nhất quán với v11 dataset.

Các fix từ loader_training cũng được áp dụng:
  - prefetch_factor / persistent_workers guarded khi num_workers=0
  - pin_memory guard CUDA + num_workers > 0
  - _find_tcnd_root check path trước
  - worker_init_fn cho reproducibility
"""
from __future__ import annotations

import inspect
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _here not in sys.path:
    sys.path.insert(0, _here)

# BUG-3 FIX: dùng v11 training dataset (split= kwarg) cho cả inference
from Model.data.trajectoriesWithMe_unet_training import (
    TrajectoryDataset,
    seq_collate,
)


def _find_tcnd_root(path: str) -> str:
    path = os.path.abspath(path)
    if os.path.exists(os.path.join(path, "Data1d")):
        return path
    check = os.path.dirname(path)
    for _ in range(6):
        if os.path.exists(os.path.join(check, "Data1d")):
            return check
        parent = os.path.dirname(check)
        if parent == check:
            break
        check = parent
    for sub in ("TCND_vn", "tcnd_vn", "data", "TCND"):
        candidate = os.path.join(path, sub)
        if os.path.exists(os.path.join(candidate, "Data1d")):
            return candidate
    return path


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def data_loader(
    args,
    path_config,
    test: bool = False,
    test_year=None,
    batch_size: int | None = None,
):
    """
    Unified data loader for train / val / test splits.
    BUG-3 FIX: always uses split= kwarg for v11 TrajectoryDataset.
    """
    if isinstance(path_config, dict):
        raw_path  = path_config.get("root", "")
        dset_type = path_config.get("type", "test" if test else "train")
    else:
        raw_path  = str(path_config)
        dset_type = "test" if test else "train"

    root = _find_tcnd_root(raw_path)
    print(f"DataLoader | root={root} | type={dset_type} | year={test_year}")

    ds_sig    = inspect.signature(TrajectoryDataset.__init__)
    ds_kwargs: dict = dict(
        data_dir    = root,
        obs_len     = args.obs_len,
        pred_len    = args.pred_len,
        skip        = getattr(args, "skip",        1),
        threshold   = getattr(args, "threshold",   0.002),
        min_ped     = getattr(args, "min_ped",     1),
        delim       = getattr(args, "delim",       " "),
        other_modal = getattr(args, "other_modal", "gph"),
        is_test     = test,
    )

    # v11: prefer split=, fallback to type= for legacy datasets
    if "split" in ds_sig.parameters:
        ds_kwargs["split"] = dset_type
    elif "type" in ds_sig.parameters:
        ds_kwargs["type"] = dset_type
    else:
        ds_kwargs["split"] = dset_type

    if "test_year" in ds_sig.parameters and test_year is not None:
        ds_kwargs["test_year"] = test_year

    dataset = TrajectoryDataset(**ds_kwargs)

    num_workers    = getattr(args, "num_workers", 0)
    use_persistent = num_workers > 0
    prefetch       = 2 if num_workers > 0 else None
    use_pin_memory = _cuda_available() and num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size         = batch_size or args.batch_size,
        shuffle            = not test,
        collate_fn         = seq_collate,
        num_workers        = num_workers,
        persistent_workers = use_persistent,
        prefetch_factor    = prefetch,
        drop_last          = False,
        pin_memory         = use_pin_memory,
        worker_init_fn     = _worker_init_fn if num_workers > 0 else None,
    )
    print(f"  {len(dataset)} sequences  (workers={num_workers})")
    return dataset, loader
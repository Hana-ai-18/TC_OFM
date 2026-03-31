
# """
# Model/data/loader_training.py  ── v10-fixed
# ============================================
# Data loader for training.

# FIXES vs original:
#   1. REMOVED _normalize_data() — Data1d files are already normalised
#      (confirmed from data screenshot: LONG=-9.5, LAT=2.9, etc.)
#      Re-normalising produced garbage values.
#   2. seq_collate_training env_data moved to index 13 (was 15) to match
#      trajectoriesWithMe_unet_training.seq_collate and
#      VelocityField._context which reads batch_list[13].
#   3. Removed unused TCTrainingDataset class entirely — TrajectoryDataset
#      from trajectoriesWithMe_unet_training is the canonical dataset.
#      Using two dataset classes for the same data caused the normalisation
#      mismatch in the first place.

# ADDITIONAL FIXES (v10-fixed-2):
#   4. prefetch_factor must be None (not 2) when num_workers=0 — PyTorch
#      raises ValueError: "prefetch_factor option could only be specified
#      in multiprocessing" otherwise.
#   5. persistent_workers must be False when num_workers=0 — PyTorch raises
#      ValueError: "persistent_workers option needs num_workers > 0".
#      Both 4 and 5 were already guarded by use_persistent but prefetch was
#      set independently and could still be 2 with num_workers=0.
#   6. The 'type' parameter name in TrajectoryDataset() shadowed Python's
#      builtin. Renamed to 'split' in the call-site kwarg; dataset classes
#      that accept the old name still work because we pass it as a keyword.
#   7. _find_tcnd_root walked up the tree but never checked the path itself
#      first, so passing the TCND root directly returned the wrong folder.
#      Fixed: check path before walking up AND before scanning subdirs.
#   8. pin_memory guard: pin_memory=True with num_workers=0 is a no-op and
#      generates a UserWarning on some PyTorch versions. Only enable when
#      both CUDA is available AND num_workers > 0.
#   9. test_year forwarded to TrajectoryDataset only when the dataset
#      constructor actually accepts it (checked via inspect), avoiding
#      TypeError on dataset classes that don't have that parameter.
# """
# from __future__ import annotations

# import inspect
# import os
# from torch.utils.data import DataLoader

# from Model.data.trajectoriesWithMe_unet_training import (
#     TrajectoryDataset,
#     seq_collate,
# )


# def _find_tcnd_root(path: str) -> str:
#     """
#     Walk up the directory tree to find the folder that contains Data1d/.

#     FIX: check `path` itself before ascending, so passing the root
#     directly works without an off-by-one miss.
#     """
#     path = os.path.abspath(path)

#     # FIX 7a: check the given path before walking upward
#     if os.path.exists(os.path.join(path, "Data1d")):
#         return path

#     # Walk upward
#     check = os.path.dirname(path)
#     for _ in range(6):
#         if os.path.exists(os.path.join(check, "Data1d")):
#             return check
#         parent = os.path.dirname(check)
#         if parent == check:
#             break
#         check = parent

#     # FIX 7b: scan well-known sub-directory names under the original path
#     for sub in ("TCND_vn", "tcnd_vn", "data", "TCND"):
#         candidate = os.path.join(path, sub)
#         if os.path.exists(os.path.join(candidate, "Data1d")):
#             return candidate

#     return path


# def data_loader(
#     args,
#     path_config,
#     test: bool = False,
#     test_year=None,
#     batch_size: int | None = None,
# ):
#     """
#     Unified data loader for train / val / test splits.

#     path_config : str  or  {"root": ..., "type": "train"|"val"|"test"}
#     """
#     if isinstance(path_config, dict):
#         raw_path  = path_config.get("root", "")
#         dset_type = path_config.get("type", "test" if test else "train")
#     else:
#         raw_path  = str(path_config)
#         dset_type = "test" if test else "train"

#     root = _find_tcnd_root(raw_path)
#     print(f"DataLoader | root={root} | type={dset_type} | year={test_year}")

#     # FIX 9: only forward test_year if the dataset constructor accepts it
#     ds_sig    = inspect.signature(TrajectoryDataset.__init__)
#     ds_kwargs = dict(
#         data_dir    = root,
#         obs_len     = args.obs_len,
#         pred_len    = args.pred_len,
#         skip        = getattr(args, "skip",        1),
#         threshold   = getattr(args, "threshold",   0.002),
#         min_ped     = getattr(args, "min_ped",     1),
#         delim       = getattr(args, "delim",       " "),
#         other_modal = getattr(args, "other_modal", "gph"),
#         # FIX 6: avoid shadowing the builtin 'type'; pass as keyword
#         split       = dset_type,
#         is_test     = test,
#     )
#     # Some older dataset versions use 'type' instead of 'split'
#     if "split" not in ds_sig.parameters and "type" in ds_sig.parameters:
#         ds_kwargs["type"] = ds_kwargs.pop("split")

#     if "test_year" in ds_sig.parameters and test_year is not None:
#         ds_kwargs["test_year"] = test_year

#     dataset = TrajectoryDataset(**ds_kwargs)

#     num_workers = getattr(args, "num_workers", 0)

#     # FIX 4 + 5: both persistent_workers and prefetch_factor require
#     # num_workers > 0; guard them together under a single condition.
#     use_persistent = num_workers > 0
#     prefetch       = 2 if num_workers > 0 else None   # was always 2

#     # FIX 8: pin_memory is only useful (and warning-free) when CUDA is
#     # available AND a background worker is actually copying tensors.
#     use_pin_memory = _cuda_available() and num_workers > 0

#     loader = DataLoader(
#         dataset,
#         batch_size         = batch_size or args.batch_size,
#         shuffle            = not test,
#         collate_fn         = seq_collate,   # canonical collate — env at index 13
#         num_workers        = num_workers,
#         persistent_workers = use_persistent,
#         prefetch_factor    = prefetch,
#         drop_last          = False,
#         pin_memory         = use_pin_memory,
#     )
#     print(f"  {len(dataset)} sequences  (workers={num_workers})")
#     return dataset, loader


# def _cuda_available() -> bool:
#     try:
#         import torch
#         return torch.cuda.is_available()
#     except ImportError:
#         return False

"""
Model/data/loader_training.py  ── v11
======================================
Data loader for training.

FIXES từ v10-fixed-2 (giữ nguyên):
  1. REMOVED _normalize_data() — Data1d files already normalised.
  2. seq_collate env_data tại index 13.
  3. Removed unused TCTrainingDataset.
  4. prefetch_factor=None khi num_workers=0.
  5. persistent_workers=False khi num_workers=0.
  6. 'type' → 'split' kwarg cho TrajectoryDataset.
  7. _find_tcnd_root check path trước khi walk up.
  8. pin_memory guard: chỉ True khi CUDA available AND num_workers > 0.
  9. test_year forwarded chỉ khi dataset constructor chấp nhận.

FIXES mới v11:
  10. BUG-3 FIX: import từ trajectoriesWithMe_unet_training (v11 dataset).
      Dùng split= kwarg (không phải type=) nhất quán.
  11. Thêm worker_init_fn để seed numpy/random per worker → reproducible.
  12. Thêm drop_last=True cho train split để tránh batch size 1 ở cuối.
"""
from __future__ import annotations

import inspect
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

# BUG-3 FIX: import từ v11 training dataset (dùng split= kwarg)
from Model.data.trajectoriesWithMe_unet_training import (
    TrajectoryDataset,
    seq_collate,
)


def _find_tcnd_root(path: str) -> str:
    """
    Walk up the directory tree to find the folder that contains Data1d/.
    FIX-7: check `path` itself trước khi ascend.
    """
    path = os.path.abspath(path)

    # Check given path first
    if os.path.exists(os.path.join(path, "Data1d")):
        return path

    # Walk upward
    check = os.path.dirname(path)
    for _ in range(6):
        if os.path.exists(os.path.join(check, "Data1d")):
            return check
        parent = os.path.dirname(check)
        if parent == check:
            break
        check = parent

    # Scan well-known sub-directory names
    for sub in ("TCND_vn", "tcnd_vn", "data", "TCND"):
        candidate = os.path.join(path, sub)
        if os.path.exists(os.path.join(candidate, "Data1d")):
            return candidate

    return path


def _worker_init_fn(worker_id: int) -> None:
    """Seed numpy/random per DataLoader worker for reproducibility."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def _cuda_available() -> bool:
    try:
        import torch as _torch
        return _torch.cuda.is_available()
    except ImportError:
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

    path_config : str  or  {"root": ..., "type": "train"|"val"|"test"}

    BUG-3 FIX: TrajectoryDataset v11 uses split= kwarg; this loader
    always passes split=dset_type (never the old type= kwarg) except when
    the dataset signature only has 'type' (legacy fallback).
    """
    if isinstance(path_config, dict):
        raw_path  = path_config.get("root", "")
        dset_type = path_config.get("type", "test" if test else "train")
    else:
        raw_path  = str(path_config)
        dset_type = "test" if test else "train"

    root = _find_tcnd_root(raw_path)
    print(f"DataLoader | root={root} | type={dset_type} | year={test_year}")

    # FIX-9: only forward test_year if constructor accepts it
    # FIX-6 / BUG-3: use split= kwarg preferentially
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

    # v11 dataset uses 'split', legacy uses 'type'
    if "split" in ds_sig.parameters:
        ds_kwargs["split"] = dset_type
    elif "type" in ds_sig.parameters:
        ds_kwargs["type"] = dset_type
    else:
        ds_kwargs["split"] = dset_type  # default to split

    if "test_year" in ds_sig.parameters and test_year is not None:
        ds_kwargs["test_year"] = test_year

    dataset = TrajectoryDataset(**ds_kwargs)

    num_workers = getattr(args, "num_workers", 0)

    # FIX-4+5: guard both persistent_workers and prefetch_factor
    use_persistent = num_workers > 0
    prefetch       = 2 if num_workers > 0 else None

    # FIX-8: pin_memory only useful with CUDA + background workers
    use_pin_memory = _cuda_available() and num_workers > 0

    # drop_last=True for train split to avoid single-sample batches
    # that break BatchNorm in FNO layers
    is_train   = dset_type == "train" and not test
    drop_last  = is_train and len(dataset) > (batch_size or args.batch_size)

    loader = DataLoader(
        dataset,
        batch_size         = batch_size or args.batch_size,
        shuffle            = not test,
        collate_fn         = seq_collate,
        num_workers        = num_workers,
        persistent_workers = use_persistent,
        prefetch_factor    = prefetch,
        drop_last          = drop_last,
        pin_memory         = use_pin_memory,
        worker_init_fn     = _worker_init_fn if num_workers > 0 else None,
    )
    print(f"  {len(dataset)} sequences  "
          f"(workers={num_workers}, drop_last={drop_last})")
    return dataset, loader
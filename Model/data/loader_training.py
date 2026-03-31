# """
# Model/data/loader_training.py - Data loader for training
# Handles Vietnam TC data structure
# """
# import os
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader


# class TCTrainingDataset(Dataset):
#     """
#     TC Dataset for training
#     Handles normalized data for training
#     """
#     def __init__(self, data_root, obs_len=8, pred_len=4):
#         self.data_root = data_root
#         self.obs_len = obs_len
#         self.pred_len = pred_len
        
#         self.sequences = []
#         self._load_data()
    
#     def _load_data(self):
#         """Load and normalize training data"""
#         # data1d_dir = os.path.join(self.data_root, 'Data1d')
#         data1d_dir = self.data_root 
        
#         if not os.path.exists(data1d_dir):
#             print(f"Warning: {data1d_dir} not found")
#             return
        
#         txt_files = [f for f in os.listdir(data1d_dir) if f.endswith('.txt')]
        
#         for txt_file in txt_files:
#             year_str = txt_file.split('_')[0]
#             self._process_file(os.path.join(data1d_dir, txt_file), year_str)
        
#         print(f"Loaded {len(self.sequences)} training sequences")
    
#     def _normalize_data(self, lon, lat, pres, wind):
#         """Normalize according to paper's formula"""
#         # LONG normalization: (LONG - 1800) / 50
#         lon_norm = (lon - 1800.0) / 50.0
        
#         # LAT normalization: LAT / 50
#         lat_norm = lat / 50.0
        
#         # PRES normalization: (PRES - 960) / 50
#         pres_norm = (pres - 960.0) / 50.0
        
#         # WND normalization: (WND - 40) / 25
#         wind_norm = (wind - 40.0) / 25.0
        
#         return lon_norm, lat_norm, pres_norm, wind_norm
    
#     def _process_file(self, file_path, year):
#         """Process training file with normalization"""
#         try:
#             with open(file_path, 'r') as f:
#                 lines = f.readlines()
            
#             storm_name = None
#             tc_data = []
            
#             for line in lines:
#                 line = line.strip()
#                 if not line or line.startswith('#'):
#                     continue
                
#                 parts = line.split()
#                 if len(parts) < 5:
#                     continue
                
#                 try:
#                     timestamp = parts[0]
#                     lon = float(parts[1])
#                     lat = float(parts[2])
#                     pres = float(parts[3])
#                     wind = float(parts[4])
                    
#                     if storm_name is None and len(parts) > 5:
#                         storm_name = parts[5]
                    
#                     # Normalize
#                     lon_norm, lat_norm, pres_norm, wind_norm = self._normalize_data(
#                         lon, lat, pres, wind
#                     )
                    
#                     tc_data.append({
#                         'timestamp': timestamp,
#                         'lon': lon_norm,
#                         'lat': lat_norm,
#                         'pres': pres_norm,
#                         'wind': wind_norm,
#                         'year': year,
#                         'storm_name': storm_name or 'UNKNOWN'
#                     })
#                 except (ValueError, IndexError):
#                     continue
            
#             # Create sequences
#             total_len = self.obs_len + self.pred_len
#             for i in range(len(tc_data) - total_len + 1):
#                 seq = tc_data[i:i + total_len]
#                 self.sequences.append(seq)
        
#         except Exception as e:
#             print(f"Error processing {file_path}: {e}")
    
#     def __len__(self):
#         return len(self.sequences)
    
#     def __getitem__(self, idx):
#         seq = self.sequences[idx]
        
#         obs_seq = seq[:self.obs_len]
#         pred_seq = seq[self.obs_len:]
        
#         # Extract normalized features
#         obs_traj = torch.tensor([[d['lon'], d['lat']] for d in obs_seq], dtype=torch.float32)
#         pred_traj = torch.tensor([[d['lon'], d['lat']] for d in pred_seq], dtype=torch.float32)
        
#         obs_Me = torch.tensor([[d['pres'], d['wind']] for d in obs_seq], dtype=torch.float32)
#         pred_Me = torch.tensor([[d['pres'], d['wind']] for d in pred_seq], dtype=torch.float32)
        
#         # Load 3D and env data
#         year = obs_seq[0]['year']
#         storm_name = obs_seq[0]['storm_name']
        
#         image_obs = self._load_3d_data(year, storm_name, obs_seq)
#         env_data = self._load_env_data(year, storm_name, obs_seq)
        
#         return (
#             obs_traj, pred_traj, None, None, None, None, None,
#             obs_Me, pred_Me, None, None, image_obs, None, env_data
#         )
    
#     def _load_3d_data(self, year, storm_name, obs_seq):
#         """Load 3D data"""
#         try:
#             data3d_dir = os.path.join(self.data_root.replace('Data1d', 'Data3d'), 
#                                      year, storm_name)
            
#             images = []
#             for obs in obs_seq:
#                 timestamp = obs['timestamp']
#                 npy_path = os.path.join(data3d_dir, f"{timestamp}.npy")
                
#                 if os.path.exists(npy_path):
#                     img = np.load(npy_path)
#                     images.append(img)
#                 else:
#                     images.append(np.zeros((64, 64)))
            
#             images = np.stack(images)
#             return torch.from_numpy(images).float().unsqueeze(0).unsqueeze(0)
        
#         except:
#             return torch.zeros(1, 1, self.obs_len, 64, 64)
    
#     def _load_env_data(self, year, storm_name, obs_seq):
#         """Load environment data"""
#         try:
#             env_dir = os.path.join(self.data_root.replace('Data1d', 'Env'), 
#                                   year, storm_name)
            
#             env_data = {
#                 'wind': torch.zeros(self.obs_len, 1),
#                 'month': torch.zeros(self.obs_len, 12),
#                 'move_velocity': torch.zeros(self.obs_len, 1)
#             }
            
#             for i, obs in enumerate(obs_seq):
#                 timestamp = obs['timestamp']
#                 npy_path = os.path.join(env_dir, f"{timestamp}.npy")
                
#                 if os.path.exists(npy_path):
#                     env = np.load(npy_path, allow_pickle=True).item()
                    
#                     if 'wind' in env:
#                         env_data['wind'][i] = torch.tensor([env['wind']])
#                     if 'month' in env:
#                         month_onehot = torch.zeros(12)
#                         month_onehot[env['month'] - 1] = 1
#                         env_data['month'][i] = month_onehot
            
#             return env_data
        
#         except:
#             return {
#                 'wind': torch.zeros(self.obs_len, 1),
#                 'month': torch.zeros(self.obs_len, 12),
#                 'move_velocity': torch.zeros(self.obs_len, 1)
#             }


# def seq_collate_training(data):
#     """Collate for training"""
#     batch_size = len(data)
    
#     obs_traj_list = []
#     pred_traj_list = []
#     obs_Me_list = []
#     pred_Me_list = []
#     image_obs_list = []
#     env_data_list = []
    
#     for item in data:
#         obs_traj_list.append(item[0])
#         pred_traj_list.append(item[1])
#         obs_Me_list.append(item[7])
#         pred_Me_list.append(item[8])
#         image_obs_list.append(item[11])
#         env_data_list.append(item[13])
    
#     obs_traj = torch.stack(obs_traj_list, dim=1)
#     pred_traj = torch.stack(pred_traj_list, dim=1)
#     obs_Me = torch.stack(obs_Me_list, dim=1)
#     pred_Me = torch.stack(pred_Me_list, dim=1)
    
#     image_obs = torch.cat(image_obs_list, dim=0)
    
#     env_data = {}
#     for key in env_data_list[0].keys():
#         env_data[key] = torch.stack([d[key] for d in env_data_list], dim=0)
    
#     return [
#         obs_traj,      # 0
#         pred_traj,     # 1
#         None,          # 2
#         None,          # 3
#         None,          # 4
#         None,          # 5
#         None,          # 6
#         obs_Me,        # 7
#         pred_Me,       # 8
#         None,          # 9
#         None,          # 10
#         image_obs,     # 11
#         None,          # 12
#         None,          # 13
#         None,          # 14
#         env_data       # 15 <--- Đưa về index 15
#     ]


# def data_loader(args, path, test=False, batch_size=None):
#     """Create training data loader"""
#     dataset = TCTrainingDataset(
#         data_root=path['root'],
#         obs_len=args.obs_len,
#         pred_len=args.pred_len
#     )
    
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size or args.batch_size,
#         shuffle=not test,
#         collate_fn=seq_collate_training,
#         num_workers=4,
#         pin_memory=True, 
#         drop_last=True
#     )
    
#     return dataset, loader
"""
Model/data/loader_training.py  ── v10-fixed
============================================
Data loader for training.

FIXES vs original:
  1. REMOVED _normalize_data() — Data1d files are already normalised
     (confirmed from data screenshot: LONG=-9.5, LAT=2.9, etc.)
     Re-normalising produced garbage values.
  2. seq_collate_training env_data moved to index 13 (was 15) to match
     trajectoriesWithMe_unet_training.seq_collate and
     VelocityField._context which reads batch_list[13].
  3. Removed unused TCTrainingDataset class entirely — TrajectoryDataset
     from trajectoriesWithMe_unet_training is the canonical dataset.
     Using two dataset classes for the same data caused the normalisation
     mismatch in the first place.

ADDITIONAL FIXES (v10-fixed-2):
  4. prefetch_factor must be None (not 2) when num_workers=0 — PyTorch
     raises ValueError: "prefetch_factor option could only be specified
     in multiprocessing" otherwise.
  5. persistent_workers must be False when num_workers=0 — PyTorch raises
     ValueError: "persistent_workers option needs num_workers > 0".
     Both 4 and 5 were already guarded by use_persistent but prefetch was
     set independently and could still be 2 with num_workers=0.
  6. The 'type' parameter name in TrajectoryDataset() shadowed Python's
     builtin. Renamed to 'split' in the call-site kwarg; dataset classes
     that accept the old name still work because we pass it as a keyword.
  7. _find_tcnd_root walked up the tree but never checked the path itself
     first, so passing the TCND root directly returned the wrong folder.
     Fixed: check path before walking up AND before scanning subdirs.
  8. pin_memory guard: pin_memory=True with num_workers=0 is a no-op and
     generates a UserWarning on some PyTorch versions. Only enable when
     both CUDA is available AND num_workers > 0.
  9. test_year forwarded to TrajectoryDataset only when the dataset
     constructor actually accepts it (checked via inspect), avoiding
     TypeError on dataset classes that don't have that parameter.
"""
from __future__ import annotations

import inspect
import os
from torch.utils.data import DataLoader

from Model.data.trajectoriesWithMe_unet_training import (
    TrajectoryDataset,
    seq_collate,
)


def _find_tcnd_root(path: str) -> str:
    """
    Walk up the directory tree to find the folder that contains Data1d/.

    FIX: check `path` itself before ascending, so passing the root
    directly works without an off-by-one miss.
    """
    path = os.path.abspath(path)

    # FIX 7a: check the given path before walking upward
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

    # FIX 7b: scan well-known sub-directory names under the original path
    for sub in ("TCND_vn", "tcnd_vn", "data", "TCND"):
        candidate = os.path.join(path, sub)
        if os.path.exists(os.path.join(candidate, "Data1d")):
            return candidate

    return path


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
    """
    if isinstance(path_config, dict):
        raw_path  = path_config.get("root", "")
        dset_type = path_config.get("type", "test" if test else "train")
    else:
        raw_path  = str(path_config)
        dset_type = "test" if test else "train"

    root = _find_tcnd_root(raw_path)
    print(f"DataLoader | root={root} | type={dset_type} | year={test_year}")

    # FIX 9: only forward test_year if the dataset constructor accepts it
    ds_sig    = inspect.signature(TrajectoryDataset.__init__)
    ds_kwargs = dict(
        data_dir    = root,
        obs_len     = args.obs_len,
        pred_len    = args.pred_len,
        skip        = getattr(args, "skip",        1),
        threshold   = getattr(args, "threshold",   0.002),
        min_ped     = getattr(args, "min_ped",     1),
        delim       = getattr(args, "delim",       " "),
        other_modal = getattr(args, "other_modal", "gph"),
        # FIX 6: avoid shadowing the builtin 'type'; pass as keyword
        split       = dset_type,
        is_test     = test,
    )
    # Some older dataset versions use 'type' instead of 'split'
    if "split" not in ds_sig.parameters and "type" in ds_sig.parameters:
        ds_kwargs["type"] = ds_kwargs.pop("split")

    if "test_year" in ds_sig.parameters and test_year is not None:
        ds_kwargs["test_year"] = test_year

    dataset = TrajectoryDataset(**ds_kwargs)

    num_workers = getattr(args, "num_workers", 0)

    # FIX 4 + 5: both persistent_workers and prefetch_factor require
    # num_workers > 0; guard them together under a single condition.
    use_persistent = num_workers > 0
    prefetch       = 2 if num_workers > 0 else None   # was always 2

    # FIX 8: pin_memory is only useful (and warning-free) when CUDA is
    # available AND a background worker is actually copying tensors.
    use_pin_memory = _cuda_available() and num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size         = batch_size or args.batch_size,
        shuffle            = not test,
        collate_fn         = seq_collate,   # canonical collate — env at index 13
        num_workers        = num_workers,
        persistent_workers = use_persistent,
        prefetch_factor    = prefetch,
        drop_last          = False,
        pin_memory         = use_pin_memory,
    )
    print(f"  {len(dataset)} sequences  (workers={num_workers})")
    return dataset, loader


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
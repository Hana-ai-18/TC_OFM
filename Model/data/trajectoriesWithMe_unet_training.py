# """
# Model/data/trajectoriesWithMe_unet_training.py  ── v18
# ======================================================
# TC trajectory dataset — TRAINING VERSION.

# FIXES vs v17:

#   FIX-DATA-15 [CRITICAL] GPH500 = 0 dù Env_data .npy tồn tại.

#               Nguyên nhân gốc (pipeline cũ — build_env_data_scs_v10.py):
#                 _parse_data3d() lưu key "gph500_mean_n" và "gph500_center_n"
#                 (đã pre-normalized bằng GPH500_MEAN_NORM=5900, STD=200 → range
#                 rất khác với raw dam). Nhưng env_data_processing() và
#                 build_env_features_one_step() đọc key "gph500_mean" và
#                 "gph500_center" → KHÔNG TÌM THẤY → gph500 = 0.

#               Fix:
#                 - _load_env_npy(): sau khi load dict từ .npy, nếu key
#                   "gph500_mean" không có nhưng "gph500_mean_n" có → copy sang
#                   đúng key, đánh dấu cờ "gph500_already_normed" = True.
#                 - env_data_processing(): đọc cờ đó, KHÔNG apply sentinel guard
#                   [25, 95] (guard đó dành cho raw dam, không phải pre-normed).
#                   Truyền thẳng value qua.
#                 - build_env_features_one_step() (env_net_transformer v19):
#                   nếu cờ "gph500_already_normed" set → bỏ qua z-score lần 2,
#                   chỉ clip [-5, 5].

#   FIX-DATA-16 CSV fallback: column mapping từ all_storms_final.csv đã dùng
#               "env_gph500_mean" (raw dam) → truyền thẳng vào env_data_processing
#               với sentinel guard [25, 95] — đây là ĐÚNG cho CSV path vì CSV
#               lưu raw dam. Giữ nguyên, không cần thay.

#   FIX-DATA-17 env_data_processing(): cờ "gph500_already_normed" được bảo toàn
#               qua cleaned dict để downstream đọc được.

# Kept from v17:
#   FIX-DATA-12 CSV fallback auto-discover all_storms_final.csv
#   FIX-DATA-13 Warning message phân biệt sentinel vs env files missing
#   FIX-DATA-10 DATA3D_MEAN/STD ch0 gph500 dùng đơn vị /380 dam
#   FIX-DATA-11 env_data_processing gph500 sentinel guard [25.0, 95.0]
#               (chỉ áp dụng khi NOT already_normed)
#   FIX-DATA-5  ch3/ch4 u500_center/v500_center corrected mean/std
#   FIX-DATA-4  Zero-sentinel only ch0 (gph500_mean)
#   FIX-CACHE-1 Stable env cache key
#   FIX-DATA-1/2/3 Large sentinel, SST fill, gph guard
# """
# from __future__ import annotations

# import logging
# import math
# import os

# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset

# try:
#     import cv2
#     HAS_CV2 = True
# except ImportError:
#     HAS_CV2 = False

# try:
#     import netCDF4 as nc
#     HAS_NC = True
# except ImportError:
#     HAS_NC = False

# from Model.env_net_transformer_gphsplit import (
#     bearing_to_scs_center_onehot, dist_to_scs_boundary_onehot,
#     delta_velocity_onehot, intensity_class_onehot,
#     build_env_features_one_step, feat_to_tensor, ENV_FEATURE_DIMS,
# )

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# DATA3D_H  = 81
# DATA3D_W  = 81
# DATA3D_CH = 13

# # ── DATA3D normalisation constants ────────────────────────────────────────────
# DATA3D_MEAN = np.array([
#     33.64,      # ch0  gph500_mean  FIX-DATA-10: /380 dam (was 1287.66 in v15)
#     5843.14,    # ch1  u500_mean    ✓ confirmed
#     1482.47,    # ch2  v500_mean    ✓ confirmed
#     5930.27,    # ch3  u500_center  FIX-DATA-5 (v14)
#     1622.27,    # ch4  v500_center  FIX-DATA-5 (v14)
#     0.27,       # ch5
#     -0.34,      # ch6
#     -0.86,      # ch7
#     0.25,       # ch8
#     1.76,       # ch9
#     1.34,       # ch10
#     0.94,       # ch11
#     300.95,     # ch12 SST
# ], dtype=np.float32)

# DATA3D_STD = np.array([
#     7.08,       # ch0  gph500_mean  FIX-DATA-10
#     50.55,      # ch1  u500_mean    ✓
#     29.42,      # ch2  v500_mean    ✓
#     1025.26,    # ch3  u500_center  FIX-DATA-5 (v14)
#     1600.32,    # ch4  v500_center  FIX-DATA-5 (v14)
#     4.73,
#     2.98,
#     2.75,
#     5.37,
#     2.29,
#     2.21,
#     2.68,
#     3.05,       # ch12 SST
# ], dtype=np.float32)

# # ── Sentinel thresholds ───────────────────────────────────────────────────────
# _DATA3D_SENTINEL_LARGE         = 20000.0
# _DATA3D_SENTINEL_ZERO_CHANNELS = {0}

# # Guard cho raw dam (build_env_data lưu CSV path → truyền raw dam)
# _DATA3D_GPH_VALID_MIN = 25.0
# _DATA3D_GPH_VALID_MAX = 95.0

# _DATA3D_SST_CHANNEL = 12
# _SST_VALID_MIN = 270.0
# _SST_FILL_K    = 298.0

# _MOVE_VEL_NORM = 1219.84

# # ── Key mapping: build_env_data_scs_v10.py lưu suffix "_n" ───────────────────
# # FIX-DATA-15: Map từ key cũ (có _n) sang key chuẩn mà downstream đọc.
# _NPY_KEY_REMAP = {
#     "gph500_mean_n"   : "gph500_mean",
#     "gph500_center_n" : "gph500_center",
# }
# # u500/v500 trong .npy cũ đã normalize /U500_NORM → [-1,1].
# # build_env_features expects float 0-1 (boolean-style). Clip [0,1] là OK.
# _NPY_U500_KEY_REMAP = {
#     "u500_mean_n"   : "u500_mean",
#     "u500_center_n" : "u500_center",
#     "v500_mean_n"   : "v500_mean",
#     "v500_center_n" : "v500_center",
# }


# # ── CSV fallback builder ──────────────────────────────────────────────────────

# def _build_csv_env_lookup(csv_path: str) -> dict:
#     """
#     Load all_storms_final.csv và build lookup dict cho env features.
#     CSV lưu raw dam cho gph500 → truyền thẳng, sentinel guard [25,95] đúng.
#     """
#     try:
#         import pandas as pd
#     except ImportError:
#         logger.warning("pandas không có → CSV fallback không khả dụng")
#         return {}

#     if not os.path.exists(csv_path):
#         logger.warning(f"CSV fallback không tìm thấy: {csv_path}")
#         return {}

#     logger.info(f"Loading CSV env fallback: {csv_path}")
#     try:
#         df = pd.read_csv(csv_path, dtype={"storm_name": str, "dt": str})
#     except Exception as e:
#         logger.warning(f"CSV load failed: {e}")
#         return {}

#     lookup: dict = {}
#     for _, row in df.iterrows():
#         yr         = str(int(float(row["year"])))
#         name_raw   = str(row["storm_name"])
#         name_strip = name_raw.lstrip("0") or "0"
#         ts         = str(row["dt"])

#         d = {
#             # CSV lưu raw dam → env_data_processing sẽ apply sentinel [25,95]
#             "gph500_mean"  : float(row.get("env_gph500_mean",   -29.5)),
#             "gph500_center": float(row.get("env_gph500_center", -29.5)),
#             # CSV lưu boolean 0/1 → build_env_features dùng as-is
#             "u500_mean"    : float(row.get("env_u500_mean",   0.0)),
#             "u500_center"  : float(row.get("env_u500_center", 0.0)),
#             "v500_mean"    : float(row.get("env_v500_mean",   0.0)),
#             "v500_center"  : float(row.get("env_v500_center", 0.0)),
#             # move_velocity: CSV lưu normalized, nhân lại để downstream chia
#             "move_velocity": float(row.get("env_move_velocity", -1)) * _MOVE_VEL_NORM,
#             "history_direction12"  : [float(row.get(f"env_dir12_{i}",  -1)) for i in range(8)],
#             "history_direction24"  : [float(row.get(f"env_dir24_{i}",  -1)) for i in range(8)],
#             "history_inte_change24": [float(row.get(f"env_inten24_{i}", -1)) for i in range(4)],
#             # CSV path: gph500 là raw dam → KHÔNG đánh dấu already_normed
#             "gph500_already_normed": False,
#         }

#         lookup[(yr, name_strip, ts)] = d
#         lookup[(yr, name_raw,   ts)] = d

#     logger.info(f"CSV env lookup built: {len(lookup)} entries "
#                 f"({len(df)} rows × 2 name formats)")
#     return lookup


# def _auto_discover_csv(root_path: str) -> str | None:
#     candidates = [
#         os.path.join(root_path, "all_storms_final.csv"),
#         os.path.join(os.path.dirname(root_path), "all_storms_final.csv"),
#         os.path.join(os.path.dirname(os.path.dirname(root_path)),
#                      "all_storms_final.csv"),
#         "/kaggle/input/datasets/gmnguynhng/data-tc-finall/all_storms_final.csv",
#     ]
#     for p in candidates:
#         if os.path.exists(p):
#             return p
#     return None


# # ── env_data_processing ───────────────────────────────────────────────────────

# def env_data_processing(env_dict: dict) -> dict:
#     """
#     Clean env_npy dictionary.

#     FIX-DATA-15: Phân biệt hai nguồn gph500:
#       (A) .npy cũ (build_env_data_scs_v10): key "gph500_mean_n",
#           đã pre-normalized bằng mean=5900/std=200 → range ~[-28, 2].
#           Cờ "gph500_already_normed"=True được set bởi _load_env_npy().
#           → KHÔNG apply sentinel [25, 95], truyền thẳng qua.
#       (B) CSV fallback: key "gph500_mean", raw dam ~[27, 90].
#           Cờ "gph500_already_normed"=False (hoặc không có).
#           → Apply sentinel [25, 95] bình thường.

#     FIX-DATA-17: Bảo toàn cờ "gph500_already_normed" qua cleaned dict.
#     """
#     if not isinstance(env_dict, dict):
#         return {}

#     cleaned = {}
#     for k, v in env_dict.items():
#         if isinstance(v, (list, np.ndarray)):
#             cleaned[k] = v
#         elif k.endswith("_already_normed") or k.endswith("_valid") or k == "has_data3d":
#             # Giữ nguyên boolean flags
#             cleaned[k] = v
#         else:
#             cleaned[k] = 0.0 if v == -1 else v

#     # SST sentinel (giữ nguyên)
#     for sst_key in ("sst_mean", "sst_center", "sst"):
#         if sst_key in cleaned:
#             val = cleaned[sst_key]
#             if val is None or val == 0 or (isinstance(val, float) and val < _SST_VALID_MIN):
#                 cleaned[sst_key] = _SST_FILL_K

#     # FIX-DATA-15: GPH500 sentinel guard chỉ apply khi NOT already_normed
#     already_normed = cleaned.get("gph500_already_normed", False)
#     if not already_normed:
#         for gph_key in ("gph500_mean", "gph500_center"):
#             if gph_key in cleaned:
#                 val = cleaned[gph_key]
#                 if val is not None and isinstance(val, (int, float)):
#                     if val < _DATA3D_GPH_VALID_MIN or val > _DATA3D_GPH_VALID_MAX:
#                         cleaned[gph_key] = None  # sentinel → feature = 0
#     # Nếu already_normed=True: bỏ qua guard, giữ nguyên giá trị

#     return cleaned


# # ── seq_collate ───────────────────────────────────────────────────────────────

# def seq_collate(data):
#     (obs_traj, pred_traj, obs_rel, pred_rel,
#      nlp, mask, obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
#      obs_date, pred_date, img_obs, img_pred, env_data_raw, tyID) = zip(*data)

#     def traj_TBC(lst):
#         cat = torch.cat(lst, dim=0)
#         return cat.permute(2, 0, 1)

#     obs_traj_out    = traj_TBC(obs_traj)
#     pred_traj_out   = traj_TBC(pred_traj)
#     obs_rel_out     = traj_TBC(obs_rel)
#     pred_rel_out    = traj_TBC(pred_rel)
#     obs_Me_out      = traj_TBC(obs_Me)
#     pred_Me_out     = traj_TBC(pred_Me)
#     obs_Me_rel_out  = traj_TBC(obs_Me_rel)
#     pred_Me_rel_out = traj_TBC(pred_Me_rel)

#     nlp_out = torch.tensor(
#         [v for sl in nlp for v in (sl if hasattr(sl, "__iter__") else [sl])],
#         dtype=torch.float,
#     )
#     mask_out = torch.cat(list(mask), dim=0).permute(1, 0)

#     counts        = torch.tensor([t.shape[0] for t in obs_traj])
#     cum           = torch.cumsum(counts, dim=0)
#     starts        = torch.cat([torch.tensor([0]), cum[:-1]])
#     seq_start_end = torch.stack([starts, cum], dim=1)

#     img_obs_out  = torch.stack(list(img_obs), dim=0).permute(0, 4, 1, 2, 3).float()
#     img_pred_out = torch.stack(list(img_pred), dim=0).permute(0, 4, 1, 2, 3).float()

#     env_out = None
#     valid_envs = [d for d in env_data_raw if isinstance(d, dict)]
#     if valid_envs:
#         env_out  = {}
#         all_keys = set()
#         for d in valid_envs:
#             all_keys.update(d.keys())
#         # Loại bỏ các cờ internal trước khi collate
#         all_keys -= {"gph500_already_normed", "has_data3d",
#                      "gph500_mean_already_normed", "gph500_center_already_normed",
#                      "history_direction12_valid", "history_direction24_valid",
#                      "history_inte_change24_valid"}
#         for key in all_keys:
#             vals = []
#             for d in env_data_raw:
#                 if isinstance(d, dict) and key in d:
#                     v = d[key]
#                     v = torch.tensor(v, dtype=torch.float) if not torch.is_tensor(v) else v.float()
#                     vals.append(v)
#                 else:
#                     ref = next((d[key] for d in valid_envs if key in d), None)
#                     if ref is not None:
#                         rt = torch.tensor(ref, dtype=torch.float) if not torch.is_tensor(ref) else ref.float()
#                         vals.append(torch.zeros_like(rt))
#                     else:
#                         vals.append(torch.zeros(1))
#             try:
#                 env_out[key] = torch.stack(vals, dim=0)
#             except Exception:
#                 try:
#                     mx     = max(v.numel() for v in vals)
#                     padded = [F.pad(v.flatten(), (0, mx - v.numel())) for v in vals]
#                     env_out[key] = torch.stack(padded, dim=0)
#                 except Exception:
#                     pass

#     return (
#         obs_traj_out, pred_traj_out, obs_rel_out, pred_rel_out,
#         nlp_out, mask_out, seq_start_end,
#         obs_Me_out, pred_Me_out, obs_Me_rel_out, pred_Me_rel_out,
#         img_obs_out, img_pred_out, env_out, None, list(tyID),
#     )


# # ── TrajectoryDataset ─────────────────────────────────────────────────────────

# class TrajectoryDataset(Dataset):
#     """TC trajectory dataset for TCND_VN."""

#     def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,
#                  threshold=0.002, min_ped=1, delim=" ", other_modal="gph",
#                  test_year=None, type="train", split=None, is_test=False,
#                  csv_env_path=None,
#                  **kwargs):
#         super().__init__()

#         dtype = split if split is not None else type

#         if isinstance(data_dir, dict):
#             root  = data_dir["root"]
#             dtype = data_dir.get("type", dtype)
#         else:
#             root = data_dir
#         if is_test and dtype not in ("val", "test"):
#             dtype = "test"

#         root = os.path.abspath(root)
#         bn   = os.path.basename(root)
#         if bn in ("train", "test", "val"):
#             self.root_path = os.path.dirname(os.path.dirname(root))
#         elif bn == "Data1d":
#             self.root_path = os.path.dirname(root)
#         else:
#             self.root_path = root

#         self.data1d_path = os.path.join(self.root_path, "Data1d", dtype)
#         self.data3d_path = os.path.join(self.root_path, "Data3d")
#         for env_name in ("Env_data", "ENV_DATA", "env_data", "Env_Data"):
#             candidate = os.path.join(self.root_path, env_name)
#             if os.path.exists(candidate):
#                 self.env_path = candidate
#                 break
#         else:
#             self.env_path = os.path.join(self.root_path, "Env_data")

#         logger.info(f"root ({dtype}) : {self.root_path}")

#         # ── CSV fallback ─────────────────────────────────────────────────────
#         self._csv_env_lookup: dict = {}
#         self._env_path_missing = not os.path.exists(self.env_path)

#         if self._env_path_missing or csv_env_path is not None:
#             if csv_env_path is None:
#                 csv_env_path = _auto_discover_csv(self.root_path)
#             if csv_env_path:
#                 self._csv_env_lookup = _build_csv_env_lookup(csv_env_path)
#                 if self._env_path_missing:
#                     logger.info(
#                         f"Env_data không tìm thấy tại {self.env_path}. "
#                         f"Dùng CSV fallback: {csv_env_path} "
#                         f"({len(self._csv_env_lookup)//2} entries)"
#                     )
#             elif self._env_path_missing:
#                 logger.warning(
#                     f"Env_data không tìm thấy: {self.env_path} "
#                     f"VÀ không tìm thấy all_storms_final.csv. "
#                     f"Tất cả env features sẽ = 0 (model vẫn chạy được)."
#                 )

#         self.obs_len    = obs_len
#         self.pred_len   = pred_len
#         self.seq_len    = obs_len + pred_len
#         self.skip       = skip
#         self.modal_name = other_modal

#         if not os.path.exists(self.data1d_path):
#             logger.error(f"Missing Data1d: {self.data1d_path}")
#             self.num_seq       = 0
#             self.seq_start_end = []
#             self.tyID          = []
#             return

#         all_files = [
#             os.path.join(self.data1d_path, f)
#             for f in os.listdir(self.data1d_path)
#             if f.endswith(".txt") and (test_year is None or str(test_year) in f)
#         ]
#         logger.info(f"{len(all_files)} Data1d files (year={test_year})")

#         self.obs_traj_raw    = []
#         self.pred_traj_raw   = []
#         self.obs_Me_raw      = []
#         self.pred_Me_raw     = []
#         self.obs_rel_raw     = []
#         self.pred_rel_raw    = []
#         self.obs_Me_rel_raw  = []
#         self.pred_Me_rel_raw = []
#         self.non_linear_ped  = []
#         self.tyID            = []
#         num_peds_in_seq      = []

#         self.env_cache: dict[tuple, dict] = {}

#         for path in all_files:
#             base   = os.path.splitext(os.path.basename(path))[0]
#             parts  = base.split("_")
#             f_year = parts[0] if parts else "unknown"
#             f_name = parts[1] if len(parts) > 1 else base

#             d    = self._read_file(path, delim)
#             data = d["main"]
#             add  = d["addition"]
#             if len(data) < self.seq_len:
#                 continue

#             frames     = np.unique(data[:, 0]).tolist()
#             frame_data = [data[data[:, 0] == f] for f in frames]

#             n_frames = len(frames)
#             if n_frames < self.seq_len:
#                 continue
#             n_seq = (n_frames - self.seq_len) // skip + 1

#             for idx in range(0, n_seq * skip, skip):
#                 if idx + self.seq_len > len(frame_data):
#                     break

#                 seg  = np.concatenate(frame_data[idx: idx + self.seq_len])
#                 peds = np.unique(seg[:, 1])

#                 buf_obs_traj    = []
#                 buf_pred_traj   = []
#                 buf_obs_rel     = []
#                 buf_pred_rel    = []
#                 buf_obs_Me      = []
#                 buf_pred_Me     = []
#                 buf_obs_Me_rel  = []
#                 buf_pred_Me_rel = []
#                 buf_nlp         = []
#                 cnt             = 0

#                 for pid in peds:
#                     ps = seg[seg[:, 1] == pid]
#                     if len(ps) != self.seq_len:
#                         continue
#                     ps_t = np.transpose(ps[:, 2:])
#                     rel  = np.zeros_like(ps_t)
#                     rel[:, 1:] = ps_t[:, 1:] - ps_t[:, :-1]

#                     buf_obs_traj.append(torch.from_numpy(ps_t[:2, :obs_len]).float())
#                     buf_pred_traj.append(torch.from_numpy(ps_t[:2, obs_len:]).float())
#                     buf_obs_rel.append(torch.from_numpy(rel[:2, :obs_len]).float())
#                     buf_pred_rel.append(torch.from_numpy(rel[:2, obs_len:]).float())
#                     buf_obs_Me.append(torch.from_numpy(ps_t[2:, :obs_len]).float())
#                     buf_pred_Me.append(torch.from_numpy(ps_t[2:, obs_len:]).float())
#                     buf_obs_Me_rel.append(torch.from_numpy(rel[2:, :obs_len]).float())
#                     buf_pred_Me_rel.append(torch.from_numpy(rel[2:, obs_len:]).float())
#                     buf_nlp.append(self._poly_fit(ps_t, pred_len, threshold))
#                     cnt += 1

#                 if cnt >= min_ped:
#                     self.obs_traj_raw.extend(buf_obs_traj)
#                     self.pred_traj_raw.extend(buf_pred_traj)
#                     self.obs_rel_raw.extend(buf_obs_rel)
#                     self.pred_rel_raw.extend(buf_pred_rel)
#                     self.obs_Me_raw.extend(buf_obs_Me)
#                     self.pred_Me_raw.extend(buf_pred_Me)
#                     self.obs_Me_rel_raw.extend(buf_obs_Me_rel)
#                     self.pred_Me_rel_raw.extend(buf_pred_Me_rel)
#                     self.non_linear_ped.extend(buf_nlp)

#                     num_peds_in_seq.append(cnt)
#                     self.tyID.append({
#                         "old":    [f_year, f_name, idx],
#                         "tydate": [add[i][0] for i in range(idx, idx + self.seq_len)],
#                     })

#         self.num_seq = len(self.tyID)
#         cum = np.cumsum(num_peds_in_seq).tolist()
#         self.seq_start_end = list(zip([0] + cum[:-1], cum))
#         logger.info(f"Loaded {self.num_seq} sequences")

#     def _read_file(self, path: str, delim: str) -> dict:
#         data, add = [], []
#         with open(path, encoding="utf-8", errors="ignore") as f:
#             raw_lines = f.readlines()
#         for line in raw_lines:
#             line = line.strip()
#             if not line or line.startswith(("#", "//", "-", "=")):
#                 continue
#             parts = line.split()
#             if len(parts) < 7:
#                 continue
#             try:
#                 int(parts[0])
#             except ValueError:
#                 continue
#             try:
#                 frame_id  = float(parts[0])
#                 lon_norm  = float(parts[1])
#                 lat_norm  = float(parts[2])
#                 pres_norm = float(parts[3])
#                 wnd_norm  = float(parts[4])
#                 date      = parts[5]
#                 name      = parts[6]
#                 add.append([date, name])
#                 data.append([frame_id, 1.0, lon_norm, lat_norm, pres_norm, wnd_norm])
#             except (ValueError, IndexError):
#                 continue
#         return {
#             "main":     np.asarray(data, dtype=np.float32) if data else np.zeros((0, 6), dtype=np.float32),
#             "addition": add,
#         }

#     def _poly_fit(self, traj, tlen, threshold):
#         t  = np.linspace(0, tlen - 1, tlen)
#         rx = np.polyfit(t, traj[0, -tlen:], 2, full=True)[1]
#         ry = np.polyfit(t, traj[1, -tlen:], 2, full=True)[1]
#         return 1.0 if (len(rx) > 0 and rx[0] + ry[0] >= threshold) else 0.0

#     def _normalize_data3d(self, arr: np.ndarray) -> np.ndarray:
#         arr = arr.copy()
#         for c in range(DATA3D_CH):
#             ch = arr[:, :, c]
#             ch[ch > _DATA3D_SENTINEL_LARGE] = np.nan
#             if c in _DATA3D_SENTINEL_ZERO_CHANNELS:
#                 ch[ch == 0.0] = np.nan
#             if c == _DATA3D_SST_CHANNEL:
#                 ch[ch < _SST_VALID_MIN] = _SST_FILL_K
#             if np.any(np.isnan(ch)):
#                 valid_vals = ch[~np.isnan(ch)]
#                 fill_val = float(np.median(valid_vals)) if len(valid_vals) > 0 else float(DATA3D_MEAN[c])
#                 ch[np.isnan(ch)] = fill_val
#             arr[:, :, c] = (ch - DATA3D_MEAN[c]) / (DATA3D_STD[c] + 1e-6)
#         return np.clip(arr, -5.0, 5.0)

#     def _load_data3d_file(self, path: str):
#         try:
#             if path.endswith(".npy"):
#                 arr = np.load(path).astype(np.float32)
#             elif path.endswith(".nc") and HAS_NC:
#                 with nc.Dataset(path) as ds:
#                     keys = list(ds.variables.keys())
#                     arr  = np.array(ds.variables[keys[-1]][:]).astype(np.float32)
#             else:
#                 return None
#             if arr.ndim == 2:
#                 arr = arr[:, :, np.newaxis]
#             if arr.ndim == 3:
#                 if arr.shape[0] == DATA3D_CH:
#                     arr = arr.transpose(1, 2, 0)
#                 H, W, C = arr.shape
#                 if H != DATA3D_H or W != DATA3D_W:
#                     if HAS_CV2:
#                         arr = cv2.resize(arr, (DATA3D_W, DATA3D_H))
#                     else:
#                         arr = arr[:DATA3D_H, :DATA3D_W, :]
#                         if arr.shape[0] < DATA3D_H:
#                             arr = np.pad(arr, ((0, DATA3D_H - arr.shape[0]), (0, 0), (0, 0)))
#                         if arr.shape[1] < DATA3D_W:
#                             arr = np.pad(arr, ((0, 0), (0, DATA3D_W - arr.shape[1]), (0, 0)))
#                 if arr.shape[2] < DATA3D_CH:
#                     arr = np.concatenate([
#                         arr,
#                         np.zeros((DATA3D_H, DATA3D_W, DATA3D_CH - arr.shape[2]), dtype=np.float32),
#                     ], axis=2)
#                 arr = arr[:, :, :DATA3D_CH]
#                 return self._normalize_data3d(arr)
#         except Exception as e:
#             logger.debug(f"Data3d load error {path}: {e}")
#         return None

#     def img_read(self, year, ty_name, timestamp) -> torch.Tensor:
#         folder = os.path.join(self.data3d_path, str(year), str(ty_name))
#         if not os.path.exists(folder):
#             return torch.zeros(DATA3D_H, DATA3D_W, DATA3D_CH)
#         prefix = f"WP{year}{ty_name}_{timestamp}"
#         for ext in (".npy", ".nc"):
#             p = os.path.join(folder, prefix + ext)
#             if os.path.exists(p):
#                 arr = self._load_data3d_file(p)
#                 if arr is not None:
#                     return torch.from_numpy(arr).float()
#         try:
#             for fname in sorted(os.listdir(folder)):
#                 if timestamp in fname and fname.endswith((".npy", ".nc")):
#                     arr = self._load_data3d_file(os.path.join(folder, fname))
#                     if arr is not None:
#                         return torch.from_numpy(arr).float()
#         except Exception:
#             pass
#         return torch.zeros(DATA3D_H, DATA3D_W, DATA3D_CH)

#     def _load_env_npy(self, year, ty_name, timestamp):
#         """
#         FIX-DATA-15: Load .npy và remap key "gph500_mean_n" → "gph500_mean"
#         với cờ "gph500_already_normed"=True để downstream KHÔNG apply
#         sentinel guard [25,95] và KHÔNG z-score lần 2.
#         """
#         folder = os.path.join(self.env_path, str(year), str(ty_name))
#         if os.path.exists(folder):
#             # Thử load file theo thứ tự ưu tiên
#             candidates = []
#             for fname in [f"WP{year}{ty_name}_{timestamp}.npy", f"{timestamp}.npy"]:
#                 p = os.path.join(folder, fname)
#                 if os.path.exists(p):
#                     candidates.append(p)
#             if not candidates:
#                 try:
#                     candidates = [
#                         os.path.join(folder, f)
#                         for f in os.listdir(folder)
#                         if timestamp in f and f.endswith(".npy")
#                     ]
#                 except Exception:
#                     pass

#             for p in candidates:
#                 try:
#                     raw = np.load(p, allow_pickle=True).item()
#                     if not isinstance(raw, dict):
#                         continue

#                     # FIX-DATA-15: Remap key cũ "_n" → key chuẩn
#                     remapped = dict(raw)

#                     # GPH500: key cũ là "gph500_mean_n" (pre-normed bởi build_env)
#                     has_old_gph = False
#                     for old_k, new_k in _NPY_KEY_REMAP.items():
#                         if old_k in remapped and new_k not in remapped:
#                             remapped[new_k] = remapped.pop(old_k)
#                             has_old_gph = True

#                     # u500/v500: key cũ là "u500_mean_n" v.v.
#                     for old_k, new_k in _NPY_U500_KEY_REMAP.items():
#                         if old_k in remapped and new_k not in remapped:
#                             remapped[new_k] = remapped.pop(old_k)

#                     # Đánh dấu gph500 đã normalized nếu đến từ key "_n"
#                     if has_old_gph:
#                         remapped["gph500_already_normed"] = True

#                     return env_data_processing(remapped)
#                 except Exception as e:
#                     logger.debug(f"env npy load error {p}: {e}")

#         # CSV fallback
#         if self._csv_env_lookup:
#             yr_str = str(year)
#             ts_str = str(timestamp)
#             name_strip  = str(ty_name).lstrip("0") or "0"
#             name_padded = str(ty_name).zfill(4)
#             for name_try in (str(ty_name), name_strip, name_padded):
#                 raw_dict = self._csv_env_lookup.get((yr_str, name_try, ts_str))
#                 if raw_dict is not None:
#                     return env_data_processing(dict(raw_dict))

#         return None

#     def _get_env_features(self, year, ty_name, dates, obs_traj, obs_Me):
#         T          = len(dates)
#         all_feats  = []
#         prev_speed = None
#         for t in range(T):
#             env_npy = self._load_env_npy(year, ty_name, dates[t])
#             feat    = build_env_features_one_step(
#                 lon_norm=float(obs_traj[0, t]), lat_norm=float(obs_traj[1, t]),
#                 wind_norm=float(obs_Me[1, t]),
#                 pres_norm=float(obs_Me[0, t]),
#                 timestamp=dates[t],
#                 env_npy=env_npy, prev_speed_kmh=prev_speed,
#             )
#             all_feats.append(feat)
#             if isinstance(env_npy, dict):
#                 mv = float(env_npy.get("move_velocity", 0.0) or 0.0)
#                 prev_speed = mv if mv != -1 else 0.0
#         env_out = {}
#         for key in ENV_FEATURE_DIMS:
#             dim  = ENV_FEATURE_DIMS[key]
#             rows = []
#             for feat in all_feats:
#                 v = feat.get(key, [0.0] * dim)
#                 t = torch.tensor(v, dtype=torch.float)
#                 if t.numel() < dim:
#                     t = F.pad(t, (0, dim - t.numel()))
#                 rows.append(t[:dim])
#             env_out[key] = torch.stack(rows, dim=0)
#         return env_out

#     def _embed_time(self, date_list):
#         rows = []
#         for d in date_list:
#             try:
#                 rows.append([
#                     (float(d[:4]) - 1949) / 70.0 - 0.5,
#                     (float(d[4:6]) - 1)   / 11.0 - 0.5,
#                     (float(d[6:8]) - 1)   / 30.0 - 0.5,
#                     float(d[8:10])         / 18.0 - 0.5,
#                 ])
#             except Exception:
#                 rows.append([0.0, 0.0, 0.0, 0.0])
#         return torch.tensor(rows, dtype=torch.float).t().unsqueeze(0)

#     def __len__(self):
#         return self.num_seq

#     def __getitem__(self, index):
#         if self.num_seq == 0:
#             raise IndexError("Empty dataset")
#         s, e   = self.seq_start_end[index]
#         info   = self.tyID[index]
#         year   = str(info["old"][0])
#         tyname = str(info["old"][1])
#         dates  = info["tydate"]

#         imgs     = [self.img_read(year, tyname, ts) for ts in dates[:self.obs_len]]
#         img_obs  = torch.stack(imgs, dim=0)
#         img_pred = torch.zeros(self.pred_len, DATA3D_H, DATA3D_W, DATA3D_CH)

#         obs_traj    = torch.stack([self.obs_traj_raw[i]    for i in range(s, e)])
#         pred_traj   = torch.stack([self.pred_traj_raw[i]   for i in range(s, e)])
#         obs_rel     = torch.stack([self.obs_rel_raw[i]     for i in range(s, e)])
#         pred_rel    = torch.stack([self.pred_rel_raw[i]    for i in range(s, e)])
#         obs_Me      = torch.stack([self.obs_Me_raw[i]      for i in range(s, e)])
#         pred_Me     = torch.stack([self.pred_Me_raw[i]     for i in range(s, e)])
#         obs_Me_rel  = torch.stack([self.obs_Me_rel_raw[i]  for i in range(s, e)])
#         pred_Me_rel = torch.stack([self.pred_Me_rel_raw[i] for i in range(s, e)])

#         n    = e - s
#         nlp  = [self.non_linear_ped[i] for i in range(s, e)]
#         mask = torch.ones(n, self.seq_len)

#         obs_traj_np = obs_traj[0].numpy()
#         obs_Me_np   = obs_Me[0].numpy()

#         cache_key = (year, tyname, tuple(dates[:self.obs_len]))
#         if cache_key not in self.env_cache:
#             self.env_cache[cache_key] = self._get_env_features(
#                 year, tyname, dates[:self.obs_len], obs_traj_np, obs_Me_np)
#         env_out = self.env_cache[cache_key]

#         return [
#             obs_traj, pred_traj, obs_rel, pred_rel, nlp, mask,
#             obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
#             self._embed_time(dates[:self.obs_len]),
#             self._embed_time(dates[self.obs_len:]),
#             img_obs, img_pred, env_out, info,
#         ]

"""
Model/data/trajectoriesWithMe_unet_training.py  ── v23
=======================================================
FIXES vs v18:

  FIX-DATA-18 [CRITICAL] _build_csv_env_lookup: env_gph500_mean trong CSV
              lưu RAW dam (27-90), KHÔNG phải pre-normalized. Đã xác nhận
              env_gph500_mean == d3d_gph500_mean_n và cả hai có range 27-90
              (tên "_n" trong CSV là misleading, data thực là raw dam).
              → gph500_already_normed = False cho CSV path (đúng).
              → sentinel guard [25, 95] apply bình thường.
              → -29.5 sentinel được lọc ra đúng (< 25).

  FIX-DATA-19 [CRITICAL] _build_csv_env_lookup: move_velocity trong CSV
              đã normalized (/1219.84), range 0-0.55. Code cũ nhân lại
              * _MOVE_VEL_NORM để downstream chia lại. Hành vi này đúng
              về mặt toán học nhưng không rõ ràng. Giữ nguyên (correct).

  FIX-DATA-20 [CRITICAL] _build_csv_env_lookup: cột history_direction12/24
              và history_inte_change24 trong CSV là one-hot encoded floats,
              KHÔNG phải -1 sentinels. build_env_features_one_step cần nhận
              list of floats [0.0, 1.0, 0.0, ...] trực tiếp, không qua
              sentinel check "all == -1". Đảm bảo truyền đúng format.

  FIX-DATA-21 [BUG] _load_env_npy: sau khi remap key "_n" → chuẩn, gọi
              env_data_processing(remapped) nhưng env_data_processing lại
              apply sentinel guard trên gph500 dù cờ already_normed=True.
              Nguyên nhân: cờ đặt trong dict TRƯỚC khi gọi processing,
              nhưng bị lọc ra bởi "cleaned[k] = 0.0 if v == -1 else v"
              khi v=True/False. Fix: xử lý boolean flags TRƯỚC trong loop.

  FIX-DATA-22 CURRICULUM REMOVED: Curriculum learning gây ADE tụt nghiêm
              trọng mỗi lần tăng len (282→444 km khi len 5→6 ở ep 10).
              Với 13M params và data 8473 sequences, model cần học toàn bộ
              pred_len=12 từ đầu. Thay curriculum bằng sequence weighting:
              các bước xa hơn có weight thấp hơn lúc đầu (soft weighting),
              tăng dần theo epoch thông qua loss weight, không cắt seq.

  FIX-DATA-23 CSV lookup: storm_name format. CSV có names như "0002" (4 digits
              với leading zeros) và stripped "2". Lookup cần thử cả hai.

Kept from v17:
  FIX-DATA-12 CSV fallback auto-discover all_storms_final.csv
  FIX-DATA-10 DATA3D_MEAN/STD ch0 gph500 đúng /380 dam
  FIX-DATA-5  ch3/ch4 u500_center/v500_center corrected
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import netCDF4 as nc
    HAS_NC = True
except ImportError:
    HAS_NC = False

from Model.env_net_transformer_gphsplit import (
    bearing_to_scs_center_onehot, dist_to_scs_boundary_onehot,
    delta_velocity_onehot, intensity_class_onehot,
    build_env_features_one_step, feat_to_tensor, ENV_FEATURE_DIMS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA3D_H  = 81
DATA3D_W  = 81
DATA3D_CH = 13

# ── DATA3D normalisation constants ────────────────────────────────────────────
DATA3D_MEAN = np.array([
    33.64,      # ch0  gph500_mean  (raw dam /380 unit)
    5843.14,    # ch1  u500_mean
    1482.47,    # ch2  v500_mean
    5930.27,    # ch3  u500_center
    1622.27,    # ch4  v500_center
    0.27,       # ch5
    -0.34,      # ch6
    -0.86,      # ch7
    0.25,       # ch8
    1.76,       # ch9
    1.34,       # ch10
    0.94,       # ch11
    300.95,     # ch12 SST
], dtype=np.float32)

DATA3D_STD = np.array([
    7.08,       # ch0
    50.55,      # ch1
    29.42,      # ch2
    1025.26,    # ch3
    1600.32,    # ch4
    4.73,       # ch5
    2.98,       # ch6
    2.75,       # ch7
    5.37,       # ch8
    2.29,       # ch9
    2.21,       # ch10
    2.68,       # ch11
    3.05,       # ch12 SST
], dtype=np.float32)

# ── Sentinel thresholds ───────────────────────────────────────────────────────
_DATA3D_SENTINEL_LARGE         = 20000.0
_DATA3D_SENTINEL_ZERO_CHANNELS = {0}
_DATA3D_GPH_VALID_MIN          = 25.0
_DATA3D_GPH_VALID_MAX          = 95.0
_DATA3D_SST_CHANNEL            = 12
_SST_VALID_MIN                 = 270.0
_SST_FILL_K                    = 298.0
_MOVE_VEL_NORM                 = 1219.84

# ── Key mapping: build_env_data_scs_v10.py lưu suffix "_n" ───────────────────
# FIX-DATA-15 (kept): Map từ key cũ (có _n) sang key chuẩn
_NPY_KEY_REMAP = {
    "gph500_mean_n"   : "gph500_mean",
    "gph500_center_n" : "gph500_center",
}
_NPY_U500_KEY_REMAP = {
    "u500_mean_n"   : "u500_mean",
    "u500_center_n" : "u500_center",
    "v500_mean_n"   : "v500_mean",
    "v500_center_n" : "v500_center",
}


# ── CSV fallback builder ──────────────────────────────────────────────────────

def _build_csv_env_lookup(csv_path: str) -> dict:
    """
    Load all_storms_final.csv → lookup dict.

    FIX-DATA-18: env_gph500_mean trong CSV là raw dam (27-90), sentinel=-29.5.
                 Đây là KHÔNG phải pre-normalized. gph500_already_normed=False.
    FIX-DATA-20: history_direction12/24 và inten24 trong CSV là one-hot floats
                 [0.0, 1.0, 0.0, ...]. Truyền thẳng không cần xử lý sentinel.
    FIX-DATA-23: Lưu cả key (yr, name_original, ts) lẫn (yr, name_stripped, ts).
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas không có → CSV fallback không khả dụng")
        return {}

    if not os.path.exists(csv_path):
        logger.warning(f"CSV fallback không tìm thấy: {csv_path}")
        return {}

    logger.info(f"Loading CSV env fallback: {csv_path}")
    try:
        df = pd.read_csv(csv_path, dtype={"storm_name": str, "dt": str})
    except Exception as e:
        logger.warning(f"CSV load failed: {e}")
        return {}

    lookup: dict = {}
    for _, row in df.iterrows():
        yr          = str(int(float(row["year"])))
        name_raw    = str(row["storm_name"])
        name_strip  = name_raw.lstrip("0") or "0"
        ts          = str(row["dt"])

        # FIX-DATA-20: history_direction → one-hot floats từ CSV columns
        dir12  = [float(row.get(f"env_dir12_{i}",  0.0)) for i in range(8)]
        dir24  = [float(row.get(f"env_dir24_{i}",  0.0)) for i in range(8)]
        inten24 = [float(row.get(f"env_inten24_{i}", 0.0)) for i in range(4)]

        # FIX-DATA-18: env_gph500_mean là raw dam → gph500_already_normed=False
        gph_mean   = float(row.get("env_gph500_mean",   -29.5))
        gph_center = float(row.get("env_gph500_center", -29.5))

        # move_velocity: CSV lưu normalized [0-0.55], nhân lại để downstream chia
        mv_norm = float(row.get("env_move_velocity", 0.0))
        mv_raw  = mv_norm * _MOVE_VEL_NORM  # → raw km/h, downstream /1219.84

        d = {
            "gph500_mean"            : gph_mean,
            "gph500_center"          : gph_center,
            "gph500_already_normed"  : False,  # FIX-DATA-18: raw dam
            "u500_mean"              : float(row.get("env_u500_mean",   0.0)),
            "u500_center"            : float(row.get("env_u500_center", 0.0)),
            "v500_mean"              : float(row.get("env_v500_mean",   0.0)),
            "v500_center"            : float(row.get("env_v500_center", 0.0)),
            "move_velocity"          : mv_raw,
            "history_direction12"    : dir12,   # FIX-DATA-20: already one-hot
            "history_direction24"    : dir24,
            "history_inte_change24"  : inten24,
        }

        # FIX-DATA-23: store both name formats
        lookup[(yr, name_raw,   ts)] = d
        lookup[(yr, name_strip, ts)] = d
        # Also zero-padded variants
        name_padded4 = name_raw.zfill(4)
        name_padded2 = name_raw.zfill(2)
        lookup[(yr, name_padded4, ts)] = d
        lookup[(yr, name_padded2, ts)] = d

    n_storms = df['storm_name'].nunique()
    logger.info(f"CSV env lookup built: {len(lookup)} entries ({n_storms} storms)")
    return lookup


def _auto_discover_csv(root_path: str) -> str | None:
    candidates = [
        os.path.join(root_path, "all_storms_final.csv"),
        os.path.join(os.path.dirname(root_path), "all_storms_final.csv"),
        os.path.join(os.path.dirname(os.path.dirname(root_path)),
                     "all_storms_final.csv"),
        "/kaggle/input/datasets/gmnguynhng/data-tc-finall/all_storms_final.csv",
        "/kaggle/working/all_storms_final.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            logger.info(f"Auto-discovered CSV: {p}")
            return p
    return None


# ── env_data_processing ───────────────────────────────────────────────────────

def env_data_processing(env_dict: dict) -> dict:
    """
    Clean env_npy dictionary.

    FIX-DATA-21: Boolean flags (already_normed, has_data3d) phải được xử lý
                 TRƯỚC vòng lặp chính để tránh bị convert sang float(0.0).
                 Sau đó mới apply sentinel guards.
    """
    if not isinstance(env_dict, dict):
        return {}

    # FIX-DATA-21: Trích xuất boolean flags trước
    already_normed = bool(env_dict.get("gph500_already_normed", False))

    cleaned = {"gph500_already_normed": already_normed}

    for k, v in env_dict.items():
        # Skip boolean flags đã xử lý
        if k in ("gph500_already_normed", "has_data3d",
                 "gph500_mean_already_normed", "gph500_center_already_normed"):
            continue
        if isinstance(v, (list, np.ndarray)):
            cleaned[k] = v
        elif isinstance(v, bool):
            cleaned[k] = v
        elif v == -1:
            cleaned[k] = 0.0
        else:
            cleaned[k] = v

    # SST sentinel
    for sst_key in ("sst_mean", "sst_center", "sst"):
        if sst_key in cleaned:
            val = cleaned[sst_key]
            if val is None or val == 0 or (isinstance(val, float) and val < _SST_VALID_MIN):
                cleaned[sst_key] = _SST_FILL_K

    # GPH500 sentinel guard chỉ apply khi NOT already_normed
    if not already_normed:
        for gph_key in ("gph500_mean", "gph500_center"):
            if gph_key in cleaned:
                val = cleaned[gph_key]
                if val is not None and isinstance(val, (int, float)):
                    if val < _DATA3D_GPH_VALID_MIN or val > _DATA3D_GPH_VALID_MAX:
                        cleaned[gph_key] = None  # sentinel → feature = 0
    # already_normed=True: giữ nguyên, không apply guard

    return cleaned


# ── seq_collate ───────────────────────────────────────────────────────────────

def seq_collate(data):
    (obs_traj, pred_traj, obs_rel, pred_rel,
     nlp, mask, obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
     obs_date, pred_date, img_obs, img_pred, env_data_raw, tyID) = zip(*data)

    def traj_TBC(lst):
        cat = torch.cat(lst, dim=0)
        return cat.permute(2, 0, 1)

    obs_traj_out    = traj_TBC(obs_traj)
    pred_traj_out   = traj_TBC(pred_traj)
    obs_rel_out     = traj_TBC(obs_rel)
    pred_rel_out    = traj_TBC(pred_rel)
    obs_Me_out      = traj_TBC(obs_Me)
    pred_Me_out     = traj_TBC(pred_Me)
    obs_Me_rel_out  = traj_TBC(obs_Me_rel)
    pred_Me_rel_out = traj_TBC(pred_Me_rel)

    nlp_out = torch.tensor(
        [v for sl in nlp for v in (sl if hasattr(sl, "__iter__") else [sl])],
        dtype=torch.float,
    )
    mask_out = torch.cat(list(mask), dim=0).permute(1, 0)

    counts        = torch.tensor([t.shape[0] for t in obs_traj])
    cum           = torch.cumsum(counts, dim=0)
    starts        = torch.cat([torch.tensor([0]), cum[:-1]])
    seq_start_end = torch.stack([starts, cum], dim=1)

    img_obs_out  = torch.stack(list(img_obs),  dim=0).permute(0, 4, 1, 2, 3).float()
    img_pred_out = torch.stack(list(img_pred), dim=0).permute(0, 4, 1, 2, 3).float()

    env_out    = None
    valid_envs = [d for d in env_data_raw if isinstance(d, dict)]
    if valid_envs:
        env_out  = {}
        all_keys = set()
        for d in valid_envs:
            all_keys.update(d.keys())

        # FIX-DATA-21: Không collate boolean/internal flags
        _skip_keys = {
            "gph500_already_normed", "has_data3d",
            "gph500_mean_already_normed", "gph500_center_already_normed",
            "history_direction12_valid", "history_direction24_valid",
            "history_inte_change24_valid",
        }
        all_keys -= _skip_keys

        for key in all_keys:
            vals = []
            for d in env_data_raw:
                if isinstance(d, dict) and key in d:
                    v = d[key]
                    v = torch.tensor(v, dtype=torch.float) if not torch.is_tensor(v) else v.float()
                    vals.append(v)
                else:
                    ref = next((d[key] for d in valid_envs if key in d), None)
                    if ref is not None:
                        rt = torch.tensor(ref, dtype=torch.float) if not torch.is_tensor(ref) else ref.float()
                        vals.append(torch.zeros_like(rt))
                    else:
                        vals.append(torch.zeros(1))
            try:
                env_out[key] = torch.stack(vals, dim=0)
            except Exception:
                try:
                    mx     = max(v.numel() for v in vals)
                    padded = [F.pad(v.flatten(), (0, mx - v.numel())) for v in vals]
                    env_out[key] = torch.stack(padded, dim=0)
                except Exception:
                    pass

    return (
        obs_traj_out, pred_traj_out, obs_rel_out, pred_rel_out,
        nlp_out, mask_out, seq_start_end,
        obs_Me_out, pred_Me_out, obs_Me_rel_out, pred_Me_rel_out,
        img_obs_out, img_pred_out, env_out, None, list(tyID),
    )


# ── TrajectoryDataset ─────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    TC trajectory dataset for TCND_VN.
    FIX-DATA-22: CURRICULUM REMOVED. Luôn train trên pred_len=12 đầy đủ.
                 Soft weighting theo step được xử lý trong loss function.
    """

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,
                 threshold=0.002, min_ped=1, delim=" ", other_modal="gph",
                 test_year=None, type="train", split=None, is_test=False,
                 csv_env_path=None,
                 **kwargs):
        super().__init__()

        dtype = split if split is not None else type

        if isinstance(data_dir, dict):
            root  = data_dir["root"]
            dtype = data_dir.get("type", dtype)
        else:
            root = data_dir
        if is_test and dtype not in ("val", "test"):
            dtype = "test"

        root = os.path.abspath(root)
        bn   = os.path.basename(root)
        if bn in ("train", "test", "val"):
            self.root_path = os.path.dirname(os.path.dirname(root))
        elif bn == "Data1d":
            self.root_path = os.path.dirname(root)
        else:
            self.root_path = root

        self.data1d_path = os.path.join(self.root_path, "Data1d", dtype)
        self.data3d_path = os.path.join(self.root_path, "Data3d")
        for env_name in ("Env_data", "ENV_DATA", "env_data", "Env_Data"):
            candidate = os.path.join(self.root_path, env_name)
            if os.path.exists(candidate):
                self.env_path = candidate
                break
        else:
            self.env_path = os.path.join(self.root_path, "Env_data")

        logger.info(f"root ({dtype}) : {self.root_path}")

        # ── CSV fallback ─────────────────────────────────────────────────────
        self._csv_env_lookup: dict  = {}
        self._env_path_missing      = not os.path.exists(self.env_path)

        if self._env_path_missing or csv_env_path is not None:
            if csv_env_path is None:
                csv_env_path = _auto_discover_csv(self.root_path)
            if csv_env_path:
                self._csv_env_lookup = _build_csv_env_lookup(csv_env_path)
                if self._env_path_missing:
                    logger.info(
                        f"Env_data không tìm thấy: {self.env_path}. "
                        f"Dùng CSV fallback ({len(self._csv_env_lookup)//4} entries)"
                    )
            elif self._env_path_missing:
                logger.warning(
                    f"Env_data không tìm thấy: {self.env_path} "
                    f"VÀ không tìm thấy CSV. Env features sẽ = 0."
                )

        self.obs_len    = obs_len
        self.pred_len   = pred_len
        self.seq_len    = obs_len + pred_len
        self.skip       = skip
        self.modal_name = other_modal

        if not os.path.exists(self.data1d_path):
            logger.error(f"Missing Data1d: {self.data1d_path}")
            self.num_seq       = 0
            self.seq_start_end = []
            self.tyID          = []
            return

        all_files = [
            os.path.join(self.data1d_path, f)
            for f in os.listdir(self.data1d_path)
            if f.endswith(".txt") and (test_year is None or str(test_year) in f)
        ]
        logger.info(f"{len(all_files)} Data1d files (year={test_year})")

        self.obs_traj_raw    = []
        self.pred_traj_raw   = []
        self.obs_Me_raw      = []
        self.pred_Me_raw     = []
        self.obs_rel_raw     = []
        self.pred_rel_raw    = []
        self.obs_Me_rel_raw  = []
        self.pred_Me_rel_raw = []
        self.non_linear_ped  = []
        self.tyID            = []
        num_peds_in_seq      = []
        self.env_cache: dict = {}

        for path in all_files:
            base   = os.path.splitext(os.path.basename(path))[0]
            parts  = base.split("_")
            f_year = parts[0] if parts else "unknown"
            f_name = parts[1] if len(parts) > 1 else base

            d    = self._read_file(path, delim)
            data = d["main"]
            add  = d["addition"]
            if len(data) < self.seq_len:
                continue

            frames     = np.unique(data[:, 0]).tolist()
            frame_data = [data[data[:, 0] == f] for f in frames]
            n_frames   = len(frames)
            if n_frames < self.seq_len:
                continue
            n_seq = (n_frames - self.seq_len) // skip + 1

            for idx in range(0, n_seq * skip, skip):
                if idx + self.seq_len > len(frame_data):
                    break

                seg  = np.concatenate(frame_data[idx: idx + self.seq_len])
                peds = np.unique(seg[:, 1])

                buf_obs_traj    = []
                buf_pred_traj   = []
                buf_obs_rel     = []
                buf_pred_rel    = []
                buf_obs_Me      = []
                buf_pred_Me     = []
                buf_obs_Me_rel  = []
                buf_pred_Me_rel = []
                buf_nlp         = []
                cnt             = 0

                for pid in peds:
                    ps = seg[seg[:, 1] == pid]
                    if len(ps) != self.seq_len:
                        continue
                    ps_t = np.transpose(ps[:, 2:])
                    rel  = np.zeros_like(ps_t)
                    rel[:, 1:] = ps_t[:, 1:] - ps_t[:, :-1]

                    buf_obs_traj.append(
                        torch.from_numpy(ps_t[:2, :obs_len]).float())
                    buf_pred_traj.append(
                        torch.from_numpy(ps_t[:2, obs_len:]).float())
                    buf_obs_rel.append(
                        torch.from_numpy(rel[:2, :obs_len]).float())
                    buf_pred_rel.append(
                        torch.from_numpy(rel[:2, obs_len:]).float())
                    buf_obs_Me.append(
                        torch.from_numpy(ps_t[2:, :obs_len]).float())
                    buf_pred_Me.append(
                        torch.from_numpy(ps_t[2:, obs_len:]).float())
                    buf_obs_Me_rel.append(
                        torch.from_numpy(rel[2:, :obs_len]).float())
                    buf_pred_Me_rel.append(
                        torch.from_numpy(rel[2:, obs_len:]).float())
                    buf_nlp.append(self._poly_fit(ps_t, pred_len, threshold))
                    cnt += 1

                if cnt >= min_ped:
                    self.obs_traj_raw.extend(buf_obs_traj)
                    self.pred_traj_raw.extend(buf_pred_traj)
                    self.obs_rel_raw.extend(buf_obs_rel)
                    self.pred_rel_raw.extend(buf_pred_rel)
                    self.obs_Me_raw.extend(buf_obs_Me)
                    self.pred_Me_raw.extend(buf_pred_Me)
                    self.obs_Me_rel_raw.extend(buf_obs_Me_rel)
                    self.pred_Me_rel_raw.extend(buf_pred_Me_rel)
                    self.non_linear_ped.extend(buf_nlp)
                    num_peds_in_seq.append(cnt)
                    self.tyID.append({
                        "old":    [f_year, f_name, idx],
                        "tydate": [add[i][0] for i in range(idx, idx + self.seq_len)],
                    })

        self.num_seq = len(self.tyID)
        cum = np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = list(zip([0] + cum[:-1], cum))
        logger.info(f"Loaded {self.num_seq} sequences")

    def _read_file(self, path: str, delim: str) -> dict:
        data, add = [], []
        with open(path, encoding="utf-8", errors="ignore") as f:
            raw_lines = f.readlines()
        for line in raw_lines:
            line = line.strip()
            if not line or line.startswith(("#", "//", "-", "=")):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                int(parts[0])
            except ValueError:
                continue
            try:
                frame_id  = float(parts[0])
                lon_norm  = float(parts[1])
                lat_norm  = float(parts[2])
                pres_norm = float(parts[3])
                wnd_norm  = float(parts[4])
                date      = parts[5]
                name      = parts[6]
                add.append([date, name])
                data.append([frame_id, 1.0, lon_norm, lat_norm, pres_norm, wnd_norm])
            except (ValueError, IndexError):
                continue
        return {
            "main":     np.asarray(data, dtype=np.float32) if data else np.zeros((0, 6), dtype=np.float32),
            "addition": add,
        }

    def _poly_fit(self, traj, tlen, threshold):
        t  = np.linspace(0, tlen - 1, tlen)
        rx = np.polyfit(t, traj[0, -tlen:], 2, full=True)[1]
        ry = np.polyfit(t, traj[1, -tlen:], 2, full=True)[1]
        return 1.0 if (len(rx) > 0 and rx[0] + ry[0] >= threshold) else 0.0

    def _normalize_data3d(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        for c in range(DATA3D_CH):
            ch = arr[:, :, c]
            ch[ch > _DATA3D_SENTINEL_LARGE] = np.nan
            if c in _DATA3D_SENTINEL_ZERO_CHANNELS:
                ch[ch == 0.0] = np.nan
            if c == _DATA3D_SST_CHANNEL:
                ch[ch < _SST_VALID_MIN] = _SST_FILL_K
            if np.any(np.isnan(ch)):
                valid_vals = ch[~np.isnan(ch)]
                fill_val = (float(np.median(valid_vals)) if len(valid_vals) > 0
                            else float(DATA3D_MEAN[c]))
                ch[np.isnan(ch)] = fill_val
            arr[:, :, c] = (ch - DATA3D_MEAN[c]) / (DATA3D_STD[c] + 1e-6)
        return np.clip(arr, -5.0, 5.0)

    def _load_data3d_file(self, path: str):
        try:
            if path.endswith(".npy"):
                arr = np.load(path).astype(np.float32)
            elif path.endswith(".nc") and HAS_NC:
                with nc.Dataset(path) as ds:
                    keys = list(ds.variables.keys())
                    arr  = np.array(ds.variables[keys[-1]][:]).astype(np.float32)
            else:
                return None
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            if arr.ndim == 3:
                if arr.shape[0] == DATA3D_CH:
                    arr = arr.transpose(1, 2, 0)
                H, W, C = arr.shape
                if H != DATA3D_H or W != DATA3D_W:
                    if HAS_CV2:
                        arr = cv2.resize(arr, (DATA3D_W, DATA3D_H))
                    else:
                        arr = arr[:DATA3D_H, :DATA3D_W, :]
                        if arr.shape[0] < DATA3D_H:
                            arr = np.pad(arr, ((0, DATA3D_H - arr.shape[0]), (0, 0), (0, 0)))
                        if arr.shape[1] < DATA3D_W:
                            arr = np.pad(arr, ((0, 0), (0, DATA3D_W - arr.shape[1]), (0, 0)))
                if arr.shape[2] < DATA3D_CH:
                    arr = np.concatenate([
                        arr,
                        np.zeros((DATA3D_H, DATA3D_W, DATA3D_CH - arr.shape[2]),
                                 dtype=np.float32),
                    ], axis=2)
                arr = arr[:, :, :DATA3D_CH]
                return self._normalize_data3d(arr)
        except Exception as e:
            logger.debug(f"Data3d load error {path}: {e}")
        return None

    def img_read(self, year, ty_name, timestamp) -> torch.Tensor:
        folder = os.path.join(self.data3d_path, str(year), str(ty_name))
        if not os.path.exists(folder):
            return torch.zeros(DATA3D_H, DATA3D_W, DATA3D_CH)
        prefix = f"WP{year}{ty_name}_{timestamp}"
        for ext in (".npy", ".nc"):
            p = os.path.join(folder, prefix + ext)
            if os.path.exists(p):
                arr = self._load_data3d_file(p)
                if arr is not None:
                    return torch.from_numpy(arr).float()
        try:
            for fname in sorted(os.listdir(folder)):
                if timestamp in fname and fname.endswith((".npy", ".nc")):
                    arr = self._load_data3d_file(os.path.join(folder, fname))
                    if arr is not None:
                        return torch.from_numpy(arr).float()
        except Exception:
            pass
        return torch.zeros(DATA3D_H, DATA3D_W, DATA3D_CH)

    def _load_env_npy(self, year, ty_name, timestamp):
        """
        FIX-DATA-21: Remap keys, set already_normed flag, THEN call
                     env_data_processing() which respects the flag.
        """
        folder = os.path.join(self.env_path, str(year), str(ty_name))
        if os.path.exists(folder):
            candidates = []
            for fname in [f"WP{year}{ty_name}_{timestamp}.npy", f"{timestamp}.npy"]:
                p = os.path.join(folder, fname)
                if os.path.exists(p):
                    candidates.append(p)
            if not candidates:
                try:
                    candidates = [
                        os.path.join(folder, f)
                        for f in os.listdir(folder)
                        if timestamp in f and f.endswith(".npy")
                    ]
                except Exception:
                    pass

            for p in candidates:
                try:
                    raw = np.load(p, allow_pickle=True).item()
                    if not isinstance(raw, dict):
                        continue

                    remapped = dict(raw)

                    # Remap GPH500 "_n" keys → chuẩn, đặt flag
                    has_old_gph = False
                    for old_k, new_k in _NPY_KEY_REMAP.items():
                        if old_k in remapped and new_k not in remapped:
                            remapped[new_k] = remapped.pop(old_k)
                            has_old_gph = True

                    # Remap u/v500 "_n" keys
                    for old_k, new_k in _NPY_U500_KEY_REMAP.items():
                        if old_k in remapped and new_k not in remapped:
                            remapped[new_k] = remapped.pop(old_k)

                    # FIX-DATA-21: Đặt flag TRƯỚC khi gọi processing
                    if has_old_gph:
                        remapped["gph500_already_normed"] = True

                    return env_data_processing(remapped)
                except Exception as e:
                    logger.debug(f"env npy load error {p}: {e}")

        # CSV fallback
        if self._csv_env_lookup:
            yr_str = str(year)
            ts_str = str(timestamp)
            name_strip  = str(ty_name).lstrip("0") or "0"
            name_padded = str(ty_name).zfill(4)
            name_padded2 = str(ty_name).zfill(2)
            for name_try in (str(ty_name), name_strip, name_padded, name_padded2):
                raw_dict = self._csv_env_lookup.get((yr_str, name_try, ts_str))
                if raw_dict is not None:
                    return env_data_processing(dict(raw_dict))

        return None

    def _get_env_features(self, year, ty_name, dates, obs_traj, obs_Me):
        T         = len(dates)
        all_feats = []
        prev_speed = None

        for t in range(T):
            env_npy = self._load_env_npy(year, ty_name, dates[t])
            feat    = build_env_features_one_step(
                lon_norm  = float(obs_traj[0, t]),
                lat_norm  = float(obs_traj[1, t]),
                wind_norm = float(obs_Me[1, t]),
                pres_norm = float(obs_Me[0, t]),
                timestamp = dates[t],
                env_npy   = env_npy,
                prev_speed_kmh = prev_speed,
            )
            all_feats.append(feat)
            if isinstance(env_npy, dict):
                mv = float(env_npy.get("move_velocity", 0.0) or 0.0)
                prev_speed = mv if mv != -1 else 0.0

        env_out = {}
        for key in ENV_FEATURE_DIMS:
            dim  = ENV_FEATURE_DIMS[key]
            rows = []
            for feat in all_feats:
                v = feat.get(key, [0.0] * dim)
                t = torch.tensor(v, dtype=torch.float)
                if t.numel() < dim:
                    t = F.pad(t, (0, dim - t.numel()))
                rows.append(t[:dim])
            env_out[key] = torch.stack(rows, dim=0)
        return env_out

    def _embed_time(self, date_list):
        rows = []
        for d in date_list:
            try:
                rows.append([
                    (float(d[:4]) - 1949) / 70.0 - 0.5,
                    (float(d[4:6]) - 1)   / 11.0 - 0.5,
                    (float(d[6:8]) - 1)   / 30.0 - 0.5,
                    float(d[8:10])         / 18.0 - 0.5,
                ])
            except Exception:
                rows.append([0.0, 0.0, 0.0, 0.0])
        return torch.tensor(rows, dtype=torch.float).t().unsqueeze(0)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        if self.num_seq == 0:
            raise IndexError("Empty dataset")
        s, e   = self.seq_start_end[index]
        info   = self.tyID[index]
        year   = str(info["old"][0])
        tyname = str(info["old"][1])
        dates  = info["tydate"]

        imgs    = [self.img_read(year, tyname, ts) for ts in dates[:self.obs_len]]
        img_obs = torch.stack(imgs, dim=0)
        img_pred = torch.zeros(self.pred_len, DATA3D_H, DATA3D_W, DATA3D_CH)

        obs_traj    = torch.stack([self.obs_traj_raw[i]    for i in range(s, e)])
        pred_traj   = torch.stack([self.pred_traj_raw[i]   for i in range(s, e)])
        obs_rel     = torch.stack([self.obs_rel_raw[i]     for i in range(s, e)])
        pred_rel    = torch.stack([self.pred_rel_raw[i]    for i in range(s, e)])
        obs_Me      = torch.stack([self.obs_Me_raw[i]      for i in range(s, e)])
        pred_Me     = torch.stack([self.pred_Me_raw[i]     for i in range(s, e)])
        obs_Me_rel  = torch.stack([self.obs_Me_rel_raw[i]  for i in range(s, e)])
        pred_Me_rel = torch.stack([self.pred_Me_rel_raw[i] for i in range(s, e)])

        n    = e - s
        nlp  = [self.non_linear_ped[i] for i in range(s, e)]
        mask = torch.ones(n, self.seq_len)

        obs_traj_np = obs_traj[0].numpy()
        obs_Me_np   = obs_Me[0].numpy()

        cache_key = (year, tyname, tuple(dates[:self.obs_len]))
        if cache_key not in self.env_cache:
            self.env_cache[cache_key] = self._get_env_features(
                year, tyname, dates[:self.obs_len], obs_traj_np, obs_Me_np)
        env_out = self.env_cache[cache_key]

        return [
            obs_traj, pred_traj, obs_rel, pred_rel, nlp, mask,
            obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
            self._embed_time(dates[:self.obs_len]),
            self._embed_time(dates[self.obs_len:]),
            img_obs, img_pred, env_out, info,
        ]
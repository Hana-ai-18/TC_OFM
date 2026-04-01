# # """
# # Model/data/trajectoriesWithMe_unet_training.py  ── v11
# # ======================================================
# # TC trajectory dataset — TRAINING VERSION.

# # FIXES vs v10:
# #   FIX-A  Buffer ped tensors trước, chỉ flush vào raw lists SAU KHI check
# #          cnt >= min_ped.  Trước đây tensor được append trước check → nếu
# #          min_ped > 1 mà cnt < min_ped thì seq_start_end bị lệch index,
# #          __getitem__ đọc sai range [s, e).

# #   FIX-B  Thay ceil(.../ skip) bằng floor((N - seq_len) / skip) + 1 để
# #          tính đúng số sliding-window với skip > 1.
# #          ceil có thể trả về n_seq lớn hơn thực tế; vòng lặp phải break
# #          sớm (lãng phí) và trong một số edge-case có thể bỏ sót window
# #          cuối cùng khi len(frames) - seq_len không chia hết cho skip.

# # Data3d: 81×81×13 tensor.  Env: 90-dim feature vector.
# # """
# # from __future__ import annotations

# # import logging
# # import math
# # import os

# # import numpy as np
# # import torch
# # import torch.nn.functional as F
# # from torch.utils.data import Dataset

# # try:
# #     import cv2
# #     HAS_CV2 = True
# # except ImportError:
# #     HAS_CV2 = False

# # try:
# #     import netCDF4 as nc
# #     HAS_NC = True
# # except ImportError:
# #     HAS_NC = False

# # from Model.env_net_transformer_gphsplit import (
# #     bearing_to_scs_center_onehot, dist_to_scs_boundary_onehot,
# #     delta_velocity_onehot, intensity_class_onehot,
# #     build_env_features_one_step, feat_to_tensor, ENV_FEATURE_DIMS,
# # )

# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # DATA3D_H  = 81
# # DATA3D_W  = 81
# # DATA3D_CH = 13

# # DATA3D_MEAN = np.array([
# #     12439.46, 5843.14, 1482.47, 752.80,
# #     -0.52, 0.27, -0.34, -0.86,
# #     0.25, 1.76, 1.34, 0.94, 300.95,
# # ], dtype=np.float32)

# # DATA3D_STD = np.array([
# #     91.59, 50.55, 29.42, 28.49,
# #     8.97, 4.73, 2.98, 2.75,
# #     5.37, 2.29, 2.21, 2.68, 3.05,
# # ], dtype=np.float32)


# # # ── env_data_processing ───────────────────────────────────────────────────────

# # def env_data_processing(env_dict: dict) -> dict:
# #     if not isinstance(env_dict, dict):
# #         return {}
# #     cleaned = {}
# #     for k, v in env_dict.items():
# #         if isinstance(v, (list, np.ndarray)):
# #             cleaned[k] = v
# #         else:
# #             cleaned[k] = 0.0 if v == -1 else v
# #     return cleaned


# # # ── seq_collate ───────────────────────────────────────────────────────────────

# # def seq_collate(data):
# #     (obs_traj, pred_traj, obs_rel, pred_rel,
# #      nlp, mask, obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
# #      obs_date, pred_date, img_obs, img_pred, env_data_raw, tyID) = zip(*data)

# #     def traj_TBC(lst):
# #         cat = torch.cat(lst, dim=0)
# #         return cat.permute(2, 0, 1)

# #     obs_traj_out    = traj_TBC(obs_traj)
# #     pred_traj_out   = traj_TBC(pred_traj)
# #     obs_rel_out     = traj_TBC(obs_rel)
# #     pred_rel_out    = traj_TBC(pred_rel)
# #     obs_Me_out      = traj_TBC(obs_Me)
# #     pred_Me_out     = traj_TBC(pred_Me)
# #     obs_Me_rel_out  = traj_TBC(obs_Me_rel)
# #     pred_Me_rel_out = traj_TBC(pred_Me_rel)

# #     nlp_out = torch.tensor(
# #         [v for sl in nlp for v in (sl if hasattr(sl, "__iter__") else [sl])],
# #         dtype=torch.float,
# #     )
# #     mask_out = torch.cat(list(mask), dim=0).permute(1, 0)

# #     counts        = torch.tensor([t.shape[0] for t in obs_traj])
# #     cum           = torch.cumsum(counts, dim=0)
# #     starts        = torch.cat([torch.tensor([0]), cum[:-1]])
# #     seq_start_end = torch.stack([starts, cum], dim=1)

# #     img_obs_out  = torch.stack(list(img_obs), dim=0).permute(0, 4, 1, 2, 3).float()
# #     img_pred_out = torch.stack(list(img_pred), dim=0).permute(0, 4, 1, 2, 3).float()

# #     env_out = None
# #     valid_envs = [d for d in env_data_raw if isinstance(d, dict)]
# #     if valid_envs:
# #         env_out  = {}
# #         all_keys = set()
# #         for d in valid_envs:
# #             all_keys.update(d.keys())
# #         for key in all_keys:
# #             vals = []
# #             for d in env_data_raw:
# #                 if isinstance(d, dict) and key in d:
# #                     v = d[key]
# #                     v = torch.tensor(v, dtype=torch.float) if not torch.is_tensor(v) else v.float()
# #                     vals.append(v)
# #                 else:
# #                     ref = next((d[key] for d in valid_envs if key in d), None)
# #                     if ref is not None:
# #                         rt = torch.tensor(ref, dtype=torch.float) if not torch.is_tensor(ref) else ref.float()
# #                         vals.append(torch.zeros_like(rt))
# #                     else:
# #                         vals.append(torch.zeros(1))
# #             try:
# #                 env_out[key] = torch.stack(vals, dim=0)
# #             except Exception:
# #                 try:
# #                     mx     = max(v.numel() for v in vals)
# #                     padded = [F.pad(v.flatten(), (0, mx - v.numel())) for v in vals]
# #                     env_out[key] = torch.stack(padded, dim=0)
# #                 except Exception:
# #                     pass

# #     return (
# #         obs_traj_out, pred_traj_out, obs_rel_out, pred_rel_out,
# #         nlp_out, mask_out, seq_start_end,
# #         obs_Me_out, pred_Me_out, obs_Me_rel_out, pred_Me_rel_out,
# #         img_obs_out, img_pred_out, env_out, None, list(tyID),
# #     )


# # # ── TrajectoryDataset ─────────────────────────────────────────────────────────

# # class TrajectoryDataset(Dataset):
# #     """TC trajectory dataset for TCND_VN."""

# #     def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,
# #                  threshold=0.002, min_ped=1, delim=" ", other_modal="gph",
# #                  test_year=None, type="train", split=None, is_test=False,
# #                  **kwargs):
# #         super().__init__()

# #         # 'split' kwarg (v11 loader) takes precedence over 'type' (legacy)
# #         dtype = split if split is not None else type

# #         if isinstance(data_dir, dict):
# #             root  = data_dir["root"]
# #             dtype = data_dir.get("type", dtype)
# #         else:
# #             root = data_dir
# #         if is_test and dtype not in ("val", "test"):
# #             dtype = "test"

# #         root = os.path.abspath(root)
# #         bn   = os.path.basename(root)
# #         if bn in ("train", "test", "val"):
# #             self.root_path = os.path.dirname(os.path.dirname(root))
# #         elif bn == "Data1d":
# #             self.root_path = os.path.dirname(root)
# #         else:
# #             self.root_path = root

# #         self.data1d_path = os.path.join(self.root_path, "Data1d", dtype)
# #         self.data3d_path = os.path.join(self.root_path, "Data3d")
# #         for env_name in ("Env_data", "ENV_DATA", "env_data"):
# #             candidate = os.path.join(self.root_path, env_name)
# #             if os.path.exists(candidate):
# #                 self.env_path = candidate
# #                 break
# #         else:
# #             self.env_path = os.path.join(self.root_path, "Env_data")

# #         logger.info(f"root ({dtype}) : {self.root_path}")
# #         self.obs_len    = obs_len
# #         self.pred_len   = pred_len
# #         self.seq_len    = obs_len + pred_len
# #         self.skip       = skip
# #         self.modal_name = other_modal

# #         if not os.path.exists(self.data1d_path):
# #             logger.error(f"Missing Data1d: {self.data1d_path}")
# #             self.num_seq       = 0
# #             self.seq_start_end = []
# #             self.tyID          = []
# #             return

# #         all_files = [
# #             os.path.join(self.data1d_path, f)
# #             for f in os.listdir(self.data1d_path)
# #             if f.endswith(".txt") and (test_year is None or str(test_year) in f)
# #         ]
# #         logger.info(f"{len(all_files)} Data1d files (year={test_year})")

# #         self.obs_traj_raw    = []
# #         self.pred_traj_raw   = []
# #         self.obs_Me_raw      = []
# #         self.pred_Me_raw     = []
# #         self.obs_rel_raw     = []
# #         self.pred_rel_raw    = []
# #         self.obs_Me_rel_raw  = []
# #         self.pred_Me_rel_raw = []
# #         self.non_linear_ped  = []
# #         self.tyID            = []
# #         num_peds_in_seq      = []
# #         self.env_cache       = {}

# #         for path in all_files:
# #             base   = os.path.splitext(os.path.basename(path))[0]
# #             parts  = base.split("_")
# #             f_year = parts[0] if parts else "unknown"
# #             f_name = parts[1] if len(parts) > 1 else base

# #             d    = self._read_file(path, delim)
# #             data = d["main"]
# #             add  = d["addition"]
# #             if len(data) < self.seq_len:
# #                 continue

# #             frames     = np.unique(data[:, 0]).tolist()
# #             frame_data = [data[data[:, 0] == f] for f in frames]

# #             # FIX-B: correct sliding-window count for any skip value.
# #             #   floor((N - seq_len) / skip) + 1
# #             # ceil((N - seq_len + 1) / skip) from v10 could overshoot by 1
# #             # when (N - seq_len) is not divisible by skip, causing an extra
# #             # loop iteration that immediately hits the break guard.
# #             n_frames = len(frames)
# #             if n_frames < self.seq_len:
# #                 continue
# #             n_seq = (n_frames - self.seq_len) // skip + 1

# #             for idx in range(0, n_seq * skip, skip):
# #                 # Defensive guard — should never trigger with the fixed n_seq
# #                 if idx + self.seq_len > len(frame_data):
# #                     break

# #                 seg  = np.concatenate(frame_data[idx: idx + self.seq_len])
# #                 peds = np.unique(seg[:, 1])

# #                 # FIX-A: buffer tensors per-ped; only flush to shared raw
# #                 # lists AFTER cnt >= min_ped is confirmed.  In v10 tensors
# #                 # were appended unconditionally inside the pid loop, so
# #                 # rejected sequences (cnt < min_ped) left orphaned tensors
# #                 # in the raw lists, making seq_start_end indices wrong.
# #                 buf_obs_traj    = []
# #                 buf_pred_traj   = []
# #                 buf_obs_rel     = []
# #                 buf_pred_rel    = []
# #                 buf_obs_Me      = []
# #                 buf_pred_Me     = []
# #                 buf_obs_Me_rel  = []
# #                 buf_pred_Me_rel = []
# #                 buf_nlp         = []
# #                 cnt             = 0

# #                 for pid in peds:
# #                     ps = seg[seg[:, 1] == pid]
# #                     if len(ps) != self.seq_len:
# #                         continue
# #                     ps_t = np.transpose(ps[:, 2:])
# #                     rel  = np.zeros_like(ps_t)
# #                     rel[:, 1:] = ps_t[:, 1:] - ps_t[:, :-1]

# #                     buf_obs_traj.append(torch.from_numpy(ps_t[:2, :obs_len]).float())
# #                     buf_pred_traj.append(torch.from_numpy(ps_t[:2, obs_len:]).float())
# #                     buf_obs_rel.append(torch.from_numpy(rel[:2, :obs_len]).float())
# #                     buf_pred_rel.append(torch.from_numpy(rel[:2, obs_len:]).float())
# #                     buf_obs_Me.append(torch.from_numpy(ps_t[2:, :obs_len]).float())
# #                     buf_pred_Me.append(torch.from_numpy(ps_t[2:, obs_len:]).float())
# #                     buf_obs_Me_rel.append(torch.from_numpy(rel[2:, :obs_len]).float())
# #                     buf_pred_Me_rel.append(torch.from_numpy(rel[2:, obs_len:]).float())
# #                     buf_nlp.append(self._poly_fit(ps_t, pred_len, threshold))
# #                     cnt += 1

# #                 if cnt >= min_ped:
# #                     # Flush buffer → shared raw lists atomically
# #                     self.obs_traj_raw.extend(buf_obs_traj)
# #                     self.pred_traj_raw.extend(buf_pred_traj)
# #                     self.obs_rel_raw.extend(buf_obs_rel)
# #                     self.pred_rel_raw.extend(buf_pred_rel)
# #                     self.obs_Me_raw.extend(buf_obs_Me)
# #                     self.pred_Me_raw.extend(buf_pred_Me)
# #                     self.obs_Me_rel_raw.extend(buf_obs_Me_rel)
# #                     self.pred_Me_rel_raw.extend(buf_pred_Me_rel)
# #                     self.non_linear_ped.extend(buf_nlp)

# #                     num_peds_in_seq.append(cnt)
# #                     self.tyID.append({
# #                         "old":    [f_year, f_name, idx],
# #                         "tydate": [add[i][0] for i in range(idx, idx + self.seq_len)],
# #                     })

# #         self.num_seq = len(self.tyID)
# #         cum = np.cumsum(num_peds_in_seq).tolist()
# #         self.seq_start_end = list(zip([0] + cum[:-1], cum))
# #         logger.info(f"Loaded {self.num_seq} sequences")

# #     # ── internal helpers ───────────────────────────────────────────────────────

# #     def _read_file(self, path: str, delim: str) -> dict:
# #         data, add = [], []
# #         with open(path, encoding="utf-8", errors="ignore") as f:
# #             raw_lines = f.readlines()
# #         for line in raw_lines:
# #             line = line.strip()
# #             if not line or line.startswith(("#", "//", "-", "=")):
# #                 continue
# #             parts = line.split()
# #             if len(parts) < 7:
# #                 continue
# #             try:
# #                 int(parts[0])
# #             except ValueError:
# #                 continue
# #             try:
# #                 frame_id  = float(parts[0])
# #                 lon_norm  = float(parts[1])
# #                 lat_norm  = float(parts[2])
# #                 pres_norm = float(parts[3])
# #                 wnd_norm  = float(parts[4])
# #                 date      = parts[5]
# #                 name      = parts[6]
# #                 add.append([date, name])
# #                 data.append([frame_id, 1.0, lon_norm, lat_norm, pres_norm, wnd_norm])
# #             except (ValueError, IndexError):
# #                 continue
# #         return {
# #             "main":     np.asarray(data, dtype=np.float32) if data else np.zeros((0, 6), dtype=np.float32),
# #             "addition": add,
# #         }

# #     def _poly_fit(self, traj, tlen, threshold):
# #         t  = np.linspace(0, tlen - 1, tlen)
# #         rx = np.polyfit(t, traj[0, -tlen:], 2, full=True)[1]
# #         ry = np.polyfit(t, traj[1, -tlen:], 2, full=True)[1]
# #         return 1.0 if (len(rx) > 0 and rx[0] + ry[0] >= threshold) else 0.0

# #     def _normalize_data3d(self, arr: np.ndarray) -> np.ndarray:
# #         for c in range(DATA3D_CH):
# #             arr[:, :, c] = (arr[:, :, c] - DATA3D_MEAN[c]) / (DATA3D_STD[c] + 1e-6)
# #         return np.clip(arr, -5.0, 5.0)

# #     def _load_data3d_file(self, path: str):
# #         try:
# #             if path.endswith(".npy"):
# #                 arr = np.load(path).astype(np.float32)
# #             elif path.endswith(".nc") and HAS_NC:
# #                 with nc.Dataset(path) as ds:
# #                     keys = list(ds.variables.keys())
# #                     arr  = np.array(ds.variables[keys[-1]][:]).astype(np.float32)
# #             else:
# #                 return None
# #             if arr.ndim == 2:
# #                 arr = arr[:, :, np.newaxis]
# #             if arr.ndim == 3:
# #                 if arr.shape[0] == DATA3D_CH:
# #                     arr = arr.transpose(1, 2, 0)
# #                 H, W, C = arr.shape
# #                 if H != DATA3D_H or W != DATA3D_W:
# #                     if HAS_CV2:
# #                         arr = cv2.resize(arr, (DATA3D_W, DATA3D_H))
# #                     else:
# #                         arr = arr[:DATA3D_H, :DATA3D_W, :]
# #                         if arr.shape[0] < DATA3D_H:
# #                             arr = np.pad(arr, ((0, DATA3D_H - arr.shape[0]), (0, 0), (0, 0)))
# #                         if arr.shape[1] < DATA3D_W:
# #                             arr = np.pad(arr, ((0, 0), (0, DATA3D_W - arr.shape[1]), (0, 0)))
# #                 if arr.shape[2] < DATA3D_CH:
# #                     arr = np.concatenate([
# #                         arr,
# #                         np.zeros((DATA3D_H, DATA3D_W, DATA3D_CH - arr.shape[2]), dtype=np.float32),
# #                     ], axis=2)
# #                 arr = arr[:, :, :DATA3D_CH]
# #                 return self._normalize_data3d(arr)
# #         except Exception as e:
# #             logger.debug(f"Data3d load error {path}: {e}")
# #         return None

# #     def img_read(self, year, ty_name, timestamp) -> torch.Tensor:
# #         folder = os.path.join(self.data3d_path, str(year), str(ty_name))
# #         if not os.path.exists(folder):
# #             return torch.zeros(DATA3D_H, DATA3D_W, DATA3D_CH)
# #         prefix = f"WP{year}{ty_name}_{timestamp}"
# #         for ext in (".npy", ".nc"):
# #             p = os.path.join(folder, prefix + ext)
# #             if os.path.exists(p):
# #                 arr = self._load_data3d_file(p)
# #                 if arr is not None:
# #                     return torch.from_numpy(arr).float()
# #         try:
# #             for fname in sorted(os.listdir(folder)):
# #                 if timestamp in fname and fname.endswith((".npy", ".nc")):
# #                     arr = self._load_data3d_file(os.path.join(folder, fname))
# #                     if arr is not None:
# #                         return torch.from_numpy(arr).float()
# #         except Exception:
# #             pass
# #         return torch.zeros(DATA3D_H, DATA3D_W, DATA3D_CH)

# #     def _load_env_npy(self, year, ty_name, timestamp):
# #         folder = os.path.join(self.env_path, str(year), str(ty_name))
# #         if not os.path.exists(folder):
# #             return None
# #         for fname in [f"WP{year}{ty_name}_{timestamp}.npy", f"{timestamp}.npy"]:
# #             p = os.path.join(folder, fname)
# #             if os.path.exists(p):
# #                 try:
# #                     return env_data_processing(np.load(p, allow_pickle=True).item())
# #                 except Exception:
# #                     pass
# #         try:
# #             cands = [f for f in os.listdir(folder) if timestamp in f and f.endswith(".npy")]
# #             if cands:
# #                 return env_data_processing(
# #                     np.load(os.path.join(folder, cands[0]), allow_pickle=True).item())
# #         except Exception:
# #             pass
# #         return None

# #     def _get_env_features(self, year, ty_name, dates, obs_traj, obs_Me):
# #         T          = len(dates)
# #         all_feats  = []
# #         prev_speed = None
# #         for t in range(T):
# #             env_npy = self._load_env_npy(year, ty_name, dates[t])
# #             feat    = build_env_features_one_step(
# #                 lon_norm=float(obs_traj[0, t]), lat_norm=float(obs_traj[1, t]),
# #                 wind_norm=float(obs_Me[1, t]), timestamp=dates[t],
# #                 env_npy=env_npy, prev_speed_kmh=prev_speed,
# #             )
# #             all_feats.append(feat)
# #             if isinstance(env_npy, dict):
# #                 mv = float(env_npy.get("move_velocity", 0.0) or 0.0)
# #                 prev_speed = mv if mv != -1 else 0.0
# #         env_out = {}
# #         for key in ENV_FEATURE_DIMS:
# #             dim  = ENV_FEATURE_DIMS[key]
# #             rows = []
# #             for feat in all_feats:
# #                 v = feat.get(key, [0.0] * dim)
# #                 t = torch.tensor(v, dtype=torch.float)
# #                 if t.numel() < dim:
# #                     t = F.pad(t, (0, dim - t.numel()))
# #                 rows.append(t[:dim])
# #             env_out[key] = torch.stack(rows, dim=0)
# #         return env_out

# #     def _embed_time(self, date_list):
# #         rows = []
# #         for d in date_list:
# #             try:
# #                 rows.append([
# #                     (float(d[:4]) - 1949) / 70.0 - 0.5,
# #                     (float(d[4:6]) - 1)   / 11.0 - 0.5,
# #                     (float(d[6:8]) - 1)   / 30.0 - 0.5,
# #                     float(d[8:10])         / 18.0 - 0.5,
# #                 ])
# #             except Exception:
# #                 rows.append([0.0, 0.0, 0.0, 0.0])
# #         return torch.tensor(rows, dtype=torch.float).t().unsqueeze(0)

# #     # ── Dataset interface ──────────────────────────────────────────────────────

# #     def __len__(self):
# #         return self.num_seq

# #     def __getitem__(self, index):
# #         if self.num_seq == 0:
# #             raise IndexError("Empty dataset")
# #         s, e   = self.seq_start_end[index]
# #         info   = self.tyID[index]
# #         year   = str(info["old"][0])
# #         tyname = str(info["old"][1])
# #         dates  = info["tydate"]

# #         imgs     = [self.img_read(year, tyname, ts) for ts in dates[:self.obs_len]]
# #         img_obs  = torch.stack(imgs, dim=0)
# #         img_pred = torch.zeros(self.pred_len, DATA3D_H, DATA3D_W, DATA3D_CH)

# #         obs_traj    = torch.stack([self.obs_traj_raw[i]    for i in range(s, e)])
# #         pred_traj   = torch.stack([self.pred_traj_raw[i]   for i in range(s, e)])
# #         obs_rel     = torch.stack([self.obs_rel_raw[i]     for i in range(s, e)])
# #         pred_rel    = torch.stack([self.pred_rel_raw[i]    for i in range(s, e)])
# #         obs_Me      = torch.stack([self.obs_Me_raw[i]      for i in range(s, e)])
# #         pred_Me     = torch.stack([self.pred_Me_raw[i]     for i in range(s, e)])
# #         obs_Me_rel  = torch.stack([self.obs_Me_rel_raw[i]  for i in range(s, e)])
# #         pred_Me_rel = torch.stack([self.pred_Me_rel_raw[i] for i in range(s, e)])

# #         n    = e - s
# #         nlp  = [self.non_linear_ped[i] for i in range(s, e)]
# #         mask = torch.ones(n, self.seq_len)

# #         obs_traj_np = obs_traj[0].numpy()
# #         obs_Me_np   = obs_Me[0].numpy()
# #         if index not in self.env_cache:
# #             self.env_cache[index] = self._get_env_features(
# #                 year, tyname, dates[:self.obs_len], obs_traj_np, obs_Me_np)
# #         env_out = self.env_cache[index]

# #         return [
# #             obs_traj, pred_traj, obs_rel, pred_rel, nlp, mask,
# #             obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
# #             self._embed_time(dates[:self.obs_len]),
# #             self._embed_time(dates[self.obs_len:]),
# #             img_obs, img_pred, env_out, info,
# #         ]

# """
# Model/data/trajectoriesWithMe_unet_training.py  ── v12
# ======================================================
# TC trajectory dataset — TRAINING VERSION.

# FIXES vs v11:
#   FIX-DATA-1  _normalize_data3d: sentinel values in 3D arrays.
#               Before normalization, replace:
#                 - arr[:,:,c] > 20000 → np.nan → median imputation
#                 - arr[:,:,c] == 0    → np.nan → median imputation
#               Prevents sentinel ±5 clips polluting FNO input.

#   FIX-DATA-2  env_data_processing: SST sentinel fix.
#               SST=0 in env_npy → replace with 298.0 K (≈25°C).
#               Affects 887 rows across 2019–2025 storms.

#   FIX-DATA-3  _load_env_npy: additional sentinel guard for
#               gph500_mean/center values outside valid range.
#               Valid: ~27–35. Sentinels: -29.5 or ~88.
#               Replace sentinel gph500 with None (→ code returns 0.0).

# Kept from v11:
#   FIX-A  Buffer ped tensors before min_ped check.
#   FIX-B  Correct sliding-window count with skip.

# Data3d: 81×81×13 tensor.  Env: 90-dim feature vector.
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

# DATA3D_MEAN = np.array([
#     12439.46, 5843.14, 1482.47, 752.80,
#     -0.52, 0.27, -0.34, -0.86,
#     0.25, 1.76, 1.34, 0.94, 300.95,
# ], dtype=np.float32)

# DATA3D_STD = np.array([
#     91.59, 50.55, 29.42, 28.49,
#     8.97, 4.73, 2.98, 2.75,
#     5.37, 2.29, 2.21, 2.68, 3.05,
# ], dtype=np.float32)

# # Sentinel thresholds for DATA3D channels
# # Channels 0–5 are gph500/u500/v500 (mean+center): large sentinels >20000, zero sentinels =0
# _DATA3D_SENTINEL_LARGE = 20000.0
# _DATA3D_SENTINEL_ZERO_CHANNELS = {0, 1, 2, 3, 4, 5}  # gph500/u500/v500 mean+center

# # SST channel index in DATA3D (channel 12, 0-indexed)
# _DATA3D_SST_CHANNEL = 12
# _SST_VALID_MIN = 270.0   # ~-3°C (sea water won't freeze much below this)
# _SST_FILL_K    = 298.0   # ~25°C, typical tropical SST


# # ── env_data_processing ───────────────────────────────────────────────────────

# def env_data_processing(env_dict: dict) -> dict:
#     """
#     Clean env_npy dictionary.
#     FIX-DATA-2: Replace SST=0 with 298.0K (tropical fill value).
#     """
#     if not isinstance(env_dict, dict):
#         return {}
#     cleaned = {}
#     for k, v in env_dict.items():
#         if isinstance(v, (list, np.ndarray)):
#             cleaned[k] = v
#         else:
#             cleaned[k] = 0.0 if v == -1 else v

#     # FIX-DATA-2: SST sentinel fix
#     for sst_key in ("sst_mean", "sst_center", "sst"):
#         if sst_key in cleaned:
#             val = cleaned[sst_key]
#             if val is None or val == 0 or (isinstance(val, float) and val < _SST_VALID_MIN):
#                 cleaned[sst_key] = _SST_FILL_K

#     # FIX-DATA-3: gph500 sentinel guard in env_dict
#     # Valid gph500 (pre-scaled): ~27–35. Sentinels: -29.5 (zero) or ~88 (large).
#     for gph_key in ("gph500_mean", "gph500_center"):
#         if gph_key in cleaned:
#             val = cleaned[gph_key]
#             if val is not None and isinstance(val, (int, float)):
#                 if val < 25.0 or val > 50.0:
#                     # Sentinel value — set to None so build_env_features uses 0.0
#                     cleaned[gph_key] = None

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
#                  **kwargs):
#         super().__init__()

#         # 'split' kwarg (v11 loader) takes precedence over 'type' (legacy)
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
#         for env_name in ("Env_data", "ENV_DATA", "env_data"):
#             candidate = os.path.join(self.root_path, env_name)
#             if os.path.exists(candidate):
#                 self.env_path = candidate
#                 break
#         else:
#             self.env_path = os.path.join(self.root_path, "Env_data")

#         logger.info(f"root ({dtype}) : {self.root_path}")
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
#         self.env_cache       = {}

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

#     # ── internal helpers ───────────────────────────────────────────────────────

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
#         """
#         FIX-DATA-1: Before z-score normalization, replace sentinel values
#         in DATA3D channels 0–5 (gph500/u500/v500 mean+center).
#           - Large sentinel (> 20000): set to np.nan → impute with valid median
#           - Zero sentinel (== 0) in channels 0–5: set to np.nan → impute
#         This prevents sentinel values from becoming ±5 clips that
#         corrupt FNO input and backprop gradients.
#         """
#         arr = arr.copy()
#         for c in range(DATA3D_CH):
#             ch = arr[:, :, c]
#             if c in _DATA3D_SENTINEL_ZERO_CHANNELS:
#                 # Large sentinel
#                 ch[ch > _DATA3D_SENTINEL_LARGE] = np.nan
#                 # Zero sentinel (only for wind/gph channels, 0 is unphysical)
#                 ch[ch == 0.0] = np.nan
#             # SST channel: 0 means missing, fill with tropical value
#             if c == _DATA3D_SST_CHANNEL:
#                 ch[ch < _SST_VALID_MIN] = _SST_FILL_K
#             # Impute NaN with channel median
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
#                 return self._normalize_data3d(arr)  # FIX-DATA-1 applied here
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
#         folder = os.path.join(self.env_path, str(year), str(ty_name))
#         if not os.path.exists(folder):
#             return None
#         for fname in [f"WP{year}{ty_name}_{timestamp}.npy", f"{timestamp}.npy"]:
#             p = os.path.join(folder, fname)
#             if os.path.exists(p):
#                 try:
#                     # FIX-DATA-2+3: env_data_processing now cleans SST and gph500 sentinels
#                     return env_data_processing(np.load(p, allow_pickle=True).item())
#                 except Exception:
#                     pass
#         try:
#             cands = [f for f in os.listdir(folder) if timestamp in f and f.endswith(".npy")]
#             if cands:
#                 return env_data_processing(
#                     np.load(os.path.join(folder, cands[0]), allow_pickle=True).item())
#         except Exception:
#             pass
#         return None

#     def _get_env_features(self, year, ty_name, dates, obs_traj, obs_Me):
#         T          = len(dates)
#         all_feats  = []
#         prev_speed = None
#         for t in range(T):
#             env_npy = self._load_env_npy(year, ty_name, dates[t])
#             feat    = build_env_features_one_step(
#                 lon_norm=float(obs_traj[0, t]), lat_norm=float(obs_traj[1, t]),
#                 wind_norm=float(obs_Me[1, t]), timestamp=dates[t],
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

#     # ── Dataset interface ──────────────────────────────────────────────────────

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
#         if index not in self.env_cache:
#             self.env_cache[index] = self._get_env_features(
#                 year, tyname, dates[:self.obs_len], obs_traj_np, obs_Me_np)
#         env_out = self.env_cache[index]

#         return [
#             obs_traj, pred_traj, obs_rel, pred_rel, nlp, mask,
#             obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
#             self._embed_time(dates[:self.obs_len]),
#             self._embed_time(dates[self.obs_len:]),
#             img_obs, img_pred, env_out, info,
#         ]

"""
Model/data/trajectoriesWithMe_unet_training.py  ── v13
======================================================
TC trajectory dataset — TRAINING VERSION.

FIXES vs v12:
  FIX-DATA-4  _normalize_data3d: zero-sentinel applied to wrong channels.
              v12 used _DATA3D_SENTINEL_ZERO_CHANNELS = {0,1,2,3,4,5} which
              included channels 4 (v500_center) and 5 (u500_center).
              Wind components (u/v) at storm center CAN legitimately be 0
              (e.g. storm in weak-shear environment). Treating 0 as sentinel
              for wind channels corrupts valid data.
              Fix: Only channel 0 (gph500_mean) is physically incapable of
              being 0. Channels 1–5 (u/v wind components) may be 0 validly.
              _DATA3D_SENTINEL_ZERO_CHANNELS = {0}

  FIX-CACHE-1 env_cache keyed by sequence index, not stable identity.
              DataLoader shuffle means the same index maps to different
              sequences across epochs → stale / cross-contaminated cache.
              Fix: cache key = (year, tyname, tuple(dates[:obs_len]))
              which is unique per sequence regardless of shuffle order.

Kept from v12:
  FIX-DATA-1  Sentinel values (> 20000) in DATA3D channels → NaN → median.
  FIX-DATA-2  SST=0 fill with 298K.
  FIX-DATA-3  gph500 sentinel guard in env_dict.
  FIX-A  Buffer ped tensors before min_ped check.
  FIX-B  Correct sliding-window count with skip.

Data3d: 81×81×13 tensor.  Env: 90-dim feature vector.
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

DATA3D_MEAN = np.array([
    12439.46, 5843.14, 1482.47, 752.80,
    480.31,   # ch4 v500_center — separate stats (v13)
    0.27, -0.34, -0.86,
    0.25, 1.76, 1.34, 0.94, 300.95,
], dtype=np.float32)

DATA3D_STD = np.array([
    91.59, 50.55, 29.42, 28.49,
    24.17,   # ch4 v500_center — separate stats (v13)
    4.73, 2.98, 2.75,
    5.37, 2.29, 2.21, 2.68, 3.05,
], dtype=np.float32)

# ── Sentinel thresholds for DATA3D channels ───────────────────────────────────
# Large sentinel: any channel with value > 20000 is bad data.
_DATA3D_SENTINEL_LARGE = 20000.0

# FIX-DATA-4: Only gph500_mean (channel 0) is physically incapable of being 0.
# Wind components (channels 1-5, u/v at mean and center) CAN validly be 0.
# v12 incorrectly flagged channels {0,1,2,3,4,5} — this corrupted wind data.
_DATA3D_SENTINEL_ZERO_CHANNELS = {0}   # gph500_mean only

# SST channel index in DATA3D (channel 12, 0-indexed)
_DATA3D_SST_CHANNEL = 12
_SST_VALID_MIN = 270.0   # ~-3°C (sea water won't freeze much below this)
_SST_FILL_K    = 298.0   # ~25°C, typical tropical SST


# ── env_data_processing ───────────────────────────────────────────────────────

def env_data_processing(env_dict: dict) -> dict:
    """
    Clean env_npy dictionary.
    FIX-DATA-2: Replace SST=0 with 298.0K (tropical fill value).
    FIX-DATA-3: Guard gph500 sentinels (out of valid range 25–50).
    """
    if not isinstance(env_dict, dict):
        return {}
    cleaned = {}
    for k, v in env_dict.items():
        if isinstance(v, (list, np.ndarray)):
            cleaned[k] = v
        else:
            cleaned[k] = 0.0 if v == -1 else v

    # FIX-DATA-2: SST sentinel fix
    for sst_key in ("sst_mean", "sst_center", "sst"):
        if sst_key in cleaned:
            val = cleaned[sst_key]
            if val is None or val == 0 or (isinstance(val, float) and val < _SST_VALID_MIN):
                cleaned[sst_key] = _SST_FILL_K

    # FIX-DATA-3: gph500 sentinel guard in env_dict
    # Valid gph500 (pre-scaled): ~27–35. Sentinels: -29.5 (zero) or ~88 (large).
    for gph_key in ("gph500_mean", "gph500_center"):
        if gph_key in cleaned:
            val = cleaned[gph_key]
            if val is not None and isinstance(val, (int, float)):
                if val < 25.0 or val > 50.0:
                    cleaned[gph_key] = None

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

    img_obs_out  = torch.stack(list(img_obs), dim=0).permute(0, 4, 1, 2, 3).float()
    img_pred_out = torch.stack(list(img_pred), dim=0).permute(0, 4, 1, 2, 3).float()

    env_out = None
    valid_envs = [d for d in env_data_raw if isinstance(d, dict)]
    if valid_envs:
        env_out  = {}
        all_keys = set()
        for d in valid_envs:
            all_keys.update(d.keys())
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
    """TC trajectory dataset for TCND_VN."""

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,
                 threshold=0.002, min_ped=1, delim=" ", other_modal="gph",
                 test_year=None, type="train", split=None, is_test=False,
                 **kwargs):
        super().__init__()

        # 'split' kwarg (v11 loader) takes precedence over 'type' (legacy)
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
        for env_name in ("Env_data", "ENV_DATA", "env_data"):
            candidate = os.path.join(self.root_path, env_name)
            if os.path.exists(candidate):
                self.env_path = candidate
                break
        else:
            self.env_path = os.path.join(self.root_path, "Env_data")

        logger.info(f"root ({dtype}) : {self.root_path}")
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

        # FIX-CACHE-1: env_cache keyed by stable (year, tyname, dates) tuple
        # NOT by dataset index — index→sequence mapping changes with shuffle.
        self.env_cache: dict[tuple, dict] = {}

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

            n_frames = len(frames)
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

                    buf_obs_traj.append(torch.from_numpy(ps_t[:2, :obs_len]).float())
                    buf_pred_traj.append(torch.from_numpy(ps_t[:2, obs_len:]).float())
                    buf_obs_rel.append(torch.from_numpy(rel[:2, :obs_len]).float())
                    buf_pred_rel.append(torch.from_numpy(rel[:2, obs_len:]).float())
                    buf_obs_Me.append(torch.from_numpy(ps_t[2:, :obs_len]).float())
                    buf_pred_Me.append(torch.from_numpy(ps_t[2:, obs_len:]).float())
                    buf_obs_Me_rel.append(torch.from_numpy(rel[2:, :obs_len]).float())
                    buf_pred_Me_rel.append(torch.from_numpy(rel[2:, obs_len:]).float())
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

    # ── internal helpers ───────────────────────────────────────────────────────

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
        """
        FIX-DATA-1 (v12): Before z-score normalization, replace large sentinel
        values (> 20000) in all channels with NaN → impute with valid median.

        FIX-DATA-4 (v13): Zero-sentinel only for channel 0 (gph500_mean).
        v12 applied zero-sentinel to channels {0,1,2,3,4,5} which includes
        wind components that CAN legitimately be 0. Only gph500 is physically
        incapable of being 0 Pa.
        """
        arr = arr.copy()
        for c in range(DATA3D_CH):
            ch = arr[:, :, c]
            # Large sentinel: always bad regardless of channel
            ch[ch > _DATA3D_SENTINEL_LARGE] = np.nan
            # Zero sentinel: ONLY for gph500_mean (channel 0) — not wind channels
            if c in _DATA3D_SENTINEL_ZERO_CHANNELS:
                ch[ch == 0.0] = np.nan
            # SST channel: values below freezing are fill values
            if c == _DATA3D_SST_CHANNEL:
                ch[ch < _SST_VALID_MIN] = _SST_FILL_K
            # Impute NaN with channel median
            if np.any(np.isnan(ch)):
                valid_vals = ch[~np.isnan(ch)]
                fill_val = float(np.median(valid_vals)) if len(valid_vals) > 0 else float(DATA3D_MEAN[c])
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
                        np.zeros((DATA3D_H, DATA3D_W, DATA3D_CH - arr.shape[2]), dtype=np.float32),
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
        folder = os.path.join(self.env_path, str(year), str(ty_name))
        if not os.path.exists(folder):
            return None
        for fname in [f"WP{year}{ty_name}_{timestamp}.npy", f"{timestamp}.npy"]:
            p = os.path.join(folder, fname)
            if os.path.exists(p):
                try:
                    return env_data_processing(np.load(p, allow_pickle=True).item())
                except Exception:
                    pass
        try:
            cands = [f for f in os.listdir(folder) if timestamp in f and f.endswith(".npy")]
            if cands:
                return env_data_processing(
                    np.load(os.path.join(folder, cands[0]), allow_pickle=True).item())
        except Exception:
            pass
        return None

    def _get_env_features(self, year, ty_name, dates, obs_traj, obs_Me):
        T          = len(dates)
        all_feats  = []
        prev_speed = None
        for t in range(T):
            env_npy = self._load_env_npy(year, ty_name, dates[t])
            feat    = build_env_features_one_step(
                lon_norm=float(obs_traj[0, t]), lat_norm=float(obs_traj[1, t]),
                wind_norm=float(obs_Me[1, t]), timestamp=dates[t],
                env_npy=env_npy, prev_speed_kmh=prev_speed,
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

    # ── Dataset interface ──────────────────────────────────────────────────────

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

        imgs     = [self.img_read(year, tyname, ts) for ts in dates[:self.obs_len]]
        img_obs  = torch.stack(imgs, dim=0)
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

        # FIX-CACHE-1: stable cache key = (year, tyname, obs_dates_tuple)
        # DataLoader shuffle changes dataset index → sequence mapping each epoch,
        # so caching by index caused stale / cross-contaminated lookups.
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

# """
# Model/env_net_transformer_gphsplit.py  ── v19
# ===================================================================
# FIXES vs v18:

#   FIX-ENV-19  [CRITICAL] GPH500 = 0 khi dùng Env_data .npy được build
#               bởi build_env_data_scs_v10.py (pipeline cũ).

#               Pipeline cũ lưu gph500 với key "gph500_mean_n" đã
#               pre-normalized bằng mean=5900/std=200 → range ~[-28, 2].
#               Sau khi trajectoriesWithMe v18 remap sang key "gph500_mean"
#               và set cờ "gph500_already_normed"=True, hàm này nhận value
#               đã normalize → KHÔNG được z-score lần 2.

#               Fix trong build_env_features_one_step():
#                 if env_npy.get("gph500_already_normed", False):
#                     val = float(raw)   # dùng trực tiếp, chỉ clip [-5,5]
#                 else:
#                     val = (raw - mean_val) / (std_val + 1e-8)  # z-score bình thường

#               Kết quả: GPH500 features sẽ có giá trị thực thay vì = 0.

# Kept from v18 / earlier:
#   FIX-ENV-18  gph500 normalisation constants /380 dam:
#               _GPH500_MEAN=33.64, _GPH500_STD=7.08 (dùng cho CSV path
#               và .npy mới, không dùng cho .npy cũ đã pre-normed)
#   FIX-ENV-16B u500/v500 stored as boolean flags [0,1], no z-score
#   FIX-ENV-15  pres_norm parameter
#   FIX-ENV-14  wind /150.0
#   Feature vector — 90 dims total (unchanged).
# """
# from __future__ import annotations

# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ── Feature dimensions ────────────────────────────────────────────────────────
# ENV_FEATURE_DIMS: dict[str, int] = {
#     "wind":                   1,
#     "intensity_class":        6,
#     "move_velocity":          1,
#     "month":                 12,
#     "location_lon_scs":      10,
#     "location_lat_scs":       8,
#     "bearing_to_scs_center": 16,
#     "dist_to_scs_boundary":   5,
#     "delta_velocity":         5,
#     "history_direction12":    8,
#     "history_direction24":    8,
#     "history_inte_change24":  4,
#     "gph500_mean":            1,
#     "gph500_center":          1,
#     "u500_mean":              1,
#     "u500_center":            1,
#     "v500_mean":              1,
#     "v500_center":            1,
# }

# ENV_DIM_TOTAL = sum(ENV_FEATURE_DIMS.values())   # 90

# _D3D_KEYS = {
#     "gph500_mean", "gph500_center",
#     "u500_mean",   "u500_center",
#     "v500_mean",   "v500_center",
# }
# ENV_1D_DIM = sum(d for k, d in ENV_FEATURE_DIMS.items() if k not in _D3D_KEYS)  # 84
# ENV_3D_DIM = ENV_DIM_TOTAL - ENV_1D_DIM   # 6

# # ── SCS geography constants ───────────────────────────────────────────────────
# SCS_BBOX        = dict(lon_min=100.0, lon_max=125.0, lat_min=5.0, lat_max=20.0)
# SCS_CENTER      = (112.5, 12.5)
# SCS_DIAGONAL_KM = 3100.0
# BOUNDARY_THRESHOLDS = [0.05, 0.15, 0.30]
# DELTA_VEL_BINS      = [-20.0, -5.0, 5.0, 20.0]

# # ── Normalisation constants ───────────────────────────────────────────────────
# # Dùng cho CSV path và .npy mới (raw dam ~[27-90])
# _GPH500_MEAN  = 33.64
# _GPH500_STD   =  7.08
# _GPH500C_MEAN = 32.84
# _GPH500C_STD  =  1.03

# # Sentinel: valid range ~27-95 dam. Value < 25 là sentinel (-29.5).
# _GPH500_SENTINEL_LO = 25.0
# _GPH500_SENTINEL_HI = 95.0

# # FIX-ENV-16B (kept): u500/v500 env values are boolean flags (0.0 or 1.0)
# _UV500_SENTINEL_HI = 20000.0

# _WIND_NORM_DENOM = 150.0
# _INTENSITY_THRESHOLDS_MS = [17.2, 32.7, 41.5, 51.5, 65.0]


# # ── Feature-engineering helpers ───────────────────────────────────────────────

# def _pos_onehot(val: float, lo: float, hi: float, n: int) -> list[int]:
#     idx = int((val - lo) / (hi - lo) * n)
#     v   = [0] * n
#     v[max(0, min(n - 1, idx))] = 1
#     return v


# def bearing_to_scs_center_onehot(lon_deg: float, lat_deg: float) -> list[int]:
#     c_lon, c_lat = SCS_CENTER
#     mid_lat = math.radians((lat_deg + c_lat) / 2.0)
#     dx      = (c_lon - lon_deg) * math.cos(mid_lat)
#     dy      = c_lat - lat_deg
#     bearing = math.degrees(math.atan2(dx, dy)) % 360.0
#     idx     = int((bearing + 11.25) / 22.5) % 16
#     v       = [0] * 16
#     v[idx]  = 1
#     return v


# def dist_to_scs_boundary_onehot(lon_deg: float, lat_deg: float) -> list[int]:
#     lo, hi = SCS_BBOX["lon_min"], SCS_BBOX["lon_max"]
#     la, lb = SCS_BBOX["lat_min"], SCS_BBOX["lat_max"]
#     if not (lo <= lon_deg <= hi and la <= lat_deg <= lb):
#         return [1, 0, 0, 0, 0]
#     d_min = min(
#         (lon_deg - lo) * 111.0 * math.cos(math.radians(lat_deg)),
#         (hi - lon_deg) * 111.0 * math.cos(math.radians(lat_deg)),
#         (lat_deg - la) * 111.0,
#         (lb - lat_deg) * 111.0,
#     )
#     r = d_min / SCS_DIAGONAL_KM
#     if   r < BOUNDARY_THRESHOLDS[0]: idx = 4
#     elif r < BOUNDARY_THRESHOLDS[1]: idx = 3
#     elif r < BOUNDARY_THRESHOLDS[2]: idx = 2
#     else:                             idx = 1
#     v      = [0] * 5
#     v[idx] = 1
#     return v


# def delta_velocity_onehot(delta_km_h: float) -> list[int]:
#     bins = DELTA_VEL_BINS
#     if   delta_km_h <= bins[0]: idx = 0
#     elif delta_km_h <= bins[1]: idx = 1
#     elif delta_km_h <= bins[2]: idx = 2
#     elif delta_km_h <= bins[3]: idx = 3
#     else:                        idx = 4
#     v      = [0] * 5
#     v[idx] = 1
#     return v


# def intensity_class_onehot(wind_ms: float) -> list[int]:
#     thresholds = _INTENSITY_THRESHOLDS_MS
#     idx = sum(wind_ms >= t for t in thresholds)
#     v   = [0] * 6
#     v[min(idx, 5)] = 1
#     return v


# def build_env_features_one_step(
#     lon_norm: float,
#     lat_norm: float,
#     wind_norm: float,
#     timestamp: str,
#     env_npy,
#     prev_speed_kmh,
#     pres_norm: float = 0.0,
# ) -> dict:
#     """
#     Build 90-dim env feature vector for one timestep.

#     FIX-ENV-19: gph500 handling theo nguồn dữ liệu:
#       - .npy cũ (build_env_data_scs_v10): env_npy["gph500_already_normed"]=True
#         → value đã normalized, chỉ clip [-5,5], không z-score lại.
#       - CSV path: env_npy["gph500_already_normed"]=False (default)
#         → value là raw dam ~[27-90], apply sentinel guard + z-score bình thường.

#     FIX-ENV-16B (kept): u500/v500 stored as boolean flags [0,1], no z-score.
#     """
#     lon_deg = (lon_norm * 50.0 + 1800.0) / 10.0
#     lat_deg = (lat_norm * 50.0) / 10.0
#     wind_kt = wind_norm * 25.0 + 40.0
#     wind_ms = wind_kt * 0.5144

#     feat: dict = {}

#     feat["wind"]            = [wind_ms / _WIND_NORM_DENOM]
#     feat["intensity_class"] = intensity_class_onehot(wind_ms)

#     mv = 0.0
#     if isinstance(env_npy, dict):
#         v  = env_npy.get("move_velocity", 0.0)
#         mv = 0.0 if (v is None or v == -1) else float(v)
#     feat["move_velocity"] = [mv / 1219.84]

#     try:    mi = int(timestamp[4:6]) - 1
#     except: mi = 0
#     oh        = [0] * 12
#     oh[max(0, min(11, mi))] = 1
#     feat["month"] = oh

#     feat["location_lon_scs"]      = _pos_onehot(lon_deg, 100.0, 125.0, 10)
#     feat["location_lat_scs"]      = _pos_onehot(lat_deg,   5.0,  25.0,  8)
#     feat["bearing_to_scs_center"] = bearing_to_scs_center_onehot(lon_deg, lat_deg)
#     feat["dist_to_scs_boundary"]  = dist_to_scs_boundary_onehot(lon_deg, lat_deg)

#     delta = (mv - prev_speed_kmh) if prev_speed_kmh is not None else 0.0
#     feat["delta_velocity"] = delta_velocity_onehot(delta)

#     for key, dim in [("history_direction12", 8), ("history_direction24", 8)]:
#         if isinstance(env_npy, dict) and key in env_npy:
#             v = env_npy[key]
#             v = list(v)[:dim] if hasattr(v, "__iter__") else [-1] * dim
#             v = v + [0] * (dim - len(v))
#             v = [-1] * dim if all(x == -1 for x in v) else v
#         else:
#             v = [-1] * dim
#         feat[key] = v

#     key = "history_inte_change24"
#     if isinstance(env_npy, dict) and key in env_npy:
#         v = env_npy[key]
#         v = list(v)[:4] if hasattr(v, "__iter__") else [-1] * 4
#         v = v + [0] * (4 - len(v))
#         v = [-1] * 4 if all(x == -1 for x in v) else v
#     else:
#         v = [-1] * 4
#     feat["history_inte_change24"] = v

#     # ── GPH500 ─────────────────────────────────────────────────────────────────
#     # FIX-ENV-19: Phân biệt .npy cũ (already_normed) vs CSV/raw (z-score cần thiết)
#     already_normed = isinstance(env_npy, dict) and env_npy.get("gph500_already_normed", False)

#     for k, mean_val, std_val in [
#         ("gph500_mean",   _GPH500_MEAN,  _GPH500_STD),
#         ("gph500_center", _GPH500C_MEAN, _GPH500C_STD),
#     ]:
#         val = 0.0
#         if isinstance(env_npy, dict) and k in env_npy:
#             raw = env_npy[k]
#             try:
#                 raw = float(raw)
#             except (TypeError, ValueError):
#                 raw = None

#             if raw is None or math.isnan(raw):
#                 val = 0.0
#             elif already_normed:
#                 # FIX-ENV-19: .npy cũ — value đã normalized bởi build_env_data
#                 # (mean=5900, std=200). Range ~[-28, 2].
#                 # Chỉ clip để tránh outlier cực đoan, không z-score lại.
#                 val = float(np.clip(raw, -5.0, 5.0))
#             else:
#                 # CSV hoặc .npy mới — raw dam, cần sentinel check + z-score
#                 if raw < _GPH500_SENTINEL_LO or raw > _GPH500_SENTINEL_HI:
#                     val = 0.0
#                 else:
#                     val = (raw - mean_val) / (std_val + 1e-8)
#                     val = float(np.clip(val, -5.0, 5.0))

#         feat[k] = [val]

#     # ── U500 / V500 ────────────────────────────────────────────────────────────
#     # FIX-ENV-16B (kept): boolean availability flags, no z-score.
#     # .npy cũ lưu "u500_mean_n" (normalized /U500_NORM → [-1,1]) đã được
#     # remap sang "u500_mean" bởi _load_env_npy. Clip [0,1] là hợp lý.
#     for k in ("u500_mean", "u500_center", "v500_mean", "v500_center"):
#         if isinstance(env_npy, dict) and k in env_npy:
#             raw = env_npy[k]
#             try:
#                 val = float(np.clip(float(raw), 0.0, 1.0))
#             except (TypeError, ValueError):
#                 val = 0.0
#         else:
#             val = 0.0
#         feat[k] = [val]

#     return feat


# def feat_to_tensor(feat: dict) -> torch.Tensor:
#     parts = []
#     for key in ENV_FEATURE_DIMS:
#         dim = ENV_FEATURE_DIMS[key]
#         v   = feat.get(key, None)
#         if v is None:
#             parts.append(torch.zeros(dim))
#             continue
#         if not isinstance(v, (list, torch.Tensor)):
#             v = [float(v)]
#         t = torch.tensor(v, dtype=torch.float)
#         if t.numel() < dim:
#             t = F.pad(t, (0, dim - t.numel()))
#         parts.append(t[:dim])
#     return torch.cat(parts)


# def build_env_vector(env_data, B: int, T: int,
#                      device: torch.device) -> torch.Tensor:
#     parts = []
#     for key, dim in ENV_FEATURE_DIMS.items():
#         slot = torch.zeros(B, T, dim, device=device)
#         if env_data is None or key not in env_data:
#             parts.append(slot)
#             continue
#         v = env_data[key]
#         if v is None:
#             parts.append(slot)
#             continue
#         if not torch.is_tensor(v):
#             try:
#                 v = torch.tensor(v, dtype=torch.float, device=device)
#             except Exception:
#                 parts.append(slot)
#                 continue
#         v = v.float().to(device)
#         try:
#             if v.dim() == 0:
#                 slot = v.expand(B, T, 1) if dim == 1 else slot
#             elif v.dim() == 1:
#                 if v.numel() == dim:
#                     slot = v.view(1, 1, dim).expand(B, T, dim)
#             elif v.dim() == 2:
#                 if v.shape == (B, T):
#                     slot = v.unsqueeze(-1).expand(-1, -1, dim)
#                 elif v.shape == (B, dim):
#                     slot = v.unsqueeze(1).expand(-1, T, -1)
#                 elif v.shape[0] == T and v.shape[1] == dim:
#                     slot = v.unsqueeze(0).expand(B, -1, -1)
#             elif v.dim() == 3:
#                 if v.shape[:2] == (B, T):
#                     d_in = v.shape[-1]
#                     slot = v[..., :dim] if d_in >= dim else F.pad(v, (0, dim - d_in))
#                 elif v.shape[0] == T:
#                     s    = v.permute(1, 0, 2)[..., :dim]
#                     slot = s if s.shape[-1] == dim else F.pad(s, (0, dim - s.shape[-1]))
#         except Exception:
#             pass
#         parts.append(slot.float())
#     return torch.cat(parts, dim=-1)


# # ── Env-T-Net ─────────────────────────────────────────────────────────────────

# class Env_net(nn.Module):
#     def __init__(self, obs_len: int = 8, embed_dim: int = 16, d_model: int = 64):
#         super().__init__()
#         self.obs_len = obs_len
#         self.d_model = d_model
#         H1, H2, H3  = 64, 32, 64

#         self.mlp_env_1d = nn.Sequential(
#             nn.Linear(ENV_1D_DIM, H1), nn.LayerNorm(H1), nn.GELU(),
#             nn.Linear(H1, H1),         nn.LayerNorm(H1), nn.GELU(),
#         )
#         self.cnn_env_3d = nn.Sequential(
#             nn.Conv1d(ENV_3D_DIM, H2, 3, padding=1), nn.BatchNorm1d(H2), nn.GELU(),
#             nn.Conv1d(H2, H2, 3, padding=1),         nn.BatchNorm1d(H2), nn.GELU(),
#         )
#         self.mlp_fusion = nn.Sequential(
#             nn.Linear(H1 + H2, H3), nn.LayerNorm(H3), nn.GELU(),
#         )
#         self.pos_enc_env = nn.Parameter(torch.randn(1, obs_len, H3) * 0.02)
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=H3, nhead=4, dim_feedforward=H3 * 2,
#             dropout=0.1, activation="gelu", batch_first=True,
#         )
#         self.transformer_env = nn.TransformerEncoder(enc_layer, num_layers=2)
#         self.out_proj = nn.Sequential(nn.Linear(H3, d_model), nn.LayerNorm(d_model))

#     def forward(self, env_data, gph: torch.Tensor) -> tuple:
#         if gph.dim() == 4:
#             gph = gph.unsqueeze(1)
#         B, C, T, H, W = gph.shape
#         device = gph.device

#         feat    = build_env_vector(env_data, B, T, device)
#         feat_1d = feat[:, :, :ENV_1D_DIM]
#         feat_3d = feat[:, :, ENV_1D_DIM:]

#         e_1d  = self.mlp_env_1d(feat_1d)
#         e_3d  = self.cnn_env_3d(feat_3d.permute(0, 2, 1)).permute(0, 2, 1)
#         e_env = self.mlp_fusion(torch.cat([e_1d, e_3d], dim=-1))

#         t_actual = min(T, self.pos_enc_env.shape[1])
#         e_env    = e_env[:, :t_actual, :] + self.pos_enc_env[:, :t_actual, :]
#         e_env_time = self.transformer_env(e_env)
#         ctx = self.out_proj(e_env_time[:, -1, :])
#         return ctx, 0, 0

"""
Model/env_net_transformer_gphsplit.py  ── v20
===================================================================
FIXES vs v19:

  FIX-ENV-20A [CRITICAL] u500/v500 luôn = 0 vì key mismatch:
              trajectoriesWithMe remap "u500_mean_n" → "u500_raw_mean"
              nhưng build_env_features đọc "u500_mean" → miss → 0.
              Fix: đọc cả "u500_mean" VÀ "u500_raw_mean", ưu tiên raw.
              Normalize u/v500 raw (m/s) về [-1,1] bằng /30.0.

  FIX-ENV-20B [HIGH] Loại bỏ boolean flag (0/1) cho u/v500.
              v16B-v19 dùng boolean flag vô nghĩa. Bây giờ dùng
              giá trị normalized thực sự của steering flow.
              Nếu has raw: val = clip(raw/30.0, -1, 1)
              Nếu has normalized (pre-normed .npy): dùng trực tiếp
              Nếu không có: val = 0.0

  FIX-ENV-20C [MEDIUM] gph500 đã normalized (.npy cũ): clip [-3,3]
              thay vì [-5,5] để giảm outlier ảnh hưởng.

Kept from v19:
  FIX-ENV-19  gph500 already_normed handling
  FIX-ENV-18  gph500 normalisation /380 dam
  FIX-ENV-15  pres_norm parameter
  FIX-ENV-14  wind /150.0
  Feature vector — 90 dims total (unchanged).
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Feature dimensions ────────────────────────────────────────────────────────
ENV_FEATURE_DIMS: dict[str, int] = {
    "wind":                   1,
    "intensity_class":        6,
    "move_velocity":          1,
    "month":                 12,
    "location_lon_scs":      10,
    "location_lat_scs":       8,
    "bearing_to_scs_center": 16,
    "dist_to_scs_boundary":   5,
    "delta_velocity":         5,
    "history_direction12":    8,
    "history_direction24":    8,
    "history_inte_change24":  4,
    "gph500_mean":            1,
    "gph500_center":          1,
    "u500_mean":              1,
    "u500_center":            1,
    "v500_mean":              1,
    "v500_center":            1,
}

ENV_DIM_TOTAL = sum(ENV_FEATURE_DIMS.values())   # 90

_D3D_KEYS = {
    "gph500_mean", "gph500_center",
    "u500_mean",   "u500_center",
    "v500_mean",   "v500_center",
}
ENV_1D_DIM = sum(d for k, d in ENV_FEATURE_DIMS.items() if k not in _D3D_KEYS)  # 84
ENV_3D_DIM = ENV_DIM_TOTAL - ENV_1D_DIM   # 6

# ── SCS geography constants ───────────────────────────────────────────────────
SCS_BBOX        = dict(lon_min=100.0, lon_max=125.0, lat_min=5.0, lat_max=20.0)
SCS_CENTER      = (112.5, 12.5)
SCS_DIAGONAL_KM = 3100.0
BOUNDARY_THRESHOLDS = [0.05, 0.15, 0.30]
DELTA_VEL_BINS      = [-20.0, -5.0, 5.0, 20.0]

# ── Normalisation constants ───────────────────────────────────────────────────
# ── GPH500 ──────────────────────────────────────────────────────
# _GPH500_MEAN_M  = 5880.0
# _GPH500_STD_M   =  150.0
# _GPH500_MIN_M   = 4000.0   # sentinel lo
# _GPH500_MAX_M   = 7000.0   # sentinel hi
# Đổi constants về đơn vị dam (decameters), vì .npy lưu dam
_GPH500_MIN_DAM = 400.0   # ~4000m / 10
_GPH500_MAX_DAM = 700.0   # ~7000m / 10
_GPH500_MEAN_DAM = 587.0  # ~5870m / 10
_GPH500_STD_DAM  = 8.0    # ~80m / 10


# FIX-ENV-20A/B: u/v500 raw normalization constant (m/s)
_UV500_NORM = 30.0   # clip raw u/v500 (m/s) to [-30,30] → [-1,1]

_WIND_NORM_DENOM = 150.0
_INTENSITY_THRESHOLDS_MS = [17.2, 32.7, 41.5, 51.5, 65.0]


# ── Feature-engineering helpers ───────────────────────────────────────────────

def _pos_onehot(val: float, lo: float, hi: float, n: int) -> list[int]:
    idx = int((val - lo) / (hi - lo) * n)
    v   = [0] * n
    v[max(0, min(n - 1, idx))] = 1
    return v


def bearing_to_scs_center_onehot(lon_deg: float, lat_deg: float) -> list[int]:
    c_lon, c_lat = SCS_CENTER
    mid_lat = math.radians((lat_deg + c_lat) / 2.0)
    dx      = (c_lon - lon_deg) * math.cos(mid_lat)
    dy      = c_lat - lat_deg
    bearing = math.degrees(math.atan2(dx, dy)) % 360.0
    idx     = int((bearing + 11.25) / 22.5) % 16
    v       = [0] * 16
    v[idx]  = 1
    return v


def dist_to_scs_boundary_onehot(lon_deg: float, lat_deg: float) -> list[int]:
    lo, hi = SCS_BBOX["lon_min"], SCS_BBOX["lon_max"]
    la, lb = SCS_BBOX["lat_min"], SCS_BBOX["lat_max"]
    if not (lo <= lon_deg <= hi and la <= lat_deg <= lb):
        return [1, 0, 0, 0, 0]
    d_min = min(
        (lon_deg - lo) * 111.0 * math.cos(math.radians(lat_deg)),
        (hi - lon_deg) * 111.0 * math.cos(math.radians(lat_deg)),
        (lat_deg - la) * 111.0,
        (lb - lat_deg) * 111.0,
    )
    r = d_min / SCS_DIAGONAL_KM
    if   r < BOUNDARY_THRESHOLDS[0]: idx = 4
    elif r < BOUNDARY_THRESHOLDS[1]: idx = 3
    elif r < BOUNDARY_THRESHOLDS[2]: idx = 2
    else:                             idx = 1
    v      = [0] * 5
    v[idx] = 1
    return v


def delta_velocity_onehot(delta_km_h: float) -> list[int]:
    bins = DELTA_VEL_BINS
    if   delta_km_h <= bins[0]: idx = 0
    elif delta_km_h <= bins[1]: idx = 1
    elif delta_km_h <= bins[2]: idx = 2
    elif delta_km_h <= bins[3]: idx = 3
    else:                        idx = 4
    v      = [0] * 5
    v[idx] = 1
    return v


def intensity_class_onehot(wind_ms: float) -> list[int]:
    thresholds = _INTENSITY_THRESHOLDS_MS
    idx = sum(wind_ms >= t for t in thresholds)
    v   = [0] * 6
    v[min(idx, 5)] = 1
    return v


# def _get_uv500_val(env_npy: dict, key_base: str, already_normed_uv: bool) -> float:
#     """
#     FIX-ENV-20A/B: Đọc u/v500 đúng cách.
    
#     Priority:
#     1. "{key_base}_raw_mean" → raw m/s từ CSV (FIX-DATA-24) → normalize /30
#     2. "{key_base}_mean" với already_normed_uv=True → .npy cũ đã normed → dùng trực tiếp
#     3. "{key_base}_mean" với already_normed_uv=False → boolean flag legacy → bỏ qua (→ 0)
#     4. Không có → 0.0
    
#     Ví dụ: key_base="u500" → đọc "u500_raw_mean" hoặc "u500_mean"
#     """
#     if not isinstance(env_npy, dict):
#         return 0.0

#     # Mapping: key_base → các key name có thể có
#     # "u500" → "u500_raw_mean", "u500_mean"
#     # "u500_center" → "u500_raw_center", "u500_center"
#     raw_key    = key_base + "_raw"   # e.g. "u500_mean_raw" → not correct
#     # Correct: raw key được lưu là "u500_raw_mean", "v500_raw_mean" etc
#     # key_base = "u500_mean" → base = "u500", sub = "mean"
#     # Let's parse properly
#     parts = key_base.rsplit("_", 1)  # ["u500", "mean"] or ["u500", "center"]
#     if len(parts) == 2:
#         uv_var, stat = parts  # e.g. "u500", "mean"
#         raw_key_csv = f"{uv_var}_raw_{stat}"  # "u500_raw_mean"
#     else:
#         raw_key_csv = None

#     # Priority 1: raw từ CSV
#     if raw_key_csv and raw_key_csv in env_npy:
#         raw = env_npy[raw_key_csv]
#         try:
#             raw = float(raw)
#             if math.isnan(raw) or raw == 0.0:
#                 return 0.0
#             return float(np.clip(raw / _UV500_RAW_NORM, -1.0, 1.0))
#         except (TypeError, ValueError):
#             pass

#     # Priority 2: key thường
#     if key_base in env_npy:
#         raw = env_npy[key_base]
#         try:
#             raw = float(raw)
#             if math.isnan(raw):
#                 return 0.0
#             if already_normed_uv:
#                 # .npy cũ: đã normalized bởi build_env (/U500_NORM=30) → range [-1,1]
#                 return float(np.clip(raw, -1.0, 1.0))
#             else:
#                 # CSV legacy: boolean flag 0.0 hoặc 1.0 → vô nghĩa → bỏ qua
#                 # Trả 0.0 thay vì dùng boolean
#                 return 0.0
#         except (TypeError, ValueError):
#             pass

#     return 0.0

# def _read_uv500_from_npy(env_npy: dict, key: str) -> float:
#     """
#     FIX-ENV-22: Đọc u/v500 từ .npy (đã normalized [-1,1] bởi /30.0).
#     Trả về 0.0 nếu missing hoặc has_data3d=False.
#     Clip [-1, 1] để đảm bảo không có outlier.
#     """
#     if not isinstance(env_npy, dict):
#         return 0.0
#     # has_data3d=False → tất cả u/v = 0.0 (fallback trong builder)
#     if not env_npy.get("has_data3d", True):
#         return 0.0
#     raw = env_npy.get(key, None)
#     if raw is None:
#         return 0.0
#     try:
#         v = float(raw)
#     except (TypeError, ValueError):
#         return 0.0
#     # Trong .npy: đã clip(-1,1) — giá trị hợp lệ trong [-1,1]
#     # Nếu = 0.0 → Data3d miss cho timestep này (builder fallback)
#     return float(np.clip(v, -1.0, 1.0))

def _read_uv500_from_npy(env_npy: dict, key: str) -> float:
    if not isinstance(env_npy, dict): return 0.0
    if not env_npy.get("has_data3d", True): return 0.0
    raw = env_npy.get(key, None)
    if raw is None: return 0.0
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    # FIX: raw là m/s (~±30), cần /30 rồi clip
    return float(np.clip(v / _UV500_NORM, -1.0, 1.0))
    # _UV500_NORM = 30.0 đã có sẵn trong file

def _read_uv500_from_csv(raw_val, norm: float = _UV500_NORM) -> float:
    """
    FIX-ENV-22: Normalize u/v500 raw từ CSV (m²/s² ~5843).
    Công thức giống build_env_data_scs_v10.py: clip(raw/30.0, -1, 1).
    """
    if raw_val is None:
        return 0.0
    try:
        v = float(raw_val)
    except (TypeError, ValueError):
        return 0.0
    if v <= 0.0 or v > 50000.0:   # sentinel/missing
        return 0.0
    return float(np.clip(v / norm, -1.0, 1.0))

def build_env_features_one_step(
    lon_norm: float,
    lat_norm: float,
    wind_norm: float,
    timestamp: str,
    env_npy,
    prev_speed_kmh,
    pres_norm: float = 0.0,
) -> dict:
    """
    Build 90-dim env feature vector for one timestep.

    FIX-ENV-20A/B: u/v500 đọc từ raw key CSV hoặc pre-normed .npy key.
    FIX-ENV-20C: gph500 already_normed clip to [-3,3].
    FIX-ENV-19:  gph500 already_normed handling.
    """
    lon_deg = (lon_norm * 50.0 + 1800.0) / 10.0
    lat_deg = (lat_norm * 50.0) / 10.0
    wind_kt = wind_norm * 25.0 + 40.0
    wind_ms = wind_kt * 0.5144

    feat: dict = {}

    # feat["wind"]            = [wind_ms / _WIND_NORM_DENOM]
    if isinstance(env_npy, dict) and "wind" in env_npy:
        feat["wind"] = [float(env_npy["wind"])]   # wnd_norm gốc, nhất quán Data1d
    else:
        feat["wind"] = [(wind_norm * 25.0 + 40.0) * 0.5144 / _WIND_NORM_DENOM]

    feat["intensity_class"] = intensity_class_onehot(wind_ms)

    mv = 0.0
    if isinstance(env_npy, dict):
        v  = env_npy.get("move_velocity", 0.0)
        mv = 0.0 if (v is None or v == -1) else float(v)
    feat["move_velocity"] = [mv / 1219.84]

    try:    mi = int(timestamp[4:6]) - 1
    except: mi = 0
    oh        = [0] * 12
    oh[max(0, min(11, mi))] = 1
    feat["month"] = oh

    feat["location_lon_scs"]      = _pos_onehot(lon_deg, 100.0, 125.0, 10)
    feat["location_lat_scs"]      = _pos_onehot(lat_deg,   5.0,  25.0,  8)
    feat["bearing_to_scs_center"] = bearing_to_scs_center_onehot(lon_deg, lat_deg)
    feat["dist_to_scs_boundary"]  = dist_to_scs_boundary_onehot(lon_deg, lat_deg)

    delta = (mv - prev_speed_kmh) if prev_speed_kmh is not None else 0.0
    feat["delta_velocity"] = delta_velocity_onehot(delta)

    for key, dim in [("history_direction12", 8), ("history_direction24", 8)]:
        if isinstance(env_npy, dict) and key in env_npy:
            v = env_npy[key]
            v = list(v)[:dim] if hasattr(v, "__iter__") else [-1] * dim
            v = v + [0] * (dim - len(v))
            v = [-1] * dim if all(x == -1 for x in v) else v
        else:
            v = [-1] * dim
        feat[key] = v

    key = "history_inte_change24"
    if isinstance(env_npy, dict) and key in env_npy:
        v = env_npy[key]
        v = list(v)[:4] if hasattr(v, "__iter__") else [-1] * 4
        v = v + [0] * (4 - len(v))
        v = [-1] * 4 if all(x == -1 for x in v) else v
    else:
        v = [-1] * 4
    feat["history_inte_change24"] = v
    
    # Trong build_env_features_one_step:
    # for feat_key, npy_key in [
    #     ("gph500_mean",   "gph500_mean"),
    #     ("gph500_center", "gph500_center"),
    # ]:
    #     val = 0.0
    #     if isinstance(env_npy, dict):
    #         has_3d = env_npy.get("has_data3d", False)
    #         if has_3d and npy_key in env_npy:
    #             try:
    #                 raw = float(env_npy[npy_key])
    #                 if _GPH500_MIN_M <= raw <= _GPH500_MAX_M:
    #                     # raw là geopotential height (m) → normalize trực tiếp
    #                     val = (raw - _GPH500_MEAN_M) / (_GPH500_STD_M + 1e-8)
    #                     val = float(np.clip(val, -3.0, 3.0))
    #                 # else: ngoài range → val = 0.0 (missing/sentinel)
    #             except (TypeError, ValueError):
    #                 val = 0.0
    #     feat[feat_key] = [val]
    for feat_key, npy_key in [("gph500_mean", "gph500_mean"), ("gph500_center", "gph500_center")]:
        val = 0.0
        if isinstance(env_npy, dict):
            has_3d = env_npy.get("has_data3d", False)
            if has_3d and npy_key in env_npy:
                try:
                    raw = float(env_npy[npy_key])
                    if _GPH500_MIN_DAM <= raw <= _GPH500_MAX_DAM:
                        val = (raw - _GPH500_MEAN_DAM) / (_GPH500_STD_DAM + 1e-8)
                        val = float(np.clip(val, -3.0, 3.0))
                    # else: missing/sentinel → val = 0.0
                except (TypeError, ValueError):
                    val = 0.0
        feat[feat_key] = [val]
    # ── U500 / V500 ──────────────────────────────────────────────
    for feat_key, npy_key in [
        ("u500_mean",   "u500_mean"),
        ("u500_center", "u500_center"),
        ("v500_mean",   "v500_mean"),
        ("v500_center", "v500_center"),
    ]:
        val = 0.0
        if isinstance(env_npy, dict):
            has_3d = env_npy.get("has_data3d", False)
            if has_3d and npy_key in env_npy:
                try:
                    raw = float(env_npy[npy_key])
                    # raw là m/s từ build_env_data_scs_v12 → ÷30 → [-1,1]
                    val = float(np.clip(raw / _UV500_NORM, -1.0, 1.0))
                except (TypeError, ValueError):
                    val = 0.0
        feat[feat_key] = [val]
    
    return feat


def feat_to_tensor(feat: dict) -> torch.Tensor:
    parts = []
    for key in ENV_FEATURE_DIMS:
        dim = ENV_FEATURE_DIMS[key]
        v   = feat.get(key, None)
        if v is None:
            parts.append(torch.zeros(dim))
            continue
        if not isinstance(v, (list, torch.Tensor)):
            v = [float(v)]
        t = torch.tensor(v, dtype=torch.float)
        if t.numel() < dim:
            t = F.pad(t, (0, dim - t.numel()))
        parts.append(t[:dim])
    return torch.cat(parts)


def build_env_vector(env_data, B: int, T: int,
                     device: torch.device) -> torch.Tensor:
    parts = []
    for key, dim in ENV_FEATURE_DIMS.items():
        slot = torch.zeros(B, T, dim, device=device)
        if env_data is None or key not in env_data:
            parts.append(slot)
            continue
        v = env_data[key]
        if v is None:
            parts.append(slot)
            continue
        if not torch.is_tensor(v):
            try:
                v = torch.tensor(v, dtype=torch.float, device=device)
            except Exception:
                parts.append(slot)
                continue
        v = v.float().to(device)
        try:
            if v.dim() == 0:
                slot = v.expand(B, T, 1) if dim == 1 else slot
            elif v.dim() == 1:
                if v.numel() == dim:
                    slot = v.view(1, 1, dim).expand(B, T, dim)
            elif v.dim() == 2:
                if v.shape == (B, T):
                    slot = v.unsqueeze(-1).expand(-1, -1, dim)
                elif v.shape == (B, dim):
                    slot = v.unsqueeze(1).expand(-1, T, -1)
                elif v.shape[0] == T and v.shape[1] == dim:
                    slot = v.unsqueeze(0).expand(B, -1, -1)
            elif v.dim() == 3:
                if v.shape[:2] == (B, T):
                    d_in = v.shape[-1]
                    slot = v[..., :dim] if d_in >= dim else F.pad(v, (0, dim - d_in))
                elif v.shape[0] == T:
                    s    = v.permute(1, 0, 2)[..., :dim]
                    slot = s if s.shape[-1] == dim else F.pad(s, (0, dim - s.shape[-1]))
        except Exception:
            pass
        parts.append(slot.float())
    return torch.cat(parts, dim=-1)


# ── Env-T-Net ─────────────────────────────────────────────────────────────────

class Env_net(nn.Module):
    def __init__(self, obs_len: int = 8, embed_dim: int = 16, d_model: int = 64):
        super().__init__()
        self.obs_len = obs_len
        self.d_model = d_model
        H1, H2, H3  = 64, 32, 64

        self.mlp_env_1d = nn.Sequential(
            nn.Linear(ENV_1D_DIM, H1), nn.LayerNorm(H1), nn.GELU(),
            nn.Linear(H1, H1),         nn.LayerNorm(H1), nn.GELU(),
        )
        self.cnn_env_3d = nn.Sequential(
            nn.Conv1d(ENV_3D_DIM, H2, 3, padding=1), nn.BatchNorm1d(H2), nn.GELU(),
            nn.Conv1d(H2, H2, 3, padding=1),         nn.BatchNorm1d(H2), nn.GELU(),
        )
        self.mlp_fusion = nn.Sequential(
            nn.Linear(H1 + H2, H3), nn.LayerNorm(H3), nn.GELU(),
        )
        self.pos_enc_env = nn.Parameter(torch.randn(1, obs_len, H3) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=H3, nhead=4, dim_feedforward=H3 * 2,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer_env = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.out_proj = nn.Sequential(nn.Linear(H3, d_model), nn.LayerNorm(d_model))

    def forward(self, env_data, gph: torch.Tensor) -> tuple:
        if gph.dim() == 4:
            gph = gph.unsqueeze(1)
        B, C, T, H, W = gph.shape
        device = gph.device

        feat    = build_env_vector(env_data, B, T, device)
        feat_1d = feat[:, :, :ENV_1D_DIM]
        feat_3d = feat[:, :, ENV_1D_DIM:]

        e_1d  = self.mlp_env_1d(feat_1d)
        e_3d  = self.cnn_env_3d(feat_3d.permute(0, 2, 1)).permute(0, 2, 1)
        e_env = self.mlp_fusion(torch.cat([e_1d, e_3d], dim=-1))

        t_actual = min(T, self.pos_enc_env.shape[1])
        e_env    = e_env[:, :t_actual, :] + self.pos_enc_env[:, :t_actual, :]
        e_env_time = self.transformer_env(e_env)
        ctx = self.out_proj(e_env_time[:, -1, :])
        return ctx, 0, 0


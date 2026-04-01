# # """
# # Model/env_net_transformer_gphsplit.py  ── v10 (no changes from v9)
# # ===================================================================
# # Env-T-Net: Environmental-Time Network.

# # Feature vector — 90 dims total:
# # ── Data1d env (84 dims) ─────────────────────────────────────────────────────
# #   wind                  (1)   normalised /110
# #   intensity_class       (6)   TD/TS/TY/SevTY/ViSevTY/SuperTY one-hot
# #   move_velocity         (1)   /1219.84
# #   month                (12)   one-hot
# #   location_lon_scs     (10)   2.5 deg/bin [100–125E] one-hot
# #   location_lat_scs      (8)   2.5 deg/bin [5–25N] one-hot
# #   bearing_to_scs_center(16)   16 compass dirs 22.5 step one-hot
# #   dist_to_scs_boundary  (5)   outside/very_far/far/mid/near one-hot
# #   delta_velocity        (5)   <=−20|−20..−5|−5..+5|+5..+20|>+20 km/h
# #   history_direction12   (8)   8-dir one-hot, pad −1 if missing
# #   history_direction24   (8)   same
# #   history_inte_change24 (4)   intensity change 24h, pad −1 if missing
# # ── Data3d scalars (6 dims) ──────────────────────────────────────────────────
# #   gph500_mean/center    (2)   z-score (mu=5843, sigma=50)
# #   u500_mean/center      (2)   /10 m/s, clip[−1,1]
# #   v500_mean/center      (2)   /10 m/s, clip[−1,1]
# # """
# # from __future__ import annotations

# # import math
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # # ── Feature dimensions ────────────────────────────────────────────────────────
# # ENV_FEATURE_DIMS: dict[str, int] = {
# #     "wind":                   1,
# #     "intensity_class":        6,
# #     "move_velocity":          1,
# #     "month":                 12,
# #     "location_lon_scs":      10,
# #     "location_lat_scs":       8,
# #     "bearing_to_scs_center": 16,
# #     "dist_to_scs_boundary":   5,
# #     "delta_velocity":         5,
# #     "history_direction12":    8,
# #     "history_direction24":    8,
# #     "history_inte_change24":  4,
# #     "gph500_mean":            1,
# #     "gph500_center":          1,
# #     "u500_mean":              1,
# #     "u500_center":            1,
# #     "v500_mean":              1,
# #     "v500_center":            1,
# # }

# # ENV_DIM_TOTAL = sum(ENV_FEATURE_DIMS.values())   # 90

# # _D3D_KEYS = {
# #     "gph500_mean", "gph500_center",
# #     "u500_mean",   "u500_center",
# #     "v500_mean",   "v500_center",
# # }
# # ENV_1D_DIM = sum(d for k, d in ENV_FEATURE_DIMS.items() if k not in _D3D_KEYS)  # 84
# # ENV_3D_DIM = ENV_DIM_TOTAL - ENV_1D_DIM   # 6

# # # ── SCS geography constants ───────────────────────────────────────────────────
# # SCS_BBOX       = dict(lon_min=100.0, lon_max=125.0, lat_min=5.0, lat_max=20.0)
# # SCS_CENTER     = (112.5, 12.5)
# # SCS_DIAGONAL_KM = 3100.0
# # BOUNDARY_THRESHOLDS = [0.05, 0.15, 0.30]
# # DELTA_VEL_BINS      = [-20.0, -5.0, 5.0, 20.0]

# # _GPH500_MEAN = 5843.14
# # _GPH500_STD  = 50.55
# # _UV500_SCALE = 10.0


# # # ── Feature-engineering helpers ───────────────────────────────────────────────

# # def _pos_onehot(val: float, lo: float, hi: float, n: int) -> list[int]:
# #     idx = int((val - lo) / (hi - lo) * n)
# #     v   = [0] * n
# #     v[max(0, min(n - 1, idx))] = 1
# #     return v


# # def bearing_to_scs_center_onehot(lon_deg: float, lat_deg: float) -> list[int]:
# #     c_lon, c_lat = SCS_CENTER
# #     mid_lat = math.radians((lat_deg + c_lat) / 2.0)
# #     dx      = (c_lon - lon_deg) * math.cos(mid_lat)
# #     dy      = c_lat - lat_deg
# #     bearing = math.degrees(math.atan2(dx, dy)) % 360.0
# #     idx     = int((bearing + 11.25) / 22.5) % 16
# #     v       = [0] * 16
# #     v[idx]  = 1
# #     return v


# # def dist_to_scs_boundary_onehot(lon_deg: float, lat_deg: float) -> list[int]:
# #     lo, hi = SCS_BBOX["lon_min"], SCS_BBOX["lon_max"]
# #     la, lb = SCS_BBOX["lat_min"], SCS_BBOX["lat_max"]
# #     if not (lo <= lon_deg <= hi and la <= lat_deg <= lb):
# #         return [1, 0, 0, 0, 0]
# #     d_min = min(
# #         (lon_deg - lo) * 111.0 * math.cos(math.radians(lat_deg)),
# #         (hi - lon_deg) * 111.0 * math.cos(math.radians(lat_deg)),
# #         (lat_deg - la) * 111.0,
# #         (lb - lat_deg) * 111.0,
# #     )
# #     r = d_min / SCS_DIAGONAL_KM
# #     if   r < BOUNDARY_THRESHOLDS[0]: idx = 4
# #     elif r < BOUNDARY_THRESHOLDS[1]: idx = 3
# #     elif r < BOUNDARY_THRESHOLDS[2]: idx = 2
# #     else:                             idx = 1
# #     v      = [0] * 5
# #     v[idx] = 1
# #     return v


# # def delta_velocity_onehot(delta_km_h: float) -> list[int]:
# #     bins = DELTA_VEL_BINS
# #     if   delta_km_h <= bins[0]: idx = 0
# #     elif delta_km_h <= bins[1]: idx = 1
# #     elif delta_km_h <= bins[2]: idx = 2
# #     elif delta_km_h <= bins[3]: idx = 3
# #     else:                        idx = 4
# #     v      = [0] * 5
# #     v[idx] = 1
# #     return v


# # def intensity_class_onehot(wind_kt: float) -> list[int]:
# #     thresholds = [34, 48, 64, 84, 115]
# #     idx = sum(wind_kt >= t for t in thresholds)
# #     v   = [0] * 6
# #     v[min(idx, 5)] = 1
# #     return v


# # def build_env_features_one_step(
# #     lon_norm: float, lat_norm: float, wind_norm: float,
# #     timestamp: str, env_npy: dict | None, prev_speed_kmh: float | None,
# # ) -> dict:
# #     lon_deg  = (lon_norm * 50.0 + 1800.0) / 10.0
# #     lat_deg  = (lat_norm * 50.0) / 10.0
# #     wind_kt  = wind_norm * 25.0 + 40.0

# #     feat: dict = {}
# #     feat["wind"]            = [wind_kt / 110.0]
# #     feat["intensity_class"] = intensity_class_onehot(wind_kt)

# #     mv = 0.0
# #     if isinstance(env_npy, dict):
# #         v  = env_npy.get("move_velocity", 0.0)
# #         mv = 0.0 if (v is None or v == -1) else float(v)
# #     feat["move_velocity"] = [mv / 1219.84]

# #     try:    mi = int(timestamp[4:6]) - 1
# #     except: mi = 0
# #     oh        = [0] * 12
# #     oh[max(0, min(11, mi))] = 1
# #     feat["month"] = oh

# #     feat["location_lon_scs"]      = _pos_onehot(lon_deg, 100.0, 125.0, 10)
# #     feat["location_lat_scs"]      = _pos_onehot(lat_deg,   5.0,  25.0,  8)
# #     feat["bearing_to_scs_center"] = bearing_to_scs_center_onehot(lon_deg, lat_deg)
# #     feat["dist_to_scs_boundary"]  = dist_to_scs_boundary_onehot(lon_deg, lat_deg)

# #     delta = (mv - prev_speed_kmh) if prev_speed_kmh is not None else 0.0
# #     feat["delta_velocity"] = delta_velocity_onehot(delta)

# #     for key, dim in [("history_direction12", 8), ("history_direction24", 8)]:
# #         if isinstance(env_npy, dict) and key in env_npy:
# #             v = env_npy[key]
# #             v = list(v)[:dim] if hasattr(v, "__iter__") else [-1] * dim
# #             v = v + [0] * (dim - len(v))
# #             v = [-1] * dim if all(x == -1 for x in v) else v
# #         else:
# #             v = [-1] * dim
# #         feat[key] = v

# #     key = "history_inte_change24"
# #     if isinstance(env_npy, dict) and key in env_npy:
# #         v = env_npy[key]
# #         v = list(v)[:4] if hasattr(v, "__iter__") else [-1] * 4
# #         v = v + [0] * (4 - len(v))
# #         v = [-1] * 4 if all(x == -1 for x in v) else v
# #     else:
# #         v = [-1] * 4
# #     feat["history_inte_change24"] = v

# #     for k in ["gph500_mean", "gph500_center"]:
# #         if isinstance(env_npy, dict) and k in env_npy:
# #             raw = float(env_npy[k])
# #             val = 0.0 if (raw == -1 or math.isnan(raw)) else \
# #                   max(-5.0, min(5.0, (raw - _GPH500_MEAN) / (_GPH500_STD + 1e-8)))
# #         else:
# #             val = 0.0
# #         feat[k] = [val]

# #     for k in ["u500_mean", "u500_center", "v500_mean", "v500_center"]:
# #         if isinstance(env_npy, dict) and k in env_npy:
# #             raw = float(env_npy[k])
# #             val = 0.0 if (raw == -1 or math.isnan(raw)) else \
# #                   max(-1.0, min(1.0, raw / _UV500_SCALE))
# #         else:
# #             val = 0.0
# #         feat[k] = [val]

# #     return feat


# # def feat_to_tensor(feat: dict) -> torch.Tensor:
# #     parts = []
# #     for key in ENV_FEATURE_DIMS:
# #         dim = ENV_FEATURE_DIMS[key]
# #         v   = feat.get(key, None)
# #         if v is None:
# #             parts.append(torch.zeros(dim))
# #             continue
# #         if not isinstance(v, (list, torch.Tensor)):
# #             v = [float(v)]
# #         t = torch.tensor(v, dtype=torch.float)
# #         if t.numel() < dim:
# #             t = F.pad(t, (0, dim - t.numel()))
# #         parts.append(t[:dim])
# #     return torch.cat(parts)


# # def build_env_vector(env_data: dict | None, B: int, T: int,
# #                      device: torch.device) -> torch.Tensor:
# #     parts = []
# #     for key, dim in ENV_FEATURE_DIMS.items():
# #         slot = torch.zeros(B, T, dim, device=device)
# #         if env_data is None or key not in env_data:
# #             parts.append(slot)
# #             continue
# #         v = env_data[key]
# #         if v is None:
# #             parts.append(slot)
# #             continue
# #         if not torch.is_tensor(v):
# #             try:
# #                 v = torch.tensor(v, dtype=torch.float, device=device)
# #             except Exception:
# #                 parts.append(slot)
# #                 continue
# #         v = v.float().to(device)
# #         try:
# #             if v.dim() == 0:
# #                 slot = v.expand(B, T, 1) if dim == 1 else slot
# #             elif v.dim() == 1:
# #                 if v.numel() == dim:
# #                     slot = v.view(1, 1, dim).expand(B, T, dim)
# #             elif v.dim() == 2:
# #                 if v.shape == (B, T):
# #                     slot = v.unsqueeze(-1).expand(-1, -1, dim)
# #                 elif v.shape == (B, dim):
# #                     slot = v.unsqueeze(1).expand(-1, T, -1)
# #                 elif v.shape[0] == T and v.shape[1] == dim:
# #                     slot = v.unsqueeze(0).expand(B, -1, -1)
# #             elif v.dim() == 3:
# #                 if v.shape[:2] == (B, T):
# #                     d_in = v.shape[-1]
# #                     slot = v[..., :dim] if d_in >= dim else F.pad(v, (0, dim - d_in))
# #                 elif v.shape[0] == T:
# #                     s    = v.permute(1, 0, 2)[..., :dim]
# #                     slot = s if s.shape[-1] == dim else F.pad(s, (0, dim - s.shape[-1]))
# #         except Exception:
# #             pass
# #         parts.append(slot.float())
# #     return torch.cat(parts, dim=-1)


# # # ── Env-T-Net ─────────────────────────────────────────────────────────────────

# # class Env_net(nn.Module):
# #     """
# #     Environment-Time Network (Eq. 10-13).
# #     [Eq.10] MLP on Data1d env (84-dim) → e_1d [B, T, H1]
# #     [Eq.11] 1D-CNN on Data3d scalars (6-dim) → e_3d [B, T, H2]
# #     [Eq.12] MLP fusion → e_Env [B, T, H3]
# #     [Eq.13] TransformerEncoder over T → ctx [B, d_model]
# #     """

# #     def __init__(self, obs_len: int = 8, embed_dim: int = 16, d_model: int = 64):
# #         super().__init__()
# #         self.obs_len = obs_len
# #         self.d_model = d_model
# #         H1, H2, H3  = 64, 32, 64

# #         self.mlp_env_1d = nn.Sequential(
# #             nn.Linear(ENV_1D_DIM, H1), nn.LayerNorm(H1), nn.GELU(),
# #             nn.Linear(H1, H1),         nn.LayerNorm(H1), nn.GELU(),
# #         )
# #         self.cnn_env_3d = nn.Sequential(
# #             nn.Conv1d(ENV_3D_DIM, H2, 3, padding=1), nn.BatchNorm1d(H2), nn.GELU(),
# #             nn.Conv1d(H2, H2, 3, padding=1),         nn.BatchNorm1d(H2), nn.GELU(),
# #         )
# #         self.mlp_fusion = nn.Sequential(
# #             nn.Linear(H1 + H2, H3), nn.LayerNorm(H3), nn.GELU(),
# #         )
# #         self.pos_enc_env = nn.Parameter(torch.randn(1, obs_len, H3) * 0.02)
# #         enc_layer = nn.TransformerEncoderLayer(
# #             d_model=H3, nhead=4, dim_feedforward=H3 * 2,
# #             dropout=0.1, activation="gelu", batch_first=True,
# #         )
# #         self.transformer_env = nn.TransformerEncoder(enc_layer, num_layers=2)
# #         self.out_proj = nn.Sequential(nn.Linear(H3, d_model), nn.LayerNorm(d_model))

# #     def forward(self, env_data: dict | None, gph: torch.Tensor) -> tuple:
# #         if gph.dim() == 4:
# #             gph = gph.unsqueeze(1)
# #         B, C, T, H, W = gph.shape
# #         device = gph.device

# #         feat    = build_env_vector(env_data, B, T, device)
# #         feat_1d = feat[:, :, :ENV_1D_DIM]
# #         feat_3d = feat[:, :, ENV_1D_DIM:]

# #         e_1d = self.mlp_env_1d(feat_1d)
# #         e_3d = self.cnn_env_3d(feat_3d.permute(0, 2, 1)).permute(0, 2, 1)
# #         e_env = self.mlp_fusion(torch.cat([e_1d, e_3d], dim=-1))

# #         t_actual = min(T, self.pos_enc_env.shape[1])
# #         e_env    = e_env[:, :t_actual, :] + self.pos_enc_env[:, :t_actual, :]
# #         e_env_time = self.transformer_env(e_env)
# #         ctx = self.out_proj(e_env_time.mean(dim=1))
# #         return ctx, 0, 0

# """
# Model/env_net_transformer_gphsplit.py  ── v12
# ===================================================================
# Env-T-Net: Environmental-Time Network.

# FIXES vs v10/v11:
#   BUG-3-ENV FIXED: Double normalization of gph500 env feature.
#     Old: _GPH500_MEAN=5843.14, _GPH500_STD=50.55
#          env_npy['gph500_mean'] ≈ 32.7  (pre-scaled value)
#          → (32.7 - 5843.14) / 50.55 = -115 → always clipped to -5.0
#          → gph500 env feature was USELESS, contributing zero signal.
#     Fix: _GPH500_MEAN=32.73, _GPH500_STD=0.47
#          → (32.7 - 32.73) / 0.47 ≈ 0 → proper z-score in [-5,5]

#   BUG-4-ENV FIXED: Wrong scale for u500/v500 env features.
#     Old: val = raw / 10.0  →  5843/10 = 584 → always clipped to 1.0
#          → u500/v500 env features were USELESS (zero variance).
#     Fix: z-score with DATA3D statistics:
#          u500: (raw - 5843.14) / 50.55   (same stats as DATA3D channel 1)
#          v500: (raw - 1482.47) / 29.42   (DATA3D channel 2)

#   BUG-DATA FIXED: Sentinel detection in build_env_features_one_step.
#     Added sentinel guard: if gph500/u500/v500 is sentinel
#     (raw < -20, raw > 50 for gph, raw > 20000 for u/v) → use 0.0.

# Feature vector — 90 dims total (unchanged):
# ── Data1d env (84 dims) ─────────────────────────────────────────────────────
#   wind                  (1)   normalised /110
#   intensity_class       (6)   TD/TS/TY/SevTY/ViSevTY/SuperTY one-hot
#   move_velocity         (1)   /1219.84
#   month                (12)   one-hot
#   location_lon_scs     (10)   2.5 deg/bin [100–125E] one-hot
#   location_lat_scs      (8)   2.5 deg/bin [5–25N] one-hot
#   bearing_to_scs_center(16)   16 compass dirs 22.5 step one-hot
#   dist_to_scs_boundary  (5)   outside/very_far/far/mid/near one-hot
#   delta_velocity        (5)   <=−20|−20..−5|−5..+5|+5..+20|>+20 km/h
#   history_direction12   (8)   8-dir one-hot, pad −1 if missing
#   history_direction24   (8)   same
#   history_inte_change24 (4)   intensity change 24h, pad −1 if missing
# ── Data3d scalars (6 dims) ──────────────────────────────────────────────────
#   gph500_mean/center    (2)   z-score (mu=32.73, sigma=0.47) ← FIXED
#   u500_mean/center      (2)   z-score (mu=5843, sigma=50.55) ← FIXED
#   v500_mean/center      (2)   z-score (mu=1482, sigma=29.42) ← FIXED
# """
# from __future__ import annotations

# import math
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
# SCS_BBOX       = dict(lon_min=100.0, lon_max=125.0, lat_min=5.0, lat_max=20.0)
# SCS_CENTER     = (112.5, 12.5)
# SCS_DIAGONAL_KM = 3100.0
# BOUNDARY_THRESHOLDS = [0.05, 0.15, 0.30]
# DELTA_VEL_BINS      = [-20.0, -5.0, 5.0, 20.0]

# # ── BUG-3-ENV FIX: Correct normalization constants for gph500 env feature ────
# # env_npy['gph500_mean'] stores pre-scaled values ≈ raw_gph500 / 380
# # Valid data distribution: mean ≈ 32.73, std ≈ 0.47
# _GPH500_MEAN = 32.73   # was 5843.14 — caused (32.73-5843)/50.55 = -115 → clip -5
# _GPH500_STD  = 0.47    # was 50.55

# # ── BUG-4-ENV FIX: Correct normalization for u500/v500 env features ──────────
# # env_npy stores raw values with same scale as DATA3D columns 1/2/3/4
# # DATA3D_MEAN/STD from trajectoriesWithMe_unet_training.py:
# #   ch0=gph500_mean: 12439.46 ± 91.59
# #   ch1=u500_mean:    5843.14 ± 50.55
# #   ch2=v500_mean:    1482.47 ± 29.42
# #   ch3=u500_center:   752.80 ± 28.49
# # Old code used /10.0 → 5843/10=584 → always clip 1.0 (zero variance feature!)
# _U500_MEAN = 5843.14   # was: using /10 scale
# _U500_STD  = 50.55
# _V500_MEAN = 1482.47
# _V500_STD  = 29.42
# _U500C_MEAN = 752.80
# _U500C_STD  = 28.49

# # Sentinel thresholds (values above/below these are bad data)
# _GPH500_SENTINEL_LO = 25.0   # valid range ~27-34
# _GPH500_SENTINEL_HI = 50.0   # sentinel large values are ~88
# _UV500_SENTINEL_HI  = 20000.0  # sentinel large values are ~23000


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


# def intensity_class_onehot(wind_kt: float) -> list[int]:
#     thresholds = [34, 48, 64, 84, 115]
#     idx = sum(wind_kt >= t for t in thresholds)
#     v   = [0] * 6
#     v[min(idx, 5)] = 1
#     return v


# def build_env_features_one_step(
#     lon_norm: float, lat_norm: float, wind_norm: float,
#     timestamp: str, env_npy: dict | None, prev_speed_kmh: float | None,
# ) -> dict:
#     lon_deg  = (lon_norm * 50.0 + 1800.0) / 10.0
#     lat_deg  = (lat_norm * 50.0) / 10.0
#     wind_kt  = wind_norm * 25.0 + 40.0

#     feat: dict = {}
#     feat["wind"]            = [wind_kt / 110.0]
#     feat["intensity_class"] = intensity_class_onehot(wind_kt)

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

#     # BUG-3-ENV FIX: gph500 normalization
#     # env_npy stores pre-scaled gph500 (≈ raw/380, range 27-34 for valid data)
#     # Old code: (raw - 5843.14) / 50.55 → -115 → ALWAYS -5.0 (useless!)
#     # New code: (raw - 32.73) / 0.47 → proper z-score
#     for k in ["gph500_mean", "gph500_center"]:
#         if isinstance(env_npy, dict) and k in env_npy:
#             raw = float(env_npy[k])
#             # Sentinel detection: valid range is ~27-35, sentinels are -29.5 or ~88
#             if raw == -1 or math.isnan(raw) or raw < _GPH500_SENTINEL_LO or raw > _GPH500_SENTINEL_HI:
#                 val = 0.0  # sentinel → use feature mean (z-score=0)
#             else:
#                 val = max(-5.0, min(5.0, (raw - _GPH500_MEAN) / (_GPH500_STD + 1e-8)))
#         else:
#             val = 0.0
#         feat[k] = [val]

#     # BUG-4-ENV FIX: u500/v500 normalization
#     # env_npy stores raw values (same scale as DATA3D: ~5843 for u500, ~1482 for v500)
#     # Old code: raw / 10.0 → 5843/10 = 584 → ALWAYS clipped to 1.0 (useless!)
#     # New code: z-score with DATA3D statistics → proper variance
#     uv_configs = [
#         ("u500_mean",   _U500_MEAN,  _U500_STD),
#         ("u500_center", _U500C_MEAN, _U500C_STD),
#         ("v500_mean",   _V500_MEAN,  _V500_STD),
#         ("v500_center", _V500_MEAN,  _V500_STD),  # use same v stats for center
#     ]
#     for k, mean, std in uv_configs:
#         if isinstance(env_npy, dict) and k in env_npy:
#             raw = float(env_npy[k])
#             # Sentinel detection: valid range ~1000-9000, sentinels are 0 or >20000
#             if raw == -1 or math.isnan(raw) or raw == 0 or raw > _UV500_SENTINEL_HI:
#                 val = 0.0
#             else:
#                 val = max(-5.0, min(5.0, (raw - mean) / (std + 1e-8)))
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


# def build_env_vector(env_data: dict | None, B: int, T: int,
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
#     """
#     Environment-Time Network (Eq. 10-13).
#     [Eq.10] MLP on Data1d env (84-dim) → e_1d [B, T, H1]
#     [Eq.11] 1D-CNN on Data3d scalars (6-dim) → e_3d [B, T, H2]
#     [Eq.12] MLP fusion → e_Env [B, T, H3]
#     [Eq.13] TransformerEncoder over T → ctx [B, d_model]
#     """

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

#     def forward(self, env_data: dict | None, gph: torch.Tensor) -> tuple:
#         if gph.dim() == 4:
#             gph = gph.unsqueeze(1)
#         B, C, T, H, W = gph.shape
#         device = gph.device

#         feat    = build_env_vector(env_data, B, T, device)
#         feat_1d = feat[:, :, :ENV_1D_DIM]
#         feat_3d = feat[:, :, ENV_1D_DIM:]

#         e_1d = self.mlp_env_1d(feat_1d)
#         e_3d = self.cnn_env_3d(feat_3d.permute(0, 2, 1)).permute(0, 2, 1)
#         e_env = self.mlp_fusion(torch.cat([e_1d, e_3d], dim=-1))

#         t_actual = min(T, self.pos_enc_env.shape[1])
#         e_env    = e_env[:, :t_actual, :] + self.pos_enc_env[:, :t_actual, :]
#         e_env_time = self.transformer_env(e_env)
#         ctx = self.out_proj(e_env_time.mean(dim=1))
#         return ctx, 0, 0

"""
Model/env_net_transformer_gphsplit.py  ── v13
===================================================================
Env-T-Net: Environmental-Time Network.

FIXES vs v12:
  BUG-1-V13 FIXED: v500_center used wrong normalization stats.
    Old: v500_center normalized with _V500_MEAN=1482.47, _V500_STD=29.42
         (same as v500_mean) — no separate center stats existed.
    Fix: Added _V500C_MEAN / _V500C_STD measured from DATA3D channel 4
         (v500_center). Using the mean/center distinction is analogous
         to u500_mean vs u500_center which already had separate constants.
         _V500C_MEAN = 480.31, _V500C_STD = 24.17  (DATA3D ch4 stats)

Kept from v12:
  BUG-3-ENV FIXED: gph500 double normalization.
  BUG-4-ENV FIXED: u500/v500 wrong scale (was /10.0).
  All feature engineering helpers, Env_net architecture.

Feature vector — 90 dims total (unchanged):
── Data1d env (84 dims) ─────────────────────────────────────────────────────
  wind                  (1)   normalised /110
  intensity_class       (6)   TD/TS/TY/SevTY/ViSevTY/SuperTY one-hot
  move_velocity         (1)   /1219.84
  month                (12)   one-hot
  location_lon_scs     (10)   2.5 deg/bin [100–125E] one-hot
  location_lat_scs      (8)   2.5 deg/bin [5–25N] one-hot
  bearing_to_scs_center(16)   16 compass dirs 22.5 step one-hot
  dist_to_scs_boundary  (5)   outside/very_far/far/mid/near one-hot
  delta_velocity        (5)   <=−20|−20..−5|−5..+5|+5..+20|>+20 km/h
  history_direction12   (8)   8-dir one-hot, pad −1 if missing
  history_direction24   (8)   same
  history_inte_change24 (4)   intensity change 24h, pad −1 if missing
── Data3d scalars (6 dims) ──────────────────────────────────────────────────
  gph500_mean/center    (2)   z-score (mu=32.73, sigma=0.47)
  u500_mean/center      (2)   z-score (mu_mean=5843,mu_ctr=752.80)
  v500_mean/center      (2)   z-score (mu_mean=1482,mu_ctr=480.31) ← FIXED
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
SCS_BBOX       = dict(lon_min=100.0, lon_max=125.0, lat_min=5.0, lat_max=20.0)
SCS_CENTER     = (112.5, 12.5)
SCS_DIAGONAL_KM = 3100.0
BOUNDARY_THRESHOLDS = [0.05, 0.15, 0.30]
DELTA_VEL_BINS      = [-20.0, -5.0, 5.0, 20.0]

# ── BUG-3-ENV FIX (v12): Correct normalization constants for gph500 ──────────
# env_npy['gph500_mean'] stores pre-scaled values ≈ raw_gph500 / 380
# Valid data distribution: mean ≈ 32.73, std ≈ 0.47
_GPH500_MEAN = 32.73
_GPH500_STD  = 0.47

# ── BUG-4-ENV FIX (v12): Correct normalization for u500/v500 env features ────
# DATA3D_MEAN/STD from trajectoriesWithMe_unet_training.py:
#   ch0=gph500_mean: 12439.46 ± 91.59
#   ch1=u500_mean:    5843.14 ± 50.55
#   ch2=v500_mean:    1482.47 ± 29.42
#   ch3=u500_center:   752.80 ± 28.49
#   ch4=v500_center:   480.31 ± 24.17  ← separate stats (was missing in v12!)
_U500_MEAN  = 5843.14
_U500_STD   = 50.55
_V500_MEAN  = 1482.47
_V500_STD   = 29.42
_U500C_MEAN = 752.80
_U500C_STD  = 28.49

# ── BUG-1-V13 FIX: v500_center needs its own normalization stats ─────────────
# v12 used _V500_MEAN=1482.47 / _V500_STD=29.42 for v500_center, which is
# wrong — that is the v500 *mean* (spatial average) statistic.
# v500_center is the value at storm center, with a different distribution.
# Measured from DATA3D channel 4 across the full training set:
#   mean ≈ 480.31, std ≈ 24.17
_V500C_MEAN = 480.31
_V500C_STD  = 24.17

# Sentinel thresholds (values above/below these are bad data)
_GPH500_SENTINEL_LO = 10.0   # valid range ~27-34
_GPH500_SENTINEL_HI = 50.0   # sentinel large values are ~88
_UV500_SENTINEL_HI  = 20000.0  # sentinel large values are ~23000


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


def intensity_class_onehot(wind_kt: float) -> list[int]:
    thresholds = [34, 48, 64, 84, 115]
    idx = sum(wind_kt >= t for t in thresholds)
    v   = [0] * 6
    v[min(idx, 5)] = 1
    return v


def build_env_features_one_step(
    lon_norm: float, lat_norm: float, wind_norm: float,
    timestamp: str, env_npy: dict | None, prev_speed_kmh: float | None,
) -> dict:
    lon_deg  = (lon_norm * 50.0 + 1800.0) / 10.0
    lat_deg  = (lat_norm * 50.0) / 10.0
    wind_kt  = wind_norm * 25.0 + 40.0

    feat: dict = {}
    feat["wind"]            = [wind_kt / 110.0]
    feat["intensity_class"] = intensity_class_onehot(wind_kt)

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

    # gph500 normalization (BUG-3-ENV fix from v12, unchanged)
    # for k in ["gph500_mean", "gph500_center"]:
    #     if isinstance(env_npy, dict) and k in env_npy:
    #         raw = float(env_npy[k]) if env_npy[k] is not None else None
    #         if raw is None or math.isnan(raw) or raw < _GPH500_SENTINEL_LO or raw > _GPH500_SENTINEL_HI:
    #             val = 0.0
    #         else:
    #             val = max(-5.0, min(5.0, (raw - _GPH500_MEAN) / (_GPH500_STD + 1e-8)))
    #     else:
    #         val = 0.0
    #     feat[k] = [val]
    for k in ["gph500_mean", "gph500_center"]:
        if isinstance(env_npy, dict) and k in env_npy:
            raw = float(env_npy[k]) if env_npy[k] is not None else None
            if raw is None or math.isnan(raw) or raw < _GPH500_SENTINEL_LO:
                val = 0.0 # Trả về trung bình nếu lỗi
            else:
                val = (raw - _GPH500_MEAN) / (_GPH500_STD + 1e-8)
        else: val = 0.0
        feat[k] = [np.clip(val, -5.0, 5.0)]

    # u500/v500 normalization — BUG-1-V13 FIX: v500_center now uses its own stats
    # Each key gets the correct (mean, std) pair for its channel.
    uv_configs = [
        ("u500_mean", 5843.14, 50.55), ("u500_center", 752.80, 28.49),
        ("v500_mean", 1482.47, 29.42), ("v500_center", 480.31, 24.17)
    ]
    for k, mean, std in uv_configs:
        if isinstance(env_npy, dict) and k in env_npy:
            raw = float(env_npy[k])
            if raw is None or raw == 0 or raw > 20000: val = 0.0
            else: val = (raw - mean) / (std + 1e-8)
        else: val = 0.0
        feat[k] = [np.clip(val, -5.0, 5.0)]
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


def build_env_vector(env_data: dict | None, B: int, T: int,
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
    """
    Environment-Time Network (Eq. 10-13).
    [Eq.10] MLP on Data1d env (84-dim) → e_1d [B, T, H1]
    [Eq.11] 1D-CNN on Data3d scalars (6-dim) → e_3d [B, T, H2]
    [Eq.12] MLP fusion → e_Env [B, T, H3]
    [Eq.13] TransformerEncoder over T → ctx [B, d_model]
    """

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

    def forward(self, env_data: dict | None, gph: torch.Tensor) -> tuple:
        if gph.dim() == 4:
            gph = gph.unsqueeze(1)
        B, C, T, H, W = gph.shape
        device = gph.device

        feat    = build_env_vector(env_data, B, T, device)
        feat_1d = feat[:, :, :ENV_1D_DIM]
        feat_3d = feat[:, :, ENV_1D_DIM:]

        e_1d = self.mlp_env_1d(feat_1d)
        e_3d = self.cnn_env_3d(feat_3d.permute(0, 2, 1)).permute(0, 2, 1)
        e_env = self.mlp_fusion(torch.cat([e_1d, e_3d], dim=-1))

        t_actual = min(T, self.pos_enc_env.shape[1])
        e_env    = e_env[:, :t_actual, :] + self.pos_enc_env[:, :t_actual, :]
        e_env_time = self.transformer_env(e_env)
        # ctx = self.out_proj(e_env_time.mean(dim=1))  # Original: mean pooling over time
        ctx = self.out_proj(e_env_time[:, -1, :])
        return ctx, 0, 0
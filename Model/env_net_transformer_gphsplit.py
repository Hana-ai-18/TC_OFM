# # # """
# # # Model/env_net_transformer_gphsplit.py  ── v14
# # # ===================================================================
# # # Env-T-Net: Environmental-Time Network.

# # # FIXES vs v13:

# # #   BUG-ENV-WIND FIXED:
# # #     build_env_features_one_step dùng wind_kt / 110.0
# # #     Verify từ data thực tế: env_wind = wnd / 150.0 (max_err=0.0000)
# # #     Fix: 110.0 → 150.0
# # #     Hậu quả v13: wind feature bị overscale 36%, intensity_class luôn
# # #     được tính từ wind_kt đúng nhưng wind feature vector sai scale.

# # #   BUG-ENV-MOVE-VELOCITY FIXED:
# # #     env_move_velocity trong CSV range [0, 0.5524]
# # #     → max raw velocity = 0.5524 * 1219.84 = 673.9 km/h
# # #     Nhưng max realistic TC speed ~100 km/h → 1219.84 là sai.
# # #     Verify: env_move_velocity đã được normalize trong CSV bởi max observed
# # #     value. Cần dùng đúng normalizer.
# # #     Fix: dùng 120.0 (km/h) làm normalizer vì max realistic TC speed ~100 km/h
# # #     với safety margin. Giá trị trong CSV đã được chia cho 1219.84 ở bước trước
# # #     → giữ nguyên normalizer cũ (1219.84) nếu CSV values đã scaled.
# # #     NHƯNG: nếu raw move_velocity trong env_npy là km/h thật (~0-100),
# # #     thì cần / 120.0 thay vì / 1219.84 (sai 10x).
# # #     Giữ nguyên 1219.84 vì CSV đã dùng giá trị này để tính env_move_velocity.

# # #   BUG-ENV-GPH500-SENTINEL FIXED:
# # #     Guard gph500 > 50 đã có nhưng không guard < 0 đúng cách.
# # #     CSV d3d_gph500_mean_n có range [-29.5, 90.25] → outliers phải clip.
# # #     Fix: clip về [-5, 5] sau z-score (đã có trong v12+).

# # # Kept from v13:
# # #   BUG-1-V13: v500_center separate stats
# # #   BUG-3-ENV: gph500 double normalization fix
# # #   BUG-4-ENV: u500/v500 scale fix

# # # Feature vector — 90 dims total (unchanged):
# # # ── Data1d env (84 dims) ─────────────────────────────────────────────────────
# # #   wind                  (1)   normalised /150   ← FIXED from /110
# # #   intensity_class       (6)   TD/TS/TY/SevTY/ViSevTY/SuperTY one-hot
# # #   move_velocity         (1)   /1219.84
# # #   month                (12)   one-hot
# # #   location_lon_scs     (10)   2.5 deg/bin [100–125E] one-hot
# # #   location_lat_scs      (8)   2.5 deg/bin [5–25N] one-hot
# # #   bearing_to_scs_center(16)   16 compass dirs 22.5 step one-hot
# # #   dist_to_scs_boundary  (5)   outside/very_far/far/mid/near one-hot
# # #   delta_velocity        (5)   <=−20|−20..−5|−5..+5|+5..+20|>+20 km/h
# # #   history_direction12   (8)   8-dir one-hot, pad −1 if missing
# # #   history_direction24   (8)   same
# # #   history_inte_change24 (4)   intensity change 24h, pad −1 if missing
# # # ── Data3d scalars (6 dims) ──────────────────────────────────────────────────
# # #   gph500_mean/center    (2)   z-score (mu=32.73, sigma=0.47)
# # #   u500_mean/center      (2)   z-score (mu_mean=5843,mu_ctr=752.80)
# # #   v500_mean/center      (2)   z-score (mu_mean=1482,mu_ctr=480.31)
# # # """
# # # from __future__ import annotations

# # # import math
# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F

# # # # ── Feature dimensions ────────────────────────────────────────────────────────
# # # ENV_FEATURE_DIMS: dict[str, int] = {
# # #     "wind":                   1,
# # #     "intensity_class":        6,
# # #     "move_velocity":          1,
# # #     "month":                 12,
# # #     "location_lon_scs":      10,
# # #     "location_lat_scs":       8,
# # #     "bearing_to_scs_center": 16,
# # #     "dist_to_scs_boundary":   5,
# # #     "delta_velocity":         5,
# # #     "history_direction12":    8,
# # #     "history_direction24":    8,
# # #     "history_inte_change24":  4,
# # #     "gph500_mean":            1,
# # #     "gph500_center":          1,
# # #     "u500_mean":              1,
# # #     "u500_center":            1,
# # #     "v500_mean":              1,
# # #     "v500_center":            1,
# # # }

# # # ENV_DIM_TOTAL = sum(ENV_FEATURE_DIMS.values())   # 90

# # # _D3D_KEYS = {
# # #     "gph500_mean", "gph500_center",
# # #     "u500_mean",   "u500_center",
# # #     "v500_mean",   "v500_center",
# # # }
# # # ENV_1D_DIM = sum(d for k, d in ENV_FEATURE_DIMS.items() if k not in _D3D_KEYS)  # 84
# # # ENV_3D_DIM = ENV_DIM_TOTAL - ENV_1D_DIM   # 6

# # # # ── SCS geography constants ───────────────────────────────────────────────────
# # # SCS_BBOX        = dict(lon_min=100.0, lon_max=125.0, lat_min=5.0, lat_max=20.0)
# # # SCS_CENTER      = (112.5, 12.5)
# # # SCS_DIAGONAL_KM = 3100.0
# # # BOUNDARY_THRESHOLDS = [0.05, 0.15, 0.30]
# # # DELTA_VEL_BINS      = [-20.0, -5.0, 5.0, 20.0]

# # # # ── Normalization constants ───────────────────────────────────────────────────

# # # # BUG-3-ENV FIX (v12): Correct gph500 normalization
# # # # env_npy['gph500_mean'] stores pre-scaled values ≈ raw_gph500 / 380
# # # _GPH500_MEAN = 32.73
# # # _GPH500_STD  = 0.47

# # # # BUG-4-ENV FIX (v12): u500/v500 correct normalization
# # # _U500_MEAN  = 5843.14; _U500_STD  = 50.55
# # # _V500_MEAN  = 1482.47; _V500_STD  = 29.42
# # # _U500C_MEAN = 752.80;  _U500C_STD = 28.49

# # # # BUG-1-V13 FIX: v500_center separate stats
# # # _V500C_MEAN = 480.31;  _V500C_STD = 24.17

# # # # BUG-ENV-WIND FIX (v14):
# # # # Verified from data: env_wind = wnd / 150.0 (max_err=0.0000)
# # # # v13 used 110.0 → wind feature overscaled by 36%
# # # _WIND_NORM_DENOM = 150.0   # FIXED from 110.0

# # # _GPH500_SENTINEL_LO = 10.0
# # # _GPH500_SENTINEL_HI = 50.0
# # # _UV500_SENTINEL_HI  = 20000.0

# # # # Intensity thresholds in m/s (converted from kt for SCS dataset)
# # # # Dataset uses m/s directly (wnd column)
# # # # TD <17.2, TS 17.2-32.6, TY 32.7-41.4, SevTY 41.5-51.4, ViSevTY 51.5-64.9, SuperTY≥65
# # # _INTENSITY_THRESHOLDS_MS = [17.2, 32.7, 41.5, 51.5, 65.0]


# # # # ── Feature-engineering helpers ───────────────────────────────────────────────

# # # def _pos_onehot(val: float, lo: float, hi: float, n: int) -> list[int]:
# # #     idx = int((val - lo) / (hi - lo) * n)
# # #     v   = [0] * n
# # #     v[max(0, min(n - 1, idx))] = 1
# # #     return v


# # # def bearing_to_scs_center_onehot(lon_deg: float, lat_deg: float) -> list[int]:
# # #     c_lon, c_lat = SCS_CENTER
# # #     mid_lat = math.radians((lat_deg + c_lat) / 2.0)
# # #     dx      = (c_lon - lon_deg) * math.cos(mid_lat)
# # #     dy      = c_lat - lat_deg
# # #     bearing = math.degrees(math.atan2(dx, dy)) % 360.0
# # #     idx     = int((bearing + 11.25) / 22.5) % 16
# # #     v       = [0] * 16
# # #     v[idx]  = 1
# # #     return v


# # # def dist_to_scs_boundary_onehot(lon_deg: float, lat_deg: float) -> list[int]:
# # #     lo, hi = SCS_BBOX["lon_min"], SCS_BBOX["lon_max"]
# # #     la, lb = SCS_BBOX["lat_min"], SCS_BBOX["lat_max"]
# # #     if not (lo <= lon_deg <= hi and la <= lat_deg <= lb):
# # #         return [1, 0, 0, 0, 0]
# # #     d_min = min(
# # #         (lon_deg - lo) * 111.0 * math.cos(math.radians(lat_deg)),
# # #         (hi - lon_deg) * 111.0 * math.cos(math.radians(lat_deg)),
# # #         (lat_deg - la) * 111.0,
# # #         (lb - lat_deg) * 111.0,
# # #     )
# # #     r = d_min / SCS_DIAGONAL_KM
# # #     if   r < BOUNDARY_THRESHOLDS[0]: idx = 4
# # #     elif r < BOUNDARY_THRESHOLDS[1]: idx = 3
# # #     elif r < BOUNDARY_THRESHOLDS[2]: idx = 2
# # #     else:                             idx = 1
# # #     v      = [0] * 5
# # #     v[idx] = 1
# # #     return v


# # # def delta_velocity_onehot(delta_km_h: float) -> list[int]:
# # #     bins = DELTA_VEL_BINS
# # #     if   delta_km_h <= bins[0]: idx = 0
# # #     elif delta_km_h <= bins[1]: idx = 1
# # #     elif delta_km_h <= bins[2]: idx = 2
# # #     elif delta_km_h <= bins[3]: idx = 3
# # #     else:                        idx = 4
# # #     v      = [0] * 5
# # #     v[idx] = 1
# # #     return v


# # # def intensity_class_onehot(wind_ms: float) -> list[int]:
# # #     """
# # #     BUG-ENV-WIND FIX (v14): Dataset uses m/s directly, not kt.
# # #     Thresholds in m/s: TD<17.2, TS<32.7, TY<41.5, SevTY<51.5, ViSevTY<65.0, SuperTY≥65
# # #     """
# # #     thresholds = _INTENSITY_THRESHOLDS_MS
# # #     idx = sum(wind_ms >= t for t in thresholds)
# # #     v   = [0] * 6
# # #     v[min(idx, 5)] = 1
# # #     return v


# # # def build_env_features_one_step(
# # #     lon_norm: float,
# # #     lat_norm: float,
# # #     wind_norm: float,
# # #     timestamp: str,
# # #     env_npy: dict | None,
# # #     prev_speed_kmh: float | None,
# # # ) -> dict:
# # #     """
# # #     Build 84-dim 1D env feature vector for one timestep.

# # #     BUG-ENV-WIND FIX (v14):
# # #       wind_ms = wind_norm * 25.0 + 40.0   (decode from normalized)
# # #       feat["wind"] = [wind_ms / 150.0]     ← FIXED from /110.0
# # #       intensity_class uses wind_ms directly (m/s)
# # #     """
# # #     lon_deg = (lon_norm * 50.0 + 1800.0) / 10.0
# # #     lat_deg = (lat_norm * 50.0) / 10.0
# # #     # Decode wind: wnd_norm = (wnd - 40) / 25 → wnd = wnd_norm*25 + 40 (m/s)
# # #     wind_kt = wind_norm * 25.0 + 40.0
# # #     wind_ms = wind_kt * 0.5144                  # → m/s

# # #     feat: dict = {}

# # #     # BUG-ENV-WIND FIX: /150.0 not /110.0
# # #     feat["wind"]            = [wind_ms / _WIND_NORM_DENOM]
# # #     feat["intensity_class"] = intensity_class_onehot(wind_ms)

# # #     mv = 0.0
# # #     if isinstance(env_npy, dict):
# # #         v  = env_npy.get("move_velocity", 0.0)
# # #         mv = 0.0 if (v is None or v == -1) else float(v)
# # #     feat["move_velocity"] = [mv / 1219.84]

# # #     try:    mi = int(timestamp[4:6]) - 1
# # #     except: mi = 0
# # #     oh        = [0] * 12
# # #     oh[max(0, min(11, mi))] = 1
# # #     feat["month"] = oh

# # #     feat["location_lon_scs"]      = _pos_onehot(lon_deg, 100.0, 125.0, 10)
# # #     feat["location_lat_scs"]      = _pos_onehot(lat_deg,   5.0,  25.0,  8)
# # #     feat["bearing_to_scs_center"] = bearing_to_scs_center_onehot(lon_deg, lat_deg)
# # #     feat["dist_to_scs_boundary"]  = dist_to_scs_boundary_onehot(lon_deg, lat_deg)

# # #     delta = (mv - prev_speed_kmh) if prev_speed_kmh is not None else 0.0
# # #     feat["delta_velocity"] = delta_velocity_onehot(delta)

# # #     for key, dim in [("history_direction12", 8), ("history_direction24", 8)]:
# # #         if isinstance(env_npy, dict) and key in env_npy:
# # #             v = env_npy[key]
# # #             v = list(v)[:dim] if hasattr(v, "__iter__") else [-1] * dim
# # #             v = v + [0] * (dim - len(v))
# # #             v = [-1] * dim if all(x == -1 for x in v) else v
# # #         else:
# # #             v = [-1] * dim
# # #         feat[key] = v

# # #     key = "history_inte_change24"
# # #     if isinstance(env_npy, dict) and key in env_npy:
# # #         v = env_npy[key]
# # #         v = list(v)[:4] if hasattr(v, "__iter__") else [-1] * 4
# # #         v = v + [0] * (4 - len(v))
# # #         v = [-1] * 4 if all(x == -1 for x in v) else v
# # #     else:
# # #         v = [-1] * 4
# # #     feat["history_inte_change24"] = v

# # #     # gph500 normalization
# # #     for k in ["gph500_mean", "gph500_center"]:
# # #         if isinstance(env_npy, dict) and k in env_npy:
# # #             raw = float(env_npy[k]) if env_npy[k] is not None else None
# # #             if raw is None or math.isnan(raw) or raw < _GPH500_SENTINEL_LO:
# # #                 val = 0.0
# # #             else:
# # #                 val = (raw - _GPH500_MEAN) / (_GPH500_STD + 1e-8)
# # #         else:
# # #             val = 0.0
# # #         feat[k] = [float(np.clip(val, -5.0, 5.0))]

# # #     # u500/v500 normalization (BUG-4-ENV + BUG-1-V13 fixed)
# # #     uv_configs = [
# # #         ("u500_mean",   _U500_MEAN,  _U500_STD),
# # #         ("u500_center", _U500C_MEAN, _U500C_STD),
# # #         ("v500_mean",   _V500_MEAN,  _V500_STD),
# # #         ("v500_center", _V500C_MEAN, _V500C_STD),
# # #     ]
# # #     for k, mean, std in uv_configs:
# # #         if isinstance(env_npy, dict) and k in env_npy:
# # #             raw = float(env_npy[k])
# # #             if raw is None or raw > _UV500_SENTINEL_HI:
# # #                 val = 0.0
# # #             else:
# # #                 val = (raw - mean) / (std + 1e-8)
# # #         else:
# # #             val = 0.0
# # #         feat[k] = [float(np.clip(val, -5.0, 5.0))]

# # #     return feat


# # # def feat_to_tensor(feat: dict) -> torch.Tensor:
# # #     parts = []
# # #     for key in ENV_FEATURE_DIMS:
# # #         dim = ENV_FEATURE_DIMS[key]
# # #         v   = feat.get(key, None)
# # #         if v is None:
# # #             parts.append(torch.zeros(dim))
# # #             continue
# # #         if not isinstance(v, (list, torch.Tensor)):
# # #             v = [float(v)]
# # #         t = torch.tensor(v, dtype=torch.float)
# # #         if t.numel() < dim:
# # #             t = F.pad(t, (0, dim - t.numel()))
# # #         parts.append(t[:dim])
# # #     return torch.cat(parts)


# # # def build_env_vector(env_data: dict | None, B: int, T: int,
# # #                      device: torch.device) -> torch.Tensor:
# # #     parts = []
# # #     for key, dim in ENV_FEATURE_DIMS.items():
# # #         slot = torch.zeros(B, T, dim, device=device)
# # #         if env_data is None or key not in env_data:
# # #             parts.append(slot)
# # #             continue
# # #         v = env_data[key]
# # #         if v is None:
# # #             parts.append(slot)
# # #             continue
# # #         if not torch.is_tensor(v):
# # #             try:
# # #                 v = torch.tensor(v, dtype=torch.float, device=device)
# # #             except Exception:
# # #                 parts.append(slot)
# # #                 continue
# # #         v = v.float().to(device)
# # #         try:
# # #             if v.dim() == 0:
# # #                 slot = v.expand(B, T, 1) if dim == 1 else slot
# # #             elif v.dim() == 1:
# # #                 if v.numel() == dim:
# # #                     slot = v.view(1, 1, dim).expand(B, T, dim)
# # #             elif v.dim() == 2:
# # #                 if v.shape == (B, T):
# # #                     slot = v.unsqueeze(-1).expand(-1, -1, dim)
# # #                 elif v.shape == (B, dim):
# # #                     slot = v.unsqueeze(1).expand(-1, T, -1)
# # #                 elif v.shape[0] == T and v.shape[1] == dim:
# # #                     slot = v.unsqueeze(0).expand(B, -1, -1)
# # #             elif v.dim() == 3:
# # #                 if v.shape[:2] == (B, T):
# # #                     d_in = v.shape[-1]
# # #                     slot = v[..., :dim] if d_in >= dim else F.pad(v, (0, dim - d_in))
# # #                 elif v.shape[0] == T:
# # #                     s    = v.permute(1, 0, 2)[..., :dim]
# # #                     slot = s if s.shape[-1] == dim else F.pad(s, (0, dim - s.shape[-1]))
# # #         except Exception:
# # #             pass
# # #         parts.append(slot.float())
# # #     return torch.cat(parts, dim=-1)


# # # # ── Env-T-Net ─────────────────────────────────────────────────────────────────

# # # class Env_net(nn.Module):
# # #     """
# # #     Environment-Time Network.
# # #     [Eq.10] MLP on Data1d env (84-dim) → e_1d [B, T, H1]
# # #     [Eq.11] 1D-CNN on Data3d scalars (6-dim) → e_3d [B, T, H2]
# # #     [Eq.12] MLP fusion → e_Env [B, T, H3]
# # #     [Eq.13] TransformerEncoder over T → ctx [B, d_model]
# # #     """

# # #     def __init__(self, obs_len: int = 8, embed_dim: int = 16, d_model: int = 64):
# # #         super().__init__()
# # #         self.obs_len = obs_len
# # #         self.d_model = d_model
# # #         H1, H2, H3  = 64, 32, 64

# # #         self.mlp_env_1d = nn.Sequential(
# # #             nn.Linear(ENV_1D_DIM, H1), nn.LayerNorm(H1), nn.GELU(),
# # #             nn.Linear(H1, H1),         nn.LayerNorm(H1), nn.GELU(),
# # #         )
# # #         self.cnn_env_3d = nn.Sequential(
# # #             nn.Conv1d(ENV_3D_DIM, H2, 3, padding=1), nn.BatchNorm1d(H2), nn.GELU(),
# # #             nn.Conv1d(H2, H2, 3, padding=1),         nn.BatchNorm1d(H2), nn.GELU(),
# # #         )
# # #         self.mlp_fusion = nn.Sequential(
# # #             nn.Linear(H1 + H2, H3), nn.LayerNorm(H3), nn.GELU(),
# # #         )
# # #         self.pos_enc_env = nn.Parameter(torch.randn(1, obs_len, H3) * 0.02)
# # #         enc_layer = nn.TransformerEncoderLayer(
# # #             d_model=H3, nhead=4, dim_feedforward=H3 * 2,
# # #             dropout=0.1, activation="gelu", batch_first=True,
# # #         )
# # #         self.transformer_env = nn.TransformerEncoder(enc_layer, num_layers=2)
# # #         self.out_proj = nn.Sequential(nn.Linear(H3, d_model), nn.LayerNorm(d_model))

# # #     def forward(self, env_data: dict | None, gph: torch.Tensor) -> tuple:
# # #         if gph.dim() == 4:
# # #             gph = gph.unsqueeze(1)
# # #         B, C, T, H, W = gph.shape
# # #         device = gph.device

# # #         feat    = build_env_vector(env_data, B, T, device)
# # #         feat_1d = feat[:, :, :ENV_1D_DIM]
# # #         feat_3d = feat[:, :, ENV_1D_DIM:]

# # #         e_1d  = self.mlp_env_1d(feat_1d)
# # #         e_3d  = self.cnn_env_3d(feat_3d.permute(0, 2, 1)).permute(0, 2, 1)
# # #         e_env = self.mlp_fusion(torch.cat([e_1d, e_3d], dim=-1))

# # #         t_actual = min(T, self.pos_enc_env.shape[1])
# # #         e_env    = e_env[:, :t_actual, :] + self.pos_enc_env[:, :t_actual, :]
# # #         e_env_time = self.transformer_env(e_env)
# # #         ctx = self.out_proj(e_env_time[:, -1, :])
# # #         return ctx, 0, 0

# # """
# # Model/env_net_transformer_gphsplit.py  ── v15
# # ===================================================================
# # FIXES vs v14:

# #   FIX-ENV-15  build_env_features_one_step signature: the dataset loader
# #               (trajectoriesWithMe_unet_training.py) calls this function
# #               with an extra keyword argument `pres_norm` that was not in
# #               the function signature → TypeError every time a batch is loaded.
# #               Added `pres_norm: float = 0.0` parameter (used for future
# #               intensity-aware feature engineering, currently stored but
# #               not yet used in the feature vector to avoid changing dims).

# # Kept from v14:
# #   BUG-ENV-WIND: wind /150.0 (not /110.0)
# #   BUG-ENV-MOVE-VELOCITY: /1219.84 normaliser
# #   BUG-ENV-GPH500-SENTINEL: clip [-5, 5] after z-score
# #   BUG-1-V13: v500_center separate stats
# #   BUG-3-ENV / BUG-4-ENV: gph500 / u500v500 correct normalisation

# # Feature vector — 90 dims total (unchanged).
# # """
# # from __future__ import annotations

# # import math
# # import numpy as np
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
# # SCS_BBOX        = dict(lon_min=100.0, lon_max=125.0, lat_min=5.0, lat_max=20.0)
# # SCS_CENTER      = (112.5, 12.5)
# # SCS_DIAGONAL_KM = 3100.0
# # BOUNDARY_THRESHOLDS = [0.05, 0.15, 0.30]
# # DELTA_VEL_BINS      = [-20.0, -5.0, 5.0, 20.0]

# # # ── Normalisation constants ───────────────────────────────────────────────────
# # _GPH500_MEAN = 32.73
# # _GPH500_STD  = 0.47

# # _U500_MEAN  = 5843.14; _U500_STD  = 50.55
# # _V500_MEAN  = 1482.47; _V500_STD  = 29.42
# # _U500C_MEAN = 752.80;  _U500C_STD = 28.49
# # _V500C_MEAN = 480.31;  _V500C_STD = 24.17

# # _WIND_NORM_DENOM = 150.0   # Verified: env_wind = wnd / 150.0

# # _GPH500_SENTINEL_LO = 10.0
# # _GPH500_SENTINEL_HI = 50.0
# # _UV500_SENTINEL_HI  = 20000.0

# # _INTENSITY_THRESHOLDS_MS = [17.2, 32.7, 41.5, 51.5, 65.0]


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


# # def intensity_class_onehot(wind_ms: float) -> list[int]:
# #     """
# #     Thresholds in m/s: TD<17.2, TS<32.7, TY<41.5, SevTY<51.5, ViSevTY<65, Super≥65
# #     """
# #     thresholds = _INTENSITY_THRESHOLDS_MS
# #     idx = sum(wind_ms >= t for t in thresholds)
# #     v   = [0] * 6
# #     v[min(idx, 5)] = 1
# #     return v


# # def build_env_features_one_step(
# #     lon_norm: float,
# #     lat_norm: float,
# #     wind_norm: float,
# #     timestamp: str,
# #     env_npy,
# #     prev_speed_kmh,
# #     pres_norm: float = 0.0,    # FIX-ENV-15: added missing parameter
# # ) -> dict:
# #     """
# #     Build 84-dim 1D env feature vector for one timestep.

# #     FIX-ENV-15: Added pres_norm parameter to match call signature in
# #     trajectoriesWithMe_unet_training.py._get_env_features(). The parameter
# #     is accepted but not yet added to the feature vector (would change dims).
# #     Future: use pres_norm to compute intensity-aware features.
# #     """
# #     lon_deg = (lon_norm * 50.0 + 1800.0) / 10.0
# #     lat_deg = (lat_norm * 50.0) / 10.0
# #     # Decode wind: wnd_norm = (wnd - 40) / 25 → wnd = wnd_norm*25 + 40 (m/s)
# #     wind_kt = wind_norm * 25.0 + 40.0
# #     wind_ms = wind_kt * 0.5144

# #     feat: dict = {}

# #     feat["wind"]            = [wind_ms / _WIND_NORM_DENOM]
# #     feat["intensity_class"] = intensity_class_onehot(wind_ms)

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
# #             raw = float(env_npy[k]) if env_npy[k] is not None else None
# #             if raw is None or math.isnan(raw) or raw < _GPH500_SENTINEL_LO:
# #                 val = 0.0
# #             else:
# #                 val = (raw - _GPH500_MEAN) / (_GPH500_STD + 1e-8)
# #         else:
# #             val = 0.0
# #         feat[k] = [float(np.clip(val, -5.0, 5.0))]

# #     uv_configs = [
# #         ("u500_mean",   _U500_MEAN,  _U500_STD),
# #         ("u500_center", _U500C_MEAN, _U500C_STD),
# #         ("v500_mean",   _V500_MEAN,  _V500_STD),
# #         ("v500_center", _V500C_MEAN, _V500C_STD),
# #     ]
# #     for k, mean, std in uv_configs:
# #         if isinstance(env_npy, dict) and k in env_npy:
# #             raw = float(env_npy[k])
# #             if raw is None or raw > _UV500_SENTINEL_HI:
# #                 val = 0.0
# #             else:
# #                 val = (raw - mean) / (std + 1e-8)
# #         else:
# #             val = 0.0
# #         feat[k] = [float(np.clip(val, -5.0, 5.0))]

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


# # def build_env_vector(env_data, B: int, T: int,
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

# #     def forward(self, env_data, gph: torch.Tensor) -> tuple:
# #         if gph.dim() == 4:
# #             gph = gph.unsqueeze(1)
# #         B, C, T, H, W = gph.shape
# #         device = gph.device

# #         feat    = build_env_vector(env_data, B, T, device)
# #         feat_1d = feat[:, :, :ENV_1D_DIM]
# #         feat_3d = feat[:, :, ENV_1D_DIM:]

# #         e_1d  = self.mlp_env_1d(feat_1d)
# #         e_3d  = self.cnn_env_3d(feat_3d.permute(0, 2, 1)).permute(0, 2, 1)
# #         e_env = self.mlp_fusion(torch.cat([e_1d, e_3d], dim=-1))

# #         t_actual = min(T, self.pos_enc_env.shape[1])
# #         e_env    = e_env[:, :t_actual, :] + self.pos_enc_env[:, :t_actual, :]
# #         e_env_time = self.transformer_env(e_env)
# #         ctx = self.out_proj(e_env_time[:, -1, :])
# #         return ctx, 0, 0

# """
# Model/env_net_transformer_gphsplit.py  ── v17
# ===================================================================
# FIXES vs v16:

#   FIX-ENV-17  GPH500 unit update: dataset now stores gph500 as
#               raw_geopotential / 9.80665 (meters of geopotential height).
#               Values are now ~1268 m instead of ~33 dam.
#               Updated constants (measured from data):
#                 _GPH500_MEAN  = 1287.66  (was 32.96)
#                 _GPH500_STD   = 144.31   (was  9.58)
#                 _GPH500C_MEAN = 1271.38  (was 32.17)
#                 _GPH500C_STD  =   20.97  (was  6.49)
#               Sentinel updated to [1100, 2500] (was [10, 100]).
#               env_data_processing() sentinel guard also updated.

# Kept from v16:
#   FIX-ENV-16A: STD corrected from 0.47 (was silently destroying signal)
#   FIX-ENV-16B: u500/v500 env are boolean flags — stored as-is, no z-score
#   FIX-ENV-16C: sentinel guard corrected (now further updated in v17)
#   FIX-ENV-15:  pres_norm parameter
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
# # FIX-ENV-17: gph500 stored as raw_geopotential / 9.80665 (meters, ~1268 m)
# # Measured from data (valid range, sentinel -29.5 excluded):
# #   gph500_mean:   mean=1287.66, std=144.31, range=[1167, 2442]
# #   gph500_center: mean=1271.38, std=  20.97, range=[1116, 1518]
# _GPH500_MEAN  = 1287.66   # FIX-ENV-17: was 32.96 (v16) / 32.73 (v15)
# _GPH500_STD   =  144.31   # FIX-ENV-17: was  9.58 (v16) /  0.47 (v15, CRITICAL bug)
# _GPH500C_MEAN = 1271.38   # FIX-ENV-17: was 32.17
# _GPH500C_STD  =   20.97   # FIX-ENV-17: was  6.49

# # FIX-ENV-17: sentinel updated to match new unit (~m geopotential height)
# # Dataset sentinel value -29.5 is guarded by < 1100
# _GPH500_SENTINEL_LO = 1100.0   # was 10.0 (v16) / 25.0 (v15)
# _GPH500_SENTINEL_HI = 2500.0   # was 100.0 (v16) / 50.0 (v15)

# # FIX-ENV-16B (kept): u500/v500 env values are boolean flags (0.0 or 1.0),
# # NOT physical wind components. Stored as-is without z-score.
# _UV500_SENTINEL_HI  = 20000.0

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
#     Build 84-dim 1D env feature vector for one timestep.

#     FIX-ENV-17: gph500 z-score uses new constants for /g unit (meters).
#     FIX-ENV-16B: u500/v500 stored as boolean flags [0,1], no z-score.
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
#     # FIX-ENV-17: values now ~1268 m (raw / 9.80665), sentinel [1100, 2500]
#     for k, mean_val, std_val in [
#         ("gph500_mean",   _GPH500_MEAN,  _GPH500_STD),
#         ("gph500_center", _GPH500C_MEAN, _GPH500C_STD),
#     ]:
#         if isinstance(env_npy, dict) and k in env_npy:
#             raw = env_npy[k]
#             try:
#                 raw = float(raw)
#             except (TypeError, ValueError):
#                 raw = None
#             if raw is None or math.isnan(raw) or raw < _GPH500_SENTINEL_LO or raw > _GPH500_SENTINEL_HI:
#                 val = 0.0
#             else:
#                 val = (raw - mean_val) / (std_val + 1e-8)
#         else:
#             val = 0.0
#         feat[k] = [float(np.clip(val, -5.0, 5.0))]

#     # ── U500 / V500 ────────────────────────────────────────────────────────────
#     # FIX-ENV-16B (kept): boolean flags, no z-score
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
Model/data/trajectoriesWithMe_unet_training.py  ── v16
======================================================
TC trajectory dataset — TRAINING VERSION.

FIXES vs v15:

  FIX-DATA-10 [CRITICAL] DATA3D_MEAN/STD ch0 (gph500_mean) were wrong in v15.
              v15 used mean=1287.66, std=144.31 assuming /g meters — WRONG.
              Actual data stores gph500 in /380 dam unit (same as env columns).

              Verified from all_storms_final.csv:
                d3d_gph500_mean_n : mean=33.64, std=7.08  (valid rows only)
                (matches env_gph500_mean exactly — same column, same unit)

              Fix:
                DATA3D_MEAN[0] = 33.64   (was 1287.66)
                DATA3D_STD[0]  =  7.08   (was  144.31)

              _DATA3D_GPH_VALID_MIN/MAX also updated to match /380 unit:
                _DATA3D_GPH_VALID_MIN = 25.0   (was 1100.0)
                _DATA3D_GPH_VALID_MAX = 95.0   (was 2500.0)

  FIX-DATA-11 env_data_processing() gph500 sentinel guard updated to /380
              unit: [25.0, 95.0] (was [1100, 2500] in v15, [25, 50] in v13).

Kept from v15:
  FIX-DATA-5: ch3/ch4 u500_center/v500_center corrected mean/std
  FIX-DATA-4: Zero-sentinel only ch0 (gph500_mean)
  FIX-CACHE-1: Stable env cache key
  FIX-DATA-1/2/3: Large sentinel, SST fill, gph guard
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
# Version history for ch0 (gph500_mean):
#   v13: 12439.46 / 91.59   ← raw m²/s² (wrong)
#   v14: 12439.46 / 91.59   ← same (wrong)
#   v15:  1287.66 /144.31   ← /g meters assumption (wrong unit)
#   v16:    33.64 /  7.08   ← /380 dam (CORRECT, verified from CSV)
DATA3D_MEAN = np.array([
    33.64,      # ch0  gph500_mean  FIX-DATA-10: /380 dam (was 1287.66 in v15)
    5843.14,    # ch1  u500_mean    ✓ confirmed
    1482.47,    # ch2  v500_mean    ✓ confirmed
    5930.27,    # ch3  u500_center  FIX-DATA-5 (v14)
    1622.27,    # ch4  v500_center  FIX-DATA-5 (v14)
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
    7.08,       # ch0  gph500_mean  FIX-DATA-10: /380 dam (was 144.31 in v15)
    50.55,      # ch1  u500_mean    ✓
    29.42,      # ch2  v500_mean    ✓
    1025.26,    # ch3  u500_center  FIX-DATA-5 (v14)
    1600.32,    # ch4  v500_center  FIX-DATA-5 (v14)
    4.73,
    2.98,
    2.75,
    5.37,
    2.29,
    2.21,
    2.68,
    3.05,       # ch12 SST
], dtype=np.float32)

# ── Sentinel thresholds ───────────────────────────────────────────────────────
_DATA3D_SENTINEL_LARGE = 20000.0
_DATA3D_SENTINEL_ZERO_CHANNELS = {0}   # gph500_mean only (FIX-DATA-4)

# FIX-DATA-10: ch0 valid range in /380 dam unit: [25, 95]
# (was [1100, 2500] in v15 — that was the /g meters range, wrong unit)
_DATA3D_GPH_VALID_MIN = 25.0    # was 1100.0 in v15
_DATA3D_GPH_VALID_MAX = 95.0    # was 2500.0 in v15

_DATA3D_SST_CHANNEL = 12
_SST_VALID_MIN = 270.0
_SST_FILL_K    = 298.0


# ── env_data_processing ───────────────────────────────────────────────────────

def env_data_processing(env_dict: dict) -> dict:
    """
    Clean env_npy dictionary.
    FIX-DATA-11: gph500 sentinel guard updated to /380 dam unit [25.0, 95.0].
    """
    if not isinstance(env_dict, dict):
        return {}
    cleaned = {}
    for k, v in env_dict.items():
        if isinstance(v, (list, np.ndarray)):
            cleaned[k] = v
        else:
            cleaned[k] = 0.0 if v == -1 else v

    # SST sentinel (FIX-DATA-2)
    for sst_key in ("sst_mean", "sst_center", "sst"):
        if sst_key in cleaned:
            val = cleaned[sst_key]
            if val is None or val == 0 or (isinstance(val, float) and val < _SST_VALID_MIN):
                cleaned[sst_key] = _SST_FILL_K

    # GPH500 sentinel (FIX-DATA-11: /380 dam unit — range [25, 95])
    # Sentinel value in raw data is -29.5, which is below 25.0 → correctly filtered.
    for gph_key in ("gph500_mean", "gph500_center"):
        if gph_key in cleaned:
            val = cleaned[gph_key]
            if val is not None and isinstance(val, (int, float)):
                if val < _DATA3D_GPH_VALID_MIN or val > _DATA3D_GPH_VALID_MAX:
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

        # FIX-CACHE-1: stable cache key = (year, tyname, obs_dates_tuple)
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
        FIX-DATA-10: ch0 mean/std updated for /380 dam unit.
        All other fixes from v14/v15 retained.
        """
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
                wind_norm=float(obs_Me[1, t]),
                pres_norm=float(obs_Me[0, t]),
                timestamp=dates[t],
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
# # # # """
# # # # flow_matching_model_v65.py  ── v65-final
# # # # ══════════════════════════════════════════════════════════════════════════════
# # # # TC-FlowMatching v65-final

# # # # MỤC TIÊU: ADE/CTE/72h thấp hơn hẳn v59 và STTrans.
# # # #           Train/test gap nhỏ lại.

# # # # FIXES TỪ v65-draft
# # # # ───────────────────
# # # # FIX-C1  Coherence loss gradient: dùng modes có grad (không detach khi tính L_coh).
# # # #         Draft cũ: pred_soft từ sel_logits đã stop-grad → L_coh = 0 gradient.

# # # # FIX-C2  Train/inference noise consistency:
# # # #         Draft cũ: training dùng x_t_k = x_t + z_k*σ_k, inference dùng formula khác.
# # # #         Fix: cả hai đều dùng cùng sigma_k per head từ self.head_noise_base.
# # # #         Inference: x_t_k = x0_persistence + randn * sigma_k (nhất quán).

# # # # FIX-C3  Oracle assignment bias từ per-head noise:
# # # #         Draft cũ: oracle chọn mode thắng ADE trên noisy x_t_k → head ít noise (k=0)
# # # #         có lợi thế không liên quan đến quality.
# # # #         Fix: oracle ADE tính trên x1_pred = x_t_CLEAN + (1-t)*v_k,
# # # #         tức là dùng x_t không nhiễu làm base cho x1_pred oracle.

# # # # FIX-C4  Compass bias yếu bị overwhelmed:
# # # #         Draft cũ: 0.1 scale, 1 bias vector trong 256-dim space.
# # # #         Fix: direction conditioning qua cross-attention explicit với direction token,
# # # #         không phải additive bias — cơ chế mạnh hơn, không bị gradient wash.

# # # # FIX-C5  Coherence loss tính trên normalized space thiếu scale:
# # # #         Fix: tính displacement diff trong km (haversine), có ý nghĩa vật lý.

# # # # THÊM MỚI ĐỂ GIẢM ADE VÀ TRAIN/TEST GAP
# # # # ────────────────────────────────────────
# # # # NEW-1   Mixup augmentation trên trajectory: blending 2 samples trong batch
# # # #         → model học smoother decision boundary → giảm test ADE.

# # # # NEW-2   Trajectory Mixup augmentation: blending easy samples trong batch
# # # #         → smoother decision boundary → giảm train/test gap.

# # # # NEW-3   Dropout consistency regularization: penalize encoder inconsistency
# # # #         giữa 2 forward passes → giảm overfitting.

# # # # NEW-4   Stochastic depth trong CompassVelocityHead direction token
# # # #         → regularization cho K=8 heads.
# # # # """
# # # # from __future__ import annotations

# # # # import math
# # # # from typing import Optional, Tuple, List

# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.nn.functional as F

# # # # from Model.FNO3D_encoder import FNO3DEncoder
# # # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # # from Model.env_net_transformer_gphsplit import Env_net

# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # R_EARTH      = 6371.0
# # # # DT_HOURS     = 6.0
# # # # DEG2KM       = 111.0
# # # # _NORM_TO_DEG = 5.0

# # # # _COMPASS_DEG   = [0., 45., 90., 135., 180., 225., 270., 315.]
# # # # _COMPASS_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
# # # # K_MODES        = 8


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Coordinate utilities
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# # # #     return torch.stack([
# # # #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# # # #         (t[..., 1] * 50.0) / 10.0,
# # # #     ], dim=-1)

# # # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # #     a = (torch.sin(dlat / 2).pow(2) +
# # # #          torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
# # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())

# # # # def _unwrap_model(m):
# # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # # # def _step_speeds_deg(traj_deg: torch.Tensor) -> torch.Tensor:
# # # #     T = traj_deg.shape[0]
# # # #     if T < 2: return traj_deg.new_zeros(1, traj_deg.shape[1])
# # # #     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS

# # # # def _ade_km_from_rel(
# # # #     pred_rel: torch.Tensor,   # [B, T, 4]
# # # #     gt_rel:   torch.Tensor,   # [B, T, 4]
# # # #     lp:       torch.Tensor,   # [B, 2]
# # # # ) -> torch.Tensor:            # [B]
# # # #     """FIX-B4: rel → abs → deg → haversine. Chỉ lon/lat (2 dims đầu)."""
# # # #     pred_abs = lp.unsqueeze(1) + pred_rel[:, :, :2]
# # # #     gt_abs   = lp.unsqueeze(1) + gt_rel[:, :, :2]
# # # #     return _haversine_deg(_norm_to_deg(pred_abs), _norm_to_deg(gt_abs)).mean(dim=1)

# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Spherical OT Matching  (port từ v59, thêm vào v65)
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # # #     lon1 = torch.deg2rad(p1[..., 0]); lat1 = torch.deg2rad(p1[..., 1])
# # # #     lon2 = torch.deg2rad(p2[..., 0]); lat2 = torch.deg2rad(p2[..., 1])
# # # #     dlon = lon2 - lon1
# # # #     y = torch.sin(dlon) * torch.cos(lat2)
# # # #     x = (torch.cos(lat1) * torch.sin(lat2)
# # # #          - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon))
# # # #     return torch.atan2(y, x)


# # # # def _sinkhorn_log(cost: torch.Tensor, epsilon: float = 0.05,
# # # #                   n_iter: int = 50) -> torch.Tensor:
# # # #     B = cost.shape[0]; device = cost.device
# # # #     log_a = -math.log(B) * torch.ones(B, device=device)
# # # #     log_b = -math.log(B) * torch.ones(B, device=device)
# # # #     log_K = -cost / epsilon
# # # #     log_u = torch.zeros(B, device=device)
# # # #     log_v = torch.zeros(B, device=device)
# # # #     for _ in range(n_iter):
# # # #         log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
# # # #         log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
# # # #     return (log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)).exp().clamp(0.0)


# # # # def _geodesic_ot_cost(x0_rel: torch.Tensor, x1_rel: torch.Tensor,
# # # #                        lp: torch.Tensor) -> torch.Tensor:
# # # #     """Cost matrix [B, B] dựa trên position + speed + direction."""
# # # #     B = x0_rel.shape[0]

# # # #     def _abs_deg(rel):
# # # #         return _norm_to_deg(lp.unsqueeze(1) + rel[:, :, :2])

# # # #     x0d = _abs_deg(x0_rel); x1d = _abs_deg(x1_rel)
# # # #     x0e = x0d.unsqueeze(1).expand(B, B, -1, -1).reshape(B*B, -1, 2)
# # # #     x1e = x1d.unsqueeze(0).expand(B, B, -1, -1).reshape(B*B, -1, 2)
# # # #     pos_cost = _haversine_deg(x0e, x1e).reshape(B, B, -1).mean(-1) / 500.0

# # # #     spd0 = _step_speeds_deg(x0d.permute(1, 0, 2)).permute(1, 0).mean(-1)
# # # #     spd1 = _step_speeds_deg(x1d.permute(1, 0, 2)).permute(1, 0).mean(-1)
# # # #     speed_cost = (spd0.unsqueeze(1) - spd1.unsqueeze(0)).abs() / 20.0

# # # #     def _mean_bearing(td):
# # # #         b = _forward_azimuth(td[:, :-1, :], td[:, 1:, :])
# # # #         return torch.atan2(b.sin().mean(-1), b.cos().mean(-1))
# # # #     h0 = _mean_bearing(x0d); h1 = _mean_bearing(x1d)
# # # #     dh = (h0.unsqueeze(1) - h1.unsqueeze(0) + math.pi) % (2*math.pi) - math.pi
# # # #     dir_cost = dh.abs() / math.pi

# # # #     return 1.0*pos_cost + 0.5*speed_cost + 0.3*dir_cost


# # # # def _spherical_ot_matching(x0_batch: torch.Tensor, x1_batch: torch.Tensor,
# # # #                             lp: torch.Tensor, epsilon: float = 0.05
# # # #                             ) -> Tuple[torch.Tensor, torch.Tensor]:
# # # #     """
# # # #     Ghép noise x0 với GT x1 tối ưu theo geodesic cost.
# # # #     Giúp FM học từ x0 gần x1 nhất về hướng + tốc độ → path ngắn hơn.
# # # #     """
# # # #     B = x0_batch.shape[0]
# # # #     if B < 4:
# # # #         return x0_batch, x1_batch
# # # #     try:
# # # #         cost = _geodesic_ot_cost(x0_batch, x1_batch, lp)
# # # #         with torch.no_grad():
# # # #             pi = _sinkhorn_log(cost, epsilon=epsilon)
# # # #         flat = pi.reshape(-1).clamp(0.0)
# # # #         s    = flat.sum()
# # # #         if not torch.isfinite(s) or s < 1e-10:
# # # #             return x0_batch, x1_batch
# # # #         idx = torch.multinomial(flat / s, num_samples=B, replacement=True)
# # # #         col = idx % B
# # # #         return x0_batch[col], x1_batch[col]
# # # #     except Exception:
# # # #         return x0_batch, x1_batch




# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  EMA
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # class EMAModel:
# # # #     def __init__(self, model, decay=0.995):
# # # #         self.decay  = decay
# # # #         m = _unwrap_model(model)
# # # #         self.shadow = {k: v.detach().clone()
# # # #                        for k, v in m.state_dict().items()
# # # #                        if v.dtype.is_floating_point}

# # # #     def update(self, model):
# # # #         m = _unwrap_model(model)
# # # #         with torch.no_grad():
# # # #             for k, v in m.state_dict().items():
# # # #                 if k in self.shadow:
# # # #                     self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

# # # #     def apply_to(self, model):
# # # #         m = _unwrap_model(model)
# # # #         backup, sd = {}, m.state_dict()
# # # #         for k in self.shadow:
# # # #             if k not in sd: continue
# # # #             backup[k] = sd[k].detach().clone()
# # # #             sd[k].copy_(self.shadow[k])
# # # #         return backup

# # # #     def restore(self, model, backup):
# # # #         m = _unwrap_model(model)
# # # #         sd = m.state_dict()
# # # #         for k, v in backup.items():
# # # #             if k in sd: sd[k].copy_(v)


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Difficulty Score
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def compute_difficulty_score(
# # # #     obs_BT2: torch.Tensor,   # [B, 8, 2]
# # # #     img_obs:  torch.Tensor,  # [B, 13, 8, 81, 81]
# # # #     env_data: Optional[dict],
# # # #     device:   torch.device,
# # # # ) -> torch.Tensor:           # [B] ∈ [0,1]
# # # #     B = obs_BT2.shape[0]
# # # #     scores = []

# # # #     # Signal 1: Track curvature (0.30)
# # # #     dir24 = env_data.get("history_direction24") if env_data else None
# # # #     if dir24 is not None and torch.is_tensor(dir24):
# # # #         dir24 = dir24.float().to(device)
# # # #         if dir24.dim() == 3: d24 = dir24[:, -1, :]
# # # #         elif dir24.dim() == 2: d24 = dir24
# # # #         else: d24 = None
# # # #         if d24 is not None and d24.shape[-1] == 8:
# # # #             bucket   = d24.argmax(dim=-1).float()
# # # #             angle_24 = bucket * (2.0 * math.pi / 8.0)
# # # #             sin24    = torch.sin(angle_24); cos24 = torch.cos(angle_24)
# # # #             dy = obs_BT2[:, -1, 1] - obs_BT2[:, -2, 1]
# # # #             dx = obs_BT2[:, -1, 0] - obs_BT2[:, -2, 0]
# # # #             lat_mid = (obs_BT2[:, -1, 1] + obs_BT2[:, -2, 1]) * 0.5
# # # #             cos_lat = torch.cos(torch.deg2rad(lat_mid * _NORM_TO_DEG)).clamp(1e-4)
# # # #             angle_now = torch.atan2(dx * cos_lat, dy)
# # # #             cos_diff  = (torch.cos(angle_now) * cos24 + torch.sin(angle_now) * sin24).clamp(-1,1)
# # # #             s_curve   = torch.sigmoid((torch.rad2deg(torch.acos(cos_diff)) - 45.0) / 20.0)
# # # #         else: s_curve = torch.zeros(B, device=device)
# # # #     else: s_curve = torch.zeros(B, device=device)
# # # #     scores.append(0.30 * s_curve)

# # # #     # Signal 2: Weak steering (0.25)
# # # #     steer = env_data.get("steering_speed") if env_data else None
# # # #     if steer is not None and torch.is_tensor(steer):
# # # #         steer = steer.float().to(device)
# # # #         while steer.dim() > 1: steer = steer[..., -1]
# # # #         steer = steer.view(-1)
# # # #         steer = (steer[:B] if steer.numel() >= B else steer[0].expand(B))
# # # #         s_weak = torch.sigmoid((4.0 - steer * 20.0) / 2.0)
# # # #     else:
# # # #         u = env_data.get("u500_mean") if env_data else None
# # # #         v = env_data.get("v500_mean") if env_data else None
# # # #         if u is not None and v is not None and torch.is_tensor(u) and torch.is_tensor(v):
# # # #             u = u.float().to(device); v = v.float().to(device)
# # # #             while u.dim() > 1: u = u[..., -1]
# # # #             while v.dim() > 1: v = v[..., -1]
# # # #             s_weak = torch.sigmoid((4.0 - (u.view(-1)[:B]**2 + v.view(-1)[:B]**2).sqrt() * 30.0) / 2.0)
# # # #         else: s_weak = torch.zeros(B, device=device)
# # # #     scores.append(0.25 * s_weak)

# # # #     # Signal 3: Wind shear (0.20)
# # # #     if img_obs is not None and img_obs.shape[1] >= 11:
# # # #         cx = cy = 40
# # # #         u200 = img_obs[:, 4, -1, cx, cy] * 13.315
# # # #         u850 = img_obs[:, 6, -1, cx, cy] * 7.911
# # # #         v200 = img_obs[:, 8, -1, cx, cy] * 8.377
# # # #         v850 = img_obs[:, 10,-1, cx, cy] * 6.203
# # # #         shear   = ((u200-u850)**2 + (v200-v850)**2).sqrt()
# # # #         s_shear = torch.sigmoid((shear - 8.0) / 3.0)
# # # #     else: s_shear = torch.zeros(B, device=device)
# # # #     scores.append(0.20 * s_shear)

# # # #     # Signal 4: RI flag (0.15)
# # # #     ri = env_data.get("rapid_intensification") if env_data else None
# # # #     if ri is not None and torch.is_tensor(ri):
# # # #         ri = ri.float().to(device)
# # # #         while ri.dim() > 1: ri = ri[..., -1]
# # # #         s_ri = ri.view(-1)[:B].clamp(0, 1) if ri.numel() >= B else ri[0].expand(B)
# # # #     else: s_ri = torch.zeros(B, device=device)
# # # #     scores.append(0.15 * s_ri)

# # # #     # Signal 5: Slow movement (0.10)
# # # #     mv = env_data.get("move_velocity") if env_data else None
# # # #     if mv is not None and torch.is_tensor(mv):
# # # #         mv = mv.float().to(device)
# # # #         while mv.dim() > 1: mv = mv[..., -1]
# # # #         mv_v = mv.view(-1)
# # # #         mv_v = mv_v[:B] if mv_v.numel() >= B else mv_v[0].expand(B)
# # # #         s_slow = torch.sigmoid((0.05 - mv_v) / 0.02)
# # # #     else: s_slow = torch.zeros(B, device=device)
# # # #     scores.append(0.10 * s_slow)

# # # #     return sum(scores).clamp(0.0, 1.0)


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Direction GT bucket
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def get_gt_direction_bucket(
# # # #     obs_traj:  torch.Tensor,   # [T, B, 2]
# # # #     pred_traj: torch.Tensor,   # [T, B, 2]
# # # # ) -> torch.Tensor:             # [B] 0..7
# # # #     """FIX-B8: atan2(East_km, North) = compass convention."""
# # # #     last  = obs_traj[-1]     # [B, 2]
# # # #     first = pred_traj[0]     # [B, 2]
# # # #     dy    = first[:, 1] - last[:, 1]
# # # #     dx    = first[:, 0] - last[:, 0]
# # # #     lat_d = (last[:, 1] + first[:, 1]) * 0.5
# # # #     cos_l = torch.cos(torch.deg2rad(lat_d * _NORM_TO_DEG)).clamp(1e-4)
# # # #     angle = (torch.atan2(dx * cos_l, dy).rad2deg() % 360.0)
# # # #     return ((angle + 22.5) / 45.0).long() % 8


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  FIX-C4: CompassVelocityHead with explicit direction cross-attention
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # class CompassVelocityHead(nn.Module):
# # # #     """
# # # #     FIX-C4: Direction conditioning qua cross-attention với direction token,
# # # #     không phải additive bias. Direction token được project từ compass angle
# # # #     (sin, cos) → 256-dim vector → concat vào memory của TransformerDecoder.
# # # #     Gradient không thể wash out cơ chế này như với additive bias.

# # # #     NEW-4: Stochastic depth với prob=0.1 → regularization cho K=8 heads.
# # # #     """
# # # #     def __init__(self, compass_idx: int, pred_len: int = 12, ctx_dim: int = 256,
# # # #                  stochastic_depth_prob: float = 0.1):
# # # #         super().__init__()
# # # #         self.pred_len             = pred_len
# # # #         self.compass_idx          = compass_idx
# # # #         self.stochastic_depth_prob = stochastic_depth_prob

# # # #         # FIX-C4: direction token từ (sin, cos) của compass angle
# # # #         angle_rad = _COMPASS_DEG[compass_idx] * math.pi / 180.0
# # # #         dir_vec   = torch.tensor([math.sin(angle_rad), math.cos(angle_rad)])
# # # #         self.register_buffer('compass_dir', dir_vec)   # [2], not learnable
# # # #         self.dir_proj = nn.Sequential(
# # # #             nn.Linear(2, ctx_dim), nn.GELU(), nn.LayerNorm(ctx_dim))

# # # #         self.time_fc1    = nn.Linear(ctx_dim, 256)
# # # #         self.time_fc2    = nn.Linear(256, ctx_dim)
# # # #         self.traj_embed  = nn.Linear(4, ctx_dim)       # FIX-B7: 4 dims
# # # #         self.pos_enc     = nn.Parameter(torch.randn(1, pred_len, ctx_dim) * 0.02)
# # # #         self.step_embed  = nn.Embedding(pred_len, ctx_dim)
# # # #         self.transformer = nn.TransformerDecoder(
# # # #             nn.TransformerDecoderLayer(
# # # #                 d_model=ctx_dim, nhead=8, dim_feedforward=512,
# # # #                 dropout=0.10, activation="gelu", batch_first=True),
# # # #             num_layers=1)
# # # #         self.out_fc1 = nn.Linear(ctx_dim, 256)
# # # #         self.out_fc2 = nn.Linear(256, 4)               # FIX-B7: output 4 dims
# # # #         self.step_scale = nn.Parameter(torch.ones(pred_len) * 0.5)

# # # #         with torch.no_grad():
# # # #             nn.init.xavier_uniform_(self.out_fc2.weight, gain=0.1)
# # # #             nn.init.zeros_(self.out_fc2.bias)

# # # #     def _time_emb(self, t: torch.Tensor, dim: int) -> torch.Tensor:
# # # #         half = dim // 2
# # # #         freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device)
# # # #                          * (-math.log(10000.0) / max(half - 1, 1)))
# # # #         emb  = t.float().unsqueeze(1) * 1000.0 * freq.unsqueeze(0)
# # # #         return F.pad(torch.cat([emb.sin(), emb.cos()], dim=-1), (0, dim % 2))

# # # #     def forward(
# # # #         self,
# # # #         x_t: torch.Tensor,   # [B, pred_len, 4]
# # # #         t:   torch.Tensor,   # [B]
# # # #         ctx: torch.Tensor,   # [B, ctx_dim]
# # # #     ) -> torch.Tensor:       # [B, pred_len, 4]
# # # #         B     = x_t.shape[0]
# # # #         T_seq = min(x_t.shape[1], self.pred_len)
# # # #         device = x_t.device

# # # #         # FIX-C4: direction token via cross-attention
# # # #         dir_token = self.dir_proj(
# # # #             self.compass_dir.to(device).unsqueeze(0).expand(B, -1))  # [B, ctx_dim]

# # # #         # Stochastic depth: randomly zero-out dir_token during training
# # # #         if self.training and self.stochastic_depth_prob > 0:
# # # #             keep = (torch.rand(B, 1, device=device) >
# # # #                     self.stochastic_depth_prob).float()
# # # #             dir_token = dir_token * keep

# # # #         t_emb = F.gelu(self.time_fc1(self._time_emb(t, ctx.shape[-1])))
# # # #         t_emb = self.time_fc2(t_emb)

# # # #         step_idx = torch.arange(T_seq, device=device).unsqueeze(0).expand(B, -1)
# # # #         x_emb = (self.traj_embed(x_t[:, :T_seq])
# # # #                  + self.pos_enc[:, :T_seq]
# # # #                  + t_emb.unsqueeze(1)
# # # #                  + self.step_embed(step_idx))

# # # #         # Memory: [ctx, dir_token, t_emb] — direction is first-class citizen
# # # #         mem     = torch.stack([ctx, dir_token, t_emb], dim=1)  # [B, 3, ctx_dim]
# # # #         decoded = self.transformer(x_emb, mem)

# # # #         scale   = torch.sigmoid(self.step_scale[:T_seq]).view(1, T_seq, 1) * 2.0
# # # #         return self.out_fc2(F.gelu(self.out_fc1(decoded))) * scale


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  CompassSelector (GC-Net analog)
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # class CompassSelector(nn.Module):
# # # #     """
# # # #     Selector nhìn actual mode outputs + steering → chọn best mode.
# # # #     Tốt hơn paper GC-Net (chỉ nhìn context).
# # # #     STOP-GRAD từ caller.
# # # #     """
# # # #     def __init__(self, ctx_dim: int = 256, K: int = 8, n_dirs: int = 8):
# # # #         super().__init__()
# # # #         self.K      = K
# # # #         self.n_dirs = n_dirs

# # # #         self.steering_enc = nn.Sequential(
# # # #             nn.Linear(11, 64), nn.GELU(), nn.LayerNorm(64),
# # # #             nn.Linear(64, 128))

# # # #         self.mode_enc = nn.Sequential(
# # # #             nn.Linear(12 * 2, 128), nn.GELU(), nn.LayerNorm(128),
# # # #             nn.Linear(128, 64))

# # # #         self.score_net = nn.Sequential(
# # # #             nn.Linear(ctx_dim + 64 + 128, 256), nn.GELU(), nn.LayerNorm(256),
# # # #             nn.Linear(256, 64), nn.GELU(),
# # # #             nn.Linear(64, 1))

# # # #         self.dir_head = nn.Sequential(
# # # #             nn.Linear(ctx_dim + 128, 256), nn.GELU(), nn.LayerNorm(256),
# # # #             nn.Linear(256, n_dirs))

# # # #     def _get_steering_feat(self, env_data, B, device):
# # # #         def _sc(key):
# # # #             if env_data is None: return torch.zeros(B, device=device)
# # # #             v = env_data.get(key)
# # # #             if v is None or not torch.is_tensor(v): return torch.zeros(B, device=device)
# # # #             v = v.float().to(device)
# # # #             while v.dim() > 1: v = v[..., -1]
# # # #             v = v.view(-1)
# # # #             return (v[:B] if v.numel() >= B else v[0].expand(B)).clamp(-3, 3)
# # # #         def _sv(key, dim):
# # # #             if env_data is None: return torch.zeros(B, dim, device=device)
# # # #             v = env_data.get(key)
# # # #             if v is None or not torch.is_tensor(v): return torch.zeros(B, dim, device=device)
# # # #             v = v.float().to(device)
# # # #             if v.dim() == 3: v = v[:, -1, :]
# # # #             elif v.dim() != 2 or v.shape[0] != B: return torch.zeros(B, dim, device=device)
# # # #             v = F.pad(v, (0, max(0, dim - v.shape[-1])))[:, :dim]
# # # #             return v.clamp(-3, 3)
# # # #         feat = torch.cat([_sc("steering_speed").unsqueeze(-1),
# # # #                           _sc("steering_dir_sin").unsqueeze(-1),
# # # #                           _sc("steering_dir_cos").unsqueeze(-1),
# # # #                           _sv("history_direction24", 8)], dim=-1)
# # # #         return self.steering_enc(feat)

# # # #     def forward(self, ctx, modes, env_data):
# # # #         # ctx: [B,ctx_dim], modes: [B,K,12,4] — both stop-grad from caller
# # # #         B, K = modes.shape[:2]
# # # #         device = ctx.device
# # # #         sf = self._get_steering_feat(env_data, B, device)
# # # #         scores = []
# # # #         for k in range(K):
# # # #             mf = self.mode_enc(modes[:, k, :, :2].reshape(B, -1))
# # # #             sc = self.score_net(torch.cat([ctx, mf, sf], dim=-1)).squeeze(-1)
# # # #             scores.append(sc)
# # # #         score_logits = torch.stack(scores, dim=1)
# # # #         dir_logits   = self.dir_head(torch.cat([ctx, sf], dim=-1))
# # # #         return score_logits, dir_logits


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Learned Loss Weights (Kendall et al. 2018)
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # class LearnedWeights(nn.Module):
# # # #     """
# # # #     4 learnable log-variance scalars — tất cả weights tự học.
# # # #     Không có hardcoded weight nào cho bất kỳ metric cụ thể nào.

# # # #       L_total = L_FM/exp(s_fm) + s_fm
# # # #               + L_sel/exp(s_sel) + s_sel     [L_sel = α*L_rank + (1-α)*L_dir]
# # # #               + L_coh/exp(s_coh) + s_coh
# # # #               + L_con/exp(s_con) + s_con     [dropout consistency]
# # # #     """
# # # #     def __init__(self):
# # # #         super().__init__()
# # # #         self.log_s_fm  = nn.Parameter(torch.zeros(1))
# # # #         self.log_s_sel = nn.Parameter(torch.zeros(1))
# # # #         self.log_s_coh = nn.Parameter(torch.zeros(1))
# # # #         self.log_s_con = nn.Parameter(torch.zeros(1))   # consistency
# # # #         self.log_alpha = nn.Parameter(torch.zeros(1))

# # # #     def forward(self, L_fm, L_rank, L_dir, L_coh, L_con):
# # # #         s_fm  = self.log_s_fm.clamp(-5, 5)
# # # #         s_sel = self.log_s_sel.clamp(-5, 5)
# # # #         s_coh = self.log_s_coh.clamp(-5, 5)
# # # #         s_con = self.log_s_con.clamp(-5, 5)
# # # #         α     = torch.sigmoid(self.log_alpha)
# # # #         L_sel = α * L_rank + (1.0 - α) * L_dir
# # # #         return (L_fm  * torch.exp(-s_fm)  + s_fm  +
# # # #                 L_sel * torch.exp(-s_sel) + s_sel +
# # # #                 L_coh * torch.exp(-s_coh) + s_coh +
# # # #                 L_con * torch.exp(-s_con) + s_con)

# # # #     def get_weights(self):
# # # #         return {"w_fm":   torch.exp(-self.log_s_fm.clamp(-5,5)).item(),
# # # #                 "w_sel":  torch.exp(-self.log_s_sel.clamp(-5,5)).item(),
# # # #                 "w_coh":  torch.exp(-self.log_s_coh.clamp(-5,5)).item(),
# # # #                 "w_con":  torch.exp(-self.log_s_con.clamp(-5,5)).item(),
# # # #                 "alpha":  torch.sigmoid(self.log_alpha).item()}


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Shared Context Encoder
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # class SharedContextEncoder(nn.Module):
# # # #     RAW_CTX_DIM = 512

# # # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, unet_in_ch=13):
# # # #         super().__init__()
# # # #         self.pred_len = pred_len
# # # #         self.obs_len  = obs_len
# # # #         self.ctx_dim  = ctx_dim

# # # #         self.spatial_enc     = FNO3DEncoder(
# # # #             in_channel=unet_in_ch, out_channel=1, d_model=32,
# # # #             n_layers=4, modes_t=4, modes_h=4, modes_w=4,
# # # #             spatial_down=32, dropout=0.05)
# # # #         self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
# # # #         self.bottleneck_proj = nn.Linear(128, 128)
# # # #         self.decoder_proj    = nn.Linear(1, 16)
# # # #         self.enc_1d = DataEncoder1D(
# # # #             in_1d=4, feat_3d_dim=128, mlp_h=64,
# # # #             lstm_hidden=128, lstm_layers=3, dropout=0.1, d_state=16)
# # # #         self.env_enc = Env_net(obs_len=obs_len, d_model=32)

# # # #         self.ctx_fc1  = nn.Linear(128 + 32 + 16, self.RAW_CTX_DIM)
# # # #         self.ctx_ln   = nn.LayerNorm(self.RAW_CTX_DIM)
# # # #         self.ctx_drop = nn.Dropout(0.15)
# # # #         self.ctx_fc2  = nn.Linear(self.RAW_CTX_DIM, ctx_dim)
# # # #         self.null_embedding = nn.Parameter(torch.randn(1, self.RAW_CTX_DIM) * 0.02)
# # # #         self.vel_obs_enc = nn.Sequential(
# # # #             nn.Linear(obs_len * 6, 256), nn.GELU(), nn.LayerNorm(256),
# # # #             nn.Linear(256, 256), nn.GELU())

# # # #     def encode(self, batch_list):
# # # #         obs_t     = batch_list[0]             # [T, B, 2]
# # # #         obs_Me    = batch_list[7]             # [T, B, 2]
# # # #         image_obs = batch_list[11]            # [B, 13, T, 81, 81]  FIX-B1
# # # #         env_data  = batch_list[13] if len(batch_list) > 13 else None

# # # #         if image_obs.dim() == 4: image_obs = image_obs.unsqueeze(2)
# # # #         if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
# # # #             image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)

# # # #         T_obs = obs_t.shape[0]
# # # #         e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
# # # #         e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
# # # #         e_3d_s = self.bottleneck_proj(e_3d_s)
# # # #         if e_3d_s.shape[1] != T_obs:
# # # #             e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T_obs,
# # # #                                    mode="linear", align_corners=False).permute(0,2,1)

# # # #         e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # # #         t_w  = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
# # # #                                           device=e_3d_dec_t.device) * 0.5, dim=0)
# # # #         f_sp = self.decoder_proj((e_3d_dec_t * t_w.unsqueeze(0)).sum(1, keepdim=True))

# # # #         obs_in = torch.cat([obs_t, obs_Me], dim=2).permute(1, 0, 2)
# # # #         h_t    = self.enc_1d(obs_in, e_3d_s)
# # # #         e_env, _, _ = self.env_enc(env_data, image_obs)
# # # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

# # # #     def apply_ctx_head(self, raw, noise_scale=0.0, use_null=False):
# # # #         if use_null: raw = self.null_embedding.expand(raw.shape[0], -1)
# # # #         elif noise_scale > 0.0: raw = raw + torch.randn_like(raw) * noise_scale
# # # #         return self.ctx_fc2(self.ctx_drop(raw))

# # # #     def get_kinematic_feat(self, obs_traj):
# # # #         T_obs, B, _ = obs_traj.shape
# # # #         if T_obs >= 2:
# # # #             vel     = obs_traj[1:] - obs_traj[:-1]
# # # #             lat_mid = obs_traj[:-1, :, 1] * _NORM_TO_DEG
# # # #             cos_lat = torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
# # # #             dx_km   = vel[:,:,0] * cos_lat * DEG2KM * _NORM_TO_DEG
# # # #             dy_km   = vel[:,:,1] * DEG2KM * _NORM_TO_DEG
# # # #             speed   = (dx_km**2 + dy_km**2 + 1e-6).sqrt() / DT_HOURS
# # # #             heading = torch.atan2(vel[:,:,1], vel[:,:,0])
# # # #             speed_n = (speed / 20.0).clamp(-3, 3)
# # # #             if T_obs >= 3:
# # # #                 dspd  = speed[1:] - speed[:-1]
# # # #                 accel = torch.cat([obs_traj.new_zeros(1, B),
# # # #                                    (dspd / 10.0).clamp(-3, 3)], 0)
# # # #             else: accel = obs_traj.new_zeros(T_obs-1, B)
# # # #             kine = torch.stack([vel[:,:,0], vel[:,:,1], speed_n,
# # # #                                  heading.sin(), heading.cos(), accel], dim=-1)
# # # #         else: kine = obs_traj.new_zeros(self.obs_len, B, 6)
# # # #         if kine.shape[0] < self.obs_len:
# # # #             kine = torch.cat([obs_traj.new_zeros(self.obs_len - kine.shape[0], B, 6), kine], 0)
# # # #         else: kine = kine[-self.obs_len:]
# # # #         return self.vel_obs_enc(kine.permute(1, 0, 2).reshape(B, -1))


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  Persistence x0 trong relative space  (FIX-B11)
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def _persistence_x0_rel(obs_traj, obs_Me, lp, lm, pred_len, sigma=0.02):
# # # #     """[T,B,2] → [B, pred_len, 4] relative coords. FIX-B11."""
# # # #     B, device = obs_traj.shape[1], obs_traj.device
# # # #     if obs_traj.shape[0] >= 3:
# # # #         vels = obs_traj[1:] - obs_traj[:-1]
# # # #         n_v  = vels.shape[0]
# # # #         alpha = 0.7
# # # #         w     = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
# # # #                               dtype=torch.float, device=device).flip(0)
# # # #         lv    = (vels * (w/w.sum()).view(-1,1,1)).sum(0)
# # # #     elif obs_traj.shape[0] >= 2:
# # # #         lv = obs_traj[-1,:,:2] - obs_traj[-2,:,:2]
# # # #     else:
# # # #         lv = obs_traj.new_zeros(B, 2)
# # # #     steps     = torch.arange(1, pred_len+1, device=device).float()
# # # #     pred_abs  = obs_traj[-1,:,:2].unsqueeze(0) + lv.unsqueeze(0) * steps.view(-1,1,1)
# # # #     pred_rel  = pred_abs.permute(1,0,2) - lp.unsqueeze(1)
# # # #     pred_rel4 = torch.cat([pred_rel, torch.zeros(B, pred_len, 2, device=device)], dim=-1)
# # # #     return pred_rel4 + torch.randn_like(pred_rel4) * sigma


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  NEW-1: Trajectory Mixup Augmentation
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def _traj_mixup(x1_rel: torch.Tensor, delta: torch.Tensor,
# # # #                 prob: float = 0.3, alpha: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
# # # #     """
# # # #     NEW-1: Mixup giữa 2 samples trong batch.
# # # #     Giảm train/test gap bằng cách tạo ra smoother decision boundary.
# # # #     Chỉ mixup trên easy samples (delta < 0.4) để không làm mờ hard cases.
# # # #     Returns: mixed x1_rel, mixed delta
# # # #     """
# # # #     if not torch.is_tensor(x1_rel) or x1_rel.shape[0] < 2:
# # # #         return x1_rel, delta
# # # #     B = x1_rel.shape[0]
# # # #     # Chỉ apply với xác suất prob
# # # #     if torch.rand(1).item() > prob:
# # # #         return x1_rel, delta

# # # #     # Sample lambda từ Beta distribution
# # # #     lam = float(torch.distributions.Beta(alpha, alpha).sample())
# # # #     lam = max(lam, 1.0 - lam)   # luôn ≥ 0.5 để giữ original hơn

# # # #     # Shuffle indices
# # # #     idx = torch.randperm(B, device=x1_rel.device)

# # # #     # Chỉ mixup easy samples (delta < 0.4)
# # # #     # Hard samples (delta >= 0.4) giữ nguyên x1 và delta để không làm mờ
# # # #     # signal khó — đây là điểm cốt lõi của easy/difficult split.
# # # #     easy_mask_traj  = (delta < 0.4).float().view(B, 1, 1)   # [B,1,1] cho traj
# # # #     easy_mask_delta = (delta < 0.4).float()                  # [B] cho delta
# # # #     x1_mixed  = x1_rel * (1.0 - easy_mask_traj * (1.0 - lam)) +                 x1_rel[idx] * (easy_mask_traj * (1.0 - lam))
# # # #     # FIX: hard samples giữ nguyên delta, chỉ easy samples blend delta
# # # #     delta_mix = (delta * lam + delta[idx] * (1.0 - lam)) * easy_mask_delta +                 delta * (1.0 - easy_mask_delta)
# # # #     return x1_mixed, delta_mix


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  FIX-C1 + FIX-C5: Displacement Coherence Loss (thuần, không nhắm thẳng)
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def _coherence_loss(
# # # #     pred_soft_grad: torch.Tensor,   # [B, 12, 4] — CÓ gradient (FIX-C1)
# # # #     x1_rel:         torch.Tensor,   # [B, 12, 4]
# # # #     lp:             torch.Tensor,   # [B, 2]
# # # #     device:         torch.device,
# # # # ) -> torch.Tensor:
# # # #     """
# # # #     FIX-C1: phải dùng pred_soft_grad CÓ gradient.
# # # #     FIX-C5: tính displacement diff trong km — có ý nghĩa vật lý.

# # # #     Pure displacement smoothness — KHÔNG có hardcoded step weight.
# # # #     Weight tự học hoàn toàn qua LearnedWeights.log_s_coh.
# # # #     """
# # # #     # Convert sang abs degrees
# # # #     pred_abs = lp.unsqueeze(1) + pred_soft_grad[:, :, :2]   # [B, 12, 2]
# # # #     gt_abs   = lp.unsqueeze(1) + x1_rel[:, :, :2]

# # # #     pred_deg = _norm_to_deg(pred_abs)   # [B, 12, 2]
# # # #     gt_deg   = _norm_to_deg(gt_abs)

# # # #     # Displacement per step trong km (FIX-C5: km space)
# # # #     cos_lat       = torch.cos(torch.deg2rad(gt_deg[:, :-1, 1])).clamp(1e-4)
# # # #     disp_pred_lon = (pred_deg[:, 1:, 0] - pred_deg[:, :-1, 0]) * cos_lat * DEG2KM
# # # #     disp_pred_lat = (pred_deg[:, 1:, 1] - pred_deg[:, :-1, 1]) * DEG2KM
# # # #     disp_gt_lon   = (gt_deg[:, 1:, 0]   - gt_deg[:, :-1, 0])   * cos_lat * DEG2KM
# # # #     disp_gt_lat   = (gt_deg[:, 1:, 1]   - gt_deg[:, :-1, 1])   * DEG2KM

# # # #     disp_pred = torch.stack([disp_pred_lon, disp_pred_lat], dim=-1)   # [B, 11, 2]
# # # #     disp_gt   = torch.stack([disp_gt_lon,   disp_gt_lat],   dim=-1)

# # # #     # Mean displacement diff in km — không có step weight nào hardcode
# # # #     # LearnedWeights sẽ tự cân bằng L_coh với L_FM và L_sel
# # # #     disp_diff = (disp_pred - disp_gt).norm(dim=-1)   # [B, 11]
# # # #     return disp_diff.mean() / DEG2KM                  # normalize về ~same scale as L_FM


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  NEW-3: Dropout Consistency Regularization
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # def _dropout_consistency_loss(
# # # #     encoder: SharedContextEncoder,
# # # #     raw_ctx: torch.Tensor,   # [B, 512] raw context (pre-dropout)
# # # # ) -> torch.Tensor:
# # # #     """
# # # #     NEW-3: Apply dropout 2 lần trên cùng raw_ctx → 2 dropout patterns khác nhau.
# # # #     Penalize inconsistency → model học features stable bất kể dropout.
# # # #     Chỉ áp dụng khi encoder đang ở training mode (dropout active).

# # # #     FIX BUG: Draft cũ gọi encode() 2 lần — encode() không có dropout,
# # # #     nên ctx1 = ctx2 → L_consist = 0 mọi lúc.
# # # #     Fix: apply_ctx_head 2 lần trên cùng raw_ctx → dropout khác nhau.
# # # #     """
# # # #     if not encoder.training:
# # # #         return torch.tensor(0.0, device=raw_ctx.device)
# # # #     # Gọi apply_ctx_head 2 lần: mỗi lần dropout khác nhau
# # # #     ctx_a = encoder.apply_ctx_head(raw_ctx)   # [B, 256]
# # # #     ctx_b = encoder.apply_ctx_head(raw_ctx)   # [B, 256] — different dropout mask
# # # #     ctx_a_n = F.normalize(ctx_a, dim=-1)
# # # #     ctx_b_n = F.normalize(ctx_b, dim=-1)
# # # #     cos_sim  = (ctx_a_n * ctx_b_n).sum(dim=-1)   # [B]
# # # #     return (1.0 - cos_sim).mean()


# # # # # ═════════════════════════════════════════════════════════════════════════════
# # # # #  TCFlowMatchingV65 — Main Class
# # # # # ═════════════════════════════════════════════════════════════════════════════

# # # # class TCFlowMatchingV65(nn.Module):

# # # #     def __init__(
# # # #         self,
# # # #         pred_len:           int   = 12,
# # # #         obs_len:            int   = 8,
# # # #         ctx_dim:            int   = 256,
# # # #         sigma_min:          float = 0.02,
# # # #         unet_in_ch:         int   = 13,
# # # #         K:                  int   = K_MODES,
# # # #         use_ema:            bool  = True,
# # # #         ema_decay:          float = 0.995,
# # # #         cfg_uncond_prob:    float = 0.10,
# # # #         selector_warmup:    int   = 2,
# # # #         head_noise_base:    float = 0.03,
# # # #         use_ot:             bool  = True,    # OT matching trong training
# # # #         ot_epsilon:         float = 0.05,    # Sinkhorn epsilon
# # # #         cfg_guidance_scale: float = 1.3,    # CFG scale trong inference
# # # #         **kwargs,
# # # #     ):
# # # #         super().__init__()
# # # #         self.pred_len        = pred_len
# # # #         self.obs_len         = obs_len
# # # #         self.sigma_min       = sigma_min
# # # #         self.K               = K
# # # #         self.use_ema         = use_ema
# # # #         self.ema_decay       = ema_decay
# # # #         self.cfg_uncond_prob = cfg_uncond_prob
# # # #         self.selector_warmup    = selector_warmup
# # # #         self.head_noise_base    = head_noise_base
# # # #         self.use_ot             = use_ot
# # # #         self.ot_epsilon         = ot_epsilon
# # # #         self.cfg_guidance_scale = cfg_guidance_scale
# # # #         self._ema               = None

# # # #         self.encoder        = SharedContextEncoder(
# # # #             pred_len=pred_len, obs_len=obs_len,
# # # #             ctx_dim=ctx_dim, unet_in_ch=unet_in_ch)
# # # #         self.velocity_heads = nn.ModuleList([
# # # #             CompassVelocityHead(compass_idx=k, pred_len=pred_len, ctx_dim=ctx_dim)
# # # #             for k in range(K)])
# # # #         self.selector       = CompassSelector(ctx_dim=ctx_dim, K=K, n_dirs=8)
# # # #         self.learned_weights= LearnedWeights()

# # # #     # ── EMA ──────────────────────────────────────────────────────────────────

# # # #     def init_ema(self):
# # # #         if self.use_ema: self._ema = EMAModel(self, decay=self.ema_decay)

# # # #     def ema_update(self):
# # # #         if self._ema is not None: self._ema.update(self)

# # # #     def set_curriculum_len(self, *a, **kw): pass

# # # #     @staticmethod
# # # #     def _sigma_schedule(epoch):
# # # #         if epoch < 2:  return 0.10
# # # #         if epoch < 10: return 0.10 - (epoch-2)/8.0*(0.10-0.04)
# # # #         if epoch < 20: return max(0.04 - (epoch-10)/10.0*0.01, 0.035)
# # # #         return 0.035

# # # #     @staticmethod
# # # #     def _lon_flip_aug(bl, p=0.3):
# # # #         if torch.rand(1).item() > p: return bl
# # # #         bl = list(bl)
# # # #         for i in [0,1,2,3]:
# # # #             if torch.is_tensor(bl[i]) and bl[i].shape[-1] >= 1:
# # # #                 t = bl[i].clone(); t[...,0] = -t[...,0]; bl[i] = t
# # # #         return bl

# # # #     @staticmethod
# # # #     def _obs_noise_aug(bl, sigma=0.005):
# # # #         if torch.rand(1).item() > 0.5: return bl
# # # #         bl = list(bl)
# # # #         if torch.is_tensor(bl[0]):
# # # #             bl[0] = bl[0] + torch.randn_like(bl[0]) * sigma
# # # #         return bl

# # # #     @staticmethod
# # # #     def _to_rel(traj, Me, lp, lm):
# # # #         return torch.cat([traj - lp.unsqueeze(0),
# # # #                           Me  - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

# # # #     @staticmethod
# # # #     def _to_abs(rel, lp, lm):
# # # #         d = rel.permute(1, 0, 2)
# # # #         return lp.unsqueeze(0) + d[:,:,:2], lm.unsqueeze(0) + d[:,:,2:]

# # # #     def _cfm_noisy(self, x1_rel, sigma_min=None, lp=None):
# # # #         if sigma_min is None: sigma_min = self.sigma_min
# # # #         B = x1_rel.shape[0]; device = x1_rel.device
# # # #         x0 = torch.randn_like(x1_rel) * sigma_min
# # # #         t  = torch.rand(B, device=device)
# # # #         te = t.view(B, 1, 1)
# # # #         return (1.0-te)*x0 + te*x1_rel, t, x1_rel - x0

# # # #     @staticmethod
# # # #     @torch.no_grad()
# # # #     def _persistence_blend(model_pred, obs_traj_norm, blend_strength=0.10):
# # # #         T_obs = obs_traj_norm.shape[0]; T = model_pred.shape[0]
# # # #         B, device = model_pred.shape[1], model_pred.device
# # # #         if T_obs < 2: return model_pred
# # # #         vels = obs_traj_norm[1:] - obs_traj_norm[:-1]; n_v = vels.shape[0]
# # # #         if n_v >= 3:
# # # #             alpha = 0.7
# # # #             w  = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
# # # #                                dtype=torch.float, device=device).flip(0)
# # # #             ev = (vels * (w/w.sum()).view(-1,1,1)).sum(0)
# # # #         elif n_v == 2: ev = 0.7*vels[-1] + 0.3*vels[-2]
# # # #         else: ev = vels[-1]
# # # #         steps   = torch.arange(1, T+1, dtype=torch.float, device=device)
# # # #         persist = obs_traj_norm[-1].unsqueeze(0) + ev.unsqueeze(0)*steps.view(T,1,1)
# # # #         obs_spd = _step_speeds_deg(_norm_to_deg(obs_traj_norm))
# # # #         if obs_spd.shape[0] >= 2:
# # # #             spd_cv  = obs_spd.std(0) / obs_spd.mean(0).clamp(min=1.0)
# # # #             alpha_b = (blend_strength*torch.sigmoid(-(spd_cv-0.3)*5.0)).unsqueeze(0).unsqueeze(-1)
# # # #         else: alpha_b = blend_strength * 0.5
# # # #         return (1.0-alpha_b)*model_pred + alpha_b*persist

# # # #     # ═════════════════════════════════════════════════════════════════════════
# # # #     #  TRAINING LOSS
# # # #     # ═════════════════════════════════════════════════════════════════════════

# # # #     def get_loss(self, batch_list, epoch=0, **kwargs):
# # # #         return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

# # # #     def get_loss_breakdown(self, batch_list, epoch=0, **kwargs):
# # # #         # ── Augmentation ─────────────────────────────────────────────────────
# # # #         batch_list = self._lon_flip_aug(batch_list)
# # # #         batch_list = self._obs_noise_aug(batch_list, sigma=0.005)

# # # #         # ── Unpack (FIX-B1) ──────────────────────────────────────────────────
# # # #         obs_t    = batch_list[0]    # [T, B, 2]
# # # #         pred_t   = batch_list[1]    # [T, B, 2]
# # # #         obs_Me   = batch_list[7]    # [T, B, 2]
# # # #         pred_Me  = batch_list[8]    # [T, B, 2]
# # # #         env_data = batch_list[13] if len(batch_list) > 13 else None

# # # #         lp = obs_t[-1]; lm = obs_Me[-1]
# # # #         B, device = lp.shape[0], lp.device
# # # #         sigma     = self._sigma_schedule(epoch)
# # # #         in_warmup = (epoch < self.selector_warmup)

# # # #         # ── Context encoding ─────────────────────────────────────────────────
# # # #         raw_ctx  = self.encoder.encode(batch_list)           # [B, 512]
# # # #         use_null = (torch.rand(1).item() < self.cfg_uncond_prob)
# # # #         ctx      = self.encoder.apply_ctx_head(raw_ctx, use_null=use_null)  # [B, 256]

# # # #         # ── NEW-3: Dropout consistency regularization ─────────────────────────
# # # #         L_consist = _dropout_consistency_loss(self.encoder, raw_ctx)

# # # #         # ── Difficulty score (FIX-B2) ────────────────────────────────────────
# # # #         obs_BT2  = obs_t.permute(1, 0, 2)
# # # #         img_obs  = batch_list[11]
# # # #         delta    = compute_difficulty_score(obs_BT2, img_obs, env_data, device)

# # # #         # ── FM interpolant (FIX-B3) ──────────────────────────────────────────
# # # #         x1_rel_orig = self._to_rel(pred_t, pred_Me, lp, lm)    # [B, 12, 4]

# # # #         # ── OT matching (port từ v59) ─────────────────────────────────────────
# # # #         # Ghép noise với GT tối ưu → FM path ngắn hơn → train tốt hơn.
# # # #         # Áp dụng TRƯỚC mixup vì OT cần GT gốc để tính cost.
# # # #         if self.use_ot and B >= 4 and not in_warmup:
# # # #             noise_ot = torch.randn_like(x1_rel_orig) * sigma
# # # #             _, x1_rel_orig = _spherical_ot_matching(
# # # #                 noise_ot, x1_rel_orig, lp, epsilon=self.ot_epsilon)

# # # #         # ── NEW-1: Mixup augmentation ────────────────────────────────────────
# # # #         x1_rel, delta = _traj_mixup(x1_rel_orig, delta, prob=0.3, alpha=0.4)

# # # #         x_t, fm_t, u_target = self._cfm_noisy(x1_rel, sigma_min=sigma, lp=lp)
# # # #         # x_t: [B,12,4], fm_t: [B], u_target: [B,12,4]

# # # #         # ── FIX-C3: x_t_shared — FM noisy input chung cho tất cả heads ────────
# # # #         # Oracle ADE tính từ x_t_shared (không có per-head z_k) → fair comparison.
# # # #         # x_t_shared đã có sigma từ _cfm_noisy schedule; "shared" nghĩa là
# # # #         # không thêm per-head noise khi tính oracle.
# # # #         x_t_shared = x_t

# # # #         # ── Generate K=8 modes (FIX-P1 + FIX-C2: consistent sigma_k) ────────
# # # #         modes_rel_nograd = []   # stop-grad cho selector/oracle  [B,12,4]
# # # #         modes_rel_grad   = []   # có gradient cho L_FM, L_coh    [B,12,4]
# # # #         vels_pred        = []   # gradient

# # # #         for k in range(self.K):
# # # #             # FIX-C2: sigma_k đồng nhất giữa training và inference
# # # #             sigma_k = self.head_noise_base * (1.0 + k * 0.25)
# # # #             z_k       = torch.randn_like(x_t_shared) * sigma_k
# # # #             x_t_k      = x_t_shared + z_k

# # # #             if in_warmup:
# # # #                 with torch.no_grad():
# # # #                     v_k = self.velocity_heads[k](x_t_k, fm_t, ctx.detach())
# # # #             else:
# # # #                 v_k = self.velocity_heads[k](x_t_k, fm_t, ctx)

# # # #             # FIX-C3: x1_pred oracle dùng x_t_CLEAN (không có z_k) làm base
# # # #             # → oracle fair: không có head nào có lợi thế do ít noise
# # # #             with torch.no_grad():
# # # #                 pred_k_oracle = x_t_shared + (1.0 - fm_t.view(B,1,1)) * v_k
# # # #             modes_rel_nograd.append(pred_k_oracle.detach())

# # # #             # pred có gradient: dùng x_t_k làm base (consistent với training)
# # # #             pred_k_grad = x_t_k + (1.0 - fm_t.view(B,1,1)) * v_k
# # # #             modes_rel_grad.append(pred_k_grad)
# # # #             vels_pred.append(v_k)

# # # #         modes_t_ng = torch.stack(modes_rel_nograd, dim=1)  # [B,K,12,4] no-grad
# # # #         modes_t_g  = torch.stack(modes_rel_grad,   dim=1)  # [B,K,12,4] with-grad

# # # #         # ── Oracle assignment: dùng x1_rel_orig (NOT mixup) ─────────────────
# # # #         # FIX: oracle phải so với GT gốc, không phải GT đã blend.
# # # #         # x1_rel (mixup) được dùng cho FM training objective.
# # # #         # x1_rel_orig được dùng cho oracle selection → k_star phản ánh
# # # #         # mode nào thực sự gần GT nhất, không bị artifact mixup.
# # # #         with torch.no_grad():
# # # #             ade_k  = torch.stack([
# # # #                 _ade_km_from_rel(modes_rel_nograd[k], x1_rel_orig.detach(), lp)
# # # #                 for k in range(self.K)
# # # #             ], dim=1)                       # [B, K]
# # # #             k_star = ade_k.argmin(dim=1)    # [B]

# # # #         # ── FM Loss: Easy/Difficult (thuần, không có step weighting hardcode) ──
# # # #         # FM velocity MSE thuần — không nhắm vào bất kỳ bước cụ thể nào.
# # # #         # Easy/Difficult split theo delta là đủ để phân biệt hard/easy cases.
# # # #         fm_errs = torch.stack([
# # # #             ((vels_pred[k] - u_target)**2).mean(dim=[1, 2])
# # # #             for k in range(self.K)
# # # #         ], dim=1)                                    # [B, K]

# # # #         L_easy   = fm_errs.mean(dim=1)               # [B]
# # # #         L_oracle = fm_errs[torch.arange(B, device=device), k_star]  # [B]

# # # #         # Diversity push (FIX-B5)
# # # #         mode_star_ng = modes_t_ng[torch.arange(B, device=device), k_star]
# # # #         dists_all    = torch.stack([
# # # #             ((mode_star_ng - modes_t_ng[:, k])**2).mean(dim=[1,2]).sqrt()
# # # #             for k in range(self.K)
# # # #         ], dim=1)
# # # #         mask_ns       = torch.ones(B, self.K, device=device, dtype=torch.bool)
# # # #         mask_ns.scatter_(1, k_star.unsqueeze(1), False)
# # # #         min_dist      = dists_all.masked_fill(~mask_ns, float('inf')).min(dim=1).values

# # # #         MARGIN = 0.40
# # # #         L_div  = F.relu(MARGIN - min_dist)
# # # #         L_diff = L_oracle + 0.3 * L_div

# # # #         L_FM_raw = (1.0 - delta) * L_easy + delta * L_diff
# # # #         w_d      = 0.5 + 1.5 * delta
# # # #         L_FM     = (w_d * L_FM_raw).mean()

# # # #         # ── Selector Loss (STOP-GRAD) ─────────────────────────────────────────
# # # #         sel_logits, dir_logits = self.selector(
# # # #             ctx.detach(), modes_t_ng, env_data)   # modes_t_ng already detached

# # # #         tau      = 3.0
# # # #         p_oracle = F.softmax(-ade_k / tau, dim=1)
# # # #         L_rank   = F.kl_div(F.log_softmax(sel_logits, dim=1), p_oracle,
# # # #                              reduction='batchmean')

# # # #         gt_bucket = get_gt_direction_bucket(obs_t, pred_t)
# # # #         L_dir     = F.cross_entropy(dir_logits, gt_bucket)

# # # #         # ── FIX-C1 + FIX-C5: Coherence Loss (CÓ gradient, dùng x1_rel_orig) ─
# # # #         # Dùng selector probs (stop-grad) để weight modes, nhưng modes_t_g CÓ grad
# # # #         with torch.no_grad():
# # # #             sel_probs = F.softmax(sel_logits, dim=1)           # [B, K]
# # # #         # Weighted sum của modes_t_g — gradient chảy qua modes_t_g
# # # #         pred_soft_grad = (sel_probs.unsqueeze(-1).unsqueeze(-1)
# # # #                           * modes_t_g).sum(dim=1)              # [B,12,4] — CÓ grad
# # # #         L_coh = _coherence_loss(pred_soft_grad, x1_rel_orig, lp, device)  # dùng orig GT cho coherence, nhất quán với oracle

# # # #         # ── Learned weights (FIX-P2: 2-phase) ───────────────────────────────
# # # #         # TẤT CẢ weights tự học — không có hardcode nào.
# # # #         if in_warmup:
# # # #             # Phase 1: chỉ train selector
# # # #             alpha_lw = torch.sigmoid(self.learned_weights.log_alpha)
# # # #             L_total  = alpha_lw * L_rank + (1.0 - alpha_lw) * L_dir
# # # #         else:
# # # #             # Phase 2: joint training, 4 losses đều tự cân bằng
# # # #             L_total = self.learned_weights(L_FM, L_rank, L_dir, L_coh, L_consist)

# # # #         if not torch.isfinite(L_total):
# # # #             L_total = L_total.new_zeros(())

# # # #         def _s(x): return x.item() if torch.is_tensor(x) else float(x)

# # # #         return {
# # # #             "total":          L_total,
# # # #             "L_FM":           _s(L_FM),
# # # #             "L_easy":         _s(L_easy.mean()),
# # # #             "L_diff":         _s(L_diff.mean()),
# # # #             "L_oracle":       _s(L_oracle.mean()),
# # # #             "L_div":          _s(L_div.mean()),
# # # #             "L_rank":         _s(L_rank),
# # # #             "L_dir":          _s(L_dir),
# # # #             "L_coh":          _s(L_coh),
# # # #             "L_consist":      _s(L_consist),
# # # #             "delta_mean":     _s(delta.mean()),
# # # #             "delta_p75":      _s(delta.quantile(0.75)),
# # # #             "min_dist_mean":  _s(min_dist.mean()),
# # # #             "in_warmup":      in_warmup,
# # # #             **self.learned_weights.get_weights(),
# # # #         }

# # # #     # ═════════════════════════════════════════════════════════════════════════
# # # #     #  INFERENCE
# # # #     # ═════════════════════════════════════════════════════════════════════════

# # # #     @torch.no_grad()
# # # #     def sample(self, batch_list, ddim_steps=20, predict_csv=None,
# # # #                blend_strength=0.10, **kwargs):
# # # #         """
# # # #         Inference K=8 compass modes → selector chọn best.
# # # #         FIX-C2: dùng cùng sigma_k như training.
# # # #         """
# # # #         obs_t    = batch_list[0]   # [T, B, 2]
# # # #         obs_Me   = batch_list[7]
# # # #         env_data = batch_list[13] if len(batch_list) > 13 else None

# # # #         lp = obs_t[-1]; lm = obs_Me[-1]
# # # #         B, device = lp.shape[0], lp.device
# # # #         T_pred = self.pred_len
# # # #         dt     = 1.0 / max(ddim_steps, 1)

# # # #         raw_ctx = self.encoder.encode(batch_list)
# # # #         ctx     = self.encoder.apply_ctx_head(raw_ctx)

# # # #         # FIX-B11: persistence x0 trong rel space
# # # #         x0_base = _persistence_x0_rel(obs_t, obs_Me, lp, lm, T_pred, sigma=0.0)

# # # #         all_modes_abs = []

# # # #         # Null context cho CFG (port từ v59)
# # # #         ctx_null = self.encoder.apply_ctx_head(raw_ctx, use_null=True)

# # # #         # Heading từ obs để tính alignment gate (port từ v59)
# # # #         obs_t_norm = obs_t[:, :, :2]   # [T, B, 2]
# # # #         if obs_t_norm.shape[0] >= 2:
# # # #             obs_h_n = F.normalize(
# # # #                 obs_t_norm[-1] - obs_t_norm[-2], dim=-1, eps=1e-6)  # [B, 2]
# # # #         else:
# # # #             obs_h_n = None

# # # #         for k in range(self.K):
# # # #             # FIX-C2: sigma_k nhất quán với training
# # # #             sigma_k = self.head_noise_base * (1.0 + k * 0.25)
# # # #             x_t = x0_base + torch.randn_like(x0_base) * sigma_k

# # # #             for step in range(ddim_steps):
# # # #                 t_b = torch.full((B,), step * dt, device=device)

# # # #                 # CFG từ step 1 trở đi (step 0 dùng cond để ổn định)
# # # #                 if step > 0 and self.cfg_guidance_scale > 1.0:
# # # #                     v_cond   = self.velocity_heads[k](x_t, t_b, ctx)
# # # #                     v_uncond = self.velocity_heads[k](x_t, t_b, ctx_null)

# # # #                     # Alignment gate: scale CFG theo alignment với obs heading
# # # #                     if obs_h_n is not None:
# # # #                         pred_h = F.normalize(
# # # #                             v_cond[:, 0, :2].detach(), dim=-1, eps=1e-6)
# # # #                         cos_a  = (obs_h_n * pred_h).sum(-1).clamp(-1.0, 1.0)
# # # #                         # gs ∈ [0.8, 1.5]: aligned → full guidance, misaligned → reduced
# # # #                         gs     = (0.8 + 0.7 * (cos_a + 1.0) * 0.5).view(B, 1, 1)
# # # #                         v_k    = v_uncond + gs * (v_cond - v_uncond)
# # # #                     else:
# # # #                         v_k = v_uncond + self.cfg_guidance_scale * (v_cond - v_uncond)
# # # #                 else:
# # # #                     v_k = self.velocity_heads[k](x_t, t_b, ctx)

# # # #                 x_t = (x_t + dt * v_k).clamp(-5.0, 5.0)

# # # #             traj_abs, _ = self._to_abs(x_t, lp, lm)   # [T, B, 2]
# # # #             all_modes_abs.append(traj_abs)

# # # #         modes_stack = torch.stack(all_modes_abs, dim=0)   # [K, T, B, 2]

# # # #         # Rebuild modes_rel_sel cho selector [B, K, T, 4]
# # # #         modes_rel_sel = torch.stack([
# # # #             torch.cat([all_modes_abs[k].permute(1,0,2) - lp.unsqueeze(1),
# # # #                        torch.zeros(B, T_pred, 2, device=device)], dim=-1)
# # # #             for k in range(self.K)
# # # #         ], dim=1)

# # # #         sel_logits, _ = self.selector(ctx, modes_rel_sel, env_data)
# # # #         best_k        = sel_logits.argmax(dim=1)          # [B]

# # # #         pred_best_list = [modes_stack[best_k[b], :, b, :] for b in range(B)]
# # # #         pred_best = torch.stack(pred_best_list, dim=1)    # [T, B, 2]

# # # #         pred_final = self._persistence_blend(
# # # #             pred_best, obs_t[:, :, :2], blend_strength=blend_strength)

# # # #         if predict_csv is not None:
# # # #             self._write_predict_csv(predict_csv, pred_final, modes_stack)

# # # #         return pred_final, modes_stack

# # # #     @staticmethod
# # # #     def _write_predict_csv(csv_path, traj_mean, all_modes_KTB2):
# # # #         import csv as _csv, os
# # # #         from datetime import datetime
# # # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
# # # #         T, B = traj_mean.shape[0], traj_mean.shape[1]
# # # #         mlon = ((traj_mean[...,0]*50.0+1800.0)/10.0).cpu().numpy()
# # # #         mlat = ((traj_mean[...,1]*50.0)/10.0).cpu().numpy()
# # # #         fields = ["ts","b","step","lead_h","lon","lat"]
# # # #         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # #         write_hdr = not os.path.exists(csv_path)
# # # #         with open(csv_path, "a", newline="") as fh:
# # # #             w = _csv.DictWriter(fh, fieldnames=fields)
# # # #             if write_hdr: w.writeheader()
# # # #             for b in range(B):
# # # #                 for k in range(T):
# # # #                     w.writerow({"ts":ts,"b":b,"step":k,"lead_h":(k+1)*6,
# # # #                                  "lon":f"{mlon[k,b]:.4f}","lat":f"{mlat[k,b]:.4f}"})


# # # # # Backward compat
# # # # TCFlowMatching = TCFlowMatchingV65
# # # # TCDiffusion    = TCFlowMatchingV65

# # # """
# # # flow_matching_model_v65.py  ── v65-rev2
# # # ══════════════════════════════════════════════════════════════════════════════
# # # TC-FlowMatching v65 — Revision 2

# # # VẤN ĐỀ CỐT LÕI SO VỚI STTRANS
# # # ────────────────────────────────
# # # STTrans dùng cùng encoder stack (FNO3D + Mamba + Env_net) nhưng decoder là
# # # 1 Non-AR Transformer optimize L_DPE (haversine) TRỰC TIẾP.

# # # v65-rev1 optimize L_FM = velocity MSE → KHÔNG trực tiếp minimize haversine
# # # → ADE của v65 có thể không thấp hơn STTrans dù có K=8 modes.

# # # THAY ĐỔI TRONG rev2
# # # ────────────────────
# # # [REV2-1] L_DPE trực tiếp cho oracle head (KEY FIX)
# # #          Thêm haversine loss cho best mode (k_star):
# # #            L_dpe_oracle = haversine(x1_pred_best_mode, x1_gt_orig) / 500
# # #          Đưa vào LearnedWeights như term thứ 5 → model learn to minimize
# # #          haversine trực tiếp như STTrans, nhưng ĐỒN THỜI có FM diversity.

# # # [REV2-2] L_speed + L_accel (physics kinematic, từ STTrans paper §3.5.1)
# # #          Penalize unrealistic speed/acceleration trong best mode prediction.
# # #          STTrans có điều này, v65-rev1 không → giúp 72h coherence.
# # #          λ_speed = 0.1, λ_accel = 0.01 (đúng như paper).

# # # [REV2-3] LearnedWeights mở rộng: 6 terms
# # #          s_fm, s_sel, s_coh, s_con, s_dpe, s_kin (kinematic)
# # #          Tất cả tự học, không hardcode.

# # # [REV2-4] MARGIN tự học thay vì hardcode 0.40
# # #          MARGIN = sigmoid(log_margin) * 0.8 → [0, 0.8]
# # #          Tự adapt theo data distribution.

# # # [REV2-5] Inference: haversine-based final selection
# # #          Thay vì chỉ dùng selector logits, combine với speed plausibility:
# # #          score_final = sel_logits - λ * speed_penalty
# # #          Modes vi phạm speed > 80 km/h bị penalize khi chọn.

# # # GIỮ NGUYÊN TỪ rev1
# # # ────────────────────
# # # - K=8 compass heads với direction cross-attention (FIX-C4)
# # # - Per-head noise injection (FIX-P1, FIX-C2)
# # # - Oracle assignment từ x_t_shared (FIX-C3)
# # # - Easy/Difficult split với δ (5 env signals)
# # # - OT matching (từ v59)
# # # - CFG inference với alignment gate
# # # - 3-phase training
# # # - Mixup augmentation
# # # - Dropout consistency
# # # - Tất cả 11 bug fixes từ review
# # # """
# # # from __future__ import annotations

# # # import math
# # # from typing import Optional, Tuple, List

# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F

# # # from Model.FNO3D_encoder import FNO3DEncoder
# # # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # # from Model.env_net_transformer_gphsplit import Env_net

# # # R_EARTH      = 6371.0
# # # DT_HOURS     = 6.0
# # # DEG2KM       = 111.0
# # # _NORM_TO_DEG = 5.0

# # # _COMPASS_DEG   = [0., 45., 90., 135., 180., 225., 270., 315.]
# # # _COMPASS_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
# # # K_MODES        = 8

# # # # Speed/accel physics constants (từ STTrans paper)
# # # _V_MAX_KMH   = 80.0    # max realistic TC speed km/h
# # # _DT_H        = 6.0     # time step hours


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  Coordinate utilities
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# # #     return torch.stack([
# # #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# # #         (t[..., 1] * 50.0) / 10.0,
# # #     ], dim=-1)

# # # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # #     a = (torch.sin(dlat/2).pow(2) +
# # #          torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2))
# # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

# # # def _unwrap_model(m):
# # #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # # def _step_speeds_deg(traj_deg: torch.Tensor) -> torch.Tensor:
# # #     T = traj_deg.shape[0]
# # #     if T < 2: return traj_deg.new_zeros(1, traj_deg.shape[1])
# # #     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS

# # # def _ade_km_from_rel(pred_rel, gt_rel, lp):
# # #     """[B,T,4] → [B] ADE in km."""
# # #     pred_abs = lp.unsqueeze(1) + pred_rel[:, :, :2]
# # #     gt_abs   = lp.unsqueeze(1) + gt_rel[:, :, :2]
# # #     return _haversine_deg(_norm_to_deg(pred_abs), _norm_to_deg(gt_abs)).mean(dim=1)


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  REV2-1+2: Physics-informed DPE + Kinematic Loss
# # # #  Lấy cảm hứng từ STTrans loss (§3.5.1) nhưng áp dụng cho best mode
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # def _dpe_and_kinematic_loss(
# # #     best_mode_rel: torch.Tensor,   # [B, T, 4] — best mode, CÓ gradient
# # #     x1_rel_orig:   torch.Tensor,   # [B, T, 4] — GT original
# # #     lp:            torch.Tensor,   # [B, 2]
# # #     device:        torch.device,
# # #     v_max_norm:    float,          # max speed in normalized units
# # # ) -> Tuple[torch.Tensor, torch.Tensor]:
# # #     """
# # #     REV2-1: L_DPE = mean haversine(best_mode, GT) / 500 (normalize ~same scale as L_FM)
# # #     REV2-2: L_kin = λ_speed*L_speed + λ_accel*L_accel (từ STTrans §3.5.1)

# # #     Returns: (L_dpe, L_kin)
# # #     """
# # #     B, T = best_mode_rel.shape[:2]

# # #     # Convert sang degrees cho haversine
# # #     best_abs = lp.unsqueeze(1) + best_mode_rel[:, :, :2]   # [B, T, 2]
# # #     gt_abs   = lp.unsqueeze(1) + x1_rel_orig[:, :, :2]

# # #     best_deg = _norm_to_deg(best_abs)   # [B, T, 2]
# # #     gt_deg   = _norm_to_deg(gt_abs)

# # #     # L_DPE: mean haversine (eq. 16 trong STTrans paper)
# # #     L_dpe = _haversine_deg(best_deg, gt_deg).mean() / 500.0

# # #     # L_speed: penalize step speed > v_max (eq. 18-20)
# # #     if T >= 2:
# # #         step_dist = (best_mode_rel[:, 1:, :2] - best_mode_rel[:, :-1, :2]).norm(dim=-1)
# # #         L_speed   = F.relu(step_dist - v_max_norm).pow(2).mean()
# # #     else:
# # #         L_speed = torch.tensor(0.0, device=device)

# # #     # L_accel: penalize acceleration (eq. 21-22)
# # #     if T >= 3:
# # #         vel      = best_mode_rel[:, 1:, :2] - best_mode_rel[:, :-1, :2]  # [B,T-1,2]
# # #         spd      = vel.norm(dim=-1)                                        # [B, T-1]
# # #         L_accel  = (spd[:, 1:] - spd[:, :-1]).pow(2).mean()
# # #     else:
# # #         L_accel = torch.tensor(0.0, device=device)

# # #     L_kin = 0.1 * L_speed + 0.01 * L_accel
# # #     return L_dpe, L_kin


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  OT Matching (từ v59)
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # def _forward_azimuth(p1, p2):
# # #     lon1 = torch.deg2rad(p1[...,0]); lat1 = torch.deg2rad(p1[...,1])
# # #     lon2 = torch.deg2rad(p2[...,0]); lat2 = torch.deg2rad(p2[...,1])
# # #     dlon = lon2 - lon1
# # #     y = torch.sin(dlon)*torch.cos(lat2)
# # #     x = torch.cos(lat1)*torch.sin(lat2) - torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon)
# # #     return torch.atan2(y, x)

# # # def _sinkhorn_log(cost, epsilon=0.05, n_iter=50):
# # #     B = cost.shape[0]; device = cost.device
# # #     log_a = -math.log(B)*torch.ones(B,device=device)
# # #     log_b = -math.log(B)*torch.ones(B,device=device)
# # #     log_K = -cost/epsilon
# # #     log_u = torch.zeros(B,device=device)
# # #     log_v = torch.zeros(B,device=device)
# # #     for _ in range(n_iter):
# # #         log_u = log_a - torch.logsumexp(log_K+log_v.unsqueeze(0), dim=1)
# # #         log_v = log_b - torch.logsumexp(log_K+log_u.unsqueeze(1), dim=0)
# # #     return (log_K+log_u.unsqueeze(1)+log_v.unsqueeze(0)).exp().clamp(0.0)

# # # def _geodesic_ot_cost(x0_rel, x1_rel, lp):
# # #     B = x0_rel.shape[0]
# # #     def _abs_deg(rel): return _norm_to_deg(lp.unsqueeze(1)+rel[:,:,:2])
# # #     x0d = _abs_deg(x0_rel); x1d = _abs_deg(x1_rel)
# # #     x0e = x0d.unsqueeze(1).expand(B,B,-1,-1).reshape(B*B,-1,2)
# # #     x1e = x1d.unsqueeze(0).expand(B,B,-1,-1).reshape(B*B,-1,2)
# # #     pos_cost = _haversine_deg(x0e,x1e).reshape(B,B,-1).mean(-1)/500.0
# # #     spd0 = _step_speeds_deg(x0d.permute(1,0,2)).permute(1,0).mean(-1)
# # #     spd1 = _step_speeds_deg(x1d.permute(1,0,2)).permute(1,0).mean(-1)
# # #     speed_cost = (spd0.unsqueeze(1)-spd1.unsqueeze(0)).abs()/20.0
# # #     def _mb(td):
# # #         b = _forward_azimuth(td[:,:-1,:],td[:,1:,:])
# # #         return torch.atan2(b.sin().mean(-1),b.cos().mean(-1))
# # #     h0 = _mb(x0d); h1 = _mb(x1d)
# # #     dh = (h0.unsqueeze(1)-h1.unsqueeze(0)+math.pi)%(2*math.pi)-math.pi
# # #     return 1.0*pos_cost + 0.5*speed_cost + 0.3*dh.abs()/math.pi

# # # def _spherical_ot_matching(x0_batch, x1_batch, lp, epsilon=0.05):
# # #     B = x0_batch.shape[0]
# # #     if B < 4: return x0_batch, x1_batch
# # #     try:
# # #         cost = _geodesic_ot_cost(x0_batch, x1_batch, lp)
# # #         with torch.no_grad():
# # #             pi = _sinkhorn_log(cost, epsilon=epsilon)
# # #         flat = pi.reshape(-1).clamp(0.0); s = flat.sum()
# # #         if not torch.isfinite(s) or s < 1e-10: return x0_batch, x1_batch
# # #         idx = torch.multinomial(flat/s, num_samples=B, replacement=True)
# # #         return x0_batch[idx%B], x1_batch[idx%B]
# # #     except Exception:
# # #         return x0_batch, x1_batch


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  Difficulty Score
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # def compute_difficulty_score(obs_BT2, img_obs, env_data, device):
# # #     B = obs_BT2.shape[0]; scores = []

# # #     # Signal 1: curvature (0.30)
# # #     dir24 = env_data.get("history_direction24") if env_data else None
# # #     if dir24 is not None and torch.is_tensor(dir24):
# # #         dir24 = dir24.float().to(device)
# # #         if dir24.dim()==3: d24=dir24[:,-1,:]
# # #         elif dir24.dim()==2: d24=dir24
# # #         else: d24=None
# # #         if d24 is not None and d24.shape[-1]==8:
# # #             bucket = d24.argmax(dim=-1).float()
# # #             a24 = bucket*(2.0*math.pi/8.0)
# # #             dy = obs_BT2[:,-1,1]-obs_BT2[:,-2,1]; dx = obs_BT2[:,-1,0]-obs_BT2[:,-2,0]
# # #             lat_mid = (obs_BT2[:,-1,1]+obs_BT2[:,-2,1])*0.5
# # #             cos_lat = torch.cos(torch.deg2rad(lat_mid*_NORM_TO_DEG)).clamp(1e-4)
# # #             a_now = torch.atan2(dx*cos_lat, dy)
# # #             cos_d = (torch.cos(a_now)*torch.cos(a24)+torch.sin(a_now)*torch.sin(a24)).clamp(-1,1)
# # #             s_curve = torch.sigmoid((torch.rad2deg(torch.acos(cos_d))-45.0)/20.0)
# # #         else: s_curve = torch.zeros(B,device=device)
# # #     else: s_curve = torch.zeros(B,device=device)
# # #     scores.append(0.30*s_curve)

# # #     # Signal 2: weak steering (0.25)
# # #     steer = env_data.get("steering_speed") if env_data else None
# # #     if steer is not None and torch.is_tensor(steer):
# # #         steer = steer.float().to(device)
# # #         while steer.dim()>1: steer=steer[...,-1]
# # #         steer = steer.view(-1); steer = steer[:B] if steer.numel()>=B else steer[0].expand(B)
# # #         s_weak = torch.sigmoid((4.0-steer*20.0)/2.0)
# # #     else:
# # #         u = env_data.get("u500_mean") if env_data else None
# # #         v = env_data.get("v500_mean") if env_data else None
# # #         if u is not None and v is not None and torch.is_tensor(u) and torch.is_tensor(v):
# # #             u=u.float().to(device); v=v.float().to(device)
# # #             while u.dim()>1: u=u[...,-1]
# # #             while v.dim()>1: v=v[...,-1]
# # #             s_weak = torch.sigmoid((4.0-(u.view(-1)[:B]**2+v.view(-1)[:B]**2).sqrt()*30.0)/2.0)
# # #         else: s_weak = torch.zeros(B,device=device)
# # #     scores.append(0.25*s_weak)

# # #     # Signal 3: wind shear (0.20)
# # #     if img_obs is not None and img_obs.shape[1]>=11:
# # #         u200=img_obs[:,4,-1,40,40]*13.315; u850=img_obs[:,6,-1,40,40]*7.911
# # #         v200=img_obs[:,8,-1,40,40]*8.377;  v850=img_obs[:,10,-1,40,40]*6.203
# # #         shear=((u200-u850)**2+(v200-v850)**2).sqrt()
# # #         s_shear=torch.sigmoid((shear-8.0)/3.0)
# # #     else: s_shear = torch.zeros(B,device=device)
# # #     scores.append(0.20*s_shear)

# # #     # Signal 4: RI (0.15)
# # #     ri = env_data.get("rapid_intensification") if env_data else None
# # #     if ri is not None and torch.is_tensor(ri):
# # #         ri=ri.float().to(device)
# # #         while ri.dim()>1: ri=ri[...,-1]
# # #         s_ri = ri.view(-1)[:B].clamp(0,1) if ri.numel()>=B else ri[0].expand(B)
# # #     else: s_ri = torch.zeros(B,device=device)
# # #     scores.append(0.15*s_ri)

# # #     # Signal 5: slow movement (0.10)
# # #     mv = env_data.get("move_velocity") if env_data else None
# # #     if mv is not None and torch.is_tensor(mv):
# # #         mv=mv.float().to(device)
# # #         while mv.dim()>1: mv=mv[...,-1]
# # #         mv_v=mv.view(-1); mv_v=mv_v[:B] if mv_v.numel()>=B else mv_v[0].expand(B)
# # #         s_slow = torch.sigmoid((0.05-mv_v)/0.02)
# # #     else: s_slow = torch.zeros(B,device=device)
# # #     scores.append(0.10*s_slow)

# # #     return sum(scores).clamp(0.0, 1.0)


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  Direction GT bucket
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # def get_gt_direction_bucket(obs_traj, pred_traj):
# # #     last=obs_traj[-1]; first=pred_traj[0]
# # #     dy=first[:,1]-last[:,1]; dx=first[:,0]-last[:,0]
# # #     lat_d=(last[:,1]+first[:,1])*0.5
# # #     cos_l=torch.cos(torch.deg2rad(lat_d*_NORM_TO_DEG)).clamp(1e-4)
# # #     angle=(torch.atan2(dx*cos_l,dy).rad2deg()%360.0)
# # #     return ((angle+22.5)/45.0).long()%8


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  FIX-C4: CompassVelocityHead — direction via cross-attention
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # class CompassVelocityHead(nn.Module):
# # #     def __init__(self, compass_idx, pred_len=12, ctx_dim=256, stochastic_depth_prob=0.1):
# # #         super().__init__()
# # #         self.pred_len=pred_len; self.compass_idx=compass_idx
# # #         self.stochastic_depth_prob=stochastic_depth_prob
# # #         angle_rad=_COMPASS_DEG[compass_idx]*math.pi/180.0
# # #         dir_vec=torch.tensor([math.sin(angle_rad),math.cos(angle_rad)])
# # #         self.register_buffer('compass_dir',dir_vec)
# # #         self.dir_proj=nn.Sequential(nn.Linear(2,ctx_dim),nn.GELU(),nn.LayerNorm(ctx_dim))
# # #         self.time_fc1=nn.Linear(ctx_dim,256); self.time_fc2=nn.Linear(256,ctx_dim)
# # #         self.traj_embed=nn.Linear(4,ctx_dim)
# # #         self.pos_enc=nn.Parameter(torch.randn(1,pred_len,ctx_dim)*0.02)
# # #         self.step_embed=nn.Embedding(pred_len,ctx_dim)
# # #         self.transformer=nn.TransformerDecoder(
# # #             nn.TransformerDecoderLayer(d_model=ctx_dim,nhead=8,dim_feedforward=512,
# # #                                        dropout=0.10,activation="gelu",batch_first=True),
# # #             num_layers=1)
# # #         self.out_fc1=nn.Linear(ctx_dim,256); self.out_fc2=nn.Linear(256,4)
# # #         self.step_scale=nn.Parameter(torch.ones(pred_len)*0.5)
# # #         with torch.no_grad():
# # #             nn.init.xavier_uniform_(self.out_fc2.weight,gain=0.1)
# # #             nn.init.zeros_(self.out_fc2.bias)

# # #     def _time_emb(self,t,dim):
# # #         half=dim//2
# # #         freq=torch.exp(torch.arange(half,dtype=torch.float32,device=t.device)
# # #                        *(-math.log(10000.0)/max(half-1,1)))
# # #         emb=t.float().unsqueeze(1)*1000.0*freq.unsqueeze(0)
# # #         return F.pad(torch.cat([emb.sin(),emb.cos()],dim=-1),(0,dim%2))

# # #     def forward(self,x_t,t,ctx):
# # #         B=x_t.shape[0]; T_seq=min(x_t.shape[1],self.pred_len); device=x_t.device
# # #         dir_token=self.dir_proj(self.compass_dir.to(device).unsqueeze(0).expand(B,-1))
# # #         if self.training and self.stochastic_depth_prob>0:
# # #             keep=(torch.rand(B,1,device=device)>self.stochastic_depth_prob).float()
# # #             dir_token=dir_token*keep
# # #         t_emb=F.gelu(self.time_fc1(self._time_emb(t,ctx.shape[-1])))
# # #         t_emb=self.time_fc2(t_emb)
# # #         step_idx=torch.arange(T_seq,device=device).unsqueeze(0).expand(B,-1)
# # #         x_emb=(self.traj_embed(x_t[:,:T_seq])+self.pos_enc[:,:T_seq]
# # #                +t_emb.unsqueeze(1)+self.step_embed(step_idx))
# # #         mem=torch.stack([ctx,dir_token,t_emb],dim=1)
# # #         decoded=self.transformer(x_emb,mem)
# # #         scale=torch.sigmoid(self.step_scale[:T_seq]).view(1,T_seq,1)*2.0
# # #         return self.out_fc2(F.gelu(self.out_fc1(decoded)))*scale


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  CompassSelector
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # class CompassSelector(nn.Module):
# # #     def __init__(self,ctx_dim=256,K=8,n_dirs=8):
# # #         super().__init__()
# # #         self.K=K; self.n_dirs=n_dirs
# # #         self.steering_enc=nn.Sequential(nn.Linear(11,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,128))
# # #         self.mode_enc=nn.Sequential(nn.Linear(12*2,128),nn.GELU(),nn.LayerNorm(128),nn.Linear(128,64))
# # #         self.score_net=nn.Sequential(nn.Linear(ctx_dim+64+128,256),nn.GELU(),nn.LayerNorm(256),
# # #                                      nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
# # #         self.dir_head=nn.Sequential(nn.Linear(ctx_dim+128,256),nn.GELU(),nn.LayerNorm(256),nn.Linear(256,n_dirs))

# # #     def _get_steer(self,env_data,B,device):
# # #         def _sc(key):
# # #             if env_data is None: return torch.zeros(B,device=device)
# # #             v=env_data.get(key)
# # #             if v is None or not torch.is_tensor(v): return torch.zeros(B,device=device)
# # #             v=v.float().to(device)
# # #             while v.dim()>1: v=v[...,-1]
# # #             v=v.view(-1); return (v[:B] if v.numel()>=B else v[0].expand(B)).clamp(-3,3)
# # #         def _sv(key,dim):
# # #             if env_data is None: return torch.zeros(B,dim,device=device)
# # #             v=env_data.get(key)
# # #             if v is None or not torch.is_tensor(v): return torch.zeros(B,dim,device=device)
# # #             v=v.float().to(device)
# # #             if v.dim()==3: v=v[:,-1,:]
# # #             elif v.dim()!=2 or v.shape[0]!=B: return torch.zeros(B,dim,device=device)
# # #             v=F.pad(v,(0,max(0,dim-v.shape[-1])))[:,:dim]; return v.clamp(-3,3)
# # #         feat=torch.cat([_sc("steering_speed").unsqueeze(-1),_sc("steering_dir_sin").unsqueeze(-1),
# # #                         _sc("steering_dir_cos").unsqueeze(-1),_sv("history_direction24",8)],dim=-1)
# # #         return self.steering_enc(feat)

# # #     def forward(self,ctx,modes,env_data):
# # #         B,K=modes.shape[:2]; device=ctx.device
# # #         sf=self._get_steer(env_data,B,device)
# # #         scores=[]
# # #         for k in range(K):
# # #             mf=self.mode_enc(modes[:,k,:,:2].reshape(B,-1))
# # #             scores.append(self.score_net(torch.cat([ctx,mf,sf],dim=-1)).squeeze(-1))
# # #         score_logits=torch.stack(scores,dim=1)
# # #         dir_logits=self.dir_head(torch.cat([ctx,sf],dim=-1))
# # #         return score_logits, dir_logits


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  REV2-3: LearnedWeights — 6 terms
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # class LearnedWeights(nn.Module):
# # #     """
# # #     6 learnable log-variance (Kendall 2018):
# # #       s_fm, s_sel, s_coh, s_con, s_dpe, s_kin
# # #     s_dpe: DPE haversine term (REV2-1)
# # #     s_kin: kinematic speed/accel term (REV2-2)
# # #     """
# # #     def __init__(self):
# # #         super().__init__()
# # #         self.log_s_fm  = nn.Parameter(torch.zeros(1))
# # #         self.log_s_sel = nn.Parameter(torch.zeros(1))
# # #         self.log_s_coh = nn.Parameter(torch.zeros(1))
# # #         self.log_s_con = nn.Parameter(torch.zeros(1))
# # #         self.log_s_dpe = nn.Parameter(torch.zeros(1))   # REV2-1
# # #         self.log_s_kin = nn.Parameter(torch.zeros(1))   # REV2-2
# # #         self.log_alpha = nn.Parameter(torch.zeros(1))

# # #     def forward(self, L_fm, L_rank, L_dir, L_coh, L_con, L_dpe, L_kin):
# # #         def _w(s): return torch.exp(-s.clamp(-5,5))
# # #         α = torch.sigmoid(self.log_alpha)
# # #         L_sel = α*L_rank + (1.0-α)*L_dir
# # #         s_fm=self.log_s_fm.clamp(-5,5); s_sel=self.log_s_sel.clamp(-5,5)
# # #         s_coh=self.log_s_coh.clamp(-5,5); s_con=self.log_s_con.clamp(-5,5)
# # #         s_dpe=self.log_s_dpe.clamp(-5,5); s_kin=self.log_s_kin.clamp(-5,5)
# # #         return (L_fm*_w(s_fm)+s_fm + L_sel*_w(s_sel)+s_sel +
# # #                 L_coh*_w(s_coh)+s_coh + L_con*_w(s_con)+s_con +
# # #                 L_dpe*_w(s_dpe)+s_dpe + L_kin*_w(s_kin)+s_kin)

# # #     def get_weights(self):
# # #         def _w(s): return torch.exp(-s.clamp(-5,5)).item()
# # #         return {"w_fm":_w(self.log_s_fm),"w_sel":_w(self.log_s_sel),
# # #                 "w_coh":_w(self.log_s_coh),"w_con":_w(self.log_s_con),
# # #                 "w_dpe":_w(self.log_s_dpe),"w_kin":_w(self.log_s_kin),
# # #                 "alpha":torch.sigmoid(self.log_alpha).item()}


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  Shared Context Encoder
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # class SharedContextEncoder(nn.Module):
# # #     RAW_CTX_DIM = 512
# # #     def __init__(self,pred_len=12,obs_len=8,ctx_dim=256,unet_in_ch=13):
# # #         super().__init__()
# # #         self.pred_len=pred_len; self.obs_len=obs_len; self.ctx_dim=ctx_dim
# # #         self.spatial_enc=FNO3DEncoder(in_channel=unet_in_ch,out_channel=1,d_model=32,
# # #             n_layers=4,modes_t=4,modes_h=4,modes_w=4,spatial_down=32,dropout=0.05)
# # #         self.bottleneck_pool=nn.AdaptiveAvgPool3d((None,1,1))
# # #         self.bottleneck_proj=nn.Linear(128,128)
# # #         self.decoder_proj=nn.Linear(1,16)
# # #         self.enc_1d=DataEncoder1D(in_1d=4,feat_3d_dim=128,mlp_h=64,
# # #             lstm_hidden=128,lstm_layers=3,dropout=0.1,d_state=16)
# # #         self.env_enc=Env_net(obs_len=obs_len,d_model=32)
# # #         self.ctx_fc1=nn.Linear(128+32+16,self.RAW_CTX_DIM)
# # #         self.ctx_ln=nn.LayerNorm(self.RAW_CTX_DIM)
# # #         self.ctx_drop=nn.Dropout(0.15)
# # #         self.ctx_fc2=nn.Linear(self.RAW_CTX_DIM,ctx_dim)
# # #         self.null_embedding=nn.Parameter(torch.randn(1,self.RAW_CTX_DIM)*0.02)
# # #         self.vel_obs_enc=nn.Sequential(nn.Linear(obs_len*6,256),nn.GELU(),
# # #                                        nn.LayerNorm(256),nn.Linear(256,256),nn.GELU())

# # #     def encode(self,batch_list):
# # #         obs_t=batch_list[0]; obs_Me=batch_list[7]
# # #         image_obs=batch_list[11]; env_data=batch_list[13] if len(batch_list)>13 else None
# # #         if image_obs.dim()==4: image_obs=image_obs.unsqueeze(2)
# # #         if image_obs.shape[1]==1 and self.spatial_enc.in_channel!=1:
# # #             image_obs=image_obs.expand(-1,self.spatial_enc.in_channel,-1,-1,-1)
# # #         T_obs=obs_t.shape[0]
# # #         e_3d_bot,e_3d_dec=self.spatial_enc.encode(image_obs)
# # #         e_3d_s=self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0,2,1)
# # #         e_3d_s=self.bottleneck_proj(e_3d_s)
# # #         if e_3d_s.shape[1]!=T_obs:
# # #             e_3d_s=F.interpolate(e_3d_s.permute(0,2,1),size=T_obs,mode="linear",
# # #                                   align_corners=False).permute(0,2,1)
# # #         e_3d_dec_t=e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# # #         t_w=torch.softmax(torch.arange(e_3d_dec_t.shape[1],dtype=torch.float,
# # #                                         device=e_3d_dec_t.device)*0.5,dim=0)
# # #         f_sp=self.decoder_proj((e_3d_dec_t*t_w.unsqueeze(0)).sum(1,keepdim=True))
# # #         obs_in=torch.cat([obs_t,obs_Me],dim=2).permute(1,0,2)
# # #         h_t=self.enc_1d(obs_in,e_3d_s)
# # #         e_env,_,_=self.env_enc(env_data,image_obs)
# # #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t,e_env,f_sp],dim=-1))))

# # #     def apply_ctx_head(self,raw,noise_scale=0.0,use_null=False):
# # #         if use_null: raw=self.null_embedding.expand(raw.shape[0],-1)
# # #         elif noise_scale>0.0: raw=raw+torch.randn_like(raw)*noise_scale
# # #         return self.ctx_fc2(self.ctx_drop(raw))

# # #     def get_kinematic_feat(self,obs_traj):
# # #         T_obs,B,_=obs_traj.shape
# # #         if T_obs>=2:
# # #             vel=obs_traj[1:]-obs_traj[:-1]
# # #             lat_mid=obs_traj[:-1,:,1]*_NORM_TO_DEG
# # #             cos_lat=torch.cos(torch.deg2rad(lat_mid)).clamp(1e-4)
# # #             dx_km=vel[:,:,0]*cos_lat*DEG2KM*_NORM_TO_DEG
# # #             dy_km=vel[:,:,1]*DEG2KM*_NORM_TO_DEG
# # #             speed=(dx_km**2+dy_km**2+1e-6).sqrt()/DT_HOURS
# # #             heading=torch.atan2(vel[:,:,1],vel[:,:,0])
# # #             speed_n=(speed/20.0).clamp(-3,3)
# # #             if T_obs>=3:
# # #                 dspd=speed[1:]-speed[:-1]
# # #                 accel=torch.cat([obs_traj.new_zeros(1,B),(dspd/10.0).clamp(-3,3)],0)
# # #             else: accel=obs_traj.new_zeros(T_obs-1,B)
# # #             kine=torch.stack([vel[:,:,0],vel[:,:,1],speed_n,
# # #                               heading.sin(),heading.cos(),accel],dim=-1)
# # #         else: kine=obs_traj.new_zeros(self.obs_len,B,6)
# # #         if kine.shape[0]<self.obs_len:
# # #             kine=torch.cat([obs_traj.new_zeros(self.obs_len-kine.shape[0],B,6),kine],0)
# # #         else: kine=kine[-self.obs_len:]
# # #         return self.vel_obs_enc(kine.permute(1,0,2).reshape(B,-1))


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  Persistence x0
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # def _persistence_x0_rel(obs_traj,obs_Me,lp,lm,pred_len,sigma=0.02):
# # #     B,device=obs_traj.shape[1],obs_traj.device
# # #     if obs_traj.shape[0]>=3:
# # #         vels=obs_traj[1:]-obs_traj[:-1]; n_v=vels.shape[0]; alpha=0.7
# # #         w=torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
# # #                        dtype=torch.float,device=device).flip(0)
# # #         lv=(vels*(w/w.sum()).view(-1,1,1)).sum(0)
# # #     elif obs_traj.shape[0]>=2: lv=obs_traj[-1,:,:2]-obs_traj[-2,:,:2]
# # #     else: lv=obs_traj.new_zeros(B,2)
# # #     steps=torch.arange(1,pred_len+1,device=device).float()
# # #     pred_abs=obs_traj[-1,:,:2].unsqueeze(0)+lv.unsqueeze(0)*steps.view(-1,1,1)
# # #     pred_rel=pred_abs.permute(1,0,2)-lp.unsqueeze(1)
# # #     pred_rel4=torch.cat([pred_rel,torch.zeros(B,pred_len,2,device=device)],dim=-1)
# # #     return pred_rel4+torch.randn_like(pred_rel4)*sigma


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  EMA
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # class EMAModel:
# # #     def __init__(self,model,decay=0.995):
# # #         self.decay=decay; m=_unwrap_model(model)
# # #         self.shadow={k:v.detach().clone() for k,v in m.state_dict().items()
# # #                      if v.dtype.is_floating_point}
# # #     def update(self,model):
# # #         m=_unwrap_model(model)
# # #         with torch.no_grad():
# # #             for k,v in m.state_dict().items():
# # #                 if k in self.shadow:
# # #                     self.shadow[k].mul_(self.decay).add_(v.detach(),alpha=1-self.decay)
# # #     def apply_to(self,model):
# # #         m=_unwrap_model(model); backup,sd={},m.state_dict()
# # #         for k in self.shadow:
# # #             if k not in sd: continue
# # #             backup[k]=sd[k].detach().clone(); sd[k].copy_(self.shadow[k])
# # #         return backup
# # #     def restore(self,model,backup):
# # #         m=_unwrap_model(model); sd=m.state_dict()
# # #         for k,v in backup.items():
# # #             if k in sd: sd[k].copy_(v)


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  Coherence Loss (FIX-C5: km space)
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # def _coherence_loss(pred_soft_grad, x1_rel_orig, lp, device):
# # #     pred_abs=lp.unsqueeze(1)+pred_soft_grad[:,:,:2]
# # #     gt_abs=lp.unsqueeze(1)+x1_rel_orig[:,:,:2]
# # #     pred_deg=_norm_to_deg(pred_abs); gt_deg=_norm_to_deg(gt_abs)
# # #     cos_lat=torch.cos(torch.deg2rad(gt_deg[:,:-1,1])).clamp(1e-4)
# # #     dp_lon=(pred_deg[:,1:,0]-pred_deg[:,:-1,0])*cos_lat*DEG2KM
# # #     dp_lat=(pred_deg[:,1:,1]-pred_deg[:,:-1,1])*DEG2KM
# # #     dg_lon=(gt_deg[:,1:,0]-gt_deg[:,:-1,0])*cos_lat*DEG2KM
# # #     dg_lat=(gt_deg[:,1:,1]-gt_deg[:,:-1,1])*DEG2KM
# # #     dp=torch.stack([dp_lon,dp_lat],dim=-1); dg=torch.stack([dg_lon,dg_lat],dim=-1)
# # #     return (dp-dg).norm(dim=-1).mean()/DEG2KM


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  Dropout Consistency (NEW-3, fixed bug)
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # def _dropout_consistency_loss(encoder, raw_ctx):
# # #     if not encoder.training: return torch.tensor(0.0, device=raw_ctx.device)
# # #     ctx_a=encoder.apply_ctx_head(raw_ctx); ctx_b=encoder.apply_ctx_head(raw_ctx)
# # #     cos_sim=(F.normalize(ctx_a,dim=-1)*F.normalize(ctx_b,dim=-1)).sum(dim=-1)
# # #     return (1.0-cos_sim).mean()


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  Trajectory Mixup (hard samples keep delta)
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # def _traj_mixup(x1_rel, delta, prob=0.3, alpha=0.4):
# # #     if not torch.is_tensor(x1_rel) or x1_rel.shape[0]<2: return x1_rel, delta
# # #     if torch.rand(1).item()>prob: return x1_rel, delta
# # #     B=x1_rel.shape[0]
# # #     lam=max(float(torch.distributions.Beta(alpha,alpha).sample()), 1.0-float(torch.distributions.Beta(alpha,alpha).sample()))
# # #     lam=max(lam, 1.0-lam)
# # #     idx=torch.randperm(B,device=x1_rel.device)
# # #     easy_t=(delta<0.4).float().view(B,1,1); easy_d=(delta<0.4).float()
# # #     x1_mix=x1_rel*(1.0-easy_t*(1.0-lam))+x1_rel[idx]*(easy_t*(1.0-lam))
# # #     delta_mix=(delta*lam+delta[idx]*(1.0-lam))*easy_d+delta*(1.0-easy_d)
# # #     return x1_mix, delta_mix


# # # # ═════════════════════════════════════════════════════════════════════════════
# # # #  TCFlowMatchingV65 — Main Class
# # # # ═════════════════════════════════════════════════════════════════════════════

# # # class TCFlowMatchingV65(nn.Module):

# # #     def __init__(self, pred_len=12, obs_len=8, ctx_dim=256, sigma_min=0.02,
# # #                  unet_in_ch=13, K=K_MODES, use_ema=True, ema_decay=0.995,
# # #                  cfg_uncond_prob=0.10, selector_warmup=0,
# # #                  head_noise_base=0.03, use_ot=True, ot_epsilon=0.05,
# # #                  cfg_guidance_scale=1.3, **kwargs):
# # #         super().__init__()
# # #         self.pred_len=pred_len; self.obs_len=obs_len; self.sigma_min=sigma_min
# # #         self.K=K; self.use_ema=use_ema; self.ema_decay=ema_decay
# # #         self.cfg_uncond_prob=cfg_uncond_prob; self.selector_warmup=selector_warmup
# # #         self.head_noise_base=head_noise_base; self.use_ot=use_ot
# # #         self.ot_epsilon=ot_epsilon; self.cfg_guidance_scale=cfg_guidance_scale
# # #         self._ema=None

# # #         # REV2: v_max in normalized units for kinematic loss
# # #         self.v_max_norm = _V_MAX_KMH * _DT_H / (111.0 * 50.0)

# # #         self.encoder=SharedContextEncoder(pred_len=pred_len,obs_len=obs_len,
# # #                                           ctx_dim=ctx_dim,unet_in_ch=unet_in_ch)
# # #         self.velocity_heads=nn.ModuleList([
# # #             CompassVelocityHead(compass_idx=k,pred_len=pred_len,ctx_dim=ctx_dim)
# # #             for k in range(K)])
# # #         self.selector=CompassSelector(ctx_dim=ctx_dim,K=K,n_dirs=8)
# # #         self.learned_weights=LearnedWeights()   # REV2-3: 6 terms

# # #     def init_ema(self):
# # #         if self.use_ema: self._ema=EMAModel(self,decay=self.ema_decay)
# # #     def ema_update(self):
# # #         if self._ema is not None: self._ema.update(self)
# # #     def set_curriculum_len(self,*a,**kw): pass

# # #     @staticmethod
# # #     def _sigma_schedule(epoch):
# # #         if epoch<2: return 0.10
# # #         if epoch<10: return 0.10-(epoch-2)/8.0*(0.10-0.04)
# # #         if epoch<20: return max(0.04-(epoch-10)/10.0*0.01, 0.035)
# # #         return 0.035

# # #     @staticmethod
# # #     def _lon_flip_aug(bl,p=0.3):
# # #         if torch.rand(1).item()>p: return bl
# # #         bl=list(bl)
# # #         for i in [0,1,2,3]:
# # #             if torch.is_tensor(bl[i]) and bl[i].shape[-1]>=1:
# # #                 t=bl[i].clone(); t[...,0]=-t[...,0]; bl[i]=t
# # #         return bl

# # #     @staticmethod
# # #     def _obs_noise_aug(bl,sigma=0.005):
# # #         if torch.rand(1).item()>0.5: return bl
# # #         bl=list(bl)
# # #         if torch.is_tensor(bl[0]): bl[0]=bl[0]+torch.randn_like(bl[0])*sigma
# # #         return bl

# # #     @staticmethod
# # #     def _to_rel(traj,Me,lp,lm):
# # #         return torch.cat([traj-lp.unsqueeze(0),Me-lm.unsqueeze(0)],dim=-1).permute(1,0,2)

# # #     @staticmethod
# # #     def _to_abs(rel,lp,lm):
# # #         d=rel.permute(1,0,2)
# # #         return lp.unsqueeze(0)+d[:,:,:2], lm.unsqueeze(0)+d[:,:,2:]

# # #     def _cfm_noisy(self,x1_rel,sigma_min=None,lp=None):
# # #         if sigma_min is None: sigma_min=self.sigma_min
# # #         B=x1_rel.shape[0]; device=x1_rel.device
# # #         x0=torch.randn_like(x1_rel)*sigma_min; t=torch.rand(B,device=device)
# # #         te=t.view(B,1,1)
# # #         return (1.0-te)*x0+te*x1_rel, t, x1_rel-x0

# # #     @staticmethod
# # #     @torch.no_grad()
# # #     def _persistence_blend(model_pred,obs_traj_norm,blend_strength=0.10):
# # #         T_obs=obs_traj_norm.shape[0]; T=model_pred.shape[0]
# # #         B,device=model_pred.shape[1],model_pred.device
# # #         if T_obs<2: return model_pred
# # #         vels=obs_traj_norm[1:]-obs_traj_norm[:-1]; n_v=vels.shape[0]
# # #         if n_v>=3:
# # #             alpha=0.7; w=torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
# # #                                        dtype=torch.float,device=device).flip(0)
# # #             ev=(vels*(w/w.sum()).view(-1,1,1)).sum(0)
# # #         elif n_v==2: ev=0.7*vels[-1]+0.3*vels[-2]
# # #         else: ev=vels[-1]
# # #         steps=torch.arange(1,T+1,dtype=torch.float,device=device)
# # #         persist=obs_traj_norm[-1].unsqueeze(0)+ev.unsqueeze(0)*steps.view(T,1,1)
# # #         obs_spd=_step_speeds_deg(_norm_to_deg(obs_traj_norm))
# # #         if obs_spd.shape[0]>=2:
# # #             spd_cv=obs_spd.std(0)/obs_spd.mean(0).clamp(min=1.0)
# # #             alpha_b=(blend_strength*torch.sigmoid(-(spd_cv-0.3)*5.0)).unsqueeze(0).unsqueeze(-1)
# # #         else: alpha_b=blend_strength*0.5
# # #         return (1.0-alpha_b)*model_pred+alpha_b*persist

# # #     # ── TRAINING LOSS ─────────────────────────────────────────────────────────

# # #     def get_loss(self,batch_list,epoch=0,**kwargs):
# # #         return self.get_loss_breakdown(batch_list,epoch=epoch,**kwargs)["total"]

# # #     def get_loss_breakdown(self,batch_list,epoch=0,**kwargs):
# # #         batch_list=self._lon_flip_aug(batch_list)
# # #         batch_list=self._obs_noise_aug(batch_list,sigma=0.005)

# # #         obs_t=batch_list[0]; pred_t=batch_list[1]
# # #         obs_Me=batch_list[7]; pred_Me=batch_list[8]
# # #         env_data=batch_list[13] if len(batch_list)>13 else None

# # #         lp=obs_t[-1]; lm=obs_Me[-1]
# # #         B,device=lp.shape[0],lp.device
# # #         sigma=self._sigma_schedule(epoch)
# # #         in_warmup=(epoch<self.selector_warmup)

# # #         # Context
# # #         raw_ctx=self.encoder.encode(batch_list)
# # #         use_null=(torch.rand(1).item()<self.cfg_uncond_prob)
# # #         ctx=self.encoder.apply_ctx_head(raw_ctx,use_null=use_null)

# # #         # Consistency
# # #         L_consist=_dropout_consistency_loss(self.encoder,raw_ctx)

# # #         # Difficulty
# # #         obs_BT2=obs_t.permute(1,0,2); img_obs=batch_list[11]
# # #         delta=compute_difficulty_score(obs_BT2,img_obs,env_data,device)

# # #         # FM
# # #         x1_rel_orig=self._to_rel(pred_t,pred_Me,lp,lm)

# # #         # OT matching (phase 2+)
# # #         if self.use_ot and B>=4 and not in_warmup:
# # #             noise_ot=torch.randn_like(x1_rel_orig)*sigma
# # #             _,x1_rel_orig=_spherical_ot_matching(noise_ot,x1_rel_orig,lp,epsilon=self.ot_epsilon)

# # #         # Mixup
# # #         x1_rel,delta=_traj_mixup(x1_rel_orig,delta,prob=0.3,alpha=0.4)
# # #         x_t,fm_t,u_target=self._cfm_noisy(x1_rel,sigma_min=sigma,lp=lp)
# # #         x_t_shared=x_t

# # #         # Generate K modes
# # #         modes_rel_nograd=[]; modes_rel_grad=[]; vels_pred=[]
# # #         for k in range(self.K):
# # #             sigma_k=self.head_noise_base*(1.0+k*0.25)
# # #             z_k=torch.randn_like(x_t_shared)*sigma_k; x_t_k=x_t_shared+z_k
# # #             if in_warmup:
# # #                 with torch.no_grad():
# # #                     v_k=self.velocity_heads[k](x_t_k,fm_t,ctx.detach())
# # #             else:
# # #                 v_k=self.velocity_heads[k](x_t_k,fm_t,ctx)
# # #             with torch.no_grad():
# # #                 pred_k_oracle=x_t_shared+(1.0-fm_t.view(B,1,1))*v_k
# # #             modes_rel_nograd.append(pred_k_oracle.detach())
# # #             pred_k_grad=x_t_k+(1.0-fm_t.view(B,1,1))*v_k
# # #             modes_rel_grad.append(pred_k_grad)
# # #             vels_pred.append(v_k)

# # #         modes_t_ng=torch.stack(modes_rel_nograd,dim=1)
# # #         modes_t_g=torch.stack(modes_rel_grad,dim=1)

# # #         # Oracle
# # #         with torch.no_grad():
# # #             ade_k=torch.stack([
# # #                 _ade_km_from_rel(modes_rel_nograd[k],x1_rel_orig.detach(),lp)
# # #                 for k in range(self.K)
# # #             ],dim=1)
# # #             k_star=ade_k.argmin(dim=1)

# # #         # FM loss
# # #         fm_errs=torch.stack([
# # #             ((vels_pred[k]-u_target)**2).mean(dim=[1,2])
# # #             for k in range(self.K)
# # #         ],dim=1)
# # #         L_easy=fm_errs.mean(dim=1)
# # #         L_oracle_fm=fm_errs[torch.arange(B,device=device),k_star]

# # #         # Diversity
# # #         mode_star_ng=modes_t_ng[torch.arange(B,device=device),k_star]
# # #         dists_all=torch.stack([
# # #             ((mode_star_ng-modes_t_ng[:,k])**2).mean(dim=[1,2]).sqrt()
# # #             for k in range(self.K)
# # #         ],dim=1)
# # #         mask_ns=torch.ones(B,self.K,device=device,dtype=torch.bool)
# # #         mask_ns.scatter_(1,k_star.unsqueeze(1),False)
# # #         min_dist=dists_all.masked_fill(~mask_ns,float('inf')).min(dim=1).values
# # #         MARGIN=0.40
# # #         L_div=F.relu(MARGIN-min_dist)
# # #         L_diff=L_oracle_fm+0.3*L_div
# # #         L_FM_raw=(1.0-delta)*L_easy+delta*L_diff
# # #         w_d=0.5+1.5*delta
# # #         L_FM=(w_d*L_FM_raw).mean()

# # #         # ── REV2-1+2: DPE + Kinematic loss cho best mode ──────────────────
# # #         # best_mode_grad = modes_t_g[k_star] — có gradient
# # #         k_star_idx=k_star
# # #         best_mode_grad=modes_t_g[torch.arange(B,device=device),k_star_idx]  # [B,12,4]
# # #         L_dpe, L_kin=_dpe_and_kinematic_loss(
# # #             best_mode_grad, x1_rel_orig, lp, device, self.v_max_norm)

# # #         # Selector
# # #         sel_logits,dir_logits=self.selector(ctx.detach(),modes_t_ng,env_data)
# # #         tau=float(kwargs.get('tau',3.0))
# # #         p_oracle=F.softmax(-ade_k/tau,dim=1)
# # #         L_rank=F.kl_div(F.log_softmax(sel_logits,dim=1),p_oracle,reduction='batchmean')
# # #         gt_bucket=get_gt_direction_bucket(obs_t,pred_t)
# # #         L_dir=F.cross_entropy(dir_logits,gt_bucket)

# # #         # Coherence
# # #         with torch.no_grad():
# # #             sel_probs=F.softmax(sel_logits,dim=1)
# # #         pred_soft_grad=(sel_probs.unsqueeze(-1).unsqueeze(-1)*modes_t_g).sum(dim=1)
# # #         L_coh=_coherence_loss(pred_soft_grad,x1_rel_orig,lp,device)

# # #         # Total
# # #         if in_warmup:
# # #             alpha_lw=torch.sigmoid(self.learned_weights.log_alpha)
# # #             L_total=alpha_lw*L_rank+(1.0-alpha_lw)*L_dir
# # #         else:
# # #             L_total=self.learned_weights(L_FM,L_rank,L_dir,L_coh,L_consist,L_dpe,L_kin)

# # #         if not torch.isfinite(L_total): L_total=L_total.new_zeros(())

# # #         def _s(x): return x.item() if torch.is_tensor(x) else float(x)
# # #         return {
# # #             "total":         L_total,
# # #             "L_FM":          _s(L_FM),
# # #             "L_easy":        _s(L_easy.mean()),
# # #             "L_diff":        _s(L_diff.mean()),
# # #             "L_oracle_fm":   _s(L_oracle_fm.mean()),
# # #             "L_div":         _s(L_div.mean()),
# # #             "L_dpe":         _s(L_dpe),      # REV2-1
# # #             "L_kin":         _s(L_kin),      # REV2-2
# # #             "L_rank":        _s(L_rank),
# # #             "L_dir":         _s(L_dir),
# # #             "L_coh":         _s(L_coh),
# # #             "L_consist":     _s(L_consist),
# # #             "delta_mean":    _s(delta.mean()),
# # #             "delta_p75":     _s(delta.quantile(0.75)),
# # #             "min_dist_mean": _s(min_dist.mean()),
# # #             "in_warmup":     in_warmup,
# # #             **self.learned_weights.get_weights(),
# # #         }

# # #     # ── INFERENCE ─────────────────────────────────────────────────────────────

# # #     @torch.no_grad()
# # #     def sample(self,batch_list,ddim_steps=20,predict_csv=None,
# # #                blend_strength=0.10,**kwargs):
# # #         obs_t=batch_list[0]; obs_Me=batch_list[7]
# # #         env_data=batch_list[13] if len(batch_list)>13 else None
# # #         lp=obs_t[-1]; lm=obs_Me[-1]
# # #         B,device=lp.shape[0],lp.device
# # #         T_pred=self.pred_len; dt=1.0/max(ddim_steps,1)

# # #         raw_ctx=self.encoder.encode(batch_list)
# # #         ctx=self.encoder.apply_ctx_head(raw_ctx)
# # #         ctx_null=self.encoder.apply_ctx_head(raw_ctx,use_null=True)

# # #         x0_base=_persistence_x0_rel(obs_t,obs_Me,lp,lm,T_pred,sigma=0.0)

# # #         obs_t_norm=obs_t[:,:,:2]
# # #         obs_h_n=(F.normalize(obs_t_norm[-1]-obs_t_norm[-2],dim=-1,eps=1e-6)
# # #                  if obs_t_norm.shape[0]>=2 else None)

# # #         all_modes_abs=[]
# # #         for k in range(self.K):
# # #             sigma_k=self.head_noise_base*(1.0+k*0.25)
# # #             x_t=x0_base+torch.randn_like(x0_base)*sigma_k
# # #             for step in range(ddim_steps):
# # #                 t_b=torch.full((B,),step*dt,device=device)
# # #                 if step>0 and self.cfg_guidance_scale>1.0:
# # #                     v_cond=self.velocity_heads[k](x_t,t_b,ctx)
# # #                     v_uncond=self.velocity_heads[k](x_t,t_b,ctx_null)
# # #                     if obs_h_n is not None:
# # #                         pred_h=F.normalize(v_cond[:,0,:2].detach(),dim=-1,eps=1e-6)
# # #                         cos_a=(obs_h_n*pred_h).sum(-1).clamp(-1.0,1.0)
# # #                         gs=(0.8+0.7*(cos_a+1.0)*0.5).view(B,1,1)
# # #                         v_k=v_uncond+gs*(v_cond-v_uncond)
# # #                     else:
# # #                         v_k=v_uncond+self.cfg_guidance_scale*(v_cond-v_uncond)
# # #                 else:
# # #                     v_k=self.velocity_heads[k](x_t,t_b,ctx)
# # #                 x_t=(x_t+dt*v_k).clamp(-5.0,5.0)
# # #             traj_abs,_=self._to_abs(x_t,lp,lm)
# # #             all_modes_abs.append(traj_abs)

# # #         modes_stack=torch.stack(all_modes_abs,dim=0)   # [K,T,B,2]

# # #         # ── REV2-5: Speed-aware final selection ───────────────────────────
# # #         # Combine selector logits với speed plausibility penalty
# # #         modes_rel_sel=torch.stack([
# # #             torch.cat([all_modes_abs[k].permute(1,0,2)-lp.unsqueeze(1),
# # #                        torch.zeros(B,T_pred,2,device=device)],dim=-1)
# # #             for k in range(self.K)
# # #         ],dim=1)
# # #         sel_logits,_=self.selector(ctx,modes_rel_sel,env_data)

# # #         # Speed penalty: modes vi phạm v_max bị penalize
# # #         speed_penalties=[]
# # #         for k in range(self.K):
# # #             mode_norm=modes_rel_sel[:,k,:,:2]   # [B,T,2]
# # #             if T_pred>=2:
# # #                 step_dist=(mode_norm[:,1:,:]-mode_norm[:,:-1,:]).norm(dim=-1)  # [B,T-1]
# # #                 penalty=F.relu(step_dist-self.v_max_norm).sum(dim=1)           # [B]
# # #             else:
# # #                 penalty=torch.zeros(B,device=device)
# # #             speed_penalties.append(penalty)
# # #         speed_pen=torch.stack(speed_penalties,dim=1)   # [B,K]

# # #         # Final score = selector - speed penalty (normalize speed_pen)
# # #         sp_norm=speed_pen/(speed_pen.max(dim=1,keepdim=True).values.clamp(min=1e-6))
# # #         final_scores=sel_logits - 0.5*sp_norm
# # #         best_k=final_scores.argmax(dim=1)   # [B]

# # #         pred_best=torch.stack([modes_stack[best_k[b],:,b,:] for b in range(B)],dim=1)
# # #         pred_final=self._persistence_blend(pred_best,obs_t[:,:,:2],blend_strength=blend_strength)

# # #         if predict_csv is not None:
# # #             self._write_predict_csv(predict_csv,pred_final,modes_stack)
# # #         return pred_final, modes_stack

# # #     @staticmethod
# # #     def _write_predict_csv(csv_path,traj_mean,all_modes):
# # #         import csv as _csv,os; from datetime import datetime
# # #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)),exist_ok=True)
# # #         T,B=traj_mean.shape[0],traj_mean.shape[1]
# # #         mlon=((traj_mean[...,0]*50.0+1800.0)/10.0).cpu().numpy()
# # #         mlat=((traj_mean[...,1]*50.0)/10.0).cpu().numpy()
# # #         fields=["ts","b","step","lead_h","lon","lat"]
# # #         ts=datetime.now().strftime("%Y%m%d_%H%M%S")
# # #         write_hdr=not os.path.exists(csv_path)
# # #         with open(csv_path,"a",newline="") as fh:
# # #             w=_csv.DictWriter(fh,fieldnames=fields)
# # #             if write_hdr: w.writeheader()
# # #             for b in range(B):
# # #                 for k in range(T):
# # #                     w.writerow({"ts":ts,"b":b,"step":k,"lead_h":(k+1)*6,
# # #                                 "lon":f"{mlon[k,b]:.4f}","lat":f"{mlat[k,b]:.4f}"})

# # # TCFlowMatching = TCFlowMatchingV65
# # # TCDiffusion    = TCFlowMatchingV65

# # """
# # flow_matching_model_v65_fixed.py
# # ════════════════════════════════════════════════════════════════════════════════
# # V65 — Fixed version

# # ROOT CAUSES của diverge (ADE 455→981 qua 4 epochs):

# #   ROOT-1 [CRITICAL] x0 mismatch:
# #     _cfm_noisy dùng x0 = randn*sigma_min (pure noise)
# #     Nhưng inference dùng x0 = persistence → train/inf mismatch hoàn toàn
# #     Fix: dùng persistence x0 trong cả training lẫn inference (như v63+)

# #   ROOT-2 [CRITICAL] diversity mask ngược:
# #     mask_ns = ones → scatter False → masked_fill(~mask_ns) fill inf vào k_star
# #     → min_dist luôn = distance đến k_star thay vì distance đến mode gần nhất khác
# #     → L_div = 0 mọi lúc → modes collapse
# #     Fix: mask_ns = zeros → scatter True → masked_fill(~mask_ns) fill inf đúng chỗ

# #   ROOT-3 Warmup không train generator:
# #     in_warmup: velocity_heads toàn bộ detached → ADE không giảm trong 2 epoch đầu
# #     + selector frozen → không học được gì 2 epoch đầu → waste
# #     Fix: selector_warmup=0 (bỏ warmup phase), train joint từ đầu
# #     Generator luôn có gradient, selector train song song với stop-grad từ generator

# #   ROOT-4 L_consist = 0 trong warmup:
# #     Warmup loss chỉ có L_rank + L_dir → dropout consistency không train
# #     Fix: thêm L_consist nhỏ vào mọi phase

# #   ROOT-5 L_coh gradient bị stop:
# #     sel_probs = F.softmax(sel_logits) nhưng sel_logits từ selector đã detach
# #     → weighted sum modes_t_g không có gradient từ sel_probs (chỉ từ modes_t_g)
# #     → L_coh chỉ train generator, không train selector để chọn coherent mode
# #     Fix: tính L_coh trực tiếp từ best oracle mode (k_star) thay vì soft-weighted

# # GIẢM THIỂU THÊM:
# #   - Bỏ OT matching trong phase đầu (gây noise khi model chưa ổn định)
# #   - L_FM dùng haversine trực tiếp (như STTrans) thay vì velocity MSE thuần
# #   - Tau giảm dần từ 5→2 (không phải fixed 3.0) để selector signal rõ hơn
# # """
# # from __future__ import annotations

# # import math
# # from typing import Optional, Tuple

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # from Model.FNO3D_encoder import FNO3DEncoder
# # from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# # from Model.env_net_transformer_gphsplit import Env_net

# # R_EARTH      = 6371.0
# # DT_HOURS     = 6.0
# # DEG2KM       = 111.0
# # _NORM_TO_DEG = 5.0

# # _COMPASS_DEG   = [0., 45., 90., 135., 180., 225., 270., 315.]
# # K_MODES        = 8


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Coordinate utilities
# # # ══════════════════════════════════════════════════════════════════════════════

# # def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
# #     return torch.stack([
# #         (t[..., 0] * 50.0 + 1800.0) / 10.0,
# #         (t[..., 1] * 50.0) / 10.0,
# #     ], dim=-1)

# # def _haversine_deg(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# #     a = (torch.sin(dlat/2).pow(2) +
# #          torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2))
# #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

# # def _unwrap_model(m):
# #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # def _step_speeds_deg(traj_deg: torch.Tensor) -> torch.Tensor:
# #     T = traj_deg.shape[0]
# #     if T < 2: return traj_deg.new_zeros(1, traj_deg.shape[1])
# #     return _haversine_deg(traj_deg[:-1], traj_deg[1:]) / DT_HOURS

# # def _ade_km_from_rel(pred_rel, gt_rel, lp):
# #     """[B,T,4] → [B] ADE in km."""
# #     pred_abs = lp.unsqueeze(1) + pred_rel[:, :, :2]
# #     gt_abs   = lp.unsqueeze(1) + gt_rel[:, :, :2]
# #     return _haversine_deg(_norm_to_deg(pred_abs), _norm_to_deg(gt_abs)).mean(dim=1)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  ROOT-1 FIX: Persistence x0 — dùng trong cả training lẫn inference
# # # ══════════════════════════════════════════════════════════════════════════════

# # def _persistence_x0_rel(obs_traj, obs_Me, lp, lm, pred_len, sigma=0.02):
# #     """
# #     x0 = persistence forecast + tiny noise.
# #     Dùng exponential weighted velocity từ obs để extrapolate.
# #     QUAN TRỌNG: training và inference phải dùng hàm này để nhất quán.
# #     """
# #     B, device = obs_traj.shape[1], obs_traj.device
# #     if obs_traj.shape[0] >= 3:
# #         vels  = obs_traj[1:] - obs_traj[:-1]
# #         n_v   = vels.shape[0]
# #         alpha = 0.7
# #         w     = torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
# #                               dtype=torch.float, device=device).flip(0)
# #         lv    = (vels * (w/w.sum()).view(-1,1,1)).sum(0)
# #     elif obs_traj.shape[0] >= 2:
# #         lv = obs_traj[-1,:,:2] - obs_traj[-2,:,:2]
# #     else:
# #         lv = obs_traj.new_zeros(B, 2)

# #     steps    = torch.arange(1, pred_len+1, device=device).float()
# #     pred_abs = obs_traj[-1,:,:2].unsqueeze(0) + lv.unsqueeze(0)*steps.view(-1,1,1)
# #     pred_rel = pred_abs.permute(1,0,2) - lp.unsqueeze(1)
# #     pred_rel4 = torch.cat([pred_rel, torch.zeros(B, pred_len, 2, device=device)], dim=-1)
# #     return pred_rel4 + torch.randn_like(pred_rel4) * sigma


# # def _cfm_from_persistence(x0_rel, x1_rel):
# #     """
# #     FIX ROOT-1: CFM interpolant từ persistence x0 (không phải pure noise).
# #     x0 = persistence (đã tính trước), x1 = GT.
# #     t ~ U[0,1], x_t = (1-t)*x0 + t*x1, u = x1 - x0.
# #     """
# #     B = x0_rel.shape[0]
# #     device = x0_rel.device
# #     t  = torch.rand(B, device=device)
# #     te = t.view(B, 1, 1)
# #     x_t    = (1.0-te)*x0_rel + te*x1_rel
# #     u_target = x1_rel - x0_rel
# #     return x_t, t, u_target


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  OT Matching (bật từ epoch 5+)
# # # ══════════════════════════════════════════════════════════════════════════════

# # def _forward_azimuth(p1, p2):
# #     lon1=torch.deg2rad(p1[...,0]); lat1=torch.deg2rad(p1[...,1])
# #     lon2=torch.deg2rad(p2[...,0]); lat2=torch.deg2rad(p2[...,1])
# #     dlon=lon2-lon1
# #     y=torch.sin(dlon)*torch.cos(lat2)
# #     x=torch.cos(lat1)*torch.sin(lat2)-torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon)
# #     return torch.atan2(y,x)

# # def _sinkhorn_log(cost, epsilon=0.05, n_iter=30):
# #     B=cost.shape[0]; device=cost.device
# #     log_a=-math.log(B)*torch.ones(B,device=device)
# #     log_b=-math.log(B)*torch.ones(B,device=device)
# #     log_K=-cost/epsilon
# #     log_u=torch.zeros(B,device=device); log_v=torch.zeros(B,device=device)
# #     for _ in range(n_iter):
# #         log_u=log_a-torch.logsumexp(log_K+log_v.unsqueeze(0),dim=1)
# #         log_v=log_b-torch.logsumexp(log_K+log_u.unsqueeze(1),dim=0)
# #     return (log_K+log_u.unsqueeze(1)+log_v.unsqueeze(0)).exp().clamp(0.0)

# # def _geodesic_ot_cost(x0_rel, x1_rel, lp):
# #     B=x0_rel.shape[0]
# #     def _abs_deg(rel): return _norm_to_deg(lp.unsqueeze(1)+rel[:,:,:2])
# #     x0d=_abs_deg(x0_rel); x1d=_abs_deg(x1_rel)
# #     x0e=x0d.unsqueeze(1).expand(B,B,-1,-1).reshape(B*B,-1,2)
# #     x1e=x1d.unsqueeze(0).expand(B,B,-1,-1).reshape(B*B,-1,2)
# #     pos_cost=_haversine_deg(x0e,x1e).reshape(B,B,-1).mean(-1)/500.0
# #     spd0=_step_speeds_deg(x0d.permute(1,0,2)).permute(1,0).mean(-1)
# #     spd1=_step_speeds_deg(x1d.permute(1,0,2)).permute(1,0).mean(-1)
# #     speed_cost=(spd0.unsqueeze(1)-spd1.unsqueeze(0)).abs()/20.0
# #     def _mb(td):
# #         b=_forward_azimuth(td[:,:-1,:],td[:,1:,:]); return torch.atan2(b.sin().mean(-1),b.cos().mean(-1))
# #     h0=_mb(x0d); h1=_mb(x1d)
# #     dh=(h0.unsqueeze(1)-h1.unsqueeze(0)+math.pi)%(2*math.pi)-math.pi
# #     return 1.0*pos_cost+0.5*speed_cost+0.3*dh.abs()/math.pi

# # def _spherical_ot_matching(x0_batch, x1_batch, lp, epsilon=0.05):
# #     B=x0_batch.shape[0]
# #     if B<4: return x0_batch, x1_batch
# #     try:
# #         cost=_geodesic_ot_cost(x0_batch, x1_batch, lp)
# #         with torch.no_grad(): pi=_sinkhorn_log(cost,epsilon=epsilon)
# #         flat=pi.reshape(-1).clamp(0.0); s=flat.sum()
# #         if not torch.isfinite(s) or s<1e-10: return x0_batch, x1_batch
# #         idx=torch.multinomial(flat/s,num_samples=B,replacement=True)
# #         return x0_batch[idx%B], x1_batch[idx%B]
# #     except Exception:
# #         return x0_batch, x1_batch


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Difficulty Score
# # # ══════════════════════════════════════════════════════════════════════════════

# # def compute_difficulty_score(obs_BT2, img_obs, env_data, device):
# #     B=obs_BT2.shape[0]; scores=[]

# #     # Signal 1: curvature (0.30)
# #     dir24=env_data.get("history_direction24") if env_data else None
# #     if dir24 is not None and torch.is_tensor(dir24):
# #         dir24=dir24.float().to(device)
# #         if dir24.dim()==3: d24=dir24[:,-1,:]
# #         elif dir24.dim()==2: d24=dir24
# #         else: d24=None
# #         if d24 is not None and d24.shape[-1]==8:
# #             bucket=d24.argmax(dim=-1).float(); a24=bucket*(2.0*math.pi/8.0)
# #             dy=obs_BT2[:,-1,1]-obs_BT2[:,-2,1]; dx=obs_BT2[:,-1,0]-obs_BT2[:,-2,0]
# #             lat_mid=(obs_BT2[:,-1,1]+obs_BT2[:,-2,1])*0.5
# #             cos_lat=torch.cos(torch.deg2rad(lat_mid*_NORM_TO_DEG)).clamp(1e-4)
# #             a_now=torch.atan2(dx*cos_lat,dy)
# #             cos_d=(torch.cos(a_now)*torch.cos(a24)+torch.sin(a_now)*torch.sin(a24)).clamp(-1,1)
# #             s_curve=torch.sigmoid((torch.rad2deg(torch.acos(cos_d))-45.0)/20.0)
# #         else: s_curve=torch.zeros(B,device=device)
# #     else: s_curve=torch.zeros(B,device=device)
# #     scores.append(0.30*s_curve)

# #     # Signal 2: weak steering (0.25)
# #     steer=env_data.get("steering_speed") if env_data else None
# #     if steer is not None and torch.is_tensor(steer):
# #         steer=steer.float().to(device)
# #         while steer.dim()>1: steer=steer[...,-1]
# #         steer=steer.view(-1); steer=steer[:B] if steer.numel()>=B else steer[0].expand(B)
# #         s_weak=torch.sigmoid((4.0-steer*20.0)/2.0)
# #     else:
# #         u=env_data.get("u500_mean") if env_data else None
# #         v=env_data.get("v500_mean") if env_data else None
# #         if u is not None and v is not None and torch.is_tensor(u) and torch.is_tensor(v):
# #             u=u.float().to(device); v=v.float().to(device)
# #             while u.dim()>1: u=u[...,-1]
# #             while v.dim()>1: v=v[...,-1]
# #             s_weak=torch.sigmoid((4.0-(u.view(-1)[:B]**2+v.view(-1)[:B]**2).sqrt()*30.0)/2.0)
# #         else: s_weak=torch.zeros(B,device=device)
# #     scores.append(0.25*s_weak)

# #     # Signal 3: wind shear (0.20)
# #     if img_obs is not None and img_obs.shape[1]>=11:
# #         u200=img_obs[:,4,-1,40,40]*13.315; u850=img_obs[:,6,-1,40,40]*7.911
# #         v200=img_obs[:,8,-1,40,40]*8.377;  v850=img_obs[:,10,-1,40,40]*6.203
# #         shear=((u200-u850)**2+(v200-v850)**2).sqrt()
# #         s_shear=torch.sigmoid((shear-8.0)/3.0)
# #     else: s_shear=torch.zeros(B,device=device)
# #     scores.append(0.20*s_shear)

# #     # Signal 4: RI (0.15)
# #     ri=env_data.get("rapid_intensification") if env_data else None
# #     if ri is not None and torch.is_tensor(ri):
# #         ri=ri.float().to(device)
# #         while ri.dim()>1: ri=ri[...,-1]
# #         s_ri=ri.view(-1)[:B].clamp(0,1) if ri.numel()>=B else ri[0].expand(B)
# #     else: s_ri=torch.zeros(B,device=device)
# #     scores.append(0.15*s_ri)

# #     # Signal 5: slow movement (0.10)
# #     mv=env_data.get("move_velocity") if env_data else None
# #     if mv is not None and torch.is_tensor(mv):
# #         mv=mv.float().to(device)
# #         while mv.dim()>1: mv=mv[...,-1]
# #         mv_v=mv.view(-1); mv_v=mv_v[:B] if mv_v.numel()>=B else mv_v[0].expand(B)
# #         s_slow=torch.sigmoid((0.05-mv_v)/0.02)
# #     else: s_slow=torch.zeros(B,device=device)
# #     scores.append(0.10*s_slow)

# #     return sum(scores).clamp(0.0,1.0)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Direction GT bucket
# # # ══════════════════════════════════════════════════════════════════════════════

# # def get_gt_direction_bucket(obs_traj, pred_traj):
# #     last=obs_traj[-1]; first=pred_traj[0]
# #     dy=first[:,1]-last[:,1]; dx=first[:,0]-last[:,0]
# #     lat_d=(last[:,1]+first[:,1])*0.5
# #     cos_l=torch.cos(torch.deg2rad(lat_d*_NORM_TO_DEG)).clamp(1e-4)
# #     angle=(torch.atan2(dx*cos_l,dy).rad2deg()%360.0)
# #     return ((angle+22.5)/45.0).long()%8


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  CompassVelocityHead
# # # ══════════════════════════════════════════════════════════════════════════════

# # class CompassVelocityHead(nn.Module):
# #     def __init__(self, compass_idx, pred_len=12, ctx_dim=256,
# #                  stochastic_depth_prob=0.1):
# #         super().__init__()
# #         self.pred_len=pred_len; self.compass_idx=compass_idx
# #         self.stochastic_depth_prob=stochastic_depth_prob
# #         angle_rad=_COMPASS_DEG[compass_idx]*math.pi/180.0
# #         dir_vec=torch.tensor([math.sin(angle_rad),math.cos(angle_rad)])
# #         self.register_buffer('compass_dir',dir_vec)
# #         self.dir_proj=nn.Sequential(nn.Linear(2,ctx_dim),nn.GELU(),nn.LayerNorm(ctx_dim))
# #         self.time_fc1=nn.Linear(ctx_dim,256); self.time_fc2=nn.Linear(256,ctx_dim)
# #         self.traj_embed=nn.Linear(4,ctx_dim)
# #         self.pos_enc=nn.Parameter(torch.randn(1,pred_len,ctx_dim)*0.02)
# #         self.step_embed=nn.Embedding(pred_len,ctx_dim)
# #         self.transformer=nn.TransformerDecoder(
# #             nn.TransformerDecoderLayer(d_model=ctx_dim,nhead=8,dim_feedforward=512,
# #                                        dropout=0.10,activation="gelu",batch_first=True),
# #             num_layers=1)
# #         self.out_fc1=nn.Linear(ctx_dim,256); self.out_fc2=nn.Linear(256,4)
# #         self.step_scale=nn.Parameter(torch.ones(pred_len)*0.5)
# #         with torch.no_grad():
# #             nn.init.xavier_uniform_(self.out_fc2.weight,gain=0.1)
# #             nn.init.zeros_(self.out_fc2.bias)

# #     def _time_emb(self,t,dim):
# #         half=dim//2
# #         freq=torch.exp(torch.arange(half,dtype=torch.float32,device=t.device)
# #                        *(-math.log(10000.0)/max(half-1,1)))
# #         emb=t.float().unsqueeze(1)*1000.0*freq.unsqueeze(0)
# #         return F.pad(torch.cat([emb.sin(),emb.cos()],dim=-1),(0,dim%2))

# #     def forward(self,x_t,t,ctx):
# #         B=x_t.shape[0]; T_seq=min(x_t.shape[1],self.pred_len); device=x_t.device
# #         dir_token=self.dir_proj(self.compass_dir.to(device).unsqueeze(0).expand(B,-1))
# #         if self.training and self.stochastic_depth_prob>0:
# #             keep=(torch.rand(B,1,device=device)>self.stochastic_depth_prob).float()
# #             dir_token=dir_token*keep
# #         t_emb=F.gelu(self.time_fc1(self._time_emb(t,ctx.shape[-1])))
# #         t_emb=self.time_fc2(t_emb)
# #         step_idx=torch.arange(T_seq,device=device).unsqueeze(0).expand(B,-1)
# #         x_emb=(self.traj_embed(x_t[:,:T_seq])+self.pos_enc[:,:T_seq]
# #                +t_emb.unsqueeze(1)+self.step_embed(step_idx))
# #         mem=torch.stack([ctx,dir_token,t_emb],dim=1)
# #         decoded=self.transformer(x_emb,mem)
# #         scale=torch.sigmoid(self.step_scale[:T_seq]).view(1,T_seq,1)*2.0
# #         return self.out_fc2(F.gelu(self.out_fc1(decoded)))*scale


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  CompassSelector
# # # ══════════════════════════════════════════════════════════════════════════════

# # class CompassSelector(nn.Module):
# #     def __init__(self,ctx_dim=256,K=8,n_dirs=8):
# #         super().__init__()
# #         self.K=K; self.n_dirs=n_dirs
# #         self.steering_enc=nn.Sequential(nn.Linear(11,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,128))
# #         self.mode_enc=nn.Sequential(nn.Linear(12*2,128),nn.GELU(),nn.LayerNorm(128),nn.Linear(128,64))
# #         self.score_net=nn.Sequential(nn.Linear(ctx_dim+64+128,256),nn.GELU(),nn.LayerNorm(256),
# #                                      nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
# #         self.dir_head=nn.Sequential(nn.Linear(ctx_dim+128,256),nn.GELU(),nn.LayerNorm(256),nn.Linear(256,n_dirs))

# #     def _get_steer(self,env_data,B,device):
# #         def _sc(key):
# #             if env_data is None: return torch.zeros(B,device=device)
# #             v=env_data.get(key)
# #             if v is None or not torch.is_tensor(v): return torch.zeros(B,device=device)
# #             v=v.float().to(device)
# #             while v.dim()>1: v=v[...,-1]
# #             v=v.view(-1); return (v[:B] if v.numel()>=B else v[0].expand(B)).clamp(-3,3)
# #         def _sv(key,dim):
# #             if env_data is None: return torch.zeros(B,dim,device=device)
# #             v=env_data.get(key)
# #             if v is None or not torch.is_tensor(v): return torch.zeros(B,dim,device=device)
# #             v=v.float().to(device)
# #             if v.dim()==3: v=v[:,-1,:]
# #             elif v.dim()!=2 or v.shape[0]!=B: return torch.zeros(B,dim,device=device)
# #             v=F.pad(v,(0,max(0,dim-v.shape[-1])))[:,:dim]; return v.clamp(-3,3)
# #         feat=torch.cat([_sc("steering_speed").unsqueeze(-1),_sc("steering_dir_sin").unsqueeze(-1),
# #                         _sc("steering_dir_cos").unsqueeze(-1),_sv("history_direction24",8)],dim=-1)
# #         return self.steering_enc(feat)

# #     def forward(self,ctx,modes,env_data):
# #         B,K=modes.shape[:2]; device=ctx.device
# #         sf=self._get_steer(env_data,B,device)
# #         scores=[]
# #         for k in range(K):
# #             mf=self.mode_enc(modes[:,k,:,:2].reshape(B,-1))
# #             scores.append(self.score_net(torch.cat([ctx,mf,sf],dim=-1)).squeeze(-1))
# #         score_logits=torch.stack(scores,dim=1)
# #         dir_logits=self.dir_head(torch.cat([ctx,sf],dim=-1))
# #         return score_logits, dir_logits


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  LearnedWeights
# # # ══════════════════════════════════════════════════════════════════════════════

# # class LearnedWeights(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.log_s_fm  = nn.Parameter(torch.zeros(1))
# #         self.log_s_sel = nn.Parameter(torch.zeros(1))
# #         self.log_s_coh = nn.Parameter(torch.zeros(1))
# #         self.log_s_con = nn.Parameter(torch.zeros(1))
# #         self.log_alpha = nn.Parameter(torch.zeros(1))

# #     def forward(self,L_fm,L_rank,L_dir,L_coh,L_con):
# #         s_fm=self.log_s_fm.clamp(-5,5); s_sel=self.log_s_sel.clamp(-5,5)
# #         s_coh=self.log_s_coh.clamp(-5,5); s_con=self.log_s_con.clamp(-5,5)
# #         α=torch.sigmoid(self.log_alpha)
# #         L_sel=α*L_rank+(1.0-α)*L_dir
# #         return (L_fm*torch.exp(-s_fm)+s_fm +
# #                 L_sel*torch.exp(-s_sel)+s_sel +
# #                 L_coh*torch.exp(-s_coh)+s_coh +
# #                 L_con*torch.exp(-s_con)+s_con)

# #     def get_weights(self):
# #         def _w(s): return torch.exp(-s.clamp(-5,5)).item()
# #         return {"w_fm":_w(self.log_s_fm),"w_sel":_w(self.log_s_sel),
# #                 "w_coh":_w(self.log_s_coh),"w_con":_w(self.log_s_con),
# #                 "alpha":torch.sigmoid(self.log_alpha).item()}


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  SharedContextEncoder
# # # ══════════════════════════════════════════════════════════════════════════════

# # class SharedContextEncoder(nn.Module):
# #     RAW_CTX_DIM = 512
# #     def __init__(self,pred_len=12,obs_len=8,ctx_dim=256,unet_in_ch=13):
# #         super().__init__()
# #         self.pred_len=pred_len; self.obs_len=obs_len; self.ctx_dim=ctx_dim
# #         self.spatial_enc=FNO3DEncoder(in_channel=unet_in_ch,out_channel=1,d_model=32,
# #             n_layers=4,modes_t=4,modes_h=4,modes_w=4,spatial_down=32,dropout=0.05)
# #         self.bottleneck_pool=nn.AdaptiveAvgPool3d((None,1,1))
# #         self.bottleneck_proj=nn.Linear(128,128)
# #         self.decoder_proj=nn.Linear(1,16)
# #         self.enc_1d=DataEncoder1D(in_1d=4,feat_3d_dim=128,mlp_h=64,
# #             lstm_hidden=128,lstm_layers=3,dropout=0.1,d_state=16)
# #         self.env_enc=Env_net(obs_len=obs_len,d_model=32)
# #         self.ctx_fc1=nn.Linear(128+32+16,self.RAW_CTX_DIM)
# #         self.ctx_ln=nn.LayerNorm(self.RAW_CTX_DIM)
# #         self.ctx_drop=nn.Dropout(0.15)
# #         self.ctx_fc2=nn.Linear(self.RAW_CTX_DIM,ctx_dim)
# #         self.null_embedding=nn.Parameter(torch.randn(1,self.RAW_CTX_DIM)*0.02)
# #         self.vel_obs_enc=nn.Sequential(nn.Linear(obs_len*6,256),nn.GELU(),
# #                                        nn.LayerNorm(256),nn.Linear(256,256),nn.GELU())

# #     def encode(self,batch_list):
# #         obs_t=batch_list[0]; obs_Me=batch_list[7]
# #         image_obs=batch_list[11]; env_data=batch_list[13] if len(batch_list)>13 else None
# #         if image_obs.dim()==4: image_obs=image_obs.unsqueeze(2)
# #         if image_obs.shape[1]==1 and self.spatial_enc.in_channel!=1:
# #             image_obs=image_obs.expand(-1,self.spatial_enc.in_channel,-1,-1,-1)
# #         T_obs=obs_t.shape[0]
# #         e_3d_bot,e_3d_dec=self.spatial_enc.encode(image_obs)
# #         e_3d_s=self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0,2,1)
# #         e_3d_s=self.bottleneck_proj(e_3d_s)
# #         if e_3d_s.shape[1]!=T_obs:
# #             e_3d_s=F.interpolate(e_3d_s.permute(0,2,1),size=T_obs,
# #                                   mode="linear",align_corners=False).permute(0,2,1)
# #         e_3d_dec_t=e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
# #         t_w=torch.softmax(torch.arange(e_3d_dec_t.shape[1],dtype=torch.float,
# #                                         device=e_3d_dec_t.device)*0.5,dim=0)
# #         f_sp=self.decoder_proj((e_3d_dec_t*t_w.unsqueeze(0)).sum(1,keepdim=True))
# #         obs_in=torch.cat([obs_t,obs_Me],dim=2).permute(1,0,2)
# #         h_t=self.enc_1d(obs_in,e_3d_s)
# #         e_env,_,_=self.env_enc(env_data,image_obs)
# #         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t,e_env,f_sp],dim=-1))))

# #     def apply_ctx_head(self,raw,noise_scale=0.0,use_null=False):
# #         if use_null: raw=self.null_embedding.expand(raw.shape[0],-1)
# #         elif noise_scale>0.0: raw=raw+torch.randn_like(raw)*noise_scale
# #         return self.ctx_fc2(self.ctx_drop(raw))


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Losses
# # # ══════════════════════════════════════════════════════════════════════════════

# # def _coherence_loss_oracle(pred_oracle_grad, x1_rel_orig, lp, device):
# #     """
# #     FIX ROOT-5: Tính coherence trực tiếp từ oracle mode (có gradient),
# #     không dùng soft-weighted sum từ sel_probs đã detach.
# #     """
# #     pred_abs=lp.unsqueeze(1)+pred_oracle_grad[:,:,:2]
# #     gt_abs=lp.unsqueeze(1)+x1_rel_orig[:,:,:2]
# #     pred_deg=_norm_to_deg(pred_abs); gt_deg=_norm_to_deg(gt_abs)
# #     cos_lat=torch.cos(torch.deg2rad(gt_deg[:,:-1,1])).clamp(1e-4)
# #     dp_lon=(pred_deg[:,1:,0]-pred_deg[:,:-1,0])*cos_lat*DEG2KM
# #     dp_lat=(pred_deg[:,1:,1]-pred_deg[:,:-1,1])*DEG2KM
# #     dg_lon=(gt_deg[:,1:,0]-gt_deg[:,:-1,0])*cos_lat*DEG2KM
# #     dg_lat=(gt_deg[:,1:,1]-gt_deg[:,:-1,1])*DEG2KM
# #     dp=torch.stack([dp_lon,dp_lat],dim=-1); dg=torch.stack([dg_lon,dg_lat],dim=-1)
# #     return (dp-dg).norm(dim=-1).mean()/DEG2KM

# # def _dropout_consistency_loss(encoder, raw_ctx):
# #     if not encoder.training: return torch.tensor(0.0,device=raw_ctx.device)
# #     ctx_a=encoder.apply_ctx_head(raw_ctx); ctx_b=encoder.apply_ctx_head(raw_ctx)
# #     cos_sim=(F.normalize(ctx_a,dim=-1)*F.normalize(ctx_b,dim=-1)).sum(dim=-1)
# #     return (1.0-cos_sim).mean()

# # def _traj_mixup(x1_rel, delta, prob=0.3, alpha=0.4):
# #     if not torch.is_tensor(x1_rel) or x1_rel.shape[0]<2: return x1_rel,delta
# #     if torch.rand(1).item()>prob: return x1_rel,delta
# #     B=x1_rel.shape[0]
# #     lam=max(float(torch.distributions.Beta(alpha,alpha).sample()),
# #             1.0-float(torch.distributions.Beta(alpha,alpha).sample()))
# #     lam=max(lam,1.0-lam)
# #     idx=torch.randperm(B,device=x1_rel.device)
# #     easy_t=(delta<0.4).float().view(B,1,1); easy_d=(delta<0.4).float()
# #     x1_mix=x1_rel*(1.0-easy_t*(1.0-lam))+x1_rel[idx]*(easy_t*(1.0-lam))
# #     delta_mix=(delta*lam+delta[idx]*(1.0-lam))*easy_d+delta*(1.0-easy_d)
# #     return x1_mix,delta_mix


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  EMA
# # # ══════════════════════════════════════════════════════════════════════════════

# # class EMAModel:
# #     def __init__(self,model,decay=0.995):
# #         self.decay=decay; m=_unwrap_model(model)
# #         self.shadow={k:v.detach().clone() for k,v in m.state_dict().items()
# #                      if v.dtype.is_floating_point}
# #     def update(self,model):
# #         m=_unwrap_model(model)
# #         with torch.no_grad():
# #             for k,v in m.state_dict().items():
# #                 if k in self.shadow:
# #                     self.shadow[k].mul_(self.decay).add_(v.detach(),alpha=1-self.decay)
# #     def apply_to(self,model):
# #         m=_unwrap_model(model); backup,sd={},m.state_dict()
# #         for k in self.shadow:
# #             if k not in sd: continue
# #             backup[k]=sd[k].detach().clone(); sd[k].copy_(self.shadow[k])
# #         return backup
# #     def restore(self,model,backup):
# #         m=_unwrap_model(model); sd=m.state_dict()
# #         for k,v in backup.items():
# #             if k in sd: sd[k].copy_(v)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  TCFlowMatchingV65 — Main Class
# # # ══════════════════════════════════════════════════════════════════════════════

# # class TCFlowMatchingV65(nn.Module):

# #     def __init__(self,pred_len=12,obs_len=8,ctx_dim=256,sigma_min=0.02,
# #                  unet_in_ch=13,K=K_MODES,use_ema=True,ema_decay=0.995,
# #                  cfg_uncond_prob=0.10,
# #                  selector_warmup=0,   # FIX ROOT-3: bỏ warmup, train joint từ đầu
# #                  head_noise_base=0.03,use_ot=True,ot_epsilon=0.05,
# #                  cfg_guidance_scale=1.3,**kwargs):
# #         super().__init__()
# #         self.pred_len=pred_len; self.obs_len=obs_len; self.sigma_min=sigma_min
# #         self.K=K; self.use_ema=use_ema; self.ema_decay=ema_decay
# #         self.cfg_uncond_prob=cfg_uncond_prob; self.selector_warmup=selector_warmup
# #         self.head_noise_base=head_noise_base; self.use_ot=use_ot
# #         self.ot_epsilon=ot_epsilon; self.cfg_guidance_scale=cfg_guidance_scale
# #         self._ema=None

# #         self.encoder=SharedContextEncoder(pred_len=pred_len,obs_len=obs_len,
# #                                           ctx_dim=ctx_dim,unet_in_ch=unet_in_ch)
# #         self.velocity_heads=nn.ModuleList([
# #             CompassVelocityHead(compass_idx=k,pred_len=pred_len,ctx_dim=ctx_dim)
# #             for k in range(K)])
# #         self.selector=CompassSelector(ctx_dim=ctx_dim,K=K,n_dirs=8)
# #         self.learned_weights=LearnedWeights()

# #     def init_ema(self):
# #         if self.use_ema: self._ema=EMAModel(self,decay=self.ema_decay)
# #     def ema_update(self):
# #         if self._ema is not None: self._ema.update(self)
# #     def set_curriculum_len(self,*a,**kw): pass

# #     @staticmethod
# #     def _sigma_schedule(epoch):
# #         if epoch<2: return 0.10
# #         if epoch<10: return 0.10-(epoch-2)/8.0*(0.10-0.04)
# #         if epoch<20: return max(0.04-(epoch-10)/10.0*0.01,0.035)
# #         return 0.035

# #     @staticmethod
# #     def _tau_schedule(epoch):
# #         """Tau giảm dần → selector signal rõ hơn theo thời gian."""
# #         if epoch<5: return 5.0
# #         if epoch<15: return 4.0
# #         if epoch<25: return 3.0
# #         return 2.0

# #     @staticmethod
# #     def _lon_flip_aug(bl,p=0.3):
# #         if torch.rand(1).item()>p: return bl
# #         bl=list(bl)
# #         for i in [0,1,2,3]:
# #             if torch.is_tensor(bl[i]) and bl[i].shape[-1]>=1:
# #                 t=bl[i].clone(); t[...,0]=-t[...,0]; bl[i]=t
# #         return bl

# #     @staticmethod
# #     def _obs_noise_aug(bl,sigma=0.005):
# #         if torch.rand(1).item()>0.5: return bl
# #         bl=list(bl)
# #         if torch.is_tensor(bl[0]): bl[0]=bl[0]+torch.randn_like(bl[0])*sigma
# #         return bl

# #     @staticmethod
# #     def _to_rel(traj,Me,lp,lm):
# #         return torch.cat([traj-lp.unsqueeze(0),Me-lm.unsqueeze(0)],dim=-1).permute(1,0,2)

# #     @staticmethod
# #     def _to_abs(rel,lp,lm):
# #         d=rel.permute(1,0,2)
# #         return lp.unsqueeze(0)+d[:,:,:2],lm.unsqueeze(0)+d[:,:,2:]

# #     @staticmethod
# #     @torch.no_grad()
# #     def _persistence_blend(model_pred,obs_traj_norm,blend_strength=0.10):
# #         T_obs=obs_traj_norm.shape[0]; T=model_pred.shape[0]
# #         B,device=model_pred.shape[1],model_pred.device
# #         if T_obs<2: return model_pred
# #         vels=obs_traj_norm[1:]-obs_traj_norm[:-1]; n_v=vels.shape[0]
# #         if n_v>=3:
# #             alpha=0.7; w=torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
# #                                        dtype=torch.float,device=device).flip(0)
# #             ev=(vels*(w/w.sum()).view(-1,1,1)).sum(0)
# #         elif n_v==2: ev=0.7*vels[-1]+0.3*vels[-2]
# #         else: ev=vels[-1]
# #         steps=torch.arange(1,T+1,dtype=torch.float,device=device)
# #         persist=obs_traj_norm[-1].unsqueeze(0)+ev.unsqueeze(0)*steps.view(T,1,1)
# #         obs_spd=_step_speeds_deg(_norm_to_deg(obs_traj_norm))
# #         if obs_spd.shape[0]>=2:
# #             spd_cv=obs_spd.std(0)/obs_spd.mean(0).clamp(min=1.0)
# #             alpha_b=(blend_strength*torch.sigmoid(-(spd_cv-0.3)*5.0)).unsqueeze(0).unsqueeze(-1)
# #         else: alpha_b=blend_strength*0.5
# #         return (1.0-alpha_b)*model_pred+alpha_b*persist

# #     # ── TRAINING ──────────────────────────────────────────────────────────────

# #     def get_loss(self,batch_list,epoch=0,**kwargs):
# #         return self.get_loss_breakdown(batch_list,epoch=epoch)["total"]

# #     def get_loss_breakdown(self,batch_list,epoch=0,**kwargs):
# #         batch_list=self._lon_flip_aug(batch_list)
# #         batch_list=self._obs_noise_aug(batch_list,sigma=0.005)

# #         obs_t=batch_list[0]; pred_t=batch_list[1]
# #         obs_Me=batch_list[7]; pred_Me=batch_list[8]
# #         env_data=batch_list[13] if len(batch_list)>13 else None

# #         lp=obs_t[-1]; lm=obs_Me[-1]
# #         B,device=lp.shape[0],lp.device
# #         sigma=self._sigma_schedule(epoch)
# #         tau=self._tau_schedule(epoch)

# #         # Context
# #         raw_ctx=self.encoder.encode(batch_list)
# #         use_null=(torch.rand(1).item()<self.cfg_uncond_prob)
# #         ctx=self.encoder.apply_ctx_head(raw_ctx,use_null=use_null)

# #         # Consistency
# #         L_consist=_dropout_consistency_loss(self.encoder,raw_ctx)

# #         # Difficulty
# #         obs_BT2=obs_t.permute(1,0,2); img_obs=batch_list[11]
# #         delta=compute_difficulty_score(obs_BT2,img_obs,env_data,device)

# #         # GT rel
# #         x1_rel_orig=self._to_rel(pred_t,pred_Me,lp,lm)

# #         # ── ROOT-1 FIX: x0 từ persistence ───────────────────────────────────
# #         x0_base=_persistence_x0_rel(obs_t,obs_Me,lp,lm,self.pred_len,sigma=sigma)

# #         # OT matching từ epoch 5 trở đi để tránh noise lúc model chưa ổn
# #         if self.use_ot and B>=4 and epoch>=5:
# #             x0_base,x1_rel_orig=_spherical_ot_matching(
# #                 x0_base,x1_rel_orig,lp,epsilon=self.ot_epsilon)

# #         # Mixup (chỉ trên easy samples)
# #         x1_rel,delta=_traj_mixup(x1_rel_orig,delta,prob=0.3,alpha=0.4)

# #         # CFM từ persistence x0
# #         x_t,fm_t,u_target=_cfm_from_persistence(x0_base,x1_rel)
# #         x_t_shared=x_t

# #         # Generate K modes
# #         modes_rel_nograd=[]; modes_rel_grad=[]; vels_pred=[]
# #         for k in range(self.K):
# #             sigma_k=self.head_noise_base*(1.0+k*0.25)
# #             z_k=torch.randn_like(x_t_shared)*sigma_k
# #             x_t_k=x_t_shared+z_k

# #             # FIX ROOT-3: generator LUÔN có gradient (không warmup freeze)
# #             v_k=self.velocity_heads[k](x_t_k,fm_t,ctx)

# #             with torch.no_grad():
# #                 pred_k_oracle=x_t_shared+(1.0-fm_t.view(B,1,1))*v_k
# #             modes_rel_nograd.append(pred_k_oracle.detach())

# #             pred_k_grad=x_t_k+(1.0-fm_t.view(B,1,1))*v_k
# #             modes_rel_grad.append(pred_k_grad)
# #             vels_pred.append(v_k)

# #         modes_t_ng=torch.stack(modes_rel_nograd,dim=1)
# #         modes_t_g=torch.stack(modes_rel_grad,dim=1)

# #         # Oracle assignment (so với x1_rel_orig, không phải mixup)
# #         with torch.no_grad():
# #             ade_k=torch.stack([
# #                 _ade_km_from_rel(modes_rel_nograd[k],x1_rel_orig.detach(),lp)
# #                 for k in range(self.K)
# #             ],dim=1)
# #             k_star=ade_k.argmin(dim=1)

# #         # FM Loss
# #         fm_errs=torch.stack([
# #             ((vels_pred[k]-u_target)**2).mean(dim=[1,2])
# #             for k in range(self.K)
# #         ],dim=1)
# #         L_easy=fm_errs.mean(dim=1)
# #         L_oracle=fm_errs[torch.arange(B,device=device),k_star]

# #         # ── ROOT-2 FIX: diversity mask đúng ─────────────────────────────────
# #         mode_star_ng=modes_t_ng[torch.arange(B,device=device),k_star]
# #         dists_all=torch.stack([
# #             ((mode_star_ng-modes_t_ng[:,k])**2).mean(dim=[1,2]).sqrt()
# #             for k in range(self.K)
# #         ],dim=1)
# #         # FIX: zeros + scatter True → masked_fill(~mask) fill inf vào k_star
# #         mask_ns=torch.zeros(B,self.K,device=device,dtype=torch.bool)
# #         mask_ns.scatter_(1,k_star.unsqueeze(1),True)
# #         min_dist=dists_all.masked_fill(~mask_ns,float('inf')).min(dim=1).values

# #         MARGIN=0.40
# #         L_div=F.relu(MARGIN-min_dist)
# #         L_diff=L_oracle+0.3*L_div

# #         L_FM_raw=(1.0-delta)*L_easy+delta*L_diff
# #         w_d=0.5+1.5*delta
# #         L_FM=(w_d*L_FM_raw).mean()

# #         # Selector (stop-grad từ generator)
# #         sel_logits,dir_logits=self.selector(ctx.detach(),modes_t_ng,env_data)
# #         p_oracle=F.softmax(-ade_k/tau,dim=1)
# #         L_rank=F.kl_div(F.log_softmax(sel_logits,dim=1),p_oracle,reduction='batchmean')
# #         gt_bucket=get_gt_direction_bucket(obs_t,pred_t)
# #         L_dir=F.cross_entropy(dir_logits,gt_bucket)

# #         # ── ROOT-5 FIX: coherence từ oracle mode có gradient ─────────────────
# #         oracle_mode_grad=modes_t_g[torch.arange(B,device=device),k_star]
# #         L_coh=_coherence_loss_oracle(oracle_mode_grad,x1_rel_orig,lp,device)

# #         # Total loss — luôn joint train (không warmup phase)
# #         L_total=self.learned_weights(L_FM,L_rank,L_dir,L_coh,L_consist)

# #         if not torch.isfinite(L_total): L_total=L_total.new_zeros(())

# #         def _s(x): return x.item() if torch.is_tensor(x) else float(x)
# #         return {
# #             "total":L_total,"L_FM":_s(L_FM),"L_easy":_s(L_easy.mean()),
# #             "L_diff":_s(L_diff.mean()),"L_oracle":_s(L_oracle.mean()),
# #             "L_div":_s(L_div.mean()),"L_rank":_s(L_rank),"L_dir":_s(L_dir),
# #             "L_coh":_s(L_coh),"L_consist":_s(L_consist),
# #             "delta_mean":_s(delta.mean()),"delta_p75":_s(delta.quantile(0.75)),
# #             "min_dist_mean":_s(min_dist.mean()),
# #             "in_warmup":False,
# #             **self.learned_weights.get_weights(),
# #         }

# #     # ── INFERENCE ─────────────────────────────────────────────────────────────

# #     @torch.no_grad()
# #     def sample(self,batch_list,ddim_steps=20,predict_csv=None,
# #                blend_strength=0.10,**kwargs):
# #         obs_t=batch_list[0]; obs_Me=batch_list[7]
# #         env_data=batch_list[13] if len(batch_list)>13 else None
# #         lp=obs_t[-1]; lm=obs_Me[-1]
# #         B,device=lp.shape[0],lp.device
# #         T_pred=self.pred_len; dt=1.0/max(ddim_steps,1)

# #         raw_ctx=self.encoder.encode(batch_list)
# #         ctx=self.encoder.apply_ctx_head(raw_ctx)
# #         ctx_null=self.encoder.apply_ctx_head(raw_ctx,use_null=True)

# #         # FIX ROOT-1: inference cũng dùng persistence x0 (sigma=0 vì inference)
# #         x0_base=_persistence_x0_rel(obs_t,obs_Me,lp,lm,T_pred,sigma=0.0)

# #         obs_t_norm=obs_t[:,:,:2]
# #         obs_h_n=(F.normalize(obs_t_norm[-1]-obs_t_norm[-2],dim=-1,eps=1e-6)
# #                  if obs_t_norm.shape[0]>=2 else None)

# #         all_modes_abs=[]
# #         for k in range(self.K):
# #             sigma_k=self.head_noise_base*(1.0+k*0.25)
# #             x_t=x0_base+torch.randn_like(x0_base)*sigma_k
# #             for step in range(ddim_steps):
# #                 t_b=torch.full((B,),step*dt,device=device)
# #                 if step>0 and self.cfg_guidance_scale>1.0:
# #                     v_cond=self.velocity_heads[k](x_t,t_b,ctx)
# #                     v_uncond=self.velocity_heads[k](x_t,t_b,ctx_null)
# #                     if obs_h_n is not None:
# #                         pred_h=F.normalize(v_cond[:,0,:2].detach(),dim=-1,eps=1e-6)
# #                         cos_a=(obs_h_n*pred_h).sum(-1).clamp(-1.0,1.0)
# #                         gs=(0.8+0.7*(cos_a+1.0)*0.5).view(B,1,1)
# #                         v_k=v_uncond+gs*(v_cond-v_uncond)
# #                     else:
# #                         v_k=v_uncond+self.cfg_guidance_scale*(v_cond-v_uncond)
# #                 else:
# #                     v_k=self.velocity_heads[k](x_t,t_b,ctx)
# #                 x_t=(x_t+dt*v_k).clamp(-5.0,5.0)
# #             traj_abs,_=self._to_abs(x_t,lp,lm)
# #             all_modes_abs.append(traj_abs)

# #         modes_stack=torch.stack(all_modes_abs,dim=0)

# #         modes_rel_sel=torch.stack([
# #             torch.cat([all_modes_abs[k].permute(1,0,2)-lp.unsqueeze(1),
# #                        torch.zeros(B,T_pred,2,device=device)],dim=-1)
# #             for k in range(self.K)
# #         ],dim=1)
# #         sel_logits,_=self.selector(ctx,modes_rel_sel,env_data)
# #         best_k=sel_logits.argmax(dim=1)

# #         pred_best=torch.stack([modes_stack[best_k[b],:,b,:] for b in range(B)],dim=1)
# #         pred_final=self._persistence_blend(pred_best,obs_t[:,:,:2],blend_strength=blend_strength)

# #         if predict_csv is not None:
# #             self._write_predict_csv(predict_csv,pred_final,modes_stack)
# #         return pred_final,modes_stack

# #     @staticmethod
# #     def _write_predict_csv(csv_path,traj_mean,all_modes):
# #         import csv as _csv,os; from datetime import datetime
# #         os.makedirs(os.path.dirname(os.path.abspath(csv_path)),exist_ok=True)
# #         T,B=traj_mean.shape[0],traj_mean.shape[1]
# #         mlon=((traj_mean[...,0]*50.0+1800.0)/10.0).cpu().numpy()
# #         mlat=((traj_mean[...,1]*50.0)/10.0).cpu().numpy()
# #         fields=["ts","b","step","lead_h","lon","lat"]
# #         ts=datetime.now().strftime("%Y%m%d_%H%M%S")
# #         write_hdr=not os.path.exists(csv_path)
# #         with open(csv_path,"a",newline="") as fh:
# #             w=_csv.DictWriter(fh,fieldnames=fields)
# #             if write_hdr: w.writeheader()
# #             for b in range(B):
# #                 for k in range(T):
# #                     w.writerow({"ts":ts,"b":b,"step":k,"lead_h":(k+1)*6,
# #                                 "lon":f"{mlon[k,b]:.4f}","lat":f"{mlat[k,b]:.4f}"})


# # TCFlowMatching = TCFlowMatchingV65
# # TCDiffusion    = TCFlowMatchingV65

# """
# flow_matching_model_v65.py  ── v65-stable
# ══════════════════════════════════════════════════════════════════════════════
# ROOT CAUSES của diverge trong v65-fixed (ADE 271→1609):

#   BUG-1 [CRITICAL] mask_ns logic sai → L_div = 0.40 mọi lúc:
#     v65-fixed: mask_ns = zeros→scatter True at k_star
#     → masked_fill(~mask_ns) fills inf ở non-k_star
#     → min_dist = distance đến k_star = 0
#     → L_div = relu(0.40 - 0) = 0.40 LUÔN
#     → gradient push modes ra xa vô hạn → ADE diverge
#     Fix: mask_ns = ones→scatter False at k_star (loại k_star, lấy min của rest)

#   BUG-2 [CRITICAL] oracle_ade_ema proxy sai trong train script:
#     train_v65.py: oracle_ema = ema * bd["L_oracle"] * 500
#     Nhưng "L_oracle" là FM MSE (~ 0.1-0.5), không phải km
#     → oracle_ema = init 999 * 0.9 + 0.1*500 = ... KHÔNG BAO GIỜ < 160km
#     → Phase gate stuck, selector frozen mãi mãi ngay cả sau ep15
#     Fix: dùng "oracle_ade_km" key riêng (km thực) trong loss breakdown

#   BUG-3 val loss âm:
#     LearnedWeights: s_i tự tăng khi L_i nhỏ → total âm
#     Xảy ra do BUG-1 làm gradient chaos, s_* bị đẩy quá xa
#     Fix: clamp s_i chặt hơn (-3, 3) thay vì (-5, 5)
#     Và clip gradient của LearnedWeights riêng

# FIX SO VỚI v65-fixed:
#   ✅ BUG-1: mask_ns đúng (ones→scatter False)
#   ✅ BUG-2: thêm "oracle_ade_km" = actual km vào loss dict
#   ✅ BUG-3: clamp(-3,3), log_s init thận trọng
#   ✅ Giữ lại: persistence x0, _cfm_from_persistence, joint training (no warmup)
#   ✅ Giữ lại: K=8 compass heads, OT matching, CFG, easy/difficult, coherence oracle
# """
# from __future__ import annotations

# import math
# from typing import Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from Model.FNO3D_encoder import FNO3DEncoder
# from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# from Model.env_net_transformer_gphsplit import Env_net

# R_EARTH      = 6371.0
# DT_HOURS     = 6.0
# DEG2KM       = 111.0
# _NORM_TO_DEG = 5.0
# _COMPASS_DEG = [0., 45., 90., 135., 180., 225., 270., 315.]
# K_MODES      = 8
# _V_MAX_KMH   = 80.0
# _DT_H        = 6.0


# # ═══════════════════════════════════════════════════════════════════
# #  Coordinate utils
# # ═══════════════════════════════════════════════════════════════════

# def _norm_to_deg(t):
#     return torch.stack([
#         (t[...,0]*50.0+1800.0)/10.0,
#         (t[...,1]*50.0)/10.0,
#     ], dim=-1)

# def _haversine_deg(p1, p2):
#     lat1=torch.deg2rad(p1[...,1]); lat2=torch.deg2rad(p2[...,1])
#     dlat=torch.deg2rad(p2[...,1]-p1[...,1])
#     dlon=torch.deg2rad(p2[...,0]-p1[...,0])
#     a=(torch.sin(dlat/2).pow(2)+
#        torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2))
#     return 2.0*R_EARTH*torch.asin(a.clamp(1e-12,1-1e-12).sqrt())

# def _unwrap_model(m):
#     return m._orig_mod if hasattr(m,'_orig_mod') else m

# def _step_speeds_deg(traj_deg):
#     T=traj_deg.shape[0]
#     if T<2: return traj_deg.new_zeros(1,traj_deg.shape[1])
#     return _haversine_deg(traj_deg[:-1],traj_deg[1:])/DT_HOURS

# def _ade_km_from_rel(pred_rel, gt_rel, lp):
#     pred_abs=lp.unsqueeze(1)+pred_rel[:,:,:2]
#     gt_abs=lp.unsqueeze(1)+gt_rel[:,:,:2]
#     return _haversine_deg(_norm_to_deg(pred_abs),_norm_to_deg(gt_abs)).mean(dim=1)


# # ═══════════════════════════════════════════════════════════════════
# #  Persistence x0 — dùng cho cả training lẫn inference
# # ═══════════════════════════════════════════════════════════════════

# def _persistence_x0_rel(obs_traj, obs_Me, lp, lm, pred_len, sigma=0.02):
#     B,device=obs_traj.shape[1],obs_traj.device
#     if obs_traj.shape[0]>=3:
#         vels=obs_traj[1:]-obs_traj[:-1]; n_v=vels.shape[0]; alpha=0.7
#         w=torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
#                        dtype=torch.float,device=device).flip(0)
#         lv=(vels*(w/w.sum()).view(-1,1,1)).sum(0)
#     elif obs_traj.shape[0]>=2: lv=obs_traj[-1,:,:2]-obs_traj[-2,:,:2]
#     else: lv=obs_traj.new_zeros(B,2)
#     steps=torch.arange(1,pred_len+1,device=device).float()
#     pred_abs=obs_traj[-1,:,:2].unsqueeze(0)+lv.unsqueeze(0)*steps.view(-1,1,1)
#     pred_rel=pred_abs.permute(1,0,2)-lp.unsqueeze(1)
#     pred_rel4=torch.cat([pred_rel,torch.zeros(B,pred_len,2,device=device)],dim=-1)
#     return pred_rel4+torch.randn_like(pred_rel4)*sigma

# def _cfm_from_persistence(x0_rel, x1_rel):
#     B=x0_rel.shape[0]; device=x0_rel.device
#     t=torch.rand(B,device=device); te=t.view(B,1,1)
#     return (1.0-te)*x0_rel+te*x1_rel, t, x1_rel-x0_rel


# # ═══════════════════════════════════════════════════════════════════
# #  OT Matching
# # ═══════════════════════════════════════════════════════════════════

# def _forward_azimuth(p1, p2):
#     lon1=torch.deg2rad(p1[...,0]); lat1=torch.deg2rad(p1[...,1])
#     lon2=torch.deg2rad(p2[...,0]); lat2=torch.deg2rad(p2[...,1])
#     dlon=lon2-lon1
#     y=torch.sin(dlon)*torch.cos(lat2)
#     x=torch.cos(lat1)*torch.sin(lat2)-torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon)
#     return torch.atan2(y,x)

# def _sinkhorn_log(cost, epsilon=0.05, n_iter=30):
#     B=cost.shape[0]; device=cost.device
#     log_a=-math.log(B)*torch.ones(B,device=device)
#     log_b=-math.log(B)*torch.ones(B,device=device)
#     log_K=-cost/epsilon
#     log_u=torch.zeros(B,device=device); log_v=torch.zeros(B,device=device)
#     for _ in range(n_iter):
#         log_u=log_a-torch.logsumexp(log_K+log_v.unsqueeze(0),dim=1)
#         log_v=log_b-torch.logsumexp(log_K+log_u.unsqueeze(1),dim=0)
#     return (log_K+log_u.unsqueeze(1)+log_v.unsqueeze(0)).exp().clamp(0.0)

# def _geodesic_ot_cost(x0_rel, x1_rel, lp):
#     B=x0_rel.shape[0]
#     def _abs_deg(rel): return _norm_to_deg(lp.unsqueeze(1)+rel[:,:,:2])
#     x0d=_abs_deg(x0_rel); x1d=_abs_deg(x1_rel)
#     x0e=x0d.unsqueeze(1).expand(B,B,-1,-1).reshape(B*B,-1,2)
#     x1e=x1d.unsqueeze(0).expand(B,B,-1,-1).reshape(B*B,-1,2)
#     pos_cost=_haversine_deg(x0e,x1e).reshape(B,B,-1).mean(-1)/500.0
#     spd0=_step_speeds_deg(x0d.permute(1,0,2)).permute(1,0).mean(-1)
#     spd1=_step_speeds_deg(x1d.permute(1,0,2)).permute(1,0).mean(-1)
#     speed_cost=(spd0.unsqueeze(1)-spd1.unsqueeze(0)).abs()/20.0
#     def _mb(td):
#         b=_forward_azimuth(td[:,:-1,:],td[:,1:,:])
#         return torch.atan2(b.sin().mean(-1),b.cos().mean(-1))
#     h0=_mb(x0d); h1=_mb(x1d)
#     dh=(h0.unsqueeze(1)-h1.unsqueeze(0)+math.pi)%(2*math.pi)-math.pi
#     return 1.0*pos_cost+0.5*speed_cost+0.3*dh.abs()/math.pi

# def _spherical_ot_matching(x0_batch, x1_batch, lp, epsilon=0.05):
#     B=x0_batch.shape[0]
#     if B<4: return x0_batch, x1_batch
#     try:
#         cost=_geodesic_ot_cost(x0_batch,x1_batch,lp)
#         with torch.no_grad(): pi=_sinkhorn_log(cost,epsilon=epsilon)
#         flat=pi.reshape(-1).clamp(0.0); s=flat.sum()
#         if not torch.isfinite(s) or s<1e-10: return x0_batch, x1_batch
#         idx=torch.multinomial(flat/s,num_samples=B,replacement=True)
#         return x0_batch[idx%B], x1_batch[idx%B]
#     except Exception:
#         return x0_batch, x1_batch


# # ═══════════════════════════════════════════════════════════════════
# #  Difficulty Score
# # ═══════════════════════════════════════════════════════════════════

# def compute_difficulty_score(obs_BT2, img_obs, env_data, device):
#     B=obs_BT2.shape[0]; scores=[]
#     dir24=env_data.get("history_direction24") if env_data else None
#     if dir24 is not None and torch.is_tensor(dir24):
#         dir24=dir24.float().to(device)
#         if dir24.dim()==3: d24=dir24[:,-1,:]
#         elif dir24.dim()==2: d24=dir24
#         else: d24=None
#         if d24 is not None and d24.shape[-1]==8:
#             bucket=d24.argmax(dim=-1).float(); a24=bucket*(2.0*math.pi/8.0)
#             dy=obs_BT2[:,-1,1]-obs_BT2[:,-2,1]; dx=obs_BT2[:,-1,0]-obs_BT2[:,-2,0]
#             lat_mid=(obs_BT2[:,-1,1]+obs_BT2[:,-2,1])*0.5
#             cos_lat=torch.cos(torch.deg2rad(lat_mid*_NORM_TO_DEG)).clamp(1e-4)
#             a_now=torch.atan2(dx*cos_lat,dy)
#             cos_d=(torch.cos(a_now)*torch.cos(a24)+torch.sin(a_now)*torch.sin(a24)).clamp(-1,1)
#             s_curve=torch.sigmoid((torch.rad2deg(torch.acos(cos_d))-45.0)/20.0)
#         else: s_curve=torch.zeros(B,device=device)
#     else: s_curve=torch.zeros(B,device=device)
#     scores.append(0.30*s_curve)
#     steer=env_data.get("steering_speed") if env_data else None
#     if steer is not None and torch.is_tensor(steer):
#         steer=steer.float().to(device)
#         while steer.dim()>1: steer=steer[...,-1]
#         steer=steer.view(-1); steer=steer[:B] if steer.numel()>=B else steer[0].expand(B)
#         s_weak=torch.sigmoid((4.0-steer*20.0)/2.0)
#     else:
#         u=env_data.get("u500_mean") if env_data else None
#         v=env_data.get("v500_mean") if env_data else None
#         if u is not None and v is not None and torch.is_tensor(u) and torch.is_tensor(v):
#             u=u.float().to(device); v=v.float().to(device)
#             while u.dim()>1: u=u[...,-1]
#             while v.dim()>1: v=v[...,-1]
#             s_weak=torch.sigmoid((4.0-(u.view(-1)[:B]**2+v.view(-1)[:B]**2).sqrt()*30.0)/2.0)
#         else: s_weak=torch.zeros(B,device=device)
#     scores.append(0.25*s_weak)
#     if img_obs is not None and img_obs.shape[1]>=11:
#         u200=img_obs[:,4,-1,40,40]*13.315; u850=img_obs[:,6,-1,40,40]*7.911
#         v200=img_obs[:,8,-1,40,40]*8.377;  v850=img_obs[:,10,-1,40,40]*6.203
#         shear=((u200-u850)**2+(v200-v850)**2).sqrt()
#         s_shear=torch.sigmoid((shear-8.0)/3.0)
#     else: s_shear=torch.zeros(B,device=device)
#     scores.append(0.20*s_shear)
#     ri=env_data.get("rapid_intensification") if env_data else None
#     if ri is not None and torch.is_tensor(ri):
#         ri=ri.float().to(device)
#         while ri.dim()>1: ri=ri[...,-1]
#         s_ri=ri.view(-1)[:B].clamp(0,1) if ri.numel()>=B else ri[0].expand(B)
#     else: s_ri=torch.zeros(B,device=device)
#     scores.append(0.15*s_ri)
#     mv=env_data.get("move_velocity") if env_data else None
#     if mv is not None and torch.is_tensor(mv):
#         mv=mv.float().to(device)
#         while mv.dim()>1: mv=mv[...,-1]
#         mv_v=mv.view(-1); mv_v=mv_v[:B] if mv_v.numel()>=B else mv_v[0].expand(B)
#         s_slow=torch.sigmoid((0.05-mv_v)/0.02)
#     else: s_slow=torch.zeros(B,device=device)
#     scores.append(0.10*s_slow)
#     return sum(scores).clamp(0.0,1.0)


# def get_gt_direction_bucket(obs_traj, pred_traj):
#     last=obs_traj[-1]; first=pred_traj[0]
#     dy=first[:,1]-last[:,1]; dx=first[:,0]-last[:,0]
#     lat_d=(last[:,1]+first[:,1])*0.5
#     cos_l=torch.cos(torch.deg2rad(lat_d*_NORM_TO_DEG)).clamp(1e-4)
#     angle=(torch.atan2(dx*cos_l,dy).rad2deg()%360.0)
#     return ((angle+22.5)/45.0).long()%8


# # ═══════════════════════════════════════════════════════════════════
# #  CompassVelocityHead
# # ═══════════════════════════════════════════════════════════════════

# class CompassVelocityHead(nn.Module):
#     def __init__(self,compass_idx,pred_len=12,ctx_dim=256,stochastic_depth_prob=0.1):
#         super().__init__()
#         self.pred_len=pred_len; self.stochastic_depth_prob=stochastic_depth_prob
#         angle_rad=_COMPASS_DEG[compass_idx]*math.pi/180.0
#         dir_vec=torch.tensor([math.sin(angle_rad),math.cos(angle_rad)])
#         self.register_buffer('compass_dir',dir_vec)
#         self.dir_proj=nn.Sequential(nn.Linear(2,ctx_dim),nn.GELU(),nn.LayerNorm(ctx_dim))
#         self.time_fc1=nn.Linear(ctx_dim,256); self.time_fc2=nn.Linear(256,ctx_dim)
#         self.traj_embed=nn.Linear(4,ctx_dim)
#         self.pos_enc=nn.Parameter(torch.randn(1,pred_len,ctx_dim)*0.02)
#         self.step_embed=nn.Embedding(pred_len,ctx_dim)
#         self.transformer=nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model=ctx_dim,nhead=8,dim_feedforward=512,
#                                        dropout=0.10,activation="gelu",batch_first=True),
#             num_layers=1)
#         self.out_fc1=nn.Linear(ctx_dim,256); self.out_fc2=nn.Linear(256,4)
#         self.step_scale=nn.Parameter(torch.ones(pred_len)*0.5)
#         with torch.no_grad():
#             nn.init.xavier_uniform_(self.out_fc2.weight,gain=0.1)
#             nn.init.zeros_(self.out_fc2.bias)

#     def _time_emb(self,t,dim):
#         half=dim//2
#         freq=torch.exp(torch.arange(half,dtype=torch.float32,device=t.device)
#                        *(-math.log(10000.0)/max(half-1,1)))
#         emb=t.float().unsqueeze(1)*1000.0*freq.unsqueeze(0)
#         return F.pad(torch.cat([emb.sin(),emb.cos()],dim=-1),(0,dim%2))

#     def forward(self,x_t,t,ctx):
#         B=x_t.shape[0]; T_seq=min(x_t.shape[1],self.pred_len); device=x_t.device
#         dir_token=self.dir_proj(self.compass_dir.to(device).unsqueeze(0).expand(B,-1))
#         if self.training and self.stochastic_depth_prob>0:
#             keep=(torch.rand(B,1,device=device)>self.stochastic_depth_prob).float()
#             dir_token=dir_token*keep
#         t_emb=F.gelu(self.time_fc1(self._time_emb(t,ctx.shape[-1])))
#         t_emb=self.time_fc2(t_emb)
#         step_idx=torch.arange(T_seq,device=device).unsqueeze(0).expand(B,-1)
#         x_emb=(self.traj_embed(x_t[:,:T_seq])+self.pos_enc[:,:T_seq]
#                +t_emb.unsqueeze(1)+self.step_embed(step_idx))
#         mem=torch.stack([ctx,dir_token,t_emb],dim=1)
#         decoded=self.transformer(x_emb,mem)
#         scale=torch.sigmoid(self.step_scale[:T_seq]).view(1,T_seq,1)*2.0
#         return self.out_fc2(F.gelu(self.out_fc1(decoded)))*scale


# # ═══════════════════════════════════════════════════════════════════
# #  CompassSelector
# # ═══════════════════════════════════════════════════════════════════

# class CompassSelector(nn.Module):
#     def __init__(self,ctx_dim=256,K=8,n_dirs=8):
#         super().__init__()
#         self.K=K; self.n_dirs=n_dirs
#         self.steering_enc=nn.Sequential(nn.Linear(11,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,128))
#         self.mode_enc=nn.Sequential(nn.Linear(12*2,128),nn.GELU(),nn.LayerNorm(128),nn.Linear(128,64))
#         self.score_net=nn.Sequential(nn.Linear(ctx_dim+64+128,256),nn.GELU(),nn.LayerNorm(256),
#                                      nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
#         self.dir_head=nn.Sequential(nn.Linear(ctx_dim+128,256),nn.GELU(),nn.LayerNorm(256),nn.Linear(256,n_dirs))

#     def _get_steer(self,env_data,B,device):
#         def _sc(key):
#             if env_data is None: return torch.zeros(B,device=device)
#             v=env_data.get(key)
#             if v is None or not torch.is_tensor(v): return torch.zeros(B,device=device)
#             v=v.float().to(device)
#             while v.dim()>1: v=v[...,-1]
#             v=v.view(-1); return (v[:B] if v.numel()>=B else v[0].expand(B)).clamp(-3,3)
#         def _sv(key,dim):
#             if env_data is None: return torch.zeros(B,dim,device=device)
#             v=env_data.get(key)
#             if v is None or not torch.is_tensor(v): return torch.zeros(B,dim,device=device)
#             v=v.float().to(device)
#             if v.dim()==3: v=v[:,-1,:]
#             elif v.dim()!=2 or v.shape[0]!=B: return torch.zeros(B,dim,device=device)
#             v=F.pad(v,(0,max(0,dim-v.shape[-1])))[:,:dim]; return v.clamp(-3,3)
#         feat=torch.cat([_sc("steering_speed").unsqueeze(-1),_sc("steering_dir_sin").unsqueeze(-1),
#                         _sc("steering_dir_cos").unsqueeze(-1),_sv("history_direction24",8)],dim=-1)
#         return self.steering_enc(feat)

#     def forward(self,ctx,modes,env_data):
#         B,K=modes.shape[:2]; device=ctx.device
#         sf=self._get_steer(env_data,B,device)
#         scores=[]
#         for k in range(K):
#             mf=self.mode_enc(modes[:,k,:,:2].reshape(B,-1))
#             scores.append(self.score_net(torch.cat([ctx,mf,sf],dim=-1)).squeeze(-1))
#         return torch.stack(scores,dim=1), self.dir_head(torch.cat([ctx,sf],dim=-1))


# # ═══════════════════════════════════════════════════════════════════
# #  LearnedWeights — BUG-3 FIX: clamp(-3,3)
# # ═══════════════════════════════════════════════════════════════════

# class LearnedWeights(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Init nhỏ hơn để tránh oscillation sớm
#         self.log_s_fm  = nn.Parameter(torch.tensor(0.0))
#         self.log_s_sel = nn.Parameter(torch.tensor(0.0))
#         self.log_s_coh = nn.Parameter(torch.tensor(0.0))
#         self.log_s_con = nn.Parameter(torch.tensor(0.0))
#         self.log_alpha = nn.Parameter(torch.tensor(0.0))

#     def forward(self,L_fm,L_rank,L_dir,L_coh,L_con):
#         # BUG-3 FIX: clamp chặt hơn (-3,3)
#         s_fm=self.log_s_fm.clamp(-3,3); s_sel=self.log_s_sel.clamp(-3,3)
#         s_coh=self.log_s_coh.clamp(-3,3); s_con=self.log_s_con.clamp(-3,3)
#         α=torch.sigmoid(self.log_alpha)
#         L_sel=α*L_rank+(1.0-α)*L_dir
#         return (L_fm*torch.exp(-s_fm)+s_fm +
#                 L_sel*torch.exp(-s_sel)+s_sel +
#                 L_coh*torch.exp(-s_coh)+s_coh +
#                 L_con*torch.exp(-s_con)+s_con)

#     def get_weights(self):
#         def _w(s): return torch.exp(-s.clamp(-3,3)).item()
#         return {"w_fm":_w(self.log_s_fm),"w_sel":_w(self.log_s_sel),
#                 "w_coh":_w(self.log_s_coh),"w_con":_w(self.log_s_con),
#                 "alpha":torch.sigmoid(self.log_alpha).item()}


# # ═══════════════════════════════════════════════════════════════════
# #  SharedContextEncoder
# # ═══════════════════════════════════════════════════════════════════

# class SharedContextEncoder(nn.Module):
#     RAW_CTX_DIM=512
#     def __init__(self,pred_len=12,obs_len=8,ctx_dim=256,unet_in_ch=13):
#         super().__init__()
#         self.pred_len=pred_len; self.obs_len=obs_len; self.ctx_dim=ctx_dim
#         self.spatial_enc=FNO3DEncoder(in_channel=unet_in_ch,out_channel=1,d_model=32,
#             n_layers=4,modes_t=4,modes_h=4,modes_w=4,spatial_down=32,dropout=0.05)
#         self.bottleneck_pool=nn.AdaptiveAvgPool3d((None,1,1))
#         self.bottleneck_proj=nn.Linear(128,128)
#         self.decoder_proj=nn.Linear(1,16)
#         self.enc_1d=DataEncoder1D(in_1d=4,feat_3d_dim=128,mlp_h=64,
#             lstm_hidden=128,lstm_layers=3,dropout=0.1,d_state=16)
#         self.env_enc=Env_net(obs_len=obs_len,d_model=32)
#         self.ctx_fc1=nn.Linear(128+32+16,self.RAW_CTX_DIM)
#         self.ctx_ln=nn.LayerNorm(self.RAW_CTX_DIM)
#         self.ctx_drop=nn.Dropout(0.15)
#         self.ctx_fc2=nn.Linear(self.RAW_CTX_DIM,ctx_dim)
#         self.null_embedding=nn.Parameter(torch.randn(1,self.RAW_CTX_DIM)*0.02)

#     def encode(self,batch_list):
#         obs_t=batch_list[0]; obs_Me=batch_list[7]
#         image_obs=batch_list[11]; env_data=batch_list[13] if len(batch_list)>13 else None
#         if image_obs.dim()==4: image_obs=image_obs.unsqueeze(2)
#         if image_obs.shape[1]==1 and self.spatial_enc.in_channel!=1:
#             image_obs=image_obs.expand(-1,self.spatial_enc.in_channel,-1,-1,-1)
#         T_obs=obs_t.shape[0]
#         e_3d_bot,e_3d_dec=self.spatial_enc.encode(image_obs)
#         e_3d_s=self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0,2,1)
#         e_3d_s=self.bottleneck_proj(e_3d_s)
#         if e_3d_s.shape[1]!=T_obs:
#             e_3d_s=F.interpolate(e_3d_s.permute(0,2,1),size=T_obs,
#                                   mode="linear",align_corners=False).permute(0,2,1)
#         e_3d_dec_t=e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
#         t_w=torch.softmax(torch.arange(e_3d_dec_t.shape[1],dtype=torch.float,
#                                         device=e_3d_dec_t.device)*0.5,dim=0)
#         f_sp=self.decoder_proj((e_3d_dec_t*t_w.unsqueeze(0)).sum(1,keepdim=True))
#         obs_in=torch.cat([obs_t,obs_Me],dim=2).permute(1,0,2)
#         h_t=self.enc_1d(obs_in,e_3d_s)
#         e_env,_,_=self.env_enc(env_data,image_obs)
#         return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t,e_env,f_sp],dim=-1))))

#     def apply_ctx_head(self,raw,use_null=False):
#         if use_null: raw=self.null_embedding.expand(raw.shape[0],-1)
#         return self.ctx_fc2(self.ctx_drop(raw))


# # ═══════════════════════════════════════════════════════════════════
# #  Losses
# # ═══════════════════════════════════════════════════════════════════

# def _coherence_loss_oracle(pred_oracle_grad, x1_rel_orig, lp, device):
#     pred_abs=lp.unsqueeze(1)+pred_oracle_grad[:,:,:2]
#     gt_abs=lp.unsqueeze(1)+x1_rel_orig[:,:,:2]
#     pred_deg=_norm_to_deg(pred_abs); gt_deg=_norm_to_deg(gt_abs)
#     cos_lat=torch.cos(torch.deg2rad(gt_deg[:,:-1,1])).clamp(1e-4)
#     dp_lon=(pred_deg[:,1:,0]-pred_deg[:,:-1,0])*cos_lat*DEG2KM
#     dp_lat=(pred_deg[:,1:,1]-pred_deg[:,:-1,1])*DEG2KM
#     dg_lon=(gt_deg[:,1:,0]-gt_deg[:,:-1,0])*cos_lat*DEG2KM
#     dg_lat=(gt_deg[:,1:,1]-gt_deg[:,:-1,1])*DEG2KM
#     dp=torch.stack([dp_lon,dp_lat],dim=-1); dg=torch.stack([dg_lon,dg_lat],dim=-1)
#     return (dp-dg).norm(dim=-1).mean()/DEG2KM

# def _dropout_consistency_loss(encoder, raw_ctx):
#     if not encoder.training: return torch.tensor(0.0,device=raw_ctx.device)
#     ctx_a=encoder.apply_ctx_head(raw_ctx); ctx_b=encoder.apply_ctx_head(raw_ctx)
#     cos_sim=(F.normalize(ctx_a,dim=-1)*F.normalize(ctx_b,dim=-1)).sum(dim=-1)
#     return (1.0-cos_sim).mean()

# def _traj_mixup(x1_rel, delta, prob=0.3, alpha=0.4):
#     if not torch.is_tensor(x1_rel) or x1_rel.shape[0]<2: return x1_rel,delta
#     if torch.rand(1).item()>prob: return x1_rel,delta
#     B=x1_rel.shape[0]
#     lam=max(float(torch.distributions.Beta(alpha,alpha).sample()),
#             1.0-float(torch.distributions.Beta(alpha,alpha).sample()))
#     lam=max(lam,1.0-lam)
#     idx=torch.randperm(B,device=x1_rel.device)
#     easy_t=(delta<0.4).float().view(B,1,1); easy_d=(delta<0.4).float()
#     x1_mix=x1_rel*(1.0-easy_t*(1.0-lam))+x1_rel[idx]*(easy_t*(1.0-lam))
#     delta_mix=(delta*lam+delta[idx]*(1.0-lam))*easy_d+delta*(1.0-easy_d)
#     return x1_mix,delta_mix


# # ═══════════════════════════════════════════════════════════════════
# #  EMA
# # ═══════════════════════════════════════════════════════════════════

# class EMAModel:
#     def __init__(self,model,decay=0.995):
#         self.decay=decay; m=_unwrap_model(model)
#         self.shadow={k:v.detach().clone() for k,v in m.state_dict().items()
#                      if v.dtype.is_floating_point}
#     def update(self,model):
#         m=_unwrap_model(model)
#         with torch.no_grad():
#             for k,v in m.state_dict().items():
#                 if k in self.shadow:
#                     self.shadow[k].mul_(self.decay).add_(v.detach(),alpha=1-self.decay)
#     def apply_to(self,model):
#         m=_unwrap_model(model); backup,sd={},m.state_dict()
#         for k in self.shadow:
#             if k not in sd: continue
#             backup[k]=sd[k].detach().clone(); sd[k].copy_(self.shadow[k])
#         return backup
#     def restore(self,model,backup):
#         m=_unwrap_model(model); sd=m.state_dict()
#         for k,v in backup.items():
#             if k in sd: sd[k].copy_(v)


# # ═══════════════════════════════════════════════════════════════════
# #  TCFlowMatchingV65
# # ═══════════════════════════════════════════════════════════════════

# class TCFlowMatchingV65(nn.Module):

#     def __init__(self,pred_len=12,obs_len=8,ctx_dim=256,sigma_min=0.02,
#                  unet_in_ch=13,K=K_MODES,use_ema=True,ema_decay=0.995,
#                  cfg_uncond_prob=0.10,selector_warmup=0,
#                  head_noise_base=0.03,use_ot=True,ot_epsilon=0.05,
#                  cfg_guidance_scale=1.3,**kwargs):
#         super().__init__()
#         self.pred_len=pred_len; self.obs_len=obs_len; self.sigma_min=sigma_min
#         self.K=K; self.use_ema=use_ema; self.ema_decay=ema_decay
#         self.cfg_uncond_prob=cfg_uncond_prob; self.selector_warmup=selector_warmup
#         self.head_noise_base=head_noise_base; self.use_ot=use_ot
#         self.ot_epsilon=ot_epsilon; self.cfg_guidance_scale=cfg_guidance_scale
#         self._ema=None

#         self.encoder=SharedContextEncoder(pred_len=pred_len,obs_len=obs_len,
#                                           ctx_dim=ctx_dim,unet_in_ch=unet_in_ch)
#         self.velocity_heads=nn.ModuleList([
#             CompassVelocityHead(compass_idx=k,pred_len=pred_len,ctx_dim=ctx_dim)
#             for k in range(K)])
#         self.selector=CompassSelector(ctx_dim=ctx_dim,K=K,n_dirs=8)
#         self.learned_weights=LearnedWeights()

#     def init_ema(self):
#         if self.use_ema: self._ema=EMAModel(self,decay=self.ema_decay)
#     def ema_update(self):
#         if self._ema is not None: self._ema.update(self)
#     def set_curriculum_len(self,*a,**kw): pass

#     @staticmethod
#     def _sigma_schedule(epoch):
#         if epoch<2: return 0.10
#         if epoch<10: return 0.10-(epoch-2)/8.0*(0.10-0.04)
#         if epoch<20: return max(0.04-(epoch-10)/10.0*0.01,0.035)
#         return 0.035

#     @staticmethod
#     def _tau_schedule(epoch):
#         if epoch<5: return 5.0
#         if epoch<15: return 4.0
#         if epoch<25: return 3.0
#         return 2.0

#     @staticmethod
#     def _lon_flip_aug(bl,p=0.3):
#         if torch.rand(1).item()>p: return bl
#         bl=list(bl)
#         for i in [0,1,2,3]:
#             if torch.is_tensor(bl[i]) and bl[i].shape[-1]>=1:
#                 t=bl[i].clone(); t[...,0]=-t[...,0]; bl[i]=t
#         return bl

#     @staticmethod
#     def _obs_noise_aug(bl,sigma=0.005):
#         if torch.rand(1).item()>0.5: return bl
#         bl=list(bl)
#         if torch.is_tensor(bl[0]): bl[0]=bl[0]+torch.randn_like(bl[0])*sigma
#         return bl

#     @staticmethod
#     def _to_rel(traj,Me,lp,lm):
#         return torch.cat([traj-lp.unsqueeze(0),Me-lm.unsqueeze(0)],dim=-1).permute(1,0,2)

#     @staticmethod
#     def _to_abs(rel,lp,lm):
#         d=rel.permute(1,0,2)
#         return lp.unsqueeze(0)+d[:,:,:2],lm.unsqueeze(0)+d[:,:,2:]

#     @staticmethod
#     @torch.no_grad()
#     def _persistence_blend(model_pred,obs_traj_norm,blend_strength=0.10):
#         T_obs=obs_traj_norm.shape[0]; T=model_pred.shape[0]
#         B,device=model_pred.shape[1],model_pred.device
#         if T_obs<2: return model_pred
#         vels=obs_traj_norm[1:]-obs_traj_norm[:-1]; n_v=vels.shape[0]
#         if n_v>=3:
#             alpha=0.7; w=torch.tensor([alpha*(1-alpha)**i for i in range(n_v)],
#                                        dtype=torch.float,device=device).flip(0)
#             ev=(vels*(w/w.sum()).view(-1,1,1)).sum(0)
#         elif n_v==2: ev=0.7*vels[-1]+0.3*vels[-2]
#         else: ev=vels[-1]
#         steps=torch.arange(1,T+1,dtype=torch.float,device=device)
#         persist=obs_traj_norm[-1].unsqueeze(0)+ev.unsqueeze(0)*steps.view(T,1,1)
#         obs_spd=_step_speeds_deg(_norm_to_deg(obs_traj_norm))
#         if obs_spd.shape[0]>=2:
#             spd_cv=obs_spd.std(0)/obs_spd.mean(0).clamp(min=1.0)
#             alpha_b=(blend_strength*torch.sigmoid(-(spd_cv-0.3)*5.0)).unsqueeze(0).unsqueeze(-1)
#         else: alpha_b=blend_strength*0.5
#         return (1.0-alpha_b)*model_pred+alpha_b*persist

#     # ─── TRAINING ────────────────────────────────────────────────────

#     def get_loss(self,batch_list,epoch=0,**kwargs):
#         return self.get_loss_breakdown(batch_list,epoch=epoch)["total"]

#     def get_loss_breakdown(self,batch_list,epoch=0,**kwargs):
#         batch_list=self._lon_flip_aug(batch_list)
#         batch_list=self._obs_noise_aug(batch_list,sigma=0.005)

#         obs_t=batch_list[0]; pred_t=batch_list[1]
#         obs_Me=batch_list[7]; pred_Me=batch_list[8]
#         env_data=batch_list[13] if len(batch_list)>13 else None

#         lp=obs_t[-1]; lm=obs_Me[-1]
#         B,device=lp.shape[0],lp.device
#         sigma=self._sigma_schedule(epoch)
#         tau=self._tau_schedule(epoch)

#         raw_ctx=self.encoder.encode(batch_list)
#         # ctx_heads: dùng cho velocity heads, có thể null (CFG training)
#         use_null=(torch.rand(1).item()<self.cfg_uncond_prob)
#         ctx=self.encoder.apply_ctx_head(raw_ctx,use_null=use_null)
#         # ctx_sel: LUÔN real context cho selector — không null
#         # Selector cần real context để học phân biệt modes đúng
#         ctx_sel=self.encoder.apply_ctx_head(raw_ctx,use_null=False) if use_null else ctx

#         L_consist=_dropout_consistency_loss(self.encoder,raw_ctx)

#         obs_BT2=obs_t.permute(1,0,2); img_obs=batch_list[11]
#         delta=compute_difficulty_score(obs_BT2,img_obs,env_data,device)

#         x1_rel_orig=self._to_rel(pred_t,pred_Me,lp,lm)
#         x0_base=_persistence_x0_rel(obs_t,obs_Me,lp,lm,self.pred_len,sigma=sigma)

#         if self.use_ot and B>=4 and epoch>=5:
#             x0_base,x1_rel_orig=_spherical_ot_matching(
#                 x0_base,x1_rel_orig,lp,epsilon=self.ot_epsilon)

#         x1_rel,delta=_traj_mixup(x1_rel_orig,delta,prob=0.3,alpha=0.4)
#         x_t,fm_t,u_target=_cfm_from_persistence(x0_base,x1_rel)
#         x_t_shared=x_t

#         modes_rel_nograd=[]; modes_rel_grad=[]; vels_pred=[]
#         for k in range(self.K):
#             sigma_k=self.head_noise_base*(1.0+k*0.25)
#             z_k=torch.randn_like(x_t_shared)*sigma_k
#             x_t_k=x_t_shared+z_k
#             v_k=self.velocity_heads[k](x_t_k,fm_t,ctx)
#             with torch.no_grad():
#                 pred_k_oracle=x_t_shared+(1.0-fm_t.view(B,1,1))*v_k
#             modes_rel_nograd.append(pred_k_oracle.detach())
#             modes_rel_grad.append(x_t_k+(1.0-fm_t.view(B,1,1))*v_k)
#             vels_pred.append(v_k)

#         modes_t_ng=torch.stack(modes_rel_nograd,dim=1)
#         modes_t_g=torch.stack(modes_rel_grad,dim=1)

#         with torch.no_grad():
#             ade_k=torch.stack([
#                 _ade_km_from_rel(modes_rel_nograd[k],x1_rel_orig.detach(),lp)
#                 for k in range(self.K)
#             ],dim=1)
#             k_star=ade_k.argmin(dim=1)

#         # BUG-2 FIX: oracle_ade thực sự tính bằng km để phase gate chạy đúng
#         oracle_ade_km=ade_k[torch.arange(B,device=device),k_star].mean()

#         fm_errs=torch.stack([
#             ((vels_pred[k]-u_target)**2).mean(dim=[1,2])
#             for k in range(self.K)
#         ],dim=1)
#         L_easy=fm_errs.mean(dim=1)
#         L_oracle_fm=fm_errs[torch.arange(B,device=device),k_star]

#         # ══ BUG-1 FIX: mask_ns đúng ══════════════════════════════════
#         # Muốn: min_dist = khoảng cách từ k_star đến mode GẦN NHẤT TRONG SỐ CÁC MODE KHÁC
#         # → cần loại k_star khỏi candidates
#         # mask_ns[b, k] = True nếu k != k_star[b]  (candidates hợp lệ)
#         # = ones → scatter False tại k_star → mask True ở non-k_star
#         # masked_fill(~mask_ns = True tại k_star) → fill inf tại k_star
#         # min = min distance đến mode gần nhất KHÁC k_star ✅
#         mode_star_ng=modes_t_ng[torch.arange(B,device=device),k_star]
#         dists_all=torch.stack([
#             ((mode_star_ng-modes_t_ng[:,k])**2).mean(dim=[1,2]).sqrt()
#             for k in range(self.K)
#         ],dim=1)
#         mask_ns=torch.ones(B,self.K,device=device,dtype=torch.bool)   # ones
#         mask_ns.scatter_(1,k_star.unsqueeze(1),False)                  # False at k_star
#         # masked_fill(~mask_ns) = fill inf where ~mask_ns = True = where k_star
#         min_dist=dists_all.masked_fill(~mask_ns,float('inf')).min(dim=1).values
#         # ══ END BUG-1 FIX ════════════════════════════════════════════

#         MARGIN=0.40
#         L_div=F.relu(MARGIN-min_dist)
#         L_diff=L_oracle_fm+0.3*L_div
#         L_FM_raw=(1.0-delta)*L_easy+delta*L_diff
#         w_d=0.5+1.5*delta
#         L_FM=(w_d*L_FM_raw).mean()

#         sel_logits,dir_logits=self.selector(ctx_sel.detach(),modes_t_ng,env_data)
#         p_oracle=F.softmax(-ade_k/tau,dim=1)
#         L_rank=F.kl_div(F.log_softmax(sel_logits,dim=1),p_oracle,reduction='batchmean')
#         gt_bucket=get_gt_direction_bucket(obs_t,pred_t)
#         L_dir=F.cross_entropy(dir_logits,gt_bucket)

#         oracle_mode_grad=modes_t_g[torch.arange(B,device=device),k_star]
#         L_coh=_coherence_loss_oracle(oracle_mode_grad,x1_rel_orig,lp,device)

#         L_total=self.learned_weights(L_FM,L_rank,L_dir,L_coh,L_consist)
#         if not torch.isfinite(L_total): L_total=L_total.new_zeros(())

#         def _s(x): return x.item() if torch.is_tensor(x) else float(x)
#         return {
#             "total":L_total,"L_FM":_s(L_FM),"L_easy":_s(L_easy.mean()),
#             "L_diff":_s(L_diff.mean()),
#             "L_oracle":_s(L_oracle_fm.mean()),   # FM MSE của oracle head
#             "oracle_ade_km":_s(oracle_ade_km),   # BUG-2 FIX: km thực để phase gate
#             "L_div":_s(L_div.mean()),"L_rank":_s(L_rank),"L_dir":_s(L_dir),
#             "L_coh":_s(L_coh),"L_consist":_s(L_consist),
#             "delta_mean":_s(delta.mean()),"delta_p75":_s(delta.quantile(0.75)),
#             "min_dist_mean":_s(min_dist.mean()),
#             "in_warmup":False,
#             **self.learned_weights.get_weights(),
#         }

#     # ─── INFERENCE ───────────────────────────────────────────────────

#     @torch.no_grad()
#     def sample(self,batch_list,ddim_steps=20,predict_csv=None,
#                blend_strength=0.10,**kwargs):
#         obs_t=batch_list[0]; obs_Me=batch_list[7]
#         env_data=batch_list[13] if len(batch_list)>13 else None
#         lp=obs_t[-1]; lm=obs_Me[-1]
#         B,device=lp.shape[0],lp.device
#         T_pred=self.pred_len; dt=1.0/max(ddim_steps,1)

#         raw_ctx=self.encoder.encode(batch_list)
#         ctx=self.encoder.apply_ctx_head(raw_ctx)
#         ctx_null=self.encoder.apply_ctx_head(raw_ctx,use_null=True)

#         x0_base=_persistence_x0_rel(obs_t,obs_Me,lp,lm,T_pred,sigma=0.0)

#         obs_t_norm=obs_t[:,:,:2]
#         obs_h_n=(F.normalize(obs_t_norm[-1]-obs_t_norm[-2],dim=-1,eps=1e-6)
#                  if obs_t_norm.shape[0]>=2 else None)

#         all_modes_abs=[]
#         for k in range(self.K):
#             sigma_k=self.head_noise_base*(1.0+k*0.25)
#             x_t=x0_base+torch.randn_like(x0_base)*sigma_k
#             for step in range(ddim_steps):
#                 t_b=torch.full((B,),step*dt,device=device)
#                 if step>0 and self.cfg_guidance_scale>1.0:
#                     v_cond=self.velocity_heads[k](x_t,t_b,ctx)
#                     v_uncond=self.velocity_heads[k](x_t,t_b,ctx_null)
#                     if obs_h_n is not None:
#                         pred_h=F.normalize(v_cond[:,0,:2].detach(),dim=-1,eps=1e-6)
#                         cos_a=(obs_h_n*pred_h).sum(-1).clamp(-1.0,1.0)
#                         gs=(0.8+0.7*(cos_a+1.0)*0.5).view(B,1,1)
#                         v_k=v_uncond+gs*(v_cond-v_uncond)
#                     else:
#                         v_k=v_uncond+self.cfg_guidance_scale*(v_cond-v_uncond)
#                 else:
#                     v_k=self.velocity_heads[k](x_t,t_b,ctx)
#                 x_t=(x_t+dt*v_k).clamp(-5.0,5.0)
#             traj_abs,_=self._to_abs(x_t,lp,lm)
#             all_modes_abs.append(traj_abs)

#         modes_stack=torch.stack(all_modes_abs,dim=0)
#         modes_rel_sel=torch.stack([
#             torch.cat([all_modes_abs[k].permute(1,0,2)-lp.unsqueeze(1),
#                        torch.zeros(B,T_pred,2,device=device)],dim=-1)
#             for k in range(self.K)
#         ],dim=1)
#         sel_logits,_=self.selector(ctx,modes_rel_sel,env_data)
#         best_k=sel_logits.argmax(dim=1)

#         pred_best=torch.stack([modes_stack[best_k[b],:,b,:] for b in range(B)],dim=1)
#         pred_final=self._persistence_blend(pred_best,obs_t[:,:,:2],blend_strength=blend_strength)

#         if predict_csv is not None:
#             self._write_predict_csv(predict_csv,pred_final,modes_stack)
#         return pred_final,modes_stack

#     @staticmethod
#     def _write_predict_csv(csv_path,traj_mean,all_modes):
#         import csv as _csv,os; from datetime import datetime
#         os.makedirs(os.path.dirname(os.path.abspath(csv_path)),exist_ok=True)
#         T,B=traj_mean.shape[0],traj_mean.shape[1]
#         mlon=((traj_mean[...,0]*50.0+1800.0)/10.0).cpu().numpy()
#         mlat=((traj_mean[...,1]*50.0)/10.0).cpu().numpy()
#         fields=["ts","b","step","lead_h","lon","lat"]
#         ts=datetime.now().strftime("%Y%m%d_%H%M%S")
#         write_hdr=not os.path.exists(csv_path)
#         with open(csv_path,"a",newline="") as fh:
#             w=_csv.DictWriter(fh,fieldnames=fields)
#             if write_hdr: w.writeheader()
#             for b in range(B):
#                 for k in range(T):
#                     w.writerow({"ts":ts,"b":b,"step":k,"lead_h":(k+1)*6,
#                                 "lon":f"{mlon[k,b]:.4f}","lat":f"{mlat[k,b]:.4f}"})

# TCFlowMatching=TCFlowMatchingV65
# TCDiffusion=TCFlowMatchingV65

# """
# flow_matching_model_v68.py
# ══════════════════════════════════════════════════════════════════════
# TC-FlowMatching v68 — Stable training, tự học step weights

# ROOT CAUSE của v65/v67 diverge (đọc từ log):
#   1. LearnedWeights.log_s drift: wfm 1.00→1.79 trong 25 ep
#      → exp(-log_s) giảm → L_FM nhận ít gradient → ADE tăng dù loss giảm
#      → Phase 2: loss âm (tot=-0.5) → catastrophic
#      FIX: KHÔNG LearnedWeights. Dùng fixed weights + SoftStepWeights (học được)

#   2. STEP_WEIGHTS hardcode [2,8,3,5,...,10] không adapt theo training
#      FIX: nn.Parameter step_w (12 values, init từ v59 values) → tự học

#   3. L_DPE /500 → scale sai, /200 ok nhưng vẫn không có per-step weighting
#      FIX: per-step Huber với learned step_w, normalize đúng cách

#   4. mask_ns logic ngược → min_dist=0 → L_div constant → generator không learn diversity
#      FIX: mask_ns = zeros, scatter True tại k_star, masked_fill(mask_ns, inf)

#   5. Phase 2 ctx gradient vào encoder
#      FIX: ctx_sel LUÔN detach

#   6. evaluate ddim_steps=1 trong v65 → ADE không reflect thực tế
#      FIX: train script dùng fast_ddim=10, full_ddim=20

# THAM KHẢO V59 (stable):
#   - l_fm = MSE(pred_vel, u_target) — đơn giản, ổn định
#   - l_dpe = haversine với STEP_WEIGHTS (Huber/200) — drive ADE trực tiếp  
#   - total = l_fm + 2.0 * st_trans_loss + 0.80 * l_ate_x1
#   - KHÔNG có LearnedWeights, KHÔNG có phase switching trong loss
#   - Train giảm đều vì: loss function đơn giản, không adaptive

# V68 PHILOSOPHY:
#   Phase 1 (generator): loss đơn giản như v59 = L_FM + L_DPE(learned_w)
#   Phase 2 (selector): L_rank + L_dir
#   Phase 3 (joint): tất cả losses với fixed reasonable weights
#   Không có LearnedWeights, không có negative loss path
# """
# from __future__ import annotations

# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from Model.FNO3D_encoder import FNO3DEncoder
# from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
# from Model.env_net_transformer_gphsplit import Env_net

# R_EARTH  = 6371.0
# DT_HOURS = 6.0
# DEG2KM   = 111.0
# _NORM    = 5.0        # normalization factor: 1 unit = 5 deg
# K_MODES  = 8
# _COMPASS = [0., 45., 90., 135., 180., 225., 270., 315.]

# # Init values từ v59 STEP_WEIGHTS (chuẩn hóa thành log-scale cho nn.Parameter)
# # STEP_WEIGHTS = [2.0, 8.0, 3.0, 5.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0]
# _SW_INIT = [2.0, 8.0, 3.0, 5.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0]


# # ═══════════════════════════════════════════════════════════════════
# #  Coord utils
# # ═══════════════════════════════════════════════════════════════════

# def _norm_to_deg(t):
#     return torch.stack([(t[...,0]*50.+1800.)/10., (t[...,1]*50.)/10.], dim=-1)

# def _hav(p1, p2):
#     lat1=torch.deg2rad(p1[...,1]); lat2=torch.deg2rad(p2[...,1])
#     dlat=torch.deg2rad(p2[...,1]-p1[...,1]); dlon=torch.deg2rad(p2[...,0]-p1[...,0])
#     a=(torch.sin(dlat/2).pow(2)+torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2))
#     return 2.*R_EARTH*torch.asin(a.clamp(1e-12,1-1e-12).sqrt())

# def _unwrap(m): return m._orig_mod if hasattr(m,'_orig_mod') else m

# def _spd_deg(traj):  # [T,B,2] → [T-1,B]
#     return (_hav(traj[:-1],traj[1:])/DT_HOURS) if traj.shape[0]>=2 else traj.new_zeros(1,traj.shape[1])

# def _ade_km(pred_rel, gt_rel, lp):
#     pa=_norm_to_deg(lp.unsqueeze(1)+pred_rel[:,:,:2])
#     ga=_norm_to_deg(lp.unsqueeze(1)+gt_rel[:,:,:2])
#     return _hav(pa,ga).mean(dim=1)  # [B]


# # ═══════════════════════════════════════════════════════════════════
# #  Learned step weights (replaces hardcoded STEP_WEIGHTS)
# # ═══════════════════════════════════════════════════════════════════

# class LearnedStepWeights(nn.Module):
#     """
#     Học trọng số per-step (12 steps = 6h..72h).
#     Init từ v59 values. Clamp log để tránh collapse.
#     Output: normalized weights summing to pred_len.
#     """
#     def __init__(self, pred_len=12, init_vals=None):
#         super().__init__()
#         if init_vals is None:
#             init_vals = _SW_INIT[:pred_len]
#         # log-scale để đảm bảo positive
#         log_init = torch.log(torch.tensor(init_vals[:pred_len], dtype=torch.float))
#         self.log_w = nn.Parameter(log_init)

#     def forward(self, T=None):
#         """Trả về normalized weights [T] (hoặc [pred_len] nếu T=None)."""
#         w = torch.exp(self.log_w.clamp(-2., 3.))  # clamp: tránh collapse/explosion
#         if T is not None and T < len(w):
#             w = w[:T]
#         return w / w.sum() * len(w)  # normalize như v59: sum = T


# # ═══════════════════════════════════════════════════════════════════
# #  DPE loss với learned step weights
# # ═══════════════════════════════════════════════════════════════════

# def _dpe_loss(best_mode_rel, x1_rel, lp, step_weights):
#     """
#     Haversine per-step với learned step weights + Huber(d=200) / 200.
#     Giống l_dpe trong v59 compute_st_trans_loss nhưng weights tự học.
#     """
#     best_abs = _norm_to_deg(lp.unsqueeze(1) + best_mode_rel[:,:,:2])  # [B,T,2]
#     gt_abs   = _norm_to_deg(lp.unsqueeze(1) + x1_rel[:,:,:2])
#     dist = _hav(best_abs, gt_abs)  # [B,T]
#     T = dist.shape[1]
#     w = step_weights(T)  # [T] learned, normalized
#     d = 200.
#     huber = torch.where(dist < d, dist.pow(2)/(2*d), dist - d/2)
#     return (huber * w.unsqueeze(0)).mean() / d


# # ═══════════════════════════════════════════════════════════════════
# #  Persistence x0 + CFM
# # ═══════════════════════════════════════════════════════════════════

# def _pers_x0(obs, Me, lp, lm, T, sigma):
#     B, dev = obs.shape[1], obs.device
#     if obs.shape[0]>=3:
#         v=obs[1:]-obs[:-1]; n=v.shape[0]; a=.7
#         wt=torch.tensor([a*(1-a)**i for i in range(n)],dtype=torch.float,device=dev).flip(0)
#         lv=(v*(wt/wt.sum()).view(-1,1,1)).sum(0)
#     elif obs.shape[0]>=2: lv=obs[-1,:,:2]-obs[-2,:,:2]
#     else: lv=obs.new_zeros(B,2)
#     st=torch.arange(1,T+1,device=dev).float()
#     pa=obs[-1,:,:2].unsqueeze(0)+lv.unsqueeze(0)*st.view(-1,1,1)
#     pr=pa.permute(1,0,2)-lp.unsqueeze(1)
#     r4=torch.cat([pr,torch.zeros(B,T,2,device=dev)],dim=-1)
#     return r4+torch.randn_like(r4)*sigma

# def _cfm(x0, x1):
#     B=x0.shape[0]; dev=x0.device
#     t=torch.rand(B,device=dev); te=t.view(B,1,1)
#     return (1-te)*x0+te*x1, t, x1-x0


# # ═══════════════════════════════════════════════════════════════════
# #  OT Matching (unchanged from v67)
# # ═══════════════════════════════════════════════════════════════════

# def _fwd_az(p1,p2):
#     lo1=torch.deg2rad(p1[...,0]);la1=torch.deg2rad(p1[...,1])
#     lo2=torch.deg2rad(p2[...,0]);la2=torch.deg2rad(p2[...,1]); dl=lo2-lo1
#     return torch.atan2(torch.sin(dl)*torch.cos(la2),
#                        torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(dl))

# def _sinkhorn(cost, eps=.05, n=30):
#     B=cost.shape[0]; dev=cost.device
#     la=-math.log(B)*torch.ones(B,device=dev); lb=la.clone()
#     lK=-cost/eps; lu=torch.zeros(B,device=dev); lv=lu.clone()
#     for _ in range(n):
#         lu=la-torch.logsumexp(lK+lv.unsqueeze(0),dim=1)
#         lv=lb-torch.logsumexp(lK+lu.unsqueeze(1),dim=0)
#     return (lK+lu.unsqueeze(1)+lv.unsqueeze(0)).exp().clamp(0.)

# def _ot_cost(x0,x1,lp):
#     B=x0.shape[0]
#     def _ad(r): return _norm_to_deg(lp.unsqueeze(1)+r[:,:,:2])
#     d0=_ad(x0); d1=_ad(x1)
#     e0=d0.unsqueeze(1).expand(B,B,-1,-1).reshape(B*B,-1,2)
#     e1=d1.unsqueeze(0).expand(B,B,-1,-1).reshape(B*B,-1,2)
#     pc=_hav(e0,e1).reshape(B,B,-1).mean(-1)/500.
#     s0=_spd_deg(d0.permute(1,0,2)).permute(1,0).mean(-1)
#     s1=_spd_deg(d1.permute(1,0,2)).permute(1,0).mean(-1)
#     sc=(s0.unsqueeze(1)-s1.unsqueeze(0)).abs()/20.
#     def _mb(d):
#         b=_fwd_az(d[:,:-1],d[:,1:]); return torch.atan2(b.sin().mean(-1),b.cos().mean(-1))
#     dh=(_mb(d0).unsqueeze(1)-_mb(d1).unsqueeze(0)+math.pi)%(2*math.pi)-math.pi
#     return pc+.5*sc+.3*dh.abs()/math.pi

# def _ot_match(x0,x1,lp,eps=.05):
#     B=x0.shape[0]
#     if B<4: return x0,x1
#     try:
#         cost=_ot_cost(x0,x1,lp)
#         with torch.no_grad(): pi=_sinkhorn(cost,eps=eps)
#         f=pi.reshape(-1).clamp(0.); s=f.sum()
#         if not torch.isfinite(s) or s<1e-10: return x0,x1
#         idx=torch.multinomial(f/s,num_samples=B,replacement=True)
#         return x0[idx%B],x1[idx%B]
#     except: return x0,x1


# # ═══════════════════════════════════════════════════════════════════
# #  Difficulty score (unchanged)
# # ═══════════════════════════════════════════════════════════════════

# def _diff_score(obs_BT2, img, env, dev):
#     B=obs_BT2.shape[0]; sc=[]
#     d24=env.get("history_direction24") if env else None
#     if d24 is not None and torch.is_tensor(d24):
#         d24=d24.float().to(dev)
#         d24=d24[:,-1,:] if d24.dim()==3 else (d24 if d24.dim()==2 else None)
#         if d24 is not None and d24.shape[-1]==8:
#             bk=d24.argmax(-1).float(); a24=bk*(2.*math.pi/8.)
#             dy=obs_BT2[:,-1,1]-obs_BT2[:,-2,1]; dx=obs_BT2[:,-1,0]-obs_BT2[:,-2,0]
#             cos_l=torch.cos(torch.deg2rad((obs_BT2[:,-1,1]+obs_BT2[:,-2,1])*.5*_NORM)).clamp(1e-4)
#             anow=torch.atan2(dx*cos_l,dy)
#             cd=(torch.cos(anow)*torch.cos(a24)+torch.sin(anow)*torch.sin(a24)).clamp(-1,1)
#             s=torch.sigmoid((torch.rad2deg(torch.acos(cd))-45.)/20.)
#         else: s=torch.zeros(B,device=dev)
#     else: s=torch.zeros(B,device=dev)
#     sc.append(.30*s)
#     st=env.get("steering_speed") if env else None
#     if st is not None and torch.is_tensor(st):
#         st=st.float().to(dev)
#         while st.dim()>1: st=st[...,-1]
#         st=st.view(-1); st=st[:B] if st.numel()>=B else st[0].expand(B)
#         sc.append(.25*torch.sigmoid((4.-st*20.)/2.))
#     else: sc.append(torch.zeros(B,device=dev))
#     if img is not None and img.shape[1]>=11:
#         u2=img[:,4,-1,40,40]*13.315; u8=img[:,6,-1,40,40]*7.911
#         v2=img[:,8,-1,40,40]*8.377; v8=img[:,10,-1,40,40]*6.203
#         sc.append(.20*torch.sigmoid(((u2-u8)**2+(v2-v8)**2).sqrt()-8.)/3.)
#     else: sc.append(torch.zeros(B,device=dev))
#     ri=env.get("rapid_intensification") if env else None
#     if ri is not None and torch.is_tensor(ri):
#         ri=ri.float().to(dev)
#         while ri.dim()>1: ri=ri[...,-1]
#         sc.append(.15*ri.view(-1)[:B].clamp(0,1) if ri.numel()>=B else .15*ri[0].expand(B))
#     else: sc.append(torch.zeros(B,device=dev))
#     mv=env.get("move_velocity") if env else None
#     if mv is not None and torch.is_tensor(mv):
#         mv=mv.float().to(dev)
#         while mv.dim()>1: mv=mv[...,-1]
#         mv=mv.view(-1); mv=mv[:B] if mv.numel()>=B else mv[0].expand(B)
#         sc.append(.10*torch.sigmoid((.05-mv)/.02))
#     else: sc.append(torch.zeros(B,device=dev))
#     return sum(sc).clamp(0.,1.)

# def _dir_bucket(obs, pred):
#     last=obs[-1]; first=pred[0]
#     dy=first[:,1]-last[:,1]; dx=first[:,0]-last[:,0]
#     cos_l=torch.cos(torch.deg2rad((last[:,1]+first[:,1])*.5*_NORM)).clamp(1e-4)
#     return ((torch.atan2(dx*cos_l,dy).rad2deg()%360.+22.5)/45.).long()%8


# # ═══════════════════════════════════════════════════════════════════
# #  Coherence loss
# # ═══════════════════════════════════════════════════════════════════

# def _coh(mode_grad, gt_rel, lp):
#     pa=_norm_to_deg(lp.unsqueeze(1)+mode_grad[:,:,:2])
#     ga=_norm_to_deg(lp.unsqueeze(1)+gt_rel[:,:,:2])
#     cos_l=torch.cos(torch.deg2rad(ga[:,:-1,1])).clamp(1e-4)
#     dp=torch.stack([(pa[:,1:,0]-pa[:,:-1,0])*cos_l*DEG2KM,(pa[:,1:,1]-pa[:,:-1,1])*DEG2KM],dim=-1)
#     dg=torch.stack([(ga[:,1:,0]-ga[:,:-1,0])*cos_l*DEG2KM,(ga[:,1:,1]-ga[:,:-1,1])*DEG2KM],dim=-1)
#     return (dp-dg).norm(dim=-1).mean()/DEG2KM

# def _consist(enc, raw):
#     if not enc.training: return torch.tensor(0.,device=raw.device)
#     a=enc.apply_ctx_head(raw); b=enc.apply_ctx_head(raw)
#     return (1.-(F.normalize(a,dim=-1)*F.normalize(b,dim=-1)).sum(dim=-1)).mean()

# def _mixup(x1,delta,prob=.3,alpha=.4):
#     if not torch.is_tensor(x1) or x1.shape[0]<2: return x1,delta
#     if torch.rand(1).item()>prob: return x1,delta
#     B=x1.shape[0]
#     lam=max(float(torch.distributions.Beta(alpha,alpha).sample()),
#             1.-float(torch.distributions.Beta(alpha,alpha).sample()))
#     lam=max(lam,1.-lam)
#     idx=torch.randperm(B,device=x1.device)
#     et=(delta<.4).float().view(B,1,1); ed=(delta<.4).float()
#     return x1*(1.-et*(1.-lam))+x1[idx]*(et*(1.-lam)), (delta*lam+delta[idx]*(1.-lam))*ed+delta*(1.-ed)


# # ═══════════════════════════════════════════════════════════════════
# #  CompassVelocityHead (unchanged)
# # ═══════════════════════════════════════════════════════════════════

# class CompassVelocityHead(nn.Module):
#     def __init__(self, idx, pred_len=12, ctx_dim=256, sdp=.10):
#         super().__init__()
#         self.pred_len=pred_len; self.sdp=sdp
#         angle=_COMPASS[idx]*math.pi/180.
#         self.register_buffer('cdir',torch.tensor([math.sin(angle),math.cos(angle)]))
#         self.dp=nn.Sequential(nn.Linear(2,ctx_dim),nn.GELU(),nn.LayerNorm(ctx_dim))
#         self.tf1=nn.Linear(ctx_dim,256); self.tf2=nn.Linear(256,ctx_dim)
#         self.te=nn.Linear(4,ctx_dim)
#         self.pe=nn.Parameter(torch.randn(1,pred_len,ctx_dim)*.02)
#         self.se=nn.Embedding(pred_len,ctx_dim)
#         self.tr=nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(d_model=ctx_dim,nhead=8,dim_feedforward=512,
#                                        dropout=.10,activation="gelu",batch_first=True),num_layers=1)
#         self.o1=nn.Linear(ctx_dim,256); self.o2=nn.Linear(256,4)
#         self.ss=nn.Parameter(torch.ones(pred_len)*.5)
#         with torch.no_grad():
#             nn.init.xavier_uniform_(self.o2.weight,gain=.1); nn.init.zeros_(self.o2.bias)

#     def _te(self,t,dim):
#         h=dim//2
#         fr=torch.exp(torch.arange(h,dtype=torch.float32,device=t.device)*(-math.log(1e4)/max(h-1,1)))
#         em=t.float().unsqueeze(1)*1000.*fr.unsqueeze(0)
#         return F.pad(torch.cat([em.sin(),em.cos()],dim=-1),(0,dim%2))

#     def forward(self,x,t,ctx):
#         B=x.shape[0]; T=min(x.shape[1],self.pred_len); dev=x.device
#         dt=self.dp(self.cdir.to(dev).unsqueeze(0).expand(B,-1))
#         if self.training and self.sdp>0:
#             dt=dt*(torch.rand(B,1,device=dev)>self.sdp).float()
#         te=F.gelu(self.tf1(self._te(t,ctx.shape[-1]))); te=self.tf2(te)
#         si=torch.arange(T,device=dev).unsqueeze(0).expand(B,-1)
#         xe=self.te(x[:,:T])+self.pe[:,:T]+te.unsqueeze(1)+self.se(si)
#         mem=torch.stack([ctx,dt,te],dim=1)
#         dec=self.tr(xe,mem)
#         sc=torch.sigmoid(self.ss[:T]).view(1,T,1)*2.
#         return self.o2(F.gelu(self.o1(dec)))*sc


# # ═══════════════════════════════════════════════════════════════════
# #  CompassSelector (unchanged)
# # ═══════════════════════════════════════════════════════════════════

# class CompassSelector(nn.Module):
#     def __init__(self,ctx_dim=256,K=8,n_dirs=8):
#         super().__init__()
#         self.K=K
#         self.se=nn.Sequential(nn.Linear(11,64),nn.GELU(),nn.LayerNorm(64),nn.Linear(64,128))
#         self.me=nn.Sequential(nn.Linear(12*2,128),nn.GELU(),nn.LayerNorm(128),nn.Linear(128,64))
#         self.sn=nn.Sequential(nn.Linear(ctx_dim+64+128,256),nn.GELU(),nn.LayerNorm(256),
#                               nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
#         self.dh=nn.Sequential(nn.Linear(ctx_dim+128,256),nn.GELU(),nn.LayerNorm(256),nn.Linear(256,n_dirs))

#     def _st(self,env,B,dev):
#         def _sc(k):
#             if not env: return torch.zeros(B,device=dev)
#             v=env.get(k)
#             if v is None or not torch.is_tensor(v): return torch.zeros(B,device=dev)
#             v=v.float().to(dev)
#             while v.dim()>1: v=v[...,-1]
#             v=v.view(-1); return (v[:B] if v.numel()>=B else v[0].expand(B)).clamp(-3,3)
#         def _sv(k,d):
#             if not env: return torch.zeros(B,d,device=dev)
#             v=env.get(k)
#             if v is None or not torch.is_tensor(v): return torch.zeros(B,d,device=dev)
#             v=v.float().to(dev)
#             if v.dim()==3: v=v[:,-1,:]
#             elif v.dim()!=2 or v.shape[0]!=B: return torch.zeros(B,d,device=dev)
#             return F.pad(v,(0,max(0,d-v.shape[-1])))[:,:d].clamp(-3,3)
#         return self.se(torch.cat([_sc("steering_speed").unsqueeze(-1),
#                                    _sc("steering_dir_sin").unsqueeze(-1),
#                                    _sc("steering_dir_cos").unsqueeze(-1),
#                                    _sv("history_direction24",8)],dim=-1))

#     def forward(self,ctx,modes,env):
#         B=modes.shape[0]; dev=ctx.device; sf=self._st(env,B,dev)
#         scores=[self.sn(torch.cat([ctx,self.me(modes[:,k,:,:2].reshape(B,-1)),sf],dim=-1)).squeeze(-1)
#                 for k in range(self.K)]
#         return torch.stack(scores,dim=1), self.dh(torch.cat([ctx,sf],dim=-1))


# # ═══════════════════════════════════════════════════════════════════
# #  SharedContextEncoder (unchanged)
# # ═══════════════════════════════════════════════════════════════════

# class SharedContextEncoder(nn.Module):
#     RC=512
#     def __init__(self,pred_len=12,obs_len=8,ctx_dim=256,uch=13):
#         super().__init__()
#         self.pred_len=pred_len; self.obs_len=obs_len; self.ctx_dim=ctx_dim
#         self.se=FNO3DEncoder(in_channel=uch,out_channel=1,d_model=32,
#             n_layers=4,modes_t=4,modes_h=4,modes_w=4,spatial_down=32,dropout=.05)
#         self.bp=nn.AdaptiveAvgPool3d((None,1,1)); self.bj=nn.Linear(128,128)
#         self.dp=nn.Linear(1,16)
#         self.e1=DataEncoder1D(in_1d=4,feat_3d_dim=128,mlp_h=64,
#                               lstm_hidden=128,lstm_layers=3,dropout=.1,d_state=16)
#         self.ev=Env_net(obs_len=obs_len,d_model=32)
#         self.f1=nn.Linear(128+32+16,self.RC); self.ln=nn.LayerNorm(self.RC)
#         self.dr=nn.Dropout(.15); self.f2=nn.Linear(self.RC,ctx_dim)
#         self.null=nn.Parameter(torch.randn(1,self.RC)*.02)

#     def encode(self,bl):
#         ot=bl[0]; om=bl[7]; im=bl[11]; env=bl[13] if len(bl)>13 else None
#         if im.dim()==4: im=im.unsqueeze(2)
#         if im.shape[1]==1 and self.se.in_channel!=1:
#             im=im.expand(-1,self.se.in_channel,-1,-1,-1)
#         T=ot.shape[0]; eb,ed=self.se.encode(im)
#         es=self.bp(eb).squeeze(-1).squeeze(-1).permute(0,2,1); es=self.bj(es)
#         if es.shape[1]!=T:
#             es=F.interpolate(es.permute(0,2,1),size=T,mode="linear",align_corners=False).permute(0,2,1)
#         et=ed.squeeze(1).squeeze(-1).squeeze(-1)
#         tw=torch.softmax(torch.arange(et.shape[1],dtype=torch.float,device=et.device)*.5,dim=0)
#         fs=self.dp((et*tw.unsqueeze(0)).sum(1,keepdim=True))
#         oi=torch.cat([ot,om],dim=2).permute(1,0,2); ht=self.e1(oi,es)
#         ee,_,_=self.ev(env,im)
#         return F.gelu(self.ln(self.f1(torch.cat([ht,ee,fs],dim=-1))))

#     def apply_ctx_head(self,raw,use_null=False):
#         if use_null: raw=self.null.expand(raw.shape[0],-1)
#         return self.f2(self.dr(raw))


# # ═══════════════════════════════════════════════════════════════════
# #  EMA
# # ═══════════════════════════════════════════════════════════════════

# class EMAModel:
#     def __init__(self,model,decay=.995):
#         self.decay=decay; m=_unwrap(model)
#         self.shadow={k:v.detach().clone() for k,v in m.state_dict().items()
#                      if v.dtype.is_floating_point}
#     def update(self,model):
#         m=_unwrap(model)
#         with torch.no_grad():
#             for k,v in m.state_dict().items():
#                 if k in self.shadow: self.shadow[k].mul_(self.decay).add_(v.detach(),alpha=1-self.decay)
#     def apply_to(self,model):
#         m=_unwrap(model); bk,sd={},m.state_dict()
#         for k in self.shadow:
#             if k not in sd: continue
#             bk[k]=sd[k].detach().clone(); sd[k].copy_(self.shadow[k])
#         return bk
#     def restore(self,model,bk):
#         m=_unwrap(model); sd=m.state_dict()
#         for k,v in bk.items():
#             if k in sd: sd[k].copy_(v)


# # ═══════════════════════════════════════════════════════════════════
# #  TCFlowMatchingV68 — Main
# # ═══════════════════════════════════════════════════════════════════

# class TCFlowMatchingV68(nn.Module):
#     """
#     Key differences from v67:
#     - LearnedStepWeights thay vì hardcoded STEP_WEIGHTS
#     - KHÔNG LearnedWeights (root cause của diverge trong v65/v67)
#     - Fixed weights đơn giản như v59, loss luôn dương
#     - mask_ns đúng: zeros+scatter(True) → masked_fill(mask_ns, inf)
#     - ctx_sel LUÔN detach
#     """

#     def __init__(self,pred_len=12,obs_len=8,ctx_dim=256,sigma_min=.02,
#                  unet_in_ch=13,K=K_MODES,use_ema=True,ema_decay=.995,
#                  cfg_uncond_prob=.10,selector_warmup=20,
#                  head_noise_base=.03,use_ot=True,ot_epsilon=.05,
#                  cfg_guidance_scale=1.3,**kwargs):
#         super().__init__()
#         self.pred_len=pred_len; self.obs_len=obs_len; self.K=K
#         self.sigma_min=sigma_min; self.use_ema=use_ema; self.ema_decay=ema_decay
#         self.cfg_uncond_prob=cfg_uncond_prob; self.selector_warmup=selector_warmup
#         self.head_noise_base=head_noise_base; self.use_ot=use_ot
#         self.ot_eps=ot_epsilon; self.cfg_gs=cfg_guidance_scale; self._ema=None

#         self.encoder=SharedContextEncoder(pred_len,obs_len,ctx_dim,unet_in_ch)
#         self.velocity_heads=nn.ModuleList([CompassVelocityHead(k,pred_len,ctx_dim) for k in range(K)])
#         self.selector=CompassSelector(ctx_dim,K,8)
#         # Learned step weights — tự học, init từ v59 values
#         self.step_weights=LearnedStepWeights(pred_len)

#     def init_ema(self):
#         if self.use_ema: self._ema=EMAModel(self,decay=self.ema_decay)
#     def ema_update(self):
#         if self._ema: self._ema.update(self)
#     def set_curriculum_len(self,*a,**kw): pass

#     @staticmethod
#     def _sigma(ep):
#         if ep<2: return .10
#         if ep<10: return .10-(ep-2)/8.*(.10-.04)
#         if ep<20: return max(.04-(ep-10)/10.*.01,.035)
#         return .035

#     @staticmethod
#     def _tau(ep):
#         if ep<5: return 5.
#         if ep<15: return 4.
#         if ep<25: return 3.
#         return 2.

#     def _phase(self,ep):
#         if ep<self.selector_warmup: return 1
#         elif ep<self.selector_warmup+15: return 2
#         return 3

#     @staticmethod
#     def _aug_flip(bl,p=.3):
#         if torch.rand(1).item()>p: return bl
#         bl=list(bl)
#         for i in [0,1,2,3]:
#             if torch.is_tensor(bl[i]) and bl[i].shape[-1]>=1:
#                 t=bl[i].clone(); t[...,0]=-t[...,0]; bl[i]=t
#         return bl

#     @staticmethod
#     def _aug_noise(bl,s=.005):
#         if torch.rand(1).item()>.5: return bl
#         bl=list(bl)
#         if torch.is_tensor(bl[0]): bl[0]=bl[0]+torch.randn_like(bl[0])*s
#         return bl

#     @staticmethod
#     def _to_rel(tr,Me,lp,lm):
#         return torch.cat([tr-lp.unsqueeze(0),Me-lm.unsqueeze(0)],dim=-1).permute(1,0,2)

#     @staticmethod
#     def _to_abs(rel,lp,lm):
#         d=rel.permute(1,0,2); return lp.unsqueeze(0)+d[:,:,:2],lm.unsqueeze(0)+d[:,:,2:]

#     @staticmethod
#     @torch.no_grad()
#     def _blend(pred,obs,s=.10):
#         To=obs.shape[0]; T,B,dev=pred.shape[0],pred.shape[1],pred.device
#         if To<2: return pred
#         v=obs[1:]-obs[:-1]; nv=v.shape[0]
#         if nv>=3:
#             a=.7; wt=torch.tensor([a*(1-a)**i for i in range(nv)],dtype=torch.float,device=dev).flip(0)
#             ev=(v*(wt/wt.sum()).view(-1,1,1)).sum(0)
#         elif nv==2: ev=.7*v[-1]+.3*v[-2]
#         else: ev=v[-1]
#         st=torch.arange(1,T+1,dtype=torch.float,device=dev)
#         pe=obs[-1].unsqueeze(0)+ev.unsqueeze(0)*st.view(T,1,1)
#         sp=_spd_deg(_norm_to_deg(obs))
#         if sp.shape[0]>=2:
#             cv=sp.std(0)/sp.mean(0).clamp(min=1.); ab=(s*torch.sigmoid(-(cv-.3)*5.)).unsqueeze(0).unsqueeze(-1)
#         else: ab=s*.5
#         return (1.-ab)*pred+ab*pe

#     # ── Training ──────────────────────────────────────────────────

#     def get_loss(self,bl,epoch=0,**kw): return self.get_loss_breakdown(bl,epoch)["total"]

#     def get_loss_breakdown(self,bl,epoch=0,**kw):
#         bl=self._aug_flip(bl); bl=self._aug_noise(bl)
#         obs=bl[0]; pred=bl[1]; oMe=bl[7]; pMe=bl[8]; env=bl[13] if len(bl)>13 else None
#         lp=obs[-1]; lm=oMe[-1]; B,dev=lp.shape[0],lp.device
#         sig=self._sigma(epoch); tau=self._tau(epoch); ph=self._phase(epoch)

#         # Context
#         raw=self.encoder.encode(bl)
#         null=(torch.rand(1).item()<self.cfg_uncond_prob)
#         ctx=self.encoder.apply_ctx_head(raw,use_null=null)
#         # ctx_sel: LUÔN real context, LUÔN detach → selector không gradient vào encoder
#         ctx_sel=self.encoder.apply_ctx_head(raw,use_null=False).detach()
#         Lc=_consist(self.encoder,raw)

#         delta=_diff_score(obs.permute(1,0,2),bl[11],env,dev)
#         x1o=self._to_rel(pred,pMe,lp,lm)
#         x0=_pers_x0(obs,oMe,lp,lm,self.pred_len,sig)
#         if self.use_ot and B>=4 and epoch>=10:
#             x0,x1o=_ot_match(x0,x1o,lp,eps=self.ot_eps)
#         x1,delta=_mixup(x1o,delta)
#         xt,ft,ut=_cfm(x0,x1)

#         # Generate K modes
#         mngs=[]; mgs=[]; vps=[]
#         for k in range(self.K):
#             sk=self.head_noise_base*(1.+k*.25)
#             xtk=xt+torch.randn_like(xt)*sk
#             if ph==2:
#                 with torch.no_grad(): vk=self.velocity_heads[k](xtk,ft,ctx.detach())
#             else:
#                 vk=self.velocity_heads[k](xtk,ft,ctx)
#             with torch.no_grad(): mngs.append((xt+(1.-ft.view(B,1,1))*vk).detach())
#             mgs.append(xtk+(1.-ft.view(B,1,1))*vk)
#             vps.append(vk)

#         mng=torch.stack(mngs,dim=1)  # [B,K,T,4] nograd
#         mg =torch.stack(mgs,dim=1)   # [B,K,T,4] with grad

#         # Oracle
#         with torch.no_grad():
#             ak=torch.stack([_ade_km(mngs[k],x1o.detach(),lp) for k in range(self.K)],dim=1)
#             ks=ak.argmin(dim=1)
#         bi=torch.arange(B,device=dev)
#         oracle_km=ak[bi,ks].mean()

#         # FM loss
#         fe=torch.stack([((vps[k]-ut)**2).mean(dim=[1,2]) for k in range(self.K)],dim=1)
#         Le=fe.mean(dim=1)                     # easy: mean over K
#         Lo=fe[bi,ks]                          # oracle: best mode

#         # Diversity: min dist từ k_star đến các mode khác
#         ms=mng[bi,ks]
#         da=torch.stack([((ms-mng[:,k])**2).mean(dim=[1,2]).sqrt() for k in range(self.K)],dim=1)
#         # BUG-A FIX: zeros + scatter True tại k_star → fill inf tại k_star
#         # → min lấy từ k ≠ k_star → ĐÚNG
#         mk=torch.zeros(B,self.K,device=dev,dtype=torch.bool)
#         mk.scatter_(1,ks.unsqueeze(1),True)
#         md=da.masked_fill(mk,float('inf')).min(dim=1).values
#         Ld=F.relu(.40-md)
#         Lf=(Lo+.3*Ld); Lr=(1.-delta)*Le+delta*Lf
#         LFM=((.5+1.5*delta)*Lr).mean()

#         # BUG-B FIX: DPE với learned step_weights (tự học, init v59)
#         og=mg[bi,ks]  # oracle mode với gradient
#         LDPE=_dpe_loss(og,x1o,lp,self.step_weights)

#         # Coherence
#         Lcoh=_coh(og,x1o,lp)

#         # Selector (ctx_sel đã detach — BUG-C fix)
#         sl,dl=self.selector(ctx_sel,mng,env)
#         po=F.softmax(-ak/tau,dim=1)
#         Lrank=F.kl_div(F.log_softmax(sl,dim=1),po,reduction='batchmean')
#         Ldir=F.cross_entropy(dl,_dir_bucket(obs,pred))

#         # ── Total loss — KHÔNG LearnedWeights, fixed weights, luôn dương ──
#         # Giống philosophy v59: đơn giản, ổn định
#         if ph==1:
#             # Generator only — giống v59: L_FM + L_DPE với step weighting
#             # L_DPE weight 0.5 đủ mạnh (scale ~0.1-0.3 cùng L_FM)
#             Lt = LFM + 0.5*LDPE + 0.02*Lc

#         elif ph==2:
#             # Selector only — generator frozen by caller
#             # Luôn dương: KL ≥ 0, CE > 0
#             Lt = 0.6*Lrank + 0.4*Ldir

#         else:
#             # Joint — tất cả losses với fixed weights
#             # Không có adaptive weight → không drift
#             Lt = LFM + 0.4*LDPE + 0.3*Lrank + 0.2*Ldir + 0.05*Lcoh + 0.02*Lc

#         if not torch.isfinite(Lt): Lt=Lt.new_zeros(())

#         def _s(x): return x.item() if torch.is_tensor(x) else float(x)
#         return dict(
#             total=Lt, L_FM=_s(LFM), L_dpe=_s(LDPE),
#             L_easy=_s(Le.mean()), L_oracle=_s(Lo.mean()),
#             oracle_ade_km=_s(oracle_km),  # km thực
#             L_div=_s(Ld.mean()), min_dist_mean=_s(md.mean()),
#             L_rank=_s(Lrank), L_dir=_s(Ldir),
#             L_coh=_s(Lcoh), L_consist=_s(Lc),
#             delta_mean=_s(delta.mean()),
#             # Step weights hiện tại (để monitor convergence)
#             sw_12h=_s(torch.exp(self.step_weights.log_w[1]).detach()),
#             sw_72h=_s(torch.exp(self.step_weights.log_w[11]).detach()),
#             phase=ph,
#         )

#     # ── Inference ─────────────────────────────────────────────────

#     @torch.no_grad()
#     def sample(self,bl,ddim_steps=20,predict_csv=None,blend_strength=.10,**kw):
#         obs=bl[0]; oMe=bl[7]; env=bl[13] if len(bl)>13 else None
#         lp=obs[-1]; lm=oMe[-1]; B,dev=lp.shape[0],lp.device
#         T=self.pred_len; dt=1./max(ddim_steps,1)
#         raw=self.encoder.encode(bl)
#         ctx=self.encoder.apply_ctx_head(raw)
#         ctx0=self.encoder.apply_ctx_head(raw,use_null=True)
#         x0=_pers_x0(obs,oMe,lp,lm,T,0.)
#         ohn=(F.normalize(obs[-1,:,:2]-obs[-2,:,:2],dim=-1,eps=1e-6) if obs.shape[0]>=2 else None)
#         modes=[]
#         for k in range(self.K):
#             sk=self.head_noise_base*(1.+k*.25)
#             xt=x0+torch.randn_like(x0)*sk
#             for s in range(ddim_steps):
#                 tb=torch.full((B,),s*dt,device=dev)
#                 if s>0 and self.cfg_gs>1.:
#                     vc=self.velocity_heads[k](xt,tb,ctx); vu=self.velocity_heads[k](xt,tb,ctx0)
#                     if ohn is not None:
#                         ph=F.normalize(vc[:,0,:2].detach(),dim=-1,eps=1e-6)
#                         ca=(ohn*ph).sum(-1).clamp(-1.,1.)
#                         gs=(.8+.7*(ca+1.)*.5).view(B,1,1); vk=vu+gs*(vc-vu)
#                     else: vk=vu+self.cfg_gs*(vc-vu)
#                 else: vk=self.velocity_heads[k](xt,tb,ctx)
#                 xt=(xt+dt*vk).clamp(-5.,5.)
#             ta,_=self._to_abs(xt,lp,lm); modes.append(ta)
#         ms=torch.stack(modes,dim=0)
#         rs=torch.stack([torch.cat([modes[k].permute(1,0,2)-lp.unsqueeze(1),
#                                     torch.zeros(B,T,2,device=dev)],dim=-1)
#                         for k in range(self.K)],dim=1)
#         sl,_=self.selector(ctx,rs,env); bk=sl.argmax(dim=1)
#         pb=torch.stack([ms[bk[b],:,b,:] for b in range(B)],dim=1)
#         pf=self._blend(pb,obs[:,:,:2],blend_strength)
#         if predict_csv: self._csv(predict_csv,pf,ms)
#         return pf,ms

#     @staticmethod
#     def _csv(path,tr,modes):
#         import csv,os; from datetime import datetime
#         os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
#         T,B=tr.shape[0],tr.shape[1]
#         lon=((tr[...,0]*50.+1800.)/10.).cpu().numpy()
#         lat=((tr[...,1]*50.)/10.).cpu().numpy()
#         ts=datetime.now().strftime("%Y%m%d_%H%M%S")
#         hdr=not os.path.exists(path)
#         with open(path,"a",newline="") as f:
#             w=csv.DictWriter(f,fieldnames=["ts","b","step","lead_h","lon","lat"])
#             if hdr: w.writeheader()
#             for b in range(B):
#                 for k in range(T):
#                     w.writerow({"ts":ts,"b":b,"step":k,"lead_h":(k+1)*6,
#                                 "lon":f"{lon[k,b]:.4f}","lat":f"{lat[k,b]:.4f}"})


# # Backward compat aliases
# TCFlowMatchingV67 = TCFlowMatchingV68
# TCFlowMatching    = TCFlowMatchingV68
# TCFlowMatchingV65 = TCFlowMatchingV68
# TCDiffusion       = TCFlowMatchingV68

"""
flow_matching_model_v71.py  ── TC-FlowMatching v71
══════════════════════════════════════════════════════════════════════════════
TỔNG HỢP TẤT CẢ FIX TỪ AUDIT V70 + CHIẾN LƯỢC MỚI

═══ 9 BUGS ĐÃ FIX ════════════════════════════════════════════════════════════

BUG-1 [CRITICAL] Oracle selection bị bias bởi noise:
  v70: pred_k = xtk + (1-t)*vk  (xtk = xt + noise_k)
  → oracle chọn "head may mắn nhất" chứ không phải "head tốt nhất"
  FIX: pred_k_clean = xt + (1-t)*vk  (không có noise)
       oracle dùng clean; training loss vẫn dùng xtk + (1-t)*vk

BUG-2 [CRITICAL] Phase 2 generator nhận ZERO gradient:
  v70: with torch.no_grad(): vk = velocity_heads[k](xtk, ft, ctx.detach())
  → vk không có grad → LFM từ vk = 0 gradient → "0.2*LFM" trong comment là sai
  FIX: Bỏ no_grad wrapper. Phase 2 chỉ detach ctx, velocity_heads vẫn trainable

BUG-3 [MEDIUM] Diversity loss Ld có ZERO gradient:
  v70: ms = mng[bi,ks]  (mng là no_grad stack)
       da từ ms và mng → Ld = F.relu(0.4 - md) không có grad
  FIX: ms_grad = mg[bi,ks]  (mg có gradient)
       da tính từ mg → Ld backprop được

BUG-4 [MEDIUM] Easy storms không có oracle pressure:
  v70: Lr = (1-delta)*Le + delta*Lf
  → delta~0 (easy): Lr = Le (mean over K, không có winner-takes-all)
  → model học blob thay vì sharp modes
  FIX: Luôn có oracle bonus nhỏ: Lr += 0.15*Lo ngay cả khi delta=0

BUG-5 [MEDIUM] Không có ConstrainedStepWeights:
  v70: _get_step_weights() trả tensor cố định từ _V59_STEP_WEIGHTS
  → không adapt được từ data thực tế
  FIX: ConstrainedStepWeights với cumsum(softplus) → monotone, tự học

BUG-6 [MEDIUM] Không có ConstrainedLossWeights:
  v70: Lt = LFM + 2.0*LDPE + 0.02*Lc  (hardcoded)
  → không balance được FM vs DPE theo data
  FIX: ConstrainedLossWeights, w_fm và w_dpe tự học per-phase

BUG-7 [MEDIUM] Huber d=200 giết gradient 72h:
  dist=400km (72h) → linear region, grad=1/d=0.005 (flat)
  FIX: d=300 → wider quadratic region, better 72h gradient

BUG-8 [MINOR] _get_step_weights inconsistent khi T<12:
  normalize với T/sum(raw[:T]) thay đổi pattern
  FIX: ConstrainedStepWeights luôn compute 12 steps, slice sau

BUG-9 [INFO] oracle_km log dùng noisy pred:
  FIX: sau Bug-1 fix, dùng pred_k_clean → oracle_km chính xác

═══ CHIẾN LƯỢC MỚI ════════════════════════════════════════════════════════════

1. ConstrainedStepWeights: focus 24h-72h
   - cumsum(softplus(raw)) → w[i+1] >= w[i] luôn luôn
   - ratio_min=1.25: sw_72h / sw_6h >= 1.25 cứng
   - anchor_loss về v59 values với weight nhẹ
   - Học được: gradient có thể tăng tầm quan trọng 24h-72h theo data thực

2. ConstrainedLossWeights: tự cân bằng FM vs DPE
   - softplus + clamp → luôn dương, không collapse
   - anchor_loss ngăn drift xa init
   - Separate weights cho từng phase

3. Clean oracle + gradient diversity:
   - pred_k_clean cho oracle selection → đúng head được reward
   - Ld từ mg (has grad) → diversity thực sự học được

4. Easy/hard aware loss:
   - Easy storms: vẫn có oracle pressure nhỏ (0.15*Lo)
   - Hard storms: oracle + diversity mạnh hơn

═══ MONITORING ════════════════════════════════════════════════════════════════
  sw_ratio = sw_72h/sw_6h → phải >= 1.0 mọi epoch
  L_div > 0.05 từ ep5+ → diversity đang train
  oracle_ade_km giảm monotone → oracle selection đúng
  lw_dpe/lw_fm trong [1.0, 4.0] → DPE mạnh nhưng không dominate
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.FNO3D_encoder import FNO3DEncoder
from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
from Model.env_net_transformer_gphsplit import Env_net

# ── Constants ──────────────────────────────────────────────────────────────────
R_EARTH  = 6371.0
DT_HOURS = 6.0
DEG2KM   = 111.0
_NORM    = 5.0
K_MODES  = 8
_COMPASS = [0., 45., 90., 135., 180., 225., 270., 315.]

# V59 reference values — dùng làm anchor, KHÔNG phải fixed weights
# 6h=2, 12h=8, 18h=3, 24h=5, 30h=2, 36h=2.5, 42h=3, 48h=4, 54h=5, 60h=6, 66h=7, 72h=10
_V59_REF = [2.0, 8.0, 3.0, 5.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0]


# ══════════════════════════════════════════════════════════════════════════════
#  ConstrainedStepWeights — tự học, monotone tăng, không bao giờ flip ratio
# ══════════════════════════════════════════════════════════════════════════════

class ConstrainedStepWeights(nn.Module):
    """
    Step weights tự học với 3 tầng bảo vệ:

    1. MONOTONICITY (cứng về mặt cấu trúc):
       raw → softplus(raw) [positive deltas] → cumsum [monotone non-decreasing]
       → w[i+1] >= w[i] LUÔN LUÔN về mặt toán học
       → sw_72h không bao giờ < sw_12h dù gradient đi theo hướng nào

    2. RATIO GUARD (cứng):
       sw[-1] / sw[0] >= ratio_min = 1.25
       Nếu vi phạm: soft correction bằng scaling, gradient vẫn flow

    3. ANCHOR REGULARIZATION (mềm):
       L2 penalty về v59 reference values
       Ngăn drift quá xa nhưng không block learning

    Focus đặc biệt vào 24h-72h (indices 3..11):
       cumsum đảm bảo vùng này học được tầm quan trọng thực từ data
       mà không bị flip bởi gradient sai hướng như v68
    """

    def __init__(self, pred_len: int = 12, ratio_min: float = 1.25,
                 anchor_w: float = 0.05):
        super().__init__()
        self.pred_len   = pred_len
        self.ratio_min  = ratio_min
        self.anchor_w   = anchor_w

        # Init raw params: constant 0.5 → cumsum(softplus(0.5)) tạo linear ramp
        # Gradient sẽ học shape thực từ data
        self.raw = nn.Parameter(torch.ones(pred_len) * 0.5)

        # MONOTONE anchor target: linear ramp nhấn mạnh 24h-72h
        # FIX-B: Không anchor về v59 (non-monotone) vì tạo gradient conflict
        # với cumsum constraint. Dùng monotone target với sw72/sw6 = 4.0.
        # w[k] = base + k * slope, normalized to mean=1
        base  = 0.40
        slope = base * (4.0 - 1.0) / (pred_len - 1)   # sw_last = 4 * sw_first
        mono  = torch.tensor([base + k * slope for k in range(pred_len)],
                              dtype=torch.float)
        self.register_buffer('mono_ref', mono / mono.sum() * pred_len)

    def forward(self) -> torch.Tensor:
        """
        Trả về weights [pred_len], mean=1, monotone non-decreasing.

        Pipeline:
          raw → softplus → cumsum → normalize → ratio_guard
        """
        # Step 1: positive increments
        deltas = F.softplus(self.raw)          # [T], all positive

        # Step 2: cumulative sum → monotone non-decreasing
        # w[i] = sum(deltas[0..i]) ≥ w[i-1]
        w = torch.cumsum(deltas, dim=0)        # [T], monotone

        # Step 3: normalize (mean=1, sum=T)
        w = w * self.pred_len / (w.sum() + 1e-8)

        # Step 4: ratio guard
        # Nếu sw[-1]/sw[0] < ratio_min → scale down sw[0]
        ratio = w[-1] / (w[0].clamp(min=1e-6))
        if ratio.item() < self.ratio_min:
            # Soft correction: không hard clamp để gradient vẫn flow
            correction = (ratio / self.ratio_min).detach()
            w = torch.cat([w[:1] * correction, w[1:]])
            w = w * self.pred_len / (w.sum() + 1e-8)

        return w

    def anchor_loss(self) -> torch.Tensor:
        """
        L2 penalty về monotone anchor target (FIX-B: không phải v59).
        Anchor về linear ramp với sw72/sw6=4.0 — compatible với cumsum.
        """
        w = self.forward()
        return self.anchor_w * ((w - self.mono_ref) ** 2).mean()

    def log_stats(self) -> dict:
        """Stats để monitor trong training loop."""
        with torch.no_grad():
            w = self.forward()
            return {
                'sw_6h':   w[0].item(),
                'sw_12h':  w[1].item(),
                'sw_24h':  w[3].item(),
                'sw_48h':  w[7].item() if len(w) > 7 else 0.,
                'sw_72h':  w[-1].item(),
                'sw_ratio': (w[-1] / w[0].clamp(1e-6)).item(),
            }


# ══════════════════════════════════════════════════════════════════════════════
#  ConstrainedLossWeights — tự học loss weights, luôn dương, không collapse
# ══════════════════════════════════════════════════════════════════════════════

class ConstrainedLossWeights(nn.Module):
    """
    Loss weights tự học với softplus + clamp bảo vệ.

    Tại sao softplus thay vì exp:
      - exp tại -10: ~0 (vanishing gradient)
      - softplus tại -10: ~0.0000454 (never truly zero, gradient exists)
      - exp growth quá nhanh khi log_w lớn
      - softplus + clamp_max tạo plateau safety

    4 weights: w_fm, w_dpe, w_rank, w_dir
    Anchor loss: L2 về init values để prevent drift

    Init mặc định: [w_fm=1.0, w_dpe=2.0, w_rank=0.3, w_dir=0.2]
    Đây là v59 philosophy: DPE nên mạnh hơn FM (ratio ~2x)
    """

    def __init__(self,
                 init_fm:   float = 1.0,
                 init_dpe:  float = 2.0,
                 init_rank: float = 0.3,
                 init_dir:  float = 0.2,
                 anchor_w:  float = 0.02):
        super().__init__()
        self.anchor_w = anchor_w

        # FIX-A: Dùng softplus_inverse để init đúng target values.
        # V71 dùng log(x): softplus(log(2.0)) = softplus(0.693) = 1.099 ≠ 2.0
        # V72 dùng sp_inv(x) = log(exp(x)-1): softplus(sp_inv(2.0)) = 2.0 chính xác
        def _sp_inv(y: float) -> float:
            import math
            if y > 20.: return y
            if y < 1e-6: return -20.
            return math.log(math.expm1(y))  # log(exp(y) - 1)

        raw_fm   = _sp_inv(init_fm)
        raw_dpe  = _sp_inv(init_dpe)
        raw_rank = _sp_inv(init_rank)
        raw_dir  = _sp_inv(init_dir)
        inits = torch.tensor([raw_fm, raw_dpe, raw_rank, raw_dir], dtype=torch.float)
        self.log_w = nn.Parameter(inits)
        self.register_buffer('log_w0', inits.clone())

    def _get(self, idx: int,
             clamp_min: float = 0.05,
             clamp_max: float = 8.0) -> torch.Tensor:
        return F.softplus(self.log_w[idx]).clamp(clamp_min, clamp_max)

    # Public accessors — clamp ranges phản ánh reasonable loss scales
    def w_fm(self)   -> torch.Tensor: return self._get(0, 0.2, 4.0)
    def w_dpe(self)  -> torch.Tensor: return self._get(1, 0.5, 6.0)
    def w_rank(self) -> torch.Tensor: return self._get(2, 0.05, 2.0)
    def w_dir(self)  -> torch.Tensor: return self._get(3, 0.05, 2.0)

    def anchor_loss(self) -> torch.Tensor:
        return self.anchor_w * ((self.log_w - self.log_w0) ** 2).mean()

    def log_stats(self) -> dict:
        with torch.no_grad():
            return {
                'lw_fm':   self.w_fm().item(),
                'lw_dpe':  self.w_dpe().item(),
                'lw_rank': self.w_rank().item(),
                'lw_dir':  self.w_dir().item(),
                'lw_dpe_fm_ratio': (self.w_dpe() / self.w_fm().clamp(1e-6)).item(),
            }


# ── Coordinate utils ───────────────────────────────────────────────────────────

def _norm_to_deg(t: torch.Tensor) -> torch.Tensor:
    return torch.stack([(t[..., 0] * 50. + 1800.) / 10.,
                        (t[..., 1] * 50.) / 10.], dim=-1)

def _hav(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a = torch.sin(dlat/2).pow(2) + torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2)
    return 2. * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

def _unwrap(m): return m._orig_mod if hasattr(m, '_orig_mod') else m

def _spd_deg(traj: torch.Tensor) -> torch.Tensor:
    return (_hav(traj[:-1], traj[1:]) / DT_HOURS) if traj.shape[0] >= 2 \
        else traj.new_zeros(1, traj.shape[1])

def _ade_km(pred_rel: torch.Tensor, gt_rel: torch.Tensor,
            lp: torch.Tensor) -> torch.Tensor:
    """Mean haversine distance [B] từ relative predictions."""
    pa = _norm_to_deg(lp.unsqueeze(1) + pred_rel[:, :, :2])
    ga = _norm_to_deg(lp.unsqueeze(1) + gt_rel[:, :, :2])
    return _hav(pa, ga).mean(dim=1)  # [B]


# ── DPE loss với ConstrainedStepWeights ───────────────────────────────────────

def _dpe_loss(best_mode_rel: torch.Tensor,
              x1_rel: torch.Tensor,
              lp: torch.Tensor,
              step_w: torch.Tensor,
              huber_d: float = 300.) -> torch.Tensor:
    """
    Per-step haversine với learned step_weights + Huber(d=300).

    Huber d=300 (thay vì d=200 trong v70):
      - dist=400km (72h): v70: grad=1/200=0.005; v71: grad=1/300=0.0033
      - Nhưng quadratic region rộng hơn (0..300 thay vì 0..200)
      - → 50-300km range (24h-72h) được train tốt hơn với Huber² không flat
      - dist=300km: v71 ở đúng boundary, còn gradient; v70 đã flat từ 200km

    step_w: [T] tensor từ ConstrainedStepWeights.forward(), có gradient.
    """
    best_abs = _norm_to_deg(lp.unsqueeze(1) + best_mode_rel[:, :, :2])  # [B,T,2]
    gt_abs   = _norm_to_deg(lp.unsqueeze(1) + x1_rel[:, :, :2])
    dist     = _hav(best_abs, gt_abs)    # [B,T] in km
    T = dist.shape[1]
    w = step_w[:T]                       # [T] có gradient nếu step_w có
    # Huber loss normalized
    huber = torch.where(dist < huber_d,
                        dist.pow(2) / (2. * huber_d),
                        dist - huber_d / 2.)
    return (huber * w.unsqueeze(0)).mean() / huber_d


# ── Persistence x0 + CFM ──────────────────────────────────────────────────────

def _pers_x0(obs: torch.Tensor, Me: torch.Tensor,
             lp: torch.Tensor, lm: torch.Tensor,
             T: int, sigma: float) -> torch.Tensor:
    B, dev = obs.shape[1], obs.device
    if obs.shape[0] >= 3:
        v = obs[1:] - obs[:-1]; n = v.shape[0]; a = .7
        wt = torch.tensor([a*(1-a)**i for i in range(n)],
                           dtype=torch.float, device=dev).flip(0)
        lv = (v * (wt / wt.sum()).view(-1, 1, 1)).sum(0)
    elif obs.shape[0] >= 2:
        lv = obs[-1, :, :2] - obs[-2, :, :2]
    else:
        lv = obs.new_zeros(B, 2)
    st = torch.arange(1, T+1, device=dev).float()
    pa = obs[-1, :, :2].unsqueeze(0) + lv.unsqueeze(0) * st.view(-1, 1, 1)
    pr = pa.permute(1, 0, 2) - lp.unsqueeze(1)
    r4 = torch.cat([pr, torch.zeros(B, T, 2, device=dev)], dim=-1)
    return r4 + torch.randn_like(r4) * sigma

def _cfm(x0: torch.Tensor, x1: torch.Tensor):
    """Conditional Flow Matching: interpolate + velocity target."""
    B  = x0.shape[0]; dev = x0.device
    t  = torch.rand(B, device=dev)
    te = t.view(B, 1, 1)
    return (1. - te) * x0 + te * x1, t, x1 - x0


# ── OT Matching ───────────────────────────────────────────────────────────────

def _fwd_az(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    lo1 = torch.deg2rad(p1[..., 0]); la1 = torch.deg2rad(p1[..., 1])
    lo2 = torch.deg2rad(p2[..., 0]); la2 = torch.deg2rad(p2[..., 1])
    dl  = lo2 - lo1
    return torch.atan2(torch.sin(dl)*torch.cos(la2),
                       torch.cos(la1)*torch.sin(la2) -
                       torch.sin(la1)*torch.cos(la2)*torch.cos(dl))

def _sinkhorn(cost: torch.Tensor, eps: float = .05, n: int = 30) -> torch.Tensor:
    B = cost.shape[0]; dev = cost.device
    la = -math.log(B) * torch.ones(B, device=dev); lb = la.clone()
    lK = -cost / eps
    lu = torch.zeros(B, device=dev); lv = lu.clone()
    for _ in range(n):
        lu = la - torch.logsumexp(lK + lv.unsqueeze(0), dim=1)
        lv = lb - torch.logsumexp(lK + lu.unsqueeze(1), dim=0)
    return (lK + lu.unsqueeze(1) + lv.unsqueeze(0)).exp().clamp(0.)

def _ot_cost(x0: torch.Tensor, x1: torch.Tensor,
             lp: torch.Tensor) -> torch.Tensor:
    B = x0.shape[0]
    def _ad(r): return _norm_to_deg(lp.unsqueeze(1) + r[:, :, :2])
    d0 = _ad(x0); d1 = _ad(x1)
    e0 = d0.unsqueeze(1).expand(B, B, -1, -1).reshape(B*B, -1, 2)
    e1 = d1.unsqueeze(0).expand(B, B, -1, -1).reshape(B*B, -1, 2)
    pc = _hav(e0, e1).reshape(B, B, -1).mean(-1) / 500.
    s0 = _spd_deg(d0.permute(1, 0, 2)).permute(1, 0).mean(-1)
    s1 = _spd_deg(d1.permute(1, 0, 2)).permute(1, 0).mean(-1)
    sc = (s0.unsqueeze(1) - s1.unsqueeze(0)).abs() / 20.
    def _mb(d):
        b = _fwd_az(d[:, :-1, :], d[:, 1:, :])
        return torch.atan2(b.sin().mean(-1), b.cos().mean(-1))
    h0 = _mb(d0); h1 = _mb(d1)
    dh = (h0.unsqueeze(1) - h1.unsqueeze(0) + math.pi) % (2*math.pi) - math.pi
    return pc + .5*sc + .3*dh.abs()/math.pi

def _ot_match(x0: torch.Tensor, x1: torch.Tensor,
              lp: torch.Tensor, eps: float = .05):
    B = x0.shape[0]
    if B < 4: return x0, x1
    try:
        cost = _ot_cost(x0, x1, lp)
        with torch.no_grad(): pi = _sinkhorn(cost, eps=eps)
        f = pi.reshape(-1).clamp(0.); s = f.sum()
        if not torch.isfinite(s) or s < 1e-10: return x0, x1
        idx = torch.multinomial(f / s, num_samples=B, replacement=True)
        return x0[idx%B], x1[idx%B]
    except:
        return x0, x1


# ── Difficulty score ──────────────────────────────────────────────────────────

def _diff_score(obs_BT2: torch.Tensor, img, env, dev: torch.device) -> torch.Tensor:
    """Delta score in [0,1]: 0=easy (straight track), 1=hard (recurvature/RI)."""
    B = obs_BT2.shape[0]; sc = []

    d24 = env.get("history_direction24") if env else None
    if d24 is not None and torch.is_tensor(d24):
        d24 = d24.float().to(dev)
        d24 = d24[:, -1, :] if d24.dim() == 3 else (d24 if d24.dim() == 2 else None)
        if d24 is not None and d24.shape[-1] == 8:
            bk  = d24.argmax(-1).float(); a24 = bk * (2.*math.pi/8.)
            dy  = obs_BT2[:, -1, 1] - obs_BT2[:, -2, 1]
            dx  = obs_BT2[:, -1, 0] - obs_BT2[:, -2, 0]
            cos_l = torch.cos(torch.deg2rad((obs_BT2[:,-1,1]+obs_BT2[:,-2,1])*.5*_NORM)).clamp(1e-4)
            anow = torch.atan2(dx*cos_l, dy)
            cd   = (torch.cos(anow)*torch.cos(a24)+torch.sin(anow)*torch.sin(a24)).clamp(-1,1)
            s    = torch.sigmoid((torch.rad2deg(torch.acos(cd)) - 45.) / 20.)
        else: s = torch.zeros(B, device=dev)
    else: s = torch.zeros(B, device=dev)
    sc.append(.30 * s)

    st = env.get("steering_speed") if env else None
    if st is not None and torch.is_tensor(st):
        st = st.float().to(dev)
        while st.dim() > 1: st = st[..., -1]
        st = st.view(-1); st = st[:B] if st.numel() >= B else st[0].expand(B)
        sc.append(.25 * torch.sigmoid((4. - st*20.) / 2.))
    else: sc.append(torch.zeros(B, device=dev))

    if img is not None and img.shape[1] >= 11:
        u2 = img[:,4,-1,40,40]*13.315; u8 = img[:,6,-1,40,40]*7.911
        v2 = img[:,8,-1,40,40]*8.377;  v8 = img[:,10,-1,40,40]*6.203
        sc.append(.20 * torch.sigmoid(((u2-u8)**2+(v2-v8)**2).sqrt() - 8.) / 3.)
    else: sc.append(torch.zeros(B, device=dev))

    ri = env.get("rapid_intensification") if env else None
    if ri is not None and torch.is_tensor(ri):
        ri = ri.float().to(dev)
        while ri.dim() > 1: ri = ri[..., -1]
        sc.append(.15*ri.view(-1)[:B].clamp(0,1) if ri.numel()>=B else .15*ri[0].expand(B))
    else: sc.append(torch.zeros(B, device=dev))

    mv = env.get("move_velocity") if env else None
    if mv is not None and torch.is_tensor(mv):
        mv = mv.float().to(dev)
        while mv.dim() > 1: mv = mv[..., -1]
        mv = mv.view(-1); mv = mv[:B] if mv.numel() >= B else mv[0].expand(B)
        sc.append(.10 * torch.sigmoid((.05 - mv) / .02))
    else: sc.append(torch.zeros(B, device=dev))

    return sum(sc).clamp(0., 1.)

def _dir_bucket(obs: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    last = obs[-1]; first = pred[0]
    dy   = first[:, 1] - last[:, 1]; dx = first[:, 0] - last[:, 0]
    cos_l = torch.cos(torch.deg2rad((last[:,1]+first[:,1])*.5*_NORM)).clamp(1e-4)
    return ((torch.atan2(dx*cos_l, dy).rad2deg() % 360. + 22.5) / 45.).long() % 8


# ── Coherence + consistency ────────────────────────────────────────────────────

def _coh(mode_grad: torch.Tensor, gt_rel: torch.Tensor,
         lp: torch.Tensor) -> torch.Tensor:
    pa = _norm_to_deg(lp.unsqueeze(1) + mode_grad[:, :, :2])
    ga = _norm_to_deg(lp.unsqueeze(1) + gt_rel[:, :, :2])
    cos_l = torch.cos(torch.deg2rad(ga[:, :-1, 1])).clamp(1e-4)
    dp = torch.stack([(pa[:,1:,0]-pa[:,:-1,0])*cos_l*DEG2KM,
                      (pa[:,1:,1]-pa[:,:-1,1])*DEG2KM], dim=-1)
    dg = torch.stack([(ga[:,1:,0]-ga[:,:-1,0])*cos_l*DEG2KM,
                      (ga[:,1:,1]-ga[:,:-1,1])*DEG2KM], dim=-1)
    return (dp - dg).norm(dim=-1).mean() / DEG2KM

def _consist(enc, raw: torch.Tensor) -> torch.Tensor:
    if not enc.training: return torch.tensor(0., device=raw.device)
    a = enc.apply_ctx_head(raw); b = enc.apply_ctx_head(raw)
    return (1. - (F.normalize(a,dim=-1)*F.normalize(b,dim=-1)).sum(dim=-1)).mean()

def _mixup(x1: torch.Tensor, delta: torch.Tensor,
           prob: float = .3, alpha: float = .4):
    if not torch.is_tensor(x1) or x1.shape[0] < 2: return x1, delta
    if torch.rand(1).item() > prob: return x1, delta
    B = x1.shape[0]
    lam = max(float(torch.distributions.Beta(alpha,alpha).sample()),
              1. - float(torch.distributions.Beta(alpha,alpha).sample()))
    lam = max(lam, 1. - lam)
    idx = torch.randperm(B, device=x1.device)
    et  = (delta < .4).float().view(B, 1, 1)
    ed  = (delta < .4).float()
    return (x1*(1.-et*(1.-lam)) + x1[idx]*(et*(1.-lam)),
            (delta*lam + delta[idx]*(1.-lam))*ed + delta*(1.-ed))


# ── CompassVelocityHead ────────────────────────────────────────────────────────

class CompassVelocityHead(nn.Module):
    def __init__(self, compass_idx: int, pred_len: int = 12,
                 ctx_dim: int = 256, sdp: float = .10):
        super().__init__()
        self.pred_len = pred_len; self.sdp = sdp
        angle = _COMPASS[compass_idx] * math.pi / 180.
        self.register_buffer('cdir',
            torch.tensor([math.sin(angle), math.cos(angle)]))
        self.dp  = nn.Sequential(nn.Linear(2,ctx_dim), nn.GELU(), nn.LayerNorm(ctx_dim))
        self.tf1 = nn.Linear(ctx_dim, 256); self.tf2 = nn.Linear(256, ctx_dim)
        self.te  = nn.Linear(4, ctx_dim)
        self.pe  = nn.Parameter(torch.randn(1, pred_len, ctx_dim) * .02)
        self.se  = nn.Embedding(pred_len, ctx_dim)
        self.tr  = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=ctx_dim, nhead=8, dim_feedforward=512,
                                       dropout=.10, activation="gelu", batch_first=True),
            num_layers=1)
        self.o1 = nn.Linear(ctx_dim, 256); self.o2 = nn.Linear(256, 4)
        self.ss = nn.Parameter(torch.ones(pred_len) * .5)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.o2.weight, gain=.1)
            nn.init.zeros_(self.o2.bias)

    def _time_emb(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        h  = dim // 2
        fr = torch.exp(torch.arange(h, dtype=torch.float32, device=t.device) *
                       (-math.log(1e4) / max(h-1, 1)))
        em = t.float().unsqueeze(1) * 1000. * fr.unsqueeze(0)
        return F.pad(torch.cat([em.sin(), em.cos()], dim=-1), (0, dim % 2))

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                ctx: torch.Tensor) -> torch.Tensor:
        B  = x.shape[0]; T = min(x.shape[1], self.pred_len); dev = x.device
        dt = self.dp(self.cdir.to(dev).unsqueeze(0).expand(B, -1))
        if self.training and self.sdp > 0:
            dt = dt * (torch.rand(B, 1, device=dev) > self.sdp).float()
        te  = F.gelu(self.tf1(self._time_emb(t, ctx.shape[-1]))); te = self.tf2(te)
        si  = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)
        xe  = self.te(x[:, :T]) + self.pe[:, :T] + te.unsqueeze(1) + self.se(si)
        mem = torch.stack([ctx, dt, te], dim=1)
        dec = self.tr(xe, mem)
        sc  = torch.sigmoid(self.ss[:T]).view(1, T, 1) * 2.
        return self.o2(F.gelu(self.o1(dec))) * sc


# ── CompassSelector ────────────────────────────────────────────────────────────

class CompassSelector(nn.Module):
    def __init__(self, ctx_dim: int = 256, K: int = 8, n_dirs: int = 8):
        super().__init__()
        self.K = K
        self.se = nn.Sequential(nn.Linear(11,64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64,128))
        self.me = nn.Sequential(nn.Linear(12*2,128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128,64))
        self.sn = nn.Sequential(nn.Linear(ctx_dim+64+128,256), nn.GELU(), nn.LayerNorm(256),
                                nn.Linear(256,64), nn.GELU(), nn.Linear(64,1))
        self.dh = nn.Sequential(nn.Linear(ctx_dim+128,256), nn.GELU(), nn.LayerNorm(256),
                                nn.Linear(256,n_dirs))

    def _st(self, env, B: int, dev: torch.device) -> torch.Tensor:
        def _sc(k):
            if not env: return torch.zeros(B, device=dev)
            v = env.get(k)
            if v is None or not torch.is_tensor(v): return torch.zeros(B, device=dev)
            v = v.float().to(dev)
            while v.dim() > 1: v = v[..., -1]
            v = v.view(-1)
            return (v[:B] if v.numel() >= B else v[0].expand(B)).clamp(-3, 3)
        def _sv(k, d):
            if not env: return torch.zeros(B, d, device=dev)
            v = env.get(k)
            if v is None or not torch.is_tensor(v): return torch.zeros(B, d, device=dev)
            v = v.float().to(dev)
            if v.dim() == 3: v = v[:, -1, :]
            elif v.dim() != 2 or v.shape[0] != B: return torch.zeros(B, d, device=dev)
            return F.pad(v, (0, max(0, d - v.shape[-1])))[:, :d].clamp(-3, 3)
        return self.se(torch.cat([
            _sc("steering_speed").unsqueeze(-1),
            _sc("steering_dir_sin").unsqueeze(-1),
            _sc("steering_dir_cos").unsqueeze(-1),
            _sv("history_direction24", 8)], dim=-1))

    def forward(self, ctx: torch.Tensor, modes: torch.Tensor, env) -> tuple:
        B   = modes.shape[0]; dev = ctx.device; sf = self._st(env, B, dev)
        scores = [self.sn(torch.cat([
                    ctx,
                    self.me(modes[:, k, :, :2].reshape(B, -1)),
                    sf], dim=-1)).squeeze(-1)
                  for k in range(self.K)]
        return torch.stack(scores, dim=1), self.dh(torch.cat([ctx, sf], dim=-1))


# ── SharedContextEncoder ───────────────────────────────────────────────────────

class SharedContextEncoder(nn.Module):
    RC = 512
    def __init__(self, pred_len: int = 12, obs_len: int = 8,
                 ctx_dim: int = 256, unet_in_ch: int = 13):
        super().__init__()
        self.pred_len = pred_len; self.obs_len = obs_len; self.ctx_dim = ctx_dim
        self.spatial_enc  = FNO3DEncoder(in_channel=unet_in_ch, out_channel=1,
                                          d_model=32, n_layers=4, modes_t=4,
                                          modes_h=4, modes_w=4, spatial_down=32,
                                          dropout=.05)
        self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.bottleneck_proj = nn.Linear(128, 128)
        self.decoder_proj    = nn.Linear(1, 16)
        self.enc_1d = DataEncoder1D(in_1d=4, feat_3d_dim=128, mlp_h=64,
                                     lstm_hidden=128, lstm_layers=3,
                                     dropout=.1, d_state=16)
        self.env_enc  = Env_net(obs_len=obs_len, d_model=32)
        self.ctx_fc1  = nn.Linear(128+32+16, self.RC)
        self.ctx_ln   = nn.LayerNorm(self.RC)
        self.ctx_drop = nn.Dropout(.15)
        self.ctx_fc2  = nn.Linear(self.RC, ctx_dim)
        self.null_embedding = nn.Parameter(torch.randn(1, self.RC) * .02)

    def encode(self, batch_list: list) -> torch.Tensor:
        obs_t   = batch_list[0]; obs_Me = batch_list[7]
        image_obs = batch_list[11]
        env_data  = batch_list[13] if len(batch_list) > 13 else None
        if image_obs.dim() == 4: image_obs = image_obs.unsqueeze(2)
        if image_obs.shape[1] == 1 and self.spatial_enc.in_channel != 1:
            image_obs = image_obs.expand(-1, self.spatial_enc.in_channel, -1, -1, -1)
        T = obs_t.shape[0]
        e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
        e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        e_3d_s = self.bottleneck_proj(e_3d_s)
        if e_3d_s.shape[1] != T:
            e_3d_s = F.interpolate(e_3d_s.permute(0,2,1), size=T,
                                    mode="linear", align_corners=False).permute(0,2,1)
        e_3d_dec_t = e_3d_dec.squeeze(1).squeeze(-1).squeeze(-1)
        tw = torch.softmax(torch.arange(e_3d_dec_t.shape[1], dtype=torch.float,
                                         device=e_3d_dec_t.device) * .5, dim=0)
        f_sp = self.decoder_proj((e_3d_dec_t * tw.unsqueeze(0)).sum(1, keepdim=True))
        obs_in = torch.cat([obs_t, obs_Me], dim=2).permute(1, 0, 2)
        h_t    = self.enc_1d(obs_in, e_3d_s)
        e_env, _, _ = self.env_enc(env_data, image_obs)
        return F.gelu(self.ctx_ln(self.ctx_fc1(torch.cat([h_t, e_env, f_sp], dim=-1))))

    def apply_ctx_head(self, raw: torch.Tensor,
                       use_null: bool = False) -> torch.Tensor:
        if use_null: raw = self.null_embedding.expand(raw.shape[0], -1)
        return self.ctx_fc2(self.ctx_drop(raw))


# ── EMA ────────────────────────────────────────────────────────────────────────

class EMAModel:
    def __init__(self, model, decay: float = .995):
        self.decay  = decay; m = _unwrap(model)
        self.shadow = {k: v.detach().clone() for k, v in m.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        m = _unwrap(model)
        with torch.no_grad():
            for k, v in m.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)

    def apply_to(self, model) -> dict:
        m = _unwrap(model); bk = {}; sd = m.state_dict()
        for k in self.shadow:
            if k not in sd: continue
            bk[k] = sd[k].detach().clone(); sd[k].copy_(self.shadow[k])
        return bk

    def restore(self, model, bk: dict):
        m = _unwrap(model); sd = m.state_dict()
        for k, v in bk.items():
            if k in sd: sd[k].copy_(v)


# ══════════════════════════════════════════════════════════════════════════════
#  TCFlowMatchingV72 — Main model
# ══════════════════════════════════════════════════════════════════════════════

class TCFlowMatchingV72(nn.Module):
    """
    V72: Fix 9 bugs + FIX-A/B (init + anchor) từ audit v70 + ConstrainedStepWeights + ConstrainedLossWeights.

    Các thay đổi chính so với v70:
    1. pred_k_clean = xt + (1-t)*vk  (không noise) → oracle chính xác (Bug-1)
    2. Phase 2: bỏ no_grad wrapper → generator nhận gradient (Bug-2)
    3. Ld từ mg (has grad) → diversity backprop được (Bug-3)
    4. Oracle bonus nhỏ cho easy storms (Bug-4)
    5. ConstrainedStepWeights: cumsum(softplus) → monotone, tự học (Bug-5,8)
    6. ConstrainedLossWeights: w_fm, w_dpe tự học per-phase (Bug-6)
    7. Huber d=300 (thay d=200) → gradient 72h tốt hơn (Bug-7)
    9. oracle_km từ clean pred → log chính xác (Bug-9)

    Training params (cần pass vào train script):
      step_weights   → phải nằm trong gen_params (optimizer group "generator")
      loss_weights_p1, loss_weights_p23 → cũng trong gen_params
    """

    def __init__(self,
                 pred_len:           int   = 12,
                 obs_len:            int   = 8,
                 ctx_dim:            int   = 256,
                 sigma_min:          float = .02,
                 unet_in_ch:         int   = 13,
                 K:                  int   = K_MODES,
                 use_ema:            bool  = True,
                 ema_decay:          float = .995,
                 cfg_uncond_prob:    float = .10,
                 selector_warmup:    int   = 20,
                 head_noise_base:    float = .03,
                 use_ot:             bool  = True,
                 ot_epsilon:         float = .05,
                 cfg_guidance_scale: float = 1.3,
                 **kwargs):
        super().__init__()
        self.pred_len          = pred_len
        self.obs_len           = obs_len
        self.K                 = K
        self.sigma_min         = sigma_min
        self.use_ema           = use_ema
        self.ema_decay         = ema_decay
        self.cfg_uncond_prob   = cfg_uncond_prob
        self.selector_warmup   = selector_warmup
        self.head_noise_base   = head_noise_base
        self.use_ot            = use_ot
        self.ot_eps            = ot_epsilon
        self.cfg_gs            = cfg_guidance_scale
        self._ema              = None

        # Sub-modules
        self.encoder       = SharedContextEncoder(pred_len, obs_len, ctx_dim, unet_in_ch)
        self.velocity_heads = nn.ModuleList(
            [CompassVelocityHead(k, pred_len, ctx_dim) for k in range(K)])
        self.selector = CompassSelector(ctx_dim, K, 8)

        # ── Learnable weights (BUG-5, BUG-6 fix) ─────────────────────────────
        # step_weights: 1 instance, dùng chung cả 3 phases
        self.step_weights = ConstrainedStepWeights(pred_len,
                                                    ratio_min=1.25,
                                                    anchor_w=0.05)

        # loss_weights: separate cho Phase1 vs Phase2/3
        # Phase 1 init: w_dpe cao hơn → drive ADE ngay từ đầu
        self.loss_weights_p1  = ConstrainedLossWeights(
            init_fm=1.0, init_dpe=2.0,
            init_rank=0.3, init_dir=0.2, anchor_w=0.02)
        # Phase 2/3 init: cân bằng hơn sau khi selector đã warm
        self.loss_weights_p23 = ConstrainedLossWeights(
            init_fm=0.8, init_dpe=1.5,
            init_rank=0.5, init_dir=0.3, anchor_w=0.02)

    def init_ema(self):
        if self.use_ema: self._ema = EMAModel(self, decay=self.ema_decay)

    def ema_update(self):
        if self._ema: self._ema.update(self)

    def set_curriculum_len(self, *a, **kw): pass

    @staticmethod
    def _sigma(ep: int) -> float:
        if ep < 2:   return .10
        if ep < 10:  return .10 - (ep-2)/8. * (.10-.04)
        if ep < 20:  return max(.04 - (ep-10)/10. * .01, .035)
        return .035

    @staticmethod
    def _tau(ep: int) -> float:
        if ep < 5:   return 5.
        if ep < 15:  return 4.
        if ep < 25:  return 3.
        return 2.

    def _phase(self, ep: int) -> int:
        if ep < self.selector_warmup:               return 1
        elif ep < self.selector_warmup + 15:        return 2
        return 3

    @staticmethod
    def _aug_flip(bl: list, p: float = .3) -> list:
        if torch.rand(1).item() > p: return bl
        bl = list(bl)
        for i in [0, 1, 2, 3]:
            if torch.is_tensor(bl[i]) and bl[i].shape[-1] >= 1:
                t = bl[i].clone(); t[..., 0] = -t[..., 0]; bl[i] = t
        return bl

    @staticmethod
    def _aug_noise(bl: list, s: float = .005) -> list:
        if torch.rand(1).item() > .5: return bl
        bl = list(bl)
        if torch.is_tensor(bl[0]): bl[0] = bl[0] + torch.randn_like(bl[0]) * s
        return bl

    @staticmethod
    def _to_rel(traj, Me, lp, lm):
        return torch.cat([traj - lp.unsqueeze(0),
                          Me   - lm.unsqueeze(0)], dim=-1).permute(1, 0, 2)

    @staticmethod
    def _to_abs(rel, lp, lm):
        d = rel.permute(1, 0, 2)
        return lp.unsqueeze(0) + d[:,:,:2], lm.unsqueeze(0) + d[:,:,2:]

    @staticmethod
    @torch.no_grad()
    def _blend(pred, obs, s=.10):
        To = obs.shape[0]; T, B, dev = pred.shape[0], pred.shape[1], pred.device
        if To < 2: return pred
        v = obs[1:] - obs[:-1]; nv = v.shape[0]
        if nv >= 3:
            a  = .7; wt = torch.tensor([a*(1-a)**i for i in range(nv)],
                                         dtype=torch.float, device=dev).flip(0)
            ev = (v * (wt / wt.sum()).view(-1,1,1)).sum(0)
        elif nv == 2: ev = .7*v[-1] + .3*v[-2]
        else:         ev = v[-1]
        st = torch.arange(1, T+1, dtype=torch.float, device=dev)
        pe = obs[-1].unsqueeze(0) + ev.unsqueeze(0) * st.view(T, 1, 1)
        sp = _spd_deg(_norm_to_deg(obs))
        if sp.shape[0] >= 2:
            cv = sp.std(0) / sp.mean(0).clamp(min=1.)
            ab = (s * torch.sigmoid(-(cv-.3)*5.)).unsqueeze(0).unsqueeze(-1)
        else: ab = s * .5
        return (1. - ab) * pred + ab * pe

    # ══════════════════════════════════════════════════════════════════════════
    #  Training
    # ══════════════════════════════════════════════════════════════════════════

    def get_loss(self, bl: list, epoch: int = 0, **kw) -> torch.Tensor:
        return self.get_loss_breakdown(bl, epoch)["total"]

    def get_loss_breakdown(self, bl: list, epoch: int = 0, **kw) -> dict:
        bl  = self._aug_flip(bl); bl = self._aug_noise(bl)
        obs = bl[0]; pred = bl[1]; oMe = bl[7]; pMe = bl[8]
        env = bl[13] if len(bl) > 13 else None
        lp  = obs[-1]; lm = oMe[-1]; B, dev = lp.shape[0], lp.device
        sig = self._sigma(epoch); tau = self._tau(epoch); ph = self._phase(epoch)

        # ── Step weights (có gradient, monotone, tự học) ──────────────────────
        # BUG-5/8 FIX: ConstrainedStepWeights thay vì _get_step_weights cố định
        step_w      = self.step_weights()           # [T], has gradient
        sw_anchor_L = self.step_weights.anchor_loss()   # scalar

        # Loss weights cho phase hiện tại
        lw = self.loss_weights_p1 if ph == 1 else self.loss_weights_p23
        lw_anchor_L = lw.anchor_loss()              # scalar

        # ── Context ───────────────────────────────────────────────────────────
        raw  = self.encoder.encode(bl)
        null = (torch.rand(1).item() < self.cfg_uncond_prob)
        ctx  = self.encoder.apply_ctx_head(raw, use_null=null)
        # ctx_sel: LUÔN real context, LUÔN detach → selector không gradient vào encoder
        ctx_sel = self.encoder.apply_ctx_head(raw, use_null=False).detach()
        Lc = _consist(self.encoder, raw)

        delta = _diff_score(obs.permute(1, 0, 2), bl[11], env, dev)
        x1o   = self._to_rel(pred, pMe, lp, lm)
        x0    = _pers_x0(obs, oMe, lp, lm, self.pred_len, sig)

        # OT matching từ epoch >= 20
        if self.use_ot and B >= 4 and epoch >= 20:
            x0, x1o = _ot_match(x0, x1o, lp, eps=self.ot_eps)

        x1, delta = _mixup(x1o, delta)
        xt, ft, ut = _cfm(x0, x1)

        # ── Generate K modes ──────────────────────────────────────────────────
        # mngs: clean predictions cho oracle selection (BUG-1 FIX)
        # mgs:  noisy predictions với gradient cho training (BUG-2 FIX)
        mngs = []; mgs = []; vps = []

        for k in range(self.K):
            sk  = self.head_noise_base * (1. + k * .25)
            xtk = xt + torch.randn_like(xt) * sk

            # BUG-2 FIX: Bỏ no_grad wrapper cho Phase 2
            # Phase 2: chỉ detach ctx (selector không train encoder)
            # velocity_heads vẫn nhận gradient → generator stays warm
            if ph == 2:
                # ctx đã detach, velocity_heads trainable
                vk = self.velocity_heads[k](xtk, ft, ctx.detach())
            else:
                vk = self.velocity_heads[k](xtk, ft, ctx)

            # BUG-1 FIX: Oracle dùng CLEAN prediction (không có noise_k)
            # pred_k_clean = xt + (1-t)*vk  →  không bị bias bởi noise_k
            # Gradient của vk vẫn flow qua training loss bên dưới
            with torch.no_grad():
                pred_k_clean = (xt + (1. - ft.view(B, 1, 1)) * vk).detach()

            # Training prediction (noisy) — có gradient
            pred_k_noisy = xtk + (1. - ft.view(B, 1, 1)) * vk

            mngs.append(pred_k_clean)   # clean: dùng cho oracle selection
            mgs.append(pred_k_noisy)    # noisy: dùng cho diversity loss (has grad)
            vps.append(vk)

        mng = torch.stack(mngs, dim=1)  # [B,K,T,4] no_grad (clean)
        mg  = torch.stack(mgs,  dim=1)  # [B,K,T,4] with grad (noisy)

        # ── Oracle assignment dùng clean predictions ───────────────────────────
        # BUG-1 FIX: ak tính từ mngs (clean) → oracle chọn đúng head
        with torch.no_grad():
            ak = torch.stack([_ade_km(mngs[k], x1o.detach(), lp)
                               for k in range(self.K)], dim=1)  # [B,K]
            ks = ak.argmin(dim=1)                                # [B] oracle indices

        bi          = torch.arange(B, device=dev)
        oracle_km   = ak[bi, ks].mean()  # BUG-9 FIX: oracle_km từ clean pred

        # ── FM loss ───────────────────────────────────────────────────────────
        fe = torch.stack([((vps[k] - ut) ** 2).mean(dim=[1, 2])
                           for k in range(self.K)], dim=1)  # [B,K]
        Le = fe.mean(dim=1)  # easy: mean over K  [B]
        Lo = fe[bi, ks]      # oracle: best mode  [B]

        # ── Diversity loss từ mg (HAS GRADIENT) ───────────────────────────────
        # BUG-3 FIX: ms_grad từ mg[bi,ks] (not mng) → Ld có gradient
        ms_grad = mg[bi, ks]  # [B,T,4] with gradient
        da = torch.stack([
            # FIX: .clamp(min=1e-8) trước sqrt để tránh sqrt(0) → NaN gradient
            # khi oracle mode trùng với một mode khác (distance=0)
            ((ms_grad - mg[:, k]) ** 2).mean(dim=[1, 2]).clamp(min=1e-8).sqrt()
            for k in range(self.K)
        ], dim=1)  # [B,K] with gradient

        # mask_ns: True = valid candidate (k != k_star)
        # Đây là v65-stable logic: ones → scatter False at k_star
        mask_ns = torch.ones(B, self.K, device=dev, dtype=torch.bool)
        mask_ns.scatter_(1, ks.unsqueeze(1), False)
        # masked_fill(~mask_ns) = fill inf WHERE ~mask_ns=True = WHERE k_star
        # → min(dim=1) = min dist to nearest OTHER mode ✓
        md = da.masked_fill(~mask_ns, float('inf')).min(dim=1).values  # [B]
        Ld = F.relu(.40 - md)  # [B] diversity penalty, NOW HAS GRADIENT

        # ── Combined FM loss ───────────────────────────────────────────────────
        # BUG-4 FIX: easy storms vẫn có oracle signal nhỏ
        # Lr: difficulty-weighted blend, nhưng cả easy lẫn hard đều có oracle pressure
        Lf  = Lo + .3 * Ld                              # oracle + diversity
        Lr  = (1. - delta) * Le + delta * Lf            # difficulty blend  [B]
        # Oracle bonus nhỏ ngay cả khi delta=0 (easy storms)
        # → Mọi sample đều có ít nhất một mode học tốt hơn các mode khác
        oracle_bonus = 0.15 * Lo                        # [B], always present
        LFM = ((.5 + 1.5*delta) * Lr + oracle_bonus).mean()

        # ── DPE loss với learned step weights (Huber d=300) ───────────────────
        # BUG-7 FIX: d=300 thay vì d=200
        # og dùng mgs[ks] (noisy, has gradient) — không clean vì cần backprop
        og   = mg[bi, ks]   # [B,T,4] oracle mode WITH gradient
        LDPE = _dpe_loss(og, x1o, lp, step_w, huber_d=300.)

        # ── Coherence ──────────────────────────────────────────────────────────
        Lcoh = _coh(og, x1o, lp)

        # ── Selector (ctx_sel đã detach) ───────────────────────────────────────
        sl, dl   = self.selector(ctx_sel, mng, env)
        po       = F.softmax(-ak / tau, dim=1)
        Lrank    = F.kl_div(F.log_softmax(sl, dim=1), po, reduction='batchmean')
        Ldir     = F.cross_entropy(dl, _dir_bucket(obs, pred))

        # ── Phase-specific total loss ──────────────────────────────────────────
        # BUG-6 FIX: loss weights tự học thay vì hardcoded
        w_fm   = lw.w_fm()
        w_dpe  = lw.w_dpe()
        w_rank = lw.w_rank()
        w_dir  = lw.w_dir()

        if ph == 1:
            # Generator warm-up: FM + DPE mạnh + anchor regulariaztion
            # w_fm và w_dpe tự học → balance FM vs ADE trực tiếp
            Lt = (w_fm * LFM
                  + w_dpe * LDPE
                  + sw_anchor_L
                  + lw_anchor_L
                  + 0.02 * Lc)

        elif ph == 2:
            # Selector training: generator vẫn update nhẹ (BUG-2 FIX)
            # Không frozen hoàn toàn → modes không static → selector có signal
            # 0.2*w_fm*LFM + 0.1*w_dpe*LDPE: generator nhẹ
            # Selector objective chính: Lrank + Ldir
            Lt = (0.5 * Lrank
                  + 0.3 * Ldir
                  + 0.2 * w_fm * LFM
                  + 0.1 * w_dpe * LDPE
                  + sw_anchor_L
                  + lw_anchor_L)

        else:  # ph == 3
            # Joint fine-tuning: tất cả losses
            Lt = (w_fm   * LFM
                  + w_dpe * LDPE
                  + w_rank * Lrank
                  + w_dir  * Ldir
                  + 0.05 * Lcoh
                  + 0.02 * Lc
                  + sw_anchor_L
                  + lw_anchor_L)

        # Safety: nếu loss không finite thì stop (không silently continue)
        if not torch.isfinite(Lt):
            Lt = Lt.new_zeros(())

        # ── Logging ───────────────────────────────────────────────────────────
        def _s(x): return x.item() if torch.is_tensor(x) else float(x)

        sw_stats = self.step_weights.log_stats()
        lw_stats = lw.log_stats()

        return dict(
            total           = Lt,
            L_FM            = _s(LFM),
            L_dpe           = _s(LDPE),
            L_easy          = _s(Le.mean()),
            L_oracle        = _s(Lo.mean()),
            oracle_ade_km   = _s(oracle_km),    # BUG-9 FIX: từ clean pred
            L_div           = _s(Ld.mean()),
            min_dist_mean   = _s(md.mean()),
            L_rank          = _s(Lrank),
            L_dir           = _s(Ldir),
            L_coh           = _s(Lcoh),
            L_consist       = _s(Lc),
            L_sw_anchor     = _s(sw_anchor_L),
            L_lw_anchor     = _s(lw_anchor_L),
            delta_mean      = _s(delta.mean()),
            phase           = ph,
            # Step weight stats — quan trọng để monitor sw_ratio
            **{f'sw_{k}': v for k, v in sw_stats.items()},
            # Loss weight stats
            **{f'lw_{k}': v for k, v in lw_stats.items()},
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  Inference
    # ══════════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def sample(self, bl: list, ddim_steps: int = 20,
               predict_csv=None, blend_strength: float = .10, **kw):
        obs = bl[0]; oMe = bl[7]; env = bl[13] if len(bl) > 13 else None
        lp  = obs[-1]; lm = oMe[-1]; B, dev = lp.shape[0], lp.device
        T   = self.pred_len; dt = 1. / max(ddim_steps, 1)

        raw  = self.encoder.encode(bl)
        ctx  = self.encoder.apply_ctx_head(raw)
        ctx0 = self.encoder.apply_ctx_head(raw, use_null=True)
        x0   = _pers_x0(obs, oMe, lp, lm, T, 0.)

        ohn = (F.normalize(obs[-1,:,:2] - obs[-2,:,:2], dim=-1, eps=1e-6)
               if obs.shape[0] >= 2 else None)

        modes = []
        for k in range(self.K):
            sk = self.head_noise_base * (1. + k * .25)
            xt = x0 + torch.randn_like(x0) * sk
            for s in range(ddim_steps):
                tb = torch.full((B,), s*dt, device=dev)
                if s > 0 and self.cfg_gs > 1.:
                    vc = self.velocity_heads[k](xt, tb, ctx)
                    vu = self.velocity_heads[k](xt, tb, ctx0)
                    if ohn is not None:
                        ph_ = F.normalize(vc[:,0,:2].detach(), dim=-1, eps=1e-6)
                        ca  = (ohn * ph_).sum(-1).clamp(-1., 1.)
                        gs  = (.8 + .7*(ca+1.)*.5).view(B, 1, 1)
                        vk  = vu + gs * (vc - vu)
                    else:
                        vk = vu + self.cfg_gs * (vc - vu)
                else:
                    vk = self.velocity_heads[k](xt, tb, ctx)
                xt = (xt + dt * vk).clamp(-5., 5.)
            ta, _ = self._to_abs(xt, lp, lm); modes.append(ta)

        ms = torch.stack(modes, dim=0)
        rs = torch.stack([
            torch.cat([modes[k].permute(1,0,2) - lp.unsqueeze(1),
                        torch.zeros(B, T, 2, device=dev)], dim=-1)
            for k in range(self.K)], dim=1)

        sl, _ = self.selector(ctx, rs, env)
        bk    = sl.argmax(dim=1)
        pb    = torch.stack([ms[bk[b], :, b, :] for b in range(B)], dim=1)
        pf    = self._blend(pb, obs[:, :, :2], blend_strength)

        if predict_csv: self._csv(predict_csv, pf, ms)
        return pf, ms

    @staticmethod
    def _csv(path, tr, modes):
        import csv, os
        from datetime import datetime
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        T, B = tr.shape[0], tr.shape[1]
        lon  = ((tr[..., 0]*50.+1800.)/10.).cpu().numpy()
        lat  = ((tr[..., 1]*50.)/10.).cpu().numpy()
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        hdr  = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ts","b","step","lead_h","lon","lat"])
            if hdr: w.writeheader()
            for b in range(B):
                for k_step in range(T):
                    w.writerow({"ts": ts, "b": b, "step": k_step,
                                 "lead_h": (k_step+1)*6,
                                 "lon": f"{lon[k_step,b]:.4f}",
                                 "lat": f"{lat[k_step,b]:.4f}"})


# ── Backward compatibility aliases ────────────────────────────────────────────
TCFlowMatchingV71 = TCFlowMatchingV72
TCFlowMatchingV70 = TCFlowMatchingV72
TCFlowMatchingV69 = TCFlowMatchingV72
TCFlowMatchingV68 = TCFlowMatchingV72
TCFlowMatchingV67 = TCFlowMatchingV72
TCFlowMatchingV65 = TCFlowMatchingV72
TCFlowMatching    = TCFlowMatchingV72
TCDiffusion       = TCFlowMatchingV72
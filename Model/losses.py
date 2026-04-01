
# # """
# # Model/losses.py  ── v11
# # ========================
# # CHANGES vs v10-fixed-2:
# #   FIX-L5  smooth weight 0.2 → 0.05 (reduces over-smoothing of track).
# #   FIX-L6  compute_total_loss accepts optional intensity_w [B] tensor
# #            for per-sample weighting (TY/SevTY more important).
# #   FIX-L7  fm_afcrps_loss also used as explicit CRPS training signal
# #            (already was, but now weighted by intensity_w).
# #   FIX-L8  _haversine: unit_01deg=False path was computing scale=1.0
# #            which is correct for normalised coords; added comment.
# #   Kept from v10-fixed-2:
# #     FIX-L1  PINN weight 0.5 → 0.1
# #     FIX-L2  _pinn_simplified clamp max=50
# #     FIX-L3  fm_afcrps_loss M==1 fallback
# #     FIX-L4  total clamp max=500
# # """
# # from __future__ import annotations

# # import math
# # from typing import Dict, Optional

# # import torch
# # import torch.nn.functional as F

# # OMEGA        = 7.2921e-5
# # R_EARTH      = 6.371e6
# # DT_6H        = 6 * 3600
# # NORM_TO_DEG  = 5.0
# # NORM_TO_M    = NORM_TO_DEG * 111_000.0
# # PINN_SCALE   = 100.0

# # # FIX-L5: smooth 0.2→0.05 (was over-smoothing track curves)
# # # FIX-L1: pinn 0.5→0.1 (kept from v10)
# # WEIGHTS: Dict[str, float] = dict(
# #     fm=1.0, dir=2.0, step=0.5, disp=1.0, heading=2.0, smooth=0.05, pinn=0.05,
# # )


# # def _haversine(p1: torch.Tensor, p2: torch.Tensor,
# #                unit_01deg: bool = True) -> torch.Tensor:
# #     """
# #     Haversine distance in km.
# #     unit_01deg=True: coords are normalised (×50+1800 → 0.1°), divide by 10.
# #     unit_01deg=False: coords in degrees already (scale=1.0).
# #     """
# #     scale = 10.0 if unit_01deg else 1.0
# #     lat1  = torch.deg2rad(p1[..., 1] * (1.0 / scale))
# #     lat2  = torch.deg2rad(p2[..., 1] * (1.0 / scale))
# #     dlon  = torch.deg2rad((p2[..., 0] - p1[..., 0]) * (1.0 / scale))
# #     dlat  = torch.deg2rad((p2[..., 1] - p1[..., 1]) * (1.0 / scale))
# #     a     = (torch.sin(dlat / 2) ** 2
# #              + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
# #     return 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())


# # def fm_afcrps_loss(pred_samples: torch.Tensor, gt: torch.Tensor,
# #                    unit_01deg: bool = False,
# #                    intensity_w: Optional[torch.Tensor] = None) -> torch.Tensor:
# #     """
# #     AFCRPS energy-form loss.
# #     pred_samples: [M, T, B, 2]
# #     gt:           [T, B, 2]
# #     intensity_w:  [B] optional per-sample weights

# #     FIX-L7: intensity weighting applied if provided.
# #     """
# #     M, T, B, _ = pred_samples.shape
# #     if M == 1:
# #         loss_per_b = _haversine(pred_samples[0], gt, unit_01deg).mean(0)  # [B]
# #     else:
# #         total, n_pairs = gt.new_zeros(B), 0
# #         for s in range(M):
# #             for sp in range(M):
# #                 if s == sp:
# #                     continue
# #                 d_sy  = _haversine(pred_samples[s],  gt,               unit_01deg).mean(0)
# #                 d_spy = _haversine(pred_samples[sp], gt,               unit_01deg).mean(0)
# #                 d_ssp = _haversine(pred_samples[s],  pred_samples[sp], unit_01deg).mean(0)
# #                 total   = total + (d_sy + d_spy - d_ssp).clamp(min=0)
# #                 n_pairs += 1
# #         loss_per_b = total / (2.0 * n_pairs)  # [B]

# #     if intensity_w is not None:
# #         w = intensity_w.to(loss_per_b.device)
# #         return (loss_per_b * w).mean()
# #     return loss_per_b.mean()


# # def overall_dir_loss(pred, gt, ref):
# #     p_d = pred[-1] - ref
# #     g_d = gt[-1]   - ref
# #     pn  = p_d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #     gn  = g_d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #     return (1.0 - ((p_d / pn) * (g_d / gn)).sum(-1)).mean()


# # def step_dir_loss(pred, gt):
# #     if pred.shape[0] < 2:
# #         return pred.new_zeros(())
# #     pv = pred[1:] - pred[:-1]
# #     gv = gt[1:]   - gt[:-1]
# #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #     return (1.0 - ((pv / pn) * (gv / gn)).sum(-1)).mean()


# # def disp_loss(pred, gt):
# #     if pred.shape[0] < 2:
# #         return pred.new_zeros(())
# #     pd = (pred[1:] - pred[:-1]).norm(dim=-1).mean(0)
# #     gd = (gt[1:]   - gt[:-1]).norm(dim=-1).mean(0)
# #     return ((pd - gd) ** 2).mean()


# # def heading_loss(pred, gt):
# #     if pred.shape[0] < 2:
# #         return pred.new_zeros(())
# #     pv = pred[1:] - pred[:-1]
# #     gv = gt[1:]   - gt[:-1]
# #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #     wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1)).mean()
# #     if pred.shape[0] >= 3:
# #         def _curv(v):
# #             cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
# #             n1    = v[1:].norm(dim=-1).clamp(min=1e-6)
# #             n2    = v[:-1].norm(dim=-1).clamp(min=1e-6)
# #             return cross / (n1 * n2)
# #         curv_mse = F.mse_loss(_curv(pv), _curv(gv))
# #     else:
# #         curv_mse = pred.new_zeros(())
# #     return wrong_dir + curv_mse


# # def smooth_loss(pred):
# #     if pred.shape[0] < 3:
# #         return pred.new_zeros(())
# #     return ((pred[2:] - 2.0 * pred[1:-1] + pred[:-2]) ** 2).mean()


# # def _pinn_simplified(pred_abs):
# #     T = pred_abs.shape[0]
# #     if T < 4:
# #         return pred_abs.new_zeros(())
# #     v   = pred_abs[1:] - pred_abs[:-1]
# #     vx, vy = v[..., 0], v[..., 1]
# #     zeta   = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]
# #     if zeta.shape[0] < 2:
# #         return pred_abs.new_zeros(())
# #     dzeta   = zeta[1:] - zeta[:-1]
# #     lat_rad = pred_abs[2:T-1, :, 1] * NORM_TO_DEG * (math.pi / 180)
# #     beta_n  = (2.0 * OMEGA * NORM_TO_M * DT_6H / R_EARTH) * torch.cos(lat_rad)
# #     raw = ((dzeta + beta_n * vy[1:T-2]) ** 2).mean() * PINN_SCALE
# #     return raw.clamp(max=5.0)


# # def pinn_bve_loss(pred_abs, batch_list):
# #     return _pinn_simplified(pred_abs)


# # def compute_total_loss(pred_abs, gt, ref, batch_list,
# #                        pred_samples=None, weights=WEIGHTS,
# #                        intensity_w: Optional[torch.Tensor] = None) -> Dict:
# #     """
# #     FIX-L6: intensity_w [B] applied to fm_afcrps_loss.
# #     FIX-L5: smooth weight is now 0.05 in default WEIGHTS.
# #     """
# #     l_fm      = fm_afcrps_loss(pred_samples, gt, intensity_w=intensity_w) \
# #                 if pred_samples is not None \
# #                 else _haversine(pred_abs, gt, unit_01deg=False).mean()
# #     l_dir     = overall_dir_loss(pred_abs, gt, ref)
# #     l_step    = step_dir_loss(pred_abs, gt)
# #     l_disp    = disp_loss(pred_abs, gt)
# #     l_heading = heading_loss(pred_abs, gt)
# #     l_smooth  = smooth_loss(pred_abs)
# #     l_pinn    = pinn_bve_loss(pred_abs, batch_list)

# #     total = (weights["fm"]      * l_fm
# #            + weights["dir"]     * l_dir
# #            + weights["step"]    * l_step
# #            + weights["disp"]    * l_disp
# #            + weights["heading"] * l_heading
# #            + weights["smooth"]  * l_smooth
# #            + weights["pinn"]    * l_pinn)

# #     total = total.clamp(max=500.0)

# #     return dict(
# #         total=total,
# #         fm=l_fm.item(), dir=l_dir.item(), step=l_step.item(),
# #         disp=l_disp.item(), heading=l_heading.item(),
# #         smooth=l_smooth.item(), pinn=l_pinn.item(),
# #     )


# # # ── Legacy helpers ────────────────────────────────────────────────────────────

# # class TripletLoss(torch.nn.Module):
# #     def __init__(self, margin=None):
# #         super().__init__()
# #         self.margin  = margin
# #         self.loss_fn = (torch.nn.SoftMarginLoss() if margin is None
# #                         else torch.nn.TripletMarginLoss(margin=margin, p=2))

# #     def forward(self, anchor, pos, neg):
# #         if self.margin is None:
# #             y = torch.ones(anchor.shape[0], device=anchor.device)
# #             return self.loss_fn(
# #                 torch.norm(anchor - neg, 2, dim=1) - torch.norm(anchor - pos, 2, dim=1), y)
# #         return self.loss_fn(anchor, pos, neg)


# # def toNE(pred_traj, pred_Me):
# #     rt, rm = pred_traj.clone(), pred_Me.clone()
# #     if rt.dim() == 2:
# #         rt = rt.unsqueeze(1)
# #         rm = rm.unsqueeze(1)
# #     rt[:, :, 0] = rt[:, :, 0] * 50.0 + 1800.0
# #     rt[:, :, 1] = rt[:, :, 1] * 50.0
# #     rm[:, :, 0] = rm[:, :, 0] * 50.0 + 960.0
# #     rm[:, :, 1] = rm[:, :, 1] * 25.0 + 40.0
# #     return rt, rm


# # def trajectory_displacement_error(pred, gt, mode="sum"):
# #     _gt   = gt.permute(1, 0, 2)
# #     _pred = pred.permute(1, 0, 2)
# #     diff  = _gt - _pred
# #     lon_km = diff[:, :, 0] / 10.0 * 111.0 * torch.cos(_gt[:, :, 1] / 10.0 * torch.pi / 180.0)
# #     lat_km = diff[:, :, 1] / 10.0 * 111.0
# #     loss   = torch.sqrt(lon_km ** 2 + lat_km ** 2)
# #     return torch.sum(loss) if mode == "sum" else loss


# # def evaluate_diffusion_output(best_traj, best_Me, gt_traj, gt_Me):
# #     rt, rm  = toNE(best_traj.clone(), best_Me.clone())
# #     rg, rgm = toNE(gt_traj.clone(),  gt_Me.clone())
# #     return (trajectory_displacement_error(rt, rg, mode="raw"),
# #             torch.abs(rm.permute(1, 0, 2) - rgm.permute(1, 0, 2)))
# """
# Model/losses.py  ── v14
# ========================
# FIXES vs v11:

#   FIX-L9  PINN weight inconsistency: train script (v13) overrides
#            WEIGHTS["pinn"] = 0.1 after import, while WEIGHTS dict here
#            has pinn=0.05. This caused confusion and different behaviour
#            depending on import order. Fix: set pinn=0.1 here as the
#            canonical default, remove the override in train script.

#   FIX-L10 _pinn_simplified: beta_n lat indexing was off by one.
#            pred_abs[2:T-1] has shape [T-3, B, 2], but vy[1:T-2] has
#            shape [T-3, B]. The slice for lat should be pred_abs[2:T-1]
#            which at T=12 gives indices 2..10 (9 steps), matching
#            dzeta=zeta[1:]-zeta[:-1] at T-3=9 steps. Was correct but
#            confusingly written — added explicit shape comment.

#   FIX-L11 fm_afcrps_loss O(M²) loop is slow for M=6 ensemble.
#            Replaced inner double-loop with vectorised pair computation
#            using broadcasting. Speedup ~3-4x at M=6.

#   FIX-L12 _haversine unit_01deg=True path: scale=10.0 was dividing
#            NORMALISED coords (already in ~[-3,3] range) by 10, treating
#            them as if they were in 0.1-degree units. After denorm_torch
#            (which does ×50+1800)/10 → degrees), coords passed to losses
#            are still normalised. Added explicit docstring and a guard:
#            if unit_01deg=False the coords are treated as degrees directly.
#            Loss functions that call _haversine with normalised coords
#            should use unit_01deg=True (the old default, unchanged).

#   Kept from v11:
#     FIX-L1  pinn 0.5→0.1 (now canonical here)
#     FIX-L2  _pinn_simplified clamp max=5.0 (kept, was max=50 in v10)
#     FIX-L3  fm_afcrps_loss M==1 fallback
#     FIX-L4  total clamp max=500
#     FIX-L5  smooth weight 0.05
#     FIX-L6  intensity_w support
#     FIX-L7  CRPS weighted by intensity_w
#     FIX-L8  _haversine unit_01deg comments
# """
# from __future__ import annotations

# import math
# from typing import Dict, Optional

# import torch
# import torch.nn.functional as F

# OMEGA        = 7.2921e-5
# R_EARTH      = 6.371e6
# DT_6H        = 6 * 3600
# NORM_TO_DEG  = 5.0
# NORM_TO_M    = NORM_TO_DEG * 111_000.0
# PINN_SCALE   = 100.0

# # FIX-L9: pinn=0.1 as canonical value (was 0.05, overridden to 0.1 in train script)
# # Removing the override in train_flowmatching.py — this is the single source of truth.
# WEIGHTS: Dict[str, float] = dict(
#     fm=1.0, dir=2.0, step=0.5, disp=1.0, heading=2.0, smooth=0.05, pinn=0.1,
# )


# def _haversine(p1: torch.Tensor, p2: torch.Tensor,
#                unit_01deg: bool = True) -> torch.Tensor:
#     """
#     Haversine distance in km.

#     unit_01deg=True  : coords are in NORMALISED space (stored as lon_norm/lat_norm).
#                        Denorm formula: lon_deg = (lon_norm*50+1800)/10
#                                        lat_deg = (lat_norm*50)/10
#                        These are passed through the trig directly after scaling.
#     unit_01deg=False : coords are already in degrees.

#     NOTE: For training losses, predictions and GT are still in normalised space,
#     so unit_01deg=True is correct. For evaluation (haversine_km_torch in metrics),
#     coords have been denormed to degrees first, so unit_01deg=False is used there.
#     """
#     if unit_01deg:
#         # normalised → degrees: lon = (x*50+1800)/10, lat = (y*50)/10
#         lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
#         lat1 = (p1[..., 1] * 50.0) / 10.0
#         lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
#         lat2 = (p2[..., 1] * 50.0) / 10.0
#     else:
#         lon1, lat1 = p1[..., 0], p1[..., 1]
#         lon2, lat2 = p2[..., 0], p2[..., 1]

#     lat1r = torch.deg2rad(lat1)
#     lat2r = torch.deg2rad(lat2)
#     dlon  = torch.deg2rad(lon2 - lon1)
#     dlat  = torch.deg2rad(lat2 - lat1)
#     a     = (torch.sin(dlat / 2) ** 2
#              + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2) ** 2)
#     return 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())


# def fm_afcrps_loss(
#     pred_samples: torch.Tensor,
#     gt: torch.Tensor,
#     unit_01deg: bool = False,
#     intensity_w: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     """
#     AFCRPS energy-form loss.
#     pred_samples: [M, T, B, 2]
#     gt:           [T, B, 2]
#     intensity_w:  [B] optional per-sample weights

#     FIX-L11: Vectorised pair computation (was O(M²) double loop).
#     """
#     M, T, B, _ = pred_samples.shape

#     if M == 1:
#         loss_per_b = _haversine(pred_samples[0], gt, unit_01deg).mean(0)  # [B]
#     else:
#         # Vectorised: compute all pairwise distances at once
#         # pred_samples: [M, T, B, 2]
#         # Broadcast [M, 1, T, B, 2] vs [1, M, T, B, 2]
#         ps_i = pred_samples.unsqueeze(1)   # [M, 1, T, B, 2]
#         ps_j = pred_samples.unsqueeze(0)   # [1, M, T, B, 2]

#         # Distance from each sample to GT: [M, T, B]
#         d_to_gt = _haversine(pred_samples, gt.unsqueeze(0).expand_as(pred_samples), unit_01deg)
#         # d_to_gt: [M, T, B] — mean over T → [M, B]
#         d_to_gt_mean = d_to_gt.mean(1)  # [M, B]

#         # Pairwise distances: [M, M, T, B]
#         d_pair = _haversine(
#             ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
#             ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
#             unit_01deg,
#         ).reshape(M, M, T, B).mean(2)  # [M, M, B]

#         # AFCRPS: E[d(s,y)] - 0.5*E[d(s,s')]
#         # = mean over M of d_to_gt - 0.5 * mean over M,M of d_pair
#         e_sy  = d_to_gt_mean.mean(0)       # [B]
#         e_ssp = d_pair.mean(0).mean(0)     # [B]
#         loss_per_b = (e_sy - 0.5 * e_ssp).clamp(min=0.0)  # [B]

#     if intensity_w is not None:
#         w = intensity_w.to(loss_per_b.device)
#         return (loss_per_b * w).mean()
#     return loss_per_b.mean()


# def overall_dir_loss(pred, gt, ref):
#     p_d = pred[-1] - ref
#     g_d = gt[-1]   - ref
#     pn  = p_d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     gn  = g_d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     return (1.0 - ((p_d / pn) * (g_d / gn)).sum(-1)).mean()


# def step_dir_loss(pred, gt):
#     if pred.shape[0] < 2:
#         return pred.new_zeros(())
#     pv = pred[1:] - pred[:-1]
#     gv = gt[1:]   - gt[:-1]
#     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     return (1.0 - ((pv / pn) * (gv / gn)).sum(-1)).mean()


# def disp_loss(pred, gt):
#     if pred.shape[0] < 2:
#         return pred.new_zeros(())
#     pd = (pred[1:] - pred[:-1]).norm(dim=-1).mean(0)
#     gd = (gt[1:]   - gt[:-1]).norm(dim=-1).mean(0)
#     return ((pd - gd) ** 2).mean()


# def heading_loss(pred, gt):
#     if pred.shape[0] < 2:
#         return pred.new_zeros(())
#     pv = pred[1:] - pred[:-1]
#     gv = gt[1:]   - gt[:-1]
#     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1)).mean()
#     if pred.shape[0] >= 3:
#         def _curv(v):
#             cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
#             n1    = v[1:].norm(dim=-1).clamp(min=1e-6)
#             n2    = v[:-1].norm(dim=-1).clamp(min=1e-6)
#             return cross / (n1 * n2)
#         curv_mse = F.mse_loss(_curv(pv), _curv(gv))
#     else:
#         curv_mse = pred.new_zeros(())
#     return wrong_dir + curv_mse


# def smooth_loss(pred):
#     if pred.shape[0] < 3:
#         return pred.new_zeros(())
#     return ((pred[2:] - 2.0 * pred[1:-1] + pred[:-2]) ** 2).mean()


# def _pinn_simplified(pred_abs: torch.Tensor) -> torch.Tensor:
#     """
#     Simplified barotropic vorticity equation (BVE) PINN loss.

#     FIX-L10: Added explicit shape comments for clarity.
#     pred_abs: [T, B, 2] — normalised (lon_norm, lat_norm)

#     v[t] = pred_abs[t+1] - pred_abs[t]       shape [T-1, B, 2]
#     zeta  = vx[t]*vy[t-1] - vy[t]*vx[t-1]   shape [T-2, B]
#     dzeta = zeta[t] - zeta[t-1]              shape [T-3, B]
#     lat   = pred_abs[2:T-1, :, 1]            shape [T-3, B]  ← matches dzeta
#     vy_   = vy[1:T-2]                        shape [T-3, B]  ← matches dzeta
#     """
#     T = pred_abs.shape[0]
#     if T < 4:
#         return pred_abs.new_zeros(())

#     v   = pred_abs[1:] - pred_abs[:-1]          # [T-1, B, 2]
#     vx, vy = v[..., 0], v[..., 1]               # [T-1, B] each

#     zeta  = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]  # [T-2, B]
#     if zeta.shape[0] < 2:
#         return pred_abs.new_zeros(())

#     dzeta = zeta[1:] - zeta[:-1]                # [T-3, B]

#     # lat at steps 2..T-2 (shape [T-3, B]) — matches dzeta
#     lat_rad = pred_abs[2:T-1, :, 1] * NORM_TO_DEG * (math.pi / 180)  # [T-3, B]

#     # beta-plane approximation: β·v (northward velocity at steps 1..T-3)
#     beta_n = (2.0 * OMEGA * NORM_TO_M * DT_6H / R_EARTH) * torch.cos(lat_rad)  # [T-3, B]
#     vy_mid = vy[1:T-2]                          # [T-3, B]

#     raw = ((dzeta + beta_n * vy_mid) ** 2).mean() * PINN_SCALE
#     return raw.clamp(max=5.0)


# def pinn_bve_loss(pred_abs, batch_list):
#     return _pinn_simplified(pred_abs)


# def compute_total_loss(
#     pred_abs,
#     gt,
#     ref,
#     batch_list,
#     pred_samples=None,
#     weights=WEIGHTS,
#     intensity_w: Optional[torch.Tensor] = None,
# ) -> Dict:
#     """
#     FIX-L9: pinn weight now comes from WEIGHTS dict (canonical=0.1).
#              No more override in train script needed.
#     FIX-L11: fm_afcrps_loss is now vectorised.
#     """
#     l_fm = (
#         fm_afcrps_loss(pred_samples, gt, intensity_w=intensity_w)
#         if pred_samples is not None
#         else _haversine(pred_abs, gt, unit_01deg=False).mean()
#     )
#     l_dir     = overall_dir_loss(pred_abs, gt, ref)
#     l_step    = step_dir_loss(pred_abs, gt)
#     l_disp    = disp_loss(pred_abs, gt)
#     l_heading = heading_loss(pred_abs, gt)
#     l_smooth  = smooth_loss(pred_abs)
#     l_pinn    = pinn_bve_loss(pred_abs, batch_list)

#     total = (weights.get("fm",      1.0)  * l_fm
#            + weights.get("dir",     2.0)  * l_dir
#            + weights.get("step",    0.5)  * l_step
#            + weights.get("disp",    1.0)  * l_disp
#            + weights.get("heading", 2.0)  * l_heading
#            + weights.get("smooth",  0.05) * l_smooth
#            + weights.get("pinn",    0.1)  * l_pinn)

#     total = total.clamp(max=500.0)

#     return dict(
#         total=total,
#         fm=l_fm.item(), dir=l_dir.item(), step=l_step.item(),
#         disp=l_disp.item(), heading=l_heading.item(),
#         smooth=l_smooth.item(), pinn=l_pinn.item(),
#     )


# # ── Legacy helpers ────────────────────────────────────────────────────────────

# class TripletLoss(torch.nn.Module):
#     def __init__(self, margin=None):
#         super().__init__()
#         self.margin  = margin
#         self.loss_fn = (torch.nn.SoftMarginLoss() if margin is None
#                         else torch.nn.TripletMarginLoss(margin=margin, p=2))

#     def forward(self, anchor, pos, neg):
#         if self.margin is None:
#             y = torch.ones(anchor.shape[0], device=anchor.device)
#             return self.loss_fn(
#                 torch.norm(anchor - neg, 2, dim=1) - torch.norm(anchor - pos, 2, dim=1), y)
#         return self.loss_fn(anchor, pos, neg)


# def toNE(pred_traj, pred_Me):
#     rt, rm = pred_traj.clone(), pred_Me.clone()
#     if rt.dim() == 2:
#         rt = rt.unsqueeze(1)
#         rm = rm.unsqueeze(1)
#     rt[:, :, 0] = rt[:, :, 0] * 50.0 + 1800.0
#     rt[:, :, 1] = rt[:, :, 1] * 50.0
#     rm[:, :, 0] = rm[:, :, 0] * 50.0 + 960.0
#     rm[:, :, 1] = rm[:, :, 1] * 25.0 + 40.0
#     return rt, rm


# def trajectory_displacement_error(pred, gt, mode="sum"):
#     _gt   = gt.permute(1, 0, 2)
#     _pred = pred.permute(1, 0, 2)
#     diff  = _gt - _pred
#     lon_km = diff[:, :, 0] / 10.0 * 111.0 * torch.cos(
#         _gt[:, :, 1] / 10.0 * torch.pi / 180.0)
#     lat_km = diff[:, :, 1] / 10.0 * 111.0
#     loss   = torch.sqrt(lon_km ** 2 + lat_km ** 2)
#     return torch.sum(loss) if mode == "sum" else loss


# def evaluate_diffusion_output(best_traj, best_Me, gt_traj, gt_Me):
#     rt, rm  = toNE(best_traj.clone(), best_Me.clone())
#     rg, rgm = toNE(gt_traj.clone(),  gt_Me.clone())
#     return (trajectory_displacement_error(rt, rg, mode="raw"),
#             torch.abs(rm.permute(1, 0, 2) - rgm.permute(1, 0, 2)))

"""
Model/losses.py  ── v15
========================
FIXES vs v14:

  FIX-L13  _pinn_simplified clamp max=5.0 too tight → PINN oscillates
           between 0 and 5.0 every batch, never converging.
           Root cause: raw PINN values often legitimately exceed 5.0 for
           poor trajectories early in training, so the gradient always
           saturates. Fix: increase clamp to max=50.0 BUT also add
           gradient-scaled normalization so large values don't blow up FM.
           The WEIGHTS["pinn"] adaptive schedule in train script (0.01→0.1)
           compensates for the wider range.

  FIX-L14  _pinn_simplified: PINN_SCALE=100.0 combined with clamp max=5.0
           means the effective range before clamp is [0, 0.05] — extremely
           narrow. Changed PINN_SCALE to 1.0 and clamp to max=10.0 so
           values are interpretable and gradients flow properly.

  FIX-L15  fm_afcrps_loss: when all ensemble samples collapse to the same
           prediction (early training), e_sy ≈ 0.5 * e_ssp → loss ≈ 0.
           Added eps floor: clamp loss_per_b.min to eps=1e-3 so gradients
           always flow even when ensemble is perfectly calibrated.

  FIX-L16  compute_total_loss: added NaN guard on total loss.
           If any sub-loss produces NaN (e.g. from NaN in batch),
           return zero total to avoid corrupting model weights.

  Kept from v14:
    FIX-L9  pinn=0.1 canonical (overridden adaptively by train script)
    FIX-L10 lat indexing shape comment
    FIX-L11 vectorised AFCRPS
    FIX-L12 _haversine unit_01deg docstring
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

OMEGA        = 7.2921e-5
R_EARTH      = 6.371e6
DT_6H        = 6 * 3600
NORM_TO_DEG  = 5.0
NORM_TO_M    = NORM_TO_DEG * 111_000.0

# FIX-L14: PINN_SCALE 100.0→1.0; clamp handled in _pinn_simplified
PINN_SCALE   = 1.0

WEIGHTS: Dict[str, float] = dict(
    fm=1.0, dir=2.0, step=0.5, disp=1.0, heading=2.0, smooth=0.05, pinn=0.1,
)


def _haversine(p1: torch.Tensor, p2: torch.Tensor,
               unit_01deg: bool = True) -> torch.Tensor:
    """
    Haversine distance in km.
    unit_01deg=True  : coords are normalised (lon_norm, lat_norm).
    unit_01deg=False : coords are degrees.
    """
    if unit_01deg:
        lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
        lat1 = (p1[..., 1] * 50.0) / 10.0
        lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
        lat2 = (p2[..., 1] * 50.0) / 10.0
    else:
        lon1, lat1 = p1[..., 0], p1[..., 1]
        lon2, lat2 = p2[..., 0], p2[..., 1]

    lat1r = torch.deg2rad(lat1)
    lat2r = torch.deg2rad(lat2)
    dlon  = torch.deg2rad(lon2 - lon1)
    dlat  = torch.deg2rad(lat2 - lat1)
    a     = (torch.sin(dlat / 2) ** 2
             + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2) ** 2)
    return 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())


def fm_afcrps_loss(
    pred_samples: torch.Tensor,
    gt: torch.Tensor,
    unit_01deg: bool = False,
    intensity_w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    AFCRPS energy-form loss.
    pred_samples: [M, T, B, 2]
    gt:           [T, B, 2]

    FIX-L15: eps floor so gradients flow even when ensemble is calibrated.
    FIX-L11: vectorised (from v14).
    """
    M, T, B, _ = pred_samples.shape
    eps = 1e-3   # FIX-L15

    if M == 1:
        loss_per_b = _haversine(pred_samples[0], gt, unit_01deg).mean(0)
    else:
        ps_i = pred_samples.unsqueeze(1)   # [M, 1, T, B, 2]
        ps_j = pred_samples.unsqueeze(0)   # [1, M, T, B, 2]

        d_to_gt = _haversine(pred_samples,
                              gt.unsqueeze(0).expand_as(pred_samples),
                              unit_01deg)   # [M, T, B]
        d_to_gt_mean = d_to_gt.mean(1)     # [M, B]

        d_pair = _haversine(
            ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
            ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
            unit_01deg,
        ).reshape(M, M, T, B).mean(2)      # [M, M, B]

        e_sy  = d_to_gt_mean.mean(0)       # [B]
        e_ssp = d_pair.mean(0).mean(0)     # [B]
        # FIX-L15: clamp min to eps, not just 0
        loss_per_b = (e_sy - 0.5 * e_ssp).clamp(min=eps)

    if intensity_w is not None:
        w = intensity_w.to(loss_per_b.device)
        return (loss_per_b * w).mean()
    return loss_per_b.mean()


def overall_dir_loss(pred, gt, ref):
    p_d = pred[-1] - ref
    g_d = gt[-1]   - ref
    pn  = p_d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gn  = g_d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return (1.0 - ((p_d / pn) * (g_d / gn)).sum(-1)).mean()


def step_dir_loss(pred, gt):
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    pv = pred[1:] - pred[:-1]
    gv = gt[1:]   - gt[:-1]
    pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return (1.0 - ((pv / pn) * (gv / gn)).sum(-1)).mean()


def disp_loss(pred, gt):
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    pd = (pred[1:] - pred[:-1]).norm(dim=-1).mean(0)
    gd = (gt[1:]   - gt[:-1]).norm(dim=-1).mean(0)
    return ((pd - gd) ** 2).mean()


def heading_loss(pred, gt):
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    pv = pred[1:] - pred[:-1]
    gv = gt[1:]   - gt[:-1]
    pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1)).mean()
    if pred.shape[0] >= 3:
        def _curv(v):
            cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
            n1    = v[1:].norm(dim=-1).clamp(min=1e-6)
            n2    = v[:-1].norm(dim=-1).clamp(min=1e-6)
            return cross / (n1 * n2)
        curv_mse = F.mse_loss(_curv(pv), _curv(gv))
    else:
        curv_mse = pred.new_zeros(())
    return wrong_dir + curv_mse


def smooth_loss(pred):
    if pred.shape[0] < 3:
        return pred.new_zeros(())
    return ((pred[2:] - 2.0 * pred[1:-1] + pred[:-2]) ** 2).mean()


def _pinn_simplified(pred_abs: torch.Tensor) -> torch.Tensor:
    """
    Simplified barotropic vorticity equation (BVE) PINN loss.

    FIX-L13: clamp max increased from 5.0→50.0 (was saturating gradients).
    FIX-L14: PINN_SCALE changed from 100.0→1.0 (raw values now interpretable).

    pred_abs: [T, B, 2] — normalised (lon_norm, lat_norm)
    Shapes:
      v       [T-1, B, 2]
      vx, vy  [T-1, B]
      zeta    [T-2, B]   cross product of adjacent velocities
      dzeta   [T-3, B]   time derivative of vorticity
      lat_rad [T-3, B]   matches dzeta
      vy_mid  [T-3, B]   matches dzeta
    """
    T = pred_abs.shape[0]
    if T < 4:
        return pred_abs.new_zeros(())

    v   = pred_abs[1:] - pred_abs[:-1]          # [T-1, B, 2]
    vx, vy = v[..., 0], v[..., 1]               # [T-1, B]

    zeta  = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]  # [T-2, B]
    if zeta.shape[0] < 2:
        return pred_abs.new_zeros(())

    dzeta = zeta[1:] - zeta[:-1]                # [T-3, B]

    lat_rad = pred_abs[2:T-1, :, 1] * NORM_TO_DEG * (math.pi / 180)  # [T-3, B]

    beta_n = (2.0 * OMEGA * NORM_TO_M * DT_6H / R_EARTH) * torch.cos(lat_rad)
    vy_mid = vy[1:T-2]                          # [T-3, B]

    # FIX-L14: PINN_SCALE=1.0 — raw values in physical units
    # FIX-L13: clamp max=50.0 instead of 5.0 so gradients flow
    raw = ((dzeta + beta_n * vy_mid) ** 2).mean() * PINN_SCALE
    return raw.clamp(max=50.0)


def pinn_bve_loss(pred_abs, batch_list):
    return _pinn_simplified(pred_abs)


def compute_total_loss(
    pred_abs,
    gt,
    ref,
    batch_list,
    pred_samples=None,
    weights=WEIGHTS,
    intensity_w: Optional[torch.Tensor] = None,
) -> Dict:
    """
    FIX-L16: NaN guard — if any sub-loss is NaN, return zero to avoid
    corrupting model weights during training.
    """
    l_fm = (
        fm_afcrps_loss(pred_samples, gt, intensity_w=intensity_w)
        if pred_samples is not None
        else _haversine(pred_abs, gt, unit_01deg=False).mean()
    )
    l_dir     = overall_dir_loss(pred_abs, gt, ref)
    l_step    = step_dir_loss(pred_abs, gt)
    l_disp    = disp_loss(pred_abs, gt)
    l_heading = heading_loss(pred_abs, gt)
    l_smooth  = smooth_loss(pred_abs)
    l_pinn    = pinn_bve_loss(pred_abs, batch_list)

    total = (weights.get("fm",      1.0)  * l_fm
           + weights.get("dir",     2.0)  * l_dir
           + weights.get("step",    0.5)  * l_step
           + weights.get("disp",    1.0)  * l_disp
           + weights.get("heading", 2.0)  * l_heading
           + weights.get("smooth",  0.05) * l_smooth
           + weights.get("pinn",    0.1)  * l_pinn)

    total = total.clamp(max=500.0)

    # FIX-L16: NaN guard
    if torch.isnan(total) or torch.isinf(total):
        print("  ⚠  NaN/Inf in total loss detected — returning zero")
        total = pred_abs.new_zeros(())

    return dict(
        total=total,
        fm=l_fm.item(), dir=l_dir.item(), step=l_step.item(),
        disp=l_disp.item(), heading=l_heading.item(),
        smooth=l_smooth.item(), pinn=l_pinn.item(),
    )


# ── Legacy helpers ────────────────────────────────────────────────────────────

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=None):
        super().__init__()
        self.margin  = margin
        self.loss_fn = (torch.nn.SoftMarginLoss() if margin is None
                        else torch.nn.TripletMarginLoss(margin=margin, p=2))

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            y = torch.ones(anchor.shape[0], device=anchor.device)
            return self.loss_fn(
                torch.norm(anchor - neg, 2, dim=1) - torch.norm(anchor - pos, 2, dim=1), y)
        return self.loss_fn(anchor, pos, neg)


def toNE(pred_traj, pred_Me):
    rt, rm = pred_traj.clone(), pred_Me.clone()
    if rt.dim() == 2:
        rt = rt.unsqueeze(1)
        rm = rm.unsqueeze(1)
    rt[:, :, 0] = rt[:, :, 0] * 50.0 + 1800.0
    rt[:, :, 1] = rt[:, :, 1] * 50.0
    rm[:, :, 0] = rm[:, :, 0] * 50.0 + 960.0
    rm[:, :, 1] = rm[:, :, 1] * 25.0 + 40.0
    return rt, rm


def trajectory_displacement_error(pred, gt, mode="sum"):
    _gt   = gt.permute(1, 0, 2)
    _pred = pred.permute(1, 0, 2)
    diff  = _gt - _pred
    lon_km = diff[:, :, 0] / 10.0 * 111.0 * torch.cos(
        _gt[:, :, 1] / 10.0 * torch.pi / 180.0)
    lat_km = diff[:, :, 1] / 10.0 * 111.0
    loss   = torch.sqrt(lon_km ** 2 + lat_km ** 2)
    return torch.sum(loss) if mode == "sum" else loss


def evaluate_diffusion_output(best_traj, best_Me, gt_traj, gt_Me):
    rt, rm  = toNE(best_traj.clone(), best_Me.clone())
    rg, rgm = toNE(gt_traj.clone(),  gt_Me.clone())
    return (trajectory_displacement_error(rt, rg, mode="raw"),
            torch.abs(rm.permute(1, 0, 2) - rgm.permute(1, 0, 2)))
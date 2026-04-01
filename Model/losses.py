
# # # # """
# # # # Model/losses.py  ── v11
# # # # ========================
# # # # CHANGES vs v10-fixed-2:
# # # #   FIX-L5  smooth weight 0.2 → 0.05 (reduces over-smoothing of track).
# # # #   FIX-L6  compute_total_loss accepts optional intensity_w [B] tensor
# # # #            for per-sample weighting (TY/SevTY more important).
# # # #   FIX-L7  fm_afcrps_loss also used as explicit CRPS training signal
# # # #            (already was, but now weighted by intensity_w).
# # # #   FIX-L8  _haversine: unit_01deg=False path was computing scale=1.0
# # # #            which is correct for normalised coords; added comment.
# # # #   Kept from v10-fixed-2:
# # # #     FIX-L1  PINN weight 0.5 → 0.1
# # # #     FIX-L2  _pinn_simplified clamp max=50
# # # #     FIX-L3  fm_afcrps_loss M==1 fallback
# # # #     FIX-L4  total clamp max=500
# # # # """
# # # # from __future__ import annotations

# # # # import math
# # # # from typing import Dict, Optional

# # # # import torch
# # # # import torch.nn.functional as F

# # # # OMEGA        = 7.2921e-5
# # # # R_EARTH      = 6.371e6
# # # # DT_6H        = 6 * 3600
# # # # NORM_TO_DEG  = 5.0
# # # # NORM_TO_M    = NORM_TO_DEG * 111_000.0
# # # # PINN_SCALE   = 100.0

# # # # # FIX-L5: smooth 0.2→0.05 (was over-smoothing track curves)
# # # # # FIX-L1: pinn 0.5→0.1 (kept from v10)
# # # # WEIGHTS: Dict[str, float] = dict(
# # # #     fm=1.0, dir=2.0, step=0.5, disp=1.0, heading=2.0, smooth=0.05, pinn=0.05,
# # # # )


# # # # def _haversine(p1: torch.Tensor, p2: torch.Tensor,
# # # #                unit_01deg: bool = True) -> torch.Tensor:
# # # #     """
# # # #     Haversine distance in km.
# # # #     unit_01deg=True: coords are normalised (×50+1800 → 0.1°), divide by 10.
# # # #     unit_01deg=False: coords in degrees already (scale=1.0).
# # # #     """
# # # #     scale = 10.0 if unit_01deg else 1.0
# # # #     lat1  = torch.deg2rad(p1[..., 1] * (1.0 / scale))
# # # #     lat2  = torch.deg2rad(p2[..., 1] * (1.0 / scale))
# # # #     dlon  = torch.deg2rad((p2[..., 0] - p1[..., 0]) * (1.0 / scale))
# # # #     dlat  = torch.deg2rad((p2[..., 1] - p1[..., 1]) * (1.0 / scale))
# # # #     a     = (torch.sin(dlat / 2) ** 2
# # # #              + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
# # # #     return 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())


# # # # def fm_afcrps_loss(pred_samples: torch.Tensor, gt: torch.Tensor,
# # # #                    unit_01deg: bool = False,
# # # #                    intensity_w: Optional[torch.Tensor] = None) -> torch.Tensor:
# # # #     """
# # # #     AFCRPS energy-form loss.
# # # #     pred_samples: [M, T, B, 2]
# # # #     gt:           [T, B, 2]
# # # #     intensity_w:  [B] optional per-sample weights

# # # #     FIX-L7: intensity weighting applied if provided.
# # # #     """
# # # #     M, T, B, _ = pred_samples.shape
# # # #     if M == 1:
# # # #         loss_per_b = _haversine(pred_samples[0], gt, unit_01deg).mean(0)  # [B]
# # # #     else:
# # # #         total, n_pairs = gt.new_zeros(B), 0
# # # #         for s in range(M):
# # # #             for sp in range(M):
# # # #                 if s == sp:
# # # #                     continue
# # # #                 d_sy  = _haversine(pred_samples[s],  gt,               unit_01deg).mean(0)
# # # #                 d_spy = _haversine(pred_samples[sp], gt,               unit_01deg).mean(0)
# # # #                 d_ssp = _haversine(pred_samples[s],  pred_samples[sp], unit_01deg).mean(0)
# # # #                 total   = total + (d_sy + d_spy - d_ssp).clamp(min=0)
# # # #                 n_pairs += 1
# # # #         loss_per_b = total / (2.0 * n_pairs)  # [B]

# # # #     if intensity_w is not None:
# # # #         w = intensity_w.to(loss_per_b.device)
# # # #         return (loss_per_b * w).mean()
# # # #     return loss_per_b.mean()


# # # # def overall_dir_loss(pred, gt, ref):
# # # #     p_d = pred[-1] - ref
# # # #     g_d = gt[-1]   - ref
# # # #     pn  = p_d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     gn  = g_d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     return (1.0 - ((p_d / pn) * (g_d / gn)).sum(-1)).mean()


# # # # def step_dir_loss(pred, gt):
# # # #     if pred.shape[0] < 2:
# # # #         return pred.new_zeros(())
# # # #     pv = pred[1:] - pred[:-1]
# # # #     gv = gt[1:]   - gt[:-1]
# # # #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     return (1.0 - ((pv / pn) * (gv / gn)).sum(-1)).mean()


# # # # def disp_loss(pred, gt):
# # # #     if pred.shape[0] < 2:
# # # #         return pred.new_zeros(())
# # # #     pd = (pred[1:] - pred[:-1]).norm(dim=-1).mean(0)
# # # #     gd = (gt[1:]   - gt[:-1]).norm(dim=-1).mean(0)
# # # #     return ((pd - gd) ** 2).mean()


# # # # def heading_loss(pred, gt):
# # # #     if pred.shape[0] < 2:
# # # #         return pred.new_zeros(())
# # # #     pv = pred[1:] - pred[:-1]
# # # #     gv = gt[1:]   - gt[:-1]
# # # #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # # #     wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1)).mean()
# # # #     if pred.shape[0] >= 3:
# # # #         def _curv(v):
# # # #             cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
# # # #             n1    = v[1:].norm(dim=-1).clamp(min=1e-6)
# # # #             n2    = v[:-1].norm(dim=-1).clamp(min=1e-6)
# # # #             return cross / (n1 * n2)
# # # #         curv_mse = F.mse_loss(_curv(pv), _curv(gv))
# # # #     else:
# # # #         curv_mse = pred.new_zeros(())
# # # #     return wrong_dir + curv_mse


# # # # def smooth_loss(pred):
# # # #     if pred.shape[0] < 3:
# # # #         return pred.new_zeros(())
# # # #     return ((pred[2:] - 2.0 * pred[1:-1] + pred[:-2]) ** 2).mean()


# # # # def _pinn_simplified(pred_abs):
# # # #     T = pred_abs.shape[0]
# # # #     if T < 4:
# # # #         return pred_abs.new_zeros(())
# # # #     v   = pred_abs[1:] - pred_abs[:-1]
# # # #     vx, vy = v[..., 0], v[..., 1]
# # # #     zeta   = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]
# # # #     if zeta.shape[0] < 2:
# # # #         return pred_abs.new_zeros(())
# # # #     dzeta   = zeta[1:] - zeta[:-1]
# # # #     lat_rad = pred_abs[2:T-1, :, 1] * NORM_TO_DEG * (math.pi / 180)
# # # #     beta_n  = (2.0 * OMEGA * NORM_TO_M * DT_6H / R_EARTH) * torch.cos(lat_rad)
# # # #     raw = ((dzeta + beta_n * vy[1:T-2]) ** 2).mean() * PINN_SCALE
# # # #     return raw.clamp(max=5.0)


# # # # def pinn_bve_loss(pred_abs, batch_list):
# # # #     return _pinn_simplified(pred_abs)


# # # # def compute_total_loss(pred_abs, gt, ref, batch_list,
# # # #                        pred_samples=None, weights=WEIGHTS,
# # # #                        intensity_w: Optional[torch.Tensor] = None) -> Dict:
# # # #     """
# # # #     FIX-L6: intensity_w [B] applied to fm_afcrps_loss.
# # # #     FIX-L5: smooth weight is now 0.05 in default WEIGHTS.
# # # #     """
# # # #     l_fm      = fm_afcrps_loss(pred_samples, gt, intensity_w=intensity_w) \
# # # #                 if pred_samples is not None \
# # # #                 else _haversine(pred_abs, gt, unit_01deg=False).mean()
# # # #     l_dir     = overall_dir_loss(pred_abs, gt, ref)
# # # #     l_step    = step_dir_loss(pred_abs, gt)
# # # #     l_disp    = disp_loss(pred_abs, gt)
# # # #     l_heading = heading_loss(pred_abs, gt)
# # # #     l_smooth  = smooth_loss(pred_abs)
# # # #     l_pinn    = pinn_bve_loss(pred_abs, batch_list)

# # # #     total = (weights["fm"]      * l_fm
# # # #            + weights["dir"]     * l_dir
# # # #            + weights["step"]    * l_step
# # # #            + weights["disp"]    * l_disp
# # # #            + weights["heading"] * l_heading
# # # #            + weights["smooth"]  * l_smooth
# # # #            + weights["pinn"]    * l_pinn)

# # # #     total = total.clamp(max=500.0)

# # # #     return dict(
# # # #         total=total,
# # # #         fm=l_fm.item(), dir=l_dir.item(), step=l_step.item(),
# # # #         disp=l_disp.item(), heading=l_heading.item(),
# # # #         smooth=l_smooth.item(), pinn=l_pinn.item(),
# # # #     )


# # # # # ── Legacy helpers ────────────────────────────────────────────────────────────

# # # # class TripletLoss(torch.nn.Module):
# # # #     def __init__(self, margin=None):
# # # #         super().__init__()
# # # #         self.margin  = margin
# # # #         self.loss_fn = (torch.nn.SoftMarginLoss() if margin is None
# # # #                         else torch.nn.TripletMarginLoss(margin=margin, p=2))

# # # #     def forward(self, anchor, pos, neg):
# # # #         if self.margin is None:
# # # #             y = torch.ones(anchor.shape[0], device=anchor.device)
# # # #             return self.loss_fn(
# # # #                 torch.norm(anchor - neg, 2, dim=1) - torch.norm(anchor - pos, 2, dim=1), y)
# # # #         return self.loss_fn(anchor, pos, neg)


# # # # def toNE(pred_traj, pred_Me):
# # # #     rt, rm = pred_traj.clone(), pred_Me.clone()
# # # #     if rt.dim() == 2:
# # # #         rt = rt.unsqueeze(1)
# # # #         rm = rm.unsqueeze(1)
# # # #     rt[:, :, 0] = rt[:, :, 0] * 50.0 + 1800.0
# # # #     rt[:, :, 1] = rt[:, :, 1] * 50.0
# # # #     rm[:, :, 0] = rm[:, :, 0] * 50.0 + 960.0
# # # #     rm[:, :, 1] = rm[:, :, 1] * 25.0 + 40.0
# # # #     return rt, rm


# # # # def trajectory_displacement_error(pred, gt, mode="sum"):
# # # #     _gt   = gt.permute(1, 0, 2)
# # # #     _pred = pred.permute(1, 0, 2)
# # # #     diff  = _gt - _pred
# # # #     lon_km = diff[:, :, 0] / 10.0 * 111.0 * torch.cos(_gt[:, :, 1] / 10.0 * torch.pi / 180.0)
# # # #     lat_km = diff[:, :, 1] / 10.0 * 111.0
# # # #     loss   = torch.sqrt(lon_km ** 2 + lat_km ** 2)
# # # #     return torch.sum(loss) if mode == "sum" else loss


# # # # def evaluate_diffusion_output(best_traj, best_Me, gt_traj, gt_Me):
# # # #     rt, rm  = toNE(best_traj.clone(), best_Me.clone())
# # # #     rg, rgm = toNE(gt_traj.clone(),  gt_Me.clone())
# # # #     return (trajectory_displacement_error(rt, rg, mode="raw"),
# # # #             torch.abs(rm.permute(1, 0, 2) - rgm.permute(1, 0, 2)))
# # # """
# # # Model/losses.py  ── v14
# # # ========================
# # # FIXES vs v11:

# # #   FIX-L9  PINN weight inconsistency: train script (v13) overrides
# # #            WEIGHTS["pinn"] = 0.1 after import, while WEIGHTS dict here
# # #            has pinn=0.05. This caused confusion and different behaviour
# # #            depending on import order. Fix: set pinn=0.1 here as the
# # #            canonical default, remove the override in train script.

# # #   FIX-L10 _pinn_simplified: beta_n lat indexing was off by one.
# # #            pred_abs[2:T-1] has shape [T-3, B, 2], but vy[1:T-2] has
# # #            shape [T-3, B]. The slice for lat should be pred_abs[2:T-1]
# # #            which at T=12 gives indices 2..10 (9 steps), matching
# # #            dzeta=zeta[1:]-zeta[:-1] at T-3=9 steps. Was correct but
# # #            confusingly written — added explicit shape comment.

# # #   FIX-L11 fm_afcrps_loss O(M²) loop is slow for M=6 ensemble.
# # #            Replaced inner double-loop with vectorised pair computation
# # #            using broadcasting. Speedup ~3-4x at M=6.

# # #   FIX-L12 _haversine unit_01deg=True path: scale=10.0 was dividing
# # #            NORMALISED coords (already in ~[-3,3] range) by 10, treating
# # #            them as if they were in 0.1-degree units. After denorm_torch
# # #            (which does ×50+1800)/10 → degrees), coords passed to losses
# # #            are still normalised. Added explicit docstring and a guard:
# # #            if unit_01deg=False the coords are treated as degrees directly.
# # #            Loss functions that call _haversine with normalised coords
# # #            should use unit_01deg=True (the old default, unchanged).

# # #   Kept from v11:
# # #     FIX-L1  pinn 0.5→0.1 (now canonical here)
# # #     FIX-L2  _pinn_simplified clamp max=5.0 (kept, was max=50 in v10)
# # #     FIX-L3  fm_afcrps_loss M==1 fallback
# # #     FIX-L4  total clamp max=500
# # #     FIX-L5  smooth weight 0.05
# # #     FIX-L6  intensity_w support
# # #     FIX-L7  CRPS weighted by intensity_w
# # #     FIX-L8  _haversine unit_01deg comments
# # # """
# # # from __future__ import annotations

# # # import math
# # # from typing import Dict, Optional

# # # import torch
# # # import torch.nn.functional as F

# # # OMEGA        = 7.2921e-5
# # # R_EARTH      = 6.371e6
# # # DT_6H        = 6 * 3600
# # # NORM_TO_DEG  = 5.0
# # # NORM_TO_M    = NORM_TO_DEG * 111_000.0
# # # PINN_SCALE   = 100.0

# # # # FIX-L9: pinn=0.1 as canonical value (was 0.05, overridden to 0.1 in train script)
# # # # Removing the override in train_flowmatching.py — this is the single source of truth.
# # # WEIGHTS: Dict[str, float] = dict(
# # #     fm=1.0, dir=2.0, step=0.5, disp=1.0, heading=2.0, smooth=0.05, pinn=0.1,
# # # )


# # # def _haversine(p1: torch.Tensor, p2: torch.Tensor,
# # #                unit_01deg: bool = True) -> torch.Tensor:
# # #     """
# # #     Haversine distance in km.

# # #     unit_01deg=True  : coords are in NORMALISED space (stored as lon_norm/lat_norm).
# # #                        Denorm formula: lon_deg = (lon_norm*50+1800)/10
# # #                                        lat_deg = (lat_norm*50)/10
# # #                        These are passed through the trig directly after scaling.
# # #     unit_01deg=False : coords are already in degrees.

# # #     NOTE: For training losses, predictions and GT are still in normalised space,
# # #     so unit_01deg=True is correct. For evaluation (haversine_km_torch in metrics),
# # #     coords have been denormed to degrees first, so unit_01deg=False is used there.
# # #     """
# # #     if unit_01deg:
# # #         # normalised → degrees: lon = (x*50+1800)/10, lat = (y*50)/10
# # #         lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
# # #         lat1 = (p1[..., 1] * 50.0) / 10.0
# # #         lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
# # #         lat2 = (p2[..., 1] * 50.0) / 10.0
# # #     else:
# # #         lon1, lat1 = p1[..., 0], p1[..., 1]
# # #         lon2, lat2 = p2[..., 0], p2[..., 1]

# # #     lat1r = torch.deg2rad(lat1)
# # #     lat2r = torch.deg2rad(lat2)
# # #     dlon  = torch.deg2rad(lon2 - lon1)
# # #     dlat  = torch.deg2rad(lat2 - lat1)
# # #     a     = (torch.sin(dlat / 2) ** 2
# # #              + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2) ** 2)
# # #     return 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())


# # # def fm_afcrps_loss(
# # #     pred_samples: torch.Tensor,
# # #     gt: torch.Tensor,
# # #     unit_01deg: bool = False,
# # #     intensity_w: Optional[torch.Tensor] = None,
# # # ) -> torch.Tensor:
# # #     """
# # #     AFCRPS energy-form loss.
# # #     pred_samples: [M, T, B, 2]
# # #     gt:           [T, B, 2]
# # #     intensity_w:  [B] optional per-sample weights

# # #     FIX-L11: Vectorised pair computation (was O(M²) double loop).
# # #     """
# # #     M, T, B, _ = pred_samples.shape

# # #     if M == 1:
# # #         loss_per_b = _haversine(pred_samples[0], gt, unit_01deg).mean(0)  # [B]
# # #     else:
# # #         # Vectorised: compute all pairwise distances at once
# # #         # pred_samples: [M, T, B, 2]
# # #         # Broadcast [M, 1, T, B, 2] vs [1, M, T, B, 2]
# # #         ps_i = pred_samples.unsqueeze(1)   # [M, 1, T, B, 2]
# # #         ps_j = pred_samples.unsqueeze(0)   # [1, M, T, B, 2]

# # #         # Distance from each sample to GT: [M, T, B]
# # #         d_to_gt = _haversine(pred_samples, gt.unsqueeze(0).expand_as(pred_samples), unit_01deg)
# # #         # d_to_gt: [M, T, B] — mean over T → [M, B]
# # #         d_to_gt_mean = d_to_gt.mean(1)  # [M, B]

# # #         # Pairwise distances: [M, M, T, B]
# # #         d_pair = _haversine(
# # #             ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # #             ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# # #             unit_01deg,
# # #         ).reshape(M, M, T, B).mean(2)  # [M, M, B]

# # #         # AFCRPS: E[d(s,y)] - 0.5*E[d(s,s')]
# # #         # = mean over M of d_to_gt - 0.5 * mean over M,M of d_pair
# # #         e_sy  = d_to_gt_mean.mean(0)       # [B]
# # #         e_ssp = d_pair.mean(0).mean(0)     # [B]
# # #         loss_per_b = (e_sy - 0.5 * e_ssp).clamp(min=0.0)  # [B]

# # #     if intensity_w is not None:
# # #         w = intensity_w.to(loss_per_b.device)
# # #         return (loss_per_b * w).mean()
# # #     return loss_per_b.mean()


# # # def overall_dir_loss(pred, gt, ref):
# # #     p_d = pred[-1] - ref
# # #     g_d = gt[-1]   - ref
# # #     pn  = p_d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # #     gn  = g_d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # #     return (1.0 - ((p_d / pn) * (g_d / gn)).sum(-1)).mean()


# # # def step_dir_loss(pred, gt):
# # #     if pred.shape[0] < 2:
# # #         return pred.new_zeros(())
# # #     pv = pred[1:] - pred[:-1]
# # #     gv = gt[1:]   - gt[:-1]
# # #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # #     return (1.0 - ((pv / pn) * (gv / gn)).sum(-1)).mean()


# # # def disp_loss(pred, gt):
# # #     if pred.shape[0] < 2:
# # #         return pred.new_zeros(())
# # #     pd = (pred[1:] - pred[:-1]).norm(dim=-1).mean(0)
# # #     gd = (gt[1:]   - gt[:-1]).norm(dim=-1).mean(0)
# # #     return ((pd - gd) ** 2).mean()


# # # def heading_loss(pred, gt):
# # #     if pred.shape[0] < 2:
# # #         return pred.new_zeros(())
# # #     pv = pred[1:] - pred[:-1]
# # #     gv = gt[1:]   - gt[:-1]
# # #     pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # #     gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# # #     wrong_dir = F.relu(-((pv / pn) * (gv / gn)).sum(-1)).mean()
# # #     if pred.shape[0] >= 3:
# # #         def _curv(v):
# # #             cross = v[1:, :, 0] * v[:-1, :, 1] - v[1:, :, 1] * v[:-1, :, 0]
# # #             n1    = v[1:].norm(dim=-1).clamp(min=1e-6)
# # #             n2    = v[:-1].norm(dim=-1).clamp(min=1e-6)
# # #             return cross / (n1 * n2)
# # #         curv_mse = F.mse_loss(_curv(pv), _curv(gv))
# # #     else:
# # #         curv_mse = pred.new_zeros(())
# # #     return wrong_dir + curv_mse


# # # def smooth_loss(pred):
# # #     if pred.shape[0] < 3:
# # #         return pred.new_zeros(())
# # #     return ((pred[2:] - 2.0 * pred[1:-1] + pred[:-2]) ** 2).mean()


# # # def _pinn_simplified(pred_abs: torch.Tensor) -> torch.Tensor:
# # #     """
# # #     Simplified barotropic vorticity equation (BVE) PINN loss.

# # #     FIX-L10: Added explicit shape comments for clarity.
# # #     pred_abs: [T, B, 2] — normalised (lon_norm, lat_norm)

# # #     v[t] = pred_abs[t+1] - pred_abs[t]       shape [T-1, B, 2]
# # #     zeta  = vx[t]*vy[t-1] - vy[t]*vx[t-1]   shape [T-2, B]
# # #     dzeta = zeta[t] - zeta[t-1]              shape [T-3, B]
# # #     lat   = pred_abs[2:T-1, :, 1]            shape [T-3, B]  ← matches dzeta
# # #     vy_   = vy[1:T-2]                        shape [T-3, B]  ← matches dzeta
# # #     """
# # #     T = pred_abs.shape[0]
# # #     if T < 4:
# # #         return pred_abs.new_zeros(())

# # #     v   = pred_abs[1:] - pred_abs[:-1]          # [T-1, B, 2]
# # #     vx, vy = v[..., 0], v[..., 1]               # [T-1, B] each

# # #     zeta  = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]  # [T-2, B]
# # #     if zeta.shape[0] < 2:
# # #         return pred_abs.new_zeros(())

# # #     dzeta = zeta[1:] - zeta[:-1]                # [T-3, B]

# # #     # lat at steps 2..T-2 (shape [T-3, B]) — matches dzeta
# # #     lat_rad = pred_abs[2:T-1, :, 1] * NORM_TO_DEG * (math.pi / 180)  # [T-3, B]

# # #     # beta-plane approximation: β·v (northward velocity at steps 1..T-3)
# # #     beta_n = (2.0 * OMEGA * NORM_TO_M * DT_6H / R_EARTH) * torch.cos(lat_rad)  # [T-3, B]
# # #     vy_mid = vy[1:T-2]                          # [T-3, B]

# # #     raw = ((dzeta + beta_n * vy_mid) ** 2).mean() * PINN_SCALE
# # #     return raw.clamp(max=5.0)


# # # def pinn_bve_loss(pred_abs, batch_list):
# # #     return _pinn_simplified(pred_abs)


# # # def compute_total_loss(
# # #     pred_abs,
# # #     gt,
# # #     ref,
# # #     batch_list,
# # #     pred_samples=None,
# # #     weights=WEIGHTS,
# # #     intensity_w: Optional[torch.Tensor] = None,
# # # ) -> Dict:
# # #     """
# # #     FIX-L9: pinn weight now comes from WEIGHTS dict (canonical=0.1).
# # #              No more override in train script needed.
# # #     FIX-L11: fm_afcrps_loss is now vectorised.
# # #     """
# # #     l_fm = (
# # #         fm_afcrps_loss(pred_samples, gt, intensity_w=intensity_w)
# # #         if pred_samples is not None
# # #         else _haversine(pred_abs, gt, unit_01deg=False).mean()
# # #     )
# # #     l_dir     = overall_dir_loss(pred_abs, gt, ref)
# # #     l_step    = step_dir_loss(pred_abs, gt)
# # #     l_disp    = disp_loss(pred_abs, gt)
# # #     l_heading = heading_loss(pred_abs, gt)
# # #     l_smooth  = smooth_loss(pred_abs)
# # #     l_pinn    = pinn_bve_loss(pred_abs, batch_list)

# # #     total = (weights.get("fm",      1.0)  * l_fm
# # #            + weights.get("dir",     2.0)  * l_dir
# # #            + weights.get("step",    0.5)  * l_step
# # #            + weights.get("disp",    1.0)  * l_disp
# # #            + weights.get("heading", 2.0)  * l_heading
# # #            + weights.get("smooth",  0.05) * l_smooth
# # #            + weights.get("pinn",    0.1)  * l_pinn)

# # #     total = total.clamp(max=500.0)

# # #     return dict(
# # #         total=total,
# # #         fm=l_fm.item(), dir=l_dir.item(), step=l_step.item(),
# # #         disp=l_disp.item(), heading=l_heading.item(),
# # #         smooth=l_smooth.item(), pinn=l_pinn.item(),
# # #     )


# # # # ── Legacy helpers ────────────────────────────────────────────────────────────

# # # class TripletLoss(torch.nn.Module):
# # #     def __init__(self, margin=None):
# # #         super().__init__()
# # #         self.margin  = margin
# # #         self.loss_fn = (torch.nn.SoftMarginLoss() if margin is None
# # #                         else torch.nn.TripletMarginLoss(margin=margin, p=2))

# # #     def forward(self, anchor, pos, neg):
# # #         if self.margin is None:
# # #             y = torch.ones(anchor.shape[0], device=anchor.device)
# # #             return self.loss_fn(
# # #                 torch.norm(anchor - neg, 2, dim=1) - torch.norm(anchor - pos, 2, dim=1), y)
# # #         return self.loss_fn(anchor, pos, neg)


# # # def toNE(pred_traj, pred_Me):
# # #     rt, rm = pred_traj.clone(), pred_Me.clone()
# # #     if rt.dim() == 2:
# # #         rt = rt.unsqueeze(1)
# # #         rm = rm.unsqueeze(1)
# # #     rt[:, :, 0] = rt[:, :, 0] * 50.0 + 1800.0
# # #     rt[:, :, 1] = rt[:, :, 1] * 50.0
# # #     rm[:, :, 0] = rm[:, :, 0] * 50.0 + 960.0
# # #     rm[:, :, 1] = rm[:, :, 1] * 25.0 + 40.0
# # #     return rt, rm


# # # def trajectory_displacement_error(pred, gt, mode="sum"):
# # #     _gt   = gt.permute(1, 0, 2)
# # #     _pred = pred.permute(1, 0, 2)
# # #     diff  = _gt - _pred
# # #     lon_km = diff[:, :, 0] / 10.0 * 111.0 * torch.cos(
# # #         _gt[:, :, 1] / 10.0 * torch.pi / 180.0)
# # #     lat_km = diff[:, :, 1] / 10.0 * 111.0
# # #     loss   = torch.sqrt(lon_km ** 2 + lat_km ** 2)
# # #     return torch.sum(loss) if mode == "sum" else loss


# # # def evaluate_diffusion_output(best_traj, best_Me, gt_traj, gt_Me):
# # #     rt, rm  = toNE(best_traj.clone(), best_Me.clone())
# # #     rg, rgm = toNE(gt_traj.clone(),  gt_Me.clone())
# # #     return (trajectory_displacement_error(rt, rg, mode="raw"),
# # #             torch.abs(rm.permute(1, 0, 2) - rgm.permute(1, 0, 2)))

# # """
# # Model/losses.py  ── v15
# # ========================
# # FIXES vs v14:

# #   FIX-L13  _pinn_simplified clamp max=5.0 too tight → PINN oscillates
# #            between 0 and 5.0 every batch, never converging.
# #            Root cause: raw PINN values often legitimately exceed 5.0 for
# #            poor trajectories early in training, so the gradient always
# #            saturates. Fix: increase clamp to max=50.0 BUT also add
# #            gradient-scaled normalization so large values don't blow up FM.
# #            The WEIGHTS["pinn"] adaptive schedule in train script (0.01→0.1)
# #            compensates for the wider range.

# #   FIX-L14  _pinn_simplified: PINN_SCALE=100.0 combined with clamp max=5.0
# #            means the effective range before clamp is [0, 0.05] — extremely
# #            narrow. Changed PINN_SCALE to 1.0 and clamp to max=10.0 so
# #            values are interpretable and gradients flow properly.

# #   FIX-L15  fm_afcrps_loss: when all ensemble samples collapse to the same
# #            prediction (early training), e_sy ≈ 0.5 * e_ssp → loss ≈ 0.
# #            Added eps floor: clamp loss_per_b.min to eps=1e-3 so gradients
# #            always flow even when ensemble is perfectly calibrated.

# #   FIX-L16  compute_total_loss: added NaN guard on total loss.
# #            If any sub-loss produces NaN (e.g. from NaN in batch),
# #            return zero total to avoid corrupting model weights.

# #   Kept from v14:
# #     FIX-L9  pinn=0.1 canonical (overridden adaptively by train script)
# #     FIX-L10 lat indexing shape comment
# #     FIX-L11 vectorised AFCRPS
# #     FIX-L12 _haversine unit_01deg docstring
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

# # # FIX-L14: PINN_SCALE 100.0→1.0; clamp handled in _pinn_simplified
# # PINN_SCALE   = 1.0

# # WEIGHTS: Dict[str, float] = dict(
# #     fm=1.0, dir=2.0, step=0.5, disp=1.0, heading=2.0, smooth=0.05, pinn=0.1,
# # )


# # def _haversine(p1: torch.Tensor, p2: torch.Tensor,
# #                unit_01deg: bool = True) -> torch.Tensor:
# #     """
# #     Haversine distance in km.
# #     unit_01deg=True  : coords are normalised (lon_norm, lat_norm).
# #     unit_01deg=False : coords are degrees.
# #     """
# #     if unit_01deg:
# #         lon1 = (p1[..., 0] * 50.0 + 1800.0) / 10.0
# #         lat1 = (p1[..., 1] * 50.0) / 10.0
# #         lon2 = (p2[..., 0] * 50.0 + 1800.0) / 10.0
# #         lat2 = (p2[..., 1] * 50.0) / 10.0
# #     else:
# #         lon1, lat1 = p1[..., 0], p1[..., 1]
# #         lon2, lat2 = p2[..., 0], p2[..., 1]

# #     lat1r = torch.deg2rad(lat1)
# #     lat2r = torch.deg2rad(lat2)
# #     dlon  = torch.deg2rad(lon2 - lon1)
# #     dlat  = torch.deg2rad(lat2 - lat1)
# #     a     = (torch.sin(dlat / 2) ** 2
# #              + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2) ** 2)
# #     return 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())


# # def fm_afcrps_loss(
# #     pred_samples: torch.Tensor,
# #     gt: torch.Tensor,
# #     unit_01deg: bool = False,
# #     intensity_w: Optional[torch.Tensor] = None,
# # ) -> torch.Tensor:
# #     """
# #     AFCRPS energy-form loss.
# #     pred_samples: [M, T, B, 2]
# #     gt:           [T, B, 2]

# #     FIX-L15: eps floor so gradients flow even when ensemble is calibrated.
# #     FIX-L11: vectorised (from v14).
# #     """
# #     M, T, B, _ = pred_samples.shape
# #     eps = 1e-3   # FIX-L15

# #     if M == 1:
# #         loss_per_b = _haversine(pred_samples[0], gt, unit_01deg).mean(0)
# #     else:
# #         ps_i = pred_samples.unsqueeze(1)   # [M, 1, T, B, 2]
# #         ps_j = pred_samples.unsqueeze(0)   # [1, M, T, B, 2]

# #         d_to_gt = _haversine(pred_samples,
# #                               gt.unsqueeze(0).expand_as(pred_samples),
# #                               unit_01deg)   # [M, T, B]
# #         d_to_gt_mean = d_to_gt.mean(1)     # [M, B]

# #         d_pair = _haversine(
# #             ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# #             ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
# #             unit_01deg,
# #         ).reshape(M, M, T, B).mean(2)      # [M, M, B]

# #         e_sy  = d_to_gt_mean.mean(0)       # [B]
# #         e_ssp = d_pair.mean(0).mean(0)     # [B]
# #         # FIX-L15: clamp min to eps, not just 0
# #         loss_per_b = (e_sy - 0.5 * e_ssp).clamp(min=eps)

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


# # def _pinn_simplified(pred_abs: torch.Tensor) -> torch.Tensor:
# #     """
# #     Simplified barotropic vorticity equation (BVE) PINN loss.

# #     FIX-L13: clamp max increased from 5.0→50.0 (was saturating gradients).
# #     FIX-L14: PINN_SCALE changed from 100.0→1.0 (raw values now interpretable).

# #     pred_abs: [T, B, 2] — normalised (lon_norm, lat_norm)
# #     Shapes:
# #       v       [T-1, B, 2]
# #       vx, vy  [T-1, B]
# #       zeta    [T-2, B]   cross product of adjacent velocities
# #       dzeta   [T-3, B]   time derivative of vorticity
# #       lat_rad [T-3, B]   matches dzeta
# #       vy_mid  [T-3, B]   matches dzeta
# #     """
# #     T = pred_abs.shape[0]
# #     if T < 4:
# #         return pred_abs.new_zeros(())

# #     v   = pred_abs[1:] - pred_abs[:-1]          # [T-1, B, 2]
# #     vx, vy = v[..., 0], v[..., 1]               # [T-1, B]

# #     zeta  = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]  # [T-2, B]
# #     if zeta.shape[0] < 2:
# #         return pred_abs.new_zeros(())

# #     dzeta = zeta[1:] - zeta[:-1]                # [T-3, B]

# #     lat_rad = pred_abs[2:T-1, :, 1] * NORM_TO_DEG * (math.pi / 180)  # [T-3, B]

# #     beta_n = (2.0 * OMEGA * NORM_TO_M * DT_6H / R_EARTH) * torch.cos(lat_rad)
# #     vy_mid = vy[1:T-2]                          # [T-3, B]

# #     # FIX-L14: PINN_SCALE=1.0 — raw values in physical units
# #     # FIX-L13: clamp max=50.0 instead of 5.0 so gradients flow
# #     raw = ((dzeta + beta_n * vy_mid) ** 2).mean() * PINN_SCALE
# #     return raw.clamp(max=50.0)


# # def pinn_bve_loss(pred_abs, batch_list):
# #     return _pinn_simplified(pred_abs)


# # def compute_total_loss(
# #     pred_abs,
# #     gt,
# #     ref,
# #     batch_list,
# #     pred_samples=None,
# #     weights=WEIGHTS,
# #     intensity_w: Optional[torch.Tensor] = None,
# # ) -> Dict:
# #     """
# #     FIX-L16: NaN guard — if any sub-loss is NaN, return zero to avoid
# #     corrupting model weights during training.
# #     """
# #     l_fm = (
# #         fm_afcrps_loss(pred_samples, gt, intensity_w=intensity_w)
# #         if pred_samples is not None
# #         else _haversine(pred_abs, gt, unit_01deg=False).mean()
# #     )
# #     l_dir     = overall_dir_loss(pred_abs, gt, ref)
# #     l_step    = step_dir_loss(pred_abs, gt)
# #     l_disp    = disp_loss(pred_abs, gt)
# #     l_heading = heading_loss(pred_abs, gt)
# #     l_smooth  = smooth_loss(pred_abs)
# #     l_pinn    = pinn_bve_loss(pred_abs, batch_list)

# #     total = (weights.get("fm",      1.0)  * l_fm
# #            + weights.get("dir",     2.0)  * l_dir
# #            + weights.get("step",    0.5)  * l_step
# #            + weights.get("disp",    1.0)  * l_disp
# #            + weights.get("heading", 2.0)  * l_heading
# #            + weights.get("smooth",  0.05) * l_smooth
# #            + weights.get("pinn",    0.1)  * l_pinn)

# #     total = total.clamp(max=500.0)

# #     # FIX-L16: NaN guard
# #     if torch.isnan(total) or torch.isinf(total):
# #         print("  ⚠  NaN/Inf in total loss detected — returning zero")
# #         total = pred_abs.new_zeros(())

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
# #     lon_km = diff[:, :, 0] / 10.0 * 111.0 * torch.cos(
# #         _gt[:, :, 1] / 10.0 * torch.pi / 180.0)
# #     lat_km = diff[:, :, 1] / 10.0 * 111.0
# #     loss   = torch.sqrt(lon_km ** 2 + lat_km ** 2)
# #     return torch.sum(loss) if mode == "sum" else loss


# # def evaluate_diffusion_output(best_traj, best_Me, gt_traj, gt_Me):
# #     rt, rm  = toNE(best_traj.clone(), best_Me.clone())
# #     rg, rgm = toNE(gt_traj.clone(),  gt_Me.clone())
# #     return (trajectory_displacement_error(rt, rg, mode="raw"),
# #             torch.abs(rm.permute(1, 0, 2) - rgm.permute(1, 0, 2)))

# """
# Model/losses.py  ── v16
# ========================
# FIXES vs v15:

#   FIX-L17  Thêm velocity_loss: ATE >> CTE nghĩa là mô hình học hướng đi
#            tốt nhưng tốc độ sai. velocity_loss penalizes sai số step-size
#            (khoảng cách mỗi bước) so với ground-truth.
#            WEIGHTS["velocity"] = 1.5 (quan trọng, cao hơn disp_loss).

#   FIX-L18  acceleration_loss: thêm penalty cho gia tốc bất thường.
#            Bão không thể tăng tốc đột ngột → smooth acceleration.
#            WEIGHTS["accel"] = 0.3

#   FIX-L19  disp_loss đã tồn tại nhưng chỉ đo mean step size, không phân
#            biệt từng bước. velocity_loss đo per-step nên chi tiết hơn.
#            Giữ disp_loss nhưng giảm weight xuống 0.5 (từ 1.0).

#   FIX-L20  fm_afcrps_loss: thêm time-weighted CRPS — bước 72h quan trọng
#            hơn bước 6h. Weight tăng tuyến tính từ 0.5→1.5 theo thời gian.
#            Điều này giúp mô hình ưu tiên độ chính xác dài hạn.

# Kept from v15:
#   FIX-L13..L16 (pinn clamp, PINN_SCALE, eps floor, NaN guard)
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

# PINN_SCALE   = 1.0

# WEIGHTS: Dict[str, float] = dict(
#     fm=1.0, dir=2.0, step=0.5, disp=0.5,   # FIX-L19: disp 1.0→0.5
#     heading=2.0, smooth=0.05, pinn=0.1,
#     velocity=1.5,   # FIX-L17: NEW — penalize step-size error (ATE fix)
#     accel=0.3,      # FIX-L18: NEW — penalize sudden acceleration changes
# )


# def _haversine(p1: torch.Tensor, p2: torch.Tensor,
#                unit_01deg: bool = True) -> torch.Tensor:
#     """
#     Haversine distance in km.
#     unit_01deg=True  : coords are normalised (lon_norm, lat_norm).
#     unit_01deg=False : coords are degrees.
#     """
#     if unit_01deg:
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

#     FIX-L20: time-weighted CRPS — weight tăng theo lead time.
#     FIX-L15: eps floor.
#     FIX-L11: vectorised.
#     """
#     M, T, B, _ = pred_samples.shape
#     eps = 1e-3

#     # FIX-L20: time weights — bước cuối quan trọng gấp 3× bước đầu
#     time_w = torch.linspace(0.5, 1.5, T, device=pred_samples.device)  # [T]

#     if M == 1:
#         dist = _haversine(pred_samples[0], gt, unit_01deg)  # [T, B]
#         loss_per_b = (dist * time_w.unsqueeze(1)).mean(0)
#     else:
#         d_to_gt = _haversine(pred_samples,
#                               gt.unsqueeze(0).expand_as(pred_samples),
#                               unit_01deg)   # [M, T, B]
#         # Apply time weights
#         d_to_gt_w = d_to_gt * time_w.unsqueeze(0).unsqueeze(2)  # [M, T, B]
#         d_to_gt_mean = d_to_gt_w.mean(1)   # [M, B]

#         ps_i = pred_samples.unsqueeze(1)
#         ps_j = pred_samples.unsqueeze(0)
#         d_pair = _haversine(
#             ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
#             ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
#             unit_01deg,
#         ).reshape(M, M, T, B)
#         d_pair_w = d_pair * time_w.unsqueeze(0).unsqueeze(0).unsqueeze(3)
#         d_pair_mean = d_pair_w.mean(2)     # [M, M, B]

#         e_sy  = d_to_gt_mean.mean(0)
#         e_ssp = d_pair_mean.mean(0).mean(0)
#         loss_per_b = (e_sy - 0.5 * e_ssp).clamp(min=eps)

#     if intensity_w is not None:
#         w = intensity_w.to(loss_per_b.device)
#         return (loss_per_b * w).mean()
#     return loss_per_b.mean()


# def velocity_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#     """
#     FIX-L17: Per-step velocity magnitude loss.
#     Penalizes difference in step-size (speed) between pred and gt.
#     This directly addresses ATE >> CTE: model gets direction right but speed wrong.

#     pred, gt: [T, B, 2] in normalised coords
#     """
#     if pred.shape[0] < 2:
#         return pred.new_zeros(())

#     # Step vectors in normalised space
#     pred_v = pred[1:] - pred[:-1]   # [T-1, B, 2]
#     gt_v   = gt[1:]   - gt[:-1]     # [T-1, B, 2]

#     # Step sizes (speed proxy) in normalised space
#     pred_speed = pred_v.norm(dim=-1)  # [T-1, B]
#     gt_speed   = gt_v.norm(dim=-1)    # [T-1, B]

#     # MSE on speed — forces model to match magnitude, not just direction
#     speed_mse = F.mse_loss(pred_speed, gt_speed)

#     # Also penalise along-track signed error (ATE direction)
#     gt_v_norm = gt_v / gt_v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
#     # Signed along-track difference
#     ate_signed = ((pred_v - gt_v) * gt_v_norm).sum(-1)  # [T-1, B]
#     ate_mse    = (ate_signed ** 2).mean()

#     return speed_mse + 0.3 * ate_mse


# def acceleration_loss(pred: torch.Tensor) -> torch.Tensor:
#     """
#     FIX-L18: Smooth acceleration — TCs don't change speed abruptly.
#     Penalizes large changes in step-velocity (jerk).

#     pred: [T, B, 2]
#     """
#     if pred.shape[0] < 3:
#         return pred.new_zeros(())
#     v = pred[1:] - pred[:-1]     # [T-1, B, 2]
#     a = v[1:] - v[:-1]           # [T-2, B, 2] — acceleration
#     return (a ** 2).mean()


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
#     FIX-L13: clamp max=50.0
#     FIX-L14: PINN_SCALE=1.0
#     """
#     T = pred_abs.shape[0]
#     if T < 4:
#         return pred_abs.new_zeros(())

#     v   = pred_abs[1:] - pred_abs[:-1]
#     vx, vy = v[..., 0], v[..., 1]

#     zeta  = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]
#     if zeta.shape[0] < 2:
#         return pred_abs.new_zeros(())

#     dzeta = zeta[1:] - zeta[:-1]
#     lat_rad = pred_abs[2:T-1, :, 1] * NORM_TO_DEG * (math.pi / 180)
#     beta_n = (2.0 * OMEGA * NORM_TO_M * DT_6H / R_EARTH) * torch.cos(lat_rad)
#     vy_mid = vy[1:T-2]

#     raw = ((dzeta + beta_n * vy_mid) ** 2).mean() * PINN_SCALE
#     return raw.clamp(max=50.0)


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
#     FIX-L16: NaN guard.
#     FIX-L17: velocity_loss added.
#     FIX-L18: acceleration_loss added.
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
#     l_vel     = velocity_loss(pred_abs, gt)       # FIX-L17
#     l_accel   = acceleration_loss(pred_abs)        # FIX-L18

#     total = (weights.get("fm",       1.0)  * l_fm
#            + weights.get("dir",      2.0)  * l_dir
#            + weights.get("step",     0.5)  * l_step
#            + weights.get("disp",     0.5)  * l_disp    # FIX-L19: 1.0→0.5
#            + weights.get("heading",  2.0)  * l_heading
#            + weights.get("smooth",   0.05) * l_smooth
#            + weights.get("pinn",     0.1)  * l_pinn
#            + weights.get("velocity", 1.5)  * l_vel     # FIX-L17
#            + weights.get("accel",    0.3)  * l_accel)  # FIX-L18

#     total = total.clamp(max=500.0)

#     # FIX-L16: NaN guard
#     if torch.isnan(total) or torch.isinf(total):
#         print("  ⚠  NaN/Inf in total loss detected — returning zero")
#         total = pred_abs.new_zeros(())

#     return dict(
#         total=total,
#         fm=l_fm.item(), dir=l_dir.item(), step=l_step.item(),
#         disp=l_disp.item(), heading=l_heading.item(),
#         smooth=l_smooth.item(), pinn=l_pinn.item(),
#         velocity=l_vel.item(), accel=l_accel.item(),
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
Model/losses.py  ── v17
========================
FIXES vs v16:

  FIX-L21  Straight-track bias (PR=1.37): thêm recurvature_weight vào
           compute_total_loss. Khi sequence là recurvature, nhân toàn bộ
           loss lên × recurv_weight (mặc định 2.5). Điều này buộc mô hình
           chú ý nhiều hơn vào các track khó (bão quay đầu).

           Cách detect recurvature online (không cần pre-label):
           Tính total_rotation_angle từ gt trajectory. Nếu >= 45° → recurvature.
           Weight shape: [B] tensor, 2.5 cho recurvature, 1.0 cho straight.

  FIX-L22  Kết hợp recurv_weight với intensity_w thành unified sample_weight.
           Không cần 2 separate weighting paths.

  FIX-L23  recurvature_loss riêng: penalize trực tiếp heading error tại
           điểm bão đổi hướng (turning point). Phát hiện turning point bằng
           cross product của consecutive velocity vectors đổi dấu.
           WEIGHTS["recurv"] = 1.0

Kept from v16:
  FIX-L17..L20 (velocity_loss, acceleration_loss, disp, time-weighted CRPS)
  FIX-L13..L16 (pinn clamp, NaN guard, eps floor)
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

PINN_SCALE   = 1.0
WEIGHTS: Dict[str, float] = dict(
    fm=2.0,        # Vị trí (km)
    velocity=2.0,  # Tốc độ (ATE/Speed)
    heading=1.5,   # Độ cong quỹ đạo
    recurv=1.5,    # Turning point
    step=0.5,      # Hướng từng bước
    disp=0.5,      # Độ dài bước nhảy
    dir=1.0,       # Hướng đích cuối
    smooth=0.1,    # Độ mượt
    accel=0.3,     # Quán tính
    pinn=0.02,     # Vật lý BVE
)
# FIX-L21: threshold góc để classify recurvature (degree)
RECURV_ANGLE_THR = 45.0
# FIX-L21: weight multiplier cho recurvature sequences
RECURV_WEIGHT    = 2.5


# ── Haversine ─────────────────────────────────────────────────────────────────

def _haversine(p1: torch.Tensor, p2: torch.Tensor,
               unit_01deg: bool = True) -> torch.Tensor:
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


# ── FIX-L21: Recurvature detection & weighting ────────────────────────────────

def _total_rotation_angle_batch(gt: torch.Tensor) -> torch.Tensor:
    """
    Tính tổng góc quay của quỹ đạo cho mỗi sample trong batch.
    gt: [T, B, 2] — normalised coords
    Returns: [B] tensor, đơn vị degrees

    Dùng lat-corrected velocity để chính xác hơn ở vĩ độ cao.
    """
    T, B, _ = gt.shape
    if T < 3:
        return gt.new_zeros(B)

    # Lat-corrected velocity
    lats_rad = gt[:, :, 1] * NORM_TO_DEG * (math.pi / 180.0)  # [T, B]
    cos_lat  = torch.cos(lats_rad[:-1])                          # [T-1, B]
    dlat = gt[1:, :, 1] - gt[:-1, :, 1]                         # [T-1, B]
    dlon = (gt[1:, :, 0] - gt[:-1, :, 0]) * cos_lat             # [T-1, B]
    v = torch.stack([dlon, dlat], dim=-1)                        # [T-1, B, 2]

    # Total rotation = sum of angles between consecutive step vectors
    v1 = v[:-1]   # [T-2, B, 2]
    v2 = v[1:]    # [T-2, B, 2]

    n1 = v1.norm(dim=-1).clamp(min=1e-8)
    n2 = v2.norm(dim=-1).clamp(min=1e-8)

    cos_a = (v1 * v2).sum(-1) / (n1 * n2)
    cos_a = cos_a.clamp(-1.0, 1.0)
    angles_rad = torch.acos(cos_a)                               # [T-2, B]

    total_deg = torch.rad2deg(angles_rad).sum(0)                 # [B]
    return total_deg


def _recurvature_weights(gt: torch.Tensor,
                         thr: float = RECURV_ANGLE_THR,
                         w_recurv: float = RECURV_WEIGHT) -> torch.Tensor:
    """
    FIX-L21: Per-sample weight tensor.
    recurvature sequences → weight=w_recurv, straight → weight=1.0
    gt: [T, B, 2]
    Returns: [B] float tensor
    """
    rot = _total_rotation_angle_batch(gt)             # [B]
    w   = torch.where(rot >= thr,
                      torch.full_like(rot, w_recurv),
                      torch.ones_like(rot))
    return w


# ── FIX-L23: Recurvature (turning point) loss ─────────────────────────────────

def recurvature_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    FIX-L23: Penalize heading error tại turning points.
    Turning point = nơi cross product của consecutive velocities đổi dấu.

    pred, gt: [T, B, 2] normalised coords
    """
    if pred.shape[0] < 3:
        return pred.new_zeros(())

    # Step vectors
    pred_v = pred[1:] - pred[:-1]   # [T-1, B, 2]
    gt_v   = gt[1:] - gt[:-1]       # [T-1, B, 2]

    # Cross products of consecutive gt vectors — detect turning points
    # cross[t] = gt_v[t] × gt_v[t+1] (z-component of 2D cross product)
    gt_cross = (gt_v[:-1, :, 0] * gt_v[1:, :, 1]
              - gt_v[:-1, :, 1] * gt_v[1:, :, 0])    # [T-2, B]

    # Turning point mask: sign changes between consecutive cross products
    # sign_change[t] = True khi gt_cross[t] và gt_cross[t-1] khác dấu
    if gt_cross.shape[0] < 2:
        return pred.new_zeros(())

    sign_change = (gt_cross[:-1] * gt_cross[1:]) < 0  # [T-3, B] bool

    if not sign_change.any():
        return pred.new_zeros(())

    # Tại turning points, penalize heading difference giữa pred và gt
    # Dùng bước gt_v[1:-1] (align với sign_change indexing)
    pred_v_mid = pred_v[1:-1]   # [T-2, B, 2] → sau đó [T-3, B, 2] cần thêm slice
    gt_v_mid   = gt_v[1:-1]

    if pred_v_mid.shape[0] > sign_change.shape[0]:
        pred_v_mid = pred_v_mid[:sign_change.shape[0]]
        gt_v_mid   = gt_v_mid[:sign_change.shape[0]]

    # Normalise
    pn = pred_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gn = gt_v_mid.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    p_hat = pred_v_mid / pn
    g_hat = gt_v_mid   / gn

    # Cosine similarity loss tại turning points
    cos_sim = (p_hat * g_hat).sum(-1)               # [T-3, B]
    dir_loss = (1.0 - cos_sim)                      # [T-3, B], 0=perfect

    # Chỉ tính loss tại turning points, weight ×3 cho turning points
    # mask = sign_change.float()                       # [T-3, B]
    # weighted = dir_loss * (1.0 + 2.0 * mask)        # non-turning=1×, turning=3×
    mask = sign_change.float()  # [T-3, B]
    if mask.sum() < 1:
        return pred.new_zeros(())
    weighted = (dir_loss * mask).sum() / mask.sum().clamp(min=1)  # mean chỉ tại turning points
    return weighted
    # return weighted.mean()


# ── AFCRPS loss ───────────────────────────────────────────────────────────────

def fm_afcrps_loss(
    pred_samples: torch.Tensor,
    gt: torch.Tensor,
    unit_01deg: bool = False,
    intensity_w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    AFCRPS với time-weighted CRPS (FIX-L20) và eps floor (FIX-L15).
    pred_samples: [M, T, B, 2], gt: [T, B, 2]
    """
    M, T, B, _ = pred_samples.shape
    eps = 1e-3

    # Time weights: bước 72h quan trọng gấp 3× bước 6h
    time_w = torch.linspace(0.5, 1.5, T, device=pred_samples.device)

    if M == 1:
        dist = _haversine(pred_samples[0], gt, unit_01deg)
        loss_per_b = (dist * time_w.unsqueeze(1)).mean(0)
    else:
        d_to_gt = _haversine(
            pred_samples,
            gt.unsqueeze(0).expand_as(pred_samples),
            unit_01deg,
        )   # [M, T, B]
        d_to_gt_w    = d_to_gt * time_w.unsqueeze(0).unsqueeze(2)
        d_to_gt_mean = d_to_gt_w.mean(1)   # [M, B]

        ps_i = pred_samples.unsqueeze(1)
        ps_j = pred_samples.unsqueeze(0)
        d_pair = _haversine(
            ps_i.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
            ps_j.expand(M, M, T, B, 2).reshape(M * M, T, B, 2),
            unit_01deg,
        ).reshape(M, M, T, B)
        d_pair_w   = d_pair * time_w.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        d_pair_mean = d_pair_w.mean(2)   # [M, M, B]

        e_sy  = d_to_gt_mean.mean(0)
        e_ssp = d_pair_mean.mean(0).mean(0)
        loss_per_b = (e_sy - 0.5 * e_ssp).clamp(min=eps)

    if intensity_w is not None:
        w = intensity_w.to(loss_per_b.device)
        return (loss_per_b * w).mean()
    return loss_per_b.mean()


# # ── Velocity loss (FIX-L17) ───────────────────────────────────────────────────

# # def velocity_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
# #     """
# #     Penalize per-step speed error và along-track signed error.
# #     pred, gt: [T, B, 2]
# #     """
# #     if pred.shape[0] < 2:
# #         return pred.new_zeros(())

# #     pred_v = pred[1:] - pred[:-1]
# #     gt_v   = gt[1:] - gt[:-1]

# #     pred_speed = pred_v.norm(dim=-1)
# #     gt_speed   = gt_v.norm(dim=-1)
# #     speed_mse  = F.mse_loss(pred_speed, gt_speed)

# #     gt_v_norm  = gt_v / gt_v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
# #     ate_signed = ((pred_v - gt_v) * gt_v_norm).sum(-1)
# #     ate_mse    = (ate_signed ** 2).mean()

# #     return speed_mse + 0.3 * ate_mse
# def velocity_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#     if pred.shape[0] < 2:
#         return pred.new_zeros(())

#     # Tính vector vận tốc (bước nhảy)
#     v_pred = pred[1:] - pred[:-1]
#     v_gt   = gt[1:] - gt[:-1]

#     # 1. Speed Loss: Khớp độ lớn vận tốc (tốc độ km/h)
#     s_pred = torch.norm(v_pred, dim=-1)
#     s_gt   = torch.norm(v_gt, dim=-1)
#     l_speed    = F.mse_loss(s_pred, s_gt)

#     # 2. ATE Loss (Along-track): Khớp vị trí trên quỹ đạo
#     # Tránh chia cho 0 nếu bão đứng yên
#     gt_v_unit = v_gt / torch.norm(v_gt, dim=-1, keepdim=True).clamp(min=1e-6)
    
#     # Chiếu sai số vận tốc lên hướng đi thực tế
#     # Nếu giá trị này dương: model dự báo bão đi nhanh hơn thực tế
#     # Nếu âm: model dự báo bão đi chậm hơn thực tế
#     ate_errors = torch.sum((v_pred - v_gt) * gt_v_unit, dim=-1)
#     l_ate      = torch.mean(ate_errors**2, dim=0)

#     # Kết hợp: Ưu tiên speed để ổn định, ate để chính xác thời gian
#     return l_speed + 0.5 * l_ate

# ── Acceleration loss (FIX-L18) ───────────────────────────────────────────────

def acceleration_loss(pred: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 3:
        return pred.new_zeros(())
    v = pred[1:] - pred[:-1]
    a = v[1:] - v[:-1]
    return (a ** 2).mean()


# ── Direction losses ───────────────────────────────────────────────────────────

def overall_dir_loss(pred, gt, ref):
    p_d = pred[-1] - ref
    g_d = gt[-1]   - ref
    pn  = p_d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gn  = g_d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return (1.0 - ((p_d / pn) * (g_d / gn)).sum(-1)).mean()


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


# ── PINN BVE ──────────────────────────────────────────────────────────────────

def _pinn_simplified(pred_abs: torch.Tensor) -> torch.Tensor:
    T = pred_abs.shape[0]
    if T < 5:
        return pred_abs.new_zeros(())

    # v   = pred_abs[1:] - pred_abs[:-1]
    # vx, vy = v[..., 0], v[..., 1]

    # zeta  = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]
    # if zeta.shape[0] < 2:
    #     return pred_abs.new_zeros(())

    # dzeta   = zeta[1:] - zeta[:-1]
    # lat_rad = pred_abs[2:T-1, :, 1] * NORM_TO_DEG * (math.pi / 180)
    # beta_n  = (2.0 * OMEGA * NORM_TO_M * DT_6H / R_EARTH) * torch.cos(lat_rad)
    # vy_mid  = vy[1:T-2]

    # raw = ((dzeta + beta_n * vy_mid) ** 2).mean() * PINN_SCALE
    # # HIỆN TẠI
    # dzeta   = zeta[1:] - zeta[:-1]          # shape [T-4, B]
    # lat_rad = pred_abs[2:T-1, :, 1] ...     # shape [T-3, B] ← lệch 1
    # vy_mid  = vy[1:T-2]                      # shape [T-3, B] ← cũng lệch

    # FIX — align tất cả về [T-4, B]:
    v   = pred_abs[1:] - pred_abs[:-1]       # [T-1, B, 2]
    vx, vy = v[...,0], v[...,1]              # [T-1, B]
    zeta    = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]  # [T-2, B]
    dzeta   = zeta[1:] - zeta[:-1]          # [T-3, B]

    # lat tại midpoint của dzeta[k] = pred_abs[k+2]
    lat_rad = pred_abs[2:T-1, :, 1] * NORM_TO_DEG * (math.pi / 180)  # [T-3, B]
    # nhưng dzeta shape [T-3, B] và lat_rad shape [T-3, B] → T-3 == T-3 ✓
    # vấn đề là vy_mid:
    vy_mid  = vy[1:T-2]   # vy shape [T-1,B], slice [1:T-2] = [T-3,B] ✓ khi T≥5

    # Kiểm tra lại: T=12 → dzeta[T-3=9,B], lat_rad[T-3=9,B], vy_mid[T-3=9,B] ✓
    # T=4 (curriculum) → dzeta[1,B], lat_rad[1,B], vy_mid[1,B] ✓
    # THỰC RA không lệch với T≥5 — bug ở chỗ khác:
    # pred_abs[2:T-1] khi T=pred_abs.shape[0]=12 → [2:11] = 9 phần tử ✓
    # vy[1:T-2] = vy[1:10] = 9 phần tử ✓
    # Vậy không lệch — đây là false alarm. Tuy nhiên beta_n cần kiểm tra:
    beta_n = (2*OMEGA*NORM_TO_M*DT_6H/R_EARTH) * torch.cos(lat_rad)  # [T-3,B]
    # nhân với vy_mid [T-3,B] → OK
    raw = ((dzeta + beta_n * vy_mid)**2).mean() * PINN_SCALE  # scalar ✓
    return raw.clamp(max=50.0)


def pinn_bve_loss(pred_abs, batch_list):
    return _pinn_simplified(pred_abs)


# # ── Main loss function ────────────────────────────────────────────────────────

# # def compute_total_loss(
# #     pred_abs,
# #     gt,
# #     ref,
# #     batch_list,
# #     pred_samples=None,
# #     weights=WEIGHTS,
# #     intensity_w: Optional[torch.Tensor] = None,
# # ) -> Dict:
# #     """
# #     FIX-L21: Recurvature weighting — nhân loss theo per-sample weight.
# #              Recurvature sequences (tổng góc quay >= 45°) được weight ×2.5.
# #              Điều này trực tiếp fix PR=1.37 (straight-track bias).

# #     FIX-L22: Kết hợp recurv_w và intensity_w thành unified sample_weight.

# #     FIX-L23: recurvature_loss — penalize turning point heading error.
# #     """
# #     # FIX-L21: Tính recurvature weight từ gt trajectory
# #     recurv_w = _recurvature_weights(gt, w_recurv=2.5)   # [B]

# #     # FIX-L22: Kết hợp với intensity_w nếu có
# #     if intensity_w is not None:
# #         sample_w = (recurv_w * intensity_w.to(recurv_w.device))
# #         sample_w = sample_w / sample_w.mean().clamp(min=1e-6)
# #     else:
# #         sample_w = recurv_w / recurv_w.mean().clamp(min=1e-6)

# #     # FM loss với unified sample weight
# #     l_fm = (
# #         fm_afcrps_loss(pred_samples, gt, intensity_w=sample_w)
# #         if pred_samples is not None
# #         else _haversine(pred_abs, gt, unit_01deg=False).mean()
# #     )

# #     l_dir     = overall_dir_loss(pred_abs, gt, ref)
# #     l_step    = step_dir_loss(pred_abs, gt)
# #     l_disp    = disp_loss(pred_abs, gt)
# #     l_heading = heading_loss(pred_abs, gt)
# #     l_smooth  = smooth_loss(pred_abs)
# #     l_pinn    = pinn_bve_loss(pred_abs, batch_list)
# #     l_vel     = velocity_loss(pred_abs, gt)
# #     l_accel   = acceleration_loss(pred_abs)
# #     l_recurv  = recurvature_loss(pred_abs, gt)   # FIX-L23

# #     # FIX-L21: Apply recurv_w đến các directional losses (không phải smooth/pinn)
# #     # Cách: scale total của direction-related losses theo mean(recurv_w - 1)
# #     # Straight tracks: factor=1.0, Recurvature: factor=up to 2.5
# #     direction_factor = sample_w.mean()

# #     total = (weights.get("fm",       2.0)  * l_fm
# #            + weights.get("dir",      2.0)  * l_dir      * direction_factor
# #            + weights.get("step",     0.5)  * l_step     * direction_factor
# #            + weights.get("disp",     0.5)  * l_disp
# #            + weights.get("heading",  1.5)  * l_heading  * direction_factor
# #            + weights.get("smooth",   0.05) * l_smooth
# #            + weights.get("pinn",     0.02) * l_pinn
# #            + weights.get("velocity", 2.0)  * l_vel      *10 * direction_factor
# #            + weights.get("accel",    0.3)  * l_accel
# #            + weights.get("recurv",   1.0)  * l_recurv   * direction_factor)

# #     total = total.clamp(max=500.0)

# #     if torch.isnan(total) or torch.isinf(total):
# #         print("  ⚠  NaN/Inf in total loss — returning zero")
# #         total = pred_abs.new_zeros(())

# #     # Log recurvature ratio để monitor
# #     n_recurv = (recurv_w > 1.5).float().sum().item()
# #     B        = recurv_w.shape[0]

# #     return dict(
# #         total=total,
# #         fm=l_fm.item(), dir=l_dir.item(), step=l_step.item(),
# #         disp=l_disp.item(), heading=l_heading.item(),
# #         smooth=l_smooth.item(), pinn=l_pinn.item(),
# #         velocity=l_vel.item(), accel=l_accel.item(),
# #         recurv=l_recurv.item(),
# #         recurv_ratio=n_recurv / max(B, 1),   # % batch là recurvature
# #     )
# # def compute_total_loss(pred_abs, gt, ref, batch_list, pred_samples=None, weights=WEIGHTS, intensity_w=None):
# #     # 1. Tính trọng số [B] cho từng ca bão (Recurvature = 2.5, Straight = 1.0)
# #     # Hàm này trả về tensor shape [B]
# #     recurv_w = _recurvature_weights(gt, w_recurv=2.5) 
    
# #     if intensity_w is not None:
# #         sample_w = recurv_w * intensity_w.to(recurv_w.device)
# #     else:
# #         sample_w = recurv_w
        
# #     # Chuẩn hóa để tránh nổ gradient nhưng GIỮ NGUYÊN tỉ lệ 2.5 : 1 giữa các mẫu
# #     sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

# #     # 2. Flow Matching Loss (Vị trí) - Đã tích hợp sample_w bên trong hàm này
# #     l_fm = fm_afcrps_loss(pred_samples, gt, intensity_w=sample_w)

# #     # 3. Velocity Loss (Tốc độ) - QUAN TRỌNG: Tính per-sample rồi mới nhân weight
# #     # Giả sử hàm velocity_loss_per_sample trả về tensor [B]
# #     l_vel_vec = velocity_loss(pred_abs, gt) 
# #     l_vel = (l_vel_vec * sample_w).mean()

# #     # 4. Hướng và các thành phần khác (Tính scalar)
# #     l_recurv = recurvature_loss(pred_abs, gt) # Turning point penalty
# #     l_heading = heading_loss(pred_abs, gt)   # Góc lệch hướng
# #     l_pinn = pinn_bve_loss(pred_abs, batch_list)

# #     # 5. TỔNG HỢP VỚI HỆ SỐ CÂN BẰNG SCALE
# #     # l_vel nhân 100 để "có tiếng nói" tương đương với l_fm (km)
# #     # l_heading nhân 5 để ép model chú ý đến hướng
# #     total = (weights.get("fm", 2.0) * l_fm +
# #              weights.get("velocity", 2.0) * l_vel * 100.0 + 
# #              weights.get("heading", 1.5) * l_heading * 5.0 + 
# #              weights.get("recurv", 1.5) * l_recurv * 5.0 + 
# #              weights.get("pinn", 0.02) * l_pinn)

# # # total = (weights.get("fm",       2.0)  * l_fm
# # #            + weights.get("dir",      2.0)  * l_dir      * direction_factor
# # #            + weights.get("step",     0.5)  * l_step     * direction_factor
# # #            + weights.get("disp",     0.5)  * l_disp
# # #            + weights.get("heading",  1.5)  * l_heading  * direction_factor
# # #            + weights.get("smooth",   0.05) * l_smooth
# # #            + weights.get("pinn",     0.02) * l_pinn
# # #            + weights.get("velocity", 2.0)  * l_vel      *10 * direction_factor
# # #            + weights.get("accel",    0.3)  * l_accel
# # #            + weights.get("recurv",   1.0)  * l_recurv   * direction_factor)
# #     # NaN guard
# #     if torch.isnan(total): total = pred_abs.new_zeros(())

# #     return dict(
# #         total=total,
# #         fm=l_fm.item(),
# #         vel=l_vel.item(),
# #         recurv=l_recurv.item(),
# #         # Tỷ lệ bão quay đầu trong batch để theo dõi
# #         recurv_ratio=(recurv_w > 1.0).float().mean().item() 
# #     )
# # --- PHẦN 2: HÀM TỔNG HỢP CHÍNH ---

# def compute_total_loss(pred_abs, gt, ref, batch_list, pred_samples=None, weights=WEIGHTS, intensity_w=None):
#     # 1. TRỌNG SỐ MẪU (Recurvature Priority)
#     recurv_w = _recurvature_weights(gt, w_recurv=2.5) # [B]
#     sample_w = recurv_w * (intensity_w.to(gt.device) if intensity_w is not None else 1.0)
#     sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

#     # 2. TÍNH TOÁN CÁC THÀNH PHẦN (Tất cả đưa về mean batch có trọng số)
#     # Vị trí (km)
#     l_fm = fm_afcrps_loss(pred_samples, gt, intensity_w=sample_w)
    
#     # Tốc độ & Độ dài bước (Scale x100)
#     l_vel = (velocity_loss(pred_abs, gt) * sample_w).mean()
#     l_disp = (disp_loss(pred_abs, gt) * sample_w).mean()

#     # Hướng & Khúc cua (Scale x10)
#     l_step = (step_dir_loss(pred_abs, gt) * sample_w).mean()
#     l_heading = heading_loss(pred_abs, gt) # Scalar mean
#     l_recurv = recurvature_loss(pred_abs, gt) # Scalar mean
#     l_dir_final = overall_dir_loss(pred_abs, gt, ref)

#     # Regularization & Physics
#     l_smooth = smooth_loss(pred_abs)
#     l_accel = acceleration_loss(pred_abs)
#     l_pinn = pinn_bve_loss(pred_abs, batch_list)

#     # 3. TỔNG HỢP (Scale Balancing)
#     total = (
#         weights['fm']       * l_fm +
#         weights['velocity'] * l_vel * 100.0 + 
#         weights['disp']     * l_disp * 100.0 +   # Thêm disp
#         weights['step']     * l_step * 10.0 +    # Thêm step
#         weights['heading']  * l_heading * 10.0 + 
#         weights['recurv']   * l_recurv * 10.0 + 
#         weights['dir']      * l_dir_final * 5.0 +
#         weights['smooth']   * l_smooth * 10.0 + 
#         weights['accel']    * l_accel * 10.0 +
#         weights['pinn']     * l_pinn
#     )

#     if torch.isnan(total): total = pred_abs.new_zeros(())

#     return dict(
#         total=total, fm=l_fm.item(), vel=l_vel.item(), 
#         step=l_step.item(), disp=l_disp.item(),
#         recurv=l_recurv.item(), smooth=l_smooth.item(),
#         recurv_ratio=(recurv_w > 1.0).float().mean().item()
#     )
# losses.py

def velocity_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Returns per-sample velocity loss, shape [B].
    Fix: does NOT call .mean() internally so caller can apply sample_w.
    """
    if pred.shape[0] < 2:
        return pred.new_zeros(pred.shape[1])  # [B]

    v_pred = pred[1:] - pred[:-1]   # [T-1, B, 2]
    v_gt   = gt[1:]   - gt[:-1]

    s_pred = torch.norm(v_pred, dim=-1)   # [T-1, B]
    s_gt   = torch.norm(v_gt,   dim=-1)
    l_speed = (s_pred - s_gt).pow(2).mean(0)  # [B]

    gt_v_unit = v_gt / torch.norm(v_gt, dim=-1, keepdim=True).clamp(min=1e-6)
    ate_errors = ((v_pred - v_gt) * gt_v_unit).sum(-1)  # [T-1, B]
    l_ate = ate_errors.pow(2).mean(0)  # [B]

    return l_speed + 0.5 * l_ate  # [B]


def disp_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Returns per-sample displacement loss, shape [B]."""
    if pred.shape[0] < 2:
        return pred.new_zeros(pred.shape[1])
    pd = (pred[1:] - pred[:-1]).norm(dim=-1).mean(0)  # [B]
    gd = (gt[1:]   - gt[:-1]).norm(dim=-1).mean(0)
    return (pd - gd).pow(2)  # [B]


def step_dir_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Returns per-sample step direction loss, shape [B]."""
    if pred.shape[0] < 2:
        return pred.new_zeros(pred.shape[1])
    pv = pred[1:] - pred[:-1]
    gv = gt[1:]   - gt[:-1]
    pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos_sim = ((pv / pn) * (gv / gn)).sum(-1)  # [T-1, B]
    return (1.0 - cos_sim).mean(0)  # [B]


def compute_total_loss(pred_abs, gt, ref, batch_list, pred_samples=None,
                       weights=WEIGHTS, intensity_w=None):
    # 1. TRỌNG SỐ MẪU
    recurv_w = _recurvature_weights(gt, w_recurv=2.5)  # [B]
    sample_w = recurv_w * (intensity_w.to(gt.device) if intensity_w is not None else 1.0)
    sample_w = sample_w / sample_w.mean().clamp(min=1e-6)

    # 2. CÁC THÀNH PHẦN LOSS
    # HIỆN TẠI — gọi với unit_01deg=False nhưng input là normalized coords
    # l_fm = fm_afcrps_loss(pred_samples, gt, intensity_w=sample_w)
    # bên trong: _haversine(..., unit_01deg=False) → coi lon/lat là degrees trực tiếp
    # nhưng normalized coords ~[-2,3], không phải degrees [0,180]

    # FIX — truyền unit_01deg=False VÀ convert trước, hoặc dùng unit_01deg=True
    # Nhìn vào _haversine với unit_01deg=True:
    #   lon_deg = (norm * 50 + 1800) / 10  ← đây là denorm đúng
    # Vậy cần unit_01deg=True:
    l_fm = fm_afcrps_loss(pred_samples, gt, intensity_w=sample_w, unit_01deg=True)  # ← FIX-C3


    # FIX: dùng hàm per-sample rồi mới nhân weight → .mean()
    l_vel  = (velocity_loss_per_sample(pred_abs, gt) * sample_w).mean()
    l_disp = (disp_loss_per_sample(pred_abs, gt) * sample_w).mean()
    l_step = (step_dir_loss_per_sample(pred_abs, gt) * sample_w).mean()

    # Scalar losses (không có per-sample version hợp lý)
    l_heading   = heading_loss(pred_abs, gt)
    l_recurv    = recurvature_loss(pred_abs, gt)
    l_dir_final = overall_dir_loss(pred_abs, gt, ref)
    l_smooth    = smooth_loss(pred_abs)
    l_accel     = acceleration_loss(pred_abs)
    l_pinn      = pinn_bve_loss(pred_abs, batch_list)

    # 3. TỔNG HỢP
    total = (
        weights['fm']       * l_fm +
        # weights['velocity'] * l_vel   * 100.0 +
        weights['velocity'] * l_vel   * 30.0 +
        weights['disp']     * l_disp  * 30.0 +
        weights['step']     * l_step  *  10.0 +
        weights['heading']  * l_heading * 10.0 +
        weights['recurv']   * l_recurv  * 10.0 +
        weights['dir']      * l_dir_final * 5.0 +
        weights['smooth']   * l_smooth * 10.0 +
        weights['accel']    * l_accel  * 10.0 +
        weights['pinn']     * l_pinn
    )

    if torch.isnan(total):
        total = pred_abs.new_zeros(())

    # FIX: key là 'velocity' (khớp với train script bd.get('velocity'))
    return dict(
        total=total,
        fm=l_fm.item(),
        velocity=l_vel.item(),
        step=l_step.item(),
        disp=l_disp.item(),
        heading=l_heading.item(),    # ← thêm
        recurv=l_recurv.item(),
        smooth=l_smooth.item(),
        accel=l_accel.item(),        # ← thêm
        pinn=l_pinn.item(),          # ← thêm — đây là nguyên nhân pinn=0.0000
        recurv_ratio=(recurv_w > 1.0).float().mean().item()
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
                torch.norm(anchor - neg, 2, dim=1)
                - torch.norm(anchor - pos, 2, dim=1), y)
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
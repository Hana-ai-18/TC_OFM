# """
# Model/losses.py  ── v10 (algorithm unchanged from v9)
# ======================================================
# Loss functions: OT-CFM + PINN-BVE.

# L1  L_FM      : afCRPS
# L2  L_dir     : overall direction
# L3  L_step    : per-step direction
# L4  L_disp    : displacement speed matching
# L5  L_heading : anti-parallel penalty + curvature MSE
# L6  L_smooth  : 2nd-order finite differences
# L7  L_PINN    : beta-plane BVE residual

# Total: L = 1.0·L1 + 2.0·L2 + 0.5·L3 + 1.0·L4 + 2.0·L5 + 0.2·L6 + 0.5·L7
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

# WEIGHTS: Dict[str, float] = dict(
#     fm=1.0, dir=2.0, step=0.5, disp=1.0, heading=2.0, smooth=0.2, pinn=0.5,
# )


# def _haversine(p1: torch.Tensor, p2: torch.Tensor,
#                unit_01deg: bool = True) -> torch.Tensor:
#     scale = 10.0 if unit_01deg else 1.0
#     lat1  = torch.deg2rad(p1[..., 1] * (1.0 / scale))
#     lat2  = torch.deg2rad(p2[..., 1] * (1.0 / scale))
#     dlon  = torch.deg2rad((p2[..., 0] - p1[..., 0]) * (1.0 / scale))
#     dlat  = torch.deg2rad((p2[..., 1] - p1[..., 1]) * (1.0 / scale))
#     a     = (torch.sin(dlat / 2) ** 2
#              + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
#     return 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())


# def fm_afcrps_loss(pred_samples: torch.Tensor, gt: torch.Tensor,
#                    unit_01deg: bool = False) -> torch.Tensor:
#     M, T, B, _ = pred_samples.shape
#     if M == 1:
#         return _haversine(pred_samples[0], gt, unit_01deg).mean()
#     total, n_pairs = gt.new_zeros(()), 0
#     for s in range(M):
#         for sp in range(M):
#             if s == sp:
#                 continue
#             d_sy  = _haversine(pred_samples[s],  gt,               unit_01deg)
#             d_spy = _haversine(pred_samples[sp], gt,               unit_01deg)
#             d_ssp = _haversine(pred_samples[s],  pred_samples[sp], unit_01deg)
#             total   = total + (d_sy + d_spy - d_ssp).clamp(min=0).mean()
#             n_pairs += 1
#     return total / (2.0 * n_pairs)


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


# def _pinn_simplified(pred_abs):
#     T = pred_abs.shape[0]
#     if T < 4:
#         return pred_abs.new_zeros(())
#     v   = pred_abs[1:] - pred_abs[:-1]
#     vx, vy = v[..., 0], v[..., 1]
#     zeta   = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]
#     if zeta.shape[0] < 2:
#         return pred_abs.new_zeros(())
#     dzeta   = zeta[1:] - zeta[:-1]
#     lat_rad = pred_abs[2:T-1, :, 1] * NORM_TO_DEG * (math.pi / 180)
#     beta_n  = (2.0 * OMEGA * NORM_TO_M * DT_6H / R_EARTH) * torch.cos(lat_rad)
#     return ((dzeta + beta_n * vy[1:T-2]) ** 2).mean() * PINN_SCALE


# def pinn_bve_loss(pred_abs, batch_list):
#     return _pinn_simplified(pred_abs)


# def compute_total_loss(pred_abs, gt, ref, batch_list,
#                        pred_samples=None, weights=WEIGHTS) -> Dict:
#     l_fm      = fm_afcrps_loss(pred_samples, gt) if pred_samples is not None \
#                 else _haversine(pred_abs, gt, unit_01deg=False).mean()
#     l_dir     = overall_dir_loss(pred_abs, gt, ref)
#     l_step    = step_dir_loss(pred_abs, gt)
#     l_disp    = disp_loss(pred_abs, gt)
#     l_heading = heading_loss(pred_abs, gt)
#     l_smooth  = smooth_loss(pred_abs)
#     l_pinn    = pinn_bve_loss(pred_abs, batch_list)

#     total = (weights["fm"]      * l_fm
#            + weights["dir"]     * l_dir
#            + weights["step"]    * l_step
#            + weights["disp"]    * l_disp
#            + weights["heading"] * l_heading
#            + weights["smooth"]  * l_smooth
#            + weights["pinn"]    * l_pinn)

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
#     lon_km = diff[:, :, 0] / 10.0 * 111.0 * torch.cos(_gt[:, :, 1] / 10.0 * torch.pi / 180.0)
#     lat_km = diff[:, :, 1] / 10.0 * 111.0
#     loss   = torch.sqrt(lon_km ** 2 + lat_km ** 2)
#     return torch.sum(loss) if mode == "sum" else loss


# def evaluate_diffusion_output(best_traj, best_Me, gt_traj, gt_Me):
#     rt, rm  = toNE(best_traj.clone(), best_Me.clone())
#     rg, rgm = toNE(gt_traj.clone(),  gt_Me.clone())
#     return (trajectory_displacement_error(rt, rg, mode="raw"),
#             torch.abs(rm.permute(1, 0, 2) - rgm.permute(1, 0, 2)))

"""
Model/losses.py  ── v11
========================
CHANGES vs v10-fixed-2:
  FIX-L5  smooth weight 0.2 → 0.05 (reduces over-smoothing of track).
  FIX-L6  compute_total_loss accepts optional intensity_w [B] tensor
           for per-sample weighting (TY/SevTY more important).
  FIX-L7  fm_afcrps_loss also used as explicit CRPS training signal
           (already was, but now weighted by intensity_w).
  FIX-L8  _haversine: unit_01deg=False path was computing scale=1.0
           which is correct for normalised coords; added comment.
  Kept from v10-fixed-2:
    FIX-L1  PINN weight 0.5 → 0.1
    FIX-L2  _pinn_simplified clamp max=50
    FIX-L3  fm_afcrps_loss M==1 fallback
    FIX-L4  total clamp max=500
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
PINN_SCALE   = 100.0

# FIX-L5: smooth 0.2→0.05 (was over-smoothing track curves)
# FIX-L1: pinn 0.5→0.1 (kept from v10)
WEIGHTS: Dict[str, float] = dict(
    fm=1.0, dir=2.0, step=0.5, disp=1.0, heading=2.0, smooth=0.05, pinn=0.05,
)


def _haversine(p1: torch.Tensor, p2: torch.Tensor,
               unit_01deg: bool = True) -> torch.Tensor:
    """
    Haversine distance in km.
    unit_01deg=True: coords are normalised (×50+1800 → 0.1°), divide by 10.
    unit_01deg=False: coords in degrees already (scale=1.0).
    """
    scale = 10.0 if unit_01deg else 1.0
    lat1  = torch.deg2rad(p1[..., 1] * (1.0 / scale))
    lat2  = torch.deg2rad(p2[..., 1] * (1.0 / scale))
    dlon  = torch.deg2rad((p2[..., 0] - p1[..., 0]) * (1.0 / scale))
    dlat  = torch.deg2rad((p2[..., 1] - p1[..., 1]) * (1.0 / scale))
    a     = (torch.sin(dlat / 2) ** 2
             + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
    return 2.0 * 6371.0 * torch.asin(a.clamp(0.0, 1.0).sqrt())


def fm_afcrps_loss(pred_samples: torch.Tensor, gt: torch.Tensor,
                   unit_01deg: bool = False,
                   intensity_w: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    AFCRPS energy-form loss.
    pred_samples: [M, T, B, 2]
    gt:           [T, B, 2]
    intensity_w:  [B] optional per-sample weights

    FIX-L7: intensity weighting applied if provided.
    """
    M, T, B, _ = pred_samples.shape
    if M == 1:
        loss_per_b = _haversine(pred_samples[0], gt, unit_01deg).mean(0)  # [B]
    else:
        total, n_pairs = gt.new_zeros(B), 0
        for s in range(M):
            for sp in range(M):
                if s == sp:
                    continue
                d_sy  = _haversine(pred_samples[s],  gt,               unit_01deg).mean(0)
                d_spy = _haversine(pred_samples[sp], gt,               unit_01deg).mean(0)
                d_ssp = _haversine(pred_samples[s],  pred_samples[sp], unit_01deg).mean(0)
                total   = total + (d_sy + d_spy - d_ssp).clamp(min=0)
                n_pairs += 1
        loss_per_b = total / (2.0 * n_pairs)  # [B]

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


def _pinn_simplified(pred_abs):
    T = pred_abs.shape[0]
    if T < 4:
        return pred_abs.new_zeros(())
    v   = pred_abs[1:] - pred_abs[:-1]
    vx, vy = v[..., 0], v[..., 1]
    zeta   = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]
    if zeta.shape[0] < 2:
        return pred_abs.new_zeros(())
    dzeta   = zeta[1:] - zeta[:-1]
    lat_rad = pred_abs[2:T-1, :, 1] * NORM_TO_DEG * (math.pi / 180)
    beta_n  = (2.0 * OMEGA * NORM_TO_M * DT_6H / R_EARTH) * torch.cos(lat_rad)
    raw = ((dzeta + beta_n * vy[1:T-2]) ** 2).mean() * PINN_SCALE
    return raw.clamp(max=5.0)


def pinn_bve_loss(pred_abs, batch_list):
    return _pinn_simplified(pred_abs)


def compute_total_loss(pred_abs, gt, ref, batch_list,
                       pred_samples=None, weights=WEIGHTS,
                       intensity_w: Optional[torch.Tensor] = None) -> Dict:
    """
    FIX-L6: intensity_w [B] applied to fm_afcrps_loss.
    FIX-L5: smooth weight is now 0.05 in default WEIGHTS.
    """
    l_fm      = fm_afcrps_loss(pred_samples, gt, intensity_w=intensity_w) \
                if pred_samples is not None \
                else _haversine(pred_abs, gt, unit_01deg=False).mean()
    l_dir     = overall_dir_loss(pred_abs, gt, ref)
    l_step    = step_dir_loss(pred_abs, gt)
    l_disp    = disp_loss(pred_abs, gt)
    l_heading = heading_loss(pred_abs, gt)
    l_smooth  = smooth_loss(pred_abs)
    l_pinn    = pinn_bve_loss(pred_abs, batch_list)

    total = (weights["fm"]      * l_fm
           + weights["dir"]     * l_dir
           + weights["step"]    * l_step
           + weights["disp"]    * l_disp
           + weights["heading"] * l_heading
           + weights["smooth"]  * l_smooth
           + weights["pinn"]    * l_pinn)

    total = total.clamp(max=500.0)

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
    lon_km = diff[:, :, 0] / 10.0 * 111.0 * torch.cos(_gt[:, :, 1] / 10.0 * torch.pi / 180.0)
    lat_km = diff[:, :, 1] / 10.0 * 111.0
    loss   = torch.sqrt(lon_km ** 2 + lat_km ** 2)
    return torch.sum(loss) if mode == "sum" else loss


def evaluate_diffusion_output(best_traj, best_Me, gt_traj, gt_Me):
    rt, rm  = toNE(best_traj.clone(), best_Me.clone())
    rg, rgm = toNE(gt_traj.clone(),  gt_Me.clone())
    return (trajectory_displacement_error(rt, rg, mode="raw"),
            torch.abs(rm.permute(1, 0, 2) - rgm.permute(1, 0, 2)))
"""

# ─────────────────────────────────────────────────────────────────────────────
# FILE PLACEMENT:
#
#   SOURCE:        ablation_runner.py
#   KAGGLE TARGET: /kaggle/working/ablation_runner.py
#   LOCAL DEV:     ablation_runner.py
#
#   ODE steps sweep (sau khi có checkpoint):
#   python ablation_runner.py --mode ode_steps \
#     --checkpoint runs/best_model.pth \
#     --dataset_root /kaggle/input/datasets/tc-ofm \
#     --output_dir ablations/
#
#   Summarize ablation JSONs:
#   python ablation_runner.py --mode summarize --ablation_dir ablations/
# ─────────────────────────────────────────────────────────────────────────────

ablation_runner.py — ESWA Section C: Ablation Studies
════════════════════════════════════════════════════════════════════════════════
Ablation experiments required by ESWA paper:
  C4. Loss function ablation: no L_heading, no L_calib, no AUG-C, no learned weights
  C5. ODE steps: 1 vs 2 vs 5 vs 10 (accuracy vs compute trade-off)
  C6. Input features: track-only vs track+ERA5
  C7. FM vs Diffusion same backbone (generation mechanism only)

Usage:
  # A) Ablation via config flags (train a specific variant):
  python ablation_runner.py \\
    --mode train_variant \\
    --variant no_L_heading \\
    --dataset_root /path/to/tc-ofm \\
    --output_dir ablations/

  # B) ODE steps sweep (inference only, from existing checkpoint):
  python ablation_runner.py \\
    --mode ode_steps \\
    --checkpoint runs/best_model.pth \\
    --dataset_root /path/to/tc-ofm \\
    --output_dir ablations/

  # C) Summarize all ablation results from JSON files:
  python ablation_runner.py \\
    --mode summarize \\
    --ablation_dir ablations/
"""
from __future__ import annotations

import os, sys, argparse, json, time, copy
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, EMAModel,
    _norm_to_deg, _haversine_deg, _unwrap,
    augment_batch,
)


def _ate_cte_full_local(pred_deg, gt_deg):
    """
    Local copy of evaluate_full.py's _ate_cte_full (identical formula —
    ATE/CTE signed, [T-1, B] shape). Copied rather than imported to
    remove the circular dependency that existed before (evaluate_full.py
    imports ode_steps_sweep from THIS file at module load; this file
    used to import _ate_cte_full back from evaluate_full.py inside a
    function body). That lazy in-function import happened to not crash
    because it only ran when the function was called, never at module
    load time — but it was fragile (moving it to module level would
    break both files), so it's removed in favor of this local copy.
    If evaluate_full.py's _ate_cte_full formula ever changes, update
    both copies.
    """
    from Model.flow_matching_model import _forward_azimuth
    import torch as _torch
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    bear_ref = _forward_azimuth(gt_deg[:T-1], gt_deg[1:T])
    bear_err = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])
    dist_err = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang      = bear_err - bear_ref
    return dist_err * _torch.cos(ang), dist_err * _torch.sin(ang)

# ─────────────────────────────────────────────────────────────────────────────
#  Ablation variant definitions
# ─────────────────────────────────────────────────────────────────────────────

# Each variant specifies which components to disable/modify
ABLATION_VARIANTS = {
    # ── Full model (baseline for ablation table) ───────────────────────────
    "full": {
        "desc": "Full model v2.5 (L_CFM+L_reg+L_heading+L_calib_along_track+AUG-C+Kendall)",
        "disable_l_heading": False, "disable_l_calib": False,
        "disable_aug_c": False, "disable_learned_weights": False,
        "disable_l_reg": False, "n_inference_steps": 1,
    },
    # ── Loss component ablations ───────────────────────────────────────────
    "no_L_heading": {
        "desc": "w/o L_heading (no multi-step heading constraint)",
        "disable_l_heading": True, "disable_l_calib": False,
        "disable_aug_c": False, "disable_learned_weights": False,
        "disable_l_reg": False, "n_inference_steps": 1,
    },
    "no_L_calib": {
        "desc": "w/o L_calib (no along-track speed calibration training)",
        "disable_l_heading": False, "disable_l_calib": True,
        "disable_aug_c": False, "disable_learned_weights": False,
        "disable_l_reg": False, "n_inference_steps": 1,
    },
    "no_L_reg": {
        "desc": "w/o L_reg (no ADE auxiliary loss)",
        "disable_l_heading": False, "disable_l_calib": False,
        "disable_aug_c": False, "disable_learned_weights": False,
        "disable_l_reg": True, "n_inference_steps": 1,
    },
    "cfm_only": {
        "desc": "L_CFM only — no auxiliary losses",
        "disable_l_heading": True, "disable_l_calib": True,
        "disable_aug_c": False, "disable_learned_weights": False,
        "disable_l_reg": True, "n_inference_steps": 1,
    },
    "no_L_reg_heading": {
        "desc": "w/o L_reg + L_heading (L_calib only)",
        "disable_l_heading": True, "disable_l_calib": False,
        "disable_aug_c": False, "disable_learned_weights": False,
        "disable_l_reg": True, "n_inference_steps": 1,
    },
    # ── Augmentation ablations ─────────────────────────────────────────────
    "no_aug_c": {
        "desc": "w/o AUG-C recurvature augmentation",
        "disable_l_heading": False, "disable_l_calib": False,
        "disable_aug_c": True, "disable_learned_weights": False,
        "disable_l_reg": False, "n_inference_steps": 1,
    },
    "no_aug": {
        "desc": "w/o ALL augmentation (A+B+C disabled)",
        "disable_l_heading": False, "disable_l_calib": False,
        "disable_aug_c": True, "disable_learned_weights": False,
        "disable_l_reg": False, "n_inference_steps": 1,
        "disable_all_aug": True,   # extra flag handled in AblationModel
    },
    # ── Learning mechanism ablations ───────────────────────────────────────
    "no_learned_weights": {
        "desc": "Fixed lambda weights (no Kendall uncertainty weighting)",
        "disable_l_heading": False, "disable_l_calib": False,
        "disable_aug_c": False, "disable_learned_weights": True,
        "disable_l_reg": False, "n_inference_steps": 1,
    },
    "no_speed_correction": {
        "desc": "w/o per-horizon speed correction (correction=1.0 at inference)",
        "disable_l_heading": False, "disable_l_calib": True,
        "disable_aug_c": False, "disable_learned_weights": False,
        "disable_l_reg": False, "n_inference_steps": 1,
        "freeze_speed_correction": True,
    },
    # ── Ensemble size ablation (inference only) ────────────────────────────
    "ensemble_5":  {"desc": "K=5 ensemble", "n_inference_steps": 1,
                    "n_ensemble_override": 5},
    "ensemble_10": {"desc": "K=10 ensemble", "n_inference_steps": 1,
                    "n_ensemble_override": 10},
    "ensemble_20": {"desc": "K=20 ensemble (default)", "n_inference_steps": 1,
                    "n_ensemble_override": 20},
    "ensemble_40": {"desc": "K=40 ensemble", "n_inference_steps": 1,
                    "n_ensemble_override": 40},
    # ── ODE steps (inference only) ─────────────────────────────────────────
    "ode_1step":  {"desc": "1-step ODE (Euler)", "n_inference_steps": 1},
    "ode_2step":  {"desc": "2-step ODE",          "n_inference_steps": 2},
    "ode_5step":  {"desc": "5-step ODE",          "n_inference_steps": 5},
    "ode_10step": {"desc": "10-step ODE",         "n_inference_steps": 10},

    # NEWLY ADDED -- components that had a real, working constructor
    # flag on TCFlowMatching but NO ablation entry, so they were never
    # actually tested despite the flag existing. VERIFIED against
    # train_flowmatching.py's actual argparse block (not assumed):
    #   --no_ot              -> use_ot=False              [READY]
    #   --disable_horizon_nll -> enable_horizon_nll=False  [READY]
    #   --disable_hard_reg    -> lambda_hard_reg forced 0  [READY]
    #   --no_ema              -> use_ema=False             [READY, inference-only via evaluate_full.py --no_ema, no retrain needed]
    # These need RETRAINING (they change training-time behavior), run
    # via `python train_flowmatching.py <the corresponding --flag>` --
    # NOT via ablation_runner.py's --mode train_variant/AblationModel
    # path, which does NOT forward these particular kwargs today.
    "no_ot_matching": {
        "desc": "w/o OT (Sinkhorn) coupling -- random x0/x1 pairing",
        "train_flag": "--no_ot", "_status": "READY -- real train_flowmatching.py flag",
    },
    "no_horizon_nll": {
        "desc": "w/o log_b_horizon Laplace-NLL horizon normalization",
        "train_flag": "--disable_horizon_nll", "_status": "READY -- real train_flowmatching.py flag",
    },
    "no_hard_reg": {
        "desc": "w/o lambda_hard_reg pull-to-uniform on hard_score weights",
        "train_flag": "--disable_hard_reg", "_status": "READY -- real train_flowmatching.py flag",
    },
    "no_ema": {
        "desc": "w/o EMA weights at inference (raw trained weights)",
        "train_flag": "--no_ema",
        "_status": "READY, but INFERENCE-ONLY -- use evaluate_full.py "
                   "--no_ema on an EXISTING checkpoint (any checkpoint "
                   "trained with EMA on works fine for this comparison; "
                   "no retraining needed at all, unlike the other 3).",
    },
    # NOT YET IMPLEMENTED -- these describe real architectural
    # components (see NEW_VARIANT_RATIONALE below for why each matters)
    # but, unlike the 4 above, VERIFIED to have NO existing constructor
    # kwarg or CLI flag in flow_matching_model.py / train_flowmatching.py.
    # Listed here as a TODO, not a runnable variant -- do not attempt
    # `--variant fixed_hard_score_weights` or `--variant no_kinematic_feat`,
    # they will silently no-op (AblationModel has no code path for them).
    "fixed_hard_score_weights": {
        "desc": "[NOT IMPLEMENTED] hard_score with FIXED weights "
                "(0.35/0.25/0.25/0.15) instead of learned weight_logits",
        "_status": "NEEDS NEW CODE -- requires a new constructor kwarg "
                   "(e.g. freeze_hard_score_weights: bool) that, in "
                   "hard_score_from_obs()'s call site, passes a fixed "
                   "tensor instead of self.hard_score_weight_logits, "
                   "and disables that Parameter's gradient. Not present "
                   "in flow_matching_model.py today.",
    },
    "no_kinematic_feat": {
        "desc": "[NOT IMPLEMENTED] ContextEncoder w/o 6-dim kinematic "
                "feature branch",
        "_status": "NEEDS NEW CODE -- requires a new constructor kwarg "
                   "threaded into ContextEncoder.__init__ / .forward to "
                   "zero out or skip self.vel_obs_enc and adjust self.fuse's "
                   "input dim accordingly (currently hardcoded to "
                   "d_cond + d_cond//2 + d_cond//4). Not present today.",
    },
}

# Rationale for each newly-added variant (kept out of the dict above to
# keep it readable -- ABLATION_VARIANTS stays a flat config table):
NEW_VARIANT_RATIONALE = {
    "no_ot_matching": (
        "OT matching (_ot_match/_sinkhorn_log) is claimed to give smoother "
        "conditional flow paths than i.i.d. random pairing, but this is "
        "asserted from general FM literature, not measured on this "
        "dataset. --no_ot is a real, ready-to-run flag with zero ablation "
        "evidence collected yet."
    ),
    "no_horizon_nll": (
        "Newest of the 4 CTE/spread fixes (log train only reached ~ep81 "
        "at time of writing) and the LEAST validated -- never isolated "
        "from the other 3 fixes (sigma_min/max, sigma_decay_end, "
        "n_inference_steps) in the current run. Needed to know how much "
        "of the CTE gain is horizon-NLL specifically vs the sigma changes."
    ),
    "no_hard_reg": (
        "hard_score's 4 component weights are LEARNED (LEARN-3) with "
        "lambda_hard_reg=0.02 pulling toward uniform to prevent collapse. "
        "No evidence yet that the regularizer at 0.02 does anything vs 0.0."
    ),
    "no_ema": (
        "EMA (decay=0.995) is on by default for reported paper numbers, "
        "but there is no side-by-side EMA-vs-raw table showing how much "
        "of the final ADE/ATE/CTE improvement is attributable to EMA "
        "specifically. Inference-only -- use evaluate_full.py --no_ema "
        "on an EXISTING checkpoint, no retraining needed."
    ),
    "fixed_hard_score_weights": (
        "Directly answers whether learning the hard_score weights helps "
        "at all vs the original hand-picked 0.35/0.25/0.25/0.15. Currently "
        "asserted, not tested -- AND not yet even codeable without adding "
        "a new constructor kwarg first (see _status above)."
    ),
    "no_kinematic_feat": (
        "ContextEncoder fuses 3 feature sources; only the whole encoder's "
        "value is implicitly tested via baseline comparison. No ablation "
        "isolates whether the kinematic branch specifically contributes "
        "vs being redundant with what the 1D/spatial encoders already "
        "see -- AND not yet even codeable without adding a new "
        "constructor kwarg first (see _status above)."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
#  Modified model for ablation (patches get_loss_breakdown at runtime)
# ─────────────────────────────────────────────────────────────────────────────

class AblationModel(TCFlowMatching):
    """
    TCFlowMatching with ablation switches injected.
    Inherits all parameters from base class, only disables specific loss terms.
    """
    def __init__(self, variant_cfg: Dict, **kwargs):
        super().__init__(**kwargs)
        self._abl = variant_cfg

    def get_loss_breakdown(self, batch_list, epoch: int = 0, **kwargs):
        import math as _math
        import torch.nn.functional as F

        bd = super().get_loss_breakdown(batch_list, epoch=epoch, **kwargs)

        # Re-compute total with ablations
        # We override specific terms to zero
        abl = self._abl

        # Get the current total (float tensor)
        total = bd["total"]
        device = total.device

        # Apply ablations by zeroing out terms
        # Note: super() already computed total; we rebuild it
        l_cfm     = torch.tensor(bd["l_cfm"],    device=device, requires_grad=False)
        l_reg     = torch.tensor(bd["l_reg"],    device=device)
        l_heading = torch.tensor(bd["l_heading"], device=device)
        l_calib   = torch.tensor(bd["l_calib"],  device=device)

        if abl.get("disable_l_reg", False):
            l_reg = torch.zeros((), device=device)
        if abl.get("disable_l_heading", False):
            l_heading = torch.zeros((), device=device)
        if abl.get("disable_l_calib", False):
            l_calib = torch.zeros((), device=device)

        if abl.get("disable_learned_weights", False):
            # Use fixed lambda weights (baseline behavior)
            lam_reg  = bd["lam_reg"] * 0.2
            lam_dir  = bd["lam_dir"] * 0.07
            lam_cal  = bd["lam_calib"] * 0.1
            total = l_cfm + lam_reg * l_reg + lam_dir * l_heading + lam_cal * l_calib
        else:
            # Already computed by super() — just rebuild with zeroed terms
            HALF = 0.5 * _math.log(2.0 * _math.pi)
            prec_r = torch.exp(-2.0 * self.log_sigma_reg.clamp(min=-3.0))
            prec_h = torch.exp(-2.0 * self.log_sigma_heading.clamp(min=-3.0))
            prec_c = torch.exp(-2.0 * self.log_sigma_calib.clamp(min=-3.0))
            lam_r = bd["lam_reg"];  lam_d = bd["lam_dir"]; lam_c = bd["lam_calib"]
            total = (l_cfm
                     + lam_r * (0.5 * prec_r * l_reg     + self.log_sigma_reg.clamp(-3) + HALF)
                     + lam_d * (0.5 * prec_h * l_heading + self.log_sigma_heading.clamp(-3) + HALF)
                     + lam_c * (0.5 * prec_c * l_calib   + self.log_sigma_calib.clamp(-3) + HALF))
            if not torch.isfinite(total):
                total = torch.zeros((), device=device)

        bd["total"] = total
        bd["l_reg"]     = float(l_reg)
        bd["l_heading"] = float(l_heading)
        bd["l_calib"]   = float(l_calib)
        return bd

    def get_loss(self, batch_list, epoch: int = 0, **kwargs):
        return self.get_loss_breakdown(batch_list, epoch=epoch)["total"]

    def augment_batch(self, batch_list, device):
        """
        Override to disable AUG-C if requested.

        [FIX] Trước đây khi disable_aug_c=True, hàm này tắt TOÀN BỘ 6
        nhánh augment (A/B/C/D-E/F), không chỉ riêng AUG-C (nhánh C,
        recurvature) — comment cũ tự thừa nhận đây là "simplest approach"
        chứ không phải đúng ý nghĩa tên biến disable_aug_c. Giờ
        augment_batch() (flow_matching_model.py) đã có tham số disable_c
        thật (thêm cùng lúc với fix --disable_aug_c ở train_flowmatching.py
        — trước đó --disable_aug_c ở train_flowmatching.py cũng hoàn toàn
        không có tác dụng vì cùng lý do), dùng đúng nó để chỉ tắt nhánh C,
        giữ nguyên A/B/D-E/F — khớp đúng ý nghĩa "w/o AUG-C" trong
        ABLATION_VARIANTS["no_aug_c"]'s desc, và nhất quán với
        train_flowmatching.py's --disable_aug_c giờ đã hoạt động đúng.

        disable_all_aug (variant "no_aug"): vẫn giữ hành vi tắt TOÀN BỘ
        augmentation như cũ — đây là ablation "no_aug" thật (w/o ALL
        augmentation A+B+C), khác no_aug_c ở chỗ tắt cả A/B/D-E/F, không
        chỉ riêng C.
        """
        if self._abl.get("disable_all_aug", False):
            return [b.clone() if torch.is_tensor(b) else b for b in batch_list]
        return augment_batch(batch_list,
                             disable_c=self._abl.get("disable_aug_c", False))


# ─────────────────────────────────────────────────────────────────────────────
#  ODE steps sweep (inference only)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ode_steps_sweep(model, loader, device,
                     steps_list: List[int] = [1, 4, 8, 10, 12, 16, 20],
                     n_ensemble: int = 20,
                     collect_spread: bool = True) -> Dict:
    """
    Evaluate model with different numbers of ODE integration (N)
    steps. Reports ADE/ATE/CTE (not just ADE, as the earlier version
    did -- the project's own experiment log shows N is NOT monotonic
    across all 3 metrics: N=1->10 improves spread/CRPS but slightly
    WORSENS ADE/ATE/CTE, so a table that only shows ADE would hide
    that trade-off) plus ensemble spread (mean pairwise distance
    between ensemble members at the final step), the metric that
    actually motivated raising N in the first place.

    default steps_list = [1,4,8,10,12,16,20] matches the ablation the
    project asked for. N=16 here is an ODE integration step count,
    unrelated to pred_len=16 from the reference paper -- same
    distinction flagged before, restated so this output isn't misread.

    [BỔ SUNG, per-lead-time] Trước đây d.mean(0) gộp trung bình qua
    TOÀN BỘ T lead-time (6h..72h) trước khi lưu -- mỗi storm chỉ đóng
    góp 1 con số duy nhất, không có cách nào biết N ảnh hưởng CTE tại
    72h khác gì tại 6h. Giờ vẫn giữ "*_mean"/"*_std" tổng thể (tương
    thích ngược với print_ode_sweep/plot_ode_n_sweep, KHÔNG đổi hành vi
    hiện có), nhưng THÊM "by_lead_time" -- dict {lead_time: {ADE/ATE/
    CTE mean}} dùng CÙNG convention 1-indexed (1=6h...T=72h) đã thống
    nhất với evaluate_multi_model.py, để build bảng/plot per-horizon
    cho riêng ablation N nếu cần sau này.
    """
    model.eval()
    results = {}
    raw = _unwrap(model)
    from Model.flow_matching_model import _forward_azimuth

    for n_steps in steps_list:
        print(f"  ODE steps={n_steps}...")
        all_ade, all_ate, all_cte, all_spread = [], [], [], []
        by_lt_ade = defaultdict(list)
        by_lt_ate = defaultdict(list)
        by_lt_cte = defaultdict(list)
        t_start = time.time()

        for batch in loader:
            bl = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            gt = bl[1]

            try:
                orig_steps = getattr(raw, "n_inference_steps", 1)
                raw.n_inference_steps = n_steps
                pred, _, all_t = model.sample(bl, num_ensemble=n_ensemble)
                raw.n_inference_steps = orig_steps
            except Exception as e:
                print(f"    Error: {e}"); continue

            T   = min(pred.shape[0], gt.shape[0])
            pd  = _norm_to_deg(pred[:T])
            gd  = _norm_to_deg(gt[:T, :, :2])
            d   = _haversine_deg(pd, gd)          # [T, B]
            all_ade.extend(d.mean(0).tolist())     # tổng thể (giữ nguyên, tương thích ngược)
            for step0 in range(T):                 # per-lead-time: 0-indexed step -> 1-indexed lead_time
                by_lt_ade[step0 + 1].extend(d[step0].tolist())

            if T >= 2:
                bear_ref = _forward_azimuth(gd[:T-1], gd[1:T])
                bear_err = _forward_azimuth(gd[1:T], pd[1:T])
                dist_err = _haversine_deg(pd[1:T], gd[1:T])
                ang = bear_err - bear_ref
                ate_full = (dist_err * torch.cos(ang)).abs()   # [T-1, B]
                cte_full = (dist_err * torch.sin(ang)).abs()   # [T-1, B]
                all_ate.extend(ate_full.mean(0).tolist())
                all_cte.extend(cte_full.mean(0).tolist())
                # ate_full[k] ứng original step k+1 (0-indexed) = lead_time k+2
                # (không định nghĩa ở lead_time=1/6h -- xem
                # evaluate_multi_model.py's docstring cho lý do toán học)
                for k in range(T - 1):
                    by_lt_ate[k + 2].extend(ate_full[k].tolist())
                    by_lt_cte[k + 2].extend(cte_full[k].tolist())

            if collect_spread and all_t is not None and all_t.shape[0] >= 2:
                K = all_t.shape[0]
                last = _norm_to_deg(all_t[:, -1, :, :2])   # [K, B, 2]
                idx1 = torch.randperm(K)[:min(K, 10)]
                idx2 = torch.randperm(K)[:min(K, 10)]
                for a, b in zip(idx1.tolist(), idx2.tolist()):
                    if a != b:
                        dab = _haversine_deg(last[a:a+1], last[b:b+1]).squeeze(0)
                        all_spread.append(float(dab.mean()))

        elapsed = time.time() - t_start
        by_lead_time = {}
        all_lts = sorted(set(by_lt_ade.keys()) | set(by_lt_ate.keys()) | set(by_lt_cte.keys()))
        for lt in all_lts:
            by_lead_time[lt] = {
                "ADE": float(np.mean(by_lt_ade[lt])) if by_lt_ade.get(lt) else float("nan"),
                "ATE": float(np.mean(by_lt_ate[lt])) if by_lt_ate.get(lt) else float("nan"),
                "CTE": float(np.mean(by_lt_cte[lt])) if by_lt_cte.get(lt) else float("nan"),
                "n":   len(by_lt_ade.get(lt, [])),
            }

        results[n_steps] = {
            "n_steps":     n_steps,
            "ADE_mean":    float(np.mean(all_ade)) if all_ade else float("nan"),
            "ADE_std":     float(np.std(all_ade))  if all_ade else float("nan"),
            "ATE_mean":    float(np.mean(all_ate)) if all_ate else float("nan"),
            "CTE_mean":    float(np.mean(all_cte)) if all_cte else float("nan"),
            "spread_mean": float(np.mean(all_spread)) if all_spread else float("nan"),
            "time_s":      elapsed,
            "n_storms":    len(all_ade),
            "by_lead_time": by_lead_time,
        }
        r = results[n_steps]
        print(f"    ADE={r['ADE_mean']:.2f}  ATE={r['ATE_mean']:.2f}  "
              f"CTE={r['CTE_mean']:.2f}  spread={r['spread_mean']:.2f}km  "
              f"t={elapsed:.1f}s")

    return results


def print_ode_sweep(results: Dict):
    print(f"\n  {'='*88}")
    print(f"  ODE STEPS (N) SWEEP -- ADE/ATE/CTE + ensemble spread vs compute")
    print(f"  {'-'*88}")
    print(f"  {'N':>4} {'ADE(km)':>9} {'ATE(km)':>9} {'CTE(km)':>9} "
          f"{'Spread(km)':>11} {'dADE vs N=1':>12} {'Time(s)':>9}")
    print(f"  {'-'*88}")
    ref_ade = results.get(1, {}).get("ADE_mean", float("nan"))
    for n, r in sorted(results.items()):
        delta = r["ADE_mean"] - ref_ade
        print(f"  {n:>4} {r['ADE_mean']:>9.2f} {r['ATE_mean']:>9.2f} "
              f"{r['CTE_mean']:>9.2f} {r['spread_mean']:>11.2f} "
              f"{delta:>+12.2f} {r['time_s']:>9.1f}")
    print(f"  {'='*88}")
    print(f"  Known project result (empirical, N=1->10): spread 4km->55km, "
          f"CRPS -9.5%, ADE/ATE/CTE +1.7%/+1.9%/+7.2%. N=3-4 was WORSE than")
    print(f"  both N=1 and N=8-10 (non-monotonic -- Euler discretization "
          f"error at low N). Use this sweep to confirm/update that curve on")
    print(f"  the current 3-seed checkpoints, and check whether N=12/16/20")
    print(f"  keep improving spread or plateau/regress past N=10.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Summarize ablation results from JSON files
# ─────────────────────────────────────────────────────────────────────────────

def summarize_ablations(ablation_dir: str) -> Dict:
    """Load all ablation JSON files and print comparison table."""
    jsons = [f for f in os.listdir(ablation_dir) if f.endswith(".json")]
    if not jsons:
        print(f"  No JSON files found in {ablation_dir}")
        return {}

    all_results = {}
    for jf in sorted(jsons):
        path = os.path.join(ablation_dir, jf)
        try:
            with open(path) as f:
                d = json.load(f)
            name = d.get("variant", jf.replace(".json", ""))
            all_results[name] = d
        except Exception as e:
            print(f"  Error loading {jf}: {e}")

    print(f"\n  {'='*80}")
    print(f"  ABLATION STUDY SUMMARY")
    print(f"  {'─'*80}")
    print(f"  {'Variant':<30} {'ADE':>8} {'ATE':>8} {'CTE':>8}  Description")
    print(f"  {'─'*80}")

    full_ade = all_results.get("full", {}).get("ADE", float("nan"))
    for name, r in sorted(all_results.items()):
        ade = r.get("ADE", float("nan"))
        ate = r.get("ATE", float("nan"))
        cte = r.get("CTE", float("nan"))
        delta = ade - full_ade if not (isinstance(ade, float) and isinstance(full_ade, float) and (ade != ade or full_ade != full_ade)) else float("nan")
        delta_str = f"({delta:+.1f})" if abs(delta) < 999 else ""
        desc = ABLATION_VARIANTS.get(name, {}).get("desc", "")[:35]
        print(f"  {name:<30} {ade:>8.1f} {ate:>8.1f} {cte:>8.1f}  {delta_str} {desc}")

    print(f"  {'='*80}\n")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-seed runner
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_seed(checkpoint_pattern: str,
                   seeds: List[int],
                   loader, device,
                   n_ensemble: int = 20) -> Dict:
    """
    Evaluate model across multiple seeds and aggregate mean ± std.
    checkpoint_pattern: e.g. "runs/seed{seed}/best_model.pth"
    """
    seed_results = []
    for seed in seeds:
        ck_path = checkpoint_pattern.format(seed=seed)
        if not os.path.exists(ck_path):
            print(f"  Missing checkpoint: {ck_path}")
            continue
        print(f"  Seed {seed}: {ck_path}")
        ck = torch.load(ck_path, map_location="cpu")
        cfg = ck.get("model_cfg", {})
        model = TCFlowMatching(**cfg).to(device)
        model.load_state_dict(ck.get("model", ck), strict=False)
        model.eval()

        all_ade, all_ate, all_cte = [], [], []
        with torch.no_grad():
            for batch in loader:
                bl = [x.to(device) if torch.is_tensor(x) else x for x in batch]
                gt = bl[1]
                try:
                    pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
                except Exception:
                    continue
                T   = min(pred.shape[0], gt.shape[0])
                pd  = _norm_to_deg(pred[:T])
                gd  = _norm_to_deg(gt[:T, :, :2])
                d   = _haversine_deg(pd, gd)
                ate, cte = _ate_cte_full_local(pd, gd)
                all_ade.extend(d.mean(0).tolist())
                all_ate.extend(ate.abs().mean(0).tolist())
                all_cte.extend(cte.abs().mean(0).tolist())

        seed_results.append({
            "seed": seed,
            "ADE": float(np.mean(all_ade)),
            "ATE": float(np.mean(all_ate)),
            "CTE": float(np.mean(all_cte)),
        })
        print(f"    ADE={seed_results[-1]['ADE']:.2f}  "
              f"ATE={seed_results[-1]['ATE']:.2f}  "
              f"CTE={seed_results[-1]['CTE']:.2f}")

    if not seed_results:
        return {}

    aggregated = {}
    for metric in ["ADE", "ATE", "CTE"]:
        vals = [r[metric] for r in seed_results]
        aggregated[metric]          = float(np.mean(vals))
        aggregated[f"{metric}_std"] = float(np.std(vals))
        aggregated[f"{metric}_vals"] = vals

    print(f"\n  Multi-seed ({len(seed_results)} seeds):")
    for metric in ["ADE", "ATE", "CTE"]:
        print(f"    {metric}: {aggregated[metric]:.2f} ± {aggregated[f'{metric}_std']:.2f}")
    return {"per_seed": seed_results, "aggregated": aggregated}


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


@torch.no_grad()
def quick_eval(model, loader, device, n_ensemble=20) -> Dict:
    """
    Quick evaluation: ADE/ATE/CTE, tổng thể + per-lead-time.

    [BỔ SUNG, per-lead-time] Giữ nguyên "ADE"/"ATE"/"CTE" tổng thể
    (tương thích ngược với mọi nơi đang đọc quick_eval() cho 21 variant
    trong ABLATION_VARIANTS), thêm "by_lead_time" cùng schema với
    ode_steps_sweep()'s bổ sung -- convention 1-indexed (1=6h...T=72h),
    khớp evaluate_multi_model.py.
    """
    model.eval()
    all_ade, all_ate, all_cte = [], [], []
    by_lt_ade = defaultdict(list)
    by_lt_ate = defaultdict(list)
    by_lt_cte = defaultdict(list)

    for batch in loader:
        bl = move(list(batch), device)
        gt = bl[1]
        try:
            pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
        except Exception:
            continue
        T   = min(pred.shape[0], gt.shape[0])
        pd  = _norm_to_deg(pred[:T])
        gd  = _norm_to_deg(gt[:T, :, :2])
        d   = _haversine_deg(pd, gd)          # [T, B]

        from Model.flow_matching_model import _forward_azimuth
        if T >= 2:
            bear_ref = _forward_azimuth(gd[:T-1], gd[1:T])
            bear_err = _forward_azimuth(gd[1:T], pd[1:T])
            dist_err = _haversine_deg(pd[1:T], gd[1:T])
            ang  = bear_err - bear_ref
            ate  = (dist_err * torch.cos(ang)).abs()   # [T-1, B]
            cte  = (dist_err * torch.sin(ang)).abs()   # [T-1, B]
            all_ate.extend(ate.mean(0).tolist())
            all_cte.extend(cte.mean(0).tolist())
            for k in range(T - 1):   # k+2 = lead_time (see ode_steps_sweep's comment)
                by_lt_ate[k + 2].extend(ate[k].tolist())
                by_lt_cte[k + 2].extend(cte[k].tolist())

        all_ade.extend(d.mean(0).tolist())
        for step0 in range(T):
            by_lt_ade[step0 + 1].extend(d[step0].tolist())

    by_lead_time = {}
    all_lts = sorted(set(by_lt_ade.keys()) | set(by_lt_ate.keys()) | set(by_lt_cte.keys()))
    for lt in all_lts:
        by_lead_time[lt] = {
            "ADE": float(np.mean(by_lt_ade[lt])) if by_lt_ade.get(lt) else float("nan"),
            "ATE": float(np.mean(by_lt_ate[lt])) if by_lt_ate.get(lt) else float("nan"),
            "CTE": float(np.mean(by_lt_cte[lt])) if by_lt_cte.get(lt) else float("nan"),
            "n":   len(by_lt_ade.get(lt, [])),
        }

    return {
        "ADE": float(np.mean(all_ade)) if all_ade else float("nan"),
        "ATE": float(np.mean(all_ate)) if all_ate else float("nan"),
        "CTE": float(np.mean(all_cte)) if all_cte else float("nan"),
        "n":   len(all_ade),
        "by_lead_time": by_lead_time,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ode_steps", "summarize", "multi_seed",
                                       "eval_variant"],
                   default="ode_steps")
    p.add_argument("--checkpoint",        type=str, default=None)
    p.add_argument("--checkpoint_pattern",type=str, default=None,
                   help="For multi_seed: runs/seed{seed}/best_model.pth")
    p.add_argument("--seeds",             type=int, nargs="+", default=[0,1,2])
    p.add_argument("--dataset_root",      type=str, default=None)
    p.add_argument("--split",             default="test")
    p.add_argument("--n_ensemble",        type=int, default=20)
    p.add_argument("--variant",           type=str, default="full",
                   choices=list(ABLATION_VARIANTS.keys()))
    p.add_argument("--output_dir",        type=str, default="ablations")
    p.add_argument("--ablation_dir",      type=str, default="ablations")
    p.add_argument("--gpu",               type=int, default=0)
    p.add_argument("--ode_steps_list",    type=int, nargs="+",
                   default=[1, 4, 8, 10, 12, 16, 20],
                   help="N values to sweep for --mode ode_steps. Default "
                        "matches the requested ablation N=[1,4,8,10,12,16,20]. "
                        "These are ODE integration steps, NOT related to "
                        "pred_len.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ── Summarize mode ────────────────────────────────────────────────────
    if args.mode == "summarize":
        summarize_ablations(args.ablation_dir)
        return

    # ── Need data and checkpoint for other modes ──────────────────────────
    if args.dataset_root is None:
        print("  ERROR: --dataset_root required"); return

    import argparse as _ap
    _, loader = data_loader(
        _ap.Namespace(dataset_root=args.dataset_root),
        {"root": args.dataset_root, "type": args.split}, test=True)
    print(f"  Data: {len(loader)} batches ({args.split})")

    # ── Multi-seed mode ───────────────────────────────────────────────────
    if args.mode == "multi_seed":
        if args.checkpoint_pattern is None:
            print("  ERROR: --checkpoint_pattern required"); return
        result = run_multi_seed(args.checkpoint_pattern, args.seeds,
                                 loader, device, args.n_ensemble)
        out_path = os.path.join(args.output_dir, "multi_seed_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {out_path}")
        return

    # ── Need checkpoint for remaining modes ───────────────────────────────
    if args.checkpoint is None:
        print("  ERROR: --checkpoint required"); return
    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = ck.get("model_cfg", {})

    # ── ODE steps mode (inference only) ──────────────────────────────────
    if args.mode == "ode_steps":
        model = TCFlowMatching(**cfg).to(device)
        model.load_state_dict(ck.get("model", ck), strict=False)
        steps_list = args.ode_steps_list
        results = ode_steps_sweep(model, loader, device,
                                   steps_list=steps_list,
                                   n_ensemble=args.n_ensemble)
        print_ode_sweep(results)
        out_path = os.path.join(args.output_dir, "ode_steps_sweep.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {out_path}")
        return

    # ── Variant evaluation mode ───────────────────────────────────────────
    if args.mode == "eval_variant":
        variant_cfg = ABLATION_VARIANTS[args.variant]
        print(f"  Variant: {args.variant}")
        print(f"  Desc: {variant_cfg['desc']}")

        # Create ablation model
        model = AblationModel(variant_cfg, **cfg).to(device)
        model.load_state_dict(ck.get("model", ck), strict=False)

        r = quick_eval(model, loader, device, args.n_ensemble)
        r["variant"] = args.variant
        r["desc"]    = variant_cfg["desc"]

        print(f"  ADE={r['ADE']:.2f}  ATE={r['ATE']:.2f}  CTE={r['CTE']:.2f}")

        out_path = os.path.join(args.output_dir, f"{args.variant}.json")
        with open(out_path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
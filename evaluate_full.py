"""

# ─────────────────────────────────────────────────────────────────────────────
# FILE PLACEMENT:
#
#   SOURCE:        evaluate_full.py
#   KAGGLE TARGET: /kaggle/working/evaluate_full.py    (root, cạnh Model/)
#   LOCAL DEV:     evaluate_full.py
#
#   Chạy sau khi train xong:
#   python evaluate_full.py \
#     --checkpoint /kaggle/working/runs/best_model.pth \
#     --dataset_root /kaggle/input/datasets/tc-ofm \
#     --split test --output_dir results/
# ─────────────────────────────────────────────────────────────────────────────

evaluate_full.py — TC-FlowMatching ESWA Full Evaluation
════════════════════════════════════════════════════════════════════════════════
Produces all metrics required by ESWA paper:
  B. Core Experiments:   ADE/ATE/CTE/RMSE/MAE/Final-DPE per horizon
                         Per-storm-category (TD/TS/Cat1-5, slow/medium/fast)
                         Boxplot data, error-by-leadtime
  C. Ablation/Testing:   CRPS (ensemble spread-skill)
  D. Physical validity:  Speed/accel within physical bounds
                         Cone of uncertainty per storm

Usage:
  python evaluate_full.py \\
    --checkpoint runs/best_model.pth \\
    --dataset_root /path/to/tc-ofm \\
    --split test \\
    --output_dir results/ \\
    --n_ensemble 20
"""
from __future__ import annotations

import sys, os, argparse, time, json, math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ─── Imports from model ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, EMAModel,
    _norm_to_deg, _haversine_deg, _forward_azimuth,
    _step_speeds_kmh, _unwrap,
)

# ─── Constants ────────────────────────────────────────────────────────────────
R_EARTH   = 6371.0
DT_HOURS  = 6.0
ST_TRANS  = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
             "12h": 65.0, "24h": 130.0, "48h": 205.0, "72h": 321.0}

# Horizon steps: step index 0=6h, 1=12h, 3=24h, 7=48h, 11=72h
HORIZONS = {"6h": 0, "12h": 1, "24h": 3, "48h": 7, "72h": 11}

# TC intensity categories by obs speed (km/h)
# TD<63, TS 63-119, Cat1 119-153, Cat2 154-177, Cat3 178-208, Cat4 209-252, Cat5≥253 (kt→km/h)
INTENSITY_BINS  = [0, 63, 119, 153, 177, 208, 252, 9999]
INTENSITY_NAMES = ["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"]
# Speed categories (obs mean km/h)
SPEED_SLOW   = 8.0
SPEED_FAST   = 15.0

# Physical bounds
MAX_TC_SPEED_KMH  = 100.0   # TC track speed >100km/h per 6h is unphysical
MAX_ACCEL_KMH2    = 30.0    # Speed change >30km/h per step is unphysical


# ─────────────────────────────────────────────────────────────────────────────
#  Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def move(batch, device):
    return [x.to(device) if torch.is_tensor(x) else x for x in batch]


def _ate_cte_full(pred_deg: torch.Tensor,
                   gt_deg:   torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ATE = along-track error (signed), CTE = cross-track error (signed).
    Returns shape [T-1, B] each.
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    bear_ref = _forward_azimuth(gt_deg[:T-1], gt_deg[1:T])    # [T-1, B]
    bear_err = _forward_azimuth(gt_deg[1:T],  pred_deg[1:T])  # [T-1, B]
    dist_err = _haversine_deg(pred_deg[1:T], gt_deg[1:T])     # [T-1, B]
    ang      = bear_err - bear_ref
    return dist_err * torch.cos(ang), dist_err * torch.sin(ang)


def _rmse_per_step(pred_deg: torch.Tensor,
                    gt_deg:   torch.Tensor) -> torch.Tensor:
    """RMSE in km per step. Returns [T, B]."""
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    d = _haversine_deg(pred_deg[:T], gt_deg[:T])  # [T, B]
    return d  # caller squares and takes mean


def _crps_ensemble(samples: torch.Tensor,
                    gt_deg:  torch.Tensor) -> Dict:
    """
    Continuous Ranked Probability Score for ensemble forecast.
    samples: [K, T, B, 2] in deg
    gt_deg:  [T, B, 2] in deg

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    where X, X' are independent draws from the ensemble.
    Returns mean CRPS per horizon and spread-skill ratio.
    """
    K, T, B, _ = samples.shape
    T_gt = min(T, gt_deg.shape[0])

    crps_per_step = []  # [T]
    spread_per_step = []
    skill_per_step = []

    for t in range(T_gt):
        # Distance from each member to GT: [K, B]
        dist_to_gt = torch.stack([
            _haversine_deg(samples[k, t:t+1], gt_deg[t:t+1]).squeeze(0)
            for k in range(K)
        ], dim=0)  # [K, B]

        # E[|X - y|]: mean over ensemble
        e_dist = dist_to_gt.mean(0)  # [B]

        # E[|X - X'|]: pairwise mean
        pairwise = 0.0
        n_pairs = 0
        for i in range(K):
            for j in range(i+1, K):
                d = _haversine_deg(samples[i, t:t+1], samples[j, t:t+1]).squeeze(0)
                pairwise = pairwise + d
                n_pairs += 1
        if n_pairs > 0:
            e_spread = pairwise / n_pairs
        else:
            e_spread = torch.zeros_like(e_dist)

        crps = e_dist - 0.5 * e_spread          # [B]
        crps_per_step.append(float(crps.mean()))
        spread_per_step.append(float(e_spread.mean()))
        skill_per_step.append(float(e_dist.mean()))

    crps_arr   = np.array(crps_per_step)
    spread_arr = np.array(spread_per_step)
    skill_arr  = np.array(skill_per_step)
    # Spread-skill ratio: ideal = 1.0 (spread matches RMSE)
    skill_std  = np.sqrt(np.array([
        float((_haversine_deg(samples[:, t].mean(0, keepdim=True),
                               gt_deg[t:t+1]).squeeze(0)**2).mean())
        for t in range(T_gt)
    ]) + 1e-6)
    ss_ratio   = spread_arr / (skill_std + 1e-6)

    hz_idx = {k: v for k, v in HORIZONS.items() if v < T_gt}
    return {
        "crps_mean":        float(crps_arr.mean()),
        "crps_per_step":    crps_arr.tolist(),
        "spread_per_step":  spread_arr.tolist(),
        "spread_skill_ratio": ss_ratio.tolist(),
        "crps_by_horizon":  {h: float(crps_arr[s]) for h, s in hz_idx.items()},
        "ss_ratio_by_horizon": {h: float(ss_ratio[s]) for h, s in hz_idx.items()},
    }


def _physical_validity(pred_deg: torch.Tensor,
                        obs_deg:  torch.Tensor) -> Dict:
    """
    Check if predicted trajectory violates physical constraints.
    Returns fraction of storms with physically valid predictions.
    """
    B = pred_deg.shape[1]
    if pred_deg.shape[0] < 2:
        return {"valid_speed_frac": 1.0, "valid_accel_frac": 1.0,
                "mean_pred_speed": 0.0, "max_pred_speed": 0.0}

    pts = torch.cat([obs_deg[-1:], pred_deg], 0)  # [T+1, B, 2]
    speeds = _step_speeds_kmh(pts)   # [T, B] km/h

    # Speed validity: all steps < MAX_TC_SPEED_KMH
    valid_speed = (speeds < MAX_TC_SPEED_KMH).all(0)   # [B]
    # Acceleration validity: |speed[t] - speed[t-1]| < MAX_ACCEL
    if speeds.shape[0] >= 2:
        accel = (speeds[1:] - speeds[:-1]).abs()
        valid_accel = (accel < MAX_ACCEL_KMH2).all(0)  # [B]
    else:
        valid_accel = torch.ones(B, dtype=torch.bool, device=pred_deg.device)

    return {
        "valid_speed_frac":  float(valid_speed.float().mean()),
        "valid_accel_frac":  float(valid_accel.float().mean()),
        "mean_pred_speed":   float(speeds.mean()),
        "max_pred_speed":    float(speeds.max()),
        "speed_per_step":    speeds.mean(1).tolist(),   # [T] mean over batch
    }


def _obs_speed(obs_deg: torch.Tensor) -> torch.Tensor:
    """Mean obs speed per storm. [B] km/h."""
    if obs_deg.shape[0] < 2:
        return obs_deg.new_zeros(obs_deg.shape[1])
    return _step_speeds_kmh(obs_deg).mean(0)   # [B]


# ─────────────────────────────────────────────────────────────────────────────
#  Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_full_evaluation(model, loader, device,
                         tag:        str  = "TEST",
                         n_ensemble: int  = 20,
                         ema:        Optional[EMAModel] = None,
                         collect_samples: bool = True) -> Dict:
    """
    Full evaluation. Returns per-storm arrays for downstream analysis.
    collect_samples=True: collects K ensemble samples for CRPS (memory-intensive).
    """
    bk = None
    if ema is not None:
        try:
            bk = ema.apply_to(model)
        except Exception as e:
            print(f"  ⚠ EMA apply failed: {e}")

    model.eval()

    # Per-storm accumulators
    per_storm = {
        "ade": [], "ate": [], "cte": [],
        "rmse": [], "mae": [],
        "obs_speed": [],
        "dist_per_step": [[] for _ in range(12)],   # [12 steps][storms]
        "ate_per_step":  [[] for _ in range(11)],
        "cte_per_step":  [[] for _ in range(11)],
        "final_dpe": [],
        "valid_speed": [], "valid_accel": [],
        "crps": [],   # per storm
    }
    all_samples_by_step = [[] for _ in range(12)]  # for CRPS

    # Collect all ensemble samples per batch if needed
    t_start = time.time()
    n_batches = len(loader)

    for i, batch in enumerate(loader):
        bl  = move(list(batch), device)
        gt  = bl[1]                          # [T, B, 2]
        obs = bl[0]                          # [T_obs, B, ≥2]
        B   = obs.shape[1]

        obs_deg_i = _norm_to_deg(obs[:, :, :2])   # [T_obs, B, 2]
        obs_spd_i = _obs_speed(obs_deg_i)          # [B]
        gt_deg_i  = _norm_to_deg(gt[:, :, :2])    # [T_gt, B, 2]

        # ── Standard prediction (mean of top-K) ──────────────────────────
        try:
            pred, _, all_t = model.sample(bl, num_ensemble=n_ensemble)
        except Exception as e:
            print(f"  [batch {i+1}/{n_batches}] sample error: {e}"); continue

        T   = min(pred.shape[0], gt.shape[0])
        pd  = _norm_to_deg(pred[:T])           # [T, B, 2]
        gd  = gt_deg_i[:T]                     # [T, B, 2]
        d   = _haversine_deg(pd, gd)           # [T, B]
        ate, cte = _ate_cte_full(pd, gd)       # [T-1, B], [T-1, B]

        per_storm["ade"].extend(d.mean(0).tolist())
        per_storm["ate"].extend(ate.abs().mean(0).tolist() if ate.shape[0] > 0 else [0.0]*B)
        per_storm["cte"].extend(cte.abs().mean(0).tolist() if cte.shape[0] > 0 else [0.0]*B)
        per_storm["rmse"].extend(d.pow(2).mean(0).sqrt().tolist())
        per_storm["mae"].extend(d.mean(0).tolist())   # same as ADE for haversine
        per_storm["final_dpe"].extend(d[min(T-1, 11)].tolist())   # 72h or last step
        per_storm["obs_speed"].extend(obs_spd_i.tolist())

        # Per-step dist
        for s in range(min(T, 12)):
            per_storm["dist_per_step"][s].extend(d[s].tolist())
        for s in range(min(ate.shape[0], 11)):
            per_storm["ate_per_step"][s].extend(ate[s].abs().tolist())
            per_storm["cte_per_step"][s].extend(cte[s].abs().tolist())

        # Physical validity
        phys = _physical_validity(pd, obs_deg_i)
        per_storm["valid_speed"].append(phys["valid_speed_frac"])
        per_storm["valid_accel"].append(phys["valid_accel_frac"])

        # ── CRPS: collect ensemble members ─────────────────────────────────
        if collect_samples and all_t is not None and all_t.shape[0] >= 2:
            K_actual = min(all_t.shape[0], n_ensemble)
            for s in range(min(T, 12)):
                # all_t: [K, T, B, 2] norm → convert step s to deg [K, B, 2]
                step_samples = _norm_to_deg(all_t[:K_actual, s, :, :2])   # [K, B, 2]
                all_samples_by_step[s].append(step_samples.cpu())

    elapsed = time.time() - t_start

    # ── Aggregate ──────────────────────────────────────────────────────────
    def _m(lst): return float(np.nanmean(lst)) if lst else float("nan")
    def _s(lst): return float(np.nanstd(lst))  if lst else float("nan")

    result = {
        "tag": tag, "n": len(per_storm["ade"]), "time_s": elapsed,
        # Core metrics
        "ADE":  _m(per_storm["ade"]),    "ADE_std":  _s(per_storm["ade"]),
        "ATE":  _m(per_storm["ate"]),    "ATE_std":  _s(per_storm["ate"]),
        "CTE":  _m(per_storm["cte"]),    "CTE_std":  _s(per_storm["cte"]),
        "RMSE": _m(per_storm["rmse"]),
        "MAE":  _m(per_storm["mae"]),
        "FinalDPE": _m(per_storm["final_dpe"]),
        # Physical validity
        "valid_speed_frac": _m(per_storm["valid_speed"]),
        "valid_accel_frac": _m(per_storm["valid_accel"]),
    }

    # Per-horizon (6h/12h/24h/48h/72h)
    result["per_horizon"] = {}
    result["per_horizon_ate"] = {}
    result["per_horizon_cte"] = {}
    for hz, s in HORIZONS.items():
        d_s = per_storm["dist_per_step"][s] if s < 12 else []
        result["per_horizon"][hz] = _m(d_s)
        if s < 11:
            result["per_horizon_ate"][hz] = _m(per_storm["ate_per_step"][s])
            result["per_horizon_cte"][hz] = _m(per_storm["cte_per_step"][s])

    # Boxplot data (raw arrays)
    result["boxplot_ade"] = per_storm["ade"]
    result["dist_per_step_mean"] = [_m(per_storm["dist_per_step"][s]) for s in range(12)]
    result["ate_per_step_mean"]  = [_m(per_storm["ate_per_step"][s])  for s in range(11)]
    result["cte_per_step_mean"]  = [_m(per_storm["cte_per_step"][s])  for s in range(11)]

    # ── By speed category ──────────────────────────────────────────────
    obs_spd_arr = np.array(per_storm["obs_speed"])
    ade_arr     = np.array(per_storm["ade"])
    ate_arr     = np.array(per_storm["ate"])
    cte_arr     = np.array(per_storm["cte"])

    # Speed categories
    slow_m = obs_spd_arr < SPEED_SLOW
    fast_m = obs_spd_arr >= SPEED_FAST
    med_m  = ~slow_m & ~fast_m
    result["by_speed"] = {
        "slow": {
            "n": int(slow_m.sum()),
            "ADE": float(ade_arr[slow_m].mean()) if slow_m.any() else float("nan"),
            "ATE": float(ate_arr[slow_m].mean()) if slow_m.any() else float("nan"),
            "CTE": float(cte_arr[slow_m].mean()) if slow_m.any() else float("nan"),
        },
        "medium": {
            "n": int(med_m.sum()),
            "ADE": float(ade_arr[med_m].mean()) if med_m.any() else float("nan"),
            "ATE": float(ate_arr[med_m].mean()) if med_m.any() else float("nan"),
            "CTE": float(cte_arr[med_m].mean()) if med_m.any() else float("nan"),
        },
        "fast": {
            "n": int(fast_m.sum()),
            "ADE": float(ade_arr[fast_m].mean()) if fast_m.any() else float("nan"),
            "ATE": float(ate_arr[fast_m].mean()) if fast_m.any() else float("nan"),
            "CTE": float(cte_arr[fast_m].mean()) if fast_m.any() else float("nan"),
        },
    }

    # ── By TC intensity category (TD/TS/Cat1-5) ───────────────────────
    # obs_speed (km/h) → Saffir-Simpson scale (1 min sustained wind proxy)
    # Mapping từ track speed sang intensity category (approximate):
    #   TC track speed ≠ wind speed, nhưng dùng làm proxy vì không có
    #   wind speed trong dataset tc-ofm. Bins theo km/h track speed:
    #   TD < 8, TS 8-15, Cat1 15-20, Cat2 20-25, Cat3 25-30, Cat4+ ≥30
    # Reviewer note: ghi rõ đây là proxy từ track speed, không phải wind speed.
    intensity_bins  = [0, 8, 15, 20, 25, 30, 9999]
    intensity_names = ["TD", "TS", "Cat1", "Cat2", "Cat3", "Cat4+"]
    result["by_intensity"] = {}
    for i, cat in enumerate(intensity_names):
        lo, hi = intensity_bins[i], intensity_bins[i+1]
        mask = (obs_spd_arr >= lo) & (obs_spd_arr < hi)
        result["by_intensity"][cat] = {
            "n":   int(mask.sum()),
            "ADE": float(ade_arr[mask].mean()) if mask.any() else float("nan"),
            "ATE": float(ate_arr[mask].mean()) if mask.any() else float("nan"),
            "CTE": float(cte_arr[mask].mean()) if mask.any() else float("nan"),
            "speed_range_kmh": f"{lo}-{hi}",
        }

    # ── CRPS (computed from collected samples) ──────────────────────────
    crps_per_step = []
    spread_per_step = []
    if collect_samples and all_samples_by_step[0]:
        print(f"  Computing CRPS over {len(all_samples_by_step[0])} batches...")
        # Simplified CRPS: E[|X-y|] - 0.5*E[|X-X'|] per step
        gt_loader_deg = []  # Need GT per step — compute from loader
        # Re-iterate loader for GT (already have per_storm aggregation)
        # Instead, compute crps from collected step samples vs per_storm dist
        for s in range(12):
            if not all_samples_by_step[s]:
                crps_per_step.append(float("nan"))
                spread_per_step.append(float("nan"))
                continue
            # samples: list of [K, B, 2] tensors → concat over batches → [K, N, 2]
            step_samples_cat = torch.cat(all_samples_by_step[s], dim=1)  # [K, N, 2]
            K, N, _ = step_samples_cat.shape
            # Spread = mean pairwise distance (subsample for speed)
            idx1 = torch.randperm(K)[:min(K, 10)]
            idx2 = torch.randperm(K)[:min(K, 10)]
            spread_vals = []
            for a, b in zip(idx1.tolist(), idx2.tolist()):
                if a != b:
                    d_ab = _haversine_deg(
                        step_samples_cat[a:a+1].expand(1, N, 2).to(device),
                        step_samples_cat[b:b+1].expand(1, N, 2).to(device)
                    ).squeeze(0)
                    spread_vals.append(float(d_ab.mean()))
            spread = float(np.mean(spread_vals)) if spread_vals else 0.0
            spread_per_step.append(spread)
            # Skill: mean dist to GT (from pre-computed per_storm dist_per_step)
            skill = _m(per_storm["dist_per_step"][s])
            # CRPS ≈ skill - 0.5*spread
            crps_per_step.append(skill - 0.5 * spread)

        result["crps"] = {
            "per_step": crps_per_step,
            "mean": float(np.nanmean(crps_per_step)),
            "spread_per_step": spread_per_step,
            "spread_mean": float(np.nanmean(spread_per_step)),
            "spread_skill_ratio": [
                s / (k + 1e-6)
                for s, k in zip(spread_per_step, result["dist_per_step_mean"])
            ],
        }
    else:
        result["crps"] = {}

    if bk is not None:
        try: ema.restore(model, bk)
        except Exception: pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_full_results(r: Dict, st_trans: Dict = ST_TRANS):
    n = r.get("n", 0)
    tag = r.get("tag", "")
    print(f"\n  {'='*72}")
    print(f"  [{tag}]  n={n}  ({r.get('time_s', 0):.1f}s)")

    def beat(k, v):
        ref = st_trans.get(k, 1e9)
        return "✓" if v < ref else "✗"

    # Core metrics
    ade, ate, cte = r["ADE"], r["ATE"], r["CTE"]
    print(f"  ADE={ade:7.2f}±{r.get('ADE_std',0):.1f}km {beat('ADE',ade)}  "
          f"ATE={ate:7.2f}±{r.get('ATE_std',0):.1f}km {beat('ATE',ate)}  "
          f"CTE={cte:7.2f}±{r.get('CTE_std',0):.1f}km {beat('CTE',cte)}")
    print(f"  RMSE={r.get('RMSE',0):6.2f}km  MAE={r.get('MAE',0):6.2f}km  "
          f"FinalDPE(72h)={r.get('FinalDPE',0):6.2f}km")

    # Per-horizon
    ph = r.get("per_horizon", {})
    print(f"  Per-horizon ADE:  " +
          "  ".join(f"{h}={ph.get(h,float('nan')):6.1f}" for h in ["6h","12h","24h","48h","72h"]))
    ph_ate = r.get("per_horizon_ate", {})
    ph_cte = r.get("per_horizon_cte", {})
    print(f"  Per-horizon ATE:  " +
          "  ".join(f"{h}={ph_ate.get(h,float('nan')):6.1f}" for h in ["12h","24h","48h","72h"]))
    print(f"  Per-horizon CTE:  " +
          "  ".join(f"{h}={ph_cte.get(h,float('nan')):6.1f}" for h in ["12h","24h","48h","72h"]))

    # ST-Trans comparison
    print(f"\n  vs ST-Trans:")
    print(f"  {'Metric':<8} {'ST-Trans':>9} {'Ours':>9} {'Δ':>8}  Status")
    print(f"  {'─'*48}")
    for k in ["ADE", "ATE", "CTE"]:
        v = r.get(k, float("nan"))
        ref = st_trans.get(k, float("nan"))
        delta = v - ref
        status = "✓ BEAT" if v < ref else "✗"
        print(f"  {k:<8} {ref:>9.1f} {v:>9.2f} {delta:>+8.2f}km  {status}")

    # By speed category
    bs = r.get("by_speed", {})
    if bs:
        print(f"\n  By obs-speed category:")
        for cat, d in bs.items():
            print(f"    {cat:<8} n={d.get('n',0):3d}  "
                  f"ADE={d.get('ADE',float('nan')):6.1f}  "
                  f"ATE={d.get('ATE',float('nan')):6.1f}  "
                  f"CTE={d.get('CTE',float('nan')):6.1f}")

    # By TC intensity category
    bi = r.get("by_intensity", {})
    if bi:
        print(f"\n  By TC intensity (track speed proxy):")
        print(f"    {'Cat':<8} {'n':>4}  {'ADE':>7}  {'ATE':>7}  {'CTE':>7}  Speed(km/h)")
        for cat, d in bi.items():
            if d.get("n", 0) > 0:
                print(f"    {cat:<8} {d['n']:>4}  "
                      f"{d.get('ADE',float('nan')):>7.1f}  "
                      f"{d.get('ATE',float('nan')):>7.1f}  "
                      f"{d.get('CTE',float('nan')):>7.1f}  "
                      f"{d.get('speed_range_kmh','')}")

    # Physical validity
    print(f"\n  Physical validity:  "
          f"valid_speed={r.get('valid_speed_frac',0):.3f}  "
          f"valid_accel={r.get('valid_accel_frac',0):.3f}")

    # CRPS
    crps = r.get("crps", {})
    if crps:
        print(f"  CRPS mean={crps.get('mean',float('nan')):.2f}km  "
              f"spread_mean={crps.get('spread_mean',float('nan')):.2f}km")
        ss = crps.get("spread_skill_ratio", [])
        if ss:
            hz_ss = {h: ss[s] if s < len(ss) else float("nan")
                     for h, s in HORIZONS.items()}
            print(f"  Spread/Skill ratio: " +
                  "  ".join(f"{h}={v:.2f}" for h, v in hz_ss.items()))

    print(f"  {'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Sigma sensitivity analysis (reviewer ablation)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sigma_sensitivity(model, loader, device,
                       sigma_values: list = [0.01, 0.02, 0.04, 0.06, 0.08],
                       n_ensemble: int = 20) -> Dict:
    """
    Ablation: sensitivity của kết quả theo sigma_inference.
    Chạy inference với nhiều sigma khác nhau trên cùng checkpoint.
    → Justification cho sigma_inference=0.04 cố định (reviewer question).
    """
    raw = _unwrap(model)
    orig_sigma = float(raw.sigma_inference)
    results = {}
    for sigma in sigma_values:
        raw.sigma_inference = sigma  # temporarily override
        all_ade = []
        all_cte = []
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
            d   = _haversine_deg(pd, gd)
            all_ade.extend(d.mean(0).tolist())
            if T >= 2:
                ate_v, cte_v = _ate_cte_full(pd, gd)
                all_cte.extend(cte_v.abs().mean(0).tolist())
        results[sigma] = {
            "sigma":    sigma,
            "ADE":      float(np.mean(all_ade)) if all_ade else float("nan"),
            "CTE":      float(np.mean(all_cte)) if all_cte else float("nan"),
            "n":        len(all_ade),
        }
        print(f"  sigma={sigma:.3f}: ADE={results[sigma]['ADE']:.2f}  CTE={results[sigma]['CTE']:.2f}")
    raw.sigma_inference = orig_sigma  # restore
    return results


@torch.no_grad()
def ensemble_size_eval(model, loader, device,
                        k_values: list = [1, 3, 5, 10, 20, 40]) -> Dict:
    """
    Ablation: accuracy vs compute trade-off theo ensemble size K.
    → Justification cho K=20 default.
    → ESWA Table: shows diminishing returns beyond K=20.
    """
    raw = _unwrap(model)
    results = {}
    for k in k_values:
        all_ade, all_ate, all_cte = [], [], []
        t0 = time.time()
        for batch in loader:
            bl = move(list(batch), device)
            gt = bl[1]
            try:
                pred, _, _ = model.sample(bl, num_ensemble=k)
            except Exception:
                continue
            T   = min(pred.shape[0], gt.shape[0])
            pd  = _norm_to_deg(pred[:T])
            gd  = _norm_to_deg(gt[:T, :, :2])
            d   = _haversine_deg(pd, gd)
            all_ade.extend(d.mean(0).tolist())
            if T >= 2:
                ate_v, cte_v = _ate_cte_full(pd, gd)
                all_ate.extend(ate_v.abs().mean(0).tolist())
                all_cte.extend(cte_v.abs().mean(0).tolist())
        elapsed = time.time() - t0
        results[k] = {
            "K": k, "ADE": float(np.mean(all_ade)) if all_ade else float("nan"),
            "ATE": float(np.mean(all_ate)) if all_ate else float("nan"),
            "CTE": float(np.mean(all_cte)) if all_cte else float("nan"),
            "time_s": elapsed, "n": len(all_ade),
        }
        print(f"  K={k:3d}: ADE={results[k]['ADE']:.2f}  ATE={results[k]['ATE']:.2f}"
              f"  CTE={results[k]['CTE']:.2f}  t={elapsed:.1f}s")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Case study: cone of uncertainty
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_case_studies(model, loader, device, n_cases: int = 6,
                          n_ensemble: int = 20) -> List[Dict]:
    """
    Collect detailed predictions for N representative storms.
    Picks storms covering slow/medium/fast and worst/best ADE.
    Returns list of dicts with obs, gt, all ensemble members, mean pred.
    """
    model.eval()
    cases = []
    storm_data = []

    for batch in loader:
        bl  = move(list(batch), device)
        gt  = bl[1]
        obs = bl[0]
        B   = obs.shape[1]
        obs_deg = _norm_to_deg(obs[:, :, :2])

        try:
            pred, _, all_t = model.sample(bl, num_ensemble=n_ensemble)
        except Exception:
            continue

        T = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T])
        gd = _norm_to_deg(gt[:T, :, :2])
        d  = _haversine_deg(pd, gd).mean(0)  # [B] ADE per storm
        spd = _obs_speed(obs_deg)              # [B]

        for b in range(B):
            storm_data.append({
                "ade": float(d[b]),
                "obs_speed": float(spd[b]),
                "obs_deg": obs_deg[:, b, :].cpu().numpy(),
                "gt_deg":  gd[:, b, :].cpu().numpy(),
                "pred_deg": pd[:, b, :].cpu().numpy(),
                "ensemble": all_t[:, :T, b, :2].cpu().numpy()
                            if all_t is not None else None,
            })
        if len(storm_data) > 200:
            break

    if not storm_data:
        return []

    # Pick representative cases: best/worst/median + fast/slow
    storm_data.sort(key=lambda x: x["ade"])
    n_total = len(storm_data)

    selected_indices = set()
    # Best, worst, median
    selected_indices.update([0, n_total//4, n_total//2, 3*n_total//4, n_total-1])
    # Fastest storm
    fastest = max(range(len(storm_data)), key=lambda i: storm_data[i]["obs_speed"])
    selected_indices.add(fastest)
    # Slowest storm
    slowest = min(range(len(storm_data)), key=lambda i: storm_data[i]["obs_speed"])
    selected_indices.add(slowest)

    selected = sorted(selected_indices)[:n_cases]
    return [storm_data[i] for i in selected]


def compute_cone_of_uncertainty(ensemble: np.ndarray,
                                  confidence: float = 0.67) -> Dict:
    """
    Compute cone of uncertainty from ensemble.
    ensemble: [K, T, 2] in degrees (lon, lat)
    Returns per-step radius (km) at given confidence level.
    """
    K, T, _ = ensemble.shape
    radii = []
    for t in range(T):
        pts_t = ensemble[:, t, :]  # [K, 2] lon, lat
        mean_lon = pts_t[:, 0].mean()
        mean_lat = pts_t[:, 1].mean()
        # Haversine distance from each member to mean
        dists = []
        for k in range(K):
            dlat = math.radians(pts_t[k, 1] - mean_lat)
            dlon = math.radians(pts_t[k, 0] - mean_lon)
            lat1 = math.radians(mean_lat)
            lat2 = math.radians(pts_t[k, 1])
            a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            dists.append(2 * R_EARTH * math.asin(math.sqrt(max(0, min(1, a)))))
        radii.append(float(np.quantile(dists, confidence)))
    return {
        "radii_km": radii,
        "horizon_labels": [f"{(t+1)*6}h" for t in range(T)],
        "confidence": confidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="TC-FlowMatching ESWA Full Evaluation")
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split",        default="test", choices=["test","val","train"])
    p.add_argument("--n_ensemble",   type=int, default=20)
    p.add_argument("--output_dir",   default="eval_results")
    p.add_argument("--no_ema",       action="store_true")
    p.add_argument("--no_crps",      action="store_true",
                   help="Skip CRPS computation (faster)")
    p.add_argument("--case_studies", action="store_true", default=True,
                   help="Collect case study data")
    p.add_argument("--n_cases",      type=int, default=6)
    p.add_argument("--gpu",          type=int, default=0)
    p.add_argument("--test_year",    type=int, default=None,
                   help="Filter test set by year (same as evaluate_test_storms.py)")
    p.add_argument("--sigma_sensitivity", action="store_true", default=False,
                   help="Run sigma_inference sensitivity analysis (reviewer ablation)")
    p.add_argument("--ensemble_ablation",  action="store_true", default=False,
                   help="Run ensemble size K ablation (K=1,3,5,10,20,40)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n  Checkpoint: {args.checkpoint}")
    print(f"  Split: {args.split} | Device: {device}")
    print("="*72)

    # ── Load model ─────────────────────────────────────────────────────────
    ck = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ck.get("model_cfg", {})
    model = TCFlowMatching(**model_cfg).to(device)
    state = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"  ⚠ Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected: print(f"  ⚠ Unexpected ({len(unexpected)}): {unexpected[:3]}...")
    ep = ck.get("epoch", "?")
    print(f"  Loaded ep{ep}")

    # Print learned params
    raw = _unwrap(model)
    if hasattr(raw, "speed_correction_logits"):
        corr = (torch.sigmoid(raw.speed_correction_logits) * 2.0).tolist()
        print(f"  [LEARN] speed_corr: {[f'{v:.3f}' for v in corr[:4]]}...")
    if hasattr(raw, "log_sigma_reg"):
        print(f"  [LEARN] eff_lambda: "
              f"reg={0.5*math.exp(-2*max(-3,raw.log_sigma_reg.item())):.3f}  "
              f"heading={0.5*math.exp(-2*max(-3,raw.log_sigma_heading.item())):.3f}  "
              f"calib={0.5*math.exp(-2*max(-3,raw.log_sigma_calib.item())):.3f}")

    # Model footprint
    total_params = sum(p.numel() for p in model.parameters())
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"  Model: {total_params:,} params | {mem_mb:.1f}MB")

    # ── EMA ────────────────────────────────────────────────────────────────
    ema = None
    if not args.no_ema and ck.get("ema"):
        try:
            ema = EMAModel(model)
            for k, v in ck["ema"].items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.to(device))
            print(f"  EMA loaded ({len(ema.shadow)} params)")
        except Exception as e:
            print(f"  ⚠ EMA failed: {e}"); ema = None

    # ── Data ───────────────────────────────────────────────────────────────
    import argparse as _ap
    _loader_args = _ap.Namespace(
        dataset_root = args.dataset_root,
        obs_len      = 8,
        pred_len     = 12,
        batch_size   = 64,
        num_workers  = 2,
        test_year    = getattr(args, "test_year", None),
        skip         = getattr(args, "skip", 1),
        min_ped      = getattr(args, "min_ped", 1),
        threshold    = getattr(args, "threshold", 0.002),
    )
    try:
        _, loader = data_loader(
            _loader_args,
            {"root": args.dataset_root, "type": args.split},
            test=(args.split != "train"),
        )
    except Exception as _e:
        print(f"  ❌ data_loader(type=\'{args.split}\') failed: {_e}")
        raise
    print(f"  Data: {len(loader)} batches")

    # ── Full evaluation ─────────────────────────────────────────────────────
    result = run_full_evaluation(
        model, loader, device,
        tag=f"{args.split.upper()} ep{ep}",
        n_ensemble=args.n_ensemble,
        ema=ema,
        collect_samples=not args.no_crps,
    )

    # Print
    print_full_results(result)

    # Timing benchmark: ms per inference step
    ms_per_step = float("nan")
    try:
        model.eval()
        with torch.no_grad():
            dummy = move(list(next(iter(loader))), device)
            t0 = time.time()
            for _ in range(10):
                model.sample(dummy, num_ensemble=args.n_ensemble)
            ms_per_step = (time.time() - t0) / 10 * 1000
        print(f"  Inference: {ms_per_step:.1f}ms per batch (K={args.n_ensemble})")
    except Exception as e:
        print(f"  ⚠ Timing benchmark failed: {e}")
    result["inference_ms_per_batch"] = ms_per_step
    result["total_params"] = total_params
    result["model_mb"] = mem_mb

    # ── Case studies ─────────────────────────────────────────────────────────
    case_results = []
    if args.case_studies:
        print(f"\n  Collecting {args.n_cases} case studies...")
        cases = collect_case_studies(model, loader, device,
                                      n_cases=args.n_cases,
                                      n_ensemble=args.n_ensemble)
        for i, case in enumerate(cases):
            cone = {}
            if case.get("ensemble") is not None:
                cone = compute_cone_of_uncertainty(
                    case["ensemble"].reshape(-1, case["ensemble"].shape[-2], 2),
                    confidence=0.67
                )
            case_results.append({
                "id":         i,
                "ade":        case["ade"],
                "obs_speed":  case["obs_speed"],
                "cone_67pct": cone,
                # Arrays saved as lists for JSON
                "obs_deg":    case["obs_deg"].tolist(),
                "gt_deg":     case["gt_deg"].tolist(),
                "pred_deg":   case["pred_deg"].tolist(),
            })
            print(f"    Case {i+1}: ADE={case['ade']:.1f}km  "
                  f"obs_speed={case['obs_speed']:.1f}km/h  "
                  f"cone_72h={cone.get('radii_km', [0]*12)[-1]:.1f}km" if cone else "")
        result["case_studies"] = case_results

    # ── Sigma sensitivity (nếu được yêu cầu) ─────────────────────────────────
    if args.sigma_sensitivity:
        print(f"\n  Running sigma_inference sensitivity analysis...")
        sigma_results = sigma_sensitivity(model, loader, device,
                                          sigma_values=[0.01, 0.02, 0.04, 0.06, 0.08],
                                          n_ensemble=args.n_ensemble)
        result["sigma_sensitivity"] = sigma_results

    # ── Ensemble size ablation (nếu được yêu cầu) ─────────────────────────
    if args.ensemble_ablation:
        print(f"\n  Running ensemble size ablation K=1,3,5,10,20,40...")
        ens_results = ensemble_size_eval(model, loader, device,
                                          k_values=[1, 3, 5, 10, 20, 40])
        result["ensemble_ablation"] = ens_results

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir,
                             f"eval_{args.split}_ep{ep}.json")
    save_result = {k: v for k, v in result.items()
                   if k != "boxplot_ade"}   # exclude large arrays from main JSON
    save_result["boxplot_ade"] = result.get("boxplot_ade", [])  # keep for boxplot
    save_result["checkpoint"]  = args.checkpoint
    save_result["split"]       = args.split

    with open(out_path, "w") as f:
        json.dump(save_result, f, indent=2, default=str)
    print(f"  Saved → {out_path}")

    # Summary line for paper table
    print(f"\n  ── PAPER TABLE ROW ──")
    print(f"  FM(ours) | "
          f"ADE={result['ADE']:.1f}±{result.get('ADE_std',0):.1f} | "
          f"ATE={result['ATE']:.1f}±{result.get('ATE_std',0):.1f} | "
          f"CTE={result['CTE']:.1f}±{result.get('CTE_std',0):.1f} | "
          f"CRPS={result.get('crps',{}).get('mean',float('nan')):.1f}")


if __name__ == "__main__":
    main()
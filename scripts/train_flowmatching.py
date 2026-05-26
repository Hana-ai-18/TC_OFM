"""
scripts/train_v65.py — TC-FlowMatching v65
════════════════════════════════════════════════════════════════════════════
Phased training cho TCFlowMatchingV65.

PHASES (giống v64 logic):
  Phase 1  ep 0  .. GEN_ONLY_EPOCHS-1  : Generator only, selector FROZEN
  Phase 2  ep GEN_ONLY_EPOCHS .. SEL_WARM_EPOCHS-1 : Selector warmup
  Phase 3  ep SEL_WARM_EPOCHS+          : Full joint

HOW TO RUN:
  python scripts/train_v65.py \
      --dataset_root TCND_vn \
      --output_dir   runs/v65 \
      --batch_size   32 \
      --num_epochs   120 \
      --use_amp \
      --gpu_num 0

RESUME:
  python scripts/train_v65.py \
      --resume runs/v65/best_composite.pth \
      --output_dir runs/v65 ...
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, csv, time, random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader

# Import v65 model — alias đầy đủ
from Model.flow_matching_model_v65 import TCFlowMatchingV65 as TCFlowMatchingV65

try:
    from Model.utils import get_cosine_schedule_with_warmup
except ImportError:
    # Fallback scheduler nếu không có utils
    from torch.optim.lr_scheduler import CosineAnnealingLR
    def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
        return CosineAnnealingLR(opt, T_max=total_steps, eta_min=min_lr)

try:
    from utils.metrics import (
        StepErrorAccumulator, save_metrics_csv,
        haversine_km_torch, denorm_torch,
        HORIZON_STEPS, DatasetMetrics,
    )
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    print("  ⚠  utils.metrics not found — using fallback eval")

try:
    from utils.metrics import haversine_and_atecte_torch
    HAS_ATECTE = True
except ImportError:
    HAS_ATECTE = False

# ── ST-Trans baseline (từ train v64) ──────────────────────────────────────────
TARGETS = {
    "ADE": 172.68,
    "72h": 321.39,
    "ATE": 142.21,
    "CTE":  42.04,
    "12h":  65.42,
    "24h": 104.67,
    "48h": 205.10,
}

R_EARTH = 6371.0

# ── Phase thresholds ───────────────────────────────────────────────────────────
GEN_ONLY_EPOCHS = 15
SEL_WARM_EPOCHS = 30
GEN_READY_ADE   = 160.0
SEL_READY_ACC   = 0.40


def _get_phase(epoch: int) -> int:
    if epoch < GEN_ONLY_EPOCHS: return 1
    if epoch < SEL_WARM_EPOCHS: return 2
    return 3


def _get_tau(epoch: int) -> float:
    if epoch < GEN_ONLY_EPOCHS: return 8.0
    if epoch < SEL_WARM_EPOCHS: return 6.0
    if epoch < 40:              return 5.0
    return 3.0


def _compute_rank_w(epoch, oracle_ade, sel_acc_ema, phase) -> float:
    if phase == 1: return 0.0
    if oracle_ade > GEN_READY_ADE: return 0.02
    if phase == 2:
        ep_frac = min((epoch - GEN_ONLY_EPOCHS) / max(SEL_WARM_EPOCHS - GEN_ONLY_EPOCHS, 1), 1.0)
        base = 0.02 + (0.15 - 0.02) * ep_frac
        if sel_acc_ema < 0.35: base = min(base, 0.04)
        return base
    ep_in = epoch - SEL_WARM_EPOCHS
    base = 0.15 + (0.40 - 0.15) * min(ep_in / 20.0, 1.0)
    if sel_acc_ema < SEL_READY_ACC: base = min(base, 0.12)
    return base


# ── Utilities ─────────────────────────────────────────────────────────────────

def _unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m


def _selector_params(model):
    return list(_unwrap(model).selector.parameters())


def _generator_params(model):
    sel_ids = {id(p) for p in _selector_params(model)}
    return [p for p in model.parameters() if id(p) not in sel_ids]


def _freeze_selector(model):
    for p in _selector_params(model):
        p.requires_grad_(False)


def _unfreeze_selector(model):
    for p in _selector_params(model):
        p.requires_grad_(True)


def _selector_is_frozen(model) -> bool:
    params = _selector_params(model)
    return len(params) > 0 and not params[0].requires_grad


def move(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return out


def _norm_to_deg(arr):
    out = arr.clone()
    out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
    out[..., 1] = (arr[..., 1] * 50.0) / 10.0
    return out


def _haversine(p1, p2):
    lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a = (torch.sin(dlat/2).pow(2) +
         torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2))
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())


def _compute_ate_cte(pred_deg, gt_deg):
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    lon1 = torch.deg2rad(gt_deg[:T-1,:,0]); lat1 = torch.deg2rad(gt_deg[:T-1,:,1])
    lon2 = torch.deg2rad(gt_deg[1:T, :,0]); lat2 = torch.deg2rad(gt_deg[1:T, :,1])
    lon3 = torch.deg2rad(pred_deg[1:T,:,0]); lat3 = torch.deg2rad(pred_deg[1:T,:,1])
    y_a  = torch.sin(lon2-lon1)*torch.cos(lat2)
    x_a  = torch.cos(lat1)*torch.sin(lat2) - torch.sin(lat1)*torch.cos(lat2)*torch.cos(lon2-lon1)
    bear = torch.atan2(y_a, x_a)
    y_e  = torch.sin(lon3-lon2)*torch.cos(lat3)
    x_e  = torch.cos(lat2)*torch.sin(lat3) - torch.sin(lat2)*torch.cos(lat3)*torch.cos(lon3-lon2)
    bear_e = torch.atan2(y_e, x_e)
    tot = _haversine(pred_deg[1:T], gt_deg[1:T])
    angle = bear_e - bear
    return tot * torch.cos(angle), tot * torch.sin(angle)


# ── Metrics accumulators ───────────────────────────────────────────────────────

class SimpleAccumulator:
    def __init__(self):
        self.dists = []; self.ates = []; self.ctes = []
        self.step_dists = defaultdict(list)
        _HORIZON = {12:1, 24:3, 48:7, 72:11}
        self._h = _HORIZON

    def update(self, dist, ate=None, cte=None):
        # dist: [T, B]
        self.dists.extend(dist.mean(0).tolist())
        for h, s in self._h.items():
            if s < dist.shape[0]:
                self.step_dists[h].extend(dist[s].tolist())
        if ate is not None: self.ates.extend(ate.abs().mean(0).tolist())
        if cte is not None: self.ctes.extend(cte.abs().mean(0).tolist())

    def compute(self) -> dict:
        r = {
            "ADE":      float(np.mean(self.dists)) if self.dists else float("nan"),
            "ATE_mean": float(np.mean(self.ates))  if self.ates  else float("nan"),
            "CTE_mean": float(np.mean(self.ctes))  if self.ctes  else float("nan"),
            "n_samples": len(self.dists),
        }
        for h in self._h:
            vals = self.step_dists.get(h, [])
            r[f"{h}h"] = float(np.mean(vals)) if vals else float("nan")
        return r


def _composite_score(r: dict) -> float:
    ade = r.get("ADE", float("inf"))
    h72 = r.get("72h", float("inf"))
    ate = r.get("ATE_mean", float("inf"))
    cte = r.get("CTE_mean", float("inf"))
    if not np.isfinite(ate): ate = ade * 0.46
    if not np.isfinite(cte): cte = ade * 0.53
    return 100.0 * (
        0.05*(ade/136.0) + 0.10*(r.get("12h",ade)/50.0) +
        0.15*(r.get("24h",ade)/100.0) + 0.20*(r.get("48h",ade)/200.0) +
        0.25*(h72/300.0) + 0.13*(ate/80.0) + 0.12*(cte/94.0))


def _beat_str(r: dict) -> str:
    parts = []
    for k, t in [("ADE",172.68),("ATE_mean",142.21),("CTE_mean",42.04),
                  ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
        v = r.get(k, float("inf"))
        if np.isfinite(v) and v < t:
            parts.append(f"{k.replace('_mean','')}✅{v:.1f}")
    return "🏆 BEAT: " + " ".join(parts) if parts else ""


def _gap_str(r: dict) -> str:
    parts = []
    for k, ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
        v = r.get(k, float("nan"))
        if np.isfinite(v):
            parts.append(f"{k.replace('_mean','')}:{v:.0f}({'↓' if v<ref else '↑'}{abs(v-ref):.0f})")
    return " | ".join(parts)


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, tag="", ema_obj=None) -> dict:
    backup = None
    if ema_obj is not None:
        try: backup = ema_obj.apply_to(model)
        except: pass

    model.eval()
    acc = SimpleAccumulator()
    t0 = time.perf_counter(); n = 0

    for batch in loader:
        bl = move(list(batch), device)
        # v65 sample() returns (pred_final, modes_stack) — 2 values
        result = model.sample(bl, ddim_steps=1)
        pred   = result[0]   # [T, B, 2]
        gt     = bl[1]
        T = min(pred.shape[0], gt.shape[0])
        pd = _norm_to_deg(pred[:T])
        gd = _norm_to_deg(gt[:T])
        dist = _haversine(pd, gd)
        ate, cte = _compute_ate_cte(pd, gd)
        acc.update(dist, ate, cte)
        n += 1

    if backup is not None:
        try: ema_obj.restore(model, backup)
        except: pass

    r = acc.compute()
    r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)

    def _v(k): return r.get(k, float("nan"))
    def _m(v, t): return "✅" if np.isfinite(v) and v < t else "❌"

    elapsed = (time.perf_counter() - t0)
    print(f"\n{'='*68}")
    print(f"  [{tag}  {elapsed:.0f}s  {n} batches]")
    print(f"  ADE={_v('ADE'):.1f}{_m(_v('ADE'),172.68)}  "
          f"12h={_v('12h'):.0f}{_m(_v('12h'),65.42)}  "
          f"24h={_v('24h'):.0f}{_m(_v('24h'),104.67)}  "
          f"48h={_v('48h'):.0f}{_m(_v('48h'),205.10)}  "
          f"72h={_v('72h'):.0f}{_m(_v('72h'),321.39)}")
    if np.isfinite(_v("ATE_mean")):
        print(f"  ATE={_v('ATE_mean'):.1f}{_m(_v('ATE_mean'),142.21)}  "
              f"CTE={_v('CTE_mean'):.1f}{_m(_v('CTE_mean'),42.04)}")
    print(f"  vs ST-Trans: {_gap_str(r)}")
    beat = _beat_str(r)
    if beat: print(f"  {beat}")
    print(f"  Score={_composite_score(r):.2f}")
    print(f"{'='*68}\n")
    return r


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _save_ckpt(path, epoch, model, opt, sched, saver, tl, vl, extra=None):
    m = _unwrap(model)
    ema = getattr(m, "_ema", None)
    ema_sd = None
    if ema and hasattr(ema, "shadow"):
        try: ema_sd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
        except: pass
    payload = {
        "epoch": epoch,
        "model_state_dict": m.state_dict(),
        "optimizer_state":  opt.state_dict(),
        "scheduler_state":  sched.state_dict(),
        "ema_shadow": ema_sd,
        "best_score": saver.best_score,
        "best_ade": saver.best_ade, "best_72h": saver.best_72h,
        "best_ate": saver.best_ate, "best_cte": saver.best_cte,
        "train_loss": tl, "val_loss": vl,
    }
    if extra: payload.update(extra)
    torch.save(payload, path)


class BestSaver:
    def __init__(self, patience=35):
        self.patience    = patience
        self.counter     = 0
        self.early_stop  = False
        self.best_score  = self.best_ade = self.best_72h = \
            self.best_ate = self.best_cte = float("inf")

    def update(self, r, model, out_dir, epoch, opt, sched, tl, vl,
               tag="", min_epochs=25):
        score = _composite_score(r)
        ade   = r.get("ADE", float("inf"))
        h72   = r.get("72h", float("inf"))
        ate   = r.get("ATE_mean", float("inf"))
        cte   = r.get("CTE_mean", float("inf"))

        improved = False
        for v, attr, fname in [
            (ade,   "best_ade",   "best_ade.pth"),
            (h72,   "best_72h",   "best_72h.pth"),
            (ate,   "best_ate",   "best_ate.pth"),
            (cte,   "best_cte",   "best_cte.pth"),
        ]:
            if v < getattr(self, attr):
                setattr(self, attr, v); improved = True
                _save_ckpt(os.path.join(out_dir, fname),
                           epoch, model, opt, sched, self, tl, vl)

        if score < self.best_score:
            self.best_score = score; self.counter = 0
            _save_ckpt(os.path.join(out_dir, f"best_{tag or 'composite'}.pth"),
                       epoch, model, opt, sched, self, tl, vl,
                       {"score": score, "ade": ade, "h72": h72,
                        "ate": ate, "cte": cte})
            print(f"  ✅ Best {tag} score={score:.2f}  ADE={ade:.1f}  "
                  f"72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}  (ep{epoch})")
        else:
            self.counter += 1
            print(f"  No improve {self.counter}/{self.patience}  "
                  f"(best={self.best_score:.2f} cur={score:.2f})")

        if epoch >= min_epochs and self.counter >= self.patience:
            self.early_stop = True


def make_val_subset(val_ds, size, bs, collate_fn):
    idx = random.Random(42).sample(range(len(val_ds)), min(size, len(val_ds)))
    return DataLoader(Subset(val_ds, idx), batch_size=bs, shuffle=False,
                      collate_fn=collate_fn, num_workers=0, drop_last=False)


# ── Args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root",    default="TCND_vn")
    p.add_argument("--obs_len",         default=8,      type=int)
    p.add_argument("--pred_len",        default=12,     type=int)
    p.add_argument("--batch_size",      default=32,     type=int)
    p.add_argument("--num_epochs",      default=120,    type=int)
    p.add_argument("--g_learning_rate", default=1e-4,   type=float)
    p.add_argument("--sel_lr_ratio",    default=0.10,   type=float)
    p.add_argument("--weight_decay",    default=1e-3,   type=float)
    p.add_argument("--warmup_epochs",   default=5,      type=int)
    p.add_argument("--grad_clip",       default=1.0,    type=float)
    p.add_argument("--patience",        default=35,     type=int)
    p.add_argument("--min_epochs",      default=25,     type=int)
    p.add_argument("--use_amp",         action="store_true")
    p.add_argument("--num_workers",     default=2,      type=int)
    p.add_argument("--sigma_min",       default=0.02,   type=float)
    p.add_argument("--head_noise_base", default=0.03,   type=float)
    p.add_argument("--use_ot",          default=True,   action="store_true")
    p.add_argument("--no_ot",           dest="use_ot",  action="store_false")
    p.add_argument("--cfg_guidance_scale", default=1.3, type=float)
    p.add_argument("--gen_only_epochs", default=GEN_ONLY_EPOCHS, type=int)
    p.add_argument("--sel_warm_epochs", default=SEL_WARM_EPOCHS, type=int)
    p.add_argument("--gen_ready_ade",   default=GEN_READY_ADE,   type=float)
    p.add_argument("--val_freq",        default=3,      type=int)
    p.add_argument("--val_subset_size", default=500,    type=int)
    p.add_argument("--use_ema",         default=True,   action="store_true")
    p.add_argument("--no_ema",          dest="use_ema", action="store_false")
    p.add_argument("--ema_decay",       default=0.995,  type=float)
    p.add_argument("--output_dir",      default="runs/v65")
    p.add_argument("--gpu_num",         default="0")
    p.add_argument("--delim",           default=" ")
    p.add_argument("--skip",            default=1,      type=int)
    p.add_argument("--min_ped",         default=1,      type=int)
    p.add_argument("--threshold",       default=0.002,  type=float)
    p.add_argument("--other_modal",     default="gph")
    p.add_argument("--test_year",       default=None,   type=int)
    p.add_argument("--resume",          default=None)
    p.add_argument("--resume_epoch",    default=None,   type=int)
    p.add_argument("--eval_test_after_train", default=True, action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    global GEN_ONLY_EPOCHS, SEL_WARM_EPOCHS, GEN_READY_ADE
    GEN_ONLY_EPOCHS = args.gen_only_epochs
    SEL_WARM_EPOCHS = args.sel_warm_epochs
    GEN_READY_ADE   = args.gen_ready_ade

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("  TC-FlowMatching v65  —  K=8 Compass + OT + CFG + Easy/Diff")
    print(f"  Phase 1 (ep 0–{GEN_ONLY_EPOCHS-1})  : Generator only  (selector FROZEN)")
    print(f"  Phase 2 (ep {GEN_ONLY_EPOCHS}–{SEL_WARM_EPOCHS-1}) : Selector warmup")
    print(f"  Phase 3 (ep {SEL_WARM_EPOCHS}+)   : Full joint")
    print(f"  ST-Trans target: ADE<{TARGETS['ADE']}  72h<{TARGETS['72h']}"
          f"  ATE<{TARGETS['ATE']}  CTE<{TARGETS['CTE']}")
    print("=" * 72)

    train_ds, train_loader = data_loader(
        args, {"root": args.dataset_root, "type": "train"}, test=False)
    val_ds, val_loader = data_loader(
        args, {"root": args.dataset_root, "type": "val"}, test=True)

    from Model.data.trajectoriesWithMe_unet_training import seq_collate
    val_sub = make_val_subset(val_ds, args.val_subset_size,
                               args.batch_size, seq_collate)

    print(f"  train: {len(train_ds)} seqs  val: {len(val_ds)} seqs")

    model = TCFlowMatchingV65(
        pred_len          = args.pred_len,
        obs_len           = args.obs_len,
        sigma_min         = args.sigma_min,
        use_ema           = args.use_ema,
        ema_decay         = args.ema_decay,
        use_ot            = args.use_ot,
        head_noise_base   = args.head_noise_base,
        cfg_guidance_scale= args.cfg_guidance_scale,
        # selector_warmup handled EXTERNALLY by phase logic — set to 0
        # so model never auto-freezes (we control freezing here)
        selector_warmup   = 0,
    ).to(device)

    model.init_ema()
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_p:,}")

    # Separate LR groups: generator vs selector
    optimizer = optim.AdamW([
        {"params": _generator_params(model), "lr": args.g_learning_rate,
         "name": "generator"},
        {"params": _selector_params(model),  "lr": args.g_learning_rate * args.sel_lr_ratio,
         "name": "selector"},
    ], weight_decay=args.weight_decay)

    saver  = BestSaver(patience=args.patience)
    scaler = GradScaler("cuda", enabled=args.use_amp)

    steps_ep   = len(train_loader)
    total_s    = steps_ep * args.num_epochs
    warmup_s   = steps_ep * args.warmup_epochs
    scheduler  = get_cosine_schedule_with_warmup(optimizer, warmup_s, total_s, min_lr=1e-6)

    start_epoch = 0

    # Resume
    if args.resume and os.path.exists(args.resume):
        print(f"  Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        m = _unwrap(model)
        miss, unexp = m.load_state_dict(ckpt["model_state_dict"], strict=False)
        if miss:  print(f"  Missing keys: {len(miss)}")
        if unexp: print(f"  Unexpected:   {len(unexp)}")

        ema = getattr(m, "_ema", None)
        if ema and ckpt.get("ema_shadow"):
            for k, v in ckpt["ema_shadow"].items():
                if k in ema.shadow: ema.shadow[k].copy_(v.to(device))

        try: optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e: print(f"  Opt restore: {e}")
        try: scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception as e:
            for _ in range(ckpt.get("epoch", 0) * steps_ep): scheduler.step()

        for attr in ("best_score","best_ade","best_72h","best_ate","best_cte"):
            if attr in ckpt: setattr(saver, attr, ckpt[attr])

        start_epoch = (args.resume_epoch if args.resume_epoch is not None
                       else ckpt.get("epoch", 0) + 1)
        print(f"  → Resuming from epoch {start_epoch}")

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: enabled")
    except Exception:
        pass

    # Phase state
    oracle_ade_ema = 999.0
    sel_acc_ema    = 1.0 / 8   # 1/K
    _pa            = 0.10

    prev_phase = -1
    last_tl = last_vl = 0.0
    train_start = time.perf_counter()

    print(f"  Training: {steps_ep} steps/epoch, start ep={start_epoch}")
    print("=" * 72)

    for epoch in range(start_epoch, args.num_epochs):
        phase = _get_phase(epoch)
        rank_w = _compute_rank_w(epoch, oracle_ade_ema, sel_acc_ema, phase)

        # Phase transitions
        if phase == 1:
            if not _selector_is_frozen(model):
                _freeze_selector(model)
                print(f"  🔒 ep{epoch}: Selector FROZEN (Phase 1)")
        else:
            if _selector_is_frozen(model):
                _unfreeze_selector(model)
                print(f"  🔓 ep{epoch}: Selector UNFROZEN (Phase {phase})")

        # Adjust selector LR per phase
        for pg in optimizer.param_groups:
            if pg.get("name") == "selector":
                if phase == 1:
                    pg["lr"] = 0.0
                elif phase == 2:
                    pg["lr"] = args.g_learning_rate * args.sel_lr_ratio
                else:
                    pg["lr"] = args.g_learning_rate * 0.50

        if phase != prev_phase:
            print(f"\n  ┌── Phase {phase}/3  ep={epoch}"
                  f"  oracle_ema={oracle_ade_ema:.1f}km"
                  f"  sel_acc={sel_acc_ema:.2f}"
                  f"  rank_w={rank_w:.3f}"
                  f"  tau={_get_tau(epoch):.0f}")
            prev_phase = phase

        # ── Train one epoch ───────────────────────────────────────────────
        model.train()
        sum_loss = 0.0; t0 = time.perf_counter()

        for i, batch in enumerate(train_loader):
            bl = move(list(batch), device)

            with autocast(device_type="cuda", enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl, epoch=epoch)

            optimizer.zero_grad()
            scaler.scale(bd["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            sc_before = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= sc_before: scheduler.step()
            model.ema_update()

            sum_loss += bd["total"].item()

            # Update EMAs từ v65 loss keys
            o_ade = bd.get("L_oracle", oracle_ade_ema)
            # L_oracle là MSE trong rel space, không phải km
            # Dùng delta_mean như proxy cho difficulty signal
            oracle_ade_ema = (1-_pa)*oracle_ade_ema + _pa*float(o_ade)*500.0
            # sel_acc không có trong v65 — dùng proxy: nếu L_rank nhỏ → selector tốt
            l_rank = bd.get("L_rank", 1.0)
            # Approximate: acc tăng khi rank loss giảm
            sel_acc_approx = max(0.0, min(1.0, 1.0 - float(l_rank)))
            sel_acc_ema = (1-0.05)*sel_acc_ema + 0.05*sel_acc_approx

            if i % 20 == 0:
                lr_g = next(pg["lr"] for pg in optimizer.param_groups
                             if pg.get("name") == "generator")
                frozen = "🔒" if _selector_is_frozen(model) else "🔓"
                print(
                    f"  [P{phase}][{epoch:>3}][{i:>3}/{steps_ep}]"
                    f"  tot={bd['total'].item():.3f}"
                    f"  fm={bd.get('L_FM',0):.4f}"
                    f"  rank={bd.get('L_rank',0):.4f}(w={rank_w:.3f})"
                    f"  dir={bd.get('L_dir',0):.4f}"
                    f"  coh={bd.get('L_coh',0):.4f}"
                    f"  con={bd.get('L_consist',0):.4f}"
                    f"  δ={bd.get('delta_mean',0):.2f}"
                    f"  div={bd.get('min_dist_mean',0):.3f}"
                    f"  wfm={bd.get('w_fm',1):.2f}"
                    f"  wsel={bd.get('w_sel',1):.2f}"
                    f"  wcoh={bd.get('w_coh',1):.2f}"
                    f"  wcon={bd.get('w_con',1):.2f}"
                    f"  {frozen}  lr={lr_g:.2e}"
                )

        ep_s = time.perf_counter() - t0
        avg_t = sum_loss / steps_ep

        # Val loss
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                bv = move(list(batch), device)
                with autocast(device_type="cuda", enabled=args.use_amp):
                    val_loss += model.get_loss(bv, epoch=epoch).item()
        avg_vl = val_loss / len(val_loader)
        last_tl = avg_t; last_vl = avg_vl
        print(f"  Epoch {epoch:>3}  P{phase}  "
              f"train={avg_t:.4f}  val={avg_vl:.4f}  {ep_s:.0f}s  "
              f"oracle_ema={oracle_ade_ema:.1f}  sel_acc={sel_acc_ema:.2f}")

        # Fast eval every epoch (subset)
        m_fast = evaluate(model, val_sub, device,
                          tag=f"FAST/P{phase} ep{epoch}")
        saver.update(m_fast, model, args.output_dir, epoch,
                     optimizer, scheduler, avg_t, avg_vl, tag="fast")

        # Full val every val_freq epochs
        if epoch % args.val_freq == 0:
            ema = getattr(_unwrap(model), "_ema", None)
            r_raw = evaluate(model, val_loader, device,
                             tag=f"RAW/P{phase} ep{epoch}")
            saver.update(r_raw, model, args.output_dir, epoch,
                         optimizer, scheduler, avg_t, avg_vl, tag="raw")

            if ema is not None and epoch >= 5:
                r_ema = evaluate(model, val_loader, device,
                                 tag=f"EMA/P{phase} ep{epoch}", ema_obj=ema)
                saver.update(r_ema, model, args.output_dir, epoch,
                             optimizer, scheduler, avg_t, avg_vl, tag="ema")

        # Periodic checkpoint
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            _save_ckpt(
                os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"),
                epoch, model, optimizer, scheduler, saver, avg_t, avg_vl,
                {"oracle_ade_ema": oracle_ade_ema, "sel_acc_ema": sel_acc_ema,
                 "phase": phase})

        if saver.early_stop:
            print(f"  ⛔ Early stopping at epoch {epoch}")
            break

    total_h = (time.perf_counter() - train_start) / 3600
    print(f"\n  Best: ADE={saver.best_ade:.1f}  72h={saver.best_72h:.0f}"
          f"  ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}"
          f"  ({total_h:.2f}h)")

    # Post-training test
    if args.eval_test_after_train:
        print("\n" + "="*72)
        print("  POST-TRAINING TEST EVALUATION")
        print("="*72)
        try:
            test_ds, test_loader = data_loader(
                args, {"root": args.dataset_root, "type": "test"}, test=True)
        except Exception as e:
            print(f"  ⚠  Test set unavailable: {e} — using val")
            test_loader = val_loader

        for ckpt_name, label in [
            ("best_composite.pth", "COMPOSITE"),
            ("best_72h.pth", "72h"),
            ("best_cte.pth", "CTE"),
            ("best_ema.pth", "EMA"),
        ]:
            p = os.path.join(args.output_dir, ckpt_name)
            if not os.path.exists(p): continue
            ckpt = torch.load(p, map_location=device)
            _unwrap(model).load_state_dict(ckpt["model_state_dict"], strict=False)
            ema = getattr(_unwrap(model), "_ema", None)
            if ema and ckpt.get("ema_shadow"):
                for k, v in ckpt["ema_shadow"].items():
                    if k in ema.shadow: ema.shadow[k].copy_(v.to(device))

            r = evaluate(model, test_loader, device, tag=f"TEST/{label}")
            overall = r
            print(f"\n  ── {label} TEST ──")
            for key, ref in [("ADE",172.68),("72h",321.39),
                               ("ATE_mean",142.21),("CTE_mean",42.04)]:
                v = overall.get(key, float("nan"))
                mark = "✅ BEAT!" if np.isfinite(v) and v < ref else f"❌ target:{ref:.0f}"
                print(f"    {key:<10}: {v:>8.1f} km  {mark}  [gap:{v-ref:+.1f}]")

    print("=" * 72)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    main(args)
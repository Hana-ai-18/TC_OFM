# # # # # # """
# # # # # # scripts/train_v65.py — TC-FlowMatching v65
# # # # # # ════════════════════════════════════════════════════════════════════════════
# # # # # # Phased training cho TCFlowMatchingV65.

# # # # # # PHASES (giống v64 logic):
# # # # # #   Phase 1  ep 0  .. GEN_ONLY_EPOCHS-1  : Generator only, selector FROZEN
# # # # # #   Phase 2  ep GEN_ONLY_EPOCHS .. SEL_WARM_EPOCHS-1 : Selector warmup
# # # # # #   Phase 3  ep SEL_WARM_EPOCHS+          : Full joint

# # # # # # HOW TO RUN:
# # # # # #   python scripts/train_v65.py \
# # # # # #       --dataset_root TCND_vn \
# # # # # #       --output_dir   runs/v65 \
# # # # # #       --batch_size   32 \
# # # # # #       --num_epochs   120 \
# # # # # #       --use_amp \
# # # # # #       --gpu_num 0

# # # # # # RESUME:
# # # # # #   python scripts/train_v65.py \
# # # # # #       --resume runs/v65/best_composite.pth \
# # # # # #       --output_dir runs/v65 ...
# # # # # # """
# # # # # # from __future__ import annotations

# # # # # # import sys, os
# # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # import argparse, csv, time, random
# # # # # # from collections import defaultdict
# # # # # # from datetime import datetime
# # # # # # from typing import Dict, List, Optional

# # # # # # import numpy as np
# # # # # # import torch
# # # # # # import torch.optim as optim
# # # # # # from torch.amp import autocast, GradScaler
# # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # from Model.data.loader_training import data_loader

# # # # # # # Import v65 model — alias đầy đủ
# # # # # # from Model.flow_matching_model import TCFlowMatchingV65 as TCFlowMatchingV65

# # # # # # try:
# # # # # #     from Model.utils import get_cosine_schedule_with_warmup
# # # # # # except ImportError:
# # # # # #     # Fallback scheduler nếu không có utils
# # # # # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # # # # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # # # # #         return CosineAnnealingLR(opt, T_max=total_steps, eta_min=min_lr)

# # # # # # try:
# # # # # #     from utils.metrics import (
# # # # # #         StepErrorAccumulator, save_metrics_csv,
# # # # # #         haversine_km_torch, denorm_torch,
# # # # # #         HORIZON_STEPS, DatasetMetrics,
# # # # # #     )
# # # # # #     HAS_METRICS = True
# # # # # # except ImportError:
# # # # # #     HAS_METRICS = False
# # # # # #     print("  ⚠  utils.metrics not found — using fallback eval")

# # # # # # try:
# # # # # #     from utils.metrics import haversine_and_atecte_torch
# # # # # #     HAS_ATECTE = True
# # # # # # except ImportError:
# # # # # #     HAS_ATECTE = False

# # # # # # # ── ST-Trans baseline (từ train v64) ──────────────────────────────────────────
# # # # # # TARGETS = {
# # # # # #     "ADE": 172.68,
# # # # # #     "72h": 321.39,
# # # # # #     "ATE": 142.21,
# # # # # #     "CTE":  42.04,
# # # # # #     "12h":  65.42,
# # # # # #     "24h": 104.67,
# # # # # #     "48h": 205.10,
# # # # # # }

# # # # # # R_EARTH = 6371.0

# # # # # # # ── Phase thresholds ───────────────────────────────────────────────────────────
# # # # # # GEN_ONLY_EPOCHS = 15
# # # # # # SEL_WARM_EPOCHS = 30
# # # # # # GEN_READY_ADE   = 160.0
# # # # # # SEL_READY_ACC   = 0.40


# # # # # # def _get_phase(epoch: int) -> int:
# # # # # #     if epoch < GEN_ONLY_EPOCHS: return 1
# # # # # #     if epoch < SEL_WARM_EPOCHS: return 2
# # # # # #     return 3


# # # # # # def _get_tau(epoch: int) -> float:
# # # # # #     if epoch < GEN_ONLY_EPOCHS: return 8.0
# # # # # #     if epoch < SEL_WARM_EPOCHS: return 6.0
# # # # # #     if epoch < 40:              return 5.0
# # # # # #     return 3.0


# # # # # # def _compute_rank_w(epoch, oracle_ade, sel_acc_ema, phase) -> float:
# # # # # #     if phase == 1: return 0.0
# # # # # #     if oracle_ade > GEN_READY_ADE: return 0.02
# # # # # #     if phase == 2:
# # # # # #         ep_frac = min((epoch - GEN_ONLY_EPOCHS) / max(SEL_WARM_EPOCHS - GEN_ONLY_EPOCHS, 1), 1.0)
# # # # # #         base = 0.02 + (0.15 - 0.02) * ep_frac
# # # # # #         if sel_acc_ema < 0.35: base = min(base, 0.04)
# # # # # #         return base
# # # # # #     ep_in = epoch - SEL_WARM_EPOCHS
# # # # # #     base = 0.15 + (0.40 - 0.15) * min(ep_in / 20.0, 1.0)
# # # # # #     if sel_acc_ema < SEL_READY_ACC: base = min(base, 0.12)
# # # # # #     return base


# # # # # # # ── Utilities ─────────────────────────────────────────────────────────────────

# # # # # # def _unwrap(m):
# # # # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m


# # # # # # def _selector_params(model):
# # # # # #     return list(_unwrap(model).selector.parameters())


# # # # # # def _generator_params(model):
# # # # # #     sel_ids = {id(p) for p in _selector_params(model)}
# # # # # #     return [p for p in model.parameters() if id(p) not in sel_ids]


# # # # # # def _freeze_selector(model):
# # # # # #     for p in _selector_params(model):
# # # # # #         p.requires_grad_(False)


# # # # # # def _unfreeze_selector(model):
# # # # # #     for p in _selector_params(model):
# # # # # #         p.requires_grad_(True)


# # # # # # def _selector_is_frozen(model) -> bool:
# # # # # #     params = _selector_params(model)
# # # # # #     return len(params) > 0 and not params[0].requires_grad


# # # # # # def move(batch, device):
# # # # # #     out = list(batch)
# # # # # #     for i, x in enumerate(out):
# # # # # #         if torch.is_tensor(x):
# # # # # #             out[i] = x.to(device)
# # # # # #         elif isinstance(x, dict):
# # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # #                       for k, v in x.items()}
# # # # # #     return out


# # # # # # def _norm_to_deg(arr):
# # # # # #     out = arr.clone()
# # # # # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# # # # # #     return out


# # # # # # def _haversine(p1, p2):
# # # # # #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# # # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # # #     a = (torch.sin(dlat/2).pow(2) +
# # # # # #          torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2))
# # # # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())


# # # # # # def _compute_ate_cte(pred_deg, gt_deg):
# # # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # # #     if T < 2:
# # # # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # # #         return z, z
# # # # # #     lon1 = torch.deg2rad(gt_deg[:T-1,:,0]); lat1 = torch.deg2rad(gt_deg[:T-1,:,1])
# # # # # #     lon2 = torch.deg2rad(gt_deg[1:T, :,0]); lat2 = torch.deg2rad(gt_deg[1:T, :,1])
# # # # # #     lon3 = torch.deg2rad(pred_deg[1:T,:,0]); lat3 = torch.deg2rad(pred_deg[1:T,:,1])
# # # # # #     y_a  = torch.sin(lon2-lon1)*torch.cos(lat2)
# # # # # #     x_a  = torch.cos(lat1)*torch.sin(lat2) - torch.sin(lat1)*torch.cos(lat2)*torch.cos(lon2-lon1)
# # # # # #     bear = torch.atan2(y_a, x_a)
# # # # # #     y_e  = torch.sin(lon3-lon2)*torch.cos(lat3)
# # # # # #     x_e  = torch.cos(lat2)*torch.sin(lat3) - torch.sin(lat2)*torch.cos(lat3)*torch.cos(lon3-lon2)
# # # # # #     bear_e = torch.atan2(y_e, x_e)
# # # # # #     tot = _haversine(pred_deg[1:T], gt_deg[1:T])
# # # # # #     angle = bear_e - bear
# # # # # #     return tot * torch.cos(angle), tot * torch.sin(angle)


# # # # # # # ── Metrics accumulators ───────────────────────────────────────────────────────

# # # # # # class SimpleAccumulator:
# # # # # #     def __init__(self):
# # # # # #         self.dists = []; self.ates = []; self.ctes = []
# # # # # #         self.step_dists = defaultdict(list)
# # # # # #         _HORIZON = {12:1, 24:3, 48:7, 72:11}
# # # # # #         self._h = _HORIZON

# # # # # #     def update(self, dist, ate=None, cte=None):
# # # # # #         # dist: [T, B]
# # # # # #         self.dists.extend(dist.mean(0).tolist())
# # # # # #         for h, s in self._h.items():
# # # # # #             if s < dist.shape[0]:
# # # # # #                 self.step_dists[h].extend(dist[s].tolist())
# # # # # #         if ate is not None: self.ates.extend(ate.abs().mean(0).tolist())
# # # # # #         if cte is not None: self.ctes.extend(cte.abs().mean(0).tolist())

# # # # # #     def compute(self) -> dict:
# # # # # #         r = {
# # # # # #             "ADE":      float(np.mean(self.dists)) if self.dists else float("nan"),
# # # # # #             "ATE_mean": float(np.mean(self.ates))  if self.ates  else float("nan"),
# # # # # #             "CTE_mean": float(np.mean(self.ctes))  if self.ctes  else float("nan"),
# # # # # #             "n_samples": len(self.dists),
# # # # # #         }
# # # # # #         for h in self._h:
# # # # # #             vals = self.step_dists.get(h, [])
# # # # # #             r[f"{h}h"] = float(np.mean(vals)) if vals else float("nan")
# # # # # #         return r


# # # # # # def _composite_score(r: dict) -> float:
# # # # # #     ade = r.get("ADE", float("inf"))
# # # # # #     h72 = r.get("72h", float("inf"))
# # # # # #     ate = r.get("ATE_mean", float("inf"))
# # # # # #     cte = r.get("CTE_mean", float("inf"))
# # # # # #     if not np.isfinite(ate): ate = ade * 0.46
# # # # # #     if not np.isfinite(cte): cte = ade * 0.53
# # # # # #     return 100.0 * (
# # # # # #         0.05*(ade/136.0) + 0.10*(r.get("12h",ade)/50.0) +
# # # # # #         0.15*(r.get("24h",ade)/100.0) + 0.20*(r.get("48h",ade)/200.0) +
# # # # # #         0.25*(h72/300.0) + 0.13*(ate/80.0) + 0.12*(cte/94.0))


# # # # # # def _beat_str(r: dict) -> str:
# # # # # #     parts = []
# # # # # #     for k, t in [("ADE",172.68),("ATE_mean",142.21),("CTE_mean",42.04),
# # # # # #                   ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
# # # # # #         v = r.get(k, float("inf"))
# # # # # #         if np.isfinite(v) and v < t:
# # # # # #             parts.append(f"{k.replace('_mean','')}✅{v:.1f}")
# # # # # #     return "🏆 BEAT: " + " ".join(parts) if parts else ""


# # # # # # def _gap_str(r: dict) -> str:
# # # # # #     parts = []
# # # # # #     for k, ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # # # #         v = r.get(k, float("nan"))
# # # # # #         if np.isfinite(v):
# # # # # #             parts.append(f"{k.replace('_mean','')}:{v:.0f}({'↓' if v<ref else '↑'}{abs(v-ref):.0f})")
# # # # # #     return " | ".join(parts)


# # # # # # # ── Evaluation ─────────────────────────────────────────────────────────────────

# # # # # # @torch.no_grad()
# # # # # # def evaluate(model, loader, device, tag="", ema_obj=None) -> dict:
# # # # # #     backup = None
# # # # # #     if ema_obj is not None:
# # # # # #         try: backup = ema_obj.apply_to(model)
# # # # # #         except: pass

# # # # # #     model.eval()
# # # # # #     acc = SimpleAccumulator()
# # # # # #     t0 = time.perf_counter(); n = 0

# # # # # #     for batch in loader:
# # # # # #         bl = move(list(batch), device)
# # # # # #         # v65 sample() returns (pred_final, modes_stack) — 2 values
# # # # # #         result = model.sample(bl, ddim_steps=1)
# # # # # #         pred   = result[0]   # [T, B, 2]
# # # # # #         gt     = bl[1]
# # # # # #         T = min(pred.shape[0], gt.shape[0])
# # # # # #         pd = _norm_to_deg(pred[:T])
# # # # # #         gd = _norm_to_deg(gt[:T])
# # # # # #         dist = _haversine(pd, gd)
# # # # # #         ate, cte = _compute_ate_cte(pd, gd)
# # # # # #         acc.update(dist, ate, cte)
# # # # # #         n += 1

# # # # # #     if backup is not None:
# # # # # #         try: ema_obj.restore(model, backup)
# # # # # #         except: pass

# # # # # #     r = acc.compute()
# # # # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)

# # # # # #     def _v(k): return r.get(k, float("nan"))
# # # # # #     def _m(v, t): return "✅" if np.isfinite(v) and v < t else "❌"

# # # # # #     elapsed = (time.perf_counter() - t0)
# # # # # #     print(f"\n{'='*68}")
# # # # # #     print(f"  [{tag}  {elapsed:.0f}s  {n} batches]")
# # # # # #     print(f"  ADE={_v('ADE'):.1f}{_m(_v('ADE'),172.68)}  "
# # # # # #           f"12h={_v('12h'):.0f}{_m(_v('12h'),65.42)}  "
# # # # # #           f"24h={_v('24h'):.0f}{_m(_v('24h'),104.67)}  "
# # # # # #           f"48h={_v('48h'):.0f}{_m(_v('48h'),205.10)}  "
# # # # # #           f"72h={_v('72h'):.0f}{_m(_v('72h'),321.39)}")
# # # # # #     if np.isfinite(_v("ATE_mean")):
# # # # # #         print(f"  ATE={_v('ATE_mean'):.1f}{_m(_v('ATE_mean'),142.21)}  "
# # # # # #               f"CTE={_v('CTE_mean'):.1f}{_m(_v('CTE_mean'),42.04)}")
# # # # # #     print(f"  vs ST-Trans: {_gap_str(r)}")
# # # # # #     beat = _beat_str(r)
# # # # # #     if beat: print(f"  {beat}")
# # # # # #     print(f"  Score={_composite_score(r):.2f}")
# # # # # #     print(f"{'='*68}\n")
# # # # # #     return r


# # # # # # # ── Checkpoint ────────────────────────────────────────────────────────────────

# # # # # # def _save_ckpt(path, epoch, model, opt, sched, saver, tl, vl, extra=None):
# # # # # #     m = _unwrap(model)
# # # # # #     ema = getattr(m, "_ema", None)
# # # # # #     ema_sd = None
# # # # # #     if ema and hasattr(ema, "shadow"):
# # # # # #         try: ema_sd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # # # # #         except: pass
# # # # # #     payload = {
# # # # # #         "epoch": epoch,
# # # # # #         "model_state_dict": m.state_dict(),
# # # # # #         "optimizer_state":  opt.state_dict(),
# # # # # #         "scheduler_state":  sched.state_dict(),
# # # # # #         "ema_shadow": ema_sd,
# # # # # #         "best_score": saver.best_score,
# # # # # #         "best_ade": saver.best_ade, "best_72h": saver.best_72h,
# # # # # #         "best_ate": saver.best_ate, "best_cte": saver.best_cte,
# # # # # #         "train_loss": tl, "val_loss": vl,
# # # # # #     }
# # # # # #     if extra: payload.update(extra)
# # # # # #     torch.save(payload, path)


# # # # # # class BestSaver:
# # # # # #     def __init__(self, patience=35):
# # # # # #         self.patience    = patience
# # # # # #         self.counter     = 0
# # # # # #         self.early_stop  = False
# # # # # #         self.best_score  = self.best_ade = self.best_72h = \
# # # # # #             self.best_ate = self.best_cte = float("inf")

# # # # # #     def update(self, r, model, out_dir, epoch, opt, sched, tl, vl,
# # # # # #                tag="", min_epochs=25):
# # # # # #         score = _composite_score(r)
# # # # # #         ade   = r.get("ADE", float("inf"))
# # # # # #         h72   = r.get("72h", float("inf"))
# # # # # #         ate   = r.get("ATE_mean", float("inf"))
# # # # # #         cte   = r.get("CTE_mean", float("inf"))

# # # # # #         improved = False
# # # # # #         for v, attr, fname in [
# # # # # #             (ade,   "best_ade",   "best_ade.pth"),
# # # # # #             (h72,   "best_72h",   "best_72h.pth"),
# # # # # #             (ate,   "best_ate",   "best_ate.pth"),
# # # # # #             (cte,   "best_cte",   "best_cte.pth"),
# # # # # #         ]:
# # # # # #             if v < getattr(self, attr):
# # # # # #                 setattr(self, attr, v); improved = True
# # # # # #                 _save_ckpt(os.path.join(out_dir, fname),
# # # # # #                            epoch, model, opt, sched, self, tl, vl)

# # # # # #         if score < self.best_score:
# # # # # #             self.best_score = score; self.counter = 0
# # # # # #             _save_ckpt(os.path.join(out_dir, f"best_{tag or 'composite'}.pth"),
# # # # # #                        epoch, model, opt, sched, self, tl, vl,
# # # # # #                        {"score": score, "ade": ade, "h72": h72,
# # # # # #                         "ate": ate, "cte": cte})
# # # # # #             print(f"  ✅ Best {tag} score={score:.2f}  ADE={ade:.1f}  "
# # # # # #                   f"72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}  (ep{epoch})")
# # # # # #         else:
# # # # # #             self.counter += 1
# # # # # #             print(f"  No improve {self.counter}/{self.patience}  "
# # # # # #                   f"(best={self.best_score:.2f} cur={score:.2f})")

# # # # # #         if epoch >= min_epochs and self.counter >= self.patience:
# # # # # #             self.early_stop = True


# # # # # # def make_val_subset(val_ds, size, bs, collate_fn):
# # # # # #     idx = random.Random(42).sample(range(len(val_ds)), min(size, len(val_ds)))
# # # # # #     return DataLoader(Subset(val_ds, idx), batch_size=bs, shuffle=False,
# # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # def get_args():
# # # # # #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # #     p.add_argument("--dataset_root",    default="TCND_vn")
# # # # # #     p.add_argument("--obs_len",         default=8,      type=int)
# # # # # #     p.add_argument("--pred_len",        default=12,     type=int)
# # # # # #     p.add_argument("--batch_size",      default=32,     type=int)
# # # # # #     p.add_argument("--num_epochs",      default=120,    type=int)
# # # # # #     p.add_argument("--g_learning_rate", default=1e-4,   type=float)
# # # # # #     p.add_argument("--sel_lr_ratio",    default=0.10,   type=float)
# # # # # #     p.add_argument("--weight_decay",    default=1e-3,   type=float)
# # # # # #     p.add_argument("--warmup_epochs",   default=5,      type=int)
# # # # # #     p.add_argument("--grad_clip",       default=1.0,    type=float)
# # # # # #     p.add_argument("--patience",        default=35,     type=int)
# # # # # #     p.add_argument("--min_epochs",      default=25,     type=int)
# # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # #     p.add_argument("--num_workers",     default=2,      type=int)
# # # # # #     p.add_argument("--sigma_min",       default=0.02,   type=float)
# # # # # #     p.add_argument("--head_noise_base", default=0.03,   type=float)
# # # # # #     p.add_argument("--use_ot",          default=True,   action="store_true")
# # # # # #     p.add_argument("--no_ot",           dest="use_ot",  action="store_false")
# # # # # #     p.add_argument("--cfg_guidance_scale", default=1.3, type=float)
# # # # # #     p.add_argument("--gen_only_epochs", default=GEN_ONLY_EPOCHS, type=int)
# # # # # #     p.add_argument("--sel_warm_epochs", default=SEL_WARM_EPOCHS, type=int)
# # # # # #     p.add_argument("--gen_ready_ade",   default=GEN_READY_ADE,   type=float)
# # # # # #     p.add_argument("--val_freq",        default=3,      type=int)
# # # # # #     p.add_argument("--val_subset_size", default=500,    type=int)
# # # # # #     p.add_argument("--use_ema",         default=True,   action="store_true")
# # # # # #     p.add_argument("--no_ema",          dest="use_ema", action="store_false")
# # # # # #     p.add_argument("--ema_decay",       default=0.995,  type=float)
# # # # # #     p.add_argument("--output_dir",      default="runs/v65")
# # # # # #     p.add_argument("--gpu_num",         default="0")
# # # # # #     p.add_argument("--delim",           default=" ")
# # # # # #     p.add_argument("--skip",            default=1,      type=int)
# # # # # #     p.add_argument("--min_ped",         default=1,      type=int)
# # # # # #     p.add_argument("--threshold",       default=0.002,  type=float)
# # # # # #     p.add_argument("--other_modal",     default="gph")
# # # # # #     p.add_argument("--test_year",       default=None,   type=int)
# # # # # #     p.add_argument("--resume",          default=None)
# # # # # #     p.add_argument("--resume_epoch",    default=None,   type=int)
# # # # # #     p.add_argument("--eval_test_after_train", default=True, action="store_true")
# # # # # #     return p.parse_args()


# # # # # # # ── Main ──────────────────────────────────────────────────────────────────────

# # # # # # def main(args):
# # # # # #     global GEN_ONLY_EPOCHS, SEL_WARM_EPOCHS, GEN_READY_ADE
# # # # # #     GEN_ONLY_EPOCHS = args.gen_only_epochs
# # # # # #     SEL_WARM_EPOCHS = args.sel_warm_epochs
# # # # # #     GEN_READY_ADE   = args.gen_ready_ade

# # # # # #     if torch.cuda.is_available():
# # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # #     print("=" * 72)
# # # # # #     print("  TC-FlowMatching v65  —  K=8 Compass + OT + CFG + Easy/Diff")
# # # # # #     print(f"  Phase 1 (ep 0–{GEN_ONLY_EPOCHS-1})  : Generator only  (selector FROZEN)")
# # # # # #     print(f"  Phase 2 (ep {GEN_ONLY_EPOCHS}–{SEL_WARM_EPOCHS-1}) : Selector warmup")
# # # # # #     print(f"  Phase 3 (ep {SEL_WARM_EPOCHS}+)   : Full joint")
# # # # # #     print(f"  ST-Trans target: ADE<{TARGETS['ADE']}  72h<{TARGETS['72h']}"
# # # # # #           f"  ATE<{TARGETS['ATE']}  CTE<{TARGETS['CTE']}")
# # # # # #     print("=" * 72)

# # # # # #     train_ds, train_loader = data_loader(
# # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # #     val_ds, val_loader = data_loader(
# # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # #     val_sub = make_val_subset(val_ds, args.val_subset_size,
# # # # # #                                args.batch_size, seq_collate)

# # # # # #     print(f"  train: {len(train_ds)} seqs  val: {len(val_ds)} seqs")

# # # # # #     model = TCFlowMatchingV65(
# # # # # #         pred_len          = args.pred_len,
# # # # # #         obs_len           = args.obs_len,
# # # # # #         sigma_min         = args.sigma_min,
# # # # # #         use_ema           = args.use_ema,
# # # # # #         ema_decay         = args.ema_decay,
# # # # # #         use_ot            = args.use_ot,
# # # # # #         head_noise_base   = args.head_noise_base,
# # # # # #         cfg_guidance_scale= args.cfg_guidance_scale,
# # # # # #         # selector_warmup handled EXTERNALLY by phase logic — set to 0
# # # # # #         # so model never auto-freezes (we control freezing here)
# # # # # #         selector_warmup   = 0,
# # # # # #     ).to(device)

# # # # # #     model.init_ema()
# # # # # #     n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # #     print(f"  params: {n_p:,}")

# # # # # #     # Separate LR groups: generator vs selector
# # # # # #     optimizer = optim.AdamW([
# # # # # #         {"params": _generator_params(model), "lr": args.g_learning_rate,
# # # # # #          "name": "generator"},
# # # # # #         {"params": _selector_params(model),  "lr": args.g_learning_rate * args.sel_lr_ratio,
# # # # # #          "name": "selector"},
# # # # # #     ], weight_decay=args.weight_decay)

# # # # # #     saver  = BestSaver(patience=args.patience)
# # # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)

# # # # # #     steps_ep   = len(train_loader)
# # # # # #     total_s    = steps_ep * args.num_epochs
# # # # # #     warmup_s   = steps_ep * args.warmup_epochs
# # # # # #     scheduler  = get_cosine_schedule_with_warmup(optimizer, warmup_s, total_s, min_lr=1e-6)

# # # # # #     start_epoch = 0

# # # # # #     # Resume
# # # # # #     if args.resume and os.path.exists(args.resume):
# # # # # #         print(f"  Loading checkpoint: {args.resume}")
# # # # # #         ckpt = torch.load(args.resume, map_location=device)
# # # # # #         m = _unwrap(model)
# # # # # #         miss, unexp = m.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # # #         if miss:  print(f"  Missing keys: {len(miss)}")
# # # # # #         if unexp: print(f"  Unexpected:   {len(unexp)}")

# # # # # #         ema = getattr(m, "_ema", None)
# # # # # #         if ema and ckpt.get("ema_shadow"):
# # # # # #             for k, v in ckpt["ema_shadow"].items():
# # # # # #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(device))

# # # # # #         try: optimizer.load_state_dict(ckpt["optimizer_state"])
# # # # # #         except Exception as e: print(f"  Opt restore: {e}")
# # # # # #         try: scheduler.load_state_dict(ckpt["scheduler_state"])
# # # # # #         except Exception as e:
# # # # # #             for _ in range(ckpt.get("epoch", 0) * steps_ep): scheduler.step()

# # # # # #         for attr in ("best_score","best_ade","best_72h","best_ate","best_cte"):
# # # # # #             if attr in ckpt: setattr(saver, attr, ckpt[attr])

# # # # # #         start_epoch = (args.resume_epoch if args.resume_epoch is not None
# # # # # #                        else ckpt.get("epoch", 0) + 1)
# # # # # #         print(f"  → Resuming from epoch {start_epoch}")

# # # # # #     try:
# # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # #         print("  torch.compile: enabled")
# # # # # #     except Exception:
# # # # # #         pass

# # # # # #     # Phase state
# # # # # #     oracle_ade_ema = 999.0
# # # # # #     sel_acc_ema    = 1.0 / 8   # 1/K
# # # # # #     _pa            = 0.10

# # # # # #     prev_phase = -1
# # # # # #     last_tl = last_vl = 0.0
# # # # # #     train_start = time.perf_counter()

# # # # # #     print(f"  Training: {steps_ep} steps/epoch, start ep={start_epoch}")
# # # # # #     print("=" * 72)

# # # # # #     for epoch in range(start_epoch, args.num_epochs):
# # # # # #         phase = _get_phase(epoch)
# # # # # #         rank_w = _compute_rank_w(epoch, oracle_ade_ema, sel_acc_ema, phase)

# # # # # #         # Phase transitions
# # # # # #         if phase == 1:
# # # # # #             if not _selector_is_frozen(model):
# # # # # #                 _freeze_selector(model)
# # # # # #                 print(f"  🔒 ep{epoch}: Selector FROZEN (Phase 1)")
# # # # # #         else:
# # # # # #             if _selector_is_frozen(model):
# # # # # #                 _unfreeze_selector(model)
# # # # # #                 print(f"  🔓 ep{epoch}: Selector UNFROZEN (Phase {phase})")

# # # # # #         # Adjust selector LR per phase
# # # # # #         for pg in optimizer.param_groups:
# # # # # #             if pg.get("name") == "selector":
# # # # # #                 if phase == 1:
# # # # # #                     pg["lr"] = 0.0
# # # # # #                 elif phase == 2:
# # # # # #                     pg["lr"] = args.g_learning_rate * args.sel_lr_ratio
# # # # # #                 else:
# # # # # #                     pg["lr"] = args.g_learning_rate * 0.50

# # # # # #         if phase != prev_phase:
# # # # # #             print(f"\n  ┌── Phase {phase}/3  ep={epoch}"
# # # # # #                   f"  oracle_ema={oracle_ade_ema:.1f}km"
# # # # # #                   f"  sel_acc={sel_acc_ema:.2f}"
# # # # # #                   f"  rank_w={rank_w:.3f}"
# # # # # #                   f"  tau={_get_tau(epoch):.0f}")
# # # # # #             prev_phase = phase

# # # # # #         # ── Train one epoch ───────────────────────────────────────────────
# # # # # #         model.train()
# # # # # #         sum_loss = 0.0; t0 = time.perf_counter()

# # # # # #         for i, batch in enumerate(train_loader):
# # # # # #             bl = move(list(batch), device)

# # # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # #                 bd = model.get_loss_breakdown(bl, epoch=epoch)

# # # # # #             optimizer.zero_grad()
# # # # # #             scaler.scale(bd["total"]).backward()
# # # # # #             scaler.unscale_(optimizer)
# # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

# # # # # #             sc_before = scaler.get_scale()
# # # # # #             scaler.step(optimizer); scaler.update()
# # # # # #             if scaler.get_scale() >= sc_before: scheduler.step()
# # # # # #             model.ema_update()

# # # # # #             sum_loss += bd["total"].item()

# # # # # #             # Update EMAs từ v65 loss keys
# # # # # #             o_ade = bd.get("L_oracle", oracle_ade_ema)
# # # # # #             # L_oracle là MSE trong rel space, không phải km
# # # # # #             # Dùng delta_mean như proxy cho difficulty signal
# # # # # #             oracle_ade_ema = (1-_pa)*oracle_ade_ema + _pa*float(o_ade)*500.0
# # # # # #             # sel_acc không có trong v65 — dùng proxy: nếu L_rank nhỏ → selector tốt
# # # # # #             l_rank = bd.get("L_rank", 1.0)
# # # # # #             # Approximate: acc tăng khi rank loss giảm
# # # # # #             sel_acc_approx = max(0.0, min(1.0, 1.0 - float(l_rank)))
# # # # # #             sel_acc_ema = (1-0.05)*sel_acc_ema + 0.05*sel_acc_approx

# # # # # #             if i % 20 == 0:
# # # # # #                 lr_g = next(pg["lr"] for pg in optimizer.param_groups
# # # # # #                              if pg.get("name") == "generator")
# # # # # #                 frozen = "🔒" if _selector_is_frozen(model) else "🔓"
# # # # # #                 print(
# # # # # #                     f"  [P{phase}][{epoch:>3}][{i:>3}/{steps_ep}]"
# # # # # #                     f"  tot={bd['total'].item():.3f}"
# # # # # #                     f"  fm={bd.get('L_FM',0):.4f}"
# # # # # #                     f"  rank={bd.get('L_rank',0):.4f}(w={rank_w:.3f})"
# # # # # #                     f"  dir={bd.get('L_dir',0):.4f}"
# # # # # #                     f"  coh={bd.get('L_coh',0):.4f}"
# # # # # #                     f"  con={bd.get('L_consist',0):.4f}"
# # # # # #                     f"  δ={bd.get('delta_mean',0):.2f}"
# # # # # #                     f"  div={bd.get('min_dist_mean',0):.3f}"
# # # # # #                     f"  wfm={bd.get('w_fm',1):.2f}"
# # # # # #                     f"  wsel={bd.get('w_sel',1):.2f}"
# # # # # #                     f"  wcoh={bd.get('w_coh',1):.2f}"
# # # # # #                     f"  wcon={bd.get('w_con',1):.2f}"
# # # # # #                     f"  {frozen}  lr={lr_g:.2e}"
# # # # # #                 )

# # # # # #         ep_s = time.perf_counter() - t0
# # # # # #         avg_t = sum_loss / steps_ep

# # # # # #         # Val loss
# # # # # #         model.eval(); val_loss = 0.0
# # # # # #         with torch.no_grad():
# # # # # #             for batch in val_loader:
# # # # # #                 bv = move(list(batch), device)
# # # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # #                     val_loss += model.get_loss(bv, epoch=epoch).item()
# # # # # #         avg_vl = val_loss / len(val_loader)
# # # # # #         last_tl = avg_t; last_vl = avg_vl
# # # # # #         print(f"  Epoch {epoch:>3}  P{phase}  "
# # # # # #               f"train={avg_t:.4f}  val={avg_vl:.4f}  {ep_s:.0f}s  "
# # # # # #               f"oracle_ema={oracle_ade_ema:.1f}  sel_acc={sel_acc_ema:.2f}")

# # # # # #         # Fast eval every epoch (subset)
# # # # # #         m_fast = evaluate(model, val_sub, device,
# # # # # #                           tag=f"FAST/P{phase} ep{epoch}")
# # # # # #         saver.update(m_fast, model, args.output_dir, epoch,
# # # # # #                      optimizer, scheduler, avg_t, avg_vl, tag="fast")

# # # # # #         # Full val every val_freq epochs
# # # # # #         if epoch % args.val_freq == 0:
# # # # # #             ema = getattr(_unwrap(model), "_ema", None)
# # # # # #             r_raw = evaluate(model, val_loader, device,
# # # # # #                              tag=f"RAW/P{phase} ep{epoch}")
# # # # # #             saver.update(r_raw, model, args.output_dir, epoch,
# # # # # #                          optimizer, scheduler, avg_t, avg_vl, tag="raw")

# # # # # #             if ema is not None and epoch >= 5:
# # # # # #                 r_ema = evaluate(model, val_loader, device,
# # # # # #                                  tag=f"EMA/P{phase} ep{epoch}", ema_obj=ema)
# # # # # #                 saver.update(r_ema, model, args.output_dir, epoch,
# # # # # #                              optimizer, scheduler, avg_t, avg_vl, tag="ema")

# # # # # #         # Periodic checkpoint
# # # # # #         if epoch % 10 == 0 or epoch == args.num_epochs - 1:
# # # # # #             _save_ckpt(
# # # # # #                 os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"),
# # # # # #                 epoch, model, optimizer, scheduler, saver, avg_t, avg_vl,
# # # # # #                 {"oracle_ade_ema": oracle_ade_ema, "sel_acc_ema": sel_acc_ema,
# # # # # #                  "phase": phase})

# # # # # #         if saver.early_stop:
# # # # # #             print(f"  ⛔ Early stopping at epoch {epoch}")
# # # # # #             break

# # # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # # #     print(f"\n  Best: ADE={saver.best_ade:.1f}  72h={saver.best_72h:.0f}"
# # # # # #           f"  ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}"
# # # # # #           f"  ({total_h:.2f}h)")

# # # # # #     # Post-training test
# # # # # #     if args.eval_test_after_train:
# # # # # #         print("\n" + "="*72)
# # # # # #         print("  POST-TRAINING TEST EVALUATION")
# # # # # #         print("="*72)
# # # # # #         try:
# # # # # #             test_ds, test_loader = data_loader(
# # # # # #                 args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # # # #         except Exception as e:
# # # # # #             print(f"  ⚠  Test set unavailable: {e} — using val")
# # # # # #             test_loader = val_loader

# # # # # #         for ckpt_name, label in [
# # # # # #             ("best_composite.pth", "COMPOSITE"),
# # # # # #             ("best_72h.pth", "72h"),
# # # # # #             ("best_cte.pth", "CTE"),
# # # # # #             ("best_ema.pth", "EMA"),
# # # # # #         ]:
# # # # # #             p = os.path.join(args.output_dir, ckpt_name)
# # # # # #             if not os.path.exists(p): continue
# # # # # #             ckpt = torch.load(p, map_location=device)
# # # # # #             _unwrap(model).load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # # #             ema = getattr(_unwrap(model), "_ema", None)
# # # # # #             if ema and ckpt.get("ema_shadow"):
# # # # # #                 for k, v in ckpt["ema_shadow"].items():
# # # # # #                     if k in ema.shadow: ema.shadow[k].copy_(v.to(device))

# # # # # #             r = evaluate(model, test_loader, device, tag=f"TEST/{label}")
# # # # # #             overall = r
# # # # # #             print(f"\n  ── {label} TEST ──")
# # # # # #             for key, ref in [("ADE",172.68),("72h",321.39),
# # # # # #                                ("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # # # #                 v = overall.get(key, float("nan"))
# # # # # #                 mark = "✅ BEAT!" if np.isfinite(v) and v < ref else f"❌ target:{ref:.0f}"
# # # # # #                 print(f"    {key:<10}: {v:>8.1f} km  {mark}  [gap:{v-ref:+.1f}]")

# # # # # #     print("=" * 72)


# # # # # # if __name__ == "__main__":
# # # # # #     args = get_args()
# # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # # # # #     main(args)

# # # # # """
# # # # # scripts/train_v65.py — TC-FlowMatching v65
# # # # # ════════════════════════════════════════════════════════════════════════════
# # # # # Phased training cho TCFlowMatchingV65.

# # # # # PHASES (giống v64 logic):
# # # # #   Phase 1  ep 0  .. GEN_ONLY_EPOCHS-1  : Generator only, selector FROZEN
# # # # #   Phase 2  ep GEN_ONLY_EPOCHS .. SEL_WARM_EPOCHS-1 : Selector warmup
# # # # #   Phase 3  ep SEL_WARM_EPOCHS+          : Full joint

# # # # # HOW TO RUN:
# # # # #   python scripts/train_v65.py \
# # # # #       --dataset_root TCND_vn \
# # # # #       --output_dir   runs/v65 \
# # # # #       --batch_size   32 \
# # # # #       --num_epochs   120 \
# # # # #       --use_amp \
# # # # #       --gpu_num 0

# # # # # RESUME:
# # # # #   python scripts/train_v65.py \
# # # # #       --resume runs/v65/best_composite.pth \
# # # # #       --output_dir runs/v65 ...
# # # # # """
# # # # # from __future__ import annotations

# # # # # import sys, os
# # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # import argparse, csv, time, random
# # # # # from collections import defaultdict
# # # # # from datetime import datetime
# # # # # from typing import Dict, List, Optional

# # # # # import numpy as np
# # # # # import torch
# # # # # import torch.optim as optim
# # # # # from torch.amp import autocast, GradScaler
# # # # # from torch.utils.data import DataLoader, Subset

# # # # # from Model.data.loader_training import data_loader

# # # # # # Import v65 model — alias đầy đủ
# # # # # from Model.flow_matching_model import TCFlowMatchingV65 as TCFlowMatchingV65

# # # # # try:
# # # # #     from Model.utils import get_cosine_schedule_with_warmup
# # # # # except ImportError:
# # # # #     # Fallback scheduler nếu không có utils
# # # # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # # # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # # # #         return CosineAnnealingLR(opt, T_max=total_steps, eta_min=min_lr)

# # # # # try:
# # # # #     from utils.metrics import (
# # # # #         StepErrorAccumulator, save_metrics_csv,
# # # # #         haversine_km_torch, denorm_torch,
# # # # #         HORIZON_STEPS, DatasetMetrics,
# # # # #     )
# # # # #     HAS_METRICS = True
# # # # # except ImportError:
# # # # #     HAS_METRICS = False
# # # # #     print("  ⚠  utils.metrics not found — using fallback eval")

# # # # # try:
# # # # #     from utils.metrics import haversine_and_atecte_torch
# # # # #     HAS_ATECTE = True
# # # # # except ImportError:
# # # # #     HAS_ATECTE = False

# # # # # # ── ST-Trans baseline (từ train v64) ──────────────────────────────────────────
# # # # # TARGETS = {
# # # # #     "ADE": 172.68,
# # # # #     "72h": 321.39,
# # # # #     "ATE": 142.21,
# # # # #     "CTE":  42.04,
# # # # #     "12h":  65.42,
# # # # #     "24h": 104.67,
# # # # #     "48h": 205.10,
# # # # # }

# # # # # R_EARTH = 6371.0

# # # # # # ── Phase thresholds ───────────────────────────────────────────────────────────
# # # # # GEN_ONLY_EPOCHS = 15
# # # # # SEL_WARM_EPOCHS = 30
# # # # # GEN_READY_ADE   = 160.0
# # # # # SEL_READY_ACC   = 0.40


# # # # # def _get_phase(epoch: int) -> int:
# # # # #     if epoch < GEN_ONLY_EPOCHS: return 1
# # # # #     if epoch < SEL_WARM_EPOCHS: return 2
# # # # #     return 3


# # # # # def _get_tau(epoch: int) -> float:
# # # # #     if epoch < GEN_ONLY_EPOCHS: return 8.0
# # # # #     if epoch < SEL_WARM_EPOCHS: return 6.0
# # # # #     if epoch < 40:              return 5.0
# # # # #     return 3.0


# # # # # def _compute_rank_w(epoch, oracle_ade, sel_acc_ema, phase) -> float:
# # # # #     if phase == 1: return 0.0
# # # # #     if oracle_ade > GEN_READY_ADE: return 0.02
# # # # #     if phase == 2:
# # # # #         ep_frac = min((epoch - GEN_ONLY_EPOCHS) / max(SEL_WARM_EPOCHS - GEN_ONLY_EPOCHS, 1), 1.0)
# # # # #         base = 0.02 + (0.15 - 0.02) * ep_frac
# # # # #         if sel_acc_ema < 0.35: base = min(base, 0.04)
# # # # #         return base
# # # # #     ep_in = epoch - SEL_WARM_EPOCHS
# # # # #     base = 0.15 + (0.40 - 0.15) * min(ep_in / 20.0, 1.0)
# # # # #     if sel_acc_ema < SEL_READY_ACC: base = min(base, 0.12)
# # # # #     return base


# # # # # # ── Utilities ─────────────────────────────────────────────────────────────────

# # # # # def _unwrap(m):
# # # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m


# # # # # def _selector_params(model):
# # # # #     return list(_unwrap(model).selector.parameters())


# # # # # def _generator_params(model):
# # # # #     sel_ids = {id(p) for p in _selector_params(model)}
# # # # #     return [p for p in model.parameters() if id(p) not in sel_ids]


# # # # # def _freeze_selector(model):
# # # # #     for p in _selector_params(model):
# # # # #         p.requires_grad_(False)


# # # # # def _unfreeze_selector(model):
# # # # #     for p in _selector_params(model):
# # # # #         p.requires_grad_(True)


# # # # # def _selector_is_frozen(model) -> bool:
# # # # #     params = _selector_params(model)
# # # # #     return len(params) > 0 and not params[0].requires_grad


# # # # # def move(batch, device):
# # # # #     out = list(batch)
# # # # #     for i, x in enumerate(out):
# # # # #         if torch.is_tensor(x):
# # # # #             out[i] = x.to(device)
# # # # #         elif isinstance(x, dict):
# # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # #                       for k, v in x.items()}
# # # # #     return out


# # # # # def _norm_to_deg(arr):
# # # # #     out = arr.clone()
# # # # #     out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
# # # # #     out[..., 1] = (arr[..., 1] * 50.0) / 10.0
# # # # #     return out


# # # # # def _haversine(p1, p2):
# # # # #     lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
# # # # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # # # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # # # #     a = (torch.sin(dlat/2).pow(2) +
# # # # #          torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2))
# # # # #     return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())


# # # # # def _compute_ate_cte(pred_deg, gt_deg):
# # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # #     if T < 2:
# # # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # #         return z, z
# # # # #     lon1 = torch.deg2rad(gt_deg[:T-1,:,0]); lat1 = torch.deg2rad(gt_deg[:T-1,:,1])
# # # # #     lon2 = torch.deg2rad(gt_deg[1:T, :,0]); lat2 = torch.deg2rad(gt_deg[1:T, :,1])
# # # # #     lon3 = torch.deg2rad(pred_deg[1:T,:,0]); lat3 = torch.deg2rad(pred_deg[1:T,:,1])
# # # # #     y_a  = torch.sin(lon2-lon1)*torch.cos(lat2)
# # # # #     x_a  = torch.cos(lat1)*torch.sin(lat2) - torch.sin(lat1)*torch.cos(lat2)*torch.cos(lon2-lon1)
# # # # #     bear = torch.atan2(y_a, x_a)
# # # # #     y_e  = torch.sin(lon3-lon2)*torch.cos(lat3)
# # # # #     x_e  = torch.cos(lat2)*torch.sin(lat3) - torch.sin(lat2)*torch.cos(lat3)*torch.cos(lon3-lon2)
# # # # #     bear_e = torch.atan2(y_e, x_e)
# # # # #     tot = _haversine(pred_deg[1:T], gt_deg[1:T])
# # # # #     angle = bear_e - bear
# # # # #     return tot * torch.cos(angle), tot * torch.sin(angle)


# # # # # # ── Metrics accumulators ───────────────────────────────────────────────────────

# # # # # class SimpleAccumulator:
# # # # #     def __init__(self):
# # # # #         self.dists = []; self.ates = []; self.ctes = []
# # # # #         self.step_dists = defaultdict(list)
# # # # #         _HORIZON = {12:1, 24:3, 48:7, 72:11}
# # # # #         self._h = _HORIZON

# # # # #     def update(self, dist, ate=None, cte=None):
# # # # #         # dist: [T, B]
# # # # #         self.dists.extend(dist.mean(0).tolist())
# # # # #         for h, s in self._h.items():
# # # # #             if s < dist.shape[0]:
# # # # #                 self.step_dists[h].extend(dist[s].tolist())
# # # # #         if ate is not None: self.ates.extend(ate.abs().mean(0).tolist())
# # # # #         if cte is not None: self.ctes.extend(cte.abs().mean(0).tolist())

# # # # #     def compute(self) -> dict:
# # # # #         r = {
# # # # #             "ADE":      float(np.mean(self.dists)) if self.dists else float("nan"),
# # # # #             "ATE_mean": float(np.mean(self.ates))  if self.ates  else float("nan"),
# # # # #             "CTE_mean": float(np.mean(self.ctes))  if self.ctes  else float("nan"),
# # # # #             "n_samples": len(self.dists),
# # # # #         }
# # # # #         for h in self._h:
# # # # #             vals = self.step_dists.get(h, [])
# # # # #             r[f"{h}h"] = float(np.mean(vals)) if vals else float("nan")
# # # # #         return r


# # # # # def _composite_score(r: dict) -> float:
# # # # #     ade = r.get("ADE", float("inf"))
# # # # #     h72 = r.get("72h", float("inf"))
# # # # #     ate = r.get("ATE_mean", float("inf"))
# # # # #     cte = r.get("CTE_mean", float("inf"))
# # # # #     if not np.isfinite(ate): ate = ade * 0.46
# # # # #     if not np.isfinite(cte): cte = ade * 0.53
# # # # #     return 100.0 * (
# # # # #         0.05*(ade/136.0) + 0.10*(r.get("12h",ade)/50.0) +
# # # # #         0.15*(r.get("24h",ade)/100.0) + 0.20*(r.get("48h",ade)/200.0) +
# # # # #         0.25*(h72/300.0) + 0.13*(ate/80.0) + 0.12*(cte/94.0))


# # # # # def _beat_str(r: dict) -> str:
# # # # #     parts = []
# # # # #     for k, t in [("ADE",172.68),("ATE_mean",142.21),("CTE_mean",42.04),
# # # # #                   ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
# # # # #         v = r.get(k, float("inf"))
# # # # #         if np.isfinite(v) and v < t:
# # # # #             parts.append(f"{k.replace('_mean','')}✅{v:.1f}")
# # # # #     return "🏆 BEAT: " + " ".join(parts) if parts else ""


# # # # # def _gap_str(r: dict) -> str:
# # # # #     parts = []
# # # # #     for k, ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # # #         v = r.get(k, float("nan"))
# # # # #         if np.isfinite(v):
# # # # #             parts.append(f"{k.replace('_mean','')}:{v:.0f}({'↓' if v<ref else '↑'}{abs(v-ref):.0f})")
# # # # #     return " | ".join(parts)


# # # # # # ── Evaluation ─────────────────────────────────────────────────────────────────

# # # # # @torch.no_grad()
# # # # # def evaluate(model, loader, device, tag="", ema_obj=None) -> dict:
# # # # #     backup = None
# # # # #     if ema_obj is not None:
# # # # #         try: backup = ema_obj.apply_to(model)
# # # # #         except: pass

# # # # #     model.eval()
# # # # #     acc = SimpleAccumulator()
# # # # #     t0 = time.perf_counter(); n = 0

# # # # #     for batch in loader:
# # # # #         bl = move(list(batch), device)
# # # # #         # v65 sample() returns (pred_final, modes_stack) — 2 values
# # # # #         result = model.sample(bl, ddim_steps=1)
# # # # #         pred   = result[0]   # [T, B, 2]
# # # # #         gt     = bl[1]
# # # # #         T = min(pred.shape[0], gt.shape[0])
# # # # #         pd = _norm_to_deg(pred[:T])
# # # # #         gd = _norm_to_deg(gt[:T])
# # # # #         dist = _haversine(pd, gd)
# # # # #         ate, cte = _compute_ate_cte(pd, gd)
# # # # #         acc.update(dist, ate, cte)
# # # # #         n += 1

# # # # #     if backup is not None:
# # # # #         try: ema_obj.restore(model, backup)
# # # # #         except: pass

# # # # #     r = acc.compute()
# # # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)

# # # # #     def _v(k): return r.get(k, float("nan"))
# # # # #     def _m(v, t): return "✅" if np.isfinite(v) and v < t else "❌"

# # # # #     elapsed = (time.perf_counter() - t0)
# # # # #     print(f"\n{'='*68}")
# # # # #     print(f"  [{tag}  {elapsed:.0f}s  {n} batches]")
# # # # #     print(f"  ADE={_v('ADE'):.1f}{_m(_v('ADE'),172.68)}  "
# # # # #           f"12h={_v('12h'):.0f}{_m(_v('12h'),65.42)}  "
# # # # #           f"24h={_v('24h'):.0f}{_m(_v('24h'),104.67)}  "
# # # # #           f"48h={_v('48h'):.0f}{_m(_v('48h'),205.10)}  "
# # # # #           f"72h={_v('72h'):.0f}{_m(_v('72h'),321.39)}")
# # # # #     if np.isfinite(_v("ATE_mean")):
# # # # #         print(f"  ATE={_v('ATE_mean'):.1f}{_m(_v('ATE_mean'),142.21)}  "
# # # # #               f"CTE={_v('CTE_mean'):.1f}{_m(_v('CTE_mean'),42.04)}")
# # # # #     print(f"  vs ST-Trans: {_gap_str(r)}")
# # # # #     beat = _beat_str(r)
# # # # #     if beat: print(f"  {beat}")
# # # # #     print(f"  Score={_composite_score(r):.2f}")
# # # # #     print(f"{'='*68}\n")
# # # # #     return r


# # # # # # ── Checkpoint ────────────────────────────────────────────────────────────────

# # # # # def _save_ckpt(path, epoch, model, opt, sched, saver, tl, vl, extra=None):
# # # # #     m = _unwrap(model)
# # # # #     ema = getattr(m, "_ema", None)
# # # # #     ema_sd = None
# # # # #     if ema and hasattr(ema, "shadow"):
# # # # #         try: ema_sd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # # # #         except: pass
# # # # #     payload = {
# # # # #         "epoch": epoch,
# # # # #         "model_state_dict": m.state_dict(),
# # # # #         "optimizer_state":  opt.state_dict(),
# # # # #         "scheduler_state":  sched.state_dict(),
# # # # #         "ema_shadow": ema_sd,
# # # # #         "best_score": saver.best_score,
# # # # #         "best_ade": saver.best_ade, "best_72h": saver.best_72h,
# # # # #         "best_ate": saver.best_ate, "best_cte": saver.best_cte,
# # # # #         "train_loss": tl, "val_loss": vl,
# # # # #     }
# # # # #     if extra: payload.update(extra)
# # # # #     torch.save(payload, path)


# # # # # class BestSaver:
# # # # #     def __init__(self, patience=35):
# # # # #         self.patience    = patience
# # # # #         self.counter     = 0
# # # # #         self.early_stop  = False
# # # # #         self.best_score  = self.best_ade = self.best_72h = \
# # # # #             self.best_ate = self.best_cte = float("inf")

# # # # #     def update(self, r, model, out_dir, epoch, opt, sched, tl, vl,
# # # # #                tag="", min_epochs=25):
# # # # #         score = _composite_score(r)
# # # # #         ade   = r.get("ADE", float("inf"))
# # # # #         h72   = r.get("72h", float("inf"))
# # # # #         ate   = r.get("ATE_mean", float("inf"))
# # # # #         cte   = r.get("CTE_mean", float("inf"))

# # # # #         improved = False
# # # # #         for v, attr, fname in [
# # # # #             (ade,   "best_ade",   "best_ade.pth"),
# # # # #             (h72,   "best_72h",   "best_72h.pth"),
# # # # #             (ate,   "best_ate",   "best_ate.pth"),
# # # # #             (cte,   "best_cte",   "best_cte.pth"),
# # # # #         ]:
# # # # #             if v < getattr(self, attr):
# # # # #                 setattr(self, attr, v); improved = True
# # # # #                 _save_ckpt(os.path.join(out_dir, fname),
# # # # #                            epoch, model, opt, sched, self, tl, vl)

# # # # #         if score < self.best_score:
# # # # #             self.best_score = score; self.counter = 0
# # # # #             _save_ckpt(os.path.join(out_dir, f"best_{tag or 'composite'}.pth"),
# # # # #                        epoch, model, opt, sched, self, tl, vl,
# # # # #                        {"score": score, "ade": ade, "h72": h72,
# # # # #                         "ate": ate, "cte": cte})
# # # # #             print(f"  ✅ Best {tag} score={score:.2f}  ADE={ade:.1f}  "
# # # # #                   f"72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}  (ep{epoch})")
# # # # #         else:
# # # # #             self.counter += 1
# # # # #             print(f"  No improve {self.counter}/{self.patience}  "
# # # # #                   f"(best={self.best_score:.2f} cur={score:.2f})")

# # # # #         if epoch >= min_epochs and self.counter >= self.patience:
# # # # #             self.early_stop = True


# # # # # def make_val_subset(val_ds, size, bs, collate_fn):
# # # # #     idx = random.Random(42).sample(range(len(val_ds)), min(size, len(val_ds)))
# # # # #     return DataLoader(Subset(val_ds, idx), batch_size=bs, shuffle=False,
# # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # def get_args():
# # # # #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # #     p.add_argument("--dataset_root",    default="TCND_vn")
# # # # #     p.add_argument("--obs_len",         default=8,      type=int)
# # # # #     p.add_argument("--pred_len",        default=12,     type=int)
# # # # #     p.add_argument("--batch_size",      default=32,     type=int)
# # # # #     p.add_argument("--num_epochs",      default=120,    type=int)
# # # # #     p.add_argument("--g_learning_rate", default=1e-4,   type=float)
# # # # #     p.add_argument("--sel_lr_ratio",    default=0.10,   type=float)
# # # # #     p.add_argument("--weight_decay",    default=1e-3,   type=float)
# # # # #     p.add_argument("--warmup_epochs",   default=5,      type=int)
# # # # #     p.add_argument("--grad_clip",       default=1.0,    type=float)
# # # # #     p.add_argument("--patience",        default=35,     type=int)
# # # # #     p.add_argument("--min_epochs",      default=25,     type=int)
# # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # #     p.add_argument("--num_workers",     default=2,      type=int)
# # # # #     p.add_argument("--sigma_min",       default=0.02,   type=float)
# # # # #     p.add_argument("--head_noise_base", default=0.03,   type=float)
# # # # #     p.add_argument("--use_ot",          default=True,   action="store_true")
# # # # #     p.add_argument("--no_ot",           dest="use_ot",  action="store_false")
# # # # #     p.add_argument("--cfg_guidance_scale", default=1.3, type=float)
# # # # #     p.add_argument("--gen_only_epochs", default=GEN_ONLY_EPOCHS, type=int)
# # # # #     p.add_argument("--sel_warm_epochs", default=SEL_WARM_EPOCHS, type=int)
# # # # #     p.add_argument("--gen_ready_ade",   default=GEN_READY_ADE,   type=float)
# # # # #     p.add_argument("--val_freq",        default=3,      type=int)
# # # # #     p.add_argument("--val_subset_size", default=500,    type=int)
# # # # #     p.add_argument("--use_ema",         default=True,   action="store_true")
# # # # #     p.add_argument("--no_ema",          dest="use_ema", action="store_false")
# # # # #     p.add_argument("--ema_decay",       default=0.995,  type=float)
# # # # #     p.add_argument("--output_dir",      default="runs/v65")
# # # # #     p.add_argument("--gpu_num",         default="0")
# # # # #     p.add_argument("--delim",           default=" ")
# # # # #     p.add_argument("--skip",            default=1,      type=int)
# # # # #     p.add_argument("--min_ped",         default=1,      type=int)
# # # # #     p.add_argument("--threshold",       default=0.002,  type=float)
# # # # #     p.add_argument("--other_modal",     default="gph")
# # # # #     p.add_argument("--test_year",       default=None,   type=int)
# # # # #     p.add_argument("--resume",          default=None)
# # # # #     p.add_argument("--resume_epoch",    default=None,   type=int)
# # # # #     p.add_argument("--eval_test_after_train", default=True, action="store_true")
# # # # #     return p.parse_args()


# # # # # # ── Main ──────────────────────────────────────────────────────────────────────

# # # # # def main(args):
# # # # #     global GEN_ONLY_EPOCHS, SEL_WARM_EPOCHS, GEN_READY_ADE
# # # # #     GEN_ONLY_EPOCHS = args.gen_only_epochs
# # # # #     SEL_WARM_EPOCHS = args.sel_warm_epochs
# # # # #     GEN_READY_ADE   = args.gen_ready_ade

# # # # #     if torch.cuda.is_available():
# # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # #     print("=" * 72)
# # # # #     print("  TC-FlowMatching v65  —  K=8 Compass + OT + CFG + Easy/Diff")
# # # # #     print(f"  Phase 1 (ep 0–{GEN_ONLY_EPOCHS-1})  : Generator only  (selector FROZEN)")
# # # # #     print(f"  Phase 2 (ep {GEN_ONLY_EPOCHS}–{SEL_WARM_EPOCHS-1}) : Selector warmup")
# # # # #     print(f"  Phase 3 (ep {SEL_WARM_EPOCHS}+)   : Full joint")
# # # # #     print(f"  ST-Trans target: ADE<{TARGETS['ADE']}  72h<{TARGETS['72h']}"
# # # # #           f"  ATE<{TARGETS['ATE']}  CTE<{TARGETS['CTE']}")
# # # # #     print("=" * 72)

# # # # #     train_ds, train_loader = data_loader(
# # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # #     val_ds, val_loader = data_loader(
# # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # #     val_sub = make_val_subset(val_ds, args.val_subset_size,
# # # # #                                args.batch_size, seq_collate)

# # # # #     print(f"  train: {len(train_ds)} seqs  val: {len(val_ds)} seqs")

# # # # #     model = TCFlowMatchingV65(
# # # # #         pred_len          = args.pred_len,
# # # # #         obs_len           = args.obs_len,
# # # # #         sigma_min         = args.sigma_min,
# # # # #         use_ema           = args.use_ema,
# # # # #         ema_decay         = args.ema_decay,
# # # # #         use_ot            = args.use_ot,
# # # # #         head_noise_base   = args.head_noise_base,
# # # # #         cfg_guidance_scale= args.cfg_guidance_scale,
# # # # #         # selector_warmup handled EXTERNALLY by phase logic — set to 0
# # # # #         # so model never auto-freezes (we control freezing here)
# # # # #         selector_warmup   = 0,
# # # # #     ).to(device)

# # # # #     model.init_ema()
# # # # #     n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # #     print(f"  params: {n_p:,}")

# # # # #     # Separate LR groups: generator vs selector
# # # # #     optimizer = optim.AdamW([
# # # # #         {"params": _generator_params(model), "lr": args.g_learning_rate,
# # # # #          "name": "generator"},
# # # # #         {"params": _selector_params(model),  "lr": args.g_learning_rate * args.sel_lr_ratio,
# # # # #          "name": "selector"},
# # # # #     ], weight_decay=args.weight_decay)

# # # # #     saver  = BestSaver(patience=args.patience)
# # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)

# # # # #     steps_ep   = len(train_loader)
# # # # #     total_s    = steps_ep * args.num_epochs
# # # # #     warmup_s   = steps_ep * args.warmup_epochs
# # # # #     scheduler  = get_cosine_schedule_with_warmup(optimizer, warmup_s, total_s, min_lr=1e-6)

# # # # #     start_epoch = 0

# # # # #     # Resume
# # # # #     if args.resume and os.path.exists(args.resume):
# # # # #         print(f"  Loading checkpoint: {args.resume}")
# # # # #         ckpt = torch.load(args.resume, map_location=device)
# # # # #         m = _unwrap(model)
# # # # #         miss, unexp = m.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # #         if miss:  print(f"  Missing keys: {len(miss)}")
# # # # #         if unexp: print(f"  Unexpected:   {len(unexp)}")

# # # # #         ema = getattr(m, "_ema", None)
# # # # #         if ema and ckpt.get("ema_shadow"):
# # # # #             for k, v in ckpt["ema_shadow"].items():
# # # # #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(device))

# # # # #         try: optimizer.load_state_dict(ckpt["optimizer_state"])
# # # # #         except Exception as e: print(f"  Opt restore: {e}")
# # # # #         try: scheduler.load_state_dict(ckpt["scheduler_state"])
# # # # #         except Exception as e:
# # # # #             for _ in range(ckpt.get("epoch", 0) * steps_ep): scheduler.step()

# # # # #         for attr in ("best_score","best_ade","best_72h","best_ate","best_cte"):
# # # # #             if attr in ckpt: setattr(saver, attr, ckpt[attr])

# # # # #         start_epoch = (args.resume_epoch if args.resume_epoch is not None
# # # # #                        else ckpt.get("epoch", 0) + 1)
# # # # #         print(f"  → Resuming from epoch {start_epoch}")

# # # # #     try:
# # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # #         print("  torch.compile: enabled")
# # # # #     except Exception:
# # # # #         pass

# # # # #     # Phase state
# # # # #     oracle_ade_ema = 999.0
# # # # #     sel_acc_ema    = 1.0 / 8   # 1/K
# # # # #     _pa            = 0.10

# # # # #     prev_phase = -1
# # # # #     last_tl = last_vl = 0.0
# # # # #     train_start = time.perf_counter()

# # # # #     print(f"  Training: {steps_ep} steps/epoch, start ep={start_epoch}")
# # # # #     print("=" * 72)

# # # # #     for epoch in range(start_epoch, args.num_epochs):
# # # # #         phase = _get_phase(epoch)
# # # # #         rank_w = _compute_rank_w(epoch, oracle_ade_ema, sel_acc_ema, phase)

# # # # #         # Phase transitions
# # # # #         if phase == 1:
# # # # #             if not _selector_is_frozen(model):
# # # # #                 _freeze_selector(model)
# # # # #                 print(f"  🔒 ep{epoch}: Selector FROZEN (Phase 1)")
# # # # #         else:
# # # # #             if _selector_is_frozen(model):
# # # # #                 _unfreeze_selector(model)
# # # # #                 print(f"  🔓 ep{epoch}: Selector UNFROZEN (Phase {phase})")

# # # # #         # Adjust selector LR per phase
# # # # #         for pg in optimizer.param_groups:
# # # # #             if pg.get("name") == "selector":
# # # # #                 if phase == 1:
# # # # #                     pg["lr"] = 0.0
# # # # #                 elif phase == 2:
# # # # #                     pg["lr"] = args.g_learning_rate * args.sel_lr_ratio
# # # # #                 else:
# # # # #                     pg["lr"] = args.g_learning_rate * 0.50

# # # # #         if phase != prev_phase:
# # # # #             print(f"\n  ┌── Phase {phase}/3  ep={epoch}"
# # # # #                   f"  oracle_ema={oracle_ade_ema:.1f}km"
# # # # #                   f"  sel_acc={sel_acc_ema:.2f}"
# # # # #                   f"  rank_w={rank_w:.3f}"
# # # # #                   f"  tau={_get_tau(epoch):.0f}")
# # # # #             prev_phase = phase

# # # # #         # ── Train one epoch ───────────────────────────────────────────────
# # # # #         model.train()
# # # # #         sum_loss = 0.0; t0 = time.perf_counter()

# # # # #         for i, batch in enumerate(train_loader):
# # # # #             bl = move(list(batch), device)

# # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # #                 tau_ep = _get_tau(epoch)
# # # # #                 bd = model.get_loss_breakdown(bl, epoch=epoch, tau=tau_ep)

# # # # #             optimizer.zero_grad()
# # # # #             scaler.scale(bd["total"]).backward()
# # # # #             scaler.unscale_(optimizer)
# # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

# # # # #             sc_before = scaler.get_scale()
# # # # #             scaler.step(optimizer); scaler.update()
# # # # #             if scaler.get_scale() >= sc_before: scheduler.step()
# # # # #             model.ema_update()

# # # # #             sum_loss += bd["total"].item()

# # # # #             # BUG-2 FIX: dùng oracle_ade_km (km thực) thay vì MSE*500
# # # # #             o_ade_km = bd.get("oracle_ade_km", oracle_ade_ema)
# # # # #             oracle_ade_ema = (1-_pa)*oracle_ade_ema + _pa*float(o_ade_km)
# # # # #             # sel_acc proxy: L_rank nhỏ → selector tốt hơn
# # # # #             l_rank = bd.get("L_rank", 1.0)
# # # # #             sel_acc_approx = max(0.0, min(1.0, 1.0 - float(l_rank)))
# # # # #             sel_acc_ema = (1-0.05)*sel_acc_ema + 0.05*sel_acc_approx

# # # # #             if i % 20 == 0:
# # # # #                 lr_g = next(pg["lr"] for pg in optimizer.param_groups
# # # # #                              if pg.get("name") == "generator")
# # # # #                 frozen = "🔒" if _selector_is_frozen(model) else "🔓"
# # # # #                 print(
# # # # #                     f"  [P{phase}][{epoch:>3}][{i:>3}/{steps_ep}]"
# # # # #                     f"  tot={bd['total'].item():.3f}"
# # # # #                     f"  fm={bd.get('L_FM',0):.4f}"
# # # # #                     f"  rank={bd.get('L_rank',0):.4f}(w={rank_w:.3f})"
# # # # #                     f"  dir={bd.get('L_dir',0):.4f}"
# # # # #                     f"  coh={bd.get('L_coh',0):.4f}"
# # # # #                     f"  con={bd.get('L_consist',0):.4f}"
# # # # #                     f"  oracle={bd.get('oracle_ade_km',0):.0f}km"
# # # # #                     f"  δ={bd.get('delta_mean',0):.2f}"
# # # # #                     f"  div={bd.get('min_dist_mean',0):.3f}"
# # # # #                     f"  wfm={bd.get('w_fm',1):.2f}"
# # # # #                     f"  wsel={bd.get('w_sel',1):.2f}"
# # # # #                     f"  wcoh={bd.get('w_coh',1):.2f}"
# # # # #                     f"  wcon={bd.get('w_con',1):.2f}"
# # # # #                     f"  {frozen}  lr={lr_g:.2e}"
# # # # #                 )

# # # # #         ep_s = time.perf_counter() - t0
# # # # #         avg_t = sum_loss / steps_ep

# # # # #         # Val loss
# # # # #         model.eval(); val_loss = 0.0
# # # # #         with torch.no_grad():
# # # # #             for batch in val_loader:
# # # # #                 bv = move(list(batch), device)
# # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # #                     val_loss += model.get_loss(bv, epoch=epoch, tau=_get_tau(epoch)).item()
# # # # #         avg_vl = val_loss / len(val_loader)
# # # # #         last_tl = avg_t; last_vl = avg_vl
# # # # #         print(f"  Epoch {epoch:>3}  P{phase}  "
# # # # #               f"train={avg_t:.4f}  val={avg_vl:.4f}  {ep_s:.0f}s  "
# # # # #               f"oracle_ema={oracle_ade_ema:.1f}  sel_acc={sel_acc_ema:.2f}")

# # # # #         # Fast eval every epoch (subset)
# # # # #         m_fast = evaluate(model, val_sub, device,
# # # # #                           tag=f"FAST/P{phase} ep{epoch}")
# # # # #         saver.update(m_fast, model, args.output_dir, epoch,
# # # # #                      optimizer, scheduler, avg_t, avg_vl, tag="fast")

# # # # #         # Full val every val_freq epochs
# # # # #         if epoch % args.val_freq == 0:
# # # # #             ema = getattr(_unwrap(model), "_ema", None)
# # # # #             r_raw = evaluate(model, val_loader, device,
# # # # #                              tag=f"RAW/P{phase} ep{epoch}")
# # # # #             saver.update(r_raw, model, args.output_dir, epoch,
# # # # #                          optimizer, scheduler, avg_t, avg_vl, tag="raw")

# # # # #             if ema is not None and epoch >= 5:
# # # # #                 r_ema = evaluate(model, val_loader, device,
# # # # #                                  tag=f"EMA/P{phase} ep{epoch}", ema_obj=ema)
# # # # #                 saver.update(r_ema, model, args.output_dir, epoch,
# # # # #                              optimizer, scheduler, avg_t, avg_vl, tag="ema")

# # # # #         # Periodic checkpoint
# # # # #         if epoch % 10 == 0 or epoch == args.num_epochs - 1:
# # # # #             _save_ckpt(
# # # # #                 os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"),
# # # # #                 epoch, model, optimizer, scheduler, saver, avg_t, avg_vl,
# # # # #                 {"oracle_ade_ema": oracle_ade_ema, "sel_acc_ema": sel_acc_ema,
# # # # #                  "phase": phase})

# # # # #         if saver.early_stop:
# # # # #             print(f"  ⛔ Early stopping at epoch {epoch}")
# # # # #             break

# # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # #     print(f"\n  Best: ADE={saver.best_ade:.1f}  72h={saver.best_72h:.0f}"
# # # # #           f"  ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}"
# # # # #           f"  ({total_h:.2f}h)")

# # # # #     # Post-training test
# # # # #     if args.eval_test_after_train:
# # # # #         print("\n" + "="*72)
# # # # #         print("  POST-TRAINING TEST EVALUATION")
# # # # #         print("="*72)
# # # # #         try:
# # # # #             test_ds, test_loader = data_loader(
# # # # #                 args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # # #         except Exception as e:
# # # # #             print(f"  ⚠  Test set unavailable: {e} — using val")
# # # # #             test_loader = val_loader

# # # # #         for ckpt_name, label in [
# # # # #             ("best_composite.pth", "COMPOSITE"),
# # # # #             ("best_72h.pth", "72h"),
# # # # #             ("best_cte.pth", "CTE"),
# # # # #             ("best_ema.pth", "EMA"),
# # # # #         ]:
# # # # #             p = os.path.join(args.output_dir, ckpt_name)
# # # # #             if not os.path.exists(p): continue
# # # # #             ckpt = torch.load(p, map_location=device)
# # # # #             _unwrap(model).load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # #             ema = getattr(_unwrap(model), "_ema", None)
# # # # #             if ema and ckpt.get("ema_shadow"):
# # # # #                 for k, v in ckpt["ema_shadow"].items():
# # # # #                     if k in ema.shadow: ema.shadow[k].copy_(v.to(device))

# # # # #             r = evaluate(model, test_loader, device, tag=f"TEST/{label}")
# # # # #             overall = r
# # # # #             print(f"\n  ── {label} TEST ──")
# # # # #             for key, ref in [("ADE",172.68),("72h",321.39),
# # # # #                                ("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # # #                 v = overall.get(key, float("nan"))
# # # # #                 mark = "✅ BEAT!" if np.isfinite(v) and v < ref else f"❌ target:{ref:.0f}"
# # # # #                 print(f"    {key:<10}: {v:>8.1f} km  {mark}  [gap:{v-ref:+.1f}]")

# # # # #     print("=" * 72)


# # # # # if __name__ == "__main__":
# # # # #     args = get_args()
# # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # # # #     main(args)


# # # # """
# # # # train_v68.py — TC-FlowMatching v68
# # # # ════════════════════════════════════════════════════════════════════════
# # # # 3-phase training với freeze/unfreeze chính xác:
# # # #   Phase 1 (0..SEL_WARMUP-1):       Generator trains, selector FROZEN
# # # #   Phase 2 (SEL_WARMUP..+14):       Selector trains, generator+encoder FROZEN
# # # #   Phase 3 (SEL_WARMUP+15..end):    Joint, tất cả unfrozen

# # # # Key fixes so với train_v65:
# # # #   - oracle_ade_ema từ bd["oracle_ade_km"] (km thực)
# # # #   - Phase 2: freeze velocity_heads + encoder (không chỉ selector)
# # # #   - ddim_steps=10 cho fast eval (không phải 1)
# # # #   - Log step_weights để monitor learning
# # # #   - Generator LR không set 0 trong Phase 2 (dùng requires_grad=False thay vì)

# # # # HOW TO RUN:
# # # #   python scripts/train_v68.py \
# # # #       --dataset_root /kaggle/input/.../tc-ofm \
# # # #       --output_dir   runs/v68 \
# # # #       --batch_size   32 \
# # # #       --num_epochs   120 \
# # # #       --sel_warmup   20 \
# # # #       --use_amp
# # # # """
# # # # from __future__ import annotations

# # # # import sys, os
# # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # import argparse, time, random
# # # # from collections import defaultdict

# # # # import numpy as np
# # # # import torch
# # # # import torch.optim as optim
# # # # from torch.amp import autocast, GradScaler
# # # # from torch.utils.data import DataLoader, Subset

# # # # from Model.data.loader_training import data_loader
# # # # from Model.flow_matching_model import TCFlowMatchingV68

# # # # try:
# # # #     from Model.utils import get_cosine_schedule_with_warmup
# # # # except ImportError:
# # # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # # #         return CosineAnnealingLR(opt, T_max=total_steps, eta_min=min_lr)

# # # # TARGETS = {"ADE":172.68,"72h":321.39,"ATE":142.21,"CTE":42.04,
# # # #            "12h":65.42,"24h":104.67,"48h":205.10}
# # # # R_EARTH = 6371.0


# # # # def _unwrap(m): return m._orig_mod if hasattr(m,'_orig_mod') else m
# # # # def _sel_params(m): return list(_unwrap(m).selector.parameters())
# # # # def _gen_params(m):  # encoder + velocity_heads + step_weights
# # # #     u=_unwrap(m)
# # # #     return list(u.encoder.parameters())+list(u.velocity_heads.parameters())+list(u.step_weights.parameters())

# # # # def _freeze(params):
# # # #     for p in params: p.requires_grad_(False)
# # # # def _unfreeze(params):
# # # #     for p in params: p.requires_grad_(True)
# # # # def _sel_frozen(m):
# # # #     p=_sel_params(m); return len(p)>0 and not p[0].requires_grad

# # # # def _get_phase(ep, sw):
# # # #     if ep<sw: return 1
# # # #     elif ep<sw+15: return 2
# # # #     return 3

# # # # def move(b,dev):
# # # #     out=list(b)
# # # #     for i,x in enumerate(out):
# # # #         if torch.is_tensor(x): out[i]=x.to(dev)
# # # #         elif isinstance(x,dict): out[i]={k:v.to(dev) if torch.is_tensor(v) else v for k,v in x.items()}
# # # #     return out

# # # # def _ntd(a):
# # # #     o=a.clone(); o[...,0]=(a[...,0]*50.+1800.)/10.; o[...,1]=(a[...,1]*50.)/10.; return o

# # # # def _hav(p1,p2):
# # # #     la1=torch.deg2rad(p1[...,1]);la2=torch.deg2rad(p2[...,1])
# # # #     dlat=torch.deg2rad(p2[...,1]-p1[...,1]);dlon=torch.deg2rad(p2[...,0]-p1[...,0])
# # # #     a=(torch.sin(dlat/2).pow(2)+torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
# # # #     return 2.*R_EARTH*torch.asin(a.clamp(1e-12,1-1e-12).sqrt())

# # # # def _atecte(pd,gd):
# # # #     T=min(pd.shape[0],gd.shape[0])
# # # #     if T<2: z=pd.new_zeros(1,pd.shape[1]); return z,z
# # # #     lo1=torch.deg2rad(gd[:T-1,:,0]);la1=torch.deg2rad(gd[:T-1,:,1])
# # # #     lo2=torch.deg2rad(gd[1:T, :,0]);la2=torch.deg2rad(gd[1:T, :,1])
# # # #     lo3=torch.deg2rad(pd[1:T,:,0]);la3=torch.deg2rad(pd[1:T,:,1])
# # # #     ya=torch.sin(lo2-lo1)*torch.cos(la2); xa=torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # # #     be=torch.atan2(ya,xa)
# # # #     ye=torch.sin(lo3-lo2)*torch.cos(la3); xe=torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # # #     bee=torch.atan2(ye,xe)
# # # #     tot=_hav(pd[1:T],gd[1:T]); ang=bee-be
# # # #     return tot*torch.cos(ang),tot*torch.sin(ang)


# # # # class Acc:
# # # #     def __init__(self):
# # # #         self.d=[]; self.a=[]; self.c=[]; self.sd=defaultdict(list); self._h={12:1,24:3,48:7,72:11}
# # # #     def update(self,dist,ate=None,cte=None):
# # # #         self.d.extend(dist.mean(0).tolist())
# # # #         for h,s in self._h.items():
# # # #             if s<dist.shape[0]: self.sd[h].extend(dist[s].tolist())
# # # #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# # # #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())
# # # #     def compute(self):
# # # #         r={"ADE":float(np.mean(self.d)) if self.d else float("nan"),
# # # #            "ATE_mean":float(np.mean(self.a)) if self.a else float("nan"),
# # # #            "CTE_mean":float(np.mean(self.c)) if self.c else float("nan"),
# # # #            "n":len(self.d)}
# # # #         for h in self._h:
# # # #             v=self.sd.get(h,[]); r[f"{h}h"]=float(np.mean(v)) if v else float("nan")
# # # #         return r


# # # # def _score(r):
# # # #     ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# # # #     ate=r.get("ATE_mean",1e9); cte=r.get("CTE_mean",1e9)
# # # #     if not np.isfinite(ate): ate=ade*.46
# # # #     if not np.isfinite(cte): cte=ade*.53
# # # #     return 100.*(0.05*(ade/136.)+0.10*(r.get("12h",ade)/50.)+0.15*(r.get("24h",ade)/100.)+
# # # #                  0.20*(r.get("48h",ade)/200.)+0.25*(h72/300.)+0.13*(ate/80.)+0.12*(cte/94.))

# # # # def _beat(r):
# # # #     p=[]
# # # #     for k,t in [("ADE",172.68),("ATE_mean",142.21),("CTE_mean",42.04),("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
# # # #         v=r.get(k,1e9)
# # # #         if np.isfinite(v) and v<t: p.append(f"{k.replace('_mean','')}✅{v:.1f}")
# # # #     return "🏆 BEAT: "+" ".join(p) if p else ""

# # # # def _gap(r):
# # # #     p=[]
# # # #     for k,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # #         v=r.get(k,float("nan"))
# # # #         if np.isfinite(v): p.append(f"{k.replace('_mean','')}:{v:.0f}({'↓' if v<ref else '↑'}{abs(v-ref):.0f})")
# # # #     return " | ".join(p)


# # # # @torch.no_grad()
# # # # def evaluate(model,loader,dev,tag="",ema=None,steps=10):
# # # #     bk=None
# # # #     if ema:
# # # #         try: bk=ema.apply_to(model)
# # # #         except: pass
# # # #     model.eval(); acc=Acc(); t0=time.perf_counter(); n=0
# # # #     for b in loader:
# # # #         bl=move(list(b),dev)
# # # #         p,_=model.sample(bl,ddim_steps=steps)
# # # #         g=bl[1]; T=min(p.shape[0],g.shape[0])
# # # #         pd=_ntd(p[:T]); gd=_ntd(g[:T])
# # # #         dist=_hav(pd,gd); at,ct=_atecte(pd,gd)
# # # #         acc.update(dist,at,ct); n+=1
# # # #     if bk:
# # # #         try: ema.restore(model,bk)
# # # #         except: pass
# # # #     r=acc.compute(); r["ms"]=(time.perf_counter()-t0)*1e3/max(n,1)
# # # #     def _v(k): return r.get(k,float("nan"))
# # # #     def _m(v,t): return "✅" if np.isfinite(v) and v<t else "❌"
# # # #     el=time.perf_counter()-t0
# # # #     print(f"\n{'='*68}")
# # # #     print(f"  [{tag}  {el:.0f}s  {n} batches]")
# # # #     print(f"  ADE={_v('ADE'):.1f}{_m(_v('ADE'),172.68)}  "
# # # #           f"12h={_v('12h'):.0f}{_m(_v('12h'),65.42)}  "
# # # #           f"24h={_v('24h'):.0f}{_m(_v('24h'),104.67)}  "
# # # #           f"48h={_v('48h'):.0f}{_m(_v('48h'),205.10)}  "
# # # #           f"72h={_v('72h'):.0f}{_m(_v('72h'),321.39)}")
# # # #     if np.isfinite(_v("ATE_mean")):
# # # #         print(f"  ATE={_v('ATE_mean'):.1f}{_m(_v('ATE_mean'),142.21)}  CTE={_v('CTE_mean'):.1f}{_m(_v('CTE_mean'),42.04)}")
# # # #     print(f"  vs ST-Trans: {_gap(r)}")
# # # #     bt=_beat(r)
# # # #     if bt: print(f"  {bt}")
# # # #     print(f"  Score={_score(r):.2f}")
# # # #     print(f"{'='*68}\n")
# # # #     return r


# # # # def _save(path,ep,model,opt,sched,saver,tl,vl,extra=None):
# # # #     m=_unwrap(model); ema=getattr(m,"_ema",None); esd=None
# # # #     if ema and hasattr(ema,"shadow"):
# # # #         try: esd={k:v.cpu().clone() for k,v in ema.shadow.items()}
# # # #         except: pass
# # # #     d={"epoch":ep,"model_state_dict":m.state_dict(),
# # # #        "optimizer_state":opt.state_dict(),"scheduler_state":sched.state_dict(),
# # # #        "ema_shadow":esd,"best_score":saver.bs,"best_ade":saver.ba,
# # # #        "best_72h":saver.b7,"best_ate":saver.bat,"best_cte":saver.bc,
# # # #        "train_loss":tl,"val_loss":vl}
# # # #     if extra: d.update(extra)
# # # #     torch.save(d,path)


# # # # class Saver:
# # # #     def __init__(self,patience=35):
# # # #         self.patience=patience; self.cnt=0; self.stop=False
# # # #         self.bs=self.ba=self.b7=self.bat=self.bc=float("inf")

# # # #     def update(self,r,model,out,ep,opt,sched,tl,vl,tag="",min_ep=30):
# # # #         sc=_score(r); ade=r.get("ADE",1e9); h72=r.get("72h",1e9)
# # # #         ate=r.get("ATE_mean",1e9); cte=r.get("CTE_mean",1e9)
# # # #         for v,a,fn in [(ade,"ba","best_ade.pth"),(h72,"b7","best_72h.pth"),
# # # #                        (ate,"bat","best_ate.pth"),(cte,"bc","best_cte.pth")]:
# # # #             if v<getattr(self,a):
# # # #                 setattr(self,a,v); _save(os.path.join(out,fn),ep,model,opt,sched,self,tl,vl)
# # # #         if sc<self.bs:
# # # #             self.bs=sc; self.cnt=0
# # # #             _save(os.path.join(out,f"best_{tag or 'composite'}.pth"),ep,model,opt,sched,self,tl,vl,
# # # #                   {"score":sc,"ade":ade,"h72":h72})
# # # #             print(f"  ✅ Best {tag} score={sc:.2f}  ADE={ade:.1f}  72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}")
# # # #         else:
# # # #             self.cnt+=1
# # # #             print(f"  No improve {self.cnt}/{self.patience}  (best={self.bs:.2f} cur={sc:.2f})")
# # # #         if ep>=min_ep and self.cnt>=self.patience: self.stop=True


# # # # def mksub(ds,n,bs,cf):
# # # #     idx=random.Random(42).sample(range(len(ds)),min(n,len(ds)))
# # # #     return DataLoader(Subset(ds,idx),batch_size=bs,shuffle=False,collate_fn=cf,num_workers=0,drop_last=False)


# # # # def get_args():
# # # #     p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # #     p.add_argument("--dataset_root",    default="TCND_vn")
# # # #     p.add_argument("--obs_len",         default=8,    type=int)
# # # #     p.add_argument("--pred_len",        default=12,   type=int)
# # # #     p.add_argument("--batch_size",      default=32,   type=int)
# # # #     p.add_argument("--num_epochs",      default=120,  type=int)
# # # #     p.add_argument("--g_learning_rate", default=1e-4, type=float)
# # # #     p.add_argument("--sel_lr_ratio",    default=0.10, type=float)
# # # #     p.add_argument("--weight_decay",    default=1e-3, type=float)
# # # #     p.add_argument("--warmup_epochs",   default=5,    type=int)
# # # #     p.add_argument("--grad_clip",       default=1.0,  type=float)
# # # #     p.add_argument("--patience",        default=35,   type=int)
# # # #     p.add_argument("--min_epochs",      default=30,   type=int)
# # # #     p.add_argument("--use_amp",         action="store_true")
# # # #     p.add_argument("--num_workers",     default=2,    type=int)
# # # #     p.add_argument("--sigma_min",       default=0.02, type=float)
# # # #     p.add_argument("--head_noise_base", default=0.03, type=float)
# # # #     p.add_argument("--use_ot",          default=True, action="store_true")
# # # #     p.add_argument("--no_ot",           dest="use_ot",action="store_false")
# # # #     p.add_argument("--cfg_guidance_scale",default=1.3,type=float)
# # # #     p.add_argument("--sel_warmup",      default=20,   type=int)
# # # #     p.add_argument("--val_freq",        default=3,    type=int)
# # # #     p.add_argument("--val_subset_size", default=500,  type=int)
# # # #     p.add_argument("--fast_ddim",       default=10,   type=int)
# # # #     p.add_argument("--full_ddim",       default=20,   type=int)
# # # #     p.add_argument("--use_ema",         default=True, action="store_true")
# # # #     p.add_argument("--no_ema",          dest="use_ema",action="store_false")
# # # #     p.add_argument("--ema_decay",       default=0.995,type=float)
# # # #     p.add_argument("--output_dir",      default="runs/v68")
# # # #     p.add_argument("--gpu_num",         default="0")
# # # #     p.add_argument("--delim",           default=" ")
# # # #     p.add_argument("--skip",            default=1,    type=int)
# # # #     p.add_argument("--min_ped",         default=1,    type=int)
# # # #     p.add_argument("--threshold",       default=0.002,type=float)
# # # #     p.add_argument("--other_modal",     default="gph")
# # # #     p.add_argument("--test_year",       default=None, type=int)
# # # #     p.add_argument("--resume",          default=None)
# # # #     p.add_argument("--resume_epoch",    default=None, type=int)
# # # #     p.add_argument("--eval_test_after_train",default=True,action="store_true")
# # # #     return p.parse_args()


# # # # def main(args):
# # # #     if torch.cuda.is_available(): os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_num)
# # # #     dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # #     os.makedirs(args.output_dir,exist_ok=True)
# # # #     SW=args.sel_warmup

# # # #     print("="*72)
# # # #     print(f"  TC-FlowMatching v68  —  Stable, LearnedStepWeights, 3-phase")
# # # #     print(f"  Phase 1 (ep 0–{SW-1}):    Generator+StepW, selector FROZEN")
# # # #     print(f"  Phase 2 (ep {SW}–{SW+14}): Selector, generator+encoder FROZEN")
# # # #     print(f"  Phase 3 (ep {SW+15}+):  Full joint")
# # # #     print(f"  Target: ADE<{TARGETS['ADE']}  72h<{TARGETS['72h']}  ATE<{TARGETS['ATE']}  CTE<{TARGETS['CTE']}")
# # # #     print("="*72)

# # # #     trd,trl=data_loader(args,{"root":args.dataset_root,"type":"train"},test=False)
# # # #     vd,vl=data_loader(args,{"root":args.dataset_root,"type":"val"},test=True)
# # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # #     vsub=mksub(vd,args.val_subset_size,args.batch_size,seq_collate)
# # # #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# # # #     model=TCFlowMatchingV68(
# # # #         pred_len=args.pred_len,obs_len=args.obs_len,
# # # #         sigma_min=args.sigma_min,use_ema=args.use_ema,ema_decay=args.ema_decay,
# # # #         use_ot=args.use_ot,head_noise_base=args.head_noise_base,
# # # #         cfg_guidance_scale=args.cfg_guidance_scale,selector_warmup=SW,
# # # #     ).to(dev)
# # # #     model.init_ema()
# # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # #     print(f"  params: {n_params:,}")

# # # #     opt=optim.AdamW([
# # # #         {"params":_gen_params(model),"lr":args.g_learning_rate,"name":"generator"},
# # # #         {"params":_sel_params(model),"lr":args.g_learning_rate*args.sel_lr_ratio,"name":"selector"},
# # # #     ],weight_decay=args.weight_decay)

# # # #     saver=Saver(patience=args.patience)
# # # #     scaler=GradScaler("cuda",enabled=args.use_amp)
# # # #     nstep=len(trl); total=nstep*args.num_epochs; wstp=nstep*args.warmup_epochs
# # # #     sched=get_cosine_schedule_with_warmup(opt,wstp,total,min_lr=1e-6)

# # # #     start=0
# # # #     if args.resume and os.path.exists(args.resume):
# # # #         print(f"  Loading: {args.resume}")
# # # #         ck=torch.load(args.resume,map_location=dev); m=_unwrap(model)
# # # #         ms,us=m.load_state_dict(ck["model_state_dict"],strict=False)
# # # #         if ms: print(f"  Missing: {len(ms)}"); 
# # # #         if us: print(f"  Unexpected: {len(us)}")
# # # #         ema=getattr(m,"_ema",None)
# # # #         if ema and ck.get("ema_shadow"):
# # # #             for k,v in ck["ema_shadow"].items():
# # # #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
# # # #         try: opt.load_state_dict(ck["optimizer_state"])
# # # #         except Exception as e: print(f"  Opt: {e}")
# # # #         try: sched.load_state_dict(ck["scheduler_state"])
# # # #         except Exception as e:
# # # #             for _ in range(ck.get("epoch",0)*nstep): sched.step()
# # # #         for a in ("best_score","best_ade","best_72h","best_ate","best_cte"):
# # # #             if a in ck: setattr(saver,{"best_score":"bs","best_ade":"ba","best_72h":"b7","best_ate":"bat","best_cte":"bc"}[a],ck[a])
# # # #         start=args.resume_epoch if args.resume_epoch else ck.get("epoch",0)+1
# # # #         print(f"  → Resume ep {start}")

# # # #     try:
# # # #         model=torch.compile(model,mode="reduce-overhead"); print("  torch.compile: enabled")
# # # #     except: pass

# # # #     oracle_ema=999.; _pa=.05
# # # #     prev_ph=-1; ts=time.perf_counter()
# # # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # # #     print("="*72)

# # # #     for ep in range(start,args.num_epochs):
# # # #         ph=_get_phase(ep,SW)

# # # #         # Phase transitions
# # # #         if ph==1 and prev_ph!=1:
# # # #             _freeze(_sel_params(model)); _unfreeze(_gen_params(model))
# # # #             print(f"  🔒 ep{ep}: selector FROZEN, generator active (Phase 1)")
# # # #         elif ph==2 and prev_ph!=2:
# # # #             _unfreeze(_sel_params(model)); _freeze(_gen_params(model))
# # # #             print(f"  🔀 ep{ep}: generator FROZEN, selector active (Phase 2)")
# # # #         elif ph==3 and prev_ph!=3:
# # # #             _unfreeze(_gen_params(model)); _unfreeze(_sel_params(model))
# # # #             print(f"  🔓 ep{ep}: all UNFROZEN (Phase 3)")

# # # #         # Adjust LR for frozen params
# # # #         for pg in opt.param_groups:
# # # #             if pg.get("name")=="selector":
# # # #                 pg["lr"]=0. if ph==1 else (args.g_learning_rate*args.sel_lr_ratio if ph==2 else args.g_learning_rate*.5)
# # # #             elif pg.get("name")=="generator":
# # # #                 pg["lr"]=0. if ph==2 else args.g_learning_rate  # cosine applies separately

# # # #         if ph!=prev_ph:
# # # #             print(f"\n  ┌── Phase {ph}/3  ep={ep}  oracle_ema={oracle_ema:.1f}km  tau={TCFlowMatchingV68._tau(ep):.0f}")
# # # #             prev_ph=ph

# # # #         # ── Train epoch ───────────────────────────────────────────
# # # #         model.train(); sl=0.; t0=time.perf_counter()
# # # #         for i,batch in enumerate(trl):
# # # #             bl=move(list(batch),dev)
# # # #             with autocast(device_type="cuda",enabled=args.use_amp):
# # # #                 bd=model.get_loss_breakdown(bl,epoch=ep)
# # # #             opt.zero_grad()
# # # #             scaler.scale(bd["total"]).backward()
# # # #             scaler.unscale_(opt)
# # # #             torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
# # # #             sb=scaler.get_scale(); scaler.step(opt); scaler.update()
# # # #             if scaler.get_scale()>=sb: sched.step()
# # # #             model.ema_update(); sl+=bd["total"].item()

# # # #             # oracle_ema từ km thực
# # # #             ok = bd.get("oracle_ade_km", oracle_ema)
# # # #             # convert to float safely and check finiteness
# # # #             try:
# # # #                 np_mod = float(ok)
# # # #             except Exception:
# # # #                 try:
# # # #                     np_mod = float(np.array(ok).item())
# # # #                 except Exception:
# # # #                     np_mod = None
# # # #             if np_mod is not None and np.isfinite(np_mod):
# # # #                 oracle_ema = (1 - _pa) * oracle_ema + _pa * np_mod

# # # #             if i%20==0:
# # # #                 lg=next((pg["lr"] for pg in opt.param_groups if pg.get("name")=="generator"),0.)
# # # #                 ls=next((pg["lr"] for pg in opt.param_groups if pg.get("name")=="selector"),0.)
# # # #                 icon="🔒" if ph==1 else ("🔀" if ph==2 else "🔓")
# # # #                 print(
# # # #                     f"  [P{ph}][{ep:>3}][{i:>3}/{nstep}]"
# # # #                     f"  tot={bd['total'].item():.3f}"
# # # #                     f"  fm={bd.get('L_FM',0):.4f}"
# # # #                     f"  dpe={bd.get('L_dpe',0):.4f}"
# # # #                     f"  rank={bd.get('L_rank',0):.4f}"
# # # #                     f"  div={bd.get('min_dist_mean',0):.3f}"
# # # #                     f"  oracle={bd.get('oracle_ade_km',0):.1f}km"
# # # #                     f"  sw12={bd.get('sw_12h',0):.2f}"
# # # #                     f"  sw72={bd.get('sw_72h',0):.2f}"
# # # #                     f"  δ={bd.get('delta_mean',0):.2f}"
# # # #                     f"  {icon}  lg={lg:.2e}  ls={ls:.2e}"
# # # #                 )

# # # #         eps=time.perf_counter()-t0; avt=sl/nstep

# # # #         # Val loss
# # # #         model.eval(); vls=0.
# # # #         with torch.no_grad():
# # # #             for batch in vl:
# # # #                 bv=move(list(batch),dev)
# # # #                 with autocast(device_type="cuda",enabled=args.use_amp):
# # # #                     vls+=model.get_loss(bv,epoch=ep).item()
# # # #         avv=vls/len(vl)
# # # #         print(f"  Epoch {ep:>3}  P{ph}  train={avt:.4f}  val={avv:.4f}  {eps:.0f}s  oracle_ema={oracle_ema:.1f}km")

# # # #         # Fast eval
# # # #         rf=evaluate(model,vsub,dev,tag=f"FAST/P{ph} ep{ep}",steps=args.fast_ddim)
# # # #         saver.update(rf,model,args.output_dir,ep,opt,sched,avt,avv,tag="fast")

# # # #         # Full val every val_freq
# # # #         if ep%args.val_freq==0:
# # # #             em=getattr(_unwrap(model),"_ema",None)
# # # #             rr=evaluate(model,vl,dev,tag=f"RAW/P{ph} ep{ep}",steps=args.full_ddim)
# # # #             saver.update(rr,model,args.output_dir,ep,opt,sched,avt,avv,tag="raw")
# # # #             if em and ep>=5:
# # # #                 re=evaluate(model,vl,dev,tag=f"EMA/P{ph} ep{ep}",ema=em,steps=args.full_ddim)
# # # #                 saver.update(re,model,args.output_dir,ep,opt,sched,avt,avv,tag="ema")

# # # #         if ep%10==0 or ep==args.num_epochs-1:
# # # #             _save(os.path.join(args.output_dir,f"ckpt_ep{ep:03d}.pth"),ep,model,opt,sched,saver,avt,avv,
# # # #                   {"oracle_ema":oracle_ema,"phase":ph})

# # # #         if saver.stop: print(f"  ⛔ Early stop ep{ep}"); break

# # # #     th=(time.perf_counter()-ts)/3600
# # # #     print(f"\n  Best: ADE={saver.ba:.1f}  72h={saver.b7:.0f}  ATE={saver.bat:.1f}  CTE={saver.bc:.1f}  ({th:.2f}h)")

# # # #     if args.eval_test_after_train:
# # # #         print("\n"+"="*72+"  POST-TRAINING TEST\n"+"="*72)
# # # #         try: _,tl2=data_loader(args,{"root":args.dataset_root,"type":"test"},test=True)
# # # #         except: print("  ⚠ No test set, using val"); tl2=vl
# # # #         for fn,lb in [("best_composite.pth","COMPOSITE"),("best_72h.pth","72h"),
# # # #                        ("best_cte.pth","CTE"),("best_ema.pth","EMA")]:
# # # #             pp=os.path.join(args.output_dir,fn)
# # # #             if not os.path.exists(pp): continue
# # # #             ck=torch.load(pp,map_location=dev); _unwrap(model).load_state_dict(ck["model_state_dict"],strict=False)
# # # #             em=getattr(_unwrap(model),"_ema",None)
# # # #             if em and ck.get("ema_shadow"):
# # # #                 for k,v in ck["ema_shadow"].items():
# # # #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
# # # #             r=evaluate(model,tl2,dev,tag=f"TEST/{lb}",steps=args.full_ddim)
# # # #             print(f"\n  ── {lb} TEST ──")
# # # #             for key,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
# # # #                 v=r.get(key,float("nan"))
# # # #                 mk="✅ BEAT!" if np.isfinite(v) and v<ref else f"❌ target:{ref:.0f}"
# # # #                 print(f"    {key:<10}: {v:>8.1f} km  {mk}  [gap:{v-ref:+.1f}]")
# # # #     print("="*72)


# # # # if __name__=="__main__":
# # # #     args=get_args()
# # # #     import numpy as np
# # # #     np.random.seed(42); torch.manual_seed(42)
# # # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # # #     main(args)

# # # """
# # # train_v72.py — TC-FlowMatching v72
# # # ════════════════════════════════════════════════════════════════════════
# # # 3-phase training với freeze/unfreeze chính xác:
# # #   Phase 1 (0..SEL_WARMUP-1):       Generator + step_weights + loss_weights trains
# # #                                     Selector FROZEN, selector LR = 0
# # #   Phase 2 (SEL_WARMUP..+14):       Selector trains
# # #                                     Generator NOT fully frozen (BUG-2 fix)
# # #                                     generator LR giảm xuống 0.05x (không 0)
# # #   Phase 3 (SEL_WARMUP+15..end):    Joint, tất cả unfrozen, cân bằng LR

# # # Key fixes so với train_v68:
# # #   - _gen_params bao gồm step_weights + loss_weights_p1 + loss_weights_p23
# # #   - Phase 2: generator LR = 0.05x (không phải 0) → BUG-2 fix
# # #   - Log sw_ratio, lw_dpe, lw_fm để monitor stability
# # #   - Red flag alerts: sw_ratio < 1.0 hoặc L_div constant

# # # STABILITY MONITORS (xem trong log):
# # #   sw_ratio >= 1.0     → step weights không flip (nếu < 1.0: dừng debug ngay)
# # #   L_div > 0.05        → diversity đang train (nếu = 0.4 constant: Bug-3 chưa fix)
# # #   oracle_ade_km ↓     → oracle selection đúng (nếu tăng: Bug-1 chưa fix)
# # #   lw_dpe/lw_fm ∈ 1-4  → DPE/FM balance hợp lý

# # # HOW TO RUN:
# # #   python scripts/train_v72.py \\
# # #       --dataset_root /kaggle/input/.../tc-ofm \\
# # #       --output_dir   runs/v72 \\
# # #       --batch_size   32 \\
# # #       --num_epochs   120 \\
# # #       --sel_warmup   20 \\
# # #       --use_amp
# # # """
# # # from __future__ import annotations

# # # import sys, os
# # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # import argparse, time, random
# # # from collections import defaultdict

# # # import numpy as np
# # # import torch
# # # import torch.optim as optim
# # # from torch.amp import autocast, GradScaler
# # # from torch.utils.data import DataLoader, Subset

# # # from Model.data.loader_training import data_loader
# # # from Model.flow_matching_model import TCFlowMatchingV72

# # # try:
# # #     from Model.utils import get_cosine_schedule_with_warmup
# # # except ImportError:
# # #     from torch.optim.lr_scheduler import CosineAnnealingLR
# # #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# # #         return CosineAnnealingLR(opt, T_max=total_steps, eta_min=min_lr)

# # # TARGETS = {
# # #     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
# # #     "12h": 65.42,  "24h": 104.67, "48h": 205.10
# # # }
# # # R_EARTH = 6371.0


# # # # ── Helper functions ──────────────────────────────────────────────────────────

# # # def _unwrap(m):
# # #     return m._orig_mod if hasattr(m, '_orig_mod') else m


# # # def _sel_params(m):
# # #     """Selector parameters."""
# # #     return list(_unwrap(m).selector.parameters())


# # # def _gen_params(m):
# # #     """
# # #     Generator parameters: encoder + velocity_heads + learnable weights.
# # #     V71 thêm: step_weights, loss_weights_p1, loss_weights_p23
# # #     Đây là những params cần đưa vào optimizer group "generator".
# # #     """
# # #     u = _unwrap(m)
# # #     params = (list(u.encoder.parameters()) +
# # #               list(u.velocity_heads.parameters()) +
# # #               list(u.step_weights.parameters()) +
# # #               list(u.loss_weights_p1.parameters()) +
# # #               list(u.loss_weights_p23.parameters()))
# # #     return params


# # # def _freeze(params):
# # #     for p in params: p.requires_grad_(False)


# # # def _unfreeze(params):
# # #     for p in params: p.requires_grad_(True)


# # # def _sel_frozen(m):
# # #     p = _sel_params(m)
# # #     return len(p) > 0 and not p[0].requires_grad


# # # def _get_phase(ep: int, sw: int) -> int:
# # #     if ep < sw: return 1
# # #     elif ep < sw + 15: return 2
# # #     return 3


# # # def move(b, dev):
# # #     out = list(b)
# # #     for i, x in enumerate(out):
# # #         if torch.is_tensor(x):
# # #             out[i] = x.to(dev)
# # #         elif isinstance(x, dict):
# # #             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v
# # #                        for k, v in x.items()}
# # #     return out


# # # # ── Metric utils ──────────────────────────────────────────────────────────────

# # # def _ntd(a: torch.Tensor) -> torch.Tensor:
# # #     o = a.clone()
# # #     o[..., 0] = (a[..., 0]*50. + 1800.) / 10.
# # #     o[..., 1] = (a[..., 1]*50.) / 10.
# # #     return o


# # # def _hav(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
# # #     la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
# # #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# # #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# # #     a = (torch.sin(dlat/2).pow(2) +
# # #          torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
# # #     return 2. * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())


# # # def _atecte(pd: torch.Tensor, gd: torch.Tensor):
# # #     T = min(pd.shape[0], gd.shape[0])
# # #     if T < 2:
# # #         z = pd.new_zeros(1, pd.shape[1]); return z, z
# # #     lo1 = torch.deg2rad(gd[:T-1,:,0]); la1 = torch.deg2rad(gd[:T-1,:,1])
# # #     lo2 = torch.deg2rad(gd[1:T, :,0]); la2 = torch.deg2rad(gd[1:T, :,1])
# # #     lo3 = torch.deg2rad(pd[1:T, :,0]); la3 = torch.deg2rad(pd[1:T, :,1])
# # #     ya  = torch.sin(lo2-lo1)*torch.cos(la2)
# # #     xa  = torch.cos(la1)*torch.sin(la2) - torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # #     be  = torch.atan2(ya, xa)
# # #     ye  = torch.sin(lo3-lo2)*torch.cos(la3)
# # #     xe  = torch.cos(la2)*torch.sin(la3) - torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # #     bee = torch.atan2(ye, xe)
# # #     tot = _hav(pd[1:T], gd[1:T])
# # #     ang = bee - be
# # #     return tot * torch.cos(ang), tot * torch.sin(ang)


# # # class Acc:
# # #     def __init__(self):
# # #         self.d = []; self.a = []; self.c = []
# # #         self.sd = defaultdict(list)
# # #         self._h = {12: 1, 24: 3, 48: 7, 72: 11}

# # #     def update(self, dist, ate=None, cte=None):
# # #         self.d.extend(dist.mean(0).tolist())
# # #         for h, s in self._h.items():
# # #             if s < dist.shape[0]: self.sd[h].extend(dist[s].tolist())
# # #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# # #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())

# # #     def compute(self) -> dict:
# # #         r = {
# # #             "ADE":      float(np.mean(self.d))  if self.d else float("nan"),
# # #             "ATE_mean": float(np.mean(self.a))  if self.a else float("nan"),
# # #             "CTE_mean": float(np.mean(self.c))  if self.c else float("nan"),
# # #             "n":        len(self.d)
# # #         }
# # #         for h in self._h:
# # #             v = self.sd.get(h, [])
# # #             r[f"{h}h"] = float(np.mean(v)) if v else float("nan")
# # #         return r


# # # def _score(r: dict) -> float:
# # #     ade  = r.get("ADE",   1e9); h72 = r.get("72h",   1e9)
# # #     ate  = r.get("ATE_mean", 1e9); cte = r.get("CTE_mean", 1e9)
# # #     if not np.isfinite(ate): ate = ade * .46
# # #     if not np.isfinite(cte): cte = ade * .53
# # #     return 100. * (
# # #         0.05 * (ade / 136.) +
# # #         0.10 * (r.get("12h", ade) / 50.) +
# # #         0.15 * (r.get("24h", ade) / 100.) +
# # #         0.20 * (r.get("48h", ade) / 200.) +
# # #         0.25 * (h72 / 300.) +
# # #         0.13 * (ate / 80.) +
# # #         0.12 * (cte / 94.)
# # #     )


# # # def _beat(r: dict) -> str:
# # #     p = []
# # #     for k, t in [("ADE", 172.68), ("ATE_mean", 142.21), ("CTE_mean", 42.04),
# # #                   ("72h", 321.39), ("12h", 65.42), ("24h", 104.67), ("48h", 205.10)]:
# # #         v = r.get(k, 1e9)
# # #         if np.isfinite(v) and v < t:
# # #             p.append(f"{k.replace('_mean','')}:{v:.1f}")
# # #     return "BEAT: " + " ".join(p) if p else ""


# # # def _gap(r: dict) -> str:
# # #     p = []
# # #     for k, ref in [("ADE", 172.68), ("72h", 321.39),
# # #                     ("ATE_mean", 142.21), ("CTE_mean", 42.04)]:
# # #         v = r.get(k, float("nan"))
# # #         if np.isfinite(v):
# # #             p.append(f"{k.replace('_mean','')}:{v:.0f}"
# # #                      f"({'down' if v < ref else 'up'}{abs(v-ref):.0f})")
# # #     return " | ".join(p)


# # # # ── Evaluation ────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def evaluate(model, loader, dev, tag: str = "",
# # #              ema=None, steps: int = 10) -> dict:
# # #     bk = None
# # #     if ema:
# # #         try: bk = ema.apply_to(model)
# # #         except: pass
# # #     model.eval(); acc = Acc(); t0 = time.perf_counter(); n = 0
# # #     for b in loader:
# # #         bl = move(list(b), dev)
# # #         p, _ = model.sample(bl, ddim_steps=steps)
# # #         g = bl[1]; T = min(p.shape[0], g.shape[0])
# # #         pd = _ntd(p[:T]); gd = _ntd(g[:T])
# # #         dist = _hav(pd, gd); at, ct = _atecte(pd, gd)
# # #         acc.update(dist, at, ct); n += 1
# # #     if bk:
# # #         try: ema.restore(model, bk)
# # #         except: pass
# # #     r = acc.compute(); r["ms"] = (time.perf_counter()-t0)*1e3 / max(n, 1)

# # #     def _v(k): return r.get(k, float("nan"))
# # #     def _m(v, t): return "ok" if np.isfinite(v) and v < t else "no"

# # #     el = time.perf_counter() - t0
# # #     print(f"\n{'='*68}")
# # #     print(f"  [{tag}  {el:.0f}s  {n} batches]")
# # #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]  "
# # #           f"12h={_v('12h'):.0f}[{_m(_v('12h'),65.42)}]  "
# # #           f"24h={_v('24h'):.0f}[{_m(_v('24h'),104.67)}]  "
# # #           f"48h={_v('48h'):.0f}[{_m(_v('48h'),205.10)}]  "
# # #           f"72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
# # #     if np.isfinite(_v("ATE_mean")):
# # #         print(f"  ATE={_v('ATE_mean'):.1f}[{_m(_v('ATE_mean'),142.21)}]  "
# # #               f"CTE={_v('CTE_mean'):.1f}[{_m(_v('CTE_mean'),42.04)}]")
# # #     print(f"  vs ST-Trans: {_gap(r)}")
# # #     bt = _beat(r)
# # #     if bt: print(f"  *** {bt} ***")
# # #     print(f"  Score={_score(r):.2f}")
# # #     print(f"{'='*68}\n")
# # #     return r


# # # # ── Checkpoint ────────────────────────────────────────────────────────────────

# # # def _save(path, ep, model, opt, sched, saver, tl, vl, extra=None):
# # #     m = _unwrap(model); ema = getattr(m, "_ema", None); esd = None
# # #     if ema and hasattr(ema, "shadow"):
# # #         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # #         except: pass
# # #     d = {
# # #         "epoch": ep,
# # #         "model_state_dict": m.state_dict(),
# # #         "optimizer_state": opt.state_dict(),
# # #         "scheduler_state": sched.state_dict(),
# # #         "ema_shadow": esd,
# # #         "best_score": saver.bs,
# # #         "best_ade":   saver.ba,
# # #         "best_72h":   saver.b7,
# # #         "best_ate":   saver.bat,
# # #         "best_cte":   saver.bc,
# # #         "train_loss": tl,
# # #         "val_loss":   vl,
# # #     }
# # #     if extra: d.update(extra)
# # #     torch.save(d, path)


# # # class Saver:
# # #     def __init__(self, patience: int = 35):
# # #         self.patience = patience; self.cnt = 0; self.stop = False
# # #         self.bs = self.ba = self.b7 = self.bat = self.bc = float("inf")

# # #     def update(self, r, model, out, ep, opt, sched, tl, vl,
# # #                tag: str = "", min_ep: int = 30):
# # #         sc  = _score(r)
# # #         ade = r.get("ADE",   1e9); h72 = r.get("72h",   1e9)
# # #         ate = r.get("ATE_mean", 1e9); cte = r.get("CTE_mean", 1e9)

# # #         for v, a, fn in [(ade, "ba", "best_ade.pth"),
# # #                           (h72, "b7", "best_72h.pth"),
# # #                           (ate, "bat","best_ate.pth"),
# # #                           (cte, "bc", "best_cte.pth")]:
# # #             if v < getattr(self, a):
# # #                 setattr(self, a, v)
# # #                 _save(os.path.join(out, fn), ep, model, opt, sched, self, tl, vl)

# # #         if sc < self.bs:
# # #             self.bs = sc; self.cnt = 0
# # #             _save(os.path.join(out, f"best_{tag or 'composite'}.pth"),
# # #                   ep, model, opt, sched, self, tl, vl,
# # #                   {"score": sc, "ade": ade, "h72": h72})
# # #             print(f"  Best {tag} score={sc:.2f}  ADE={ade:.1f}  "
# # #                   f"72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}")
# # #         else:
# # #             self.cnt += 1
# # #             print(f"  No improve {self.cnt}/{self.patience}  "
# # #                   f"(best={self.bs:.2f} cur={sc:.2f})")

# # #         if ep >= min_ep and self.cnt >= self.patience:
# # #             self.stop = True


# # # def mksub(ds, n: int, bs: int, cf):
# # #     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
# # #     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
# # #                        collate_fn=cf, num_workers=0, drop_last=False)


# # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # def get_args():
# # #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # #     p.add_argument("--dataset_root",       default="TCND_vn")
# # #     p.add_argument("--obs_len",            default=8,    type=int)
# # #     p.add_argument("--pred_len",           default=12,   type=int)
# # #     p.add_argument("--batch_size",         default=32,   type=int)
# # #     p.add_argument("--num_epochs",         default=120,  type=int)
# # #     p.add_argument("--g_learning_rate",    default=1e-4, type=float)
# # #     p.add_argument("--sel_lr_ratio",       default=0.10, type=float)
# # #     # Phase 2 generator LR ratio (BUG-2 fix: không phải 0)
# # #     p.add_argument("--gen_lr_ph2_ratio",   default=0.05, type=float,
# # #                     help="Generator LR multiplier in Phase 2 (default 0.05 = 5%% of g_lr)")
# # #     p.add_argument("--weight_decay",       default=1e-3, type=float)
# # #     p.add_argument("--warmup_epochs",      default=5,    type=int)
# # #     p.add_argument("--grad_clip",          default=1.0,  type=float)
# # #     p.add_argument("--patience",           default=35,   type=int)
# # #     p.add_argument("--min_epochs",         default=30,   type=int)
# # #     p.add_argument("--use_amp",            action="store_true")
# # #     p.add_argument("--num_workers",        default=2,    type=int)
# # #     p.add_argument("--sigma_min",          default=0.02, type=float)
# # #     p.add_argument("--head_noise_base",    default=0.03, type=float)
# # #     p.add_argument("--use_ot",             default=True, action="store_true")
# # #     p.add_argument("--no_ot",              dest="use_ot", action="store_false")
# # #     p.add_argument("--cfg_guidance_scale", default=1.3,  type=float)
# # #     p.add_argument("--sel_warmup",         default=20,   type=int)
# # #     p.add_argument("--val_freq",           default=3,    type=int)
# # #     p.add_argument("--val_subset_size",    default=500,  type=int)
# # #     p.add_argument("--fast_ddim",          default=10,   type=int)
# # #     p.add_argument("--full_ddim",          default=20,   type=int)
# # #     p.add_argument("--use_ema",            default=True, action="store_true")
# # #     p.add_argument("--no_ema",             dest="use_ema", action="store_false")
# # #     p.add_argument("--ema_decay",          default=0.995, type=float)
# # #     p.add_argument("--output_dir",         default="runs/v72")
# # #     p.add_argument("--gpu_num",            default="0")
# # #     p.add_argument("--delim",              default=" ")
# # #     p.add_argument("--skip",               default=1,    type=int)
# # #     p.add_argument("--min_ped",            default=1,    type=int)
# # #     p.add_argument("--threshold",          default=0.002, type=float)
# # #     p.add_argument("--other_modal",        default="gph")
# # #     p.add_argument("--test_year",          default=None, type=int)
# # #     p.add_argument("--resume",             default=None)
# # #     p.add_argument("--resume_epoch",       default=None, type=int)
# # #     p.add_argument("--eval_test_after_train", default=True, action="store_true")
# # #     return p.parse_args()


# # # # ── Main ──────────────────────────────────────────────────────────────────────

# # # def main(args):
# # #     if torch.cuda.is_available():
# # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     os.makedirs(args.output_dir, exist_ok=True)
# # #     SW = args.sel_warmup

# # #     print("=" * 72)
# # #     print(f"  TC-FlowMatching v72  —  9-bug-fix, ConstrainedWeights")
# # #     print(f"  Phase 1 (ep 0-{SW-1}):      Generator+StepW+LossW, selector FROZEN")
# # #     print(f"  Phase 2 (ep {SW}-{SW+14}):  Selector trains, generator LR={args.gen_lr_ph2_ratio:.0%}")
# # #     print(f"  Phase 3 (ep {SW+15}+):  Full joint training")
# # #     print(f"  Target: ADE<{TARGETS['ADE']}  72h<{TARGETS['72h']}  "
# # #           f"ATE<{TARGETS['ATE']}  CTE<{TARGETS['CTE']}")
# # #     print(f"  Stability: sw_ratio>=1.0, L_div>0.05, oracle_km down, lw_dpe/fm in 1-4")
# # #     print("=" * 72)

# # #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
# # #     vd,  vl  = data_loader(args, {"root": args.dataset_root, "type": "val"},   test=True)
# # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # #     vsub = mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
# # #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# # #     model = TCFlowMatchingV72(
# # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # #         sigma_min=args.sigma_min, use_ema=args.use_ema, ema_decay=args.ema_decay,
# # #         use_ot=args.use_ot, head_noise_base=args.head_noise_base,
# # #         cfg_guidance_scale=args.cfg_guidance_scale, selector_warmup=SW,
# # #     ).to(dev)
# # #     model.init_ema()

# # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # #     print(f"  params: {n_params:,}")

# # #     # ── Optimizer ─────────────────────────────────────────────────────────────
# # #     # Generator group bao gồm step_weights và loss_weights (V71 addition)
# # #     opt = optim.AdamW([
# # #         {"params": _gen_params(model), "lr": args.g_learning_rate, "name": "generator"},
# # #         {"params": _sel_params(model), "lr": args.g_learning_rate * args.sel_lr_ratio,
# # #           "name": "selector"},
# # #     ], weight_decay=args.weight_decay)

# # #     saver  = Saver(patience=args.patience)
# # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # #     nstep  = len(trl)
# # #     total  = nstep * args.num_epochs
# # #     wstp   = nstep * args.warmup_epochs
# # #     sched  = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)

# # #     # ── Resume ────────────────────────────────────────────────────────────────
# # #     start = 0
# # #     if args.resume and os.path.exists(args.resume):
# # #         print(f"  Loading: {args.resume}")
# # #         ck = torch.load(args.resume, map_location=dev)
# # #         m  = _unwrap(model)
# # #         ms, us = m.load_state_dict(ck["model_state_dict"], strict=False)
# # #         if ms: print(f"  Missing keys: {len(ms)}")
# # #         if us: print(f"  Unexpected keys: {len(us)}")
# # #         ema = getattr(m, "_ema", None)
# # #         if ema and ck.get("ema_shadow"):
# # #             for k, v in ck["ema_shadow"].items():
# # #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
# # #         try:
# # #             opt.load_state_dict(ck["optimizer_state"])
# # #         except Exception as e:
# # #             print(f"  Opt state not loaded: {e}")
# # #         try:
# # #             sched.load_state_dict(ck["scheduler_state"])
# # #         except Exception as e:
# # #             for _ in range(ck.get("epoch", 0) * nstep): sched.step()
# # #         for a in ("best_score", "best_ade", "best_72h", "best_ate", "best_cte"):
# # #             key_map = {"best_score":"bs","best_ade":"ba","best_72h":"b7",
# # #                         "best_ate":"bat","best_cte":"bc"}
# # #             if a in ck: setattr(saver, key_map[a], ck[a])
# # #         start = args.resume_epoch if args.resume_epoch else ck.get("epoch", 0) + 1
# # #         print(f"  Resume from epoch {start}")

# # #     # ── Compile ───────────────────────────────────────────────────────────────
# # #     try:
# # #         model = torch.compile(model, mode="reduce-overhead")
# # #         print("  torch.compile: enabled")
# # #     except:
# # #         pass

# # #     # ── Training loop ─────────────────────────────────────────────────────────
# # #     oracle_ema = 999.;  _pa = .05
# # #     prev_ph = -1; ts = time.perf_counter()
# # #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# # #     print("=" * 72)

# # #     for ep in range(start, args.num_epochs):
# # #         ph = _get_phase(ep, SW)

# # #         # ── Phase transitions ─────────────────────────────────────────────────
# # #         if ph == 1 and prev_ph != 1:
# # #             # Phase 1: selector frozen, generator active
# # #             _freeze(_sel_params(model))
# # #             _unfreeze(_gen_params(model))
# # #             print(f"  [PHASE 1] ep{ep}: selector FROZEN, generator+weights ACTIVE")

# # #         elif ph == 2 and prev_ph != 2:
# # #             # Phase 2: selector unfrozen
# # #             # BUG-2 FIX: generator NOT frozen — just lower LR
# # #             # _freeze(_gen_params(model)) ← REMOVED (was causing Bug-2)
# # #             _unfreeze(_sel_params(model))
# # #             _unfreeze(_gen_params(model))  # generator stays trainable
# # #             print(f"  [PHASE 2] ep{ep}: selector ACTIVE, generator LR={args.gen_lr_ph2_ratio:.0%}")

# # #         elif ph == 3 and prev_ph != 3:
# # #             # Phase 3: all unfrozen
# # #             _unfreeze(_gen_params(model))
# # #             _unfreeze(_sel_params(model))
# # #             print(f"  [PHASE 3] ep{ep}: ALL UNFROZEN (joint fine-tuning)")

# # #         # ── LR adjustment per phase ────────────────────────────────────────────
# # #         # BUG-2 FIX: Phase 2 generator LR = gen_lr_ph2_ratio * g_lr (não phải 0)
# # #         for pg in opt.param_groups:
# # #             if pg.get("name") == "selector":
# # #                 pg["lr"] = (0. if ph == 1
# # #                              else args.g_learning_rate * args.sel_lr_ratio if ph == 2
# # #                              else args.g_learning_rate * 0.5)
# # #             elif pg.get("name") == "generator":
# # #                 pg["lr"] = (args.g_learning_rate                    if ph == 1
# # #                              else args.g_learning_rate * args.gen_lr_ph2_ratio  if ph == 2
# # #                              else args.g_learning_rate * 0.8)   # phase 3: slightly lower

# # #         if ph != prev_ph:
# # #             print(f"\n  Phase {ph}/3  ep={ep}  oracle_ema={oracle_ema:.1f}km  "
# # #                   f"tau={TCFlowMatchingV72._tau(ep):.0f}")
# # #             prev_ph = ph

# # #         # ── Train epoch ───────────────────────────────────────────────────────
# # #         model.train()
# # #         sl = 0.; t0 = time.perf_counter()

# # #         for i, batch in enumerate(trl):
# # #             bl = move(list(batch), dev)
# # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # #                 bd = model.get_loss_breakdown(bl, epoch=ep)

# # #             opt.zero_grad()
# # #             scaler.scale(bd["total"]).backward()
# # #             scaler.unscale_(opt)
# # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # #             sb = scaler.get_scale()
# # #             scaler.step(opt); scaler.update()
# # #             if scaler.get_scale() >= sb: sched.step()
# # #             model.ema_update()
# # #             sl += bd["total"].item()

# # #             # Oracle EMA từ clean predictions (BUG-9 fix)
# # #             ok = bd.get("oracle_ade_km", oracle_ema)
# # #             try:
# # #                 np_mod = float(ok)
# # #             except Exception:
# # #                 try:   np_mod = float(np.array(ok).item())
# # #                 except: np_mod = None
# # #             if np_mod is not None and np.isfinite(np_mod):
# # #                 oracle_ema = (1 - _pa) * oracle_ema + _pa * np_mod

# # #             # ── Step logging ──────────────────────────────────────────────────
# # #             if i % 20 == 0:
# # #                 lg   = next((pg["lr"] for pg in opt.param_groups
# # #                               if pg.get("name") == "generator"), 0.)
# # #                 ls   = next((pg["lr"] for pg in opt.param_groups
# # #                               if pg.get("name") == "selector"), 0.)
# # #                 icon = {"1": "P1", "2": "P2", "3": "P3"}.get(str(ph), "??")

# # #                 # Stability check: sw_ratio
# # #                 sw_ratio = bd.get('sw_ratio', 1.0)
# # #                 sw_warn  = " [WARN:sw_ratio<1!]" if sw_ratio < 1.0 else ""

# # #                 # Stability check: L_div (nên > 0.05 từ ep5+)
# # #                 ldiv = bd.get('L_div', 0.)
# # #                 div_warn = " [WARN:div=0?]" if ep > 5 and ldiv < 0.01 else ""

# # #                 print(
# # #                     f"  [{icon}][{ep:>3}][{i:>3}/{nstep}]"
# # #                     f"  tot={bd['total'].item():.3f}"
# # #                     f"  fm={bd.get('L_FM', 0):.4f}"
# # #                     f"  dpe={bd.get('L_dpe', 0):.4f}"
# # #                     f"  rank={bd.get('L_rank', 0):.4f}"
# # #                     f"  div={ldiv:.3f}"
# # #                     f"  oracle={bd.get('oracle_ade_km', 0):.1f}km"
# # #                     f"  sw24={bd.get('sw_24h', 0):.2f}"
# # #                     f"  sw72={bd.get('sw_72h', 0):.2f}"
# # #                     f"  swr={sw_ratio:.2f}{sw_warn}"
# # #                     f"  dpe/fm={bd.get('lw_dpe_fm_ratio', 0):.2f}"
# # #                     f"  d={bd.get('delta_mean', 0):.2f}"
# # #                     f"{div_warn}"
# # #                     f"  lg={lg:.2e}  ls={ls:.2e}"
# # #                 )

# # #         eps = time.perf_counter() - t0
# # #         avt = sl / nstep

# # #         # ── Val loss ──────────────────────────────────────────────────────────
# # #         model.eval(); vls = 0.
# # #         with torch.no_grad():
# # #             for batch in vl:
# # #                 bv  = move(list(batch), dev)
# # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # #                     vls += model.get_loss(bv, epoch=ep).item()
# # #         avv = vls / len(vl)
# # #         print(f"  Epoch {ep:>3}  P{ph}  train={avt:.4f}  val={avv:.4f}  "
# # #               f"{eps:.0f}s  oracle_ema={oracle_ema:.1f}km")

# # #         # ── Fast eval (subset) ────────────────────────────────────────────────
# # #         rf = evaluate(model, vsub, dev,
# # #                        tag=f"FAST/P{ph} ep{ep}", steps=args.fast_ddim)
# # #         saver.update(rf, model, args.output_dir, ep, opt, sched, avt, avv, tag="fast")

# # #         # ── Full val every val_freq epochs ────────────────────────────────────
# # #         if ep % args.val_freq == 0:
# # #             em = getattr(_unwrap(model), "_ema", None)

# # #             rr = evaluate(model, vl, dev,
# # #                            tag=f"RAW/P{ph} ep{ep}", steps=args.full_ddim)
# # #             saver.update(rr, model, args.output_dir, ep, opt, sched, avt, avv, tag="raw")

# # #             if em and ep >= 5:
# # #                 re = evaluate(model, vl, dev,
# # #                                tag=f"EMA/P{ph} ep{ep}", ema=em, steps=args.full_ddim)
# # #                 saver.update(re, model, args.output_dir, ep, opt, sched, avt, avv, tag="ema")

# # #         # ── Periodic checkpoint ───────────────────────────────────────────────
# # #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# # #             _save(os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# # #                   ep, model, opt, sched, saver, avt, avv,
# # #                   {"oracle_ema": oracle_ema, "phase": ph})

# # #         if saver.stop:
# # #             print(f"  Early stop ep{ep}"); break

# # #     th = (time.perf_counter() - ts) / 3600.
# # #     print(f"\n  Best: ADE={saver.ba:.1f}  72h={saver.b7:.0f}  "
# # #           f"ATE={saver.bat:.1f}  CTE={saver.bc:.1f}  ({th:.2f}h)")

# # #     # ── Post-training test ────────────────────────────────────────────────────
# # #     if args.eval_test_after_train:
# # #         print("\n" + "="*72)
# # #         print("  POST-TRAINING TEST")
# # #         print("="*72)
# # #         try:
# # #             _, tl2 = data_loader(args, {"root": args.dataset_root, "type": "test"},
# # #                                    test=True)
# # #         except:
# # #             print("  No test set, using val"); tl2 = vl

# # #         for fn, lb in [("best_composite.pth", "COMPOSITE"),
# # #                          ("best_72h.pth", "72H"),
# # #                          ("best_cte.pth", "CTE"),
# # #                          ("best_ema.pth", "EMA")]:
# # #             pp = os.path.join(args.output_dir, fn)
# # #             if not os.path.exists(pp): continue
# # #             ck = torch.load(pp, map_location=dev)
# # #             _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
# # #             em = getattr(_unwrap(model), "_ema", None)
# # #             if em and ck.get("ema_shadow"):
# # #                 for k, v in ck["ema_shadow"].items():
# # #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
# # #             r = evaluate(model, tl2, dev, tag=f"TEST/{lb}", steps=args.full_ddim)
# # #             print(f"\n  --- {lb} TEST ---")
# # #             for key, ref in [("ADE", 172.68), ("72h", 321.39),
# # #                                ("ATE_mean", 142.21), ("CTE_mean", 42.04)]:
# # #                 v = r.get(key, float("nan"))
# # #                 mk = "BEAT!" if np.isfinite(v) and v < ref else f"target:{ref:.0f}"
# # #                 print(f"    {key:<10}: {v:>8.1f} km  [{mk}  gap:{v-ref:+.1f}]")

# # #     print("=" * 72)


# # # if __name__ == "__main__":
# # #     args = get_args()
# # #     np.random.seed(42); torch.manual_seed(42)
# # #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# # #     main(args)

# # """
# # train_v73.py — TC-FlowMatching v72 model + V73 training strategy
# # ════════════════════════════════════════════════════════════════════════
# # CHIẾN LƯỢC TỐI ƯU DỰA TRÊN ANALYSIS + PAPER TCNM:

# # KEY INSIGHT từ paper TCNM:
# #   - GC-Net chỉ cần q=2 epochs warm-up, sau đó 100 epochs joint
# #   - Most improvement từ joint training, không phải phase separation
# #   - V68/V72 dùng 20ep generator-only → selector học với frozen modes
# #     → suboptimal vì modes và selector không co-adapt

# # KEY INSIGHT từ log v68:
# #   - Best luôn tại Phase 1 (EMA ep9=144.66)
# #   - Phase 2 plateau → 15 no-improve → ăn hết patience trước Phase 3
# #   - Phase 3 chỉ còn ~20ep → không đủ cho joint improvement
# #   - → Cần per-phase patience reset để Phase 3 có đủ epochs

# # V73 STRATEGY:
# #   Phase 1 (ep0..SEL_WARMUP-1): Generator trains hard
# #     - selector_warmup=8 (ngắn hơn, đủ để generator init tốt)
# #     - Tất cả train, selector ở 10% LR (đủ để không đứng yên)
# #     - Patience P1=12: nếu 12ep không improve → move to Phase 2

# #   Phase 2 (ep..+5): Selector focus, rất ngắn
# #     - Generator 10% LR (không frozen hoàn toàn)
# #     - Selector full LR
# #     - Patience P2=6 (ngắn, chỉ selector warmup)
# #     - Phase 2 KHÔNG count vào main patience

# #   Phase 3 (joint, ep..end): Main training
# #     - RESET patience về 0 khi bắt đầu Phase 3
# #     - LR reset về max_lr * 0.5 (fresh start)
# #     - Patience P3=50: đủ dài để joint improvement
# #     - min_ep=SEL_WARMUP+5+10=23 để không stop quá sớm

# # EARLY STOP DESIGN:
# #   - Saver có 3 patience counters: p1, p2, p3
# #   - Mỗi phase chỉ dùng counter của phase đó
# #   - Phase transition reset counter tương ứng
# #   - Phase 3 counter bắt đầu từ 0 → full 50 epochs available

# # LR STRATEGY:
# #   - Phase 1: cosine warmup từ 0 → 1e-4 trong 2 ep, sau đó cosine decay
# #   - Phase 3: reset optimizer momentum + LR về 5e-5
# #     → Tránh stale momentum từ Phase 1 ảnh hưởng Phase 3
# #     → Giống paper TCNM restart sau GC-Net warmup

# # CHECKPOINT:
# #   - best_ema: chỉ save khi EMA eval improve (EMA smoother)
# #   - best_composite: save khi any eval improve
# #   - Cả 2 track separately → không bị Phase 2 plateau xóa Phase 1 best

# # HOW TO RUN:
# #   python scripts/train_v73.py \\
# #       --dataset_root /kaggle/input/.../tc-ofm \\
# #       --output_dir   runs/v73 \\
# #       --batch_size   32 \\
# #       --sel_warmup   8 \\
# #       --num_epochs   120 \\
# #       --use_amp
# # """
# # from __future__ import annotations

# # import sys, os
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # import argparse, time, random
# # from collections import defaultdict

# # import numpy as np
# # import torch
# # import torch.optim as optim
# # from torch.amp import autocast, GradScaler
# # from torch.utils.data import DataLoader, Subset

# # from Model.data.loader_training import data_loader
# # from Model.flow_matching_model import TCFlowMatchingV72

# # try:
# #     from Model.utils import get_cosine_schedule_with_warmup
# # except ImportError:
# #     from torch.optim.lr_scheduler import CosineAnnealingLR
# #     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
# #         return CosineAnnealingLR(opt, T_max=total_steps, eta_min=min_lr)

# # TARGETS = {
# #     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
# #     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# # }
# # R_EARTH = 6371.0


# # # ── Helpers ───────────────────────────────────────────────────────────────────

# # def _unwrap(m):
# #     return m._orig_mod if hasattr(m, '_orig_mod') else m

# # def _sel_params(m):
# #     return list(_unwrap(m).selector.parameters())

# # def _gen_params(m):
# #     """Generator + learnable weights (step_weights, loss_weights_*)."""
# #     u = _unwrap(m)
# #     return (list(u.encoder.parameters()) +
# #             list(u.velocity_heads.parameters()) +
# #             list(u.step_weights.parameters()) +
# #             list(u.loss_weights_p1.parameters()) +
# #             list(u.loss_weights_p23.parameters()))

# # def _freeze(params):
# #     for p in params: p.requires_grad_(False)

# # def _unfreeze(params):
# #     for p in params: p.requires_grad_(True)

# # def _get_phase(ep: int, sw: int) -> int:
# #     if ep < sw:        return 1
# #     if ep < sw + 5:    return 2   # Phase 2 chỉ 5 epochs (ngắn hơn V72)
# #     return 3

# # def move(b, dev):
# #     out = list(b)
# #     for i, x in enumerate(out):
# #         if torch.is_tensor(x):          out[i] = x.to(dev)
# #         elif isinstance(x, dict):
# #             out[i] = {k: v.to(dev) if torch.is_tensor(v) else v
# #                        for k, v in x.items()}
# #     return out


# # # ── Metric utils ──────────────────────────────────────────────────────────────

# # def _ntd(a):
# #     o = a.clone()
# #     o[..., 0] = (a[..., 0]*50. + 1800.) / 10.
# #     o[..., 1] = (a[..., 1]*50.) / 10.
# #     return o

# # def _hav(p1, p2):
# #     la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
# #     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
# #     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
# #     a = (torch.sin(dlat/2).pow(2) +
# #          torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2).pow(2))
# #     return 2. * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())

# # def _atecte(pd, gd):
# #     T = min(pd.shape[0], gd.shape[0])
# #     if T < 2:
# #         z = pd.new_zeros(1, pd.shape[1]); return z, z
# #     lo1=torch.deg2rad(gd[:T-1,:,0]); la1=torch.deg2rad(gd[:T-1,:,1])
# #     lo2=torch.deg2rad(gd[1:T, :,0]); la2=torch.deg2rad(gd[1:T, :,1])
# #     lo3=torch.deg2rad(pd[1:T, :,0]); la3=torch.deg2rad(pd[1:T, :,1])
# #     ya = torch.sin(lo2-lo1)*torch.cos(la2)
# #     xa = torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# #     be = torch.atan2(ya, xa)
# #     ye = torch.sin(lo3-lo2)*torch.cos(la3)
# #     xe = torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# #     bee = torch.atan2(ye, xe)
# #     tot = _hav(pd[1:T], gd[1:T])
# #     return tot*torch.cos(bee-be), tot*torch.sin(bee-be)


# # class Acc:
# #     def __init__(self):
# #         self.d=[]; self.a=[]; self.c=[]
# #         self.sd = defaultdict(list)
# #         self._h = {12:1, 24:3, 48:7, 72:11}

# #     def update(self, dist, ate=None, cte=None):
# #         self.d.extend(dist.mean(0).tolist())
# #         for h, s in self._h.items():
# #             if s < dist.shape[0]: self.sd[h].extend(dist[s].tolist())
# #         if ate is not None: self.a.extend(ate.abs().mean(0).tolist())
# #         if cte is not None: self.c.extend(cte.abs().mean(0).tolist())

# #     def compute(self):
# #         r = {"ADE":      float(np.mean(self.d)) if self.d else float("nan"),
# #              "ATE_mean": float(np.mean(self.a)) if self.a else float("nan"),
# #              "CTE_mean": float(np.mean(self.c)) if self.c else float("nan"),
# #              "n":        len(self.d)}
# #         for h in self._h:
# #             v = self.sd.get(h, [])
# #             r[f"{h}h"] = float(np.mean(v)) if v else float("nan")
# #         return r


# # def _score(r):
# #     ade = r.get("ADE",   1e9); h72 = r.get("72h",   1e9)
# #     ate = r.get("ATE_mean", 1e9); cte = r.get("CTE_mean", 1e9)
# #     if not np.isfinite(ate): ate = ade*.46
# #     if not np.isfinite(cte): cte = ade*.53
# #     return 100.*(0.05*(ade/136.) + 0.10*(r.get("12h",ade)/50.) +
# #                  0.15*(r.get("24h",ade)/100.) + 0.20*(r.get("48h",ade)/200.) +
# #                  0.25*(h72/300.) + 0.13*(ate/80.) + 0.12*(cte/94.))

# # def _beat(r):
# #     p = []
# #     for k, t in [("ADE",172.68),("ATE_mean",142.21),("CTE_mean",42.04),
# #                   ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
# #         v = r.get(k, 1e9)
# #         if np.isfinite(v) and v < t: p.append(f"{k.replace('_mean','')}:{v:.1f}")
# #     return "*** BEAT: " + " ".join(p) + " ***" if p else ""

# # def _gap(r):
# #     p = []
# #     for k, ref in [("ADE",172.68),("72h",321.39),
# #                     ("ATE_mean",142.21),("CTE_mean",42.04)]:
# #         v = r.get(k, float("nan"))
# #         if np.isfinite(v):
# #             p.append(f"{k.replace('_mean','')}:{v:.0f}"
# #                      f"({'dn' if v<ref else 'up'}{abs(v-ref):.0f})")
# #     return " | ".join(p)


# # # ── Evaluation ────────────────────────────────────────────────────────────────

# # @torch.no_grad()
# # def evaluate(model, loader, dev, tag="", ema=None, steps=10):
# #     bk = None
# #     if ema:
# #         try: bk = ema.apply_to(model)
# #         except: pass
# #     model.eval(); acc = Acc(); t0 = time.perf_counter(); n = 0
# #     for b in loader:
# #         bl = move(list(b), dev)
# #         p, _ = model.sample(bl, ddim_steps=steps)
# #         g = bl[1]; T = min(p.shape[0], g.shape[0])
# #         pd = _ntd(p[:T]); gd = _ntd(g[:T])
# #         dist = _hav(pd, gd); at, ct = _atecte(pd, gd)
# #         acc.update(dist, at, ct); n += 1
# #     if bk:
# #         try: ema.restore(model, bk)
# #         except: pass
# #     r = acc.compute()
# #     def _v(k): return r.get(k, float("nan"))
# #     def _m(v, t): return "ok" if np.isfinite(v) and v < t else "no"
# #     el = time.perf_counter() - t0
# #     print(f"\n{'='*68}")
# #     print(f"  [{tag}  {el:.0f}s  {n} batches]")
# #     print(f"  ADE={_v('ADE'):.1f}[{_m(_v('ADE'),172.68)}]  "
# #           f"12h={_v('12h'):.0f}  24h={_v('24h'):.0f}  "
# #           f"48h={_v('48h'):.0f}  72h={_v('72h'):.0f}[{_m(_v('72h'),321.39)}]")
# #     if np.isfinite(_v("ATE_mean")):
# #         print(f"  ATE={_v('ATE_mean'):.1f}[{_m(_v('ATE_mean'),142.21)}]  "
# #               f"CTE={_v('CTE_mean'):.1f}[{_m(_v('CTE_mean'),42.04)}]")
# #     print(f"  vs ST-Trans: {_gap(r)}")
# #     bt = _beat(r)
# #     if bt: print(f"  {bt}")
# #     print(f"  Score={_score(r):.2f}")
# #     print(f"{'='*68}\n")
# #     return r


# # # ── Per-Phase Saver ────────────────────────────────────────────────────────────

# # class PerPhaseSaver:
# #     """
# #     Saver với per-phase patience.
# #     Key design:
# #       - cnt riêng cho mỗi phase: cnt1, cnt2, cnt3
# #       - Chỉ cnt của phase hiện tại tăng khi no-improve
# #       - Phase transition: reset cnt của phase mới về 0
# #       - best_score tracking vẫn global (không reset khi phase đổi)
# #       - stop chỉ khi cnt[current_phase] >= patience[current_phase]
# #         VÀ ep >= min_ep
# #     """
# #     def __init__(self,
# #                  patience_p1: int = 12,
# #                  patience_p2: int = 6,
# #                  patience_p3: int = 50,
# #                  min_ep: int = 25):
# #         self.pat = {1: patience_p1, 2: patience_p2, 3: patience_p3}
# #         self.cnt = {1: 0, 2: 0, 3: 0}
# #         self.min_ep = min_ep
# #         self.stop = False
# #         # Global bests
# #         self.bs = self.ba = self.b7 = self.bat = self.bc = float("inf")
# #         self.best_ep = -1

# #     def on_phase_change(self, new_phase: int):
# #         """Reset counter cho phase mới."""
# #         self.cnt[new_phase] = 0
# #         print(f"  [Saver] Phase {new_phase} started — patience reset to 0/{self.pat[new_phase]}")

# #     def _save(self, path, ep, model, opt, sched, tl, vl, extra=None):
# #         m = _unwrap(model); ema = getattr(m, "_ema", None); esd = None
# #         if ema and hasattr(ema, "shadow"):
# #             try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# #             except: pass
# #         d = {"epoch": ep, "model_state_dict": m.state_dict(),
# #              "optimizer_state": opt.state_dict(),
# #              "scheduler_state": sched.state_dict(),
# #              "ema_shadow": esd,
# #              "best_score": self.bs, "best_ade": self.ba,
# #              "best_72h": self.b7, "best_ate": self.bat, "best_cte": self.bc,
# #              "train_loss": tl, "val_loss": vl}
# #         if extra: d.update(extra)
# #         torch.save(d, path)

# #     def update(self, r, ph, model, out, ep, opt, sched, tl, vl, tag=""):
# #         sc  = _score(r)
# #         ade = r.get("ADE",   1e9); h72 = r.get("72h",   1e9)
# #         ate = r.get("ATE_mean", 1e9); cte = r.get("CTE_mean", 1e9)

# #         # Save per-metric bests
# #         for v, attr, fn in [(ade,"ba","best_ade.pth"),
# #                              (h72,"b7","best_72h.pth"),
# #                              (ate,"bat","best_ate.pth"),
# #                              (cte,"bc","best_cte.pth")]:
# #             if v < getattr(self, attr):
# #                 setattr(self, attr, v)
# #                 self._save(os.path.join(out, fn), ep, model, opt, sched, tl, vl)

# #         # Global best
# #         if sc < self.bs:
# #             self.bs = sc; self.cnt[ph] = 0; self.best_ep = ep
# #             self._save(os.path.join(out, f"best_{tag or 'composite'}.pth"),
# #                        ep, model, opt, sched, tl, vl,
# #                        {"score": sc, "ade": ade, "h72": h72, "phase": ph})
# #             print(f"  [BEST] {tag} ep={ep} score={sc:.2f} "
# #                   f"ADE={ade:.1f} 72h={h72:.0f} ATE={ate:.1f} CTE={cte:.1f}")
# #         else:
# #             self.cnt[ph] += 1
# #             print(f"  No improve (P{ph}) {self.cnt[ph]}/{self.pat[ph]}  "
# #                   f"best={self.bs:.2f} cur={sc:.2f}  best_ep={self.best_ep}")

# #         # Early stop: only based on current phase patience
# #         if ep >= self.min_ep and self.cnt[ph] >= self.pat[ph]:
# #             if ph == 3:
# #                 self.stop = True
# #                 print(f"  [STOP] Phase 3 patience exhausted at ep={ep}")
# #             elif ph == 1:
# #                 print(f"  [P1 patience done] Moving to Phase 2 early")
# #                 # Don't set stop=True, let phase transition handle it
# #             # Phase 2 always completes (only 5 epochs)


# # def mksub(ds, n, bs, cf):
# #     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
# #     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
# #                        collate_fn=cf, num_workers=0, drop_last=False)


# # # ── Args ──────────────────────────────────────────────────────────────────────

# # def get_args():
# #     p = argparse.ArgumentParser()
# #     p.add_argument("--dataset_root",       default="TCND_vn")
# #     p.add_argument("--obs_len",            default=8,    type=int)
# #     p.add_argument("--pred_len",           default=12,   type=int)
# #     p.add_argument("--batch_size",         default=32,   type=int)
# #     p.add_argument("--num_epochs",         default=120,  type=int)
# #     p.add_argument("--g_learning_rate",    default=1e-4, type=float)
# #     p.add_argument("--sel_lr_ratio",       default=0.10, type=float)
# #     p.add_argument("--gen_lr_p2_ratio",    default=0.10, type=float,
# #                     help="Generator LR in Phase 2 (10% of g_lr)")
# #     p.add_argument("--gen_lr_p3_reset",    default=0.50, type=float,
# #                     help="Generator LR reset at Phase 3 start (50% of g_lr)")
# #     p.add_argument("--weight_decay",       default=1e-3, type=float)
# #     p.add_argument("--warmup_epochs",      default=2,    type=int)
# #     p.add_argument("--grad_clip",          default=1.0,  type=float)
# #     # Per-phase patience
# #     p.add_argument("--patience_p1",        default=12,   type=int)
# #     p.add_argument("--patience_p2",        default=6,    type=int)
# #     p.add_argument("--patience_p3",        default=50,   type=int)
# #     p.add_argument("--min_ep",             default=25,   type=int)
# #     p.add_argument("--use_amp",            action="store_true")
# #     p.add_argument("--num_workers",        default=2,    type=int)
# #     p.add_argument("--sigma_min",          default=0.02, type=float)
# #     p.add_argument("--head_noise_base",    default=0.03, type=float)
# #     p.add_argument("--use_ot",             default=True, action="store_true")
# #     p.add_argument("--no_ot",              dest="use_ot", action="store_false")
# #     p.add_argument("--cfg_guidance_scale", default=1.3,  type=float)
# #     p.add_argument("--sel_warmup",         default=8,    type=int,
# #                     help="Phase 1 length (shorter than V72's 20, like paper)")
# #     p.add_argument("--val_freq",           default=3,    type=int)
# #     p.add_argument("--val_subset_size",    default=500,  type=int)
# #     p.add_argument("--fast_ddim",          default=10,   type=int)
# #     p.add_argument("--full_ddim",          default=20,   type=int)
# #     p.add_argument("--use_ema",            default=True, action="store_true")
# #     p.add_argument("--no_ema",             dest="use_ema", action="store_false")
# #     p.add_argument("--ema_decay",          default=0.995, type=float)
# #     p.add_argument("--output_dir",         default="runs/v73")
# #     p.add_argument("--gpu_num",            default="0")
# #     p.add_argument("--delim",              default=" ")
# #     p.add_argument("--skip",               default=1,    type=int)
# #     p.add_argument("--min_ped",            default=1,    type=int)
# #     p.add_argument("--threshold",          default=0.002, type=float)
# #     p.add_argument("--other_modal",        default="gph")
# #     p.add_argument("--test_year",          default=None, type=int)
# #     p.add_argument("--resume",             default=None)
# #     p.add_argument("--resume_epoch",       default=None, type=int)
# #     p.add_argument("--eval_test_after_train", default=True, action="store_true")
# #     return p.parse_args()


# # # ── Main ──────────────────────────────────────────────────────────────────────

# # def main(args):
# #     if torch.cuda.is_available():
# #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# #     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     os.makedirs(args.output_dir, exist_ok=True)
# #     SW = args.sel_warmup  # default 8

# #     print("=" * 72)
# #     print(f"  TC-FlowMatching V73 training strategy")
# #     print(f"  Phase 1 (ep 0-{SW-1}):      Generator+weights, sel 10% LR")
# #     print(f"  Phase 2 (ep {SW}-{SW+4}):   Selector focus, gen 10% LR (5ep)")
# #     print(f"  Phase 3 (ep {SW+5}+):        Joint, LR reset, patience={args.patience_p3}")
# #     print(f"  Per-phase patience: P1={args.patience_p1} P2={args.patience_p2} P3={args.patience_p3}")
# #     print(f"  Key: Phase 3 resets patience → full {args.patience_p3}ep joint training")
# #     print(f"  Target: ADE<172.68 72h<321.39 ATE<142.21 CTE<42.04")
# #     print("=" * 72)

# #     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
# #     vd,  vl  = data_loader(args, {"root": args.dataset_root, "type": "val"},   test=True)
# #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# #     vsub = mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
# #     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

# #     model = TCFlowMatchingV72(
# #         pred_len=args.pred_len, obs_len=args.obs_len,
# #         sigma_min=args.sigma_min, use_ema=args.use_ema, ema_decay=args.ema_decay,
# #         use_ot=args.use_ot, head_noise_base=args.head_noise_base,
# #         cfg_guidance_scale=args.cfg_guidance_scale, selector_warmup=SW,
# #     ).to(dev)
# #     model.init_ema()
# #     print(f"  params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# #     # ── Optimizer ─────────────────────────────────────────────────────────────
# #     opt = optim.AdamW([
# #         {"params": _gen_params(model), "lr": args.g_learning_rate,
# #           "name": "generator"},
# #         {"params": _sel_params(model), "lr": args.g_learning_rate * args.sel_lr_ratio,
# #           "name": "selector"},
# #     ], weight_decay=args.weight_decay)

# #     nstep = len(trl)
# #     total = nstep * args.num_epochs
# #     wstp  = nstep * args.warmup_epochs
# #     sched = get_cosine_schedule_with_warmup(opt, wstp, total, min_lr=1e-6)
# #     scaler = GradScaler("cuda", enabled=args.use_amp)

# #     saver = PerPhaseSaver(
# #         patience_p1=args.patience_p1,
# #         patience_p2=args.patience_p2,
# #         patience_p3=args.patience_p3,
# #         min_ep=args.min_ep,
# #     )

# #     # ── Resume ────────────────────────────────────────────────────────────────
# #     start = 0
# #     if args.resume and os.path.exists(args.resume):
# #         print(f"  Loading: {args.resume}")
# #         ck = torch.load(args.resume, map_location=dev)
# #         ms, us = _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
# #         if ms: print(f"  Missing: {len(ms)}")
# #         ema = getattr(_unwrap(model), "_ema", None)
# #         if ema and ck.get("ema_shadow"):
# #             for k, v in ck["ema_shadow"].items():
# #                 if k in ema.shadow: ema.shadow[k].copy_(v.to(dev))
# #         try: opt.load_state_dict(ck["optimizer_state"])
# #         except: pass
# #         try: sched.load_state_dict(ck["scheduler_state"])
# #         except:
# #             for _ in range(ck.get("epoch", 0) * nstep): sched.step()
# #         for a, attr in [("best_score","bs"),("best_ade","ba"),
# #                           ("best_72h","b7"),("best_ate","bat"),("best_cte","bc")]:
# #             if a in ck: setattr(saver, attr, ck[a])
# #         start = args.resume_epoch or ck.get("epoch", 0) + 1
# #         print(f"  Resume ep={start}")

# #     try:
# #         model = torch.compile(model, mode="reduce-overhead")
# #         print("  torch.compile: enabled")
# #     except: pass

# #     # ── Training loop ─────────────────────────────────────────────────────────
# #     oracle_ema = 999.; _pa = .05
# #     prev_ph = -1; ts = time.perf_counter()
# #     p3_lr_reset_done = False
# #     print(f"  Training: {nstep} steps/ep, start ep={start}")
# #     print("=" * 72)

# #     for ep in range(start, args.num_epochs):
# #         ph = _get_phase(ep, SW)

# #         # ── Phase transitions ─────────────────────────────────────────────────
# #         if ph != prev_ph:
# #             saver.on_phase_change(ph)

# #             if ph == 1:
# #                 # Phase 1: tất cả train, selector ở LR thấp
# #                 _unfreeze(_gen_params(model))
# #                 _unfreeze(_sel_params(model))
# #                 print(f"  [P1 ep{ep}] Generator active, selector 10% LR")

# #             elif ph == 2:
# #                 # Phase 2: selector focus, generator chậm
# #                 _unfreeze(_gen_params(model))
# #                 _unfreeze(_sel_params(model))
# #                 print(f"  [P2 ep{ep}] Selector focus (5ep only)")

# #             elif ph == 3 and not p3_lr_reset_done:
# #                 # Phase 3: reset LR để fresh joint training
# #                 _unfreeze(_gen_params(model))
# #                 _unfreeze(_sel_params(model))
# #                 p3_reset_lr = args.g_learning_rate * args.gen_lr_p3_reset
# #                 # Reset optimizer state (momentum) để tránh stale từ Phase 1
# #                 for pg in opt.param_groups:
# #                     pg["lr"] = (p3_reset_lr if pg.get("name") == "generator"
# #                                 else p3_reset_lr * args.sel_lr_ratio)
# #                     # Reset Adam state cho smooth start
# #                     for p_ in pg["params"]:
# #                         state = opt.state.get(p_)
# #                         if state:
# #                             state.pop("exp_avg", None)
# #                             state.pop("exp_avg_sq", None)
# #                 # Reset scheduler: create new one for Phase 3
# #                 p3_total = nstep * (args.num_epochs - ep)
# #                 p3_warmup = nstep * 1  # 1 epoch warmup
# #                 sched = get_cosine_schedule_with_warmup(
# #                     opt, p3_warmup, p3_total, min_lr=5e-7)
# #                 p3_lr_reset_done = True
# #                 print(f"  [P3 ep{ep}] LR reset to {p3_reset_lr:.1e}, "
# #                       f"fresh cosine schedule, patience={args.patience_p3}")
# #                 print(f"  [P3] This is main joint training phase — "
# #                       f"expect steady ADE decrease")

# #             prev_ph = ph

# #         # ── LR per phase ──────────────────────────────────────────────────────
# #         if ph == 1:
# #             for pg in opt.param_groups:
# #                 if pg.get("name") == "selector":
# #                     # Selector trains from start but at low LR
# #                     pg["lr"] = args.g_learning_rate * args.sel_lr_ratio * 0.1
# #         elif ph == 2:
# #             for pg in opt.param_groups:
# #                 if pg.get("name") == "generator":
# #                     pg["lr"] = args.g_learning_rate * args.gen_lr_p2_ratio
# #                 elif pg.get("name") == "selector":
# #                     pg["lr"] = args.g_learning_rate * args.sel_lr_ratio
# #         # Phase 3: sched handles LR (reset above)

# #         # ── Train epoch ───────────────────────────────────────────────────────
# #         model.train(); sl = 0.; t0 = time.perf_counter()

# #         for i, batch in enumerate(trl):
# #             bl = move(list(batch), dev)
# #             with autocast(device_type="cuda", enabled=args.use_amp):
# #                 bd = model.get_loss_breakdown(bl, epoch=ep)

# #             opt.zero_grad()
# #             scaler.scale(bd["total"]).backward()
# #             scaler.unscale_(opt)
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# #             sb = scaler.get_scale()
# #             scaler.step(opt); scaler.update()
# #             if scaler.get_scale() >= sb: sched.step()
# #             model.ema_update()
# #             sl += bd["total"].item()

# #             # oracle EMA
# #             ok = bd.get("oracle_ade_km", oracle_ema)
# #             try:
# #                 nv = float(ok)
# #                 if np.isfinite(nv): oracle_ema = (1-_pa)*oracle_ema + _pa*nv
# #             except: pass

# #             if i % 20 == 0:
# #                 lg = next((pg["lr"] for pg in opt.param_groups
# #                             if pg.get("name")=="generator"), 0.)
# #                 ls = next((pg["lr"] for pg in opt.param_groups
# #                             if pg.get("name")=="selector"), 0.)
# #                 sw_r = bd.get('sw_ratio', 0.)
# #                 sw_warn = " [!sw_ratio<1]" if sw_r < 1.0 else ""
# #                 print(
# #                     f"  [P{ph}][{ep:>3}][{i:>3}/{nstep}]"
# #                     f"  tot={bd['total'].item():.3f}"
# #                     f"  fm={bd.get('L_FM',0):.4f}"
# #                     f"  dpe={bd.get('L_dpe',0):.4f}"
# #                     f"  rank={bd.get('L_rank',0):.4f}"
# #                     f"  div={bd.get('L_div',0):.3f}"
# #                     f"  orc={bd.get('oracle_ade_km',0):.1f}km"
# #                     f"  sw24={bd.get('sw_24h',0):.2f}"
# #                     f"  sw72={bd.get('sw_72h',0):.2f}"
# #                     f"  swr={sw_r:.2f}{sw_warn}"
# #                     f"  dpe/fm={bd.get('lw_dpe_fm_ratio',0):.2f}"
# #                     f"  d={bd.get('delta_mean',0):.2f}"
# #                     f"  lg={lg:.2e}  ls={ls:.2e}"
# #                 )

# #         avt = sl / nstep

# #         # ── Val loss ──────────────────────────────────────────────────────────
# #         model.eval(); vls = 0.
# #         with torch.no_grad():
# #             for batch in vl:
# #                 bv = move(list(batch), dev)
# #                 with autocast(device_type="cuda", enabled=args.use_amp):
# #                     vls += model.get_loss(bv, epoch=ep).item()
# #         avv = vls / len(vl)
# #         eps = time.perf_counter() - t0
# #         print(f"  Epoch {ep:>3} P{ph} train={avt:.4f} val={avv:.4f} "
# #               f"{eps:.0f}s oracle_ema={oracle_ema:.1f}km "
# #               f"p1={saver.cnt[1]} p2={saver.cnt[2]} p3={saver.cnt[3]}")

# #         # ── Fast eval ─────────────────────────────────────────────────────────
# #         rf = evaluate(model, vsub, dev, tag=f"FAST/P{ph} ep{ep}",
# #                        steps=args.fast_ddim)
# #         saver.update(rf, ph, model, args.output_dir, ep,
# #                       opt, sched, avt, avv, tag="fast")

# #         # ── Full val every val_freq ───────────────────────────────────────────
# #         if ep % args.val_freq == 0:
# #             em = getattr(_unwrap(model), "_ema", None)
# #             rr = evaluate(model, vl, dev, tag=f"RAW/P{ph} ep{ep}",
# #                            steps=args.full_ddim)
# #             saver.update(rr, ph, model, args.output_dir, ep,
# #                           opt, sched, avt, avv, tag="raw")
# #             if em and ep >= 3:
# #                 re = evaluate(model, vl, dev, tag=f"EMA/P{ph} ep{ep}",
# #                                ema=em, steps=args.full_ddim)
# #                 saver.update(re, ph, model, args.output_dir, ep,
# #                               opt, sched, avt, avv, tag="ema")

# #         # ── Checkpoint ───────────────────────────────────────────────────────
# #         if ep % 10 == 0 or ep == args.num_epochs - 1:
# #             path = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
# #             m = _unwrap(model); ema_obj = getattr(m, "_ema", None); esd = None
# #             if ema_obj and hasattr(ema_obj, "shadow"):
# #                 try: esd = {k: v.cpu().clone() for k, v in ema_obj.shadow.items()}
# #                 except: pass
# #             torch.save({"epoch": ep, "model_state_dict": m.state_dict(),
# #                          "optimizer_state": opt.state_dict(),
# #                          "scheduler_state": sched.state_dict(),
# #                          "ema_shadow": esd, "oracle_ema": oracle_ema,
# #                          "phase": ph, "best_score": saver.bs,
# #                          "best_ade": saver.ba, "best_72h": saver.b7,
# #                          "best_ate": saver.bat, "best_cte": saver.bc}, path)

# #         # ── Early stop: only Phase 3 can trigger full stop ────────────────────
# #         if saver.stop:
# #             print(f"  Early stop at ep={ep}")
# #             break

# #     th = (time.perf_counter() - ts) / 3600.
# #     print(f"\n  Best: ADE={saver.ba:.1f} 72h={saver.b7:.0f} "
# #           f"ATE={saver.bat:.1f} CTE={saver.bc:.1f} ({th:.2f}h)")

# #     # ── Post-training test ────────────────────────────────────────────────────
# #     if args.eval_test_after_train:
# #         print("\n" + "="*72 + "\n  POST-TRAINING TEST\n" + "="*72)
# #         try:
# #             _, tl2 = data_loader(args, {"root": args.dataset_root,
# #                                           "type": "test"}, test=True)
# #         except:
# #             print("  No test set, using val"); tl2 = vl
# #         for fn, lb in [("best_composite.pth","COMPOSITE"),
# #                          ("best_72h.pth","72H"),
# #                          ("best_ema.pth","EMA")]:
# #             pp = os.path.join(args.output_dir, fn)
# #             if not os.path.exists(pp): continue
# #             ck = torch.load(pp, map_location=dev)
# #             _unwrap(model).load_state_dict(ck["model_state_dict"], strict=False)
# #             em = getattr(_unwrap(model), "_ema", None)
# #             if em and ck.get("ema_shadow"):
# #                 for k, v in ck["ema_shadow"].items():
# #                     if k in em.shadow: em.shadow[k].copy_(v.to(dev))
# #             r = evaluate(model, tl2, dev, tag=f"TEST/{lb}",
# #                           steps=args.full_ddim)
# #             print(f"\n  --- {lb} ---")
# #             for key, ref in [("ADE",172.68),("72h",321.39),
# #                                ("ATE_mean",142.21),("CTE_mean",42.04)]:
# #                 v = r.get(key, float("nan"))
# #                 mk = "BEAT!" if np.isfinite(v) and v < ref else f"need:{ref:.0f}"
# #                 print(f"    {key:<10}: {v:>8.1f}  [{mk} gap:{v-ref:+.1f}]")
# #     print("=" * 72)


# # if __name__ == "__main__":
# #     args = get_args()
# #     np.random.seed(42); torch.manual_seed(42)
# #     if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
# #     main(args)

# """
# train_flowmatching.py — TC-FlowMatching v69 training script
# ════════════════════════════════════════════════════════════════════════
# Đơn giản như v59: 1 phase, không freeze/unfreeze, không phase switching.
# Thêm monitor adaptive weights (aw_6h, aw_72h, w_dpe, w_disp).

# HOW TO RUN:
#   python scripts/train_flowmatching.py \
#       --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
#       --output_dir   runs/v69 \
#       --batch_size   32 \
#       --num_epochs   120 \
#       --learning_rate 1e-4 \
#       --use_amp \
#       --use_ot
# """
# from __future__ import annotations

# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse, time, random, math
# from collections import defaultdict

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.amp import autocast, GradScaler
# from torch.utils.data import DataLoader, Subset

# from Model.data.loader_training import data_loader
# from Model.flow_matching_model import TCFlowMatching, EMAModel

# try:
#     from Model.utils import get_cosine_schedule_with_warmup
# except ImportError:
#     from torch.optim.lr_scheduler import CosineAnnealingLR
#     def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
#         return CosineAnnealingLR(opt, T_max=total_steps, eta_min=min_lr)


# # ── Targets ───────────────────────────────────────────────────────────────────
# TARGETS = {
#     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
#     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# }
# R_EARTH = 6371.0


# # ── Helpers ───────────────────────────────────────────────────────────────────

# def _unwrap(m):
#     return m._orig_mod if hasattr(m, '_orig_mod') else m


# def move(batch, dev):
#     out = list(batch)
#     for i, x in enumerate(out):
#         if torch.is_tensor(x):
#             out[i] = x.to(dev)
#         elif isinstance(x, dict):
#             out[i] = {k: (v.to(dev) if torch.is_tensor(v) else v)
#                        for k, v in x.items()}
#     return out


# def _ntd(a):
#     """Normalized coords → degrees."""
#     o = a.clone()
#     o[..., 0] = (a[..., 0] * 50. + 1800.) / 10.
#     o[..., 1] = (a[..., 1] * 50.) / 10.
#     return o


# def _hav(p1, p2):
#     """Haversine distance in km."""
#     la1 = torch.deg2rad(p1[..., 1]); la2 = torch.deg2rad(p2[..., 1])
#     dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
#     dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
#     a = (torch.sin(dlat/2).pow(2) +
#          torch.cos(la1) * torch.cos(la2) * torch.sin(dlon/2).pow(2))
#     return 2. * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())


# def _atecte(pd, gd):
#     """Compute ATE and CTE from predicted and GT trajectories in degrees."""
#     T = min(pd.shape[0], gd.shape[0])
#     if T < 2:
#         z = pd.new_zeros(1, pd.shape[1])
#         return z, z
#     lo1 = torch.deg2rad(gd[:T-1,:,0]); la1 = torch.deg2rad(gd[:T-1,:,1])
#     lo2 = torch.deg2rad(gd[1:T, :,0]); la2 = torch.deg2rad(gd[1:T, :,1])
#     lo3 = torch.deg2rad(pd[1:T, :,0]); la3 = torch.deg2rad(pd[1:T, :,1])
#     ya = torch.sin(lo2-lo1) * torch.cos(la2)
#     xa = (torch.cos(la1)*torch.sin(la2) -
#           torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1))
#     be = torch.atan2(ya, xa)
#     ye = torch.sin(lo3-lo2) * torch.cos(la3)
#     xe = (torch.cos(la2)*torch.sin(la3) -
#           torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2))
#     bee = torch.atan2(ye, xe)
#     tot = _hav(pd[1:T], gd[1:T])
#     ang = bee - be
#     return tot * torch.cos(ang), tot * torch.sin(ang)


# # ── Accumulator ───────────────────────────────────────────────────────────────

# class MetricAccumulator:
#     def __init__(self):
#         self.dists = []
#         self.ates = []
#         self.ctes = []
#         self.step_dists = defaultdict(list)
#         self._step_map = {12: 1, 24: 3, 48: 7, 72: 11}

#     def update(self, dist, ate=None, cte=None):
#         self.dists.extend(dist.mean(0).tolist())
#         for h, s in self._step_map.items():
#             if s < dist.shape[0]:
#                 self.step_dists[h].extend(dist[s].tolist())
#         if ate is not None:
#             self.ates.extend(ate.abs().mean(0).tolist())
#         if cte is not None:
#             self.ctes.extend(cte.abs().mean(0).tolist())

#     def compute(self):
#         r = {
#             "ADE":      float(np.mean(self.dists)) if self.dists else float("nan"),
#             "ATE_mean": float(np.mean(self.ates))  if self.ates  else float("nan"),
#             "CTE_mean": float(np.mean(self.ctes))  if self.ctes  else float("nan"),
#             "n":        len(self.dists),
#         }
#         for h in self._step_map:
#             v = self.step_dists.get(h, [])
#             r[f"{h}h"] = float(np.mean(v)) if v else float("nan")
#         return r


# def _score(r):
#     """Composite score — lower is better."""
#     ade = r.get("ADE", 1e9); h72 = r.get("72h", 1e9)
#     ate = r.get("ATE_mean", 1e9); cte = r.get("CTE_mean", 1e9)
#     if not np.isfinite(ate): ate = ade * 0.46
#     if not np.isfinite(cte): cte = ade * 0.53
#     return 100. * (0.05*(ade/136.) + 0.10*(r.get("12h",ade)/50.) +
#                    0.15*(r.get("24h",ade)/100.) + 0.20*(r.get("48h",ade)/200.) +
#                    0.25*(h72/300.) + 0.13*(ate/80.) + 0.12*(cte/94.))


# def _gap_str(r):
#     p = []
#     for k, ref in [("ADE",172.68), ("72h",321.39),
#                     ("ATE_mean",142.21), ("CTE_mean",42.04)]:
#         v = r.get(k, float("nan"))
#         if np.isfinite(v):
#             arrow = "↓" if v < ref else "↑"
#             p.append(f"{k.replace('_mean','')}:{v:.0f}({arrow}{abs(v-ref):.0f})")
#     return " | ".join(p)


# def _beat_str(r):
#     p = []
#     for k, t in [("ADE",172.68), ("ATE_mean",142.21), ("CTE_mean",42.04),
#                   ("72h",321.39), ("12h",65.42), ("24h",104.67), ("48h",205.10)]:
#         v = r.get(k, 1e9)
#         if np.isfinite(v) and v < t:
#             p.append(f"{k.replace('_mean','')}✅{v:.1f}")
#     return "🏆 BEAT: " + " ".join(p) if p else ""


# # ── Evaluate ──────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def evaluate(model, loader, dev, tag="", ema=None, steps=20,
#              num_ensemble=50):
#     backup = None
#     if ema:
#         try: backup = ema.apply_to(model)
#         except: pass

#     model.eval()
#     acc = MetricAccumulator()
#     t0 = time.perf_counter()
#     n_batches = 0

#     for batch in loader:
#         bl = move(list(batch), dev)
#         pred, _, _ = model.sample(bl, num_ensemble=num_ensemble,
#                                    ddim_steps=steps)
#         gt = bl[1]
#         T = min(pred.shape[0], gt.shape[0])
#         pd = _ntd(pred[:T]); gd = _ntd(gt[:T])
#         dist = _hav(pd, gd)
#         ate, cte = _atecte(pd, gd)
#         acc.update(dist, ate, cte)
#         n_batches += 1

#     if backup:
#         try: ema.restore(model, backup)
#         except: pass

#     r = acc.compute()
#     el = time.perf_counter() - t0

#     def _v(k): return r.get(k, float("nan"))
#     def _m(v, t): return "✅" if np.isfinite(v) and v < t else "❌"

#     print(f"\n{'='*68}")
#     print(f"  [{tag}  {el:.0f}s  {n_batches} batches]")
#     print(f"  ADE={_v('ADE'):.1f}{_m(_v('ADE'),172.68)}  "
#           f"12h={_v('12h'):.0f}{_m(_v('12h'),65.42)}  "
#           f"24h={_v('24h'):.0f}{_m(_v('24h'),104.67)}  "
#           f"48h={_v('48h'):.0f}{_m(_v('48h'),205.10)}  "
#           f"72h={_v('72h'):.0f}{_m(_v('72h'),321.39)}")
#     if np.isfinite(_v("ATE_mean")):
#         print(f"  ATE={_v('ATE_mean'):.1f}{_m(_v('ATE_mean'),142.21)}  "
#               f"CTE={_v('CTE_mean'):.1f}{_m(_v('CTE_mean'),42.04)}")
#     print(f"  vs ST-Trans: {_gap_str(r)}")
#     bt = _beat_str(r)
#     if bt: print(f"  {bt}")
#     print(f"  Score={_score(r):.2f}")
#     print(f"{'='*68}\n")
#     return r


# # ── Save/Load ─────────────────────────────────────────────────────────────────

# def _save_ckpt(path, ep, model, opt, sched, best_score, best_ade,
#                best_72h, best_ate, best_cte, train_loss, val_loss,
#                extra=None):
#     m = _unwrap(model)
#     ema = getattr(m, "_ema", None)
#     esd = None
#     if ema and hasattr(ema, "shadow"):
#         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
#         except: pass
#     d = {
#         "epoch": ep, "model_state_dict": m.state_dict(),
#         "optimizer_state": opt.state_dict(),
#         "scheduler_state": sched.state_dict(),
#         "ema_shadow": esd,
#         "best_score": best_score, "best_ade": best_ade,
#         "best_72h": best_72h, "best_ate": best_ate, "best_cte": best_cte,
#         "train_loss": train_loss, "val_loss": val_loss,
#     }
#     if extra: d.update(extra)
#     torch.save(d, path)


# class BestTracker:
#     def __init__(self, patience=35):
#         self.patience = patience
#         self.cnt = 0
#         self.stop = False
#         self.best_score = float("inf")
#         self.best_ade = float("inf")
#         self.best_72h = float("inf")
#         self.best_ate = float("inf")
#         self.best_cte = float("inf")

#     def update(self, r, model, out_dir, ep, opt, sched, tl, vl,
#                tag="", min_ep=20):
#         sc = _score(r)
#         ade = r.get("ADE", 1e9)
#         h72 = r.get("72h", 1e9)
#         ate = r.get("ATE_mean", 1e9)
#         cte = r.get("CTE_mean", 1e9)

#         # Per-metric bests
#         for v, attr, fn in [(ade, "best_ade", "best_ade.pth"),
#                              (h72, "best_72h", "best_72h.pth"),
#                              (ate, "best_ate", "best_ate.pth"),
#                              (cte, "best_cte", "best_cte.pth")]:
#             if v < getattr(self, attr):
#                 setattr(self, attr, v)
#                 _save_ckpt(os.path.join(out_dir, fn), ep, model, opt, sched,
#                            self.best_score, self.best_ade, self.best_72h,
#                            self.best_ate, self.best_cte, tl, vl)

#         # Composite best
#         if sc < self.best_score:
#             self.best_score = sc
#             self.cnt = 0
#             _save_ckpt(
#                 os.path.join(out_dir, f"best_{tag or 'composite'}.pth"),
#                 ep, model, opt, sched,
#                 self.best_score, self.best_ade, self.best_72h,
#                 self.best_ate, self.best_cte, tl, vl,
#                 {"score": sc, "ade": ade, "h72": h72})
#             print(f"  ✅ Best {tag} score={sc:.2f}  "
#                   f"ADE={ade:.1f}  72h={h72:.0f}  "
#                   f"ATE={ate:.1f}  CTE={cte:.1f}")
#         else:
#             self.cnt += 1
#             print(f"  No improve {self.cnt}/{self.patience}  "
#                   f"(best={self.best_score:.2f} cur={sc:.2f})")

#         if ep >= min_ep and self.cnt >= self.patience:
#             self.stop = True


# def mksub(ds, n, bs, collate_fn):
#     idx = random.Random(42).sample(range(len(ds)), min(n, len(ds)))
#     return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False,
#                        collate_fn=collate_fn, num_workers=0, drop_last=False)


# # ── Args ──────────────────────────────────────────────────────────────────────

# def get_args():
#     p = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     p.add_argument("--dataset_root",    default="TCND_vn")
#     p.add_argument("--obs_len",         default=8,     type=int)
#     p.add_argument("--pred_len",        default=12,    type=int)
#     p.add_argument("--batch_size",      default=32,    type=int)
#     p.add_argument("--num_epochs",      default=120,   type=int)
#     p.add_argument("--learning_rate",   dest="learning_rate", default=1e-4,  type=float)
#     p.add_argument("--g_learning_rate", dest="learning_rate", default=1e-4,  type=float,
#                     help="Alias for --learning_rate (for backward compatibility)")
#     p.add_argument("--weight_decay",    default=1e-3,  type=float)
#     p.add_argument("--warmup_epochs",   default=5,     type=int)
#     p.add_argument("--grad_clip",       default=1.0,   type=float)
#     p.add_argument("--patience",        default=35,    type=int)
#     p.add_argument("--min_epochs",      default=20,    type=int)
#     p.add_argument("--use_amp",         action="store_true")
#     p.add_argument("--num_workers",     default=2,     type=int)
#     # Model args
#     p.add_argument("--sigma_min",       default=0.02,  type=float)
#     p.add_argument("--use_ot",          default=True,  action="store_true")
#     p.add_argument("--no_ot",           dest="use_ot", action="store_false")
#     p.add_argument("--cfg_guidance_scale", default=1.5, type=float)
#     # Eval args
#     p.add_argument("--val_freq",        default=3,     type=int)
#     p.add_argument("--val_subset_size", default=500,   type=int)
#     p.add_argument("--fast_ensemble",   default=20,    type=int)
#     p.add_argument("--fast_ddim",       default=15,    type=int)
#     p.add_argument("--full_ensemble",   default=50,    type=int)
#     p.add_argument("--full_ddim",       default=20,    type=int)
#     # EMA
#     p.add_argument("--use_ema",         default=True,  action="store_true")
#     p.add_argument("--no_ema",          dest="use_ema", action="store_false")
#     p.add_argument("--ema_decay",       default=0.995, type=float)
#     # Output
#     p.add_argument("--output_dir",      default="runs/v69")
#     p.add_argument("--gpu_num",         default="0")
#     # Data loader compat args
#     p.add_argument("--delim",           default=" ")
#     p.add_argument("--skip",            default=1,     type=int)
#     p.add_argument("--min_ped",         default=1,     type=int)
#     p.add_argument("--threshold",       default=0.002, type=float)
#     p.add_argument("--other_modal",     default="gph")
#     p.add_argument("--test_year",       default=None,  type=int)
#     # Resume
#     p.add_argument("--resume",          default=None)
#     p.add_argument("--resume_epoch",    default=None,  type=int)
#     p.add_argument("--eval_test_after_train", default=True, action="store_true")
#     return p.parse_args()


# # ── Main ──────────────────────────────────────────────────────────────────────

# def main(args):
#     if torch.cuda.is_available():
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
#     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output_dir, exist_ok=True)

#     print("=" * 72)
#     print(f"  TC-FlowMatching v69 — v59 base + adaptive difficulty")
#     print(f"  Single phase, no freeze/unfreeze")
#     print(f"  Adaptive: per-step weights (running_error), "
#           f"per-subloss weights (running_subloss)")
#     print(f"  Easy/difficult: per-sample reweight by actual ADE")
#     print(f"  Target: ADE<{TARGETS['ADE']}  72h<{TARGETS['72h']}  "
#           f"ATE<{TARGETS['ATE']}  CTE<{TARGETS['CTE']}")
#     print("=" * 72)

#     # ── Data ──────────────────────────────────────────────────────────────
#     trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"},
#                             test=False)
#     vd, vl = data_loader(args, {"root": args.dataset_root, "type": "val"},
#                            test=True)
#     from Model.data.trajectoriesWithMe_unet_training import seq_collate
#     vsub = mksub(vd, args.val_subset_size, args.batch_size, seq_collate)
#     print(f"  train:{len(trd)} seqs  val:{len(vd)} seqs")

#     # ── Model ─────────────────────────────────────────────────────────────
#     model = TCFlowMatching(
#         pred_len=args.pred_len, obs_len=args.obs_len,
#         sigma_min=args.sigma_min,
#         use_ema=args.use_ema, ema_decay=args.ema_decay,
#         use_ate_ot=args.use_ot,
#         cfg_guidance_scale=args.cfg_guidance_scale,
#     ).to(dev)
#     model.init_ema()
#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"  params: {n_params:,}")

#     # ── Optimizer ─────────────────────────────────────────────────────────
#     opt = optim.AdamW(model.parameters(), lr=args.learning_rate,
#                        weight_decay=args.weight_decay)
#     nstep = len(trl)
#     total_steps = nstep * args.num_epochs
#     warmup_steps = nstep * args.warmup_epochs
#     sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps,
#                                              min_lr=1e-6)
#     scaler = GradScaler("cuda", enabled=args.use_amp)
#     tracker = BestTracker(patience=args.patience)

#     # ── Resume ────────────────────────────────────────────────────────────
#     start_ep = 0
#     if args.resume and os.path.exists(args.resume):
#         print(f"  Loading: {args.resume}")
#         ck = torch.load(args.resume, map_location=dev)
#         m = _unwrap(model)
#         ms, us = m.load_state_dict(ck["model_state_dict"], strict=False)
#         if ms: print(f"  Missing keys: {len(ms)}")
#         if us: print(f"  Unexpected keys: {len(us)}")
#         # EMA
#         ema = getattr(m, "_ema", None)
#         if ema and ck.get("ema_shadow"):
#             for k, v in ck["ema_shadow"].items():
#                 if k in ema.shadow:
#                     ema.shadow[k].copy_(v.to(dev))
#         # Optimizer
#         try: opt.load_state_dict(ck["optimizer_state"])
#         except Exception as e: print(f"  Optimizer load: {e}")
#         # Scheduler
#         try: sched.load_state_dict(ck["scheduler_state"])
#         except:
#             for _ in range(ck.get("epoch", 0) * nstep):
#                 sched.step()
#         # Tracker bests
#         for src, dst in [("best_score", "best_score"), ("best_ade", "best_ade"),
#                           ("best_72h", "best_72h"), ("best_ate", "best_ate"),
#                           ("best_cte", "best_cte")]:
#             if src in ck:
#                 setattr(tracker, dst, ck[src])
#         start_ep = args.resume_epoch if args.resume_epoch else ck.get("epoch", 0) + 1
#         print(f"  → Resume from ep {start_ep}")

#     # ── Compile ───────────────────────────────────────────────────────────
#     try:
#         model = torch.compile(model, mode="reduce-overhead")
#         print("  torch.compile: enabled")
#     except:
#         pass

#     print(f"  Training: {nstep} steps/ep, start ep={start_ep}")
#     print("=" * 72)

#     # ── Training loop ─────────────────────────────────────────────────────
#     ts_start = time.perf_counter()

#     for ep in range(start_ep, args.num_epochs):
#         model.train()
#         sum_loss = 0.
#         t0 = time.perf_counter()

#         for i, batch in enumerate(trl):
#             bl = move(list(batch), dev)

#             with autocast(device_type="cuda", enabled=args.use_amp):
#                 bd = model.get_loss_breakdown(bl, epoch=ep)

#             opt.zero_grad()
#             scaler.scale(bd["total"]).backward()
#             scaler.unscale_(opt)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#             sb = scaler.get_scale()
#             scaler.step(opt)
#             scaler.update()
#             if scaler.get_scale() >= sb:
#                 sched.step()
#             model.ema_update()
#             sum_loss += bd["total"].item()

#             # ── Log every 20 steps ────────────────────────────────────────
#             if i % 20 == 0:
#                 lr = opt.param_groups[0]["lr"]
#                 print(
#                     f"  [{ep:>3}][{i:>3}/{nstep}]"
#                     f"  tot={bd['total'].item():.3f}"
#                     f"  fm={bd.get('fm_mse',0):.4f}"
#                     f"  dpe={bd.get('dpe',0):.4f}"
#                     f"  head={bd.get('heading',0):.4f}"
#                     f"  disp={bd.get('disp_smooth',0):.4f}"
#                     f"  sph={bd.get('sph_ate',0):.4f}"
#                     f"  aw72={bd.get('aw_72h',0):.1f}"
#                     f"  wD={bd.get('w_dpe',0):.2f}"
#                     f"  wS={bd.get('w_disp',0):.2f}"
#                     f"  lr={lr:.2e}"
#                 )

#         ep_time = time.perf_counter() - t0
#         avg_train = sum_loss / nstep

#         # ── Val loss ──────────────────────────────────────────────────────
#         model.eval()
#         val_sum = 0.
#         with torch.no_grad():
#             for batch in vl:
#                 bv = move(list(batch), dev)
#                 with autocast(device_type="cuda", enabled=args.use_amp):
#                     val_sum += model.get_loss(bv, epoch=ep).item()
#         avg_val = val_sum / len(vl)

#         print(f"  Epoch {ep:>3}  train={avg_train:.4f}  val={avg_val:.4f}  "
#               f"{ep_time:.0f}s")

#         # ── Fast eval (subset, fewer ensembles) ───────────────────────────
#         rf = evaluate(model, vsub, dev,
#                        tag=f"FAST ep{ep}",
#                        steps=args.fast_ddim,
#                        num_ensemble=args.fast_ensemble)
#         tracker.update(rf, model, args.output_dir, ep, opt, sched,
#                         avg_train, avg_val, tag="fast",
#                         min_ep=args.min_epochs)

#         # ── Full val every val_freq epochs ────────────────────────────────
#         if ep % args.val_freq == 0:
#             ema_obj = getattr(_unwrap(model), "_ema", None)

#             rr = evaluate(model, vl, dev,
#                            tag=f"RAW ep{ep}",
#                            steps=args.full_ddim,
#                            num_ensemble=args.full_ensemble)
#             tracker.update(rr, model, args.output_dir, ep, opt, sched,
#                             avg_train, avg_val, tag="raw",
#                             min_ep=args.min_epochs)

#             if ema_obj and ep >= 5:
#                 re = evaluate(model, vl, dev,
#                                tag=f"EMA ep{ep}",
#                                ema=ema_obj,
#                                steps=args.full_ddim,
#                                num_ensemble=args.full_ensemble)
#                 tracker.update(re, model, args.output_dir, ep, opt, sched,
#                                 avg_train, avg_val, tag="ema",
#                                 min_ep=args.min_epochs)

#         # ── Periodic checkpoint ───────────────────────────────────────────
#         if ep % 10 == 0 or ep == args.num_epochs - 1:
#             _save_ckpt(
#                 os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
#                 ep, model, opt, sched,
#                 tracker.best_score, tracker.best_ade, tracker.best_72h,
#                 tracker.best_ate, tracker.best_cte,
#                 avg_train, avg_val)

#         # ── Early stop ────────────────────────────────────────────────────
#         if tracker.stop:
#             print(f"  ⛔ Early stop at ep{ep}")
#             break

#     total_h = (time.perf_counter() - ts_start) / 3600.
#     print(f"\n  Training done in {total_h:.2f}h")
#     print(f"  Best: ADE={tracker.best_ade:.1f}  72h={tracker.best_72h:.0f}  "
#           f"ATE={tracker.best_ate:.1f}  CTE={tracker.best_cte:.1f}")

#     # ── Post-training test ────────────────────────────────────────────────
#     if args.eval_test_after_train:
#         print(f"\n{'='*72}\n  POST-TRAINING TEST\n{'='*72}")
#         try:
#             _, test_loader = data_loader(
#                 args, {"root": args.dataset_root, "type": "test"}, test=True)
#         except:
#             print("  ⚠ No test set, using val")
#             test_loader = vl

#         for fn, label in [("best_fast.pth", "FAST"),
#                            ("best_raw.pth", "RAW"),
#                            ("best_ema.pth", "EMA"),
#                            ("best_ade.pth", "ADE"),
#                            ("best_cte.pth", "CTE")]:
#             pp = os.path.join(args.output_dir, fn)
#             if not os.path.exists(pp):
#                 continue
#             ck = torch.load(pp, map_location=dev)
#             _unwrap(model).load_state_dict(
#                 ck["model_state_dict"], strict=False)
#             ema_obj = getattr(_unwrap(model), "_ema", None)
#             if ema_obj and ck.get("ema_shadow"):
#                 for k, v in ck["ema_shadow"].items():
#                     if k in ema_obj.shadow:
#                         ema_obj.shadow[k].copy_(v.to(dev))

#             r = evaluate(model, test_loader, dev,
#                           tag=f"TEST/{label}",
#                           steps=args.full_ddim,
#                           num_ensemble=args.full_ensemble)
#             print(f"\n  ── {label} TEST ──")
#             for key, ref in [("ADE", 172.68), ("72h", 321.39),
#                                ("ATE_mean", 142.21), ("CTE_mean", 42.04)]:
#                 v = r.get(key, float("nan"))
#                 mk = "BEAT!" if np.isfinite(v) and v < ref \
#                      else f" target:{ref:.0f}"
#                 print(f"    {key:<10}: {v:>8.1f} km  {mk}  "
#                       f"[gap:{v-ref:+.1f}]")

#     print("=" * 72)


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(42)
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
#     main(args)

"""
train_v69.py — TC-FlowMatching v69 Training Script
════════════════════════════════════════════════════════════════════════
3-phase training với fix từ log v68:

  Phase 1 (ep 0..SEL_WARMUP-1):
    Generator+Selector trains, selector LR=0 (effectively frozen)
    Loss: L_FM + 0.5*L_DPE + 0.02*L_consist

  Phase 2 (ep SEL_WARMUP..+14):
    [V69 FIX vs v68] Generator KHÔNG bị fully freeze
    Generator LR = g_lr * 0.1 (nhẹ, để modes vẫn update)
    Selector LR = g_lr * sel_lr_ratio (normal)
    Loss: 0.5*L_rank + 0.3*L_dir + 0.2*L_FM
    → modes di chuyển nhẹ → selector có signal để học
    → tránh plateau của v68 (train=1.42-1.49 không đổi)

  Phase 3 (ep SEL_WARMUP+15..end):
    All unfrozen, joint training
    Loss: L_FM + 0.4*L_DPE + 0.3*L_rank + 0.2*L_dir + 0.05*L_coh

KEY FIXES vs v68:
  - Không freeze generator Phase 2 (fix Phase 2 plateau)
  - OT bật ep>=20 (tránh đồng thời với step_w transition ep10)
  - _ot_cost._mb index fix (d[:,1:,:] không phải d[:,1,:])
  - step_weights: _curriculum_weights() on-the-fly, không nn.Parameter

HOW TO RUN:
  python scripts/train_v69.py \
      --dataset_root /kaggle/input/.../tc-ofm \
      --output_dir   runs/v69 \
      --batch_size   32 \
      --num_epochs   120 \
      --sel_warmup   20 \
      --use_amp
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, time, random
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader
from Model.flow_matching_model import TCFlowMatchingV69

try:
    from Model.utils import get_cosine_schedule_with_warmup
except ImportError:
    from torch.optim.lr_scheduler import CosineAnnealingLR
    def get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps, min_lr=1e-6):
        return CosineAnnealingLR(opt, T_max=total_steps, eta_min=min_lr)

TARGETS = {
    "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
    "12h": 65.42,  "24h": 104.67, "48h": 205.10,
}
R_EARTH = 6371.0


# ── Phase helpers ──────────────────────────────────────────────────

def _get_phase(epoch, sel_warmup):
    if epoch < sel_warmup:          return 1
    elif epoch < sel_warmup + 15:   return 2
    else:                            return 3


def _unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m

def _selector_params(model):
    return list(_unwrap(model).selector.parameters())

def _generator_params(model):
    """encoder + velocity_heads params."""
    m = _unwrap(model)
    gen = list(m.encoder.parameters()) + list(m.velocity_heads.parameters())
    return gen

def _freeze_selector(model):
    for p in _selector_params(model): p.requires_grad_(False)

def _unfreeze_selector(model):
    for p in _selector_params(model): p.requires_grad_(True)

def _freeze_generator(model):
    for p in _generator_params(model): p.requires_grad_(False)

def _unfreeze_generator(model):
    for p in _generator_params(model): p.requires_grad_(True)

def _selector_is_frozen(model):
    params = _selector_params(model)
    return len(params) > 0 and not params[0].requires_grad


def move(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x): out[i] = x.to(device)
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
    lat1=torch.deg2rad(p1[...,1]); lat2=torch.deg2rad(p2[...,1])
    dlat=torch.deg2rad(p2[...,1]-p1[...,1]); dlon=torch.deg2rad(p2[...,0]-p1[...,0])
    a=(torch.sin(dlat/2).pow(2)+torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2))
    return 2.0*R_EARTH*torch.asin(a.clamp(1e-12,1-1e-12).sqrt())

def _compute_ate_cte(pred_deg, gt_deg):
    T=min(pred_deg.shape[0],gt_deg.shape[0])
    if T<2:
        z=pred_deg.new_zeros(1,pred_deg.shape[1]); return z,z
    lon1=torch.deg2rad(gt_deg[:T-1,:,0]); lat1=torch.deg2rad(gt_deg[:T-1,:,1])
    lon2=torch.deg2rad(gt_deg[1:T, :,0]); lat2=torch.deg2rad(gt_deg[1:T, :,1])
    lon3=torch.deg2rad(pred_deg[1:T,:,0]); lat3=torch.deg2rad(pred_deg[1:T,:,1])
    y_a=torch.sin(lon2-lon1)*torch.cos(lat2)
    x_a=torch.cos(lat1)*torch.sin(lat2)-torch.sin(lat1)*torch.cos(lat2)*torch.cos(lon2-lon1)
    bear=torch.atan2(y_a,x_a)
    y_e=torch.sin(lon3-lon2)*torch.cos(lat3)
    x_e=torch.cos(lat2)*torch.sin(lat3)-torch.sin(lat2)*torch.cos(lat3)*torch.cos(lon3-lon2)
    bear_e=torch.atan2(y_e,x_e)
    tot=_haversine(pred_deg[1:T],gt_deg[1:T])
    angle=bear_e-bear
    return tot*torch.cos(angle), tot*torch.sin(angle)


class SimpleAccumulator:
    def __init__(self):
        self.dists=[]; self.ates=[]; self.ctes=[]
        self.step_dists=defaultdict(list)
        self._h={12:1,24:3,48:7,72:11}

    def update(self,dist,ate=None,cte=None):
        self.dists.extend(dist.mean(0).tolist())
        for h,s in self._h.items():
            if s<dist.shape[0]: self.step_dists[h].extend(dist[s].tolist())
        if ate is not None: self.ates.extend(ate.abs().mean(0).tolist())
        if cte is not None: self.ctes.extend(cte.abs().mean(0).tolist())

    def compute(self):
        r={"ADE":float(np.mean(self.dists)) if self.dists else float("nan"),
           "ATE_mean":float(np.mean(self.ates)) if self.ates else float("nan"),
           "CTE_mean":float(np.mean(self.ctes)) if self.ctes else float("nan"),
           "n_samples":len(self.dists)}
        for h in self._h:
            vals=self.step_dists.get(h,[])
            r[f"{h}h"]=float(np.mean(vals)) if vals else float("nan")
        return r


def _composite_score(r):
    ade=r.get("ADE",float("inf")); h72=r.get("72h",float("inf"))
    ate=r.get("ATE_mean",float("inf")); cte=r.get("CTE_mean",float("inf"))
    if not np.isfinite(ate): ate=ade*0.46
    if not np.isfinite(cte): cte=ade*0.53
    return 100.0*(0.05*(ade/136.0)+0.10*(r.get("12h",ade)/50.0)+
                  0.15*(r.get("24h",ade)/100.0)+0.20*(r.get("48h",ade)/200.0)+
                  0.25*(h72/300.0)+0.13*(ate/80.0)+0.12*(cte/94.0))

def _beat_str(r):
    parts=[]
    for k,t in [("ADE",172.68),("ATE_mean",142.21),("CTE_mean",42.04),
                ("72h",321.39),("12h",65.42),("24h",104.67),("48h",205.10)]:
        v=r.get(k,float("inf"))
        if np.isfinite(v) and v<t: parts.append(f"{k.replace('_mean','')}✅{v:.1f}")
    return "🏆 BEAT: "+" ".join(parts) if parts else ""

def _gap_str(r):
    parts=[]
    for k,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
        v=r.get(k,float("nan"))
        if np.isfinite(v):
            parts.append(f"{k.replace('_mean','')}:{v:.0f}({'↓' if v<ref else '↑'}{abs(v-ref):.0f})")
    return " | ".join(parts)


@torch.no_grad()
def evaluate(model, loader, device, tag="", ema_obj=None, ddim_steps=10):
    backup=None
    if ema_obj is not None:
        try: backup=ema_obj.apply_to(model)
        except: pass
    model.eval()
    acc=SimpleAccumulator(); t0=time.perf_counter(); n=0
    for batch in loader:
        bl=move(list(batch),device)
        result=model.sample(bl,ddim_steps=ddim_steps)
        pred=result[0]; gt=bl[1]
        T=min(pred.shape[0],gt.shape[0])
        pd=_norm_to_deg(pred[:T]); gd=_norm_to_deg(gt[:T])
        dist=_haversine(pd,gd); ate,cte=_compute_ate_cte(pd,gd)
        acc.update(dist,ate,cte); n+=1
    if backup is not None:
        try: ema_obj.restore(model,backup)
        except: pass
    r=acc.compute(); r["ms_per_batch"]=(time.perf_counter()-t0)*1e3/max(n,1)
    def _v(k): return r.get(k,float("nan"))
    def _m(v,t): return "✅" if np.isfinite(v) and v<t else "❌"
    elapsed=(time.perf_counter()-t0)
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
    beat=_beat_str(r)
    if beat: print(f"  {beat}")
    print(f"  Score={_composite_score(r):.2f}")
    print(f"{'='*68}\n")
    return r


def _save_ckpt(path,epoch,model,opt,sched,saver,tl,vl,extra=None):
    m=_unwrap(model); ema=getattr(m,"_ema",None); ema_sd=None
    if ema and hasattr(ema,"shadow"):
        try: ema_sd={k:v.cpu().clone() for k,v in ema.shadow.items()}
        except: pass
    payload={"epoch":epoch,"model_state_dict":m.state_dict(),
             "optimizer_state":opt.state_dict(),"scheduler_state":sched.state_dict(),
             "ema_shadow":ema_sd,"best_score":saver.best_score,
             "best_ade":saver.best_ade,"best_72h":saver.best_72h,
             "best_ate":saver.best_ate,"best_cte":saver.best_cte,
             "train_loss":tl,"val_loss":vl}
    if extra: payload.update(extra)
    torch.save(payload,path)


class BestSaver:
    def __init__(self,patience=35):
        self.patience=patience; self.counter=0; self.early_stop=False
        self.best_score=self.best_ade=self.best_72h=self.best_ate=self.best_cte=float("inf")

    def update(self,r,model,out_dir,epoch,opt,sched,tl,vl,tag="",min_epochs=25):
        score=_composite_score(r)
        ade=r.get("ADE",float("inf")); h72=r.get("72h",float("inf"))
        ate=r.get("ATE_mean",float("inf")); cte=r.get("CTE_mean",float("inf"))
        for v,attr,fname in [(ade,"best_ade","best_ade.pth"),(h72,"best_72h","best_72h.pth"),
                              (ate,"best_ate","best_ate.pth"),(cte,"best_cte","best_cte.pth")]:
            if v<getattr(self,attr):
                setattr(self,attr,v)
                _save_ckpt(os.path.join(out_dir,fname),epoch,model,opt,sched,self,tl,vl)
        if score<self.best_score:
            self.best_score=score; self.counter=0
            _save_ckpt(os.path.join(out_dir,f"best_{tag or 'composite'}.pth"),
                       epoch,model,opt,sched,self,tl,vl,
                       {"score":score,"ade":ade,"h72":h72,"ate":ate,"cte":cte})
            print(f"  ✅ Best {tag} score={score:.2f}  ADE={ade:.1f}  "
                  f"72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}  (ep{epoch})")
        else:
            self.counter+=1
            print(f"  No improve {self.counter}/{self.patience}  "
                  f"(best={self.best_score:.2f} cur={score:.2f})")
        if epoch>=min_epochs and self.counter>=self.patience:
            self.early_stop=True


def make_val_subset(val_ds,size,bs,collate_fn):
    idx=random.Random(42).sample(range(len(val_ds)),min(size,len(val_ds)))
    return DataLoader(Subset(val_ds,idx),batch_size=bs,shuffle=False,
                      collate_fn=collate_fn,num_workers=0,drop_last=False)


def get_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root",    default="TCND_vn")
    p.add_argument("--obs_len",         default=8,     type=int)
    p.add_argument("--pred_len",        default=12,    type=int)
    p.add_argument("--batch_size",      default=32,    type=int)
    p.add_argument("--num_epochs",      default=120,   type=int)
    p.add_argument("--g_learning_rate", default=1e-4,  type=float)
    p.add_argument("--sel_lr_ratio",    default=0.10,  type=float)
    p.add_argument("--weight_decay",    default=1e-3,  type=float)
    p.add_argument("--warmup_epochs",   default=5,     type=int)
    p.add_argument("--grad_clip",       default=1.0,   type=float)
    p.add_argument("--patience",        default=35,    type=int)
    p.add_argument("--min_epochs",      default=30,    type=int)
    p.add_argument("--use_amp",         action="store_true")
    p.add_argument("--num_workers",     default=2,     type=int)
    p.add_argument("--sigma_min",       default=0.02,  type=float)
    p.add_argument("--head_noise_base", default=0.03,  type=float)
    p.add_argument("--use_ot",          default=True,  action="store_true")
    p.add_argument("--no_ot",           dest="use_ot", action="store_false")
    p.add_argument("--cfg_guidance_scale", default=1.3, type=float)
    p.add_argument("--sel_warmup",      default=20,    type=int,
                   help="Phase 1 ends at this epoch (selector frozen 0..sel_warmup-1)")
    p.add_argument("--val_freq",        default=3,     type=int)
    p.add_argument("--val_subset_size", default=500,   type=int)
    p.add_argument("--fast_ddim",       default=10,    type=int,
                   help="ddim_steps for fast eval (per epoch)")
    p.add_argument("--full_ddim",       default=20,    type=int,
                   help="ddim_steps for full val eval")
    p.add_argument("--use_ema",         default=True,  action="store_true")
    p.add_argument("--no_ema",          dest="use_ema",action="store_false")
    p.add_argument("--ema_decay",       default=0.995, type=float)
    p.add_argument("--output_dir",      default="runs/v67")
    p.add_argument("--gpu_num",         default="0")
    p.add_argument("--delim",           default=" ")
    p.add_argument("--skip",            default=1,     type=int)
    p.add_argument("--min_ped",         default=1,     type=int)
    p.add_argument("--threshold",       default=0.002, type=float)
    p.add_argument("--other_modal",     default="gph")
    p.add_argument("--test_year",       default=None,  type=int)
    p.add_argument("--resume",          default=None)
    p.add_argument("--resume_epoch",    default=None,  type=int)
    p.add_argument("--eval_test_after_train", default=True, action="store_true")
    return p.parse_args()


def main(args):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_num)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir,exist_ok=True)

    SEL_WARMUP=args.sel_warmup

    print("="*72)
    print("  TC-FlowMatching v69  —  Stable, curriculum weights, 3-Phase")
    print(f"  Phase 1 (ep 0–{SEL_WARMUP-1}):    Generator, selector LR=0")
    print(f"  Phase 2 (ep {SEL_WARMUP}–{SEL_WARMUP+14}): Selector+Generator(LR*0.05)")
    print(f"  Phase 3 (ep {SEL_WARMUP+15}+):  Full joint")
    print(f"  OT: epoch>=20 (v69 fix: avoid simultaneous with step_w transition)")
    print(f"  ST-Trans target: ADE<{TARGETS['ADE']}  72h<{TARGETS['72h']}"
          f"  ATE<{TARGETS['ATE']}  CTE<{TARGETS['CTE']}")
    print("="*72)

    train_ds,train_loader=data_loader(args,{"root":args.dataset_root,"type":"train"},test=False)
    val_ds,val_loader=data_loader(args,{"root":args.dataset_root,"type":"val"},test=True)
    from Model.data.trajectoriesWithMe_unet_training import seq_collate
    val_sub=make_val_subset(val_ds,args.val_subset_size,args.batch_size,seq_collate)
    print(f"  train: {len(train_ds)} seqs  val: {len(val_ds)} seqs")

    model=TCFlowMatchingV69(
        pred_len=args.pred_len,obs_len=args.obs_len,
        sigma_min=args.sigma_min,use_ema=args.use_ema,ema_decay=args.ema_decay,
        use_ot=args.use_ot,head_noise_base=args.head_noise_base,
        cfg_guidance_scale=args.cfg_guidance_scale,
        selector_warmup=SEL_WARMUP,
    ).to(device)
    model.init_ema()
    n_p=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_p:,}")

    optimizer=optim.AdamW([
        {"params":_generator_params(model),"lr":args.g_learning_rate,"name":"generator"},
        {"params":_selector_params(model),"lr":args.g_learning_rate*args.sel_lr_ratio,"name":"selector"},
    ],weight_decay=args.weight_decay)

    saver=BestSaver(patience=args.patience)
    scaler=GradScaler("cuda",enabled=args.use_amp)
    steps_ep=len(train_loader)
    total_s=steps_ep*args.num_epochs
    warmup_s=steps_ep*args.warmup_epochs
    scheduler=get_cosine_schedule_with_warmup(optimizer,warmup_s,total_s,min_lr=1e-6)

    start_epoch=0
    if args.resume and os.path.exists(args.resume):
        print(f"  Loading checkpoint: {args.resume}")
        ckpt=torch.load(args.resume,map_location=device)
        m=_unwrap(model)
        miss,unexp=m.load_state_dict(ckpt["model_state_dict"],strict=False)
        if miss:  print(f"  Missing: {len(miss)}")
        if unexp: print(f"  Unexpected: {len(unexp)}")
        ema=getattr(m,"_ema",None)
        if ema and ckpt.get("ema_shadow"):
            for k,v in ckpt["ema_shadow"].items():
                if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
        try: optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e: print(f"  Opt restore: {e}")
        try: scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception as e:
            for _ in range(ckpt.get("epoch",0)*steps_ep): scheduler.step()
        for attr in ("best_score","best_ade","best_72h","best_ate","best_cte"):
            if attr in ckpt: setattr(saver,attr,ckpt[attr])
        start_epoch=(args.resume_epoch if args.resume_epoch is not None
                     else ckpt.get("epoch",0)+1)
        print(f"  → Resuming from epoch {start_epoch}")

    try:
        model=torch.compile(model,mode="reduce-overhead")
        print("  torch.compile: enabled")
    except Exception:
        pass

    # oracle_ade_ema: update từ oracle_ade_km THỰC (km) từ loss dict
    oracle_ade_ema=999.0
    _pa=0.05  # EMA smoothing

    prev_phase=-1; last_tl=last_vl=0.0
    train_start=time.perf_counter()
    print(f"  Training: {steps_ep} steps/epoch, start ep={start_epoch}")
    print("="*72)

    for epoch in range(start_epoch,args.num_epochs):
        phase=_get_phase(epoch,SEL_WARMUP)

        # Phase transitions — freeze/unfreeze
        # V69 FIX: Phase 2 KHÔNG fully freeze generator
        # → dùng LR thấp thay vì requires_grad=False
        if phase==1:
            if not _selector_is_frozen(model):
                _freeze_selector(model)
                _unfreeze_generator(model)
                print(f"  🔒 ep{epoch}: Selector FROZEN, Generator active (Phase 1)")
        elif phase==2:
            if prev_phase==1:
                _unfreeze_selector(model)
                _unfreeze_generator(model)  # V69: không freeze, dùng LR thấp
                print(f"  🔀 ep{epoch}: Selector+Generator active (Phase 2, gen LR reduced)")
        else:
            if prev_phase==2:
                _unfreeze_generator(model)
                print(f"  🔓 ep{epoch}: All UNFROZEN (Phase 3 Joint)")

        # LR per phase
        for pg in optimizer.param_groups:
            if pg.get("name")=="selector":
                if phase==1:   pg["lr"]=0.0
                elif phase==2: pg["lr"]=args.g_learning_rate*args.sel_lr_ratio
                else:          pg["lr"]=args.g_learning_rate*0.50
            elif pg.get("name")=="generator":
                if phase==1:
                    # Cosine schedule
                    pass  # scheduler handles it
                elif phase==2:
                    # V69 FIX: generator nhẹ, không freeze
                    # LR rất thấp: modes vẫn update nhưng không overwrite selector signal
                    pg["lr"]=args.g_learning_rate*0.05
                else:
                    pass  # scheduler handles it

        if phase!=prev_phase:
            print(f"\n  ┌── Phase {phase}/3  ep={epoch}"
                  f"  oracle_ema={oracle_ade_ema:.1f}km"
                  f"  tau={TCFlowMatchingV69._tau(epoch):.0f}")
            prev_phase=phase

        # ── Train one epoch ───────────────────────────────────────
        model.train()
        sum_loss=0.0; t0=time.perf_counter()

        for i,batch in enumerate(train_loader):
            bl=move(list(batch),device)
            with autocast(device_type="cuda",enabled=args.use_amp):
                bd=model.get_loss_breakdown(bl,epoch=epoch)

            optimizer.zero_grad()
            scaler.scale(bd["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
            sc_before=scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale()>=sc_before: scheduler.step()
            model.ema_update()
            sum_loss+=bd["total"].item()

            # Update oracle_ade_ema từ km thực
            oracle_km=bd.get("oracle_ade_km",oracle_ade_ema)
            if np.isfinite(float(oracle_km)):
                oracle_ade_ema=(1-_pa)*oracle_ade_ema+_pa*float(oracle_km)

            if i%20==0:
                # LR hiển thị
                lr_g=next((pg["lr"] for pg in optimizer.param_groups
                            if pg.get("name")=="generator"),0.0)
                lr_s=next((pg["lr"] for pg in optimizer.param_groups
                            if pg.get("name")=="selector"),0.0)
                phase_icon="🔒" if phase==1 else ("🔀" if phase==2 else "🔓")
                print(
                    f"  [P{phase}][{epoch:>3}][{i:>3}/{steps_ep}]"
                    f"  tot={bd['total'].item():.3f}"
                    f"  fm={bd.get('L_FM',0):.4f}"
                    f"  dpe={bd.get('L_dpe',0):.4f}"
                    f"  rank={bd.get('L_rank',0):.4f}"
                    f"  dir={bd.get('L_dir',0):.4f}"
                    f"  oracle={bd.get('oracle_ade_km',0):.1f}km"
                    f"  δ={bd.get('delta_mean',0):.2f}"
                    f"  div={bd.get('min_dist_mean',0):.3f}"
                    f"  sw12={bd.get('sw_12h',0):.2f}"
                    f"  sw72={bd.get('sw_72h',0):.2f}"
                    f"  {phase_icon}  lr_g={lr_g:.2e}  lr_s={lr_s:.2e}"
                )

        ep_s=time.perf_counter()-t0
        avg_t=sum_loss/steps_ep

        # Val loss
        model.eval(); val_loss=0.0
        with torch.no_grad():
            for batch in val_loader:
                bv=move(list(batch),device)
                with autocast(device_type="cuda",enabled=args.use_amp):
                    val_loss+=model.get_loss(bv,epoch=epoch).item()
        avg_vl=val_loss/len(val_loader)
        last_tl=avg_t; last_vl=avg_vl

        print(f"  Epoch {epoch:>3}  P{phase}  "
              f"train={avg_t:.4f}  val={avg_vl:.4f}  {ep_s:.0f}s  "
              f"oracle_ema={oracle_ade_ema:.1f}km")

        # Fast eval
        m_fast=evaluate(model,val_sub,device,
                        tag=f"FAST/P{phase} ep{epoch}",
                        ddim_steps=args.fast_ddim)
        saver.update(m_fast,model,args.output_dir,epoch,
                     optimizer,scheduler,avg_t,avg_vl,tag="fast")

        # Full val every val_freq epochs
        if epoch%args.val_freq==0:
            ema=getattr(_unwrap(model),"_ema",None)
            r_raw=evaluate(model,val_loader,device,
                           tag=f"RAW/P{phase} ep{epoch}",
                           ddim_steps=args.full_ddim)
            saver.update(r_raw,model,args.output_dir,epoch,
                         optimizer,scheduler,avg_t,avg_vl,tag="raw")
            if ema is not None and epoch>=5:
                r_ema=evaluate(model,val_loader,device,
                               tag=f"EMA/P{phase} ep{epoch}",
                               ema_obj=ema,ddim_steps=args.full_ddim)
                saver.update(r_ema,model,args.output_dir,epoch,
                             optimizer,scheduler,avg_t,avg_vl,tag="ema")

        if epoch%10==0 or epoch==args.num_epochs-1:
            _save_ckpt(
                os.path.join(args.output_dir,f"ckpt_ep{epoch:03d}.pth"),
                epoch,model,optimizer,scheduler,saver,avg_t,avg_vl,
                {"oracle_ade_ema":oracle_ade_ema,"phase":phase})

        if saver.early_stop:
            print(f"  ⛔ Early stopping at epoch {epoch}")
            break

    total_h=(time.perf_counter()-train_start)/3600
    print(f"\n  Best: ADE={saver.best_ade:.1f}  72h={saver.best_72h:.0f}"
          f"  ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}"
          f"  ({total_h:.2f}h)")

    if args.eval_test_after_train:
        print("\n"+"="*72)
        print("  POST-TRAINING TEST EVALUATION")
        print("="*72)
        try:
            test_ds,test_loader=data_loader(args,{"root":args.dataset_root,"type":"test"},test=True)
        except Exception as e:
            print(f"  ⚠  Test unavailable: {e} — using val"); test_loader=val_loader

        for ckpt_name,label in [("best_composite.pth","COMPOSITE"),("best_72h.pth","72h"),
                                  ("best_cte.pth","CTE"),("best_ema.pth","EMA")]:
            p=os.path.join(args.output_dir,ckpt_name)
            if not os.path.exists(p): continue
            ckpt=torch.load(p,map_location=device)
            _unwrap(model).load_state_dict(ckpt["model_state_dict"],strict=False)
            ema=getattr(_unwrap(model),"_ema",None)
            if ema and ckpt.get("ema_shadow"):
                for k,v in ckpt["ema_shadow"].items():
                    if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
            r=evaluate(model,test_loader,device,
                       tag=f"TEST/{label}",ddim_steps=args.full_ddim)
            print(f"\n  ── {label} TEST ──")
            for key,ref in [("ADE",172.68),("72h",321.39),("ATE_mean",142.21),("CTE_mean",42.04)]:
                v=r.get(key,float("nan"))
                mark="✅ BEAT!" if np.isfinite(v) and v<ref else f"❌ target:{ref:.0f}"
                print(f"    {key:<10}: {v:>8.1f} km  {mark}  [gap:{v-ref:+.1f}]")
    print("="*72)


if __name__=="__main__":
    args=get_args()
    np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    main(args)
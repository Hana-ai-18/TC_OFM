"""
scripts/train_fm.py  ——  TC-FlowMatching v2.4
  Core v2.1 + augmentation mạnh (rotation/mixup) + exp step weights
  XAI: attribution, hard_score components, physics score, uncertainty
"""
from __future__ import annotations

import sys, os
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse, math, time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler

from Model.data.loader_training import data_loader
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _haversine_deg,
    EMAModel, augment_batch, hard_score_from_obs,
    compute_obs_attribution, compute_ensemble_uncertainty,
    load_checkpoint_compat,
)

HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
ST_TRANS_VAL  = {"ADE": 172.68, "ATE": 142.21, "CTE": 42.04,
                 "12h": 65.42, "24h": 104.67, "48h": 205.10, "72h": 321.39}
ST_TRANS_TEST = {"ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
                 "12h": 77.5, "24h": 130.5, "48h": 269.9, "72h": 423.3}


def _unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m

def move(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x): out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
    return out

def _ate_cte(pred_deg, gt_deg):
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2: z = pred_deg.new_zeros(1, pred_deg.shape[1]); return z, z
    lo1=torch.deg2rad(gt_deg[:T-1,:,0]); la1=torch.deg2rad(gt_deg[:T-1,:,1])
    lo2=torch.deg2rad(gt_deg[1:T,:,0]);  la2=torch.deg2rad(gt_deg[1:T,:,1])
    lo3=torch.deg2rad(pred_deg[1:T,:,0]);la3=torch.deg2rad(pred_deg[1:T,:,1])
    ya=torch.sin(lo2-lo1)*torch.cos(la2)
    xa=torch.cos(la1)*torch.sin(la2)-torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
    be=torch.atan2(ya,xa)
    ye=torch.sin(lo3-lo2)*torch.cos(la3)
    xe=torch.cos(la2)*torch.sin(la3)-torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
    bee=torch.atan2(ye,xe)
    tot=_haversine_deg(pred_deg[1:T],gt_deg[1:T]); ang=bee-be
    return tot*torch.cos(ang), tot*torch.sin(ang)


# ─────────────────────────────────────────────────────────────────────────────
#  2-group optimizer + scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model, lr_vel, lr_enc, weight_decay):
    raw = _unwrap(model)
    enc_params = list(raw.encoder.parameters())
    vel_params = list(raw.velocity.parameters())
    return optim.AdamW([
        {"params": enc_params,  "lr": lr_enc, "name": "encoder"},
        {"params": vel_params,  "lr": lr_vel, "name": "velocity"},
    ], weight_decay=weight_decay)


class TwoGroupScheduler:
    def __init__(self, opt, warmup_ep, total_ep, lr_vel, lr_enc,
                 lr_min, freeze_enc_ep, enc_warmup_ep):
        self.opt = opt; self.warmup = warmup_ep; self.total = total_ep
        self.lr_vel = lr_vel; self.lr_enc = lr_enc; self.lr_min = lr_min
        self.freeze_enc = freeze_enc_ep; self.enc_warmup = enc_warmup_ep
        self.epoch = 0

    def _cos(self, ep_s, ep_e, lr_s, lr_e, ep):
        if ep <= ep_s: return lr_s
        if ep >= ep_e: return lr_e
        t = (ep - ep_s) / (ep_e - ep_s)
        return lr_e + 0.5*(lr_s - lr_e)*(1 + math.cos(math.pi * t))

    def step(self):
        ep = self.epoch
        lr_vel = (self.lr_vel * max(0.1, ep/max(self.warmup-1,1)) if ep < self.warmup
                  else self._cos(self.warmup, self.total, self.lr_vel, self.lr_min, ep))
        enc_start = self.freeze_enc + self.enc_warmup
        if ep < self.freeze_enc:
            lr_enc = 0.0
        elif ep < enc_start:
            lr_enc = self.lr_enc * (ep - self.freeze_enc) / max(self.enc_warmup, 1)
        else:
            lr_enc = self._cos(enc_start, self.total, self.lr_enc, self.lr_min, ep)
        for pg in self.opt.param_groups:
            pg["lr"] = lr_enc if pg.get("name") == "encoder" else lr_vel
        self.epoch += 1
        return lr_vel, lr_enc

    def get_lrs(self):
        d = {pg.get("name","?"): pg["lr"] for pg in self.opt.param_groups}
        return d.get("velocity", 0.0), d.get("encoder", 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, tag="", ema=None, ref_targets=None,
             use_tta=False, n_tta=8, epoch_for_loss=9999,
             run_xai=False, xai_batch=None) -> Dict:
    bk = None
    if ema is not None:
        try: bk = ema.apply_to(model)
        except Exception as e: print(f"  ⚠ EMA: {e}")

    model.eval()
    all_ade, all_ate, all_cte = [], [], []
    step_dist = defaultdict(list)
    sum_loss = 0.0; sum_n = 0

    for batch in loader:
        bl = move(list(batch), device)
        gt = bl[1]; B = bl[0].shape[1]

        try:
            bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
            if torch.isfinite(bd["total"]):
                sum_loss += bd["total"].item() * B; sum_n += B
        except Exception:
            pass

        if use_tta:
            obs  = bl[0]
            anchor = obs[-1:, :, :2]   # [1, B, 2] — BUG-3 FIXED
            preds_tta = []
            for tta_i in range(n_tta):
                if tta_i == 0:
                    bl_tta = bl
                else:
                    angle = (tta_i / (n_tta-1) - 0.5) * (math.pi/18)
                    cos_a, sin_a = math.cos(angle), math.sin(angle)
                    rot = torch.tensor([[cos_a,-sin_a],[sin_a,cos_a]],
                                       dtype=obs.dtype, device=device)
                    T_o = obs.shape[0]
                    rel_obs = (obs[..., :2] - anchor).reshape(T_o*B, 2)
                    obs_rot = obs.clone()
                    obs_rot[..., :2] = (rot @ rel_obs.T).T.reshape(T_o, B, 2) + anchor
                    bl_tta = list(bl); bl_tta[0] = obs_rot

                try:
                    pred, _, _ = model.sample(bl_tta)
                except Exception as e:
                    print(f"  TTA error: {e}"); continue

                if tta_i > 0:
                    rot_inv = rot.T
                    T_p = pred.shape[0]
                    rel_p = (pred - anchor).reshape(T_p*B, 2)
                    pred  = (rot_inv @ rel_p.T).T.reshape(T_p, B, 2) + anchor
                preds_tta.append(pred)

            if not preds_tta: continue
            weights = [2.0 if i == 0 else 1.0 for i in range(len(preds_tta))]
            total_w = sum(weights)
            pred = sum(w/total_w * p for w, p in zip(weights, preds_tta))
        else:
            try:
                pred, _, _ = model.sample(bl)
            except Exception as e:
                print(f"  sample error: {e}"); continue

        T        = min(pred.shape[0], gt.shape[0])
        pred_deg = _norm_to_deg(pred[:T]); gt_deg = _norm_to_deg(gt[:T])
        dist     = _haversine_deg(pred_deg, gt_deg)
        ate, cte = _ate_cte(pred_deg, gt_deg)

        all_ade.extend(dist.mean(0).tolist())
        if ate.shape[0] > 0:
            all_ate.extend(ate.abs().mean(0).tolist())
            all_cte.extend(cte.abs().mean(0).tolist())
        for h, s in HORIZON_STEPS.items():
            if s < T: step_dist[h].extend(dist[s].tolist())

    if bk is not None:
        try: ema.restore(model, bk)
        except Exception: pass

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")
    val_loss = sum_loss / max(sum_n, 1)
    result = {"ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
              "n": len(all_ade), "val_loss": val_loss}
    for h in HORIZON_STEPS: result[f"{h}h"] = _m(step_dist[h])

    ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
    result["combined"] = (0.5*ade + 0.3*ate_ + 0.2*cte_
                          if all(np.isfinite(x) for x in [ade,ate_,cte_]) else ade)

    ref = ref_targets or ST_TRANS_VAL
    def _v(k): return result.get(k, float("nan"))
    def _ok(k): return "✓" if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9) else "✗"

    tta_str = " [TTA]" if use_tta else ""
    print(f"\n  {'='*72}")
    print(f"  [{tag}]{tta_str}  n={result['n']}")
    print(f"  Val Loss : {val_loss:.6f}")
    print(f"  ADE={_v('ADE'):7.1f}km {_ok('ADE')}  ATE={_v('ATE'):7.1f}km {_ok('ATE')}"
          f"  CTE={_v('CTE'):7.1f}km {_ok('CTE')}")
    print(f"  Combined = {_v('combined'):.1f}")
    print(f"  12h={_v('12h'):6.1f}  24h={_v('24h'):6.1f}"
          f"  48h={_v('48h'):6.1f}  72h={_v('72h'):6.1f} km")
    beat = [f"{k}={_v(k):.0f}<{ref.get(k,999):.0f}"
            for k in ["ADE","ATE","CTE","12h","24h","48h","72h"]
            if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9)]
    print(f"  BEAT: {' | '.join(beat) if beat else 'none'}")

    if run_xai and xai_batch is not None:
        try:
            model.train()  # BUG-C fix: không dùng eval() cho XAI
            with torch.no_grad():
                _, _, all_t, xai = _unwrap(model).sample(xai_batch, return_xai=True)
            print(f"  ── XAI Summary {'─'*48}")
            print(f"  [XAI-4] Uncertainty: 12h={xai['mean_12h_std']:.1f}km  "
                  f"72h={xai['mean_72h_std']:.1f}km  "
                  f"ratio={float(xai['uncertainty_ratio'].mean()):.2f}x")
            pc = xai.get("physics_components", {})
            if pc:
                print(f"  [XAI-3] Physics(best): speed={float(pc['speed'].mean()):.3f}  "
                      f"smooth={float(pc['smooth'].mean()):.3f}  "
                      f"heading={float(pc['heading'].mean()):.3f}")
            hc = xai.get("hard_components", {})
            if hc:
                print(f"  [XAI-2] HardScore: curvature={float(hc['curvature'].mean()):.3f}  "
                      f"speed_var={float(hc['speed_var'].mean()):.3f}  "
                      f"dir_change={float(hc['dir_change'].mean()):.3f}")
            sc = xai.get("speed_comparison", {})
            if sc:
                ratio = sc.get("speed_ratio", 1.0)
                flag  = "⚠ OVER" if ratio > 1.15 else ("⚠ UNDER" if ratio < 0.85 else "✓ OK")
                print(f"  [XAI-5] Speed: obs={sc['obs_speed_mean']:.1f}km/h  "
                      f"pred={sc['pred_speed_mean']:.1f}km/h  "
                      f"ratio={ratio:.2f} {flag}")
            print(f"  {'─'*60}")
            result["xai"] = xai
        except Exception as e:
            print(f"  XAI error: {e}")

    print(f"  {'='*72}\n")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _save(path, epoch, model, opt, sched, best_score,
          ema=None, scaler=None, patience_cnt=0, extra=None):
    m   = _unwrap(model)
    esd = None
    if ema is not None:
        try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
        except Exception: pass
    payload = {"epoch": epoch, "model": m.state_dict(), "ema": esd,
               "optimizer": opt.state_dict(), "scheduler_ep": sched.epoch,
               "scaler": scaler.state_dict() if scaler else None,
               "best_score": best_score, "patience_cnt": patience_cnt}
    if extra: payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, model, opt, sched, ema, scaler, device):
    # Dùng compat loader để handle vel_obs_enc 6→7 feature expansion
    ck = load_checkpoint_compat(path, _unwrap(model), device)
    print(f"  Loaded: {path}")
    if "optimizer" in ck:
        try: opt.load_state_dict(ck["optimizer"])
        except Exception as e: print(f"  ⚠ Optimizer: {e}")
    sched.epoch = ck.get("scheduler_ep", ck.get("scheduler", 0))
    if scaler and ck.get("scaler"):
        try: scaler.load_state_dict(ck["scaler"])
        except Exception: pass
    if ema and ck.get("ema"):
        for k, v in ck["ema"].items():
            if k in ema.shadow: ema.shadow[k].copy_(v.to(device))
    return (ck.get("epoch", 0) + 1,
            ck.get("best_score", ck.get("best_ade", float("inf"))),
            ck.get("patience_cnt", 0))


def ensemble_checkpoints(ckpt_paths, model, device):
    """[BUG-4 FIXED] Chỉ ensemble periodic ckpts."""
    if not ckpt_paths: return
    sds = []
    for p in ckpt_paths:
        if not os.path.exists(p): continue
        try:
            ck = torch.load(p, map_location=device)
            sds.append(ck.get("model", ck))
            print(f"  Ensemble: {os.path.basename(p)}")
        except Exception as e:
            print(f"  ⚠ {p}: {e}")
    if len(sds) < 2: return
    avg_sd = {}
    for key in sds[0]:
        try: avg_sd[key] = sum(sd[key].float() for sd in sds) / len(sds)
        except Exception: avg_sd[key] = sds[0][key]
    _unwrap(model).load_state_dict(avg_sd, strict=False)
    print(f"  Ensemble: averaged {len(sds)} ckpts")


# ─────────────────────────────────────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root",      default="TCND_vn")
    p.add_argument("--obs_len",           default=8,     type=int)
    p.add_argument("--pred_len",          default=12,    type=int)
    p.add_argument("--num_workers",       default=2,     type=int)
    p.add_argument("--other_modal",       default="gph")
    p.add_argument("--delim",             default=" ")
    p.add_argument("--skip",              default=1,     type=int)
    p.add_argument("--min_ped",           default=1,     type=int)
    p.add_argument("--threshold",         default=0.002, type=float)
    # Model
    p.add_argument("--d_cond",            default=256,   type=int)
    p.add_argument("--d_model",           default=256,   type=int)
    p.add_argument("--nhead",             default=8,     type=int)
    p.add_argument("--num_dec_layers",    default=4,     type=int)
    p.add_argument("--dim_ff",            default=512,   type=int)
    p.add_argument("--dropout",           default=0.1,   type=float)
    p.add_argument("--unet_in_ch",        default=13,    type=int)
    p.add_argument("--sigma_min",         default=0.04,  type=float)
    p.add_argument("--sigma_max",         default=0.08,  type=float)
    p.add_argument("--lambda_reg",        default=0.2,   type=float)
    p.add_argument("--use_ot",            default=True,  action="store_true")
    p.add_argument("--no_ot",             dest="use_ot", action="store_false")
    p.add_argument("--ot_epsilon",        default=0.05,  type=float)
    p.add_argument("--n_ensemble",        default=20,    type=int)
    p.add_argument("--sigma_inference",   default=0.04,  type=float)
    p.add_argument("--n_inference_steps", default=1,     type=int)
    # Training
    p.add_argument("--num_epochs",        default=150,   type=int)
    p.add_argument("--batch_size",        default=64,    type=int)
    p.add_argument("--lr",                default=1e-4,  type=float)
    p.add_argument("--lr_enc",            default=2e-5,  type=float)
    p.add_argument("--lr_min",            default=1e-6,  type=float)
    p.add_argument("--warmup_epochs",     default=5,     type=int)
    p.add_argument("--freeze_enc_epochs", default=5,     type=int)
    p.add_argument("--enc_warmup_epochs", default=3,     type=int)
    p.add_argument("--weight_decay",      default=1e-4,  type=float)
    p.add_argument("--grad_clip",         default=1.0,   type=float)
    p.add_argument("--use_amp",           default=False, action="store_true")
    p.add_argument("--use_ema",           default=True,  action="store_true")
    p.add_argument("--no_ema",            dest="use_ema",action="store_false")
    # Eval
    p.add_argument("--val_freq",          default=5,     type=int)     # giảm từ 10 → 5
    p.add_argument("--patience",          default=30,    type=int)
    p.add_argument("--min_ep",            default=10,    type=int)
    # IO
    p.add_argument("--output_dir",        default="runs/fm_v24")
    p.add_argument("--gpu_num",           default="0")
    p.add_argument("--resume",            default=None)
    p.add_argument("--test_at_end",       default=True,  action="store_true")
    p.add_argument("--no_test",           dest="test_at_end", action="store_false")
    p.add_argument("--tta_test",          default=True,  action="store_true")
    p.add_argument("--n_tta",             default=8,     type=int)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt = os.path.join(args.output_dir, "best_model.pth")
    last_ckpt = os.path.join(args.output_dir, "last_model.pth")

    print("=" * 72)
    print("  TC-FlowMatching v2.4 — v2.1 core + aug mạnh hơn + exp step weights")
    print(f"  sigma_max={args.sigma_max}  sigma_min={args.sigma_min}")
    print(f"  lambda_reg={args.lambda_reg}  L_reg ramp: ep10→ep30")
    print(f"  Encoder freeze {args.freeze_enc_epochs}ep + warmup {args.enc_warmup_epochs}ep")
    print(f"  n_inference_steps={args.n_inference_steps} (1-step Euler)")
    print(f"  XAI: hard_score components, physics score, uncertainty")
    print("=" * 72)

    trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
    vd, val_loader = data_loader(args, {"root": args.dataset_root, "type": "val"}, test=True)
    print(f"  train: {len(trd)} ({len(trl)} batches/ep)")
    # Lấy 1 batch val để dùng cho XAI logging
    _xai_batch_raw = next(iter(val_loader))
    xai_batch = move(list(_xai_batch_raw), device) if _xai_batch_raw else None
    print(f"  val  : {len(vd)} ({len(val_loader)} batches)")

    model = TCFlowMatching(
        pred_len=args.pred_len, obs_len=args.obs_len,
        unet_in_ch=args.unet_in_ch, d_cond=args.d_cond,
        d_model=args.d_model, nhead=args.nhead,
        num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
        dropout=args.dropout, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        lambda_reg=args.lambda_reg, use_ot=args.use_ot, ot_epsilon=args.ot_epsilon,
        use_ema=args.use_ema, n_ensemble=args.n_ensemble,
        sigma_inference=args.sigma_inference,
        n_inference_steps=args.n_inference_steps,
    ).to(device)

    model.init_ema()
    ema = getattr(_unwrap(model), "_ema", None)
    raw = _unwrap(model)
    n_enc = sum(p.numel() for p in raw.encoder.parameters())
    n_vel = sum(p.numel() for p in raw.velocity.parameters())
    print(f"  Encoder: {n_enc:,}  Velocity: {n_vel:,}  Total: {n_enc+n_vel:,}")

    opt    = build_optimizer(model, args.lr, args.lr_enc, args.weight_decay)
    scaler = GradScaler("cuda", enabled=args.use_amp)
    sched  = TwoGroupScheduler(
        opt, args.warmup_epochs, args.num_epochs,
        args.lr, args.lr_enc, args.lr_min,
        args.freeze_enc_epochs, args.enc_warmup_epochs)

    start_ep = 0; best_score = float("inf"); patience_cnt = 0
    if args.resume and os.path.exists(args.resume):
        start_ep, best_score, patience_cnt = load_checkpoint(
            args.resume, model, opt, sched, ema, scaler, device)
        print(f"  ↩ Resume ep{start_ep}  best={best_score:.1f}")

    # [BUG-H FIXED] torch.compile TRƯỚC freeze logic.
    # Freeze thực sự được handle bằng zero_grad + skip update (không phải
    # requires_grad=False sau compile vì compile có thể cache graph).
    # Thay vào đó: vẫn set requires_grad nhưng THÊM zero_grad cho frozen params
    # sau backward để đảm bảo optimizer không update chúng.
    try:
        model = torch.compile(model, mode="reduce-overhead"); print("  torch.compile: ok")
    except Exception: pass

    nstep = len(trl)
    val_ade_history = []
    periodic_ckpts  = []   # [BUG-4 FIXED]

    print(f"\n  TRAINING ({nstep} steps/ep × {args.num_epochs} ep)")

    for ep in range(start_ep, start_ep + args.num_epochs):
        model.train()
        sum_loss = sum_cfm = sum_reg = sum_ade1 = 0.0
        t0 = time.perf_counter()

        rel_ep = ep - start_ep

        # Freeze logic — [BUG-H FIXED]
        # Sau torch.compile, KHÔNG dùng requires_grad=False/True để freeze/unfreeze
        # vì compiled graph có thể không re-trace khi requires_grad thay đổi.
        # Thay vào đó: set requires_grad NHƯNG sau backward, zero gradient của
        # frozen params để optimizer không update chúng.
        # requires_grad=False vẫn set để tránh tính gradient không cần thiết,
        # nhưng không dựa vào nó hoàn toàn.
        # v2.4: không có ar_enc/ar_gate nữa — chỉ freeze/unfreeze encoder backbone
        freeze_enc = rel_ep < args.freeze_enc_epochs
        raw_enc = _unwrap(model).encoder
        for p in raw_enc.parameters():
            p.requires_grad_(not freeze_enc)

        if rel_ep == 0 and freeze_enc:
            print(f"  *** Ep{ep}: encoder frozen ({args.freeze_enc_epochs}ep) ***")
        if rel_ep == args.freeze_enc_epochs:
            print(f"\n  *** Ep{ep}: encoder unfrozen ***")

        for i, batch in enumerate(trl):
            bl = move(list(batch), device)
            bl_aug = augment_batch(bl)

            opt.zero_grad()
            with autocast(device_type="cuda", enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl_aug, epoch=ep)

            scaler.scale(bd["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # Zero grad cho encoder frozen params sau backward
            if freeze_enc:
                for p in _unwrap(model).encoder.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

            scaler.step(opt); scaler.update()
            model.ema_update()

            sum_loss += bd["total"].item(); sum_cfm += bd["l_cfm"]
            sum_reg  += bd["l_reg"];       sum_ade1 += bd["ade_1step"]

            if i % 30 == 0:
                lr_vel, lr_enc = sched.get_lrs()
                enc_status = "frozen" if freeze_enc else "active"
                print(f"  [{ep:>3}][{i:>3}/{nstep}]"
                      f"  total={bd['total'].item():.4f}"
                      f"  cfm={bd['l_cfm']:.4f}"
                      f"  reg={bd['l_reg']:.4f}"
                      f"  lam={bd['lam_reg']:.3f}"
                      f"  ade1={bd['ade_1step']:.0f}km"
                      f"  σ={bd['sigma']:.3f}"
                      f"  enc={enc_status}"
                      f"  lr_vel={lr_vel:.2e}")

        train_loss = sum_loss / nstep
        lr_vel, lr_enc = sched.get_lrs()
        sched.step()

        print(f"\n  ── Ep{ep:>3}  train={train_loss:.6f}"
              f"  cfm={sum_cfm/nstep:.4f}  reg={sum_reg/nstep:.4f}"
              f"  ade1={sum_ade1/nstep:.0f}km"
              f"  lr_vel={lr_vel:.2e}  lr_enc={lr_enc:.2e}"
              f"  t={time.perf_counter()-t0:.0f}s")

        _save(last_ckpt, ep, model, opt, sched, best_score, ema, scaler, patience_cnt)

        # Periodic checkpoint — [BUG-4 FIXED] track riêng
        if rel_ep % 5 == 0:
            ckpt_path = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
            _save(ckpt_path, ep, model, opt, sched, best_score, ema, scaler, patience_cnt)
            periodic_ckpts.append(ckpt_path)
            if len(periodic_ckpts) > 5: periodic_ckpts.pop(0)
            print(f"  💾 ckpt_ep{ep:03d}.pth")

        # Val
        if rel_ep % args.val_freq == 0:
            r = evaluate(model, val_loader, device,
                         tag=f"VAL ep{ep}", ema=ema,
                         ref_targets=ST_TRANS_VAL,
                         use_tta=False, epoch_for_loss=ep,
                         run_xai=(rel_ep % 10 == 0), xai_batch=xai_batch)

            val_ade   = r["ADE"]
            score     = r["combined"]
            val_ade_history.append(val_ade)

            if len(val_ade_history) >= 4:
                trend = float(np.mean(val_ade_history[-2:])) - float(np.mean(val_ade_history[-4:-2]))
                trend_str = (f"↑{trend:+.1f}km ⚠" if trend > 5
                             else f"↓{trend:+.1f}km ✓" if trend < -5
                             else f"→{trend:+.1f}km")
            else:
                trend_str = "—"

            print(f"  train={train_loss:.6f}  val_ADE={val_ade:.1f}km"
                  f"  combined={score:.1f}  trend={trend_str}")

            if score < best_score:
                best_score = score; patience_cnt = 0
                _save(best_ckpt, ep, model, opt, sched, best_score, ema, scaler, 0,
                      extra={"val_ade": r["ADE"], "val_ate": r["ATE"],
                             "val_cte": r["CTE"], "val_loss": r["val_loss"]})
                print(f"  ✅ Best! score={best_score:.2f}"
                      f"  ADE={r['ADE']:.1f} ATE={r['ATE']:.1f} CTE={r['CTE']:.1f}")
            else:
                if rel_ep >= args.min_ep: patience_cnt += args.val_freq
                print(f"  No improve {patience_cnt}/{args.patience} (best={best_score:.1f})")
                if rel_ep >= args.min_ep and patience_cnt >= args.patience:
                    print(f"  ⛔ Early stop @ ep{ep}"); break

    print(f"\n  Done! best_score={best_score:.2f}")

    # Test
    if args.test_at_end and os.path.exists(best_ckpt):
        print("\n  Loading best checkpoint...")
        ck = torch.load(best_ckpt, map_location=device)
        _unwrap(model).load_state_dict(ck["model"], strict=False)
        if ema and ck.get("ema"):
            for k, v in ck["ema"].items():
                if k in ema.shadow: ema.shadow[k].copy_(v.to(device))

        # [BUG-4 FIXED] ensemble từ periodic ckpts
        if len(periodic_ckpts) >= 3:
            ensemble_checkpoints(periodic_ckpts[-3:], model, device)

        try:
            _, test_loader = data_loader(args, {"root": args.dataset_root, "type": "test"}, test=True)
        except Exception:
            print("  No test set → val"); test_loader = val_loader

        r_test = evaluate(model, test_loader, device,
                          tag="TEST (best + ensemble)",
                          ema=ema, ref_targets=ST_TRANS_TEST,
                          use_tta=args.tta_test, n_tta=args.n_tta)

        val_ade = ck.get("val_ade", float("nan"))
        print("\n  VAL vs TEST gap:")
        for m_name, ref in [("ADE", ST_TRANS_TEST["ADE"]),
                             ("72h", ST_TRANS_TEST["72h"])]:
            tv  = r_test.get(m_name, float("nan"))
            vv  = val_ade if m_name == "ADE" else float("nan")
            gap = tv - vv if (np.isfinite(tv) and np.isfinite(vv)) else float("nan")
            beat = "✅" if np.isfinite(tv) and tv < ref else "✗"
            print(f"  {m_name}: val={vv:.1f}  test={tv:.1f}  gap={gap:+.1f}"
                  f"  ST-Trans={ref:.1f}  {beat}")


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    if args.dataset_root == "TCND_vn":
        _auto = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
        if os.path.isdir(_auto): args.dataset_root = _auto
    main(args)
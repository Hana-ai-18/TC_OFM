"""
scripts/train_fm.py  ——  TC-FlowMatching v2 Training
═══════════════════════════════════════════════════════

Tương ứng với flow_matching_model.py (FM đúng cách).

THIẾT KẾ TRAINING ĐƠN GIẢN:
  - 1 optimizer, 1 scheduler (Cosine + warmup)
  - Không GradNorm (chỉ có 2 loss terms)
  - Không phase transitions phức tạp
  - Không alpha_hard, không selector training
  - EMA + early stopping trên val ADE

HYPERPARAMETERS MẶC ĐỊNH:
  lr = 2e-4   (nhỏ hơn bản cũ vì FM transformer deeper)
  batch_size = 64
  num_epochs = 150
  warmup = 5 epoch
  sigma_max = 0.3, sigma_min = 0.1 (schedule tự động)
  lambda_reg = 0.2 (ramp từ epoch 10-30)
  n_ensemble = 20, ddim_steps = 8

CHẠY:
  python scripts/train_fm.py \
    --dataset_root /kaggle/input/.../tc-ofm \
    --output_dir   runs/fm_v2 \
    --num_epochs   150 \
    --batch_size   64

RESUME:
  python scripts/train_fm.py \
    --resume runs/fm_v2/best_model.pth \
    --dataset_root ...
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, math, random, time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader
from Model.data.trajectoriesWithMe_unet_training import seq_collate
from Model.flow_matching_model import (
    TCFlowMatching, _norm_to_deg, _haversine_deg, EMAModel,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
R_EARTH = 6371.0
HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}

ST_TRANS_TARGETS = {
    "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
    "12h": 65.42,  "24h": 104.67, "48h": 205.10,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m

def move(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return out

def make_subset_loader(dataset, n, batch_size, collate_fn):
    idx = random.Random(42).sample(range(len(dataset)), min(n, len(dataset)))
    return DataLoader(Subset(dataset, idx), batch_size=batch_size,
                      shuffle=False, collate_fn=collate_fn,
                      num_workers=0, drop_last=False)

def _ate_cte(pred_deg, gt_deg):
    """ATE, CTE per step"""
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z
    lo1 = torch.deg2rad(gt_deg[:T-1, :, 0])
    la1 = torch.deg2rad(gt_deg[:T-1, :, 1])
    lo2 = torch.deg2rad(gt_deg[1:T,  :, 0])
    la2 = torch.deg2rad(gt_deg[1:T,  :, 1])
    lo3 = torch.deg2rad(pred_deg[1:T, :, 0])
    la3 = torch.deg2rad(pred_deg[1:T, :, 1])
    ya  = torch.sin(lo2 - lo1) * torch.cos(la2)
    xa  = (torch.cos(la1) * torch.sin(la2)
           - torch.sin(la1) * torch.cos(la2) * torch.cos(lo2 - lo1))
    be  = torch.atan2(ya, xa)
    ye  = torch.sin(lo3 - lo2) * torch.cos(la3)
    xe  = (torch.cos(la2) * torch.sin(la3)
           - torch.sin(la2) * torch.cos(la3) * torch.cos(lo3 - lo2))
    bee = torch.atan2(ye, xe)
    tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
    ang = bee - be
    return tot * torch.cos(ang), tot * torch.sin(ang)


# ─────────────────────────────────────────────────────────────────────────────
#  Scheduler: linear warmup + cosine annealing
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 lr_init: float, lr_min: float = 1e-6):
        self.opt     = optimizer
        self.warmup  = warmup_epochs
        self.total   = total_epochs
        self.lr_init = lr_init
        self.lr_min  = lr_min
        self.epoch   = 0

    def step(self) -> float:
        e = self.epoch
        if e < self.warmup:
            lr = self.lr_init * (0.1 + 0.9 * e / max(self.warmup - 1, 1))
        else:
            t  = e - self.warmup
            T  = self.total - self.warmup
            lr = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (
                1 + math.cos(math.pi * t / max(T, 1)))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        self.epoch += 1
        return lr


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, tag: str = "",
             ddim_steps: int = 8, ema: Optional[EMAModel] = None) -> Dict:
    bk = None
    if ema is not None:
        try: bk = ema.apply_to(model)
        except: pass

    model.eval()
    all_ade, all_ate, all_cte = [], [], []
    step_ade = defaultdict(list)

    for batch in loader:
        bl   = move(list(batch), device)
        gt   = bl[1]
        pred, _, _ = model.sample(bl, ddim_steps=ddim_steps)

        T        = min(pred.shape[0], gt.shape[0])
        pred_deg = _norm_to_deg(pred[:T])
        gt_deg   = _norm_to_deg(gt[:T])
        dist     = _haversine_deg(pred_deg, gt_deg)  # [T, B]
        ate, cte = _ate_cte(pred_deg, gt_deg)

        all_ade.extend(dist.mean(0).tolist())
        if ate.shape[0] > 0:
            all_ate.extend(ate.abs().mean(0).tolist())
            all_cte.extend(cte.abs().mean(0).tolist())

        for h, s in HORIZON_STEPS.items():
            if s < T:
                step_ade[h].extend(dist[s].tolist())

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")

    result = {
        "ADE": _m(all_ade),
        "ATE": _m(all_ate),
        "CTE": _m(all_cte),
        "n":   len(all_ade),
    }
    for h in HORIZON_STEPS:
        result[f"{h}h"] = _m(step_ade[h])

    if bk is not None:
        try: ema.restore(model, bk)
        except: pass

    # Print
    def _v(k): return result.get(k, float("nan"))
    def _ok(k, ref): return "✓" if np.isfinite(_v(k)) and _v(k) < ref else "✗"

    print(f"\n  {'='*60}")
    print(f"  [{tag}]")
    print(f"  ADE={_v('ADE'):.1f} {_ok('ADE', ST_TRANS_TARGETS['ADE'])}"
          f"  (target < {ST_TRANS_TARGETS['ADE']})")
    print(f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
          f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}"
          f"  {_ok('72h', ST_TRANS_TARGETS['72h'])}")
    print(f"  ATE={_v('ATE'):.1f}  CTE={_v('CTE'):.1f}")

    beat = []
    for k, ref in ST_TRANS_TARGETS.items():
        v = _v(k)
        if np.isfinite(v) and v < ref:
            beat.append(f"{k}:{v:.1f}<{ref:.1f}")
    if beat:
        print(f"  *** BEAT ST-TRANS: {' | '.join(beat)} ***")
    print(f"  {'='*60}\n")

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _save(path, epoch, model, opt, sched, best_ade, ema=None,
          patience_cnt=0, scaler=None, args=None,
          val_loss_history=None, best_val_loss=float("inf")):
    m   = _unwrap(model)
    esd = None
    if ema is not None:
        try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
        except: pass
    torch.save({
        # ── Core weights ──────────────────────────────────────────────
        "epoch":        epoch,
        "model":        m.state_dict(),
        "ema":          esd,
        # ── Optimizer / scheduler / scaler ────────────────────────────
        # Lưu đủ để resume tiếp tục chính xác như chưa bị interrupt:
        #   optimizer : Adam momentum + variance -> không mất đà
        #   scheduler : sched.epoch -> WarmupCosine biết đang ở đâu
        #   scaler    : AMP loss scale đã calibrate -> không bị reset
        "optimizer":    opt.state_dict(),
        "scheduler":    sched.epoch,
        "sched_total":  sched.total,    # giữ cosine tail đúng khi resume với num_epochs khác
        "scaler":       scaler.state_dict() if scaler is not None else None,
        # ── Training state ────────────────────────────────────────────
        "best_ade":     best_ade,
        "patience_cnt":    patience_cnt,
        "val_loss_history": val_loss_history,
        "best_val_loss":    best_val_loss,
        # ── Hyperparams snapshot ──────────────────────────────────────
        "args":         vars(args) if args is not None else None,
    }, path)


# ─────────────────────────────────────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    p.add_argument("--dataset_root",  default="TCND_vn")
    p.add_argument("--obs_len",       default=8,     type=int)
    p.add_argument("--pred_len",      default=12,    type=int)
    p.add_argument("--num_workers",   default=2,     type=int)
    p.add_argument("--other_modal",   default="gph")
    p.add_argument("--delim",         default=" ")
    p.add_argument("--skip",          default=1,     type=int)
    p.add_argument("--min_ped",       default=1,     type=int)
    p.add_argument("--threshold",     default=0.002, type=float)
    # Model
    p.add_argument("--d_cond",        default=256,   type=int)
    p.add_argument("--d_model",       default=256,   type=int)
    p.add_argument("--nhead",         default=8,     type=int)
    p.add_argument("--num_dec_layers",default=4,     type=int)
    p.add_argument("--dim_ff",        default=512,   type=int)
    p.add_argument("--dropout",       default=0.1,   type=float)
    p.add_argument("--unet_in_ch",    default=13,    type=int)
    p.add_argument("--sigma_min",     default=0.04,  type=float,
                   help="FM sigma min (~22km, 15%% ADE). Phải < ADE_norm=0.32")
    p.add_argument("--sigma_max",     default=0.08,  type=float,
                   help="FM sigma max (~43km, 25%% ADE). Phải << ADE_norm")
    p.add_argument("--lambda_reg",    default=0.2,   type=float,
                   help="ADE signal weight (FIX-FATAL-1)")
    p.add_argument("--use_ot",        default=True,  action="store_true")
    p.add_argument("--ot_epsilon",    default=0.05,  type=float,
                   help="Sinkhorn epsilon cho OT matching")
    p.add_argument("--no_ot",         dest="use_ot", action="store_false")
    p.add_argument("--n_ensemble",    default=20,    type=int)
    p.add_argument("--sigma_inference", default=0.04, type=float,
                   help="Inference noise std. Dat = sigma_min (0.04) de tranh "
                        "mismatch: epoch 40+ model chi thay sigma=0.04, "
                        "dung 0.079 se extrapolate ngoai training range.")
    # Training
    p.add_argument("--num_epochs",    default=150,   type=int)
    p.add_argument("--batch_size",    default=64,    type=int)
    p.add_argument("--lr",            default=2e-4,  type=float)
    p.add_argument("--lr_min",        default=1e-6,  type=float)
    p.add_argument("--warmup_epochs", default=5,     type=int)
    p.add_argument("--weight_decay",  default=1e-4,  type=float)
    p.add_argument("--grad_clip",     default=1.0,   type=float)
    p.add_argument("--use_amp",       action="store_true", default=False)
    p.add_argument("--use_ema",       default=True,  action="store_true")
    p.add_argument("--no_ema",        dest="use_ema",action="store_false")
    p.add_argument("--ema_decay",     default=0.995, type=float)
    # Eval
    p.add_argument("--val_freq",      default=5,     type=int)
    p.add_argument("--val_subset",    default=1000,  type=int,
                   help="Số samples val mỗi lần eval (default 1000 ≈ 30% val set)")
    p.add_argument("--ddim_steps",    default=8,     type=int)
    p.add_argument("--patience",      default=30,    type=int)
    p.add_argument("--min_ep",        default=20,    type=int)
    # IO
    p.add_argument("--output_dir",    default="runs/fm_v2")
    p.add_argument("--gpu_num",       default="0")
    p.add_argument("--resume",        default=None)
    p.add_argument("--test_at_end",   action="store_true", default=True)
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

    print("=" * 65)
    print("  TC-FlowMatching v2  |  FM đúng cách")
    print(f"  sigma: {args.sigma_max}→{args.sigma_min}  "
          f"lambda_reg={args.lambda_reg}  K={args.n_ensemble}")
    print(f"  N_steps={args.ddim_steps}  sigma_inf={args.sigma_inference}")
    print(f"  FIX-FATAL-1: L_reg tại t=0 (gradient không dilute)")
    print(f"  FIX-FATAL-2: 2 loss terms (l_cfm + lambda_reg*l_reg)")
    print(f"  FIX-FATAL-3: Best-of-K thay SelectorNet")
    print(f"  FIX-MAJOR-4: sigma_max={args.sigma_max}, "
          f"sigma_inf={args.sigma_inference}")
    print(f"  FIX-MAJOR-5: FM trên 2D (không 4D)")
    print("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────────
    trd, trl = data_loader(
        args, {"root": args.dataset_root, "type": "train"}, test=False)
    vd, _    = data_loader(
        args, {"root": args.dataset_root, "type": "val"},   test=True)
    val_sub  = make_subset_loader(vd, args.val_subset, args.batch_size, seq_collate)
    # Full val loader — dùng để ADE evaluation chính xác (không bị bias subset)
    val_full = DataLoader(vd, batch_size=args.batch_size, shuffle=False,
                          collate_fn=seq_collate, num_workers=0, drop_last=False)
    print(f"  train: {len(trd)}  val: {len(vd)}  (full_val batches={len(val_full)})")

    # ── Model ─────────────────────────────────────────────────────────────
    model = TCFlowMatching(
        pred_len=args.pred_len, obs_len=args.obs_len,
        unet_in_ch=args.unet_in_ch,
        d_cond=args.d_cond, d_model=args.d_model,
        nhead=args.nhead, num_dec_layers=args.num_dec_layers,
        dim_ff=args.dim_ff, dropout=args.dropout,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        lambda_reg=args.lambda_reg,
        use_ot=args.use_ot,
        ot_epsilon=args.ot_epsilon,
        use_ema=args.use_ema,
        n_ensemble=args.n_ensemble,
        sigma_inference=args.sigma_inference,
    ).to(device)

    model.init_ema()
    ema = getattr(_unwrap(model), "_ema", None)

    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_enc   = sum(p.numel() for p in model.encoder.parameters())
    n_vel   = sum(p.numel() for p in model.velocity.parameters())
    print(f"\n  Params total  : {n_total:,}")
    print(f"  Encoder       : {n_enc:,}  (FNO3D+Mamba, giữ nguyên)")
    print(f"  VelocityTrans : {n_vel:,}  (4-layer transformer, 2D only)")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    opt    = optim.AdamW(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=args.use_amp)
    sched  = WarmupCosineScheduler(
        opt, warmup_epochs=args.warmup_epochs,
        total_epochs=args.num_epochs,
        lr_init=args.lr, lr_min=args.lr_min)

    # ── Resume ────────────────────────────────────────────────────────────
    start_ep = 0
    best_ade = float("inf")
    patience_cnt = 0
    val_loss_history: list = []    # track trend để phát hiện overfit
    best_val_loss = float("inf")   # track best val_loss riêng

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        _unwrap(model).load_state_dict(ck["model"], strict=False)
        try: opt.load_state_dict(ck["optimizer"])
        except: pass
        sched.epoch  = ck.get("scheduler", 0)
        if "sched_total" in ck:
            sched.total = ck["sched_total"]   # giữ đúng cosine curve khi resume
        start_ep     = ck.get("epoch", 0) + 1
        best_ade     = ck.get("best_ade", float("inf"))
        patience_cnt = ck.get("patience_cnt", 0)   # ← restore patience
        if scaler is not None and ck.get("scaler") is not None:
            try: scaler.load_state_dict(ck["scaler"])   # ← restore AMP scale
            except: pass
        if ema and ck.get("ema"):
            for k, v in ck["ema"].items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.to(device))
        print(f"  ↩ Resumed ep{start_ep}  best_ADE={best_ade:.1f} km"
              f"  patience={patience_cnt}/{args.patience}")

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: ok")
    except:
        pass

    print()
    print("=" * 65)
    print(f"  TRAINING  ({len(trl)} steps/epoch, {args.num_epochs} epochs)")
    print(f"  Warmup: {args.warmup_epochs} ep  lr: {args.lr}→{args.lr_min}")
    print(f"  L_reg ramp: ep10→ep30 (0→{args.lambda_reg})")
    print("=" * 65)

    nstep = len(trl)

    for ep in range(start_ep, args.num_epochs):
        model.train()
        sum_loss, sum_cfm, sum_reg, sum_ade1 = 0.0, 0.0, 0.0, 0.0
        t0 = time.perf_counter()

        for i, batch in enumerate(trl):
            bl = move(list(batch), device)

            opt.zero_grad()
            with autocast(device_type="cuda", enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl, epoch=ep)

            scaler.scale(bd["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            model.ema_update()

            sum_loss += bd["total"].item()
            sum_cfm  += bd["l_cfm"]
            sum_reg  += bd["l_reg"]
            sum_ade1 += bd["ade_1step"]

            if i % 30 == 0:
                lr = opt.param_groups[0]["lr"]
                print(f"  [{ep:>3}][{i:>3}/{nstep}]"
                      f"  total={bd['total'].item():.4f}"
                      f"  cfm={bd['l_cfm']:.4f}"
                      f"  reg={bd['l_reg']:.4f}"
                      f"  lam={bd['lam_reg']:.3f}"
                      f"  ade_1step={bd['ade_1step']:.0f}km"
                      f"  sigma={bd['sigma']:.3f}"
                      f"  hs_mean={bd.get('hard_score_mean',0):.2f}"
                      f"  lr={lr:.2e}")

        avg_loss = sum_loss / nstep
        avg_cfm  = sum_cfm  / nstep
        avg_reg  = sum_reg  / nstep
        avg_ade1 = sum_ade1 / nstep
        # Lấy lr hiện tại (đang dùng trong epoch này) TRƯỚC khi sched.step() thay đổi
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()   # advance lr cho epoch tiếp theo

        print(f"  Epoch {ep:>3}"
              f"  loss={avg_loss:.4f}"
              f"  cfm={avg_cfm:.4f}"
              f"  reg={avg_reg:.4f}"
              f"  ade_1step={avg_ade1:.0f}km"
              f"  lr={cur_lr:.2e}"      # lr epoch này (không phải epoch sau)
              f"  t={time.perf_counter()-t0:.0f}s")

        # ── Periodic ckpt mỗi 10 epoch ──────────────────────────────────────
        # Đặt TRƯỚC val block để không bị skip khi early stop break
        if ep % 5 == 0:
            _save(os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
                  ep, model, opt, sched, best_ade, ema,
                  patience_cnt=patience_cnt, scaler=scaler, args=args,
                      val_loss_history=val_loss_history, best_val_loss=best_val_loss)

        # ── Val ───────────────────────────────────────────────────────────
        if ep % args.val_freq == 0:
            # Val loss: dùng val_sub (nhanh) để theo dõi overfitting
            model.eval()
            val_loss_buf = []
            with torch.no_grad():
                for vbatch in val_sub:
                    vbl = move(list(vbatch), device)
                    with autocast(device_type="cuda", enabled=args.use_amp):
                        vbd = model.get_loss_breakdown(vbl, epoch=ep)
                    if torch.isfinite(vbd["total"]):
                        val_loss_buf.append(vbd["total"].item())
            avg_val_loss = float(np.mean(val_loss_buf)) if val_loss_buf else float("nan")
            val_loss_history.append(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            # Trend: trung bình 3 lần gần nhất vs 3 lần trước
            if len(val_loss_history) >= 6:
                recent = float(np.mean(val_loss_history[-3:]))
                before = float(np.mean(val_loss_history[-6:-3]))
                trend  = recent - before
                if   trend >  0.002: trend_str = f"↑ +{trend:.4f} ⚠TĂNG"
                elif trend < -0.002: trend_str = f"↓ {trend:.4f} ✓giảm"
                else:                trend_str = f"→ {trend:+.4f} flat"
            elif len(val_loss_history) >= 2:
                d = val_loss_history[-1] - val_loss_history[-2]
                trend_str = f"{'↑' if d > 0 else '↓'} {d:+.4f}"
            else:
                trend_str = "—"

            gap     = avg_val_loss - avg_loss
            gap_pct = gap / max(avg_loss, 1e-8) * 100
            if   gap_pct > 50: warn = "  🔴 OVERFIT"
            elif gap_pct > 25: warn = "  🟡 watch"
            else:              warn = ""

            print(f"  val_loss={avg_val_loss:.4f}  best={best_val_loss:.4f}"
                  f"  trend={trend_str}"
                  f"  gap={gap:+.4f}({gap_pct:+.0f}%){warn}"
                  f"  train={avg_loss:.4f}")

            # Val ADE: full val set để early stop không bị bias subset
            r = evaluate(model, val_full, device,
                         tag=f"val ep{ep}  [full {len(vd)} samples]",
                         ddim_steps=args.ddim_steps,
                         ema=ema)
            ade = r["ADE"]

            if ade < best_ade:
                best_ade     = ade
                patience_cnt = 0
                # Lưu ngay từ epoch 0, không chờ min_ep
                _save(best_ckpt, ep, model, opt, sched, best_ade, ema,
                      patience_cnt=patience_cnt, scaler=scaler, args=args,
                      val_loss_history=val_loss_history, best_val_loss=best_val_loss)
                print(f"  ✅ Best ADE = {best_ade:.2f} km  (ep {ep})")
            else:
                # Patience chỉ đếm sau min_ep — tránh early stop quá sớm
                if ep >= args.min_ep:
                    patience_cnt += args.val_freq
                print(f"  No improve {patience_cnt}/{args.patience}"
                      f"  (best={best_ade:.1f})")
                if ep >= args.min_ep and patience_cnt >= args.patience:
                    print(f"  ⛔ Early stop @ ep{ep}")
                    break

    print("=" * 65)
    print(f"  Best ADE: {best_ade:.2f} km")
    print(f"  Target:   {ST_TRANS_TARGETS['ADE']:.2f} km (ST-Trans)")
    print(f"  Gap:      {best_ade - ST_TRANS_TARGETS['ADE']:+.2f} km")
    print("=" * 65)

    # ── Test ──────────────────────────────────────────────────────────────
    if args.test_at_end and os.path.exists(best_ckpt):
        print("\n  Test set evaluation...")
        try:
            _, test_loader = data_loader(
                args, {"root": args.dataset_root, "type": "test"}, test=True)
        except:
            print("  No test set → using val")
            _, test_loader = data_loader(
                args, {"root": args.dataset_root, "type": "val"}, test=True)

        ck = torch.load(best_ckpt, map_location=device)
        _unwrap(model).load_state_dict(ck["model"], strict=False)
        if ema and ck.get("ema"):
            for k, v in ck["ema"].items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.to(device))

        evaluate(model, test_loader, device,
                 tag="TEST (EMA best)",
                 ddim_steps=args.ddim_steps,
                 ema=ema)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main(args)
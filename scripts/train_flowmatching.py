# # # # # """
# # # # # scripts/train_fm.py  ——  TC-FlowMatching v2 Training
# # # # # ═══════════════════════════════════════════════════════

# # # # # Tương ứng với flow_matching_model.py (FM đúng cách).

# # # # # THIẾT KẾ TRAINING ĐƠN GIẢN:
# # # # #   - 1 optimizer, 1 scheduler (Cosine + warmup)
# # # # #   - Không GradNorm (chỉ có 2 loss terms)
# # # # #   - Không phase transitions phức tạp
# # # # #   - Không alpha_hard, không selector training
# # # # #   - EMA + early stopping trên val ADE

# # # # # HYPERPARAMETERS MẶC ĐỊNH:
# # # # #   lr = 2e-4   (nhỏ hơn bản cũ vì FM transformer deeper)
# # # # #   batch_size = 64
# # # # #   num_epochs = 150
# # # # #   warmup = 5 epoch
# # # # #   sigma_max = 0.3, sigma_min = 0.1 (schedule tự động)
# # # # #   lambda_reg = 0.2 (ramp từ epoch 10-30)
# # # # #   n_ensemble = 20, ddim_steps = 8

# # # # # CHẠY:
# # # # #   python scripts/train_fm.py \
# # # # #     --dataset_root /kaggle/input/.../tc-ofm \
# # # # #     --output_dir   runs/fm_v2 \
# # # # #     --num_epochs   150 \
# # # # #     --batch_size   64

# # # # # RESUME:
# # # # #   python scripts/train_fm.py \
# # # # #     --resume runs/fm_v2/best_model.pth \
# # # # #     --dataset_root ...
# # # # # """
# # # # # from __future__ import annotations
# # # # # import sys, os
# # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # import argparse, math, random, time
# # # # # from collections import defaultdict
# # # # # from typing import Dict, Optional

# # # # # import numpy as np
# # # # # import torch
# # # # # import torch.optim as optim
# # # # # from torch.amp import autocast, GradScaler
# # # # # from torch.utils.data import DataLoader, Subset

# # # # # from Model.data.loader_training import data_loader
# # # # # from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # from Model.flow_matching_model import (
# # # # #     TCFlowMatching, _norm_to_deg, _haversine_deg, EMAModel,
# # # # # )

# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Constants
# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # R_EARTH = 6371.0
# # # # # HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}

# # # # # ST_TRANS_TARGETS = {
# # # # #     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
# # # # #     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# # # # # }


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Utilities
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def _unwrap(m):
# # # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # # # # def move(batch, device):
# # # # #     out = list(batch)
# # # # #     for i, x in enumerate(out):
# # # # #         if torch.is_tensor(x):
# # # # #             out[i] = x.to(device)
# # # # #         elif isinstance(x, dict):
# # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # #                       for k, v in x.items()}
# # # # #     return out

# # # # # def make_subset_loader(dataset, n, batch_size, collate_fn):
# # # # #     idx = random.Random(42).sample(range(len(dataset)), min(n, len(dataset)))
# # # # #     return DataLoader(Subset(dataset, idx), batch_size=batch_size,
# # # # #                       shuffle=False, collate_fn=collate_fn,
# # # # #                       num_workers=0, drop_last=False)

# # # # # def _ate_cte(pred_deg, gt_deg):
# # # # #     """ATE, CTE per step"""
# # # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # # #     if T < 2:
# # # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # # #         return z, z
# # # # #     lo1 = torch.deg2rad(gt_deg[:T-1, :, 0])
# # # # #     la1 = torch.deg2rad(gt_deg[:T-1, :, 1])
# # # # #     lo2 = torch.deg2rad(gt_deg[1:T,  :, 0])
# # # # #     la2 = torch.deg2rad(gt_deg[1:T,  :, 1])
# # # # #     lo3 = torch.deg2rad(pred_deg[1:T, :, 0])
# # # # #     la3 = torch.deg2rad(pred_deg[1:T, :, 1])
# # # # #     ya  = torch.sin(lo2 - lo1) * torch.cos(la2)
# # # # #     xa  = (torch.cos(la1) * torch.sin(la2)
# # # # #            - torch.sin(la1) * torch.cos(la2) * torch.cos(lo2 - lo1))
# # # # #     be  = torch.atan2(ya, xa)
# # # # #     ye  = torch.sin(lo3 - lo2) * torch.cos(la3)
# # # # #     xe  = (torch.cos(la2) * torch.sin(la3)
# # # # #            - torch.sin(la2) * torch.cos(la3) * torch.cos(lo3 - lo2))
# # # # #     bee = torch.atan2(ye, xe)
# # # # #     tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# # # # #     ang = bee - be
# # # # #     return tot * torch.cos(ang), tot * torch.sin(ang)


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Scheduler: linear warmup + cosine annealing
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # class WarmupCosineScheduler:
# # # # #     def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
# # # # #                  lr_init: float, lr_min: float = 1e-6):
# # # # #         self.opt     = optimizer
# # # # #         self.warmup  = warmup_epochs
# # # # #         self.total   = total_epochs
# # # # #         self.lr_init = lr_init
# # # # #         self.lr_min  = lr_min
# # # # #         self.epoch   = 0

# # # # #     def step(self) -> float:
# # # # #         e = self.epoch
# # # # #         if e < self.warmup:
# # # # #             lr = self.lr_init * (0.1 + 0.9 * e / max(self.warmup - 1, 1))
# # # # #         else:
# # # # #             t  = e - self.warmup
# # # # #             T  = self.total - self.warmup
# # # # #             lr = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (
# # # # #                 1 + math.cos(math.pi * t / max(T, 1)))
# # # # #         for pg in self.opt.param_groups:
# # # # #             pg["lr"] = lr
# # # # #         self.epoch += 1
# # # # #         return lr


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Evaluation
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # @torch.no_grad()
# # # # # def evaluate(model, loader, device, tag: str = "",
# # # # #              ddim_steps: int = 8, ema: Optional[EMAModel] = None) -> Dict:
# # # # #     bk = None
# # # # #     if ema is not None:
# # # # #         try: bk = ema.apply_to(model)
# # # # #         except: pass

# # # # #     model.eval()
# # # # #     all_ade, all_ate, all_cte = [], [], []
# # # # #     step_ade = defaultdict(list)

# # # # #     for batch in loader:
# # # # #         bl   = move(list(batch), device)
# # # # #         gt   = bl[1]
# # # # #         pred, _, _ = model.sample(bl, ddim_steps=ddim_steps)

# # # # #         T        = min(pred.shape[0], gt.shape[0])
# # # # #         pred_deg = _norm_to_deg(pred[:T])
# # # # #         gt_deg   = _norm_to_deg(gt[:T])
# # # # #         dist     = _haversine_deg(pred_deg, gt_deg)  # [T, B]
# # # # #         ate, cte = _ate_cte(pred_deg, gt_deg)

# # # # #         all_ade.extend(dist.mean(0).tolist())
# # # # #         if ate.shape[0] > 0:
# # # # #             all_ate.extend(ate.abs().mean(0).tolist())
# # # # #             all_cte.extend(cte.abs().mean(0).tolist())

# # # # #         for h, s in HORIZON_STEPS.items():
# # # # #             if s < T:
# # # # #                 step_ade[h].extend(dist[s].tolist())

# # # # #     def _m(lst): return float(np.mean(lst)) if lst else float("nan")

# # # # #     result = {
# # # # #         "ADE": _m(all_ade),
# # # # #         "ATE": _m(all_ate),
# # # # #         "CTE": _m(all_cte),
# # # # #         "n":   len(all_ade),
# # # # #     }
# # # # #     for h in HORIZON_STEPS:
# # # # #         result[f"{h}h"] = _m(step_ade[h])

# # # # #     if bk is not None:
# # # # #         try: ema.restore(model, bk)
# # # # #         except: pass

# # # # #     # Print
# # # # #     def _v(k): return result.get(k, float("nan"))
# # # # #     def _ok(k, ref): return "✓" if np.isfinite(_v(k)) and _v(k) < ref else "✗"

# # # # #     print(f"\n  {'='*60}")
# # # # #     print(f"  [{tag}]")
# # # # #     print(f"  ADE={_v('ADE'):.1f} {_ok('ADE', ST_TRANS_TARGETS['ADE'])}"
# # # # #           f"  (target < {ST_TRANS_TARGETS['ADE']})")
# # # # #     print(f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
# # # # #           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}"
# # # # #           f"  {_ok('72h', ST_TRANS_TARGETS['72h'])}")
# # # # #     print(f"  ATE={_v('ATE'):.1f}  CTE={_v('CTE'):.1f}")

# # # # #     beat = []
# # # # #     for k, ref in ST_TRANS_TARGETS.items():
# # # # #         v = _v(k)
# # # # #         if np.isfinite(v) and v < ref:
# # # # #             beat.append(f"{k}:{v:.1f}<{ref:.1f}")
# # # # #     if beat:
# # # # #         print(f"  *** BEAT ST-TRANS: {' | '.join(beat)} ***")
# # # # #     print(f"  {'='*60}\n")

# # # # #     return result


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Checkpoint
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def _save(path, epoch, model, opt, sched, best_ade, ema=None,
# # # # #           patience_cnt=0, scaler=None, args=None,
# # # # #           val_loss_history=None, best_val_loss=float("inf")):
# # # # #     m   = _unwrap(model)
# # # # #     esd = None
# # # # #     if ema is not None:
# # # # #         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # # # #         except: pass
# # # # #     torch.save({
# # # # #         # ── Core weights ──────────────────────────────────────────────
# # # # #         "epoch":        epoch,
# # # # #         "model":        m.state_dict(),
# # # # #         "ema":          esd,
# # # # #         # ── Optimizer / scheduler / scaler ────────────────────────────
# # # # #         # Lưu đủ để resume tiếp tục chính xác như chưa bị interrupt:
# # # # #         #   optimizer : Adam momentum + variance -> không mất đà
# # # # #         #   scheduler : sched.epoch -> WarmupCosine biết đang ở đâu
# # # # #         #   scaler    : AMP loss scale đã calibrate -> không bị reset
# # # # #         "optimizer":    opt.state_dict(),
# # # # #         "scheduler":    sched.epoch,
# # # # #         "sched_total":  sched.total,    # giữ cosine tail đúng khi resume với num_epochs khác
# # # # #         "scaler":       scaler.state_dict() if scaler is not None else None,
# # # # #         # ── Training state ────────────────────────────────────────────
# # # # #         "best_ade":     best_ade,
# # # # #         "patience_cnt":    patience_cnt,
# # # # #         "val_loss_history": val_loss_history,
# # # # #         "best_val_loss":    best_val_loss,
# # # # #         # ── Hyperparams snapshot ──────────────────────────────────────
# # # # #         "args":         vars(args) if args is not None else None,
# # # # #     }, path)


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Args
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def get_args():
# # # # #     p = argparse.ArgumentParser(
# # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # #     # Data
# # # # #     p.add_argument("--dataset_root",  default="TCND_vn")
# # # # #     p.add_argument("--obs_len",       default=8,     type=int)
# # # # #     p.add_argument("--pred_len",      default=12,    type=int)
# # # # #     p.add_argument("--num_workers",   default=2,     type=int)
# # # # #     p.add_argument("--other_modal",   default="gph")
# # # # #     p.add_argument("--delim",         default=" ")
# # # # #     p.add_argument("--skip",          default=1,     type=int)
# # # # #     p.add_argument("--min_ped",       default=1,     type=int)
# # # # #     p.add_argument("--threshold",     default=0.002, type=float)
# # # # #     # Model
# # # # #     p.add_argument("--d_cond",        default=256,   type=int)
# # # # #     p.add_argument("--d_model",       default=256,   type=int)
# # # # #     p.add_argument("--nhead",         default=8,     type=int)
# # # # #     p.add_argument("--num_dec_layers",default=4,     type=int)
# # # # #     p.add_argument("--dim_ff",        default=512,   type=int)
# # # # #     p.add_argument("--dropout",       default=0.1,   type=float)
# # # # #     p.add_argument("--unet_in_ch",    default=13,    type=int)
# # # # #     p.add_argument("--sigma_min",     default=0.04,  type=float,
# # # # #                    help="FM sigma min (~22km, 15%% ADE). Phải < ADE_norm=0.32")
# # # # #     p.add_argument("--sigma_max",     default=0.08,  type=float,
# # # # #                    help="FM sigma max (~43km, 25%% ADE). Phải << ADE_norm")
# # # # #     p.add_argument("--lambda_reg",    default=0.2,   type=float,
# # # # #                    help="ADE signal weight (FIX-FATAL-1)")
# # # # #     p.add_argument("--use_ot",        default=True,  action="store_true")
# # # # #     p.add_argument("--ot_epsilon",    default=0.05,  type=float,
# # # # #                    help="Sinkhorn epsilon cho OT matching")
# # # # #     p.add_argument("--no_ot",         dest="use_ot", action="store_false")
# # # # #     p.add_argument("--n_ensemble",    default=20,    type=int)
# # # # #     p.add_argument("--sigma_inference", default=0.04, type=float,
# # # # #                    help="Inference noise std. Dat = sigma_min (0.04) de tranh "
# # # # #                         "mismatch: epoch 40+ model chi thay sigma=0.04, "
# # # # #                         "dung 0.079 se extrapolate ngoai training range.")
# # # # #     # Training
# # # # #     p.add_argument("--num_epochs",    default=150,   type=int)
# # # # #     p.add_argument("--batch_size",    default=64,    type=int)
# # # # #     p.add_argument("--lr",            default=2e-4,  type=float)
# # # # #     p.add_argument("--lr_min",        default=1e-6,  type=float)
# # # # #     p.add_argument("--warmup_epochs", default=5,     type=int)
# # # # #     p.add_argument("--weight_decay",  default=1e-4,  type=float)
# # # # #     p.add_argument("--grad_clip",     default=1.0,   type=float)
# # # # #     p.add_argument("--use_amp",       action="store_true", default=False)
# # # # #     p.add_argument("--use_ema",       default=True,  action="store_true")
# # # # #     p.add_argument("--no_ema",        dest="use_ema",action="store_false")
# # # # #     p.add_argument("--ema_decay",     default=0.995, type=float)
# # # # #     # Eval
# # # # #     p.add_argument("--val_freq",      default=5,     type=int)
# # # # #     p.add_argument("--val_subset",    default=1000,  type=int,
# # # # #                    help="Số samples val mỗi lần eval (default 1000 ≈ 30% val set)")
# # # # #     p.add_argument("--ddim_steps",    default=8,     type=int)
# # # # #     p.add_argument("--patience",      default=30,    type=int)
# # # # #     p.add_argument("--min_ep",        default=20,    type=int)
# # # # #     # IO
# # # # #     p.add_argument("--output_dir",    default="runs/fm_v2")
# # # # #     p.add_argument("--gpu_num",       default="0")
# # # # #     p.add_argument("--resume",        default=None)
# # # # #     p.add_argument("--test_at_end",   action="store_true", default=True)
# # # # #     return p.parse_args()


# # # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # # #  Main
# # # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # # def main(args):
# # # # #     if torch.cuda.is_available():
# # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # #     os.makedirs(args.output_dir, exist_ok=True)
# # # # #     best_ckpt = os.path.join(args.output_dir, "best_model.pth")

# # # # #     print("=" * 65)
# # # # #     print("  TC-FlowMatching v2  |  FM đúng cách")
# # # # #     print(f"  sigma: {args.sigma_max}→{args.sigma_min}  "
# # # # #           f"lambda_reg={args.lambda_reg}  K={args.n_ensemble}")
# # # # #     print(f"  N_steps={args.ddim_steps}  sigma_inf={args.sigma_inference}")
# # # # #     print(f"  FIX-FATAL-1: L_reg tại t=0 (gradient không dilute)")
# # # # #     print(f"  FIX-FATAL-2: 2 loss terms (l_cfm + lambda_reg*l_reg)")
# # # # #     print(f"  FIX-FATAL-3: Best-of-K thay SelectorNet")
# # # # #     print(f"  FIX-MAJOR-4: sigma_max={args.sigma_max}, "
# # # # #           f"sigma_inf={args.sigma_inference}")
# # # # #     print(f"  FIX-MAJOR-5: FM trên 2D (không 4D)")
# # # # #     print("=" * 65)

# # # # #     # ── Data ──────────────────────────────────────────────────────────────
# # # # #     trd, trl = data_loader(
# # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # #     vd, _    = data_loader(
# # # # #         args, {"root": args.dataset_root, "type": "val"},   test=True)
# # # # #     val_sub  = make_subset_loader(vd, args.val_subset, args.batch_size, seq_collate)
# # # # #     # Full val loader — dùng để ADE evaluation chính xác (không bị bias subset)
# # # # #     val_full = DataLoader(vd, batch_size=args.batch_size, shuffle=False,
# # # # #                           collate_fn=seq_collate, num_workers=0, drop_last=False)
# # # # #     print(f"  train: {len(trd)}  val: {len(vd)}  (full_val batches={len(val_full)})")

# # # # #     # ── Model ─────────────────────────────────────────────────────────────
# # # # #     model = TCFlowMatching(
# # # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # # #         unet_in_ch=args.unet_in_ch,
# # # # #         d_cond=args.d_cond, d_model=args.d_model,
# # # # #         nhead=args.nhead, num_dec_layers=args.num_dec_layers,
# # # # #         dim_ff=args.dim_ff, dropout=args.dropout,
# # # # #         sigma_min=args.sigma_min, sigma_max=args.sigma_max,
# # # # #         lambda_reg=args.lambda_reg,
# # # # #         use_ot=args.use_ot,
# # # # #         ot_epsilon=args.ot_epsilon,
# # # # #         use_ema=args.use_ema,
# # # # #         n_ensemble=args.n_ensemble,
# # # # #         sigma_inference=args.sigma_inference,
# # # # #     ).to(device)

# # # # #     model.init_ema()
# # # # #     ema = getattr(_unwrap(model), "_ema", None)

# # # # #     n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # #     n_enc   = sum(p.numel() for p in model.encoder.parameters())
# # # # #     n_vel   = sum(p.numel() for p in model.velocity.parameters())
# # # # #     print(f"\n  Params total  : {n_total:,}")
# # # # #     print(f"  Encoder       : {n_enc:,}  (FNO3D+Mamba, giữ nguyên)")
# # # # #     print(f"  VelocityTrans : {n_vel:,}  (4-layer transformer, 2D only)")

# # # # #     # ── Optimizer & scheduler ─────────────────────────────────────────────
# # # # #     opt    = optim.AdamW(model.parameters(), lr=args.lr,
# # # # #                           weight_decay=args.weight_decay)
# # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # # #     sched  = WarmupCosineScheduler(
# # # # #         opt, warmup_epochs=args.warmup_epochs,
# # # # #         total_epochs=args.num_epochs,
# # # # #         lr_init=args.lr, lr_min=args.lr_min)

# # # # #     # ── Resume ────────────────────────────────────────────────────────────
# # # # #     start_ep = 0
# # # # #     best_ade = float("inf")
# # # # #     patience_cnt = 0
# # # # #     val_loss_history: list = []    # track trend để phát hiện overfit
# # # # #     best_val_loss = float("inf")   # track best val_loss riêng

# # # # #     if args.resume and os.path.exists(args.resume):
# # # # #         ck = torch.load(args.resume, map_location=device)
# # # # #         _unwrap(model).load_state_dict(ck["model"], strict=False)
# # # # #         try: opt.load_state_dict(ck["optimizer"])
# # # # #         except: pass
# # # # #         sched.epoch  = ck.get("scheduler", 0)
# # # # #         if "sched_total" in ck:
# # # # #             sched.total = ck["sched_total"]   # giữ đúng cosine curve khi resume
# # # # #         start_ep     = ck.get("epoch", 0) + 1
# # # # #         best_ade     = ck.get("best_ade", float("inf"))
# # # # #         patience_cnt = ck.get("patience_cnt", 0)   # ← restore patience
# # # # #         if scaler is not None and ck.get("scaler") is not None:
# # # # #             try: scaler.load_state_dict(ck["scaler"])   # ← restore AMP scale
# # # # #             except: pass
# # # # #         if ema and ck.get("ema"):
# # # # #             for k, v in ck["ema"].items():
# # # # #                 if k in ema.shadow:
# # # # #                     ema.shadow[k].copy_(v.to(device))
# # # # #         print(f"  ↩ Resumed ep{start_ep}  best_ADE={best_ade:.1f} km"
# # # # #               f"  patience={patience_cnt}/{args.patience}")

# # # # #     try:
# # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # #         print("  torch.compile: ok")
# # # # #     except:
# # # # #         pass

# # # # #     print()
# # # # #     print("=" * 65)
# # # # #     print(f"  TRAINING  ({len(trl)} steps/epoch, {args.num_epochs} epochs)")
# # # # #     print(f"  Warmup: {args.warmup_epochs} ep  lr: {args.lr}→{args.lr_min}")
# # # # #     print(f"  L_reg ramp: ep10→ep30 (0→{args.lambda_reg})")
# # # # #     print("=" * 65)

# # # # #     nstep = len(trl)

# # # # #     for ep in range(start_ep, args.num_epochs):
# # # # #         model.train()
# # # # #         sum_loss, sum_cfm, sum_reg, sum_ade1 = 0.0, 0.0, 0.0, 0.0
# # # # #         t0 = time.perf_counter()

# # # # #         for i, batch in enumerate(trl):
# # # # #             bl = move(list(batch), device)

# # # # #             opt.zero_grad()
# # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # #                 bd = model.get_loss_breakdown(bl, epoch=ep)

# # # # #             scaler.scale(bd["total"]).backward()
# # # # #             scaler.unscale_(opt)
# # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # #             scaler.step(opt)
# # # # #             scaler.update()
# # # # #             model.ema_update()

# # # # #             sum_loss += bd["total"].item()
# # # # #             sum_cfm  += bd["l_cfm"]
# # # # #             sum_reg  += bd["l_reg"]
# # # # #             sum_ade1 += bd["ade_1step"]

# # # # #             if i % 30 == 0:
# # # # #                 lr = opt.param_groups[0]["lr"]
# # # # #                 print(f"  [{ep:>3}][{i:>3}/{nstep}]"
# # # # #                       f"  total={bd['total'].item():.4f}"
# # # # #                       f"  cfm={bd['l_cfm']:.4f}"
# # # # #                       f"  reg={bd['l_reg']:.4f}"
# # # # #                       f"  lam={bd['lam_reg']:.3f}"
# # # # #                       f"  ade_1step={bd['ade_1step']:.0f}km"
# # # # #                       f"  sigma={bd['sigma']:.3f}"
# # # # #                       f"  hs_mean={bd.get('hard_score_mean',0):.2f}"
# # # # #                       f"  lr={lr:.2e}")

# # # # #         avg_loss = sum_loss / nstep
# # # # #         avg_cfm  = sum_cfm  / nstep
# # # # #         avg_reg  = sum_reg  / nstep
# # # # #         avg_ade1 = sum_ade1 / nstep
# # # # #         # Lấy lr hiện tại (đang dùng trong epoch này) TRƯỚC khi sched.step() thay đổi
# # # # #         cur_lr = opt.param_groups[0]["lr"]
# # # # #         sched.step()   # advance lr cho epoch tiếp theo

# # # # #         print(f"  Epoch {ep:>3}"
# # # # #               f"  loss={avg_loss:.4f}"
# # # # #               f"  cfm={avg_cfm:.4f}"
# # # # #               f"  reg={avg_reg:.4f}"
# # # # #               f"  ade_1step={avg_ade1:.0f}km"
# # # # #               f"  lr={cur_lr:.2e}"      # lr epoch này (không phải epoch sau)
# # # # #               f"  t={time.perf_counter()-t0:.0f}s")

# # # # #         # ── Periodic ckpt mỗi 10 epoch ──────────────────────────────────────
# # # # #         # Đặt TRƯỚC val block để không bị skip khi early stop break
# # # # #         if ep % 5 == 0:
# # # # #             _save(os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# # # # #                   ep, model, opt, sched, best_ade, ema,
# # # # #                   patience_cnt=patience_cnt, scaler=scaler, args=args,
# # # # #                       val_loss_history=val_loss_history, best_val_loss=best_val_loss)

# # # # #         # ── Val ───────────────────────────────────────────────────────────
# # # # #         if ep % args.val_freq == 0:
# # # # #             # Val loss: dùng val_sub (nhanh) để theo dõi overfitting
# # # # #             model.eval()
# # # # #             val_loss_buf = []
# # # # #             with torch.no_grad():
# # # # #                 for vbatch in val_sub:
# # # # #                     vbl = move(list(vbatch), device)
# # # # #                     with autocast(device_type="cuda", enabled=args.use_amp):
# # # # #                         vbd = model.get_loss_breakdown(vbl, epoch=ep)
# # # # #                     if torch.isfinite(vbd["total"]):
# # # # #                         val_loss_buf.append(vbd["total"].item())
# # # # #             avg_val_loss = float(np.mean(val_loss_buf)) if val_loss_buf else float("nan")
# # # # #             val_loss_history.append(avg_val_loss)
# # # # #             if avg_val_loss < best_val_loss:
# # # # #                 best_val_loss = avg_val_loss

# # # # #             # Trend: trung bình 3 lần gần nhất vs 3 lần trước
# # # # #             if len(val_loss_history) >= 6:
# # # # #                 recent = float(np.mean(val_loss_history[-3:]))
# # # # #                 before = float(np.mean(val_loss_history[-6:-3]))
# # # # #                 trend  = recent - before
# # # # #                 if   trend >  0.002: trend_str = f"↑ +{trend:.4f} ⚠TĂNG"
# # # # #                 elif trend < -0.002: trend_str = f"↓ {trend:.4f} ✓giảm"
# # # # #                 else:                trend_str = f"→ {trend:+.4f} flat"
# # # # #             elif len(val_loss_history) >= 2:
# # # # #                 d = val_loss_history[-1] - val_loss_history[-2]
# # # # #                 trend_str = f"{'↑' if d > 0 else '↓'} {d:+.4f}"
# # # # #             else:
# # # # #                 trend_str = "—"

# # # # #             gap     = avg_val_loss - avg_loss
# # # # #             gap_pct = gap / max(avg_loss, 1e-8) * 100
# # # # #             if   gap_pct > 50: warn = "  🔴 OVERFIT"
# # # # #             elif gap_pct > 25: warn = "  🟡 watch"
# # # # #             else:              warn = ""

# # # # #             print(f"  val_loss={avg_val_loss:.4f}  best={best_val_loss:.4f}"
# # # # #                   f"  trend={trend_str}"
# # # # #                   f"  gap={gap:+.4f}({gap_pct:+.0f}%){warn}"
# # # # #                   f"  train={avg_loss:.4f}")

# # # # #             # Val ADE: full val set để early stop không bị bias subset
# # # # #             r = evaluate(model, val_full, device,
# # # # #                          tag=f"val ep{ep}  [full {len(vd)} samples]",
# # # # #                          ddim_steps=args.ddim_steps,
# # # # #                          ema=ema)
# # # # #             ade = r["ADE"]

# # # # #             if ade < best_ade:
# # # # #                 best_ade     = ade
# # # # #                 patience_cnt = 0
# # # # #                 # Lưu ngay từ epoch 0, không chờ min_ep
# # # # #                 _save(best_ckpt, ep, model, opt, sched, best_ade, ema,
# # # # #                       patience_cnt=patience_cnt, scaler=scaler, args=args,
# # # # #                       val_loss_history=val_loss_history, best_val_loss=best_val_loss)
# # # # #                 print(f"  ✅ Best ADE = {best_ade:.2f} km  (ep {ep})")
# # # # #             else:
# # # # #                 # Patience chỉ đếm sau min_ep — tránh early stop quá sớm
# # # # #                 if ep >= args.min_ep:
# # # # #                     patience_cnt += args.val_freq
# # # # #                 print(f"  No improve {patience_cnt}/{args.patience}"
# # # # #                       f"  (best={best_ade:.1f})")
# # # # #                 if ep >= args.min_ep and patience_cnt >= args.patience:
# # # # #                     print(f"  ⛔ Early stop @ ep{ep}")
# # # # #                     break

# # # # #     print("=" * 65)
# # # # #     print(f"  Best ADE: {best_ade:.2f} km")
# # # # #     print(f"  Target:   {ST_TRANS_TARGETS['ADE']:.2f} km (ST-Trans)")
# # # # #     print(f"  Gap:      {best_ade - ST_TRANS_TARGETS['ADE']:+.2f} km")
# # # # #     print("=" * 65)

# # # # #     # ── Test ──────────────────────────────────────────────────────────────
# # # # #     if args.test_at_end and os.path.exists(best_ckpt):
# # # # #         print("\n  Test set evaluation...")
# # # # #         try:
# # # # #             _, test_loader = data_loader(
# # # # #                 args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # # #         except:
# # # # #             print("  No test set → using val")
# # # # #             _, test_loader = data_loader(
# # # # #                 args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # #         ck = torch.load(best_ckpt, map_location=device)
# # # # #         _unwrap(model).load_state_dict(ck["model"], strict=False)
# # # # #         if ema and ck.get("ema"):
# # # # #             for k, v in ck["ema"].items():
# # # # #                 if k in ema.shadow:
# # # # #                     ema.shadow[k].copy_(v.to(device))

# # # # #         evaluate(model, test_loader, device,
# # # # #                  tag="TEST (EMA best)",
# # # # #                  ddim_steps=args.ddim_steps,
# # # # #                  ema=ema)


# # # # # if __name__ == "__main__":
# # # # #     args = get_args()
# # # # #     np.random.seed(42)
# # # # #     torch.manual_seed(42)
# # # # #     if torch.cuda.is_available():
# # # # #         torch.cuda.manual_seed_all(42)
# # # # #     main(args)

# # # # """
# # # # scripts/train_fm.py  ——  TC-FlowMatching v2.1 Training
# # # # ════════════════════════════════════════════════════════

# # # # Dựa trên v2 (document được paste), với 3 fix:

# # # #   [FIX-1] augment_batch() ở train loop, KHÔNG trong get_loss_breakdown
# # # #           → val loss chính xác (không bị augment)

# # # #   [FIX-2] 1-shot inference (n_inference_steps=1, sigma_inference=0.04)
# # # #           → zero error accumulation ở 72h

# # # #   [FIX-3] Augmentation mạnh hơn: track shift + speed scale, 100%

# # # #   GIỮ NGUYÊN từ v2:
# # # #   - L_CFM + L_reg(t=0), lambda_reg ramp ep10→ep30
# # # #   - d_model=256, 4 layers — đủ capacity
# # # #   - OT matching
# # # #   - 2-group optimizer (encoder freeze) từ v4
# # # #   - Full val set cho val ADE
# # # #   - Val loss tracking + trend detection
# # # #   - Best ckpt = combined_score (0.6*ADE + 0.2*ATE + 0.2*CTE)
# # # #   - Val/Test comparison table cuối training
# # # # """
# # # # from __future__ import annotations
# # # # import sys, os
# # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # import argparse, math, time
# # # # from collections import defaultdict
# # # # from typing import Dict, Optional

# # # # import numpy as np
# # # # import torch
# # # # import torch.optim as optim
# # # # from torch.amp import autocast, GradScaler

# # # # from Model.data.loader_training import data_loader
# # # # from Model.flow_matching_model import (
# # # #     TCFlowMatching, _norm_to_deg, _haversine_deg,
# # # #     EMAModel, augment_batch,
# # # # )

# # # # HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
# # # # ST_TRANS_TARGETS = {
# # # #     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
# # # #     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# # # # }


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Utilities
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def _unwrap(m):
# # # #     return m._orig_mod if hasattr(m, "_orig_mod") else m


# # # # def move(batch, device):
# # # #     out = list(batch)
# # # #     for i, x in enumerate(out):
# # # #         if torch.is_tensor(x):
# # # #             out[i] = x.to(device)
# # # #         elif isinstance(x, dict):
# # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # #                       for k, v in x.items()}
# # # #     return out


# # # # def _ate_cte(pred_deg, gt_deg):
# # # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # # #     if T < 2:
# # # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # # #         return z, z
# # # #     lo1 = torch.deg2rad(gt_deg[:T-1,:,0]); la1 = torch.deg2rad(gt_deg[:T-1,:,1])
# # # #     lo2 = torch.deg2rad(gt_deg[1:T, :,0]); la2 = torch.deg2rad(gt_deg[1:T, :,1])
# # # #     lo3 = torch.deg2rad(pred_deg[1:T,:,0]); la3 = torch.deg2rad(pred_deg[1:T,:,1])
# # # #     ya  = torch.sin(lo2-lo1)*torch.cos(la2)
# # # #     xa  = torch.cos(la1)*torch.sin(la2) - torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # # #     be  = torch.atan2(ya, xa)
# # # #     ye  = torch.sin(lo3-lo2)*torch.cos(la3)
# # # #     xe  = torch.cos(la2)*torch.sin(la3) - torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # # #     bee = torch.atan2(ye, xe)
# # # #     tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# # # #     ang = bee - be
# # # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  2-group optimizer + gradual unfreeze (từ v4)
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def build_optimizer(model, lr_velocity, lr_encoder, weight_decay):
# # # #     raw = _unwrap(model)
# # # #     enc_params = list(raw.encoder.parameters())
# # # #     vel_params = list(raw.velocity.parameters())
# # # #     for p in enc_params + vel_params:
# # # #         p.requires_grad_(True)
# # # #     return optim.AdamW([
# # # #         {"params": enc_params, "lr": lr_encoder,  "name": "encoder"},
# # # #         {"params": vel_params, "lr": lr_velocity, "name": "velocity"},
# # # #     ], weight_decay=weight_decay)


# # # # def get_lrs(opt):
# # # #     return opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]


# # # # class TwoGroupScheduler:
# # # #     def __init__(self, opt, warmup_epochs, total_epochs,
# # # #                  lr_vel, lr_vel_min, freeze_end_ep,
# # # #                  lr_enc_peak, encoder_warmup_epochs=5):
# # # #         self.opt         = opt
# # # #         self.warmup      = warmup_epochs
# # # #         self.total       = total_epochs
# # # #         self.lr_vel      = lr_vel
# # # #         self.lr_vel_min  = lr_vel_min
# # # #         self.freeze_end  = freeze_end_ep
# # # #         self.lr_enc_peak = lr_enc_peak
# # # #         self.enc_warmup  = encoder_warmup_epochs
# # # #         self.epoch       = 0

# # # #     def _cosine(self, ep_from, ep_to, lr_start, lr_end, ep):
# # # #         t = max(0.0, min(1.0, (ep - ep_from) / max(ep_to - ep_from, 1)))
# # # #         return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t))

# # # #     def step(self):
# # # #         ep = self.epoch
# # # #         # Velocity: warmup → cosine
# # # #         if ep < self.warmup:
# # # #             lr_vel = self.lr_vel * (0.1 + 0.9 * ep / max(self.warmup-1, 1))
# # # #         else:
# # # #             lr_vel = self._cosine(self.warmup, self.total,
# # # #                                   self.lr_vel, self.lr_vel_min, ep)
# # # #         # Encoder: 0 → linear warmup → cosine
# # # #         if ep < self.freeze_end:
# # # #             lr_enc = 0.0
# # # #         elif ep < self.freeze_end + self.enc_warmup:
# # # #             lr_enc = self.lr_enc_peak * (ep - self.freeze_end) / self.enc_warmup
# # # #         else:
# # # #             lr_enc = self._cosine(
# # # #                 self.freeze_end + self.enc_warmup, self.total,
# # # #                 self.lr_enc_peak, self.lr_vel_min, ep)
# # # #         self.opt.param_groups[0]["lr"] = lr_enc
# # # #         self.opt.param_groups[1]["lr"] = lr_vel
# # # #         self.epoch += 1
# # # #         return lr_vel, lr_enc


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Evaluation — FULL val, NO augment, in val loss đầy đủ
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # @torch.no_grad()
# # # # def evaluate(model, loader, device, tag="",
# # # #              n_ensemble=20, ema=None, epoch_for_loss=9999) -> Dict:
# # # #     """
# # # #     [FIX-1] KHÔNG augment.
# # # #     In đầy đủ: val loss (l_cfm, l_reg, total) + ADE/ATE/CTE + lead times.
# # # #     Loss weighted by batch size để chính xác.
# # # #     """
# # # #     bk = None
# # # #     if ema is not None:
# # # #         try:
# # # #             bk = ema.apply_to(model)
# # # #         except Exception as e:
# # # #             print(f"    ⚠ EMA apply: {e}")

# # # #     model.eval()
# # # #     all_ade, all_ate, all_cte = [], [], []
# # # #     step_dist = defaultdict(list)

# # # #     # Loss accumulation weighted by batch size
# # # #     sum_total_w = sum_cfm_w = sum_reg_w = 0.0
# # # #     sum_n = 0

# # # #     for batch in loader:
# # # #         bl = move(list(batch), device)
# # # #         gt = bl[1]
# # # #         B  = bl[0].shape[1]

# # # #         # [FIX-1] Val loss: KHÔNG augment
# # # #         try:
# # # #             bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
# # # #             if torch.isfinite(bd["total"]):
# # # #                 sum_total_w += bd["total"].item() * B
# # # #                 sum_cfm_w   += bd["l_cfm"] * B
# # # #                 sum_reg_w   += bd["l_reg"] * B
# # # #                 sum_n       += B
# # # #         except Exception:
# # # #             pass

# # # #         # 1-shot sample (ddim_steps=1 → 1-shot via n_inference_steps=1)
# # # #         try:
# # # #             pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
# # # #         except Exception as e:
# # # #             print(f"  sample error: {e}")
# # # #             continue

# # # #         T        = min(pred.shape[0], gt.shape[0])
# # # #         pred_deg = _norm_to_deg(pred[:T])
# # # #         gt_deg   = _norm_to_deg(gt[:T])
# # # #         dist     = _haversine_deg(pred_deg, gt_deg)
# # # #         ate, cte = _ate_cte(pred_deg, gt_deg)

# # # #         all_ade.extend(dist.mean(0).tolist())
# # # #         if ate.shape[0] > 0:
# # # #             all_ate.extend(ate.abs().mean(0).tolist())
# # # #             all_cte.extend(cte.abs().mean(0).tolist())
# # # #         for h, s in HORIZON_STEPS.items():
# # # #             if s < T:
# # # #                 step_dist[h].extend(dist[s].tolist())

# # # #     if bk is not None:
# # # #         try:
# # # #             ema.restore(model, bk)
# # # #         except Exception:
# # # #             pass

# # # #     def _m(lst): return float(np.mean(lst)) if lst else float("nan")

# # # #     val_loss     = sum_total_w / max(sum_n, 1)
# # # #     val_cfm_loss = sum_cfm_w   / max(sum_n, 1)
# # # #     val_reg_loss = sum_reg_w   / max(sum_n, 1)

# # # #     result = {
# # # #         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
# # # #         "n":   len(all_ade),
# # # #         "val_loss":     val_loss,
# # # #         "val_cfm_loss": val_cfm_loss,
# # # #         "val_reg_loss": val_reg_loss,
# # # #     }
# # # #     for h in HORIZON_STEPS:
# # # #         result[f"{h}h"] = _m(step_dist[h])

# # # #     ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
# # # #     result["combined_score"] = (
# # # #         0.6*ade + 0.2*ate_ + 0.2*cte_
# # # #         if all(np.isfinite(x) for x in [ade, ate_, cte_]) else ade)

# # # #     # ── Print ─────────────────────────────────────────────────────────────
# # # #     def _v(k): return result.get(k, float("nan"))
# # # #     def _ok(k, ref): return "✓" if np.isfinite(_v(k)) and _v(k) < ref else "✗"

# # # #     print(f"\n  {'='*70}")
# # # #     print(f"  [{tag}]  n={result['n']} seqs  (1-shot, NO augment)")
# # # #     print(f"  {'─'*70}")
# # # #     print(f"  Val Loss : total={val_loss:.6f}"
# # # #           f"  cfm={val_cfm_loss:.6f}"
# # # #           f"  reg={val_reg_loss:.6f}"
# # # #           f"  (weighted mean, {sum_n} samples)")
# # # #     print(f"  {'─'*70}")
# # # #     print(f"  ADE = {_v('ADE'):7.1f} km  {_ok('ADE', ST_TRANS_TARGETS['ADE'])}"
# # # #           f"  (< {ST_TRANS_TARGETS['ADE']:.0f})")
# # # #     print(f"  ATE = {_v('ATE'):7.1f} km  {_ok('ATE', ST_TRANS_TARGETS['ATE'])}"
# # # #           f"  (< {ST_TRANS_TARGETS['ATE']:.0f})")
# # # #     print(f"  CTE = {_v('CTE'):7.1f} km  {_ok('CTE', ST_TRANS_TARGETS['CTE'])}"
# # # #           f"  (< {ST_TRANS_TARGETS['CTE']:.0f})")
# # # #     print(f"  Combined = {_v('combined_score'):.1f}"
# # # #           f"  (0.6×ADE + 0.2×ATE + 0.2×CTE)")
# # # #     print(f"  {'─'*70}")
# # # #     print(f"  12h={_v('12h'):6.1f}  24h={_v('24h'):6.1f}"
# # # #           f"  48h={_v('48h'):6.1f}  72h={_v('72h'):6.1f} km")
# # # #     beat = [f"{k}={_v(k):.0f}<{ref:.0f}"
# # # #             for k, ref in ST_TRANS_TARGETS.items()
# # # #             if np.isfinite(_v(k)) and _v(k) < ref]
# # # #     print(f"  {'─'*70}")
# # # #     print(f"  BEAT ST-Trans: {' | '.join(beat) if beat else 'none yet'}")
# # # #     print(f"  {'='*70}\n")
# # # #     return result


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Checkpoint
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def _save(path, epoch, model, opt, sched, best_score,
# # # #           ema=None, scaler=None, extra=None):
# # # #     m   = _unwrap(model)
# # # #     esd = None
# # # #     if ema is not None:
# # # #         try:
# # # #             esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # # #         except Exception:
# # # #             pass
# # # #     payload = {
# # # #         "epoch": epoch, "model": m.state_dict(),
# # # #         "optimizer": opt.state_dict(),
# # # #         "scheduler": sched.epoch,
# # # #         "scaler": scaler.state_dict() if scaler is not None else None,
# # # #         "best_score": best_score,
# # # #         "best_ade":   best_score,   # compat
# # # #         "ema": esd,
# # # #     }
# # # #     if extra:
# # # #         payload.update(extra)
# # # #     torch.save(payload, path)


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Args
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def get_args():
# # # #     p = argparse.ArgumentParser(
# # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # #     p.add_argument("--dataset_root",           default="TCND_vn")
# # # #     p.add_argument("--obs_len",                default=8,      type=int)
# # # #     p.add_argument("--pred_len",               default=12,     type=int)
# # # #     p.add_argument("--num_workers",            default=2,      type=int)
# # # #     p.add_argument("--other_modal",            default="gph")
# # # #     p.add_argument("--delim",                  default=" ")
# # # #     p.add_argument("--skip",                   default=1,      type=int)
# # # #     p.add_argument("--min_ped",                default=1,      type=int)
# # # #     p.add_argument("--threshold",              default=0.002,  type=float)
# # # #     # Model — giữ v2 defaults (256d, 4 layers)
# # # #     p.add_argument("--d_cond",                 default=256,    type=int)
# # # #     p.add_argument("--d_model",                default=256,    type=int)
# # # #     p.add_argument("--nhead",                  default=8,      type=int)
# # # #     p.add_argument("--num_dec_layers",         default=4,      type=int)
# # # #     p.add_argument("--dim_ff",                 default=512,    type=int)
# # # #     p.add_argument("--dropout",                default=0.1,    type=float)
# # # #     p.add_argument("--unet_in_ch",             default=13,     type=int)
# # # #     p.add_argument("--sigma_min",              default=0.04,   type=float)
# # # #     p.add_argument("--sigma_max",              default=0.08,   type=float)
# # # #     p.add_argument("--lambda_reg",             default=0.2,    type=float)
# # # #     p.add_argument("--use_ot",                 default=True,   action="store_true")
# # # #     p.add_argument("--ot_epsilon",             default=0.05,   type=float)
# # # #     p.add_argument("--no_ot",                  dest="use_ot",  action="store_false")
# # # #     p.add_argument("--n_ensemble",             default=20,     type=int)
# # # #     p.add_argument("--sigma_inference",        default=0.04,   type=float,
# # # #                    help="[FIX-2] match sigma_min để tránh mismatch ở epoch 40+")
# # # #     p.add_argument("--n_inference_steps",      default=1,      type=int,
# # # #                    help="[FIX-2] 1=1-shot, >1=Euler")
# # # #     # Training
# # # #     p.add_argument("--num_epochs",             default=150,    type=int)
# # # #     p.add_argument("--batch_size",             default=64,     type=int)
# # # #     p.add_argument("--lr",                     default=2e-4,   type=float)
# # # #     p.add_argument("--lr_min",                 default=1e-6,   type=float)
# # # #     p.add_argument("--warmup_epochs",          default=5,      type=int)
# # # #     p.add_argument("--weight_decay",           default=1e-4,   type=float)
# # # #     p.add_argument("--grad_clip",              default=1.0,    type=float)
# # # #     p.add_argument("--use_amp",                action="store_true", default=False)
# # # #     p.add_argument("--use_ema",                default=True,   action="store_true")
# # # #     p.add_argument("--no_ema",                 dest="use_ema", action="store_false")
# # # #     p.add_argument("--ema_decay",              default=0.995,  type=float)
# # # #     # Encoder freeze (từ v4)
# # # #     p.add_argument("--freeze_encoder_epochs",  default=10,     type=int)
# # # #     p.add_argument("--encoder_warmup_epochs",  default=5,      type=int)
# # # #     p.add_argument("--lr_enc_peak",            default=5e-5,   type=float)
# # # #     # Eval
# # # #     p.add_argument("--val_freq",               default=5,      type=int)
# # # #     p.add_argument("--patience",               default=30,     type=int)
# # # #     p.add_argument("--min_ep",                 default=20,     type=int)
# # # #     # IO
# # # #     p.add_argument("--output_dir",             default="runs/fm_v2")
# # # #     p.add_argument("--gpu_num",                default="0")
# # # #     p.add_argument("--resume",                 default=None)
# # # #     p.add_argument("--test_at_end",            action="store_true", default=True)
# # # #     return p.parse_args()


# # # # # ─────────────────────────────────────────────────────────────────────────────
# # # # #  Main
# # # # # ─────────────────────────────────────────────────────────────────────────────

# # # # def main(args):
# # # #     if torch.cuda.is_available():
# # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # #     os.makedirs(args.output_dir, exist_ok=True)
# # # #     best_ckpt = os.path.join(args.output_dir, "best_model.pth")
# # # #     last_ckpt = os.path.join(args.output_dir, "last_model.pth")

# # # #     print("=" * 72)
# # # #     print("  TC-FlowMatching v2.1")
# # # #     print(f"  [FIX-1] augment_batch() ở train loop, val KHÔNG augment")
# # # #     print(f"  [FIX-2] 1-shot inference (n_steps={args.n_inference_steps},"
# # # #           f" sigma_inf={args.sigma_inference})")
# # # #     print(f"  [FIX-3] Augment mạnh: track_shift±5km + speed_scale×0.85-1.15, 100%")
# # # #     print(f"  GIỮ: d_model={args.d_model}, layers={args.num_dec_layers},"
# # # #           f" L_CFM+L_reg, OT")
# # # #     print(f"  Encoder freeze {args.freeze_encoder_epochs} ep"
# # # #           f" → warmup {args.encoder_warmup_epochs} ep"
# # # #           f" → lr_peak={args.lr_enc_peak:.0e}")
# # # #     print("=" * 72)

# # # #     # ── Data ──────────────────────────────────────────────────────────────
# # # #     print("\n  Loading data...")
# # # #     trd, trl = data_loader(
# # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # #     vd, val_loader = data_loader(
# # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)
# # # #     print(f"  train : {len(trd)} ({len(trl)} batches/ep)")
# # # #     print(f"  val   : {len(vd)} ({len(val_loader)} batches) ← FULL VAL")

# # # #     # ── Model ─────────────────────────────────────────────────────────────
# # # #     model = TCFlowMatching(
# # # #         pred_len=args.pred_len,        obs_len=args.obs_len,
# # # #         unet_in_ch=args.unet_in_ch,   d_cond=args.d_cond,
# # # #         d_model=args.d_model,          nhead=args.nhead,
# # # #         num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
# # # #         dropout=args.dropout,
# # # #         sigma_min=args.sigma_min,      sigma_max=args.sigma_max,
# # # #         lambda_reg=args.lambda_reg,    use_ot=args.use_ot,
# # # #         ot_epsilon=args.ot_epsilon,    use_ema=args.use_ema,
# # # #         n_ensemble=args.n_ensemble,
# # # #         n_inference_steps=args.n_inference_steps,
# # # #         sigma_inference=args.sigma_inference,
# # # #     ).to(device)

# # # #     model.init_ema()
# # # #     ema = getattr(_unwrap(model), "_ema", None)

# # # #     raw     = _unwrap(model)
# # # #     n_enc   = sum(p.numel() for p in raw.encoder.parameters())
# # # #     n_vel   = sum(p.numel() for p in raw.velocity.parameters())
# # # #     print(f"\n  Encoder       : {n_enc:,} params")
# # # #     print(f"  VelocityTrans : {n_vel:,} params")
# # # #     print(f"  Total         : {n_enc+n_vel:,} params")

# # # #     # ── 2-group optimizer ─────────────────────────────────────────────────
# # # #     lr_enc_init = 0.0 if args.freeze_encoder_epochs > 0 else args.lr_enc_peak
# # # #     opt = build_optimizer(model, lr_velocity=args.lr,
# # # #                           lr_encoder=lr_enc_init,
# # # #                           weight_decay=args.weight_decay)
# # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # #     sched  = TwoGroupScheduler(
# # # #         opt=opt,
# # # #         warmup_epochs=args.warmup_epochs,
# # # #         total_epochs=args.num_epochs,
# # # #         lr_vel=args.lr,
# # # #         lr_vel_min=args.lr_min,
# # # #         freeze_end_ep=args.freeze_encoder_epochs,
# # # #         lr_enc_peak=args.lr_enc_peak,
# # # #         encoder_warmup_epochs=args.encoder_warmup_epochs,
# # # #     )

# # # #     print(f"\n  LR velocity   : {args.lr:.0e} → {args.lr_min:.0e}")
# # # #     print(f"  LR encoder    : 0 ({args.freeze_encoder_epochs} ep)"
# # # #           f" → {args.lr_enc_peak:.0e} ({args.encoder_warmup_epochs} ep)"
# # # #           f" → cosine")

# # # #     # ── Resume ────────────────────────────────────────────────────────────
# # # #     start_ep     = 0
# # # #     best_score   = float("inf")
# # # #     patience_cnt = 0
# # # #     val_loss_history = []

# # # #     if args.resume and os.path.exists(args.resume):
# # # #         ck = torch.load(args.resume, map_location=device)
# # # #         _unwrap(model).load_state_dict(ck["model"], strict=False)
# # # #         try:
# # # #             opt.load_state_dict(ck["optimizer"])
# # # #         except Exception as e:
# # # #             print(f"  ⚠ Opt state: {e}")
# # # #         sched.epoch  = ck.get("scheduler", 0)
# # # #         start_ep     = ck.get("epoch", 0) + 1
# # # #         best_score   = ck.get("best_score", ck.get("best_ade", float("inf")))
# # # #         patience_cnt = ck.get("patience_cnt", 0)
# # # #         if scaler is not None and ck.get("scaler"):
# # # #             try:
# # # #                 scaler.load_state_dict(ck["scaler"])
# # # #             except Exception:
# # # #                 pass
# # # #         if ema and ck.get("ema"):
# # # #             for k, v in ck["ema"].items():
# # # #                 if k in ema.shadow:
# # # #                     ema.shadow[k].copy_(v.to(device))
# # # #         print(f"  ↩ Resumed ep{start_ep}  best_score={best_score:.1f}"
# # # #               f"  patience={patience_cnt}/{args.patience}")

# # # #     try:
# # # #         model = torch.compile(model, mode="reduce-overhead")
# # # #         print("  torch.compile: ok")
# # # #     except Exception:
# # # #         pass

# # # #     print()
# # # #     print("=" * 72)
# # # #     print(f"  TRAINING  ({len(trl)} steps/ep × {args.num_epochs} ep)")
# # # #     print(f"  Train: augment=True  Val: augment=False  ← FIX-1")
# # # #     print(f"  Inference: 1-shot at t=0  ← FIX-2")
# # # #     print("=" * 72)

# # # #     nstep = len(trl)

# # # #     for ep in range(start_ep, args.num_epochs):
# # # #         model.train()
# # # #         sum_loss_w = 0.0; sum_n_train = 0
# # # #         sum_cfm = sum_reg = sum_ade1 = 0.0
# # # #         t0 = time.perf_counter()

# # # #         for i, batch in enumerate(trl):
# # # #             bl = move(list(batch), device)
# # # #             B  = bl[0].shape[1]

# # # #             # [FIX-1] Augment CHỈ ở train loop
# # # #             bl_aug = augment_batch(bl)

# # # #             opt.zero_grad()
# # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # #                 bd = model.get_loss_breakdown(bl_aug, epoch=ep)

# # # #             scaler.scale(bd["total"]).backward()
# # # #             scaler.unscale_(opt)
# # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # #             scaler.step(opt)
# # # #             scaler.update()
# # # #             model.ema_update()

# # # #             sum_loss_w  += bd["total"].item() * B
# # # #             sum_n_train += B
# # # #             sum_cfm     += bd["l_cfm"]
# # # #             sum_reg     += bd["l_reg"]
# # # #             sum_ade1    += bd["ade_1step"]

# # # #             if i % 30 == 0:
# # # #                 lr_enc, lr_vel = get_lrs(opt)
# # # #                 print(f"  [{ep:>3}][{i:>3}/{nstep}]"
# # # #                       f"  total={bd['total'].item():.4f}"
# # # #                       f"  cfm={bd['l_cfm']:.4f}"
# # # #                       f"  reg={bd['l_reg']:.4f}"
# # # #                       f"  lam={bd['lam_reg']:.3f}"
# # # #                       f"  ade1={bd['ade_1step']:.0f}km"
# # # #                       f"  σ={bd['sigma']:.3f}"
# # # #                       f"  lr_vel={lr_vel:.2e}"
# # # #                       f"  lr_enc={lr_enc:.2e}")

# # # #         train_loss = sum_loss_w / max(sum_n_train, 1)
# # # #         avg_cfm    = sum_cfm / nstep
# # # #         avg_reg    = sum_reg / nstep
# # # #         avg_ade1   = sum_ade1 / nstep
# # # #         lr_vel, lr_enc = sched.step()

# # # #         # Encoder unfreeze log
# # # #         if ep == args.freeze_encoder_epochs:
# # # #             print(f"\n  *** Ep{ep}: encoder warmup 0→{args.lr_enc_peak:.0e}"
# # # #                   f" ({args.encoder_warmup_epochs} ep) ***")
# # # #         if ep == args.freeze_encoder_epochs + args.encoder_warmup_epochs:
# # # #             print(f"\n  *** Ep{ep}: encoder fully active ***")

# # # #         print(f"\n  ── Ep{ep:>3}"
# # # #               f"  train_loss={train_loss:.6f}"
# # # #               f"  cfm={avg_cfm:.4f}"
# # # #               f"  reg={avg_reg:.4f}"
# # # #               f"  ade1={avg_ade1:.0f}km"
# # # #               f"  lr_vel={lr_vel:.2e}"
# # # #               f"  lr_enc={lr_enc:.2e}"
# # # #               f"  t={time.perf_counter()-t0:.0f}s")

# # # #         _save(last_ckpt, ep, model, opt, sched, best_score, ema, scaler)

# # # #         # ── Val ───────────────────────────────────────────────────────────
# # # #         if ep % args.val_freq == 0:
# # # #             print(f"\n  ── Val FULL SET (ep{ep}) ──")
# # # #             r = evaluate(model, val_loader, device,
# # # #                          tag=f"VAL ep{ep}",
# # # #                          n_ensemble=args.n_ensemble,
# # # #                          ema=ema,
# # # #                          epoch_for_loss=ep)

# # # #             val_loss = r["val_loss"]
# # # #             val_ade  = r["ADE"]
# # # #             val_loss_history.append(val_ade)   # track ADE trend (reliable hơn loss gap)
# # # #             score    = r["combined_score"]

# # # #             # [FIX-7] Overfit detection bằng val ADE trend
# # # #             # KHÔNG dùng val_loss vs train_loss gap vì:
# # # #             #   train_loss có augment → distribution khác → gap không reliable
# # # #             # Thay bằng: val ADE tăng liên tục = overfit signal
# # # #             if len(val_loss_history) >= 4:
# # # #                 recent = float(np.mean(val_loss_history[-2:]))
# # # #                 before = float(np.mean(val_loss_history[-4:-2]))
# # # #                 trend  = recent - before
# # # #                 if   trend >  5.0:  trend_str = f"ADE↑{trend:+.1f}km ⚠TĂNG"
# # # #                 elif trend < -5.0:  trend_str = f"ADE↓{trend:+.1f}km ✓giảm"
# # # #                 else:               trend_str = f"ADE→{trend:+.1f}km flat"
# # # #             else:
# # # #                 trend_str = "—"

# # # #             print(f"  train_loss={train_loss:.6f}"
# # # #                   f"  val_loss={val_loss:.6f}"
# # # #                   f"  val_ADE={val_ade:.1f}km"
# # # #                   f"  trend={trend_str}")
# # # #             print(f"  combined={score:.1f}  best={best_score:.1f}")

# # # #             if score < best_score and ep >= args.min_ep:
# # # #                 best_score   = score
# # # #                 patience_cnt = 0
# # # #                 _save(best_ckpt, ep, model, opt, sched, best_score, ema, scaler,
# # # #                       extra={"val_ade":   r["ADE"],
# # # #                              "val_ate":   r["ATE"],
# # # #                              "val_cte":   r["CTE"],
# # # #                              "val_loss":  val_loss,
# # # #                              "val_score": score,
# # # #                              "patience_cnt": patience_cnt})
# # # #                 print(f"  ✅ New best! score={best_score:.2f}"
# # # #                       f"  ADE={r['ADE']:.1f} ATE={r['ATE']:.1f}"
# # # #                       f"  CTE={r['CTE']:.1f} → {best_ckpt}")
# # # #             else:
# # # #                 if ep >= args.min_ep:
# # # #                     patience_cnt += args.val_freq
# # # #                 print(f"  No improve {patience_cnt}/{args.patience}")
# # # #                 if ep >= args.min_ep and patience_cnt >= args.patience:
# # # #                     print(f"  ⛔ Early stop @ ep{ep}")
# # # #                     break

# # # #         if ep % 5 == 0:
# # # #             ckpt_ep = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
# # # #             _save(ckpt_ep, ep, model, opt, sched, best_score, ema, scaler)
# # # #             print(f"  💾 {ckpt_ep}")

# # # #     print("\n" + "=" * 72)
# # # #     print(f"  Done! best_score={best_score:.2f}")
# # # #     print("=" * 72)

# # # #     # ── Test ──────────────────────────────────────────────────────────────
# # # #     if args.test_at_end and os.path.exists(best_ckpt):
# # # #         print("\n  Loading best ckpt for TEST...")
# # # #         ck = torch.load(best_ckpt, map_location=device)
# # # #         _unwrap(model).load_state_dict(ck["model"], strict=False)
# # # #         if ema and ck.get("ema"):
# # # #             for k, v in ck["ema"].items():
# # # #                 if k in ema.shadow:
# # # #                     ema.shadow[k].copy_(v.to(device))

# # # #         try:
# # # #             _, test_loader = data_loader(
# # # #                 args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # #             print(f"  Test: {len(test_loader)} batches")
# # # #         except Exception:
# # # #             print("  No test set → using val")
# # # #             test_loader = val_loader

# # # #         r_test = evaluate(model, test_loader, device,
# # # #                           tag="TEST (best ckpt, NO augment)",
# # # #                           n_ensemble=args.n_ensemble, ema=ema)

# # # #         val_ade  = ck.get("val_ade",  float("nan"))
# # # #         val_ate  = ck.get("val_ate",  float("nan"))
# # # #         val_cte  = ck.get("val_cte",  float("nan"))
# # # #         val_loss_saved = ck.get("val_loss", float("nan"))

# # # #         print()
# # # #         print("=" * 72)
# # # #         print("  VAL vs TEST (1-shot, kỳ vọng gap < 20km)")
# # # #         print("=" * 72)
# # # #         print(f"  {'Metric':<12}  {'Val':>10}  {'Test':>10}  {'Gap':>10}  Status")
# # # #         print("  " + "─" * 58)
# # # #         for m, v, t in [("ADE (km)",  val_ade,        r_test["ADE"]),
# # # #                         ("ATE (km)",  val_ate,        r_test["ATE"]),
# # # #                         ("CTE (km)",  val_cte,        r_test["CTE"]),
# # # #                         ("Loss",      val_loss_saved, r_test["val_loss"])]:
# # # #             gap  = t - v if (np.isfinite(t) and np.isfinite(v)) else float("nan")
# # # #             flag = ("✅" if np.isfinite(gap) and abs(gap) < 20
# # # #                     else "⚠" if np.isfinite(gap) else "?")
# # # #             print(f"  {m:<12}  {v:>10.2f}  {t:>10.2f}  {gap:>+10.2f}  {flag}")
# # # #         print("=" * 72)


# # # # if __name__ == "__main__":
# # # #     args = get_args()
# # # #     np.random.seed(42); torch.manual_seed(42)
# # # #     if torch.cuda.is_available():
# # # #         torch.cuda.manual_seed_all(42)

# # # #     if args.dataset_root == "TCND_vn":
# # # #         _auto = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
# # # #         if os.path.isdir(_auto):
# # # #             args.dataset_root = _auto
# # # #             print(f"  Auto dataset: {_auto}")

# # # #     main(args)

# # # """
# # # scripts/train_fm.py  ——  TC-FlowMatching v2.1 Training
# # # ════════════════════════════════════════════════════════

# # # Dựa trên v2 (document được paste), với 3 fix:

# # #   [FIX-1] augment_batch() ở train loop, KHÔNG trong get_loss_breakdown
# # #           → val loss chính xác (không bị augment)

# # #   [FIX-2] 1-shot inference (n_inference_steps=1, sigma_inference=0.04)
# # #           → zero error accumulation ở 72h

# # #   [FIX-3] Augmentation mạnh hơn: track shift + speed scale, 100%

# # #   GIỮ NGUYÊN từ v2:
# # #   - L_CFM + L_reg(t=0), lambda_reg ramp ep10→ep30
# # #   - d_model=256, 4 layers — đủ capacity
# # #   - OT matching
# # #   - 2-group optimizer (encoder freeze) từ v4
# # #   - Full val set cho val ADE
# # #   - Val loss tracking + trend detection
# # #   - Best ckpt = combined_score (0.6*ADE + 0.2*ATE + 0.2*CTE)
# # #   - Val/Test comparison table cuối training
# # # """
# # # from __future__ import annotations
# # # import sys, os
# # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # import argparse, math, time
# # # from collections import defaultdict
# # # from typing import Dict, Optional

# # # import numpy as np
# # # import torch
# # # import torch.optim as optim
# # # from torch.amp import autocast, GradScaler

# # # from Model.data.loader_training import data_loader
# # # from Model.flow_matching_model import (
# # #     TCFlowMatching, _norm_to_deg, _haversine_deg,
# # #     EMAModel, augment_batch,
# # # )

# # # HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
# # # ST_TRANS_TARGETS = {
# # #     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
# # #     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# # # }


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Utilities
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _unwrap(m):
# # #     return m._orig_mod if hasattr(m, "_orig_mod") else m


# # # def move(batch, device):
# # #     out = list(batch)
# # #     for i, x in enumerate(out):
# # #         if torch.is_tensor(x):
# # #             out[i] = x.to(device)
# # #         elif isinstance(x, dict):
# # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # #                       for k, v in x.items()}
# # #     return out


# # # def _ate_cte(pred_deg, gt_deg):
# # #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# # #     if T < 2:
# # #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# # #         return z, z
# # #     lo1 = torch.deg2rad(gt_deg[:T-1,:,0]); la1 = torch.deg2rad(gt_deg[:T-1,:,1])
# # #     lo2 = torch.deg2rad(gt_deg[1:T, :,0]); la2 = torch.deg2rad(gt_deg[1:T, :,1])
# # #     lo3 = torch.deg2rad(pred_deg[1:T,:,0]); la3 = torch.deg2rad(pred_deg[1:T,:,1])
# # #     ya  = torch.sin(lo2-lo1)*torch.cos(la2)
# # #     xa  = torch.cos(la1)*torch.sin(la2) - torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
# # #     be  = torch.atan2(ya, xa)
# # #     ye  = torch.sin(lo3-lo2)*torch.cos(la3)
# # #     xe  = torch.cos(la2)*torch.sin(la3) - torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
# # #     bee = torch.atan2(ye, xe)
# # #     tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# # #     ang = bee - be
# # #     return tot*torch.cos(ang), tot*torch.sin(ang)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  2-group optimizer + gradual unfreeze (từ v4)
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def build_optimizer(model, lr_velocity, lr_encoder, weight_decay):
# # #     raw = _unwrap(model)
# # #     enc_params = list(raw.encoder.parameters())
# # #     vel_params = list(raw.velocity.parameters())
# # #     for p in enc_params + vel_params:
# # #         p.requires_grad_(True)
# # #     return optim.AdamW([
# # #         {"params": enc_params, "lr": lr_encoder,  "name": "encoder"},
# # #         {"params": vel_params, "lr": lr_velocity, "name": "velocity"},
# # #     ], weight_decay=weight_decay)


# # # def get_lrs(opt):
# # #     return opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]


# # # class TwoGroupScheduler:
# # #     def __init__(self, opt, warmup_epochs, total_epochs,
# # #                  lr_vel, lr_vel_min, freeze_end_ep,
# # #                  lr_enc_peak, encoder_warmup_epochs=5):
# # #         self.opt         = opt
# # #         self.warmup      = warmup_epochs
# # #         self.total       = total_epochs
# # #         self.lr_vel      = lr_vel
# # #         self.lr_vel_min  = lr_vel_min
# # #         self.freeze_end  = freeze_end_ep
# # #         self.lr_enc_peak = lr_enc_peak
# # #         self.enc_warmup  = encoder_warmup_epochs
# # #         self.epoch       = 0

# # #     def _cosine(self, ep_from, ep_to, lr_start, lr_end, ep):
# # #         t = max(0.0, min(1.0, (ep - ep_from) / max(ep_to - ep_from, 1)))
# # #         return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t))

# # #     def step(self):
# # #         ep = self.epoch
# # #         # Velocity: warmup → cosine
# # #         if ep < self.warmup:
# # #             lr_vel = self.lr_vel * (0.1 + 0.9 * ep / max(self.warmup-1, 1))
# # #         else:
# # #             lr_vel = self._cosine(self.warmup, self.total,
# # #                                   self.lr_vel, self.lr_vel_min, ep)
# # #         # Encoder: 0 → linear warmup → cosine
# # #         if ep < self.freeze_end:
# # #             lr_enc = 0.0
# # #         elif ep < self.freeze_end + self.enc_warmup:
# # #             lr_enc = self.lr_enc_peak * (ep - self.freeze_end) / self.enc_warmup
# # #         else:
# # #             lr_enc = self._cosine(
# # #                 self.freeze_end + self.enc_warmup, self.total,
# # #                 self.lr_enc_peak, self.lr_vel_min, ep)
# # #         self.opt.param_groups[0]["lr"] = lr_enc
# # #         self.opt.param_groups[1]["lr"] = lr_vel
# # #         self.epoch += 1
# # #         return lr_vel, lr_enc


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Evaluation — FULL val, NO augment, in val loss đầy đủ
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # @torch.no_grad()
# # # def evaluate(model, loader, device, tag="",
# # #              n_ensemble=20, ema=None, epoch_for_loss=9999) -> Dict:
# # #     """
# # #     [FIX-1] KHÔNG augment.
# # #     In đầy đủ: val loss (l_cfm, l_reg, total) + ADE/ATE/CTE + lead times.
# # #     Loss weighted by batch size để chính xác.
# # #     """
# # #     bk = None
# # #     if ema is not None:
# # #         try:
# # #             bk = ema.apply_to(model)
# # #         except Exception as e:
# # #             print(f"    ⚠ EMA apply: {e}")

# # #     model.eval()
# # #     all_ade, all_ate, all_cte = [], [], []
# # #     step_dist = defaultdict(list)

# # #     # Loss accumulation weighted by batch size
# # #     sum_total_w = sum_cfm_w = sum_reg_w = 0.0
# # #     sum_n = 0

# # #     for batch in loader:
# # #         bl = move(list(batch), device)
# # #         gt = bl[1]
# # #         B  = bl[0].shape[1]

# # #         # [FIX-1] Val loss: KHÔNG augment
# # #         try:
# # #             bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
# # #             if torch.isfinite(bd["total"]):
# # #                 sum_total_w += bd["total"].item() * B
# # #                 sum_cfm_w   += bd["l_cfm"] * B
# # #                 sum_reg_w   += bd["l_reg"] * B
# # #                 sum_n       += B
# # #         except Exception:
# # #             pass

# # #         # 1-shot sample (ddim_steps=1 → 1-shot via n_inference_steps=1)
# # #         try:
# # #             pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
# # #         except Exception as e:
# # #             print(f"  sample error: {e}")
# # #             continue

# # #         T        = min(pred.shape[0], gt.shape[0])
# # #         pred_deg = _norm_to_deg(pred[:T])
# # #         gt_deg   = _norm_to_deg(gt[:T])
# # #         dist     = _haversine_deg(pred_deg, gt_deg)
# # #         ate, cte = _ate_cte(pred_deg, gt_deg)

# # #         all_ade.extend(dist.mean(0).tolist())
# # #         if ate.shape[0] > 0:
# # #             all_ate.extend(ate.abs().mean(0).tolist())
# # #             all_cte.extend(cte.abs().mean(0).tolist())
# # #         for h, s in HORIZON_STEPS.items():
# # #             if s < T:
# # #                 step_dist[h].extend(dist[s].tolist())

# # #     if bk is not None:
# # #         try:
# # #             ema.restore(model, bk)
# # #         except Exception:
# # #             pass

# # #     def _m(lst): return float(np.mean(lst)) if lst else float("nan")

# # #     val_loss     = sum_total_w / max(sum_n, 1)
# # #     val_cfm_loss = sum_cfm_w   / max(sum_n, 1)
# # #     val_reg_loss = sum_reg_w   / max(sum_n, 1)

# # #     result = {
# # #         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
# # #         "n":   len(all_ade),
# # #         "val_loss":     val_loss,
# # #         "val_cfm_loss": val_cfm_loss,
# # #         "val_reg_loss": val_reg_loss,
# # #     }
# # #     for h in HORIZON_STEPS:
# # #         result[f"{h}h"] = _m(step_dist[h])

# # #     ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
# # #     result["combined_score"] = (
# # #         0.6*ade + 0.2*ate_ + 0.2*cte_
# # #         if all(np.isfinite(x) for x in [ade, ate_, cte_]) else ade)

# # #     # ── Print ─────────────────────────────────────────────────────────────
# # #     def _v(k): return result.get(k, float("nan"))
# # #     def _ok(k, ref): return "✓" if np.isfinite(_v(k)) and _v(k) < ref else "✗"

# # #     print(f"\n  {'='*70}")
# # #     print(f"  [{tag}]  n={result['n']} seqs  (1-shot, NO augment)")
# # #     print(f"  {'─'*70}")
# # #     print(f"  Val Loss : total={val_loss:.6f}"
# # #           f"  cfm={val_cfm_loss:.6f}"
# # #           f"  reg={val_reg_loss:.6f}"
# # #           f"  (weighted mean, {sum_n} samples)")
# # #     print(f"  {'─'*70}")
# # #     print(f"  ADE = {_v('ADE'):7.1f} km  {_ok('ADE', ST_TRANS_TARGETS['ADE'])}"
# # #           f"  (< {ST_TRANS_TARGETS['ADE']:.0f})")
# # #     print(f"  ATE = {_v('ATE'):7.1f} km  {_ok('ATE', ST_TRANS_TARGETS['ATE'])}"
# # #           f"  (< {ST_TRANS_TARGETS['ATE']:.0f})")
# # #     print(f"  CTE = {_v('CTE'):7.1f} km  {_ok('CTE', ST_TRANS_TARGETS['CTE'])}"
# # #           f"  (< {ST_TRANS_TARGETS['CTE']:.0f})")
# # #     print(f"  Combined = {_v('combined_score'):.1f}"
# # #           f"  (0.6×ADE + 0.2×ATE + 0.2×CTE)")
# # #     print(f"  {'─'*70}")
# # #     print(f"  12h={_v('12h'):6.1f}  24h={_v('24h'):6.1f}"
# # #           f"  48h={_v('48h'):6.1f}  72h={_v('72h'):6.1f} km")
# # #     beat = [f"{k}={_v(k):.0f}<{ref:.0f}"
# # #             for k, ref in ST_TRANS_TARGETS.items()
# # #             if np.isfinite(_v(k)) and _v(k) < ref]
# # #     print(f"  {'─'*70}")
# # #     print(f"  BEAT ST-Trans: {' | '.join(beat) if beat else 'none yet'}")
# # #     print(f"  {'='*70}\n")
# # #     return result


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Checkpoint
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def _save(path, epoch, model, opt, sched, best_score,
# # #           ema=None, scaler=None, extra=None):
# # #     m   = _unwrap(model)
# # #     esd = None
# # #     if ema is not None:
# # #         try:
# # #             esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# # #         except Exception:
# # #             pass
# # #     payload = {
# # #         "epoch": epoch, "model": m.state_dict(),
# # #         "optimizer": opt.state_dict(),
# # #         "scheduler": sched.epoch,
# # #         "scaler": scaler.state_dict() if scaler is not None else None,
# # #         "best_score": best_score,
# # #         "best_ade":   best_score,   # compat
# # #         "ema": esd,
# # #     }
# # #     if extra:
# # #         payload.update(extra)
# # #     torch.save(payload, path)


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Args
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def get_args():
# # #     p = argparse.ArgumentParser(
# # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # #     p.add_argument("--dataset_root",           default="TCND_vn")
# # #     p.add_argument("--obs_len",                default=8,      type=int)
# # #     p.add_argument("--pred_len",               default=12,     type=int)
# # #     p.add_argument("--num_workers",            default=2,      type=int)
# # #     p.add_argument("--other_modal",            default="gph")
# # #     p.add_argument("--delim",                  default=" ")
# # #     p.add_argument("--skip",                   default=1,      type=int)
# # #     p.add_argument("--min_ped",                default=1,      type=int)
# # #     p.add_argument("--threshold",              default=0.002,  type=float)
# # #     # Model — giữ v2 defaults (256d, 4 layers)
# # #     p.add_argument("--d_cond",                 default=256,    type=int)
# # #     p.add_argument("--d_model",                default=256,    type=int)
# # #     p.add_argument("--nhead",                  default=8,      type=int)
# # #     p.add_argument("--num_dec_layers",         default=4,      type=int)
# # #     p.add_argument("--dim_ff",                 default=512,    type=int)
# # #     p.add_argument("--dropout",                default=0.1,    type=float)
# # #     p.add_argument("--unet_in_ch",             default=13,     type=int)
# # #     p.add_argument("--sigma_min",              default=0.04,   type=float)
# # #     p.add_argument("--sigma_max",              default=0.08,   type=float)
# # #     p.add_argument("--lambda_reg",             default=0.2,    type=float)
# # #     p.add_argument("--use_ot",                 default=True,   action="store_true")
# # #     p.add_argument("--ot_epsilon",             default=0.05,   type=float)
# # #     p.add_argument("--no_ot",                  dest="use_ot",  action="store_false")
# # #     p.add_argument("--n_ensemble",             default=20,     type=int)
# # #     p.add_argument("--sigma_inference",        default=0.04,   type=float,
# # #                    help="[FIX-2] match sigma_min để tránh mismatch ở epoch 40+")
# # #     p.add_argument("--n_inference_steps",      default=1,      type=int,
# # #                    help="[FIX-2] 1=1-shot, >1=Euler")
# # #     # Training
# # #     p.add_argument("--num_epochs",             default=150,    type=int)
# # #     p.add_argument("--batch_size",             default=64,     type=int)
# # #     p.add_argument("--lr",                     default=2e-4,   type=float)
# # #     p.add_argument("--lr_min",                 default=1e-6,   type=float)
# # #     p.add_argument("--warmup_epochs",          default=5,      type=int)
# # #     p.add_argument("--weight_decay",           default=1e-4,   type=float)
# # #     p.add_argument("--grad_clip",              default=1.0,    type=float)
# # #     p.add_argument("--use_amp",                action="store_true", default=False)
# # #     p.add_argument("--use_ema",                default=True,   action="store_true")
# # #     p.add_argument("--no_ema",                 dest="use_ema", action="store_false")
# # #     p.add_argument("--ema_decay",              default=0.995,  type=float)
# # #     # Encoder freeze (từ v4)
# # #     p.add_argument("--freeze_encoder_epochs",  default=10,     type=int)
# # #     p.add_argument("--encoder_warmup_epochs",  default=5,      type=int)
# # #     p.add_argument("--lr_enc_peak",            default=5e-5,   type=float)
# # #     # Eval
# # #     p.add_argument("--val_freq",               default=5,      type=int)
# # #     p.add_argument("--patience",               default=30,     type=int)
# # #     p.add_argument("--min_ep",                 default=20,     type=int)
# # #     # IO
# # #     p.add_argument("--output_dir",             default="runs/fm_v2")
# # #     p.add_argument("--gpu_num",                default="0")
# # #     p.add_argument("--resume",                 default=None)
# # #     p.add_argument("--test_at_end",            action="store_true", default=True)
# # #     return p.parse_args()


# # # # ─────────────────────────────────────────────────────────────────────────────
# # # #  Main
# # # # ─────────────────────────────────────────────────────────────────────────────

# # # def main(args):
# # #     if torch.cuda.is_available():
# # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     os.makedirs(args.output_dir, exist_ok=True)
# # #     best_ckpt = os.path.join(args.output_dir, "best_model.pth")
# # #     last_ckpt = os.path.join(args.output_dir, "last_model.pth")

# # #     print("=" * 72)
# # #     print("  TC-FlowMatching v2.1")
# # #     print(f"  [FIX-1] augment_batch() ở train loop, val KHÔNG augment")
# # #     print(f"  [FIX-2] 1-shot inference (n_steps={args.n_inference_steps},"
# # #           f" sigma_inf={args.sigma_inference})")
# # #     print(f"  [FIX-3] Augment mạnh: track_shift±5km + speed_scale×0.85-1.15, 100%")
# # #     print(f"  GIỮ: d_model={args.d_model}, layers={args.num_dec_layers},"
# # #           f" L_CFM+L_reg, OT")
# # #     print(f"  Encoder freeze {args.freeze_encoder_epochs} ep"
# # #           f" → warmup {args.encoder_warmup_epochs} ep"
# # #           f" → lr_peak={args.lr_enc_peak:.0e}")
# # #     print("=" * 72)

# # #     # ── Data ──────────────────────────────────────────────────────────────
# # #     print("\n  Loading data...")
# # #     trd, trl = data_loader(
# # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # #     vd, val_loader = data_loader(
# # #         args, {"root": args.dataset_root, "type": "val"}, test=True)
# # #     print(f"  train : {len(trd)} ({len(trl)} batches/ep)")
# # #     print(f"  val   : {len(vd)} ({len(val_loader)} batches) ← FULL VAL")

# # #     # ── Model ─────────────────────────────────────────────────────────────
# # #     model = TCFlowMatching(
# # #         pred_len=args.pred_len,        obs_len=args.obs_len,
# # #         unet_in_ch=args.unet_in_ch,   d_cond=args.d_cond,
# # #         d_model=args.d_model,          nhead=args.nhead,
# # #         num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
# # #         dropout=args.dropout,
# # #         sigma_min=args.sigma_min,      sigma_max=args.sigma_max,
# # #         lambda_reg=args.lambda_reg,    use_ot=args.use_ot,
# # #         ot_epsilon=args.ot_epsilon,    use_ema=args.use_ema,
# # #         n_ensemble=args.n_ensemble,
# # #         n_inference_steps=args.n_inference_steps,
# # #         sigma_inference=args.sigma_inference,
# # #     ).to(device)

# # #     model.init_ema()
# # #     ema = getattr(_unwrap(model), "_ema", None)

# # #     raw     = _unwrap(model)
# # #     n_enc   = sum(p.numel() for p in raw.encoder.parameters())
# # #     n_vel   = sum(p.numel() for p in raw.velocity.parameters())
# # #     print(f"\n  Encoder       : {n_enc:,} params")
# # #     print(f"  VelocityTrans : {n_vel:,} params")
# # #     print(f"  Total         : {n_enc+n_vel:,} params")

# # #     # ── 2-group optimizer ─────────────────────────────────────────────────
# # #     lr_enc_init = 0.0 if args.freeze_encoder_epochs > 0 else args.lr_enc_peak
# # #     opt = build_optimizer(model, lr_velocity=args.lr,
# # #                           lr_encoder=lr_enc_init,
# # #                           weight_decay=args.weight_decay)
# # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # #     sched  = TwoGroupScheduler(
# # #         opt=opt,
# # #         warmup_epochs=args.warmup_epochs,
# # #         total_epochs=args.num_epochs,
# # #         lr_vel=args.lr,
# # #         lr_vel_min=args.lr_min,
# # #         freeze_end_ep=args.freeze_encoder_epochs,
# # #         lr_enc_peak=args.lr_enc_peak,
# # #         encoder_warmup_epochs=args.encoder_warmup_epochs,
# # #     )

# # #     print(f"\n  LR velocity   : {args.lr:.0e} → {args.lr_min:.0e}")
# # #     print(f"  LR encoder    : 0 ({args.freeze_encoder_epochs} ep)"
# # #           f" → {args.lr_enc_peak:.0e} ({args.encoder_warmup_epochs} ep)"
# # #           f" → cosine")

# # #     # ── Resume ────────────────────────────────────────────────────────────
# # #     start_ep     = 0
# # #     best_score   = float("inf")
# # #     patience_cnt = 0
# # #     val_loss_history = []

# # #     if args.resume and os.path.exists(args.resume):
# # #         ck = torch.load(args.resume, map_location=device)
# # #         _unwrap(model).load_state_dict(ck["model"], strict=False)
# # #         try:
# # #             opt.load_state_dict(ck["optimizer"])
# # #         except Exception as e:
# # #             print(f"  ⚠ Opt state: {e}")
# # #         sched.epoch  = ck.get("scheduler", 0)
# # #         start_ep     = ck.get("epoch", 0) + 1
# # #         best_score   = ck.get("best_score", ck.get("best_ade", float("inf")))
# # #         patience_cnt = ck.get("patience_cnt", 0)
# # #         if scaler is not None and ck.get("scaler"):
# # #             try:
# # #                 scaler.load_state_dict(ck["scaler"])
# # #             except Exception:
# # #                 pass
# # #         if ema and ck.get("ema"):
# # #             for k, v in ck["ema"].items():
# # #                 if k in ema.shadow:
# # #                     ema.shadow[k].copy_(v.to(device))
# # #         print(f"  ↩ Resumed ep{start_ep}  best_score={best_score:.1f}"
# # #               f"  patience={patience_cnt}/{args.patience}")

# # #     try:
# # #         model = torch.compile(model, mode="reduce-overhead")
# # #         print("  torch.compile: ok")
# # #     except Exception:
# # #         pass

# # #     print()
# # #     print("=" * 72)
# # #     print(f"  TRAINING  ({len(trl)} steps/ep × {args.num_epochs} ep)")
# # #     print(f"  Train: augment=True  Val: augment=False  ← FIX-1")
# # #     print(f"  Inference: 1-shot at t=0  ← FIX-2")
# # #     print("=" * 72)

# # #     nstep = len(trl)

# # #     for ep in range(start_ep, args.num_epochs):
# # #         model.train()
# # #         sum_loss_w = 0.0; sum_n_train = 0
# # #         sum_cfm = sum_reg = sum_ade1 = 0.0
# # #         t0 = time.perf_counter()

# # #         for i, batch in enumerate(trl):
# # #             bl = move(list(batch), device)
# # #             B  = bl[0].shape[1]

# # #             # [FIX-1] Augment CHỈ ở train loop
# # #             bl_aug = augment_batch(bl)

# # #             opt.zero_grad()
# # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # #                 bd = model.get_loss_breakdown(bl_aug, epoch=ep)

# # #             scaler.scale(bd["total"]).backward()
# # #             scaler.unscale_(opt)
# # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # #             scaler.step(opt)
# # #             scaler.update()
# # #             model.ema_update()

# # #             sum_loss_w  += bd["total"].item() * B
# # #             sum_n_train += B
# # #             sum_cfm     += bd["l_cfm"]
# # #             sum_reg     += bd["l_reg"]
# # #             sum_ade1    += bd["ade_1step"]

# # #             if i % 30 == 0:
# # #                 lr_enc, lr_vel = get_lrs(opt)
# # #                 print(f"  [{ep:>3}][{i:>3}/{nstep}]"
# # #                       f"  total={bd['total'].item():.4f}"
# # #                       f"  cfm={bd['l_cfm']:.4f}"
# # #                       f"  reg={bd['l_reg']:.4f}"
# # #                       f"  lam={bd['lam_reg']:.3f}"
# # #                       f"  ade1={bd['ade_1step']:.0f}km"
# # #                       f"  σ={bd['sigma']:.3f}"
# # #                       f"  lr_vel={lr_vel:.2e}"
# # #                       f"  lr_enc={lr_enc:.2e}")

# # #         train_loss = sum_loss_w / max(sum_n_train, 1)
# # #         avg_cfm    = sum_cfm / nstep
# # #         avg_reg    = sum_reg / nstep
# # #         avg_ade1   = sum_ade1 / nstep
# # #         # Lưu lr đang dùng TRƯỚC khi advance scheduler
# # #         lr_enc_used, lr_vel_used = get_lrs(opt)
# # #         lr_vel, lr_enc = sched.step()   # advance cho epoch tiếp theo

# # #         # Encoder unfreeze log
# # #         if ep == args.freeze_encoder_epochs:
# # #             print(f"\n  *** Ep{ep}: encoder warmup 0→{args.lr_enc_peak:.0e}"
# # #                   f" ({args.encoder_warmup_epochs} ep) ***")
# # #         if ep == args.freeze_encoder_epochs + args.encoder_warmup_epochs:
# # #             print(f"\n  *** Ep{ep}: encoder fully active ***")

# # #         print(f"\n  ── Ep{ep:>3}"
# # #               f"  train_loss={train_loss:.6f}"
# # #               f"  cfm={avg_cfm:.4f}"
# # #               f"  reg={avg_reg:.4f}"
# # #               f"  ade1={avg_ade1:.0f}km"
# # #               f"  lr_vel={lr_vel_used:.2e}"   # lr dùng epoch này
# # #               f"  lr_enc={lr_enc_used:.2e}"   # lr dùng epoch này
# # #               f"  → next lr_vel={lr_vel:.2e}"  # lr epoch tiếp theo
# # #               f"  t={time.perf_counter()-t0:.0f}s")

# # #         _save(last_ckpt, ep, model, opt, sched, best_score, ema, scaler)

# # #         # ── Val ───────────────────────────────────────────────────────────
# # #         if ep % args.val_freq == 0:
# # #             print(f"\n  ── Val FULL SET (ep{ep}) ──")
# # #             r = evaluate(model, val_loader, device,
# # #                          tag=f"VAL ep{ep}",
# # #                          n_ensemble=args.n_ensemble,
# # #                          ema=ema,
# # #                          epoch_for_loss=ep)

# # #             val_loss = r["val_loss"]
# # #             val_ade  = r["ADE"]
# # #             val_loss_history.append(val_ade)   # track ADE trend (reliable hơn loss gap)
# # #             score    = r["combined_score"]

# # #             # [FIX-7] Overfit detection bằng val ADE trend
# # #             # KHÔNG dùng val_loss vs train_loss gap vì:
# # #             #   train_loss có augment → distribution khác → gap không reliable
# # #             # Thay bằng: val ADE tăng liên tục = overfit signal
# # #             if len(val_loss_history) >= 4:
# # #                 recent = float(np.mean(val_loss_history[-2:]))
# # #                 before = float(np.mean(val_loss_history[-4:-2]))
# # #                 trend  = recent - before
# # #                 if   trend >  5.0:  trend_str = f"ADE↑{trend:+.1f}km ⚠TĂNG"
# # #                 elif trend < -5.0:  trend_str = f"ADE↓{trend:+.1f}km ✓giảm"
# # #                 else:               trend_str = f"ADE→{trend:+.1f}km flat"
# # #             else:
# # #                 trend_str = "—"

# # #             print(f"  train_loss={train_loss:.6f}"
# # #                   f"  val_loss={val_loss:.6f}"
# # #                   f"  val_ADE={val_ade:.1f}km"
# # #                   f"  trend={trend_str}")
# # #             print(f"  combined={score:.1f}  best={best_score:.1f}")

# # #             if score < best_score:
# # #                 # Save best từ epoch 0 — không cần min_ep
# # #                 best_score   = score
# # #                 patience_cnt = 0
# # #                 _save(best_ckpt, ep, model, opt, sched, best_score, ema, scaler,
# # #                       extra={"val_ade":   r["ADE"],
# # #                              "val_ate":   r["ATE"],
# # #                              "val_cte":   r["CTE"],
# # #                              "val_loss":  val_loss,
# # #                              "val_score": score,
# # #                              "patience_cnt": patience_cnt})
# # #                 print(f"  ✅ New best! score={best_score:.2f}"
# # #                       f"  ADE={r['ADE']:.1f} ATE={r['ATE']:.1f}"
# # #                       f"  CTE={r['CTE']:.1f} → {best_ckpt}")
# # #             else:
# # #                 # Patience chỉ đếm sau min_ep — tránh early stop quá sớm
# # #                 if ep >= args.min_ep:
# # #                     patience_cnt += args.val_freq
# # #                 print(f"  No improve {patience_cnt}/{args.patience}")
# # #                 if ep >= args.min_ep and patience_cnt >= args.patience:
# # #                     print(f"  ⛔ Early stop @ ep{ep}")
# # #                     break

# # #         # Checkpoint mỗi 5 epoch
# # #         if ep % 5 == 0:
# # #             ckpt_ep = os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth")
# # #             _save(ckpt_ep, ep, model, opt, sched, best_score, ema, scaler)
# # #             print(f"  💾 {ckpt_ep}")

# # #     print("\n" + "=" * 72)
# # #     print(f"  Done! best_score={best_score:.2f}")
# # #     print("=" * 72)

# # #     # ── Test ──────────────────────────────────────────────────────────────
# # #     if args.test_at_end and os.path.exists(best_ckpt):
# # #         print("\n  Loading best ckpt for TEST...")
# # #         ck = torch.load(best_ckpt, map_location=device)
# # #         _unwrap(model).load_state_dict(ck["model"], strict=False)
# # #         if ema and ck.get("ema"):
# # #             for k, v in ck["ema"].items():
# # #                 if k in ema.shadow:
# # #                     ema.shadow[k].copy_(v.to(device))

# # #         try:
# # #             _, test_loader = data_loader(
# # #                 args, {"root": args.dataset_root, "type": "test"}, test=True)
# # #             print(f"  Test: {len(test_loader)} batches")
# # #         except Exception:
# # #             print("  No test set → using val")
# # #             test_loader = val_loader

# # #         r_test = evaluate(model, test_loader, device,
# # #                           tag="TEST (best ckpt, NO augment)",
# # #                           n_ensemble=args.n_ensemble, ema=ema)

# # #         val_ade  = ck.get("val_ade",  float("nan"))
# # #         val_ate  = ck.get("val_ate",  float("nan"))
# # #         val_cte  = ck.get("val_cte",  float("nan"))
# # #         val_loss_saved = ck.get("val_loss", float("nan"))

# # #         print()
# # #         print("=" * 72)
# # #         print("  VAL vs TEST (1-shot, kỳ vọng gap < 20km)")
# # #         print("=" * 72)
# # #         print(f"  {'Metric':<12}  {'Val':>10}  {'Test':>10}  {'Gap':>10}  Status")
# # #         print("  " + "─" * 58)
# # #         for m, v, t in [("ADE (km)",  val_ade,        r_test["ADE"]),
# # #                         ("ATE (km)",  val_ate,        r_test["ATE"]),
# # #                         ("CTE (km)",  val_cte,        r_test["CTE"]),
# # #                         ("Loss",      val_loss_saved, r_test["val_loss"])]:
# # #             gap  = t - v if (np.isfinite(t) and np.isfinite(v)) else float("nan")
# # #             flag = ("✅" if np.isfinite(gap) and abs(gap) < 20
# # #                     else "⚠" if np.isfinite(gap) else "?")
# # #             print(f"  {m:<12}  {v:>10.2f}  {t:>10.2f}  {gap:>+10.2f}  {flag}")
# # #         print("=" * 72)


# # # if __name__ == "__main__":
# # #     args = get_args()
# # #     np.random.seed(42); torch.manual_seed(42)
# # #     if torch.cuda.is_available():
# # #         torch.cuda.manual_seed_all(42)

# # #     if args.dataset_root == "TCND_vn":
# # #         _auto = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
# # #         if os.path.isdir(_auto):
# # #             args.dataset_root = _auto
# # #             print(f"  Auto dataset: {_auto}")

# # #     main(args)

# # """
# # scripts/train_fm.py  ——  TC-FlowMatching v2 Training
# # ═══════════════════════════════════════════════════════

# # Tương ứng với flow_matching_model.py (FM đúng cách).

# # THIẾT KẾ TRAINING ĐƠN GIẢN:
# #   - 1 optimizer, 1 scheduler (Cosine + warmup)
# #   - Không GradNorm (chỉ có 2 loss terms)
# #   - Không phase transitions phức tạp
# #   - Không alpha_hard, không selector training
# #   - EMA + early stopping trên val ADE

# # HYPERPARAMETERS MẶC ĐỊNH:
# #   lr = 2e-4   (nhỏ hơn bản cũ vì FM transformer deeper)
# #   batch_size = 64
# #   num_epochs = 150
# #   warmup = 5 epoch
# #   sigma_max = 0.3, sigma_min = 0.1 (schedule tự động)
# #   lambda_reg = 0.2 (ramp từ epoch 10-30)
# #   n_ensemble = 20, ddim_steps = 8

# # CHẠY:
# #   python scripts/train_fm.py \
# #     --dataset_root /kaggle/input/.../tc-ofm \
# #     --output_dir   runs/fm_v2 \
# #     --num_epochs   150 \
# #     --batch_size   64

# # RESUME:
# #   python scripts/train_fm.py \
# #     --resume runs/fm_v2/best_model.pth \
# #     --dataset_root ...
# # """
# # from __future__ import annotations
# # import sys, os
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # import argparse, math, random, time
# # from collections import defaultdict
# # from typing import Dict, Optional

# # import numpy as np
# # import torch
# # import torch.optim as optim
# # from torch.amp import autocast, GradScaler
# # from torch.utils.data import DataLoader, Subset

# # from Model.data.loader_training import data_loader
# # from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # from Model.flow_matching_model import (
# #     TCFlowMatching, _norm_to_deg, _haversine_deg, EMAModel,
# # )

# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Constants
# # # ─────────────────────────────────────────────────────────────────────────────
# # R_EARTH = 6371.0
# # HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}

# # ST_TRANS_TARGETS = {
# #     "ADE": 172.68, "72h": 321.39, "ATE": 142.21, "CTE": 42.04,
# #     "12h": 65.42,  "24h": 104.67, "48h": 205.10,
# # }


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Utilities
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _unwrap(m):
# #     return m._orig_mod if hasattr(m, "_orig_mod") else m

# # def move(batch, device):
# #     out = list(batch)
# #     for i, x in enumerate(out):
# #         if torch.is_tensor(x):
# #             out[i] = x.to(device)
# #         elif isinstance(x, dict):
# #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# #                       for k, v in x.items()}
# #     return out

# # def make_subset_loader(dataset, n, batch_size, collate_fn):
# #     idx = random.Random(42).sample(range(len(dataset)), min(n, len(dataset)))
# #     return DataLoader(Subset(dataset, idx), batch_size=batch_size,
# #                       shuffle=False, collate_fn=collate_fn,
# #                       num_workers=0, drop_last=False)

# # def _ate_cte(pred_deg, gt_deg):
# #     """ATE, CTE per step"""
# #     T = min(pred_deg.shape[0], gt_deg.shape[0])
# #     if T < 2:
# #         z = pred_deg.new_zeros(1, pred_deg.shape[1])
# #         return z, z
# #     lo1 = torch.deg2rad(gt_deg[:T-1, :, 0])
# #     la1 = torch.deg2rad(gt_deg[:T-1, :, 1])
# #     lo2 = torch.deg2rad(gt_deg[1:T,  :, 0])
# #     la2 = torch.deg2rad(gt_deg[1:T,  :, 1])
# #     lo3 = torch.deg2rad(pred_deg[1:T, :, 0])
# #     la3 = torch.deg2rad(pred_deg[1:T, :, 1])
# #     ya  = torch.sin(lo2 - lo1) * torch.cos(la2)
# #     xa  = (torch.cos(la1) * torch.sin(la2)
# #            - torch.sin(la1) * torch.cos(la2) * torch.cos(lo2 - lo1))
# #     be  = torch.atan2(ya, xa)
# #     ye  = torch.sin(lo3 - lo2) * torch.cos(la3)
# #     xe  = (torch.cos(la2) * torch.sin(la3)
# #            - torch.sin(la2) * torch.cos(la3) * torch.cos(lo3 - lo2))
# #     bee = torch.atan2(ye, xe)
# #     tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
# #     ang = bee - be
# #     return tot * torch.cos(ang), tot * torch.sin(ang)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Scheduler: linear warmup + cosine annealing
# # # ─────────────────────────────────────────────────────────────────────────────

# # class WarmupCosineScheduler:
# #     def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
# #                  lr_init: float, lr_min: float = 1e-6):
# #         self.opt     = optimizer
# #         self.warmup  = warmup_epochs
# #         self.total   = total_epochs
# #         self.lr_init = lr_init
# #         self.lr_min  = lr_min
# #         self.epoch   = 0

# #     def step(self) -> float:
# #         e = self.epoch
# #         if e < self.warmup:
# #             lr = self.lr_init * (0.1 + 0.9 * e / max(self.warmup - 1, 1))
# #         else:
# #             t  = e - self.warmup
# #             T  = self.total - self.warmup
# #             lr = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (
# #                 1 + math.cos(math.pi * t / max(T, 1)))
# #         for pg in self.opt.param_groups:
# #             pg["lr"] = lr
# #         self.epoch += 1
# #         return lr


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Evaluation
# # # ─────────────────────────────────────────────────────────────────────────────

# # @torch.no_grad()
# # def evaluate(model, loader, device, tag: str = "",
# #              ddim_steps: int = 8, ema: Optional[EMAModel] = None) -> Dict:
# #     bk = None
# #     if ema is not None:
# #         try: bk = ema.apply_to(model)
# #         except: pass

# #     model.eval()
# #     all_ade, all_ate, all_cte = [], [], []
# #     step_ade = defaultdict(list)

# #     for batch in loader:
# #         bl   = move(list(batch), device)
# #         gt   = bl[1]
# #         pred, _, _ = model.sample(bl, ddim_steps=ddim_steps)

# #         T        = min(pred.shape[0], gt.shape[0])
# #         pred_deg = _norm_to_deg(pred[:T])
# #         gt_deg   = _norm_to_deg(gt[:T])
# #         dist     = _haversine_deg(pred_deg, gt_deg)  # [T, B]
# #         ate, cte = _ate_cte(pred_deg, gt_deg)

# #         all_ade.extend(dist.mean(0).tolist())
# #         if ate.shape[0] > 0:
# #             all_ate.extend(ate.abs().mean(0).tolist())
# #             all_cte.extend(cte.abs().mean(0).tolist())

# #         for h, s in HORIZON_STEPS.items():
# #             if s < T:
# #                 step_ade[h].extend(dist[s].tolist())

# #     def _m(lst): return float(np.mean(lst)) if lst else float("nan")

# #     result = {
# #         "ADE": _m(all_ade),
# #         "ATE": _m(all_ate),
# #         "CTE": _m(all_cte),
# #         "n":   len(all_ade),
# #     }
# #     for h in HORIZON_STEPS:
# #         result[f"{h}h"] = _m(step_ade[h])

# #     if bk is not None:
# #         try: ema.restore(model, bk)
# #         except: pass

# #     # Print
# #     def _v(k): return result.get(k, float("nan"))
# #     def _ok(k, ref): return "✓" if np.isfinite(_v(k)) and _v(k) < ref else "✗"

# #     print(f"\n  {'='*60}")
# #     print(f"  [{tag}]")
# #     print(f"  ADE={_v('ADE'):.1f} {_ok('ADE', ST_TRANS_TARGETS['ADE'])}"
# #           f"  (target < {ST_TRANS_TARGETS['ADE']})")
# #     print(f"  12h={_v('12h'):.0f}  24h={_v('24h'):.0f}"
# #           f"  48h={_v('48h'):.0f}  72h={_v('72h'):.0f}"
# #           f"  {_ok('72h', ST_TRANS_TARGETS['72h'])}")
# #     print(f"  ATE={_v('ATE'):.1f}  CTE={_v('CTE'):.1f}")

# #     beat = []
# #     for k, ref in ST_TRANS_TARGETS.items():
# #         v = _v(k)
# #         if np.isfinite(v) and v < ref:
# #             beat.append(f"{k}:{v:.1f}<{ref:.1f}")
# #     if beat:
# #         print(f"  *** BEAT ST-TRANS: {' | '.join(beat)} ***")
# #     print(f"  {'='*60}\n")

# #     return result


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Checkpoint
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _save(path, epoch, model, opt, sched, best_ade, ema=None,
# #           patience_cnt=0, scaler=None, args=None,
# #           val_loss_history=None, best_val_loss=float("inf")):
# #     m   = _unwrap(model)
# #     esd = None
# #     if ema is not None:
# #         try: esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
# #         except: pass
# #     torch.save({
# #         # ── Core weights ──────────────────────────────────────────────
# #         "epoch":        epoch,
# #         "model":        m.state_dict(),
# #         "ema":          esd,
# #         # ── Optimizer / scheduler / scaler ────────────────────────────
# #         # Lưu đủ để resume tiếp tục chính xác như chưa bị interrupt:
# #         #   optimizer : Adam momentum + variance -> không mất đà
# #         #   scheduler : sched.epoch -> WarmupCosine biết đang ở đâu
# #         #   scaler    : AMP loss scale đã calibrate -> không bị reset
# #         "optimizer":    opt.state_dict(),
# #         "scheduler":    sched.epoch,
# #         "sched_total":  sched.total,    # giữ cosine tail đúng khi resume với num_epochs khác
# #         "scaler":       scaler.state_dict() if scaler is not None else None,
# #         # ── Training state ────────────────────────────────────────────
# #         "best_ade":     best_ade,
# #         "patience_cnt":    patience_cnt,
# #         "val_loss_history": val_loss_history,
# #         "best_val_loss":    best_val_loss,
# #         # ── Hyperparams snapshot ──────────────────────────────────────
# #         "args":         vars(args) if args is not None else None,
# #     }, path)


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Args
# # # ─────────────────────────────────────────────────────────────────────────────

# # def get_args():
# #     p = argparse.ArgumentParser(
# #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# #     # Data
# #     p.add_argument("--dataset_root",  default="TCND_vn")
# #     p.add_argument("--obs_len",       default=8,     type=int)
# #     p.add_argument("--pred_len",      default=12,    type=int)
# #     p.add_argument("--num_workers",   default=2,     type=int)
# #     p.add_argument("--other_modal",   default="gph")
# #     p.add_argument("--delim",         default=" ")
# #     p.add_argument("--skip",          default=1,     type=int)
# #     p.add_argument("--min_ped",       default=1,     type=int)
# #     p.add_argument("--threshold",     default=0.002, type=float)
# #     # Model
# #     p.add_argument("--d_cond",        default=256,   type=int)
# #     p.add_argument("--d_model",       default=256,   type=int)
# #     p.add_argument("--nhead",         default=8,     type=int)
# #     p.add_argument("--num_dec_layers",default=4,     type=int)
# #     p.add_argument("--dim_ff",        default=512,   type=int)
# #     p.add_argument("--dropout",       default=0.1,   type=float)
# #     p.add_argument("--unet_in_ch",    default=13,    type=int)
# #     p.add_argument("--sigma_min",     default=0.04,  type=float,
# #                    help="FM sigma min (~22km, 15%% ADE). Phải < ADE_norm=0.32")
# #     p.add_argument("--sigma_max",     default=0.08,  type=float,
# #                    help="FM sigma max (~43km, 25%% ADE). Phải << ADE_norm")
# #     p.add_argument("--lambda_reg",    default=0.2,   type=float,
# #                    help="ADE signal weight (FIX-FATAL-1)")
# #     p.add_argument("--use_ot",        default=True,  action="store_true")
# #     p.add_argument("--ot_epsilon",    default=0.05,  type=float,
# #                    help="Sinkhorn epsilon cho OT matching")
# #     p.add_argument("--no_ot",         dest="use_ot", action="store_false")
# #     p.add_argument("--n_ensemble",    default=20,    type=int)
# #     p.add_argument("--sigma_inference", default=0.04, type=float,
# #                    help="Inference noise std. Dat = sigma_min (0.04) de tranh "
# #                         "mismatch: epoch 40+ model chi thay sigma=0.04, "
# #                         "dung 0.079 se extrapolate ngoai training range.")
# #     # Training
# #     p.add_argument("--num_epochs",    default=150,   type=int)
# #     p.add_argument("--batch_size",    default=64,    type=int)
# #     p.add_argument("--lr",            default=2e-4,  type=float)
# #     p.add_argument("--lr_min",        default=1e-6,  type=float)
# #     p.add_argument("--warmup_epochs", default=5,     type=int)
# #     p.add_argument("--weight_decay",  default=1e-4,  type=float)
# #     p.add_argument("--grad_clip",     default=1.0,   type=float)
# #     p.add_argument("--use_amp",       action="store_true", default=False)
# #     p.add_argument("--use_ema",       default=True,  action="store_true")
# #     p.add_argument("--no_ema",        dest="use_ema",action="store_false")
# #     p.add_argument("--ema_decay",     default=0.995, type=float)
# #     # Eval
# #     p.add_argument("--val_freq",      default=5,     type=int)
# #     p.add_argument("--val_subset",    default=1000,  type=int,
# #                    help="Số samples val mỗi lần eval (default 1000 ≈ 30% val set)")
# #     p.add_argument("--ddim_steps",    default=8,     type=int)
# #     p.add_argument("--patience",      default=30,    type=int)
# #     p.add_argument("--min_ep",        default=20,    type=int)
# #     # IO
# #     p.add_argument("--output_dir",    default="runs/fm_v2")
# #     p.add_argument("--gpu_num",       default="0")
# #     p.add_argument("--resume",        default=None)
# #     p.add_argument("--test_at_end",   action="store_true", default=True)
# #     return p.parse_args()


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Main
# # # ─────────────────────────────────────────────────────────────────────────────

# # def main(args):
# #     if torch.cuda.is_available():
# #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     os.makedirs(args.output_dir, exist_ok=True)
# #     best_ckpt = os.path.join(args.output_dir, "best_model.pth")

# #     print("=" * 65)
# #     print("  TC-FlowMatching v2  |  FM đúng cách")
# #     print(f"  sigma: {args.sigma_max}→{args.sigma_min}  "
# #           f"lambda_reg={args.lambda_reg}  K={args.n_ensemble}")
# #     print(f"  N_steps={args.ddim_steps}  sigma_inf={args.sigma_inference}")
# #     print(f"  FIX-FATAL-1: L_reg tại t=0 (gradient không dilute)")
# #     print(f"  FIX-FATAL-2: 2 loss terms (l_cfm + lambda_reg*l_reg)")
# #     print(f"  FIX-FATAL-3: Best-of-K thay SelectorNet")
# #     print(f"  FIX-MAJOR-4: sigma_max={args.sigma_max}, "
# #           f"sigma_inf={args.sigma_inference}")
# #     print(f"  FIX-MAJOR-5: FM trên 2D (không 4D)")
# #     print("=" * 65)

# #     # ── Data ──────────────────────────────────────────────────────────────
# #     trd, trl = data_loader(
# #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# #     vd, _    = data_loader(
# #         args, {"root": args.dataset_root, "type": "val"},   test=True)
# #     val_sub  = make_subset_loader(vd, args.val_subset, args.batch_size, seq_collate)
# #     # Full val loader — dùng để ADE evaluation chính xác (không bị bias subset)
# #     val_full = DataLoader(vd, batch_size=args.batch_size, shuffle=False,
# #                           collate_fn=seq_collate, num_workers=0, drop_last=False)
# #     print(f"  train: {len(trd)}  val: {len(vd)}  (full_val batches={len(val_full)})")

# #     # ── Model ─────────────────────────────────────────────────────────────
# #     model = TCFlowMatching(
# #         pred_len=args.pred_len, obs_len=args.obs_len,
# #         unet_in_ch=args.unet_in_ch,
# #         d_cond=args.d_cond, d_model=args.d_model,
# #         nhead=args.nhead, num_dec_layers=args.num_dec_layers,
# #         dim_ff=args.dim_ff, dropout=args.dropout,
# #         sigma_min=args.sigma_min, sigma_max=args.sigma_max,
# #         lambda_reg=args.lambda_reg,
# #         use_ot=args.use_ot,
# #         ot_epsilon=args.ot_epsilon,
# #         use_ema=args.use_ema,
# #         n_ensemble=args.n_ensemble,
# #         sigma_inference=args.sigma_inference,
# #     ).to(device)

# #     model.init_ema()
# #     ema = getattr(_unwrap(model), "_ema", None)

# #     n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #     n_enc   = sum(p.numel() for p in model.encoder.parameters())
# #     n_vel   = sum(p.numel() for p in model.velocity.parameters())
# #     print(f"\n  Params total  : {n_total:,}")
# #     print(f"  Encoder       : {n_enc:,}  (FNO3D+Mamba, giữ nguyên)")
# #     print(f"  VelocityTrans : {n_vel:,}  (4-layer transformer, 2D only)")

# #     # ── Optimizer & scheduler ─────────────────────────────────────────────
# #     opt    = optim.AdamW(model.parameters(), lr=args.lr,
# #                           weight_decay=args.weight_decay)
# #     scaler = GradScaler("cuda", enabled=args.use_amp)
# #     sched  = WarmupCosineScheduler(
# #         opt, warmup_epochs=args.warmup_epochs,
# #         total_epochs=args.num_epochs,
# #         lr_init=args.lr, lr_min=args.lr_min)

# #     # ── Resume ────────────────────────────────────────────────────────────
# #     start_ep = 0
# #     best_ade = float("inf")
# #     patience_cnt = 0
# #     val_loss_history: list = []    # track trend để phát hiện overfit
# #     best_val_loss = float("inf")   # track best val_loss riêng

# #     if args.resume and os.path.exists(args.resume):
# #         ck = torch.load(args.resume, map_location=device)
# #         _unwrap(model).load_state_dict(ck["model"], strict=False)
# #         try: opt.load_state_dict(ck["optimizer"])
# #         except: pass
# #         sched.epoch  = ck.get("scheduler", 0)
# #         if "sched_total" in ck:
# #             sched.total = ck["sched_total"]   # giữ đúng cosine curve khi resume
# #         start_ep     = ck.get("epoch", 0) + 1
# #         best_ade     = ck.get("best_ade", float("inf"))
# #         patience_cnt = ck.get("patience_cnt", 0)   # ← restore patience
# #         if scaler is not None and ck.get("scaler") is not None:
# #             try: scaler.load_state_dict(ck["scaler"])   # ← restore AMP scale
# #             except: pass
# #         if ema and ck.get("ema"):
# #             for k, v in ck["ema"].items():
# #                 if k in ema.shadow:
# #                     ema.shadow[k].copy_(v.to(device))
# #         print(f"  ↩ Resumed ep{start_ep}  best_ADE={best_ade:.1f} km"
# #               f"  patience={patience_cnt}/{args.patience}")

# #     try:
# #         model = torch.compile(model, mode="reduce-overhead")
# #         print("  torch.compile: ok")
# #     except:
# #         pass

# #     print()
# #     print("=" * 65)
# #     print(f"  TRAINING  ({len(trl)} steps/epoch, {args.num_epochs} epochs)")
# #     print(f"  Warmup: {args.warmup_epochs} ep  lr: {args.lr}→{args.lr_min}")
# #     print(f"  L_reg ramp: ep10→ep30 (0→{args.lambda_reg})")
# #     print("=" * 65)

# #     nstep = len(trl)

# #     for ep in range(start_ep, args.num_epochs):
# #         model.train()
# #         sum_loss, sum_cfm, sum_reg, sum_ade1 = 0.0, 0.0, 0.0, 0.0
# #         t0 = time.perf_counter()

# #         for i, batch in enumerate(trl):
# #             bl = move(list(batch), device)

# #             opt.zero_grad()
# #             with autocast(device_type="cuda", enabled=args.use_amp):
# #                 bd = model.get_loss_breakdown(bl, epoch=ep)

# #             scaler.scale(bd["total"]).backward()
# #             scaler.unscale_(opt)
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# #             scaler.step(opt)
# #             scaler.update()
# #             model.ema_update()

# #             sum_loss += bd["total"].item()
# #             sum_cfm  += bd["l_cfm"]
# #             sum_reg  += bd["l_reg"]
# #             sum_ade1 += bd["ade_1step"]

# #             if i % 30 == 0:
# #                 lr = opt.param_groups[0]["lr"]
# #                 print(f"  [{ep:>3}][{i:>3}/{nstep}]"
# #                       f"  total={bd['total'].item():.4f}"
# #                       f"  cfm={bd['l_cfm']:.4f}"
# #                       f"  reg={bd['l_reg']:.4f}"
# #                       f"  lam={bd['lam_reg']:.3f}"
# #                       f"  ade_1step={bd['ade_1step']:.0f}km"
# #                       f"  sigma={bd['sigma']:.3f}"
# #                       f"  hs_mean={bd.get('hard_score_mean',0):.2f}"
# #                       f"  lr={lr:.2e}")

# #         avg_loss = sum_loss / nstep
# #         avg_cfm  = sum_cfm  / nstep
# #         avg_reg  = sum_reg  / nstep
# #         avg_ade1 = sum_ade1 / nstep
# #         # Lấy lr hiện tại (đang dùng trong epoch này) TRƯỚC khi sched.step() thay đổi
# #         cur_lr = opt.param_groups[0]["lr"]
# #         sched.step()   # advance lr cho epoch tiếp theo

# #         print(f"  Epoch {ep:>3}"
# #               f"  loss={avg_loss:.4f}"
# #               f"  cfm={avg_cfm:.4f}"
# #               f"  reg={avg_reg:.4f}"
# #               f"  ade_1step={avg_ade1:.0f}km"
# #               f"  lr={cur_lr:.2e}"      # lr epoch này (không phải epoch sau)
# #               f"  t={time.perf_counter()-t0:.0f}s")

# #         # ── Periodic ckpt mỗi 10 epoch ──────────────────────────────────────
# #         # Đặt TRƯỚC val block để không bị skip khi early stop break
# #         if ep % 5 == 0:
# #             _save(os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
# #                   ep, model, opt, sched, best_ade, ema,
# #                   patience_cnt=patience_cnt, scaler=scaler, args=args,
# #                       val_loss_history=val_loss_history, best_val_loss=best_val_loss)

# #         # ── Val ───────────────────────────────────────────────────────────
# #         if ep % args.val_freq == 0:
# #             # Val loss: dùng val_sub (nhanh) để theo dõi overfitting
# #             model.eval()
# #             val_loss_buf = []
# #             with torch.no_grad():
# #                 for vbatch in val_sub:
# #                     vbl = move(list(vbatch), device)
# #                     with autocast(device_type="cuda", enabled=args.use_amp):
# #                         vbd = model.get_loss_breakdown(vbl, epoch=ep)
# #                     if torch.isfinite(vbd["total"]):
# #                         val_loss_buf.append(vbd["total"].item())
# #             avg_val_loss = float(np.mean(val_loss_buf)) if val_loss_buf else float("nan")
# #             val_loss_history.append(avg_val_loss)
# #             if avg_val_loss < best_val_loss:
# #                 best_val_loss = avg_val_loss

# #             # Trend: trung bình 3 lần gần nhất vs 3 lần trước
# #             if len(val_loss_history) >= 6:
# #                 recent = float(np.mean(val_loss_history[-3:]))
# #                 before = float(np.mean(val_loss_history[-6:-3]))
# #                 trend  = recent - before
# #                 if   trend >  0.002: trend_str = f"↑ +{trend:.4f} ⚠TĂNG"
# #                 elif trend < -0.002: trend_str = f"↓ {trend:.4f} ✓giảm"
# #                 else:                trend_str = f"→ {trend:+.4f} flat"
# #             elif len(val_loss_history) >= 2:
# #                 d = val_loss_history[-1] - val_loss_history[-2]
# #                 trend_str = f"{'↑' if d > 0 else '↓'} {d:+.4f}"
# #             else:
# #                 trend_str = "—"

# #             gap     = avg_val_loss - avg_loss
# #             gap_pct = gap / max(avg_loss, 1e-8) * 100
# #             if   gap_pct > 50: warn = "  🔴 OVERFIT"
# #             elif gap_pct > 25: warn = "  🟡 watch"
# #             else:              warn = ""

# #             print(f"  val_loss={avg_val_loss:.4f}  best={best_val_loss:.4f}"
# #                   f"  trend={trend_str}"
# #                   f"  gap={gap:+.4f}({gap_pct:+.0f}%){warn}"
# #                   f"  train={avg_loss:.4f}")

# #             # Val ADE: full val set để early stop không bị bias subset
# #             r = evaluate(model, val_full, device,
# #                          tag=f"val ep{ep}  [full {len(vd)} samples]",
# #                          ddim_steps=args.ddim_steps,
# #                          ema=ema)
# #             ade = r["ADE"]

# #             if ade < best_ade:
# #                 best_ade     = ade
# #                 patience_cnt = 0
# #                 # Lưu ngay từ epoch 0, không chờ min_ep
# #                 _save(best_ckpt, ep, model, opt, sched, best_ade, ema,
# #                       patience_cnt=patience_cnt, scaler=scaler, args=args,
# #                       val_loss_history=val_loss_history, best_val_loss=best_val_loss)
# #                 print(f"  ✅ Best ADE = {best_ade:.2f} km  (ep {ep})")
# #             else:
# #                 # Patience chỉ đếm sau min_ep — tránh early stop quá sớm
# #                 if ep >= args.min_ep:
# #                     patience_cnt += args.val_freq
# #                 print(f"  No improve {patience_cnt}/{args.patience}"
# #                       f"  (best={best_ade:.1f})")
# #                 if ep >= args.min_ep and patience_cnt >= args.patience:
# #                     print(f"  ⛔ Early stop @ ep{ep}")
# #                     break

# #     print("=" * 65)
# #     print(f"  Best ADE: {best_ade:.2f} km")
# #     print(f"  Target:   {ST_TRANS_TARGETS['ADE']:.2f} km (ST-Trans)")
# #     print(f"  Gap:      {best_ade - ST_TRANS_TARGETS['ADE']:+.2f} km")
# #     print("=" * 65)

# #     # ── Test ──────────────────────────────────────────────────────────────
# #     if args.test_at_end and os.path.exists(best_ckpt):
# #         print("\n  Test set evaluation...")
# #         try:
# #             _, test_loader = data_loader(
# #                 args, {"root": args.dataset_root, "type": "test"}, test=True)
# #         except:
# #             print("  No test set → using val")
# #             _, test_loader = data_loader(
# #                 args, {"root": args.dataset_root, "type": "val"}, test=True)

# #         ck = torch.load(best_ckpt, map_location=device)
# #         _unwrap(model).load_state_dict(ck["model"], strict=False)
# #         if ema and ck.get("ema"):
# #             for k, v in ck["ema"].items():
# #                 if k in ema.shadow:
# #                     ema.shadow[k].copy_(v.to(device))

# #         evaluate(model, test_loader, device,
# #                  tag="TEST (EMA best)",
# #                  ddim_steps=args.ddim_steps,
# #                  ema=ema)


# # if __name__ == "__main__":
# #     args = get_args()
# #     np.random.seed(42)
# #     torch.manual_seed(42)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(42)
# #     main(args)

# """
# scripts/train_fm.py  ——  TC-FlowMatching v3 Training
# ═══════════════════════════════════════════════════════

# Thay đổi so với v2.1:

#   [TRAIN-1] Fine-tune từ checkpoint v2.1 (không train từ đầu)
#     Load weights encoder + velocity với strict=False
#     → Encoder đã tốt, chỉ cần học AR stages + learned modules mới
#     → Tiết kiệm ~30 epoch encoder warm-up

#   [TRAIN-2] 3-group optimizer
#     Group 1: encoder       lr = lr_enc (nhỏ, đã converge)
#     Group 2: velocity      lr = lr_vel (trung bình, cần adapt AR)
#     Group 3: score_net     lr = lr_vel * 2 (mới hoàn toàn, cần lr cao hơn)

#   [TRAIN-3] Val metric = combined_score = 0.5*ADE + 0.3*ATE + 0.2*CTE
#     Thay vì chỉ ADE → model không chỉ tối ưu ADE mà bỏ qua ATE

#   [TRAIN-4] Best checkpoint theo combined_score trên VAL FULL
#     Save mỗi 5 epoch, save best ngay từ epoch 0

#   [TRAIN-5] Early stopping patience = 30 epoch, min_ep = 15
#     Tăng min_ep từ 20 lên vì v3 cần thêm thời gian warm-up AR stages
# """
# from __future__ import annotations
# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse, math, time
# from collections import defaultdict
# from typing import Dict, Optional

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.amp import autocast, GradScaler

# from Model.data.loader_training import data_loader
# from Model.flow_matching_model import (
#     TCFlowMatching, _norm_to_deg, _haversine_deg,
#     EMAModel, augment_batch,
# )

# HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}
# ST_TRANS_TEST = {
#     "ADE": 224.4, "ATE": 213.7, "CTE": 59.4,
#     "12h": 77.5,  "24h": 130.5, "48h": 269.9, "72h": 423.3,
# }
# ST_TRANS_VAL = {
#     "ADE": 172.68, "ATE": 142.21, "CTE": 42.04,
#     "12h": 65.42,  "24h": 104.67, "48h": 205.10, "72h": 321.39,
# }


# def _unwrap(m):
#     return m._orig_mod if hasattr(m, "_orig_mod") else m


# def move(batch, device):
#     out = list(batch)
#     for i, x in enumerate(out):
#         if torch.is_tensor(x):
#             out[i] = x.to(device)
#         elif isinstance(x, dict):
#             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
#                       for k, v in x.items()}
#     return out


# def _ate_cte(pred_deg, gt_deg):
#     T = min(pred_deg.shape[0], gt_deg.shape[0])
#     if T < 2:
#         z = pred_deg.new_zeros(1, pred_deg.shape[1])
#         return z, z
#     lo1 = torch.deg2rad(gt_deg[:T-1,:,0]); la1 = torch.deg2rad(gt_deg[:T-1,:,1])
#     lo2 = torch.deg2rad(gt_deg[1:T, :,0]); la2 = torch.deg2rad(gt_deg[1:T, :,1])
#     lo3 = torch.deg2rad(pred_deg[1:T,:,0]); la3 = torch.deg2rad(pred_deg[1:T,:,1])
#     ya  = torch.sin(lo2-lo1)*torch.cos(la2)
#     xa  = torch.cos(la1)*torch.sin(la2) - torch.sin(la1)*torch.cos(la2)*torch.cos(lo2-lo1)
#     be  = torch.atan2(ya, xa)
#     ye  = torch.sin(lo3-lo2)*torch.cos(la3)
#     xe  = torch.cos(la2)*torch.sin(la3) - torch.sin(la2)*torch.cos(la3)*torch.cos(lo3-lo2)
#     bee = torch.atan2(ye, xe)
#     tot = _haversine_deg(pred_deg[1:T], gt_deg[1:T])
#     ang = bee - be
#     return tot*torch.cos(ang), tot*torch.sin(ang)


# # ─────────────────────────────────────────────────────────────────────────────
# #  3-group optimizer + scheduler
# # ─────────────────────────────────────────────────────────────────────────────

# def build_optimizer(model, lr_vel, lr_enc, lr_score, weight_decay):
#     raw = _unwrap(model)
#     enc_params   = list(raw.encoder.parameters())
#     vel_params   = list(raw.velocity.parameters())
#     score_params = list(raw.score_net.parameters())
#     for p in enc_params + vel_params + score_params:
#         p.requires_grad_(True)
#     return optim.AdamW([
#         {"params": enc_params,   "lr": lr_enc,   "name": "encoder"},
#         {"params": vel_params,   "lr": lr_vel,   "name": "velocity"},
#         {"params": score_params, "lr": lr_score, "name": "score_net"},
#     ], weight_decay=weight_decay)


# def get_lrs(opt):
#     return tuple(pg["lr"] for pg in opt.param_groups)


# class ThreeGroupScheduler:
#     """
#     Encoder: freeze → linear warmup → cosine
#     Velocity: warmup → cosine
#     ScoreNet: warmup (nhỏ) → cosine (score net cần ổn định hơn)
#     """
#     def __init__(self, opt, warmup_ep, total_ep,
#                  lr_vel, lr_vel_min, lr_enc_peak,
#                  lr_score, freeze_end_ep, enc_warmup_ep):
#         self.opt          = opt
#         self.warmup       = warmup_ep
#         self.total        = total_ep
#         self.lr_vel       = lr_vel
#         self.lr_vel_min   = lr_vel_min
#         self.lr_enc_peak  = lr_enc_peak
#         self.lr_score     = lr_score
#         self.freeze_end   = freeze_end_ep
#         self.enc_warmup   = enc_warmup_ep
#         self.epoch        = 0

#     def _cos(self, ep_from, ep_to, lr_start, lr_end, ep):
#         t = max(0.0, min(1.0, (ep - ep_from) / max(ep_to - ep_from, 1)))
#         return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t))

#     def step(self):
#         ep = self.epoch
#         # Velocity
#         if ep < self.warmup:
#             lr_vel = self.lr_vel * (0.1 + 0.9 * ep / max(self.warmup-1, 1))
#         else:
#             lr_vel = self._cos(self.warmup, self.total,
#                                self.lr_vel, self.lr_vel_min, ep)
#         # Encoder
#         if ep < self.freeze_end:
#             lr_enc = 0.0
#         elif ep < self.freeze_end + self.enc_warmup:
#             lr_enc = self.lr_enc_peak * (ep - self.freeze_end) / self.enc_warmup
#         else:
#             lr_enc = self._cos(self.freeze_end + self.enc_warmup,
#                                self.total, self.lr_enc_peak, self.lr_vel_min, ep)
#         # ScoreNet: cosine từ lr_score → lr_vel_min
#         if ep < self.warmup:
#             lr_score = self.lr_score * (0.1 + 0.9 * ep / max(self.warmup-1, 1))
#         else:
#             lr_score = self._cos(self.warmup, self.total,
#                                  self.lr_score, self.lr_vel_min, ep)

#         self.opt.param_groups[0]["lr"] = lr_enc
#         self.opt.param_groups[1]["lr"] = lr_vel
#         self.opt.param_groups[2]["lr"] = lr_score
#         self.epoch += 1
#         return lr_vel, lr_enc, lr_score


# # ─────────────────────────────────────────────────────────────────────────────
# #  Evaluation — full val, NO augment, AR-1shot
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def evaluate(model, loader, device, tag="",
#              n_ensemble=20, ema=None, epoch_for_loss=9999,
#              ref_targets=None) -> Dict:
#     bk = None
#     if ema is not None:
#         try:
#             bk = ema.apply_to(model)
#         except Exception as e:
#             print(f"    ⚠ EMA apply: {e}")

#     model.eval()
#     all_ade, all_ate, all_cte = [], [], []
#     step_dist = defaultdict(list)
#     sum_total_w = sum_cfm_w = 0.0
#     sum_n = 0

#     for batch in loader:
#         bl = move(list(batch), device)
#         gt = bl[1]
#         B  = bl[0].shape[1]

#         # Val loss (no augment)
#         try:
#             bd = model.get_loss_breakdown(bl, epoch=epoch_for_loss)
#             if torch.isfinite(bd["total"]):
#                 sum_total_w += bd["total"].item() * B
#                 sum_cfm_w   += bd["l_cfm"] * B
#                 sum_n       += B
#         except Exception:
#             pass

#         # AR-1shot sample
#         try:
#             pred, _, _ = model.sample(bl, num_ensemble=n_ensemble)
#         except Exception as e:
#             print(f"  sample error: {e}")
#             continue

#         T        = min(pred.shape[0], gt.shape[0])
#         pred_deg = _norm_to_deg(pred[:T])
#         gt_deg   = _norm_to_deg(gt[:T])
#         dist     = _haversine_deg(pred_deg, gt_deg)
#         ate, cte = _ate_cte(pred_deg, gt_deg)

#         all_ade.extend(dist.mean(0).tolist())
#         if ate.shape[0] > 0:
#             all_ate.extend(ate.abs().mean(0).tolist())
#             all_cte.extend(cte.abs().mean(0).tolist())
#         for h, s in HORIZON_STEPS.items():
#             if s < T:
#                 step_dist[h].extend(dist[s].tolist())

#     if bk is not None:
#         try:
#             ema.restore(model, bk)
#         except Exception:
#             pass

#     def _m(lst): return float(np.mean(lst)) if lst else float("nan")

#     val_loss = sum_total_w / max(sum_n, 1)
#     result = {
#         "ADE": _m(all_ade), "ATE": _m(all_ate), "CTE": _m(all_cte),
#         "n":   len(all_ade),
#         "val_loss": val_loss,
#         "val_cfm":  sum_cfm_w / max(sum_n, 1),
#     }
#     for h in HORIZON_STEPS:
#         result[f"{h}h"] = _m(step_dist[h])

#     ade, ate_, cte_ = result["ADE"], result["ATE"], result["CTE"]
#     # [TRAIN-3] Combined score: 0.5*ADE + 0.3*ATE + 0.2*CTE
#     result["combined_score"] = (
#         0.5*ade + 0.3*ate_ + 0.2*cte_
#         if all(np.isfinite(x) for x in [ade, ate_, cte_]) else ade)

#     ref = ref_targets or ST_TRANS_VAL
#     def _v(k): return result.get(k, float("nan"))
#     def _ok(k): return "✓" if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9) else "✗"

#     print(f"\n  {'='*72}")
#     print(f"  [{tag}]  n={result['n']}  (AR-1shot, NO augment)")
#     print(f"  {'─'*72}")
#     print(f"  Val Loss : total={val_loss:.6f}  cfm={result['val_cfm']:.6f}")
#     print(f"  {'─'*72}")
#     print(f"  ADE = {_v('ADE'):7.1f} km  {_ok('ADE')}  (< {ref.get('ADE',999):.1f})")
#     print(f"  ATE = {_v('ATE'):7.1f} km  {_ok('ATE')}  (< {ref.get('ATE',999):.1f})")
#     print(f"  CTE = {_v('CTE'):7.1f} km  {_ok('CTE')}  (< {ref.get('CTE',999):.1f})")
#     print(f"  Combined = {_v('combined_score'):.1f}  (0.5×ADE + 0.3×ATE + 0.2×CTE)")
#     print(f"  {'─'*72}")
#     print(f"  12h={_v('12h'):6.1f}  24h={_v('24h'):6.1f}"
#           f"  48h={_v('48h'):6.1f}  72h={_v('72h'):6.1f} km")
#     beat = [f"{k}={_v(k):.0f}<{ref[k]:.0f}"
#             for k in ["ADE","ATE","CTE","12h","24h","48h","72h"]
#             if np.isfinite(_v(k)) and _v(k) < ref.get(k, 1e9)]
#     print(f"  {'─'*72}")
#     print(f"  BEAT: {' | '.join(beat) if beat else 'none yet'}")
#     print(f"  {'='*72}\n")
#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# #  Checkpoint
# # ─────────────────────────────────────────────────────────────────────────────

# def _save(path, epoch, model, opt, sched, best_score,
#           ema=None, scaler=None, extra=None):
#     m   = _unwrap(model)
#     esd = None
#     if ema is not None:
#         try:
#             esd = {k: v.cpu().clone() for k, v in ema.shadow.items()}
#         except Exception:
#             pass
#     payload = {
#         "epoch": epoch, "model": m.state_dict(),
#         "optimizer": opt.state_dict(),
#         "scheduler": sched.epoch,
#         "scaler": scaler.state_dict() if scaler is not None else None,
#         "best_score": best_score,
#         "best_ade": best_score,
#         "ema": esd,
#     }
#     if extra:
#         payload.update(extra)
#     torch.save(payload, path)


# # ─────────────────────────────────────────────────────────────────────────────
# #  Args
# # ─────────────────────────────────────────────────────────────────────────────

# def get_args():
#     p = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     p.add_argument("--dataset_root",          default="TCND_vn")
#     p.add_argument("--obs_len",               default=8,      type=int)
#     p.add_argument("--pred_len",              default=12,     type=int)
#     p.add_argument("--num_workers",           default=2,      type=int)
#     p.add_argument("--other_modal",           default="gph")
#     p.add_argument("--delim",                 default=" ")
#     p.add_argument("--skip",                  default=1,      type=int)
#     p.add_argument("--min_ped",               default=1,      type=int)
#     p.add_argument("--threshold",             default=0.002,  type=float)
#     # Model
#     p.add_argument("--d_cond",                default=256,    type=int)
#     p.add_argument("--d_model",               default=256,    type=int)
#     p.add_argument("--nhead",                 default=8,      type=int)
#     p.add_argument("--num_dec_layers",        default=4,      type=int)
#     p.add_argument("--dim_ff",               default=512,    type=int)
#     p.add_argument("--dropout",               default=0.1,    type=float)
#     p.add_argument("--unet_in_ch",            default=13,     type=int)
#     p.add_argument("--sigma_min",             default=0.02,   type=float)
#     p.add_argument("--sigma_max",             default=0.12,   type=float)
#     p.add_argument("--lambda_reg",            default=0.3,    type=float)
#     p.add_argument("--lambda_score",          default=0.1,    type=float)
#     p.add_argument("--use_ot",                default=True,   action="store_true")
#     p.add_argument("--no_ot",                 dest="use_ot",  action="store_false")
#     p.add_argument("--ot_epsilon",            default=0.05,   type=float)
#     p.add_argument("--n_ensemble",            default=20,     type=int)
#     p.add_argument("--ar_stage_len",          default=4,      type=int)
#     # Training
#     p.add_argument("--num_epochs",            default=120,    type=int)
#     p.add_argument("--batch_size",            default=64,     type=int)
#     p.add_argument("--lr",                    default=1e-4,   type=float,
#                    help="lr velocity (nhỏ hơn vì fine-tune)")
#     p.add_argument("--lr_min",               default=1e-6,   type=float)
#     p.add_argument("--lr_enc",               default=2e-5,   type=float,
#                    help="lr encoder (nhỏ, đã converge)")
#     p.add_argument("--lr_score",              default=2e-4,   type=float,
#                    help="lr score_net (cao hơn vì mới hoàn toàn)")
#     p.add_argument("--warmup_epochs",         default=3,      type=int)
#     p.add_argument("--weight_decay",          default=1e-4,   type=float)
#     p.add_argument("--grad_clip",             default=1.0,    type=float)
#     p.add_argument("--use_amp",               action="store_true", default=False)
#     p.add_argument("--use_ema",               default=True,   action="store_true")
#     p.add_argument("--no_ema",                dest="use_ema", action="store_false")
#     # Encoder freeze (nhỏ hơn vì fine-tune)
#     p.add_argument("--freeze_encoder_epochs", default=5,      type=int)
#     p.add_argument("--encoder_warmup_epochs", default=3,      type=int)
#     # Eval
#     p.add_argument("--val_freq",              default=5,      type=int)
#     p.add_argument("--patience",              default=30,     type=int)
#     p.add_argument("--min_ep",               default=15,     type=int)
#     # IO
#     p.add_argument("--output_dir",            default="runs/fm_v3")
#     p.add_argument("--gpu_num",               default="0")
#     p.add_argument("--resume",                default=None,
#                    help="Resume từ checkpoint v3")
#     p.add_argument("--pretrain",              default=None,
#                    help="[TRAIN-1] Load từ checkpoint v2.1 (strict=False)")
#     p.add_argument("--test_at_end",           action="store_true", default=True)
#     return p.parse_args()


# # ─────────────────────────────────────────────────────────────────────────────
# #  Main
# # ─────────────────────────────────────────────────────────────────────────────

# def main(args):
#     if torch.cuda.is_available():
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output_dir, exist_ok=True)
#     best_ckpt = os.path.join(args.output_dir, "best_model.pth")
#     last_ckpt = os.path.join(args.output_dir, "last_model.pth")

#     print("=" * 72)
#     print("  TC-FlowMatching v3")
#     print(f"  [FIX-A] AR-1shot: {args.pred_len//args.ar_stage_len} stages × {args.ar_stage_len} steps")
#     print(f"  [FIX-B] Learned step weights per sample")
#     print(f"  [FIX-C] Learned sigma per sample ∈ [{args.sigma_min},{args.sigma_max}]")
#     print(f"  [FIX-D] ScoreNet learned selection")
#     print(f"  [FIX-E] Aug: shift±15km + scale×0.6-1.5 + rot±10° + mixup + noise")
#     print(f"  [FIX-F] Stage weights: 1.0, 1.5, 2.0 → 72h penalized more")
#     print(f"  [TRAIN-1] Fine-tune từ: {args.pretrain or 'scratch'}")
#     print(f"  [TRAIN-3] Val metric: combined = 0.5×ADE + 0.3×ATE + 0.2×CTE")
#     print("=" * 72)

#     # ── Data ──────────────────────────────────────────────────────────────
#     print("\n  Loading data...")
#     trd, trl = data_loader(
#         args, {"root": args.dataset_root, "type": "train"}, test=False)
#     vd, val_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "val"}, test=True)
#     print(f"  train : {len(trd)} ({len(trl)} batches/ep)")
#     print(f"  val   : {len(vd)} ({len(val_loader)} batches) ← FULL VAL")

#     # ── Model ─────────────────────────────────────────────────────────────
#     model = TCFlowMatching(
#         pred_len=args.pred_len,       obs_len=args.obs_len,
#         unet_in_ch=args.unet_in_ch,  d_cond=args.d_cond,
#         d_model=args.d_model,         nhead=args.nhead,
#         num_dec_layers=args.num_dec_layers, dim_ff=args.dim_ff,
#         dropout=args.dropout,
#         sigma_min=args.sigma_min,     sigma_max=args.sigma_max,
#         lambda_reg=args.lambda_reg,   lambda_score=args.lambda_score,
#         use_ot=args.use_ot,           ot_epsilon=args.ot_epsilon,
#         use_ema=args.use_ema,
#         n_ensemble=args.n_ensemble,   ar_stage_len=args.ar_stage_len,
#     ).to(device)

#     model.init_ema()
#     ema = getattr(_unwrap(model), "_ema", None)

#     raw   = _unwrap(model)
#     n_enc = sum(p.numel() for p in raw.encoder.parameters())
#     n_vel = sum(p.numel() for p in raw.velocity.parameters())
#     n_sc  = sum(p.numel() for p in raw.score_net.parameters())
#     print(f"\n  Encoder   : {n_enc:,}")
#     print(f"  Velocity  : {n_vel:,}")
#     print(f"  ScoreNet  : {n_sc:,}")
#     print(f"  Total     : {n_enc+n_vel+n_sc:,}")

#     # [TRAIN-1] Load pretrained weights từ v2.1
#     if args.pretrain and os.path.exists(args.pretrain):
#         ck = torch.load(args.pretrain, map_location=device)
#         sd = ck.get("model", ck)
#         missing, unexpected = _unwrap(model).load_state_dict(sd, strict=False)
#         print(f"\n  [TRAIN-1] Loaded pretrain: {args.pretrain}")
#         print(f"  Missing : {len(missing)} keys (new modules: ar_enc, ar_gate, step_weight_net, sigma_net, score_net)")
#         print(f"  Unexpected: {len(unexpected)} keys")
#         if ema and ck.get("ema"):
#             loaded_ema = 0
#             for k, v in ck["ema"].items():
#                 if k in ema.shadow:
#                     ema.shadow[k].copy_(v.to(device))
#                     loaded_ema += 1
#             print(f"  EMA loaded: {loaded_ema} keys")

#     # ── 3-group optimizer ─────────────────────────────────────────────────
#     lr_enc_init = 0.0 if args.freeze_encoder_epochs > 0 else args.lr_enc
#     opt = build_optimizer(model,
#                           lr_vel=args.lr,
#                           lr_enc=lr_enc_init,
#                           lr_score=args.lr_score,
#                           weight_decay=args.weight_decay)
#     scaler = GradScaler("cuda", enabled=args.use_amp)
#     sched  = ThreeGroupScheduler(
#         opt=opt, warmup_ep=args.warmup_epochs,
#         total_ep=args.num_epochs,
#         lr_vel=args.lr, lr_vel_min=args.lr_min,
#         lr_enc_peak=args.lr_enc,
#         lr_score=args.lr_score,
#         freeze_end_ep=args.freeze_encoder_epochs,
#         enc_warmup_ep=args.encoder_warmup_epochs)

#     print(f"\n  LR velocity   : {args.lr:.0e} → {args.lr_min:.0e}")
#     print(f"  LR encoder    : 0 ({args.freeze_encoder_epochs} ep) → {args.lr_enc:.0e}")
#     print(f"  LR score_net  : {args.lr_score:.0e} → {args.lr_min:.0e}")

#     # ── Resume từ v3 checkpoint ────────────────────────────────────────────
#     start_ep     = 0
#     best_score   = float("inf")
#     patience_cnt = 0
#     val_ade_history = []

#     if args.resume and os.path.exists(args.resume):
#         ck = torch.load(args.resume, map_location=device)
#         _unwrap(model).load_state_dict(ck["model"], strict=False)
#         try:
#             opt.load_state_dict(ck["optimizer"])
#         except Exception as e:
#             print(f"  ⚠ Opt: {e}")
#         sched.epoch  = ck.get("scheduler", 0)
#         start_ep     = ck.get("epoch", 0) + 1
#         best_score   = ck.get("best_score", float("inf"))
#         patience_cnt = ck.get("patience_cnt", 0)
#         if scaler and ck.get("scaler"):
#             try:
#                 scaler.load_state_dict(ck["scaler"])
#             except Exception:
#                 pass
#         if ema and ck.get("ema"):
#             for k, v in ck["ema"].items():
#                 if k in ema.shadow:
#                     ema.shadow[k].copy_(v.to(device))
#         print(f"  ↩ Resumed ep{start_ep}  best_score={best_score:.1f}")

#     try:
#         model = torch.compile(model, mode="reduce-overhead")
#         print("  torch.compile: ok")
#     except Exception:
#         pass

#     nstep = len(trl)
#     print()
#     print("=" * 72)
#     print(f"  TRAINING  ({nstep} steps/ep × {args.num_epochs} ep)")
#     print(f"  Augment: shift±15km + scale×0.6-1.5 + rot±10° + mixup")
#     print(f"  Val: NO augment, AR-1shot, combined score")
#     print("=" * 72)

#     for ep in range(start_ep, args.num_epochs):
#         model.train()
#         sum_loss_w = sum_n_train = 0
#         sum_cfm = sum_reg = sum_score = sum_ade1 = 0.0
#         t0 = time.perf_counter()

#         for i, batch in enumerate(trl):
#             bl = move(list(batch), device)
#             B  = bl[0].shape[1]

#             # Augment (chỉ train)
#             bl_aug = augment_batch(bl)

#             opt.zero_grad()
#             with autocast(device_type="cuda", enabled=args.use_amp):
#                 bd = model.get_loss_breakdown(bl_aug, epoch=ep)

#             scaler.scale(bd["total"]).backward()
#             scaler.unscale_(opt)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#             scaler.step(opt)
#             scaler.update()
#             model.ema_update()

#             sum_loss_w  += bd["total"].item() * B
#             sum_n_train += B
#             sum_cfm     += bd["l_cfm"]
#             sum_reg     += bd["l_reg"]
#             sum_score   += bd["l_score"]
#             sum_ade1    += bd["ade_1step"]

#             if i % 30 == 0:
#                 lr_enc, lr_vel, lr_sc = get_lrs(opt)
#                 print(f"  [{ep:>3}][{i:>3}/{nstep}]"
#                       f"  total={bd['total'].item():.4f}"
#                       f"  cfm={bd['l_cfm']:.4f}"
#                       f"  reg={bd['l_reg']:.4f}"
#                       f"  sc={bd['l_score']:.4f}"
#                       f"  lam={bd['lam_reg']:.3f}"
#                       f"  ade1={bd['ade_1step']:.0f}km"
#                       f"  σ={bd['sigma']:.3f}"
#                       f"  lr_vel={lr_vel:.2e}"
#                       f"  lr_enc={lr_enc:.2e}")

#         train_loss = sum_loss_w / max(sum_n_train, 1)
#         avg_cfm    = sum_cfm / nstep
#         avg_reg    = sum_reg / nstep
#         avg_score  = sum_score / nstep
#         avg_ade1   = sum_ade1 / nstep
#         lr_enc_used, lr_vel_used, lr_sc_used = get_lrs(opt)
#         lr_vel_next, lr_enc_next, lr_sc_next = sched.step()

#         if ep == args.freeze_encoder_epochs:
#             print(f"\n  *** Ep{ep}: encoder warmup bắt đầu ***")
#         if ep == args.freeze_encoder_epochs + args.encoder_warmup_epochs:
#             print(f"\n  *** Ep{ep}: encoder fully active ***")

#         print(f"\n  ── Ep{ep:>3}"
#               f"  train={train_loss:.6f}"
#               f"  cfm={avg_cfm:.4f}"
#               f"  reg={avg_reg:.4f}"
#               f"  score={avg_score:.4f}"
#               f"  ade1={avg_ade1:.0f}km"
#               f"  lr_vel={lr_vel_used:.2e}"
#               f"  lr_enc={lr_enc_used:.2e}"
#               f"  t={time.perf_counter()-t0:.0f}s")

#         _save(last_ckpt, ep, model, opt, sched, best_score, ema, scaler)

#         # ── Val ───────────────────────────────────────────────────────────
#         if ep % args.val_freq == 0:
#             r = evaluate(model, val_loader, device,
#                          tag=f"VAL ep{ep}",
#                          n_ensemble=args.n_ensemble,
#                          ema=ema, epoch_for_loss=ep,
#                          ref_targets=ST_TRANS_VAL)

#             val_ade  = r["ADE"]
#             score    = r["combined_score"]
#             val_loss = r["val_loss"]
#             val_ade_history.append(val_ade)

#             # Val ADE trend
#             if len(val_ade_history) >= 4:
#                 recent = float(np.mean(val_ade_history[-2:]))
#                 before = float(np.mean(val_ade_history[-4:-2]))
#                 trend  = recent - before
#                 if   trend >  5.0: trend_str = f"ADE↑{trend:+.1f}km ⚠"
#                 elif trend < -5.0: trend_str = f"ADE↓{trend:+.1f}km ✓"
#                 else:              trend_str = f"ADE→{trend:+.1f}km"
#             else:
#                 trend_str = "—"

#             print(f"  train={train_loss:.6f}  val={val_loss:.6f}"
#                   f"  val_ADE={val_ade:.1f}km  combined={score:.1f}"
#                   f"  trend={trend_str}")

#             # [TRAIN-4] Save best từ ep 0, không cần min_ep
#             if score < best_score:
#                 best_score   = score
#                 patience_cnt = 0
#                 _save(best_ckpt, ep, model, opt, sched, best_score, ema, scaler,
#                       extra={"val_ade":   r["ADE"],
#                              "val_ate":   r["ATE"],
#                              "val_cte":   r["CTE"],
#                              "val_loss":  val_loss,
#                              "val_score": score,
#                              "patience_cnt": 0})
#                 print(f"  ✅ Best! score={best_score:.2f}"
#                       f"  ADE={r['ADE']:.1f} ATE={r['ATE']:.1f}"
#                       f"  CTE={r['CTE']:.1f}")
#             else:
#                 if ep >= args.min_ep:
#                     patience_cnt += args.val_freq
#                 print(f"  No improve {patience_cnt}/{args.patience}"
#                       f"  (best={best_score:.1f})")
#                 if ep >= args.min_ep and patience_cnt >= args.patience:
#                     print(f"  ⛔ Early stop @ ep{ep}")
#                     break

#         # Checkpoint mỗi 5 epoch
#         if ep % 5 == 0:
#             _save(os.path.join(args.output_dir, f"ckpt_ep{ep:03d}.pth"),
#                   ep, model, opt, sched, best_score, ema, scaler)
#             print(f"  💾 ckpt_ep{ep:03d}.pth")

#     print("\n" + "=" * 72)
#     print(f"  Done! best_score={best_score:.2f}")
#     print("=" * 72)

#     # ── Test ──────────────────────────────────────────────────────────────
#     if args.test_at_end and os.path.exists(best_ckpt):
#         print("\n  Loading best ckpt cho TEST...")
#         ck = torch.load(best_ckpt, map_location=device)
#         _unwrap(model).load_state_dict(ck["model"], strict=False)
#         if ema and ck.get("ema"):
#             for k, v in ck["ema"].items():
#                 if k in ema.shadow:
#                     ema.shadow[k].copy_(v.to(device))

#         try:
#             _, test_loader = data_loader(
#                 args, {"root": args.dataset_root, "type": "test"}, test=True)
#         except Exception:
#             print("  No test set → val")
#             test_loader = val_loader

#         r_test = evaluate(model, test_loader, device,
#                           tag="TEST (best ckpt, AR-1shot)",
#                           n_ensemble=args.n_ensemble, ema=ema,
#                           ref_targets=ST_TRANS_TEST)

#         val_ade = ck.get("val_ade", float("nan"))
#         val_ate = ck.get("val_ate", float("nan"))
#         val_cte = ck.get("val_cte", float("nan"))

#         print()
#         print("=" * 72)
#         print("  VAL vs TEST gap (mục tiêu: gap < 53km như ST-Trans)")
#         print("=" * 72)
#         print(f"  {'Metric':<10}  {'Val':>8}  {'Test':>8}  {'Gap':>8}  ST-Trans Test  Beat?")
#         print("  " + "─" * 62)
#         for m, v, t, ref in [
#             ("ADE",  val_ade,  r_test["ADE"],  ST_TRANS_TEST["ADE"]),
#             ("ATE",  val_ate,  r_test["ATE"],  ST_TRANS_TEST["ATE"]),
#             ("CTE",  val_cte,  r_test["CTE"],  ST_TRANS_TEST["CTE"]),
#             ("72h",  float("nan"), r_test["72h"], ST_TRANS_TEST["72h"]),
#         ]:
#             gap  = t - v if (np.isfinite(t) and np.isfinite(v)) else float("nan")
#             beat = "✅" if np.isfinite(t) and t < ref else "✗"
#             print(f"  {m:<10}  {v:>8.1f}  {t:>8.1f}  {gap:>+8.1f}  {ref:>13.1f}  {beat}")
#         print("=" * 72)


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(42); torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)

#     if args.dataset_root == "TCND_vn":
#         _auto = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
#         if os.path.isdir(_auto):
#             args.dataset_root = _auto
#             print(f"  Auto dataset: {_auto}")

#     main(args)

"""
scripts/train_fm.py  ——  TC-FlowMatching v2.2 (train from scratch)

Adjustments cho train từ đầu:
  [SCRATCH-3] AR freeze: ar_enc/ar_gate freeze đến ar_freeze_until=15
  Encoder backbone freeze 10 epoch đầu (tăng từ 5), warmup 5 epoch
  val_freq=5 (giảm từ 10) vì train từ đầu cần monitor sát hơn
  patience=40 (tăng từ 30) vì cần nhiều epoch hơn để hội tụ
  num_epochs=150

XAI logging: log hard_score components mỗi val để track model behavior
"""
from __future__ import annotations

import argparse, math, os, time
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
             use_tta=False, n_tta=8, epoch_for_loss=9999) -> Dict:
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
    ck = torch.load(path, map_location=device)
    sd = ck.get("model", ck)
    missing, unexpected = _unwrap(model).load_state_dict(sd, strict=False)
    print(f"  Loaded: {path}")
    if missing: print(f"  Missing: {len(missing)} keys")
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
    p.add_argument("--sigma_max",         default=0.12,  type=float)   # SCRATCH-1
    p.add_argument("--lambda_reg",        default=0.3,   type=float)   # SCRATCH-4
    p.add_argument("--use_ot",            default=True,  action="store_true")
    p.add_argument("--no_ot",             dest="use_ot", action="store_false")
    p.add_argument("--ot_epsilon",        default=0.05,  type=float)
    p.add_argument("--n_ensemble",        default=20,    type=int)
    p.add_argument("--sigma_inference",   default=0.04,  type=float)
    p.add_argument("--ar_inference",      default=True,  action="store_true")
    p.add_argument("--no_ar",             dest="ar_inference", action="store_false")
    p.add_argument("--ar_stage_len",      default=4,     type=int)
    p.add_argument("--ar_freeze_until",   default=15,    type=int)     # SCRATCH-3
    p.add_argument("--n_inference_steps", default=1,     type=int)     # SCRATCH-5
    # Training
    p.add_argument("--num_epochs",        default=150,   type=int)
    p.add_argument("--batch_size",        default=64,    type=int)
    p.add_argument("--lr",                default=2e-4,  type=float)   # cao hơn khi train scratch
    p.add_argument("--lr_enc",            default=5e-5,  type=float)   # cao hơn khi train scratch
    p.add_argument("--lr_min",            default=1e-6,  type=float)
    p.add_argument("--warmup_epochs",     default=5,     type=int)
    p.add_argument("--freeze_enc_epochs", default=10,    type=int)     # tăng từ 5 → 10
    p.add_argument("--enc_warmup_epochs", default=5,     type=int)     # tăng từ 3 → 5
    p.add_argument("--weight_decay",      default=1e-4,  type=float)
    p.add_argument("--grad_clip",         default=1.0,   type=float)
    p.add_argument("--use_amp",           default=False, action="store_true")
    p.add_argument("--use_ema",           default=True,  action="store_true")
    p.add_argument("--no_ema",            dest="use_ema",action="store_false")
    # Eval
    p.add_argument("--val_freq",          default=5,     type=int)     # giảm từ 10 → 5
    p.add_argument("--patience",          default=40,    type=int)     # tăng từ 30 → 40
    p.add_argument("--min_ep",            default=30,    type=int)     # tăng từ 10 → 30
    # IO
    p.add_argument("--output_dir",        default="runs/fm_v22_scratch")
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
    print("  TC-FlowMatching v2.2 — TRAIN FROM SCRATCH")
    print(f"  sigma_max={args.sigma_max} (tăng cho exploration)")
    print(f"  lambda_reg={args.lambda_reg}  L_reg ramp: ep20→ep40")
    print(f"  AR freeze đến ep{args.ar_freeze_until}")
    print(f"  Encoder freeze {args.freeze_enc_epochs}ep + warmup {args.enc_warmup_epochs}ep")
    print(f"  n_inference_steps={args.n_inference_steps} (1-step Euler)")
    print(f"  XAI: hard_score components, AR gate values, uncertainty")
    print("=" * 72)

    trd, trl = data_loader(args, {"root": args.dataset_root, "type": "train"}, test=False)
    vd, val_loader = data_loader(args, {"root": args.dataset_root, "type": "val"}, test=True)
    print(f"  train: {len(trd)} ({len(trl)} batches/ep)")
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
        ar_inference=args.ar_inference, ar_stage_len=args.ar_stage_len,
        ar_freeze_until=args.ar_freeze_until,
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
        freeze_enc = rel_ep < args.freeze_enc_epochs
        ar_frozen  = rel_ep < args.ar_freeze_until
        raw_enc = _unwrap(model).encoder
        ar_param_ids = set(
            id(p) for p in list(raw_enc.ar_enc.parameters())
                           + list(raw_enc.ar_gate.parameters()))
        for p in raw_enc.parameters():
            if id(p) in ar_param_ids:
                p.requires_grad_(not ar_frozen)
            else:
                p.requires_grad_(not freeze_enc)

        if rel_ep == 0:
            print(f"  *** Encoder backbone frozen ({args.freeze_enc_epochs}ep) ***")
            print(f"  *** AR modules frozen đến ep{args.ar_freeze_until} ***")
        if rel_ep == args.freeze_enc_epochs:
            print(f"\n  *** Ep{ep}: encoder backbone unfrozen ***")
        if rel_ep == args.ar_freeze_until:
            print(f"\n  *** Ep{ep}: AR modules unfrozen — AR conditioning bắt đầu học ***")

        for i, batch in enumerate(trl):
            bl = move(list(batch), device)
            bl_aug = augment_batch(bl)

            opt.zero_grad()
            with autocast(device_type="cuda", enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl_aug, epoch=ep)

            scaler.scale(bd["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # [BUG-H FIXED] Zero gradient của frozen params SAU backward,
            # TRƯỚC optimizer.step() để đảm bảo chúng không được update.
            raw_enc_inner = _unwrap(model).encoder
            for p in raw_enc_inner.parameters():
                if p.grad is None: continue
                pid = id(p)
                is_ar = pid in ar_param_ids
                if (is_ar and ar_frozen) or (not is_ar and freeze_enc):
                    p.grad.zero_()

            scaler.step(opt); scaler.update()
            model.ema_update()

            sum_loss += bd["total"].item(); sum_cfm += bd["l_cfm"]
            sum_reg  += bd["l_reg"];       sum_ade1 += bd["ade_1step"]

            if i % 30 == 0:
                lr_vel, lr_enc = sched.get_lrs()
                enc_status = "frozen" if freeze_enc else "active"
                ar_status  = "frozen" if ar_frozen  else "active"
                print(f"  [{ep:>3}][{i:>3}/{nstep}]"
                      f"  total={bd['total'].item():.4f}"
                      f"  cfm={bd['l_cfm']:.4f}"
                      f"  reg={bd['l_reg']:.4f}"
                      f"  lam={bd['lam_reg']:.3f}"
                      f"  ade1={bd['ade_1step']:.0f}km"
                      f"  σ={bd['sigma']:.3f}"
                      f"  enc={enc_status}"
                      f"  ar={ar_status}"
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
                         use_tta=False, epoch_for_loss=ep)

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
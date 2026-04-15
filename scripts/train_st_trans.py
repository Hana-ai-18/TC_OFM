"""
scripts/train_st_trans.py  ── ST-Trans Baseline Training
=========================================================
THUẬT TOÁN: Faiaz et al. (2026) Expert Systems With Applications 317, 131972
"Physics-guided non-autoregressive transformer for lightweight cyclone
 track prediction in the Bay of Bengal"

HYPERPARAMETERS theo paper (§3.6):
  - Epochs    : 1200 (paper) → early stopping
  - Batch     : 90 (paper)
  - Optimizer : AdamW + weight decay
  - LR Sched  : ReduceLROnPlateau (monitor val DPE)
  - Grad clip : 0.5
  - Patience  : 100 epochs
  - Loss      : Physics-guided composite (DPE + MSE + speed + accel)
  - d_model   : 64
  - nhead     : 4
  - dim_ff    : 512

SO SÁNH với bài của bạn:
  - Dùng chung DataLoader
  - Log ADE tại 12h/24h/48h/72h cùng format
  - Save metrics CSV cùng format
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import random
import csv
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader
from Model.st_trans_model import (
    STTrans, STTransAR,
    haversine_km, _norm_to_deg, HORIZON_STEPS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def move(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return out


def make_val_subset_loader(val_dataset, subset_size, batch_size, collate_fn):
    n   = len(val_dataset)
    rng = random.Random(42)
    idx = rng.sample(range(n), min(subset_size, n))
    return DataLoader(Subset(val_dataset, idx),
                      batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=0, drop_last=False)


def save_metrics_csv(row: dict, csv_path: str):
    write_hdr = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_hdr:
            w.writeheader()
        w.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_ade   = []
    step_buf  = {h: [] for h in HORIZON_STEPS}

    for batch in loader:
        bl      = move(list(batch), device)
        pred, _, _ = model.sample(bl)
        gt      = bl[1]
        T       = min(pred.shape[0], gt.shape[0])
        pred_d  = _norm_to_deg(pred[:T])
        gt_d    = _norm_to_deg(gt[:T])
        dist    = haversine_km(pred_d, gt_d)   # [T, B]
        all_ade.extend(dist.mean(0).tolist())
        for h, s in HORIZON_STEPS.items():
            if s < T:
                step_buf[h].extend(dist[s].tolist())

    result = dict(ADE=float(np.mean(all_ade)) if all_ade else float("nan"))
    for h, vals in step_buf.items():
        result[f"{h}h"] = float(np.mean(vals)) if vals else float("nan")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train ST-Trans baseline (Faiaz et al. 2026)")

    p.add_argument("--dataset_root", default="TCND_vn",  type=str)
    p.add_argument("--obs_len",      default=8,          type=int)
    p.add_argument("--pred_len",     default=12,         type=int)

    # Model hyperparameters
    p.add_argument("--model_type",   default="non_ar",   type=str,
                   choices=["non_ar", "ar"],
                   help="non_ar: ST-Trans | ar: ST-Trans-AR")
    p.add_argument("--d_model",      default=64,         type=int)
    p.add_argument("--nhead",        default=4,          type=int)
    p.add_argument("--num_enc_layers", default=1,        type=int)
    p.add_argument("--num_dec_layers", default=3,        type=int)
    p.add_argument("--dim_ff",       default=512,        type=int)
    p.add_argument("--dropout",      default=0.1,        type=float)

    # Physics loss weights (paper §3.5.1)
    p.add_argument("--lambda_speed", default=0.1,        type=float)
    p.add_argument("--lambda_accel", default=0.01,       type=float)
    p.add_argument("--w_mse",        default=0.05,       type=float)
    p.add_argument("--v_max_kmh",    default=80.0,       type=float)

    # Training
    p.add_argument("--num_epochs",   default=1200,       type=int)
    p.add_argument("--batch_size",   default=90,         type=int)
    p.add_argument("--lr",           default=1e-3,       type=float)
    p.add_argument("--weight_decay", default=1e-4,       type=float)
    p.add_argument("--grad_clip",    default=0.5,        type=float,
                   help="Paper: gradient clip = 0.5")
    p.add_argument("--patience",     default=100,        type=int,
                   help="Early stopping patience (paper: 100)")
    p.add_argument("--min_epochs",   default=50,         type=int)
    p.add_argument("--lr_patience",  default=20,         type=int)
    p.add_argument("--lr_factor",    default=0.5,        type=float)
    p.add_argument("--lr_min",       default=1e-6,       type=float)
    p.add_argument("--val_freq",     default=5,          type=int)
    p.add_argument("--val_subset",   default=600,        type=int)
    p.add_argument("--num_workers",  default=2,          type=int)

    # I/O
    p.add_argument("--output_dir",   default="runs/st_trans", type=str)
    p.add_argument("--metrics_csv",  default="metrics.csv",   type=str)
    p.add_argument("--gpu_num",      default="0",             type=str)

    # DataLoader compat
    p.add_argument("--delim",        default=" ")
    p.add_argument("--skip",         default=1,   type=int)
    p.add_argument("--min_ped",      default=1,   type=int)
    p.add_argument("--threshold",    default=0.002, type=float)
    p.add_argument("--other_modal",  default="gph")
    p.add_argument("--unet_in_ch",   default=13,  type=int)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_csv = os.path.join(args.output_dir, args.metrics_csv)

    print("=" * 70)
    print(f"  ST-TRANS BASELINE  |  type={args.model_type.upper()}")
    print(f"  Faiaz et al. (2026) Expert Systems With Applications 317, 131972")
    print(f"  d_model={args.d_model}  nhead={args.nhead}  dim_ff={args.dim_ff}")
    print(f"  λ_speed={args.lambda_speed}  λ_accel={args.lambda_accel}")
    print(f"  v_max={args.v_max_kmh}km/h  loss=Physics-guided")
    print("=" * 70)

    # ── Data ──
    train_dataset, train_loader = data_loader(
        args, {"root": args.dataset_root, "type": "train"}, test=False)
    val_dataset, val_loader = data_loader(
        args, {"root": args.dataset_root, "type": "val"}, test=True)

    from Model.data.trajectoriesWithMe_unet_training import seq_collate
    val_sub_loader = make_val_subset_loader(
        val_dataset, args.val_subset, args.batch_size, seq_collate)

    print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
    print(f"  val   : {len(val_dataset)} seq")

    # ── Model ──
    if args.model_type == "non_ar":
        model = STTrans(
            obs_len      = args.obs_len,
            pred_len     = args.pred_len,
            d_model      = args.d_model,
            nhead        = args.nhead,
            num_enc_layers = args.num_enc_layers,
            num_dec_layers = args.num_dec_layers,
            dim_ff       = args.dim_ff,
            dropout      = args.dropout,
            lambda_speed = args.lambda_speed,
            lambda_accel = args.lambda_accel,
            w_mse        = args.w_mse,
            v_max_kmh    = args.v_max_kmh,
        ).to(device)
    else:
        model = STTransAR(
            obs_len      = args.obs_len,
            pred_len     = args.pred_len,
            d_model      = args.d_model,
            nhead        = args.nhead,
            num_enc_layers = args.num_enc_layers,
            dim_ff       = args.dim_ff,
            dropout      = args.dropout,
            lambda_speed = args.lambda_speed,
            lambda_accel = args.lambda_accel,
            w_mse        = args.w_mse,
            v_max_kmh    = args.v_max_kmh,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params : {n_params:,}")

    # ── Optimizer + Scheduler (paper §3.6) ──
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.lr_min,
    )

    best_ade     = float("inf")
    patience_cnt = 0
    train_start  = time.perf_counter()

    print("=" * 70)
    print(f"  TRAINING  ({len(train_loader)} steps/epoch)")
    print("=" * 70)

    for epoch in range(args.num_epochs):
        model.train()
        sum_loss = 0.0
        t0 = time.perf_counter()

        for i, batch in enumerate(train_loader):
            bl = move(list(batch), device)

            bd   = model.get_loss_breakdown(bl)
            loss = bd["total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            sum_loss += loss.item()

            if i % 30 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
                      f"  loss={loss.item():.4f}"
                      f"  dpe={bd.get('dpe',0):.2f}km"
                      f"  speed={bd.get('speed',0):.4f}"
                      f"  lr={lr:.2e}")

        avg_train = sum_loss / len(train_loader)

        # ── Val loss (DPE on val set) ──
        model.eval()
        val_dpe = 0.0
        n_val   = 0
        with torch.no_grad():
            for batch in val_loader:
                bl_v = move(list(batch), device)
                bd_v = model.get_loss_breakdown(bl_v)
                val_dpe += bd_v["dpe"]
                n_val   += 1
        avg_val_dpe = val_dpe / max(n_val, 1)

        # Paper §3.6: ReduceLROnPlateau monitors val DPE
        scheduler.step(avg_val_dpe)

        ep_t = time.perf_counter() - t0
        print(f"  Epoch {epoch:>4}  train_loss={avg_train:.4f}"
              f"  val_dpe={avg_val_dpe:.2f}km  t={ep_t:.0f}s")

        # ── ADE evaluation ──
        if epoch % args.val_freq == 0:
            r = evaluate(model, val_sub_loader, device)
            ade12 = r.get("12h", float("nan"))
            ade24 = r.get("24h", float("nan"))
            ade48 = r.get("48h", float("nan"))
            ade72 = r.get("72h", float("nan"))
            ade   = r.get("ADE", float("nan"))

            t12 = "🎯" if ade12 < 50  else "❌"
            t24 = "🎯" if ade24 < 100 else "❌"
            t48 = "🎯" if ade48 < 200 else "❌"
            t72 = "🎯" if ade72 < 300 else "❌"

            print(f"  [VAL ep{epoch}]"
                  f"  ADE={ade:.1f}"
                  f"  12h={ade12:.0f}{t12}"
                  f"  24h={ade24:.0f}{t24}"
                  f"  48h={ade48:.0f}{t48}"
                  f"  72h={ade72:.0f}{t72} km")

            save_metrics_csv({
                "timestamp"  : datetime.now().strftime("%Y%m%d_%H%M%S"),
                "epoch"      : epoch,
                "model_type" : f"ST-Trans-{args.model_type}",
                "train_loss" : f"{avg_train:.6f}",
                "val_dpe_km" : f"{avg_val_dpe:.2f}",
                "ADE_km"     : f"{ade:.2f}",
                "12h_km"     : f"{ade12:.2f}",
                "24h_km"     : f"{ade24:.2f}",
                "48h_km"     : f"{ade48:.2f}",
                "72h_km"     : f"{ade72:.2f}",
            }, metrics_csv)

            if ade < best_ade:
                best_ade     = ade
                patience_cnt = 0
                torch.save({
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "best_ade"   : best_ade,
                    "model_type" : args.model_type,
                    "paper"      : "Faiaz et al. 2026",
                }, os.path.join(args.output_dir, "best_model.pth"))
                print(f"  ✅ Best ADE {best_ade:.1f} km  (epoch {epoch})")
            else:
                patience_cnt += args.val_freq
                print(f"  No improvement {patience_cnt}/{args.patience}"
                      f"  (best={best_ade:.1f} km)")

            if epoch >= args.min_epochs and patience_cnt >= args.patience:
                print(f"  ⛔ Early stop @ epoch {epoch}")
                break

        if epoch % 100 == 0:
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "train_loss" : avg_train,
                "val_dpe"    : avg_val_dpe,
            }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

    total_h = (time.perf_counter() - train_start) / 3600
    print("=" * 70)
    print(f"  Model   : ST-Trans-{args.model_type.upper()}")
    print(f"  Best ADE: {best_ade:.1f} km")
    print(f"  Total   : {total_h:.2f}h")
    print(f"  Metrics : {metrics_csv}")
    print("=" * 70)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main(args)
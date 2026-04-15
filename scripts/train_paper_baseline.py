"""
scripts/train_paper_baseline.py  ── PAPER BASELINE TRAINING
============================================================
THUẬT TOÁN: Rahman et al. (2025) Results in Engineering 26, 105009

HYPERPARAMETERS theo paper (Table 2):
  - Epochs    : 7000  (paper) → dùng early stopping
  - Hidden    : 28    (paper) → 256 vì ctx phức tạp hơn (configurable)
  - Layers    : 3     (paper)
  - Batch     : 90    (paper) → configurable
  - Optimizer : Adam lr=0.001 (paper)
  - LR Sched  : ReduceLROnPlateau (paper §2.8)
  - Loss      : MSE   (paper §2.7)
  - Dropout   : 0.2   (GRU only, paper §2.8)
  - Split     : 60:20:20 (paper §2.7)

SO SÁNH với bài của bạn:
  - Dùng chung DataLoader + collate_fn
  - Log ADE tại 12h / 24h / 48h / 72h cùng format
  - Save metrics CSV cùng format với train_flowmatching.py
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import math
import random
import copy
import csv
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader
from Model.paper_baseline_model import (
    PaperBaseline, compute_ade_per_horizon,
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


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, fast=True) -> dict:
    model.eval()
    all_ade  = []
    all_fde  = []
    step_buf = {h: [] for h in HORIZON_STEPS}

    for batch in loader:
        bl   = move(list(batch), device)
        pred, _, _ = model.sample(bl)
        gt   = bl[1]
        T    = min(pred.shape[0], gt.shape[0])
        pred_deg = _norm_to_deg(pred[:T])
        gt_deg   = _norm_to_deg(gt[:T])
        dist = haversine_km(pred_deg, gt_deg)  # [T, B]
        all_ade.extend(dist.mean(0).tolist())
        all_fde.extend(dist[-1].tolist())
        for h, step_idx in HORIZON_STEPS.items():
            if step_idx < T:
                step_buf[h].extend(dist[step_idx].tolist())

    result = dict(
        ADE = float(np.mean(all_ade)) if all_ade else float("nan"),
        FDE = float(np.mean(all_fde)) if all_fde else float("nan"),
    )
    for h, vals in step_buf.items():
        result[f"{h}h"] = float(np.mean(vals)) if vals else float("nan")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CSV logger
# ══════════════════════════════════════════════════════════════════════════════

def save_metrics_csv(row: dict, csv_path: str):
    write_hdr = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_hdr:
            w.writeheader()
        w.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train paper baseline (Rahman et al. 2025)")

    # Dataset
    p.add_argument("--dataset_root", default="TCND_vn",  type=str)
    p.add_argument("--obs_len",      default=8,          type=int)
    p.add_argument("--pred_len",     default=12,         type=int)

    # Paper hyperparameters (Table 2)
    p.add_argument("--model_type",   default="lstm",     type=str,
                   choices=["lstm", "gru", "rnn"],
                   help="lstm | gru | rnn  (paper §2.4-2.6)")
    p.add_argument("--num_epochs",   default=7000,       type=int,
                   help="Paper: 7000 epochs")
    p.add_argument("--hidden_dim",   default=256,        type=int,
                   help="Paper: 28 (ta dùng 256 vì ctx phức tạp hơn)")
    p.add_argument("--n_layers",     default=3,          type=int,
                   help="Paper: layer_dim=3")
    p.add_argument("--batch_size",   default=90,         type=int,
                   help="Paper: batch_size=90")
    p.add_argument("--lr",           default=0.001,      type=float,
                   help="Paper: Adam lr=0.001")
    p.add_argument("--weight_decay", default=0.0,        type=float,
                   help="Paper: không nói đến weight_decay → 0")
    p.add_argument("--dropout",      default=0.20,       type=float,
                   help="Paper: GRU dropout=0.2 (§2.8)")

    # ReduceLROnPlateau (paper §2.8)
    p.add_argument("--lr_patience",  default=50,         type=int,
                   help="ReduceLROnPlateau patience")
    p.add_argument("--lr_factor",    default=0.5,        type=float,
                   help="ReduceLROnPlateau factor")
    p.add_argument("--lr_min",       default=1e-6,       type=float)

    # Training infra
    p.add_argument("--patience",     default=200,        type=int,
                   help="Early stopping patience (epochs)")
    p.add_argument("--min_epochs",   default=100,        type=int)
    p.add_argument("--use_amp",      action="store_true")
    p.add_argument("--num_workers",  default=2,          type=int)
    p.add_argument("--grad_clip",    default=1.0,        type=float)
    p.add_argument("--val_freq",     default=5,          type=int)
    p.add_argument("--val_subset",   default=600,        type=int)

    # I/O
    p.add_argument("--output_dir",   default="runs/paper_baseline", type=str)
    p.add_argument("--metrics_csv",  default="metrics.csv",         type=str)
    p.add_argument("--gpu_num",      default="0",                   type=str)

    # Unused but kept for data_loader compatibility
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
    from typing import Dict   # local để tránh lỗi ở top-level

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_csv = os.path.join(args.output_dir, args.metrics_csv)

    print("=" * 70)
    print(f"  PAPER BASELINE  |  model={args.model_type.upper()}")
    print(f"  Rahman et al. (2025) Results in Engineering 26, 105009")
    print(f"  hidden={args.hidden_dim}  layers={args.n_layers}  dropout={args.dropout}")
    print(f"  lr={args.lr}  batch={args.batch_size}  loss=MSE")
    print(f"  LR scheduler: ReduceLROnPlateau(patience={args.lr_patience})")
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
    model = PaperBaseline(
        model_type = args.model_type,
        pred_len   = args.pred_len,
        obs_len    = args.obs_len,
        hidden_dim = args.hidden_dim,
        n_layers   = args.n_layers,
        unet_in_ch = args.unet_in_ch,
        dropout    = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params : {n_params:,}")

    # ── Optimizer + Scheduler (paper §2.7-2.8) ──
    optimizer = optim.Adam(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    # Paper §2.8: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = args.lr_factor,
        patience = args.lr_patience,
        min_lr   = args.lr_min,
        # verbose  = True,
    )

    scaler = GradScaler("cuda", enabled=args.use_amp)

    best_ade      = float("inf")
    best_val_loss = float("inf")
    patience_cnt  = 0
    train_start   = time.perf_counter()

    print("=" * 70)
    print(f"  TRAINING  ({len(train_loader)} steps/epoch)")
    print("=" * 70)

    for epoch in range(args.num_epochs):
        model.train()
        sum_loss = 0.0
        t0 = time.perf_counter()

        for i, batch in enumerate(train_loader):
            bl = move(list(batch), device)

            with autocast(device_type="cuda", enabled=args.use_amp):
                loss = model.get_loss(bl)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            sum_loss += loss.item()

            if i % 30 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
                      f"  mse={loss.item():.4f}"
                      f"  lr={lr:.2e}")

        avg_train = sum_loss / len(train_loader)

        # ── Validation loss ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                bl_v = move(list(batch), device)
                with autocast(device_type="cuda", enabled=args.use_amp):
                    val_loss += model.get_loss(bl_v).item()
        avg_val = val_loss / len(val_loader)

        # Paper §2.8: ReduceLROnPlateau step on val loss
        scheduler.step(avg_val)

        ep_t = time.perf_counter() - t0
        print(f"  Epoch {epoch:>4}  train_mse={avg_train:.4f}  "
              f"val_mse={avg_val:.4f}  t={ep_t:.0f}s")

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

            # Save metrics CSV (cùng format để so sánh với bài bạn)
            save_metrics_csv({
                "timestamp"   : datetime.now().strftime("%Y%m%d_%H%M%S"),
                "epoch"       : epoch,
                "model_type"  : args.model_type,
                "train_mse"   : f"{avg_train:.6f}",
                "val_mse"     : f"{avg_val:.6f}",
                "ADE_km"      : f"{ade:.2f}",
                "FDE_km"      : f"{r.get('FDE', float('nan')):.2f}",
                "12h_km"      : f"{ade12:.2f}",
                "24h_km"      : f"{ade24:.2f}",
                "48h_km"      : f"{ade48:.2f}",
                "72h_km"      : f"{ade72:.2f}",
            }, metrics_csv)

            # Save best model
            if ade < best_ade:
                best_ade = ade
                patience_cnt = 0
                torch.save({
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "opt_state"  : optimizer.state_dict(),
                    "best_ade"   : best_ade,
                    "model_type" : args.model_type,
                    "paper"      : "Rahman et al. 2025",
                }, os.path.join(args.output_dir, "best_model.pth"))
                print(f"  ✅ Best ADE {best_ade:.1f} km  (epoch {epoch})")
            else:
                patience_cnt += args.val_freq
                print(f"  No improvement {patience_cnt}/{args.patience}"
                      f"  (best={best_ade:.1f} km)")

            if epoch >= args.min_epochs and patience_cnt >= args.patience:
                print(f"  ⛔ Early stop @ epoch {epoch}")
                break

        # Periodic checkpoint
        if epoch % 100 == 0:
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "train_mse"  : avg_train,
                "val_mse"    : avg_val,
            }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

    total_h = (time.perf_counter() - train_start) / 3600
    print("=" * 70)
    print(f"  Model   : {args.model_type.upper()} (Rahman et al. 2025)")
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
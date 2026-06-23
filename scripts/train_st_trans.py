# # # # # # # # """
# # # # # # # # scripts/train_st_trans.py  ── ST-Trans Baseline Training
# # # # # # # # =========================================================
# # # # # # # # THUẬT TOÁN: Faiaz et al. (2026) Expert Systems With Applications 317, 131972
# # # # # # # # "Physics-guided non-autoregressive transformer for lightweight cyclone
# # # # # # # #  track prediction in the Bay of Bengal"

# # # # # # # # HYPERPARAMETERS theo paper (§3.6):
# # # # # # # #   - Epochs    : 1200 (paper) → early stopping
# # # # # # # #   - Batch     : 90 (paper)
# # # # # # # #   - Optimizer : AdamW + weight decay
# # # # # # # #   - LR Sched  : ReduceLROnPlateau (monitor val DPE)
# # # # # # # #   - Grad clip : 0.5
# # # # # # # #   - Patience  : 100 epochs
# # # # # # # #   - Loss      : Physics-guided composite (DPE + MSE + speed + accel)
# # # # # # # #   - d_model   : 64
# # # # # # # #   - nhead     : 4
# # # # # # # #   - dim_ff    : 512

# # # # # # # # SO SÁNH với bài của bạn:
# # # # # # # #   - Dùng chung DataLoader
# # # # # # # #   - Log ADE tại 12h/24h/48h/72h cùng format
# # # # # # # #   - Save metrics CSV cùng format
# # # # # # # # """
# # # # # # # # from __future__ import annotations

# # # # # # # # import sys, os
# # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # import argparse
# # # # # # # # import time
# # # # # # # # import random
# # # # # # # # import csv
# # # # # # # # from datetime import datetime

# # # # # # # # import numpy as np
# # # # # # # # import torch
# # # # # # # # import torch.optim as optim
# # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # from Model.data.loader_training import data_loader
# # # # # # # # from Model.st_trans_model import (
# # # # # # # #     STTrans, STTransAR,
# # # # # # # #     haversine_km, _norm_to_deg, HORIZON_STEPS,
# # # # # # # # )


# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Helpers
# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # def move(batch, device):
# # # # # # # #     out = list(batch)
# # # # # # # #     for i, x in enumerate(out):
# # # # # # # #         if torch.is_tensor(x):
# # # # # # # #             out[i] = x.to(device)
# # # # # # # #         elif isinstance(x, dict):
# # # # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # #                       for k, v in x.items()}
# # # # # # # #     return out


# # # # # # # # def make_val_subset_loader(val_dataset, subset_size, batch_size, collate_fn):
# # # # # # # #     n   = len(val_dataset)
# # # # # # # #     rng = random.Random(42)
# # # # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # # # #     return DataLoader(Subset(val_dataset, idx),
# # # # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # # # def save_metrics_csv(row: dict, csv_path: str):
# # # # # # # #     write_hdr = not os.path.exists(csv_path)
# # # # # # # #     with open(csv_path, "a", newline="") as fh:
# # # # # # # #         w = csv.DictWriter(fh, fieldnames=list(row.keys()))
# # # # # # # #         if write_hdr:
# # # # # # # #             w.writeheader()
# # # # # # # #         w.writerow(row)


# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Evaluation
# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # @torch.no_grad()
# # # # # # # # def evaluate(model, loader, device) -> dict:
# # # # # # # #     model.eval()
# # # # # # # #     all_ade   = []
# # # # # # # #     step_buf  = {h: [] for h in HORIZON_STEPS}

# # # # # # # #     for batch in loader:
# # # # # # # #         bl      = move(list(batch), device)
# # # # # # # #         pred, _, _ = model.sample(bl)
# # # # # # # #         gt      = bl[1]
# # # # # # # #         T       = min(pred.shape[0], gt.shape[0])
# # # # # # # #         pred_d  = _norm_to_deg(pred[:T])
# # # # # # # #         gt_d    = _norm_to_deg(gt[:T])
# # # # # # # #         dist    = haversine_km(pred_d, gt_d)   # [T, B]
# # # # # # # #         all_ade.extend(dist.mean(0).tolist())
# # # # # # # #         for h, s in HORIZON_STEPS.items():
# # # # # # # #             if s < T:
# # # # # # # #                 step_buf[h].extend(dist[s].tolist())

# # # # # # # #     result = dict(ADE=float(np.mean(all_ade)) if all_ade else float("nan"))
# # # # # # # #     for h, vals in step_buf.items():
# # # # # # # #         result[f"{h}h"] = float(np.mean(vals)) if vals else float("nan")
# # # # # # # #     return result


# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  Args
# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # def get_args():
# # # # # # # #     p = argparse.ArgumentParser(
# # # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
# # # # # # # #         description="Train ST-Trans baseline (Faiaz et al. 2026)")

# # # # # # # #     p.add_argument("--dataset_root", default="TCND_vn",  type=str)
# # # # # # # #     p.add_argument("--obs_len",      default=8,          type=int)
# # # # # # # #     p.add_argument("--pred_len",     default=12,         type=int)

# # # # # # # #     # Model hyperparameters
# # # # # # # #     p.add_argument("--model_type",   default="non_ar",   type=str,
# # # # # # # #                    choices=["non_ar", "ar"],
# # # # # # # #                    help="non_ar: ST-Trans | ar: ST-Trans-AR")
# # # # # # # #     p.add_argument("--d_model",      default=64,         type=int)
# # # # # # # #     p.add_argument("--nhead",        default=4,          type=int)
# # # # # # # #     p.add_argument("--num_enc_layers", default=1,        type=int)
# # # # # # # #     p.add_argument("--num_dec_layers", default=3,        type=int)
# # # # # # # #     p.add_argument("--dim_ff",       default=512,        type=int)
# # # # # # # #     p.add_argument("--dropout",      default=0.1,        type=float)

# # # # # # # #     # Physics loss weights (paper §3.5.1)
# # # # # # # #     p.add_argument("--lambda_speed", default=0.1,        type=float)
# # # # # # # #     p.add_argument("--lambda_accel", default=0.01,       type=float)
# # # # # # # #     p.add_argument("--w_mse",        default=0.05,       type=float)
# # # # # # # #     p.add_argument("--v_max_kmh",    default=80.0,       type=float)

# # # # # # # #     # Training
# # # # # # # #     p.add_argument("--num_epochs",   default=1200,       type=int)
# # # # # # # #     p.add_argument("--batch_size",   default=90,         type=int)
# # # # # # # #     p.add_argument("--lr",           default=1e-3,       type=float)
# # # # # # # #     p.add_argument("--weight_decay", default=1e-4,       type=float)
# # # # # # # #     p.add_argument("--grad_clip",    default=0.5,        type=float,
# # # # # # # #                    help="Paper: gradient clip = 0.5")
# # # # # # # #     p.add_argument("--patience",     default=100,        type=int,
# # # # # # # #                    help="Early stopping patience (paper: 100)")
# # # # # # # #     p.add_argument("--min_epochs",   default=50,         type=int)
# # # # # # # #     p.add_argument("--lr_patience",  default=20,         type=int)
# # # # # # # #     p.add_argument("--lr_factor",    default=0.5,        type=float)
# # # # # # # #     p.add_argument("--lr_min",       default=1e-6,       type=float)
# # # # # # # #     p.add_argument("--val_freq",     default=5,          type=int)
# # # # # # # #     p.add_argument("--val_subset",   default=600,        type=int)
# # # # # # # #     p.add_argument("--num_workers",  default=2,          type=int)

# # # # # # # #     # I/O
# # # # # # # #     p.add_argument("--output_dir",   default="runs/st_trans", type=str)
# # # # # # # #     p.add_argument("--metrics_csv",  default="metrics.csv",   type=str)
# # # # # # # #     p.add_argument("--gpu_num",      default="0",             type=str)

# # # # # # # #     # DataLoader compat
# # # # # # # #     p.add_argument("--delim",        default=" ")
# # # # # # # #     p.add_argument("--skip",         default=1,   type=int)
# # # # # # # #     p.add_argument("--min_ped",      default=1,   type=int)
# # # # # # # #     p.add_argument("--threshold",    default=0.002, type=float)
# # # # # # # #     p.add_argument("--other_modal",  default="gph")
# # # # # # # #     p.add_argument("--unet_in_ch",   default=13,  type=int)

# # # # # # # #     return p.parse_args()


# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # # #  MAIN
# # # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # # def main(args):
# # # # # # # #     if torch.cuda.is_available():
# # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)

# # # # # # # #     print("=" * 70)
# # # # # # # #     print(f"  ST-TRANS BASELINE  |  type={args.model_type.upper()}")
# # # # # # # #     print(f"  Faiaz et al. (2026) Expert Systems With Applications 317, 131972")
# # # # # # # #     print(f"  d_model={args.d_model}  nhead={args.nhead}  dim_ff={args.dim_ff}")
# # # # # # # #     print(f"  λ_speed={args.lambda_speed}  λ_accel={args.lambda_accel}")
# # # # # # # #     print(f"  v_max={args.v_max_kmh}km/h  loss=Physics-guided")
# # # # # # # #     print("=" * 70)

# # # # # # # #     # ── Data ──
# # # # # # # #     train_dataset, train_loader = data_loader(
# # # # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # # # #     val_dataset, val_loader = data_loader(
# # # # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # #     val_sub_loader = make_val_subset_loader(
# # # # # # # #         val_dataset, args.val_subset, args.batch_size, seq_collate)

# # # # # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # # # # #     print(f"  val   : {len(val_dataset)} seq")

# # # # # # # #     # ── Model ──
# # # # # # # #     if args.model_type == "non_ar":
# # # # # # # #         model = STTrans(
# # # # # # # #             obs_len      = args.obs_len,
# # # # # # # #             pred_len     = args.pred_len,
# # # # # # # #             d_model      = args.d_model,
# # # # # # # #             nhead        = args.nhead,
# # # # # # # #             num_enc_layers = args.num_enc_layers,
# # # # # # # #             num_dec_layers = args.num_dec_layers,
# # # # # # # #             dim_ff       = args.dim_ff,
# # # # # # # #             dropout      = args.dropout,
# # # # # # # #             lambda_speed = args.lambda_speed,
# # # # # # # #             lambda_accel = args.lambda_accel,
# # # # # # # #             w_mse        = args.w_mse,
# # # # # # # #             v_max_kmh    = args.v_max_kmh,
# # # # # # # #         ).to(device)
# # # # # # # #     else:
# # # # # # # #         model = STTransAR(
# # # # # # # #             obs_len      = args.obs_len,
# # # # # # # #             pred_len     = args.pred_len,
# # # # # # # #             d_model      = args.d_model,
# # # # # # # #             nhead        = args.nhead,
# # # # # # # #             num_enc_layers = args.num_enc_layers,
# # # # # # # #             dim_ff       = args.dim_ff,
# # # # # # # #             dropout      = args.dropout,
# # # # # # # #             lambda_speed = args.lambda_speed,
# # # # # # # #             lambda_accel = args.lambda_accel,
# # # # # # # #             w_mse        = args.w_mse,
# # # # # # # #             v_max_kmh    = args.v_max_kmh,
# # # # # # # #         ).to(device)

# # # # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # # #     print(f"  params : {n_params:,}")

# # # # # # # #     # ── Optimizer + Scheduler (paper §3.6) ──
# # # # # # # #     optimizer = optim.AdamW(
# # # # # # # #         model.parameters(),
# # # # # # # #         lr=args.lr,
# # # # # # # #         weight_decay=args.weight_decay,
# # # # # # # #     )
# # # # # # # #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# # # # # # # #         optimizer,
# # # # # # # #         mode="min", factor=args.lr_factor,
# # # # # # # #         patience=args.lr_patience, min_lr=args.lr_min,
# # # # # # # #     )

# # # # # # # #     best_ade     = float("inf")
# # # # # # # #     patience_cnt = 0
# # # # # # # #     train_start  = time.perf_counter()

# # # # # # # #     print("=" * 70)
# # # # # # # #     print(f"  TRAINING  ({len(train_loader)} steps/epoch)")
# # # # # # # #     print("=" * 70)

# # # # # # # #     for epoch in range(args.num_epochs):
# # # # # # # #         model.train()
# # # # # # # #         sum_loss = 0.0
# # # # # # # #         t0 = time.perf_counter()

# # # # # # # #         for i, batch in enumerate(train_loader):
# # # # # # # #             bl = move(list(batch), device)

# # # # # # # #             bd   = model.get_loss_breakdown(bl)
# # # # # # # #             loss = bd["total"]

# # # # # # # #             optimizer.zero_grad()
# # # # # # # #             loss.backward()
# # # # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # # # # #             optimizer.step()

# # # # # # # #             sum_loss += loss.item()

# # # # # # # #             if i % 30 == 0:
# # # # # # # #                 lr = optimizer.param_groups[0]["lr"]
# # # # # # # #                 print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
# # # # # # # #                       f"  loss={loss.item():.4f}"
# # # # # # # #                       f"  dpe={bd.get('dpe',0):.2f}km"
# # # # # # # #                       f"  speed={bd.get('speed',0):.4f}"
# # # # # # # #                       f"  lr={lr:.2e}")

# # # # # # # #         avg_train = sum_loss / len(train_loader)

# # # # # # # #         # ── Val loss (DPE on val set) ──
# # # # # # # #         model.eval()
# # # # # # # #         val_dpe = 0.0
# # # # # # # #         n_val   = 0
# # # # # # # #         with torch.no_grad():
# # # # # # # #             for batch in val_loader:
# # # # # # # #                 bl_v = move(list(batch), device)
# # # # # # # #                 bd_v = model.get_loss_breakdown(bl_v)
# # # # # # # #                 val_dpe += bd_v["dpe"]
# # # # # # # #                 n_val   += 1
# # # # # # # #         avg_val_dpe = val_dpe / max(n_val, 1)

# # # # # # # #         # Paper §3.6: ReduceLROnPlateau monitors val DPE
# # # # # # # #         scheduler.step(avg_val_dpe)

# # # # # # # #         ep_t = time.perf_counter() - t0
# # # # # # # #         print(f"  Epoch {epoch:>4}  train_loss={avg_train:.4f}"
# # # # # # # #               f"  val_dpe={avg_val_dpe:.2f}km  t={ep_t:.0f}s")

# # # # # # # #         # ── ADE evaluation ──
# # # # # # # #         if epoch % args.val_freq == 0:
# # # # # # # #             r = evaluate(model, val_sub_loader, device)
# # # # # # # #             ade12 = r.get("12h", float("nan"))
# # # # # # # #             ade24 = r.get("24h", float("nan"))
# # # # # # # #             ade48 = r.get("48h", float("nan"))
# # # # # # # #             ade72 = r.get("72h", float("nan"))
# # # # # # # #             ade   = r.get("ADE", float("nan"))

# # # # # # # #             t12 = "🎯" if ade12 < 50  else "❌"
# # # # # # # #             t24 = "🎯" if ade24 < 100 else "❌"
# # # # # # # #             t48 = "🎯" if ade48 < 200 else "❌"
# # # # # # # #             t72 = "🎯" if ade72 < 300 else "❌"

# # # # # # # #             print(f"  [VAL ep{epoch}]"
# # # # # # # #                   f"  ADE={ade:.1f}"
# # # # # # # #                   f"  12h={ade12:.0f}{t12}"
# # # # # # # #                   f"  24h={ade24:.0f}{t24}"
# # # # # # # #                   f"  48h={ade48:.0f}{t48}"
# # # # # # # #                   f"  72h={ade72:.0f}{t72} km")

# # # # # # # #             save_metrics_csv({
# # # # # # # #                 "timestamp"  : datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # # # #                 "epoch"      : epoch,
# # # # # # # #                 "model_type" : f"ST-Trans-{args.model_type}",
# # # # # # # #                 "train_loss" : f"{avg_train:.6f}",
# # # # # # # #                 "val_dpe_km" : f"{avg_val_dpe:.2f}",
# # # # # # # #                 "ADE_km"     : f"{ade:.2f}",
# # # # # # # #                 "12h_km"     : f"{ade12:.2f}",
# # # # # # # #                 "24h_km"     : f"{ade24:.2f}",
# # # # # # # #                 "48h_km"     : f"{ade48:.2f}",
# # # # # # # #                 "72h_km"     : f"{ade72:.2f}",
# # # # # # # #             }, metrics_csv)

# # # # # # # #             if ade < best_ade:
# # # # # # # #                 best_ade     = ade
# # # # # # # #                 patience_cnt = 0
# # # # # # # #                 torch.save({
# # # # # # # #                     "epoch"      : epoch,
# # # # # # # #                     "model_state": model.state_dict(),
# # # # # # # #                     "best_ade"   : best_ade,
# # # # # # # #                     "model_type" : args.model_type,
# # # # # # # #                     "paper"      : "Faiaz et al. 2026",
# # # # # # # #                 }, os.path.join(args.output_dir, "best_model.pth"))
# # # # # # # #                 print(f"  ✅ Best ADE {best_ade:.1f} km  (epoch {epoch})")
# # # # # # # #             else:
# # # # # # # #                 patience_cnt += args.val_freq
# # # # # # # #                 print(f"  No improvement {patience_cnt}/{args.patience}"
# # # # # # # #                       f"  (best={best_ade:.1f} km)")

# # # # # # # #             if epoch >= args.min_epochs and patience_cnt >= args.patience:
# # # # # # # #                 print(f"  ⛔ Early stop @ epoch {epoch}")
# # # # # # # #                 break

# # # # # # # #         if epoch % 100 == 0:
# # # # # # # #             torch.save({
# # # # # # # #                 "epoch"      : epoch,
# # # # # # # #                 "model_state": model.state_dict(),
# # # # # # # #                 "train_loss" : avg_train,
# # # # # # # #                 "val_dpe"    : avg_val_dpe,
# # # # # # # #             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

# # # # # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # # # # #     print("=" * 70)
# # # # # # # #     print(f"  Model   : ST-Trans-{args.model_type.upper()}")
# # # # # # # #     print(f"  Best ADE: {best_ade:.1f} km")
# # # # # # # #     print(f"  Total   : {total_h:.2f}h")
# # # # # # # #     print(f"  Metrics : {metrics_csv}")
# # # # # # # #     print("=" * 70)


# # # # # # # # if __name__ == "__main__":
# # # # # # # #     args = get_args()
# # # # # # # #     np.random.seed(42)
# # # # # # # #     torch.manual_seed(42)
# # # # # # # #     if torch.cuda.is_available():
# # # # # # # #         torch.cuda.manual_seed_all(42)
# # # # # # # #     main(args)

# # # # # # # """
# # # # # # # scripts/train_st_trans.py  ── ST-Trans Baseline Training
# # # # # # # =========================================================
# # # # # # # THUẬT TOÁN: Faiaz et al. (2026) Expert Systems With Applications 317, 131972

# # # # # # # THAY ĐỔI SO VỚI PHIÊN BẢN TRƯỚC:
# # # # # # #   ✅ Dùng cùng DataLoader / batch_list với paper_baseline (full modal: ảnh + env)
# # # # # # #   ✅ STTrans bây giờ sử dụng PaperEncoder (FNO3D + Mamba + Env_net) làm backbone
# # # # # # #   ✅ Thêm ATE / CTE trong mỗi lần eval
# # # # # # #   ✅ Đánh giá trên tập TEST ở cuối training (--test_at_end)
# # # # # # #   ✅ CSV log bao gồm đầy đủ ADE / ATE / CTE tại 12h / 24h / 48h / 72h

# # # # # # # HYPERPARAMETERS theo paper (§3.6):
# # # # # # #   - Epochs    : 1200 → early stopping
# # # # # # #   - Batch     : 90
# # # # # # #   - Optimizer : AdamW + weight decay
# # # # # # #   - LR Sched  : ReduceLROnPlateau (monitor val DPE)
# # # # # # #   - Grad clip : 0.5
# # # # # # #   - d_model   : 64, nhead : 4, dim_ff : 512
# # # # # # # """
# # # # # # # from __future__ import annotations

# # # # # # # import sys, os
# # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # import argparse
# # # # # # # import time
# # # # # # # import random
# # # # # # # import csv
# # # # # # # from datetime import datetime

# # # # # # # import numpy as np
# # # # # # # import torch
# # # # # # # import torch.optim as optim
# # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # from Model.data.loader_training import data_loader
# # # # # # # from Model.st_trans_model import (
# # # # # # #     STTrans, STTransAR,
# # # # # # # )
# # # # # # # from Model.paper_baseline_model import (
# # # # # # #     haversine_km, _norm_to_deg, _ate_cte_tensors,
# # # # # # #     HORIZON_STEPS,
# # # # # # # )


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Helpers
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # def move(batch, device):
# # # # # # #     out = list(batch)
# # # # # # #     for i, x in enumerate(out):
# # # # # # #         if torch.is_tensor(x):
# # # # # # #             out[i] = x.to(device)
# # # # # # #         elif isinstance(x, dict):
# # # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # #                       for k, v in x.items()}
# # # # # # #     return out


# # # # # # # def make_subset_loader(dataset, subset_size, batch_size, collate_fn, seed=42):
# # # # # # #     n   = len(dataset)
# # # # # # #     rng = random.Random(seed)
# # # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # # #     return DataLoader(Subset(dataset, idx),
# # # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # # def save_metrics_csv(row: dict, csv_path: str):
# # # # # # #     write_hdr = not os.path.exists(csv_path)
# # # # # # #     with open(csv_path, "a", newline="") as fh:
# # # # # # #         w = csv.DictWriter(fh, fieldnames=list(row.keys()))
# # # # # # #         if write_hdr:
# # # # # # #             w.writeheader()
# # # # # # #         w.writerow(row)


# # # # # # # def _fmt(v) -> str:
# # # # # # #     return f"{v:.2f}" if isinstance(v, float) and not np.isnan(v) else "nan"


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Evaluation  (ADE + ATE + CTE)
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # @torch.no_grad()
# # # # # # # def evaluate(model, loader, device) -> dict:
# # # # # # #     """
# # # # # # #     Đánh giá model trên toàn bộ loader.
# # # # # # #     batch_list dùng cùng format với paper_baseline (full multi-modal).
# # # # # # #     """
# # # # # # #     model.eval()

# # # # # # #     all_ade, all_fde = [], []
# # # # # # #     ade_buf = {h: [] for h in HORIZON_STEPS}
# # # # # # #     ate_buf = {h: [] for h in HORIZON_STEPS}
# # # # # # #     cte_buf = {h: [] for h in HORIZON_STEPS}
# # # # # # #     all_ate_abs, all_cte_abs = [], []

# # # # # # #     for batch in loader:
# # # # # # #         bl         = move(list(batch), device)
# # # # # # #         pred, _, _ = model.sample(bl)           # batch_list đầy đủ
# # # # # # #         gt         = bl[1]
# # # # # # #         T          = min(pred.shape[0], gt.shape[0])

# # # # # # #         pred_d = _norm_to_deg(pred[:T])
# # # # # # #         gt_d   = _norm_to_deg(gt[:T])
# # # # # # #         dist   = haversine_km(pred_d, gt_d)     # [T, B]

# # # # # # #         ate, cte = _ate_cte_tensors(pred[:T], gt[:T])   # each [T, B]

# # # # # # #         all_ade.extend(dist.mean(0).tolist())
# # # # # # #         all_fde.extend(dist[-1].tolist())
# # # # # # #         all_ate_abs.extend(ate.abs().mean(0).tolist())
# # # # # # #         all_cte_abs.extend(cte.abs().mean(0).tolist())

# # # # # # #         for h, s in HORIZON_STEPS.items():
# # # # # # #             if s < T:
# # # # # # #                 ade_buf[h].extend(dist[s].tolist())
# # # # # # #                 ate_buf[h].extend(ate[s].abs().tolist())
# # # # # # #                 cte_buf[h].extend(cte[s].abs().tolist())

# # # # # # #     def _mean(lst):
# # # # # # #         return float(np.mean(lst)) if lst else float("nan")

# # # # # # #     result = dict(
# # # # # # #         ADE     = _mean(all_ade),
# # # # # # #         FDE     = _mean(all_fde),
# # # # # # #         ATE_abs = _mean(all_ate_abs),
# # # # # # #         CTE_abs = _mean(all_cte_abs),
# # # # # # #     )
# # # # # # #     for h in HORIZON_STEPS:
# # # # # # #         result[f"{h}h"]         = _mean(ade_buf[h])
# # # # # # #         result[f"ATE_abs_{h}h"] = _mean(ate_buf[h])
# # # # # # #         result[f"CTE_abs_{h}h"] = _mean(cte_buf[h])

# # # # # # #     return result


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Test set evaluation
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # def run_test_evaluation(model, ckpt_path: str, args, device,
# # # # # # #                         collate_fn, csv_path: str):
# # # # # # #     """Load best checkpoint rồi đánh giá toàn bộ test set."""
# # # # # # #     print("\n" + "=" * 70)
# # # # # # #     print("  TEST SET EVALUATION  (ST-Trans)")
# # # # # # #     print("=" * 70)

# # # # # # #     ckpt = torch.load(ckpt_path, map_location=device)
# # # # # # #     model.load_state_dict(ckpt["model_state"])
# # # # # # #     print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}"
# # # # # # #           f"  (best val ADE = {ckpt.get('best_ade', float('nan')):.1f} km)")

# # # # # # #     test_dataset, test_loader = data_loader(
# # # # # # #         args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # # # # #     print(f"  test : {len(test_dataset)} sequences  ({len(test_loader)} batches)")

# # # # # # #     metrics = evaluate(model, test_loader, device)

# # # # # # #     print(f"\n  {'Metric':<20} {'Value (km)':>12}")
# # # # # # #     print(f"  {'-'*34}")
# # # # # # #     for key, val in metrics.items():
# # # # # # #         print(f"  {key:<20} {_fmt(val):>12}")

# # # # # # #     row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # # #            "split": "test",
# # # # # # #            "model_type": f"ST-Trans-{args.model_type}"}
# # # # # # #     row.update({k: _fmt(v) for k, v in metrics.items()})
# # # # # # #     save_metrics_csv(row, csv_path)
# # # # # # #     print(f"\n  Test metrics saved → {csv_path}")
# # # # # # #     print("=" * 70)
# # # # # # #     return metrics


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Args
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # def get_args():
# # # # # # #     p = argparse.ArgumentParser(
# # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
# # # # # # #         description="Train ST-Trans baseline (Faiaz et al. 2026)")

# # # # # # #     p.add_argument("--dataset_root", default="TCND_vn",  type=str)
# # # # # # #     p.add_argument("--obs_len",      default=8,          type=int)
# # # # # # #     p.add_argument("--pred_len",     default=12,         type=int)

# # # # # # #     # Model
# # # # # # #     p.add_argument("--model_type",   default="non_ar",   type=str,
# # # # # # #                    choices=["non_ar", "ar"])
# # # # # # #     p.add_argument("--d_model",      default=64,         type=int)
# # # # # # #     p.add_argument("--nhead",        default=4,          type=int)
# # # # # # #     p.add_argument("--num_enc_layers", default=1,        type=int)
# # # # # # #     p.add_argument("--num_dec_layers", default=3,        type=int)
# # # # # # #     p.add_argument("--dim_ff",       default=512,        type=int)
# # # # # # #     p.add_argument("--dropout",      default=0.1,        type=float)
# # # # # # #     p.add_argument("--unet_in_ch",   default=13,         type=int)

# # # # # # #     # Physics loss
# # # # # # #     p.add_argument("--lambda_speed", default=0.1,        type=float)
# # # # # # #     p.add_argument("--lambda_accel", default=0.01,       type=float)
# # # # # # #     p.add_argument("--w_mse",        default=0.05,       type=float)
# # # # # # #     p.add_argument("--v_max_kmh",    default=80.0,       type=float)

# # # # # # #     # Training
# # # # # # #     p.add_argument("--num_epochs",   default=1200,       type=int)
# # # # # # #     p.add_argument("--batch_size",   default=90,         type=int)
# # # # # # #     p.add_argument("--lr",           default=1e-3,       type=float)
# # # # # # #     p.add_argument("--weight_decay", default=1e-4,       type=float)
# # # # # # #     p.add_argument("--grad_clip",    default=0.5,        type=float)
# # # # # # #     p.add_argument("--patience",     default=100,        type=int)
# # # # # # #     p.add_argument("--min_epochs",   default=50,         type=int)
# # # # # # #     p.add_argument("--lr_patience",  default=20,         type=int)
# # # # # # #     p.add_argument("--lr_factor",    default=0.5,        type=float)
# # # # # # #     p.add_argument("--lr_min",       default=1e-6,       type=float)
# # # # # # #     p.add_argument("--val_freq",     default=5,          type=int)
# # # # # # #     p.add_argument("--val_subset",   default=600,        type=int)
# # # # # # #     p.add_argument("--num_workers",  default=2,          type=int)

# # # # # # #     # Test
# # # # # # #     p.add_argument("--test_at_end",  action="store_true",
# # # # # # #                    help="Đánh giá trên tập test sau khi training xong")

# # # # # # #     # I/O
# # # # # # #     p.add_argument("--output_dir",   default="runs/st_trans", type=str)
# # # # # # #     p.add_argument("--metrics_csv",  default="metrics.csv",   type=str)
# # # # # # #     p.add_argument("--gpu_num",      default="0",             type=str)

# # # # # # #     # DataLoader compat (cùng với train_paper_baseline)
# # # # # # #     p.add_argument("--delim",        default=" ")
# # # # # # #     p.add_argument("--skip",         default=1,   type=int)
# # # # # # #     p.add_argument("--min_ped",      default=1,   type=int)
# # # # # # #     p.add_argument("--threshold",    default=0.002, type=float)
# # # # # # #     p.add_argument("--other_modal",  default="gph")

# # # # # # #     return p.parse_args()


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  MAIN
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # def main(args):
# # # # # # #     if torch.cuda.is_available():
# # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# # # # # # #     best_ckpt   = os.path.join(args.output_dir, "best_model.pth")

# # # # # # #     print("=" * 70)
# # # # # # #     print(f"  ST-TRANS BASELINE  |  type={args.model_type.upper()}")
# # # # # # #     print(f"  Faiaz et al. (2026) Expert Systems With Applications 317, 131972")
# # # # # # #     print(f"  Encoder: PaperEncoder (FNO3D + Mamba + Env_net)  ← cùng với LSTM baseline")
# # # # # # #     print(f"  d_model={args.d_model}  nhead={args.nhead}  dim_ff={args.dim_ff}")
# # # # # # #     print(f"  λ_speed={args.lambda_speed}  λ_accel={args.lambda_accel}")
# # # # # # #     print(f"  Metrics: ADE / ATE / CTE @ 12h / 24h / 48h / 72h")
# # # # # # #     print("=" * 70)

# # # # # # #     # ── Data ── (cùng pipeline với train_paper_baseline) ──────────────────
# # # # # # #     train_dataset, train_loader = data_loader(
# # # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # # #     val_dataset, val_loader = data_loader(
# # # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # #     val_sub_loader = make_subset_loader(
# # # # # # #         val_dataset, args.val_subset, args.batch_size, seq_collate)

# # # # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # # # #     print(f"  val   : {len(val_dataset)} seq")

# # # # # # #     # ── Model ─────────────────────────────────────────────────────────────
# # # # # # #     if args.model_type == "non_ar":
# # # # # # #         model = STTrans(
# # # # # # #             obs_len        = args.obs_len,
# # # # # # #             pred_len       = args.pred_len,
# # # # # # #             unet_in_ch     = args.unet_in_ch,
# # # # # # #             d_model        = args.d_model,
# # # # # # #             nhead          = args.nhead,
# # # # # # #             num_enc_layers = args.num_enc_layers,
# # # # # # #             num_dec_layers = args.num_dec_layers,
# # # # # # #             dim_ff         = args.dim_ff,
# # # # # # #             dropout        = args.dropout,
# # # # # # #             lambda_speed   = args.lambda_speed,
# # # # # # #             lambda_accel   = args.lambda_accel,
# # # # # # #             w_mse          = args.w_mse,
# # # # # # #             v_max_kmh      = args.v_max_kmh,
# # # # # # #         ).to(device)
# # # # # # #     else:
# # # # # # #         model = STTransAR(
# # # # # # #             obs_len        = args.obs_len,
# # # # # # #             pred_len       = args.pred_len,
# # # # # # #             unet_in_ch     = args.unet_in_ch,
# # # # # # #             d_model        = args.d_model,
# # # # # # #             nhead          = args.nhead,
# # # # # # #             num_enc_layers = args.num_enc_layers,
# # # # # # #             dim_ff         = args.dim_ff,
# # # # # # #             dropout        = args.dropout,
# # # # # # #             lambda_speed   = args.lambda_speed,
# # # # # # #             lambda_accel   = args.lambda_accel,
# # # # # # #             w_mse          = args.w_mse,
# # # # # # #             v_max_kmh      = args.v_max_kmh,
# # # # # # #         ).to(device)

# # # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # #     print(f"  params : {n_params:,}")

# # # # # # #     # ── Optimizer + Scheduler ─────────────────────────────────────────────
# # # # # # #     optimizer = optim.AdamW(model.parameters(),
# # # # # # #                             lr=args.lr, weight_decay=args.weight_decay)
# # # # # # #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# # # # # # #         optimizer, mode="min", factor=args.lr_factor,
# # # # # # #         patience=args.lr_patience, min_lr=args.lr_min)

# # # # # # #     best_ade     = float("inf")
# # # # # # #     patience_cnt = 0
# # # # # # #     train_start  = time.perf_counter()

# # # # # # #     print("=" * 70)
# # # # # # #     print(f"  TRAINING  ({len(train_loader)} steps/epoch)")
# # # # # # #     print("=" * 70)

# # # # # # #     for epoch in range(args.num_epochs):
# # # # # # #         model.train()
# # # # # # #         sum_loss = 0.0
# # # # # # #         t0 = time.perf_counter()

# # # # # # #         for i, batch in enumerate(train_loader):
# # # # # # #             bl   = move(list(batch), device)   # full batch_list (ảnh + env + ...)
# # # # # # #             bd   = model.get_loss_breakdown(bl)
# # # # # # #             loss = bd["total"]

# # # # # # #             optimizer.zero_grad()
# # # # # # #             loss.backward()
# # # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # # # #             optimizer.step()

# # # # # # #             sum_loss += loss.item()

# # # # # # #             if i % 30 == 0:
# # # # # # #                 lr = optimizer.param_groups[0]["lr"]
# # # # # # #                 print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
# # # # # # #                       f"  loss={loss.item():.4f}"
# # # # # # #                       f"  dpe={bd.get('dpe', 0):.2f}km"
# # # # # # #                       f"  speed={bd.get('speed', 0):.4f}"
# # # # # # #                       f"  lr={lr:.2e}")

# # # # # # #         avg_train = sum_loss / len(train_loader)

# # # # # # #         # ── Val loss (DPE) ────────────────────────────────────────────────
# # # # # # #         model.eval()
# # # # # # #         val_dpe = 0.0
# # # # # # #         n_val   = 0
# # # # # # #         with torch.no_grad():
# # # # # # #             for batch in val_loader:
# # # # # # #                 bl_v    = move(list(batch), device)
# # # # # # #                 bd_v    = model.get_loss_breakdown(bl_v)
# # # # # # #                 val_dpe += bd_v["dpe"]
# # # # # # #                 n_val   += 1
# # # # # # #         avg_val_dpe = val_dpe / max(n_val, 1)

# # # # # # #         scheduler.step(avg_val_dpe)

# # # # # # #         ep_t = time.perf_counter() - t0
# # # # # # #         print(f"  Epoch {epoch:>4}  train_loss={avg_train:.4f}"
# # # # # # #               f"  val_dpe={avg_val_dpe:.2f}km  t={ep_t:.0f}s")

# # # # # # #         # ── ADE + ATE + CTE evaluation ────────────────────────────────────
# # # # # # #         if epoch % args.val_freq == 0:
# # # # # # #             r = evaluate(model, val_sub_loader, device)

# # # # # # #             ade12 = r.get("12h",         float("nan"))
# # # # # # #             ade24 = r.get("24h",         float("nan"))
# # # # # # #             ade48 = r.get("48h",         float("nan"))
# # # # # # #             ade72 = r.get("72h",         float("nan"))
# # # # # # #             ade   = r.get("ADE",         float("nan"))
# # # # # # #             ate12 = r.get("ATE_abs_12h", float("nan"))
# # # # # # #             cte12 = r.get("CTE_abs_12h", float("nan"))
# # # # # # #             ate72 = r.get("ATE_abs_72h", float("nan"))
# # # # # # #             cte72 = r.get("CTE_abs_72h", float("nan"))

# # # # # # #             t12 = "🎯" if ade12 < 50  else "❌"
# # # # # # #             t24 = "🎯" if ade24 < 100 else "❌"
# # # # # # #             t48 = "🎯" if ade48 < 200 else "❌"
# # # # # # #             t72 = "🎯" if ade72 < 300 else "❌"

# # # # # # #             print(f"  [VAL ep{epoch}]"
# # # # # # #                   f"  ADE={ade:.1f}"
# # # # # # #                   f"  12h={ade12:.0f}{t12}"
# # # # # # #                   f"  24h={ade24:.0f}{t24}"
# # # # # # #                   f"  48h={ade48:.0f}{t48}"
# # # # # # #                   f"  72h={ade72:.0f}{t72} km")
# # # # # # #             print(f"           "
# # # # # # #                   f"  ATE@12h={ate12:.1f}  CTE@12h={cte12:.1f}"
# # # # # # #                   f"  ATE@72h={ate72:.1f}  CTE@72h={cte72:.1f} km")

# # # # # # #             save_metrics_csv({
# # # # # # #                 "timestamp"      : datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # # #                 "split"          : "val",
# # # # # # #                 "epoch"          : epoch,
# # # # # # #                 "model_type"     : f"ST-Trans-{args.model_type}",
# # # # # # #                 "train_loss"     : _fmt(avg_train),
# # # # # # #                 "val_dpe_km"     : _fmt(avg_val_dpe),
# # # # # # #                 "ADE_km"         : _fmt(ade),
# # # # # # #                 "FDE_km"         : _fmt(r.get("FDE", float("nan"))),
# # # # # # #                 "12h_km"         : _fmt(ade12),
# # # # # # #                 "24h_km"         : _fmt(ade24),
# # # # # # #                 "48h_km"         : _fmt(ade48),
# # # # # # #                 "72h_km"         : _fmt(ade72),
# # # # # # #                 "ATE_abs_km"     : _fmt(r.get("ATE_abs", float("nan"))),
# # # # # # #                 "CTE_abs_km"     : _fmt(r.get("CTE_abs", float("nan"))),
# # # # # # #                 "ATE_abs_12h_km" : _fmt(ate12),
# # # # # # #                 "CTE_abs_12h_km" : _fmt(cte12),
# # # # # # #                 "ATE_abs_24h_km" : _fmt(r.get("ATE_abs_24h", float("nan"))),
# # # # # # #                 "CTE_abs_24h_km" : _fmt(r.get("CTE_abs_24h", float("nan"))),
# # # # # # #                 "ATE_abs_48h_km" : _fmt(r.get("ATE_abs_48h", float("nan"))),
# # # # # # #                 "CTE_abs_48h_km" : _fmt(r.get("CTE_abs_48h", float("nan"))),
# # # # # # #                 "ATE_abs_72h_km" : _fmt(ate72),
# # # # # # #                 "CTE_abs_72h_km" : _fmt(cte72),
# # # # # # #             }, metrics_csv)

# # # # # # #             if ade < best_ade:
# # # # # # #                 best_ade     = ade
# # # # # # #                 patience_cnt = 0
# # # # # # #                 torch.save({
# # # # # # #                     "epoch"      : epoch,
# # # # # # #                     "model_state": model.state_dict(),
# # # # # # #                     "best_ade"   : best_ade,
# # # # # # #                     "model_type" : args.model_type,
# # # # # # #                     "paper"      : "Faiaz et al. 2026",
# # # # # # #                 }, best_ckpt)
# # # # # # #                 print(f"  ✅ Best ADE {best_ade:.1f} km  (epoch {epoch})")
# # # # # # #             else:
# # # # # # #                 patience_cnt += args.val_freq
# # # # # # #                 print(f"  No improvement {patience_cnt}/{args.patience}"
# # # # # # #                       f"  (best={best_ade:.1f} km)")

# # # # # # #             if epoch >= args.min_epochs and patience_cnt >= args.patience:
# # # # # # #                 print(f"  ⛔ Early stop @ epoch {epoch}")
# # # # # # #                 break

# # # # # # #         if epoch % 100 == 0:
# # # # # # #             torch.save({
# # # # # # #                 "epoch"      : epoch,
# # # # # # #                 "model_state": model.state_dict(),
# # # # # # #                 "train_loss" : avg_train,
# # # # # # #                 "val_dpe"    : avg_val_dpe,
# # # # # # #             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

# # # # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # # # #     print("=" * 70)
# # # # # # #     print(f"  Model   : ST-Trans-{args.model_type.upper()}")
# # # # # # #     print(f"  Best ADE: {best_ade:.1f} km")
# # # # # # #     print(f"  Total   : {total_h:.2f}h")
# # # # # # #     print(f"  Metrics : {metrics_csv}")
# # # # # # #     print("=" * 70)

# # # # # # #     # ── Test evaluation ────────────────────────────────────────────────────
# # # # # # #     if args.test_at_end and os.path.exists(best_ckpt):
# # # # # # #         from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # #         run_test_evaluation(model, best_ckpt, args, device,
# # # # # # #                             seq_collate, metrics_csv)


# # # # # # # if __name__ == "__main__":
# # # # # # #     args = get_args()
# # # # # # #     np.random.seed(42)
# # # # # # #     torch.manual_seed(42)
# # # # # # #     if torch.cuda.is_available():
# # # # # # #         torch.cuda.manual_seed_all(42)
# # # # # # #     main(args)

# # # # # # """
# # # # # # scripts/train_st_trans_v2.py  ── ST-Trans v2 Training
# # # # # # =======================================================
# # # # # # Mở rộng từ train_st_trans.py với:
# # # # # #   ✅ Easy/Hard loss split (easy = ST-Trans gốc, hard = extended)
# # # # # #   ✅ Tự động tính easy/hard threshold từ train set trước khi train
# # # # # #   ✅ Log easy_ADE và hard_ADE riêng biệt mỗi lần eval
# # # # # #   ✅ Monitor alpha (gate) và step_weights
# # # # # #   ✅ Tương thích 100% với DataLoader của train_st_trans.py

# # # # # # CHẠY:
# # # # # #   python scripts/train_st_trans_v2.py \\
# # # # # #     --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \\
# # # # # #     --output_dir   runs/st_trans_v2 \\
# # # # # #     --num_epochs   600 \\
# # # # # #     --batch_size   90 \\
# # # # # #     --w_heading    0.3 \\
# # # # # #     --w_recurv     0.05 \\
# # # # # #     --test_at_end

# # # # # # HYPERPARAMETERS:
# # # # # #   Giữ nguyên mọi thứ từ train_st_trans.py, thêm:
# # # # # #     --w_heading         weight L_heading (hard only), default 0.3
# # # # # #     --w_recurv          weight L_recurv  (hard only), default 0.05
# # # # # #     --w_gate_reg        weight gate regularization, default 0.01
# # # # # #     --step_weight_slope slope của step weights, default 0.1
# # # # # #     --recurv_threshold  ngưỡng góc để label recurvature (degrees), default 45
# # # # # #     --threshold_pct     percentile để tính easy/hard threshold, default 70
# # # # # # """
# # # # # # from __future__ import annotations

# # # # # # import sys, os
# # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # import argparse
# # # # # # import time
# # # # # # import random
# # # # # # import csv
# # # # # # from datetime import datetime

# # # # # # import numpy as np
# # # # # # import torch
# # # # # # import torch.optim as optim
# # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # from Model.data.loader_training import data_loader
# # # # # # from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # from Model.st_trans_model import (
# # # # # #     STTransV2, build_st_trans_v2,
# # # # # #     classify_hard_obs, compute_hard_thresholds,
# # # # # # )
# # # # # # from Model.paper_baseline_model import (
# # # # # #     haversine_km, _norm_to_deg, _ate_cte_tensors, HORIZON_STEPS,
# # # # # # )


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Helpers (copy từ train_st_trans.py, không thay đổi)
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def move(batch, device):
# # # # # #     out = list(batch)
# # # # # #     for i, x in enumerate(out):
# # # # # #         if torch.is_tensor(x):
# # # # # #             out[i] = x.to(device)
# # # # # #         elif isinstance(x, dict):
# # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # #                       for k, v in x.items()}
# # # # # #     return out


# # # # # # def make_subset_loader(dataset, subset_size, batch_size, seed=42):
# # # # # #     n   = len(dataset)
# # # # # #     rng = random.Random(seed)
# # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # #     return DataLoader(Subset(dataset, idx),
# # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # #                       collate_fn=seq_collate, num_workers=0, drop_last=False)


# # # # # # def save_metrics_csv(row: dict, csv_path: str):
# # # # # #     write_hdr = not os.path.exists(csv_path)
# # # # # #     with open(csv_path, "a", newline="") as fh:
# # # # # #         w = csv.DictWriter(fh, fieldnames=list(row.keys()))
# # # # # #         if write_hdr:
# # # # # #             w.writeheader()
# # # # # #         w.writerow(row)


# # # # # # def _fmt(v) -> str:
# # # # # #     if isinstance(v, float):
# # # # # #         return "nan" if np.isnan(v) else f"{v:.4f}"
# # # # # #     return str(v)


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Evaluation — tách easy/hard ADE
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # @torch.no_grad()
# # # # # # def evaluate(model: STTransV2, loader, device) -> dict:
# # # # # #     """
# # # # # #     Đánh giá model. Tách ADE thành easy_ADE và hard_ADE.
# # # # # #     Dùng cùng threshold đã compute từ train set (model.threshold_curv/spd).
# # # # # #     """
# # # # # #     model.eval()

# # # # # #     all_ade, all_fde    = [], []
# # # # # #     easy_ade_list       = []
# # # # # #     hard_ade_list       = []
# # # # # #     ade_buf = {h: [] for h in HORIZON_STEPS}
# # # # # #     ate_buf = {h: [] for h in HORIZON_STEPS}
# # # # # #     cte_buf = {h: [] for h in HORIZON_STEPS}

# # # # # #     for batch in loader:
# # # # # #         bl         = move(list(batch), device)
# # # # # #         obs_traj   = bl[0]   # [T_obs, B, 2]
# # # # # #         gt         = bl[1]   # [T_pred, B, 2]

# # # # # #         pred, _, _ = model.sample(bl)
# # # # # #         T          = min(pred.shape[0], gt.shape[0])
# # # # # #         pred_d     = _norm_to_deg(pred[:T])
# # # # # #         gt_d       = _norm_to_deg(gt[:T])
# # # # # #         dist       = haversine_km(pred_d, gt_d)         # [T, B]
# # # # # #         ate, cte   = _ate_cte_tensors(pred[:T], gt[:T]) # [T, B] each

# # # # # #         ade_per_sample = dist.mean(0)   # [B]
# # # # # #         all_ade.extend(ade_per_sample.tolist())
# # # # # #         all_fde.extend(dist[-1].tolist())

# # # # # #         # Easy/hard split cho monitoring
# # # # # #         is_hard = classify_hard_obs(
# # # # # #             obs_traj, model.threshold_curv, model.threshold_spd)
# # # # # #         is_easy = ~is_hard
# # # # # #         if is_easy.any():
# # # # # #             easy_ade_list.extend(ade_per_sample[is_easy].tolist())
# # # # # #         if is_hard.any():
# # # # # #             hard_ade_list.extend(ade_per_sample[is_hard].tolist())

# # # # # #         for h, s in HORIZON_STEPS.items():
# # # # # #             if s < T:
# # # # # #                 ade_buf[h].extend(dist[s].tolist())
# # # # # #                 ate_buf[h].extend(ate[s].abs().tolist())
# # # # # #                 cte_buf[h].extend(cte[s].abs().tolist())

# # # # # #     def _mean(lst):
# # # # # #         return float(np.mean(lst)) if lst else float("nan")

# # # # # #     result = dict(
# # # # # #         ADE      = _mean(all_ade),
# # # # # #         FDE      = _mean(all_fde),
# # # # # #         easy_ADE = _mean(easy_ade_list),
# # # # # #         hard_ADE = _mean(hard_ade_list),
# # # # # #         n_easy   = len(easy_ade_list),
# # # # # #         n_hard   = len(hard_ade_list),
# # # # # #     )
# # # # # #     for h in HORIZON_STEPS:
# # # # # #         result[f"{h}h"]         = _mean(ade_buf[h])
# # # # # #         result[f"ATE_abs_{h}h"] = _mean(ate_buf[h])
# # # # # #         result[f"CTE_abs_{h}h"] = _mean(cte_buf[h])

# # # # # #     return result


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Test evaluation
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def run_test_evaluation(model, ckpt_path, args, device, csv_path):
# # # # # #     print("\n" + "=" * 70)
# # # # # #     print("  TEST SET EVALUATION  (ST-Trans v2)")
# # # # # #     print("=" * 70)

# # # # # #     ckpt = torch.load(ckpt_path, map_location=device)
# # # # # #     model.load_state_dict(ckpt["model_state"])
# # # # # #     # Restore thresholds
# # # # # #     model.threshold_curv = ckpt.get("threshold_curv", model.threshold_curv)
# # # # # #     model.threshold_spd  = ckpt.get("threshold_spd",  model.threshold_spd)
# # # # # #     print(f"  Loaded checkpoint epoch {ckpt.get('epoch','?')}"
# # # # # #           f"  best_val_ADE={ckpt.get('best_ade',float('nan')):.1f} km")

# # # # # #     _, test_loader = data_loader(
# # # # # #         args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # # # #     metrics = evaluate(model, test_loader, device)

# # # # # #     print(f"\n  {'Metric':<22} {'Value':>10}")
# # # # # #     print(f"  {'-'*34}")
# # # # # #     for k, v in metrics.items():
# # # # # #         print(f"  {k:<22} {_fmt(v):>10}")

# # # # # #     row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # #            "split": "test", "model": "ST-Trans-v2"}
# # # # # #     row.update({k: _fmt(v) for k, v in metrics.items()})
# # # # # #     save_metrics_csv(row, csv_path)
# # # # # #     print(f"\n  Saved → {csv_path}")
# # # # # #     print("=" * 70)
# # # # # #     return metrics


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Args
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def get_args():
# # # # # #     p = argparse.ArgumentParser(
# # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
# # # # # #         description="Train ST-Trans v2 (Easy/Hard split + Physics Gate)")

# # # # # #     # Data
# # # # # #     p.add_argument("--dataset_root",     default="TCND_vn",   type=str)
# # # # # #     p.add_argument("--obs_len",          default=8,           type=int)
# # # # # #     p.add_argument("--pred_len",         default=12,          type=int)

# # # # # #     # Model architecture (giống STTrans gốc)
# # # # # #     p.add_argument("--d_model",          default=64,          type=int)
# # # # # #     p.add_argument("--nhead",            default=4,           type=int)
# # # # # #     p.add_argument("--num_enc_layers",   default=1,           type=int)
# # # # # #     p.add_argument("--num_dec_layers",   default=3,           type=int)
# # # # # #     p.add_argument("--dim_ff",           default=512,         type=int)
# # # # # #     p.add_argument("--dropout",          default=0.1,         type=float)
# # # # # #     p.add_argument("--unet_in_ch",       default=13,          type=int)

# # # # # #     # Easy loss weights (ST-Trans gốc — không nên thay đổi)
# # # # # #     p.add_argument("--lambda_speed",     default=0.1,         type=float)
# # # # # #     p.add_argument("--lambda_accel",     default=0.01,        type=float)
# # # # # #     p.add_argument("--w_mse",            default=0.05,        type=float)
# # # # # #     p.add_argument("--v_max_kmh",        default=80.0,        type=float)

# # # # # #     # Hard loss extras (mới)
# # # # # #     p.add_argument("--w_heading",        default=0.3,         type=float,
# # # # # #                    help="Weight L_heading (hard samples only)")
# # # # # #     p.add_argument("--w_recurv",         default=0.05,        type=float,
# # # # # #                    help="Weight L_recurv auxiliary (hard samples only)")
# # # # # #     p.add_argument("--w_gate_reg",       default=0.01,        type=float,
# # # # # #                    help="Weight gate regularization")
# # # # # #     p.add_argument("--step_weight_slope",default=0.1,         type=float,
# # # # # #                    help="Initial slope of learnable step weights")
# # # # # #     p.add_argument("--recurv_threshold", default=45.0,        type=float,
# # # # # #                    help="Angle threshold (degrees) for recurvature label")

# # # # # #     # Easy/hard threshold
# # # # # #     p.add_argument("--threshold_pct",    default=70.0,        type=float,
# # # # # #                    help="Percentile for easy/hard threshold computation")
# # # # # #     p.add_argument("--gate_hidden",      default=32,          type=int)
# # # # # #     p.add_argument("--recurv_hidden",    default=64,          type=int)

# # # # # #     # Training (giống ST-Trans gốc)
# # # # # #     p.add_argument("--num_epochs",       default=600,         type=int)
# # # # # #     p.add_argument("--batch_size",       default=90,          type=int)
# # # # # #     p.add_argument("--lr",               default=1e-3,        type=float)
# # # # # #     p.add_argument("--weight_decay",     default=1e-4,        type=float)
# # # # # #     p.add_argument("--grad_clip",        default=0.5,         type=float)
# # # # # #     p.add_argument("--patience",         default=100,         type=int)
# # # # # #     p.add_argument("--min_epochs",       default=50,          type=int)
# # # # # #     p.add_argument("--lr_patience",      default=20,          type=int)
# # # # # #     p.add_argument("--lr_factor",        default=0.5,         type=float)
# # # # # #     p.add_argument("--lr_min",           default=1e-6,        type=float)
# # # # # #     p.add_argument("--val_freq",         default=5,           type=int)
# # # # # #     p.add_argument("--val_subset",       default=600,         type=int)
# # # # # #     p.add_argument("--num_workers",      default=2,           type=int)

# # # # # #     # Test
# # # # # #     p.add_argument("--test_at_end",      action="store_true")

# # # # # #     # I/O
# # # # # #     p.add_argument("--output_dir",       default="runs/st_trans_v2", type=str)
# # # # # #     p.add_argument("--metrics_csv",      default="metrics.csv",      type=str)
# # # # # #     p.add_argument("--gpu_num",          default="0",                type=str)
# # # # # #     p.add_argument("--resume",           default=None,               type=str,
# # # # # #                    help="Path to checkpoint để resume training")

# # # # # #     # DataLoader compat
# # # # # #     p.add_argument("--delim",        default=" ")
# # # # # #     p.add_argument("--skip",         default=1,     type=int)
# # # # # #     p.add_argument("--min_ped",      default=1,     type=int)
# # # # # #     p.add_argument("--threshold",    default=0.002, type=float)
# # # # # #     p.add_argument("--other_modal",  default="gph")

# # # # # #     return p.parse_args()


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  MAIN
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def main(args):
# # # # # #     if torch.cuda.is_available():
# # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# # # # # #     best_ckpt   = os.path.join(args.output_dir, "best_model.pth")

# # # # # #     print("=" * 70)
# # # # # #     print("  ST-TRANS v2  |  Easy/Hard Split + Physics Steering Gate")
# # # # # #     print(f"  Easy loss : ST-Trans gốc (unchanged)")
# # # # # #     print(f"  Hard loss : DPE_weighted + heading + recurv + gate_reg")
# # # # # #     print(f"  w_heading={args.w_heading}  w_recurv={args.w_recurv}"
# # # # # #           f"  slope={args.step_weight_slope}")
# # # # # #     print("=" * 70)

# # # # # #     # ── Data ──────────────────────────────────────────────────────────────
# # # # # #     train_dataset, train_loader = data_loader(
# # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # #     val_dataset, val_loader = data_loader(
# # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)
# # # # # #     val_sub_loader = make_subset_loader(
# # # # # #         val_dataset, args.val_subset, args.batch_size)

# # # # # #     print(f"  train: {len(train_dataset)} seq  val: {len(val_dataset)} seq")

# # # # # #     # ── Compute easy/hard threshold từ train set ──────────────────────────
# # # # # #     print(f"\n  Computing easy/hard threshold (p{args.threshold_pct:.0f}"
# # # # # #           f" from train set)...")
# # # # # #     t0 = time.perf_counter()
# # # # # #     threshold_curv, threshold_spd = compute_hard_thresholds(
# # # # # #         train_loader, device, percentile=args.threshold_pct)
# # # # # #     print(f"  threshold_curv = {threshold_curv:.3f}°"
# # # # # #           f"  threshold_spd = {threshold_spd:.4f}"
# # # # # #           f"  ({time.perf_counter()-t0:.1f}s)")

# # # # # #     # Estimate hard fraction để thông báo
# # # # # #     n_hard_est, n_total = 0, 0
# # # # # #     with torch.no_grad():
# # # # # #         for batch in train_loader:
# # # # # #             obs = batch[0].to(device)
# # # # # #             is_hard = classify_hard_obs(obs, threshold_curv, threshold_spd)
# # # # # #             n_hard_est += is_hard.sum().item()
# # # # # #             n_total    += obs.shape[1]
# # # # # #     print(f"  Hard fraction: {n_hard_est}/{n_total}"
# # # # # #           f" = {100*n_hard_est/max(n_total,1):.1f}%  "
# # # # # #           f"(target ~30%, p{args.threshold_pct:.0f})")

# # # # # #     # ── Model ─────────────────────────────────────────────────────────────
# # # # # #     model = build_st_trans_v2(args, threshold_curv, threshold_spd).to(device)

# # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # #     print(f"\n  Params total : {n_params:,}")
# # # # # #     print(f"  Encoder      : {sum(p.numel() for p in model.encoder.parameters()):,}")
# # # # # #     print(f"  Gate         : {sum(p.numel() for p in model.steering_gate.parameters()):,}")
# # # # # #     print(f"  Recurv head  : {sum(p.numel() for p in model.recurv_head.parameters()):,}")

# # # # # #     # ── Resume nếu có ─────────────────────────────────────────────────────
# # # # # #     start_epoch  = 0
# # # # # #     best_ade     = float("inf")
# # # # # #     patience_cnt = 0

# # # # # #     if args.resume and os.path.exists(args.resume):
# # # # # #         ckpt = torch.load(args.resume, map_location=device)
# # # # # #         model.load_state_dict(ckpt["model_state"])
# # # # # #         start_epoch  = ckpt.get("epoch", 0) + 1
# # # # # #         best_ade     = ckpt.get("best_ade", float("inf"))
# # # # # #         patience_cnt = ckpt.get("patience_cnt", 0)
# # # # # #         # Restore thresholds từ checkpoint
# # # # # #         if "threshold_curv" in ckpt:
# # # # # #             model.set_thresholds(ckpt["threshold_curv"], ckpt["threshold_spd"])
# # # # # #         print(f"\n  ↩ Resumed from {args.resume}"
# # # # # #               f"  (epoch {start_epoch}, best_ADE={best_ade:.1f})")

# # # # # #     # ── Optimizer + Scheduler ─────────────────────────────────────────────
# # # # # #     optimizer = optim.AdamW(
# # # # # #         model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# # # # # #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# # # # # #         optimizer, mode="min", factor=args.lr_factor,
# # # # # #         patience=args.lr_patience, min_lr=args.lr_min)

# # # # # #     train_start = time.perf_counter()
# # # # # #     print("=" * 70)
# # # # # #     print(f"  TRAINING  ({len(train_loader)} steps/epoch,"
# # # # # #           f" max {args.num_epochs} epochs)")
# # # # # #     print("=" * 70)

# # # # # #     for epoch in range(start_epoch, args.num_epochs):
# # # # # #         model.train()
# # # # # #         sum_loss = 0.0
# # # # # #         sum_easy_dpe = sum_hard_dpe = 0.0
# # # # # #         n_easy_total = n_hard_total = 0
# # # # # #         t0 = time.perf_counter()

# # # # # #         for i, batch in enumerate(train_loader):
# # # # # #             bl  = move(list(batch), device)
# # # # # #             bd  = model.get_loss_breakdown(bl)
# # # # # #             loss = bd["total"]

# # # # # #             optimizer.zero_grad()
# # # # # #             loss.backward()
# # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # # #             optimizer.step()

# # # # # #             sum_loss += loss.item()
# # # # # #             ne = bd.get("n_easy", 0)
# # # # # #             nh = bd.get("n_hard", 0)
# # # # # #             n_easy_total += ne
# # # # # #             n_hard_total += nh
# # # # # #             if ne > 0:
# # # # # #                 sum_easy_dpe += bd.get("easy_dpe", 0)
# # # # # #             if nh > 0:
# # # # # #                 sum_hard_dpe += bd.get("hard_dpe", 0)

# # # # # #             if i % 30 == 0:
# # # # # #                 lr    = optimizer.param_groups[0]["lr"]
# # # # # #                 alpha = bd.get("alpha_mean", float("nan"))
# # # # # #                 sw72  = bd.get("step_w_72h", float("nan"))
# # # # # #                 head  = bd.get("hard_heading", float("nan"))
# # # # # #                 print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
# # # # # #                       f"  loss={loss.item():.4f}"
# # # # # #                       f"  easy_dpe={bd.get('easy_dpe',0):.1f}"
# # # # # #                       f"  hard_dpe={bd.get('hard_dpe',0):.1f}"
# # # # # #                       f"  heading={head:.3f}"
# # # # # #                       f"  alpha={alpha:.3f}"
# # # # # #                       f"  sw72={sw72:.2f}"
# # # # # #                       f"  n_e={ne}/n_h={nh}"
# # # # # #                       f"  lr={lr:.2e}")

# # # # # #         avg_loss     = sum_loss / len(train_loader)
# # # # # #         avg_easy_dpe = sum_easy_dpe / max(n_easy_total // 90, 1)
# # # # # #         avg_hard_dpe = sum_hard_dpe / max(n_hard_total // 90, 1)

# # # # # #         # ── Val DPE (cho scheduler) ────────────────────────────────────────
# # # # # #         model.eval()
# # # # # #         val_dpe_sum = 0.0
# # # # # #         n_val = 0
# # # # # #         with torch.no_grad():
# # # # # #             for batch in val_loader:
# # # # # #                 bl_v    = move(list(batch), device)
# # # # # #                 bd_v    = model.get_loss_breakdown(bl_v)
# # # # # #                 val_dpe_sum += bd_v.get("easy_dpe", 0) + bd_v.get("hard_dpe", 0)
# # # # # #                 n_val   += 1
# # # # # #         avg_val_dpe = val_dpe_sum / max(n_val, 1)

# # # # # #         scheduler.step(avg_val_dpe)

# # # # # #         ep_t = time.perf_counter() - t0
# # # # # #         sw72 = model.step_weights[-1].item()
# # # # # #         print(f"  Epoch {epoch:>4}"
# # # # # #               f"  loss={avg_loss:.4f}"
# # # # # #               f"  val_dpe={avg_val_dpe:.2f}"
# # # # # #               f"  easy_dpe≈{avg_easy_dpe:.1f}"
# # # # # #               f"  hard_dpe≈{avg_hard_dpe:.1f}"
# # # # # #               f"  sw72={sw72:.3f}"
# # # # # #               f"  t={ep_t:.0f}s")

# # # # # #         # ── ADE evaluation + easy/hard split ──────────────────────────────
# # # # # #         if epoch % args.val_freq == 0:
# # # # # #             r = evaluate(model, val_sub_loader, device)

# # # # # #             ade     = r.get("ADE",         float("nan"))
# # # # # #             easy_ad = r.get("easy_ADE",    float("nan"))
# # # # # #             hard_ad = r.get("hard_ADE",    float("nan"))
# # # # # #             ade72   = r.get("72h",         float("nan"))
# # # # # #             ate72   = r.get("ATE_abs_72h", float("nan"))
# # # # # #             cte72   = r.get("CTE_abs_72h", float("nan"))
# # # # # #             n_e     = r.get("n_easy",      0)
# # # # # #             n_h     = r.get("n_hard",      0)

# # # # # #             print(f"\n  ╔═ VAL ep{epoch}")
# # # # # #             print(f"  ║  ADE={ade:.1f}  easy={easy_ad:.1f}({n_e})"
# # # # # #                   f"  hard={hard_ad:.1f}({n_h})")
# # # # # #             print(f"  ║  72h={ade72:.1f}  ATE@72h={ate72:.1f}"
# # # # # #                   f"  CTE@72h={cte72:.1f}")
# # # # # #             print(f"  ╚═ step_w_72h={sw72:.3f}  threshold_curv={model.threshold_curv:.2f}\n")

# # # # # #             save_metrics_csv({
# # # # # #                 "timestamp"      : datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # #                 "split"          : "val",
# # # # # #                 "epoch"          : epoch,
# # # # # #                 "model"          : "ST-Trans-v2",
# # # # # #                 "train_loss"     : _fmt(avg_loss),
# # # # # #                 "val_dpe"        : _fmt(avg_val_dpe),
# # # # # #                 "ADE"            : _fmt(ade),
# # # # # #                 "easy_ADE"       : _fmt(easy_ad),
# # # # # #                 "hard_ADE"       : _fmt(hard_ad),
# # # # # #                 "n_easy"         : n_e,
# # # # # #                 "n_hard"         : n_h,
# # # # # #                 "12h"            : _fmt(r.get("12h",         float("nan"))),
# # # # # #                 "24h"            : _fmt(r.get("24h",         float("nan"))),
# # # # # #                 "48h"            : _fmt(r.get("48h",         float("nan"))),
# # # # # #                 "72h"            : _fmt(ade72),
# # # # # #                 "ATE_abs_72h"    : _fmt(ate72),
# # # # # #                 "CTE_abs_72h"    : _fmt(cte72),
# # # # # #                 "step_w_72h"     : _fmt(sw72),
# # # # # #                 "threshold_curv" : _fmt(model.threshold_curv),
# # # # # #                 "threshold_spd"  : _fmt(model.threshold_spd),
# # # # # #             }, metrics_csv)

# # # # # #             if ade < best_ade:
# # # # # #                 best_ade     = ade
# # # # # #                 patience_cnt = 0
# # # # # #                 torch.save({
# # # # # #                     "epoch"          : epoch,
# # # # # #                     "model_state"    : model.state_dict(),
# # # # # #                     "best_ade"       : best_ade,
# # # # # #                     "threshold_curv" : model.threshold_curv,
# # # # # #                     "threshold_spd"  : model.threshold_spd,
# # # # # #                     "step_weights"   : model.step_weights.detach().cpu().tolist(),
# # # # # #                 }, best_ckpt)
# # # # # #                 print(f"  ✅ Best ADE = {best_ade:.1f} km  (epoch {epoch})")
# # # # # #             else:
# # # # # #                 patience_cnt += args.val_freq
# # # # # #                 print(f"  No improve {patience_cnt}/{args.patience}"
# # # # # #                       f"  (best={best_ade:.1f})")

# # # # # #             if epoch >= args.min_epochs and patience_cnt >= args.patience:
# # # # # #                 print(f"\n  ⛔ Early stop @ epoch {epoch}")
# # # # # #                 break

# # # # # #         # Periodic checkpoint
# # # # # #         if epoch % 50 == 0 and epoch > 0:
# # # # # #             torch.save({
# # # # # #                 "epoch"          : epoch,
# # # # # #                 "model_state"    : model.state_dict(),
# # # # # #                 "train_loss"     : avg_loss,
# # # # # #                 "val_dpe"        : avg_val_dpe,
# # # # # #                 "best_ade"       : best_ade,
# # # # # #                 "patience_cnt"   : patience_cnt,
# # # # # #                 "threshold_curv" : model.threshold_curv,
# # # # # #                 "threshold_spd"  : model.threshold_spd,
# # # # # #             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

# # # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # # #     print("=" * 70)
# # # # # #     print(f"  Model    : ST-Trans-v2")
# # # # # #     print(f"  Best ADE : {best_ade:.1f} km")
# # # # # #     print(f"  Total    : {total_h:.2f}h")
# # # # # #     print("=" * 70)

# # # # # #     if args.test_at_end and os.path.exists(best_ckpt):
# # # # # #         run_test_evaluation(model, best_ckpt, args, device, metrics_csv)


# # # # # # if __name__ == "__main__":
# # # # # #     args = get_args()
# # # # # #     np.random.seed(42)
# # # # # #     torch.manual_seed(42)
# # # # # #     if torch.cuda.is_available():
# # # # # #         torch.cuda.manual_seed_all(42)
# # # # # #     main(args)

# # # # # """
# # # # # scripts/train_st_trans_v2.py  ── ST-Trans v2 Training (GradNorm + Easy/Hard Split)
# # # # # ====================================================================================
# # # # # Chạy:
# # # # #   python train_st_trans_v2.py \
# # # # #     --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# # # # #     --output_dir   /kaggle/working/runs/st_trans_v2 \
# # # # #     --num_epochs   600 \
# # # # #     --batch_size   32 \
# # # # #     --w_heading    0.3 \
# # # # #     --w_recurv     0.05 \
# # # # #     --step_weight_slope 0.1 \
# # # # #     --threshold_pct 70 \
# # # # #     --val_freq     5 \
# # # # #     --val_subset   600 \
# # # # #     --test_at_end \
# # # # #     --gpu_num      0
# # # # # """
# # # # # from __future__ import annotations

# # # # # import sys
# # # # # import os
# # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # import argparse
# # # # # import time
# # # # # import random
# # # # # import csv
# # # # # import math
# # # # # from datetime import datetime

# # # # # import numpy as np
# # # # # import torch
# # # # # import torch.optim as optim
# # # # # from torch.utils.data import DataLoader, Subset

# # # # # from Model.data.loader_training import data_loader
# # # # # from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # from Model.st_trans_model import (
# # # # #     STTransV2,
# # # # #     build_st_trans_v2,
# # # # #     classify_hard_obs,
# # # # #     compute_hard_thresholds,
# # # # # )
# # # # # from Model.paper_baseline_model import (
# # # # #     haversine_km,
# # # # #     _norm_to_deg,
# # # # #     _ate_cte_tensors,
# # # # #     HORIZON_STEPS,
# # # # # )


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Helpers
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def move(batch, device):
# # # # #     out = list(batch)
# # # # #     for i, x in enumerate(out):
# # # # #         if torch.is_tensor(x):
# # # # #             out[i] = x.to(device)
# # # # #         elif isinstance(x, dict):
# # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # #                       for k, v in x.items()}
# # # # #     return out


# # # # # def make_subset_loader(dataset, subset_size, batch_size, seed=42):
# # # # #     n   = len(dataset)
# # # # #     rng = random.Random(seed)
# # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # #     return DataLoader(
# # # # #         Subset(dataset, idx),
# # # # #         batch_size=batch_size,
# # # # #         shuffle=False,
# # # # #         collate_fn=seq_collate,
# # # # #         num_workers=0,
# # # # #         drop_last=False,
# # # # #     )


# # # # # def save_metrics_csv(row: dict, csv_path: str):
# # # # #     write_hdr = not os.path.exists(csv_path)
# # # # #     with open(csv_path, "a", newline="") as fh:
# # # # #         w = csv.DictWriter(fh, fieldnames=list(row.keys()))
# # # # #         if write_hdr:
# # # # #             w.writeheader()
# # # # #         w.writerow(row)


# # # # # def _fmt(v) -> str:
# # # # #     if isinstance(v, float):
# # # # #         return "nan" if math.isnan(v) else f"{v:.4f}"
# # # # #     return str(v)


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Evaluation
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # @torch.no_grad()
# # # # # def evaluate(model: STTransV2, loader, device) -> dict:
# # # # #     model.eval()

# # # # #     all_ade, all_fde = [], []
# # # # #     easy_ade_list    = []
# # # # #     hard_ade_list    = []
# # # # #     ade_buf = {h: [] for h in HORIZON_STEPS}
# # # # #     ate_buf = {h: [] for h in HORIZON_STEPS}
# # # # #     cte_buf = {h: [] for h in HORIZON_STEPS}

# # # # #     for batch in loader:
# # # # #         bl       = move(list(batch), device)
# # # # #         obs_traj = bl[0]   # [T_obs, B, 2]
# # # # #         gt       = bl[1]   # [T_pred, B, 2]

# # # # #         pred, _, _ = model.sample(bl)
# # # # #         T          = min(pred.shape[0], gt.shape[0])
# # # # #         pred_d     = _norm_to_deg(pred[:T])
# # # # #         gt_d       = _norm_to_deg(gt[:T])
# # # # #         dist       = haversine_km(pred_d, gt_d)          # [T, B]
# # # # #         ate, cte   = _ate_cte_tensors(pred[:T], gt[:T])  # [T, B]

# # # # #         ade_per_sample = dist.mean(0)   # [B]
# # # # #         all_ade.extend(ade_per_sample.tolist())
# # # # #         all_fde.extend(dist[-1].tolist())

# # # # #         is_hard = classify_hard_obs(
# # # # #             obs_traj, model.threshold_curv, model.threshold_spd)
# # # # #         is_easy = ~is_hard
# # # # #         if is_easy.any():
# # # # #             easy_ade_list.extend(ade_per_sample[is_easy].tolist())
# # # # #         if is_hard.any():
# # # # #             hard_ade_list.extend(ade_per_sample[is_hard].tolist())

# # # # #         for h, s in HORIZON_STEPS.items():
# # # # #             if s < T:
# # # # #                 ade_buf[h].extend(dist[s].tolist())
# # # # #                 ate_buf[h].extend(ate[s].abs().tolist())
# # # # #                 cte_buf[h].extend(cte[s].abs().tolist())

# # # # #     def _mean(lst):
# # # # #         return float(np.mean(lst)) if lst else float("nan")

# # # # #     result = dict(
# # # # #         ADE      = _mean(all_ade),
# # # # #         FDE      = _mean(all_fde),
# # # # #         easy_ADE = _mean(easy_ade_list),
# # # # #         hard_ADE = _mean(hard_ade_list),
# # # # #         n_easy   = len(easy_ade_list),
# # # # #         n_hard   = len(hard_ade_list),
# # # # #     )
# # # # #     for h in HORIZON_STEPS:
# # # # #         result[f"{h}h"]         = _mean(ade_buf[h])
# # # # #         result[f"ATE_abs_{h}h"] = _mean(ate_buf[h])
# # # # #         result[f"CTE_abs_{h}h"] = _mean(cte_buf[h])

# # # # #     return result


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Test evaluation
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def run_test_evaluation(model, ckpt_path, args, device, csv_path):
# # # # #     print("\n" + "=" * 70)
# # # # #     print("  TEST SET EVALUATION  (ST-Trans v2)")
# # # # #     print("=" * 70)

# # # # #     ckpt = torch.load(ckpt_path, map_location=device)
# # # # #     model.load_state_dict(ckpt["model_state"])
# # # # #     model.threshold_curv = ckpt.get("threshold_curv", model.threshold_curv)
# # # # #     model.threshold_spd  = ckpt.get("threshold_spd",  model.threshold_spd)
# # # # #     print(f"  Loaded checkpoint epoch {ckpt.get('epoch', '?')}"
# # # # #           f"  best_val_ADE={ckpt.get('best_ade', float('nan')):.4f} km")

# # # # #     _, test_loader = data_loader(
# # # # #         args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # # #     metrics = evaluate(model, test_loader, device)

# # # # #     print(f"\n  {'Metric':<22} {'Value':>10}")
# # # # #     print(f"  {'-'*34}")
# # # # #     for k, v in metrics.items():
# # # # #         print(f"  {k:<22} {_fmt(v):>10}")

# # # # #     row = {
# # # # #         "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # #         "split": "test",
# # # # #         "model": "ST-Trans-v2",
# # # # #     }
# # # # #     row.update({k: _fmt(v) for k, v in metrics.items()})
# # # # #     save_metrics_csv(row, csv_path)
# # # # #     print(f"\n  Saved → {csv_path}")
# # # # #     print("=" * 70)
# # # # #     return metrics


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Args
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def get_args():
# # # # #     p = argparse.ArgumentParser(
# # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
# # # # #         description="Train ST-Trans v2 (GradNorm + Easy/Hard split)",
# # # # #     )

# # # # #     # Data
# # # # #     p.add_argument("--dataset_root",      default="TCND_vn",   type=str)
# # # # #     p.add_argument("--obs_len",           default=8,           type=int)
# # # # #     p.add_argument("--pred_len",          default=12,          type=int)

# # # # #     # Model architecture
# # # # #     p.add_argument("--d_model",           default=64,          type=int)
# # # # #     p.add_argument("--nhead",             default=4,           type=int)
# # # # #     p.add_argument("--num_enc_layers",    default=1,           type=int)
# # # # #     p.add_argument("--num_dec_layers",    default=3,           type=int)
# # # # #     p.add_argument("--dim_ff",            default=512,         type=int)
# # # # #     p.add_argument("--dropout",           default=0.1,         type=float)
# # # # #     p.add_argument("--unet_in_ch",        default=13,          type=int)

# # # # #     # Physics / loss hyperparams (giữ cho argparse compat; GradNorm tự học)
# # # # #     p.add_argument("--lambda_speed",      default=0.1,         type=float)
# # # # #     p.add_argument("--lambda_accel",      default=0.01,        type=float)
# # # # #     p.add_argument("--w_mse",            default=0.05,        type=float)
# # # # #     p.add_argument("--v_max_kmh",         default=80.0,        type=float)
# # # # #     p.add_argument("--w_heading",         default=0.3,         type=float)
# # # # #     p.add_argument("--w_recurv",          default=0.05,        type=float)
# # # # #     p.add_argument("--w_gate_reg",        default=0.01,        type=float)
# # # # #     p.add_argument("--step_weight_slope", default=0.1,         type=float)
# # # # #     p.add_argument("--recurv_threshold",  default=45.0,        type=float)
# # # # #     p.add_argument("--threshold_pct",     default=70.0,        type=float)
# # # # #     p.add_argument("--gradnorm_alpha",    default=1.5,         type=float)
# # # # #     p.add_argument("--gate_hidden",       default=32,          type=int)
# # # # #     p.add_argument("--recurv_hidden",     default=64,          type=int)

# # # # #     # Training
# # # # #     p.add_argument("--num_epochs",        default=600,         type=int)
# # # # #     p.add_argument("--batch_size",        default=32,          type=int)
# # # # #     p.add_argument("--lr",                default=1e-3,        type=float)
# # # # #     p.add_argument("--lr_gradnorm",       default=1e-4,        type=float,
# # # # #                    help="Learning rate cho GradNorm λ parameters")
# # # # #     p.add_argument("--weight_decay",      default=1e-4,        type=float)
# # # # #     p.add_argument("--grad_clip",         default=0.5,         type=float)
# # # # #     p.add_argument("--patience",          default=100,         type=int)
# # # # #     p.add_argument("--min_epochs",        default=50,          type=int)
# # # # #     p.add_argument("--lr_patience",       default=20,          type=int)
# # # # #     p.add_argument("--lr_factor",         default=0.5,         type=float)
# # # # #     p.add_argument("--lr_min",            default=1e-6,        type=float)
# # # # #     p.add_argument("--val_freq",          default=5,           type=int)
# # # # #     p.add_argument("--val_subset",        default=600,         type=int)
# # # # #     p.add_argument("--num_workers",       default=2,           type=int)
# # # # #     p.add_argument("--gradnorm_start_epoch", default=1,        type=int,
# # # # #                    help="Epoch bắt đầu update GradNorm (sau khi set L0)")

# # # # #     # Test
# # # # #     p.add_argument("--test_at_end",       action="store_true")

# # # # #     # I/O
# # # # #     p.add_argument("--output_dir",        default="runs/st_trans_v2", type=str)
# # # # #     p.add_argument("--metrics_csv",       default="metrics.csv",      type=str)
# # # # #     p.add_argument("--gpu_num",           default="0",                type=str)
# # # # #     p.add_argument("--resume",            default=None,               type=str)

# # # # #     # DataLoader compat
# # # # #     p.add_argument("--delim",        default=" ")
# # # # #     p.add_argument("--skip",         default=1,     type=int)
# # # # #     p.add_argument("--min_ped",      default=1,     type=int)
# # # # #     p.add_argument("--threshold",    default=0.002, type=float)
# # # # #     p.add_argument("--other_modal",  default="gph")

# # # # #     return p.parse_args()


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  MAIN
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def main(args):
# # # # #     if torch.cuda.is_available():
# # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# # # # #     best_ckpt   = os.path.join(args.output_dir, "best_model.pth")

# # # # #     print("=" * 70)
# # # # #     print("  ST-TRANS v2  |  GradNorm + Easy/Hard Split + Physics Gate")
# # # # #     print(f"  gradnorm_alpha={args.gradnorm_alpha}"
# # # # #           f"  step_slope={args.step_weight_slope}"
# # # # #           f"  threshold_pct={args.threshold_pct}")
# # # # #     print("=" * 70)

# # # # #     # ── Data ──────────────────────────────────────────────────────────────
# # # # #     train_dataset, train_loader = data_loader(
# # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # #     val_dataset, val_loader = data_loader(
# # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)
# # # # #     val_sub_loader = make_subset_loader(
# # # # #         val_dataset, args.val_subset, args.batch_size)

# # # # #     print(f"  train: {len(train_dataset)} seq  val: {len(val_dataset)} seq")

# # # # #     # ── Compute easy/hard threshold từ train set ──────────────────────────
# # # # #     print(f"\n  Computing easy/hard threshold"
# # # # #           f" (p{args.threshold_pct:.0f} from train set)...")
# # # # #     t0 = time.perf_counter()
# # # # #     threshold_curv, threshold_spd = compute_hard_thresholds(
# # # # #         train_loader, device, percentile=args.threshold_pct)
# # # # #     print(f"  threshold_curv = {threshold_curv:.3f}°"
# # # # #           f"  threshold_spd = {threshold_spd:.4f}"
# # # # #           f"  ({time.perf_counter() - t0:.1f}s)")

# # # # #     # Estimate hard fraction
# # # # #     n_hard_est, n_total = 0, 0
# # # # #     with torch.no_grad():
# # # # #         for batch in train_loader:
# # # # #             obs = batch[0].to(device)
# # # # #             is_hard = classify_hard_obs(obs, threshold_curv, threshold_spd)
# # # # #             n_hard_est += is_hard.sum().item()
# # # # #             n_total    += obs.shape[1]
# # # # #     print(f"  Hard fraction: {n_hard_est}/{n_total}"
# # # # #           f" = {100 * n_hard_est / max(n_total, 1):.1f}%"
# # # # #           f"  (target ~30%, p{args.threshold_pct:.0f})")

# # # # #     # ── Model ─────────────────────────────────────────────────────────────
# # # # #     model = build_st_trans_v2(args, threshold_curv, threshold_spd).to(device)

# # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # #     print(f"\n  Params total : {n_params:,}")
# # # # #     print(f"  Encoder      : {sum(p.numel() for p in model.encoder.parameters()):,}")
# # # # #     print(f"  Gate         : {sum(p.numel() for p in model.steering_gate.parameters()):,}")
# # # # #     print(f"  Recurv head  : {sum(p.numel() for p in model.recurv_head.parameters()):,}")
# # # # #     print(f"  GradNorm λ   : {sum(p.numel() for p in model.gradnorm.parameters()):,}")

# # # # #     # ── Resume ────────────────────────────────────────────────────────────
# # # # #     start_epoch  = 0
# # # # #     best_ade     = float("inf")
# # # # #     patience_cnt = 0
# # # # #     L0_set       = False

# # # # #     if args.resume and os.path.exists(args.resume):
# # # # #         ckpt = torch.load(args.resume, map_location=device)
# # # # #         model.load_state_dict(ckpt["model_state"])
# # # # #         start_epoch  = ckpt.get("epoch", 0) + 1
# # # # #         best_ade     = ckpt.get("best_ade", float("inf"))
# # # # #         patience_cnt = ckpt.get("patience_cnt", 0)
# # # # #         if "threshold_curv" in ckpt:
# # # # #             model.set_thresholds(ckpt["threshold_curv"], ckpt["threshold_spd"])
# # # # #         if "gradnorm_L0" in ckpt:
# # # # #             model.gradnorm.set_initial_losses(ckpt["gradnorm_L0"])
# # # # #             L0_set = True
# # # # #         print(f"\n  ↩ Resumed from {args.resume}"
# # # # #               f"  (epoch {start_epoch}, best_ADE={best_ade:.4f})")

# # # # #     # ── Optimizers ────────────────────────────────────────────────────────
# # # # #     # Optimizer 1: model parameters (không bao gồm GradNorm λ)
# # # # #     optimizer = optim.AdamW(
# # # # #         model.model_params(),
# # # # #         lr=args.lr,
# # # # #         weight_decay=args.weight_decay,
# # # # #     )
# # # # #     # Optimizer 2: GradNorm λ parameters (lr thấp hơn)
# # # # #     gn_optimizer = optim.Adam(
# # # # #         model.gradnorm_params(),
# # # # #         lr=args.lr_gradnorm,
# # # # #     )

# # # # #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# # # # #         optimizer, mode="min", factor=args.lr_factor,
# # # # #         patience=args.lr_patience, min_lr=args.lr_min,
# # # # #     )

# # # # #     train_start = time.perf_counter()
# # # # #     print("=" * 70)
# # # # #     print(f"  TRAINING  ({len(train_loader)} steps/epoch,"
# # # # #           f" max {args.num_epochs} epochs)")
# # # # #     print(f"  GradNorm starts at epoch {args.gradnorm_start_epoch}")
# # # # #     print("=" * 70)

# # # # #     # Lưu L0 để restore sau này
# # # # #     epoch_L0_log: dict = {}

# # # # #     for epoch in range(start_epoch, args.num_epochs):
# # # # #         model.train()
# # # # #         sum_loss     = 0.0
# # # # #         sum_easy_dpe = 0.0
# # # # #         sum_hard_dpe = 0.0
# # # # #         n_easy_total = 0
# # # # #         n_hard_total = 0
# # # # #         t0 = time.perf_counter()

# # # # #         for i, batch in enumerate(train_loader):
# # # # #             bl = move(list(batch), device)

# # # # #             # ── Forward ───────────────────────────────────────────────────
# # # # #             bd   = model.get_loss_breakdown(bl)
# # # # #             loss = bd["total"]

# # # # #             # ── Backward: model params ────────────────────────────────────
# # # # #             optimizer.zero_grad()
# # # # #             gn_optimizer.zero_grad()

# # # # #             # retain_graph=True để GradNorm có thể tính grad của λ sau đó
# # # # #             use_gradnorm = (epoch >= args.gradnorm_start_epoch
# # # # #                             and L0_set
# # # # #                             and len(bd.get("_raw_losses", {})) > 0)

# # # # #             loss.backward(retain_graph=use_gradnorm)
# # # # #             torch.nn.utils.clip_grad_norm_(model.model_params(), args.grad_clip)
# # # # #             optimizer.step()

# # # # #             # ── Backward: GradNorm λ ──────────────────────────────────────
# # # # #             if use_gradnorm:
# # # # #                 try:
# # # # #                     gn_loss = model.gradnorm_loss(bd)
# # # # #                     gn_loss.backward()
# # # # #                     gn_optimizer.step()
# # # # #                     model.gradnorm.renormalize()
# # # # #                 except Exception as e:
# # # # #                     # GradNorm có thể fail nếu graph đã bị free
# # # # #                     pass
# # # # #             elif not use_gradnorm and epoch >= args.gradnorm_start_epoch and not L0_set:
# # # # #                 # Vẫn giải phóng graph nếu retain_graph=True không cần thiết
# # # # #                 pass

# # # # #             # ── Logging ───────────────────────────────────────────────────
# # # # #             sum_loss += loss.item()
# # # # #             ne = bd.get("n_easy", 0)
# # # # #             nh = bd.get("n_hard", 0)
# # # # #             n_easy_total += ne
# # # # #             n_hard_total += nh
# # # # #             if ne > 0:
# # # # #                 sum_easy_dpe += bd.get("easy_dpe", 0.0)
# # # # #             if nh > 0:
# # # # #                 sum_hard_dpe += bd.get("hard_dpe", 0.0)

# # # # #             if i % 30 == 0:
# # # # #                 lr       = optimizer.param_groups[0]["lr"]
# # # # #                 alpha    = bd.get("alpha_mean",  float("nan"))
# # # # #                 sw72     = bd.get("step_w_72h",  float("nan"))
# # # # #                 head     = bd.get("hard_heading", float("nan"))
# # # # #                 lam_mse  = bd.get("λ_mse",       float("nan"))
# # # # #                 lam_head = bd.get("λ_heading",    float("nan"))
# # # # #                 print(
# # # # #                     f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
# # # # #                     f"  loss={loss.item():.4f}"
# # # # #                     f"  easy_dpe={bd.get('easy_dpe', 0.0):.1f}"
# # # # #                     f"  hard_dpe={bd.get('hard_dpe', 0.0):.1f}"
# # # # #                     f"  head={head:.3f}"
# # # # #                     f"  α={alpha:.3f}"
# # # # #                     f"  sw72={sw72:.2f}"
# # # # #                     f"  λ_mse={lam_mse:.3f}"
# # # # #                     f"  λ_head={lam_head:.3f}"
# # # # #                     f"  ne={ne}/nh={nh}"
# # # # #                     f"  lr={lr:.2e}"
# # # # #                 )

# # # # #         # ── Sau epoch đầu tiên: set L0 cho GradNorm ───────────────────────
# # # # #         if epoch == args.gradnorm_start_epoch - 1 and not L0_set:
# # # # #             # Chạy 1 pass nhanh để lấy loss trung bình
# # # # #             model.eval()
# # # # #             raw_accum: dict = {}
# # # # #             raw_count = 0
# # # # #             with torch.no_grad():
# # # # #                 for batch in train_loader:
# # # # #                     bl_tmp = move(list(batch), device)
# # # # #                     bd_tmp = model.get_loss_breakdown(bl_tmp)
# # # # #                     for k, v in bd_tmp.get("_raw_losses", {}).items():
# # # # #                         val = v.item() if isinstance(v, torch.Tensor) else float(v)
# # # # #                         raw_accum[k] = raw_accum.get(k, 0.0) + val
# # # # #                     raw_count += 1
# # # # #             if raw_count > 0:
# # # # #                 L0_dict = {k: v / raw_count for k, v in raw_accum.items()}
# # # # #                 model.gradnorm.set_initial_losses(L0_dict)
# # # # #                 epoch_L0_log = L0_dict
# # # # #                 L0_set = True
# # # # #                 print(f"\n  ✅ GradNorm L0 set: "
# # # # #                       + "  ".join(f"{k}={v:.4f}" for k, v in L0_dict.items()))
# # # # #             model.train()

# # # # #         # ── Val DPE (cho scheduler) ────────────────────────────────────────
# # # # #         avg_loss     = sum_loss / max(len(train_loader), 1)
# # # # #         avg_easy_dpe = sum_easy_dpe / max(n_easy_total // max(args.batch_size, 1), 1)
# # # # #         avg_hard_dpe = sum_hard_dpe / max(n_hard_total // max(args.batch_size, 1), 1)

# # # # #         model.eval()
# # # # #         val_dpe_sum = 0.0
# # # # #         n_val = 0
# # # # #         with torch.no_grad():
# # # # #             for batch in val_loader:
# # # # #                 bl_v    = move(list(batch), device)
# # # # #                 bd_v    = model.get_loss_breakdown(bl_v)
# # # # #                 val_dpe_sum += (bd_v.get("easy_dpe", 0.0)
# # # # #                                 + bd_v.get("hard_dpe", 0.0))
# # # # #                 n_val += 1
# # # # #         avg_val_dpe = val_dpe_sum / max(n_val, 1)

# # # # #         scheduler.step(avg_val_dpe)

# # # # #         ep_t = time.perf_counter() - t0
# # # # #         sw72 = model.step_weights[-1].item()
# # # # #         lam_dict = model.gradnorm.lambda_dict()
# # # # #         lam_str  = "  ".join(f"{k}={v:.3f}" for k, v in lam_dict.items())

# # # # #         print(
# # # # #             f"  Epoch {epoch:>4}"
# # # # #             f"  loss={avg_loss:.4f}"
# # # # #             f"  val_dpe={avg_val_dpe:.2f}"
# # # # #             f"  easy≈{avg_easy_dpe:.1f}"
# # # # #             f"  hard≈{avg_hard_dpe:.1f}"
# # # # #             f"  sw72={sw72:.3f}"
# # # # #             f"  {lam_str}"
# # # # #             f"  t={ep_t:.0f}s"
# # # # #         )

# # # # #         # ── ADE evaluation + easy/hard split ──────────────────────────────
# # # # #         if epoch % args.val_freq == 0:
# # # # #             r = evaluate(model, val_sub_loader, device)

# # # # #             ade     = r.get("ADE",         float("nan"))
# # # # #             easy_ad = r.get("easy_ADE",    float("nan"))
# # # # #             hard_ad = r.get("hard_ADE",    float("nan"))
# # # # #             ade72   = r.get("72h",         float("nan"))
# # # # #             ate72   = r.get("ATE_abs_72h", float("nan"))
# # # # #             cte72   = r.get("CTE_abs_72h", float("nan"))
# # # # #             n_e     = r.get("n_easy",      0)
# # # # #             n_h     = r.get("n_hard",      0)

# # # # #             print(f"\n  ╔═ VAL ep{epoch}")
# # # # #             print(f"  ║  ADE={ade:.4f}  easy={easy_ad:.4f}({n_e})"
# # # # #                   f"  hard={hard_ad:.4f}({n_h})")
# # # # #             print(f"  ║  72h={ade72:.4f}  ATE@72h={ate72:.4f}"
# # # # #                   f"  CTE@72h={cte72:.4f}")
# # # # #             print(f"  ╚═ sw72={sw72:.3f}"
# # # # #                   f"  curv_thr={model.threshold_curv:.2f}\n")

# # # # #             csv_row = {
# # # # #                 "timestamp":       datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # #                 "split":           "val",
# # # # #                 "epoch":           epoch,
# # # # #                 "model":           "ST-Trans-v2",
# # # # #                 "train_loss":      _fmt(avg_loss),
# # # # #                 "val_dpe":         _fmt(avg_val_dpe),
# # # # #                 "ADE":             _fmt(ade),
# # # # #                 "easy_ADE":        _fmt(easy_ad),
# # # # #                 "hard_ADE":        _fmt(hard_ad),
# # # # #                 "n_easy":          n_e,
# # # # #                 "n_hard":          n_h,
# # # # #                 "12h":             _fmt(r.get("12h",         float("nan"))),
# # # # #                 "24h":             _fmt(r.get("24h",         float("nan"))),
# # # # #                 "48h":             _fmt(r.get("48h",         float("nan"))),
# # # # #                 "72h":             _fmt(ade72),
# # # # #                 "ATE_abs_72h":     _fmt(ate72),
# # # # #                 "CTE_abs_72h":     _fmt(cte72),
# # # # #                 "step_w_72h":      _fmt(sw72),
# # # # #                 "threshold_curv":  _fmt(model.threshold_curv),
# # # # #                 "threshold_spd":   _fmt(model.threshold_spd),
# # # # #             }
# # # # #             csv_row.update({k: _fmt(v) for k, v in lam_dict.items()})
# # # # #             save_metrics_csv(csv_row, metrics_csv)

# # # # #             # ── Save best ─────────────────────────────────────────────────
# # # # #             if ade < best_ade:
# # # # #                 best_ade     = ade
# # # # #                 patience_cnt = 0
# # # # #                 torch.save(
# # # # #                     {
# # # # #                         "epoch":          epoch,
# # # # #                         "model_state":    model.state_dict(),
# # # # #                         "best_ade":       best_ade,
# # # # #                         "threshold_curv": model.threshold_curv,
# # # # #                         "threshold_spd":  model.threshold_spd,
# # # # #                         "step_weights":   model.step_weights.detach().cpu().tolist(),
# # # # #                         "gradnorm_L0":    epoch_L0_log,
# # # # #                     },
# # # # #                     best_ckpt,
# # # # #                 )
# # # # #                 print(f"  ✅ Best ADE = {best_ade:.4f} km  (epoch {epoch})")
# # # # #             else:
# # # # #                 patience_cnt += args.val_freq
# # # # #                 print(f"  No improve {patience_cnt}/{args.patience}"
# # # # #                       f"  (best={best_ade:.4f})")

# # # # #             if epoch >= args.min_epochs and patience_cnt >= args.patience:
# # # # #                 print(f"\n  ⛔ Early stop @ epoch {epoch}")
# # # # #                 break

# # # # #         # ── Periodic checkpoint ───────────────────────────────────────────
# # # # #         if epoch % 50 == 0 and epoch > 0:
# # # # #             torch.save(
# # # # #                 {
# # # # #                     "epoch":          epoch,
# # # # #                     "model_state":    model.state_dict(),
# # # # #                     "train_loss":     avg_loss,
# # # # #                     "val_dpe":        avg_val_dpe,
# # # # #                     "best_ade":       best_ade,
# # # # #                     "patience_cnt":   patience_cnt,
# # # # #                     "threshold_curv": model.threshold_curv,
# # # # #                     "threshold_spd":  model.threshold_spd,
# # # # #                     "gradnorm_L0":    epoch_L0_log,
# # # # #                 },
# # # # #                 os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"),
# # # # #             )

# # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # #     print("=" * 70)
# # # # #     print(f"  Model    : ST-Trans-v2")
# # # # #     print(f"  Best ADE : {best_ade:.4f} km")
# # # # #     print(f"  Total    : {total_h:.2f}h")
# # # # #     print("=" * 70)

# # # # #     if args.test_at_end and os.path.exists(best_ckpt):
# # # # #         run_test_evaluation(model, best_ckpt, args, device, metrics_csv)


# # # # # if __name__ == "__main__":
# # # # #     args = get_args()
# # # # #     np.random.seed(42)
# # # # #     torch.manual_seed(42)
# # # # #     if torch.cuda.is_available():
# # # # #         torch.cuda.manual_seed_all(42)
# # # # #     main(args)

# # # # """
# # # # scripts/train_st_trans_v2.py  ── ST-Trans v2 Training
# # # # ======================================================
# # # # Tương thích với STTransV2 mới:
# # # #   - Không có GradNorm (đã thay bằng UncertaintyWeighting)
# # # #   - Không có w_easy_boost (UW tự balance)
# # # #   - step_weights = softmax (không collapse)
# # # #   - lp_h.detach() bảo vệ decoder
# # # #   - 1 optimizer duy nhất: model params + UW σ params

# # # # CHẠY:
# # # #   python scripts/train_st_trans_v2.py \
# # # #     --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# # # #     --output_dir   runs/st_trans_v2 \
# # # #     --num_epochs   600 \
# # # #     --batch_size   90 \
# # # #     --threshold_pct 70 \
# # # #     --val_freq  5 \
# # # #     --val_subset 600 \
# # # #     --test_at_end

# # # # RESUME:
# # # #   python scripts/train_st_trans_v2.py \
# # # #     --resume runs/st_trans_v2/best_model.pth \
# # # #     --dataset_root ...
# # # # """
# # # # from __future__ import annotations

# # # # import sys, os
# # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # import argparse, time, random, csv, math
# # # # from datetime import datetime

# # # # import numpy as np
# # # # import torch
# # # # import torch.optim as optim
# # # # from torch.utils.data import DataLoader, Subset

# # # # from Model.data.loader_training import data_loader
# # # # from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # from Model.st_trans_model import (
# # # #     STTransV2, build_st_trans_v2,
# # # #     classify_hard_obs, compute_hard_thresholds,
# # # # )
# # # # from Model.paper_baseline_model import (
# # # #     haversine_km, _norm_to_deg, HORIZON_STEPS,
# # # # )
# # # # try:
# # # #     from Model.paper_baseline_model import _ate_cte_tensors
# # # # except ImportError:
# # # #     def _ate_cte_tensors(pred, gt):
# # # #         return torch.zeros_like(pred[...,0]), torch.zeros_like(pred[...,0])


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Helpers
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def move(batch, device):
# # # #     out = list(batch)
# # # #     for i, x in enumerate(out):
# # # #         if torch.is_tensor(x):
# # # #             out[i] = x.to(device)
# # # #         elif isinstance(x, dict):
# # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # #                       for k, v in x.items()}
# # # #     return out


# # # # def make_subset_loader(dataset, subset_size, batch_size, seed=42):
# # # #     rng = random.Random(seed)
# # # #     idx = rng.sample(range(len(dataset)), min(subset_size, len(dataset)))
# # # #     return DataLoader(Subset(dataset, idx),
# # # #                       batch_size=batch_size, shuffle=False,
# # # #                       collate_fn=seq_collate, num_workers=0, drop_last=False)


# # # # def save_csv(row: dict, path: str):
# # # #     write_hdr = not os.path.exists(path)
# # # #     with open(path, "a", newline="") as f:
# # # #         w = csv.DictWriter(f, fieldnames=list(row.keys()))
# # # #         if write_hdr:
# # # #             w.writeheader()
# # # #         w.writerow(row)


# # # # def _fmt(v) -> str:
# # # #     if isinstance(v, float):
# # # #         return "nan" if (math.isnan(v) if not math.isinf(v) else False) else f"{v:.4f}"
# # # #     return str(v)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Evaluation
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # @torch.no_grad()
# # # # def evaluate(model: STTransV2, loader, device) -> dict:
# # # #     model.eval()
# # # #     all_ade, all_fde     = [], []
# # # #     easy_ade_list        = []
# # # #     hard_ade_list        = []
# # # #     ade_buf = {h: [] for h in HORIZON_STEPS}
# # # #     ate_buf = {h: [] for h in HORIZON_STEPS}
# # # #     cte_buf = {h: [] for h in HORIZON_STEPS}

# # # #     for batch in loader:
# # # #         bl       = move(list(batch), device)
# # # #         obs_traj = bl[0]
# # # #         gt       = bl[1]
# # # #         pred, _, _ = model.sample(bl)

# # # #         T      = min(pred.shape[0], gt.shape[0])
# # # #         pred_d = _norm_to_deg(pred[:T])
# # # #         gt_d   = _norm_to_deg(gt[:T])
# # # #         dist   = haversine_km(pred_d, gt_d)          # [T, B]

# # # #         try:
# # # #             ate, cte = _ate_cte_tensors(pred[:T], gt[:T])
# # # #         except Exception:
# # # #             ate = cte = torch.zeros_like(dist)

# # # #         ade_per = dist.mean(0)
# # # #         all_ade.extend(ade_per.tolist())
# # # #         all_fde.extend(dist[-1].tolist())

# # # #         is_hard = classify_hard_obs(obs_traj,
# # # #                                     model.threshold_curv, model.threshold_spd)
# # # #         if (~is_hard).any():
# # # #             easy_ade_list.extend(ade_per[~is_hard].tolist())
# # # #         if is_hard.any():
# # # #             hard_ade_list.extend(ade_per[is_hard].tolist())

# # # #         for h, s in HORIZON_STEPS.items():
# # # #             if s < T:
# # # #                 ade_buf[h].extend(dist[s].tolist())
# # # #                 ate_buf[h].extend(ate[s].abs().tolist())
# # # #                 cte_buf[h].extend(cte[s].abs().tolist())

# # # #     def _mean(lst):
# # # #         return float(np.mean(lst)) if lst else float("nan")

# # # #     result = dict(
# # # #         ADE      = _mean(all_ade),
# # # #         FDE      = _mean(all_fde),
# # # #         easy_ADE = _mean(easy_ade_list),
# # # #         hard_ADE = _mean(hard_ade_list),
# # # #         n_easy   = len(easy_ade_list),
# # # #         n_hard   = len(hard_ade_list),
# # # #     )
# # # #     for h in HORIZON_STEPS:
# # # #         result[f"{h}h"]         = _mean(ade_buf[h])
# # # #         result[f"ATE_abs_{h}h"] = _mean(ate_buf[h])
# # # #         result[f"CTE_abs_{h}h"] = _mean(cte_buf[h])
# # # #     return result


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Test
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def run_test(model, ckpt_path, args, device, csv_path):
# # # #     print("\n" + "="*65)
# # # #     print("  TEST SET EVALUATION  (ST-Trans v2)")
# # # #     print("="*65)
# # # #     ckpt = torch.load(ckpt_path, map_location=device)
# # # #     model.load_state_dict(ckpt["model_state"])
# # # #     model.set_thresholds(
# # # #         ckpt.get("threshold_curv", model.threshold_curv),
# # # #         ckpt.get("threshold_spd",  model.threshold_spd))
# # # #     print(f"  Loaded ep{ckpt.get('epoch','?')}"
# # # #           f"  best_val={ckpt.get('best_ade', float('nan')):.2f} km")

# # # #     _, test_loader = data_loader(
# # # #         args, {"root": args.dataset_root, "type": "test"}, test=True)
# # # #     metrics = evaluate(model, test_loader, device)

# # # #     print()
# # # #     for k, v in metrics.items():
# # # #         print(f"  {k:<22} {_fmt(v)}")

# # # #     row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # #            "split": "test", "model": "ST-Trans-v2-UW"}
# # # #     row.update({k: _fmt(v) for k, v in metrics.items()})
# # # #     save_csv(row, csv_path)
# # # #     print("="*65)
# # # #     return metrics


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Args
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def get_args():
# # # #     p = argparse.ArgumentParser(
# # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # #     # Data
# # # #     p.add_argument("--dataset_root",     default="TCND_vn")
# # # #     p.add_argument("--obs_len",          default=8,     type=int)
# # # #     p.add_argument("--pred_len",         default=12,    type=int)
# # # #     # Model arch
# # # #     p.add_argument("--d_model",          default=64,    type=int)
# # # #     p.add_argument("--nhead",            default=4,     type=int)
# # # #     p.add_argument("--num_enc_layers",   default=1,     type=int)
# # # #     p.add_argument("--num_dec_layers",   default=3,     type=int)
# # # #     p.add_argument("--dim_ff",           default=512,   type=int)
# # # #     p.add_argument("--dropout",          default=0.1,   type=float)
# # # #     p.add_argument("--unet_in_ch",       default=13,    type=int)
# # # #     p.add_argument("--v_max_kmh",        default=80.0,  type=float)
# # # #     p.add_argument("--recurv_threshold", default=45.0,  type=float)
# # # #     p.add_argument("--gate_hidden",      default=32,    type=int)
# # # #     p.add_argument("--recurv_hidden",    default=64,    type=int)
# # # #     # Threshold
# # # #     p.add_argument("--threshold_pct",    default=70.0,  type=float)
# # # #     # Training
# # # #     p.add_argument("--num_epochs",       default=600,   type=int)
# # # #     p.add_argument("--batch_size",       default=90,    type=int)
# # # #     p.add_argument("--lr",               default=1e-3,  type=float)
# # # #     p.add_argument("--lr_uw",            default=1e-4,  type=float,
# # # #                    help="LR cho UW sigma params (thường nhỏ hơn lr)")
# # # #     p.add_argument("--weight_decay",     default=1e-4,  type=float)
# # # #     p.add_argument("--grad_clip",        default=0.5,   type=float)
# # # #     p.add_argument("--patience",         default=100,   type=int)
# # # #     p.add_argument("--min_epochs",       default=50,    type=int)
# # # #     p.add_argument("--lr_patience",      default=20,    type=int)
# # # #     p.add_argument("--lr_factor",        default=0.5,   type=float)
# # # #     p.add_argument("--lr_min",           default=1e-6,  type=float)
# # # #     # Eval
# # # #     p.add_argument("--val_freq",         default=5,     type=int)
# # # #     p.add_argument("--val_subset",       default=600,   type=int)
# # # #     p.add_argument("--num_workers",      default=2,     type=int)
# # # #     p.add_argument("--test_at_end",      action="store_true")
# # # #     # IO
# # # #     p.add_argument("--output_dir",       default="runs/st_trans_v2")
# # # #     p.add_argument("--metrics_csv",      default="metrics.csv")
# # # #     p.add_argument("--gpu_num",          default="0")
# # # #     p.add_argument("--resume",           default=None)
# # # #     # DataLoader compat
# # # #     p.add_argument("--delim",      default=" ")
# # # #     p.add_argument("--skip",       default=1,     type=int)
# # # #     p.add_argument("--min_ped",    default=1,     type=int)
# # # #     p.add_argument("--threshold",  default=0.002, type=float)
# # # #     p.add_argument("--other_modal",default="gph")
# # # #     return p.parse_args()


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  MAIN
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def main(args):
# # # #     if torch.cuda.is_available():
# # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # #     os.makedirs(args.output_dir, exist_ok=True)
# # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# # # #     best_ckpt   = os.path.join(args.output_dir, "best_model.pth")

# # # #     print("="*65)
# # # #     print("  ST-TRANS v2  |  Uncertainty Weighting + Easy/Hard Split")
# # # #     print(f"  threshold_pct={args.threshold_pct}%")
# # # #     print(f"  lr={args.lr}  lr_uw={args.lr_uw}")
# # # #     print("="*65)

# # # #     # ── Data ──────────────────────────────────────────────────────────────
# # # #     train_dataset, train_loader = data_loader(
# # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # #     val_dataset, _ = data_loader(
# # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)
# # # #     val_sub = make_subset_loader(val_dataset, args.val_subset, args.batch_size)
# # # #     print(f"  train: {len(train_dataset)}  val: {len(val_dataset)}")

# # # #     # ── Thresholds ────────────────────────────────────────────────────────
# # # #     print(f"\n  Computing easy/hard threshold (p{args.threshold_pct:.0f})...")
# # # #     t0 = time.perf_counter()
# # # #     tc, ts = compute_hard_thresholds(
# # # #         train_loader, device, percentile=args.threshold_pct)
# # # #     print(f"  threshold_curv={tc:.3f}°  threshold_spd={ts:.4f}"
# # # #           f"  ({time.perf_counter()-t0:.1f}s)")

# # # #     # Hard fraction check
# # # #     n_h, n_tot = 0, 0
# # # #     with torch.no_grad():
# # # #         for b in train_loader:
# # # #             obs = b[0].to(device)
# # # #             n_h   += classify_hard_obs(obs, tc, ts).sum().item()
# # # #             n_tot += obs.shape[1]
# # # #     print(f"  Hard fraction: {n_h}/{n_tot} = {100*n_h/max(n_tot,1):.1f}%"
# # # #           f"  (target ~30%)")

# # # #     # ── Model ─────────────────────────────────────────────────────────────
# # # #     model = build_st_trans_v2(args, tc, ts).to(device)

# # # #     # Log params
# # # #     n_total   = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # #     n_encoder = sum(p.numel() for p in model.encoder.parameters())
# # # #     n_gate    = sum(p.numel() for p in model.steering_gate.parameters())
# # # #     n_recurv  = sum(p.numel() for p in model.recurv_head.parameters())
# # # #     n_uw      = sum(p.numel() for p in model.uw.parameters())
# # # #     n_sw      = model.raw_step_weights.numel()
# # # #     print(f"\n  Params total : {n_total:,}")
# # # #     print(f"  Encoder      : {n_encoder:,}")
# # # #     print(f"  Gate         : {n_gate:,}")
# # # #     print(f"  Recurv head  : {n_recurv:,}")
# # # #     print(f"  UW σ params  : {n_uw}  (tasks: {model.uw.task_names})")
# # # #     print(f"  Step weights : {n_sw}  (softmax, init uniform)")
# # # #     print(f"  Init σ = 1.0 for all tasks → effective weight = 0.5")

# # # #     # ── Resume ────────────────────────────────────────────────────────────
# # # #     start_epoch  = 0
# # # #     best_ade     = float("inf")
# # # #     patience_cnt = 0

# # # #     if args.resume and os.path.exists(args.resume):
# # # #         ckpt = torch.load(args.resume, map_location=device)
# # # #         model.load_state_dict(ckpt["model_state"])
# # # #         start_epoch  = ckpt.get("epoch", 0) + 1
# # # #         best_ade     = ckpt.get("best_ade", float("inf"))
# # # #         patience_cnt = ckpt.get("patience_cnt", 0)
# # # #         model.set_thresholds(
# # # #             ckpt.get("threshold_curv", tc),
# # # #             ckpt.get("threshold_spd",  ts))
# # # #         print(f"\n  ↩ Resumed ep{start_epoch}  best={best_ade:.2f} km")

# # # #     # ── Optimizer ─────────────────────────────────────────────────────────
# # # #     # UW σ params dùng lr nhỏ hơn để sigma không oscillate
# # # #     uw_param_ids = set(id(p) for p in model.uw.parameters())
# # # #     model_params = [p for p in model.parameters()
# # # #                     if id(p) not in uw_param_ids and p.requires_grad]
# # # #     uw_params    = list(model.uw.parameters())

# # # #     optimizer = optim.AdamW([
# # # #         {"params": model_params, "lr": args.lr,    "weight_decay": args.weight_decay},
# # # #         {"params": uw_params,    "lr": args.lr_uw, "weight_decay": 0.0},
# # # #     ])
# # # #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# # # #         optimizer, mode="min", factor=args.lr_factor,
# # # #         patience=args.lr_patience, min_lr=args.lr_min)

# # # #     train_start = time.perf_counter()
# # # #     print()
# # # #     print("="*65)
# # # #     print(f"  TRAINING  ({len(train_loader)} steps/epoch, max {args.num_epochs} epochs)")
# # # #     print("="*65)

# # # #     for epoch in range(start_epoch, args.num_epochs):
# # # #         model.train()
# # # #         sum_loss = 0.0
# # # #         t0 = time.perf_counter()

# # # #         for i, batch in enumerate(train_loader):
# # # #             bl = move(list(batch), device)
# # # #             bd = model.get_loss_breakdown(bl)

# # # #             optimizer.zero_grad()
# # # #             bd["total"].backward()   # UW σ cũng được update ở đây
# # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # #             optimizer.step()

# # # #             sum_loss += bd["total"].item()

# # # #             if i % 30 == 0:
# # # #                 lr     = optimizer.param_groups[0]["lr"]
# # # #                 # Log σ để monitor UW tự học
# # # #                 sigma  = model.uw.sigma_dict()
# # # #                 s_dpe_e = sigma.get("σ_dpe_easy",  float("nan"))
# # # #                 s_head  = sigma.get("σ_heading",   float("nan"))
# # # #                 s_rec   = sigma.get("σ_recurv",    float("nan"))
# # # #                 sw_min  = bd.get("sw_min",  float("nan"))
# # # #                 sw_max  = bd.get("sw_max",  float("nan"))
# # # #                 sw72    = bd.get("sw72",    float("nan"))
# # # #                 alpha   = bd.get("alpha_mean", float("nan"))
# # # #                 n_e     = bd.get("n_easy", 0)
# # # #                 n_h     = bd.get("n_hard", 0)
# # # #                 print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
# # # #                       f"  loss={bd['total'].item():.4f}"
# # # #                       f"  easy_dpe={bd.get('easy_dpe', 0):.1f}"
# # # #                       f"  hard_dpe={bd.get('hard_dpe', 0):.1f}"
# # # #                       f"  head={bd.get('hard_heading', 0):.3f}"
# # # #                       f"  α={alpha:.3f}"
# # # #                       f"  sw=[{sw_min:.2f},{sw_max:.2f}]"
# # # #                       f"  sw72={sw72:.3f}"
# # # #                       f"  σ_dpe_e={s_dpe_e:.3f}"
# # # #                       f"  σ_head={s_head:.3f}"
# # # #                       f"  σ_rec={s_rec:.3f}"
# # # #                       f"  ne={n_e}/nh={n_h}"
# # # #                       f"  lr={lr:.2e}")

# # # #         avg_loss = sum_loss / len(train_loader)

# # # #         # Val DPE cho scheduler (dùng subset nhanh)
# # # #         model.eval()
# # # #         val_sum, val_n = 0.0, 0
# # # #         with torch.no_grad():
# # # #             for batch in val_sub:
# # # #                 bl  = move(list(batch), device)
# # # #                 bd  = model.get_loss_breakdown(bl)
# # # #                 val_sum += bd["total"].item()
# # # #                 val_n   += 1
# # # #         avg_val = val_sum / max(val_n, 1)
# # # #         scheduler.step(avg_val)

# # # #         # Log σ của epoch
# # # #         sigma_d   = model.uw.sigma_dict()
# # # #         sigma_str = "  ".join(f"{k}={v:.3f}" for k,v in sigma_d.items())
# # # #         sw_vals   = model.step_weights.detach()
# # # #         lr_cur    = optimizer.param_groups[0]["lr"]

# # # #         print(f"  Epoch {epoch:>4}"
# # # #               f"  loss={avg_loss:.4f}"
# # # #               f"  val={avg_val:.2f}"
# # # #               f"  sw72={sw_vals[-1].item():.3f}"
# # # #               f"  sw_max={sw_vals.max().item():.3f}"
# # # #               f"  {sigma_str}"
# # # #               f"  lr={lr_cur:.2e}"
# # # #               f"  t={time.perf_counter()-t0:.0f}s")

# # # #         # ── Val ADE (full eval) ───────────────────────────────────────────
# # # #         if epoch % args.val_freq == 0:
# # # #             r = evaluate(model, val_sub, device)
# # # #             ade     = r["ADE"]
# # # #             easy_ad = r["easy_ADE"]
# # # #             hard_ad = r["hard_ADE"]
# # # #             ade72   = r.get("72h",         float("nan"))
# # # #             ate72   = r.get("ATE_abs_72h", float("nan"))
# # # #             cte72   = r.get("CTE_abs_72h", float("nan"))

# # # #             print(f"\n  ╔═ VAL ep{epoch}")
# # # #             print(f"  ║  ADE={ade:.4f}"
# # # #                   f"  easy={easy_ad:.4f}({r['n_easy']})"
# # # #                   f"  hard={hard_ad:.4f}({r['n_hard']})")
# # # #             print(f"  ║  72h={ade72:.4f}"
# # # #                   f"  ATE@72h={ate72:.4f}"
# # # #                   f"  CTE@72h={cte72:.4f}")
# # # #             print(f"  ╚═ sw72={sw_vals[-1].item():.3f}"
# # # #                   f"  {sigma_str}\n")

# # # #             # Save CSV
# # # #             save_csv({
# # # #                 "timestamp":   datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # #                 "split":       "val",
# # # #                 "epoch":       epoch,
# # # #                 "model":       "ST-Trans-v2-UW",
# # # #                 "train_loss":  _fmt(avg_loss),
# # # #                 "val_loss":    _fmt(avg_val),
# # # #                 "ADE":         _fmt(ade),
# # # #                 "easy_ADE":    _fmt(easy_ad),
# # # #                 "hard_ADE":    _fmt(hard_ad),
# # # #                 "72h":         _fmt(ade72),
# # # #                 "ATE_abs_72h": _fmt(ate72),
# # # #                 "CTE_abs_72h": _fmt(cte72),
# # # #                 "sw72":        _fmt(sw_vals[-1].item()),
# # # #                 "sw_max":      _fmt(sw_vals.max().item()),
# # # #                 **{k: _fmt(v) for k,v in sigma_d.items()},
# # # #             }, metrics_csv)

# # # #             # Best model
# # # #             if ade < best_ade:
# # # #                 best_ade     = ade
# # # #                 patience_cnt = 0
# # # #                 torch.save({
# # # #                     "epoch":          epoch,
# # # #                     "model_state":    model.state_dict(),
# # # #                     "best_ade":       best_ade,
# # # #                     "threshold_curv": model.threshold_curv,
# # # #                     "threshold_spd":  model.threshold_spd,
# # # #                     "sigma":          sigma_d,
# # # #                     "sw":             sw_vals.tolist(),
# # # #                 }, best_ckpt)
# # # #                 print(f"  ✅ Best ADE = {best_ade:.4f} km  (epoch {epoch})")
# # # #             else:
# # # #                 patience_cnt += args.val_freq
# # # #                 print(f"  No improve {patience_cnt}/{args.patience}"
# # # #                       f"  (best={best_ade:.4f})")

# # # #             if epoch >= args.min_epochs and patience_cnt >= args.patience:
# # # #                 print(f"\n  ⛔ Early stop @ epoch {epoch}")
# # # #                 break

# # # #         # Periodic checkpoint
# # # #         if epoch % 50 == 0 and epoch > 0:
# # # #             torch.save({
# # # #                 "epoch":          epoch,
# # # #                 "model_state":    model.state_dict(),
# # # #                 "train_loss":     avg_loss,
# # # #                 "best_ade":       best_ade,
# # # #                 "patience_cnt":   patience_cnt,
# # # #                 "threshold_curv": model.threshold_curv,
# # # #                 "threshold_spd":  model.threshold_spd,
# # # #             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

# # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # #     print("="*65)
# # # #     print(f"  Best ADE : {best_ade:.4f} km")
# # # #     print(f"  Total    : {total_h:.2f}h")
# # # #     print(f"  Final σ  : {model.uw.sigma_dict()}")
# # # #     print(f"  Final sw : {model.step_weights.detach().tolist()}")
# # # #     print("="*65)

# # # #     if args.test_at_end and os.path.exists(best_ckpt):
# # # #         run_test(model, best_ckpt, args, device, metrics_csv)


# # # # if __name__ == "__main__":
# # # #     args = get_args()
# # # #     np.random.seed(42)
# # # #     torch.manual_seed(42)
# # # #     if torch.cuda.is_available():
# # # #         torch.cuda.manual_seed_all(42)
# # # #     main(args)

# # # """
# # # scripts/train_st_trans_v2.py  ── ST-Trans v2 Training
# # # ======================================================
# # # Tương thích với STTransV2 mới:
# # #   - Không có GradNorm (đã thay bằng UncertaintyWeighting)
# # #   - Không có w_easy_boost (UW tự balance)
# # #   - step_weights = softmax (không collapse)
# # #   - lp_h.detach() bảo vệ decoder
# # #   - 1 optimizer duy nhất: model params + UW σ params

# # # CHẠY:
# # #   python scripts/train_st_trans_v2.py \
# # #     --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# # #     --output_dir   runs/st_trans_v2 \
# # #     --num_epochs   600 \
# # #     --batch_size   90 \
# # #     --threshold_pct 70 \
# # #     --val_freq  5 \
# # #     --val_subset 600 \
# # #     --test_at_end

# # # RESUME:
# # #   python scripts/train_st_trans_v2.py \
# # #     --resume runs/st_trans_v2/best_model.pth \
# # #     --dataset_root ...
# # # """
# # # from __future__ import annotations

# # # import sys, os
# # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # import argparse, time, random, csv, math
# # # from datetime import datetime

# # # import numpy as np
# # # import torch
# # # import torch.optim as optim
# # # from torch.utils.data import DataLoader, Subset

# # # from Model.data.loader_training import data_loader
# # # from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # from Model.st_trans_model import (
# # #     STTransV2, build_st_trans_v2,
# # #     classify_hard_obs, compute_hard_thresholds,
# # # )
# # # from Model.paper_baseline_model import (
# # #     haversine_km, _norm_to_deg, HORIZON_STEPS,
# # # )
# # # try:
# # #     from Model.paper_baseline_model import _ate_cte_tensors
# # # except ImportError:
# # #     def _ate_cte_tensors(pred, gt):
# # #         return torch.zeros_like(pred[...,0]), torch.zeros_like(pred[...,0])


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Helpers
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def move(batch, device):
# # #     out = list(batch)
# # #     for i, x in enumerate(out):
# # #         if torch.is_tensor(x):
# # #             out[i] = x.to(device)
# # #         elif isinstance(x, dict):
# # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # #                       for k, v in x.items()}
# # #     return out


# # # def make_subset_loader(dataset, subset_size, batch_size, seed=42):
# # #     rng = random.Random(seed)
# # #     idx = rng.sample(range(len(dataset)), min(subset_size, len(dataset)))
# # #     return DataLoader(Subset(dataset, idx),
# # #                       batch_size=batch_size, shuffle=False,
# # #                       collate_fn=seq_collate, num_workers=0, drop_last=False)


# # # def save_csv(row: dict, path: str):
# # #     write_hdr = not os.path.exists(path)
# # #     with open(path, "a", newline="") as f:
# # #         w = csv.DictWriter(f, fieldnames=list(row.keys()))
# # #         if write_hdr:
# # #             w.writeheader()
# # #         w.writerow(row)


# # # def _fmt(v) -> str:
# # #     if isinstance(v, float):
# # #         return "nan" if (math.isnan(v) if not math.isinf(v) else False) else f"{v:.4f}"
# # #     return str(v)


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Evaluation
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # @torch.no_grad()
# # # def evaluate(model: STTransV2, loader, device) -> dict:
# # #     model.eval()
# # #     all_ade, all_fde     = [], []
# # #     easy_ade_list        = []
# # #     hard_ade_list        = []
# # #     ade_buf = {h: [] for h in HORIZON_STEPS}
# # #     ate_buf = {h: [] for h in HORIZON_STEPS}
# # #     cte_buf = {h: [] for h in HORIZON_STEPS}

# # #     for batch in loader:
# # #         bl       = move(list(batch), device)
# # #         obs_traj = bl[0]
# # #         gt       = bl[1]
# # #         pred, _, _ = model.sample(bl)

# # #         T      = min(pred.shape[0], gt.shape[0])
# # #         pred_d = _norm_to_deg(pred[:T])
# # #         gt_d   = _norm_to_deg(gt[:T])
# # #         dist   = haversine_km(pred_d, gt_d)          # [T, B]

# # #         try:
# # #             ate, cte = _ate_cte_tensors(pred[:T], gt[:T])
# # #         except Exception:
# # #             ate = cte = torch.zeros_like(dist)

# # #         ade_per = dist.mean(0)
# # #         all_ade.extend(ade_per.tolist())
# # #         all_fde.extend(dist[-1].tolist())

# # #         is_hard = classify_hard_obs(obs_traj,
# # #                                     model.threshold_curv, model.threshold_spd)
# # #         if (~is_hard).any():
# # #             easy_ade_list.extend(ade_per[~is_hard].tolist())
# # #         if is_hard.any():
# # #             hard_ade_list.extend(ade_per[is_hard].tolist())

# # #         for h, s in HORIZON_STEPS.items():
# # #             if s < T:
# # #                 ade_buf[h].extend(dist[s].tolist())
# # #                 ate_buf[h].extend(ate[s].abs().tolist())
# # #                 cte_buf[h].extend(cte[s].abs().tolist())

# # #     def _mean(lst):
# # #         return float(np.mean(lst)) if lst else float("nan")

# # #     result = dict(
# # #         ADE      = _mean(all_ade),
# # #         FDE      = _mean(all_fde),
# # #         easy_ADE = _mean(easy_ade_list),
# # #         hard_ADE = _mean(hard_ade_list),
# # #         n_easy   = len(easy_ade_list),
# # #         n_hard   = len(hard_ade_list),
# # #     )
# # #     for h in HORIZON_STEPS:
# # #         result[f"{h}h"]         = _mean(ade_buf[h])
# # #         result[f"ATE_abs_{h}h"] = _mean(ate_buf[h])
# # #         result[f"CTE_abs_{h}h"] = _mean(cte_buf[h])
# # #     return result


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Test
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def run_test(model, ckpt_path, args, device, csv_path):
# # #     print("\n" + "="*65)
# # #     print("  TEST SET EVALUATION  (ST-Trans v2)")
# # #     print("="*65)
# # #     ckpt = torch.load(ckpt_path, map_location=device)
# # #     model.load_state_dict(ckpt["model_state"])
# # #     model.set_thresholds(
# # #         ckpt.get("threshold_curv", model.threshold_curv),
# # #         ckpt.get("threshold_spd",  model.threshold_spd))
# # #     print(f"  Loaded ep{ckpt.get('epoch','?')}"
# # #           f"  best_val={ckpt.get('best_ade', float('nan')):.2f} km")

# # #     _, test_loader = data_loader(
# # #         args, {"root": args.dataset_root, "type": "test"}, test=True)
# # #     metrics = evaluate(model, test_loader, device)

# # #     print()
# # #     for k, v in metrics.items():
# # #         print(f"  {k:<22} {_fmt(v)}")

# # #     row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
# # #            "split": "test", "model": "ST-Trans-v2-UW"}
# # #     row.update({k: _fmt(v) for k, v in metrics.items()})
# # #     save_csv(row, csv_path)
# # #     print("="*65)
# # #     return metrics


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Args
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def get_args():
# # #     p = argparse.ArgumentParser(
# # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # #     # Data
# # #     p.add_argument("--dataset_root",     default="TCND_vn")
# # #     p.add_argument("--obs_len",          default=8,     type=int)
# # #     p.add_argument("--pred_len",         default=12,    type=int)
# # #     # Model arch
# # #     p.add_argument("--d_model",          default=64,    type=int)
# # #     p.add_argument("--nhead",            default=4,     type=int)
# # #     p.add_argument("--num_enc_layers",   default=1,     type=int)
# # #     p.add_argument("--num_dec_layers",   default=3,     type=int)
# # #     p.add_argument("--dim_ff",           default=512,   type=int)
# # #     p.add_argument("--dropout",          default=0.1,   type=float)
# # #     p.add_argument("--unet_in_ch",       default=13,    type=int)
# # #     p.add_argument("--v_max_kmh",        default=80.0,  type=float)
# # #     p.add_argument("--recurv_threshold", default=45.0,  type=float)
# # #     p.add_argument("--gate_hidden",      default=32,    type=int)
# # #     p.add_argument("--recurv_hidden",    default=64,    type=int)
# # #     # Threshold
# # #     p.add_argument("--threshold_pct",    default=70.0,  type=float)
# # #     # Training
# # #     p.add_argument("--num_epochs",       default=600,   type=int)
# # #     p.add_argument("--batch_size",       default=90,    type=int)
# # #     p.add_argument("--lr",               default=1e-3,  type=float)
# # #     p.add_argument("--lr_uw",            default=1e-4,  type=float,
# # #                    help="LR cho UW sigma params (thường nhỏ hơn lr)")
# # #     p.add_argument("--weight_decay",     default=1e-4,  type=float)
# # #     p.add_argument("--grad_clip",        default=0.5,   type=float)
# # #     p.add_argument("--patience",         default=100,   type=int)
# # #     p.add_argument("--min_epochs",       default=50,    type=int)
# # #     p.add_argument("--lr_patience",      default=20,    type=int)
# # #     p.add_argument("--lr_factor",        default=0.5,   type=float)
# # #     p.add_argument("--lr_min",           default=1e-6,  type=float)
# # #     # Eval
# # #     p.add_argument("--val_freq",         default=5,     type=int)
# # #     p.add_argument("--val_subset",       default=600,   type=int)
# # #     p.add_argument("--num_workers",      default=2,     type=int)
# # #     p.add_argument("--test_at_end",      action="store_true")
# # #     # IO
# # #     p.add_argument("--output_dir",       default="runs/st_trans_v2")
# # #     p.add_argument("--metrics_csv",      default="metrics.csv")
# # #     p.add_argument("--gpu_num",          default="0")
# # #     p.add_argument("--resume",           default=None)
# # #     # DataLoader compat
# # #     p.add_argument("--delim",      default=" ")
# # #     p.add_argument("--skip",       default=1,     type=int)
# # #     p.add_argument("--min_ped",    default=1,     type=int)
# # #     p.add_argument("--threshold",  default=0.002, type=float)
# # #     p.add_argument("--other_modal",default="gph")
# # #     return p.parse_args()


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  MAIN
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def main(args):
# # #     if torch.cuda.is_available():
# # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     os.makedirs(args.output_dir, exist_ok=True)
# # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# # #     best_ckpt   = os.path.join(args.output_dir, "best_model.pth")

# # #     print("="*65)
# # #     print("  ST-TRANS v2  |  Uncertainty Weighting + Easy/Hard Split")
# # #     print(f"  threshold_pct={args.threshold_pct}%")
# # #     print(f"  lr={args.lr}  lr_uw={args.lr_uw}")
# # #     print("="*65)

# # #     # ── Data ──────────────────────────────────────────────────────────────
# # #     train_dataset, train_loader = data_loader(
# # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # #     val_dataset, _ = data_loader(
# # #         args, {"root": args.dataset_root, "type": "val"}, test=True)
# # #     val_sub = make_subset_loader(val_dataset, args.val_subset, args.batch_size)
# # #     print(f"  train: {len(train_dataset)}  val: {len(val_dataset)}")

# # #     # ── Thresholds ────────────────────────────────────────────────────────
# # #     print(f"\n  Computing easy/hard threshold (p{args.threshold_pct:.0f})...")
# # #     t0 = time.perf_counter()
# # #     tc, ts = compute_hard_thresholds(
# # #         train_loader, device, percentile=args.threshold_pct)
# # #     print(f"  threshold_curv={tc:.3f}°  threshold_spd={ts:.4f}"
# # #           f"  ({time.perf_counter()-t0:.1f}s)")

# # #     # Hard fraction check
# # #     n_h, n_tot = 0, 0
# # #     with torch.no_grad():
# # #         for b in train_loader:
# # #             obs = b[0].to(device)
# # #             n_h   += classify_hard_obs(obs, tc, ts).sum().item()
# # #             n_tot += obs.shape[1]
# # #     print(f"  Hard fraction: {n_h}/{n_tot} = {100*n_h/max(n_tot,1):.1f}%"
# # #           f"  (target ~30%)")

# # #     # ── Model ─────────────────────────────────────────────────────────────
# # #     model = build_st_trans_v2(args, tc, ts).to(device)

# # #     # Log params
# # #     n_total   = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # #     n_encoder = sum(p.numel() for p in model.encoder.parameters())
# # #     n_gate    = sum(p.numel() for p in model.steering_gate.parameters())
# # #     n_recurv  = sum(p.numel() for p in model.recurv_head.parameters())
# # #     n_uw      = sum(p.numel() for p in model.uw.parameters())
# # #     print(f"\n  Params total : {n_total:,}")
# # #     print(f"  Encoder      : {n_encoder:,}")
# # #     print(f"  Gate         : {n_gate:,}")
# # #     print(f"  Recurv head  : {n_recurv:,}")
# # #     print(f"  UW σ params  : {n_uw}  (tasks: {model.uw.task_names})")
# # #     print(f"  UW tasks     : {len(model.uw.TASKS)}  {model.uw.TASKS}")
# # #     print(f"  Init σ = 1.0 for all tasks → effective weight = 0.5")

# # #     # ── Resume ────────────────────────────────────────────────────────────
# # #     start_epoch  = 0
# # #     best_ade     = float("inf")
# # #     patience_cnt = 0

# # #     if args.resume and os.path.exists(args.resume):
# # #         ckpt = torch.load(args.resume, map_location=device)
# # #         model.load_state_dict(ckpt["model_state"])
# # #         start_epoch  = ckpt.get("epoch", 0) + 1
# # #         best_ade     = ckpt.get("best_ade", float("inf"))
# # #         patience_cnt = ckpt.get("patience_cnt", 0)
# # #         model.set_thresholds(
# # #             ckpt.get("threshold_curv", tc),
# # #             ckpt.get("threshold_spd",  ts))
# # #         print(f"\n  ↩ Resumed ep{start_epoch}  best={best_ade:.2f} km")

# # #     # ── Optimizer ─────────────────────────────────────────────────────────
# # #     # UW σ params dùng lr nhỏ hơn để sigma không oscillate
# # #     uw_param_ids = set(id(p) for p in model.uw.parameters())
# # #     model_params = [p for p in model.parameters()
# # #                     if id(p) not in uw_param_ids and p.requires_grad]
# # #     uw_params    = list(model.uw.parameters())

# # #     optimizer = optim.AdamW([
# # #         {"params": model_params, "lr": args.lr,    "weight_decay": args.weight_decay},
# # #         {"params": uw_params,    "lr": args.lr_uw, "weight_decay": 0.0},
# # #     ])
# # #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# # #         optimizer, mode="min", factor=args.lr_factor,
# # #         patience=args.lr_patience, min_lr=args.lr_min)

# # #     train_start = time.perf_counter()
# # #     print()
# # #     print("="*65)
# # #     print(f"  TRAINING  ({len(train_loader)} steps/epoch, max {args.num_epochs} epochs)")
# # #     print("="*65)

# # #     for epoch in range(start_epoch, args.num_epochs):
# # #         model.train()
# # #         sum_loss = 0.0
# # #         t0 = time.perf_counter()

# # #         for i, batch in enumerate(train_loader):
# # #             bl = move(list(batch), device)
# # #             bd = model.get_loss_breakdown(bl)

# # #             optimizer.zero_grad()
# # #             bd["total"].backward()   # UW σ cũng được update ở đây
# # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # #             optimizer.step()

# # #             sum_loss += bd["total"].item()

# # #             if i % 30 == 0:
# # #                 lr     = optimizer.param_groups[0]["lr"]
# # #                 # Log σ để monitor UW tự học
# # #                 sigma  = model.uw.sigma_dict()
# # #                 s_head  = sigma.get("σ_heading",   float("nan"))
# # #                 s_head  = sigma.get("σ_heading",   float("nan"))
# # #                 s_rec   = sigma.get("σ_recurv",    float("nan"))
# # #                 alpha   = bd.get("alpha_mean", float("nan"))
# # #                 alpha   = bd.get("alpha_mean", float("nan"))
# # #                 n_e     = bd.get("n_easy", 0)
# # #                 n_h     = bd.get("n_hard", 0)
# # #                 print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
# # #                       f"  loss={bd['total'].item():.4f}"
# # #                       f"  easy_dpe={bd.get('easy_dpe', 0):.1f}"
# # #                       f"  hard_dpe={bd.get('hard_dpe', 0):.1f}"
# # #                       f"  head={bd.get('hard_heading', 0):.3f}"
# # #                       f"  α={alpha:.3f}"
# # #                       f"  α={alpha:.3f}"
# # #                       f"  σ_head={s_head:.3f}"
# # #                       f"  σ_rec={s_rec:.3f}"
# # #                       f"  ne={n_e}/nh={n_h}"
# # #                       f"  lr={lr:.2e}")

# # #         avg_loss = sum_loss / len(train_loader)

# # #         # Val DPE cho scheduler (dùng subset nhanh)
# # #         model.eval()
# # #         val_sum, val_n = 0.0, 0
# # #         with torch.no_grad():
# # #             for batch in val_sub:
# # #                 bl  = move(list(batch), device)
# # #                 bd  = model.get_loss_breakdown(bl)
# # #                 val_sum += bd["total"].item()
# # #                 val_n   += 1
# # #         avg_val = val_sum / max(val_n, 1)
# # #         scheduler.step(avg_val)

# # #         # Log σ của epoch
# # #         sigma_d   = model.uw.sigma_dict()
# # #         sigma_str = "  ".join(f"{k}={v:.3f}" for k,v in sigma_d.items())
# # #         lr_cur    = optimizer.param_groups[0]["lr"]

# # #         print(f"  Epoch {epoch:>4}"
# # #               f"  loss={avg_loss:.4f}"
# # #               f"  val={avg_val:.2f}"
# # #               f"  {sigma_str}"
# # #               f"  lr={lr_cur:.2e}"
# # #               f"  t={time.perf_counter()-t0:.0f}s")

# # #         # ── Val ADE (full eval) ───────────────────────────────────────────
# # #         if epoch % args.val_freq == 0:
# # #             r = evaluate(model, val_sub, device)
# # #             ade     = r["ADE"]
# # #             easy_ad = r["easy_ADE"]
# # #             hard_ad = r["hard_ADE"]
# # #             ade72   = r.get("72h",         float("nan"))
# # #             ate72   = r.get("ATE_abs_72h", float("nan"))
# # #             cte72   = r.get("CTE_abs_72h", float("nan"))

# # #             print(f"\n  ╔═ VAL ep{epoch}")
# # #             print(f"  ║  ADE={ade:.4f}"
# # #                   f"  easy={easy_ad:.4f}({r['n_easy']})"
# # #                   f"  hard={hard_ad:.4f}({r['n_hard']})")
# # #             print(f"  ║  72h={ade72:.4f}"
# # #                   f"  ATE@72h={ate72:.4f}"
# # #                   f"  CTE@72h={cte72:.4f}")
# # #             print(f"  ╚═ {sigma_str}\n")

# # #             # Save CSV
# # #             save_csv({
# # #                 "timestamp":   datetime.now().strftime("%Y%m%d_%H%M%S"),
# # #                 "split":       "val",
# # #                 "epoch":       epoch,
# # #                 "model":       "ST-Trans-v2-UW",
# # #                 "train_loss":  _fmt(avg_loss),
# # #                 "val_loss":    _fmt(avg_val),
# # #                 "ADE":         _fmt(ade),
# # #                 "easy_ADE":    _fmt(easy_ad),
# # #                 "hard_ADE":    _fmt(hard_ad),
# # #                 "72h":         _fmt(ade72),
# # #                 "ATE_abs_72h": _fmt(ate72),
# # #                 "CTE_abs_72h": _fmt(cte72),
# # #                 **{k: _fmt(v) for k,v in sigma_d.items()},
# # #             }, metrics_csv)

# # #             # Best model
# # #             if ade < best_ade:
# # #                 best_ade     = ade
# # #                 patience_cnt = 0
# # #                 torch.save({
# # #                     "epoch":          epoch,
# # #                     "model_state":    model.state_dict(),
# # #                     "best_ade":       best_ade,
# # #                     "threshold_curv": model.threshold_curv,
# # #                     "threshold_spd":  model.threshold_spd,
# # #                     "sigma":          sigma_d,
# # #                 }, best_ckpt)
# # #                 print(f"  ✅ Best ADE = {best_ade:.4f} km  (epoch {epoch})")
# # #             else:
# # #                 patience_cnt += args.val_freq
# # #                 print(f"  No improve {patience_cnt}/{args.patience}"
# # #                       f"  (best={best_ade:.4f})")

# # #             if epoch >= args.min_epochs and patience_cnt >= args.patience:
# # #                 print(f"\n  ⛔ Early stop @ epoch {epoch}")
# # #                 break

# # #         # Periodic checkpoint
# # #         if epoch % 50 == 0 and epoch > 0:
# # #             torch.save({
# # #                 "epoch":          epoch,
# # #                 "model_state":    model.state_dict(),
# # #                 "train_loss":     avg_loss,
# # #                 "best_ade":       best_ade,
# # #                 "patience_cnt":   patience_cnt,
# # #                 "threshold_curv": model.threshold_curv,
# # #                 "threshold_spd":  model.threshold_spd,
# # #             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

# # #     total_h = (time.perf_counter() - train_start) / 3600
# # #     print("="*65)
# # #     print(f"  Best ADE : {best_ade:.4f} km")
# # #     print(f"  Total    : {total_h:.2f}h")
# # #     print(f"  Final σ  : {model.uw.sigma_dict()}")
# # #     print("="*65)

# # #     if args.test_at_end and os.path.exists(best_ckpt):
# # #         run_test(model, best_ckpt, args, device, metrics_csv)


# # # if __name__ == "__main__":
# # #     args = get_args()
# # #     np.random.seed(42)
# # #     torch.manual_seed(42)
# # #     if torch.cuda.is_available():
# # #         torch.cuda.manual_seed_all(42)
# # #     main(args)


# # """
# # scripts/train_st_trans_v2.py  ── ST-Trans v2 Training
# # ======================================================
# # Tương thích với STTransV2 mới:
# #   - Không có GradNorm (đã thay bằng UncertaintyWeighting)
# #   - Không có w_easy_boost (UW tự balance)
# #   - step_weights = softmax (không collapse)
# #   - lp_h.detach() bảo vệ decoder
# #   - 1 optimizer duy nhất: model params + UW σ params

# # CHẠY:
# #   python scripts/train_st_trans_v2.py \
# #     --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
# #     --output_dir   runs/st_trans_v2 \
# #     --num_epochs   600 \
# #     --batch_size   90 \
# #     --threshold_pct 70 \
# #     --val_freq  5 \
# #     --val_subset 600 \
# #     --test_at_end

# # RESUME:
# #   python scripts/train_st_trans_v2.py \
# #     --resume runs/st_trans_v2/best_model.pth \
# #     --dataset_root ...
# # """
# # from __future__ import annotations

# # import sys, os
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # import argparse, time, random, csv, math
# # from datetime import datetime

# # import numpy as np
# # import torch
# # import torch.optim as optim
# # from torch.utils.data import DataLoader, Subset

# # from Model.data.loader_training import data_loader
# # from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # from Model.st_trans_model import (
# #     STTransV2, build_st_trans_v2,
# #     classify_hard_obs, compute_hard_thresholds,
# # )
# # from Model.paper_baseline_model import (
# #     haversine_km, _norm_to_deg, HORIZON_STEPS,
# # )
# # try:
# #     from Model.paper_baseline_model import _ate_cte_tensors
# # except ImportError:
# #     def _ate_cte_tensors(pred, gt):
# #         return torch.zeros_like(pred[...,0]), torch.zeros_like(pred[...,0])


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Helpers
# # # ══════════════════════════════════════════════════════════════════════════════

# # def move(batch, device):
# #     out = list(batch)
# #     for i, x in enumerate(out):
# #         if torch.is_tensor(x):
# #             out[i] = x.to(device)
# #         elif isinstance(x, dict):
# #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# #                       for k, v in x.items()}
# #     return out


# # def make_subset_loader(dataset, subset_size, batch_size, seed=42):
# #     rng = random.Random(seed)
# #     idx = rng.sample(range(len(dataset)), min(subset_size, len(dataset)))
# #     return DataLoader(Subset(dataset, idx),
# #                       batch_size=batch_size, shuffle=False,
# #                       collate_fn=seq_collate, num_workers=0, drop_last=False)


# # def save_csv(row: dict, path: str):
# #     write_hdr = not os.path.exists(path)
# #     with open(path, "a", newline="") as f:
# #         w = csv.DictWriter(f, fieldnames=list(row.keys()))
# #         if write_hdr:
# #             w.writeheader()
# #         w.writerow(row)


# # def _fmt(v) -> str:
# #     if isinstance(v, float):
# #         return "nan" if (math.isnan(v) if not math.isinf(v) else False) else f"{v:.4f}"
# #     return str(v)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Evaluation
# # # ══════════════════════════════════════════════════════════════════════════════

# # @torch.no_grad()
# # def evaluate(model: STTransV2, loader, device) -> dict:
# #     model.eval()
# #     all_ade, all_fde     = [], []
# #     easy_ade_list        = []
# #     hard_ade_list        = []
# #     ade_buf = {h: [] for h in HORIZON_STEPS}
# #     ate_buf = {h: [] for h in HORIZON_STEPS}
# #     cte_buf = {h: [] for h in HORIZON_STEPS}

# #     for batch in loader:
# #         bl       = move(list(batch), device)
# #         obs_traj = bl[0]
# #         gt       = bl[1]
# #         pred, _, _ = model.sample(bl)

# #         T      = min(pred.shape[0], gt.shape[0])
# #         pred_d = _norm_to_deg(pred[:T])
# #         gt_d   = _norm_to_deg(gt[:T])
# #         dist   = haversine_km(pred_d, gt_d)          # [T, B]

# #         try:
# #             ate, cte = _ate_cte_tensors(pred[:T], gt[:T])
# #         except Exception:
# #             ate = cte = torch.zeros_like(dist)

# #         ade_per = dist.mean(0)
# #         all_ade.extend(ade_per.tolist())
# #         all_fde.extend(dist[-1].tolist())

# #         is_hard = classify_hard_obs(obs_traj,
# #                                     model.threshold_curv, model.threshold_spd)
# #         if (~is_hard).any():
# #             easy_ade_list.extend(ade_per[~is_hard].tolist())
# #         if is_hard.any():
# #             hard_ade_list.extend(ade_per[is_hard].tolist())

# #         for h, s in HORIZON_STEPS.items():
# #             if s < T:
# #                 ade_buf[h].extend(dist[s].tolist())
# #                 ate_buf[h].extend(ate[s].abs().tolist())
# #                 cte_buf[h].extend(cte[s].abs().tolist())

# #     def _mean(lst):
# #         return float(np.mean(lst)) if lst else float("nan")

# #     result = dict(
# #         ADE      = _mean(all_ade),
# #         FDE      = _mean(all_fde),
# #         easy_ADE = _mean(easy_ade_list),
# #         hard_ADE = _mean(hard_ade_list),
# #         n_easy   = len(easy_ade_list),
# #         n_hard   = len(hard_ade_list),
# #     )
# #     for h in HORIZON_STEPS:
# #         result[f"{h}h"]         = _mean(ade_buf[h])
# #         result[f"ATE_abs_{h}h"] = _mean(ate_buf[h])
# #         result[f"CTE_abs_{h}h"] = _mean(cte_buf[h])
# #     return result


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Test
# # # ══════════════════════════════════════════════════════════════════════════════

# # def run_test(model, ckpt_path, args, device, csv_path):
# #     print("\n" + "="*65)
# #     print("  TEST SET EVALUATION  (ST-Trans v2)")
# #     print("="*65)
# #     ckpt = torch.load(ckpt_path, map_location=device)
# #     model.load_state_dict(ckpt["model_state"])
# #     model.set_thresholds(
# #         ckpt.get("threshold_curv", model.threshold_curv),
# #         ckpt.get("threshold_spd",  model.threshold_spd))
# #     print(f"  Loaded ep{ckpt.get('epoch','?')}"
# #           f"  best_val={ckpt.get('best_ade', float('nan')):.2f} km")

# #     _, test_loader = data_loader(
# #         args, {"root": args.dataset_root, "type": "test"}, test=True)
# #     metrics = evaluate(model, test_loader, device)

# #     print()
# #     for k, v in metrics.items():
# #         print(f"  {k:<22} {_fmt(v)}")

# #     row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
# #            "split": "test", "model": "ST-Trans-v2-UW"}
# #     row.update({k: _fmt(v) for k, v in metrics.items()})
# #     save_csv(row, csv_path)
# #     print("="*65)
# #     return metrics


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Args
# # # ══════════════════════════════════════════════════════════════════════════════

# # def get_args():
# #     p = argparse.ArgumentParser(
# #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# #     # Data
# #     p.add_argument("--dataset_root",     default="TCND_vn")
# #     p.add_argument("--obs_len",          default=8,     type=int)
# #     p.add_argument("--pred_len",         default=12,    type=int)
# #     # Model arch
# #     p.add_argument("--d_model",          default=64,    type=int)
# #     p.add_argument("--nhead",            default=4,     type=int)
# #     p.add_argument("--num_enc_layers",   default=1,     type=int)
# #     p.add_argument("--num_dec_layers",   default=3,     type=int)
# #     p.add_argument("--dim_ff",           default=512,   type=int)
# #     p.add_argument("--dropout",          default=0.1,   type=float)
# #     p.add_argument("--unet_in_ch",       default=13,    type=int)
# #     p.add_argument("--v_max_kmh",        default=80.0,  type=float)
# #     p.add_argument("--recurv_threshold", default=45.0,  type=float)
# #     p.add_argument("--gate_hidden",      default=32,    type=int)
# #     p.add_argument("--recurv_hidden",    default=64,    type=int)
# #     # Threshold
# #     p.add_argument("--threshold_pct",    default=70.0,  type=float)
# #     # Training
# #     p.add_argument("--num_epochs",       default=600,   type=int)
# #     p.add_argument("--batch_size",       default=90,    type=int)
# #     p.add_argument("--lr",               default=1e-3,  type=float)
# #     p.add_argument("--lr_uw",            default=1e-4,  type=float,
# #                    help="LR cho UW sigma params (thường nhỏ hơn lr)")
# #     p.add_argument("--weight_decay",     default=1e-4,  type=float)
# #     p.add_argument("--grad_clip",        default=0.5,   type=float)
# #     p.add_argument("--patience",         default=100,   type=int)
# #     p.add_argument("--min_epochs",       default=50,    type=int)
# #     p.add_argument("--lr_patience",      default=20,    type=int)
# #     p.add_argument("--lr_factor",        default=0.5,   type=float)
# #     p.add_argument("--lr_min",           default=1e-6,  type=float)
# #     # Eval
# #     p.add_argument("--val_freq",         default=5,     type=int)
# #     p.add_argument("--val_subset",       default=600,   type=int)
# #     p.add_argument("--num_workers",      default=2,     type=int)
# #     p.add_argument("--test_at_end",      action="store_true")
# #     # IO
# #     p.add_argument("--output_dir",       default="runs/st_trans_v2")
# #     p.add_argument("--metrics_csv",      default="metrics.csv")
# #     p.add_argument("--gpu_num",          default="0")
# #     p.add_argument("--resume",           default=None)
# #     # DataLoader compat
# #     p.add_argument("--delim",      default=" ")
# #     p.add_argument("--skip",       default=1,     type=int)
# #     p.add_argument("--min_ped",    default=1,     type=int)
# #     p.add_argument("--threshold",  default=0.002, type=float)
# #     p.add_argument("--other_modal",default="gph")
# #     return p.parse_args()


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  MAIN
# # # ══════════════════════════════════════════════════════════════════════════════

# # def main(args):
# #     if torch.cuda.is_available():
# #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     os.makedirs(args.output_dir, exist_ok=True)
# #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# #     best_ckpt   = os.path.join(args.output_dir, "best_model.pth")

# #     print("="*65)
# #     print("  ST-TRANS v2  |  Uncertainty Weighting + Easy/Hard Split")
# #     print(f"  threshold_pct={args.threshold_pct}%")
# #     print(f"  lr={args.lr}  lr_uw={args.lr_uw}")
# #     print("="*65)

# #     # ── Data ──────────────────────────────────────────────────────────────
# #     train_dataset, train_loader = data_loader(
# #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# #     val_dataset, _ = data_loader(
# #         args, {"root": args.dataset_root, "type": "val"}, test=True)
# #     val_sub = make_subset_loader(val_dataset, args.val_subset, args.batch_size)
# #     print(f"  train: {len(train_dataset)}  val: {len(val_dataset)}")

# #     # ── Thresholds ────────────────────────────────────────────────────────
# #     print(f"\n  Computing easy/hard threshold (p{args.threshold_pct:.0f})...")
# #     t0 = time.perf_counter()
# #     tc, ts = compute_hard_thresholds(
# #         train_loader, device, percentile=args.threshold_pct)
# #     print(f"  threshold_curv={tc:.3f}°  threshold_spd={ts:.4f}"
# #           f"  ({time.perf_counter()-t0:.1f}s)")

# #     # Hard fraction check
# #     n_h, n_tot = 0, 0
# #     with torch.no_grad():
# #         for b in train_loader:
# #             obs = b[0].to(device)
# #             n_h   += classify_hard_obs(obs, tc, ts).sum().item()
# #             n_tot += obs.shape[1]
# #     print(f"  Hard fraction: {n_h}/{n_tot} = {100*n_h/max(n_tot,1):.1f}%"
# #           f"  (target ~30%)")

# #     # ── Model ─────────────────────────────────────────────────────────────
# #     model = build_st_trans_v2(args, tc, ts).to(device)

# #     # Log params
# #     n_total   = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #     n_encoder = sum(p.numel() for p in model.encoder.parameters())
# #     n_gate    = sum(p.numel() for p in model.steering_gate.parameters())
# #     n_recurv  = sum(p.numel() for p in model.recurv_head.parameters())
# #     n_uw      = sum(p.numel() for p in model.uw.parameters())
# #     print(f"\n  Params total : {n_total:,}")
# #     print(f"  Encoder      : {n_encoder:,}")
# #     print(f"  Gate         : {n_gate:,}")
# #     print(f"  Recurv head  : {n_recurv:,}")
# #     print(f"  UW σ params  : {n_uw}  (tasks: {model.uw.TASKS})")
# #     print(f"  UW tasks     : {len(model.uw.TASKS)}  {model.uw.TASKS}")
# #     print(f"  Init σ = 1.0 for all tasks → effective weight = 0.5")

# #     # ── Resume ────────────────────────────────────────────────────────────
# #     start_epoch  = 0
# #     best_ade     = float("inf")
# #     patience_cnt = 0

# #     if args.resume and os.path.exists(args.resume):
# #         ckpt = torch.load(args.resume, map_location=device)
# #         model.load_state_dict(ckpt["model_state"])
# #         start_epoch  = ckpt.get("epoch", 0) + 1
# #         best_ade     = ckpt.get("best_ade", float("inf"))
# #         patience_cnt = ckpt.get("patience_cnt", 0)
# #         model.set_thresholds(
# #             ckpt.get("threshold_curv", tc),
# #             ckpt.get("threshold_spd",  ts))
# #         print(f"\n  ↩ Resumed ep{start_epoch}  best={best_ade:.2f} km")

# #     # ── Optimizer ─────────────────────────────────────────────────────────
# #     # UW σ params dùng lr nhỏ hơn để sigma không oscillate
# #     uw_param_ids = set(id(p) for p in model.uw.parameters())
# #     model_params = [p for p in model.parameters()
# #                     if id(p) not in uw_param_ids and p.requires_grad]
# #     uw_params    = list(model.uw.parameters())

# #     optimizer = optim.AdamW([
# #         {"params": model_params, "lr": args.lr,    "weight_decay": args.weight_decay},
# #         {"params": uw_params,    "lr": args.lr_uw, "weight_decay": 0.0},
# #     ])
# #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# #         optimizer, mode="min", factor=args.lr_factor,
# #         patience=args.lr_patience, min_lr=args.lr_min)

# #     train_start = time.perf_counter()
# #     print()
# #     print("="*65)
# #     print(f"  TRAINING  ({len(train_loader)} steps/epoch, max {args.num_epochs} epochs)")
# #     print("="*65)

# #     for epoch in range(start_epoch, args.num_epochs):
# #         model.train()
# #         sum_loss = 0.0
# #         t0 = time.perf_counter()

# #         for i, batch in enumerate(train_loader):
# #             bl = move(list(batch), device)
# #             bd = model.get_loss_breakdown(bl)

# #             optimizer.zero_grad()
# #             bd["total"].backward()   # UW σ cũng được update ở đây
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# #             optimizer.step()

# #             sum_loss += bd["total"].item()

# #             if i % 30 == 0:
# #                 lr     = optimizer.param_groups[0]["lr"]
# #                 # Log σ để monitor UW tự học
# #                 sigma  = model.uw.sigma_dict()
# #                 s_head  = sigma.get("σ_heading",   float("nan"))
# #                 s_head  = sigma.get("σ_heading",   float("nan"))
# #                 s_rec   = sigma.get("σ_recurv",    float("nan"))
# #                 alpha   = bd.get("alpha_mean", float("nan"))
# #                 alpha   = bd.get("alpha_mean", float("nan"))
# #                 n_e     = bd.get("n_easy", 0)
# #                 n_h     = bd.get("n_hard", 0)
# #                 print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
# #                       f"  loss={bd['total'].item():.4f}"
# #                       f"  easy_dpe={bd.get('easy_dpe', 0):.1f}"
# #                       f"  hard_dpe={bd.get('hard_dpe', 0):.1f}"
# #                       f"  head={bd.get('hard_heading', 0):.3f}"
# #                       f"  α={alpha:.3f}"
# #                       f"  α={alpha:.3f}"
# #                       f"  σ_head={s_head:.3f}"
# #                       f"  σ_rec={s_rec:.3f}"
# #                       f"  ne={n_e}/nh={n_h}"
# #                       f"  lr={lr:.2e}")

# #         avg_loss = sum_loss / len(train_loader)

# #         # Val DPE cho scheduler (dùng subset nhanh)
# #         model.eval()
# #         val_sum, val_n = 0.0, 0
# #         with torch.no_grad():
# #             for batch in val_sub:
# #                 bl  = move(list(batch), device)
# #                 bd  = model.get_loss_breakdown(bl)
# #                 val_sum += bd["total"].item()
# #                 val_n   += 1
# #         avg_val = val_sum / max(val_n, 1)
# #         scheduler.step(avg_val)

# #         # Log σ của epoch
# #         sigma_d   = model.uw.sigma_dict()
# #         sigma_str = "  ".join(f"{k}={v:.3f}" for k,v in sigma_d.items())
# #         lr_cur    = optimizer.param_groups[0]["lr"]

# #         print(f"  Epoch {epoch:>4}"
# #               f"  loss={avg_loss:.4f}"
# #               f"  val={avg_val:.2f}"
# #               f"  {sigma_str}"
# #               f"  lr={lr_cur:.2e}"
# #               f"  t={time.perf_counter()-t0:.0f}s")

# #         # ── Val ADE (full eval) ───────────────────────────────────────────
# #         if epoch % args.val_freq == 0:
# #             r = evaluate(model, val_sub, device)
# #             ade     = r["ADE"]
# #             easy_ad = r["easy_ADE"]
# #             hard_ad = r["hard_ADE"]
# #             ade72   = r.get("72h",         float("nan"))
# #             ate72   = r.get("ATE_abs_72h", float("nan"))
# #             cte72   = r.get("CTE_abs_72h", float("nan"))

# #             print(f"\n  ╔═ VAL ep{epoch}")
# #             print(f"  ║  ADE={ade:.4f}"
# #                   f"  easy={easy_ad:.4f}({r['n_easy']})"
# #                   f"  hard={hard_ad:.4f}({r['n_hard']})")
# #             print(f"  ║  72h={ade72:.4f}"
# #                   f"  ATE@72h={ate72:.4f}"
# #                   f"  CTE@72h={cte72:.4f}")
# #             print(f"  ╚═ {sigma_str}\n")

# #             # Save CSV
# #             save_csv({
# #                 "timestamp":   datetime.now().strftime("%Y%m%d_%H%M%S"),
# #                 "split":       "val",
# #                 "epoch":       epoch,
# #                 "model":       "ST-Trans-v2-UW",
# #                 "train_loss":  _fmt(avg_loss),
# #                 "val_loss":    _fmt(avg_val),
# #                 "ADE":         _fmt(ade),
# #                 "easy_ADE":    _fmt(easy_ad),
# #                 "hard_ADE":    _fmt(hard_ad),
# #                 "72h":         _fmt(ade72),
# #                 "ATE_abs_72h": _fmt(ate72),
# #                 "CTE_abs_72h": _fmt(cte72),
# #                 **{k: _fmt(v) for k,v in sigma_d.items()},
# #             }, metrics_csv)

# #             # Best model
# #             if ade < best_ade:
# #                 best_ade     = ade
# #                 patience_cnt = 0
# #                 torch.save({
# #                     "epoch":          epoch,
# #                     "model_state":    model.state_dict(),
# #                     "best_ade":       best_ade,
# #                     "threshold_curv": model.threshold_curv,
# #                     "threshold_spd":  model.threshold_spd,
# #                     "sigma":          sigma_d,
# #                 }, best_ckpt)
# #                 print(f"  ✅ Best ADE = {best_ade:.4f} km  (epoch {epoch})")
# #             else:
# #                 patience_cnt += args.val_freq
# #                 print(f"  No improve {patience_cnt}/{args.patience}"
# #                       f"  (best={best_ade:.4f})")

# #             if epoch >= args.min_epochs and patience_cnt >= args.patience:
# #                 print(f"\n  ⛔ Early stop @ epoch {epoch}")
# #                 break

# #         # Periodic checkpoint
# #         if epoch % 50 == 0 and epoch > 0:
# #             torch.save({
# #                 "epoch":          epoch,
# #                 "model_state":    model.state_dict(),
# #                 "train_loss":     avg_loss,
# #                 "best_ade":       best_ade,
# #                 "patience_cnt":   patience_cnt,
# #                 "threshold_curv": model.threshold_curv,
# #                 "threshold_spd":  model.threshold_spd,
# #             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

# #     total_h = (time.perf_counter() - train_start) / 3600
# #     print("="*65)
# #     print(f"  Best ADE : {best_ade:.4f} km")
# #     print(f"  Total    : {total_h:.2f}h")
# #     print(f"  Final σ  : {model.uw.sigma_dict()}")
# #     print("="*65)

# #     if args.test_at_end and os.path.exists(best_ckpt):
# #         run_test(model, best_ckpt, args, device, metrics_csv)


# # if __name__ == "__main__":
# #     args = get_args()
# #     np.random.seed(42)
# #     torch.manual_seed(42)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(42)
# #     main(args)

# """
# scripts/train_st_trans_v3.py  ── ST-Trans v3 Training
# ======================================================
# Tương thích với STTransV3:
#   - d_model=128, nhead=8, num_enc_layers=2
#   - SteeringContextEncoder (ERA5 cho tất cả samples)
#   - RoPETransformerDecoder (thay Sinusoidal+TransformerDecoder)
#   - StepWeightedDPE (horizon weights tăng dần)
#   - UncertaintyWeighting (3 tasks: dpe_hard, heading, recurv)

# THAY ĐỔI SO VỚI TRAIN v2:
#   1. Import từ st_trans_v3_model thay vì v2
#   2. Thêm --d_model 128, --nhead 8, --num_enc_layers 2 defaults
#   3. Thêm --dpe_ramp arg
#   4. grad_clip tăng từ 0.5 → 1.0 (d_model lớn hơn cần clip rộng hơn)
#   5. lr giảm từ 1e-3 → 5e-4 (d_model=128 nhạy cảm hơn với lr cao)
#   6. warmup scheduler: linear warmup 5 epochs + CosineAnnealing
#      (giúp ổn định RoPE + SteeringEncoder ở early training)

# CHẠY:
#   python scripts/train_st_trans_v3.py \
#     --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
#     --output_dir   runs/st_trans_v3 \
#     --num_epochs   600 \
#     --batch_size   90 \
#     --d_model      128 \
#     --nhead        8 \
#     --num_enc_layers 2 \
#     --threshold_pct 70 \
#     --val_freq  5 \
#     --val_subset 600 \
#     --test_at_end

# RESUME:
#   python scripts/train_st_trans_v3.py \
#     --resume runs/st_trans_v3/best_model.pth \
#     --dataset_root ...
# """
# from __future__ import annotations

# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse, time, random, csv, math
# from datetime import datetime

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset

# from Model.data.loader_training import data_loader
# from Model.data.trajectoriesWithMe_unet_training import seq_collate
# from Model.st_trans_model import (
#     STTransV3, build_st_trans_v3,
#     classify_hard_obs, compute_hard_thresholds,
# )
# from Model.paper_baseline_model import (
#     haversine_km, _norm_to_deg, HORIZON_STEPS,
# )
# try:
#     from Model.paper_baseline_model import _ate_cte_tensors
# except ImportError:
#     def _ate_cte_tensors(pred, gt):
#         return torch.zeros_like(pred[..., 0]), torch.zeros_like(pred[..., 0])


# # ══════════════════════════════════════════════════════════════════════════════
# #  Helpers
# # ══════════════════════════════════════════════════════════════════════════════

# def move(batch, device):
#     out = list(batch)
#     for i, x in enumerate(out):
#         if torch.is_tensor(x):
#             out[i] = x.to(device)
#         elif isinstance(x, dict):
#             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
#                       for k, v in x.items()}
#     return out


# def make_subset_loader(dataset, subset_size, batch_size, seed=42):
#     rng = random.Random(seed)
#     idx = rng.sample(range(len(dataset)), min(subset_size, len(dataset)))
#     return DataLoader(Subset(dataset, idx),
#                       batch_size=batch_size, shuffle=False,
#                       collate_fn=seq_collate, num_workers=0, drop_last=False)


# def save_csv(row: dict, path: str):
#     write_hdr = not os.path.exists(path)
#     with open(path, "a", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=list(row.keys()))
#         if write_hdr:
#             w.writeheader()
#         w.writerow(row)


# def _fmt(v) -> str:
#     if isinstance(v, float):
#         return "nan" if (math.isnan(v) if not math.isinf(v) else False) else f"{v:.4f}"
#     return str(v)


# # ══════════════════════════════════════════════════════════════════════════════
# #  Warmup + Cosine Scheduler
# # ══════════════════════════════════════════════════════════════════════════════

# class WarmupCosineScheduler:
#     """
#     Linear warmup N_warmup epochs → Cosine anneal T_max epochs.

#     Lý do cần warmup cho v3:
#     - RoPE parameters (cos/sin cache) cần vài epoch để decoder
#       học cách sử dụng relative position info đúng cách
#     - SteeringContextEncoder mới hoàn toàn → gradient lớn ban đầu
#     - d_model=128 → nhiều params hơn → cần lr nhỏ hơn ở đầu

#     warmup_epochs: lr tuyến tính từ lr_init/10 → lr_init
#     sau đó: CosineAnnealingLR từ lr_init → lr_min trong T_max epochs
#     """

#     def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
#                  lr_init: float, lr_min: float = 1e-6):
#         self.optimizer      = optimizer
#         self.warmup_epochs  = warmup_epochs
#         self.total_epochs   = total_epochs
#         self.lr_init        = lr_init
#         self.lr_min         = lr_min
#         self.current_epoch  = 0

#     def step(self):
#         e = self.current_epoch
#         if e < self.warmup_epochs:
#             # Linear warmup: lr_init/10 → lr_init
#             lr = self.lr_init * (0.1 + 0.9 * e / max(self.warmup_epochs - 1, 1))
#         else:
#             # Cosine annealing
#             t = e - self.warmup_epochs
#             T = self.total_epochs - self.warmup_epochs
#             lr = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (
#                 1 + math.cos(math.pi * t / max(T, 1)))
#         for pg in self.optimizer.param_groups:
#             # UW params dùng lr/10 để sigma không oscillate
#             pg["lr"] = lr if not pg.get("is_uw", False) else lr * 0.1
#         self.current_epoch += 1
#         return lr


# # ══════════════════════════════════════════════════════════════════════════════
# #  Evaluation
# # ══════════════════════════════════════════════════════════════════════════════

# @torch.no_grad()
# def evaluate(model: STTransV3, loader, device) -> dict:
#     model.eval()
#     all_ade, all_fde  = [], []
#     easy_ade_list     = []
#     hard_ade_list     = []
#     ade_buf = {h: [] for h in HORIZON_STEPS}
#     ate_buf = {h: [] for h in HORIZON_STEPS}
#     cte_buf = {h: [] for h in HORIZON_STEPS}

#     for batch in loader:
#         bl       = move(list(batch), device)
#         obs_traj = bl[0]
#         gt       = bl[1]
#         pred, _, _ = model.sample(bl)

#         T      = min(pred.shape[0], gt.shape[0])
#         pred_d = _norm_to_deg(pred[:T])
#         gt_d   = _norm_to_deg(gt[:T])
#         dist   = haversine_km(pred_d, gt_d)   # [T, B]

#         try:
#             ate, cte = _ate_cte_tensors(pred[:T], gt[:T])
#         except Exception:
#             ate = cte = torch.zeros_like(dist)

#         ade_per = dist.mean(0)
#         all_ade.extend(ade_per.tolist())
#         all_fde.extend(dist[-1].tolist())

#         is_hard = classify_hard_obs(obs_traj,
#                                     model.threshold_curv, model.threshold_spd)
#         if (~is_hard).any():
#             easy_ade_list.extend(ade_per[~is_hard].tolist())
#         if is_hard.any():
#             hard_ade_list.extend(ade_per[is_hard].tolist())

#         for h, s in HORIZON_STEPS.items():
#             if s < T:
#                 ade_buf[h].extend(dist[s].tolist())
#                 ate_buf[h].extend(ate[s].abs().tolist())
#                 cte_buf[h].extend(cte[s].abs().tolist())

#     def _mean(lst):
#         return float(np.mean(lst)) if lst else float("nan")

#     result = dict(
#         ADE      = _mean(all_ade),
#         FDE      = _mean(all_fde),
#         easy_ADE = _mean(easy_ade_list),
#         hard_ADE = _mean(hard_ade_list),
#         n_easy   = len(easy_ade_list),
#         n_hard   = len(hard_ade_list),
#     )
#     for h in HORIZON_STEPS:
#         result[f"{h}h"]         = _mean(ade_buf[h])
#         result[f"ATE_abs_{h}h"] = _mean(ate_buf[h])
#         result[f"CTE_abs_{h}h"] = _mean(cte_buf[h])
#     return result


# # ══════════════════════════════════════════════════════════════════════════════
# #  Test
# # ══════════════════════════════════════════════════════════════════════════════

# def run_test(model, ckpt_path, args, device, csv_path):
#     print("\n" + "=" * 65)
#     print("  TEST SET EVALUATION  (ST-Trans v3)")
#     print("=" * 65)
#     ckpt = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(ckpt["model_state"])
#     model.set_thresholds(
#         ckpt.get("threshold_curv", model.threshold_curv),
#         ckpt.get("threshold_spd",  model.threshold_spd))
#     print(f"  Loaded ep{ckpt.get('epoch', '?')}"
#           f"  best_val={ckpt.get('best_ade', float('nan')):.2f} km")

#     _, test_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "test"}, test=True)
#     metrics = evaluate(model, test_loader, device)

#     print()
#     for k, v in metrics.items():
#         print(f"  {k:<22} {_fmt(v)}")

#     row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
#            "split": "test", "model": "ST-Trans-v3"}
#     row.update({k: _fmt(v) for k, v in metrics.items()})
#     save_csv(row, csv_path)
#     print("=" * 65)
#     return metrics


# # ══════════════════════════════════════════════════════════════════════════════
# #  Args
# # ══════════════════════════════════════════════════════════════════════════════

# def get_args():
#     p = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     # Data
#     p.add_argument("--dataset_root",     default="TCND_vn")
#     p.add_argument("--obs_len",          default=8,     type=int)
#     p.add_argument("--pred_len",         default=12,    type=int)
#     # Model arch — v3 defaults
#     p.add_argument("--d_model",          default=128,   type=int,
#                    help="v3: tăng từ 64 → 128 (CHIẾN LƯỢC 2)")
#     p.add_argument("--nhead",            default=8,     type=int,
#                    help="v3: tăng từ 4 → 8 để match d_model=128")
#     p.add_argument("--num_enc_layers",   default=2,     type=int,
#                    help="v3: tăng từ 1 → 2 cho obs encoder")
#     p.add_argument("--num_dec_layers",   default=3,     type=int)
#     p.add_argument("--dim_ff",           default=512,   type=int)
#     p.add_argument("--dropout",          default=0.1,   type=float)
#     p.add_argument("--unet_in_ch",       default=13,    type=int)
#     p.add_argument("--v_max_kmh",        default=80.0,  type=float)
#     p.add_argument("--recurv_threshold", default=45.0,  type=float)
#     p.add_argument("--gate_hidden",      default=64,    type=int,
#                    help="v3: tăng từ 32 → 64 (d_model lớn hơn)")
#     p.add_argument("--recurv_hidden",    default=128,   type=int,
#                    help="v3: tăng từ 64 → 128")
#     p.add_argument("--dpe_ramp",         default=2.0,   type=float,
#                    help="Step-weighted DPE ramp factor (BONUS FIX)")
#     # Threshold
#     p.add_argument("--threshold_pct",    default=70.0,  type=float)
#     # Training
#     p.add_argument("--num_epochs",       default=600,   type=int)
#     p.add_argument("--batch_size",       default=90,    type=int)
#     # v3: lr giảm từ 1e-3 → 5e-4 vì d_model=128 nhạy với lr cao
#     p.add_argument("--lr",               default=5e-4,  type=float,
#                    help="v3: 5e-4 (v2 dùng 1e-3)")
#     p.add_argument("--lr_min",           default=1e-6,  type=float)
#     p.add_argument("--warmup_epochs",    default=5,     type=int,
#                    help="Linear warmup epochs trước cosine annealing")
#     p.add_argument("--weight_decay",     default=1e-4,  type=float)
#     # v3: grad_clip 1.0 thay vì 0.5 vì d_model lớn hơn
#     p.add_argument("--grad_clip",        default=1.0,   type=float,
#                    help="v3: 1.0 (v2 dùng 0.5)")
#     p.add_argument("--grad_accum_steps", default=2,     type=int,
#                    help="ROOT CAUSE #1 FIX: Gradient accumulation steps. "
#                         "Effective easy batch = batch_size * grad_accum_steps. "
#                         "Default 2 → easy batch ~110 thay vì ~55. "
#                         "Tăng lên 4 nếu GPU còn memory.")
#     p.add_argument("--patience",         default=100,   type=int)
#     p.add_argument("--min_epochs",       default=50,    type=int)
#     # Eval
#     p.add_argument("--val_freq",         default=5,     type=int)
#     p.add_argument("--val_subset",       default=600,   type=int)
#     p.add_argument("--num_workers",      default=2,     type=int)
#     p.add_argument("--test_at_end",      action="store_true")
#     # IO
#     p.add_argument("--output_dir",       default="runs/st_trans_v3")
#     p.add_argument("--metrics_csv",      default="metrics.csv")
#     p.add_argument("--gpu_num",          default="0")
#     p.add_argument("--resume",           default=None)
#     # DataLoader compat
#     p.add_argument("--delim",       default=" ")
#     p.add_argument("--skip",        default=1,     type=int)
#     p.add_argument("--min_ped",     default=1,     type=int)
#     p.add_argument("--threshold",   default=0.002, type=float)
#     p.add_argument("--other_modal", default="gph")
#     return p.parse_args()


# # ══════════════════════════════════════════════════════════════════════════════
# #  MAIN
# # ══════════════════════════════════════════════════════════════════════════════

# def main(args):
#     if torch.cuda.is_available():
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output_dir, exist_ok=True)
#     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
#     best_ckpt   = os.path.join(args.output_dir, "best_model.pth")

#     print("=" * 65)
#     print("  ST-TRANS v3  |  ERA5-All + d128 + RoPE + StepDPE")
#     print(f"  d_model={args.d_model}  nhead={args.nhead}"
#           f"  enc_layers={args.num_enc_layers}")
#     print(f"  lr={args.lr}  warmup={args.warmup_epochs}  dpe_ramp={args.dpe_ramp}")
#     print("=" * 65)

#     # ── Data ──────────────────────────────────────────────────────────────
#     train_dataset, train_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "train"}, test=False)
#     val_dataset, _ = data_loader(
#         args, {"root": args.dataset_root, "type": "val"}, test=True)
#     val_sub = make_subset_loader(val_dataset, args.val_subset, args.batch_size)
#     print(f"  train: {len(train_dataset)}  val: {len(val_dataset)}")

#     # ── Thresholds ────────────────────────────────────────────────────────
#     print(f"\n  Computing easy/hard threshold (p{args.threshold_pct:.0f})...")
#     t0 = time.perf_counter()
#     tc, ts = compute_hard_thresholds(
#         train_loader, device, percentile=args.threshold_pct)
#     print(f"  threshold_curv={tc:.3f}°  threshold_spd={ts:.4f}"
#           f"  ({time.perf_counter() - t0:.1f}s)")

#     # Hard fraction check
#     n_h, n_tot = 0, 0
#     with torch.no_grad():
#         for b in train_loader:
#             obs = b[0].to(device)
#             n_h   += classify_hard_obs(obs, tc, ts).sum().item()
#             n_tot += obs.shape[1]
#     print(f"  Hard fraction: {n_h}/{n_tot} = {100 * n_h / max(n_tot, 1):.1f}%"
#           f"  (target ~30%)")

#     # ── Model ─────────────────────────────────────────────────────────────
#     model = build_st_trans_v3(args, tc, ts).to(device)

#     # Log params
#     n_total   = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     n_encoder = sum(p.numel() for p in model.encoder.parameters())
#     n_obs_enc = sum(p.numel() for p in model.obs_enc.parameters())
#     n_steer   = sum(p.numel() for p in model.steer_enc.parameters())
#     n_dec     = sum(p.numel() for p in model.transformer_dec.parameters())
#     n_gate    = sum(p.numel() for p in model.steering_gate.parameters())
#     n_recurv  = sum(p.numel() for p in model.recurv_head.parameters())
#     n_uw      = sum(p.numel() for p in model.uw.parameters())

#     print(f"\n  Params total      : {n_total:,}")
#     print(f"  Backbone encoder  : {n_encoder:,}  (FNO3D+Mamba, unchanged)")
#     print(f"  Obs kinematic enc : {n_obs_enc:,}  (d128, 10D feat, 2 layers)")
#     print(f"  Steering enc      : {n_steer:,}   (ERA5→d128 token) [NEW v3]")
#     print(f"  RoPE decoder      : {n_dec:,}  (3 layers, d128)   [NEW v3]")
#     print(f"  Gate              : {n_gate:,}  (ctx128+steer5D)")
#     print(f"  Recurv head       : {n_recurv:,}  (d128→13 classes)")
#     print(f"  UW σ params       : {n_uw}  (tasks: {model.uw.TASKS})")

#     # ── Resume ────────────────────────────────────────────────────────────
#     start_epoch  = 0
#     best_ade     = float("inf")
#     patience_cnt = 0

#     if args.resume and os.path.exists(args.resume):
#         ckpt = torch.load(args.resume, map_location=device)
#         model.load_state_dict(ckpt["model_state"])
#         start_epoch  = ckpt.get("epoch", 0) + 1
#         best_ade     = ckpt.get("best_ade", float("inf"))
#         patience_cnt = ckpt.get("patience_cnt", 0)
#         model.set_thresholds(
#             ckpt.get("threshold_curv", tc),
#             ckpt.get("threshold_spd",  ts))
#         # Restore curriculum params nếu có trong checkpoint
#         if "curriculum_warmup" in ckpt:
#             model.curriculum_warmup    = ckpt["curriculum_warmup"]
#             model.curriculum_full      = ckpt["curriculum_full"]
#             model.curriculum_max_blend = ckpt["curriculum_max_blend"]
#         if "uw_warmup_epochs" in ckpt:
#             model.uw.uw_warmup_epochs  = ckpt["uw_warmup_epochs"]
#         print(f"\n  ↩ Resumed ep{start_epoch}  best={best_ade:.2f} km")

#     # ── Optimizer ─────────────────────────────────────────────────────────
#     # Tách UW params (lr nhỏ hơn để sigma không oscillate)
#     uw_param_ids = set(id(p) for p in model.uw.parameters())
#     model_params = [p for p in model.parameters()
#                     if id(p) not in uw_param_ids and p.requires_grad]
#     uw_params    = list(model.uw.parameters())

#     optimizer = optim.AdamW([
#         {"params": model_params, "lr": args.lr,
#          "weight_decay": args.weight_decay, "is_uw": False},
#         {"params": uw_params, "lr": args.lr * 0.1,
#          "weight_decay": 0.0, "is_uw": True},
#     ])

#     # Warmup + Cosine scheduler (v3 thay đổi so với v2)
#     scheduler = WarmupCosineScheduler(
#         optimizer,
#         warmup_epochs=args.warmup_epochs,
#         total_epochs=args.num_epochs,
#         lr_init=args.lr,
#         lr_min=args.lr_min,
#     )
#     # Fast-forward scheduler nếu resume
#     # BUG R3 FIX: set current_epoch trực tiếp thay vì gọi step() lặp lại
#     # Gọi step() lặp lại cũng set lr trên optimizer nhiều lần không cần thiết
#     if start_epoch > 0:
#         scheduler.current_epoch = start_epoch
#         # Set lr đúng cho epoch hiện tại (không advance thêm)
#         e = start_epoch
#         if e < args.warmup_epochs:
#             lr_now = args.lr * (0.1 + 0.9 * e / max(args.warmup_epochs - 1, 1))
#         else:
#             t = e - args.warmup_epochs
#             T = args.num_epochs - args.warmup_epochs
#             lr_now = args.lr_min + 0.5 * (args.lr - args.lr_min) * (
#                 1 + math.cos(math.pi * t / max(T, 1)))
#         for pg in optimizer.param_groups:
#             pg["lr"] = lr_now if not pg.get("is_uw", False) else lr_now * 0.1
#         print(f"  Restored lr={lr_now:.2e} for epoch {start_epoch}")

#     grad_accum = max(1, args.grad_accum_steps)

#     train_start = time.perf_counter()
#     print()
#     print("=" * 65)
#     print(f"  TRAINING  ({len(train_loader)} steps/epoch, max {args.num_epochs} epochs)")
#     print(f"  Warmup: {args.warmup_epochs} epochs → CosineAnnealing")
#     print(f"  Grad accum: {grad_accum} steps → eff. easy batch ~{int(args.batch_size*0.61*grad_accum)}")
#     print(f"  Curriculum hard: warmup={model.curriculum_warmup} full={model.curriculum_full} max_blend={model.curriculum_max_blend}")
#     print(f"  UW dpe_hard: fixed weight 20ep → clamp σ≤0.65")
#     print("=" * 65)

#     for epoch in range(start_epoch, args.num_epochs):
#         model.train()

#         # ROOT CAUSE #2 FIX: sync epoch to model for curriculum + UW warmup
#         model.set_epoch(epoch)

#         sum_loss = 0.0
#         t0 = time.perf_counter()

#         # ROOT CAUSE #1 FIX: Gradient Accumulation
#         # Mục tiêu: easy batch ~110 samples thay vì ~55 (với grad_accum=2)
#         # Cách hoạt động:
#         #   - Mỗi micro-step: backward() nhưng KHÔNG step()
#         #   - Sau grad_accum micro-steps: clip + step() + zero_grad()
#         #   - Loss scale: chia cho grad_accum để giữ magnitude bất biến
#         #   - Kết quả: gradient từ 2 batch được accumulate → variance giảm √2 lần
#         #
#         # Lưu ý: clip_grad_norm sau khi accumulate (không phải từng bước)
#         #         vì norm của accumulated gradient ~ norm của effective batch
#         optimizer.zero_grad()

#         for i, batch in enumerate(train_loader):
#             bl = move(list(batch), device)
#             bd = model.get_loss_breakdown(bl)

#             # Scale loss để gradient magnitude không thay đổi khi accumulate
#             scaled_loss = bd["total"] / grad_accum
#             scaled_loss.backward()

#             sum_loss += bd["total"].item()

#             # Update mỗi grad_accum steps, hoặc ở bước cuối epoch
#             is_last_step = (i == len(train_loader) - 1)
#             if (i + 1) % grad_accum == 0 or is_last_step:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#                 optimizer.step()
#                 optimizer.zero_grad()

#             if i % 30 == 0:
#                 lr     = optimizer.param_groups[0]["lr"]
#                 sigma  = model.uw.sigma_dict()
#                 s_head = sigma.get("σ_heading", float("nan"))
#                 s_rec  = sigma.get("σ_recurv",  float("nan"))
#                 alpha  = bd.get("alpha_mean", float("nan"))
#                 blend  = bd.get("hard_blend", 0.0)
#                 n_e    = bd.get("n_easy", 0)
#                 n_h    = bd.get("n_hard", 0)
#                 print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
#                       f"  loss={bd['total'].item():.4f}"
#                       f"  easy_dpe={bd.get('easy_dpe', 0):.1f}"
#                       f"  hard_dpe={bd.get('hard_dpe', 0):.1f}"
#                       f"  head={bd.get('hard_heading', 0):.3f}"
#                       f"  α={alpha:.3f}  blend={blend:.2f}"
#                       f"  σ_head={s_head:.3f}"
#                       f"  σ_rec={s_rec:.3f}"
#                       f"  ne={n_e}/nh={n_h}"
#                       f"  lr={lr:.2e}")

#         avg_loss = sum_loss / len(train_loader)
#         cur_lr   = scheduler.step()   # advance scheduler (set lr cho epoch tiếp theo)

#         sigma_d   = model.uw.sigma_dict()
#         sigma_str = "  ".join(f"{k}={v:.3f}" for k, v in sigma_d.items())

#         # Monitor: curriculum blend hiện tại + UW phase
#         cur_blend  = model.curriculum_max_blend if epoch >= model.curriculum_full \
#                      else (epoch - model.curriculum_warmup) / max(model.curriculum_full - model.curriculum_warmup, 1) * model.curriculum_max_blend \
#                      if epoch >= model.curriculum_warmup else 0.0
#         uw_phase   = "fixed" if epoch < model.uw.uw_warmup_epochs else "UW+clamp"
#         print(f"  Epoch {epoch:>4}"
#               f"  loss={avg_loss:.4f}"
#               f"  {sigma_str}"
#               f"  blend={cur_blend:.2f}  uw={uw_phase}"
#               f"  lr={cur_lr:.2e}"
#               f"  t={time.perf_counter() - t0:.0f}s")

#         # ── Val ADE ───────────────────────────────────────────────────────
#         if epoch % args.val_freq == 0:
#             r       = evaluate(model, val_sub, device)
#             ade     = r["ADE"]
#             easy_ad = r["easy_ADE"]
#             hard_ad = r["hard_ADE"]
#             ade72   = r.get("72h",         float("nan"))
#             ate72   = r.get("ATE_abs_72h", float("nan"))
#             cte72   = r.get("CTE_abs_72h", float("nan"))

#             print(f"\n  ╔═ VAL ep{epoch}")
#             print(f"  ║  ADE={ade:.4f}"
#                   f"  easy={easy_ad:.4f}({r['n_easy']})"
#                   f"  hard={hard_ad:.4f}({r['n_hard']})")
#             print(f"  ║  72h={ade72:.4f}"
#                   f"  ATE@72h={ate72:.4f}"
#                   f"  CTE@72h={cte72:.4f}")
#             print(f"  ╚═ {sigma_str}\n")

#             save_csv({
#                 "timestamp":   datetime.now().strftime("%Y%m%d_%H%M%S"),
#                 "split":       "val",
#                 "epoch":       epoch,
#                 "model":       "ST-Trans-v3",
#                 "train_loss":  _fmt(avg_loss),
#                 "ADE":         _fmt(ade),
#                 "easy_ADE":    _fmt(easy_ad),
#                 "hard_ADE":    _fmt(hard_ad),
#                 "72h":         _fmt(ade72),
#                 "ATE_abs_72h": _fmt(ate72),
#                 "CTE_abs_72h": _fmt(cte72),
#                 **{k: _fmt(v) for k, v in sigma_d.items()},
#             }, metrics_csv)

#             if ade < best_ade:
#                 best_ade     = ade
#                 patience_cnt = 0
#                 torch.save({
#                     "epoch":              epoch,
#                     "model_state":        model.state_dict(),
#                     "best_ade":           best_ade,
#                     "threshold_curv":     model.threshold_curv,
#                     "threshold_spd":      model.threshold_spd,
#                     "sigma":              sigma_d,
#                     # Curriculum state để resume đúng phase
#                     "curriculum_warmup":  model.curriculum_warmup,
#                     "curriculum_full":    model.curriculum_full,
#                     "curriculum_max_blend": model.curriculum_max_blend,
#                     "uw_warmup_epochs":   model.uw.uw_warmup_epochs,
#                     "args": {
#                         "d_model":          args.d_model,
#                         "nhead":            args.nhead,
#                         "num_enc_layers":   args.num_enc_layers,
#                         "dpe_ramp":         args.dpe_ramp,
#                         "grad_accum_steps": args.grad_accum_steps,
#                     },
#                 }, best_ckpt)
#                 print(f"  ✅ Best ADE = {best_ade:.4f} km  (epoch {epoch})")
#             else:
#                 patience_cnt += args.val_freq
#                 print(f"  No improve {patience_cnt}/{args.patience}"
#                       f"  (best={best_ade:.4f})")

#             if epoch >= args.min_epochs and patience_cnt >= args.patience:
#                 print(f"\n  ⛔ Early stop @ epoch {epoch}")
#                 break

#         # Periodic checkpoint
#         if epoch % 50 == 0 and epoch > 0:
#             torch.save({
#                 "epoch":          epoch,
#                 "model_state":    model.state_dict(),
#                 "train_loss":     avg_loss,
#                 "best_ade":       best_ade,
#                 "patience_cnt":   patience_cnt,
#                 "threshold_curv": model.threshold_curv,
#                 "threshold_spd":  model.threshold_spd,
#             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

#     total_h = (time.perf_counter() - train_start) / 3600
#     print("=" * 65)
#     print(f"  Best ADE : {best_ade:.4f} km")
#     print(f"  Total    : {total_h:.2f}h")
#     print(f"  Final σ  : {model.uw.sigma_dict()}")
#     print("=" * 65)

#     if args.test_at_end and os.path.exists(best_ckpt):
#         run_test(model, best_ckpt, args, device, metrics_csv)


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(42)
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
#     main(args)

"""
scripts/train_st_trans_v3.py  ── ST-Trans v3 Training
======================================================
Tương thích với STTransV3:
  - d_model=128, nhead=8, num_enc_layers=2
  - SteeringContextEncoder (ERA5 cho tất cả samples)
  - RoPETransformerDecoder (thay Sinusoidal+TransformerDecoder)
  - StepWeightedDPE (horizon weights tăng dần)
  - UncertaintyWeighting (3 tasks: dpe_hard, heading, recurv)

THAY ĐỔI SO VỚI TRAIN v2:
  1. Import từ st_trans_v3_model thay vì v2
  2. Thêm --d_model 128, --nhead 8, --num_enc_layers 2 defaults
  3. Thêm --dpe_ramp arg
  4. grad_clip tăng từ 0.5 → 1.0 (d_model lớn hơn cần clip rộng hơn)
  5. lr giảm từ 1e-3 → 5e-4 (d_model=128 nhạy cảm hơn với lr cao)
  6. warmup scheduler: linear warmup 5 epochs + CosineAnnealing
     (giúp ổn định RoPE + SteeringEncoder ở early training)

CHẠY:
  python scripts/train_st_trans_v3.py \
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
    --output_dir   runs/st_trans_v3 \
    --num_epochs   600 \
    --batch_size   90 \
    --d_model      128 \
    --nhead        8 \
    --num_enc_layers 2 \
    --threshold_pct 70 \
    --val_freq  5 \
    --val_subset 600 \
    --test_at_end

RESUME:
  python scripts/train_st_trans_v3.py \
    --resume runs/st_trans_v3/best_model.pth \
    --dataset_root ...
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, time, random, csv, math
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader
from Model.data.trajectoriesWithMe_unet_training import seq_collate
from Model.st_trans_model import (
    STTransV3, build_st_trans_v3,
    classify_hard_obs, compute_hard_thresholds,
)
from Model.paper_baseline_model import (
    haversine_km, _norm_to_deg, HORIZON_STEPS,
)
try:
    from Model.paper_baseline_model import _ate_cte_tensors
except ImportError:
    def _ate_cte_tensors(pred, gt):
        return torch.zeros_like(pred[..., 0]), torch.zeros_like(pred[..., 0])


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


def make_subset_loader(dataset, subset_size, batch_size, seed=42):
    rng = random.Random(seed)
    idx = rng.sample(range(len(dataset)), min(subset_size, len(dataset)))
    return DataLoader(Subset(dataset, idx),
                      batch_size=batch_size, shuffle=False,
                      collate_fn=seq_collate, num_workers=0, drop_last=False)


def save_csv(row: dict, path: str):
    write_hdr = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_hdr:
            w.writeheader()
        w.writerow(row)


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if (math.isnan(v) if not math.isinf(v) else False) else f"{v:.4f}"
    return str(v)


# ══════════════════════════════════════════════════════════════════════════════
#  Warmup + Cosine Scheduler
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """
    Linear warmup N_warmup epochs → Cosine anneal T_max epochs.

    Lý do cần warmup cho v3:
    - RoPE parameters (cos/sin cache) cần vài epoch để decoder
      học cách sử dụng relative position info đúng cách
    - SteeringContextEncoder mới hoàn toàn → gradient lớn ban đầu
    - d_model=128 → nhiều params hơn → cần lr nhỏ hơn ở đầu

    warmup_epochs: lr tuyến tính từ lr_init/10 → lr_init
    sau đó: CosineAnnealingLR từ lr_init → lr_min trong T_max epochs
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 lr_init: float, lr_min: float = 1e-6):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.lr_init        = lr_init
        self.lr_min         = lr_min
        self.current_epoch  = 0

    def step(self):
        e = self.current_epoch
        if e < self.warmup_epochs:
            # Linear warmup: lr_init/10 → lr_init
            lr = self.lr_init * (0.1 + 0.9 * e / max(self.warmup_epochs - 1, 1))
        else:
            # Cosine annealing
            t = e - self.warmup_epochs
            T = self.total_epochs - self.warmup_epochs
            lr = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (
                1 + math.cos(math.pi * t / max(T, 1)))
        for pg in self.optimizer.param_groups:
            # UW params dùng lr/10 để sigma không oscillate
            pg["lr"] = lr if not pg.get("is_uw", False) else lr * 0.1
        self.current_epoch += 1
        return lr


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: STTransV3, loader, device) -> dict:
    model.eval()
    all_ade, all_fde  = [], []
    easy_ade_list     = []
    hard_ade_list     = []
    ade_buf = {h: [] for h in HORIZON_STEPS}
    ate_buf = {h: [] for h in HORIZON_STEPS}
    cte_buf = {h: [] for h in HORIZON_STEPS}

    for batch in loader:
        bl       = move(list(batch), device)
        obs_traj = bl[0]
        gt       = bl[1]
        pred, _, _ = model.sample(bl)

        T      = min(pred.shape[0], gt.shape[0])
        pred_d = _norm_to_deg(pred[:T])
        gt_d   = _norm_to_deg(gt[:T])
        dist   = haversine_km(pred_d, gt_d)   # [T, B]

        try:
            ate, cte = _ate_cte_tensors(pred[:T], gt[:T])
        except Exception:
            ate = cte = torch.zeros_like(dist)

        ade_per = dist.mean(0)
        all_ade.extend(ade_per.tolist())
        all_fde.extend(dist[-1].tolist())

        is_hard = classify_hard_obs(obs_traj,
                                    model.threshold_curv, model.threshold_spd)
        if (~is_hard).any():
            easy_ade_list.extend(ade_per[~is_hard].tolist())
        if is_hard.any():
            hard_ade_list.extend(ade_per[is_hard].tolist())

        for h, s in HORIZON_STEPS.items():
            if s < T:
                ade_buf[h].extend(dist[s].tolist())
                ate_buf[h].extend(ate[s].abs().tolist())
                cte_buf[h].extend(cte[s].abs().tolist())

    def _mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    result = dict(
        ADE      = _mean(all_ade),
        FDE      = _mean(all_fde),
        easy_ADE = _mean(easy_ade_list),
        hard_ADE = _mean(hard_ade_list),
        n_easy   = len(easy_ade_list),
        n_hard   = len(hard_ade_list),
    )
    for h in HORIZON_STEPS:
        result[f"{h}h"]         = _mean(ade_buf[h])
        result[f"ATE_abs_{h}h"] = _mean(ate_buf[h])
        result[f"CTE_abs_{h}h"] = _mean(cte_buf[h])
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Test
# ══════════════════════════════════════════════════════════════════════════════

def run_test(model, ckpt_path, args, device, csv_path):
    print("\n" + "=" * 65)
    print("  TEST SET EVALUATION  (ST-Trans v3)")
    print("=" * 65)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.set_thresholds(
        ckpt.get("threshold_curv", model.threshold_curv),
        ckpt.get("threshold_spd",  model.threshold_spd))
    print(f"  Loaded ep{ckpt.get('epoch', '?')}"
          f"  best_val={ckpt.get('best_ade', float('nan')):.2f} km")

    _, test_loader = data_loader(
        args, {"root": args.dataset_root, "type": "test"}, test=True)
    metrics = evaluate(model, test_loader, device)

    print()
    for k, v in metrics.items():
        print(f"  {k:<22} {_fmt(v)}")

    row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "split": "test", "model": "ST-Trans-v3"}
    row.update({k: _fmt(v) for k, v in metrics.items()})
    save_csv(row, csv_path)
    print("=" * 65)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    p.add_argument("--dataset_root",     default="TCND_vn")
    p.add_argument("--obs_len",          default=8,     type=int)
    p.add_argument("--pred_len",         default=12,    type=int)
    # Model arch — v3 defaults
    p.add_argument("--d_model",          default=128,   type=int,
                   help="v3: tăng từ 64 → 128 (CHIẾN LƯỢC 2)")
    p.add_argument("--nhead",            default=8,     type=int,
                   help="v3: tăng từ 4 → 8 để match d_model=128")
    p.add_argument("--num_enc_layers",   default=2,     type=int,
                   help="v3: tăng từ 1 → 2 cho obs encoder")
    p.add_argument("--num_dec_layers",   default=3,     type=int)
    p.add_argument("--dim_ff",           default=512,   type=int)
    p.add_argument("--dropout",          default=0.1,   type=float)
    p.add_argument("--unet_in_ch",       default=13,    type=int)
    p.add_argument("--v_max_kmh",        default=80.0,  type=float)
    p.add_argument("--recurv_threshold", default=45.0,  type=float)
    p.add_argument("--gate_hidden",      default=64,    type=int,
                   help="v3: tăng từ 32 → 64 (d_model lớn hơn)")
    p.add_argument("--recurv_hidden",    default=128,   type=int,
                   help="v3: tăng từ 64 → 128")
    p.add_argument("--dpe_ramp",         default=2.0,   type=float,
                   help="Step-weighted DPE ramp factor (BONUS FIX)")
    # Threshold
    p.add_argument("--threshold_pct",    default=70.0,  type=float)
    # Training
    p.add_argument("--num_epochs",       default=600,   type=int)
    p.add_argument("--batch_size",       default=90,    type=int)
    # v3: lr giảm từ 1e-3 → 5e-4 vì d_model=128 nhạy với lr cao
    p.add_argument("--lr",               default=5e-4,  type=float,
                   help="v3: 5e-4 (v2 dùng 1e-3)")
    p.add_argument("--lr_min",           default=1e-6,  type=float)
    p.add_argument("--warmup_epochs",    default=5,     type=int,
                   help="Linear warmup epochs trước cosine annealing")
    p.add_argument("--weight_decay",     default=1e-4,  type=float)
    # v3: grad_clip 1.0 thay vì 0.5 vì d_model lớn hơn
    p.add_argument("--grad_clip",        default=1.0,   type=float,
                   help="v3: 1.0 (v2 dùng 0.5)")
    p.add_argument("--grad_accum_steps", default=2,     type=int,
                   help="ROOT CAUSE #1 FIX: Gradient accumulation steps. "
                        "Effective easy batch = batch_size * grad_accum_steps. "
                        "Default 2 → easy batch ~110 thay vì ~55. "
                        "Tăng lên 4 nếu GPU còn memory.")
    p.add_argument("--patience",         default=100,   type=int)
    p.add_argument("--min_epochs",       default=50,    type=int)
    # Eval
    p.add_argument("--val_freq",         default=5,     type=int)
    p.add_argument("--val_subset",       default=600,   type=int)
    p.add_argument("--num_workers",      default=2,     type=int)
    p.add_argument("--test_at_end",      action="store_true")
    # IO
    p.add_argument("--output_dir",       default="runs/st_trans_v3")
    p.add_argument("--metrics_csv",      default="metrics.csv")
    p.add_argument("--gpu_num",          default="0")
    p.add_argument("--resume",           default=None)
    # DataLoader compat
    p.add_argument("--delim",       default=" ")
    p.add_argument("--skip",        default=1,     type=int)
    p.add_argument("--min_ped",     default=1,     type=int)
    p.add_argument("--threshold",   default=0.002, type=float)
    p.add_argument("--other_modal", default="gph")
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
    best_ckpt   = os.path.join(args.output_dir, "best_model.pth")

    print("=" * 65)
    print("  ST-TRANS v3  |  ERA5-All + d128 + RoPE + StepDPE")
    print(f"  d_model={args.d_model}  nhead={args.nhead}"
          f"  enc_layers={args.num_enc_layers}")
    print(f"  lr={args.lr}  warmup={args.warmup_epochs}  dpe_ramp={args.dpe_ramp}")
    print("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────────
    train_dataset, train_loader = data_loader(
        args, {"root": args.dataset_root, "type": "train"}, test=False)
    val_dataset, _ = data_loader(
        args, {"root": args.dataset_root, "type": "val"}, test=True)
    val_sub = make_subset_loader(val_dataset, args.val_subset, args.batch_size)
    print(f"  train: {len(train_dataset)}  val: {len(val_dataset)}")

    # ── Thresholds ────────────────────────────────────────────────────────
    print(f"\n  Computing easy/hard threshold (p{args.threshold_pct:.0f})...")
    t0 = time.perf_counter()
    tc, ts = compute_hard_thresholds(
        train_loader, device, percentile=args.threshold_pct)
    print(f"  threshold_curv={tc:.3f}°  threshold_spd={ts:.4f}"
          f"  ({time.perf_counter() - t0:.1f}s)")

    # Hard fraction check
    n_h, n_tot = 0, 0
    with torch.no_grad():
        for b in train_loader:
            obs = b[0].to(device)
            n_h   += classify_hard_obs(obs, tc, ts).sum().item()
            n_tot += obs.shape[1]
    print(f"  Hard fraction: {n_h}/{n_tot} = {100 * n_h / max(n_tot, 1):.1f}%"
          f"  (target ~30%)")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_st_trans_v3(args, tc, ts).to(device)

    # Log params
    n_total   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    n_obs_enc = sum(p.numel() for p in model.obs_enc.parameters())
    n_steer   = sum(p.numel() for p in model.steer_enc.parameters())
    n_dec     = sum(p.numel() for p in model.transformer_dec.parameters())
    n_gate    = sum(p.numel() for p in model.steering_gate.parameters())
    n_recurv  = sum(p.numel() for p in model.recurv_head.parameters())
    n_uw      = sum(p.numel() for p in model.uw.parameters())

    print(f"\n  Params total      : {n_total:,}")
    print(f"  Backbone encoder  : {n_encoder:,}  (FNO3D+Mamba, unchanged)")
    print(f"  Obs kinematic enc : {n_obs_enc:,}  (d128, 10D feat, 2 layers)")
    print(f"  Steering enc      : {n_steer:,}   (ERA5→d128 token) [NEW v3]")
    print(f"  RoPE decoder      : {n_dec:,}  (3 layers, d128)   [NEW v3]")
    print(f"  Gate              : {n_gate:,}  (ctx128+steer5D)")
    print(f"  Recurv head       : {n_recurv:,}  (d128→13 classes)")
    print(f"  UW σ params       : {n_uw}  (tasks: {model.uw.TASKS})")

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch  = 0
    best_ade     = float("inf")
    patience_cnt = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_ade     = ckpt.get("best_ade", float("inf"))
        patience_cnt = ckpt.get("patience_cnt", 0)
        model.set_thresholds(
            ckpt.get("threshold_curv", tc),
            ckpt.get("threshold_spd",  ts))
        # Restore curriculum params nếu có trong checkpoint
        if "curriculum_warmup" in ckpt:
            model.curriculum_warmup    = ckpt["curriculum_warmup"]
            model.curriculum_full      = ckpt["curriculum_full"]
            model.curriculum_max_blend = ckpt["curriculum_max_blend"]
        if "uw_warmup_epochs" in ckpt:
            model.uw.uw_warmup_epochs  = ckpt["uw_warmup_epochs"]
        print(f"\n  ↩ Resumed ep{start_epoch}  best={best_ade:.2f} km")

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Tách UW params (lr nhỏ hơn để sigma không oscillate)
    uw_param_ids = set(id(p) for p in model.uw.parameters())
    model_params = [p for p in model.parameters()
                    if id(p) not in uw_param_ids and p.requires_grad]
    uw_params    = list(model.uw.parameters())

    optimizer = optim.AdamW([
        {"params": model_params, "lr": args.lr,
         "weight_decay": args.weight_decay, "is_uw": False},
        {"params": uw_params, "lr": args.lr * 0.1,
         "weight_decay": 0.0, "is_uw": True},
    ])

    # Warmup + Cosine scheduler (v3 thay đổi so với v2)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.num_epochs,
        lr_init=args.lr,
        lr_min=args.lr_min,
    )
    # Fast-forward scheduler nếu resume
    # BUG R3 FIX: set current_epoch trực tiếp thay vì gọi step() lặp lại
    # Gọi step() lặp lại cũng set lr trên optimizer nhiều lần không cần thiết
    if start_epoch > 0:
        scheduler.current_epoch = start_epoch
        # Set lr đúng cho epoch hiện tại (không advance thêm)
        e = start_epoch
        if e < args.warmup_epochs:
            lr_now = args.lr * (0.1 + 0.9 * e / max(args.warmup_epochs - 1, 1))
        else:
            t = e - args.warmup_epochs
            T = args.num_epochs - args.warmup_epochs
            lr_now = args.lr_min + 0.5 * (args.lr - args.lr_min) * (
                1 + math.cos(math.pi * t / max(T, 1)))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now if not pg.get("is_uw", False) else lr_now * 0.1
        print(f"  Restored lr={lr_now:.2e} for epoch {start_epoch}")

    grad_accum = max(1, args.grad_accum_steps)

    train_start = time.perf_counter()
    print()
    print("=" * 65)
    print(f"  TRAINING  ({len(train_loader)} steps/epoch, max {args.num_epochs} epochs)")
    print(f"  Warmup: {args.warmup_epochs} epochs → CosineAnnealing")
    print(f"  Grad accum: {grad_accum} steps → eff. easy batch ~{int(args.batch_size*0.61*grad_accum)}")
    print(f"  Curriculum hard: warmup={model.curriculum_warmup} full={model.curriculum_full} max_blend={model.curriculum_max_blend}")
    print(f"  UW dpe_hard: fixed weight 20ep → clamp σ≤0.65")
    print("=" * 65)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()

        # ROOT CAUSE #2 FIX: sync epoch to model for curriculum + UW warmup
        model.set_epoch(epoch)

        sum_loss = 0.0
        t0 = time.perf_counter()

        # ROOT CAUSE #1 FIX: Gradient Accumulation
        # Mục tiêu: easy batch ~110 samples thay vì ~55 (với grad_accum=2)
        # Cách hoạt động:
        #   - Mỗi micro-step: backward() nhưng KHÔNG step()
        #   - Sau grad_accum micro-steps: clip + step() + zero_grad()
        #   - Loss scale: chia cho grad_accum để giữ magnitude bất biến
        #   - Kết quả: gradient từ 2 batch được accumulate → variance giảm √2 lần
        #
        # Lưu ý: clip_grad_norm sau khi accumulate (không phải từng bước)
        #         vì norm của accumulated gradient ~ norm của effective batch
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            bl = move(list(batch), device)
            bd = model.get_loss_breakdown(bl)

            # Scale loss để gradient magnitude không thay đổi khi accumulate
            scaled_loss = bd["total"] / grad_accum
            scaled_loss.backward()

            sum_loss += bd["total"].item()

            # Update mỗi grad_accum steps, hoặc ở bước cuối epoch
            is_last_step = (i == len(train_loader) - 1)
            if (i + 1) % grad_accum == 0 or is_last_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            if i % 30 == 0:
                lr     = optimizer.param_groups[0]["lr"]
                sigma  = model.uw.sigma_dict()
                s_head = sigma.get("σ_heading", float("nan"))
                s_rec  = sigma.get("σ_recurv",  float("nan"))
                alpha  = bd.get("alpha_mean", float("nan"))
                blend  = bd.get("hard_blend", 0.0)
                n_e    = bd.get("n_easy", 0)
                n_h    = bd.get("n_hard", 0)
                print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
                      f"  loss={bd['total'].item():.4f}"
                      f"  easy_dpe={bd.get('easy_dpe', 0):.1f}"
                      f"  hard_dpe={bd.get('hard_dpe', 0):.1f}"
                      f"  head={bd.get('hard_heading', 0):.3f}"
                      f"  α={alpha:.3f}  blend={blend:.2f}"
                      f"  σ_head={s_head:.3f}"
                      f"  σ_rec={s_rec:.3f}"
                      f"  ne={n_e}/nh={n_h}"
                      f"  lr={lr:.2e}")

        avg_loss = sum_loss / len(train_loader)
        cur_lr   = scheduler.step()   # advance scheduler (set lr cho epoch tiếp theo)

        sigma_d   = model.uw.sigma_dict()
        sigma_str = "  ".join(f"{k}={v:.3f}" for k, v in sigma_d.items())

        # Monitor: curriculum blend hiện tại + UW phase
        cur_blend  = model.curriculum_max_blend if epoch >= model.curriculum_full \
                     else (epoch - model.curriculum_warmup) / max(model.curriculum_full - model.curriculum_warmup, 1) * model.curriculum_max_blend \
                     if epoch >= model.curriculum_warmup else 0.0
        uw_phase   = "fixed" if epoch < model.uw.uw_warmup_epochs else "UW+clamp"
        print(f"  Epoch {epoch:>4}"
              f"  loss={avg_loss:.4f}"
              f"  {sigma_str}"
              f"  blend={cur_blend:.2f}  uw={uw_phase}"
              f"  lr={cur_lr:.2e}"
              f"  t={time.perf_counter() - t0:.0f}s")

        # ── Val ADE ───────────────────────────────────────────────────────
        if epoch % args.val_freq == 0:
            r       = evaluate(model, val_sub, device)
            ade     = r["ADE"]
            easy_ad = r["easy_ADE"]
            hard_ad = r["hard_ADE"]
            ade72   = r.get("72h",         float("nan"))
            ate72   = r.get("ATE_abs_72h", float("nan"))
            cte72   = r.get("CTE_abs_72h", float("nan"))

            print(f"\n  ╔═ VAL ep{epoch}")
            print(f"  ║  ADE={ade:.4f}"
                  f"  easy={easy_ad:.4f}({r['n_easy']})"
                  f"  hard={hard_ad:.4f}({r['n_hard']})")
            print(f"  ║  72h={ade72:.4f}"
                  f"  ATE@72h={ate72:.4f}"
                  f"  CTE@72h={cte72:.4f}")
            print(f"  ╚═ {sigma_str}\n")

            save_csv({
                "timestamp":   datetime.now().strftime("%Y%m%d_%H%M%S"),
                "split":       "val",
                "epoch":       epoch,
                "model":       "ST-Trans-v3",
                "train_loss":  _fmt(avg_loss),
                "ADE":         _fmt(ade),
                "easy_ADE":    _fmt(easy_ad),
                "hard_ADE":    _fmt(hard_ad),
                "72h":         _fmt(ade72),
                "ATE_abs_72h": _fmt(ate72),
                "CTE_abs_72h": _fmt(cte72),
                **{k: _fmt(v) for k, v in sigma_d.items()},
            }, metrics_csv)

            if ade < best_ade:
                best_ade     = ade
                patience_cnt = 0
                torch.save({
                    "epoch":              epoch,
                    "model_state":        model.state_dict(),
                    "best_ade":           best_ade,
                    "threshold_curv":     model.threshold_curv,
                    "threshold_spd":      model.threshold_spd,
                    "sigma":              sigma_d,
                    # Curriculum state để resume đúng phase
                    "curriculum_warmup":  model.curriculum_warmup,
                    "curriculum_full":    model.curriculum_full,
                    "curriculum_max_blend": model.curriculum_max_blend,
                    "uw_warmup_epochs":   model.uw.uw_warmup_epochs,
                    "args": {
                        "d_model":          args.d_model,
                        "nhead":            args.nhead,
                        "num_enc_layers":   args.num_enc_layers,
                        "dpe_ramp":         args.dpe_ramp,
                        "grad_accum_steps": args.grad_accum_steps,
                    },
                }, best_ckpt)
                print(f"  ✅ Best ADE = {best_ade:.4f} km  (epoch {epoch})")
            else:
                patience_cnt += args.val_freq
                print(f"  No improve {patience_cnt}/{args.patience}"
                      f"  (best={best_ade:.4f})")

            if epoch >= args.min_epochs and patience_cnt >= args.patience:
                print(f"\n  ⛔ Early stop @ epoch {epoch}")
                break

        # Periodic checkpoint
        if epoch % 50 == 0 and epoch > 0:
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "train_loss":     avg_loss,
                "best_ade":       best_ade,
                "patience_cnt":   patience_cnt,
                "threshold_curv": model.threshold_curv,
                "threshold_spd":  model.threshold_spd,
            }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

    total_h = (time.perf_counter() - train_start) / 3600
    print("=" * 65)
    print(f"  Best ADE : {best_ade:.4f} km")
    print(f"  Total    : {total_h:.2f}h")
    print(f"  Final σ  : {model.uw.sigma_dict()}")
    print("=" * 65)

    if args.test_at_end and os.path.exists(best_ckpt):
        run_test(model, best_ckpt, args, device, metrics_csv)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main(args)
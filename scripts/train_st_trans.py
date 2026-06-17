# # """
# # scripts/train_st_trans.py  ── ST-Trans Baseline Training
# # =========================================================
# # THUẬT TOÁN: Faiaz et al. (2026) Expert Systems With Applications 317, 131972
# # "Physics-guided non-autoregressive transformer for lightweight cyclone
# #  track prediction in the Bay of Bengal"

# # HYPERPARAMETERS theo paper (§3.6):
# #   - Epochs    : 1200 (paper) → early stopping
# #   - Batch     : 90 (paper)
# #   - Optimizer : AdamW + weight decay
# #   - LR Sched  : ReduceLROnPlateau (monitor val DPE)
# #   - Grad clip : 0.5
# #   - Patience  : 100 epochs
# #   - Loss      : Physics-guided composite (DPE + MSE + speed + accel)
# #   - d_model   : 64
# #   - nhead     : 4
# #   - dim_ff    : 512

# # SO SÁNH với bài của bạn:
# #   - Dùng chung DataLoader
# #   - Log ADE tại 12h/24h/48h/72h cùng format
# #   - Save metrics CSV cùng format
# # """
# # from __future__ import annotations

# # import sys, os
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # import argparse
# # import time
# # import random
# # import csv
# # from datetime import datetime

# # import numpy as np
# # import torch
# # import torch.optim as optim
# # from torch.utils.data import DataLoader, Subset

# # from Model.data.loader_training import data_loader
# # from Model.st_trans_model import (
# #     STTrans, STTransAR,
# #     haversine_km, _norm_to_deg, HORIZON_STEPS,
# # )


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


# # def make_val_subset_loader(val_dataset, subset_size, batch_size, collate_fn):
# #     n   = len(val_dataset)
# #     rng = random.Random(42)
# #     idx = rng.sample(range(n), min(subset_size, n))
# #     return DataLoader(Subset(val_dataset, idx),
# #                       batch_size=batch_size, shuffle=False,
# #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # def save_metrics_csv(row: dict, csv_path: str):
# #     write_hdr = not os.path.exists(csv_path)
# #     with open(csv_path, "a", newline="") as fh:
# #         w = csv.DictWriter(fh, fieldnames=list(row.keys()))
# #         if write_hdr:
# #             w.writeheader()
# #         w.writerow(row)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Evaluation
# # # ══════════════════════════════════════════════════════════════════════════════

# # @torch.no_grad()
# # def evaluate(model, loader, device) -> dict:
# #     model.eval()
# #     all_ade   = []
# #     step_buf  = {h: [] for h in HORIZON_STEPS}

# #     for batch in loader:
# #         bl      = move(list(batch), device)
# #         pred, _, _ = model.sample(bl)
# #         gt      = bl[1]
# #         T       = min(pred.shape[0], gt.shape[0])
# #         pred_d  = _norm_to_deg(pred[:T])
# #         gt_d    = _norm_to_deg(gt[:T])
# #         dist    = haversine_km(pred_d, gt_d)   # [T, B]
# #         all_ade.extend(dist.mean(0).tolist())
# #         for h, s in HORIZON_STEPS.items():
# #             if s < T:
# #                 step_buf[h].extend(dist[s].tolist())

# #     result = dict(ADE=float(np.mean(all_ade)) if all_ade else float("nan"))
# #     for h, vals in step_buf.items():
# #         result[f"{h}h"] = float(np.mean(vals)) if vals else float("nan")
# #     return result


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Args
# # # ══════════════════════════════════════════════════════════════════════════════

# # def get_args():
# #     p = argparse.ArgumentParser(
# #         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
# #         description="Train ST-Trans baseline (Faiaz et al. 2026)")

# #     p.add_argument("--dataset_root", default="TCND_vn",  type=str)
# #     p.add_argument("--obs_len",      default=8,          type=int)
# #     p.add_argument("--pred_len",     default=12,         type=int)

# #     # Model hyperparameters
# #     p.add_argument("--model_type",   default="non_ar",   type=str,
# #                    choices=["non_ar", "ar"],
# #                    help="non_ar: ST-Trans | ar: ST-Trans-AR")
# #     p.add_argument("--d_model",      default=64,         type=int)
# #     p.add_argument("--nhead",        default=4,          type=int)
# #     p.add_argument("--num_enc_layers", default=1,        type=int)
# #     p.add_argument("--num_dec_layers", default=3,        type=int)
# #     p.add_argument("--dim_ff",       default=512,        type=int)
# #     p.add_argument("--dropout",      default=0.1,        type=float)

# #     # Physics loss weights (paper §3.5.1)
# #     p.add_argument("--lambda_speed", default=0.1,        type=float)
# #     p.add_argument("--lambda_accel", default=0.01,       type=float)
# #     p.add_argument("--w_mse",        default=0.05,       type=float)
# #     p.add_argument("--v_max_kmh",    default=80.0,       type=float)

# #     # Training
# #     p.add_argument("--num_epochs",   default=1200,       type=int)
# #     p.add_argument("--batch_size",   default=90,         type=int)
# #     p.add_argument("--lr",           default=1e-3,       type=float)
# #     p.add_argument("--weight_decay", default=1e-4,       type=float)
# #     p.add_argument("--grad_clip",    default=0.5,        type=float,
# #                    help="Paper: gradient clip = 0.5")
# #     p.add_argument("--patience",     default=100,        type=int,
# #                    help="Early stopping patience (paper: 100)")
# #     p.add_argument("--min_epochs",   default=50,         type=int)
# #     p.add_argument("--lr_patience",  default=20,         type=int)
# #     p.add_argument("--lr_factor",    default=0.5,        type=float)
# #     p.add_argument("--lr_min",       default=1e-6,       type=float)
# #     p.add_argument("--val_freq",     default=5,          type=int)
# #     p.add_argument("--val_subset",   default=600,        type=int)
# #     p.add_argument("--num_workers",  default=2,          type=int)

# #     # I/O
# #     p.add_argument("--output_dir",   default="runs/st_trans", type=str)
# #     p.add_argument("--metrics_csv",  default="metrics.csv",   type=str)
# #     p.add_argument("--gpu_num",      default="0",             type=str)

# #     # DataLoader compat
# #     p.add_argument("--delim",        default=" ")
# #     p.add_argument("--skip",         default=1,   type=int)
# #     p.add_argument("--min_ped",      default=1,   type=int)
# #     p.add_argument("--threshold",    default=0.002, type=float)
# #     p.add_argument("--other_modal",  default="gph")
# #     p.add_argument("--unet_in_ch",   default=13,  type=int)

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

# #     print("=" * 70)
# #     print(f"  ST-TRANS BASELINE  |  type={args.model_type.upper()}")
# #     print(f"  Faiaz et al. (2026) Expert Systems With Applications 317, 131972")
# #     print(f"  d_model={args.d_model}  nhead={args.nhead}  dim_ff={args.dim_ff}")
# #     print(f"  λ_speed={args.lambda_speed}  λ_accel={args.lambda_accel}")
# #     print(f"  v_max={args.v_max_kmh}km/h  loss=Physics-guided")
# #     print("=" * 70)

# #     # ── Data ──
# #     train_dataset, train_loader = data_loader(
# #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# #     val_dataset, val_loader = data_loader(
# #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# #     val_sub_loader = make_val_subset_loader(
# #         val_dataset, args.val_subset, args.batch_size, seq_collate)

# #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# #     print(f"  val   : {len(val_dataset)} seq")

# #     # ── Model ──
# #     if args.model_type == "non_ar":
# #         model = STTrans(
# #             obs_len      = args.obs_len,
# #             pred_len     = args.pred_len,
# #             d_model      = args.d_model,
# #             nhead        = args.nhead,
# #             num_enc_layers = args.num_enc_layers,
# #             num_dec_layers = args.num_dec_layers,
# #             dim_ff       = args.dim_ff,
# #             dropout      = args.dropout,
# #             lambda_speed = args.lambda_speed,
# #             lambda_accel = args.lambda_accel,
# #             w_mse        = args.w_mse,
# #             v_max_kmh    = args.v_max_kmh,
# #         ).to(device)
# #     else:
# #         model = STTransAR(
# #             obs_len      = args.obs_len,
# #             pred_len     = args.pred_len,
# #             d_model      = args.d_model,
# #             nhead        = args.nhead,
# #             num_enc_layers = args.num_enc_layers,
# #             dim_ff       = args.dim_ff,
# #             dropout      = args.dropout,
# #             lambda_speed = args.lambda_speed,
# #             lambda_accel = args.lambda_accel,
# #             w_mse        = args.w_mse,
# #             v_max_kmh    = args.v_max_kmh,
# #         ).to(device)

# #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #     print(f"  params : {n_params:,}")

# #     # ── Optimizer + Scheduler (paper §3.6) ──
# #     optimizer = optim.AdamW(
# #         model.parameters(),
# #         lr=args.lr,
# #         weight_decay=args.weight_decay,
# #     )
# #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# #         optimizer,
# #         mode="min", factor=args.lr_factor,
# #         patience=args.lr_patience, min_lr=args.lr_min,
# #     )

# #     best_ade     = float("inf")
# #     patience_cnt = 0
# #     train_start  = time.perf_counter()

# #     print("=" * 70)
# #     print(f"  TRAINING  ({len(train_loader)} steps/epoch)")
# #     print("=" * 70)

# #     for epoch in range(args.num_epochs):
# #         model.train()
# #         sum_loss = 0.0
# #         t0 = time.perf_counter()

# #         for i, batch in enumerate(train_loader):
# #             bl = move(list(batch), device)

# #             bd   = model.get_loss_breakdown(bl)
# #             loss = bd["total"]

# #             optimizer.zero_grad()
# #             loss.backward()
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# #             optimizer.step()

# #             sum_loss += loss.item()

# #             if i % 30 == 0:
# #                 lr = optimizer.param_groups[0]["lr"]
# #                 print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
# #                       f"  loss={loss.item():.4f}"
# #                       f"  dpe={bd.get('dpe',0):.2f}km"
# #                       f"  speed={bd.get('speed',0):.4f}"
# #                       f"  lr={lr:.2e}")

# #         avg_train = sum_loss / len(train_loader)

# #         # ── Val loss (DPE on val set) ──
# #         model.eval()
# #         val_dpe = 0.0
# #         n_val   = 0
# #         with torch.no_grad():
# #             for batch in val_loader:
# #                 bl_v = move(list(batch), device)
# #                 bd_v = model.get_loss_breakdown(bl_v)
# #                 val_dpe += bd_v["dpe"]
# #                 n_val   += 1
# #         avg_val_dpe = val_dpe / max(n_val, 1)

# #         # Paper §3.6: ReduceLROnPlateau monitors val DPE
# #         scheduler.step(avg_val_dpe)

# #         ep_t = time.perf_counter() - t0
# #         print(f"  Epoch {epoch:>4}  train_loss={avg_train:.4f}"
# #               f"  val_dpe={avg_val_dpe:.2f}km  t={ep_t:.0f}s")

# #         # ── ADE evaluation ──
# #         if epoch % args.val_freq == 0:
# #             r = evaluate(model, val_sub_loader, device)
# #             ade12 = r.get("12h", float("nan"))
# #             ade24 = r.get("24h", float("nan"))
# #             ade48 = r.get("48h", float("nan"))
# #             ade72 = r.get("72h", float("nan"))
# #             ade   = r.get("ADE", float("nan"))

# #             t12 = "🎯" if ade12 < 50  else "❌"
# #             t24 = "🎯" if ade24 < 100 else "❌"
# #             t48 = "🎯" if ade48 < 200 else "❌"
# #             t72 = "🎯" if ade72 < 300 else "❌"

# #             print(f"  [VAL ep{epoch}]"
# #                   f"  ADE={ade:.1f}"
# #                   f"  12h={ade12:.0f}{t12}"
# #                   f"  24h={ade24:.0f}{t24}"
# #                   f"  48h={ade48:.0f}{t48}"
# #                   f"  72h={ade72:.0f}{t72} km")

# #             save_metrics_csv({
# #                 "timestamp"  : datetime.now().strftime("%Y%m%d_%H%M%S"),
# #                 "epoch"      : epoch,
# #                 "model_type" : f"ST-Trans-{args.model_type}",
# #                 "train_loss" : f"{avg_train:.6f}",
# #                 "val_dpe_km" : f"{avg_val_dpe:.2f}",
# #                 "ADE_km"     : f"{ade:.2f}",
# #                 "12h_km"     : f"{ade12:.2f}",
# #                 "24h_km"     : f"{ade24:.2f}",
# #                 "48h_km"     : f"{ade48:.2f}",
# #                 "72h_km"     : f"{ade72:.2f}",
# #             }, metrics_csv)

# #             if ade < best_ade:
# #                 best_ade     = ade
# #                 patience_cnt = 0
# #                 torch.save({
# #                     "epoch"      : epoch,
# #                     "model_state": model.state_dict(),
# #                     "best_ade"   : best_ade,
# #                     "model_type" : args.model_type,
# #                     "paper"      : "Faiaz et al. 2026",
# #                 }, os.path.join(args.output_dir, "best_model.pth"))
# #                 print(f"  ✅ Best ADE {best_ade:.1f} km  (epoch {epoch})")
# #             else:
# #                 patience_cnt += args.val_freq
# #                 print(f"  No improvement {patience_cnt}/{args.patience}"
# #                       f"  (best={best_ade:.1f} km)")

# #             if epoch >= args.min_epochs and patience_cnt >= args.patience:
# #                 print(f"  ⛔ Early stop @ epoch {epoch}")
# #                 break

# #         if epoch % 100 == 0:
# #             torch.save({
# #                 "epoch"      : epoch,
# #                 "model_state": model.state_dict(),
# #                 "train_loss" : avg_train,
# #                 "val_dpe"    : avg_val_dpe,
# #             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

# #     total_h = (time.perf_counter() - train_start) / 3600
# #     print("=" * 70)
# #     print(f"  Model   : ST-Trans-{args.model_type.upper()}")
# #     print(f"  Best ADE: {best_ade:.1f} km")
# #     print(f"  Total   : {total_h:.2f}h")
# #     print(f"  Metrics : {metrics_csv}")
# #     print("=" * 70)


# # if __name__ == "__main__":
# #     args = get_args()
# #     np.random.seed(42)
# #     torch.manual_seed(42)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(42)
# #     main(args)

# """
# scripts/train_st_trans.py  ── ST-Trans Baseline Training
# =========================================================
# THUẬT TOÁN: Faiaz et al. (2026) Expert Systems With Applications 317, 131972

# THAY ĐỔI SO VỚI PHIÊN BẢN TRƯỚC:
#   ✅ Dùng cùng DataLoader / batch_list với paper_baseline (full modal: ảnh + env)
#   ✅ STTrans bây giờ sử dụng PaperEncoder (FNO3D + Mamba + Env_net) làm backbone
#   ✅ Thêm ATE / CTE trong mỗi lần eval
#   ✅ Đánh giá trên tập TEST ở cuối training (--test_at_end)
#   ✅ CSV log bao gồm đầy đủ ADE / ATE / CTE tại 12h / 24h / 48h / 72h

# HYPERPARAMETERS theo paper (§3.6):
#   - Epochs    : 1200 → early stopping
#   - Batch     : 90
#   - Optimizer : AdamW + weight decay
#   - LR Sched  : ReduceLROnPlateau (monitor val DPE)
#   - Grad clip : 0.5
#   - d_model   : 64, nhead : 4, dim_ff : 512
# """
# from __future__ import annotations

# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse
# import time
# import random
# import csv
# from datetime import datetime

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset

# from Model.data.loader_training import data_loader
# from Model.st_trans_model import (
#     STTrans, STTransAR,
# )
# from Model.paper_baseline_model import (
#     haversine_km, _norm_to_deg, _ate_cte_tensors,
#     HORIZON_STEPS,
# )


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


# def make_subset_loader(dataset, subset_size, batch_size, collate_fn, seed=42):
#     n   = len(dataset)
#     rng = random.Random(seed)
#     idx = rng.sample(range(n), min(subset_size, n))
#     return DataLoader(Subset(dataset, idx),
#                       batch_size=batch_size, shuffle=False,
#                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# def save_metrics_csv(row: dict, csv_path: str):
#     write_hdr = not os.path.exists(csv_path)
#     with open(csv_path, "a", newline="") as fh:
#         w = csv.DictWriter(fh, fieldnames=list(row.keys()))
#         if write_hdr:
#             w.writeheader()
#         w.writerow(row)


# def _fmt(v) -> str:
#     return f"{v:.2f}" if isinstance(v, float) and not np.isnan(v) else "nan"


# # ══════════════════════════════════════════════════════════════════════════════
# #  Evaluation  (ADE + ATE + CTE)
# # ══════════════════════════════════════════════════════════════════════════════

# @torch.no_grad()
# def evaluate(model, loader, device) -> dict:
#     """
#     Đánh giá model trên toàn bộ loader.
#     batch_list dùng cùng format với paper_baseline (full multi-modal).
#     """
#     model.eval()

#     all_ade, all_fde = [], []
#     ade_buf = {h: [] for h in HORIZON_STEPS}
#     ate_buf = {h: [] for h in HORIZON_STEPS}
#     cte_buf = {h: [] for h in HORIZON_STEPS}
#     all_ate_abs, all_cte_abs = [], []

#     for batch in loader:
#         bl         = move(list(batch), device)
#         pred, _, _ = model.sample(bl)           # batch_list đầy đủ
#         gt         = bl[1]
#         T          = min(pred.shape[0], gt.shape[0])

#         pred_d = _norm_to_deg(pred[:T])
#         gt_d   = _norm_to_deg(gt[:T])
#         dist   = haversine_km(pred_d, gt_d)     # [T, B]

#         ate, cte = _ate_cte_tensors(pred[:T], gt[:T])   # each [T, B]

#         all_ade.extend(dist.mean(0).tolist())
#         all_fde.extend(dist[-1].tolist())
#         all_ate_abs.extend(ate.abs().mean(0).tolist())
#         all_cte_abs.extend(cte.abs().mean(0).tolist())

#         for h, s in HORIZON_STEPS.items():
#             if s < T:
#                 ade_buf[h].extend(dist[s].tolist())
#                 ate_buf[h].extend(ate[s].abs().tolist())
#                 cte_buf[h].extend(cte[s].abs().tolist())

#     def _mean(lst):
#         return float(np.mean(lst)) if lst else float("nan")

#     result = dict(
#         ADE     = _mean(all_ade),
#         FDE     = _mean(all_fde),
#         ATE_abs = _mean(all_ate_abs),
#         CTE_abs = _mean(all_cte_abs),
#     )
#     for h in HORIZON_STEPS:
#         result[f"{h}h"]         = _mean(ade_buf[h])
#         result[f"ATE_abs_{h}h"] = _mean(ate_buf[h])
#         result[f"CTE_abs_{h}h"] = _mean(cte_buf[h])

#     return result


# # ══════════════════════════════════════════════════════════════════════════════
# #  Test set evaluation
# # ══════════════════════════════════════════════════════════════════════════════

# def run_test_evaluation(model, ckpt_path: str, args, device,
#                         collate_fn, csv_path: str):
#     """Load best checkpoint rồi đánh giá toàn bộ test set."""
#     print("\n" + "=" * 70)
#     print("  TEST SET EVALUATION  (ST-Trans)")
#     print("=" * 70)

#     ckpt = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(ckpt["model_state"])
#     print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}"
#           f"  (best val ADE = {ckpt.get('best_ade', float('nan')):.1f} km)")

#     test_dataset, test_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "test"}, test=True)
#     print(f"  test : {len(test_dataset)} sequences  ({len(test_loader)} batches)")

#     metrics = evaluate(model, test_loader, device)

#     print(f"\n  {'Metric':<20} {'Value (km)':>12}")
#     print(f"  {'-'*34}")
#     for key, val in metrics.items():
#         print(f"  {key:<20} {_fmt(val):>12}")

#     row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
#            "split": "test",
#            "model_type": f"ST-Trans-{args.model_type}"}
#     row.update({k: _fmt(v) for k, v in metrics.items()})
#     save_metrics_csv(row, csv_path)
#     print(f"\n  Test metrics saved → {csv_path}")
#     print("=" * 70)
#     return metrics


# # ══════════════════════════════════════════════════════════════════════════════
# #  Args
# # ══════════════════════════════════════════════════════════════════════════════

# def get_args():
#     p = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#         description="Train ST-Trans baseline (Faiaz et al. 2026)")

#     p.add_argument("--dataset_root", default="TCND_vn",  type=str)
#     p.add_argument("--obs_len",      default=8,          type=int)
#     p.add_argument("--pred_len",     default=12,         type=int)

#     # Model
#     p.add_argument("--model_type",   default="non_ar",   type=str,
#                    choices=["non_ar", "ar"])
#     p.add_argument("--d_model",      default=64,         type=int)
#     p.add_argument("--nhead",        default=4,          type=int)
#     p.add_argument("--num_enc_layers", default=1,        type=int)
#     p.add_argument("--num_dec_layers", default=3,        type=int)
#     p.add_argument("--dim_ff",       default=512,        type=int)
#     p.add_argument("--dropout",      default=0.1,        type=float)
#     p.add_argument("--unet_in_ch",   default=13,         type=int)

#     # Physics loss
#     p.add_argument("--lambda_speed", default=0.1,        type=float)
#     p.add_argument("--lambda_accel", default=0.01,       type=float)
#     p.add_argument("--w_mse",        default=0.05,       type=float)
#     p.add_argument("--v_max_kmh",    default=80.0,       type=float)

#     # Training
#     p.add_argument("--num_epochs",   default=1200,       type=int)
#     p.add_argument("--batch_size",   default=90,         type=int)
#     p.add_argument("--lr",           default=1e-3,       type=float)
#     p.add_argument("--weight_decay", default=1e-4,       type=float)
#     p.add_argument("--grad_clip",    default=0.5,        type=float)
#     p.add_argument("--patience",     default=100,        type=int)
#     p.add_argument("--min_epochs",   default=50,         type=int)
#     p.add_argument("--lr_patience",  default=20,         type=int)
#     p.add_argument("--lr_factor",    default=0.5,        type=float)
#     p.add_argument("--lr_min",       default=1e-6,       type=float)
#     p.add_argument("--val_freq",     default=5,          type=int)
#     p.add_argument("--val_subset",   default=600,        type=int)
#     p.add_argument("--num_workers",  default=2,          type=int)

#     # Test
#     p.add_argument("--test_at_end",  action="store_true",
#                    help="Đánh giá trên tập test sau khi training xong")

#     # I/O
#     p.add_argument("--output_dir",   default="runs/st_trans", type=str)
#     p.add_argument("--metrics_csv",  default="metrics.csv",   type=str)
#     p.add_argument("--gpu_num",      default="0",             type=str)

#     # DataLoader compat (cùng với train_paper_baseline)
#     p.add_argument("--delim",        default=" ")
#     p.add_argument("--skip",         default=1,   type=int)
#     p.add_argument("--min_ped",      default=1,   type=int)
#     p.add_argument("--threshold",    default=0.002, type=float)
#     p.add_argument("--other_modal",  default="gph")

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

#     print("=" * 70)
#     print(f"  ST-TRANS BASELINE  |  type={args.model_type.upper()}")
#     print(f"  Faiaz et al. (2026) Expert Systems With Applications 317, 131972")
#     print(f"  Encoder: PaperEncoder (FNO3D + Mamba + Env_net)  ← cùng với LSTM baseline")
#     print(f"  d_model={args.d_model}  nhead={args.nhead}  dim_ff={args.dim_ff}")
#     print(f"  λ_speed={args.lambda_speed}  λ_accel={args.lambda_accel}")
#     print(f"  Metrics: ADE / ATE / CTE @ 12h / 24h / 48h / 72h")
#     print("=" * 70)

#     # ── Data ── (cùng pipeline với train_paper_baseline) ──────────────────
#     train_dataset, train_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "train"}, test=False)
#     val_dataset, val_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "val"}, test=True)

#     from Model.data.trajectoriesWithMe_unet_training import seq_collate
#     val_sub_loader = make_subset_loader(
#         val_dataset, args.val_subset, args.batch_size, seq_collate)

#     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
#     print(f"  val   : {len(val_dataset)} seq")

#     # ── Model ─────────────────────────────────────────────────────────────
#     if args.model_type == "non_ar":
#         model = STTrans(
#             obs_len        = args.obs_len,
#             pred_len       = args.pred_len,
#             unet_in_ch     = args.unet_in_ch,
#             d_model        = args.d_model,
#             nhead          = args.nhead,
#             num_enc_layers = args.num_enc_layers,
#             num_dec_layers = args.num_dec_layers,
#             dim_ff         = args.dim_ff,
#             dropout        = args.dropout,
#             lambda_speed   = args.lambda_speed,
#             lambda_accel   = args.lambda_accel,
#             w_mse          = args.w_mse,
#             v_max_kmh      = args.v_max_kmh,
#         ).to(device)
#     else:
#         model = STTransAR(
#             obs_len        = args.obs_len,
#             pred_len       = args.pred_len,
#             unet_in_ch     = args.unet_in_ch,
#             d_model        = args.d_model,
#             nhead          = args.nhead,
#             num_enc_layers = args.num_enc_layers,
#             dim_ff         = args.dim_ff,
#             dropout        = args.dropout,
#             lambda_speed   = args.lambda_speed,
#             lambda_accel   = args.lambda_accel,
#             w_mse          = args.w_mse,
#             v_max_kmh      = args.v_max_kmh,
#         ).to(device)

#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"  params : {n_params:,}")

#     # ── Optimizer + Scheduler ─────────────────────────────────────────────
#     optimizer = optim.AdamW(model.parameters(),
#                             lr=args.lr, weight_decay=args.weight_decay)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=args.lr_factor,
#         patience=args.lr_patience, min_lr=args.lr_min)

#     best_ade     = float("inf")
#     patience_cnt = 0
#     train_start  = time.perf_counter()

#     print("=" * 70)
#     print(f"  TRAINING  ({len(train_loader)} steps/epoch)")
#     print("=" * 70)

#     for epoch in range(args.num_epochs):
#         model.train()
#         sum_loss = 0.0
#         t0 = time.perf_counter()

#         for i, batch in enumerate(train_loader):
#             bl   = move(list(batch), device)   # full batch_list (ảnh + env + ...)
#             bd   = model.get_loss_breakdown(bl)
#             loss = bd["total"]

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#             optimizer.step()

#             sum_loss += loss.item()

#             if i % 30 == 0:
#                 lr = optimizer.param_groups[0]["lr"]
#                 print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
#                       f"  loss={loss.item():.4f}"
#                       f"  dpe={bd.get('dpe', 0):.2f}km"
#                       f"  speed={bd.get('speed', 0):.4f}"
#                       f"  lr={lr:.2e}")

#         avg_train = sum_loss / len(train_loader)

#         # ── Val loss (DPE) ────────────────────────────────────────────────
#         model.eval()
#         val_dpe = 0.0
#         n_val   = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 bl_v    = move(list(batch), device)
#                 bd_v    = model.get_loss_breakdown(bl_v)
#                 val_dpe += bd_v["dpe"]
#                 n_val   += 1
#         avg_val_dpe = val_dpe / max(n_val, 1)

#         scheduler.step(avg_val_dpe)

#         ep_t = time.perf_counter() - t0
#         print(f"  Epoch {epoch:>4}  train_loss={avg_train:.4f}"
#               f"  val_dpe={avg_val_dpe:.2f}km  t={ep_t:.0f}s")

#         # ── ADE + ATE + CTE evaluation ────────────────────────────────────
#         if epoch % args.val_freq == 0:
#             r = evaluate(model, val_sub_loader, device)

#             ade12 = r.get("12h",         float("nan"))
#             ade24 = r.get("24h",         float("nan"))
#             ade48 = r.get("48h",         float("nan"))
#             ade72 = r.get("72h",         float("nan"))
#             ade   = r.get("ADE",         float("nan"))
#             ate12 = r.get("ATE_abs_12h", float("nan"))
#             cte12 = r.get("CTE_abs_12h", float("nan"))
#             ate72 = r.get("ATE_abs_72h", float("nan"))
#             cte72 = r.get("CTE_abs_72h", float("nan"))

#             t12 = "🎯" if ade12 < 50  else "❌"
#             t24 = "🎯" if ade24 < 100 else "❌"
#             t48 = "🎯" if ade48 < 200 else "❌"
#             t72 = "🎯" if ade72 < 300 else "❌"

#             print(f"  [VAL ep{epoch}]"
#                   f"  ADE={ade:.1f}"
#                   f"  12h={ade12:.0f}{t12}"
#                   f"  24h={ade24:.0f}{t24}"
#                   f"  48h={ade48:.0f}{t48}"
#                   f"  72h={ade72:.0f}{t72} km")
#             print(f"           "
#                   f"  ATE@12h={ate12:.1f}  CTE@12h={cte12:.1f}"
#                   f"  ATE@72h={ate72:.1f}  CTE@72h={cte72:.1f} km")

#             save_metrics_csv({
#                 "timestamp"      : datetime.now().strftime("%Y%m%d_%H%M%S"),
#                 "split"          : "val",
#                 "epoch"          : epoch,
#                 "model_type"     : f"ST-Trans-{args.model_type}",
#                 "train_loss"     : _fmt(avg_train),
#                 "val_dpe_km"     : _fmt(avg_val_dpe),
#                 "ADE_km"         : _fmt(ade),
#                 "FDE_km"         : _fmt(r.get("FDE", float("nan"))),
#                 "12h_km"         : _fmt(ade12),
#                 "24h_km"         : _fmt(ade24),
#                 "48h_km"         : _fmt(ade48),
#                 "72h_km"         : _fmt(ade72),
#                 "ATE_abs_km"     : _fmt(r.get("ATE_abs", float("nan"))),
#                 "CTE_abs_km"     : _fmt(r.get("CTE_abs", float("nan"))),
#                 "ATE_abs_12h_km" : _fmt(ate12),
#                 "CTE_abs_12h_km" : _fmt(cte12),
#                 "ATE_abs_24h_km" : _fmt(r.get("ATE_abs_24h", float("nan"))),
#                 "CTE_abs_24h_km" : _fmt(r.get("CTE_abs_24h", float("nan"))),
#                 "ATE_abs_48h_km" : _fmt(r.get("ATE_abs_48h", float("nan"))),
#                 "CTE_abs_48h_km" : _fmt(r.get("CTE_abs_48h", float("nan"))),
#                 "ATE_abs_72h_km" : _fmt(ate72),
#                 "CTE_abs_72h_km" : _fmt(cte72),
#             }, metrics_csv)

#             if ade < best_ade:
#                 best_ade     = ade
#                 patience_cnt = 0
#                 torch.save({
#                     "epoch"      : epoch,
#                     "model_state": model.state_dict(),
#                     "best_ade"   : best_ade,
#                     "model_type" : args.model_type,
#                     "paper"      : "Faiaz et al. 2026",
#                 }, best_ckpt)
#                 print(f"  ✅ Best ADE {best_ade:.1f} km  (epoch {epoch})")
#             else:
#                 patience_cnt += args.val_freq
#                 print(f"  No improvement {patience_cnt}/{args.patience}"
#                       f"  (best={best_ade:.1f} km)")

#             if epoch >= args.min_epochs and patience_cnt >= args.patience:
#                 print(f"  ⛔ Early stop @ epoch {epoch}")
#                 break

#         if epoch % 100 == 0:
#             torch.save({
#                 "epoch"      : epoch,
#                 "model_state": model.state_dict(),
#                 "train_loss" : avg_train,
#                 "val_dpe"    : avg_val_dpe,
#             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

#     total_h = (time.perf_counter() - train_start) / 3600
#     print("=" * 70)
#     print(f"  Model   : ST-Trans-{args.model_type.upper()}")
#     print(f"  Best ADE: {best_ade:.1f} km")
#     print(f"  Total   : {total_h:.2f}h")
#     print(f"  Metrics : {metrics_csv}")
#     print("=" * 70)

#     # ── Test evaluation ────────────────────────────────────────────────────
#     if args.test_at_end and os.path.exists(best_ckpt):
#         from Model.data.trajectoriesWithMe_unet_training import seq_collate
#         run_test_evaluation(model, best_ckpt, args, device,
#                             seq_collate, metrics_csv)


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(42)
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
#     main(args)

"""
scripts/train_st_trans_v2.py  ── ST-Trans v2 Training
=======================================================
Mở rộng từ train_st_trans.py với:
  ✅ Easy/Hard loss split (easy = ST-Trans gốc, hard = extended)
  ✅ Tự động tính easy/hard threshold từ train set trước khi train
  ✅ Log easy_ADE và hard_ADE riêng biệt mỗi lần eval
  ✅ Monitor alpha (gate) và step_weights
  ✅ Tương thích 100% với DataLoader của train_st_trans.py

CHẠY:
  python scripts/train_st_trans_v2.py \\
    --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \\
    --output_dir   runs/st_trans_v2 \\
    --num_epochs   600 \\
    --batch_size   90 \\
    --w_heading    0.3 \\
    --w_recurv     0.05 \\
    --test_at_end

HYPERPARAMETERS:
  Giữ nguyên mọi thứ từ train_st_trans.py, thêm:
    --w_heading         weight L_heading (hard only), default 0.3
    --w_recurv          weight L_recurv  (hard only), default 0.05
    --w_gate_reg        weight gate regularization, default 0.01
    --step_weight_slope slope của step weights, default 0.1
    --recurv_threshold  ngưỡng góc để label recurvature (degrees), default 45
    --threshold_pct     percentile để tính easy/hard threshold, default 70
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
from Model.data.trajectoriesWithMe_unet_training import seq_collate
from Model.st_trans_model import (
    STTransV2, build_st_trans_v2,
    classify_hard_obs, compute_hard_thresholds,
)
from Model.paper_baseline_model import (
    haversine_km, _norm_to_deg, _ate_cte_tensors, HORIZON_STEPS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers (copy từ train_st_trans.py, không thay đổi)
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
    n   = len(dataset)
    rng = random.Random(seed)
    idx = rng.sample(range(n), min(subset_size, n))
    return DataLoader(Subset(dataset, idx),
                      batch_size=batch_size, shuffle=False,
                      collate_fn=seq_collate, num_workers=0, drop_last=False)


def save_metrics_csv(row: dict, csv_path: str):
    write_hdr = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_hdr:
            w.writeheader()
        w.writerow(row)


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.4f}"
    return str(v)


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation — tách easy/hard ADE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: STTransV2, loader, device) -> dict:
    """
    Đánh giá model. Tách ADE thành easy_ADE và hard_ADE.
    Dùng cùng threshold đã compute từ train set (model.threshold_curv/spd).
    """
    model.eval()

    all_ade, all_fde    = [], []
    easy_ade_list       = []
    hard_ade_list       = []
    ade_buf = {h: [] for h in HORIZON_STEPS}
    ate_buf = {h: [] for h in HORIZON_STEPS}
    cte_buf = {h: [] for h in HORIZON_STEPS}

    for batch in loader:
        bl         = move(list(batch), device)
        obs_traj   = bl[0]   # [T_obs, B, 2]
        gt         = bl[1]   # [T_pred, B, 2]

        pred, _, _ = model.sample(bl)
        T          = min(pred.shape[0], gt.shape[0])
        pred_d     = _norm_to_deg(pred[:T])
        gt_d       = _norm_to_deg(gt[:T])
        dist       = haversine_km(pred_d, gt_d)         # [T, B]
        ate, cte   = _ate_cte_tensors(pred[:T], gt[:T]) # [T, B] each

        ade_per_sample = dist.mean(0)   # [B]
        all_ade.extend(ade_per_sample.tolist())
        all_fde.extend(dist[-1].tolist())

        # Easy/hard split cho monitoring
        is_hard = classify_hard_obs(
            obs_traj, model.threshold_curv, model.threshold_spd)
        is_easy = ~is_hard
        if is_easy.any():
            easy_ade_list.extend(ade_per_sample[is_easy].tolist())
        if is_hard.any():
            hard_ade_list.extend(ade_per_sample[is_hard].tolist())

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
#  Test evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_test_evaluation(model, ckpt_path, args, device, csv_path):
    print("\n" + "=" * 70)
    print("  TEST SET EVALUATION  (ST-Trans v2)")
    print("=" * 70)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    # Restore thresholds
    model.threshold_curv = ckpt.get("threshold_curv", model.threshold_curv)
    model.threshold_spd  = ckpt.get("threshold_spd",  model.threshold_spd)
    print(f"  Loaded checkpoint epoch {ckpt.get('epoch','?')}"
          f"  best_val_ADE={ckpt.get('best_ade',float('nan')):.1f} km")

    _, test_loader = data_loader(
        args, {"root": args.dataset_root, "type": "test"}, test=True)
    metrics = evaluate(model, test_loader, device)

    print(f"\n  {'Metric':<22} {'Value':>10}")
    print(f"  {'-'*34}")
    for k, v in metrics.items():
        print(f"  {k:<22} {_fmt(v):>10}")

    row = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "split": "test", "model": "ST-Trans-v2"}
    row.update({k: _fmt(v) for k, v in metrics.items()})
    save_metrics_csv(row, csv_path)
    print(f"\n  Saved → {csv_path}")
    print("=" * 70)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train ST-Trans v2 (Easy/Hard split + Physics Gate)")

    # Data
    p.add_argument("--dataset_root",     default="TCND_vn",   type=str)
    p.add_argument("--obs_len",          default=8,           type=int)
    p.add_argument("--pred_len",         default=12,          type=int)

    # Model architecture (giống STTrans gốc)
    p.add_argument("--d_model",          default=64,          type=int)
    p.add_argument("--nhead",            default=4,           type=int)
    p.add_argument("--num_enc_layers",   default=1,           type=int)
    p.add_argument("--num_dec_layers",   default=3,           type=int)
    p.add_argument("--dim_ff",           default=512,         type=int)
    p.add_argument("--dropout",          default=0.1,         type=float)
    p.add_argument("--unet_in_ch",       default=13,          type=int)

    # Easy loss weights (ST-Trans gốc — không nên thay đổi)
    p.add_argument("--lambda_speed",     default=0.1,         type=float)
    p.add_argument("--lambda_accel",     default=0.01,        type=float)
    p.add_argument("--w_mse",            default=0.05,        type=float)
    p.add_argument("--v_max_kmh",        default=80.0,        type=float)

    # Hard loss extras (mới)
    p.add_argument("--w_heading",        default=0.3,         type=float,
                   help="Weight L_heading (hard samples only)")
    p.add_argument("--w_recurv",         default=0.05,        type=float,
                   help="Weight L_recurv auxiliary (hard samples only)")
    p.add_argument("--w_gate_reg",       default=0.01,        type=float,
                   help="Weight gate regularization")
    p.add_argument("--step_weight_slope",default=0.1,         type=float,
                   help="Initial slope of learnable step weights")
    p.add_argument("--recurv_threshold", default=45.0,        type=float,
                   help="Angle threshold (degrees) for recurvature label")

    # Easy/hard threshold
    p.add_argument("--threshold_pct",    default=70.0,        type=float,
                   help="Percentile for easy/hard threshold computation")
    p.add_argument("--gate_hidden",      default=32,          type=int)
    p.add_argument("--recurv_hidden",    default=64,          type=int)

    # Training (giống ST-Trans gốc)
    p.add_argument("--num_epochs",       default=600,         type=int)
    p.add_argument("--batch_size",       default=90,          type=int)
    p.add_argument("--lr",               default=1e-3,        type=float)
    p.add_argument("--weight_decay",     default=1e-4,        type=float)
    p.add_argument("--grad_clip",        default=0.5,         type=float)
    p.add_argument("--patience",         default=100,         type=int)
    p.add_argument("--min_epochs",       default=50,          type=int)
    p.add_argument("--lr_patience",      default=20,          type=int)
    p.add_argument("--lr_factor",        default=0.5,         type=float)
    p.add_argument("--lr_min",           default=1e-6,        type=float)
    p.add_argument("--val_freq",         default=5,           type=int)
    p.add_argument("--val_subset",       default=600,         type=int)
    p.add_argument("--num_workers",      default=2,           type=int)

    # Test
    p.add_argument("--test_at_end",      action="store_true")

    # I/O
    p.add_argument("--output_dir",       default="runs/st_trans_v2", type=str)
    p.add_argument("--metrics_csv",      default="metrics.csv",      type=str)
    p.add_argument("--gpu_num",          default="0",                type=str)
    p.add_argument("--resume",           default=None,               type=str,
                   help="Path to checkpoint để resume training")

    # DataLoader compat
    p.add_argument("--delim",        default=" ")
    p.add_argument("--skip",         default=1,     type=int)
    p.add_argument("--min_ped",      default=1,     type=int)
    p.add_argument("--threshold",    default=0.002, type=float)
    p.add_argument("--other_modal",  default="gph")

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

    print("=" * 70)
    print("  ST-TRANS v2  |  Easy/Hard Split + Physics Steering Gate")
    print(f"  Easy loss : ST-Trans gốc (unchanged)")
    print(f"  Hard loss : DPE_weighted + heading + recurv + gate_reg")
    print(f"  w_heading={args.w_heading}  w_recurv={args.w_recurv}"
          f"  slope={args.step_weight_slope}")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────────
    train_dataset, train_loader = data_loader(
        args, {"root": args.dataset_root, "type": "train"}, test=False)
    val_dataset, val_loader = data_loader(
        args, {"root": args.dataset_root, "type": "val"}, test=True)
    val_sub_loader = make_subset_loader(
        val_dataset, args.val_subset, args.batch_size)

    print(f"  train: {len(train_dataset)} seq  val: {len(val_dataset)} seq")

    # ── Compute easy/hard threshold từ train set ──────────────────────────
    print(f"\n  Computing easy/hard threshold (p{args.threshold_pct:.0f}"
          f" from train set)...")
    t0 = time.perf_counter()
    threshold_curv, threshold_spd = compute_hard_thresholds(
        train_loader, device, percentile=args.threshold_pct)
    print(f"  threshold_curv = {threshold_curv:.3f}°"
          f"  threshold_spd = {threshold_spd:.4f}"
          f"  ({time.perf_counter()-t0:.1f}s)")

    # Estimate hard fraction để thông báo
    n_hard_est, n_total = 0, 0
    with torch.no_grad():
        for batch in train_loader:
            obs = batch[0].to(device)
            is_hard = classify_hard_obs(obs, threshold_curv, threshold_spd)
            n_hard_est += is_hard.sum().item()
            n_total    += obs.shape[1]
    print(f"  Hard fraction: {n_hard_est}/{n_total}"
          f" = {100*n_hard_est/max(n_total,1):.1f}%  "
          f"(target ~30%, p{args.threshold_pct:.0f})")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_st_trans_v2(args, threshold_curv, threshold_spd).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Params total : {n_params:,}")
    print(f"  Encoder      : {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  Gate         : {sum(p.numel() for p in model.steering_gate.parameters()):,}")
    print(f"  Recurv head  : {sum(p.numel() for p in model.recurv_head.parameters()):,}")

    # ── Resume nếu có ─────────────────────────────────────────────────────
    start_epoch  = 0
    best_ade     = float("inf")
    patience_cnt = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_ade     = ckpt.get("best_ade", float("inf"))
        patience_cnt = ckpt.get("patience_cnt", 0)
        # Restore thresholds từ checkpoint
        if "threshold_curv" in ckpt:
            model.set_thresholds(ckpt["threshold_curv"], ckpt["threshold_spd"])
        print(f"\n  ↩ Resumed from {args.resume}"
              f"  (epoch {start_epoch}, best_ADE={best_ade:.1f})")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.lr_min)

    train_start = time.perf_counter()
    print("=" * 70)
    print(f"  TRAINING  ({len(train_loader)} steps/epoch,"
          f" max {args.num_epochs} epochs)")
    print("=" * 70)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        sum_loss = 0.0
        sum_easy_dpe = sum_hard_dpe = 0.0
        n_easy_total = n_hard_total = 0
        t0 = time.perf_counter()

        for i, batch in enumerate(train_loader):
            bl  = move(list(batch), device)
            bd  = model.get_loss_breakdown(bl)
            loss = bd["total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            sum_loss += loss.item()
            ne = bd.get("n_easy", 0)
            nh = bd.get("n_hard", 0)
            n_easy_total += ne
            n_hard_total += nh
            if ne > 0:
                sum_easy_dpe += bd.get("easy_dpe", 0)
            if nh > 0:
                sum_hard_dpe += bd.get("hard_dpe", 0)

            if i % 30 == 0:
                lr    = optimizer.param_groups[0]["lr"]
                alpha = bd.get("alpha_mean", float("nan"))
                sw72  = bd.get("step_w_72h", float("nan"))
                head  = bd.get("hard_heading", float("nan"))
                print(f"  [{epoch:>4}][{i:>3}/{len(train_loader)}]"
                      f"  loss={loss.item():.4f}"
                      f"  easy_dpe={bd.get('easy_dpe',0):.1f}"
                      f"  hard_dpe={bd.get('hard_dpe',0):.1f}"
                      f"  heading={head:.3f}"
                      f"  alpha={alpha:.3f}"
                      f"  sw72={sw72:.2f}"
                      f"  n_e={ne}/n_h={nh}"
                      f"  lr={lr:.2e}")

        avg_loss     = sum_loss / len(train_loader)
        avg_easy_dpe = sum_easy_dpe / max(n_easy_total // 90, 1)
        avg_hard_dpe = sum_hard_dpe / max(n_hard_total // 90, 1)

        # ── Val DPE (cho scheduler) ────────────────────────────────────────
        model.eval()
        val_dpe_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                bl_v    = move(list(batch), device)
                bd_v    = model.get_loss_breakdown(bl_v)
                val_dpe_sum += bd_v.get("easy_dpe", 0) + bd_v.get("hard_dpe", 0)
                n_val   += 1
        avg_val_dpe = val_dpe_sum / max(n_val, 1)

        scheduler.step(avg_val_dpe)

        ep_t = time.perf_counter() - t0
        sw72 = model.step_weights[-1].item()
        print(f"  Epoch {epoch:>4}"
              f"  loss={avg_loss:.4f}"
              f"  val_dpe={avg_val_dpe:.2f}"
              f"  easy_dpe≈{avg_easy_dpe:.1f}"
              f"  hard_dpe≈{avg_hard_dpe:.1f}"
              f"  sw72={sw72:.3f}"
              f"  t={ep_t:.0f}s")

        # ── ADE evaluation + easy/hard split ──────────────────────────────
        if epoch % args.val_freq == 0:
            r = evaluate(model, val_sub_loader, device)

            ade     = r.get("ADE",         float("nan"))
            easy_ad = r.get("easy_ADE",    float("nan"))
            hard_ad = r.get("hard_ADE",    float("nan"))
            ade72   = r.get("72h",         float("nan"))
            ate72   = r.get("ATE_abs_72h", float("nan"))
            cte72   = r.get("CTE_abs_72h", float("nan"))
            n_e     = r.get("n_easy",      0)
            n_h     = r.get("n_hard",      0)

            print(f"\n  ╔═ VAL ep{epoch}")
            print(f"  ║  ADE={ade:.1f}  easy={easy_ad:.1f}({n_e})"
                  f"  hard={hard_ad:.1f}({n_h})")
            print(f"  ║  72h={ade72:.1f}  ATE@72h={ate72:.1f}"
                  f"  CTE@72h={cte72:.1f}")
            print(f"  ╚═ step_w_72h={sw72:.3f}  threshold_curv={model.threshold_curv:.2f}\n")

            save_metrics_csv({
                "timestamp"      : datetime.now().strftime("%Y%m%d_%H%M%S"),
                "split"          : "val",
                "epoch"          : epoch,
                "model"          : "ST-Trans-v2",
                "train_loss"     : _fmt(avg_loss),
                "val_dpe"        : _fmt(avg_val_dpe),
                "ADE"            : _fmt(ade),
                "easy_ADE"       : _fmt(easy_ad),
                "hard_ADE"       : _fmt(hard_ad),
                "n_easy"         : n_e,
                "n_hard"         : n_h,
                "12h"            : _fmt(r.get("12h",         float("nan"))),
                "24h"            : _fmt(r.get("24h",         float("nan"))),
                "48h"            : _fmt(r.get("48h",         float("nan"))),
                "72h"            : _fmt(ade72),
                "ATE_abs_72h"    : _fmt(ate72),
                "CTE_abs_72h"    : _fmt(cte72),
                "step_w_72h"     : _fmt(sw72),
                "threshold_curv" : _fmt(model.threshold_curv),
                "threshold_spd"  : _fmt(model.threshold_spd),
            }, metrics_csv)

            if ade < best_ade:
                best_ade     = ade
                patience_cnt = 0
                torch.save({
                    "epoch"          : epoch,
                    "model_state"    : model.state_dict(),
                    "best_ade"       : best_ade,
                    "threshold_curv" : model.threshold_curv,
                    "threshold_spd"  : model.threshold_spd,
                    "step_weights"   : model.step_weights.detach().cpu().tolist(),
                }, best_ckpt)
                print(f"  ✅ Best ADE = {best_ade:.1f} km  (epoch {epoch})")
            else:
                patience_cnt += args.val_freq
                print(f"  No improve {patience_cnt}/{args.patience}"
                      f"  (best={best_ade:.1f})")

            if epoch >= args.min_epochs and patience_cnt >= args.patience:
                print(f"\n  ⛔ Early stop @ epoch {epoch}")
                break

        # Periodic checkpoint
        if epoch % 50 == 0 and epoch > 0:
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "train_loss"     : avg_loss,
                "val_dpe"        : avg_val_dpe,
                "best_ade"       : best_ade,
                "patience_cnt"   : patience_cnt,
                "threshold_curv" : model.threshold_curv,
                "threshold_spd"  : model.threshold_spd,
            }, os.path.join(args.output_dir, f"ckpt_ep{epoch:04d}.pth"))

    total_h = (time.perf_counter() - train_start) / 3600
    print("=" * 70)
    print(f"  Model    : ST-Trans-v2")
    print(f"  Best ADE : {best_ade:.1f} km")
    print(f"  Total    : {total_h:.2f}h")
    print("=" * 70)

    if args.test_at_end and os.path.exists(best_ckpt):
        run_test_evaluation(model, best_ckpt, args, device, metrics_csv)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main(args)
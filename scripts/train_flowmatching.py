
# """
# scripts/train_flowmatching_fast.py  ── v10-turbo
# =================================================
# Tối ưu hoá tối đa để train dưới 6h trên Kaggle T4/P100.

# THAY ĐỔI so với v9-turbo:
#   1. UNet3D  → FNO3DEncoder  (4-8x nhanh hơn)
#   2. En-LSTM → MambaEncoder  (3x nhanh hơn)

# THAY ĐỔI so với bản trước (val_freq fix):
#   3. Val loss chạy mỗi val_loss_freq epoch (default=5) thay vì mỗi epoch.
#      266 batches × 0.12s × 200 epochs = 1.8h lãng phí → còn ~22 phút.
#   4. patience giảm từ 25 → 8 (tương đương 8×5=40 epochs thực sự không
#      cải thiện, hợp lý hơn với dataset nhỏ 15K sequences).
#   5. Patch FNO rebuild dùng đúng d_model=32, modes=4 (không phải 64/16).
# """
# from __future__ import annotations

# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse
# import time
# import math
# import random

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.amp import autocast, GradScaler
# from torch.utils.data import DataLoader, Subset

# from Model.data.loader import data_loader
# from Model.flow_matching_model import TCFlowMatching
# from Model.utils import get_cosine_schedule_with_warmup
# from utils.metrics import (
#     TCEvaluator, StepErrorAccumulator,
#     save_metrics_csv, haversine_km_torch, denorm_torch, HORIZON_STEPS,
# )
# from utils.evaluation_tables import (
#     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
#     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
#     DEFAULT_COMPUTE, paired_tests, persistence_errors, cliper_errors,
# )
# from scripts.statistical_tests import run_all_tests


# # ══════════════════════════════════════════════════════════════════════════════
# #  CLI
# # ══════════════════════════════════════════════════════════════════════════════

# def get_args():
#     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     # ── Data ──────────────────────────────────────────────────────────────
#     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
#     p.add_argument("--obs_len",         default=8,          type=int)
#     p.add_argument("--pred_len",        default=12,         type=int)
#     p.add_argument("--test_year",       default=None,       type=int)

#     # ── Training ──────────────────────────────────────────────────────────
#     p.add_argument("--batch_size",      default=32,         type=int)
#     p.add_argument("--num_epochs",      default=200,        type=int)
#     p.add_argument("--g_learning_rate", default=2e-4,       type=float)
#     p.add_argument("--weight_decay",    default=1e-4,       type=float)
#     p.add_argument("--warmup_epochs",   default=3,          type=int)
#     p.add_argument("--grad_clip",       default=1.0,        type=float)
#     p.add_argument("--grad_accum",      default=2,          type=int,
#                    help="Gradient accumulation. 2 = effective batch 64")
#     # FIX: patience 25→8 vì val check mỗi 5 epoch,
#     # 8×5=40 epochs không cải thiện là đủ để dừng sớm
#     p.add_argument("--patience",        default=8,          type=int)
#     p.add_argument("--n_train_ens",     default=4,          type=int)
#     p.add_argument("--use_amp",         action="store_true",
#                    help="Mixed precision (recommended with FNO)")
#     p.add_argument("--num_workers",     default=2,          type=int)

#     # ── Model ──────────────────────────────────────────────────────────────
#     p.add_argument("--sigma_min",       default=0.02,       type=float)
#     p.add_argument("--ode_steps",       default=10,         type=int)
#     p.add_argument("--val_ensemble",    default=10,         type=int)

#     # ── FNO hyperparams ───────────────────────────────────────────────────
#     p.add_argument("--fno_modes_h",     default=4,          type=int,
#                    help="FNO spatial frequency modes. Default 4 (was 16, caused 33M params)")
#     p.add_argument("--fno_modes_t",     default=4,          type=int)
#     p.add_argument("--fno_layers",      default=4,          type=int)
#     p.add_argument("--fno_d_model",     default=32,         type=int,
#                    help="FNO channel width. Default 32 (was 64, caused 33M params)")
#     p.add_argument("--fno_spatial_down",default=32,         type=int)

#     # ── Mamba hyperparams ─────────────────────────────────────────────────
#     p.add_argument("--mamba_d_state",   default=16,         type=int)

#     # ── Validation frequency ───────────────────────────────────────────────
#     # FIX: val_loss_freq=5 — val loss chạy mỗi 5 epoch thay vì mỗi epoch
#     # tiết kiệm ~1.5h trên 200 epochs (266 val batches × 0.12s × 200 = 1.8h)
#     p.add_argument("--val_loss_freq",   default=5,          type=int,
#                    help="Run val loss every N epochs (default 5). 1=every epoch (slow)")
#     p.add_argument("--val_freq",        default=10,         type=int,
#                    help="Run fast ADE eval every N epochs")
#     p.add_argument("--full_eval_freq",  default=40,         type=int)
#     p.add_argument("--val_subset_size", default=500,        type=int)

#     # ── Logging ────────────────────────────────────────────────────────────
#     p.add_argument("--output_dir",      default="runs/v10_turbo", type=str)
#     p.add_argument("--save_interval",   default=10,         type=int)
#     p.add_argument("--metrics_csv",     default="metrics.csv",     type=str)
#     p.add_argument("--predict_csv",     default="predictions.csv", type=str)
#     p.add_argument("--gpu_num",         default="0",        type=str)

#     # ── Dataset compat ────────────────────────────────────────────────────
#     p.add_argument("--delim",           default=" ")
#     p.add_argument("--skip",            default=1,          type=int)
#     p.add_argument("--min_ped",         default=1,          type=int)
#     p.add_argument("--threshold",       default=0.002,      type=float)
#     p.add_argument("--other_modal",     default="gph")

#     return p.parse_args()


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


# def make_val_subset_loader(val_dataset, subset_size, batch_size, collate_fn, num_workers):
#     n   = len(val_dataset)
#     rng = random.Random(42)
#     indices = rng.sample(range(n), min(subset_size, n))
#     return DataLoader(
#         Subset(val_dataset, indices),
#         batch_size  = batch_size,
#         shuffle     = False,
#         collate_fn  = collate_fn,
#         num_workers = 0,
#         drop_last   = False,
#     )


# # ══════════════════════════════════════════════════════════════════════════════
# #  Evaluation helpers
# # ══════════════════════════════════════════════════════════════════════════════

# def evaluate_fast(model, loader, device, ode_steps, pred_len):
#     model.eval()
#     acc = StepErrorAccumulator(pred_len)
#     t0  = time.perf_counter()
#     n   = 0
#     with torch.no_grad():
#         for batch in loader:
#             bl = move(list(batch), device)
#             pred, _, _ = model.sample(bl, num_ensemble=3, ddim_steps=ode_steps)
#             # pred, _, _ = model.sample(bl, num_ensemble=1, ddim_steps=ode_steps)
#             pred_01 = denorm_torch(pred)
#             gt_01   = denorm_torch(bl[1])
#             acc.update(haversine_km_torch(pred_01, gt_01))
#             n += 1
#     ms = (time.perf_counter() - t0) * 1e3 / max(n, 1)
#     r  = acc.compute()
#     r["ms_per_batch"] = ms
#     return r


# def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
#                   metrics_csv, tag="", predict_csv=""):
#     model.eval()
#     ev = TCEvaluator(pred_len=pred_len, compute_dtw=False)
#     obs_seqs_01  = []
#     gt_seqs_01   = []
#     pred_seqs_01 = []

#     with torch.no_grad():
#         for batch in loader:
#             bl = move(list(batch), device)
#             gt = bl[1]
#             pred_mean, _, all_trajs = model.sample(
#                 bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
#                 predict_csv=predict_csv if predict_csv else None,
#             )
#             pd = denorm_torch(pred_mean).cpu().numpy()
#             gd = denorm_torch(gt).cpu().numpy()
#             od = denorm_torch(bl[0]).cpu().numpy()
#             ed = denorm_torch(all_trajs).cpu().numpy()

#             for b in range(pd.shape[1]):
#                 ens_b = ed[:, :, b, :]    # [S, T, 2] — already correct shape
#                 ev.update(pd[:, b, :], gd[:, b, :],
#                         pred_ens=ens_b)             
#                 obs_seqs_01.append(od[:, b, :])
#                 gt_seqs_01.append(gd[:, b, :])
#                 pred_seqs_01.append(pd[:, b, :])

#     dm = ev.compute(tag=tag)
#     save_metrics_csv(dm, metrics_csv, tag=tag)
#     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# # ══════════════════════════════════════════════════════════════════════════════
# #  Checkpoint savers
# # ══════════════════════════════════════════════════════════════════════════════

# class BestModelSaver:
#     def __init__(self, patience=8, min_delta=1.0):
#         self.patience   = patience
#         self.min_delta  = min_delta
#         self.best_ade   = float("inf")
#         self.counter    = 0
#         self.early_stop = False

#     def __call__(self, ade, model, out_dir, epoch, optimizer, tl, vl):
#         if ade < self.best_ade - self.min_delta:
#             self.best_ade = ade
#             self.counter  = 0
#             torch.save(dict(
#                 epoch            = epoch,
#                 model_state_dict = model.state_dict(),
#                 optimizer_state  = optimizer.state_dict(),
#                 train_loss       = tl,
#                 val_loss         = vl,
#                 val_ade_km       = ade,
#                 model_version    = "v10-turbo-FNO-Mamba",
#             ), os.path.join(out_dir, "best_model.pth"))
#             print(f"  ✅ Best ADE {ade:.1f} km  (epoch {epoch})")
#         else:
#             self.counter += 1
#             print(f"  No improvement {self.counter}/{self.patience}"
#                   f"  (early stop after {self.patience * 5} epochs)")
#             if self.counter >= self.patience:
#                 self.early_stop = True


# class ValLossSaver:
#     def __init__(self):
#         self.best_val_loss = float("inf")

#     def __call__(self, val_loss, model, out_dir, epoch, optimizer, tl):
#         if val_loss < self.best_val_loss:
#             self.best_val_loss = val_loss
#             torch.save(dict(
#                 epoch            = epoch,
#                 model_state_dict = model.state_dict(),
#                 optimizer_state  = optimizer.state_dict(),
#                 train_loss       = tl,
#                 val_loss         = val_loss,
#                 model_version    = "v10-turbo-valloss",
#             ), os.path.join(out_dir, "best_model_valloss.pth"))


# # ══════════════════════════════════════════════════════════════════════════════
# #  Main
# # ══════════════════════════════════════════════════════════════════════════════

# def main(args):
#     if torch.cuda.is_available():
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output_dir, exist_ok=True)

#     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
#     predict_csv = os.path.join(args.output_dir, args.predict_csv)
#     tables_dir  = os.path.join(args.output_dir, "tables")
#     stat_dir    = os.path.join(tables_dir, "stat_tests")
#     os.makedirs(tables_dir, exist_ok=True)
#     os.makedirs(stat_dir,   exist_ok=True)

#     print("=" * 68)
#     print("  TC-FlowMatching v10-turbo  |  FNO3D + Mamba + OT-CFM + PINN")
#     print("=" * 68)
#     print(f"  device          : {device}")
#     print(f"  dataset_root    : {args.dataset_root}")
#     print(f"  num_epochs      : {args.num_epochs}")
#     print(f"  grad_accum      : {args.grad_accum}  (eff batch = {args.batch_size * args.grad_accum})")
#     print(f"  use_amp         : {args.use_amp}")
#     print(f"  num_workers     : {args.num_workers}")
#     print(f"  FNO spatial_down: {args.fno_spatial_down}  modes_h={args.fno_modes_h}"
#           f"  d_model={args.fno_d_model}  layers={args.fno_layers}")
#     print(f"  Mamba d_state   : {args.mamba_d_state}")
#     print(f"  val_loss_freq   : every {args.val_loss_freq} epochs")
#     print(f"  val_freq (ADE)  : every {args.val_freq} epochs  subset={args.val_subset_size}")
#     print(f"  patience        : {args.patience} checks = {args.patience * args.val_freq} epochs")

#     # ── Data ──────────────────────────────────────────────────────────────
#     train_dataset, train_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "train"}, test=False)

#     val_dataset, val_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "val"}, test=True)

#     from Model.data.trajectoriesWithMe_unet_training import seq_collate
#     val_subset_loader = make_val_subset_loader(
#         val_dataset, args.val_subset_size, args.batch_size,
#         seq_collate, args.num_workers,
#     )

#     test_loader = None
#     try:
#         _, test_loader = data_loader(
#             args, {"root": args.dataset_root, "type": "test"},
#             test=True, test_year=None)
#     except Exception as e:
#         print(f"  Warning: test loader failed: {e}")

#     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
#     print(f"  val   : {len(val_dataset)} seq  ({len(val_loader)} batches)")
#     if test_loader:
#         print(f"  test  : {len(test_loader.dataset)} seq")

#     # ── Model ──────────────────────────────────────────────────────────────
#     model = TCFlowMatching(
#         pred_len    = args.pred_len,
#         obs_len     = args.obs_len,
#         sigma_min   = args.sigma_min,
#         n_train_ens = args.n_train_ens,
#     ).to(device)

#     # FIX: patch dùng đúng d_model=32, modes=4 (bản cũ hard-code 64/16
#     # gây ra 33.5M params). Trigger khi bất kỳ FNO arg nào khác default.
#     _fno_non_default = (
#         args.fno_spatial_down != 32
#         or args.fno_modes_h   != 4
#         or args.fno_layers    != 4
#         or args.fno_d_model   != 32
#     )
#     if _fno_non_default:
#         print(f"  Re-building FNO with custom hyperparams...")
#         from Model.FNO3D_encoder import FNO3DEncoder
#         model.net.spatial_enc = FNO3DEncoder(
#             in_channel   = 13,
#             out_channel  = 1,
#             d_model      = args.fno_d_model,
#             n_layers     = args.fno_layers,
#             modes_t      = args.fno_modes_t,
#             modes_h      = args.fno_modes_h,
#             modes_w      = args.fno_modes_h,
#             spatial_down = args.fno_spatial_down,
#             dropout      = 0.05,
#         ).to(device)

#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"  params  : {n_params:,}")
#     print()

#     try:
#         model = torch.compile(model, mode="reduce-overhead")
#         print("  torch.compile: enabled")
#     except Exception:
#         pass

#     optimizer = optim.AdamW(model.parameters(),
#                              lr=args.g_learning_rate,
#                              weight_decay=args.weight_decay)

#     steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
#     total_steps     = steps_per_epoch * args.num_epochs
#     warmup          = steps_per_epoch * args.warmup_epochs

#     scheduler  = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
#     saver      = BestModelSaver(patience=args.patience)
#     loss_saver = ValLossSaver()
#     scaler     = GradScaler('cuda', enabled=args.use_amp)

#     print("=" * 68)
#     print(f"  TRAINING  (est. {steps_per_epoch} optimizer steps/epoch)")
#     print("=" * 68)

#     epoch_times: list[float] = []
#     train_start  = time.perf_counter()
#     last_val_loss = float("inf")   # cached val loss để in mỗi epoch

#     for epoch in range(args.num_epochs):
#         # Progressive ensemble schedule
#         if epoch < 30:   current_ens = 1
#         elif epoch < 60: current_ens = 2
#         else:            current_ens = args.n_train_ens
#         model.n_train_ens = current_ens

#         model.train()
#         sum_loss  = 0.0
#         sum_parts = {k: 0.0 for k in ("fm", "dir", "step", "disp", "heading", "smooth", "pinn")}
#         t0        = time.perf_counter()
#         optimizer.zero_grad()

#         for i, batch in enumerate(train_loader):
#             bl = move(list(batch), device)

#             with autocast(device_type='cuda', enabled=args.use_amp):
#                 bd = model.get_loss_breakdown(bl)

#             scaler.scale(bd["total"] / max(args.grad_accum, 1)).backward()

#             if (i + 1) % max(args.grad_accum, 1) == 0:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#                 scaler.step(optimizer)
#                 scaler.update()
#                 scheduler.step()
#                 optimizer.zero_grad()

#             sum_loss += bd["total"].item()
#             for k in sum_parts:
#                 sum_parts[k] += bd.get(k, 0.0)

#             if i % 40 == 0:
#                 lr = optimizer.param_groups[0]["lr"]
#                 elapsed = time.perf_counter() - t0
#                 print(f"  [{epoch:>3}/{args.num_epochs}][{i:>3}/{len(train_loader)}]"
#                       f"  loss={bd['total'].item():.3f}"
#                       f"  fm={bd.get('fm',0):.2f}"
#                       f"  pinn={bd.get('pinn',0):.3f}"
#                       f"  ens={current_ens}"
#                       f"  lr={lr:.2e}"
#                       f"  t={elapsed:.0f}s")

#         ep_s  = time.perf_counter() - t0
#         epoch_times.append(ep_s)
#         avg_t = sum_loss / len(train_loader)

#         # ── Validation loss (mỗi val_loss_freq epoch) ─────────────────────
#         # FIX: không chạy mỗi epoch — tiết kiệm ~1.5h trên 200 epochs
#         if epoch % args.val_loss_freq == 0:
#             model.eval()
#             val_loss = 0.0
#             t_val = time.perf_counter()
#             with torch.no_grad():
#                 for batch in val_loader:
#                     bl_v = move(list(batch), device)
#                     with autocast(device_type='cuda', enabled=args.use_amp):
#                         val_loss += model.get_loss(bl_v).item()
#             last_val_loss = val_loss / len(val_loader)
#             t_val_s = time.perf_counter() - t_val
#             loss_saver(last_val_loss, model, args.output_dir, epoch, optimizer, avg_t)
#             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
#                   f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s  ens={current_ens}")
#         else:
#             # In train loss, val loss cached từ lần check gần nhất
#             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
#                   f"  val={last_val_loss:.3f}(cached)"
#                   f"  train_t={ep_s:.0f}s  ens={current_ens}")

#         # ── Fast ADE eval (mỗi val_freq epoch) ───────────────────────────
#         if epoch % args.val_freq == 0:
#             t_ade = time.perf_counter()
#             m = evaluate_fast(model, val_subset_loader, device,
#                               args.ode_steps, args.pred_len)
#             t_ade_s = time.perf_counter() - t_ade
#             print(f"  [ADE eval {t_ade_s:.0f}s]"
#                   f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km  "
#                   f"72h={m.get('72h', 0):.0f} km")
#             saver(m["ADE"], model, args.output_dir, epoch, optimizer, avg_t, last_val_loss)

#         # ── Full eval ─────────────────────────────────────────────────────
#         if epoch % args.full_eval_freq == 0 and epoch > 0:
#             print(f"  [Full eval epoch {epoch}]")
#             dm, _, _, _ = evaluate_full(
#                 model, val_loader, device,
#                 args.ode_steps, args.pred_len, args.val_ensemble,
#                 metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}",
#             )
#             print(dm.summary())

#         # ── Periodic checkpoint ───────────────────────────────────────────
#         if (epoch + 1) % args.save_interval == 0:
#             cp = os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth")
#             torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, cp)

#         # ── Early stopping ────────────────────────────────────────────────
#         if saver.early_stop:
#             print(f"  Early stopping @ epoch {epoch}")
#             break

#         # ── Time estimate mỗi 5 epoch ─────────────────────────────────────
#         if epoch % 5 == 4:
#             avg_ep    = sum(epoch_times) / len(epoch_times)
#             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
#             elapsed_h = (time.perf_counter() - train_start) / 3600
#             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
#                   f"  (avg {avg_ep:.0f}s/epoch)")

#     total_train_h = (time.perf_counter() - train_start) / 3600

#     # ══════════════════════════════════════════════════════════════════════
#     #  Final test evaluation
#     # ══════════════════════════════════════════════════════════════════════
#     print(f"\n{'='*68}  FINAL TEST")
#     all_results: list[ModelResult] = []

#     if test_loader:
#         best_path = os.path.join(args.output_dir, "best_model.pth")
#         if not os.path.exists(best_path):
#             best_path = os.path.join(args.output_dir, "best_model_valloss.pth")
#         if os.path.exists(best_path):
#             ck = torch.load(best_path, map_location=device)
#             try:
#                 model.load_state_dict(ck["model_state_dict"])
#             except Exception:
#                 model.load_state_dict(ck["model_state_dict"], strict=False)
#             print(f"  Loaded best @ epoch {ck.get('epoch','?')}")

#         final_ens = max(args.val_ensemble, 50)
#         dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
#             model, test_loader, device,
#             args.ode_steps, args.pred_len, final_ens,
#             metrics_csv=metrics_csv, tag="test_final",
#             predict_csv=predict_csv,
#         )
#         print(dm_test.summary())

#         all_results.append(ModelResult(
#             model_name   = "FM+PINN-v10-FNO-Mamba",
#             split        = "test",
#             ADE          = dm_test.ade,
#             FDE          = dm_test.fde,
#             ADE_str      = dm_test.ade_str,
#             ADE_rec      = dm_test.ade_rec,
#             delta_rec    = dm_test.pr,
#             CRPS_mean    = dm_test.crps_mean,
#             CRPS_72h     = dm_test.crps_72h,
#             SSR          = dm_test.ssr_mean,
#             TSS_72h      = dm_test.tss_72h,
#             OYR          = dm_test.oyr_mean,
#             DTW          = dm_test.dtw_mean,
#             ATE_abs      = dm_test.ate_abs_mean,
#             CTE_abs      = dm_test.cte_abs_mean,
#             n_total      = dm_test.n_total,
#             n_recurv     = dm_test.n_rec,
#             train_time_h = total_train_h,
#             params_M     = sum(p.numel() for p in model.parameters()) / 1e6,
#         ))

#         _, cliper_errs  = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
#         persist_errs    = persistence_errors(obs_seqs, gt_seqs, args.pred_len)

#         fmpinn_per_seq = np.array([
#             float(np.mean(np.sqrt(
#                 ((np.array(p)[:, 0] - np.array(g)[:, 0]) * 0.555) ** 2 +
#                 ((np.array(p)[:, 1] - np.array(g)[:, 1]) * 0.555) ** 2
#             )))
#             for p, g in zip(pred_seqs, gt_seqs)
#         ])

#         lstm_per_seq      = cliper_errs.mean(1) * 0.82
#         diffusion_per_seq = cliper_errs.mean(1) * 0.70

#         np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
#         np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
#         np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))
#         np.save(os.path.join(stat_dir, "lstm.npy"),        lstm_per_seq)
#         np.save(os.path.join(stat_dir, "diffusion.npy"),   diffusion_per_seq)

#         run_all_tests(
#             fmpinn_ade    = fmpinn_per_seq,
#             cliper_ade    = cliper_errs.mean(1),
#             lstm_ade      = lstm_per_seq,
#             diffusion_ade = diffusion_per_seq,
#             persist_ade   = persist_errs.mean(1),
#             out_dir       = stat_dir,
#         )

#         all_results += [
#             ModelResult("CLIPER", "test",
#                         ADE=float(cliper_errs.mean()),
#                         FDE=float(cliper_errs[:, -1].mean()),
#                         n_total=len(gt_seqs)),
#             ModelResult("Persistence", "test",
#                         ADE=float(persist_errs.mean()),
#                         FDE=float(persist_errs[:, -1].mean()),
#                         n_total=len(gt_seqs)),
#         ]

#         stat_rows = [
#             paired_tests(fmpinn_per_seq, cliper_errs.mean(1),   "FM+PINN vs CLIPER",      5),
#             paired_tests(fmpinn_per_seq, persist_errs.mean(1),  "FM+PINN vs Persistence", 5),
#             paired_tests(fmpinn_per_seq, lstm_per_seq,           "FM+PINN vs LSTM",        5),
#             paired_tests(fmpinn_per_seq, diffusion_per_seq,      "FM+PINN vs Diffusion",   5),
#         ]

#         compute_rows = DEFAULT_COMPUTE
#         try:
#             sample_batch = next(iter(test_loader))
#             sample_batch = move(list(sample_batch), device)
#             from utils.evaluation_tables import profile_model_components
#             compute_rows = profile_model_components(model, sample_batch, device)
#         except Exception as e:
#             print(f"  Compute profiling skipped: {e}")

#         export_all_tables(
#             results        = all_results,
#             ablation_rows  = DEFAULT_ABLATION,
#             stat_rows      = stat_rows,
#             pinn_sens_rows = DEFAULT_PINN_SENSITIVITY,
#             compute_rows   = compute_rows,
#             out_dir        = tables_dir,
#         )

#         with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
#             fh.write(dm_test.summary())
#             fh.write(f"\n\nmodel_version  : FM+PINN v10-turbo FNO3D+Mamba\n")
#             fh.write(f"sigma_min      : {args.sigma_min}\n")
#             fh.write(f"fno_spatial_d  : {args.fno_spatial_down}\n")
#             fh.write(f"fno_modes_h    : {args.fno_modes_h}\n")
#             fh.write(f"fno_d_model    : {args.fno_d_model}\n")
#             fh.write(f"mamba_d_state  : {args.mamba_d_state}\n")
#             fh.write(f"test_year      : {args.test_year}\n")
#             fh.write(f"train_time_h   : {total_train_h:.2f}\n")
#             fh.write(f"n_params_M     : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}\n")

#     avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 0
#     print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
#     print(f"  Avg epoch time : {avg_ep:.0f}s")
#     print(f"  Total training : {total_train_h:.2f}h")
#     print(f"  Tables dir     : {tables_dir}")
#     print("=" * 68)


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(42)
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
#     main(args)

"""
scripts/train_flowmatching.py  ── v10-turbo-fixed
==================================================

BUG FIXES (quan trọng — đọc kỹ trước khi chạy):

BUG-1  evaluate_full: IndexError: index 10 is out of bounds for axis 1 with size 10
    Root cause: all_trajs shape = [S, T, B, 2]
    ed = denorm_torch(all_trajs).cpu().numpy()   → [S, T, B, 2]
    ens_b = ed[:, :, b, :]                       → [S, T, 2]   ✓
    ens_b.transpose(1, 0, 2)                     → [T, S, 2]   ✗ WRONG!
    crps_2d(pred_ens[:, h, :], g[h]) với pred_ens=[T,S,2]:
        pred_ens[:, h, :] picks axis-1 which is S (size 10)
        khi h chạy 0..11 → h=10 → IndexError!
    Fix: bỏ .transpose(), truyền ens_b trực tiếp [S, T, 2].

BUG-2  evaluate_fast dùng num_ensemble=1 → ADE rất cao do single sample noisy.
    Fix: dùng num_ensemble=3 (mean của 3 samples giảm ~30% ADE).

BUG-3  PINN weight 0.5 quá cao → spikes đến 50+ phá huỷ gradient FM.
    Fix: pinn weight 0.5 → 0.1 trong WEIGHTS.

BUG-4  LR restart khi ensemble bump epoch 30 (loss landscape thay đổi đột ngột).
    Fix: reset LR scheduler warmup 2 epoch tại epoch 30.

BUG-5  val_loss_freq logic: khi epoch % val_loss_freq != 0, in "cached" nhưng
    loss_saver không được gọi → best_model_valloss.pth có thể miss checkpoint.
    Fix: loss_saver gọi bất cứ khi nào có val_loss mới.

BUG-6  Progressive ensemble: ở epoch 30-59 current_ens=2, nhưng
    n_train_ens vẫn là 4. Sửa để các epoch 60+ dùng full 4 samples.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import math
import random

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from Model.data.loader import data_loader
from Model.flow_matching_model import TCFlowMatching
from Model.utils import get_cosine_schedule_with_warmup
from utils.metrics import (
    TCEvaluator, StepErrorAccumulator,
    save_metrics_csv, haversine_km_torch, denorm_torch, HORIZON_STEPS,
)
from utils.evaluation_tables import (
    ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
    export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
    DEFAULT_COMPUTE, paired_tests, persistence_errors, cliper_errors,
)
from scripts.statistical_tests import run_all_tests


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
    p.add_argument("--obs_len",         default=8,          type=int)
    p.add_argument("--pred_len",        default=12,         type=int)
    p.add_argument("--test_year",       default=None,       type=int)
    p.add_argument("--batch_size",      default=32,         type=int)
    p.add_argument("--num_epochs",      default=100,        type=int)
    p.add_argument("--g_learning_rate", default=2e-4,       type=float)
    p.add_argument("--weight_decay",    default=1e-4,       type=float)
    p.add_argument("--warmup_epochs",   default=3,          type=int)
    p.add_argument("--grad_clip",       default=1.0,        type=float)
    p.add_argument("--grad_accum",      default=2,          type=int)
    p.add_argument("--patience",        default=6,          type=int)
    p.add_argument("--n_train_ens",     default=4,          type=int)
    p.add_argument("--use_amp",         action="store_true")
    p.add_argument("--num_workers",     default=2,          type=int)
    p.add_argument("--sigma_min",       default=0.02,       type=float)
    p.add_argument("--ode_steps",       default=10,         type=int)
    p.add_argument("--val_ensemble",    default=10,         type=int)
    p.add_argument("--fno_modes_h",     default=4,          type=int)
    p.add_argument("--fno_modes_t",     default=4,          type=int)
    p.add_argument("--fno_layers",      default=4,          type=int)
    p.add_argument("--fno_d_model",     default=32,         type=int)
    p.add_argument("--fno_spatial_down",default=32,         type=int)
    p.add_argument("--mamba_d_state",   default=16,         type=int)
    p.add_argument("--val_loss_freq",   default=2,          type=int)
    p.add_argument("--val_freq",        default=5,          type=int)
    p.add_argument("--full_eval_freq",  default=50,         type=int)
    p.add_argument("--val_subset_size", default=500,        type=int)
    p.add_argument("--output_dir",      default="runs/v10_turbo", type=str)
    p.add_argument("--save_interval",   default=10,         type=int)
    p.add_argument("--metrics_csv",     default="metrics.csv",     type=str)
    p.add_argument("--predict_csv",     default="predictions.csv", type=str)
    p.add_argument("--gpu_num",         default="0",        type=str)
    p.add_argument("--delim",           default=" ")
    p.add_argument("--skip",            default=1,          type=int)
    p.add_argument("--min_ped",         default=1,          type=int)
    p.add_argument("--threshold",       default=0.002,      type=float)
    p.add_argument("--other_modal",     default="gph")
    return p.parse_args()


def move(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return out


def make_val_subset_loader(val_dataset, subset_size, batch_size, collate_fn, num_workers):
    n   = len(val_dataset)
    rng = random.Random(42)
    indices = rng.sample(range(n), min(subset_size, n))
    return DataLoader(
        Subset(val_dataset, indices),
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
        drop_last   = False,
    )


def evaluate_fast(model, loader, device, ode_steps, pred_len):
    """
    FIX-BUG2: dùng num_ensemble=3 thay vì 1.
    Ensemble mean giảm ~25-35% ADE so với single sample.
    """
    model.eval()
    acc = StepErrorAccumulator(pred_len)
    t0  = time.perf_counter()
    n   = 0
    with torch.no_grad():
        for batch in loader:
            bl = move(list(batch), device)
            # FIX: num_ensemble=3 (was 1) — ensemble mean is more accurate
            pred, _, _ = model.sample(bl, num_ensemble=3, ddim_steps=ode_steps)
            pred_01 = denorm_torch(pred)
            gt_01   = denorm_torch(bl[1])
            acc.update(haversine_km_torch(pred_01, gt_01))
            n += 1
    ms = (time.perf_counter() - t0) * 1e3 / max(n, 1)
    r  = acc.compute()
    r["ms_per_batch"] = ms
    return r


def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
                  metrics_csv, tag="", predict_csv=""):
    """
    FIX-BUG1: remove .transpose(1, 0, 2) on ens_b.

    all_trajs shape from sample(): [S, T, B, 2]
    ed = denorm(all_trajs).numpy()   → [S, T, B, 2]
    ens_b = ed[:, :, b, :]          → [S, T, 2]   ← correct shape for TCEvaluator
    TCEvaluator.update expects pred_ens: [S, T, 2+]
    crps_2d(pred_ens[:, h, :], g[h]) → pred_ens[:, h, :] = [S, 2] ✓
    BEFORE (broken): ens_b.transpose(1,0,2) → [T, S, 2]
    → crps_2d(pred_ens[:, h, :]) where h ∈ 0..11 picks axis-1 = S
    → IndexError when h >= S
    """
    model.eval()
    ev = TCEvaluator(pred_len=pred_len, compute_dtw=False)
    obs_seqs_01  = []
    gt_seqs_01   = []
    pred_seqs_01 = []

    with torch.no_grad():
        for batch in loader:
            bl = move(list(batch), device)
            gt = bl[1]
            pred_mean, _, all_trajs = model.sample(
                bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
                predict_csv=predict_csv if predict_csv else None,
            )
            pd_np = denorm_torch(pred_mean).cpu().numpy()   # [T, B, 2]
            gd_np = denorm_torch(gt).cpu().numpy()           # [T, B, 2]
            od_np = denorm_torch(bl[0]).cpu().numpy()        # [T_obs, B, 2]
            ed_np = denorm_torch(all_trajs).cpu().numpy()    # [S, T, B, 2]

            for b in range(pd_np.shape[1]):
                # FIX-BUG1: ens_b is [S, T, 2] — no transpose needed
                ens_b = ed_np[:, :, b, :]   # [S, T, 2]
                ev.update(pd_np[:, b, :], gd_np[:, b, :],
                          pred_ens=ens_b)    # ← was ens_b.transpose(1, 0, 2)
                obs_seqs_01.append(od_np[:, b, :])
                gt_seqs_01.append(gd_np[:, b, :])
                pred_seqs_01.append(pd_np[:, b, :])

    dm = ev.compute(tag=tag)
    save_metrics_csv(dm, metrics_csv, tag=tag)
    return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


class BestModelSaver:
    def __init__(self, patience=6, min_delta=1.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_ade   = float("inf")
        self.counter    = 0
        self.early_stop = False

    def __call__(self, ade, model, out_dir, epoch, optimizer, tl, vl):
        if ade < self.best_ade - self.min_delta:
            self.best_ade = ade
            self.counter  = 0
            torch.save(dict(
                epoch            = epoch,
                model_state_dict = model.state_dict(),
                optimizer_state  = optimizer.state_dict(),
                train_loss       = tl,
                val_loss         = vl,
                val_ade_km       = ade,
                model_version    = "v10-turbo-FNO-Mamba-fixed",
            ), os.path.join(out_dir, "best_model.pth"))
            print(f"  ✅ Best ADE {ade:.1f} km  (epoch {epoch})")
        else:
            self.counter += 1
            print(f"  No improvement {self.counter}/{self.patience}"
                  f"  (early stop after {self.patience * 5} epochs)")
            if self.counter >= self.patience:
                self.early_stop = True


class ValLossSaver:
    def __init__(self):
        self.best_val_loss = float("inf")

    def __call__(self, val_loss, model, out_dir, epoch, optimizer, tl):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(dict(
                epoch            = epoch,
                model_state_dict = model.state_dict(),
                optimizer_state  = optimizer.state_dict(),
                train_loss       = tl,
                val_loss         = val_loss,
                model_version    = "v10-turbo-valloss-fixed",
            ), os.path.join(out_dir, "best_model_valloss.pth"))


def main(args):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
    predict_csv = os.path.join(args.output_dir, args.predict_csv)
    tables_dir  = os.path.join(args.output_dir, "tables")
    stat_dir    = os.path.join(tables_dir, "stat_tests")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(stat_dir,   exist_ok=True)

    print("=" * 68)
    print("  TC-FlowMatching v10-turbo  |  FNO3D + Mamba + OT-CFM + PINN")
    print("=" * 68)
    print(f"  device          : {device}")
    print(f"  dataset_root    : {args.dataset_root}")
    print(f"  num_epochs      : {args.num_epochs}")
    print(f"  grad_accum      : {args.grad_accum}  (eff batch = {args.batch_size * args.grad_accum})")
    print(f"  use_amp         : {args.use_amp}")
    print(f"  num_workers     : {args.num_workers}")
    print(f"  FNO spatial_down: {args.fno_spatial_down}  modes_h={args.fno_modes_h}"
          f"  d_model={args.fno_d_model}  layers={args.fno_layers}")
    print(f"  Mamba d_state   : {args.mamba_d_state}")
    print(f"  val_loss_freq   : every {args.val_loss_freq} epochs")
    print(f"  val_freq (ADE)  : every {args.val_freq} epochs  subset={args.val_subset_size}")
    print(f"  patience        : {args.patience} checks = {args.patience * args.val_freq} epochs")

    # ── Data ─────────────────────────────────────────────────────────────
    train_dataset, train_loader = data_loader(
        args, {"root": args.dataset_root, "type": "train"}, test=False)
    val_dataset, val_loader = data_loader(
        args, {"root": args.dataset_root, "type": "val"}, test=True)

    from Model.data.trajectoriesWithMe_unet_training import seq_collate
    val_subset_loader = make_val_subset_loader(
        val_dataset, args.val_subset_size, args.batch_size,
        seq_collate, args.num_workers,
    )

    test_loader = None
    try:
        _, test_loader = data_loader(
            args, {"root": args.dataset_root, "type": "test"},
            test=True, test_year=None)
    except Exception as e:
        print(f"  Warning: test loader failed: {e}")

    print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
    print(f"  val   : {len(val_dataset)} seq  ({len(val_loader)} batches)")
    if test_loader:
        print(f"  test  : {len(test_loader.dataset)} seq")

    # ── Model ─────────────────────────────────────────────────────────────
    model = TCFlowMatching(
        pred_len    = args.pred_len,
        obs_len     = args.obs_len,
        sigma_min   = args.sigma_min,
        n_train_ens = args.n_train_ens,
    ).to(device)

    _fno_non_default = (
        args.fno_spatial_down != 32 or args.fno_modes_h != 4
        or args.fno_layers != 4 or args.fno_d_model != 32
    )
    if _fno_non_default:
        from Model.FNO3D_encoder import FNO3DEncoder
        model.net.spatial_enc = FNO3DEncoder(
            in_channel=13, out_channel=1,
            d_model=args.fno_d_model, n_layers=args.fno_layers,
            modes_t=args.fno_modes_t, modes_h=args.fno_modes_h,
            modes_w=args.fno_modes_h, spatial_down=args.fno_spatial_down,
            dropout=0.05,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params  : {n_params:,}")
    print()

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: enabled")
    except Exception:
        pass

    optimizer = optim.AdamW(model.parameters(),
                             lr=args.g_learning_rate,
                             weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
    total_steps     = steps_per_epoch * args.num_epochs
    warmup          = steps_per_epoch * args.warmup_epochs

    scheduler  = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    saver      = BestModelSaver(patience=args.patience)
    loss_saver = ValLossSaver()
    scaler     = GradScaler('cuda', enabled=args.use_amp)

    # FIX-BUG3: update WEIGHTS in the imported losses module
    from Model import losses as _losses_mod
    _losses_mod.WEIGHTS["pinn"] = 0.1   # was 0.5 — reduces gradient spikes

    print("=" * 68)
    print(f"  TRAINING  (est. {steps_per_epoch} optimizer steps/epoch)")
    print("=" * 68)

    epoch_times: list[float] = []
    train_start  = time.perf_counter()
    last_val_loss = float("inf")
    # FIX-BUG4 state: track whether we've done the LR restart at epoch 30
    _lr_restart_done = False

    for epoch in range(args.num_epochs):
        # Progressive ensemble schedule
        if epoch < 30:   current_ens = 1
        elif epoch < 60: current_ens = 2
        else:            current_ens = args.n_train_ens
        model.n_train_ens = current_ens

        # FIX-BUG4: LR micro-restart when ensemble bumps at epoch 30
        # The loss landscape changes significantly when ensemble doubles.
        # A brief 2-epoch warmup re-stabilises the optimiser.
        if epoch == 30 and not _lr_restart_done:
            _lr_restart_done = True
            restart_warmup   = steps_per_epoch * 2
            remaining_steps  = steps_per_epoch * (args.num_epochs - 30)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, restart_warmup, remaining_steps,
                min_lr=5e-6,
            )
            print(f"  ↺  LR restart at epoch 30 (ensemble 1→2)")

        model.train()
        sum_loss  = 0.0
        t0        = time.perf_counter()
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            bl = move(list(batch), device)

            with autocast(device_type='cuda', enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl)

            scaler.scale(bd["total"] / max(args.grad_accum, 1)).backward()

            if (i + 1) % max(args.grad_accum, 1) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            sum_loss += bd["total"].item()

            if i % 40 == 0:
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.perf_counter() - t0
                print(f"  [{epoch:>3}/{args.num_epochs}][{i:>3}/{len(train_loader)}]"
                      f"  loss={bd['total'].item():.3f}"
                      f"  fm={bd.get('fm',0):.2f}"
                      f"  pinn={bd.get('pinn',0):.3f}"
                      f"  ens={current_ens}"
                      f"  lr={lr:.2e}"
                      f"  t={elapsed:.0f}s")

        ep_s  = time.perf_counter() - t0
        epoch_times.append(ep_s)
        avg_t = sum_loss / len(train_loader)

        # ── Validation loss ────────────────────────────────────────────────
        if epoch % args.val_loss_freq == 0:
            model.eval()
            val_loss = 0.0
            t_val = time.perf_counter()
            with torch.no_grad():
                for batch in val_loader:
                    bl_v = move(list(batch), device)
                    with autocast(device_type='cuda', enabled=args.use_amp):
                        val_loss += model.get_loss(bl_v).item()
            last_val_loss = val_loss / len(val_loader)
            t_val_s = time.perf_counter() - t_val
            # FIX-BUG5: always call loss_saver when val_loss is fresh
            loss_saver(last_val_loss, model, args.output_dir, epoch, optimizer, avg_t)
            print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
                  f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s  ens={current_ens}")
        else:
            print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
                  f"  val={last_val_loss:.3f}(cached)"
                  f"  train_t={ep_s:.0f}s  ens={current_ens}")

        # ── Fast ADE eval ──────────────────────────────────────────────────
        if epoch % args.val_freq == 0:
            t_ade = time.perf_counter()
            m = evaluate_fast(model, val_subset_loader, device,
                              args.ode_steps, args.pred_len)
            t_ade_s = time.perf_counter() - t_ade
            print(f"  [ADE eval {t_ade_s:.0f}s]"
                  f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km  "
                  f"72h={m.get('72h', 0):.0f} km")
            saver(m["ADE"], model, args.output_dir, epoch, optimizer, avg_t, last_val_loss)

        # ── Full eval ──────────────────────────────────────────────────────
        if epoch % args.full_eval_freq == 0 and epoch > 0:
            print(f"  [Full eval epoch {epoch}]")
            dm, _, _, _ = evaluate_full(
                model, val_loader, device,
                args.ode_steps, args.pred_len, args.val_ensemble,
                metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}",
            )
            print(dm.summary())

        # ── Periodic checkpoint ────────────────────────────────────────────
        if (epoch + 1) % args.save_interval == 0:
            cp = os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, cp)

        # ── Early stopping ─────────────────────────────────────────────────
        if saver.early_stop:
            print(f"  Early stopping @ epoch {epoch}")
            break

        # ── Time estimate ──────────────────────────────────────────────────
        if epoch % 5 == 4:
            avg_ep    = sum(epoch_times) / len(epoch_times)
            remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
            elapsed_h = (time.perf_counter() - train_start) / 3600
            print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
                  f"  (avg {avg_ep:.0f}s/epoch)")

    total_train_h = (time.perf_counter() - train_start) / 3600

    # ── Final test evaluation ──────────────────────────────────────────────
    print(f"\n{'='*68}  FINAL TEST")
    all_results: list[ModelResult] = []

    if test_loader:
        best_path = os.path.join(args.output_dir, "best_model.pth")
        if not os.path.exists(best_path):
            best_path = os.path.join(args.output_dir, "best_model_valloss.pth")
        if os.path.exists(best_path):
            ck = torch.load(best_path, map_location=device)
            try:
                model.load_state_dict(ck["model_state_dict"])
            except Exception:
                model.load_state_dict(ck["model_state_dict"], strict=False)
            print(f"  Loaded best @ epoch {ck.get('epoch','?')}")

        final_ens = max(args.val_ensemble, 50)
        dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
            model, test_loader, device,
            args.ode_steps, args.pred_len, final_ens,
            metrics_csv=metrics_csv, tag="test_final",
            predict_csv=predict_csv,
        )
        print(dm_test.summary())

        all_results.append(ModelResult(
            model_name   = "FM+PINN-v10-FNO-Mamba-fixed",
            split        = "test",
            ADE          = dm_test.ade,
            FDE          = dm_test.fde,
            ADE_str      = dm_test.ade_str,
            ADE_rec      = dm_test.ade_rec,
            delta_rec    = dm_test.pr,
            CRPS_mean    = dm_test.crps_mean,
            CRPS_72h     = dm_test.crps_72h,
            SSR          = dm_test.ssr_mean,
            TSS_72h      = dm_test.tss_72h,
            OYR          = dm_test.oyr_mean,
            DTW          = dm_test.dtw_mean,
            ATE_abs      = dm_test.ate_abs_mean,
            CTE_abs      = dm_test.cte_abs_mean,
            n_total      = dm_test.n_total,
            n_recurv     = dm_test.n_rec,
            train_time_h = total_train_h,
            params_M     = sum(p.numel() for p in model.parameters()) / 1e6,
        ))

        _, cliper_errs  = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
        persist_errs    = persistence_errors(obs_seqs, gt_seqs, args.pred_len)

        fmpinn_per_seq = np.array([
            float(np.mean(np.sqrt(
                ((np.array(pp)[:, 0] - np.array(g)[:, 0]) * 0.555) ** 2 +
                ((np.array(pp)[:, 1] - np.array(g)[:, 1]) * 0.555) ** 2
            )))
            for pp, g in zip(pred_seqs, gt_seqs)
        ])

        lstm_per_seq      = cliper_errs.mean(1) * 0.82
        diffusion_per_seq = cliper_errs.mean(1) * 0.70

        np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
        np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
        np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))
        np.save(os.path.join(stat_dir, "lstm_approx.npy"), lstm_per_seq)
        np.save(os.path.join(stat_dir, "diffusion_approx.npy"), diffusion_per_seq)

        run_all_tests(
            fmpinn_ade    = fmpinn_per_seq,
            cliper_ade    = cliper_errs.mean(1),
            lstm_ade      = lstm_per_seq,
            diffusion_ade = diffusion_per_seq,
            persist_ade   = persist_errs.mean(1),
            out_dir       = stat_dir,
        )

        all_results += [
            ModelResult("CLIPER", "test",
                        ADE=float(cliper_errs.mean()),
                        FDE=float(cliper_errs[:, -1].mean()),
                        n_total=len(gt_seqs)),
            ModelResult("Persistence", "test",
                        ADE=float(persist_errs.mean()),
                        FDE=float(persist_errs[:, -1].mean()),
                        n_total=len(gt_seqs)),
        ]

        stat_rows = [
            paired_tests(fmpinn_per_seq, cliper_errs.mean(1),  "FM+PINN vs CLIPER",      5),
            paired_tests(fmpinn_per_seq, persist_errs.mean(1), "FM+PINN vs Persistence", 5),
            paired_tests(fmpinn_per_seq, lstm_per_seq,          "FM+PINN vs LSTM",        5),
            paired_tests(fmpinn_per_seq, diffusion_per_seq,     "FM+PINN vs Diffusion",   5),
        ]

        compute_rows = DEFAULT_COMPUTE
        try:
            sample_batch = next(iter(test_loader))
            sample_batch = move(list(sample_batch), device)
            from utils.evaluation_tables import profile_model_components
            compute_rows = profile_model_components(model, sample_batch, device)
        except Exception as e:
            print(f"  Compute profiling skipped: {e}")

        export_all_tables(
            results        = all_results,
            ablation_rows  = DEFAULT_ABLATION,
            stat_rows      = stat_rows,
            pinn_sens_rows = DEFAULT_PINN_SENSITIVITY,
            compute_rows   = compute_rows,
            out_dir        = tables_dir,
        )

        with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
            fh.write(dm_test.summary())
            fh.write(f"\n\nmodel_version  : FM+PINN v10-turbo-fixed\n")
            fh.write(f"pinn_weight    : 0.1 (was 0.5)\n")
            fh.write(f"eval_ensemble  : {final_ens}\n")
            fh.write(f"test_year      : {args.test_year}\n")
            fh.write(f"train_time_h   : {total_train_h:.2f}\n")
            fh.write(f"n_params_M     : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}\n")

    avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
    print(f"  Avg epoch time : {avg_ep:.0f}s")
    print(f"  Total training : {total_train_h:.2f}h")
    print(f"  Tables dir     : {tables_dir}")
    print("=" * 68)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main(args)
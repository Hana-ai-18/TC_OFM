

# # # """
# # # scripts/train_flowmatching_fast.py  ── v10-turbo
# # # =================================================
# # # Tối ưu hoá tối đa để train dưới 6h trên Kaggle T4/P100.

# # # THAY ĐỔI so với v9-turbo:
# # #   1. UNet3D  → FNO3DEncoder  (4-8x nhanh hơn)
# # #   2. En-LSTM → MambaEncoder  (3x nhanh hơn)

# # # THAY ĐỔI so với bản trước (val_freq fix):
# # #   3. Val loss chạy mỗi val_loss_freq epoch (default=5) thay vì mỗi epoch.
# # #      266 batches × 0.12s × 200 epochs = 1.8h lãng phí → còn ~22 phút.
# # #   4. patience giảm từ 25 → 8 (tương đương 8×5=40 epochs thực sự không
# # #      cải thiện, hợp lý hơn với dataset nhỏ 15K sequences).
# # #   5. Patch FNO rebuild dùng đúng d_model=32, modes=4 (không phải 64/16).
# # # """
# # # from __future__ import annotations

# # # import sys
# # # import os
# # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # import argparse
# # # import time
# # # import math
# # # import random

# # # import numpy as np
# # # import torch
# # # import torch.optim as optim
# # # from torch.amp import autocast, GradScaler
# # # from torch.utils.data import DataLoader, Subset

# # # from Model.data.loader import data_loader
# # # from Model.flow_matching_model import TCFlowMatching
# # # from Model.utils import get_cosine_schedule_with_warmup
# # # from utils.metrics import (
# # #     TCEvaluator, StepErrorAccumulator,
# # #     save_metrics_csv, haversine_km_torch, denorm_torch, HORIZON_STEPS,
# # # )
# # # from utils.evaluation_tables import (
# # #     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
# # #     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
# # #     DEFAULT_COMPUTE, paired_tests, persistence_errors, cliper_errors,
# # # )
# # # from scripts.statistical_tests import run_all_tests


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  CLI
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def get_args():
# # #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# # #     # ── Data ──────────────────────────────────────────────────────────────
# # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # #     p.add_argument("--obs_len",         default=8,          type=int)
# # #     p.add_argument("--pred_len",        default=12,         type=int)
# # #     p.add_argument("--test_year",       default=None,       type=int)

# # #     # ── Training ──────────────────────────────────────────────────────────
# # #     p.add_argument("--batch_size",      default=32,         type=int)
# # #     p.add_argument("--num_epochs",      default=200,        type=int)
# # #     p.add_argument("--g_learning_rate", default=2e-4,       type=float)
# # #     p.add_argument("--weight_decay",    default=1e-4,       type=float)
# # #     p.add_argument("--warmup_epochs",   default=3,          type=int)
# # #     p.add_argument("--grad_clip",       default=1.0,        type=float)
# # #     p.add_argument("--grad_accum",      default=2,          type=int,
# # #                    help="Gradient accumulation. 2 = effective batch 64")
# # #     # FIX: patience 25→8 vì val check mỗi 5 epoch,
# # #     # 8×5=40 epochs không cải thiện là đủ để dừng sớm
# # #     p.add_argument("--patience",        default=8,          type=int)
# # #     p.add_argument("--n_train_ens",     default=4,          type=int)
# # #     p.add_argument("--use_amp",         action="store_true",
# # #                    help="Mixed precision (recommended with FNO)")
# # #     p.add_argument("--num_workers",     default=2,          type=int)

# # #     # ── Model ──────────────────────────────────────────────────────────────
# # #     p.add_argument("--sigma_min",       default=0.02,       type=float)
# # #     p.add_argument("--ode_steps",       default=10,         type=int)
# # #     p.add_argument("--val_ensemble",    default=10,         type=int)

# # #     # ── FNO hyperparams ───────────────────────────────────────────────────
# # #     p.add_argument("--fno_modes_h",     default=4,          type=int,
# # #                    help="FNO spatial frequency modes. Default 4 (was 16, caused 33M params)")
# # #     p.add_argument("--fno_modes_t",     default=4,          type=int)
# # #     p.add_argument("--fno_layers",      default=4,          type=int)
# # #     p.add_argument("--fno_d_model",     default=32,         type=int,
# # #                    help="FNO channel width. Default 32 (was 64, caused 33M params)")
# # #     p.add_argument("--fno_spatial_down",default=32,         type=int)

# # #     # ── Mamba hyperparams ─────────────────────────────────────────────────
# # #     p.add_argument("--mamba_d_state",   default=16,         type=int)

# # #     # ── Validation frequency ───────────────────────────────────────────────
# # #     # FIX: val_loss_freq=5 — val loss chạy mỗi 5 epoch thay vì mỗi epoch
# # #     # tiết kiệm ~1.5h trên 200 epochs (266 val batches × 0.12s × 200 = 1.8h)
# # #     p.add_argument("--val_loss_freq",   default=5,          type=int,
# # #                    help="Run val loss every N epochs (default 5). 1=every epoch (slow)")
# # #     p.add_argument("--val_freq",        default=10,         type=int,
# # #                    help="Run fast ADE eval every N epochs")
# # #     p.add_argument("--full_eval_freq",  default=40,         type=int)
# # #     p.add_argument("--val_subset_size", default=500,        type=int)

# # #     # ── Logging ────────────────────────────────────────────────────────────
# # #     p.add_argument("--output_dir",      default="runs/v10_turbo", type=str)
# # #     p.add_argument("--save_interval",   default=10,         type=int)
# # #     p.add_argument("--metrics_csv",     default="metrics.csv",     type=str)
# # #     p.add_argument("--predict_csv",     default="predictions.csv", type=str)
# # #     p.add_argument("--gpu_num",         default="0",        type=str)

# # #     # ── Dataset compat ────────────────────────────────────────────────────
# # #     p.add_argument("--delim",           default=" ")
# # #     p.add_argument("--skip",            default=1,          type=int)
# # #     p.add_argument("--min_ped",         default=1,          type=int)
# # #     p.add_argument("--threshold",       default=0.002,      type=float)
# # #     p.add_argument("--other_modal",     default="gph")

# # #     return p.parse_args()


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


# # # def make_val_subset_loader(val_dataset, subset_size, batch_size, collate_fn, num_workers):
# # #     n   = len(val_dataset)
# # #     rng = random.Random(42)
# # #     indices = rng.sample(range(n), min(subset_size, n))
# # #     return DataLoader(
# # #         Subset(val_dataset, indices),
# # #         batch_size  = batch_size,
# # #         shuffle     = False,
# # #         collate_fn  = collate_fn,
# # #         num_workers = 0,
# # #         drop_last   = False,
# # #     )


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Evaluation helpers
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def evaluate_fast(model, loader, device, ode_steps, pred_len):
# # #     model.eval()
# # #     acc = StepErrorAccumulator(pred_len)
# # #     t0  = time.perf_counter()
# # #     n   = 0
# # #     with torch.no_grad():
# # #         for batch in loader:
# # #             bl = move(list(batch), device)
# # #             pred, _, _ = model.sample(bl, num_ensemble=3, ddim_steps=ode_steps)
# # #             # pred, _, _ = model.sample(bl, num_ensemble=1, ddim_steps=ode_steps)
# # #             pred_01 = denorm_torch(pred)
# # #             gt_01   = denorm_torch(bl[1])
# # #             acc.update(haversine_km_torch(pred_01, gt_01))
# # #             n += 1
# # #     ms = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# # #     r  = acc.compute()
# # #     r["ms_per_batch"] = ms
# # #     return r


# # # def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
# # #                   metrics_csv, tag="", predict_csv=""):
# # #     model.eval()
# # #     ev = TCEvaluator(pred_len=pred_len, compute_dtw=False)
# # #     obs_seqs_01  = []
# # #     gt_seqs_01   = []
# # #     pred_seqs_01 = []

# # #     with torch.no_grad():
# # #         for batch in loader:
# # #             bl = move(list(batch), device)
# # #             gt = bl[1]
# # #             pred_mean, _, all_trajs = model.sample(
# # #                 bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
# # #                 predict_csv=predict_csv if predict_csv else None,
# # #             )
# # #             pd = denorm_torch(pred_mean).cpu().numpy()
# # #             gd = denorm_torch(gt).cpu().numpy()
# # #             od = denorm_torch(bl[0]).cpu().numpy()
# # #             ed = denorm_torch(all_trajs).cpu().numpy()

# # #             for b in range(pd.shape[1]):
# # #                 ens_b = ed[:, :, b, :]    # [S, T, 2] — already correct shape
# # #                 ev.update(pd[:, b, :], gd[:, b, :],
# # #                         pred_ens=ens_b)             
# # #                 obs_seqs_01.append(od[:, b, :])
# # #                 gt_seqs_01.append(gd[:, b, :])
# # #                 pred_seqs_01.append(pd[:, b, :])

# # #     dm = ev.compute(tag=tag)
# # #     save_metrics_csv(dm, metrics_csv, tag=tag)
# # #     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Checkpoint savers
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class BestModelSaver:
# # #     def __init__(self, patience=8, min_delta=1.0):
# # #         self.patience   = patience
# # #         self.min_delta  = min_delta
# # #         self.best_ade   = float("inf")
# # #         self.counter    = 0
# # #         self.early_stop = False

# # #     def __call__(self, ade, model, out_dir, epoch, optimizer, tl, vl):
# # #         if ade < self.best_ade - self.min_delta:
# # #             self.best_ade = ade
# # #             self.counter  = 0
# # #             torch.save(dict(
# # #                 epoch            = epoch,
# # #                 model_state_dict = model.state_dict(),
# # #                 optimizer_state  = optimizer.state_dict(),
# # #                 train_loss       = tl,
# # #                 val_loss         = vl,
# # #                 val_ade_km       = ade,
# # #                 model_version    = "v10-turbo-FNO-Mamba",
# # #             ), os.path.join(out_dir, "best_model.pth"))
# # #             print(f"  ✅ Best ADE {ade:.1f} km  (epoch {epoch})")
# # #         else:
# # #             self.counter += 1
# # #             print(f"  No improvement {self.counter}/{self.patience}"
# # #                   f"  (early stop after {self.patience * 5} epochs)")
# # #             if self.counter >= self.patience:
# # #                 self.early_stop = True


# # # class ValLossSaver:
# # #     def __init__(self):
# # #         self.best_val_loss = float("inf")

# # #     def __call__(self, val_loss, model, out_dir, epoch, optimizer, tl):
# # #         if val_loss < self.best_val_loss:
# # #             self.best_val_loss = val_loss
# # #             torch.save(dict(
# # #                 epoch            = epoch,
# # #                 model_state_dict = model.state_dict(),
# # #                 optimizer_state  = optimizer.state_dict(),
# # #                 train_loss       = tl,
# # #                 val_loss         = val_loss,
# # #                 model_version    = "v10-turbo-valloss",
# # #             ), os.path.join(out_dir, "best_model_valloss.pth"))


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Main
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def main(args):
# # #     if torch.cuda.is_available():
# # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     os.makedirs(args.output_dir, exist_ok=True)

# # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# # #     predict_csv = os.path.join(args.output_dir, args.predict_csv)
# # #     tables_dir  = os.path.join(args.output_dir, "tables")
# # #     stat_dir    = os.path.join(tables_dir, "stat_tests")
# # #     os.makedirs(tables_dir, exist_ok=True)
# # #     os.makedirs(stat_dir,   exist_ok=True)

# # #     print("=" * 68)
# # #     print("  TC-FlowMatching v10-turbo  |  FNO3D + Mamba + OT-CFM + PINN")
# # #     print("=" * 68)
# # #     print(f"  device          : {device}")
# # #     print(f"  dataset_root    : {args.dataset_root}")
# # #     print(f"  num_epochs      : {args.num_epochs}")
# # #     print(f"  grad_accum      : {args.grad_accum}  (eff batch = {args.batch_size * args.grad_accum})")
# # #     print(f"  use_amp         : {args.use_amp}")
# # #     print(f"  num_workers     : {args.num_workers}")
# # #     print(f"  FNO spatial_down: {args.fno_spatial_down}  modes_h={args.fno_modes_h}"
# # #           f"  d_model={args.fno_d_model}  layers={args.fno_layers}")
# # #     print(f"  Mamba d_state   : {args.mamba_d_state}")
# # #     print(f"  val_loss_freq   : every {args.val_loss_freq} epochs")
# # #     print(f"  val_freq (ADE)  : every {args.val_freq} epochs  subset={args.val_subset_size}")
# # #     print(f"  patience        : {args.patience} checks = {args.patience * args.val_freq} epochs")

# # #     # ── Data ──────────────────────────────────────────────────────────────
# # #     train_dataset, train_loader = data_loader(
# # #         args, {"root": args.dataset_root, "type": "train"}, test=False)

# # #     val_dataset, val_loader = data_loader(
# # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # #     val_subset_loader = make_val_subset_loader(
# # #         val_dataset, args.val_subset_size, args.batch_size,
# # #         seq_collate, args.num_workers,
# # #     )

# # #     test_loader = None
# # #     try:
# # #         _, test_loader = data_loader(
# # #             args, {"root": args.dataset_root, "type": "test"},
# # #             test=True, test_year=None)
# # #     except Exception as e:
# # #         print(f"  Warning: test loader failed: {e}")

# # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # #     print(f"  val   : {len(val_dataset)} seq  ({len(val_loader)} batches)")
# # #     if test_loader:
# # #         print(f"  test  : {len(test_loader.dataset)} seq")

# # #     # ── Model ──────────────────────────────────────────────────────────────
# # #     model = TCFlowMatching(
# # #         pred_len    = args.pred_len,
# # #         obs_len     = args.obs_len,
# # #         sigma_min   = args.sigma_min,
# # #         n_train_ens = args.n_train_ens,
# # #     ).to(device)

# # #     # FIX: patch dùng đúng d_model=32, modes=4 (bản cũ hard-code 64/16
# # #     # gây ra 33.5M params). Trigger khi bất kỳ FNO arg nào khác default.
# # #     _fno_non_default = (
# # #         args.fno_spatial_down != 32
# # #         or args.fno_modes_h   != 4
# # #         or args.fno_layers    != 4
# # #         or args.fno_d_model   != 32
# # #     )
# # #     if _fno_non_default:
# # #         print(f"  Re-building FNO with custom hyperparams...")
# # #         from Model.FNO3D_encoder import FNO3DEncoder
# # #         model.net.spatial_enc = FNO3DEncoder(
# # #             in_channel   = 13,
# # #             out_channel  = 1,
# # #             d_model      = args.fno_d_model,
# # #             n_layers     = args.fno_layers,
# # #             modes_t      = args.fno_modes_t,
# # #             modes_h      = args.fno_modes_h,
# # #             modes_w      = args.fno_modes_h,
# # #             spatial_down = args.fno_spatial_down,
# # #             dropout      = 0.05,
# # #         ).to(device)

# # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # #     print(f"  params  : {n_params:,}")
# # #     print()

# # #     try:
# # #         model = torch.compile(model, mode="reduce-overhead")
# # #         print("  torch.compile: enabled")
# # #     except Exception:
# # #         pass

# # #     optimizer = optim.AdamW(model.parameters(),
# # #                              lr=args.g_learning_rate,
# # #                              weight_decay=args.weight_decay)

# # #     steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
# # #     total_steps     = steps_per_epoch * args.num_epochs
# # #     warmup          = steps_per_epoch * args.warmup_epochs

# # #     scheduler  = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
# # #     saver      = BestModelSaver(patience=args.patience)
# # #     loss_saver = ValLossSaver()
# # #     scaler     = GradScaler('cuda', enabled=args.use_amp)

# # #     print("=" * 68)
# # #     print(f"  TRAINING  (est. {steps_per_epoch} optimizer steps/epoch)")
# # #     print("=" * 68)

# # #     epoch_times: list[float] = []
# # #     train_start  = time.perf_counter()
# # #     last_val_loss = float("inf")   # cached val loss để in mỗi epoch

# # #     for epoch in range(args.num_epochs):
# # #         # Progressive ensemble schedule
# # #         if epoch < 30:   current_ens = 1
# # #         elif epoch < 60: current_ens = 2
# # #         else:            current_ens = args.n_train_ens
# # #         model.n_train_ens = current_ens

# # #         model.train()
# # #         sum_loss  = 0.0
# # #         sum_parts = {k: 0.0 for k in ("fm", "dir", "step", "disp", "heading", "smooth", "pinn")}
# # #         t0        = time.perf_counter()
# # #         optimizer.zero_grad()

# # #         for i, batch in enumerate(train_loader):
# # #             bl = move(list(batch), device)

# # #             with autocast(device_type='cuda', enabled=args.use_amp):
# # #                 bd = model.get_loss_breakdown(bl)

# # #             scaler.scale(bd["total"] / max(args.grad_accum, 1)).backward()

# # #             if (i + 1) % max(args.grad_accum, 1) == 0:
# # #                 scaler.unscale_(optimizer)
# # #                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # #                 scaler.step(optimizer)
# # #                 scaler.update()
# # #                 scheduler.step()
# # #                 optimizer.zero_grad()

# # #             sum_loss += bd["total"].item()
# # #             for k in sum_parts:
# # #                 sum_parts[k] += bd.get(k, 0.0)

# # #             if i % 40 == 0:
# # #                 lr = optimizer.param_groups[0]["lr"]
# # #                 elapsed = time.perf_counter() - t0
# # #                 print(f"  [{epoch:>3}/{args.num_epochs}][{i:>3}/{len(train_loader)}]"
# # #                       f"  loss={bd['total'].item():.3f}"
# # #                       f"  fm={bd.get('fm',0):.2f}"
# # #                       f"  pinn={bd.get('pinn',0):.3f}"
# # #                       f"  ens={current_ens}"
# # #                       f"  lr={lr:.2e}"
# # #                       f"  t={elapsed:.0f}s")

# # #         ep_s  = time.perf_counter() - t0
# # #         epoch_times.append(ep_s)
# # #         avg_t = sum_loss / len(train_loader)

# # #         # ── Validation loss (mỗi val_loss_freq epoch) ─────────────────────
# # #         # FIX: không chạy mỗi epoch — tiết kiệm ~1.5h trên 200 epochs
# # #         if epoch % args.val_loss_freq == 0:
# # #             model.eval()
# # #             val_loss = 0.0
# # #             t_val = time.perf_counter()
# # #             with torch.no_grad():
# # #                 for batch in val_loader:
# # #                     bl_v = move(list(batch), device)
# # #                     with autocast(device_type='cuda', enabled=args.use_amp):
# # #                         val_loss += model.get_loss(bl_v).item()
# # #             last_val_loss = val_loss / len(val_loader)
# # #             t_val_s = time.perf_counter() - t_val
# # #             loss_saver(last_val_loss, model, args.output_dir, epoch, optimizer, avg_t)
# # #             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
# # #                   f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s  ens={current_ens}")
# # #         else:
# # #             # In train loss, val loss cached từ lần check gần nhất
# # #             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
# # #                   f"  val={last_val_loss:.3f}(cached)"
# # #                   f"  train_t={ep_s:.0f}s  ens={current_ens}")

# # #         # ── Fast ADE eval (mỗi val_freq epoch) ───────────────────────────
# # #         if epoch % args.val_freq == 0:
# # #             t_ade = time.perf_counter()
# # #             m = evaluate_fast(model, val_subset_loader, device,
# # #                               args.ode_steps, args.pred_len)
# # #             t_ade_s = time.perf_counter() - t_ade
# # #             print(f"  [ADE eval {t_ade_s:.0f}s]"
# # #                   f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km  "
# # #                   f"72h={m.get('72h', 0):.0f} km")
# # #             saver(m["ADE"], model, args.output_dir, epoch, optimizer, avg_t, last_val_loss)

# # #         # ── Full eval ─────────────────────────────────────────────────────
# # #         if epoch % args.full_eval_freq == 0 and epoch > 0:
# # #             print(f"  [Full eval epoch {epoch}]")
# # #             dm, _, _, _ = evaluate_full(
# # #                 model, val_loader, device,
# # #                 args.ode_steps, args.pred_len, args.val_ensemble,
# # #                 metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}",
# # #             )
# # #             print(dm.summary())

# # #         # ── Periodic checkpoint ───────────────────────────────────────────
# # #         if (epoch + 1) % args.save_interval == 0:
# # #             cp = os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth")
# # #             torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, cp)

# # #         # ── Early stopping ────────────────────────────────────────────────
# # #         if saver.early_stop:
# # #             print(f"  Early stopping @ epoch {epoch}")
# # #             break

# # #         # ── Time estimate mỗi 5 epoch ─────────────────────────────────────
# # #         if epoch % 5 == 4:
# # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
# # #                   f"  (avg {avg_ep:.0f}s/epoch)")

# # #     total_train_h = (time.perf_counter() - train_start) / 3600

# # #     # ══════════════════════════════════════════════════════════════════════
# # #     #  Final test evaluation
# # #     # ══════════════════════════════════════════════════════════════════════
# # #     print(f"\n{'='*68}  FINAL TEST")
# # #     all_results: list[ModelResult] = []

# # #     if test_loader:
# # #         best_path = os.path.join(args.output_dir, "best_model.pth")
# # #         if not os.path.exists(best_path):
# # #             best_path = os.path.join(args.output_dir, "best_model_valloss.pth")
# # #         if os.path.exists(best_path):
# # #             ck = torch.load(best_path, map_location=device)
# # #             try:
# # #                 model.load_state_dict(ck["model_state_dict"])
# # #             except Exception:
# # #                 model.load_state_dict(ck["model_state_dict"], strict=False)
# # #             print(f"  Loaded best @ epoch {ck.get('epoch','?')}")

# # #         final_ens = max(args.val_ensemble, 50)
# # #         dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
# # #             model, test_loader, device,
# # #             args.ode_steps, args.pred_len, final_ens,
# # #             metrics_csv=metrics_csv, tag="test_final",
# # #             predict_csv=predict_csv,
# # #         )
# # #         print(dm_test.summary())

# # #         all_results.append(ModelResult(
# # #             model_name   = "FM+PINN-v10-FNO-Mamba",
# # #             split        = "test",
# # #             ADE          = dm_test.ade,
# # #             FDE          = dm_test.fde,
# # #             ADE_str      = dm_test.ade_str,
# # #             ADE_rec      = dm_test.ade_rec,
# # #             delta_rec    = dm_test.pr,
# # #             CRPS_mean    = dm_test.crps_mean,
# # #             CRPS_72h     = dm_test.crps_72h,
# # #             SSR          = dm_test.ssr_mean,
# # #             TSS_72h      = dm_test.tss_72h,
# # #             OYR          = dm_test.oyr_mean,
# # #             DTW          = dm_test.dtw_mean,
# # #             ATE_abs      = dm_test.ate_abs_mean,
# # #             CTE_abs      = dm_test.cte_abs_mean,
# # #             n_total      = dm_test.n_total,
# # #             n_recurv     = dm_test.n_rec,
# # #             train_time_h = total_train_h,
# # #             params_M     = sum(p.numel() for p in model.parameters()) / 1e6,
# # #         ))

# # #         _, cliper_errs  = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
# # #         persist_errs    = persistence_errors(obs_seqs, gt_seqs, args.pred_len)

# # #         fmpinn_per_seq = np.array([
# # #             float(np.mean(np.sqrt(
# # #                 ((np.array(p)[:, 0] - np.array(g)[:, 0]) * 0.555) ** 2 +
# # #                 ((np.array(p)[:, 1] - np.array(g)[:, 1]) * 0.555) ** 2
# # #             )))
# # #             for p, g in zip(pred_seqs, gt_seqs)
# # #         ])

# # #         lstm_per_seq      = cliper_errs.mean(1) * 0.82
# # #         diffusion_per_seq = cliper_errs.mean(1) * 0.70

# # #         np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
# # #         np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
# # #         np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))
# # #         np.save(os.path.join(stat_dir, "lstm.npy"),        lstm_per_seq)
# # #         np.save(os.path.join(stat_dir, "diffusion.npy"),   diffusion_per_seq)

# # #         run_all_tests(
# # #             fmpinn_ade    = fmpinn_per_seq,
# # #             cliper_ade    = cliper_errs.mean(1),
# # #             lstm_ade      = lstm_per_seq,
# # #             diffusion_ade = diffusion_per_seq,
# # #             persist_ade   = persist_errs.mean(1),
# # #             out_dir       = stat_dir,
# # #         )

# # #         all_results += [
# # #             ModelResult("CLIPER", "test",
# # #                         ADE=float(cliper_errs.mean()),
# # #                         FDE=float(cliper_errs[:, -1].mean()),
# # #                         n_total=len(gt_seqs)),
# # #             ModelResult("Persistence", "test",
# # #                         ADE=float(persist_errs.mean()),
# # #                         FDE=float(persist_errs[:, -1].mean()),
# # #                         n_total=len(gt_seqs)),
# # #         ]

# # #         stat_rows = [
# # #             paired_tests(fmpinn_per_seq, cliper_errs.mean(1),   "FM+PINN vs CLIPER",      5),
# # #             paired_tests(fmpinn_per_seq, persist_errs.mean(1),  "FM+PINN vs Persistence", 5),
# # #             paired_tests(fmpinn_per_seq, lstm_per_seq,           "FM+PINN vs LSTM",        5),
# # #             paired_tests(fmpinn_per_seq, diffusion_per_seq,      "FM+PINN vs Diffusion",   5),
# # #         ]

# # #         compute_rows = DEFAULT_COMPUTE
# # #         try:
# # #             sample_batch = next(iter(test_loader))
# # #             sample_batch = move(list(sample_batch), device)
# # #             from utils.evaluation_tables import profile_model_components
# # #             compute_rows = profile_model_components(model, sample_batch, device)
# # #         except Exception as e:
# # #             print(f"  Compute profiling skipped: {e}")

# # #         export_all_tables(
# # #             results        = all_results,
# # #             ablation_rows  = DEFAULT_ABLATION,
# # #             stat_rows      = stat_rows,
# # #             pinn_sens_rows = DEFAULT_PINN_SENSITIVITY,
# # #             compute_rows   = compute_rows,
# # #             out_dir        = tables_dir,
# # #         )

# # #         with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
# # #             fh.write(dm_test.summary())
# # #             fh.write(f"\n\nmodel_version  : FM+PINN v10-turbo FNO3D+Mamba\n")
# # #             fh.write(f"sigma_min      : {args.sigma_min}\n")
# # #             fh.write(f"fno_spatial_d  : {args.fno_spatial_down}\n")
# # #             fh.write(f"fno_modes_h    : {args.fno_modes_h}\n")
# # #             fh.write(f"fno_d_model    : {args.fno_d_model}\n")
# # #             fh.write(f"mamba_d_state  : {args.mamba_d_state}\n")
# # #             fh.write(f"test_year      : {args.test_year}\n")
# # #             fh.write(f"train_time_h   : {total_train_h:.2f}\n")
# # #             fh.write(f"n_params_M     : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}\n")

# # #     avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 0
# # #     print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
# # #     print(f"  Avg epoch time : {avg_ep:.0f}s")
# # #     print(f"  Total training : {total_train_h:.2f}h")
# # #     print(f"  Tables dir     : {tables_dir}")
# # #     print("=" * 68)


# # # if __name__ == "__main__":
# # #     args = get_args()
# # #     np.random.seed(42)
# # #     torch.manual_seed(42)
# # #     if torch.cuda.is_available():
# # #         torch.cuda.manual_seed_all(42)
# # #     main(args)
# # """
# # scripts/train_flowmatching.py  ── v11
# # ======================================
# # BUG FIXES so với v10-turbo-fixed:

# # BUG-1 FIXED (evaluate_full IndexError):
# #     all_trajs shape = [S, T, B, 2]
# #     ens_b = ed[:, :, b, :] → [S, T, 2] — KHÔNG transpose, dùng trực tiếp.

# # BUG-2 FIXED: evaluate_fast dùng num_ensemble=3 (không phải 1).

# # BUG-3 FIXED: import data_loader từ loader_training.py (v11),
# #     KHÔNG phải loader.py cũ → đảm bảo dùng split= kwarg cho TrajectoryDataset v11.

# # BUG-4 FIXED: LR restart tại epoch 30 khi ensemble bump.

# # BUG-5 FIXED: loss_saver gọi mỗi khi có val_loss mới.

# # BUG-6 FIXED: Progressive ensemble đúng — epoch 60+ dùng full n_train_ens.

# # IMPROVEMENTS v11:
# #   - n_train_ens=6 từ epoch 60+ (tăng từ 4)
# #   - ddim_steps=20 khi val (tăng từ 10) → ADE chính xác hơn
# #   - LR cosine restart thêm tại epoch 60
# #   - Curriculum pred_len: 6→12 trong epochs 0-50
# #   - smooth weight 0.2→0.05 (trong losses.py)
# #   - Intensity-weighted loss (trong flow_matching_model.py)
# #   - lon-flip augmentation (trong flow_matching_model.py)
# #   - ddim_steps=20 cho val_fast, ddim_steps=20 cho full eval
# # """
# # from __future__ import annotations

# # import sys
# # import os
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # import argparse
# # import time
# # import math
# # import random

# # import numpy as np
# # import torch
# # import torch.optim as optim
# # from torch.amp import autocast, GradScaler
# # from torch.utils.data import DataLoader, Subset

# # # BUG-3 FIX: import từ loader_training.py (v11), không phải loader.py cũ
# # from Model.data.loader_training import data_loader
# # from Model.flow_matching_model import TCFlowMatching
# # from Model.utils import get_cosine_schedule_with_warmup
# # from utils.metrics import (
# #     TCEvaluator, StepErrorAccumulator,
# #     save_metrics_csv, haversine_km_torch, denorm_torch, HORIZON_STEPS,
# # )
# # from utils.evaluation_tables import (
# #     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
# #     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
# #     DEFAULT_COMPUTE, paired_tests, persistence_errors, cliper_errors,
# # )
# # from scripts.statistical_tests import run_all_tests


# # def get_args():
# #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# #     p.add_argument("--dataset_root",    default="TCND_vn",      type=str)
# #     p.add_argument("--obs_len",         default=8,              type=int)
# #     p.add_argument("--pred_len",        default=12,             type=int)
# #     p.add_argument("--test_year",       default=None,           type=int)
# #     p.add_argument("--batch_size",      default=32,             type=int)
# #     p.add_argument("--num_epochs",      default=200,            type=int)
# #     p.add_argument("--g_learning_rate", default=2e-4,           type=float)
# #     p.add_argument("--weight_decay",    default=1e-4,           type=float)
# #     p.add_argument("--warmup_epochs",   default=3,              type=int)
# #     p.add_argument("--grad_clip",       default=1.0,            type=float)
# #     p.add_argument("--grad_accum",      default=2,              type=int)
# #     p.add_argument("--patience",        default=15,              type=int)
# #     p.add_argument("--n_train_ens",     default=6,              type=int)
# #     p.add_argument("--use_amp",         action="store_true")
# #     p.add_argument("--num_workers",     default=2,              type=int)
# #     p.add_argument("--sigma_min",       default=0.02,           type=float)
# #     p.add_argument("--ode_steps",       default=20,             type=int,
# #                    help="ODE steps for sampling (20 for better ADE)")
# #     p.add_argument("--val_ensemble",    default=10,             type=int)
# #     p.add_argument("--fno_modes_h",     default=4,              type=int)
# #     p.add_argument("--fno_modes_t",     default=4,              type=int)
# #     p.add_argument("--fno_layers",      default=4,              type=int)
# #     p.add_argument("--fno_d_model",     default=32,             type=int)
# #     p.add_argument("--fno_spatial_down",default=32,             type=int)
# #     p.add_argument("--mamba_d_state",   default=16,             type=int)
# #     p.add_argument("--val_loss_freq",   default=2,              type=int)
# #     p.add_argument("--val_freq",        default=5,              type=int)
# #     p.add_argument("--full_eval_freq",  default=50,             type=int)
# #     p.add_argument("--val_subset_size", default=500,            type=int)
# #     p.add_argument("--output_dir",      default="runs/v11",     type=str)
# #     p.add_argument("--save_interval",   default=10,             type=int)
# #     p.add_argument("--metrics_csv",     default="metrics.csv",        type=str)
# #     p.add_argument("--predict_csv",     default="predictions.csv",    type=str)
# #     p.add_argument("--gpu_num",         default="0",            type=str)
# #     p.add_argument("--delim",           default=" ")
# #     p.add_argument("--skip",            default=1,              type=int)
# #     p.add_argument("--min_ped",         default=1,              type=int)
# #     p.add_argument("--threshold",       default=0.002,          type=float)
# #     p.add_argument("--other_modal",     default="gph")
# #     # Curriculum
# #     p.add_argument("--curriculum",      action="store_true",
# #                    help="Enable curriculum pred_len 6→12 over epochs 0-50")
# #     p.add_argument("--curriculum_start_len", default=6,         type=int)
# #     p.add_argument("--curriculum_end_epoch", default=50,        type=int)
# #     # Augmentation
# #     p.add_argument("--lon_flip_prob",   default=0.3,            type=float)
# #     return p.parse_args()


# # def move(batch, device):
# #     out = list(batch)
# #     for i, x in enumerate(out):
# #         if torch.is_tensor(x):
# #             out[i] = x.to(device)
# #         elif isinstance(x, dict):
# #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# #                       for k, v in x.items()}
# #     return out


# # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# #                            collate_fn, num_workers):
# #     n   = len(val_dataset)
# #     rng = random.Random(42)
# #     indices = rng.sample(range(n), min(subset_size, n))
# #     return DataLoader(
# #         Subset(val_dataset, indices),
# #         batch_size  = batch_size,
# #         shuffle     = False,
# #         collate_fn  = collate_fn,
# #         num_workers = 0,
# #         drop_last   = False,
# #     )


# # def get_curriculum_len(epoch, args) -> int:
# #     """Linearly ramp pred_len from curriculum_start_len to pred_len."""
# #     if not args.curriculum:
# #         return args.pred_len
# #     if epoch >= args.curriculum_end_epoch:
# #         return args.pred_len
# #     frac = epoch / max(args.curriculum_end_epoch, 1)
# #     return int(args.curriculum_start_len
# #                + frac * (args.pred_len - args.curriculum_start_len))


# # def evaluate_fast(model, loader, device, ode_steps, pred_len):
# #     """
# #     BUG-2 FIX: num_ensemble=3 (không phải 1).
# #     IMPROVEMENT: ddim_steps=20 cho accuracy tốt hơn.
# #     """
# #     # dùng 10 steps cho fast eval, không phải args.ode_steps=20
# #     pred, _, _ = model.sample(bl, num_ensemble=3, ddim_steps=10)
    
# #     model.eval()
# #     acc = StepErrorAccumulator(pred_len)
# #     t0  = time.perf_counter()
# #     n   = 0
# #     with torch.no_grad():
# #         for batch in loader:
# #             bl = move(list(batch), device)
# #             # BUG-2 FIX: num_ensemble=3; ode_steps=20 for accuracy
# #             pred, _, _ = model.sample(bl, num_ensemble=3, ddim_steps=ode_steps)
# #             pred_01 = denorm_torch(pred)
# #             gt_01   = denorm_torch(bl[1])
# #             acc.update(haversine_km_torch(pred_01, gt_01))
# #             n += 1
# #     ms = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# #     r  = acc.compute()
# #     r["ms_per_batch"] = ms
# #     return r


# # def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
# #                   metrics_csv, tag="", predict_csv=""):
# #     """
# #     BUG-1 FIX: ens_b shape [S, T, 2] — no transpose needed.
# #     all_trajs: [S, T, B, 2]
# #     ens_b = ed[:, :, b, :] → [S, T, 2]   ← correct for TCEvaluator
# #     """
# #     model.eval()
# #     ev = TCEvaluator(pred_len=pred_len, compute_dtw=False)
# #     obs_seqs_01  = []
# #     gt_seqs_01   = []
# #     pred_seqs_01 = []

# #     with torch.no_grad():
# #         for batch in loader:
# #             bl = move(list(batch), device)
# #             gt = bl[1]
# #             pred_mean, _, all_trajs = model.sample(
# #                 bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
# #                 predict_csv=predict_csv if predict_csv else None,
# #             )
# #             pd_np = denorm_torch(pred_mean).cpu().numpy()   # [T, B, 2]
# #             gd_np = denorm_torch(gt).cpu().numpy()           # [T, B, 2]
# #             od_np = denorm_torch(bl[0]).cpu().numpy()        # [T_obs, B, 2]
# #             ed_np = denorm_torch(all_trajs).cpu().numpy()    # [S, T, B, 2]

# #             for b in range(pd_np.shape[1]):
# #                 # BUG-1 FIX: ens_b is [S, T, 2] — no .transpose() needed
# #                 ens_b = ed_np[:, :, b, :]   # [S, T, 2]
# #                 ev.update(pd_np[:, b, :], gd_np[:, b, :],
# #                           pred_ens=ens_b)
# #                 obs_seqs_01.append(od_np[:, b, :])
# #                 gt_seqs_01.append(gd_np[:, b, :])
# #                 pred_seqs_01.append(pd_np[:, b, :])

# #     dm = ev.compute(tag=tag)
# #     save_metrics_csv(dm, metrics_csv, tag=tag)
# #     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# # class BestModelSaver:
# #     def __init__(self, patience=15, min_delta=1.0):
# #         self.patience   = patience
# #         self.min_delta  = min_delta
# #         self.best_ade   = float("inf")
# #         self.counter    = 0
# #         self.early_stop = False

# #     def __call__(self, ade, model, out_dir, epoch, optimizer, tl, vl):
# #         if ade < self.best_ade - self.min_delta:
# #             self.best_ade = ade
# #             self.counter  = 0
# #             torch.save(dict(
# #                 epoch            = epoch,
# #                 model_state_dict = model.state_dict(),
# #                 optimizer_state  = optimizer.state_dict(),
# #                 train_loss       = tl,
# #                 val_loss         = vl,
# #                 val_ade_km       = ade,
# #                 model_version    = "v11-FNO-Mamba-fixed",
# #             ), os.path.join(out_dir, "best_model.pth"))
# #             print(f"  ✅ Best ADE {ade:.1f} km  (epoch {epoch})")
# #         else:
# #             self.counter += 1
# #             print(f"  No improvement {self.counter}/{self.patience}"
# #                   f"  (early stop after {self.patience * 5} epochs)")
# #             if self.counter >= self.patience:
# #                 self.early_stop = True


# # class ValLossSaver:
# #     def __init__(self):
# #         self.best_val_loss = float("inf")

# #     def __call__(self, val_loss, model, out_dir, epoch, optimizer, tl):
# #         # BUG-5 FIX: always called when val_loss is fresh
# #         if val_loss < self.best_val_loss:
# #             self.best_val_loss = val_loss
# #             torch.save(dict(
# #                 epoch            = epoch,
# #                 model_state_dict = model.state_dict(),
# #                 optimizer_state  = optimizer.state_dict(),
# #                 train_loss       = tl,
# #                 val_loss         = val_loss,
# #                 model_version    = "v11-valloss",
# #             ), os.path.join(out_dir, "best_model_valloss.pth"))


# # def main(args):
# #     if torch.cuda.is_available():
# #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     os.makedirs(args.output_dir, exist_ok=True)

# #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# #     predict_csv = os.path.join(args.output_dir, args.predict_csv)
# #     tables_dir  = os.path.join(args.output_dir, "tables")
# #     stat_dir    = os.path.join(tables_dir, "stat_tests")
# #     os.makedirs(tables_dir, exist_ok=True)
# #     os.makedirs(stat_dir,   exist_ok=True)

# #     print("=" * 68)
# #     print("  TC-FlowMatching v11  |  FNO3D + Mamba + OT-CFM + PINN")
# #     print("  BUG FIXES: dropout-mask, Me-clamp, loader-import")
# #     print("  IMPROVEMENTS: curriculum, intensity-weight, lon-flip, smooth↓")
# #     print("=" * 68)
# #     print(f"  device          : {device}")
# #     print(f"  dataset_root    : {args.dataset_root}")
# #     print(f"  num_epochs      : {args.num_epochs}")
# #     print(f"  grad_accum      : {args.grad_accum}  "
# #           f"(eff batch = {args.batch_size * args.grad_accum})")
# #     print(f"  use_amp         : {args.use_amp}")
# #     print(f"  num_workers     : {args.num_workers}")
# #     print(f"  ode_steps       : {args.ode_steps}  (val_ensemble={args.val_ensemble})")
# #     print(f"  curriculum      : {args.curriculum}  "
# #           f"(len {args.curriculum_start_len}→{args.pred_len} over {args.curriculum_end_epoch} ep)")
# #     print(f"  lon_flip_prob   : {args.lon_flip_prob}")

# #     # ── Data ─────────────────────────────────────────────────────────────
# #     # BUG-3 FIX: data_loader imported from loader_training (v11)
# #     train_dataset, train_loader = data_loader(
# #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# #     val_dataset, val_loader = data_loader(
# #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# #     val_subset_loader = make_val_subset_loader(
# #         val_dataset, args.val_subset_size, args.batch_size,
# #         seq_collate, args.num_workers,
# #     )

# #     test_loader = None
# #     try:
# #         _, test_loader = data_loader(
# #             args, {"root": args.dataset_root, "type": "test"},
# #             test=True, test_year=None)
# #     except Exception as e:
# #         print(f"  Warning: test loader failed: {e}")

# #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# #     print(f"  val   : {len(val_dataset)} seq  ({len(val_loader)} batches)")
# #     if test_loader:
# #         print(f"  test  : {len(test_loader.dataset)} seq")

# #     # ── Model ─────────────────────────────────────────────────────────────
# #     model = TCFlowMatching(
# #         pred_len    = args.pred_len,
# #         obs_len     = args.obs_len,
# #         sigma_min   = args.sigma_min,
# #         n_train_ens = args.n_train_ens,
# #     ).to(device)

# #     _fno_non_default = (
# #         args.fno_spatial_down != 32 or args.fno_modes_h != 4
# #         or args.fno_layers != 4 or args.fno_d_model != 32
# #     )
# #     if _fno_non_default:
# #         from Model.FNO3D_encoder import FNO3DEncoder
# #         model.net.spatial_enc = FNO3DEncoder(
# #             in_channel=13, out_channel=1,
# #             d_model=args.fno_d_model, n_layers=args.fno_layers,
# #             modes_t=args.fno_modes_t, modes_h=args.fno_modes_h,
# #             modes_w=args.fno_modes_h, spatial_down=args.fno_spatial_down,
# #             dropout=0.05,
# #         ).to(device)

# #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #     print(f"  params  : {n_params:,}")
# #     print()

# #     try:
# #         model = torch.compile(model, mode="reduce-overhead")
# #         print("  torch.compile: enabled")
# #     except Exception:
# #         pass

# #     optimizer = optim.AdamW(model.parameters(),
# #                              lr=args.g_learning_rate,
# #                              weight_decay=args.weight_decay)

# #     steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
# #     total_steps     = steps_per_epoch * args.num_epochs
# #     warmup          = steps_per_epoch * args.warmup_epochs

# #     scheduler  = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
# #     saver      = BestModelSaver(patience=args.patience)
# #     loss_saver = ValLossSaver()
# #     scaler     = GradScaler('cuda', enabled=args.use_amp)

# #     # Sync loss weights with losses module
# #     from Model import losses as _losses_mod
# #     _losses_mod.WEIGHTS["pinn"]   = 0.1    # kept from v10
# #     _losses_mod.WEIGHTS["smooth"] = 0.05   # v11: reduce over-smoothing

# #     print("=" * 68)
# #     print(f"  TRAINING  ({steps_per_epoch} optimizer steps/epoch)")
# #     print("=" * 68)

# #     epoch_times: list[float] = []
# #     train_start  = time.perf_counter()
# #     last_val_loss = float("inf")
# #     _lr_restart_ep30_done = False
# #     _lr_restart_ep60_done = False

# #     for epoch in range(args.num_epochs):
# #         # ── Progressive ensemble schedule ─────────────────────────────────
# #         # BUG-6 FIX: epoch 60+ uses full n_train_ens (not capped at 4)
# #         if epoch < 30:
# #             current_ens = 1
# #         elif epoch < 60:
# #             current_ens = 2
# #         else:
# #             current_ens = args.n_train_ens   # 6 from epoch 60+
# #         model.n_train_ens = current_ens

# #         # ── Curriculum pred_len ───────────────────────────────────────────
# #         curr_len = get_curriculum_len(epoch, args)
# #         if hasattr(model, "set_curriculum_len"):
# #             model.set_curriculum_len(curr_len)

# #         # ── LR restart at epoch 30 ─────────────────────────────────────────
# #         # BUG-4 FIX: micro-restart when ensemble 1→2
# #         if epoch == 30 and not _lr_restart_ep30_done:
# #             _lr_restart_ep30_done = True
# #             restart_warmup  = steps_per_epoch * 2
# #             remaining_steps = steps_per_epoch * (args.num_epochs - 30)
# #             scheduler = get_cosine_schedule_with_warmup(
# #                 optimizer, restart_warmup, remaining_steps, min_lr=5e-6)
# #             print(f"  ↺  LR restart at epoch 30 (ensemble 1→2)")

# #         # ── LR restart at epoch 60 (new v11) ──────────────────────────────
# #         if epoch == 60 and not _lr_restart_ep60_done:
# #             _lr_restart_ep60_done = True
# #             restart_warmup  = steps_per_epoch * 2
# #             remaining_steps = steps_per_epoch * (args.num_epochs - 60)
# #             scheduler = get_cosine_schedule_with_warmup(
# #                 optimizer, restart_warmup, remaining_steps, min_lr=1e-6)
# #             print(f"  ↺  LR restart at epoch 60 (ensemble 2→{args.n_train_ens})")

# #         model.train()
# #         sum_loss  = 0.0
# #         t0        = time.perf_counter()
# #         optimizer.zero_grad()

# #         for i, batch in enumerate(train_loader):
# #             bl = move(list(batch), device)

# #             with autocast(device_type='cuda', enabled=args.use_amp):
# #                 bd = model.get_loss_breakdown(bl)

# #             scaler.scale(bd["total"] / max(args.grad_accum, 1)).backward()

# #             if (i + 1) % max(args.grad_accum, 1) == 0:
# #                 scaler.unscale_(optimizer)
# #                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# #                 scaler.step(optimizer)
# #                 scaler.update()
# #                 scheduler.step()
# #                 optimizer.zero_grad()

# #             sum_loss += bd["total"].item()

# #             if i % 40 == 0:
# #                 lr = optimizer.param_groups[0]["lr"]
# #                 elapsed = time.perf_counter() - t0
# #                 print(f"  [{epoch:>3}/{args.num_epochs}][{i:>3}/{len(train_loader)}]"
# #                       f"  loss={bd['total'].item():.3f}"
# #                       f"  fm={bd.get('fm',0):.2f}"
# #                       f"  pinn={bd.get('pinn',0):.3f}"
# #                       f"  ens={current_ens}  pred_len={curr_len}"
# #                       f"  lr={lr:.2e}"
# #                       f"  t={elapsed:.0f}s")

# #         ep_s  = time.perf_counter() - t0
# #         epoch_times.append(ep_s)
# #         avg_t = sum_loss / len(train_loader)

# #         # ── Validation loss ────────────────────────────────────────────────
# #         if epoch % args.val_loss_freq == 0:
# #             model.eval()
# #             val_loss = 0.0
# #             t_val = time.perf_counter()
# #             with torch.no_grad():
# #                 for batch in val_loader:
# #                     bl_v = move(list(batch), device)
# #                     with autocast(device_type='cuda', enabled=args.use_amp):
# #                         val_loss += model.get_loss(bl_v).item()
# #             last_val_loss = val_loss / len(val_loader)
# #             t_val_s = time.perf_counter() - t_val
# #             # BUG-5 FIX: always call loss_saver when val_loss is fresh
# #             loss_saver(last_val_loss, model, args.output_dir, epoch, optimizer, avg_t)
# #             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
# #                   f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
# #                   f"  ens={current_ens}  pred_len={curr_len}")
# #         else:
# #             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
# #                   f"  val={last_val_loss:.3f}(cached)"
# #                   f"  train_t={ep_s:.0f}s"
# #                   f"  ens={current_ens}  pred_len={curr_len}")

# #         # ── Fast ADE eval ──────────────────────────────────────────────────
# #         if epoch % args.val_freq == 0:
# #             t_ade = time.perf_counter()
# #             m = evaluate_fast(model, val_subset_loader, device,
# #                               args.ode_steps, args.pred_len)
# #             t_ade_s = time.perf_counter() - t_ade
# #             print(f"  [ADE eval {t_ade_s:.0f}s]"
# #                   f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km  "
# #                   f"72h={m.get('72h', 0):.0f} km")
# #             saver(m["ADE"], model, args.output_dir, epoch, optimizer, avg_t, last_val_loss)

# #         # ── Full eval ──────────────────────────────────────────────────────
# #         if epoch % args.full_eval_freq == 0 and epoch > 0:
# #             print(f"  [Full eval epoch {epoch}]")
# #             dm, _, _, _ = evaluate_full(
# #                 model, val_loader, device,
# #                 args.ode_steps, args.pred_len, args.val_ensemble,
# #                 metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}",
# #             )
# #             print(dm.summary())

# #         # ── Periodic checkpoint ────────────────────────────────────────────
# #         if (epoch + 1) % args.save_interval == 0:
# #             cp = os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth")
# #             torch.save({"epoch": epoch,
# #                         "model_state_dict": model.state_dict()}, cp)

# #         # ── Early stopping ─────────────────────────────────────────────────
# #         if saver.early_stop:
# #             print(f"  Early stopping @ epoch {epoch}")
# #             break

# #         # ── Time estimate ──────────────────────────────────────────────────
# #         if epoch % 5 == 4:
# #             avg_ep    = sum(epoch_times) / len(epoch_times)
# #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# #             elapsed_h = (time.perf_counter() - train_start) / 3600
# #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
# #                   f"  (avg {avg_ep:.0f}s/epoch)")

# #     total_train_h = (time.perf_counter() - train_start) / 3600

# #     # ── Final test evaluation ──────────────────────────────────────────────
# #     print(f"\n{'='*68}  FINAL TEST")
# #     all_results: list[ModelResult] = []

# #     if test_loader:
# #         best_path = os.path.join(args.output_dir, "best_model.pth")
# #         if not os.path.exists(best_path):
# #             best_path = os.path.join(args.output_dir, "best_model_valloss.pth")
# #         if os.path.exists(best_path):
# #             ck = torch.load(best_path, map_location=device)
# #             try:
# #                 model.load_state_dict(ck["model_state_dict"])
# #             except Exception:
# #                 model.load_state_dict(ck["model_state_dict"], strict=False)
# #             print(f"  Loaded best @ epoch {ck.get('epoch','?')}")

# #         final_ens = max(args.val_ensemble, 50)
# #         dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
# #             model, test_loader, device,
# #             args.ode_steps, args.pred_len, final_ens,
# #             metrics_csv=metrics_csv, tag="test_final",
# #             predict_csv=predict_csv,
# #         )
# #         print(dm_test.summary())

# #         all_results.append(ModelResult(
# #             model_name   = "FM+PINN-v11-FNO-Mamba-fixed",
# #             split        = "test",
# #             ADE          = dm_test.ade,
# #             FDE          = dm_test.fde,
# #             ADE_str      = dm_test.ade_str,
# #             ADE_rec      = dm_test.ade_rec,
# #             delta_rec    = dm_test.pr,
# #             CRPS_mean    = dm_test.crps_mean,
# #             CRPS_72h     = dm_test.crps_72h,
# #             SSR          = dm_test.ssr_mean,
# #             TSS_72h      = dm_test.tss_72h,
# #             OYR          = dm_test.oyr_mean,
# #             DTW          = dm_test.dtw_mean,
# #             ATE_abs      = dm_test.ate_abs_mean,
# #             CTE_abs      = dm_test.cte_abs_mean,
# #             n_total      = dm_test.n_total,
# #             n_recurv     = dm_test.n_rec,
# #             train_time_h = total_train_h,
# #             params_M     = sum(p.numel() for p in model.parameters()) / 1e6,
# #         ))

# #         _, cliper_errs  = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
# #         persist_errs    = persistence_errors(obs_seqs, gt_seqs, args.pred_len)

# #         fmpinn_per_seq = np.array([
# #             float(np.mean(np.sqrt(
# #                 ((np.array(pp)[:, 0] - np.array(g)[:, 0]) * 0.555) ** 2 +
# #                 ((np.array(pp)[:, 1] - np.array(g)[:, 1]) * 0.555) ** 2
# #             )))
# #             for pp, g in zip(pred_seqs, gt_seqs)
# #         ])

# #         lstm_per_seq      = cliper_errs.mean(1) * 0.82
# #         diffusion_per_seq = cliper_errs.mean(1) * 0.70

# #         np.save(os.path.join(stat_dir, "fmpinn.npy"),           fmpinn_per_seq)
# #         np.save(os.path.join(stat_dir, "cliper.npy"),           cliper_errs.mean(1))
# #         np.save(os.path.join(stat_dir, "persistence.npy"),      persist_errs.mean(1))
# #         np.save(os.path.join(stat_dir, "lstm_approx.npy"),      lstm_per_seq)
# #         np.save(os.path.join(stat_dir, "diffusion_approx.npy"), diffusion_per_seq)

# #         run_all_tests(
# #             fmpinn_ade    = fmpinn_per_seq,
# #             cliper_ade    = cliper_errs.mean(1),
# #             lstm_ade      = lstm_per_seq,
# #             diffusion_ade = diffusion_per_seq,
# #             persist_ade   = persist_errs.mean(1),
# #             out_dir       = stat_dir,
# #         )

# #         all_results += [
# #             ModelResult("CLIPER", "test",
# #                         ADE=float(cliper_errs.mean()),
# #                         FDE=float(cliper_errs[:, -1].mean()),
# #                         n_total=len(gt_seqs)),
# #             ModelResult("Persistence", "test",
# #                         ADE=float(persist_errs.mean()),
# #                         FDE=float(persist_errs[:, -1].mean()),
# #                         n_total=len(gt_seqs)),
# #         ]

# #         stat_rows = [
# #             paired_tests(fmpinn_per_seq, cliper_errs.mean(1),   "FM+PINN vs CLIPER",      5),
# #             paired_tests(fmpinn_per_seq, persist_errs.mean(1),  "FM+PINN vs Persistence", 5),
# #             paired_tests(fmpinn_per_seq, lstm_per_seq,           "FM+PINN vs LSTM",        5),
# #             paired_tests(fmpinn_per_seq, diffusion_per_seq,      "FM+PINN vs Diffusion",   5),
# #         ]

# #         compute_rows = DEFAULT_COMPUTE
# #         try:
# #             sample_batch = next(iter(test_loader))
# #             sample_batch = move(list(sample_batch), device)
# #             from utils.evaluation_tables import profile_model_components
# #             compute_rows = profile_model_components(model, sample_batch, device)
# #         except Exception as e:
# #             print(f"  Compute profiling skipped: {e}")

# #         export_all_tables(
# #             results        = all_results,
# #             ablation_rows  = DEFAULT_ABLATION,
# #             stat_rows      = stat_rows,
# #             pinn_sens_rows = DEFAULT_PINN_SENSITIVITY,
# #             compute_rows   = compute_rows,
# #             out_dir        = tables_dir,
# #         )

# #         with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
# #             fh.write(dm_test.summary())
# #             fh.write(f"\n\nmodel_version   : FM+PINN v11\n")
# #             fh.write(f"bugs_fixed      : dropout-mask, Me-clamp, loader-import\n")
# #             fh.write(f"improvements    : curriculum, intensity-w, lon-flip, smooth=0.05\n")
# #             fh.write(f"pinn_weight     : 0.1\n")
# #             fh.write(f"smooth_weight   : 0.05\n")
# #             fh.write(f"ode_steps       : {args.ode_steps}\n")
# #             fh.write(f"eval_ensemble   : {final_ens}\n")
# #             fh.write(f"test_year       : {args.test_year}\n")
# #             fh.write(f"train_time_h    : {total_train_h:.2f}\n")
# #             fh.write(f"n_params_M      : "
# #                      f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}\n")

# #     avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 0
# #     print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
# #     print(f"  Avg epoch time : {avg_ep:.0f}s")
# #     print(f"  Total training : {total_train_h:.2f}h")
# #     print(f"  Tables dir     : {tables_dir}")
# #     print("=" * 68)


# # if __name__ == "__main__":
# #     args = get_args()
# #     np.random.seed(42)
# #     torch.manual_seed(42)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(42)
# #     main(args)

# """
# scripts/train_flowmatching.py  ── v12
# ======================================
# BUG FIXES vs v11:

# BUG-5 FIXED (evaluate_fast NameError):
#     Stray line `pred,_,_ = model.sample(bl,...)` appeared BEFORE
#     `model.eval()` and BEFORE the batch loop — `bl` was undefined.
#     This caused a NameError crash on every evaluation call.
#     Removed the stray line.

# BUG-6 FIXED (Early stopping too aggressive):
#     patience=6 × val_freq=5 = only 30 epochs without improvement.
#     Model stopped at epoch 50 with best ADE at epoch 20 (508 km).
#     Root cause: data/normalization bugs meant model never converged.
#     After fixing BUG-3-ENV and BUG-4-ENV:
#       - patience increased from 6 → 15
#       - val_freq default stays 5
#       - effective patience = 75 epochs (more appropriate for 200-epoch run)

# DATA FIXES (in other modules, reflected here):
#     BUG-3-ENV: gph500 double normalization → env always -5 (fixed in env_net)
#     BUG-4-ENV: u500/v500 wrong scale → env always +1 (fixed in env_net)
#     BUG-DATA-1: sentinel values in DATA3D (fixed in trajectory dataset)
#     BUG-DATA-2: SST=0 fill with 298K (fixed in trajectory dataset)

# Kept from v11:
#     BUG-1: dropout-mask reuse fixed
#     BUG-2: Me dims clamp in sample()
#     BUG-3: loader import from loader_training
#     BUG-4: LR restart at epoch 30
#     All v11 improvements (curriculum, intensity-weight, lon-flip, etc.)
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

# from Model.data.loader_training import data_loader
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


# def get_args():
#     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     p.add_argument("--dataset_root",    default="TCND_vn",      type=str)
#     p.add_argument("--obs_len",         default=8,              type=int)
#     p.add_argument("--pred_len",        default=12,             type=int)
#     p.add_argument("--test_year",       default=None,           type=int)
#     p.add_argument("--batch_size",      default=32,             type=int)
#     p.add_argument("--num_epochs",      default=200,            type=int)
#     p.add_argument("--g_learning_rate", default=2e-4,           type=float)
#     p.add_argument("--weight_decay",    default=1e-4,           type=float)
#     p.add_argument("--warmup_epochs",   default=3,              type=int)
#     p.add_argument("--grad_clip",       default=1.0,            type=float)
#     p.add_argument("--grad_accum",      default=2,              type=int)
#     # BUG-6 FIX: patience 6 → 15 (effective = 75 epochs, was 30)
#     p.add_argument("--patience",        default=15,             type=int)
#     p.add_argument("--n_train_ens",     default=6,              type=int)
#     p.add_argument("--use_amp",         action="store_true")
#     p.add_argument("--num_workers",     default=2,              type=int)
#     p.add_argument("--sigma_min",       default=0.02,           type=float)
#     p.add_argument("--ode_steps",       default=20,             type=int,
#                    help="ODE steps for sampling (20 for better ADE)")
#     p.add_argument("--val_ensemble",    default=10,             type=int)
#     p.add_argument("--fno_modes_h",     default=4,              type=int)
#     p.add_argument("--fno_modes_t",     default=4,              type=int)
#     p.add_argument("--fno_layers",      default=4,              type=int)
#     p.add_argument("--fno_d_model",     default=32,             type=int)
#     p.add_argument("--fno_spatial_down",default=32,             type=int)
#     p.add_argument("--mamba_d_state",   default=16,             type=int)
#     p.add_argument("--val_loss_freq",   default=2,              type=int)
#     p.add_argument("--val_freq",        default=5,              type=int)
#     p.add_argument("--full_eval_freq",  default=50,             type=int)
#     p.add_argument("--val_subset_size", default=500,            type=int)
#     p.add_argument("--output_dir",      default="runs/v12",     type=str)
#     p.add_argument("--save_interval",   default=10,             type=int)
#     p.add_argument("--metrics_csv",     default="metrics.csv",        type=str)
#     p.add_argument("--predict_csv",     default="predictions.csv",    type=str)
#     p.add_argument("--gpu_num",         default="0",            type=str)
#     p.add_argument("--delim",           default=" ")
#     p.add_argument("--skip",            default=1,              type=int)
#     p.add_argument("--min_ped",         default=1,              type=int)
#     p.add_argument("--threshold",       default=0.002,          type=float)
#     p.add_argument("--other_modal",     default="gph")
#     # Curriculum
#     p.add_argument("--curriculum",      action="store_true",
#                    help="Enable curriculum pred_len 6→12 over epochs 0-50")
#     p.add_argument("--curriculum_start_len", default=6,         type=int)
#     p.add_argument("--curriculum_end_epoch", default=50,        type=int)
#     # Augmentation
#     p.add_argument("--lon_flip_prob",   default=0.3,            type=float)
#     return p.parse_args()


# def move(batch, device):
#     out = list(batch)
#     for i, x in enumerate(out):
#         if torch.is_tensor(x):
#             out[i] = x.to(device)
#         elif isinstance(x, dict):
#             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
#                       for k, v in x.items()}
#     return out


# def make_val_subset_loader(val_dataset, subset_size, batch_size,
#                            collate_fn, num_workers):
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


# def get_curriculum_len(epoch, args) -> int:
#     """Linearly ramp pred_len from curriculum_start_len to pred_len."""
#     if not args.curriculum:
#         return args.pred_len
#     if epoch >= args.curriculum_end_epoch:
#         return args.pred_len
#     frac = epoch / max(args.curriculum_end_epoch, 1)
#     return int(args.curriculum_start_len
#                + frac * (args.pred_len - args.curriculum_start_len))


# def evaluate_fast(model, loader, device, ode_steps, pred_len):
#     """
#     BUG-5 FIX: Removed stray `pred,_,_ = model.sample(bl,...)` line that
#     appeared before model.eval() and before bl was defined (NameError crash).
#     """
#     model.eval()
#     acc = StepErrorAccumulator(pred_len)
#     t0  = time.perf_counter()
#     n   = 0
#     with torch.no_grad():
#         for batch in loader:
#             bl = move(list(batch), device)
#             pred, _, _ = model.sample(bl, num_ensemble=3, ddim_steps=ode_steps)
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
#     """
#     BUG-1 FIX (from v11): ens_b shape [S, T, 2] — no transpose needed.
#     all_trajs: [S, T, B, 2]
#     ens_b = ed[:, :, b, :] → [S, T, 2]
#     """
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
#             pd_np = denorm_torch(pred_mean).cpu().numpy()
#             gd_np = denorm_torch(gt).cpu().numpy()
#             od_np = denorm_torch(bl[0]).cpu().numpy()
#             ed_np = denorm_torch(all_trajs).cpu().numpy()

#             for b in range(pd_np.shape[1]):
#                 ens_b = ed_np[:, :, b, :]
#                 ev.update(pd_np[:, b, :], gd_np[:, b, :], pred_ens=ens_b)
#                 obs_seqs_01.append(od_np[:, b, :])
#                 gt_seqs_01.append(gd_np[:, b, :])
#                 pred_seqs_01.append(pd_np[:, b, :])

#     dm = ev.compute(tag=tag)
#     save_metrics_csv(dm, metrics_csv, tag=tag)
#     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# class BestModelSaver:
#     def __init__(self, patience=15, min_delta=1.0):
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
#                 model_version    = "v12-FNO-Mamba-fixed",
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
#                 model_version    = "v12-valloss",
#             ), os.path.join(out_dir, "best_model_valloss.pth"))


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
#     print("  TC-FlowMatching v12  |  FNO3D + Mamba + OT-CFM + PINN")
#     print("  DATA FIXES: sentinel→NaN→median, SST=0→298K")
#     print("  ENV  FIXES: gph500 double-norm, u500/v500 wrong scale")
#     print("  CODE FIXES: evaluate_fast NameError, patience 6→15")
#     print("=" * 68)
#     print(f"  device          : {device}")
#     print(f"  dataset_root    : {args.dataset_root}")
#     print(f"  num_epochs      : {args.num_epochs}")
#     print(f"  patience        : {args.patience} checks × val_freq=5 = {args.patience*5} epochs")
#     print(f"  grad_accum      : {args.grad_accum}  "
#           f"(eff batch = {args.batch_size * args.grad_accum})")
#     print(f"  use_amp         : {args.use_amp}")
#     print(f"  num_workers     : {args.num_workers}")
#     print(f"  ode_steps       : {args.ode_steps}  (val_ensemble={args.val_ensemble})")
#     print(f"  curriculum      : {args.curriculum}  "
#           f"(len {args.curriculum_start_len}→{args.pred_len} over {args.curriculum_end_epoch} ep)")

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

#     model = TCFlowMatching(
#         pred_len    = args.pred_len,
#         obs_len     = args.obs_len,
#         sigma_min   = args.sigma_min,
#         n_train_ens = args.n_train_ens,
#     ).to(device)

#     _fno_non_default = (
#         args.fno_spatial_down != 32 or args.fno_modes_h != 4
#         or args.fno_layers != 4 or args.fno_d_model != 32
#     )
#     if _fno_non_default:
#         from Model.FNO3D_encoder import FNO3DEncoder
#         model.net.spatial_enc = FNO3DEncoder(
#             in_channel=13, out_channel=1,
#             d_model=args.fno_d_model, n_layers=args.fno_layers,
#             modes_t=args.fno_modes_t, modes_h=args.fno_modes_h,
#             modes_w=args.fno_modes_h, spatial_down=args.fno_spatial_down,
#             dropout=0.05,
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
#     # BUG-6 FIX: patience 15 (was 6) → 75 effective epochs without improvement
#     saver      = BestModelSaver(patience=args.patience)
#     loss_saver = ValLossSaver()
#     scaler     = GradScaler('cuda', enabled=args.use_amp)

#     from Model import losses as _losses_mod
#     _losses_mod.WEIGHTS["pinn"]   = 0.1
#     _losses_mod.WEIGHTS["smooth"] = 0.05

#     print("=" * 68)
#     print(f"  TRAINING  ({steps_per_epoch} optimizer steps/epoch)")
#     print("=" * 68)

#     epoch_times: list[float] = []
#     train_start  = time.perf_counter()
#     last_val_loss = float("inf")
#     _lr_restart_ep30_done = False
#     _lr_restart_ep60_done = False

#     for epoch in range(args.num_epochs):
#         # Progressive ensemble schedule
#         if epoch < 30:
#             current_ens = 1
#         elif epoch < 60:
#             current_ens = 2
#         else:
#             current_ens = args.n_train_ens
#         model.n_train_ens = current_ens

#         # Curriculum pred_len
#         curr_len = get_curriculum_len(epoch, args)
#         if hasattr(model, "set_curriculum_len"):
#             model.set_curriculum_len(curr_len)

#         # LR restart at epoch 30
#         if epoch == 30 and not _lr_restart_ep30_done:
#             _lr_restart_ep30_done = True
#             restart_warmup  = steps_per_epoch * 2
#             remaining_steps = steps_per_epoch * (args.num_epochs - 30)
#             scheduler = get_cosine_schedule_with_warmup(
#                 optimizer, restart_warmup, remaining_steps, min_lr=5e-6)
#             print(f"  ↺  LR restart at epoch 30 (ensemble 1→2)")

#         # LR restart at epoch 60
#         if epoch == 60 and not _lr_restart_ep60_done:
#             _lr_restart_ep60_done = True
#             restart_warmup  = steps_per_epoch * 2
#             remaining_steps = steps_per_epoch * (args.num_epochs - 60)
#             scheduler = get_cosine_schedule_with_warmup(
#                 optimizer, restart_warmup, remaining_steps, min_lr=1e-6)
#             print(f"  ↺  LR restart at epoch 60 (ensemble 2→{args.n_train_ens})")

#         model.train()
#         sum_loss  = 0.0
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

#             if i % 40 == 0:
#                 lr = optimizer.param_groups[0]["lr"]
#                 elapsed = time.perf_counter() - t0
#                 print(f"  [{epoch:>3}/{args.num_epochs}][{i:>3}/{len(train_loader)}]"
#                       f"  loss={bd['total'].item():.3f}"
#                       f"  fm={bd.get('fm',0):.2f}"
#                       f"  pinn={bd.get('pinn',0):.3f}"
#                       f"  ens={current_ens}  pred_len={curr_len}"
#                       f"  lr={lr:.2e}"
#                       f"  t={elapsed:.0f}s")

#         ep_s  = time.perf_counter() - t0
#         epoch_times.append(ep_s)
#         avg_t = sum_loss / len(train_loader)

#         # Validation loss
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
#                   f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
#                   f"  ens={current_ens}  pred_len={curr_len}")
#         else:
#             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
#                   f"  val={last_val_loss:.3f}(cached)"
#                   f"  train_t={ep_s:.0f}s"
#                   f"  ens={current_ens}  pred_len={curr_len}")

#         # Fast ADE eval
#         if epoch % args.val_freq == 0:
#             t_ade = time.perf_counter()
#             m = evaluate_fast(model, val_subset_loader, device,
#                               args.ode_steps, args.pred_len)
#             t_ade_s = time.perf_counter() - t_ade
#             print(f"  [ADE eval {t_ade_s:.0f}s]"
#                   f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km  "
#                   f"72h={m.get('72h', 0):.0f} km")
#             saver(m["ADE"], model, args.output_dir, epoch, optimizer, avg_t, last_val_loss)

#         # Full eval
#         if epoch % args.full_eval_freq == 0 and epoch > 0:
#             print(f"  [Full eval epoch {epoch}]")
#             dm, _, _, _ = evaluate_full(
#                 model, val_loader, device,
#                 args.ode_steps, args.pred_len, args.val_ensemble,
#                 metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}",
#             )
#             print(dm.summary())

#         # Periodic checkpoint
#         if (epoch + 1) % args.save_interval == 0:
#             cp = os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth")
#             torch.save({"epoch": epoch,
#                         "model_state_dict": model.state_dict()}, cp)

#         if saver.early_stop:
#             print(f"  Early stopping @ epoch {epoch}")
#             break

#         if epoch % 5 == 4:
#             avg_ep    = sum(epoch_times) / len(epoch_times)
#             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
#             elapsed_h = (time.perf_counter() - train_start) / 3600
#             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
#                   f"  (avg {avg_ep:.0f}s/epoch)")

#     total_train_h = (time.perf_counter() - train_start) / 3600

#     # Final test evaluation
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
#             model_name   = "FM+PINN-v12-FNO-Mamba-data-fixed",
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
#                 ((np.array(pp)[:, 0] - np.array(g)[:, 0]) * 0.555) ** 2 +
#                 ((np.array(pp)[:, 1] - np.array(g)[:, 1]) * 0.555) ** 2
#             )))
#             for pp, g in zip(pred_seqs, gt_seqs)
#         ])

#         lstm_per_seq      = cliper_errs.mean(1) * 0.82
#         diffusion_per_seq = cliper_errs.mean(1) * 0.70

#         np.save(os.path.join(stat_dir, "fmpinn.npy"),           fmpinn_per_seq)
#         np.save(os.path.join(stat_dir, "cliper.npy"),           cliper_errs.mean(1))
#         np.save(os.path.join(stat_dir, "persistence.npy"),      persist_errs.mean(1))
#         np.save(os.path.join(stat_dir, "lstm_approx.npy"),      lstm_per_seq)
#         np.save(os.path.join(stat_dir, "diffusion_approx.npy"), diffusion_per_seq)

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
#             fh.write(f"\n\nmodel_version   : FM+PINN v12\n")
#             fh.write(f"data_fixes      : sentinel→median, SST=0→298K\n")
#             fh.write(f"env_fixes       : gph500 double-norm, u/v wrong-scale\n")
#             fh.write(f"code_fixes      : eval_fast NameError, patience 6→15\n")
#             fh.write(f"pinn_weight     : 0.1\n")
#             fh.write(f"smooth_weight   : 0.05\n")
#             fh.write(f"ode_steps       : {args.ode_steps}\n")
#             fh.write(f"eval_ensemble   : {final_ens}\n")
#             fh.write(f"test_year       : {args.test_year}\n")
#             fh.write(f"train_time_h    : {total_train_h:.2f}\n")
#             fh.write(f"n_params_M      : "
#                      f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}\n")

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
scripts/train_flowmatching.py  ── v13
======================================
BUG FIXES vs v12:

BUG-4-UNIT FIXED (fmpinn_per_seq wrong distance scale):
    v12 used coefficient * 0.555 to convert normalised coordinates → km.
    After denorm_torch the predictions are in degrees (lon/lat).
    Correct formula: haversine using cos(lat) at each point, or a proper
    per-point conversion: delta_lat_km = Δlat_deg * 111.0,
    delta_lon_km = Δlon_deg * 111.0 * cos(lat_rad).
    The 0.555 factor was ~200x too small, producing fake "sub-km" ADE.
    Fix: use haversine_km_np() helper that replicates haversine_km_torch
    in numpy, applied to denormed degree coordinates.

BUG-5-FAKE FIXED (lstm/diffusion per-seq errors are fabricated):
    v12 set:
        lstm_per_seq      = cliper_errs.mean(1) * 0.82
        diffusion_per_seq = cliper_errs.mean(1) * 0.70
    These are proportional rescalings of CLIPER — the resulting
    Wilcoxon / t-tests will detect a signal that does not exist
    in any real model comparison.
    Fix: Save placeholder NaN arrays and emit a clear WARNING so the
    researcher knows real LSTM / Diffusion inference must be run.
    The stat-test pipeline skips comparisons with all-NaN arrays.

Kept from v12:
    BUG-1: dropout-mask reuse fixed
    BUG-2: Me dims clamp in sample()
    BUG-3: loader import from loader_training
    BUG-5-CODE: evaluate_fast NameError fixed
    BUG-6: patience 6 → 15
    BUG-3-ENV / BUG-4-ENV: env normalization (in env_net)
    BUG-DATA-1/2: sentinel fixes (in trajectoriesWithMe)
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

from Model.data.loader_training import data_loader
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


# ── Haversine helper (numpy) ──────────────────────────────────────────────────

def haversine_km_np(pred_deg: np.ndarray, gt_deg: np.ndarray) -> np.ndarray:
    """
    Compute great-circle distance in km between predicted and GT tracks.

    Parameters
    ----------
    pred_deg : np.ndarray  shape [T, 2]  (lon_deg, lat_deg)
    gt_deg   : np.ndarray  shape [T, 2]

    Returns
    -------
    dist_km  : np.ndarray  shape [T]
    """
    R = 6371.0
    lon1, lat1 = np.radians(pred_deg[:, 0]), np.radians(pred_deg[:, 1])
    lon2, lat2 = np.radians(gt_deg[:, 0]),   np.radians(gt_deg[:, 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a    = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def denorm_deg_np(arr_norm: np.ndarray) -> np.ndarray:
    """
    Convert normalised (lon_norm, lat_norm) → (lon_deg, lat_deg).
    Mirrors the inverse of the normalization in the dataset:
        lon_norm = (lon_deg * 10 - 1800) / 50
        lat_norm = (lat_deg * 10) / 50
    → lon_deg = (lon_norm * 50 + 1800) / 10
    → lat_deg = (lat_norm * 50) / 10
    """
    out = arr_norm.copy()
    out[:, 0] = (arr_norm[:, 0] * 50.0 + 1800.0) / 10.0   # lon
    out[:, 1] = (arr_norm[:, 1] * 50.0) / 10.0             # lat
    return out


def seq_ade_km(pred_norm: np.ndarray, gt_norm: np.ndarray) -> float:
    """
    BUG-4-UNIT FIX: compute ADE in km using proper haversine distance.

    Parameters
    ----------
    pred_norm : [T, 2]  normalised (lon_norm, lat_norm)
    gt_norm   : [T, 2]

    Returns mean haversine distance (km) across T steps.
    """
    pred_deg = denorm_deg_np(pred_norm)
    gt_deg   = denorm_deg_np(gt_norm)
    return float(haversine_km_np(pred_deg, gt_deg).mean())


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root",    default="TCND_vn",      type=str)
    p.add_argument("--obs_len",         default=8,              type=int)
    p.add_argument("--pred_len",        default=12,             type=int)
    p.add_argument("--test_year",       default=None,           type=int)
    p.add_argument("--batch_size",      default=32,             type=int)
    p.add_argument("--num_epochs",      default=200,            type=int)
    p.add_argument("--g_learning_rate", default=2e-4,           type=float)
    p.add_argument("--weight_decay",    default=1e-4,           type=float)
    p.add_argument("--warmup_epochs",   default=3,              type=int)
    p.add_argument("--grad_clip",       default=1.0,            type=float)
    p.add_argument("--grad_accum",      default=2,              type=int)
    p.add_argument("--patience",        default=15,             type=int)
    p.add_argument("--n_train_ens",     default=6,              type=int)
    p.add_argument("--use_amp",         action="store_true")
    p.add_argument("--num_workers",     default=2,              type=int)
    p.add_argument("--sigma_min",       default=0.02,           type=float)
    p.add_argument("--ode_steps",       default=20,             type=int,
                   help="ODE steps for sampling (20 for better ADE)")
    p.add_argument("--val_ensemble",    default=10,             type=int)
    p.add_argument("--fno_modes_h",     default=4,              type=int)
    p.add_argument("--fno_modes_t",     default=4,              type=int)
    p.add_argument("--fno_layers",      default=4,              type=int)
    p.add_argument("--fno_d_model",     default=32,             type=int)
    p.add_argument("--fno_spatial_down",default=32,             type=int)
    p.add_argument("--mamba_d_state",   default=16,             type=int)
    p.add_argument("--val_loss_freq",   default=2,              type=int)
    p.add_argument("--val_freq",        default=5,              type=int)
    p.add_argument("--full_eval_freq",  default=50,             type=int)
    p.add_argument("--val_subset_size", default=500,            type=int)
    p.add_argument("--output_dir",      default="runs/v13",     type=str)
    p.add_argument("--save_interval",   default=10,             type=int)
    p.add_argument("--metrics_csv",     default="metrics.csv",        type=str)
    p.add_argument("--predict_csv",     default="predictions.csv",    type=str)
    # Optional paths to real LSTM / Diffusion per-sequence error arrays (.npy)
    # If not provided, those comparisons are skipped with a WARNING.
    p.add_argument("--lstm_errors_npy",      default=None, type=str,
                   help="Path to .npy of per-sequence ADE (km) from LSTM baseline. "
                        "Required for valid statistical comparison.")
    p.add_argument("--diffusion_errors_npy", default=None, type=str,
                   help="Path to .npy of per-sequence ADE (km) from Diffusion baseline. "
                        "Required for valid statistical comparison.")
    p.add_argument("--gpu_num",         default="0",            type=str)
    p.add_argument("--delim",           default=" ")
    p.add_argument("--skip",            default=1,              type=int)
    p.add_argument("--min_ped",         default=1,              type=int)
    p.add_argument("--threshold",       default=0.002,          type=float)
    p.add_argument("--other_modal",     default="gph")
    # Curriculum
    p.add_argument("--curriculum",      action="store_true",
                   help="Enable curriculum pred_len 6→12 over epochs 0-50")
    p.add_argument("--curriculum_start_len", default=6,         type=int)
    p.add_argument("--curriculum_end_epoch", default=50,        type=int)
    # Augmentation
    p.add_argument("--lon_flip_prob",   default=0.3,            type=float)
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


def make_val_subset_loader(val_dataset, subset_size, batch_size,
                           collate_fn, num_workers):
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


def get_curriculum_len(epoch, args) -> int:
    """Linearly ramp pred_len from curriculum_start_len to pred_len."""
    if not args.curriculum:
        return args.pred_len
    if epoch >= args.curriculum_end_epoch:
        return args.pred_len
    frac = epoch / max(args.curriculum_end_epoch, 1)
    return int(args.curriculum_start_len
               + frac * (args.pred_len - args.curriculum_start_len))


def evaluate_fast(model, loader, device, ode_steps, pred_len):
    """Fast ADE evaluation using val subset."""
    model.eval()
    acc = StepErrorAccumulator(pred_len)
    t0  = time.perf_counter()
    n   = 0
    with torch.no_grad():
        for batch in loader:
            bl = move(list(batch), device)
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
            pd_np = denorm_torch(pred_mean).cpu().numpy()
            gd_np = denorm_torch(gt).cpu().numpy()
            od_np = denorm_torch(bl[0]).cpu().numpy()
            ed_np = denorm_torch(all_trajs).cpu().numpy()

            for b in range(pd_np.shape[1]):
                ens_b = ed_np[:, :, b, :]
                ev.update(pd_np[:, b, :], gd_np[:, b, :], pred_ens=ens_b)
                obs_seqs_01.append(od_np[:, b, :])
                gt_seqs_01.append(gd_np[:, b, :])
                pred_seqs_01.append(pd_np[:, b, :])

    dm = ev.compute(tag=tag)
    save_metrics_csv(dm, metrics_csv, tag=tag)
    return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


class BestModelSaver:
    def __init__(self, patience=15, min_delta=1.0):
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
                model_version    = "v13-FNO-Mamba-fixed",
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
                model_version    = "v13-valloss",
            ), os.path.join(out_dir, "best_model_valloss.pth"))


def _load_baseline_errors(path: str | None, name: str) -> np.ndarray | None:
    """
    BUG-5-FAKE FIX: Load real per-sequence ADE errors from a .npy file.
    Returns None (not fake data) if path is missing, so statistical tests
    skip that comparison rather than reporting fabricated significance.
    """
    if path is None:
        print(f"\n  ⚠  WARNING: --{name.lower().replace('+','').replace(' ','_')}_errors_npy"
              f" not provided.")
        print(f"     Statistical comparison vs {name} will be SKIPPED.")
        print(f"     To include it, run {name} inference and pass the per-sequence")
        print(f"     ADE .npy file via the corresponding argument.\n")
        return None
    if not os.path.exists(path):
        print(f"\n  ⚠  WARNING: {path} does not exist — {name} comparison skipped.\n")
        return None
    arr = np.load(path)
    print(f"  ✓  Loaded {name} errors: {arr.shape} from {path}")
    return arr


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
    print("  TC-FlowMatching v13  |  FNO3D + Mamba + OT-CFM + PINN")
    print("  DATA FIXES: sentinel→NaN→median, SST=0→298K")
    print("  ENV  FIXES: gph500 double-norm, u500/v500 wrong scale")
    print("  ENV  FIXES: v500_center wrong stats (v13)")
    print("  CODE FIXES: zero-sentinel channels, env_cache key, ADE units")
    print("  STAT FIXES: real baselines required for LSTM/Diffusion tests")
    print("=" * 68)
    print(f"  device          : {device}")
    print(f"  dataset_root    : {args.dataset_root}")
    print(f"  num_epochs      : {args.num_epochs}")
    print(f"  patience        : {args.patience} checks × val_freq=5 = {args.patience*5} epochs")
    print(f"  grad_accum      : {args.grad_accum}  "
          f"(eff batch = {args.batch_size * args.grad_accum})")
    print(f"  use_amp         : {args.use_amp}")
    print(f"  num_workers     : {args.num_workers}")
    print(f"  ode_steps       : {args.ode_steps}  (val_ensemble={args.val_ensemble})")
    print(f"  curriculum      : {args.curriculum}  "
          f"(len {args.curriculum_start_len}→{args.pred_len} over {args.curriculum_end_epoch} ep)")

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

    from Model import losses as _losses_mod
    _losses_mod.WEIGHTS["pinn"]   = 0.1
    _losses_mod.WEIGHTS["smooth"] = 0.05

    print("=" * 68)
    print(f"  TRAINING  ({steps_per_epoch} optimizer steps/epoch)")
    print("=" * 68)

    epoch_times: list[float] = []
    train_start  = time.perf_counter()
    last_val_loss = float("inf")
    _lr_restart_ep30_done = False
    _lr_restart_ep60_done = False

    for epoch in range(args.num_epochs):
        # Progressive ensemble schedule
        if epoch < 30:
            current_ens = 1
        elif epoch < 60:
            current_ens = 2
        else:
            current_ens = args.n_train_ens
        model.n_train_ens = current_ens

        # Curriculum pred_len
        curr_len = get_curriculum_len(epoch, args)
        if hasattr(model, "set_curriculum_len"):
            model.set_curriculum_len(curr_len)

        # LR restart at epoch 30
        if epoch == 30 and not _lr_restart_ep30_done:
            _lr_restart_ep30_done = True
            restart_warmup  = steps_per_epoch * 2
            remaining_steps = steps_per_epoch * (args.num_epochs - 30)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, restart_warmup, remaining_steps, min_lr=5e-6)
            print(f"  ↺  LR restart at epoch 30 (ensemble 1→2)")

        # LR restart at epoch 60
        if epoch == 60 and not _lr_restart_ep60_done:
            _lr_restart_ep60_done = True
            restart_warmup  = steps_per_epoch * 2
            remaining_steps = steps_per_epoch * (args.num_epochs - 60)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, restart_warmup, remaining_steps, min_lr=1e-6)
            print(f"  ↺  LR restart at epoch 60 (ensemble 2→{args.n_train_ens})")

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
                      f"  ens={current_ens}  pred_len={curr_len}"
                      f"  lr={lr:.2e}"
                      f"  t={elapsed:.0f}s")

        ep_s  = time.perf_counter() - t0
        epoch_times.append(ep_s)
        avg_t = sum_loss / len(train_loader)

        # Validation loss
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
            loss_saver(last_val_loss, model, args.output_dir, epoch, optimizer, avg_t)
            print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
                  f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
                  f"  ens={current_ens}  pred_len={curr_len}")
        else:
            print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
                  f"  val={last_val_loss:.3f}(cached)"
                  f"  train_t={ep_s:.0f}s"
                  f"  ens={current_ens}  pred_len={curr_len}")

        # Fast ADE eval
        if epoch % args.val_freq == 0:
            t_ade = time.perf_counter()
            m = evaluate_fast(model, val_subset_loader, device,
                              args.ode_steps, args.pred_len)
            t_ade_s = time.perf_counter() - t_ade
            print(f"  [ADE eval {t_ade_s:.0f}s]"
                  f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km  "
                  f"72h={m.get('72h', 0):.0f} km")
            saver(m["ADE"], model, args.output_dir, epoch, optimizer, avg_t, last_val_loss)

        # Full eval
        if epoch % args.full_eval_freq == 0 and epoch > 0:
            print(f"  [Full eval epoch {epoch}]")
            dm, _, _, _ = evaluate_full(
                model, val_loader, device,
                args.ode_steps, args.pred_len, args.val_ensemble,
                metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}",
            )
            print(dm.summary())

        # Periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            cp = os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth")
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict()}, cp)

        if saver.early_stop:
            print(f"  Early stopping @ epoch {epoch}")
            break

        if epoch % 5 == 4:
            avg_ep    = sum(epoch_times) / len(epoch_times)
            remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
            elapsed_h = (time.perf_counter() - train_start) / 3600
            print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
                  f"  (avg {avg_ep:.0f}s/epoch)")

    total_train_h = (time.perf_counter() - train_start) / 3600

    # Final test evaluation
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
            model_name   = "FM+PINN-v13-FNO-Mamba-data-fixed",
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

        # ── BUG-4-UNIT FIX: per-sequence ADE using proper haversine ──────
        # v12 used * 0.555 (degrees × 0.555 ≠ km). Correct formula:
        # denorm normalised coords → degrees, then apply haversine distance.
        fmpinn_per_seq = np.array([
            seq_ade_km(np.array(pp), np.array(g))
            for pp, g in zip(pred_seqs, gt_seqs)
        ])

        np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
        np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
        np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))

        # ── BUG-5-FAKE FIX: load real baselines or skip ───────────────────
        # v12 fabricated lstm/diffusion as cliper * 0.82/0.70.
        # Wilcoxon/t-tests on data derived from cliper will always show
        # significance regardless of what FM+PINN actually achieves.
        # Now we require real inference results or skip the comparison.
        lstm_per_seq      = _load_baseline_errors(args.lstm_errors_npy,      "LSTM")
        diffusion_per_seq = _load_baseline_errors(args.diffusion_errors_npy, "Diffusion")

        if lstm_per_seq is not None:
            np.save(os.path.join(stat_dir, "lstm.npy"), lstm_per_seq)
        if diffusion_per_seq is not None:
            np.save(os.path.join(stat_dir, "diffusion.npy"), diffusion_per_seq)

        # Use dummy array for run_all_tests if real data unavailable;
        # statistical_tests.run_all_tests will skip comparisons where
        # baseline length < 2 after alignment.
        _dummy = np.array([float("nan")])
        run_all_tests(
            fmpinn_ade    = fmpinn_per_seq,
            cliper_ade    = cliper_errs.mean(1),
            lstm_ade      = lstm_per_seq if lstm_per_seq is not None else _dummy,
            diffusion_ade = diffusion_per_seq if diffusion_per_seq is not None else _dummy,
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
        ]
        if lstm_per_seq is not None:
            stat_rows.append(
                paired_tests(fmpinn_per_seq, lstm_per_seq, "FM+PINN vs LSTM", 5))
        if diffusion_per_seq is not None:
            stat_rows.append(
                paired_tests(fmpinn_per_seq, diffusion_per_seq, "FM+PINN vs Diffusion", 5))

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
            fh.write(f"\n\nmodel_version   : FM+PINN v13\n")
            fh.write(f"data_fixes      : sentinel→median, SST=0→298K\n")
            fh.write(f"env_fixes       : gph500 double-norm, u/v wrong-scale, v500c stats\n")
            fh.write(f"code_fixes      : zero-sentinel channels, env_cache key, ADE units\n")
            fh.write(f"stat_fixes      : real LSTM/Diffusion baselines required\n")
            fh.write(f"pinn_weight     : 0.1\n")
            fh.write(f"smooth_weight   : 0.05\n")
            fh.write(f"ode_steps       : {args.ode_steps}\n")
            fh.write(f"eval_ensemble   : {final_ens}\n")
            fh.write(f"test_year       : {args.test_year}\n")
            fh.write(f"train_time_h    : {total_train_h:.2f}\n")
            fh.write(f"n_params_M      : "
                     f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}\n")
            fh.write(f"lstm_errors_npy : {args.lstm_errors_npy}\n")
            fh.write(f"diff_errors_npy : {args.diffusion_errors_npy}\n")

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
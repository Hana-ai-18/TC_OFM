# """
# scripts/train_flowmatching.py  ── v23
# ======================================
# FIXES vs v22:

#   FIX-T23-1  CURRICULUM REMOVED (FIX-DATA-22).
#              Curriculum gây ADE tụt 282→444 km mỗi lần tăng len.
#              Thay bằng step_weight_alpha: giảm dần từ 1.0 → 0.0 theo epoch,
#              làm cho AFCRPS weight các bước gần hơn ở epoch đầu (soft curriculum),
#              KHÔNG cắt ngắn sequence pred_len=12.

#   FIX-T23-2  BestModelSaver: CHỈ dùng full_val ADE làm criteria lưu best model.
#              Subset ADE chỉ dùng để monitor nhanh, KHÔNG ảnh hưởng patience.
#              Khi full-val chưa chạy, saver giữ nguyên counter.

#   FIX-T23-3  ODE steps tăng: train=20, val=30, test=50 (từ 10/10/10).
#              10 steps quá thấp cho OT-CFM với 12-step trajectory.

#   FIX-T23-4  initial_sample_sigma: 0.1 (từ 0.3). 0.3 quá lớn trong
#              normalised space, gây spread bùng nổ ngay từ đầu.

#   FIX-T23-5  ctx_noise_scale: 0.02 (từ 0.05). Giảm context noise để
#              kiểm soát ensemble spread.

#   FIX-T23-6  patience: 15 (từ 6). Với val_ade_freq=2, patience=6 chỉ
#              tương đương 12 epoch thực → stop quá sớm sau curriculum jump.

#   FIX-T23-7  step_weight_alpha schedule: giảm từ 1.0 (ep 0) → 0.0 (ep 30)
#              tuyến tính. Sau ep 30 → uniform weights = standard AFCRPS.

#   FIX-T23-8  evaluate_full_val_ade: log thêm PINN loss trung bình để
#              verify PINN đang học (không còn = 100 constant).

#   FIX-T23-9  GPH500 verification: check mean trong range (27-90) thay vì
#              check == 0 (mean=-0.058 là pre-normed gph500 cũ, không đúng).
#              Với FIX-DATA-18, CSV gph500 là raw dam → mean ≈ 33 sau sentinel.

# Kept from v22:
#   FIX-V22-1  StepErrorAccumulator pad zeros (v5/v6)
#   FIX-V22-2  evaluate_full_val_ade mỗi val_ade_freq epoch
#   FIX-V22-3  Log active_steps từ accumulator
# """
# from __future__ import annotations

# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse
# import time
# import math
# import random
# import copy

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.amp import autocast, GradScaler
# from torch.utils.data import DataLoader, Subset

# from Model.data.loader_training import data_loader
# from Model.flow_matching_model import TCFlowMatching
# from Model.utils import get_cosine_schedule_with_warmup
# from Model.losses import WEIGHTS as _BASE_WEIGHTS
# from utils.metrics import (
#     TCEvaluator, StepErrorAccumulator,
#     save_metrics_csv, haversine_km_torch,
#     denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
#     cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
#     brier_skill_score, cliper_errors, persistence_errors,
# )
# from utils.evaluation_tables import (
#     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
#     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
#     DEFAULT_COMPUTE, paired_tests,
# )
# from scripts.statistical_tests import run_all_tests


# # ── Helpers ───────────────────────────────────────────────────────────────────

# def haversine_km_np_local(pred_deg: np.ndarray,
#                            gt_deg: np.ndarray) -> np.ndarray:
#     pred_deg = np.atleast_2d(pred_deg)
#     gt_deg   = np.atleast_2d(gt_deg)
#     R = 6371.0
#     lon1, lat1 = np.radians(pred_deg[:, 0]), np.radians(pred_deg[:, 1])
#     lon2, lat2 = np.radians(gt_deg[:, 0]),   np.radians(gt_deg[:, 1])
#     dlon = lon2 - lon1;  dlat = lat2 - lat1
#     a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
#     return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# def seq_ade_km(pred_norm: np.ndarray, gt_norm: np.ndarray) -> float:
#     return float(haversine_km_np_local(denorm_deg_np(pred_norm),
#                                        denorm_deg_np(gt_norm)).mean())


# # ── Adaptive weight schedules ─────────────────────────────────────────────────

# # def get_pinn_weight(epoch, warmup_epochs=30, w_start=0.001, w_end=0.05):
# #     """
# #     FIX-T23-8: w_start=0.001 (từ 0.01), w_end=0.05 (từ 0.1).
# #     PINN bắt đầu rất nhỏ để FM học trước, sau đó tăng dần.
# #     """
# #     if epoch >= warmup_epochs:
# #         return w_end
# #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)
# # def get_pinn_weight(epoch, warmup_epochs=30, w_start=0.01, w_end=0.05): # Sửa w_start ở đây
# #     if epoch >= warmup_epochs:
# #         return w_end
# #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)

 
# def get_pinn_weight(epoch, warmup_epochs=50, w_start=0.001, w_end=0.05):
#     """
#     FIX-T24-1: warmup dài hơn (50 ep) để PINN học sau khi FM ổn định.
#     w_start giảm hơn để không interfere với FM học ban đầu.
#     """
#     if epoch >= warmup_epochs:
#         return w_end
#     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)

# def get_velocity_weight(epoch, warmup_epochs=20, w_start=0.5, w_end=1.5):
#     if epoch >= warmup_epochs:
#         return w_end
#     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)

 
 
# def get_progressive_ens(epoch, n_train_ens=6):
#     """
#     FIX-T24-4: Progressive ensemble tăng chậm hơn.
#     ep 0-19: 1 sample (fast training)
#     ep 20-49: 2 samples (begin diversity)
#     ep 50+: n_train_ens (full ensemble)
#     """
#     if epoch < 20:
#         return 1
#     elif epoch < 50:
#         return 2
#     else:
#         return n_train_ens
 
 
# def _check_uv500(bl):
#     """
#     FIX-T24-6: Verify u/v500 sau FIX-ENV-20 và FIX-DATA-28.
#     Expected: mean trong range [-1, 1] và không phải 0.
#     """
#     env_data = bl[13]
#     if env_data is None:
#         print("  ⚠️  env_data is None at epoch 0")
#         return
 
#     # Check u500
#     for key in ("u500_mean", "v500_mean"):
#         if key not in env_data:
#             print(f"  ⚠️  {key} không có trong env_data!")
#             continue
 
#         val = env_data[key]
#         n_zero  = (val == 0).sum().item()
#         n_total = val.numel()
#         zero_pct = 100.0 * n_zero / max(n_total, 1)
#         mean_val = val.mean().item()
#         std_val  = val.std().item()
 
#         if zero_pct > 80.0:
#             print(f"\n{'!' * 60}")
#             print(f"  ⚠️  {key} = 0 cho {zero_pct:.1f}% samples!")
#             print(f"     mean={mean_val:.4f}, std={std_val:.4f}")
#             print(f"     FIX-DATA-28 chưa được apply hoặc CSV không có d3d_u500_mean_raw")
#             print(f"{'!' * 60}\n")
#         elif abs(mean_val) < 0.01 and std_val < 0.01:
#             print(f"  ⚠️  {key} gần 0 (mean={mean_val:.4f}, std={std_val:.4f})")
#         else:
#             print(f"  ✅ {key} OK: mean={mean_val:.4f}, std={std_val:.4f}, zero={zero_pct:.1f}%")
 
#     # Check gph500
#     if "gph500_mean" in env_data:
#         gph_val = env_data["gph500_mean"]
#         gph_mean = gph_val.mean().item()
#         zero_pct = 100.0 * (gph_val == 0).sum().item() / max(gph_val.numel(), 1)
 
#         if abs(gph_mean) < 1.0 and zero_pct > 50.0:
#             print(f"  ⚠️  GPH500 mean≈0 ({gph_mean:.4f}) → data issue")
#         elif 25.0 < gph_mean < 95.0:
#             print(f"  ✅ GPH500 OK (mean={gph_mean:.2f} dam, zero={zero_pct:.1f}%)")
#         elif -30.0 < gph_mean < 5.0:
#             print(f"  ℹ️  GPH500 pre-normalized (mean={gph_mean:.4f}) - OK if from .npy")
#         else:
#             print(f"  ⚠️  GPH500 unexpected (mean={gph_mean:.4f})")
 

# def get_recurv_weight(epoch, warmup_epochs=10, w_start=0.3, w_end=1.0):
#     if epoch >= warmup_epochs:
#         return w_end
#     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# def get_grad_clip(epoch, warmup_epochs=20, clip_start=2.0, clip_end=1.0):
#     if epoch >= warmup_epochs:
#         return clip_end
#     return clip_start - (epoch / max(warmup_epochs-1, 1)) * (clip_start - clip_end)


# def get_step_weight_alpha(epoch, decay_epochs=60) -> float: # 30→60
#     """
#     FIX-T23-7: step_weight_alpha replaces curriculum.
#     alpha=1.0 at ep 0 → 0.0 at ep decay_epochs.
#     After decay_epochs: uniform AFCRPS weights.
#     """
#     if epoch >= decay_epochs:
#         return 0.0
#     return 1.0 - (epoch / decay_epochs)


# def get_args():
#     p = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     p.add_argument("--dataset_root",    default="TCND_vn",      type=str)
#     p.add_argument("--obs_len",         default=8,              type=int)
#     p.add_argument("--pred_len",        default=12,             type=int)
#     p.add_argument("--test_year",       default=None,           type=int)
#     p.add_argument("--batch_size",      default=32,             type=int)
#     p.add_argument("--num_epochs",      default=200,            type=int)
#     p.add_argument("--g_learning_rate", default=2e-4,           type=float)
#     p.add_argument("--weight_decay",    default=1e-4,           type=float)
#     p.add_argument("--warmup_epochs",   default=3,              type=int)
#     p.add_argument("--grad_clip",       default=2.0,            type=float)
#     p.add_argument("--grad_accum",      default=2,              type=int)
#     p.add_argument("--patience",        default=15,             type=int,
#                    help="FIX-T23-6: tăng từ 6 lên 15 để tránh stop sớm.")
#     p.add_argument("--min_epochs",      default=80,             type=int)
#     p.add_argument("--n_train_ens",     default=6,              type=int)
#     p.add_argument("--use_amp",         action="store_true")
#     p.add_argument("--num_workers",     default=2,              type=int)

#     # FIX-T23-4/5: sigma giảm
#     p.add_argument("--sigma_min",            default=0.02,  type=float)
#     # p.add_argument("--ctx_noise_scale",      default=0.02,  type=float,
#     #                help="FIX-T23-5: giảm từ 0.05 → 0.02")
#     # p.add_argument("--initial_sample_sigma", default=0.1,   type=float,
#     #                help="FIX-T23-4: giảm từ 0.3 → 0.1")
#     # train_flowmatching.py — get_args()
#     p.add_argument("--ctx_noise_scale",      default=0.005,  type=float,
#                 help="FIX-T24-B: giảm từ 0.02 → 0.005, kiểm soát spread")
#     p.add_argument("--initial_sample_sigma", default=0.05,   type=float,
#                 help="FIX-T24-B: giảm từ 0.1 → 0.05")

#     # FIX-T23-3: ODE steps tăng
#     p.add_argument("--ode_steps_train", default=20,  type=int,
#                    help="FIX-T23-3: từ 10 → 20")
#     p.add_argument("--ode_steps_val",   default=30,  type=int,
#                    help="FIX-T23-3: từ 10 → 30")
#     p.add_argument("--ode_steps_test",  default=50,  type=int)
#     p.add_argument("--ode_steps",       default=None, type=int,
#                    help="Override train/val/test steps (for testing)")

#     p.add_argument("--val_ensemble",    default=30,             type=int)
#     p.add_argument("--fast_ensemble",   default=8,              type=int)

#     p.add_argument("--fno_modes_h",      default=4,             type=int)
#     p.add_argument("--fno_modes_t",      default=4,             type=int)
#     p.add_argument("--fno_layers",       default=4,             type=int)
#     p.add_argument("--fno_d_model",      default=32,            type=int)
#     p.add_argument("--fno_spatial_down", default=32,            type=int)
#     p.add_argument("--mamba_d_state",    default=16,            type=int)

#     p.add_argument("--val_loss_freq",   default=1,              type=int)
#     p.add_argument("--val_freq",        default=1,              type=int)
#     p.add_argument("--val_ade_freq",    default=1,              type=int)
#     p.add_argument("--full_eval_freq",  default=10,             type=int)
#     p.add_argument("--val_subset_size", default=600,            type=int)

#     p.add_argument("--output_dir",      default="runs/v23",     type=str)
#     p.add_argument("--save_interval",   default=10,             type=int)
#     p.add_argument("--metrics_csv",     default="metrics.csv",     type=str)
#     p.add_argument("--predict_csv",     default="predictions.csv", type=str)
#     p.add_argument("--lstm_errors_npy",      default=None, type=str)
#     p.add_argument("--diffusion_errors_npy", default=None, type=str)

#     p.add_argument("--gpu_num",         default="0",            type=str)
#     p.add_argument("--delim",           default=" ")
#     p.add_argument("--skip",            default=1,              type=int)
#     p.add_argument("--min_ped",         default=1,              type=int)
#     p.add_argument("--threshold",       default=0.002,          type=float)
#     p.add_argument("--other_modal",     default="gph")

#     # FIX-T23-1: curriculum params REMOVED, replaced by step_weight_alpha
#     # p.add_argument("--step_weight_decay_epochs", default=30, type=int,
#     #                help="FIX-T23-7: epochs over which alpha decays 1→0")

#     p.add_argument("--step_weight_decay_epochs", default=60, type=int,   # 30→60
#                help="FIX-T24-A: epochs over which alpha decays 1→0")
    
#     p.add_argument("--lon_flip_prob",   default=0.3,            type=float)

#     # PINN warmup with smaller values
#     p.add_argument("--pinn_warmup_epochs", default=30,          type=int)
#     p.add_argument("--pinn_w_start",    default=0.001,          type=float,
#                    help="FIX-T23-8: 0.001 (từ 0.01)")
#     p.add_argument("--pinn_w_end",      default=0.05,           type=float,
#                    help="FIX-T23-8: 0.05 (từ 0.1)")

#     p.add_argument("--vel_warmup_epochs",  default=20,          type=float)
#     p.add_argument("--vel_w_start",        default=0.5,         type=float)
#     p.add_argument("--vel_w_end",          default=1.5,         type=float)
#     p.add_argument("--recurv_warmup_epochs", default=10,        type=int)
#     p.add_argument("--recurv_w_start",       default=0.3,       type=float)
#     p.add_argument("--recurv_w_end",         default=1.0,       type=float)

#     return p.parse_args()


# def _resolve_ode_steps(args):
#     if args.ode_steps is not None:
#         return args.ode_steps, args.ode_steps, args.ode_steps
#     return args.ode_steps_train, args.ode_steps_val, args.ode_steps_test


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
#                             collate_fn, num_workers):
#     n   = len(val_dataset)
#     rng = random.Random(42)
#     idx = rng.sample(range(n), min(subset_size, n))
#     return DataLoader(Subset(val_dataset, idx),
#                       batch_size=batch_size, shuffle=False,
#                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # ── evaluate_fast ─────────────────────────────────────────────────────────────

# # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
# #     """Monitor nhanh trên val subset. Không dùng để quyết định best model."""
# #     model.eval()
# #     acc = StepErrorAccumulator(pred_len)
# #     t0  = time.perf_counter()
# #     n   = 0
# #     spread_buf = []

# #     with torch.no_grad():
# #         for batch in loader:
# #             bl = move(list(batch), device)
# #             pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
# #                                               ddim_steps=ode_steps)
# #             T_active  = pred.shape[0]
# #             gt_sliced = bl[1][:T_active]
# #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
# #             acc.update(dist)

# #             last_step = all_trajs[:, -1, :, :]
# #             std_lon   = last_step[:, :, 0].std(0)
# #             std_lat   = last_step[:, :, 1].std(0)
# #             spread_km = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# #             spread_buf.append(spread_km)
# #             n += 1

# #     r = acc.compute()
# #     r["ms_per_batch"]  = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# #     r["spread_72h_km"] = float(np.mean(spread_buf)) if spread_buf else 0.0
# #     return r

# def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
#     """
#     FIX-T24-7: Log thêm per-step spread để monitor FIX-L46.
#     """
#     import torch
#     import numpy as np
#     import time
 
#     model.eval()
#     from utils.metrics import StepErrorAccumulator, haversine_km_torch, denorm_torch
#     acc = StepErrorAccumulator(pred_len)
#     t0  = time.perf_counter()
#     n   = 0
#     spread_per_step = []
 
#     with torch.no_grad():
#         for batch in loader:
#             bl = list(batch)
#             for i, x in enumerate(bl):
#                 if torch.is_tensor(x):
#                     bl[i] = x.to(device)
#                 elif isinstance(x, dict):
#                     bl[i] = {k: v.to(device) if torch.is_tensor(v) else v
#                               for k, v in x.items()}
 
#             pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
#                                               ddim_steps=ode_steps)
#             T_active  = pred.shape[0]
#             gt_sliced = bl[1][:T_active]
#             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
#             acc.update(dist)
 
#             # FIX-T24-7: per-step spread
#             step_spreads = []
#             for t in range(all_trajs.shape[1]):
#                 step_data = all_trajs[:, t, :, :]
#                 std_lon = step_data[:, :, 0].std(0)
#                 std_lat = step_data[:, :, 1].std(0)
#                 spread = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
#                 step_spreads.append(spread)
#             spread_per_step.append(step_spreads)
#             n += 1
 
#     r = acc.compute()
#     r["ms_per_batch"]  = (time.perf_counter() - t0) * 1e3 / max(n, 1)
 
#     if spread_per_step:
#         spreads = np.array(spread_per_step)  # [n_batches, T]
#         r["spread_12h_km"] = float(spreads[:, 1].mean()) if spreads.shape[1] > 1 else 0.0
#         r["spread_24h_km"] = float(spreads[:, 3].mean()) if spreads.shape[1] > 3 else 0.0
#         r["spread_72h_km"] = float(spreads[:, -1].mean())
#     return r

# # ── evaluate_full_val_ade ─────────────────────────────────────────────────────

# def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
#                            fast_ensemble, metrics_csv, epoch, tag=""):
#     """
#     FIX-T23-2: Full val ADE là DUY NHẤT criteria để lưu best model.
#     FIX-T23-8: Log PINN loss trung bình để verify không còn saturate.
#     """
#     model.eval()
#     acc      = StepErrorAccumulator(pred_len)
#     t0       = time.perf_counter()
#     n_batch  = 0
#     pinn_buf = []

#     with torch.no_grad():
#         for batch in val_loader:
#             bl = move(list(batch), device)

#             # Luôn sample full pred_len (không curriculum)
#             pred, _, _ = model.sample(bl, num_ensemble=fast_ensemble,
#                                        ddim_steps=ode_steps)
#             T_pred = pred.shape[0]
#             gt     = bl[1][:T_pred]
#             dist   = haversine_km_torch(denorm_torch(pred), denorm_torch(gt))
#             acc.update(dist)

#             # FIX-T23-8: compute PINN để verify
#             try:
#                 from Model.losses import pinn_bve_loss, _haversine_deg
#                 from utils.metrics import denorm_deg_np
#                 pred_deg = pred.clone()
#                 pred_deg[..., 0] = (pred[..., 0] * 50.0 + 1800.0) / 10.0
#                 pred_deg[..., 1] = (pred[..., 1] * 50.0) / 10.0
#                 env_d = bl[13] if len(bl) > 13 else None
#                 pinn_val = pinn_bve_loss(pred_deg, bl, env_data=env_d).item()
#                 pinn_buf.append(pinn_val)
#             except Exception:
#                 pass

#             n_batch += 1

#     elapsed = time.perf_counter() - t0
#     r       = acc.compute()

#     ade_str = f"{r.get('ADE', float('nan')):.1f}"
#     fde_str = f"{r.get('FDE', float('nan')):.1f}"
#     h12     = f"{r.get('12h', float('nan')):.0f}"
#     h24     = f"{r.get('24h', float('nan')):.0f}"
#     h48     = f"{r.get('48h', float('nan')):.0f}"
#     h72     = f"{r.get('72h', float('nan')):.0f}"
#     pinn_mean = f"{np.mean(pinn_buf):.3f}" if pinn_buf else "N/A"

#     print(f"\n{'='*64}")
#     print(f"  [FULL VAL ADE  ep={epoch}  {elapsed:.0f}s  {n_batch} batches]")
#     print(f"  ADE={ade_str} km  FDE={fde_str} km")
#     print(f"  12h={h12}  24h={h24}  48h={h48}  72h={h72} km")
#     print(f"  PINN_mean={pinn_mean}  "  # FIX-T23-8
#           f"samples={r.get('n_samples',0)}  ens={fast_ensemble}  steps={ode_steps}")
#     print(f"{'='*64}\n")

#     from datetime import datetime
#     tag_str = tag or f"val_full_ep{epoch:03d}"
#     from utils.metrics import DatasetMetrics, save_metrics_csv
#     dm = DatasetMetrics(
#         ade      = r.get("ADE",  float("nan")),
#         fde      = r.get("FDE",  float("nan")),
#         ugde_12h = r.get("12h",  float("nan")),
#         ugde_24h = r.get("24h",  float("nan")),
#         ugde_48h = r.get("48h",  float("nan")),
#         ugde_72h = r.get("72h",  float("nan")),
#         n_total  = r.get("n_samples", 0),
#         timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
#     )
#     save_metrics_csv(dm, metrics_csv, tag=tag_str)
#     return r


# def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
#                   metrics_csv, tag="", predict_csv=""):
#     """Full 4-tier evaluation."""
#     model.eval()
#     cliper_step_errors = []
#     ev = TCEvaluator(pred_len=pred_len, compute_dtw=True)
#     obs_seqs_01 = []; gt_seqs_01 = []; pred_seqs_01 = []; ens_seqs_01 = []

#     with torch.no_grad():
#         for batch in loader:
#             bl  = move(list(batch), device)
#             gt  = bl[1];  obs = bl[0]
#             pred_mean, _, all_trajs = model.sample(
#                 bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
#                 predict_csv=predict_csv if predict_csv else None)

#             pd_np = denorm_torch(pred_mean).cpu().numpy()
#             gd_np = denorm_torch(gt).cpu().numpy()
#             od_np = denorm_torch(obs).cpu().numpy()
#             ed_np = denorm_torch(all_trajs).cpu().numpy()

#             B = pd_np.shape[1]
#             for b in range(B):
#                 ens_b = ed_np[:, :, b, :]
#                 ev.update(pd_np[:, b, :], gd_np[:, b, :], pred_ens=ens_b)
#                 obs_seqs_01.append(od_np[:, b, :])
#                 gt_seqs_01.append(gd_np[:, b, :])
#                 pred_seqs_01.append(pd_np[:, b, :])
#                 ens_seqs_01.append(ens_b)

#                 obs_b_norm = obs.cpu().numpy()[:, b, :]
#                 cliper_errors_b = np.zeros(pred_len)
#                 for h in range(pred_len):
#                     pred_cliper_norm = cliper_forecast(obs_b_norm, h + 1)
#                     pred_cliper_01   = denorm_np(pred_cliper_norm[np.newaxis])
#                     gt_01            = gd_np[h, b, :][np.newaxis]
#                     from utils.metrics import haversine_km_np
#                     cliper_errors_b[h] = float(
#                         haversine_km_np(pred_cliper_01, gt_01, unit_01deg=True)[0])
#                 cliper_step_errors.append(cliper_errors_b)

#     if cliper_step_errors:
#         cliper_mat       = np.stack(cliper_step_errors)
#         cliper_ugde_dict = {h: float(cliper_mat[:, s].mean())
#                             for h, s in HORIZON_STEPS.items()
#                             if s < cliper_mat.shape[1]}
#         ev.cliper_ugde   = cliper_ugde_dict
#         print(f"  [CLIPER UGDE] 72h={cliper_ugde_dict.get(72, float('nan')):.1f} km")

#     dm = ev.compute(tag=tag)

#     try:
#         if LANDFALL_TARGETS and ens_seqs_01:
#             bss_vals = []
#             step_72  = HORIZON_STEPS.get(72, pred_len - 1)
#             for tname, t_lon, t_lat in LANDFALL_TARGETS:
#                 bv = brier_skill_score(
#                     ens_seqs_01, gt_seqs_01, min(step_72, pred_len-1),
#                     (t_lon * 10.0, t_lat * 10.0), LANDFALL_RADIUS_KM)
#                 if not math.isnan(bv):
#                     bss_vals.append(bv)
#             if bss_vals:
#                 dm.bss_mean = float(np.mean(bss_vals))
#                 print(f"  [BSS] mean={dm.bss_mean:.4f}")
#     except Exception as e:
#         print(f"  ⚠  BSS failed: {e}")

#     save_metrics_csv(dm, metrics_csv, tag=tag)
#     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# # ── BestModelSaver ────────────────────────────────────────────────────────────

# class BestModelSaver:
#     """
#     FIX-T23-2: CHỈ dùng full_val ADE để quyết định best model và patience.
#     Subset ADE không ảnh hưởng đến saver logic.
#     """

#     def __init__(self, patience=15, ade_tol=5.0):
#         self.patience      = patience
#         self.ade_tol       = ade_tol
#         self.best_ade      = float("inf")
#         self.best_val_loss = float("inf")
#         self.counter_ade   = 0
#         self.counter_loss  = 0
#         self.early_stop    = False

#     def reset_counters(self, reason=""):
#         self.counter_ade  = 0
#         self.counter_loss = 0
#         if reason:
#             print(f"  [SAVER] Patience reset: {reason}")

#     def update_val_loss(self, val_loss, model, out_dir, epoch, optimizer, tl):
#         if val_loss < self.best_val_loss - 1e-4:
#             self.best_val_loss = val_loss
#             self.counter_loss  = 0
#             torch.save(dict(
#                 epoch=epoch, model_state_dict=model.state_dict(),
#                 optimizer_state=optimizer.state_dict(),
#                 train_loss=tl, val_loss=val_loss,
#                 model_version="v23-valloss"),
#                 os.path.join(out_dir, "best_model_valloss.pth"))
#         else:
#             self.counter_loss += 1

#     def update_ade_full_val(self, ade, model, out_dir, epoch,
#                              optimizer, tl, vl, min_epochs=80):
#         """
#         FIX-T23-2: Chỉ gọi khi có full_val ADE. Đây là criteria duy nhất.
#         """
#         if ade < self.best_ade - self.ade_tol:
#             self.best_ade     = ade
#             self.counter_ade  = 0
#             torch.save(dict(
#                 epoch=epoch, model_state_dict=model.state_dict(),
#                 optimizer_state=optimizer.state_dict(),
#                 train_loss=tl, val_loss=vl, val_ade_km=ade,
#                 model_version="v23-FNO-Mamba-OT-CFM"),
#                 os.path.join(out_dir, "best_model.pth"))
#             print(f"  ✅ Best full-val ADE {ade:.1f} km  (epoch {epoch})")
#         else:
#             self.counter_ade += 1
#             print(f"  No ADE improvement {self.counter_ade}/{self.patience}"
#                   f"  (Δ={self.best_ade - ade:.1f} km < tol={self.ade_tol} km)"
#                   f"  | Loss counter {self.counter_loss}/{self.patience}"
#                   f"  [full_val]")

#         if epoch >= min_epochs:
#             if (self.counter_ade >= self.patience
#                     and self.counter_loss >= self.patience):
#                 self.early_stop = True
#                 print(f"  ⛔ Early stop @ epoch {epoch}")
#         else:
#             if (self.counter_ade >= self.patience
#                     and self.counter_loss >= self.patience):
#                 print(f"  ⚠  Would stop but min_epochs={min_epochs} not reached.")
#                 self.counter_ade  = 0
#                 self.counter_loss = 0

#     def log_subset_ade(self, ade: float, epoch: int):
#         """FIX-T23-2: subset ADE chỉ để log, không ảnh hưởng patience."""
#         print(f"  [SUBSET-ADE ep{epoch}]  {ade:.1f} km  (monitor only, not used for best model)")


# def _load_baseline_errors(path, name):
#     if path is None:
#         print(f"\n  ⚠  {name} errors not provided — skip stat comparison.\n")
#         return None
#     if not os.path.exists(path):
#         print(f"\n  ⚠  {path} not found — {name} skipped.\n")
#         return None
#     arr = np.load(path)
#     print(f"  ✓  Loaded {name}: {arr.shape}")
#     return arr


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

#     ode_train, ode_val, ode_test = _resolve_ode_steps(args)

#     print("=" * 68)
#     print("  TC-FlowMatching v23  |  FNO3D + Mamba + OT-CFM + PINN")
#     print("  v23 FIXES:")
#     print("    FIX-T23-1: CURRICULUM REMOVED → step_weight_alpha soft weighting")
#     print("    FIX-T23-2: Best model CHỈ từ full_val ADE (không dùng subset)")
#     print("    FIX-T23-3: ODE steps train=20, val=30 (từ 10/10)")
#     print("    FIX-T23-4: initial_sample_sigma=0.1 (từ 0.3)")
#     print("    FIX-T23-5: ctx_noise_scale=0.02 (từ 0.05)")
#     print("    FIX-T23-6: patience=15 (từ 6)")
#     print("    FIX-T23-7: step_weight_alpha 1.0→0.0 over 30 epochs")
#     print("    FIX-T23-8: PINN w_start=0.001 (từ 0.01), log PINN trong eval")
#     print("    FIX-DATA-18/21: GPH500 từ CSV = raw dam, xử lý đúng")
#     print("    FIX-L39/42: PINN scale=1e-3, clamp=50 → gradient không saturate")
#     print("=" * 68)
#     print(f"  device               : {device}")
#     print(f"  sigma_min            : {args.sigma_min}")
#     print(f"  ctx_noise_scale      : {args.ctx_noise_scale}")
#     print(f"  initial_sample_sigma : {args.initial_sample_sigma}")
#     print(f"  ode_steps            : train={ode_train}  val={ode_val}  test={ode_test}")
#     print(f"  val_ensemble         : {args.val_ensemble}")
#     print(f"  val_ade_freq         : every {args.val_ade_freq} epochs (full val set)")
#     print(f"  patience             : {args.patience} epochs  (min_epochs={args.min_epochs})")
#     print(f"  step_weight_decay    : {args.step_weight_decay_epochs} epochs")
#     print(f"  NO CURRICULUM        : pred_len={args.pred_len} from epoch 0")
#     print()

#     train_dataset, train_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "train"}, test=False)
#     val_dataset, val_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "val"}, test=True)

#     from Model.data.trajectoriesWithMe_unet_training import seq_collate
#     val_subset_loader = make_val_subset_loader(
#         val_dataset, args.val_subset_size, args.batch_size,
#         seq_collate, args.num_workers)

#     test_loader = None
#     try:
#         _, test_loader = data_loader(
#             args, {"root": args.dataset_root, "type": "test"},
#             test=True, test_year=None)
#     except Exception as e:
#         print(f"  Warning: test loader: {e}")

#     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
#     print(f"  val   : {len(val_dataset)} seq")
#     if test_loader:
#         print(f"  test  : {len(test_loader.dataset)} seq")

#     model = TCFlowMatching(
#         pred_len             = args.pred_len,
#         obs_len              = args.obs_len,
#         sigma_min            = args.sigma_min,
#         n_train_ens          = args.n_train_ens,
#         ctx_noise_scale      = args.ctx_noise_scale,
#         initial_sample_sigma = args.initial_sample_sigma,
#     ).to(device)

#     if (args.fno_spatial_down != 32 or args.fno_modes_h != 4
#             or args.fno_layers != 4 or args.fno_d_model != 32):
#         from Model.FNO3D_encoder import FNO3DEncoder
#         model.net.spatial_enc = FNO3DEncoder(
#             in_channel=13, out_channel=1,
#             d_model=args.fno_d_model, n_layers=args.fno_layers,
#             modes_t=args.fno_modes_t, modes_h=args.fno_modes_h,
#             modes_w=args.fno_modes_h, spatial_down=args.fno_spatial_down,
#             dropout=0.05).to(device)

#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"  params  : {n_params:,}")

#     try:
#         model = torch.compile(model, mode="reduce-overhead")
#         print("  torch.compile: enabled")
#     except Exception:
#         pass

#     optimizer       = optim.AdamW(model.parameters(),
#                                    lr=args.g_learning_rate,
#                                    weight_decay=args.weight_decay)
#     steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
#     total_steps     = steps_per_epoch * args.num_epochs
#     warmup          = steps_per_epoch * args.warmup_epochs
#     scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
#     saver     = BestModelSaver(patience=args.patience, ade_tol=1.0)
#     scaler    = GradScaler('cuda', enabled=args.use_amp)

#     print("=" * 68)
#     print(f"  TRAINING  ({steps_per_epoch} steps/epoch, NO CURRICULUM)")
#     print("=" * 68)

#     epoch_times   = []
#     train_start   = time.perf_counter()
#     last_val_loss = float("inf")
#     _lr_ep30_done = False
#     _lr_ep60_done = False
#     _prev_ens     = 1

#     import Model.losses as _losses_mod

#     for epoch in range(args.num_epochs):
#         # Progressive ensemble
#         # current_ens = 1 if epoch < 30 else (2 if epoch < 60 else args.n_train_ens)
#         current_ens = get_progressive_ens(epoch, args.n_train_ens)
#         model.n_train_ens = current_ens
#         effective_fast_ens = min(args.fast_ensemble, max(current_ens*2, args.fast_ensemble))

#         if current_ens != _prev_ens:
#             saver.reset_counters(f"n_train_ens {_prev_ens}→{current_ens} at ep {epoch}")
#             _prev_ens = current_ens

#         # FIX-T23-1: NO curriculum. Always train on full pred_len.
#         # FIX-T23-7: step_weight_alpha replaces curriculum
#         step_alpha = get_step_weight_alpha(epoch, args.step_weight_decay_epochs)

#         # Weight schedule
#         epoch_weights = copy.copy(_BASE_WEIGHTS)
#         epoch_weights["pinn"] = get_pinn_weight(
#             epoch, args.pinn_warmup_epochs, args.pinn_w_start, args.pinn_w_end)
#         epoch_weights["velocity"] = get_velocity_weight(
#             epoch, args.vel_warmup_epochs, args.vel_w_start, args.vel_w_end)
#         epoch_weights["recurv"]   = get_recurv_weight(
#             epoch, args.recurv_warmup_epochs, args.recurv_w_start, args.recurv_w_end)
#         _losses_mod.WEIGHTS.update(epoch_weights)
#         if hasattr(model, 'weights'):
#             model.weights = epoch_weights

#         current_clip = get_grad_clip(epoch, warmup_epochs=20,
#                                       clip_start=args.grad_clip, clip_end=1.0)

#         # LR warm restarts
#         if epoch == 30 and not _lr_ep30_done:
#             _lr_ep30_done = True
#             scheduler = get_cosine_schedule_with_warmup(
#                 optimizer, steps_per_epoch,
#                 steps_per_epoch * (args.num_epochs - 30), min_lr=5e-6)
#             saver.reset_counters("LR warm restart at epoch 30")
#             print(f"  ↺  Warm Restart LR at epoch 30")

#         if epoch == 60 and not _lr_ep60_done:
#             _lr_ep60_done = True
#             scheduler = get_cosine_schedule_with_warmup(
#                 optimizer, steps_per_epoch,
#                 steps_per_epoch * (args.num_epochs - 60), min_lr=1e-6)
#             saver.reset_counters("LR warm restart at epoch 60")
#             print(f"  ↺  Warm Restart LR at epoch 60")

#         # ── Training loop ─────────────────────────────────────────────────────
#         model.train()
#         sum_loss      = 0.0
#         t0            = time.perf_counter()
#         optimizer.zero_grad()
#         recurv_ratio_buf = []

#         for i, batch in enumerate(train_loader):
#             bl = move(list(batch), device)

#             if epoch == 0 and i == 0:
#                 _check_gph500(bl, train_dataset)
#                 _check_uv500(bl)

#             with autocast(device_type='cuda', enabled=args.use_amp):
#                 # FIX-T23-1: pass step_weight_alpha to model
#                 bd = model.get_loss_breakdown(bl, step_weight_alpha=step_alpha)

#             loss_to_backpass = bd["total"] / max(args.grad_accum, 1)
#             scaler.scale(loss_to_backpass).backward()

#             if ((i + 1) % args.grad_accum == 0
#                     or (i + 1) == len(train_loader)):
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(
#                     model.parameters(), current_clip)
#                 scaler.step(optimizer)
#                 scaler.update()
#                 scheduler.step()
#                 optimizer.zero_grad()

#             sum_loss += bd["total"].item()
#             if "recurv_ratio" in bd:
#                 recurv_ratio_buf.append(bd["recurv_ratio"])

#             if i % 20 == 0:
#                 lr       = optimizer.param_groups[0]["lr"]
#                 rr       = bd.get("recurv_ratio", 0.0)
#                 elapsed  = time.perf_counter() - t0
#                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
#                       f"  loss={bd['total'].item():.3f}"
#                       f"  fm={bd.get('fm',0):.2f}"
#                       f"  vel={bd.get('velocity',0):.4f}"
#                       f"  pinn={bd.get('pinn', 0):.4f}"  # FIX-T23-8: more precision
#                       f"  recurv={bd.get('recurv',0):.3f}"
#                       f"  rr={rr:.2f}"
#                       f"  pinn_w={epoch_weights['pinn']:.4f}"
#                       f"  alpha={step_alpha:.2f}"
#                       f"  clip={current_clip:.1f}"
#                       f"  ens={current_ens}"
#                       f"  lr={lr:.2e}  t={elapsed:.0f}s")

#         ep_s    = time.perf_counter() - t0
#         epoch_times.append(ep_s)
#         avg_t   = sum_loss / len(train_loader)
#         mean_rr = float(np.mean(recurv_ratio_buf)) if recurv_ratio_buf else 0.0

#         # ── Val loss ───────────────────────────────────────────────────────────
#         # if epoch % args.val_freq == 0:
#         #     model.eval()
#         #     val_loss = 0.0
#         #     t_val    = time.perf_counter()
#         #     with torch.no_grad():
#         #         for batch in val_loader:
#         #             bl_v = move(list(batch), device)
#         #             with autocast(device_type='cuda', enabled=args.use_amp):
#         #                 val_loss += model.get_loss(bl_v).item()
#         #     last_val_loss = val_loss / len(val_loader)
#         #     t_val_s = time.perf_counter() - t_val
#         #     saver.update_val_loss(last_val_loss, model, args.output_dir,
#         #                            epoch, optimizer, avg_t)
#         #     print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
#         #           f"  rr={mean_rr:.2f}"
#         #           f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
#         #           f"  ens={current_ens}  alpha={step_alpha:.2f}"
#         #           f"  recurv_w={epoch_weights['recurv']:.2f}")
#         # else:
#         #     print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
#         #           f"  val={last_val_loss:.3f}(cached)"
#         #           f"  rr={mean_rr:.2f}  t={ep_s:.0f}s")

#         # ── Val loss (Tính mỗi epoch, không cached) ──────────────────────────
#         model.eval()
#         val_loss = 0.0
#         t_val    = time.perf_counter()
#         with torch.no_grad():
#             for batch in val_loader:
#                 bl_v = move(list(batch), device)
#                 with autocast(device_type='cuda', enabled=args.use_amp):
#                     val_loss += model.get_loss(bl_v).item()
        
#         last_val_loss = val_loss / len(val_loader)
#         t_val_s = time.perf_counter() - t_val
        
#         saver.update_val_loss(last_val_loss, model, args.output_dir,
#                                epoch, optimizer, avg_t)
        
#         print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
#               f"  rr={mean_rr:.2f}"
#               f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
#               f"  ens={current_ens}  alpha={step_alpha:.2f}"
#               f"  recurv_w={epoch_weights['recurv']:.2f}")

#         # ── Fast ADE (subset, monitor only) ───────────────────────────────────
#         t_ade  = time.perf_counter()
#         m_fast = evaluate_fast(model, val_subset_loader, device,
#                                ode_train, args.pred_len, effective_fast_ens)
#         t_ade_s = time.perf_counter() - t_ade

#         spread_72h    = m_fast.get("spread_72h_km", 0.0)
#         active_steps  = m_fast.get("active_steps", args.pred_len)
#         collapse_warn = "  ⚠️ COLLAPSE!" if spread_72h < 10.0 else ""
#         spread_warn   = "  ⚠️ SPREAD HIGH!" if spread_72h > 600.0 else ""

#         print(f"  [FAST-ADE ep{epoch} {t_ade_s:.0f}s]"
#               f"  ADE={m_fast['ADE']:.1f} km  FDE={m_fast['FDE']:.1f} km"
#               f"  12h={m_fast.get('12h', float('nan')):.0f}"
#               f"  24h={m_fast.get('24h', float('nan')):.0f}"
#               f"  72h={m_fast.get('72h', float('nan')):.0f} km"
#               f"  spread={spread_72h:.1f} km"
#               f"  active_steps={active_steps}/{args.pred_len}"
#               f"  (subset, monitor only)"
#               f"{collapse_warn}{spread_warn}")

#         # FIX-T23-2: Subset ADE chỉ để log, KHÔNG dùng cho best model
#         saver.log_subset_ade(m_fast["ADE"], epoch)

#         # ── Full val ADE (mỗi val_ade_freq epoch) → là criteria chính ────────
#         if epoch % args.val_ade_freq == 0:
#             try:
#                 r_full = evaluate_full_val_ade(
#                     model, val_loader, device,
#                     ode_steps     = ode_train,
#                     pred_len      = args.pred_len,
#                     fast_ensemble = effective_fast_ens,
#                     metrics_csv   = metrics_csv,
#                     epoch         = epoch,
#                     tag           = f"val_full_ep{epoch:03d}",
#                 )
#                 full_ade = r_full.get("ADE", float("inf"))

#                 # FIX-T23-2: CHỈ đây mới trigger best model save và patience
#                 saver.update_ade_full_val(
#                     full_ade, model, args.output_dir, epoch,
#                     optimizer, avg_t, last_val_loss,
#                     min_epochs=args.min_epochs)
#             except Exception as e:
#                 print(f"  ⚠  Full val ADE failed: {e}")
#                 import traceback; traceback.print_exc()

#         # ── Full eval (4-tier) ────────────────────────────────────────────────
#         if epoch % args.full_eval_freq == 0 and epoch > 0:
#             print(f"  [Full eval epoch {epoch}, ode_steps={ode_val}]")
#             try:
#                 dm, _, _, _ = evaluate_full(
#                     model, val_loader, device,
#                     ode_val, args.pred_len, args.val_ensemble,
#                     metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}")
#                 print(dm.summary())
#             except Exception as e:
#                 print(f"  ⚠  full_eval failed at epoch {epoch}: {e}")
#                 import traceback; traceback.print_exc()

#         if (epoch + 1) % args.save_interval == 0:
#             torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
#                        os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))

#         if saver.early_stop:
#             print(f"  Early stopping @ epoch {epoch}")
#             break

#         if epoch % 5 == 4:
#             avg_ep    = sum(epoch_times) / len(epoch_times)
#             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
#             elapsed_h = (time.perf_counter() - train_start) / 3600
#             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
#                   f"  (avg {avg_ep:.0f}s/epoch)")

#     # Restore final weights
#     _losses_mod.WEIGHTS["pinn"]     = args.pinn_w_end
#     _losses_mod.WEIGHTS["velocity"] = args.vel_w_end
#     _losses_mod.WEIGHTS["recurv"]   = args.recurv_w_end

#     total_train_h = (time.perf_counter() - train_start) / 3600

#     # ── Final test eval ───────────────────────────────────────────────────────
#     print(f"\n{'='*68}  FINAL TEST (ode_steps={ode_test})")
#     all_results = []

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
#             print(f"  Loaded best @ epoch {ck.get('epoch','?')}"
#                   f"  ADE={ck.get('val_ade_km','?')}")

#         final_ens = max(args.val_ensemble, 50)
#         dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
#             model, test_loader, device,
#             ode_test, args.pred_len, final_ens,
#             metrics_csv=metrics_csv, tag="test_final",
#             predict_csv=predict_csv)
#         print(dm_test.summary())

#         all_results.append(ModelResult(
#             model_name   = "FM+PINN-v23",
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
#             params_M     = n_params / 1e6,
#         ))

#         _, cliper_errs = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
#         persist_errs   = persistence_errors(obs_seqs, gt_seqs, args.pred_len)
#         fmpinn_per_seq = np.array([seq_ade_km(np.array(pp), np.array(g))
#                                     for pp, g in zip(pred_seqs, gt_seqs)])

#         np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
#         np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
#         np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))

#         lstm_per_seq      = _load_baseline_errors(args.lstm_errors_npy, "LSTM")
#         diffusion_per_seq = _load_baseline_errors(args.diffusion_errors_npy, "Diffusion")

#         stat_rows = [
#             paired_tests(fmpinn_per_seq, cliper_errs.mean(1),  "FM+PINN vs CLIPER", 5),
#             paired_tests(fmpinn_per_seq, persist_errs.mean(1), "FM+PINN vs Persist", 5),
#         ]
#         if lstm_per_seq is not None:
#             stat_rows.append(
#                 paired_tests(fmpinn_per_seq, lstm_per_seq, "FM+PINN vs LSTM", 5))
#         if diffusion_per_seq is not None:
#             stat_rows.append(
#                 paired_tests(fmpinn_per_seq, diffusion_per_seq, "FM+PINN vs Diffusion", 5))

#         compute_rows = DEFAULT_COMPUTE
#         try:
#             sb = next(iter(test_loader))
#             sb = move(list(sb), device)
#             from utils.evaluation_tables import profile_model_components
#             compute_rows = profile_model_components(model, sb, device)
#         except Exception as e:
#             print(f"  Profiling skipped: {e}")

#         export_all_tables(
#             results=all_results, ablation_rows=DEFAULT_ABLATION,
#             stat_rows=stat_rows, pinn_sens_rows=DEFAULT_PINN_SENSITIVITY,
#             compute_rows=compute_rows, out_dir=tables_dir)

#         with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
#             fh.write(dm_test.summary())
#             fh.write(f"\n\nmodel_version         : FM+PINN v23\n")
#             fh.write(f"sigma_min             : {args.sigma_min}\n")
#             fh.write(f"ctx_noise_scale       : {args.ctx_noise_scale}\n")
#             fh.write(f"initial_sample_sigma  : {args.initial_sample_sigma}\n")
#             fh.write(f"ode_steps_test        : {ode_test}\n")
#             fh.write(f"eval_ensemble         : {final_ens}\n")
#             fh.write(f"train_time_h          : {total_train_h:.2f}\n")
#             fh.write(f"n_params_M            : {n_params/1e6:.2f}\n")

#     avg_ep = sum(epoch_times)/len(epoch_times) if epoch_times else 0
#     print(f"\n  Best full-val ADE  : {saver.best_ade:.1f} km")
#     print(f"  Best val loss      : {saver.best_val_loss:.4f}")
#     print(f"  Avg epoch time     : {avg_ep:.0f}s")
#     print(f"  Total training     : {total_train_h:.2f}h")
#     print(f"  Tables dir         : {tables_dir}")
#     print("=" * 68)


# # ── GPH500 verification ───────────────────────────────────────────────────────

# def _check_gph500(bl, train_dataset):
#     """
#     FIX-T23-9: Verify GPH500 range (27-90 raw dam từ CSV).
#     Mean ~33 là đúng. Mean ≈ 0 hoặc ≈ -0.06 là sai (pre-normalized chưa fix).
#     """
#     env_data = bl[13]
#     if env_data is None or "gph500_mean" not in env_data:
#         print("  ⚠️  GPH500 key not found in env_data")
#         return

#     gph_val = env_data["gph500_mean"]
#     n_zero  = (gph_val == 0).sum().item()
#     n_total = gph_val.numel()
#     zero_pct = 100.0 * n_zero / max(n_total, 1)
#     gph_mean = gph_val.mean().item()

#     if abs(gph_mean) < 1.0 and zero_pct > 50.0:
#         print("\n" + "!" * 60)
#         print("  ⚠️  GPH500 mean ≈ 0 → Data not loading correctly from CSV")
#         print(f"     mean={gph_mean:.4f}, zero={zero_pct:.1f}%")
#         print("     Check FIX-DATA-18: env_gph500_mean should be raw dam (27-90)")
#         print("!" * 60 + "\n")
#     elif 25.0 < gph_mean < 95.0:
#         print(f"  ✅ GPH500 OK (mean={gph_mean:.2f} dam, zero={zero_pct:.1f}%)")
#     elif -30.0 < gph_mean < 5.0:
#         # Pre-normalized: mean ≈ -0.06 is the old npy format with _n keys
#         print(f"  ℹ️  GPH500 pre-normalized detected (mean={gph_mean:.4f})")
#         print(f"     This is acceptable if loading from .npy with _n keys")
#         print(f"     zero={zero_pct:.1f}%")
#     else:
#         print(f"  ⚠️  GPH500 unexpected range (mean={gph_mean:.4f}, zero={zero_pct:.1f}%)")


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(42); torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
#     main(args)

"""
scripts/train_flowmatching.py  ── v24
======================================
FIXES vs v23:

  FIX-T24-A  [TWO-PHASE TRAINING]
             Phase 1 (ep 0-30): Đóng băng FNO3D + enc_1d + env_enc.
               Chỉ train ShortRangeHead + ctx_fc/transformer. LR=5e-4.
               Mục tiêu: ShortRangeHead học motion pattern nhanh.
             Phase 2 (ep 30+): Unfreeze toàn bộ. LR=2e-4.
               Mục tiêu: Fine-tune end-to-end với đầy đủ loss.

  FIX-T24-B  [PRIMARY METRIC: 12h ADE]
             BestModelSaver giờ track best_12h_ade thay vì overall ADE.
             Early stopping dựa trên 12h ADE (tol=3 km).
             Overall ADE vẫn log nhưng không quyết định best model.

  FIX-T24-C  [SHORT-RANGE LOSS WEIGHT SCHEDULE]
             Ep 0-30 : short_range weight = 8.0 (focus mạnh vào 12h/24h)
             Ep 30-60: short_range weight = 5.0 (cân bằng)
             Ep 60+  : short_range weight = 3.0 (duy trì)

  FIX-T24-D  [SPREAD CONTROL ĐẦU TRAINING]
             initial_sample_sigma=0.03, ctx_noise_scale=0.002
             Tăng spread_weight lên 1.2 trong 30 epoch đầu để ép
             ensemble không bùng nổ ngay từ đầu.

  FIX-T24-E  [PINN WARMUP DÀI HƠN]
             pinn_warmup_epochs=80 (từ 50). PINN chỉ bắt đầu có ý nghĩa
             sau khi FM và ShortRangeHead đã học được pattern.

  FIX-T24-F  [EVALUATE SHORT-RANGE SEPARATELY]
             evaluate_fast() log riêng 12h và 24h ADE từ ShortRangeHead.
             evaluate_full_val_ade() cũng log sr_ade_12h, sr_ade_24h.

Kept from v23:
  FIX-T23-1  Curriculum removed
  FIX-T23-2  Full val ADE là criteria lưu best model (v24: 12h ADE)
  FIX-T23-6  patience=15
  FIX-T23-7  step_weight_alpha schedule
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import math
import random
import copy

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader
from Model.flow_matching_model import TCFlowMatching, ShortRangeHead
from Model.utils import get_cosine_schedule_with_warmup
from Model.losses import WEIGHTS as _BASE_WEIGHTS
from utils.metrics import (
    TCEvaluator, StepErrorAccumulator,
    save_metrics_csv, haversine_km_torch,
    denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
    cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
    brier_skill_score, cliper_errors, persistence_errors,
)
from utils.evaluation_tables import (
    ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
    export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
    DEFAULT_COMPUTE, paired_tests,
)
from scripts.statistical_tests import run_all_tests


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_km_np_local(pred_deg: np.ndarray,
                           gt_deg: np.ndarray) -> np.ndarray:
    pred_deg = np.atleast_2d(pred_deg)
    gt_deg   = np.atleast_2d(gt_deg)
    R = 6371.0
    lon1, lat1 = np.radians(pred_deg[:, 0]), np.radians(pred_deg[:, 1])
    lon2, lat2 = np.radians(gt_deg[:, 0]),   np.radians(gt_deg[:, 1])
    dlon = lon2 - lon1;  dlat = lat2 - lat1
    a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def seq_ade_km(pred_norm: np.ndarray, gt_norm: np.ndarray) -> float:
    return float(haversine_km_np_local(denorm_deg_np(pred_norm),
                                       denorm_deg_np(gt_norm)).mean())


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
    idx = rng.sample(range(n), min(subset_size, n))
    return DataLoader(Subset(val_dataset, idx),
                      batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=0, drop_last=False)


# ── Phase control  (FIX-T24-A) ───────────────────────────────────────────────

def get_phase(epoch: int, phase2_start: int = 30) -> int:
    """1 = short-range focus, 2 = full model."""
    return 1 if epoch < phase2_start else 2


def freeze_backbone(model: TCFlowMatching) -> None:
    """Phase 1: freeze FNO3D, enc_1d, env_enc."""
    for name, param in model.named_parameters():
        if any(k in name for k in ["spatial_enc", "enc_1d", "env_enc"]):
            param.requires_grad_(False)
    n_frozen = sum(not p.requires_grad for p in model.parameters())
    print(f"  [Phase-1] Frozen {n_frozen:,} params (backbone)")


def unfreeze_all(model: TCFlowMatching) -> None:
    """Phase 2: unfreeze everything."""
    for param in model.parameters():
        param.requires_grad_(True)
    print(f"  [Phase-2] Unfrozen all params")


# ── Weight schedules ──────────────────────────────────────────────────────────

# def get_short_range_weight(epoch: int) -> float:
#     """FIX-T24-C: short_range weight schedule."""
#     if epoch < 30:
#         return 8.0
#     elif epoch < 60:
#         return 5.0
#     else:
#         return 3.0
def get_short_range_weight(epoch):
    # FIX-T24-C giữ nguyên nhưng tăng mạnh hơn ở phase 1
    if epoch < 15:
        return 12.0   # ép ShortRangeHead học rất mạnh giai đoạn đầu
    elif epoch < 30:
        return 8.0
    elif epoch < 60:
        return 5.0
    else:
        return 3.0

def get_fm_weight(epoch):
    """FIX MỚI: FM loss nhỏ ở phase 1 để không cản ShortRangeHead."""
    if epoch < 30:
        return 0.5   # giảm FM weight trong phase 1
    return 2.0       # về bình thường ở phase 2

def get_spread_weight(epoch: int) -> float:
    """FIX-T24-D: higher spread penalty in first 30 epochs."""
    if epoch < 30:
        return 1.2
    return 0.8


# def get_pinn_weight(epoch: int, warmup_epochs: int = 80,
#                     w_start: float = 0.001, w_end: float = 0.05) -> float:
#     """FIX-T24-E: longer PINN warmup (80 epochs)."""
#     if epoch >= warmup_epochs:
#         return w_end
#     return w_start + (epoch / max(warmup_epochs - 1, 1)) * (w_end - w_start)

def get_pinn_weight(epoch, warmup_epochs=80,
                    w_start=0.001, w_end=0.05):
    # FIX: tăng nhanh hơn trong phase 2 (sau ep 30)
    if epoch < 30:
        # Phase 1: PINN rất nhỏ, focus ShortRangeHead
        return w_start
    elif epoch < warmup_epochs:
        # Phase 2: tăng tuyến tính từ 0.005 → w_end
        t = (epoch - 30) / max(warmup_epochs - 30, 1)
        return 0.005 + t * (w_end - 0.005)
    return w_end
    

def get_velocity_weight(epoch, warmup_epochs=20, w_start=0.5, w_end=1.5):
    if epoch >= warmup_epochs:
        return w_end
    return w_start + (epoch / max(warmup_epochs - 1, 1)) * (w_end - w_start)


def get_recurv_weight(epoch, warmup_epochs=10, w_start=0.3, w_end=1.0):
    if epoch >= warmup_epochs:
        return w_end
    return w_start + (epoch / max(warmup_epochs - 1, 1)) * (w_end - w_start)


def get_grad_clip(epoch, warmup_epochs=20, clip_start=2.0, clip_end=1.0):
    if epoch >= warmup_epochs:
        return clip_end
    return clip_start - (epoch / max(warmup_epochs - 1, 1)) * (clip_start - clip_end)


def get_step_weight_alpha(epoch: int, decay_epochs: int = 60) -> float:
    if epoch >= decay_epochs:
        return 0.0
    return 1.0 - (epoch / decay_epochs)


def get_progressive_ens(epoch: int, n_train_ens: int = 6) -> int:
    if epoch < 20:
        return 1
    elif epoch < 50:
        return 2
    else:
        return n_train_ens


# ── Short-range ADE helper ────────────────────────────────────────────────────

@torch.no_grad()
def compute_sr_ade(model: TCFlowMatching,
                   batch_list: list,
                   device) -> tuple[float, float]:
    """
    FIX-T24-F: Compute 12h and 24h ADE from ShortRangeHead.
    Returns (ade_12h_km, ade_24h_km).
    """
    obs_t  = batch_list[0]
    traj_gt = batch_list[1]

    raw_ctx  = model.net._context(batch_list)
    sr_pred  = model.net.forward_short_range(obs_t, raw_ctx)  # [4, B, 2]

    # Denorm to 0.1° units for haversine
    pred_01 = sr_pred.clone()
    pred_01[..., 0] = sr_pred[..., 0] * 50.0 + 1800.0
    pred_01[..., 1] = sr_pred[..., 1] * 50.0

    gt_01 = traj_gt.clone()
    gt_01[..., 0] = traj_gt[..., 0] * 50.0 + 1800.0
    gt_01[..., 1] = traj_gt[..., 1] * 50.0

    # step 1 = 6h (idx 0), step 2 = 12h (idx 1), step 4 = 24h (idx 3)
    ade_12h = haversine_km_torch(pred_01[1], gt_01[1]).mean().item()
    ade_24h = haversine_km_torch(pred_01[3], gt_01[3]).mean().item() \
              if pred_01.shape[0] >= 4 else float("nan")
    return ade_12h, ade_24h


# ── evaluate_fast ─────────────────────────────────────────────────────────────

def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
    """
    FIX-T24-F: Log short-range 12h/24h ADE from ShortRangeHead separately.
    """
    model.eval()
    acc = StepErrorAccumulator(pred_len)
    t0  = time.perf_counter()
    n   = 0
    spread_per_step = []
    sr_12h_buf, sr_24h_buf = [], []

    with torch.no_grad():
        for batch in loader:
            bl = list(batch)
            for i, x in enumerate(bl):
                if torch.is_tensor(x):
                    bl[i] = x.to(device)
                elif isinstance(x, dict):
                    bl[i] = {k: v.to(device) if torch.is_tensor(v) else v
                              for k, v in x.items()}

            pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
                                              ddim_steps=ode_steps)
            T_active  = pred.shape[0]
            gt_sliced = bl[1][:T_active]
            dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
            acc.update(dist)

            # Short-range ADE from head
            a12, a24 = compute_sr_ade(model, bl, device)
            sr_12h_buf.append(a12)
            sr_24h_buf.append(a24)

            # Per-step spread
            step_spreads = []
            for t in range(all_trajs.shape[1]):
                step_data = all_trajs[:, t, :, :]
                std_lon = step_data[:, :, 0].std(0)
                std_lat = step_data[:, :, 1].std(0)
                spread = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
                step_spreads.append(spread)
            spread_per_step.append(step_spreads)
            n += 1

    r = acc.compute()
    r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)

    if spread_per_step:
        spreads = np.array(spread_per_step)
        r["spread_12h_km"] = float(spreads[:, 1].mean()) if spreads.shape[1] > 1 else 0.0
        r["spread_24h_km"] = float(spreads[:, 3].mean()) if spreads.shape[1] > 3 else 0.0
        r["spread_72h_km"] = float(spreads[:, -1].mean())

    # FIX-T24-F: short-range ADE from head
    r["sr_ade_12h"] = float(np.mean(sr_12h_buf)) if sr_12h_buf else float("nan")
    r["sr_ade_24h"] = float(np.mean(sr_24h_buf)) if sr_24h_buf else float("nan")

    return r


# ── evaluate_full_val_ade ─────────────────────────────────────────────────────

def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
                           fast_ensemble, metrics_csv, epoch, tag=""):
    """
    FIX-T24-B: Primary metric is now 12h ADE (from ShortRangeHead).
    Returns dict with 'ADE', '12h', '24h', 'sr_12h', 'sr_24h'.
    """
    model.eval()
    acc       = StepErrorAccumulator(pred_len)
    t0        = time.perf_counter()
    n_batch   = 0
    pinn_buf  = []
    sr_12h_buf, sr_24h_buf = [], []

    with torch.no_grad():
        for batch in val_loader:
            bl = move(list(batch), device)

            pred, _, _ = model.sample(bl, num_ensemble=fast_ensemble,
                                       ddim_steps=ode_steps)
            T_pred = pred.shape[0]
            gt     = bl[1][:T_pred]
            dist   = haversine_km_torch(denorm_torch(pred), denorm_torch(gt))
            acc.update(dist)

            # Short-range from head
            a12, a24 = compute_sr_ade(model, bl, device)
            sr_12h_buf.append(a12)
            sr_24h_buf.append(a24)

            # PINN
            try:
                from Model.losses import pinn_bve_loss
                pred_deg = pred.clone()
                pred_deg[..., 0] = (pred[..., 0] * 50.0 + 1800.0) / 10.0
                pred_deg[..., 1] = (pred[..., 1] * 50.0) / 10.0
                env_d = bl[13] if len(bl) > 13 else None
                pinn_val = pinn_bve_loss(pred_deg, bl, env_data=env_d).item()
                pinn_buf.append(pinn_val)
            except Exception:
                pass

            n_batch += 1

    elapsed  = time.perf_counter() - t0
    r        = acc.compute()
    sr_12h   = float(np.mean(sr_12h_buf)) if sr_12h_buf else float("nan")
    sr_24h   = float(np.mean(sr_24h_buf)) if sr_24h_buf else float("nan")
    pinn_mean = f"{np.mean(pinn_buf):.3f}" if pinn_buf else "N/A"

    print(f"\n{'='*64}")
    print(f"  [FULL VAL ADE  ep={epoch}  {elapsed:.0f}s  {n_batch} batches]")
    print(f"  ADE (blend) = {r.get('ADE', float('nan')):.1f} km  "
          f"FDE = {r.get('FDE', float('nan')):.1f} km")
    print(f"  12h={r.get('12h', float('nan')):.0f}  "
          f"24h={r.get('24h', float('nan')):.0f}  "
          f"48h={r.get('48h', float('nan')):.0f}  "
          f"72h={r.get('72h', float('nan')):.0f} km")
    print(f"  ── ShortRangeHead ──")
    print(f"  SR-12h={sr_12h:.1f} km  SR-24h={sr_24h:.1f} km  "
          f"[TARGET: <50 / <100 km]")
    print(f"  PINN_mean={pinn_mean}  samples={r.get('n_samples',0)}  "
          f"ens={fast_ensemble}  steps={ode_steps}")
    print(f"{'='*64}\n")

    r["sr_12h"] = sr_12h
    r["sr_24h"] = sr_24h

    from datetime import datetime
    from utils.metrics import DatasetMetrics, save_metrics_csv as _save_csv
    dm = DatasetMetrics(
        ade      = r.get("ADE",  float("nan")),
        fde      = r.get("FDE",  float("nan")),
        ugde_12h = r.get("12h",  float("nan")),
        ugde_24h = r.get("24h",  float("nan")),
        ugde_48h = r.get("48h",  float("nan")),
        ugde_72h = r.get("72h",  float("nan")),
        n_total  = r.get("n_samples", 0),
        timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    _save_csv(dm, metrics_csv, tag=tag or f"val_full_ep{epoch:03d}")
    return r


def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
                  metrics_csv, tag="", predict_csv=""):
    """Full 4-tier evaluation (unchanged from v23)."""
    model.eval()
    cliper_step_errors = []
    ev = TCEvaluator(pred_len=pred_len, compute_dtw=True)
    obs_seqs_01 = []; gt_seqs_01 = []; pred_seqs_01 = []; ens_seqs_01 = []

    with torch.no_grad():
        for batch in loader:
            bl  = move(list(batch), device)
            gt  = bl[1];  obs = bl[0]
            pred_mean, _, all_trajs = model.sample(
                bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
                predict_csv=predict_csv if predict_csv else None)

            pd_np = denorm_torch(pred_mean).cpu().numpy()
            gd_np = denorm_torch(gt).cpu().numpy()
            od_np = denorm_torch(obs).cpu().numpy()
            ed_np = denorm_torch(all_trajs).cpu().numpy()

            B = pd_np.shape[1]
            for b in range(B):
                ens_b = ed_np[:, :, b, :]
                ev.update(pd_np[:, b, :], gd_np[:, b, :], pred_ens=ens_b)
                obs_seqs_01.append(od_np[:, b, :])
                gt_seqs_01.append(gd_np[:, b, :])
                pred_seqs_01.append(pd_np[:, b, :])
                ens_seqs_01.append(ens_b)

                obs_b_norm = obs.cpu().numpy()[:, b, :]
                cliper_errors_b = np.zeros(pred_len)
                for h in range(pred_len):
                    pred_cliper_norm = cliper_forecast(obs_b_norm, h + 1)
                    pred_cliper_01   = denorm_np(pred_cliper_norm[np.newaxis])
                    gt_01            = gd_np[h, b, :][np.newaxis]
                    from utils.metrics import haversine_km_np
                    cliper_errors_b[h] = float(
                        haversine_km_np(pred_cliper_01, gt_01, unit_01deg=True)[0])
                cliper_step_errors.append(cliper_errors_b)

    if cliper_step_errors:
        cliper_mat       = np.stack(cliper_step_errors)
        cliper_ugde_dict = {h: float(cliper_mat[:, s].mean())
                            for h, s in HORIZON_STEPS.items()
                            if s < cliper_mat.shape[1]}
        ev.cliper_ugde   = cliper_ugde_dict

    dm = ev.compute(tag=tag)

    try:
        if LANDFALL_TARGETS and ens_seqs_01:
            bss_vals = []
            step_72  = HORIZON_STEPS.get(72, pred_len - 1)
            for tname, t_lon, t_lat in LANDFALL_TARGETS:
                bv = brier_skill_score(
                    ens_seqs_01, gt_seqs_01, min(step_72, pred_len-1),
                    (t_lon * 10.0, t_lat * 10.0), LANDFALL_RADIUS_KM)
                if not math.isnan(bv):
                    bss_vals.append(bv)
            if bss_vals:
                dm.bss_mean = float(np.mean(bss_vals))
    except Exception as e:
        print(f"  ⚠  BSS failed: {e}")

    save_metrics_csv(dm, metrics_csv, tag=tag)
    return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# ── BestModelSaver  (FIX-T24-B) ───────────────────────────────────────────────

class BestModelSaver:
    """
    FIX-T24-B: Primary metric = ShortRangeHead 12h ADE.
    Also track overall blend ADE separately.
    patience triggers only from 12h ADE stagnation.
    """

    def __init__(self, patience: int = 15, sr_tol: float = 3.0,
                 ade_tol: float = 5.0):
        self.patience      = patience
        self.sr_tol        = sr_tol       # km tolerance for 12h ADE
        self.ade_tol       = ade_tol      # km tolerance for overall ADE
        self.best_sr12h    = float("inf")
        self.best_ade      = float("inf")
        self.best_val_loss = float("inf")
        self.counter_12h   = 0
        self.counter_loss  = 0
        self.early_stop    = False

    def reset_counters(self, reason: str = "") -> None:
        self.counter_12h  = 0
        self.counter_loss = 0
        if reason:
            print(f"  [SAVER] Patience reset: {reason}")

    def update_val_loss(self, val_loss, model, out_dir, epoch, optimizer, tl):
        if val_loss < self.best_val_loss - 1e-4:
            self.best_val_loss = val_loss
            self.counter_loss  = 0
            torch.save(dict(
                epoch=epoch, model_state_dict=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                train_loss=tl, val_loss=val_loss,
                model_version="v24-valloss"),
                os.path.join(out_dir, "best_model_valloss.pth"))
        else:
            self.counter_loss += 1

    def update_full_val(self, result: dict, model, out_dir, epoch,
                        optimizer, tl, vl, min_epochs: int = 80):
        """
        FIX-T24-B: Primary trigger = sr_12h (ShortRangeHead 12h ADE).
        """
        sr_12h    = result.get("sr_12h", float("inf"))
        blend_ade = result.get("ADE",    float("inf"))

        # Save best model if SR 12h ADE improved
        improved = sr_12h < self.best_sr12h - self.sr_tol
        if improved:
            self.best_sr12h = sr_12h
            self.best_ade   = min(self.best_ade, blend_ade)
            self.counter_12h = 0
            torch.save(dict(
                epoch=epoch, model_state_dict=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                train_loss=tl, val_loss=vl,
                sr_ade_12h=sr_12h, blend_ade=blend_ade,
                model_version="v24-SR+FM"),
                os.path.join(out_dir, "best_model.pth"))
            print(f"  ✅ Best SR-12h {sr_12h:.1f} km  "
                  f"blend-ADE {blend_ade:.1f} km  (epoch {epoch})")
        else:
            self.counter_12h += 1
            print(f"  No SR-12h improvement {self.counter_12h}/{self.patience}"
                  f"  (Δ={self.best_sr12h - sr_12h:.1f} km < tol={self.sr_tol} km)"
                  f"  | Loss counter {self.counter_loss}/{self.patience}"
                  f"  | SR-12h={sr_12h:.1f}  blend={blend_ade:.1f}")

        if epoch >= min_epochs:
            if (self.counter_12h >= self.patience
                    and self.counter_loss >= self.patience):
                self.early_stop = True
                print(f"  ⛔ Early stop @ epoch {epoch}")
        else:
            if (self.counter_12h >= self.patience
                    and self.counter_loss >= self.patience):
                print(f"  ⚠  Would stop but min_epochs={min_epochs} not reached.")
                self.counter_12h  = 0
                self.counter_loss = 0

    def log_subset_sr(self, sr_12h: float, sr_24h: float, epoch: int):
        print(f"  [SUBSET SR ep{epoch}]  12h={sr_12h:.1f}  24h={sr_24h:.1f} km"
              f"  [target <50/<100]  (monitor only)")


# ── GPH/UV checks (unchanged) ─────────────────────────────────────────────────

# def _check_gph500(bl, train_dataset):
#     env_data = bl[13]
#     if env_data is None or "gph500_mean" not in env_data:
#         print("  ⚠️  GPH500 key not found"); return
#     gph_val  = env_data["gph500_mean"]
#     gph_mean = gph_val.mean().item()
#     zero_pct = 100.0 * (gph_val == 0).sum().item() / max(gph_val.numel(), 1)
#     if 25.0 < gph_mean < 95.0:
#         print(f"  ✅ GPH500 OK (mean={gph_mean:.2f} dam, zero={zero_pct:.1f}%)")
#     else:
#         print(f"  ⚠️  GPH500 unexpected (mean={gph_mean:.4f}, zero={zero_pct:.1f}%)")


def _check_gph500(bl, train_dataset):
    try:
        env_dir = train_dataset.env_path
    except AttributeError:
        try:
            env_dir = train_dataset.dataset.env_path
        except AttributeError:
            env_dir = "UNKNOWN"
    print(f"  Env path   : {env_dir}  exists={os.path.exists(env_dir) if env_dir != 'UNKNOWN' else 'N/A'}")

    env_data = bl[13]
    if env_data is None:
        print("  ⚠️  env_data is None"); return

    # has_data3d bị skip trong collate → không check nữa
    for key, expected_range in [
        ("gph500_mean",   (-3.0, 3.0)),   # đã normalized
        ("gph500_center", (-3.0, 3.0)),
    ]:
        if key not in env_data:
            print(f"  ⚠️  {key} MISSING"); continue
        v    = env_data[key]
        mn   = v.mean().item()
        std  = v.std().item()
        zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
        lo, hi = expected_range
        if zero > 80.0:
            print(f"  ⚠️  {key}: zero={zero:.1f}% → env missing!")
        elif lo <= mn <= hi:
            print(f"  ✅ {key}: mean={mn:.3f} std={std:.3f} zero={zero:.1f}%")
        else:
            print(f"  ⚠️  {key}: mean={mn:.3f} ngoài range [{lo},{hi}]")

def _check_uv500(bl):
    env_data = bl[13]
    if env_data is None: return
    for key in ("u500_mean", "v500_mean"):
        if key not in env_data:
            print(f"  ⚠️  {key} MISSING"); continue
        v    = env_data[key]
        mn   = v.mean().item()
        std  = v.std().item()
        zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
        if zero > 80.0:
            print(f"  ⚠️  {key}: zero={zero:.1f}% → u/v500 missing!")
        else:
            # expected: normalized [-1,1], mean ~0, std ~0.1-0.3
            print(f"  ✅ {key}: mean={mn:.4f} std={std:.4f} zero={zero:.1f}%")

def _load_baseline_errors(path, name):
    if path is None:
        print(f"\n  ⚠  {name} errors not provided.\n"); return None
    if not os.path.exists(path):
        print(f"\n  ⚠  {path} not found.\n"); return None
    arr = np.load(path)
    print(f"  ✓  Loaded {name}: {arr.shape}")
    return arr


# ── Args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
    p.add_argument("--obs_len",         default=8,          type=int)
    p.add_argument("--pred_len",        default=12,         type=int)
    p.add_argument("--test_year",       default=None,       type=int)
    p.add_argument("--batch_size",      default=32,         type=int)
    p.add_argument("--num_epochs",      default=200,        type=int)
    p.add_argument("--g_learning_rate", default=2e-4,       type=float)
    p.add_argument("--phase1_lr",       default=5e-4,       type=float,
                   help="FIX-T24-A: LR cho phase 1 (ShortRangeHead focus)")
    p.add_argument("--phase2_start",    default=30,         type=int,
                   help="FIX-T24-A: epoch bắt đầu phase 2 (unfreeze all)")
    p.add_argument("--weight_decay",    default=1e-4,       type=float)
    p.add_argument("--warmup_epochs",   default=3,          type=int)
    p.add_argument("--grad_clip",       default=2.0,        type=float)
    p.add_argument("--grad_accum",      default=2,          type=int)
    p.add_argument("--patience",        default=15,         type=int)
    p.add_argument("--min_epochs",      default=80,         type=int)
    p.add_argument("--n_train_ens",     default=6,          type=int)
    p.add_argument("--use_amp",         action="store_true")
    p.add_argument("--num_workers",     default=2,          type=int)

    # FIX-M25: smaller sigma / noise
    p.add_argument("--sigma_min",            default=0.02,  type=float)
    p.add_argument("--ctx_noise_scale",      default=0.002, type=float,
                   help="FIX-M25: 0.02→0.002")
    p.add_argument("--initial_sample_sigma", default=0.03,  type=float,
                   help="FIX-M25: 0.1→0.03")

    p.add_argument("--ode_steps_train", default=20,  type=int)
    p.add_argument("--ode_steps_val",   default=30,  type=int)
    p.add_argument("--ode_steps_test",  default=50,  type=int)
    p.add_argument("--ode_steps",       default=None, type=int)

    p.add_argument("--val_ensemble",    default=30,  type=int)
    p.add_argument("--fast_ensemble",   default=8,   type=int)

    p.add_argument("--fno_modes_h",      default=4,  type=int)
    p.add_argument("--fno_modes_t",      default=4,  type=int)
    p.add_argument("--fno_layers",       default=4,  type=int)
    p.add_argument("--fno_d_model",      default=32, type=int)
    p.add_argument("--fno_spatial_down", default=32, type=int)
    p.add_argument("--mamba_d_state",    default=16, type=int)

    p.add_argument("--val_loss_freq",    default=1,   type=int)
    p.add_argument("--val_freq",         default=1,   type=int)
    p.add_argument("--val_ade_freq",     default=1,   type=int)
    p.add_argument("--full_eval_freq",   default=10,  type=int)
    p.add_argument("--val_subset_size",  default=600, type=int)

    p.add_argument("--output_dir",      default="runs/v24",      type=str)
    p.add_argument("--save_interval",   default=10,              type=int)
    p.add_argument("--metrics_csv",     default="metrics.csv",   type=str)
    p.add_argument("--predict_csv",     default="predictions.csv", type=str)
    p.add_argument("--lstm_errors_npy",      default=None, type=str)
    p.add_argument("--diffusion_errors_npy", default=None, type=str)

    p.add_argument("--gpu_num",         default="0", type=str)
    p.add_argument("--delim",           default=" ")
    p.add_argument("--skip",            default=1,   type=int)
    p.add_argument("--min_ped",         default=1,   type=int)
    p.add_argument("--threshold",       default=0.002, type=float)
    p.add_argument("--other_modal",     default="gph")

    p.add_argument("--step_weight_decay_epochs", default=60, type=int)
    p.add_argument("--lon_flip_prob",            default=0.3, type=float)

    # FIX-T24-E: longer PINN warmup
    p.add_argument("--pinn_warmup_epochs", default=80,    type=int,
                   help="FIX-T24-E: 50→80")
    p.add_argument("--pinn_w_start",      default=0.001, type=float)
    p.add_argument("--pinn_w_end",        default=0.05,  type=float)

    p.add_argument("--vel_warmup_epochs",    default=20,  type=float)
    p.add_argument("--vel_w_start",          default=0.5, type=float)
    p.add_argument("--vel_w_end",            default=1.5, type=float)
    p.add_argument("--recurv_warmup_epochs", default=10,  type=int)
    p.add_argument("--recurv_w_start",       default=0.3, type=float)
    p.add_argument("--recurv_w_end",         default=1.0, type=float)

    return p.parse_args()


def _resolve_ode_steps(args):
    if args.ode_steps is not None:
        return args.ode_steps, args.ode_steps, args.ode_steps
    return args.ode_steps_train, args.ode_steps_val, args.ode_steps_test


# ── MAIN ──────────────────────────────────────────────────────────────────────

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

    ode_train, ode_val, ode_test = _resolve_ode_steps(args)

    print("=" * 70)
    print("  TC-FlowMatching v24  |  ShortRangeHead + OT-CFM + PINN")
    print("  v24 FIXES:")
    print("    FIX-T24-A: Two-phase training (backbone freeze ep 0-30)")
    print("    FIX-T24-B: Primary metric = SR-12h ADE (ShortRangeHead)")
    print("    FIX-T24-C: short_range_weight 8→5→3 schedule")
    print("    FIX-T24-D: initial_sample_sigma=0.03, spread_weight↑")
    print("    FIX-T24-E: pinn_warmup=80 ep")
    print("    FIX-T24-F: evaluate SR-12h/24h separately")
    print("    FIX-M23:   ShortRangeHead GRU for deterministic 12h/24h")
    print("    FIX-L49:   short_range_regression_loss (Huber)")
    print("    FIX-L50:   PINN scale 1e-3→1e-2")
    print("    FIX-L52:   ensemble_spread_loss max_spread 200→150 km")
    print("=" * 70)
    print(f"  device               : {device}")
    print(f"  phase2_start         : ep {args.phase2_start}")
    print(f"  phase1_lr            : {args.phase1_lr:.2e}")
    print(f"  phase2_lr            : {args.g_learning_rate:.2e}")
    print(f"  ctx_noise_scale      : {args.ctx_noise_scale}")
    print(f"  initial_sample_sigma : {args.initial_sample_sigma}")
    print(f"  ode_steps            : train={ode_train} val={ode_val} test={ode_test}")
    print(f"  patience             : {args.patience}  min_epochs={args.min_epochs}")
    print()

    train_dataset, train_loader = data_loader(
        args, {"root": args.dataset_root, "type": "train"}, test=False)
    val_dataset, val_loader = data_loader(
        args, {"root": args.dataset_root, "type": "val"}, test=True)

    from Model.data.trajectoriesWithMe_unet_training import seq_collate
    val_subset_loader = make_val_subset_loader(
        val_dataset, args.val_subset_size, args.batch_size,
        seq_collate, args.num_workers)

    test_loader = None
    try:
        _, test_loader = data_loader(
            args, {"root": args.dataset_root, "type": "test"},
            test=True, test_year=None)
    except Exception as e:
        print(f"  Warning: test loader: {e}")

    print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
    print(f"  val   : {len(val_dataset)} seq")
    if test_loader:
        print(f"  test  : {len(test_loader.dataset)} seq")

    model = TCFlowMatching(
        pred_len             = args.pred_len,
        obs_len              = args.obs_len,
        sigma_min            = args.sigma_min,
        n_train_ens          = args.n_train_ens,
        ctx_noise_scale      = args.ctx_noise_scale,
        initial_sample_sigma = args.initial_sample_sigma,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params  : {n_params:,}")

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: enabled")
    except Exception:
        pass

    # ── Phase 1: freeze backbone, higher LR for ShortRangeHead ──────────────
    freeze_backbone(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase1_lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
    total_steps     = steps_per_epoch * args.num_epochs
    warmup          = steps_per_epoch * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    saver  = BestModelSaver(patience=args.patience, sr_tol=3.0, ade_tol=1.0)
    scaler = GradScaler("cuda", enabled=args.use_amp)

    print("=" * 70)
    print(f"  TRAINING  ({steps_per_epoch} steps/epoch)")
    print("=" * 70)

    epoch_times   = []
    train_start   = time.perf_counter()
    last_val_loss = float("inf")
    _phase        = 1
    _prev_ens     = 1
    _lr_ep30_done = False
    _lr_ep60_done = False

    import Model.losses as _losses_mod

    for epoch in range(args.num_epochs):

        # ── Phase transition ──────────────────────────────────────────────
        if epoch == args.phase2_start and _phase == 1:
            _phase = 2
            unfreeze_all(model)
            # Reset optimizer with phase-2 LR
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.g_learning_rate,
                weight_decay=args.weight_decay,
            )
            rem_steps = steps_per_epoch * (args.num_epochs - epoch)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, steps_per_epoch, rem_steps, min_lr=5e-6)
            saver.reset_counters(f"Phase 2 started @ ep {epoch}")
            print(f"\n  ↺  PHASE 2 START ep {epoch}: unfreeze all, "
                  f"LR={args.g_learning_rate:.2e}")

        # ── Weight schedules ──────────────────────────────────────────────
        current_ens = get_progressive_ens(epoch, args.n_train_ens)
        model.n_train_ens = current_ens
        eff_fast_ens = min(args.fast_ensemble,
                           max(current_ens * 2, args.fast_ensemble))

        if current_ens != _prev_ens:
            saver.reset_counters(f"n_train_ens {_prev_ens}→{current_ens}")
            _prev_ens = current_ens

        step_alpha    = get_step_weight_alpha(epoch, args.step_weight_decay_epochs)
        epoch_weights = copy.copy(_BASE_WEIGHTS)
        epoch_weights["fm"]          = get_fm_weight(epoch)
        epoch_weights["short_range"] = get_short_range_weight(epoch)
        epoch_weights["spread"]      = get_spread_weight(epoch)
        epoch_weights["pinn"]        = get_pinn_weight(
            epoch, args.pinn_warmup_epochs, args.pinn_w_start, args.pinn_w_end)
        epoch_weights["velocity"]    = get_velocity_weight(
            epoch, args.vel_warmup_epochs, args.vel_w_start, args.vel_w_end)
        epoch_weights["recurv"]      = get_recurv_weight(
            epoch, args.recurv_warmup_epochs, args.recurv_w_start, args.recurv_w_end)
        _losses_mod.WEIGHTS.update(epoch_weights)

        current_clip = get_grad_clip(epoch, warmup_epochs=20,
                                      clip_start=args.grad_clip, clip_end=1.0)

        # LR warm restarts (phase 2 only)
        if _phase == 2:
            if epoch == 30 and not _lr_ep30_done:
                _lr_ep30_done = True
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer, steps_per_epoch,
                    steps_per_epoch * (args.num_epochs - 30), min_lr=5e-6)
                saver.reset_counters("LR warm restart ep 30")
                print(f"  ↺  Warm Restart LR @ ep 30")

            if epoch == 60 and not _lr_ep60_done:
                _lr_ep60_done = True
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer, steps_per_epoch,
                    steps_per_epoch * (args.num_epochs - 60), min_lr=1e-6)
                saver.reset_counters("LR warm restart ep 60")
                print(f"  ↺  Warm Restart LR @ ep 60")

        # ── Training loop ─────────────────────────────────────────────────
        model.train()
        sum_loss   = 0.0
        t0         = time.perf_counter()
        optimizer.zero_grad()
        recurv_ratio_buf = []

        for i, batch in enumerate(train_loader):
            bl = move(list(batch), device)

            if epoch == 0 and i == 0:
                # Đúng attribute name
                try:
                    env_dir = train_dataset.env_path
                except AttributeError:
                    try:
                        env_dir = train_dataset.dataset.env_path
                    except AttributeError:
                        env_dir = "UNKNOWN"
                print(f"  Env path     : {env_dir}")
                print(f"  Env exists   : {os.path.exists(env_dir) if env_dir != 'UNKNOWN' else 'N/A'}")
                print(f"  Root path    : {getattr(train_dataset, 'root_path', 'N/A')}")
                        # Check tensor trong batch (sau collate)
                _check_gph500(bl, train_dataset)
                _check_uv500(bl)

            with autocast(device_type="cuda", enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl, step_weight_alpha=step_alpha)

            loss_to_bp = bd["total"] / max(args.grad_accum, 1)
            scaler.scale(loss_to_bp).backward()

            if ((i + 1) % args.grad_accum == 0
                    or (i + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), current_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            sum_loss += bd["total"].item()
            if "recurv_ratio" in bd:
                recurv_ratio_buf.append(bd["recurv_ratio"])

            if i % 20 == 0:
                lr      = optimizer.param_groups[0]["lr"]
                sr_loss = bd.get("short_range", 0.0)
                elapsed = time.perf_counter() - t0
                print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
                      f"  loss={bd['total'].item():.3f}"
                      f"  fm={bd.get('fm',0):.2f}"
                      f"  sr={sr_loss:.4f}"
                      f"  vel={bd.get('velocity',0):.3f}"
                      f"  pinn={bd.get('pinn',0):.4f}"
                      f"  spread={bd.get('spread',0):.3f}"
                      f"  sr_w={epoch_weights['short_range']:.1f}"
                      f"  alpha={step_alpha:.2f}"
                      f"  ph={_phase}"
                      f"  ens={current_ens}"
                      f"  lr={lr:.2e}  t={elapsed:.0f}s")

        ep_s    = time.perf_counter() - t0
        epoch_times.append(ep_s)
        avg_t   = sum_loss / len(train_loader)
        mean_rr = float(np.mean(recurv_ratio_buf)) if recurv_ratio_buf else 0.0

        # ── Val loss ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        t_val    = time.perf_counter()
        with torch.no_grad():
            for batch in val_loader:
                bl_v = move(list(batch), device)
                with autocast(device_type="cuda", enabled=args.use_amp):
                    val_loss += model.get_loss(bl_v).item()
        last_val_loss = val_loss / len(val_loader)
        t_val_s = time.perf_counter() - t_val
        saver.update_val_loss(last_val_loss, model, args.output_dir,
                               epoch, optimizer, avg_t)

        print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
              f"  rr={mean_rr:.2f}"
              f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
              f"  ens={current_ens}  alpha={step_alpha:.2f}")

        # ── Fast eval ─────────────────────────────────────────────────────
        t_ade   = time.perf_counter()
        m_fast  = evaluate_fast(model, val_subset_loader, device,
                                ode_train, args.pred_len, eff_fast_ens)
        t_ade_s = time.perf_counter() - t_ade

        sr_12h   = m_fast.get("sr_ade_12h", float("nan"))
        sr_24h   = m_fast.get("sr_ade_24h", float("nan"))
        spread72 = m_fast.get("spread_72h_km", 0.0)
        collapse = "  ⚠️ COLLAPSE!" if spread72 < 10.0 else ""
        hi_sprd  = "  ⚠️ SPREAD!" if spread72 > 400.0 else ""
        sr_hit12 = "  🎯 <50km!" if sr_12h < 50.0 else ""
        sr_hit24 = "  🎯 <100km!" if sr_24h < 100.0 else ""

        print(f"  [FAST ep{epoch} {t_ade_s:.0f}s]"
              f"  ADE={m_fast['ADE']:.1f}  FDE={m_fast['FDE']:.1f} km"
              f"  12h={m_fast.get('12h',float('nan')):.0f}"
              f"  24h={m_fast.get('24h',float('nan')):.0f}"
              f"  72h={m_fast.get('72h',float('nan')):.0f} km"
              f"  spread={spread72:.1f} km"
              f"{collapse}{hi_sprd}")
        print(f"         ShortRange: SR-12h={sr_12h:.1f} km"
              f"  SR-24h={sr_24h:.1f} km"
              f"{sr_hit12}{sr_hit24}  (monitor only)")

        saver.log_subset_sr(sr_12h, sr_24h, epoch)

        # ── Full val ADE ──────────────────────────────────────────────────
        if epoch % args.val_ade_freq == 0:
            try:
                r_full = evaluate_full_val_ade(
                    model, val_loader, device,
                    ode_steps     = ode_train,
                    pred_len      = args.pred_len,
                    fast_ensemble = eff_fast_ens,
                    metrics_csv   = metrics_csv,
                    epoch         = epoch,
                    tag           = f"val_full_ep{epoch:03d}",
                )
                # FIX-T24-B: primary = sr_12h
                saver.update_full_val(
                    r_full, model, args.output_dir, epoch,
                    optimizer, avg_t, last_val_loss,
                    min_epochs=args.min_epochs)
            except Exception as e:
                print(f"  ⚠  Full val failed: {e}")
                import traceback; traceback.print_exc()

        # ── Full eval ─────────────────────────────────────────────────────
        if epoch % args.full_eval_freq == 0 and epoch > 0:
            try:
                dm, _, _, _ = evaluate_full(
                    model, val_loader, device,
                    ode_val, args.pred_len, args.val_ensemble,
                    metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}")
                print(dm.summary())
            except Exception as e:
                print(f"  ⚠  full_eval failed ep {epoch}: {e}")

        if (epoch + 1) % args.save_interval == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))

        if saver.early_stop:
            print(f"  Early stopping @ epoch {epoch}")
            break

        if epoch % 5 == 4:
            avg_ep    = sum(epoch_times) / len(epoch_times)
            remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
            elapsed_h = (time.perf_counter() - train_start) / 3600
            print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
                  f"  (avg {avg_ep:.0f}s/epoch)")

    # ── Final ─────────────────────────────────────────────────────────────────
    _losses_mod.WEIGHTS["pinn"]        = args.pinn_w_end
    _losses_mod.WEIGHTS["velocity"]    = args.vel_w_end
    _losses_mod.WEIGHTS["recurv"]      = args.recurv_w_end
    _losses_mod.WEIGHTS["short_range"] = 3.0

    total_train_h = (time.perf_counter() - train_start) / 3600

    print(f"\n{'='*70}  FINAL TEST (ode_steps={ode_test})")
    all_results = []

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
            print(f"  Loaded best @ epoch {ck.get('epoch','?')}"
                  f"  SR-12h={ck.get('sr_ade_12h','?')}")

        final_ens = max(args.val_ensemble, 50)
        dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
            model, test_loader, device,
            ode_test, args.pred_len, final_ens,
            metrics_csv=metrics_csv, tag="test_final",
            predict_csv=predict_csv)
        print(dm_test.summary())

        all_results.append(ModelResult(
            model_name   = "FM+SR+PINN-v24",
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
            params_M     = n_params / 1e6,
        ))

        _, cliper_errs = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
        persist_errs   = persistence_errors(obs_seqs, gt_seqs, args.pred_len)
        fmpinn_per_seq = np.array([seq_ade_km(np.array(pp), np.array(g))
                                    for pp, g in zip(pred_seqs, gt_seqs)])

        np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
        np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
        np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))

        stat_rows = [
            paired_tests(fmpinn_per_seq, cliper_errs.mean(1),  "FM+SR vs CLIPER", 5),
            paired_tests(fmpinn_per_seq, persist_errs.mean(1), "FM+SR vs Persist", 5),
        ]
        lstm_per_seq      = _load_baseline_errors(args.lstm_errors_npy, "LSTM")
        diffusion_per_seq = _load_baseline_errors(args.diffusion_errors_npy, "Diffusion")
        if lstm_per_seq is not None:
            stat_rows.append(paired_tests(fmpinn_per_seq, lstm_per_seq, "FM+SR vs LSTM", 5))
        if diffusion_per_seq is not None:
            stat_rows.append(paired_tests(fmpinn_per_seq, diffusion_per_seq, "FM+SR vs Diffusion", 5))

        export_all_tables(
            results=all_results, ablation_rows=DEFAULT_ABLATION,
            stat_rows=stat_rows, pinn_sens_rows=DEFAULT_PINN_SENSITIVITY,
            compute_rows=DEFAULT_COMPUTE, out_dir=tables_dir)

        with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
            fh.write(dm_test.summary())
            fh.write(f"\n\nmodel_version         : FM+SR+PINN v24\n")
            fh.write(f"ctx_noise_scale       : {args.ctx_noise_scale}\n")
            fh.write(f"initial_sample_sigma  : {args.initial_sample_sigma}\n")
            fh.write(f"ode_steps_test        : {ode_test}\n")
            fh.write(f"eval_ensemble         : {final_ens}\n")
            fh.write(f"train_time_h          : {total_train_h:.2f}\n")
            fh.write(f"n_params_M            : {n_params/1e6:.2f}\n")

    avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    print(f"\n  Best SR-12h ADE    : {saver.best_sr12h:.1f} km")
    print(f"  Best blend ADE     : {saver.best_ade:.1f} km")
    print(f"  Best val loss      : {saver.best_val_loss:.4f}")
    print(f"  Avg epoch time     : {avg_ep:.0f}s")
    print(f"  Total training     : {total_train_h:.2f}h")
    print("=" * 70)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main(args)
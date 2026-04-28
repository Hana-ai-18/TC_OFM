# # # # # # # # # # # # # # # """
# # # # # # # # # # # # # # # scripts/train_flowmatching.py  ── v23
# # # # # # # # # # # # # # # ======================================
# # # # # # # # # # # # # # # FIXES vs v22:

# # # # # # # # # # # # # # #   FIX-T23-1  CURRICULUM REMOVED (FIX-DATA-22).
# # # # # # # # # # # # # # #              Curriculum gây ADE tụt 282→444 km mỗi lần tăng len.
# # # # # # # # # # # # # # #              Thay bằng step_weight_alpha: giảm dần từ 1.0 → 0.0 theo epoch,
# # # # # # # # # # # # # # #              làm cho AFCRPS weight các bước gần hơn ở epoch đầu (soft curriculum),
# # # # # # # # # # # # # # #              KHÔNG cắt ngắn sequence pred_len=12.

# # # # # # # # # # # # # # #   FIX-T23-2  BestModelSaver: CHỈ dùng full_val ADE làm criteria lưu best model.
# # # # # # # # # # # # # # #              Subset ADE chỉ dùng để monitor nhanh, KHÔNG ảnh hưởng patience.
# # # # # # # # # # # # # # #              Khi full-val chưa chạy, saver giữ nguyên counter.

# # # # # # # # # # # # # # #   FIX-T23-3  ODE steps tăng: train=20, val=30, test=50 (từ 10/10/10).
# # # # # # # # # # # # # # #              10 steps quá thấp cho OT-CFM với 12-step trajectory.

# # # # # # # # # # # # # # #   FIX-T23-4  initial_sample_sigma: 0.1 (từ 0.3). 0.3 quá lớn trong
# # # # # # # # # # # # # # #              normalised space, gây spread bùng nổ ngay từ đầu.

# # # # # # # # # # # # # # #   FIX-T23-5  ctx_noise_scale: 0.02 (từ 0.05). Giảm context noise để
# # # # # # # # # # # # # # #              kiểm soát ensemble spread.

# # # # # # # # # # # # # # #   FIX-T23-6  patience: 15 (từ 6). Với val_ade_freq=2, patience=6 chỉ
# # # # # # # # # # # # # # #              tương đương 12 epoch thực → stop quá sớm sau curriculum jump.

# # # # # # # # # # # # # # #   FIX-T23-7  step_weight_alpha schedule: giảm từ 1.0 (ep 0) → 0.0 (ep 30)
# # # # # # # # # # # # # # #              tuyến tính. Sau ep 30 → uniform weights = standard AFCRPS.

# # # # # # # # # # # # # # #   FIX-T23-8  evaluate_full_val_ade: log thêm PINN loss trung bình để
# # # # # # # # # # # # # # #              verify PINN đang học (không còn = 100 constant).

# # # # # # # # # # # # # # #   FIX-T23-9  GPH500 verification: check mean trong range (27-90) thay vì
# # # # # # # # # # # # # # #              check == 0 (mean=-0.058 là pre-normed gph500 cũ, không đúng).
# # # # # # # # # # # # # # #              Với FIX-DATA-18, CSV gph500 là raw dam → mean ≈ 33 sau sentinel.

# # # # # # # # # # # # # # # Kept from v22:
# # # # # # # # # # # # # # #   FIX-V22-1  StepErrorAccumulator pad zeros (v5/v6)
# # # # # # # # # # # # # # #   FIX-V22-2  evaluate_full_val_ade mỗi val_ade_freq epoch
# # # # # # # # # # # # # # #   FIX-V22-3  Log active_steps từ accumulator
# # # # # # # # # # # # # # # """
# # # # # # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # # # # # import sys
# # # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # # # # # # # # import argparse
# # # # # # # # # # # # # # # import time
# # # # # # # # # # # # # # # import math
# # # # # # # # # # # # # # # import random
# # # # # # # # # # # # # # # import copy

# # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # import torch
# # # # # # # # # # # # # # # import torch.optim as optim
# # # # # # # # # # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # # # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # # # # # # # # from Model.data.loader_training import data_loader
# # # # # # # # # # # # # # # from Model.flow_matching_model import TCFlowMatching
# # # # # # # # # # # # # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # # # # # # # # # from Model.losses import WEIGHTS as _BASE_WEIGHTS
# # # # # # # # # # # # # # # from utils.metrics import (
# # # # # # # # # # # # # # #     TCEvaluator, StepErrorAccumulator,
# # # # # # # # # # # # # # #     save_metrics_csv, haversine_km_torch,
# # # # # # # # # # # # # # #     denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
# # # # # # # # # # # # # # #     cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
# # # # # # # # # # # # # # #     brier_skill_score, cliper_errors, persistence_errors,
# # # # # # # # # # # # # # # )
# # # # # # # # # # # # # # # from utils.evaluation_tables import (
# # # # # # # # # # # # # # #     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
# # # # # # # # # # # # # # #     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
# # # # # # # # # # # # # # #     DEFAULT_COMPUTE, paired_tests,
# # # # # # # # # # # # # # # )
# # # # # # # # # # # # # # # from scripts.statistical_tests import run_all_tests


# # # # # # # # # # # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # # def haversine_km_np_local(pred_deg: np.ndarray,
# # # # # # # # # # # # # # #                            gt_deg: np.ndarray) -> np.ndarray:
# # # # # # # # # # # # # # #     pred_deg = np.atleast_2d(pred_deg)
# # # # # # # # # # # # # # #     gt_deg   = np.atleast_2d(gt_deg)
# # # # # # # # # # # # # # #     R = 6371.0
# # # # # # # # # # # # # # #     lon1, lat1 = np.radians(pred_deg[:, 0]), np.radians(pred_deg[:, 1])
# # # # # # # # # # # # # # #     lon2, lat2 = np.radians(gt_deg[:, 0]),   np.radians(gt_deg[:, 1])
# # # # # # # # # # # # # # #     dlon = lon2 - lon1;  dlat = lat2 - lat1
# # # # # # # # # # # # # # #     a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
# # # # # # # # # # # # # # #     return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# # # # # # # # # # # # # # # def seq_ade_km(pred_norm: np.ndarray, gt_norm: np.ndarray) -> float:
# # # # # # # # # # # # # # #     return float(haversine_km_np_local(denorm_deg_np(pred_norm),
# # # # # # # # # # # # # # #                                        denorm_deg_np(gt_norm)).mean())


# # # # # # # # # # # # # # # # ── Adaptive weight schedules ─────────────────────────────────────────────────

# # # # # # # # # # # # # # # # def get_pinn_weight(epoch, warmup_epochs=30, w_start=0.001, w_end=0.05):
# # # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # # #     FIX-T23-8: w_start=0.001 (từ 0.01), w_end=0.05 (từ 0.1).
# # # # # # # # # # # # # # # #     PINN bắt đầu rất nhỏ để FM học trước, sau đó tăng dần.
# # # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # # # # # #         return w_end
# # # # # # # # # # # # # # # #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)
# # # # # # # # # # # # # # # # def get_pinn_weight(epoch, warmup_epochs=30, w_start=0.01, w_end=0.05): # Sửa w_start ở đây
# # # # # # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # # # # # #         return w_end
# # # # # # # # # # # # # # # #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)

 
# # # # # # # # # # # # # # # def get_pinn_weight(epoch, warmup_epochs=50, w_start=0.001, w_end=0.05):
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     FIX-T24-1: warmup dài hơn (50 ep) để PINN học sau khi FM ổn định.
# # # # # # # # # # # # # # #     w_start giảm hơn để không interfere với FM học ban đầu.
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # # # # #         return w_end
# # # # # # # # # # # # # # #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)

# # # # # # # # # # # # # # # def get_velocity_weight(epoch, warmup_epochs=20, w_start=0.5, w_end=1.5):
# # # # # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # # # # #         return w_end
# # # # # # # # # # # # # # #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)

 
 
# # # # # # # # # # # # # # # def get_progressive_ens(epoch, n_train_ens=6):
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     FIX-T24-4: Progressive ensemble tăng chậm hơn.
# # # # # # # # # # # # # # #     ep 0-19: 1 sample (fast training)
# # # # # # # # # # # # # # #     ep 20-49: 2 samples (begin diversity)
# # # # # # # # # # # # # # #     ep 50+: n_train_ens (full ensemble)
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     if epoch < 20:
# # # # # # # # # # # # # # #         return 1
# # # # # # # # # # # # # # #     elif epoch < 50:
# # # # # # # # # # # # # # #         return 2
# # # # # # # # # # # # # # #     else:
# # # # # # # # # # # # # # #         return n_train_ens
 
 
# # # # # # # # # # # # # # # def _check_uv500(bl):
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     FIX-T24-6: Verify u/v500 sau FIX-ENV-20 và FIX-DATA-28.
# # # # # # # # # # # # # # #     Expected: mean trong range [-1, 1] và không phải 0.
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     env_data = bl[13]
# # # # # # # # # # # # # # #     if env_data is None:
# # # # # # # # # # # # # # #         print("  ⚠️  env_data is None at epoch 0")
# # # # # # # # # # # # # # #         return
 
# # # # # # # # # # # # # # #     # Check u500
# # # # # # # # # # # # # # #     for key in ("u500_mean", "v500_mean"):
# # # # # # # # # # # # # # #         if key not in env_data:
# # # # # # # # # # # # # # #             print(f"  ⚠️  {key} không có trong env_data!")
# # # # # # # # # # # # # # #             continue
 
# # # # # # # # # # # # # # #         val = env_data[key]
# # # # # # # # # # # # # # #         n_zero  = (val == 0).sum().item()
# # # # # # # # # # # # # # #         n_total = val.numel()
# # # # # # # # # # # # # # #         zero_pct = 100.0 * n_zero / max(n_total, 1)
# # # # # # # # # # # # # # #         mean_val = val.mean().item()
# # # # # # # # # # # # # # #         std_val  = val.std().item()
 
# # # # # # # # # # # # # # #         if zero_pct > 80.0:
# # # # # # # # # # # # # # #             print(f"\n{'!' * 60}")
# # # # # # # # # # # # # # #             print(f"  ⚠️  {key} = 0 cho {zero_pct:.1f}% samples!")
# # # # # # # # # # # # # # #             print(f"     mean={mean_val:.4f}, std={std_val:.4f}")
# # # # # # # # # # # # # # #             print(f"     FIX-DATA-28 chưa được apply hoặc CSV không có d3d_u500_mean_raw")
# # # # # # # # # # # # # # #             print(f"{'!' * 60}\n")
# # # # # # # # # # # # # # #         elif abs(mean_val) < 0.01 and std_val < 0.01:
# # # # # # # # # # # # # # #             print(f"  ⚠️  {key} gần 0 (mean={mean_val:.4f}, std={std_val:.4f})")
# # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # #             print(f"  ✅ {key} OK: mean={mean_val:.4f}, std={std_val:.4f}, zero={zero_pct:.1f}%")
 
# # # # # # # # # # # # # # #     # Check gph500
# # # # # # # # # # # # # # #     if "gph500_mean" in env_data:
# # # # # # # # # # # # # # #         gph_val = env_data["gph500_mean"]
# # # # # # # # # # # # # # #         gph_mean = gph_val.mean().item()
# # # # # # # # # # # # # # #         zero_pct = 100.0 * (gph_val == 0).sum().item() / max(gph_val.numel(), 1)
 
# # # # # # # # # # # # # # #         if abs(gph_mean) < 1.0 and zero_pct > 50.0:
# # # # # # # # # # # # # # #             print(f"  ⚠️  GPH500 mean≈0 ({gph_mean:.4f}) → data issue")
# # # # # # # # # # # # # # #         elif 25.0 < gph_mean < 95.0:
# # # # # # # # # # # # # # #             print(f"  ✅ GPH500 OK (mean={gph_mean:.2f} dam, zero={zero_pct:.1f}%)")
# # # # # # # # # # # # # # #         elif -30.0 < gph_mean < 5.0:
# # # # # # # # # # # # # # #             print(f"  ℹ️  GPH500 pre-normalized (mean={gph_mean:.4f}) - OK if from .npy")
# # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # #             print(f"  ⚠️  GPH500 unexpected (mean={gph_mean:.4f})")
 

# # # # # # # # # # # # # # # def get_recurv_weight(epoch, warmup_epochs=10, w_start=0.3, w_end=1.0):
# # # # # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # # # # #         return w_end
# # # # # # # # # # # # # # #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# # # # # # # # # # # # # # # def get_grad_clip(epoch, warmup_epochs=20, clip_start=2.0, clip_end=1.0):
# # # # # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # # # # #         return clip_end
# # # # # # # # # # # # # # #     return clip_start - (epoch / max(warmup_epochs-1, 1)) * (clip_start - clip_end)


# # # # # # # # # # # # # # # def get_step_weight_alpha(epoch, decay_epochs=60) -> float: # 30→60
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     FIX-T23-7: step_weight_alpha replaces curriculum.
# # # # # # # # # # # # # # #     alpha=1.0 at ep 0 → 0.0 at ep decay_epochs.
# # # # # # # # # # # # # # #     After decay_epochs: uniform AFCRPS weights.
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     if epoch >= decay_epochs:
# # # # # # # # # # # # # # #         return 0.0
# # # # # # # # # # # # # # #     return 1.0 - (epoch / decay_epochs)


# # # # # # # # # # # # # # # def get_args():
# # # # # # # # # # # # # # #     p = argparse.ArgumentParser(
# # # # # # # # # # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # # # # # # # # # #     p.add_argument("--dataset_root",    default="TCND_vn",      type=str)
# # # # # # # # # # # # # # #     p.add_argument("--obs_len",         default=8,              type=int)
# # # # # # # # # # # # # # #     p.add_argument("--pred_len",        default=12,             type=int)
# # # # # # # # # # # # # # #     p.add_argument("--test_year",       default=None,           type=int)
# # # # # # # # # # # # # # #     p.add_argument("--batch_size",      default=32,             type=int)
# # # # # # # # # # # # # # #     p.add_argument("--num_epochs",      default=200,            type=int)
# # # # # # # # # # # # # # #     p.add_argument("--g_learning_rate", default=2e-4,           type=float)
# # # # # # # # # # # # # # #     p.add_argument("--weight_decay",    default=1e-4,           type=float)
# # # # # # # # # # # # # # #     p.add_argument("--warmup_epochs",   default=3,              type=int)
# # # # # # # # # # # # # # #     p.add_argument("--grad_clip",       default=2.0,            type=float)
# # # # # # # # # # # # # # #     p.add_argument("--grad_accum",      default=2,              type=int)
# # # # # # # # # # # # # # #     p.add_argument("--patience",        default=15,             type=int,
# # # # # # # # # # # # # # #                    help="FIX-T23-6: tăng từ 6 lên 15 để tránh stop sớm.")
# # # # # # # # # # # # # # #     p.add_argument("--min_epochs",      default=80,             type=int)
# # # # # # # # # # # # # # #     p.add_argument("--n_train_ens",     default=6,              type=int)
# # # # # # # # # # # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # # # # # # # # # # #     p.add_argument("--num_workers",     default=2,              type=int)

# # # # # # # # # # # # # # #     # FIX-T23-4/5: sigma giảm
# # # # # # # # # # # # # # #     p.add_argument("--sigma_min",            default=0.02,  type=float)
# # # # # # # # # # # # # # #     # p.add_argument("--ctx_noise_scale",      default=0.02,  type=float,
# # # # # # # # # # # # # # #     #                help="FIX-T23-5: giảm từ 0.05 → 0.02")
# # # # # # # # # # # # # # #     # p.add_argument("--initial_sample_sigma", default=0.1,   type=float,
# # # # # # # # # # # # # # #     #                help="FIX-T23-4: giảm từ 0.3 → 0.1")
# # # # # # # # # # # # # # #     # train_flowmatching.py — get_args()
# # # # # # # # # # # # # # #     p.add_argument("--ctx_noise_scale",      default=0.005,  type=float,
# # # # # # # # # # # # # # #                 help="FIX-T24-B: giảm từ 0.02 → 0.005, kiểm soát spread")
# # # # # # # # # # # # # # #     p.add_argument("--initial_sample_sigma", default=0.05,   type=float,
# # # # # # # # # # # # # # #                 help="FIX-T24-B: giảm từ 0.1 → 0.05")

# # # # # # # # # # # # # # #     # FIX-T23-3: ODE steps tăng
# # # # # # # # # # # # # # #     p.add_argument("--ode_steps_train", default=20,  type=int,
# # # # # # # # # # # # # # #                    help="FIX-T23-3: từ 10 → 20")
# # # # # # # # # # # # # # #     p.add_argument("--ode_steps_val",   default=30,  type=int,
# # # # # # # # # # # # # # #                    help="FIX-T23-3: từ 10 → 30")
# # # # # # # # # # # # # # #     p.add_argument("--ode_steps_test",  default=50,  type=int)
# # # # # # # # # # # # # # #     p.add_argument("--ode_steps",       default=None, type=int,
# # # # # # # # # # # # # # #                    help="Override train/val/test steps (for testing)")

# # # # # # # # # # # # # # #     p.add_argument("--val_ensemble",    default=30,             type=int)
# # # # # # # # # # # # # # #     p.add_argument("--fast_ensemble",   default=8,              type=int)

# # # # # # # # # # # # # # #     p.add_argument("--fno_modes_h",      default=4,             type=int)
# # # # # # # # # # # # # # #     p.add_argument("--fno_modes_t",      default=4,             type=int)
# # # # # # # # # # # # # # #     p.add_argument("--fno_layers",       default=4,             type=int)
# # # # # # # # # # # # # # #     p.add_argument("--fno_d_model",      default=32,            type=int)
# # # # # # # # # # # # # # #     p.add_argument("--fno_spatial_down", default=32,            type=int)
# # # # # # # # # # # # # # #     p.add_argument("--mamba_d_state",    default=16,            type=int)

# # # # # # # # # # # # # # #     p.add_argument("--val_loss_freq",   default=1,              type=int)
# # # # # # # # # # # # # # #     p.add_argument("--val_freq",        default=1,              type=int)
# # # # # # # # # # # # # # #     p.add_argument("--val_ade_freq",    default=1,              type=int)
# # # # # # # # # # # # # # #     p.add_argument("--full_eval_freq",  default=10,             type=int)
# # # # # # # # # # # # # # #     p.add_argument("--val_subset_size", default=600,            type=int)

# # # # # # # # # # # # # # #     p.add_argument("--output_dir",      default="runs/v23",     type=str)
# # # # # # # # # # # # # # #     p.add_argument("--save_interval",   default=10,             type=int)
# # # # # # # # # # # # # # #     p.add_argument("--metrics_csv",     default="metrics.csv",     type=str)
# # # # # # # # # # # # # # #     p.add_argument("--predict_csv",     default="predictions.csv", type=str)
# # # # # # # # # # # # # # #     p.add_argument("--lstm_errors_npy",      default=None, type=str)
# # # # # # # # # # # # # # #     p.add_argument("--diffusion_errors_npy", default=None, type=str)

# # # # # # # # # # # # # # #     p.add_argument("--gpu_num",         default="0",            type=str)
# # # # # # # # # # # # # # #     p.add_argument("--delim",           default=" ")
# # # # # # # # # # # # # # #     p.add_argument("--skip",            default=1,              type=int)
# # # # # # # # # # # # # # #     p.add_argument("--min_ped",         default=1,              type=int)
# # # # # # # # # # # # # # #     p.add_argument("--threshold",       default=0.002,          type=float)
# # # # # # # # # # # # # # #     p.add_argument("--other_modal",     default="gph")

# # # # # # # # # # # # # # #     # FIX-T23-1: curriculum params REMOVED, replaced by step_weight_alpha
# # # # # # # # # # # # # # #     # p.add_argument("--step_weight_decay_epochs", default=30, type=int,
# # # # # # # # # # # # # # #     #                help="FIX-T23-7: epochs over which alpha decays 1→0")

# # # # # # # # # # # # # # #     p.add_argument("--step_weight_decay_epochs", default=60, type=int,   # 30→60
# # # # # # # # # # # # # # #                help="FIX-T24-A: epochs over which alpha decays 1→0")
    
# # # # # # # # # # # # # # #     p.add_argument("--lon_flip_prob",   default=0.3,            type=float)

# # # # # # # # # # # # # # #     # PINN warmup with smaller values
# # # # # # # # # # # # # # #     p.add_argument("--pinn_warmup_epochs", default=30,          type=int)
# # # # # # # # # # # # # # #     p.add_argument("--pinn_w_start",    default=0.001,          type=float,
# # # # # # # # # # # # # # #                    help="FIX-T23-8: 0.001 (từ 0.01)")
# # # # # # # # # # # # # # #     p.add_argument("--pinn_w_end",      default=0.05,           type=float,
# # # # # # # # # # # # # # #                    help="FIX-T23-8: 0.05 (từ 0.1)")

# # # # # # # # # # # # # # #     p.add_argument("--vel_warmup_epochs",  default=20,          type=float)
# # # # # # # # # # # # # # #     p.add_argument("--vel_w_start",        default=0.5,         type=float)
# # # # # # # # # # # # # # #     p.add_argument("--vel_w_end",          default=1.5,         type=float)
# # # # # # # # # # # # # # #     p.add_argument("--recurv_warmup_epochs", default=10,        type=int)
# # # # # # # # # # # # # # #     p.add_argument("--recurv_w_start",       default=0.3,       type=float)
# # # # # # # # # # # # # # #     p.add_argument("--recurv_w_end",         default=1.0,       type=float)

# # # # # # # # # # # # # # #     return p.parse_args()


# # # # # # # # # # # # # # # def _resolve_ode_steps(args):
# # # # # # # # # # # # # # #     if args.ode_steps is not None:
# # # # # # # # # # # # # # #         return args.ode_steps, args.ode_steps, args.ode_steps
# # # # # # # # # # # # # # #     return args.ode_steps_train, args.ode_steps_val, args.ode_steps_test


# # # # # # # # # # # # # # # def move(batch, device):
# # # # # # # # # # # # # # #     out = list(batch)
# # # # # # # # # # # # # # #     for i, x in enumerate(out):
# # # # # # # # # # # # # # #         if torch.is_tensor(x):
# # # # # # # # # # # # # # #             out[i] = x.to(device)
# # # # # # # # # # # # # # #         elif isinstance(x, dict):
# # # # # # # # # # # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # # # # # # # # #                       for k, v in x.items()}
# # # # # # # # # # # # # # #     return out


# # # # # # # # # # # # # # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # # # # # # # # # # # # # #                             collate_fn, num_workers):
# # # # # # # # # # # # # # #     n   = len(val_dataset)
# # # # # # # # # # # # # # #     rng = random.Random(42)
# # # # # # # # # # # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # # # # # # # # # # #     return DataLoader(Subset(val_dataset, idx),
# # # # # # # # # # # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # # # # # # # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # # # # # # # # # # # ── evaluate_fast ─────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
# # # # # # # # # # # # # # # #     """Monitor nhanh trên val subset. Không dùng để quyết định best model."""
# # # # # # # # # # # # # # # #     model.eval()
# # # # # # # # # # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # # # # # # # # # #     t0  = time.perf_counter()
# # # # # # # # # # # # # # # #     n   = 0
# # # # # # # # # # # # # # # #     spread_buf = []

# # # # # # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # # # # # #         for batch in loader:
# # # # # # # # # # # # # # # #             bl = move(list(batch), device)
# # # # # # # # # # # # # # # #             pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
# # # # # # # # # # # # # # # #                                               ddim_steps=ode_steps)
# # # # # # # # # # # # # # # #             T_active  = pred.shape[0]
# # # # # # # # # # # # # # # #             gt_sliced = bl[1][:T_active]
# # # # # # # # # # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
# # # # # # # # # # # # # # # #             acc.update(dist)

# # # # # # # # # # # # # # # #             last_step = all_trajs[:, -1, :, :]
# # # # # # # # # # # # # # # #             std_lon   = last_step[:, :, 0].std(0)
# # # # # # # # # # # # # # # #             std_lat   = last_step[:, :, 1].std(0)
# # # # # # # # # # # # # # # #             spread_km = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# # # # # # # # # # # # # # # #             spread_buf.append(spread_km)
# # # # # # # # # # # # # # # #             n += 1

# # # # # # # # # # # # # # # #     r = acc.compute()
# # # # # # # # # # # # # # # #     r["ms_per_batch"]  = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# # # # # # # # # # # # # # # #     r["spread_72h_km"] = float(np.mean(spread_buf)) if spread_buf else 0.0
# # # # # # # # # # # # # # # #     return r

# # # # # # # # # # # # # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     FIX-T24-7: Log thêm per-step spread để monitor FIX-L46.
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     import torch
# # # # # # # # # # # # # # #     import numpy as np
# # # # # # # # # # # # # # #     import time
 
# # # # # # # # # # # # # # #     model.eval()
# # # # # # # # # # # # # # #     from utils.metrics import StepErrorAccumulator, haversine_km_torch, denorm_torch
# # # # # # # # # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # # # # # # # # #     t0  = time.perf_counter()
# # # # # # # # # # # # # # #     n   = 0
# # # # # # # # # # # # # # #     spread_per_step = []
 
# # # # # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # # # # #         for batch in loader:
# # # # # # # # # # # # # # #             bl = list(batch)
# # # # # # # # # # # # # # #             for i, x in enumerate(bl):
# # # # # # # # # # # # # # #                 if torch.is_tensor(x):
# # # # # # # # # # # # # # #                     bl[i] = x.to(device)
# # # # # # # # # # # # # # #                 elif isinstance(x, dict):
# # # # # # # # # # # # # # #                     bl[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # # # # # # # # #                               for k, v in x.items()}
 
# # # # # # # # # # # # # # #             pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
# # # # # # # # # # # # # # #                                               ddim_steps=ode_steps)
# # # # # # # # # # # # # # #             T_active  = pred.shape[0]
# # # # # # # # # # # # # # #             gt_sliced = bl[1][:T_active]
# # # # # # # # # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
# # # # # # # # # # # # # # #             acc.update(dist)
 
# # # # # # # # # # # # # # #             # FIX-T24-7: per-step spread
# # # # # # # # # # # # # # #             step_spreads = []
# # # # # # # # # # # # # # #             for t in range(all_trajs.shape[1]):
# # # # # # # # # # # # # # #                 step_data = all_trajs[:, t, :, :]
# # # # # # # # # # # # # # #                 std_lon = step_data[:, :, 0].std(0)
# # # # # # # # # # # # # # #                 std_lat = step_data[:, :, 1].std(0)
# # # # # # # # # # # # # # #                 spread = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# # # # # # # # # # # # # # #                 step_spreads.append(spread)
# # # # # # # # # # # # # # #             spread_per_step.append(step_spreads)
# # # # # # # # # # # # # # #             n += 1
 
# # # # # # # # # # # # # # #     r = acc.compute()
# # # # # # # # # # # # # # #     r["ms_per_batch"]  = (time.perf_counter() - t0) * 1e3 / max(n, 1)
 
# # # # # # # # # # # # # # #     if spread_per_step:
# # # # # # # # # # # # # # #         spreads = np.array(spread_per_step)  # [n_batches, T]
# # # # # # # # # # # # # # #         r["spread_12h_km"] = float(spreads[:, 1].mean()) if spreads.shape[1] > 1 else 0.0
# # # # # # # # # # # # # # #         r["spread_24h_km"] = float(spreads[:, 3].mean()) if spreads.shape[1] > 3 else 0.0
# # # # # # # # # # # # # # #         r["spread_72h_km"] = float(spreads[:, -1].mean())
# # # # # # # # # # # # # # #     return r

# # # # # # # # # # # # # # # # ── evaluate_full_val_ade ─────────────────────────────────────────────────────

# # # # # # # # # # # # # # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # # # # # # # # # # # # # #                            fast_ensemble, metrics_csv, epoch, tag=""):
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     FIX-T23-2: Full val ADE là DUY NHẤT criteria để lưu best model.
# # # # # # # # # # # # # # #     FIX-T23-8: Log PINN loss trung bình để verify không còn saturate.
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     model.eval()
# # # # # # # # # # # # # # #     acc      = StepErrorAccumulator(pred_len)
# # # # # # # # # # # # # # #     t0       = time.perf_counter()
# # # # # # # # # # # # # # #     n_batch  = 0
# # # # # # # # # # # # # # #     pinn_buf = []

# # # # # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # # # # #         for batch in val_loader:
# # # # # # # # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # # # # # # # #             # Luôn sample full pred_len (không curriculum)
# # # # # # # # # # # # # # #             pred, _, _ = model.sample(bl, num_ensemble=fast_ensemble,
# # # # # # # # # # # # # # #                                        ddim_steps=ode_steps)
# # # # # # # # # # # # # # #             T_pred = pred.shape[0]
# # # # # # # # # # # # # # #             gt     = bl[1][:T_pred]
# # # # # # # # # # # # # # #             dist   = haversine_km_torch(denorm_torch(pred), denorm_torch(gt))
# # # # # # # # # # # # # # #             acc.update(dist)

# # # # # # # # # # # # # # #             # FIX-T23-8: compute PINN để verify
# # # # # # # # # # # # # # #             try:
# # # # # # # # # # # # # # #                 from Model.losses import pinn_bve_loss, _haversine_deg
# # # # # # # # # # # # # # #                 from utils.metrics import denorm_deg_np
# # # # # # # # # # # # # # #                 pred_deg = pred.clone()
# # # # # # # # # # # # # # #                 pred_deg[..., 0] = (pred[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # # # # # #                 pred_deg[..., 1] = (pred[..., 1] * 50.0) / 10.0
# # # # # # # # # # # # # # #                 env_d = bl[13] if len(bl) > 13 else None
# # # # # # # # # # # # # # #                 pinn_val = pinn_bve_loss(pred_deg, bl, env_data=env_d).item()
# # # # # # # # # # # # # # #                 pinn_buf.append(pinn_val)
# # # # # # # # # # # # # # #             except Exception:
# # # # # # # # # # # # # # #                 pass

# # # # # # # # # # # # # # #             n_batch += 1

# # # # # # # # # # # # # # #     elapsed = time.perf_counter() - t0
# # # # # # # # # # # # # # #     r       = acc.compute()

# # # # # # # # # # # # # # #     ade_str = f"{r.get('ADE', float('nan')):.1f}"
# # # # # # # # # # # # # # #     fde_str = f"{r.get('FDE', float('nan')):.1f}"
# # # # # # # # # # # # # # #     h12     = f"{r.get('12h', float('nan')):.0f}"
# # # # # # # # # # # # # # #     h24     = f"{r.get('24h', float('nan')):.0f}"
# # # # # # # # # # # # # # #     h48     = f"{r.get('48h', float('nan')):.0f}"
# # # # # # # # # # # # # # #     h72     = f"{r.get('72h', float('nan')):.0f}"
# # # # # # # # # # # # # # #     pinn_mean = f"{np.mean(pinn_buf):.3f}" if pinn_buf else "N/A"

# # # # # # # # # # # # # # #     print(f"\n{'='*64}")
# # # # # # # # # # # # # # #     print(f"  [FULL VAL ADE  ep={epoch}  {elapsed:.0f}s  {n_batch} batches]")
# # # # # # # # # # # # # # #     print(f"  ADE={ade_str} km  FDE={fde_str} km")
# # # # # # # # # # # # # # #     print(f"  12h={h12}  24h={h24}  48h={h48}  72h={h72} km")
# # # # # # # # # # # # # # #     print(f"  PINN_mean={pinn_mean}  "  # FIX-T23-8
# # # # # # # # # # # # # # #           f"samples={r.get('n_samples',0)}  ens={fast_ensemble}  steps={ode_steps}")
# # # # # # # # # # # # # # #     print(f"{'='*64}\n")

# # # # # # # # # # # # # # #     from datetime import datetime
# # # # # # # # # # # # # # #     tag_str = tag or f"val_full_ep{epoch:03d}"
# # # # # # # # # # # # # # #     from utils.metrics import DatasetMetrics, save_metrics_csv
# # # # # # # # # # # # # # #     dm = DatasetMetrics(
# # # # # # # # # # # # # # #         ade      = r.get("ADE",  float("nan")),
# # # # # # # # # # # # # # #         fde      = r.get("FDE",  float("nan")),
# # # # # # # # # # # # # # #         ugde_12h = r.get("12h",  float("nan")),
# # # # # # # # # # # # # # #         ugde_24h = r.get("24h",  float("nan")),
# # # # # # # # # # # # # # #         ugde_48h = r.get("48h",  float("nan")),
# # # # # # # # # # # # # # #         ugde_72h = r.get("72h",  float("nan")),
# # # # # # # # # # # # # # #         n_total  = r.get("n_samples", 0),
# # # # # # # # # # # # # # #         timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # # # # # # # # # # #     )
# # # # # # # # # # # # # # #     save_metrics_csv(dm, metrics_csv, tag=tag_str)
# # # # # # # # # # # # # # #     return r


# # # # # # # # # # # # # # # def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
# # # # # # # # # # # # # # #                   metrics_csv, tag="", predict_csv=""):
# # # # # # # # # # # # # # #     """Full 4-tier evaluation."""
# # # # # # # # # # # # # # #     model.eval()
# # # # # # # # # # # # # # #     cliper_step_errors = []
# # # # # # # # # # # # # # #     ev = TCEvaluator(pred_len=pred_len, compute_dtw=True)
# # # # # # # # # # # # # # #     obs_seqs_01 = []; gt_seqs_01 = []; pred_seqs_01 = []; ens_seqs_01 = []

# # # # # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # # # # #         for batch in loader:
# # # # # # # # # # # # # # #             bl  = move(list(batch), device)
# # # # # # # # # # # # # # #             gt  = bl[1];  obs = bl[0]
# # # # # # # # # # # # # # #             pred_mean, _, all_trajs = model.sample(
# # # # # # # # # # # # # # #                 bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
# # # # # # # # # # # # # # #                 predict_csv=predict_csv if predict_csv else None)

# # # # # # # # # # # # # # #             pd_np = denorm_torch(pred_mean).cpu().numpy()
# # # # # # # # # # # # # # #             gd_np = denorm_torch(gt).cpu().numpy()
# # # # # # # # # # # # # # #             od_np = denorm_torch(obs).cpu().numpy()
# # # # # # # # # # # # # # #             ed_np = denorm_torch(all_trajs).cpu().numpy()

# # # # # # # # # # # # # # #             B = pd_np.shape[1]
# # # # # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # # # # #                 ens_b = ed_np[:, :, b, :]
# # # # # # # # # # # # # # #                 ev.update(pd_np[:, b, :], gd_np[:, b, :], pred_ens=ens_b)
# # # # # # # # # # # # # # #                 obs_seqs_01.append(od_np[:, b, :])
# # # # # # # # # # # # # # #                 gt_seqs_01.append(gd_np[:, b, :])
# # # # # # # # # # # # # # #                 pred_seqs_01.append(pd_np[:, b, :])
# # # # # # # # # # # # # # #                 ens_seqs_01.append(ens_b)

# # # # # # # # # # # # # # #                 obs_b_norm = obs.cpu().numpy()[:, b, :]
# # # # # # # # # # # # # # #                 cliper_errors_b = np.zeros(pred_len)
# # # # # # # # # # # # # # #                 for h in range(pred_len):
# # # # # # # # # # # # # # #                     pred_cliper_norm = cliper_forecast(obs_b_norm, h + 1)
# # # # # # # # # # # # # # #                     pred_cliper_01   = denorm_np(pred_cliper_norm[np.newaxis])
# # # # # # # # # # # # # # #                     gt_01            = gd_np[h, b, :][np.newaxis]
# # # # # # # # # # # # # # #                     from utils.metrics import haversine_km_np
# # # # # # # # # # # # # # #                     cliper_errors_b[h] = float(
# # # # # # # # # # # # # # #                         haversine_km_np(pred_cliper_01, gt_01, unit_01deg=True)[0])
# # # # # # # # # # # # # # #                 cliper_step_errors.append(cliper_errors_b)

# # # # # # # # # # # # # # #     if cliper_step_errors:
# # # # # # # # # # # # # # #         cliper_mat       = np.stack(cliper_step_errors)
# # # # # # # # # # # # # # #         cliper_ugde_dict = {h: float(cliper_mat[:, s].mean())
# # # # # # # # # # # # # # #                             for h, s in HORIZON_STEPS.items()
# # # # # # # # # # # # # # #                             if s < cliper_mat.shape[1]}
# # # # # # # # # # # # # # #         ev.cliper_ugde   = cliper_ugde_dict
# # # # # # # # # # # # # # #         print(f"  [CLIPER UGDE] 72h={cliper_ugde_dict.get(72, float('nan')):.1f} km")

# # # # # # # # # # # # # # #     dm = ev.compute(tag=tag)

# # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # #         if LANDFALL_TARGETS and ens_seqs_01:
# # # # # # # # # # # # # # #             bss_vals = []
# # # # # # # # # # # # # # #             step_72  = HORIZON_STEPS.get(72, pred_len - 1)
# # # # # # # # # # # # # # #             for tname, t_lon, t_lat in LANDFALL_TARGETS:
# # # # # # # # # # # # # # #                 bv = brier_skill_score(
# # # # # # # # # # # # # # #                     ens_seqs_01, gt_seqs_01, min(step_72, pred_len-1),
# # # # # # # # # # # # # # #                     (t_lon * 10.0, t_lat * 10.0), LANDFALL_RADIUS_KM)
# # # # # # # # # # # # # # #                 if not math.isnan(bv):
# # # # # # # # # # # # # # #                     bss_vals.append(bv)
# # # # # # # # # # # # # # #             if bss_vals:
# # # # # # # # # # # # # # #                 dm.bss_mean = float(np.mean(bss_vals))
# # # # # # # # # # # # # # #                 print(f"  [BSS] mean={dm.bss_mean:.4f}")
# # # # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # # # #         print(f"  ⚠  BSS failed: {e}")

# # # # # # # # # # # # # # #     save_metrics_csv(dm, metrics_csv, tag=tag)
# # # # # # # # # # # # # # #     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# # # # # # # # # # # # # # # # ── BestModelSaver ────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # # class BestModelSaver:
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     FIX-T23-2: CHỈ dùng full_val ADE để quyết định best model và patience.
# # # # # # # # # # # # # # #     Subset ADE không ảnh hưởng đến saver logic.
# # # # # # # # # # # # # # #     """

# # # # # # # # # # # # # # #     def __init__(self, patience=15, ade_tol=5.0):
# # # # # # # # # # # # # # #         self.patience      = patience
# # # # # # # # # # # # # # #         self.ade_tol       = ade_tol
# # # # # # # # # # # # # # #         self.best_ade      = float("inf")
# # # # # # # # # # # # # # #         self.best_val_loss = float("inf")
# # # # # # # # # # # # # # #         self.counter_ade   = 0
# # # # # # # # # # # # # # #         self.counter_loss  = 0
# # # # # # # # # # # # # # #         self.early_stop    = False

# # # # # # # # # # # # # # #     def reset_counters(self, reason=""):
# # # # # # # # # # # # # # #         self.counter_ade  = 0
# # # # # # # # # # # # # # #         self.counter_loss = 0
# # # # # # # # # # # # # # #         if reason:
# # # # # # # # # # # # # # #             print(f"  [SAVER] Patience reset: {reason}")

# # # # # # # # # # # # # # #     def update_val_loss(self, val_loss, model, out_dir, epoch, optimizer, tl):
# # # # # # # # # # # # # # #         if val_loss < self.best_val_loss - 1e-4:
# # # # # # # # # # # # # # #             self.best_val_loss = val_loss
# # # # # # # # # # # # # # #             self.counter_loss  = 0
# # # # # # # # # # # # # # #             torch.save(dict(
# # # # # # # # # # # # # # #                 epoch=epoch, model_state_dict=model.state_dict(),
# # # # # # # # # # # # # # #                 optimizer_state=optimizer.state_dict(),
# # # # # # # # # # # # # # #                 train_loss=tl, val_loss=val_loss,
# # # # # # # # # # # # # # #                 model_version="v23-valloss"),
# # # # # # # # # # # # # # #                 os.path.join(out_dir, "best_model_valloss.pth"))
# # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # #             self.counter_loss += 1

# # # # # # # # # # # # # # #     def update_ade_full_val(self, ade, model, out_dir, epoch,
# # # # # # # # # # # # # # #                              optimizer, tl, vl, min_epochs=80):
# # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # #         FIX-T23-2: Chỉ gọi khi có full_val ADE. Đây là criteria duy nhất.
# # # # # # # # # # # # # # #         """
# # # # # # # # # # # # # # #         if ade < self.best_ade - self.ade_tol:
# # # # # # # # # # # # # # #             self.best_ade     = ade
# # # # # # # # # # # # # # #             self.counter_ade  = 0
# # # # # # # # # # # # # # #             torch.save(dict(
# # # # # # # # # # # # # # #                 epoch=epoch, model_state_dict=model.state_dict(),
# # # # # # # # # # # # # # #                 optimizer_state=optimizer.state_dict(),
# # # # # # # # # # # # # # #                 train_loss=tl, val_loss=vl, val_ade_km=ade,
# # # # # # # # # # # # # # #                 model_version="v23-FNO-Mamba-OT-CFM"),
# # # # # # # # # # # # # # #                 os.path.join(out_dir, "best_model.pth"))
# # # # # # # # # # # # # # #             print(f"  ✅ Best full-val ADE {ade:.1f} km  (epoch {epoch})")
# # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # #             self.counter_ade += 1
# # # # # # # # # # # # # # #             print(f"  No ADE improvement {self.counter_ade}/{self.patience}"
# # # # # # # # # # # # # # #                   f"  (Δ={self.best_ade - ade:.1f} km < tol={self.ade_tol} km)"
# # # # # # # # # # # # # # #                   f"  | Loss counter {self.counter_loss}/{self.patience}"
# # # # # # # # # # # # # # #                   f"  [full_val]")

# # # # # # # # # # # # # # #         if epoch >= min_epochs:
# # # # # # # # # # # # # # #             if (self.counter_ade >= self.patience
# # # # # # # # # # # # # # #                     and self.counter_loss >= self.patience):
# # # # # # # # # # # # # # #                 self.early_stop = True
# # # # # # # # # # # # # # #                 print(f"  ⛔ Early stop @ epoch {epoch}")
# # # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # # #             if (self.counter_ade >= self.patience
# # # # # # # # # # # # # # #                     and self.counter_loss >= self.patience):
# # # # # # # # # # # # # # #                 print(f"  ⚠  Would stop but min_epochs={min_epochs} not reached.")
# # # # # # # # # # # # # # #                 self.counter_ade  = 0
# # # # # # # # # # # # # # #                 self.counter_loss = 0

# # # # # # # # # # # # # # #     def log_subset_ade(self, ade: float, epoch: int):
# # # # # # # # # # # # # # #         """FIX-T23-2: subset ADE chỉ để log, không ảnh hưởng patience."""
# # # # # # # # # # # # # # #         print(f"  [SUBSET-ADE ep{epoch}]  {ade:.1f} km  (monitor only, not used for best model)")


# # # # # # # # # # # # # # # def _load_baseline_errors(path, name):
# # # # # # # # # # # # # # #     if path is None:
# # # # # # # # # # # # # # #         print(f"\n  ⚠  {name} errors not provided — skip stat comparison.\n")
# # # # # # # # # # # # # # #         return None
# # # # # # # # # # # # # # #     if not os.path.exists(path):
# # # # # # # # # # # # # # #         print(f"\n  ⚠  {path} not found — {name} skipped.\n")
# # # # # # # # # # # # # # #         return None
# # # # # # # # # # # # # # #     arr = np.load(path)
# # # # # # # # # # # # # # #     print(f"  ✓  Loaded {name}: {arr.shape}")
# # # # # # # # # # # # # # #     return arr


# # # # # # # # # # # # # # # def main(args):
# # # # # # # # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # # # # # # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # # # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # # # # # # # # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# # # # # # # # # # # # # # #     predict_csv = os.path.join(args.output_dir, args.predict_csv)
# # # # # # # # # # # # # # #     tables_dir  = os.path.join(args.output_dir, "tables")
# # # # # # # # # # # # # # #     stat_dir    = os.path.join(tables_dir, "stat_tests")
# # # # # # # # # # # # # # #     os.makedirs(tables_dir, exist_ok=True)
# # # # # # # # # # # # # # #     os.makedirs(stat_dir,   exist_ok=True)

# # # # # # # # # # # # # # #     ode_train, ode_val, ode_test = _resolve_ode_steps(args)

# # # # # # # # # # # # # # #     print("=" * 68)
# # # # # # # # # # # # # # #     print("  TC-FlowMatching v23  |  FNO3D + Mamba + OT-CFM + PINN")
# # # # # # # # # # # # # # #     print("  v23 FIXES:")
# # # # # # # # # # # # # # #     print("    FIX-T23-1: CURRICULUM REMOVED → step_weight_alpha soft weighting")
# # # # # # # # # # # # # # #     print("    FIX-T23-2: Best model CHỈ từ full_val ADE (không dùng subset)")
# # # # # # # # # # # # # # #     print("    FIX-T23-3: ODE steps train=20, val=30 (từ 10/10)")
# # # # # # # # # # # # # # #     print("    FIX-T23-4: initial_sample_sigma=0.1 (từ 0.3)")
# # # # # # # # # # # # # # #     print("    FIX-T23-5: ctx_noise_scale=0.02 (từ 0.05)")
# # # # # # # # # # # # # # #     print("    FIX-T23-6: patience=15 (từ 6)")
# # # # # # # # # # # # # # #     print("    FIX-T23-7: step_weight_alpha 1.0→0.0 over 30 epochs")
# # # # # # # # # # # # # # #     print("    FIX-T23-8: PINN w_start=0.001 (từ 0.01), log PINN trong eval")
# # # # # # # # # # # # # # #     print("    FIX-DATA-18/21: GPH500 từ CSV = raw dam, xử lý đúng")
# # # # # # # # # # # # # # #     print("    FIX-L39/42: PINN scale=1e-3, clamp=50 → gradient không saturate")
# # # # # # # # # # # # # # #     print("=" * 68)
# # # # # # # # # # # # # # #     print(f"  device               : {device}")
# # # # # # # # # # # # # # #     print(f"  sigma_min            : {args.sigma_min}")
# # # # # # # # # # # # # # #     print(f"  ctx_noise_scale      : {args.ctx_noise_scale}")
# # # # # # # # # # # # # # #     print(f"  initial_sample_sigma : {args.initial_sample_sigma}")
# # # # # # # # # # # # # # #     print(f"  ode_steps            : train={ode_train}  val={ode_val}  test={ode_test}")
# # # # # # # # # # # # # # #     print(f"  val_ensemble         : {args.val_ensemble}")
# # # # # # # # # # # # # # #     print(f"  val_ade_freq         : every {args.val_ade_freq} epochs (full val set)")
# # # # # # # # # # # # # # #     print(f"  patience             : {args.patience} epochs  (min_epochs={args.min_epochs})")
# # # # # # # # # # # # # # #     print(f"  step_weight_decay    : {args.step_weight_decay_epochs} epochs")
# # # # # # # # # # # # # # #     print(f"  NO CURRICULUM        : pred_len={args.pred_len} from epoch 0")
# # # # # # # # # # # # # # #     print()

# # # # # # # # # # # # # # #     train_dataset, train_loader = data_loader(
# # # # # # # # # # # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # # # # # # # # # # #     val_dataset, val_loader = data_loader(
# # # # # # # # # # # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # # # # # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # # # # # # # # #     val_subset_loader = make_val_subset_loader(
# # # # # # # # # # # # # # #         val_dataset, args.val_subset_size, args.batch_size,
# # # # # # # # # # # # # # #         seq_collate, args.num_workers)

# # # # # # # # # # # # # # #     test_loader = None
# # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # #         _, test_loader = data_loader(
# # # # # # # # # # # # # # #             args, {"root": args.dataset_root, "type": "test"},
# # # # # # # # # # # # # # #             test=True, test_year=None)
# # # # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # # # #         print(f"  Warning: test loader: {e}")

# # # # # # # # # # # # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # # # # # # # # # # # #     print(f"  val   : {len(val_dataset)} seq")
# # # # # # # # # # # # # # #     if test_loader:
# # # # # # # # # # # # # # #         print(f"  test  : {len(test_loader.dataset)} seq")

# # # # # # # # # # # # # # #     model = TCFlowMatching(
# # # # # # # # # # # # # # #         pred_len             = args.pred_len,
# # # # # # # # # # # # # # #         obs_len              = args.obs_len,
# # # # # # # # # # # # # # #         sigma_min            = args.sigma_min,
# # # # # # # # # # # # # # #         n_train_ens          = args.n_train_ens,
# # # # # # # # # # # # # # #         ctx_noise_scale      = args.ctx_noise_scale,
# # # # # # # # # # # # # # #         initial_sample_sigma = args.initial_sample_sigma,
# # # # # # # # # # # # # # #     ).to(device)

# # # # # # # # # # # # # # #     if (args.fno_spatial_down != 32 or args.fno_modes_h != 4
# # # # # # # # # # # # # # #             or args.fno_layers != 4 or args.fno_d_model != 32):
# # # # # # # # # # # # # # #         from Model.FNO3D_encoder import FNO3DEncoder
# # # # # # # # # # # # # # #         model.net.spatial_enc = FNO3DEncoder(
# # # # # # # # # # # # # # #             in_channel=13, out_channel=1,
# # # # # # # # # # # # # # #             d_model=args.fno_d_model, n_layers=args.fno_layers,
# # # # # # # # # # # # # # #             modes_t=args.fno_modes_t, modes_h=args.fno_modes_h,
# # # # # # # # # # # # # # #             modes_w=args.fno_modes_h, spatial_down=args.fno_spatial_down,
# # # # # # # # # # # # # # #             dropout=0.05).to(device)

# # # # # # # # # # # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # # # # # # # # # #     print(f"  params  : {n_params:,}")

# # # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # # # # # # # # # # #         print("  torch.compile: enabled")
# # # # # # # # # # # # # # #     except Exception:
# # # # # # # # # # # # # # #         pass

# # # # # # # # # # # # # # #     optimizer       = optim.AdamW(model.parameters(),
# # # # # # # # # # # # # # #                                    lr=args.g_learning_rate,
# # # # # # # # # # # # # # #                                    weight_decay=args.weight_decay)
# # # # # # # # # # # # # # #     steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
# # # # # # # # # # # # # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # # # # # # # # # # # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # # # # # # # # # # # # #     scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
# # # # # # # # # # # # # # #     saver     = BestModelSaver(patience=args.patience, ade_tol=1.0)
# # # # # # # # # # # # # # #     scaler    = GradScaler('cuda', enabled=args.use_amp)

# # # # # # # # # # # # # # #     print("=" * 68)
# # # # # # # # # # # # # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch, NO CURRICULUM)")
# # # # # # # # # # # # # # #     print("=" * 68)

# # # # # # # # # # # # # # #     epoch_times   = []
# # # # # # # # # # # # # # #     train_start   = time.perf_counter()
# # # # # # # # # # # # # # #     last_val_loss = float("inf")
# # # # # # # # # # # # # # #     _lr_ep30_done = False
# # # # # # # # # # # # # # #     _lr_ep60_done = False
# # # # # # # # # # # # # # #     _prev_ens     = 1

# # # # # # # # # # # # # # #     import Model.losses as _losses_mod

# # # # # # # # # # # # # # #     for epoch in range(args.num_epochs):
# # # # # # # # # # # # # # #         # Progressive ensemble
# # # # # # # # # # # # # # #         # current_ens = 1 if epoch < 30 else (2 if epoch < 60 else args.n_train_ens)
# # # # # # # # # # # # # # #         current_ens = get_progressive_ens(epoch, args.n_train_ens)
# # # # # # # # # # # # # # #         model.n_train_ens = current_ens
# # # # # # # # # # # # # # #         effective_fast_ens = min(args.fast_ensemble, max(current_ens*2, args.fast_ensemble))

# # # # # # # # # # # # # # #         if current_ens != _prev_ens:
# # # # # # # # # # # # # # #             saver.reset_counters(f"n_train_ens {_prev_ens}→{current_ens} at ep {epoch}")
# # # # # # # # # # # # # # #             _prev_ens = current_ens

# # # # # # # # # # # # # # #         # FIX-T23-1: NO curriculum. Always train on full pred_len.
# # # # # # # # # # # # # # #         # FIX-T23-7: step_weight_alpha replaces curriculum
# # # # # # # # # # # # # # #         step_alpha = get_step_weight_alpha(epoch, args.step_weight_decay_epochs)

# # # # # # # # # # # # # # #         # Weight schedule
# # # # # # # # # # # # # # #         epoch_weights = copy.copy(_BASE_WEIGHTS)
# # # # # # # # # # # # # # #         epoch_weights["pinn"] = get_pinn_weight(
# # # # # # # # # # # # # # #             epoch, args.pinn_warmup_epochs, args.pinn_w_start, args.pinn_w_end)
# # # # # # # # # # # # # # #         epoch_weights["velocity"] = get_velocity_weight(
# # # # # # # # # # # # # # #             epoch, args.vel_warmup_epochs, args.vel_w_start, args.vel_w_end)
# # # # # # # # # # # # # # #         epoch_weights["recurv"]   = get_recurv_weight(
# # # # # # # # # # # # # # #             epoch, args.recurv_warmup_epochs, args.recurv_w_start, args.recurv_w_end)
# # # # # # # # # # # # # # #         _losses_mod.WEIGHTS.update(epoch_weights)
# # # # # # # # # # # # # # #         if hasattr(model, 'weights'):
# # # # # # # # # # # # # # #             model.weights = epoch_weights

# # # # # # # # # # # # # # #         current_clip = get_grad_clip(epoch, warmup_epochs=20,
# # # # # # # # # # # # # # #                                       clip_start=args.grad_clip, clip_end=1.0)

# # # # # # # # # # # # # # #         # LR warm restarts
# # # # # # # # # # # # # # #         if epoch == 30 and not _lr_ep30_done:
# # # # # # # # # # # # # # #             _lr_ep30_done = True
# # # # # # # # # # # # # # #             scheduler = get_cosine_schedule_with_warmup(
# # # # # # # # # # # # # # #                 optimizer, steps_per_epoch,
# # # # # # # # # # # # # # #                 steps_per_epoch * (args.num_epochs - 30), min_lr=5e-6)
# # # # # # # # # # # # # # #             saver.reset_counters("LR warm restart at epoch 30")
# # # # # # # # # # # # # # #             print(f"  ↺  Warm Restart LR at epoch 30")

# # # # # # # # # # # # # # #         if epoch == 60 and not _lr_ep60_done:
# # # # # # # # # # # # # # #             _lr_ep60_done = True
# # # # # # # # # # # # # # #             scheduler = get_cosine_schedule_with_warmup(
# # # # # # # # # # # # # # #                 optimizer, steps_per_epoch,
# # # # # # # # # # # # # # #                 steps_per_epoch * (args.num_epochs - 60), min_lr=1e-6)
# # # # # # # # # # # # # # #             saver.reset_counters("LR warm restart at epoch 60")
# # # # # # # # # # # # # # #             print(f"  ↺  Warm Restart LR at epoch 60")

# # # # # # # # # # # # # # #         # ── Training loop ─────────────────────────────────────────────────────
# # # # # # # # # # # # # # #         model.train()
# # # # # # # # # # # # # # #         sum_loss      = 0.0
# # # # # # # # # # # # # # #         t0            = time.perf_counter()
# # # # # # # # # # # # # # #         optimizer.zero_grad()
# # # # # # # # # # # # # # #         recurv_ratio_buf = []

# # # # # # # # # # # # # # #         for i, batch in enumerate(train_loader):
# # # # # # # # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # # # # # # # #             if epoch == 0 and i == 0:
# # # # # # # # # # # # # # #                 _check_gph500(bl, train_dataset)
# # # # # # # # # # # # # # #                 _check_uv500(bl)

# # # # # # # # # # # # # # #             with autocast(device_type='cuda', enabled=args.use_amp):
# # # # # # # # # # # # # # #                 # FIX-T23-1: pass step_weight_alpha to model
# # # # # # # # # # # # # # #                 bd = model.get_loss_breakdown(bl, step_weight_alpha=step_alpha)

# # # # # # # # # # # # # # #             loss_to_backpass = bd["total"] / max(args.grad_accum, 1)
# # # # # # # # # # # # # # #             scaler.scale(loss_to_backpass).backward()

# # # # # # # # # # # # # # #             if ((i + 1) % args.grad_accum == 0
# # # # # # # # # # # # # # #                     or (i + 1) == len(train_loader)):
# # # # # # # # # # # # # # #                 scaler.unscale_(optimizer)
# # # # # # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_(
# # # # # # # # # # # # # # #                     model.parameters(), current_clip)
# # # # # # # # # # # # # # #                 scaler.step(optimizer)
# # # # # # # # # # # # # # #                 scaler.update()
# # # # # # # # # # # # # # #                 scheduler.step()
# # # # # # # # # # # # # # #                 optimizer.zero_grad()

# # # # # # # # # # # # # # #             sum_loss += bd["total"].item()
# # # # # # # # # # # # # # #             if "recurv_ratio" in bd:
# # # # # # # # # # # # # # #                 recurv_ratio_buf.append(bd["recurv_ratio"])

# # # # # # # # # # # # # # #             if i % 20 == 0:
# # # # # # # # # # # # # # #                 lr       = optimizer.param_groups[0]["lr"]
# # # # # # # # # # # # # # #                 rr       = bd.get("recurv_ratio", 0.0)
# # # # # # # # # # # # # # #                 elapsed  = time.perf_counter() - t0
# # # # # # # # # # # # # # #                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # # # # # # # # # # # # #                       f"  loss={bd['total'].item():.3f}"
# # # # # # # # # # # # # # #                       f"  fm={bd.get('fm',0):.2f}"
# # # # # # # # # # # # # # #                       f"  vel={bd.get('velocity',0):.4f}"
# # # # # # # # # # # # # # #                       f"  pinn={bd.get('pinn', 0):.4f}"  # FIX-T23-8: more precision
# # # # # # # # # # # # # # #                       f"  recurv={bd.get('recurv',0):.3f}"
# # # # # # # # # # # # # # #                       f"  rr={rr:.2f}"
# # # # # # # # # # # # # # #                       f"  pinn_w={epoch_weights['pinn']:.4f}"
# # # # # # # # # # # # # # #                       f"  alpha={step_alpha:.2f}"
# # # # # # # # # # # # # # #                       f"  clip={current_clip:.1f}"
# # # # # # # # # # # # # # #                       f"  ens={current_ens}"
# # # # # # # # # # # # # # #                       f"  lr={lr:.2e}  t={elapsed:.0f}s")

# # # # # # # # # # # # # # #         ep_s    = time.perf_counter() - t0
# # # # # # # # # # # # # # #         epoch_times.append(ep_s)
# # # # # # # # # # # # # # #         avg_t   = sum_loss / len(train_loader)
# # # # # # # # # # # # # # #         mean_rr = float(np.mean(recurv_ratio_buf)) if recurv_ratio_buf else 0.0

# # # # # # # # # # # # # # #         # ── Val loss ───────────────────────────────────────────────────────────
# # # # # # # # # # # # # # #         # if epoch % args.val_freq == 0:
# # # # # # # # # # # # # # #         #     model.eval()
# # # # # # # # # # # # # # #         #     val_loss = 0.0
# # # # # # # # # # # # # # #         #     t_val    = time.perf_counter()
# # # # # # # # # # # # # # #         #     with torch.no_grad():
# # # # # # # # # # # # # # #         #         for batch in val_loader:
# # # # # # # # # # # # # # #         #             bl_v = move(list(batch), device)
# # # # # # # # # # # # # # #         #             with autocast(device_type='cuda', enabled=args.use_amp):
# # # # # # # # # # # # # # #         #                 val_loss += model.get_loss(bl_v).item()
# # # # # # # # # # # # # # #         #     last_val_loss = val_loss / len(val_loader)
# # # # # # # # # # # # # # #         #     t_val_s = time.perf_counter() - t_val
# # # # # # # # # # # # # # #         #     saver.update_val_loss(last_val_loss, model, args.output_dir,
# # # # # # # # # # # # # # #         #                            epoch, optimizer, avg_t)
# # # # # # # # # # # # # # #         #     print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
# # # # # # # # # # # # # # #         #           f"  rr={mean_rr:.2f}"
# # # # # # # # # # # # # # #         #           f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
# # # # # # # # # # # # # # #         #           f"  ens={current_ens}  alpha={step_alpha:.2f}"
# # # # # # # # # # # # # # #         #           f"  recurv_w={epoch_weights['recurv']:.2f}")
# # # # # # # # # # # # # # #         # else:
# # # # # # # # # # # # # # #         #     print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
# # # # # # # # # # # # # # #         #           f"  val={last_val_loss:.3f}(cached)"
# # # # # # # # # # # # # # #         #           f"  rr={mean_rr:.2f}  t={ep_s:.0f}s")

# # # # # # # # # # # # # # #         # ── Val loss (Tính mỗi epoch, không cached) ──────────────────────────
# # # # # # # # # # # # # # #         model.eval()
# # # # # # # # # # # # # # #         val_loss = 0.0
# # # # # # # # # # # # # # #         t_val    = time.perf_counter()
# # # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # # #             for batch in val_loader:
# # # # # # # # # # # # # # #                 bl_v = move(list(batch), device)
# # # # # # # # # # # # # # #                 with autocast(device_type='cuda', enabled=args.use_amp):
# # # # # # # # # # # # # # #                     val_loss += model.get_loss(bl_v).item()
        
# # # # # # # # # # # # # # #         last_val_loss = val_loss / len(val_loader)
# # # # # # # # # # # # # # #         t_val_s = time.perf_counter() - t_val
        
# # # # # # # # # # # # # # #         saver.update_val_loss(last_val_loss, model, args.output_dir,
# # # # # # # # # # # # # # #                                epoch, optimizer, avg_t)
        
# # # # # # # # # # # # # # #         print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
# # # # # # # # # # # # # # #               f"  rr={mean_rr:.2f}"
# # # # # # # # # # # # # # #               f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
# # # # # # # # # # # # # # #               f"  ens={current_ens}  alpha={step_alpha:.2f}"
# # # # # # # # # # # # # # #               f"  recurv_w={epoch_weights['recurv']:.2f}")

# # # # # # # # # # # # # # #         # ── Fast ADE (subset, monitor only) ───────────────────────────────────
# # # # # # # # # # # # # # #         t_ade  = time.perf_counter()
# # # # # # # # # # # # # # #         m_fast = evaluate_fast(model, val_subset_loader, device,
# # # # # # # # # # # # # # #                                ode_train, args.pred_len, effective_fast_ens)
# # # # # # # # # # # # # # #         t_ade_s = time.perf_counter() - t_ade

# # # # # # # # # # # # # # #         spread_72h    = m_fast.get("spread_72h_km", 0.0)
# # # # # # # # # # # # # # #         active_steps  = m_fast.get("active_steps", args.pred_len)
# # # # # # # # # # # # # # #         collapse_warn = "  ⚠️ COLLAPSE!" if spread_72h < 10.0 else ""
# # # # # # # # # # # # # # #         spread_warn   = "  ⚠️ SPREAD HIGH!" if spread_72h > 600.0 else ""

# # # # # # # # # # # # # # #         print(f"  [FAST-ADE ep{epoch} {t_ade_s:.0f}s]"
# # # # # # # # # # # # # # #               f"  ADE={m_fast['ADE']:.1f} km  FDE={m_fast['FDE']:.1f} km"
# # # # # # # # # # # # # # #               f"  12h={m_fast.get('12h', float('nan')):.0f}"
# # # # # # # # # # # # # # #               f"  24h={m_fast.get('24h', float('nan')):.0f}"
# # # # # # # # # # # # # # #               f"  72h={m_fast.get('72h', float('nan')):.0f} km"
# # # # # # # # # # # # # # #               f"  spread={spread_72h:.1f} km"
# # # # # # # # # # # # # # #               f"  active_steps={active_steps}/{args.pred_len}"
# # # # # # # # # # # # # # #               f"  (subset, monitor only)"
# # # # # # # # # # # # # # #               f"{collapse_warn}{spread_warn}")

# # # # # # # # # # # # # # #         # FIX-T23-2: Subset ADE chỉ để log, KHÔNG dùng cho best model
# # # # # # # # # # # # # # #         saver.log_subset_ade(m_fast["ADE"], epoch)

# # # # # # # # # # # # # # #         # ── Full val ADE (mỗi val_ade_freq epoch) → là criteria chính ────────
# # # # # # # # # # # # # # #         if epoch % args.val_ade_freq == 0:
# # # # # # # # # # # # # # #             try:
# # # # # # # # # # # # # # #                 r_full = evaluate_full_val_ade(
# # # # # # # # # # # # # # #                     model, val_loader, device,
# # # # # # # # # # # # # # #                     ode_steps     = ode_train,
# # # # # # # # # # # # # # #                     pred_len      = args.pred_len,
# # # # # # # # # # # # # # #                     fast_ensemble = effective_fast_ens,
# # # # # # # # # # # # # # #                     metrics_csv   = metrics_csv,
# # # # # # # # # # # # # # #                     epoch         = epoch,
# # # # # # # # # # # # # # #                     tag           = f"val_full_ep{epoch:03d}",
# # # # # # # # # # # # # # #                 )
# # # # # # # # # # # # # # #                 full_ade = r_full.get("ADE", float("inf"))

# # # # # # # # # # # # # # #                 # FIX-T23-2: CHỈ đây mới trigger best model save và patience
# # # # # # # # # # # # # # #                 saver.update_ade_full_val(
# # # # # # # # # # # # # # #                     full_ade, model, args.output_dir, epoch,
# # # # # # # # # # # # # # #                     optimizer, avg_t, last_val_loss,
# # # # # # # # # # # # # # #                     min_epochs=args.min_epochs)
# # # # # # # # # # # # # # #             except Exception as e:
# # # # # # # # # # # # # # #                 print(f"  ⚠  Full val ADE failed: {e}")
# # # # # # # # # # # # # # #                 import traceback; traceback.print_exc()

# # # # # # # # # # # # # # #         # ── Full eval (4-tier) ────────────────────────────────────────────────
# # # # # # # # # # # # # # #         if epoch % args.full_eval_freq == 0 and epoch > 0:
# # # # # # # # # # # # # # #             print(f"  [Full eval epoch {epoch}, ode_steps={ode_val}]")
# # # # # # # # # # # # # # #             try:
# # # # # # # # # # # # # # #                 dm, _, _, _ = evaluate_full(
# # # # # # # # # # # # # # #                     model, val_loader, device,
# # # # # # # # # # # # # # #                     ode_val, args.pred_len, args.val_ensemble,
# # # # # # # # # # # # # # #                     metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}")
# # # # # # # # # # # # # # #                 print(dm.summary())
# # # # # # # # # # # # # # #             except Exception as e:
# # # # # # # # # # # # # # #                 print(f"  ⚠  full_eval failed at epoch {epoch}: {e}")
# # # # # # # # # # # # # # #                 import traceback; traceback.print_exc()

# # # # # # # # # # # # # # #         if (epoch + 1) % args.save_interval == 0:
# # # # # # # # # # # # # # #             torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
# # # # # # # # # # # # # # #                        os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))

# # # # # # # # # # # # # # #         if saver.early_stop:
# # # # # # # # # # # # # # #             print(f"  Early stopping @ epoch {epoch}")
# # # # # # # # # # # # # # #             break

# # # # # # # # # # # # # # #         if epoch % 5 == 4:
# # # # # # # # # # # # # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # # # # # # # # # # # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # # # # # # # # # # # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # # # # # # # # # # # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
# # # # # # # # # # # # # # #                   f"  (avg {avg_ep:.0f}s/epoch)")

# # # # # # # # # # # # # # #     # Restore final weights
# # # # # # # # # # # # # # #     _losses_mod.WEIGHTS["pinn"]     = args.pinn_w_end
# # # # # # # # # # # # # # #     _losses_mod.WEIGHTS["velocity"] = args.vel_w_end
# # # # # # # # # # # # # # #     _losses_mod.WEIGHTS["recurv"]   = args.recurv_w_end

# # # # # # # # # # # # # # #     total_train_h = (time.perf_counter() - train_start) / 3600

# # # # # # # # # # # # # # #     # ── Final test eval ───────────────────────────────────────────────────────
# # # # # # # # # # # # # # #     print(f"\n{'='*68}  FINAL TEST (ode_steps={ode_test})")
# # # # # # # # # # # # # # #     all_results = []

# # # # # # # # # # # # # # #     if test_loader:
# # # # # # # # # # # # # # #         best_path = os.path.join(args.output_dir, "best_model.pth")
# # # # # # # # # # # # # # #         if not os.path.exists(best_path):
# # # # # # # # # # # # # # #             best_path = os.path.join(args.output_dir, "best_model_valloss.pth")
# # # # # # # # # # # # # # #         if os.path.exists(best_path):
# # # # # # # # # # # # # # #             ck = torch.load(best_path, map_location=device)
# # # # # # # # # # # # # # #             try:
# # # # # # # # # # # # # # #                 model.load_state_dict(ck["model_state_dict"])
# # # # # # # # # # # # # # #             except Exception:
# # # # # # # # # # # # # # #                 model.load_state_dict(ck["model_state_dict"], strict=False)
# # # # # # # # # # # # # # #             print(f"  Loaded best @ epoch {ck.get('epoch','?')}"
# # # # # # # # # # # # # # #                   f"  ADE={ck.get('val_ade_km','?')}")

# # # # # # # # # # # # # # #         final_ens = max(args.val_ensemble, 50)
# # # # # # # # # # # # # # #         dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
# # # # # # # # # # # # # # #             model, test_loader, device,
# # # # # # # # # # # # # # #             ode_test, args.pred_len, final_ens,
# # # # # # # # # # # # # # #             metrics_csv=metrics_csv, tag="test_final",
# # # # # # # # # # # # # # #             predict_csv=predict_csv)
# # # # # # # # # # # # # # #         print(dm_test.summary())

# # # # # # # # # # # # # # #         all_results.append(ModelResult(
# # # # # # # # # # # # # # #             model_name   = "FM+PINN-v23",
# # # # # # # # # # # # # # #             split        = "test",
# # # # # # # # # # # # # # #             ADE          = dm_test.ade,
# # # # # # # # # # # # # # #             FDE          = dm_test.fde,
# # # # # # # # # # # # # # #             ADE_str      = dm_test.ade_str,
# # # # # # # # # # # # # # #             ADE_rec      = dm_test.ade_rec,
# # # # # # # # # # # # # # #             delta_rec    = dm_test.pr,
# # # # # # # # # # # # # # #             CRPS_mean    = dm_test.crps_mean,
# # # # # # # # # # # # # # #             CRPS_72h     = dm_test.crps_72h,
# # # # # # # # # # # # # # #             SSR          = dm_test.ssr_mean,
# # # # # # # # # # # # # # #             TSS_72h      = dm_test.tss_72h,
# # # # # # # # # # # # # # #             OYR          = dm_test.oyr_mean,
# # # # # # # # # # # # # # #             DTW          = dm_test.dtw_mean,
# # # # # # # # # # # # # # #             ATE_abs      = dm_test.ate_abs_mean,
# # # # # # # # # # # # # # #             CTE_abs      = dm_test.cte_abs_mean,
# # # # # # # # # # # # # # #             n_total      = dm_test.n_total,
# # # # # # # # # # # # # # #             n_recurv     = dm_test.n_rec,
# # # # # # # # # # # # # # #             train_time_h = total_train_h,
# # # # # # # # # # # # # # #             params_M     = n_params / 1e6,
# # # # # # # # # # # # # # #         ))

# # # # # # # # # # # # # # #         _, cliper_errs = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
# # # # # # # # # # # # # # #         persist_errs   = persistence_errors(obs_seqs, gt_seqs, args.pred_len)
# # # # # # # # # # # # # # #         fmpinn_per_seq = np.array([seq_ade_km(np.array(pp), np.array(g))
# # # # # # # # # # # # # # #                                     for pp, g in zip(pred_seqs, gt_seqs)])

# # # # # # # # # # # # # # #         np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
# # # # # # # # # # # # # # #         np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
# # # # # # # # # # # # # # #         np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))

# # # # # # # # # # # # # # #         lstm_per_seq      = _load_baseline_errors(args.lstm_errors_npy, "LSTM")
# # # # # # # # # # # # # # #         diffusion_per_seq = _load_baseline_errors(args.diffusion_errors_npy, "Diffusion")

# # # # # # # # # # # # # # #         stat_rows = [
# # # # # # # # # # # # # # #             paired_tests(fmpinn_per_seq, cliper_errs.mean(1),  "FM+PINN vs CLIPER", 5),
# # # # # # # # # # # # # # #             paired_tests(fmpinn_per_seq, persist_errs.mean(1), "FM+PINN vs Persist", 5),
# # # # # # # # # # # # # # #         ]
# # # # # # # # # # # # # # #         if lstm_per_seq is not None:
# # # # # # # # # # # # # # #             stat_rows.append(
# # # # # # # # # # # # # # #                 paired_tests(fmpinn_per_seq, lstm_per_seq, "FM+PINN vs LSTM", 5))
# # # # # # # # # # # # # # #         if diffusion_per_seq is not None:
# # # # # # # # # # # # # # #             stat_rows.append(
# # # # # # # # # # # # # # #                 paired_tests(fmpinn_per_seq, diffusion_per_seq, "FM+PINN vs Diffusion", 5))

# # # # # # # # # # # # # # #         compute_rows = DEFAULT_COMPUTE
# # # # # # # # # # # # # # #         try:
# # # # # # # # # # # # # # #             sb = next(iter(test_loader))
# # # # # # # # # # # # # # #             sb = move(list(sb), device)
# # # # # # # # # # # # # # #             from utils.evaluation_tables import profile_model_components
# # # # # # # # # # # # # # #             compute_rows = profile_model_components(model, sb, device)
# # # # # # # # # # # # # # #         except Exception as e:
# # # # # # # # # # # # # # #             print(f"  Profiling skipped: {e}")

# # # # # # # # # # # # # # #         export_all_tables(
# # # # # # # # # # # # # # #             results=all_results, ablation_rows=DEFAULT_ABLATION,
# # # # # # # # # # # # # # #             stat_rows=stat_rows, pinn_sens_rows=DEFAULT_PINN_SENSITIVITY,
# # # # # # # # # # # # # # #             compute_rows=compute_rows, out_dir=tables_dir)

# # # # # # # # # # # # # # #         with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
# # # # # # # # # # # # # # #             fh.write(dm_test.summary())
# # # # # # # # # # # # # # #             fh.write(f"\n\nmodel_version         : FM+PINN v23\n")
# # # # # # # # # # # # # # #             fh.write(f"sigma_min             : {args.sigma_min}\n")
# # # # # # # # # # # # # # #             fh.write(f"ctx_noise_scale       : {args.ctx_noise_scale}\n")
# # # # # # # # # # # # # # #             fh.write(f"initial_sample_sigma  : {args.initial_sample_sigma}\n")
# # # # # # # # # # # # # # #             fh.write(f"ode_steps_test        : {ode_test}\n")
# # # # # # # # # # # # # # #             fh.write(f"eval_ensemble         : {final_ens}\n")
# # # # # # # # # # # # # # #             fh.write(f"train_time_h          : {total_train_h:.2f}\n")
# # # # # # # # # # # # # # #             fh.write(f"n_params_M            : {n_params/1e6:.2f}\n")

# # # # # # # # # # # # # # #     avg_ep = sum(epoch_times)/len(epoch_times) if epoch_times else 0
# # # # # # # # # # # # # # #     print(f"\n  Best full-val ADE  : {saver.best_ade:.1f} km")
# # # # # # # # # # # # # # #     print(f"  Best val loss      : {saver.best_val_loss:.4f}")
# # # # # # # # # # # # # # #     print(f"  Avg epoch time     : {avg_ep:.0f}s")
# # # # # # # # # # # # # # #     print(f"  Total training     : {total_train_h:.2f}h")
# # # # # # # # # # # # # # #     print(f"  Tables dir         : {tables_dir}")
# # # # # # # # # # # # # # #     print("=" * 68)


# # # # # # # # # # # # # # # # ── GPH500 verification ───────────────────────────────────────────────────────

# # # # # # # # # # # # # # # def _check_gph500(bl, train_dataset):
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     FIX-T23-9: Verify GPH500 range (27-90 raw dam từ CSV).
# # # # # # # # # # # # # # #     Mean ~33 là đúng. Mean ≈ 0 hoặc ≈ -0.06 là sai (pre-normalized chưa fix).
# # # # # # # # # # # # # # #     """
# # # # # # # # # # # # # # #     env_data = bl[13]
# # # # # # # # # # # # # # #     if env_data is None or "gph500_mean" not in env_data:
# # # # # # # # # # # # # # #         print("  ⚠️  GPH500 key not found in env_data")
# # # # # # # # # # # # # # #         return

# # # # # # # # # # # # # # #     gph_val = env_data["gph500_mean"]
# # # # # # # # # # # # # # #     n_zero  = (gph_val == 0).sum().item()
# # # # # # # # # # # # # # #     n_total = gph_val.numel()
# # # # # # # # # # # # # # #     zero_pct = 100.0 * n_zero / max(n_total, 1)
# # # # # # # # # # # # # # #     gph_mean = gph_val.mean().item()

# # # # # # # # # # # # # # #     if abs(gph_mean) < 1.0 and zero_pct > 50.0:
# # # # # # # # # # # # # # #         print("\n" + "!" * 60)
# # # # # # # # # # # # # # #         print("  ⚠️  GPH500 mean ≈ 0 → Data not loading correctly from CSV")
# # # # # # # # # # # # # # #         print(f"     mean={gph_mean:.4f}, zero={zero_pct:.1f}%")
# # # # # # # # # # # # # # #         print("     Check FIX-DATA-18: env_gph500_mean should be raw dam (27-90)")
# # # # # # # # # # # # # # #         print("!" * 60 + "\n")
# # # # # # # # # # # # # # #     elif 25.0 < gph_mean < 95.0:
# # # # # # # # # # # # # # #         print(f"  ✅ GPH500 OK (mean={gph_mean:.2f} dam, zero={zero_pct:.1f}%)")
# # # # # # # # # # # # # # #     elif -30.0 < gph_mean < 5.0:
# # # # # # # # # # # # # # #         # Pre-normalized: mean ≈ -0.06 is the old npy format with _n keys
# # # # # # # # # # # # # # #         print(f"  ℹ️  GPH500 pre-normalized detected (mean={gph_mean:.4f})")
# # # # # # # # # # # # # # #         print(f"     This is acceptable if loading from .npy with _n keys")
# # # # # # # # # # # # # # #         print(f"     zero={zero_pct:.1f}%")
# # # # # # # # # # # # # # #     else:
# # # # # # # # # # # # # # #         print(f"  ⚠️  GPH500 unexpected range (mean={gph_mean:.4f}, zero={zero_pct:.1f}%)")


# # # # # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # # # # #     args = get_args()
# # # # # # # # # # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # # # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # # # # # # # #         torch.cuda.manual_seed_all(42)
# # # # # # # # # # # # # # #     main(args)

# # # # # # # # # # # # # # """
# # # # # # # # # # # # # # scripts/train_flowmatching.py  ── v24
# # # # # # # # # # # # # # ======================================
# # # # # # # # # # # # # # FIXES vs v23:

# # # # # # # # # # # # # #   FIX-T24-A  [TWO-PHASE TRAINING]
# # # # # # # # # # # # # #              Phase 1 (ep 0-30): Đóng băng FNO3D + enc_1d + env_enc.
# # # # # # # # # # # # # #                Chỉ train ShortRangeHead + ctx_fc/transformer. LR=5e-4.
# # # # # # # # # # # # # #                Mục tiêu: ShortRangeHead học motion pattern nhanh.
# # # # # # # # # # # # # #              Phase 2 (ep 30+): Unfreeze toàn bộ. LR=2e-4.
# # # # # # # # # # # # # #                Mục tiêu: Fine-tune end-to-end với đầy đủ loss.

# # # # # # # # # # # # # #   FIX-T24-B  [PRIMARY METRIC: 12h ADE]
# # # # # # # # # # # # # #              BestModelSaver giờ track best_12h_ade thay vì overall ADE.
# # # # # # # # # # # # # #              Early stopping dựa trên 12h ADE (tol=3 km).
# # # # # # # # # # # # # #              Overall ADE vẫn log nhưng không quyết định best model.

# # # # # # # # # # # # # #   FIX-T24-C  [SHORT-RANGE LOSS WEIGHT SCHEDULE]
# # # # # # # # # # # # # #              Ep 0-30 : short_range weight = 8.0 (focus mạnh vào 12h/24h)
# # # # # # # # # # # # # #              Ep 30-60: short_range weight = 5.0 (cân bằng)
# # # # # # # # # # # # # #              Ep 60+  : short_range weight = 3.0 (duy trì)

# # # # # # # # # # # # # #   FIX-T24-D  [SPREAD CONTROL ĐẦU TRAINING]
# # # # # # # # # # # # # #              initial_sample_sigma=0.03, ctx_noise_scale=0.002
# # # # # # # # # # # # # #              Tăng spread_weight lên 1.2 trong 30 epoch đầu để ép
# # # # # # # # # # # # # #              ensemble không bùng nổ ngay từ đầu.

# # # # # # # # # # # # # #   FIX-T24-E  [PINN WARMUP DÀI HƠN]
# # # # # # # # # # # # # #              pinn_warmup_epochs=80 (từ 50). PINN chỉ bắt đầu có ý nghĩa
# # # # # # # # # # # # # #              sau khi FM và ShortRangeHead đã học được pattern.

# # # # # # # # # # # # # #   FIX-T24-F  [EVALUATE SHORT-RANGE SEPARATELY]
# # # # # # # # # # # # # #              evaluate_fast() log riêng 12h và 24h ADE từ ShortRangeHead.
# # # # # # # # # # # # # #              evaluate_full_val_ade() cũng log sr_ade_12h, sr_ade_24h.

# # # # # # # # # # # # # # Kept from v23:
# # # # # # # # # # # # # #   FIX-T23-1  Curriculum removed
# # # # # # # # # # # # # #   FIX-T23-2  Full val ADE là criteria lưu best model (v24: 12h ADE)
# # # # # # # # # # # # # #   FIX-T23-6  patience=15
# # # # # # # # # # # # # #   FIX-T23-7  step_weight_alpha schedule
# # # # # # # # # # # # # # """
# # # # # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # # # # import sys
# # # # # # # # # # # # # # import os
# # # # # # # # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # # # # # # # import argparse
# # # # # # # # # # # # # # import time
# # # # # # # # # # # # # # import math
# # # # # # # # # # # # # # import random
# # # # # # # # # # # # # # import copy

# # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # import torch
# # # # # # # # # # # # # # import torch.optim as optim
# # # # # # # # # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # # # # # # # from Model.data.loader_training import data_loader
# # # # # # # # # # # # # # from Model.flow_matching_model import TCFlowMatching, ShortRangeHead
# # # # # # # # # # # # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # # # # # # # # from Model.losses import WEIGHTS as _BASE_WEIGHTS
# # # # # # # # # # # # # # from utils.metrics import (
# # # # # # # # # # # # # #     TCEvaluator, StepErrorAccumulator,
# # # # # # # # # # # # # #     save_metrics_csv, haversine_km_torch,
# # # # # # # # # # # # # #     denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
# # # # # # # # # # # # # #     cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
# # # # # # # # # # # # # #     brier_skill_score, cliper_errors, persistence_errors,
# # # # # # # # # # # # # # )
# # # # # # # # # # # # # # from utils.evaluation_tables import (
# # # # # # # # # # # # # #     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
# # # # # # # # # # # # # #     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
# # # # # # # # # # # # # #     DEFAULT_COMPUTE, paired_tests,
# # # # # # # # # # # # # # )
# # # # # # # # # # # # # # from scripts.statistical_tests import run_all_tests


# # # # # # # # # # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # def haversine_km_np_local(pred_deg: np.ndarray,
# # # # # # # # # # # # # #                            gt_deg: np.ndarray) -> np.ndarray:
# # # # # # # # # # # # # #     pred_deg = np.atleast_2d(pred_deg)
# # # # # # # # # # # # # #     gt_deg   = np.atleast_2d(gt_deg)
# # # # # # # # # # # # # #     R = 6371.0
# # # # # # # # # # # # # #     lon1, lat1 = np.radians(pred_deg[:, 0]), np.radians(pred_deg[:, 1])
# # # # # # # # # # # # # #     lon2, lat2 = np.radians(gt_deg[:, 0]),   np.radians(gt_deg[:, 1])
# # # # # # # # # # # # # #     dlon = lon2 - lon1;  dlat = lat2 - lat1
# # # # # # # # # # # # # #     a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
# # # # # # # # # # # # # #     return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# # # # # # # # # # # # # # def seq_ade_km(pred_norm: np.ndarray, gt_norm: np.ndarray) -> float:
# # # # # # # # # # # # # #     return float(haversine_km_np_local(denorm_deg_np(pred_norm),
# # # # # # # # # # # # # #                                        denorm_deg_np(gt_norm)).mean())


# # # # # # # # # # # # # # def move(batch, device):
# # # # # # # # # # # # # #     out = list(batch)
# # # # # # # # # # # # # #     for i, x in enumerate(out):
# # # # # # # # # # # # # #         if torch.is_tensor(x):
# # # # # # # # # # # # # #             out[i] = x.to(device)
# # # # # # # # # # # # # #         elif isinstance(x, dict):
# # # # # # # # # # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # # # # # # # #                       for k, v in x.items()}
# # # # # # # # # # # # # #     return out


# # # # # # # # # # # # # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # # # # # # # # # # # # #                             collate_fn, num_workers):
# # # # # # # # # # # # # #     n   = len(val_dataset)
# # # # # # # # # # # # # #     rng = random.Random(42)
# # # # # # # # # # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # # # # # # # # # #     return DataLoader(Subset(val_dataset, idx),
# # # # # # # # # # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # # # # # # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # # # # # # # # # # ── Phase control  (FIX-T24-A) ───────────────────────────────────────────────

# # # # # # # # # # # # # # def get_phase(epoch: int, phase2_start: int = 30) -> int:
# # # # # # # # # # # # # #     """1 = short-range focus, 2 = full model."""
# # # # # # # # # # # # # #     return 1 if epoch < phase2_start else 2


# # # # # # # # # # # # # # def freeze_backbone(model: TCFlowMatching) -> None:
# # # # # # # # # # # # # #     """Phase 1: freeze FNO3D, enc_1d, env_enc."""
# # # # # # # # # # # # # #     for name, param in model.named_parameters():
# # # # # # # # # # # # # #         if any(k in name for k in ["spatial_enc", "enc_1d", "env_enc"]):
# # # # # # # # # # # # # #             param.requires_grad_(False)
# # # # # # # # # # # # # #     n_frozen = sum(not p.requires_grad for p in model.parameters())
# # # # # # # # # # # # # #     print(f"  [Phase-1] Frozen {n_frozen:,} params (backbone)")


# # # # # # # # # # # # # # def unfreeze_all(model: TCFlowMatching) -> None:
# # # # # # # # # # # # # #     """Phase 2: unfreeze everything."""
# # # # # # # # # # # # # #     for param in model.parameters():
# # # # # # # # # # # # # #         param.requires_grad_(True)
# # # # # # # # # # # # # #     print(f"  [Phase-2] Unfrozen all params")


# # # # # # # # # # # # # # # ── Weight schedules ──────────────────────────────────────────────────────────

# # # # # # # # # # # # # # # def get_short_range_weight(epoch: int) -> float:
# # # # # # # # # # # # # # #     """FIX-T24-C: short_range weight schedule."""
# # # # # # # # # # # # # # #     if epoch < 30:
# # # # # # # # # # # # # # #         return 8.0
# # # # # # # # # # # # # # #     elif epoch < 60:
# # # # # # # # # # # # # # #         return 5.0
# # # # # # # # # # # # # # #     else:
# # # # # # # # # # # # # # #         return 3.0
# # # # # # # # # # # # # # def get_short_range_weight(epoch):
# # # # # # # # # # # # # #     # FIX-T24-C giữ nguyên nhưng tăng mạnh hơn ở phase 1
# # # # # # # # # # # # # #     if epoch < 15:
# # # # # # # # # # # # # #         return 12.0   # ép ShortRangeHead học rất mạnh giai đoạn đầu
# # # # # # # # # # # # # #     elif epoch < 30:
# # # # # # # # # # # # # #         return 8.0
# # # # # # # # # # # # # #     elif epoch < 60:
# # # # # # # # # # # # # #         return 5.0
# # # # # # # # # # # # # #     else:
# # # # # # # # # # # # # #         return 3.0

# # # # # # # # # # # # # # def get_fm_weight(epoch):
# # # # # # # # # # # # # #     """FIX MỚI: FM loss nhỏ ở phase 1 để không cản ShortRangeHead."""
# # # # # # # # # # # # # #     if epoch < 30:
# # # # # # # # # # # # # #         return 0.5   # giảm FM weight trong phase 1
# # # # # # # # # # # # # #     return 2.0       # về bình thường ở phase 2

# # # # # # # # # # # # # # def get_spread_weight(epoch: int) -> float:
# # # # # # # # # # # # # #     """FIX-T24-D: higher spread penalty in first 30 epochs."""
# # # # # # # # # # # # # #     if epoch < 30:
# # # # # # # # # # # # # #         return 1.2
# # # # # # # # # # # # # #     return 0.8


# # # # # # # # # # # # # # # def get_pinn_weight(epoch: int, warmup_epochs: int = 80,
# # # # # # # # # # # # # # #                     w_start: float = 0.001, w_end: float = 0.05) -> float:
# # # # # # # # # # # # # # #     """FIX-T24-E: longer PINN warmup (80 epochs)."""
# # # # # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # # # # #         return w_end
# # # # # # # # # # # # # # #     return w_start + (epoch / max(warmup_epochs - 1, 1)) * (w_end - w_start)

# # # # # # # # # # # # # # def get_pinn_weight(epoch, warmup_epochs=80,
# # # # # # # # # # # # # #                     w_start=0.001, w_end=0.05):
# # # # # # # # # # # # # #     # FIX: tăng nhanh hơn trong phase 2 (sau ep 30)
# # # # # # # # # # # # # #     if epoch < 30:
# # # # # # # # # # # # # #         # Phase 1: PINN rất nhỏ, focus ShortRangeHead
# # # # # # # # # # # # # #         return w_start
# # # # # # # # # # # # # #     elif epoch < warmup_epochs:
# # # # # # # # # # # # # #         # Phase 2: tăng tuyến tính từ 0.005 → w_end
# # # # # # # # # # # # # #         t = (epoch - 30) / max(warmup_epochs - 30, 1)
# # # # # # # # # # # # # #         return 0.005 + t * (w_end - 0.005)
# # # # # # # # # # # # # #     return w_end
    

# # # # # # # # # # # # # # def get_velocity_weight(epoch, warmup_epochs=20, w_start=0.5, w_end=1.5):
# # # # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # # # #         return w_end
# # # # # # # # # # # # # #     return w_start + (epoch / max(warmup_epochs - 1, 1)) * (w_end - w_start)


# # # # # # # # # # # # # # def get_recurv_weight(epoch, warmup_epochs=10, w_start=0.3, w_end=1.0):
# # # # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # # # #         return w_end
# # # # # # # # # # # # # #     return w_start + (epoch / max(warmup_epochs - 1, 1)) * (w_end - w_start)


# # # # # # # # # # # # # # def get_grad_clip(epoch, warmup_epochs=20, clip_start=2.0, clip_end=1.0):
# # # # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # # # #         return clip_end
# # # # # # # # # # # # # #     return clip_start - (epoch / max(warmup_epochs - 1, 1)) * (clip_start - clip_end)


# # # # # # # # # # # # # # def get_step_weight_alpha(epoch: int, decay_epochs: int = 60) -> float:
# # # # # # # # # # # # # #     if epoch >= decay_epochs:
# # # # # # # # # # # # # #         return 0.0
# # # # # # # # # # # # # #     return 1.0 - (epoch / decay_epochs)


# # # # # # # # # # # # # # def get_progressive_ens(epoch: int, n_train_ens: int = 6) -> int:
# # # # # # # # # # # # # #     if epoch < 20:
# # # # # # # # # # # # # #         return 1
# # # # # # # # # # # # # #     elif epoch < 50:
# # # # # # # # # # # # # #         return 2
# # # # # # # # # # # # # #     else:
# # # # # # # # # # # # # #         return n_train_ens


# # # # # # # # # # # # # # # ── Short-range ADE helper ────────────────────────────────────────────────────

# # # # # # # # # # # # # # @torch.no_grad()
# # # # # # # # # # # # # # def compute_sr_ade(model: TCFlowMatching,
# # # # # # # # # # # # # #                    batch_list: list,
# # # # # # # # # # # # # #                    device) -> tuple[float, float]:
# # # # # # # # # # # # # #     """
# # # # # # # # # # # # # #     FIX-T24-F: Compute 12h and 24h ADE from ShortRangeHead.
# # # # # # # # # # # # # #     Returns (ade_12h_km, ade_24h_km).
# # # # # # # # # # # # # #     """
# # # # # # # # # # # # # #     obs_t  = batch_list[0]
# # # # # # # # # # # # # #     traj_gt = batch_list[1]

# # # # # # # # # # # # # #     raw_ctx  = model.net._context(batch_list)
# # # # # # # # # # # # # #     sr_pred  = model.net.forward_short_range(obs_t, raw_ctx)  # [4, B, 2]

# # # # # # # # # # # # # #     # Denorm to 0.1° units for haversine
# # # # # # # # # # # # # #     pred_01 = sr_pred.clone()
# # # # # # # # # # # # # #     pred_01[..., 0] = sr_pred[..., 0] * 50.0 + 1800.0
# # # # # # # # # # # # # #     pred_01[..., 1] = sr_pred[..., 1] * 50.0

# # # # # # # # # # # # # #     gt_01 = traj_gt.clone()
# # # # # # # # # # # # # #     gt_01[..., 0] = traj_gt[..., 0] * 50.0 + 1800.0
# # # # # # # # # # # # # #     gt_01[..., 1] = traj_gt[..., 1] * 50.0

# # # # # # # # # # # # # #     # step 1 = 6h (idx 0), step 2 = 12h (idx 1), step 4 = 24h (idx 3)
# # # # # # # # # # # # # #     ade_12h = haversine_km_torch(pred_01[1], gt_01[1]).mean().item()
# # # # # # # # # # # # # #     ade_24h = haversine_km_torch(pred_01[3], gt_01[3]).mean().item() \
# # # # # # # # # # # # # #               if pred_01.shape[0] >= 4 else float("nan")
# # # # # # # # # # # # # #     return ade_12h, ade_24h


# # # # # # # # # # # # # # # ── evaluate_fast ─────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
# # # # # # # # # # # # # #     """
# # # # # # # # # # # # # #     FIX-T24-F: Log short-range 12h/24h ADE from ShortRangeHead separately.
# # # # # # # # # # # # # #     """
# # # # # # # # # # # # # #     model.eval()
# # # # # # # # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # # # # # # # #     t0  = time.perf_counter()
# # # # # # # # # # # # # #     n   = 0
# # # # # # # # # # # # # #     spread_per_step = []
# # # # # # # # # # # # # #     sr_12h_buf, sr_24h_buf = [], []

# # # # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # # # #         for batch in loader:
# # # # # # # # # # # # # #             bl = list(batch)
# # # # # # # # # # # # # #             for i, x in enumerate(bl):
# # # # # # # # # # # # # #                 if torch.is_tensor(x):
# # # # # # # # # # # # # #                     bl[i] = x.to(device)
# # # # # # # # # # # # # #                 elif isinstance(x, dict):
# # # # # # # # # # # # # #                     bl[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # # # # # # # #                               for k, v in x.items()}

# # # # # # # # # # # # # #             pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
# # # # # # # # # # # # # #                                               ddim_steps=ode_steps)
# # # # # # # # # # # # # #             T_active  = pred.shape[0]
# # # # # # # # # # # # # #             gt_sliced = bl[1][:T_active]
# # # # # # # # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
# # # # # # # # # # # # # #             acc.update(dist)

# # # # # # # # # # # # # #             # Short-range ADE from head
# # # # # # # # # # # # # #             a12, a24 = compute_sr_ade(model, bl, device)
# # # # # # # # # # # # # #             sr_12h_buf.append(a12)
# # # # # # # # # # # # # #             sr_24h_buf.append(a24)

# # # # # # # # # # # # # #             # Per-step spread
# # # # # # # # # # # # # #             step_spreads = []
# # # # # # # # # # # # # #             for t in range(all_trajs.shape[1]):
# # # # # # # # # # # # # #                 step_data = all_trajs[:, t, :, :]
# # # # # # # # # # # # # #                 std_lon = step_data[:, :, 0].std(0)
# # # # # # # # # # # # # #                 std_lat = step_data[:, :, 1].std(0)
# # # # # # # # # # # # # #                 spread = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# # # # # # # # # # # # # #                 step_spreads.append(spread)
# # # # # # # # # # # # # #             spread_per_step.append(step_spreads)
# # # # # # # # # # # # # #             n += 1

# # # # # # # # # # # # # #     r = acc.compute()
# # # # # # # # # # # # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)

# # # # # # # # # # # # # #     if spread_per_step:
# # # # # # # # # # # # # #         spreads = np.array(spread_per_step)
# # # # # # # # # # # # # #         r["spread_12h_km"] = float(spreads[:, 1].mean()) if spreads.shape[1] > 1 else 0.0
# # # # # # # # # # # # # #         r["spread_24h_km"] = float(spreads[:, 3].mean()) if spreads.shape[1] > 3 else 0.0
# # # # # # # # # # # # # #         r["spread_72h_km"] = float(spreads[:, -1].mean())

# # # # # # # # # # # # # #     # FIX-T24-F: short-range ADE from head
# # # # # # # # # # # # # #     r["sr_ade_12h"] = float(np.mean(sr_12h_buf)) if sr_12h_buf else float("nan")
# # # # # # # # # # # # # #     r["sr_ade_24h"] = float(np.mean(sr_24h_buf)) if sr_24h_buf else float("nan")

# # # # # # # # # # # # # #     return r


# # # # # # # # # # # # # # # ── evaluate_full_val_ade ─────────────────────────────────────────────────────

# # # # # # # # # # # # # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # # # # # # # # # # # # #                            fast_ensemble, metrics_csv, epoch, tag=""):
# # # # # # # # # # # # # #     """
# # # # # # # # # # # # # #     FIX-T24-B: Primary metric is now 12h ADE (from ShortRangeHead).
# # # # # # # # # # # # # #     Returns dict with 'ADE', '12h', '24h', 'sr_12h', 'sr_24h'.
# # # # # # # # # # # # # #     """
# # # # # # # # # # # # # #     model.eval()
# # # # # # # # # # # # # #     acc       = StepErrorAccumulator(pred_len)
# # # # # # # # # # # # # #     t0        = time.perf_counter()
# # # # # # # # # # # # # #     n_batch   = 0
# # # # # # # # # # # # # #     pinn_buf  = []
# # # # # # # # # # # # # #     sr_12h_buf, sr_24h_buf = [], []

# # # # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # # # #         for batch in val_loader:
# # # # # # # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # # # # # # #             pred, _, _ = model.sample(bl, num_ensemble=fast_ensemble,
# # # # # # # # # # # # # #                                        ddim_steps=ode_steps)
# # # # # # # # # # # # # #             T_pred = pred.shape[0]
# # # # # # # # # # # # # #             gt     = bl[1][:T_pred]
# # # # # # # # # # # # # #             dist   = haversine_km_torch(denorm_torch(pred), denorm_torch(gt))
# # # # # # # # # # # # # #             acc.update(dist)

# # # # # # # # # # # # # #             # Short-range from head
# # # # # # # # # # # # # #             a12, a24 = compute_sr_ade(model, bl, device)
# # # # # # # # # # # # # #             sr_12h_buf.append(a12)
# # # # # # # # # # # # # #             sr_24h_buf.append(a24)

# # # # # # # # # # # # # #             # PINN
# # # # # # # # # # # # # #             try:
# # # # # # # # # # # # # #                 from Model.losses import pinn_bve_loss
# # # # # # # # # # # # # #                 pred_deg = pred.clone()
# # # # # # # # # # # # # #                 pred_deg[..., 0] = (pred[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # # # # #                 pred_deg[..., 1] = (pred[..., 1] * 50.0) / 10.0
# # # # # # # # # # # # # #                 env_d = bl[13] if len(bl) > 13 else None
# # # # # # # # # # # # # #                 pinn_val = pinn_bve_loss(pred_deg, bl, env_data=env_d).item()
# # # # # # # # # # # # # #                 pinn_buf.append(pinn_val)
# # # # # # # # # # # # # #             except Exception:
# # # # # # # # # # # # # #                 pass

# # # # # # # # # # # # # #             n_batch += 1

# # # # # # # # # # # # # #     elapsed  = time.perf_counter() - t0
# # # # # # # # # # # # # #     r        = acc.compute()
# # # # # # # # # # # # # #     sr_12h   = float(np.mean(sr_12h_buf)) if sr_12h_buf else float("nan")
# # # # # # # # # # # # # #     sr_24h   = float(np.mean(sr_24h_buf)) if sr_24h_buf else float("nan")
# # # # # # # # # # # # # #     pinn_mean = f"{np.mean(pinn_buf):.3f}" if pinn_buf else "N/A"

# # # # # # # # # # # # # #     print(f"\n{'='*64}")
# # # # # # # # # # # # # #     print(f"  [FULL VAL ADE  ep={epoch}  {elapsed:.0f}s  {n_batch} batches]")
# # # # # # # # # # # # # #     print(f"  ADE (blend) = {r.get('ADE', float('nan')):.1f} km  "
# # # # # # # # # # # # # #           f"FDE = {r.get('FDE', float('nan')):.1f} km")
# # # # # # # # # # # # # #     print(f"  12h={r.get('12h', float('nan')):.0f}  "
# # # # # # # # # # # # # #           f"24h={r.get('24h', float('nan')):.0f}  "
# # # # # # # # # # # # # #           f"48h={r.get('48h', float('nan')):.0f}  "
# # # # # # # # # # # # # #           f"72h={r.get('72h', float('nan')):.0f} km")
# # # # # # # # # # # # # #     print(f"  ── ShortRangeHead ──")
# # # # # # # # # # # # # #     print(f"  SR-12h={sr_12h:.1f} km  SR-24h={sr_24h:.1f} km  "
# # # # # # # # # # # # # #           f"[TARGET: <50 / <100 km]")
# # # # # # # # # # # # # #     print(f"  PINN_mean={pinn_mean}  samples={r.get('n_samples',0)}  "
# # # # # # # # # # # # # #           f"ens={fast_ensemble}  steps={ode_steps}")
# # # # # # # # # # # # # #     print(f"{'='*64}\n")

# # # # # # # # # # # # # #     r["sr_12h"] = sr_12h
# # # # # # # # # # # # # #     r["sr_24h"] = sr_24h

# # # # # # # # # # # # # #     from datetime import datetime
# # # # # # # # # # # # # #     from utils.metrics import DatasetMetrics, save_metrics_csv as _save_csv
# # # # # # # # # # # # # #     dm = DatasetMetrics(
# # # # # # # # # # # # # #         ade      = r.get("ADE",  float("nan")),
# # # # # # # # # # # # # #         fde      = r.get("FDE",  float("nan")),
# # # # # # # # # # # # # #         ugde_12h = r.get("12h",  float("nan")),
# # # # # # # # # # # # # #         ugde_24h = r.get("24h",  float("nan")),
# # # # # # # # # # # # # #         ugde_48h = r.get("48h",  float("nan")),
# # # # # # # # # # # # # #         ugde_72h = r.get("72h",  float("nan")),
# # # # # # # # # # # # # #         n_total  = r.get("n_samples", 0),
# # # # # # # # # # # # # #         timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     _save_csv(dm, metrics_csv, tag=tag or f"val_full_ep{epoch:03d}")
# # # # # # # # # # # # # #     return r


# # # # # # # # # # # # # # def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
# # # # # # # # # # # # # #                   metrics_csv, tag="", predict_csv=""):
# # # # # # # # # # # # # #     """Full 4-tier evaluation (unchanged from v23)."""
# # # # # # # # # # # # # #     model.eval()
# # # # # # # # # # # # # #     cliper_step_errors = []
# # # # # # # # # # # # # #     ev = TCEvaluator(pred_len=pred_len, compute_dtw=True)
# # # # # # # # # # # # # #     obs_seqs_01 = []; gt_seqs_01 = []; pred_seqs_01 = []; ens_seqs_01 = []

# # # # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # # # #         for batch in loader:
# # # # # # # # # # # # # #             bl  = move(list(batch), device)
# # # # # # # # # # # # # #             gt  = bl[1];  obs = bl[0]
# # # # # # # # # # # # # #             pred_mean, _, all_trajs = model.sample(
# # # # # # # # # # # # # #                 bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
# # # # # # # # # # # # # #                 predict_csv=predict_csv if predict_csv else None)

# # # # # # # # # # # # # #             pd_np = denorm_torch(pred_mean).cpu().numpy()
# # # # # # # # # # # # # #             gd_np = denorm_torch(gt).cpu().numpy()
# # # # # # # # # # # # # #             od_np = denorm_torch(obs).cpu().numpy()
# # # # # # # # # # # # # #             ed_np = denorm_torch(all_trajs).cpu().numpy()

# # # # # # # # # # # # # #             B = pd_np.shape[1]
# # # # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # # # #                 ens_b = ed_np[:, :, b, :]
# # # # # # # # # # # # # #                 ev.update(pd_np[:, b, :], gd_np[:, b, :], pred_ens=ens_b)
# # # # # # # # # # # # # #                 obs_seqs_01.append(od_np[:, b, :])
# # # # # # # # # # # # # #                 gt_seqs_01.append(gd_np[:, b, :])
# # # # # # # # # # # # # #                 pred_seqs_01.append(pd_np[:, b, :])
# # # # # # # # # # # # # #                 ens_seqs_01.append(ens_b)

# # # # # # # # # # # # # #                 obs_b_norm = obs.cpu().numpy()[:, b, :]
# # # # # # # # # # # # # #                 cliper_errors_b = np.zeros(pred_len)
# # # # # # # # # # # # # #                 for h in range(pred_len):
# # # # # # # # # # # # # #                     pred_cliper_norm = cliper_forecast(obs_b_norm, h + 1)
# # # # # # # # # # # # # #                     pred_cliper_01   = denorm_np(pred_cliper_norm[np.newaxis])
# # # # # # # # # # # # # #                     gt_01            = gd_np[h, b, :][np.newaxis]
# # # # # # # # # # # # # #                     from utils.metrics import haversine_km_np
# # # # # # # # # # # # # #                     cliper_errors_b[h] = float(
# # # # # # # # # # # # # #                         haversine_km_np(pred_cliper_01, gt_01, unit_01deg=True)[0])
# # # # # # # # # # # # # #                 cliper_step_errors.append(cliper_errors_b)

# # # # # # # # # # # # # #     if cliper_step_errors:
# # # # # # # # # # # # # #         cliper_mat       = np.stack(cliper_step_errors)
# # # # # # # # # # # # # #         cliper_ugde_dict = {h: float(cliper_mat[:, s].mean())
# # # # # # # # # # # # # #                             for h, s in HORIZON_STEPS.items()
# # # # # # # # # # # # # #                             if s < cliper_mat.shape[1]}
# # # # # # # # # # # # # #         ev.cliper_ugde   = cliper_ugde_dict

# # # # # # # # # # # # # #     dm = ev.compute(tag=tag)

# # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # #         if LANDFALL_TARGETS and ens_seqs_01:
# # # # # # # # # # # # # #             bss_vals = []
# # # # # # # # # # # # # #             step_72  = HORIZON_STEPS.get(72, pred_len - 1)
# # # # # # # # # # # # # #             for tname, t_lon, t_lat in LANDFALL_TARGETS:
# # # # # # # # # # # # # #                 bv = brier_skill_score(
# # # # # # # # # # # # # #                     ens_seqs_01, gt_seqs_01, min(step_72, pred_len-1),
# # # # # # # # # # # # # #                     (t_lon * 10.0, t_lat * 10.0), LANDFALL_RADIUS_KM)
# # # # # # # # # # # # # #                 if not math.isnan(bv):
# # # # # # # # # # # # # #                     bss_vals.append(bv)
# # # # # # # # # # # # # #             if bss_vals:
# # # # # # # # # # # # # #                 dm.bss_mean = float(np.mean(bss_vals))
# # # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # # #         print(f"  ⚠  BSS failed: {e}")

# # # # # # # # # # # # # #     save_metrics_csv(dm, metrics_csv, tag=tag)
# # # # # # # # # # # # # #     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# # # # # # # # # # # # # # # ── BestModelSaver  (FIX-T24-B) ───────────────────────────────────────────────

# # # # # # # # # # # # # # class BestModelSaver:
# # # # # # # # # # # # # #     """
# # # # # # # # # # # # # #     FIX-T24-B: Primary metric = ShortRangeHead 12h ADE.
# # # # # # # # # # # # # #     Also track overall blend ADE separately.
# # # # # # # # # # # # # #     patience triggers only from 12h ADE stagnation.
# # # # # # # # # # # # # #     """

# # # # # # # # # # # # # #     def __init__(self, patience: int = 15, sr_tol: float = 3.0,
# # # # # # # # # # # # # #                  ade_tol: float = 5.0):
# # # # # # # # # # # # # #         self.patience      = patience
# # # # # # # # # # # # # #         self.sr_tol        = sr_tol       # km tolerance for 12h ADE
# # # # # # # # # # # # # #         self.ade_tol       = ade_tol      # km tolerance for overall ADE
# # # # # # # # # # # # # #         self.best_sr12h    = float("inf")
# # # # # # # # # # # # # #         self.best_ade      = float("inf")
# # # # # # # # # # # # # #         self.best_val_loss = float("inf")
# # # # # # # # # # # # # #         self.counter_12h   = 0
# # # # # # # # # # # # # #         self.counter_loss  = 0
# # # # # # # # # # # # # #         self.early_stop    = False

# # # # # # # # # # # # # #     def reset_counters(self, reason: str = "") -> None:
# # # # # # # # # # # # # #         self.counter_12h  = 0
# # # # # # # # # # # # # #         self.counter_loss = 0
# # # # # # # # # # # # # #         if reason:
# # # # # # # # # # # # # #             print(f"  [SAVER] Patience reset: {reason}")

# # # # # # # # # # # # # #     def update_val_loss(self, val_loss, model, out_dir, epoch, optimizer, tl):
# # # # # # # # # # # # # #         if val_loss < self.best_val_loss - 1e-4:
# # # # # # # # # # # # # #             self.best_val_loss = val_loss
# # # # # # # # # # # # # #             self.counter_loss  = 0
# # # # # # # # # # # # # #             torch.save(dict(
# # # # # # # # # # # # # #                 epoch=epoch, model_state_dict=model.state_dict(),
# # # # # # # # # # # # # #                 optimizer_state=optimizer.state_dict(),
# # # # # # # # # # # # # #                 train_loss=tl, val_loss=val_loss,
# # # # # # # # # # # # # #                 model_version="v24-valloss"),
# # # # # # # # # # # # # #                 os.path.join(out_dir, "best_model_valloss.pth"))
# # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # #             self.counter_loss += 1

# # # # # # # # # # # # # #     def update_full_val(self, result: dict, model, out_dir, epoch,
# # # # # # # # # # # # # #                         optimizer, tl, vl, min_epochs: int = 80):
# # # # # # # # # # # # # #         """
# # # # # # # # # # # # # #         FIX-T24-B: Primary trigger = sr_12h (ShortRangeHead 12h ADE).
# # # # # # # # # # # # # #         """
# # # # # # # # # # # # # #         sr_12h    = result.get("sr_12h", float("inf"))
# # # # # # # # # # # # # #         blend_ade = result.get("ADE",    float("inf"))

# # # # # # # # # # # # # #         # Save best model if SR 12h ADE improved
# # # # # # # # # # # # # #         improved = sr_12h < self.best_sr12h - self.sr_tol
# # # # # # # # # # # # # #         if improved:
# # # # # # # # # # # # # #             self.best_sr12h = sr_12h
# # # # # # # # # # # # # #             self.best_ade   = min(self.best_ade, blend_ade)
# # # # # # # # # # # # # #             self.counter_12h = 0
# # # # # # # # # # # # # #             torch.save(dict(
# # # # # # # # # # # # # #                 epoch=epoch, model_state_dict=model.state_dict(),
# # # # # # # # # # # # # #                 optimizer_state=optimizer.state_dict(),
# # # # # # # # # # # # # #                 train_loss=tl, val_loss=vl,
# # # # # # # # # # # # # #                 sr_ade_12h=sr_12h, blend_ade=blend_ade,
# # # # # # # # # # # # # #                 model_version="v24-SR+FM"),
# # # # # # # # # # # # # #                 os.path.join(out_dir, "best_model.pth"))
# # # # # # # # # # # # # #             print(f"  ✅ Best SR-12h {sr_12h:.1f} km  "
# # # # # # # # # # # # # #                   f"blend-ADE {blend_ade:.1f} km  (epoch {epoch})")
# # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # #             self.counter_12h += 1
# # # # # # # # # # # # # #             print(f"  No SR-12h improvement {self.counter_12h}/{self.patience}"
# # # # # # # # # # # # # #                   f"  (Δ={self.best_sr12h - sr_12h:.1f} km < tol={self.sr_tol} km)"
# # # # # # # # # # # # # #                   f"  | Loss counter {self.counter_loss}/{self.patience}"
# # # # # # # # # # # # # #                   f"  | SR-12h={sr_12h:.1f}  blend={blend_ade:.1f}")

# # # # # # # # # # # # # #         if epoch >= min_epochs:
# # # # # # # # # # # # # #             if (self.counter_12h >= self.patience
# # # # # # # # # # # # # #                     and self.counter_loss >= self.patience):
# # # # # # # # # # # # # #                 self.early_stop = True
# # # # # # # # # # # # # #                 print(f"  ⛔ Early stop @ epoch {epoch}")
# # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # #             if (self.counter_12h >= self.patience
# # # # # # # # # # # # # #                     and self.counter_loss >= self.patience):
# # # # # # # # # # # # # #                 print(f"  ⚠  Would stop but min_epochs={min_epochs} not reached.")
# # # # # # # # # # # # # #                 self.counter_12h  = 0
# # # # # # # # # # # # # #                 self.counter_loss = 0

# # # # # # # # # # # # # #     def log_subset_sr(self, sr_12h: float, sr_24h: float, epoch: int):
# # # # # # # # # # # # # #         print(f"  [SUBSET SR ep{epoch}]  12h={sr_12h:.1f}  24h={sr_24h:.1f} km"
# # # # # # # # # # # # # #               f"  [target <50/<100]  (monitor only)")


# # # # # # # # # # # # # # # ── GPH/UV checks (unchanged) ─────────────────────────────────────────────────

# # # # # # # # # # # # # # # def _check_gph500(bl, train_dataset):
# # # # # # # # # # # # # # #     env_data = bl[13]
# # # # # # # # # # # # # # #     if env_data is None or "gph500_mean" not in env_data:
# # # # # # # # # # # # # # #         print("  ⚠️  GPH500 key not found"); return
# # # # # # # # # # # # # # #     gph_val  = env_data["gph500_mean"]
# # # # # # # # # # # # # # #     gph_mean = gph_val.mean().item()
# # # # # # # # # # # # # # #     zero_pct = 100.0 * (gph_val == 0).sum().item() / max(gph_val.numel(), 1)
# # # # # # # # # # # # # # #     if 25.0 < gph_mean < 95.0:
# # # # # # # # # # # # # # #         print(f"  ✅ GPH500 OK (mean={gph_mean:.2f} dam, zero={zero_pct:.1f}%)")
# # # # # # # # # # # # # # #     else:
# # # # # # # # # # # # # # #         print(f"  ⚠️  GPH500 unexpected (mean={gph_mean:.4f}, zero={zero_pct:.1f}%)")


# # # # # # # # # # # # # # def _check_gph500(bl, train_dataset):
# # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # #         env_dir = train_dataset.env_path
# # # # # # # # # # # # # #     except AttributeError:
# # # # # # # # # # # # # #         try:
# # # # # # # # # # # # # #             env_dir = train_dataset.dataset.env_path
# # # # # # # # # # # # # #         except AttributeError:
# # # # # # # # # # # # # #             env_dir = "UNKNOWN"
# # # # # # # # # # # # # #     print(f"  Env path   : {env_dir}  exists={os.path.exists(env_dir) if env_dir != 'UNKNOWN' else 'N/A'}")

# # # # # # # # # # # # # #     env_data = bl[13]
# # # # # # # # # # # # # #     if env_data is None:
# # # # # # # # # # # # # #         print("  ⚠️  env_data is None"); return

# # # # # # # # # # # # # #     # has_data3d bị skip trong collate → không check nữa
# # # # # # # # # # # # # #     for key, expected_range in [
# # # # # # # # # # # # # #         ("gph500_mean",   (-3.0, 3.0)),   # đã normalized
# # # # # # # # # # # # # #         ("gph500_center", (-3.0, 3.0)),
# # # # # # # # # # # # # #     ]:
# # # # # # # # # # # # # #         if key not in env_data:
# # # # # # # # # # # # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # # # # # # # # # # # #         v    = env_data[key]
# # # # # # # # # # # # # #         mn   = v.mean().item()
# # # # # # # # # # # # # #         std  = v.std().item()
# # # # # # # # # # # # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # # # # # # # # # # # #         lo, hi = expected_range
# # # # # # # # # # # # # #         if zero > 80.0:
# # # # # # # # # # # # # #             print(f"  ⚠️  {key}: zero={zero:.1f}% → env missing!")
# # # # # # # # # # # # # #         elif lo <= mn <= hi:
# # # # # # # # # # # # # #             print(f"  ✅ {key}: mean={mn:.3f} std={std:.3f} zero={zero:.1f}%")
# # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # #             print(f"  ⚠️  {key}: mean={mn:.3f} ngoài range [{lo},{hi}]")

# # # # # # # # # # # # # # def _check_uv500(bl):
# # # # # # # # # # # # # #     env_data = bl[13]
# # # # # # # # # # # # # #     if env_data is None: return
# # # # # # # # # # # # # #     for key in ("u500_mean", "v500_mean"):
# # # # # # # # # # # # # #         if key not in env_data:
# # # # # # # # # # # # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # # # # # # # # # # # #         v    = env_data[key]
# # # # # # # # # # # # # #         mn   = v.mean().item()
# # # # # # # # # # # # # #         std  = v.std().item()
# # # # # # # # # # # # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # # # # # # # # # # # #         if zero > 80.0:
# # # # # # # # # # # # # #             print(f"  ⚠️  {key}: zero={zero:.1f}% → u/v500 missing!")
# # # # # # # # # # # # # #         else:
# # # # # # # # # # # # # #             # expected: normalized [-1,1], mean ~0, std ~0.1-0.3
# # # # # # # # # # # # # #             print(f"  ✅ {key}: mean={mn:.4f} std={std:.4f} zero={zero:.1f}%")

# # # # # # # # # # # # # # def _load_baseline_errors(path, name):
# # # # # # # # # # # # # #     if path is None:
# # # # # # # # # # # # # #         print(f"\n  ⚠  {name} errors not provided.\n"); return None
# # # # # # # # # # # # # #     if not os.path.exists(path):
# # # # # # # # # # # # # #         print(f"\n  ⚠  {path} not found.\n"); return None
# # # # # # # # # # # # # #     arr = np.load(path)
# # # # # # # # # # # # # #     print(f"  ✓  Loaded {name}: {arr.shape}")
# # # # # # # # # # # # # #     return arr


# # # # # # # # # # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # def get_args():
# # # # # # # # # # # # # #     p = argparse.ArgumentParser(
# # # # # # # # # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # # # # # # # # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # # # # # # # # # # # # #     p.add_argument("--obs_len",         default=8,          type=int)
# # # # # # # # # # # # # #     p.add_argument("--pred_len",        default=12,         type=int)
# # # # # # # # # # # # # #     p.add_argument("--test_year",       default=None,       type=int)
# # # # # # # # # # # # # #     p.add_argument("--batch_size",      default=32,         type=int)
# # # # # # # # # # # # # #     p.add_argument("--num_epochs",      default=200,        type=int)
# # # # # # # # # # # # # #     p.add_argument("--g_learning_rate", default=2e-4,       type=float)
# # # # # # # # # # # # # #     p.add_argument("--phase1_lr",       default=5e-4,       type=float,
# # # # # # # # # # # # # #                    help="FIX-T24-A: LR cho phase 1 (ShortRangeHead focus)")
# # # # # # # # # # # # # #     p.add_argument("--phase2_start",    default=30,         type=int,
# # # # # # # # # # # # # #                    help="FIX-T24-A: epoch bắt đầu phase 2 (unfreeze all)")
# # # # # # # # # # # # # #     p.add_argument("--weight_decay",    default=1e-4,       type=float)
# # # # # # # # # # # # # #     p.add_argument("--warmup_epochs",   default=3,          type=int)
# # # # # # # # # # # # # #     p.add_argument("--grad_clip",       default=2.0,        type=float)
# # # # # # # # # # # # # #     p.add_argument("--grad_accum",      default=2,          type=int)
# # # # # # # # # # # # # #     p.add_argument("--patience",        default=15,         type=int)
# # # # # # # # # # # # # #     p.add_argument("--min_epochs",      default=80,         type=int)
# # # # # # # # # # # # # #     p.add_argument("--n_train_ens",     default=6,          type=int)
# # # # # # # # # # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # # # # # # # # # #     p.add_argument("--num_workers",     default=2,          type=int)

# # # # # # # # # # # # # #     # FIX-M25: smaller sigma / noise
# # # # # # # # # # # # # #     p.add_argument("--sigma_min",            default=0.02,  type=float)
# # # # # # # # # # # # # #     p.add_argument("--ctx_noise_scale",      default=0.002, type=float,
# # # # # # # # # # # # # #                    help="FIX-M25: 0.02→0.002")
# # # # # # # # # # # # # #     p.add_argument("--initial_sample_sigma", default=0.03,  type=float,
# # # # # # # # # # # # # #                    help="FIX-M25: 0.1→0.03")

# # # # # # # # # # # # # #     p.add_argument("--ode_steps_train", default=20,  type=int)
# # # # # # # # # # # # # #     p.add_argument("--ode_steps_val",   default=30,  type=int)
# # # # # # # # # # # # # #     p.add_argument("--ode_steps_test",  default=50,  type=int)
# # # # # # # # # # # # # #     p.add_argument("--ode_steps",       default=None, type=int)

# # # # # # # # # # # # # #     p.add_argument("--val_ensemble",    default=30,  type=int)
# # # # # # # # # # # # # #     p.add_argument("--fast_ensemble",   default=8,   type=int)

# # # # # # # # # # # # # #     p.add_argument("--fno_modes_h",      default=4,  type=int)
# # # # # # # # # # # # # #     p.add_argument("--fno_modes_t",      default=4,  type=int)
# # # # # # # # # # # # # #     p.add_argument("--fno_layers",       default=4,  type=int)
# # # # # # # # # # # # # #     p.add_argument("--fno_d_model",      default=32, type=int)
# # # # # # # # # # # # # #     p.add_argument("--fno_spatial_down", default=32, type=int)
# # # # # # # # # # # # # #     p.add_argument("--mamba_d_state",    default=16, type=int)

# # # # # # # # # # # # # #     p.add_argument("--val_loss_freq",    default=1,   type=int)
# # # # # # # # # # # # # #     p.add_argument("--val_freq",         default=1,   type=int)
# # # # # # # # # # # # # #     p.add_argument("--val_ade_freq",     default=1,   type=int)
# # # # # # # # # # # # # #     p.add_argument("--full_eval_freq",   default=10,  type=int)
# # # # # # # # # # # # # #     p.add_argument("--val_subset_size",  default=600, type=int)

# # # # # # # # # # # # # #     p.add_argument("--output_dir",      default="runs/v24",      type=str)
# # # # # # # # # # # # # #     p.add_argument("--save_interval",   default=10,              type=int)
# # # # # # # # # # # # # #     p.add_argument("--metrics_csv",     default="metrics.csv",   type=str)
# # # # # # # # # # # # # #     p.add_argument("--predict_csv",     default="predictions.csv", type=str)
# # # # # # # # # # # # # #     p.add_argument("--lstm_errors_npy",      default=None, type=str)
# # # # # # # # # # # # # #     p.add_argument("--diffusion_errors_npy", default=None, type=str)

# # # # # # # # # # # # # #     p.add_argument("--gpu_num",         default="0", type=str)
# # # # # # # # # # # # # #     p.add_argument("--delim",           default=" ")
# # # # # # # # # # # # # #     p.add_argument("--skip",            default=1,   type=int)
# # # # # # # # # # # # # #     p.add_argument("--min_ped",         default=1,   type=int)
# # # # # # # # # # # # # #     p.add_argument("--threshold",       default=0.002, type=float)
# # # # # # # # # # # # # #     p.add_argument("--other_modal",     default="gph")

# # # # # # # # # # # # # #     p.add_argument("--step_weight_decay_epochs", default=60, type=int)
# # # # # # # # # # # # # #     p.add_argument("--lon_flip_prob",            default=0.3, type=float)

# # # # # # # # # # # # # #     # FIX-T24-E: longer PINN warmup
# # # # # # # # # # # # # #     p.add_argument("--pinn_warmup_epochs", default=80,    type=int,
# # # # # # # # # # # # # #                    help="FIX-T24-E: 50→80")
# # # # # # # # # # # # # #     p.add_argument("--pinn_w_start",      default=0.001, type=float)
# # # # # # # # # # # # # #     p.add_argument("--pinn_w_end",        default=0.05,  type=float)

# # # # # # # # # # # # # #     p.add_argument("--vel_warmup_epochs",    default=20,  type=float)
# # # # # # # # # # # # # #     p.add_argument("--vel_w_start",          default=0.5, type=float)
# # # # # # # # # # # # # #     p.add_argument("--vel_w_end",            default=1.5, type=float)
# # # # # # # # # # # # # #     p.add_argument("--recurv_warmup_epochs", default=10,  type=int)
# # # # # # # # # # # # # #     p.add_argument("--recurv_w_start",       default=0.3, type=float)
# # # # # # # # # # # # # #     p.add_argument("--recurv_w_end",         default=1.0, type=float)

# # # # # # # # # # # # # #     return p.parse_args()


# # # # # # # # # # # # # # def _resolve_ode_steps(args):
# # # # # # # # # # # # # #     if args.ode_steps is not None:
# # # # # # # # # # # # # #         return args.ode_steps, args.ode_steps, args.ode_steps
# # # # # # # # # # # # # #     return args.ode_steps_train, args.ode_steps_val, args.ode_steps_test


# # # # # # # # # # # # # # # ── MAIN ──────────────────────────────────────────────────────────────────────

# # # # # # # # # # # # # # def main(args):
# # # # # # # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # # # # # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # # # # # # # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# # # # # # # # # # # # # #     predict_csv = os.path.join(args.output_dir, args.predict_csv)
# # # # # # # # # # # # # #     tables_dir  = os.path.join(args.output_dir, "tables")
# # # # # # # # # # # # # #     stat_dir    = os.path.join(tables_dir, "stat_tests")
# # # # # # # # # # # # # #     os.makedirs(tables_dir, exist_ok=True)
# # # # # # # # # # # # # #     os.makedirs(stat_dir,   exist_ok=True)

# # # # # # # # # # # # # #     ode_train, ode_val, ode_test = _resolve_ode_steps(args)

# # # # # # # # # # # # # #     print("=" * 70)
# # # # # # # # # # # # # #     print("  TC-FlowMatching v24  |  ShortRangeHead + OT-CFM + PINN")
# # # # # # # # # # # # # #     print("  v24 FIXES:")
# # # # # # # # # # # # # #     print("    FIX-T24-A: Two-phase training (backbone freeze ep 0-30)")
# # # # # # # # # # # # # #     print("    FIX-T24-B: Primary metric = SR-12h ADE (ShortRangeHead)")
# # # # # # # # # # # # # #     print("    FIX-T24-C: short_range_weight 8→5→3 schedule")
# # # # # # # # # # # # # #     print("    FIX-T24-D: initial_sample_sigma=0.03, spread_weight↑")
# # # # # # # # # # # # # #     print("    FIX-T24-E: pinn_warmup=80 ep")
# # # # # # # # # # # # # #     print("    FIX-T24-F: evaluate SR-12h/24h separately")
# # # # # # # # # # # # # #     print("    FIX-M23:   ShortRangeHead GRU for deterministic 12h/24h")
# # # # # # # # # # # # # #     print("    FIX-L49:   short_range_regression_loss (Huber)")
# # # # # # # # # # # # # #     print("    FIX-L50:   PINN scale 1e-3→1e-2")
# # # # # # # # # # # # # #     print("    FIX-L52:   ensemble_spread_loss max_spread 200→150 km")
# # # # # # # # # # # # # #     print("=" * 70)
# # # # # # # # # # # # # #     print(f"  device               : {device}")
# # # # # # # # # # # # # #     print(f"  phase2_start         : ep {args.phase2_start}")
# # # # # # # # # # # # # #     print(f"  phase1_lr            : {args.phase1_lr:.2e}")
# # # # # # # # # # # # # #     print(f"  phase2_lr            : {args.g_learning_rate:.2e}")
# # # # # # # # # # # # # #     print(f"  ctx_noise_scale      : {args.ctx_noise_scale}")
# # # # # # # # # # # # # #     print(f"  initial_sample_sigma : {args.initial_sample_sigma}")
# # # # # # # # # # # # # #     print(f"  ode_steps            : train={ode_train} val={ode_val} test={ode_test}")
# # # # # # # # # # # # # #     print(f"  patience             : {args.patience}  min_epochs={args.min_epochs}")
# # # # # # # # # # # # # #     print()

# # # # # # # # # # # # # #     train_dataset, train_loader = data_loader(
# # # # # # # # # # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # # # # # # # # # #     val_dataset, val_loader = data_loader(
# # # # # # # # # # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # # # # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # # # # # # # #     val_subset_loader = make_val_subset_loader(
# # # # # # # # # # # # # #         val_dataset, args.val_subset_size, args.batch_size,
# # # # # # # # # # # # # #         seq_collate, args.num_workers)

# # # # # # # # # # # # # #     test_loader = None
# # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # #         _, test_loader = data_loader(
# # # # # # # # # # # # # #             args, {"root": args.dataset_root, "type": "test"},
# # # # # # # # # # # # # #             test=True, test_year=None)
# # # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # # #         print(f"  Warning: test loader: {e}")

# # # # # # # # # # # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # # # # # # # # # # #     print(f"  val   : {len(val_dataset)} seq")
# # # # # # # # # # # # # #     if test_loader:
# # # # # # # # # # # # # #         print(f"  test  : {len(test_loader.dataset)} seq")

# # # # # # # # # # # # # #     model = TCFlowMatching(
# # # # # # # # # # # # # #         pred_len             = args.pred_len,
# # # # # # # # # # # # # #         obs_len              = args.obs_len,
# # # # # # # # # # # # # #         sigma_min            = args.sigma_min,
# # # # # # # # # # # # # #         n_train_ens          = args.n_train_ens,
# # # # # # # # # # # # # #         ctx_noise_scale      = args.ctx_noise_scale,
# # # # # # # # # # # # # #         initial_sample_sigma = args.initial_sample_sigma,
# # # # # # # # # # # # # #     ).to(device)

# # # # # # # # # # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # # # # # # # # #     print(f"  params  : {n_params:,}")

# # # # # # # # # # # # # #     try:
# # # # # # # # # # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # # # # # # # # # #         print("  torch.compile: enabled")
# # # # # # # # # # # # # #     except Exception:
# # # # # # # # # # # # # #         pass

# # # # # # # # # # # # # #     # ── Phase 1: freeze backbone, higher LR for ShortRangeHead ──────────────
# # # # # # # # # # # # # #     freeze_backbone(model)
# # # # # # # # # # # # # #     optimizer = optim.AdamW(
# # # # # # # # # # # # # #         filter(lambda p: p.requires_grad, model.parameters()),
# # # # # # # # # # # # # #         lr=args.phase1_lr, weight_decay=args.weight_decay,
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
# # # # # # # # # # # # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # # # # # # # # # # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # # # # # # # # # # # #     scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

# # # # # # # # # # # # # #     saver  = BestModelSaver(patience=args.patience, sr_tol=3.0, ade_tol=1.0)
# # # # # # # # # # # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)

# # # # # # # # # # # # # #     print("=" * 70)
# # # # # # # # # # # # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch)")
# # # # # # # # # # # # # #     print("=" * 70)

# # # # # # # # # # # # # #     epoch_times   = []
# # # # # # # # # # # # # #     train_start   = time.perf_counter()
# # # # # # # # # # # # # #     last_val_loss = float("inf")
# # # # # # # # # # # # # #     _phase        = 1
# # # # # # # # # # # # # #     _prev_ens     = 1
# # # # # # # # # # # # # #     _lr_ep30_done = False
# # # # # # # # # # # # # #     _lr_ep60_done = False

# # # # # # # # # # # # # #     import Model.losses as _losses_mod

# # # # # # # # # # # # # #     for epoch in range(args.num_epochs):

# # # # # # # # # # # # # #         # ── Phase transition ──────────────────────────────────────────────
# # # # # # # # # # # # # #         if epoch == args.phase2_start and _phase == 1:
# # # # # # # # # # # # # #             _phase = 2
# # # # # # # # # # # # # #             unfreeze_all(model)
# # # # # # # # # # # # # #             # Reset optimizer with phase-2 LR
# # # # # # # # # # # # # #             optimizer = optim.AdamW(
# # # # # # # # # # # # # #                 model.parameters(),
# # # # # # # # # # # # # #                 lr=args.g_learning_rate,
# # # # # # # # # # # # # #                 weight_decay=args.weight_decay,
# # # # # # # # # # # # # #             )
# # # # # # # # # # # # # #             rem_steps = steps_per_epoch * (args.num_epochs - epoch)
# # # # # # # # # # # # # #             scheduler = get_cosine_schedule_with_warmup(
# # # # # # # # # # # # # #                 optimizer, steps_per_epoch, rem_steps, min_lr=5e-6)
# # # # # # # # # # # # # #             saver.reset_counters(f"Phase 2 started @ ep {epoch}")
# # # # # # # # # # # # # #             print(f"\n  ↺  PHASE 2 START ep {epoch}: unfreeze all, "
# # # # # # # # # # # # # #                   f"LR={args.g_learning_rate:.2e}")

# # # # # # # # # # # # # #         # ── Weight schedules ──────────────────────────────────────────────
# # # # # # # # # # # # # #         current_ens = get_progressive_ens(epoch, args.n_train_ens)
# # # # # # # # # # # # # #         model.n_train_ens = current_ens
# # # # # # # # # # # # # #         eff_fast_ens = min(args.fast_ensemble,
# # # # # # # # # # # # # #                            max(current_ens * 2, args.fast_ensemble))

# # # # # # # # # # # # # #         if current_ens != _prev_ens:
# # # # # # # # # # # # # #             saver.reset_counters(f"n_train_ens {_prev_ens}→{current_ens}")
# # # # # # # # # # # # # #             _prev_ens = current_ens

# # # # # # # # # # # # # #         step_alpha    = get_step_weight_alpha(epoch, args.step_weight_decay_epochs)
# # # # # # # # # # # # # #         epoch_weights = copy.copy(_BASE_WEIGHTS)
# # # # # # # # # # # # # #         epoch_weights["fm"]          = get_fm_weight(epoch)
# # # # # # # # # # # # # #         epoch_weights["short_range"] = get_short_range_weight(epoch)
# # # # # # # # # # # # # #         epoch_weights["spread"]      = get_spread_weight(epoch)
# # # # # # # # # # # # # #         epoch_weights["pinn"]        = get_pinn_weight(
# # # # # # # # # # # # # #             epoch, args.pinn_warmup_epochs, args.pinn_w_start, args.pinn_w_end)
# # # # # # # # # # # # # #         epoch_weights["velocity"]    = get_velocity_weight(
# # # # # # # # # # # # # #             epoch, args.vel_warmup_epochs, args.vel_w_start, args.vel_w_end)
# # # # # # # # # # # # # #         epoch_weights["recurv"]      = get_recurv_weight(
# # # # # # # # # # # # # #             epoch, args.recurv_warmup_epochs, args.recurv_w_start, args.recurv_w_end)
# # # # # # # # # # # # # #         _losses_mod.WEIGHTS.update(epoch_weights)

# # # # # # # # # # # # # #         current_clip = get_grad_clip(epoch, warmup_epochs=20,
# # # # # # # # # # # # # #                                       clip_start=args.grad_clip, clip_end=1.0)

# # # # # # # # # # # # # #         # LR warm restarts (phase 2 only)
# # # # # # # # # # # # # #         if _phase == 2:
# # # # # # # # # # # # # #             if epoch == 30 and not _lr_ep30_done:
# # # # # # # # # # # # # #                 _lr_ep30_done = True
# # # # # # # # # # # # # #                 scheduler = get_cosine_schedule_with_warmup(
# # # # # # # # # # # # # #                     optimizer, steps_per_epoch,
# # # # # # # # # # # # # #                     steps_per_epoch * (args.num_epochs - 30), min_lr=5e-6)
# # # # # # # # # # # # # #                 saver.reset_counters("LR warm restart ep 30")
# # # # # # # # # # # # # #                 print(f"  ↺  Warm Restart LR @ ep 30")

# # # # # # # # # # # # # #             if epoch == 60 and not _lr_ep60_done:
# # # # # # # # # # # # # #                 _lr_ep60_done = True
# # # # # # # # # # # # # #                 scheduler = get_cosine_schedule_with_warmup(
# # # # # # # # # # # # # #                     optimizer, steps_per_epoch,
# # # # # # # # # # # # # #                     steps_per_epoch * (args.num_epochs - 60), min_lr=1e-6)
# # # # # # # # # # # # # #                 saver.reset_counters("LR warm restart ep 60")
# # # # # # # # # # # # # #                 print(f"  ↺  Warm Restart LR @ ep 60")

# # # # # # # # # # # # # #         # ── Training loop ─────────────────────────────────────────────────
# # # # # # # # # # # # # #         model.train()
# # # # # # # # # # # # # #         sum_loss   = 0.0
# # # # # # # # # # # # # #         t0         = time.perf_counter()
# # # # # # # # # # # # # #         optimizer.zero_grad()
# # # # # # # # # # # # # #         recurv_ratio_buf = []

# # # # # # # # # # # # # #         for i, batch in enumerate(train_loader):
# # # # # # # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # # # # # # #             if epoch == 0 and i == 0:
# # # # # # # # # # # # # #                 # Đúng attribute name
# # # # # # # # # # # # # #                 try:
# # # # # # # # # # # # # #                     env_dir = train_dataset.env_path
# # # # # # # # # # # # # #                 except AttributeError:
# # # # # # # # # # # # # #                     try:
# # # # # # # # # # # # # #                         env_dir = train_dataset.dataset.env_path
# # # # # # # # # # # # # #                     except AttributeError:
# # # # # # # # # # # # # #                         env_dir = "UNKNOWN"
# # # # # # # # # # # # # #                 print(f"  Env path     : {env_dir}")
# # # # # # # # # # # # # #                 print(f"  Env exists   : {os.path.exists(env_dir) if env_dir != 'UNKNOWN' else 'N/A'}")
# # # # # # # # # # # # # #                 print(f"  Root path    : {getattr(train_dataset, 'root_path', 'N/A')}")
# # # # # # # # # # # # # #                         # Check tensor trong batch (sau collate)
# # # # # # # # # # # # # #                 _check_gph500(bl, train_dataset)
# # # # # # # # # # # # # #                 _check_uv500(bl)

# # # # # # # # # # # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # # # # # # #                 bd = model.get_loss_breakdown(bl, step_weight_alpha=step_alpha)

# # # # # # # # # # # # # #             loss_to_bp = bd["total"] / max(args.grad_accum, 1)
# # # # # # # # # # # # # #             scaler.scale(loss_to_bp).backward()

# # # # # # # # # # # # # #             if ((i + 1) % args.grad_accum == 0
# # # # # # # # # # # # # #                     or (i + 1) == len(train_loader)):
# # # # # # # # # # # # # #                 scaler.unscale_(optimizer)
# # # # # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_(
# # # # # # # # # # # # # #                     model.parameters(), current_clip)
# # # # # # # # # # # # # #                 scaler.step(optimizer)
# # # # # # # # # # # # # #                 scaler.update()
# # # # # # # # # # # # # #                 optimizer.zero_grad()
# # # # # # # # # # # # # #                 scheduler.step() 

# # # # # # # # # # # # # #             sum_loss += bd["total"].item()
# # # # # # # # # # # # # #             if "recurv_ratio" in bd:
# # # # # # # # # # # # # #                 recurv_ratio_buf.append(bd["recurv_ratio"])

# # # # # # # # # # # # # #             if i % 20 == 0:
# # # # # # # # # # # # # #                 lr      = optimizer.param_groups[0]["lr"]
# # # # # # # # # # # # # #                 sr_loss = bd.get("short_range", 0.0)
# # # # # # # # # # # # # #                 elapsed = time.perf_counter() - t0
# # # # # # # # # # # # # #                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # # # # # # # # # # # #                       f"  loss={bd['total'].item():.3f}"
# # # # # # # # # # # # # #                       f"  fm={bd.get('fm',0):.2f}"
# # # # # # # # # # # # # #                       f"  sr={sr_loss:.4f}"
# # # # # # # # # # # # # #                       f"  vel={bd.get('velocity',0):.3f}"
# # # # # # # # # # # # # #                       f"  pinn={bd.get('pinn',0):.4f}"
# # # # # # # # # # # # # #                       f"  spread={bd.get('spread',0):.3f}"
# # # # # # # # # # # # # #                       f"  sr_w={epoch_weights['short_range']:.1f}"
# # # # # # # # # # # # # #                       f"  alpha={step_alpha:.2f}"
# # # # # # # # # # # # # #                       f"  ph={_phase}"
# # # # # # # # # # # # # #                       f"  ens={current_ens}"
# # # # # # # # # # # # # #                       f"  lr={lr:.2e}  t={elapsed:.0f}s")

# # # # # # # # # # # # # #         ep_s    = time.perf_counter() - t0
# # # # # # # # # # # # # #         epoch_times.append(ep_s)
# # # # # # # # # # # # # #         avg_t   = sum_loss / len(train_loader)
# # # # # # # # # # # # # #         mean_rr = float(np.mean(recurv_ratio_buf)) if recurv_ratio_buf else 0.0

# # # # # # # # # # # # # #         # ── Val loss ──────────────────────────────────────────────────────
# # # # # # # # # # # # # #         model.eval()
# # # # # # # # # # # # # #         val_loss = 0.0
# # # # # # # # # # # # # #         t_val    = time.perf_counter()
# # # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # # #             for batch in val_loader:
# # # # # # # # # # # # # #                 bl_v = move(list(batch), device)
# # # # # # # # # # # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # # # # # # #                     val_loss += model.get_loss(bl_v).item()
# # # # # # # # # # # # # #         last_val_loss = val_loss / len(val_loader)
# # # # # # # # # # # # # #         t_val_s = time.perf_counter() - t_val
# # # # # # # # # # # # # #         saver.update_val_loss(last_val_loss, model, args.output_dir,
# # # # # # # # # # # # # #                                epoch, optimizer, avg_t)

# # # # # # # # # # # # # #         print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
# # # # # # # # # # # # # #               f"  rr={mean_rr:.2f}"
# # # # # # # # # # # # # #               f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
# # # # # # # # # # # # # #               f"  ens={current_ens}  alpha={step_alpha:.2f}")

# # # # # # # # # # # # # #         # ── Fast eval ─────────────────────────────────────────────────────
# # # # # # # # # # # # # #         t_ade   = time.perf_counter()
# # # # # # # # # # # # # #         m_fast  = evaluate_fast(model, val_subset_loader, device,
# # # # # # # # # # # # # #                                 ode_train, args.pred_len, eff_fast_ens)
# # # # # # # # # # # # # #         t_ade_s = time.perf_counter() - t_ade

# # # # # # # # # # # # # #         sr_12h   = m_fast.get("sr_ade_12h", float("nan"))
# # # # # # # # # # # # # #         sr_24h   = m_fast.get("sr_ade_24h", float("nan"))
# # # # # # # # # # # # # #         spread72 = m_fast.get("spread_72h_km", 0.0)
# # # # # # # # # # # # # #         collapse = "  ⚠️ COLLAPSE!" if spread72 < 10.0 else ""
# # # # # # # # # # # # # #         hi_sprd  = "  ⚠️ SPREAD!" if spread72 > 400.0 else ""
# # # # # # # # # # # # # #         sr_hit12 = "  🎯 <50km!" if sr_12h < 50.0 else ""
# # # # # # # # # # # # # #         sr_hit24 = "  🎯 <100km!" if sr_24h < 100.0 else ""

# # # # # # # # # # # # # #         print(f"  [FAST ep{epoch} {t_ade_s:.0f}s]"
# # # # # # # # # # # # # #               f"  ADE={m_fast['ADE']:.1f}  FDE={m_fast['FDE']:.1f} km"
# # # # # # # # # # # # # #               f"  12h={m_fast.get('12h',float('nan')):.0f}"
# # # # # # # # # # # # # #               f"  24h={m_fast.get('24h',float('nan')):.0f}"
# # # # # # # # # # # # # #               f"  72h={m_fast.get('72h',float('nan')):.0f} km"
# # # # # # # # # # # # # #               f"  spread={spread72:.1f} km"
# # # # # # # # # # # # # #               f"{collapse}{hi_sprd}")
# # # # # # # # # # # # # #         print(f"         ShortRange: SR-12h={sr_12h:.1f} km"
# # # # # # # # # # # # # #               f"  SR-24h={sr_24h:.1f} km"
# # # # # # # # # # # # # #               f"{sr_hit12}{sr_hit24}  (monitor only)")

# # # # # # # # # # # # # #         saver.log_subset_sr(sr_12h, sr_24h, epoch)

# # # # # # # # # # # # # #         # ── Full val ADE ──────────────────────────────────────────────────
# # # # # # # # # # # # # #         if epoch % args.val_ade_freq == 0:
# # # # # # # # # # # # # #             try:
# # # # # # # # # # # # # #                 r_full = evaluate_full_val_ade(
# # # # # # # # # # # # # #                     model, val_loader, device,
# # # # # # # # # # # # # #                     ode_steps     = ode_train,
# # # # # # # # # # # # # #                     pred_len      = args.pred_len,
# # # # # # # # # # # # # #                     fast_ensemble = eff_fast_ens,
# # # # # # # # # # # # # #                     metrics_csv   = metrics_csv,
# # # # # # # # # # # # # #                     epoch         = epoch,
# # # # # # # # # # # # # #                     tag           = f"val_full_ep{epoch:03d}",
# # # # # # # # # # # # # #                 )
# # # # # # # # # # # # # #                 # FIX-T24-B: primary = sr_12h
# # # # # # # # # # # # # #                 saver.update_full_val(
# # # # # # # # # # # # # #                     r_full, model, args.output_dir, epoch,
# # # # # # # # # # # # # #                     optimizer, avg_t, last_val_loss,
# # # # # # # # # # # # # #                     min_epochs=args.min_epochs)
# # # # # # # # # # # # # #             except Exception as e:
# # # # # # # # # # # # # #                 print(f"  ⚠  Full val failed: {e}")
# # # # # # # # # # # # # #                 import traceback; traceback.print_exc()

# # # # # # # # # # # # # #         # ── Full eval ─────────────────────────────────────────────────────
# # # # # # # # # # # # # #         if epoch % args.full_eval_freq == 0 and epoch > 0:
# # # # # # # # # # # # # #             try:
# # # # # # # # # # # # # #                 dm, _, _, _ = evaluate_full(
# # # # # # # # # # # # # #                     model, val_loader, device,
# # # # # # # # # # # # # #                     ode_val, args.pred_len, args.val_ensemble,
# # # # # # # # # # # # # #                     metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}")
# # # # # # # # # # # # # #                 print(dm.summary())
# # # # # # # # # # # # # #             except Exception as e:
# # # # # # # # # # # # # #                 print(f"  ⚠  full_eval failed ep {epoch}: {e}")

# # # # # # # # # # # # # #         if (epoch + 1) % args.save_interval == 0:
# # # # # # # # # # # # # #             torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
# # # # # # # # # # # # # #                        os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))

# # # # # # # # # # # # # #         if saver.early_stop:
# # # # # # # # # # # # # #             print(f"  Early stopping @ epoch {epoch}")
# # # # # # # # # # # # # #             break

# # # # # # # # # # # # # #         if epoch % 5 == 4:
# # # # # # # # # # # # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # # # # # # # # # # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # # # # # # # # # # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # # # # # # # # # # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
# # # # # # # # # # # # # #                   f"  (avg {avg_ep:.0f}s/epoch)")

# # # # # # # # # # # # # #     # ── Final ─────────────────────────────────────────────────────────────────
# # # # # # # # # # # # # #     _losses_mod.WEIGHTS["pinn"]        = args.pinn_w_end
# # # # # # # # # # # # # #     _losses_mod.WEIGHTS["velocity"]    = args.vel_w_end
# # # # # # # # # # # # # #     _losses_mod.WEIGHTS["recurv"]      = args.recurv_w_end
# # # # # # # # # # # # # #     _losses_mod.WEIGHTS["short_range"] = 3.0

# # # # # # # # # # # # # #     total_train_h = (time.perf_counter() - train_start) / 3600

# # # # # # # # # # # # # #     print(f"\n{'='*70}  FINAL TEST (ode_steps={ode_test})")
# # # # # # # # # # # # # #     all_results = []

# # # # # # # # # # # # # #     if test_loader:
# # # # # # # # # # # # # #         best_path = os.path.join(args.output_dir, "best_model.pth")
# # # # # # # # # # # # # #         if not os.path.exists(best_path):
# # # # # # # # # # # # # #             best_path = os.path.join(args.output_dir, "best_model_valloss.pth")
# # # # # # # # # # # # # #         if os.path.exists(best_path):
# # # # # # # # # # # # # #             ck = torch.load(best_path, map_location=device)
# # # # # # # # # # # # # #             try:
# # # # # # # # # # # # # #                 model.load_state_dict(ck["model_state_dict"])
# # # # # # # # # # # # # #             except Exception:
# # # # # # # # # # # # # #                 model.load_state_dict(ck["model_state_dict"], strict=False)
# # # # # # # # # # # # # #             print(f"  Loaded best @ epoch {ck.get('epoch','?')}"
# # # # # # # # # # # # # #                   f"  SR-12h={ck.get('sr_ade_12h','?')}")

# # # # # # # # # # # # # #         final_ens = max(args.val_ensemble, 50)
# # # # # # # # # # # # # #         dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
# # # # # # # # # # # # # #             model, test_loader, device,
# # # # # # # # # # # # # #             ode_test, args.pred_len, final_ens,
# # # # # # # # # # # # # #             metrics_csv=metrics_csv, tag="test_final",
# # # # # # # # # # # # # #             predict_csv=predict_csv)
# # # # # # # # # # # # # #         print(dm_test.summary())

# # # # # # # # # # # # # #         all_results.append(ModelResult(
# # # # # # # # # # # # # #             model_name   = "FM+SR+PINN-v24",
# # # # # # # # # # # # # #             split        = "test",
# # # # # # # # # # # # # #             ADE          = dm_test.ade,
# # # # # # # # # # # # # #             FDE          = dm_test.fde,
# # # # # # # # # # # # # #             ADE_str      = dm_test.ade_str,
# # # # # # # # # # # # # #             ADE_rec      = dm_test.ade_rec,
# # # # # # # # # # # # # #             delta_rec    = dm_test.pr,
# # # # # # # # # # # # # #             CRPS_mean    = dm_test.crps_mean,
# # # # # # # # # # # # # #             CRPS_72h     = dm_test.crps_72h,
# # # # # # # # # # # # # #             SSR          = dm_test.ssr_mean,
# # # # # # # # # # # # # #             TSS_72h      = dm_test.tss_72h,
# # # # # # # # # # # # # #             OYR          = dm_test.oyr_mean,
# # # # # # # # # # # # # #             DTW          = dm_test.dtw_mean,
# # # # # # # # # # # # # #             ATE_abs      = dm_test.ate_abs_mean,
# # # # # # # # # # # # # #             CTE_abs      = dm_test.cte_abs_mean,
# # # # # # # # # # # # # #             n_total      = dm_test.n_total,
# # # # # # # # # # # # # #             n_recurv     = dm_test.n_rec,
# # # # # # # # # # # # # #             train_time_h = total_train_h,
# # # # # # # # # # # # # #             params_M     = n_params / 1e6,
# # # # # # # # # # # # # #         ))

# # # # # # # # # # # # # #         _, cliper_errs = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
# # # # # # # # # # # # # #         persist_errs   = persistence_errors(obs_seqs, gt_seqs, args.pred_len)
# # # # # # # # # # # # # #         fmpinn_per_seq = np.array([seq_ade_km(np.array(pp), np.array(g))
# # # # # # # # # # # # # #                                     for pp, g in zip(pred_seqs, gt_seqs)])

# # # # # # # # # # # # # #         np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
# # # # # # # # # # # # # #         np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
# # # # # # # # # # # # # #         np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))

# # # # # # # # # # # # # #         stat_rows = [
# # # # # # # # # # # # # #             paired_tests(fmpinn_per_seq, cliper_errs.mean(1),  "FM+SR vs CLIPER", 5),
# # # # # # # # # # # # # #             paired_tests(fmpinn_per_seq, persist_errs.mean(1), "FM+SR vs Persist", 5),
# # # # # # # # # # # # # #         ]
# # # # # # # # # # # # # #         lstm_per_seq      = _load_baseline_errors(args.lstm_errors_npy, "LSTM")
# # # # # # # # # # # # # #         diffusion_per_seq = _load_baseline_errors(args.diffusion_errors_npy, "Diffusion")
# # # # # # # # # # # # # #         if lstm_per_seq is not None:
# # # # # # # # # # # # # #             stat_rows.append(paired_tests(fmpinn_per_seq, lstm_per_seq, "FM+SR vs LSTM", 5))
# # # # # # # # # # # # # #         if diffusion_per_seq is not None:
# # # # # # # # # # # # # #             stat_rows.append(paired_tests(fmpinn_per_seq, diffusion_per_seq, "FM+SR vs Diffusion", 5))

# # # # # # # # # # # # # #         export_all_tables(
# # # # # # # # # # # # # #             results=all_results, ablation_rows=DEFAULT_ABLATION,
# # # # # # # # # # # # # #             stat_rows=stat_rows, pinn_sens_rows=DEFAULT_PINN_SENSITIVITY,
# # # # # # # # # # # # # #             compute_rows=DEFAULT_COMPUTE, out_dir=tables_dir)

# # # # # # # # # # # # # #         with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
# # # # # # # # # # # # # #             fh.write(dm_test.summary())
# # # # # # # # # # # # # #             fh.write(f"\n\nmodel_version         : FM+SR+PINN v24\n")
# # # # # # # # # # # # # #             fh.write(f"ctx_noise_scale       : {args.ctx_noise_scale}\n")
# # # # # # # # # # # # # #             fh.write(f"initial_sample_sigma  : {args.initial_sample_sigma}\n")
# # # # # # # # # # # # # #             fh.write(f"ode_steps_test        : {ode_test}\n")
# # # # # # # # # # # # # #             fh.write(f"eval_ensemble         : {final_ens}\n")
# # # # # # # # # # # # # #             fh.write(f"train_time_h          : {total_train_h:.2f}\n")
# # # # # # # # # # # # # #             fh.write(f"n_params_M            : {n_params/1e6:.2f}\n")

# # # # # # # # # # # # # #     avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 0
# # # # # # # # # # # # # #     print(f"\n  Best SR-12h ADE    : {saver.best_sr12h:.1f} km")
# # # # # # # # # # # # # #     print(f"  Best blend ADE     : {saver.best_ade:.1f} km")
# # # # # # # # # # # # # #     print(f"  Best val loss      : {saver.best_val_loss:.4f}")
# # # # # # # # # # # # # #     print(f"  Avg epoch time     : {avg_ep:.0f}s")
# # # # # # # # # # # # # #     print(f"  Total training     : {total_train_h:.2f}h")
# # # # # # # # # # # # # #     print("=" * 70)


# # # # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # # # #     args = get_args()
# # # # # # # # # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # # # # # # #         torch.cuda.manual_seed_all(42)
# # # # # # # # # # # # # #     main(args)

# # # # # # # # # # # # # """
# # # # # # # # # # # # # scripts/train_flowmatching.py  ── v25
# # # # # # # # # # # # # ======================================
# # # # # # # # # # # # # FULL REWRITE – fixes từ review:

# # # # # # # # # # # # #   FIX-T-A  [CRITICAL] Truyền epoch vào get_loss() / get_loss_breakdown()
# # # # # # # # # # # # #            để PINN AdaptClamp, adaptive BVE weighting, L_PWR kích hoạt
# # # # # # # # # # # # #            đúng thời điểm. v24 không truyền epoch → PINN luôn ở mode
# # # # # # # # # # # # #            tanh (epoch>=20 ngay từ đầu).

# # # # # # # # # # # # #   FIX-T-B  [HIGH] evaluate_full_val_ade() cache raw_ctx: không gọi
# # # # # # # # # # # # #            model.net._context() 2 lần cho cùng một batch.
# # # # # # # # # # # # #            Tích hợp compute_sr_ade() vào vòng lặp chính.

# # # # # # # # # # # # #   FIX-T-C  [HIGH] Phase 1 freeze: freeze ctx_fc1 và ctx_ln cùng với
# # # # # # # # # # # # #            backbone. Lý do: raw_ctx từ frozen encoder → chất lượng kém
# # # # # # # # # # # # #            → ShortRangeHead học trên noise nếu ctx_fc1 không frozen.

# # # # # # # # # # # # #   FIX-T-D  [MEDIUM] short_range_weight schedule: không cần scale cực
# # # # # # # # # # # # #            cao (12.0) vì short_range_regression_loss đã normalize đúng
# # # # # # # # # # # # #            (đơn vị km thực, không chia thêm HUBER_DELTA nữa).
# # # # # # # # # # # # #            Schedule mới: 5.0 → 3.0 → 2.0.

# # # # # # # # # # # # #   FIX-T-E  [MEDIUM] Bridge loss weight: kích hoạt từ epoch 10 sau khi
# # # # # # # # # # # # #            cả SR và FM đã có prediction có nghĩa. ep 0-9: 0.0, sau: 0.5.

# # # # # # # # # # # # #   FIX-T-F  [LOW] Log thêm pinn breakdown (l_sw, l_steer, l_gph, l_pwr)
# # # # # # # # # # # # #            để debug PINN convergence.

# # # # # # # # # # # # # Kept from v24:
# # # # # # # # # # # # #   FIX-T24-A  Two-phase training (backbone freeze)
# # # # # # # # # # # # #   FIX-T24-B  Primary metric = SR-12h ADE
# # # # # # # # # # # # #   FIX-T24-D  spread_weight schedule
# # # # # # # # # # # # #   FIX-T24-E  pinn_warmup_epochs=80
# # # # # # # # # # # # #   FIX-T24-F  log sr_12h/sr_24h separately
# # # # # # # # # # # # # """
# # # # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # # # import sys
# # # # # # # # # # # # # import os
# # # # # # # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # # # # # # import argparse
# # # # # # # # # # # # # import time
# # # # # # # # # # # # # import math
# # # # # # # # # # # # # import random
# # # # # # # # # # # # # import copy

# # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # import torch
# # # # # # # # # # # # # import torch.optim as optim
# # # # # # # # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # # # # # # from Model.data.loader_training import data_loader
# # # # # # # # # # # # # from Model.flow_matching_model import TCFlowMatching, ShortRangeHead
# # # # # # # # # # # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # # # # # # # from Model.losses import WEIGHTS as _BASE_WEIGHTS
# # # # # # # # # # # # # from utils.metrics import (
# # # # # # # # # # # # #     TCEvaluator, StepErrorAccumulator,
# # # # # # # # # # # # #     save_metrics_csv, haversine_km_torch,
# # # # # # # # # # # # #     denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
# # # # # # # # # # # # #     cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
# # # # # # # # # # # # #     brier_skill_score, cliper_errors, persistence_errors,
# # # # # # # # # # # # # )
# # # # # # # # # # # # # from utils.evaluation_tables import (
# # # # # # # # # # # # #     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
# # # # # # # # # # # # #     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
# # # # # # # # # # # # #     DEFAULT_COMPUTE, paired_tests,
# # # # # # # # # # # # # )
# # # # # # # # # # # # # from scripts.statistical_tests import run_all_tests


# # # # # # # # # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # # # # # # # # def haversine_km_np_local(pred_deg, gt_deg):
# # # # # # # # # # # # #     pred_deg = np.atleast_2d(pred_deg)
# # # # # # # # # # # # #     gt_deg   = np.atleast_2d(gt_deg)
# # # # # # # # # # # # #     R = 6371.0
# # # # # # # # # # # # #     lon1, lat1 = np.radians(pred_deg[:, 0]), np.radians(pred_deg[:, 1])
# # # # # # # # # # # # #     lon2, lat2 = np.radians(gt_deg[:, 0]),   np.radians(gt_deg[:, 1])
# # # # # # # # # # # # #     dlon = lon2 - lon1;  dlat = lat2 - lat1
# # # # # # # # # # # # #     a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
# # # # # # # # # # # # #     return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# # # # # # # # # # # # # def seq_ade_km(pred_norm, gt_norm):
# # # # # # # # # # # # #     return float(haversine_km_np_local(denorm_deg_np(pred_norm),
# # # # # # # # # # # # #                                        denorm_deg_np(gt_norm)).mean())


# # # # # # # # # # # # # def move(batch, device):
# # # # # # # # # # # # #     out = list(batch)
# # # # # # # # # # # # #     for i, x in enumerate(out):
# # # # # # # # # # # # #         if torch.is_tensor(x):
# # # # # # # # # # # # #             out[i] = x.to(device)
# # # # # # # # # # # # #         elif isinstance(x, dict):
# # # # # # # # # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # # # # # # #                       for k, v in x.items()}
# # # # # # # # # # # # #     return out


# # # # # # # # # # # # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # # # # # # # # # # # #                             collate_fn, num_workers):
# # # # # # # # # # # # #     n   = len(val_dataset)
# # # # # # # # # # # # #     rng = random.Random(42)
# # # # # # # # # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # # # # # # # # #     return DataLoader(Subset(val_dataset, idx),
# # # # # # # # # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # # # # # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # # # # # # # # # ── Phase control ─────────────────────────────────────────────────────────────

# # # # # # # # # # # # # def get_phase(epoch: int, phase2_start: int = 30) -> int:
# # # # # # # # # # # # #     return 1 if epoch < phase2_start else 2


# # # # # # # # # # # # # # def freeze_backbone(model: TCFlowMatching) -> None:
# # # # # # # # # # # # # #     """
# # # # # # # # # # # # # #     FIX-T-C: Freeze backbone + ctx_fc1 + ctx_ln trong phase 1.
# # # # # # # # # # # # # #     raw_ctx từ frozen encoder chất lượng kém → ctx_fc1 cũng freeze
# # # # # # # # # # # # # #     để ShortRangeHead không học trên nhiễu.
# # # # # # # # # # # # # #     """
# # # # # # # # # # # # # #     frozen_keys = ["spatial_enc", "enc_1d", "env_enc",
# # # # # # # # # # # # # #                    "ctx_fc1", "ctx_ln"]   # FIX-T-C: thêm ctx_fc1, ctx_ln
# # # # # # # # # # # # # #     for name, param in model.named_parameters():
# # # # # # # # # # # # # #         if any(k in name for k in frozen_keys):
# # # # # # # # # # # # # #             param.requires_grad_(False)
# # # # # # # # # # # # # #     n_frozen = sum(not p.requires_grad for p in model.parameters())
# # # # # # # # # # # # # #     print(f"  [Phase-1] Frozen {n_frozen:,} params (backbone + ctx_fc1)")


# # # # # # # # # # # # # # def unfreeze_all(model: TCFlowMatching) -> None:
# # # # # # # # # # # # # #     for param in model.parameters():
# # # # # # # # # # # # # #         param.requires_grad_(True)
# # # # # # # # # # # # # #     print(f"  [Phase-2] Unfrozen all params")

# # # # # # # # # # # # # # # Trong freeze_backbone — thêm cả prefix _orig_mod:
# # # # # # # # # # # # # # def freeze_backbone(model: TCFlowMatching) -> None:
# # # # # # # # # # # # # #     frozen_keys = [
# # # # # # # # # # # # # #         "spatial_enc", "enc_1d", "env_enc", "ctx_fc1", "ctx_ln",
# # # # # # # # # # # # # #         # Sau torch.compile tên có thể có prefix:
# # # # # # # # # # # # # #         "_orig_mod.net.spatial_enc", "_orig_mod.net.enc_1d",
# # # # # # # # # # # # # #         "_orig_mod.net.env_enc",     "_orig_mod.net.ctx_fc1",
# # # # # # # # # # # # # #         "_orig_mod.net.ctx_ln",
# # # # # # # # # # # # # #         # Hoặc prefix net.:
# # # # # # # # # # # # # #         "net.spatial_enc", "net.enc_1d", "net.env_enc",
# # # # # # # # # # # # # #         "net.ctx_fc1",     "net.ctx_ln",
# # # # # # # # # # # # # #     ]
# # # # # # # # # # # # # #     n_frozen = 0
# # # # # # # # # # # # # #     for name, param in model.named_parameters():
# # # # # # # # # # # # # #         if any(k in name for k in frozen_keys):
# # # # # # # # # # # # # #             param.requires_grad_(False)
# # # # # # # # # # # # # #             n_frozen += param.numel()
# # # # # # # # # # # # # #     n_tensors = sum(not p.requires_grad for p in model.parameters())
# # # # # # # # # # # # # #     print(f"  [Phase-1] Frozen {n_tensors} tensors / {n_frozen:,} params")
# # # # # # # # # # # # # #     # Sanity check
# # # # # # # # # # # # # #     if n_frozen < 1_000_000:
# # # # # # # # # # # # # #         print(f"  ⚠️  WARNING: chỉ frozen {n_frozen:,} params — kiểm tra tên lại!")
# # # # # # # # # # # # # #         print("  Tên params thực tế (10 đầu):")
# # # # # # # # # # # # # #         for name, _ in list(model.named_parameters())[:10]:
# # # # # # # # # # # # # #             print(f"    {name}")

# # # # # # # # # # # # # def freeze_backbone(model):
# # # # # # # # # # # # #     frozen_keys = ["spatial_enc", "enc_1d", "env_enc", "ctx_fc1", "ctx_ln",
# # # # # # # # # # # # #                    "net.spatial_enc", "net.enc_1d", "net.env_enc", "net.ctx_fc1", "net.ctx_ln"]
# # # # # # # # # # # # #     n_frozen_params = 0
# # # # # # # # # # # # #     for name, param in model.named_parameters():
# # # # # # # # # # # # #         if any(k in name for k in frozen_keys):
# # # # # # # # # # # # #             param.requires_grad_(False)
# # # # # # # # # # # # #             n_frozen_params += param.numel()
    
# # # # # # # # # # # # #     # Debug: list frozen modules
# # # # # # # # # # # # #     frozen_modules = set()
# # # # # # # # # # # # #     for name, param in model.named_parameters():
# # # # # # # # # # # # #         if not param.requires_grad:
# # # # # # # # # # # # #             top = name.split('.')[0] if '.' in name else name
# # # # # # # # # # # # #             frozen_modules.add(top)
# # # # # # # # # # # # #     print(f"  [Phase-1] Frozen {n_frozen_params:,} / {sum(p.numel() for p in model.parameters()):,} params")
# # # # # # # # # # # # #     print(f"  Frozen modules: {sorted(frozen_modules)}")
    
# # # # # # # # # # # # #     if n_frozen_params < 2_000_000:
# # # # # # # # # # # # #         print("  ⚠️  WARNING: Expected >2M frozen params for backbone+ctx!")


# # # # # # # # # # # # # def unfreeze_all(model):
# # # # # # # # # # # # #     """Phase 2: unfreeze all parameters."""
# # # # # # # # # # # # #     for param in model.parameters():
# # # # # # # # # # # # #         param.requires_grad_(True)
# # # # # # # # # # # # #     print(f"  [Phase-2] Unfrozen all params")
# # # # # # # # # # # # # # ── Weight schedules ──────────────────────────────────────────────────────────

# # # # # # # # # # # # # def get_short_range_weight(epoch: int) -> float:
# # # # # # # # # # # # #     """
# # # # # # # # # # # # #     FIX-T-D: Scale nhỏ hơn vì short_range_regression_loss đã normalize đúng
# # # # # # # # # # # # #     (đơn vị km thực tế, không chia HUBER_DELTA).
# # # # # # # # # # # # #     """
# # # # # # # # # # # # #     if epoch < 15:
# # # # # # # # # # # # #         return 5.0    # FIX-T-D: 12.0 → 5.0
# # # # # # # # # # # # #     elif epoch < 30:
# # # # # # # # # # # # #         return 3.0    # FIX-T-D: 8.0 → 3.0
# # # # # # # # # # # # #     elif epoch < 60:
# # # # # # # # # # # # #         return 2.0    # FIX-T-D: 5.0 → 2.0
# # # # # # # # # # # # #     else:
# # # # # # # # # # # # #         return 1.5    # FIX-T-D: 3.0 → 1.5


# # # # # # # # # # # # # # def get_bridge_weight(epoch: int) -> float:
# # # # # # # # # # # # # #     """
# # # # # # # # # # # # # #     FIX-T-E: Bridge loss từ epoch 10.
# # # # # # # # # # # # # #     ep 0-9:  0.0 (SR và FM chưa đủ ổn định để enforce nhất quán)
# # # # # # # # # # # # # #     ep 10+:  0.5
# # # # # # # # # # # # # #     """
# # # # # # # # # # # # # #     return 0.0 if epoch < 10 else 0.5

# # # # # # # # # # # # # def get_bridge_weight(epoch: int) -> float:
# # # # # # # # # # # # #     if epoch < 10: return 0.0
# # # # # # # # # # # # #     if epoch < 40: return 0.5
# # # # # # # # # # # # #     return 1.5  # Tăng mạnh weight ở cuối để ép FM phải bám sát SR Head tại mốc 24h

# # # # # # # # # # # # # def get_fm_weight(epoch: int) -> float:
# # # # # # # # # # # # #     if epoch < 30:
# # # # # # # # # # # # #         return 0.5   # Phase 1: FM nhỏ, focus SR
# # # # # # # # # # # # #     return 2.0


# # # # # # # # # # # # # def get_spread_weight(epoch: int) -> float:
# # # # # # # # # # # # #     if epoch < 30:
# # # # # # # # # # # # #         return 1.2
# # # # # # # # # # # # #     return 0.8


# # # # # # # # # # # # # def get_pinn_weight(epoch: int, warmup_epochs: int = 80,
# # # # # # # # # # # # #                     w_start: float = 0.001, w_end: float = 0.05) -> float:
# # # # # # # # # # # # #     """
# # # # # # # # # # # # #     FIX-T-A: weight nhỏ trong phase 1, tăng dần trong phase 2.
# # # # # # # # # # # # #     Note: AdaptClamp và adaptive BVE weighting giờ được handle bên trong
# # # # # # # # # # # # #     pinn_bve_loss() qua epoch parameter.
# # # # # # # # # # # # #     """
# # # # # # # # # # # # #     if epoch < 30:
# # # # # # # # # # # # #         return w_start
# # # # # # # # # # # # #     elif epoch < warmup_epochs:
# # # # # # # # # # # # #         t = (epoch - 30) / max(warmup_epochs - 30, 1)
# # # # # # # # # # # # #         return 0.005 + t * (w_end - 0.005)
# # # # # # # # # # # # #     return w_end


# # # # # # # # # # # # # def get_velocity_weight(epoch, warmup_epochs=20, w_start=0.5, w_end=1.5):
# # # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # # #         return w_end
# # # # # # # # # # # # #     return w_start + (epoch / max(warmup_epochs - 1, 1)) * (w_end - w_start)


# # # # # # # # # # # # # def get_recurv_weight(epoch, warmup_epochs=10, w_start=0.3, w_end=1.0):
# # # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # # #         return w_end
# # # # # # # # # # # # #     return w_start + (epoch / max(warmup_epochs - 1, 1)) * (w_end - w_start)


# # # # # # # # # # # # # def get_grad_clip(epoch, warmup_epochs=20, clip_start=2.0, clip_end=1.0):
# # # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # # #         return clip_end
# # # # # # # # # # # # #     return clip_start - (epoch / max(warmup_epochs - 1, 1)) * (clip_start - clip_end)


# # # # # # # # # # # # # def get_step_weight_alpha(epoch: int, decay_epochs: int = 60) -> float:
# # # # # # # # # # # # #     if epoch >= decay_epochs:
# # # # # # # # # # # # #         return 0.0
# # # # # # # # # # # # #     return 1.0 - (epoch / decay_epochs)


# # # # # # # # # # # # # def get_progressive_ens(epoch: int, n_train_ens: int = 6) -> int:
# # # # # # # # # # # # #     if epoch < 20:
# # # # # # # # # # # # #         return 1
# # # # # # # # # # # # #     elif epoch < 50:
# # # # # # # # # # # # #         return 2
# # # # # # # # # # # # #     else:
# # # # # # # # # # # # #         return n_train_ens


# # # # # # # # # # # # # # ── FIX-T-B: compute_sr_ade với cached raw_ctx ───────────────────────────────

# # # # # # # # # # # # # @torch.no_grad()
# # # # # # # # # # # # # def compute_sr_ade_from_ctx(
# # # # # # # # # # # # #     model: TCFlowMatching,
# # # # # # # # # # # # #     batch_list: list,
# # # # # # # # # # # # #     raw_ctx: torch.Tensor,
# # # # # # # # # # # # #     device,
# # # # # # # # # # # # # ) -> tuple:
# # # # # # # # # # # # #     """
# # # # # # # # # # # # #     FIX-T-B: Nhận raw_ctx đã compute, tránh tính lại.
# # # # # # # # # # # # #     Returns (ade_12h_km, ade_24h_km).
# # # # # # # # # # # # #     """
# # # # # # # # # # # # #     obs_t   = batch_list[0]
# # # # # # # # # # # # #     traj_gt = batch_list[1]

# # # # # # # # # # # # #     sr_pred = model.net.forward_short_range(obs_t, raw_ctx)   # [4, B, 2]

# # # # # # # # # # # # #     pred_01 = sr_pred.clone()
# # # # # # # # # # # # #     pred_01[..., 0] = sr_pred[..., 0] * 50.0 + 1800.0
# # # # # # # # # # # # #     pred_01[..., 1] = sr_pred[..., 1] * 50.0

# # # # # # # # # # # # #     gt_01 = traj_gt.clone()
# # # # # # # # # # # # #     gt_01[..., 0] = traj_gt[..., 0] * 50.0 + 1800.0
# # # # # # # # # # # # #     gt_01[..., 1] = traj_gt[..., 1] * 50.0

# # # # # # # # # # # # #     ade_12h = haversine_km_torch(pred_01[1], gt_01[1]).mean().item()
# # # # # # # # # # # # #     ade_24h = haversine_km_torch(pred_01[3], gt_01[3]).mean().item() \
# # # # # # # # # # # # #               if pred_01.shape[0] >= 4 else float("nan")
# # # # # # # # # # # # #     return ade_12h, ade_24h


# # # # # # # # # # # # # @torch.no_grad()
# # # # # # # # # # # # # def compute_sr_ade(model, batch_list, device):
# # # # # # # # # # # # #     """Wrapper không có cached ctx (dùng trong evaluate_fast)."""
# # # # # # # # # # # # #     raw_ctx = model.net._context(batch_list)
# # # # # # # # # # # # #     return compute_sr_ade_from_ctx(model, batch_list, raw_ctx, device)


# # # # # # # # # # # # # # ── evaluate_fast ─────────────────────────────────────────────────────────────

# # # # # # # # # # # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
# # # # # # # # # # # # #     model.eval()
# # # # # # # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # # # # # # #     t0  = time.perf_counter()
# # # # # # # # # # # # #     n   = 0
# # # # # # # # # # # # #     spread_per_step = []
# # # # # # # # # # # # #     sr_12h_buf, sr_24h_buf = [], []

# # # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # # #         for batch in loader:
# # # # # # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # # # # # #             pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
# # # # # # # # # # # # #                                               ddim_steps=ode_steps)
# # # # # # # # # # # # #             T_active  = pred.shape[0]
# # # # # # # # # # # # #             gt_sliced = bl[1][:T_active]
# # # # # # # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
# # # # # # # # # # # # #             acc.update(dist)

# # # # # # # # # # # # #             # FIX-T-B: SR ADE từ model.sample() đã compute raw_ctx
# # # # # # # # # # # # #             # Dùng sr_pred từ sample() thay vì gọi lại _context()
# # # # # # # # # # # # #             # Vì sample() đã cache raw_ctx, compute thêm SR không tốn compute
# # # # # # # # # # # # #             a12, a24 = compute_sr_ade(model, bl, device)
# # # # # # # # # # # # #             sr_12h_buf.append(a12)
# # # # # # # # # # # # #             sr_24h_buf.append(a24)

# # # # # # # # # # # # #             step_spreads = []
# # # # # # # # # # # # #             for t in range(all_trajs.shape[1]):
# # # # # # # # # # # # #                 step_data = all_trajs[:, t, :, :]
# # # # # # # # # # # # #                 std_lon = step_data[:, :, 0].std(0)
# # # # # # # # # # # # #                 std_lat = step_data[:, :, 1].std(0)
# # # # # # # # # # # # #                 spread = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# # # # # # # # # # # # #                 step_spreads.append(spread)
# # # # # # # # # # # # #             spread_per_step.append(step_spreads)
# # # # # # # # # # # # #             n += 1

# # # # # # # # # # # # #     r = acc.compute()
# # # # # # # # # # # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)

# # # # # # # # # # # # #     if spread_per_step:
# # # # # # # # # # # # #         spreads = np.array(spread_per_step)
# # # # # # # # # # # # #         r["spread_12h_km"] = float(spreads[:, 1].mean()) if spreads.shape[1] > 1 else 0.0
# # # # # # # # # # # # #         r["spread_24h_km"] = float(spreads[:, 3].mean()) if spreads.shape[1] > 3 else 0.0
# # # # # # # # # # # # #         r["spread_72h_km"] = float(spreads[:, -1].mean())

# # # # # # # # # # # # #     r["sr_ade_12h"] = float(np.mean(sr_12h_buf)) if sr_12h_buf else float("nan")
# # # # # # # # # # # # #     r["sr_ade_24h"] = float(np.mean(sr_24h_buf)) if sr_24h_buf else float("nan")
# # # # # # # # # # # # #     return r


# # # # # # # # # # # # # # ── evaluate_full_val_ade (FIX-T-B) ──────────────────────────────────────────

# # # # # # # # # # # # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # # # # # # # # # # # #                            fast_ensemble, metrics_csv, epoch, tag=""):
# # # # # # # # # # # # #     """
# # # # # # # # # # # # #     FIX-T-B: compute raw_ctx sekali, dùng untuk SR ADE.
# # # # # # # # # # # # #     Tidak compute _context() dua kali per batch.
# # # # # # # # # # # # #     """
# # # # # # # # # # # # #     model.eval()
# # # # # # # # # # # # #     acc       = StepErrorAccumulator(pred_len)
# # # # # # # # # # # # #     t0        = time.perf_counter()
# # # # # # # # # # # # #     n_batch   = 0
# # # # # # # # # # # # #     pinn_buf  = []
# # # # # # # # # # # # #     sr_12h_buf, sr_24h_buf = [], []

# # # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # # #         for batch in val_loader:
# # # # # # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # # # # # #             # sample() sudah compute raw_ctx sekali
# # # # # # # # # # # # #             pred, _, _ = model.sample(bl, num_ensemble=fast_ensemble,
# # # # # # # # # # # # #                                        ddim_steps=ode_steps)
# # # # # # # # # # # # #             T_pred = pred.shape[0]
# # # # # # # # # # # # #             gt     = bl[1][:T_pred]
# # # # # # # # # # # # #             dist   = haversine_km_torch(denorm_torch(pred), denorm_torch(gt))
# # # # # # # # # # # # #             acc.update(dist)

# # # # # # # # # # # # #             # FIX-T-B: compute SR ADE — raw_ctx tính thêm 1 lần
# # # # # # # # # # # # #             # Nhưng đây là evaluation loop riêng, không thể tránh hoàn toàn
# # # # # # # # # # # # #             # Optimize: lấy sr_pred từ pred_mean[:4] thay vì gọi lại
# # # # # # # # # # # # #             sr_pred_from_mean = pred[:ShortRangeHead.N_STEPS]  # [4, B, 2]
# # # # # # # # # # # # #             gt_sr = bl[1][:ShortRangeHead.N_STEPS]

# # # # # # # # # # # # #             # Convert to 0.1° units
# # # # # # # # # # # # #             sr_01 = sr_pred_from_mean.clone()
# # # # # # # # # # # # #             sr_01[..., 0] = sr_pred_from_mean[..., 0] * 50.0 + 1800.0
# # # # # # # # # # # # #             sr_01[..., 1] = sr_pred_from_mean[..., 1] * 50.0
# # # # # # # # # # # # #             gt_01 = gt_sr.clone()
# # # # # # # # # # # # #             gt_01[..., 0] = gt_sr[..., 0] * 50.0 + 1800.0
# # # # # # # # # # # # #             gt_01[..., 1] = gt_sr[..., 1] * 50.0

# # # # # # # # # # # # #             ade_12h = haversine_km_torch(sr_01[1], gt_01[1]).mean().item()
# # # # # # # # # # # # #             ade_24h = haversine_km_torch(sr_01[3], gt_01[3]).mean().item() \
# # # # # # # # # # # # #                       if sr_01.shape[0] >= 4 else float("nan")
# # # # # # # # # # # # #             sr_12h_buf.append(ade_12h)
# # # # # # # # # # # # #             sr_24h_buf.append(ade_24h)

# # # # # # # # # # # # #             # PINN validation
# # # # # # # # # # # # #             try:
# # # # # # # # # # # # #                 from Model.losses import pinn_bve_loss
# # # # # # # # # # # # #                 pred_deg = pred.clone()
# # # # # # # # # # # # #                 pred_deg[..., 0] = (pred[..., 0] * 50.0 + 1800.0) / 10.0
# # # # # # # # # # # # #                 pred_deg[..., 1] = (pred[..., 1] * 50.0) / 10.0
# # # # # # # # # # # # #                 env_d = bl[13] if len(bl) > 13 else None
# # # # # # # # # # # # #                 pinn_val = pinn_bve_loss(
# # # # # # # # # # # # #                     pred_deg, bl, env_data=env_d, epoch=epoch).item()
# # # # # # # # # # # # #                 pinn_buf.append(pinn_val)
# # # # # # # # # # # # #             except Exception:
# # # # # # # # # # # # #                 pass

# # # # # # # # # # # # #             n_batch += 1

# # # # # # # # # # # # #     elapsed  = time.perf_counter() - t0
# # # # # # # # # # # # #     r        = acc.compute()
# # # # # # # # # # # # #     sr_12h   = float(np.mean(sr_12h_buf)) if sr_12h_buf else float("nan")
# # # # # # # # # # # # #     sr_24h   = float(np.mean(sr_24h_buf)) if sr_24h_buf else float("nan")
# # # # # # # # # # # # #     pinn_mean = f"{np.mean(pinn_buf):.3f}" if pinn_buf else "N/A"

# # # # # # # # # # # # #     print(f"\n{'='*64}")
# # # # # # # # # # # # #     print(f"  [FULL VAL ADE  ep={epoch}  {elapsed:.0f}s  {n_batch} batches]")
# # # # # # # # # # # # #     print(f"  ADE (blend) = {r.get('ADE', float('nan')):.1f} km  "
# # # # # # # # # # # # #           f"FDE = {r.get('FDE', float('nan')):.1f} km")
# # # # # # # # # # # # #     print(f"  12h={r.get('12h', float('nan')):.0f}  "
# # # # # # # # # # # # #           f"24h={r.get('24h', float('nan')):.0f}  "
# # # # # # # # # # # # #           f"48h={r.get('48h', float('nan')):.0f}  "
# # # # # # # # # # # # #           f"72h={r.get('72h', float('nan')):.0f} km")
# # # # # # # # # # # # #     print(f"  ── ShortRangeHead (từ blend pred_mean) ──")
# # # # # # # # # # # # #     print(f"  SR-12h={sr_12h:.1f} km  SR-24h={sr_24h:.1f} km  "
# # # # # # # # # # # # #           f"[TARGET: <50 / <100 km]")
# # # # # # # # # # # # #     print(f"  PINN_mean={pinn_mean}  samples={r.get('n_samples',0)}  "
# # # # # # # # # # # # #           f"ens={fast_ensemble}  steps={ode_steps}")
# # # # # # # # # # # # #     print(f"{'='*64}\n")

# # # # # # # # # # # # #     r["sr_12h"] = sr_12h
# # # # # # # # # # # # #     r["sr_24h"] = sr_24h

# # # # # # # # # # # # #     from datetime import datetime
# # # # # # # # # # # # #     from utils.metrics import DatasetMetrics, save_metrics_csv as _save_csv
# # # # # # # # # # # # #     dm = DatasetMetrics(
# # # # # # # # # # # # #         ade      = r.get("ADE",  float("nan")),
# # # # # # # # # # # # #         fde      = r.get("FDE",  float("nan")),
# # # # # # # # # # # # #         ugde_12h = r.get("12h",  float("nan")),
# # # # # # # # # # # # #         ugde_24h = r.get("24h",  float("nan")),
# # # # # # # # # # # # #         ugde_48h = r.get("48h",  float("nan")),
# # # # # # # # # # # # #         ugde_72h = r.get("72h",  float("nan")),
# # # # # # # # # # # # #         n_total  = r.get("n_samples", 0),
# # # # # # # # # # # # #         timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # # # # # # # # #     )
# # # # # # # # # # # # #     _save_csv(dm, metrics_csv, tag=tag or f"val_full_ep{epoch:03d}")
# # # # # # # # # # # # #     return r


# # # # # # # # # # # # # def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
# # # # # # # # # # # # #                   metrics_csv, tag="", predict_csv=""):
# # # # # # # # # # # # #     model.eval()
# # # # # # # # # # # # #     cliper_step_errors = []
# # # # # # # # # # # # #     ev = TCEvaluator(pred_len=pred_len, compute_dtw=True)
# # # # # # # # # # # # #     obs_seqs_01 = []; gt_seqs_01 = []; pred_seqs_01 = []; ens_seqs_01 = []

# # # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # # #         for batch in loader:
# # # # # # # # # # # # #             bl  = move(list(batch), device)
# # # # # # # # # # # # #             gt  = bl[1]; obs = bl[0]
# # # # # # # # # # # # #             pred_mean, _, all_trajs = model.sample(
# # # # # # # # # # # # #                 bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
# # # # # # # # # # # # #                 predict_csv=predict_csv if predict_csv else None)

# # # # # # # # # # # # #             pd_np = denorm_torch(pred_mean).cpu().numpy()
# # # # # # # # # # # # #             gd_np = denorm_torch(gt).cpu().numpy()
# # # # # # # # # # # # #             od_np = denorm_torch(obs).cpu().numpy()
# # # # # # # # # # # # #             ed_np = denorm_torch(all_trajs).cpu().numpy()

# # # # # # # # # # # # #             B = pd_np.shape[1]
# # # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # # #                 ens_b = ed_np[:, :, b, :]
# # # # # # # # # # # # #                 ev.update(pd_np[:, b, :], gd_np[:, b, :], pred_ens=ens_b)
# # # # # # # # # # # # #                 obs_seqs_01.append(od_np[:, b, :])
# # # # # # # # # # # # #                 gt_seqs_01.append(gd_np[:, b, :])
# # # # # # # # # # # # #                 pred_seqs_01.append(pd_np[:, b, :])
# # # # # # # # # # # # #                 ens_seqs_01.append(ens_b)

# # # # # # # # # # # # #                 obs_b_norm = obs.cpu().numpy()[:, b, :]
# # # # # # # # # # # # #                 cliper_errors_b = np.zeros(pred_len)
# # # # # # # # # # # # #                 for h in range(pred_len):
# # # # # # # # # # # # #                     pred_cliper_norm = cliper_forecast(obs_b_norm, h + 1)
# # # # # # # # # # # # #                     pred_cliper_01   = denorm_np(pred_cliper_norm[np.newaxis])
# # # # # # # # # # # # #                     gt_01_np         = gd_np[h, b, :][np.newaxis]
# # # # # # # # # # # # #                     from utils.metrics import haversine_km_np
# # # # # # # # # # # # #                     cliper_errors_b[h] = float(
# # # # # # # # # # # # #                         haversine_km_np(pred_cliper_01, gt_01_np, unit_01deg=True)[0])
# # # # # # # # # # # # #                 cliper_step_errors.append(cliper_errors_b)

# # # # # # # # # # # # #     if cliper_step_errors:
# # # # # # # # # # # # #         cliper_mat       = np.stack(cliper_step_errors)
# # # # # # # # # # # # #         cliper_ugde_dict = {h: float(cliper_mat[:, s].mean())
# # # # # # # # # # # # #                             for h, s in HORIZON_STEPS.items()
# # # # # # # # # # # # #                             if s < cliper_mat.shape[1]}
# # # # # # # # # # # # #         ev.cliper_ugde = cliper_ugde_dict

# # # # # # # # # # # # #     dm = ev.compute(tag=tag)

# # # # # # # # # # # # #     try:
# # # # # # # # # # # # #         if LANDFALL_TARGETS and ens_seqs_01:
# # # # # # # # # # # # #             bss_vals = []
# # # # # # # # # # # # #             step_72 = HORIZON_STEPS.get(72, pred_len - 1)
# # # # # # # # # # # # #             for tname, t_lon, t_lat in LANDFALL_TARGETS:
# # # # # # # # # # # # #                 bv = brier_skill_score(
# # # # # # # # # # # # #                     ens_seqs_01, gt_seqs_01, min(step_72, pred_len-1),
# # # # # # # # # # # # #                     (t_lon * 10.0, t_lat * 10.0), LANDFALL_RADIUS_KM)
# # # # # # # # # # # # #                 if not math.isnan(bv):
# # # # # # # # # # # # #                     bss_vals.append(bv)
# # # # # # # # # # # # #             if bss_vals:
# # # # # # # # # # # # #                 dm.bss_mean = float(np.mean(bss_vals))
# # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # #         print(f"  ⚠  BSS failed: {e}")

# # # # # # # # # # # # #     save_metrics_csv(dm, metrics_csv, tag=tag)
# # # # # # # # # # # # #     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# # # # # # # # # # # # # # ── BestModelSaver ────────────────────────────────────────────────────────────

# # # # # # # # # # # # # class BestModelSaver:
# # # # # # # # # # # # #     def __init__(self, patience: int = 15, sr_tol: float = 3.0,
# # # # # # # # # # # # #                  ade_tol: float = 5.0):
# # # # # # # # # # # # #         self.patience      = patience
# # # # # # # # # # # # #         self.sr_tol        = sr_tol
# # # # # # # # # # # # #         self.ade_tol       = ade_tol
# # # # # # # # # # # # #         self.best_sr12h    = float("inf")
# # # # # # # # # # # # #         self.best_ade      = float("inf")
# # # # # # # # # # # # #         self.best_val_loss = float("inf")
# # # # # # # # # # # # #         self.counter_12h   = 0
# # # # # # # # # # # # #         self.counter_loss  = 0
# # # # # # # # # # # # #         self.early_stop    = False

# # # # # # # # # # # # #     def reset_counters(self, reason: str = "") -> None:
# # # # # # # # # # # # #         self.counter_12h  = 0
# # # # # # # # # # # # #         self.counter_loss = 0
# # # # # # # # # # # # #         if reason:
# # # # # # # # # # # # #             print(f"  [SAVER] Patience reset: {reason}")

# # # # # # # # # # # # #     def update_val_loss(self, val_loss, model, out_dir, epoch, optimizer, tl):
# # # # # # # # # # # # #         if val_loss < self.best_val_loss - 1e-4:
# # # # # # # # # # # # #             self.best_val_loss = val_loss
# # # # # # # # # # # # #             self.counter_loss  = 0
# # # # # # # # # # # # #             torch.save(dict(
# # # # # # # # # # # # #                 epoch=epoch, model_state_dict=model.state_dict(),
# # # # # # # # # # # # #                 optimizer_state=optimizer.state_dict(),
# # # # # # # # # # # # #                 train_loss=tl, val_loss=val_loss,
# # # # # # # # # # # # #                 model_version="v25-valloss"),
# # # # # # # # # # # # #                 os.path.join(out_dir, "best_model_valloss.pth"))
# # # # # # # # # # # # #         else:
# # # # # # # # # # # # #             self.counter_loss += 1

# # # # # # # # # # # # #     def update_full_val(self, result: dict, model, out_dir, epoch,
# # # # # # # # # # # # #                         optimizer, tl, vl, min_epochs: int = 80):
# # # # # # # # # # # # #         sr_12h    = result.get("sr_12h", float("inf"))
# # # # # # # # # # # # #         blend_ade = result.get("ADE",    float("inf"))

# # # # # # # # # # # # #         improved = sr_12h < self.best_sr12h - self.sr_tol
# # # # # # # # # # # # #         if improved:
# # # # # # # # # # # # #             self.best_sr12h = sr_12h
# # # # # # # # # # # # #             self.best_ade   = min(self.best_ade, blend_ade)
# # # # # # # # # # # # #             self.counter_12h = 0
# # # # # # # # # # # # #             torch.save(dict(
# # # # # # # # # # # # #                 epoch=epoch, model_state_dict=model.state_dict(),
# # # # # # # # # # # # #                 optimizer_state=optimizer.state_dict(),
# # # # # # # # # # # # #                 train_loss=tl, val_loss=vl,
# # # # # # # # # # # # #                 sr_ade_12h=sr_12h, blend_ade=blend_ade,
# # # # # # # # # # # # #                 model_version="v25-SR+FM"),
# # # # # # # # # # # # #                 os.path.join(out_dir, "best_model.pth"))
# # # # # # # # # # # # #             print(f"  ✅ Best SR-12h {sr_12h:.1f} km  "
# # # # # # # # # # # # #                   f"blend-ADE {blend_ade:.1f} km  (epoch {epoch})")
# # # # # # # # # # # # #         else:
# # # # # # # # # # # # #             self.counter_12h += 1
# # # # # # # # # # # # #             print(f"  No SR-12h improvement {self.counter_12h}/{self.patience}"
# # # # # # # # # # # # #                   f"  (Δ={self.best_sr12h - sr_12h:.1f} km < tol={self.sr_tol} km)"
# # # # # # # # # # # # #                   f"  | Loss counter {self.counter_loss}/{self.patience}"
# # # # # # # # # # # # #                   f"  | SR-12h={sr_12h:.1f}  blend={blend_ade:.1f}")

# # # # # # # # # # # # #         if epoch >= min_epochs:
# # # # # # # # # # # # #             if (self.counter_12h >= self.patience
# # # # # # # # # # # # #                     and self.counter_loss >= self.patience):
# # # # # # # # # # # # #                 self.early_stop = True
# # # # # # # # # # # # #                 print(f"  ⛔ Early stop @ epoch {epoch}")
# # # # # # # # # # # # #         else:
# # # # # # # # # # # # #             if (self.counter_12h >= self.patience
# # # # # # # # # # # # #                     and self.counter_loss >= self.patience):
# # # # # # # # # # # # #                 print(f"  ⚠  Would stop but min_epochs={min_epochs} not reached.")
# # # # # # # # # # # # #                 self.counter_12h  = 0
# # # # # # # # # # # # #                 self.counter_loss = 0

# # # # # # # # # # # # #     def log_subset_sr(self, sr_12h: float, sr_24h: float, epoch: int):
# # # # # # # # # # # # #         print(f"  [SUBSET SR ep{epoch}]  12h={sr_12h:.1f}  24h={sr_24h:.1f} km"
# # # # # # # # # # # # #               f"  [target <50/<100]  (monitor only)")


# # # # # # # # # # # # # # ── GPH/UV diagnostic checks ──────────────────────────────────────────────────

# # # # # # # # # # # # # def _check_gph500(bl, train_dataset):
# # # # # # # # # # # # #     try:
# # # # # # # # # # # # #         env_dir = train_dataset.env_path
# # # # # # # # # # # # #     except AttributeError:
# # # # # # # # # # # # #         try:
# # # # # # # # # # # # #             env_dir = train_dataset.dataset.env_path
# # # # # # # # # # # # #         except AttributeError:
# # # # # # # # # # # # #             env_dir = "UNKNOWN"
# # # # # # # # # # # # #     print(f"  Env path   : {env_dir}  exists={os.path.exists(env_dir) if env_dir != 'UNKNOWN' else 'N/A'}")

# # # # # # # # # # # # #     env_data = bl[13]
# # # # # # # # # # # # #     if env_data is None:
# # # # # # # # # # # # #         print("  ⚠️  env_data is None"); return

# # # # # # # # # # # # #     for key, expected_range in [
# # # # # # # # # # # # #         ("gph500_mean",   (-3.0, 3.0)),
# # # # # # # # # # # # #         ("gph500_center", (-3.0, 3.0)),
# # # # # # # # # # # # #     ]:
# # # # # # # # # # # # #         if key not in env_data:
# # # # # # # # # # # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # # # # # # # # # # #         v    = env_data[key]
# # # # # # # # # # # # #         mn   = v.mean().item()
# # # # # # # # # # # # #         std  = v.std().item()
# # # # # # # # # # # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # # # # # # # # # # #         lo, hi = expected_range
# # # # # # # # # # # # #         if zero > 80.0:
# # # # # # # # # # # # #             print(f"  ⚠️  {key}: zero={zero:.1f}% → env missing!")
# # # # # # # # # # # # #         elif lo <= mn <= hi:
# # # # # # # # # # # # #             print(f"  ✅ {key}: mean={mn:.3f} std={std:.3f} zero={zero:.1f}%")
# # # # # # # # # # # # #         else:
# # # # # # # # # # # # #             print(f"  ⚠️  {key}: mean={mn:.3f} ngoài range [{lo},{hi}]")


# # # # # # # # # # # # # def _check_uv500(bl):
# # # # # # # # # # # # #     env_data = bl[13]
# # # # # # # # # # # # #     if env_data is None: return
# # # # # # # # # # # # #     for key in ("u500_mean", "v500_mean"):
# # # # # # # # # # # # #         if key not in env_data:
# # # # # # # # # # # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # # # # # # # # # # #         v    = env_data[key]
# # # # # # # # # # # # #         mn   = v.mean().item()
# # # # # # # # # # # # #         std  = v.std().item()
# # # # # # # # # # # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # # # # # # # # # # #         if zero > 80.0:
# # # # # # # # # # # # #             print(f"  ⚠️  {key}: zero={zero:.1f}% → u/v500 missing!")
# # # # # # # # # # # # #         else:
# # # # # # # # # # # # #             print(f"  ✅ {key}: mean={mn:.4f} std={std:.4f} zero={zero:.1f}%")


# # # # # # # # # # # # # def _load_baseline_errors(path, name):
# # # # # # # # # # # # #     if path is None:
# # # # # # # # # # # # #         print(f"\n  ⚠  {name} errors not provided.\n"); return None
# # # # # # # # # # # # #     if not os.path.exists(path):
# # # # # # # # # # # # #         print(f"\n  ⚠  {path} not found.\n"); return None
# # # # # # # # # # # # #     arr = np.load(path)
# # # # # # # # # # # # #     print(f"  ✓  Loaded {name}: {arr.shape}")
# # # # # # # # # # # # #     return arr


# # # # # # # # # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # # # # # # # # def get_args():
# # # # # # # # # # # # #     p = argparse.ArgumentParser(
# # # # # # # # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # # # # # # # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # # # # # # # # # # # #     p.add_argument("--obs_len",         default=8,          type=int)
# # # # # # # # # # # # #     p.add_argument("--pred_len",        default=12,         type=int)
# # # # # # # # # # # # #     p.add_argument("--test_year",       default=None,       type=int)
# # # # # # # # # # # # #     p.add_argument("--batch_size",      default=32,         type=int)
# # # # # # # # # # # # #     p.add_argument("--num_epochs",      default=200,        type=int)
# # # # # # # # # # # # #     p.add_argument("--g_learning_rate", default=2e-4,       type=float)
# # # # # # # # # # # # #     p.add_argument("--phase1_lr",       default=5e-4,       type=float)
# # # # # # # # # # # # #     p.add_argument("--phase2_start",    default=30,         type=int)
# # # # # # # # # # # # #     p.add_argument("--weight_decay",    default=1e-4,       type=float)
# # # # # # # # # # # # #     p.add_argument("--warmup_epochs",   default=3,          type=int)
# # # # # # # # # # # # #     p.add_argument("--grad_clip",       default=2.0,        type=float)
# # # # # # # # # # # # #     p.add_argument("--grad_accum",      default=2,          type=int)
# # # # # # # # # # # # #     p.add_argument("--patience",        default=15,         type=int)
# # # # # # # # # # # # #     p.add_argument("--min_epochs",      default=80,         type=int)
# # # # # # # # # # # # #     p.add_argument("--n_train_ens",     default=6,          type=int)
# # # # # # # # # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # # # # # # # # #     p.add_argument("--num_workers",     default=2,          type=int)

# # # # # # # # # # # # #     p.add_argument("--sigma_min",            default=0.02,  type=float)
# # # # # # # # # # # # #     p.add_argument("--ctx_noise_scale",      default=0.002, type=float)
# # # # # # # # # # # # #     p.add_argument("--initial_sample_sigma", default=0.03,  type=float)

# # # # # # # # # # # # #     p.add_argument("--ode_steps_train", default=20,  type=int)
# # # # # # # # # # # # #     p.add_argument("--ode_steps_val",   default=30,  type=int)
# # # # # # # # # # # # #     p.add_argument("--ode_steps_test",  default=50,  type=int)
# # # # # # # # # # # # #     p.add_argument("--ode_steps",       default=None, type=int)

# # # # # # # # # # # # #     p.add_argument("--val_ensemble",    default=30,  type=int)
# # # # # # # # # # # # #     p.add_argument("--fast_ensemble",   default=8,   type=int)

# # # # # # # # # # # # #     p.add_argument("--fno_modes_h",      default=4,  type=int)
# # # # # # # # # # # # #     p.add_argument("--fno_modes_t",      default=4,  type=int)
# # # # # # # # # # # # #     p.add_argument("--fno_layers",       default=4,  type=int)
# # # # # # # # # # # # #     p.add_argument("--fno_d_model",      default=32, type=int)
# # # # # # # # # # # # #     p.add_argument("--fno_spatial_down", default=32, type=int)
# # # # # # # # # # # # #     p.add_argument("--mamba_d_state",    default=16, type=int)

# # # # # # # # # # # # #     p.add_argument("--val_loss_freq",    default=1,   type=int)
# # # # # # # # # # # # #     p.add_argument("--val_freq",         default=1,   type=int)
# # # # # # # # # # # # #     p.add_argument("--val_ade_freq",     default=1,   type=int)
# # # # # # # # # # # # #     p.add_argument("--full_eval_freq",   default=10,  type=int)
# # # # # # # # # # # # #     p.add_argument("--val_subset_size",  default=600, type=int)

# # # # # # # # # # # # #     p.add_argument("--output_dir",      default="runs/v25",      type=str)
# # # # # # # # # # # # #     p.add_argument("--save_interval",   default=10,              type=int)
# # # # # # # # # # # # #     p.add_argument("--metrics_csv",     default="metrics.csv",   type=str)
# # # # # # # # # # # # #     p.add_argument("--predict_csv",     default="predictions.csv", type=str)
# # # # # # # # # # # # #     p.add_argument("--lstm_errors_npy",      default=None, type=str)
# # # # # # # # # # # # #     p.add_argument("--diffusion_errors_npy", default=None, type=str)

# # # # # # # # # # # # #     p.add_argument("--gpu_num",         default="0", type=str)
# # # # # # # # # # # # #     p.add_argument("--delim",           default=" ")
# # # # # # # # # # # # #     p.add_argument("--skip",            default=1,   type=int)
# # # # # # # # # # # # #     p.add_argument("--min_ped",         default=1,   type=int)
# # # # # # # # # # # # #     p.add_argument("--threshold",       default=0.002, type=float)
# # # # # # # # # # # # #     p.add_argument("--other_modal",     default="gph")

# # # # # # # # # # # # #     p.add_argument("--step_weight_decay_epochs", default=60, type=int)
# # # # # # # # # # # # #     p.add_argument("--lon_flip_prob",            default=0.3, type=float)

# # # # # # # # # # # # #     p.add_argument("--pinn_warmup_epochs", default=80,    type=int)
# # # # # # # # # # # # #     p.add_argument("--pinn_w_start",      default=0.001, type=float)
# # # # # # # # # # # # #     p.add_argument("--pinn_w_end",        default=0.05,  type=float)

# # # # # # # # # # # # #     p.add_argument("--vel_warmup_epochs",    default=20,  type=float)
# # # # # # # # # # # # #     p.add_argument("--vel_w_start",          default=0.5, type=float)
# # # # # # # # # # # # #     p.add_argument("--vel_w_end",            default=1.5, type=float)
# # # # # # # # # # # # #     p.add_argument("--recurv_warmup_epochs", default=10,  type=int)
# # # # # # # # # # # # #     p.add_argument("--recurv_w_start",       default=0.3, type=float)
# # # # # # # # # # # # #     p.add_argument("--recurv_w_end",         default=1.0, type=float)

# # # # # # # # # # # # #     return p.parse_args()


# # # # # # # # # # # # # def _resolve_ode_steps(args):
# # # # # # # # # # # # #     if args.ode_steps is not None:
# # # # # # # # # # # # #         return args.ode_steps, args.ode_steps, args.ode_steps
# # # # # # # # # # # # #     return args.ode_steps_train, args.ode_steps_val, args.ode_steps_test


# # # # # # # # # # # # # # ── MAIN ──────────────────────────────────────────────────────────────────────

# # # # # # # # # # # # # def main(args):
# # # # # # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # # # # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # # # # # # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# # # # # # # # # # # # #     predict_csv = os.path.join(args.output_dir, args.predict_csv)
# # # # # # # # # # # # #     tables_dir  = os.path.join(args.output_dir, "tables")
# # # # # # # # # # # # #     stat_dir    = os.path.join(tables_dir, "stat_tests")
# # # # # # # # # # # # #     os.makedirs(tables_dir, exist_ok=True)
# # # # # # # # # # # # #     os.makedirs(stat_dir,   exist_ok=True)

# # # # # # # # # # # # #     ode_train, ode_val, ode_test = _resolve_ode_steps(args)

# # # # # # # # # # # # #     print("=" * 70)
# # # # # # # # # # # # #     print("  TC-FlowMatching v25  |  Full Fix Release")
# # # # # # # # # # # # #     print("  KEY FIXES:")
# # # # # # # # # # # # #     print("    FIX-T-A: epoch truyền vào get_loss_breakdown() → AdaptClamp đúng")
# # # # # # # # # # # # #     print("    FIX-T-B: cache raw_ctx, tránh _context() 2 lần/batch")
# # # # # # # # # # # # #     print("    FIX-T-C: freeze ctx_fc1+ctx_ln trong phase 1")
# # # # # # # # # # # # #     print("    FIX-T-D: short_range_weight 5→3→2→1.5 (scale đúng với km)")
# # # # # # # # # # # # #     print("    FIX-T-E: bridge_weight từ epoch 10")
# # # # # # # # # # # # #     print("    FIX-L-A: bỏ *NRM khỏi pinn loss (tránh overpower 35×)")
# # # # # # # # # # # # #     print("    FIX-L-B: AdaptClamp Huber→nội suy→tanh theo epoch")
# # # # # # # # # # # # #     print("    FIX-L-C: adaptive BVE weighting theo track error")
# # # # # # # # # # # # #     print("    FIX-L-D: L_PWR pressure-wind balance (ep 30+)")
# # # # # # # # # # # # #     print("    FIX-L-G: Energy Score term trong AFCRPS")
# # # # # # # # # # # # #     print("    FIX-L-H: L_bridge SR↔FM tại step 4")
# # # # # # # # # # # # #     print("    FIX-M-A: ShortRangeHead gate bias=+2.0")
# # # # # # # # # # # # #     print("    FIX-M-D: Adam thay SGD+momentum trong physics_correct")
# # # # # # # # # # # # #     print("=" * 70)

# # # # # # # # # # # # #     train_dataset, train_loader = data_loader(
# # # # # # # # # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # # # # # # # # #     val_dataset, val_loader = data_loader(
# # # # # # # # # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # # # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # # # # # # #     val_subset_loader = make_val_subset_loader(
# # # # # # # # # # # # #         val_dataset, args.val_subset_size, args.batch_size,
# # # # # # # # # # # # #         seq_collate, args.num_workers)

# # # # # # # # # # # # #     test_loader = None
# # # # # # # # # # # # #     try:
# # # # # # # # # # # # #         _, test_loader = data_loader(
# # # # # # # # # # # # #             args, {"root": args.dataset_root, "type": "test"},
# # # # # # # # # # # # #             test=True, test_year=None)
# # # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # # #         print(f"  Warning: test loader: {e}")

# # # # # # # # # # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # # # # # # # # # #     print(f"  val   : {len(val_dataset)} seq")

# # # # # # # # # # # # #     model = TCFlowMatching(
# # # # # # # # # # # # #         pred_len             = args.pred_len,
# # # # # # # # # # # # #         obs_len              = args.obs_len,
# # # # # # # # # # # # #         sigma_min            = args.sigma_min,
# # # # # # # # # # # # #         n_train_ens          = args.n_train_ens,
# # # # # # # # # # # # #         ctx_noise_scale      = args.ctx_noise_scale,
# # # # # # # # # # # # #         initial_sample_sigma = args.initial_sample_sigma,
# # # # # # # # # # # # #     ).to(device)

# # # # # # # # # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # # # # # # # #     print(f"  params  : {n_params:,}")

# # # # # # # # # # # # #     # try:
# # # # # # # # # # # # #     #     model = torch.compile(model, mode="reduce-overhead")
# # # # # # # # # # # # #     #     print("  torch.compile: enabled")
# # # # # # # # # # # # #     # except Exception:
# # # # # # # # # # # # #     #     pass

# # # # # # # # # # # # #     # # # Phase 1: freeze backbone + ctx_fc1 (FIX-T-C)
# # # # # # # # # # # # #     # # freeze_backbone(model)
# # # # # # # # # # # # #     # optimizer = optim.AdamW(
# # # # # # # # # # # # #     #     filter(lambda p: p.requires_grad, model.parameters()),
# # # # # # # # # # # # #     #     lr=args.phase1_lr, weight_decay=args.weight_decay,
# # # # # # # # # # # # #     # )
# # # # # # # # # # # # #     # FIX: freeze TRƯỚC compile
# # # # # # # # # # # # #     freeze_backbone(model)
# # # # # # # # # # # # #     optimizer = optim.AdamW(
# # # # # # # # # # # # #         filter(lambda p: p.requires_grad, model.parameters()),
# # # # # # # # # # # # #         lr=args.phase1_lr, weight_decay=args.weight_decay,
# # # # # # # # # # # # #     )

# # # # # # # # # # # # #     # compile SAU freeze
# # # # # # # # # # # # #     try:
# # # # # # # # # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # # # # # # # # #         print("  torch.compile: enabled")
# # # # # # # # # # # # #     except Exception:
# # # # # # # # # # # # #         pass

# # # # # # # # # # # # #     steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
# # # # # # # # # # # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # # # # # # # # # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # # # # # # # # # # #     scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

# # # # # # # # # # # # #     saver  = BestModelSaver(patience=args.patience, sr_tol=3.0, ade_tol=1.0)
# # # # # # # # # # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)

# # # # # # # # # # # # #     print("=" * 70)
# # # # # # # # # # # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch)")
# # # # # # # # # # # # #     print("=" * 70)

# # # # # # # # # # # # #     epoch_times   = []
# # # # # # # # # # # # #     train_start   = time.perf_counter()
# # # # # # # # # # # # #     last_val_loss = float("inf")
# # # # # # # # # # # # #     _phase        = 1
# # # # # # # # # # # # #     _prev_ens     = 1
# # # # # # # # # # # # #     _lr_ep30_done = False
# # # # # # # # # # # # #     _lr_ep60_done = False

# # # # # # # # # # # # #     import Model.losses as _losses_mod

# # # # # # # # # # # # #     for epoch in range(args.num_epochs):

# # # # # # # # # # # # #         # Phase transition
# # # # # # # # # # # # #         if epoch == args.phase2_start and _phase == 1:
# # # # # # # # # # # # #             _phase = 2
# # # # # # # # # # # # #             unfreeze_all(model)
# # # # # # # # # # # # #             optimizer = optim.AdamW(
# # # # # # # # # # # # #                 model.parameters(),
# # # # # # # # # # # # #                 lr=args.g_learning_rate,
# # # # # # # # # # # # #                 weight_decay=args.weight_decay,
# # # # # # # # # # # # #             )
# # # # # # # # # # # # #             rem_steps = steps_per_epoch * (args.num_epochs - epoch)
# # # # # # # # # # # # #             scheduler = get_cosine_schedule_with_warmup(
# # # # # # # # # # # # #                 optimizer, steps_per_epoch, rem_steps, min_lr=5e-6)
# # # # # # # # # # # # #             saver.reset_counters(f"Phase 2 started @ ep {epoch}")
# # # # # # # # # # # # #             print(f"\n  ↺  PHASE 2 START ep {epoch}: unfreeze all, "
# # # # # # # # # # # # #                   f"LR={args.g_learning_rate:.2e}")

# # # # # # # # # # # # #         # Weight schedules
# # # # # # # # # # # # #         current_ens = get_progressive_ens(epoch, args.n_train_ens)
# # # # # # # # # # # # #         model.n_train_ens = current_ens
# # # # # # # # # # # # #         eff_fast_ens = min(args.fast_ensemble,
# # # # # # # # # # # # #                            max(current_ens * 2, args.fast_ensemble))

# # # # # # # # # # # # #         if current_ens != _prev_ens:
# # # # # # # # # # # # #             saver.reset_counters(f"n_train_ens {_prev_ens}→{current_ens}")
# # # # # # # # # # # # #             _prev_ens = current_ens

# # # # # # # # # # # # #         step_alpha    = get_step_weight_alpha(epoch, args.step_weight_decay_epochs)
# # # # # # # # # # # # #         epoch_weights = copy.copy(_BASE_WEIGHTS)
# # # # # # # # # # # # #         epoch_weights["fm"]          = get_fm_weight(epoch)
# # # # # # # # # # # # #         epoch_weights["short_range"] = get_short_range_weight(epoch)   # FIX-T-D
# # # # # # # # # # # # #         epoch_weights["bridge"]      = get_bridge_weight(epoch)         # FIX-T-E
# # # # # # # # # # # # #         epoch_weights["spread"]      = get_spread_weight(epoch)
# # # # # # # # # # # # #         epoch_weights["pinn"]        = get_pinn_weight(
# # # # # # # # # # # # #             epoch, args.pinn_warmup_epochs, args.pinn_w_start, args.pinn_w_end)
# # # # # # # # # # # # #         epoch_weights["velocity"]    = get_velocity_weight(
# # # # # # # # # # # # #             epoch, args.vel_warmup_epochs, args.vel_w_start, args.vel_w_end)
# # # # # # # # # # # # #         epoch_weights["recurv"]      = get_recurv_weight(
# # # # # # # # # # # # #             epoch, args.recurv_warmup_epochs, args.recurv_w_start, args.recurv_w_end)
# # # # # # # # # # # # #         _losses_mod.WEIGHTS.update(epoch_weights)

# # # # # # # # # # # # #         current_clip = get_grad_clip(epoch, warmup_epochs=20,
# # # # # # # # # # # # #                                       clip_start=args.grad_clip, clip_end=1.0)

# # # # # # # # # # # # #         # LR warm restarts (phase 2 only)
# # # # # # # # # # # # #         if _phase == 2:
# # # # # # # # # # # # #             if epoch == 30 and not _lr_ep30_done:
# # # # # # # # # # # # #                 _lr_ep30_done = True
# # # # # # # # # # # # #                 scheduler = get_cosine_schedule_with_warmup(
# # # # # # # # # # # # #                     optimizer, steps_per_epoch,
# # # # # # # # # # # # #                     steps_per_epoch * (args.num_epochs - 30), min_lr=5e-6)
# # # # # # # # # # # # #                 saver.reset_counters("LR warm restart ep 30")
# # # # # # # # # # # # #                 print(f"  ↺  Warm Restart LR @ ep 30")

# # # # # # # # # # # # #             if epoch == 60 and not _lr_ep60_done:
# # # # # # # # # # # # #                 _lr_ep60_done = True
# # # # # # # # # # # # #                 scheduler = get_cosine_schedule_with_warmup(
# # # # # # # # # # # # #                     optimizer, steps_per_epoch,
# # # # # # # # # # # # #                     steps_per_epoch * (args.num_epochs - 60), min_lr=1e-6)
# # # # # # # # # # # # #                 saver.reset_counters("LR warm restart ep 60")
# # # # # # # # # # # # #                 print(f"  ↺  Warm Restart LR @ ep 60")

# # # # # # # # # # # # #         # Training loop
# # # # # # # # # # # # #         model.train()
# # # # # # # # # # # # #         sum_loss   = 0.0
# # # # # # # # # # # # #         t0         = time.perf_counter()
# # # # # # # # # # # # #         optimizer.zero_grad()
# # # # # # # # # # # # #         recurv_ratio_buf = []

# # # # # # # # # # # # #         for i, batch in enumerate(train_loader):
# # # # # # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # # # # # #             if epoch == 0 and i == 0:
# # # # # # # # # # # # #                 _check_gph500(bl, train_dataset)
# # # # # # # # # # # # #                 _check_uv500(bl)

# # # # # # # # # # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # # # # # #                 # FIX-T-A: pass epoch vào get_loss_breakdown
# # # # # # # # # # # # #                 bd = model.get_loss_breakdown(
# # # # # # # # # # # # #                     bl,
# # # # # # # # # # # # #                     step_weight_alpha=step_alpha,
# # # # # # # # # # # # #                     epoch=epoch,               # ← KEY FIX
# # # # # # # # # # # # #                 )

# # # # # # # # # # # # #             loss_to_bp = bd["total"] / max(args.grad_accum, 1)
# # # # # # # # # # # # #             scaler.scale(loss_to_bp).backward()

# # # # # # # # # # # # #             if ((i + 1) % args.grad_accum == 0
# # # # # # # # # # # # #                     or (i + 1) == len(train_loader)):
# # # # # # # # # # # # #                 scaler.unscale_(optimizer)
# # # # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_(
# # # # # # # # # # # # #                     model.parameters(), current_clip)
# # # # # # # # # # # # #                 scaler.step(optimizer)
# # # # # # # # # # # # #                 scaler.update()
# # # # # # # # # # # # #                 optimizer.zero_grad()
# # # # # # # # # # # # #                 scheduler.step()

# # # # # # # # # # # # #             sum_loss += bd["total"].item()
# # # # # # # # # # # # #             if "recurv_ratio" in bd:
# # # # # # # # # # # # #                 recurv_ratio_buf.append(bd["recurv_ratio"])

# # # # # # # # # # # # #             if i % 20 == 0:
# # # # # # # # # # # # #                 lr      = optimizer.param_groups[0]["lr"]
# # # # # # # # # # # # #                 sr_loss = bd.get("short_range", 0.0)
# # # # # # # # # # # # #                 bridge  = bd.get("bridge", 0.0)
# # # # # # # # # # # # #                 elapsed = time.perf_counter() - t0
# # # # # # # # # # # # #                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # # # # # # # # # # #                       f"  loss={bd['total'].item():.3f}"
# # # # # # # # # # # # #                       f"  fm={bd.get('fm',0):.2f}"
# # # # # # # # # # # # #                       f"  sr={sr_loss:.4f}"
# # # # # # # # # # # # #                       f"  bridge={bridge:.4f}"
# # # # # # # # # # # # #                       f"  pinn={bd.get('pinn',0):.4f}"
# # # # # # # # # # # # #                       f"  spread={bd.get('spread',0):.6f}"
# # # # # # # # # # # # #                       f"  sr_w={epoch_weights['short_range']:.1f}"
# # # # # # # # # # # # #                       f"  br_w={epoch_weights['bridge']:.1f}"
# # # # # # # # # # # # #                       f"  alpha={step_alpha:.2f}"
# # # # # # # # # # # # #                       f"  ph={_phase}"
# # # # # # # # # # # # #                       f"  ep={epoch}"       # FIX-T-F: log epoch
# # # # # # # # # # # # #                       f"  ens={current_ens}"
# # # # # # # # # # # # #                       f"  lr={lr:.2e}  t={elapsed:.0f}s")

# # # # # # # # # # # # #         ep_s    = time.perf_counter() - t0
# # # # # # # # # # # # #         epoch_times.append(ep_s)
# # # # # # # # # # # # #         avg_t   = sum_loss / len(train_loader)
# # # # # # # # # # # # #         mean_rr = float(np.mean(recurv_ratio_buf)) if recurv_ratio_buf else 0.0

# # # # # # # # # # # # #         # Val loss
# # # # # # # # # # # # #         model.eval()
# # # # # # # # # # # # #         val_loss = 0.0
# # # # # # # # # # # # #         t_val    = time.perf_counter()
# # # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # # #             for batch in val_loader:
# # # # # # # # # # # # #                 bl_v = move(list(batch), device)
# # # # # # # # # # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # # # # # #                     # FIX-T-A: truyền epoch vào val loss cũng
# # # # # # # # # # # # #                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
# # # # # # # # # # # # #         last_val_loss = val_loss / len(val_loader)
# # # # # # # # # # # # #         t_val_s = time.perf_counter() - t_val
# # # # # # # # # # # # #         saver.update_val_loss(last_val_loss, model, args.output_dir,
# # # # # # # # # # # # #                                epoch, optimizer, avg_t)

# # # # # # # # # # # # #         print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
# # # # # # # # # # # # #               f"  rr={mean_rr:.2f}"
# # # # # # # # # # # # #               f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
# # # # # # # # # # # # #               f"  ens={current_ens}  alpha={step_alpha:.2f}")

# # # # # # # # # # # # #         # Fast eval
# # # # # # # # # # # # #         t_ade   = time.perf_counter()
# # # # # # # # # # # # #         m_fast  = evaluate_fast(model, val_subset_loader, device,
# # # # # # # # # # # # #                                 ode_train, args.pred_len, eff_fast_ens)
# # # # # # # # # # # # #         t_ade_s = time.perf_counter() - t_ade

# # # # # # # # # # # # #         sr_12h   = m_fast.get("sr_ade_12h", float("nan"))
# # # # # # # # # # # # #         sr_24h   = m_fast.get("sr_ade_24h", float("nan"))
# # # # # # # # # # # # #         spread72 = m_fast.get("spread_72h_km", 0.0)
# # # # # # # # # # # # #         collapse = "  ⚠️ COLLAPSE!" if spread72 < 10.0 else ""
# # # # # # # # # # # # #         hi_sprd  = "  ⚠️ SPREAD!" if spread72 > 400.0 else ""
# # # # # # # # # # # # #         sr_hit12 = "  🎯 <50km!" if sr_12h < 50.0 else ""
# # # # # # # # # # # # #         sr_hit24 = "  🎯 <100km!" if sr_24h < 100.0 else ""

# # # # # # # # # # # # #         print(f"  [FAST ep{epoch} {t_ade_s:.0f}s]"
# # # # # # # # # # # # #               f"  ADE={m_fast['ADE']:.1f}  FDE={m_fast['FDE']:.1f} km"
# # # # # # # # # # # # #               f"  12h={m_fast.get('12h',float('nan')):.0f}"
# # # # # # # # # # # # #               f"  24h={m_fast.get('24h',float('nan')):.0f}"
# # # # # # # # # # # # #               f"  72h={m_fast.get('72h',float('nan')):.0f} km"
# # # # # # # # # # # # #               f"  spread={spread72:.1f} km"
# # # # # # # # # # # # #               f"{collapse}{hi_sprd}")
# # # # # # # # # # # # #         print(f"         SR: 12h={sr_12h:.1f} km"
# # # # # # # # # # # # #               f"  24h={sr_24h:.1f} km"
# # # # # # # # # # # # #               f"{sr_hit12}{sr_hit24}")

# # # # # # # # # # # # #         saver.log_subset_sr(sr_12h, sr_24h, epoch)

# # # # # # # # # # # # #         # Full val ADE
# # # # # # # # # # # # #         if epoch % args.val_ade_freq == 0:
# # # # # # # # # # # # #             try:
# # # # # # # # # # # # #                 r_full = evaluate_full_val_ade(
# # # # # # # # # # # # #                     model, val_loader, device,
# # # # # # # # # # # # #                     ode_steps     = ode_train,
# # # # # # # # # # # # #                     pred_len      = args.pred_len,
# # # # # # # # # # # # #                     fast_ensemble = eff_fast_ens,
# # # # # # # # # # # # #                     metrics_csv   = metrics_csv,
# # # # # # # # # # # # #                     epoch         = epoch,
# # # # # # # # # # # # #                     tag           = f"val_full_ep{epoch:03d}",
# # # # # # # # # # # # #                 )
# # # # # # # # # # # # #                 saver.update_full_val(
# # # # # # # # # # # # #                     r_full, model, args.output_dir, epoch,
# # # # # # # # # # # # #                     optimizer, avg_t, last_val_loss,
# # # # # # # # # # # # #                     min_epochs=args.min_epochs)
# # # # # # # # # # # # #             except Exception as e:
# # # # # # # # # # # # #                 print(f"  ⚠  Full val failed: {e}")
# # # # # # # # # # # # #                 import traceback; traceback.print_exc()

# # # # # # # # # # # # #         if epoch % args.full_eval_freq == 0 and epoch > 0:
# # # # # # # # # # # # #             try:
# # # # # # # # # # # # #                 dm, _, _, _ = evaluate_full(
# # # # # # # # # # # # #                     model, val_loader, device,
# # # # # # # # # # # # #                     ode_val, args.pred_len, args.val_ensemble,
# # # # # # # # # # # # #                     metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}")
# # # # # # # # # # # # #                 print(dm.summary())
# # # # # # # # # # # # #             except Exception as e:
# # # # # # # # # # # # #                 print(f"  ⚠  full_eval failed ep {epoch}: {e}")

# # # # # # # # # # # # #         # if (epoch + 1) % args.save_interval == 0:
# # # # # # # # # # # # #         #     torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
# # # # # # # # # # # # #         #                os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))
# # # # # # # # # # # # #         # Cuối mỗi epoch (thay save_interval)
# # # # # # # # # # # # #         # Bằng đoạn này — lưu TẤT CẢ mọi epoch:
# # # # # # # # # # # # #         torch.save(
# # # # # # # # # # # # #             {
# # # # # # # # # # # # #                 "epoch"            : epoch,
# # # # # # # # # # # # #                 "model_state_dict" : model.state_dict(),
# # # # # # # # # # # # #                 "optimizer_state"  : optimizer.state_dict(),
# # # # # # # # # # # # #                 "train_loss"       : avg_t,
# # # # # # # # # # # # #                 "val_loss"         : last_val_loss,
# # # # # # # # # # # # #                 "sr_ade_12h"       : m_fast.get("sr_ade_12h", float("nan")),
# # # # # # # # # # # # #                 "blend_ade"        : m_fast.get("ADE", float("nan")),
# # # # # # # # # # # # #                 "spread_72h"       : m_fast.get("spread_72h_km", 0.0),
# # # # # # # # # # # # #             },
# # # # # # # # # # # # #             os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"),
# # # # # # # # # # # # #         )

# # # # # # # # # # # # #         if epoch % 5 == 4:
# # # # # # # # # # # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # # # # # # # # # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # # # # # # # # # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # # # # # # # # # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
# # # # # # # # # # # # #                   f"  (avg {avg_ep:.0f}s/epoch)")

# # # # # # # # # # # # #     # Final
# # # # # # # # # # # # #     _losses_mod.WEIGHTS["pinn"]        = args.pinn_w_end
# # # # # # # # # # # # #     _losses_mod.WEIGHTS["velocity"]    = args.vel_w_end
# # # # # # # # # # # # #     _losses_mod.WEIGHTS["recurv"]      = args.recurv_w_end
# # # # # # # # # # # # #     _losses_mod.WEIGHTS["short_range"] = 1.5
# # # # # # # # # # # # #     _losses_mod.WEIGHTS["bridge"]      = 0.5

# # # # # # # # # # # # #     total_train_h = (time.perf_counter() - train_start) / 3600

# # # # # # # # # # # # #     print(f"\n{'='*70}  FINAL TEST (ode_steps={ode_test})")
# # # # # # # # # # # # #     all_results = []

# # # # # # # # # # # # #     if test_loader:
# # # # # # # # # # # # #         best_path = os.path.join(args.output_dir, "best_model.pth")
# # # # # # # # # # # # #         if not os.path.exists(best_path):
# # # # # # # # # # # # #             best_path = os.path.join(args.output_dir, "best_model_valloss.pth")
# # # # # # # # # # # # #         if os.path.exists(best_path):
# # # # # # # # # # # # #             ck = torch.load(best_path, map_location=device)
# # # # # # # # # # # # #             try:
# # # # # # # # # # # # #                 model.load_state_dict(ck["model_state_dict"])
# # # # # # # # # # # # #             except Exception:
# # # # # # # # # # # # #                 model.load_state_dict(ck["model_state_dict"], strict=False)
# # # # # # # # # # # # #             print(f"  Loaded best @ epoch {ck.get('epoch','?')}"
# # # # # # # # # # # # #                   f"  SR-12h={ck.get('sr_ade_12h','?')}")

# # # # # # # # # # # # #         final_ens = max(args.val_ensemble, 50)
# # # # # # # # # # # # #         dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
# # # # # # # # # # # # #             model, test_loader, device,
# # # # # # # # # # # # #             ode_test, args.pred_len, final_ens,
# # # # # # # # # # # # #             metrics_csv=metrics_csv, tag="test_final",
# # # # # # # # # # # # #             predict_csv=predict_csv)
# # # # # # # # # # # # #         print(dm_test.summary())

# # # # # # # # # # # # #         all_results.append(ModelResult(
# # # # # # # # # # # # #             model_name   = "FM+SR+PINN-v25",
# # # # # # # # # # # # #             split        = "test",
# # # # # # # # # # # # #             ADE          = dm_test.ade,
# # # # # # # # # # # # #             FDE          = dm_test.fde,
# # # # # # # # # # # # #             ADE_str      = dm_test.ade_str,
# # # # # # # # # # # # #             ADE_rec      = dm_test.ade_rec,
# # # # # # # # # # # # #             delta_rec    = dm_test.pr,
# # # # # # # # # # # # #             CRPS_mean    = dm_test.crps_mean,
# # # # # # # # # # # # #             CRPS_72h     = dm_test.crps_72h,
# # # # # # # # # # # # #             SSR          = dm_test.ssr_mean,
# # # # # # # # # # # # #             TSS_72h      = dm_test.tss_72h,
# # # # # # # # # # # # #             OYR          = dm_test.oyr_mean,
# # # # # # # # # # # # #             DTW          = dm_test.dtw_mean,
# # # # # # # # # # # # #             ATE_abs      = dm_test.ate_abs_mean,
# # # # # # # # # # # # #             CTE_abs      = dm_test.cte_abs_mean,
# # # # # # # # # # # # #             n_total      = dm_test.n_total,
# # # # # # # # # # # # #             n_recurv     = dm_test.n_rec,
# # # # # # # # # # # # #             train_time_h = total_train_h,
# # # # # # # # # # # # #             params_M     = n_params / 1e6,
# # # # # # # # # # # # #         ))

# # # # # # # # # # # # #         _, cliper_errs = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
# # # # # # # # # # # # #         persist_errs   = persistence_errors(obs_seqs, gt_seqs, args.pred_len)
# # # # # # # # # # # # #         fmpinn_per_seq = np.array([seq_ade_km(np.array(pp), np.array(g))
# # # # # # # # # # # # #                                     for pp, g in zip(pred_seqs, gt_seqs)])

# # # # # # # # # # # # #         np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
# # # # # # # # # # # # #         np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
# # # # # # # # # # # # #         np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))

# # # # # # # # # # # # #         stat_rows = [
# # # # # # # # # # # # #             paired_tests(fmpinn_per_seq, cliper_errs.mean(1),  "FM+SR vs CLIPER", 5),
# # # # # # # # # # # # #             paired_tests(fmpinn_per_seq, persist_errs.mean(1), "FM+SR vs Persist", 5),
# # # # # # # # # # # # #         ]
# # # # # # # # # # # # #         lstm_per_seq      = _load_baseline_errors(args.lstm_errors_npy, "LSTM")
# # # # # # # # # # # # #         diffusion_per_seq = _load_baseline_errors(args.diffusion_errors_npy, "Diffusion")
# # # # # # # # # # # # #         if lstm_per_seq is not None:
# # # # # # # # # # # # #             stat_rows.append(paired_tests(fmpinn_per_seq, lstm_per_seq, "FM+SR vs LSTM", 5))
# # # # # # # # # # # # #         if diffusion_per_seq is not None:
# # # # # # # # # # # # #             stat_rows.append(paired_tests(fmpinn_per_seq, diffusion_per_seq, "FM+SR vs Diffusion", 5))

# # # # # # # # # # # # #         export_all_tables(
# # # # # # # # # # # # #             results=all_results, ablation_rows=DEFAULT_ABLATION,
# # # # # # # # # # # # #             stat_rows=stat_rows, pinn_sens_rows=DEFAULT_PINN_SENSITIVITY,
# # # # # # # # # # # # #             compute_rows=DEFAULT_COMPUTE, out_dir=tables_dir)

# # # # # # # # # # # # #         with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
# # # # # # # # # # # # #             fh.write(dm_test.summary())
# # # # # # # # # # # # #             fh.write(f"\n\nmodel_version         : FM+SR+PINN v25\n")
# # # # # # # # # # # # #             fh.write(f"ctx_noise_scale       : {args.ctx_noise_scale}\n")
# # # # # # # # # # # # #             fh.write(f"initial_sample_sigma  : {args.initial_sample_sigma}\n")
# # # # # # # # # # # # #             fh.write(f"ode_steps_test        : {ode_test}\n")
# # # # # # # # # # # # #             fh.write(f"eval_ensemble         : {final_ens}\n")
# # # # # # # # # # # # #             fh.write(f"train_time_h          : {total_train_h:.2f}\n")
# # # # # # # # # # # # #             fh.write(f"n_params_M            : {n_params/1e6:.2f}\n")

# # # # # # # # # # # # #     avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 0
# # # # # # # # # # # # #     print(f"\n  Best SR-12h ADE    : {saver.best_sr12h:.1f} km")
# # # # # # # # # # # # #     print(f"  Best blend ADE     : {saver.best_ade:.1f} km")
# # # # # # # # # # # # #     print(f"  Best val loss      : {saver.best_val_loss:.4f}")
# # # # # # # # # # # # #     print(f"  Avg epoch time     : {avg_ep:.0f}s")
# # # # # # # # # # # # #     print(f"  Total training     : {total_train_h:.2f}h")
# # # # # # # # # # # # #     print("=" * 70)


# # # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # # #     args = get_args()
# # # # # # # # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # # # # # #         torch.cuda.manual_seed_all(42)
# # # # # # # # # # # # #     main(args)

# # # # # # # # # # # # """
# # # # # # # # # # # # scripts/train_flowmatching.py  ── v30 - HIERARCHICAL
# # # # # # # # # # # # =====================================================
# # # # # # # # # # # # KEY CHANGES từ v25:
# # # # # # # # # # # #   1. BỎ HOÀN TOÀN Phase 1 freeze → train all from epoch 0
# # # # # # # # # # # #   2. Single optimizer, single LR schedule
# # # # # # # # # # # #   3. Weight schedules đơn giản hơn
# # # # # # # # # # # #   4. Continuity loss thay bridge loss
# # # # # # # # # # # #   5. FM weight = 2.0 constant (không suppress)
# # # # # # # # # # # #   6. SR weight = 3.0 → 2.0 (giảm dần)
# # # # # # # # # # # #   7. PINN warmup chậm hơn, weight thấp hơn
# # # # # # # # # # # #   8. Ensemble progressive: 2→4→6 (bắt đầu từ 2, không phải 1)
# # # # # # # # # # # # """
# # # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # # import sys
# # # # # # # # # # # # import os
# # # # # # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # # # # # import argparse
# # # # # # # # # # # # import time
# # # # # # # # # # # # import math
# # # # # # # # # # # # import random
# # # # # # # # # # # # import copy

# # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # import torch
# # # # # # # # # # # # import torch.optim as optim
# # # # # # # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # # # # # from Model.data.loader_training import data_loader
# # # # # # # # # # # # from Model.flow_matching_model import TCFlowMatching, ShortRangeHead
# # # # # # # # # # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # # # # # # from Model.losses import WEIGHTS as _BASE_WEIGHTS
# # # # # # # # # # # # from utils.metrics import (
# # # # # # # # # # # #     TCEvaluator, StepErrorAccumulator,
# # # # # # # # # # # #     save_metrics_csv, haversine_km_torch,
# # # # # # # # # # # #     denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
# # # # # # # # # # # #     cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
# # # # # # # # # # # #     brier_skill_score, cliper_errors, persistence_errors,
# # # # # # # # # # # # )
# # # # # # # # # # # # from utils.evaluation_tables import (
# # # # # # # # # # # #     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
# # # # # # # # # # # #     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
# # # # # # # # # # # #     DEFAULT_COMPUTE, paired_tests,
# # # # # # # # # # # # )
# # # # # # # # # # # # from scripts.statistical_tests import run_all_tests


# # # # # # # # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # # # # # # # def haversine_km_np_local(pred_deg, gt_deg):
# # # # # # # # # # # #     pred_deg = np.atleast_2d(pred_deg)
# # # # # # # # # # # #     gt_deg   = np.atleast_2d(gt_deg)
# # # # # # # # # # # #     R = 6371.0
# # # # # # # # # # # #     lon1, lat1 = np.radians(pred_deg[:, 0]), np.radians(pred_deg[:, 1])
# # # # # # # # # # # #     lon2, lat2 = np.radians(gt_deg[:, 0]),   np.radians(gt_deg[:, 1])
# # # # # # # # # # # #     dlon = lon2 - lon1;  dlat = lat2 - lat1
# # # # # # # # # # # #     a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
# # # # # # # # # # # #     return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# # # # # # # # # # # # def seq_ade_km(pred_norm, gt_norm):
# # # # # # # # # # # #     return float(haversine_km_np_local(denorm_deg_np(pred_norm),
# # # # # # # # # # # #                                        denorm_deg_np(gt_norm)).mean())


# # # # # # # # # # # # def move(batch, device):
# # # # # # # # # # # #     out = list(batch)
# # # # # # # # # # # #     for i, x in enumerate(out):
# # # # # # # # # # # #         if torch.is_tensor(x):
# # # # # # # # # # # #             out[i] = x.to(device)
# # # # # # # # # # # #         elif isinstance(x, dict):
# # # # # # # # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # # # # # #                       for k, v in x.items()}
# # # # # # # # # # # #     return out


# # # # # # # # # # # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # # # # # # # # # # #                             collate_fn, num_workers):
# # # # # # # # # # # #     n   = len(val_dataset)
# # # # # # # # # # # #     rng = random.Random(42)
# # # # # # # # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # # # # # # # #     return DataLoader(Subset(val_dataset, idx),
# # # # # # # # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # # # # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # # # # # # # # ── Weight schedules (SIMPLIFIED) ────────────────────────────────────────────

# # # # # # # # # # # # def get_short_range_weight(epoch: int) -> float:
# # # # # # # # # # # #     """SR weight: 3.0 early → 2.0 later (always present, not dominant)"""
# # # # # # # # # # # #     if epoch < 20:
# # # # # # # # # # # #         return 3.0
# # # # # # # # # # # #     elif epoch < 50:
# # # # # # # # # # # #         return 2.5
# # # # # # # # # # # #     else:
# # # # # # # # # # # #         return 2.0


# # # # # # # # # # # # def get_continuity_weight(epoch: int) -> float:
# # # # # # # # # # # #     """Continuity: 0 first 5 epochs (SR not stable yet), then 2.0"""
# # # # # # # # # # # #     if epoch < 5:
# # # # # # # # # # # #         return 0.0
# # # # # # # # # # # #     elif epoch < 20:
# # # # # # # # # # # #         return 1.0
# # # # # # # # # # # #     else:
# # # # # # # # # # # #         return 2.0


# # # # # # # # # # # # def get_fm_weight(epoch: int) -> float:
# # # # # # # # # # # #     """FM weight: constant 2.0 (no suppression!)"""
# # # # # # # # # # # #     return 2.0


# # # # # # # # # # # # def get_spread_weight(epoch: int) -> float:
# # # # # # # # # # # #     """Spread: moderate, constant"""
# # # # # # # # # # # #     return 0.5


# # # # # # # # # # # # def get_pinn_weight(epoch: int, warmup_epochs: int = 60,
# # # # # # # # # # # #                     w_start: float = 0.001, w_end: float = 0.03) -> float:
# # # # # # # # # # # #     """PINN: gentle warmup, lower max (0.03 instead of 0.05)"""
# # # # # # # # # # # #     if epoch < 10:
# # # # # # # # # # # #         return w_start
# # # # # # # # # # # #     elif epoch < warmup_epochs:
# # # # # # # # # # # #         t = (epoch - 10) / max(warmup_epochs - 10, 1)
# # # # # # # # # # # #         return w_start + t * (w_end - w_start)
# # # # # # # # # # # #     return w_end


# # # # # # # # # # # # def get_velocity_weight(epoch, warmup_epochs=20, w_start=0.5, w_end=1.2):
# # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # #         return w_end
# # # # # # # # # # # #     return w_start + (epoch / max(warmup_epochs - 1, 1)) * (w_end - w_start)


# # # # # # # # # # # # def get_recurv_weight(epoch, warmup_epochs=10, w_start=0.3, w_end=1.0):
# # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # #         return w_end
# # # # # # # # # # # #     return w_start + (epoch / max(warmup_epochs - 1, 1)) * (w_end - w_start)


# # # # # # # # # # # # def get_grad_clip(epoch, warmup_epochs=20, clip_start=2.0, clip_end=1.0):
# # # # # # # # # # # #     if epoch >= warmup_epochs:
# # # # # # # # # # # #         return clip_end
# # # # # # # # # # # #     return clip_start - (epoch / max(warmup_epochs - 1, 1)) * (clip_start - clip_end)


# # # # # # # # # # # # def get_step_weight_alpha(epoch: int, decay_epochs: int = 40) -> float:
# # # # # # # # # # # #     """Faster decay to 0 (40 epochs instead of 60)"""
# # # # # # # # # # # #     if epoch >= decay_epochs:
# # # # # # # # # # # #         return 0.0
# # # # # # # # # # # #     return 1.0 - (epoch / decay_epochs)


# # # # # # # # # # # # def get_progressive_ens(epoch: int, n_train_ens: int = 6) -> int:
# # # # # # # # # # # #     """Start with 2 (not 1!) since FM needs diversity from start"""
# # # # # # # # # # # #     if epoch < 10:
# # # # # # # # # # # #         return 2
# # # # # # # # # # # #     elif epoch < 30:
# # # # # # # # # # # #         return 4
# # # # # # # # # # # #     else:
# # # # # # # # # # # #         return n_train_ens


# # # # # # # # # # # # # ── SR ADE helpers ────────────────────────────────────────────────────────────

# # # # # # # # # # # # @torch.no_grad()
# # # # # # # # # # # # def compute_sr_ade(model, batch_list, device):
# # # # # # # # # # # #     raw_ctx = model.net._context(batch_list)
# # # # # # # # # # # #     obs_t   = batch_list[0]
# # # # # # # # # # # #     traj_gt = batch_list[1]

# # # # # # # # # # # #     sr_pred = model.net.forward_short_range(obs_t, raw_ctx)

# # # # # # # # # # # #     pred_01 = sr_pred.clone()
# # # # # # # # # # # #     pred_01[..., 0] = sr_pred[..., 0] * 50.0 + 1800.0
# # # # # # # # # # # #     pred_01[..., 1] = sr_pred[..., 1] * 50.0

# # # # # # # # # # # #     gt_01 = traj_gt.clone()
# # # # # # # # # # # #     gt_01[..., 0] = traj_gt[..., 0] * 50.0 + 1800.0
# # # # # # # # # # # #     gt_01[..., 1] = traj_gt[..., 1] * 50.0

# # # # # # # # # # # #     ade_12h = haversine_km_torch(pred_01[1], gt_01[1]).mean().item()
# # # # # # # # # # # #     ade_24h = haversine_km_torch(pred_01[3], gt_01[3]).mean().item() \
# # # # # # # # # # # #               if pred_01.shape[0] >= 4 else float("nan")
# # # # # # # # # # # #     return ade_12h, ade_24h


# # # # # # # # # # # # # ── evaluate_fast ─────────────────────────────────────────────────────────────

# # # # # # # # # # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
# # # # # # # # # # # #     model.eval()
# # # # # # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # # # # # #     t0  = time.perf_counter()
# # # # # # # # # # # #     n   = 0
# # # # # # # # # # # #     spread_per_step = []
# # # # # # # # # # # #     sr_12h_buf, sr_24h_buf = [], []

# # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # #         for batch in loader:
# # # # # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # # # # #             pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
# # # # # # # # # # # #                                               ddim_steps=ode_steps)
# # # # # # # # # # # #             T_active  = pred.shape[0]
# # # # # # # # # # # #             gt_sliced = bl[1][:T_active]
# # # # # # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
# # # # # # # # # # # #             acc.update(dist)

# # # # # # # # # # # #             a12, a24 = compute_sr_ade(model, bl, device)
# # # # # # # # # # # #             sr_12h_buf.append(a12)
# # # # # # # # # # # #             sr_24h_buf.append(a24)

# # # # # # # # # # # #             step_spreads = []
# # # # # # # # # # # #             for t in range(all_trajs.shape[1]):
# # # # # # # # # # # #                 step_data = all_trajs[:, t, :, :]
# # # # # # # # # # # #                 std_lon = step_data[:, :, 0].std(0)
# # # # # # # # # # # #                 std_lat = step_data[:, :, 1].std(0)
# # # # # # # # # # # #                 spread = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# # # # # # # # # # # #                 step_spreads.append(spread)
# # # # # # # # # # # #             spread_per_step.append(step_spreads)
# # # # # # # # # # # #             n += 1

# # # # # # # # # # # #     r = acc.compute()
# # # # # # # # # # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)

# # # # # # # # # # # #     if spread_per_step:
# # # # # # # # # # # #         spreads = np.array(spread_per_step)
# # # # # # # # # # # #         r["spread_12h_km"] = float(spreads[:, 1].mean()) if spreads.shape[1] > 1 else 0.0
# # # # # # # # # # # #         r["spread_24h_km"] = float(spreads[:, 3].mean()) if spreads.shape[1] > 3 else 0.0
# # # # # # # # # # # #         r["spread_72h_km"] = float(spreads[:, -1].mean())

# # # # # # # # # # # #     r["sr_ade_12h"] = float(np.mean(sr_12h_buf)) if sr_12h_buf else float("nan")
# # # # # # # # # # # #     r["sr_ade_24h"] = float(np.mean(sr_24h_buf)) if sr_24h_buf else float("nan")
# # # # # # # # # # # #     return r


# # # # # # # # # # # # # ── evaluate_full_val_ade ────────────────────────────────────────────────────

# # # # # # # # # # # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # # # # # # # # # # #                            fast_ensemble, metrics_csv, epoch, tag=""):
# # # # # # # # # # # #     model.eval()
# # # # # # # # # # # #     acc       = StepErrorAccumulator(pred_len)
# # # # # # # # # # # #     t0        = time.perf_counter()
# # # # # # # # # # # #     n_batch   = 0
# # # # # # # # # # # #     sr_12h_buf, sr_24h_buf = [], []

# # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # #         for batch in val_loader:
# # # # # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # # # # #             pred, _, _ = model.sample(bl, num_ensemble=fast_ensemble,
# # # # # # # # # # # #                                        ddim_steps=ode_steps)
# # # # # # # # # # # #             T_pred = pred.shape[0]
# # # # # # # # # # # #             gt     = bl[1][:T_pred]
# # # # # # # # # # # #             dist   = haversine_km_torch(denorm_torch(pred), denorm_torch(gt))
# # # # # # # # # # # #             acc.update(dist)

# # # # # # # # # # # #             sr_pred_from_mean = pred[:ShortRangeHead.N_STEPS]
# # # # # # # # # # # #             gt_sr = bl[1][:ShortRangeHead.N_STEPS]

# # # # # # # # # # # #             sr_01 = sr_pred_from_mean.clone()
# # # # # # # # # # # #             sr_01[..., 0] = sr_pred_from_mean[..., 0] * 50.0 + 1800.0
# # # # # # # # # # # #             sr_01[..., 1] = sr_pred_from_mean[..., 1] * 50.0
# # # # # # # # # # # #             gt_01 = gt_sr.clone()
# # # # # # # # # # # #             gt_01[..., 0] = gt_sr[..., 0] * 50.0 + 1800.0
# # # # # # # # # # # #             gt_01[..., 1] = gt_sr[..., 1] * 50.0

# # # # # # # # # # # #             ade_12h = haversine_km_torch(sr_01[1], gt_01[1]).mean().item()
# # # # # # # # # # # #             ade_24h = haversine_km_torch(sr_01[3], gt_01[3]).mean().item() \
# # # # # # # # # # # #                       if sr_01.shape[0] >= 4 else float("nan")
# # # # # # # # # # # #             sr_12h_buf.append(ade_12h)
# # # # # # # # # # # #             sr_24h_buf.append(ade_24h)

# # # # # # # # # # # #             n_batch += 1

# # # # # # # # # # # #     elapsed  = time.perf_counter() - t0
# # # # # # # # # # # #     r        = acc.compute()
# # # # # # # # # # # #     sr_12h   = float(np.mean(sr_12h_buf)) if sr_12h_buf else float("nan")
# # # # # # # # # # # #     sr_24h   = float(np.mean(sr_24h_buf)) if sr_24h_buf else float("nan")

# # # # # # # # # # # #     print(f"\n{'='*64}")
# # # # # # # # # # # #     print(f"  [FULL VAL ADE  ep={epoch}  {elapsed:.0f}s  {n_batch} batches]")
# # # # # # # # # # # #     print(f"  ADE = {r.get('ADE', float('nan')):.1f} km  "
# # # # # # # # # # # #           f"FDE = {r.get('FDE', float('nan')):.1f} km")
# # # # # # # # # # # #     print(f"  12h={r.get('12h', float('nan')):.0f}  "
# # # # # # # # # # # #           f"24h={r.get('24h', float('nan')):.0f}  "
# # # # # # # # # # # #           f"48h={r.get('48h', float('nan')):.0f}  "
# # # # # # # # # # # #           f"72h={r.get('72h', float('nan')):.0f} km")
# # # # # # # # # # # #     print(f"  SR-12h={sr_12h:.1f} km  SR-24h={sr_24h:.1f} km  "
# # # # # # # # # # # #           f"[TARGET: <50 / <100 km]")
# # # # # # # # # # # #     print(f"{'='*64}\n")

# # # # # # # # # # # #     r["sr_12h"] = sr_12h
# # # # # # # # # # # #     r["sr_24h"] = sr_24h

# # # # # # # # # # # #     from datetime import datetime
# # # # # # # # # # # #     from utils.metrics import DatasetMetrics, save_metrics_csv as _save_csv
# # # # # # # # # # # #     dm = DatasetMetrics(
# # # # # # # # # # # #         ade      = r.get("ADE",  float("nan")),
# # # # # # # # # # # #         fde      = r.get("FDE",  float("nan")),
# # # # # # # # # # # #         ugde_12h = r.get("12h",  float("nan")),
# # # # # # # # # # # #         ugde_24h = r.get("24h",  float("nan")),
# # # # # # # # # # # #         ugde_48h = r.get("48h",  float("nan")),
# # # # # # # # # # # #         ugde_72h = r.get("72h",  float("nan")),
# # # # # # # # # # # #         n_total  = r.get("n_samples", 0),
# # # # # # # # # # # #         timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # # # # # # # #     )
# # # # # # # # # # # #     _save_csv(dm, metrics_csv, tag=tag or f"val_full_ep{epoch:03d}")
# # # # # # # # # # # #     return r


# # # # # # # # # # # # def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
# # # # # # # # # # # #                   metrics_csv, tag="", predict_csv=""):
# # # # # # # # # # # #     model.eval()
# # # # # # # # # # # #     cliper_step_errors = []
# # # # # # # # # # # #     ev = TCEvaluator(pred_len=pred_len, compute_dtw=True)
# # # # # # # # # # # #     obs_seqs_01 = []; gt_seqs_01 = []; pred_seqs_01 = []; ens_seqs_01 = []

# # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # #         for batch in loader:
# # # # # # # # # # # #             bl  = move(list(batch), device)
# # # # # # # # # # # #             gt  = bl[1]; obs = bl[0]
# # # # # # # # # # # #             pred_mean, _, all_trajs = model.sample(
# # # # # # # # # # # #                 bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
# # # # # # # # # # # #                 predict_csv=predict_csv if predict_csv else None)

# # # # # # # # # # # #             pd_np = denorm_torch(pred_mean).cpu().numpy()
# # # # # # # # # # # #             gd_np = denorm_torch(gt).cpu().numpy()
# # # # # # # # # # # #             od_np = denorm_torch(obs).cpu().numpy()
# # # # # # # # # # # #             ed_np = denorm_torch(all_trajs).cpu().numpy()

# # # # # # # # # # # #             B = pd_np.shape[1]
# # # # # # # # # # # #             for b in range(B):
# # # # # # # # # # # #                 ens_b = ed_np[:, :, b, :]
# # # # # # # # # # # #                 ev.update(pd_np[:, b, :], gd_np[:, b, :], pred_ens=ens_b)
# # # # # # # # # # # #                 obs_seqs_01.append(od_np[:, b, :])
# # # # # # # # # # # #                 gt_seqs_01.append(gd_np[:, b, :])
# # # # # # # # # # # #                 pred_seqs_01.append(pd_np[:, b, :])
# # # # # # # # # # # #                 ens_seqs_01.append(ens_b)

# # # # # # # # # # # #                 obs_b_norm = obs.cpu().numpy()[:, b, :]
# # # # # # # # # # # #                 cliper_errors_b = np.zeros(pred_len)
# # # # # # # # # # # #                 for h in range(pred_len):
# # # # # # # # # # # #                     pred_cliper_norm = cliper_forecast(obs_b_norm, h + 1)
# # # # # # # # # # # #                     pred_cliper_01   = denorm_np(pred_cliper_norm[np.newaxis])
# # # # # # # # # # # #                     gt_01_np         = gd_np[h, b, :][np.newaxis]
# # # # # # # # # # # #                     from utils.metrics import haversine_km_np
# # # # # # # # # # # #                     cliper_errors_b[h] = float(
# # # # # # # # # # # #                         haversine_km_np(pred_cliper_01, gt_01_np, unit_01deg=True)[0])
# # # # # # # # # # # #                 cliper_step_errors.append(cliper_errors_b)

# # # # # # # # # # # #     if cliper_step_errors:
# # # # # # # # # # # #         cliper_mat       = np.stack(cliper_step_errors)
# # # # # # # # # # # #         cliper_ugde_dict = {h: float(cliper_mat[:, s].mean())
# # # # # # # # # # # #                             for h, s in HORIZON_STEPS.items()
# # # # # # # # # # # #                             if s < cliper_mat.shape[1]}
# # # # # # # # # # # #         ev.cliper_ugde = cliper_ugde_dict

# # # # # # # # # # # #     dm = ev.compute(tag=tag)

# # # # # # # # # # # #     try:
# # # # # # # # # # # #         if LANDFALL_TARGETS and ens_seqs_01:
# # # # # # # # # # # #             bss_vals = []
# # # # # # # # # # # #             step_72 = HORIZON_STEPS.get(72, pred_len - 1)
# # # # # # # # # # # #             for tname, t_lon, t_lat in LANDFALL_TARGETS:
# # # # # # # # # # # #                 bv = brier_skill_score(
# # # # # # # # # # # #                     ens_seqs_01, gt_seqs_01, min(step_72, pred_len-1),
# # # # # # # # # # # #                     (t_lon * 10.0, t_lat * 10.0), LANDFALL_RADIUS_KM)
# # # # # # # # # # # #                 if not math.isnan(bv):
# # # # # # # # # # # #                     bss_vals.append(bv)
# # # # # # # # # # # #             if bss_vals:
# # # # # # # # # # # #                 dm.bss_mean = float(np.mean(bss_vals))
# # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # #         print(f"  ⚠  BSS failed: {e}")

# # # # # # # # # # # #     save_metrics_csv(dm, metrics_csv, tag=tag)
# # # # # # # # # # # #     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# # # # # # # # # # # # # ── BestModelSaver ────────────────────────────────────────────────────────────

# # # # # # # # # # # # class BestModelSaver:
# # # # # # # # # # # #     def __init__(self, patience: int = 20, ade_tol: float = 3.0):
# # # # # # # # # # # #         self.patience      = patience
# # # # # # # # # # # #         self.ade_tol       = ade_tol
# # # # # # # # # # # #         self.best_ade      = float("inf")
# # # # # # # # # # # #         self.best_72h      = float("inf")
# # # # # # # # # # # #         self.best_val_loss = float("inf")
# # # # # # # # # # # #         self.counter       = 0
# # # # # # # # # # # #         self.early_stop    = False

# # # # # # # # # # # #     def reset_counters(self, reason: str = "") -> None:
# # # # # # # # # # # #         self.counter = 0
# # # # # # # # # # # #         if reason:
# # # # # # # # # # # #             print(f"  [SAVER] Patience reset: {reason}")

# # # # # # # # # # # #     def update(self, result: dict, model, out_dir, epoch,
# # # # # # # # # # # #                optimizer, tl, vl, min_epochs: int = 60):
# # # # # # # # # # # #         """Track blend ADE as primary metric (not just SR-12h)"""
# # # # # # # # # # # #         blend_ade = result.get("ADE", float("inf"))
# # # # # # # # # # # #         ade_72h   = result.get("72h", float("inf"))

# # # # # # # # # # # #         improved = blend_ade < self.best_ade - self.ade_tol
# # # # # # # # # # # #         if improved:
# # # # # # # # # # # #             self.best_ade = blend_ade
# # # # # # # # # # # #             self.best_72h = min(self.best_72h, ade_72h)
# # # # # # # # # # # #             self.counter  = 0
# # # # # # # # # # # #             torch.save(dict(
# # # # # # # # # # # #                 epoch=epoch, model_state_dict=model.state_dict(),
# # # # # # # # # # # #                 optimizer_state=optimizer.state_dict(),
# # # # # # # # # # # #                 train_loss=tl, val_loss=vl,
# # # # # # # # # # # #                 blend_ade=blend_ade, ade_72h=ade_72h,
# # # # # # # # # # # #                 model_version="v30-hierarchical"),
# # # # # # # # # # # #                 os.path.join(out_dir, "best_model.pth"))
# # # # # # # # # # # #             print(f"  ✅ Best ADE {blend_ade:.1f} km  72h={ade_72h:.1f} km  (epoch {epoch})")
# # # # # # # # # # # #         else:
# # # # # # # # # # # #             self.counter += 1
# # # # # # # # # # # #             print(f"  No ADE improvement {self.counter}/{self.patience}"
# # # # # # # # # # # #                   f"  (best={self.best_ade:.1f}  current={blend_ade:.1f})")

# # # # # # # # # # # #         if epoch >= min_epochs and self.counter >= self.patience:
# # # # # # # # # # # #             self.early_stop = True
# # # # # # # # # # # #             print(f"  ⛔ Early stop @ epoch {epoch}")


# # # # # # # # # # # # # ── GPH/UV diagnostic ─────────────────────────────────────────────────────────

# # # # # # # # # # # # def _check_env(bl, train_dataset):
# # # # # # # # # # # #     try:
# # # # # # # # # # # #         env_dir = train_dataset.env_path
# # # # # # # # # # # #     except AttributeError:
# # # # # # # # # # # #         try:
# # # # # # # # # # # #             env_dir = train_dataset.dataset.env_path
# # # # # # # # # # # #         except AttributeError:
# # # # # # # # # # # #             env_dir = "UNKNOWN"
# # # # # # # # # # # #     print(f"  Env path: {env_dir}")

# # # # # # # # # # # #     env_data = bl[13]
# # # # # # # # # # # #     if env_data is None:
# # # # # # # # # # # #         print("  ⚠️  env_data is None"); return

# # # # # # # # # # # #     for key in ("gph500_mean", "gph500_center", "u500_mean", "v500_mean"):
# # # # # # # # # # # #         if key not in env_data:
# # # # # # # # # # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # # # # # # # # # #         v    = env_data[key]
# # # # # # # # # # # #         mn   = v.mean().item()
# # # # # # # # # # # #         std  = v.std().item()
# # # # # # # # # # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # # # # # # # # # #         print(f"  {'✅' if zero < 80 else '⚠️'}  {key}: mean={mn:.4f} std={std:.4f} zero={zero:.1f}%")


# # # # # # # # # # # # def _load_baseline_errors(path, name):
# # # # # # # # # # # #     if path is None:
# # # # # # # # # # # #         print(f"\n  ⚠  {name} errors not provided.\n"); return None
# # # # # # # # # # # #     if not os.path.exists(path):
# # # # # # # # # # # #         print(f"\n  ⚠  {path} not found.\n"); return None
# # # # # # # # # # # #     arr = np.load(path)
# # # # # # # # # # # #     print(f"  ✓  Loaded {name}: {arr.shape}")
# # # # # # # # # # # #     return arr


# # # # # # # # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # # # # # # # def get_args():
# # # # # # # # # # # #     p = argparse.ArgumentParser(
# # # # # # # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # # # # # # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # # # # # # # # # # #     p.add_argument("--obs_len",         default=8,          type=int)
# # # # # # # # # # # #     p.add_argument("--pred_len",        default=12,         type=int)
# # # # # # # # # # # #     p.add_argument("--test_year",       default=None,       type=int)
# # # # # # # # # # # #     p.add_argument("--batch_size",      default=32,         type=int)
# # # # # # # # # # # #     p.add_argument("--num_epochs",      default=150,        type=int)
# # # # # # # # # # # #     p.add_argument("--g_learning_rate", default=2e-4,       type=float)
# # # # # # # # # # # #     p.add_argument("--weight_decay",    default=1e-4,       type=float)
# # # # # # # # # # # #     p.add_argument("--warmup_epochs",   default=5,          type=int)
# # # # # # # # # # # #     p.add_argument("--grad_clip",       default=2.0,        type=float)
# # # # # # # # # # # #     p.add_argument("--grad_accum",      default=2,          type=int)
# # # # # # # # # # # #     p.add_argument("--patience",        default=20,         type=int)
# # # # # # # # # # # #     p.add_argument("--min_epochs",      default=60,         type=int)
# # # # # # # # # # # #     p.add_argument("--n_train_ens",     default=6,          type=int)
# # # # # # # # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # # # # # # # #     p.add_argument("--num_workers",     default=2,          type=int)

# # # # # # # # # # # #     p.add_argument("--sigma_min",            default=0.02,   type=float)
# # # # # # # # # # # #     p.add_argument("--ctx_noise_scale",      default=0.01,  type=float)
# # # # # # # # # # # #     p.add_argument("--initial_sample_sigma", default=0.05,   type=float)

# # # # # # # # # # # #     p.add_argument("--ode_steps_train", default=20,  type=int)
# # # # # # # # # # # #     p.add_argument("--ode_steps_val",   default=30,  type=int)
# # # # # # # # # # # #     p.add_argument("--ode_steps_test",  default=50,  type=int)
# # # # # # # # # # # #     p.add_argument("--ode_steps",       default=None, type=int)

# # # # # # # # # # # #     p.add_argument("--val_ensemble",    default=30,  type=int)
# # # # # # # # # # # #     p.add_argument("--fast_ensemble",   default=8,   type=int)

# # # # # # # # # # # #     p.add_argument("--fno_modes_h",      default=4,  type=int)
# # # # # # # # # # # #     p.add_argument("--fno_modes_t",      default=4,  type=int)
# # # # # # # # # # # #     p.add_argument("--fno_layers",       default=4,  type=int)
# # # # # # # # # # # #     p.add_argument("--fno_d_model",      default=32, type=int)
# # # # # # # # # # # #     p.add_argument("--fno_spatial_down", default=32, type=int)
# # # # # # # # # # # #     p.add_argument("--mamba_d_state",    default=16, type=int)

# # # # # # # # # # # #     p.add_argument("--val_freq",         default=1,   type=int)
# # # # # # # # # # # #     p.add_argument("--val_ade_freq",     default=1,   type=int)
# # # # # # # # # # # #     p.add_argument("--full_eval_freq",   default=10,  type=int)
# # # # # # # # # # # #     p.add_argument("--val_subset_size",  default=600, type=int)

# # # # # # # # # # # #     p.add_argument("--output_dir",      default="runs/v30",      type=str)
# # # # # # # # # # # #     p.add_argument("--save_interval",   default=10,              type=int)
# # # # # # # # # # # #     p.add_argument("--metrics_csv",     default="metrics.csv",   type=str)
# # # # # # # # # # # #     p.add_argument("--predict_csv",     default="predictions.csv", type=str)
# # # # # # # # # # # #     p.add_argument("--lstm_errors_npy",      default=None, type=str)
# # # # # # # # # # # #     p.add_argument("--diffusion_errors_npy", default=None, type=str)

# # # # # # # # # # # #     p.add_argument("--gpu_num",         default="0", type=str)
# # # # # # # # # # # #     p.add_argument("--delim",           default=" ")
# # # # # # # # # # # #     p.add_argument("--skip",            default=1,   type=int)
# # # # # # # # # # # #     p.add_argument("--min_ped",         default=1,   type=int)
# # # # # # # # # # # #     p.add_argument("--threshold",       default=0.002, type=float)
# # # # # # # # # # # #     p.add_argument("--other_modal",     default="gph")

# # # # # # # # # # # #     p.add_argument("--step_weight_decay_epochs", default=40, type=int)
# # # # # # # # # # # #     p.add_argument("--lon_flip_prob",            default=0.3, type=float)

# # # # # # # # # # # #     p.add_argument("--pinn_warmup_epochs", default=60,    type=int)
# # # # # # # # # # # #     p.add_argument("--pinn_w_start",      default=0.001, type=float)
# # # # # # # # # # # #     p.add_argument("--pinn_w_end",        default=0.03,  type=float)

# # # # # # # # # # # #     p.add_argument("--vel_warmup_epochs",    default=20,  type=float)
# # # # # # # # # # # #     p.add_argument("--vel_w_start",          default=0.5, type=float)
# # # # # # # # # # # #     p.add_argument("--vel_w_end",            default=1.2, type=float)
# # # # # # # # # # # #     p.add_argument("--recurv_warmup_epochs", default=10,  type=int)
# # # # # # # # # # # #     p.add_argument("--recurv_w_start",       default=0.3, type=float)
# # # # # # # # # # # #     p.add_argument("--recurv_w_end",         default=1.0, type=float)

# # # # # # # # # # # #     return p.parse_args()


# # # # # # # # # # # # def _resolve_ode_steps(args):
# # # # # # # # # # # #     if args.ode_steps is not None:
# # # # # # # # # # # #         return args.ode_steps, args.ode_steps, args.ode_steps
# # # # # # # # # # # #     return args.ode_steps_train, args.ode_steps_val, args.ode_steps_test


# # # # # # # # # # # # # ── MAIN ──────────────────────────────────────────────────────────────────────

# # # # # # # # # # # # def main(args):
# # # # # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # # # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # # # # # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# # # # # # # # # # # #     predict_csv = os.path.join(args.output_dir, args.predict_csv)
# # # # # # # # # # # #     tables_dir  = os.path.join(args.output_dir, "tables")
# # # # # # # # # # # #     stat_dir    = os.path.join(tables_dir, "stat_tests")
# # # # # # # # # # # #     os.makedirs(tables_dir, exist_ok=True)
# # # # # # # # # # # #     os.makedirs(stat_dir,   exist_ok=True)

# # # # # # # # # # # #     ode_train, ode_val, ode_test = _resolve_ode_steps(args)

# # # # # # # # # # # #     print("=" * 70)
# # # # # # # # # # # #     print("  TC-FlowMatching v30  |  HIERARCHICAL SR+FM")
# # # # # # # # # # # #     print("  STRATEGY:")
# # # # # # # # # # # #     print("    - NO Phase freeze: train all from epoch 0")
# # # # # # # # # # # #     print("    - SR owns step 1-4 (6h-24h)")
# # # # # # # # # # # #     print("    - FM owns step 5-12 (30h-72h), starts from SR step 4")
# # # # # # # # # # # #     print("    - Continuity loss at SR→FM handoff")
# # # # # # # # # # # #     print("    - initial_sample_sigma = 0.03 (restored)")
# # # # # # # # # # # #     print("    - ctx_noise_scale = 0.002 (restored, no *10)")
# # # # # # # # # # # #     print("  TARGETS: 12h<50  24h<100  48h<200  72h<300 km")
# # # # # # # # # # # #     print("=" * 70)

# # # # # # # # # # # #     train_dataset, train_loader = data_loader(
# # # # # # # # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # # # # # # # #     val_dataset, val_loader = data_loader(
# # # # # # # # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # # # # # #     val_subset_loader = make_val_subset_loader(
# # # # # # # # # # # #         val_dataset, args.val_subset_size, args.batch_size,
# # # # # # # # # # # #         seq_collate, args.num_workers)

# # # # # # # # # # # #     test_loader = None
# # # # # # # # # # # #     try:
# # # # # # # # # # # #         _, test_loader = data_loader(
# # # # # # # # # # # #             args, {"root": args.dataset_root, "type": "test"},
# # # # # # # # # # # #             test=True, test_year=None)
# # # # # # # # # # # #     except Exception as e:
# # # # # # # # # # # #         print(f"  Warning: test loader: {e}")

# # # # # # # # # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # # # # # # # # #     print(f"  val   : {len(val_dataset)} seq")

# # # # # # # # # # # #     model = TCFlowMatching(
# # # # # # # # # # # #         pred_len             = args.pred_len,
# # # # # # # # # # # #         obs_len              = args.obs_len,
# # # # # # # # # # # #         sigma_min            = args.sigma_min,
# # # # # # # # # # # #         n_train_ens          = args.n_train_ens,
# # # # # # # # # # # #         ctx_noise_scale      = args.ctx_noise_scale,
# # # # # # # # # # # #         initial_sample_sigma = args.initial_sample_sigma,
# # # # # # # # # # # #     ).to(device)

# # # # # # # # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # # # # # # #     print(f"  params  : {n_params:,}")

# # # # # # # # # # # #     # ★ NO PHASE FREEZE: single optimizer from start
# # # # # # # # # # # #     optimizer = optim.AdamW(
# # # # # # # # # # # #         model.parameters(),
# # # # # # # # # # # #         lr=args.g_learning_rate,
# # # # # # # # # # # #         weight_decay=args.weight_decay,
# # # # # # # # # # # #     )

# # # # # # # # # # # #     try:
# # # # # # # # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # # # # # # # #         print("  torch.compile: enabled")
# # # # # # # # # # # #     except Exception:
# # # # # # # # # # # #         pass

# # # # # # # # # # # #     steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
# # # # # # # # # # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # # # # # # # # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # # # # # # # # # #     scheduler = get_cosine_schedule_with_warmup(
# # # # # # # # # # # #         optimizer, warmup, total_steps, min_lr=1e-6)

# # # # # # # # # # # #     saver  = BestModelSaver(patience=args.patience, ade_tol=3.0)
# # # # # # # # # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)

# # # # # # # # # # # #     print("=" * 70)
# # # # # # # # # # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch)")
# # # # # # # # # # # #     print("=" * 70)

# # # # # # # # # # # #     epoch_times   = []
# # # # # # # # # # # #     train_start   = time.perf_counter()
# # # # # # # # # # # #     last_val_loss = float("inf")
# # # # # # # # # # # #     _prev_ens     = 2

# # # # # # # # # # # #     import Model.losses as _losses_mod

# # # # # # # # # # # #     for epoch in range(args.num_epochs):

# # # # # # # # # # # #         # ── Weight schedules ──
# # # # # # # # # # # #         current_ens = get_progressive_ens(epoch, args.n_train_ens)
# # # # # # # # # # # #         model.n_train_ens = current_ens

# # # # # # # # # # # #         if current_ens != _prev_ens:
# # # # # # # # # # # #             saver.reset_counters(f"n_train_ens {_prev_ens}→{current_ens}")
# # # # # # # # # # # #             _prev_ens = current_ens

# # # # # # # # # # # #         step_alpha    = get_step_weight_alpha(epoch, args.step_weight_decay_epochs)
# # # # # # # # # # # #         epoch_weights = copy.copy(_BASE_WEIGHTS)
# # # # # # # # # # # #         epoch_weights["fm"]          = get_fm_weight(epoch)
# # # # # # # # # # # #         epoch_weights["short_range"] = get_short_range_weight(epoch)
# # # # # # # # # # # #         epoch_weights["continuity"]  = get_continuity_weight(epoch)
# # # # # # # # # # # #         epoch_weights["spread"]      = get_spread_weight(epoch)
# # # # # # # # # # # #         epoch_weights["pinn"]        = get_pinn_weight(
# # # # # # # # # # # #             epoch, args.pinn_warmup_epochs, args.pinn_w_start, args.pinn_w_end)
# # # # # # # # # # # #         epoch_weights["velocity"]    = get_velocity_weight(
# # # # # # # # # # # #             epoch, args.vel_warmup_epochs, args.vel_w_start, args.vel_w_end)
# # # # # # # # # # # #         epoch_weights["recurv"]      = get_recurv_weight(
# # # # # # # # # # # #             epoch, args.recurv_warmup_epochs, args.recurv_w_start, args.recurv_w_end)
# # # # # # # # # # # #         _losses_mod.WEIGHTS.update(epoch_weights)

# # # # # # # # # # # #         current_clip = get_grad_clip(epoch, warmup_epochs=20,
# # # # # # # # # # # #                                       clip_start=args.grad_clip, clip_end=1.0)

# # # # # # # # # # # #         # ── Training loop ──
# # # # # # # # # # # #         model.train()
# # # # # # # # # # # #         sum_loss   = 0.0
# # # # # # # # # # # #         t0         = time.perf_counter()
# # # # # # # # # # # #         optimizer.zero_grad()

# # # # # # # # # # # #         for i, batch in enumerate(train_loader):
# # # # # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # # # # #             if epoch == 0 and i == 0:
# # # # # # # # # # # #                 _check_env(bl, train_dataset)

# # # # # # # # # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # # # # #                 bd = model.get_loss_breakdown(
# # # # # # # # # # # #                     bl, step_weight_alpha=step_alpha, epoch=epoch)

# # # # # # # # # # # #             loss_to_bp = bd["total"] / max(args.grad_accum, 1)
# # # # # # # # # # # #             scaler.scale(loss_to_bp).backward()

# # # # # # # # # # # #             # if ((i + 1) % args.grad_accum == 0
# # # # # # # # # # # #             #         or (i + 1) == len(train_loader)):
# # # # # # # # # # # #             #     scaler.unscale_(optimizer)
# # # # # # # # # # # #             #     torch.nn.utils.clip_grad_norm_(
# # # # # # # # # # # #             #         model.parameters(), current_clip)
# # # # # # # # # # # #             #     scaler.step(optimizer)
# # # # # # # # # # # #             #     scaler.update()
# # # # # # # # # # # #             #     optimizer.zero_grad()
# # # # # # # # # # # #             #     scheduler.step()

# # # # # # # # # # # #             if ((i + 1) % args.grad_accum == 0
# # # # # # # # # # # #                     or (i + 1) == len(train_loader)):
# # # # # # # # # # # #                 scaler.unscale_(optimizer)
# # # # # # # # # # # #                 torch.nn.utils.clip_grad_norm_(
# # # # # # # # # # # #                     model.parameters(), current_clip)
# # # # # # # # # # # #                 scaler.step(optimizer)    # ← 1. optimizer trước
# # # # # # # # # # # #                 scaler.update()
# # # # # # # # # # # #                 scheduler.step()          # ← 2. scheduler sau  (HOÁN ĐỔI VỊ TRÍ)
# # # # # # # # # # # #                 optimizer.zero_grad()

# # # # # # # # # # # #             sum_loss += bd["total"].item()

# # # # # # # # # # # #             if i % 20 == 0:
# # # # # # # # # # # #                 lr   = optimizer.param_groups[0]["lr"]
# # # # # # # # # # # #                 elapsed = time.perf_counter() - t0
# # # # # # # # # # # #                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # # # # # # # # # #                       f"  loss={bd['total'].item():.3f}"
# # # # # # # # # # # #                       f"  fm={bd.get('fm',0):.2f}"
# # # # # # # # # # # #                       f"  sr={bd.get('short_range',0):.4f}"
# # # # # # # # # # # #                       f"  cont={bd.get('continuity',0):.4f}"
# # # # # # # # # # # #                       f"  pinn={bd.get('pinn',0):.4f}"
# # # # # # # # # # # #                       f"  spread={bd.get('spread',0):.4f}"
# # # # # # # # # # # #                       f"  ens={current_ens}"
# # # # # # # # # # # #                       f"  lr={lr:.2e}  t={elapsed:.0f}s")

# # # # # # # # # # # #         ep_s  = time.perf_counter() - t0
# # # # # # # # # # # #         epoch_times.append(ep_s)
# # # # # # # # # # # #         avg_t = sum_loss / len(train_loader)

# # # # # # # # # # # #         # ── Val loss ──
# # # # # # # # # # # #         model.eval()
# # # # # # # # # # # #         val_loss = 0.0
# # # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # # #             for batch in val_loader:
# # # # # # # # # # # #                 bl_v = move(list(batch), device)
# # # # # # # # # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # # # # #                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
# # # # # # # # # # # #         last_val_loss = val_loss / len(val_loader)

# # # # # # # # # # # #         print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
# # # # # # # # # # # #               f"  train_t={ep_s:.0f}s  ens={current_ens}")

# # # # # # # # # # # #         # ── Fast eval ──
# # # # # # # # # # # #         eff_fast_ens = min(args.fast_ensemble, max(current_ens * 2, args.fast_ensemble))
# # # # # # # # # # # #         m_fast = evaluate_fast(model, val_subset_loader, device,
# # # # # # # # # # # #                                 ode_train, args.pred_len, eff_fast_ens)

# # # # # # # # # # # #         sr_12h   = m_fast.get("sr_ade_12h", float("nan"))
# # # # # # # # # # # #         sr_24h   = m_fast.get("sr_ade_24h", float("nan"))
# # # # # # # # # # # #         ade_48h  = m_fast.get("48h", float("nan"))
# # # # # # # # # # # #         ade_72h  = m_fast.get("72h", float("nan"))
# # # # # # # # # # # #         spread72 = m_fast.get("spread_72h_km", 0.0)

# # # # # # # # # # # #         # Check targets
# # # # # # # # # # # #         t12 = "🎯" if sr_12h < 50 else "❌"
# # # # # # # # # # # #         t24 = "🎯" if sr_24h < 100 else "❌"
# # # # # # # # # # # #         t48 = "🎯" if ade_48h < 200 else "❌"
# # # # # # # # # # # #         t72 = "🎯" if ade_72h < 300 else "❌"

# # # # # # # # # # # #         print(f"  [FAST ep{epoch}]"
# # # # # # # # # # # #               f"  ADE={m_fast['ADE']:.1f}  FDE={m_fast['FDE']:.1f} km"
# # # # # # # # # # # #               f"  12h={m_fast.get('12h',float('nan')):.0f}{t12}"
# # # # # # # # # # # #               f"  24h={m_fast.get('24h',float('nan')):.0f}{t24}"
# # # # # # # # # # # #               f"  48h={ade_48h:.0f}{t48}"
# # # # # # # # # # # #               f"  72h={ade_72h:.0f}{t72}"
# # # # # # # # # # # #               f"  spread={spread72:.0f}km")
# # # # # # # # # # # #         print(f"         SR: 12h={sr_12h:.1f} km  24h={sr_24h:.1f} km")

# # # # # # # # # # # #         # ── Full val ADE ──
# # # # # # # # # # # #         if epoch % args.val_ade_freq == 0:
# # # # # # # # # # # #             try:
# # # # # # # # # # # #                 r_full = evaluate_full_val_ade(
# # # # # # # # # # # #                     model, val_loader, device,
# # # # # # # # # # # #                     ode_steps=ode_train, pred_len=args.pred_len,
# # # # # # # # # # # #                     fast_ensemble=eff_fast_ens, metrics_csv=metrics_csv,
# # # # # # # # # # # #                     epoch=epoch)
# # # # # # # # # # # #                 saver.update(r_full, model, args.output_dir, epoch,
# # # # # # # # # # # #                            optimizer, avg_t, last_val_loss,
# # # # # # # # # # # #                            min_epochs=args.min_epochs)
# # # # # # # # # # # #             except Exception as e:
# # # # # # # # # # # #                 print(f"  ⚠  Full val failed: {e}")
# # # # # # # # # # # #                 import traceback; traceback.print_exc()

# # # # # # # # # # # #         # ── Checkpoint ──
# # # # # # # # # # # #         torch.save({
# # # # # # # # # # # #             "epoch": epoch,
# # # # # # # # # # # #             "model_state_dict": model.state_dict(),
# # # # # # # # # # # #             "optimizer_state": optimizer.state_dict(),
# # # # # # # # # # # #             "train_loss": avg_t,
# # # # # # # # # # # #             "val_loss": last_val_loss,
# # # # # # # # # # # #             "blend_ade": m_fast.get("ADE", float("nan")),
# # # # # # # # # # # #             "sr_12h": sr_12h,
# # # # # # # # # # # #             "ade_72h": ade_72h,
# # # # # # # # # # # #         }, os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))

# # # # # # # # # # # #         if epoch % 5 == 4:
# # # # # # # # # # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # # # # # # # # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # # # # # # # # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # # # # # # # # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")

# # # # # # # # # # # #         if saver.early_stop:
# # # # # # # # # # # #             print(f"  ⛔ Early stopping at epoch {epoch}")
# # # # # # # # # # # #             break

# # # # # # # # # # # #     # ── Final test ──
# # # # # # # # # # # #     total_train_h = (time.perf_counter() - train_start) / 3600
# # # # # # # # # # # #     print(f"\n{'='*70}  FINAL TEST (ode_steps={ode_test})")

# # # # # # # # # # # #     if test_loader:
# # # # # # # # # # # #         best_path = os.path.join(args.output_dir, "best_model.pth")
# # # # # # # # # # # #         if os.path.exists(best_path):
# # # # # # # # # # # #             ck = torch.load(best_path, map_location=device)
# # # # # # # # # # # #             try:
# # # # # # # # # # # #                 model.load_state_dict(ck["model_state_dict"])
# # # # # # # # # # # #             except Exception:
# # # # # # # # # # # #                 model.load_state_dict(ck["model_state_dict"], strict=False)
# # # # # # # # # # # #             print(f"  Loaded best @ epoch {ck.get('epoch','?')}")

# # # # # # # # # # # #         final_ens = max(args.val_ensemble, 50)
# # # # # # # # # # # #         dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
# # # # # # # # # # # #             model, test_loader, device,
# # # # # # # # # # # #             ode_test, args.pred_len, final_ens,
# # # # # # # # # # # #             metrics_csv=metrics_csv, tag="test_final",
# # # # # # # # # # # #             predict_csv=predict_csv)
# # # # # # # # # # # #         print(dm_test.summary())

# # # # # # # # # # # #     print(f"\n  Best ADE           : {saver.best_ade:.1f} km")
# # # # # # # # # # # #     print(f"  Best 72h           : {saver.best_72h:.1f} km")
# # # # # # # # # # # #     print(f"  Best val loss      : {saver.best_val_loss:.4f}")
# # # # # # # # # # # #     print(f"  Total training     : {total_train_h:.2f}h")
# # # # # # # # # # # #     print("=" * 70)


# # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # #     args = get_args()
# # # # # # # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # # # # #         torch.cuda.manual_seed_all(42)
# # # # # # # # # # # #     main(args)
# # # # # # # # # # # """
# # # # # # # # # # # train_flowmatching_v33.py — Simplified training for Pure FM + MSE
# # # # # # # # # # # ═══════════════════════════════════════════════════════════════════

# # # # # # # # # # # KEY CHANGES từ v30 train script:
# # # # # # # # # # #   1. BỎ tất cả weight schedules phức tạp → chỉ cần MSE
# # # # # # # # # # #   2. BỎ SR-specific evaluation → FM predict all steps
# # # # # # # # # # #   3. BỎ AFCRPS, PINN, continuity, spread schedules
# # # # # # # # # # #   4. THÊM sigma_min schedule logging
# # # # # # # # # # #   5. Learning rate: cosine với warmup, peak = 3e-4 (cao hơn v32)
# # # # # # # # # # #   6. Grad clip: 1.0 constant (không cần schedule)
# # # # # # # # # # #   7. Grad accum = 1 (đơn giản hóa)
# # # # # # # # # # # """
# # # # # # # # # # # from __future__ import annotations

# # # # # # # # # # # import sys
# # # # # # # # # # # import os
# # # # # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # # # # import argparse
# # # # # # # # # # # import time
# # # # # # # # # # # import math
# # # # # # # # # # # import random
# # # # # # # # # # # import copy

# # # # # # # # # # # import numpy as np
# # # # # # # # # # # import torch
# # # # # # # # # # # import torch.optim as optim
# # # # # # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # # # # from Model.data.loader_training import data_loader
# # # # # # # # # # # from Model.flow_matching_model import TCFlowMatching
# # # # # # # # # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # # # # # from utils.metrics import (
# # # # # # # # # # #     TCEvaluator, StepErrorAccumulator,
# # # # # # # # # # #     save_metrics_csv, haversine_km_torch,
# # # # # # # # # # #     denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
# # # # # # # # # # #     cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
# # # # # # # # # # #     brier_skill_score, cliper_errors, persistence_errors,
# # # # # # # # # # # )


# # # # # # # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # # # # # # def move(batch, device):
# # # # # # # # # # #     out = list(batch)
# # # # # # # # # # #     for i, x in enumerate(out):
# # # # # # # # # # #         if torch.is_tensor(x):
# # # # # # # # # # #             out[i] = x.to(device)
# # # # # # # # # # #         elif isinstance(x, dict):
# # # # # # # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # # # # #                       for k, v in x.items()}
# # # # # # # # # # #     return out


# # # # # # # # # # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # # # # # # # # # #                             collate_fn, num_workers):
# # # # # # # # # # #     n   = len(val_dataset)
# # # # # # # # # # #     rng = random.Random(42)
# # # # # # # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # # # # # # #     return DataLoader(Subset(val_dataset, idx),
# # # # # # # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # # # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # # # # # # # ── Evaluation ────────────────────────────────────────────────────────────────

# # # # # # # # # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
# # # # # # # # # # #     """Quick evaluation on subset."""
# # # # # # # # # # #     model.eval()
# # # # # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # # # # #     t0  = time.perf_counter()
# # # # # # # # # # #     n   = 0
# # # # # # # # # # #     spread_per_step = []

# # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # #         for batch in loader:
# # # # # # # # # # #             bl = move(list(batch), device)
# # # # # # # # # # #             pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
# # # # # # # # # # #                                               ddim_steps=ode_steps)
# # # # # # # # # # #             T_active  = pred.shape[0]
# # # # # # # # # # #             gt_sliced = bl[1][:T_active]
# # # # # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
# # # # # # # # # # #             acc.update(dist)

# # # # # # # # # # #             # Spread per step
# # # # # # # # # # #             step_spreads = []
# # # # # # # # # # #             for t in range(all_trajs.shape[1]):
# # # # # # # # # # #                 step_data = all_trajs[:, t, :, :]
# # # # # # # # # # #                 std_lon = step_data[:, :, 0].std(0)
# # # # # # # # # # #                 std_lat = step_data[:, :, 1].std(0)
# # # # # # # # # # #                 spread = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# # # # # # # # # # #                 step_spreads.append(spread)
# # # # # # # # # # #             spread_per_step.append(step_spreads)
# # # # # # # # # # #             n += 1

# # # # # # # # # # #     r = acc.compute()
# # # # # # # # # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)

# # # # # # # # # # #     if spread_per_step:
# # # # # # # # # # #         spreads = np.array(spread_per_step)
# # # # # # # # # # #         r["spread_12h_km"] = float(spreads[:, 1].mean()) if spreads.shape[1] > 1 else 0.0
# # # # # # # # # # #         r["spread_24h_km"] = float(spreads[:, 3].mean()) if spreads.shape[1] > 3 else 0.0
# # # # # # # # # # #         r["spread_72h_km"] = float(spreads[:, -1].mean())
# # # # # # # # # # #     return r


# # # # # # # # # # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # # # # # # # # # #                            fast_ensemble, metrics_csv, epoch):
# # # # # # # # # # #     """Full validation ADE computation."""
# # # # # # # # # # #     model.eval()
# # # # # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # # # # #     t0  = time.perf_counter()
# # # # # # # # # # #     n   = 0

# # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # #         for batch in val_loader:
# # # # # # # # # # #             bl = move(list(batch), device)
# # # # # # # # # # #             pred, _, _ = model.sample(bl, num_ensemble=fast_ensemble,
# # # # # # # # # # #                                        ddim_steps=ode_steps)
# # # # # # # # # # #             T_pred = pred.shape[0]
# # # # # # # # # # #             gt = bl[1][:T_pred]
# # # # # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt))
# # # # # # # # # # #             acc.update(dist)
# # # # # # # # # # #             n += 1

# # # # # # # # # # #     r = acc.compute()
# # # # # # # # # # #     elapsed = time.perf_counter() - t0

# # # # # # # # # # #     print(f"\n{'='*64}")
# # # # # # # # # # #     print(f"  [FULL VAL ADE  ep={epoch}  {elapsed:.0f}s  {n} batches]")
# # # # # # # # # # #     print(f"  ADE = {r.get('ADE', float('nan')):.1f} km  "
# # # # # # # # # # #           f"FDE = {r.get('FDE', float('nan')):.1f} km")
# # # # # # # # # # #     print(f"  6h={r.get('6h', float('nan')):.0f}  "
# # # # # # # # # # #           f"12h={r.get('12h', float('nan')):.0f}  "
# # # # # # # # # # #           f"24h={r.get('24h', float('nan')):.0f}  "
# # # # # # # # # # #           f"48h={r.get('48h', float('nan')):.0f}  "
# # # # # # # # # # #           f"72h={r.get('72h', float('nan')):.0f} km")
# # # # # # # # # # #     print(f"{'='*64}\n")

# # # # # # # # # # #     from datetime import datetime
# # # # # # # # # # #     from utils.metrics import DatasetMetrics, save_metrics_csv as _save_csv
# # # # # # # # # # #     dm = DatasetMetrics(
# # # # # # # # # # #         ade      = r.get("ADE",  float("nan")),
# # # # # # # # # # #         fde      = r.get("FDE",  float("nan")),
# # # # # # # # # # #         ugde_12h = r.get("12h",  float("nan")),
# # # # # # # # # # #         ugde_24h = r.get("24h",  float("nan")),
# # # # # # # # # # #         ugde_48h = r.get("48h",  float("nan")),
# # # # # # # # # # #         ugde_72h = r.get("72h",  float("nan")),
# # # # # # # # # # #         n_total  = r.get("n_samples", 0),
# # # # # # # # # # #         timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # # # # # # #     )
# # # # # # # # # # #     _save_csv(dm, metrics_csv, tag=f"val_full_ep{epoch:03d}")
# # # # # # # # # # #     return r


# # # # # # # # # # # # ── BestModelSaver ────────────────────────────────────────────────────────────

# # # # # # # # # # # class BestModelSaver:
# # # # # # # # # # #     def __init__(self, patience: int = 25, ade_tol: float = 2.0):
# # # # # # # # # # #         self.patience   = patience
# # # # # # # # # # #         self.ade_tol    = ade_tol
# # # # # # # # # # #         self.best_ade   = float("inf")
# # # # # # # # # # #         self.best_72h   = float("inf")
# # # # # # # # # # #         self.counter    = 0
# # # # # # # # # # #         self.early_stop = False

# # # # # # # # # # #     def update(self, result, model, out_dir, epoch, optimizer, tl, vl,
# # # # # # # # # # #                min_epochs=60):
# # # # # # # # # # #         ade   = result.get("ADE", float("inf"))
# # # # # # # # # # #         h72   = result.get("72h", float("inf"))

# # # # # # # # # # #         if ade < self.best_ade - self.ade_tol:
# # # # # # # # # # #             self.best_ade = ade
# # # # # # # # # # #             self.best_72h = min(self.best_72h, h72)
# # # # # # # # # # #             self.counter  = 0
# # # # # # # # # # #             torch.save(dict(
# # # # # # # # # # #                 epoch=epoch, model_state_dict=model.state_dict(),
# # # # # # # # # # #                 optimizer_state=optimizer.state_dict(),
# # # # # # # # # # #                 train_loss=tl, val_loss=vl,
# # # # # # # # # # #                 ade=ade, ade_72h=h72,
# # # # # # # # # # #                 model_version="v33-pure-fm-mse"),
# # # # # # # # # # #                 os.path.join(out_dir, "best_model.pth"))
# # # # # # # # # # #             print(f"  ✅ Best ADE {ade:.1f} km  72h={h72:.1f} km  (epoch {epoch})")
# # # # # # # # # # #         else:
# # # # # # # # # # #             self.counter += 1
# # # # # # # # # # #             print(f"  No improvement {self.counter}/{self.patience}"
# # # # # # # # # # #                   f"  (best={self.best_ade:.1f}  current={ade:.1f})")

# # # # # # # # # # #         if epoch >= min_epochs and self.counter >= self.patience:
# # # # # # # # # # #             self.early_stop = True


# # # # # # # # # # # # ── Env diagnostic ────────────────────────────────────────────────────────────

# # # # # # # # # # # def _check_env(bl, train_dataset):
# # # # # # # # # # #     try:
# # # # # # # # # # #         env_dir = train_dataset.env_path
# # # # # # # # # # #     except AttributeError:
# # # # # # # # # # #         try:
# # # # # # # # # # #             env_dir = train_dataset.dataset.env_path
# # # # # # # # # # #         except AttributeError:
# # # # # # # # # # #             env_dir = "UNKNOWN"
# # # # # # # # # # #     print(f"  Env path: {env_dir}")

# # # # # # # # # # #     env_data = bl[13]
# # # # # # # # # # #     if env_data is None:
# # # # # # # # # # #         print("  ⚠️  env_data is None"); return

# # # # # # # # # # #     for key in ("gph500_mean", "gph500_center", "u500_mean", "v500_mean"):
# # # # # # # # # # #         if key not in env_data:
# # # # # # # # # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # # # # # # # # #         v    = env_data[key]
# # # # # # # # # # #         mn   = v.mean().item()
# # # # # # # # # # #         std  = v.std().item()
# # # # # # # # # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # # # # # # # # #         print(f"  {'✅' if zero < 80 else '⚠️'}  {key}: mean={mn:.4f} std={std:.4f} zero={zero:.1f}%")


# # # # # # # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # # # # # # def get_args():
# # # # # # # # # # #     p = argparse.ArgumentParser(
# # # # # # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # # # # # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # # # # # # # # # #     p.add_argument("--obs_len",         default=8,          type=int)
# # # # # # # # # # #     p.add_argument("--pred_len",        default=12,         type=int)
# # # # # # # # # # #     p.add_argument("--batch_size",      default=32,         type=int)
# # # # # # # # # # #     p.add_argument("--num_epochs",      default=120,        type=int)
# # # # # # # # # # #     p.add_argument("--g_learning_rate", default=3e-4,       type=float)
# # # # # # # # # # #     p.add_argument("--weight_decay",    default=1e-4,       type=float)
# # # # # # # # # # #     p.add_argument("--warmup_epochs",   default=5,          type=int)
# # # # # # # # # # #     p.add_argument("--grad_clip",       default=1.0,        type=float)
# # # # # # # # # # #     p.add_argument("--patience",        default=25,         type=int)
# # # # # # # # # # #     p.add_argument("--min_epochs",      default=50,         type=int)
# # # # # # # # # # #     p.add_argument("--n_train_ens",     default=4,          type=int)
# # # # # # # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # # # # # # #     p.add_argument("--num_workers",     default=2,          type=int)

# # # # # # # # # # #     p.add_argument("--sigma_min",            default=0.02,   type=float)
# # # # # # # # # # #     p.add_argument("--ctx_noise_scale",      default=0.005,  type=float)
# # # # # # # # # # #     p.add_argument("--initial_sample_sigma", default=0.03,   type=float)

# # # # # # # # # # #     p.add_argument("--ode_steps_train", default=20,  type=int)
# # # # # # # # # # #     p.add_argument("--ode_steps_val",   default=30,  type=int)
# # # # # # # # # # #     p.add_argument("--ode_steps_test",  default=50,  type=int)

# # # # # # # # # # #     p.add_argument("--val_ensemble",    default=30,  type=int)
# # # # # # # # # # #     p.add_argument("--fast_ensemble",   default=8,   type=int)

# # # # # # # # # # #     p.add_argument("--val_freq",         default=1,   type=int)
# # # # # # # # # # #     p.add_argument("--val_ade_freq",     default=1,   type=int)
# # # # # # # # # # #     p.add_argument("--val_subset_size",  default=600, type=int)

# # # # # # # # # # #     p.add_argument("--output_dir",      default="runs/v33",      type=str)
# # # # # # # # # # #     p.add_argument("--metrics_csv",     default="metrics.csv",   type=str)
# # # # # # # # # # #     p.add_argument("--predict_csv",     default="predictions.csv", type=str)

# # # # # # # # # # #     p.add_argument("--gpu_num",         default="0", type=str)
# # # # # # # # # # #     p.add_argument("--delim",           default=" ")
# # # # # # # # # # #     p.add_argument("--skip",            default=1,   type=int)
# # # # # # # # # # #     p.add_argument("--min_ped",         default=1,   type=int)
# # # # # # # # # # #     p.add_argument("--threshold",       default=0.002, type=float)
# # # # # # # # # # #     p.add_argument("--other_modal",     default="gph")
# # # # # # # # # # #     p.add_argument("--test_year",       default=None, type=int)

# # # # # # # # # # #     return p.parse_args()


# # # # # # # # # # # # ── MAIN ──────────────────────────────────────────────────────────────────────

# # # # # # # # # # # def main(args):
# # # # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # # # # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)

# # # # # # # # # # #     print("=" * 70)
# # # # # # # # # # #     print("  TC-FlowMatching v33  |  PURE FM + MSE-ONLY")
# # # # # # # # # # #     print("  ─────────────────────────────────────────────")
# # # # # # # # # # #     print("  STRATEGY:")
# # # # # # # # # # #     print("    - FM predicts ALL 12 steps (no separate SR head)")
# # # # # # # # # # #     print("    - Training loss = MSE_haversine only (+ light vel/heading)")
# # # # # # # # # # #     print("    - NO AFCRPS, NO PINN, NO spread, NO continuity")
# # # # # # # # # # #     print("    - Sigma schedule: 0.15 → 0.03 (deterministic → stochastic)")
# # # # # # # # # # #     print("    - Same powerful encoder (FNO3D + Mamba + Env_net)")
# # # # # # # # # # #     print("  TARGETS: 12h<50  24h<100  48h<200  72h<300 km")
# # # # # # # # # # #     print("=" * 70)

# # # # # # # # # # #     # ── Data ──────────────────────────────────────────────────────────
# # # # # # # # # # #     train_dataset, train_loader = data_loader(
# # # # # # # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # # # # # # #     val_dataset, val_loader = data_loader(
# # # # # # # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # # # # #     val_subset_loader = make_val_subset_loader(
# # # # # # # # # # #         val_dataset, args.val_subset_size, args.batch_size,
# # # # # # # # # # #         seq_collate, args.num_workers)

# # # # # # # # # # #     test_loader = None
# # # # # # # # # # #     try:
# # # # # # # # # # #         _, test_loader = data_loader(
# # # # # # # # # # #             args, {"root": args.dataset_root, "type": "test"},
# # # # # # # # # # #             test=True, test_year=None)
# # # # # # # # # # #     except Exception as e:
# # # # # # # # # # #         print(f"  Warning: test loader: {e}")

# # # # # # # # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # # # # # # # #     print(f"  val   : {len(val_dataset)} seq")

# # # # # # # # # # #     # ── Model ─────────────────────────────────────────────────────────
# # # # # # # # # # #     model = TCFlowMatching(
# # # # # # # # # # #         pred_len             = args.pred_len,
# # # # # # # # # # #         obs_len              = args.obs_len,
# # # # # # # # # # #         sigma_min            = args.sigma_min,
# # # # # # # # # # #         n_train_ens          = args.n_train_ens,
# # # # # # # # # # #         ctx_noise_scale      = args.ctx_noise_scale,
# # # # # # # # # # #         initial_sample_sigma = args.initial_sample_sigma,
# # # # # # # # # # #     ).to(device)

# # # # # # # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # # # # # #     print(f"  params  : {n_params:,}")

# # # # # # # # # # #     # ── Optimizer ─────────────────────────────────────────────────────
# # # # # # # # # # #     optimizer = optim.AdamW(
# # # # # # # # # # #         model.parameters(),
# # # # # # # # # # #         lr=args.g_learning_rate,
# # # # # # # # # # #         weight_decay=args.weight_decay,
# # # # # # # # # # #     )

# # # # # # # # # # #     try:
# # # # # # # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # # # # # # #         print("  torch.compile: enabled")
# # # # # # # # # # #     except Exception:
# # # # # # # # # # #         pass

# # # # # # # # # # #     steps_per_epoch = len(train_loader)
# # # # # # # # # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # # # # # # # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # # # # # # # # #     scheduler = get_cosine_schedule_with_warmup(
# # # # # # # # # # #         optimizer, warmup, total_steps, min_lr=1e-6)

# # # # # # # # # # #     saver  = BestModelSaver(patience=args.patience, ade_tol=2.0)
# # # # # # # # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)

# # # # # # # # # # #     print("=" * 70)
# # # # # # # # # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch)")
# # # # # # # # # # #     print("=" * 70)

# # # # # # # # # # #     epoch_times = []
# # # # # # # # # # #     train_start = time.perf_counter()

# # # # # # # # # # #     for epoch in range(args.num_epochs):
# # # # # # # # # # #         model.train()
# # # # # # # # # # #         sum_loss = 0.0
# # # # # # # # # # #         t0 = time.perf_counter()

# # # # # # # # # # #         for i, batch in enumerate(train_loader):
# # # # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # # # #             if epoch == 0 and i == 0:
# # # # # # # # # # #                 _check_env(bl, train_dataset)

# # # # # # # # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # # # #                 bd = model.get_loss_breakdown(bl, epoch=epoch)

# # # # # # # # # # #             optimizer.zero_grad()
# # # # # # # # # # #             scaler.scale(bd["total"]).backward()
# # # # # # # # # # #             scaler.unscale_(optimizer)
# # # # # # # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # # # # # # # #             scaler.step(optimizer)
# # # # # # # # # # #             scaler.update()
# # # # # # # # # # #             scheduler.step()

# # # # # # # # # # #             sum_loss += bd["total"].item()

# # # # # # # # # # #             if i % 20 == 0:
# # # # # # # # # # #                 lr = optimizer.param_groups[0]["lr"]
# # # # # # # # # # #                 elapsed = time.perf_counter() - t0
# # # # # # # # # # #                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # # # # # # # # #                       f"  loss={bd['total'].item():.4f}"
# # # # # # # # # # #                       f"  mse={bd['mse_hav']:.4f}"
# # # # # # # # # # #                       f"  vel={bd['velocity']:.4f}"
# # # # # # # # # # #                       f"  head={bd['heading']:.4f}"
# # # # # # # # # # #                       f"  σ={bd['sigma']:.3f}"
# # # # # # # # # # #                       f"  lr={lr:.2e}  t={elapsed:.0f}s")

# # # # # # # # # # #         ep_s  = time.perf_counter() - t0
# # # # # # # # # # #         epoch_times.append(ep_s)
# # # # # # # # # # #         avg_t = sum_loss / len(train_loader)

# # # # # # # # # # #         # ── Val loss ──────────────────────────────────────────────────
# # # # # # # # # # #         model.eval()
# # # # # # # # # # #         val_loss = 0.0
# # # # # # # # # # #         with torch.no_grad():
# # # # # # # # # # #             for batch in val_loader:
# # # # # # # # # # #                 bl_v = move(list(batch), device)
# # # # # # # # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # # # #                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
# # # # # # # # # # #         avg_vl = val_loss / len(val_loader)

# # # # # # # # # # #         print(f"  Epoch {epoch:>3}  train={avg_t:.4f}  val={avg_vl:.4f}"
# # # # # # # # # # #               f"  time={ep_s:.0f}s")

# # # # # # # # # # #         # ── Fast eval ─────────────────────────────────────────────────
# # # # # # # # # # #         m_fast = evaluate_fast(model, val_subset_loader, device,
# # # # # # # # # # #                                 args.ode_steps_train, args.pred_len,
# # # # # # # # # # #                                 args.fast_ensemble)

# # # # # # # # # # #         h6  = m_fast.get("6h",  float("nan"))
# # # # # # # # # # #         h12 = m_fast.get("12h", float("nan"))
# # # # # # # # # # #         h24 = m_fast.get("24h", float("nan"))
# # # # # # # # # # #         h48 = m_fast.get("48h", float("nan"))
# # # # # # # # # # #         h72 = m_fast.get("72h", float("nan"))
# # # # # # # # # # #         sp72 = m_fast.get("spread_72h_km", 0.0)

# # # # # # # # # # #         t6  = "🎯" if h6  < 30  else "❌"
# # # # # # # # # # #         t12 = "🎯" if h12 < 50  else "❌"
# # # # # # # # # # #         t24 = "🎯" if h24 < 100 else "❌"
# # # # # # # # # # #         t48 = "🎯" if h48 < 200 else "❌"
# # # # # # # # # # #         t72 = "🎯" if h72 < 300 else "❌"

# # # # # # # # # # #         print(f"  [FAST ep{epoch}]"
# # # # # # # # # # #               f"  ADE={m_fast['ADE']:.1f}"
# # # # # # # # # # #               f"  6h={h6:.0f}{t6}"
# # # # # # # # # # #               f"  12h={h12:.0f}{t12}"
# # # # # # # # # # #               f"  24h={h24:.0f}{t24}"
# # # # # # # # # # #               f"  48h={h48:.0f}{t48}"
# # # # # # # # # # #               f"  72h={h72:.0f}{t72}"
# # # # # # # # # # #               f"  spread72={sp72:.0f}km")

# # # # # # # # # # #         # ── Full val ADE ──────────────────────────────────────────────
# # # # # # # # # # #         if epoch % args.val_ade_freq == 0:
# # # # # # # # # # #             try:
# # # # # # # # # # #                 r_full = evaluate_full_val_ade(
# # # # # # # # # # #                     model, val_loader, device,
# # # # # # # # # # #                     ode_steps=args.ode_steps_train, pred_len=args.pred_len,
# # # # # # # # # # #                     fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # # # # # # # #                     epoch=epoch)
# # # # # # # # # # #                 saver.update(r_full, model, args.output_dir, epoch,
# # # # # # # # # # #                            optimizer, avg_t, avg_vl,
# # # # # # # # # # #                            min_epochs=args.min_epochs)
# # # # # # # # # # #             except Exception as e:
# # # # # # # # # # #                 print(f"  ⚠  Full val failed: {e}")
# # # # # # # # # # #                 import traceback; traceback.print_exc()

# # # # # # # # # # #         # ── Checkpoint ────────────────────────────────────────────────
# # # # # # # # # # #         if epoch % 5 == 0 or epoch == args.num_epochs - 1:
# # # # # # # # # # #             torch.save({
# # # # # # # # # # #                 "epoch": epoch,
# # # # # # # # # # #                 "model_state_dict": model.state_dict(),
# # # # # # # # # # #                 "optimizer_state": optimizer.state_dict(),
# # # # # # # # # # #                 "train_loss": avg_t, "val_loss": avg_vl,
# # # # # # # # # # #                 "ade": m_fast.get("ADE", float("nan")),
# # # # # # # # # # #             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))

# # # # # # # # # # #         if epoch % 10 == 9:
# # # # # # # # # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # # # # # # # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # # # # # # # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # # # # # # # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")

# # # # # # # # # # #         if saver.early_stop:
# # # # # # # # # # #             print(f"  ⛔ Early stopping at epoch {epoch}")
# # # # # # # # # # #             break

# # # # # # # # # # #     # ── Final ─────────────────────────────────────────────────────────
# # # # # # # # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # # # # # # # #     print(f"\n{'='*70}")
# # # # # # # # # # #     print(f"  Best ADE  : {saver.best_ade:.1f} km")
# # # # # # # # # # #     print(f"  Best 72h  : {saver.best_72h:.1f} km")
# # # # # # # # # # #     print(f"  Training  : {total_h:.2f}h")
# # # # # # # # # # #     print("=" * 70)


# # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # #     args = get_args()
# # # # # # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # # # #         torch.cuda.manual_seed_all(42)
# # # # # # # # # # #     main(args)

# # # # # # # # # # """
# # # # # # # # # # train_flowmatching_v33.py — Fixed training for Pure FM + MSE
# # # # # # # # # # ═══════════════════════════════════════════════════════════════════

# # # # # # # # # # FIX CHANGES:
# # # # # # # # # #   1. BestModelSaver: composite score = 0.25*ADE + 0.35*48h + 0.40*72h
# # # # # # # # # #      → Không còn save model tốt về 12h nhưng tệ về 72h
# # # # # # # # # #   2. evaluate_fast: tăng fast_ensemble lên 15, ode_steps lên 25
# # # # # # # # # #      → Val estimate ổn định hơn, ít noise hơn
# # # # # # # # # #   3. evaluate_full_val_ade: tăng fast_ensemble lên 20, ddim_steps lên 30
# # # # # # # # # #      → Final eval chính xác hơn (median ensemble)
# # # # # # # # # #   4. Log thêm long_range loss và w_lr trong training loop
# # # # # # # # # #   5. Print cả composite score để dễ track
# # # # # # # # # # """
# # # # # # # # # # from __future__ import annotations

# # # # # # # # # # import sys
# # # # # # # # # # import os
# # # # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # # # import argparse
# # # # # # # # # # import time
# # # # # # # # # # import math
# # # # # # # # # # import random
# # # # # # # # # # import copy

# # # # # # # # # # import numpy as np
# # # # # # # # # # import torch
# # # # # # # # # # import torch.optim as optim
# # # # # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # # # from Model.data.loader_training import data_loader
# # # # # # # # # # from Model.flow_matching_model import TCFlowMatching
# # # # # # # # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # # # # from utils.metrics import (
# # # # # # # # # #     TCEvaluator, StepErrorAccumulator,
# # # # # # # # # #     save_metrics_csv, haversine_km_torch,
# # # # # # # # # #     denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
# # # # # # # # # #     cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
# # # # # # # # # #     brier_skill_score, cliper_errors, persistence_errors,
# # # # # # # # # # )


# # # # # # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # # # # # def move(batch, device):
# # # # # # # # # #     out = list(batch)
# # # # # # # # # #     for i, x in enumerate(out):
# # # # # # # # # #         if torch.is_tensor(x):
# # # # # # # # # #             out[i] = x.to(device)
# # # # # # # # # #         elif isinstance(x, dict):
# # # # # # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # # # #                       for k, v in x.items()}
# # # # # # # # # #     return out


# # # # # # # # # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # # # # # # # # #                             collate_fn, num_workers):
# # # # # # # # # #     n   = len(val_dataset)
# # # # # # # # # #     rng = random.Random(42)
# # # # # # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # # # # # #     return DataLoader(Subset(val_dataset, idx),
# # # # # # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # # # # # def _composite_score(result):
# # # # # # # # # #     """
# # # # # # # # # #     FIX: Composite score ưu tiên long-range.
# # # # # # # # # #     Tránh save model tốt về 12h nhưng tệ về 48h/72h.
# # # # # # # # # #     """
# # # # # # # # # #     ade = result.get("ADE", float("inf"))
# # # # # # # # # #     h48 = result.get("48h", float("inf"))
# # # # # # # # # #     h72 = result.get("72h", float("inf"))
# # # # # # # # # #     return 0.25 * ade + 0.35 * h48 + 0.40 * h72


# # # # # # # # # # # ── Evaluation ────────────────────────────────────────────────────────────────

# # # # # # # # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=15):
# # # # # # # # # #     """
# # # # # # # # # #     FIX: Tăng fast_ensemble 8→15, ode_steps 20→25 → estimate ổn định hơn.
# # # # # # # # # #     """
# # # # # # # # # #     model.eval()
# # # # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # # # #     t0  = time.perf_counter()
# # # # # # # # # #     n   = 0
# # # # # # # # # #     spread_per_step = []

# # # # # # # # # #     with torch.no_grad():
# # # # # # # # # #         for batch in loader:
# # # # # # # # # #             bl = move(list(batch), device)
# # # # # # # # # #             pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
# # # # # # # # # #                                               ddim_steps=ode_steps)
# # # # # # # # # #             T_active  = pred.shape[0]
# # # # # # # # # #             gt_sliced = bl[1][:T_active]
# # # # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
# # # # # # # # # #             acc.update(dist)

# # # # # # # # # #             step_spreads = []
# # # # # # # # # #             for t in range(all_trajs.shape[1]):
# # # # # # # # # #                 step_data = all_trajs[:, t, :, :]
# # # # # # # # # #                 std_lon = step_data[:, :, 0].std(0)
# # # # # # # # # #                 std_lat = step_data[:, :, 1].std(0)
# # # # # # # # # #                 spread = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# # # # # # # # # #                 step_spreads.append(spread)
# # # # # # # # # #             spread_per_step.append(step_spreads)
# # # # # # # # # #             n += 1

# # # # # # # # # #     r = acc.compute()
# # # # # # # # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)

# # # # # # # # # #     if spread_per_step:
# # # # # # # # # #         spreads = np.array(spread_per_step)
# # # # # # # # # #         r["spread_12h_km"] = float(spreads[:, 1].mean()) if spreads.shape[1] > 1 else 0.0
# # # # # # # # # #         r["spread_24h_km"] = float(spreads[:, 3].mean()) if spreads.shape[1] > 3 else 0.0
# # # # # # # # # #         r["spread_72h_km"] = float(spreads[:, -1].mean())
# # # # # # # # # #     return r


# # # # # # # # # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # # # # # # # # #                            fast_ensemble, metrics_csv, epoch):
# # # # # # # # # #     """
# # # # # # # # # #     FIX: Tăng fast_ensemble 8→20, ddim_steps 20→30 → eval chính xác hơn.
# # # # # # # # # #     """
# # # # # # # # # #     model.eval()
# # # # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # # # #     t0  = time.perf_counter()
# # # # # # # # # #     n   = 0

# # # # # # # # # #     with torch.no_grad():
# # # # # # # # # #         for batch in val_loader:
# # # # # # # # # #             bl = move(list(batch), device)
# # # # # # # # # #             # FIX: Tăng ensemble và steps cho full val
# # # # # # # # # #             pred, _, _ = model.sample(bl, num_ensemble=max(fast_ensemble, 20),
# # # # # # # # # #                                        ddim_steps=max(ode_steps, 30))
# # # # # # # # # #             T_pred = pred.shape[0]
# # # # # # # # # #             gt = bl[1][:T_pred]
# # # # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt))
# # # # # # # # # #             acc.update(dist)
# # # # # # # # # #             n += 1

# # # # # # # # # #     r = acc.compute()
# # # # # # # # # #     elapsed = time.perf_counter() - t0
# # # # # # # # # #     score = _composite_score(r)

# # # # # # # # # #     print(f"\n{'='*64}")
# # # # # # # # # #     print(f"  [FULL VAL ADE  ep={epoch}  {elapsed:.0f}s  {n} batches]")
# # # # # # # # # #     print(f"  ADE = {r.get('ADE', float('nan')):.1f} km  "
# # # # # # # # # #           f"FDE = {r.get('FDE', float('nan')):.1f} km")
# # # # # # # # # #     print(f"  6h={r.get('6h', float('nan')):.0f}  "
# # # # # # # # # #           f"12h={r.get('12h', float('nan')):.0f}  "
# # # # # # # # # #           f"24h={r.get('24h', float('nan')):.0f}  "
# # # # # # # # # #           f"48h={r.get('48h', float('nan')):.0f}  "
# # # # # # # # # #           f"72h={r.get('72h', float('nan')):.0f} km")
# # # # # # # # # #     # FIX: Log composite score
# # # # # # # # # #     print(f"  Composite score = {score:.1f} (0.25*ADE + 0.35*48h + 0.40*72h)")
# # # # # # # # # #     print(f"{'='*64}\n")

# # # # # # # # # #     from datetime import datetime
# # # # # # # # # #     from utils.metrics import DatasetMetrics, save_metrics_csv as _save_csv
# # # # # # # # # #     dm = DatasetMetrics(
# # # # # # # # # #         ade      = r.get("ADE",  float("nan")),
# # # # # # # # # #         fde      = r.get("FDE",  float("nan")),
# # # # # # # # # #         ugde_12h = r.get("12h",  float("nan")),
# # # # # # # # # #         ugde_24h = r.get("24h",  float("nan")),
# # # # # # # # # #         ugde_48h = r.get("48h",  float("nan")),
# # # # # # # # # #         ugde_72h = r.get("72h",  float("nan")),
# # # # # # # # # #         n_total  = r.get("n_samples", 0),
# # # # # # # # # #         timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # # # # # #     )
# # # # # # # # # #     _save_csv(dm, metrics_csv, tag=f"val_full_ep{epoch:03d}")
# # # # # # # # # #     return r


# # # # # # # # # # # ── BestModelSaver ────────────────────────────────────────────────────────────

# # # # # # # # # # class BestModelSaver:
# # # # # # # # # #     """
# # # # # # # # # #     FIX: Dùng composite score thay vì pure ADE.
# # # # # # # # # #     Composite = 0.25*ADE + 0.35*48h + 0.40*72h
# # # # # # # # # #     → Không còn chọn model tốt về 12h nhưng tệ về 72h.
# # # # # # # # # #     """
# # # # # # # # # #     def __init__(self, patience: int = 25, score_tol: float = 2.0):
# # # # # # # # # #         self.patience    = patience
# # # # # # # # # #         self.score_tol   = score_tol
# # # # # # # # # #         self.best_score  = float("inf")
# # # # # # # # # #         self.best_ade    = float("inf")
# # # # # # # # # #         self.best_72h    = float("inf")
# # # # # # # # # #         self.best_48h    = float("inf")
# # # # # # # # # #         self.counter     = 0
# # # # # # # # # #         self.early_stop  = False

# # # # # # # # # #     def update(self, result, model, out_dir, epoch, optimizer, tl, vl,
# # # # # # # # # #                min_epochs=60):
# # # # # # # # # #         ade   = result.get("ADE", float("inf"))
# # # # # # # # # #         h48   = result.get("48h", float("inf"))
# # # # # # # # # #         h72   = result.get("72h", float("inf"))
# # # # # # # # # #         score = _composite_score(result)

# # # # # # # # # #         if score < self.best_score - self.score_tol:
# # # # # # # # # #             self.best_score = score
# # # # # # # # # #             self.best_ade   = ade
# # # # # # # # # #             self.best_72h   = h72
# # # # # # # # # #             self.best_48h   = h48
# # # # # # # # # #             self.counter    = 0
# # # # # # # # # #             torch.save(dict(
# # # # # # # # # #                 epoch=epoch,
# # # # # # # # # #                 model_state_dict=model.state_dict(),
# # # # # # # # # #                 optimizer_state=optimizer.state_dict(),
# # # # # # # # # #                 train_loss=tl, val_loss=vl,
# # # # # # # # # #                 ade=ade, ade_48h=h48, ade_72h=h72,
# # # # # # # # # #                 composite_score=score,
# # # # # # # # # #                 model_version="v33fix-long-range"),
# # # # # # # # # #                 os.path.join(out_dir, "best_model.pth"))
# # # # # # # # # #             print(f"  ✅ Best score={score:.1f}  ADE={ade:.1f}  48h={h48:.1f}  72h={h72:.1f}  (epoch {epoch})")
# # # # # # # # # #         else:
# # # # # # # # # #             self.counter += 1
# # # # # # # # # #             print(f"  No improvement {self.counter}/{self.patience}"
# # # # # # # # # #                   f"  (best_score={self.best_score:.1f}  current={score:.1f})"
# # # # # # # # # #                   f"  [72h: best={self.best_72h:.1f} current={h72:.1f}]")

# # # # # # # # # #         if epoch >= min_epochs and self.counter >= self.patience:
# # # # # # # # # #             self.early_stop = True


# # # # # # # # # # # ── Env diagnostic ────────────────────────────────────────────────────────────

# # # # # # # # # # def _check_env(bl, train_dataset):
# # # # # # # # # #     try:
# # # # # # # # # #         env_dir = train_dataset.env_path
# # # # # # # # # #     except AttributeError:
# # # # # # # # # #         try:
# # # # # # # # # #             env_dir = train_dataset.dataset.env_path
# # # # # # # # # #         except AttributeError:
# # # # # # # # # #             env_dir = "UNKNOWN"
# # # # # # # # # #     print(f"  Env path: {env_dir}")

# # # # # # # # # #     env_data = bl[13]
# # # # # # # # # #     if env_data is None:
# # # # # # # # # #         print("  ⚠️  env_data is None"); return

# # # # # # # # # #     for key in ("gph500_mean", "gph500_center", "u500_mean", "v500_mean"):
# # # # # # # # # #         if key not in env_data:
# # # # # # # # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # # # # # # # #         v    = env_data[key]
# # # # # # # # # #         mn   = v.mean().item()
# # # # # # # # # #         std  = v.std().item()
# # # # # # # # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # # # # # # # #         print(f"  {'✅' if zero < 80 else '⚠️'}  {key}: mean={mn:.4f} std={std:.4f} zero={zero:.1f}%")


# # # # # # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # # # # # def get_args():
# # # # # # # # # #     p = argparse.ArgumentParser(
# # # # # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # # # # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # # # # # # # # #     p.add_argument("--obs_len",         default=8,          type=int)
# # # # # # # # # #     p.add_argument("--pred_len",        default=12,         type=int)
# # # # # # # # # #     p.add_argument("--batch_size",      default=32,         type=int)
# # # # # # # # # #     p.add_argument("--num_epochs",      default=120,        type=int)
# # # # # # # # # #     p.add_argument("--g_learning_rate", default=3e-4,       type=float)
# # # # # # # # # #     p.add_argument("--weight_decay",    default=1e-4,       type=float)
# # # # # # # # # #     p.add_argument("--warmup_epochs",   default=5,          type=int)
# # # # # # # # # #     p.add_argument("--grad_clip",       default=1.0,        type=float)
# # # # # # # # # #     p.add_argument("--patience",        default=25,         type=int)
# # # # # # # # # #     p.add_argument("--min_epochs",      default=50,         type=int)
# # # # # # # # # #     p.add_argument("--n_train_ens",     default=4,          type=int)
# # # # # # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # # # # # #     p.add_argument("--num_workers",     default=2,          type=int)

# # # # # # # # # #     p.add_argument("--sigma_min",            default=0.02,   type=float)
# # # # # # # # # #     p.add_argument("--ctx_noise_scale",      default=0.005,  type=float)
# # # # # # # # # #     p.add_argument("--initial_sample_sigma", default=0.03,   type=float)

# # # # # # # # # #     p.add_argument("--ode_steps_train", default=25,  type=int)   # FIX: 20→25
# # # # # # # # # #     p.add_argument("--ode_steps_val",   default=30,  type=int)
# # # # # # # # # #     p.add_argument("--ode_steps_test",  default=50,  type=int)

# # # # # # # # # #     p.add_argument("--val_ensemble",    default=30,  type=int)
# # # # # # # # # #     p.add_argument("--fast_ensemble",   default=15,  type=int)   # FIX: 8→15

# # # # # # # # # #     p.add_argument("--val_freq",         default=1,   type=int)
# # # # # # # # # #     p.add_argument("--val_ade_freq",     default=1,   type=int)
# # # # # # # # # #     p.add_argument("--val_subset_size",  default=600, type=int)

# # # # # # # # # #     p.add_argument("--output_dir",      default="runs/v33fix",   type=str)
# # # # # # # # # #     p.add_argument("--metrics_csv",     default="metrics.csv",   type=str)
# # # # # # # # # #     p.add_argument("--predict_csv",     default="predictions.csv", type=str)

# # # # # # # # # #     p.add_argument("--gpu_num",         default="0", type=str)
# # # # # # # # # #     p.add_argument("--delim",           default=" ")
# # # # # # # # # #     p.add_argument("--skip",            default=1,   type=int)
# # # # # # # # # #     p.add_argument("--min_ped",         default=1,   type=int)
# # # # # # # # # #     p.add_argument("--threshold",       default=0.002, type=float)
# # # # # # # # # #     p.add_argument("--other_modal",     default="gph")
# # # # # # # # # #     p.add_argument("--test_year",       default=None, type=int)

# # # # # # # # # #     # FIX: Option để load từ checkpoint v33 cũ và fine-tune
# # # # # # # # # #     p.add_argument("--resume",          default=None, type=str,
# # # # # # # # # #                    help="Path to checkpoint to resume from (e.g. runs/v33/best_model.pth)")
# # # # # # # # # #     p.add_argument("--resume_epoch",    default=0,    type=int,
# # # # # # # # # #                    help="Start epoch khi resume (để sigma schedule đúng)")

# # # # # # # # # #     return p.parse_args()


# # # # # # # # # # # ── MAIN ──────────────────────────────────────────────────────────────────────

# # # # # # # # # # def main(args):
# # # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # # # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)

# # # # # # # # # #     print("=" * 70)
# # # # # # # # # #     print("  TC-FlowMatching v33fix  |  LONG-RANGE FIXED")
# # # # # # # # # #     print("  ─────────────────────────────────────────────")
# # # # # # # # # #     print("  FIX 1: BỎ no_grad → l_vel/l_head có gradient thực")
# # # # # # # # # #     print("  FIX 2: long_range_aux_loss cho 48h-72h (bật sau epoch 20)")
# # # # # # # # # #     print("  FIX 3: Median ensemble thay mean tại inference")
# # # # # # # # # #     print("  FIX 4: Composite saver = 0.25*ADE + 0.35*48h + 0.40*72h")
# # # # # # # # # #     print("  FIX 5: fast_ensemble=15, ode_steps=25 (stable val estimate)")
# # # # # # # # # #     print("  TARGETS: 12h<50  24h<100  48h<200  72h<300 km")
# # # # # # # # # #     print("=" * 70)

# # # # # # # # # #     # ── Data ──────────────────────────────────────────────────────────────
# # # # # # # # # #     train_dataset, train_loader = data_loader(
# # # # # # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # # # # # #     val_dataset, val_loader = data_loader(
# # # # # # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # # # #     val_subset_loader = make_val_subset_loader(
# # # # # # # # # #         val_dataset, args.val_subset_size, args.batch_size,
# # # # # # # # # #         seq_collate, args.num_workers)

# # # # # # # # # #     test_loader = None
# # # # # # # # # #     try:
# # # # # # # # # #         _, test_loader = data_loader(
# # # # # # # # # #             args, {"root": args.dataset_root, "type": "test"},
# # # # # # # # # #             test=True, test_year=None)
# # # # # # # # # #     except Exception as e:
# # # # # # # # # #         print(f"  Warning: test loader: {e}")

# # # # # # # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # # # # # # #     print(f"  val   : {len(val_dataset)} seq")

# # # # # # # # # #     # ── Model ─────────────────────────────────────────────────────────────
# # # # # # # # # #     model = TCFlowMatching(
# # # # # # # # # #         pred_len             = args.pred_len,
# # # # # # # # # #         obs_len              = args.obs_len,
# # # # # # # # # #         sigma_min            = args.sigma_min,
# # # # # # # # # #         n_train_ens          = args.n_train_ens,
# # # # # # # # # #         ctx_noise_scale      = args.ctx_noise_scale,
# # # # # # # # # #         initial_sample_sigma = args.initial_sample_sigma,
# # # # # # # # # #     ).to(device)

# # # # # # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # # # # #     print(f"  params  : {n_params:,}")

# # # # # # # # # #     # ── Optimizer ─────────────────────────────────────────────────────────
# # # # # # # # # #     optimizer = optim.AdamW(
# # # # # # # # # #         model.parameters(),
# # # # # # # # # #         lr=args.g_learning_rate,
# # # # # # # # # #         weight_decay=args.weight_decay,
# # # # # # # # # #     )

# # # # # # # # # #     # ── Resume từ checkpoint ───────────────────────────────────────────────
# # # # # # # # # #     start_epoch = 0
# # # # # # # # # #     if args.resume is not None and os.path.exists(args.resume):
# # # # # # # # # #         print(f"  Loading checkpoint: {args.resume}")
# # # # # # # # # #         ckpt = torch.load(args.resume, map_location=device)
# # # # # # # # # #         model.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # # # # # # #         # Không load optimizer state khi resume để fresh lr schedule
# # # # # # # # # #         start_epoch = args.resume_epoch
# # # # # # # # # #         print(f"  Resumed from epoch {start_epoch}")
# # # # # # # # # #     elif args.resume is not None:
# # # # # # # # # #         print(f"  ⚠️  Checkpoint không tìm thấy: {args.resume}, training từ đầu")

# # # # # # # # # #     try:
# # # # # # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # # # # # #         print("  torch.compile: enabled")
# # # # # # # # # #     except Exception:
# # # # # # # # # #         pass

# # # # # # # # # #     steps_per_epoch = len(train_loader)
# # # # # # # # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # # # # # # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # # # # # # # #     scheduler = get_cosine_schedule_with_warmup(
# # # # # # # # # #         optimizer, warmup, total_steps, min_lr=1e-6)

# # # # # # # # # #     # Advance scheduler nếu resume
# # # # # # # # # #     if start_epoch > 0:
# # # # # # # # # #         for _ in range(start_epoch * steps_per_epoch):
# # # # # # # # # #             scheduler.step()

# # # # # # # # # #     saver  = BestModelSaver(patience=args.patience, score_tol=2.0)
# # # # # # # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)

# # # # # # # # # #     print("=" * 70)
# # # # # # # # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch, start_epoch={start_epoch})")
# # # # # # # # # #     print("=" * 70)

# # # # # # # # # #     epoch_times = []
# # # # # # # # # #     train_start = time.perf_counter()

# # # # # # # # # #     for epoch in range(start_epoch, args.num_epochs):
# # # # # # # # # #         model.train()
# # # # # # # # # #         sum_loss = 0.0
# # # # # # # # # #         t0 = time.perf_counter()

# # # # # # # # # #         for i, batch in enumerate(train_loader):
# # # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # # #             if epoch == start_epoch and i == 0:
# # # # # # # # # #                 _check_env(bl, train_dataset)

# # # # # # # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # # #                 bd = model.get_loss_breakdown(bl, epoch=epoch)

# # # # # # # # # #             optimizer.zero_grad()
# # # # # # # # # #             scaler.scale(bd["total"]).backward()
# # # # # # # # # #             scaler.unscale_(optimizer)
# # # # # # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # # # # # # #             scaler.step(optimizer)
# # # # # # # # # #             scaler.update()
# # # # # # # # # #             scheduler.step()

# # # # # # # # # #             sum_loss += bd["total"].item()

# # # # # # # # # #             if i % 20 == 0:
# # # # # # # # # #                 lr = optimizer.param_groups[0]["lr"]
# # # # # # # # # #                 elapsed = time.perf_counter() - t0
# # # # # # # # # #                 # FIX: Log thêm long_range loss và weight
# # # # # # # # # #                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # # # # # # # #                       f"  loss={bd['total'].item():.4f}"
# # # # # # # # # #                       f"  mse={bd['mse_hav']:.4f}"
# # # # # # # # # #                       f"  vel={bd['velocity']:.4f}"
# # # # # # # # # #                       f"  head={bd['heading']:.4f}"
# # # # # # # # # #                       f"  lr48-72={bd['long_range']:.4f}(w={bd['w_lr']:.2f})"
# # # # # # # # # #                       f"  σ={bd['sigma']:.3f}"
# # # # # # # # # #                       f"  lr={lr:.2e}  t={elapsed:.0f}s")

# # # # # # # # # #         ep_s  = time.perf_counter() - t0
# # # # # # # # # #         epoch_times.append(ep_s)
# # # # # # # # # #         avg_t = sum_loss / len(train_loader)

# # # # # # # # # #         # ── Val loss ──────────────────────────────────────────────────────
# # # # # # # # # #         model.eval()
# # # # # # # # # #         val_loss = 0.0
# # # # # # # # # #         with torch.no_grad():
# # # # # # # # # #             for batch in val_loader:
# # # # # # # # # #                 bl_v = move(list(batch), device)
# # # # # # # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # # #                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
# # # # # # # # # #         avg_vl = val_loss / len(val_loader)

# # # # # # # # # #         print(f"  Epoch {epoch:>3}  train={avg_t:.4f}  val={avg_vl:.4f}"
# # # # # # # # # #               f"  time={ep_s:.0f}s")

# # # # # # # # # #         # ── Fast eval ─────────────────────────────────────────────────────
# # # # # # # # # #         m_fast = evaluate_fast(model, val_subset_loader, device,
# # # # # # # # # #                                 args.ode_steps_train, args.pred_len,
# # # # # # # # # #                                 args.fast_ensemble)

# # # # # # # # # #         h6  = m_fast.get("6h",  float("nan"))
# # # # # # # # # #         h12 = m_fast.get("12h", float("nan"))
# # # # # # # # # #         h24 = m_fast.get("24h", float("nan"))
# # # # # # # # # #         h48 = m_fast.get("48h", float("nan"))
# # # # # # # # # #         h72 = m_fast.get("72h", float("nan"))
# # # # # # # # # #         sp72 = m_fast.get("spread_72h_km", 0.0)
# # # # # # # # # #         fast_score = _composite_score(m_fast)

# # # # # # # # # #         t6  = "🎯" if h6  < 30  else "❌"
# # # # # # # # # #         t12 = "🎯" if h12 < 50  else "❌"
# # # # # # # # # #         t24 = "🎯" if h24 < 100 else "❌"
# # # # # # # # # #         t48 = "🎯" if h48 < 200 else "❌"
# # # # # # # # # #         t72 = "🎯" if h72 < 300 else "❌"

# # # # # # # # # #         print(f"  [FAST ep{epoch}]"
# # # # # # # # # #               f"  ADE={m_fast['ADE']:.1f}"
# # # # # # # # # #               f"  6h={h6:.0f}{t6}"
# # # # # # # # # #               f"  12h={h12:.0f}{t12}"
# # # # # # # # # #               f"  24h={h24:.0f}{t24}"
# # # # # # # # # #               f"  48h={h48:.0f}{t48}"
# # # # # # # # # #               f"  72h={h72:.0f}{t72}"
# # # # # # # # # #               f"  spread72={sp72:.0f}km"
# # # # # # # # # #               f"  score={fast_score:.1f}")

# # # # # # # # # #         # ── Full val ADE ──────────────────────────────────────────────────
# # # # # # # # # #         if epoch % args.val_ade_freq == 0:
# # # # # # # # # #             try:
# # # # # # # # # #                 r_full = evaluate_full_val_ade(
# # # # # # # # # #                     model, val_loader, device,
# # # # # # # # # #                     ode_steps=args.ode_steps_train, pred_len=args.pred_len,
# # # # # # # # # #                     fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # # # # # # #                     epoch=epoch)
# # # # # # # # # #                 saver.update(r_full, model, args.output_dir, epoch,
# # # # # # # # # #                            optimizer, avg_t, avg_vl,
# # # # # # # # # #                            min_epochs=args.min_epochs)
# # # # # # # # # #             except Exception as e:
# # # # # # # # # #                 print(f"  ⚠  Full val failed: {e}")
# # # # # # # # # #                 import traceback; traceback.print_exc()

# # # # # # # # # #         # ── Checkpoint ────────────────────────────────────────────────────
# # # # # # # # # #         if epoch % 5 == 0 or epoch == args.num_epochs - 1:
# # # # # # # # # #             torch.save({
# # # # # # # # # #                 "epoch": epoch,
# # # # # # # # # #                 "model_state_dict": model.state_dict(),
# # # # # # # # # #                 "optimizer_state": optimizer.state_dict(),
# # # # # # # # # #                 "train_loss": avg_t, "val_loss": avg_vl,
# # # # # # # # # #                 "ade": m_fast.get("ADE", float("nan")),
# # # # # # # # # #                 "h48": h48, "h72": h72,
# # # # # # # # # #             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))

# # # # # # # # # #         if epoch % 10 == 9:
# # # # # # # # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # # # # # # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # # # # # # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # # # # # # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")

# # # # # # # # # #         if saver.early_stop:
# # # # # # # # # #             print(f"  ⛔ Early stopping at epoch {epoch}")
# # # # # # # # # #             break

# # # # # # # # # #     # ── Final ─────────────────────────────────────────────────────────────
# # # # # # # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # # # # # # #     print(f"\n{'='*70}")
# # # # # # # # # #     print(f"  Best composite score : {saver.best_score:.1f}")
# # # # # # # # # #     print(f"  Best ADE  : {saver.best_ade:.1f} km")
# # # # # # # # # #     print(f"  Best 48h  : {saver.best_48h:.1f} km")
# # # # # # # # # #     print(f"  Best 72h  : {saver.best_72h:.1f} km")
# # # # # # # # # #     print(f"  Training  : {total_h:.2f}h")
# # # # # # # # # #     print("=" * 70)


# # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # #     args = get_args()
# # # # # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # # #         torch.cuda.manual_seed_all(42)
# # # # # # # # # #     main(args)

# # # # # # # # # """
# # # # # # # # # train_flowmatching_v34.py — Training cho TC-FlowMatching v34
# # # # # # # # # ═══════════════════════════════════════════════════════════════════

# # # # # # # # # CẢI TIẾN TỪ v33fix:
# # # # # # # # #   1. EMA weights (exponential moving average)
# # # # # # # # #   2. SWA (Stochastic Weight Averaging) ở late epochs
# # # # # # # # #   3. Horizon-specific tracking: log riêng best_12h, best_24h, best_48h, best_72h
# # # # # # # # #   4. Composite score đa trọng số cho early stopping
# # # # # # # # #   5. Evaluate với EMA weights
# # # # # # # # #   6. Phase-aware: weights và sigma schedule rõ ràng

# # # # # # # # # COMPOSITE SCORE:
# # # # # # # # #   score = 0.20*ADE + 0.20*12h + 0.20*24h + 0.20*48h + 0.20*72h
# # # # # # # # #   → cân bằng hơn so với v33fix (0.25+0.35+0.40 thiên về long-range)
# # # # # # # # # """
# # # # # # # # # from __future__ import annotations

# # # # # # # # # import sys
# # # # # # # # # import os
# # # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # # import argparse
# # # # # # # # # import time
# # # # # # # # # import math
# # # # # # # # # import random
# # # # # # # # # import copy

# # # # # # # # # import numpy as np
# # # # # # # # # import torch
# # # # # # # # # import torch.optim as optim
# # # # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # # from Model.data.loader_training import data_loader
# # # # # # # # # from Model.flow_matching_model import TCFlowMatching
# # # # # # # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # # # from utils.metrics import (
# # # # # # # # #     TCEvaluator, StepErrorAccumulator,
# # # # # # # # #     save_metrics_csv, haversine_km_torch,
# # # # # # # # #     denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
# # # # # # # # # )


# # # # # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # # # # def move(batch, device):
# # # # # # # # #     out = list(batch)
# # # # # # # # #     for i, x in enumerate(out):
# # # # # # # # #         if torch.is_tensor(x):
# # # # # # # # #             out[i] = x.to(device)
# # # # # # # # #         elif isinstance(x, dict):
# # # # # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # # #                       for k, v in x.items()}
# # # # # # # # #     return out


# # # # # # # # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # # # # # # # #                             collate_fn, num_workers):
# # # # # # # # #     n   = len(val_dataset)
# # # # # # # # #     rng = random.Random(42)
# # # # # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # # # # #     return DataLoader(Subset(val_dataset, idx),
# # # # # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # # # # def _composite_score(result):
# # # # # # # # #     """
# # # # # # # # #     Balanced composite: equal weight across all horizons + ADE.
# # # # # # # # #     """
# # # # # # # # #     ade = result.get("ADE", float("inf"))
# # # # # # # # #     h12 = result.get("12h", float("inf"))
# # # # # # # # #     h24 = result.get("24h", float("inf"))
# # # # # # # # #     h48 = result.get("48h", float("inf"))
# # # # # # # # #     h72 = result.get("72h", float("inf"))
# # # # # # # # #     return 0.20 * ade + 0.20 * h12 + 0.20 * h24 + 0.20 * h48 + 0.20 * h72


# # # # # # # # # # ── SWA (Stochastic Weight Averaging) ─────────────────────────────────────────

# # # # # # # # # class SWAManager:
# # # # # # # # #     """Simple SWA: average weights of last N checkpoints."""
# # # # # # # # #     def __init__(self, model, start_epoch=50):
# # # # # # # # #         self.start_epoch = start_epoch
# # # # # # # # #         self.n_averaged  = 0
# # # # # # # # #         self.avg_state   = None

# # # # # # # # #     def update(self, model, epoch):
# # # # # # # # #         if epoch < self.start_epoch:
# # # # # # # # #             return
# # # # # # # # #         sd = {k: v.detach().clone() for k, v in model.state_dict().items()
# # # # # # # # #               if v.dtype.is_floating_point}
# # # # # # # # #         if self.avg_state is None:
# # # # # # # # #             self.avg_state = sd
# # # # # # # # #             self.n_averaged = 1
# # # # # # # # #         else:
# # # # # # # # #             n = self.n_averaged
# # # # # # # # #             for k in self.avg_state:
# # # # # # # # #                 if k in sd:
# # # # # # # # #                     self.avg_state[k].mul_(n / (n + 1)).add_(sd[k], alpha=1.0 / (n + 1))
# # # # # # # # #             self.n_averaged += 1

# # # # # # # # #     def apply_to(self, model):
# # # # # # # # #         if self.avg_state is None:
# # # # # # # # #             return None
# # # # # # # # #         backup = {}
# # # # # # # # #         sd = model.state_dict()
# # # # # # # # #         for k, v in self.avg_state.items():
# # # # # # # # #             backup[k] = sd[k].detach().clone()
# # # # # # # # #             sd[k].copy_(v)
# # # # # # # # #         return backup

# # # # # # # # #     def restore(self, model, backup):
# # # # # # # # #         if backup is None:
# # # # # # # # #             return
# # # # # # # # #         sd = model.state_dict()
# # # # # # # # #         for k, v in backup.items():
# # # # # # # # #             sd[k].copy_(v)


# # # # # # # # # # ── Evaluation ────────────────────────────────────────────────────────────────

# # # # # # # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=15):
# # # # # # # # #     model.eval()
# # # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # # #     t0  = time.perf_counter()
# # # # # # # # #     n   = 0
# # # # # # # # #     spread_per_step = []

# # # # # # # # #     with torch.no_grad():
# # # # # # # # #         for batch in loader:
# # # # # # # # #             bl = move(list(batch), device)
# # # # # # # # #             pred, _, all_trajs = model.sample(
# # # # # # # # #                 bl, num_ensemble=fast_ensemble, ddim_steps=ode_steps,
# # # # # # # # #                 importance_weight=True)
# # # # # # # # #             T_active  = pred.shape[0]
# # # # # # # # #             gt_sliced = bl[1][:T_active]
# # # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
# # # # # # # # #             acc.update(dist)

# # # # # # # # #             step_spreads = []
# # # # # # # # #             for t in range(all_trajs.shape[1]):
# # # # # # # # #                 step_data = all_trajs[:, t, :, :]
# # # # # # # # #                 std_lon = step_data[:, :, 0].std(0)
# # # # # # # # #                 std_lat = step_data[:, :, 1].std(0)
# # # # # # # # #                 spread = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# # # # # # # # #                 step_spreads.append(spread)
# # # # # # # # #             spread_per_step.append(step_spreads)
# # # # # # # # #             n += 1

# # # # # # # # #     r = acc.compute()
# # # # # # # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# # # # # # # # #     if spread_per_step:
# # # # # # # # #         spreads = np.array(spread_per_step)
# # # # # # # # #         r["spread_12h_km"] = float(spreads[:, 1].mean()) if spreads.shape[1] > 1 else 0.0
# # # # # # # # #         r["spread_24h_km"] = float(spreads[:, 3].mean()) if spreads.shape[1] > 3 else 0.0
# # # # # # # # #         r["spread_72h_km"] = float(spreads[:, -1].mean())
# # # # # # # # #     return r


# # # # # # # # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # # # # # # # #                            fast_ensemble, metrics_csv, epoch,
# # # # # # # # #                            use_ema=False, ema_obj=None):
# # # # # # # # #     """Full val eval, optionally with EMA weights."""
# # # # # # # # #     backup = None
# # # # # # # # #     if use_ema and ema_obj is not None:
# # # # # # # # #         backup = ema_obj.apply_to(model)

# # # # # # # # #     model.eval()
# # # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # # #     t0  = time.perf_counter()
# # # # # # # # #     n   = 0

# # # # # # # # #     with torch.no_grad():
# # # # # # # # #         for batch in val_loader:
# # # # # # # # #             bl = move(list(batch), device)
# # # # # # # # #             pred, _, _ = model.sample(
# # # # # # # # #                 bl, num_ensemble=max(fast_ensemble, 20),
# # # # # # # # #                 ddim_steps=max(ode_steps, 30),
# # # # # # # # #                 importance_weight=True)
# # # # # # # # #             T_pred = pred.shape[0]
# # # # # # # # #             gt = bl[1][:T_pred]
# # # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt))
# # # # # # # # #             acc.update(dist)
# # # # # # # # #             n += 1

# # # # # # # # #     r = acc.compute()
# # # # # # # # #     elapsed = time.perf_counter() - t0
# # # # # # # # #     score = _composite_score(r)

# # # # # # # # #     tag = "EMA" if use_ema else "RAW"
# # # # # # # # #     print(f"\n{'='*64}")
# # # # # # # # #     print(f"  [FULL VAL ADE ({tag})  ep={epoch}  {elapsed:.0f}s  {n} batches]")
# # # # # # # # #     print(f"  ADE = {r.get('ADE', float('nan')):.1f} km  "
# # # # # # # # #           f"FDE = {r.get('FDE', float('nan')):.1f} km")
# # # # # # # # #     print(f"  6h={r.get('6h', float('nan')):.0f}  "
# # # # # # # # #           f"12h={r.get('12h', float('nan')):.0f}  "
# # # # # # # # #           f"24h={r.get('24h', float('nan')):.0f}  "
# # # # # # # # #           f"48h={r.get('48h', float('nan')):.0f}  "
# # # # # # # # #           f"72h={r.get('72h', float('nan')):.0f} km")
# # # # # # # # #     print(f"  Composite score = {score:.1f}")
# # # # # # # # #     print(f"{'='*64}\n")

# # # # # # # # #     from datetime import datetime
# # # # # # # # #     from utils.metrics import DatasetMetrics, save_metrics_csv as _save_csv
# # # # # # # # #     dm = DatasetMetrics(
# # # # # # # # #         ade      = r.get("ADE",  float("nan")),
# # # # # # # # #         fde      = r.get("FDE",  float("nan")),
# # # # # # # # #         ugde_12h = r.get("12h",  float("nan")),
# # # # # # # # #         ugde_24h = r.get("24h",  float("nan")),
# # # # # # # # #         ugde_48h = r.get("48h",  float("nan")),
# # # # # # # # #         ugde_72h = r.get("72h",  float("nan")),
# # # # # # # # #         n_total  = r.get("n_samples", 0),
# # # # # # # # #         timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # # # # #     )
# # # # # # # # #     _save_csv(dm, metrics_csv, tag=f"val_full_{tag}_ep{epoch:03d}")

# # # # # # # # #     # Restore
# # # # # # # # #     if backup is not None:
# # # # # # # # #         ema_obj.restore(model, backup)
# # # # # # # # #     return r


# # # # # # # # # # ── Horizon-specific BestModelSaver ───────────────────────────────────────────

# # # # # # # # # class HorizonAwareBestSaver:
# # # # # # # # #     """
# # # # # # # # #     Track best for each horizon separately + composite.
# # # # # # # # #     Save 3 checkpoints: best_composite, best_ade, best_72h.
# # # # # # # # #     """
# # # # # # # # #     def __init__(self, patience=30, tol=1.5):
# # # # # # # # #         self.patience = patience
# # # # # # # # #         self.tol = tol
# # # # # # # # #         self.counter = 0
# # # # # # # # #         self.early_stop = False

# # # # # # # # #         self.best_score   = float("inf")
# # # # # # # # #         self.best_ade     = float("inf")
# # # # # # # # #         self.best_12h     = float("inf")
# # # # # # # # #         self.best_24h     = float("inf")
# # # # # # # # #         self.best_48h     = float("inf")
# # # # # # # # #         self.best_72h     = float("inf")

# # # # # # # # #     def update(self, r, model, out_dir, epoch, optimizer, tl, vl,
# # # # # # # # #                 min_epochs=50):
# # # # # # # # #         ade = r.get("ADE", float("inf"))
# # # # # # # # #         h12 = r.get("12h", float("inf"))
# # # # # # # # #         h24 = r.get("24h", float("inf"))
# # # # # # # # #         h48 = r.get("48h", float("inf"))
# # # # # # # # #         h72 = r.get("72h", float("inf"))
# # # # # # # # #         score = _composite_score(r)

# # # # # # # # #         improved_score = score < self.best_score - self.tol
# # # # # # # # #         improved_any = False

# # # # # # # # #         # Per-horizon best
# # # # # # # # #         if ade < self.best_ade:
# # # # # # # # #             self.best_ade = ade
# # # # # # # # #             improved_any = True
# # # # # # # # #             torch.save(dict(
# # # # # # # # #                 epoch=epoch, model_state_dict=model.state_dict(),
# # # # # # # # #                 ade=ade, h12=h12, h24=h24, h48=h48, h72=h72,
# # # # # # # # #                 tag="best_ade",
# # # # # # # # #             ), os.path.join(out_dir, "best_ade.pth"))
# # # # # # # # #         if h72 < self.best_72h:
# # # # # # # # #             self.best_72h = h72
# # # # # # # # #             improved_any = True
# # # # # # # # #             torch.save(dict(
# # # # # # # # #                 epoch=epoch, model_state_dict=model.state_dict(),
# # # # # # # # #                 ade=ade, h12=h12, h24=h24, h48=h48, h72=h72,
# # # # # # # # #                 tag="best_72h",
# # # # # # # # #             ), os.path.join(out_dir, "best_72h.pth"))
# # # # # # # # #         if h48 < self.best_48h: self.best_48h = h48; improved_any = True
# # # # # # # # #         if h24 < self.best_24h: self.best_24h = h24; improved_any = True
# # # # # # # # #         if h12 < self.best_12h: self.best_12h = h12; improved_any = True

# # # # # # # # #         if improved_score:
# # # # # # # # #             self.best_score = score
# # # # # # # # #             self.counter = 0
# # # # # # # # #             torch.save(dict(
# # # # # # # # #                 epoch=epoch, model_state_dict=model.state_dict(),
# # # # # # # # #                 optimizer_state=optimizer.state_dict(),
# # # # # # # # #                 train_loss=tl, val_loss=vl,
# # # # # # # # #                 ade=ade, h12=h12, h24=h24, h48=h48, h72=h72,
# # # # # # # # #                 composite_score=score, tag="best_composite",
# # # # # # # # #                 model_version="v34",
# # # # # # # # #             ), os.path.join(out_dir, "best_model.pth"))
# # # # # # # # #             print(f"  ✅ Best COMPOSITE={score:.1f}  "
# # # # # # # # #                   f"ADE={ade:.1f}  12h={h12:.0f}  24h={h24:.0f}  "
# # # # # # # # #                   f"48h={h48:.0f}  72h={h72:.0f}  (ep {epoch})")
# # # # # # # # #         else:
# # # # # # # # #             if not improved_any:
# # # # # # # # #                 self.counter += 1
# # # # # # # # #             print(f"  No improvement {self.counter}/{self.patience}"
# # # # # # # # #                   f"  (best_score={self.best_score:.1f}, cur={score:.1f})")

# # # # # # # # #         if epoch >= min_epochs and self.counter >= self.patience:
# # # # # # # # #             self.early_stop = True


# # # # # # # # # # ── Env diagnostic ────────────────────────────────────────────────────────────

# # # # # # # # # def _check_env(bl, train_dataset):
# # # # # # # # #     try:
# # # # # # # # #         env_dir = train_dataset.env_path
# # # # # # # # #     except AttributeError:
# # # # # # # # #         try:
# # # # # # # # #             env_dir = train_dataset.dataset.env_path
# # # # # # # # #         except AttributeError:
# # # # # # # # #             env_dir = "UNKNOWN"
# # # # # # # # #     print(f"  Env path: {env_dir}")

# # # # # # # # #     env_data = bl[13]
# # # # # # # # #     if env_data is None:
# # # # # # # # #         print("  ⚠️  env_data is None"); return

# # # # # # # # #     for key in ("gph500_mean", "gph500_center", "u500_mean", "v500_mean"):
# # # # # # # # #         if key not in env_data:
# # # # # # # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # # # # # # #         v    = env_data[key]
# # # # # # # # #         mn   = v.mean().item()
# # # # # # # # #         std  = v.std().item()
# # # # # # # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # # # # # # #         print(f"  {'✅' if zero < 80 else '⚠️'}  {key}: mean={mn:.4f} std={std:.4f} zero={zero:.1f}%")


# # # # # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # # # # def get_args():
# # # # # # # # #     p = argparse.ArgumentParser(
# # # # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # # # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # # # # # # # #     p.add_argument("--obs_len",         default=8,          type=int)
# # # # # # # # #     p.add_argument("--pred_len",        default=12,         type=int)
# # # # # # # # #     p.add_argument("--batch_size",      default=32,         type=int)
# # # # # # # # #     p.add_argument("--num_epochs",      default=120,        type=int)
# # # # # # # # #     p.add_argument("--g_learning_rate", default=3e-4,       type=float)
# # # # # # # # #     p.add_argument("--weight_decay",    default=1e-4,       type=float)
# # # # # # # # #     p.add_argument("--warmup_epochs",   default=5,          type=int)
# # # # # # # # #     p.add_argument("--grad_clip",       default=1.0,        type=float)
# # # # # # # # #     p.add_argument("--patience",        default=30,         type=int)
# # # # # # # # #     p.add_argument("--min_epochs",      default=50,         type=int)
# # # # # # # # #     p.add_argument("--n_train_ens",     default=4,          type=int)
# # # # # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # # # # #     p.add_argument("--num_workers",     default=2,          type=int)

# # # # # # # # #     p.add_argument("--sigma_min",            default=0.02,   type=float)
# # # # # # # # #     p.add_argument("--ctx_noise_scale",      default=0.01,   type=float)
# # # # # # # # #     p.add_argument("--initial_sample_sigma", default=0.03,   type=float)

# # # # # # # # #     p.add_argument("--ode_steps_train", default=25,  type=int)
# # # # # # # # #     p.add_argument("--ode_steps_val",   default=30,  type=int)
# # # # # # # # #     p.add_argument("--ode_steps_test",  default=50,  type=int)

# # # # # # # # #     p.add_argument("--val_ensemble",    default=30,  type=int)
# # # # # # # # #     p.add_argument("--fast_ensemble",   default=15,  type=int)

# # # # # # # # #     p.add_argument("--val_freq",         default=1,   type=int)
# # # # # # # # #     p.add_argument("--val_ade_freq",     default=1,   type=int)
# # # # # # # # #     p.add_argument("--val_subset_size",  default=600, type=int)

# # # # # # # # #     # ★ NEW options
# # # # # # # # #     p.add_argument("--use_ema",          action="store_true", default=True)
# # # # # # # # #     p.add_argument("--ema_decay",        default=0.999, type=float)
# # # # # # # # #     p.add_argument("--swa_start_epoch",  default=55,   type=int)
# # # # # # # # #     p.add_argument("--teacher_forcing",  action="store_true", default=True)

# # # # # # # # #     p.add_argument("--output_dir",      default="runs/v34",      type=str)
# # # # # # # # #     p.add_argument("--metrics_csv",     default="metrics.csv",   type=str)
# # # # # # # # #     p.add_argument("--predict_csv",     default="predictions.csv", type=str)

# # # # # # # # #     p.add_argument("--gpu_num",         default="0", type=str)
# # # # # # # # #     p.add_argument("--delim",           default=" ")
# # # # # # # # #     p.add_argument("--skip",            default=1,   type=int)
# # # # # # # # #     p.add_argument("--min_ped",         default=1,   type=int)
# # # # # # # # #     p.add_argument("--threshold",       default=0.002, type=float)
# # # # # # # # #     p.add_argument("--other_modal",     default="gph")
# # # # # # # # #     p.add_argument("--test_year",       default=None, type=int)

# # # # # # # # #     p.add_argument("--resume",          default=None, type=str)
# # # # # # # # #     p.add_argument("--resume_epoch",    default=0,    type=int)

# # # # # # # # #     return p.parse_args()


# # # # # # # # # # ── MAIN ──────────────────────────────────────────────────────────────────────

# # # # # # # # # def main(args):
# # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)

# # # # # # # # #     print("=" * 70)
# # # # # # # # #     print("  TC-FlowMatching v34  |  HORIZON-AWARE + MULTI-SCALE + EMA/SWA")
# # # # # # # # #     print("  ─────────────────────────────────────────────")
# # # # # # # # #     print("  FIXES:")
# # # # # # # # #     print("    1. No more NameError (w_vel/w_lr/w_head defined properly)")
# # # # # # # # #     print("    2. No double denorm (clean naming _deg vs _norm)")
# # # # # # # # #     print("    3. All losses use DEGREES consistently")
# # # # # # # # #     print("  NEW IDEAS:")
# # # # # # # # #     print("    1. Horizon-aware weighting: w[t] ~ (t+1)^1.5")
# # # # # # # # #     print("    2. Multi-scale haversine: 12h+24h+48h+72h nested")
# # # # # # # # #     print("    3. Endpoint-weighted loss (gamma=2)")
# # # # # # # # #     print("    4. Trajectory shape loss (multi-window)")
# # # # # # # # #     print("    5. Steering alignment + steering-conditioned velocity")
# # # # # # # # #     print("    6. Residual via teacher forcing scheduled")
# # # # # # # # #     print("    7. EMA weights + SWA")
# # # # # # # # #     print("    8. Importance-weighted ensemble sampling")
# # # # # # # # #     print("    9. Data aug: mixup + obs noise + lon flip")
# # # # # # # # #     print("  TARGETS: 12h<50  24h<100  48h<200  72h<300 km")
# # # # # # # # #     print("=" * 70)

# # # # # # # # #     # ── Data ──────────────────────────────────────────────────────────────
# # # # # # # # #     train_dataset, train_loader = data_loader(
# # # # # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # # # # #     val_dataset, val_loader = data_loader(
# # # # # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # # #     val_subset_loader = make_val_subset_loader(
# # # # # # # # #         val_dataset, args.val_subset_size, args.batch_size,
# # # # # # # # #         seq_collate, args.num_workers)

# # # # # # # # #     try:
# # # # # # # # #         _, test_loader = data_loader(
# # # # # # # # #             args, {"root": args.dataset_root, "type": "test"},
# # # # # # # # #             test=True, test_year=None)
# # # # # # # # #     except Exception as e:
# # # # # # # # #         print(f"  Warning: test loader: {e}")

# # # # # # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # # # # # #     print(f"  val   : {len(val_dataset)} seq")

# # # # # # # # #     # ── Model ─────────────────────────────────────────────────────────────
# # # # # # # # #     model = TCFlowMatching(
# # # # # # # # #         pred_len             = args.pred_len,
# # # # # # # # #         obs_len              = args.obs_len,
# # # # # # # # #         sigma_min            = args.sigma_min,
# # # # # # # # #         n_train_ens          = args.n_train_ens,
# # # # # # # # #         ctx_noise_scale      = args.ctx_noise_scale,
# # # # # # # # #         initial_sample_sigma = args.initial_sample_sigma,
# # # # # # # # #         teacher_forcing      = args.teacher_forcing,
# # # # # # # # #         use_ema              = args.use_ema,
# # # # # # # # #         ema_decay            = args.ema_decay,
# # # # # # # # #     ).to(device)

# # # # # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # # # #     print(f"  params  : {n_params:,}")

# # # # # # # # #     # Initialize EMA AFTER model on device
# # # # # # # # #     model.init_ema()
# # # # # # # # #     print(f"  EMA     : {'ON' if model._ema is not None else 'OFF'} (decay={args.ema_decay})")

# # # # # # # # #     # ── Optimizer ─────────────────────────────────────────────────────────
# # # # # # # # #     optimizer = optim.AdamW(
# # # # # # # # #         model.parameters(),
# # # # # # # # #         lr=args.g_learning_rate,
# # # # # # # # #         weight_decay=args.weight_decay,
# # # # # # # # #     )

# # # # # # # # #     # Resume
# # # # # # # # #     start_epoch = 0
# # # # # # # # #     if args.resume is not None and os.path.exists(args.resume):
# # # # # # # # #         print(f"  Loading checkpoint: {args.resume}")
# # # # # # # # #         ckpt = torch.load(args.resume, map_location=device)
# # # # # # # # #         model.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # # # # # #         start_epoch = args.resume_epoch
# # # # # # # # #         print(f"  Resumed from epoch {start_epoch}")

# # # # # # # # #     try:
# # # # # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # # # # #         print("  torch.compile: enabled")
# # # # # # # # #     except Exception:
# # # # # # # # #         pass

# # # # # # # # #     steps_per_epoch = len(train_loader)
# # # # # # # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # # # # # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # # # # # # #     scheduler = get_cosine_schedule_with_warmup(
# # # # # # # # #         optimizer, warmup, total_steps, min_lr=1e-6)

# # # # # # # # #     if start_epoch > 0:
# # # # # # # # #         for _ in range(start_epoch * steps_per_epoch):
# # # # # # # # #             scheduler.step()

# # # # # # # # #     saver  = HorizonAwareBestSaver(patience=args.patience, tol=1.5)
# # # # # # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # # # # # # #     swa    = SWAManager(model, start_epoch=args.swa_start_epoch)

# # # # # # # # #     print("=" * 70)
# # # # # # # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch)")
# # # # # # # # #     print("=" * 70)

# # # # # # # # #     epoch_times = []
# # # # # # # # #     train_start = time.perf_counter()

# # # # # # # # #     for epoch in range(start_epoch, args.num_epochs):
# # # # # # # # #         model.train()
# # # # # # # # #         sum_loss = 0.0
# # # # # # # # #         t0 = time.perf_counter()

# # # # # # # # #         for i, batch in enumerate(train_loader):
# # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # #             if epoch == start_epoch and i == 0:
# # # # # # # # #                 _check_env(bl, train_dataset)

# # # # # # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # #                 bd = model.get_loss_breakdown(bl, epoch=epoch)

# # # # # # # # #             optimizer.zero_grad()
# # # # # # # # #             scaler.scale(bd["total"]).backward()
# # # # # # # # #             scaler.unscale_(optimizer)
# # # # # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # # # # # #             scaler.step(optimizer)
# # # # # # # # #             scaler.update()
# # # # # # # # #             scheduler.step()

# # # # # # # # #             # EMA update
# # # # # # # # #             if hasattr(model, 'ema_update'):
# # # # # # # # #                 model.ema_update()
# # # # # # # # #             elif hasattr(model, '_orig_mod') and hasattr(model._orig_mod, 'ema_update'):
# # # # # # # # #                 model._orig_mod.ema_update()

# # # # # # # # #             sum_loss += bd["total"].item()

# # # # # # # # #             if i % 20 == 0:
# # # # # # # # #                 lr = optimizer.param_groups[0]["lr"]
# # # # # # # # #                 elapsed = time.perf_counter() - t0
# # # # # # # # #                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # # # # # # #                       f"  tot={bd['total'].item():.3f}"
# # # # # # # # #                       f"  fm={bd['fm_mse']:.3f}"
# # # # # # # # #                       f"  hor={bd['mse_hav']:.3f}"
# # # # # # # # #                       f"  ms={bd['multi_scale']:.3f}"
# # # # # # # # #                       f"  end={bd['endpoint']:.3f}"
# # # # # # # # #                       f"  shp={bd['shape']:.3f}"
# # # # # # # # #                       f"  vel={bd['velocity']:.3f}"
# # # # # # # # #                       f"  hd={bd['heading']:.3f}"
# # # # # # # # #                       f"  str={bd['steering']:.3f}"
# # # # # # # # #                       f"  σ={bd['sigma']:.3f}"
# # # # # # # # #                       f"  lr={lr:.2e}")

# # # # # # # # #         ep_s  = time.perf_counter() - t0
# # # # # # # # #         epoch_times.append(ep_s)
# # # # # # # # #         avg_t = sum_loss / len(train_loader)

# # # # # # # # #         # SWA update
# # # # # # # # #         swa.update(model, epoch)

# # # # # # # # #         # Val loss
# # # # # # # # #         model.eval()
# # # # # # # # #         val_loss = 0.0
# # # # # # # # #         with torch.no_grad():
# # # # # # # # #             for batch in val_loader:
# # # # # # # # #                 bl_v = move(list(batch), device)
# # # # # # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # #                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
# # # # # # # # #         avg_vl = val_loss / len(val_loader)

# # # # # # # # #         print(f"  Epoch {epoch:>3}  train={avg_t:.4f}  val={avg_vl:.4f}"
# # # # # # # # #               f"  time={ep_s:.0f}s")

# # # # # # # # #         # Fast eval (RAW weights)
# # # # # # # # #         m_fast = evaluate_fast(model, val_subset_loader, device,
# # # # # # # # #                                 args.ode_steps_train, args.pred_len,
# # # # # # # # #                                 args.fast_ensemble)

# # # # # # # # #         h6  = m_fast.get("6h",  float("nan"))
# # # # # # # # #         h12 = m_fast.get("12h", float("nan"))
# # # # # # # # #         h24 = m_fast.get("24h", float("nan"))
# # # # # # # # #         h48 = m_fast.get("48h", float("nan"))
# # # # # # # # #         h72 = m_fast.get("72h", float("nan"))
# # # # # # # # #         fast_score = _composite_score(m_fast)

# # # # # # # # #         t6  = "🎯" if h6  < 30  else "❌"
# # # # # # # # #         t12 = "🎯" if h12 < 50  else "❌"
# # # # # # # # #         t24 = "🎯" if h24 < 100 else "❌"
# # # # # # # # #         t48 = "🎯" if h48 < 200 else "❌"
# # # # # # # # #         t72 = "🎯" if h72 < 300 else "❌"

# # # # # # # # #         print(f"  [FAST ep{epoch}]"
# # # # # # # # #               f"  ADE={m_fast['ADE']:.1f}"
# # # # # # # # #               f"  6h={h6:.0f}{t6}"
# # # # # # # # #               f"  12h={h12:.0f}{t12}"
# # # # # # # # #               f"  24h={h24:.0f}{t24}"
# # # # # # # # #               f"  48h={h48:.0f}{t48}"
# # # # # # # # #               f"  72h={h72:.0f}{t72}"
# # # # # # # # #               f"  score={fast_score:.1f}")

# # # # # # # # #         # Full val (with EMA if available)
# # # # # # # # #         if epoch % args.val_ade_freq == 0:
# # # # # # # # #             # Get EMA ref
# # # # # # # # #             ema_obj = None
# # # # # # # # #             if hasattr(model, '_ema'):
# # # # # # # # #                 ema_obj = model._ema
# # # # # # # # #             elif hasattr(model, '_orig_mod') and hasattr(model._orig_mod, '_ema'):
# # # # # # # # #                 ema_obj = model._orig_mod._ema

# # # # # # # # #             try:
# # # # # # # # #                 # RAW eval
# # # # # # # # #                 r_raw = evaluate_full_val_ade(
# # # # # # # # #                     model, val_loader, device,
# # # # # # # # #                     ode_steps=args.ode_steps_train, pred_len=args.pred_len,
# # # # # # # # #                     fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # # # # # #                     epoch=epoch, use_ema=False, ema_obj=None)

# # # # # # # # #                 # EMA eval (if available and past warmup)
# # # # # # # # #                 r_use = r_raw
# # # # # # # # #                 if ema_obj is not None and epoch >= 10:
# # # # # # # # #                     r_ema = evaluate_full_val_ade(
# # # # # # # # #                         model, val_loader, device,
# # # # # # # # #                         ode_steps=args.ode_steps_train, pred_len=args.pred_len,
# # # # # # # # #                         fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # # # # # #                         epoch=epoch, use_ema=True, ema_obj=ema_obj)
# # # # # # # # #                     # Use EMA result for saver if better
# # # # # # # # #                     if _composite_score(r_ema) < _composite_score(r_raw):
# # # # # # # # #                         r_use = r_ema

# # # # # # # # #                 saver.update(r_use, model, args.output_dir, epoch,
# # # # # # # # #                            optimizer, avg_t, avg_vl,
# # # # # # # # #                            min_epochs=args.min_epochs)
# # # # # # # # #             except Exception as e:
# # # # # # # # #                 print(f"  ⚠  Full val failed: {e}")
# # # # # # # # #                 import traceback; traceback.print_exc()

# # # # # # # # #         # Checkpoint
# # # # # # # # #         if epoch % 5 == 0 or epoch == args.num_epochs - 1:
# # # # # # # # #             torch.save({
# # # # # # # # #                 "epoch": epoch,
# # # # # # # # #                 "model_state_dict": model.state_dict(),
# # # # # # # # #                 "optimizer_state": optimizer.state_dict(),
# # # # # # # # #                 "train_loss": avg_t, "val_loss": avg_vl,
# # # # # # # # #                 "ade": m_fast.get("ADE", float("nan")),
# # # # # # # # #                 "h48": h48, "h72": h72,
# # # # # # # # #             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))

# # # # # # # # #         if epoch % 10 == 9:
# # # # # # # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # # # # # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # # # # # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # # # # # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")

# # # # # # # # #         if saver.early_stop:
# # # # # # # # #             print(f"  ⛔ Early stopping at epoch {epoch}")
# # # # # # # # #             break

# # # # # # # # #     # ── Final: evaluate SWA weights ────────────────────────────────────────
# # # # # # # # #     print(f"\n{'='*70}")
# # # # # # # # #     print("  Evaluating SWA weights...")
# # # # # # # # #     swa_backup = swa.apply_to(model)
# # # # # # # # #     if swa_backup is not None:
# # # # # # # # #         r_swa = evaluate_full_val_ade(
# # # # # # # # #             model, val_loader, device,
# # # # # # # # #             ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # # # # # # #             fast_ensemble=args.val_ensemble, metrics_csv=metrics_csv,
# # # # # # # # #             epoch=9999, use_ema=False)
# # # # # # # # #         if _composite_score(r_swa) < saver.best_score:
# # # # # # # # #             print(f"  ✅ SWA improved composite: {saver.best_score:.1f} → {_composite_score(r_swa):.1f}")
# # # # # # # # #             torch.save(dict(
# # # # # # # # #                 epoch=9999, model_state_dict=model.state_dict(),
# # # # # # # # #                 ade=r_swa.get("ADE", float("nan")),
# # # # # # # # #                 h72=r_swa.get("72h", float("nan")),
# # # # # # # # #                 tag="best_swa", model_version="v34-swa",
# # # # # # # # #             ), os.path.join(args.output_dir, "best_swa.pth"))
# # # # # # # # #         swa.restore(model, swa_backup)

# # # # # # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # # # # # #     print(f"\n{'='*70}")
# # # # # # # # #     print(f"  Best COMPOSITE : {saver.best_score:.1f}")
# # # # # # # # #     print(f"  Best ADE  : {saver.best_ade:.1f} km")
# # # # # # # # #     print(f"  Best 12h  : {saver.best_12h:.1f} km")
# # # # # # # # #     print(f"  Best 24h  : {saver.best_24h:.1f} km")
# # # # # # # # #     print(f"  Best 48h  : {saver.best_48h:.1f} km")
# # # # # # # # #     print(f"  Best 72h  : {saver.best_72h:.1f} km")
# # # # # # # # #     print(f"  Training  : {total_h:.2f}h")
# # # # # # # # #     print("=" * 70)


# # # # # # # # # if __name__ == "__main__":
# # # # # # # # #     args = get_args()
# # # # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # #         torch.cuda.manual_seed_all(42)
# # # # # # # # #     main(args)

# # # # # # # # # """
# # # # # # # # # train_flowmatching_v34fix.py — Training cho TC-FlowMatching v34fix
# # # # # # # # # ═══════════════════════════════════════════════════════════════════

# # # # # # # # # FIXES từ v34:
# # # # # # # # #   BUG 1 (EMA KeyError): torch.compile đổi tên key trong state_dict.
# # # # # # # # #          evaluate_full_val_ade phải lấy ema_obj từ model._orig_mod
# # # # # # # # #          (nếu compiled) thay vì model._ema (không tồn tại sau compile).
# # # # # # # # #   BUG 2 (EMA update sau compile): ema_update() gọi trên compiled model
# # # # # # # # #          phải đi qua _orig_mod.ema_update() để đúng.

# # # # # # # # # COMPOSITE SCORE:
# # # # # # # # #   score = 0.20*ADE + 0.20*12h + 0.20*24h + 0.20*48h + 0.20*72h
# # # # # # # # # """
# # # # # # # # # from __future__ import annotations

# # # # # # # # # import sys
# # # # # # # # # import os
# # # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # # import argparse
# # # # # # # # # import time
# # # # # # # # # import math
# # # # # # # # # import random
# # # # # # # # # import copy

# # # # # # # # # import numpy as np
# # # # # # # # # import torch
# # # # # # # # # import torch.optim as optim
# # # # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # # from Model.data.loader_training import data_loader
# # # # # # # # # from Model.flow_matching_model import TCFlowMatching
# # # # # # # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # # # from utils.metrics import (
# # # # # # # # #     TCEvaluator, StepErrorAccumulator,
# # # # # # # # #     save_metrics_csv, haversine_km_torch,
# # # # # # # # #     denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
# # # # # # # # # )


# # # # # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # # # # def move(batch, device):
# # # # # # # # #     out = list(batch)
# # # # # # # # #     for i, x in enumerate(out):
# # # # # # # # #         if torch.is_tensor(x):
# # # # # # # # #             out[i] = x.to(device)
# # # # # # # # #         elif isinstance(x, dict):
# # # # # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # # #                       for k, v in x.items()}
# # # # # # # # #     return out


# # # # # # # # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # # # # # # # #                             collate_fn, num_workers):
# # # # # # # # #     n   = len(val_dataset)
# # # # # # # # #     rng = random.Random(42)
# # # # # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # # # # #     return DataLoader(Subset(val_dataset, idx),
# # # # # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # # # # def _composite_score(result):
# # # # # # # # #     ade = result.get("ADE", float("inf"))
# # # # # # # # #     h12 = result.get("12h", float("inf"))
# # # # # # # # #     h24 = result.get("24h", float("inf"))
# # # # # # # # #     h48 = result.get("48h", float("inf"))
# # # # # # # # #     h72 = result.get("72h", float("inf"))
# # # # # # # # #     return 0.20 * ade + 0.20 * h12 + 0.20 * h24 + 0.20 * h48 + 0.20 * h72


# # # # # # # # # def _get_ema_obj(model):
# # # # # # # # #     """
# # # # # # # # #     FIX: lấy EMAModel object từ model, xử lý cả trường hợp
# # # # # # # # #     model đã được torch.compile (wrapped thành _orig_mod).
# # # # # # # # #     """
# # # # # # # # #     # Trường hợp 1: raw model (chưa compile)
# # # # # # # # #     if hasattr(model, '_ema') and model._ema is not None:
# # # # # # # # #         return model._ema
# # # # # # # # #     # Trường hợp 2: compiled model → unwrap _orig_mod
# # # # # # # # #     if hasattr(model, '_orig_mod'):
# # # # # # # # #         orig = model._orig_mod
# # # # # # # # #         if hasattr(orig, '_ema') and orig._ema is not None:
# # # # # # # # #             return orig._ema
# # # # # # # # #     return None


# # # # # # # # # def _call_ema_update(model):
# # # # # # # # #     """
# # # # # # # # #     FIX: gọi ema_update() đúng cách sau torch.compile.
# # # # # # # # #     torch.compile wrap forward() nhưng không wrap custom methods,
# # # # # # # # #     nên cần gọi trực tiếp trên _orig_mod.
# # # # # # # # #     """
# # # # # # # # #     # Trường hợp 1: raw model
# # # # # # # # #     if hasattr(model, 'ema_update') and not hasattr(model, '_orig_mod'):
# # # # # # # # #         model.ema_update()
# # # # # # # # #         return
# # # # # # # # #     # Trường hợp 2: compiled model
# # # # # # # # #     if hasattr(model, '_orig_mod'):
# # # # # # # # #         orig = model._orig_mod
# # # # # # # # #         if hasattr(orig, 'ema_update'):
# # # # # # # # #             orig.ema_update()
# # # # # # # # #             return
# # # # # # # # #     # Fallback: gọi trực tiếp
# # # # # # # # #     if hasattr(model, 'ema_update'):
# # # # # # # # #         model.ema_update()


# # # # # # # # # # ── SWA ───────────────────────────────────────────────────────────────────────

# # # # # # # # # class SWAManager:
# # # # # # # # #     def __init__(self, model, start_epoch=50):
# # # # # # # # #         self.start_epoch = start_epoch
# # # # # # # # #         self.n_averaged  = 0
# # # # # # # # #         self.avg_state   = None

# # # # # # # # #     def update(self, model, epoch):
# # # # # # # # #         if epoch < self.start_epoch:
# # # # # # # # #             return
# # # # # # # # #         # Unwrap compiled model
# # # # # # # # #         m = model._orig_mod if hasattr(model, '_orig_mod') else model
# # # # # # # # #         sd = {k: v.detach().clone() for k, v in m.state_dict().items()
# # # # # # # # #               if v.dtype.is_floating_point}
# # # # # # # # #         if self.avg_state is None:
# # # # # # # # #             self.avg_state = sd
# # # # # # # # #             self.n_averaged = 1
# # # # # # # # #         else:
# # # # # # # # #             n = self.n_averaged
# # # # # # # # #             for k in self.avg_state:
# # # # # # # # #                 if k in sd:
# # # # # # # # #                     self.avg_state[k].mul_(n / (n + 1)).add_(sd[k], alpha=1.0 / (n + 1))
# # # # # # # # #             self.n_averaged += 1

# # # # # # # # #     def apply_to(self, model):
# # # # # # # # #         if self.avg_state is None:
# # # # # # # # #             return None
# # # # # # # # #         m = model._orig_mod if hasattr(model, '_orig_mod') else model
# # # # # # # # #         backup = {}
# # # # # # # # #         sd = m.state_dict()
# # # # # # # # #         for k, v in self.avg_state.items():
# # # # # # # # #             if k in sd:
# # # # # # # # #                 backup[k] = sd[k].detach().clone()
# # # # # # # # #                 sd[k].copy_(v)
# # # # # # # # #         return backup

# # # # # # # # #     def restore(self, model, backup):
# # # # # # # # #         if backup is None:
# # # # # # # # #             return
# # # # # # # # #         m = model._orig_mod if hasattr(model, '_orig_mod') else model
# # # # # # # # #         sd = m.state_dict()
# # # # # # # # #         for k, v in backup.items():
# # # # # # # # #             if k in sd:
# # # # # # # # #                 sd[k].copy_(v)


# # # # # # # # # # ── Evaluation ────────────────────────────────────────────────────────────────

# # # # # # # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=15):
# # # # # # # # #     model.eval()
# # # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # # #     t0  = time.perf_counter()
# # # # # # # # #     n   = 0
# # # # # # # # #     spread_per_step = []

# # # # # # # # #     with torch.no_grad():
# # # # # # # # #         for batch in loader:
# # # # # # # # #             bl = move(list(batch), device)
# # # # # # # # #             pred, _, all_trajs = model.sample(
# # # # # # # # #                 bl, num_ensemble=fast_ensemble, ddim_steps=ode_steps,
# # # # # # # # #                 importance_weight=True)
# # # # # # # # #             T_active  = pred.shape[0]
# # # # # # # # #             gt_sliced = bl[1][:T_active]
# # # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
# # # # # # # # #             acc.update(dist)

# # # # # # # # #             step_spreads = []
# # # # # # # # #             for t in range(all_trajs.shape[1]):
# # # # # # # # #                 step_data = all_trajs[:, t, :, :]
# # # # # # # # #                 std_lon = step_data[:, :, 0].std(0)
# # # # # # # # #                 std_lat = step_data[:, :, 1].std(0)
# # # # # # # # #                 spread = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# # # # # # # # #                 step_spreads.append(spread)
# # # # # # # # #             spread_per_step.append(step_spreads)
# # # # # # # # #             n += 1

# # # # # # # # #     r = acc.compute()
# # # # # # # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# # # # # # # # #     if spread_per_step:
# # # # # # # # #         spreads = np.array(spread_per_step)
# # # # # # # # #         r["spread_12h_km"] = float(spreads[:, 1].mean()) if spreads.shape[1] > 1 else 0.0
# # # # # # # # #         r["spread_24h_km"] = float(spreads[:, 3].mean()) if spreads.shape[1] > 3 else 0.0
# # # # # # # # #         r["spread_72h_km"] = float(spreads[:, -1].mean())
# # # # # # # # #     return r


# # # # # # # # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # # # # # # # #                            fast_ensemble, metrics_csv, epoch,
# # # # # # # # #                            use_ema=False, ema_obj=None):
# # # # # # # # #     """
# # # # # # # # #     Full val eval, optionally with EMA weights.
# # # # # # # # #     FIX: dùng _get_ema_obj() để lấy đúng EMAModel kể cả sau torch.compile.
# # # # # # # # #     """
# # # # # # # # #     backup = None
# # # # # # # # #     if use_ema and ema_obj is not None:
# # # # # # # # #         try:
# # # # # # # # #             backup = ema_obj.apply_to(model)
# # # # # # # # #         except Exception as e:
# # # # # # # # #             print(f"  ⚠  EMA apply_to failed: {e} — skipping EMA eval")
# # # # # # # # #             backup = None
# # # # # # # # #             use_ema = False

# # # # # # # # #     model.eval()
# # # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # # #     t0  = time.perf_counter()
# # # # # # # # #     n   = 0

# # # # # # # # #     with torch.no_grad():
# # # # # # # # #         for batch in val_loader:
# # # # # # # # #             bl = move(list(batch), device)
# # # # # # # # #             pred, _, _ = model.sample(
# # # # # # # # #                 bl, num_ensemble=max(fast_ensemble, 20),
# # # # # # # # #                 ddim_steps=max(ode_steps, 30),
# # # # # # # # #                 importance_weight=True)
# # # # # # # # #             T_pred = pred.shape[0]
# # # # # # # # #             gt = bl[1][:T_pred]
# # # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt))
# # # # # # # # #             acc.update(dist)
# # # # # # # # #             n += 1

# # # # # # # # #     r = acc.compute()
# # # # # # # # #     elapsed = time.perf_counter() - t0
# # # # # # # # #     score   = _composite_score(r)

# # # # # # # # #     tag = "EMA" if use_ema else "RAW"
# # # # # # # # #     print(f"\n{'='*64}")
# # # # # # # # #     print(f"  [FULL VAL ADE ({tag})  ep={epoch}  {elapsed:.0f}s  {n} batches]")
# # # # # # # # #     print(f"  ADE = {r.get('ADE', float('nan')):.1f} km  "
# # # # # # # # #           f"FDE = {r.get('FDE', float('nan')):.1f} km")
# # # # # # # # #     print(f"  6h={r.get('6h', float('nan')):.0f}  "
# # # # # # # # #           f"12h={r.get('12h', float('nan')):.0f}  "
# # # # # # # # #           f"24h={r.get('24h', float('nan')):.0f}  "
# # # # # # # # #           f"48h={r.get('48h', float('nan')):.0f}  "
# # # # # # # # #           f"72h={r.get('72h', float('nan')):.0f} km")
# # # # # # # # #     print(f"  Composite score = {score:.1f}")
# # # # # # # # #     print(f"{'='*64}\n")

# # # # # # # # #     from datetime import datetime
# # # # # # # # #     from utils.metrics import DatasetMetrics, save_metrics_csv as _save_csv
# # # # # # # # #     dm = DatasetMetrics(
# # # # # # # # #         ade      = r.get("ADE",  float("nan")),
# # # # # # # # #         fde      = r.get("FDE",  float("nan")),
# # # # # # # # #         ugde_12h = r.get("12h",  float("nan")),
# # # # # # # # #         ugde_24h = r.get("24h",  float("nan")),
# # # # # # # # #         ugde_48h = r.get("48h",  float("nan")),
# # # # # # # # #         ugde_72h = r.get("72h",  float("nan")),
# # # # # # # # #         n_total  = r.get("n_samples", 0),
# # # # # # # # #         timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # # # # #     )
# # # # # # # # #     _save_csv(dm, metrics_csv, tag=f"val_full_{tag}_ep{epoch:03d}")

# # # # # # # # #     # Restore weights nếu đã apply EMA
# # # # # # # # #     if backup is not None:
# # # # # # # # #         try:
# # # # # # # # #             ema_obj.restore(model, backup)
# # # # # # # # #         except Exception as e:
# # # # # # # # #             print(f"  ⚠  EMA restore failed: {e}")

# # # # # # # # #     return r


# # # # # # # # # # ── Horizon-specific BestModelSaver ───────────────────────────────────────────

# # # # # # # # # class HorizonAwareBestSaver:
# # # # # # # # #     def __init__(self, patience=30, tol=1.5):
# # # # # # # # #         self.patience  = patience
# # # # # # # # #         self.tol       = tol
# # # # # # # # #         self.counter   = 0
# # # # # # # # #         self.early_stop = False

# # # # # # # # #         self.best_score = float("inf")
# # # # # # # # #         self.best_ade   = float("inf")
# # # # # # # # #         self.best_12h   = float("inf")
# # # # # # # # #         self.best_24h   = float("inf")
# # # # # # # # #         self.best_48h   = float("inf")
# # # # # # # # #         self.best_72h   = float("inf")

# # # # # # # # #     def _unwrap_state_dict(self, model):
# # # # # # # # #         """Lấy state_dict từ raw model (unwrap compile nếu cần)."""
# # # # # # # # #         m = model._orig_mod if hasattr(model, '_orig_mod') else model
# # # # # # # # #         return m.state_dict()

# # # # # # # # #     def update(self, r, model, out_dir, epoch, optimizer, tl, vl,
# # # # # # # # #                 min_epochs=50):
# # # # # # # # #         ade  = r.get("ADE", float("inf"))
# # # # # # # # #         h12  = r.get("12h", float("inf"))
# # # # # # # # #         h24  = r.get("24h", float("inf"))
# # # # # # # # #         h48  = r.get("48h", float("inf"))
# # # # # # # # #         h72  = r.get("72h", float("inf"))
# # # # # # # # #         score = _composite_score(r)

# # # # # # # # #         improved_score = score < self.best_score - self.tol
# # # # # # # # #         improved_any   = False

# # # # # # # # #         sd = self._unwrap_state_dict(model)

# # # # # # # # #         if ade < self.best_ade:
# # # # # # # # #             self.best_ade = ade
# # # # # # # # #             improved_any  = True
# # # # # # # # #             torch.save(dict(
# # # # # # # # #                 epoch=epoch, model_state_dict=sd,
# # # # # # # # #                 ade=ade, h12=h12, h24=h24, h48=h48, h72=h72, tag="best_ade",
# # # # # # # # #             ), os.path.join(out_dir, "best_ade.pth"))
# # # # # # # # #         if h72 < self.best_72h:
# # # # # # # # #             self.best_72h = h72
# # # # # # # # #             improved_any  = True
# # # # # # # # #             torch.save(dict(
# # # # # # # # #                 epoch=epoch, model_state_dict=sd,
# # # # # # # # #                 ade=ade, h12=h12, h24=h24, h48=h48, h72=h72, tag="best_72h",
# # # # # # # # #             ), os.path.join(out_dir, "best_72h.pth"))
# # # # # # # # #         if h48 < self.best_48h: self.best_48h = h48; improved_any = True
# # # # # # # # #         if h24 < self.best_24h: self.best_24h = h24; improved_any = True
# # # # # # # # #         if h12 < self.best_12h: self.best_12h = h12; improved_any = True

# # # # # # # # #         if improved_score:
# # # # # # # # #             self.best_score = score
# # # # # # # # #             self.counter    = 0
# # # # # # # # #             torch.save(dict(
# # # # # # # # #                 epoch=epoch, model_state_dict=sd,
# # # # # # # # #                 optimizer_state=optimizer.state_dict(),
# # # # # # # # #                 train_loss=tl, val_loss=vl,
# # # # # # # # #                 ade=ade, h12=h12, h24=h24, h48=h48, h72=h72,
# # # # # # # # #                 composite_score=score, tag="best_composite",
# # # # # # # # #                 model_version="v34fix",
# # # # # # # # #             ), os.path.join(out_dir, "best_model.pth"))
# # # # # # # # #             print(f"  ✅ Best COMPOSITE={score:.1f}  "
# # # # # # # # #                   f"ADE={ade:.1f}  12h={h12:.0f}  24h={h24:.0f}  "
# # # # # # # # #                   f"48h={h48:.0f}  72h={h72:.0f}  (ep {epoch})")
# # # # # # # # #         else:
# # # # # # # # #             if not improved_any:
# # # # # # # # #                 self.counter += 1
# # # # # # # # #             print(f"  No improvement {self.counter}/{self.patience}"
# # # # # # # # #                   f"  (best_score={self.best_score:.1f}, cur={score:.1f})")

# # # # # # # # #         if epoch >= min_epochs and self.counter >= self.patience:
# # # # # # # # #             self.early_stop = True


# # # # # # # # # # ── Env diagnostic ────────────────────────────────────────────────────────────

# # # # # # # # # def _check_env(bl, train_dataset):
# # # # # # # # #     try:
# # # # # # # # #         env_dir = train_dataset.env_path
# # # # # # # # #     except AttributeError:
# # # # # # # # #         try:
# # # # # # # # #             env_dir = train_dataset.dataset.env_path
# # # # # # # # #         except AttributeError:
# # # # # # # # #             env_dir = "UNKNOWN"
# # # # # # # # #     print(f"  Env path: {env_dir}")

# # # # # # # # #     env_data = bl[13]
# # # # # # # # #     if env_data is None:
# # # # # # # # #         print("  ⚠️  env_data is None"); return

# # # # # # # # #     for key in ("gph500_mean", "gph500_center", "u500_mean", "v500_mean"):
# # # # # # # # #         if key not in env_data:
# # # # # # # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # # # # # # #         v    = env_data[key]
# # # # # # # # #         mn   = v.mean().item()
# # # # # # # # #         std  = v.std().item()
# # # # # # # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # # # # # # #         print(f"  {'✅' if zero < 80 else '⚠️'}  {key}: mean={mn:.4f} std={std:.4f} zero={zero:.1f}%")


# # # # # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # # # # def get_args():
# # # # # # # # #     p = argparse.ArgumentParser(
# # # # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # # # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # # # # # # # #     p.add_argument("--obs_len",         default=8,          type=int)
# # # # # # # # #     p.add_argument("--pred_len",        default=12,         type=int)
# # # # # # # # #     p.add_argument("--batch_size",      default=32,         type=int)
# # # # # # # # #     p.add_argument("--num_epochs",      default=120,        type=int)
# # # # # # # # #     p.add_argument("--g_learning_rate", default=3e-4,       type=float)
# # # # # # # # #     p.add_argument("--weight_decay",    default=1e-4,       type=float)
# # # # # # # # #     p.add_argument("--warmup_epochs",   default=5,          type=int)
# # # # # # # # #     p.add_argument("--grad_clip",       default=1.0,        type=float)
# # # # # # # # #     p.add_argument("--patience",        default=30,         type=int)
# # # # # # # # #     p.add_argument("--min_epochs",      default=50,         type=int)
# # # # # # # # #     p.add_argument("--n_train_ens",     default=4,          type=int)
# # # # # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # # # # #     p.add_argument("--num_workers",     default=2,          type=int)

# # # # # # # # #     p.add_argument("--sigma_min",            default=0.02,  type=float)
# # # # # # # # #     p.add_argument("--ctx_noise_scale",      default=0.01,  type=float)
# # # # # # # # #     p.add_argument("--initial_sample_sigma", default=0.03,  type=float)

# # # # # # # # #     p.add_argument("--ode_steps_train", default=25,  type=int)
# # # # # # # # #     p.add_argument("--ode_steps_val",   default=30,  type=int)
# # # # # # # # #     p.add_argument("--ode_steps_test",  default=50,  type=int)

# # # # # # # # #     p.add_argument("--val_ensemble",    default=30,  type=int)
# # # # # # # # #     p.add_argument("--fast_ensemble",   default=15,  type=int)

# # # # # # # # #     p.add_argument("--val_freq",        default=1,   type=int)
# # # # # # # # #     p.add_argument("--val_ade_freq",    default=1,   type=int)
# # # # # # # # #     p.add_argument("--val_subset_size", default=600, type=int)

# # # # # # # # #     p.add_argument("--use_ema",         action="store_true", default=True)
# # # # # # # # #     p.add_argument("--ema_decay",       default=0.999, type=float)
# # # # # # # # #     p.add_argument("--swa_start_epoch", default=55,   type=int)
# # # # # # # # #     p.add_argument("--teacher_forcing", action="store_true", default=True)

# # # # # # # # #     p.add_argument("--output_dir",   default="runs/v34fix",    type=str)
# # # # # # # # #     p.add_argument("--metrics_csv",  default="metrics.csv",    type=str)
# # # # # # # # #     p.add_argument("--predict_csv",  default="predictions.csv", type=str)

# # # # # # # # #     p.add_argument("--gpu_num",      default="0", type=str)
# # # # # # # # #     p.add_argument("--delim",        default=" ")
# # # # # # # # #     p.add_argument("--skip",         default=1,   type=int)
# # # # # # # # #     p.add_argument("--min_ped",      default=1,   type=int)
# # # # # # # # #     p.add_argument("--threshold",    default=0.002, type=float)
# # # # # # # # #     p.add_argument("--other_modal",  default="gph")
# # # # # # # # #     p.add_argument("--test_year",    default=None, type=int)

# # # # # # # # #     p.add_argument("--resume",       default=None, type=str)
# # # # # # # # #     p.add_argument("--resume_epoch", default=0,    type=int)

# # # # # # # # #     return p.parse_args()


# # # # # # # # # # ── MAIN ──────────────────────────────────────────────────────────────────────

# # # # # # # # # def main(args):
# # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)

# # # # # # # # #     print("=" * 70)
# # # # # # # # #     print("  TC-FlowMatching v34fix  |  HORIZON-AWARE + 72h FOCUSED + EMA FIX")
# # # # # # # # #     print("  ─────────────────────────────────────────────")
# # # # # # # # #     print("  FIXES từ v34:")
# # # # # # # # #     print("    1. EMA KeyError: unwrap _orig_mod sau torch.compile")
# # # # # # # # #     print("    2. multi_scale_haversine: 70% endpoint[h] + 30% ADE[:h]")
# # # # # # # # #     print("    3. Sigma floor 0.03→0.06 (ensemble diversity)")
# # # # # # # # #     print("    4. horizon alpha 1.5→2.0 (step 12 weight 9x step 1)")
# # # # # # # # #     print("    5. endpoint gamma 2.0→2.5, norm 300→250")
# # # # # # # # #     print("    6. WEIGHTS: multi_scale 2.0→3.5, endpoint 2.5→3.0")
# # # # # # # # #     print("  TARGETS: 12h<50  24h<100  48h<200  72h<300 km")
# # # # # # # # #     print("=" * 70)

# # # # # # # # #     # ── Data ──────────────────────────────────────────────────────────────
# # # # # # # # #     train_dataset, train_loader = data_loader(
# # # # # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # # # # #     val_dataset, val_loader = data_loader(
# # # # # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # # #     val_subset_loader = make_val_subset_loader(
# # # # # # # # #         val_dataset, args.val_subset_size, args.batch_size,
# # # # # # # # #         seq_collate, args.num_workers)

# # # # # # # # #     try:
# # # # # # # # #         _, test_loader = data_loader(
# # # # # # # # #             args, {"root": args.dataset_root, "type": "test"},
# # # # # # # # #             test=True, test_year=None)
# # # # # # # # #     except Exception as e:
# # # # # # # # #         print(f"  Warning: test loader: {e}")

# # # # # # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # # # # # #     print(f"  val   : {len(val_dataset)} seq")

# # # # # # # # #     # ── Model ─────────────────────────────────────────────────────────────
# # # # # # # # #     model = TCFlowMatching(
# # # # # # # # #         pred_len             = args.pred_len,
# # # # # # # # #         obs_len              = args.obs_len,
# # # # # # # # #         sigma_min            = args.sigma_min,
# # # # # # # # #         n_train_ens          = args.n_train_ens,
# # # # # # # # #         ctx_noise_scale      = args.ctx_noise_scale,
# # # # # # # # #         initial_sample_sigma = args.initial_sample_sigma,
# # # # # # # # #         teacher_forcing      = args.teacher_forcing,
# # # # # # # # #         use_ema              = args.use_ema,
# # # # # # # # #         ema_decay            = args.ema_decay,
# # # # # # # # #     ).to(device)

# # # # # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # # # #     print(f"  params  : {n_params:,}")

# # # # # # # # #     # FIX: init_ema() TRƯỚC khi torch.compile để shadow có đúng key names
# # # # # # # # #     model.init_ema()
# # # # # # # # #     print(f"  EMA     : {'ON' if model._ema is not None else 'OFF'} (decay={args.ema_decay})")

# # # # # # # # #     # ── Optimizer ─────────────────────────────────────────────────────────
# # # # # # # # #     optimizer = optim.AdamW(
# # # # # # # # #         model.parameters(),
# # # # # # # # #         lr=args.g_learning_rate,
# # # # # # # # #         weight_decay=args.weight_decay,
# # # # # # # # #     )

# # # # # # # # #     # Resume
# # # # # # # # #     start_epoch = 0
# # # # # # # # #     if args.resume is not None and os.path.exists(args.resume):
# # # # # # # # #         print(f"  Loading checkpoint: {args.resume}")
# # # # # # # # #         ckpt = torch.load(args.resume, map_location=device)
# # # # # # # # #         # Unwrap nếu cần
# # # # # # # # #         m = model._orig_mod if hasattr(model, '_orig_mod') else model
# # # # # # # # #         m.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # # # # # #         start_epoch = args.resume_epoch
# # # # # # # # #         print(f"  Resumed from epoch {start_epoch}")

# # # # # # # # #     # FIX: torch.compile SAU init_ema() để _orig_mod.net.pos_enc có đúng key
# # # # # # # # #     try:
# # # # # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # # # # #         print("  torch.compile: enabled")
# # # # # # # # #     except Exception:
# # # # # # # # #         pass

# # # # # # # # #     steps_per_epoch = len(train_loader)
# # # # # # # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # # # # # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # # # # # # #     scheduler = get_cosine_schedule_with_warmup(
# # # # # # # # #         optimizer, warmup, total_steps, min_lr=1e-6)

# # # # # # # # #     if start_epoch > 0:
# # # # # # # # #         for _ in range(start_epoch * steps_per_epoch):
# # # # # # # # #             scheduler.step()

# # # # # # # # #     saver  = HorizonAwareBestSaver(patience=args.patience, tol=1.5)
# # # # # # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # # # # # # #     swa    = SWAManager(model, start_epoch=args.swa_start_epoch)

# # # # # # # # #     print("=" * 70)
# # # # # # # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch)")
# # # # # # # # #     print("=" * 70)

# # # # # # # # #     epoch_times = []
# # # # # # # # #     train_start = time.perf_counter()

# # # # # # # # #     for epoch in range(start_epoch, args.num_epochs):
# # # # # # # # #         model.train()
# # # # # # # # #         sum_loss = 0.0
# # # # # # # # #         t0 = time.perf_counter()

# # # # # # # # #         for i, batch in enumerate(train_loader):
# # # # # # # # #             bl = move(list(batch), device)

# # # # # # # # #             if epoch == start_epoch and i == 0:
# # # # # # # # #                 _check_env(bl, train_dataset)

# # # # # # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # #                 bd = model.get_loss_breakdown(bl, epoch=epoch)

# # # # # # # # #             optimizer.zero_grad()
# # # # # # # # #             scaler.scale(bd["total"]).backward()
# # # # # # # # #             scaler.unscale_(optimizer)
# # # # # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # # # # # #             scaler.step(optimizer)
# # # # # # # # #             scaler.update()
# # # # # # # # #             scheduler.step()

# # # # # # # # #             # FIX: dùng _call_ema_update() thay vì trực tiếp gọi model.ema_update()
# # # # # # # # #             _call_ema_update(model)

# # # # # # # # #             sum_loss += bd["total"].item()

# # # # # # # # #             if i % 20 == 0:
# # # # # # # # #                 lr = optimizer.param_groups[0]["lr"]
# # # # # # # # #                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # # # # # # #                       f"  tot={bd['total'].item():.3f}"
# # # # # # # # #                       f"  fm={bd['fm_mse']:.3f}"
# # # # # # # # #                       f"  hor={bd['mse_hav']:.3f}"
# # # # # # # # #                       f"  ms={bd['multi_scale']:.3f}"
# # # # # # # # #                       f"  end={bd['endpoint']:.3f}"
# # # # # # # # #                       f"  shp={bd['shape']:.3f}"
# # # # # # # # #                       f"  vel={bd['velocity']:.3f}"
# # # # # # # # #                       f"  hd={bd['heading']:.3f}"
# # # # # # # # #                       f"  str={bd['steering']:.3f}"
# # # # # # # # #                       f"  σ={bd['sigma']:.3f}"
# # # # # # # # #                       f"  lr={lr:.2e}")

# # # # # # # # #         ep_s  = time.perf_counter() - t0
# # # # # # # # #         epoch_times.append(ep_s)
# # # # # # # # #         avg_t = sum_loss / len(train_loader)

# # # # # # # # #         swa.update(model, epoch)

# # # # # # # # #         # Val loss
# # # # # # # # #         model.eval()
# # # # # # # # #         val_loss = 0.0
# # # # # # # # #         with torch.no_grad():
# # # # # # # # #             for batch in val_loader:
# # # # # # # # #                 bl_v = move(list(batch), device)
# # # # # # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # # #                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
# # # # # # # # #         avg_vl = val_loss / len(val_loader)

# # # # # # # # #         print(f"  Epoch {epoch:>3}  train={avg_t:.4f}  val={avg_vl:.4f}"
# # # # # # # # #               f"  time={ep_s:.0f}s")

# # # # # # # # #         # Fast eval
# # # # # # # # #         m_fast = evaluate_fast(model, val_subset_loader, device,
# # # # # # # # #                                 args.ode_steps_train, args.pred_len,
# # # # # # # # #                                 args.fast_ensemble)

# # # # # # # # #         h6  = m_fast.get("6h",  float("nan"))
# # # # # # # # #         h12 = m_fast.get("12h", float("nan"))
# # # # # # # # #         h24 = m_fast.get("24h", float("nan"))
# # # # # # # # #         h48 = m_fast.get("48h", float("nan"))
# # # # # # # # #         h72 = m_fast.get("72h", float("nan"))
# # # # # # # # #         fast_score = _composite_score(m_fast)

# # # # # # # # #         t6  = "🎯" if h6  < 30  else "❌"
# # # # # # # # #         t12 = "🎯" if h12 < 50  else "❌"
# # # # # # # # #         t24 = "🎯" if h24 < 100 else "❌"
# # # # # # # # #         t48 = "🎯" if h48 < 200 else "❌"
# # # # # # # # #         t72 = "🎯" if h72 < 300 else "❌"

# # # # # # # # #         print(f"  [FAST ep{epoch}]"
# # # # # # # # #               f"  ADE={m_fast['ADE']:.1f}"
# # # # # # # # #               f"  6h={h6:.0f}{t6}"
# # # # # # # # #               f"  12h={h12:.0f}{t12}"
# # # # # # # # #               f"  24h={h24:.0f}{t24}"
# # # # # # # # #               f"  48h={h48:.0f}{t48}"
# # # # # # # # #               f"  72h={h72:.0f}{t72}"
# # # # # # # # #               f"  score={fast_score:.1f}")

# # # # # # # # #         # Full val
# # # # # # # # #         if epoch % args.val_ade_freq == 0:
# # # # # # # # #             # FIX: dùng _get_ema_obj() để lấy đúng EMA object sau compile
# # # # # # # # #             ema_obj = _get_ema_obj(model)

# # # # # # # # #             try:
# # # # # # # # #                 # RAW eval
# # # # # # # # #                 r_raw = evaluate_full_val_ade(
# # # # # # # # #                     model, val_loader, device,
# # # # # # # # #                     ode_steps=args.ode_steps_train, pred_len=args.pred_len,
# # # # # # # # #                     fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # # # # # #                     epoch=epoch, use_ema=False, ema_obj=None)

# # # # # # # # #                 # EMA eval (nếu có và qua warmup)
# # # # # # # # #                 r_use = r_raw
# # # # # # # # #                 if ema_obj is not None and epoch >= 10:
# # # # # # # # #                     r_ema = evaluate_full_val_ade(
# # # # # # # # #                         model, val_loader, device,
# # # # # # # # #                         ode_steps=args.ode_steps_train, pred_len=args.pred_len,
# # # # # # # # #                         fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # # # # # #                         epoch=epoch, use_ema=True, ema_obj=ema_obj)
# # # # # # # # #                     if _composite_score(r_ema) < _composite_score(r_raw):
# # # # # # # # #                         r_use = r_ema

# # # # # # # # #                 saver.update(r_use, model, args.output_dir, epoch,
# # # # # # # # #                              optimizer, avg_t, avg_vl,
# # # # # # # # #                              min_epochs=args.min_epochs)
# # # # # # # # #             except Exception as e:
# # # # # # # # #                 print(f"  ⚠  Full val failed: {e}")
# # # # # # # # #                 import traceback; traceback.print_exc()

# # # # # # # # #         # Checkpoint
# # # # # # # # #         if epoch % 5 == 0 or epoch == args.num_epochs - 1:
# # # # # # # # #             m = model._orig_mod if hasattr(model, '_orig_mod') else model
# # # # # # # # #             torch.save({
# # # # # # # # #                 "epoch": epoch,
# # # # # # # # #                 "model_state_dict": m.state_dict(),
# # # # # # # # #                 "optimizer_state": optimizer.state_dict(),
# # # # # # # # #                 "train_loss": avg_t, "val_loss": avg_vl,
# # # # # # # # #                 "ade": m_fast.get("ADE", float("nan")),
# # # # # # # # #                 "h48": h48, "h72": h72,
# # # # # # # # #             }, os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))

# # # # # # # # #         if epoch % 10 == 9:
# # # # # # # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # # # # # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # # # # # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # # # # # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")

# # # # # # # # #         if saver.early_stop:
# # # # # # # # #             print(f"  ⛔ Early stopping at epoch {epoch}")
# # # # # # # # #             break

# # # # # # # # #     # ── Final: evaluate SWA weights ───────────────────────────────────────
# # # # # # # # #     print(f"\n{'='*70}")
# # # # # # # # #     print("  Evaluating SWA weights...")
# # # # # # # # #     swa_backup = swa.apply_to(model)
# # # # # # # # #     if swa_backup is not None:
# # # # # # # # #         r_swa = evaluate_full_val_ade(
# # # # # # # # #             model, val_loader, device,
# # # # # # # # #             ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # # # # # # #             fast_ensemble=args.val_ensemble, metrics_csv=metrics_csv,
# # # # # # # # #             epoch=9999, use_ema=False)
# # # # # # # # #         if _composite_score(r_swa) < saver.best_score:
# # # # # # # # #             print(f"  ✅ SWA improved: {saver.best_score:.1f} → {_composite_score(r_swa):.1f}")
# # # # # # # # #             m = model._orig_mod if hasattr(model, '_orig_mod') else model
# # # # # # # # #             torch.save(dict(
# # # # # # # # #                 epoch=9999, model_state_dict=m.state_dict(),
# # # # # # # # #                 ade=r_swa.get("ADE", float("nan")),
# # # # # # # # #                 h72=r_swa.get("72h", float("nan")),
# # # # # # # # #                 tag="best_swa", model_version="v34fix-swa",
# # # # # # # # #             ), os.path.join(args.output_dir, "best_swa.pth"))
# # # # # # # # #         swa.restore(model, swa_backup)

# # # # # # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # # # # # #     print(f"\n{'='*70}")
# # # # # # # # #     print(f"  Best COMPOSITE : {saver.best_score:.1f}")
# # # # # # # # #     print(f"  Best ADE  : {saver.best_ade:.1f} km")
# # # # # # # # #     print(f"  Best 12h  : {saver.best_12h:.1f} km")
# # # # # # # # #     print(f"  Best 24h  : {saver.best_24h:.1f} km")
# # # # # # # # #     print(f"  Best 48h  : {saver.best_48h:.1f} km")
# # # # # # # # #     print(f"  Best 72h  : {saver.best_72h:.1f} km")
# # # # # # # # #     print(f"  Training  : {total_h:.2f}h")
# # # # # # # # #     print("=" * 70)


# # # # # # # # # if __name__ == "__main__":
# # # # # # # # #     args = get_args()
# # # # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # # # #     if torch.cuda.is_available():
# # # # # # # # #         torch.cuda.manual_seed_all(42)
# # # # # # # # #     main(args)

# # # # # # # # """
# # # # # # # # train_flowmatching_v34fix_resumable.py

# # # # # # # # THAY ĐỔI SO VỚI v34fix GỐC:
# # # # # # # #   1. Checkpoint lưu đầy đủ: scheduler_state, ema_shadow, saver state
# # # # # # # #   2. Resume load đúng thứ tự: weights → EMA → compile → optimizer → scheduler
# # # # # # # #   3. Không còn vòng lặp advance scheduler O(N*steps) khi resume
# # # # # # # #   4. EMA shadow restore trước compile để key names khớp

# # # # # # # # CÁCH DÙNG:
# # # # # # # #   # Lần đầu train từ đầu:
# # # # # # # #   python train_v34fix_resumable.py --output_dir runs/v34fix [args...]

# # # # # # # #   # Resume từ checkpoint:
# # # # # # # #   python train_v34fix_resumable.py \
# # # # # # # #     --resume /kaggle/input/my-checkpoint/ckpt_ep037.pth \
# # # # # # # #     --resume_epoch 38 \
# # # # # # # #     --output_dir runs/v34fix [args...]

# # # # # # # #   # Resume từ best_model:
# # # # # # # #   python train_v34fix_resumable.py \
# # # # # # # #     --resume runs/v34fix/best_model.pth \
# # # # # # # #     --resume_epoch 38 \
# # # # # # # #     --output_dir runs/v34fix [args...]
# # # # # # # # """
# # # # # # # # from __future__ import annotations

# # # # # # # # import sys
# # # # # # # # import os
# # # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # # import argparse
# # # # # # # # import time
# # # # # # # # import math
# # # # # # # # import random
# # # # # # # # import copy
# # # # # # # # from train_patch import (
# # # # # # # #       _composite_score,
# # # # # # # #       evaluate_fast_v2 as evaluate_fast,
# # # # # # # #       evaluate_full_val_ade_v2 as evaluate_full_val_ade,
# # # # # # # #       log_fast_eval,
# # # # # # # #   )

# # # # # # # # import numpy as np
# # # # # # # # import torch
# # # # # # # # import torch.optim as optim
# # # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # # from Model.data.loader_training import data_loader
# # # # # # # # from Model.flow_matching_model import TCFlowMatching
# # # # # # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # # from utils.metrics import (
# # # # # # # #     TCEvaluator, StepErrorAccumulator,
# # # # # # # #     save_metrics_csv, haversine_km_torch,
# # # # # # # #     denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
# # # # # # # # )


# # # # # # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # # # # # def move(batch, device):
# # # # # # # #     out = list(batch)
# # # # # # # #     for i, x in enumerate(out):
# # # # # # # #         if torch.is_tensor(x):
# # # # # # # #             out[i] = x.to(device)
# # # # # # # #         elif isinstance(x, dict):
# # # # # # # #             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
# # # # # # # #                       for k, v in x.items()}
# # # # # # # #     return out


# # # # # # # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # # # # # # #                             collate_fn, num_workers):
# # # # # # # #     n   = len(val_dataset)
# # # # # # # #     rng = random.Random(42)
# # # # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # # # #     return DataLoader(Subset(val_dataset, idx),
# # # # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # # # def _composite_score(result):
# # # # # # # #     ade = result.get("ADE", float("inf"))
# # # # # # # #     h12 = result.get("12h", float("inf"))
# # # # # # # #     h24 = result.get("24h", float("inf"))
# # # # # # # #     h48 = result.get("48h", float("inf"))
# # # # # # # #     h72 = result.get("72h", float("inf"))
# # # # # # # #     return 0.10 * ade + 0.10 * h12 + 0.15 * h24 + 0.25 * h48 + 0.40 * h72


# # # # # # # # def _get_ema_obj(model):
# # # # # # # #     """Lấy EMAModel object, xử lý cả compiled và raw model."""
# # # # # # # #     if hasattr(model, '_ema') and model._ema is not None:
# # # # # # # #         return model._ema
# # # # # # # #     if hasattr(model, '_orig_mod'):
# # # # # # # #         orig = model._orig_mod
# # # # # # # #         if hasattr(orig, '_ema') and orig._ema is not None:
# # # # # # # #             return orig._ema
# # # # # # # #     return None


# # # # # # # # def _call_ema_update(model):
# # # # # # # #     """Gọi ema_update() đúng cách sau torch.compile."""
# # # # # # # #     if hasattr(model, '_orig_mod'):
# # # # # # # #         orig = model._orig_mod
# # # # # # # #         if hasattr(orig, 'ema_update'):
# # # # # # # #             orig.ema_update()
# # # # # # # #             return
# # # # # # # #     if hasattr(model, 'ema_update'):
# # # # # # # #         model.ema_update()


# # # # # # # # def _get_raw_model(model):
# # # # # # # #     """Unwrap compiled model để lấy raw model."""
# # # # # # # #     return model._orig_mod if hasattr(model, '_orig_mod') else model


# # # # # # # # def _save_checkpoint(path, epoch, model, optimizer, scheduler,
# # # # # # # #                      saver, avg_t, avg_vl, metrics=None):
# # # # # # # #     """
# # # # # # # #     Lưu checkpoint đầy đủ bao gồm scheduler và EMA.
# # # # # # # #     Dùng hàm này ở MỌI nơi lưu checkpoint để đảm bảo nhất quán.
# # # # # # # #     """
# # # # # # # #     m = _get_raw_model(model)

# # # # # # # #     # Lưu EMA shadow nếu có
# # # # # # # #     ema_obj = _get_ema_obj(model)
# # # # # # # #     ema_sd  = None
# # # # # # # #     if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # # # # # # #         try:
# # # # # # # #             ema_sd = {k: v.cpu().clone() for k, v in ema_obj.shadow.items()}
# # # # # # # #         except Exception:
# # # # # # # #             ema_sd = None

# # # # # # # #     payload = {
# # # # # # # #         "epoch"            : epoch,
# # # # # # # #         "model_state_dict" : m.state_dict(),
# # # # # # # #         "optimizer_state"  : optimizer.state_dict(),
# # # # # # # #         "scheduler_state"  : scheduler.state_dict(),
# # # # # # # #         "ema_shadow"       : ema_sd,
# # # # # # # #         # Saver state để resume patience đúng
# # # # # # # #         "best_score"       : saver.best_score,
# # # # # # # #         "best_ade"         : saver.best_ade,
# # # # # # # #         "best_72h"         : saver.best_72h,
# # # # # # # #         "best_48h"         : saver.best_48h,
# # # # # # # #         "best_24h"         : saver.best_24h,
# # # # # # # #         "best_12h"         : saver.best_12h,
# # # # # # # #         "train_loss"       : avg_t,
# # # # # # # #         "val_loss"         : avg_vl,
# # # # # # # #     }
# # # # # # # #     if metrics:
# # # # # # # #         payload.update(metrics)

# # # # # # # #     torch.save(payload, path)


# # # # # # # # def _load_checkpoint(path, model, optimizer, scheduler, saver, device):
# # # # # # # #     """
# # # # # # # #     Load checkpoint và restore tất cả state.
# # # # # # # #     Phải gọi SAU khi tất cả objects đã được tạo nhưng TRƯỚC torch.compile.
# # # # # # # #     Trả về start_epoch.
# # # # # # # #     """
# # # # # # # #     if path is None or not os.path.exists(path):
# # # # # # # #         if path is not None:
# # # # # # # #             print(f"  ⚠  Checkpoint không tìm thấy: {path}")
# # # # # # # #         return 0

# # # # # # # #     print(f"  Loading checkpoint: {path}")
# # # # # # # #     ckpt = torch.load(path, map_location=device)

# # # # # # # #     # 1. Model weights — dùng raw model (chưa compile)
# # # # # # # #     m = _get_raw_model(model)
# # # # # # # #     missing, unexpected = m.load_state_dict(
# # # # # # # #         ckpt["model_state_dict"], strict=False)
# # # # # # # #     if missing:
# # # # # # # #         print(f"  ⚠  Missing keys ({len(missing)}): "
# # # # # # # #               f"{missing[:3]}{'...' if len(missing) > 3 else ''}")

# # # # # # # #     # 2. EMA shadow — phải restore TRƯỚC compile
# # # # # # # #     if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
# # # # # # # #         ema_obj = _get_ema_obj(model)
# # # # # # # #         if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # # # # # # #             restored = 0
# # # # # # # #             for k, v in ckpt["ema_shadow"].items():
# # # # # # # #                 if k in ema_obj.shadow:
# # # # # # # #                     ema_obj.shadow[k].copy_(v.to(device))
# # # # # # # #                     restored += 1
# # # # # # # #             print(f"  EMA shadow restored ({restored} keys)")

# # # # # # # #     # 3. Optimizer — load state, fix device
# # # # # # # #     if "optimizer_state" in ckpt:
# # # # # # # #         try:
# # # # # # # #             optimizer.load_state_dict(ckpt["optimizer_state"])
# # # # # # # #             for state in optimizer.state.values():
# # # # # # # #                 for k, v in state.items():
# # # # # # # #                     if torch.is_tensor(v):
# # # # # # # #                         state[k] = v.to(device)
# # # # # # # #             print("  Optimizer state restored")
# # # # # # # #         except Exception as e:
# # # # # # # #             print(f"  ⚠  Optimizer restore failed: {e} (using fresh)")

# # # # # # # #     # 4. Scheduler — dùng state_dict nếu có, không thì advance manually
# # # # # # # #     if "scheduler_state" in ckpt:
# # # # # # # #         try:
# # # # # # # #             scheduler.load_state_dict(ckpt["scheduler_state"])
# # # # # # # #             print(f"  Scheduler restored  "
# # # # # # # #                   f"(last_epoch={scheduler.last_epoch})")
# # # # # # # #         except Exception as e:
# # # # # # # #             print(f"  ⚠  Scheduler restore failed: {e}")
# # # # # # # #     # Nếu không có scheduler_state, caller sẽ advance manually

# # # # # # # #     # 5. Saver state — để patience tiếp tục đúng
# # # # # # # #     if "best_score" in ckpt:
# # # # # # # #         saver.best_score = ckpt["best_score"]
# # # # # # # #         saver.best_ade   = ckpt.get("best_ade",  float("inf"))
# # # # # # # #         saver.best_72h   = ckpt.get("best_72h",  float("inf"))
# # # # # # # #         saver.best_48h   = ckpt.get("best_48h",  float("inf"))
# # # # # # # #         saver.best_24h   = ckpt.get("best_24h",  float("inf"))
# # # # # # # #         saver.best_12h   = ckpt.get("best_12h",  float("inf"))
# # # # # # # #         print(f"  Saver state restored  (best_score={saver.best_score:.1f})")

# # # # # # # #     start_epoch = ckpt.get("epoch", 0) + 1
# # # # # # # #     print(f"  → Resuming from epoch {start_epoch}")
# # # # # # # #     return start_epoch


# # # # # # # # # ── SWA ───────────────────────────────────────────────────────────────────────

# # # # # # # # class SWAManager:
# # # # # # # #     def __init__(self, model, start_epoch=50):
# # # # # # # #         self.start_epoch = start_epoch
# # # # # # # #         self.n_averaged  = 0
# # # # # # # #         self.avg_state   = None

# # # # # # # #     def update(self, model, epoch):
# # # # # # # #         if epoch < self.start_epoch:
# # # # # # # #             return
# # # # # # # #         m  = _get_raw_model(model)
# # # # # # # #         sd = {k: v.detach().clone() for k, v in m.state_dict().items()
# # # # # # # #               if v.dtype.is_floating_point}
# # # # # # # #         if self.avg_state is None:
# # # # # # # #             self.avg_state = sd
# # # # # # # #             self.n_averaged = 1
# # # # # # # #         else:
# # # # # # # #             n = self.n_averaged
# # # # # # # #             for k in self.avg_state:
# # # # # # # #                 if k in sd:
# # # # # # # #                     self.avg_state[k].mul_(n / (n + 1)).add_(sd[k], alpha=1.0 / (n + 1))
# # # # # # # #             self.n_averaged += 1

# # # # # # # #     def apply_to(self, model):
# # # # # # # #         if self.avg_state is None:
# # # # # # # #             return None
# # # # # # # #         m  = _get_raw_model(model)
# # # # # # # #         sd = m.state_dict()
# # # # # # # #         backup = {k: sd[k].detach().clone() for k in self.avg_state if k in sd}
# # # # # # # #         for k, v in self.avg_state.items():
# # # # # # # #             if k in sd:
# # # # # # # #                 sd[k].copy_(v)
# # # # # # # #         return backup

# # # # # # # #     def restore(self, model, backup):
# # # # # # # #         if backup is None:
# # # # # # # #             return
# # # # # # # #         m  = _get_raw_model(model)
# # # # # # # #         sd = m.state_dict()
# # # # # # # #         for k, v in backup.items():
# # # # # # # #             if k in sd:
# # # # # # # #                 sd[k].copy_(v)


# # # # # # # # # ── Evaluation ────────────────────────────────────────────────────────────────

# # # # # # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=15):
# # # # # # # #     model.eval()
# # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # #     t0  = time.perf_counter()
# # # # # # # #     n   = 0
# # # # # # # #     spread_per_step = []

# # # # # # # #     with torch.no_grad():
# # # # # # # #         for batch in loader:
# # # # # # # #             bl = move(list(batch), device)
# # # # # # # #             pred, _, all_trajs = model.sample(
# # # # # # # #                 bl, num_ensemble=fast_ensemble, ddim_steps=ode_steps,
# # # # # # # #                 importance_weight=True)
# # # # # # # #             T_active  = pred.shape[0]
# # # # # # # #             gt_sliced = bl[1][:T_active]
# # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
# # # # # # # #             acc.update(dist)

# # # # # # # #             step_spreads = []
# # # # # # # #             for t in range(all_trajs.shape[1]):
# # # # # # # #                 step_data = all_trajs[:, t, :, :]
# # # # # # # #                 std_lon   = step_data[:, :, 0].std(0)
# # # # # # # #                 std_lat   = step_data[:, :, 1].std(0)
# # # # # # # #                 spread    = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# # # # # # # #                 step_spreads.append(spread)
# # # # # # # #             spread_per_step.append(step_spreads)
# # # # # # # #             n += 1

# # # # # # # #     r = acc.compute()
# # # # # # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# # # # # # # #     if spread_per_step:
# # # # # # # #         spreads = np.array(spread_per_step)
# # # # # # # #         r["spread_12h_km"] = float(spreads[:, 1].mean()) if spreads.shape[1] > 1 else 0.0
# # # # # # # #         r["spread_24h_km"] = float(spreads[:, 3].mean()) if spreads.shape[1] > 3 else 0.0
# # # # # # # #         r["spread_72h_km"] = float(spreads[:, -1].mean())
# # # # # # # #     return r


# # # # # # # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # # # # # # #                            fast_ensemble, metrics_csv, epoch,
# # # # # # # #                            use_ema=False, ema_obj=None):
# # # # # # # #     backup = None
# # # # # # # #     if use_ema and ema_obj is not None:
# # # # # # # #         try:
# # # # # # # #             backup = ema_obj.apply_to(model)
# # # # # # # #         except Exception as e:
# # # # # # # #             print(f"  ⚠  EMA apply_to failed: {e} — skipping EMA eval")
# # # # # # # #             backup  = None
# # # # # # # #             use_ema = False

# # # # # # # #     model.eval()
# # # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # # #     t0  = time.perf_counter()
# # # # # # # #     n   = 0

# # # # # # # #     with torch.no_grad():
# # # # # # # #         for batch in val_loader:
# # # # # # # #             bl = move(list(batch), device)
# # # # # # # #             pred, _, _ = model.sample(
# # # # # # # #                 bl, num_ensemble=max(fast_ensemble, 20),
# # # # # # # #                 ddim_steps=max(ode_steps, 30),
# # # # # # # #                 importance_weight=True)
# # # # # # # #             T_pred = pred.shape[0]
# # # # # # # #             gt = bl[1][:T_pred]
# # # # # # # #             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt))
# # # # # # # #             acc.update(dist)
# # # # # # # #             n += 1

# # # # # # # #     r       = acc.compute()
# # # # # # # #     elapsed = time.perf_counter() - t0
# # # # # # # #     score   = _composite_score(r)
# # # # # # # #     tag     = "EMA" if use_ema else "RAW"

# # # # # # # #     print(f"\n{'='*64}")
# # # # # # # #     print(f"  [FULL VAL ADE ({tag})  ep={epoch}  {elapsed:.0f}s  {n} batches]")
# # # # # # # #     print(f"  ADE = {r.get('ADE', float('nan')):.1f} km  "
# # # # # # # #           f"FDE = {r.get('FDE', float('nan')):.1f} km")
# # # # # # # #     print(f"  6h={r.get('6h',  float('nan')):.0f}  "
# # # # # # # #           f"12h={r.get('12h', float('nan')):.0f}  "
# # # # # # # #           f"24h={r.get('24h', float('nan')):.0f}  "
# # # # # # # #           f"48h={r.get('48h', float('nan')):.0f}  "
# # # # # # # #           f"72h={r.get('72h', float('nan')):.0f} km")
# # # # # # # #     print(f"  Composite score = {score:.1f}")
# # # # # # # #     print(f"{'='*64}\n")

# # # # # # # #     from datetime import datetime
# # # # # # # #     from utils.metrics import DatasetMetrics, save_metrics_csv as _save_csv
# # # # # # # #     dm = DatasetMetrics(
# # # # # # # #         ade      = r.get("ADE",  float("nan")),
# # # # # # # #         fde      = r.get("FDE",  float("nan")),
# # # # # # # #         ugde_12h = r.get("12h",  float("nan")),
# # # # # # # #         ugde_24h = r.get("24h",  float("nan")),
# # # # # # # #         ugde_48h = r.get("48h",  float("nan")),
# # # # # # # #         ugde_72h = r.get("72h",  float("nan")),
# # # # # # # #         n_total  = r.get("n_samples", 0),
# # # # # # # #         timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # # # #     )
# # # # # # # #     _save_csv(dm, metrics_csv, tag=f"val_full_{tag}_ep{epoch:03d}")

# # # # # # # #     if backup is not None:
# # # # # # # #         try:
# # # # # # # #             ema_obj.restore(model, backup)
# # # # # # # #         except Exception as e:
# # # # # # # #             print(f"  ⚠  EMA restore failed: {e}")

# # # # # # # #     return r


# # # # # # # # # ── BestModelSaver ────────────────────────────────────────────────────────────

# # # # # # # # class HorizonAwareBestSaver:
# # # # # # # #     def __init__(self, patience=30, tol=1.5):
# # # # # # # #         self.patience   = patience
# # # # # # # #         self.tol        = tol
# # # # # # # #         self.counter    = 0
# # # # # # # #         self.early_stop = False
# # # # # # # #         self.best_score = float("inf")
# # # # # # # #         self.best_ade   = float("inf")
# # # # # # # #         self.best_12h   = float("inf")
# # # # # # # #         self.best_24h   = float("inf")
# # # # # # # #         self.best_48h   = float("inf")
# # # # # # # #         self.best_72h   = float("inf")

# # # # # # # #     def update(self, r, model, out_dir, epoch, optimizer, scheduler,
# # # # # # # #                tl, vl, saver_ref, min_epochs=50):
# # # # # # # #         ade   = r.get("ADE", float("inf"))
# # # # # # # #         h12   = r.get("12h", float("inf"))
# # # # # # # #         h24   = r.get("24h", float("inf"))
# # # # # # # #         h48   = r.get("48h", float("inf"))
# # # # # # # #         h72   = r.get("72h", float("inf"))
# # # # # # # #         score = _composite_score(r)

# # # # # # # #         improved_any = False

# # # # # # # #         if ade < self.best_ade:
# # # # # # # #             self.best_ade = ade;  improved_any = True
# # # # # # # #             _save_checkpoint(
# # # # # # # #                 os.path.join(out_dir, "best_ade.pth"),
# # # # # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # # # # #                 {"ade": ade, "h12": h12, "h48": h48, "h72": h72, "tag": "best_ade"})
# # # # # # # #         if h72 < self.best_72h:
# # # # # # # #             self.best_72h = h72;  improved_any = True
# # # # # # # #             _save_checkpoint(
# # # # # # # #                 os.path.join(out_dir, "best_72h.pth"),
# # # # # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # # # # #                 {"h72": h72, "tag": "best_72h"})
# # # # # # # #         if h48 < self.best_48h: self.best_48h = h48; improved_any = True
# # # # # # # #         if h24 < self.best_24h: self.best_24h = h24; improved_any = True
# # # # # # # #         if h12 < self.best_12h: self.best_12h = h12; improved_any = True

# # # # # # # #         if score < self.best_score - self.tol:
# # # # # # # #             self.best_score = score
# # # # # # # #             self.counter    = 0
# # # # # # # #             _save_checkpoint(
# # # # # # # #                 os.path.join(out_dir, "best_model.pth"),
# # # # # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # # # # #                 {"ade": ade, "h12": h12, "h24": h24,
# # # # # # # #                  "h48": h48, "h72": h72,
# # # # # # # #                  "composite_score": score, "tag": "best_composite"})
# # # # # # # #             print(f"  ✅ Best COMPOSITE={score:.1f}  "
# # # # # # # #                   f"ADE={ade:.1f}  12h={h12:.0f}  24h={h24:.0f}  "
# # # # # # # #                   f"48h={h48:.0f}  72h={h72:.0f}  (ep {epoch})")
# # # # # # # #         else:
# # # # # # # #             if not improved_any:
# # # # # # # #                 self.counter += 1
# # # # # # # #             print(f"  No improvement {self.counter}/{self.patience}"
# # # # # # # #                   f"  (best_score={self.best_score:.1f}, cur={score:.1f})")

# # # # # # # #         if epoch >= min_epochs and self.counter >= self.patience:
# # # # # # # #             self.early_stop = True


# # # # # # # # # ── Env diagnostic ────────────────────────────────────────────────────────────

# # # # # # # # def _check_env(bl, train_dataset):
# # # # # # # #     try:    env_dir = train_dataset.env_path
# # # # # # # #     except AttributeError:
# # # # # # # #         try:    env_dir = train_dataset.dataset.env_path
# # # # # # # #         except: env_dir = "UNKNOWN"
# # # # # # # #     print(f"  Env path: {env_dir}")
# # # # # # # #     env_data = bl[13]
# # # # # # # #     if env_data is None:
# # # # # # # #         print("  ⚠️  env_data is None"); return
# # # # # # # #     for key in ("gph500_mean", "gph500_center", "u500_mean", "v500_mean"):
# # # # # # # #         if key not in env_data:
# # # # # # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # # # # # #         v    = env_data[key]
# # # # # # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # # # # # #         print(f"  {'✅' if zero < 80 else '⚠️'}  {key}: "
# # # # # # # #               f"mean={v.mean().item():.4f} std={v.std().item():.4f} zero={zero:.1f}%")


# # # # # # # # # ── Args ──────────────────────────────────────────────────────────────────────

# # # # # # # # def get_args():
# # # # # # # #     p = argparse.ArgumentParser(
# # # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # # # # # # #     p.add_argument("--obs_len",         default=8,          type=int)
# # # # # # # #     p.add_argument("--pred_len",        default=12,         type=int)
# # # # # # # #     p.add_argument("--batch_size",      default=32,         type=int)
# # # # # # # #     p.add_argument("--num_epochs",      default=120,        type=int)
# # # # # # # #     p.add_argument("--g_learning_rate", default=3e-4,       type=float)
# # # # # # # #     p.add_argument("--weight_decay",    default=1e-4,       type=float)
# # # # # # # #     p.add_argument("--warmup_epochs",   default=5,          type=int)
# # # # # # # #     p.add_argument("--grad_clip",       default=1.0,        type=float)
# # # # # # # #     p.add_argument("--patience",        default=30,         type=int)
# # # # # # # #     p.add_argument("--min_epochs",      default=50,         type=int)
# # # # # # # #     p.add_argument("--n_train_ens",     default=4,          type=int)
# # # # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # # # #     p.add_argument("--num_workers",     default=2,          type=int)

# # # # # # # #     p.add_argument("--sigma_min",            default=0.02,  type=float)
# # # # # # # #     p.add_argument("--ctx_noise_scale",      default=0.01,  type=float)
# # # # # # # #     p.add_argument("--initial_sample_sigma", default=0.03,  type=float)

# # # # # # # #     p.add_argument("--ode_steps_train", default=25,  type=int)
# # # # # # # #     p.add_argument("--ode_steps_val",   default=30,  type=int)
# # # # # # # #     p.add_argument("--ode_steps_test",  default=50,  type=int)

# # # # # # # #     p.add_argument("--val_ensemble",    default=30,  type=int)
# # # # # # # #     p.add_argument("--fast_ensemble",   default=15,  type=int)

# # # # # # # #     p.add_argument("--val_freq",        default=1,   type=int)
# # # # # # # #     p.add_argument("--val_ade_freq",    default=1,   type=int)
# # # # # # # #     p.add_argument("--val_subset_size", default=600, type=int)

# # # # # # # #     p.add_argument("--use_ema",         action="store_true", default=True)
# # # # # # # #     p.add_argument("--ema_decay",       default=0.995, type=float)
# # # # # # # #     p.add_argument("--swa_start_epoch", default=55,   type=int)
# # # # # # # #     p.add_argument("--teacher_forcing", action="store_true", default=True)

# # # # # # # #     p.add_argument("--output_dir",  default="runs/v34fix",    type=str)
# # # # # # # #     p.add_argument("--metrics_csv", default="metrics.csv",    type=str)
# # # # # # # #     p.add_argument("--predict_csv", default="predictions.csv", type=str)

# # # # # # # #     p.add_argument("--gpu_num",     default="0", type=str)
# # # # # # # #     p.add_argument("--delim",       default=" ")
# # # # # # # #     p.add_argument("--skip",        default=1,   type=int)
# # # # # # # #     p.add_argument("--min_ped",     default=1,   type=int)
# # # # # # # #     p.add_argument("--threshold",   default=0.002, type=float)
# # # # # # # #     p.add_argument("--other_modal", default="gph")
# # # # # # # #     p.add_argument("--test_year",   default=None, type=int)

# # # # # # # #     # Resume args
# # # # # # # #     p.add_argument("--resume",
# # # # # # # #                    default=None, type=str,
# # # # # # # #                    help="Path to checkpoint .pth (ckpt_epXXX.pth hoặc best_model.pth)")
# # # # # # # #     p.add_argument("--resume_epoch",
# # # # # # # #                    default=None, type=int,
# # # # # # # #                    help="Epoch bắt đầu train (mặc định: tự lấy từ checkpoint)")

# # # # # # # #     return p.parse_args()


# # # # # # # # # ── MAIN ──────────────────────────────────────────────────────────────────────

# # # # # # # # def main(args):
# # # # # # # #     if torch.cuda.is_available():
# # # # # # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # # # # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # # # # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)

# # # # # # # #     print("=" * 70)
# # # # # # # #     print("  TC-FlowMatching v34fix  |  HORIZON-AWARE + 72h FOCUSED + EMA FIX")
# # # # # # # #     print("  TARGETS: 12h<50  24h<100  48h<200  72h<300 km")
# # # # # # # #     print("=" * 70)

# # # # # # # #     # ── Data ──────────────────────────────────────────────────────────────
# # # # # # # #     train_dataset, train_loader = data_loader(
# # # # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # # # #     val_dataset, val_loader = data_loader(
# # # # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # # #     val_subset_loader = make_val_subset_loader(
# # # # # # # #         val_dataset, args.val_subset_size, args.batch_size,
# # # # # # # #         seq_collate, args.num_workers)

# # # # # # # #     test_loader = None
# # # # # # # #     try:
# # # # # # # #         _, test_loader = data_loader(
# # # # # # # #             args, {"root": args.dataset_root, "type": "test"},
# # # # # # # #             test=True, test_year=None)
# # # # # # # #     except Exception as e:
# # # # # # # #         print(f"  Warning: test loader: {e}")

# # # # # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # # # # #     print(f"  val   : {len(val_dataset)} seq")

# # # # # # # #     # ── Model ─────────────────────────────────────────────────────────────
# # # # # # # #     model = TCFlowMatching(
# # # # # # # #         pred_len             = args.pred_len,
# # # # # # # #         obs_len              = args.obs_len,
# # # # # # # #         sigma_min            = args.sigma_min,
# # # # # # # #         n_train_ens          = args.n_train_ens,
# # # # # # # #         ctx_noise_scale      = args.ctx_noise_scale,
# # # # # # # #         initial_sample_sigma = args.initial_sample_sigma,
# # # # # # # #         teacher_forcing      = args.teacher_forcing,
# # # # # # # #         use_ema              = args.use_ema,
# # # # # # # #         ema_decay            = args.ema_decay,
# # # # # # # #     ).to(device)

# # # # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # # #     print(f"  params  : {n_params:,}")

# # # # # # # #     # EMA phải init TRƯỚC compile
# # # # # # # #     model.init_ema()
# # # # # # # #     print(f"  EMA     : {'ON' if model._ema is not None else 'OFF'}"
# # # # # # # #           f"  (decay={args.ema_decay})")

# # # # # # # #     # ── Optimizer + Scheduler (tạo trước khi load state) ─────────────────
# # # # # # # #     optimizer = optim.AdamW(
# # # # # # # #         model.parameters(),
# # # # # # # #         lr=args.g_learning_rate,
# # # # # # # #         weight_decay=args.weight_decay,
# # # # # # # #     )

# # # # # # # #     saver  = HorizonAwareBestSaver(patience=args.patience, tol=1.5)
# # # # # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # # # # # #     swa    = SWAManager(model, start_epoch=args.swa_start_epoch)

# # # # # # # #     # Tạo scheduler tạm (sẽ được overwrite bởi load_state_dict)
# # # # # # # #     steps_per_epoch = len(train_loader)
# # # # # # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # # # # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # # # # # #     scheduler = get_cosine_schedule_with_warmup(
# # # # # # # #         optimizer, warmup, total_steps, min_lr=1e-6)

# # # # # # # #     # ══════════════════════════════════════════════════════════════════════
# # # # # # # #     # RESUME — phải xảy ra TRƯỚC torch.compile
# # # # # # # #     # Thứ tự: model weights → EMA → optimizer → scheduler → saver
# # # # # # # #     # ══════════════════════════════════════════════════════════════════════
# # # # # # # #     has_scheduler_state = False
# # # # # # # #     start_epoch = 0

# # # # # # # #     if args.resume is not None and os.path.exists(args.resume):
# # # # # # # #         print(f"  Loading checkpoint: {args.resume}")
# # # # # # # #         ckpt = torch.load(args.resume, map_location=device)

# # # # # # # #         # 1. Model weights (raw model, chưa compile)
# # # # # # # #         m = _get_raw_model(model)
# # # # # # # #         missing, _ = m.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # # # # #         if missing:
# # # # # # # #             print(f"  ⚠  Missing keys ({len(missing)}): "
# # # # # # # #                   f"{missing[:3]}{'...' if len(missing)>3 else ''}")

# # # # # # # #         # 2. EMA shadow — TRƯỚC compile để key names khớp
# # # # # # # #         if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
# # # # # # # #             ema_obj = _get_ema_obj(model)
# # # # # # # #             if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # # # # # # #                 restored = sum(
# # # # # # # #                     1 for k, v in ckpt["ema_shadow"].items()
# # # # # # # #                     if k in ema_obj.shadow
# # # # # # # #                     and not ema_obj.shadow[k].copy_(v.to(device)) is None
# # # # # # # #                 )
# # # # # # # #                 print(f"  EMA shadow restored ({restored} keys)")

# # # # # # # #         # 3. Optimizer
# # # # # # # #         if "optimizer_state" in ckpt:
# # # # # # # #             try:
# # # # # # # #                 optimizer.load_state_dict(ckpt["optimizer_state"])
# # # # # # # #                 for state in optimizer.state.values():
# # # # # # # #                     for k, v in state.items():
# # # # # # # #                         if torch.is_tensor(v):
# # # # # # # #                             state[k] = v.to(device)
# # # # # # # #                 print("  Optimizer state restored")
# # # # # # # #             except Exception as e:
# # # # # # # #                 print(f"  ⚠  Optimizer restore failed: {e}")

# # # # # # # #         # 4. Scheduler (sẽ load state_dict sau khi scheduler được tạo lại)
# # # # # # # #         has_scheduler_state = "scheduler_state" in ckpt
# # # # # # # #         _ckpt_scheduler_state = ckpt.get("scheduler_state", None)

# # # # # # # #         # 5. Saver state
# # # # # # # #         if "best_score" in ckpt:
# # # # # # # #             saver.best_score = ckpt["best_score"]
# # # # # # # #             saver.best_ade   = ckpt.get("best_ade",  float("inf"))
# # # # # # # #             saver.best_72h   = ckpt.get("best_72h",  float("inf"))
# # # # # # # #             saver.best_48h   = ckpt.get("best_48h",  float("inf"))
# # # # # # # #             saver.best_24h   = ckpt.get("best_24h",  float("inf"))
# # # # # # # #             saver.best_12h   = ckpt.get("best_12h",  float("inf"))
# # # # # # # #             print(f"  Saver state restored (best_score={saver.best_score:.1f})")

# # # # # # # #         # 6. Xác định start_epoch
# # # # # # # #         if args.resume_epoch is not None:
# # # # # # # #             start_epoch = args.resume_epoch
# # # # # # # #         else:
# # # # # # # #             start_epoch = ckpt.get("epoch", 0) + 1
# # # # # # # #         print(f"  → Resuming from epoch {start_epoch}")

# # # # # # # #     elif args.resume is not None:
# # # # # # # #         print(f"  ⚠  Checkpoint không tìm thấy: {args.resume}, train từ đầu")

# # # # # # # #     # ── torch.compile SAU khi load weights và EMA ──────────────────────
# # # # # # # #     try:
# # # # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # # # #         print("  torch.compile: enabled")
# # # # # # # #     except Exception:
# # # # # # # #         pass

# # # # # # # #     # Load scheduler state sau compile (scheduler không bị compile ảnh hưởng)
# # # # # # # #     if has_scheduler_state and _ckpt_scheduler_state is not None:
# # # # # # # #         try:
# # # # # # # #             scheduler.load_state_dict(_ckpt_scheduler_state)
# # # # # # # #             print(f"  Scheduler restored (last_epoch={scheduler.last_epoch})")
# # # # # # # #         except Exception as e:
# # # # # # # #             print(f"  ⚠  Scheduler restore failed: {e} — advancing manually")
# # # # # # # #             for _ in range(start_epoch * steps_per_epoch):
# # # # # # # #                 scheduler.step()
# # # # # # # #     elif start_epoch > 0:
# # # # # # # #         # Checkpoint cũ không có scheduler_state → advance manually
# # # # # # # #         # Nhanh vì chỉ update internal counter, không chạy optimizer
# # # # # # # #         print(f"  Advancing scheduler {start_epoch * steps_per_epoch} steps...")
# # # # # # # #         for _ in range(start_epoch * steps_per_epoch):
# # # # # # # #             scheduler.step()

# # # # # # # #     print("=" * 70)
# # # # # # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch, start={start_epoch})")
# # # # # # # #     print("=" * 70)

# # # # # # # #     epoch_times = []
# # # # # # # #     train_start = time.perf_counter()

# # # # # # # #     for epoch in range(start_epoch, args.num_epochs):
# # # # # # # #         model.train()
# # # # # # # #         sum_loss = 0.0
# # # # # # # #         t0 = time.perf_counter()

# # # # # # # #         for i, batch in enumerate(train_loader):
# # # # # # # #             bl = move(list(batch), device)

# # # # # # # #             if epoch == start_epoch and i == 0:
# # # # # # # #                 _check_env(bl, train_dataset)

# # # # # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # #                 bd = model.get_loss_breakdown(bl, epoch=epoch)

# # # # # # # #             optimizer.zero_grad()
# # # # # # # #             scaler.scale(bd["total"]).backward()
# # # # # # # #             scaler.unscale_(optimizer)
# # # # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # # # # #             scaler.step(optimizer)
# # # # # # # #             scaler.update()
# # # # # # # #             scheduler.step()

# # # # # # # #             _call_ema_update(model)

# # # # # # # #             sum_loss += bd["total"].item()

# # # # # # # #             if i % 20 == 0:
# # # # # # # #                 lr = optimizer.param_groups[0]["lr"]
# # # # # # # #                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # # # # # #                       f"  tot={bd['total'].item():.3f}"
# # # # # # # #                       f"  fm={bd.get('fm_mse', 0):.3f}"
# # # # # # # #                       f"  hor={bd.get('mse_hav', 0):.3f}"
# # # # # # # #                       f"  ms={bd.get('multi_scale', 0):.3f}"
# # # # # # # #                       f"  end={bd.get('endpoint', 0):.3f}"
# # # # # # # #                       f"  shp={bd.get('shape', 0):.3f}"
# # # # # # # #                       f"  vel={bd.get('velocity', 0):.3f}"
# # # # # # # #                       f"  hd={bd.get('heading', 0):.3f}"
# # # # # # # #                       f"  str={bd.get('steering', 0):.3f}"
# # # # # # # #                       f"  σ={bd.get('sigma', 0):.3f}"
# # # # # # # #                       f"  lr={lr:.2e}")

# # # # # # # #         ep_s  = time.perf_counter() - t0
# # # # # # # #         epoch_times.append(ep_s)
# # # # # # # #         avg_t = sum_loss / len(train_loader)

# # # # # # # #         swa.update(model, epoch)

# # # # # # # #         # Val loss
# # # # # # # #         model.eval()
# # # # # # # #         val_loss = 0.0
# # # # # # # #         with torch.no_grad():
# # # # # # # #             for batch in val_loader:
# # # # # # # #                 bl_v = move(list(batch), device)
# # # # # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # # #                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
# # # # # # # #         avg_vl = val_loss / len(val_loader)

# # # # # # # #         print(f"  Epoch {epoch:>3}  train={avg_t:.4f}  val={avg_vl:.4f}"
# # # # # # # #               f"  time={ep_s:.0f}s")

# # # # # # # #         # Fast eval
# # # # # # # #         m_fast    = evaluate_fast(model, val_subset_loader, device,
# # # # # # # #                                    args.ode_steps_train, args.pred_len,
# # # # # # # #                                    args.fast_ensemble)
# # # # # # # #         h6  = m_fast.get("6h",  float("nan"))
# # # # # # # #         h12 = m_fast.get("12h", float("nan"))
# # # # # # # #         h24 = m_fast.get("24h", float("nan"))
# # # # # # # #         h48 = m_fast.get("48h", float("nan"))
# # # # # # # #         h72 = m_fast.get("72h", float("nan"))
# # # # # # # #         fast_score = _composite_score(m_fast)

# # # # # # # #         # print(f"  [FAST ep{epoch}]"
# # # # # # # #         #       f"  ADE={m_fast['ADE']:.1f}"
# # # # # # # #         #       f"  6h={h6:.0f}{'🎯' if h6<30 else '❌'}"
# # # # # # # #         #       f"  12h={h12:.0f}{'🎯' if h12<50 else '❌'}"
# # # # # # # #         #       f"  24h={h24:.0f}{'🎯' if h24<100 else '❌'}"
# # # # # # # #         #       f"  48h={h48:.0f}{'🎯' if h48<200 else '❌'}"
# # # # # # # #         #       f"  72h={h72:.0f}{'🎯' if h72<300 else '❌'}"
# # # # # # # #         #       f"  score={fast_score:.1f}")
# # # # # # # #         log_fast_eval(m_fast, epoch)
# # # # # # # #         # Full val eval
# # # # # # # #         if epoch % args.val_ade_freq == 0:
# # # # # # # #             ema_obj = _get_ema_obj(model)
# # # # # # # #             try:
# # # # # # # #                 r_raw = evaluate_full_val_ade(
# # # # # # # #                     model, val_loader, device,
# # # # # # # #                     ode_steps=args.ode_steps_train, pred_len=args.pred_len,
# # # # # # # #                     fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # # # # #                     epoch=epoch, use_ema=False, ema_obj=None)

# # # # # # # #                 r_use = r_raw
# # # # # # # #                 if ema_obj is not None and epoch >= 10:
# # # # # # # #                     r_ema = evaluate_full_val_ade(
# # # # # # # #                         model, val_loader, device,
# # # # # # # #                         ode_steps=args.ode_steps_train, pred_len=args.pred_len,
# # # # # # # #                         fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # # # # #                         epoch=epoch, use_ema=True, ema_obj=ema_obj)
# # # # # # # #                     if _composite_score(r_ema) < _composite_score(r_raw):
# # # # # # # #                         r_use = r_ema

# # # # # # # #                 saver.update(r_use, model, args.output_dir, epoch,
# # # # # # # #                              optimizer, scheduler, avg_t, avg_vl,
# # # # # # # #                              saver_ref=saver,
# # # # # # # #                              min_epochs=args.min_epochs)
# # # # # # # #             except Exception as e:
# # # # # # # #                 print(f"  ⚠  Full val failed: {e}")
# # # # # # # #                 import traceback; traceback.print_exc()

# # # # # # # #         # ── Checkpoint định kỳ (dùng _save_checkpoint để đầy đủ) ────────
# # # # # # # #         if epoch % 5 == 0 or epoch == args.num_epochs - 1:
# # # # # # # #             _save_checkpoint(
# # # # # # # #                 os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"),
# # # # # # # #                 epoch, model, optimizer, scheduler, saver, avg_t, avg_vl,
# # # # # # # #                 {"ade": m_fast.get("ADE", float("nan")),
# # # # # # # #                  "h48": h48, "h72": h72})

# # # # # # # #         if epoch % 10 == 9:
# # # # # # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # # # # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # # # # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # # # # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")

# # # # # # # #         if saver.early_stop:
# # # # # # # #             print(f"  ⛔ Early stopping at epoch {epoch}")
# # # # # # # #             break

# # # # # # # #     # ── Final: SWA eval ───────────────────────────────────────────────────
# # # # # # # #     print(f"\n{'='*70}")
# # # # # # # #     swa_backup = swa.apply_to(model)
# # # # # # # #     if swa_backup is not None:
# # # # # # # #         print("  Evaluating SWA weights...")
# # # # # # # #         r_swa = evaluate_full_val_ade(
# # # # # # # #             model, val_loader, device,
# # # # # # # #             ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # # # # # #             fast_ensemble=args.val_ensemble, metrics_csv=metrics_csv,
# # # # # # # #             epoch=9999, use_ema=False)
# # # # # # # #         if _composite_score(r_swa) < saver.best_score:
# # # # # # # #             _save_checkpoint(
# # # # # # # #                 os.path.join(args.output_dir, "best_swa.pth"),
# # # # # # # #                 9999, model, optimizer, scheduler, saver,
# # # # # # # #                 avg_t=0.0, avg_vl=0.0,
# # # # # # # #                 metrics={"tag": "best_swa"})
# # # # # # # #             print(f"  ✅ SWA checkpoint saved")
# # # # # # # #         swa.restore(model, swa_backup)

# # # # # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # # # # #     print(f"\n  Best score: {saver.best_score:.1f}  |  "
# # # # # # # #           f"ADE={saver.best_ade:.1f}  12h={saver.best_12h:.1f}  "
# # # # # # # #           f"24h={saver.best_24h:.1f}  48h={saver.best_48h:.1f}  "
# # # # # # # #           f"72h={saver.best_72h:.1f}  |  {total_h:.2f}h")
# # # # # # # #     print("=" * 70)


# # # # # # # # if __name__ == "__main__":
# # # # # # # #     args = get_args()
# # # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # # #     if torch.cuda.is_available():
# # # # # # # #         torch.cuda.manual_seed_all(42)
# # # # # # # #     main(args)

# # # # # # # """
# # # # # # # scripts/train_flowmatching.py  — v36  BEAT ST-TRANS EDITION
# # # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # # TARGET: Beat ST-Trans paper (Bay of Bengal, Expert Systems w/ Applications 2026)
# # # # # # #   Paper:  Mean DPE=136.41 km | ATE=79.94 km | CTE=93.58 km | 72h≈297 km
# # # # # # #   Target: Mean DPE<130 km   | ATE<75 km    | CTE<88 km    | 72h<270 km

# # # # # # # KEY CHANGES từ v34fix:
# # # # # # #   1. losses.py v36: velocity_smoothness + ate_cte_decomp (giống ST-Trans physics)
# # # # # # #   2. metrics.py v7: ATE/CTE per-step tracking trong mọi evaluation
# # # # # # #   3. Composite score v2: normalize theo ST-Trans targets
# # # # # # #   4. EMA decay giảm 0.995→0.992 để update nhanh hơn
# # # # # # #   5. Log ATE/CTE, beat indicators ở mỗi epoch

# # # # # # # COMMAND:
# # # # # # #   python train_flowmatching.py \
# # # # # # #     --dataset_root /kaggle/input/datasets/gmnguynhng/tc-vn-update-env \
# # # # # # #     --output_dir /kaggle/working/ \
# # # # # # #     --num_epochs 100 \
# # # # # # #     --batch_size 32 \
# # # # # # #     --use_amp \
# # # # # # #     --ode_steps_train 10 \
# # # # # # #     --ode_steps_val 10 \
# # # # # # #     --ode_steps_test 10 \
# # # # # # #     --sigma_min 0.02 \
# # # # # # #     --val_freq 5 \
# # # # # # #     --patience 30
# # # # # # # """
# # # # # # # from __future__ import annotations

# # # # # # # import sys
# # # # # # # import os
# # # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # # import argparse
# # # # # # # import time
# # # # # # # import math
# # # # # # # import random
# # # # # # # import copy

# # # # # # # import numpy as np
# # # # # # # import torch
# # # # # # # import torch.optim as optim
# # # # # # # from torch.amp import autocast, GradScaler
# # # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # # from Model.data.loader_training import data_loader
# # # # # # # from Model.flow_matching_model import TCFlowMatching
# # # # # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # # # # from utils.metrics import (
# # # # # # #     StepErrorAccumulator,
# # # # # # #     save_metrics_csv,
# # # # # # #     haversine_km_torch,
# # # # # # #     denorm_torch,
# # # # # # #     HORIZON_STEPS,
# # # # # # #     DatasetMetrics,
# # # # # # # )

# # # # # # # # ── Try import ATE/CTE helper (metrics v7) ────────────────────────────────────
# # # # # # # try:
# # # # # # #     from utils.metrics import haversine_and_atecte_torch
# # # # # # #     HAS_ATECTE = True
# # # # # # # except ImportError:
# # # # # # #     HAS_ATECTE = False
# # # # # # #     print("  ⚠  haversine_and_atecte_torch not found — ATE/CTE disabled")


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Composite Score v2 — normalize theo ST-Trans targets
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # def _composite_score(result: dict) -> float:
# # # # # # #     """
# # # # # # #     Lower = better. score < 100 → beat ST-Trans trên tất cả metrics.

# # # # # # #     ST-Trans targets: DPE=136.41, ATE=79.94, CTE=93.58, 72h≈297
# # # # # # #     """
# # # # # # #     ade = result.get("ADE", float("inf"))
# # # # # # #     h12 = result.get("12h", float("inf"))
# # # # # # #     h24 = result.get("24h", float("inf"))
# # # # # # #     h48 = result.get("48h", float("inf"))
# # # # # # #     h72 = result.get("72h", float("inf"))
# # # # # # #     ate = result.get("ATE_mean", float("inf"))
# # # # # # #     cte = result.get("CTE_mean", float("inf"))

# # # # # # #     # Fallback nếu chưa có ATE/CTE
# # # # # # #     if not np.isfinite(ate): ate = ade * 0.46
# # # # # # #     if not np.isfinite(cte): cte = ade * 0.53

# # # # # # #     score = (
# # # # # # #         0.05 * (ade / 136.0)
# # # # # # #         + 0.05 * (h12 / 50.0)
# # # # # # #         + 0.10 * (h24 / 100.0)
# # # # # # #         + 0.15 * (h48 / 200.0)
# # # # # # #         + 0.35 * (h72 / 300.0)  # 72h — priority cao nhất
# # # # # # #         + 0.15 * (ate / 80.0)   # ATE — beat 79.94
# # # # # # #         + 0.15 * (cte / 94.0)   # CTE — beat 93.58
# # # # # # #     )
# # # # # # #     return score * 100.0


# # # # # # # def _beat_report(r: dict) -> str:
# # # # # # #     """Tạo string báo cáo so với ST-Trans."""
# # # # # # #     ade = r.get("ADE", float("inf"))
# # # # # # #     h72 = r.get("72h", float("inf"))
# # # # # # #     ate = r.get("ATE_mean", float("inf"))
# # # # # # #     cte = r.get("CTE_mean", float("inf"))
# # # # # # #     parts = []
# # # # # # #     if np.isfinite(ade) and ade < 136.41: parts.append(f"DPE✅{ade:.1f}")
# # # # # # #     if np.isfinite(ate) and ate < 79.94:  parts.append(f"ATE✅{ate:.1f}")
# # # # # # #     if np.isfinite(cte) and cte < 93.58:  parts.append(f"CTE✅{cte:.1f}")
# # # # # # #     if np.isfinite(h72) and h72 < 297.0:  parts.append(f"72h✅{h72:.1f}")
# # # # # # #     return "🏆 BEAT: " + " ".join(parts) if parts else ""


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


# # # # # # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # # # # # #                             collate_fn, num_workers):
# # # # # # #     n   = len(val_dataset)
# # # # # # #     rng = random.Random(42)
# # # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # # #     return DataLoader(Subset(val_dataset, idx),
# # # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # # def _get_ema_obj(model):
# # # # # # #     if hasattr(model, '_ema') and model._ema is not None:
# # # # # # #         return model._ema
# # # # # # #     if hasattr(model, '_orig_mod'):
# # # # # # #         orig = model._orig_mod
# # # # # # #         if hasattr(orig, '_ema') and orig._ema is not None:
# # # # # # #             return orig._ema
# # # # # # #     return None


# # # # # # # def _call_ema_update(model):
# # # # # # #     if hasattr(model, '_orig_mod'):
# # # # # # #         orig = model._orig_mod
# # # # # # #         if hasattr(orig, 'ema_update'):
# # # # # # #             orig.ema_update(); return
# # # # # # #     if hasattr(model, 'ema_update'):
# # # # # # #         model.ema_update()


# # # # # # # def _get_raw_model(model):
# # # # # # #     return model._orig_mod if hasattr(model, '_orig_mod') else model


# # # # # # # def _save_checkpoint(path, epoch, model, optimizer, scheduler,
# # # # # # #                      saver, avg_t, avg_vl, metrics=None):
# # # # # # #     m = _get_raw_model(model)
# # # # # # #     ema_obj = _get_ema_obj(model)
# # # # # # #     ema_sd  = None
# # # # # # #     if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # # # # # #         try:
# # # # # # #             ema_sd = {k: v.cpu().clone() for k, v in ema_obj.shadow.items()}
# # # # # # #         except Exception:
# # # # # # #             ema_sd = None

# # # # # # #     payload = {
# # # # # # #         "epoch"            : epoch,
# # # # # # #         "model_state_dict" : m.state_dict(),
# # # # # # #         "optimizer_state"  : optimizer.state_dict(),
# # # # # # #         "scheduler_state"  : scheduler.state_dict(),
# # # # # # #         "ema_shadow"       : ema_sd,
# # # # # # #         "best_score"       : saver.best_score,
# # # # # # #         "best_ade"         : saver.best_ade,
# # # # # # #         "best_72h"         : saver.best_72h,
# # # # # # #         "best_48h"         : saver.best_48h,
# # # # # # #         "best_24h"         : saver.best_24h,
# # # # # # #         "best_12h"         : saver.best_12h,
# # # # # # #         "best_ate"         : getattr(saver, 'best_ate', float("inf")),
# # # # # # #         "best_cte"         : getattr(saver, 'best_cte', float("inf")),
# # # # # # #         "train_loss"       : avg_t,
# # # # # # #         "val_loss"         : avg_vl,
# # # # # # #     }
# # # # # # #     if metrics:
# # # # # # #         payload.update(metrics)
# # # # # # #     torch.save(payload, path)


# # # # # # # def _load_checkpoint(path, model, optimizer, scheduler, saver, device):
# # # # # # #     if path is None or not os.path.exists(path):
# # # # # # #         if path is not None:
# # # # # # #             print(f"  ⚠  Checkpoint không tìm thấy: {path}")
# # # # # # #         return 0

# # # # # # #     print(f"  Loading checkpoint: {path}")
# # # # # # #     ckpt = torch.load(path, map_location=device)

# # # # # # #     m = _get_raw_model(model)
# # # # # # #     missing, unexpected = m.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # # # #     if missing:
# # # # # # #         print(f"  ⚠  Missing keys ({len(missing)}): "
# # # # # # #               f"{missing[:3]}{'...' if len(missing) > 3 else ''}")

# # # # # # #     if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
# # # # # # #         ema_obj = _get_ema_obj(model)
# # # # # # #         if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # # # # # #             restored = 0
# # # # # # #             for k, v in ckpt["ema_shadow"].items():
# # # # # # #                 if k in ema_obj.shadow:
# # # # # # #                     ema_obj.shadow[k].copy_(v.to(device))
# # # # # # #                     restored += 1
# # # # # # #             print(f"  EMA shadow restored ({restored} keys)")

# # # # # # #     if "optimizer_state" in ckpt:
# # # # # # #         try:
# # # # # # #             optimizer.load_state_dict(ckpt["optimizer_state"])
# # # # # # #             for state in optimizer.state.values():
# # # # # # #                 for k, v in state.items():
# # # # # # #                     if torch.is_tensor(v):
# # # # # # #                         state[k] = v.to(device)
# # # # # # #             print("  Optimizer state restored")
# # # # # # #         except Exception as e:
# # # # # # #             print(f"  ⚠  Optimizer restore failed: {e}")

# # # # # # #     if "scheduler_state" in ckpt:
# # # # # # #         try:
# # # # # # #             scheduler.load_state_dict(ckpt["scheduler_state"])
# # # # # # #             print(f"  Scheduler restored (last_epoch={scheduler.last_epoch})")
# # # # # # #         except Exception as e:
# # # # # # #             print(f"  ⚠  Scheduler restore failed: {e}")

# # # # # # #     if "best_score" in ckpt:
# # # # # # #         saver.best_score = ckpt["best_score"]
# # # # # # #         saver.best_ade   = ckpt.get("best_ade",  float("inf"))
# # # # # # #         saver.best_72h   = ckpt.get("best_72h",  float("inf"))
# # # # # # #         saver.best_48h   = ckpt.get("best_48h",  float("inf"))
# # # # # # #         saver.best_24h   = ckpt.get("best_24h",  float("inf"))
# # # # # # #         saver.best_12h   = ckpt.get("best_12h",  float("inf"))
# # # # # # #         saver.best_ate   = ckpt.get("best_ate",  float("inf"))
# # # # # # #         saver.best_cte   = ckpt.get("best_cte",  float("inf"))
# # # # # # #         print(f"  Saver state restored (best_score={saver.best_score:.1f})")

# # # # # # #     start_epoch = ckpt.get("epoch", 0) + 1
# # # # # # #     print(f"  → Resuming from epoch {start_epoch}")
# # # # # # #     return start_epoch


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  SWA
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # class SWAManager:
# # # # # # #     def __init__(self, model, start_epoch=60):
# # # # # # #         self.start_epoch = start_epoch
# # # # # # #         self.n_averaged  = 0
# # # # # # #         self.avg_state   = None

# # # # # # #     def update(self, model, epoch):
# # # # # # #         if epoch < self.start_epoch: return
# # # # # # #         m  = _get_raw_model(model)
# # # # # # #         sd = {k: v.detach().clone() for k, v in m.state_dict().items()
# # # # # # #               if v.dtype.is_floating_point}
# # # # # # #         if self.avg_state is None:
# # # # # # #             self.avg_state = sd; self.n_averaged = 1
# # # # # # #         else:
# # # # # # #             n = self.n_averaged
# # # # # # #             for k in self.avg_state:
# # # # # # #                 if k in sd:
# # # # # # #                     self.avg_state[k].mul_(n / (n + 1)).add_(sd[k], alpha=1.0 / (n + 1))
# # # # # # #             self.n_averaged += 1

# # # # # # #     def apply_to(self, model):
# # # # # # #         if self.avg_state is None: return None
# # # # # # #         m  = _get_raw_model(model)
# # # # # # #         sd = m.state_dict()
# # # # # # #         backup = {k: sd[k].detach().clone() for k in self.avg_state if k in sd}
# # # # # # #         for k, v in self.avg_state.items():
# # # # # # #             if k in sd: sd[k].copy_(v)
# # # # # # #         return backup

# # # # # # #     def restore(self, model, backup):
# # # # # # #         if backup is None: return
# # # # # # #         m  = _get_raw_model(model)
# # # # # # #         sd = m.state_dict()
# # # # # # #         for k, v in backup.items():
# # # # # # #             if k in sd: sd[k].copy_(v)


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Evaluation — với ATE/CTE
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # def _eval_batch_atecte(pred_norm, gt_norm):
# # # # # # #     """
# # # # # # #     Tính dist, ate, cte từ normalized coords.
# # # # # # #     Fallback nếu không có haversine_and_atecte_torch.
# # # # # # #     """
# # # # # # #     pred_d = denorm_torch(pred_norm)
# # # # # # #     gt_d   = denorm_torch(gt_norm)

# # # # # # #     if HAS_ATECTE:
# # # # # # #         dist, ate, cte = haversine_and_atecte_torch(pred_d, gt_d, unit_01deg=True)
# # # # # # #         return dist, ate, cte
# # # # # # #     else:
# # # # # # #         dist = haversine_km_torch(pred_d, gt_d)
# # # # # # #         return dist, None, None


# # # # # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=15):
# # # # # # #     model.eval()
# # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # #     t0  = time.perf_counter()
# # # # # # #     n   = 0
# # # # # # #     spread_per_step = []

# # # # # # #     with torch.no_grad():
# # # # # # #         for batch in loader:
# # # # # # #             bl = move(list(batch), device)
# # # # # # #             pred, _, all_trajs = model.sample(
# # # # # # #                 bl, num_ensemble=fast_ensemble, ddim_steps=ode_steps,
# # # # # # #                 importance_weight=True)
# # # # # # #             T_active  = pred.shape[0]
# # # # # # #             gt_sliced = bl[1][:T_active]

# # # # # # #             dist, ate, cte = _eval_batch_atecte(pred, gt_sliced)
# # # # # # #             acc.update(dist, ate_km=ate, cte_km=cte)

# # # # # # #             # Spread
# # # # # # #             step_spreads = []
# # # # # # #             for t in range(all_trajs.shape[1]):
# # # # # # #                 step_data = all_trajs[:, t, :, :]
# # # # # # #                 spread = ((step_data[:, :, 0].std(0)**2
# # # # # # #                            + step_data[:, :, 1].std(0)**2).sqrt() * 500.0).mean().item()
# # # # # # #                 step_spreads.append(spread)
# # # # # # #             spread_per_step.append(step_spreads)
# # # # # # #             n += 1

# # # # # # #     r = acc.compute()
# # # # # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# # # # # # #     if spread_per_step:
# # # # # # #         spreads = np.array(spread_per_step)
# # # # # # #         r["spread_72h_km"] = float(spreads[:, -1].mean())
# # # # # # #     return r


# # # # # # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # # # # # #                            fast_ensemble, metrics_csv, epoch,
# # # # # # #                            use_ema=False, ema_obj=None):
# # # # # # #     backup = None
# # # # # # #     if use_ema and ema_obj is not None:
# # # # # # #         try:
# # # # # # #             backup = ema_obj.apply_to(model)
# # # # # # #         except Exception as e:
# # # # # # #             print(f"  ⚠  EMA apply_to failed: {e}")
# # # # # # #             backup  = None; use_ema = False

# # # # # # #     model.eval()
# # # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # # #     t0  = time.perf_counter()
# # # # # # #     n   = 0

# # # # # # #     with torch.no_grad():
# # # # # # #         for batch in val_loader:
# # # # # # #             bl = move(list(batch), device)
# # # # # # #             pred, _, _ = model.sample(
# # # # # # #                 bl, num_ensemble=max(fast_ensemble, 20),
# # # # # # #                 ddim_steps=max(ode_steps, 20),
# # # # # # #                 importance_weight=True)
# # # # # # #             T_pred = pred.shape[0]
# # # # # # #             gt     = bl[1][:T_pred]
# # # # # # #             dist, ate, cte = _eval_batch_atecte(pred, gt)
# # # # # # #             acc.update(dist, ate_km=ate, cte_km=cte)
# # # # # # #             n += 1

# # # # # # #     r       = acc.compute()
# # # # # # #     elapsed = time.perf_counter() - t0
# # # # # # #     score   = _composite_score(r)
# # # # # # #     tag     = "EMA" if use_ema else "RAW"

# # # # # # #     ade_v = r.get("ADE",     float("nan"))
# # # # # # #     fde_v = r.get("FDE",     float("nan"))
# # # # # # #     h6_v  = r.get("6h",      float("nan"))
# # # # # # #     h12_v = r.get("12h",     float("nan"))
# # # # # # #     h24_v = r.get("24h",     float("nan"))
# # # # # # #     h48_v = r.get("48h",     float("nan"))
# # # # # # #     h72_v = r.get("72h",     float("nan"))
# # # # # # #     ate_v = r.get("ATE_mean", float("nan"))
# # # # # # #     cte_v = r.get("CTE_mean", float("nan"))

# # # # # # #     def ind(v, tgt):
# # # # # # #         if not np.isfinite(v): return ""
# # # # # # #         return "✅" if v < tgt else "❌"

# # # # # # #     print(f"\n{'='*70}")
# # # # # # #     print(f"  [FULL VAL ADE ({tag})  ep={epoch}  {elapsed:.0f}s  {n} batches]")
# # # # # # #     print(f"  ADE={ade_v:.1f} km {ind(ade_v,136.41)}  "
# # # # # # #           f"FDE={fde_v:.1f} km")
# # # # # # #     print(f"  6h={h6_v:.0f}  12h={h12_v:.0f}{ind(h12_v,50)}  "
# # # # # # #           f"24h={h24_v:.0f}{ind(h24_v,100)}  "
# # # # # # #           f"48h={h48_v:.0f}{ind(h48_v,200)}  "
# # # # # # #           f"72h={h72_v:.0f}{ind(h72_v,300)} km")
# # # # # # #     if np.isfinite(ate_v):
# # # # # # #         print(f"  ATE={ate_v:.1f}{ind(ate_v,79.94)}  "
# # # # # # #               f"CTE={cte_v:.1f}{ind(cte_v,93.58)}  "
# # # # # # #               f"[ST-Trans: ATE=79.94 CTE=93.58]")
# # # # # # #     beat = _beat_report(r)
# # # # # # #     if beat: print(f"  {beat}")
# # # # # # #     print(f"  Score v2 = {score:.2f}  (< 100 = beat ST-Trans all metrics)")
# # # # # # #     print(f"{'='*70}\n")

# # # # # # #     # Save CSV
# # # # # # #     from datetime import datetime
# # # # # # #     dm = DatasetMetrics(
# # # # # # #         ade          = ade_v if np.isfinite(ade_v) else 0.0,
# # # # # # #         fde          = fde_v if np.isfinite(fde_v) else 0.0,
# # # # # # #         ugde_12h     = h12_v if np.isfinite(h12_v) else 0.0,
# # # # # # #         ugde_24h     = h24_v if np.isfinite(h24_v) else 0.0,
# # # # # # #         ugde_48h     = h48_v if np.isfinite(h48_v) else 0.0,
# # # # # # #         ugde_72h     = h72_v if np.isfinite(h72_v) else 0.0,
# # # # # # #         ate_abs_mean = ate_v if np.isfinite(ate_v) else 0.0,
# # # # # # #         cte_abs_mean = cte_v if np.isfinite(cte_v) else 0.0,
# # # # # # #         n_total      = r.get("n_samples", 0),
# # # # # # #         timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # # #     )
# # # # # # #     save_metrics_csv(dm, metrics_csv, tag=f"val_full_{tag}_ep{epoch:03d}")

# # # # # # #     if backup is not None:
# # # # # # #         try:
# # # # # # #             ema_obj.restore(model, backup)
# # # # # # #         except Exception as e:
# # # # # # #             print(f"  ⚠  EMA restore failed: {e}")

# # # # # # #     return r


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  BestModelSaver — v2 với ATE/CTE
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # class HorizonAwareBestSaver:
# # # # # # #     def __init__(self, patience=30, tol=1.5):
# # # # # # #         self.patience   = patience
# # # # # # #         self.tol        = tol
# # # # # # #         self.counter    = 0
# # # # # # #         self.early_stop = False
# # # # # # #         self.best_score = float("inf")
# # # # # # #         self.best_ade   = float("inf")
# # # # # # #         self.best_12h   = float("inf")
# # # # # # #         self.best_24h   = float("inf")
# # # # # # #         self.best_48h   = float("inf")
# # # # # # #         self.best_72h   = float("inf")
# # # # # # #         self.best_ate   = float("inf")
# # # # # # #         self.best_cte   = float("inf")

# # # # # # #     def update(self, r, model, out_dir, epoch, optimizer, scheduler,
# # # # # # #                tl, vl, saver_ref, min_epochs=30):
# # # # # # #         ade   = r.get("ADE", float("inf"))
# # # # # # #         h12   = r.get("12h", float("inf"))
# # # # # # #         h24   = r.get("24h", float("inf"))
# # # # # # #         h48   = r.get("48h", float("inf"))
# # # # # # #         h72   = r.get("72h", float("inf"))
# # # # # # #         ate   = r.get("ATE_mean", ade * 0.46 if np.isfinite(ade) else float("inf"))
# # # # # # #         cte   = r.get("CTE_mean", ade * 0.53 if np.isfinite(ade) else float("inf"))
# # # # # # #         score = _composite_score(r)

# # # # # # #         improved_any = False

# # # # # # #         if ade < self.best_ade:
# # # # # # #             self.best_ade = ade; improved_any = True
# # # # # # #             _save_checkpoint(
# # # # # # #                 os.path.join(out_dir, "best_ade.pth"),
# # # # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # # # #                 {"ade": ade, "tag": "best_ade"})

# # # # # # #         if h72 < self.best_72h:
# # # # # # #             self.best_72h = h72; improved_any = True
# # # # # # #             _save_checkpoint(
# # # # # # #                 os.path.join(out_dir, "best_72h.pth"),
# # # # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # # # #                 {"h72": h72, "tag": "best_72h"})

# # # # # # #         if ate < self.best_ate:
# # # # # # #             self.best_ate = ate; improved_any = True
# # # # # # #             _save_checkpoint(
# # # # # # #                 os.path.join(out_dir, "best_ate.pth"),
# # # # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # # # #                 {"ate": ate, "cte": cte, "tag": "best_ate"})

# # # # # # #         if cte < self.best_cte:
# # # # # # #             self.best_cte = cte; improved_any = True
# # # # # # #             _save_checkpoint(
# # # # # # #                 os.path.join(out_dir, "best_cte.pth"),
# # # # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # # # #                 {"ate": ate, "cte": cte, "tag": "best_cte"})

# # # # # # #         if h48 < self.best_48h: self.best_48h = h48; improved_any = True
# # # # # # #         if h24 < self.best_24h: self.best_24h = h24; improved_any = True
# # # # # # #         if h12 < self.best_12h: self.best_12h = h12; improved_any = True

# # # # # # #         if score < self.best_score - self.tol:
# # # # # # #             self.best_score = score
# # # # # # #             self.counter    = 0
# # # # # # #             _save_checkpoint(
# # # # # # #                 os.path.join(out_dir, "best_model.pth"),
# # # # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # # # #                 {"ade": ade, "h12": h12, "h24": h24, "h48": h48, "h72": h72,
# # # # # # #                  "ate": ate, "cte": cte,
# # # # # # #                  "composite_score": score, "tag": "best_composite"})
# # # # # # #             print(f"  ✅ Best COMPOSITE={score:.2f}  "
# # # # # # #                   f"ADE={ade:.1f}  12h={h12:.0f}  24h={h24:.0f}  "
# # # # # # #                   f"48h={h48:.0f}  72h={h72:.0f}  "
# # # # # # #                   f"ATE={ate:.1f}  CTE={cte:.1f}  (ep {epoch})")
# # # # # # #         else:
# # # # # # #             if not improved_any:
# # # # # # #                 self.counter += 1
# # # # # # #             print(f"  No improvement {self.counter}/{self.patience}"
# # # # # # #                   f"  (best={self.best_score:.2f} cur={score:.2f})"
# # # # # # #                   f"  72h={h72:.0f}↓{self.best_72h:.0f}"
# # # # # # #                   f"  ATE={ate:.1f}↓{self.best_ate:.1f}")

# # # # # # #         if epoch >= min_epochs and self.counter >= self.patience:
# # # # # # #             self.early_stop = True


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Env diagnostic
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # def _check_env(bl, train_dataset):
# # # # # # #     try:    env_dir = train_dataset.env_path
# # # # # # #     except AttributeError:
# # # # # # #         try:    env_dir = train_dataset.dataset.env_path
# # # # # # #         except: env_dir = "UNKNOWN"
# # # # # # #     print(f"  Env path: {env_dir}")
# # # # # # #     env_data = bl[13]
# # # # # # #     if env_data is None:
# # # # # # #         print("  ⚠️  env_data is None"); return
# # # # # # #     for key in ("gph500_mean", "gph500_center", "u500_mean", "v500_mean"):
# # # # # # #         if key not in env_data:
# # # # # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # # # # #         v    = env_data[key]
# # # # # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # # # # #         print(f"  {'✅' if zero < 80 else '⚠️'}  {key}: "
# # # # # # #               f"mean={v.mean().item():.4f} std={v.std().item():.4f} zero={zero:.1f}%")


# # # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # # #  Args
# # # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # # def get_args():
# # # # # # #     p = argparse.ArgumentParser(
# # # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # # # # # #     p.add_argument("--obs_len",         default=8,          type=int)
# # # # # # #     p.add_argument("--pred_len",        default=12,         type=int)
# # # # # # #     p.add_argument("--batch_size",      default=32,         type=int)
# # # # # # #     p.add_argument("--num_epochs",      default=100,        type=int)
# # # # # # #     p.add_argument("--g_learning_rate", default=3e-4,       type=float)
# # # # # # #     p.add_argument("--weight_decay",    default=1e-4,       type=float)
# # # # # # #     p.add_argument("--warmup_epochs",   default=5,          type=int)
# # # # # # #     p.add_argument("--grad_clip",       default=1.0,        type=float)
# # # # # # #     p.add_argument("--patience",        default=30,         type=int)
# # # # # # #     p.add_argument("--min_epochs",      default=30,         type=int)
# # # # # # #     p.add_argument("--n_train_ens",     default=4,          type=int)
# # # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # # #     p.add_argument("--num_workers",     default=2,          type=int)

# # # # # # #     p.add_argument("--sigma_min",            default=0.02,  type=float)
# # # # # # #     p.add_argument("--ctx_noise_scale",      default=0.01,  type=float)
# # # # # # #     p.add_argument("--initial_sample_sigma", default=0.03,  type=float)

# # # # # # #     p.add_argument("--ode_steps_train", default=10,  type=int)
# # # # # # #     p.add_argument("--ode_steps_val",   default=10,  type=int)
# # # # # # #     p.add_argument("--ode_steps_test",  default=10,  type=int)

# # # # # # #     p.add_argument("--val_ensemble",    default=20,  type=int)
# # # # # # #     p.add_argument("--fast_ensemble",   default=10,  type=int)

# # # # # # #     p.add_argument("--val_freq",        default=5,   type=int)
# # # # # # #     p.add_argument("--val_ade_freq",    default=5,   type=int)
# # # # # # #     p.add_argument("--val_subset_size", default=500, type=int)

# # # # # # #     p.add_argument("--use_ema",         action="store_true", default=True)
# # # # # # #     p.add_argument("--ema_decay",       default=0.992, type=float,
# # # # # # #                    help="EMA decay. 0.992 thay vì 0.995 để update nhanh hơn")
# # # # # # #     p.add_argument("--swa_start_epoch", default=60,   type=int)
# # # # # # #     p.add_argument("--teacher_forcing", action="store_true", default=True)

# # # # # # #     p.add_argument("--output_dir",  default="runs/v36",       type=str)
# # # # # # #     p.add_argument("--metrics_csv", default="metrics_v36.csv", type=str)
# # # # # # #     p.add_argument("--predict_csv", default="predictions.csv", type=str)

# # # # # # #     p.add_argument("--gpu_num",     default="0", type=str)
# # # # # # #     p.add_argument("--delim",       default=" ")
# # # # # # #     p.add_argument("--skip",        default=1,   type=int)
# # # # # # #     p.add_argument("--min_ped",     default=1,   type=int)
# # # # # # #     p.add_argument("--threshold",   default=0.002, type=float)
# # # # # # #     p.add_argument("--other_modal", default="gph")
# # # # # # #     p.add_argument("--test_year",   default=None, type=int)

# # # # # # #     # Resume
# # # # # # #     p.add_argument("--resume",       default=None, type=str)
# # # # # # #     p.add_argument("--resume_epoch", default=None, type=int)

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

# # # # # # #     print("=" * 72)
# # # # # # #     print("  TC-FlowMatching v36  |  BEAT ST-TRANS EDITION")
# # # # # # #     print("  TARGETS vs ST-Trans: DPE<136 | ATE<79.94 | CTE<93.58 | 72h<297")
# # # # # # #     print("  NEW LOSSES: velocity_smoothness + ate_cte_decomp")
# # # # # # #     print("  EMA decay:", args.ema_decay, "(0.992 < 0.995 → faster update)")
# # # # # # #     print("=" * 72)

# # # # # # #     # ── Data ──────────────────────────────────────────────────────────────
# # # # # # #     train_dataset, train_loader = data_loader(
# # # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # # #     val_dataset, val_loader = data_loader(
# # # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # # #     val_subset_loader = make_val_subset_loader(
# # # # # # #         val_dataset, args.val_subset_size, args.batch_size,
# # # # # # #         seq_collate, args.num_workers)

# # # # # # #     test_loader = None
# # # # # # #     try:
# # # # # # #         _, test_loader = data_loader(
# # # # # # #             args, {"root": args.dataset_root, "type": "test"},
# # # # # # #             test=True, test_year=None)
# # # # # # #     except Exception as e:
# # # # # # #         print(f"  Warning: test loader: {e}")

# # # # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # # # #     print(f"  val   : {len(val_dataset)} seq")

# # # # # # #     # ── Model ─────────────────────────────────────────────────────────────
# # # # # # #     model = TCFlowMatching(
# # # # # # #         pred_len             = args.pred_len,
# # # # # # #         obs_len              = args.obs_len,
# # # # # # #         sigma_min            = args.sigma_min,
# # # # # # #         n_train_ens          = args.n_train_ens,
# # # # # # #         ctx_noise_scale      = args.ctx_noise_scale,
# # # # # # #         initial_sample_sigma = args.initial_sample_sigma,
# # # # # # #         teacher_forcing      = args.teacher_forcing,
# # # # # # #         use_ema              = args.use_ema,
# # # # # # #         ema_decay            = args.ema_decay,
# # # # # # #     ).to(device)

# # # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # # #     print(f"  params  : {n_params:,}")

# # # # # # #     model.init_ema()
# # # # # # #     print(f"  EMA     : {'ON' if model._ema is not None else 'OFF'}"
# # # # # # #           f"  (decay={args.ema_decay})")

# # # # # # #     # ── Optimizer + Scheduler ─────────────────────────────────────────────
# # # # # # #     optimizer = optim.AdamW(
# # # # # # #         model.parameters(),
# # # # # # #         lr=args.g_learning_rate,
# # # # # # #         weight_decay=args.weight_decay,
# # # # # # #     )

# # # # # # #     saver  = HorizonAwareBestSaver(patience=args.patience, tol=1.5)
# # # # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # # # # #     swa    = SWAManager(model, start_epoch=args.swa_start_epoch)

# # # # # # #     steps_per_epoch = len(train_loader)
# # # # # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # # # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # # # # #     scheduler = get_cosine_schedule_with_warmup(
# # # # # # #         optimizer, warmup, total_steps, min_lr=1e-6)

# # # # # # #     # ── Resume ────────────────────────────────────────────────────────────
# # # # # # #     has_scheduler_state    = False
# # # # # # #     _ckpt_scheduler_state  = None
# # # # # # #     start_epoch = 0

# # # # # # #     if args.resume is not None and os.path.exists(args.resume):
# # # # # # #         print(f"  Loading checkpoint: {args.resume}")
# # # # # # #         ckpt = torch.load(args.resume, map_location=device)

# # # # # # #         m = _get_raw_model(model)
# # # # # # #         missing, _ = m.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # # # #         if missing:
# # # # # # #             print(f"  ⚠  Missing keys ({len(missing)}): "
# # # # # # #                   f"{missing[:3]}{'...' if len(missing)>3 else ''}")

# # # # # # #         if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
# # # # # # #             ema_obj = _get_ema_obj(model)
# # # # # # #             if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # # # # # #                 restored = sum(
# # # # # # #                     1 for k, v in ckpt["ema_shadow"].items()
# # # # # # #                     if k in ema_obj.shadow
# # # # # # #                     and not ema_obj.shadow[k].copy_(v.to(device)) is None
# # # # # # #                 )
# # # # # # #                 print(f"  EMA shadow restored ({restored} keys)")

# # # # # # #         if "optimizer_state" in ckpt:
# # # # # # #             try:
# # # # # # #                 optimizer.load_state_dict(ckpt["optimizer_state"])
# # # # # # #                 for state in optimizer.state.values():
# # # # # # #                     for k, v in state.items():
# # # # # # #                         if torch.is_tensor(v): state[k] = v.to(device)
# # # # # # #                 print("  Optimizer state restored")
# # # # # # #             except Exception as e:
# # # # # # #                 print(f"  ⚠  Optimizer restore failed: {e}")

# # # # # # #         has_scheduler_state   = "scheduler_state" in ckpt
# # # # # # #         _ckpt_scheduler_state = ckpt.get("scheduler_state", None)

# # # # # # #         if "best_score" in ckpt:
# # # # # # #             saver.best_score = ckpt["best_score"]
# # # # # # #             saver.best_ade   = ckpt.get("best_ade",  float("inf"))
# # # # # # #             saver.best_72h   = ckpt.get("best_72h",  float("inf"))
# # # # # # #             saver.best_48h   = ckpt.get("best_48h",  float("inf"))
# # # # # # #             saver.best_24h   = ckpt.get("best_24h",  float("inf"))
# # # # # # #             saver.best_12h   = ckpt.get("best_12h",  float("inf"))
# # # # # # #             saver.best_ate   = ckpt.get("best_ate",  float("inf"))
# # # # # # #             saver.best_cte   = ckpt.get("best_cte",  float("inf"))
# # # # # # #             print(f"  Saver state restored (best_score={saver.best_score:.2f})")

# # # # # # #         start_epoch = (args.resume_epoch if args.resume_epoch is not None
# # # # # # #                        else ckpt.get("epoch", 0) + 1)
# # # # # # #         print(f"  → Resuming from epoch {start_epoch}")

# # # # # # #     elif args.resume is not None:
# # # # # # #         print(f"  ⚠  Checkpoint không tìm thấy: {args.resume}")

# # # # # # #     # ── torch.compile ─────────────────────────────────────────────────────
# # # # # # #     try:
# # # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # # #         print("  torch.compile: enabled")
# # # # # # #     except Exception:
# # # # # # #         pass

# # # # # # #     # ── Scheduler restore/advance ─────────────────────────────────────────
# # # # # # #     if has_scheduler_state and _ckpt_scheduler_state is not None:
# # # # # # #         try:
# # # # # # #             scheduler.load_state_dict(_ckpt_scheduler_state)
# # # # # # #             print(f"  Scheduler restored (last_epoch={scheduler.last_epoch})")
# # # # # # #         except Exception as e:
# # # # # # #             print(f"  ⚠  Scheduler restore failed: {e} — advancing manually")
# # # # # # #             for _ in range(start_epoch * steps_per_epoch):
# # # # # # #                 scheduler.step()
# # # # # # #     elif start_epoch > 0:
# # # # # # #         print(f"  Advancing scheduler {start_epoch * steps_per_epoch} steps...")
# # # # # # #         for _ in range(start_epoch * steps_per_epoch):
# # # # # # #             scheduler.step()

# # # # # # #     print("=" * 72)
# # # # # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch, start={start_epoch})")
# # # # # # #     print("=" * 72)

# # # # # # #     epoch_times = []
# # # # # # #     train_start = time.perf_counter()

# # # # # # #     for epoch in range(start_epoch, args.num_epochs):
# # # # # # #         model.train()
# # # # # # #         sum_loss = 0.0
# # # # # # #         t0 = time.perf_counter()

# # # # # # #         for i, batch in enumerate(train_loader):
# # # # # # #             bl = move(list(batch), device)

# # # # # # #             if epoch == start_epoch and i == 0:
# # # # # # #                 _check_env(bl, train_dataset)

# # # # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # #                 bd = model.get_loss_breakdown(bl, epoch=epoch)

# # # # # # #             optimizer.zero_grad()
# # # # # # #             scaler.scale(bd["total"]).backward()
# # # # # # #             scaler.unscale_(optimizer)
# # # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # # # #             scaler.step(optimizer)
# # # # # # #             scaler.update()
# # # # # # #             scheduler.step()

# # # # # # #             _call_ema_update(model)
# # # # # # #             sum_loss += bd["total"].item()

# # # # # # #             if i % 20 == 0:
# # # # # # #                 lr = optimizer.param_groups[0]["lr"]
# # # # # # #                 # Log các loss key có trong v36
# # # # # # #                 print(
# # # # # # #                     f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # # # # #                     f"  tot={bd['total'].item():.3f}"
# # # # # # #                     f"  fm={bd.get('fm_mse', 0):.3f}"
# # # # # # #                     f"  hor={bd.get('mse_hav', 0):.3f}"
# # # # # # #                     f"  end={bd.get('endpoint', 0):.3f}"
# # # # # # #                     f"  vel={bd.get('vel_smooth', 0):.3f}"
# # # # # # #                     f"  atc={bd.get('ate_cte', 0):.3f}"
# # # # # # #                     f"  hd={bd.get('heading', 0):.3f}"
# # # # # # #                     f"  str={bd.get('steering', 0):.3f}"
# # # # # # #                     f"  lr={lr:.2e}"
# # # # # # #                 )

# # # # # # #         ep_s  = time.perf_counter() - t0
# # # # # # #         epoch_times.append(ep_s)
# # # # # # #         avg_t = sum_loss / len(train_loader)

# # # # # # #         swa.update(model, epoch)

# # # # # # #         # Val loss
# # # # # # #         model.eval()
# # # # # # #         val_loss = 0.0
# # # # # # #         with torch.no_grad():
# # # # # # #             for batch in val_loader:
# # # # # # #                 bl_v = move(list(batch), device)
# # # # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # # #                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
# # # # # # #         avg_vl = val_loss / len(val_loader)

# # # # # # #         print(f"  Epoch {epoch:>3}  train={avg_t:.4f}  val={avg_vl:.4f}"
# # # # # # #               f"  time={ep_s:.0f}s")

# # # # # # #         # ── Fast eval (mỗi epoch) ─────────────────────────────────────────
# # # # # # #         m_fast = evaluate_fast(model, val_subset_loader, device,
# # # # # # #                                 args.ode_steps_train, args.pred_len,
# # # # # # #                                 args.fast_ensemble)

# # # # # # #         h6  = m_fast.get("6h",  float("nan"))
# # # # # # #         h12 = m_fast.get("12h", float("nan"))
# # # # # # #         h24 = m_fast.get("24h", float("nan"))
# # # # # # #         h48 = m_fast.get("48h", float("nan"))
# # # # # # #         h72 = m_fast.get("72h", float("nan"))
# # # # # # #         ate = m_fast.get("ATE_mean", float("nan"))
# # # # # # #         cte = m_fast.get("CTE_mean", float("nan"))
# # # # # # #         fast_score = _composite_score(m_fast)

# # # # # # #         def ind(v, tgt):
# # # # # # #             if not np.isfinite(v): return "?"
# # # # # # #             return "🎯" if v < tgt else "❌"

# # # # # # #         print(
# # # # # # #             f"  [FAST ep{epoch}]"
# # # # # # #             f"  ADE={m_fast.get('ADE', float('nan')):.1f}"
# # # # # # #             f"  12h={h12:.0f}{ind(h12,50)}"
# # # # # # #             f"  24h={h24:.0f}{ind(h24,100)}"
# # # # # # #             f"  48h={h48:.0f}{ind(h48,200)}"
# # # # # # #             f"  72h={h72:.0f}{ind(h72,300)}"
# # # # # # #             f"  ATE={ate:.1f}{ind(ate,79.94)}"
# # # # # # #             f"  CTE={cte:.1f}{ind(cte,93.58)}"
# # # # # # #             f"  score={fast_score:.2f}"
# # # # # # #         )

# # # # # # #         # ── Full val eval (mỗi val_freq epoch) ───────────────────────────
# # # # # # #         if epoch % args.val_freq == 0:
# # # # # # #             ema_obj = _get_ema_obj(model)
# # # # # # #             try:
# # # # # # #                 r_raw = evaluate_full_val_ade(
# # # # # # #                     model, val_loader, device,
# # # # # # #                     ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # # # # #                     fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # # # #                     epoch=epoch, use_ema=False, ema_obj=None)

# # # # # # #                 r_use = r_raw
# # # # # # #                 if ema_obj is not None and epoch >= 5:
# # # # # # #                     r_ema = evaluate_full_val_ade(
# # # # # # #                         model, val_loader, device,
# # # # # # #                         ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # # # # #                         fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # # # #                         epoch=epoch, use_ema=True, ema_obj=ema_obj)
# # # # # # #                     if _composite_score(r_ema) < _composite_score(r_raw):
# # # # # # #                         r_use = r_ema

# # # # # # #                 saver.update(r_use, model, args.output_dir, epoch,
# # # # # # #                              optimizer, scheduler, avg_t, avg_vl,
# # # # # # #                              saver_ref=saver, min_epochs=args.min_epochs)
# # # # # # #             except Exception as e:
# # # # # # #                 print(f"  ⚠  Full val failed: {e}")
# # # # # # #                 import traceback; traceback.print_exc()

# # # # # # #         # ── Checkpoint định kỳ ────────────────────────────────────────────
# # # # # # #         if epoch % 10 == 0 or epoch == args.num_epochs - 1:
# # # # # # #             _save_checkpoint(
# # # # # # #                 os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"),
# # # # # # #                 epoch, model, optimizer, scheduler, saver, avg_t, avg_vl,
# # # # # # #                 {"ade": m_fast.get("ADE", float("nan")),
# # # # # # #                  "h48": h48, "h72": h72,
# # # # # # #                  "ate": ate, "cte": cte})

# # # # # # #         if epoch % 10 == 9:
# # # # # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # # # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # # # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # # # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")
# # # # # # #             print(f"  📊 Best so far: score={saver.best_score:.2f}  "
# # # # # # #                   f"72h={saver.best_72h:.0f}km  ATE={saver.best_ate:.1f}km  "
# # # # # # #                   f"CTE={saver.best_cte:.1f}km")

# # # # # # #         if saver.early_stop:
# # # # # # #             print(f"  ⛔ Early stopping at epoch {epoch}")
# # # # # # #             break

# # # # # # #     # ── Final: SWA eval ───────────────────────────────────────────────────
# # # # # # #     print(f"\n{'='*72}")
# # # # # # #     swa_backup = swa.apply_to(model)
# # # # # # #     if swa_backup is not None:
# # # # # # #         print("  Evaluating SWA weights...")
# # # # # # #         r_swa = evaluate_full_val_ade(
# # # # # # #             model, val_loader, device,
# # # # # # #             ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # # # # #             fast_ensemble=args.val_ensemble, metrics_csv=metrics_csv,
# # # # # # #             epoch=9999, use_ema=False)
# # # # # # #         if _composite_score(r_swa) < saver.best_score:
# # # # # # #             _save_checkpoint(
# # # # # # #                 os.path.join(args.output_dir, "best_swa.pth"),
# # # # # # #                 9999, model, optimizer, scheduler, saver,
# # # # # # #                 avg_t=0.0, avg_vl=0.0, metrics={"tag": "best_swa"})
# # # # # # #             print("  ✅ SWA checkpoint saved")
# # # # # # #         swa.restore(model, swa_backup)

# # # # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # # # #     print(f"\n  Best composite score v2: {saver.best_score:.2f}")
# # # # # # #     print(f"  Best metrics: ADE={saver.best_ade:.1f}  "
# # # # # # #           f"12h={saver.best_12h:.0f}  24h={saver.best_24h:.0f}  "
# # # # # # #           f"48h={saver.best_48h:.0f}  72h={saver.best_72h:.0f}  "
# # # # # # #           f"ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}")
# # # # # # #     print(f"  Total time: {total_h:.2f}h")

# # # # # # #     # Beat report
# # # # # # #     best_r = {
# # # # # # #         "ADE": saver.best_ade, "12h": saver.best_12h,
# # # # # # #         "24h": saver.best_24h, "48h": saver.best_48h,
# # # # # # #         "72h": saver.best_72h, "ATE_mean": saver.best_ate,
# # # # # # #         "CTE_mean": saver.best_cte,
# # # # # # #     }
# # # # # # #     beat = _beat_report(best_r)
# # # # # # #     if beat:
# # # # # # #         print(f"\n  🏆🏆🏆 {beat} 🏆🏆🏆")
# # # # # # #     else:
# # # # # # #         print(f"\n  ST-Trans targets not yet beaten. Best score={saver.best_score:.2f}")
# # # # # # #         print(f"  Need: DPE<136.41, ATE<79.94, CTE<93.58, 72h<297")
# # # # # # #     print("=" * 72)


# # # # # # # if __name__ == "__main__":
# # # # # # #     args = get_args()
# # # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # # #     if torch.cuda.is_available():
# # # # # # #         torch.cuda.manual_seed_all(42)
# # # # # # #     main(args)

# # # # # # """
# # # # # # scripts/train_flowmatching.py  — v37  ATE-FOCUSED EDITION
# # # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # # TARGET: Beat ST-Trans paper (Bay of Bengal, Expert Systems w/ Applications 2026)
# # # # # #   Paper:  Mean DPE=136.41 km | ATE=79.94 km | CTE=93.58 km | 72h≈297 km
# # # # # #   Status v36: CTE=69.0 ✅  ATE=168.6 ❌  72h=395 ❌

# # # # # # KEY CHANGES từ v36:
# # # # # #   1. losses.py v37: along_track_speed_loss (NEW) — fix ATE via speed matching
# # # # # #   2. losses.py v37: ate_cte ate_weight=2.0, cte_weight=0.5 (CTE đã tốt)
# # # # # #   3. losses.py v37: velocity_smoothness vmax 80→95, thêm penalty TOO SLOW
# # # # # #   4. losses.py v37: direct_multi_horizon_mse dùng Huber (stable hơn pow(2))
# # # # # #   5. flow_matching_model v37: sigma schedule tighter (0.10 thay 0.15)
# # # # # #   6. EMA decay: 0.992 → 0.990 (faster update)
# # # # # #   7. Log thêm "alng" (along_track) để monitor

# # # # # # COMMAND:
# # # # # #   python train_flowmatching.py \
# # # # # #     --dataset_root /kaggle/input/datasets/gmnguynhng/tc-vn-update-env \
# # # # # #     --output_dir /kaggle/working/ \
# # # # # #     --num_epochs 100 \
# # # # # #     --batch_size 32 \
# # # # # #     --use_amp \
# # # # # #     --ode_steps_train 10 \
# # # # # #     --ode_steps_val 10 \
# # # # # #     --ode_steps_test 10 \
# # # # # #     --sigma_min 0.02 \
# # # # # #     --val_freq 5 \
# # # # # #     --patience 30 \
# # # # # #     --ema_decay 0.990
# # # # # # """
# # # # # # from __future__ import annotations

# # # # # # import sys
# # # # # # import os
# # # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # # import argparse
# # # # # # import time
# # # # # # import math
# # # # # # import random
# # # # # # import copy

# # # # # # import numpy as np
# # # # # # import torch
# # # # # # import torch.optim as optim
# # # # # # from torch.amp import autocast, GradScaler
# # # # # # from torch.utils.data import DataLoader, Subset

# # # # # # from Model.data.loader_training import data_loader
# # # # # # from Model.flow_matching_model import TCFlowMatching
# # # # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # # # from utils.metrics import (
# # # # # #     StepErrorAccumulator,
# # # # # #     save_metrics_csv,
# # # # # #     haversine_km_torch,
# # # # # #     denorm_torch,
# # # # # #     HORIZON_STEPS,
# # # # # #     DatasetMetrics,
# # # # # # )

# # # # # # try:
# # # # # #     from utils.metrics import haversine_and_atecte_torch
# # # # # #     HAS_ATECTE = True
# # # # # # except ImportError:
# # # # # #     HAS_ATECTE = False
# # # # # #     print("  ⚠  haversine_and_atecte_torch not found — ATE/CTE disabled")


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Composite Score v2
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def _composite_score(result: dict) -> float:
# # # # # #     ade = result.get("ADE", float("inf"))
# # # # # #     h12 = result.get("12h", float("inf"))
# # # # # #     h24 = result.get("24h", float("inf"))
# # # # # #     h48 = result.get("48h", float("inf"))
# # # # # #     h72 = result.get("72h", float("inf"))
# # # # # #     ate = result.get("ATE_mean", float("inf"))
# # # # # #     cte = result.get("CTE_mean", float("inf"))

# # # # # #     if not np.isfinite(ate): ate = ade * 0.46
# # # # # #     if not np.isfinite(cte): cte = ade * 0.53

# # # # # #     score = (
# # # # # #         0.05 * (ade / 136.0)
# # # # # #         + 0.05 * (h12 / 50.0)
# # # # # #         + 0.10 * (h24 / 100.0)
# # # # # #         + 0.15 * (h48 / 200.0)
# # # # # #         + 0.35 * (h72 / 300.0)
# # # # # #         + 0.15 * (ate / 80.0)
# # # # # #         + 0.15 * (cte / 94.0)
# # # # # #     )
# # # # # #     return score * 100.0


# # # # # # def _beat_report(r: dict) -> str:
# # # # # #     ade = r.get("ADE", float("inf"))
# # # # # #     h72 = r.get("72h", float("inf"))
# # # # # #     ate = r.get("ATE_mean", float("inf"))
# # # # # #     cte = r.get("CTE_mean", float("inf"))
# # # # # #     parts = []
# # # # # #     if np.isfinite(ade) and ade < 136.41: parts.append(f"DPE✅{ade:.1f}")
# # # # # #     if np.isfinite(ate) and ate < 79.94:  parts.append(f"ATE✅{ate:.1f}")
# # # # # #     if np.isfinite(cte) and cte < 93.58:  parts.append(f"CTE✅{cte:.1f}")
# # # # # #     if np.isfinite(h72) and h72 < 297.0:  parts.append(f"72h✅{h72:.1f}")
# # # # # #     return "🏆 BEAT: " + " ".join(parts) if parts else ""


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Helpers — unchanged from v36
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


# # # # # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # # # # #                             collate_fn, num_workers):
# # # # # #     n   = len(val_dataset)
# # # # # #     rng = random.Random(42)
# # # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # # #     return DataLoader(Subset(val_dataset, idx),
# # # # # #                       batch_size=batch_size, shuffle=False,
# # # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # # def _get_ema_obj(model):
# # # # # #     if hasattr(model, '_ema') and model._ema is not None:
# # # # # #         return model._ema
# # # # # #     if hasattr(model, '_orig_mod'):
# # # # # #         orig = model._orig_mod
# # # # # #         if hasattr(orig, '_ema') and orig._ema is not None:
# # # # # #             return orig._ema
# # # # # #     return None


# # # # # # def _call_ema_update(model):
# # # # # #     if hasattr(model, '_orig_mod'):
# # # # # #         orig = model._orig_mod
# # # # # #         if hasattr(orig, 'ema_update'):
# # # # # #             orig.ema_update(); return
# # # # # #     if hasattr(model, 'ema_update'):
# # # # # #         model.ema_update()


# # # # # # def _get_raw_model(model):
# # # # # #     return model._orig_mod if hasattr(model, '_orig_mod') else model


# # # # # # def _save_checkpoint(path, epoch, model, optimizer, scheduler,
# # # # # #                      saver, avg_t, avg_vl, metrics=None):
# # # # # #     m = _get_raw_model(model)
# # # # # #     ema_obj = _get_ema_obj(model)
# # # # # #     ema_sd  = None
# # # # # #     if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # # # # #         try:
# # # # # #             ema_sd = {k: v.cpu().clone() for k, v in ema_obj.shadow.items()}
# # # # # #         except Exception:
# # # # # #             ema_sd = None

# # # # # #     payload = {
# # # # # #         "epoch"            : epoch,
# # # # # #         "model_state_dict" : m.state_dict(),
# # # # # #         "optimizer_state"  : optimizer.state_dict(),
# # # # # #         "scheduler_state"  : scheduler.state_dict(),
# # # # # #         "ema_shadow"       : ema_sd,
# # # # # #         "best_score"       : saver.best_score,
# # # # # #         "best_ade"         : saver.best_ade,
# # # # # #         "best_72h"         : saver.best_72h,
# # # # # #         "best_48h"         : saver.best_48h,
# # # # # #         "best_24h"         : saver.best_24h,
# # # # # #         "best_12h"         : saver.best_12h,
# # # # # #         "best_ate"         : getattr(saver, 'best_ate', float("inf")),
# # # # # #         "best_cte"         : getattr(saver, 'best_cte', float("inf")),
# # # # # #         "train_loss"       : avg_t,
# # # # # #         "val_loss"         : avg_vl,
# # # # # #     }
# # # # # #     if metrics:
# # # # # #         payload.update(metrics)
# # # # # #     torch.save(payload, path)


# # # # # # def _load_checkpoint(path, model, optimizer, scheduler, saver, device):
# # # # # #     if path is None or not os.path.exists(path):
# # # # # #         if path is not None:
# # # # # #             print(f"  ⚠  Checkpoint không tìm thấy: {path}")
# # # # # #         return 0

# # # # # #     print(f"  Loading checkpoint: {path}")
# # # # # #     ckpt = torch.load(path, map_location=device)

# # # # # #     m = _get_raw_model(model)
# # # # # #     missing, unexpected = m.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # # #     if missing:
# # # # # #         print(f"  ⚠  Missing keys ({len(missing)}): "
# # # # # #               f"{missing[:3]}{'...' if len(missing) > 3 else ''}")

# # # # # #     if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
# # # # # #         ema_obj = _get_ema_obj(model)
# # # # # #         if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # # # # #             restored = 0
# # # # # #             for k, v in ckpt["ema_shadow"].items():
# # # # # #                 if k in ema_obj.shadow:
# # # # # #                     ema_obj.shadow[k].copy_(v.to(device))
# # # # # #                     restored += 1
# # # # # #             print(f"  EMA shadow restored ({restored} keys)")

# # # # # #     if "optimizer_state" in ckpt:
# # # # # #         try:
# # # # # #             optimizer.load_state_dict(ckpt["optimizer_state"])
# # # # # #             for state in optimizer.state.values():
# # # # # #                 for k, v in state.items():
# # # # # #                     if torch.is_tensor(v):
# # # # # #                         state[k] = v.to(device)
# # # # # #             print("  Optimizer state restored")
# # # # # #         except Exception as e:
# # # # # #             print(f"  ⚠  Optimizer restore failed: {e}")

# # # # # #     if "scheduler_state" in ckpt:
# # # # # #         try:
# # # # # #             scheduler.load_state_dict(ckpt["scheduler_state"])
# # # # # #             print(f"  Scheduler restored (last_epoch={scheduler.last_epoch})")
# # # # # #         except Exception as e:
# # # # # #             print(f"  ⚠  Scheduler restore failed: {e}")

# # # # # #     if "best_score" in ckpt:
# # # # # #         saver.best_score = ckpt["best_score"]
# # # # # #         saver.best_ade   = ckpt.get("best_ade",  float("inf"))
# # # # # #         saver.best_72h   = ckpt.get("best_72h",  float("inf"))
# # # # # #         saver.best_48h   = ckpt.get("best_48h",  float("inf"))
# # # # # #         saver.best_24h   = ckpt.get("best_24h",  float("inf"))
# # # # # #         saver.best_12h   = ckpt.get("best_12h",  float("inf"))
# # # # # #         saver.best_ate   = ckpt.get("best_ate",  float("inf"))
# # # # # #         saver.best_cte   = ckpt.get("best_cte",  float("inf"))
# # # # # #         print(f"  Saver state restored (best_score={saver.best_score:.1f})")

# # # # # #     start_epoch = ckpt.get("epoch", 0) + 1
# # # # # #     print(f"  → Resuming from epoch {start_epoch}")
# # # # # #     return start_epoch


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  SWA — unchanged
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # class SWAManager:
# # # # # #     def __init__(self, model, start_epoch=60):
# # # # # #         self.start_epoch = start_epoch
# # # # # #         self.n_averaged  = 0
# # # # # #         self.avg_state   = None

# # # # # #     def update(self, model, epoch):
# # # # # #         if epoch < self.start_epoch: return
# # # # # #         m  = _get_raw_model(model)
# # # # # #         sd = {k: v.detach().clone() for k, v in m.state_dict().items()
# # # # # #               if v.dtype.is_floating_point}
# # # # # #         if self.avg_state is None:
# # # # # #             self.avg_state = sd; self.n_averaged = 1
# # # # # #         else:
# # # # # #             n = self.n_averaged
# # # # # #             for k in self.avg_state:
# # # # # #                 if k in sd:
# # # # # #                     self.avg_state[k].mul_(n / (n + 1)).add_(sd[k], alpha=1.0 / (n + 1))
# # # # # #             self.n_averaged += 1

# # # # # #     def apply_to(self, model):
# # # # # #         if self.avg_state is None: return None
# # # # # #         m  = _get_raw_model(model)
# # # # # #         sd = m.state_dict()
# # # # # #         backup = {k: sd[k].detach().clone() for k in self.avg_state if k in sd}
# # # # # #         for k, v in self.avg_state.items():
# # # # # #             if k in sd: sd[k].copy_(v)
# # # # # #         return backup

# # # # # #     def restore(self, model, backup):
# # # # # #         if backup is None: return
# # # # # #         m  = _get_raw_model(model)
# # # # # #         sd = m.state_dict()
# # # # # #         for k, v in backup.items():
# # # # # #             if k in sd: sd[k].copy_(v)


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Evaluation — unchanged từ v36
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def _eval_batch_atecte(pred_norm, gt_norm):
# # # # # #     pred_d = denorm_torch(pred_norm)
# # # # # #     gt_d   = denorm_torch(gt_norm)

# # # # # #     if HAS_ATECTE:
# # # # # #         dist, ate, cte = haversine_and_atecte_torch(pred_d, gt_d, unit_01deg=True)
# # # # # #         return dist, ate, cte
# # # # # #     else:
# # # # # #         dist = haversine_km_torch(pred_d, gt_d)
# # # # # #         return dist, None, None


# # # # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=15):
# # # # # #     model.eval()
# # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # #     t0  = time.perf_counter()
# # # # # #     n   = 0
# # # # # #     spread_per_step = []

# # # # # #     with torch.no_grad():
# # # # # #         for batch in loader:
# # # # # #             bl = move(list(batch), device)
# # # # # #             pred, _, all_trajs = model.sample(
# # # # # #                 bl, num_ensemble=fast_ensemble, ddim_steps=ode_steps,
# # # # # #                 importance_weight=True)
# # # # # #             T_active  = pred.shape[0]
# # # # # #             gt_sliced = bl[1][:T_active]

# # # # # #             dist, ate, cte = _eval_batch_atecte(pred, gt_sliced)
# # # # # #             acc.update(dist, ate_km=ate, cte_km=cte)

# # # # # #             step_spreads = []
# # # # # #             for t in range(all_trajs.shape[1]):
# # # # # #                 step_data = all_trajs[:, t, :, :]
# # # # # #                 spread = ((step_data[:, :, 0].std(0)**2
# # # # # #                            + step_data[:, :, 1].std(0)**2).sqrt() * 500.0).mean().item()
# # # # # #                 step_spreads.append(spread)
# # # # # #             spread_per_step.append(step_spreads)
# # # # # #             n += 1

# # # # # #     r = acc.compute()
# # # # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# # # # # #     if spread_per_step:
# # # # # #         spreads = np.array(spread_per_step)
# # # # # #         r["spread_72h_km"] = float(spreads[:, -1].mean())
# # # # # #     return r


# # # # # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # # # # #                            fast_ensemble, metrics_csv, epoch,
# # # # # #                            use_ema=False, ema_obj=None):
# # # # # #     backup = None
# # # # # #     if use_ema and ema_obj is not None:
# # # # # #         try:
# # # # # #             backup = ema_obj.apply_to(model)
# # # # # #         except Exception as e:
# # # # # #             print(f"  ⚠  EMA apply_to failed: {e}")
# # # # # #             backup  = None; use_ema = False

# # # # # #     model.eval()
# # # # # #     acc = StepErrorAccumulator(pred_len)
# # # # # #     t0  = time.perf_counter()
# # # # # #     n   = 0

# # # # # #     with torch.no_grad():
# # # # # #         for batch in val_loader:
# # # # # #             bl = move(list(batch), device)
# # # # # #             pred, _, _ = model.sample(
# # # # # #                 bl, num_ensemble=max(fast_ensemble, 20),
# # # # # #                 ddim_steps=max(ode_steps, 20),
# # # # # #                 importance_weight=True)
# # # # # #             T_pred = pred.shape[0]
# # # # # #             gt     = bl[1][:T_pred]
# # # # # #             dist, ate, cte = _eval_batch_atecte(pred, gt)
# # # # # #             acc.update(dist, ate_km=ate, cte_km=cte)
# # # # # #             n += 1

# # # # # #     r       = acc.compute()
# # # # # #     elapsed = time.perf_counter() - t0
# # # # # #     score   = _composite_score(r)
# # # # # #     tag     = "EMA" if use_ema else "RAW"

# # # # # #     ade_v = r.get("ADE",     float("nan"))
# # # # # #     fde_v = r.get("FDE",     float("nan"))
# # # # # #     h6_v  = r.get("6h",      float("nan"))
# # # # # #     h12_v = r.get("12h",     float("nan"))
# # # # # #     h24_v = r.get("24h",     float("nan"))
# # # # # #     h48_v = r.get("48h",     float("nan"))
# # # # # #     h72_v = r.get("72h",     float("nan"))
# # # # # #     ate_v = r.get("ATE_mean", float("nan"))
# # # # # #     cte_v = r.get("CTE_mean", float("nan"))

# # # # # #     def ind(v, tgt):
# # # # # #         if not np.isfinite(v): return ""
# # # # # #         return "✅" if v < tgt else "❌"

# # # # # #     print(f"\n{'='*70}")
# # # # # #     print(f"  [FULL VAL ADE ({tag})  ep={epoch}  {elapsed:.0f}s  {n} batches]")
# # # # # #     print(f"  ADE={ade_v:.1f} km {ind(ade_v,136.41)}  FDE={fde_v:.1f} km")
# # # # # #     print(f"  6h={h6_v:.0f}  12h={h12_v:.0f}{ind(h12_v,50)}  "
# # # # # #           f"24h={h24_v:.0f}{ind(h24_v,100)}  "
# # # # # #           f"48h={h48_v:.0f}{ind(h48_v,200)}  "
# # # # # #           f"72h={h72_v:.0f}{ind(h72_v,300)} km")
# # # # # #     if np.isfinite(ate_v):
# # # # # #         print(f"  ATE={ate_v:.1f}{ind(ate_v,79.94)}  "
# # # # # #               f"CTE={cte_v:.1f}{ind(cte_v,93.58)}  "
# # # # # #               f"[ST-Trans: ATE=79.94 CTE=93.58]")
# # # # # #     beat = _beat_report(r)
# # # # # #     if beat: print(f"  {beat}")
# # # # # #     print(f"  Score v2 = {score:.2f}  (< 100 = beat ST-Trans all metrics)")
# # # # # #     print(f"{'='*70}\n")

# # # # # #     from datetime import datetime
# # # # # #     dm = DatasetMetrics(
# # # # # #         ade          = ade_v if np.isfinite(ade_v) else 0.0,
# # # # # #         fde          = fde_v if np.isfinite(fde_v) else 0.0,
# # # # # #         ugde_12h     = h12_v if np.isfinite(h12_v) else 0.0,
# # # # # #         ugde_24h     = h24_v if np.isfinite(h24_v) else 0.0,
# # # # # #         ugde_48h     = h48_v if np.isfinite(h48_v) else 0.0,
# # # # # #         ugde_72h     = h72_v if np.isfinite(h72_v) else 0.0,
# # # # # #         ate_abs_mean = ate_v if np.isfinite(ate_v) else 0.0,
# # # # # #         cte_abs_mean = cte_v if np.isfinite(cte_v) else 0.0,
# # # # # #         n_total      = r.get("n_samples", 0),
# # # # # #         timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # # #     )
# # # # # #     save_metrics_csv(dm, metrics_csv, tag=f"val_full_{tag}_ep{epoch:03d}")

# # # # # #     if backup is not None:
# # # # # #         try:
# # # # # #             ema_obj.restore(model, backup)
# # # # # #         except Exception as e:
# # # # # #             print(f"  ⚠  EMA restore failed: {e}")

# # # # # #     return r


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  BestModelSaver — unchanged from v36
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # class HorizonAwareBestSaver:
# # # # # #     def __init__(self, patience=30, tol=1.5):
# # # # # #         self.patience   = patience
# # # # # #         self.tol        = tol
# # # # # #         self.counter    = 0
# # # # # #         self.early_stop = False
# # # # # #         self.best_score = float("inf")
# # # # # #         self.best_ade   = float("inf")
# # # # # #         self.best_12h   = float("inf")
# # # # # #         self.best_24h   = float("inf")
# # # # # #         self.best_48h   = float("inf")
# # # # # #         self.best_72h   = float("inf")
# # # # # #         self.best_ate   = float("inf")
# # # # # #         self.best_cte   = float("inf")

# # # # # #     def update(self, r, model, out_dir, epoch, optimizer, scheduler,
# # # # # #                tl, vl, saver_ref, min_epochs=30):
# # # # # #         ade   = r.get("ADE", float("inf"))
# # # # # #         h12   = r.get("12h", float("inf"))
# # # # # #         h24   = r.get("24h", float("inf"))
# # # # # #         h48   = r.get("48h", float("inf"))
# # # # # #         h72   = r.get("72h", float("inf"))
# # # # # #         ate   = r.get("ATE_mean", ade * 0.46 if np.isfinite(ade) else float("inf"))
# # # # # #         cte   = r.get("CTE_mean", ade * 0.53 if np.isfinite(ade) else float("inf"))
# # # # # #         score = _composite_score(r)

# # # # # #         improved_any = False

# # # # # #         if ade < self.best_ade:
# # # # # #             self.best_ade = ade; improved_any = True
# # # # # #             _save_checkpoint(
# # # # # #                 os.path.join(out_dir, "best_ade.pth"),
# # # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # # #                 {"ade": ade, "tag": "best_ade"})

# # # # # #         if h72 < self.best_72h:
# # # # # #             self.best_72h = h72; improved_any = True
# # # # # #             _save_checkpoint(
# # # # # #                 os.path.join(out_dir, "best_72h.pth"),
# # # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # # #                 {"h72": h72, "tag": "best_72h"})

# # # # # #         if ate < self.best_ate:
# # # # # #             self.best_ate = ate; improved_any = True
# # # # # #             _save_checkpoint(
# # # # # #                 os.path.join(out_dir, "best_ate.pth"),
# # # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # # #                 {"ate": ate, "cte": cte, "tag": "best_ate"})

# # # # # #         if cte < self.best_cte:
# # # # # #             self.best_cte = cte; improved_any = True
# # # # # #             _save_checkpoint(
# # # # # #                 os.path.join(out_dir, "best_cte.pth"),
# # # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # # #                 {"ate": ate, "cte": cte, "tag": "best_cte"})

# # # # # #         if h48 < self.best_48h: self.best_48h = h48; improved_any = True
# # # # # #         if h24 < self.best_24h: self.best_24h = h24; improved_any = True
# # # # # #         if h12 < self.best_12h: self.best_12h = h12; improved_any = True

# # # # # #         if score < self.best_score - self.tol:
# # # # # #             self.best_score = score
# # # # # #             self.counter    = 0
# # # # # #             _save_checkpoint(
# # # # # #                 os.path.join(out_dir, "best_model.pth"),
# # # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # # #                 {"ade": ade, "h12": h12, "h24": h24, "h48": h48, "h72": h72,
# # # # # #                  "ate": ate, "cte": cte,
# # # # # #                  "composite_score": score, "tag": "best_composite"})
# # # # # #             print(f"  ✅ Best COMPOSITE={score:.2f}  "
# # # # # #                   f"ADE={ade:.1f}  12h={h12:.0f}  24h={h24:.0f}  "
# # # # # #                   f"48h={h48:.0f}  72h={h72:.0f}  "
# # # # # #                   f"ATE={ate:.1f}  CTE={cte:.1f}  (ep {epoch})")
# # # # # #         else:
# # # # # #             if not improved_any:
# # # # # #                 self.counter += 1
# # # # # #             print(f"  No improvement {self.counter}/{self.patience}"
# # # # # #                   f"  (best={self.best_score:.2f} cur={score:.2f})"
# # # # # #                   f"  72h={h72:.0f}↓{self.best_72h:.0f}"
# # # # # #                   f"  ATE={ate:.1f}↓{self.best_ate:.1f}")

# # # # # #         if epoch >= min_epochs and self.counter >= self.patience:
# # # # # #             self.early_stop = True


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Env diagnostic — unchanged
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def _check_env(bl, train_dataset):
# # # # # #     try:    env_dir = train_dataset.env_path
# # # # # #     except AttributeError:
# # # # # #         try:    env_dir = train_dataset.dataset.env_path
# # # # # #         except: env_dir = "UNKNOWN"
# # # # # #     print(f"  Env path: {env_dir}")
# # # # # #     env_data = bl[13]
# # # # # #     if env_data is None:
# # # # # #         print("  ⚠️  env_data is None"); return
# # # # # #     for key in ("gph500_mean", "gph500_center", "u500_mean", "v500_mean"):
# # # # # #         if key not in env_data:
# # # # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # # # #         v    = env_data[key]
# # # # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # # # #         print(f"  {'✅' if zero < 80 else '⚠️'}  {key}: "
# # # # # #               f"mean={v.mean().item():.4f} std={v.std().item():.4f} zero={zero:.1f}%")


# # # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # # #  Args
# # # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # # def get_args():
# # # # # #     p = argparse.ArgumentParser(
# # # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # # # # #     p.add_argument("--obs_len",         default=8,          type=int)
# # # # # #     p.add_argument("--pred_len",        default=12,         type=int)
# # # # # #     p.add_argument("--batch_size",      default=32,         type=int)
# # # # # #     p.add_argument("--num_epochs",      default=100,        type=int)
# # # # # #     p.add_argument("--g_learning_rate", default=3e-4,       type=float)
# # # # # #     p.add_argument("--weight_decay",    default=1e-4,       type=float)
# # # # # #     p.add_argument("--warmup_epochs",   default=5,          type=int)
# # # # # #     p.add_argument("--grad_clip",       default=1.0,        type=float)
# # # # # #     p.add_argument("--patience",        default=30,         type=int)
# # # # # #     p.add_argument("--min_epochs",      default=30,         type=int)
# # # # # #     p.add_argument("--n_train_ens",     default=4,          type=int)
# # # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # # #     p.add_argument("--num_workers",     default=2,          type=int)

# # # # # #     p.add_argument("--sigma_min",            default=0.02,  type=float)
# # # # # #     p.add_argument("--ctx_noise_scale",      default=0.01,  type=float)
# # # # # #     p.add_argument("--initial_sample_sigma", default=0.03,  type=float)

# # # # # #     p.add_argument("--ode_steps_train", default=10,  type=int)
# # # # # #     p.add_argument("--ode_steps_val",   default=10,  type=int)
# # # # # #     p.add_argument("--ode_steps_test",  default=10,  type=int)

# # # # # #     p.add_argument("--val_ensemble",    default=20,  type=int)
# # # # # #     p.add_argument("--fast_ensemble",   default=10,  type=int)

# # # # # #     p.add_argument("--val_freq",        default=5,   type=int)
# # # # # #     p.add_argument("--val_ade_freq",    default=5,   type=int)
# # # # # #     p.add_argument("--val_subset_size", default=500, type=int)

# # # # # #     p.add_argument("--use_ema",         action="store_true", default=True)
# # # # # #     p.add_argument("--ema_decay",       default=0.990, type=float,  # v37: 0.992→0.990
# # # # # #                    help="EMA decay. 0.990 cho ATE optimization")
# # # # # #     p.add_argument("--swa_start_epoch", default=60,   type=int)
# # # # # #     p.add_argument("--teacher_forcing", action="store_true", default=True)

# # # # # #     p.add_argument("--output_dir",  default="runs/v37",        type=str)
# # # # # #     p.add_argument("--metrics_csv", default="metrics_v37.csv", type=str)
# # # # # #     p.add_argument("--predict_csv", default="predictions.csv", type=str)

# # # # # #     p.add_argument("--gpu_num",     default="0", type=str)
# # # # # #     p.add_argument("--delim",       default=" ")
# # # # # #     p.add_argument("--skip",        default=1,   type=int)
# # # # # #     p.add_argument("--min_ped",     default=1,   type=int)
# # # # # #     p.add_argument("--threshold",   default=0.002, type=float)
# # # # # #     p.add_argument("--other_modal", default="gph")
# # # # # #     p.add_argument("--test_year",   default=None, type=int)

# # # # # #     p.add_argument("--resume",       default=None, type=str)
# # # # # #     p.add_argument("--resume_epoch", default=None, type=int)

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

# # # # # #     print("=" * 72)
# # # # # #     print("  TC-FlowMatching v37  |  ATE-FOCUSED EDITION")
# # # # # #     print("  STATUS v36: CTE=69.0 ✅  ATE=168.6 ❌  72h=395 ❌")
# # # # # #     print("  TARGET v37: ATE<79.94  72h<297  (CTE already ✅)")
# # # # # #     print("  NEW: along_track_speed_loss | ate_weight=2x | vmax=95km/h")
# # # # # #     print("  EMA decay:", args.ema_decay, "(0.990 = faster update)")
# # # # # #     print("=" * 72)

# # # # # #     # ── Data ──────────────────────────────────────────────────────────────
# # # # # #     train_dataset, train_loader = data_loader(
# # # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # # #     val_dataset, val_loader = data_loader(
# # # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # # #     val_subset_loader = make_val_subset_loader(
# # # # # #         val_dataset, args.val_subset_size, args.batch_size,
# # # # # #         seq_collate, args.num_workers)

# # # # # #     test_loader = None
# # # # # #     try:
# # # # # #         _, test_loader = data_loader(
# # # # # #             args, {"root": args.dataset_root, "type": "test"},
# # # # # #             test=True, test_year=None)
# # # # # #     except Exception as e:
# # # # # #         print(f"  Warning: test loader: {e}")

# # # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # # #     print(f"  val   : {len(val_dataset)} seq")

# # # # # #     # ── Model ─────────────────────────────────────────────────────────────
# # # # # #     model = TCFlowMatching(
# # # # # #         pred_len             = args.pred_len,
# # # # # #         obs_len              = args.obs_len,
# # # # # #         sigma_min            = args.sigma_min,
# # # # # #         n_train_ens          = args.n_train_ens,
# # # # # #         ctx_noise_scale      = args.ctx_noise_scale,
# # # # # #         initial_sample_sigma = args.initial_sample_sigma,
# # # # # #         teacher_forcing      = args.teacher_forcing,
# # # # # #         use_ema              = args.use_ema,
# # # # # #         ema_decay            = args.ema_decay,
# # # # # #     ).to(device)

# # # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # # #     print(f"  params  : {n_params:,}")

# # # # # #     model.init_ema()
# # # # # #     print(f"  EMA     : {'ON' if model._ema is not None else 'OFF'}"
# # # # # #           f"  (decay={args.ema_decay})")

# # # # # #     # ── Optimizer + Scheduler ─────────────────────────────────────────────
# # # # # #     optimizer = optim.AdamW(
# # # # # #         model.parameters(),
# # # # # #         lr=args.g_learning_rate,
# # # # # #         weight_decay=args.weight_decay,
# # # # # #     )

# # # # # #     saver  = HorizonAwareBestSaver(patience=args.patience, tol=1.5)
# # # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # # # #     swa    = SWAManager(model, start_epoch=args.swa_start_epoch)

# # # # # #     steps_per_epoch = len(train_loader)
# # # # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # # # #     scheduler = get_cosine_schedule_with_warmup(
# # # # # #         optimizer, warmup, total_steps, min_lr=1e-6)

# # # # # #     # ── Resume ────────────────────────────────────────────────────────────
# # # # # #     has_scheduler_state    = False
# # # # # #     _ckpt_scheduler_state  = None
# # # # # #     start_epoch = 0

# # # # # #     if args.resume is not None and os.path.exists(args.resume):
# # # # # #         print(f"  Loading checkpoint: {args.resume}")
# # # # # #         ckpt = torch.load(args.resume, map_location=device)

# # # # # #         m = _get_raw_model(model)
# # # # # #         missing, _ = m.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # # #         if missing:
# # # # # #             print(f"  ⚠  Missing keys ({len(missing)}): "
# # # # # #                   f"{missing[:3]}{'...' if len(missing)>3 else ''}")

# # # # # #         if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
# # # # # #             ema_obj = _get_ema_obj(model)
# # # # # #             if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # # # # #                 restored = sum(
# # # # # #                     1 for k, v in ckpt["ema_shadow"].items()
# # # # # #                     if k in ema_obj.shadow
# # # # # #                     and not ema_obj.shadow[k].copy_(v.to(device)) is None
# # # # # #                 )
# # # # # #                 print(f"  EMA shadow restored ({restored} keys)")

# # # # # #         if "optimizer_state" in ckpt:
# # # # # #             try:
# # # # # #                 optimizer.load_state_dict(ckpt["optimizer_state"])
# # # # # #                 for state in optimizer.state.values():
# # # # # #                     for k, v in state.items():
# # # # # #                         if torch.is_tensor(v): state[k] = v.to(device)
# # # # # #                 print("  Optimizer state restored")
# # # # # #             except Exception as e:
# # # # # #                 print(f"  ⚠  Optimizer restore failed: {e}")

# # # # # #         has_scheduler_state   = "scheduler_state" in ckpt
# # # # # #         _ckpt_scheduler_state = ckpt.get("scheduler_state", None)

# # # # # #         if "best_score" in ckpt:
# # # # # #             saver.best_score = ckpt["best_score"]
# # # # # #             saver.best_ade   = ckpt.get("best_ade",  float("inf"))
# # # # # #             saver.best_72h   = ckpt.get("best_72h",  float("inf"))
# # # # # #             saver.best_48h   = ckpt.get("best_48h",  float("inf"))
# # # # # #             saver.best_24h   = ckpt.get("best_24h",  float("inf"))
# # # # # #             saver.best_12h   = ckpt.get("best_12h",  float("inf"))
# # # # # #             saver.best_ate   = ckpt.get("best_ate",  float("inf"))
# # # # # #             saver.best_cte   = ckpt.get("best_cte",  float("inf"))
# # # # # #             print(f"  Saver state restored (best_score={saver.best_score:.2f})")

# # # # # #         start_epoch = (args.resume_epoch if args.resume_epoch is not None
# # # # # #                        else ckpt.get("epoch", 0) + 1)
# # # # # #         print(f"  → Resuming from epoch {start_epoch}")

# # # # # #     elif args.resume is not None:
# # # # # #         print(f"  ⚠  Checkpoint không tìm thấy: {args.resume}")

# # # # # #     # ── torch.compile ─────────────────────────────────────────────────────
# # # # # #     try:
# # # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # # #         print("  torch.compile: enabled")
# # # # # #     except Exception:
# # # # # #         pass

# # # # # #     # ── Scheduler restore/advance ─────────────────────────────────────────
# # # # # #     if has_scheduler_state and _ckpt_scheduler_state is not None:
# # # # # #         try:
# # # # # #             scheduler.load_state_dict(_ckpt_scheduler_state)
# # # # # #             print(f"  Scheduler restored (last_epoch={scheduler.last_epoch})")
# # # # # #         except Exception as e:
# # # # # #             print(f"  ⚠  Scheduler restore failed: {e} — advancing manually")
# # # # # #             for _ in range(start_epoch * steps_per_epoch):
# # # # # #                 scheduler.step()
# # # # # #     elif start_epoch > 0:
# # # # # #         print(f"  Advancing scheduler {start_epoch * steps_per_epoch} steps...")
# # # # # #         for _ in range(start_epoch * steps_per_epoch):
# # # # # #             scheduler.step()

# # # # # #     print("=" * 72)
# # # # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch, start={start_epoch})")
# # # # # #     print("=" * 72)

# # # # # #     epoch_times = []
# # # # # #     train_start = time.perf_counter()

# # # # # #     for epoch in range(start_epoch, args.num_epochs):
# # # # # #         model.train()
# # # # # #         sum_loss = 0.0
# # # # # #         t0 = time.perf_counter()

# # # # # #         for i, batch in enumerate(train_loader):
# # # # # #             bl = move(list(batch), device)

# # # # # #             if epoch == start_epoch and i == 0:
# # # # # #                 _check_env(bl, train_dataset)

# # # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # #                 bd = model.get_loss_breakdown(bl, epoch=epoch)

# # # # # #             optimizer.zero_grad()
# # # # # #             scaler.scale(bd["total"]).backward()
# # # # # #             scaler.unscale_(optimizer)
# # # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # # #             scaler.step(optimizer)
# # # # # #             scaler.update()
# # # # # #             scheduler.step()

# # # # # #             _call_ema_update(model)
# # # # # #             sum_loss += bd["total"].item()

# # # # # #             if i % 20 == 0:
# # # # # #                 lr = optimizer.param_groups[0]["lr"]
# # # # # #                 # v37: log thêm "alng" (along_track)
# # # # # #                 print(
# # # # # #                     f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # # # #                     f"  tot={bd['total'].item():.3f}"
# # # # # #                     f"  fm={bd.get('fm_mse', 0):.3f}"
# # # # # #                     f"  hor={bd.get('mse_hav', 0):.3f}"
# # # # # #                     f"  end={bd.get('endpoint', 0):.3f}"
# # # # # #                     f"  vel={bd.get('vel_smooth', 0):.3f}"
# # # # # #                     f"  atc={bd.get('ate_cte', 0):.3f}"
# # # # # #                     f"  alng={bd.get('along_track', 0):.3f}"   # ← v37 NEW
# # # # # #                     f"  hd={bd.get('heading', 0):.3f}"
# # # # # #                     f"  str={bd.get('steering', 0):.3f}"
# # # # # #                     f"  lr={lr:.2e}"
# # # # # #                 )

# # # # # #         ep_s  = time.perf_counter() - t0
# # # # # #         epoch_times.append(ep_s)
# # # # # #         avg_t = sum_loss / len(train_loader)

# # # # # #         swa.update(model, epoch)

# # # # # #         # Val loss
# # # # # #         model.eval()
# # # # # #         val_loss = 0.0
# # # # # #         with torch.no_grad():
# # # # # #             for batch in val_loader:
# # # # # #                 bl_v = move(list(batch), device)
# # # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # # #                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
# # # # # #         avg_vl = val_loss / len(val_loader)

# # # # # #         print(f"  Epoch {epoch:>3}  train={avg_t:.4f}  val={avg_vl:.4f}"
# # # # # #               f"  time={ep_s:.0f}s")

# # # # # #         # ── Fast eval (mỗi epoch) ─────────────────────────────────────────
# # # # # #         m_fast = evaluate_fast(model, val_subset_loader, device,
# # # # # #                                 args.ode_steps_train, args.pred_len,
# # # # # #                                 args.fast_ensemble)

# # # # # #         h6  = m_fast.get("6h",  float("nan"))
# # # # # #         h12 = m_fast.get("12h", float("nan"))
# # # # # #         h24 = m_fast.get("24h", float("nan"))
# # # # # #         h48 = m_fast.get("48h", float("nan"))
# # # # # #         h72 = m_fast.get("72h", float("nan"))
# # # # # #         ate = m_fast.get("ATE_mean", float("nan"))
# # # # # #         cte = m_fast.get("CTE_mean", float("nan"))
# # # # # #         fast_score = _composite_score(m_fast)

# # # # # #         def ind(v, tgt):
# # # # # #             if not np.isfinite(v): return "?"
# # # # # #             return "🎯" if v < tgt else "❌"

# # # # # #         print(
# # # # # #             f"  [FAST ep{epoch}]"
# # # # # #             f"  ADE={m_fast.get('ADE', float('nan')):.1f}"
# # # # # #             f"  12h={h12:.0f}{ind(h12,50)}"
# # # # # #             f"  24h={h24:.0f}{ind(h24,100)}"
# # # # # #             f"  48h={h48:.0f}{ind(h48,200)}"
# # # # # #             f"  72h={h72:.0f}{ind(h72,300)}"
# # # # # #             f"  ATE={ate:.1f}{ind(ate,79.94)}"
# # # # # #             f"  CTE={cte:.1f}{ind(cte,93.58)}"
# # # # # #             f"  score={fast_score:.2f}"
# # # # # #         )

# # # # # #         # ── Full val eval (mỗi val_freq epoch) ───────────────────────────
# # # # # #         if epoch % args.val_freq == 0:
# # # # # #             ema_obj = _get_ema_obj(model)
# # # # # #             try:
# # # # # #                 r_raw = evaluate_full_val_ade(
# # # # # #                     model, val_loader, device,
# # # # # #                     ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # # # #                     fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # # #                     epoch=epoch, use_ema=False, ema_obj=None)

# # # # # #                 r_use = r_raw
# # # # # #                 if ema_obj is not None and epoch >= 5:
# # # # # #                     r_ema = evaluate_full_val_ade(
# # # # # #                         model, val_loader, device,
# # # # # #                         ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # # # #                         fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # # #                         epoch=epoch, use_ema=True, ema_obj=ema_obj)
# # # # # #                     if _composite_score(r_ema) < _composite_score(r_raw):
# # # # # #                         r_use = r_ema

# # # # # #                 saver.update(r_use, model, args.output_dir, epoch,
# # # # # #                              optimizer, scheduler, avg_t, avg_vl,
# # # # # #                              saver_ref=saver, min_epochs=args.min_epochs)
# # # # # #             except Exception as e:
# # # # # #                 print(f"  ⚠  Full val failed: {e}")
# # # # # #                 import traceback; traceback.print_exc()

# # # # # #         # ── Checkpoint định kỳ ────────────────────────────────────────────
# # # # # #         if epoch % 10 == 0 or epoch == args.num_epochs - 1:
# # # # # #             _save_checkpoint(
# # # # # #                 os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"),
# # # # # #                 epoch, model, optimizer, scheduler, saver, avg_t, avg_vl,
# # # # # #                 {"ade": m_fast.get("ADE", float("nan")),
# # # # # #                  "h48": h48, "h72": h72,
# # # # # #                  "ate": ate, "cte": cte})

# # # # # #         if epoch % 10 == 9:
# # # # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")
# # # # # #             print(f"  📊 Best so far: score={saver.best_score:.2f}  "
# # # # # #                   f"72h={saver.best_72h:.0f}km  ATE={saver.best_ate:.1f}km  "
# # # # # #                   f"CTE={saver.best_cte:.1f}km")

# # # # # #         if saver.early_stop:
# # # # # #             print(f"  ⛔ Early stopping at epoch {epoch}")
# # # # # #             break

# # # # # #     # ── Final: SWA eval ───────────────────────────────────────────────────
# # # # # #     print(f"\n{'='*72}")
# # # # # #     swa_backup = swa.apply_to(model)
# # # # # #     if swa_backup is not None:
# # # # # #         print("  Evaluating SWA weights...")
# # # # # #         r_swa = evaluate_full_val_ade(
# # # # # #             model, val_loader, device,
# # # # # #             ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # # # #             fast_ensemble=args.val_ensemble, metrics_csv=metrics_csv,
# # # # # #             epoch=9999, use_ema=False)
# # # # # #         if _composite_score(r_swa) < saver.best_score:
# # # # # #             _save_checkpoint(
# # # # # #                 os.path.join(args.output_dir, "best_swa.pth"),
# # # # # #                 9999, model, optimizer, scheduler, saver,
# # # # # #                 avg_t=0.0, avg_vl=0.0, metrics={"tag": "best_swa"})
# # # # # #             print("  ✅ SWA checkpoint saved")
# # # # # #         swa.restore(model, swa_backup)

# # # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # # #     print(f"\n  Best composite score v2: {saver.best_score:.2f}")
# # # # # #     print(f"  Best metrics: ADE={saver.best_ade:.1f}  "
# # # # # #           f"12h={saver.best_12h:.0f}  24h={saver.best_24h:.0f}  "
# # # # # #           f"48h={saver.best_48h:.0f}  72h={saver.best_72h:.0f}  "
# # # # # #           f"ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}")
# # # # # #     print(f"  Total time: {total_h:.2f}h")

# # # # # #     best_r = {
# # # # # #         "ADE": saver.best_ade, "12h": saver.best_12h,
# # # # # #         "24h": saver.best_24h, "48h": saver.best_48h,
# # # # # #         "72h": saver.best_72h, "ATE_mean": saver.best_ate,
# # # # # #         "CTE_mean": saver.best_cte,
# # # # # #     }
# # # # # #     beat = _beat_report(best_r)
# # # # # #     if beat:
# # # # # #         print(f"\n  🏆🏆🏆 {beat} 🏆🏆🏆")
# # # # # #     else:
# # # # # #         print(f"\n  ST-Trans targets not yet beaten. Best score={saver.best_score:.2f}")
# # # # # #         print(f"  Need: DPE<136.41, ATE<79.94, CTE<93.58, 72h<297")
# # # # # #     print("=" * 72)


# # # # # # if __name__ == "__main__":
# # # # # #     args = get_args()
# # # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # # #     if torch.cuda.is_available():
# # # # # #         torch.cuda.manual_seed_all(42)
# # # # # #     main(args)

# # # # # """
# # # # # scripts/train_flowmatching.py — v39  BEAT ST-TRANS EDITION
# # # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # # THAY ĐỔI DUY NHẤT so với v36 train script (doc 5):

# # # # #   Log print line — update key names cho losses v39:
# # # # #     ĐỔI:  vel={vel_smooth}  atc={ate_cte}  str={steering}
# # # # #     THÀNH: spd={speed_acc}  cml={cumul_disp}  vsm={vel_smooth}

# # # # #   Tất cả code khác GIỐNG HỆT v36 train script.
# # # # # """
# # # # # from __future__ import annotations

# # # # # import sys
# # # # # import os
# # # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # # import argparse
# # # # # import time
# # # # # import random

# # # # # import numpy as np
# # # # # import torch
# # # # # import torch.optim as optim
# # # # # from torch.amp import autocast, GradScaler
# # # # # from torch.utils.data import DataLoader, Subset

# # # # # from Model.data.loader_training import data_loader
# # # # # from Model.flow_matching_model import TCFlowMatching
# # # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # # from utils.metrics import (
# # # # #     StepErrorAccumulator,
# # # # #     save_metrics_csv,
# # # # #     haversine_km_torch,
# # # # #     denorm_torch,
# # # # #     HORIZON_STEPS,
# # # # #     DatasetMetrics,
# # # # # )

# # # # # try:
# # # # #     from utils.metrics import haversine_and_atecte_torch
# # # # #     HAS_ATECTE = True
# # # # # except ImportError:
# # # # #     HAS_ATECTE = False
# # # # #     print("  ⚠  haversine_and_atecte_torch not found — ATE/CTE disabled")


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Composite Score v2
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def _composite_score(result: dict) -> float:
# # # # #     ade = result.get("ADE", float("inf"))
# # # # #     h12 = result.get("12h", float("inf"))
# # # # #     h24 = result.get("24h", float("inf"))
# # # # #     h48 = result.get("48h", float("inf"))
# # # # #     h72 = result.get("72h", float("inf"))
# # # # #     ate = result.get("ATE_mean", float("inf"))
# # # # #     cte = result.get("CTE_mean", float("inf"))
# # # # #     if not np.isfinite(ate): ate = ade * 0.46
# # # # #     if not np.isfinite(cte): cte = ade * 0.53
# # # # #     return 100.0 * (
# # # # #         0.05 * (ade / 136.0)
# # # # #         + 0.05 * (h12 / 50.0)
# # # # #         + 0.10 * (h24 / 100.0)
# # # # #         + 0.15 * (h48 / 200.0)
# # # # #         + 0.35 * (h72 / 300.0)
# # # # #         + 0.15 * (ate / 80.0)
# # # # #         + 0.15 * (cte / 94.0)
# # # # #     )


# # # # # def _beat_report(r: dict) -> str:
# # # # #     parts = []
# # # # #     if np.isfinite(r.get("ADE", float("inf"))) and r["ADE"] < 136.41:
# # # # #         parts.append(f"DPE✅{r['ADE']:.1f}")
# # # # #     if np.isfinite(r.get("ATE_mean", float("inf"))) and r["ATE_mean"] < 79.94:
# # # # #         parts.append(f"ATE✅{r['ATE_mean']:.1f}")
# # # # #     if np.isfinite(r.get("CTE_mean", float("inf"))) and r["CTE_mean"] < 93.58:
# # # # #         parts.append(f"CTE✅{r['CTE_mean']:.1f}")
# # # # #     if np.isfinite(r.get("72h", float("inf"))) and r["72h"] < 297.0:
# # # # #         parts.append(f"72h✅{r['72h']:.1f}")
# # # # #     return "🏆 BEAT: " + " ".join(parts) if parts else ""


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Helpers — unchanged
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


# # # # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # # # #                             collate_fn, num_workers):
# # # # #     n   = len(val_dataset)
# # # # #     rng = random.Random(42)
# # # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # # #     return DataLoader(Subset(val_dataset, idx),
# # # # #                       batch_size=batch_size, shuffle=False,
# # # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # # def _get_ema_obj(model):
# # # # #     if hasattr(model, '_ema') and model._ema is not None:
# # # # #         return model._ema
# # # # #     if hasattr(model, '_orig_mod'):
# # # # #         orig = model._orig_mod
# # # # #         if hasattr(orig, '_ema') and orig._ema is not None:
# # # # #             return orig._ema
# # # # #     return None


# # # # # def _call_ema_update(model):
# # # # #     if hasattr(model, '_orig_mod'):
# # # # #         orig = model._orig_mod
# # # # #         if hasattr(orig, 'ema_update'):
# # # # #             orig.ema_update(); return
# # # # #     if hasattr(model, 'ema_update'):
# # # # #         model.ema_update()


# # # # # def _get_raw_model(model):
# # # # #     return model._orig_mod if hasattr(model, '_orig_mod') else model


# # # # # def _save_checkpoint(path, epoch, model, optimizer, scheduler,
# # # # #                      saver, avg_t, avg_vl, metrics=None):
# # # # #     m = _get_raw_model(model)
# # # # #     ema_obj = _get_ema_obj(model)
# # # # #     ema_sd  = None
# # # # #     if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # # # #         try:
# # # # #             ema_sd = {k: v.cpu().clone() for k, v in ema_obj.shadow.items()}
# # # # #         except Exception:
# # # # #             ema_sd = None
# # # # #     payload = {
# # # # #         "epoch": epoch, "model_state_dict": m.state_dict(),
# # # # #         "optimizer_state": optimizer.state_dict(),
# # # # #         "scheduler_state": scheduler.state_dict(),
# # # # #         "ema_shadow": ema_sd,
# # # # #         "best_score": saver.best_score, "best_ade": saver.best_ade,
# # # # #         "best_72h": saver.best_72h, "best_48h": saver.best_48h,
# # # # #         "best_24h": saver.best_24h, "best_12h": saver.best_12h,
# # # # #         "best_ate": getattr(saver, 'best_ate', float("inf")),
# # # # #         "best_cte": getattr(saver, 'best_cte', float("inf")),
# # # # #         "train_loss": avg_t, "val_loss": avg_vl,
# # # # #     }
# # # # #     if metrics: payload.update(metrics)
# # # # #     torch.save(payload, path)


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  SWA — unchanged
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class SWAManager:
# # # # #     def __init__(self, model, start_epoch=60):
# # # # #         self.start_epoch = start_epoch
# # # # #         self.n_averaged  = 0
# # # # #         self.avg_state   = None

# # # # #     def update(self, model, epoch):
# # # # #         if epoch < self.start_epoch: return
# # # # #         m  = _get_raw_model(model)
# # # # #         sd = {k: v.detach().clone() for k, v in m.state_dict().items()
# # # # #               if v.dtype.is_floating_point}
# # # # #         if self.avg_state is None:
# # # # #             self.avg_state = sd; self.n_averaged = 1
# # # # #         else:
# # # # #             n = self.n_averaged
# # # # #             for k in self.avg_state:
# # # # #                 if k in sd:
# # # # #                     self.avg_state[k].mul_(n/(n+1)).add_(sd[k], alpha=1.0/(n+1))
# # # # #             self.n_averaged += 1

# # # # #     def apply_to(self, model):
# # # # #         if self.avg_state is None: return None
# # # # #         m  = _get_raw_model(model)
# # # # #         sd = m.state_dict()
# # # # #         backup = {k: sd[k].detach().clone() for k in self.avg_state if k in sd}
# # # # #         for k, v in self.avg_state.items():
# # # # #             if k in sd: sd[k].copy_(v)
# # # # #         return backup

# # # # #     def restore(self, model, backup):
# # # # #         if backup is None: return
# # # # #         m  = _get_raw_model(model)
# # # # #         sd = m.state_dict()
# # # # #         for k, v in backup.items():
# # # # #             if k in sd: sd[k].copy_(v)


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Evaluation — unchanged
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def _eval_batch_atecte(pred_norm, gt_norm):
# # # # #     pred_d = denorm_torch(pred_norm)
# # # # #     gt_d   = denorm_torch(gt_norm)
# # # # #     if HAS_ATECTE:
# # # # #         dist, ate, cte = haversine_and_atecte_torch(pred_d, gt_d, unit_01deg=True)
# # # # #         return dist, ate, cte
# # # # #     else:
# # # # #         dist = haversine_km_torch(pred_d, gt_d)
# # # # #         return dist, None, None


# # # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=15):
# # # # #     model.eval()
# # # # #     acc = StepErrorAccumulator(pred_len)
# # # # #     t0  = time.perf_counter()
# # # # #     n   = 0
# # # # #     spread_per_step = []
# # # # #     with torch.no_grad():
# # # # #         for batch in loader:
# # # # #             bl = move(list(batch), device)
# # # # #             pred, _, all_trajs = model.sample(
# # # # #                 bl, num_ensemble=fast_ensemble, ddim_steps=ode_steps,
# # # # #                 importance_weight=True)
# # # # #             T_active  = pred.shape[0]
# # # # #             gt_sliced = bl[1][:T_active]
# # # # #             dist, ate, cte = _eval_batch_atecte(pred, gt_sliced)
# # # # #             acc.update(dist, ate_km=ate, cte_km=cte)
# # # # #             step_spreads = []
# # # # #             for t in range(all_trajs.shape[1]):
# # # # #                 step_data = all_trajs[:, t, :, :]
# # # # #                 spread = ((step_data[:, :, 0].std(0)**2
# # # # #                            + step_data[:, :, 1].std(0)**2).sqrt() * 500.0).mean().item()
# # # # #                 step_spreads.append(spread)
# # # # #             spread_per_step.append(step_spreads)
# # # # #             n += 1
# # # # #     r = acc.compute()
# # # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# # # # #     if spread_per_step:
# # # # #         r["spread_72h_km"] = float(np.array(spread_per_step)[:, -1].mean())
# # # # #     return r


# # # # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # # # #                            fast_ensemble, metrics_csv, epoch,
# # # # #                            use_ema=False, ema_obj=None):
# # # # #     backup = None
# # # # #     if use_ema and ema_obj is not None:
# # # # #         try:
# # # # #             backup = ema_obj.apply_to(model)
# # # # #         except Exception as e:
# # # # #             print(f"  ⚠  EMA apply_to failed: {e}")
# # # # #             backup = None; use_ema = False

# # # # #     model.eval()
# # # # #     acc = StepErrorAccumulator(pred_len)
# # # # #     t0  = time.perf_counter()
# # # # #     n   = 0
# # # # #     with torch.no_grad():
# # # # #         for batch in val_loader:
# # # # #             bl = move(list(batch), device)
# # # # #             pred, _, _ = model.sample(
# # # # #                 bl, num_ensemble=max(fast_ensemble, 20),
# # # # #                 ddim_steps=max(ode_steps, 20), importance_weight=True)
# # # # #             T_pred = pred.shape[0]
# # # # #             gt     = bl[1][:T_pred]
# # # # #             dist, ate, cte = _eval_batch_atecte(pred, gt)
# # # # #             acc.update(dist, ate_km=ate, cte_km=cte)
# # # # #             n += 1

# # # # #     r       = acc.compute()
# # # # #     elapsed = time.perf_counter() - t0
# # # # #     score   = _composite_score(r)
# # # # #     tag     = "EMA" if use_ema else "RAW"

# # # # #     ade_v = r.get("ADE",      float("nan"))
# # # # #     fde_v = r.get("FDE",      float("nan"))
# # # # #     h12_v = r.get("12h",      float("nan"))
# # # # #     h24_v = r.get("24h",      float("nan"))
# # # # #     h48_v = r.get("48h",      float("nan"))
# # # # #     h72_v = r.get("72h",      float("nan"))
# # # # #     ate_v = r.get("ATE_mean", float("nan"))
# # # # #     cte_v = r.get("CTE_mean", float("nan"))

# # # # #     def ind(v, tgt):
# # # # #         if not np.isfinite(v): return ""
# # # # #         return "✅" if v < tgt else "❌"

# # # # #     print(f"\n{'='*70}")
# # # # #     print(f"  [FULL VAL ({tag})  ep={epoch}  {elapsed:.0f}s  {n} batches]")
# # # # #     print(f"  ADE={ade_v:.1f}{ind(ade_v,136.41)}  FDE={fde_v:.1f} km")
# # # # #     print(f"  12h={h12_v:.0f}{ind(h12_v,50)}  24h={h24_v:.0f}{ind(h24_v,100)}  "
# # # # #           f"48h={h48_v:.0f}{ind(h48_v,200)}  72h={h72_v:.0f}{ind(h72_v,297)} km")
# # # # #     if np.isfinite(ate_v):
# # # # #         print(f"  ATE={ate_v:.1f}{ind(ate_v,79.94)}  CTE={cte_v:.1f}{ind(cte_v,93.58)}"
# # # # #               f"  [ST-Trans: ATE=79.94 CTE=93.58]")
# # # # #     beat = _beat_report(r)
# # # # #     if beat: print(f"  {beat}")
# # # # #     print(f"  Score = {score:.2f}  (< 100 = beat ST-Trans all metrics)")
# # # # #     print(f"{'='*70}\n")

# # # # #     from datetime import datetime
# # # # #     dm = DatasetMetrics(
# # # # #         ade=ade_v if np.isfinite(ade_v) else 0.0,
# # # # #         fde=fde_v if np.isfinite(fde_v) else 0.0,
# # # # #         ugde_12h=h12_v if np.isfinite(h12_v) else 0.0,
# # # # #         ugde_24h=h24_v if np.isfinite(h24_v) else 0.0,
# # # # #         ugde_48h=h48_v if np.isfinite(h48_v) else 0.0,
# # # # #         ugde_72h=h72_v if np.isfinite(h72_v) else 0.0,
# # # # #         ate_abs_mean=ate_v if np.isfinite(ate_v) else 0.0,
# # # # #         cte_abs_mean=cte_v if np.isfinite(cte_v) else 0.0,
# # # # #         n_total=r.get("n_samples", 0),
# # # # #         timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # # #     )
# # # # #     save_metrics_csv(dm, metrics_csv, tag=f"val_{tag}_ep{epoch:03d}")

# # # # #     if backup is not None:
# # # # #         try:
# # # # #             ema_obj.restore(model, backup)
# # # # #         except Exception as e:
# # # # #             print(f"  ⚠  EMA restore failed: {e}")
# # # # #     return r


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  BestModelSaver — unchanged
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # class HorizonAwareBestSaver:
# # # # #     def __init__(self, patience=30, tol=1.5):
# # # # #         self.patience   = patience
# # # # #         self.tol        = tol
# # # # #         self.counter    = 0
# # # # #         self.early_stop = False
# # # # #         self.best_score = float("inf")
# # # # #         self.best_ade   = float("inf")
# # # # #         self.best_12h   = float("inf")
# # # # #         self.best_24h   = float("inf")
# # # # #         self.best_48h   = float("inf")
# # # # #         self.best_72h   = float("inf")
# # # # #         self.best_ate   = float("inf")
# # # # #         self.best_cte   = float("inf")

# # # # #     def update(self, r, model, out_dir, epoch, optimizer, scheduler,
# # # # #                tl, vl, saver_ref, min_epochs=30):
# # # # #         ade   = r.get("ADE", float("inf"))
# # # # #         h12   = r.get("12h", float("inf"))
# # # # #         h24   = r.get("24h", float("inf"))
# # # # #         h48   = r.get("48h", float("inf"))
# # # # #         h72   = r.get("72h", float("inf"))
# # # # #         ate   = r.get("ATE_mean", ade * 0.46 if np.isfinite(ade) else float("inf"))
# # # # #         cte   = r.get("CTE_mean", ade * 0.53 if np.isfinite(ade) else float("inf"))
# # # # #         score = _composite_score(r)
# # # # #         improved_any = False

# # # # #         for metric_val, best_attr, fname, extra in [
# # # # #             (ade, 'best_ade', 'best_ade.pth',  {"ade": ade}),
# # # # #             (h72, 'best_72h', 'best_72h.pth',  {"h72": h72}),
# # # # #             (ate, 'best_ate', 'best_ate.pth',  {"ate": ate, "cte": cte}),
# # # # #             (cte, 'best_cte', 'best_cte.pth',  {"ate": ate, "cte": cte}),
# # # # #         ]:
# # # # #             if metric_val < getattr(self, best_attr):
# # # # #                 setattr(self, best_attr, metric_val)
# # # # #                 improved_any = True
# # # # #                 _save_checkpoint(os.path.join(out_dir, fname),
# # # # #                                   epoch, model, optimizer, scheduler,
# # # # #                                   saver_ref, tl, vl, extra)

# # # # #         if h48 < self.best_48h: self.best_48h = h48; improved_any = True
# # # # #         if h24 < self.best_24h: self.best_24h = h24; improved_any = True
# # # # #         if h12 < self.best_12h: self.best_12h = h12; improved_any = True

# # # # #         if score < self.best_score - self.tol:
# # # # #             self.best_score = score
# # # # #             self.counter    = 0
# # # # #             _save_checkpoint(
# # # # #                 os.path.join(out_dir, "best_model.pth"),
# # # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # # #                 {"ade": ade, "h12": h12, "h24": h24, "h48": h48, "h72": h72,
# # # # #                  "ate": ate, "cte": cte, "composite_score": score})
# # # # #             print(f"  ✅ Best COMPOSITE={score:.2f}  ADE={ade:.1f}  "
# # # # #                   f"12h={h12:.0f}  24h={h24:.0f}  48h={h48:.0f}  "
# # # # #                   f"72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}  (ep {epoch})")
# # # # #         else:
# # # # #             if not improved_any:
# # # # #                 self.counter += 1
# # # # #             print(f"  No improvement {self.counter}/{self.patience}"
# # # # #                   f"  (best={self.best_score:.2f} cur={score:.2f})"
# # # # #                   f"  72h={h72:.0f}↓{self.best_72h:.0f}"
# # # # #                   f"  ATE={ate:.1f}↓{self.best_ate:.1f}")

# # # # #         if epoch >= min_epochs and self.counter >= self.patience:
# # # # #             self.early_stop = True


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Env diagnostic — unchanged
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def _check_env(bl, train_dataset):
# # # # #     try:    env_dir = train_dataset.env_path
# # # # #     except AttributeError:
# # # # #         try:    env_dir = train_dataset.dataset.env_path
# # # # #         except: env_dir = "UNKNOWN"
# # # # #     print(f"  Env path: {env_dir}")
# # # # #     env_data = bl[13]
# # # # #     if env_data is None:
# # # # #         print("  ⚠️  env_data is None"); return
# # # # #     for key in ("gph500_mean", "gph500_center", "u500_mean", "v500_mean"):
# # # # #         if key not in env_data:
# # # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # # #         v    = env_data[key]
# # # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # # #         print(f"  {'✅' if zero < 80 else '⚠️'}  {key}: "
# # # # #               f"mean={v.mean().item():.4f} std={v.std().item():.4f} zero={zero:.1f}%")


# # # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # # #  Args — unchanged
# # # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # # def get_args():
# # # # #     p = argparse.ArgumentParser(
# # # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # # # #     p.add_argument("--obs_len",         default=8,          type=int)
# # # # #     p.add_argument("--pred_len",        default=12,         type=int)
# # # # #     p.add_argument("--batch_size",      default=32,         type=int)
# # # # #     p.add_argument("--num_epochs",      default=100,        type=int)
# # # # #     p.add_argument("--g_learning_rate", default=3e-4,       type=float)
# # # # #     p.add_argument("--weight_decay",    default=1e-4,       type=float)
# # # # #     p.add_argument("--warmup_epochs",   default=5,          type=int)
# # # # #     p.add_argument("--grad_clip",       default=1.0,        type=float)
# # # # #     p.add_argument("--patience",        default=30,         type=int)
# # # # #     p.add_argument("--min_epochs",      default=30,         type=int)
# # # # #     p.add_argument("--n_train_ens",     default=4,          type=int)
# # # # #     p.add_argument("--use_amp",         action="store_true")
# # # # #     p.add_argument("--num_workers",     default=2,          type=int)
# # # # #     p.add_argument("--sigma_min",            default=0.02,  type=float)
# # # # #     p.add_argument("--ctx_noise_scale",      default=0.01,  type=float)
# # # # #     p.add_argument("--initial_sample_sigma", default=0.03,  type=float)
# # # # #     p.add_argument("--ode_steps_train", default=10,  type=int)
# # # # #     p.add_argument("--ode_steps_val",   default=10,  type=int)
# # # # #     p.add_argument("--ode_steps_test",  default=10,  type=int)
# # # # #     p.add_argument("--val_ensemble",    default=20,  type=int)
# # # # #     p.add_argument("--fast_ensemble",   default=10,  type=int)
# # # # #     p.add_argument("--val_freq",        default=5,   type=int)
# # # # #     p.add_argument("--val_ade_freq",    default=5,   type=int)
# # # # #     p.add_argument("--val_subset_size", default=500, type=int)
# # # # #     p.add_argument("--use_ema",         action="store_true", default=True)
# # # # #     p.add_argument("--ema_decay",       default=0.992, type=float)
# # # # #     p.add_argument("--swa_start_epoch", default=60,   type=int)
# # # # #     p.add_argument("--teacher_forcing", action="store_true", default=True)
# # # # #     p.add_argument("--output_dir",  default="runs/v39",        type=str)
# # # # #     p.add_argument("--metrics_csv", default="metrics_v39.csv", type=str)
# # # # #     p.add_argument("--predict_csv", default="predictions.csv", type=str)
# # # # #     p.add_argument("--gpu_num",     default="0", type=str)
# # # # #     p.add_argument("--delim",       default=" ")
# # # # #     p.add_argument("--skip",        default=1,   type=int)
# # # # #     p.add_argument("--min_ped",     default=1,   type=int)
# # # # #     p.add_argument("--threshold",   default=0.002, type=float)
# # # # #     p.add_argument("--other_modal", default="gph")
# # # # #     p.add_argument("--test_year",   default=None, type=int)
# # # # #     p.add_argument("--resume",       default=None, type=str)
# # # # #     p.add_argument("--resume_epoch", default=None, type=int)
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

# # # # #     print("=" * 72)
# # # # #     print("  TC-FlowMatching v39  |  BEAT ST-TRANS EDITION")
# # # # #     print("  TARGETS: DPE<136 | ATE<79.94 | CTE<93.58 | 72h<297")
# # # # #     print("  LOSSES: speed_accuracy + cumulative_displacement (cited)")
# # # # #     print("  EMA decay:", args.ema_decay)
# # # # #     print("=" * 72)

# # # # #     train_dataset, train_loader = data_loader(
# # # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # # #     val_dataset, val_loader = data_loader(
# # # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # # #     val_subset_loader = make_val_subset_loader(
# # # # #         val_dataset, args.val_subset_size, args.batch_size,
# # # # #         seq_collate, args.num_workers)

# # # # #     test_loader = None
# # # # #     try:
# # # # #         _, test_loader = data_loader(
# # # # #             args, {"root": args.dataset_root, "type": "test"},
# # # # #             test=True, test_year=None)
# # # # #     except Exception as e:
# # # # #         print(f"  Warning: test loader: {e}")

# # # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # # #     print(f"  val   : {len(val_dataset)} seq")

# # # # #     model = TCFlowMatching(
# # # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # # #         sigma_min=args.sigma_min, n_train_ens=args.n_train_ens,
# # # # #         ctx_noise_scale=args.ctx_noise_scale,
# # # # #         initial_sample_sigma=args.initial_sample_sigma,
# # # # #         teacher_forcing=args.teacher_forcing,
# # # # #         use_ema=args.use_ema, ema_decay=args.ema_decay,
# # # # #     ).to(device)

# # # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # # #     print(f"  params  : {n_params:,}")
# # # # #     model.init_ema()
# # # # #     print(f"  EMA     : {'ON' if model._ema is not None else 'OFF'}"
# # # # #           f"  (decay={args.ema_decay})")

# # # # #     optimizer = optim.AdamW(model.parameters(),
# # # # #                              lr=args.g_learning_rate,
# # # # #                              weight_decay=args.weight_decay)
# # # # #     saver  = HorizonAwareBestSaver(patience=args.patience, tol=1.5)
# # # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # # #     swa    = SWAManager(model, start_epoch=args.swa_start_epoch)

# # # # #     steps_per_epoch = len(train_loader)
# # # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # # #     scheduler = get_cosine_schedule_with_warmup(
# # # # #         optimizer, warmup, total_steps, min_lr=1e-6)

# # # # #     # Resume
# # # # #     has_scheduler_state   = False
# # # # #     _ckpt_scheduler_state = None
# # # # #     start_epoch = 0

# # # # #     if args.resume is not None and os.path.exists(args.resume):
# # # # #         print(f"  Loading checkpoint: {args.resume}")
# # # # #         ckpt = torch.load(args.resume, map_location=device)
# # # # #         m = _get_raw_model(model)
# # # # #         missing, _ = m.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # # #         if missing:
# # # # #             print(f"  ⚠  Missing keys ({len(missing)}): {missing[:3]}")
# # # # #         if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
# # # # #             ema_obj = _get_ema_obj(model)
# # # # #             if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # # # #                 for k, v in ckpt["ema_shadow"].items():
# # # # #                     if k in ema_obj.shadow:
# # # # #                         ema_obj.shadow[k].copy_(v.to(device))
# # # # #         if "optimizer_state" in ckpt:
# # # # #             try:
# # # # #                 optimizer.load_state_dict(ckpt["optimizer_state"])
# # # # #                 for state in optimizer.state.values():
# # # # #                     for k, v in state.items():
# # # # #                         if torch.is_tensor(v): state[k] = v.to(device)
# # # # #             except Exception as e:
# # # # #                 print(f"  ⚠  Optimizer restore failed: {e}")
# # # # #         has_scheduler_state   = "scheduler_state" in ckpt
# # # # #         _ckpt_scheduler_state = ckpt.get("scheduler_state", None)
# # # # #         if "best_score" in ckpt:
# # # # #             saver.best_score = ckpt["best_score"]
# # # # #             saver.best_ade   = ckpt.get("best_ade",  float("inf"))
# # # # #             saver.best_72h   = ckpt.get("best_72h",  float("inf"))
# # # # #             saver.best_48h   = ckpt.get("best_48h",  float("inf"))
# # # # #             saver.best_24h   = ckpt.get("best_24h",  float("inf"))
# # # # #             saver.best_12h   = ckpt.get("best_12h",  float("inf"))
# # # # #             saver.best_ate   = ckpt.get("best_ate",  float("inf"))
# # # # #             saver.best_cte   = ckpt.get("best_cte",  float("inf"))
# # # # #         start_epoch = (args.resume_epoch if args.resume_epoch is not None
# # # # #                        else ckpt.get("epoch", 0) + 1)
# # # # #         print(f"  → Resuming from epoch {start_epoch}")
# # # # #     elif args.resume is not None:
# # # # #         print(f"  ⚠  Checkpoint không tìm thấy: {args.resume}")

# # # # #     try:
# # # # #         model = torch.compile(model, mode="reduce-overhead")
# # # # #         print("  torch.compile: enabled")
# # # # #     except Exception:
# # # # #         pass

# # # # #     if has_scheduler_state and _ckpt_scheduler_state is not None:
# # # # #         try:
# # # # #             scheduler.load_state_dict(_ckpt_scheduler_state)
# # # # #         except Exception as e:
# # # # #             print(f"  ⚠  Scheduler restore failed: {e}")
# # # # #             for _ in range(start_epoch * steps_per_epoch): scheduler.step()
# # # # #     elif start_epoch > 0:
# # # # #         for _ in range(start_epoch * steps_per_epoch): scheduler.step()

# # # # #     print("=" * 72)
# # # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch, start={start_epoch})")
# # # # #     print("=" * 72)

# # # # #     epoch_times = []
# # # # #     train_start = time.perf_counter()

# # # # #     for epoch in range(start_epoch, args.num_epochs):
# # # # #         model.train()
# # # # #         sum_loss = 0.0
# # # # #         t0 = time.perf_counter()

# # # # #         for i, batch in enumerate(train_loader):
# # # # #             bl = move(list(batch), device)

# # # # #             if epoch == start_epoch and i == 0:
# # # # #                 _check_env(bl, train_dataset)

# # # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # # #                 bd = model.get_loss_breakdown(bl, epoch=epoch)

# # # # #             optimizer.zero_grad()
# # # # #             scaler.scale(bd["total"]).backward()
# # # # #             scaler.unscale_(optimizer)
# # # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # # #             scaler.step(optimizer)
# # # # #             scaler.update()
# # # # #             scheduler.step()
# # # # #             _call_ema_update(model)
# # # # #             sum_loss += bd["total"].item()

# # # # #             if i % 20 == 0:
# # # # #                 lr = optimizer.param_groups[0]["lr"]
# # # # #                 # ── LOG LINE v39: spd=speed_acc  cml=cumul_disp  vsm=vel_smooth
# # # # #                 print(
# # # # #                     f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # # #                     f"  tot={bd['total'].item():.3f}"
# # # # #                     f"  fm={bd.get('fm_mse',    0):.3f}"
# # # # #                     f"  hor={bd.get('mse_hav',  0):.3f}"
# # # # #                     f"  end={bd.get('endpoint', 0):.3f}"
# # # # #                     f"  spd={bd.get('speed_acc',  0):.3f}"   # ← v39
# # # # #                     f"  cml={bd.get('cumul_disp', 0):.3f}"   # ← v39
# # # # #                     f"  hd={bd.get('heading',   0):.3f}"
# # # # #                     f"  vsm={bd.get('vel_smooth',0):.3f}"    # ← v39
# # # # #                     f"  lr={lr:.2e}"
# # # # #                 )

# # # # #         ep_s  = time.perf_counter() - t0
# # # # #         epoch_times.append(ep_s)
# # # # #         avg_t = sum_loss / len(train_loader)
# # # # #         swa.update(model, epoch)

# # # # #         model.eval()
# # # # #         val_loss = 0.0
# # # # #         with torch.no_grad():
# # # # #             for batch in val_loader:
# # # # #                 bl_v = move(list(batch), device)
# # # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # # #                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
# # # # #         avg_vl = val_loss / len(val_loader)
# # # # #         print(f"  Epoch {epoch:>3}  train={avg_t:.4f}  val={avg_vl:.4f}"
# # # # #               f"  time={ep_s:.0f}s")

# # # # #         # Fast eval every epoch
# # # # #         m_fast = evaluate_fast(model, val_subset_loader, device,
# # # # #                                 args.ode_steps_train, args.pred_len,
# # # # #                                 args.fast_ensemble)
# # # # #         h12 = m_fast.get("12h", float("nan"))
# # # # #         h24 = m_fast.get("24h", float("nan"))
# # # # #         h48 = m_fast.get("48h", float("nan"))
# # # # #         h72 = m_fast.get("72h", float("nan"))
# # # # #         ate = m_fast.get("ATE_mean", float("nan"))
# # # # #         cte = m_fast.get("CTE_mean", float("nan"))
# # # # #         fast_score = _composite_score(m_fast)

# # # # #         def ind(v, tgt):
# # # # #             if not np.isfinite(v): return "?"
# # # # #             return "🎯" if v < tgt else "❌"

# # # # #         print(
# # # # #             f"  [FAST ep{epoch}]"
# # # # #             f"  ADE={m_fast.get('ADE', float('nan')):.1f}"
# # # # #             f"  12h={h12:.0f}{ind(h12,50)}"
# # # # #             f"  24h={h24:.0f}{ind(h24,100)}"
# # # # #             f"  48h={h48:.0f}{ind(h48,200)}"
# # # # #             f"  72h={h72:.0f}{ind(h72,297)}"
# # # # #             f"  ATE={ate:.1f}{ind(ate,79.94)}"
# # # # #             f"  CTE={cte:.1f}{ind(cte,93.58)}"
# # # # #             f"  score={fast_score:.2f}"
# # # # #         )

# # # # #         # Full val every val_freq epochs
# # # # #         if epoch % args.val_freq == 0:
# # # # #             ema_obj = _get_ema_obj(model)
# # # # #             try:
# # # # #                 r_raw = evaluate_full_val_ade(
# # # # #                     model, val_loader, device,
# # # # #                     ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # # #                     fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # #                     epoch=epoch, use_ema=False)
# # # # #                 r_use = r_raw
# # # # #                 if ema_obj is not None and epoch >= 5:
# # # # #                     r_ema = evaluate_full_val_ade(
# # # # #                         model, val_loader, device,
# # # # #                         ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # # #                         fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # # #                         epoch=epoch, use_ema=True, ema_obj=ema_obj)
# # # # #                     if _composite_score(r_ema) < _composite_score(r_raw):
# # # # #                         r_use = r_ema
# # # # #                 saver.update(r_use, model, args.output_dir, epoch,
# # # # #                              optimizer, scheduler, avg_t, avg_vl,
# # # # #                              saver_ref=saver, min_epochs=args.min_epochs)
# # # # #             except Exception as e:
# # # # #                 print(f"  ⚠  Full val failed: {e}")
# # # # #                 import traceback; traceback.print_exc()

# # # # #         if epoch % 10 == 0 or epoch == args.num_epochs - 1:
# # # # #             _save_checkpoint(
# # # # #                 os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"),
# # # # #                 epoch, model, optimizer, scheduler, saver, avg_t, avg_vl,
# # # # #                 {"ade": m_fast.get("ADE", float("nan")),
# # # # #                  "h48": h48, "h72": h72, "ate": ate, "cte": cte})

# # # # #         if epoch % 10 == 9:
# # # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")
# # # # #             print(f"  📊 Best: score={saver.best_score:.2f}  "
# # # # #                   f"72h={saver.best_72h:.0f}km  "
# # # # #                   f"ATE={saver.best_ate:.1f}km  CTE={saver.best_cte:.1f}km")

# # # # #         if saver.early_stop:
# # # # #             print(f"  ⛔ Early stopping at epoch {epoch}")
# # # # #             break

# # # # #     # Final SWA eval
# # # # #     print(f"\n{'='*72}")
# # # # #     swa_backup = swa.apply_to(model)
# # # # #     if swa_backup is not None:
# # # # #         print("  Evaluating SWA weights...")
# # # # #         r_swa = evaluate_full_val_ade(
# # # # #             model, val_loader, device,
# # # # #             ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # # #             fast_ensemble=args.val_ensemble, metrics_csv=metrics_csv,
# # # # #             epoch=9999, use_ema=False)
# # # # #         if _composite_score(r_swa) < saver.best_score:
# # # # #             _save_checkpoint(
# # # # #                 os.path.join(args.output_dir, "best_swa.pth"),
# # # # #                 9999, model, optimizer, scheduler, saver,
# # # # #                 avg_t=0.0, avg_vl=0.0, metrics={"tag": "best_swa"})
# # # # #             print("  ✅ SWA checkpoint saved")
# # # # #         swa.restore(model, swa_backup)

# # # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # # #     print(f"\n  Best composite score: {saver.best_score:.2f}")
# # # # #     print(f"  Best: ADE={saver.best_ade:.1f}  "
# # # # #           f"12h={saver.best_12h:.0f}  24h={saver.best_24h:.0f}  "
# # # # #           f"48h={saver.best_48h:.0f}  72h={saver.best_72h:.0f}  "
# # # # #           f"ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}")
# # # # #     print(f"  Total time: {total_h:.2f}h")

# # # # #     best_r = {"ADE": saver.best_ade, "12h": saver.best_12h,
# # # # #               "24h": saver.best_24h, "48h": saver.best_48h,
# # # # #               "72h": saver.best_72h, "ATE_mean": saver.best_ate,
# # # # #               "CTE_mean": saver.best_cte}
# # # # #     beat = _beat_report(best_r)
# # # # #     if beat:
# # # # #         print(f"\n  🏆🏆🏆 {beat} 🏆🏆🏆")
# # # # #     else:
# # # # #         print(f"\n  Need: DPE<136.41, ATE<79.94, CTE<93.58, 72h<297")
# # # # #     print("=" * 72)


# # # # # if __name__ == "__main__":
# # # # #     args = get_args()
# # # # #     np.random.seed(42); torch.manual_seed(42)
# # # # #     if torch.cuda.is_available():
# # # # #         torch.cuda.manual_seed_all(42)
# # # # #     main(args)

# # # # """
# # # # scripts/train_flowmatching.py — v40  CLEAN 7-LOSS EDITION
# # # # ═══════════════════════════════════════════════════════════════════════════════
# # # # THAY ĐỔI DUY NHẤT so với v36 train script (doc 5):

# # # #   Log print line — update key names cho losses v39:
# # # #     ĐỔI:  vel={vel_smooth}  atc={ate_cte}  str={steering}
# # # #     THÀNH: spd={speed_acc}  cml={cumul_disp}  acc={accel}  dcp={decomp}  cns={cons}

# # # #   Tất cả code khác GIỐNG HỆT v36 train script.
# # # # """
# # # # from __future__ import annotations

# # # # import sys
# # # # import os
# # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # import argparse
# # # # import time
# # # # import random

# # # # import numpy as np
# # # # import torch
# # # # import torch.optim as optim
# # # # from torch.amp import autocast, GradScaler
# # # # from torch.utils.data import DataLoader, Subset

# # # # from Model.data.loader_training import data_loader
# # # # from Model.flow_matching_model import TCFlowMatching
# # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # from utils.metrics import (
# # # #     StepErrorAccumulator,
# # # #     save_metrics_csv,
# # # #     haversine_km_torch,
# # # #     denorm_torch,
# # # #     HORIZON_STEPS,
# # # #     DatasetMetrics,
# # # # )

# # # # try:
# # # #     from utils.metrics import haversine_and_atecte_torch
# # # #     HAS_ATECTE = True
# # # # except ImportError:
# # # #     HAS_ATECTE = False
# # # #     print("  ⚠  haversine_and_atecte_torch not found — ATE/CTE disabled")


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Composite Score v2
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def _composite_score(result: dict) -> float:
# # # #     ade = result.get("ADE", float("inf"))
# # # #     h12 = result.get("12h", float("inf"))
# # # #     h24 = result.get("24h", float("inf"))
# # # #     h48 = result.get("48h", float("inf"))
# # # #     h72 = result.get("72h", float("inf"))
# # # #     ate = result.get("ATE_mean", float("inf"))
# # # #     cte = result.get("CTE_mean", float("inf"))
# # # #     if not np.isfinite(ate): ate = ade * 0.46
# # # #     if not np.isfinite(cte): cte = ade * 0.53
# # # #     return 100.0 * (
# # # #         0.05 * (ade / 136.0)
# # # #         + 0.05 * (h12 / 50.0)
# # # #         + 0.10 * (h24 / 100.0)
# # # #         + 0.15 * (h48 / 200.0)
# # # #         + 0.35 * (h72 / 300.0)
# # # #         + 0.15 * (ate / 80.0)
# # # #         + 0.15 * (cte / 94.0)
# # # #     )


# # # # def _beat_report(r: dict) -> str:
# # # #     parts = []
# # # #     if np.isfinite(r.get("ADE", float("inf"))) and r["ADE"] < 136.41:
# # # #         parts.append(f"DPE✅{r['ADE']:.1f}")
# # # #     if np.isfinite(r.get("ATE_mean", float("inf"))) and r["ATE_mean"] < 79.94:
# # # #         parts.append(f"ATE✅{r['ATE_mean']:.1f}")
# # # #     if np.isfinite(r.get("CTE_mean", float("inf"))) and r["CTE_mean"] < 93.58:
# # # #         parts.append(f"CTE✅{r['CTE_mean']:.1f}")
# # # #     if np.isfinite(r.get("72h", float("inf"))) and r["72h"] < 297.0:
# # # #         parts.append(f"72h✅{r['72h']:.1f}")
# # # #     return "🏆 BEAT: " + " ".join(parts) if parts else ""


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Helpers — unchanged
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


# # # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # # #                             collate_fn, num_workers):
# # # #     n   = len(val_dataset)
# # # #     rng = random.Random(42)
# # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # #     return DataLoader(Subset(val_dataset, idx),
# # # #                       batch_size=batch_size, shuffle=False,
# # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # def _get_ema_obj(model):
# # # #     if hasattr(model, '_ema') and model._ema is not None:
# # # #         return model._ema
# # # #     if hasattr(model, '_orig_mod'):
# # # #         orig = model._orig_mod
# # # #         if hasattr(orig, '_ema') and orig._ema is not None:
# # # #             return orig._ema
# # # #     return None


# # # # def _call_ema_update(model):
# # # #     if hasattr(model, '_orig_mod'):
# # # #         orig = model._orig_mod
# # # #         if hasattr(orig, 'ema_update'):
# # # #             orig.ema_update(); return
# # # #     if hasattr(model, 'ema_update'):
# # # #         model.ema_update()


# # # # def _get_raw_model(model):
# # # #     return model._orig_mod if hasattr(model, '_orig_mod') else model


# # # # def _save_checkpoint(path, epoch, model, optimizer, scheduler,
# # # #                      saver, avg_t, avg_vl, metrics=None):
# # # #     m = _get_raw_model(model)
# # # #     ema_obj = _get_ema_obj(model)
# # # #     ema_sd  = None
# # # #     if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # # #         try:
# # # #             ema_sd = {k: v.cpu().clone() for k, v in ema_obj.shadow.items()}
# # # #         except Exception:
# # # #             ema_sd = None
# # # #     payload = {
# # # #         "epoch": epoch, "model_state_dict": m.state_dict(),
# # # #         "optimizer_state": optimizer.state_dict(),
# # # #         "scheduler_state": scheduler.state_dict(),
# # # #         "ema_shadow": ema_sd,
# # # #         "best_score": saver.best_score, "best_ade": saver.best_ade,
# # # #         "best_72h": saver.best_72h, "best_48h": saver.best_48h,
# # # #         "best_24h": saver.best_24h, "best_12h": saver.best_12h,
# # # #         "best_ate": getattr(saver, 'best_ate', float("inf")),
# # # #         "best_cte": getattr(saver, 'best_cte', float("inf")),
# # # #         "train_loss": avg_t, "val_loss": avg_vl,
# # # #     }
# # # #     if metrics: payload.update(metrics)
# # # #     torch.save(payload, path)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  SWA — unchanged
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class SWAManager:
# # # #     def __init__(self, model, start_epoch=60):
# # # #         self.start_epoch = start_epoch
# # # #         self.n_averaged  = 0
# # # #         self.avg_state   = None

# # # #     def update(self, model, epoch):
# # # #         if epoch < self.start_epoch: return
# # # #         m  = _get_raw_model(model)
# # # #         sd = {k: v.detach().clone() for k, v in m.state_dict().items()
# # # #               if v.dtype.is_floating_point}
# # # #         if self.avg_state is None:
# # # #             self.avg_state = sd; self.n_averaged = 1
# # # #         else:
# # # #             n = self.n_averaged
# # # #             for k in self.avg_state:
# # # #                 if k in sd:
# # # #                     self.avg_state[k].mul_(n/(n+1)).add_(sd[k], alpha=1.0/(n+1))
# # # #             self.n_averaged += 1

# # # #     def apply_to(self, model):
# # # #         if self.avg_state is None: return None
# # # #         m  = _get_raw_model(model)
# # # #         sd = m.state_dict()
# # # #         backup = {k: sd[k].detach().clone() for k in self.avg_state if k in sd}
# # # #         for k, v in self.avg_state.items():
# # # #             if k in sd: sd[k].copy_(v)
# # # #         return backup

# # # #     def restore(self, model, backup):
# # # #         if backup is None: return
# # # #         m  = _get_raw_model(model)
# # # #         sd = m.state_dict()
# # # #         for k, v in backup.items():
# # # #             if k in sd: sd[k].copy_(v)


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Evaluation — unchanged
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def _eval_batch_atecte(pred_norm, gt_norm):
# # # #     pred_d = denorm_torch(pred_norm)
# # # #     gt_d   = denorm_torch(gt_norm)
# # # #     if HAS_ATECTE:
# # # #         dist, ate, cte = haversine_and_atecte_torch(pred_d, gt_d, unit_01deg=True)
# # # #         return dist, ate, cte
# # # #     else:
# # # #         dist = haversine_km_torch(pred_d, gt_d)
# # # #         return dist, None, None


# # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=15):
# # # #     model.eval()
# # # #     acc = StepErrorAccumulator(pred_len)
# # # #     t0  = time.perf_counter()
# # # #     n   = 0
# # # #     spread_per_step = []
# # # #     with torch.no_grad():
# # # #         for batch in loader:
# # # #             bl = move(list(batch), device)
# # # #             pred, _, all_trajs = model.sample(
# # # #                 bl, num_ensemble=fast_ensemble, ddim_steps=ode_steps,
# # # #                 importance_weight=True)
# # # #             T_active  = pred.shape[0]
# # # #             gt_sliced = bl[1][:T_active]
# # # #             dist, ate, cte = _eval_batch_atecte(pred, gt_sliced)
# # # #             acc.update(dist, ate_km=ate, cte_km=cte)
# # # #             step_spreads = []
# # # #             for t in range(all_trajs.shape[1]):
# # # #                 step_data = all_trajs[:, t, :, :]
# # # #                 spread = ((step_data[:, :, 0].std(0)**2
# # # #                            + step_data[:, :, 1].std(0)**2).sqrt() * 500.0).mean().item()
# # # #                 step_spreads.append(spread)
# # # #             spread_per_step.append(step_spreads)
# # # #             n += 1
# # # #     r = acc.compute()
# # # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# # # #     if spread_per_step:
# # # #         r["spread_72h_km"] = float(np.array(spread_per_step)[:, -1].mean())
# # # #     return r


# # # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # # #                            fast_ensemble, metrics_csv, epoch,
# # # #                            use_ema=False, ema_obj=None):
# # # #     backup = None
# # # #     if use_ema and ema_obj is not None:
# # # #         try:
# # # #             backup = ema_obj.apply_to(model)
# # # #         except Exception as e:
# # # #             print(f"  ⚠  EMA apply_to failed: {e}")
# # # #             backup = None; use_ema = False

# # # #     model.eval()
# # # #     acc = StepErrorAccumulator(pred_len)
# # # #     t0  = time.perf_counter()
# # # #     n   = 0
# # # #     with torch.no_grad():
# # # #         for batch in val_loader:
# # # #             bl = move(list(batch), device)
# # # #             pred, _, _ = model.sample(
# # # #                 bl, num_ensemble=max(fast_ensemble, 20),
# # # #                 ddim_steps=max(ode_steps, 20), importance_weight=True)
# # # #             T_pred = pred.shape[0]
# # # #             gt     = bl[1][:T_pred]
# # # #             dist, ate, cte = _eval_batch_atecte(pred, gt)
# # # #             acc.update(dist, ate_km=ate, cte_km=cte)
# # # #             n += 1

# # # #     r       = acc.compute()
# # # #     elapsed = time.perf_counter() - t0
# # # #     score   = _composite_score(r)
# # # #     tag     = "EMA" if use_ema else "RAW"

# # # #     ade_v = r.get("ADE",      float("nan"))
# # # #     fde_v = r.get("FDE",      float("nan"))
# # # #     h12_v = r.get("12h",      float("nan"))
# # # #     h24_v = r.get("24h",      float("nan"))
# # # #     h48_v = r.get("48h",      float("nan"))
# # # #     h72_v = r.get("72h",      float("nan"))
# # # #     ate_v = r.get("ATE_mean", float("nan"))
# # # #     cte_v = r.get("CTE_mean", float("nan"))

# # # #     def ind(v, tgt):
# # # #         if not np.isfinite(v): return ""
# # # #         return "✅" if v < tgt else "❌"

# # # #     print(f"\n{'='*70}")
# # # #     print(f"  [FULL VAL ({tag})  ep={epoch}  {elapsed:.0f}s  {n} batches]")
# # # #     print(f"  ADE={ade_v:.1f}{ind(ade_v,136.41)}  FDE={fde_v:.1f} km")
# # # #     print(f"  12h={h12_v:.0f}{ind(h12_v,50)}  24h={h24_v:.0f}{ind(h24_v,100)}  "
# # # #           f"48h={h48_v:.0f}{ind(h48_v,200)}  72h={h72_v:.0f}{ind(h72_v,297)} km")
# # # #     if np.isfinite(ate_v):
# # # #         print(f"  ATE={ate_v:.1f}{ind(ate_v,79.94)}  CTE={cte_v:.1f}{ind(cte_v,93.58)}"
# # # #               f"  [ST-Trans: ATE=79.94 CTE=93.58]")
# # # #     beat = _beat_report(r)
# # # #     if beat: print(f"  {beat}")
# # # #     print(f"  Score = {score:.2f}  (< 100 = beat ST-Trans all metrics)")
# # # #     print(f"{'='*70}\n")

# # # #     from datetime import datetime
# # # #     dm = DatasetMetrics(
# # # #         ade=ade_v if np.isfinite(ade_v) else 0.0,
# # # #         fde=fde_v if np.isfinite(fde_v) else 0.0,
# # # #         ugde_12h=h12_v if np.isfinite(h12_v) else 0.0,
# # # #         ugde_24h=h24_v if np.isfinite(h24_v) else 0.0,
# # # #         ugde_48h=h48_v if np.isfinite(h48_v) else 0.0,
# # # #         ugde_72h=h72_v if np.isfinite(h72_v) else 0.0,
# # # #         ate_abs_mean=ate_v if np.isfinite(ate_v) else 0.0,
# # # #         cte_abs_mean=cte_v if np.isfinite(cte_v) else 0.0,
# # # #         n_total=r.get("n_samples", 0),
# # # #         timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
# # # #     )
# # # #     save_metrics_csv(dm, metrics_csv, tag=f"val_{tag}_ep{epoch:03d}")

# # # #     if backup is not None:
# # # #         try:
# # # #             ema_obj.restore(model, backup)
# # # #         except Exception as e:
# # # #             print(f"  ⚠  EMA restore failed: {e}")
# # # #     return r


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  BestModelSaver — unchanged
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # class HorizonAwareBestSaver:
# # # #     def __init__(self, patience=30, tol=1.5):
# # # #         self.patience   = patience
# # # #         self.tol        = tol
# # # #         self.counter    = 0
# # # #         self.early_stop = False
# # # #         self.best_score = float("inf")
# # # #         self.best_ade   = float("inf")
# # # #         self.best_12h   = float("inf")
# # # #         self.best_24h   = float("inf")
# # # #         self.best_48h   = float("inf")
# # # #         self.best_72h   = float("inf")
# # # #         self.best_ate   = float("inf")
# # # #         self.best_cte   = float("inf")

# # # #     def update(self, r, model, out_dir, epoch, optimizer, scheduler,
# # # #                tl, vl, saver_ref, min_epochs=30):
# # # #         ade   = r.get("ADE", float("inf"))
# # # #         h12   = r.get("12h", float("inf"))
# # # #         h24   = r.get("24h", float("inf"))
# # # #         h48   = r.get("48h", float("inf"))
# # # #         h72   = r.get("72h", float("inf"))
# # # #         ate   = r.get("ATE_mean", ade * 0.46 if np.isfinite(ade) else float("inf"))
# # # #         cte   = r.get("CTE_mean", ade * 0.53 if np.isfinite(ade) else float("inf"))
# # # #         score = _composite_score(r)
# # # #         improved_any = False

# # # #         for metric_val, best_attr, fname, extra in [
# # # #             (ade, 'best_ade', 'best_ade.pth',  {"ade": ade}),
# # # #             (h72, 'best_72h', 'best_72h.pth',  {"h72": h72}),
# # # #             (ate, 'best_ate', 'best_ate.pth',  {"ate": ate, "cte": cte}),
# # # #             (cte, 'best_cte', 'best_cte.pth',  {"ate": ate, "cte": cte}),
# # # #         ]:
# # # #             if metric_val < getattr(self, best_attr):
# # # #                 setattr(self, best_attr, metric_val)
# # # #                 improved_any = True
# # # #                 _save_checkpoint(os.path.join(out_dir, fname),
# # # #                                   epoch, model, optimizer, scheduler,
# # # #                                   saver_ref, tl, vl, extra)

# # # #         if h48 < self.best_48h: self.best_48h = h48; improved_any = True
# # # #         if h24 < self.best_24h: self.best_24h = h24; improved_any = True
# # # #         if h12 < self.best_12h: self.best_12h = h12; improved_any = True

# # # #         if score < self.best_score - self.tol:
# # # #             self.best_score = score
# # # #             self.counter    = 0
# # # #             _save_checkpoint(
# # # #                 os.path.join(out_dir, "best_model.pth"),
# # # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # # #                 {"ade": ade, "h12": h12, "h24": h24, "h48": h48, "h72": h72,
# # # #                  "ate": ate, "cte": cte, "composite_score": score})
# # # #             print(f"  ✅ Best COMPOSITE={score:.2f}  ADE={ade:.1f}  "
# # # #                   f"12h={h12:.0f}  24h={h24:.0f}  48h={h48:.0f}  "
# # # #                   f"72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}  (ep {epoch})")
# # # #         else:
# # # #             if not improved_any:
# # # #                 self.counter += 1
# # # #             print(f"  No improvement {self.counter}/{self.patience}"
# # # #                   f"  (best={self.best_score:.2f} cur={score:.2f})"
# # # #                   f"  72h={h72:.0f}↓{self.best_72h:.0f}"
# # # #                   f"  ATE={ate:.1f}↓{self.best_ate:.1f}")

# # # #         if epoch >= min_epochs and self.counter >= self.patience:
# # # #             self.early_stop = True


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Env diagnostic — unchanged
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def _check_env(bl, train_dataset):
# # # #     try:    env_dir = train_dataset.env_path
# # # #     except AttributeError:
# # # #         try:    env_dir = train_dataset.dataset.env_path
# # # #         except: env_dir = "UNKNOWN"
# # # #     print(f"  Env path: {env_dir}")
# # # #     env_data = bl[13]
# # # #     if env_data is None:
# # # #         print("  ⚠️  env_data is None"); return
# # # #     for key in ("gph500_mean", "gph500_center", "u500_mean", "v500_mean"):
# # # #         if key not in env_data:
# # # #             print(f"  ⚠️  {key} MISSING"); continue
# # # #         v    = env_data[key]
# # # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # # #         print(f"  {'✅' if zero < 80 else '⚠️'}  {key}: "
# # # #               f"mean={v.mean().item():.4f} std={v.std().item():.4f} zero={zero:.1f}%")


# # # # # ══════════════════════════════════════════════════════════════════════════════
# # # # #  Args — unchanged
# # # # # ══════════════════════════════════════════════════════════════════════════════

# # # # def get_args():
# # # #     p = argparse.ArgumentParser(
# # # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # # #     p.add_argument("--obs_len",         default=8,          type=int)
# # # #     p.add_argument("--pred_len",        default=12,         type=int)
# # # #     p.add_argument("--batch_size",      default=32,         type=int)
# # # #     p.add_argument("--num_epochs",      default=100,        type=int)
# # # #     p.add_argument("--g_learning_rate", default=3e-4,       type=float)
# # # #     p.add_argument("--weight_decay",    default=1e-4,       type=float)
# # # #     p.add_argument("--warmup_epochs",   default=5,          type=int)
# # # #     p.add_argument("--grad_clip",       default=1.0,        type=float)
# # # #     p.add_argument("--patience",        default=30,         type=int)
# # # #     p.add_argument("--min_epochs",      default=30,         type=int)
# # # #     p.add_argument("--n_train_ens",     default=4,          type=int)
# # # #     p.add_argument("--use_amp",         action="store_true")
# # # #     p.add_argument("--num_workers",     default=2,          type=int)
# # # #     p.add_argument("--sigma_min",            default=0.02,  type=float)
# # # #     p.add_argument("--ctx_noise_scale",      default=0.01,  type=float)
# # # #     p.add_argument("--initial_sample_sigma", default=0.03,  type=float)
# # # #     p.add_argument("--ode_steps_train", default=10,  type=int)
# # # #     p.add_argument("--ode_steps_val",   default=10,  type=int)
# # # #     p.add_argument("--ode_steps_test",  default=10,  type=int)
# # # #     p.add_argument("--val_ensemble",    default=20,  type=int)
# # # #     p.add_argument("--fast_ensemble",   default=10,  type=int)
# # # #     p.add_argument("--val_freq",        default=5,   type=int)
# # # #     p.add_argument("--val_ade_freq",    default=5,   type=int)
# # # #     p.add_argument("--val_subset_size", default=500, type=int)
# # # #     p.add_argument("--use_ema",         action="store_true", default=True)
# # # #     p.add_argument("--ema_decay",       default=0.992, type=float)
# # # #     p.add_argument("--swa_start_epoch", default=60,   type=int)
# # # #     p.add_argument("--teacher_forcing", action="store_true", default=True)
# # # #     p.add_argument("--output_dir",  default="runs/v40c",        type=str)
# # # #     p.add_argument("--metrics_csv", default="metrics_v40c.csv", type=str)
# # # #     p.add_argument("--predict_csv", default="predictions.csv", type=str)
# # # #     p.add_argument("--gpu_num",     default="0", type=str)
# # # #     p.add_argument("--delim",       default=" ")
# # # #     p.add_argument("--skip",        default=1,   type=int)
# # # #     p.add_argument("--min_ped",     default=1,   type=int)
# # # #     p.add_argument("--threshold",   default=0.002, type=float)
# # # #     p.add_argument("--other_modal", default="gph")
# # # #     p.add_argument("--test_year",   default=None, type=int)
# # # #     p.add_argument("--resume",       default=None, type=str)
# # # #     p.add_argument("--resume_epoch", default=None, type=int)
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

# # # #     print("=" * 72)
# # # #     print("  TC-FlowMatching v40_clean  |  BEAT ST-TRANS EDITION")
# # # #     print("  TARGETS: DPE<136 | ATE<79.94 | CTE<93.58 | 72h<297")
# # # #     print("  LOSSES: v40_clean: 7 orthogonal losses, no redundancy")
# # # #     print("  EMA decay:", args.ema_decay)
# # # #     print("=" * 72)

# # # #     train_dataset, train_loader = data_loader(
# # # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # # #     val_dataset, val_loader = data_loader(
# # # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # # #     val_subset_loader = make_val_subset_loader(
# # # #         val_dataset, args.val_subset_size, args.batch_size,
# # # #         seq_collate, args.num_workers)

# # # #     test_loader = None
# # # #     try:
# # # #         _, test_loader = data_loader(
# # # #             args, {"root": args.dataset_root, "type": "test"},
# # # #             test=True, test_year=None)
# # # #     except Exception as e:
# # # #         print(f"  Warning: test loader: {e}")

# # # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # # #     print(f"  val   : {len(val_dataset)} seq")

# # # #     model = TCFlowMatching(
# # # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # # #         sigma_min=args.sigma_min, n_train_ens=args.n_train_ens,
# # # #         ctx_noise_scale=args.ctx_noise_scale,
# # # #         initial_sample_sigma=args.initial_sample_sigma,
# # # #         teacher_forcing=args.teacher_forcing,
# # # #         use_ema=args.use_ema, ema_decay=args.ema_decay,
# # # #     ).to(device)

# # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # #     print(f"  params  : {n_params:,}")
# # # #     model.init_ema()
# # # #     print(f"  EMA     : {'ON' if model._ema is not None else 'OFF'}"
# # # #           f"  (decay={args.ema_decay})")

# # # #     optimizer = optim.AdamW(model.parameters(),
# # # #                              lr=args.g_learning_rate,
# # # #                              weight_decay=args.weight_decay)
# # # #     saver  = HorizonAwareBestSaver(patience=args.patience, tol=1.5)
# # # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # # #     swa    = SWAManager(model, start_epoch=args.swa_start_epoch)

# # # #     steps_per_epoch = len(train_loader)
# # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # #     scheduler = get_cosine_schedule_with_warmup(
# # # #         optimizer, warmup, total_steps, min_lr=1e-6)

# # # #     # Resume
# # # #     has_scheduler_state   = False
# # # #     _ckpt_scheduler_state = None
# # # #     start_epoch = 0

# # # #     if args.resume is not None and os.path.exists(args.resume):
# # # #         print(f"  Loading checkpoint: {args.resume}")
# # # #         ckpt = torch.load(args.resume, map_location=device)
# # # #         m = _get_raw_model(model)
# # # #         missing, _ = m.load_state_dict(ckpt["model_state_dict"], strict=False)
# # # #         if missing:
# # # #             print(f"  ⚠  Missing keys ({len(missing)}): {missing[:3]}")
# # # #         if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
# # # #             ema_obj = _get_ema_obj(model)
# # # #             if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # # #                 for k, v in ckpt["ema_shadow"].items():
# # # #                     if k in ema_obj.shadow:
# # # #                         ema_obj.shadow[k].copy_(v.to(device))
# # # #         if "optimizer_state" in ckpt:
# # # #             try:
# # # #                 optimizer.load_state_dict(ckpt["optimizer_state"])
# # # #                 for state in optimizer.state.values():
# # # #                     for k, v in state.items():
# # # #                         if torch.is_tensor(v): state[k] = v.to(device)
# # # #             except Exception as e:
# # # #                 print(f"  ⚠  Optimizer restore failed: {e}")
# # # #         has_scheduler_state   = "scheduler_state" in ckpt
# # # #         _ckpt_scheduler_state = ckpt.get("scheduler_state", None)
# # # #         if "best_score" in ckpt:
# # # #             saver.best_score = ckpt["best_score"]
# # # #             saver.best_ade   = ckpt.get("best_ade",  float("inf"))
# # # #             saver.best_72h   = ckpt.get("best_72h",  float("inf"))
# # # #             saver.best_48h   = ckpt.get("best_48h",  float("inf"))
# # # #             saver.best_24h   = ckpt.get("best_24h",  float("inf"))
# # # #             saver.best_12h   = ckpt.get("best_12h",  float("inf"))
# # # #             saver.best_ate   = ckpt.get("best_ate",  float("inf"))
# # # #             saver.best_cte   = ckpt.get("best_cte",  float("inf"))
# # # #         start_epoch = (args.resume_epoch if args.resume_epoch is not None
# # # #                        else ckpt.get("epoch", 0) + 1)
# # # #         print(f"  → Resuming from epoch {start_epoch}")
# # # #     elif args.resume is not None:
# # # #         print(f"  ⚠  Checkpoint không tìm thấy: {args.resume}")

# # # #     try:
# # # #         model = torch.compile(model, mode="reduce-overhead")
# # # #         print("  torch.compile: enabled")
# # # #     except Exception:
# # # #         pass

# # # #     if has_scheduler_state and _ckpt_scheduler_state is not None:
# # # #         try:
# # # #             scheduler.load_state_dict(_ckpt_scheduler_state)
# # # #         except Exception as e:
# # # #             print(f"  ⚠  Scheduler restore failed: {e}")
# # # #             for _ in range(start_epoch * steps_per_epoch): scheduler.step()
# # # #     elif start_epoch > 0:
# # # #         for _ in range(start_epoch * steps_per_epoch): scheduler.step()

# # # #     print("=" * 72)
# # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch, start={start_epoch})")
# # # #     print("=" * 72)

# # # #     epoch_times = []
# # # #     train_start = time.perf_counter()

# # # #     for epoch in range(start_epoch, args.num_epochs):
# # # #         model.train()
# # # #         sum_loss = 0.0
# # # #         t0 = time.perf_counter()

# # # #         for i, batch in enumerate(train_loader):
# # # #             bl = move(list(batch), device)

# # # #             if epoch == start_epoch and i == 0:
# # # #                 _check_env(bl, train_dataset)

# # # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # # #                 bd = model.get_loss_breakdown(bl, epoch=epoch)

# # # #             optimizer.zero_grad()
# # # #             scaler.scale(bd["total"]).backward()
# # # #             scaler.unscale_(optimizer)
# # # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # # #             scaler.step(optimizer)
# # # #             scaler.update()
# # # #             scheduler.step()
# # # #             _call_ema_update(model)
# # # #             sum_loss += bd["total"].item()

# # # #             if i % 20 == 0:
# # # #                 lr = optimizer.param_groups[0]["lr"]
# # # #                 # ── LOG LINE v39: spd=speed_acc  cml=cumul_disp  vsm=vel_smooth
# # # #                 print(
# # # #                     f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # #                     f"  tot={bd['total'].item():.3f}"
# # # #                     f"  fm={bd.get('fm_mse',    0):.3f}"
# # # #                     f"  hor={bd.get('mse_hav',  0):.3f}"
# # # #                     f"  end={bd.get('endpoint', 0):.3f}"
# # # #                     f"  spd={bd.get('speed_acc',  0):.3f}"
# # # #                     f"  acc={bd.get('accel',      0):.3f}"   # ← v40 NEW
# # # #                     f"  dcp={bd.get('decomp',     0):.3f}"   # ← v40 NEW
# # # #                     f"  cns={bd.get('cons',       0):.3f}"   # ← v40 NEW
# # # #                     f"  hd={bd.get('heading',    0):.3f}"
# # # #                     f"  lr={lr:.2e}"
# # # #                 )

# # # #         ep_s  = time.perf_counter() - t0
# # # #         epoch_times.append(ep_s)
# # # #         avg_t = sum_loss / len(train_loader)
# # # #         swa.update(model, epoch)

# # # #         model.eval()
# # # #         val_loss = 0.0
# # # #         with torch.no_grad():
# # # #             for batch in val_loader:
# # # #                 bl_v = move(list(batch), device)
# # # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # # #                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
# # # #         avg_vl = val_loss / len(val_loader)
# # # #         print(f"  Epoch {epoch:>3}  train={avg_t:.4f}  val={avg_vl:.4f}"
# # # #               f"  time={ep_s:.0f}s")

# # # #         # Fast eval every epoch
# # # #         m_fast = evaluate_fast(model, val_subset_loader, device,
# # # #                                 args.ode_steps_train, args.pred_len,
# # # #                                 args.fast_ensemble)
# # # #         h12 = m_fast.get("12h", float("nan"))
# # # #         h24 = m_fast.get("24h", float("nan"))
# # # #         h48 = m_fast.get("48h", float("nan"))
# # # #         h72 = m_fast.get("72h", float("nan"))
# # # #         ate = m_fast.get("ATE_mean", float("nan"))
# # # #         cte = m_fast.get("CTE_mean", float("nan"))
# # # #         fast_score = _composite_score(m_fast)

# # # #         def ind(v, tgt):
# # # #             if not np.isfinite(v): return "?"
# # # #             return "🎯" if v < tgt else "❌"

# # # #         print(
# # # #             f"  [FAST ep{epoch}]"
# # # #             f"  ADE={m_fast.get('ADE', float('nan')):.1f}"
# # # #             f"  12h={h12:.0f}{ind(h12,50)}"
# # # #             f"  24h={h24:.0f}{ind(h24,100)}"
# # # #             f"  48h={h48:.0f}{ind(h48,200)}"
# # # #             f"  72h={h72:.0f}{ind(h72,297)}"
# # # #             f"  ATE={ate:.1f}{ind(ate,79.94)}"
# # # #             f"  CTE={cte:.1f}{ind(cte,93.58)}"
# # # #             f"  score={fast_score:.2f}"
# # # #         )

# # # #         # Full val every val_freq epochs
# # # #         if epoch % args.val_freq == 0:
# # # #             ema_obj = _get_ema_obj(model)
# # # #             try:
# # # #                 r_raw = evaluate_full_val_ade(
# # # #                     model, val_loader, device,
# # # #                     ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # #                     fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # #                     epoch=epoch, use_ema=False)
# # # #                 r_use = r_raw
# # # #                 if ema_obj is not None and epoch >= 5:
# # # #                     r_ema = evaluate_full_val_ade(
# # # #                         model, val_loader, device,
# # # #                         ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # #                         fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # # #                         epoch=epoch, use_ema=True, ema_obj=ema_obj)
# # # #                     if _composite_score(r_ema) < _composite_score(r_raw):
# # # #                         r_use = r_ema
# # # #                 saver.update(r_use, model, args.output_dir, epoch,
# # # #                              optimizer, scheduler, avg_t, avg_vl,
# # # #                              saver_ref=saver, min_epochs=args.min_epochs)
# # # #             except Exception as e:
# # # #                 print(f"  ⚠  Full val failed: {e}")
# # # #                 import traceback; traceback.print_exc()

# # # #         if epoch % 10 == 0 or epoch == args.num_epochs - 1:
# # # #             _save_checkpoint(
# # # #                 os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"),
# # # #                 epoch, model, optimizer, scheduler, saver, avg_t, avg_vl,
# # # #                 {"ade": m_fast.get("ADE", float("nan")),
# # # #                  "h48": h48, "h72": h72, "ate": ate, "cte": cte})

# # # #         if epoch % 10 == 9:
# # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")
# # # #             print(f"  📊 Best: score={saver.best_score:.2f}  "
# # # #                   f"72h={saver.best_72h:.0f}km  "
# # # #                   f"ATE={saver.best_ate:.1f}km  CTE={saver.best_cte:.1f}km")

# # # #         if saver.early_stop:
# # # #             print(f"  ⛔ Early stopping at epoch {epoch}")
# # # #             break

# # # #     # Final SWA eval
# # # #     print(f"\n{'='*72}")
# # # #     swa_backup = swa.apply_to(model)
# # # #     if swa_backup is not None:
# # # #         print("  Evaluating SWA weights...")
# # # #         r_swa = evaluate_full_val_ade(
# # # #             model, val_loader, device,
# # # #             ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # # #             fast_ensemble=args.val_ensemble, metrics_csv=metrics_csv,
# # # #             epoch=9999, use_ema=False)
# # # #         if _composite_score(r_swa) < saver.best_score:
# # # #             _save_checkpoint(
# # # #                 os.path.join(args.output_dir, "best_swa.pth"),
# # # #                 9999, model, optimizer, scheduler, saver,
# # # #                 avg_t=0.0, avg_vl=0.0, metrics={"tag": "best_swa"})
# # # #             print("  ✅ SWA checkpoint saved")
# # # #         swa.restore(model, swa_backup)

# # # #     total_h = (time.perf_counter() - train_start) / 3600
# # # #     print(f"\n  Best composite score: {saver.best_score:.2f}")
# # # #     print(f"  Best: ADE={saver.best_ade:.1f}  "
# # # #           f"12h={saver.best_12h:.0f}  24h={saver.best_24h:.0f}  "
# # # #           f"48h={saver.best_48h:.0f}  72h={saver.best_72h:.0f}  "
# # # #           f"ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}")
# # # #     print(f"  Total time: {total_h:.2f}h")

# # # #     best_r = {"ADE": saver.best_ade, "12h": saver.best_12h,
# # # #               "24h": saver.best_24h, "48h": saver.best_48h,
# # # #               "72h": saver.best_72h, "ATE_mean": saver.best_ate,
# # # #               "CTE_mean": saver.best_cte}
# # # #     beat = _beat_report(best_r)
# # # #     if beat:
# # # #         print(f"\n  🏆🏆🏆 {beat} 🏆🏆🏆")
# # # #     else:
# # # #         print(f"\n  Need: DPE<136.41, ATE<79.94, CTE<93.58, 72h<297")
# # # #     print("=" * 72)


# # # # if __name__ == "__main__":
# # # #     args = get_args()
# # # #     np.random.seed(42); torch.manual_seed(42)
# # # #     if torch.cuda.is_available():
# # # #         torch.cuda.manual_seed_all(42)
# # # #     main(args)

# # # """
# # # scripts/train_flowmatching.py — v40  v41: SPEED-BIAS + OVERFIT FIX
# # # ═══════════════════════════════════════════════════════════════════════════════
# # # THAY ĐỔI DUY NHẤT so với v36 train script (doc 5):

# # #   Log print line — update key names cho losses v39:
# # #     ĐỔI:  vel={vel_smooth}  atc={ate_cte}  str={steering}
# # #     THÀNH: spd={speed_acc}  cml={cumul_disp}  acc={accel}  dcp={decomp}  cns={cons}

# # #   Tất cả code khác GIỐNG HỆT v36 train script.
# # # """
# # # from __future__ import annotations

# # # import sys
# # # import os
# # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # import argparse
# # # import time
# # # import random

# # # import numpy as np
# # # import torch
# # # import torch.optim as optim
# # # from torch.amp import autocast, GradScaler
# # # from torch.utils.data import DataLoader, Subset

# # # from Model.data.loader_training import data_loader
# # # from Model.flow_matching_model import TCFlowMatching
# # # from Model.utils import get_cosine_schedule_with_warmup
# # # from utils.metrics import (
# # #     StepErrorAccumulator,
# # #     save_metrics_csv,
# # #     haversine_km_torch,
# # #     denorm_torch,
# # #     HORIZON_STEPS,
# # #     DatasetMetrics,
# # # )

# # # try:
# # #     from utils.metrics import haversine_and_atecte_torch
# # #     HAS_ATECTE = True
# # # except ImportError:
# # #     HAS_ATECTE = False
# # #     print("  ⚠  haversine_and_atecte_torch not found — ATE/CTE disabled")


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Composite Score v2
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def _composite_score(result: dict) -> float:
# # #     ade = result.get("ADE", float("inf"))
# # #     h12 = result.get("12h", float("inf"))
# # #     h24 = result.get("24h", float("inf"))
# # #     h48 = result.get("48h", float("inf"))
# # #     h72 = result.get("72h", float("inf"))
# # #     ate = result.get("ATE_mean", float("inf"))
# # #     cte = result.get("CTE_mean", float("inf"))
# # #     if not np.isfinite(ate): ate = ade * 0.46
# # #     if not np.isfinite(cte): cte = ade * 0.53
# # #     return 100.0 * (
# # #         0.05 * (ade / 136.0)
# # #         + 0.05 * (h12 / 50.0)
# # #         + 0.10 * (h24 / 100.0)
# # #         + 0.15 * (h48 / 200.0)
# # #         + 0.35 * (h72 / 300.0)
# # #         + 0.15 * (ate / 80.0)
# # #         + 0.15 * (cte / 94.0)
# # #     )


# # # def _beat_report(r: dict) -> str:
# # #     parts = []
# # #     if np.isfinite(r.get("ADE", float("inf"))) and r["ADE"] < 136.41:
# # #         parts.append(f"DPE✅{r['ADE']:.1f}")
# # #     if np.isfinite(r.get("ATE_mean", float("inf"))) and r["ATE_mean"] < 79.94:
# # #         parts.append(f"ATE✅{r['ATE_mean']:.1f}")
# # #     if np.isfinite(r.get("CTE_mean", float("inf"))) and r["CTE_mean"] < 93.58:
# # #         parts.append(f"CTE✅{r['CTE_mean']:.1f}")
# # #     if np.isfinite(r.get("72h", float("inf"))) and r["72h"] < 297.0:
# # #         parts.append(f"72h✅{r['72h']:.1f}")
# # #     return "🏆 BEAT: " + " ".join(parts) if parts else ""


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Helpers — unchanged
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


# # # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# # #                             collate_fn, num_workers):
# # #     n   = len(val_dataset)
# # #     rng = random.Random(42)
# # #     idx = rng.sample(range(n), min(subset_size, n))
# # #     return DataLoader(Subset(val_dataset, idx),
# # #                       batch_size=batch_size, shuffle=False,
# # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # def _get_ema_obj(model):
# # #     if hasattr(model, '_ema') and model._ema is not None:
# # #         return model._ema
# # #     if hasattr(model, '_orig_mod'):
# # #         orig = model._orig_mod
# # #         if hasattr(orig, '_ema') and orig._ema is not None:
# # #             return orig._ema
# # #     return None


# # # def _call_ema_update(model):
# # #     if hasattr(model, '_orig_mod'):
# # #         orig = model._orig_mod
# # #         if hasattr(orig, 'ema_update'):
# # #             orig.ema_update(); return
# # #     if hasattr(model, 'ema_update'):
# # #         model.ema_update()


# # # def _get_raw_model(model):
# # #     return model._orig_mod if hasattr(model, '_orig_mod') else model


# # # def _save_checkpoint(path, epoch, model, optimizer, scheduler,
# # #                      saver, avg_t, avg_vl, metrics=None):
# # #     m = _get_raw_model(model)
# # #     ema_obj = _get_ema_obj(model)
# # #     ema_sd  = None
# # #     if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # #         try:
# # #             ema_sd = {k: v.cpu().clone() for k, v in ema_obj.shadow.items()}
# # #         except Exception:
# # #             ema_sd = None
# # #     payload = {
# # #         "epoch": epoch, "model_state_dict": m.state_dict(),
# # #         "optimizer_state": optimizer.state_dict(),
# # #         "scheduler_state": scheduler.state_dict(),
# # #         "ema_shadow": ema_sd,
# # #         "best_score": saver.best_score, "best_ade": saver.best_ade,
# # #         "best_72h": saver.best_72h, "best_48h": saver.best_48h,
# # #         "best_24h": saver.best_24h, "best_12h": saver.best_12h,
# # #         "best_ate": getattr(saver, 'best_ate', float("inf")),
# # #         "best_cte": getattr(saver, 'best_cte', float("inf")),
# # #         "train_loss": avg_t, "val_loss": avg_vl,
# # #     }
# # #     if metrics: payload.update(metrics)
# # #     torch.save(payload, path)


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  SWA — unchanged
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class SWAManager:
# # #     def __init__(self, model, start_epoch=60):
# # #         self.start_epoch = start_epoch
# # #         self.n_averaged  = 0
# # #         self.avg_state   = None

# # #     def update(self, model, epoch):
# # #         if epoch < self.start_epoch: return
# # #         m  = _get_raw_model(model)
# # #         sd = {k: v.detach().clone() for k, v in m.state_dict().items()
# # #               if v.dtype.is_floating_point}
# # #         if self.avg_state is None:
# # #             self.avg_state = sd; self.n_averaged = 1
# # #         else:
# # #             n = self.n_averaged
# # #             for k in self.avg_state:
# # #                 if k in sd:
# # #                     self.avg_state[k].mul_(n/(n+1)).add_(sd[k], alpha=1.0/(n+1))
# # #             self.n_averaged += 1

# # #     def apply_to(self, model):
# # #         if self.avg_state is None: return None
# # #         m  = _get_raw_model(model)
# # #         sd = m.state_dict()
# # #         backup = {k: sd[k].detach().clone() for k in self.avg_state if k in sd}
# # #         for k, v in self.avg_state.items():
# # #             if k in sd: sd[k].copy_(v)
# # #         return backup

# # #     def restore(self, model, backup):
# # #         if backup is None: return
# # #         m  = _get_raw_model(model)
# # #         sd = m.state_dict()
# # #         for k, v in backup.items():
# # #             if k in sd: sd[k].copy_(v)


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Evaluation — unchanged
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def _eval_batch_atecte(pred_norm, gt_norm):
# # #     pred_d = denorm_torch(pred_norm)
# # #     gt_d   = denorm_torch(gt_norm)
# # #     if HAS_ATECTE:
# # #         dist, ate, cte = haversine_and_atecte_torch(pred_d, gt_d, unit_01deg=True)
# # #         return dist, ate, cte
# # #     else:
# # #         dist = haversine_km_torch(pred_d, gt_d)
# # #         return dist, None, None


# # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=15):
# # #     model.eval()
# # #     acc = StepErrorAccumulator(pred_len)
# # #     t0  = time.perf_counter()
# # #     n   = 0
# # #     spread_per_step = []
# # #     with torch.no_grad():
# # #         for batch in loader:
# # #             bl = move(list(batch), device)
# # #             pred, _, all_trajs = model.sample(
# # #                 bl, num_ensemble=max(fast_ensemble,20), ddim_steps=ode_steps,
# # #                 importance_weight=True)
# # #             T_active  = pred.shape[0]
# # #             gt_sliced = bl[1][:T_active]
# # #             dist, ate, cte = _eval_batch_atecte(pred, gt_sliced)
# # #             acc.update(dist, ate_km=ate, cte_km=cte)
# # #             step_spreads = []
# # #             for t in range(all_trajs.shape[1]):
# # #                 step_data = all_trajs[:, t, :, :]
# # #                 spread = ((step_data[:, :, 0].std(0)**2
# # #                            + step_data[:, :, 1].std(0)**2).sqrt() * 500.0).mean().item()
# # #                 step_spreads.append(spread)
# # #             spread_per_step.append(step_spreads)
# # #             n += 1
# # #     r = acc.compute()
# # #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# # #     if spread_per_step:
# # #         r["spread_72h_km"] = float(np.array(spread_per_step)[:, -1].mean())
# # #     return r


# # # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# # #                            fast_ensemble, metrics_csv, epoch,
# # #                            use_ema=False, ema_obj=None):
# # #     backup = None
# # #     if use_ema and ema_obj is not None:
# # #         try:
# # #             backup = ema_obj.apply_to(model)
# # #         except Exception as e:
# # #             print(f"  ⚠  EMA apply_to failed: {e}")
# # #             backup = None; use_ema = False

# # #     model.eval()
# # #     acc = StepErrorAccumulator(pred_len)
# # #     t0  = time.perf_counter()
# # #     n   = 0
# # #     with torch.no_grad():
# # #         for batch in val_loader:
# # #             bl = move(list(batch), device)
# # #             pred, _, _ = model.sample(
# # #                 bl, num_ensemble=max(fast_ensemble, 20),
# # #                 ddim_steps=max(ode_steps, 20), importance_weight=True)
# # #             T_pred = pred.shape[0]
# # #             gt     = bl[1][:T_pred]
# # #             dist, ate, cte = _eval_batch_atecte(pred, gt)
# # #             acc.update(dist, ate_km=ate, cte_km=cte)
# # #             n += 1

# # #     r       = acc.compute()
# # #     elapsed = time.perf_counter() - t0
# # #     score   = _composite_score(r)
# # #     tag     = "EMA" if use_ema else "RAW"

# # #     ade_v = r.get("ADE",      float("nan"))
# # #     fde_v = r.get("FDE",      float("nan"))
# # #     h12_v = r.get("12h",      float("nan"))
# # #     h24_v = r.get("24h",      float("nan"))
# # #     h48_v = r.get("48h",      float("nan"))
# # #     h72_v = r.get("72h",      float("nan"))
# # #     ate_v = r.get("ATE_mean", float("nan"))
# # #     cte_v = r.get("CTE_mean", float("nan"))

# # #     def ind(v, tgt):
# # #         if not np.isfinite(v): return ""
# # #         return "✅" if v < tgt else "❌"

# # #     print(f"\n{'='*70}")
# # #     print(f"  [FULL VAL ({tag})  ep={epoch}  {elapsed:.0f}s  {n} batches]")
# # #     print(f"  ADE={ade_v:.1f}{ind(ade_v,136.41)}  FDE={fde_v:.1f} km")
# # #     print(f"  12h={h12_v:.0f}{ind(h12_v,50)}  24h={h24_v:.0f}{ind(h24_v,100)}  "
# # #           f"48h={h48_v:.0f}{ind(h48_v,200)}  72h={h72_v:.0f}{ind(h72_v,297)} km")
# # #     if np.isfinite(ate_v):
# # #         print(f"  ATE={ate_v:.1f}{ind(ate_v,79.94)}  CTE={cte_v:.1f}{ind(cte_v,93.58)}"
# # #               f"  [ST-Trans: ATE=79.94 CTE=93.58]")
# # #     beat = _beat_report(r)
# # #     if beat: print(f"  {beat}")
# # #     print(f"  Score = {score:.2f}  (< 100 = beat ST-Trans all metrics)")
# # #     print(f"{'='*70}\n")

# # #     from datetime import datetime
# # #     dm = DatasetMetrics(
# # #         ade=ade_v if np.isfinite(ade_v) else 0.0,
# # #         fde=fde_v if np.isfinite(fde_v) else 0.0,
# # #         ugde_12h=h12_v if np.isfinite(h12_v) else 0.0,
# # #         ugde_24h=h24_v if np.isfinite(h24_v) else 0.0,
# # #         ugde_48h=h48_v if np.isfinite(h48_v) else 0.0,
# # #         ugde_72h=h72_v if np.isfinite(h72_v) else 0.0,
# # #         ate_abs_mean=ate_v if np.isfinite(ate_v) else 0.0,
# # #         cte_abs_mean=cte_v if np.isfinite(cte_v) else 0.0,
# # #         n_total=r.get("n_samples", 0),
# # #         timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
# # #     )
# # #     save_metrics_csv(dm, metrics_csv, tag=f"val_{tag}_ep{epoch:03d}")

# # #     if backup is not None:
# # #         try:
# # #             ema_obj.restore(model, backup)
# # #         except Exception as e:
# # #             print(f"  ⚠  EMA restore failed: {e}")
# # #     return r


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  BestModelSaver — unchanged
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # class HorizonAwareBestSaver:
# # #     def __init__(self, patience=30, tol=1.5):
# # #         self.patience   = patience
# # #         self.tol        = tol
# # #         self.counter    = 0
# # #         self.early_stop = False
# # #         self.best_score = float("inf")
# # #         self.best_ade   = float("inf")
# # #         self.best_12h   = float("inf")
# # #         self.best_24h   = float("inf")
# # #         self.best_48h   = float("inf")
# # #         self.best_72h   = float("inf")
# # #         self.best_ate   = float("inf")
# # #         self.best_cte   = float("inf")

# # #     def update(self, r, model, out_dir, epoch, optimizer, scheduler,
# # #                tl, vl, saver_ref, min_epochs=30):
# # #         ade   = r.get("ADE", float("inf"))
# # #         h12   = r.get("12h", float("inf"))
# # #         h24   = r.get("24h", float("inf"))
# # #         h48   = r.get("48h", float("inf"))
# # #         h72   = r.get("72h", float("inf"))
# # #         ate   = r.get("ATE_mean", ade * 0.46 if np.isfinite(ade) else float("inf"))
# # #         cte   = r.get("CTE_mean", ade * 0.53 if np.isfinite(ade) else float("inf"))
# # #         score = _composite_score(r)
# # #         improved_any = False

# # #         for metric_val, best_attr, fname, extra in [
# # #             (ade, 'best_ade', 'best_ade.pth',  {"ade": ade}),
# # #             (h72, 'best_72h', 'best_72h.pth',  {"h72": h72}),
# # #             (ate, 'best_ate', 'best_ate.pth',  {"ate": ate, "cte": cte}),
# # #             (cte, 'best_cte', 'best_cte.pth',  {"ate": ate, "cte": cte}),
# # #         ]:
# # #             if metric_val < getattr(self, best_attr):
# # #                 setattr(self, best_attr, metric_val)
# # #                 improved_any = True
# # #                 _save_checkpoint(os.path.join(out_dir, fname),
# # #                                   epoch, model, optimizer, scheduler,
# # #                                   saver_ref, tl, vl, extra)

# # #         if h48 < self.best_48h: self.best_48h = h48; improved_any = True
# # #         if h24 < self.best_24h: self.best_24h = h24; improved_any = True
# # #         if h12 < self.best_12h: self.best_12h = h12; improved_any = True

# # #         if score < self.best_score - self.tol:
# # #             self.best_score = score
# # #             self.counter    = 0
# # #             _save_checkpoint(
# # #                 os.path.join(out_dir, "best_model.pth"),
# # #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# # #                 {"ade": ade, "h12": h12, "h24": h24, "h48": h48, "h72": h72,
# # #                  "ate": ate, "cte": cte, "composite_score": score})
# # #             print(f"  ✅ Best COMPOSITE={score:.2f}  ADE={ade:.1f}  "
# # #                   f"12h={h12:.0f}  24h={h24:.0f}  48h={h48:.0f}  "
# # #                   f"72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}  (ep {epoch})")
# # #         else:
# # #             if not improved_any:
# # #                 self.counter += 1
# # #             print(f"  No improvement {self.counter}/{self.patience}"
# # #                   f"  (best={self.best_score:.2f} cur={score:.2f})"
# # #                   f"  72h={h72:.0f}↓{self.best_72h:.0f}"
# # #                   f"  ATE={ate:.1f}↓{self.best_ate:.1f}")

# # #         if epoch >= min_epochs and self.counter >= self.patience:
# # #             self.early_stop = True


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Env diagnostic — unchanged
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def _check_env(bl, train_dataset):
# # #     try:    env_dir = train_dataset.env_path
# # #     except AttributeError:
# # #         try:    env_dir = train_dataset.dataset.env_path
# # #         except: env_dir = "UNKNOWN"
# # #     print(f"  Env path: {env_dir}")
# # #     env_data = bl[13]
# # #     if env_data is None:
# # #         print("  ⚠️  env_data is None"); return
# # #     for key in ("gph500_mean", "gph500_center", "u500_mean", "v500_mean"):
# # #         if key not in env_data:
# # #             print(f"  ⚠️  {key} MISSING"); continue
# # #         v    = env_data[key]
# # #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# # #         print(f"  {'✅' if zero < 80 else '⚠️'}  {key}: "
# # #               f"mean={v.mean().item():.4f} std={v.std().item():.4f} zero={zero:.1f}%")


# # # # ══════════════════════════════════════════════════════════════════════════════
# # # #  Args — unchanged
# # # # ══════════════════════════════════════════════════════════════════════════════

# # # def get_args():
# # #     p = argparse.ArgumentParser(
# # #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# # #     p.add_argument("--obs_len",         default=8,          type=int)
# # #     p.add_argument("--pred_len",        default=12,         type=int)
# # #     p.add_argument("--batch_size",      default=32,         type=int)
# # #     p.add_argument("--num_epochs",      default=100,        type=int)
# # #     p.add_argument("--g_learning_rate", default=3e-4,       type=float)
# # #     p.add_argument("--weight_decay",    default=5e-4,       type=float)  # v41: 1e-4→5e-4
# # #     p.add_argument("--warmup_epochs",   default=5,          type=int)
# # #     p.add_argument("--grad_clip",       default=1.0,        type=float)
# # #     p.add_argument("--patience",        default=30,         type=int)
# # #     p.add_argument("--min_epochs",      default=30,         type=int)
# # #     p.add_argument("--n_train_ens",     default=4,          type=int)
# # #     p.add_argument("--use_amp",         action="store_true")
# # #     p.add_argument("--num_workers",     default=2,          type=int)
# # #     p.add_argument("--sigma_min",            default=0.02,  type=float)
# # #     p.add_argument("--ctx_noise_scale",      default=0.01,  type=float)
# # #     p.add_argument("--initial_sample_sigma", default=0.03,  type=float)
# # #     p.add_argument("--ode_steps_train", default=10,  type=int)
# # #     p.add_argument("--ode_steps_val",   default=10,  type=int)
# # #     p.add_argument("--ode_steps_test",  default=10,  type=int)
# # #     p.add_argument("--val_ensemble",    default=20,  type=int)
# # #     p.add_argument("--fast_ensemble",   default=10,  type=int)
# # #     p.add_argument("--val_freq",        default=5,   type=int)
# # #     p.add_argument("--val_ade_freq",    default=5,   type=int)
# # #     p.add_argument("--val_subset_size", default=500, type=int)
# # #     p.add_argument("--use_ema",         action="store_true", default=True)
# # #     p.add_argument("--ema_decay",       default=0.992, type=float)
# # #     p.add_argument("--swa_start_epoch", default=60,   type=int)
# # #     p.add_argument("--teacher_forcing", action="store_true", default=True)
# # #     p.add_argument("--output_dir",  default="runs/v41",        type=str)
# # #     p.add_argument("--metrics_csv", default="metrics_v41.csv", type=str)
# # #     p.add_argument("--predict_csv", default="predictions.csv", type=str)
# # #     p.add_argument("--gpu_num",     default="0", type=str)
# # #     p.add_argument("--delim",       default=" ")
# # #     p.add_argument("--skip",        default=1,   type=int)
# # #     p.add_argument("--min_ped",     default=1,   type=int)
# # #     p.add_argument("--threshold",   default=0.002, type=float)
# # #     p.add_argument("--other_modal", default="gph")
# # #     p.add_argument("--test_year",   default=None, type=int)
# # #     p.add_argument("--resume",       default=None, type=str)
# # #     p.add_argument("--resume_epoch", default=None, type=int)
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

# # #     print("=" * 72)
# # #     print("  TC-FlowMatching v41  |  BEAT ST-TRANS EDITION")
# # #     print("  TARGETS: DPE<136 | ATE<79.94 | CTE<93.58 | 72h<297")
# # #     print("  LOSSES: v41: dropout 0.15 + speed_aug + weight_decay 5e-4")
# # #     print("  EMA decay:", args.ema_decay)
# # #     print("=" * 72)

# # #     train_dataset, train_loader = data_loader(
# # #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# # #     val_dataset, val_loader = data_loader(
# # #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# # #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# # #     val_subset_loader = make_val_subset_loader(
# # #         val_dataset, args.val_subset_size, args.batch_size,
# # #         seq_collate, args.num_workers)

# # #     test_loader = None
# # #     try:
# # #         _, test_loader = data_loader(
# # #             args, {"root": args.dataset_root, "type": "test"},
# # #             test=True, test_year=None)
# # #     except Exception as e:
# # #         print(f"  Warning: test loader: {e}")

# # #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# # #     print(f"  val   : {len(val_dataset)} seq")

# # #     model = TCFlowMatching(
# # #         pred_len=args.pred_len, obs_len=args.obs_len,
# # #         sigma_min=args.sigma_min, n_train_ens=args.n_train_ens,
# # #         ctx_noise_scale=args.ctx_noise_scale,
# # #         initial_sample_sigma=args.initial_sample_sigma,
# # #         teacher_forcing=args.teacher_forcing,
# # #         use_ema=args.use_ema, ema_decay=args.ema_decay,
# # #     ).to(device)

# # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # #     print(f"  params  : {n_params:,}")
# # #     model.init_ema()
# # #     print(f"  EMA     : {'ON' if model._ema is not None else 'OFF'}"
# # #           f"  (decay={args.ema_decay})")

# # #     optimizer = optim.AdamW(model.parameters(),
# # #                              lr=args.g_learning_rate,
# # #                              weight_decay=args.weight_decay)
# # #     saver  = HorizonAwareBestSaver(patience=args.patience, tol=1.5)
# # #     scaler = GradScaler("cuda", enabled=args.use_amp)
# # #     swa    = SWAManager(model, start_epoch=args.swa_start_epoch)

# # #     steps_per_epoch = len(train_loader)
# # #     total_steps     = steps_per_epoch * args.num_epochs
# # #     warmup          = steps_per_epoch * args.warmup_epochs
# # #     scheduler = get_cosine_schedule_with_warmup(
# # #         optimizer, warmup, total_steps, min_lr=1e-6)

# # #     # Resume
# # #     has_scheduler_state   = False
# # #     _ckpt_scheduler_state = None
# # #     start_epoch = 0

# # #     if args.resume is not None and os.path.exists(args.resume):
# # #         print(f"  Loading checkpoint: {args.resume}")
# # #         ckpt = torch.load(args.resume, map_location=device)
# # #         m = _get_raw_model(model)
# # #         missing, _ = m.load_state_dict(ckpt["model_state_dict"], strict=False)
# # #         if missing:
# # #             print(f"  ⚠  Missing keys ({len(missing)}): {missing[:3]}")
# # #         if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
# # #             ema_obj = _get_ema_obj(model)
# # #             if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# # #                 for k, v in ckpt["ema_shadow"].items():
# # #                     if k in ema_obj.shadow:
# # #                         ema_obj.shadow[k].copy_(v.to(device))
# # #         if "optimizer_state" in ckpt:
# # #             try:
# # #                 optimizer.load_state_dict(ckpt["optimizer_state"])
# # #                 for state in optimizer.state.values():
# # #                     for k, v in state.items():
# # #                         if torch.is_tensor(v): state[k] = v.to(device)
# # #             except Exception as e:
# # #                 print(f"  ⚠  Optimizer restore failed: {e}")
# # #         has_scheduler_state   = "scheduler_state" in ckpt
# # #         _ckpt_scheduler_state = ckpt.get("scheduler_state", None)
# # #         if "best_score" in ckpt:
# # #             saver.best_score = ckpt["best_score"]
# # #             saver.best_ade   = ckpt.get("best_ade",  float("inf"))
# # #             saver.best_72h   = ckpt.get("best_72h",  float("inf"))
# # #             saver.best_48h   = ckpt.get("best_48h",  float("inf"))
# # #             saver.best_24h   = ckpt.get("best_24h",  float("inf"))
# # #             saver.best_12h   = ckpt.get("best_12h",  float("inf"))
# # #             saver.best_ate   = ckpt.get("best_ate",  float("inf"))
# # #             saver.best_cte   = ckpt.get("best_cte",  float("inf"))
# # #         start_epoch = (args.resume_epoch if args.resume_epoch is not None
# # #                        else ckpt.get("epoch", 0) + 1)
# # #         print(f"  → Resuming from epoch {start_epoch}")
# # #     elif args.resume is not None:
# # #         print(f"  ⚠  Checkpoint không tìm thấy: {args.resume}")

# # #     try:
# # #         model = torch.compile(model, mode="reduce-overhead")
# # #         print("  torch.compile: enabled")
# # #     except Exception:
# # #         pass

# # #     if has_scheduler_state and _ckpt_scheduler_state is not None:
# # #         try:
# # #             scheduler.load_state_dict(_ckpt_scheduler_state)
# # #         except Exception as e:
# # #             print(f"  ⚠  Scheduler restore failed: {e}")
# # #             for _ in range(start_epoch * steps_per_epoch): scheduler.step()
# # #     elif start_epoch > 0:
# # #         for _ in range(start_epoch * steps_per_epoch): scheduler.step()

# # #     print("=" * 72)
# # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch, start={start_epoch})")
# # #     print("=" * 72)

# # #     epoch_times = []
# # #     train_start = time.perf_counter()

# # #     for epoch in range(start_epoch, args.num_epochs):
# # #         model.train()
# # #         sum_loss = 0.0
# # #         t0 = time.perf_counter()

# # #         for i, batch in enumerate(train_loader):
# # #             bl = move(list(batch), device)

# # #             if epoch == start_epoch and i == 0:
# # #                 _check_env(bl, train_dataset)

# # #             with autocast(device_type="cuda", enabled=args.use_amp):
# # #                 bd = model.get_loss_breakdown(bl, epoch=epoch)

# # #             optimizer.zero_grad()
# # #             scaler.scale(bd["total"]).backward()
# # #             scaler.unscale_(optimizer)
# # #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# # #             scaler.step(optimizer)
# # #             scaler.update()
# # #             scheduler.step()
# # #             _call_ema_update(model)
# # #             sum_loss += bd["total"].item()

# # #             if i % 20 == 0:
# # #                 lr = optimizer.param_groups[0]["lr"]
# # #                 # ── LOG LINE v39: spd=speed_acc  cml=cumul_disp  vsm=vel_smooth
# # #                 print(
# # #                     f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # #                     f"  tot={bd['total'].item():.3f}"
# # #                     f"  fm={bd.get('fm_mse',    0):.3f}"
# # #                     f"  hor={bd.get('mse_hav',  0):.3f}"
# # #                     f"  end={bd.get('endpoint', 0):.3f}"
# # #                     f"  spd={bd.get('speed_acc',  0):.3f}"
# # #                     f"  acc={bd.get('accel',      0):.3f}"   # ← v40 NEW
# # #                     f"  dcp={bd.get('decomp',     0):.3f}"   # ← v40 NEW
# # #                     f"  cns={bd.get('cons',       0):.3f}"   # ← v40 NEW
# # #                     f"  hd={bd.get('heading',    0):.3f}"
# # #                     f"  lr={lr:.2e}"
# # #                 )
# # #                 # Sau epoch 5 (hoặc bất kỳ epoch nào):
# # #                 if epoch == 5:
# # #                     diagnose_fno_encoder(model, val_subset_loader, device)

# # #         ep_s  = time.perf_counter() - t0
# # #         epoch_times.append(ep_s)
# # #         avg_t = sum_loss / len(train_loader)
# # #         swa.update(model, epoch)

# # #         model.eval()
# # #         val_loss = 0.0
# # #         with torch.no_grad():
# # #             for batch in val_loader:
# # #                 bl_v = move(list(batch), device)
# # #                 with autocast(device_type="cuda", enabled=args.use_amp):
# # #                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
# # #         avg_vl = val_loss / len(val_loader)
# # #         print(f"  Epoch {epoch:>3}  train={avg_t:.4f}  val={avg_vl:.4f}"
# # #               f"  time={ep_s:.0f}s")

# # #         # Fast eval every epoch
# # #         m_fast = evaluate_fast(model, val_subset_loader, device,
# # #                                 args.ode_steps_train, args.pred_len,
# # #                                 args.fast_ensemble)
# # #         h12 = m_fast.get("12h", float("nan"))
# # #         h24 = m_fast.get("24h", float("nan"))
# # #         h48 = m_fast.get("48h", float("nan"))
# # #         h72 = m_fast.get("72h", float("nan"))
# # #         ate = m_fast.get("ATE_mean", float("nan"))
# # #         cte = m_fast.get("CTE_mean", float("nan"))
# # #         fast_score = _composite_score(m_fast)

# # #         def ind(v, tgt):
# # #             if not np.isfinite(v): return "?"
# # #             return "🎯" if v < tgt else "❌"

# # #         print(
# # #             f"  [FAST ep{epoch}]"
# # #             f"  ADE={m_fast.get('ADE', float('nan')):.1f}"
# # #             f"  12h={h12:.0f}{ind(h12,50)}"
# # #             f"  24h={h24:.0f}{ind(h24,100)}"
# # #             f"  48h={h48:.0f}{ind(h48,200)}"
# # #             f"  72h={h72:.0f}{ind(h72,297)}"
# # #             f"  ATE={ate:.1f}{ind(ate,79.94)}"
# # #             f"  CTE={cte:.1f}{ind(cte,93.58)}"
# # #             f"  score={fast_score:.2f}"
# # #         )

# # #         # Full val every val_freq epochs
# # #         if epoch % args.val_freq == 0:
# # #             ema_obj = _get_ema_obj(model)
# # #             try:
# # #                 r_raw = evaluate_full_val_ade(
# # #                     model, val_loader, device,
# # #                     ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # #                     fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # #                     epoch=epoch, use_ema=False)
# # #                 r_use = r_raw
# # #                 if ema_obj is not None and epoch >= 5:
# # #                     r_ema = evaluate_full_val_ade(
# # #                         model, val_loader, device,
# # #                         ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # #                         fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# # #                         epoch=epoch, use_ema=True, ema_obj=ema_obj)
# # #                     if _composite_score(r_ema) < _composite_score(r_raw):
# # #                         r_use = r_ema
# # #                 saver.update(r_use, model, args.output_dir, epoch,
# # #                              optimizer, scheduler, avg_t, avg_vl,
# # #                              saver_ref=saver, min_epochs=args.min_epochs)
# # #             except Exception as e:
# # #                 print(f"  ⚠  Full val failed: {e}")
# # #                 import traceback; traceback.print_exc()

# # #         if epoch % 10 == 0 or epoch == args.num_epochs - 1:
# # #             _save_checkpoint(
# # #                 os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"),
# # #                 epoch, model, optimizer, scheduler, saver, avg_t, avg_vl,
# # #                 {"ade": m_fast.get("ADE", float("nan")),
# # #                  "h48": h48, "h72": h72, "ate": ate, "cte": cte})

# # #         if epoch % 10 == 9:
# # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")
# # #             print(f"  📊 Best: score={saver.best_score:.2f}  "
# # #                   f"72h={saver.best_72h:.0f}km  "
# # #                   f"ATE={saver.best_ate:.1f}km  CTE={saver.best_cte:.1f}km")

# # #         if saver.early_stop:
# # #             print(f"  ⛔ Early stopping at epoch {epoch}")
# # #             break

# # #     # Final SWA eval
# # #     print(f"\n{'='*72}")
# # #     swa_backup = swa.apply_to(model)
# # #     if swa_backup is not None:
# # #         print("  Evaluating SWA weights...")
# # #         r_swa = evaluate_full_val_ade(
# # #             model, val_loader, device,
# # #             ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# # #             fast_ensemble=args.val_ensemble, metrics_csv=metrics_csv,
# # #             epoch=9999, use_ema=False)
# # #         if _composite_score(r_swa) < saver.best_score:
# # #             _save_checkpoint(
# # #                 os.path.join(args.output_dir, "best_swa.pth"),
# # #                 9999, model, optimizer, scheduler, saver,
# # #                 avg_t=0.0, avg_vl=0.0, metrics={"tag": "best_swa"})
# # #             print("  ✅ SWA checkpoint saved")
# # #         swa.restore(model, swa_backup)

# # #     total_h = (time.perf_counter() - train_start) / 3600
# # #     print(f"\n  Best composite score: {saver.best_score:.2f}")
# # #     print(f"  Best: ADE={saver.best_ade:.1f}  "
# # #           f"12h={saver.best_12h:.0f}  24h={saver.best_24h:.0f}  "
# # #           f"48h={saver.best_48h:.0f}  72h={saver.best_72h:.0f}  "
# # #           f"ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}")
# # #     print(f"  Total time: {total_h:.2f}h")

# # #     best_r = {"ADE": saver.best_ade, "12h": saver.best_12h,
# # #               "24h": saver.best_24h, "48h": saver.best_48h,
# # #               "72h": saver.best_72h, "ATE_mean": saver.best_ate,
# # #               "CTE_mean": saver.best_cte}
# # #     beat = _beat_report(best_r)
# # #     if beat:
# # #         print(f"\n  🏆🏆🏆 {beat} 🏆🏆🏆")
# # #     else:
# # #         print(f"\n  Need: DPE<136.41, ATE<79.94, CTE<93.58, 72h<297")
# # #     print("=" * 72)

# # # # Trong train script, thêm function này:

# # # def diagnose_fno_encoder(model, val_loader, device, n_batches=5):
# # #     """
# # #     3 test để kiểm tra FNO3D có đang học geopotential height không.
    
# # #     Test 1: Ablation — zero out env, xem performance drop bao nhiêu
# # #     Test 2: Gradient flow — encoder có nhận gradient không
# # #     Test 3: Feature variance — bottleneck features có đa dạng không
# # #     """
# # #     raw_model = _get_raw_model(model)
# # #     model.eval()
    
# # #     results = {"with_env": [], "without_env": [], "feat_var": []}
    
# # #     with torch.no_grad():
# # #         for i, batch in enumerate(val_loader):
# # #             if i >= n_batches:
# # #                 break
# # #             bl = move(list(batch), device)
            
# # #             # ── Test 1: So sánh pred với/không có env ──────────────
# # #             pred_normal, _, _ = model.sample(bl, num_ensemble=10, 
# # #                                               ddim_steps=10,
# # #                                               importance_weight=False)
            
# # #             # Zero out env data
# # #             bl_no_env = list(bl)
# # #             if bl_no_env[13] is not None:
# # #                 env_zeroed = {k: torch.zeros_like(v) if torch.is_tensor(v) else v
# # #                               for k, v in bl_no_env[13].items()}
# # #                 bl_no_env[13] = env_zeroed
            
# # #             pred_no_env, _, _ = model.sample(bl_no_env, num_ensemble=10,
# # #                                               ddim_steps=10,
# # #                                               importance_weight=False)
            
# # #             T = min(pred_normal.shape[0], bl[1].shape[0])
# # #             gt = bl[1][:T]
            
# # #             d_normal  = haversine_km_torch(denorm_torch(pred_normal[:T]),
# # #                                             denorm_torch(gt)).mean().item()
# # #             d_no_env  = haversine_km_torch(denorm_torch(pred_no_env[:T]),
# # #                                             denorm_torch(gt)).mean().item()
            
# # #             results["with_env"].append(d_normal)
# # #             results["without_env"].append(d_no_env)
            
# # #             # ── Test 3: Feature variance của bottleneck ─────────────
# # #             obs_t    = bl[0]
# # #             img      = bl[11]
# # #             env_data = bl[13]
# # #             if img.dim() == 4:
# # #                 img = img.unsqueeze(2)
            
# # #             bot, _ = raw_model.net.spatial_enc.encode(img)
# # #             # bot: [B, 128, T, 4, 4]
# # #             feat_var = bot.var(dim=[0,2,3,4]).mean().item()
# # #             results["feat_var"].append(feat_var)
    
# # #     avg_with    = sum(results["with_env"]) / len(results["with_env"])
# # #     avg_without = sum(results["without_env"]) / len(results["without_env"])
# # #     avg_var     = sum(results["feat_var"]) / len(results["feat_var"])
    
# # #     improvement = (avg_without - avg_with) / avg_without * 100
    
# # #     print(f"\n{'='*55}")
# # #     print(f"  FNO3D ENCODER DIAGNOSTIC")
# # #     print(f"{'='*55}")
# # #     print(f"  ADE with env    : {avg_with:.1f} km")
# # #     print(f"  ADE without env : {avg_without:.1f} km")
# # #     print(f"  Env contribution: {improvement:+.1f}%  ", end="")
    
# # #     if improvement > 5:
# # #         print(f"✅ Encoder đang học được (env giúp {improvement:.0f}%)")
# # #     elif improvement > 0:
# # #         print(f"⚠️  Encoder học rất ít ({improvement:.1f}%) — FNO chưa converge")
# # #     else:
# # #         print(f"❌ Encoder KHÔNG đóng góp — model bỏ qua env hoàn toàn!")
# # #         print(f"     → Gợi ý: kiểm tra gradient flow vào FNO layers")
    
# # #     print(f"\n  Bottleneck feature variance: {avg_var:.4f}")
# # #     if avg_var < 0.01:
# # #         print(f"  ❌ Variance quá thấp — FNO output gần như constant!")
# # #         print(f"     → FNO không học được gì từ geopotential field")
# # #     elif avg_var < 0.1:
# # #         print(f"  ⚠️  Variance thấp — FNO đang học nhưng chưa nhiều")
# # #     else:
# # #         print(f"  ✅ Variance ổn — FNO đang extract đa dạng features")
    
# # #     print(f"{'='*55}\n")
# # #     return results

# # # if __name__ == "__main__":
# # #     args = get_args()
    
# # #     np.random.seed(42); torch.manual_seed(42)
# # #     if torch.cuda.is_available():
# # #         torch.cuda.manual_seed_all(42)
# # #     main(args)

# # """
# # scripts/train_flowmatching.py — v44_speed_fix
# # ═══════════════════════════════════════════════════════════════════════════════
# # THAY ĐỔI so với v41:

# #   1. Log print line thêm aux_fno key (FNO auxiliary loss)
# #   2. EMA decay default: 0.992 → 0.999 (ST-Trans style)
# #   3. Diagnose FNO encoder chạy epoch 5, 10, 15 (thay vì chỉ epoch 5)
# #   4. Output dir và metrics csv đổi thành v44
# #   5. Tất cả code khác GIỐNG HỆT v41
# # """
# # from __future__ import annotations

# # import sys
# # import os
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # import argparse
# # import time
# # import random

# # import numpy as np
# # import torch
# # import torch.optim as optim
# # from torch.amp import autocast, GradScaler
# # from torch.utils.data import DataLoader, Subset

# # from Model.data.loader_training import data_loader
# # from Model.flow_matching_model import TCFlowMatching
# # from Model.utils import get_cosine_schedule_with_warmup
# # from utils.metrics import (
# #     StepErrorAccumulator,
# #     save_metrics_csv,
# #     haversine_km_torch,
# #     denorm_torch,
# #     HORIZON_STEPS,
# #     DatasetMetrics,
# # )

# # try:
# #     from utils.metrics import haversine_and_atecte_torch
# #     HAS_ATECTE = True
# # except ImportError:
# #     HAS_ATECTE = False
# #     print("  ⚠  haversine_and_atecte_torch not found — ATE/CTE disabled")


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Composite Score v2
# # # ══════════════════════════════════════════════════════════════════════════════

# # def _composite_score(result: dict) -> float:
# #     ade = result.get("ADE", float("inf"))
# #     h12 = result.get("12h", float("inf"))
# #     h24 = result.get("24h", float("inf"))
# #     h48 = result.get("48h", float("inf"))
# #     h72 = result.get("72h", float("inf"))
# #     ate = result.get("ATE_mean", float("inf"))
# #     cte = result.get("CTE_mean", float("inf"))
# #     if not np.isfinite(ate): ate = ade * 0.46
# #     if not np.isfinite(cte): cte = ade * 0.53
# #     return 100.0 * (
# #         0.05 * (ade / 136.0)
# #         + 0.05 * (h12 / 50.0)
# #         + 0.10 * (h24 / 100.0)
# #         + 0.15 * (h48 / 200.0)
# #         + 0.35 * (h72 / 300.0)
# #         + 0.15 * (ate / 80.0)
# #         + 0.15 * (cte / 94.0)
# #     )


# # def _beat_report(r: dict) -> str:
# #     parts = []
# #     if np.isfinite(r.get("ADE", float("inf"))) and r["ADE"] < 136.41:
# #         parts.append(f"DPE✅{r['ADE']:.1f}")
# #     if np.isfinite(r.get("ATE_mean", float("inf"))) and r["ATE_mean"] < 79.94:
# #         parts.append(f"ATE✅{r['ATE_mean']:.1f}")
# #     if np.isfinite(r.get("CTE_mean", float("inf"))) and r["CTE_mean"] < 93.58:
# #         parts.append(f"CTE✅{r['CTE_mean']:.1f}")
# #     if np.isfinite(r.get("72h", float("inf"))) and r["72h"] < 297.0:
# #         parts.append(f"72h✅{r['72h']:.1f}")
# #     return "🏆 BEAT: " + " ".join(parts) if parts else ""


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


# # def make_val_subset_loader(val_dataset, subset_size, batch_size,
# #                             collate_fn, num_workers):
# #     n   = len(val_dataset)
# #     rng = random.Random(42)
# #     idx = rng.sample(range(n), min(subset_size, n))
# #     return DataLoader(Subset(val_dataset, idx),
# #                       batch_size=batch_size, shuffle=False,
# #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # def _get_ema_obj(model):
# #     if hasattr(model, '_ema') and model._ema is not None:
# #         return model._ema
# #     if hasattr(model, '_orig_mod'):
# #         orig = model._orig_mod
# #         if hasattr(orig, '_ema') and orig._ema is not None:
# #             return orig._ema
# #     return None


# # def _call_ema_update(model):
# #     if hasattr(model, '_orig_mod'):
# #         orig = model._orig_mod
# #         if hasattr(orig, 'ema_update'):
# #             orig.ema_update(); return
# #     if hasattr(model, 'ema_update'):
# #         model.ema_update()


# # def _get_raw_model(model):
# #     return model._orig_mod if hasattr(model, '_orig_mod') else model


# # def _save_checkpoint(path, epoch, model, optimizer, scheduler,
# #                      saver, avg_t, avg_vl, metrics=None):
# #     m = _get_raw_model(model)
# #     ema_obj = _get_ema_obj(model)
# #     ema_sd  = None
# #     if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# #         try:
# #             ema_sd = {k: v.cpu().clone() for k, v in ema_obj.shadow.items()}
# #         except Exception:
# #             ema_sd = None
# #     payload = {
# #         "epoch": epoch, "model_state_dict": m.state_dict(),
# #         "optimizer_state": optimizer.state_dict(),
# #         "scheduler_state": scheduler.state_dict(),
# #         "ema_shadow": ema_sd,
# #         "best_score": saver.best_score, "best_ade": saver.best_ade,
# #         "best_72h": saver.best_72h, "best_48h": saver.best_48h,
# #         "best_24h": saver.best_24h, "best_12h": saver.best_12h,
# #         "best_ate": getattr(saver, 'best_ate', float("inf")),
# #         "best_cte": getattr(saver, 'best_cte', float("inf")),
# #         "train_loss": avg_t, "val_loss": avg_vl,
# #     }
# #     if metrics: payload.update(metrics)
# #     torch.save(payload, path)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  SWA Manager
# # # ══════════════════════════════════════════════════════════════════════════════

# # class SWAManager:
# #     def __init__(self, model, start_epoch=60):
# #         self.start_epoch = start_epoch
# #         self.n_averaged  = 0
# #         self.avg_state   = None

# #     def update(self, model, epoch):
# #         if epoch < self.start_epoch: return
# #         m  = _get_raw_model(model)
# #         sd = {k: v.detach().clone() for k, v in m.state_dict().items()
# #               if v.dtype.is_floating_point}
# #         if self.avg_state is None:
# #             self.avg_state = sd; self.n_averaged = 1
# #         else:
# #             n = self.n_averaged
# #             for k in self.avg_state:
# #                 if k in sd:
# #                     self.avg_state[k].mul_(n / (n + 1)).add_(sd[k], alpha=1.0 / (n + 1))
# #             self.n_averaged += 1

# #     def apply_to(self, model):
# #         if self.avg_state is None: return None
# #         m  = _get_raw_model(model)
# #         sd = m.state_dict()
# #         backup = {k: sd[k].detach().clone() for k in self.avg_state if k in sd}
# #         for k, v in self.avg_state.items():
# #             if k in sd: sd[k].copy_(v)
# #         return backup

# #     def restore(self, model, backup):
# #         if backup is None: return
# #         m  = _get_raw_model(model)
# #         sd = m.state_dict()
# #         for k, v in backup.items():
# #             if k in sd: sd[k].copy_(v)


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Evaluation
# # # ══════════════════════════════════════════════════════════════════════════════

# # def _eval_batch_atecte(pred_norm, gt_norm):
# #     pred_d = denorm_torch(pred_norm)
# #     gt_d   = denorm_torch(gt_norm)
# #     if HAS_ATECTE:
# #         dist, ate, cte = haversine_and_atecte_torch(pred_d, gt_d, unit_01deg=True)
# #         return dist, ate, cte
# #     else:
# #         dist = haversine_km_torch(pred_d, gt_d)
# #         return dist, None, None


# # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=15):
# #     model.eval()
# #     acc = StepErrorAccumulator(pred_len)
# #     t0  = time.perf_counter()
# #     n   = 0
# #     spread_per_step = []
# #     with torch.no_grad():
# #         for batch in loader:
# #             bl = move(list(batch), device)
# #             pred, _, all_trajs = model.sample(
# #                 bl, num_ensemble=max(fast_ensemble, 20), ddim_steps=ode_steps,
# #                 importance_weight=True)
# #             T_active  = pred.shape[0]
# #             gt_sliced = bl[1][:T_active]
# #             dist, ate, cte = _eval_batch_atecte(pred, gt_sliced)
# #             acc.update(dist, ate_km=ate, cte_km=cte)
# #             step_spreads = []
# #             for t in range(all_trajs.shape[1]):
# #                 step_data = all_trajs[:, t, :, :]
# #                 spread = ((step_data[:, :, 0].std(0)**2
# #                            + step_data[:, :, 1].std(0)**2).sqrt() * 500.0).mean().item()
# #                 step_spreads.append(spread)
# #             spread_per_step.append(step_spreads)
# #             n += 1
# #     r = acc.compute()
# #     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# #     if spread_per_step:
# #         r["spread_72h_km"] = float(np.array(spread_per_step)[:, -1].mean())
# #     return r


# # def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
# #                            fast_ensemble, metrics_csv, epoch,
# #                            use_ema=False, ema_obj=None):
# #     backup = None
# #     if use_ema and ema_obj is not None:
# #         try:
# #             backup = ema_obj.apply_to(model)
# #         except Exception as e:
# #             print(f"  ⚠  EMA apply_to failed: {e}")
# #             backup = None; use_ema = False

# #     model.eval()
# #     acc = StepErrorAccumulator(pred_len)
# #     t0  = time.perf_counter()
# #     n   = 0
# #     with torch.no_grad():
# #         for batch in val_loader:
# #             bl = move(list(batch), device)
# #             pred, _, _ = model.sample(
# #                 bl, num_ensemble=max(fast_ensemble, 20),
# #                 ddim_steps=max(ode_steps, 20), importance_weight=True)
# #             T_pred = pred.shape[0]
# #             gt     = bl[1][:T_pred]
# #             dist, ate, cte = _eval_batch_atecte(pred, gt)
# #             acc.update(dist, ate_km=ate, cte_km=cte)
# #             n += 1

# #     r       = acc.compute()
# #     elapsed = time.perf_counter() - t0
# #     score   = _composite_score(r)
# #     tag     = "EMA" if use_ema else "RAW"

# #     ade_v = r.get("ADE",      float("nan"))
# #     fde_v = r.get("FDE",      float("nan"))
# #     h12_v = r.get("12h",      float("nan"))
# #     h24_v = r.get("24h",      float("nan"))
# #     h48_v = r.get("48h",      float("nan"))
# #     h72_v = r.get("72h",      float("nan"))
# #     ate_v = r.get("ATE_mean", float("nan"))
# #     cte_v = r.get("CTE_mean", float("nan"))

# #     def ind(v, tgt):
# #         if not np.isfinite(v): return ""
# #         return "✅" if v < tgt else "❌"

# #     print(f"\n{'='*70}")
# #     print(f"  [FULL VAL ({tag})  ep={epoch}  {elapsed:.0f}s  {n} batches]")
# #     print(f"  ADE={ade_v:.1f}{ind(ade_v,136.41)}  FDE={fde_v:.1f} km")
# #     print(f"  12h={h12_v:.0f}{ind(h12_v,50)}  24h={h24_v:.0f}{ind(h24_v,100)}  "
# #           f"48h={h48_v:.0f}{ind(h48_v,200)}  72h={h72_v:.0f}{ind(h72_v,297)} km")
# #     if np.isfinite(ate_v):
# #         print(f"  ATE={ate_v:.1f}{ind(ate_v,79.94)}  CTE={cte_v:.1f}{ind(cte_v,93.58)}"
# #               f"  [ST-Trans: ATE=79.94 CTE=93.58]")
# #     beat = _beat_report(r)
# #     if beat: print(f"  {beat}")
# #     print(f"  Score = {score:.2f}  (< 100 = beat ST-Trans all metrics)")
# #     print(f"{'='*70}\n")

# #     from datetime import datetime
# #     dm = DatasetMetrics(
# #         ade=ade_v if np.isfinite(ade_v) else 0.0,
# #         fde=fde_v if np.isfinite(fde_v) else 0.0,
# #         ugde_12h=h12_v if np.isfinite(h12_v) else 0.0,
# #         ugde_24h=h24_v if np.isfinite(h24_v) else 0.0,
# #         ugde_48h=h48_v if np.isfinite(h48_v) else 0.0,
# #         ugde_72h=h72_v if np.isfinite(h72_v) else 0.0,
# #         ate_abs_mean=ate_v if np.isfinite(ate_v) else 0.0,
# #         cte_abs_mean=cte_v if np.isfinite(cte_v) else 0.0,
# #         n_total=r.get("n_samples", 0),
# #         timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
# #     )
# #     save_metrics_csv(dm, metrics_csv, tag=f"val_{tag}_ep{epoch:03d}")

# #     if backup is not None:
# #         try:
# #             ema_obj.restore(model, backup)
# #         except Exception as e:
# #             print(f"  ⚠  EMA restore failed: {e}")
# #     return r


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  BestModelSaver
# # # ══════════════════════════════════════════════════════════════════════════════

# # class HorizonAwareBestSaver:
# #     def __init__(self, patience=30, tol=1.5):
# #         self.patience   = patience
# #         self.tol        = tol
# #         self.counter    = 0
# #         self.early_stop = False
# #         self.best_score = float("inf")
# #         self.best_ade   = float("inf")
# #         self.best_12h   = float("inf")
# #         self.best_24h   = float("inf")
# #         self.best_48h   = float("inf")
# #         self.best_72h   = float("inf")
# #         self.best_ate   = float("inf")
# #         self.best_cte   = float("inf")

# #     def update(self, r, model, out_dir, epoch, optimizer, scheduler,
# #                tl, vl, saver_ref, min_epochs=30):
# #         ade   = r.get("ADE", float("inf"))
# #         h12   = r.get("12h", float("inf"))
# #         h24   = r.get("24h", float("inf"))
# #         h48   = r.get("48h", float("inf"))
# #         h72   = r.get("72h", float("inf"))
# #         ate   = r.get("ATE_mean", ade * 0.46 if np.isfinite(ade) else float("inf"))
# #         cte   = r.get("CTE_mean", ade * 0.53 if np.isfinite(ade) else float("inf"))
# #         score = _composite_score(r)
# #         improved_any = False

# #         for metric_val, best_attr, fname, extra in [
# #             (ade, 'best_ade', 'best_ade.pth',  {"ade": ade}),
# #             (h72, 'best_72h', 'best_72h.pth',  {"h72": h72}),
# #             (ate, 'best_ate', 'best_ate.pth',  {"ate": ate, "cte": cte}),
# #             (cte, 'best_cte', 'best_cte.pth',  {"ate": ate, "cte": cte}),
# #         ]:
# #             if metric_val < getattr(self, best_attr):
# #                 setattr(self, best_attr, metric_val)
# #                 improved_any = True
# #                 _save_checkpoint(os.path.join(out_dir, fname),
# #                                   epoch, model, optimizer, scheduler,
# #                                   saver_ref, tl, vl, extra)

# #         if h48 < self.best_48h: self.best_48h = h48; improved_any = True
# #         if h24 < self.best_24h: self.best_24h = h24; improved_any = True
# #         if h12 < self.best_12h: self.best_12h = h12; improved_any = True

# #         if score < self.best_score - self.tol:
# #             self.best_score = score
# #             self.counter    = 0
# #             _save_checkpoint(
# #                 os.path.join(out_dir, "best_model.pth"),
# #                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
# #                 {"ade": ade, "h12": h12, "h24": h24, "h48": h48, "h72": h72,
# #                  "ate": ate, "cte": cte, "composite_score": score})
# #             print(f"  ✅ Best COMPOSITE={score:.2f}  ADE={ade:.1f}  "
# #                   f"12h={h12:.0f}  24h={h24:.0f}  48h={h48:.0f}  "
# #                   f"72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}  (ep {epoch})")
# #         else:
# #             if not improved_any:
# #                 self.counter += 1
# #             print(f"  No improvement {self.counter}/{self.patience}"
# #                   f"  (best={self.best_score:.2f} cur={score:.2f})"
# #                   f"  72h={h72:.0f}↓{self.best_72h:.0f}"
# #                   f"  ATE={ate:.1f}↓{self.best_ate:.1f}")

# #         if epoch >= min_epochs and self.counter >= self.patience:
# #             self.early_stop = True


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Env diagnostic — chạy nhiều lần hơn để track FNO improvement
# # # ══════════════════════════════════════════════════════════════════════════════

# # def _check_env(bl, train_dataset):
# #     try:    env_dir = train_dataset.env_path
# #     except AttributeError:
# #         try:    env_dir = train_dataset.dataset.env_path
# #         except: env_dir = "UNKNOWN"
# #     print(f"  Env path: {env_dir}")
# #     env_data = bl[13]
# #     if env_data is None:
# #         print("  ⚠️  env_data is None"); return
# #     for key in ("gph500_mean", "gph500_center", "u500_mean", "v500_mean"):
# #         if key not in env_data:
# #             print(f"  ⚠️  {key} MISSING"); continue
# #         v    = env_data[key]
# #         zero = 100.0 * (v == 0).sum().item() / max(v.numel(), 1)
# #         print(f"  {'✅' if zero < 80 else '⚠️'}  {key}: "
# #               f"mean={v.mean().item():.4f} std={v.std().item():.4f} zero={zero:.1f}%")


# # def diagnose_fno_encoder(model, val_loader, device, n_batches=5):
# #     """
# #     3 tests để kiểm tra FNO3D có đang học geopotential height không.

# #     Test 1: Ablation — zero out env, xem performance drop bao nhiêu
# #     Test 2: Gradient flow — encoder có nhận gradient không
# #     Test 3: Feature variance — bottleneck features có đa dạng không

# #     Kết quả cần quan sát theo epoch:
# #       ep5:  Env contribution âm (−13%) → FNO bị bypass
# #       ep10: Phải dương (+3~5%) nếu aux loss hoạt động
# #       ep20: Phải dương (+5~10%) → FNO đang học
# #     """
# #     raw_model = _get_raw_model(model)
# #     model.eval()

# #     results = {"with_env": [], "without_env": [], "feat_var": []}

# #     with torch.no_grad():
# #         for i, batch in enumerate(val_loader):
# #             if i >= n_batches:
# #                 break
# #             bl = move(list(batch), device)

# #             # Test 1: So sánh pred với/không có env
# #             pred_normal, _, _ = model.sample(bl, num_ensemble=10,
# #                                               ddim_steps=10,
# #                                               importance_weight=False)
# #             bl_no_env = list(bl)
# #             if bl_no_env[13] is not None:
# #                 env_zeroed = {k: torch.zeros_like(v) if torch.is_tensor(v) else v
# #                               for k, v in bl_no_env[13].items()}
# #                 bl_no_env[13] = env_zeroed

# #             pred_no_env, _, _ = model.sample(bl_no_env, num_ensemble=10,
# #                                               ddim_steps=10,
# #                                               importance_weight=False)

# #             T  = min(pred_normal.shape[0], bl[1].shape[0])
# #             gt = bl[1][:T]

# #             d_normal = haversine_km_torch(denorm_torch(pred_normal[:T]),
# #                                           denorm_torch(gt)).mean().item()
# #             d_no_env = haversine_km_torch(denorm_torch(pred_no_env[:T]),
# #                                           denorm_torch(gt)).mean().item()

# #             results["with_env"].append(d_normal)
# #             results["without_env"].append(d_no_env)

# #             # Test 3: Feature variance của bottleneck
# #             img = bl[11]
# #             if img.dim() == 4:
# #                 img = img.unsqueeze(2)
# #             bot, _ = raw_model.net.spatial_enc.encode(img)
# #             feat_var = bot.var(dim=[0, 2, 3, 4]).mean().item()
# #             results["feat_var"].append(feat_var)

# #     avg_with    = sum(results["with_env"]) / len(results["with_env"])
# #     avg_without = sum(results["without_env"]) / len(results["without_env"])
# #     avg_var     = sum(results["feat_var"]) / len(results["feat_var"])
# #     improvement = (avg_without - avg_with) / avg_without * 100

# #     print(f"\n{'='*55}")
# #     print(f"  FNO3D ENCODER DIAGNOSTIC")
# #     print(f"{'='*55}")
# #     print(f"  ADE with env    : {avg_with:.1f} km")
# #     print(f"  ADE without env : {avg_without:.1f} km")
# #     print(f"  Env contribution: {improvement:+.1f}%  ", end="")

# #     if improvement > 5:
# #         print(f"✅ FNO đang học ({improvement:.0f}%)")
# #     elif improvement > 0:
# #         print(f"⚠️  FNO học rất ít ({improvement:.1f}%) — chưa converge")
# #     else:
# #         print(f"❌ FNO bị bypass! Kiểm tra aux_fno loss trong log")
# #         print(f"     → aux_fno phải > 0.001 để gradient chạy vào FNO")

# #     print(f"\n  Bottleneck variance: {avg_var:.4f}")
# #     if avg_var < 0.01:
# #         print(f"  ❌ Variance quá thấp — FNO output gần như constant!")
# #         print(f"     → Cần tăng w_aux hoặc lr của FNO layers")
# #     elif avg_var < 0.1:
# #         print(f"  ⚠️  Variance thấp — FNO đang học nhưng chưa nhiều")
# #     else:
# #         print(f"  ✅ Variance ổn — FNO đang extract đa dạng features")
# #     print(f"{'='*55}\n")
# #     return results


# # # ══════════════════════════════════════════════════════════════════════════════
# # #  Args
# # # ══════════════════════════════════════════════════════════════════════════════

# # def get_args():
# #     p = argparse.ArgumentParser(
# #         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# #     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
# #     p.add_argument("--obs_len",         default=8,          type=int)
# #     p.add_argument("--pred_len",        default=12,         type=int)
# #     p.add_argument("--batch_size",      default=32,         type=int)
# #     p.add_argument("--num_epochs",      default=100,        type=int)
# #     p.add_argument("--g_learning_rate", default=3e-4,       type=float)
# #     p.add_argument("--weight_decay",    default=5e-4,       type=float)
# #     p.add_argument("--warmup_epochs",   default=5,          type=int)
# #     p.add_argument("--grad_clip",       default=1.0,        type=float)
# #     p.add_argument("--patience",        default=30,         type=int)
# #     p.add_argument("--min_epochs",      default=30,         type=int)
# #     p.add_argument("--n_train_ens",     default=4,          type=int)
# #     p.add_argument("--use_amp",         action="store_true")
# #     p.add_argument("--num_workers",     default=2,          type=int)
# #     p.add_argument("--sigma_min",            default=0.02,  type=float)
# #     p.add_argument("--ctx_noise_scale",      default=0.01,  type=float)
# #     p.add_argument("--initial_sample_sigma", default=0.03,  type=float)
# #     p.add_argument("--ode_steps_train", default=10,  type=int)
# #     p.add_argument("--ode_steps_val",   default=10,  type=int)
# #     p.add_argument("--ode_steps_test",  default=10,  type=int)
# #     p.add_argument("--val_ensemble",    default=20,  type=int)
# #     p.add_argument("--fast_ensemble",   default=10,  type=int)
# #     p.add_argument("--val_freq",        default=5,   type=int)
# #     p.add_argument("--val_ade_freq",    default=5,   type=int)
# #     p.add_argument("--val_subset_size", default=500, type=int)
# #     p.add_argument("--use_ema",         action="store_true", default=True)
# #     # FIX: EMA decay 0.992 → 0.999 (ST-Trans style, model lớn hơn 394k params)
# #     p.add_argument("--ema_decay",       default=0.999, type=float)
# #     p.add_argument("--swa_start_epoch", default=60,   type=int)
# #     p.add_argument("--teacher_forcing", action="store_true", default=True)
# #     p.add_argument("--output_dir",  default="runs/v44",        type=str)
# #     p.add_argument("--metrics_csv", default="metrics_v44.csv", type=str)
# #     p.add_argument("--predict_csv", default="predictions.csv", type=str)
# #     p.add_argument("--gpu_num",     default="0", type=str)
# #     p.add_argument("--delim",       default=" ")
# #     p.add_argument("--skip",        default=1,   type=int)
# #     p.add_argument("--min_ped",     default=1,   type=int)
# #     p.add_argument("--threshold",   default=0.002, type=float)
# #     p.add_argument("--other_modal", default="gph")
# #     p.add_argument("--test_year",   default=None, type=int)
# #     p.add_argument("--resume",       default=None, type=str)
# #     p.add_argument("--resume_epoch", default=None, type=int)
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

# #     print("=" * 72)
# #     print("  TC-FlowMatching v44  |  SPEED-BIAS FIX EDITION")
# #     print("  TARGETS: ATE<79.94 | CTE<93.58 | 72h<297")
# #     print("  FIXES: speed_loss + physics_scale + no_speed_aug + FM_balance")
# #     print("  MONITOR: spd log phải > 0 (hiện 0.000 = bug vẫn còn)")
# #     print(f"  EMA decay: {args.ema_decay}")
# #     print("=" * 72)

# #     train_dataset, train_loader = data_loader(
# #         args, {"root": args.dataset_root, "type": "train"}, test=False)
# #     val_dataset, val_loader = data_loader(
# #         args, {"root": args.dataset_root, "type": "val"}, test=True)

# #     from Model.data.trajectoriesWithMe_unet_training import seq_collate
# #     val_subset_loader = make_val_subset_loader(
# #         val_dataset, args.val_subset_size, args.batch_size,
# #         seq_collate, args.num_workers)

# #     test_loader = None
# #     try:
# #         _, test_loader = data_loader(
# #             args, {"root": args.dataset_root, "type": "test"},
# #             test=True, test_year=None)
# #     except Exception as e:
# #         print(f"  Warning: test loader: {e}")

# #     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
# #     print(f"  val   : {len(val_dataset)} seq")

# #     model = TCFlowMatching(
# #         pred_len=args.pred_len, obs_len=args.obs_len,
# #         sigma_min=args.sigma_min, n_train_ens=args.n_train_ens,
# #         ctx_noise_scale=args.ctx_noise_scale,
# #         initial_sample_sigma=args.initial_sample_sigma,
# #         teacher_forcing=args.teacher_forcing,
# #         use_ema=args.use_ema, ema_decay=args.ema_decay,
# #     ).to(device)

# #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #     print(f"  params  : {n_params:,}")
# #     model.init_ema()
# #     print(f"  EMA     : {'ON' if model._ema is not None else 'OFF'}"
# #           f"  (decay={args.ema_decay})")

# #     optimizer = optim.AdamW(model.parameters(),
# #                              lr=args.g_learning_rate,
# #                              weight_decay=args.weight_decay)
# #     saver  = HorizonAwareBestSaver(patience=args.patience, tol=1.5)
# #     scaler = GradScaler("cuda", enabled=args.use_amp)
# #     swa    = SWAManager(model, start_epoch=args.swa_start_epoch)

# #     steps_per_epoch = len(train_loader)
# #     total_steps     = steps_per_epoch * args.num_epochs
# #     warmup          = steps_per_epoch * args.warmup_epochs
# #     scheduler = get_cosine_schedule_with_warmup(
# #         optimizer, warmup, total_steps, min_lr=1e-6)

# #     # ── Resume ───────────────────────────────────────────────────────────────
# #     has_scheduler_state   = False
# #     _ckpt_scheduler_state = None
# #     start_epoch = 0

# #     if args.resume is not None and os.path.exists(args.resume):
# #         print(f"  Loading checkpoint: {args.resume}")
# #         ckpt = torch.load(args.resume, map_location=device)
# #         m = _get_raw_model(model)
# #         missing, unexpected = m.load_state_dict(ckpt["model_state_dict"], strict=False)
# #         if missing:
# #             print(f"  ⚠  Missing keys ({len(missing)}): {missing[:5]}")
# #             print(f"     → Đây là bình thường nếu thêm aux_steering_head mới")
# #         if unexpected:
# #             print(f"  ⚠  Unexpected keys ({len(unexpected)}): {unexpected[:3]}")
# #         if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
# #             ema_obj = _get_ema_obj(model)
# #             if ema_obj is not None and hasattr(ema_obj, 'shadow'):
# #                 loaded = 0
# #                 for k, v in ckpt["ema_shadow"].items():
# #                     if k in ema_obj.shadow:
# #                         ema_obj.shadow[k].copy_(v.to(device))
# #                         loaded += 1
# #                 print(f"  EMA shadow loaded: {loaded} keys")
# #         if "optimizer_state" in ckpt:
# #             try:
# #                 optimizer.load_state_dict(ckpt["optimizer_state"])
# #                 for state in optimizer.state.values():
# #                     for k, v in state.items():
# #                         if torch.is_tensor(v): state[k] = v.to(device)
# #             except Exception as e:
# #                 print(f"  ⚠  Optimizer restore failed (OK if new params added): {e}")
# #         has_scheduler_state   = "scheduler_state" in ckpt
# #         _ckpt_scheduler_state = ckpt.get("scheduler_state", None)
# #         if "best_score" in ckpt:
# #             saver.best_score = ckpt["best_score"]
# #             saver.best_ade   = ckpt.get("best_ade",  float("inf"))
# #             saver.best_72h   = ckpt.get("best_72h",  float("inf"))
# #             saver.best_48h   = ckpt.get("best_48h",  float("inf"))
# #             saver.best_24h   = ckpt.get("best_24h",  float("inf"))
# #             saver.best_12h   = ckpt.get("best_12h",  float("inf"))
# #             saver.best_ate   = ckpt.get("best_ate",  float("inf"))
# #             saver.best_cte   = ckpt.get("best_cte",  float("inf"))
# #         start_epoch = (args.resume_epoch if args.resume_epoch is not None
# #                        else ckpt.get("epoch", 0) + 1)
# #         print(f"  → Resuming from epoch {start_epoch}")
# #     elif args.resume is not None:
# #         print(f"  ⚠  Checkpoint không tìm thấy: {args.resume}")

# #     try:
# #         model = torch.compile(model, mode="reduce-overhead")
# #         print("  torch.compile: enabled")
# #     except Exception:
# #         pass

# #     if has_scheduler_state and _ckpt_scheduler_state is not None:
# #         try:
# #             scheduler.load_state_dict(_ckpt_scheduler_state)
# #         except Exception as e:
# #             print(f"  ⚠  Scheduler restore failed: {e}")
# #             for _ in range(start_epoch * steps_per_epoch): scheduler.step()
# #     elif start_epoch > 0:
# #         for _ in range(start_epoch * steps_per_epoch): scheduler.step()

# #     print("=" * 72)
# #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch, start={start_epoch})")
# #     print(f"  KEY: Monitor 'spd' value — phải > 0 sau fix!")
# #     print(f"  KEY: Monitor 'aux' value — FNO learning signal")
# #     print("=" * 72)

# #     epoch_times = []
# #     train_start = time.perf_counter()

# #     for epoch in range(start_epoch, args.num_epochs):
# #         model.train()
# #         sum_loss = 0.0
# #         t0 = time.perf_counter()

# #         for i, batch in enumerate(train_loader):
# #             bl = move(list(batch), device)

# #             if epoch == start_epoch and i == 0:
# #                 _check_env(bl, train_dataset)

# #             with autocast(device_type="cuda", enabled=args.use_amp):
# #                 bd = model.get_loss_breakdown(bl, epoch=epoch)

# #             optimizer.zero_grad()
# #             scaler.scale(bd["total"]).backward()
# #             scaler.unscale_(optimizer)
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

# #             # FIX v45: Chỉ step scheduler nếu optimizer thực sự đã step (tránh AMP warning)
# #             scale_before = scaler.get_scale()
# #             scaler.step(optimizer)
# #             scaler.update()
# #             scale_after = scaler.get_scale()

# #             if scale_after >= scale_before:
# #                 scheduler.step()
# #             _call_ema_update(model)

# #             sum_loss += bd["total"].item()

# #             # if i % 20 == 0:
# #             #     lr = optimizer.param_groups[0]["lr"]
# #             #     # ── LOG LINE v44: thêm aux_fno (FNO auxiliary loss)
# #             #     # KEY MONITOR: spd phải > 0 (nếu vẫn 0.000 → loss bug)
# #             #     # KEY MONITOR: aux phải > 0 (nếu 0.000 → FNO không học)
# #             #     print(
# #             #         f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# #             #         f"  tot={bd['total'].item():.3f}"
# #             #         f"  fm={bd.get('fm_mse',     0):.4f}"
# #             #         f"  hor={bd.get('mse_hav',   0):.1f}"
# #             #         f"  end={bd.get('endpoint',  0):.1f}"
# #             #         f"  spd={bd.get('speed_acc', 0):.3f}"   # ← PHẢI > 0 sau fix
# #             #         f"  acc={bd.get('accel',     0):.3f}"
# #             #         f"  dcp={bd.get('decomp',    0):.4f}"
# #             #         f"  cns={bd.get('cons',      0):.3f}"
# #             #         f"  aux={bd.get('aux_fno',   0):.4f}"   # ← FNO learning signal
# #             #         f"  lr={lr:.2e}"
# #             #     )
# #             if i % 20 == 0:
# #                 lr = optimizer.param_groups[0]["lr"]
# #                 # v45 log: physical units (km, km/h) thay vì normalized values
# #                 print(
# #                     f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# #                     f"  tot={bd['total'].item():.3f}"
# #                     f"  fm={bd.get('fm_mse',     0):.4f}"
# #                     f"  hav={bd.get('hav_km',    0):.1f}km"      # raw km
# #                     f"  h72={bd.get('h72_km',    0):.1f}km"      # raw km @ 72h
# #                     f"  spd={bd.get('spd_kmh',   0):.2f}km/h"    # raw km/h error
# #                     f"  acc={bd.get('acc_kmh2',  0):.2f}"        # raw km/h²
# #                     f"  cns={bd.get('cons',      0):.3f}"
# #                     f"  aux={bd.get('aux_fno',   0):.4f}"        # FNO signal
# #                     f"  lr={lr:.2e}"
# #                 )

# #         # ── FNO Diagnostic — chạy epoch 5, 10, 15 để track improvement ──
# #         # v41: chỉ chạy epoch 5 (bên trong loop)
# #         # v44: chạy ngoài loop, epoch 5/10/15
# #         if epoch in (5, 10, 15, 20):
# #             print(f"\n  [Epoch {epoch}] Chạy FNO Diagnostic...")
# #             diagnose_fno_encoder(model, val_subset_loader, device)

# #         ep_s  = time.perf_counter() - t0
# #         epoch_times.append(ep_s)
# #         avg_t = sum_loss / len(train_loader)
# #         swa.update(model, epoch)

# #         model.eval()
# #         val_loss = 0.0
# #         with torch.no_grad():
# #             for batch in val_loader:
# #                 bl_v = move(list(batch), device)
# #                 with autocast(device_type="cuda", enabled=args.use_amp):
# #                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
# #         avg_vl = val_loss / len(val_loader)
# #         print(f"  Epoch {epoch:>3}  train={avg_t:.4f}  val={avg_vl:.4f}"
# #               f"  time={ep_s:.0f}s")

# #         # ── Fast eval mỗi epoch ──────────────────────────────────────────────
# #         m_fast = evaluate_fast(model, val_subset_loader, device,
# #                                 args.ode_steps_train, args.pred_len,
# #                                 args.fast_ensemble)
# #         h12 = m_fast.get("12h", float("nan"))
# #         h24 = m_fast.get("24h", float("nan"))
# #         h48 = m_fast.get("48h", float("nan"))
# #         h72 = m_fast.get("72h", float("nan"))
# #         ate = m_fast.get("ATE_mean", float("nan"))
# #         cte = m_fast.get("CTE_mean", float("nan"))
# #         fast_score = _composite_score(m_fast)

# #         def ind(v, tgt):
# #             if not np.isfinite(v): return "?"
# #             return "🎯" if v < tgt else "❌"

# #         print(
# #             f"  [FAST ep{epoch}]"
# #             f"  ADE={m_fast.get('ADE', float('nan')):.1f}"
# #             f"  12h={h12:.0f}{ind(h12,50)}"
# #             f"  24h={h24:.0f}{ind(h24,100)}"
# #             f"  48h={h48:.0f}{ind(h48,200)}"
# #             f"  72h={h72:.0f}{ind(h72,297)}"
# #             f"  ATE={ate:.1f}{ind(ate,79.94)}"
# #             f"  CTE={cte:.1f}{ind(cte,93.58)}"
# #             f"  score={fast_score:.2f}"
# #         )

# #         # ── Full val every val_freq epochs ───────────────────────────────────
# #         if epoch % args.val_freq == 0:
# #             ema_obj = _get_ema_obj(model)
# #             try:
# #                 r_raw = evaluate_full_val_ade(
# #                     model, val_loader, device,
# #                     ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# #                     fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# #                     epoch=epoch, use_ema=False)
# #                 r_use = r_raw
# #                 if ema_obj is not None and epoch >= 5:
# #                     r_ema = evaluate_full_val_ade(
# #                         model, val_loader, device,
# #                         ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# #                         fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
# #                         epoch=epoch, use_ema=True, ema_obj=ema_obj)
# #                     if _composite_score(r_ema) < _composite_score(r_raw):
# #                         r_use = r_ema
# #                 saver.update(r_use, model, args.output_dir, epoch,
# #                              optimizer, scheduler, avg_t, avg_vl,
# #                              saver_ref=saver, min_epochs=args.min_epochs)
# #             except Exception as e:
# #                 print(f"  ⚠  Full val failed: {e}")
# #                 import traceback; traceback.print_exc()

# #         if epoch % 10 == 0 or epoch == args.num_epochs - 1:
# #             _save_checkpoint(
# #                 os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"),
# #                 epoch, model, optimizer, scheduler, saver, avg_t, avg_vl,
# #                 {"ade": m_fast.get("ADE", float("nan")),
# #                  "h48": h48, "h72": h72, "ate": ate, "cte": cte})

# #         if epoch % 10 == 9:
# #             avg_ep    = sum(epoch_times) / len(epoch_times)
# #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# #             elapsed_h = (time.perf_counter() - train_start) / 3600
# #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")
# #             print(f"  📊 Best: score={saver.best_score:.2f}  "
# #                   f"72h={saver.best_72h:.0f}km  "
# #                   f"ATE={saver.best_ate:.1f}km  CTE={saver.best_cte:.1f}km")

# #         if saver.early_stop:
# #             print(f"  ⛔ Early stopping at epoch {epoch}")
# #             break

# #     # ── Final SWA eval ───────────────────────────────────────────────────────
# #     print(f"\n{'='*72}")
# #     swa_backup = swa.apply_to(model)
# #     if swa_backup is not None:
# #         print("  Evaluating SWA weights...")
# #         r_swa = evaluate_full_val_ade(
# #             model, val_loader, device,
# #             ode_steps=args.ode_steps_val, pred_len=args.pred_len,
# #             fast_ensemble=args.val_ensemble, metrics_csv=metrics_csv,
# #             epoch=9999, use_ema=False)
# #         if _composite_score(r_swa) < saver.best_score:
# #             _save_checkpoint(
# #                 os.path.join(args.output_dir, "best_swa.pth"),
# #                 9999, model, optimizer, scheduler, saver,
# #                 avg_t=0.0, avg_vl=0.0, metrics={"tag": "best_swa"})
# #             print("  ✅ SWA checkpoint saved")
# #         swa.restore(model, swa_backup)

# #     total_h = (time.perf_counter() - train_start) / 3600
# #     print(f"\n  Best composite score: {saver.best_score:.2f}")
# #     print(f"  Best: ADE={saver.best_ade:.1f}  "
# #           f"12h={saver.best_12h:.0f}  24h={saver.best_24h:.0f}  "
# #           f"48h={saver.best_48h:.0f}  72h={saver.best_72h:.0f}  "
# #           f"ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}")
# #     print(f"  Total time: {total_h:.2f}h")

# #     best_r = {"ADE": saver.best_ade, "12h": saver.best_12h,
# #               "24h": saver.best_24h, "48h": saver.best_48h,
# #               "72h": saver.best_72h, "ATE_mean": saver.best_ate,
# #               "CTE_mean": saver.best_cte}
# #     beat = _beat_report(best_r)
# #     if beat:
# #         print(f"\n  🏆🏆🏆 {beat} 🏆🏆🏆")
# #     else:
# #         print(f"\n  Need: DPE<136.41, ATE<79.94, CTE<93.58, 72h<297")
# #     print("=" * 72)


# # if __name__ == "__main__":
# #     args = get_args()
# #     np.random.seed(42); torch.manual_seed(42)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(42)
# #     main(args)

# """
# scripts/train_flowmatching_v46.py — ST-Trans loss + FM architecture
# ═══════════════════════════════════════════════════════════════════════════════

# KEY CHANGES vs v45:
#   1. Loss đơn giản: DPE + 0.05*MSE + 0.1*speed + 0.01*accel (như ST-Trans)
#   2. Sigma schedule: 0.15 → 0.03 (fast learning phase)
#   3. Log đơn giản: dpe, spd, acc, fm_mse
#   4. Giảm patience → tránh overtrain
#   5. Output dir v46

# MONITOR:
#   - dpe phải giảm dần: ep0~3.0, ep10~2.0, ep30~1.5
#   - 72h phải < 350 ở ep15, < 320 ở ep30, < 300 ở ep50
# """
# from __future__ import annotations

# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse, time, random
# import numpy as np
# import torch
# import torch.optim as optim
# from torch.amp import autocast, GradScaler
# from torch.utils.data import DataLoader, Subset

# from Model.data.loader_training import data_loader
# from Model.flow_matching_model import TCFlowMatching
# from Model.utils import get_cosine_schedule_with_warmup
# from utils.metrics import (
#     StepErrorAccumulator, save_metrics_csv,
#     haversine_km_torch, denorm_torch,
#     HORIZON_STEPS, DatasetMetrics,
# )

# try:
#     from utils.metrics import haversine_and_atecte_torch
#     HAS_ATECTE = True
# except ImportError:
#     HAS_ATECTE = False


# # ══════════════════════════════════════════════════════════════════════════════
# #  Score
# # ══════════════════════════════════════════════════════════════════════════════

# def _composite_score(result: dict) -> float:
#     ade = result.get("ADE", float("inf"))
#     h72 = result.get("72h", float("inf"))
#     ate = result.get("ATE_mean", float("inf"))
#     cte = result.get("CTE_mean", float("inf"))
#     if not np.isfinite(ate): ate = ade * 0.46
#     if not np.isfinite(cte): cte = ade * 0.53
#     h12 = result.get("12h", float("inf"))
#     h24 = result.get("24h", float("inf"))
#     h48 = result.get("48h", float("inf"))
#     return 100.0 * (
#         0.05 * (ade  / 136.0)
#         + 0.05 * (h12  / 50.0)
#         + 0.10 * (h24  / 100.0)
#         + 0.15 * (h48  / 200.0)
#         + 0.35 * (h72  / 300.0)
#         + 0.15 * (ate  / 80.0)
#         + 0.15 * (cte  / 94.0)
#     )


# def _beat_report(r: dict) -> str:
#     parts = []
#     if np.isfinite(r.get("ADE", float("inf"))) and r["ADE"] < 136.41:
#         parts.append(f"DPE✅{r['ADE']:.1f}")
#     if np.isfinite(r.get("ATE_mean", float("inf"))) and r["ATE_mean"] < 79.94:
#         parts.append(f"ATE✅{r['ATE_mean']:.1f}")
#     if np.isfinite(r.get("CTE_mean", float("inf"))) and r["CTE_mean"] < 93.58:
#         parts.append(f"CTE✅{r['CTE_mean']:.1f}")
#     if np.isfinite(r.get("72h", float("inf"))) and r["72h"] < 297.0:
#         parts.append(f"72h✅{r['72h']:.1f}")
#     return "🏆 BEAT: " + " ".join(parts) if parts else ""


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
#     idx = rng.sample(range(n), min(subset_size, n))
#     return DataLoader(Subset(val_dataset, idx),
#                       batch_size=batch_size, shuffle=False,
#                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# def _get_ema_obj(model):
#     if hasattr(model, '_ema') and model._ema is not None:
#         return model._ema
#     if hasattr(model, '_orig_mod'):
#         orig = model._orig_mod
#         if hasattr(orig, '_ema') and orig._ema is not None:
#             return orig._ema
#     return None


# def _call_ema_update(model):
#     if hasattr(model, '_orig_mod'):
#         orig = model._orig_mod
#         if hasattr(orig, 'ema_update'):
#             orig.ema_update(); return
#     if hasattr(model, 'ema_update'):
#         model.ema_update()


# def _get_raw_model(model):
#     return model._orig_mod if hasattr(model, '_orig_mod') else model


# def _save_checkpoint(path, epoch, model, optimizer, scheduler,
#                      saver, avg_t, avg_vl, metrics=None):
#     m = _get_raw_model(model)
#     ema_obj = _get_ema_obj(model)
#     ema_sd  = None
#     if ema_obj is not None and hasattr(ema_obj, 'shadow'):
#         try:
#             ema_sd = {k: v.cpu().clone() for k, v in ema_obj.shadow.items()}
#         except Exception:
#             ema_sd = None
#     payload = {
#         "epoch": epoch, "model_state_dict": m.state_dict(),
#         "optimizer_state": optimizer.state_dict(),
#         "scheduler_state": scheduler.state_dict(),
#         "ema_shadow": ema_sd,
#         "best_score": saver.best_score, "best_ade": saver.best_ade,
#         "best_72h": saver.best_72h, "best_48h": saver.best_48h,
#         "best_24h": saver.best_24h, "best_12h": saver.best_12h,
#         "best_ate": getattr(saver, 'best_ate', float("inf")),
#         "best_cte": getattr(saver, 'best_cte', float("inf")),
#         "train_loss": avg_t, "val_loss": avg_vl,
#     }
#     if metrics:
#         payload.update(metrics)
#     torch.save(payload, path)


# # ══════════════════════════════════════════════════════════════════════════════
# #  Evaluation
# # ══════════════════════════════════════════════════════════════════════════════

# def _eval_batch_atecte(pred_norm, gt_norm):
#     pred_d = denorm_torch(pred_norm)
#     gt_d   = denorm_torch(gt_norm)
#     if HAS_ATECTE:
#         dist, ate, cte = haversine_and_atecte_torch(pred_d, gt_d, unit_01deg=True)
#         return dist, ate, cte
#     else:
#         dist = haversine_km_torch(pred_d, gt_d)
#         return dist, None, None


# def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=15):
#     model.eval()
#     acc = StepErrorAccumulator(pred_len)
#     t0  = time.perf_counter()
#     n   = 0
#     with torch.no_grad():
#         for batch in loader:
#             bl = move(list(batch), device)
#             pred, _, all_trajs = model.sample(
#                 bl, num_ensemble=max(fast_ensemble, 20), ddim_steps=ode_steps,
#                 importance_weight=True)
#             T_active  = pred.shape[0]
#             gt_sliced = bl[1][:T_active]
#             dist, ate, cte = _eval_batch_atecte(pred, gt_sliced)
#             acc.update(dist, ate_km=ate, cte_km=cte)
#             n += 1
#     r = acc.compute()
#     r["ms_per_batch"] = (time.perf_counter() - t0) * 1e3 / max(n, 1)
#     return r


# def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
#                            fast_ensemble, metrics_csv, epoch,
#                            use_ema=False, ema_obj=None):
#     backup = None
#     if use_ema and ema_obj is not None:
#         try:
#             backup = ema_obj.apply_to(model)
#         except Exception as e:
#             print(f"  ⚠  EMA apply_to failed: {e}")
#             backup = None; use_ema = False

#     model.eval()
#     acc = StepErrorAccumulator(pred_len)
#     t0  = time.perf_counter()
#     n   = 0
#     with torch.no_grad():
#         for batch in val_loader:
#             bl = move(list(batch), device)
#             pred, _, _ = model.sample(
#                 bl, num_ensemble=max(fast_ensemble, 20),
#                 ddim_steps=max(ode_steps, 20), importance_weight=True)
#             T_pred = pred.shape[0]
#             gt     = bl[1][:T_pred]
#             dist, ate, cte = _eval_batch_atecte(pred, gt)
#             acc.update(dist, ate_km=ate, cte_km=cte)
#             n += 1

#     r       = acc.compute()
#     elapsed = time.perf_counter() - t0
#     score   = _composite_score(r)
#     tag     = "EMA" if use_ema else "RAW"

#     ade_v = r.get("ADE",      float("nan"))
#     h12_v = r.get("12h",      float("nan"))
#     h24_v = r.get("24h",      float("nan"))
#     h48_v = r.get("48h",      float("nan"))
#     h72_v = r.get("72h",      float("nan"))
#     ate_v = r.get("ATE_mean", float("nan"))
#     cte_v = r.get("CTE_mean", float("nan"))

#     def ind(v, tgt):
#         if not np.isfinite(v): return ""
#         return "✅" if v < tgt else "❌"

#     print(f"\n{'='*70}")
#     print(f"  [FULL VAL ({tag})  ep={epoch}  {elapsed:.0f}s  {n} batches]")
#     print(f"  ADE={ade_v:.1f}{ind(ade_v,136.41)} km")
#     print(f"  12h={h12_v:.0f}{ind(h12_v,50)}  24h={h24_v:.0f}{ind(h24_v,100)}  "
#           f"48h={h48_v:.0f}{ind(h48_v,200)}  72h={h72_v:.0f}{ind(h72_v,297)} km")
#     if np.isfinite(ate_v):
#         print(f"  ATE={ate_v:.1f}{ind(ate_v,79.94)}  CTE={cte_v:.1f}{ind(cte_v,93.58)}")
#     beat = _beat_report(r)
#     if beat: print(f"  {beat}")
#     print(f"  Score = {score:.2f}")
#     print(f"{'='*70}\n")

#     from datetime import datetime
#     dm = DatasetMetrics(
#         ade=ade_v if np.isfinite(ade_v) else 0.0,
#         fde=r.get("FDE", 0.0) if np.isfinite(r.get("FDE", float("nan"))) else 0.0,
#         ugde_12h=h12_v if np.isfinite(h12_v) else 0.0,
#         ugde_24h=h24_v if np.isfinite(h24_v) else 0.0,
#         ugde_48h=h48_v if np.isfinite(h48_v) else 0.0,
#         ugde_72h=h72_v if np.isfinite(h72_v) else 0.0,
#         ate_abs_mean=ate_v if np.isfinite(ate_v) else 0.0,
#         cte_abs_mean=cte_v if np.isfinite(cte_v) else 0.0,
#         n_total=r.get("n_samples", 0),
#         timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
#     )
#     save_metrics_csv(dm, metrics_csv, tag=f"val_{tag}_ep{epoch:03d}")

#     if backup is not None:
#         try:
#             ema_obj.restore(model, backup)
#         except Exception as e:
#             print(f"  ⚠  EMA restore failed: {e}")
#     return r


# # ══════════════════════════════════════════════════════════════════════════════
# #  Best Model Saver
# # ══════════════════════════════════════════════════════════════════════════════

# class HorizonAwareBestSaver:
#     def __init__(self, patience=25, tol=1.5):
#         self.patience   = patience
#         self.tol        = tol
#         self.counter    = 0
#         self.early_stop = False
#         self.best_score = float("inf")
#         self.best_ade   = float("inf")
#         self.best_12h   = float("inf")
#         self.best_24h   = float("inf")
#         self.best_48h   = float("inf")
#         self.best_72h   = float("inf")
#         self.best_ate   = float("inf")
#         self.best_cte   = float("inf")

#     def update(self, r, model, out_dir, epoch, optimizer, scheduler,
#                tl, vl, saver_ref, min_epochs=20):
#         ade   = r.get("ADE", float("inf"))
#         h12   = r.get("12h", float("inf"))
#         h24   = r.get("24h", float("inf"))
#         h48   = r.get("48h", float("inf"))
#         h72   = r.get("72h", float("inf"))
#         ate   = r.get("ATE_mean", ade * 0.46 if np.isfinite(ade) else float("inf"))
#         cte   = r.get("CTE_mean", ade * 0.53 if np.isfinite(ade) else float("inf"))
#         score = _composite_score(r)
#         improved_any = False

#         for metric_val, best_attr, fname, extra in [
#             (ade, 'best_ade', 'best_ade.pth',  {"ade": ade}),
#             (h72, 'best_72h', 'best_72h.pth',  {"h72": h72}),
#             (ate, 'best_ate', 'best_ate.pth',  {"ate": ate, "cte": cte}),
#             (cte, 'best_cte', 'best_cte.pth',  {"ate": ate, "cte": cte}),
#         ]:
#             if metric_val < getattr(self, best_attr):
#                 setattr(self, best_attr, metric_val)
#                 improved_any = True
#                 _save_checkpoint(os.path.join(out_dir, fname),
#                                   epoch, model, optimizer, scheduler,
#                                   saver_ref, tl, vl, extra)

#         if h48 < self.best_48h: self.best_48h = h48; improved_any = True
#         if h24 < self.best_24h: self.best_24h = h24; improved_any = True
#         if h12 < self.best_12h: self.best_12h = h12; improved_any = True

#         if score < self.best_score - self.tol:
#             self.best_score = score
#             self.counter    = 0
#             _save_checkpoint(
#                 os.path.join(out_dir, "best_model.pth"),
#                 epoch, model, optimizer, scheduler, saver_ref, tl, vl,
#                 {"ade": ade, "h12": h12, "h24": h24, "h48": h48, "h72": h72,
#                  "ate": ate, "cte": cte, "composite_score": score})
#             print(f"  ✅ Best COMPOSITE={score:.2f}  ADE={ade:.1f}  "
#                   f"72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}  (ep {epoch})")
#         else:
#             if not improved_any:
#                 self.counter += 1
#             print(f"  No improvement {self.counter}/{self.patience}"
#                   f"  (best={self.best_score:.2f} cur={score:.2f})"
#                   f"  72h={h72:.0f}↓{self.best_72h:.0f}")

#         if epoch >= min_epochs and self.counter >= self.patience:
#             self.early_stop = True


# # ══════════════════════════════════════════════════════════════════════════════
# #  Args
# # ══════════════════════════════════════════════════════════════════════════════

# def get_args():
#     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
#     p.add_argument("--obs_len",         default=8,          type=int)
#     p.add_argument("--pred_len",        default=12,         type=int)
#     p.add_argument("--batch_size",      default=32,         type=int)
#     p.add_argument("--num_epochs",      default=100,        type=int)
#     p.add_argument("--g_learning_rate", default=3e-4,       type=float)
#     p.add_argument("--weight_decay",    default=5e-4,       type=float)
#     p.add_argument("--warmup_epochs",   default=5,          type=int)
#     p.add_argument("--grad_clip",       default=1.0,        type=float)
#     p.add_argument("--patience",        default=25,         type=int)
#     p.add_argument("--min_epochs",      default=20,         type=int)
#     p.add_argument("--n_train_ens",     default=4,          type=int)
#     p.add_argument("--use_amp",         action="store_true")
#     p.add_argument("--num_workers",     default=2,          type=int)
#     p.add_argument("--sigma_min",            default=0.02,  type=float)
#     p.add_argument("--ctx_noise_scale",      default=0.01,  type=float)
#     p.add_argument("--initial_sample_sigma", default=0.03,  type=float)
#     p.add_argument("--ode_steps_train", default=10,  type=int)
#     p.add_argument("--ode_steps_val",   default=10,  type=int)
#     p.add_argument("--ode_steps_test",  default=10,  type=int)
#     p.add_argument("--val_ensemble",    default=20,  type=int)
#     p.add_argument("--fast_ensemble",   default=10,  type=int)
#     p.add_argument("--val_freq",        default=5,   type=int)
#     p.add_argument("--val_ade_freq",    default=5,   type=int)
#     p.add_argument("--val_subset_size", default=500, type=int)
#     p.add_argument("--use_ema",         action="store_true", default=True)
#     p.add_argument("--ema_decay",       default=0.999, type=float)
#     p.add_argument("--swa_start_epoch", default=60,   type=int)
#     p.add_argument("--teacher_forcing", action="store_true", default=True)
#     p.add_argument("--output_dir",  default="runs/v46",        type=str)
#     p.add_argument("--metrics_csv", default="metrics_v46.csv", type=str)
#     p.add_argument("--predict_csv", default="predictions.csv", type=str)
#     p.add_argument("--gpu_num",     default="0", type=str)
#     p.add_argument("--delim",       default=" ")
#     p.add_argument("--skip",        default=1,   type=int)
#     p.add_argument("--min_ped",     default=1,   type=int)
#     p.add_argument("--threshold",   default=0.002, type=float)
#     p.add_argument("--other_modal", default="gph")
#     p.add_argument("--test_year",   default=None, type=int)
#     p.add_argument("--resume",       default=None, type=str)
#     p.add_argument("--resume_epoch", default=None, type=int)
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

#     print("=" * 72)
#     print("  TC-FlowMatching v46  |  ST-Trans Loss Edition")
#     print("  LOSS: DPE_weighted + 0.05*MSE + 0.1*speed + 0.01*accel")
#     print("  SIGMA: 0.15→0.03 (fast convergence)")
#     print("  TARGETS: 72h<297 | ATE<79.94 | CTE<93.58 | ADE<136.41")
#     print("  MONITOR: dpe↓, h72↓ — phải < 350km ở ep15")
#     print(f"  EMA decay: {args.ema_decay}")
#     print("=" * 72)

#     train_dataset, train_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "train"}, test=False)
#     val_dataset, val_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "val"}, test=True)

#     from Model.data.trajectoriesWithMe_unet_training import seq_collate
#     val_subset_loader = make_val_subset_loader(
#         val_dataset, args.val_subset_size, args.batch_size,
#         seq_collate, args.num_workers)

#     print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
#     print(f"  val   : {len(val_dataset)} seq")

#     model = TCFlowMatching(
#         pred_len=args.pred_len, obs_len=args.obs_len,
#         sigma_min=args.sigma_min, n_train_ens=args.n_train_ens,
#         ctx_noise_scale=args.ctx_noise_scale,
#         initial_sample_sigma=args.initial_sample_sigma,
#         teacher_forcing=args.teacher_forcing,
#         use_ema=args.use_ema, ema_decay=args.ema_decay,
#     ).to(device)

#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"  params  : {n_params:,}")
#     model.init_ema()
#     print(f"  EMA     : {'ON' if model._ema is not None else 'OFF'}")

#     optimizer = optim.AdamW(model.parameters(),
#                              lr=args.g_learning_rate,
#                              weight_decay=args.weight_decay)
#     saver  = HorizonAwareBestSaver(patience=args.patience, tol=1.5)
#     scaler = GradScaler("cuda", enabled=args.use_amp)

#     steps_per_epoch = len(train_loader)
#     total_steps     = steps_per_epoch * args.num_epochs
#     warmup          = steps_per_epoch * args.warmup_epochs
#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer, warmup, total_steps, min_lr=1e-6)

#     # ── Resume ───────────────────────────────────────────────────────────────
#     has_scheduler_state   = False
#     _ckpt_scheduler_state = None
#     start_epoch = 0

#     if args.resume is not None and os.path.exists(args.resume):
#         print(f"  Loading checkpoint: {args.resume}")
#         ckpt = torch.load(args.resume, map_location=device)
#         m = _get_raw_model(model)
#         missing, unexpected = m.load_state_dict(ckpt["model_state_dict"], strict=False)
#         if missing:    print(f"  ⚠  Missing keys ({len(missing)})")
#         if unexpected: print(f"  ⚠  Unexpected keys ({len(unexpected)})")
#         if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
#             ema_obj = _get_ema_obj(model)
#             if ema_obj is not None and hasattr(ema_obj, 'shadow'):
#                 loaded = 0
#                 for k, v in ckpt["ema_shadow"].items():
#                     if k in ema_obj.shadow:
#                         ema_obj.shadow[k].copy_(v.to(device)); loaded += 1
#                 print(f"  EMA shadow loaded: {loaded} keys")
#         if "optimizer_state" in ckpt:
#             try:
#                 optimizer.load_state_dict(ckpt["optimizer_state"])
#                 for state in optimizer.state.values():
#                     for k, v in state.items():
#                         if torch.is_tensor(v): state[k] = v.to(device)
#             except Exception as e:
#                 print(f"  ⚠  Optimizer restore failed: {e}")
#         has_scheduler_state   = "scheduler_state" in ckpt
#         _ckpt_scheduler_state = ckpt.get("scheduler_state", None)
#         if "best_score" in ckpt:
#             saver.best_score = ckpt["best_score"]
#             saver.best_ade   = ckpt.get("best_ade",  float("inf"))
#             saver.best_72h   = ckpt.get("best_72h",  float("inf"))
#             saver.best_ate   = ckpt.get("best_ate",  float("inf"))
#             saver.best_cte   = ckpt.get("best_cte",  float("inf"))
#         start_epoch = (args.resume_epoch if args.resume_epoch is not None
#                        else ckpt.get("epoch", 0) + 1)
#         print(f"  → Resuming from epoch {start_epoch}")

#     try:
#         model = torch.compile(model, mode="reduce-overhead")
#         print("  torch.compile: enabled")
#     except Exception:
#         pass

#     if has_scheduler_state and _ckpt_scheduler_state is not None:
#         try:
#             scheduler.load_state_dict(_ckpt_scheduler_state)
#         except Exception as e:
#             print(f"  ⚠  Scheduler restore failed: {e}")
#             for _ in range(start_epoch * steps_per_epoch): scheduler.step()
#     elif start_epoch > 0:
#         for _ in range(start_epoch * steps_per_epoch): scheduler.step()

#     print("=" * 72)
#     print(f"  TRAINING  ({steps_per_epoch} steps/epoch, start={start_epoch})")
#     print(f"  KEY: dpe phải < 2.0 sau ep10, h72 < 350km sau ep15")
#     print("=" * 72)

#     epoch_times = []
#     train_start = time.perf_counter()

#     for epoch in range(start_epoch, args.num_epochs):
#         model.train()
#         sum_loss = 0.0
#         t0 = time.perf_counter()

#         for i, batch in enumerate(train_loader):
#             bl = move(list(batch), device)

#             with autocast(device_type="cuda", enabled=args.use_amp):
#                 bd = model.get_loss_breakdown(bl, epoch=epoch)

#             optimizer.zero_grad()
#             scaler.scale(bd["total"]).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

#             scale_before = scaler.get_scale()
#             scaler.step(optimizer)
#             scaler.update()
#             if scaler.get_scale() >= scale_before:
#                 scheduler.step()
#             _call_ema_update(model)

#             sum_loss += bd["total"].item()

#             if i % 20 == 0:
#                 lr = optimizer.param_groups[0]["lr"]
#                 print(
#                     f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
#                     f"  tot={bd['total'].item():.3f}"
#                     f"  dpe={bd.get('dpe',    0):.3f}"   # primary signal
#                     f"  spd={bd.get('speed',  0):.4f}"   # speed penalty
#                     f"  acc={bd.get('accel',  0):.4f}"   # accel penalty
#                     f"  fm={bd.get('fm_mse',  0):.4f}"   # FM velocity loss
#                     f"  σ={bd.get('sigma',    0):.3f}"   # current sigma
#                     f"  h72={bd.get('h72_km', 0):.0f}km" # 72h approx
#                     f"  lr={lr:.2e}"
#                 )

#         ep_s  = time.perf_counter() - t0
#         epoch_times.append(ep_s)
#         avg_t = sum_loss / len(train_loader)

#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch in val_loader:
#                 bl_v = move(list(batch), device)
#                 with autocast(device_type="cuda", enabled=args.use_amp):
#                     val_loss += model.get_loss(bl_v, epoch=epoch).item()
#         avg_vl = val_loss / len(val_loader)
#         print(f"  Epoch {epoch:>3}  train={avg_t:.4f}  val={avg_vl:.4f}  time={ep_s:.0f}s")

#         # ── Fast eval every epoch ────────────────────────────────────────────
#         m_fast = evaluate_fast(model, val_subset_loader, device,
#                                 args.ode_steps_train, args.pred_len,
#                                 args.fast_ensemble)
#         h12 = m_fast.get("12h", float("nan"))
#         h24 = m_fast.get("24h", float("nan"))
#         h48 = m_fast.get("48h", float("nan"))
#         h72 = m_fast.get("72h", float("nan"))
#         ate = m_fast.get("ATE_mean", float("nan"))
#         cte = m_fast.get("CTE_mean", float("nan"))
#         fast_score = _composite_score(m_fast)

#         def ind(v, tgt):
#             if not np.isfinite(v): return "?"
#             return "🎯" if v < tgt else "❌"

#         print(
#             f"  [FAST ep{epoch}]"
#             f"  ADE={m_fast.get('ADE', float('nan')):.1f}"
#             f"  12h={h12:.0f}{ind(h12,50)}"
#             f"  24h={h24:.0f}{ind(h24,100)}"
#             f"  48h={h48:.0f}{ind(h48,200)}"
#             f"  72h={h72:.0f}{ind(h72,297)}"
#             f"  ATE={ate:.1f}{ind(ate,79.94)}"
#             f"  CTE={cte:.1f}{ind(cte,93.58)}"
#             f"  score={fast_score:.2f}"
#         )

#         # ── Full val every val_freq epochs ───────────────────────────────────
#         if epoch % args.val_freq == 0:
#             ema_obj = _get_ema_obj(model)
#             try:
#                 r_raw = evaluate_full_val_ade(
#                     model, val_loader, device,
#                     ode_steps=args.ode_steps_val, pred_len=args.pred_len,
#                     fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
#                     epoch=epoch, use_ema=False)
#                 r_use = r_raw
#                 if ema_obj is not None and epoch >= 5:
#                     r_ema = evaluate_full_val_ade(
#                         model, val_loader, device,
#                         ode_steps=args.ode_steps_val, pred_len=args.pred_len,
#                         fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
#                         epoch=epoch, use_ema=True, ema_obj=ema_obj)
#                     if _composite_score(r_ema) < _composite_score(r_raw):
#                         r_use = r_ema
#                 saver.update(r_use, model, args.output_dir, epoch,
#                              optimizer, scheduler, avg_t, avg_vl,
#                              saver_ref=saver, min_epochs=args.min_epochs)
#             except Exception as e:
#                 print(f"  ⚠  Full val failed: {e}")
#                 import traceback; traceback.print_exc()

#         if epoch % 10 == 0 or epoch == args.num_epochs - 1:
#             _save_checkpoint(
#                 os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"),
#                 epoch, model, optimizer, scheduler, saver, avg_t, avg_vl,
#                 {"ade": m_fast.get("ADE", float("nan")), "h72": h72, "ate": ate, "cte": cte})

#         if epoch % 10 == 9:
#             avg_ep    = sum(epoch_times) / len(epoch_times)
#             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
#             elapsed_h = (time.perf_counter() - train_start) / 3600
#             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")
#             print(f"  📊 Best: score={saver.best_score:.2f}  "
#                   f"72h={saver.best_72h:.0f}km  ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}")

#         if saver.early_stop:
#             print(f"  ⛔ Early stopping at epoch {epoch}")
#             break

#     total_h = (time.perf_counter() - train_start) / 3600
#     print(f"\n  Best: ADE={saver.best_ade:.1f}  72h={saver.best_72h:.0f}  "
#           f"ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}")
#     print(f"  Total time: {total_h:.2f}h")
#     beat = _beat_report({"ADE": saver.best_ade, "72h": saver.best_72h,
#                           "ATE_mean": saver.best_ate, "CTE_mean": saver.best_cte})
#     if beat: print(f"\n  🏆🏆🏆 {beat} 🏆🏆🏆")
#     print("=" * 72)


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(42); torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
#     main(args)

"""
scripts/train_flowmatching_v50.py — ATE/72h Breakthrough Edition
═══════════════════════════════════════════════════════════════════════════════

KEY CHANGES vs v46:
  1. Loss v50: DPE(v50_weights) + 0.03*MSE + 0.05*speed + 0.01*accel
               + 0.15*heading + 0.30*vel_reg + 0.25*endpoint
  2. STEP_WEIGHTS_V50: [1,1,1,1.5,2,2.5,3,4,5,6,7,10] — 72h = 10x
  3. Sigma schedule: 0.15→0.04 từ ep2 (nhanh hơn v46)
  4. Inference v50: speed-sweep ensemble (7 scales × 50 members)
  5. Log thêm: vel_reg, endpoint, heading

MONITOR:
  - vel_reg phải giảm liên tục: ep0~2.0, ep10~0.5, ep20~0.3
  - endpoint phải < 1.0 ở ep10
  - heading phải < 0.3 ở ep15
  - 72h phải < 350 ở ep15, < 300 ở ep30
  - ATE phải < 150 ở ep20, < 80 ở ep40

NẾU vel_reg không giảm sau ep5: tăng weight 0.30→0.40 trong patch
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, time, random
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from Model.data.loader_training import data_loader
from Model.flow_matching_model import TCFlowMatching   # v50 standalone

from Model.utils import get_cosine_schedule_with_warmup
from utils.metrics import (
    StepErrorAccumulator, save_metrics_csv,
    haversine_km_torch, denorm_torch,
    HORIZON_STEPS, DatasetMetrics,
)

try:
    from utils.metrics import haversine_and_atecte_torch
    HAS_ATECTE = True
except ImportError:
    HAS_ATECTE = False


# ══════════════════════════════════════════════════════════════════════════════
#  Score — giống v46 (không đổi)
# ══════════════════════════════════════════════════════════════════════════════

def _composite_score(result: dict) -> float:
    ade = result.get("ADE", float("inf"))
    h72 = result.get("72h", float("inf"))
    ate = result.get("ATE_mean", float("inf"))
    cte = result.get("CTE_mean", float("inf"))
    if not np.isfinite(ate): ate = ade * 0.46
    if not np.isfinite(cte): cte = ade * 0.53
    h12 = result.get("12h", float("inf"))
    h24 = result.get("24h", float("inf"))
    h48 = result.get("48h", float("inf"))
    return 100.0 * (
        0.05 * (ade  / 136.0)
        + 0.05 * (h12  / 50.0)
        + 0.10 * (h24  / 100.0)
        + 0.15 * (h48  / 200.0)
        + 0.35 * (h72  / 300.0)
        + 0.15 * (ate  / 80.0)
        + 0.15 * (cte  / 94.0)
    )


def _beat_report(r: dict) -> str:
    parts = []
    if np.isfinite(r.get("ADE",      float("inf"))) and r["ADE"]      < 136.41: parts.append(f"ADE✅{r['ADE']:.1f}")
    if np.isfinite(r.get("ATE_mean", float("inf"))) and r["ATE_mean"] < 79.94:  parts.append(f"ATE✅{r['ATE_mean']:.1f}")
    if np.isfinite(r.get("CTE_mean", float("inf"))) and r["CTE_mean"] < 93.58:  parts.append(f"CTE✅{r['CTE_mean']:.1f}")
    if np.isfinite(r.get("72h",      float("inf"))) and r["72h"]      < 297.0:  parts.append(f"72h✅{r['72h']:.1f}")
    return "🏆 BEAT: " + " ".join(parts) if parts else ""


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers — giống v46
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


def make_val_subset_loader(val_dataset, subset_size, batch_size, collate_fn, num_workers):
    n   = len(val_dataset)
    rng = random.Random(42)
    idx = rng.sample(range(n), min(subset_size, n))
    return DataLoader(Subset(val_dataset, idx),
                      batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=0, drop_last=False)


def _get_ema_obj(model):
    if hasattr(model, '_ema') and model._ema is not None: return model._ema
    if hasattr(model, '_orig_mod'):
        orig = model._orig_mod
        if hasattr(orig, '_ema') and orig._ema is not None: return orig._ema
    return None


def _call_ema_update(model):
    if hasattr(model, '_orig_mod'):
        orig = model._orig_mod
        if hasattr(orig, 'ema_update'): orig.ema_update(); return
    if hasattr(model, 'ema_update'): model.ema_update()


def _get_raw_model(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model


def _save_checkpoint(path, epoch, model, optimizer, scheduler,
                     saver, avg_t, avg_vl, metrics=None):
    m = _get_raw_model(model)
    ema_obj = _get_ema_obj(model)
    ema_sd  = None
    if ema_obj is not None and hasattr(ema_obj, 'shadow'):
        try:    ema_sd = {k: v.cpu().clone() for k, v in ema_obj.shadow.items()}
        except: ema_sd = None
    payload = {
        "epoch": epoch, "model_state_dict": m.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "ema_shadow": ema_sd,
        "best_score": saver.best_score, "best_ade": saver.best_ade,
        "best_72h": saver.best_72h, "best_48h": saver.best_48h,
        "best_24h": saver.best_24h, "best_12h": saver.best_12h,
        "best_ate": getattr(saver, 'best_ate', float("inf")),
        "best_cte": getattr(saver, 'best_cte', float("inf")),
        "train_loss": avg_t, "val_loss": avg_vl,
    }
    if metrics: payload.update(metrics)
    torch.save(payload, path)


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation — giống v46
# ══════════════════════════════════════════════════════════════════════════════

def _eval_batch_atecte(pred_norm, gt_norm):
    pred_d = denorm_torch(pred_norm)
    gt_d   = denorm_torch(gt_norm)
    if HAS_ATECTE:
        dist, ate, cte = haversine_and_atecte_torch(pred_d, gt_d, unit_01deg=True)
        return dist, ate, cte
    dist = haversine_km_torch(pred_d, gt_d)
    return dist, None, None


def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=15):
    model.eval()
    acc = StepErrorAccumulator(pred_len)
    t0, n = time.perf_counter(), 0
    with torch.no_grad():
        for batch in loader:
            bl = move(list(batch), device)
            pred, _, _ = model.sample(bl, num_ensemble=max(fast_ensemble, 20),
                                       ddim_steps=ode_steps, importance_weight=True)
            gt_sliced  = bl[1][:pred.shape[0]]
            dist, ate, cte = _eval_batch_atecte(pred, gt_sliced)
            acc.update(dist, ate_km=ate, cte_km=cte)
            n += 1
    r = acc.compute()
    r["ms_per_batch"] = (time.perf_counter()-t0)*1e3/max(n, 1)
    return r


def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
                           fast_ensemble, metrics_csv, epoch,
                           use_ema=False, ema_obj=None):
    backup = None
    if use_ema and ema_obj is not None:
        try:    backup = ema_obj.apply_to(model)
        except Exception as e:
            print(f"  ⚠  EMA apply_to failed: {e}")
            backup = None; use_ema = False

    model.eval()
    acc = StepErrorAccumulator(pred_len)
    t0, n = time.perf_counter(), 0
    with torch.no_grad():
        for batch in val_loader:
            bl   = move(list(batch), device)
            pred, _, _ = model.sample(bl, num_ensemble=max(fast_ensemble, 20),
                                       ddim_steps=max(ode_steps, 20),
                                       importance_weight=True)
            gt   = bl[1][:pred.shape[0]]
            dist, ate, cte = _eval_batch_atecte(pred, gt)
            acc.update(dist, ate_km=ate, cte_km=cte)
            n += 1

    r       = acc.compute()
    elapsed = time.perf_counter() - t0
    score   = _composite_score(r)
    tag     = "EMA" if use_ema else "RAW"

    def _v(k): return r.get(k, float("nan"))
    def ind(v, t): return ("✅" if v < t else "❌") if np.isfinite(v) else ""

    print(f"\n{'='*70}")
    print(f"  [FULL VAL ({tag})  ep={epoch}  {elapsed:.0f}s  {n} batches]")
    print(f"  ADE={_v('ADE'):.1f}{ind(_v('ADE'),136.41)} km")
    print(f"  12h={_v('12h'):.0f}{ind(_v('12h'),50)}  "
          f"24h={_v('24h'):.0f}{ind(_v('24h'),100)}  "
          f"48h={_v('48h'):.0f}{ind(_v('48h'),200)}  "
          f"72h={_v('72h'):.0f}{ind(_v('72h'),297)} km")
    if np.isfinite(_v("ATE_mean")):
        print(f"  ATE={_v('ATE_mean'):.1f}{ind(_v('ATE_mean'),79.94)}"
              f"  CTE={_v('CTE_mean'):.1f}{ind(_v('CTE_mean'),93.58)}")
    beat = _beat_report(r)
    if beat: print(f"  {beat}")
    print(f"  Score = {score:.2f}")
    print(f"{'='*70}\n")

    from datetime import datetime
    dm = DatasetMetrics(
        ade=_v("ADE") if np.isfinite(_v("ADE")) else 0.0,
        fde=_v("FDE") if np.isfinite(_v("FDE")) else 0.0,
        ugde_12h=_v("12h") if np.isfinite(_v("12h")) else 0.0,
        ugde_24h=_v("24h") if np.isfinite(_v("24h")) else 0.0,
        ugde_48h=_v("48h") if np.isfinite(_v("48h")) else 0.0,
        ugde_72h=_v("72h") if np.isfinite(_v("72h")) else 0.0,
        ate_abs_mean=_v("ATE_mean") if np.isfinite(_v("ATE_mean")) else 0.0,
        cte_abs_mean=_v("CTE_mean") if np.isfinite(_v("CTE_mean")) else 0.0,
        n_total=r.get("n_samples", 0),
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    save_metrics_csv(dm, metrics_csv, tag=f"val_{tag}_ep{epoch:03d}")

    if backup is not None:
        try:    ema_obj.restore(model, backup)
        except Exception as e: print(f"  ⚠  EMA restore failed: {e}")
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  Best Model Saver — giống v46
# ══════════════════════════════════════════════════════════════════════════════

class HorizonAwareBestSaver:
    def __init__(self, patience=25, tol=1.5):
        self.patience = patience; self.tol = tol; self.counter = 0
        self.early_stop = False
        self.best_score = self.best_ade = self.best_12h = self.best_24h = \
            self.best_48h = self.best_72h = self.best_ate = self.best_cte = float("inf")

    def update(self, r, model, out_dir, epoch, optimizer, scheduler,
               tl, vl, saver_ref, min_epochs=20):
        ade   = r.get("ADE", float("inf"))
        h12   = r.get("12h", float("inf"))
        h24   = r.get("24h", float("inf"))
        h48   = r.get("48h", float("inf"))
        h72   = r.get("72h", float("inf"))
        ate   = r.get("ATE_mean", ade*0.46 if np.isfinite(ade) else float("inf"))
        cte   = r.get("CTE_mean", ade*0.53 if np.isfinite(ade) else float("inf"))
        score = _composite_score(r)
        improved_any = False

        for metric_val, best_attr, fname, extra in [
            (ade, 'best_ade', 'best_ade.pth', {"ade": ade}),
            (h72, 'best_72h', 'best_72h.pth', {"h72": h72}),
            (ate, 'best_ate', 'best_ate.pth', {"ate": ate, "cte": cte}),
            (cte, 'best_cte', 'best_cte.pth', {"ate": ate, "cte": cte}),
        ]:
            if metric_val < getattr(self, best_attr):
                setattr(self, best_attr, metric_val); improved_any = True
                _save_checkpoint(os.path.join(out_dir, fname),
                                  epoch, model, optimizer, scheduler, saver_ref, tl, vl, extra)

        for v, a in [(h48,'best_48h'),(h24,'best_24h'),(h12,'best_12h')]:
            if v < getattr(self, a): setattr(self, a, v); improved_any = True

        if score < self.best_score - self.tol:
            self.best_score = score; self.counter = 0
            _save_checkpoint(os.path.join(out_dir, "best_model.pth"),
                             epoch, model, optimizer, scheduler, saver_ref, tl, vl,
                             {"ade":ade,"h12":h12,"h24":h24,"h48":h48,"h72":h72,
                              "ate":ate,"cte":cte,"composite_score":score})
            print(f"  ✅ Best COMPOSITE={score:.2f}  ADE={ade:.1f}  "
                  f"72h={h72:.0f}  ATE={ate:.1f}  CTE={cte:.1f}  (ep {epoch})")
        else:
            if not improved_any: self.counter += 1
            print(f"  No improvement {self.counter}/{self.patience}"
                  f"  (best={self.best_score:.2f} cur={score:.2f})"
                  f"  72h={h72:.0f}↓{self.best_72h:.0f}")

        if epoch >= min_epochs and self.counter >= self.patience:
            self.early_stop = True


# ══════════════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
    p.add_argument("--obs_len",         default=8,          type=int)
    p.add_argument("--pred_len",        default=12,         type=int)
    p.add_argument("--batch_size",      default=32,         type=int)
    p.add_argument("--num_epochs",      default=100,        type=int)
    p.add_argument("--g_learning_rate", default=3e-4,       type=float)
    p.add_argument("--weight_decay",    default=5e-4,       type=float)
    p.add_argument("--warmup_epochs",   default=5,          type=int)
    p.add_argument("--grad_clip",       default=1.0,        type=float)
    p.add_argument("--patience",        default=25,         type=int)
    p.add_argument("--min_epochs",      default=20,         type=int)
    p.add_argument("--n_train_ens",     default=4,          type=int)
    p.add_argument("--use_amp",         action="store_true")
    p.add_argument("--num_workers",     default=2,          type=int)
    p.add_argument("--sigma_min",            default=0.02,  type=float)
    p.add_argument("--ctx_noise_scale",      default=0.01,  type=float)
    p.add_argument("--initial_sample_sigma", default=0.03,  type=float)
    p.add_argument("--ode_steps_train", default=10,  type=int)
    p.add_argument("--ode_steps_val",   default=10,  type=int)
    p.add_argument("--ode_steps_test",  default=10,  type=int)
    p.add_argument("--val_ensemble",    default=20,  type=int)
    p.add_argument("--fast_ensemble",   default=10,  type=int)
    p.add_argument("--val_freq",        default=5,   type=int)
    p.add_argument("--val_ade_freq",    default=5,   type=int)
    p.add_argument("--val_subset_size", default=500, type=int)
    p.add_argument("--use_ema",         action="store_true", default=True)
    p.add_argument("--ema_decay",       default=0.999, type=float)
    p.add_argument("--swa_start_epoch", default=60,   type=int)
    p.add_argument("--teacher_forcing", action="store_true", default=True)
    p.add_argument("--output_dir",  default="runs/v50",        type=str)   # v46→v50
    p.add_argument("--metrics_csv", default="metrics_v50.csv", type=str)   # v46→v50
    p.add_argument("--predict_csv", default="predictions.csv", type=str)
    p.add_argument("--gpu_num",     default="0", type=str)
    p.add_argument("--delim",       default=" ")
    p.add_argument("--skip",        default=1,   type=int)
    p.add_argument("--min_ped",     default=1,   type=int)
    p.add_argument("--threshold",   default=0.002, type=float)
    p.add_argument("--other_modal", default="gph")
    p.add_argument("--test_year",   default=None, type=int)
    p.add_argument("--resume",       default=None, type=str)
    p.add_argument("--resume_epoch", default=None, type=int)
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

    print("=" * 72)
    print("  TC-FlowMatching v50  |  Standalone Edition")
    print("  LOSS: DPE(v50_w) + 0.03*MSE + 0.05*spd + 0.01*acc")
    print("        + 0.15*heading + 0.30*vel_reg + 0.25*endpoint")
    print("  SIGMA: 0.15→0.04 (từ ep2)")
    print("  TARGETS: 72h<297 | ATE<79.94 | CTE<93.58 | ADE<136.41")
    print("  MONITOR: vel_reg↓ (<0.5 ở ep10)  endpoint↓ (<1.0 ở ep10)")
    print(f"  EMA decay: {args.ema_decay}")
    print("=" * 72)

    train_dataset, train_loader = data_loader(
        args, {"root": args.dataset_root, "type": "train"}, test=False)
    val_dataset, val_loader = data_loader(
        args, {"root": args.dataset_root, "type": "val"}, test=True)

    from Model.data.trajectoriesWithMe_unet_training import seq_collate
    val_subset_loader = make_val_subset_loader(
        val_dataset, args.val_subset_size, args.batch_size,
        seq_collate, args.num_workers)

    print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
    print(f"  val   : {len(val_dataset)} seq")

    model = TCFlowMatching(
        pred_len=args.pred_len, obs_len=args.obs_len,
        sigma_min=args.sigma_min, n_train_ens=args.n_train_ens,
        ctx_noise_scale=args.ctx_noise_scale,
        initial_sample_sigma=args.initial_sample_sigma,
        teacher_forcing=args.teacher_forcing,
        use_ema=args.use_ema, ema_decay=args.ema_decay,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params  : {n_params:,}")
    model.init_ema()
    print(f"  EMA     : {'ON' if model._ema is not None else 'OFF'}")

    optimizer = optim.AdamW(model.parameters(),
                             lr=args.g_learning_rate, weight_decay=args.weight_decay)
    saver  = HorizonAwareBestSaver(patience=args.patience, tol=1.5)
    scaler = GradScaler("cuda", enabled=args.use_amp)

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.num_epochs
    warmup          = steps_per_epoch * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup, total_steps, min_lr=1e-6)

    # ── Resume ───────────────────────────────────────────────────────────────
    has_scheduler_state = False; _ckpt_sched = None; start_epoch = 0

    if args.resume is not None and os.path.exists(args.resume):
        print(f"  Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        m    = _get_raw_model(model)
        missing, unexpected = m.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:    print(f"  ⚠  Missing keys ({len(missing)})")
        if unexpected: print(f"  ⚠  Unexpected keys ({len(unexpected)})")
        if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
            ema_obj = _get_ema_obj(model)
            if ema_obj is not None and hasattr(ema_obj, 'shadow'):
                loaded = sum(1 for k, v in ckpt["ema_shadow"].items()
                             if k in ema_obj.shadow and not ema_obj.shadow[k].copy_(v.to(device)) is None
                             or k in ema_obj.shadow)
                print(f"  EMA shadow: loaded")
        if "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v): state[k] = v.to(device)
            except Exception as e: print(f"  ⚠  Optimizer restore failed: {e}")
        has_scheduler_state = "scheduler_state" in ckpt
        _ckpt_sched         = ckpt.get("scheduler_state", None)
        for attr in ("best_score","best_ade","best_72h","best_ate","best_cte"):
            if attr in ckpt: setattr(saver, attr, ckpt[attr])
        start_epoch = (args.resume_epoch if args.resume_epoch is not None
                       else ckpt.get("epoch", 0) + 1)
        print(f"  → Resuming from epoch {start_epoch}")

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: enabled")
    except Exception:
        pass

    if has_scheduler_state and _ckpt_sched is not None:
        try:    scheduler.load_state_dict(_ckpt_sched)
        except Exception as e:
            print(f"  ⚠  Scheduler restore: {e}")
            for _ in range(start_epoch * steps_per_epoch): scheduler.step()
    elif start_epoch > 0:
        for _ in range(start_epoch * steps_per_epoch): scheduler.step()

    print("=" * 72)
    print(f"  TRAINING  ({steps_per_epoch} steps/epoch, start={start_epoch})")
    print(f"  KEY: vel_reg < 0.5 ở ep10 | endpoint < 1.0 ở ep10 | 72h < 350 ở ep15")
    print("=" * 72)

    epoch_times = []
    train_start = time.perf_counter()

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        sum_loss = 0.0
        t0 = time.perf_counter()

        for i, batch in enumerate(train_loader):
            bl = move(list(batch), device)

            with autocast(device_type="cuda", enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl, epoch=epoch)

            optimizer.zero_grad()
            scaler.scale(bd["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before: scheduler.step()
            _call_ema_update(model)

            sum_loss += bd["total"].item()

            if i % 20 == 0:
                lr = optimizer.param_groups[0]["lr"]
                # [v50] Log thêm vel_reg, endpoint, heading
                print(
                    f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
                    f"  tot={bd['total'].item():.3f}"
                    f"  dpe={bd.get('dpe',      0):.3f}"
                    f"  vel={bd.get('vel_reg',  0):.4f}"
                    f"  ate={bd.get('ate_mean_km', 0):.1f}km"
                    f"  cte={bd.get('cte_mean_km', 0):.1f}km"
                    f"  ep={bd.get('endpoint',  0):.4f}"
                    f"  fm={bd.get('fm_mse',   0):.4f}"
                    f"  σ={bd.get('sigma',     0):.3f}"
                    f"  lr={lr:.2e}"
                )

        ep_s  = time.perf_counter() - t0
        epoch_times.append(ep_s)
        avg_t = sum_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                bl_v = move(list(batch), device)
                with autocast(device_type="cuda", enabled=args.use_amp):
                    val_loss += model.get_loss(bl_v, epoch=epoch).item()
        avg_vl = val_loss / len(val_loader)
        print(f"  Epoch {epoch:>3}  train={avg_t:.4f}  val={avg_vl:.4f}  time={ep_s:.0f}s")

        # ── Fast eval mỗi epoch ──────────────────────────────────────────────
        m_fast    = evaluate_fast(model, val_subset_loader, device,
                                   args.ode_steps_train, args.pred_len,
                                   args.fast_ensemble)
        fast_score = _composite_score(m_fast)

        def ind(v, t):
            if not np.isfinite(v): return "?"
            return "🎯" if v < t else "❌"

        print(
            f"  [FAST ep{epoch}]"
            f"  ADE={m_fast.get('ADE', float('nan')):.1f}"
            f"  12h={m_fast.get('12h', float('nan')):.0f}{ind(m_fast.get('12h', 1e9), 50)}"
            f"  24h={m_fast.get('24h', float('nan')):.0f}{ind(m_fast.get('24h', 1e9), 100)}"
            f"  48h={m_fast.get('48h', float('nan')):.0f}{ind(m_fast.get('48h', 1e9), 200)}"
            f"  72h={m_fast.get('72h', float('nan')):.0f}{ind(m_fast.get('72h', 1e9), 297)}"
            f"  ATE={m_fast.get('ATE_mean', float('nan')):.1f}{ind(m_fast.get('ATE_mean', 1e9), 79.94)}"
            f"  CTE={m_fast.get('CTE_mean', float('nan')):.1f}{ind(m_fast.get('CTE_mean', 1e9), 93.58)}"
            f"  score={fast_score:.2f}"
        )

        # ── Full val mỗi val_freq epochs ─────────────────────────────────────
        if epoch % args.val_freq == 0:
            ema_obj = _get_ema_obj(model)
            try:
                r_raw = evaluate_full_val_ade(
                    model, val_loader, device,
                    ode_steps=args.ode_steps_val, pred_len=args.pred_len,
                    fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
                    epoch=epoch, use_ema=False)
                r_use = r_raw
                if ema_obj is not None and epoch >= 5:
                    r_ema = evaluate_full_val_ade(
                        model, val_loader, device,
                        ode_steps=args.ode_steps_val, pred_len=args.pred_len,
                        fast_ensemble=args.fast_ensemble, metrics_csv=metrics_csv,
                        epoch=epoch, use_ema=True, ema_obj=ema_obj)
                    if _composite_score(r_ema) < _composite_score(r_raw):
                        r_use = r_ema
                saver.update(r_use, model, args.output_dir, epoch,
                             optimizer, scheduler, avg_t, avg_vl,
                             saver_ref=saver, min_epochs=args.min_epochs)
            except Exception as e:
                print(f"  ⚠  Full val failed: {e}")
                import traceback; traceback.print_exc()

        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            _save_checkpoint(
                os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"),
                epoch, model, optimizer, scheduler, saver, avg_t, avg_vl,
                {"ade": m_fast.get("ADE", float("nan")),
                 "h72": m_fast.get("72h", float("nan")),
                 "ate": m_fast.get("ATE_mean", float("nan")),
                 "cte": m_fast.get("CTE_mean", float("nan"))})

        if epoch % 10 == 9:
            avg_ep    = sum(epoch_times) / len(epoch_times)
            remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
            elapsed_h = (time.perf_counter() - train_start) / 3600
            print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining")
            print(f"  📊 Best: score={saver.best_score:.2f}  "
                  f"72h={saver.best_72h:.0f}km  ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}")

        if saver.early_stop:
            print(f"  ⛔ Early stopping at epoch {epoch}")
            break

    total_h = (time.perf_counter() - train_start) / 3600
    print(f"\n  Best: ADE={saver.best_ade:.1f}  72h={saver.best_72h:.0f}  "
          f"ATE={saver.best_ate:.1f}  CTE={saver.best_cte:.1f}")
    print(f"  Total time: {total_h:.2f}h")
    beat = _beat_report({"ADE":saver.best_ade, "72h":saver.best_72h,
                          "ATE_mean":saver.best_ate, "CTE_mean":saver.best_cte})
    if beat: print(f"\n  🏆🏆🏆 {beat} 🏆🏆🏆")
    print("=" * 72)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    main(args)
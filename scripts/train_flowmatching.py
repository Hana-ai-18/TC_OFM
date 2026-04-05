
# # # # """
# # # # scripts/train_flowmatching.py  ── v20
# # # # ======================================
# # # # FIXES vs v19:

# # # # FIX-V20-1  [ENSEMBLE COLLAPSE FIX] Model v15 sửa ensemble collapse bằng
# # # #            ctx noise injection + initial_sample_sigma lớn hơn. Training
# # # #            script cần pass ctx_noise_scale và initial_sample_sigma vào
# # # #            TCFlowMatching constructor.
# # # #            Thêm args: --ctx_noise_scale (default 0.05)
# # # #                       --initial_sample_sigma (default 0.3)

# # # # FIX-V20-2  [EARLY STOP] Giữ nguyên FIX-V19-1: ADE evaluated mỗi epoch.
# # # #            counter_ade = số epoch thực không cải thiện (đúng semantic).

# # # # FIX-V20-3  [SPREAD MONITOR] In ensemble spread (1σ km) mỗi epoch trong
# # # #            evaluate_fast để detect collapse sớm. Nếu spread < 10 km
# # # #            tại 72h thì warning.

# # # # Kept from v19:
# # # #     FIX-V19-1..2 (ADE mỗi epoch, counter semantic)
# # # #     FIX-V18-1..4 (cliper shape, patience reset, denorm)
# # # # """
# # # # from __future__ import annotations

# # # # import sys
# # # # import os
# # # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # # import argparse
# # # # import time
# # # # import math
# # # # import random
# # # # import copy

# # # # import numpy as np
# # # # import torch
# # # # import torch.optim as optim
# # # # from torch.amp import autocast, GradScaler
# # # # from torch.utils.data import DataLoader, Subset

# # # # from Model.data.loader_training import data_loader
# # # # from Model.flow_matching_model import TCFlowMatching
# # # # from Model.utils import get_cosine_schedule_with_warmup
# # # # from Model.losses import WEIGHTS as _BASE_WEIGHTS
# # # # from utils.metrics import (
# # # #     TCEvaluator, StepErrorAccumulator,
# # # #     save_metrics_csv, haversine_km_torch, denorm_torch, HORIZON_STEPS,
# # # #     cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
# # # #     brier_skill_score,
# # # # )
# # # # from utils.evaluation_tables import (
# # # #     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
# # # #     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
# # # #     DEFAULT_COMPUTE, paired_tests, persistence_errors, cliper_errors,
# # # # )
# # # # from scripts.statistical_tests import run_all_tests


# # # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # # def haversine_km_np(pred_deg: np.ndarray, gt_deg: np.ndarray) -> np.ndarray:
# # # #     pred_deg = np.atleast_2d(pred_deg)
# # # #     gt_deg   = np.atleast_2d(gt_deg)
# # # #     R = 6371.0
# # # #     lon1, lat1 = np.radians(pred_deg[:, 0]), np.radians(pred_deg[:, 1])
# # # #     lon2, lat2 = np.radians(gt_deg[:, 0]),   np.radians(gt_deg[:, 1])
# # # #     dlon = lon2 - lon1;  dlat = lat2 - lat1
# # # #     a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
# # # #     return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# # # # def denorm_deg_np(arr_norm: np.ndarray) -> np.ndarray:
# # # #     arr_norm = np.atleast_2d(arr_norm)
# # # #     out = arr_norm.copy()
# # # #     out[:, 0] = (arr_norm[:, 0] * 50.0 + 1800.0) / 10.0
# # # #     out[:, 1] = (arr_norm[:, 1] * 50.0) / 10.0
# # # #     return out


# # # # def seq_ade_km(pred_norm: np.ndarray, gt_norm: np.ndarray) -> float:
# # # #     return float(haversine_km_np(denorm_deg_np(pred_norm),
# # # #                                   denorm_deg_np(gt_norm)).mean())


# # # # # ── Adaptive weight schedules ─────────────────────────────────────────────────

# # # # def get_pinn_weight(epoch, warmup_epochs=30, w_start=0.01, w_end=0.1):
# # # #     if epoch >= warmup_epochs:
# # # #         return w_end
# # # #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# # # # def get_velocity_weight(epoch, warmup_epochs=20, w_start=0.5, w_end=1.5):
# # # #     if epoch >= warmup_epochs:
# # # #         return w_end
# # # #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# # # # def get_recurv_weight(epoch, warmup_epochs=10, w_start=0.3, w_end=1.0):
# # # #     if epoch >= warmup_epochs:
# # # #         return w_end
# # # #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# # # # def get_grad_clip(epoch, warmup_epochs=20, clip_start=2.0, clip_end=1.0):
# # # #     if epoch >= warmup_epochs:
# # # #         return clip_end
# # # #     return clip_start - (epoch / max(warmup_epochs-1, 1)) * (clip_start - clip_end)


# # # # def get_args():
# # # #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # # #     p.add_argument("--dataset_root",    default="TCND_vn",      type=str)
# # # #     p.add_argument("--obs_len",         default=8,              type=int)
# # # #     p.add_argument("--pred_len",        default=12,             type=int)
# # # #     p.add_argument("--test_year",       default=None,           type=int)
# # # #     p.add_argument("--batch_size",      default=32,             type=int)
# # # #     p.add_argument("--num_epochs",      default=200,            type=int)
# # # #     p.add_argument("--g_learning_rate", default=2e-4,           type=float)
# # # #     p.add_argument("--weight_decay",    default=1e-4,           type=float)
# # # #     p.add_argument("--warmup_epochs",   default=3,              type=int)
# # # #     p.add_argument("--grad_clip",       default=2.0,            type=float)
# # # #     p.add_argument("--grad_accum",      default=2,              type=int)
# # # #     p.add_argument("--patience",        default=50,             type=int)
# # # #     p.add_argument("--min_epochs",      default=80,             type=int)
# # # #     p.add_argument("--n_train_ens",     default=6,              type=int)
# # # #     p.add_argument("--use_amp",         action="store_true")
# # # #     p.add_argument("--num_workers",     default=2,              type=int)
# # # #     p.add_argument("--sigma_min",       default=0.05,           type=float)
# # # #     # FIX-V20-1: ensemble collapse params
# # # #     p.add_argument("--ctx_noise_scale",      default=0.05, type=float,
# # # #                    help="Gaussian noise injected into raw_ctx per ensemble member at inference")
# # # #     p.add_argument("--initial_sample_sigma", default=0.3,  type=float,
# # # #                    help="Initial noise std for ODE sampling (must >> sigma_min for spread)")
# # # #     p.add_argument("--ode_steps_train", default=20,             type=int)
# # # #     p.add_argument("--ode_steps_val",   default=30,             type=int)
# # # #     p.add_argument("--ode_steps_test",  default=50,             type=int)
# # # #     p.add_argument("--ode_steps",       default=None,           type=int)
# # # #     p.add_argument("--val_ensemble",    default=30,             type=int)
# # # #     p.add_argument("--fast_ensemble",   default=8,              type=int)
# # # #     p.add_argument("--fno_modes_h",     default=4,              type=int)
# # # #     p.add_argument("--fno_modes_t",     default=4,              type=int)
# # # #     p.add_argument("--fno_layers",      default=4,              type=int)
# # # #     p.add_argument("--fno_d_model",     default=32,             type=int)
# # # #     p.add_argument("--fno_spatial_down",default=32,             type=int)
# # # #     p.add_argument("--mamba_d_state",   default=16,             type=int)
# # # #     p.add_argument("--val_loss_freq",   default=2,              type=int)
# # # #     p.add_argument("--val_freq",        default=2,              type=int)
# # # #     p.add_argument("--full_eval_freq",  default=10,             type=int)
# # # #     p.add_argument("--val_subset_size", default=600,            type=int)
# # # #     p.add_argument("--output_dir",      default="runs/v20",     type=str)
# # # #     p.add_argument("--save_interval",   default=10,             type=int)
# # # #     p.add_argument("--metrics_csv",     default="metrics.csv",        type=str)
# # # #     p.add_argument("--predict_csv",     default="predictions.csv",    type=str)
# # # #     p.add_argument("--lstm_errors_npy",      default=None, type=str)
# # # #     p.add_argument("--diffusion_errors_npy", default=None, type=str)
# # # #     p.add_argument("--gpu_num",         default="0",            type=str)
# # # #     p.add_argument("--delim",           default=" ")
# # # #     p.add_argument("--skip",            default=1,              type=int)
# # # #     p.add_argument("--min_ped",         default=1,              type=int)
# # # #     p.add_argument("--threshold",       default=0.002,          type=float)
# # # #     p.add_argument("--other_modal",     default="gph")
# # # #     p.add_argument("--curriculum",      default=True,
# # # #                    type=lambda x: x.lower() != 'false')
# # # #     p.add_argument("--curriculum_start_len", default=4,         type=int)
# # # #     p.add_argument("--curriculum_end_epoch", default=40,        type=int)
# # # #     p.add_argument("--lon_flip_prob",   default=0.3,            type=float)
# # # #     p.add_argument("--pinn_warmup_epochs", default=30,          type=int)
# # # #     p.add_argument("--pinn_w_start",    default=0.01,           type=float)
# # # #     p.add_argument("--pinn_w_end",      default=0.1,            type=float)
# # # #     p.add_argument("--vel_warmup_epochs",  default=20,          type=int)
# # # #     p.add_argument("--vel_w_start",        default=0.5,         type=float)
# # # #     p.add_argument("--vel_w_end",          default=1.5,         type=float)
# # # #     p.add_argument("--recurv_warmup_epochs", default=10,        type=int)
# # # #     p.add_argument("--recurv_w_start",       default=0.3,       type=float)
# # # #     p.add_argument("--recurv_w_end",         default=1.0,       type=float)
# # # #     return p.parse_args()


# # # # def _resolve_ode_steps(args):
# # # #     if args.ode_steps is not None:
# # # #         return args.ode_steps, args.ode_steps, args.ode_steps
# # # #     return args.ode_steps_train, args.ode_steps_val, args.ode_steps_test


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
# # # #                            collate_fn, num_workers):
# # # #     n   = len(val_dataset)
# # # #     rng = random.Random(42)
# # # #     idx = rng.sample(range(n), min(subset_size, n))
# # # #     return DataLoader(Subset(val_dataset, idx),
# # # #                       batch_size=batch_size, shuffle=False,
# # # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # # def get_curriculum_len(epoch, args) -> int:
# # # #     if not args.curriculum:
# # # #         return args.pred_len
# # # #     if epoch >= args.curriculum_end_epoch:
# # # #         return args.pred_len
# # # #     frac = epoch / max(args.curriculum_end_epoch, 1)
# # # #     return int(args.curriculum_start_len
# # # #                + frac * (args.pred_len - args.curriculum_start_len))


# # # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
# # # #     """
# # # #     FIX-V20-3: Cũng tính ensemble spread để detect collapse sớm.
# # # #     """
# # # #     model.eval()
# # # #     acc = StepErrorAccumulator(pred_len)
# # # #     t0  = time.perf_counter()
# # # #     n   = 0
# # # #     spread_buf = []   # [km] spread tại step cuối (72h proxy)

# # # #     with torch.no_grad():
# # # #         for batch in loader:
# # # #             bl = move(list(batch), device)
# # # #             pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
# # # #                                               ddim_steps=ode_steps)
# # # #             acc.update(haversine_km_torch(denorm_torch(pred), denorm_torch(bl[1])))

# # # #             # Tính spread: std của ensemble tại step cuối, convert to km
# # # #             # all_trajs: [S, T, B, 2] in normalized coords
# # # #             last_step = all_trajs[:, -1, :, :]  # [S, B, 2]
# # # #             # std across ensemble (S dim), then haversine to km
# # # #             std_lon = last_step[:, :, 0].std(0)   # [B]
# # # #             std_lat = last_step[:, :, 1].std(0)   # [B]
# # # #             # approximate km: 1 normalized unit ≈ 500 km
# # # #             spread_km = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# # # #             spread_buf.append(spread_km)
# # # #             n += 1

# # # #     r = acc.compute()
# # # #     r["ms_per_batch"]  = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# # # #     r["spread_72h_km"] = float(np.mean(spread_buf)) if spread_buf else 0.0
# # # #     return r


# # # # def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
# # # #                   metrics_csv, tag="", predict_csv=""):
# # # #     model.eval()
# # # #     cliper_step_errors = []
# # # #     ev = TCEvaluator(pred_len=pred_len, compute_dtw=True)
# # # #     obs_seqs_01 = []; gt_seqs_01 = []; pred_seqs_01 = []; ens_seqs_01 = []

# # # #     with torch.no_grad():
# # # #         for batch in loader:
# # # #             bl  = move(list(batch), device)
# # # #             gt  = bl[1];  obs = bl[0]
# # # #             pred_mean, _, all_trajs = model.sample(
# # # #                 bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
# # # #                 predict_csv=predict_csv if predict_csv else None)

# # # #             pd_np = denorm_torch(pred_mean).cpu().numpy()
# # # #             gd_np = denorm_torch(gt).cpu().numpy()
# # # #             od_np = denorm_torch(obs).cpu().numpy()
# # # #             ed_np = denorm_torch(all_trajs).cpu().numpy()

# # # #             B = pd_np.shape[1]
# # # #             for b in range(B):
# # # #                 ens_b = ed_np[:, :, b, :]
# # # #                 ev.update(pd_np[:, b, :], gd_np[:, b, :], pred_ens=ens_b)
# # # #                 obs_seqs_01.append(od_np[:, b, :])
# # # #                 gt_seqs_01.append(gd_np[:, b, :])
# # # #                 pred_seqs_01.append(pd_np[:, b, :])
# # # #                 ens_seqs_01.append(ens_b)
# # # #                 obs_b = od_np[:, b, :]

# # # #                 cliper_errors_b = np.zeros(pred_len)
# # # #                 for h in range(pred_len):
# # # #                     # pred_cliper_01  = cliper_forecast(obs_b, h + 1)
# # # #                     # pred_cliper_deg = denorm_deg_np(pred_cliper_01[np.newaxis, :])
# # # #                     pred_cliper_norm = cliper_forecast(obs_b, h + 1)
# # # #                     pred_cliper_deg  = denorm_deg_np(pred_cliper_norm[np.newaxis])
# # # #                     gt_point        = gd_np[h, b, :][np.newaxis, :]
# # # #                     # gt_deg          = denorm_deg_np(gt_point)
# # # #                     gt_deg           = denorm_deg_np(gt_point[h:h+1])
# # # #                     cliper_errors_b[h] = float(haversine_km_np(pred_cliper_deg, gt_deg)[0])

# # # #                 cliper_step_errors.append(cliper_errors_b)

# # # #     if cliper_step_errors:
# # # #         cliper_mat = np.stack(cliper_step_errors)
# # # #         cliper_ugde_dict = {h: float(cliper_mat[:, s].mean())
# # # #                             for h, s in HORIZON_STEPS.items()
# # # #                             if s < cliper_mat.shape[1]}
# # # #         ev.cliper_ugde = cliper_ugde_dict
# # # #         print(f"  [CLIPER UGDE] 72h={cliper_ugde_dict.get(72, float('nan')):.1f} km")

# # # #     dm = ev.compute(tag=tag)

# # # #     try:
# # # #         if LANDFALL_TARGETS and ens_seqs_01:
# # # #             bss_vals = []
# # # #             step_72  = HORIZON_STEPS.get(72, pred_len - 1)
# # # #             for tname, t_lon, t_lat in LANDFALL_TARGETS:
# # # #                 bv = brier_skill_score(
# # # #                     [e.transpose(1,0,2) if e.ndim==3 else e for e in ens_seqs_01],
# # # #                     gt_seqs_01, min(step_72, pred_len-1),
# # # #                     (t_lon, t_lat), LANDFALL_RADIUS_KM)
# # # #                 if not math.isnan(bv):
# # # #                     bss_vals.append(bv)
# # # #             if bss_vals:
# # # #                 dm.bss_mean = float(np.mean(bss_vals))
# # # #                 print(f"  [BSS] mean={dm.bss_mean:.4f}")
# # # #     except Exception as e:
# # # #         print(f"  ⚠  BSS failed: {e}")

# # # #     save_metrics_csv(dm, metrics_csv, tag=tag)
# # # #     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# # # # class BestModelSaver:
# # # #     def __init__(self, patience=50, ade_tol=5.0):
# # # #         self.patience      = patience
# # # #         self.ade_tol       = ade_tol
# # # #         self.best_ade      = float("inf")
# # # #         self.best_val_loss = float("inf")
# # # #         self.counter_ade   = 0
# # # #         self.counter_loss  = 0
# # # #         self.early_stop    = False

# # # #     def reset_counters(self, reason=""):
# # # #         self.counter_ade  = 0
# # # #         self.counter_loss = 0
# # # #         if reason:
# # # #             print(f"  [SAVER] Patience counters reset: {reason}")

# # # #     def update_val_loss(self, val_loss, model, out_dir, epoch, optimizer, tl):
# # # #         if val_loss < self.best_val_loss - 1e-4:
# # # #             self.best_val_loss = val_loss;  self.counter_loss = 0
# # # #             torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
# # # #                             optimizer_state=optimizer.state_dict(),
# # # #                             train_loss=tl, val_loss=val_loss,
# # # #                             model_version="v20-valloss"),
# # # #                        os.path.join(out_dir, "best_model_valloss.pth"))
# # # #         else:
# # # #             self.counter_loss += 1

# # # #     def update_ade(self, ade, model, out_dir, epoch, optimizer, tl, vl,
# # # #                    min_epochs=80):
# # # #         if ade < self.best_ade - self.ade_tol:
# # # #             self.best_ade = ade;  self.counter_ade = 0
# # # #             torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
# # # #                             optimizer_state=optimizer.state_dict(),
# # # #                             train_loss=tl, val_loss=vl, val_ade_km=ade,
# # # #                             model_version="v20-FNO-Mamba-recurv"),
# # # #                        os.path.join(out_dir, "best_model.pth"))
# # # #             print(f"  ✅ Best ADE {ade:.1f} km  (epoch {epoch})")
# # # #         else:
# # # #             self.counter_ade += 1
# # # #             print(f"  No ADE improvement {self.counter_ade}/{self.patience}"
# # # #                   f"  (Δ={self.best_ade-ade:.1f} km < tol={self.ade_tol} km)"
# # # #                   f"  | Loss counter {self.counter_loss}/{self.patience}")

# # # #         if epoch >= min_epochs:
# # # #             if self.counter_ade >= self.patience and self.counter_loss >= self.patience:
# # # #                 self.early_stop = True
# # # #                 print(f"  ⛔ Early stop @ epoch {epoch}")
# # # #         else:
# # # #             if self.counter_ade >= self.patience and self.counter_loss >= self.patience:
# # # #                 print(f"  ⚠  Would stop but min_epochs={min_epochs} not reached. Continuing...")
# # # #                 self.counter_ade = 0;  self.counter_loss = 0


# # # # def _load_baseline_errors(path, name):
# # # #     if path is None:
# # # #         print(f"\n  ⚠  WARNING: --{name.lower().replace(' ','_')}_errors_npy not provided.")
# # # #         print(f"     Statistical comparison vs {name} will be SKIPPED.\n")
# # # #         return None
# # # #     if not os.path.exists(path):
# # # #         print(f"\n  ⚠  {path} not found — {name} skipped.\n")
# # # #         return None
# # # #     arr = np.load(path)
# # # #     print(f"  ✓  Loaded {name}: {arr.shape}")
# # # #     return arr


# # # # def main(args):
# # # #     if torch.cuda.is_available():
# # # #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
# # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # #     os.makedirs(args.output_dir, exist_ok=True)

# # # #     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
# # # #     predict_csv = os.path.join(args.output_dir, args.predict_csv)
# # # #     tables_dir  = os.path.join(args.output_dir, "tables")
# # # #     stat_dir    = os.path.join(tables_dir, "stat_tests")
# # # #     os.makedirs(tables_dir, exist_ok=True)
# # # #     os.makedirs(stat_dir,   exist_ok=True)

# # # #     ode_train, ode_val, ode_test = _resolve_ode_steps(args)

# # # #     print("=" * 68)
# # # #     print("  TC-FlowMatching v20  |  FNO3D + Mamba + OT-CFM + PINN")
# # # #     print("  v20 FIXES:")
# # # #     print("    FIX-V20-1: Ensemble collapse → ctx noise + larger initial σ")
# # # #     print("    FIX-V20-2: Early stop → ADE mỗi epoch (counter = real epochs)")
# # # #     print("    FIX-V20-3: Spread monitor để detect collapse")
# # # #     print("=" * 68)
# # # #     print(f"  device               : {device}")
# # # #     print(f"  sigma_min            : {args.sigma_min}")
# # # #     print(f"  ctx_noise_scale      : {args.ctx_noise_scale}")
# # # #     print(f"  initial_sample_sigma : {args.initial_sample_sigma}")
# # # #     print(f"  ode_steps            : train={ode_train}  val={ode_val}  test={ode_test}")
# # # #     print(f"  val_ensemble         : {args.val_ensemble}")
# # # #     print(f"  patience             : {args.patience} epochs  (min_epochs={args.min_epochs})")
# # # #     print()

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
# # # #     if test_loader:
# # # #         print(f"  test  : {len(test_loader.dataset)} seq")

# # # #     # FIX-V20-1: pass ensemble collapse params to model
# # # #     model = TCFlowMatching(
# # # #         pred_len             = args.pred_len,
# # # #         obs_len              = args.obs_len,
# # # #         sigma_min            = args.sigma_min,
# # # #         n_train_ens          = args.n_train_ens,
# # # #         ctx_noise_scale      = args.ctx_noise_scale,
# # # #         initial_sample_sigma = args.initial_sample_sigma,
# # # #     ).to(device)

# # # #     if (args.fno_spatial_down != 32 or args.fno_modes_h != 4
# # # #             or args.fno_layers != 4 or args.fno_d_model != 32):
# # # #         from Model.FNO3D_encoder import FNO3DEncoder
# # # #         model.net.spatial_enc = FNO3DEncoder(
# # # #             in_channel=13, out_channel=1,
# # # #             d_model=args.fno_d_model, n_layers=args.fno_layers,
# # # #             modes_t=args.fno_modes_t, modes_h=args.fno_modes_h,
# # # #             modes_w=args.fno_modes_h, spatial_down=args.fno_spatial_down,
# # # #             dropout=0.05).to(device)

# # # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # #     print(f"  params  : {n_params:,}")

# # # #     try:
# # # #         model = torch.compile(model, mode="reduce-overhead")
# # # #         print("  torch.compile: enabled")
# # # #     except Exception:
# # # #         pass

# # # #     optimizer = optim.AdamW(model.parameters(),
# # # #                              lr=args.g_learning_rate,
# # # #                              weight_decay=args.weight_decay)
# # # #     steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
# # # #     total_steps     = steps_per_epoch * args.num_epochs
# # # #     warmup          = steps_per_epoch * args.warmup_epochs
# # # #     scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
# # # #     saver  = BestModelSaver(patience=args.patience, ade_tol=5.0)
# # # #     scaler = GradScaler('cuda', enabled=args.use_amp)

# # # #     print("=" * 68)
# # # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch)")
# # # #     print("=" * 68)

# # # #     epoch_times   = []
# # # #     train_start   = time.perf_counter()
# # # #     last_val_loss = float("inf")
# # # #     _lr_ep30_done = False
# # # #     _lr_ep60_done = False
# # # #     _prev_ens     = 1

# # # #     import Model.losses as _losses_mod

# # # #     for epoch in range(args.num_epochs):
# # # #         # Progressive ensemble
# # # #         current_ens = 1 if epoch < 30 else (2 if epoch < 60 else args.n_train_ens)
# # # #         model.n_train_ens = current_ens
# # # #         effective_fast_ens = min(args.fast_ensemble, max(current_ens*2, args.fast_ensemble))

# # # #         if current_ens != _prev_ens:
# # # #             saver.reset_counters(
# # # #                 f"n_train_ens {_prev_ens}→{current_ens} at epoch {epoch}")
# # # #             _prev_ens = current_ens

# # # #         curr_len = get_curriculum_len(epoch, args)
# # # #         if hasattr(model, "set_curriculum_len"):
# # # #             model.set_curriculum_len(curr_len)

# # # #         epoch_weights = copy.copy(_BASE_WEIGHTS)
# # # #         epoch_weights["pinn"]     = get_pinn_weight(epoch, args.pinn_warmup_epochs,
# # # #                                                     args.pinn_w_start, args.pinn_w_end)
# # # #         epoch_weights["velocity"] = get_velocity_weight(epoch, args.vel_warmup_epochs,
# # # #                                                         args.vel_w_start, args.vel_w_end)
# # # #         epoch_weights["recurv"]   = get_recurv_weight(epoch, args.recurv_warmup_epochs,
# # # #                                                       args.recurv_w_start, args.recurv_w_end)
# # # #         _losses_mod.WEIGHTS.update(epoch_weights)
# # # #         if hasattr(model, 'weights'):
# # # #             model.weights = epoch_weights
# # # #         _losses_mod.WEIGHTS["pinn"]     = epoch_weights["pinn"]
# # # #         _losses_mod.WEIGHTS["velocity"] = epoch_weights["velocity"]
# # # #         _losses_mod.WEIGHTS["recurv"]   = epoch_weights["recurv"]

# # # #         current_clip = get_grad_clip(epoch, warmup_epochs=20,
# # # #                                      clip_start=args.grad_clip, clip_end=1.0)

# # # #         # LR restarts
# # # #         if epoch == 30 and not _lr_ep30_done:
# # # #             _lr_ep30_done = True
# # # #             warmup_steps = steps_per_epoch * 1
# # # #             scheduler = get_cosine_schedule_with_warmup(
# # # #                 optimizer, warmup_steps,
# # # #                 steps_per_epoch * (args.num_epochs - 30),
# # # #                 min_lr=5e-6)
# # # #             saver.reset_counters("LR warm restart at epoch 30")
# # # #             print(f"  ↺  Warm Restart LR at epoch 30")

# # # #         if epoch == 60 and not _lr_ep60_done:
# # # #             _lr_ep60_done = True
# # # #             warmup_steps = steps_per_epoch * 1
# # # #             scheduler = get_cosine_schedule_with_warmup(
# # # #                 optimizer, warmup_steps,
# # # #                 steps_per_epoch * (args.num_epochs - 60),
# # # #                 min_lr=1e-6)
# # # #             saver.reset_counters("LR warm restart at epoch 60")
# # # #             print(f"  ↺  Warm Restart LR at epoch 60")

# # # #         # ── Training loop ─────────────────────────────────────────────────────
# # # #         model.train()
# # # #         sum_loss = 0.0
# # # #         t0 = time.perf_counter()
# # # #         optimizer.zero_grad()
# # # #         recurv_ratio_buf = []

# # # #         for i, batch in enumerate(train_loader):
# # # #             bl = move(list(batch), device)

# # # #             if epoch == 0 and i == 0:
# # # #                 test_env = bl[13]
# # # #                 if test_env is not None and "gph500_mean" in test_env:
# # # #                     gph_val = test_env["gph500_mean"]
# # # #                     if torch.all(gph_val == 0):
# # # #                         print("\n" + "!"*60)
# # # #                         print("  ⚠️  GPH500 đang bị triệt tiêu về 0!")
# # # #                         print("!"*60 + "\n")
# # # #                     else:
# # # #                         print(f"  ✅ Data Check: GPH500 OK (Mean: {gph_val.mean().item():.4f})")

# # # #             with autocast(device_type='cuda', enabled=args.use_amp):
# # # #                 bd = model.get_loss_breakdown(bl)

# # # #             loss_to_backpass = bd["total"] / max(args.grad_accum, 1)
# # # #             scaler.scale(loss_to_backpass).backward()

# # # #             if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
# # # #                 scaler.unscale_(optimizer)
# # # #                 torch.nn.utils.clip_grad_norm_(model.parameters(), current_clip)
# # # #                 scaler.step(optimizer)
# # # #                 scaler.update()
# # # #                 scheduler.step()
# # # #                 optimizer.zero_grad()

# # # #             sum_loss += bd["total"].item()

# # # #             if "recurv_ratio" in bd:
# # # #                 recurv_ratio_buf.append(bd["recurv_ratio"])

# # # #             if i % 20 == 0:
# # # #                 lr  = optimizer.param_groups[0]["lr"]
# # # #                 rr  = bd.get("recurv_ratio", 0.0)
# # # #                 elapsed = time.perf_counter() - t0
# # # #                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # # #                       f"  loss={bd['total'].item():.3f}"
# # # #                       f"  fm={bd.get('fm',0):.2f}"
# # # #                       f"  vel={bd.get('velocity',0):.6f}"
# # # #                       f"  pinn={bd.get('pinn', 0):.6f}"
# # # #                       f"  recurv={bd.get('recurv',0):.3f}"
# # # #                       f"  rr={rr:.2f}"
# # # #                       f"  pinn_w={epoch_weights['pinn']:.3f}"
# # # #                       f"  clip={current_clip:.1f}"
# # # #                       f"  ens={current_ens}  len={curr_len}"
# # # #                       f"  lr={lr:.2e}  t={elapsed:.0f}s")

# # # #         ep_s  = time.perf_counter() - t0
# # # #         epoch_times.append(ep_s)
# # # #         avg_t = sum_loss / len(train_loader)
# # # #         mean_rr = float(np.mean(recurv_ratio_buf)) if recurv_ratio_buf else 0.0

# # # #         # ── Val loss (mỗi val_freq epoch) ─────────────────────────────────────
# # # #         if epoch % args.val_freq == 0:
# # # #             model.eval()
# # # #             val_loss = 0.0
# # # #             t_val = time.perf_counter()
# # # #             with torch.no_grad():
# # # #                 for batch in val_loader:
# # # #                     bl_v = move(list(batch), device)
# # # #                     with autocast(device_type='cuda', enabled=args.use_amp):
# # # #                         val_loss += model.get_loss(bl_v).item()
# # # #             last_val_loss = val_loss / len(val_loader)
# # # #             t_val_s = time.perf_counter() - t_val
# # # #             saver.update_val_loss(last_val_loss, model, args.output_dir,
# # # #                                   epoch, optimizer, avg_t)
# # # #             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
# # # #                   f"  rr={mean_rr:.2f}"
# # # #                   f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
# # # #                   f"  ens={current_ens}  len={curr_len}"
# # # #                   f"  recurv_w={epoch_weights['recurv']:.2f}")
# # # #         else:
# # # #             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
# # # #                   f"  val={last_val_loss:.3f}(cached)"
# # # #                   f"  rr={mean_rr:.2f}  t={ep_s:.0f}s")

# # # #         # ── ADE evaluation MỖI EPOCH (FIX-V19-1 / V20-3) ─────────────────────
# # # #         t_ade = time.perf_counter()
# # # #         m = evaluate_fast(model, val_subset_loader, device,
# # # #                           ode_train, args.pred_len, effective_fast_ens)
# # # #         t_ade_s = time.perf_counter() - t_ade

# # # #         spread_72h = m.get("spread_72h_km", 0.0)
# # # #         collapse_warn = "  ⚠️ COLLAPSE!" if spread_72h < 10.0 else ""

# # # #         print(f"  [ADE ep{epoch} {t_ade_s:.0f}s]"
# # # #               f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km"
# # # #               f"  12h={m.get('12h',0):.0f}  24h={m.get('24h',0):.0f}"
# # # #               f"  72h={m.get('72h',0):.0f} km"
# # # #               f"  spread={spread_72h:.1f} km"
# # # #               f"  (ens={effective_fast_ens}, steps={ode_train})"
# # # #               f"  counter={saver.counter_ade}/{args.patience}"
# # # #               f"{collapse_warn}")

# # # #         saver.update_ade(m["ADE"], model, args.output_dir, epoch,
# # # #                          optimizer, avg_t, last_val_loss,
# # # #                          min_epochs=args.min_epochs)

# # # #         # ── Full eval (mỗi full_eval_freq epoch) ──────────────────────────────
# # # #         if epoch % args.full_eval_freq == 0 and epoch > 0:
# # # #             print(f"  [Full eval epoch {epoch}, ode_steps={ode_val}]")
# # # #             try:
# # # #                 dm, _, _, _ = evaluate_full(
# # # #                     model, val_loader, device,
# # # #                     ode_val, args.pred_len, args.val_ensemble,
# # # #                     metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}")
# # # #                 print(dm.summary())
# # # #             except Exception as e:
# # # #                 print(f"  ⚠  full_eval failed at epoch {epoch}: {e}")
# # # #                 import traceback; traceback.print_exc()

# # # #         if (epoch+1) % args.save_interval == 0:
# # # #             torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
# # # #                        os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))

# # # #         if saver.early_stop:
# # # #             print(f"  Early stopping @ epoch {epoch}")
# # # #             break

# # # #         if epoch % 5 == 4:
# # # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
# # # #                   f"  (avg {avg_ep:.0f}s/epoch)")

# # # #     _losses_mod.WEIGHTS["pinn"]     = args.pinn_w_end
# # # #     _losses_mod.WEIGHTS["velocity"] = args.vel_w_end
# # # #     _losses_mod.WEIGHTS["recurv"]   = args.recurv_w_end

# # # #     total_train_h = (time.perf_counter() - train_start) / 3600

# # # #     # ── Final test eval ───────────────────────────────────────────────────────
# # # #     print(f"\n{'='*68}  FINAL TEST (ode_steps={ode_test})")
# # # #     all_results = []

# # # #     if test_loader:
# # # #         best_path = os.path.join(args.output_dir, "best_model.pth")
# # # #         if not os.path.exists(best_path):
# # # #             best_path = os.path.join(args.output_dir, "best_model_valloss.pth")
# # # #         if os.path.exists(best_path):
# # # #             ck = torch.load(best_path, map_location=device)
# # # #             try:
# # # #                 model.load_state_dict(ck["model_state_dict"])
# # # #             except Exception:
# # # #                 model.load_state_dict(ck["model_state_dict"], strict=False)
# # # #             print(f"  Loaded best @ epoch {ck.get('epoch','?')}"
# # # #                   f"  ADE={ck.get('val_ade_km','?')}")

# # # #         final_ens = max(args.val_ensemble, 50)
# # # #         dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
# # # #             model, test_loader, device,
# # # #             ode_test, args.pred_len, final_ens,
# # # #             metrics_csv=metrics_csv, tag="test_final",
# # # #             predict_csv=predict_csv)
# # # #         print(dm_test.summary())

# # # #         all_results.append(ModelResult(
# # # #             model_name   = "FM+PINN-v20",
# # # #             split        = "test",
# # # #             ADE          = dm_test.ade,
# # # #             FDE          = dm_test.fde,
# # # #             ADE_str      = dm_test.ade_str,
# # # #             ADE_rec      = dm_test.ade_rec,
# # # #             delta_rec    = dm_test.pr,
# # # #             CRPS_mean    = dm_test.crps_mean,
# # # #             CRPS_72h     = dm_test.crps_72h,
# # # #             SSR          = dm_test.ssr_mean,
# # # #             TSS_72h      = dm_test.tss_72h,
# # # #             OYR          = dm_test.oyr_mean,
# # # #             DTW          = dm_test.dtw_mean,
# # # #             ATE_abs      = dm_test.ate_abs_mean,
# # # #             CTE_abs      = dm_test.cte_abs_mean,
# # # #             n_total      = dm_test.n_total,
# # # #             n_recurv     = dm_test.n_rec,
# # # #             train_time_h = total_train_h,
# # # #             params_M     = sum(p.numel() for p in model.parameters()) / 1e6,
# # # #         ))

# # # #         _, cliper_errs = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
# # # #         persist_errs   = persistence_errors(obs_seqs, gt_seqs, args.pred_len)
# # # #         fmpinn_per_seq = np.array([seq_ade_km(np.array(pp), np.array(g))
# # # #                                     for pp, g in zip(pred_seqs, gt_seqs)])

# # # #         np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
# # # #         np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
# # # #         np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))

# # # #         lstm_per_seq      = _load_baseline_errors(args.lstm_errors_npy,      "LSTM")
# # # #         diffusion_per_seq = _load_baseline_errors(args.diffusion_errors_npy, "Diffusion")
# # # #         if lstm_per_seq is not None:
# # # #             np.save(os.path.join(stat_dir, "lstm.npy"), lstm_per_seq)
# # # #         if diffusion_per_seq is not None:
# # # #             np.save(os.path.join(stat_dir, "diffusion.npy"), diffusion_per_seq)

# # # #         _dummy = np.array([float("nan")])
# # # #         run_all_tests(
# # # #             fmpinn_ade    = fmpinn_per_seq,
# # # #             cliper_ade    = cliper_errs.mean(1),
# # # #             lstm_ade      = lstm_per_seq if lstm_per_seq is not None else _dummy,
# # # #             diffusion_ade = diffusion_per_seq if diffusion_per_seq is not None else _dummy,
# # # #             persist_ade   = persist_errs.mean(1),
# # # #             out_dir       = stat_dir)

# # # #         all_results += [
# # # #             ModelResult("CLIPER", "test",
# # # #                         ADE=float(cliper_errs.mean()),
# # # #                         FDE=float(cliper_errs[:, -1].mean()),
# # # #                         n_total=len(gt_seqs)),
# # # #             ModelResult("Persistence", "test",
# # # #                         ADE=float(persist_errs.mean()),
# # # #                         FDE=float(persist_errs[:, -1].mean()),
# # # #                         n_total=len(gt_seqs)),
# # # #         ]

# # # #         stat_rows = [
# # # #             paired_tests(fmpinn_per_seq, cliper_errs.mean(1),  "FM+PINN vs CLIPER",      5),
# # # #             paired_tests(fmpinn_per_seq, persist_errs.mean(1), "FM+PINN vs Persistence", 5),
# # # #         ]
# # # #         if lstm_per_seq is not None:
# # # #             stat_rows.append(paired_tests(fmpinn_per_seq, lstm_per_seq, "FM+PINN vs LSTM", 5))
# # # #         if diffusion_per_seq is not None:
# # # #             stat_rows.append(paired_tests(fmpinn_per_seq, diffusion_per_seq, "FM+PINN vs Diffusion", 5))

# # # #         compute_rows = DEFAULT_COMPUTE
# # # #         try:
# # # #             sb = next(iter(test_loader))
# # # #             sb = move(list(sb), device)
# # # #             from utils.evaluation_tables import profile_model_components
# # # #             compute_rows = profile_model_components(model, sb, device)
# # # #         except Exception as e:
# # # #             print(f"  Profiling skipped: {e}")

# # # #         export_all_tables(
# # # #             results=all_results, ablation_rows=DEFAULT_ABLATION,
# # # #             stat_rows=stat_rows, pinn_sens_rows=DEFAULT_PINN_SENSITIVITY,
# # # #             compute_rows=compute_rows, out_dir=tables_dir)

# # # #         with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
# # # #             fh.write(dm_test.summary())
# # # #             fh.write(f"\n\nmodel_version         : FM+PINN v20\n")
# # # #             fh.write(f"sigma_min             : {args.sigma_min}\n")
# # # #             fh.write(f"ctx_noise_scale       : {args.ctx_noise_scale}\n")
# # # #             fh.write(f"initial_sample_sigma  : {args.initial_sample_sigma}\n")
# # # #             fh.write(f"ode_steps_test        : {ode_test}\n")
# # # #             fh.write(f"eval_ensemble         : {final_ens}\n")
# # # #             fh.write(f"train_time_h          : {total_train_h:.2f}\n")
# # # #             fh.write(f"n_params_M            : "
# # # #                      f"{sum(p.numel() for p in model.parameters())/1e6:.2f}\n")

# # # #     avg_ep = sum(epoch_times)/len(epoch_times) if epoch_times else 0
# # # #     print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
# # # #     print(f"  Best val loss  : {saver.best_val_loss:.4f}")
# # # #     print(f"  Avg epoch time : {avg_ep:.0f}s")
# # # #     print(f"  Total training : {total_train_h:.2f}h")
# # # #     print(f"  Tables dir     : {tables_dir}")
# # # #     print("=" * 68)


# # # # if __name__ == "__main__":
# # # #     args = get_args()
# # # #     np.random.seed(42);  torch.manual_seed(42)
# # # #     if torch.cuda.is_available():
# # # #         torch.cuda.manual_seed_all(42)
# # # #     main(args)

# # # """
# # # scripts/train_flowmatching.py  ── v21
# # # ======================================
# # # FIXES vs v20:

# # # FIX-V21-1  [CRASH] evaluate_fast(): pred_mean từ model.sample() có T=active_pred_len
# # #            (ví dụ T=4 khi curriculum), nhưng bl[1] (gt) luôn có T=pred_len=12.
# # #            haversine_km_torch(pred[4,B,2], gt[12,B,2]) → RuntimeError dim 0: 4 vs 12.
# # #            Fix: slice gt xuống active_pred_len trước khi tính distance.
# # #            Đây là bug gây crash được báo trong traceback.

# # # FIX-V21-2  [CRASH/WRONG] evaluate_full() cliper loop: gt_point được tạo từ
# # #            gd_np[h, b, :][np.newaxis, :] → shape [1, 2] (0.1-degree units).
# # #            Sau đó gt_deg = denorm_deg_np(gt_point[h:h+1]):
# # #              - Lỗi 1: gt_point[h:h+1] khi h>0 → EMPTY array (gt_point chỉ có 1 row).
# # #              - Lỗi 2: gd_np đã là 0.1-degree (sau denorm_torch), không phải normalized.
# # #                Nếu truyền 0.1-deg vào denorm_deg_np sẽ ra giá trị vô nghĩa (~5800 deg).
# # #            Fix: dùng haversine_km_np trực tiếp với 0.1-deg values (unit_01deg=True),
# # #            và dùng denorm_np (→ 0.1-deg) thay vì denorm_deg_np cho cliper prediction.

# # # FIX-V21-3  [WRONG BSS] evaluate_full(): ens_seqs_01 lưu ens_b với shape [S, T, 2].
# # #            Nhưng khi truyền vào brier_skill_score, code làm:
# # #              e.transpose(1,0,2) if e.ndim==3 else e
# # #            → [T, S, 2] — đảo chiều S và T, gây tính BSS sai hoàn toàn.
# # #            brier_skill_score() dùng ens_seqs[s, step] = lấy từ dim-0 (S) và dim-1 (T),
# # #            nên cần giữ nguyên [S, T, 2].
# # #            Fix: bỏ transpose, truyền ens_b trực tiếp (đã đúng [S, T, 2]).

# # # Kept from v20:
# # #     FIX-V20-1..3 (ensemble collapse, early stop, spread monitor)
# # #     FIX-V19-1..2
# # #     FIX-V18-1..4
# # # """
# # # from __future__ import annotations

# # # import sys
# # # import os
# # # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # # import argparse
# # # import time
# # # import math
# # # import random
# # # import copy

# # # import numpy as np
# # # import torch
# # # import torch.optim as optim
# # # from torch.amp import autocast, GradScaler
# # # from torch.utils.data import DataLoader, Subset

# # # from Model.data.loader_training import data_loader
# # # from Model.flow_matching_model import TCFlowMatching
# # # from Model.utils import get_cosine_schedule_with_warmup
# # # from Model.losses import WEIGHTS as _BASE_WEIGHTS
# # # from utils.metrics import (
# # #     TCEvaluator, StepErrorAccumulator,
# # #     save_metrics_csv, haversine_km_torch, haversine_km,
# # #     denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
# # #     cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
# # #     brier_skill_score,
# # # )
# # # from utils.evaluation_tables import (
# # #     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
# # #     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
# # #     DEFAULT_COMPUTE, paired_tests, persistence_errors, cliper_errors,
# # # )
# # # from scripts.statistical_tests import run_all_tests


# # # # ── Helpers ───────────────────────────────────────────────────────────────────

# # # def haversine_km_np_local(pred_deg: np.ndarray, gt_deg: np.ndarray) -> np.ndarray:
# # #     pred_deg = np.atleast_2d(pred_deg)
# # #     gt_deg   = np.atleast_2d(gt_deg)
# # #     R = 6371.0
# # #     lon1, lat1 = np.radians(pred_deg[:, 0]), np.radians(pred_deg[:, 1])
# # #     lon2, lat2 = np.radians(gt_deg[:, 0]),   np.radians(gt_deg[:, 1])
# # #     dlon = lon2 - lon1;  dlat = lat2 - lat1
# # #     a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
# # #     return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# # # def seq_ade_km(pred_norm: np.ndarray, gt_norm: np.ndarray) -> float:
# # #     return float(haversine_km_np_local(denorm_deg_np(pred_norm),
# # #                                        denorm_deg_np(gt_norm)).mean())


# # # # ── Adaptive weight schedules ─────────────────────────────────────────────────

# # # def get_pinn_weight(epoch, warmup_epochs=30, w_start=0.01, w_end=0.1):
# # #     if epoch >= warmup_epochs:
# # #         return w_end
# # #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# # # def get_velocity_weight(epoch, warmup_epochs=20, w_start=0.5, w_end=1.5):
# # #     if epoch >= warmup_epochs:
# # #         return w_end
# # #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# # # def get_recurv_weight(epoch, warmup_epochs=10, w_start=0.3, w_end=1.0):
# # #     if epoch >= warmup_epochs:
# # #         return w_end
# # #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# # # def get_grad_clip(epoch, warmup_epochs=20, clip_start=2.0, clip_end=1.0):
# # #     if epoch >= warmup_epochs:
# # #         return clip_end
# # #     return clip_start - (epoch / max(warmup_epochs-1, 1)) * (clip_start - clip_end)


# # # def get_args():
# # #     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # #     p.add_argument("--dataset_root",    default="TCND_vn",      type=str)
# # #     p.add_argument("--obs_len",         default=8,              type=int)
# # #     p.add_argument("--pred_len",        default=12,             type=int)
# # #     p.add_argument("--test_year",       default=None,           type=int)
# # #     p.add_argument("--batch_size",      default=32,             type=int)
# # #     p.add_argument("--num_epochs",      default=200,            type=int)
# # #     p.add_argument("--g_learning_rate", default=2e-4,           type=float)
# # #     p.add_argument("--weight_decay",    default=1e-4,           type=float)
# # #     p.add_argument("--warmup_epochs",   default=3,              type=int)
# # #     p.add_argument("--grad_clip",       default=2.0,            type=float)
# # #     p.add_argument("--grad_accum",      default=2,              type=int)
# # #     p.add_argument("--patience",        default=50,             type=int)
# # #     p.add_argument("--min_epochs",      default=80,             type=int)
# # #     p.add_argument("--n_train_ens",     default=6,              type=int)
# # #     p.add_argument("--use_amp",         action="store_true")
# # #     p.add_argument("--num_workers",     default=2,              type=int)
# # #     p.add_argument("--sigma_min",       default=0.05,           type=float)
# # #     p.add_argument("--ctx_noise_scale",      default=0.05, type=float,
# # #                    help="Gaussian noise injected into raw_ctx per ensemble member at inference")
# # #     p.add_argument("--initial_sample_sigma", default=0.3,  type=float,
# # #                    help="Initial noise std for ODE sampling (must >> sigma_min for spread)")
# # #     p.add_argument("--ode_steps_train", default=20,             type=int)
# # #     p.add_argument("--ode_steps_val",   default=30,             type=int)
# # #     p.add_argument("--ode_steps_test",  default=50,             type=int)
# # #     p.add_argument("--ode_steps",       default=None,           type=int)
# # #     p.add_argument("--val_ensemble",    default=30,             type=int)
# # #     p.add_argument("--fast_ensemble",   default=8,              type=int)
# # #     p.add_argument("--fno_modes_h",     default=4,              type=int)
# # #     p.add_argument("--fno_modes_t",     default=4,              type=int)
# # #     p.add_argument("--fno_layers",      default=4,              type=int)
# # #     p.add_argument("--fno_d_model",     default=32,             type=int)
# # #     p.add_argument("--fno_spatial_down",default=32,             type=int)
# # #     p.add_argument("--mamba_d_state",   default=16,             type=int)
# # #     p.add_argument("--val_loss_freq",   default=2,              type=int)
# # #     p.add_argument("--val_freq",        default=2,              type=int)
# # #     p.add_argument("--full_eval_freq",  default=10,             type=int)
# # #     p.add_argument("--val_subset_size", default=600,            type=int)
# # #     p.add_argument("--output_dir",      default="runs/v21",     type=str)
# # #     p.add_argument("--save_interval",   default=10,             type=int)
# # #     p.add_argument("--metrics_csv",     default="metrics.csv",        type=str)
# # #     p.add_argument("--predict_csv",     default="predictions.csv",    type=str)
# # #     p.add_argument("--lstm_errors_npy",      default=None, type=str)
# # #     p.add_argument("--diffusion_errors_npy", default=None, type=str)
# # #     p.add_argument("--gpu_num",         default="0",            type=str)
# # #     p.add_argument("--delim",           default=" ")
# # #     p.add_argument("--skip",            default=1,              type=int)
# # #     p.add_argument("--min_ped",         default=1,              type=int)
# # #     p.add_argument("--threshold",       default=0.002,          type=float)
# # #     p.add_argument("--other_modal",     default="gph")
# # #     p.add_argument("--curriculum",      default=True,
# # #                    type=lambda x: x.lower() != 'false')
# # #     p.add_argument("--curriculum_start_len", default=4,         type=int)
# # #     p.add_argument("--curriculum_end_epoch", default=40,        type=int)
# # #     p.add_argument("--lon_flip_prob",   default=0.3,            type=float)
# # #     p.add_argument("--pinn_warmup_epochs", default=30,          type=int)
# # #     p.add_argument("--pinn_w_start",    default=0.01,           type=float)
# # #     p.add_argument("--pinn_w_end",      default=0.1,            type=float)
# # #     p.add_argument("--vel_warmup_epochs",  default=20,          type=int)
# # #     p.add_argument("--vel_w_start",        default=0.5,         type=float)
# # #     p.add_argument("--vel_w_end",          default=1.5,         type=float)
# # #     p.add_argument("--recurv_warmup_epochs", default=10,        type=int)
# # #     p.add_argument("--recurv_w_start",       default=0.3,       type=float)
# # #     p.add_argument("--recurv_w_end",         default=1.0,       type=float)
# # #     return p.parse_args()


# # # def _resolve_ode_steps(args):
# # #     if args.ode_steps is not None:
# # #         return args.ode_steps, args.ode_steps, args.ode_steps
# # #     return args.ode_steps_train, args.ode_steps_val, args.ode_steps_test


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
# # #                            collate_fn, num_workers):
# # #     n   = len(val_dataset)
# # #     rng = random.Random(42)
# # #     idx = rng.sample(range(n), min(subset_size, n))
# # #     return DataLoader(Subset(val_dataset, idx),
# # #                       batch_size=batch_size, shuffle=False,
# # #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # # def get_curriculum_len(epoch, args) -> int:
# # #     if not args.curriculum:
# # #         return args.pred_len
# # #     if epoch >= args.curriculum_end_epoch:
# # #         return args.pred_len
# # #     frac = epoch / max(args.curriculum_end_epoch, 1)
# # #     return int(args.curriculum_start_len
# # #                + frac * (args.pred_len - args.curriculum_start_len))


# # # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
# # #     """
# # #     FIX-V21-1: Khi curriculum đang chạy, model.sample() trả về pred_mean với
# # #     T=active_pred_len (ví dụ 4), nhưng bl[1] (gt) luôn có T=pred_len=12.
# # #     Phải slice gt xuống active_pred_len để shapes khớp nhau trước khi
# # #     truyền vào haversine_km_torch.

# # #     FIX-V20-3: Tính ensemble spread để detect collapse sớm.
# # #     """
# # #     model.eval()
# # #     acc = StepErrorAccumulator(pred_len)
# # #     t0  = time.perf_counter()
# # #     n   = 0
# # #     spread_buf = []

# # #     with torch.no_grad():
# # #         for batch in loader:
# # #             bl = move(list(batch), device)
# # #             pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
# # #                                               ddim_steps=ode_steps)
# # #             # FIX-V21-1: pred.shape = [T_active, B, 2], bl[1].shape = [T_pred, B, 2]
# # #             # T_active <= T_pred (curriculum). Slice gt to match pred length.
# # #             T_active = pred.shape[0]
# # #             gt_sliced = bl[1][:T_active]  # [T_active, B, 2]
# # #             acc.update(haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced)))

# # #             # Spread: std của ensemble tại step cuối
# # #             last_step = all_trajs[:, -1, :, :]  # [S, B, 2]
# # #             std_lon = last_step[:, :, 0].std(0)   # [B]
# # #             std_lat = last_step[:, :, 1].std(0)   # [B]
# # #             spread_km = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# # #             spread_buf.append(spread_km)
# # #             n += 1

# # #     r = acc.compute()
# # #     r["ms_per_batch"]  = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# # #     r["spread_72h_km"] = float(np.mean(spread_buf)) if spread_buf else 0.0
# # #     return r


# # # def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
# # #                   metrics_csv, tag="", predict_csv=""):
# # #     """
# # #     FIX-V21-2: Sửa tính sai CLIPER error trong vòng lặp cliper.
# # #       - Bug cũ: gt_point = gd_np[h,b,:][np.newaxis,:] rồi gt_deg = denorm_deg_np(gt_point[h:h+1])
# # #         * gd_np đã là 0.1-degree (sau denorm_torch) -- không phải normalized coords
# # #         * gt_point[h:h+1] khi h>0 → empty array vì gt_point chỉ có 1 hàng
# # #       - Fix: so sánh trực tiếp ở 0.1-degree space với haversine_km_np(..., unit_01deg=True)
# # #         * dùng denorm_np() (→ 0.1-deg) cho cliper prediction
# # #         * gt đã là 0.1-deg từ gd_np, dùng trực tiếp

# # #     FIX-V21-3: Sửa shape của ens_seqs_01 khi truyền vào brier_skill_score.
# # #       - Bug cũ: [S,T,2].transpose(1,0,2) → [T,S,2] — đảo chiều S,T sai
# # #       - brier_skill_score truy cập ens_seqs[s, step] nên cần [S, T, 2]
# # #       - Fix: truyền trực tiếp ens_b [S,T,2] không transpose
# # #     """
# # #     model.eval()
# # #     cliper_step_errors = []
# # #     ev = TCEvaluator(pred_len=pred_len, compute_dtw=True)
# # #     obs_seqs_01 = []; gt_seqs_01 = []; pred_seqs_01 = []; ens_seqs_01 = []

# # #     with torch.no_grad():
# # #         for batch in loader:
# # #             bl  = move(list(batch), device)
# # #             gt  = bl[1];  obs = bl[0]
# # #             pred_mean, _, all_trajs = model.sample(
# # #                 bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
# # #                 predict_csv=predict_csv if predict_csv else None)

# # #             pd_np = denorm_torch(pred_mean).cpu().numpy()   # [T, B, 2] in 0.1-deg
# # #             gd_np = denorm_torch(gt).cpu().numpy()          # [T, B, 2] in 0.1-deg
# # #             od_np = denorm_torch(obs).cpu().numpy()         # [T_obs, B, 2] in 0.1-deg
# # #             ed_np = denorm_torch(all_trajs).cpu().numpy()   # [S, T, B, 2] in 0.1-deg

# # #             B = pd_np.shape[1]
# # #             for b in range(B):
# # #                 # ens_b: [S, T, 2] — shape đúng cho brier_skill_score
# # #                 ens_b = ed_np[:, :, b, :]
# # #                 ev.update(pd_np[:, b, :], gd_np[:, b, :], pred_ens=ens_b)
# # #                 obs_seqs_01.append(od_np[:, b, :])
# # #                 gt_seqs_01.append(gd_np[:, b, :])
# # #                 pred_seqs_01.append(pd_np[:, b, :])
# # #                 # FIX-V21-3: giữ nguyên [S, T, 2], KHÔNG transpose
# # #                 ens_seqs_01.append(ens_b)

# # #                 # ── CLIPER error per step ─────────────────────────────────────
# # #                 # FIX-V21-2: obs_b là normalized coords từ od_np (0.1-deg), cần
# # #                 # truyền normalized obs vào cliper_forecast, không phải 0.1-deg.
# # #                 # Lấy normalized obs từ bl[0] (obs_traj tensor trực tiếp).
# # #                 obs_b_norm = obs.cpu().numpy()[:, b, :]  # [T_obs, 2] normalized

# # #                 cliper_errors_b = np.zeros(pred_len)
# # #                 for h in range(pred_len):
# # #                     # cliper_forecast nhận normalized coords, trả về normalized [2]
# # #                     pred_cliper_norm = cliper_forecast(obs_b_norm, h + 1)

# # #                     # Chuyển sang 0.1-degree để so sánh với gd_np
# # #                     pred_cliper_01 = denorm_np(pred_cliper_norm[np.newaxis])  # [1, 2] 0.1-deg

# # #                     # gt tại step h: gd_np[h, b, :] đã là 0.1-deg
# # #                     gt_01 = gd_np[h, b, :][np.newaxis]  # [1, 2] 0.1-deg

# # #                     # So sánh ở 0.1-deg space
# # #                     cliper_errors_b[h] = float(
# # #                         haversine_km_np(pred_cliper_01, gt_01, unit_01deg=True)[0]
# # #                     )

# # #                 cliper_step_errors.append(cliper_errors_b)

# # #     if cliper_step_errors:
# # #         cliper_mat = np.stack(cliper_step_errors)
# # #         cliper_ugde_dict = {h: float(cliper_mat[:, s].mean())
# # #                             for h, s in HORIZON_STEPS.items()
# # #                             if s < cliper_mat.shape[1]}
# # #         ev.cliper_ugde = cliper_ugde_dict
# # #         print(f"  [CLIPER UGDE] 72h={cliper_ugde_dict.get(72, float('nan')):.1f} km")

# # #     dm = ev.compute(tag=tag)

# # #     try:
# # #         if LANDFALL_TARGETS and ens_seqs_01:
# # #             bss_vals = []
# # #             step_72  = HORIZON_STEPS.get(72, pred_len - 1)
# # #             for tname, t_lon, t_lat in LANDFALL_TARGETS:
# # #                 # FIX-V21-3: ens_seqs_01 entries are [S, T, 2] — đúng format
# # #                 # brier_skill_score expects list of [S, T, 2] in 0.1-deg units
# # #                 bv = brier_skill_score(
# # #                     ens_seqs_01,   # list of [S, T, 2], NO transpose needed
# # #                     gt_seqs_01, min(step_72, pred_len-1),
# # #                     (t_lon * 10.0, t_lat * 10.0),  # convert degrees to 0.1-deg for consistency
# # #                     LANDFALL_RADIUS_KM)
# # #                 if not math.isnan(bv):
# # #                     bss_vals.append(bv)
# # #             if bss_vals:
# # #                 dm.bss_mean = float(np.mean(bss_vals))
# # #                 print(f"  [BSS] mean={dm.bss_mean:.4f}")
# # #     except Exception as e:
# # #         print(f"  ⚠  BSS failed: {e}")

# # #     save_metrics_csv(dm, metrics_csv, tag=tag)
# # #     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# # # class BestModelSaver:
# # #     def __init__(self, patience=50, ade_tol=5.0):
# # #         self.patience      = patience
# # #         self.ade_tol       = ade_tol
# # #         self.best_ade      = float("inf")
# # #         self.best_val_loss = float("inf")
# # #         self.counter_ade   = 0
# # #         self.counter_loss  = 0
# # #         self.early_stop    = False

# # #     def reset_counters(self, reason=""):
# # #         self.counter_ade  = 0
# # #         self.counter_loss = 0
# # #         if reason:
# # #             print(f"  [SAVER] Patience counters reset: {reason}")

# # #     def update_val_loss(self, val_loss, model, out_dir, epoch, optimizer, tl):
# # #         if val_loss < self.best_val_loss - 1e-4:
# # #             self.best_val_loss = val_loss;  self.counter_loss = 0
# # #             torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
# # #                             optimizer_state=optimizer.state_dict(),
# # #                             train_loss=tl, val_loss=val_loss,
# # #                             model_version="v21-valloss"),
# # #                        os.path.join(out_dir, "best_model_valloss.pth"))
# # #         else:
# # #             self.counter_loss += 1

# # #     def update_ade(self, ade, model, out_dir, epoch, optimizer, tl, vl,
# # #                    min_epochs=80):
# # #         if ade < self.best_ade - self.ade_tol:
# # #             self.best_ade = ade;  self.counter_ade = 0
# # #             torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
# # #                             optimizer_state=optimizer.state_dict(),
# # #                             train_loss=tl, val_loss=vl, val_ade_km=ade,
# # #                             model_version="v21-FNO-Mamba-recurv"),
# # #                        os.path.join(out_dir, "best_model.pth"))
# # #             print(f"  ✅ Best ADE {ade:.1f} km  (epoch {epoch})")
# # #         else:
# # #             self.counter_ade += 1
# # #             print(f"  No ADE improvement {self.counter_ade}/{self.patience}"
# # #                   f"  (Δ={self.best_ade-ade:.1f} km < tol={self.ade_tol} km)"
# # #                   f"  | Loss counter {self.counter_loss}/{self.patience}")

# # #         if epoch >= min_epochs:
# # #             if self.counter_ade >= self.patience and self.counter_loss >= self.patience:
# # #                 self.early_stop = True
# # #                 print(f"  ⛔ Early stop @ epoch {epoch}")
# # #         else:
# # #             if self.counter_ade >= self.patience and self.counter_loss >= self.patience:
# # #                 print(f"  ⚠  Would stop but min_epochs={min_epochs} not reached. Continuing...")
# # #                 self.counter_ade = 0;  self.counter_loss = 0


# # # def _load_baseline_errors(path, name):
# # #     if path is None:
# # #         print(f"\n  ⚠  WARNING: --{name.lower().replace(' ','_')}_errors_npy not provided.")
# # #         print(f"     Statistical comparison vs {name} will be SKIPPED.\n")
# # #         return None
# # #     if not os.path.exists(path):
# # #         print(f"\n  ⚠  {path} not found — {name} skipped.\n")
# # #         return None
# # #     arr = np.load(path)
# # #     print(f"  ✓  Loaded {name}: {arr.shape}")
# # #     return arr


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

# # #     ode_train, ode_val, ode_test = _resolve_ode_steps(args)

# # #     print("=" * 68)
# # #     print("  TC-FlowMatching v21  |  FNO3D + Mamba + OT-CFM + PINN")
# # #     print("  v21 FIXES:")
# # #     print("    FIX-V21-1: evaluate_fast crash → slice gt to active_pred_len")
# # #     print("    FIX-V21-2: CLIPER errors sai đơn vị và index out of bounds")
# # #     print("    FIX-V21-3: BSS sai shape → bỏ transpose ens_seqs_01")
# # #     print("=" * 68)
# # #     print(f"  device               : {device}")
# # #     print(f"  sigma_min            : {args.sigma_min}")
# # #     print(f"  ctx_noise_scale      : {args.ctx_noise_scale}")
# # #     print(f"  initial_sample_sigma : {args.initial_sample_sigma}")
# # #     print(f"  ode_steps            : train={ode_train}  val={ode_val}  test={ode_test}")
# # #     print(f"  val_ensemble         : {args.val_ensemble}")
# # #     print(f"  patience             : {args.patience} epochs  (min_epochs={args.min_epochs})")
# # #     print()

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
# # #     if test_loader:
# # #         print(f"  test  : {len(test_loader.dataset)} seq")

# # #     model = TCFlowMatching(
# # #         pred_len             = args.pred_len,
# # #         obs_len              = args.obs_len,
# # #         sigma_min            = args.sigma_min,
# # #         n_train_ens          = args.n_train_ens,
# # #         ctx_noise_scale      = args.ctx_noise_scale,
# # #         initial_sample_sigma = args.initial_sample_sigma,
# # #     ).to(device)

# # #     if (args.fno_spatial_down != 32 or args.fno_modes_h != 4
# # #             or args.fno_layers != 4 or args.fno_d_model != 32):
# # #         from Model.FNO3D_encoder import FNO3DEncoder
# # #         model.net.spatial_enc = FNO3DEncoder(
# # #             in_channel=13, out_channel=1,
# # #             d_model=args.fno_d_model, n_layers=args.fno_layers,
# # #             modes_t=args.fno_modes_t, modes_h=args.fno_modes_h,
# # #             modes_w=args.fno_modes_h, spatial_down=args.fno_spatial_down,
# # #             dropout=0.05).to(device)

# # #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # #     print(f"  params  : {n_params:,}")

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
# # #     scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
# # #     saver  = BestModelSaver(patience=args.patience, ade_tol=5.0)
# # #     scaler = GradScaler('cuda', enabled=args.use_amp)

# # #     print("=" * 68)
# # #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch)")
# # #     print("=" * 68)

# # #     epoch_times   = []
# # #     train_start   = time.perf_counter()
# # #     last_val_loss = float("inf")
# # #     _lr_ep30_done = False
# # #     _lr_ep60_done = False
# # #     _prev_ens     = 1

# # #     import Model.losses as _losses_mod

# # #     for epoch in range(args.num_epochs):
# # #         # Progressive ensemble
# # #         current_ens = 1 if epoch < 30 else (2 if epoch < 60 else args.n_train_ens)
# # #         model.n_train_ens = current_ens
# # #         effective_fast_ens = min(args.fast_ensemble, max(current_ens*2, args.fast_ensemble))

# # #         if current_ens != _prev_ens:
# # #             saver.reset_counters(
# # #                 f"n_train_ens {_prev_ens}→{current_ens} at epoch {epoch}")
# # #             _prev_ens = current_ens

# # #         curr_len = get_curriculum_len(epoch, args)
# # #         if hasattr(model, "set_curriculum_len"):
# # #             model.set_curriculum_len(curr_len)

# # #         epoch_weights = copy.copy(_BASE_WEIGHTS)
# # #         epoch_weights["pinn"]     = get_pinn_weight(epoch, args.pinn_warmup_epochs,
# # #                                                     args.pinn_w_start, args.pinn_w_end)
# # #         epoch_weights["velocity"] = get_velocity_weight(epoch, args.vel_warmup_epochs,
# # #                                                         args.vel_w_start, args.vel_w_end)
# # #         epoch_weights["recurv"]   = get_recurv_weight(epoch, args.recurv_warmup_epochs,
# # #                                                       args.recurv_w_start, args.recurv_w_end)
# # #         _losses_mod.WEIGHTS.update(epoch_weights)
# # #         if hasattr(model, 'weights'):
# # #             model.weights = epoch_weights
# # #         _losses_mod.WEIGHTS["pinn"]     = epoch_weights["pinn"]
# # #         _losses_mod.WEIGHTS["velocity"] = epoch_weights["velocity"]
# # #         _losses_mod.WEIGHTS["recurv"]   = epoch_weights["recurv"]

# # #         current_clip = get_grad_clip(epoch, warmup_epochs=20,
# # #                                      clip_start=args.grad_clip, clip_end=1.0)

# # #         # LR restarts
# # #         if epoch == 30 and not _lr_ep30_done:
# # #             _lr_ep30_done = True
# # #             warmup_steps = steps_per_epoch * 1
# # #             scheduler = get_cosine_schedule_with_warmup(
# # #                 optimizer, warmup_steps,
# # #                 steps_per_epoch * (args.num_epochs - 30),
# # #                 min_lr=5e-6)
# # #             saver.reset_counters("LR warm restart at epoch 30")
# # #             print(f"  ↺  Warm Restart LR at epoch 30")

# # #         if epoch == 60 and not _lr_ep60_done:
# # #             _lr_ep60_done = True
# # #             warmup_steps = steps_per_epoch * 1
# # #             scheduler = get_cosine_schedule_with_warmup(
# # #                 optimizer, warmup_steps,
# # #                 steps_per_epoch * (args.num_epochs - 60),
# # #                 min_lr=1e-6)
# # #             saver.reset_counters("LR warm restart at epoch 60")
# # #             print(f"  ↺  Warm Restart LR at epoch 60")

# # #         # ── Training loop ─────────────────────────────────────────────────────
# # #         model.train()
# # #         sum_loss = 0.0
# # #         t0 = time.perf_counter()
# # #         optimizer.zero_grad()
# # #         recurv_ratio_buf = []

# # #         for i, batch in enumerate(train_loader):
# # #             bl = move(list(batch), device)

# # #             if epoch == 0 and i == 0:
# # #                 test_env = bl[13]
# # #                 if test_env is not None and "gph500_mean" in test_env:
# # #                     gph_val = test_env["gph500_mean"]
# # #                     if torch.all(gph_val == 0):
# # #                         print("\n" + "!"*60)
# # #                         print("  ⚠️  GPH500 đang bị triệt tiêu về 0!")
# # #                         print("!"*60 + "\n")
# # #                     else:
# # #                         print(f"  ✅ Data Check: GPH500 OK (Mean: {gph_val.mean().item():.4f})")

# # #             with autocast(device_type='cuda', enabled=args.use_amp):
# # #                 bd = model.get_loss_breakdown(bl)

# # #             loss_to_backpass = bd["total"] / max(args.grad_accum, 1)
# # #             scaler.scale(loss_to_backpass).backward()

# # #             if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
# # #                 scaler.unscale_(optimizer)
# # #                 torch.nn.utils.clip_grad_norm_(model.parameters(), current_clip)
# # #                 scaler.step(optimizer)
# # #                 scaler.update()
# # #                 scheduler.step()
# # #                 optimizer.zero_grad()

# # #             sum_loss += bd["total"].item()

# # #             if "recurv_ratio" in bd:
# # #                 recurv_ratio_buf.append(bd["recurv_ratio"])

# # #             if i % 20 == 0:
# # #                 lr  = optimizer.param_groups[0]["lr"]
# # #                 rr  = bd.get("recurv_ratio", 0.0)
# # #                 elapsed = time.perf_counter() - t0
# # #                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# # #                       f"  loss={bd['total'].item():.3f}"
# # #                       f"  fm={bd.get('fm',0):.2f}"
# # #                       f"  vel={bd.get('velocity',0):.6f}"
# # #                       f"  pinn={bd.get('pinn', 0):.6f}"
# # #                       f"  recurv={bd.get('recurv',0):.3f}"
# # #                       f"  rr={rr:.2f}"
# # #                       f"  pinn_w={epoch_weights['pinn']:.3f}"
# # #                       f"  clip={current_clip:.1f}"
# # #                       f"  ens={current_ens}  len={curr_len}"
# # #                       f"  lr={lr:.2e}  t={elapsed:.0f}s")

# # #         ep_s  = time.perf_counter() - t0
# # #         epoch_times.append(ep_s)
# # #         avg_t = sum_loss / len(train_loader)
# # #         mean_rr = float(np.mean(recurv_ratio_buf)) if recurv_ratio_buf else 0.0

# # #         # ── Val loss ───────────────────────────────────────────────────────────
# # #         if epoch % args.val_freq == 0:
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
# # #             saver.update_val_loss(last_val_loss, model, args.output_dir,
# # #                                   epoch, optimizer, avg_t)
# # #             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
# # #                   f"  rr={mean_rr:.2f}"
# # #                   f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
# # #                   f"  ens={current_ens}  len={curr_len}"
# # #                   f"  recurv_w={epoch_weights['recurv']:.2f}")
# # #         else:
# # #             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
# # #                   f"  val={last_val_loss:.3f}(cached)"
# # #                   f"  rr={mean_rr:.2f}  t={ep_s:.0f}s")

# # #         # ── ADE evaluation mỗi epoch ───────────────────────────────────────────
# # #         t_ade = time.perf_counter()
# # #         m = evaluate_fast(model, val_subset_loader, device,
# # #                           ode_train, args.pred_len, effective_fast_ens)
# # #         t_ade_s = time.perf_counter() - t_ade

# # #         spread_72h = m.get("spread_72h_km", 0.0)
# # #         collapse_warn = "  ⚠️ COLLAPSE!" if spread_72h < 10.0 else ""

# # #         print(f"  [ADE ep{epoch} {t_ade_s:.0f}s]"
# # #               f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km"
# # #               f"  12h={m.get('12h',0):.0f}  24h={m.get('24h',0):.0f}"
# # #               f"  72h={m.get('72h',0):.0f} km"
# # #               f"  spread={spread_72h:.1f} km"
# # #               f"  (ens={effective_fast_ens}, steps={ode_train})"
# # #               f"  counter={saver.counter_ade}/{args.patience}"
# # #               f"{collapse_warn}")

# # #         saver.update_ade(m["ADE"], model, args.output_dir, epoch,
# # #                          optimizer, avg_t, last_val_loss,
# # #                          min_epochs=args.min_epochs)

# # #         # ── Full eval ──────────────────────────────────────────────────────────
# # #         if epoch % args.full_eval_freq == 0 and epoch > 0:
# # #             print(f"  [Full eval epoch {epoch}, ode_steps={ode_val}]")
# # #             try:
# # #                 dm, _, _, _ = evaluate_full(
# # #                     model, val_loader, device,
# # #                     ode_val, args.pred_len, args.val_ensemble,
# # #                     metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}")
# # #                 print(dm.summary())
# # #             except Exception as e:
# # #                 print(f"  ⚠  full_eval failed at epoch {epoch}: {e}")
# # #                 import traceback; traceback.print_exc()

# # #         if (epoch+1) % args.save_interval == 0:
# # #             torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
# # #                        os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))

# # #         if saver.early_stop:
# # #             print(f"  Early stopping @ epoch {epoch}")
# # #             break

# # #         if epoch % 5 == 4:
# # #             avg_ep    = sum(epoch_times) / len(epoch_times)
# # #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# # #             elapsed_h = (time.perf_counter() - train_start) / 3600
# # #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
# # #                   f"  (avg {avg_ep:.0f}s/epoch)")

# # #     _losses_mod.WEIGHTS["pinn"]     = args.pinn_w_end
# # #     _losses_mod.WEIGHTS["velocity"] = args.vel_w_end
# # #     _losses_mod.WEIGHTS["recurv"]   = args.recurv_w_end

# # #     total_train_h = (time.perf_counter() - train_start) / 3600

# # #     # ── Final test eval ───────────────────────────────────────────────────────
# # #     print(f"\n{'='*68}  FINAL TEST (ode_steps={ode_test})")
# # #     all_results = []

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
# # #             print(f"  Loaded best @ epoch {ck.get('epoch','?')}"
# # #                   f"  ADE={ck.get('val_ade_km','?')}")

# # #         final_ens = max(args.val_ensemble, 50)
# # #         dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
# # #             model, test_loader, device,
# # #             ode_test, args.pred_len, final_ens,
# # #             metrics_csv=metrics_csv, tag="test_final",
# # #             predict_csv=predict_csv)
# # #         print(dm_test.summary())

# # #         all_results.append(ModelResult(
# # #             model_name   = "FM+PINN-v21",
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

# # #         _, cliper_errs = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
# # #         persist_errs   = persistence_errors(obs_seqs, gt_seqs, args.pred_len)
# # #         fmpinn_per_seq = np.array([seq_ade_km(np.array(pp), np.array(g))
# # #                                     for pp, g in zip(pred_seqs, gt_seqs)])

# # #         np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
# # #         np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
# # #         np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))

# # #         lstm_per_seq      = _load_baseline_errors(args.lstm_errors_npy,      "LSTM")
# # #         diffusion_per_seq = _load_baseline_errors(args.diffusion_errors_npy, "Diffusion")
# # #         if lstm_per_seq is not None:
# # #             np.save(os.path.join(stat_dir, "lstm.npy"), lstm_per_seq)
# # #         if diffusion_per_seq is not None:
# # #             np.save(os.path.join(stat_dir, "diffusion.npy"), diffusion_per_seq)

# # #         _dummy = np.array([float("nan")])
# # #         run_all_tests(
# # #             fmpinn_ade    = fmpinn_per_seq,
# # #             cliper_ade    = cliper_errs.mean(1),
# # #             lstm_ade      = lstm_per_seq if lstm_per_seq is not None else _dummy,
# # #             diffusion_ade = diffusion_per_seq if diffusion_per_seq is not None else _dummy,
# # #             persist_ade   = persist_errs.mean(1),
# # #             out_dir       = stat_dir)

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
# # #             paired_tests(fmpinn_per_seq, cliper_errs.mean(1),  "FM+PINN vs CLIPER",      5),
# # #             paired_tests(fmpinn_per_seq, persist_errs.mean(1), "FM+PINN vs Persistence", 5),
# # #         ]
# # #         if lstm_per_seq is not None:
# # #             stat_rows.append(paired_tests(fmpinn_per_seq, lstm_per_seq, "FM+PINN vs LSTM", 5))
# # #         if diffusion_per_seq is not None:
# # #             stat_rows.append(paired_tests(fmpinn_per_seq, diffusion_per_seq, "FM+PINN vs Diffusion", 5))

# # #         compute_rows = DEFAULT_COMPUTE
# # #         try:
# # #             sb = next(iter(test_loader))
# # #             sb = move(list(sb), device)
# # #             from utils.evaluation_tables import profile_model_components
# # #             compute_rows = profile_model_components(model, sb, device)
# # #         except Exception as e:
# # #             print(f"  Profiling skipped: {e}")

# # #         export_all_tables(
# # #             results=all_results, ablation_rows=DEFAULT_ABLATION,
# # #             stat_rows=stat_rows, pinn_sens_rows=DEFAULT_PINN_SENSITIVITY,
# # #             compute_rows=compute_rows, out_dir=tables_dir)

# # #         with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
# # #             fh.write(dm_test.summary())
# # #             fh.write(f"\n\nmodel_version         : FM+PINN v21\n")
# # #             fh.write(f"sigma_min             : {args.sigma_min}\n")
# # #             fh.write(f"ctx_noise_scale       : {args.ctx_noise_scale}\n")
# # #             fh.write(f"initial_sample_sigma  : {args.initial_sample_sigma}\n")
# # #             fh.write(f"ode_steps_test        : {ode_test}\n")
# # #             fh.write(f"eval_ensemble         : {final_ens}\n")
# # #             fh.write(f"train_time_h          : {total_train_h:.2f}\n")
# # #             fh.write(f"n_params_M            : "
# # #                      f"{sum(p.numel() for p in model.parameters())/1e6:.2f}\n")

# # #     avg_ep = sum(epoch_times)/len(epoch_times) if epoch_times else 0
# # #     print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
# # #     print(f"  Best val loss  : {saver.best_val_loss:.4f}")
# # #     print(f"  Avg epoch time : {avg_ep:.0f}s")
# # #     print(f"  Total training : {total_train_h:.2f}h")
# # #     print(f"  Tables dir     : {tables_dir}")
# # #     print("=" * 68)


# # # if __name__ == "__main__":
# # #     args = get_args()
# # #     np.random.seed(42);  torch.manual_seed(42)
# # #     if torch.cuda.is_available():
# # #         torch.cuda.manual_seed_all(42)
# # #     main(args)

# # """
# # scripts/train_flowmatching.py  ── v21
# # ======================================
# # FIXES vs v20:

# # FIX-V21-1  [CRASH] evaluate_fast(): pred_mean từ model.sample() có T=active_pred_len
# #            (ví dụ T=4 khi curriculum), nhưng bl[1] (gt) luôn có T=pred_len=12.
# #            haversine_km_torch(pred[4,B,2], gt[12,B,2]) → RuntimeError dim 0: 4 vs 12.
# #            Fix: slice gt xuống active_pred_len trước khi tính distance.
# #            Đây là bug gây crash được báo trong traceback.

# # FIX-V21-2  [CRASH/WRONG] evaluate_full() cliper loop: gt_point được tạo từ
# #            gd_np[h, b, :][np.newaxis, :] → shape [1, 2] (0.1-degree units).
# #            Sau đó gt_deg = denorm_deg_np(gt_point[h:h+1]):
# #              - Lỗi 1: gt_point[h:h+1] khi h>0 → EMPTY array (gt_point chỉ có 1 row).
# #              - Lỗi 2: gd_np đã là 0.1-degree (sau denorm_torch), không phải normalized.
# #                Nếu truyền 0.1-deg vào denorm_deg_np sẽ ra giá trị vô nghĩa (~5800 deg).
# #            Fix: dùng haversine_km_np trực tiếp với 0.1-deg values (unit_01deg=True),
# #            và dùng denorm_np (→ 0.1-deg) thay vì denorm_deg_np cho cliper prediction.

# # FIX-V21-3  [WRONG BSS] evaluate_full(): ens_seqs_01 lưu ens_b với shape [S, T, 2].
# #            Nhưng khi truyền vào brier_skill_score, code làm:
# #              e.transpose(1,0,2) if e.ndim==3 else e
# #            → [T, S, 2] — đảo chiều S và T, gây tính BSS sai hoàn toàn.
# #            brier_skill_score() dùng ens_seqs[s, step] = lấy từ dim-0 (S) và dim-1 (T),
# #            nên cần giữ nguyên [S, T, 2].
# #            Fix: bỏ transpose, truyền ens_b trực tiếp (đã đúng [S, T, 2]).

# # Kept from v20:
# #     FIX-V20-1..3 (ensemble collapse, early stop, spread monitor)
# #     FIX-V19-1..2
# #     FIX-V18-1..4
# # """
# # from __future__ import annotations

# # import sys
# # import os
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # import argparse
# # import time
# # import math
# # import random
# # import copy

# # import numpy as np
# # import torch
# # import torch.optim as optim
# # from torch.amp import autocast, GradScaler
# # from torch.utils.data import DataLoader, Subset

# # from Model.data.loader_training import data_loader
# # from Model.flow_matching_model import TCFlowMatching
# # from Model.utils import get_cosine_schedule_with_warmup
# # from Model.losses import WEIGHTS as _BASE_WEIGHTS
# # from utils.metrics import (
# #     TCEvaluator, StepErrorAccumulator,
# #     save_metrics_csv, haversine_km_torch, haversine_km,
# #     denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
# #     cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
# #     brier_skill_score,
# # )
# # from utils.evaluation_tables import (
# #     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
# #     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
# #     DEFAULT_COMPUTE, paired_tests, persistence_errors, cliper_errors,
# # )
# # from scripts.statistical_tests import run_all_tests


# # # ── Helpers ───────────────────────────────────────────────────────────────────

# # def haversine_km_np_local(pred_deg: np.ndarray, gt_deg: np.ndarray) -> np.ndarray:
# #     pred_deg = np.atleast_2d(pred_deg)
# #     gt_deg   = np.atleast_2d(gt_deg)
# #     R = 6371.0
# #     lon1, lat1 = np.radians(pred_deg[:, 0]), np.radians(pred_deg[:, 1])
# #     lon2, lat2 = np.radians(gt_deg[:, 0]),   np.radians(gt_deg[:, 1])
# #     dlon = lon2 - lon1;  dlat = lat2 - lat1
# #     a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
# #     return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# # def seq_ade_km(pred_norm: np.ndarray, gt_norm: np.ndarray) -> float:
# #     return float(haversine_km_np_local(denorm_deg_np(pred_norm),
# #                                        denorm_deg_np(gt_norm)).mean())


# # # ── Adaptive weight schedules ─────────────────────────────────────────────────

# # def get_pinn_weight(epoch, warmup_epochs=30, w_start=0.01, w_end=0.1):
# #     if epoch >= warmup_epochs:
# #         return w_end
# #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# # def get_velocity_weight(epoch, warmup_epochs=20, w_start=0.5, w_end=1.5):
# #     if epoch >= warmup_epochs:
# #         return w_end
# #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# # def get_recurv_weight(epoch, warmup_epochs=10, w_start=0.3, w_end=1.0):
# #     if epoch >= warmup_epochs:
# #         return w_end
# #     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# # def get_grad_clip(epoch, warmup_epochs=20, clip_start=2.0, clip_end=1.0):
# #     if epoch >= warmup_epochs:
# #         return clip_end
# #     return clip_start - (epoch / max(warmup_epochs-1, 1)) * (clip_start - clip_end)


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
# #     p.add_argument("--grad_clip",       default=2.0,            type=float)
# #     p.add_argument("--grad_accum",      default=2,              type=int)
# #     p.add_argument("--patience",        default=50,             type=int)
# #     p.add_argument("--min_epochs",      default=80,             type=int)
# #     p.add_argument("--n_train_ens",     default=6,              type=int)
# #     p.add_argument("--use_amp",         action="store_true")
# #     p.add_argument("--num_workers",     default=2,              type=int)
# #     p.add_argument("--sigma_min",       default=0.05,           type=float)
# #     p.add_argument("--ctx_noise_scale",      default=0.05, type=float,
# #                    help="Gaussian noise injected into raw_ctx per ensemble member at inference")
# #     p.add_argument("--initial_sample_sigma", default=0.3,  type=float,
# #                    help="Initial noise std for ODE sampling (must >> sigma_min for spread)")
# #     p.add_argument("--ode_steps_train", default=20,             type=int)
# #     p.add_argument("--ode_steps_val",   default=30,             type=int)
# #     p.add_argument("--ode_steps_test",  default=50,             type=int)
# #     p.add_argument("--ode_steps",       default=None,           type=int)
# #     p.add_argument("--val_ensemble",    default=30,             type=int)
# #     p.add_argument("--fast_ensemble",   default=8,              type=int)
# #     p.add_argument("--fno_modes_h",     default=4,              type=int)
# #     p.add_argument("--fno_modes_t",     default=4,              type=int)
# #     p.add_argument("--fno_layers",      default=4,              type=int)
# #     p.add_argument("--fno_d_model",     default=32,             type=int)
# #     p.add_argument("--fno_spatial_down",default=32,             type=int)
# #     p.add_argument("--mamba_d_state",   default=16,             type=int)
# #     p.add_argument("--val_loss_freq",   default=2,              type=int)
# #     p.add_argument("--val_freq",        default=2,              type=int)
# #     p.add_argument("--full_eval_freq",  default=10,             type=int)
# #     p.add_argument("--val_subset_size", default=600,            type=int)
# #     p.add_argument("--output_dir",      default="runs/v21",     type=str)
# #     p.add_argument("--save_interval",   default=10,             type=int)
# #     p.add_argument("--metrics_csv",     default="metrics.csv",        type=str)
# #     p.add_argument("--predict_csv",     default="predictions.csv",    type=str)
# #     p.add_argument("--lstm_errors_npy",      default=None, type=str)
# #     p.add_argument("--diffusion_errors_npy", default=None, type=str)
# #     p.add_argument("--gpu_num",         default="0",            type=str)
# #     p.add_argument("--delim",           default=" ")
# #     p.add_argument("--skip",            default=1,              type=int)
# #     p.add_argument("--min_ped",         default=1,              type=int)
# #     p.add_argument("--threshold",       default=0.002,          type=float)
# #     p.add_argument("--other_modal",     default="gph")
# #     p.add_argument("--curriculum",      default=True,
# #                    type=lambda x: x.lower() != 'false')
# #     p.add_argument("--curriculum_start_len", default=4,         type=int)
# #     p.add_argument("--curriculum_end_epoch", default=40,        type=int)
# #     p.add_argument("--lon_flip_prob",   default=0.3,            type=float)
# #     p.add_argument("--pinn_warmup_epochs", default=30,          type=int)
# #     p.add_argument("--pinn_w_start",    default=0.01,           type=float)
# #     p.add_argument("--pinn_w_end",      default=0.1,            type=float)
# #     p.add_argument("--vel_warmup_epochs",  default=20,          type=int)
# #     p.add_argument("--vel_w_start",        default=0.5,         type=float)
# #     p.add_argument("--vel_w_end",          default=1.5,         type=float)
# #     p.add_argument("--recurv_warmup_epochs", default=10,        type=int)
# #     p.add_argument("--recurv_w_start",       default=0.3,       type=float)
# #     p.add_argument("--recurv_w_end",         default=1.0,       type=float)
# #     return p.parse_args()


# # def _resolve_ode_steps(args):
# #     if args.ode_steps is not None:
# #         return args.ode_steps, args.ode_steps, args.ode_steps
# #     return args.ode_steps_train, args.ode_steps_val, args.ode_steps_test


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
# #     idx = rng.sample(range(n), min(subset_size, n))
# #     return DataLoader(Subset(val_dataset, idx),
# #                       batch_size=batch_size, shuffle=False,
# #                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# # def get_curriculum_len(epoch, args) -> int:
# #     if not args.curriculum:
# #         return args.pred_len
# #     if epoch >= args.curriculum_end_epoch:
# #         return args.pred_len
# #     frac = epoch / max(args.curriculum_end_epoch, 1)
# #     return int(args.curriculum_start_len
# #                + frac * (args.pred_len - args.curriculum_start_len))


# # def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
# #     """
# #     FIX-V21-1: Khi curriculum đang chạy, model.sample() trả về pred_mean với
# #     T=active_pred_len (ví dụ 4), nhưng bl[1] (gt) luôn có T=pred_len=12.
# #     Phải slice gt xuống active_pred_len để shapes khớp nhau trước khi
# #     truyền vào haversine_km_torch.

# #     FIX-V20-3: Tính ensemble spread để detect collapse sớm.
# #     """
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
# #             # FIX-V21-1: pred.shape = [T_active, B, 2], bl[1].shape = [T_pred, B, 2]
# #             # T_active <= T_pred (curriculum). Slice gt to match pred length.
# #             T_active = pred.shape[0]
# #             gt_sliced = bl[1][:T_active]  # [T_active, B, 2]
# #             acc.update(haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced)))

# #             # Spread: std của ensemble tại step cuối
# #             last_step = all_trajs[:, -1, :, :]  # [S, B, 2]
# #             std_lon = last_step[:, :, 0].std(0)   # [B]
# #             std_lat = last_step[:, :, 1].std(0)   # [B]
# #             spread_km = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
# #             spread_buf.append(spread_km)
# #             n += 1

# #     r = acc.compute()
# #     r["ms_per_batch"]  = (time.perf_counter() - t0) * 1e3 / max(n, 1)
# #     r["spread_72h_km"] = float(np.mean(spread_buf)) if spread_buf else 0.0
# #     return r


# # def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
# #                   metrics_csv, tag="", predict_csv=""):
# #     """
# #     FIX-V21-2: Sửa tính sai CLIPER error trong vòng lặp cliper.
# #       - Bug cũ: gt_point = gd_np[h,b,:][np.newaxis,:] rồi gt_deg = denorm_deg_np(gt_point[h:h+1])
# #         * gd_np đã là 0.1-degree (sau denorm_torch) -- không phải normalized coords
# #         * gt_point[h:h+1] khi h>0 → empty array vì gt_point chỉ có 1 hàng
# #       - Fix: so sánh trực tiếp ở 0.1-degree space với haversine_km_np(..., unit_01deg=True)
# #         * dùng denorm_np() (→ 0.1-deg) cho cliper prediction
# #         * gt đã là 0.1-deg từ gd_np, dùng trực tiếp

# #     FIX-V21-3: Sửa shape của ens_seqs_01 khi truyền vào brier_skill_score.
# #       - Bug cũ: [S,T,2].transpose(1,0,2) → [T,S,2] — đảo chiều S,T sai
# #       - brier_skill_score truy cập ens_seqs[s, step] nên cần [S, T, 2]
# #       - Fix: truyền trực tiếp ens_b [S,T,2] không transpose
# #     """
# #     model.eval()
# #     cliper_step_errors = []
# #     ev = TCEvaluator(pred_len=pred_len, compute_dtw=True)
# #     obs_seqs_01 = []; gt_seqs_01 = []; pred_seqs_01 = []; ens_seqs_01 = []

# #     with torch.no_grad():
# #         for batch in loader:
# #             bl  = move(list(batch), device)
# #             gt  = bl[1];  obs = bl[0]
# #             pred_mean, _, all_trajs = model.sample(
# #                 bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
# #                 predict_csv=predict_csv if predict_csv else None)

# #             pd_np = denorm_torch(pred_mean).cpu().numpy()   # [T, B, 2] in 0.1-deg
# #             gd_np = denorm_torch(gt).cpu().numpy()          # [T, B, 2] in 0.1-deg
# #             od_np = denorm_torch(obs).cpu().numpy()         # [T_obs, B, 2] in 0.1-deg
# #             ed_np = denorm_torch(all_trajs).cpu().numpy()   # [S, T, B, 2] in 0.1-deg

# #             B = pd_np.shape[1]
# #             for b in range(B):
# #                 # ens_b: [S, T, 2] — shape đúng cho brier_skill_score
# #                 ens_b = ed_np[:, :, b, :]
# #                 ev.update(pd_np[:, b, :], gd_np[:, b, :], pred_ens=ens_b)
# #                 obs_seqs_01.append(od_np[:, b, :])
# #                 gt_seqs_01.append(gd_np[:, b, :])
# #                 pred_seqs_01.append(pd_np[:, b, :])
# #                 # FIX-V21-3: giữ nguyên [S, T, 2], KHÔNG transpose
# #                 ens_seqs_01.append(ens_b)

# #                 # ── CLIPER error per step ─────────────────────────────────────
# #                 # FIX-V21-2: obs_b là normalized coords từ od_np (0.1-deg), cần
# #                 # truyền normalized obs vào cliper_forecast, không phải 0.1-deg.
# #                 # Lấy normalized obs từ bl[0] (obs_traj tensor trực tiếp).
# #                 obs_b_norm = obs.cpu().numpy()[:, b, :]  # [T_obs, 2] normalized

# #                 cliper_errors_b = np.zeros(pred_len)
# #                 for h in range(pred_len):
# #                     # cliper_forecast nhận normalized coords, trả về normalized [2]
# #                     pred_cliper_norm = cliper_forecast(obs_b_norm, h + 1)

# #                     # Chuyển sang 0.1-degree để so sánh với gd_np
# #                     pred_cliper_01 = denorm_np(pred_cliper_norm[np.newaxis])  # [1, 2] 0.1-deg

# #                     # gt tại step h: gd_np[h, b, :] đã là 0.1-deg
# #                     gt_01 = gd_np[h, b, :][np.newaxis]  # [1, 2] 0.1-deg

# #                     # So sánh ở 0.1-deg space
# #                     cliper_errors_b[h] = float(
# #                         haversine_km_np(pred_cliper_01, gt_01, unit_01deg=True)[0]
# #                     )

# #                 cliper_step_errors.append(cliper_errors_b)

# #     if cliper_step_errors:
# #         cliper_mat = np.stack(cliper_step_errors)
# #         cliper_ugde_dict = {h: float(cliper_mat[:, s].mean())
# #                             for h, s in HORIZON_STEPS.items()
# #                             if s < cliper_mat.shape[1]}
# #         ev.cliper_ugde = cliper_ugde_dict
# #         print(f"  [CLIPER UGDE] 72h={cliper_ugde_dict.get(72, float('nan')):.1f} km")

# #     dm = ev.compute(tag=tag)

# #     try:
# #         if LANDFALL_TARGETS and ens_seqs_01:
# #             bss_vals = []
# #             step_72  = HORIZON_STEPS.get(72, pred_len - 1)
# #             for tname, t_lon, t_lat in LANDFALL_TARGETS:
# #                 # FIX-V21-3: ens_seqs_01 entries are [S, T, 2] — đúng format
# #                 # brier_skill_score expects list of [S, T, 2] in 0.1-deg units
# #                 bv = brier_skill_score(
# #                     ens_seqs_01,   # list of [S, T, 2], NO transpose needed
# #                     gt_seqs_01, min(step_72, pred_len-1),
# #                     (t_lon * 10.0, t_lat * 10.0),  # convert degrees to 0.1-deg for consistency
# #                     LANDFALL_RADIUS_KM)
# #                 if not math.isnan(bv):
# #                     bss_vals.append(bv)
# #             if bss_vals:
# #                 dm.bss_mean = float(np.mean(bss_vals))
# #                 print(f"  [BSS] mean={dm.bss_mean:.4f}")
# #     except Exception as e:
# #         print(f"  ⚠  BSS failed: {e}")

# #     save_metrics_csv(dm, metrics_csv, tag=tag)
# #     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# # class BestModelSaver:
# #     def __init__(self, patience=50, ade_tol=5.0):
# #         self.patience      = patience
# #         self.ade_tol       = ade_tol
# #         self.best_ade      = float("inf")
# #         self.best_val_loss = float("inf")
# #         self.counter_ade   = 0
# #         self.counter_loss  = 0
# #         self.early_stop    = False

# #     def reset_counters(self, reason=""):
# #         self.counter_ade  = 0
# #         self.counter_loss = 0
# #         if reason:
# #             print(f"  [SAVER] Patience counters reset: {reason}")

# #     def update_val_loss(self, val_loss, model, out_dir, epoch, optimizer, tl):
# #         if val_loss < self.best_val_loss - 1e-4:
# #             self.best_val_loss = val_loss;  self.counter_loss = 0
# #             torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
# #                             optimizer_state=optimizer.state_dict(),
# #                             train_loss=tl, val_loss=val_loss,
# #                             model_version="v21-valloss"),
# #                        os.path.join(out_dir, "best_model_valloss.pth"))
# #         else:
# #             self.counter_loss += 1

# #     def update_ade(self, ade, model, out_dir, epoch, optimizer, tl, vl,
# #                    min_epochs=80):
# #         if ade < self.best_ade - self.ade_tol:
# #             self.best_ade = ade;  self.counter_ade = 0
# #             torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
# #                             optimizer_state=optimizer.state_dict(),
# #                             train_loss=tl, val_loss=vl, val_ade_km=ade,
# #                             model_version="v21-FNO-Mamba-recurv"),
# #                        os.path.join(out_dir, "best_model.pth"))
# #             print(f"  ✅ Best ADE {ade:.1f} km  (epoch {epoch})")
# #         else:
# #             self.counter_ade += 1
# #             print(f"  No ADE improvement {self.counter_ade}/{self.patience}"
# #                   f"  (Δ={self.best_ade-ade:.1f} km < tol={self.ade_tol} km)"
# #                   f"  | Loss counter {self.counter_loss}/{self.patience}")

# #         if epoch >= min_epochs:
# #             if self.counter_ade >= self.patience and self.counter_loss >= self.patience:
# #                 self.early_stop = True
# #                 print(f"  ⛔ Early stop @ epoch {epoch}")
# #         else:
# #             if self.counter_ade >= self.patience and self.counter_loss >= self.patience:
# #                 print(f"  ⚠  Would stop but min_epochs={min_epochs} not reached. Continuing...")
# #                 self.counter_ade = 0;  self.counter_loss = 0


# # def _load_baseline_errors(path, name):
# #     if path is None:
# #         print(f"\n  ⚠  WARNING: --{name.lower().replace(' ','_')}_errors_npy not provided.")
# #         print(f"     Statistical comparison vs {name} will be SKIPPED.\n")
# #         return None
# #     if not os.path.exists(path):
# #         print(f"\n  ⚠  {path} not found — {name} skipped.\n")
# #         return None
# #     arr = np.load(path)
# #     print(f"  ✓  Loaded {name}: {arr.shape}")
# #     return arr


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

# #     ode_train, ode_val, ode_test = _resolve_ode_steps(args)

# #     print("=" * 68)
# #     print("  TC-FlowMatching v21  |  FNO3D + Mamba + OT-CFM + PINN")
# #     print("  v21 FIXES:")
# #     print("    FIX-V21-1: evaluate_fast crash → slice gt to active_pred_len")
# #     print("    FIX-V21-2: CLIPER errors sai đơn vị và index out of bounds")
# #     print("    FIX-V21-3: BSS sai shape → bỏ transpose ens_seqs_01")
# #     print("=" * 68)
# #     print(f"  device               : {device}")
# #     print(f"  sigma_min            : {args.sigma_min}")
# #     print(f"  ctx_noise_scale      : {args.ctx_noise_scale}")
# #     print(f"  initial_sample_sigma : {args.initial_sample_sigma}")
# #     print(f"  ode_steps            : train={ode_train}  val={ode_val}  test={ode_test}")
# #     print(f"  val_ensemble         : {args.val_ensemble}")
# #     print(f"  patience             : {args.patience} epochs  (min_epochs={args.min_epochs})")
# #     print()

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
# #     if test_loader:
# #         print(f"  test  : {len(test_loader.dataset)} seq")

# #     model = TCFlowMatching(
# #         pred_len             = args.pred_len,
# #         obs_len              = args.obs_len,
# #         sigma_min            = args.sigma_min,
# #         n_train_ens          = args.n_train_ens,
# #         ctx_noise_scale      = args.ctx_noise_scale,
# #         initial_sample_sigma = args.initial_sample_sigma,
# #     ).to(device)

# #     if (args.fno_spatial_down != 32 or args.fno_modes_h != 4
# #             or args.fno_layers != 4 or args.fno_d_model != 32):
# #         from Model.FNO3D_encoder import FNO3DEncoder
# #         model.net.spatial_enc = FNO3DEncoder(
# #             in_channel=13, out_channel=1,
# #             d_model=args.fno_d_model, n_layers=args.fno_layers,
# #             modes_t=args.fno_modes_t, modes_h=args.fno_modes_h,
# #             modes_w=args.fno_modes_h, spatial_down=args.fno_spatial_down,
# #             dropout=0.05).to(device)

# #     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #     print(f"  params  : {n_params:,}")

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
# #     scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
# #     saver  = BestModelSaver(patience=args.patience, ade_tol=5.0)
# #     scaler = GradScaler('cuda', enabled=args.use_amp)

# #     print("=" * 68)
# #     print(f"  TRAINING  ({steps_per_epoch} steps/epoch)")
# #     print("=" * 68)

# #     epoch_times   = []
# #     train_start   = time.perf_counter()
# #     last_val_loss = float("inf")
# #     _lr_ep30_done = False
# #     _lr_ep60_done = False
# #     _prev_ens     = 1

# #     import Model.losses as _losses_mod

# #     for epoch in range(args.num_epochs):
# #         # Progressive ensemble
# #         current_ens = 1 if epoch < 30 else (2 if epoch < 60 else args.n_train_ens)
# #         model.n_train_ens = current_ens
# #         effective_fast_ens = min(args.fast_ensemble, max(current_ens*2, args.fast_ensemble))

# #         if current_ens != _prev_ens:
# #             saver.reset_counters(
# #                 f"n_train_ens {_prev_ens}→{current_ens} at epoch {epoch}")
# #             _prev_ens = current_ens

# #         curr_len = get_curriculum_len(epoch, args)
# #         if hasattr(model, "set_curriculum_len"):
# #             model.set_curriculum_len(curr_len)

# #         epoch_weights = copy.copy(_BASE_WEIGHTS)
# #         epoch_weights["pinn"]     = get_pinn_weight(epoch, args.pinn_warmup_epochs,
# #                                                     args.pinn_w_start, args.pinn_w_end)
# #         epoch_weights["velocity"] = get_velocity_weight(epoch, args.vel_warmup_epochs,
# #                                                         args.vel_w_start, args.vel_w_end)
# #         epoch_weights["recurv"]   = get_recurv_weight(epoch, args.recurv_warmup_epochs,
# #                                                       args.recurv_w_start, args.recurv_w_end)
# #         _losses_mod.WEIGHTS.update(epoch_weights)
# #         if hasattr(model, 'weights'):
# #             model.weights = epoch_weights
# #         _losses_mod.WEIGHTS["pinn"]     = epoch_weights["pinn"]
# #         _losses_mod.WEIGHTS["velocity"] = epoch_weights["velocity"]
# #         _losses_mod.WEIGHTS["recurv"]   = epoch_weights["recurv"]

# #         current_clip = get_grad_clip(epoch, warmup_epochs=20,
# #                                      clip_start=args.grad_clip, clip_end=1.0)

# #         # LR restarts
# #         if epoch == 30 and not _lr_ep30_done:
# #             _lr_ep30_done = True
# #             warmup_steps = steps_per_epoch * 1
# #             scheduler = get_cosine_schedule_with_warmup(
# #                 optimizer, warmup_steps,
# #                 steps_per_epoch * (args.num_epochs - 30),
# #                 min_lr=5e-6)
# #             saver.reset_counters("LR warm restart at epoch 30")
# #             print(f"  ↺  Warm Restart LR at epoch 30")

# #         if epoch == 60 and not _lr_ep60_done:
# #             _lr_ep60_done = True
# #             warmup_steps = steps_per_epoch * 1
# #             scheduler = get_cosine_schedule_with_warmup(
# #                 optimizer, warmup_steps,
# #                 steps_per_epoch * (args.num_epochs - 60),
# #                 min_lr=1e-6)
# #             saver.reset_counters("LR warm restart at epoch 60")
# #             print(f"  ↺  Warm Restart LR at epoch 60")

# #         # ── Training loop ─────────────────────────────────────────────────────
# #         model.train()
# #         sum_loss = 0.0
# #         t0 = time.perf_counter()
# #         optimizer.zero_grad()
# #         recurv_ratio_buf = []

# #         for i, batch in enumerate(train_loader):
# #             bl = move(list(batch), device)

# #             if epoch == 0 and i == 0:
# #                 test_env = bl[13]
# #                 if test_env is not None and "gph500_mean" in test_env:
# #                     gph_val = test_env["gph500_mean"]
# #                     n_zero  = (gph_val == 0).sum().item()
# #                     n_total = gph_val.numel()
# #                     zero_pct = 100.0 * n_zero / max(n_total, 1)

# #                     if torch.all(gph_val == 0):
# #                         # Phân biệt: do env files missing hay do sentinel data?
# #                         env_source = getattr(train_dataset, '_env_path_missing', None)
# #                         has_csv    = bool(getattr(train_dataset, '_csv_env_lookup', {}))
# #                         print("\n" + "!" * 60)
# #                         if env_source is True and not has_csv:
# #                             print("  ⚠️  GPH500 = 0: Env_data .npy KHÔNG TỒN TẠI")
# #                             print("  → all_storms_final.csv cũng KHÔNG TÌM THẤY")
# #                             print("  → Tất cả env features = 0. Model vẫn train")
# #                             print("     nhưng không dùng được GPH500/u500/v500 features.")
# #                             print("  FIX: Đặt all_storms_final.csv cùng thư mục dataset")
# #                             print("       HOẶC truyền csv_env_path= vào data_loader()")
# #                         elif env_source is True and has_csv:
# #                             print("  ⚠️  GPH500 = 0 dù CSV fallback đã load!")
# #                             print("  → Có thể storm_name format mismatch giữa")
# #                             print("     Data1d .txt files và all_storms_final.csv.")
# #                             print("  → Kiểm tra: f_name trong .txt vs storm_name trong CSV")
# #                         else:
# #                             print("  ⚠️  GPH500 đang bị triệt tiêu về 0!")
# #                             print("  → Env_data path tồn tại nhưng tất cả files")
# #                             print("     trả về sentinel hoặc không đọc được.")
# #                         print("!" * 60 + "\n")
# #                     elif zero_pct > 50.0:
# #                         print(f"  ⚠️  Data Check: GPH500 {zero_pct:.1f}% = 0 "
# #                               f"(mean non-zero: {gph_val[gph_val != 0].mean().item():.4f})")
# #                         print(f"     Nhiều timestep dùng sentinel (-29.5) → env features = 0. Bình thường.")
# #                     else:
# #                         # FIX-DATA-13: sentinel ~1% là bình thường, không cần warn
# #                         print(f"  ✅ Data Check: GPH500 OK "
# #                               f"(mean={gph_val.mean().item():.4f}, "
# #                               f"zero={zero_pct:.1f}% = sentinel rows)")

# #             with autocast(device_type='cuda', enabled=args.use_amp):
# #                 bd = model.get_loss_breakdown(bl)

# #             loss_to_backpass = bd["total"] / max(args.grad_accum, 1)
# #             scaler.scale(loss_to_backpass).backward()

# #             if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
# #                 scaler.unscale_(optimizer)
# #                 torch.nn.utils.clip_grad_norm_(model.parameters(), current_clip)
# #                 scaler.step(optimizer)
# #                 scaler.update()
# #                 scheduler.step()
# #                 optimizer.zero_grad()

# #             sum_loss += bd["total"].item()

# #             if "recurv_ratio" in bd:
# #                 recurv_ratio_buf.append(bd["recurv_ratio"])

# #             if i % 20 == 0:
# #                 lr  = optimizer.param_groups[0]["lr"]
# #                 rr  = bd.get("recurv_ratio", 0.0)
# #                 elapsed = time.perf_counter() - t0
# #                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
# #                       f"  loss={bd['total'].item():.3f}"
# #                       f"  fm={bd.get('fm',0):.2f}"
# #                       f"  vel={bd.get('velocity',0):.6f}"
# #                       f"  pinn={bd.get('pinn', 0):.6f}"
# #                       f"  recurv={bd.get('recurv',0):.3f}"
# #                       f"  rr={rr:.2f}"
# #                       f"  pinn_w={epoch_weights['pinn']:.3f}"
# #                       f"  clip={current_clip:.1f}"
# #                       f"  ens={current_ens}  len={curr_len}"
# #                       f"  lr={lr:.2e}  t={elapsed:.0f}s")

# #         ep_s  = time.perf_counter() - t0
# #         epoch_times.append(ep_s)
# #         avg_t = sum_loss / len(train_loader)
# #         mean_rr = float(np.mean(recurv_ratio_buf)) if recurv_ratio_buf else 0.0

# #         # ── Val loss ───────────────────────────────────────────────────────────
# #         if epoch % args.val_freq == 0:
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
# #             saver.update_val_loss(last_val_loss, model, args.output_dir,
# #                                   epoch, optimizer, avg_t)
# #             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
# #                   f"  rr={mean_rr:.2f}"
# #                   f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
# #                   f"  ens={current_ens}  len={curr_len}"
# #                   f"  recurv_w={epoch_weights['recurv']:.2f}")
# #         else:
# #             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
# #                   f"  val={last_val_loss:.3f}(cached)"
# #                   f"  rr={mean_rr:.2f}  t={ep_s:.0f}s")

# #         # ── ADE evaluation mỗi epoch ───────────────────────────────────────────
# #         t_ade = time.perf_counter()
# #         m = evaluate_fast(model, val_subset_loader, device,
# #                           ode_train, args.pred_len, effective_fast_ens)
# #         t_ade_s = time.perf_counter() - t_ade

# #         spread_72h = m.get("spread_72h_km", 0.0)
# #         collapse_warn = "  ⚠️ COLLAPSE!" if spread_72h < 10.0 else ""

# #         print(f"  [ADE ep{epoch} {t_ade_s:.0f}s]"
# #               f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km"
# #               f"  12h={m.get('12h',0):.0f}  24h={m.get('24h',0):.0f}"
# #               f"  72h={m.get('72h',0):.0f} km"
# #               f"  spread={spread_72h:.1f} km"
# #               f"  (ens={effective_fast_ens}, steps={ode_train})"
# #               f"  counter={saver.counter_ade}/{args.patience}"
# #               f"{collapse_warn}")

# #         saver.update_ade(m["ADE"], model, args.output_dir, epoch,
# #                          optimizer, avg_t, last_val_loss,
# #                          min_epochs=args.min_epochs)

# #         # ── Full eval ──────────────────────────────────────────────────────────
# #         if epoch % args.full_eval_freq == 0 and epoch > 0:
# #             print(f"  [Full eval epoch {epoch}, ode_steps={ode_val}]")
# #             try:
# #                 dm, _, _, _ = evaluate_full(
# #                     model, val_loader, device,
# #                     ode_val, args.pred_len, args.val_ensemble,
# #                     metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}")
# #                 print(dm.summary())
# #             except Exception as e:
# #                 print(f"  ⚠  full_eval failed at epoch {epoch}: {e}")
# #                 import traceback; traceback.print_exc()

# #         if (epoch+1) % args.save_interval == 0:
# #             torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
# #                        os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth"))

# #         if saver.early_stop:
# #             print(f"  Early stopping @ epoch {epoch}")
# #             break

# #         if epoch % 5 == 4:
# #             avg_ep    = sum(epoch_times) / len(epoch_times)
# #             remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
# #             elapsed_h = (time.perf_counter() - train_start) / 3600
# #             print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
# #                   f"  (avg {avg_ep:.0f}s/epoch)")

# #     _losses_mod.WEIGHTS["pinn"]     = args.pinn_w_end
# #     _losses_mod.WEIGHTS["velocity"] = args.vel_w_end
# #     _losses_mod.WEIGHTS["recurv"]   = args.recurv_w_end

# #     total_train_h = (time.perf_counter() - train_start) / 3600

# #     # ── Final test eval ───────────────────────────────────────────────────────
# #     print(f"\n{'='*68}  FINAL TEST (ode_steps={ode_test})")
# #     all_results = []

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
# #             print(f"  Loaded best @ epoch {ck.get('epoch','?')}"
# #                   f"  ADE={ck.get('val_ade_km','?')}")

# #         final_ens = max(args.val_ensemble, 50)
# #         dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
# #             model, test_loader, device,
# #             ode_test, args.pred_len, final_ens,
# #             metrics_csv=metrics_csv, tag="test_final",
# #             predict_csv=predict_csv)
# #         print(dm_test.summary())

# #         all_results.append(ModelResult(
# #             model_name   = "FM+PINN-v21",
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

# #         _, cliper_errs = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
# #         persist_errs   = persistence_errors(obs_seqs, gt_seqs, args.pred_len)
# #         fmpinn_per_seq = np.array([seq_ade_km(np.array(pp), np.array(g))
# #                                     for pp, g in zip(pred_seqs, gt_seqs)])

# #         np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
# #         np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
# #         np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))

# #         lstm_per_seq      = _load_baseline_errors(args.lstm_errors_npy,      "LSTM")
# #         diffusion_per_seq = _load_baseline_errors(args.diffusion_errors_npy, "Diffusion")
# #         if lstm_per_seq is not None:
# #             np.save(os.path.join(stat_dir, "lstm.npy"), lstm_per_seq)
# #         if diffusion_per_seq is not None:
# #             np.save(os.path.join(stat_dir, "diffusion.npy"), diffusion_per_seq)

# #         _dummy = np.array([float("nan")])
# #         run_all_tests(
# #             fmpinn_ade    = fmpinn_per_seq,
# #             cliper_ade    = cliper_errs.mean(1),
# #             lstm_ade      = lstm_per_seq if lstm_per_seq is not None else _dummy,
# #             diffusion_ade = diffusion_per_seq if diffusion_per_seq is not None else _dummy,
# #             persist_ade   = persist_errs.mean(1),
# #             out_dir       = stat_dir)

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
# #             paired_tests(fmpinn_per_seq, cliper_errs.mean(1),  "FM+PINN vs CLIPER",      5),
# #             paired_tests(fmpinn_per_seq, persist_errs.mean(1), "FM+PINN vs Persistence", 5),
# #         ]
# #         if lstm_per_seq is not None:
# #             stat_rows.append(paired_tests(fmpinn_per_seq, lstm_per_seq, "FM+PINN vs LSTM", 5))
# #         if diffusion_per_seq is not None:
# #             stat_rows.append(paired_tests(fmpinn_per_seq, diffusion_per_seq, "FM+PINN vs Diffusion", 5))

# #         compute_rows = DEFAULT_COMPUTE
# #         try:
# #             sb = next(iter(test_loader))
# #             sb = move(list(sb), device)
# #             from utils.evaluation_tables import profile_model_components
# #             compute_rows = profile_model_components(model, sb, device)
# #         except Exception as e:
# #             print(f"  Profiling skipped: {e}")

# #         export_all_tables(
# #             results=all_results, ablation_rows=DEFAULT_ABLATION,
# #             stat_rows=stat_rows, pinn_sens_rows=DEFAULT_PINN_SENSITIVITY,
# #             compute_rows=compute_rows, out_dir=tables_dir)

# #         with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
# #             fh.write(dm_test.summary())
# #             fh.write(f"\n\nmodel_version         : FM+PINN v21\n")
# #             fh.write(f"sigma_min             : {args.sigma_min}\n")
# #             fh.write(f"ctx_noise_scale       : {args.ctx_noise_scale}\n")
# #             fh.write(f"initial_sample_sigma  : {args.initial_sample_sigma}\n")
# #             fh.write(f"ode_steps_test        : {ode_test}\n")
# #             fh.write(f"eval_ensemble         : {final_ens}\n")
# #             fh.write(f"train_time_h          : {total_train_h:.2f}\n")
# #             fh.write(f"n_params_M            : "
# #                      f"{sum(p.numel() for p in model.parameters())/1e6:.2f}\n")

# #     avg_ep = sum(epoch_times)/len(epoch_times) if epoch_times else 0
# #     print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
# #     print(f"  Best val loss  : {saver.best_val_loss:.4f}")
# #     print(f"  Avg epoch time : {avg_ep:.0f}s")
# #     print(f"  Total training : {total_train_h:.2f}h")
# #     print(f"  Tables dir     : {tables_dir}")
# #     print("=" * 68)


# # if __name__ == "__main__":
# #     args = get_args()
# #     np.random.seed(42);  torch.manual_seed(42)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(42)
# #     main(args)
# """
# scripts/train_flowmatching.py  ── v22
# ======================================
# FIXES vs v21:

# FIX-V22-1  [CRASH FIX] StepErrorAccumulator.update() nhận dist_km shape
#            [T_active, B] (ví dụ [4, 32] khi curriculum) nhưng self._sum
#            shape [12] → broadcast crash.
#            Fix: metrics.py v5 StepErrorAccumulator tự pad zeros, dùng
#            per-step count để tính average đúng. evaluate_fast() không cần
#            thay đổi logic slice, chỉ cần đảm bảo truyền đúng.

# FIX-V22-2  [FULL VAL ADE] Thêm evaluate_full_val_ade() — mỗi 2 epoch
#            (val_ade_freq=2) chạy full val set với ode_steps_val ensemble
#            để tính ADE chính xác trên toàn bộ val. Kết quả được:
#              - Print ra console với format rõ ràng
#              - Lưu vào metrics_csv với tag "val_full_ep{epoch}"
#              - Dùng làm criteria cho BestModelSaver thay vì ADE từ
#                val_subset_loader
#            Khi curriculum active, dùng pred_len đầy đủ (không slice)
#            để ADE full val phản ánh performance thật.

# FIX-V22-3  [ADE REPORT] evaluate_fast() log thêm "active_steps" từ
#            StepErrorAccumulator.compute() để biết curriculum đang ở
#            bước nào.

# Kept from v21:
#     FIX-V21-1  evaluate_fast() slice gt → T_active (vẫn giữ)
#     FIX-V21-2  evaluate_full() CLIPER fix đơn vị
#     FIX-V21-3  BSS fix transpose ens_seqs_01
#     FIX-V20-1..3 ensemble collapse, early stop, spread monitor
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
#     save_metrics_csv, haversine_km_torch, haversine_km,
#     denorm_torch, denorm_np, denorm_deg_np, HORIZON_STEPS,
#     cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
#     brier_skill_score,
# )
# from utils.evaluation_tables import (
#     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
#     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
#     DEFAULT_COMPUTE, paired_tests, persistence_errors, cliper_errors,
# )
# from scripts.statistical_tests import run_all_tests


# # ── Helpers ───────────────────────────────────────────────────────────────────

# def haversine_km_np_local(pred_deg: np.ndarray, gt_deg: np.ndarray) -> np.ndarray:
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

# def get_pinn_weight(epoch, warmup_epochs=30, w_start=0.01, w_end=0.1):
#     if epoch >= warmup_epochs:
#         return w_end
#     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# def get_velocity_weight(epoch, warmup_epochs=20, w_start=0.5, w_end=1.5):
#     if epoch >= warmup_epochs:
#         return w_end
#     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# def get_recurv_weight(epoch, warmup_epochs=10, w_start=0.3, w_end=1.0):
#     if epoch >= warmup_epochs:
#         return w_end
#     return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


# def get_grad_clip(epoch, warmup_epochs=20, clip_start=2.0, clip_end=1.0):
#     if epoch >= warmup_epochs:
#         return clip_end
#     return clip_start - (epoch / max(warmup_epochs-1, 1)) * (clip_start - clip_end)


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
#     p.add_argument("--grad_clip",       default=2.0,            type=float)
#     p.add_argument("--grad_accum",      default=2,              type=int)
#     p.add_argument("--patience",        default=50,             type=int)
#     p.add_argument("--min_epochs",      default=80,             type=int)
#     p.add_argument("--n_train_ens",     default=6,              type=int)
#     p.add_argument("--use_amp",         action="store_true")
#     p.add_argument("--num_workers",     default=2,              type=int)
#     p.add_argument("--sigma_min",       default=0.05,           type=float)
#     p.add_argument("--ctx_noise_scale",      default=0.05, type=float)
#     p.add_argument("--initial_sample_sigma", default=0.3,  type=float)
#     p.add_argument("--ode_steps_train", default=20,             type=int)
#     p.add_argument("--ode_steps_val",   default=30,             type=int)
#     p.add_argument("--ode_steps_test",  default=50,             type=int)
#     p.add_argument("--ode_steps",       default=None,           type=int)
#     p.add_argument("--val_ensemble",    default=30,             type=int)
#     p.add_argument("--fast_ensemble",   default=8,              type=int)
#     p.add_argument("--fno_modes_h",     default=4,              type=int)
#     p.add_argument("--fno_modes_t",     default=4,              type=int)
#     p.add_argument("--fno_layers",      default=4,              type=int)
#     p.add_argument("--fno_d_model",     default=32,             type=int)
#     p.add_argument("--fno_spatial_down",default=32,             type=int)
#     p.add_argument("--mamba_d_state",   default=16,             type=int)
#     p.add_argument("--val_loss_freq",   default=2,              type=int)
#     p.add_argument("--val_freq",        default=2,              type=int)
#     p.add_argument("--val_ade_freq",    default=2,              type=int,
#                    help="Full val ADE evaluation frequency (epochs). "
#                         "Chạy toàn bộ val set, tốn ~2-5 phút/lần.")
#     p.add_argument("--full_eval_freq",  default=10,             type=int)
#     p.add_argument("--val_subset_size", default=600,            type=int)
#     p.add_argument("--output_dir",      default="runs/v22",     type=str)
#     p.add_argument("--save_interval",   default=10,             type=int)
#     p.add_argument("--metrics_csv",     default="metrics.csv",        type=str)
#     p.add_argument("--predict_csv",     default="predictions.csv",    type=str)
#     p.add_argument("--lstm_errors_npy",      default=None, type=str)
#     p.add_argument("--diffusion_errors_npy", default=None, type=str)
#     p.add_argument("--gpu_num",         default="0",            type=str)
#     p.add_argument("--delim",           default=" ")
#     p.add_argument("--skip",            default=1,              type=int)
#     p.add_argument("--min_ped",         default=1,              type=int)
#     p.add_argument("--threshold",       default=0.002,          type=float)
#     p.add_argument("--other_modal",     default="gph")
#     p.add_argument("--curriculum",      default=True,
#                    type=lambda x: x.lower() != 'false')
#     p.add_argument("--curriculum_start_len", default=4,         type=int)
#     p.add_argument("--curriculum_end_epoch", default=40,        type=int)
#     p.add_argument("--lon_flip_prob",   default=0.3,            type=float)
#     p.add_argument("--pinn_warmup_epochs", default=30,          type=int)
#     p.add_argument("--pinn_w_start",    default=0.01,           type=float)
#     p.add_argument("--pinn_w_end",      default=0.1,            type=float)
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
#                            collate_fn, num_workers):
#     n   = len(val_dataset)
#     rng = random.Random(42)
#     idx = rng.sample(range(n), min(subset_size, n))
#     return DataLoader(Subset(val_dataset, idx),
#                       batch_size=batch_size, shuffle=False,
#                       collate_fn=collate_fn, num_workers=0, drop_last=False)


# def get_curriculum_len(epoch, args) -> int:
#     if not args.curriculum:
#         return args.pred_len
#     if epoch >= args.curriculum_end_epoch:
#         return args.pred_len
#     frac = epoch / max(args.curriculum_end_epoch, 1)
#     return int(args.curriculum_start_len
#                + frac * (args.pred_len - args.curriculum_start_len))


# # ── evaluate_fast: monitor nhanh trên val subset ──────────────────────────────

# def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
#     """
#     FIX-V21-1: Slice gt → T_active để shapes khớp.
#     FIX-V22-1: StepErrorAccumulator v5 tự pad zeros → không crash.
#     FIX-V22-3: Log active_steps từ accumulator.
#     """
#     model.eval()
#     acc = StepErrorAccumulator(pred_len)
#     t0  = time.perf_counter()
#     n   = 0
#     spread_buf = []

#     with torch.no_grad():
#         for batch in loader:
#             bl = move(list(batch), device)
#             pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
#                                               ddim_steps=ode_steps)
#             # FIX-V21-1: pred.shape = [T_active, B, 2], bl[1].shape = [T_pred, B, 2]
#             T_active  = pred.shape[0]
#             gt_sliced = bl[1][:T_active]

#             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt_sliced))
#             # dist shape: [T_active, B] — accumulator v5 tự pad nếu T_active < pred_len
#             acc.update(dist)

#             # Spread: std ensemble tại step cuối
#             last_step = all_trajs[:, -1, :, :]   # [S, B, 2]
#             std_lon = last_step[:, :, 0].std(0)
#             std_lat = last_step[:, :, 1].std(0)
#             spread_km = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
#             spread_buf.append(spread_km)
#             n += 1

#     r = acc.compute()
#     r["ms_per_batch"]  = (time.perf_counter() - t0) * 1e3 / max(n, 1)
#     r["spread_72h_km"] = float(np.mean(spread_buf)) if spread_buf else 0.0
#     return r


# # ── evaluate_full_val_ade: full val ADE mỗi 2 epoch ──────────────────────────

# def evaluate_full_val_ade(model, val_loader, device, ode_steps, pred_len,
#                            fast_ensemble, metrics_csv, epoch, tag=""):
#     """
#     FIX-V22-2: Chạy toàn bộ val set, dùng pred_len đầy đủ (không curriculum
#     slice) để ADE phản ánh khả năng dự báo 72h thật sự.

#     Dùng fast_ensemble (không phải val_ensemble) để tiết kiệm thời gian.
#     Kết quả được lưu vào CSV và print ra console.

#     Returns: dict với ADE, FDE, per-horizon errors.
#     """
#     model.eval()
#     acc = StepErrorAccumulator(pred_len)
#     t0  = time.perf_counter()
#     n_batch = 0

#     with torch.no_grad():
#         for batch in val_loader:
#             bl = move(list(batch), device)

#             # QUAN TRỌNG: Tạm thời set active_pred_len = pred_len
#             # để sample() trả về T = pred_len đầy đủ, không bị curriculum cắt
#             original_active = getattr(model, 'active_pred_len', pred_len)
#             if hasattr(model, 'set_curriculum_len'):
#                 model.set_curriculum_len(pred_len)

#             pred, _, _ = model.sample(bl, num_ensemble=fast_ensemble,
#                                       ddim_steps=ode_steps)

#             # Restore curriculum len sau khi sample
#             if hasattr(model, 'set_curriculum_len'):
#                 model.set_curriculum_len(original_active)

#             # pred.shape: [pred_len, B, 2]
#             # bl[1].shape: [pred_len, B, 2]
#             T_pred = pred.shape[0]
#             gt = bl[1][:T_pred]

#             dist = haversine_km_torch(denorm_torch(pred), denorm_torch(gt))
#             acc.update(dist)
#             n_batch += 1

#     elapsed = time.perf_counter() - t0
#     r = acc.compute()

#     # Format output
#     tag_str = tag or f"val_full_ep{epoch:03d}"
#     ade_str = f"{r.get('ADE', float('nan')):.1f}"
#     fde_str = f"{r.get('FDE', float('nan')):.1f}"
#     h12_str = f"{r.get('12h', float('nan')):.0f}"
#     h24_str = f"{r.get('24h', float('nan')):.0f}"
#     h48_str = f"{r.get('48h', float('nan')):.0f}"
#     h72_str = f"{r.get('72h', float('nan')):.0f}"

#     print(f"\n{'='*64}")
#     print(f"  [FULL VAL ADE  ep={epoch}  {elapsed:.0f}s  {n_batch} batches]")
#     print(f"  ADE={ade_str} km  FDE={fde_str} km")
#     print(f"  12h={h12_str}  24h={h24_str}  48h={h48_str}  72h={h72_str} km")
#     print(f"  samples={r.get('n_samples',0)}  ens={fast_ensemble}  steps={ode_steps}")
#     print(f"{'='*64}\n")

#     # Lưu vào CSV
#     from dataclasses import fields as dc_fields
#     from utils.metrics import DatasetMetrics, save_metrics_csv
#     from datetime import datetime

#     dm = DatasetMetrics(
#         ade      = r.get("ADE", float("nan")),
#         fde      = r.get("FDE", float("nan")),
#         ugde_12h = r.get("12h", float("nan")),
#         ugde_24h = r.get("24h", float("nan")),
#         ugde_48h = r.get("48h", float("nan")),
#         ugde_72h = r.get("72h", float("nan")),
#         n_total  = r.get("n_samples", 0),
#         timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
#     )
#     save_metrics_csv(dm, metrics_csv, tag=tag_str)

#     return r


# def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
#                   metrics_csv, tag="", predict_csv=""):
#     """
#     FIX-V21-2: Sửa CLIPER error đơn vị.
#     FIX-V21-3: Bỏ transpose ens_seqs_01.
#     """
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

#                 # CLIPER error per step (FIX-V21-2)
#                 obs_b_norm = obs.cpu().numpy()[:, b, :]
#                 cliper_errors_b = np.zeros(pred_len)
#                 for h in range(pred_len):
#                     pred_cliper_norm = cliper_forecast(obs_b_norm, h + 1)
#                     pred_cliper_01 = denorm_np(pred_cliper_norm[np.newaxis])
#                     gt_01 = gd_np[h, b, :][np.newaxis]
#                     from utils.metrics import haversine_km_np
#                     cliper_errors_b[h] = float(
#                         haversine_km_np(pred_cliper_01, gt_01, unit_01deg=True)[0]
#                     )
#                 cliper_step_errors.append(cliper_errors_b)

#     if cliper_step_errors:
#         cliper_mat = np.stack(cliper_step_errors)
#         cliper_ugde_dict = {h: float(cliper_mat[:, s].mean())
#                             for h, s in HORIZON_STEPS.items()
#                             if s < cliper_mat.shape[1]}
#         ev.cliper_ugde = cliper_ugde_dict
#         print(f"  [CLIPER UGDE] 72h={cliper_ugde_dict.get(72, float('nan')):.1f} km")

#     dm = ev.compute(tag=tag)

#     try:
#         if LANDFALL_TARGETS and ens_seqs_01:
#             bss_vals = []
#             step_72  = HORIZON_STEPS.get(72, pred_len - 1)
#             for tname, t_lon, t_lat in LANDFALL_TARGETS:
#                 bv = brier_skill_score(
#                     ens_seqs_01,
#                     gt_seqs_01, min(step_72, pred_len-1),
#                     (t_lon * 10.0, t_lat * 10.0),
#                     LANDFALL_RADIUS_KM)
#                 if not math.isnan(bv):
#                     bss_vals.append(bv)
#             if bss_vals:
#                 dm.bss_mean = float(np.mean(bss_vals))
#                 print(f"  [BSS] mean={dm.bss_mean:.4f}")
#     except Exception as e:
#         print(f"  ⚠  BSS failed: {e}")

#     save_metrics_csv(dm, metrics_csv, tag=tag)
#     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# class BestModelSaver:
#     def __init__(self, patience=50, ade_tol=5.0):
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
#             print(f"  [SAVER] Patience counters reset: {reason}")

#     def update_val_loss(self, val_loss, model, out_dir, epoch, optimizer, tl):
#         if val_loss < self.best_val_loss - 1e-4:
#             self.best_val_loss = val_loss;  self.counter_loss = 0
#             torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
#                             optimizer_state=optimizer.state_dict(),
#                             train_loss=tl, val_loss=val_loss,
#                             model_version="v22-valloss"),
#                        os.path.join(out_dir, "best_model_valloss.pth"))
#         else:
#             self.counter_loss += 1

#     def update_ade(self, ade, model, out_dir, epoch, optimizer, tl, vl,
#                    min_epochs=80, source="full_val"):
#         """
#         FIX-V22-2: source parameter để log rõ ADE đến từ full val hay subset.
#         """
#         if ade < self.best_ade - self.ade_tol:
#             self.best_ade = ade;  self.counter_ade = 0
#             torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
#                             optimizer_state=optimizer.state_dict(),
#                             train_loss=tl, val_loss=vl, val_ade_km=ade,
#                             model_version="v22-FNO-Mamba-recurv"),
#                        os.path.join(out_dir, "best_model.pth"))
#             print(f"  ✅ Best ADE {ade:.1f} km  (epoch {epoch}, src={source})")
#         else:
#             self.counter_ade += 1
#             print(f"  No ADE improvement {self.counter_ade}/{self.patience}"
#                   f"  (Δ={self.best_ade-ade:.1f} km < tol={self.ade_tol} km)"
#                   f"  | Loss counter {self.counter_loss}/{self.patience}"
#                   f"  [src={source}]")

#         if epoch >= min_epochs:
#             if self.counter_ade >= self.patience and self.counter_loss >= self.patience:
#                 self.early_stop = True
#                 print(f"  ⛔ Early stop @ epoch {epoch}")
#         else:
#             if self.counter_ade >= self.patience and self.counter_loss >= self.patience:
#                 print(f"  ⚠  Would stop but min_epochs={min_epochs} not reached. Continuing...")
#                 self.counter_ade = 0;  self.counter_loss = 0


# def _load_baseline_errors(path, name):
#     if path is None:
#         print(f"\n  ⚠  WARNING: --{name.lower().replace(' ','_')}_errors_npy not provided.")
#         print(f"     Statistical comparison vs {name} will be SKIPPED.\n")
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
#     print("  TC-FlowMatching v22  |  FNO3D + Mamba + OT-CFM + PINN")
#     print("  v22 FIXES:")
#     print("    FIX-V22-1: StepErrorAccumulator crash → pad zeros, per-step count")
#     print("    FIX-V22-2: Full val ADE mỗi 2 epoch trên toàn bộ val set")
#     print("    FIX-V22-3: Log active_steps từ accumulator")
#     print("    FIX-DATA-15: GPH500 key remap '_n' → chuẩn (không 0 nữa)")
#     print("=" * 68)
#     print(f"  device               : {device}")
#     print(f"  sigma_min            : {args.sigma_min}")
#     print(f"  ctx_noise_scale      : {args.ctx_noise_scale}")
#     print(f"  initial_sample_sigma : {args.initial_sample_sigma}")
#     print(f"  ode_steps            : train={ode_train}  val={ode_val}  test={ode_test}")
#     print(f"  val_ensemble         : {args.val_ensemble}")
#     print(f"  val_ade_freq         : every {args.val_ade_freq} epochs (full val set)")
#     print(f"  patience             : {args.patience} epochs  (min_epochs={args.min_epochs})")
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

#     optimizer = optim.AdamW(model.parameters(),
#                              lr=args.g_learning_rate,
#                              weight_decay=args.weight_decay)
#     steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
#     total_steps     = steps_per_epoch * args.num_epochs
#     warmup          = steps_per_epoch * args.warmup_epochs
#     scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
#     saver  = BestModelSaver(patience=args.patience, ade_tol=5.0)
#     scaler = GradScaler('cuda', enabled=args.use_amp)

#     print("=" * 68)
#     print(f"  TRAINING  ({steps_per_epoch} steps/epoch)")
#     print("=" * 68)

#     epoch_times    = []
#     train_start    = time.perf_counter()
#     last_val_loss  = float("inf")
#     last_full_ade  = float("inf")   # FIX-V22-2: track full val ADE
#     _lr_ep30_done  = False
#     _lr_ep60_done  = False
#     _prev_ens      = 1

#     import Model.losses as _losses_mod

#     for epoch in range(args.num_epochs):
#         # Progressive ensemble
#         current_ens = 1 if epoch < 30 else (2 if epoch < 60 else args.n_train_ens)
#         model.n_train_ens = current_ens
#         effective_fast_ens = min(args.fast_ensemble, max(current_ens*2, args.fast_ensemble))

#         if current_ens != _prev_ens:
#             saver.reset_counters(
#                 f"n_train_ens {_prev_ens}→{current_ens} at epoch {epoch}")
#             _prev_ens = current_ens

#         curr_len = get_curriculum_len(epoch, args)
#         if hasattr(model, "set_curriculum_len"):
#             model.set_curriculum_len(curr_len)

#         epoch_weights = copy.copy(_BASE_WEIGHTS)
#         epoch_weights["pinn"]     = get_pinn_weight(epoch, args.pinn_warmup_epochs,
#                                                     args.pinn_w_start, args.pinn_w_end)
#         epoch_weights["velocity"] = get_velocity_weight(epoch, args.vel_warmup_epochs,
#                                                         args.vel_w_start, args.vel_w_end)
#         epoch_weights["recurv"]   = get_recurv_weight(epoch, args.recurv_warmup_epochs,
#                                                       args.recurv_w_start, args.recurv_w_end)
#         _losses_mod.WEIGHTS.update(epoch_weights)
#         if hasattr(model, 'weights'):
#             model.weights = epoch_weights
#         _losses_mod.WEIGHTS["pinn"]     = epoch_weights["pinn"]
#         _losses_mod.WEIGHTS["velocity"] = epoch_weights["velocity"]
#         _losses_mod.WEIGHTS["recurv"]   = epoch_weights["recurv"]

#         current_clip = get_grad_clip(epoch, warmup_epochs=20,
#                                      clip_start=args.grad_clip, clip_end=1.0)

#         # LR restarts
#         if epoch == 30 and not _lr_ep30_done:
#             _lr_ep30_done = True
#             warmup_steps = steps_per_epoch * 1
#             scheduler = get_cosine_schedule_with_warmup(
#                 optimizer, warmup_steps,
#                 steps_per_epoch * (args.num_epochs - 30),
#                 min_lr=5e-6)
#             saver.reset_counters("LR warm restart at epoch 30")
#             print(f"  ↺  Warm Restart LR at epoch 30")

#         if epoch == 60 and not _lr_ep60_done:
#             _lr_ep60_done = True
#             warmup_steps = steps_per_epoch * 1
#             scheduler = get_cosine_schedule_with_warmup(
#                 optimizer, warmup_steps,
#                 steps_per_epoch * (args.num_epochs - 60),
#                 min_lr=1e-6)
#             saver.reset_counters("LR warm restart at epoch 60")
#             print(f"  ↺  Warm Restart LR at epoch 60")

#         # ── Training loop ─────────────────────────────────────────────────────
#         model.train()
#         sum_loss = 0.0
#         t0 = time.perf_counter()
#         optimizer.zero_grad()
#         recurv_ratio_buf = []

#         for i, batch in enumerate(train_loader):
#             bl = move(list(batch), device)

#             if epoch == 0 and i == 0:
#                 test_env = bl[13]
#                 if test_env is not None and "gph500_mean" in test_env:
#                     gph_val = test_env["gph500_mean"]
#                     n_zero  = (gph_val == 0).sum().item()
#                     n_total = gph_val.numel()
#                     zero_pct = 100.0 * n_zero / max(n_total, 1)

#                     if torch.all(gph_val == 0):
#                         env_source = getattr(train_dataset, '_env_path_missing', None)
#                         has_csv    = bool(getattr(train_dataset, '_csv_env_lookup', {}))
#                         print("\n" + "!" * 60)
#                         if env_source is True and not has_csv:
#                             print("  ⚠️  GPH500 = 0: Env_data .npy KHÔNG TỒN TẠI")
#                             print("  → all_storms_final.csv cũng KHÔNG TÌM THẤY")
#                         elif env_source is True and has_csv:
#                             print("  ⚠️  GPH500 = 0 dù CSV fallback đã load!")
#                             print("  → Có thể storm_name format mismatch")
#                         else:
#                             print("  ⚠️  GPH500 = 0 dù Env_data tồn tại!")
#                             print("  → FIX-DATA-15: key '_n' chưa được remap?")
#                             print("  → Kiểm tra trajectoriesWithMe_unet_training v18")
#                         print("!" * 60 + "\n")
#                     elif zero_pct > 50.0:
#                         print(f"  ⚠️  GPH500 {zero_pct:.1f}% = 0 "
#                               f"(mean non-zero: {gph_val[gph_val != 0].mean().item():.4f})")
#                     else:
#                         print(f"  ✅ GPH500 OK "
#                               f"(mean={gph_val.mean().item():.4f}, "
#                               f"zero={zero_pct:.1f}% = sentinel rows)")

#             with autocast(device_type='cuda', enabled=args.use_amp):
#                 bd = model.get_loss_breakdown(bl)

#             loss_to_backpass = bd["total"] / max(args.grad_accum, 1)
#             scaler.scale(loss_to_backpass).backward()

#             if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), current_clip)
#                 scaler.step(optimizer)
#                 scaler.update()
#                 scheduler.step()
#                 optimizer.zero_grad()

#             sum_loss += bd["total"].item()

#             if "recurv_ratio" in bd:
#                 recurv_ratio_buf.append(bd["recurv_ratio"])

#             if i % 20 == 0:
#                 lr  = optimizer.param_groups[0]["lr"]
#                 rr  = bd.get("recurv_ratio", 0.0)
#                 elapsed = time.perf_counter() - t0
#                 print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
#                       f"  loss={bd['total'].item():.3f}"
#                       f"  fm={bd.get('fm',0):.2f}"
#                       f"  vel={bd.get('velocity',0):.6f}"
#                       f"  pinn={bd.get('pinn', 0):.6f}"
#                       f"  recurv={bd.get('recurv',0):.3f}"
#                       f"  rr={rr:.2f}"
#                       f"  pinn_w={epoch_weights['pinn']:.3f}"
#                       f"  clip={current_clip:.1f}"
#                       f"  ens={current_ens}  len={curr_len}"
#                       f"  lr={lr:.2e}  t={elapsed:.0f}s")

#         ep_s  = time.perf_counter() - t0
#         epoch_times.append(ep_s)
#         avg_t = sum_loss / len(train_loader)
#         mean_rr = float(np.mean(recurv_ratio_buf)) if recurv_ratio_buf else 0.0

#         # ── Val loss ───────────────────────────────────────────────────────────
#         if epoch % args.val_freq == 0:
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
#             saver.update_val_loss(last_val_loss, model, args.output_dir,
#                                   epoch, optimizer, avg_t)
#             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
#                   f"  rr={mean_rr:.2f}"
#                   f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
#                   f"  ens={current_ens}  len={curr_len}"
#                   f"  recurv_w={epoch_weights['recurv']:.2f}")
#         else:
#             print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
#                   f"  val={last_val_loss:.3f}(cached)"
#                   f"  rr={mean_rr:.2f}  t={ep_s:.0f}s")

#         # ── Fast ADE trên val subset (mỗi epoch, dùng để monitor) ─────────────
#         t_ade = time.perf_counter()
#         m_fast = evaluate_fast(model, val_subset_loader, device,
#                                ode_train, args.pred_len, effective_fast_ens)
#         t_ade_s = time.perf_counter() - t_ade

#         spread_72h    = m_fast.get("spread_72h_km", 0.0)
#         active_steps  = m_fast.get("active_steps", args.pred_len)
#         collapse_warn = "  ⚠️ COLLAPSE!" if spread_72h < 10.0 else ""

#         print(f"  [FAST-ADE ep{epoch} {t_ade_s:.0f}s]"
#               f"  ADE={m_fast['ADE']:.1f} km  FDE={m_fast['FDE']:.1f} km"
#               f"  12h={m_fast.get('12h', float('nan')):.0f}"
#               f"  24h={m_fast.get('24h', float('nan')):.0f}"
#               f"  72h={m_fast.get('72h', float('nan')):.0f} km"
#               f"  spread={spread_72h:.1f} km"
#               f"  active_steps={active_steps}/{args.pred_len}"
#               f"  (subset, ens={effective_fast_ens})"
#               f"{collapse_warn}")

#         # ── FIX-V22-2: Full val ADE mỗi val_ade_freq epoch ───────────────────
#         ade_for_saver = m_fast["ADE"]
#         ade_source    = "subset"

#         if epoch % args.val_ade_freq == 0:
#             t_full_ade = time.perf_counter()
#             try:
#                 r_full = evaluate_full_val_ade(
#                     model, val_loader, device,
#                     ode_steps   = ode_train,   # dùng train steps để nhanh hơn val
#                     pred_len    = args.pred_len,
#                     fast_ensemble = effective_fast_ens,
#                     metrics_csv = metrics_csv,
#                     epoch       = epoch,
#                     tag         = f"val_full_ep{epoch:03d}",
#                 )
#                 last_full_ade = r_full.get("ADE", float("inf"))
#                 ade_for_saver = last_full_ade
#                 ade_source    = "full_val"
#             except Exception as e:
#                 print(f"  ⚠  Full val ADE failed: {e}")
#                 import traceback; traceback.print_exc()

#         # Dùng ADE (full val nếu có, subset nếu không) để update saver
#         saver.update_ade(ade_for_saver, model, args.output_dir, epoch,
#                          optimizer, avg_t, last_val_loss,
#                          min_epochs=args.min_epochs,
#                          source=ade_source)

#         # ── Full eval (TCEvaluator đầy đủ 4-tier) ─────────────────────────────
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

#         if (epoch+1) % args.save_interval == 0:
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
#             model_name   = "FM+PINN-v22",
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

#         _, cliper_errs = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
#         persist_errs   = persistence_errors(obs_seqs, gt_seqs, args.pred_len)
#         fmpinn_per_seq = np.array([seq_ade_km(np.array(pp), np.array(g))
#                                     for pp, g in zip(pred_seqs, gt_seqs)])

#         np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
#         np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
#         np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))

#         lstm_per_seq      = _load_baseline_errors(args.lstm_errors_npy,      "LSTM")
#         diffusion_per_seq = _load_baseline_errors(args.diffusion_errors_npy, "Diffusion")
#         if lstm_per_seq is not None:
#             np.save(os.path.join(stat_dir, "lstm.npy"), lstm_per_seq)
#         if diffusion_per_seq is not None:
#             np.save(os.path.join(stat_dir, "diffusion.npy"), diffusion_per_seq)

#         _dummy = np.array([float("nan")])
#         run_all_tests(
#             fmpinn_ade    = fmpinn_per_seq,
#             cliper_ade    = cliper_errs.mean(1),
#             lstm_ade      = lstm_per_seq if lstm_per_seq is not None else _dummy,
#             diffusion_ade = diffusion_per_seq if diffusion_per_seq is not None else _dummy,
#             persist_ade   = persist_errs.mean(1),
#             out_dir       = stat_dir)

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
#             paired_tests(fmpinn_per_seq, cliper_errs.mean(1),  "FM+PINN vs CLIPER",      5),
#             paired_tests(fmpinn_per_seq, persist_errs.mean(1), "FM+PINN vs Persistence", 5),
#         ]
#         if lstm_per_seq is not None:
#             stat_rows.append(paired_tests(fmpinn_per_seq, lstm_per_seq, "FM+PINN vs LSTM", 5))
#         if diffusion_per_seq is not None:
#             stat_rows.append(paired_tests(fmpinn_per_seq, diffusion_per_seq, "FM+PINN vs Diffusion", 5))

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
#             fh.write(f"\n\nmodel_version         : FM+PINN v22\n")
#             fh.write(f"sigma_min             : {args.sigma_min}\n")
#             fh.write(f"ctx_noise_scale       : {args.ctx_noise_scale}\n")
#             fh.write(f"initial_sample_sigma  : {args.initial_sample_sigma}\n")
#             fh.write(f"ode_steps_test        : {ode_test}\n")
#             fh.write(f"eval_ensemble         : {final_ens}\n")
#             fh.write(f"train_time_h          : {total_train_h:.2f}\n")
#             fh.write(f"n_params_M            : "
#                      f"{sum(p.numel() for p in model.parameters())/1e6:.2f}\n")

#     avg_ep = sum(epoch_times)/len(epoch_times) if epoch_times else 0
#     print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
#     print(f"  Best val loss  : {saver.best_val_loss:.4f}")
#     print(f"  Avg epoch time : {avg_ep:.0f}s")
#     print(f"  Total training : {total_train_h:.2f}h")
#     print(f"  Tables dir     : {tables_dir}")
#     print("=" * 68)


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(42);  torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
#     main(args)
"""
Model/flow_matching_model.py  ── v23
==========================================
FIXES vs v21:

  FIX-M18  [CURRICULUM REMOVED] set_curriculum_len() vẫn giữ để backward
           compat nhưng KHÔNG được gọi từ trainer nữa. active_pred_len
           luôn = pred_len. evaluate_full_val_ade không cần restore nữa.

  FIX-M19  get_loss_breakdown(): nhận thêm step_weight_alpha parameter
           và truyền vào compute_total_loss() → fm_afcrps_loss() sử dụng
           soft weighting thay curriculum len-slicing.

  FIX-M20  get_loss_breakdown(): truyền all_trajs vào compute_total_loss()
           để tính ensemble_spread_loss. Giúp kiểm soát spread tăng quá mức.

  FIX-M21  _physics_correct(): tăng n_steps=5 (từ 3), giảm lr=0.002 (từ
           0.005) để physics correction ổn định hơn và ít overshoot.

  FIX-M22  sample(): initial_sample_sigma=0.1 (set từ constructor) đã fix
           spread. Thêm post-sampling clip chặt hơn [-3.0, 3.0] cho cả lon
           và lat (từ [-5.0, 5.0] cho lon).

Kept from v21:
  FIX-M17  _physics_correct với torch.enable_grad()
  FIX-M11..M16 OT-CFM, beta drift, env_data, physics scale
"""
from __future__ import annotations

import csv
import math
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
# Lấy đường dẫn của thư mục cha (thư mục gốc TC_FM)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)
import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.FNO3D_encoder import FNO3DEncoder
from Model.mamba_encoder import DataEncoder1D_Mamba as DataEncoder1D
from Model.env_net_transformer_gphsplit import Env_net
from Model.losses import (
    compute_total_loss, fm_physics_consistency_loss, WEIGHTS,
    pinn_speed_constraint,
)


def _denorm_to_deg(traj_norm: torch.Tensor) -> torch.Tensor:
    """Normalised → degrees. Handles [T, B, 2] and [B, 2]."""
    out = traj_norm.clone()
    out[..., 0] = (traj_norm[..., 0] * 50.0 + 1800.0) / 10.0
    out[..., 1] = (traj_norm[..., 1] * 50.0) / 10.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  VelocityField
# ══════════════════════════════════════════════════════════════════════════════

class VelocityField(nn.Module):
    """
    OT-CFM velocity field v_θ(x_t, t, context).
    Architecture: DataEncoder1D (Mamba) + FNO3D + Env-T-Net → Transformer decoder.
    Physics-guided: v_total = v_neural + sigmoid(w_physics) * v_beta_drift.
    """

    def __init__(
        self,
        pred_len:   int   = 12,
        obs_len:    int   = 8,
        ctx_dim:    int   = 256,
        sigma_min:  float = 0.02,
        unet_in_ch: int   = 13,
    ):
        super().__init__()
        self.pred_len  = pred_len
        self.obs_len   = obs_len
        self.sigma_min = sigma_min

        self.spatial_enc = FNO3DEncoder(
            in_channel   = unet_in_ch,
            out_channel  = 1,
            d_model      = 64,
            n_layers     = 4,
            modes_t      = 4,
            modes_h      = 4,
            modes_w      = 4,
            spatial_down = 32,
            dropout      = 0.05,
        )

        self.bottleneck_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.bottleneck_proj = nn.Linear(128, 128)
        self.decoder_proj    = nn.Linear(1, 16)

        self.enc_1d = DataEncoder1D(
            in_1d       = 4,
            feat_3d_dim = 128,
            mlp_h       = 64,
            lstm_hidden = 128,
            lstm_layers = 3,
            dropout     = 0.1,
            d_state     = 16,
        )

        self.env_enc = Env_net(obs_len=obs_len, d_model=64)

        self.ctx_fc1  = nn.Linear(128 + 64 + 16, 512)
        self.ctx_ln   = nn.LayerNorm(512)
        self.ctx_drop = nn.Dropout(0.15)
        self.ctx_fc2  = nn.Linear(512, ctx_dim)

        self.time_fc1 = nn.Linear(256, 512)
        self.time_fc2 = nn.Linear(512, 256)

        self.traj_embed = nn.Linear(4, 256)
        self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 256) * 0.02)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=256, nhead=8, dim_feedforward=1024,
                dropout=0.15, activation="gelu", batch_first=True,
            ),
            num_layers=4,
        )
        self.out_fc1 = nn.Linear(256, 512)
        self.out_fc2 = nn.Linear(512, 4)

        self.physics_scale = nn.Parameter(torch.ones(4) * 0.5)

    def _time_emb(self, t: torch.Tensor, dim: int = 256) -> torch.Tensor:
        half = dim // 2
        freq = torch.exp(
            torch.arange(half, dtype=torch.float32, device=t.device)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return F.pad(emb, (0, dim % 2))

    def _context(self, batch_list: List) -> torch.Tensor:
        obs_traj  = batch_list[0]
        obs_Me    = batch_list[7]
        image_obs = batch_list[11]
        env_data  = batch_list[13]

        if image_obs.dim() == 4:
            image_obs = image_obs.unsqueeze(1)

        expected_ch = self.spatial_enc.in_channel
        if image_obs.shape[1] == 1 and expected_ch != 1:
            image_obs = image_obs.expand(-1, expected_ch, -1, -1, -1)

        e_3d_bot, e_3d_dec = self.spatial_enc.encode(image_obs)
        T_obs = obs_traj.shape[0]

        e_3d_s = self.bottleneck_pool(e_3d_bot).squeeze(-1).squeeze(-1)
        e_3d_s = e_3d_s.permute(0, 2, 1)
        e_3d_s = self.bottleneck_proj(e_3d_s)

        T_bot = e_3d_s.shape[1]
        if T_bot != T_obs:
            e_3d_s = F.interpolate(
                e_3d_s.permute(0, 2, 1), size=T_obs,
                mode="linear", align_corners=False,
            ).permute(0, 2, 1)

        f_spatial_raw = e_3d_dec.mean(dim=(2, 3, 4))
        f_spatial     = self.decoder_proj(f_spatial_raw)

        obs_in = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        h_t    = self.enc_1d(obs_in, e_3d_s)

        e_env, _, _ = self.env_enc(env_data, image_obs)

        raw = torch.cat([h_t, e_env, f_spatial], dim=-1)
        raw = F.gelu(self.ctx_ln(self.ctx_fc1(raw)))
        return raw

    def _apply_ctx_head(self, raw: torch.Tensor,
                        noise_scale: float = 0.0) -> torch.Tensor:
        if noise_scale > 0.0:
            raw = raw + torch.randn_like(raw) * noise_scale
        return self.ctx_fc2(self.ctx_drop(raw))

    def _beta_drift_velocity(self, x_t: torch.Tensor) -> torch.Tensor:
        """Beta drift in normalised state space. x_t: [B, T, 4]."""
        OMEGA_val  = 7.2921e-5
        R_val      = 6.371e6
        DT         = 6 * 3600.0
        M_PER_NORM = 5.0 * 111.0 * 1000.0

        lat_norm = x_t[:, :, 1]
        lat_deg  = lat_norm * 5.0
        lat_rad  = torch.deg2rad(lat_deg.clamp(-85, 85))

        beta   = 2 * OMEGA_val * torch.cos(lat_rad) / R_val
        R_tc   = 3e5
        v_lon  = -beta * R_tc ** 2 / 2
        v_lat  =  beta * R_tc ** 2 / 4

        v_lon_norm = v_lon * DT / M_PER_NORM
        v_lat_norm = v_lat * DT / M_PER_NORM

        v_phys = torch.zeros_like(x_t)
        v_phys[:, :, 0] = v_lon_norm
        v_phys[:, :, 1] = v_lat_norm
        return v_phys

    def _decode(self, x_t: torch.Tensor, t: torch.Tensor,
                ctx: torch.Tensor) -> torch.Tensor:
        t_emb = F.gelu(self.time_fc1(self._time_emb(t, 256)))
        t_emb = self.time_fc2(t_emb)

        T_seq  = min(x_t.size(1), self.pos_enc.shape[1])
        x_emb  = (self.traj_embed(x_t[:, :T_seq, :])
                  + self.pos_enc[:, :T_seq, :]
                  + t_emb.unsqueeze(1))
        memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

        v_neural = self.out_fc2(F.gelu(self.out_fc1(
            self.transformer(x_emb, memory)
        )))  # [B, T, 4]

        with torch.no_grad():
            v_phys = self._beta_drift_velocity(x_t[:, :T_seq, :])

        scale = torch.sigmoid(self.physics_scale) * 2.0
        return v_neural + scale * v_phys

    def forward(self, x_t, t, batch_list):
        raw = self._context(batch_list)
        ctx = self._apply_ctx_head(raw, noise_scale=0.0)
        return self._decode(x_t, t, ctx)

    def forward_with_ctx(self, x_t, t, raw_ctx, noise_scale: float = 0.0):
        ctx = self._apply_ctx_head(raw_ctx, noise_scale=noise_scale)
        return self._decode(x_t, t, ctx)


# ══════════════════════════════════════════════════════════════════════════════
#  TCFlowMatching
# ══════════════════════════════════════════════════════════════════════════════

class TCFlowMatching(nn.Module):
    """TC trajectory prediction via OT-CFM + Physics-guided velocity field."""

    def __init__(
        self,
        pred_len:             int   = 12,
        obs_len:              int   = 8,
        sigma_min:            float = 0.02,
        n_train_ens:          int   = 4,
        unet_in_ch:           int   = 13,
        ctx_noise_scale:      float = 0.02,   # FIX-T23-5: 0.02 default
        initial_sample_sigma: float = 0.1,    # FIX-T23-4: 0.1 default
        **kwargs,
    ):
        super().__init__()
        self.pred_len             = pred_len
        self.obs_len              = obs_len
        self.sigma_min            = sigma_min
        self.n_train_ens          = n_train_ens
        self.active_pred_len      = pred_len   # FIX-M18: always full pred_len
        self.ctx_noise_scale      = ctx_noise_scale
        self.initial_sample_sigma = initial_sample_sigma
        self.net = VelocityField(
            pred_len   = pred_len,
            obs_len    = obs_len,
            sigma_min  = sigma_min,
            unet_in_ch = unet_in_ch,
        )

    def set_curriculum_len(self, active_len: int) -> None:
        """
        FIX-M18: Kept for backward compat but NO-OP in v23.
        Curriculum is removed. active_pred_len is always pred_len.
        """
        # self.active_pred_len = max(1, min(active_len, self.pred_len))
        pass  # no-op

    @staticmethod
    def _to_rel(traj_gt, Me_gt, last_pos, last_Me):
        return torch.cat(
            [traj_gt - last_pos.unsqueeze(0),
             Me_gt   - last_Me.unsqueeze(0)],
            dim=-1,
        ).permute(1, 0, 2)

    @staticmethod
    def _to_abs(rel, last_pos, last_Me):
        d = rel.permute(1, 0, 2)
        return (
            last_pos.unsqueeze(0) + d[:, :, :2],
            last_Me.unsqueeze(0)  + d[:, :, 2:],
        )

    def _cfm_noisy(self, x1):
        B, device = x1.shape[0], x1.device
        sm  = self.sigma_min
        x0  = torch.randn_like(x1) * sm
        t   = torch.rand(B, device=device)
        te  = t.view(B, 1, 1)
        x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
        denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
        target_vel = (x1 - (1.0 - sm) * x_t) / denom
        return x_t, t, te, denom, target_vel

    @staticmethod
    def _intensity_weights(obs_Me: torch.Tensor) -> torch.Tensor:
        wind_norm = obs_Me[-1, :, 1].detach()
        w = torch.where(wind_norm < 0.1, torch.full_like(wind_norm, 0.5),
            torch.where(wind_norm < 0.3, torch.full_like(wind_norm, 0.8),
            torch.where(wind_norm < 0.6, torch.full_like(wind_norm, 1.0),
                        torch.full_like(wind_norm, 1.5))))
        return w / w.mean().clamp(min=1e-6)

    @staticmethod
    def _lon_flip_aug(batch_list: List, p: float = 0.3) -> List:
        if torch.rand(1).item() > p:
            return batch_list
        aug = list(batch_list)
        for idx in [0, 1, 2, 3]:
            t = aug[idx]
            if torch.is_tensor(t) and t.shape[-1] >= 1:
                t = t.clone()
                t[..., 0] = -t[..., 0]
                aug[idx] = t
        return aug

    def get_loss(self, batch_list: List,
                 step_weight_alpha: float = 0.0) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list, step_weight_alpha)["total"]

    def get_loss_breakdown(self, batch_list: List,
                           step_weight_alpha: float = 0.0) -> Dict:
        """
        FIX-M19: Nhận step_weight_alpha, truyền vào compute_total_loss.
        FIX-M20: Truyền all_trajs để tính ensemble_spread_loss.
        """
        batch_list = self._lon_flip_aug(batch_list, p=0.3)

        traj_gt  = batch_list[1]
        Me_gt    = batch_list[8]
        obs_t    = batch_list[0]
        obs_Me   = batch_list[7]

        try:
            env_data = batch_list[13]
        except (IndexError, TypeError):
            env_data = None

        # FIX-M18: NO curriculum slicing. Always use full pred_len.
        lp, lm = obs_t[-1], obs_Me[-1]
        x1 = self._to_rel(traj_gt, Me_gt, lp, lm)

        raw_ctx     = self.net._context(batch_list)
        intensity_w = self._intensity_weights(obs_Me)

        x_t, t, te, denom, _ = self._cfm_noisy(x1)
        pred_vel = self.net.forward_with_ctx(x_t, t, raw_ctx, noise_scale=0.0)

        # Ensemble samples for AFCRPS + spread penalty
        samples: List[torch.Tensor] = []
        for _ in range(self.n_train_ens):
            xt_s, ts, _, dens_s, _ = self._cfm_noisy(x1)
            pv_s  = self.net.forward_with_ctx(xt_s, ts, raw_ctx, noise_scale=0.0)
            x1_s  = xt_s + dens_s * pv_s   # OT-CFM
            pa_s, _ = self._to_abs(x1_s, lp, lm)
            samples.append(pa_s)
        pred_samples = torch.stack(samples)   # [S, T, B, 2]

        # FIX-M20: all_trajs for spread penalty
        all_trajs_4d = pred_samples   # [S, T, B, 2]

        l_fm_physics = fm_physics_consistency_loss(
            pred_samples, gt_norm=traj_gt, last_pos=lp)

        x1_pred = x_t + denom * pred_vel
        pred_abs, _ = self._to_abs(x1_pred, lp, lm)

        pred_abs_deg = _denorm_to_deg(pred_abs)
        traj_gt_deg  = _denorm_to_deg(traj_gt)
        ref_deg      = _denorm_to_deg(lp)

        breakdown = compute_total_loss(
            pred_abs           = pred_abs_deg,
            gt                 = traj_gt_deg,
            ref                = ref_deg,
            batch_list         = batch_list,
            pred_samples       = pred_samples,
            gt_norm            = traj_gt,
            weights            = WEIGHTS,
            intensity_w        = intensity_w,
            env_data           = env_data,
            step_weight_alpha  = step_weight_alpha,   # FIX-M19
            all_trajs          = all_trajs_4d,         # FIX-M20
        )

        fm_phys_w = WEIGHTS.get("fm_physics", 0.3)
        breakdown["total"]      = breakdown["total"] + fm_phys_w * l_fm_physics
        breakdown["fm_physics"] = l_fm_physics.item()

        return breakdown

    # ── sample() ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        batch_list: List,
        num_ensemble: int = 50,
        ddim_steps:   int = 20,
        predict_csv:  Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        FIX-T23-3: ddim_steps default 20 (từ 10).
        FIX-M22: tighter clip [-3.0, 3.0] cho cả lon và lat.
        Returns:
            pred_mean:  [T, B, 2] mean track (normalised)
            me_mean:    [T, B, 2] mean intensity
            all_trajs:  [S, T, B, 2] all ensemble members
        """
        lp  = batch_list[0][-1]   # [B, 2]
        lm  = batch_list[7][-1]   # [B, 2]
        B   = lp.shape[0]
        device = lp.device
        T   = self.pred_len   # FIX-M18: always full pred_len
        dt  = 1.0 / max(ddim_steps, 1)

        raw_ctx = self.net._context(batch_list)

        traj_s: List[torch.Tensor] = []
        me_s:   List[torch.Tensor] = []

        for k in range(num_ensemble):
            # FIX-T23-4: initial_sample_sigma=0.1 (set in constructor)
            x_t = torch.randn(B, T, 4, device=device) * self.initial_sample_sigma

            # DDIM Euler integration
            for step in range(ddim_steps):
                t_b = torch.full((B,), step * dt, device=device)
                ns  = self.ctx_noise_scale if step == 0 else 0.0
                vel = self.net.forward_with_ctx(x_t, t_b, raw_ctx, noise_scale=ns)
                x_t = x_t + dt * vel

            # Physics correction
            x_t = self._physics_correct(x_t, lp, lm, n_steps=5, lr=0.002)

            # FIX-M22: tighter clip
            x_t = x_t.clamp(-3.0, 3.0)

            tr, me = self._to_abs(x_t, lp, lm)
            traj_s.append(tr)
            me_s.append(me)

        all_trajs = torch.stack(traj_s)   # [S, T, B, 2]
        all_me    = torch.stack(me_s)
        pred_mean = all_trajs.mean(0)
        me_mean   = all_me.mean(0)

        if predict_csv:
            self._write_predict_csv(predict_csv, pred_mean, all_trajs)

        return pred_mean, me_mean, all_trajs

    # ── Physics correction ────────────────────────────────────────────────────

    def _physics_correct(
        self,
        x_pred: torch.Tensor,
        last_pos: torch.Tensor,
        last_Me:  torch.Tensor,
        n_steps:  int   = 5,    # FIX-M21: 5 (từ 3)
        lr:       float = 0.002, # FIX-M21: 0.002 (từ 0.005)
    ) -> torch.Tensor:
        """
        FIX-M17: torch.enable_grad() inside no_grad context.
        FIX-M21: n_steps=5, lr=0.002 for more stable correction.
        """
        with torch.enable_grad():
            x = x_pred.detach().requires_grad_(True)
            optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)

            for _ in range(n_steps):
                optimizer.zero_grad()
                pred_abs, _ = self._to_abs(x, last_pos, last_Me)
                pred_deg    = _denorm_to_deg(pred_abs)

                l_speed = self._pinn_speed_constraint(pred_deg)
                l_accel = self._pinn_beta_plane_simplified(pred_deg)

                physics_loss = l_speed + 0.3 * l_accel
                physics_loss.backward()

                torch.nn.utils.clip_grad_norm_([x], max_norm=0.05)
                optimizer.step()

        return x.detach()

    @staticmethod
    def _pinn_speed_constraint(pred_deg: torch.Tensor) -> torch.Tensor:
        if pred_deg.shape[0] < 2:
            return pred_deg.new_zeros(())
        dt_deg  = pred_deg[1:] - pred_deg[:-1]
        lat_rad = torch.deg2rad(pred_deg[:-1, :, 1])
        cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
        dx_km   = dt_deg[:, :, 0] * cos_lat * 111.0
        dy_km   = dt_deg[:, :, 1] * 111.0
        speed   = torch.sqrt(dx_km ** 2 + dy_km ** 2)
        return F.relu(speed - 600.0).pow(2).mean()

    @staticmethod
    def _pinn_beta_plane_simplified(pred_deg: torch.Tensor) -> torch.Tensor:
        if pred_deg.shape[0] < 3:
            return pred_deg.new_zeros(())
        v = pred_deg[1:] - pred_deg[:-1]
        a = v[1:] - v[:-1]
        lat_rad = torch.deg2rad(pred_deg[1:-1, :, 1])
        cos_lat = torch.cos(lat_rad).clamp(min=1e-4)
        a_lon_km = a[:, :, 0] * cos_lat * 111.0
        a_lat_km = a[:, :, 1] * 111.0
        max_accel = 50.0
        violation = F.relu(torch.sqrt(a_lon_km**2 + a_lat_km**2) - max_accel)
        return violation.pow(2).mean() * 0.1

    @staticmethod
    def _write_predict_csv(csv_path: str, traj_mean: torch.Tensor,
                           all_trajs: torch.Tensor) -> None:
        import numpy as np
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        T, B, _ = traj_mean.shape
        S       = all_trajs.shape[0]

        mean_lon = ((traj_mean[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
        mean_lat = ((traj_mean[..., 1] * 50.0) / 10.0).cpu().numpy()
        all_lon  = ((all_trajs[..., 0] * 50.0 + 1800.0) / 10.0).cpu().numpy()
        all_lat  = ((all_trajs[..., 1] * 50.0) / 10.0).cpu().numpy()

        fields   = ["timestamp", "batch_idx", "step_idx", "lead_h",
                    "lon_mean_deg", "lat_mean_deg",
                    "lon_std_deg", "lat_std_deg", "ens_spread_km"]
        write_hdr = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if write_hdr:
                w.writeheader()
            for b in range(B):
                for k in range(T):
                    dlat   = all_lat[:, k, b] - mean_lat[k, b]
                    dlon   = (all_lon[:, k, b] - mean_lon[k, b]) * math.cos(
                        math.radians(mean_lat[k, b]))
                    spread = float(np.sqrt((dlat**2 + dlon**2).mean()) * 111.0)
                    w.writerow(dict(
                        timestamp     = ts,
                        batch_idx     = b,
                        step_idx      = k,
                        lead_h        = (k + 1) * 6,
                        lon_mean_deg  = f"{mean_lon[k,b]:.4f}",
                        lat_mean_deg  = f"{mean_lat[k,b]:.4f}",
                        lon_std_deg   = f"{all_lon[:,k,b].std():.4f}",
                        lat_std_deg   = f"{all_lat[:,k,b].std():.4f}",
                        ens_spread_km = f"{spread:.2f}",
                    ))
        print(f"  Predictions → {csv_path}  (B={B}, T={T}, S={S})")


# Backward-compat alias
TCDiffusion = TCFlowMatching
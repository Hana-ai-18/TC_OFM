
"""
scripts/train_flowmatching.py  ── v20
======================================
FIXES vs v19:

FIX-V20-1  [ENSEMBLE COLLAPSE FIX] Model v15 sửa ensemble collapse bằng
           ctx noise injection + initial_sample_sigma lớn hơn. Training
           script cần pass ctx_noise_scale và initial_sample_sigma vào
           TCFlowMatching constructor.
           Thêm args: --ctx_noise_scale (default 0.05)
                      --initial_sample_sigma (default 0.3)

FIX-V20-2  [EARLY STOP] Giữ nguyên FIX-V19-1: ADE evaluated mỗi epoch.
           counter_ade = số epoch thực không cải thiện (đúng semantic).

FIX-V20-3  [SPREAD MONITOR] In ensemble spread (1σ km) mỗi epoch trong
           evaluate_fast để detect collapse sớm. Nếu spread < 10 km
           tại 72h thì warning.

Kept from v19:
    FIX-V19-1..2 (ADE mỗi epoch, counter semantic)
    FIX-V18-1..4 (cliper shape, patience reset, denorm)
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
from Model.flow_matching_model import TCFlowMatching
from Model.utils import get_cosine_schedule_with_warmup
from Model.losses import WEIGHTS as _BASE_WEIGHTS
from utils.metrics import (
    TCEvaluator, StepErrorAccumulator,
    save_metrics_csv, haversine_km_torch, denorm_torch, HORIZON_STEPS,
    cliper_forecast, LANDFALL_TARGETS, LANDFALL_RADIUS_KM,
    brier_skill_score,
)
from utils.evaluation_tables import (
    ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
    export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
    DEFAULT_COMPUTE, paired_tests, persistence_errors, cliper_errors,
)
from scripts.statistical_tests import run_all_tests


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_km_np(pred_deg: np.ndarray, gt_deg: np.ndarray) -> np.ndarray:
    pred_deg = np.atleast_2d(pred_deg)
    gt_deg   = np.atleast_2d(gt_deg)
    R = 6371.0
    lon1, lat1 = np.radians(pred_deg[:, 0]), np.radians(pred_deg[:, 1])
    lon2, lat2 = np.radians(gt_deg[:, 0]),   np.radians(gt_deg[:, 1])
    dlon = lon2 - lon1;  dlat = lat2 - lat1
    a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def denorm_deg_np(arr_norm: np.ndarray) -> np.ndarray:
    arr_norm = np.atleast_2d(arr_norm)
    out = arr_norm.copy()
    out[:, 0] = (arr_norm[:, 0] * 50.0 + 1800.0) / 10.0
    out[:, 1] = (arr_norm[:, 1] * 50.0) / 10.0
    return out


def seq_ade_km(pred_norm: np.ndarray, gt_norm: np.ndarray) -> float:
    return float(haversine_km_np(denorm_deg_np(pred_norm),
                                  denorm_deg_np(gt_norm)).mean())


# ── Adaptive weight schedules ─────────────────────────────────────────────────

def get_pinn_weight(epoch, warmup_epochs=30, w_start=0.01, w_end=0.1):
    if epoch >= warmup_epochs:
        return w_end
    return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


def get_velocity_weight(epoch, warmup_epochs=20, w_start=0.5, w_end=1.5):
    if epoch >= warmup_epochs:
        return w_end
    return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


def get_recurv_weight(epoch, warmup_epochs=10, w_start=0.3, w_end=1.0):
    if epoch >= warmup_epochs:
        return w_end
    return w_start + (epoch / max(warmup_epochs-1, 1)) * (w_end - w_start)


def get_grad_clip(epoch, warmup_epochs=20, clip_start=2.0, clip_end=1.0):
    if epoch >= warmup_epochs:
        return clip_end
    return clip_start - (epoch / max(warmup_epochs-1, 1)) * (clip_start - clip_end)


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
    p.add_argument("--grad_clip",       default=2.0,            type=float)
    p.add_argument("--grad_accum",      default=2,              type=int)
    p.add_argument("--patience",        default=50,             type=int)
    p.add_argument("--min_epochs",      default=80,             type=int)
    p.add_argument("--n_train_ens",     default=6,              type=int)
    p.add_argument("--use_amp",         action="store_true")
    p.add_argument("--num_workers",     default=2,              type=int)
    p.add_argument("--sigma_min",       default=0.05,           type=float)
    # FIX-V20-1: ensemble collapse params
    p.add_argument("--ctx_noise_scale",      default=0.05, type=float,
                   help="Gaussian noise injected into raw_ctx per ensemble member at inference")
    p.add_argument("--initial_sample_sigma", default=0.3,  type=float,
                   help="Initial noise std for ODE sampling (must >> sigma_min for spread)")
    p.add_argument("--ode_steps_train", default=20,             type=int)
    p.add_argument("--ode_steps_val",   default=30,             type=int)
    p.add_argument("--ode_steps_test",  default=50,             type=int)
    p.add_argument("--ode_steps",       default=None,           type=int)
    p.add_argument("--val_ensemble",    default=30,             type=int)
    p.add_argument("--fast_ensemble",   default=8,              type=int)
    p.add_argument("--fno_modes_h",     default=4,              type=int)
    p.add_argument("--fno_modes_t",     default=4,              type=int)
    p.add_argument("--fno_layers",      default=4,              type=int)
    p.add_argument("--fno_d_model",     default=32,             type=int)
    p.add_argument("--fno_spatial_down",default=32,             type=int)
    p.add_argument("--mamba_d_state",   default=16,             type=int)
    p.add_argument("--val_loss_freq",   default=2,              type=int)
    p.add_argument("--val_freq",        default=2,              type=int)
    p.add_argument("--full_eval_freq",  default=10,             type=int)
    p.add_argument("--val_subset_size", default=600,            type=int)
    p.add_argument("--output_dir",      default="runs/v20",     type=str)
    p.add_argument("--save_interval",   default=10,             type=int)
    p.add_argument("--metrics_csv",     default="metrics.csv",        type=str)
    p.add_argument("--predict_csv",     default="predictions.csv",    type=str)
    p.add_argument("--lstm_errors_npy",      default=None, type=str)
    p.add_argument("--diffusion_errors_npy", default=None, type=str)
    p.add_argument("--gpu_num",         default="0",            type=str)
    p.add_argument("--delim",           default=" ")
    p.add_argument("--skip",            default=1,              type=int)
    p.add_argument("--min_ped",         default=1,              type=int)
    p.add_argument("--threshold",       default=0.002,          type=float)
    p.add_argument("--other_modal",     default="gph")
    p.add_argument("--curriculum",      default=True,
                   type=lambda x: x.lower() != 'false')
    p.add_argument("--curriculum_start_len", default=4,         type=int)
    p.add_argument("--curriculum_end_epoch", default=40,        type=int)
    p.add_argument("--lon_flip_prob",   default=0.3,            type=float)
    p.add_argument("--pinn_warmup_epochs", default=30,          type=int)
    p.add_argument("--pinn_w_start",    default=0.01,           type=float)
    p.add_argument("--pinn_w_end",      default=0.1,            type=float)
    p.add_argument("--vel_warmup_epochs",  default=20,          type=int)
    p.add_argument("--vel_w_start",        default=0.5,         type=float)
    p.add_argument("--vel_w_end",          default=1.5,         type=float)
    p.add_argument("--recurv_warmup_epochs", default=10,        type=int)
    p.add_argument("--recurv_w_start",       default=0.3,       type=float)
    p.add_argument("--recurv_w_end",         default=1.0,       type=float)
    return p.parse_args()


def _resolve_ode_steps(args):
    if args.ode_steps is not None:
        return args.ode_steps, args.ode_steps, args.ode_steps
    return args.ode_steps_train, args.ode_steps_val, args.ode_steps_test


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


def get_curriculum_len(epoch, args) -> int:
    if not args.curriculum:
        return args.pred_len
    if epoch >= args.curriculum_end_epoch:
        return args.pred_len
    frac = epoch / max(args.curriculum_end_epoch, 1)
    return int(args.curriculum_start_len
               + frac * (args.pred_len - args.curriculum_start_len))


def evaluate_fast(model, loader, device, ode_steps, pred_len, fast_ensemble=8):
    """
    FIX-V20-3: Cũng tính ensemble spread để detect collapse sớm.
    """
    model.eval()
    acc = StepErrorAccumulator(pred_len)
    t0  = time.perf_counter()
    n   = 0
    spread_buf = []   # [km] spread tại step cuối (72h proxy)

    with torch.no_grad():
        for batch in loader:
            bl = move(list(batch), device)
            pred, _, all_trajs = model.sample(bl, num_ensemble=fast_ensemble,
                                              ddim_steps=ode_steps)
            acc.update(haversine_km_torch(denorm_torch(pred), denorm_torch(bl[1])))

            # Tính spread: std của ensemble tại step cuối, convert to km
            # all_trajs: [S, T, B, 2] in normalized coords
            last_step = all_trajs[:, -1, :, :]  # [S, B, 2]
            # std across ensemble (S dim), then haversine to km
            std_lon = last_step[:, :, 0].std(0)   # [B]
            std_lat = last_step[:, :, 1].std(0)   # [B]
            # approximate km: 1 normalized unit ≈ 500 km
            spread_km = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
            spread_buf.append(spread_km)
            n += 1

    r = acc.compute()
    r["ms_per_batch"]  = (time.perf_counter() - t0) * 1e3 / max(n, 1)
    r["spread_72h_km"] = float(np.mean(spread_buf)) if spread_buf else 0.0
    return r


def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
                  metrics_csv, tag="", predict_csv=""):
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
                obs_b = od_np[:, b, :]

                cliper_errors_b = np.zeros(pred_len)
                for h in range(pred_len):
                    # pred_cliper_01  = cliper_forecast(obs_b, h + 1)
                    # pred_cliper_deg = denorm_deg_np(pred_cliper_01[np.newaxis, :])
                    pred_cliper_norm = cliper_forecast(obs_b, h + 1)
                    pred_cliper_deg  = denorm_deg_np(pred_cliper_norm[np.newaxis])
                    gt_point        = gd_np[h, b, :][np.newaxis, :]
                    # gt_deg          = denorm_deg_np(gt_point)
                    gt_deg           = denorm_deg_np(gt_point[h:h+1])
                    cliper_errors_b[h] = float(haversine_km_np(pred_cliper_deg, gt_deg)[0])

                cliper_step_errors.append(cliper_errors_b)

    if cliper_step_errors:
        cliper_mat = np.stack(cliper_step_errors)
        cliper_ugde_dict = {h: float(cliper_mat[:, s].mean())
                            for h, s in HORIZON_STEPS.items()
                            if s < cliper_mat.shape[1]}
        ev.cliper_ugde = cliper_ugde_dict
        print(f"  [CLIPER UGDE] 72h={cliper_ugde_dict.get(72, float('nan')):.1f} km")

    dm = ev.compute(tag=tag)

    try:
        if LANDFALL_TARGETS and ens_seqs_01:
            bss_vals = []
            step_72  = HORIZON_STEPS.get(72, pred_len - 1)
            for tname, t_lon, t_lat in LANDFALL_TARGETS:
                bv = brier_skill_score(
                    [e.transpose(1,0,2) if e.ndim==3 else e for e in ens_seqs_01],
                    gt_seqs_01, min(step_72, pred_len-1),
                    (t_lon, t_lat), LANDFALL_RADIUS_KM)
                if not math.isnan(bv):
                    bss_vals.append(bv)
            if bss_vals:
                dm.bss_mean = float(np.mean(bss_vals))
                print(f"  [BSS] mean={dm.bss_mean:.4f}")
    except Exception as e:
        print(f"  ⚠  BSS failed: {e}")

    save_metrics_csv(dm, metrics_csv, tag=tag)
    return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


class BestModelSaver:
    def __init__(self, patience=50, ade_tol=5.0):
        self.patience      = patience
        self.ade_tol       = ade_tol
        self.best_ade      = float("inf")
        self.best_val_loss = float("inf")
        self.counter_ade   = 0
        self.counter_loss  = 0
        self.early_stop    = False

    def reset_counters(self, reason=""):
        self.counter_ade  = 0
        self.counter_loss = 0
        if reason:
            print(f"  [SAVER] Patience counters reset: {reason}")

    def update_val_loss(self, val_loss, model, out_dir, epoch, optimizer, tl):
        if val_loss < self.best_val_loss - 1e-4:
            self.best_val_loss = val_loss;  self.counter_loss = 0
            torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
                            optimizer_state=optimizer.state_dict(),
                            train_loss=tl, val_loss=val_loss,
                            model_version="v20-valloss"),
                       os.path.join(out_dir, "best_model_valloss.pth"))
        else:
            self.counter_loss += 1

    def update_ade(self, ade, model, out_dir, epoch, optimizer, tl, vl,
                   min_epochs=80):
        if ade < self.best_ade - self.ade_tol:
            self.best_ade = ade;  self.counter_ade = 0
            torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
                            optimizer_state=optimizer.state_dict(),
                            train_loss=tl, val_loss=vl, val_ade_km=ade,
                            model_version="v20-FNO-Mamba-recurv"),
                       os.path.join(out_dir, "best_model.pth"))
            print(f"  ✅ Best ADE {ade:.1f} km  (epoch {epoch})")
        else:
            self.counter_ade += 1
            print(f"  No ADE improvement {self.counter_ade}/{self.patience}"
                  f"  (Δ={self.best_ade-ade:.1f} km < tol={self.ade_tol} km)"
                  f"  | Loss counter {self.counter_loss}/{self.patience}")

        if epoch >= min_epochs:
            if self.counter_ade >= self.patience and self.counter_loss >= self.patience:
                self.early_stop = True
                print(f"  ⛔ Early stop @ epoch {epoch}")
        else:
            if self.counter_ade >= self.patience and self.counter_loss >= self.patience:
                print(f"  ⚠  Would stop but min_epochs={min_epochs} not reached. Continuing...")
                self.counter_ade = 0;  self.counter_loss = 0


def _load_baseline_errors(path, name):
    if path is None:
        print(f"\n  ⚠  WARNING: --{name.lower().replace(' ','_')}_errors_npy not provided.")
        print(f"     Statistical comparison vs {name} will be SKIPPED.\n")
        return None
    if not os.path.exists(path):
        print(f"\n  ⚠  {path} not found — {name} skipped.\n")
        return None
    arr = np.load(path)
    print(f"  ✓  Loaded {name}: {arr.shape}")
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

    ode_train, ode_val, ode_test = _resolve_ode_steps(args)

    print("=" * 68)
    print("  TC-FlowMatching v20  |  FNO3D + Mamba + OT-CFM + PINN")
    print("  v20 FIXES:")
    print("    FIX-V20-1: Ensemble collapse → ctx noise + larger initial σ")
    print("    FIX-V20-2: Early stop → ADE mỗi epoch (counter = real epochs)")
    print("    FIX-V20-3: Spread monitor để detect collapse")
    print("=" * 68)
    print(f"  device               : {device}")
    print(f"  sigma_min            : {args.sigma_min}")
    print(f"  ctx_noise_scale      : {args.ctx_noise_scale}")
    print(f"  initial_sample_sigma : {args.initial_sample_sigma}")
    print(f"  ode_steps            : train={ode_train}  val={ode_val}  test={ode_test}")
    print(f"  val_ensemble         : {args.val_ensemble}")
    print(f"  patience             : {args.patience} epochs  (min_epochs={args.min_epochs})")
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

    # FIX-V20-1: pass ensemble collapse params to model
    model = TCFlowMatching(
        pred_len             = args.pred_len,
        obs_len              = args.obs_len,
        sigma_min            = args.sigma_min,
        n_train_ens          = args.n_train_ens,
        ctx_noise_scale      = args.ctx_noise_scale,
        initial_sample_sigma = args.initial_sample_sigma,
    ).to(device)

    if (args.fno_spatial_down != 32 or args.fno_modes_h != 4
            or args.fno_layers != 4 or args.fno_d_model != 32):
        from Model.FNO3D_encoder import FNO3DEncoder
        model.net.spatial_enc = FNO3DEncoder(
            in_channel=13, out_channel=1,
            d_model=args.fno_d_model, n_layers=args.fno_layers,
            modes_t=args.fno_modes_t, modes_h=args.fno_modes_h,
            modes_w=args.fno_modes_h, spatial_down=args.fno_spatial_down,
            dropout=0.05).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params  : {n_params:,}")

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
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    saver  = BestModelSaver(patience=args.patience, ade_tol=5.0)
    scaler = GradScaler('cuda', enabled=args.use_amp)

    print("=" * 68)
    print(f"  TRAINING  ({steps_per_epoch} steps/epoch)")
    print("=" * 68)

    epoch_times   = []
    train_start   = time.perf_counter()
    last_val_loss = float("inf")
    _lr_ep30_done = False
    _lr_ep60_done = False
    _prev_ens     = 1

    import Model.losses as _losses_mod

    for epoch in range(args.num_epochs):
        # Progressive ensemble
        current_ens = 1 if epoch < 30 else (2 if epoch < 60 else args.n_train_ens)
        model.n_train_ens = current_ens
        effective_fast_ens = min(args.fast_ensemble, max(current_ens*2, args.fast_ensemble))

        if current_ens != _prev_ens:
            saver.reset_counters(
                f"n_train_ens {_prev_ens}→{current_ens} at epoch {epoch}")
            _prev_ens = current_ens

        curr_len = get_curriculum_len(epoch, args)
        if hasattr(model, "set_curriculum_len"):
            model.set_curriculum_len(curr_len)

        epoch_weights = copy.copy(_BASE_WEIGHTS)
        epoch_weights["pinn"]     = get_pinn_weight(epoch, args.pinn_warmup_epochs,
                                                    args.pinn_w_start, args.pinn_w_end)
        epoch_weights["velocity"] = get_velocity_weight(epoch, args.vel_warmup_epochs,
                                                        args.vel_w_start, args.vel_w_end)
        epoch_weights["recurv"]   = get_recurv_weight(epoch, args.recurv_warmup_epochs,
                                                      args.recurv_w_start, args.recurv_w_end)
        _losses_mod.WEIGHTS.update(epoch_weights)
        if hasattr(model, 'weights'):
            model.weights = epoch_weights
        _losses_mod.WEIGHTS["pinn"]     = epoch_weights["pinn"]
        _losses_mod.WEIGHTS["velocity"] = epoch_weights["velocity"]
        _losses_mod.WEIGHTS["recurv"]   = epoch_weights["recurv"]

        current_clip = get_grad_clip(epoch, warmup_epochs=20,
                                     clip_start=args.grad_clip, clip_end=1.0)

        # LR restarts
        if epoch == 30 and not _lr_ep30_done:
            _lr_ep30_done = True
            warmup_steps = steps_per_epoch * 1
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_steps,
                steps_per_epoch * (args.num_epochs - 30),
                min_lr=5e-6)
            saver.reset_counters("LR warm restart at epoch 30")
            print(f"  ↺  Warm Restart LR at epoch 30")

        if epoch == 60 and not _lr_ep60_done:
            _lr_ep60_done = True
            warmup_steps = steps_per_epoch * 1
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_steps,
                steps_per_epoch * (args.num_epochs - 60),
                min_lr=1e-6)
            saver.reset_counters("LR warm restart at epoch 60")
            print(f"  ↺  Warm Restart LR at epoch 60")

        # ── Training loop ─────────────────────────────────────────────────────
        model.train()
        sum_loss = 0.0
        t0 = time.perf_counter()
        optimizer.zero_grad()
        recurv_ratio_buf = []

        for i, batch in enumerate(train_loader):
            bl = move(list(batch), device)

            if epoch == 0 and i == 0:
                test_env = bl[13]
                if test_env is not None and "gph500_mean" in test_env:
                    gph_val = test_env["gph500_mean"]
                    if torch.all(gph_val == 0):
                        print("\n" + "!"*60)
                        print("  ⚠️  GPH500 đang bị triệt tiêu về 0!")
                        print("!"*60 + "\n")
                    else:
                        print(f"  ✅ Data Check: GPH500 OK (Mean: {gph_val.mean().item():.4f})")

            with autocast(device_type='cuda', enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl)

            loss_to_backpass = bd["total"] / max(args.grad_accum, 1)
            scaler.scale(loss_to_backpass).backward()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), current_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            sum_loss += bd["total"].item()

            if "recurv_ratio" in bd:
                recurv_ratio_buf.append(bd["recurv_ratio"])

            if i % 20 == 0:
                lr  = optimizer.param_groups[0]["lr"]
                rr  = bd.get("recurv_ratio", 0.0)
                elapsed = time.perf_counter() - t0
                print(f"  [{epoch:>3}][{i:>3}/{len(train_loader)}]"
                      f"  loss={bd['total'].item():.3f}"
                      f"  fm={bd.get('fm',0):.2f}"
                      f"  vel={bd.get('velocity',0):.6f}"
                      f"  pinn={bd.get('pinn', 0):.6f}"
                      f"  recurv={bd.get('recurv',0):.3f}"
                      f"  rr={rr:.2f}"
                      f"  pinn_w={epoch_weights['pinn']:.3f}"
                      f"  clip={current_clip:.1f}"
                      f"  ens={current_ens}  len={curr_len}"
                      f"  lr={lr:.2e}  t={elapsed:.0f}s")

        ep_s  = time.perf_counter() - t0
        epoch_times.append(ep_s)
        avg_t = sum_loss / len(train_loader)
        mean_rr = float(np.mean(recurv_ratio_buf)) if recurv_ratio_buf else 0.0

        # ── Val loss (mỗi val_freq epoch) ─────────────────────────────────────
        if epoch % args.val_freq == 0:
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
            saver.update_val_loss(last_val_loss, model, args.output_dir,
                                  epoch, optimizer, avg_t)
            print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={last_val_loss:.3f}"
                  f"  rr={mean_rr:.2f}"
                  f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
                  f"  ens={current_ens}  len={curr_len}"
                  f"  recurv_w={epoch_weights['recurv']:.2f}")
        else:
            print(f"  Epoch {epoch:>3}  train={avg_t:.3f}"
                  f"  val={last_val_loss:.3f}(cached)"
                  f"  rr={mean_rr:.2f}  t={ep_s:.0f}s")

        # ── ADE evaluation MỖI EPOCH (FIX-V19-1 / V20-3) ─────────────────────
        t_ade = time.perf_counter()
        m = evaluate_fast(model, val_subset_loader, device,
                          ode_train, args.pred_len, effective_fast_ens)
        t_ade_s = time.perf_counter() - t_ade

        spread_72h = m.get("spread_72h_km", 0.0)
        collapse_warn = "  ⚠️ COLLAPSE!" if spread_72h < 10.0 else ""

        print(f"  [ADE ep{epoch} {t_ade_s:.0f}s]"
              f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km"
              f"  12h={m.get('12h',0):.0f}  24h={m.get('24h',0):.0f}"
              f"  72h={m.get('72h',0):.0f} km"
              f"  spread={spread_72h:.1f} km"
              f"  (ens={effective_fast_ens}, steps={ode_train})"
              f"  counter={saver.counter_ade}/{args.patience}"
              f"{collapse_warn}")

        saver.update_ade(m["ADE"], model, args.output_dir, epoch,
                         optimizer, avg_t, last_val_loss,
                         min_epochs=args.min_epochs)

        # ── Full eval (mỗi full_eval_freq epoch) ──────────────────────────────
        if epoch % args.full_eval_freq == 0 and epoch > 0:
            print(f"  [Full eval epoch {epoch}, ode_steps={ode_val}]")
            try:
                dm, _, _, _ = evaluate_full(
                    model, val_loader, device,
                    ode_val, args.pred_len, args.val_ensemble,
                    metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}")
                print(dm.summary())
            except Exception as e:
                print(f"  ⚠  full_eval failed at epoch {epoch}: {e}")
                import traceback; traceback.print_exc()

        if (epoch+1) % args.save_interval == 0:
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

    _losses_mod.WEIGHTS["pinn"]     = args.pinn_w_end
    _losses_mod.WEIGHTS["velocity"] = args.vel_w_end
    _losses_mod.WEIGHTS["recurv"]   = args.recurv_w_end

    total_train_h = (time.perf_counter() - train_start) / 3600

    # ── Final test eval ───────────────────────────────────────────────────────
    print(f"\n{'='*68}  FINAL TEST (ode_steps={ode_test})")
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
                  f"  ADE={ck.get('val_ade_km','?')}")

        final_ens = max(args.val_ensemble, 50)
        dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
            model, test_loader, device,
            ode_test, args.pred_len, final_ens,
            metrics_csv=metrics_csv, tag="test_final",
            predict_csv=predict_csv)
        print(dm_test.summary())

        all_results.append(ModelResult(
            model_name   = "FM+PINN-v20",
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

        _, cliper_errs = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
        persist_errs   = persistence_errors(obs_seqs, gt_seqs, args.pred_len)
        fmpinn_per_seq = np.array([seq_ade_km(np.array(pp), np.array(g))
                                    for pp, g in zip(pred_seqs, gt_seqs)])

        np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
        np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
        np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))

        lstm_per_seq      = _load_baseline_errors(args.lstm_errors_npy,      "LSTM")
        diffusion_per_seq = _load_baseline_errors(args.diffusion_errors_npy, "Diffusion")
        if lstm_per_seq is not None:
            np.save(os.path.join(stat_dir, "lstm.npy"), lstm_per_seq)
        if diffusion_per_seq is not None:
            np.save(os.path.join(stat_dir, "diffusion.npy"), diffusion_per_seq)

        _dummy = np.array([float("nan")])
        run_all_tests(
            fmpinn_ade    = fmpinn_per_seq,
            cliper_ade    = cliper_errs.mean(1),
            lstm_ade      = lstm_per_seq if lstm_per_seq is not None else _dummy,
            diffusion_ade = diffusion_per_seq if diffusion_per_seq is not None else _dummy,
            persist_ade   = persist_errs.mean(1),
            out_dir       = stat_dir)

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
            stat_rows.append(paired_tests(fmpinn_per_seq, lstm_per_seq, "FM+PINN vs LSTM", 5))
        if diffusion_per_seq is not None:
            stat_rows.append(paired_tests(fmpinn_per_seq, diffusion_per_seq, "FM+PINN vs Diffusion", 5))

        compute_rows = DEFAULT_COMPUTE
        try:
            sb = next(iter(test_loader))
            sb = move(list(sb), device)
            from utils.evaluation_tables import profile_model_components
            compute_rows = profile_model_components(model, sb, device)
        except Exception as e:
            print(f"  Profiling skipped: {e}")

        export_all_tables(
            results=all_results, ablation_rows=DEFAULT_ABLATION,
            stat_rows=stat_rows, pinn_sens_rows=DEFAULT_PINN_SENSITIVITY,
            compute_rows=compute_rows, out_dir=tables_dir)

        with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
            fh.write(dm_test.summary())
            fh.write(f"\n\nmodel_version         : FM+PINN v20\n")
            fh.write(f"sigma_min             : {args.sigma_min}\n")
            fh.write(f"ctx_noise_scale       : {args.ctx_noise_scale}\n")
            fh.write(f"initial_sample_sigma  : {args.initial_sample_sigma}\n")
            fh.write(f"ode_steps_test        : {ode_test}\n")
            fh.write(f"eval_ensemble         : {final_ens}\n")
            fh.write(f"train_time_h          : {total_train_h:.2f}\n")
            fh.write(f"n_params_M            : "
                     f"{sum(p.numel() for p in model.parameters())/1e6:.2f}\n")

    avg_ep = sum(epoch_times)/len(epoch_times) if epoch_times else 0
    print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
    print(f"  Best val loss  : {saver.best_val_loss:.4f}")
    print(f"  Avg epoch time : {avg_ep:.0f}s")
    print(f"  Total training : {total_train_h:.2f}h")
    print(f"  Tables dir     : {tables_dir}")
    print("=" * 68)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42);  torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main(args)
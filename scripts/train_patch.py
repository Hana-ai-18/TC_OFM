"""
PATCH cho train_flowmatching_v34fix_resumable.py
Thêm ATE/CTE tracking vào evaluate_fast và evaluate_full_val_ade
Cập nhật _composite_score để dùng v2 (normalize theo ST-Trans)
Cập nhật log để hiện ATE/CTE

CÁCH DÙNG:
  Thay thế các hàm tương ứng trong train script gốc.
  Hoặc import patch này ở đầu train script.
"""
from __future__ import annotations
import time
import numpy as np
import torch


def _composite_score(result):
    """
    V2: Normalize theo ST-Trans targets.
    ATE=79.94, CTE=93.58, Mean DPE=136.41, 72h≈300km
    """
    ade = result.get("ADE", float("inf"))
    h12 = result.get("12h", float("inf"))
    h24 = result.get("24h", float("inf"))
    h48 = result.get("48h", float("inf"))
    h72 = result.get("72h", float("inf"))
    ate = result.get("ATE_mean", float("inf"))
    cte = result.get("CTE_mean", float("inf"))

    # Fallback nếu chưa có ATE/CTE
    if not np.isfinite(ate): ate = ade * 0.45
    if not np.isfinite(cte): cte = ade * 0.52

    score = (
        0.05 * (ade / 136.0)
        + 0.05 * (h12 / 50.0)
        + 0.10 * (h24 / 100.0)
        + 0.15 * (h48 / 200.0)
        + 0.35 * (h72 / 300.0)   # 72h — priority cao nhất
        + 0.15 * (ate / 80.0)    # ATE — beat ST-Trans 79.94
        + 0.15 * (cte / 94.0)    # CTE — beat ST-Trans 93.58
    )
    return score * 100.0


def evaluate_fast_v2(model, loader, device, ode_steps, pred_len, fast_ensemble=15):
    """
    evaluate_fast với ATE/CTE tracking.
    Thay thế evaluate_fast trong train script.
    """
    # Import tại đây để tránh circular
    from utils.metrics import (
        StepErrorAccumulator, haversine_km_torch, denorm_torch,
        haversine_and_atecte_torch,
    )

    model.eval()
    acc = StepErrorAccumulator(pred_len)
    t0  = time.perf_counter()
    n   = 0
    spread_per_step = []

    with torch.no_grad():
        for batch in loader:
            bl = _move_batch(batch, device)
            pred, _, all_trajs = model.sample(
                bl, num_ensemble=fast_ensemble, ddim_steps=ode_steps,
                importance_weight=True)
            T_active  = pred.shape[0]
            gt_sliced = bl[1][:T_active]

            # Haversine + ATE/CTE
            dist, ate, cte = haversine_and_atecte_torch(
                denorm_torch(pred), denorm_torch(gt_sliced),
                unit_01deg=True
            )
            acc.update(dist, ate_km=ate, cte_km=cte)

            # Spread
            step_spreads = []
            for t in range(all_trajs.shape[1]):
                step_data = all_trajs[:, t, :, :]
                std_lon   = step_data[:, :, 0].std(0)
                std_lat   = step_data[:, :, 1].std(0)
                spread    = ((std_lon**2 + std_lat**2).sqrt() * 500.0).mean().item()
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
    return r


def evaluate_full_val_ade_v2(model, val_loader, device, ode_steps, pred_len,
                              fast_ensemble, metrics_csv, epoch,
                              use_ema=False, ema_obj=None):
    """
    evaluate_full_val_ade với ATE/CTE tracking.
    Thay thế evaluate_full_val_ade trong train script.
    """
    from utils.metrics import (
        StepErrorAccumulator, haversine_km_torch, denorm_torch,
        DatasetMetrics, save_metrics_csv as _save_csv,
        haversine_and_atecte_torch,
    )
    from datetime import datetime

    backup = None
    if use_ema and ema_obj is not None:
        try:
            backup = ema_obj.apply_to(model)
        except Exception as e:
            print(f"  ⚠  EMA apply_to failed: {e}")
            backup  = None
            use_ema = False

    model.eval()
    acc = StepErrorAccumulator(pred_len)
    t0  = time.perf_counter()
    n   = 0

    with torch.no_grad():
        for batch in val_loader:
            bl = _move_batch(list(batch), device)
            pred, _, _ = model.sample(
                bl, num_ensemble=max(fast_ensemble, 20),
                ddim_steps=max(ode_steps, 30),
                importance_weight=True)
            T_pred = pred.shape[0]
            gt = bl[1][:T_pred]
            dist, ate, cte = haversine_and_atecte_torch(
                denorm_torch(pred), denorm_torch(gt), unit_01deg=True
            )
            acc.update(dist, ate_km=ate, cte_km=cte)
            n += 1

    r       = acc.compute()
    elapsed = time.perf_counter() - t0
    score   = _composite_score(r)
    tag     = "EMA" if use_ema else "RAW"

    ate_m = r.get("ATE_mean", float("nan"))
    cte_m = r.get("CTE_mean", float("nan"))
    ate_72 = r.get("ATE_72h", float("nan"))
    cte_72 = r.get("CTE_72h", float("nan"))

    print(f"\n{'='*70}")
    print(f"  [FULL VAL ADE ({tag})  ep={epoch}  {elapsed:.0f}s  {n} batches]")
    print(f"  ADE = {r.get('ADE', float('nan')):.1f} km  "
          f"FDE = {r.get('FDE', float('nan')):.1f} km")
    print(f"  6h={r.get('6h',  float('nan')):.0f}  "
          f"12h={r.get('12h', float('nan')):.0f}  "
          f"24h={r.get('24h', float('nan')):.0f}  "
          f"48h={r.get('48h', float('nan')):.0f}  "
          f"72h={r.get('72h', float('nan')):.0f} km")
    print(f"  ATE_mean={ate_m:.1f} km  CTE_mean={cte_m:.1f} km  "
          f"  [ST-Trans: ATE=79.94 CTE=93.58]")
    print(f"  ATE@72h={ate_72:.1f} km  CTE@72h={cte_72:.1f} km")

    # Beat indicators
    ade_val = r.get("ADE", float("inf"))
    h72_val = r.get("72h", float("inf"))
    beat_str = []
    if ade_val < 136.41: beat_str.append(f"DPE✅({ade_val:.1f}<136.41)")
    if ate_m < 79.94:    beat_str.append(f"ATE✅({ate_m:.1f}<79.94)")
    if cte_m < 93.58:    beat_str.append(f"CTE✅({cte_m:.1f}<93.58)")
    if h72_val < 300.0:  beat_str.append(f"72h✅({h72_val:.1f}<300)")
    if beat_str:
        print(f"  🏆 BEAT ST-TRANS: {' '.join(beat_str)}")

    print(f"  Composite score v2 = {score:.1f}  (lower=better, <100=beat ST-Trans)")
    print(f"{'='*70}\n")

    dm = DatasetMetrics(
        ade      = r.get("ADE",  float("nan")),
        fde      = r.get("FDE",  float("nan")),
        ugde_12h = r.get("12h",  float("nan")),
        ugde_24h = r.get("24h",  float("nan")),
        ugde_48h = r.get("48h",  float("nan")),
        ugde_72h = r.get("72h",  float("nan")),
        ate_abs_mean = ate_m,
        cte_abs_mean = cte_m,
        n_total  = r.get("n_samples", 0),
        timestamp= datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    _save_csv(dm, metrics_csv, tag=f"val_full_{tag}_ep{epoch:03d}")

    if backup is not None:
        try:
            ema_obj.restore(model, backup)
        except Exception as e:
            print(f"  ⚠  EMA restore failed: {e}")

    return r


def _move_batch(batch, device):
    """Helper to move batch to device."""
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return out


# ── Log format patch cho train loop ──────────────────────────────────────────

def log_fast_eval(r: dict, epoch: int) -> None:
    """Log FAST eval với ATE/CTE indicators."""
    h6  = r.get("6h",  float("nan"))
    h12 = r.get("12h", float("nan"))
    h24 = r.get("24h", float("nan"))
    h48 = r.get("48h", float("nan"))
    h72 = r.get("72h", float("nan"))
    ate = r.get("ATE_mean", float("nan"))
    cte = r.get("CTE_mean", float("nan"))
    score = r.get("composite_score_v2", _composite_score(r))

    def ind(v, tgt):
        if not np.isfinite(v): return "?"
        return "🎯" if v < tgt else "❌"

    print(
        f"  [FAST ep{epoch}]"
        f"  ADE={r.get('ADE', float('nan')):.1f}"
        f"  6h={h6:.0f}{ind(h6,30)}"
        f"  12h={h12:.0f}{ind(h12,50)}"
        f"  24h={h24:.0f}{ind(h24,100)}"
        f"  48h={h48:.0f}{ind(h48,200)}"
        f"  72h={h72:.0f}{ind(h72,300)}"
        f"  ATE={ate:.1f}{ind(ate,79.94)}"
        f"  CTE={cte:.1f}{ind(cte,93.58)}"
        f"  score={score:.1f}"
    )


# ── HorizonAwareBestSaver patch ───────────────────────────────────────────────

class HorizonAwareBestSaverV2:
    """
    V2: Track ATE/CTE thêm vào.
    Lưu best_ate và best_cte checkpoint riêng.
    """
    def __init__(self, patience=30, tol=1.5):
        self.patience   = patience
        self.tol        = tol
        self.counter    = 0
        self.early_stop = False
        self.best_score = float("inf")
        self.best_ade   = float("inf")
        self.best_12h   = float("inf")
        self.best_24h   = float("inf")
        self.best_48h   = float("inf")
        self.best_72h   = float("inf")
        self.best_ate   = float("inf")
        self.best_cte   = float("inf")

    def update(self, r, model, out_dir, epoch, optimizer, scheduler,
               tl, vl, saver_ref, min_epochs=50, save_fn=None):
        ade   = r.get("ADE", float("inf"))
        h12   = r.get("12h", float("inf"))
        h24   = r.get("24h", float("inf"))
        h48   = r.get("48h", float("inf"))
        h72   = r.get("72h", float("inf"))
        ate   = r.get("ATE_mean", ade * 0.45)
        cte   = r.get("CTE_mean", ade * 0.52)
        score = _composite_score(r)

        improved_any = False

        if ade < self.best_ade:
            self.best_ade = ade; improved_any = True
            if save_fn:
                save_fn(f"{out_dir}/best_ade.pth", epoch, model, optimizer,
                        scheduler, saver_ref, tl, vl,
                        {"ade": ade, "tag": "best_ade"})

        if h72 < self.best_72h:
            self.best_72h = h72; improved_any = True
            if save_fn:
                save_fn(f"{out_dir}/best_72h.pth", epoch, model, optimizer,
                        scheduler, saver_ref, tl, vl,
                        {"h72": h72, "tag": "best_72h"})

        if ate < self.best_ate:
            self.best_ate = ate; improved_any = True
            if save_fn:
                save_fn(f"{out_dir}/best_ate.pth", epoch, model, optimizer,
                        scheduler, saver_ref, tl, vl,
                        {"ate": ate, "tag": "best_ate"})

        if cte < self.best_cte:
            self.best_cte = cte; improved_any = True
            if save_fn:
                save_fn(f"{out_dir}/best_cte.pth", epoch, model, optimizer,
                        scheduler, saver_ref, tl, vl,
                        {"cte": cte, "tag": "best_cte"})

        if h48 < self.best_48h: self.best_48h = h48; improved_any = True
        if h24 < self.best_24h: self.best_24h = h24; improved_any = True
        if h12 < self.best_12h: self.best_12h = h12; improved_any = True

        if score < self.best_score - self.tol:
            self.best_score = score
            self.counter    = 0
            if save_fn:
                save_fn(f"{out_dir}/best_model.pth", epoch, model, optimizer,
                        scheduler, saver_ref, tl, vl,
                        {"ade": ade, "h12": h12, "h24": h24,
                         "h48": h48, "h72": h72, "ate": ate, "cte": cte,
                         "composite_score": score, "tag": "best_composite"})
            print(f"  ✅ Best COMPOSITE={score:.1f}  "
                  f"ADE={ade:.1f}  12h={h12:.0f}  24h={h24:.0f}  "
                  f"48h={h48:.0f}  72h={h72:.0f}  "
                  f"ATE={ate:.1f}  CTE={cte:.1f}  (ep {epoch})")
        else:
            if not improved_any:
                self.counter += 1
            print(f"  No improvement {self.counter}/{self.patience}"
                  f"  (best={self.best_score:.1f}, cur={score:.1f})"
                  f"  72h_best={self.best_72h:.0f} ATE_best={self.best_ate:.1f}")

        if epoch >= min_epochs and self.counter >= self.patience:
            self.early_stop = True
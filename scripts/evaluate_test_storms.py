"""
scripts/evaluate_models.py
═══════════════════════════════════════════════════════════════════════════════

Script đánh giá tất cả models (LSTM, ST-Trans, FlowMatching) trên bộ test 12 bão.

Cách dùng:
    python scripts/evaluate_models.py \
        --dataset_root /kaggle/input/datasets/kaggle1234uitvn/tc-ofm \
        --fm_ckpt runs/v59/best_model.pth \
        --lstm_ckpt runs/lstm/best_model.pth \
        --sttrans_ckpt runs/sttrans/best_model.pth \
        --output_dir results/test_eval \
        --gpu_num 0

Output:
    results/test_eval/
        ├── FM_v59_per_storm.csv         # Per-storm, per-timestep predictions
        ├── LSTM_per_storm.csv
        ├── STTrans_per_storm.csv
        ├── summary_all_models.csv       # Bảng so sánh ADE/ATE/CTE tất cả models
        └── per_storm_detail/
            ├── FM_v59_Amphan.csv        # Từng bão riêng
            ├── FM_v59_Mocha.csv
            └── ...

CSV format per storm file:
    storm_id | lead_h | lon_pred | lat_pred | lon_true | lat_true |
    dist_km  | ate_km | cte_km
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv
import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
R_EARTH  = 6371.0
DEG2KM   = 111.0
DT_HOURS = 6.0

LEAD_HOURS  = [12, 24, 36, 48, 60, 72]          # 2, 4, 6, 8, 10, 12 steps (6h each)
LEAD_STEPS  = {h: h // 6 - 1 for h in LEAD_HOURS}  # 0-indexed: step idx

# Targets từ papers (Faiaz 2026 / Rahman 2025)
TARGETS = {
    "12h" : 50.0,
    "24h" : 100.0,
    "48h" : 200.0,
    "72h" : 297.0,
    "ADE" : 136.41,
    "ATE" : 79.94,
    "CTE" : 93.58,
}

MODEL_CONFIGS = {
    "LSTM"    : {"color": "🔵", "ref": "Rahman 2025"},
    "STTrans" : {"color": "🟡", "ref": "Faiaz 2026"},
    "FM_v59"  : {"color": "🔴", "ref": "Ours"},
}


# ═════════════════════════════════════════════════════════════════════════════
#  Coordinate utilities
# ═════════════════════════════════════════════════════════════════════════════

def norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
    out = arr.clone()
    out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
    out[..., 1] = (arr[..., 1] * 50.0) / 10.0
    return out


def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """p1, p2: [..., 2] với [..., 0]=lon, [..., 1]=lat (degrees)"""
    lat1 = torch.deg2rad(p1[..., 1]); lat2 = torch.deg2rad(p2[..., 1])
    dlat = torch.deg2rad(p2[..., 1] - p1[..., 1])
    dlon = torch.deg2rad(p2[..., 0] - p1[..., 0])
    a    = (torch.sin(dlat / 2).pow(2) +
            torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).pow(2))
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1 - 1e-12).sqrt())


def compute_ate_cte(
    pred_deg: torch.Tensor,
    gt_deg:   torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tính ATE và CTE cho từng bước dự báo.

    Args:
        pred_deg: [T, B, 2] predicted positions in degrees
        gt_deg:   [T, B, 2] ground truth positions in degrees

    Returns:
        ate: [T-1, B] along-track error (km), signed
        cte: [T-1, B] cross-track error (km), signed
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z

    # Along-track direction từ GT
    lon1 = torch.deg2rad(gt_deg[:T-1, :, 0])
    lat1 = torch.deg2rad(gt_deg[:T-1, :, 1])
    lon2 = torch.deg2rad(gt_deg[1:T,  :, 0])
    lat2 = torch.deg2rad(gt_deg[1:T,  :, 1])

    # Forward azimuth của GT track
    dlon_a = lon2 - lon1
    y_a = torch.sin(dlon_a) * torch.cos(lat2)
    x_a = (torch.cos(lat1) * torch.sin(lat2)
           - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon_a))
    bear_along = torch.atan2(y_a, x_a)  # [T-1, B]

    # Error vector direction
    lon3 = torch.deg2rad(pred_deg[1:T, :, 0])
    lat3 = torch.deg2rad(pred_deg[1:T, :, 1])
    dlon_e = lon3 - lon2
    y_e = torch.sin(dlon_e) * torch.cos(lat3)
    x_e = (torch.cos(lat2) * torch.sin(lat3)
           - torch.sin(lat2) * torch.cos(lat3) * torch.cos(dlon_e))
    bear_err = torch.atan2(y_e, x_e)   # [T-1, B]

    # Tổng lỗi & decompose
    total = haversine_km(pred_deg[1:T], gt_deg[1:T])  # [T-1, B]
    angle = bear_err - bear_along
    ate   = total * torch.cos(angle)   # [T-1, B]
    cte   = total * torch.sin(angle)   # [T-1, B]
    return ate, cte


def move_to_device(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return out


def get_raw_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def get_ema_obj(model):
    if hasattr(model, "_ema") and model._ema is not None:
        return model._ema
    if hasattr(model, "_orig_mod"):
        orig = model._orig_mod
        if hasattr(orig, "_ema") and orig._ema is not None:
            return orig._ema
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  Per-storm result accumulator
# ═════════════════════════════════════════════════════════════════════════════

class StormResult:
    """Lưu kết quả đầy đủ cho 1 cơn bão."""

    def __init__(self, storm_id: str):
        self.storm_id = storm_id
        self.rows: List[dict] = []          # per-timestep rows
        self.dist_by_lead: Dict[int, List[float]] = defaultdict(list)
        self.ate_by_lead:  Dict[int, List[float]] = defaultdict(list)
        self.cte_by_lead:  Dict[int, List[float]] = defaultdict(list)
        self.all_dist: List[float] = []
        self.all_ate:  List[float] = []
        self.all_cte:  List[float] = []
        self.n_seq = 0

    def add_batch(
        self,
        pred_deg: torch.Tensor,   # [T, B, 2]
        gt_deg:   torch.Tensor,   # [T, B, 2]
        seq_indices: Optional[torch.Tensor] = None,
    ):
        T = min(pred_deg.shape[0], gt_deg.shape[0])
        B = pred_deg.shape[1]
        if T < 2:
            return

        # Per-step distance
        dist = haversine_km(pred_deg[:T], gt_deg[:T])   # [T, B]
        ate, cte = compute_ate_cte(pred_deg, gt_deg)    # [T-1, B]

        self.n_seq += B

        for b in range(B):
            for t in range(T):
                lead_h = (t + 1) * 6
                d_val  = float(dist[t, b])
                ate_val = float(ate[t, b].abs()) if t < ate.shape[0] else float("nan")
                cte_val = float(cte[t, b].abs()) if t < cte.shape[0] else float("nan")

                self.rows.append({
                    "storm_id"  : self.storm_id,
                    "seq_idx"   : int(seq_indices[b]) if seq_indices is not None else b,
                    "lead_h"    : lead_h,
                    "lon_pred"  : float(pred_deg[t, b, 0]),
                    "lat_pred"  : float(pred_deg[t, b, 1]),
                    "lon_true"  : float(gt_deg[t, b, 0]),
                    "lat_true"  : float(gt_deg[t, b, 1]),
                    "dist_km"   : round(d_val,   2),
                    "ate_km"    : round(ate_val, 2) if np.isfinite(ate_val) else "nan",
                    "cte_km"    : round(cte_val, 2) if np.isfinite(cte_val) else "nan",
                })

                # Tích lũy theo lead time
                if lead_h in LEAD_STEPS:
                    self.dist_by_lead[lead_h].append(d_val)
                    if np.isfinite(ate_val): self.ate_by_lead[lead_h].append(ate_val)
                    if np.isfinite(cte_val): self.cte_by_lead[lead_h].append(cte_val)

                # ADE/ATE/CTE tổng
                self.all_dist.append(d_val)
                if t < ate.shape[0]:
                    if np.isfinite(ate_val): self.all_ate.append(ate_val)
                    if np.isfinite(cte_val): self.all_cte.append(cte_val)

    def summary(self) -> dict:
        def _m(lst): return float(np.mean(lst)) if lst else float("nan")

        r = {
            "storm_id" : self.storm_id,
            "n_seq"    : self.n_seq,
            "ADE"      : _m(self.all_dist),
            "ATE"      : _m(self.all_ate),
            "CTE"      : _m(self.all_cte),
        }
        for h in LEAD_HOURS:
            r[f"{h}h_dist"] = _m(self.dist_by_lead.get(h, []))
            r[f"{h}h_ate"]  = _m(self.ate_by_lead.get(h,  []))
            r[f"{h}h_cte"]  = _m(self.cte_by_lead.get(h,  []))
        return r


class MultiStormAccumulator:
    """Tích lũy kết quả cho toàn bộ test set."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.storms: Dict[str, StormResult] = {}

    def get_storm(self, storm_id: str) -> StormResult:
        if storm_id not in self.storms:
            self.storms[storm_id] = StormResult(storm_id)
        return self.storms[storm_id]

    def add_batch(
        self,
        pred_norm: torch.Tensor,   # [T, B, 2] normalized
        gt_norm:   torch.Tensor,   # [T, B, 2] normalized
        storm_ids: List[str],
    ):
        # Convert to degrees
        T   = min(pred_norm.shape[0], gt_norm.shape[0])
        B   = pred_norm.shape[1]
        pd  = norm_to_deg(pred_norm[:T])
        gd  = norm_to_deg(gt_norm[:T])

        # Group by storm_id
        sid_map: Dict[str, List[int]] = defaultdict(list)
        for b, sid in enumerate(storm_ids[:B]):
            sid_map[sid].append(b)

        for sid, bidx in sid_map.items():
            bidx_t = torch.tensor(bidx, dtype=torch.long)
            self.get_storm(sid).add_batch(
                pd[:, bidx_t, :],
                gd[:, bidx_t, :],
                seq_indices=bidx_t,
            )

    def all_summaries(self) -> List[dict]:
        rows = []
        for sid in sorted(self.storms):
            rows.append(self.storms[sid].summary())
        return rows

    def overall_summary(self) -> dict:
        """Tổng trung bình tất cả bão."""
        all_d, all_a, all_c = [], [], []
        lead_d: Dict[int, List[float]] = defaultdict(list)
        lead_a: Dict[int, List[float]] = defaultdict(list)
        lead_c: Dict[int, List[float]] = defaultdict(list)

        for sr in self.storms.values():
            all_d.extend(sr.all_dist)
            all_a.extend(sr.all_ate)
            all_c.extend(sr.all_cte)
            for h in LEAD_HOURS:
                lead_d[h].extend(sr.dist_by_lead.get(h, []))
                lead_a[h].extend(sr.ate_by_lead.get(h, []))
                lead_c[h].extend(sr.cte_by_lead.get(h, []))

        def _m(lst): return float(np.mean(lst)) if lst else float("nan")
        r = {
            "storm_id" : "📊 OVERALL",
            "n_seq"    : sum(s.n_seq for s in self.storms.values()),
            "ADE"      : _m(all_d),
            "ATE"      : _m(all_a),
            "CTE"      : _m(all_c),
        }
        for h in LEAD_HOURS:
            r[f"{h}h_dist"] = _m(lead_d[h])
            r[f"{h}h_ate"]  = _m(lead_a[h])
            r[f"{h}h_cte"]  = _m(lead_c[h])
        return r


# ═════════════════════════════════════════════════════════════════════════════
#  Extract storm IDs từ batch
# ═════════════════════════════════════════════════════════════════════════════

def extract_storm_ids(batch, B: int) -> List[str]:
    """Thử các vị trí khác nhau trong batch để lấy storm ID."""
    for idx in [2, 12, 14, 15]:
        if idx < len(batch):
            item = batch[idx]
            if isinstance(item, (list, tuple)) and len(item) >= B:
                if isinstance(item[0], str):
                    return list(item[:B])
            if torch.is_tensor(item) and item.dtype == torch.long and item.numel() >= B:
                return [str(x.item()) for x in item[:B]]
    # Fallback: dùng sequential index
    return [f"STORM_{i:03d}" for i in range(B)]


# ═════════════════════════════════════════════════════════════════════════════
#  Model loaders
# ═════════════════════════════════════════════════════════════════════════════

def load_fm_model(ckpt_path: str, device, args) -> Optional[object]:
    """Load FlowMatching model từ checkpoint."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"  ⚠  FM checkpoint not found: {ckpt_path}")
        return None

    try:
        from Model.flow_matching_model import TCFlowMatching
        model = TCFlowMatching(
            pred_len=args.pred_len,
            obs_len=args.obs_len,
            sigma_min=0.02,
            use_ema=True,
            ema_decay=0.995,
        ).to(device)
        model.init_ema()

        ckpt = torch.load(ckpt_path, map_location=device)
        m = get_raw_model(model)
        missing, unexpected = m.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:    print(f"    FM missing keys: {len(missing)}")
        if unexpected: print(f"    FM unexpected keys: {len(unexpected)}")

        # Load EMA
        if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
            ema = get_ema_obj(model)
            if ema is not None:
                for k, v in ckpt["ema_shadow"].items():
                    if k in ema.shadow:
                        ema.shadow[k].copy_(v.to(device))

        ep = ckpt.get("epoch", "?")
        print(f"  ✅ FM model loaded (epoch={ep}) from {ckpt_path}")
        return model

    except Exception as e:
        print(f"  ❌ FM model load failed: {e}")
        import traceback; traceback.print_exc()
        return None


def load_lstm_model(ckpt_path: str, device, args) -> Optional[object]:
    """Load LSTM model từ checkpoint."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"  ⚠  LSTM checkpoint not found: {ckpt_path}")
        return None

    try:
        # Thử import từ các vị trí phổ biến
        try:
            from Model.lstm_model import TCLSTMModel as LSTMModel
        except ImportError:
            try:
                from Model.baselines.lstm import LSTMPredictor as LSTMModel
            except ImportError:
                from Model.LSTM_model import LSTMModel

        model = LSTMModel(
            pred_len=args.pred_len,
            obs_len=args.obs_len,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        key  = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
        missing, unexpected = model.load_state_dict(ckpt[key], strict=False)
        if missing:    print(f"    LSTM missing keys: {len(missing)}")

        ep = ckpt.get("epoch", "?")
        print(f"  ✅ LSTM model loaded (epoch={ep}) from {ckpt_path}")
        return model

    except Exception as e:
        print(f"  ❌ LSTM model load failed: {e}")
        import traceback; traceback.print_exc()
        return None


def load_sttrans_model(ckpt_path: str, device, args) -> Optional[object]:
    """Load ST-Trans model từ checkpoint."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"  ⚠  ST-Trans checkpoint not found: {ckpt_path}")
        return None

    try:
        try:
            from Model.st_trans_model import STTransModel
        except ImportError:
            try:
                from Model.baselines.st_transformer import STTransformer as STTransModel
            except ImportError:
                from Model.STTrans_model import STTransModel

        model = STTransModel(
            pred_len=args.pred_len,
            obs_len=args.obs_len,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        key  = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
        missing, unexpected = model.load_state_dict(ckpt[key], strict=False)
        if missing:    print(f"    ST-Trans missing keys: {len(missing)}")

        ep = ckpt.get("epoch", "?")
        print(f"  ✅ ST-Trans model loaded (epoch={ep}) from {ckpt_path}")
        return model

    except Exception as e:
        print(f"  ❌ ST-Trans model load failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  Inference wrappers
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def infer_fm(model, batch_list, args, device) -> Optional[torch.Tensor]:
    """Inference với FlowMatching model, trả về [T, B, 2] normalized."""
    try:
        ema = get_ema_obj(model)
        backup = None
        if ema is not None:
            try: backup = ema.apply_to(model)
            except: pass

        model.eval()
        pred, _, _ = model.sample(
            batch_list,
            num_ensemble=args.fm_ensemble,
            ddim_steps=args.fm_ode_steps,
            importance_weight=True,
        )
        # pred: [T, B, 2]

        if backup is not None:
            try: ema.restore(model, backup)
            except: pass

        return pred

    except Exception as e:
        print(f"    FM inference error: {e}")
        return None


@torch.no_grad()
def infer_lstm(model, batch_list, args, device) -> Optional[torch.Tensor]:
    """Inference với LSTM model, trả về [T, B, 2] normalized."""
    try:
        model.eval()
        obs_traj = batch_list[0]  # [T_obs, B, D]

        # Thử các interface khác nhau
        try:
            pred = model(batch_list)
        except TypeError:
            try:
                pred = model(obs_traj)
            except Exception:
                pred = model.predict(batch_list)

        # Đảm bảo shape [T, B, 2]
        if pred.dim() == 3:
            if pred.shape[0] == pred.shape[1]:
                pass  # already [T, B, 2]
            elif pred.shape[1] != obs_traj.shape[1]:
                pred = pred.permute(1, 0, 2)

        return pred[:, :, :2]

    except Exception as e:
        print(f"    LSTM inference error: {e}")
        return None


@torch.no_grad()
def infer_sttrans(model, batch_list, args, device) -> Optional[torch.Tensor]:
    """Inference với ST-Trans model, trả về [T, B, 2] normalized."""
    try:
        model.eval()
        obs_traj = batch_list[0]

        try:
            pred = model(batch_list)
        except TypeError:
            try:
                pred = model(obs_traj)
            except Exception:
                pred = model.predict(batch_list)

        if pred.dim() == 3:
            if pred.shape[1] != obs_traj.shape[1]:
                pred = pred.permute(1, 0, 2)

        return pred[:, :, :2]

    except Exception as e:
        print(f"    ST-Trans inference error: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  CSV writers
# ═════════════════════════════════════════════════════════════════════════════

_PER_STORM_FIELDS = [
    "model", "storm_id", "seq_idx", "lead_h",
    "lon_pred", "lat_pred", "lon_true", "lat_true",
    "dist_km", "ate_km", "cte_km",
]

_SUMMARY_FIELDS = [
    "model", "storm_id", "n_seq",
    "ADE", "ATE", "CTE",
    "12h_dist", "24h_dist", "48h_dist", "72h_dist",
    "12h_ate",  "24h_ate",  "48h_ate",  "72h_ate",
    "12h_cte",  "24h_cte",  "48h_cte",  "72h_cte",
    "beat_12h", "beat_24h", "beat_48h", "beat_72h",
    "beat_ADE", "beat_ATE", "beat_CTE",
]


def write_per_storm_csv(acc: MultiStormAccumulator, out_dir: str):
    """Ghi 1 CSV tổng hợp tất cả rows per timestep."""
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(out_dir, f"{acc.model_name}_per_storm.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_PER_STORM_FIELDS)
        w.writeheader()
        for sr in acc.storms.values():
            for row in sr.rows:
                w.writerow({
                    "model"    : acc.model_name,
                    "storm_id" : row["storm_id"],
                    "seq_idx"  : row["seq_idx"],
                    "lead_h"   : row["lead_h"],
                    "lon_pred" : f"{row['lon_pred']:.4f}",
                    "lat_pred" : f"{row['lat_pred']:.4f}",
                    "lon_true" : f"{row['lon_true']:.4f}",
                    "lat_true" : f"{row['lat_true']:.4f}",
                    "dist_km"  : row["dist_km"],
                    "ate_km"   : row["ate_km"],
                    "cte_km"   : row["cte_km"],
                })
    print(f"  💾 {out}")
    return out


def write_per_storm_detail_csvs(acc: MultiStormAccumulator, out_dir: str):
    """Ghi 1 CSV riêng cho từng cơn bão."""
    detail_dir = os.path.join(out_dir, "per_storm_detail")
    os.makedirs(detail_dir, exist_ok=True)
    paths = []
    for sid, sr in sorted(acc.storms.items()):
        safe_id = sid.replace(" ", "_").replace("/", "-")
        fpath   = os.path.join(detail_dir, f"{acc.model_name}_{safe_id}.csv")
        with open(fpath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_PER_STORM_FIELDS)
            w.writeheader()
            for row in sr.rows:
                w.writerow({
                    "model"    : acc.model_name,
                    "storm_id" : row["storm_id"],
                    "seq_idx"  : row["seq_idx"],
                    "lead_h"   : row["lead_h"],
                    "lon_pred" : f"{row['lon_pred']:.4f}",
                    "lat_pred" : f"{row['lat_pred']:.4f}",
                    "lon_true" : f"{row['lon_true']:.4f}",
                    "lat_true" : f"{row['lat_true']:.4f}",
                    "dist_km"  : row["dist_km"],
                    "ate_km"   : row["ate_km"],
                    "cte_km"   : row["cte_km"],
                })
        paths.append(fpath)
    print(f"  💾 {len(paths)} per-storm detail CSVs → {detail_dir}/")
    return paths


def _fmt(v, t=None, dec=1):
    if not np.isfinite(v): return "nan"
    s = f"{v:.{dec}f}"
    if t is not None: s += "✅" if v < t else "❌"
    return s


def write_summary_csv(
    all_acc: Dict[str, MultiStormAccumulator],
    out_dir: str,
):
    """Ghi CSV so sánh tất cả models, per-storm + overall."""
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(out_dir, "summary_all_models.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
        w.writeheader()

        def _b(v, t): return "YES" if (np.isfinite(v) and v < t) else "NO"
        def _f(v):    return f"{v:.2f}" if np.isfinite(v) else "nan"

        for mname, acc in all_acc.items():
            rows = acc.all_summaries() + [acc.overall_summary()]
            for r in rows:
                w.writerow({
                    "model"    : mname,
                    "storm_id" : r["storm_id"],
                    "n_seq"    : r["n_seq"],
                    "ADE"      : _f(r["ADE"]), "ATE": _f(r["ATE"]), "CTE": _f(r["CTE"]),
                    "12h_dist" : _f(r["12h_dist"]), "24h_dist": _f(r["24h_dist"]),
                    "48h_dist" : _f(r["48h_dist"]), "72h_dist": _f(r["72h_dist"]),
                    "12h_ate"  : _f(r["12h_ate"]),  "24h_ate":  _f(r["24h_ate"]),
                    "48h_ate"  : _f(r["48h_ate"]),  "72h_ate":  _f(r["72h_ate"]),
                    "12h_cte"  : _f(r["12h_cte"]),  "24h_cte":  _f(r["24h_cte"]),
                    "48h_cte"  : _f(r["48h_cte"]),  "72h_cte":  _f(r["72h_cte"]),
                    "beat_12h" : _b(r["12h_dist"], TARGETS["12h"]),
                    "beat_24h" : _b(r["24h_dist"], TARGETS["24h"]),
                    "beat_48h" : _b(r["48h_dist"], TARGETS["48h"]),
                    "beat_72h" : _b(r["72h_dist"], TARGETS["72h"]),
                    "beat_ADE" : _b(r["ADE"],      TARGETS["ADE"]),
                    "beat_ATE" : _b(r["ATE"],      TARGETS["ATE"]),
                    "beat_CTE" : _b(r["CTE"],      TARGETS["CTE"]),
                })
    print(f"  💾 Summary: {out}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Pretty print tables
# ═════════════════════════════════════════════════════════════════════════════

def print_model_results(acc: MultiStormAccumulator, model_color: str = "🔴"):
    """In bảng kết quả từng bão + tổng thể cho 1 model."""
    print(f"\n{'═'*115}")
    cfg  = MODEL_CONFIGS.get(acc.model_name, {})
    ref  = cfg.get("ref", "")
    col  = cfg.get("color", model_color)
    print(f"  {col}  {acc.model_name}  |  {ref}")
    print(f"{'═'*115}")

    # Header
    hdr = (
        f"  {'Storm':<16} {'N':>4}  "
        f"{'ADE':>8}  {'ATE':>8}  {'CTE':>8}  "
        f"║  {'12h':>7}  {'24h':>7}  {'48h':>7}  {'72h':>7}  "
        f"║  {'ATE@72':>8}  {'CTE@72':>8}"
    )
    print(hdr)
    print("  " + "─" * 111)

    rows = acc.all_summaries()
    for r in rows:
        sid  = r["storm_id"]
        n    = r["n_seq"]
        ade  = r["ADE"]
        ate  = r["ATE"]
        cte  = r["CTE"]
        d12  = r["12h_dist"]; d24 = r["24h_dist"]
        d48  = r["48h_dist"]; d72 = r["72h_dist"]
        a72  = r["72h_ate"];  c72 = r["72h_cte"]

        def v(x, t=None, w=8, dec=1):
            if not np.isfinite(x): return f"{'nan':>{w+2}}"
            s = f"{x:>{w}.{dec}f}"
            if t: s += "✅" if x < t else "❌"
            else:  s += "  "
            return s

        print(
            f"  🌀 {sid:<14} {n:>4}  "
            f"{v(ade,TARGETS['ADE'])}  {v(ate,TARGETS['ATE'])}  {v(cte,TARGETS['CTE'])}  "
            f"║  {v(d12,TARGETS['12h'],7)}  {v(d24,TARGETS['24h'],7)}  "
            f"{v(d48,TARGETS['48h'],7)}  {v(d72,TARGETS['72h'],7)}  "
            f"║  {v(a72,None,8)}  {v(c72,None,8)}"
        )

    # Overall
    ov = acc.overall_summary()
    print("  " + "─" * 111)

    def vo(x, t=None, w=8, dec=1):
        if not np.isfinite(x): return f"{'nan':>{w+2}}"
        s = f"{x:>{w}.{dec}f}"
        if t: s += "✅" if x < t else "❌"
        else:  s += "  "
        return s

    print(
        f"  📊 {'OVERALL':<14} {ov['n_seq']:>4}  "
        f"{vo(ov['ADE'],TARGETS['ADE'])}  {vo(ov['ATE'],TARGETS['ATE'])}  "
        f"{vo(ov['CTE'],TARGETS['CTE'])}  "
        f"║  {vo(ov['12h_dist'],TARGETS['12h'],7)}  {vo(ov['24h_dist'],TARGETS['24h'],7)}  "
        f"{vo(ov['48h_dist'],TARGETS['48h'],7)}  {vo(ov['72h_dist'],TARGETS['72h'],7)}  "
        f"║  {vo(ov['72h_ate'],None,8)}  {vo(ov['72h_cte'],None,8)}"
    )
    print(f"{'═'*115}\n")


def print_comparison_table(all_acc: Dict[str, MultiStormAccumulator]):
    """In bảng so sánh tất cả models."""
    print(f"\n{'═'*100}")
    print(f"  📊  MODEL COMPARISON  —  Overall Test Set")
    print(f"{'═'*100}")
    print(
        f"  {'Model':<14}  {'ADE':>9}  {'ATE':>9}  {'CTE':>9}  "
        f"║  {'12h':>8}  {'24h':>8}  {'48h':>8}  {'72h':>8}"
    )
    print("  " + "─" * 96)

    def vv(x, t=None, w=9, dec=1):
        if not np.isfinite(x): return f"{'nan':>{w+2}}"
        s = f"{x:>{w}.{dec}f}"
        if t: s += "✅" if x < t else "❌"
        else:  s += "  "
        return s

    for mname, acc in all_acc.items():
        cfg = MODEL_CONFIGS.get(mname, {})
        col = cfg.get("color", "⚪")
        ov  = acc.overall_summary()
        print(
            f"  {col} {mname:<12}  "
            f"{vv(ov['ADE'],TARGETS['ADE'])}  {vv(ov['ATE'],TARGETS['ATE'])}  "
            f"{vv(ov['CTE'],TARGETS['CTE'])}  "
            f"║  {vv(ov['12h_dist'],TARGETS['12h'],8)}  "
            f"{vv(ov['24h_dist'],TARGETS['24h'],8)}  "
            f"{vv(ov['48h_dist'],TARGETS['48h'],8)}  "
            f"{vv(ov['72h_dist'],TARGETS['72h'],8)}"
        )

    print("  " + "─" * 96)
    print(f"  Targets: "
          f"ADE<{TARGETS['ADE']} | ATE<{TARGETS['ATE']} | CTE<{TARGETS['CTE']} | "
          f"12h<{TARGETS['12h']} | 24h<{TARGETS['24h']} | "
          f"48h<{TARGETS['48h']} | 72h<{TARGETS['72h']} (km)")
    print(f"{'═'*100}\n")


# ═════════════════════════════════════════════════════════════════════════════
#  Main evaluation loop
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model,
    model_name: str,
    test_loader: DataLoader,
    device,
    args,
    infer_fn,
) -> MultiStormAccumulator:
    """Chạy toàn bộ evaluation loop cho 1 model."""
    acc = MultiStormAccumulator(model_name)
    n_batches = len(test_loader)
    t0 = time.perf_counter()

    print(f"\n  Running {model_name} on {n_batches} batches...")

    for i, batch in enumerate(test_loader):
        bl = move_to_device(list(batch), device)
        B  = bl[0].shape[1]

        # Inference
        pred_norm = infer_fn(model, bl, args, device)
        if pred_norm is None:
            print(f"    batch {i}: inference returned None, skipping")
            continue

        gt_norm   = bl[1]

        # Lấy storm IDs
        storm_ids = extract_storm_ids(batch, B)

        # Accumulate
        T = min(pred_norm.shape[0], gt_norm.shape[0])
        acc.add_batch(pred_norm[:T], gt_norm[:T], storm_ids)

        # Progress
        if (i + 1) % max(1, n_batches // 10) == 0 or i == n_batches - 1:
            elapsed = time.perf_counter() - t0
            ov = acc.overall_summary()
            ade_str = f"{ov['ADE']:.1f}km" if np.isfinite(ov["ADE"]) else "nan"
            print(f"    [{i+1:>4}/{n_batches}]  ADE={ade_str}  "
                  f"storms={len(acc.storms)}  elapsed={elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    ov = acc.overall_summary()
    print(f"  ✅ {model_name} done in {elapsed:.1f}s  "
          f"|  {len(acc.storms)} storms  "
          f"|  ADE={ov['ADE']:.1f}  ATE={ov['ATE']:.1f}  CTE={ov['CTE']:.1f} km")
    return acc


# ═════════════════════════════════════════════════════════════════════════════
#  Args
# ═════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description="Evaluate LSTM / ST-Trans / FlowMatching on private test set (12 storms)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument("--dataset_root",   default="TCND_vn", type=str,
                   help="Root thư mục TC-OFM (chứa Data1d, Data3d, Env_Data)")
    p.add_argument("--obs_len",        default=8,  type=int)
    p.add_argument("--pred_len",       default=12, type=int)
    p.add_argument("--batch_size",     default=16, type=int)
    p.add_argument("--num_workers",    default=2,  type=int)
    p.add_argument("--skip",           default=1,  type=int)
    p.add_argument("--min_ped",        default=1,  type=int)
    p.add_argument("--threshold",      default=0.002, type=float)
    p.add_argument("--other_modal",    default="gph")
    p.add_argument("--test_year",      default=None, type=int)
    p.add_argument("--delim",          default=" ")

    # Model checkpoints
    p.add_argument("--fm_ckpt",      default=None, type=str,
                   help="Path tới FM checkpoint (best_model.pth)")
    p.add_argument("--lstm_ckpt",    default=None, type=str,
                   help="Path tới LSTM checkpoint")
    p.add_argument("--sttrans_ckpt", default=None, type=str,
                   help="Path tới ST-Trans checkpoint")

    # FM inference settings
    p.add_argument("--fm_ensemble",  default=30,  type=int,
                   help="Số ensemble members cho FM")
    p.add_argument("--fm_ode_steps", default=20,  type=int,
                   help="Số ODE steps cho FM")

    # Output
    p.add_argument("--output_dir",   default="results/test_eval", type=str)
    p.add_argument("--gpu_num",      default="0", type=str)
    p.add_argument("--no_detail_csv", action="store_true",
                   help="Không ghi CSV chi tiết từng bão")
    p.add_argument("--eval_models",  default="all", type=str,
                   help="Models cần eval: 'all' hoặc 'fm,lstm,sttrans'")

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main(args):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 80)
    print(f"  TC Model Evaluator — Private Test Set (12 storms)")
    print(f"  Started: {ts_start}")
    print(f"  Device : {device}")
    print(f"  Output : {args.output_dir}")
    print("=" * 80)

    # Xác định models cần eval
    eval_set = set()
    if args.eval_models == "all":
        eval_set = {"fm", "lstm", "sttrans"}
    else:
        eval_set = set(m.strip().lower() for m in args.eval_models.split(","))

    # ── Load test data ────────────────────────────────────────────────────────
    print("\n  Loading test dataset...")
    try:
        from Model.data.loader_training import data_loader
        test_dataset, test_loader = data_loader(
            args, {"root": args.dataset_root, "type": "test"}, test=True)
        print(f"  ✅ Test set: {len(test_dataset)} sequences, "
              f"{len(test_loader)} batches")
    except Exception as e:
        print(f"  ❌ Failed to load test data: {e}")
        import traceback; traceback.print_exc()
        return

    # ── Load & evaluate models ─────────────────────────────────────────────
    all_acc: Dict[str, MultiStormAccumulator] = {}

    # FlowMatching
    if "fm" in eval_set and args.fm_ckpt:
        print(f"\n{'─'*50}")
        print("  Loading FlowMatching (FM_v59)...")
        fm_model = load_fm_model(args.fm_ckpt, device, args)
        if fm_model is not None:
            fm_acc = run_evaluation(
                fm_model, "FM_v59", test_loader, device, args, infer_fm)
            all_acc["FM_v59"] = fm_acc
            del fm_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # LSTM
    if "lstm" in eval_set and args.lstm_ckpt:
        print(f"\n{'─'*50}")
        print("  Loading LSTM...")
        lstm_model = load_lstm_model(args.lstm_ckpt, device, args)
        if lstm_model is not None:
            lstm_acc = run_evaluation(
                lstm_model, "LSTM", test_loader, device, args, infer_lstm)
            all_acc["LSTM"] = lstm_acc
            del lstm_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ST-Trans
    if "sttrans" in eval_set and args.sttrans_ckpt:
        print(f"\n{'─'*50}")
        print("  Loading ST-Trans...")
        st_model = load_sttrans_model(args.sttrans_ckpt, device, args)
        if st_model is not None:
            st_acc = run_evaluation(
                st_model, "STTrans", test_loader, device, args, infer_sttrans)
            all_acc["STTrans"] = st_acc
            del st_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not all_acc:
        print("\n  ❌ No models were evaluated. "
              "Check --fm_ckpt / --lstm_ckpt / --sttrans_ckpt arguments.")
        return

    # ── Print results ──────────────────────────────────────────────────────
    for mname, acc in all_acc.items():
        cfg = MODEL_CONFIGS.get(mname, {})
        print_model_results(acc, cfg.get("color", "⚪"))

    # Comparison table
    if len(all_acc) > 1:
        print_comparison_table(all_acc)

    # ── Write CSVs ─────────────────────────────────────────────────────────
    print("\n  Writing CSVs...")
    csv_paths = []
    for mname, acc in all_acc.items():
        csv_paths.append(write_per_storm_csv(acc, args.output_dir))
        if not args.no_detail_csv:
            write_per_storm_detail_csvs(acc, args.output_dir)

    sum_path = write_summary_csv(all_acc, args.output_dir)
    csv_paths.append(sum_path)

    # ── Final beat summary ─────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  FINAL BEAT SUMMARY")
    print(f"{'─'*80}")
    for mname, acc in all_acc.items():
        ov = acc.overall_summary()
        cfg = MODEL_CONFIGS.get(mname, {})
        col = cfg.get("color", "⚪")
        beats = []
        for k, t in [("ADE",TARGETS["ADE"]),("ATE",TARGETS["ATE"]),("CTE",TARGETS["CTE"]),
                     ("12h_dist",TARGETS["12h"]),("24h_dist",TARGETS["24h"]),
                     ("48h_dist",TARGETS["48h"]),("72h_dist",TARGETS["72h"])]:
            label = k.replace("_dist","h") if "_dist" in k else k
            v     = ov.get(k, float("nan"))
            if np.isfinite(v) and v < t:
                beats.append(f"{label}={v:.1f}✅")
        beat_str = " | ".join(beats) if beats else "no targets beaten"
        print(f"  {col} {mname:<12}: {beat_str}")

    elapsed_total = (datetime.now() - datetime.strptime(ts_start, "%Y-%m-%d %H:%M:%S")).seconds
    print(f"\n  Total time: {elapsed_total}s")
    print(f"  Outputs saved to: {args.output_dir}/")
    for p in csv_paths:
        print(f"    → {os.path.basename(p)}")
    print(f"{'═'*80}\n")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    args = get_args()

    # Kaggle path convenience: tự điền paths nếu chạy trong Kaggle
    _kaggle_base = "/kaggle/working/TC_FM"
    if args.dataset_root == "TCND_vn" and os.path.isdir(_kaggle_base):
        args.dataset_root = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"

    # Auto-detect checkpoints nếu không chỉ định
    _runs = os.path.join(_kaggle_base, "runs") if os.path.isdir(_kaggle_base) else "runs"

    if args.fm_ckpt is None:
        for candidate in [
            os.path.join(_runs, "v59", "best_model.pth"),
            os.path.join(_runs, "v58", "best_model.pth"),
            os.path.join(_runs, "v55", "best_model.pth"),
            "best_model.pth",
        ]:
            if os.path.exists(candidate):
                args.fm_ckpt = candidate
                print(f"  Auto-detected FM ckpt: {candidate}")
                break

    if args.lstm_ckpt is None:
        for candidate in [
            os.path.join(_runs, "lstm", "best_model.pth"),
            os.path.join(_runs, "LSTM", "best_model.pth"),
            os.path.join(_runs, "lstm_baseline", "best_model.pth"),
        ]:
            if os.path.exists(candidate):
                args.lstm_ckpt = candidate
                print(f"  Auto-detected LSTM ckpt: {candidate}")
                break

    if args.sttrans_ckpt is None:
        for candidate in [
            os.path.join(_runs, "sttrans", "best_model.pth"),
            os.path.join(_runs, "STTrans", "best_model.pth"),
            os.path.join(_runs, "st_trans", "best_model.pth"),
        ]:
            if os.path.exists(candidate):
                args.sttrans_ckpt = candidate
                print(f"  Auto-detected ST-Trans ckpt: {candidate}")
                break

    # Nếu không có model nào → báo hướng dẫn
    if not any([args.fm_ckpt, args.lstm_ckpt, args.sttrans_ckpt]):
        print("  ⚠  No checkpoint paths found. Please specify:")
        print("      --fm_ckpt path/to/best_model.pth")
        print("      --lstm_ckpt path/to/lstm_best.pth")
        print("      --sttrans_ckpt path/to/sttrans_best.pth")
        print()
        print("  Or set --eval_models to include only available models:")
        print("      --eval_models fm")
        print()

    main(args)
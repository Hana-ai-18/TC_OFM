"""
scripts/evaluate_test_storms.py
════════════════════════════════════════════════════════════════════════════════
Evaluate model trên 7 test cyclones (hoặc tập test bất kỳ).
Tính per-storm và overall: ADE, 72h, ATE, CTE.

Dùng được cho cả 3 model:
  - TCFlowMatching (flow_matching_model.py)
  - PaperBaseline  (paper_baseline_model.py)
  - STTrans        (st_trans_model.py)

Usage:
  python scripts/evaluate_test_storms.py \
    --model_type flow \
    --checkpoint runs/v55/best_model.pth \
    --dataset_root TCND_vn \
    --output_csv results/test_storms_flow_v55.csv

  python scripts/evaluate_test_storms.py \
    --model_type lstm \
    --checkpoint runs/paper_baseline/best_model.pth \
    --dataset_root TCND_vn

  python scripts/evaluate_test_storms.py \
    --model_type st_trans \
    --checkpoint runs/st_trans/best_model.pth \
    --dataset_root TCND_vn
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv
import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
R_EARTH  = 6371.0
DEG2KM   = 111.0
DT_HOURS = 6.0

# Horizon steps (6h resolution): 12h=1, 24h=3, 48h=7, 72h=11
HORIZON_STEPS = {12: 1, 24: 3, 48: 7, 72: 11}

# Target benchmarks (từ paper Faiaz et al. 2026)
TARGETS = {
    "ADE"  : 136.41,
    "72h"  : 297.0,
    "ATE"  : 79.94,
    "CTE"  : 93.58,
    "12h"  : 50.0,
    "24h"  : 100.0,
    "48h"  : 200.0,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Coordinate helpers
# ══════════════════════════════════════════════════════════════════════════════

def _norm_to_deg(arr: torch.Tensor) -> torch.Tensor:
    out = arr.clone()
    out[..., 0] = (arr[..., 0] * 50.0 + 1800.0) / 10.0
    out[..., 1] = (arr[..., 1] * 50.0) / 10.0
    return out


def haversine_km(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """[..., 2] degrees → km"""
    lon1 = torch.deg2rad(p1[..., 0]); lat1 = torch.deg2rad(p1[..., 1])
    lon2 = torch.deg2rad(p2[..., 0]); lat2 = torch.deg2rad(p2[..., 1])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = (torch.sin(dlat/2).pow(2)
         + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2).pow(2))
    return 2.0 * R_EARTH * torch.asin(a.clamp(1e-12, 1-1e-12).sqrt())


def _forward_azimuth(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    lon1 = torch.deg2rad(p1[..., 0]); lat1 = torch.deg2rad(p1[..., 1])
    lon2 = torch.deg2rad(p2[..., 0]); lat2 = torch.deg2rad(p2[..., 1])
    dlon = lon2 - lon1
    y = torch.sin(dlon) * torch.cos(lat2)
    x = (torch.cos(lat1)*torch.sin(lat2)
         - torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon))
    return torch.atan2(y, x)


def compute_ate_cte(pred_deg: torch.Tensor,
                    gt_deg:   torch.Tensor) -> tuple:
    """
    Returns (ate_km, cte_km) both [T-1, B].
    ATE: along-track error (speed/timing bias)
    CTE: cross-track error (lateral bias)
    """
    T = min(pred_deg.shape[0], gt_deg.shape[0])
    if T < 2:
        z = pred_deg.new_zeros(1, pred_deg.shape[1])
        return z, z

    bear_along = _forward_azimuth(gt_deg[:T-1],   gt_deg[1:T])
    bear_error = _forward_azimuth(gt_deg[1:T],    pred_deg[1:T])
    total_err  = haversine_km(pred_deg[1:T], gt_deg[1:T])

    angle = bear_error - bear_along
    ate   = total_err * torch.cos(angle)
    cte   = total_err * torch.sin(angle)
    return ate, cte


# ══════════════════════════════════════════════════════════════════════════════
#  Move batch to device
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


# ══════════════════════════════════════════════════════════════════════════════
#  Load model
# ══════════════════════════════════════════════════════════════════════════════

def load_model(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device)

    if args.model_type == "flow":
        from Model.flow_matching_model import TCFlowMatching
        model = TCFlowMatching(
            pred_len   = args.pred_len,
            obs_len    = args.obs_len,
            unet_in_ch = args.unet_in_ch,
        ).to(device)
        state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        model.load_state_dict(state, strict=False)

        # Load EMA nếu có
        if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None and args.use_ema:
            model.init_ema()
            ema_obj = model._ema
            if ema_obj is not None:
                for k, v in ckpt["ema_shadow"].items():
                    if k in ema_obj.shadow:
                        ema_obj.shadow[k].copy_(v.to(device))
                ema_obj.apply_to(model)
                print("  EMA weights applied")

    elif args.model_type in ("lstm", "gru", "rnn"):
        from Model.paper_baseline_model import PaperBaseline
        model = PaperBaseline(
            model_type = args.model_type,
            pred_len   = args.pred_len,
            obs_len    = args.obs_len,
            hidden_dim = args.hidden_dim,
            n_layers   = args.n_layers,
            unet_in_ch = args.unet_in_ch,
            dropout    = args.dropout,
        ).to(device)
        state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        model.load_state_dict(state, strict=False)

    elif args.model_type in ("st_trans", "st_trans_ar"):
        from Model.st_trans_model import STTrans, STTransAR
        ModelCls = STTrans if args.model_type == "st_trans" else STTransAR
        model = ModelCls(
            obs_len    = args.obs_len,
            pred_len   = args.pred_len,
            d_model    = args.d_model,
            unet_in_ch = args.unet_in_ch,
        ).to(device)
        state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        model.load_state_dict(state, strict=False)

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model loaded: {args.model_type}  ({n_params:,} params)")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Per-storm accumulator
# ══════════════════════════════════════════════════════════════════════════════

class StormAccumulator:
    """Accumulate metrics per storm ID."""
    def __init__(self):
        self.data: Dict[str, Dict] = defaultdict(lambda: {
            "dists": [], "ates": [], "ctes": [],
            "step_dists": defaultdict(list),
            "n_seq": 0,
        })

    def update(self, storm_id: str,
               dist: torch.Tensor,
               ate:  Optional[torch.Tensor],
               cte:  Optional[torch.Tensor]):
        """
        dist: [T, B] km
        ate:  [T-1, B] km (signed, along-track)
        cte:  [T-1, B] km (signed, cross-track)
        """
        d = self.data[storm_id]
        B = dist.shape[1]

        # ADE per sample in batch
        d["dists"].extend(dist.mean(0).tolist())
        d["n_seq"] += B

        # Per-horizon
        for h, s in HORIZON_STEPS.items():
            if s < dist.shape[0]:
                d["step_dists"][h].extend(dist[s].tolist())

        # ATE/CTE (absolute value for reporting)
        if ate is not None:
            d["ates"].extend(ate.abs().mean(0).tolist())
        if cte is not None:
            d["ctes"].extend(cte.abs().mean(0).tolist())

    def compute_storm(self, storm_id: str) -> Dict:
        d = self.data[storm_id]
        r = {
            "storm_id" : storm_id,
            "n_seq"    : d["n_seq"],
            "ADE"      : float(np.mean(d["dists"])) if d["dists"] else float("nan"),
            "ATE"      : float(np.mean(d["ates"]))  if d["ates"]  else float("nan"),
            "CTE"      : float(np.mean(d["ctes"]))  if d["ctes"]  else float("nan"),
        }
        for h in HORIZON_STEPS:
            vals = d["step_dists"].get(h, [])
            r[f"{h}h"] = float(np.mean(vals)) if vals else float("nan")
        return r

    def compute_overall(self) -> Dict:
        all_ade, all_ate, all_cte = [], [], []
        step_all = defaultdict(list)
        n_total  = 0

        for d in self.data.values():
            all_ade.extend(d["dists"])
            all_ate.extend(d["ates"])
            all_cte.extend(d["ctes"])
            n_total += d["n_seq"]
            for h, vals in d["step_dists"].items():
                step_all[h].extend(vals)

        r = {
            "storm_id" : "OVERALL",
            "n_seq"    : n_total,
            "ADE"      : float(np.mean(all_ade)) if all_ade else float("nan"),
            "ATE"      : float(np.mean(all_ate)) if all_ate else float("nan"),
            "CTE"      : float(np.mean(all_cte)) if all_cte else float("nan"),
        }
        for h in HORIZON_STEPS:
            vals = step_all.get(h, [])
            r[f"{h}h"] = float(np.mean(vals)) if vals else float("nan")
        return r


# ══════════════════════════════════════════════════════════════════════════════
#  Inference per batch
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model, bl, args) -> tuple:
    """Returns (pred_deg [T,B,2], gt_deg [T,B,2])."""
    if args.model_type == "flow":
        pred, _, _ = model.sample(
            bl,
            num_ensemble    = args.num_ensemble,
            ddim_steps      = args.ode_steps,
            importance_weight = True,
        )
    else:
        pred, _, _ = model.sample(bl)

    gt   = bl[1]
    T    = min(pred.shape[0], gt.shape[0])
    pred_deg = _norm_to_deg(pred[:T])
    gt_deg   = _norm_to_deg(gt[:T])
    return pred_deg, gt_deg


# ══════════════════════════════════════════════════════════════════════════════
#  Print helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v: float, target: Optional[float] = None) -> str:
    if not np.isfinite(v):
        return "  nan  "
    s = f"{v:>7.1f}"
    if target is not None:
        mark = "✅" if v < target else "❌"
        s = s + mark
    return s


def _print_storm_row(r: dict):
    sid = r["storm_id"]
    n   = r["n_seq"]
    is_overall = (sid == "OVERALL")

    sep = "─" * 90 if is_overall else ""
    if sep: print(sep)

    prefix = "  📊 OVERALL " if is_overall else f"  🌀 {sid:<12}"
    print(
        f"{prefix}"
        f"  n={n:>4}"
        f"  ADE={_fmt(r['ADE'],  TARGETS['ADE'])}"
        f"  12h={_fmt(r['12h'],  TARGETS['12h'])}"
        f"  24h={_fmt(r['24h'],  TARGETS['24h'])}"
        f"  48h={_fmt(r['48h'],  TARGETS['48h'])}"
        f"  72h={_fmt(r['72h'],  TARGETS['72h'])}"
        f"  ATE={_fmt(r['ATE'],  TARGETS['ATE'])}"
        f"  CTE={_fmt(r['CTE'],  TARGETS['CTE'])}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_test_set(model, test_loader, device, args) -> List[Dict]:
    """
    Chạy inference trên toàn bộ test set.
    Giả định batch_list[3] chứa storm_id (string tensor hoặc list).
    Nếu không có storm_id → gộp tất cả vào "ALL_STORMS".
    """
    acc = StormAccumulator()
    t0  = time.perf_counter()
    n_batch = 0

    for batch in test_loader:
        bl = move(list(batch), device)

        # Lấy storm ID nếu có
        storm_ids = _extract_storm_ids(batch, bl)

        pred_deg, gt_deg = run_inference(model, bl, args)
        B = pred_deg.shape[1]

        # Tính distance
        dist = haversine_km(pred_deg, gt_deg)   # [T, B]

        # Tính ATE/CTE
        ate, cte = compute_ate_cte(pred_deg, gt_deg)

        # Update per-storm
        if storm_ids is not None and len(storm_ids) == B:
            for b in range(B):
                sid = storm_ids[b]
                d1  = dist[:, b:b+1]
                a1  = ate[:, b:b+1] if ate is not None else None
                c1  = cte[:, b:b+1] if cte is not None else None
                acc.update(sid, d1, a1, c1)
        else:
            acc.update("ALL_STORMS", dist, ate, cte)

        n_batch += 1
        if n_batch % 10 == 0:
            print(f"  Processed {n_batch} batches...", end="\r")

    elapsed = time.perf_counter() - t0
    print(f"  Inference done: {n_batch} batches in {elapsed:.1f}s")

    # Compute results
    results = []
    storm_ids_sorted = sorted(
        [s for s in acc.data.keys() if s != "ALL_STORMS"])

    for sid in storm_ids_sorted:
        r = acc.compute_storm(sid)
        results.append(r)
        _print_storm_row(r)

    # Overall
    overall = acc.compute_overall()
    results.append(overall)
    _print_storm_row(overall)

    return results


def _extract_storm_ids(batch, bl) -> Optional[List[str]]:
    """
    Cố gắng lấy storm ID từ batch.
    DataLoader của bạn có thể lưu ở index khác nhau.
    Thử các vị trí phổ biến.
    """
    # Thử batch index 2 (seq_id hoặc storm_id trong data loader)
    for idx in [2, 12, 14, 15]:
        if idx < len(batch):
            item = batch[idx]
            if isinstance(item, (list, tuple)) and len(item) > 0:
                if isinstance(item[0], str):
                    return list(item)
                # Tensor của IDs
                if torch.is_tensor(item[0]):
                    return [str(x.item()) for x in item]
            if isinstance(item, torch.Tensor) and item.dtype == torch.long:
                return [str(x.item()) for x in item]
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Save results to CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_results_csv(results: List[Dict], csv_path: str, model_name: str):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    fields = ["timestamp", "model", "storm_id", "n_seq",
              "ADE", "12h", "24h", "48h", "72h", "ATE", "CTE",
              "ADE_beat", "72h_beat", "ATE_beat", "CTE_beat"]

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {
                "timestamp" : ts,
                "model"     : model_name,
                "storm_id"  : r["storm_id"],
                "n_seq"     : r["n_seq"],
                "ADE"       : f"{r['ADE']:.2f}",
                "12h"       : f"{r['12h']:.2f}",
                "24h"       : f"{r['24h']:.2f}",
                "48h"       : f"{r['48h']:.2f}",
                "72h"       : f"{r['72h']:.2f}",
                "ATE"       : f"{r['ATE']:.2f}",
                "CTE"       : f"{r['CTE']:.2f}",
                "ADE_beat"  : "YES" if r["ADE"] < TARGETS["ADE"] else "NO",
                "72h_beat"  : "YES" if r["72h"] < TARGETS["72h"] else "NO",
                "ATE_beat"  : "YES" if r["ATE"] < TARGETS["ATE"] else "NO",
                "CTE_beat"  : "YES" if r["CTE"] < TARGETS["CTE"] else "NO",
            }
            w.writerow(row)
    print(f"\n  Results saved: {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Args
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Per-storm evaluation on test set")

    p.add_argument("--model_type",   required=True,
                   choices=["flow", "lstm", "gru", "rnn", "st_trans", "st_trans_ar"],
                   help="Model architecture")
    p.add_argument("--checkpoint",   required=True,   type=str,
                   help="Path to checkpoint .pth file")
    p.add_argument("--dataset_root", default="TCND_vn", type=str)
    p.add_argument("--obs_len",      default=8,   type=int)
    p.add_argument("--pred_len",     default=12,  type=int)
    p.add_argument("--batch_size",   default=32,  type=int)
    p.add_argument("--num_workers",  default=2,   type=int)
    p.add_argument("--gpu_num",      default="0", type=str)

    # Flow matching inference
    p.add_argument("--num_ensemble", default=50,  type=int)
    p.add_argument("--ode_steps",    default=20,  type=int)
    p.add_argument("--use_ema",      action="store_true", default=True)

    # Baseline model params
    p.add_argument("--hidden_dim",   default=256, type=int)
    p.add_argument("--n_layers",     default=3,   type=int)
    p.add_argument("--dropout",      default=0.2, type=float)
    p.add_argument("--d_model",      default=128, type=int)
    p.add_argument("--unet_in_ch",   default=13,  type=int)

    # Output
    p.add_argument("--output_csv",
                   default="results/test_storm_metrics.csv", type=str)
    p.add_argument("--model_name",   default=None, type=str,
                   help="Label for CSV (default: model_type + checkpoint name)")

    # DataLoader compat
    p.add_argument("--delim",       default=" ")
    p.add_argument("--skip",        default=1,   type=int)
    p.add_argument("--min_ped",     default=1,   type=int)
    p.add_argument("--threshold",   default=0.002, type=float)
    p.add_argument("--other_modal", default="gph")
    p.add_argument("--test_year",   default=None, type=int)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.model_name or (
        f"{args.model_type}_{os.path.basename(args.checkpoint).replace('.pth','')}")

    print("=" * 90)
    print(f"  Per-Storm Evaluation")
    print(f"  Model     : {model_name}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset   : {args.dataset_root}")
    print("=" * 90)

    # Load model
    model = load_model(args, device)

    # Load test data
    from Model.data.loader_training import data_loader
    test_dataset, test_loader = data_loader(
        args,
        {"root": args.dataset_root, "type": "test"},
        test=True,
    )
    print(f"  test  : {len(test_dataset)} sequences  ({len(test_loader)} batches)")

    print("\n  Columns: storm_id | n | ADE | 12h | 24h | 48h | 72h | ATE | CTE")
    print("  ✅ = beat target | ❌ = below target")
    print("  Targets: ADE<136 | 12h<50 | 24h<100 | 48h<200 | 72h<297 | ATE<80 | CTE<94")
    print("─" * 90)

    # Evaluate
    results = evaluate_test_set(model, test_loader, device, args)

    # Print beat summary
    overall = results[-1]
    print(f"\n  {'─'*50}")
    beats = []
    for k, t in TARGETS.items():
        key = k if k in overall else f"{k}"
        val = overall.get(key, float("nan"))
        if np.isfinite(val) and val < t:
            beats.append(f"{k}={val:.1f}✅")
    if beats:
        print(f"  🏆 BEAT: {' | '.join(beats)}")
    else:
        print(f"  No targets beaten yet")

    # Save CSV
    save_results_csv(results, args.output_csv, model_name)


if __name__ == "__main__":
    args = get_args()
    import numpy as np
    np.random.seed(42)
    torch.manual_seed(42)
    main(args)